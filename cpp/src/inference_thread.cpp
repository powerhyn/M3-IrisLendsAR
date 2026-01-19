/**
 * @file inference_thread.cpp
 * @brief 전용 추론 스레드 클래스 구현 (최적화 버전)
 *
 * 단일 슬롯 패턴으로 동기화 오버헤드 최소화
 */

#include "iris_sdk/inference_thread.h"
#include "iris_sdk/mediapipe_detector.h"

#include <chrono>
#include <cstdio>

namespace iris_sdk {

// ============================================================================
// 생성자 / 소멸자
// ============================================================================

InferenceThread::InferenceThread() = default;

InferenceThread::~InferenceThread() {
    stop();
}

// ============================================================================
// 스레드 제어
// ============================================================================

bool InferenceThread::start(const std::string& model_path, bool gpu_enabled) {
    // 이미 실행 중이면 중지
    ThreadState expected = ThreadState::Stopped;
    if (!state_.compare_exchange_strong(expected, ThreadState::Starting)) {
        // 이미 다른 상태인 경우
        if (state_ != ThreadState::Stopped) {
            stop();
            expected = ThreadState::Stopped;
            if (!state_.compare_exchange_strong(expected, ThreadState::Starting)) {
                return false;
            }
        }
    }

    model_path_ = model_path;
    gpu_enabled_ = gpu_enabled;

    // 스레드 시작
    thread_ = std::thread(&InferenceThread::threadLoop, this);

    // 초기화 완료 대기
    return waitForInitialization(10000);  // 10초 타임아웃
}

void InferenceThread::stop() {
    ThreadState current = state_.load();
    if (current == ThreadState::Stopped) {
        return;
    }

    // 중지 요청
    state_ = ThreadState::Stopping;

    // 대기 중인 스레드 깨우기
    slot_cv_.notify_one();

    // 스레드 종료 대기
    if (thread_.joinable()) {
        thread_.join();
    }

    // 상태 초기화
    state_ = ThreadState::Stopped;
    gpu_active_ = false;

    // 슬롯 초기화
    {
        std::lock_guard<std::mutex> lock(slot_mutex_);
        pending_data_ = nullptr;
        pending_width_ = 0;
        pending_height_ = 0;
        pending_format_ = 0;
        pending_result_ = IrisResult{};
    }
}

bool InferenceThread::isRunning() const noexcept {
    ThreadState current = state_.load();
    return current == ThreadState::Idle ||
           current == ThreadState::Processing ||
           current == ThreadState::ResultReady;
}

bool InferenceThread::waitForInitialization(int timeout_ms) {
    using namespace std::chrono;
    auto deadline = steady_clock::now() + milliseconds(timeout_ms);

    while (steady_clock::now() < deadline) {
        ThreadState current = state_.load();

        // 초기화 완료 (Idle 또는 Stopped)
        if (current == ThreadState::Idle) {
            return true;
        }

        // 초기화 실패
        if (current == ThreadState::Stopped || current == ThreadState::Stopping) {
            return false;
        }

        // Starting 상태면 대기
        std::this_thread::sleep_for(milliseconds(10));
    }

    std::fprintf(stderr, "[InferenceThread] Initialization timeout\n");
    return false;
}

// ============================================================================
// 검출 API
// ============================================================================

IrisResult InferenceThread::detectSync(const uint8_t* data, int width, int height, int format) {
    return submitAndWait(data, width, height, format);
}

IrisResult InferenceThread::submitAndWait(const uint8_t* data, int width, int height, int format) {
    IrisResult empty_result;
    empty_result.detected = false;

    // 유효성 검사
    if (data == nullptr || width <= 0 || height <= 0) {
        return empty_result;
    }

    // 스레드 상태 확인 및 Processing으로 전환
    ThreadState expected = ThreadState::Idle;
    if (!state_.compare_exchange_strong(expected, ThreadState::Processing)) {
        // Idle 상태가 아니면 실패
        return empty_result;
    }

    // 입력 데이터 설정
    {
        std::lock_guard<std::mutex> lock(slot_mutex_);
        pending_data_ = data;
        pending_width_ = width;
        pending_height_ = height;
        pending_format_ = format;
    }

    // 워커 스레드 깨우기
    slot_cv_.notify_one();

    // 결과 대기
    {
        std::unique_lock<std::mutex> lock(slot_mutex_);
        bool success = slot_cv_.wait_for(lock, std::chrono::seconds(5), [this] {
            ThreadState s = state_.load();
            return s == ThreadState::ResultReady ||
                   s == ThreadState::Stopped ||
                   s == ThreadState::Stopping;
        });

        if (!success || state_ != ThreadState::ResultReady) {
            std::fprintf(stderr, "[InferenceThread] Detection timeout or thread stopped\n");
            state_ = ThreadState::Idle;  // 복구
            return empty_result;
        }

        // 결과 복사 후 상태 Idle로 전환
        IrisResult result = pending_result_;
        state_ = ThreadState::Idle;
        return result;
    }
}

// ============================================================================
// 상태 조회
// ============================================================================

bool InferenceThread::isGpuActive() const noexcept {
    return gpu_active_;
}

int InferenceThread::getModelVersion() const noexcept {
    return model_version_;
}

int InferenceThread::getFaceLandmarkCount() const noexcept {
    return landmark_count_;
}

bool InferenceThread::getFaceLandmarks(float* out_landmarks) const {
    if (!out_landmarks || !isRunning()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(landmark_mutex_);
    if (last_landmarks_.empty()) {
        return false;
    }

    std::copy(last_landmarks_.begin(), last_landmarks_.end(), out_landmarks);
    return true;
}

// ============================================================================
// 스레드 메인 루프
// ============================================================================

void InferenceThread::threadLoop() {
    std::fprintf(stderr, "[InferenceThread] Thread started\n");

    // ========================================
    // 1. 검출기 초기화 (이 스레드에서!)
    // ========================================
    detector_ = std::make_unique<MediaPipeDetector>();

    if (gpu_enabled_) {
        detector_->setGpuEnabled(true);
        std::fprintf(stderr, "[InferenceThread] GPU enabled requested\n");
    }

    bool init_result = detector_->initialize(model_path_);

    if (init_result) {
        gpu_active_ = detector_->isUsingGpu();
        model_version_ = detector_->getModelVersion();
        landmark_count_ = detector_->getFaceLandmarkCount();

        std::fprintf(stderr, "[InferenceThread] Detector initialized: GPU=%s, Version=%d\n",
                     gpu_active_.load() ? "true" : "false", model_version_.load());

        // 초기화 성공 - Idle로 전환
        state_ = ThreadState::Idle;
    } else {
        std::fprintf(stderr, "[InferenceThread] Detector initialization failed\n");
        state_ = ThreadState::Stopped;
        return;
    }

    // ========================================
    // 2. 메인 루프 - 단일 슬롯 패턴
    // ========================================
    while (state_ != ThreadState::Stopping && state_ != ThreadState::Stopped) {
        const uint8_t* data;
        int width, height, format;

        // 요청 대기
        {
            std::unique_lock<std::mutex> lock(slot_mutex_);
            slot_cv_.wait(lock, [this] {
                ThreadState s = state_.load();
                return s == ThreadState::Processing ||
                       s == ThreadState::Stopping ||
                       s == ThreadState::Stopped;
            });

            ThreadState current = state_.load();
            if (current == ThreadState::Stopping || current == ThreadState::Stopped) {
                break;
            }

            // 입력 데이터 복사 (포인터만)
            data = pending_data_;
            width = pending_width_;
            height = pending_height_;
            format = pending_format_;
        }

        // 검출 수행 (락 없이 - GPU delegate와 동일 스레드에서)
        IrisResult result = detector_->detect(
            data,
            width,
            height,
            static_cast<FrameFormat>(format));

        // 랜드마크 데이터 저장 (디버그용)
        if (result.detected) {
            int count = detector_->getFaceLandmarkCount();
            if (count > 0) {
                std::lock_guard<std::mutex> lock(landmark_mutex_);
                last_landmarks_.resize(static_cast<size_t>(count) * 3);
                detector_->getFaceLandmarks(last_landmarks_.data());
            }
        }

        // 결과 저장 및 상태 전환
        {
            std::lock_guard<std::mutex> lock(slot_mutex_);
            pending_result_ = result;
            state_ = ThreadState::ResultReady;
        }
        slot_cv_.notify_one();  // notify_all() → notify_one()
    }

    // ========================================
    // 3. 정리
    // ========================================
    std::fprintf(stderr, "[InferenceThread] Thread stopping, releasing detector\n");
    detector_.reset();
    state_ = ThreadState::Stopped;
    std::fprintf(stderr, "[InferenceThread] Thread stopped\n");
}

} // namespace iris_sdk
