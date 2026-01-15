/**
 * @file inference_thread.cpp
 * @brief 전용 추론 스레드 클래스 구현
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
    if (running_) {
        stop();
    }

    model_path_ = model_path;
    gpu_enabled_ = gpu_enabled;
    running_ = true;
    initialized_ = false;
    init_success_ = false;

    // 스레드 시작
    thread_ = std::thread(&InferenceThread::threadLoop, this);

    // 초기화 완료 대기
    return waitForInitialization(10000);  // 10초 타임아웃
}

void InferenceThread::stop() {
    if (!running_) {
        return;
    }

    // 중지 요청
    running_ = false;

    // 대기 중인 스레드 깨우기
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        queue_cv_.notify_all();
    }

    // 스레드 종료 대기
    if (thread_.joinable()) {
        thread_.join();
    }

    // 상태 초기화
    initialized_ = false;
    init_success_ = false;
    gpu_active_ = false;

    // 큐 비우기
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        while (!frame_queue_.empty()) {
            frame_queue_.pop();
        }
    }

    // 결과 비우기
    {
        std::lock_guard<std::mutex> lock(result_mutex_);
        results_.clear();
    }
}

bool InferenceThread::isRunning() const noexcept {
    return running_ && initialized_ && init_success_;
}

bool InferenceThread::waitForInitialization(int timeout_ms) {
    std::unique_lock<std::mutex> lock(init_mutex_);

    if (timeout_ms <= 0) {
        init_cv_.wait(lock, [this] { return initialized_.load(); });
    } else {
        bool result = init_cv_.wait_for(lock,
            std::chrono::milliseconds(timeout_ms),
            [this] { return initialized_.load(); });
        if (!result) {
            std::fprintf(stderr, "[InferenceThread] Initialization timeout\n");
            return false;
        }
    }

    return init_success_;
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

    // 스레드가 실행 중인지 확인
    if (!isRunning() || data == nullptr || width <= 0 || height <= 0) {
        return empty_result;
    }

    // 요청 ID 생성
    uint64_t request_id = next_request_id_++;

    // 요청 생성 (동기 호출이므로 포인터만 전달 - 복사 오버헤드 제거)
    FrameRequest request;
    request.request_id = request_id;
    request.data = data;  // 포인터만 저장 (호출자가 결과 대기하므로 안전)
    request.width = width;
    request.height = height;
    request.format = format;

    // 큐에 요청 추가
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        frame_queue_.push(std::move(request));
    }
    queue_cv_.notify_one();

    // 결과 대기
    {
        std::unique_lock<std::mutex> lock(result_mutex_);
        bool found = result_cv_.wait_for(lock, std::chrono::seconds(5), [this, request_id] {
            return results_.find(request_id) != results_.end();
        });

        if (!found) {
            std::fprintf(stderr, "[InferenceThread] Detection timeout for request %llu\n",
                         static_cast<unsigned long long>(request_id));
            return empty_result;
        }

        // 결과 추출 및 제거
        auto it = results_.find(request_id);
        if (it != results_.end()) {
            IrisResult result = it->second.iris_result;
            results_.erase(it);
            return result;
        }
    }

    return empty_result;
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
    } else {
        std::fprintf(stderr, "[InferenceThread] Detector initialization failed\n");
    }

    // 초기화 완료 알림
    {
        std::lock_guard<std::mutex> lock(init_mutex_);
        init_success_ = init_result;
        initialized_ = true;
    }
    init_cv_.notify_all();

    if (!init_result) {
        running_ = false;
        return;
    }

    // ========================================
    // 2. 메인 루프 - 프레임 처리
    // ========================================
    while (running_) {
        FrameRequest request;

        // 큐에서 요청 가져오기
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] {
                return !frame_queue_.empty() || !running_;
            });

            if (!running_) {
                break;
            }

            if (frame_queue_.empty()) {
                continue;
            }

            request = std::move(frame_queue_.front());
            frame_queue_.pop();
        }

        // 검출 수행 (GPU delegate와 동일 스레드에서)
        // NOTE: 회전 처리는 FrameProcessor에서 수행 후 전달됨
        IrisResult result = detector_->detect(
            request.data,  // 이미 포인터
            request.width,
            request.height,
            static_cast<FrameFormat>(request.format));

        // 랜드마크 데이터 저장 (디버그용)
        if (result.detected) {
            int count = detector_->getFaceLandmarkCount();
            if (count > 0) {
                std::lock_guard<std::mutex> lock(landmark_mutex_);
                last_landmarks_.resize(static_cast<size_t>(count) * 3);
                detector_->getFaceLandmarks(last_landmarks_.data());
            }
        }

        // 결과 저장
        {
            std::lock_guard<std::mutex> lock(result_mutex_);
            DetectionResult det_result;
            det_result.iris_result = result;
            det_result.request_id = request.request_id;
            det_result.success = true;
            results_[request.request_id] = det_result;
        }
        result_cv_.notify_all();
    }

    // ========================================
    // 3. 정리
    // ========================================
    std::fprintf(stderr, "[InferenceThread] Thread stopping, releasing detector\n");
    detector_.reset();
    std::fprintf(stderr, "[InferenceThread] Thread stopped\n");
}

} // namespace iris_sdk
