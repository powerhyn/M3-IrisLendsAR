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
#include <cstring>

namespace iris_sdk {

namespace {

// 프레임 포맷별 바이트 크기 계산. 알 수 없는 포맷이면 0을 반환한다.
// ③-2 B1: 동기/비동기 경로가 동일 규약으로 입력을 딥카피하기 위한 공통 헬퍼.
size_t computeFrameSize(int width, int height, int format) {
    if (width <= 0 || height <= 0) {
        return 0;
    }
    const size_t w = static_cast<size_t>(width);
    const size_t h = static_cast<size_t>(height);
    switch (static_cast<FrameFormat>(format)) {
        case FrameFormat::RGB:
        case FrameFormat::BGR:
            return w * h * 3;
        case FrameFormat::RGBA:
        case FrameFormat::BGRA:
            return w * h * 4;
        case FrameFormat::Grayscale:
            return w * h;
        case FrameFormat::NV21:
        case FrameFormat::NV12:
            return w * h * 3 / 2;
        default:
            return 0;
    }
}

}  // namespace

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
    // ③-2 B1(important 1): thread_/model_path_/gpu_enabled_ 접근 전 구간을
    // lifecycle_mutex_로 직렬화한다. 이로써 다른 스레드의 start()/stop()과
    // thread_ move-대입/joinable 검사가 레이스하지 않는다.
    std::lock_guard<std::mutex> ctl(lifecycle_mutex_);

    // ③-2 B1: 이전 스레드가 남아 있으면 반드시 회수한다.
    // stopLocked()는 어떤 상태에서든 joinable이면 join하므로 안전하게 재시작
    // 가능하며, joinable 스레드에 std::thread를 move-대입(아래 thread_ = ...)하면
    // std::terminate가 발생하는 결함을 차단한다.
    // ★ 재귀 데드락 회피: 이미 lifecycle_mutex_를 보유 중이므로 public stop()이
    //   아니라 락을 잡지 않는 stopLocked()를 호출한다.
    if (state_.load() != ThreadState::Stopped || thread_.joinable()) {
        stopLocked();
    }

    ThreadState expected = ThreadState::Stopped;
    if (!state_.compare_exchange_strong(expected, ThreadState::Starting)) {
        // stopLocked() 직후에도 Stopped가 아니면(논리 오류) 시작 불가
        return false;
    }

    model_path_ = model_path;
    gpu_enabled_ = gpu_enabled;

    // 스레드 시작 (이 시점에서 thread_는 non-joinable 보장)
    thread_ = std::thread(&InferenceThread::threadLoop, this);

    // 초기화 완료 대기
    return waitForInitialization(10000);  // 10초 타임아웃
}

void InferenceThread::stop() {
    // ③-2 B1(important 1): 제어 경로를 lifecycle_mutex_로 직렬화한다.
    // 여러 스레드가 동시에 stop()을 호출해도 joinable()/join()이 직렬화되어
    // 이중 join(UB)이 발생하지 않는다.
    std::lock_guard<std::mutex> ctl(lifecycle_mutex_);
    stopLocked();
}

void InferenceThread::stopLocked() {
    // ★ 사전조건: 호출자가 lifecycle_mutex_를 보유하고 있어야 한다(public stop()
    //   또는 start()의 재시작 정리 경로). 워커 스레드는 이 함수를 호출하지 않는다.
    // ③-2 B1: 상태와 무관하게, joinable 스레드는 반드시 join한다.
    // 기존 코드는 상태가 Stopped면 조기 반환했는데, threadLoop가 초기화 실패로
    // state_=Stopped를 기록한 뒤에도 std::thread는 join 전까지 joinable로 남는다.
    // 그 상태에서 소멸자/재시작이 join을 건너뛰면 joinable 스레드 소멸 →
    // std::terminate(프로세스 abort)가 발생한다.

    // 중지 요청 (이미 Stopped여도 무해. threadLoop가 돌고 있으면 루프 종료 유도)
    state_ = ThreadState::Stopping;

    // 대기 중인 스레드 깨우기 (동기 + 비동기는 slot_cv_로 통합)
    {
        std::lock_guard<std::mutex> lock(slot_mutex_);
        slot_cv_.notify_all();
    }

    // 스레드 종료 대기 (상태와 무관하게 항상 수행)
    if (thread_.joinable()) {
        thread_.join();
    }

    // 상태 초기화
    state_ = ThreadState::Stopped;
    gpu_active_ = false;

    // 동기 슬롯 초기화
    {
        std::lock_guard<std::mutex> lock(slot_mutex_);
        pending_data_.clear();
        pending_width_ = 0;
        pending_height_ = 0;
        pending_format_ = 0;
        pending_result_ = IrisResult{};
        ++sync_request_seq_;  // 진행 중이던 요청 세대 무효화
    }

    // 비동기 슬롯 초기화
    {
        std::lock_guard<std::mutex> lock(async_input_mutex_);
        async_frame_buffer_.clear();
        async_width_ = 0;
        async_height_ = 0;
        async_format_ = 0;
        has_new_frame_ = false;
    }
    {
        std::lock_guard<std::mutex> lock(async_result_mutex_);
        async_latest_result_ = IrisResult{};
        has_async_result_ = false;
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

    // 입력 프레임 크기 계산 (딥카피 분량)
    // ③-2 B1: 호출자 raw 포인터를 워커가 직접 읽지 않도록 락 보호 딥카피한다.
    const size_t frame_size = computeFrameSize(width, height, format);
    if (frame_size == 0) {
        // ③-2 B1(important 2-a): 알 수 없는 포맷 — Processing→Idle 을 CAS로 복원한다.
        // 무조건 store 하면 그 사이 stop()이 기록한 Stopping/Stopped를 덮어써 join이
        // 영구 행에 빠진다. CAS 실패(=상태가 이미 바뀜) 시 상태를 건드리지 않는다.
        ThreadState expected = ThreadState::Processing;
        state_.compare_exchange_strong(expected, ThreadState::Idle);
        return empty_result;
    }

    uint64_t my_seq = 0;

    // 입력 데이터 설정 (딥카피) + 요청 세대 부여
    {
        std::lock_guard<std::mutex> lock(slot_mutex_);
        pending_data_.resize(frame_size);
        std::memcpy(pending_data_.data(), data, frame_size);
        pending_width_ = width;
        pending_height_ = height;
        pending_format_ = format;
        my_seq = ++sync_request_seq_;
    }

    // 워커 스레드 깨우기
    {
        std::lock_guard<std::mutex> lock(slot_mutex_);
        slot_cv_.notify_one();
    }

    // 결과 대기
    {
        std::unique_lock<std::mutex> lock(slot_mutex_);
        bool success = slot_cv_.wait_for(lock, std::chrono::seconds(5), [this] {
            ThreadState s = state_.load();
            return s == ThreadState::ResultReady ||
                   s == ThreadState::Stopped ||
                   s == ThreadState::Stopping;
        });

        if (!success || state_.load() != ThreadState::ResultReady) {
            std::fprintf(stderr, "[InferenceThread] Detection timeout or thread stopped\n");
            // ③-2 B1: 세대를 무효화해, 뒤늦게 완료된 워커가 ResultReady로
            //          상태를 고착시키지 못하게 한다. (락 보유 중이므로 워커의
            //          결과 기록과 직렬화된다.)
            ++sync_request_seq_;
            // ③-2 B1(important 2-b): TOCTOU 제거 — load 후 store의 두 원자 연산 사이에
            //          stop()이 끼어들면 Stopping/Stopped를 덮어써 join이 행에 빠졌다.
            //          Processing→Idle 단일 CAS로 복원하면, 상태가 이미 Stopping/Stopped
            //          이거나 워커가 ResultReady로 바꾼 경우 CAS가 실패해 불변이 유지된다.
            ThreadState expected = ThreadState::Processing;
            state_.compare_exchange_strong(expected, ThreadState::Idle);
            return empty_result;
        }

        // 결과 복사 후 상태 Idle로 전환 (락 보호 deep-copy)
        IrisResult result = pending_result_;
        (void)my_seq;  // 정상 경로에서는 세대가 일치하므로 검사 불필요
        // ③-2 B1(important 2-c): ResultReady→Idle 을 CAS로 복원한다. 정상 경로라도
        //          무조건 store 하면 그 직전 stop()이 기록한 Stopping/Stopped를 덮어써
        //          join이 행에 빠질 수 있다. CAS 실패 시 상태를 건드리지 않는다.
        ThreadState expected = ThreadState::ResultReady;
        state_.compare_exchange_strong(expected, ThreadState::Idle);
        return result;
    }
}

// ============================================================================
// 비동기 검출 API (P5-W1-04)
// ============================================================================

void InferenceThread::submitFrameAsync(const uint8_t* data, int width, int height, int format) {
    if (data == nullptr || width <= 0 || height <= 0) {
        return;
    }

    if (!isRunning()) {
        return;
    }

    // 프레임 크기 계산 (RGB 3채널 기준, 다른 포맷도 지원)
    const size_t frame_size = computeFrameSize(width, height, format);
    if (frame_size == 0) {
        return;  // 알 수 없는 포맷
    }

    // 딥카피 후 입력 슬롯에 저장
    {
        std::lock_guard<std::mutex> lock(async_input_mutex_);
        async_frame_buffer_.resize(frame_size);
        std::memcpy(async_frame_buffer_.data(), data, frame_size);
        async_width_ = width;
        async_height_ = height;
        async_format_ = format;
        has_new_frame_ = true;
    }

    // ③-2 B1: 워커는 slot_cv_ 단일 cv로 동기/비동기를 함께 대기한다.
    // slot_mutex_를 잠깐 잡은 채 notify하여, 워커가 predicate에서 has_new_frame_을
    // 재검사하는 시점과 직렬화해 lost-wakeup(폴링이 가리던 것)을 제거한다.
    {
        std::lock_guard<std::mutex> lock(slot_mutex_);
        slot_cv_.notify_one();
    }
}

bool InferenceThread::getLatestResult(IrisResult& out) {
    if (!has_async_result_.load()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(async_result_mutex_);
    out = async_latest_result_;
    return true;
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

        // ③-2 B1: 초기화 완료 전이를 CAS로 수행한다.
        // stop()이 Starting 중에 호출되면 state_=Stopping이 되는데, 기존 코드는
        // 무조건 state_=Idle로 덮어써 Stopping 요청을 잃고 메인 루프가 영원히
        // 돌아 join이 행(hang)으로 빠졌다. CAS가 실패하면(=Stopping) 정리 후 종료한다.
        ThreadState expected = ThreadState::Starting;
        if (!state_.compare_exchange_strong(expected, ThreadState::Idle)) {
            std::fprintf(stderr,
                         "[InferenceThread] Stop requested during init, shutting down\n");
            detector_.reset();
            state_ = ThreadState::Stopped;
            return;
        }
    } else {
        std::fprintf(stderr, "[InferenceThread] Detector initialization failed\n");
        // 초기화 실패: 스레드는 곧 반환하지만 join 전까지 joinable로 남는다.
        // stop()이 상태와 무관하게 join하도록 수정되었으므로 terminate는 없다.
        detector_.reset();
        state_ = ThreadState::Stopped;
        return;
    }

    // ========================================
    // 2. 메인 루프 - 동기 + 비동기 혼합 패턴
    // ========================================
    // ③-2 B1(minor 2): 워커 로컬 입력 버퍼를 루프 밖에서 한 번만 잡고, 매 동기
    //   요청마다 pending_data_ 와 swap 해 재사용한다. 기존 std::move + clear 패턴은
    //   pending_data_ 의 capacity 를 0으로 만들어 호출자가 매번 resize(=재할당)하게
    //   했다. swap 은 워커가 비운 버퍼의 capacity 를 pending_data_ 로 되돌려준다.
    std::vector<uint8_t> sync_frame_buffer;

    while (state_ != ThreadState::Stopping && state_ != ThreadState::Stopped) {

        // --- 동기 요청 우선 처리 ---
        {
            std::unique_lock<std::mutex> lock(slot_mutex_);

            // ③-2 B1: 동기 요청 / 비동기 프레임 / 중지를 단일 cv(slot_cv_)로 함께 대기.
            // 비동기 제출(submitFrameAsync)이 slot_cv_.notify_one()을 호출하므로
            // has_new_frame_도 wakeup 조건에 포함해 5ms 폴링 지연을 제거한다.
            auto wait_result = slot_cv_.wait_for(lock, std::chrono::milliseconds(5), [this] {
                ThreadState s = state_.load();
                return s == ThreadState::Processing ||
                       s == ThreadState::Stopping ||
                       s == ThreadState::Stopped ||
                       has_new_frame_.load();
            });

            ThreadState current = state_.load();
            if (current == ThreadState::Stopping || current == ThreadState::Stopped) {
                break;
            }

            if (wait_result && current == ThreadState::Processing) {
                // 동기 모드: 딥카피된 입력을 워커 로컬 버퍼와 swap 해 가져온다.
                // ③-2 B1(minor 2): swap 으로 pending_data_ 의 capacity 를 보존한다.
                //   (이전: std::move + clear → 호출자 측 매 호출 재할당)
                sync_frame_buffer.swap(pending_data_);
                int width = pending_width_;
                int height = pending_height_;
                int format = pending_format_;
                const uint64_t req_seq = sync_request_seq_;

                lock.unlock();

                // 검출 수행 (락 없이 - GPU delegate와 동일 스레드에서)
                IrisResult result = detector_->detect(
                    sync_frame_buffer.data(),
                    width,
                    height,
                    static_cast<FrameFormat>(format));

                // 결과 저장 및 상태 전환
                // ③-2 B1: 우리가 처리한 요청 세대가 여전히 유효할 때만 ResultReady로
                //          전이한다. detectSync가 타임아웃해 세대를 무효화했거나
                //          stop이 끼어든 경우에는 결과를 폐기해 상태 고착을 막는다.
                bool seq_valid = false;
                {
                    std::lock_guard<std::mutex> lk(slot_mutex_);
                    if (sync_request_seq_ == req_seq &&
                        state_.load() == ThreadState::Processing) {
                        pending_result_ = result;
                        state_ = ThreadState::ResultReady;
                        seq_valid = true;
                    }
                    // 세대 불일치/상태 변경 시 결과 폐기 (호출자는 이미 떠났음)
                }

                // ③-2 B1(minor 1): 랜드마크 기록은 세대 검사 통과 후로 이동한다.
                //   폐기 세대(detectSync가 이미 타임아웃해 떠난 요청)의 랜드마크가
                //   last_landmarks_ 를 덮어써 다음 getFaceLandmarks 호출자에게 노출되는
                //   결함을 차단한다. 같은 워커 스레드이며 그 사이 재추론하지 않았으므로
                //   detector_ 내부 랜드마크 상태는 detect 직후와 동일하다.
                if (seq_valid && result.detected) {
                    int count = detector_->getFaceLandmarkCount();
                    if (count > 0) {
                        std::lock_guard<std::mutex> lk(landmark_mutex_);
                        last_landmarks_.resize(static_cast<size_t>(count) * 3);
                        detector_->getFaceLandmarks(last_landmarks_.data());
                    }
                }

                slot_cv_.notify_one();
                continue;
            }
        }

        // --- 비동기 프레임 처리 ---
        if (has_new_frame_.load()) {
            std::vector<uint8_t> frame_copy;
            int width, height, format;

            // 입력 슬롯에서 딥카피
            {
                std::lock_guard<std::mutex> lock(async_input_mutex_);
                if (!has_new_frame_.load()) {
                    continue;  // 다른 스레드가 먼저 처리
                }
                frame_copy = async_frame_buffer_;  // 딥카피 (vector copy)
                width = async_width_;
                height = async_height_;
                format = async_format_;
                has_new_frame_ = false;
            }

            // 검출 수행 (락 없이 - GPU delegate와 동일 스레드에서)
            IrisResult result = detector_->detect(
                frame_copy.data(),
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

            // 비동기 결과 슬롯에 저장
            {
                std::lock_guard<std::mutex> lock(async_result_mutex_);
                async_latest_result_ = result;
                has_async_result_ = true;
            }
        }
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
