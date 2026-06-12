/**
 * @file inference_thread.h
 * @brief 전용 추론 스레드 클래스 선언
 *
 * TFLite GPU delegate는 초기화된 스레드에서만 Invoke() 가능.
 * 이 클래스는 전용 스레드에서 TFLite 초기화와 추론을 수행하여
 * GPU 가속을 안전하게 사용할 수 있게 합니다.
 *
 * 최적화: 단일 슬롯 패턴으로 동기화 오버헤드 최소화
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "export.h"
#include "types.h"

namespace iris_sdk {

// 전방 선언
class MediaPipeDetector;

/**
 * @brief 스레드 상태 열거형
 *
 * 단일 원자적 변수로 스레드 상태를 관리합니다.
 */
enum class ThreadState : int {
    Stopped = 0,     ///< 스레드 정지됨
    Starting = 1,    ///< 초기화 중
    Idle = 2,        ///< 요청 대기 중
    Processing = 3,  ///< 추론 실행 중
    ResultReady = 4, ///< 결과 준비됨
    Stopping = 5     ///< 종료 중
};

/**
 * @brief 전용 추론 스레드 클래스 (최적화 버전)
 *
 * TFLite 인터프리터와 GPU delegate를 전용 스레드에서 관리합니다.
 * 단일 슬롯 패턴으로 동기화 오버헤드를 최소화합니다.
 *
 * 동작 원리:
 * 1. start() 호출 시 전용 스레드 생성
 * 2. 스레드 내에서 MediaPipeDetector 초기화 (GPU delegate 포함)
 * 3. detectSync() 호출 시 단일 슬롯에 데이터 설정 후 결과 대기
 * 4. 스레드 내에서 검출 수행 후 결과 반환
 * 5. stop() 호출 시 스레드 종료
 *
 * @note 단일 호출자용 - 동시에 하나의 detectSync() 호출만 지원
 */
class IRIS_SDK_EXPORT InferenceThread {
public:
    InferenceThread();
    ~InferenceThread();

    // 복사 금지
    InferenceThread(const InferenceThread&) = delete;
    InferenceThread& operator=(const InferenceThread&) = delete;

    // ========================================
    // 스레드 제어
    // ========================================

    /**
     * @brief 추론 스레드 시작
     *
     * 전용 스레드를 생성하고 그 안에서 MediaPipeDetector를 초기화합니다.
     * GPU delegate가 활성화된 경우, 이 스레드에서 GPU 컨텍스트가 생성됩니다.
     *
     * @param model_path 모델 파일 디렉토리 경로
     * @param gpu_enabled GPU 가속 사용 여부
     * @return 시작 성공 여부
     */
    bool start(const std::string& model_path, bool gpu_enabled);

    /**
     * @brief 추론 스레드 중지
     *
     * 진행 중인 작업 완료 후 스레드를 종료합니다.
     *
     * @note 스레드-안전: lifecycle_mutex_로 직렬화되므로 여러 스레드가 동시에
     *       stop()을 호출해도 이중 join(UB) 없이 안전하다.
     */
    void stop();

    /**
     * @brief 스레드 실행 상태 확인
     * @return 스레드가 실행 중이면 true
     */
    bool isRunning() const noexcept;

    /**
     * @brief 초기화 완료 대기
     *
     * 스레드 내 초기화가 완료될 때까지 대기합니다.
     *
     * @param timeout_ms 타임아웃 (밀리초, 0이면 무한 대기)
     * @return 초기화 성공 여부
     */
    bool waitForInitialization(int timeout_ms = 5000);

    // ========================================
    // 검출 API (동기)
    // ========================================

    /**
     * @brief 동기 검출 (블로킹)
     *
     * 프레임을 전용 스레드로 전달하고 결과를 대기합니다.
     * 가장 간단한 사용 방법으로, 기존 API와 호환됩니다.
     *
     * @param data 프레임 데이터
     * @param width 프레임 너비
     * @param height 프레임 높이
     * @param format 픽셀 포맷 (FrameFormat enum)
     * @return 홍채 검출 결과
     */
    IrisResult detectSync(const uint8_t* data, int width, int height, int format);

    // NOTE: 회전 처리는 FrameProcessor에서 수행 후 이 클래스에 전달합니다.
    //       따라서 별도의 회전 API는 제공하지 않습니다.

    // ========================================
    // 검출 API (비동기) — P5-W1-04
    // ========================================

    /**
     * @brief 비동기 프레임 제출 (논블로킹)
     *
     * 프레임 데이터를 내부 최신 프레임 슬롯에 딥카피합니다.
     * 이전에 제출된 미처리 프레임은 덮어씌워집니다 (drop-oldest).
     * 워커 스레드가 깨어나 최신 프레임으로 추론을 수행합니다.
     *
     * @param data 프레임 데이터 (RGB 포맷, 호출자 버퍼 재사용 가능)
     * @param width 프레임 너비
     * @param height 프레임 높이
     * @param format 픽셀 포맷 (FrameFormat enum)
     */
    void submitFrameAsync(const uint8_t* data, int width, int height, int format);

    /**
     * @brief 최신 추론 결과 조회 (논블로킹)
     *
     * 워커 스레드가 마지막으로 완료한 추론 결과를 반환합니다.
     * 아직 결과가 없으면 false를 반환합니다.
     *
     * @param out 결과 출력
     * @return 유효한 결과가 있으면 true, 없으면 false
     */
    bool getLatestResult(IrisResult& out);

    // ========================================
    // 상태 조회
    // ========================================

    /**
     * @brief GPU 활성화 상태 확인
     * @return GPU가 실제로 사용 중이면 true
     */
    bool isGpuActive() const noexcept;

    /**
     * @brief 모델 버전 조회
     * @return 1: V1, 2: V2
     */
    int getModelVersion() const noexcept;

    /**
     * @brief 얼굴 랜드마크 수 조회
     * @return 랜드마크 수 (468 또는 478)
     */
    int getFaceLandmarkCount() const noexcept;

    /**
     * @brief 얼굴 랜드마크 데이터 복사
     * @param out_landmarks 출력 버퍼
     * @return 성공 여부
     */
    bool getFaceLandmarks(float* out_landmarks) const;

private:
    /**
     * @brief 스레드 메인 루프
     *
     * 스레드 내에서 실행되며, 다음을 수행:
     * 1. MediaPipeDetector 초기화
     * 2. 단일 슬롯에서 요청 수신
     * 3. 검출 수행
     * 4. 결과 저장
     */
    void threadLoop();

    /**
     * @brief 프레임 제출 및 결과 대기 (내부용)
     */
    IrisResult submitAndWait(const uint8_t* data, int width, int height, int format);

    /**
     * @brief stop()의 실제 중지 로직 (lifecycle_mutex_ 보유 가정)
     *
     * NOTE(③-2 B1): start()의 재시작 정리 경로가 stop()을 호출하면 lifecycle_mutex_를
     * 이미 보유한 상태에서 재진입해 재귀 데드락(std::mutex는 비재귀)이 발생한다.
     * 따라서 락을 잡지 않는 stopLocked()로 실제 로직을 분리하고, public stop()은
     * 락을 잡은 뒤 위임하는 얇은 래퍼로 둔다. 호출자는 반드시 lifecycle_mutex_를
     * 보유한 채로 호출해야 한다.
     */
    void stopLocked();

    // ========================================
    // 스레드 제어 직렬화 (③-2 B1)
    // ========================================
    // NOTE(③-2 B1): thread_ / model_path_ / gpu_enabled_ 는 제어 스레드(start/stop)
    //               전용 상태다. 이 뮤텍스로 start()의 thread_ move-대입과 stop()의
    //               joinable()/join()을 직렬화해 데이터 레이스와 이중 join(UB)을 막는다.
    //               ★ 락 순서 규약: 워커 스레드(threadLoop)는 이 뮤텍스를 절대 잡지
    //                 않는다. lifecycle_mutex_는 항상 slot_mutex_/async_*_mutex_/
    //                 landmark_mutex_보다 바깥(먼저 획득)에 위치하며, 제어 스레드만
    //                 사용하므로 워커가 잡는 락들과 교착 사이클을 형성하지 않는다.
    std::mutex lifecycle_mutex_;

    // ========================================
    // 스레드 상태 (단일 원자적 변수)
    // ========================================
    std::thread thread_;
    std::atomic<ThreadState> state_{ThreadState::Stopped};

    // GPU 상태
    std::atomic<bool> gpu_active_{false};
    std::atomic<int> model_version_{0};
    std::atomic<int> landmark_count_{0};

    // ========================================
    // 단일 슬롯 패턴 (동기화)
    // ========================================
    std::mutex slot_mutex_;
    std::condition_variable slot_cv_;

    // 입력 슬롯 (slot_mutex_로 보호)
    // NOTE(③-2 B1): 호출자 raw 포인터를 그대로 워커가 읽으면 타임아웃 후 호출자
    //               버퍼가 재사용/재할당될 때 데이터 레이스 또는 UAF가 발생한다.
    //               따라서 제출 시점에 입력을 딥카피해 워커에 전달한다.
    std::vector<uint8_t> pending_data_;
    int pending_width_{0};
    int pending_height_{0};
    int pending_format_{0};

    // 출력 슬롯 (slot_mutex_로 보호)
    IrisResult pending_result_;

    // 동기 요청 세대 번호 (slot_mutex_로 보호)
    // NOTE(③-2 B1): detectSync 타임아웃 후 뒤늦게 완료된 워커가 ResultReady로
    //               상태를 고착시키지 못하도록, 요청마다 세대를 부여하고 워커는
    //               자신이 처리한 세대가 여전히 유효할 때만 결과를 기록한다.
    uint64_t sync_request_seq_{0};

    // ========================================
    // 검출기 및 설정
    // ========================================
    std::unique_ptr<MediaPipeDetector> detector_;
    std::string model_path_;
    bool gpu_enabled_{false};

    // 마지막 랜드마크 데이터 (스레드 간 공유)
    mutable std::mutex landmark_mutex_;
    std::vector<float> last_landmarks_;

    // ========================================
    // 비동기 슬롯 (P5-W1-04)
    // ========================================

    // 비동기 입력 슬롯 (async_input_mutex_로 보호)
    // NOTE(③-2 B1): 비동기/동기 wakeup은 slot_cv_ 단일 cv로 통합되었으므로
    //               별도의 async_cv_는 제거되었다(죽은 대기자 제거).
    std::mutex async_input_mutex_;
    std::vector<uint8_t> async_frame_buffer_;  ///< 딥카피된 프레임 데이터
    int async_width_{0};
    int async_height_{0};
    int async_format_{0};
    std::atomic<bool> has_new_frame_{false};

    // 비동기 출력 슬롯 (async_result_mutex_로 보호)
    mutable std::mutex async_result_mutex_;
    IrisResult async_latest_result_;
    std::atomic<bool> has_async_result_{false};
};

} // namespace iris_sdk
