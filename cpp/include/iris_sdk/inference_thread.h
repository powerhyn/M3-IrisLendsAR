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
    // 검출 API
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
    const uint8_t* pending_data_{nullptr};
    int pending_width_{0};
    int pending_height_{0};
    int pending_format_{0};

    // 출력 슬롯 (slot_mutex_로 보호)
    IrisResult pending_result_;

    // ========================================
    // 검출기 및 설정
    // ========================================
    std::unique_ptr<MediaPipeDetector> detector_;
    std::string model_path_;
    bool gpu_enabled_{false};

    // 마지막 랜드마크 데이터 (스레드 간 공유)
    mutable std::mutex landmark_mutex_;
    std::vector<float> last_landmarks_;
};

} // namespace iris_sdk
