/**
 * @file mediapipe_detector.h
 * @brief MediaPipe 기반 홍채 검출기 선언
 *
 * TensorFlow Lite를 사용하여 MediaPipe의 Face Mesh 및
 * Iris Landmark 모델을 실행하는 IrisDetector 구현체
 */

#pragma once

#include "iris_sdk/iris_detector.h"
#include <memory>

namespace iris_sdk {

/**
 * @brief MediaPipe 기반 홍채 검출기
 *
 * MediaPipe Face Mesh + Iris Landmark 모델을 사용한 홍채 검출
 *
 * @note TensorFlow Lite 런타임 필요
 * @note 모델 파일 필요:
 *       - face_detection_short_range.tflite
 *       - face_landmark.tflite
 *       - iris_landmark.tflite
 */
class IRIS_SDK_EXPORT MediaPipeDetector : public IrisDetector {
public:
    MediaPipeDetector();
    ~MediaPipeDetector() override;

    // 복사/이동 금지 (Pimpl 사용)
    MediaPipeDetector(const MediaPipeDetector&) = delete;
    MediaPipeDetector& operator=(const MediaPipeDetector&) = delete;
    MediaPipeDetector(MediaPipeDetector&&) = delete;
    MediaPipeDetector& operator=(MediaPipeDetector&&) = delete;

    // ========================================
    // IrisDetector 인터페이스 구현
    // ========================================

    /**
     * @brief 검출기 초기화
     * @param model_path 모델 파일들이 있는 디렉토리 경로
     * @return 초기화 성공 여부
     */
    bool initialize(const std::string& model_path) override;

    /**
     * @brief 홍채 검출 수행
     * @param frame_data 이미지 데이터 포인터
     * @param width 이미지 너비
     * @param height 이미지 높이
     * @param format 이미지 포맷
     * @return 검출 결과
     */
    IrisResult detect(const uint8_t* frame_data,
                      int width,
                      int height,
                      FrameFormat format) override;

    /**
     * @brief 리소스 해제
     */
    void release() override;

    /**
     * @brief 초기화 상태 확인
     */
    bool isInitialized() const override;

    /**
     * @brief 검출기 타입 반환
     */
    DetectorType getDetectorType() const override;

    // ========================================
    // MediaPipe 전용 설정
    // ========================================

    /**
     * @brief 얼굴 검출 최소 신뢰도 설정
     * @param confidence 신뢰도 (0.0 ~ 1.0)
     */
    void setMinDetectionConfidence(float confidence);

    /**
     * @brief 추적 최소 신뢰도 설정
     * @param confidence 신뢰도 (0.0 ~ 1.0)
     */
    void setMinTrackingConfidence(float confidence);

    /**
     * @brief 검출할 최대 얼굴 수 설정
     * @param num_faces 얼굴 수 (1 이상)
     */
    void setNumFaces(int num_faces);

    // ========================================
    // 성능 최적화 설정
    // ========================================

    /**
     * @brief TFLite 추론에 사용할 스레드 수 설정
     *
     * 초기화 전에 호출해야 효과적임.
     * 기본값: 4
     *
     * @param num_threads 스레드 수 (1 이상)
     */
    void setNumThreads(int num_threads);

    /**
     * @brief 추적 모드 활성화/비활성화
     *
     * 추적 모드 활성화 시 연속 프레임에서 Face Detection을 스킵하여
     * 성능을 향상시킴. 얼굴이 급격히 움직이는 경우 정확도가 떨어질 수 있음.
     * 기본값: true
     *
     * @param enable 활성화 여부
     */
    void setTrackingEnabled(bool enable);

    /**
     * @brief 추적 캐시 초기화
     *
     * 이전 프레임 결과를 무효화하고 다음 detect() 호출에서
     * Face Detection을 강제로 수행함.
     */
    void resetTracking();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace iris_sdk
