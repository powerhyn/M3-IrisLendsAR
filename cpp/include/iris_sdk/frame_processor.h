/**
 * @file frame_processor.h
 * @brief 프레임 처리 파이프라인 선언
 *
 * 검출과 렌더링을 통합하여 단일 인터페이스로 제공하는 파이프라인.
 * 다양한 프레임 포맷을 지원하며, 메모리 효율적 처리를 위해 버퍼 재사용.
 */

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "export.h"
#include "types.h"

// OpenCV 전방 선언
namespace cv { class Mat; }

namespace iris_sdk {

// 전방 선언
class IrisDetector;
class LensRenderer;

/**
 * @brief 프레임 처리 결과
 */
struct IRIS_SDK_EXPORT ProcessResult {
    bool success = false;           ///< 처리 성공 여부
    IrisResult iris_result;         ///< 홍채 검출 결과
    ErrorCode error_code = ErrorCode::Success; ///< 에러 코드
    float processing_time_ms = 0.0f; ///< 총 처리 시간 (밀리초)
    float detection_time_ms = 0.0f;  ///< 검출 시간 (밀리초)
    float render_time_ms = 0.0f;     ///< 렌더링 시간 (밀리초)
    float convert_time_ms = 0.0f;    ///< 포맷 변환 시간 (밀리초)
};

/**
 * @brief 프레임 처리 콜백 타입
 */
using ProcessCallback = std::function<void(const ProcessResult&)>;

// DetectorType은 types.h에 정의됨

/**
 * @brief 프레임 처리 파이프라인
 *
 * 검출과 렌더링을 통합하여 단일 인터페이스로 제공합니다.
 * 다양한 프레임 포맷(RGBA, BGR, NV21, NV12)을 지원하며,
 * 내부적으로 최적화된 버퍼 관리를 수행합니다.
 *
 * @note Pimpl 패턴으로 구현 세부사항 은닉
 * @note 스레드 안전하지 않음 - 단일 스레드에서 사용 권장
 *
 * 사용 예시:
 * @code
 * FrameProcessor processor;
 * processor.initialize("models/");
 * processor.loadLensTexture("lens.png");
 *
 * LensConfig config;
 * config.opacity = 0.8f;
 *
 * // 프레임 처리 (검출 + 렌더링)
 * ProcessResult result = processor.process(frame_data, width, height,
 *                                          FrameFormat::RGBA, &config);
 *
 * if (result.success && result.iris_result.detected) {
 *     // 처리 성공
 * }
 * @endcode
 */
class IRIS_SDK_EXPORT FrameProcessor {
public:
    FrameProcessor();
    ~FrameProcessor();

    // 복사 금지 (Pimpl 사용)
    FrameProcessor(const FrameProcessor&) = delete;
    FrameProcessor& operator=(const FrameProcessor&) = delete;

    // 이동 지원
    FrameProcessor(FrameProcessor&&) noexcept;
    FrameProcessor& operator=(FrameProcessor&&) noexcept;

    // ========================================
    // 초기화 및 해제
    // ========================================

    /**
     * @brief 프로세서 초기화
     *
     * 검출기와 렌더러를 초기화합니다.
     *
     * @param model_path 모델 파일 디렉토리 경로
     * @param detector_type 검출기 종류 (기본: MediaPipe)
     * @return 초기화 성공 여부
     */
    bool initialize(const std::string& model_path,
                    DetectorType detector_type = DetectorType::MediaPipe);

    /**
     * @brief 리소스 해제
     *
     * 모든 내부 리소스를 해제합니다.
     */
    void release();

    /**
     * @brief 초기화 상태 확인
     * @return 초기화 완료 여부
     */
    bool isInitialized() const noexcept;

    // ========================================
    // 텍스처 관리
    // ========================================

    /**
     * @brief 렌즈 텍스처 로드 (파일)
     *
     * @param texture_path 텍스처 이미지 파일 경로
     * @return 로드 성공 여부
     */
    bool loadLensTexture(const std::string& texture_path);

    /**
     * @brief 렌즈 텍스처 로드 (메모리)
     *
     * @param data RGBA 픽셀 데이터
     * @param width 텍스처 너비
     * @param height 텍스처 높이
     * @return 로드 성공 여부
     */
    bool loadLensTexture(const uint8_t* data, int width, int height);

    /**
     * @brief 텍스처 언로드
     */
    void unloadLensTexture();

    /**
     * @brief 텍스처 로드 상태 확인
     * @return 텍스처 로드 완료 여부
     */
    bool hasLensTexture() const noexcept;

    // ========================================
    // 프레임 처리
    // ========================================

    /**
     * @brief 프레임 처리 (검출 + 렌더링)
     *
     * 프레임 데이터에서 홍채를 검출하고, 설정에 따라 렌즈를 렌더링합니다.
     * 프레임 데이터는 in-place로 수정됩니다.
     *
     * @param frame_data 프레임 데이터 (in-place 수정됨)
     * @param width 프레임 너비
     * @param height 프레임 높이
     * @param format 픽셀 포맷
     * @param config 렌더링 설정 (nullptr이면 검출만 수행)
     * @return 처리 결과
     */
    ProcessResult process(uint8_t* frame_data,
                          int width,
                          int height,
                          FrameFormat format,
                          const LensConfig* config = nullptr);

    /**
     * @brief cv::Mat 프레임 처리 (검출 + 렌더링)
     *
     * OpenCV Mat을 직접 처리합니다. BGR/BGRA 포맷 지원.
     *
     * @param frame 입출력 이미지 (in-place 수정됨)
     * @param config 렌더링 설정 (nullptr이면 검출만 수행)
     * @return 처리 결과
     */
    ProcessResult process(cv::Mat& frame,
                          const LensConfig* config = nullptr);

    /**
     * @brief 검출만 수행
     *
     * 렌더링 없이 홍채 검출만 수행합니다.
     *
     * @param frame_data 프레임 데이터 (읽기 전용)
     * @param width 프레임 너비
     * @param height 프레임 높이
     * @param format 픽셀 포맷
     * @return 홍채 검출 결과
     */
    IrisResult detectOnly(const uint8_t* frame_data,
                          int width,
                          int height,
                          FrameFormat format);

    /**
     * @brief 렌더링만 수행
     *
     * 기존 검출 결과를 사용하여 렌더링만 수행합니다.
     *
     * @param frame_data 프레임 데이터 (in-place 수정됨)
     * @param width 프레임 너비
     * @param height 프레임 높이
     * @param format 픽셀 포맷
     * @param iris_result 홍채 검출 결과
     * @param config 렌더링 설정
     * @return 렌더링 성공 여부
     */
    bool renderOnly(uint8_t* frame_data,
                    int width,
                    int height,
                    FrameFormat format,
                    const IrisResult& iris_result,
                    const LensConfig& config);

    // ========================================
    // 설정
    // ========================================

    /**
     * @brief 검출 신뢰도 임계값 설정
     *
     * @param min_confidence 최소 신뢰도 (0.0 ~ 1.0)
     */
    void setMinConfidence(float min_confidence);

    /**
     * @brief 얼굴 추적 활성화/비활성화
     *
     * @param enable 추적 활성화 여부
     */
    void setFaceTracking(bool enable);

    // ========================================
    // 통계
    // ========================================

    /**
     * @brief 마지막 처리 시간 조회
     * @return 마지막 처리 시간 (밀리초)
     */
    double getLastProcessingTimeMs() const noexcept;

    /**
     * @brief 평균 FPS 조회
     *
     * 최근 처리 기록 기반 평균 FPS를 반환합니다.
     *
     * @return 평균 FPS (처리 기록이 없으면 0.0)
     */
    double getAverageFPS() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace iris_sdk
