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
class LensRenderer;

/**
 * @brief 프레임 처리 파이프라인 (render-only)
 *
 * ④ W4-D: 검출 인프라(detector/InferenceThread)를 코어에서 제거하면서
 * 렌더링 전용 파이프라인으로 축소되었습니다. 검출(랜드마크)은 외부 추적 글루가
 * 책임지며 주입 경로(landmark_injection)로 코어에 전달됩니다.
 * 다양한 프레임 포맷(RGBA, BGR, NV21, NV12)을 지원하며,
 * 내부적으로 최적화된 버퍼 관리를 수행합니다.
 *
 * @note Pimpl 패턴으로 구현 세부사항 은닉
 * @note 스레드 안전하지 않음 - 단일 스레드에서 사용 권장
 *
 * 사용 예시:
 * @code
 * FrameProcessor processor;
 * processor.initialize();
 * processor.loadLensTexture("lens.png");
 *
 * LensConfig config;
 * config.opacity = 0.8f;
 *
 * // 외부에서 주입된 검출 결과로 렌더링만 수행
 * processor.renderOnly(frame_data, width, height,
 *                      FrameFormat::RGBA, iris_result, config);
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
     * @brief 프로세서 초기화 (render-only)
     *
     * 렌더러를 초기화합니다. ④ W4-D에서 검출 인프라가 제거되어
     * 모델 경로/검출기 타입 인자가 더 이상 필요하지 않습니다.
     *
     * @return 초기화 성공 여부
     */
    bool initialize();

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

    /**
     * @brief 기존 결과로 렌더링 (주입 워크플로우용)
     *
     * 외부에서 주입된 검출 결과를 사용하여 렌더링만 수행합니다.
     * renderOnly()와 동일한 동작이지만, 명시적 이름.
     *
     * @param frame_data 프레임 데이터 (in-place 수정됨)
     * @param width 프레임 너비
     * @param height 프레임 높이
     * @param format 픽셀 포맷
     * @param iris_result 홍채 검출 결과
     * @param config 렌더링 설정
     * @return 렌더링 성공 여부
     */
    bool renderWithResult(uint8_t* frame_data, int width, int height,
                           FrameFormat format, const IrisResult& iris_result,
                           const LensConfig& config);

    // ========================================
    // GPU 가속
    // ========================================

    /**
     * @brief GPU 가속 사용 여부 설정
     *
     * 반드시 initialize() 호출 전에 설정해야 합니다.
     *
     * @param enable true면 GPU 가속 시도, false면 CPU만 사용
     */
    void setGpuEnabled(bool enable);

    /**
     * @brief 현재 GPU 사용 상태 확인
     *
     * @return true면 GPU 사용 중, false면 CPU 사용 중
     */
    bool isUsingGpu() const noexcept;

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
