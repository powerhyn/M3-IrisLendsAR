/**
 * @file beauty_backend.h
 * @brief 뷰티 필터 백엔드 전략 인터페이스
 *
 * CPU (OpenCV) 또는 GPU (OpenGL ES/Metal) 구현을 추상화하는
 * Strategy 패턴 기반 인터페이스입니다.
 */

#ifndef IRIS_SDK_BEAUTY_BACKEND_H
#define IRIS_SDK_BEAUTY_BACKEND_H

#include "sdk_api.h"
#include "beauty_filter.h"
#include "beauty_roi_manager.h"
#include "gpu/texture_handle.h"

namespace iris_sdk {

// Forward declarations
class IRenderContext;

/**
 * @brief 뷰티 필터 백엔드 전략 인터페이스
 *
 * CPU (OpenCV) 또는 GPU (OpenGL ES/Metal) 구현을 추상화합니다.
 * 각 플랫폼별로 적절한 백엔드를 선택하여 사용합니다.
 */
class IBeautyBackend {
public:
    virtual ~IBeautyBackend() = default;

    //=========================================================================
    // 라이프사이클
    //=========================================================================

    /**
     * @brief 백엔드 초기화
     *
     * @param render_context 렌더링 컨텍스트 (GPU 백엔드에서 사용)
     * @return 성공 여부
     */
    virtual bool initialize(IRenderContext* render_context = nullptr) = 0;

    /**
     * @brief 리소스 해제
     */
    virtual void release() = 0;

    /**
     * @brief 초기화 여부 확인
     */
    virtual bool isInitialized() const = 0;

    //=========================================================================
    // 필터 적용 (CPU 버퍼)
    //=========================================================================

    /**
     * @brief 필터 적용 (in-place)
     *
     * @param frame_data 프레임 데이터 (입출력)
     * @param width 프레임 너비
     * @param height 프레임 높이
     * @param format 픽셀 포맷
     * @param config 필터 설정
     * @param roi ROI 정보 (nullptr이면 전체 프레임)
     * @return 에러 코드
     */
    virtual IrisSdkError apply(
        uint8_t* frame_data,
        int width, int height,
        IrisFrameFormat format,
        const BeautyFilterConfigV2& config,
        const BeautyROI* roi = nullptr
    ) = 0;

    //=========================================================================
    // 필터 적용 (텍스처)
    //=========================================================================

    /**
     * @brief 텍스처 기반 필터 적용 (GPU용)
     *
     * GPU 백엔드에서 Zero-Copy 처리를 위해 사용합니다.
     *
     * @param input 입력 텍스처
     * @param output 출력 텍스처
     * @param config 필터 설정
     * @param roi ROI 정보
     * @return 에러 코드
     */
    virtual IrisSdkError applyTexture(
        const TextureHandle& input,
        TextureHandle& output,
        const BeautyFilterConfigV2& config,
        const BeautyROI* roi = nullptr
    ) {
        // 기본 구현: 지원하지 않음
        (void)input;
        (void)output;
        (void)config;
        (void)roi;
        return IRIS_SDK_ERROR_NOT_SUPPORTED;
    }

    //=========================================================================
    // 메타데이터
    //=========================================================================

    /**
     * @brief 백엔드 이름 반환
     */
    virtual const char* getName() const = 0;

    /**
     * @brief GPU 지원 여부
     */
    virtual bool supportsGpu() const = 0;

    /**
     * @brief 텍스처 처리 지원 여부
     */
    virtual bool supportsTextureProcessing() const { return false; }
};

} // namespace iris_sdk

#endif // IRIS_SDK_BEAUTY_BACKEND_H
