/**
 * @file beauty_processor.h
 * @brief 뷰티 필터 통합 프로세서
 *
 * DI(Dependency Injection) 기반 설계로 CPU/GPU 백엔드를 자동 선택하고
 * ROI 캐싱을 지원하는 통합 프로세서입니다.
 */

#ifndef IRIS_SDK_BEAUTY_PROCESSOR_H
#define IRIS_SDK_BEAUTY_PROCESSOR_H

#include "beauty_backend.h"
#include "beauty_roi_manager.h"
#include "gpu/render_context.h"
#include <memory>
#include <mutex>

namespace iris_sdk {

/**
 * @brief 뷰티 필터 통합 프로세서
 *
 * - DI(Dependency Injection) 기반 설계
 * - CPU/GPU 백엔드 자동 선택
 * - ROI 캐싱 지원
 * - LensRenderer와 RenderContext 공유 가능
 */
class BeautyProcessor {
public:
    /**
     * @brief 생성자 (DI)
     *
     * @param render_context 렌더링 컨텍스트 (공유, nullptr이면 내부 생성)
     */
    explicit BeautyProcessor(std::shared_ptr<IRenderContext> render_context = nullptr);

    ~BeautyProcessor();

    // 복사 금지
    BeautyProcessor(const BeautyProcessor&) = delete;
    BeautyProcessor& operator=(const BeautyProcessor&) = delete;

    //=========================================================================
    // 초기화
    //=========================================================================

    /**
     * @brief 초기화
     *
     * @param prefer_gpu GPU 선호 여부
     * @return 성공 여부
     */
    bool initialize(bool prefer_gpu = true);

    /**
     * @brief 리소스 해제
     */
    void release();

    /**
     * @brief 초기화 여부 확인
     */
    bool isInitialized() const;

    //=========================================================================
    // 설정
    //=========================================================================

    /**
     * @brief 필터 설정 적용
     *
     * @param config 필터 설정
     * @return 에러 코드
     */
    IrisSdkError setConfig(const BeautyFilterConfigV2& config);

    /**
     * @brief 현재 필터 설정 조회
     *
     * @param out_config 출력 설정
     * @return 에러 코드
     */
    IrisSdkError getConfig(BeautyFilterConfigV2& out_config) const;

    //=========================================================================
    // 필터 적용 (CPU 버퍼)
    //=========================================================================

    /**
     * @brief 프레임에 필터 적용
     *
     * Face Mesh가 제공되면 ROI 기반 처리를 수행합니다.
     *
     * @param frame_data 프레임 데이터 (입출력)
     * @param width 프레임 너비
     * @param height 프레임 높이
     * @param format 픽셀 포맷
     * @param iris_result 홍채 검출 결과 (Face Mesh 포함, nullptr 가능)
     * @return 에러 코드
     */
    IrisSdkError process(
        uint8_t* frame_data,
        int width, int height,
        IrisFrameFormat format,
        const IrisResult* iris_result = nullptr
    );

    //=========================================================================
    // 필터 적용 (텍스처)
    //=========================================================================

    /**
     * @brief 텍스처 기반 필터 적용 (Zero-Copy)
     *
     * LensRenderer와 RenderContext 공유 시 사용합니다.
     *
     * @param input 입력 텍스처
     * @param output 출력 텍스처
     * @param iris_result 홍채 검출 결과
     * @return 에러 코드
     */
    IrisSdkError processTexture(
        const TextureHandle& input,
        TextureHandle& output,
        const IrisResult* iris_result = nullptr
    );

    //=========================================================================
    // 상태 조회
    //=========================================================================

    /**
     * @brief GPU 사용 여부
     */
    bool isUsingGpu() const;

    /**
     * @brief 필터 활성화 여부
     */
    bool isEnabled() const;

    /**
     * @brief 백엔드 이름 조회
     */
    const char* getBackendName() const;

    //=========================================================================
    // RenderContext 접근
    //=========================================================================

    /**
     * @brief RenderContext 공유
     */
    std::shared_ptr<IRenderContext> getRenderContext() const { return render_context_; }

private:
    /**
     * @brief ROI 계산 (캐싱 포함)
     */
    bool computeROI(const IrisResult* iris_result, int width, int height,
                    BeautyROI& out_roi);

    /**
     * @brief 백엔드 선택
     */
    bool selectBackend(bool prefer_gpu);

    /**
     * @brief 현재 시간 (밀리초)
     */
    static int64_t getCurrentTimeMs();

    std::shared_ptr<IRenderContext> render_context_;
    std::unique_ptr<IBeautyBackend> backend_;
    BeautyFilterConfigV2 config_;

    // ROI 캐시
    BeautyROI cached_roi_;
    int64_t cached_roi_timestamp_ = 0;
    static constexpr int64_t ROI_CACHE_TIMEOUT_MS = 100;

    mutable std::mutex mutex_;
    bool initialized_ = false;
};

} // namespace iris_sdk

#endif // IRIS_SDK_BEAUTY_PROCESSOR_H
