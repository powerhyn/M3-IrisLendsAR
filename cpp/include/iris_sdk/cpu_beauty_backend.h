/**
 * @file cpu_beauty_backend.h
 * @brief CPU 기반 뷰티 필터 백엔드 (OpenCV)
 *
 * 기존 BeautyFilter 기능을 IBeautyBackend 인터페이스로 래핑합니다.
 */

#ifndef IRIS_SDK_CPU_BEAUTY_BACKEND_H
#define IRIS_SDK_CPU_BEAUTY_BACKEND_H

#include "beauty_backend.h"
#include <opencv2/core.hpp>
#include <mutex>

namespace iris_sdk {

/**
 * @brief CPU 기반 뷰티 필터 백엔드 (OpenCV)
 *
 * 기존 BeautyFilter 로직을 IBeautyBackend 인터페이스로 구현합니다.
 * 모든 플랫폼에서 사용 가능한 폴백 백엔드입니다.
 */
class CPUBeautyBackend : public IBeautyBackend {
public:
    CPUBeautyBackend();
    ~CPUBeautyBackend() override;

    // 복사/이동 금지
    CPUBeautyBackend(const CPUBeautyBackend&) = delete;
    CPUBeautyBackend& operator=(const CPUBeautyBackend&) = delete;

    //=========================================================================
    // IBeautyBackend 구현
    //=========================================================================

    bool initialize(IRenderContext* render_context = nullptr) override;
    void release() override;
    bool isInitialized() const override;

    IrisSdkError apply(
        uint8_t* frame_data,
        int width, int height,
        IrisFrameFormat format,
        const BeautyFilterConfigV2& config,
        const BeautyROI* roi = nullptr
    ) override;

    const char* getName() const override { return "CPUBeautyBackend"; }
    bool supportsGpu() const override { return false; }

private:
    //=========================================================================
    // 필터 효과 함수
    //=========================================================================

    /**
     * @brief 피부 스무딩 적용 (Bilateral Filter)
     */
    void applySkinSmoothing(cv::Mat& frame, float strength);

    /**
     * @brief 피부 스무딩 적용 (보호 마스크 포함)
     *
     * @param frame 입력/출력 프레임
     * @param strength 스무딩 강도 (0.0 ~ 1.0)
     * @param protection_mask 보호 마스크 (눈/입술 등)
     */
    void applySkinSmoothing(cv::Mat& frame, float strength, const cv::Mat& protection_mask);

    /**
     * @brief 화이트닝 효과 적용 (보호 마스크 포함)
     *
     * @param frame 입력/출력 프레임
     * @param strength 화이트닝 강도
     * @param protection_mask 보호 마스크
     */
    void applyWhitening(cv::Mat& frame, float strength, const cv::Mat& protection_mask);

    /**
     * @brief 소프트 포커스 효과 적용
     */
    void applySoftFocus(cv::Mat& frame, float strength);

    /**
     * @brief 밝기 조정
     */
    void applyBrightness(cv::Mat& frame, float brightness);

    /**
     * @brief 화이트닝 효과 적용 (LAB 색공간)
     */
    void applyWhitening(cv::Mat& frame, float strength);

    /**
     * @brief 색상 밸런스 조정
     */
    void applyColorBalance(cv::Mat& frame, float balance);

    //=========================================================================
    // ROI 기반 처리
    //=========================================================================

    /**
     * @brief ROI 영역에만 필터 적용
     *
     * @param frame 전체 프레임
     * @param config 필터 설정
     * @param roi ROI 정보
     * @return 에러 코드
     */
    IrisSdkError applyWithROI(
        cv::Mat& frame,
        const BeautyFilterConfigV2& config,
        const BeautyROI& roi
    );

    /**
     * @brief 전체 프레임에 필터 적용
     */
    IrisSdkError applyFullFrame(
        cv::Mat& frame,
        const BeautyFilterConfigV2& config
    );

    //=========================================================================
    // 포맷 변환 헬퍼
    //=========================================================================

    /**
     * @brief 프레임 데이터를 BGR Mat로 변환
     */
    static cv::Mat convertToBGR(const uint8_t* data, int width, int height,
                                IrisFrameFormat format);

    /**
     * @brief BGR Mat를 원본 포맷으로 변환
     */
    static void convertFromBGR(const cv::Mat& bgr, uint8_t* data,
                               int width, int height, IrisFrameFormat format);

    //=========================================================================
    // 멤버 변수
    //=========================================================================

    cv::Mat work_buffer_;      ///< 작업 버퍼
    cv::Mat smooth_buffer_;    ///< 스무딩 버퍼
    cv::Mat roi_buffer_;       ///< ROI 작업 버퍼
    mutable std::mutex mutex_; ///< 스레드 안전성
    bool initialized_ = false;
};

} // namespace iris_sdk

#endif // IRIS_SDK_CPU_BEAUTY_BACKEND_H
