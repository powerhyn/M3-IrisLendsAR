/**
 * @file cpu_beauty_backend.h
 * @brief CPU 기반 뷰티 필터 백엔드 (OpenCV)
 *
 * 기존 BeautyFilter 기능을 IBeautyBackend 인터페이스로 래핑합니다.
 */

#ifndef IRIS_SDK_CPU_BEAUTY_BACKEND_H
#define IRIS_SDK_CPU_BEAUTY_BACKEND_H

#include "beauty_backend.h"
#include "types.h"
#include <opencv2/core.hpp>
#include <mutex>

namespace iris_sdk {

/**
 * @brief 주름 영역 마스크 구조체
 *
 * 얼굴 랜드마크 기반으로 주름이 발생하는 주요 영역의 마스크를 저장합니다.
 */
struct WrinkleRegions {
    cv::Mat forehead_mask;      ///< 이마 영역 마스크
    cv::Mat crow_feet_mask;     ///< 눈가 주름 영역 마스크
    cv::Mat frown_lines_mask;   ///< 미간 주름 영역 마스크
    cv::Mat combined;           ///< 모든 주름 영역 결합 마스크
};

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
    // V2 필터 효과 함수 (Guided Filter 기반)
    //=========================================================================

    /**
     * @brief 피부 스무딩 V2 (Guided Filter 기반)
     *
     * FastGuidedFilter를 사용하여 에지 보존 스무딩을 적용합니다.
     * Bilateral Filter보다 빠르고 자연스러운 결과를 제공합니다.
     *
     * @param frame 입력/출력 프레임
     * @param strength 스무딩 강도 (0.0 ~ 1.0)
     * @param protection_mask 보호 마스크 (눈/입술 등)
     */
    void applySkinSmoothingV2(cv::Mat& frame, float strength, const cv::Mat& protection_mask);

    /**
     * @brief 소프트 포커스 V2 (Guided Filter + 오버레이 블렌딩)
     *
     * Guided Filter로 에지 보존 스무딩 후 오버레이 블렌딩으로
     * 하이라이트를 강조하여 자연스러운 글로우 효과를 만듭니다.
     *
     * @param frame 입력/출력 프레임
     * @param strength 소프트 포커스 강도 (0.0 ~ 1.0)
     */
    void applySoftFocusV2(cv::Mat& frame, float strength);

    /**
     * @brief 밝기 V2 (하이라이트 보호)
     *
     * LAB 색상 공간에서 비선형 밝기 조정을 적용하여
     * 하이라이트 영역의 클리핑을 방지합니다.
     *
     * @param frame 입력/출력 프레임
     * @param brightness 밝기 조정값 (0.5~1.5, 1.0=원본)
     */
    void applyBrightnessV2(cv::Mat& frame, float brightness);

    /**
     * @brief 주름 제거 (타겟 스무딩)
     *
     * 얼굴 랜드마크를 기반으로 주름이 발생하는 영역
     * (이마, 눈가, 미간)을 선택적으로 스무딩합니다.
     *
     * @param frame 입력/출력 프레임
     * @param strength 주름 제거 강도 (0.0 ~ 1.0)
     * @param face_mesh 얼굴 랜드마크 배열 (478개)
     * @param landmark_count 랜드마크 개수
     * @param offset_x ROI 영역의 X 오프셋
     * @param offset_y ROI 영역의 Y 오프셋
     */
    void applyWrinkleRemoval(cv::Mat& frame, float strength,
                             const IrisLandmark* face_mesh, int landmark_count,
                             int offset_x, int offset_y);

    //=========================================================================
    // 헬퍼 함수
    //=========================================================================

    /**
     * @brief LAB 기반 피부톤 감지
     *
     * LAB 색상 공간의 A, B 채널을 분석하여 피부톤 영역을 감지합니다.
     *
     * @param A_channel LAB A 채널
     * @param B_channel LAB B 채널
     * @return 피부톤 마스크 (CV_8UC1)
     */
    static cv::Mat detectSkinTone(const cv::Mat& A_channel, const cv::Mat& B_channel);

    /**
     * @brief 오버레이 블렌딩
     *
     * 포토샵 스타일의 오버레이 블렌드 모드를 적용합니다.
     * 어두운 영역은 더 어둡게, 밝은 영역은 더 밝게 만듭니다.
     *
     * @param base 베이스 이미지
     * @param blend 블렌드 이미지
     * @param result 결과 이미지
     * @param opacity 블렌드 불투명도 (0.0 ~ 1.0)
     */
    static void overlayBlend(const cv::Mat& base, const cv::Mat& blend,
                             cv::Mat& result, float opacity);

    /**
     * @brief 주름 영역 마스크 생성
     *
     * 얼굴 랜드마크를 기반으로 주름이 발생하는 주요 영역의 마스크를 생성합니다.
     *
     * @param face_mesh 얼굴 랜드마크 배열
     * @param landmark_count 랜드마크 개수
     * @param frame_width 프레임 너비
     * @param frame_height 프레임 높이
     * @param offset_x ROI X 오프셋
     * @param offset_y ROI Y 오프셋
     * @return 주름 영역 마스크 구조체
     */
    static WrinkleRegions createWrinkleRegionMasks(
        const IrisLandmark* face_mesh, int landmark_count,
        int frame_width, int frame_height,
        int offset_x, int offset_y);

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
