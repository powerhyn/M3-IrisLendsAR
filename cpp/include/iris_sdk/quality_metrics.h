/**
 * @file quality_metrics.h
 * @brief FreqSep 뷰티 파이프라인 품질 측정 유틸리티
 *
 * 피부 스무딩 품질을 정량적으로 측정하는 메트릭 클래스.
 * Laplacian variance, SSIM, temporal consistency, halo detection 제공.
 *
 * @author IrisLensSDK Team
 * @version 1.0.0
 * @copyright Apache 2.0 License
 */

#pragma once

#include <opencv2/core.hpp>
#include <deque>
#include <string>
#include <vector>

namespace iris_sdk {

// ============================================================================
// 결과 구조체
// ============================================================================

/**
 * @brief Laplacian variance 감소율 측정 결과
 *
 * 피부 영역의 Laplacian variance를 비교하여
 * 스무딩 효과의 적절성을 판단한다.
 */
struct LaplacianResult {
    double original_variance  = 0.0;  ///< 원본 Laplacian variance
    double processed_variance = 0.0;  ///< 처리 후 Laplacian variance
    double reduction_ratio    = 0.0;  ///< 감소율 (0~1, 1이면 완전 제거)
    bool   passes_gate        = false; ///< 30~60% 감소면 pass
};

/**
 * @brief SSIM (Structural Similarity) 측정 결과
 *
 * 비-피부 영역의 구조적 유사도를 측정하여
 * 원본 보존 정도를 판단한다.
 */
struct SSIMResult {
    double ssim_value  = 0.0;   ///< SSIM 값 (0~1, 1이면 동일)
    bool   passes_gate = false; ///< > 0.95면 pass
};

/**
 * @brief 프레임 간 시간적 일관성 측정 결과
 *
 * 연속 프레임의 스무딩 강도 변동을 분석하여
 * 깜빡임(flickering) 없이 안정적인지 판단한다.
 */
struct TemporalResult {
    double mean_strength            = 0.0; ///< 평균 스무딩 강도
    double std_deviation            = 0.0; ///< 표준편차
    double coefficient_of_variation = 0.0; ///< CV = std/mean
    bool   passes_gate              = false; ///< CV < 5%면 pass
};

/**
 * @brief Halo artifact 검출 결과
 *
 * 피부/비-피부 경계 영역의 gradient 변화를 측정하여
 * 비정상적 halo(번짐) 발생 여부를 판단한다.
 */
struct HaloResult {
    double original_boundary_gradient  = 0.0; ///< 원본 경계 gradient magnitude
    double processed_boundary_gradient = 0.0; ///< 처리 후 경계 gradient magnitude
    double gradient_increase_ratio     = 0.0; ///< gradient 증가율
    bool   passes_gate                 = false; ///< 증가 < 15%면 pass
};

/**
 * @brief 종합 게이트 판정
 */
enum class GateVerdict {
    GO,              ///< 모든 메트릭 통과
    CONDITIONAL_GO,  ///< 일부 미통과, 허용 범위
    NO_GO            ///< 핵심 메트릭 미통과
};

/// @brief GateVerdict를 문자열로 변환
inline const char* gateVerdictToString(GateVerdict v) noexcept {
    switch (v) {
        case GateVerdict::GO:             return "GO";
        case GateVerdict::CONDITIONAL_GO: return "CONDITIONAL_GO";
        case GateVerdict::NO_GO:          return "NO_GO";
    }
    return "UNKNOWN";
}

/**
 * @brief 종합 게이트 결과
 */
struct GateResult {
    LaplacianResult laplacian;
    SSIMResult      ssim;
    HaloResult      halo;
    GateVerdict     verdict = GateVerdict::NO_GO;
    std::string     summary;
};

// ============================================================================
// QualityMetrics - 정적 메트릭 측정
// ============================================================================

/**
 * @brief FreqSep 파이프라인 품질 측정 유틸리티
 *
 * 모든 정적 메서드는 BGR CV_8UC3 이미지와
 * CV_8UC1 skin_mask (255=skin, 0=non-skin)를 입력받는다.
 *
 * @note 예외를 사용하지 않는다. 오류 시 기본값을 반환한다.
 */
class QualityMetrics {
public:
    QualityMetrics() = delete;

    /**
     * @brief Laplacian variance 감소율 측정
     *
     * 피부 영역만 추출하여 Laplacian 분산을 비교한다.
     * 적절한 스무딩은 30~60% 감소를 보인다.
     *
     * @param original   원본 이미지 (CV_8UC3 BGR)
     * @param processed  처리된 이미지 (CV_8UC3 BGR)
     * @param skin_mask  피부 마스크 (CV_8UC1, 255=skin)
     * @return LaplacianResult 측정 결과
     */
    static LaplacianResult measureLaplacianReduction(
        const cv::Mat& original,
        const cv::Mat& processed,
        const cv::Mat& skin_mask) noexcept;

    /**
     * @brief 비-피부 영역 SSIM 측정
     *
     * Wang et al. (2004) 공식으로 비-피부 영역의
     * 구조적 유사도를 측정한다.
     *
     * @param original   원본 이미지 (CV_8UC3 BGR)
     * @param processed  처리된 이미지 (CV_8UC3 BGR)
     * @param skin_mask  피부 마스크 (CV_8UC1, 255=skin)
     * @return SSIMResult 측정 결과
     */
    static SSIMResult measureNonSkinSSIM(
        const cv::Mat& original,
        const cv::Mat& processed,
        const cv::Mat& skin_mask) noexcept;

    /**
     * @brief Halo artifact 검출
     *
     * 피부/비-피부 경계를 추출하고 Sobel gradient를 측정하여
     * 처리 전후의 gradient 변화를 비교한다.
     *
     * @param original       원본 이미지 (CV_8UC3 BGR)
     * @param processed      처리된 이미지 (CV_8UC3 BGR)
     * @param skin_mask      피부 마스크 (CV_8UC1, 255=skin)
     * @param boundary_width 경계 영역 폭 (픽셀, 기본 5)
     * @return HaloResult 측정 결과
     */
    static HaloResult detectHalo(
        const cv::Mat& original,
        const cv::Mat& processed,
        const cv::Mat& skin_mask,
        int boundary_width = 5) noexcept;

    /**
     * @brief 종합 게이트 판정
     *
     * Laplacian, SSIM, Halo를 모두 측정하여 종합 판정한다.
     * - GO: 모든 메트릭 통과
     * - CONDITIONAL_GO: 1개 미통과
     * - NO_GO: 2개 이상 미통과
     *
     * @param original   원본 이미지 (CV_8UC3 BGR)
     * @param processed  처리된 이미지 (CV_8UC3 BGR)
     * @param skin_mask  피부 마스크 (CV_8UC1, 255=skin)
     * @return GateResult 종합 결과
     */
    static GateResult evaluateQuantitativeGate(
        const cv::Mat& original,
        const cv::Mat& processed,
        const cv::Mat& skin_mask) noexcept;

private:
    /// @brief 그레이스케일 변환 (이미 1채널이면 복사 없이 반환)
    static cv::Mat toGray(const cv::Mat& src) noexcept;

    /// @brief SSIM 계산 (단일 채널, 마스크 적용)
    static double computeSSIMChannel(
        const cv::Mat& img1_gray,
        const cv::Mat& img2_gray,
        const cv::Mat& mask) noexcept;

    /// @brief 마스크 영역의 Laplacian variance 계산
    static double computeMaskedLaplacianVariance(
        const cv::Mat& gray,
        const cv::Mat& mask) noexcept;

    /// @brief 경계 마스크 생성 (dilation XOR original)
    static cv::Mat createBoundaryMask(
        const cv::Mat& skin_mask,
        int boundary_width) noexcept;

    /// @brief 마스크 영역 Sobel gradient magnitude 평균
    static double computeMaskedGradientMagnitude(
        const cv::Mat& gray,
        const cv::Mat& mask) noexcept;

    /// @brief 입력 유효성 검증
    static bool validateInputs(
        const cv::Mat& img1,
        const cv::Mat& img2,
        const cv::Mat& mask) noexcept;

    /// @brief Laplacian reduction 측정 (gray Mat 직접 입력)
    static double measureLaplacianReductionImpl(
        const cv::Mat& originalGray,
        const cv::Mat& processedGray,
        const cv::Mat& mask) noexcept;

    /// @brief Halo 검출 (gray Mat 직접 입력)
    static double detectHaloImpl(
        const cv::Mat& originalGray,
        const cv::Mat& processedGray,
        const cv::Mat& mask) noexcept;
};

// ============================================================================
// TemporalAnalyzer - 시간적 일관성 분석
// ============================================================================

/**
 * @brief 프레임 간 스무딩 강도의 시간적 일관성 분석기
 *
 * 연속 프레임의 Laplacian reduction ratio를 수집하여
 * 변동 계수(CV)를 산출한다. CV < 5%이면 안정적.
 *
 * @note This class is NOT thread-safe. External synchronization is required
 *       if accessed from multiple threads. All methods including addFrame()
 *       modify internal state (reduction_ratios_, frame_count_).
 *
 * 사용법:
 * @code
 *   TemporalAnalyzer analyzer;
 *   for (auto& frame : frames) {
 *       analyzer.addFrame(original, processed, mask);
 *   }
 *   auto result = analyzer.computeTemporalVariance();
 * @endcode
 */
class TemporalAnalyzer {
public:
    TemporalAnalyzer() noexcept = default;

    /**
     * @brief 프레임 쌍 추가
     *
     * 원본/처리 프레임의 Laplacian reduction ratio를 기록한다.
     * 매 프레임 Laplacian 연산을 수행하므로 (~2-5ms/frame),
     * 호출자가 이미 ratio를 알고 있다면 addReductionRatio()를 사용한다.
     *
     * @param original   원본 프레임 (CV_8UC3 BGR)
     * @param processed  처리된 프레임 (CV_8UC3 BGR)
     * @param skin_mask  피부 마스크 (CV_8UC1, 255=skin)
     */
    void addFrame(
        const cv::Mat& original,
        const cv::Mat& processed,
        const cv::Mat& skin_mask) noexcept;

    /**
     * @brief 미리 계산된 reduction ratio를 직접 추가
     *
     * 호출자가 이미 Laplacian reduction ratio를 알고 있을 때
     * 불필요한 재계산 없이 직접 기록한다.
     * evaluateQuantitativeGate() 등에서 이미 ratio를 얻은 경우 유용.
     *
     * @param ratio Laplacian reduction ratio (0~1 범위 외 값은 무시)
     */
    void addReductionRatio(double ratio) noexcept;

    /**
     * @brief 시간적 변동 분석 결과 산출
     *
     * 수집된 프레임 데이터로부터 평균, 표준편차, CV를 계산한다.
     * 프레임 수가 2 미만이면 기본값(미통과)을 반환한다.
     *
     * @return TemporalResult 분석 결과
     */
    [[nodiscard]] TemporalResult computeTemporalVariance() const noexcept;

    /**
     * @brief 수집 데이터 초기화
     */
    void resetTemporalData() noexcept;

    /**
     * @brief 수집된 프레임 수
     */
    [[nodiscard]] std::size_t frameCount() const noexcept;

private:
    std::deque<double> reduction_ratios_; ///< 프레임별 Laplacian reduction ratio (deque: O(1) pop_front)
};

} // namespace iris_sdk
