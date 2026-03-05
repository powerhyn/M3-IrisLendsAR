/**
 * @file ab_compare.h
 * @brief Bilateral Filter vs FreqSep 파이프라인 A/B 비교 프레임워크
 *
 * 다양한 피부톤/조명/거리 조건에서 두 스무딩 파이프라인의
 * 품질 메트릭을 비교하고 피부톤별 요약 리포트를 생성한다.
 *
 * @author IrisLensSDK Team
 * @version 1.0.0
 * @copyright Apache 2.0 License
 */

#pragma once

#include <iris_sdk/quality_metrics.h>

#include <opencv2/core.hpp>
#include <cstddef>
#include <string>
#include <vector>

namespace iris_sdk {

// ============================================================================
// 피부톤 그룹
// ============================================================================

/// @brief 피부톤 분류 그룹 (ITA 기반)
enum class SkinToneGroup {
    LIGHT,   ///< 밝은 피부톤 (ITA > 55)
    MEDIUM,  ///< 중간 피부톤 (28 < ITA <= 55)
    DARK     ///< 어두운 피부톤 (ITA <= 28)
};

// ============================================================================
// 테스트 조건
// ============================================================================

/// @brief A/B 비교 테스트 조건 (피부톤, 조명, 거리, 피부 상태)
struct TestCondition {
    SkinToneGroup skin_tone = SkinToneGroup::MEDIUM;
    std::string lighting    = "natural";       ///< "natural", "fluorescent", "backlight"
    std::string distance    = "standard_40cm"; ///< "close_20cm", "standard_40cm", "far_80cm"
    std::string skin_state  = "smooth";        ///< "smooth", "large_pore", "blemished"

    /// @brief 조건 라벨 생성 (예: "LIGHT/natural/standard_40cm/smooth")
    [[nodiscard]] std::string label() const noexcept;
};

// ============================================================================
// 비교 결과
// ============================================================================

/// @brief 단일 A/B 비교 결과
struct ComparisonResult {
    GateResult bilateral_gate;   ///< Bilateral Filter 품질 게이트 결과
    GateResult freq_sep_gate;    ///< FreqSep 품질 게이트 결과

    bool freq_sep_preferred = false; ///< FreqSep이 Bilateral보다 우수한지

    double laplacian_improvement = 0.0; ///< FreqSep Laplacian 개선율
    double ssim_improvement      = 0.0; ///< FreqSep SSIM 개선율
    double halo_improvement      = 0.0; ///< FreqSep Halo 개선율

    SkinToneGroup skin_tone = SkinToneGroup::MEDIUM;
    std::string condition_label; ///< 예: "LIGHT/natural_light/standard"
};

// ============================================================================
// ABCompare 클래스
// ============================================================================

/**
 * @brief Bilateral Filter vs FreqSep A/B 비교 프레임워크
 *
 * 사용법:
 * @code
 *   ABCompare ab;
 *   for (const auto& cond : conditions) {
 *       auto result = ab.compare(original, bilateral, freqsep, mask, cond);
 *       ab.addResult(result);
 *   }
 *   auto summaries = ab.summarizeBySkinTone();
 *   std::string report = ab.generateReport();
 * @endcode
 */
class ABCompare {
public:
    ABCompare() noexcept = default;

    /// @brief 피부톤별 요약 결과
    struct SkinToneSummary {
        SkinToneGroup group = SkinToneGroup::MEDIUM;
        int total_tests              = 0;
        int freq_sep_preferred_count = 0;
        double avg_laplacian_improvement = 0.0;
        double avg_ssim_improvement      = 0.0;
        double avg_halo_improvement      = 0.0;
        double preference_ratio          = 0.0; ///< freq_sep_preferred / total (>0.7 = pass)
        bool passes_gate                 = false; ///< preference_ratio > 0.7
    };

    /**
     * @brief 단일 조건 A/B 비교 실행
     *
     * 원본 이미지에 대해 Bilateral/FreqSep 결과를 QualityMetrics로 평가하고
     * 두 결과를 비교하여 ComparisonResult를 반환한다.
     *
     * @param original         원본 이미지 (CV_8UC3 BGR)
     * @param bilateral_result Bilateral Filter 처리 결과 (CV_8UC3 BGR)
     * @param freq_sep_result  FreqSep 처리 결과 (CV_8UC3 BGR)
     * @param skin_mask        피부 마스크 (CV_8UC1, 255=skin)
     * @param condition        테스트 조건
     * @return ComparisonResult 비교 결과
     */
    ComparisonResult compare(
        const cv::Mat& original,
        const cv::Mat& bilateral_result,
        const cv::Mat& freq_sep_result,
        const cv::Mat& skin_mask,
        const TestCondition& condition) noexcept;

    /// @brief 비교 결과 추가 (배치 분석용)
    void addResult(const ComparisonResult& result) noexcept;

    /// @brief 전체 결과 피부톤별 요약 생성
    [[nodiscard]] std::vector<SkinToneSummary> summarizeBySkinTone() const noexcept;

    /// @brief JSON 형태의 리포트 문자열 생성
    [[nodiscard]] std::string generateReport() const noexcept;

    /// @brief 결과 초기화
    void reset() noexcept;

    /// @brief 총 비교 수
    [[nodiscard]] std::size_t resultCount() const noexcept;

private:
    std::vector<ComparisonResult> results_;

    /// @brief GateVerdict를 정수 점수로 변환 (GO=2, CONDITIONAL=1, NO_GO=0)
    static int verdictScore(GateVerdict v) noexcept;

    /// @brief SkinToneGroup을 문자열로 변환
    static const char* skinToneToString(SkinToneGroup group) noexcept;

    /// @brief FreqSep이 Bilateral보다 우수한지 판정
    static bool isFreqSepPreferred(
        const GateResult& bilateral,
        const GateResult& freq_sep) noexcept;
};

} // namespace iris_sdk
