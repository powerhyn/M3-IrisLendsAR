/**
 * @file ab_compare.cpp
 * @brief Bilateral Filter vs FreqSep A/B 비교 프레임워크 구현
 *
 * @note 예외 정책: noexcept 계약을 유지하며, 내부 예외는 catch(...)로 포착 후
 *       fprintf(stderr)로 최소 로그를 남기고 기본값을 반환한다.
 *       SDK 내부 로거가 없으므로 stderr 사용.
 */

#include <iris_sdk/ab_compare.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <sstream>

namespace iris_sdk {

namespace {

std::string escapeJson(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20)
                    out += ' ';
                else
                    out += c;
        }
    }
    return out;
}

} // namespace

// ============================================================================
// TestCondition
// ============================================================================

std::string TestCondition::label() const noexcept {
    return std::string(ABCompare::skinToneToString(skin_tone)) +
        "/" + lighting + "/" + distance + "/" + skin_state;
}

// ============================================================================
// ABCompare - 유틸리티
// ============================================================================

int ABCompare::verdictScore(const GateVerdict v) noexcept {
    switch (v) {
        case GateVerdict::GO:             return 2;
        case GateVerdict::CONDITIONAL_GO: return 1;
        case GateVerdict::NO_GO:          return 0;
    }
    return 0;
}

const char* ABCompare::skinToneToString(const SkinToneGroup group) noexcept {
    switch (group) {
        case SkinToneGroup::LIGHT:  return "LIGHT";
        case SkinToneGroup::MEDIUM: return "MEDIUM";
        case SkinToneGroup::DARK:   return "DARK";
    }
    return "UNKNOWN";
}

bool ABCompare::isFreqSepPreferred(
    const GateResult& bilateral,
    const GateResult& freq_sep) noexcept
{
    const int b_score = verdictScore(bilateral.verdict);
    const int f_score = verdictScore(freq_sep.verdict);

    // FreqSep verdict가 더 좋으면 우선
    if (f_score > b_score) {
        return true;
    }

    // 동일 verdict일 때 개별 메트릭 비교
    if (f_score == b_score) {
        int advantages = 0;

        // Laplacian: 감소율이 30~60% 범위에 가까울수록 좋음
        // 두 결과 모두 pass이면 reduction_ratio가 0.45(중간값)에 가까운 쪽이 유리
        const double b_lap_dist = std::abs(bilateral.laplacian.reduction_ratio - 0.45);
        const double f_lap_dist = std::abs(freq_sep.laplacian.reduction_ratio - 0.45);
        if (f_lap_dist < b_lap_dist) {
            ++advantages;
        }

        // SSIM: 높을수록 좋음
        if (freq_sep.ssim.ssim_value > bilateral.ssim.ssim_value) {
            ++advantages;
        }

        // Halo: gradient 증가율이 낮을수록 좋음
        if (freq_sep.halo.gradient_increase_ratio < bilateral.halo.gradient_increase_ratio) {
            ++advantages;
        }

        // 3개 메트릭 중 2개 이상에서 우위면 선호
        return advantages >= 2;
    }

    return false;
}

// ============================================================================
// ABCompare - 핵심 메서드
// ============================================================================

ComparisonResult ABCompare::compare(
    const cv::Mat& original,
    const cv::Mat& bilateral_result,
    const cv::Mat& freq_sep_result,
    const cv::Mat& skin_mask,
    const TestCondition& condition) noexcept
{
    try {
        ComparisonResult result;
        result.skin_tone       = condition.skin_tone;
        result.condition_label = condition.label();

        // 입력 유효성 검사
        if (original.empty() || bilateral_result.empty() ||
            freq_sep_result.empty() || skin_mask.empty()) {
            return result;
        }

        // 양쪽 파이프라인에 대해 품질 게이트 평가
        result.bilateral_gate = QualityMetrics::evaluateQuantitativeGate(
            original, bilateral_result, skin_mask);

        result.freq_sep_gate = QualityMetrics::evaluateQuantitativeGate(
            original, freq_sep_result, skin_mask);

        // 개선율 계산 (FreqSep - Bilateral, 양수이면 FreqSep이 더 나음)
        // Laplacian: 감소율이 이상적 범위(0.45)에 얼마나 가까운지
        const double b_lap = result.bilateral_gate.laplacian.reduction_ratio;
        const double f_lap = result.freq_sep_gate.laplacian.reduction_ratio;
        const double b_lap_err = std::abs(b_lap - 0.45);
        const double f_lap_err = std::abs(f_lap - 0.45);
        result.laplacian_improvement = (b_lap_err > 1e-9)
            ? (b_lap_err - f_lap_err) / b_lap_err
            : 0.0;

        // SSIM: 높을수록 좋음 (상대 개선율)
        const double b_ssim = result.bilateral_gate.ssim.ssim_value;
        const double f_ssim = result.freq_sep_gate.ssim.ssim_value;
        result.ssim_improvement = (b_ssim > 1e-9)
            ? (f_ssim - b_ssim) / b_ssim
            : 0.0;

        // Halo: gradient 증가율이 낮을수록 좋음 (감소가 개선)
        // b_halo ≤ 0 이면 Bilateral이 이미 halo-free → 직접 차이(delta)로 비교
        const double b_halo = result.bilateral_gate.halo.gradient_increase_ratio;
        const double f_halo = result.freq_sep_gate.halo.gradient_increase_ratio;
        if (b_halo > 1e-9) {
            result.halo_improvement = (b_halo - f_halo) / b_halo;
        } else {
            // Bilateral halo ≈ 0: 절대 차이로 비교 (FreqSep도 halo-free면 0)
            result.halo_improvement = b_halo - f_halo;
        }

        // FreqSep 선호 여부 판정
        result.freq_sep_preferred = isFreqSepPreferred(
            result.bilateral_gate, result.freq_sep_gate);

        return result;
    } catch (...) {
        std::fprintf(stderr, "[IrisSDK] ABCompare::compare: unknown exception\n");
        return {};
    }
}

void ABCompare::addResult(const ComparisonResult& result) noexcept {
    results_.push_back(result);
    if (results_.size() > kMaxResults) {
        results_.erase(results_.begin());
    }
}

void ABCompare::reset() noexcept {
    results_.clear();
}

std::size_t ABCompare::resultCount() const noexcept {
    return results_.size();
}

// ============================================================================
// ABCompare - 요약 및 리포트
// ============================================================================

std::vector<ABCompare::SkinToneSummary> ABCompare::summarizeBySkinTone() const noexcept {
    // 3개 피부톤 그룹에 대한 요약 생성
    constexpr SkinToneGroup groups[] = {
        SkinToneGroup::LIGHT,
        SkinToneGroup::MEDIUM,
        SkinToneGroup::DARK
    };

    std::vector<SkinToneSummary> summaries;
    summaries.reserve(3);

    for (const auto group : groups) {
        SkinToneSummary summary;
        summary.group = group;

        double sum_lap  = 0.0;
        double sum_ssim = 0.0;
        double sum_halo = 0.0;

        for (const auto& r : results_) {
            if (r.skin_tone != group) {
                continue;
            }
            ++summary.total_tests;
            if (r.freq_sep_preferred) {
                ++summary.freq_sep_preferred_count;
            }
            sum_lap  += r.laplacian_improvement;
            sum_ssim += r.ssim_improvement;
            sum_halo += r.halo_improvement;
        }

        if (summary.total_tests > 0) {
            const auto n = static_cast<double>(summary.total_tests);
            summary.avg_laplacian_improvement = sum_lap / n;
            summary.avg_ssim_improvement      = sum_ssim / n;
            summary.avg_halo_improvement      = sum_halo / n;
            summary.preference_ratio =
                static_cast<double>(summary.freq_sep_preferred_count) / n;
            summary.passes_gate = (summary.preference_ratio > 0.7);
        }

        summaries.push_back(summary);
    }

    return summaries;
}

std::string ABCompare::generateReport() const noexcept {
    const auto summaries = summarizeBySkinTone();

    std::ostringstream os;
    os.precision(4);
    os << std::fixed;

    os << "{\n";
    os << "  \"total_comparisons\": " << results_.size() << ",\n";

    // 피부톤별 요약
    os << "  \"skin_tone_summaries\": [\n";
    for (std::size_t i = 0; i < summaries.size(); ++i) {
        const auto& s = summaries[i];
        os << "    {\n";
        os << "      \"group\": \"" << skinToneToString(s.group) << "\",\n";
        os << "      \"total_tests\": " << s.total_tests << ",\n";
        os << "      \"freq_sep_preferred_count\": " << s.freq_sep_preferred_count << ",\n";
        os << "      \"avg_laplacian_improvement\": " << s.avg_laplacian_improvement << ",\n";
        os << "      \"avg_ssim_improvement\": " << s.avg_ssim_improvement << ",\n";
        os << "      \"avg_halo_improvement\": " << s.avg_halo_improvement << ",\n";
        os << "      \"preference_ratio\": " << s.preference_ratio << ",\n";
        os << "      \"passes_gate\": " << (s.passes_gate ? "true" : "false") << "\n";
        os << "    }";
        if (i + 1 < summaries.size()) {
            os << ",";
        }
        os << "\n";
    }
    os << "  ],\n";

    // 개별 결과
    os << "  \"results\": [\n";
    for (std::size_t i = 0; i < results_.size(); ++i) {
        const auto& r = results_[i];
        os << "    {\n";
        os << "      \"condition\": \"" << escapeJson(r.condition_label) << "\",\n";
        os << "      \"skin_tone\": \"" << skinToneToString(r.skin_tone) << "\",\n";
        os << "      \"freq_sep_preferred\": " << (r.freq_sep_preferred ? "true" : "false") << ",\n";
        os << "      \"laplacian_improvement\": " << r.laplacian_improvement << ",\n";
        os << "      \"ssim_improvement\": " << r.ssim_improvement << ",\n";
        os << "      \"halo_improvement\": " << r.halo_improvement << ",\n";

        os << "      \"bilateral_verdict\": \"" << gateVerdictToString(r.bilateral_gate.verdict) << "\",\n";
        os << "      \"freq_sep_verdict\": \"" << gateVerdictToString(r.freq_sep_gate.verdict) << "\"\n";

        os << "    }";
        if (i + 1 < results_.size()) {
            os << ",";
        }
        os << "\n";
    }
    os << "  ]\n";
    os << "}";

    return os.str();
}

} // namespace iris_sdk
