/**
 * @file param_tuner.cpp
 * @brief FreqSep attenuation curve parameter tuning implementation
 */

#include "iris_sdk/param_tuner.h"

#include <algorithm>
#include <cstdio>
#include <sstream>

namespace iris_sdk {

// ============================================================================
// FreqSepTuningParams
// ============================================================================

std::string FreqSepTuningParams::toString() const noexcept {
    // 고정 크기 버퍼로 snprintf 사용 (예외 없음)
    char buf[128];
    std::snprintf(buf, sizeof(buf),
        "aLow=%.2f,aHigh=%.2f,sigma=%.2f,preserve=%.2f,radius=%d",
        attenuation_low, attenuation_high, sigma_ratio,
        high_freq_preserve, blur_radius);
    return std::string(buf);
}

// ============================================================================
// ParamTuner - Grid Search
// ============================================================================

std::vector<TuningResult> ParamTuner::gridSearch(
    const cv::Mat& original,
    const cv::Mat& skin_mask,
    const TuningRange& range,
    ProcessCallback process_fn,
    const std::string& condition_label) noexcept
{
    try {
        std::vector<TuningResult> results;

        // 입력 유효성 검사
        if (original.empty() || skin_mask.empty() || !process_fn) {
            return results;
        }

        // 각 파라미터에 대한 그리드 값 생성
        const auto a_low_steps  = generateSteps(
            range.attenuation_low_min, range.attenuation_low_max, range.steps);
        const auto a_high_steps = generateSteps(
            range.attenuation_high_min, range.attenuation_high_max, range.steps);
        const auto sigma_steps  = generateSteps(
            range.sigma_ratio_min, range.sigma_ratio_max, range.steps);
        const auto preserve_steps = generateSteps(
            range.high_freq_preserve_min, range.high_freq_preserve_max, range.steps);

        // 총 조합 수 예약
        const auto total = a_low_steps.size() * a_high_steps.size()
                         * sigma_steps.size() * preserve_steps.size();
        results.reserve(total);

        // 모든 조합을 순회
        for (const float a_low : a_low_steps) {
            for (const float a_high : a_high_steps) {
                for (const float sigma : sigma_steps) {
                    for (const float preserve : preserve_steps) {
                        FreqSepTuningParams params;
                        params.attenuation_low   = a_low;
                        params.attenuation_high  = a_high;
                        params.sigma_ratio       = sigma;
                        params.high_freq_preserve = preserve;

                        // 콜백을 통해 처리된 이미지 획득
                        cv::Mat processed = process_fn(original, params);
                        if (processed.empty()) {
                            continue;
                        }

                        // QualityMetrics로 평가
                        GateResult gate = QualityMetrics::evaluateQuantitativeGate(
                            original, processed, skin_mask);

                        // 가중 점수 계산
                        const double score = computeScore(gate);

                        TuningResult result;
                        result.params          = params;
                        result.gate_result     = std::move(gate);
                        result.overall_score   = score;
                        result.condition_label = condition_label;

                        results.push_back(std::move(result));
                    }
                }
            }
        }

        return results;
    } catch (...) {
        return {};
    }
}

// ============================================================================
// ParamTuner - Score Computation
// ============================================================================

double ParamTuner::computeScore(const GateResult& gate) noexcept {
    double score = 0.0;

    // Laplacian 점수 (가중치 0.4)
    // 감소율이 45% 중심에 가까울수록 고점
    if (gate.laplacian.passes_gate) {
        const double deviation = std::abs(gate.laplacian.reduction_ratio - 0.45);
        const double laplacian_score = 1.0 - (deviation / 0.15);
        score += 0.4 * std::max(0.0, std::min(1.0, laplacian_score));
    }

    // SSIM 점수 (가중치 0.3)
    // SSIM 값이 높을수록 고점
    if (gate.ssim.passes_gate) {
        score += 0.3 * std::max(0.0, std::min(1.0, gate.ssim.ssim_value));
    }

    // Halo 점수 (가중치 0.3)
    // gradient 증가가 적을수록 고점
    if (gate.halo.passes_gate) {
        const double halo_score = 1.0 - gate.halo.gradient_increase_ratio;
        score += 0.3 * std::max(0.0, std::min(1.0, halo_score));
    }

    // 최종 clamp [0, 1]
    return std::max(0.0, std::min(1.0, score));
}

// ============================================================================
// ParamTuner - Find Best
// ============================================================================

FreqSepTuningParams ParamTuner::findBest(
    const std::vector<TuningResult>& results) noexcept
{
    if (results.empty()) {
        return FreqSepTuningParams{};
    }

    const auto it = std::max_element(results.begin(), results.end(),
        [](const TuningResult& a, const TuningResult& b) {
            return a.overall_score < b.overall_score;
        });

    return it->params;
}

// ============================================================================
// ParamTuner - Preset Recommendation
// ============================================================================

PresetRecommendation ParamTuner::recommendPresets(
    const std::vector<TuningResult>& results) noexcept
{
    PresetRecommendation presets;

    // 결과를 high_freq_preserve 기준으로 3개 그룹으로 분류
    std::vector<const TuningResult*> natural_candidates;
    std::vector<const TuningResult*> moderate_candidates;
    std::vector<const TuningResult*> strong_candidates;

    for (const auto& r : results) {
        const float preserve = r.params.high_freq_preserve;
        if (preserve > 0.6f) {
            natural_candidates.push_back(&r);
        } else if (preserve >= 0.3f) {
            moderate_candidates.push_back(&r);
        } else {
            strong_candidates.push_back(&r);
        }
    }

    // 각 그룹에서 최고 점수를 선택하는 람다
    auto pick_best = [](const std::vector<const TuningResult*>& candidates)
        -> FreqSepTuningParams {
        if (candidates.empty()) {
            return FreqSepTuningParams{};
        }
        const auto it = std::max_element(candidates.begin(), candidates.end(),
            [](const TuningResult* a, const TuningResult* b) {
                return a->overall_score < b->overall_score;
            });
        return (*it)->params;
    };

    presets.natural_params  = pick_best(natural_candidates);
    presets.moderate_params = pick_best(moderate_candidates);
    presets.strong_params   = pick_best(strong_candidates);

    // 기본 skinQuality 값 유지
    presets.natural_value  = 0.3f;
    presets.moderate_value = 0.5f;
    presets.strong_value   = 0.8f;

    return presets;
}

// ============================================================================
// ParamTuner - Blur Radius Validation
// ============================================================================

ParamTuner::BlurRadiusValidation ParamTuner::validateBlurRadiusIndependence(
    int face_width) noexcept
{
    BlurRadiusValidation result;

    // B-method: face_width * 0.05, clamp(6, 28)
    // skinQuality 값과 무관하게 동일한 공식을 사용
    result.radius_at_low  = computeBlurRadius(face_width);
    result.radius_at_mid  = computeBlurRadius(face_width);
    result.radius_at_high = computeBlurRadius(face_width);

    result.independent = (result.radius_at_low == result.radius_at_mid)
                      && (result.radius_at_mid == result.radius_at_high);

    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "face_width=%d, radius=%d (formula: clamp(face_width*0.05, 6, 28)), "
        "low=%.1f→%d, mid=%.1f→%d, high=%.1f→%d, independent=%s",
        face_width, result.radius_at_low,
        0.3f, result.radius_at_low,
        0.5f, result.radius_at_mid,
        0.8f, result.radius_at_high,
        result.independent ? "true" : "false");
    result.detail = std::string(buf);

    return result;
}

// ============================================================================
// ParamTuner - Report Generation
// ============================================================================

std::string ParamTuner::generateReport(
    const std::vector<TuningResult>& results,
    const PresetRecommendation& presets) noexcept
{
    std::ostringstream oss;

    oss << "=== FreqSep Parameter Tuning Report ===\n\n";

    // 결과 요약
    oss << "Total combinations evaluated: " << results.size() << "\n\n";

    if (results.empty()) {
        oss << "(No results available)\n";
        return oss.str();
    }

    // 상위 5개 결과
    auto sorted = results;
    std::sort(sorted.begin(), sorted.end(),
        [](const TuningResult& a, const TuningResult& b) {
            return a.overall_score > b.overall_score;
        });

    const auto top_count = std::min<std::size_t>(5, sorted.size());
    oss << "--- Top " << top_count << " Results ---\n";
    for (std::size_t i = 0; i < top_count; ++i) {
        const auto& r = sorted[i];
        char line[256];
        std::snprintf(line, sizeof(line),
            "#%zu  score=%.4f  verdict=%s  %s\n",
            i + 1,
            r.overall_score,
            r.gate_result.verdict == GateVerdict::GO ? "GO" :
            r.gate_result.verdict == GateVerdict::CONDITIONAL_GO ? "CONDITIONAL" : "NO_GO",
            r.params.toString().c_str());
        oss << line;
    }

    // 통계
    double sum = 0.0;
    double max_score = 0.0;
    double min_score = 1.0;
    int go_count = 0;
    int cond_count = 0;
    int nogo_count = 0;

    for (const auto& r : results) {
        sum += r.overall_score;
        max_score = std::max(max_score, r.overall_score);
        min_score = std::min(min_score, r.overall_score);

        switch (r.gate_result.verdict) {
            case GateVerdict::GO:             ++go_count; break;
            case GateVerdict::CONDITIONAL_GO: ++cond_count; break;
            case GateVerdict::NO_GO:          ++nogo_count; break;
        }
    }

    const double avg_score = results.empty() ? 0.0 : sum / static_cast<double>(results.size());

    oss << "\n--- Statistics ---\n";
    {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "Score: avg=%.4f, min=%.4f, max=%.4f\n",
            avg_score, min_score, max_score);
        oss << buf;
    }
    {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "Verdicts: GO=%d, CONDITIONAL=%d, NO_GO=%d\n",
            go_count, cond_count, nogo_count);
        oss << buf;
    }

    // 프리셋 추천
    oss << "\n--- Preset Recommendations ---\n";
    {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "NATURAL   (skinQuality=%.1f): %s\n",
            presets.natural_value,
            presets.natural_params.toString().c_str());
        oss << buf;
    }
    {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "MODERATE  (skinQuality=%.1f): %s\n",
            presets.moderate_value,
            presets.moderate_params.toString().c_str());
        oss << buf;
    }
    {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "STRONG    (skinQuality=%.1f): %s\n",
            presets.strong_value,
            presets.strong_params.toString().c_str());
        oss << buf;
    }

    oss << "\n=== End of Report ===\n";

    return oss.str();
}

// ============================================================================
// ParamTuner - Private Helpers
// ============================================================================

std::vector<float> ParamTuner::generateSteps(
    float min_val, float max_val, int steps) noexcept
{
    std::vector<float> values;

    if (steps <= 0) {
        return values;
    }

    if (steps == 1) {
        values.push_back((min_val + max_val) * 0.5f);
        return values;
    }

    values.reserve(static_cast<std::size_t>(steps));
    const float step_size = (max_val - min_val) / static_cast<float>(steps - 1);

    for (int i = 0; i < steps; ++i) {
        values.push_back(min_val + step_size * static_cast<float>(i));
    }

    return values;
}

int ParamTuner::computeBlurRadius(int face_width) noexcept {
    const int raw = static_cast<int>(std::round(
        static_cast<float>(face_width) * 0.05f));
    return std::clamp(raw, 6, 28);
}

} // namespace iris_sdk
