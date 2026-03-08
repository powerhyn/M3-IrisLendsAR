/**
 * @file release_gate.cpp
 * @brief 릴리즈 게이트 체크리스트 자동화 구현
 *
 * @author IrisLensSDK Team
 * @version 1.0.0
 * @copyright Apache 2.0 License
 */

#include "iris_sdk/release_gate.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>

namespace iris_sdk {

// ============================================================================
// 내부 유틸리티
// ============================================================================

namespace {

/// @brief NaN/Inf를 안전한 기본값으로 치환
double sanitize(double v, double fallback) noexcept {
    if (std::isnan(v) || std::isinf(v)) return fallback;
    return v;
}

/// @brief 소수점 2자리 문자열 변환
std::string fmt2(double v) noexcept {
    // snprintf로 안전하게 포맷
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.2f", v);
    return buf;
}

/// @brief ReleaseVerdict를 문자열로 변환
const char* verdictToString(ReleaseVerdict v) noexcept {
    switch (v) {
        case ReleaseVerdict::GO:             return "GO";
        case ReleaseVerdict::CONDITIONAL_GO: return "CONDITIONAL_GO";
        case ReleaseVerdict::NO_GO:          return "NO_GO";
    }
    return "UNKNOWN";
}

/// @brief 통과/실패 수 카운트
int countPassed(const std::vector<GateCheckResult>& results) noexcept {
    return static_cast<int>(std::count_if(
        results.begin(), results.end(),
        [](const GateCheckResult& r) { return r.passed; }));
}

} // anonymous namespace

// ============================================================================
// ReleaseGate 구현
// ============================================================================

std::vector<GateCheckResult> ReleaseGate::evaluateHardStop(
    const HardStopInput& input) const noexcept
{
    std::vector<GateCheckResult> results;
    results.reserve(4);

    // 1. Crash-free
    results.push_back({
        "crash_free",
        input.crash_free,
        input.crash_free ? "No crashes" : "Crashes detected"
    });

    // 2. Memory leak
    results.push_back({
        "memory_leak",
        input.no_memory_leak,
        input.no_memory_leak ? "No leaks" : "Memory leaks detected"
    });

    // 3. FPS (frame_time <= 33ms = 30fps)
    // NaN/Inf → 999.0 (fail-safe)
    const double frame_time = sanitize(input.frame_time_ms, 999.0);
    const bool fps_pass = frame_time <= 33.0;
    results.push_back({
        "fps",
        fps_pass,
        fmt2(frame_time) + "ms " +
            (fps_pass ? "<= " : "> ") + "33ms"
    });

    // 4. Temporal CV (< 0.05)
    const double temporal_cv = sanitize(input.temporal_cv, 1.0);
    const bool cv_pass = temporal_cv < 0.05;
    results.push_back({
        "temporal_cv",
        cv_pass,
        fmt2(temporal_cv) + (cv_pass ? " < " : " >= ") + "0.05"
    });

    return results;
}

std::vector<GateCheckResult> ReleaseGate::evaluateQuantitative(
    const QuantitativeInput& input) const noexcept
{
    std::vector<GateCheckResult> results;
    results.reserve(4);

    // NaN/Inf sanitization (fail-safe defaults)
    const double lap_red = sanitize(input.laplacian_reduction, 0.0);
    const double ns_ssim = sanitize(input.non_skin_ssim, 0.0);
    const double fs_time = sanitize(input.freq_sep_time_ms, 999.0);

    // 1. Laplacian reduction (0.3 ~ 0.6)
    const bool lap_pass = lap_red >= 0.30 && lap_red <= 0.60;
    results.push_back({
        "laplacian",
        lap_pass,
        fmt2(lap_red) + (lap_pass ? " in " : " not in ") + "[0.30, 0.60]"
    });

    // 2. Non-skin SSIM (> 0.95)
    const bool ssim_pass = ns_ssim > 0.95;
    results.push_back({
        "non_skin_ssim",
        ssim_pass,
        fmt2(ns_ssim) + (ssim_pass ? " > " : " <= ") + "0.95"
    });

    // 3. FreqSep time (tier-specific)
    const double max_time = getMaxFreqSepTimeMs(input.device_tier);
    const bool time_pass = fs_time <= max_time;
    results.push_back({
        "freq_sep_time",
        time_pass,
        fmt2(fs_time) + "ms " + (time_pass ? "<= " : "> ") + fmt2(max_time) + "ms"
    });

    // 4. TexturePool additional (<= 3)
    const bool tex_pass = input.texture_pool_additional <= 3;
    results.push_back({
        "texture_pool",
        tex_pass,
        std::to_string(input.texture_pool_additional) +
            (tex_pass ? " <= " : " > ") + "3"
    });

    return results;
}

std::vector<GateCheckResult> ReleaseGate::evaluateQualitative(
    const QualitativeInput& input) const noexcept
{
    std::vector<GateCheckResult> results;
    results.reserve(6);

    // 1. Blind A/B preference (> 0.7)
    const bool ab_pass = input.blind_ab_preference > 0.70;
    results.push_back({
        "blind_ab",
        ab_pass,
        fmt2(input.blind_ab_preference) + (ab_pass ? " > " : " <= ") + "0.70"
    });

    // 2. Blurry feedback (< 0.1)
    const bool blurry_pass = input.blurry_feedback_ratio < 0.10;
    results.push_back({
        "blurry",
        blurry_pass,
        fmt2(input.blurry_feedback_ratio) +
            (blurry_pass ? " < " : " >= ") + "0.10"
    });

    // 3. Fake feedback (< 0.1)
    const bool fake_pass = input.fake_feedback_ratio < 0.10;
    results.push_back({
        "fake_looking",
        fake_pass,
        fmt2(input.fake_feedback_ratio) +
            (fake_pass ? " < " : " >= ") + "0.10"
    });

    // 4. Skin tone uniformity
    results.push_back({
        "skin_tone_uniform",
        input.skin_tone_uniform,
        input.skin_tone_uniform ? "Uniform" : "Non-uniform"
    });

    // 5. No halo
    results.push_back({
        "no_halo",
        input.no_halo,
        input.no_halo ? "No halo detected" : "Halo present"
    });

    // 6. No contour blur
    results.push_back({
        "no_contour_blur",
        input.no_contour_blur,
        input.no_contour_blur ? "No contour blurring" : "Contour blurring present"
    });

    return results;
}

ReleaseGateResult ReleaseGate::evaluate(
    const HardStopInput& hard_stop,
    const QuantitativeInput& quantitative,
    const QualitativeInput& qualitative) const noexcept
{
    ReleaseGateResult result{};

    // 각 티어 평가
    result.hard_stop_results    = evaluateHardStop(hard_stop);
    result.quantitative_results = evaluateQuantitative(quantitative);
    result.qualitative_results  = evaluateQualitative(qualitative);

    // 통과 수 집계
    result.hard_stop_pass_count    = countPassed(result.hard_stop_results);
    result.hard_stop_total         = static_cast<int>(result.hard_stop_results.size());
    result.quantitative_pass_count = countPassed(result.quantitative_results);
    result.quantitative_total      = static_cast<int>(result.quantitative_results.size());
    result.qualitative_pass_count  = countPassed(result.qualitative_results);
    result.qualitative_total       = static_cast<int>(result.qualitative_results.size());

    // 판정 로직
    const bool hard_stop_all_pass =
        (result.hard_stop_pass_count == result.hard_stop_total);
    const bool quantitative_all_pass =
        (result.quantitative_pass_count == result.quantitative_total);
    const int qualitative_fail_count =
        result.qualitative_total - result.qualitative_pass_count;

    if (!hard_stop_all_pass || !quantitative_all_pass) {
        result.verdict = ReleaseVerdict::NO_GO;
    } else if (qualitative_fail_count == 0) {
        result.verdict = ReleaseVerdict::GO;
    } else if (qualitative_fail_count <= 1) {
        result.verdict = ReleaseVerdict::CONDITIONAL_GO;
    } else {
        result.verdict = ReleaseVerdict::NO_GO;
    }

    // 요약 생성
    std::ostringstream oss;
    oss << verdictToString(result.verdict)
        << " | Hard-Stop: " << result.hard_stop_pass_count
        << "/" << result.hard_stop_total
        << " | Quantitative: " << result.quantitative_pass_count
        << "/" << result.quantitative_total
        << " | Qualitative: " << result.qualitative_pass_count
        << "/" << result.qualitative_total;
    result.summary = oss.str();

    return result;
}

std::string ReleaseGate::formatReport(const ReleaseGateResult& result) noexcept {
    std::ostringstream oss;

    oss << "=== Release Gate Report ===\n";
    oss << "Verdict: " << verdictToString(result.verdict) << "\n\n";

    // Hard-Stop Gates
    const char* hs_status =
        (result.hard_stop_pass_count == result.hard_stop_total)
            ? "PASS" : "FAIL";
    oss << "[Hard-Stop Gates] "
        << result.hard_stop_pass_count << "/" << result.hard_stop_total
        << " " << hs_status << "\n";
    for (const auto& r : result.hard_stop_results) {
        oss << "  " << (r.passed ? "v " : "x ") << r.gate_name
            << ": " << r.detail << "\n";
    }
    oss << "\n";

    // Quantitative Gates
    const char* qt_status =
        (result.quantitative_pass_count == result.quantitative_total)
            ? "PASS" : "FAIL";
    oss << "[Quantitative Gates] "
        << result.quantitative_pass_count << "/" << result.quantitative_total
        << " " << qt_status << "\n";
    for (const auto& r : result.quantitative_results) {
        oss << "  " << (r.passed ? "v " : "x ") << r.gate_name
            << ": " << r.detail << "\n";
    }
    oss << "\n";

    // Qualitative Gates
    const char* ql_status =
        (result.qualitative_pass_count == result.qualitative_total)
            ? "PASS" : "FAIL";
    oss << "[Qualitative Gates] "
        << result.qualitative_pass_count << "/" << result.qualitative_total
        << " " << ql_status << "\n";
    for (const auto& r : result.qualitative_results) {
        oss << "  " << (r.passed ? "v " : "x ") << r.gate_name
            << ": " << r.detail << "\n";
    }

    return oss.str();
}

double ReleaseGate::getMaxFreqSepTimeMs(DeviceTier tier) noexcept {
    switch (tier) {
        case DeviceTier::HIGH: return 8.0;
        case DeviceTier::MID:  return 12.0;
        // LOW 디바이스는 FreqSep 건너뛰고 Bilateral fallback 사용.
        // 6ms는 Bilateral 경로의 허용 시간이다.
        case DeviceTier::LOW:  return 6.0;
    }
    return 8.0; // 기본값: HIGH 기준
}

} // namespace iris_sdk
