/**
 * @file quality_metrics.cpp
 * @brief FreqSep 뷰티 파이프라인 품질 측정 구현
 */

#include "iris_sdk/quality_metrics.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <sstream>

namespace iris_sdk {

// ============================================================================
// 내부 상수
// ============================================================================

namespace {

/// Laplacian reduction 통과 범위
/// 근거: 30% 미만은 스무딩 불충분 (텍스처 잔류), 60% 초과는 과도한 블러 (디테일 손실).
/// 모바일 뷰티 앱 벤치마크 (FaceApp, Snow 등) 참조 기반 경험적 범위.
constexpr double kLaplacianMinReduction = 0.30;
constexpr double kLaplacianMaxReduction = 0.60;

/// SSIM 통과 임계값
/// 근거: Wang et al. (2004) 기준 0.95 이상이면 "거의 동일" 수준.
/// 비-피부 영역 보존이 목적이므로 엄격한 임계값 적용.
constexpr double kSSIMPassThreshold = 0.95;

/// Halo gradient 증가 허용 임계값
/// 근거: 15% 이상 gradient 증가는 육안 식별 가능한 halo artifact.
/// Bilateral filter 아티팩트 문헌 참조 (Paris & Durand, 2006).
constexpr double kHaloMaxIncreaseRatio = 0.15;

/// Temporal CV 통과 임계값
/// 근거: CV 5% 이하는 30fps 영상에서 프레임 간 깜빡임이 인지 불가.
/// 비디오 품질 평가 표준 (ITU-T P.910) 기반 경험적 값.
constexpr double kTemporalCVThreshold = 0.05;

/// SSIM 안정화 상수 (Wang et al. 2004)
/// C1 = (K1 * L)^2, C2 = (K2 * L)^2  where L=255, K1=0.01, K2=0.03
constexpr double kSSIM_C1 = 6.5025;    // (0.01 * 255)^2
constexpr double kSSIM_C2 = 58.5225;   // (0.03 * 255)^2

/// 유효 마스크 픽셀 수 최소값
constexpr int kMinMaskPixels = 10;

} // anonymous namespace

// ============================================================================
// QualityMetrics - private helpers
// ============================================================================

cv::Mat QualityMetrics::toGray(const cv::Mat& src) noexcept {
    try {
        if (src.empty()) return {};

        if (src.channels() == 1) {
            return src;
        }

        cv::Mat gray;
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        return gray;
    } catch (...) {
        std::fprintf(stderr, "[IrisSDK] QualityMetrics::toGray: unknown exception\n");
        return {};
    }
}

bool QualityMetrics::validateInputs(
    const cv::Mat& img1,
    const cv::Mat& img2,
    const cv::Mat& mask) noexcept
{
    if (img1.empty() || img2.empty() || mask.empty()) {
        return false;
    }
    if (img1.size() != img2.size() || img1.size() != mask.size()) {
        return false;
    }
    if (img1.type() != img2.type()) return false;
    if (mask.type() != CV_8UC1) {
        return false;
    }
    // img1, img2는 CV_8UC3 또는 CV_8UC1 허용
    if (img1.type() != CV_8UC3 && img1.type() != CV_8UC1) {
        return false;
    }
    if (img2.type() != CV_8UC3 && img2.type() != CV_8UC1) {
        return false;
    }
    return true;
}

double QualityMetrics::computeMaskedLaplacianVariance(
    const cv::Mat& gray,
    const cv::Mat& mask) noexcept
{
    try {
        if (gray.empty() || mask.empty()) return 0.0;
        if (cv::countNonZero(mask) < kMinMaskPixels) return 0.0;

        cv::Mat laplacian;
        cv::Laplacian(gray, laplacian, CV_32F);

        cv::Scalar mean_val, stddev_val;
        cv::meanStdDev(laplacian, mean_val, stddev_val, mask);

        // variance = stddev^2
        return stddev_val[0] * stddev_val[0];
    } catch (...) {
        std::fprintf(stderr, "[IrisSDK] QualityMetrics::computeMaskedLaplacianVariance: unknown exception\n");
        return 0.0;
    }
}

double QualityMetrics::computeSSIMChannel(
    const cv::Mat& img1_gray,
    const cv::Mat& img2_gray,
    const cv::Mat& mask) noexcept
{
    try {
        if (img1_gray.empty() || img2_gray.empty()) return 0.0;
        if (cv::countNonZero(mask) < kMinMaskPixels) return 1.0;

        // float 변환
        cv::Mat i1, i2;
        img1_gray.convertTo(i1, CV_32F);
        img2_gray.convertTo(i2, CV_32F);

        // NOTE: Simplified global SSIM (not windowed). Uses whole-region statistics
        // instead of Wang et al.'s 11x11 Gaussian window for computational efficiency.
        //
        // Wang et al. (2004) SSIM 공식:
        // SSIM(x,y) = (2*mu_x*mu_y + C1)(2*sigma_xy + C2) /
        //             (mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2)
        //
        // 윈도우 기반 대신 마스크 영역 전체에 대한 글로벌 SSIM 계산

        cv::Mat i1_sq  = i1.mul(i1);
        cv::Mat i2_sq  = i2.mul(i2);
        cv::Mat i1_i2  = i1.mul(i2);

        // 마스크 영역의 평균 계산
        const cv::Scalar mu1_s = cv::mean(i1, mask);
        const cv::Scalar mu2_s = cv::mean(i2, mask);
        const float mu1 = static_cast<float>(mu1_s[0]);
        const float mu2 = static_cast<float>(mu2_s[0]);

        const cv::Scalar sigma1_sq_s = cv::mean(i1_sq, mask);
        const cv::Scalar sigma2_sq_s = cv::mean(i2_sq, mask);
        const cv::Scalar sigma12_s   = cv::mean(i1_i2, mask);

        // 분산 = E[X^2] - E[X]^2
        const float sigma1_sq = static_cast<float>(sigma1_sq_s[0]) - mu1 * mu1;
        const float sigma2_sq = static_cast<float>(sigma2_sq_s[0]) - mu2 * mu2;
        const float sigma12   = static_cast<float>(sigma12_s[0]) - mu1 * mu2;

        // SSIM
        constexpr float c1 = static_cast<float>(kSSIM_C1);
        constexpr float c2 = static_cast<float>(kSSIM_C2);
        const float numerator   = (2.0f * mu1 * mu2 + c1) *
                                  (2.0f * sigma12 + c2);
        const float denominator = (mu1 * mu1 + mu2 * mu2 + c1) *
                                  (sigma1_sq + sigma2_sq + c2);

        if (std::abs(denominator) < 1e-12f) return 1.0;

        return std::clamp(static_cast<double>(numerator / denominator), 0.0, 1.0);
    } catch (...) {
        std::fprintf(stderr, "[IrisSDK] QualityMetrics::computeSSIMChannel: unknown exception\n");
        return 0.0;
    }
}

cv::Mat QualityMetrics::createBoundaryMask(
    const cv::Mat& skin_mask,
    int boundary_width) noexcept
{
    try {
        if (skin_mask.empty() || boundary_width <= 0) return {};

        // dilate skin_mask
        cv::Mat dilated;
        const int ksize = 2 * boundary_width + 1;
        cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE, cv::Size(ksize, ksize));
        cv::dilate(skin_mask, dilated, kernel);

        // boundary = dilated XOR original
        cv::Mat boundary;
        cv::bitwise_xor(dilated, skin_mask, boundary);

        return boundary;
    } catch (...) {
        std::fprintf(stderr, "[IrisSDK] QualityMetrics::createBoundaryMask: unknown exception\n");
        return {};
    }
}

double QualityMetrics::computeMaskedGradientMagnitude(
    const cv::Mat& gray,
    const cv::Mat& mask) noexcept
{
    try {
        if (gray.empty() || mask.empty()) return 0.0;
        if (cv::countNonZero(mask) < kMinMaskPixels) return 0.0;

        cv::Mat grad_x, grad_y;
        cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3);
        cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3);

        cv::Mat magnitude;
        cv::magnitude(grad_x, grad_y, magnitude);

        const cv::Scalar mean_val = cv::mean(magnitude, mask);
        return mean_val[0];
    } catch (...) {
        std::fprintf(stderr, "[IrisSDK] QualityMetrics::computeMaskedGradientMagnitude: unknown exception\n");
        return 0.0;
    }
}

// ============================================================================
// QualityMetrics - public static methods
// ============================================================================

double QualityMetrics::measureLaplacianReductionImpl(
    const cv::Mat& originalGray,
    const cv::Mat& processedGray,
    const cv::Mat& mask) noexcept
{
    const double orig_var = computeMaskedLaplacianVariance(originalGray, mask);
    const double proc_var = computeMaskedLaplacianVariance(processedGray, mask);
    if (orig_var > 1e-12) {
        return std::clamp(1.0 - (proc_var / orig_var), 0.0, 1.0);
    }
    return 0.0;
}

double QualityMetrics::detectHaloImpl(
    const cv::Mat& originalGray,
    const cv::Mat& processedGray,
    const cv::Mat& mask) noexcept
{
    const double orig_grad = computeMaskedGradientMagnitude(originalGray, mask);
    const double proc_grad = computeMaskedGradientMagnitude(processedGray, mask);
    if (orig_grad > 1e-12) {
        return (proc_grad - orig_grad) / orig_grad;
    }
    // 원본 gradient ≈ 0 (평탄 영역): 처리 후 새 edge가 생기면 halo로 간주
    // proc_grad가 유의미하면 100% 증가로 보고 (kHaloMaxIncreaseRatio=0.15 초과 → fail)
    if (proc_grad > 1.0) {
        return 1.0;
    }
    return 0.0;
}

LaplacianResult QualityMetrics::measureLaplacianReduction(
    const cv::Mat& original,
    const cv::Mat& processed,
    const cv::Mat& skin_mask) noexcept
{
    try {
        LaplacianResult result{};

        if (!validateInputs(original, processed, skin_mask)) {
            return result;
        }

        const cv::Mat gray_orig = toGray(original);
        const cv::Mat gray_proc = toGray(processed);

        result.original_variance  = computeMaskedLaplacianVariance(gray_orig, skin_mask);
        result.processed_variance = computeMaskedLaplacianVariance(gray_proc, skin_mask);

        if (result.original_variance > 1e-12) {
            result.reduction_ratio = 1.0 -
                (result.processed_variance / result.original_variance);
            result.reduction_ratio = std::clamp(result.reduction_ratio, 0.0, 1.0);
        }

        result.passes_gate =
            (result.reduction_ratio >= kLaplacianMinReduction) &&
            (result.reduction_ratio <= kLaplacianMaxReduction);

        return result;
    } catch (...) {
        std::fprintf(stderr, "[IrisSDK] QualityMetrics::measureLaplacianReduction: unknown exception\n");
        return {};
    }
}

SSIMResult QualityMetrics::measureNonSkinSSIM(
    const cv::Mat& original,
    const cv::Mat& processed,
    const cv::Mat& skin_mask) noexcept
{
    try {
        SSIMResult result{};

        if (!validateInputs(original, processed, skin_mask)) {
            return result;
        }

        // 비-피부 마스크 (skin_mask 반전)
        cv::Mat non_skin_mask;
        cv::bitwise_not(skin_mask, non_skin_mask);

        if (cv::countNonZero(non_skin_mask) < kMinMaskPixels) {
            // 비-피부 영역이 거의 없으면 보존 완벽으로 간주
            result.ssim_value  = 1.0;
            result.passes_gate = true;
            return result;
        }

        if (original.channels() == 3) {
            // 3채널: 각 채널별 SSIM 평균
            std::vector<cv::Mat> channels_orig, channels_proc;
            cv::split(original, channels_orig);
            cv::split(processed, channels_proc);

            double ssim_sum = 0.0;
            for (int c = 0; c < 3; ++c) {
                ssim_sum += computeSSIMChannel(
                    channels_orig[c], channels_proc[c], non_skin_mask);
            }
            result.ssim_value = ssim_sum / 3.0;
        } else {
            result.ssim_value = computeSSIMChannel(
                original, processed, non_skin_mask);
        }

        result.passes_gate = (result.ssim_value > kSSIMPassThreshold);

        return result;
    } catch (...) {
        std::fprintf(stderr, "[IrisSDK] QualityMetrics::measureNonSkinSSIM: unknown exception\n");
        return {};
    }
}

HaloResult QualityMetrics::detectHalo(
    const cv::Mat& original,
    const cv::Mat& processed,
    const cv::Mat& skin_mask,
    int boundary_width) noexcept
{
    try {
        HaloResult result{};

        if (!validateInputs(original, processed, skin_mask)) {
            return result;
        }

        const cv::Mat boundary_mask = createBoundaryMask(skin_mask, boundary_width);
        if (boundary_mask.empty() || cv::countNonZero(boundary_mask) < kMinMaskPixels) {
            // 경계가 충분하지 않으면 halo 없음으로 판정
            result.passes_gate = true;
            return result;
        }

        const cv::Mat gray_orig = toGray(original);
        const cv::Mat gray_proc = toGray(processed);

        result.original_boundary_gradient  =
            computeMaskedGradientMagnitude(gray_orig, boundary_mask);
        result.processed_boundary_gradient =
            computeMaskedGradientMagnitude(gray_proc, boundary_mask);

        if (result.original_boundary_gradient > 1e-12) {
            result.gradient_increase_ratio =
                (result.processed_boundary_gradient - result.original_boundary_gradient) /
                result.original_boundary_gradient;
        } else if (result.processed_boundary_gradient > 1.0) {
            // 원본 gradient ≈ 0 인데 처리 후 새 edge 발생 → halo
            result.gradient_increase_ratio = 1.0;
        }

        // 증가율이 음수이면 gradient가 감소한 것이므로 halo 없음
        result.passes_gate = (result.gradient_increase_ratio < kHaloMaxIncreaseRatio);

        return result;
    } catch (...) {
        std::fprintf(stderr, "[IrisSDK] QualityMetrics::detectHalo: unknown exception\n");
        return {};
    }
}

GateResult QualityMetrics::evaluateQuantitativeGate(
    const cv::Mat& original,
    const cv::Mat& processed,
    const cv::Mat& skin_mask) noexcept
{
    try {
        GateResult result{};

        if (!validateInputs(original, processed, skin_mask)) {
            return result;
        }

        // toGray 중복 호출 방지: 한 번만 변환
        const cv::Mat origGray = toGray(original);
        const cv::Mat procGray = toGray(processed);

        // Laplacian (Impl 직접 호출)
        result.laplacian.original_variance  = computeMaskedLaplacianVariance(origGray, skin_mask);
        result.laplacian.processed_variance = computeMaskedLaplacianVariance(procGray, skin_mask);
        if (result.laplacian.original_variance > 1e-12) {
            result.laplacian.reduction_ratio = 1.0 -
                (result.laplacian.processed_variance / result.laplacian.original_variance);
            result.laplacian.reduction_ratio = std::clamp(result.laplacian.reduction_ratio, 0.0, 1.0);
        }
        result.laplacian.passes_gate =
            (result.laplacian.reduction_ratio >= kLaplacianMinReduction) &&
            (result.laplacian.reduction_ratio <= kLaplacianMaxReduction);

        // SSIM (채널 분리가 필요하므로 기존 메서드 호출)
        result.ssim = measureNonSkinSSIM(original, processed, skin_mask);

        // Halo (Impl 직접 호출)
        const cv::Mat boundary_mask = createBoundaryMask(skin_mask, 5);
        if (boundary_mask.empty() || cv::countNonZero(boundary_mask) < kMinMaskPixels) {
            result.halo.passes_gate = true;
        } else {
            result.halo.original_boundary_gradient  = computeMaskedGradientMagnitude(origGray, boundary_mask);
            result.halo.processed_boundary_gradient = computeMaskedGradientMagnitude(procGray, boundary_mask);
            if (result.halo.original_boundary_gradient > 1e-12) {
                result.halo.gradient_increase_ratio =
                    (result.halo.processed_boundary_gradient - result.halo.original_boundary_gradient) /
                    result.halo.original_boundary_gradient;
            } else if (result.halo.processed_boundary_gradient > 1.0) {
                result.halo.gradient_increase_ratio = 1.0;
            }
            result.halo.passes_gate = (result.halo.gradient_increase_ratio < kHaloMaxIncreaseRatio);
        }

        // 통과 수 계산
        int pass_count = 0;
        if (result.laplacian.passes_gate) ++pass_count;
        if (result.ssim.passes_gate)      ++pass_count;
        if (result.halo.passes_gate)      ++pass_count;

        // 판정
        if (pass_count == 3) {
            result.verdict = GateVerdict::GO;
        } else if (pass_count >= 2) {
            result.verdict = GateVerdict::CONDITIONAL_GO;
        } else {
            result.verdict = GateVerdict::NO_GO;
        }

        // 요약 문자열 생성
        std::ostringstream oss;
        oss << "Gate: " << gateVerdictToString(result.verdict);
        oss << " (" << pass_count << "/3 passed) | "
            << "Laplacian=" << static_cast<int>(result.laplacian.reduction_ratio * 100)
            << "% [" << (result.laplacian.passes_gate ? "OK" : "FAIL") << "] | "
            << "SSIM=" << result.ssim.ssim_value
            << " [" << (result.ssim.passes_gate ? "OK" : "FAIL") << "] | "
            << "Halo=+" << static_cast<int>(result.halo.gradient_increase_ratio * 100)
            << "% [" << (result.halo.passes_gate ? "OK" : "FAIL") << "]";

        result.summary = oss.str();

        return result;
    } catch (...) {
        std::fprintf(stderr, "[IrisSDK] QualityMetrics::evaluateQuantitativeGate: unknown exception\n");
        return {};
    }
}

// ============================================================================
// TemporalAnalyzer
// ============================================================================

void TemporalAnalyzer::addFrame(
    const cv::Mat& original,
    const cv::Mat& processed,
    const cv::Mat& skin_mask) noexcept
{
    try {
        const auto lap = QualityMetrics::measureLaplacianReduction(
            original, processed, skin_mask);

        // original_variance가 0이면 의미 없는 프레임 (빈 마스크 등)
        if (lap.original_variance > 1e-12) {
            // Keep only last 300 frames (10 seconds at 30fps) to prevent unbounded growth
            constexpr std::size_t kMaxFrames = 300;
            if (reduction_ratios_.size() >= kMaxFrames) {
                reduction_ratios_.pop_front();
            }
            reduction_ratios_.push_back(lap.reduction_ratio);
        }
    } catch (...) {
        std::fprintf(stderr, "[IrisSDK] TemporalAnalyzer::addFrame: unknown exception\n");
    }
}

void TemporalAnalyzer::addReductionRatio(double ratio) noexcept {
    // 유효 범위 외 값은 무시
    if (ratio < 0.0 || ratio > 1.0) return;

    constexpr std::size_t kMaxFrames = 300;
    if (reduction_ratios_.size() >= kMaxFrames) {
        reduction_ratios_.pop_front();
    }
    reduction_ratios_.push_back(ratio);
}

TemporalResult TemporalAnalyzer::computeTemporalVariance() const noexcept {
    TemporalResult result{};

    if (reduction_ratios_.size() < 2) {
        return result;
    }

    const auto n = static_cast<double>(reduction_ratios_.size());

    // 평균
    const double sum = std::accumulate(
        reduction_ratios_.begin(), reduction_ratios_.end(), 0.0);
    result.mean_strength = sum / n;

    // 표준편차
    double sq_sum = 0.0;
    for (const double r : reduction_ratios_) {
        const double diff = r - result.mean_strength;
        sq_sum += diff * diff;
    }
    result.std_deviation = std::sqrt(sq_sum / n);

    // 변동 계수 (CV)
    if (result.mean_strength > 1e-12) {
        result.coefficient_of_variation =
            result.std_deviation / result.mean_strength;
    }

    result.passes_gate =
        (result.coefficient_of_variation < kTemporalCVThreshold);

    return result;
}

void TemporalAnalyzer::resetTemporalData() noexcept {
    reduction_ratios_.clear();
}

std::size_t TemporalAnalyzer::frameCount() const noexcept {
    return reduction_ratios_.size();
}

} // namespace iris_sdk
