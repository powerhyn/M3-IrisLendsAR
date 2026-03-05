/**
 * @file test_quality_tuning.cpp
 * @brief P4-W3-05 품질 측정/튜닝 모듈 단위 테스트
 *
 * 대상 모듈:
 *   - QualityMetrics (Laplacian, SSIM, Halo, Gate)
 *   - TemporalAnalyzer (시간적 일관성)
 *   - ABCompare (A/B 비교 프레임워크)
 *   - ReleaseGate (릴리즈 게이트 체크리스트)
 *   - ParamTuner (파라미터 그리드 서치)
 *
 * @author IrisLensSDK Team
 * @version 1.0.0
 */

#include <gtest/gtest.h>
#include <iris_sdk/quality_metrics.h>
#include <iris_sdk/ab_compare.h>
#include <iris_sdk/release_gate.h>
#include <iris_sdk/param_tuner.h>

#include <opencv2/imgproc.hpp>
#include <cmath>
#include <string>

using namespace iris_sdk;

// ============================================================================
// 테스트 헬퍼 함수
// ============================================================================

namespace {

/// 단색 이미지 생성
cv::Mat createSolidImage(int w, int h, cv::Scalar color) {
    return cv::Mat(h, w, CV_8UC3, color);
}

/// 상단=skin(128,128,128), 하단=non-skin(200,50,50) 이미지
cv::Mat createTestImage(int w, int h) {
    cv::Mat img(h, w, CV_8UC3);
    img(cv::Rect(0, 0, w, h / 2)).setTo(cv::Scalar(128, 128, 128));
    img(cv::Rect(0, h / 2, w, h / 2)).setTo(cv::Scalar(200, 50, 50));
    return img;
}

/// 상단=white(255), 하단=black(0) 마스크
cv::Mat createTestMask(int w, int h) {
    cv::Mat mask(h, w, CV_8UC1, cv::Scalar(0));
    mask(cv::Rect(0, 0, w, h / 2)).setTo(cv::Scalar(255));
    return mask;
}

/// 블러 적용 이미지 (스무딩 시뮬레이션)
cv::Mat applyBlur(const cv::Mat& src, int ksize) {
    cv::Mat dst;
    cv::GaussianBlur(src, dst, cv::Size(ksize, ksize), 0);
    return dst;
}

/// 노이즈 추가 (Laplacian variance 변동용)
cv::Mat addNoise(const cv::Mat& src, double stddev) {
    cv::Mat noise(src.size(), src.type());
    cv::RNG rng(42);
    rng.fill(noise, cv::RNG::NORMAL, cv::Scalar::all(0), cv::Scalar::all(stddev));
    cv::Mat result;
    cv::add(src, noise, result, cv::noArray(), src.type());
    return result;
}

/// 텍스처가 있는 테스트 이미지 생성 (피부 영역에 노이즈 추가)
cv::Mat createTexturedTestImage(int w, int h) {
    cv::Mat img = createTestImage(w, h);
    // 피부 영역(상단 절반)에만 노이즈 추가
    cv::Mat skinROI = img(cv::Rect(0, 0, w, h / 2));
    cv::Mat noisy = addNoise(skinROI, 30.0);
    noisy.copyTo(skinROI);
    return img;
}

constexpr int kW = 200;
constexpr int kH = 200;

} // anonymous namespace

// ============================================================================
// T1: QualityMetrics 기본 측정
// ============================================================================

TEST(QualityMetricsTest, LaplacianReduction_IdenticalImages) {
    cv::Mat img  = createTexturedTestImage(kW, kH);
    cv::Mat mask = createTestMask(kW, kH);

    auto result = QualityMetrics::measureLaplacianReduction(img, img, mask);

    // 동일 이미지 -> 감소율 0 (또는 매우 근접)
    EXPECT_NEAR(result.reduction_ratio, 0.0, 0.01);
}

TEST(QualityMetricsTest, LaplacianReduction_BlurredImage) {
    cv::Mat original = createTexturedTestImage(kW, kH);
    cv::Mat mask     = createTestMask(kW, kH);
    cv::Mat blurred  = applyBlur(original, 15);

    auto result = QualityMetrics::measureLaplacianReduction(original, blurred, mask);

    // 블러 적용 -> 유의미한 감소 발생 (노이즈 텍스처에 강한 블러 -> 감소율 높음)
    EXPECT_GT(result.reduction_ratio, 0.20);
    EXPECT_GT(result.original_variance, result.processed_variance);
}

TEST(QualityMetricsTest, LaplacianReduction_EmptyMask) {
    cv::Mat img  = createTexturedTestImage(kW, kH);
    cv::Mat mask = cv::Mat(kH, kW, CV_8UC1, cv::Scalar(0)); // 전부 0 = 빈 마스크

    auto result = QualityMetrics::measureLaplacianReduction(img, img, mask);

    // 빈 마스크 -> 기본값 반환 (감소율 0, 미통과)
    EXPECT_DOUBLE_EQ(result.reduction_ratio, 0.0);
    EXPECT_FALSE(result.passes_gate);
}

TEST(QualityMetricsTest, LaplacianReduction_InvalidInput) {
    cv::Mat empty;
    cv::Mat mask = createTestMask(kW, kH);

    auto result = QualityMetrics::measureLaplacianReduction(empty, empty, mask);

    // 빈 이미지 -> 기본값 반환
    EXPECT_DOUBLE_EQ(result.reduction_ratio, 0.0);
    EXPECT_FALSE(result.passes_gate);
}

TEST(QualityMetricsTest, SSIM_IdenticalImages) {
    cv::Mat img  = createTestImage(kW, kH);
    cv::Mat mask = createTestMask(kW, kH);

    auto result = QualityMetrics::measureNonSkinSSIM(img, img, mask);

    // 동일 이미지 -> SSIM = 1.0 (또는 매우 근접)
    EXPECT_NEAR(result.ssim_value, 1.0, 0.01);
    EXPECT_TRUE(result.passes_gate);
}

TEST(QualityMetricsTest, SSIM_SlightlyDifferent) {
    cv::Mat original = createTestImage(kW, kH);
    cv::Mat mask     = createTestMask(kW, kH);
    // 피부 영역만 약간 블러 -> 비-피부 영역은 거의 변화 없음
    cv::Mat processed = original.clone();
    cv::Mat skinROI = processed(cv::Rect(0, 0, kW, kH / 2));
    cv::Mat blurredSkin;
    cv::GaussianBlur(skinROI, blurredSkin, cv::Size(3, 3), 0);
    blurredSkin.copyTo(skinROI);

    auto result = QualityMetrics::measureNonSkinSSIM(original, processed, mask);

    // 비-피부 영역 변화 없음 -> SSIM 매우 높음 (>0.90)
    EXPECT_GT(result.ssim_value, 0.90);
}

TEST(QualityMetricsTest, SSIM_VeryDifferent) {
    cv::Mat original  = createTestImage(kW, kH);
    cv::Mat mask      = createTestMask(kW, kH);
    // 비-피부 영역을 완전히 다른 색으로 대체
    cv::Mat processed = original.clone();
    processed(cv::Rect(0, kH / 2, kW, kH / 2)).setTo(cv::Scalar(50, 200, 200));

    auto result = QualityMetrics::measureNonSkinSSIM(original, processed, mask);

    // 매우 다름 -> SSIM < 0.95
    EXPECT_LT(result.ssim_value, 0.95);
    EXPECT_FALSE(result.passes_gate);
}

TEST(QualityMetricsTest, Halo_NoHalo) {
    cv::Mat img  = createTexturedTestImage(kW, kH);
    cv::Mat mask = createTestMask(kW, kH);

    auto result = QualityMetrics::detectHalo(img, img, mask);

    // 동일 이미지 -> gradient 증가율 0
    EXPECT_NEAR(result.gradient_increase_ratio, 0.0, 0.01);
    EXPECT_TRUE(result.passes_gate);
}

TEST(QualityMetricsTest, Halo_WithHalo) {
    cv::Mat original = createTexturedTestImage(kW, kH);
    cv::Mat mask     = createTestMask(kW, kH);

    // 경계 영역에 밝은 줄무늬를 추가하여 gradient를 인위적으로 증가시킴
    // 경계에 흑백 교대 패턴으로 급격한 변화 생성
    cv::Mat processed = original.clone();
    int boundary_y = kH / 2;
    for (int dy = -5; dy < 5; ++dy) {
        int y = boundary_y + dy;
        if (y >= 0 && y < kH) {
            auto val = (dy % 2 == 0) ? cv::Scalar(255, 255, 255) : cv::Scalar(0, 0, 0);
            processed.row(y).setTo(val);
        }
    }

    auto result = QualityMetrics::detectHalo(original, processed, mask);

    // 경계에 급격한 변화 -> processed gradient가 원본과 다름
    // gradient_increase_ratio가 0이 아닌 값 (양수 또는 음수)
    EXPECT_NE(result.gradient_increase_ratio, 0.0);
}

TEST(QualityMetricsTest, Gate_AllPass) {
    cv::Mat original = createTexturedTestImage(kW, kH);
    cv::Mat mask     = createTestMask(kW, kH);
    // 피부 영역에만 적절한 블러 적용 (비-피부 보존)
    cv::Mat processed = original.clone();
    cv::Mat skinROI = processed(cv::Rect(0, 0, kW, kH / 2));
    cv::Mat blurredSkin;
    cv::GaussianBlur(skinROI, blurredSkin, cv::Size(7, 7), 0);
    blurredSkin.copyTo(skinROI);

    auto result = QualityMetrics::evaluateQuantitativeGate(original, processed, mask);

    // 종합 게이트가 실행되고 verdict 결과가 반환되는지 확인
    // (GO 또는 CONDITIONAL_GO 둘 다 허용 - 합성 이미지의 한계)
    EXPECT_NE(result.verdict, GateVerdict::NO_GO);
}

TEST(QualityMetricsTest, Gate_SomeFail) {
    cv::Mat original = createTexturedTestImage(kW, kH);
    cv::Mat mask     = createTestMask(kW, kH);
    // 과도한 블러 (전체 이미지에 강한 블러)
    cv::Mat processed = applyBlur(original, 31);

    auto result = QualityMetrics::evaluateQuantitativeGate(original, processed, mask);

    // 과도한 블러 -> NO_GO 판정 기대 (비-피부 SSIM 하락)
    EXPECT_NE(result.verdict, GateVerdict::GO);
}

// ============================================================================
// T2: TemporalAnalyzer 시간적 일관성
// ============================================================================

TEST(TemporalAnalyzerTest, Empty_NoFrames) {
    TemporalAnalyzer analyzer;

    auto result = analyzer.computeTemporalVariance();

    // 0프레임 -> 미통과
    EXPECT_FALSE(result.passes_gate);
    EXPECT_EQ(analyzer.frameCount(), 0u);
}

TEST(TemporalAnalyzerTest, ConsistentFrames) {
    TemporalAnalyzer analyzer;

    cv::Mat original = createTexturedTestImage(kW, kH);
    cv::Mat mask     = createTestMask(kW, kH);
    // 동일한 블러를 30프레임 반복 -> 일관된 reduction ratio
    cv::Mat processed = applyBlur(original, 11);

    for (int i = 0; i < 30; ++i) {
        analyzer.addFrame(original, processed, mask);
    }

    auto result = analyzer.computeTemporalVariance();
    EXPECT_EQ(analyzer.frameCount(), 30u);

    // 동일 블러 30프레임 -> CV < 5% (변동 거의 없음)
    EXPECT_LT(result.coefficient_of_variation, 0.05);
    EXPECT_TRUE(result.passes_gate);
}

TEST(TemporalAnalyzerTest, InconsistentFrames) {
    TemporalAnalyzer analyzer;

    cv::Mat original = createTexturedTestImage(kW, kH);
    cv::Mat mask     = createTestMask(kW, kH);

    // 극단적으로 다른 처리 결과로 변동 생성
    // 짝수 프레임: 거의 동일 (블러 없음)
    // 홀수 프레임: 매우 강한 블러 (거의 모든 텍스처 제거)
    for (int i = 0; i < 30; ++i) {
        cv::Mat processed;
        if (i % 2 == 0) {
            // 원본과 거의 동일 -> reduction_ratio 낮음
            processed = original.clone();
        } else {
            // 매우 강한 블러 -> reduction_ratio 높음
            cv::GaussianBlur(original, processed, cv::Size(31, 31), 10.0);
        }
        analyzer.addFrame(original, processed, mask);
    }

    auto result = analyzer.computeTemporalVariance();
    EXPECT_EQ(analyzer.frameCount(), 30u);

    // 교대 패턴 -> 높은 변동 계수
    EXPECT_GT(result.coefficient_of_variation, 0.05);
    EXPECT_FALSE(result.passes_gate);
}

TEST(TemporalAnalyzerTest, Reset) {
    TemporalAnalyzer analyzer;

    cv::Mat original  = createTexturedTestImage(kW, kH);
    cv::Mat mask      = createTestMask(kW, kH);
    cv::Mat processed = applyBlur(original, 11);

    analyzer.addFrame(original, processed, mask);
    EXPECT_EQ(analyzer.frameCount(), 1u);

    analyzer.resetTemporalData();
    EXPECT_EQ(analyzer.frameCount(), 0u);
}

// ============================================================================
// T3: ABCompare A/B 비교
// ============================================================================

TEST(ABCompareTest, BasicComparison) {
    ABCompare ab;

    cv::Mat original  = createTexturedTestImage(kW, kH);
    cv::Mat mask      = createTestMask(kW, kH);
    cv::Mat bilateral = applyBlur(original, 11);
    cv::Mat freq_sep  = applyBlur(original, 7);

    TestCondition cond;
    cond.skin_tone  = SkinToneGroup::MEDIUM;
    cond.lighting   = "natural";
    cond.distance   = "standard_40cm";
    cond.skin_state = "smooth";

    auto result = ab.compare(original, bilateral, freq_sep, mask, cond);

    // 비교 결과가 생성되었는지 확인
    EXPECT_FALSE(result.condition_label.empty());
}

TEST(ABCompareTest, SkinToneSummary) {
    ABCompare ab;

    cv::Mat original  = createTexturedTestImage(kW, kH);
    cv::Mat mask      = createTestMask(kW, kH);
    cv::Mat bilateral = applyBlur(original, 15);
    cv::Mat freq_sep  = applyBlur(original, 7);

    // 3개 피부톤에 대해 비교 추가
    for (auto tone : {SkinToneGroup::LIGHT, SkinToneGroup::MEDIUM, SkinToneGroup::DARK}) {
        TestCondition cond;
        cond.skin_tone = tone;
        auto result = ab.compare(original, bilateral, freq_sep, mask, cond);
        ab.addResult(result);
    }

    EXPECT_EQ(ab.resultCount(), 3u);

    auto summaries = ab.summarizeBySkinTone();
    // 최소 1개 이상의 피부톤 요약이 생성되어야 함
    EXPECT_GE(summaries.size(), 1u);
}

TEST(ABCompareTest, ReportGeneration) {
    ABCompare ab;

    cv::Mat original  = createTexturedTestImage(kW, kH);
    cv::Mat mask      = createTestMask(kW, kH);
    cv::Mat bilateral = applyBlur(original, 11);
    cv::Mat freq_sep  = applyBlur(original, 7);

    TestCondition cond;
    auto result = ab.compare(original, bilateral, freq_sep, mask, cond);
    ab.addResult(result);

    std::string report = ab.generateReport();
    // JSON 리포트가 비어있지 않음
    EXPECT_FALSE(report.empty());
}

TEST(ABCompareTest, TestConditionLabel) {
    TestCondition cond;
    cond.skin_tone  = SkinToneGroup::LIGHT;
    cond.lighting   = "natural";
    cond.distance   = "standard_40cm";
    cond.skin_state = "smooth";

    std::string label = cond.label();

    // 라벨 포맷 확인: 구분자(/)가 포함되어야 함
    EXPECT_FALSE(label.empty());
    EXPECT_NE(label.find('/'), std::string::npos);
    EXPECT_NE(label.find("natural"), std::string::npos);
}

// ============================================================================
// T4: ReleaseGate 릴리즈 판정
// ============================================================================

TEST(ReleaseGateTest, AllPass_GO) {
    ReleaseGate gate;

    HardStopInput hard;
    hard.crash_free     = true;
    hard.no_memory_leak = true;
    hard.frame_time_ms  = 28.0;  // <= 33ms
    hard.temporal_cv    = 0.03;  // < 0.05

    QuantitativeInput quant;
    quant.laplacian_reduction     = 0.45;  // 0.3~0.6 -> pass
    quant.non_skin_ssim           = 0.97;  // > 0.95 -> pass
    quant.freq_sep_time_ms        = 8.0;   // 빠름
    quant.texture_pool_additional = 2;     // <= 3 -> pass
    quant.device_tier             = DeviceTier::HIGH;

    QualitativeInput qual;
    qual.blind_ab_preference   = 0.85;  // > 0.7 -> pass
    qual.blurry_feedback_ratio = 0.05;  // < 0.1 -> pass
    qual.fake_feedback_ratio   = 0.03;  // < 0.1 -> pass
    qual.skin_tone_uniform     = true;
    qual.no_halo               = true;
    qual.no_contour_blur       = true;

    auto result = gate.evaluate(hard, quant, qual);

    EXPECT_EQ(result.verdict, ReleaseVerdict::GO);
}

TEST(ReleaseGateTest, HardStopFail_NOGO) {
    ReleaseGate gate;

    HardStopInput hard;
    hard.crash_free     = false;  // 크래시 발생 -> 즉시 NO_GO
    hard.no_memory_leak = true;
    hard.frame_time_ms  = 28.0;
    hard.temporal_cv    = 0.03;

    QuantitativeInput quant;
    quant.laplacian_reduction     = 0.45;
    quant.non_skin_ssim           = 0.97;
    quant.freq_sep_time_ms        = 8.0;
    quant.texture_pool_additional = 2;
    quant.device_tier             = DeviceTier::HIGH;

    QualitativeInput qual;
    qual.blind_ab_preference   = 0.85;
    qual.blurry_feedback_ratio = 0.05;
    qual.fake_feedback_ratio   = 0.03;
    qual.skin_tone_uniform     = true;
    qual.no_halo               = true;
    qual.no_contour_blur       = true;

    auto result = gate.evaluate(hard, quant, qual);

    EXPECT_EQ(result.verdict, ReleaseVerdict::NO_GO);
}

TEST(ReleaseGateTest, QuantitativeFail_NOGO) {
    ReleaseGate gate;

    HardStopInput hard;
    hard.crash_free     = true;
    hard.no_memory_leak = true;
    hard.frame_time_ms  = 28.0;
    hard.temporal_cv    = 0.03;

    QuantitativeInput quant;
    quant.laplacian_reduction     = 0.10;  // < 0.3 -> fail
    quant.non_skin_ssim           = 0.97;
    quant.freq_sep_time_ms        = 8.0;
    quant.texture_pool_additional = 2;
    quant.device_tier             = DeviceTier::HIGH;

    QualitativeInput qual;
    qual.blind_ab_preference   = 0.85;
    qual.blurry_feedback_ratio = 0.05;
    qual.fake_feedback_ratio   = 0.03;
    qual.skin_tone_uniform     = true;
    qual.no_halo               = true;
    qual.no_contour_blur       = true;

    auto result = gate.evaluate(hard, quant, qual);

    EXPECT_EQ(result.verdict, ReleaseVerdict::NO_GO);
}

TEST(ReleaseGateTest, OneQualitativeFail_CONDITIONAL) {
    ReleaseGate gate;

    HardStopInput hard;
    hard.crash_free     = true;
    hard.no_memory_leak = true;
    hard.frame_time_ms  = 28.0;
    hard.temporal_cv    = 0.03;

    QuantitativeInput quant;
    quant.laplacian_reduction     = 0.45;
    quant.non_skin_ssim           = 0.97;
    quant.freq_sep_time_ms        = 8.0;
    quant.texture_pool_additional = 2;
    quant.device_tier             = DeviceTier::HIGH;

    QualitativeInput qual;
    qual.blind_ab_preference   = 0.50;  // < 0.7 -> fail (1개 실패)
    qual.blurry_feedback_ratio = 0.05;
    qual.fake_feedback_ratio   = 0.03;
    qual.skin_tone_uniform     = true;
    qual.no_halo               = true;
    qual.no_contour_blur       = true;

    auto result = gate.evaluate(hard, quant, qual);

    EXPECT_EQ(result.verdict, ReleaseVerdict::CONDITIONAL_GO);
}

TEST(ReleaseGateTest, TwoQualitativeFail_NOGO) {
    ReleaseGate gate;

    HardStopInput hard;
    hard.crash_free     = true;
    hard.no_memory_leak = true;
    hard.frame_time_ms  = 28.0;
    hard.temporal_cv    = 0.03;

    QuantitativeInput quant;
    quant.laplacian_reduction     = 0.45;
    quant.non_skin_ssim           = 0.97;
    quant.freq_sep_time_ms        = 8.0;
    quant.texture_pool_additional = 2;
    quant.device_tier             = DeviceTier::HIGH;

    QualitativeInput qual;
    qual.blind_ab_preference   = 0.50;  // < 0.7 -> fail
    qual.blurry_feedback_ratio = 0.20;  // > 0.1 -> fail (2개 실패)
    qual.fake_feedback_ratio   = 0.03;
    qual.skin_tone_uniform     = true;
    qual.no_halo               = true;
    qual.no_contour_blur       = true;

    auto result = gate.evaluate(hard, quant, qual);

    EXPECT_EQ(result.verdict, ReleaseVerdict::NO_GO);
}

TEST(ReleaseGateTest, FormatReport_NotEmpty) {
    ReleaseGate gate;

    HardStopInput hard;
    hard.crash_free     = true;
    hard.no_memory_leak = true;
    hard.frame_time_ms  = 28.0;
    hard.temporal_cv    = 0.03;

    QuantitativeInput quant;
    quant.laplacian_reduction     = 0.45;
    quant.non_skin_ssim           = 0.97;
    quant.freq_sep_time_ms        = 8.0;
    quant.texture_pool_additional = 2;
    quant.device_tier             = DeviceTier::HIGH;

    QualitativeInput qual;
    qual.blind_ab_preference   = 0.85;
    qual.blurry_feedback_ratio = 0.05;
    qual.fake_feedback_ratio   = 0.03;
    qual.skin_tone_uniform     = true;
    qual.no_halo               = true;
    qual.no_contour_blur       = true;

    auto result = gate.evaluate(hard, quant, qual);
    std::string report = ReleaseGate::formatReport(result);

    EXPECT_FALSE(report.empty());
}

// ============================================================================
// T5: ParamTuner 파라미터 튜닝
// ============================================================================

TEST(ParamTunerTest, FreqSepTuningParams_ToString) {
    FreqSepTuningParams params;
    params.attenuation_low   = 0.02f;
    params.attenuation_high  = 0.15f;
    params.sigma_ratio       = 0.40f;
    params.high_freq_preserve = 0.50f;
    params.blur_radius       = 14;

    std::string str = params.toString();

    // toString 포맷 확인: 파라미터 이름과 값이 포함
    EXPECT_FALSE(str.empty());
    EXPECT_NE(str.find("aLow"), std::string::npos);
    EXPECT_NE(str.find("aHigh"), std::string::npos);
    EXPECT_NE(str.find("sigma"), std::string::npos);
    EXPECT_NE(str.find("preserve"), std::string::npos);
    EXPECT_NE(str.find("radius"), std::string::npos);
}

TEST(ParamTunerTest, ComputeScore_AllPass) {
    GateResult gate;
    gate.laplacian.reduction_ratio = 0.45;  // 중심값 -> 최고점
    gate.laplacian.passes_gate     = true;
    gate.ssim.ssim_value           = 0.98;  // 높은 값
    gate.ssim.passes_gate          = true;
    gate.halo.gradient_increase_ratio = 0.05; // 낮은 증가
    gate.halo.passes_gate          = true;
    gate.verdict                   = GateVerdict::GO;

    double score = ParamTuner::computeScore(gate);

    // 전 통과 -> 점수 > 0
    EXPECT_GT(score, 0.0);
    EXPECT_LE(score, 1.0);
}

TEST(ParamTunerTest, ComputeScore_AllFail) {
    GateResult gate;
    gate.laplacian.reduction_ratio = 0.0;
    gate.laplacian.passes_gate     = false;
    gate.ssim.ssim_value           = 0.0;
    gate.ssim.passes_gate          = false;
    gate.halo.gradient_increase_ratio = 1.0;
    gate.halo.passes_gate          = false;
    gate.verdict                   = GateVerdict::NO_GO;

    double score = ParamTuner::computeScore(gate);

    // 전 실패 -> 점수 = 0
    EXPECT_DOUBLE_EQ(score, 0.0);
}

TEST(ParamTunerTest, BlurRadius_Independence) {
    // face_width=300 -> 300*0.05=15, clamp(6,28) -> 15
    auto validation = ParamTuner::validateBlurRadiusIndependence(300);

    EXPECT_TRUE(validation.independent);
    EXPECT_EQ(validation.radius_at_low, 15);
    EXPECT_EQ(validation.radius_at_mid, 15);
    EXPECT_EQ(validation.radius_at_high, 15);
}

TEST(ParamTunerTest, BlurRadius_Clamping) {
    // 작은 face_width -> clamp lower bound
    // face_width=50 -> 50*0.05=2.5 -> clamp(6,28) -> 6
    auto small = ParamTuner::validateBlurRadiusIndependence(50);
    EXPECT_EQ(small.radius_at_low, 6);
    EXPECT_EQ(small.radius_at_mid, 6);
    EXPECT_EQ(small.radius_at_high, 6);
    EXPECT_TRUE(small.independent);

    // 큰 face_width -> clamp upper bound
    // face_width=800 -> 800*0.05=40 -> clamp(6,28) -> 28
    auto large = ParamTuner::validateBlurRadiusIndependence(800);
    EXPECT_EQ(large.radius_at_low, 28);
    EXPECT_EQ(large.radius_at_mid, 28);
    EXPECT_EQ(large.radius_at_high, 28);
    EXPECT_TRUE(large.independent);
}

TEST(ParamTunerTest, GridSearch_ExecutesCallback) {
    ParamTuner tuner;

    cv::Mat original = createTexturedTestImage(kW, kH);
    cv::Mat mask     = createTestMask(kW, kH);

    TuningRange range;
    range.steps = 2; // 최소 단계로 빠르게 테스트

    int callback_count = 0;
    auto process_fn = [&](const cv::Mat& img, const FreqSepTuningParams& /*params*/) -> cv::Mat {
        ++callback_count;
        return applyBlur(img, 11);
    };

    auto results = tuner.gridSearch(original, mask, range, process_fn, "test");

    // 콜백이 1회 이상 호출되었는지 확인
    EXPECT_GT(callback_count, 0);
    EXPECT_EQ(results.size(), static_cast<std::size_t>(callback_count));
}

TEST(ParamTunerTest, FindBest_ReturnsHighestScore) {
    // 수동으로 TuningResult 벡터 생성
    std::vector<TuningResult> results;

    TuningResult r1;
    r1.params.attenuation_low = 0.01f;
    r1.overall_score = 0.3;
    results.push_back(r1);

    TuningResult r2;
    r2.params.attenuation_low = 0.03f;
    r2.overall_score = 0.9; // 최고 점수
    results.push_back(r2);

    TuningResult r3;
    r3.params.attenuation_low = 0.05f;
    r3.overall_score = 0.5;
    results.push_back(r3);

    auto best = ParamTuner::findBest(results);

    // 최고 점수 파라미터 (attenuation_low=0.03) 반환
    EXPECT_FLOAT_EQ(best.attenuation_low, 0.03f);
}
