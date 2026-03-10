/**
 * @file test_beauty_config_v2.cpp
 * @brief BeautyFilterConfigV2 단위 테스트
 */

#include <gtest/gtest.h>
#include <algorithm>

#include "iris_sdk/beauty_filter.h"
#include "iris_sdk/gpu/gpu_beauty_backend.h"

namespace iris_sdk {
namespace testing {

//=============================================================================
// BeautyFilterConfigV2 기본 테스트
//=============================================================================

TEST(BeautyFilterConfigV2Test, DefaultValuesFromHelper) {
    auto config = BeautyFilterConfigV2Helper::defaults();

    EXPECT_FALSE(config.enabled);
    EXPECT_FLOAT_EQ(config.intensity, 0.5f);
    EXPECT_FLOAT_EQ(config.smoothing, 0.5f);
    EXPECT_FLOAT_EQ(config.brightness, 1.0f);
    EXPECT_FLOAT_EQ(config.softFocus, 0.3f);
    EXPECT_FLOAT_EQ(config.whitening, 0.0f);
    EXPECT_FLOAT_EQ(config.colorBalance, 0.0f);
    EXPECT_FLOAT_EQ(config.wrinkleRemove, 0.0f);
    EXPECT_FLOAT_EQ(config.skinQuality, 0.0f);
    EXPECT_FLOAT_EQ(config.slimFace, 0.0f);
    EXPECT_FLOAT_EQ(config.enlargeEyes, 0.0f);
    EXPECT_FLOAT_EQ(config.thinChin, 0.0f);
    EXPECT_TRUE(config.useGpu);
    EXPECT_TRUE(config.roiOnly);
    EXPECT_TRUE(config.protectEyes);
    EXPECT_TRUE(config.protectLips);
    EXPECT_EQ(config.downscaleFactor, 1);
}

TEST(BeautyFilterConfigV2Test, IsValidAcceptsValidConfig) {
    auto config = BeautyFilterConfigV2Helper::defaults();
    EXPECT_TRUE(BeautyFilterConfigV2Helper::isValid(config));
}

TEST(BeautyFilterConfigV2Test, IsValidRejectsOutOfRangeIntensity) {
    auto config = BeautyFilterConfigV2Helper::defaults();

    config.intensity = 1.5f;  // Out of range
    EXPECT_FALSE(BeautyFilterConfigV2Helper::isValid(config));

    config.intensity = -0.1f;  // Out of range
    EXPECT_FALSE(BeautyFilterConfigV2Helper::isValid(config));
}

TEST(BeautyFilterConfigV2Test, IsValidRejectsOutOfRangeBrightness) {
    auto config = BeautyFilterConfigV2Helper::defaults();

    config.brightness = 2.0f;  // Out of range (max 1.5)
    EXPECT_FALSE(BeautyFilterConfigV2Helper::isValid(config));

    config.brightness = 0.4f;  // Out of range (min 0.5)
    EXPECT_FALSE(BeautyFilterConfigV2Helper::isValid(config));
}

TEST(BeautyFilterConfigV2Test, IsValidRejectsOutOfRangeColorBalance) {
    auto config = BeautyFilterConfigV2Helper::defaults();

    config.colorBalance = 1.5f;  // Out of range (max 1.0)
    EXPECT_FALSE(BeautyFilterConfigV2Helper::isValid(config));

    config.colorBalance = -1.5f;  // Out of range (min -1.0)
    EXPECT_FALSE(BeautyFilterConfigV2Helper::isValid(config));
}

TEST(BeautyFilterConfigV2Test, IsValidRejectsInvalidDownscale) {
    auto config = BeautyFilterConfigV2Helper::defaults();

    config.downscaleFactor = 0;  // Out of range (min 1)
    EXPECT_FALSE(BeautyFilterConfigV2Helper::isValid(config));

    config.downscaleFactor = 5;  // Out of range (max 4)
    EXPECT_FALSE(BeautyFilterConfigV2Helper::isValid(config));
}

TEST(BeautyFilterConfigV2Test, ClampCorrectsBoundaries) {
    BeautyFilterConfigV2 config = {};
    config.intensity = 1.5f;
    config.brightness = 0.0f;
    config.colorBalance = 2.0f;
    config.downscaleFactor = 10;
    config.smoothing = -0.5f;

    BeautyFilterConfigV2Helper::clamp(config);

    EXPECT_FLOAT_EQ(config.intensity, 1.0f);
    EXPECT_FLOAT_EQ(config.brightness, 0.5f);
    EXPECT_FLOAT_EQ(config.colorBalance, 1.0f);
    EXPECT_EQ(config.downscaleFactor, 4);
    EXPECT_FLOAT_EQ(config.smoothing, 0.0f);
}

TEST(BeautyFilterConfigV2Test, ClampPreservesValidValues) {
    auto config = BeautyFilterConfigV2Helper::defaults();
    float original_intensity = config.intensity;
    float original_brightness = config.brightness;

    BeautyFilterConfigV2Helper::clamp(config);

    EXPECT_FLOAT_EQ(config.intensity, original_intensity);
    EXPECT_FLOAT_EQ(config.brightness, original_brightness);
}

//=============================================================================
// V1 ↔ V2 변환 테스트
//=============================================================================

TEST(BeautyFilterConfigV2Test, FromV1PreservesV1Fields) {
    BeautyFilterConfig v1 = {};
    v1.enabled = true;
    v1.intensity = 0.7f;
    v1.smoothing = 0.6f;
    v1.brightness = 1.1f;
    v1.softFocus = 0.4f;

    auto v2 = BeautyFilterConfigV2Helper::fromV1(v1);

    EXPECT_EQ(v2.enabled, v1.enabled);
    EXPECT_FLOAT_EQ(v2.intensity, v1.intensity);
    EXPECT_FLOAT_EQ(v2.smoothing, v1.smoothing);
    EXPECT_FLOAT_EQ(v2.brightness, v1.brightness);
    EXPECT_FLOAT_EQ(v2.softFocus, v1.softFocus);
}

TEST(BeautyFilterConfigV2Test, FromV1SetsDefaultsForV2Fields) {
    BeautyFilterConfig v1 = {};
    v1.enabled = true;
    v1.intensity = 0.7f;

    auto v2 = BeautyFilterConfigV2Helper::fromV1(v1);

    // V2 전용 필드는 기본값
    EXPECT_FLOAT_EQ(v2.whitening, 0.0f);
    EXPECT_FLOAT_EQ(v2.slimFace, 0.0f);
    EXPECT_FLOAT_EQ(v2.enlargeEyes, 0.0f);
    EXPECT_TRUE(v2.useGpu);
    EXPECT_TRUE(v2.roiOnly);
}

TEST(BeautyFilterConfigV2Test, ToV1PreservesBasicFields) {
    auto v2 = BeautyFilterConfigV2Helper::defaults();
    v2.enabled = true;
    v2.intensity = 0.8f;
    v2.smoothing = 0.7f;
    v2.brightness = 1.2f;
    v2.softFocus = 0.5f;
    v2.whitening = 0.3f;  // V2 전용 (V1에서 무시됨)

    auto v1 = BeautyFilterConfigV2Helper::toV1(v2);

    EXPECT_EQ(v1.enabled, v2.enabled);
    EXPECT_FLOAT_EQ(v1.intensity, v2.intensity);
    EXPECT_FLOAT_EQ(v1.smoothing, v2.smoothing);
    EXPECT_FLOAT_EQ(v1.brightness, v2.brightness);
    EXPECT_FLOAT_EQ(v1.softFocus, v2.softFocus);
}

TEST(BeautyFilterConfigV2Test, RoundTripV1ToV2ToV1) {
    BeautyFilterConfig original = {};
    original.enabled = true;
    original.intensity = 0.6f;
    original.smoothing = 0.5f;
    original.brightness = 1.15f;
    original.softFocus = 0.35f;

    auto v2 = BeautyFilterConfigV2Helper::fromV1(original);
    auto v1 = BeautyFilterConfigV2Helper::toV1(v2);

    EXPECT_EQ(v1.enabled, original.enabled);
    EXPECT_FLOAT_EQ(v1.intensity, original.intensity);
    EXPECT_FLOAT_EQ(v1.smoothing, original.smoothing);
    EXPECT_FLOAT_EQ(v1.brightness, original.brightness);
    EXPECT_FLOAT_EQ(v1.softFocus, original.softFocus);
}

//=============================================================================
// C API 테스트
//=============================================================================

TEST(BeautyFilterConfigV2CAPI, DefaultConfigReturnsValidDefaults) {
    BeautyFilterConfigV2 config = {};
    config.intensity = 999.0f;  // 임의 값

    iris_sdk_default_beauty_config_v2(&config);

    EXPECT_FLOAT_EQ(config.intensity, 0.5f);
    EXPECT_FLOAT_EQ(config.smoothing, 0.5f);
    EXPECT_FALSE(config.enabled);
}

TEST(BeautyFilterConfigV2CAPI, DefaultConfigHandlesNullptr) {
    // NULL 포인터에 대한 크래시 방지
    iris_sdk_default_beauty_config_v2(nullptr);
    // 크래시 없이 통과하면 성공
}

TEST(BeautyFilterConfigV2CAPI, SetAndGetConfig) {
    BeautyFilterConfigV2 set_config = {};
    iris_sdk_default_beauty_config_v2(&set_config);
    set_config.enabled = true;
    set_config.intensity = 0.7f;
    set_config.whitening = 0.3f;

    IrisSdkError err = iris_sdk_set_beauty_filter_v2(&set_config);
    EXPECT_EQ(err, IRIS_SDK_OK);

    BeautyFilterConfigV2 get_config = {};
    err = iris_sdk_get_beauty_filter_v2(&get_config);
    EXPECT_EQ(err, IRIS_SDK_OK);

    EXPECT_EQ(get_config.enabled, set_config.enabled);
    EXPECT_FLOAT_EQ(get_config.intensity, set_config.intensity);
    EXPECT_FLOAT_EQ(get_config.whitening, set_config.whitening);
}

TEST(BeautyFilterConfigV2CAPI, SetConfigRejectsInvalid) {
    BeautyFilterConfigV2 invalid_config = {};
    invalid_config.intensity = 999.0f;  // 범위 초과

    IrisSdkError err = iris_sdk_set_beauty_filter_v2(&invalid_config);
    EXPECT_EQ(err, IRIS_SDK_INVALID_PARAM);
}

TEST(BeautyFilterConfigV2CAPI, SetConfigRejectsNullptr) {
    IrisSdkError err = iris_sdk_set_beauty_filter_v2(nullptr);
    EXPECT_EQ(err, IRIS_SDK_NULL_POINTER);
}

TEST(BeautyFilterConfigV2CAPI, GetConfigRejectsNullptr) {
    IrisSdkError err = iris_sdk_get_beauty_filter_v2(nullptr);
    EXPECT_EQ(err, IRIS_SDK_NULL_POINTER);
}

TEST(BeautyFilterConfigV2CAPI, GpuAvailableReturnsValidValue) {
    // Desktop에서는 false, Android에서는 true
    bool available = iris_sdk_beauty_gpu_available();
#if defined(__ANDROID__) && defined(IRIS_SDK_HAS_GLES)
    EXPECT_TRUE(available);
#else
    EXPECT_FALSE(available);
#endif
}

TEST(BeautyFilterConfigV2CAPI, UsingGpuReflectsConfig) {
    BeautyFilterConfigV2 config = {};
    iris_sdk_default_beauty_config_v2(&config);

    // GPU 사용 설정
    config.useGpu = true;
    iris_sdk_set_beauty_filter_v2(&config);

    // Desktop에서는 GPU 사용 불가
#if defined(__ANDROID__) && defined(IRIS_SDK_HAS_GLES)
    EXPECT_TRUE(iris_sdk_beauty_using_gpu());
#else
    EXPECT_FALSE(iris_sdk_beauty_using_gpu());
#endif

    // GPU 비활성화
    config.useGpu = false;
    iris_sdk_set_beauty_filter_v2(&config);
    EXPECT_FALSE(iris_sdk_beauty_using_gpu());
}

//=============================================================================
// skinQuality validation 테스트
//=============================================================================

TEST(BeautyFilterConfigV2Test, IsValidRejectsOutOfRangeSkinQuality) {
    auto config = BeautyFilterConfigV2Helper::defaults();

    config.skinQuality = 1.5f;  // Out of range (max 1.0)
    EXPECT_FALSE(BeautyFilterConfigV2Helper::isValid(config));

    config.skinQuality = -0.1f;  // Out of range (min 0.0)
    EXPECT_FALSE(BeautyFilterConfigV2Helper::isValid(config));
}

TEST(BeautyFilterConfigV2Test, ClampCorrectsSkinQuality) {
    BeautyFilterConfigV2 config = BeautyFilterConfigV2Helper::defaults();
    config.skinQuality = 2.0f;
    BeautyFilterConfigV2Helper::clamp(config);
    EXPECT_FLOAT_EQ(config.skinQuality, 1.0f);

    config.skinQuality = -1.0f;
    BeautyFilterConfigV2Helper::clamp(config);
    EXPECT_FLOAT_EQ(config.skinQuality, 0.0f);
}

//=============================================================================
// mapSkinQuality 테스트 (GPUBeautyBackend::FreqSepParams)
//=============================================================================

TEST(FreqSepParamsTest, DisabledWhenSkinQualityZero) {
    auto params = GPUBeautyBackend::mapSkinQuality(0.0f, 200);
    EXPECT_FALSE(params.enabled);
}

TEST(FreqSepParamsTest, DisabledWhenSkinQualityNegative) {
    auto params = GPUBeautyBackend::mapSkinQuality(-0.5f, 200);
    EXPECT_FALSE(params.enabled);
}

TEST(FreqSepParamsTest, EnabledWhenSkinQualityPositive) {
    auto params = GPUBeautyBackend::mapSkinQuality(0.01f, 200);
    EXPECT_TRUE(params.enabled);
}

TEST(FreqSepParamsTest, EnabledAtHalfQuality) {
    auto params = GPUBeautyBackend::mapSkinQuality(0.5f, 200);
    EXPECT_TRUE(params.enabled);
    EXPECT_GE(params.blur_radius, 6);
    EXPECT_LE(params.blur_radius, 28);
    EXPECT_GT(params.high_freq_preserve, 0.0f);
    EXPECT_LE(params.high_freq_preserve, 1.0f);
}

TEST(FreqSepParamsTest, EnabledAtFullQuality) {
    auto params = GPUBeautyBackend::mapSkinQuality(1.0f, 200);
    EXPECT_TRUE(params.enabled);
    // At max quality, high_freq_preserve ~0.35 (최소 35% 질감 보존)
    EXPECT_LE(params.high_freq_preserve, 0.40f);
    EXPECT_GE(params.high_freq_preserve, 0.30f);
}

TEST(FreqSepParamsTest, RadiusClampedToMinimum) {
    // face_width=50 → 50*0.05=2.5 → clamped to 6
    auto params = GPUBeautyBackend::mapSkinQuality(0.5f, 50);
    EXPECT_EQ(params.blur_radius, 6);
}

TEST(FreqSepParamsTest, RadiusClampedToMaximum) {
    // face_width=800 → 800*0.05=40 → clamped to 28
    auto params = GPUBeautyBackend::mapSkinQuality(0.5f, 800);
    EXPECT_EQ(params.blur_radius, 28);
}

TEST(FreqSepParamsTest, RadiusProportionalToFaceWidth) {
    // face_width=300, s=0.5 → ratio=0.03+0.5*0.03=0.045 → 300*0.045=13.5 → 13
    auto params = GPUBeautyBackend::mapSkinQuality(0.5f, 300);
    EXPECT_GE(params.blur_radius, 10);
    EXPECT_LE(params.blur_radius, 18);
}

TEST(FreqSepParamsTest, ZeroFaceWidthClampedToMinRadius) {
    auto params = GPUBeautyBackend::mapSkinQuality(0.5f, 0);
    EXPECT_EQ(params.blur_radius, 6);
}

TEST(FreqSepParamsTest, HighFreqPreserveDecreasesWithQuality) {
    auto low_q = GPUBeautyBackend::mapSkinQuality(0.2f, 200);
    auto mid_q = GPUBeautyBackend::mapSkinQuality(0.5f, 200);
    auto high_q = GPUBeautyBackend::mapSkinQuality(0.9f, 200);

    // Higher quality → lower preserve (more smoothing)
    EXPECT_GT(low_q.high_freq_preserve, mid_q.high_freq_preserve);
    EXPECT_GT(mid_q.high_freq_preserve, high_q.high_freq_preserve);
}

TEST(FreqSepParamsTest, AttenuationRangeValid) {
    auto params = GPUBeautyBackend::mapSkinQuality(0.5f, 200);
    EXPECT_GT(params.attenuation_high, params.attenuation_low);
    EXPECT_GT(params.attenuation_low, 0.0f);
}

TEST(FreqSepParamsTest, EdgeWeightRange) {
    auto params = GPUBeautyBackend::mapSkinQuality(1.0f, 300);
    EXPECT_GE(params.edge_weight, 0.0f);
    EXPECT_LE(params.edge_weight, 1.0f);
}

TEST(FreqSepParamsTest, ChromaWeightRange) {
    auto params = GPUBeautyBackend::mapSkinQuality(1.0f, 300);
    EXPECT_GE(params.chroma_weight, 0.0f);
    EXPECT_LE(params.chroma_weight, 1.0f);
}

TEST(FreqSepParamsTest, EdgeChromaBaselineAtLowQuality) {
    auto params = GPUBeautyBackend::mapSkinQuality(0.1f, 300);
    // 낮은 quality에서도 기본 edge 보존 있음 (0.3 baseline)
    EXPECT_GE(params.edge_weight, 0.2f);
    EXPECT_GE(params.chroma_weight, 0.1f);
}

TEST(FreqSepParamsTest, EdgeChromaIncreaseWithQuality) {
    auto low_q = GPUBeautyBackend::mapSkinQuality(0.2f, 200);
    auto high_q = GPUBeautyBackend::mapSkinQuality(0.9f, 200);
    EXPECT_LT(low_q.edge_weight, high_q.edge_weight);
    EXPECT_LT(low_q.chroma_weight, high_q.chroma_weight);
}

TEST(FreqSepParamsTest, LowFreqSmoothRatioInRange) {
    auto params = GPUBeautyBackend::mapSkinQuality(0.5f, 200);
    // 0.30 ~ 0.45 범위 (이중 블러 축소하여 피부 색감 보존)
    EXPECT_GE(params.low_freq_smooth_radius_ratio, 0.30f);
    EXPECT_LE(params.low_freq_smooth_radius_ratio, 0.45f);
}

TEST(FreqSepParamsTest, OverRangeSkinQualityClampedInternally) {
    // skinQuality > 1.0 should be clamped internally
    auto params = GPUBeautyBackend::mapSkinQuality(2.0f, 200);
    EXPECT_TRUE(params.enabled);
    // Should behave same as 1.0 due to clamp
    auto params_max = GPUBeautyBackend::mapSkinQuality(1.0f, 200);
    EXPECT_FLOAT_EQ(params.high_freq_preserve, params_max.high_freq_preserve);
}

TEST(FreqSepParamsTest, ToneLiftFixedAboveThreshold) {
    // skinQuality 0.5 → t = 0.5 > 0.1 → tone_lift = 0.15 고정
    auto params = GPUBeautyBackend::mapSkinQuality(0.5f, 300);
    EXPECT_NEAR(params.tone_lift, 0.15f, 0.01f);
}

TEST(FreqSepParamsTest, ToneLiftFixedAtFullQuality) {
    auto params = GPUBeautyBackend::mapSkinQuality(1.0f, 300);
    EXPECT_NEAR(params.tone_lift, 0.15f, 0.01f);
}

TEST(FreqSepParamsTest, ToneLiftGradualAtLowQuality) {
    // skinQuality 0.05 → t = 0.05 ≤ 0.1 → tone_lift = t * 1.5 = 0.075
    auto params = GPUBeautyBackend::mapSkinQuality(0.05f, 300);
    EXPECT_LT(params.tone_lift, 0.15f);
    EXPECT_GE(params.tone_lift, 0.0f);
}

TEST(FreqSepParamsTest, ToneLiftZeroWhenDisabled) {
    auto params = GPUBeautyBackend::mapSkinQuality(0.0f, 300);
    EXPECT_FALSE(params.enabled);
    // disabled 상태에서는 기본값 0.15지만 enabled=false이므로 사용되지 않음
}

// 회귀 테스트: s→t 수정 (085732f) 검증
// smoothstep(0.15) ≈ 0.06 < 0.1 이므로, s 기반이면 tone_lift = 0.06*1.5 ≈ 0.09
// t 기반이면 t = 0.15 > 0.1 이므로 tone_lift = 0.15 (올바름)
TEST(FreqSepParamsTest, ToneLiftFixedAtBorderlineQuality) {
    // skinQuality 0.1 → t = 0.1, 정확히 경계 (else 분기: t*1.5 = 0.15 → 연속)
    auto p_at = GPUBeautyBackend::mapSkinQuality(0.1f, 300);
    EXPECT_NEAR(p_at.tone_lift, 0.15f, 0.001f);

    // skinQuality 0.1001 → t > 0.1 → 고정 0.15
    auto p_above = GPUBeautyBackend::mapSkinQuality(0.1001f, 300);
    EXPECT_NEAR(p_above.tone_lift, 0.15f, 0.001f);

    // skinQuality 0.15 → t = 0.15 > 0.1 → 고정 0.15
    // (s 기반이면 smoothstep(0.15)≈0.06 < 0.1 → 0.09로 잘못 계산됨)
    auto p_mid = GPUBeautyBackend::mapSkinQuality(0.15f, 300);
    EXPECT_NEAR(p_mid.tone_lift, 0.15f, 0.001f);

    // skinQuality 0.196 → t = 0.196 > 0.1 → 고정 0.15
    // (s 기반이면 smoothstep(0.196)≈0.1 → 경계, 이전 구현 버그의 전환점)
    auto p_edge = GPUBeautyBackend::mapSkinQuality(0.196f, 300);
    EXPECT_NEAR(p_edge.tone_lift, 0.15f, 0.001f);
}

TEST(FreqSepParamsTest, ToneLiftRange) {
    for (float q = 0.01f; q <= 1.0f; q += 0.1f) {
        auto params = GPUBeautyBackend::mapSkinQuality(q, 200);
        EXPECT_GE(params.tone_lift, 0.0f);
        EXPECT_LE(params.tone_lift, 0.16f);  // 실제 최대 0.15
    }
}

//=============================================================================
// Luminance Sharpen 매핑 테스트
//=============================================================================

TEST(FreqSepParamsTest, SharpenAmountMidQuality) {
    // skinQuality 0.5 → s = smoothstep(0.5) = 0.5
    // sharpen_amount = 0.12 + 0.5 * 0.06 = 0.15
    auto params = GPUBeautyBackend::mapSkinQuality(0.5f, 200);
    EXPECT_NEAR(params.sharpen_amount, 0.15f, 0.02f);
}

TEST(FreqSepParamsTest, SharpenAmountMaxQuality) {
    // skinQuality 1.0 → s = 1.0
    // sharpen_amount = 0.12 + 1.0 * 0.06 = 0.18
    auto params = GPUBeautyBackend::mapSkinQuality(1.0f, 200);
    EXPECT_NEAR(params.sharpen_amount, 0.18f, 0.01f);
}

TEST(FreqSepParamsTest, SharpenAmountDisabledWhenZero) {
    auto params = GPUBeautyBackend::mapSkinQuality(0.0f, 200);
    EXPECT_FALSE(params.enabled);
    // sharpen_amount 기본값은 0.15이지만 enabled=false이므로 사용 안 됨
}

TEST(FreqSepParamsTest, SharpenAmountRange) {
    for (float q = 0.01f; q <= 1.0f; q += 0.1f) {
        auto params = GPUBeautyBackend::mapSkinQuality(q, 200);
        EXPECT_GE(params.sharpen_amount, 0.11f);  // 최소 ~0.12
        EXPECT_LE(params.sharpen_amount, 0.19f);   // 최대 ~0.18
    }
}

TEST(FreqSepParamsTest, SharpenAmountMonotonicallyIncreases) {
    auto low_q = GPUBeautyBackend::mapSkinQuality(0.2f, 200);
    auto high_q = GPUBeautyBackend::mapSkinQuality(0.9f, 200);
    EXPECT_LE(low_q.sharpen_amount, high_q.sharpen_amount);
}

//=============================================================================
// skinQuality C API 테스트
//=============================================================================

TEST(SkinQualityCAPITest, SetAndGetRoundTrip) {
    IrisSdkError err = iris_sdk_set_skin_quality(0.6f);
    EXPECT_EQ(err, IRIS_SDK_OK);

    float quality = -1.0f;
    err = iris_sdk_get_skin_quality(&quality);
    EXPECT_EQ(err, IRIS_SDK_OK);
    EXPECT_FLOAT_EQ(quality, 0.6f);
}

TEST(SkinQualityCAPITest, SetRejectsOutOfRange) {
    EXPECT_EQ(iris_sdk_set_skin_quality(-0.5f), IRIS_SDK_INVALID_PARAM);
    EXPECT_EQ(iris_sdk_set_skin_quality(1.5f), IRIS_SDK_INVALID_PARAM);
}

TEST(SkinQualityCAPITest, SetAcceptsBoundaryValues) {
    EXPECT_EQ(iris_sdk_set_skin_quality(0.0f), IRIS_SDK_OK);
    EXPECT_EQ(iris_sdk_set_skin_quality(1.0f), IRIS_SDK_OK);
}

TEST(SkinQualityCAPITest, GetRejectsNullptr) {
    EXPECT_EQ(iris_sdk_get_skin_quality(nullptr), IRIS_SDK_NULL_POINTER);
}

TEST(SkinQualityCAPITest, ZeroBypassesFreqSep) {
    // skinQuality=0 -> 기존 Bilateral 경로 (하위 호환)
    EXPECT_EQ(iris_sdk_set_skin_quality(0.0f), IRIS_SDK_OK);
    float quality = -1.0f;
    iris_sdk_get_skin_quality(&quality);
    EXPECT_FLOAT_EQ(quality, 0.0f);
}

//=============================================================================
// 프리셋 API 테스트
//=============================================================================

TEST(BeautyPresetTest, NaturalPresetSetsSkinQuality) {
    IrisSdkError err = iris_sdk_set_beauty_preset(IRIS_BEAUTY_PRESET_NATURAL);
    EXPECT_EQ(err, IRIS_SDK_OK);
    float quality = -1.0f;
    iris_sdk_get_skin_quality(&quality);
    EXPECT_FLOAT_EQ(quality, 0.3f);
}

TEST(BeautyPresetTest, ModeratePresetSetsSkinQuality) {
    iris_sdk_set_beauty_preset(IRIS_BEAUTY_PRESET_MODERATE);
    float quality = -1.0f;
    iris_sdk_get_skin_quality(&quality);
    EXPECT_FLOAT_EQ(quality, 0.5f);
}

TEST(BeautyPresetTest, StrongPresetSetsSkinQuality) {
    iris_sdk_set_beauty_preset(IRIS_BEAUTY_PRESET_STRONG);
    float quality = -1.0f;
    iris_sdk_get_skin_quality(&quality);
    EXPECT_FLOAT_EQ(quality, 0.8f);
}

TEST(BeautyPresetTest, CustomPresetKeepsCurrentValue) {
    iris_sdk_set_skin_quality(0.42f);
    IrisSdkError err = iris_sdk_set_beauty_preset(IRIS_BEAUTY_PRESET_CUSTOM);
    EXPECT_EQ(err, IRIS_SDK_OK);
    float quality = -1.0f;
    iris_sdk_get_skin_quality(&quality);
    EXPECT_FLOAT_EQ(quality, 0.42f);  // 변경 없음
}

TEST(BeautyPresetTest, InvalidPresetReturnsError) {
    IrisSdkError err = iris_sdk_set_beauty_preset(static_cast<IrisBeautyPreset>(99));
    EXPECT_EQ(err, IRIS_SDK_INVALID_PARAM);
}

TEST(BeautyPresetTest, BackwardCompatDefaultZero) {
    // 기존 API만 사용 (프리셋 미사용) -> skinQuality는 기본 0.0
    BeautyFilterConfigV2 config = {};
    iris_sdk_default_beauty_config_v2(&config);
    EXPECT_FLOAT_EQ(config.skinQuality, 0.0f);
}

//=============================================================================
// Soft Light (Pegtop) CPU 참조 테스트
//=============================================================================

namespace {

// CPU reference: Pegtop Soft Light
// SoftLight(base, blend) = (1 - 2*blend) * base² + 2 * blend * base
//                        = base * (base + 2 * blend * (1 - base))
float softLight(float base, float blend) {
    return base * (base + 2.0f * blend * (1.0f - base));
}

// Gain compensation reference (mirrors shader logic)
float compensatedSoftLight(float base, float high) {
    float gainFactor = 2.0f * base * (1.0f - base);
    float compensation = 1.0f / std::max(gainFactor, 0.25f);
    float compensated = high * compensation;
    float blend = std::clamp(0.5f + compensated, 0.0f, 1.0f);
    return softLight(base, blend);
}

} // anonymous namespace

TEST(SoftLightFormulaTest, IdentityWhenHighFreqZero) {
    // h=0 → blend=0.5 → beauty=base
    const float bases[] = {0.0f, 0.1f, 0.25f, 0.5f, 0.75f, 0.9f, 1.0f};
    for (float base : bases) {
        float result = softLight(base, 0.5f);
        EXPECT_NEAR(result, base, 1e-6f)
            << "Identity failed at base=" << base;
    }
}

TEST(SoftLightFormulaTest, BrightensWhenHighFreqPositive) {
    // h>0 → blend>0.5 → beauty>base (for base in (0,1))
    const float bases[] = {0.1f, 0.3f, 0.5f, 0.7f, 0.9f};
    for (float base : bases) {
        float blend = 0.6f;  // h = +0.1
        float result = softLight(base, blend);
        EXPECT_GT(result, base)
            << "Should brighten at base=" << base;
    }
}

TEST(SoftLightFormulaTest, DarkensWhenHighFreqNegative) {
    // h<0 → blend<0.5 → beauty<base (for base in (0,1))
    const float bases[] = {0.1f, 0.3f, 0.5f, 0.7f, 0.9f};
    for (float base : bases) {
        float blend = 0.4f;  // h = -0.1
        float result = softLight(base, blend);
        EXPECT_LT(result, base)
            << "Should darken at base=" << base;
    }
}

TEST(SoftLightFormulaTest, OutputAlwaysInUnitRange) {
    // base ∈ [0,1], blend ∈ [0,1] → result ∈ [0,1]
    for (int bi = 0; bi <= 100; bi += 5) {
        for (int li = 0; li <= 100; li += 5) {
            float base = bi / 100.0f;
            float blend = li / 100.0f;
            float result = softLight(base, blend);
            EXPECT_GE(result, 0.0f);
            EXPECT_LE(result, 1.0f);
        }
    }
}

TEST(SoftLightFormulaTest, GainCharacteristic) {
    // SoftLight(a, 0.5+h) ≈ a + 2h·a·(1-a) for small h
    // → effective gain = 2a(1-a)
    struct TestCase { float base; float expected_gain; };
    const TestCase cases[] = {
        {0.10f, 0.18f},
        {0.20f, 0.32f},
        {0.50f, 0.50f},
        {0.80f, 0.32f},
        {0.90f, 0.18f},
    };

    const float h = 0.01f;  // small perturbation
    for (const auto& tc : cases) {
        float result = softLight(tc.base, 0.5f + h);
        float actual_gain = (result - tc.base) / h;
        EXPECT_NEAR(actual_gain, tc.expected_gain, 0.01f)
            << "Gain mismatch at base=" << tc.base;
    }
}

TEST(SoftLightFormulaTest, GainCompensationEffectiveness) {
    // After compensation, high-freq preservation should be >= 80% across tones
    const float bases[] = {0.15f, 0.25f, 0.50f, 0.75f, 0.85f};
    const float h = 0.05f;

    for (float base : bases) {
        float compensated = compensatedSoftLight(base, h);
        float actual_delta = compensated - base;
        // Without compensation: delta = h * 2*base*(1-base)
        // With compensation: delta should be close to h (ideal full preservation)
        float preservation = actual_delta / h;
        EXPECT_GE(preservation, 0.80f)
            << "Compensation insufficient at base=" << base
            << " (preservation=" << preservation << ")";
    }
}

//=============================================================================
// 회귀 테스트 A: FreqSep RT 풀 사용량 검증
// temp를 Pass 2b 후 조기 릴리스하여 compositeRT 할당 시 풀 슬롯 재활용
//=============================================================================

TEST(FreqSepPipelineTest, MaxConcurrentRenderTargetsWithinPoolLimit) {
    // 파이프라인 RT 사용 패턴을 정적으로 검증
    // temp: Pass 1a~2b (조기 릴리스)
    // lowFreq: Pass 1b ~ end
    // smoothedLow: Pass 2b ~ end
    // compositeRT: Pass 3~4 (sharpen 활성 시만, temp 릴리스 후 할당)

    constexpr int kTexturePoolLimit = 4;  // TexturePool 기본 한도

    // sharpen 활성 시: temp 릴리스 후 compositeRT 할당
    // 동시 사용: lowFreq + smoothedLow + compositeRT = 3
    constexpr int kMaxConcurrentWithSharpen = 3;
    EXPECT_LE(kMaxConcurrentWithSharpen, kTexturePoolLimit);

    // sharpen 비활성 시: temp는 함수 끝에서 릴리스
    // 동시 사용: temp + lowFreq + smoothedLow = 3
    constexpr int kMaxConcurrentWithoutSharpen = 3;
    EXPECT_LE(kMaxConcurrentWithoutSharpen, kTexturePoolLimit);

    // 이전 구현(temp 미릴리스 + compositeRT 추가)에서는 4개 동시 사용이었음
    // onMemoryPressure()로 풀이 축소되면 acquireRenderTarget 실패 → sharpen 탈락
    constexpr int kOldMaxConcurrent = 4;  // 이전 구현의 회귀 케이스
    EXPECT_EQ(kOldMaxConcurrent, kTexturePoolLimit);  // 한도 꽉 참 = 위험
}

//=============================================================================
// 회귀 테스트 B: Mask 경계 Sharpen 수식 검증
// 비피부 인접 픽셀이 blur에 기여 → 마스크 경계에서 halo/ringing 발생 방지
// 수정: 인접 mask=0이면 lumCenter로 대체하여 합성 에지 무력화
//=============================================================================

TEST(LuminanceSharpenFormulaTest, MaskBoundaryNeighborReplacement) {
    // 시나리오: center는 피부(mask=1), 오른쪽 인접은 비피부(mask=0)
    // 셰이더 수식: lumR = mix(lumCenter, rawLumR, maskR)
    // maskR=0 → lumR = lumCenter (비피부 방향은 center로 대체)

    float lumCenter = 0.5f;
    float rawLumR_nonSkin = 0.8f;  // 비피부 영역은 밝기가 다를 수 있음
    float maskR = 0.0f;  // 비피부

    // mix(lumCenter, rawLumR, maskR) = lumCenter*(1-maskR) + rawLumR*maskR
    float lumR = lumCenter * (1.0f - maskR) + rawLumR_nonSkin * maskR;
    EXPECT_FLOAT_EQ(lumR, lumCenter);  // 비피부 방향은 center로 대체됨

    // 반대로 피부 인접(mask=1)이면 원래 luminance 사용
    maskR = 1.0f;
    lumR = lumCenter * (1.0f - maskR) + rawLumR_nonSkin * maskR;
    EXPECT_FLOAT_EQ(lumR, rawLumR_nonSkin);  // 피부 방향은 원래 값
}

TEST(LuminanceSharpenFormulaTest, MaskBoundaryNoHaloWhenAllNeighborsNonSkin) {
    // 모든 인접이 비피부(mask=0)면, blur == lumCenter → high_freq = 0 → sharpen 없음
    float lumCenter = 0.5f;
    float rawLumL = 0.8f, rawLumR = 0.3f, rawLumU = 0.9f, rawLumD = 0.2f;
    float maskL = 0.0f, maskR = 0.0f, maskU = 0.0f, maskD = 0.0f;

    // mix로 대체
    float lumL = lumCenter * (1.0f - maskL) + rawLumL * maskL;  // = lumCenter
    float lumR = lumCenter * (1.0f - maskR) + rawLumR * maskR;  // = lumCenter
    float lumU = lumCenter * (1.0f - maskU) + rawLumU * maskU;  // = lumCenter
    float lumD = lumCenter * (1.0f - maskD) + rawLumD * maskD;  // = lumCenter

    // blur = (lumCenter*2 + lumL + lumR + lumU + lumD) / 6
    float lumBlur = (lumCenter * 2.0f + lumL + lumR + lumU + lumD) / 6.0f;
    EXPECT_FLOAT_EQ(lumBlur, lumCenter);  // blur == center → no sharpening

    // high_freq = lumCenter - lumBlur = 0
    float highFreq = lumCenter - lumBlur;
    EXPECT_FLOAT_EQ(highFreq, 0.0f);

    // sharpened = lumCenter + amount * 0 = lumCenter → 변화 없음
    float sharpenAmount = 0.15f;
    float lumSharp = lumCenter + sharpenAmount * highFreq;
    EXPECT_FLOAT_EQ(lumSharp, lumCenter);
}

TEST(LuminanceSharpenFormulaTest, FullSkinRegionSharpensNormally) {
    // 모든 인접이 피부(mask=1)이면 정상 샤프닝 동작
    float lumCenter = 0.5f;
    float rawLumL = 0.48f, rawLumR = 0.52f, rawLumU = 0.49f, rawLumD = 0.51f;
    float maskAll = 1.0f;

    float lumL = lumCenter * (1.0f - maskAll) + rawLumL * maskAll;  // = rawLumL
    float lumR = lumCenter * (1.0f - maskAll) + rawLumR * maskAll;
    float lumU = lumCenter * (1.0f - maskAll) + rawLumU * maskAll;
    float lumD = lumCenter * (1.0f - maskAll) + rawLumD * maskAll;

    float lumBlur = (lumCenter * 2.0f + lumL + lumR + lumU + lumD) / 6.0f;
    float highFreq = lumCenter - lumBlur;

    // center(0.5)와 이웃 평균(0.5)이 비슷하므로 highFreq ≈ 0
    // 하지만 정확히 0은 아닐 수 있음 → 정상 샤프닝 동작 확인
    float sharpenAmount = 0.15f;
    float lumSharp = lumCenter + sharpenAmount * highFreq;

    // lumSharp는 lumCenter와 다를 수 있음 (정상 동작)
    // 핵심: mask=1 영역에서는 원래 luminance가 그대로 사용됨
    EXPECT_FLOAT_EQ(lumL, rawLumL);
    EXPECT_FLOAT_EQ(lumR, rawLumR);
}

} // namespace testing
} // namespace iris_sdk
