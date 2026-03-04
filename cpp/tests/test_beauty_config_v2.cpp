/**
 * @file test_beauty_config_v2.cpp
 * @brief BeautyFilterConfigV2 단위 테스트
 */

#include <gtest/gtest.h>

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
    // At max quality, high_freq_preserve should be near minimum (~0.10)
    EXPECT_LE(params.high_freq_preserve, 0.15f);
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
    // face_width=300 → 300*0.05=15 → within range
    auto params = GPUBeautyBackend::mapSkinQuality(0.5f, 300);
    EXPECT_EQ(params.blur_radius, 15);
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

TEST(FreqSepParamsTest, LowFreqSmoothRatioInRange) {
    auto params = GPUBeautyBackend::mapSkinQuality(0.5f, 200);
    EXPECT_GE(params.low_freq_smooth_radius_ratio, 0.4f);
    EXPECT_LE(params.low_freq_smooth_radius_ratio, 0.6f);
}

TEST(FreqSepParamsTest, OverRangeSkinQualityClampedInternally) {
    // skinQuality > 1.0 should be clamped internally
    auto params = GPUBeautyBackend::mapSkinQuality(2.0f, 200);
    EXPECT_TRUE(params.enabled);
    // Should behave same as 1.0 due to clamp
    auto params_max = GPUBeautyBackend::mapSkinQuality(1.0f, 200);
    EXPECT_FLOAT_EQ(params.high_freq_preserve, params_max.high_freq_preserve);
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

} // namespace testing
} // namespace iris_sdk
