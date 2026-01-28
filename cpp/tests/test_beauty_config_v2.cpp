/**
 * @file test_beauty_config_v2.cpp
 * @brief BeautyFilterConfigV2 단위 테스트
 */

#include <gtest/gtest.h>

#include "iris_sdk/beauty_filter.h"

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

} // namespace testing
} // namespace iris_sdk
