/**
 * @file test_device_tier.cpp
 * @brief DeviceTier 분류 + FreqSep 매핑 테스트 (P4-W3-04: T3, T4, T5)
 *
 * - T3: classifyGpuRenderer() 기본 분류 테스트
 * - T4: mapSkinQuality() + DeviceTier 파이프라인 분기 통합 테스트
 * - T5: GPU 문자열 파싱 에지 케이스
 *
 * classifyGpuRenderer()는 static이므로 GL 컨텍스트 없이 Desktop에서 실행 가능.
 */

#include <gtest/gtest.h>
#include "iris_sdk/gpu/gpu_beauty_backend.h"

#include <string>

namespace iris_sdk {
namespace test {

using Tier = GPUBeautyBackend::DeviceTier;

// ============================================================================
// T3: classifyGpuRenderer 기본 분류 테스트
// ============================================================================

TEST(DeviceTierTest, Adreno7xxIsHigh) {
    EXPECT_EQ(GPUBeautyBackend::classifyGpuRenderer("Adreno 750"), Tier::HIGH);
    EXPECT_EQ(GPUBeautyBackend::classifyGpuRenderer("Adreno 730"), Tier::HIGH);
    EXPECT_EQ(GPUBeautyBackend::classifyGpuRenderer("Adreno 740"), Tier::HIGH);
}

TEST(DeviceTierTest, Adreno6xxIsMid) {
    EXPECT_EQ(GPUBeautyBackend::classifyGpuRenderer("Adreno 640"), Tier::MID);
    EXPECT_EQ(GPUBeautyBackend::classifyGpuRenderer("Adreno 660"), Tier::MID);
    EXPECT_EQ(GPUBeautyBackend::classifyGpuRenderer("Adreno 690"), Tier::MID);
}

TEST(DeviceTierTest, Adreno5xxIsLow) {
    EXPECT_EQ(GPUBeautyBackend::classifyGpuRenderer("Adreno 530"), Tier::LOW);
    EXPECT_EQ(GPUBeautyBackend::classifyGpuRenderer("Adreno 512"), Tier::LOW);
}

TEST(DeviceTierTest, MaliG710IsHigh) {
    EXPECT_EQ(GPUBeautyBackend::classifyGpuRenderer("Mali-G710"), Tier::HIGH);
    EXPECT_EQ(GPUBeautyBackend::classifyGpuRenderer("Mali-G715"), Tier::HIGH);
    EXPECT_EQ(GPUBeautyBackend::classifyGpuRenderer("Mali-G720"), Tier::HIGH);
}

TEST(DeviceTierTest, MaliG78IsMid) {
    EXPECT_EQ(GPUBeautyBackend::classifyGpuRenderer("Mali-G78"), Tier::MID);
}

TEST(DeviceTierTest, MaliG71IsMid) {
    EXPECT_EQ(GPUBeautyBackend::classifyGpuRenderer("Mali-G71"), Tier::MID);
    EXPECT_EQ(GPUBeautyBackend::classifyGpuRenderer("Mali-G76"), Tier::MID);
}

TEST(DeviceTierTest, MaliG52IsLow) {
    // "Mali-G52" → G52 < 70 → LOW
    EXPECT_EQ(GPUBeautyBackend::classifyGpuRenderer("Mali-G52"), Tier::LOW);
    // "Mali G52" (하이픈 없음) → "Mali-G" 매치 안됨 → unknown → LOW
    EXPECT_EQ(GPUBeautyBackend::classifyGpuRenderer("Mali G52"), Tier::LOW);
}

TEST(DeviceTierTest, AppleGpuIsHigh) {
    EXPECT_EQ(GPUBeautyBackend::classifyGpuRenderer("Apple A15 GPU"), Tier::HIGH);
    EXPECT_EQ(GPUBeautyBackend::classifyGpuRenderer("Apple M1"), Tier::HIGH);
    EXPECT_EQ(GPUBeautyBackend::classifyGpuRenderer("Apple GPU"), Tier::HIGH);
}

TEST(DeviceTierTest, NvidiaIsHigh) {
    EXPECT_EQ(GPUBeautyBackend::classifyGpuRenderer("NVIDIA GeForce RTX 4090"), Tier::HIGH);
    EXPECT_EQ(GPUBeautyBackend::classifyGpuRenderer("NVIDIA Tegra"), Tier::HIGH);
}

TEST(DeviceTierTest, PowerVRIsMid) {
    EXPECT_EQ(GPUBeautyBackend::classifyGpuRenderer("PowerVR Rogue GE8320"), Tier::MID);
}

TEST(DeviceTierTest, UnknownIsLow) {
    EXPECT_EQ(GPUBeautyBackend::classifyGpuRenderer("Some Unknown GPU"), Tier::LOW);
}

// ============================================================================
// T5: GPU 문자열 파싱 에지 케이스
// ============================================================================

TEST(DeviceTierTest, EmptyStringIsLow) {
    EXPECT_EQ(GPUBeautyBackend::classifyGpuRenderer(""), Tier::LOW);
}

TEST(DeviceTierTest, AdrenoWithTMPrefix) {
    // "Adreno (TM) 730" — (TM) 및 공백을 스킵하고 730을 파싱
    EXPECT_EQ(GPUBeautyBackend::classifyGpuRenderer("Adreno (TM) 730"), Tier::HIGH);
    EXPECT_EQ(GPUBeautyBackend::classifyGpuRenderer("Adreno (TM) 640"), Tier::MID);
}

TEST(DeviceTierTest, AdrenoWithSpaces) {
    // 다중 공백이 있어도 숫자를 올바르게 파싱
    EXPECT_EQ(GPUBeautyBackend::classifyGpuRenderer("Adreno  640"), Tier::MID);
    EXPECT_EQ(GPUBeautyBackend::classifyGpuRenderer("Adreno   750"), Tier::HIGH);
}

TEST(DeviceTierTest, MaliNoDigits) {
    // "Mali-G" 뒤에 숫자가 없는 경우 → LOW
    EXPECT_EQ(GPUBeautyBackend::classifyGpuRenderer("Mali-G"), Tier::LOW);
}

TEST(DeviceTierTest, AdrenoNoDigits) {
    // "Adreno" 뒤에 숫자가 없는 경우 → LOW
    EXPECT_EQ(GPUBeautyBackend::classifyGpuRenderer("Adreno"), Tier::LOW);
    EXPECT_EQ(GPUBeautyBackend::classifyGpuRenderer("Adreno xyz"), Tier::LOW);
}

TEST(DeviceTierTest, GarbageString) {
    EXPECT_EQ(GPUBeautyBackend::classifyGpuRenderer("!@#$%"), Tier::LOW);
    EXPECT_EQ(GPUBeautyBackend::classifyGpuRenderer("12345"), Tier::LOW);
}

TEST(DeviceTierTest, VeryLargeNumber) {
    // Adreno 99999 → val > 0 && val <= 99999 → num=99999 >= 700 → HIGH
    EXPECT_EQ(GPUBeautyBackend::classifyGpuRenderer("Adreno 99999"), Tier::HIGH);
}

// ============================================================================
// T4: mapSkinQuality + DeviceTier 파이프라인 분기 테스트
// ============================================================================

TEST(FreqSepMappingTest, ZeroQualityDisablesFreqSep) {
    auto params = GPUBeautyBackend::mapSkinQuality(0.0f, 200);
    EXPECT_FALSE(params.enabled);
}

TEST(FreqSepMappingTest, HighQualityEnablesFreqSep) {
    auto params = GPUBeautyBackend::mapSkinQuality(0.8f, 200);
    EXPECT_TRUE(params.enabled);
    // blur_radius, high_freq_preserve 등이 유효한 범위에 있어야 함
    EXPECT_GE(params.blur_radius, 6);
    EXPECT_LE(params.blur_radius, 28);
    EXPECT_GT(params.high_freq_preserve, 0.0f);
    EXPECT_LE(params.high_freq_preserve, 1.0f);
}

TEST(FreqSepMappingTest, RadiusScalesWithFaceWidth) {
    // face_width가 클수록 blur_radius가 커야 함 (5% of face_width)
    auto params_small = GPUBeautyBackend::mapSkinQuality(0.5f, 100);
    auto params_large = GPUBeautyBackend::mapSkinQuality(0.5f, 400);

    EXPECT_TRUE(params_small.enabled);
    EXPECT_TRUE(params_large.enabled);
    // 둘 다 clamp(6, 28) 범위 내
    EXPECT_GE(params_small.blur_radius, 6);
    EXPECT_GE(params_large.blur_radius, 6);
    // 400*0.05=20 > 100*0.05=5 → clamped(6) vs 20
    EXPECT_GE(params_large.blur_radius, params_small.blur_radius);
}

TEST(FreqSepMappingTest, MinimumRadiusGuard) {
    // 매우 작은 face_width → blur_radius가 최소값(6)으로 클램프
    auto params = GPUBeautyBackend::mapSkinQuality(0.5f, 10);
    EXPECT_TRUE(params.enabled);
    EXPECT_GE(params.blur_radius, 6);
}

}  // namespace test
}  // namespace iris_sdk
