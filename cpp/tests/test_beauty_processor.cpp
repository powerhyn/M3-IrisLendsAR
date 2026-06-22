/**
 * @file test_beauty_processor.cpp
 * @brief BeautyProcessor 및 CPUBeautyBackend 단위 테스트
 */

#include <gtest/gtest.h>
#include "iris_sdk/beauty_processor.h"
#include "iris_sdk/cpu_beauty_backend.h"
#include "iris_sdk/gpu/render_context.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace iris_sdk {
namespace testing {

//=============================================================================
// 테스트 헬퍼
//=============================================================================

/**
 * @brief 테스트용 Face Mesh 생성
 */
class TestFaceMeshGenerator {
public:
    static void generateCenteredFace(IrisLandmark* landmarks, int count = 478) {
        for (int i = 0; i < count; i++) {
            float angle = static_cast<float>(i) / count * 2.0f * 3.14159f;
            float radius = 0.2f + 0.1f * std::sin(angle * 3);

            landmarks[i].x = 0.5f + radius * std::cos(angle) * 0.5f;
            landmarks[i].y = 0.5f + radius * std::sin(angle) * 0.6f;
            landmarks[i].z = 0.0f;
            landmarks[i].visibility = 1.0f;
        }

        // 얼굴 윤곽 설정
        const int face_oval[] = {
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
            361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
            176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
            162, 21, 54, 103, 67, 109
        };
        int oval_count = sizeof(face_oval) / sizeof(face_oval[0]);

        for (int i = 0; i < oval_count; i++) {
            float angle = static_cast<float>(i) / oval_count * 2.0f * 3.14159f;
            landmarks[face_oval[i]].x = 0.5f + 0.2f * std::cos(angle);
            landmarks[face_oval[i]].y = 0.5f + 0.25f * std::sin(angle);
        }
    }
};

/**
 * @brief 테스트용 프레임 생성
 */
std::vector<uint8_t> createTestFrame(int width, int height, int channels, uint8_t fill = 128) {
    return std::vector<uint8_t>(static_cast<size_t>(width) * height * channels, fill);
}

/**
 * @brief 프레임 변경 여부 확인
 */
bool isFrameModified(const std::vector<uint8_t>& original,
                      const std::vector<uint8_t>& modified) {
    if (original.size() != modified.size()) return true;
    return !std::equal(original.begin(), original.end(), modified.begin());
}

//=============================================================================
// CPUBeautyBackend 테스트
//=============================================================================

TEST(CPUBeautyBackendTest, Initialize_Success) {
    CPUBeautyBackend backend;

    EXPECT_FALSE(backend.isInitialized());
    EXPECT_TRUE(backend.initialize());
    EXPECT_TRUE(backend.isInitialized());
}

TEST(CPUBeautyBackendTest, Release_CleansUp) {
    CPUBeautyBackend backend;
    backend.initialize();

    backend.release();

    EXPECT_FALSE(backend.isInitialized());
}

TEST(CPUBeautyBackendTest, DoubleInitialize_IsIdempotent) {
    CPUBeautyBackend backend;

    EXPECT_TRUE(backend.initialize());
    EXPECT_TRUE(backend.initialize());
    EXPECT_TRUE(backend.isInitialized());
}

TEST(CPUBeautyBackendTest, Metadata_ReturnsExpectedValues) {
    CPUBeautyBackend backend;

    EXPECT_STREQ(backend.getName(), "CPUBeautyBackend");
    EXPECT_FALSE(backend.supportsGpu());
    EXPECT_FALSE(backend.supportsTextureProcessing());
}

TEST(CPUBeautyBackendTest, Apply_NotInitialized_ReturnsError) {
    CPUBeautyBackend backend;
    auto frame = createTestFrame(640, 480, 3);
    auto config = BeautyFilterConfigV2Helper::defaults();
    config.enabled = true;

    IrisSdkError result = backend.apply(
        frame.data(), 640, 480, IRIS_FORMAT_RGB, config, nullptr);

    // NotInitialized 정본은 IRIS_SDK_NOT_INITIALIZED=100 (W4-A 정정).
    EXPECT_EQ(result, IRIS_SDK_NOT_INITIALIZED);
}

TEST(CPUBeautyBackendTest, Apply_NullFrameData_ReturnsError) {
    CPUBeautyBackend backend;
    backend.initialize();
    auto config = BeautyFilterConfigV2Helper::defaults();
    config.enabled = true;

    IrisSdkError result = backend.apply(
        nullptr, 640, 480, IRIS_FORMAT_RGB, config, nullptr);

    EXPECT_EQ(result, IRIS_SDK_INVALID_PARAM);
}

TEST(CPUBeautyBackendTest, Apply_InvalidDimensions_ReturnsError) {
    CPUBeautyBackend backend;
    backend.initialize();
    auto frame = createTestFrame(640, 480, 3);
    auto config = BeautyFilterConfigV2Helper::defaults();
    config.enabled = true;

    EXPECT_EQ(backend.apply(frame.data(), 0, 480, IRIS_FORMAT_RGB, config, nullptr),
              IRIS_SDK_INVALID_PARAM);
    EXPECT_EQ(backend.apply(frame.data(), 640, 0, IRIS_FORMAT_RGB, config, nullptr),
              IRIS_SDK_INVALID_PARAM);
    EXPECT_EQ(backend.apply(frame.data(), -1, 480, IRIS_FORMAT_RGB, config, nullptr),
              IRIS_SDK_INVALID_PARAM);
}

TEST(CPUBeautyBackendTest, Apply_DisabledConfig_NoChange) {
    CPUBeautyBackend backend;
    backend.initialize();

    auto frame = createTestFrame(640, 480, 3);
    auto original = frame;

    auto config = BeautyFilterConfigV2Helper::defaults();
    config.enabled = false;

    IrisSdkError result = backend.apply(
        frame.data(), 640, 480, IRIS_FORMAT_RGB, config, nullptr);

    EXPECT_EQ(result, IRIS_SDK_OK);
    EXPECT_FALSE(isFrameModified(original, frame));
}

TEST(CPUBeautyBackendTest, Apply_EnabledConfig_ModifiesFrame) {
    CPUBeautyBackend backend;
    backend.initialize();

    // 그라데이션이 있는 프레임 생성 (필터 적용 시 변화 감지 가능)
    auto frame = createTestFrame(640, 480, 3);
    for (size_t i = 0; i < frame.size(); i++) {
        frame[i] = static_cast<uint8_t>((i * 7) % 256);  // 노이즈 패턴
    }
    auto original = frame;

    // P8-W2-C: smoothing(삭제됨) 대신 생존 효과 brightness로 프레임 변경을 검증.
    auto config = BeautyFilterConfigV2Helper::defaults();
    config.enabled = true;
    config.intensity = 1.0f;
    config.brightness = 1.2f;

    IrisSdkError result = backend.apply(
        frame.data(), 640, 480, IRIS_FORMAT_RGB, config, nullptr);

    EXPECT_EQ(result, IRIS_SDK_OK);
    EXPECT_TRUE(isFrameModified(original, frame));
}

TEST(CPUBeautyBackendTest, Apply_SupportsDifferentFormats) {
    CPUBeautyBackend backend;
    backend.initialize();

    // P8-W2-C: smoothing(삭제됨) 대신 brightness로 포맷별 apply() 경로를 검증.
    auto config = BeautyFilterConfigV2Helper::defaults();
    config.enabled = true;
    config.brightness = 1.1f;

    // RGB
    {
        auto frame = createTestFrame(100, 100, 3);
        EXPECT_EQ(backend.apply(frame.data(), 100, 100, IRIS_FORMAT_RGB, config, nullptr),
                  IRIS_SDK_OK);
    }

    // BGR
    {
        auto frame = createTestFrame(100, 100, 3);
        EXPECT_EQ(backend.apply(frame.data(), 100, 100, IRIS_FORMAT_BGR, config, nullptr),
                  IRIS_SDK_OK);
    }

    // RGBA
    {
        auto frame = createTestFrame(100, 100, 4);
        EXPECT_EQ(backend.apply(frame.data(), 100, 100, IRIS_FORMAT_RGBA, config, nullptr),
                  IRIS_SDK_OK);
    }

    // BGRA
    {
        auto frame = createTestFrame(100, 100, 4);
        EXPECT_EQ(backend.apply(frame.data(), 100, 100, IRIS_FORMAT_BGRA, config, nullptr),
                  IRIS_SDK_OK);
    }
}

//=============================================================================
// BeautyProcessor 테스트
//=============================================================================

TEST(BeautyProcessorTest, DefaultConstruction) {
    BeautyProcessor processor;

    EXPECT_FALSE(processor.isInitialized());
    EXPECT_FALSE(processor.isEnabled());
    EXPECT_FALSE(processor.isUsingGpu());
}

TEST(BeautyProcessorTest, Initialize_WithCPU_Success) {
    BeautyProcessor processor;

    EXPECT_TRUE(processor.initialize(false));  // CPU만
    EXPECT_TRUE(processor.isInitialized());
    EXPECT_FALSE(processor.isUsingGpu());
    EXPECT_STREQ(processor.getBackendName(), "CPUBeautyBackend");
}

TEST(BeautyProcessorTest, Release_CleansUp) {
    BeautyProcessor processor;
    processor.initialize(false);

    processor.release();

    EXPECT_FALSE(processor.isInitialized());
}

TEST(BeautyProcessorTest, DoubleInitialize_IsIdempotent) {
    BeautyProcessor processor;

    EXPECT_TRUE(processor.initialize(false));
    EXPECT_TRUE(processor.initialize(false));
    EXPECT_TRUE(processor.isInitialized());
}

TEST(BeautyProcessorTest, DI_AcceptsExternalRenderContext) {
    // IRenderContext::create로 CPU 컨텍스트 생성 (unique_ptr → shared_ptr 변환)
    std::shared_ptr<IRenderContext> render_ctx(IRenderContext::create(false).release());
    ASSERT_NE(render_ctx, nullptr);
    render_ctx->initialize();

    BeautyProcessor processor(render_ctx);
    EXPECT_TRUE(processor.initialize(false));

    // 동일한 RenderContext 공유 확인
    EXPECT_EQ(processor.getRenderContext().get(), render_ctx.get());
}

TEST(BeautyProcessorTest, SetConfig_Valid_Success) {
    BeautyProcessor processor;
    processor.initialize(false);

    auto config = BeautyFilterConfigV2Helper::defaults();
    config.enabled = true;
    config.brightness = 1.2f;  // P8-W2-D: smoothing 제거 → 생존 필드 brightness로 roundtrip 검증

    EXPECT_EQ(processor.setConfig(config), IRIS_SDK_OK);

    BeautyFilterConfigV2 out_config;
    EXPECT_EQ(processor.getConfig(out_config), IRIS_SDK_OK);
    EXPECT_TRUE(out_config.enabled);
    EXPECT_FLOAT_EQ(out_config.brightness, 1.2f);
}

TEST(BeautyProcessorTest, SetConfig_Invalid_ReturnsError) {
    BeautyProcessor processor;
    processor.initialize(false);

    BeautyFilterConfigV2 config = {};
    config.enabled = true;
    config.intensity = 2.0f;  // 범위 초과

    EXPECT_EQ(processor.setConfig(config), IRIS_SDK_INVALID_PARAM);
}

TEST(BeautyProcessorTest, Process_NotInitialized_ReturnsError) {
    BeautyProcessor processor;

    auto frame = createTestFrame(640, 480, 3);

    // NotInitialized 정본은 IRIS_SDK_NOT_INITIALIZED=100 (W4-A 정정).
    EXPECT_EQ(processor.process(frame.data(), 640, 480, IRIS_FORMAT_RGB, nullptr),
              IRIS_SDK_NOT_INITIALIZED);
}

TEST(BeautyProcessorTest, Process_DisabledConfig_NoChange) {
    BeautyProcessor processor;
    processor.initialize(false);

    auto frame = createTestFrame(640, 480, 3);
    auto original = frame;

    // 기본 설정은 disabled
    EXPECT_EQ(processor.process(frame.data(), 640, 480, IRIS_FORMAT_RGB, nullptr),
              IRIS_SDK_OK);
    EXPECT_FALSE(isFrameModified(original, frame));
}

TEST(BeautyProcessorTest, Process_EnabledConfig_ModifiesFrame) {
    BeautyProcessor processor;
    processor.initialize(false);

    // P8-W2-C: smoothing(삭제됨) 대신 생존 효과 brightness로 프레임 변경을 검증.
    auto config = BeautyFilterConfigV2Helper::defaults();
    config.enabled = true;
    config.intensity = 1.0f;
    config.brightness = 1.2f;
    processor.setConfig(config);

    // 그라데이션이 있는 프레임 생성 (필터 적용 시 변화 감지 가능)
    auto frame = createTestFrame(640, 480, 3);
    for (size_t i = 0; i < frame.size(); i++) {
        frame[i] = static_cast<uint8_t>((i * 7) % 256);  // 노이즈 패턴
    }
    auto original = frame;

    EXPECT_EQ(processor.process(frame.data(), 640, 480, IRIS_FORMAT_RGB, nullptr),
              IRIS_SDK_OK);
    EXPECT_TRUE(isFrameModified(original, frame));
}

TEST(BeautyProcessorTest, Process_WithROI_AppliesFilterToFaceOnly) {
    BeautyProcessor processor;
    processor.initialize(false);

    // P8-W2-C: smoothing(삭제됨) 대신 brightness로 ROI 경로 apply()를 검증.
    auto config = BeautyFilterConfigV2Helper::defaults();
    config.enabled = true;
    config.brightness = 1.2f;
    config.roiOnly = true;
    processor.setConfig(config);

    auto frame = createTestFrame(640, 480, 3);

    // IrisResult 설정
    IrisResult iris_result = {};
    iris_result.face_mesh_valid = true;
    TestFaceMeshGenerator::generateCenteredFace(iris_result.face_mesh, 478);

    EXPECT_EQ(processor.process(frame.data(), 640, 480, IRIS_FORMAT_RGB, &iris_result),
              IRIS_SDK_OK);
}

TEST(BeautyProcessorTest, ProcessTexture_CPUBackend_ReturnsNotSupported) {
    BeautyProcessor processor;
    processor.initialize(false);

    auto config = BeautyFilterConfigV2Helper::defaults();
    config.enabled = true;
    processor.setConfig(config);

    TextureHandle input, output;
    input.width = 640;
    input.height = 480;

    EXPECT_EQ(processor.processTexture(input, output, nullptr),
              IRIS_SDK_ERROR_NOT_SUPPORTED);
}

TEST(BeautyProcessorTest, IsEnabled_ReflectsConfig) {
    BeautyProcessor processor;
    processor.initialize(false);

    EXPECT_FALSE(processor.isEnabled());

    auto config = BeautyFilterConfigV2Helper::defaults();
    config.enabled = true;
    processor.setConfig(config);

    EXPECT_TRUE(processor.isEnabled());

    config.enabled = false;
    processor.setConfig(config);

    EXPECT_FALSE(processor.isEnabled());
}

//=============================================================================
// IBeautyBackend 인터페이스 테스트
//=============================================================================

TEST(IBeautyBackendTest, ApplyTexture_DefaultReturnsNotSupported) {
    CPUBeautyBackend backend;
    backend.initialize();

    TextureHandle input, output;
    auto config = BeautyFilterConfigV2Helper::defaults();

    EXPECT_EQ(backend.applyTexture(input, output, config, nullptr),
              IRIS_SDK_ERROR_NOT_SUPPORTED);
}

//=============================================================================
// 필터 효과 테스트
//=============================================================================

// P8-W2-C: SmoothingEffect_ReducesVariance 제거
//   (삭제된 CPU skin smoothing 효과를 적용만 하고 SUCCEED()로 끝나는 케이스).

TEST(CPUBeautyBackendTest, BrightnessEffect_IncreasesValues) {
    CPUBeautyBackend backend;
    backend.initialize();

    auto frame = createTestFrame(100, 100, 3, 100);
    auto original_sum = std::accumulate(frame.begin(), frame.end(), 0ULL);

    auto config = BeautyFilterConfigV2Helper::defaults();
    config.enabled = true;
    config.brightness = 1.3f;

    backend.apply(frame.data(), 100, 100, IRIS_FORMAT_RGB, config, nullptr);

    auto new_sum = std::accumulate(frame.begin(), frame.end(), 0ULL);
    EXPECT_GT(new_sum, original_sum);
}

} // namespace testing
} // namespace iris_sdk
