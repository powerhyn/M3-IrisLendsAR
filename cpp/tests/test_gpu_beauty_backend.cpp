/**
 * @file test_gpu_beauty_backend.cpp
 * @brief GPUBeautyBackend 단위 테스트
 *
 * Desktop에서는 스텁 모드로 동작하며 로직 테스트만 수행합니다.
 * 실제 GPU 테스트는 Android 디바이스에서 수행됩니다.
 */

#include <gtest/gtest.h>

#include "iris_sdk/gpu/gpu_beauty_backend.h"
#include "iris_sdk/gpu/shader_manager.h"
#include "iris_sdk/gpu/texture_pool.h"
#include "iris_sdk/beauty_filter.h"

namespace iris_sdk {
namespace test {

//=============================================================================
// ShaderManager 테스트
//=============================================================================
class ShaderManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        shader_manager_ = std::make_unique<ShaderManager>();
    }

    void TearDown() override {
        if (shader_manager_) {
            shader_manager_->releaseAll();
        }
    }

    std::unique_ptr<ShaderManager> shader_manager_;
};

TEST_F(ShaderManagerTest, CreatesProgramSuccessfully) {
    // 간단한 셰이더 소스
    const char* vertex = R"(
        #version 310 es
        void main() { gl_Position = vec4(0.0); }
    )";
    const char* fragment = R"(
        #version 310 es
        precision mediump float;
        out vec4 color;
        void main() { color = vec4(1.0); }
    )";

    GLuint program = 0;
    bool result = shader_manager_->createProgram(vertex, fragment, program);

    // Desktop 스텁에서는 항상 성공
    EXPECT_TRUE(result);
    EXPECT_NE(program, 0u);
}

TEST_F(ShaderManagerTest, CachesProgram) {
    const char* vertex = "vertex";
    const char* fragment = "fragment";

    GLuint program = 0;
    shader_manager_->createProgram(vertex, fragment, program);

    shader_manager_->cacheProgram("test_program", program);

    GLuint cached = shader_manager_->getProgram("test_program");
    EXPECT_EQ(cached, program);
}

TEST_F(ShaderManagerTest, ReturnsZeroForUncachedProgram) {
    GLuint program = shader_manager_->getProgram("nonexistent");
    EXPECT_EQ(program, 0u);
}

TEST_F(ShaderManagerTest, ReleasesAllPrograms) {
    const char* vertex = "v";
    const char* fragment = "f";

    GLuint p1 = 0, p2 = 0;
    shader_manager_->createProgram(vertex, fragment, p1);
    shader_manager_->createProgram(vertex, fragment, p2);

    shader_manager_->cacheProgram("p1", p1);
    shader_manager_->cacheProgram("p2", p2);

    EXPECT_EQ(shader_manager_->getCachedProgramCount(), 2u);

    shader_manager_->releaseAll();

    EXPECT_EQ(shader_manager_->getCachedProgramCount(), 0u);
}

//=============================================================================
// TexturePool 테스트
//=============================================================================
class TexturePoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        pool_ = std::make_unique<TexturePool>();
        ASSERT_TRUE(pool_->initialize(8, 1920, 1080));
    }

    void TearDown() override {
        if (pool_) {
            pool_->release();
        }
    }

    std::unique_ptr<TexturePool> pool_;
};

TEST_F(TexturePoolTest, InitializesSuccessfully) {
    EXPECT_TRUE(pool_->isInitialized());
}

TEST_F(TexturePoolTest, AcquiresRenderTarget) {
    auto* tex = pool_->acquireRenderTarget(640, 480);

    ASSERT_NE(tex, nullptr);
    EXPECT_EQ(tex->width, 640);
    EXPECT_EQ(tex->height, 480);
    EXPECT_TRUE(tex->in_use);
    EXPECT_NE(tex->texture_id, 0u);
    EXPECT_NE(tex->fbo_id, 0u);

    pool_->releaseTexture(tex);
    EXPECT_FALSE(tex->in_use);
}

TEST_F(TexturePoolTest, AcquiresPingPongPair) {
    TexturePool::TextureInfo* ping = nullptr;
    TexturePool::TextureInfo* pong = nullptr;

    bool result = pool_->acquirePingPongPair(800, 600, ping, pong);

    EXPECT_TRUE(result);
    ASSERT_NE(ping, nullptr);
    ASSERT_NE(pong, nullptr);
    EXPECT_NE(ping->texture_id, pong->texture_id);
    EXPECT_TRUE(ping->in_use);
    EXPECT_TRUE(pong->in_use);

    pool_->releaseTexture(ping);
    pool_->releaseTexture(pong);
}

TEST_F(TexturePoolTest, ReusesTextures) {
    auto* tex1 = pool_->acquireRenderTarget(640, 480);
    GLuint id1 = tex1->texture_id;
    pool_->releaseTexture(tex1);

    auto* tex2 = pool_->acquireRenderTarget(640, 480);
    GLuint id2 = tex2->texture_id;

    // 같은 크기면 재사용됨
    EXPECT_EQ(id1, id2);

    pool_->releaseTexture(tex2);
}

TEST_F(TexturePoolTest, ReportsStats) {
    auto stats = pool_->getStats();
    EXPECT_EQ(stats.total_textures, 0);
    EXPECT_EQ(stats.in_use, 0);
    EXPECT_EQ(stats.available, 0);

    auto* tex = pool_->acquireRenderTarget(640, 480);

    stats = pool_->getStats();
    EXPECT_EQ(stats.total_textures, 1);
    EXPECT_EQ(stats.in_use, 1);
    EXPECT_EQ(stats.available, 0);

    pool_->releaseTexture(tex);

    stats = pool_->getStats();
    EXPECT_EQ(stats.total_textures, 1);
    EXPECT_EQ(stats.in_use, 0);
    EXPECT_EQ(stats.available, 1);
}

TEST_F(TexturePoolTest, RejectsOversizedTextures) {
    auto* tex = pool_->acquireRenderTarget(4096, 4096);
    EXPECT_EQ(tex, nullptr);
}

TEST_F(TexturePoolTest, ResizesPool) {
    // 여러 텍스처 할당
    std::vector<TexturePool::TextureInfo*> textures;
    for (int i = 0; i < 6; i++) {
        auto* tex = pool_->acquireRenderTarget(320, 240);
        ASSERT_NE(tex, nullptr);
        textures.push_back(tex);
    }

    // 일부 반환
    pool_->releaseTexture(textures[0]);
    pool_->releaseTexture(textures[1]);
    pool_->releaseTexture(textures[2]);

    // 풀 크기 축소
    pool_->resizePool(4);

    auto stats = pool_->getStats();
    // 사용 중인 텍스처는 유지되어야 함
    EXPECT_GE(stats.in_use, 3);
    EXPECT_LE(stats.total_textures, 6);

    // 나머지 반환
    pool_->releaseTexture(textures[3]);
    pool_->releaseTexture(textures[4]);
    pool_->releaseTexture(textures[5]);
}

TEST_F(TexturePoolTest, HandlesMemoryPressure) {
    // 텍스처 할당
    auto* tex = pool_->acquireRenderTarget(640, 480);
    pool_->releaseTexture(tex);

    auto stats_before = pool_->getStats();
    EXPECT_EQ(stats_before.available, 1);

    // 높은 메모리 압력 시뮬레이션
    pool_->onMemoryPressure(80);  // TRIM_MEMORY_COMPLETE

    auto stats_after = pool_->getStats();
    // 미사용 텍스처가 정리되어야 함
    EXPECT_LE(stats_after.total_textures, stats_before.total_textures);
}

//=============================================================================
// GPUBeautyBackend 테스트 (Desktop 스텁 모드)
//=============================================================================
class GPUBeautyBackendTest : public ::testing::Test {
protected:
    void SetUp() override {
        backend_ = std::make_unique<GPUBeautyBackend>();
    }

    void TearDown() override {
        if (backend_) {
            backend_->release();
        }
    }

    std::unique_ptr<GPUBeautyBackend> backend_;
};

TEST_F(GPUBeautyBackendTest, ReportsMetadata) {
    EXPECT_STREQ(backend_->getName(), "GPUBeautyBackend");
    EXPECT_TRUE(backend_->supportsGpu());
    EXPECT_TRUE(backend_->supportsTextureProcessing());
}

TEST_F(GPUBeautyBackendTest, NotInitializedBeforeInit) {
    EXPECT_FALSE(backend_->isInitialized());
}

TEST_F(GPUBeautyBackendTest, FailsWithNullContext) {
    bool result = backend_->initialize(nullptr);
    EXPECT_FALSE(result);
    EXPECT_FALSE(backend_->isInitialized());
}

TEST_F(GPUBeautyBackendTest, ApplyReturnsErrorWhenNotInitialized) {
    std::vector<uint8_t> frame(640 * 480 * 4, 128);
    BeautyFilterConfigV2 config = BeautyFilterConfigV2Helper::defaults();
    config.enabled = true;

    IrisSdkError err = backend_->apply(
        frame.data(), 640, 480, IRIS_FORMAT_RGBA, config, nullptr);

    EXPECT_EQ(err, IRIS_SDK_ERROR_NOT_INITIALIZED);
}

TEST_F(GPUBeautyBackendTest, ApplyTextureReturnsErrorWhenNotInitialized) {
    TextureHandle input;
    input.type = TextureHandle::Type::OpenGLES;
    input.native_handle = new GLuint(1);
    input.width = 640;
    input.height = 480;

    TextureHandle output;
    BeautyFilterConfigV2 config = BeautyFilterConfigV2Helper::defaults();
    config.enabled = true;

    IrisSdkError err = backend_->applyTexture(input, output, config, nullptr);

    EXPECT_EQ(err, IRIS_SDK_ERROR_NOT_INITIALIZED);

    delete static_cast<GLuint*>(input.native_handle);
}

//=============================================================================
// BeautyFilterConfigV2 헬퍼 테스트
//=============================================================================
TEST(BeautyFilterConfigV2HelperTest, DefaultsAreValid) {
    BeautyFilterConfigV2 cfg = BeautyFilterConfigV2Helper::defaults();

    EXPECT_FALSE(cfg.enabled);
    EXPECT_FLOAT_EQ(cfg.intensity, 0.5f);
    EXPECT_FLOAT_EQ(cfg.smoothing, 0.0f);
    EXPECT_FLOAT_EQ(cfg.softFocus, 0.0f);
    EXPECT_FLOAT_EQ(cfg.brightness, 1.0f);
    EXPECT_TRUE(cfg.useGpu);
    EXPECT_TRUE(cfg.roiOnly);
    EXPECT_TRUE(cfg.protectEyes);
    EXPECT_TRUE(cfg.protectLips);

    EXPECT_TRUE(BeautyFilterConfigV2Helper::isValid(cfg));
}

TEST(BeautyFilterConfigV2HelperTest, ClampsOutOfRangeValues) {
    BeautyFilterConfigV2 cfg = {};
    cfg.smoothing = 2.0f;  // 범위 초과
    cfg.brightness = 0.1f;  // 범위 미만
    cfg.colorBalance = 5.0f;  // 범위 초과

    BeautyFilterConfigV2Helper::clamp(cfg);

    EXPECT_FLOAT_EQ(cfg.smoothing, 1.0f);
    EXPECT_FLOAT_EQ(cfg.brightness, 0.5f);
    EXPECT_FLOAT_EQ(cfg.colorBalance, 1.0f);
}

TEST(BeautyFilterConfigV2HelperTest, ConvertsFromV1) {
    BeautyFilterConfig v1 = {};
    v1.enabled = true;
    v1.intensity = 0.8f;
    v1.smoothing = 0.6f;
    v1.brightness = 1.1f;
    v1.softFocus = 0.4f;

    BeautyFilterConfigV2 v2 = BeautyFilterConfigV2Helper::fromV1(v1);

    EXPECT_TRUE(v2.enabled);
    EXPECT_FLOAT_EQ(v2.intensity, 0.8f);
    EXPECT_FLOAT_EQ(v2.smoothing, 0.6f);
    EXPECT_FLOAT_EQ(v2.brightness, 1.1f);
    EXPECT_FLOAT_EQ(v2.softFocus, 0.4f);
    // V2 전용 필드는 기본값
    EXPECT_FLOAT_EQ(v2.whitening, 0.0f);
    EXPECT_TRUE(v2.useGpu);
}

} // namespace test
} // namespace iris_sdk
