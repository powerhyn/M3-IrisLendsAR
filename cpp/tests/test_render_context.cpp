/**
 * @file test_render_context.cpp
 * @brief RenderContext 단위 테스트
 */

#include <gtest/gtest.h>

#include "iris_sdk/gpu/render_context.h"
#include "iris_sdk/gpu/cpu_render_context.h"
#include "iris_sdk/gpu/texture_handle.h"

#include <vector>
#include <cstring>

namespace iris_sdk {
namespace testing {

//=============================================================================
// TextureHandle 테스트
//=============================================================================

TEST(TextureHandleTest, DefaultConstructorCreatesInvalidHandle) {
    TextureHandle handle;
    EXPECT_FALSE(handle.isValid());
    EXPECT_EQ(handle.type, TextureHandle::Type::Invalid);
    EXPECT_EQ(handle.native_handle, nullptr);
}

TEST(TextureHandleTest, InvalidateResetsAllFields) {
    TextureHandle handle;
    handle.native_handle = reinterpret_cast<void*>(0x1234);
    handle.type = TextureHandle::Type::CPU;
    handle.width = 640;
    handle.height = 480;
    handle.format = TextureFormat::RGBA8;
    handle.id = 42;

    EXPECT_TRUE(handle.isValid());

    handle.invalidate();

    EXPECT_FALSE(handle.isValid());
    EXPECT_EQ(handle.native_handle, nullptr);
    EXPECT_EQ(handle.type, TextureHandle::Type::Invalid);
    EXPECT_EQ(handle.width, 0);
    EXPECT_EQ(handle.height, 0);
    EXPECT_EQ(handle.format, TextureFormat::Unknown);
    EXPECT_EQ(handle.id, 0);
}

TEST(TextureHandleTest, BytesPerPixelReturnsCorrectValue) {
    TextureHandle handle;

    handle.format = TextureFormat::RGBA8;
    EXPECT_EQ(handle.bytesPerPixel(), 4);

    handle.format = TextureFormat::RGB8;
    EXPECT_EQ(handle.bytesPerPixel(), 3);

    handle.format = TextureFormat::R8;
    EXPECT_EQ(handle.bytesPerPixel(), 1);

    handle.format = TextureFormat::RGBA16F;
    EXPECT_EQ(handle.bytesPerPixel(), 8);

    handle.format = TextureFormat::Unknown;
    EXPECT_EQ(handle.bytesPerPixel(), 0);
}

TEST(TextureHandleTest, MemorySizeCalculatesCorrectly) {
    TextureHandle handle;
    handle.width = 640;
    handle.height = 480;
    handle.format = TextureFormat::RGBA8;

    EXPECT_EQ(handle.memorySize(), 640 * 480 * 4);

    handle.format = TextureFormat::RGB8;
    EXPECT_EQ(handle.memorySize(), 640 * 480 * 3);
}

//=============================================================================
// CPURenderContext 테스트
//=============================================================================

TEST(CPURenderContextTest, InitializeSucceeds) {
    CPURenderContext ctx;
    EXPECT_TRUE(ctx.initialize());
    EXPECT_TRUE(ctx.isInitialized());
}

TEST(CPURenderContextTest, DoubleInitializeSucceeds) {
    CPURenderContext ctx;
    EXPECT_TRUE(ctx.initialize());
    EXPECT_TRUE(ctx.initialize());  // 두 번째 호출도 성공
    EXPECT_TRUE(ctx.isInitialized());
}

TEST(CPURenderContextTest, ReleaseWorks) {
    CPURenderContext ctx;
    ctx.initialize();
    ctx.release();
    EXPECT_FALSE(ctx.isInitialized());
}

TEST(CPURenderContextTest, MetadataIsCorrect) {
    CPURenderContext ctx;
    ctx.initialize();

    EXPECT_STREQ(ctx.getName(), "CPURenderContext");
    EXPECT_EQ(ctx.getMajorVersion(), 1);
    EXPECT_EQ(ctx.getMinorVersion(), 0);
    EXPECT_FALSE(ctx.supportsGpu());
}

TEST(CPURenderContextTest, CreateTextureReturnsValidHandle) {
    CPURenderContext ctx;
    ctx.initialize();

    auto handle = ctx.createTexture(640, 480, TextureFormat::RGBA8);

    EXPECT_TRUE(handle.isValid());
    EXPECT_EQ(handle.type, TextureHandle::Type::CPU);
    EXPECT_EQ(handle.width, 640);
    EXPECT_EQ(handle.height, 480);
    EXPECT_EQ(handle.format, TextureFormat::RGBA8);
    EXPECT_NE(handle.id, 0);

    ctx.deleteTexture(handle);
}

TEST(CPURenderContextTest, CreateTextureWithDifferentFormats) {
    CPURenderContext ctx;
    ctx.initialize();

    auto rgba = ctx.createTexture(100, 100, TextureFormat::RGBA8);
    auto rgb = ctx.createTexture(100, 100, TextureFormat::RGB8);
    auto r8 = ctx.createTexture(100, 100, TextureFormat::R8);

    EXPECT_TRUE(rgba.isValid());
    EXPECT_TRUE(rgb.isValid());
    EXPECT_TRUE(r8.isValid());

    ctx.deleteTexture(rgba);
    ctx.deleteTexture(rgb);
    ctx.deleteTexture(r8);
}

TEST(CPURenderContextTest, CreateTextureFailsWithInvalidParams) {
    CPURenderContext ctx;
    ctx.initialize();

    auto invalid1 = ctx.createTexture(0, 480);
    EXPECT_FALSE(invalid1.isValid());

    auto invalid2 = ctx.createTexture(640, 0);
    EXPECT_FALSE(invalid2.isValid());

    auto invalid3 = ctx.createTexture(-1, 480);
    EXPECT_FALSE(invalid3.isValid());
}

TEST(CPURenderContextTest, CreateTextureFailsBeforeInit) {
    CPURenderContext ctx;
    // 초기화 하지 않음

    auto handle = ctx.createTexture(640, 480);
    EXPECT_FALSE(handle.isValid());
}

TEST(CPURenderContextTest, DeleteTextureInvalidatesHandle) {
    CPURenderContext ctx;
    ctx.initialize();

    auto handle = ctx.createTexture(640, 480);
    EXPECT_TRUE(handle.isValid());

    ctx.deleteTexture(handle);
    EXPECT_FALSE(handle.isValid());
}

TEST(CPURenderContextTest, DeleteTextureIsIdempotent) {
    CPURenderContext ctx;
    ctx.initialize();

    auto handle = ctx.createTexture(640, 480);
    ctx.deleteTexture(handle);
    ctx.deleteTexture(handle);  // 두 번째 호출도 안전
}

TEST(CPURenderContextTest, UploadDownloadRoundTrip) {
    CPURenderContext ctx;
    ctx.initialize();

    const int width = 64;
    const int height = 64;
    const size_t size = width * height * 4;  // RGBA

    // 테스트 데이터 생성 (패턴)
    std::vector<uint8_t> input(size);
    for (size_t i = 0; i < size; ++i) {
        input[i] = static_cast<uint8_t>(i % 256);
    }

    auto handle = ctx.createTexture(width, height, TextureFormat::RGBA8);
    ASSERT_TRUE(handle.isValid());

    // 업로드
    EXPECT_TRUE(ctx.uploadTexture(handle, input.data(), width, height, TextureFormat::RGBA8));

    // 다운로드
    std::vector<uint8_t> output(size);
    EXPECT_TRUE(ctx.downloadTexture(handle, output.data(), size));

    // 비교
    EXPECT_EQ(input, output);

    ctx.deleteTexture(handle);
}

TEST(CPURenderContextTest, UploadResizesTexture) {
    CPURenderContext ctx;
    ctx.initialize();

    auto handle = ctx.createTexture(64, 64, TextureFormat::RGBA8);
    ASSERT_TRUE(handle.isValid());
    EXPECT_EQ(handle.width, 64);
    EXPECT_EQ(handle.height, 64);

    // 더 큰 데이터 업로드
    const int new_width = 128;
    const int new_height = 128;
    std::vector<uint8_t> data(new_width * new_height * 4, 128);

    EXPECT_TRUE(ctx.uploadTexture(handle, data.data(), new_width, new_height, TextureFormat::RGBA8));

    // 크기가 업데이트되어야 함
    EXPECT_EQ(handle.width, new_width);
    EXPECT_EQ(handle.height, new_height);

    ctx.deleteTexture(handle);
}

TEST(CPURenderContextTest, DownloadFailsWithSmallBuffer) {
    CPURenderContext ctx;
    ctx.initialize();

    auto handle = ctx.createTexture(64, 64, TextureFormat::RGBA8);
    std::vector<uint8_t> data(64 * 64 * 4, 0);
    ctx.uploadTexture(handle, data.data(), 64, 64, TextureFormat::RGBA8);

    std::vector<uint8_t> small_buffer(100);  // 너무 작음
    EXPECT_FALSE(ctx.downloadTexture(handle, small_buffer.data(), small_buffer.size()));

    ctx.deleteTexture(handle);
}

TEST(CPURenderContextTest, GetCvMatReturnsValidPointer) {
    CPURenderContext ctx;
    ctx.initialize();

    auto handle = ctx.createTexture(64, 64, TextureFormat::RGBA8);
    ASSERT_TRUE(handle.isValid());

    cv::Mat* mat = ctx.getCvMat(handle);
    EXPECT_NE(mat, nullptr);
    EXPECT_EQ(mat->cols, 64);
    EXPECT_EQ(mat->rows, 64);
    EXPECT_EQ(mat->channels(), 4);

    ctx.deleteTexture(handle);
}

TEST(CPURenderContextTest, GetCvMatReturnsNullForInvalidHandle) {
    CPURenderContext ctx;
    ctx.initialize();

    TextureHandle invalid_handle;
    cv::Mat* mat = ctx.getCvMat(invalid_handle);
    EXPECT_EQ(mat, nullptr);
}

TEST(CPURenderContextTest, TextureCountTracksCorrectly) {
    CPURenderContext ctx;
    ctx.initialize();

    EXPECT_EQ(ctx.getTextureCount(), 0);

    auto t1 = ctx.createTexture(64, 64);
    EXPECT_EQ(ctx.getTextureCount(), 1);

    auto t2 = ctx.createTexture(64, 64);
    EXPECT_EQ(ctx.getTextureCount(), 2);

    ctx.deleteTexture(t1);
    EXPECT_EQ(ctx.getTextureCount(), 1);

    ctx.deleteTexture(t2);
    EXPECT_EQ(ctx.getTextureCount(), 0);
}

TEST(CPURenderContextTest, MemoryUsageTracksCorrectly) {
    CPURenderContext ctx;
    ctx.initialize();

    EXPECT_EQ(ctx.getTextureMemoryUsage(), 0);

    auto t1 = ctx.createTexture(100, 100, TextureFormat::RGBA8);  // 40000 bytes
    EXPECT_EQ(ctx.getTextureMemoryUsage(), 100 * 100 * 4);

    auto t2 = ctx.createTexture(100, 100, TextureFormat::RGB8);   // 30000 bytes
    EXPECT_EQ(ctx.getTextureMemoryUsage(), 100 * 100 * 4 + 100 * 100 * 3);

    ctx.deleteTexture(t1);
    ctx.deleteTexture(t2);
}

TEST(CPURenderContextTest, MakeCurrentAlwaysSucceeds) {
    CPURenderContext ctx;
    ctx.initialize();

    EXPECT_TRUE(ctx.makeCurrent());
    ctx.doneCurrent();  // no-op
}

//=============================================================================
// RenderContext 팩토리 테스트
//=============================================================================

TEST(RenderContextFactoryTest, CreateWithGpuPreferenceReturnsCPUOnDesktop) {
    auto ctx = IRenderContext::create(true);  // GPU 선호
    ASSERT_NE(ctx, nullptr);

    // Desktop에서는 GPU 미지원이므로 CPU 컨텍스트 반환
    EXPECT_FALSE(ctx->supportsGpu());
    EXPECT_STREQ(ctx->getName(), "CPURenderContext");
}

TEST(RenderContextFactoryTest, CreateWithoutGpuPreferenceReturnsCPU) {
    auto ctx = IRenderContext::create(false);  // CPU 선호
    ASSERT_NE(ctx, nullptr);

    EXPECT_FALSE(ctx->supportsGpu());
    EXPECT_STREQ(ctx->getName(), "CPURenderContext");
}

TEST(RenderContextFactoryTest, CreatedContextIsInitialized) {
    auto ctx = IRenderContext::create(false);
    ASSERT_NE(ctx, nullptr);
    EXPECT_TRUE(ctx->isInitialized());
}

TEST(RenderContextFactoryTest, GetInfoReturnsFormattedString) {
    auto ctx = IRenderContext::create(false);
    ASSERT_NE(ctx, nullptr);

    std::string info = ctx->getInfo();
    EXPECT_FALSE(info.empty());
    EXPECT_NE(info.find("CPURenderContext"), std::string::npos);
}

//=============================================================================
// Context Loss 테스트 (CPU는 Context Loss 없음)
//=============================================================================

TEST(CPURenderContextTest, IsContextLostReturnsFalse) {
    CPURenderContext ctx;
    ctx.initialize();

    EXPECT_FALSE(ctx.isContextLost());

    // Surface 이벤트는 no-op
    ctx.onSurfaceDestroyed();
    ctx.onSurfaceCreated();

    // 여전히 작동해야 함
    EXPECT_FALSE(ctx.isContextLost());
    EXPECT_TRUE(ctx.isInitialized());
}

} // namespace testing
} // namespace iris_sdk
