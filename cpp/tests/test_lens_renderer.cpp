/**
 * @file test_lens_renderer.cpp
 * @brief LensRenderer Unit Tests
 *
 * 가상 렌즈 렌더러의 모든 기능에 대한 포괄적인 단위 테스트.
 * 초기화, 텍스처 로딩, 렌더링, LensConfig 적용 등을 검증.
 */

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <cmath>

#include "iris_sdk/lens_renderer.h"
#include "iris_sdk/types.h"

namespace iris_sdk {
namespace testing {

// ============================================================
// 테스트 헬퍼 함수
// ============================================================

/**
 * @brief 유효한 IrisResult 생성 헬퍼
 * @return 양쪽 눈이 모두 검출된 테스트용 IrisResult
 */
IrisResult createValidIrisResult() {
    IrisResult result{};
    result.detected = true;
    result.left_detected = true;
    result.right_detected = true;
    result.confidence = 0.9f;
    result.frame_width = 640;
    result.frame_height = 480;

    // 왼쪽 눈 중심 (이미지 왼쪽 1/3 지점)
    result.left_iris[0] = {0.3f, 0.4f, 0.0f, 1.0f};  // center
    result.left_iris[1] = {0.3f, 0.38f, 0.0f, 1.0f}; // top
    result.left_iris[2] = {0.3f, 0.42f, 0.0f, 1.0f}; // bottom
    result.left_iris[3] = {0.28f, 0.4f, 0.0f, 1.0f}; // left
    result.left_iris[4] = {0.32f, 0.4f, 0.0f, 1.0f}; // right
    result.left_radius = 15.0f;

    // 오른쪽 눈 (이미지 오른쪽 1/3 지점)
    result.right_iris[0] = {0.7f, 0.4f, 0.0f, 1.0f};  // center
    result.right_iris[1] = {0.7f, 0.38f, 0.0f, 1.0f}; // top
    result.right_iris[2] = {0.7f, 0.42f, 0.0f, 1.0f}; // bottom
    result.right_iris[3] = {0.68f, 0.4f, 0.0f, 1.0f}; // left
    result.right_iris[4] = {0.72f, 0.4f, 0.0f, 1.0f}; // right
    result.right_radius = 15.0f;

    result.timestamp_ms = 1000;

    return result;
}

/**
 * @brief 왼쪽 눈만 검출된 IrisResult 생성
 */
IrisResult createLeftEyeOnlyResult() {
    IrisResult result = createValidIrisResult();
    result.right_detected = false;
    return result;
}

/**
 * @brief 오른쪽 눈만 검출된 IrisResult 생성
 */
IrisResult createRightEyeOnlyResult() {
    IrisResult result = createValidIrisResult();
    result.left_detected = false;
    return result;
}

/**
 * @brief 검출 안된 IrisResult 생성
 */
IrisResult createUndetectedResult() {
    IrisResult result{};
    result.detected = false;
    result.left_detected = false;
    result.right_detected = false;
    result.confidence = 0.0f;
    result.frame_width = 640;
    result.frame_height = 480;
    return result;
}

/**
 * @brief 간단한 테스트 텍스처 생성 (RGBA)
 * @param width 텍스처 너비
 * @param height 텍스처 높이
 * @return RGBA 픽셀 데이터 벡터
 */
std::vector<uint8_t> createTestTexture(int width, int height) {
    std::vector<uint8_t> data(width * height * 4);
    for (int i = 0; i < width * height; ++i) {
        data[i * 4 + 0] = 0;     // R
        data[i * 4 + 1] = 128;   // G
        data[i * 4 + 2] = 255;   // B
        data[i * 4 + 3] = 200;   // A (투명도)
    }
    return data;
}

/**
 * @brief 테스트 프레임 생성
 * @param width 프레임 너비
 * @param height 프레임 높이
 * @return BGR 포맷 cv::Mat
 */
cv::Mat createTestFrame(int width, int height) {
    cv::Mat frame(height, width, CV_8UC3, cv::Scalar(128, 128, 128));
    return frame;
}

/**
 * @brief 기본 LensConfig 생성
 */
LensConfig createDefaultConfig() {
    LensConfig config;
    config.opacity = 0.7f;
    config.scale = 1.0f;
    config.offset_x = 0.0f;
    config.offset_y = 0.0f;
    config.blend_mode = BlendMode::Normal;
    config.edge_feather = 0.1f;
    config.apply_left = true;
    config.apply_right = true;
    return config;
}

// ============================================================
// LensRenderer 기본 테스트
// ============================================================

class LensRendererTest : public ::testing::Test {
protected:
    void SetUp() override {
        renderer = std::make_unique<LensRenderer>();
    }

    void TearDown() override {
        renderer.reset();
    }

    std::unique_ptr<LensRenderer> renderer;
};

// ============================================================
// 1. 초기화 테스트
// ============================================================

TEST_F(LensRendererTest, DefaultConstruction) {
    EXPECT_FALSE(renderer->isInitialized());
    EXPECT_FALSE(renderer->hasTexture());
}

TEST_F(LensRendererTest, InitializeSuccess) {
    EXPECT_TRUE(renderer->initialize());
    EXPECT_TRUE(renderer->isInitialized());
    EXPECT_FALSE(renderer->hasTexture());
}

TEST_F(LensRendererTest, InitializeIdempotent) {
    // 여러 번 호출해도 안전
    EXPECT_TRUE(renderer->initialize());
    EXPECT_TRUE(renderer->initialize());
    EXPECT_TRUE(renderer->isInitialized());
}

TEST_F(LensRendererTest, ReleaseAfterInit) {
    renderer->initialize();
    EXPECT_TRUE(renderer->isInitialized());

    renderer->release();
    EXPECT_FALSE(renderer->isInitialized());
    EXPECT_FALSE(renderer->hasTexture());
}

TEST_F(LensRendererTest, ReleaseWithoutInit) {
    // 초기화 없이 release 호출해도 안전
    EXPECT_NO_THROW(renderer->release());
    EXPECT_FALSE(renderer->isInitialized());
}

TEST_F(LensRendererTest, DoubleRelease) {
    renderer->initialize();
    renderer->release();
    EXPECT_NO_THROW(renderer->release());
    EXPECT_FALSE(renderer->isInitialized());
}

TEST_F(LensRendererTest, ReleaseResetsTextureState) {
    renderer->initialize();
    auto texture = createTestTexture(64, 64);
    renderer->loadTexture(texture.data(), 64, 64);
    EXPECT_TRUE(renderer->hasTexture());

    renderer->release();
    EXPECT_FALSE(renderer->hasTexture());
    EXPECT_FALSE(renderer->isInitialized());
}

// ============================================================
// 2. 텍스처 로딩 테스트
// ============================================================

TEST_F(LensRendererTest, LoadTextureFromMemorySuccess) {
    renderer->initialize();
    auto texture = createTestTexture(64, 64);

    EXPECT_TRUE(renderer->loadTexture(texture.data(), 64, 64));
    EXPECT_TRUE(renderer->hasTexture());
}

TEST_F(LensRendererTest, LoadTextureFromMemoryNullData) {
    renderer->initialize();

    EXPECT_FALSE(renderer->loadTexture(nullptr, 64, 64));
    EXPECT_FALSE(renderer->hasTexture());
}

TEST_F(LensRendererTest, LoadTextureFromMemoryInvalidDimensions) {
    renderer->initialize();
    auto texture = createTestTexture(64, 64);

    // 0 크기
    EXPECT_FALSE(renderer->loadTexture(texture.data(), 0, 64));
    EXPECT_FALSE(renderer->loadTexture(texture.data(), 64, 0));

    // 음수 크기
    EXPECT_FALSE(renderer->loadTexture(texture.data(), -10, 64));
    EXPECT_FALSE(renderer->loadTexture(texture.data(), 64, -10));

    EXPECT_FALSE(renderer->hasTexture());
}

TEST_F(LensRendererTest, LoadTextureWithoutInit) {
    auto texture = createTestTexture(64, 64);

    EXPECT_FALSE(renderer->loadTexture(texture.data(), 64, 64));
    EXPECT_FALSE(renderer->hasTexture());
}

TEST_F(LensRendererTest, LoadTextureReplacesOldTexture) {
    renderer->initialize();
    auto texture1 = createTestTexture(32, 32);
    auto texture2 = createTestTexture(64, 64);

    EXPECT_TRUE(renderer->loadTexture(texture1.data(), 32, 32));
    EXPECT_TRUE(renderer->hasTexture());

    // 새 텍스처 로드 시 이전 텍스처 교체
    EXPECT_TRUE(renderer->loadTexture(texture2.data(), 64, 64));
    EXPECT_TRUE(renderer->hasTexture());
}

TEST_F(LensRendererTest, UnloadTexture) {
    renderer->initialize();
    auto texture = createTestTexture(64, 64);
    renderer->loadTexture(texture.data(), 64, 64);
    EXPECT_TRUE(renderer->hasTexture());

    renderer->unloadTexture();
    EXPECT_FALSE(renderer->hasTexture());
    EXPECT_TRUE(renderer->isInitialized()); // 초기화 상태는 유지
}

TEST_F(LensRendererTest, UnloadTextureWithoutLoading) {
    renderer->initialize();

    EXPECT_NO_THROW(renderer->unloadTexture());
    EXPECT_FALSE(renderer->hasTexture());
}

TEST_F(LensRendererTest, LoadTextureFromFileInvalidPath) {
    renderer->initialize();

    EXPECT_FALSE(renderer->loadTexture("/nonexistent/path/texture.png"));
    EXPECT_FALSE(renderer->hasTexture());
}

TEST_F(LensRendererTest, LoadTextureFromFileEmptyPath) {
    renderer->initialize();

    EXPECT_FALSE(renderer->loadTexture(""));
    EXPECT_FALSE(renderer->hasTexture());
}

// ============================================================
// 3. 렌더링 테스트 (양눈)
// ============================================================

TEST_F(LensRendererTest, RenderBothEyesSuccess) {
    renderer->initialize();
    auto texture = createTestTexture(64, 64);
    renderer->loadTexture(texture.data(), 64, 64);

    auto frame = createTestFrame(640, 480);
    auto result = createValidIrisResult();
    auto config = createDefaultConfig();

    EXPECT_TRUE(renderer->render(frame, result, config));
}

TEST_F(LensRendererTest, RenderWithoutInit) {
    auto frame = createTestFrame(640, 480);
    auto result = createValidIrisResult();
    auto config = createDefaultConfig();

    EXPECT_FALSE(renderer->render(frame, result, config));
}

TEST_F(LensRendererTest, RenderWithoutTexture) {
    renderer->initialize();

    auto frame = createTestFrame(640, 480);
    auto result = createValidIrisResult();
    auto config = createDefaultConfig();

    EXPECT_FALSE(renderer->render(frame, result, config));
}

TEST_F(LensRendererTest, RenderUndetectedResult) {
    renderer->initialize();
    auto texture = createTestTexture(64, 64);
    renderer->loadTexture(texture.data(), 64, 64);

    auto frame = createTestFrame(640, 480);
    auto result = createUndetectedResult();
    auto config = createDefaultConfig();

    EXPECT_FALSE(renderer->render(frame, result, config));
}

TEST_F(LensRendererTest, RenderEmptyFrame) {
    renderer->initialize();
    auto texture = createTestTexture(64, 64);
    renderer->loadTexture(texture.data(), 64, 64);

    cv::Mat empty_frame;
    auto result = createValidIrisResult();
    auto config = createDefaultConfig();

    // 빈 프레임이어도 렌더링 시도는 하지만 실제로는 아무것도 그려지지 않음
    // 구현에 따라 true를 반환할 수 있음 (프레임 검증이 renderEye 내부에서 이루어지는 경우)
    renderer->render(empty_frame, result, config);
    // 결과가 어떻든 크래시가 없어야 함
}

// ============================================================
// 4. 렌더링 테스트 (왼쪽 눈만)
// ============================================================

TEST_F(LensRendererTest, RenderLeftEyeOnlySuccess) {
    renderer->initialize();
    auto texture = createTestTexture(64, 64);
    renderer->loadTexture(texture.data(), 64, 64);

    auto frame = createTestFrame(640, 480);
    auto result = createLeftEyeOnlyResult();
    auto config = createDefaultConfig();

    EXPECT_TRUE(renderer->renderLeftEye(frame, result, config));
}

TEST_F(LensRendererTest, RenderLeftEyeWhenNotDetected) {
    renderer->initialize();
    auto texture = createTestTexture(64, 64);
    renderer->loadTexture(texture.data(), 64, 64);

    auto frame = createTestFrame(640, 480);
    auto result = createRightEyeOnlyResult(); // 왼쪽 눈 검출 안됨
    auto config = createDefaultConfig();

    EXPECT_FALSE(renderer->renderLeftEye(frame, result, config));
}

TEST_F(LensRendererTest, RenderLeftEyeWithoutInit) {
    auto frame = createTestFrame(640, 480);
    auto result = createValidIrisResult();
    auto config = createDefaultConfig();

    EXPECT_FALSE(renderer->renderLeftEye(frame, result, config));
}

// ============================================================
// 5. 렌더링 테스트 (오른쪽 눈만)
// ============================================================

TEST_F(LensRendererTest, RenderRightEyeOnlySuccess) {
    renderer->initialize();
    auto texture = createTestTexture(64, 64);
    renderer->loadTexture(texture.data(), 64, 64);

    auto frame = createTestFrame(640, 480);
    auto result = createRightEyeOnlyResult();
    auto config = createDefaultConfig();

    EXPECT_TRUE(renderer->renderRightEye(frame, result, config));
}

TEST_F(LensRendererTest, RenderRightEyeWhenNotDetected) {
    renderer->initialize();
    auto texture = createTestTexture(64, 64);
    renderer->loadTexture(texture.data(), 64, 64);

    auto frame = createTestFrame(640, 480);
    auto result = createLeftEyeOnlyResult(); // 오른쪽 눈 검출 안됨
    auto config = createDefaultConfig();

    EXPECT_FALSE(renderer->renderRightEye(frame, result, config));
}

TEST_F(LensRendererTest, RenderRightEyeWithoutInit) {
    auto frame = createTestFrame(640, 480);
    auto result = createValidIrisResult();
    auto config = createDefaultConfig();

    EXPECT_FALSE(renderer->renderRightEye(frame, result, config));
}

// ============================================================
// 6. LensConfig 테스트 - Opacity
// ============================================================

TEST_F(LensRendererTest, RenderWithZeroOpacity) {
    renderer->initialize();
    auto texture = createTestTexture(64, 64);
    renderer->loadTexture(texture.data(), 64, 64);

    auto frame = createTestFrame(640, 480);
    auto result = createValidIrisResult();
    auto config = createDefaultConfig();
    config.opacity = 0.0f;

    // 렌더링은 성공하지만 시각적으로는 보이지 않음
    EXPECT_TRUE(renderer->render(frame, result, config));
}

TEST_F(LensRendererTest, RenderWithFullOpacity) {
    renderer->initialize();
    auto texture = createTestTexture(64, 64);
    renderer->loadTexture(texture.data(), 64, 64);

    auto frame = createTestFrame(640, 480);
    auto result = createValidIrisResult();
    auto config = createDefaultConfig();
    config.opacity = 1.0f;

    EXPECT_TRUE(renderer->render(frame, result, config));
}

TEST_F(LensRendererTest, RenderWithPartialOpacity) {
    renderer->initialize();
    auto texture = createTestTexture(64, 64);
    renderer->loadTexture(texture.data(), 64, 64);

    auto frame = createTestFrame(640, 480);
    auto result = createValidIrisResult();
    auto config = createDefaultConfig();
    config.opacity = 0.5f;

    EXPECT_TRUE(renderer->render(frame, result, config));
}

// ============================================================
// 7. LensConfig 테스트 - Scale
// ============================================================

TEST_F(LensRendererTest, RenderWithDefaultScale) {
    renderer->initialize();
    auto texture = createTestTexture(64, 64);
    renderer->loadTexture(texture.data(), 64, 64);

    auto frame = createTestFrame(640, 480);
    auto result = createValidIrisResult();
    auto config = createDefaultConfig();
    config.scale = 1.0f;

    EXPECT_TRUE(renderer->render(frame, result, config));
}

TEST_F(LensRendererTest, RenderWithLargeScale) {
    renderer->initialize();
    auto texture = createTestTexture(64, 64);
    renderer->loadTexture(texture.data(), 64, 64);

    auto frame = createTestFrame(640, 480);
    auto result = createValidIrisResult();
    auto config = createDefaultConfig();
    config.scale = 2.0f;

    EXPECT_TRUE(renderer->render(frame, result, config));
}

TEST_F(LensRendererTest, RenderWithSmallScale) {
    renderer->initialize();
    auto texture = createTestTexture(64, 64);
    renderer->loadTexture(texture.data(), 64, 64);

    auto frame = createTestFrame(640, 480);
    auto result = createValidIrisResult();
    auto config = createDefaultConfig();
    config.scale = 0.5f;

    EXPECT_TRUE(renderer->render(frame, result, config));
}

TEST_F(LensRendererTest, RenderWithTinyScale) {
    renderer->initialize();
    auto texture = createTestTexture(64, 64);
    renderer->loadTexture(texture.data(), 64, 64);

    auto frame = createTestFrame(640, 480);
    auto result = createValidIrisResult();
    auto config = createDefaultConfig();
    config.scale = 0.1f; // 매우 작은 크기

    // 크기가 MIN_MASK_SIZE보다 작으면 렌더링 실패 가능
    renderer->render(frame, result, config);
}

// ============================================================
// 8. LensConfig 테스트 - Apply Left/Right
// ============================================================

TEST_F(LensRendererTest, RenderOnlyLeftEyeWithConfig) {
    renderer->initialize();
    auto texture = createTestTexture(64, 64);
    renderer->loadTexture(texture.data(), 64, 64);

    auto frame = createTestFrame(640, 480);
    auto result = createValidIrisResult();
    auto config = createDefaultConfig();
    config.apply_left = true;
    config.apply_right = false;

    EXPECT_TRUE(renderer->render(frame, result, config));
}

TEST_F(LensRendererTest, RenderOnlyRightEyeWithConfig) {
    renderer->initialize();
    auto texture = createTestTexture(64, 64);
    renderer->loadTexture(texture.data(), 64, 64);

    auto frame = createTestFrame(640, 480);
    auto result = createValidIrisResult();
    auto config = createDefaultConfig();
    config.apply_left = false;
    config.apply_right = true;

    EXPECT_TRUE(renderer->render(frame, result, config));
}

TEST_F(LensRendererTest, RenderNeitherEyeWithConfig) {
    renderer->initialize();
    auto texture = createTestTexture(64, 64);
    renderer->loadTexture(texture.data(), 64, 64);

    auto frame = createTestFrame(640, 480);
    auto result = createValidIrisResult();
    auto config = createDefaultConfig();
    config.apply_left = false;
    config.apply_right = false;

    // 양쪽 다 적용 안하면 실패
    EXPECT_FALSE(renderer->render(frame, result, config));
}

// ============================================================
// 9. LensConfig 테스트 - BlendMode
// ============================================================

TEST_F(LensRendererTest, RenderWithNormalBlend) {
    renderer->initialize();
    auto texture = createTestTexture(64, 64);
    renderer->loadTexture(texture.data(), 64, 64);

    auto frame = createTestFrame(640, 480);
    auto result = createValidIrisResult();
    auto config = createDefaultConfig();
    config.blend_mode = BlendMode::Normal;

    EXPECT_TRUE(renderer->render(frame, result, config));
}

TEST_F(LensRendererTest, RenderWithMultiplyBlend) {
    renderer->initialize();
    auto texture = createTestTexture(64, 64);
    renderer->loadTexture(texture.data(), 64, 64);

    auto frame = createTestFrame(640, 480);
    auto result = createValidIrisResult();
    auto config = createDefaultConfig();
    config.blend_mode = BlendMode::Multiply;

    EXPECT_TRUE(renderer->render(frame, result, config));
}

TEST_F(LensRendererTest, RenderWithScreenBlend) {
    renderer->initialize();
    auto texture = createTestTexture(64, 64);
    renderer->loadTexture(texture.data(), 64, 64);

    auto frame = createTestFrame(640, 480);
    auto result = createValidIrisResult();
    auto config = createDefaultConfig();
    config.blend_mode = BlendMode::Screen;

    EXPECT_TRUE(renderer->render(frame, result, config));
}

TEST_F(LensRendererTest, RenderWithOverlayBlend) {
    renderer->initialize();
    auto texture = createTestTexture(64, 64);
    renderer->loadTexture(texture.data(), 64, 64);

    auto frame = createTestFrame(640, 480);
    auto result = createValidIrisResult();
    auto config = createDefaultConfig();
    config.blend_mode = BlendMode::Overlay;

    EXPECT_TRUE(renderer->render(frame, result, config));
}

// ============================================================
// 10. LensConfig 테스트 - Edge Feather
// ============================================================

TEST_F(LensRendererTest, RenderWithNoFeathering) {
    renderer->initialize();
    auto texture = createTestTexture(64, 64);
    renderer->loadTexture(texture.data(), 64, 64);

    auto frame = createTestFrame(640, 480);
    auto result = createValidIrisResult();
    auto config = createDefaultConfig();
    config.edge_feather = 0.0f;

    EXPECT_TRUE(renderer->render(frame, result, config));
}

TEST_F(LensRendererTest, RenderWithFullFeathering) {
    renderer->initialize();
    auto texture = createTestTexture(64, 64);
    renderer->loadTexture(texture.data(), 64, 64);

    auto frame = createTestFrame(640, 480);
    auto result = createValidIrisResult();
    auto config = createDefaultConfig();
    config.edge_feather = 1.0f;

    EXPECT_TRUE(renderer->render(frame, result, config));
}

TEST_F(LensRendererTest, RenderWithPartialFeathering) {
    renderer->initialize();
    auto texture = createTestTexture(64, 64);
    renderer->loadTexture(texture.data(), 64, 64);

    auto frame = createTestFrame(640, 480);
    auto result = createValidIrisResult();
    auto config = createDefaultConfig();
    config.edge_feather = 0.3f;

    EXPECT_TRUE(renderer->render(frame, result, config));
}

// ============================================================
// 11. 경계 조건 테스트
// ============================================================

TEST_F(LensRendererTest, RenderWithIrisOutOfBounds) {
    renderer->initialize();
    auto texture = createTestTexture(64, 64);
    renderer->loadTexture(texture.data(), 64, 64);

    auto frame = createTestFrame(640, 480);
    auto result = createValidIrisResult();

    // 경계 밖 좌표
    result.left_iris[0] = {-0.1f, 0.5f, 0.0f, 1.0f};
    result.right_iris[0] = {1.1f, 0.5f, 0.0f, 1.0f};

    auto config = createDefaultConfig();

    // 경계 밖 좌표는 렌더링 실패
    EXPECT_FALSE(renderer->render(frame, result, config));
}

TEST_F(LensRendererTest, RenderWithTinyRadius) {
    renderer->initialize();
    auto texture = createTestTexture(64, 64);
    renderer->loadTexture(texture.data(), 64, 64);

    auto frame = createTestFrame(640, 480);
    auto result = createValidIrisResult();
    result.left_radius = 2.0f;  // 매우 작은 반지름
    result.right_radius = 2.0f;

    auto config = createDefaultConfig();

    // 반지름 검증은 calculateIrisRegion 내부에서 이루어짐
    // MIN_IRIS_RADIUS보다 작지만 랜드마크로 재계산되므로 결과가 다를 수 있음
    // 렌더링 시도는 하되, 실제 결과는 구현에 따라 다름
    renderer->render(frame, result, config);
}

TEST_F(LensRendererTest, RenderWithHugeRadius) {
    renderer->initialize();
    auto texture = createTestTexture(64, 64);
    renderer->loadTexture(texture.data(), 64, 64);

    auto frame = createTestFrame(640, 480);
    auto result = createValidIrisResult();
    result.left_radius = 250.0f;  // 매우 큰 반지름
    result.right_radius = 250.0f;

    auto config = createDefaultConfig();

    // 반지름 검증은 calculateIrisRegion 내부에서 이루어짐
    // MAX_IRIS_RADIUS보다 크지만 랜드마크로 재계산되므로 결과가 다를 수 있음
    renderer->render(frame, result, config);
}

TEST_F(LensRendererTest, RenderWithSmallFrame) {
    renderer->initialize();
    auto texture = createTestTexture(64, 64);
    renderer->loadTexture(texture.data(), 64, 64);

    auto frame = createTestFrame(100, 100);
    auto result = createValidIrisResult();
    result.frame_width = 100;
    result.frame_height = 100;

    auto config = createDefaultConfig();

    // 작은 프레임에서 정규화 좌표가 동일하면 픽셀 반지름이 작아져 MIN_MASK_SIZE 미달 가능
    // 결과는 구현에 따라 다를 수 있음
    renderer->render(frame, result, config);
}

TEST_F(LensRendererTest, RenderIrisAtEdgeOfFrame) {
    renderer->initialize();
    auto texture = createTestTexture(64, 64);
    renderer->loadTexture(texture.data(), 64, 64);

    auto frame = createTestFrame(640, 480);
    auto result = createValidIrisResult();

    // 프레임 가장자리에 위치
    result.left_iris[0] = {0.05f, 0.05f, 0.0f, 1.0f};
    result.left_iris[1] = {0.05f, 0.03f, 0.0f, 1.0f}; // top
    result.left_iris[2] = {0.05f, 0.07f, 0.0f, 1.0f}; // bottom
    result.left_iris[3] = {0.03f, 0.05f, 0.0f, 1.0f}; // left
    result.left_iris[4] = {0.07f, 0.05f, 0.0f, 1.0f}; // right

    result.right_iris[0] = {0.95f, 0.95f, 0.0f, 1.0f};
    result.right_iris[1] = {0.95f, 0.93f, 0.0f, 1.0f}; // top
    result.right_iris[2] = {0.95f, 0.97f, 0.0f, 1.0f}; // bottom
    result.right_iris[3] = {0.93f, 0.95f, 0.0f, 1.0f}; // left
    result.right_iris[4] = {0.97f, 0.95f, 0.0f, 1.0f}; // right

    auto config = createDefaultConfig();

    // 가장자리여도 렌더링 시도, 일부는 클리핑될 수 있음
    renderer->render(frame, result, config);
}

// ============================================================
// 12. 성능 테스트
// ============================================================

TEST_F(LensRendererTest, GetLastRenderTimeInitiallyZero) {
    renderer->initialize();

    EXPECT_DOUBLE_EQ(renderer->getLastRenderTimeMs(), 0.0);
}

TEST_F(LensRendererTest, GetLastRenderTimeAfterRender) {
    renderer->initialize();
    auto texture = createTestTexture(64, 64);
    renderer->loadTexture(texture.data(), 64, 64);

    auto frame = createTestFrame(640, 480);
    auto result = createValidIrisResult();
    auto config = createDefaultConfig();

    renderer->render(frame, result, config);

    double render_time = renderer->getLastRenderTimeMs();
    EXPECT_GT(render_time, 0.0);
    EXPECT_LT(render_time, 100.0); // 100ms 이내 (합리적인 상한)
}

TEST_F(LensRendererTest, RenderTimePerformanceTarget) {
    renderer->initialize();
    auto texture = createTestTexture(64, 64);
    renderer->loadTexture(texture.data(), 64, 64);

    auto frame = createTestFrame(640, 480);
    auto result = createValidIrisResult();
    auto config = createDefaultConfig();

    // 여러 번 렌더링하여 평균 시간 측정
    const int iterations = 10;
    double total_time = 0.0;

    for (int i = 0; i < iterations; ++i) {
        renderer->render(frame, result, config);
        total_time += renderer->getLastRenderTimeMs();
    }

    double avg_time = total_time / iterations;

    // 목표: 평균 10ms 이하 (30fps 이상 가능)
    EXPECT_LT(avg_time, 10.0) << "Average render time: " << avg_time << "ms";
}

TEST_F(LensRendererTest, GetLastRenderTimeAfterFailedRender) {
    renderer->initialize();

    auto frame = createTestFrame(640, 480);
    auto result = createUndetectedResult();
    auto config = createDefaultConfig();

    // 텍스처 로드 없이 렌더링 시도 (실패)
    EXPECT_FALSE(renderer->render(frame, result, config));

    // 실패한 렌더링이어도 시간은 0일 수 있음
    EXPECT_GE(renderer->getLastRenderTimeMs(), 0.0);
}

// ============================================================
// 13. 복합 시나리오 테스트
// ============================================================

TEST_F(LensRendererTest, SequentialRenderingSuccess) {
    renderer->initialize();
    auto texture = createTestTexture(64, 64);
    renderer->loadTexture(texture.data(), 64, 64);

    auto frame = createTestFrame(640, 480);
    auto result = createValidIrisResult();
    auto config = createDefaultConfig();

    // 연속 렌더링
    EXPECT_TRUE(renderer->render(frame, result, config));
    EXPECT_TRUE(renderer->render(frame, result, config));
    EXPECT_TRUE(renderer->render(frame, result, config));
}

TEST_F(LensRendererTest, ChangeTextureAndRender) {
    renderer->initialize();

    auto texture1 = createTestTexture(32, 32);
    auto texture2 = createTestTexture(64, 64);

    auto frame = createTestFrame(640, 480);
    auto result = createValidIrisResult();
    auto config = createDefaultConfig();

    // 첫 번째 텍스처로 렌더링
    renderer->loadTexture(texture1.data(), 32, 32);
    EXPECT_TRUE(renderer->render(frame, result, config));

    // 텍스처 변경 후 렌더링
    renderer->loadTexture(texture2.data(), 64, 64);
    EXPECT_TRUE(renderer->render(frame, result, config));
}

TEST_F(LensRendererTest, ReinitializeAndRender) {
    renderer->initialize();
    auto texture = createTestTexture(64, 64);
    renderer->loadTexture(texture.data(), 64, 64);

    auto frame = createTestFrame(640, 480);
    auto result = createValidIrisResult();
    auto config = createDefaultConfig();

    EXPECT_TRUE(renderer->render(frame, result, config));

    // 재초기화
    renderer->release();
    EXPECT_FALSE(renderer->hasTexture());

    renderer->initialize();
    renderer->loadTexture(texture.data(), 64, 64);

    EXPECT_TRUE(renderer->render(frame, result, config));
}

TEST_F(LensRendererTest, DifferentConfigSequence) {
    renderer->initialize();
    auto texture = createTestTexture(64, 64);
    renderer->loadTexture(texture.data(), 64, 64);

    auto frame = createTestFrame(640, 480);
    auto result = createValidIrisResult();

    // 다양한 설정으로 렌더링
    LensConfig config1 = createDefaultConfig();
    config1.opacity = 0.5f;
    config1.scale = 1.0f;
    EXPECT_TRUE(renderer->render(frame, result, config1));

    LensConfig config2 = createDefaultConfig();
    config2.opacity = 0.8f;
    config2.scale = 1.5f;
    config2.blend_mode = BlendMode::Multiply;
    EXPECT_TRUE(renderer->render(frame, result, config2));

    LensConfig config3 = createDefaultConfig();
    config3.opacity = 1.0f;
    config3.edge_feather = 0.3f;
    config3.blend_mode = BlendMode::Overlay;
    EXPECT_TRUE(renderer->render(frame, result, config3));
}

} // namespace testing
} // namespace iris_sdk
