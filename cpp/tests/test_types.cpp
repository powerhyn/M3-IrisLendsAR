/**
 * @file test_types.cpp
 * @brief TDD RED Phase: types.h 데이터 구조체 테스트
 *
 * 이 테스트들은 types.h 구현 전에 작성됨.
 * 모든 테스트는 현재 실패해야 함 (RED phase).
 */

#include <gtest/gtest.h>
#include <type_traits>
#include "iris_sdk/types.h"

namespace iris_sdk {
namespace testing {

// ============================================================
// IrisLandmark 테스트
// ============================================================

class IrisLandmarkTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(IrisLandmarkTest, DefaultConstruction) {
    // 기본 생성 시 모든 값 0으로 초기화
    IrisLandmark landmark{};
    EXPECT_FLOAT_EQ(landmark.x, 0.0f);
    EXPECT_FLOAT_EQ(landmark.y, 0.0f);
    EXPECT_FLOAT_EQ(landmark.z, 0.0f);
    EXPECT_FLOAT_EQ(landmark.visibility, 0.0f);
}

TEST_F(IrisLandmarkTest, ValueAssignment) {
    // 값 할당 및 조회
    IrisLandmark landmark;
    landmark.x = 0.5f;
    landmark.y = 0.3f;
    landmark.z = 0.1f;
    landmark.visibility = 0.9f;

    EXPECT_FLOAT_EQ(landmark.x, 0.5f);
    EXPECT_FLOAT_EQ(landmark.y, 0.3f);
    EXPECT_FLOAT_EQ(landmark.z, 0.1f);
    EXPECT_FLOAT_EQ(landmark.visibility, 0.9f);
}

TEST_F(IrisLandmarkTest, AggregateInitialization) {
    // 집합체 초기화 지원
    IrisLandmark landmark = {0.1f, 0.2f, 0.3f, 0.8f};
    EXPECT_FLOAT_EQ(landmark.x, 0.1f);
    EXPECT_FLOAT_EQ(landmark.y, 0.2f);
    EXPECT_FLOAT_EQ(landmark.z, 0.3f);
    EXPECT_FLOAT_EQ(landmark.visibility, 0.8f);
}

TEST_F(IrisLandmarkTest, TriviallyCopyable) {
    // POD 타입 검증 - FFI 호환성 필수
    EXPECT_TRUE(std::is_trivially_copyable<IrisLandmark>::value);
    EXPECT_TRUE(std::is_standard_layout<IrisLandmark>::value);
}

TEST_F(IrisLandmarkTest, SizeVerification) {
    // 메모리 크기 검증 (4 floats = 16 bytes)
    EXPECT_EQ(sizeof(IrisLandmark), 16u);
}

TEST_F(IrisLandmarkTest, BoundaryValues) {
    // 경계값 테스트
    IrisLandmark landmark;

    // 최소값
    landmark.x = 0.0f;
    landmark.y = 0.0f;
    EXPECT_FLOAT_EQ(landmark.x, 0.0f);
    EXPECT_FLOAT_EQ(landmark.y, 0.0f);

    // 최대 정규화 값
    landmark.x = 1.0f;
    landmark.y = 1.0f;
    EXPECT_FLOAT_EQ(landmark.x, 1.0f);
    EXPECT_FLOAT_EQ(landmark.y, 1.0f);

    // 음수 (검출기에서 에러 상황)
    landmark.x = -0.1f;
    EXPECT_FLOAT_EQ(landmark.x, -0.1f);

    // 범위 초과
    landmark.x = 1.5f;
    EXPECT_FLOAT_EQ(landmark.x, 1.5f);
}

// ============================================================
// Rect 테스트
// ============================================================

class RectTest : public ::testing::Test {};

TEST_F(RectTest, DefaultConstruction) {
    Rect rect{};
    EXPECT_FLOAT_EQ(rect.x, 0.0f);
    EXPECT_FLOAT_EQ(rect.y, 0.0f);
    EXPECT_FLOAT_EQ(rect.width, 0.0f);
    EXPECT_FLOAT_EQ(rect.height, 0.0f);
}

TEST_F(RectTest, ValueAssignment) {
    Rect rect;
    rect.x = 10.0f;
    rect.y = 20.0f;
    rect.width = 100.0f;
    rect.height = 80.0f;

    EXPECT_FLOAT_EQ(rect.x, 10.0f);
    EXPECT_FLOAT_EQ(rect.y, 20.0f);
    EXPECT_FLOAT_EQ(rect.width, 100.0f);
    EXPECT_FLOAT_EQ(rect.height, 80.0f);
}

TEST_F(RectTest, AggregateInitialization) {
    Rect rect = {5.0f, 10.0f, 50.0f, 60.0f};
    EXPECT_FLOAT_EQ(rect.x, 5.0f);
    EXPECT_FLOAT_EQ(rect.y, 10.0f);
    EXPECT_FLOAT_EQ(rect.width, 50.0f);
    EXPECT_FLOAT_EQ(rect.height, 60.0f);
}

TEST_F(RectTest, TriviallyCopyable) {
    EXPECT_TRUE(std::is_trivially_copyable<Rect>::value);
    EXPECT_TRUE(std::is_standard_layout<Rect>::value);
}

TEST_F(RectTest, SizeVerification) {
    EXPECT_EQ(sizeof(Rect), 16u);
}

// ============================================================
// IrisResult 테스트
// ============================================================

class IrisResultTest : public ::testing::Test {};

TEST_F(IrisResultTest, DefaultConstruction) {
    IrisResult result{};

    // 검출 안됨 상태
    EXPECT_FALSE(result.detected);
    EXPECT_FALSE(result.left_detected);
    EXPECT_FALSE(result.right_detected);
    EXPECT_FLOAT_EQ(result.confidence, 0.0f);
}

TEST_F(IrisResultTest, LeftEyeDetection) {
    IrisResult result{};
    result.detected = true;
    result.left_detected = true;
    result.left_iris[0] = {0.3f, 0.4f, 0.0f, 0.95f};  // center
    result.left_radius = 15.0f;

    EXPECT_TRUE(result.left_detected);
    EXPECT_FLOAT_EQ(result.left_iris[0].x, 0.3f);
    EXPECT_FLOAT_EQ(result.left_iris[0].y, 0.4f);
    EXPECT_FLOAT_EQ(result.left_radius, 15.0f);
}

TEST_F(IrisResultTest, RightEyeDetection) {
    IrisResult result{};
    result.detected = true;
    result.right_detected = true;
    result.right_iris[0] = {0.7f, 0.4f, 0.0f, 0.92f};  // center
    result.right_radius = 14.5f;

    EXPECT_TRUE(result.right_detected);
    EXPECT_FLOAT_EQ(result.right_iris[0].x, 0.7f);
    EXPECT_FLOAT_EQ(result.right_iris[0].y, 0.4f);
    EXPECT_FLOAT_EQ(result.right_radius, 14.5f);
}

TEST_F(IrisResultTest, BothEyesDetection) {
    IrisResult result{};
    result.detected = true;
    result.confidence = 0.95f;
    result.left_detected = true;
    result.right_detected = true;

    result.left_iris[0] = {0.3f, 0.4f, 0.0f, 0.95f};
    result.right_iris[0] = {0.7f, 0.4f, 0.0f, 0.92f};

    EXPECT_TRUE(result.detected);
    EXPECT_TRUE(result.left_detected);
    EXPECT_TRUE(result.right_detected);
    EXPECT_FLOAT_EQ(result.confidence, 0.95f);
}

TEST_F(IrisResultTest, FaceRectStorage) {
    IrisResult result{};
    result.face_rect = {100.0f, 50.0f, 200.0f, 250.0f};

    EXPECT_FLOAT_EQ(result.face_rect.x, 100.0f);
    EXPECT_FLOAT_EQ(result.face_rect.y, 50.0f);
    EXPECT_FLOAT_EQ(result.face_rect.width, 200.0f);
    EXPECT_FLOAT_EQ(result.face_rect.height, 250.0f);
}

TEST_F(IrisResultTest, FaceRotationStorage) {
    IrisResult result{};
    result.face_rotation[0] = 5.0f;   // pitch
    result.face_rotation[1] = -10.0f; // yaw
    result.face_rotation[2] = 2.0f;   // roll

    EXPECT_FLOAT_EQ(result.face_rotation[0], 5.0f);
    EXPECT_FLOAT_EQ(result.face_rotation[1], -10.0f);
    EXPECT_FLOAT_EQ(result.face_rotation[2], 2.0f);
}

TEST_F(IrisResultTest, TimestampStorage) {
    IrisResult result{};
    result.timestamp_ms = 1234567890123LL;

    EXPECT_EQ(result.timestamp_ms, 1234567890123LL);
}

TEST_F(IrisResultTest, FrameDimensionsStorage) {
    IrisResult result{};
    result.frame_width = 1920;
    result.frame_height = 1080;

    EXPECT_EQ(result.frame_width, 1920);
    EXPECT_EQ(result.frame_height, 1080);
}

TEST_F(IrisResultTest, IrisLandmarksArray) {
    IrisResult result{};

    // 5개 랜드마크: center + 4 boundary points
    for (int i = 0; i < 5; ++i) {
        result.left_iris[i] = {0.1f * i, 0.2f * i, 0.0f, 0.9f};
        result.right_iris[i] = {0.5f + 0.1f * i, 0.2f * i, 0.0f, 0.9f};
    }

    EXPECT_FLOAT_EQ(result.left_iris[2].x, 0.2f);
    EXPECT_FLOAT_EQ(result.right_iris[3].x, 0.8f);
}

TEST_F(IrisResultTest, TriviallyCopyable) {
    EXPECT_TRUE(std::is_trivially_copyable<IrisResult>::value);
    EXPECT_TRUE(std::is_standard_layout<IrisResult>::value);
}

// ============================================================
// LensConfig 테스트
// ============================================================

class LensConfigTest : public ::testing::Test {};

TEST_F(LensConfigTest, DefaultValues) {
    LensConfig config{};

    // 기본값 검증
    EXPECT_FLOAT_EQ(config.opacity, 0.7f);
    EXPECT_FLOAT_EQ(config.scale, 1.0f);
    EXPECT_FLOAT_EQ(config.offset_x, 0.0f);
    EXPECT_FLOAT_EQ(config.offset_y, 0.0f);
    EXPECT_EQ(config.blend_mode, BlendMode::Normal);
    EXPECT_FLOAT_EQ(config.edge_feather, 0.1f);
    EXPECT_TRUE(config.apply_left);
    EXPECT_TRUE(config.apply_right);
}

TEST_F(LensConfigTest, CustomValues) {
    LensConfig config;
    config.opacity = 0.5f;
    config.scale = 1.2f;
    config.offset_x = 0.01f;
    config.offset_y = -0.02f;
    config.blend_mode = BlendMode::Multiply;
    config.edge_feather = 0.2f;
    config.apply_left = true;
    config.apply_right = false;

    EXPECT_FLOAT_EQ(config.opacity, 0.5f);
    EXPECT_FLOAT_EQ(config.scale, 1.2f);
    EXPECT_FLOAT_EQ(config.offset_x, 0.01f);
    EXPECT_FLOAT_EQ(config.offset_y, -0.02f);
    EXPECT_EQ(config.blend_mode, BlendMode::Multiply);
    EXPECT_FLOAT_EQ(config.edge_feather, 0.2f);
    EXPECT_TRUE(config.apply_left);
    EXPECT_FALSE(config.apply_right);
}

TEST_F(LensConfigTest, OpacityBoundary) {
    LensConfig config;

    // 최소값
    config.opacity = 0.0f;
    EXPECT_FLOAT_EQ(config.opacity, 0.0f);

    // 최대값
    config.opacity = 1.0f;
    EXPECT_FLOAT_EQ(config.opacity, 1.0f);
}

TEST_F(LensConfigTest, TriviallyCopyable) {
    EXPECT_TRUE(std::is_trivially_copyable<LensConfig>::value);
    EXPECT_TRUE(std::is_standard_layout<LensConfig>::value);
}

// ============================================================
// BlendMode 열거형 테스트
// ============================================================

class BlendModeTest : public ::testing::Test {};

TEST_F(BlendModeTest, EnumValues) {
    EXPECT_EQ(static_cast<int>(BlendMode::Normal), 0);
    EXPECT_EQ(static_cast<int>(BlendMode::Multiply), 1);
    EXPECT_EQ(static_cast<int>(BlendMode::Screen), 2);
    EXPECT_EQ(static_cast<int>(BlendMode::Overlay), 3);
}

// ============================================================
// FrameFormat 열거형 테스트
// ============================================================

class FrameFormatTest : public ::testing::Test {};

TEST_F(FrameFormatTest, EnumValues) {
    EXPECT_EQ(static_cast<int>(FrameFormat::RGBA), 0);
    EXPECT_EQ(static_cast<int>(FrameFormat::BGRA), 1);
    EXPECT_EQ(static_cast<int>(FrameFormat::RGB), 2);
    EXPECT_EQ(static_cast<int>(FrameFormat::BGR), 3);
    EXPECT_EQ(static_cast<int>(FrameFormat::NV21), 4);
    EXPECT_EQ(static_cast<int>(FrameFormat::NV12), 5);
    EXPECT_EQ(static_cast<int>(FrameFormat::Grayscale), 6);
}

// ============================================================
// ErrorCode 열거형 테스트
// ============================================================

class ErrorCodeTest : public ::testing::Test {};

TEST_F(ErrorCodeTest, SuccessCode) {
    EXPECT_EQ(static_cast<int>(ErrorCode::Success), 0);
}

TEST_F(ErrorCodeTest, InitializationErrors) {
    // 100번대: 초기화 에러
    EXPECT_EQ(static_cast<int>(ErrorCode::NotInitialized), 100);
    EXPECT_EQ(static_cast<int>(ErrorCode::AlreadyInitialized), 101);
    EXPECT_EQ(static_cast<int>(ErrorCode::ModelLoadFailed), 102);
    EXPECT_EQ(static_cast<int>(ErrorCode::InvalidPath), 103);
}

TEST_F(ErrorCodeTest, ParameterErrors) {
    // 200번대: 파라미터 에러
    EXPECT_EQ(static_cast<int>(ErrorCode::InvalidParameter), 200);
    EXPECT_EQ(static_cast<int>(ErrorCode::NullPointer), 201);
    EXPECT_EQ(static_cast<int>(ErrorCode::FrameFormatUnsupported), 202);
}

TEST_F(ErrorCodeTest, DetectionErrors) {
    // 300번대: 검출 에러
    EXPECT_EQ(static_cast<int>(ErrorCode::DetectionFailed), 300);
    EXPECT_EQ(static_cast<int>(ErrorCode::NoFaceDetected), 301);
}

TEST_F(ErrorCodeTest, RenderingErrors) {
    // 400번대: 렌더링 에러
    EXPECT_EQ(static_cast<int>(ErrorCode::RenderFailed), 400);
    EXPECT_EQ(static_cast<int>(ErrorCode::NoTextureLoaded), 401);
}

TEST_F(ErrorCodeTest, GeneralErrors) {
    EXPECT_EQ(static_cast<int>(ErrorCode::Unknown), 999);
}

// ============================================================
// DetectorType 열거형 테스트
// ============================================================

class DetectorTypeTest : public ::testing::Test {};

TEST_F(DetectorTypeTest, EnumValues) {
    EXPECT_EQ(static_cast<int>(DetectorType::Unknown), 0);
    EXPECT_EQ(static_cast<int>(DetectorType::MediaPipe), 1);
    EXPECT_EQ(static_cast<int>(DetectorType::EyeOnly), 2);
    EXPECT_EQ(static_cast<int>(DetectorType::Hybrid), 3);
}

} // namespace testing
} // namespace iris_sdk

// Main entry point
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
