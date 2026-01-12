/**
 * @file test_sdk_api.cpp
 * @brief IrisLensSDK C API 단위 테스트
 *
 * C API 래퍼의 기능 및 에러 처리를 검증합니다.
 */

#include <gtest/gtest.h>
#include <cstring>
#include <filesystem>

#include "iris_sdk/sdk_api.h"

namespace {

// 테스트용 모델 경로
const char* getTestModelPath() {
    // 환경에 따라 적절한 모델 경로 반환
    static const char* paths[] = {
        "models",
        "../models",
        "../../models",
        "../../../models"
    };

    for (const auto& path : paths) {
        if (std::filesystem::exists(path) && std::filesystem::is_directory(path)) {
            return path;
        }
    }
    return "models";  // 기본값
}

// 테스트 픽스처: 각 테스트 전후로 SDK 상태 정리
class SdkApiTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 이전 테스트 상태 정리
        iris_sdk_destroy();
    }

    void TearDown() override {
        // 테스트 후 SDK 정리
        iris_sdk_destroy();
    }
};

}  // namespace

// ============================================================================
// 에러 코드 테스트
// ============================================================================

TEST(SdkApiErrorTest, ErrorToStringReturnsValidStrings) {
    // 모든 에러 코드가 유효한 문자열을 반환해야 함
    EXPECT_STREQ("IRIS_SDK_OK", iris_sdk_error_to_string(IRIS_SDK_OK));
    EXPECT_STREQ("IRIS_SDK_NOT_INITIALIZED", iris_sdk_error_to_string(IRIS_SDK_NOT_INITIALIZED));
    EXPECT_STREQ("IRIS_SDK_ALREADY_INITIALIZED", iris_sdk_error_to_string(IRIS_SDK_ALREADY_INITIALIZED));
    EXPECT_STREQ("IRIS_SDK_MODEL_LOAD_FAILED", iris_sdk_error_to_string(IRIS_SDK_MODEL_LOAD_FAILED));
    EXPECT_STREQ("IRIS_SDK_INVALID_PATH", iris_sdk_error_to_string(IRIS_SDK_INVALID_PATH));
    EXPECT_STREQ("IRIS_SDK_INVALID_PARAM", iris_sdk_error_to_string(IRIS_SDK_INVALID_PARAM));
    EXPECT_STREQ("IRIS_SDK_NULL_POINTER", iris_sdk_error_to_string(IRIS_SDK_NULL_POINTER));
    EXPECT_STREQ("IRIS_SDK_INVALID_FORMAT", iris_sdk_error_to_string(IRIS_SDK_INVALID_FORMAT));
    EXPECT_STREQ("IRIS_SDK_DETECTION_FAILED", iris_sdk_error_to_string(IRIS_SDK_DETECTION_FAILED));
    EXPECT_STREQ("IRIS_SDK_NO_FACE", iris_sdk_error_to_string(IRIS_SDK_NO_FACE));
    EXPECT_STREQ("IRIS_SDK_RENDER_FAILED", iris_sdk_error_to_string(IRIS_SDK_RENDER_FAILED));
    EXPECT_STREQ("IRIS_SDK_NO_TEXTURE", iris_sdk_error_to_string(IRIS_SDK_NO_TEXTURE));
    EXPECT_STREQ("IRIS_SDK_UNKNOWN", iris_sdk_error_to_string(IRIS_SDK_UNKNOWN));
}

TEST(SdkApiErrorTest, UnknownErrorCodeReturnsUnknown) {
    // 정의되지 않은 에러 코드는 UNKNOWN 반환
    EXPECT_STREQ("IRIS_SDK_UNKNOWN", iris_sdk_error_to_string(static_cast<IrisSdkError>(12345)));
}

// ============================================================================
// 버전 및 정보 테스트
// ============================================================================

TEST(SdkApiInfoTest, GetVersionReturnsNonNull) {
    const char* version = iris_sdk_get_version();
    ASSERT_NE(nullptr, version);
    EXPECT_GT(std::strlen(version), 0u);
}

TEST(SdkApiInfoTest, GetBuildInfoReturnsNonNull) {
    const char* build_info = iris_sdk_get_build_info();
    ASSERT_NE(nullptr, build_info);
    EXPECT_GT(std::strlen(build_info), 0u);
}

TEST(SdkApiInfoTest, GetLastErrorReturnsEmptyWhenNoError) {
    // 초기 상태에서 에러 메시지는 비어있거나 이전 상태
    const char* error = iris_sdk_get_last_error();
    // 반환값은 NULL이 아님 (빈 문자열일 수 있음)
    ASSERT_NE(nullptr, error);
}

// ============================================================================
// 라이프사이클 테스트
// ============================================================================

TEST_F(SdkApiTest, InitWithNullPathReturnsError) {
    IrisSdkError err = iris_sdk_init(nullptr);
    EXPECT_EQ(IRIS_SDK_NULL_POINTER, err);
    EXPECT_FALSE(iris_sdk_is_ready());
}

TEST_F(SdkApiTest, InitWithConfigNullReturnsError) {
    IrisSdkError err = iris_sdk_init_with_config(nullptr);
    EXPECT_EQ(IRIS_SDK_NULL_POINTER, err);
    EXPECT_FALSE(iris_sdk_is_ready());
}

TEST_F(SdkApiTest, InitWithConfigNullPathReturnsError) {
    IrisSdkConfig config = {0};
    config.model_path = nullptr;

    IrisSdkError err = iris_sdk_init_with_config(&config);
    EXPECT_EQ(IRIS_SDK_NULL_POINTER, err);
    EXPECT_FALSE(iris_sdk_is_ready());
}

TEST_F(SdkApiTest, DestroyMultipleTimesIsSafe) {
    // 초기화 없이 여러 번 destroy 호출해도 안전
    iris_sdk_destroy();
    iris_sdk_destroy();
    iris_sdk_destroy();
    EXPECT_FALSE(iris_sdk_is_ready());
}

TEST_F(SdkApiTest, IsReadyReturnsFalseWhenNotInitialized) {
    EXPECT_FALSE(iris_sdk_is_ready());
}

#if defined(IRIS_SDK_HAS_TFLITE) && defined(IRIS_SDK_HAS_OPENCV)
// TFLite와 OpenCV가 있을 때만 실제 초기화 테스트

TEST_F(SdkApiTest, InitWithValidPathSucceeds) {
    const char* model_path = getTestModelPath();
    if (!std::filesystem::exists(model_path)) {
        GTEST_SKIP() << "Model path not found: " << model_path;
    }

    IrisSdkError err = iris_sdk_init(model_path);
    EXPECT_EQ(IRIS_SDK_OK, err);
    EXPECT_TRUE(iris_sdk_is_ready());
}

TEST_F(SdkApiTest, DoubleInitReturnsAlreadyInitialized) {
    const char* model_path = getTestModelPath();
    if (!std::filesystem::exists(model_path)) {
        GTEST_SKIP() << "Model path not found";
    }

    IrisSdkError err1 = iris_sdk_init(model_path);
    if (err1 != IRIS_SDK_OK) {
        GTEST_SKIP() << "Initial init failed";
    }

    IrisSdkError err2 = iris_sdk_init(model_path);
    EXPECT_EQ(IRIS_SDK_ALREADY_INITIALIZED, err2);
}

TEST_F(SdkApiTest, InitWithConfigSucceeds) {
    const char* model_path = getTestModelPath();
    if (!std::filesystem::exists(model_path)) {
        GTEST_SKIP() << "Model path not found";
    }

    IrisSdkConfig config = {0};
    config.model_path = model_path;
    config.min_confidence = 0.6f;
    config.max_faces = 1;
    config.enable_gpu = false;
    config.num_threads = 0;

    IrisSdkError err = iris_sdk_init_with_config(&config);
    EXPECT_EQ(IRIS_SDK_OK, err);
    EXPECT_TRUE(iris_sdk_is_ready());
}

TEST_F(SdkApiTest, ShutdownAfterInitSucceeds) {
    const char* model_path = getTestModelPath();
    if (!std::filesystem::exists(model_path)) {
        GTEST_SKIP() << "Model path not found";
    }

    IrisSdkError err = iris_sdk_init(model_path);
    if (err != IRIS_SDK_OK) {
        GTEST_SKIP() << "Init failed";
    }

    EXPECT_TRUE(iris_sdk_is_ready());

    iris_sdk_destroy();

    EXPECT_FALSE(iris_sdk_is_ready());
}

#endif  // IRIS_SDK_HAS_TFLITE && IRIS_SDK_HAS_OPENCV

// ============================================================================
// 검출 함수 테스트 (초기화 없이)
// ============================================================================

TEST_F(SdkApiTest, DetectWithoutInitReturnsNotInitialized) {
    uint8_t dummy_frame[640 * 480 * 4] = {0};
    IrisResult result = {0};

    IrisSdkError err = iris_sdk_detect(dummy_frame, 640, 480, IRIS_FORMAT_RGBA, &result);
    EXPECT_EQ(IRIS_SDK_NOT_INITIALIZED, err);
}

TEST_F(SdkApiTest, DetectWithNullFrameReturnsError) {
    IrisResult result = {0};

    IrisSdkError err = iris_sdk_detect(nullptr, 640, 480, IRIS_FORMAT_RGBA, &result);
    EXPECT_EQ(IRIS_SDK_NULL_POINTER, err);
}

TEST_F(SdkApiTest, DetectWithNullResultReturnsError) {
    uint8_t dummy_frame[640 * 480 * 4] = {0};

    IrisSdkError err = iris_sdk_detect(dummy_frame, 640, 480, IRIS_FORMAT_RGBA, nullptr);
    EXPECT_EQ(IRIS_SDK_NULL_POINTER, err);
}

TEST_F(SdkApiTest, DetectWithInvalidDimensionsReturnsError) {
    uint8_t dummy_frame[100] = {0};
    IrisResult result = {0};

    IrisSdkError err = iris_sdk_detect(dummy_frame, 0, 480, IRIS_FORMAT_RGBA, &result);
    EXPECT_EQ(IRIS_SDK_INVALID_PARAM, err);

    err = iris_sdk_detect(dummy_frame, 640, 0, IRIS_FORMAT_RGBA, &result);
    EXPECT_EQ(IRIS_SDK_INVALID_PARAM, err);

    err = iris_sdk_detect(dummy_frame, -1, 480, IRIS_FORMAT_RGBA, &result);
    EXPECT_EQ(IRIS_SDK_INVALID_PARAM, err);
}

// ============================================================================
// 처리 함수 테스트 (초기화 없이)
// ============================================================================

TEST_F(SdkApiTest, ProcessWithoutInitReturnsNotInitialized) {
    uint8_t dummy_frame[640 * 480 * 4] = {0};
    IrisLensConfig config;
    iris_sdk_default_lens_config(&config);
    IrisResult result = {0};

    IrisSdkError err = iris_sdk_process(dummy_frame, 640, 480, IRIS_FORMAT_RGBA, &config, &result);
    EXPECT_EQ(IRIS_SDK_NOT_INITIALIZED, err);
}

TEST_F(SdkApiTest, ProcessWithNullFrameReturnsError) {
    IrisLensConfig config;
    iris_sdk_default_lens_config(&config);
    IrisResult result = {0};

    IrisSdkError err = iris_sdk_process(nullptr, 640, 480, IRIS_FORMAT_RGBA, &config, &result);
    EXPECT_EQ(IRIS_SDK_NULL_POINTER, err);
}

// ============================================================================
// 렌더링 함수 테스트 (초기화 없이)
// ============================================================================

TEST_F(SdkApiTest, LoadTextureWithoutInitReturnsNotInitialized) {
    IrisSdkError err = iris_sdk_load_texture("test.png");
    EXPECT_EQ(IRIS_SDK_NOT_INITIALIZED, err);
}

TEST_F(SdkApiTest, LoadTextureWithNullPathReturnsError) {
    IrisSdkError err = iris_sdk_load_texture(nullptr);
    EXPECT_EQ(IRIS_SDK_NULL_POINTER, err);
}

TEST_F(SdkApiTest, LoadTextureFromMemoryWithNullDataReturnsError) {
    IrisSdkError err = iris_sdk_load_texture_from_memory(nullptr, 100, 100);
    EXPECT_EQ(IRIS_SDK_NULL_POINTER, err);
}

TEST_F(SdkApiTest, LoadTextureFromMemoryWithInvalidDimensionsReturnsError) {
    uint8_t dummy_data[100] = {0};

    IrisSdkError err = iris_sdk_load_texture_from_memory(dummy_data, 0, 100);
    EXPECT_EQ(IRIS_SDK_INVALID_PARAM, err);

    err = iris_sdk_load_texture_from_memory(dummy_data, 100, 0);
    EXPECT_EQ(IRIS_SDK_INVALID_PARAM, err);
}

TEST_F(SdkApiTest, RenderLensWithoutInitReturnsNotInitialized) {
    uint8_t dummy_frame[640 * 480 * 4] = {0};
    IrisResult iris_result = {0};
    IrisLensConfig config;
    iris_sdk_default_lens_config(&config);

    IrisSdkError err = iris_sdk_render_lens(
        dummy_frame, 640, 480, IRIS_FORMAT_RGBA, &iris_result, &config);
    EXPECT_EQ(IRIS_SDK_NOT_INITIALIZED, err);
}

TEST_F(SdkApiTest, RenderLensWithNullParamsReturnsError) {
    uint8_t dummy_frame[640 * 480 * 4] = {0};
    IrisResult iris_result = {0};
    IrisLensConfig config;
    iris_sdk_default_lens_config(&config);

    // null frame
    IrisSdkError err = iris_sdk_render_lens(
        nullptr, 640, 480, IRIS_FORMAT_RGBA, &iris_result, &config);
    EXPECT_EQ(IRIS_SDK_NULL_POINTER, err);

    // null iris_result
    err = iris_sdk_render_lens(
        dummy_frame, 640, 480, IRIS_FORMAT_RGBA, nullptr, &config);
    EXPECT_EQ(IRIS_SDK_NULL_POINTER, err);

    // null config
    err = iris_sdk_render_lens(
        dummy_frame, 640, 480, IRIS_FORMAT_RGBA, &iris_result, nullptr);
    EXPECT_EQ(IRIS_SDK_NULL_POINTER, err);
}

// ============================================================================
// 설정 함수 테스트
// ============================================================================

TEST_F(SdkApiTest, DefaultLensConfigSetsCorrectValues) {
    IrisLensConfig config = {0};
    iris_sdk_default_lens_config(&config);

    EXPECT_FLOAT_EQ(0.7f, config.opacity);
    EXPECT_FLOAT_EQ(1.0f, config.scale);
    EXPECT_FLOAT_EQ(0.0f, config.offset_x);
    EXPECT_FLOAT_EQ(0.0f, config.offset_y);
    EXPECT_EQ(IRIS_BLEND_NORMAL, config.blend_mode);
    EXPECT_FLOAT_EQ(0.1f, config.edge_feather);
    EXPECT_TRUE(config.apply_left);
    EXPECT_TRUE(config.apply_right);
}

TEST_F(SdkApiTest, DefaultLensConfigWithNullIsSafe) {
    // NULL 포인터로 호출해도 크래시 없음
    iris_sdk_default_lens_config(nullptr);
}

TEST_F(SdkApiTest, SetConfigWithoutInitReturnsNotInitialized) {
    IrisSdkError err = iris_sdk_set_config("min_confidence", "0.8");
    EXPECT_EQ(IRIS_SDK_NOT_INITIALIZED, err);
}

TEST_F(SdkApiTest, SetConfigWithNullKeyReturnsError) {
    IrisSdkError err = iris_sdk_set_config(nullptr, "0.8");
    EXPECT_EQ(IRIS_SDK_NULL_POINTER, err);
}

TEST_F(SdkApiTest, SetConfigWithNullValueReturnsError) {
    IrisSdkError err = iris_sdk_set_config("min_confidence", nullptr);
    EXPECT_EQ(IRIS_SDK_NULL_POINTER, err);
}

// ============================================================================
// 메모리 관리 테스트
// ============================================================================

TEST_F(SdkApiTest, FreeResultWithNullIsSafe) {
    // NULL 포인터로 호출해도 크래시 없음
    iris_sdk_free_result(nullptr);
}

TEST_F(SdkApiTest, FreeResultClearsStructure) {
    IrisResult result;
    result.detected = true;
    result.left_detected = true;
    result.right_detected = true;
    result.confidence = 0.95f;
    result.face_mesh_valid = true;

    iris_sdk_free_result(&result);

    EXPECT_FALSE(result.detected);
    EXPECT_FALSE(result.left_detected);
    EXPECT_FALSE(result.right_detected);
    EXPECT_FLOAT_EQ(0.0f, result.confidence);
    EXPECT_FALSE(result.face_mesh_valid);
}

// ============================================================================
// 구조체 크기 테스트 (ABI 안정성)
// ============================================================================

TEST(SdkApiStructTest, IrisLandmarkSize) {
    // 4 floats = 16 bytes
    EXPECT_EQ(sizeof(float) * 4, sizeof(IrisLandmark));
}

TEST(SdkApiStructTest, IrisRectSize) {
    // 4 floats = 16 bytes
    EXPECT_EQ(sizeof(float) * 4, sizeof(IrisRect));
}

TEST(SdkApiStructTest, IrisLensConfigHasExpectedLayout) {
    // 구조체 레이아웃이 예상대로인지 확인
    IrisLensConfig config = {0};
    EXPECT_EQ(offsetof(IrisLensConfig, opacity), 0u);
    // 기타 필드 오프셋은 컴파일러/플랫폼에 따라 다를 수 있음
}

// ============================================================================
// 프레임 포맷 테스트
// ============================================================================

TEST(SdkApiFormatTest, FrameFormatEnumValues) {
    // enum 값이 문서화된 대로인지 확인
    EXPECT_EQ(0, IRIS_FORMAT_RGBA);
    EXPECT_EQ(1, IRIS_FORMAT_BGRA);
    EXPECT_EQ(2, IRIS_FORMAT_RGB);
    EXPECT_EQ(3, IRIS_FORMAT_BGR);
    EXPECT_EQ(4, IRIS_FORMAT_NV21);
    EXPECT_EQ(5, IRIS_FORMAT_NV12);
    EXPECT_EQ(6, IRIS_FORMAT_GRAY);
}

TEST(SdkApiFormatTest, BlendModeEnumValues) {
    EXPECT_EQ(0, IRIS_BLEND_NORMAL);
    EXPECT_EQ(1, IRIS_BLEND_MULTIPLY);
    EXPECT_EQ(2, IRIS_BLEND_SCREEN);
    EXPECT_EQ(3, IRIS_BLEND_OVERLAY);
}

// ============================================================================
// 에러 코드 범위 테스트
// ============================================================================

TEST(SdkApiErrorRangeTest, InitializationErrorsIn100Range) {
    EXPECT_GE(IRIS_SDK_NOT_INITIALIZED, 100);
    EXPECT_LT(IRIS_SDK_NOT_INITIALIZED, 200);
    EXPECT_GE(IRIS_SDK_ALREADY_INITIALIZED, 100);
    EXPECT_LT(IRIS_SDK_ALREADY_INITIALIZED, 200);
    EXPECT_GE(IRIS_SDK_MODEL_LOAD_FAILED, 100);
    EXPECT_LT(IRIS_SDK_MODEL_LOAD_FAILED, 200);
    EXPECT_GE(IRIS_SDK_INVALID_PATH, 100);
    EXPECT_LT(IRIS_SDK_INVALID_PATH, 200);
}

TEST(SdkApiErrorRangeTest, ParameterErrorsIn200Range) {
    EXPECT_GE(IRIS_SDK_INVALID_PARAM, 200);
    EXPECT_LT(IRIS_SDK_INVALID_PARAM, 300);
    EXPECT_GE(IRIS_SDK_NULL_POINTER, 200);
    EXPECT_LT(IRIS_SDK_NULL_POINTER, 300);
    EXPECT_GE(IRIS_SDK_INVALID_FORMAT, 200);
    EXPECT_LT(IRIS_SDK_INVALID_FORMAT, 300);
}

TEST(SdkApiErrorRangeTest, DetectionErrorsIn300Range) {
    EXPECT_GE(IRIS_SDK_DETECTION_FAILED, 300);
    EXPECT_LT(IRIS_SDK_DETECTION_FAILED, 400);
    EXPECT_GE(IRIS_SDK_NO_FACE, 300);
    EXPECT_LT(IRIS_SDK_NO_FACE, 400);
}

TEST(SdkApiErrorRangeTest, RenderingErrorsIn400Range) {
    EXPECT_GE(IRIS_SDK_RENDER_FAILED, 400);
    EXPECT_LT(IRIS_SDK_RENDER_FAILED, 500);
    EXPECT_GE(IRIS_SDK_NO_TEXTURE, 400);
    EXPECT_LT(IRIS_SDK_NO_TEXTURE, 500);
}
