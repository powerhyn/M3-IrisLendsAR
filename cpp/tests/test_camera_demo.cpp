/**
 * @file test_camera_demo.cpp
 * @brief CameraDemo 기본 기능 테스트
 *
 * 인터랙티브 GUI 애플리케이션이므로 제한적 테스트만 수행.
 * 주요 검증은 수동 테스트 체크리스트로 진행.
 */

#include <gtest/gtest.h>
#include <filesystem>

#include "iris_sdk/frame_processor.h"
#include "iris_sdk/types.h"

namespace fs = std::filesystem;

/**
 * @brief CameraDemo 관련 기본 테스트
 */
class CameraDemoBasicTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 테스트용 모델 경로 설정
        const char* test_model_paths[] = {
            "./models",
            "../models",
            "../../shared/models",
            "../../../shared/models"
        };

        for (const char* path : test_model_paths) {
            if (fs::exists(path)) {
                model_path_ = fs::canonical(path).string();
                break;
            }
        }
    }

    std::string model_path_;
};

/**
 * @brief FrameProcessor 생성 테스트
 */
TEST_F(CameraDemoBasicTest, FrameProcessorConstruction) {
    iris_sdk::FrameProcessor processor;
    EXPECT_FALSE(processor.isInitialized());
}

/**
 * @brief 모델 경로 존재 확인
 */
TEST_F(CameraDemoBasicTest, ModelPathExists) {
    if (model_path_.empty()) {
        GTEST_SKIP() << "모델 경로를 찾을 수 없습니다";
    }

    EXPECT_TRUE(fs::exists(model_path_));

    // 필수 모델 파일 확인
    std::vector<std::string> required_models = {
        "face_detection_short_range.tflite",
        "face_landmark.tflite",
        "iris_landmark.tflite"
    };

    for (const auto& model : required_models) {
        fs::path model_file = fs::path(model_path_) / model;
        EXPECT_TRUE(fs::exists(model_file))
            << "모델 파일 없음: " << model_file;
    }
}

/**
 * @brief LensConfig 기본값 테스트
 */
TEST_F(CameraDemoBasicTest, LensConfigDefaults) {
    iris_sdk::LensConfig config;

    EXPECT_FLOAT_EQ(config.opacity, 0.7f);
    EXPECT_FLOAT_EQ(config.scale, 1.0f);
    EXPECT_FLOAT_EQ(config.offset_x, 0.0f);
    EXPECT_FLOAT_EQ(config.offset_y, 0.0f);
    EXPECT_EQ(config.blend_mode, iris_sdk::BlendMode::Normal);
    EXPECT_TRUE(config.apply_left);
    EXPECT_TRUE(config.apply_right);
}

/**
 * @brief LensConfig 범위 조절 테스트
 */
TEST_F(CameraDemoBasicTest, LensConfigRangeAdjustment) {
    iris_sdk::LensConfig config;

    // 투명도 조절 시뮬레이션 (+/- 키)
    config.opacity = std::clamp(config.opacity + 0.05f, 0.0f, 1.0f);
    EXPECT_NEAR(config.opacity, 0.75f, 0.01f);

    config.opacity = std::clamp(config.opacity - 0.1f, 0.0f, 1.0f);
    EXPECT_NEAR(config.opacity, 0.65f, 0.01f);

    // 경계값 테스트
    config.opacity = std::clamp(config.opacity + 1.0f, 0.0f, 1.0f);
    EXPECT_FLOAT_EQ(config.opacity, 1.0f);

    config.opacity = std::clamp(config.opacity - 2.0f, 0.0f, 1.0f);
    EXPECT_FLOAT_EQ(config.opacity, 0.0f);

    // 크기 조절 시뮬레이션 ([/] 키)
    config.scale = std::clamp(config.scale + 0.05f, 0.5f, 2.0f);
    EXPECT_NEAR(config.scale, 1.05f, 0.01f);

    config.scale = std::clamp(config.scale - 0.1f, 0.5f, 2.0f);
    EXPECT_NEAR(config.scale, 0.95f, 0.01f);
}

/**
 * @brief 텍스처 파일 존재 확인
 */
TEST_F(CameraDemoBasicTest, TextureFilesExist) {
    // 텍스처 경로 후보
    std::vector<std::string> texture_dirs = {
        "../../shared/test_data",
        "../../../shared/test_data",
        "./shared/test_data"
    };

    std::string texture_dir;
    for (const auto& dir : texture_dirs) {
        if (fs::exists(dir)) {
            texture_dir = fs::canonical(dir).string();
            break;
        }
    }

    if (texture_dir.empty()) {
        GTEST_SKIP() << "텍스처 디렉토리를 찾을 수 없습니다";
    }

    // 렌즈 텍스처 파일 검색
    int lens_texture_count = 0;
    for (const auto& entry : fs::directory_iterator(texture_dir)) {
        if (entry.path().extension() == ".png" &&
            entry.path().filename().string().find("lens_sample") != std::string::npos) {
            lens_texture_count++;
        }
    }

    EXPECT_GT(lens_texture_count, 0) << "렌즈 텍스처 파일이 없습니다";
}

/**
 * @brief IrisResult 초기 상태 테스트
 */
TEST_F(CameraDemoBasicTest, IrisResultInitialState) {
    iris_sdk::IrisResult result{};

    EXPECT_FALSE(result.detected);
    EXPECT_FALSE(result.left_detected);
    EXPECT_FALSE(result.right_detected);
    EXPECT_FLOAT_EQ(result.confidence, 0.0f);
}

/**
 * @brief 키 입력 매핑 테스트 (시뮬레이션)
 *
 * 실제 키 입력은 테스트 불가능하므로,
 * 키 코드 상수만 확인
 */
TEST_F(CameraDemoBasicTest, KeyCodeConstants) {
    // OpenCV 키 코드 상수 확인
    constexpr int KEY_ESC = 27;
    constexpr int KEY_SPACE = 32;
    constexpr int KEY_PLUS = '+';
    constexpr int KEY_MINUS = '-';
    constexpr int KEY_LEFT_BRACKET = '[';
    constexpr int KEY_RIGHT_BRACKET = ']';

    EXPECT_EQ(KEY_ESC, 27);
    EXPECT_EQ(KEY_SPACE, 32);
    EXPECT_EQ(KEY_PLUS, 43);
    EXPECT_EQ(KEY_MINUS, 45);
    EXPECT_EQ(KEY_LEFT_BRACKET, 91);
    EXPECT_EQ(KEY_RIGHT_BRACKET, 93);
}


//=============================================================================
// 수동 테스트 체크리스트 (TEST_F가 아닌 참조용)
//=============================================================================
// 아래 항목은 실제 GUI 환경에서 수동으로 검증해야 합니다.
//
// [ ] 웹캠 영상 정상 표시
// [ ] 홍채 검출 정상 동작 (랜드마크 표시)
// [ ] 렌즈 오버레이 정상 렌더링
// [ ] FPS 30 이상 유지
// [ ] ESC/Q 키로 종료
// [ ] SPACE 키로 렌즈 텍스처 순환
// [ ] L 키로 랜드마크 토글
// [ ] F 키로 FPS 토글
// [ ] D 키로 디버그 정보 토글
// [ ] +/- 키로 투명도 조절
// [ ] [/] 키로 크기 조절
// [ ] E 키로 렌즈 효과 ON/OFF
// [ ] S 키로 스크린샷 저장
// [ ] R 키로 설정 초기화
// [ ] H 키로 도움말 출력
//
// 성능 검증:
// [ ] 정면 얼굴에서 안정적 검출
// [ ] 좌/우 회전 (30도 내) 정상 동작
// [ ] 조명 변화에 적응
// [ ] 눈 깜빡임 시 안정성
// [ ] 다양한 거리에서 동작
//=============================================================================

