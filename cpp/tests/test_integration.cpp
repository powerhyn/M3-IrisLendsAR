/**
 * @file test_integration.cpp
 * @brief IrisLensSDK C API 통합 테스트 (P1-W4-06)
 *
 * 전체 SDK 파이프라인을 검증하는 통합 테스트입니다.
 * C API 래퍼를 통해 init → detect → render → destroy 흐름을 테스트합니다.
 *
 * 테스트 범위:
 * - 전체 파이프라인 흐름
 * - 다양한 프레임 포맷 지원
 * - 연속 처리 안정성 (1000+ 프레임)
 * - 에러 핸들링
 *
 * 필수 조건:
 * - IRIS_SDK_HAS_TFLITE: TensorFlow Lite 라이브러리
 * - IRIS_SDK_HAS_OPENCV: OpenCV 라이브러리
 * - 모델 파일: shared/models/ (*.tflite)
 * - 테스트 이미지: shared/test_data/ (*.png)
 */

#include <gtest/gtest.h>
#include <cstring>
#include <filesystem>
#include <vector>

#include "iris_sdk/sdk_api.h"

namespace fs = std::filesystem;

// ============================================================================
// 테스트 설정 상수
// ============================================================================
namespace {

/// 모델 경로를 찾기 위한 상대 경로 후보들
const std::vector<std::string> MODEL_PATH_CANDIDATES = {
    "models",
    "../models",
    "../../models",
    "../../../models",
    "shared/models",
    "../shared/models",
    "../../shared/models",
    "../../../shared/models"
};

/// 테스트 데이터 경로를 찾기 위한 상대 경로 후보들
const std::vector<std::string> TEST_DATA_PATH_CANDIDATES = {
    "shared/test_data",
    "../shared/test_data",
    "../../shared/test_data",
    "../../../shared/test_data"
};

/// 텍스처 경로를 찾기 위한 상대 경로 후보들
const std::vector<std::string> TEXTURE_PATH_CANDIDATES = {
    "shared/test_data",
    "../shared/test_data",
    "../../shared/test_data",
    "../../../shared/test_data"
};

/// 테스트 이미지 파일명
const std::vector<std::string> TEST_IMAGE_FILES = {
    "iris_test_01.png",
    "iris_test_02.png",
    "iris_test_03.png",
    "iris_test_04.png"
};

/// 텍스처 파일명
const char* TEXTURE_FILE = "lens_sample_type_alpha_01.png";

/// 연속 처리 테스트 프레임 수
constexpr int CONTINUOUS_PROCESSING_FRAMES = 1000;

/// 성능 목표 - 30fps를 위한 최대 지연 시간 (ms)
constexpr double TARGET_LATENCY_MS = 33.0;

/// 메모리 안정성 검증을 위한 반복 횟수
constexpr int MEMORY_STABILITY_ITERATIONS = 100;

}  // namespace

// ============================================================================
// 유틸리티 함수
// ============================================================================

/**
 * @brief 모델 디렉토리 경로 찾기
 * @return 찾은 모델 경로 또는 빈 문자열
 */
std::string findModelPath() {
    for (const auto& path : MODEL_PATH_CANDIDATES) {
        if (fs::exists(path) && fs::is_directory(path)) {
            // 필수 모델 파일 확인
            if (fs::exists(fs::path(path) / "face_detection_short_range.tflite") &&
                fs::exists(fs::path(path) / "face_landmark.tflite") &&
                fs::exists(fs::path(path) / "iris_landmark.tflite")) {
                return fs::canonical(path).string();
            }
        }
    }
    return "";
}

/**
 * @brief 테스트 데이터 디렉토리 경로 찾기
 * @return 찾은 테스트 데이터 경로 또는 빈 문자열
 */
std::string findTestDataPath() {
    for (const auto& path : TEST_DATA_PATH_CANDIDATES) {
        if (fs::exists(path) && fs::is_directory(path)) {
            return fs::canonical(path).string();
        }
    }
    return "";
}

/**
 * @brief 테스트 이미지 경로 반환
 * @param index 테스트 이미지 인덱스
 * @return 이미지 경로 또는 빈 문자열
 */
std::string getTestImagePath(int index = 0) {
    std::string test_data_path = findTestDataPath();
    if (test_data_path.empty()) {
        return "";
    }

    if (index < 0 || index >= static_cast<int>(TEST_IMAGE_FILES.size())) {
        index = 0;
    }

    fs::path image_path = fs::path(test_data_path) / TEST_IMAGE_FILES[index];
    if (fs::exists(image_path)) {
        return image_path.string();
    }
    return "";
}

/**
 * @brief 텍스처 파일 경로 반환
 * @return 텍스처 경로 또는 빈 문자열
 */
std::string getTexturePath() {
    for (const auto& path : TEXTURE_PATH_CANDIDATES) {
        fs::path texture_path = fs::path(path) / TEXTURE_FILE;
        if (fs::exists(texture_path)) {
            return fs::canonical(texture_path).string();
        }
    }
    return "";
}

/**
 * @brief 출력 디렉토리 생성 및 경로 반환
 * @return 출력 디렉토리 경로
 */
std::string getOutputPath() {
    std::string test_data_path = findTestDataPath();
    if (test_data_path.empty()) {
        return "test_output";
    }

    fs::path output_path = fs::path(test_data_path) / "output";
    fs::create_directories(output_path);
    return output_path.string();
}

// ============================================================================
// 통합 테스트 Fixture
// ============================================================================

#if defined(IRIS_SDK_HAS_TFLITE) && defined(IRIS_SDK_HAS_OPENCV)

#include <chrono>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

/**
 * @brief SDK 통합 테스트 Fixture
 *
 * 각 테스트 전후로 SDK 상태를 정리하여 독립적인 테스트 환경을 보장합니다.
 */
class IntegrationTest : public ::testing::Test {
protected:
    std::string model_path_;
    std::string test_data_path_;
    std::string texture_path_;
    std::string output_path_;
    std::vector<cv::Mat> test_images_;
    bool setup_success_ = false;

    void SetUp() override {
        // SDK 초기 상태 정리
        iris_sdk_destroy();

        // 경로 설정
        model_path_ = findModelPath();
        test_data_path_ = findTestDataPath();
        texture_path_ = getTexturePath();
        output_path_ = getOutputPath();

        if (model_path_.empty()) {
            std::cerr << "Warning: Model path not found. Skipping integration tests." << std::endl;
            return;
        }

        if (test_data_path_.empty()) {
            std::cerr << "Warning: Test data path not found. Skipping integration tests." << std::endl;
            return;
        }

        // 테스트 이미지 로드
        loadTestImages();

        if (test_images_.empty()) {
            std::cerr << "Warning: No test images loaded. Skipping integration tests." << std::endl;
            return;
        }

        std::cout << "=== Integration Test Setup ===" << std::endl;
        std::cout << "Model path: " << model_path_ << std::endl;
        std::cout << "Test data path: " << test_data_path_ << std::endl;
        std::cout << "Texture path: " << (texture_path_.empty() ? "(not found)" : texture_path_) << std::endl;
        std::cout << "Output path: " << output_path_ << std::endl;
        std::cout << "Test images loaded: " << test_images_.size() << std::endl;

        setup_success_ = true;
    }

    void TearDown() override {
        // SDK 정리
        iris_sdk_destroy();
        test_images_.clear();
    }

    /**
     * @brief 테스트 이미지 로드
     */
    void loadTestImages() {
        for (const auto& filename : TEST_IMAGE_FILES) {
            fs::path image_path = fs::path(test_data_path_) / filename;
            if (fs::exists(image_path)) {
                cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
                if (!image.empty()) {
                    test_images_.push_back(image);
                }
            }
        }
    }

    /**
     * @brief SDK 초기화 헬퍼
     * @return 초기화 성공 여부
     */
    bool initializeSDK() {
        IrisSdkError err = iris_sdk_init(model_path_.c_str());
        return err == IRIS_SDK_OK;
    }

    /**
     * @brief 이미지를 지정된 포맷으로 변환
     */
    cv::Mat convertToFormat(const cv::Mat& src, IrisFrameFormat format) {
        cv::Mat result;

        switch (format) {
            case IRIS_FORMAT_RGB:
                cv::cvtColor(src, result, cv::COLOR_BGR2RGB);
                break;
            case IRIS_FORMAT_BGR:
                result = src.clone();
                break;
            case IRIS_FORMAT_RGBA:
                cv::cvtColor(src, result, cv::COLOR_BGR2RGBA);
                break;
            case IRIS_FORMAT_BGRA:
                cv::cvtColor(src, result, cv::COLOR_BGR2BGRA);
                break;
            case IRIS_FORMAT_GRAY:
                cv::cvtColor(src, result, cv::COLOR_BGR2GRAY);
                break;
            default:
                result = src.clone();
                break;
        }

        return result;
    }

    /**
     * @brief 지연 시간 통계 계산
     */
    struct LatencyStats {
        double min_ms = 0.0;
        double max_ms = 0.0;
        double avg_ms = 0.0;
        double median_ms = 0.0;
        double p95_ms = 0.0;
        int sample_count = 0;

        static LatencyStats calculate(std::vector<double> times) {
            LatencyStats stats;
            if (times.empty()) return stats;

            stats.sample_count = static_cast<int>(times.size());
            std::sort(times.begin(), times.end());

            stats.min_ms = times.front();
            stats.max_ms = times.back();
            stats.avg_ms = std::accumulate(times.begin(), times.end(), 0.0) / times.size();

            size_t mid = times.size() / 2;
            stats.median_ms = (times.size() % 2 == 0)
                ? (times[mid - 1] + times[mid]) / 2.0
                : times[mid];

            size_t p95_idx = static_cast<size_t>(times.size() * 0.95);
            if (p95_idx >= times.size()) p95_idx = times.size() - 1;
            stats.p95_ms = times[p95_idx];

            return stats;
        }

        void print(const std::string& label) const {
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "\n=== " << label << " ===" << std::endl;
            std::cout << "  Samples:  " << sample_count << std::endl;
            std::cout << "  Min:      " << min_ms << " ms" << std::endl;
            std::cout << "  Max:      " << max_ms << " ms" << std::endl;
            std::cout << "  Avg:      " << avg_ms << " ms" << std::endl;
            std::cout << "  Median:   " << median_ms << " ms" << std::endl;
            std::cout << "  P95:      " << p95_ms << " ms" << std::endl;
        }
    };
};

// ============================================================================
// 기본 C API 흐름 테스트
// ============================================================================

/**
 * @brief 버전 및 빌드 정보 검증
 */
TEST_F(IntegrationTest, VersionAndBuildInfo) {
    if (!setup_success_) {
        GTEST_SKIP() << "Setup failed - models or images not available";
    }

    // 버전 정보 확인
    const char* version = iris_sdk_get_version();
    ASSERT_NE(nullptr, version);
    EXPECT_GT(std::strlen(version), 0u);
    std::cout << "SDK Version: " << version << std::endl;

    // 빌드 정보 확인
    const char* build_info = iris_sdk_get_build_info();
    ASSERT_NE(nullptr, build_info);
    EXPECT_GT(std::strlen(build_info), 0u);
    std::cout << "Build Info: " << build_info << std::endl;
}

/**
 * @brief 전체 C API 흐름 테스트 (init → detect → render → destroy)
 */
TEST_F(IntegrationTest, CAPIFullFlow) {
    if (!setup_success_) {
        GTEST_SKIP() << "Setup failed - models or images not available";
    }

    std::cout << "\n=== C API Full Flow Test ===" << std::endl;

    // 1. 버전 확인 (초기화 전에도 가능)
    const char* version = iris_sdk_get_version();
    ASSERT_NE(nullptr, version);
    std::cout << "1. Version check: " << version << std::endl;

    // 2. 초기화 전 ready 상태 확인
    EXPECT_FALSE(iris_sdk_is_ready());
    std::cout << "2. Pre-init ready check: false (expected)" << std::endl;

    // 3. SDK 초기화
    IrisSdkError init_err = iris_sdk_init(model_path_.c_str());
    ASSERT_EQ(IRIS_SDK_OK, init_err) << "SDK init failed with error: "
        << iris_sdk_error_to_string(init_err);
    std::cout << "3. SDK initialized successfully" << std::endl;

    // 4. Ready 상태 확인
    EXPECT_TRUE(iris_sdk_is_ready());
    std::cout << "4. Post-init ready check: true" << std::endl;

    // 5. 이미지 검출
    const auto& image = test_images_[0];
    cv::Mat rgb_image;
    cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);

    IrisResult detect_result = {};
    IrisSdkError detect_err = iris_sdk_detect(
        rgb_image.data,
        rgb_image.cols,
        rgb_image.rows,
        IRIS_FORMAT_RGB,
        &detect_result
    );

    EXPECT_EQ(IRIS_SDK_OK, detect_err) << "Detection failed with error: "
        << iris_sdk_error_to_string(detect_err);
    std::cout << "5. Detection completed: detected=" << detect_result.detected
              << ", confidence=" << detect_result.confidence << std::endl;

    // 6. 텍스처 로드 (있는 경우)
    if (!texture_path_.empty()) {
        IrisSdkError texture_err = iris_sdk_load_texture(texture_path_.c_str());
        EXPECT_EQ(IRIS_SDK_OK, texture_err) << "Texture load failed";
        std::cout << "6. Texture loaded: " << texture_path_ << std::endl;

        // 7. 렌더링 (검출 성공 시)
        if (detect_result.detected) {
            cv::Mat render_frame = image.clone();
            cv::cvtColor(render_frame, render_frame, cv::COLOR_BGR2RGBA);

            IrisLensConfig lens_config;
            iris_sdk_default_lens_config(&lens_config);
            lens_config.opacity = 0.7f;
            lens_config.scale = 1.2f;

            IrisSdkError render_err = iris_sdk_render_lens(
                render_frame.data,
                render_frame.cols,
                render_frame.rows,
                IRIS_FORMAT_RGBA,
                &detect_result,
                &lens_config
            );

            EXPECT_EQ(IRIS_SDK_OK, render_err) << "Render failed with error: "
                << iris_sdk_error_to_string(render_err);
            std::cout << "7. Rendering completed" << std::endl;

            // 결과 저장
            fs::path output_file = fs::path(output_path_) / "capi_full_flow_result.png";
            cv::cvtColor(render_frame, render_frame, cv::COLOR_RGBA2BGR);
            cv::imwrite(output_file.string(), render_frame);
            std::cout << "   Output saved: " << output_file << std::endl;
        }
    } else {
        std::cout << "6. Skipped texture loading (texture not found)" << std::endl;
    }

    // 8. 결과 정리
    iris_sdk_free_result(&detect_result);
    std::cout << "8. Result freed" << std::endl;

    // 9. SDK 종료
    iris_sdk_destroy();
    EXPECT_FALSE(iris_sdk_is_ready());
    std::cout << "9. SDK destroyed, ready check: false" << std::endl;

    std::cout << "\n=== C API Full Flow Test PASSED ===" << std::endl;
}

// ============================================================================
// 정적 이미지 파이프라인 테스트
// ============================================================================

/**
 * @brief 정적 이미지로 전체 detect+render 파이프라인 테스트
 */
TEST_F(IntegrationTest, FullPipelineWithImage) {
    if (!setup_success_) {
        GTEST_SKIP() << "Setup failed - models or images not available";
    }

    std::cout << "\n=== Full Pipeline With Image Test ===" << std::endl;

    // SDK 초기화
    ASSERT_TRUE(initializeSDK()) << "SDK initialization failed";

    int success_count = 0;
    double total_detect_time = 0.0;
    double total_render_time = 0.0;

    for (size_t i = 0; i < test_images_.size(); ++i) {
        const auto& image = test_images_[i];
        std::string image_name = TEST_IMAGE_FILES[i];

        // RGB 변환
        cv::Mat rgb_image;
        cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);

        // 검출 수행 및 시간 측정
        IrisResult result = {};
        auto detect_start = std::chrono::high_resolution_clock::now();
        IrisSdkError detect_err = iris_sdk_detect(
            rgb_image.data,
            rgb_image.cols,
            rgb_image.rows,
            IRIS_FORMAT_RGB,
            &result
        );
        auto detect_end = std::chrono::high_resolution_clock::now();

        double detect_time = std::chrono::duration<double, std::milli>(
            detect_end - detect_start).count();
        total_detect_time += detect_time;

        EXPECT_EQ(IRIS_SDK_OK, detect_err) << "Detection failed for " << image_name;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "[" << image_name << "] detected=" << result.detected
                  << ", confidence=" << result.confidence
                  << ", detect_time=" << detect_time << "ms" << std::endl;

        if (result.detected) {
            success_count++;

            // 텍스처 로드 및 렌더링 (텍스처 있는 경우)
            if (!texture_path_.empty()) {
                IrisSdkError tex_err = iris_sdk_load_texture(texture_path_.c_str());
                if (tex_err == IRIS_SDK_OK) {
                    cv::Mat render_frame = image.clone();
                    cv::cvtColor(render_frame, render_frame, cv::COLOR_BGR2RGBA);

                    IrisLensConfig config;
                    iris_sdk_default_lens_config(&config);

                    auto render_start = std::chrono::high_resolution_clock::now();
                    IrisSdkError render_err = iris_sdk_render_lens(
                        render_frame.data,
                        render_frame.cols,
                        render_frame.rows,
                        IRIS_FORMAT_RGBA,
                        &result,
                        &config
                    );
                    auto render_end = std::chrono::high_resolution_clock::now();

                    double render_time = std::chrono::duration<double, std::milli>(
                        render_end - render_start).count();
                    total_render_time += render_time;

                    EXPECT_EQ(IRIS_SDK_OK, render_err) << "Render failed for " << image_name;

                    std::cout << "  -> render_time=" << render_time << "ms" << std::endl;

                    // 결과 저장
                    fs::path output_file = fs::path(output_path_) /
                        (fs::path(image_name).stem().string() + "_pipeline_result.png");
                    cv::cvtColor(render_frame, render_frame, cv::COLOR_RGBA2BGR);
                    cv::imwrite(output_file.string(), render_frame);
                }
            }
        }

        iris_sdk_free_result(&result);
    }

    std::cout << "\n--- Summary ---" << std::endl;
    std::cout << "Detection success: " << success_count << "/" << test_images_.size() << std::endl;
    std::cout << "Avg detect time: " << (total_detect_time / test_images_.size()) << " ms" << std::endl;
    if (success_count > 0 && !texture_path_.empty()) {
        std::cout << "Avg render time: " << (total_render_time / success_count) << " ms" << std::endl;
    }

    EXPECT_GT(success_count, 0) << "Expected at least one successful detection";
}

// ============================================================================
// 프레임 포맷 테스트
// ============================================================================

/**
 * @brief 다양한 프레임 포맷 테스트 (BGR, RGBA, Grayscale)
 */
TEST_F(IntegrationTest, MultipleFrameFormats) {
    if (!setup_success_) {
        GTEST_SKIP() << "Setup failed - models or images not available";
    }

    ASSERT_TRUE(initializeSDK()) << "SDK initialization failed";

    std::cout << "\n=== Multiple Frame Formats Test ===" << std::endl;

    const auto& image = test_images_[0];

    struct FormatTest {
        IrisFrameFormat format;
        const char* name;
        int cv_conversion;
    };

    std::vector<FormatTest> format_tests = {
        {IRIS_FORMAT_RGB, "RGB", cv::COLOR_BGR2RGB},
        {IRIS_FORMAT_BGR, "BGR", -1},  // -1 = no conversion
        {IRIS_FORMAT_RGBA, "RGBA", cv::COLOR_BGR2RGBA},
        {IRIS_FORMAT_BGRA, "BGRA", cv::COLOR_BGR2BGRA},
        {IRIS_FORMAT_GRAY, "Grayscale", cv::COLOR_BGR2GRAY}
    };

    for (const auto& test : format_tests) {
        cv::Mat converted;
        if (test.cv_conversion == -1) {
            converted = image.clone();
        } else {
            cv::cvtColor(image, converted, test.cv_conversion);
        }

        IrisResult result = {};
        auto start = std::chrono::high_resolution_clock::now();
        IrisSdkError err = iris_sdk_detect(
            converted.data,
            converted.cols,
            converted.rows,
            test.format,
            &result
        );
        auto end = std::chrono::high_resolution_clock::now();

        double latency = std::chrono::duration<double, std::milli>(end - start).count();

        EXPECT_EQ(IRIS_SDK_OK, err) << "Detection failed for format " << test.name;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "[" << test.name << "] detected=" << result.detected
                  << ", confidence=" << result.confidence
                  << ", latency=" << latency << "ms" << std::endl;

        // 프레임 크기 정보가 올바른지 확인
        EXPECT_GT(result.frame_width, 0);
        EXPECT_GT(result.frame_height, 0);

        iris_sdk_free_result(&result);
    }
}

// ============================================================================
// 연속 처리 안정성 테스트
// ============================================================================

/**
 * @brief 연속 1000+ 프레임 처리 안정성 테스트
 */
TEST_F(IntegrationTest, ContinuousProcessingStability) {
    if (!setup_success_) {
        GTEST_SKIP() << "Setup failed - models or images not available";
    }

    ASSERT_TRUE(initializeSDK()) << "SDK initialization failed";

    std::cout << "\n=== Continuous Processing Stability Test ===" << std::endl;
    std::cout << "Processing " << CONTINUOUS_PROCESSING_FRAMES << " frames..." << std::endl;

    const auto& image = test_images_[0];
    cv::Mat rgb_image;
    cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);

    std::vector<double> latencies;
    latencies.reserve(CONTINUOUS_PROCESSING_FRAMES);

    int success_count = 0;
    int error_count = 0;

    auto total_start = std::chrono::high_resolution_clock::now();

    for (int frame = 0; frame < CONTINUOUS_PROCESSING_FRAMES; ++frame) {
        IrisResult result = {};

        auto frame_start = std::chrono::high_resolution_clock::now();
        IrisSdkError err = iris_sdk_detect(
            rgb_image.data,
            rgb_image.cols,
            rgb_image.rows,
            IRIS_FORMAT_RGB,
            &result
        );
        auto frame_end = std::chrono::high_resolution_clock::now();

        double latency = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
        latencies.push_back(latency);

        if (err == IRIS_SDK_OK) {
            if (result.detected) {
                success_count++;
            }
        } else {
            error_count++;
        }

        iris_sdk_free_result(&result);

        // 진행 상황 표시 (100프레임마다)
        if ((frame + 1) % 100 == 0) {
            std::cout << "  Processed " << (frame + 1) << " frames..." << std::endl;
        }
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time_sec = std::chrono::duration<double>(total_end - total_start).count();
    double actual_fps = CONTINUOUS_PROCESSING_FRAMES / total_time_sec;

    // 통계 계산
    LatencyStats stats = LatencyStats::calculate(latencies);
    stats.print("Continuous Processing Latency");

    std::cout << "\n--- Stability Summary ---" << std::endl;
    std::cout << "Total frames: " << CONTINUOUS_PROCESSING_FRAMES << std::endl;
    std::cout << "Detection success: " << success_count << std::endl;
    std::cout << "API errors: " << error_count << std::endl;
    std::cout << "Total time: " << std::fixed << std::setprecision(2) << total_time_sec << " sec" << std::endl;
    std::cout << "Actual FPS: " << std::fixed << std::setprecision(1) << actual_fps << std::endl;

    // 검증
    EXPECT_EQ(error_count, 0) << "Expected no API errors during continuous processing";
    EXPECT_GT(success_count, 0) << "Expected at least one successful detection";
    EXPECT_LE(stats.avg_ms, TARGET_LATENCY_MS) << "Average latency should be <= " << TARGET_LATENCY_MS << " ms";

    // 메모리 누수 없이 정상 종료 확인
    EXPECT_TRUE(iris_sdk_is_ready()) << "SDK should still be ready after continuous processing";
}

// ============================================================================
// process API 테스트 (detect + render 통합)
// ============================================================================

/**
 * @brief iris_sdk_process API 테스트 (검출 + 렌더링 통합)
 */
TEST_F(IntegrationTest, ProcessAPIIntegration) {
    if (!setup_success_) {
        GTEST_SKIP() << "Setup failed - models or images not available";
    }

    if (texture_path_.empty()) {
        GTEST_SKIP() << "Texture not found - skipping process API test";
    }

    ASSERT_TRUE(initializeSDK()) << "SDK initialization failed";

    // 텍스처 로드
    IrisSdkError tex_err = iris_sdk_load_texture(texture_path_.c_str());
    ASSERT_EQ(IRIS_SDK_OK, tex_err) << "Texture load failed";

    std::cout << "\n=== Process API Integration Test ===" << std::endl;

    const auto& image = test_images_[0];
    cv::Mat rgba_image;
    cv::cvtColor(image, rgba_image, cv::COLOR_BGR2RGBA);

    IrisLensConfig config;
    iris_sdk_default_lens_config(&config);
    config.opacity = 0.8f;
    config.scale = 1.1f;

    IrisResult result = {};

    auto start = std::chrono::high_resolution_clock::now();
    IrisSdkError err = iris_sdk_process(
        rgba_image.data,
        rgba_image.cols,
        rgba_image.rows,
        IRIS_FORMAT_RGBA,
        &config,
        &result
    );
    auto end = std::chrono::high_resolution_clock::now();

    double latency = std::chrono::duration<double, std::milli>(end - start).count();

    EXPECT_EQ(IRIS_SDK_OK, err) << "Process failed with error: " << iris_sdk_error_to_string(err);

    std::cout << "Process API: detected=" << result.detected
              << ", confidence=" << result.confidence
              << ", latency=" << std::fixed << std::setprecision(2) << latency << "ms" << std::endl;

    if (result.detected) {
        // 결과 저장
        fs::path output_file = fs::path(output_path_) / "process_api_result.png";
        cv::cvtColor(rgba_image, rgba_image, cv::COLOR_RGBA2BGR);
        cv::imwrite(output_file.string(), rgba_image);
        std::cout << "Output saved: " << output_file << std::endl;
    }

    iris_sdk_free_result(&result);
}

// ============================================================================
// 에러 핸들링 테스트
// ============================================================================

/**
 * @brief NULL 포인터 및 잘못된 파라미터 에러 핸들링 테스트
 */
TEST_F(IntegrationTest, ErrorHandling) {
    if (!setup_success_) {
        GTEST_SKIP() << "Setup failed - models or images not available";
    }

    std::cout << "\n=== Error Handling Test ===" << std::endl;

    // SDK 초기화
    ASSERT_TRUE(initializeSDK()) << "SDK initialization failed";

    // 1. NULL frame 데이터
    {
        IrisResult result = {};
        IrisSdkError err = iris_sdk_detect(nullptr, 640, 480, IRIS_FORMAT_RGB, &result);
        EXPECT_EQ(IRIS_SDK_NULL_POINTER, err);
        std::cout << "1. NULL frame data: " << iris_sdk_error_to_string(err) << " (expected)" << std::endl;
    }

    // 2. NULL result 포인터
    {
        std::vector<uint8_t> dummy(640 * 480 * 3, 128);
        IrisSdkError err = iris_sdk_detect(dummy.data(), 640, 480, IRIS_FORMAT_RGB, nullptr);
        EXPECT_EQ(IRIS_SDK_NULL_POINTER, err);
        std::cout << "2. NULL result pointer: " << iris_sdk_error_to_string(err) << " (expected)" << std::endl;
    }

    // 3. 잘못된 크기 (width = 0)
    {
        std::vector<uint8_t> dummy(100, 128);
        IrisResult result = {};
        IrisSdkError err = iris_sdk_detect(dummy.data(), 0, 480, IRIS_FORMAT_RGB, &result);
        EXPECT_EQ(IRIS_SDK_INVALID_PARAM, err);
        std::cout << "3. Invalid width (0): " << iris_sdk_error_to_string(err) << " (expected)" << std::endl;
    }

    // 4. 잘못된 크기 (height = 0)
    {
        std::vector<uint8_t> dummy(100, 128);
        IrisResult result = {};
        IrisSdkError err = iris_sdk_detect(dummy.data(), 640, 0, IRIS_FORMAT_RGB, &result);
        EXPECT_EQ(IRIS_SDK_INVALID_PARAM, err);
        std::cout << "4. Invalid height (0): " << iris_sdk_error_to_string(err) << " (expected)" << std::endl;
    }

    // 5. 음수 크기
    {
        std::vector<uint8_t> dummy(100, 128);
        IrisResult result = {};
        IrisSdkError err = iris_sdk_detect(dummy.data(), -1, 480, IRIS_FORMAT_RGB, &result);
        EXPECT_EQ(IRIS_SDK_INVALID_PARAM, err);
        std::cout << "5. Negative width: " << iris_sdk_error_to_string(err) << " (expected)" << std::endl;
    }

    // 6. 렌더링 - NULL 파라미터
    {
        std::vector<uint8_t> dummy(640 * 480 * 4, 128);
        IrisResult iris_result = {};
        IrisLensConfig config;
        iris_sdk_default_lens_config(&config);

        IrisSdkError err = iris_sdk_render_lens(nullptr, 640, 480, IRIS_FORMAT_RGBA, &iris_result, &config);
        EXPECT_EQ(IRIS_SDK_NULL_POINTER, err);
        std::cout << "6. Render with NULL frame: " << iris_sdk_error_to_string(err) << " (expected)" << std::endl;

        err = iris_sdk_render_lens(dummy.data(), 640, 480, IRIS_FORMAT_RGBA, nullptr, &config);
        EXPECT_EQ(IRIS_SDK_NULL_POINTER, err);
        std::cout << "7. Render with NULL iris_result: " << iris_sdk_error_to_string(err) << " (expected)" << std::endl;

        err = iris_sdk_render_lens(dummy.data(), 640, 480, IRIS_FORMAT_RGBA, &iris_result, nullptr);
        EXPECT_EQ(IRIS_SDK_NULL_POINTER, err);
        std::cout << "8. Render with NULL config: " << iris_sdk_error_to_string(err) << " (expected)" << std::endl;
    }

    // 7. 텍스처 없이 렌더링
    {
        // 새로 초기화하여 텍스처 없는 상태 보장
        iris_sdk_destroy();
        ASSERT_TRUE(initializeSDK());

        std::vector<uint8_t> dummy(640 * 480 * 4, 128);
        IrisResult iris_result = {};
        iris_result.detected = true;
        iris_result.left_detected = true;
        iris_result.right_detected = true;
        IrisLensConfig config;
        iris_sdk_default_lens_config(&config);

        IrisSdkError err = iris_sdk_render_lens(dummy.data(), 640, 480, IRIS_FORMAT_RGBA, &iris_result, &config);
        EXPECT_EQ(IRIS_SDK_NO_TEXTURE, err);
        std::cout << "9. Render without texture: " << iris_sdk_error_to_string(err) << " (expected)" << std::endl;
    }

    std::cout << "\n=== Error Handling Test PASSED ===" << std::endl;
}

// ============================================================================
// 초기화/종료 안정성 테스트
// ============================================================================

/**
 * @brief 반복 초기화/종료 안정성 테스트
 */
TEST_F(IntegrationTest, RepeatedInitDestroy) {
    if (!setup_success_) {
        GTEST_SKIP() << "Setup failed - models or images not available";
    }

    std::cout << "\n=== Repeated Init/Destroy Test ===" << std::endl;

    const int iterations = 5;

    for (int i = 0; i < iterations; ++i) {
        // 초기화
        IrisSdkError init_err = iris_sdk_init(model_path_.c_str());
        EXPECT_EQ(IRIS_SDK_OK, init_err) << "Init failed at iteration " << i;
        EXPECT_TRUE(iris_sdk_is_ready());

        // 간단한 검출 수행
        const auto& image = test_images_[0];
        cv::Mat rgb_image;
        cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);

        IrisResult result = {};
        IrisSdkError detect_err = iris_sdk_detect(
            rgb_image.data,
            rgb_image.cols,
            rgb_image.rows,
            IRIS_FORMAT_RGB,
            &result
        );
        EXPECT_EQ(IRIS_SDK_OK, detect_err) << "Detect failed at iteration " << i;
        iris_sdk_free_result(&result);

        // 종료
        iris_sdk_destroy();
        EXPECT_FALSE(iris_sdk_is_ready());

        std::cout << "  Iteration " << (i + 1) << "/" << iterations << " completed" << std::endl;
    }

    std::cout << "\n=== Repeated Init/Destroy Test PASSED ===" << std::endl;
}

/**
 * @brief 중복 초기화 에러 처리 테스트
 */
TEST_F(IntegrationTest, DoubleInitialization) {
    if (!setup_success_) {
        GTEST_SKIP() << "Setup failed - models or images not available";
    }

    std::cout << "\n=== Double Initialization Test ===" << std::endl;

    // 첫 번째 초기화
    IrisSdkError err1 = iris_sdk_init(model_path_.c_str());
    EXPECT_EQ(IRIS_SDK_OK, err1);
    std::cout << "First init: " << iris_sdk_error_to_string(err1) << std::endl;

    // 두 번째 초기화 시도
    IrisSdkError err2 = iris_sdk_init(model_path_.c_str());
    EXPECT_EQ(IRIS_SDK_ALREADY_INITIALIZED, err2);
    std::cout << "Second init: " << iris_sdk_error_to_string(err2) << " (expected)" << std::endl;

    // 여전히 정상 동작하는지 확인
    EXPECT_TRUE(iris_sdk_is_ready());
}

/**
 * @brief 중복 destroy 안전성 테스트
 */
TEST_F(IntegrationTest, MultipleDestroy) {
    if (!setup_success_) {
        GTEST_SKIP() << "Setup failed - models or images not available";
    }

    std::cout << "\n=== Multiple Destroy Safety Test ===" << std::endl;

    ASSERT_TRUE(initializeSDK());

    // 여러 번 destroy 호출
    iris_sdk_destroy();
    std::cout << "First destroy: OK" << std::endl;
    EXPECT_FALSE(iris_sdk_is_ready());

    iris_sdk_destroy();
    std::cout << "Second destroy: OK (no crash)" << std::endl;
    EXPECT_FALSE(iris_sdk_is_ready());

    iris_sdk_destroy();
    std::cout << "Third destroy: OK (no crash)" << std::endl;
    EXPECT_FALSE(iris_sdk_is_ready());
}

// ============================================================================
// 설정 변경 테스트
// ============================================================================

/**
 * @brief 런타임 설정 변경 테스트
 */
TEST_F(IntegrationTest, RuntimeConfigChange) {
    if (!setup_success_) {
        GTEST_SKIP() << "Setup failed - models or images not available";
    }

    ASSERT_TRUE(initializeSDK()) << "SDK initialization failed";

    std::cout << "\n=== Runtime Config Change Test ===" << std::endl;

    // 설정 변경 테스트
    IrisSdkError err = iris_sdk_set_config("min_confidence", "0.3");
    // 설정이 지원되면 OK, 아니면 다른 에러 (NOT_INITIALIZED가 아님)
    EXPECT_NE(IRIS_SDK_NOT_INITIALIZED, err);
    std::cout << "Set min_confidence=0.3: " << iris_sdk_error_to_string(err) << std::endl;

    // 잘못된 설정 키
    err = iris_sdk_set_config("invalid_key", "value");
    // 이 경우 INVALID_PARAM이 예상됨
    std::cout << "Set invalid_key: " << iris_sdk_error_to_string(err) << std::endl;
}

// ============================================================================
// 텍스처 관리 테스트
// ============================================================================

/**
 * @brief 메모리에서 텍스처 로드 테스트
 */
TEST_F(IntegrationTest, TextureFromMemory) {
    if (!setup_success_) {
        GTEST_SKIP() << "Setup failed - models or images not available";
    }

    ASSERT_TRUE(initializeSDK()) << "SDK initialization failed";

    std::cout << "\n=== Texture From Memory Test ===" << std::endl;

    // 간단한 텍스처 데이터 생성 (128x128 RGBA)
    const int tex_width = 128;
    const int tex_height = 128;
    std::vector<uint8_t> texture_data(tex_width * tex_height * 4);

    // 그라데이션 패턴 생성
    for (int y = 0; y < tex_height; ++y) {
        for (int x = 0; x < tex_width; ++x) {
            int idx = (y * tex_width + x) * 4;
            texture_data[idx + 0] = static_cast<uint8_t>(x * 2);      // R
            texture_data[idx + 1] = static_cast<uint8_t>(y * 2);      // G
            texture_data[idx + 2] = 128;                               // B
            texture_data[idx + 3] = 200;                               // A
        }
    }

    IrisSdkError err = iris_sdk_load_texture_from_memory(
        texture_data.data(), tex_width, tex_height);

    EXPECT_EQ(IRIS_SDK_OK, err) << "Texture from memory load failed";
    std::cout << "Loaded texture from memory: " << tex_width << "x" << tex_height << std::endl;
}

// ============================================================================
// 메모리 안정성 테스트
// ============================================================================

/**
 * @brief 메모리 누수 방지 테스트 - 반복 검출
 */
TEST_F(IntegrationTest, MemoryStabilityRepeatedDetection) {
    if (!setup_success_) {
        GTEST_SKIP() << "Setup failed - models or images not available";
    }

    ASSERT_TRUE(initializeSDK()) << "SDK initialization failed";

    std::cout << "\n=== Memory Stability Test (Repeated Detection) ===" << std::endl;
    std::cout << "Running " << MEMORY_STABILITY_ITERATIONS << " detection iterations..." << std::endl;

    const auto& image = test_images_[0];
    cv::Mat rgb_image;
    cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);

    for (int i = 0; i < MEMORY_STABILITY_ITERATIONS; ++i) {
        IrisResult result = {};

        IrisSdkError err = iris_sdk_detect(
            rgb_image.data,
            rgb_image.cols,
            rgb_image.rows,
            IRIS_FORMAT_RGB,
            &result
        );

        EXPECT_EQ(IRIS_SDK_OK, err) << "Detection failed at iteration " << i;

        // 결과 정리 (메모리 해제)
        iris_sdk_free_result(&result);

        if ((i + 1) % 25 == 0) {
            std::cout << "  Completed " << (i + 1) << " iterations" << std::endl;
        }
    }

    // SDK가 여전히 정상인지 확인
    EXPECT_TRUE(iris_sdk_is_ready());
    std::cout << "Memory stability test completed. SDK still ready." << std::endl;
}

#else  // !IRIS_SDK_HAS_TFLITE || !IRIS_SDK_HAS_OPENCV

// ============================================================================
// TFLite 또는 OpenCV 없이 빌드된 경우
// ============================================================================

TEST(IntegrationTest, SkippedWithoutDependencies) {
    GTEST_SKIP() << "TFLite or OpenCV not available - skipping integration tests";
}

#endif  // IRIS_SDK_HAS_TFLITE && IRIS_SDK_HAS_OPENCV

// ============================================================================
// 의존성 없이 실행 가능한 기본 테스트
// ============================================================================

TEST(IntegrationTestBasic, ErrorToStringWorks) {
    EXPECT_STREQ("IRIS_SDK_OK", iris_sdk_error_to_string(IRIS_SDK_OK));
    EXPECT_STREQ("IRIS_SDK_NOT_INITIALIZED", iris_sdk_error_to_string(IRIS_SDK_NOT_INITIALIZED));
    EXPECT_STREQ("IRIS_SDK_NULL_POINTER", iris_sdk_error_to_string(IRIS_SDK_NULL_POINTER));
    EXPECT_STREQ("IRIS_SDK_UNKNOWN", iris_sdk_error_to_string(static_cast<IrisSdkError>(9999)));
}

TEST(IntegrationTestBasic, VersionAvailable) {
    const char* version = iris_sdk_get_version();
    ASSERT_NE(nullptr, version);
    EXPECT_GT(std::strlen(version), 0u);
}

TEST(IntegrationTestBasic, DefaultLensConfigWorks) {
    IrisLensConfig config = {};
    iris_sdk_default_lens_config(&config);

    EXPECT_FLOAT_EQ(0.7f, config.opacity);
    EXPECT_FLOAT_EQ(1.0f, config.scale);
    EXPECT_FLOAT_EQ(0.0f, config.offset_x);
    EXPECT_FLOAT_EQ(0.0f, config.offset_y);
    EXPECT_FLOAT_EQ(0.0f, config.rotation);
    EXPECT_TRUE(config.apply_left);
    EXPECT_TRUE(config.apply_right);
}

TEST(IntegrationTestBasic, FreeResultWithNullSafe) {
    // NULL 포인터로 호출해도 크래시 없어야 함
    iris_sdk_free_result(nullptr);
}

TEST(IntegrationTestBasic, DestroyWithoutInitSafe) {
    // 초기화 없이 destroy 호출해도 크래시 없어야 함
    iris_sdk_destroy();
    iris_sdk_destroy();
    EXPECT_FALSE(iris_sdk_is_ready());
}
