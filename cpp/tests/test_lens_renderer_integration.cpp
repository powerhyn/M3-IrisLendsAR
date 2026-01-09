/**
 * @file test_lens_renderer_integration.cpp
 * @brief LensRenderer 통합 테스트 - MediaPipeDetector 연동
 *
 * P1-W4-01 검증: 렌더링 시간 및 시각적 품질 확인
 */

#include <gtest/gtest.h>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <iomanip>

#include "iris_sdk/mediapipe_detector.h"
#include "iris_sdk/lens_renderer.h"
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

namespace iris_sdk {
namespace test {

// 렌즈 텍스처 파일
const std::string LENS_ALPHA = "lens_sample_type_alpha_01.png";  // 배경 제거
const std::string LENS_BG = "lens_sample_type_bg_01.png";        // 배경 있음

// 테스트 이미지들
const std::vector<std::string> TEST_IMAGES = {
    "iris_test_01.png",
    "iris_test_02.png",
    "iris_test_03.png",
    "iris_test_04.png",
    "Iris_test_05.png",
    "iris_test_06.png"
};

class LensRendererIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 프로젝트 루트 찾기
        findProjectRoot();
        ASSERT_FALSE(project_root_.empty()) << "프로젝트 루트를 찾을 수 없음";

        // 출력 디렉토리 생성
        output_path_ = test_data_path_ / "output";
        fs::create_directories(output_path_);

        // MediaPipeDetector 초기화
        detector_ = std::make_unique<MediaPipeDetector>();
        ASSERT_TRUE(detector_->initialize(models_path_.string()))
            << "MediaPipeDetector 초기화 실패. 모델 경로: " << models_path_;

        // LensRenderer 초기화
        renderer_ = std::make_unique<LensRenderer>();
        ASSERT_TRUE(renderer_->initialize())
            << "LensRenderer 초기화 실패";

        std::cout << "프로젝트 루트: " << project_root_ << std::endl;
        std::cout << "모델 경로: " << models_path_ << std::endl;
        std::cout << "테스트 데이터: " << test_data_path_ << std::endl;
        std::cout << "출력 경로: " << output_path_ << std::endl;
    }

    void findProjectRoot() {
        // 가능한 프로젝트 루트 후보들
        std::vector<fs::path> possible_roots = {
            ".",
            "..",
            "../..",
            "../../..",
            "../../../..",
        };

        for (const auto& root : possible_roots) {
            fs::path models_candidate = root / "shared" / "models";
            fs::path test_data_candidate = root / "shared" / "test_data";

            if (fs::exists(models_candidate) && fs::exists(test_data_candidate)) {
                project_root_ = fs::canonical(root);
                models_path_ = fs::canonical(models_candidate);
                test_data_path_ = fs::canonical(test_data_candidate);
                break;
            }
        }
    }

    void TearDown() override {
        if (renderer_) renderer_->release();
        if (detector_) detector_->release();
    }

    // 렌더링 결과 저장 및 시간 측정
    struct RenderResult {
        bool success;
        double detect_time_ms;
        double render_time_ms;
        std::string output_path;
    };

    RenderResult renderWithTexture(const std::string& image_path,
                                   const std::string& texture_path,
                                   const std::string& output_suffix,
                                   const LensConfig& config) {
        RenderResult result{};

        // 이미지 로드 (BGR)
        cv::Mat frame = cv::imread(image_path);
        if (frame.empty()) {
            std::cerr << "이미지 로드 실패: " << image_path << std::endl;
            return result;
        }

        // 텍스처 로드
        if (!renderer_->loadTexture(texture_path)) {
            std::cerr << "텍스처 로드 실패: " << texture_path << std::endl;
            return result;
        }

        // 검출을 위해 RGB로 변환
        cv::Mat rgb_frame;
        cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);

        // 홍채 검출 (시간 측정)
        auto detect_start = std::chrono::high_resolution_clock::now();
        IrisResult iris_result = detector_->detect(
            rgb_frame.data, rgb_frame.cols, rgb_frame.rows, FrameFormat::RGB);
        auto detect_end = std::chrono::high_resolution_clock::now();
        result.detect_time_ms = std::chrono::duration<double, std::milli>(
            detect_end - detect_start).count();

        if (!iris_result.detected) {
            std::cerr << "홍채 검출 실패: " << image_path << std::endl;
            return result;
        }

        // 렌더링 (BGR 프레임에 직접 적용)
        auto render_start = std::chrono::high_resolution_clock::now();
        bool render_success = renderer_->render(frame, iris_result, config);
        auto render_end = std::chrono::high_resolution_clock::now();
        result.render_time_ms = std::chrono::duration<double, std::milli>(
            render_end - render_start).count();

        if (!render_success) {
            std::cerr << "렌더링 실패: " << image_path << std::endl;
            return result;
        }

        // 결과 저장
        fs::path input_path(image_path);
        std::string output_name = input_path.stem().string() + "_" + output_suffix + ".png";
        result.output_path = (output_path_ / output_name).string();
        cv::imwrite(result.output_path, frame);

        result.success = true;
        return result;
    }

    std::unique_ptr<MediaPipeDetector> detector_;
    std::unique_ptr<LensRenderer> renderer_;
    fs::path project_root_;
    fs::path models_path_;
    fs::path test_data_path_;
    fs::path output_path_;
};

// =============================================================================
// 알파 채널 렌즈 테스트 (배경 제거)
// =============================================================================

TEST_F(LensRendererIntegrationTest, RenderWithAlphaTexture) {
    std::string texture_path = (test_data_path_ / LENS_ALPHA).string();

    LensConfig config;
    config.opacity = 0.8f;
    config.scale = 1.2f;
    config.edge_feather = 0.15f;
    config.blend_mode = BlendMode::Normal;

    std::cout << "\n========================================\n";
    std::cout << "렌즈 타입: Alpha (배경 제거)\n";
    std::cout << "설정: opacity=" << config.opacity
              << ", scale=" << config.scale
              << ", feather=" << config.edge_feather << "\n";
    std::cout << "========================================\n";

    double total_detect_time = 0;
    double total_render_time = 0;
    int success_count = 0;

    for (const auto& image_name : TEST_IMAGES) {
        std::string image_path = (test_data_path_ / image_name).string();

        auto result = renderWithTexture(image_path, texture_path, "alpha", config);

        if (result.success) {
            success_count++;
            total_detect_time += result.detect_time_ms;
            total_render_time += result.render_time_ms;

            std::cout << std::fixed << std::setprecision(2);
            std::cout << "[" << image_name << "]\n";
            std::cout << "  검출: " << result.detect_time_ms << " ms\n";
            std::cout << "  렌더링: " << result.render_time_ms << " ms\n";
            std::cout << "  출력: " << result.output_path << "\n";

            // 렌더링 시간 목표 검증 (10ms 이하)
            EXPECT_LE(result.render_time_ms, 10.0)
                << "렌더링 시간이 목표(10ms)를 초과: " << result.render_time_ms << "ms";
        } else {
            std::cout << "[" << image_name << "] 실패\n";
        }
    }

    ASSERT_GT(success_count, 0) << "성공한 테스트가 없음";

    std::cout << "\n--- Alpha 텍스처 요약 ---\n";
    std::cout << "성공: " << success_count << "/" << TEST_IMAGES.size() << "\n";
    std::cout << "평균 검출 시간: " << (total_detect_time / success_count) << " ms\n";
    std::cout << "평균 렌더링 시간: " << (total_render_time / success_count) << " ms\n";
}

// =============================================================================
// 배경 있는 렌즈 테스트
// =============================================================================

TEST_F(LensRendererIntegrationTest, RenderWithBgTexture) {
    std::string texture_path = (test_data_path_ / LENS_BG).string();

    LensConfig config;
    config.opacity = 0.7f;
    config.scale = 1.2f;
    config.edge_feather = 0.2f;
    config.blend_mode = BlendMode::Normal;

    std::cout << "\n========================================\n";
    std::cout << "렌즈 타입: BG (배경 있음)\n";
    std::cout << "설정: opacity=" << config.opacity
              << ", scale=" << config.scale
              << ", feather=" << config.edge_feather << "\n";
    std::cout << "========================================\n";

    double total_detect_time = 0;
    double total_render_time = 0;
    int success_count = 0;

    for (const auto& image_name : TEST_IMAGES) {
        std::string image_path = (test_data_path_ / image_name).string();

        auto result = renderWithTexture(image_path, texture_path, "bg", config);

        if (result.success) {
            success_count++;
            total_detect_time += result.detect_time_ms;
            total_render_time += result.render_time_ms;

            std::cout << std::fixed << std::setprecision(2);
            std::cout << "[" << image_name << "]\n";
            std::cout << "  검출: " << result.detect_time_ms << " ms\n";
            std::cout << "  렌더링: " << result.render_time_ms << " ms\n";
            std::cout << "  출력: " << result.output_path << "\n";

            EXPECT_LE(result.render_time_ms, 10.0)
                << "렌더링 시간이 목표(10ms)를 초과: " << result.render_time_ms << "ms";
        } else {
            std::cout << "[" << image_name << "] 실패\n";
        }
    }

    ASSERT_GT(success_count, 0) << "성공한 테스트가 없음";

    std::cout << "\n--- BG 텍스처 요약 ---\n";
    std::cout << "성공: " << success_count << "/" << TEST_IMAGES.size() << "\n";
    std::cout << "평균 검출 시간: " << (total_detect_time / success_count) << " ms\n";
    std::cout << "평균 렌더링 시간: " << (total_render_time / success_count) << " ms\n";
}

// =============================================================================
// 블렌드 모드 비교 테스트
// =============================================================================

TEST_F(LensRendererIntegrationTest, CompareBlendModes) {
    std::string texture_path = (test_data_path_ / LENS_ALPHA).string();

    std::cout << "\n========================================\n";
    std::cout << "블렌드 모드 비교 테스트 (모든 이미지)\n";
    std::cout << "========================================\n";

    struct BlendModeTest {
        BlendMode mode;
        std::string name;
    };

    std::vector<BlendModeTest> blend_tests = {
        {BlendMode::Normal, "normal"},
        {BlendMode::Multiply, "multiply"},
        {BlendMode::Screen, "screen"},
        {BlendMode::Overlay, "overlay"}
    };

    for (const auto& image_name : TEST_IMAGES) {
        std::string image_path = (test_data_path_ / image_name).string();
        fs::path img_stem = fs::path(image_name).stem();

        std::cout << "\n[" << image_name << "]\n";

        for (const auto& test : blend_tests) {
            LensConfig config;
            config.opacity = 0.8f;
            config.scale = 1.2f;
            config.edge_feather = 0.15f;
            config.blend_mode = test.mode;

            std::string suffix = "blend_" + test.name;
            auto result = renderWithTexture(image_path, texture_path, suffix, config);

            if (result.success) {
                std::cout << "  " << test.name << ": " << result.render_time_ms << " ms\n";
            }
        }
    }
}

// =============================================================================
// 스케일 및 투명도 변화 테스트
// =============================================================================

TEST_F(LensRendererIntegrationTest, ScaleAndOpacityVariations) {
    std::string texture_path = (test_data_path_ / LENS_ALPHA).string();

    std::cout << "\n========================================\n";
    std::cout << "스케일 및 투명도 변화 테스트 (모든 이미지)\n";
    std::cout << "========================================\n";

    std::vector<float> scales = {0.8f, 1.0f, 1.2f, 1.5f};
    std::vector<float> opacities = {0.3f, 0.5f, 0.7f, 1.0f};

    for (const auto& image_name : TEST_IMAGES) {
        std::string image_path = (test_data_path_ / image_name).string();
        fs::path img_stem = fs::path(image_name).stem();

        std::cout << "\n[" << image_name << "]\n";

        // 스케일 변화
        std::cout << "  스케일: ";
        for (float scale : scales) {
            LensConfig config;
            config.opacity = 0.8f;
            config.scale = scale;
            config.edge_feather = 0.15f;

            std::string suffix = "scale_" + std::to_string(static_cast<int>(scale * 100));
            auto result = renderWithTexture(image_path, texture_path, suffix, config);

            if (result.success) {
                std::cout << scale << " ";
            }
        }
        std::cout << "\n";

        // 투명도 변화
        std::cout << "  투명도: ";
        for (float opacity : opacities) {
            LensConfig config;
            config.opacity = opacity;
            config.scale = 1.2f;
            config.edge_feather = 0.15f;

            std::string suffix = "opacity_" + std::to_string(static_cast<int>(opacity * 100));
            auto result = renderWithTexture(image_path, texture_path, suffix, config);

            if (result.success) {
                std::cout << opacity << " ";
            }
        }
        std::cout << "\n";
    }
}

// =============================================================================
// 연속 렌더링 성능 테스트
// =============================================================================

TEST_F(LensRendererIntegrationTest, ContinuousRenderingPerformance) {
    std::string texture_path = (test_data_path_ / LENS_ALPHA).string();
    std::string image_path = (test_data_path_ / "iris_test_01.png").string();

    cv::Mat original = cv::imread(image_path);
    ASSERT_FALSE(original.empty());

    ASSERT_TRUE(renderer_->loadTexture(texture_path));

    // 검출을 위해 RGB로 변환
    cv::Mat rgb_frame;
    cv::cvtColor(original, rgb_frame, cv::COLOR_BGR2RGB);

    IrisResult iris_result = detector_->detect(
        rgb_frame.data, rgb_frame.cols, rgb_frame.rows, FrameFormat::RGB);
    ASSERT_TRUE(iris_result.detected);

    LensConfig config;
    config.opacity = 0.8f;
    config.scale = 1.2f;
    config.edge_feather = 0.15f;

    const int NUM_ITERATIONS = 100;
    std::vector<double> render_times;
    render_times.reserve(NUM_ITERATIONS);

    std::cout << "\n========================================\n";
    std::cout << "연속 렌더링 성능 테스트 (" << NUM_ITERATIONS << "회)\n";
    std::cout << "========================================\n";

    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        cv::Mat frame = original.clone();

        auto start = std::chrono::high_resolution_clock::now();
        renderer_->render(frame, iris_result, config);
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        render_times.push_back(ms);
    }

    // 통계 계산
    double sum = 0, min_time = render_times[0], max_time = render_times[0];
    for (double t : render_times) {
        sum += t;
        min_time = std::min(min_time, t);
        max_time = std::max(max_time, t);
    }
    double avg = sum / NUM_ITERATIONS;

    // 표준편차
    double sq_sum = 0;
    for (double t : render_times) {
        sq_sum += (t - avg) * (t - avg);
    }
    double stddev = std::sqrt(sq_sum / NUM_ITERATIONS);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "평균: " << avg << " ms\n";
    std::cout << "최소: " << min_time << " ms\n";
    std::cout << "최대: " << max_time << " ms\n";
    std::cout << "표준편차: " << stddev << " ms\n";
    std::cout << "예상 FPS: " << (1000.0 / avg) << " fps (렌더링만)\n";

    // 목표 검증: 평균 10ms 이하
    EXPECT_LE(avg, 10.0) << "평균 렌더링 시간이 목표(10ms)를 초과";
}

} // namespace test
} // namespace iris_sdk
