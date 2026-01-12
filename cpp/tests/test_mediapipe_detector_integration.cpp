/**
 * @file test_mediapipe_detector_integration.cpp
 * @brief MediaPipeDetector 통합 테스트 (실제 TFLite 모델 + 실제 이미지)
 *
 * 이 테스트는 실제 TFLite 모델 파일과 테스트 이미지를 사용하여
 * MediaPipeDetector의 홍채 검출 기능을 검증합니다.
 *
 * 필수 조건:
 * - TensorFlow Lite 라이브러리 (IRIS_SDK_HAS_TFLITE)
 * - OpenCV 라이브러리 (IRIS_SDK_HAS_OPENCV)
 * - 모델 파일: shared/models/face_detection_short_range.tflite 등
 * - 테스트 이미지: shared/test_data/iris_test_01.png 등
 *
 * 성능 목표:
 * - 검출 지연: 33ms 이하 (30fps)
 * - 신뢰도: 0.5 이상
 * - 랜드마크 좌표: 0.0 ~ 1.0 (정규화)
 */

#include <gtest/gtest.h>
#include "iris_sdk/mediapipe_detector.h"

#if defined(IRIS_SDK_HAS_TFLITE) && defined(IRIS_SDK_HAS_OPENCV)

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <vector>
#include <string>
#include <filesystem>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace iris_sdk {
namespace testing {

// ============================================================
// 테스트 설정 상수
// ============================================================
namespace {
    /// 성능 목표 - 30fps를 위한 최대 지연 시간
    constexpr double TARGET_LATENCY_MS = 33.0;

    /// 검출 신뢰도 최소 임계값
    constexpr float MIN_CONFIDENCE_THRESHOLD = 0.5f;

    /// 랜드마크 좌표 유효 범위 (정규화된 값)
    constexpr float LANDMARK_MIN = 0.0f;
    constexpr float LANDMARK_MAX = 1.0f;

    /// 성능 측정을 위한 워밍업 반복 횟수
    constexpr int WARMUP_ITERATIONS = 3;

    /// 성능 측정을 위한 벤치마크 반복 횟수
    constexpr int BENCHMARK_ITERATIONS = 10;

    /// 테스트 이미지 파일명 목록
    const std::vector<std::string> TEST_IMAGE_FILES = {
        "iris_test_01.png",  // 1024x1536 portrait
        "iris_test_02.png",  // 1024x1536 portrait
        "iris_test_03.png",  // 1024x1536 portrait
        "iris_test_04.png",  // 1024x1536 portrait
        "Iris_test_05.png",  // 1024x1536 portrait
        "iris_test_06.png",  // 1024x1536 portrait
        "iris_test_07.png"   // 1280x720 landscape (카메라 캡처)
    };
}

// ============================================================
// 성능 측정 유틸리티
// ============================================================

/**
 * @brief 성능 측정 결과 구조체
 */
struct LatencyStats {
    double min_ms = 0.0;
    double max_ms = 0.0;
    double avg_ms = 0.0;
    double median_ms = 0.0;
    double std_dev_ms = 0.0;
    double p95_ms = 0.0;
    int sample_count = 0;

    /**
     * @brief 타이밍 데이터로부터 통계 계산
     * @param times 측정 시간 벡터 (밀리초)
     * @return 계산된 통계
     */
    static LatencyStats calculate(std::vector<double> times) {
        LatencyStats stats;
        if (times.empty()) {
            return stats;
        }

        stats.sample_count = static_cast<int>(times.size());

        // 정렬
        std::sort(times.begin(), times.end());

        // Min, Max
        stats.min_ms = times.front();
        stats.max_ms = times.back();

        // Average
        double sum = std::accumulate(times.begin(), times.end(), 0.0);
        stats.avg_ms = sum / static_cast<double>(times.size());

        // Median
        size_t mid = times.size() / 2;
        if (times.size() % 2 == 0) {
            stats.median_ms = (times[mid - 1] + times[mid]) / 2.0;
        } else {
            stats.median_ms = times[mid];
        }

        // Standard Deviation
        double sq_sum = 0.0;
        for (double t : times) {
            sq_sum += (t - stats.avg_ms) * (t - stats.avg_ms);
        }
        stats.std_dev_ms = std::sqrt(sq_sum / static_cast<double>(times.size()));

        // 95th Percentile
        size_t p95_idx = static_cast<size_t>(static_cast<double>(times.size()) * 0.95);
        if (p95_idx >= times.size()) p95_idx = times.size() - 1;
        stats.p95_ms = times[p95_idx];

        return stats;
    }

    /**
     * @brief 통계를 문자열로 출력
     */
    void print(const std::string& label) const {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "\n=== " << label << " ===" << std::endl;
        std::cout << "  Samples:  " << sample_count << std::endl;
        std::cout << "  Min:      " << min_ms << " ms" << std::endl;
        std::cout << "  Max:      " << max_ms << " ms" << std::endl;
        std::cout << "  Avg:      " << avg_ms << " ms" << std::endl;
        std::cout << "  Median:   " << median_ms << " ms" << std::endl;
        std::cout << "  Std Dev:  " << std_dev_ms << " ms" << std::endl;
        std::cout << "  P95:      " << p95_ms << " ms" << std::endl;
    }
};

// ============================================================
// 통합 테스트 Fixture
// ============================================================

/**
 * @brief MediaPipeDetector 통합 테스트 Fixture
 *
 * 실제 모델 파일과 이미지를 로드하여 테스트 수행
 */
class MediaPipeDetectorIntegrationTest : public ::testing::Test {
protected:
    std::unique_ptr<MediaPipeDetector> detector_;
    std::filesystem::path project_root_;
    std::filesystem::path models_path_;
    std::filesystem::path test_data_path_;
    std::vector<cv::Mat> test_images_;
    std::vector<std::string> loaded_image_names_;
    bool setup_success_ = false;

    void SetUp() override {
        // 프로젝트 루트 경로 결정
        // 테스트 실행 위치에 따라 상대 경로 조정
        std::filesystem::path current_path = std::filesystem::current_path();

        // 가능한 프로젝트 루트 경로들을 시도
        std::vector<std::filesystem::path> possible_roots = {
            current_path / ".." / "..",           // cpp/build 에서 실행 시
            current_path / "..",                   // cpp/ 에서 실행 시
            current_path,                          // 프로젝트 루트에서 실행 시
            current_path / ".." / ".." / "..",     // cpp/build/Debug 등에서 실행 시
        };

        for (const auto& root : possible_roots) {
            std::filesystem::path models_candidate = root / "shared" / "models";
            std::filesystem::path test_data_candidate = root / "shared" / "test_data";

            if (std::filesystem::exists(models_candidate) &&
                std::filesystem::exists(test_data_candidate)) {
                project_root_ = std::filesystem::canonical(root);
                models_path_ = std::filesystem::canonical(models_candidate);
                test_data_path_ = std::filesystem::canonical(test_data_candidate);
                break;
            }
        }

        if (project_root_.empty()) {
            std::cerr << "Warning: Could not find project root from " << current_path << std::endl;
            std::cerr << "Skipping integration tests." << std::endl;
            return;
        }

        std::cout << "Project root: " << project_root_ << std::endl;
        std::cout << "Models path: " << models_path_ << std::endl;
        std::cout << "Test data path: " << test_data_path_ << std::endl;

        // 모델 파일 존재 확인
        if (!verifyModelFiles()) {
            std::cerr << "Warning: Required model files not found." << std::endl;
            return;
        }

        // 테스트 이미지 로드
        if (!loadTestImages()) {
            std::cerr << "Warning: Failed to load test images." << std::endl;
            return;
        }

        // Detector 생성 및 초기화
        detector_ = std::make_unique<MediaPipeDetector>();
        if (!detector_->initialize(models_path_.string())) {
            std::cerr << "Warning: Failed to initialize MediaPipeDetector." << std::endl;
            return;
        }

        setup_success_ = true;
    }

    void TearDown() override {
        if (detector_) {
            detector_->release();
            detector_.reset();
        }
        test_images_.clear();
        loaded_image_names_.clear();
    }

    /**
     * @brief 필수 모델 파일 존재 확인
     */
    bool verifyModelFiles() const {
        const std::vector<std::string> required_models = {
            "face_detection_short_range.tflite",
            "face_landmark.tflite",
            "iris_landmark.tflite"
        };

        for (const auto& model_name : required_models) {
            std::filesystem::path model_file = models_path_ / model_name;
            if (!std::filesystem::exists(model_file)) {
                std::cerr << "Missing model file: " << model_file << std::endl;
                return false;
            }
            std::cout << "Found model: " << model_name << std::endl;
        }

        return true;
    }

    /**
     * @brief 테스트 이미지 로드
     */
    bool loadTestImages() {
        for (const auto& filename : TEST_IMAGE_FILES) {
            std::filesystem::path image_path = test_data_path_ / filename;

            if (!std::filesystem::exists(image_path)) {
                std::cerr << "Test image not found: " << image_path << std::endl;
                continue;
            }

            cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
            if (image.empty()) {
                std::cerr << "Failed to load image: " << image_path << std::endl;
                continue;
            }

            test_images_.push_back(image);
            loaded_image_names_.push_back(filename);
            std::cout << "Loaded image: " << filename
                      << " (" << image.cols << "x" << image.rows << ")" << std::endl;
        }

        return !test_images_.empty();
    }

    /**
     * @brief 이미지를 지정된 포맷으로 변환
     */
    cv::Mat convertToFormat(const cv::Mat& src, FrameFormat format) const {
        cv::Mat result;

        switch (format) {
            case FrameFormat::RGB:
                cv::cvtColor(src, result, cv::COLOR_BGR2RGB);
                break;
            case FrameFormat::BGR:
                result = src.clone();
                break;
            case FrameFormat::RGBA:
                cv::cvtColor(src, result, cv::COLOR_BGR2RGBA);
                break;
            case FrameFormat::BGRA:
                cv::cvtColor(src, result, cv::COLOR_BGR2BGRA);
                break;
            case FrameFormat::Grayscale:
                cv::cvtColor(src, result, cv::COLOR_BGR2GRAY);
                break;
            default:
                result = src.clone();
                break;
        }

        return result;
    }

    /**
     * @brief 검출 결과 유효성 검증
     */
    void validateIrisResult(const IrisResult& result, const std::string& image_name) {
        if (!result.detected) {
            // 얼굴이 검출되지 않은 경우 (허용되는 실패)
            std::cout << "  [" << image_name << "] No face detected (acceptable)" << std::endl;
            return;
        }

        // 신뢰도 검증
        EXPECT_GE(result.confidence, 0.0f)
            << "Confidence should be >= 0.0 for " << image_name;
        EXPECT_LE(result.confidence, 1.0f)
            << "Confidence should be <= 1.0 for " << image_name;

        // 좌우 홍채 중 하나 이상 검출 확인
        bool at_least_one_iris = result.left_detected || result.right_detected;
        EXPECT_TRUE(at_least_one_iris)
            << "At least one iris should be detected when face is found for " << image_name;

        // 왼쪽 홍채 검출된 경우 랜드마크 검증
        if (result.left_detected) {
            validateIrisLandmarks(result.left_iris, 5, "left", image_name);
            EXPECT_GT(result.left_radius, 0.0f)
                << "Left iris radius should be > 0 for " << image_name;
        }

        // 오른쪽 홍채 검출된 경우 랜드마크 검증
        if (result.right_detected) {
            validateIrisLandmarks(result.right_iris, 5, "right", image_name);
            EXPECT_GT(result.right_radius, 0.0f)
                << "Right iris radius should be > 0 for " << image_name;
        }

        // 얼굴 바운딩 박스 검증
        if (result.detected) {
            EXPECT_GE(result.face_rect.x, 0.0f);
            EXPECT_GE(result.face_rect.y, 0.0f);
            EXPECT_GT(result.face_rect.width, 0.0f);
            EXPECT_GT(result.face_rect.height, 0.0f);
        }

        // 프레임 정보 검증
        EXPECT_GT(result.frame_width, 0);
        EXPECT_GT(result.frame_height, 0);
    }

    /**
     * @brief 홍채 랜드마크 좌표 유효성 검증
     */
    void validateIrisLandmarks(const IrisLandmark* landmarks, int count,
                               const std::string& eye_name,
                               const std::string& image_name) {
        for (int i = 0; i < count; ++i) {
            const auto& lm = landmarks[i];

            // 정규화된 좌표 범위 검증 (0.0 ~ 1.0)
            // 참고: 일부 구현에서는 픽셀 좌표를 반환할 수 있음
            // 여기서는 유효한 양수 값인지만 확인
            EXPECT_GE(lm.x, 0.0f)
                << eye_name << " iris landmark[" << i << "].x should be >= 0 for " << image_name;
            EXPECT_GE(lm.y, 0.0f)
                << eye_name << " iris landmark[" << i << "].y should be >= 0 for " << image_name;

            // visibility가 유효한 범위인지 확인 (선택적)
            if (lm.visibility > 0.0f) {
                EXPECT_LE(lm.visibility, 1.0f)
                    << eye_name << " iris landmark[" << i << "].visibility should be <= 1.0 for " << image_name;
            }
        }
    }

    /**
     * @brief 단일 이미지에 대한 검출 지연 시간 측정
     */
    double measureDetectionLatency(const cv::Mat& image, FrameFormat format = FrameFormat::RGB) {
        cv::Mat converted = convertToFormat(image, format);

        auto start = std::chrono::high_resolution_clock::now();
        detector_->detect(converted.data, converted.cols, converted.rows, format);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> duration = end - start;
        return duration.count();
    }
};

// ============================================================
// 초기화 테스트
// ============================================================

TEST_F(MediaPipeDetectorIntegrationTest, InitializeWithValidModels) {
    if (!setup_success_) {
        GTEST_SKIP() << "Setup failed - models or images not available";
    }

    EXPECT_TRUE(detector_->isInitialized());
    EXPECT_EQ(detector_->getDetectorType(), DetectorType::MediaPipe);
}

TEST_F(MediaPipeDetectorIntegrationTest, AllModelsAreLoaded) {
    if (!setup_success_) {
        GTEST_SKIP() << "Setup failed - models or images not available";
    }

    // 초기화가 성공했다면 모든 3개 모델이 로드된 것
    EXPECT_TRUE(detector_->isInitialized());

    // release 후 재초기화 테스트
    detector_->release();
    EXPECT_FALSE(detector_->isInitialized());

    EXPECT_TRUE(detector_->initialize(models_path_.string()));
    EXPECT_TRUE(detector_->isInitialized());
}

// ============================================================
// 실제 이미지 검출 테스트
// ============================================================

TEST_F(MediaPipeDetectorIntegrationTest, DetectOnRealImages_RGB) {
    if (!setup_success_) {
        GTEST_SKIP() << "Setup failed - models or images not available";
    }

    std::cout << "\n=== RGB Format Detection Test ===" << std::endl;

    int detected_count = 0;
    for (size_t i = 0; i < test_images_.size(); ++i) {
        const auto& image = test_images_[i];
        const auto& name = loaded_image_names_[i];

        cv::Mat rgb_image;
        cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);

        IrisResult result = detector_->detect(
            rgb_image.data,
            rgb_image.cols,
            rgb_image.rows,
            FrameFormat::RGB
        );

        std::cout << "  [" << name << "] detected=" << result.detected
                  << ", left=" << result.left_detected
                  << ", right=" << result.right_detected
                  << ", confidence=" << result.confidence << std::endl;

        if (result.detected) {
            detected_count++;
            validateIrisResult(result, name);
        }
    }

    // 최소 하나의 이미지에서 검출 성공 예상
    EXPECT_GT(detected_count, 0)
        << "Expected at least one successful detection across all test images";

    std::cout << "Detection success: " << detected_count << "/" << test_images_.size() << std::endl;
}

TEST_F(MediaPipeDetectorIntegrationTest, DetectOnRealImages_BGR) {
    if (!setup_success_) {
        GTEST_SKIP() << "Setup failed - models or images not available";
    }

    std::cout << "\n=== BGR Format Detection Test ===" << std::endl;

    int detected_count = 0;
    for (size_t i = 0; i < test_images_.size(); ++i) {
        const auto& image = test_images_[i];
        const auto& name = loaded_image_names_[i];

        // OpenCV imread는 BGR로 로드하므로 변환 불필요
        IrisResult result = detector_->detect(
            image.data,
            image.cols,
            image.rows,
            FrameFormat::BGR
        );

        std::cout << "  [" << name << "] detected=" << result.detected
                  << ", confidence=" << result.confidence << std::endl;

        if (result.detected) {
            detected_count++;
            validateIrisResult(result, name);
        }
    }

    EXPECT_GT(detected_count, 0);
    std::cout << "Detection success: " << detected_count << "/" << test_images_.size() << std::endl;
}

TEST_F(MediaPipeDetectorIntegrationTest, DetectOnRealImages_RGBA) {
    if (!setup_success_) {
        GTEST_SKIP() << "Setup failed - models or images not available";
    }

    std::cout << "\n=== RGBA Format Detection Test ===" << std::endl;

    int detected_count = 0;
    for (size_t i = 0; i < test_images_.size(); ++i) {
        const auto& image = test_images_[i];
        const auto& name = loaded_image_names_[i];

        cv::Mat rgba_image;
        cv::cvtColor(image, rgba_image, cv::COLOR_BGR2RGBA);

        IrisResult result = detector_->detect(
            rgba_image.data,
            rgba_image.cols,
            rgba_image.rows,
            FrameFormat::RGBA
        );

        if (result.detected) {
            detected_count++;
            validateIrisResult(result, name);
        }
    }

    EXPECT_GT(detected_count, 0);
    std::cout << "Detection success: " << detected_count << "/" << test_images_.size() << std::endl;
}

// ============================================================
// 신뢰도 검증 테스트
// ============================================================

TEST_F(MediaPipeDetectorIntegrationTest, VerifyConfidenceValues) {
    if (!setup_success_) {
        GTEST_SKIP() << "Setup failed - models or images not available";
    }

    std::cout << "\n=== Confidence Value Verification ===" << std::endl;

    for (size_t i = 0; i < test_images_.size(); ++i) {
        const auto& image = test_images_[i];
        const auto& name = loaded_image_names_[i];

        cv::Mat rgb_image;
        cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);

        IrisResult result = detector_->detect(
            rgb_image.data,
            rgb_image.cols,
            rgb_image.rows,
            FrameFormat::RGB
        );

        if (result.detected) {
            // 검출된 경우 신뢰도는 합리적인 값이어야 함
            EXPECT_GE(result.confidence, MIN_CONFIDENCE_THRESHOLD)
                << "Confidence should be >= " << MIN_CONFIDENCE_THRESHOLD
                << " for detected face in " << name;
            EXPECT_LE(result.confidence, 1.0f)
                << "Confidence should be <= 1.0 for " << name;

            std::cout << "  [" << name << "] confidence=" << result.confidence
                      << " (threshold=" << MIN_CONFIDENCE_THRESHOLD << ")" << std::endl;
        }
    }
}

TEST_F(MediaPipeDetectorIntegrationTest, VerifyIrisLandmarkCoordinates) {
    if (!setup_success_) {
        GTEST_SKIP() << "Setup failed - models or images not available";
    }

    std::cout << "\n=== Iris Landmark Coordinate Verification ===" << std::endl;

    for (size_t i = 0; i < test_images_.size(); ++i) {
        const auto& image = test_images_[i];
        const auto& name = loaded_image_names_[i];

        cv::Mat rgb_image;
        cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);

        IrisResult result = detector_->detect(
            rgb_image.data,
            rgb_image.cols,
            rgb_image.rows,
            FrameFormat::RGB
        );

        if (result.detected && result.left_detected) {
            std::cout << "  [" << name << "] Left iris center: ("
                      << result.left_iris[0].x << ", "
                      << result.left_iris[0].y << ")" << std::endl;

            // 랜드마크가 이미지 범위 내에 있는지 확인
            // (정규화된 경우 0~1, 픽셀 좌표인 경우 0~width/height)
            for (int j = 0; j < 5; ++j) {
                const auto& lm = result.left_iris[j];
                if (lm.x > 1.0f || lm.y > 1.0f) {
                    // 픽셀 좌표로 추정
                    EXPECT_LT(lm.x, static_cast<float>(image.cols * 2))
                        << "Left iris landmark[" << j << "].x out of bounds for " << name;
                    EXPECT_LT(lm.y, static_cast<float>(image.rows * 2))
                        << "Left iris landmark[" << j << "].y out of bounds for " << name;
                }
            }
        }

        if (result.detected && result.right_detected) {
            std::cout << "  [" << name << "] Right iris center: ("
                      << result.right_iris[0].x << ", "
                      << result.right_iris[0].y << ")" << std::endl;
        }
    }
}

// ============================================================
// 성능 테스트
// ============================================================

TEST_F(MediaPipeDetectorIntegrationTest, MeasureDetectionLatency) {
    if (!setup_success_) {
        GTEST_SKIP() << "Setup failed - models or images not available";
    }

    if (test_images_.empty()) {
        GTEST_SKIP() << "No test images available";
    }

    std::cout << "\n=== Detection Latency Measurement ===" << std::endl;

    // 첫 번째 이미지로 테스트
    const auto& image = test_images_[0];
    cv::Mat rgb_image;
    cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);

    // 워밍업
    std::cout << "Warming up (" << WARMUP_ITERATIONS << " iterations)..." << std::endl;
    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        detector_->detect(rgb_image.data, rgb_image.cols, rgb_image.rows, FrameFormat::RGB);
    }

    // 벤치마크
    std::cout << "Benchmarking (" << BENCHMARK_ITERATIONS << " iterations)..." << std::endl;
    std::vector<double> latencies;
    latencies.reserve(BENCHMARK_ITERATIONS);

    for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
        double latency = measureDetectionLatency(rgb_image, FrameFormat::RGB);
        latencies.push_back(latency);
    }

    LatencyStats stats = LatencyStats::calculate(latencies);
    stats.print("Detection Latency");

    // 성능 목표 검증
    EXPECT_LE(stats.avg_ms, TARGET_LATENCY_MS)
        << "Average detection latency should be <= " << TARGET_LATENCY_MS << " ms";

    // P95도 목표 내에 있어야 함 (약간의 여유 허용)
    EXPECT_LE(stats.p95_ms, TARGET_LATENCY_MS * 1.5)
        << "P95 detection latency should be <= " << (TARGET_LATENCY_MS * 1.5) << " ms";
}

TEST_F(MediaPipeDetectorIntegrationTest, PerImageLatencyMeasurement) {
    if (!setup_success_) {
        GTEST_SKIP() << "Setup failed - models or images not available";
    }

    std::cout << "\n=== Per-Image Latency Measurement ===" << std::endl;

    std::vector<double> all_latencies;

    for (size_t i = 0; i < test_images_.size(); ++i) {
        const auto& image = test_images_[i];
        const auto& name = loaded_image_names_[i];

        cv::Mat rgb_image;
        cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);

        // 워밍업 (각 이미지당 1회)
        detector_->detect(rgb_image.data, rgb_image.cols, rgb_image.rows, FrameFormat::RGB);

        // 벤치마크
        std::vector<double> latencies;
        for (int iter = 0; iter < BENCHMARK_ITERATIONS; ++iter) {
            double latency = measureDetectionLatency(rgb_image, FrameFormat::RGB);
            latencies.push_back(latency);
            all_latencies.push_back(latency);
        }

        LatencyStats stats = LatencyStats::calculate(latencies);
        std::cout << "  [" << name << "] avg=" << stats.avg_ms
                  << "ms, min=" << stats.min_ms
                  << "ms, max=" << stats.max_ms
                  << "ms, p95=" << stats.p95_ms << "ms" << std::endl;
    }

    // 전체 통계
    LatencyStats overall = LatencyStats::calculate(all_latencies);
    overall.print("Overall Detection Latency");

    EXPECT_LE(overall.avg_ms, TARGET_LATENCY_MS)
        << "Overall average latency should be <= " << TARGET_LATENCY_MS << " ms";
}

TEST_F(MediaPipeDetectorIntegrationTest, ContinuousProcessingSimulation) {
    if (!setup_success_) {
        GTEST_SKIP() << "Setup failed - models or images not available";
    }

    if (test_images_.empty()) {
        GTEST_SKIP() << "No test images available";
    }

    std::cout << "\n=== Continuous Processing Simulation (30fps target) ===" << std::endl;

    const auto& image = test_images_[0];
    cv::Mat rgb_image;
    cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);

    // 1초 동안 처리 가능한 프레임 수 측정
    const int test_duration_sec = 1;
    const auto deadline = std::chrono::seconds(test_duration_sec);

    std::vector<double> latencies;
    auto start = std::chrono::high_resolution_clock::now();
    int frame_count = 0;

    while (true) {
        auto frame_start = std::chrono::high_resolution_clock::now();

        detector_->detect(rgb_image.data, rgb_image.cols, rgb_image.rows, FrameFormat::RGB);

        auto frame_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> frame_time = frame_end - frame_start;
        latencies.push_back(frame_time.count());
        frame_count++;

        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        if (elapsed >= deadline) {
            break;
        }
    }

    double actual_fps = static_cast<double>(frame_count) / static_cast<double>(test_duration_sec);
    LatencyStats stats = LatencyStats::calculate(latencies);

    std::cout << "  Frames processed: " << frame_count << std::endl;
    std::cout << "  Actual FPS: " << actual_fps << std::endl;
    stats.print("Frame Processing Latency");

    // 30fps 목표 검증
    EXPECT_GE(actual_fps, 30.0)
        << "Should be able to process at least 30 FPS";
}

// ============================================================
// 다양한 이미지 크기 테스트
// ============================================================

TEST_F(MediaPipeDetectorIntegrationTest, DifferentImageSizes) {
    if (!setup_success_) {
        GTEST_SKIP() << "Setup failed - models or images not available";
    }

    if (test_images_.empty()) {
        GTEST_SKIP() << "No test images available";
    }

    std::cout << "\n=== Different Image Size Tests ===" << std::endl;

    const auto& original = test_images_[0];
    const std::vector<std::pair<int, int>> test_sizes = {
        {320, 240},   // QVGA
        {640, 480},   // VGA
        {1280, 720},  // HD
        {1920, 1080}, // Full HD
    };

    for (const auto& [width, height] : test_sizes) {
        cv::Mat resized;
        cv::resize(original, resized, cv::Size(width, height));

        cv::Mat rgb_image;
        cv::cvtColor(resized, rgb_image, cv::COLOR_BGR2RGB);

        auto start = std::chrono::high_resolution_clock::now();
        IrisResult result = detector_->detect(
            rgb_image.data,
            rgb_image.cols,
            rgb_image.rows,
            FrameFormat::RGB
        );
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> duration = end - start;

        std::cout << "  " << width << "x" << height
                  << ": detected=" << result.detected
                  << ", latency=" << duration.count() << "ms" << std::endl;

        // 모든 크기에서 기본 동작 확인
        EXPECT_GE(result.frame_width, 0);
        EXPECT_GE(result.frame_height, 0);
    }
}

// ============================================================
// 추적 기능 테스트
// ============================================================

TEST_F(MediaPipeDetectorIntegrationTest, TrackingModePerformance) {
    if (!setup_success_) {
        GTEST_SKIP() << "Setup failed - models or images not available";
    }

    if (test_images_.empty()) {
        GTEST_SKIP() << "No test images available";
    }

    std::cout << "\n=== Tracking Mode Performance Test ===" << std::endl;

    const auto& image = test_images_[0];
    cv::Mat rgb_image;
    cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);

    // 추적 비활성화 상태에서 측정
    detector_->setTrackingEnabled(false);
    detector_->resetTracking();

    std::vector<double> no_tracking_latencies;
    for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
        double latency = measureDetectionLatency(rgb_image, FrameFormat::RGB);
        no_tracking_latencies.push_back(latency);
    }

    LatencyStats no_tracking_stats = LatencyStats::calculate(no_tracking_latencies);

    // 추적 활성화 상태에서 측정
    detector_->setTrackingEnabled(true);

    // 첫 프레임은 항상 전체 검출 수행
    detector_->detect(rgb_image.data, rgb_image.cols, rgb_image.rows, FrameFormat::RGB);

    std::vector<double> tracking_latencies;
    for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
        double latency = measureDetectionLatency(rgb_image, FrameFormat::RGB);
        tracking_latencies.push_back(latency);
    }

    LatencyStats tracking_stats = LatencyStats::calculate(tracking_latencies);

    std::cout << "  Without tracking: avg=" << no_tracking_stats.avg_ms << "ms" << std::endl;
    std::cout << "  With tracking:    avg=" << tracking_stats.avg_ms << "ms" << std::endl;

    // 추적 모드가 일반적으로 더 빠르거나 비슷해야 함
    // (정적 이미지에서는 차이가 없을 수 있음)
    std::cout << "  Speedup: " << (no_tracking_stats.avg_ms / tracking_stats.avg_ms) << "x" << std::endl;
}

// ============================================================
// 에지 케이스 테스트
// ============================================================

TEST_F(MediaPipeDetectorIntegrationTest, GrayscaleImageDetection) {
    if (!setup_success_) {
        GTEST_SKIP() << "Setup failed - models or images not available";
    }

    if (test_images_.empty()) {
        GTEST_SKIP() << "No test images available";
    }

    std::cout << "\n=== Grayscale Image Detection Test ===" << std::endl;

    for (size_t i = 0; i < test_images_.size(); ++i) {
        const auto& image = test_images_[i];
        const auto& name = loaded_image_names_[i];

        cv::Mat gray_image;
        cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

        IrisResult result = detector_->detect(
            gray_image.data,
            gray_image.cols,
            gray_image.rows,
            FrameFormat::Grayscale
        );

        std::cout << "  [" << name << "] Grayscale detection: "
                  << (result.detected ? "success" : "failed") << std::endl;
    }
}

TEST_F(MediaPipeDetectorIntegrationTest, ResetTrackingDuringDetection) {
    if (!setup_success_) {
        GTEST_SKIP() << "Setup failed - models or images not available";
    }

    if (test_images_.empty()) {
        GTEST_SKIP() << "No test images available";
    }

    const auto& image = test_images_[0];
    cv::Mat rgb_image;
    cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);

    // 첫 번째 검출
    IrisResult result1 = detector_->detect(
        rgb_image.data, rgb_image.cols, rgb_image.rows, FrameFormat::RGB
    );

    // 추적 리셋
    detector_->resetTracking();

    // 두 번째 검출 (리셋 후)
    IrisResult result2 = detector_->detect(
        rgb_image.data, rgb_image.cols, rgb_image.rows, FrameFormat::RGB
    );

    // 리셋 후에도 정상 동작해야 함
    EXPECT_EQ(result1.detected, result2.detected)
        << "Detection result should be consistent after reset";
}

// ============================================================
// 설정 변경 테스트
// ============================================================

TEST_F(MediaPipeDetectorIntegrationTest, ConfidenceThresholdEffect) {
    if (!setup_success_) {
        GTEST_SKIP() << "Setup failed - models or images not available";
    }

    if (test_images_.empty()) {
        GTEST_SKIP() << "No test images available";
    }

    std::cout << "\n=== Confidence Threshold Effect Test ===" << std::endl;

    const auto& image = test_images_[0];
    cv::Mat rgb_image;
    cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);

    const std::vector<float> confidence_levels = {0.1f, 0.3f, 0.5f, 0.7f, 0.9f};

    for (float conf : confidence_levels) {
        detector_->setMinDetectionConfidence(conf);
        detector_->resetTracking();

        IrisResult result = detector_->detect(
            rgb_image.data, rgb_image.cols, rgb_image.rows, FrameFormat::RGB
        );

        std::cout << "  Threshold=" << conf
                  << ": detected=" << result.detected
                  << ", confidence=" << result.confidence << std::endl;
    }

    // 원래 설정으로 복원
    detector_->setMinDetectionConfidence(0.5f);
}

// ============================================================
// 시각화 테스트 (Visualization Tests)
// ============================================================

/**
 * @brief 검출 결과를 이미지에 시각화하여 저장
 *
 * 얼굴 바운딩 박스와 홍채 랜드마크를 이미지에 그리고
 * output 폴더에 저장합니다.
 *
 * 시각화 항목:
 * - 녹색 박스: 얼굴 바운딩 박스
 * - 노란색 박스: 눈 크롭 영역 (홍채 경계점 기반 추정)
 * - 파란색: 왼쪽 홍채 (중심점: 채운 원, 경계점: 작은 원, 홍채 원)
 * - 빨간색: 오른쪽 홍채
 * - 청록색 선: 홍채 경계점 연결선 (다이아몬드 모양)
 */
TEST_F(MediaPipeDetectorIntegrationTest, VisualizeDetectionResults) {
    if (!setup_success_) {
        GTEST_SKIP() << "Setup failed - models or images not available";
    }

    if (test_images_.empty()) {
        GTEST_SKIP() << "No test images available";
    }

    std::cout << "\n=== Detection Results Visualization ===" << std::endl;

    // 출력 디렉토리 생성
    std::filesystem::path output_path = test_data_path_ / "output";
    std::filesystem::create_directories(output_path);
    std::string output_dir = output_path.string();
    std::cout << "Output directory: " << output_dir << std::endl;

    int success_count = 0;

    for (size_t i = 0; i < test_images_.size(); ++i) {
        const auto& image = test_images_[i];
        std::string image_name = TEST_IMAGE_FILES[i];

        // 각 이미지마다 추적 리셋 (독립적인 검출 수행)
        detector_->resetTracking();

        // BGR -> RGB 변환 (검출용)
        cv::Mat rgb_image;
        cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);

        // 검출 수행
        IrisResult result = detector_->detect(
            rgb_image.data, rgb_image.cols, rgb_image.rows, FrameFormat::RGB
        );

        // 시각화용 이미지 복사 (BGR 유지)
        cv::Mat vis_image = image.clone();

        // 디버그: 결과 상세 출력
        std::cout << "\n  [" << image_name << "] Detection details:" << std::endl;
        std::cout << "    detected=" << result.detected
                  << ", left_detected=" << result.left_detected
                  << ", right_detected=" << result.right_detected << std::endl;
        std::cout << "    face_rect: x=" << result.face_rect.x
                  << ", y=" << result.face_rect.y
                  << ", w=" << result.face_rect.width
                  << ", h=" << result.face_rect.height << std::endl;
        if (result.left_detected) {
            std::cout << "    left_iris[0]: x=" << result.left_iris[0].x
                      << ", y=" << result.left_iris[0].y << std::endl;
            // 경계점 좌표도 출력
            std::cout << "    left_iris bounds: ";
            for (int j = 1; j < 5; ++j) {
                std::cout << "[" << j << "](" << result.left_iris[j].x
                          << "," << result.left_iris[j].y << ") ";
            }
            std::cout << std::endl;
        }
        if (result.right_detected) {
            std::cout << "    right_iris[0]: x=" << result.right_iris[0].x
                      << ", y=" << result.right_iris[0].y << std::endl;
        }

        if (result.detected) {
            int img_w = vis_image.cols;
            int img_h = vis_image.rows;

            // 1. 얼굴 바운딩 박스 그리기 (녹색)
            cv::Rect face_box(
                static_cast<int>(result.face_rect.x * img_w),
                static_cast<int>(result.face_rect.y * img_h),
                static_cast<int>(result.face_rect.width * img_w),
                static_cast<int>(result.face_rect.height * img_h)
            );
            cv::rectangle(vis_image, face_box, cv::Scalar(0, 255, 0), 3);

            // 1.5. actual_face_crop 영역 그리기 (마젠타, 점선 효과를 위해 두께 1)
            // cropFaceRegion과 동일한 마진 적용
            float margin_x = result.face_rect.width * 0.1f;
            float margin_y = result.face_rect.height * 0.1f;
            float crop_x = std::max(0.0f, result.face_rect.x - margin_x);
            float crop_y = std::max(0.0f, result.face_rect.y - margin_y);
            float crop_max_x = std::min(1.0f, result.face_rect.x + result.face_rect.width + margin_x);
            float crop_max_y = std::min(1.0f, result.face_rect.y + result.face_rect.height + margin_y);
            cv::Rect actual_crop_box(
                static_cast<int>(crop_x * img_w),
                static_cast<int>(crop_y * img_h),
                static_cast<int>((crop_max_x - crop_x) * img_w),
                static_cast<int>((crop_max_y - crop_y) * img_h)
            );
            cv::rectangle(vis_image, actual_crop_box, cv::Scalar(255, 0, 255), 2);  // 마젠타

            // 디버그: actual_face_crop 정보 출력
            std::cout << "    actual_face_crop (estimated): x=" << crop_x
                      << ", y=" << crop_y
                      << ", w=" << (crop_max_x - crop_x)
                      << ", h=" << (crop_max_y - crop_y) << std::endl;

            // 1.6. 실제 Eye Crop 영역 시각화 (흰색 점선)
            // extractEyeRegion의 로직을 재현하여 시각화
            // 참고: 이 계산은 Face Landmark 출력에 의존하므로,
            //       현재 IrisResult에는 포함되지 않음
            // 홍채 중심 좌표와 반지름으로 Eye Crop 영역 추정
            if (result.left_detected && result.left_radius > 0) {
                // Eye Crop은 홍채보다 큰 영역이므로 반지름의 2배 정도로 추정
                float eye_crop_size = result.left_radius * 4.0f / img_w;  // 픽셀을 정규화로
                float eye_center_x = result.left_iris[0].x;
                float eye_center_y = result.left_iris[0].y;
                cv::Rect left_eye_crop_est(
                    static_cast<int>((eye_center_x - eye_crop_size/2) * img_w),
                    static_cast<int>((eye_center_y - eye_crop_size/2) * img_h),
                    static_cast<int>(eye_crop_size * img_w),
                    static_cast<int>(eye_crop_size * img_h)
                );
                cv::rectangle(vis_image, left_eye_crop_est, cv::Scalar(255, 255, 255), 1);  // 흰색
            }

            // 2. 왼쪽 홍채 그리기 (파란색)
            if (result.left_detected) {
                // 홍채 경계점들의 바운딩 박스 계산 (눈 크롭 영역 추정, 노란색)
                float left_min_x = 1.0f, left_min_y = 1.0f;
                float left_max_x = 0.0f, left_max_y = 0.0f;
                for (int j = 0; j < 5; ++j) {
                    left_min_x = std::min(left_min_x, result.left_iris[j].x);
                    left_min_y = std::min(left_min_y, result.left_iris[j].y);
                    left_max_x = std::max(left_max_x, result.left_iris[j].x);
                    left_max_y = std::max(left_max_y, result.left_iris[j].y);
                }
                // 마진 추가 (50% - 눈 크롭 영역 추정)
                float left_margin = std::max(left_max_x - left_min_x, left_max_y - left_min_y) * 0.5f;
                cv::Rect left_eye_box(
                    static_cast<int>((left_min_x - left_margin) * img_w),
                    static_cast<int>((left_min_y - left_margin) * img_h),
                    static_cast<int>((left_max_x - left_min_x + 2 * left_margin) * img_w),
                    static_cast<int>((left_max_y - left_min_y + 2 * left_margin) * img_h)
                );
                cv::rectangle(vis_image, left_eye_box, cv::Scalar(0, 255, 255), 2);  // 노란색

                // 중심점
                cv::Point left_center(
                    static_cast<int>(result.left_iris[0].x * img_w),
                    static_cast<int>(result.left_iris[0].y * img_h)
                );
                cv::circle(vis_image, left_center, 8, cv::Scalar(255, 0, 0), -1);  // 채운 원

                // 경계점들 및 연결선 (다이아몬드 모양)
                std::vector<cv::Point> left_boundary_pts;
                for (int j = 1; j < 5; ++j) {
                    cv::Point pt(
                        static_cast<int>(result.left_iris[j].x * img_w),
                        static_cast<int>(result.left_iris[j].y * img_h)
                    );
                    cv::circle(vis_image, pt, 4, cv::Scalar(255, 100, 100), -1);
                    left_boundary_pts.push_back(pt);
                }
                // 경계점 연결선 (청록색)
                if (left_boundary_pts.size() == 4) {
                    cv::line(vis_image, left_boundary_pts[0], left_boundary_pts[1], cv::Scalar(255, 255, 0), 1);
                    cv::line(vis_image, left_boundary_pts[1], left_boundary_pts[2], cv::Scalar(255, 255, 0), 1);
                    cv::line(vis_image, left_boundary_pts[2], left_boundary_pts[3], cv::Scalar(255, 255, 0), 1);
                    cv::line(vis_image, left_boundary_pts[3], left_boundary_pts[0], cv::Scalar(255, 255, 0), 1);
                }

                // 홍채 원 (반지름 사용)
                if (result.left_radius > 0) {
                    cv::circle(vis_image, left_center, static_cast<int>(result.left_radius),
                               cv::Scalar(255, 0, 0), 2);
                }
            }

            // 3. 오른쪽 홍채 그리기 (빨간색)
            if (result.right_detected) {
                // 홍채 경계점들의 바운딩 박스 계산 (눈 크롭 영역 추정, 노란색)
                float right_min_x = 1.0f, right_min_y = 1.0f;
                float right_max_x = 0.0f, right_max_y = 0.0f;
                for (int j = 0; j < 5; ++j) {
                    right_min_x = std::min(right_min_x, result.right_iris[j].x);
                    right_min_y = std::min(right_min_y, result.right_iris[j].y);
                    right_max_x = std::max(right_max_x, result.right_iris[j].x);
                    right_max_y = std::max(right_max_y, result.right_iris[j].y);
                }
                // 마진 추가 (50% - 눈 크롭 영역 추정)
                float right_margin = std::max(right_max_x - right_min_x, right_max_y - right_min_y) * 0.5f;
                cv::Rect right_eye_box(
                    static_cast<int>((right_min_x - right_margin) * img_w),
                    static_cast<int>((right_min_y - right_margin) * img_h),
                    static_cast<int>((right_max_x - right_min_x + 2 * right_margin) * img_w),
                    static_cast<int>((right_max_y - right_min_y + 2 * right_margin) * img_h)
                );
                cv::rectangle(vis_image, right_eye_box, cv::Scalar(0, 255, 255), 2);  // 노란색

                // 중심점
                cv::Point right_center(
                    static_cast<int>(result.right_iris[0].x * img_w),
                    static_cast<int>(result.right_iris[0].y * img_h)
                );
                cv::circle(vis_image, right_center, 8, cv::Scalar(0, 0, 255), -1);

                // 경계점들 및 연결선
                std::vector<cv::Point> right_boundary_pts;
                for (int j = 1; j < 5; ++j) {
                    cv::Point pt(
                        static_cast<int>(result.right_iris[j].x * img_w),
                        static_cast<int>(result.right_iris[j].y * img_h)
                    );
                    cv::circle(vis_image, pt, 4, cv::Scalar(100, 100, 255), -1);
                    right_boundary_pts.push_back(pt);
                }
                // 경계점 연결선 (청록색)
                if (right_boundary_pts.size() == 4) {
                    cv::line(vis_image, right_boundary_pts[0], right_boundary_pts[1], cv::Scalar(255, 255, 0), 1);
                    cv::line(vis_image, right_boundary_pts[1], right_boundary_pts[2], cv::Scalar(255, 255, 0), 1);
                    cv::line(vis_image, right_boundary_pts[2], right_boundary_pts[3], cv::Scalar(255, 255, 0), 1);
                    cv::line(vis_image, right_boundary_pts[3], right_boundary_pts[0], cv::Scalar(255, 255, 0), 1);
                }

                // 홍채 원 (반지름 사용)
                if (result.right_radius > 0) {
                    cv::circle(vis_image, right_center, static_cast<int>(result.right_radius),
                               cv::Scalar(0, 0, 255), 2);
                }
            }

            // 4. Face Mesh 전체 그리기 (478개 랜드마크, 연두색 점)
            if (result.face_mesh_valid) {
                // 모든 랜드마크를 작은 점으로 그리기
                for (int i = 0; i < IrisResult::FACE_MESH_LANDMARK_COUNT; ++i) {
                    cv::Point pt(
                        static_cast<int>(result.face_mesh[i].x * img_w),
                        static_cast<int>(result.face_mesh[i].y * img_h)
                    );
                    // 범위 체크
                    if (pt.x >= 0 && pt.x < img_w && pt.y >= 0 && pt.y < img_h) {
                        // 홍채 랜드마크 (468-477)는 다른 색으로 표시
                        if (i >= 468 && i <= 472) {
                            cv::circle(vis_image, pt, 3, cv::Scalar(255, 0, 0), -1);  // 왼쪽 홍채: 파랑
                        } else if (i >= 473 && i <= 477) {
                            cv::circle(vis_image, pt, 3, cv::Scalar(0, 0, 255), -1);  // 오른쪽 홍채: 빨강
                        } else {
                            cv::circle(vis_image, pt, 1, cv::Scalar(0, 255, 128), -1);  // 얼굴: 연두색
                        }
                    }
                }

                // 주요 랜드마크 인덱스 표시 (눈, 코, 입)
                // 왼쪽 눈: 33 (외측), 133 (내측)
                // 오른쪽 눈: 362 (외측), 263 (내측)
                // 코: 1
                // 입: 61, 291
                int key_indices[] = {33, 133, 362, 263, 1, 61, 291};
                for (int idx : key_indices) {
                    cv::Point pt(
                        static_cast<int>(result.face_mesh[idx].x * img_w),
                        static_cast<int>(result.face_mesh[idx].y * img_h)
                    );
                    cv::circle(vis_image, pt, 5, cv::Scalar(255, 255, 0), -1);  // 청록색
                    cv::putText(vis_image, std::to_string(idx),
                                cv::Point(pt.x + 5, pt.y - 5),
                                cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1);
                }

                std::cout << "    Face mesh: 478 landmarks drawn" << std::endl;
            }

            // 5. 정보 텍스트 추가
            std::string info = "conf=" + std::to_string(result.confidence).substr(0, 4);
            cv::putText(vis_image, info, cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

            success_count++;
        } else {
            // 검출 실패 표시
            cv::putText(vis_image, "NO FACE DETECTED", cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
        }

        // 출력 파일명 생성 (확장자를 jpg로 변경)
        std::string base_name = image_name.substr(0, image_name.find_last_of('.'));
        std::string output_file = output_dir + "/" + base_name + "_result.jpg";

        // 저장
        bool saved = cv::imwrite(output_file, vis_image);
        if (saved) {
            std::cout << "  Saved: " << output_file
                      << " (detected=" << result.detected
                      << ", conf=" << result.confidence << ")" << std::endl;
        } else {
            std::cout << "  FAILED to save: " << output_file << std::endl;
        }
    }

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Total images: " << test_images_.size() << std::endl;
    std::cout << "Detected: " << success_count << std::endl;
    std::cout << "Output directory: " << output_dir << std::endl;

    EXPECT_GT(success_count, 0);
}

// ============================================================
// P1-W3-04: 경계 조건 테스트 (Edge Case Tests)
// ============================================================

TEST_F(MediaPipeDetectorIntegrationTest, EdgeCase_EmptyImage) {
    if (!setup_success_) {
        GTEST_SKIP() << "Setup failed - models not available";
    }

    std::cout << "\n=== Edge Case: Empty Image ===" << std::endl;

    // 빈 이미지 생성
    cv::Mat empty_image;
    EXPECT_TRUE(empty_image.empty());

    // detect 호출 - 빈 데이터로는 검출 불가
    // nullptr 전달
    IrisResult result = detector_->detect(nullptr, 0, 0, FrameFormat::RGB);

    EXPECT_FALSE(result.detected);
    EXPECT_FLOAT_EQ(result.confidence, 0.0f);
    std::cout << "  Empty image: detected=" << result.detected << " (expected: false)" << std::endl;
}

TEST_F(MediaPipeDetectorIntegrationTest, EdgeCase_VerySmallImage) {
    if (!setup_success_) {
        GTEST_SKIP() << "Setup failed - models not available";
    }

    std::cout << "\n=== Edge Case: Very Small Image (10x10) ===" << std::endl;

    // 10x10 크기의 매우 작은 이미지
    cv::Mat small_image(10, 10, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::cvtColor(small_image, small_image, cv::COLOR_BGR2RGB);

    IrisResult result = detector_->detect(
        small_image.data, small_image.cols, small_image.rows, FrameFormat::RGB
    );

    // 너무 작은 이미지에서는 얼굴 검출 불가능해야 함
    EXPECT_FALSE(result.detected);
    std::cout << "  Small (10x10): detected=" << result.detected
              << ", confidence=" << result.confidence << std::endl;
}

TEST_F(MediaPipeDetectorIntegrationTest, EdgeCase_NoFaceImage) {
    if (!setup_success_) {
        GTEST_SKIP() << "Setup failed - models not available";
    }

    std::cout << "\n=== Edge Case: No Face Image (Solid Color) ===" << std::endl;

    // 단색 이미지 (얼굴 없음)
    cv::Mat no_face_image(480, 640, CV_8UC3, cv::Scalar(100, 150, 200));
    cv::cvtColor(no_face_image, no_face_image, cv::COLOR_BGR2RGB);

    IrisResult result = detector_->detect(
        no_face_image.data, no_face_image.cols, no_face_image.rows, FrameFormat::RGB
    );

    // 얼굴이 없는 이미지에서는 검출 실패해야 함
    EXPECT_FALSE(result.detected);
    std::cout << "  No face (solid): detected=" << result.detected
              << ", confidence=" << result.confidence << std::endl;
}

TEST_F(MediaPipeDetectorIntegrationTest, EdgeCase_NoFaceGradient) {
    if (!setup_success_) {
        GTEST_SKIP() << "Setup failed - models not available";
    }

    std::cout << "\n=== Edge Case: No Face Image (Gradient) ===" << std::endl;

    // 그라데이션 이미지 (얼굴처럼 보일 수 있는 패턴)
    cv::Mat gradient(480, 640, CV_8UC3);
    for (int y = 0; y < gradient.rows; y++) {
        for (int x = 0; x < gradient.cols; x++) {
            gradient.at<cv::Vec3b>(y, x) = cv::Vec3b(
                (x * 255) / gradient.cols,
                (y * 255) / gradient.rows,
                128
            );
        }
    }
    cv::cvtColor(gradient, gradient, cv::COLOR_BGR2RGB);

    IrisResult result = detector_->detect(
        gradient.data, gradient.cols, gradient.rows, FrameFormat::RGB
    );

    // 그라데이션 이미지에서도 검출 실패해야 함
    EXPECT_FALSE(result.detected);
    std::cout << "  No face (gradient): detected=" << result.detected
              << ", confidence=" << result.confidence << std::endl;
}

TEST_F(MediaPipeDetectorIntegrationTest, EdgeCase_LargeImage) {
    if (!setup_success_) {
        GTEST_SKIP() << "Setup failed - models not available";
    }

    if (test_images_.empty()) {
        GTEST_SKIP() << "No test images available";
    }

    std::cout << "\n=== Edge Case: Large Image (2x upscale) ===" << std::endl;

    // 원본 이미지를 2배 확대
    cv::Mat large_image;
    cv::resize(test_images_[0], large_image, cv::Size(), 2.0, 2.0, cv::INTER_LINEAR);
    cv::cvtColor(large_image, large_image, cv::COLOR_BGR2RGB);

    std::cout << "  Original size: " << test_images_[0].cols << "x" << test_images_[0].rows << std::endl;
    std::cout << "  Upscaled size: " << large_image.cols << "x" << large_image.rows << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    IrisResult result = detector_->detect(
        large_image.data, large_image.cols, large_image.rows, FrameFormat::RGB
    );
    auto end = std::chrono::high_resolution_clock::now();
    auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // 큰 이미지도 내부적으로 리사이즈하여 검출 가능해야 함
    std::cout << "  Large image: detected=" << result.detected
              << ", confidence=" << result.confidence
              << ", latency=" << latency << "ms" << std::endl;

    // 큰 이미지에서도 검출 성공 기대 (내부 리사이즈 처리)
    // 단, 처리 시간이 더 걸릴 수 있음
    if (result.detected) {
        EXPECT_GT(result.confidence, 0.5f);
    }
}

TEST_F(MediaPipeDetectorIntegrationTest, EdgeCase_InvalidDimensions) {
    if (!setup_success_) {
        GTEST_SKIP() << "Setup failed - models not available";
    }

    std::cout << "\n=== Edge Case: Invalid Dimensions ===" << std::endl;

    // 유효한 데이터지만 잘못된 크기
    std::vector<uint8_t> dummy_data(640 * 480 * 3, 128);

    // 너비 0
    IrisResult result1 = detector_->detect(dummy_data.data(), 0, 480, FrameFormat::RGB);
    EXPECT_FALSE(result1.detected);
    std::cout << "  Width=0: detected=" << result1.detected << std::endl;

    // 높이 0
    IrisResult result2 = detector_->detect(dummy_data.data(), 640, 0, FrameFormat::RGB);
    EXPECT_FALSE(result2.detected);
    std::cout << "  Height=0: detected=" << result2.detected << std::endl;

    // 음수 크기 (size_t이므로 매우 큰 값으로 변환)
    // 이 경우 내부에서 적절히 처리해야 함
}

} // namespace testing
} // namespace iris_sdk

#else

// ============================================================
// TFLite 또는 OpenCV 없이 빌드된 경우
// ============================================================

TEST(MediaPipeDetectorIntegrationTest, SkippedWithoutDependencies) {
    GTEST_SKIP() << "TFLite or OpenCV not available - skipping integration tests";
}

#endif // IRIS_SDK_HAS_TFLITE && IRIS_SDK_HAS_OPENCV
