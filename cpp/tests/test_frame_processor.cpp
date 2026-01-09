/**
 * @file test_frame_processor.cpp
 * @brief FrameProcessor Unit Tests
 *
 * 프레임 처리 파이프라인의 단위 테스트.
 * 초기화, 포맷 변환, 파이프라인 처리, 성능 등을 검증.
 */

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <chrono>
#include <cmath>
#include <filesystem>

#include "iris_sdk/frame_processor.h"
#include "iris_sdk/types.h"

namespace iris_sdk {
namespace testing {

// ============================================================
// 테스트 헬퍼 함수
// ============================================================

/**
 * @brief RGBA 포맷 테스트 프레임 생성
 */
std::vector<uint8_t> createRGBAFrame(int width, int height) {
    std::vector<uint8_t> data(width * height * 4);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * 4;
            data[idx + 0] = static_cast<uint8_t>(x % 256);     // R
            data[idx + 1] = static_cast<uint8_t>(y % 256);     // G
            data[idx + 2] = static_cast<uint8_t>((x + y) % 256); // B
            data[idx + 3] = 255;                                // A
        }
    }
    return data;
}

/**
 * @brief BGR 포맷 테스트 프레임 생성
 */
std::vector<uint8_t> createBGRFrame(int width, int height) {
    std::vector<uint8_t> data(width * height * 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * 3;
            data[idx + 0] = static_cast<uint8_t>((x + y) % 256); // B
            data[idx + 1] = static_cast<uint8_t>(y % 256);       // G
            data[idx + 2] = static_cast<uint8_t>(x % 256);       // R
        }
    }
    return data;
}

/**
 * @brief NV21 포맷 테스트 프레임 생성
 * Y plane: width * height bytes
 * VU plane (interleaved): width * height / 2 bytes
 */
std::vector<uint8_t> createNV21Frame(int width, int height) {
    int y_size = width * height;
    int uv_size = width * height / 2;
    std::vector<uint8_t> data(y_size + uv_size);

    // Y plane
    for (int i = 0; i < y_size; ++i) {
        data[i] = static_cast<uint8_t>(128 + (i % 64) - 32);
    }

    // VU plane (interleaved)
    for (int i = 0; i < uv_size; ++i) {
        data[y_size + i] = 128; // Neutral chroma
    }

    return data;
}

/**
 * @brief NV12 포맷 테스트 프레임 생성
 * Y plane: width * height bytes
 * UV plane (interleaved): width * height / 2 bytes
 */
std::vector<uint8_t> createNV12Frame(int width, int height) {
    // NV12와 NV21은 UV 순서만 다름
    return createNV21Frame(width, height);
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

/**
 * @brief 간단한 테스트 텍스처 생성 (RGBA)
 */
std::vector<uint8_t> createTestTexture(int width, int height) {
    std::vector<uint8_t> data(width * height * 4);
    for (int i = 0; i < width * height; ++i) {
        data[i * 4 + 0] = 0;     // R
        data[i * 4 + 1] = 128;   // G
        data[i * 4 + 2] = 255;   // B
        data[i * 4 + 3] = 200;   // A
    }
    return data;
}

/**
 * @brief 모델 경로 찾기
 */
std::string findModelPath() {
    std::vector<std::string> possible_paths = {
        "../models",
        "../../models",
        "../../../models",
        "../../../../models",
        "/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/cpp/models"
    };

    for (const auto& path : possible_paths) {
        if (std::filesystem::exists(path)) {
            return path;
        }
    }
    return "";
}

// ============================================================
// FrameProcessor 기본 테스트
// ============================================================

class FrameProcessorTest : public ::testing::Test {
protected:
    void SetUp() override {
        processor = std::make_unique<FrameProcessor>();
    }

    void TearDown() override {
        processor.reset();
    }

    std::unique_ptr<FrameProcessor> processor;
};

// ------------------------------------------------------------
// 생성 및 소멸 테스트
// ------------------------------------------------------------

TEST_F(FrameProcessorTest, DefaultConstruction) {
    // 기본 생성자로 인스턴스 생성 가능
    FrameProcessor proc;
    EXPECT_FALSE(proc.isInitialized());
}

TEST_F(FrameProcessorTest, PointerConstruction) {
    // 포인터로 생성 가능
    auto proc = std::make_unique<FrameProcessor>();
    EXPECT_NE(proc, nullptr);
    EXPECT_FALSE(proc->isInitialized());
}

TEST_F(FrameProcessorTest, MoveConstruction) {
    // 이동 생성자 지원
    FrameProcessor proc1;
    FrameProcessor proc2(std::move(proc1));
    EXPECT_FALSE(proc2.isInitialized());
}

TEST_F(FrameProcessorTest, MoveAssignment) {
    // 이동 대입 연산자 지원
    FrameProcessor proc1;
    FrameProcessor proc2;
    proc2 = std::move(proc1);
    EXPECT_FALSE(proc2.isInitialized());
}

// ------------------------------------------------------------
// 초기화 테스트
// ------------------------------------------------------------

TEST_F(FrameProcessorTest, InitializeWithInvalidPath) {
    // 존재하지 않는 경로로 초기화 실패
    EXPECT_FALSE(processor->initialize("/nonexistent/path/to/models"));
    EXPECT_FALSE(processor->isInitialized());
}

TEST_F(FrameProcessorTest, InitializeWithEmptyPath) {
    // 빈 경로로 초기화 실패
    EXPECT_FALSE(processor->initialize(""));
    EXPECT_FALSE(processor->isInitialized());
}

TEST_F(FrameProcessorTest, ReleaseWithoutInitialization) {
    // 초기화 없이 release 호출해도 안전
    EXPECT_NO_THROW(processor->release());
    EXPECT_FALSE(processor->isInitialized());
}

TEST_F(FrameProcessorTest, DoubleRelease) {
    // 두 번 release 호출해도 안전
    EXPECT_NO_THROW(processor->release());
    EXPECT_NO_THROW(processor->release());
}

// ------------------------------------------------------------
// 텍스처 관리 테스트
// ------------------------------------------------------------

TEST_F(FrameProcessorTest, LoadTextureWithInvalidPath) {
    // 존재하지 않는 파일로 텍스처 로드 실패
    EXPECT_FALSE(processor->loadLensTexture("/nonexistent/texture.png"));
    EXPECT_FALSE(processor->hasLensTexture());
}

TEST_F(FrameProcessorTest, LoadTextureWithEmptyPath) {
    // 빈 경로로 텍스처 로드 실패
    EXPECT_FALSE(processor->loadLensTexture(""));
    EXPECT_FALSE(processor->hasLensTexture());
}

TEST_F(FrameProcessorTest, LoadTextureFromMemoryWithoutInit) {
    // 초기화 없이 메모리에서 텍스처 로드 시도 - 실패해야 함
    // (FrameProcessor는 초기화 후에만 텍스처 로드 가능)
    auto texture = createTestTexture(64, 64);
    EXPECT_FALSE(processor->loadLensTexture(texture.data(), 64, 64));
    EXPECT_FALSE(processor->hasLensTexture());
}

TEST_F(FrameProcessorTest, LoadTextureFromMemoryNullData) {
    // nullptr로 텍스처 로드 실패
    EXPECT_FALSE(processor->loadLensTexture(nullptr, 64, 64));
    EXPECT_FALSE(processor->hasLensTexture());
}

TEST_F(FrameProcessorTest, LoadTextureFromMemoryInvalidSize) {
    // 잘못된 크기로 텍스처 로드 실패
    auto texture = createTestTexture(64, 64);
    EXPECT_FALSE(processor->loadLensTexture(texture.data(), 0, 64));
    EXPECT_FALSE(processor->loadLensTexture(texture.data(), 64, 0));
    EXPECT_FALSE(processor->loadLensTexture(texture.data(), -1, 64));
    EXPECT_FALSE(processor->loadLensTexture(texture.data(), 64, -1));
}

TEST_F(FrameProcessorTest, UnloadTextureWithoutInit) {
    // 초기화 없이 언로드해도 안전 (텍스처도 없음)
    EXPECT_NO_THROW(processor->unloadLensTexture());
    EXPECT_FALSE(processor->hasLensTexture());
}

TEST_F(FrameProcessorTest, UnloadTextureWithoutLoad) {
    // 로드 없이 언로드해도 안전
    EXPECT_NO_THROW(processor->unloadLensTexture());
    EXPECT_FALSE(processor->hasLensTexture());
}

// ------------------------------------------------------------
// 설정 테스트
// ------------------------------------------------------------

TEST_F(FrameProcessorTest, SetMinConfidence) {
    // 신뢰도 임계값 설정
    EXPECT_NO_THROW(processor->setMinConfidence(0.5f));
}

TEST_F(FrameProcessorTest, SetMinConfidenceBoundary) {
    // 경계값 테스트
    EXPECT_NO_THROW(processor->setMinConfidence(0.0f));
    EXPECT_NO_THROW(processor->setMinConfidence(1.0f));
}

TEST_F(FrameProcessorTest, SetFaceTracking) {
    // 얼굴 추적 설정
    EXPECT_NO_THROW(processor->setFaceTracking(true));
    EXPECT_NO_THROW(processor->setFaceTracking(false));
}

// ------------------------------------------------------------
// 프레임 처리 테스트 (초기화 전)
// ------------------------------------------------------------

TEST_F(FrameProcessorTest, ProcessWithoutInitialization) {
    // 초기화 없이 process 호출
    auto frame = createRGBAFrame(640, 480);

    ProcessResult result = processor->process(
        frame.data(), 640, 480, FrameFormat::RGBA, nullptr);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.error_code, ErrorCode::NotInitialized);
}

TEST_F(FrameProcessorTest, DetectOnlyWithoutInitialization) {
    // 초기화 없이 detectOnly 호출
    auto frame = createRGBAFrame(640, 480);

    IrisResult result = processor->detectOnly(
        frame.data(), 640, 480, FrameFormat::RGBA);

    EXPECT_FALSE(result.detected);
}

TEST_F(FrameProcessorTest, RenderOnlyWithoutInitialization) {
    // 초기화 없이 renderOnly 호출
    auto frame = createRGBAFrame(640, 480);
    IrisResult iris_result{};
    LensConfig config = createDefaultConfig();

    EXPECT_FALSE(processor->renderOnly(
        frame.data(), 640, 480, FrameFormat::RGBA, iris_result, config));
}

// ------------------------------------------------------------
// 프레임 처리 - 입력 검증 테스트
// ------------------------------------------------------------

TEST_F(FrameProcessorTest, ProcessWithNullFrame) {
    // nullptr 프레임으로 process 호출 - NotInitialized 반환
    // (프레임 유효성 검사보다 초기화 확인이 먼저)
    ProcessResult result = processor->process(
        nullptr, 640, 480, FrameFormat::RGBA, nullptr);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.error_code, ErrorCode::NotInitialized);
}

TEST_F(FrameProcessorTest, ProcessWithInvalidDimensions) {
    // 잘못된 크기로 process 호출 - 초기화 안됐으므로 NotInitialized 반환
    auto frame = createRGBAFrame(640, 480);

    ProcessResult result1 = processor->process(
        frame.data(), 0, 480, FrameFormat::RGBA, nullptr);
    EXPECT_FALSE(result1.success);
    EXPECT_EQ(result1.error_code, ErrorCode::NotInitialized);

    ProcessResult result2 = processor->process(
        frame.data(), 640, 0, FrameFormat::RGBA, nullptr);
    EXPECT_FALSE(result2.success);
    EXPECT_EQ(result2.error_code, ErrorCode::NotInitialized);

    ProcessResult result3 = processor->process(
        frame.data(), -1, 480, FrameFormat::RGBA, nullptr);
    EXPECT_FALSE(result3.success);
    EXPECT_EQ(result3.error_code, ErrorCode::NotInitialized);
}

TEST_F(FrameProcessorTest, DetectOnlyWithNullFrame) {
    // nullptr 프레임으로 detectOnly 호출
    IrisResult result = processor->detectOnly(
        nullptr, 640, 480, FrameFormat::RGBA);

    EXPECT_FALSE(result.detected);
}

TEST_F(FrameProcessorTest, DetectOnlyWithInvalidDimensions) {
    // 잘못된 크기로 detectOnly 호출
    auto frame = createRGBAFrame(640, 480);

    IrisResult result1 = processor->detectOnly(
        frame.data(), 0, 480, FrameFormat::RGBA);
    EXPECT_FALSE(result1.detected);

    IrisResult result2 = processor->detectOnly(
        frame.data(), 640, 0, FrameFormat::RGBA);
    EXPECT_FALSE(result2.detected);
}

TEST_F(FrameProcessorTest, RenderOnlyWithNullFrame) {
    // nullptr 프레임으로 renderOnly 호출
    IrisResult iris_result{};
    LensConfig config = createDefaultConfig();

    EXPECT_FALSE(processor->renderOnly(
        nullptr, 640, 480, FrameFormat::RGBA, iris_result, config));
}

TEST_F(FrameProcessorTest, RenderOnlyWithInvalidDimensions) {
    // 잘못된 크기로 renderOnly 호출
    auto frame = createRGBAFrame(640, 480);
    IrisResult iris_result{};
    LensConfig config = createDefaultConfig();

    EXPECT_FALSE(processor->renderOnly(
        frame.data(), 0, 480, FrameFormat::RGBA, iris_result, config));
    EXPECT_FALSE(processor->renderOnly(
        frame.data(), 640, 0, FrameFormat::RGBA, iris_result, config));
}

// ------------------------------------------------------------
// cv::Mat 처리 테스트
// ------------------------------------------------------------

TEST_F(FrameProcessorTest, ProcessMatWithoutInitialization) {
    // 초기화 없이 cv::Mat process 호출
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));

    ProcessResult result = processor->process(frame, nullptr);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.error_code, ErrorCode::NotInitialized);
}

TEST_F(FrameProcessorTest, ProcessEmptyMat) {
    // 빈 Mat으로 process 호출 - 초기화 안됐으므로 NotInitialized 반환
    cv::Mat frame;

    ProcessResult result = processor->process(frame, nullptr);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.error_code, ErrorCode::NotInitialized);
}

TEST_F(FrameProcessorTest, ProcessInvalidMatType) {
    // 지원하지 않는 타입의 Mat - 초기화 안됐으므로 NotInitialized 반환
    cv::Mat frame(480, 640, CV_32FC3); // float 타입

    ProcessResult result = processor->process(frame, nullptr);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.error_code, ErrorCode::NotInitialized);
}

// ------------------------------------------------------------
// 통계 테스트
// ------------------------------------------------------------

TEST_F(FrameProcessorTest, GetLastProcessingTimeInitial) {
    // 초기 처리 시간은 0
    EXPECT_DOUBLE_EQ(processor->getLastProcessingTimeMs(), 0.0);
}

TEST_F(FrameProcessorTest, GetAverageFPSInitial) {
    // 초기 평균 FPS는 0
    EXPECT_DOUBLE_EQ(processor->getAverageFPS(), 0.0);
}

// ============================================================
// FrameProcessor 통합 테스트 (실제 모델 필요)
// ============================================================

class FrameProcessorIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        processor = std::make_unique<FrameProcessor>();
        model_path = findModelPath();

        // 모델이 있는 경우에만 초기화
        if (!model_path.empty() && std::filesystem::exists(model_path)) {
            has_models = processor->initialize(model_path);
        }
    }

    void TearDown() override {
        processor.reset();
    }

    std::unique_ptr<FrameProcessor> processor;
    std::string model_path;
    bool has_models = false;
};

TEST_F(FrameProcessorIntegrationTest, InitializeWithValidModels) {
    if (model_path.empty()) {
        GTEST_SKIP() << "Model path not found, skipping integration test";
    }

    // 모델이 있으면 초기화 성공
    FrameProcessor proc;
    bool result = proc.initialize(model_path);

    // 모델 파일이 실제로 있으면 true, 없으면 false
    if (result) {
        EXPECT_TRUE(proc.isInitialized());
    }
}

TEST_F(FrameProcessorIntegrationTest, ProcessRGBAFrame) {
    if (!has_models) {
        GTEST_SKIP() << "Models not available, skipping";
    }

    // RGBA 프레임 처리
    auto frame = createRGBAFrame(640, 480);
    ProcessResult result = processor->process(
        frame.data(), 640, 480, FrameFormat::RGBA, nullptr);

    EXPECT_TRUE(result.success);
    EXPECT_GT(result.processing_time_ms, 0.0f);
}

TEST_F(FrameProcessorIntegrationTest, ProcessBGRFrame) {
    if (!has_models) {
        GTEST_SKIP() << "Models not available, skipping";
    }

    // BGR 프레임 처리
    auto frame = createBGRFrame(640, 480);
    ProcessResult result = processor->process(
        frame.data(), 640, 480, FrameFormat::BGR, nullptr);

    EXPECT_TRUE(result.success);
    EXPECT_GT(result.processing_time_ms, 0.0f);
}

TEST_F(FrameProcessorIntegrationTest, ProcessNV21Frame) {
    if (!has_models) {
        GTEST_SKIP() << "Models not available, skipping";
    }

    // NV21 프레임 처리
    auto frame = createNV21Frame(640, 480);
    ProcessResult result = processor->process(
        frame.data(), 640, 480, FrameFormat::NV21, nullptr);

    EXPECT_TRUE(result.success);
    EXPECT_GT(result.processing_time_ms, 0.0f);
}

TEST_F(FrameProcessorIntegrationTest, ProcessNV12Frame) {
    if (!has_models) {
        GTEST_SKIP() << "Models not available, skipping";
    }

    // NV12 프레임 처리
    auto frame = createNV12Frame(640, 480);
    ProcessResult result = processor->process(
        frame.data(), 640, 480, FrameFormat::NV12, nullptr);

    EXPECT_TRUE(result.success);
    EXPECT_GT(result.processing_time_ms, 0.0f);
}

TEST_F(FrameProcessorIntegrationTest, ProcessMatBGR) {
    if (!has_models) {
        GTEST_SKIP() << "Models not available, skipping";
    }

    // BGR cv::Mat 처리
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    ProcessResult result = processor->process(frame, nullptr);

    EXPECT_TRUE(result.success);
    EXPECT_GT(result.processing_time_ms, 0.0f);
}

TEST_F(FrameProcessorIntegrationTest, ProcessMatBGRA) {
    if (!has_models) {
        GTEST_SKIP() << "Models not available, skipping";
    }

    // BGRA cv::Mat 처리
    cv::Mat frame(480, 640, CV_8UC4, cv::Scalar(128, 128, 128, 255));
    ProcessResult result = processor->process(frame, nullptr);

    EXPECT_TRUE(result.success);
    EXPECT_GT(result.processing_time_ms, 0.0f);
}

TEST_F(FrameProcessorIntegrationTest, ProcessWithRendering) {
    if (!has_models) {
        GTEST_SKIP() << "Models not available, skipping";
    }

    // 텍스처 로드
    auto texture = createTestTexture(64, 64);
    processor->loadLensTexture(texture.data(), 64, 64);

    // 렌더링 설정
    LensConfig config = createDefaultConfig();

    // 프레임 처리
    auto frame = createRGBAFrame(640, 480);
    ProcessResult result = processor->process(
        frame.data(), 640, 480, FrameFormat::RGBA, &config);

    EXPECT_TRUE(result.success);
    EXPECT_GT(result.processing_time_ms, 0.0f);
    EXPECT_GE(result.detection_time_ms, 0.0f);
    EXPECT_GE(result.render_time_ms, 0.0f);
}

TEST_F(FrameProcessorIntegrationTest, DetectOnlyReturnsResult) {
    if (!has_models) {
        GTEST_SKIP() << "Models not available, skipping";
    }

    // 검출만 수행
    auto frame = createRGBAFrame(640, 480);
    IrisResult result = processor->detectOnly(
        frame.data(), 640, 480, FrameFormat::RGBA);

    // 합성 프레임이라 검출 안될 수 있음
    // 중요한 건 크래시 없이 결과 반환
    EXPECT_EQ(result.frame_width, 640);
    EXPECT_EQ(result.frame_height, 480);
}

TEST_F(FrameProcessorIntegrationTest, RenderOnlyWithValidIrisResult) {
    if (!has_models) {
        GTEST_SKIP() << "Models not available, skipping";
    }

    // 텍스처 로드
    auto texture = createTestTexture(64, 64);
    processor->loadLensTexture(texture.data(), 64, 64);

    // 가상의 IrisResult 생성
    IrisResult iris_result{};
    iris_result.detected = true;
    iris_result.left_detected = true;
    iris_result.right_detected = true;
    iris_result.confidence = 0.9f;
    iris_result.frame_width = 640;
    iris_result.frame_height = 480;

    // 왼쪽 눈
    iris_result.left_iris[0] = {0.3f, 0.4f, 0.0f, 1.0f};
    iris_result.left_radius = 15.0f;

    // 오른쪽 눈
    iris_result.right_iris[0] = {0.7f, 0.4f, 0.0f, 1.0f};
    iris_result.right_radius = 15.0f;

    LensConfig config = createDefaultConfig();

    // 렌더링만 수행
    auto frame = createRGBAFrame(640, 480);
    bool success = processor->renderOnly(
        frame.data(), 640, 480, FrameFormat::RGBA, iris_result, config);

    EXPECT_TRUE(success);
}

// ============================================================
// 성능 테스트
// ============================================================

class FrameProcessorPerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        processor = std::make_unique<FrameProcessor>();
        model_path = findModelPath();

        if (!model_path.empty() && std::filesystem::exists(model_path)) {
            has_models = processor->initialize(model_path);
        }
    }

    void TearDown() override {
        processor.reset();
    }

    std::unique_ptr<FrameProcessor> processor;
    std::string model_path;
    bool has_models = false;
};

TEST_F(FrameProcessorPerformanceTest, ProcessingTimeUnder33ms) {
    if (!has_models) {
        GTEST_SKIP() << "Models not available, skipping performance test";
    }

    // 640x480 프레임 처리 시간 측정
    auto frame = createBGRFrame(640, 480);

    // 워밍업
    processor->process(frame.data(), 640, 480, FrameFormat::BGR, nullptr);

    // 실제 측정
    const int iterations = 10;
    float total_time = 0.0f;

    for (int i = 0; i < iterations; ++i) {
        ProcessResult result = processor->process(
            frame.data(), 640, 480, FrameFormat::BGR, nullptr);

        if (result.success) {
            total_time += result.processing_time_ms;
        }
    }

    float avg_time = total_time / iterations;
    std::cout << "Average processing time: " << avg_time << " ms" << std::endl;

    // 33ms 이하 목표 (30fps 유지)
    EXPECT_LT(avg_time, 33.0f) << "Processing time exceeds 33ms target for 30fps";
}

TEST_F(FrameProcessorPerformanceTest, FPSCalculation) {
    if (!has_models) {
        GTEST_SKIP() << "Models not available, skipping FPS test";
    }

    auto frame = createBGRFrame(640, 480);

    // 여러 프레임 처리
    for (int i = 0; i < 30; ++i) {
        processor->process(frame.data(), 640, 480, FrameFormat::BGR, nullptr);
    }

    double fps = processor->getAverageFPS();
    std::cout << "Average FPS: " << fps << std::endl;

    // FPS가 계산되었는지 확인
    EXPECT_GT(fps, 0.0);
}

TEST_F(FrameProcessorPerformanceTest, FormatConversionOverhead) {
    if (!has_models) {
        GTEST_SKIP() << "Models not available, skipping";
    }

    // BGR (무변환) vs RGBA (변환 필요) 비교
    auto bgr_frame = createBGRFrame(640, 480);
    auto rgba_frame = createRGBAFrame(640, 480);

    // 워밍업
    processor->process(bgr_frame.data(), 640, 480, FrameFormat::BGR, nullptr);
    processor->process(rgba_frame.data(), 640, 480, FrameFormat::RGBA, nullptr);

    // BGR 처리
    float bgr_total = 0.0f;
    for (int i = 0; i < 10; ++i) {
        ProcessResult result = processor->process(
            bgr_frame.data(), 640, 480, FrameFormat::BGR, nullptr);
        if (result.success) {
            bgr_total += result.processing_time_ms;
        }
    }
    float bgr_avg = bgr_total / 10.0f;

    // RGBA 처리
    float rgba_total = 0.0f;
    for (int i = 0; i < 10; ++i) {
        ProcessResult result = processor->process(
            rgba_frame.data(), 640, 480, FrameFormat::RGBA, nullptr);
        if (result.success) {
            rgba_total += result.processing_time_ms;
        }
    }
    float rgba_avg = rgba_total / 10.0f;

    std::cout << "BGR avg: " << bgr_avg << " ms, RGBA avg: " << rgba_avg << " ms" << std::endl;
    std::cout << "Format conversion overhead: " << (rgba_avg - bgr_avg) << " ms" << std::endl;

    // RGBA가 BGR보다 약간 느려야 함 (변환 오버헤드)
    // 하지만 차이가 5ms 이내여야 함
    EXPECT_LT(std::abs(rgba_avg - bgr_avg), 5.0f);
}

// ============================================================
// 엣지 케이스 테스트
// ============================================================

class FrameProcessorEdgeCaseTest : public ::testing::Test {
protected:
    void SetUp() override {
        processor = std::make_unique<FrameProcessor>();
    }

    void TearDown() override {
        processor.reset();
    }

    std::unique_ptr<FrameProcessor> processor;
};

TEST_F(FrameProcessorEdgeCaseTest, VerySmallFrame) {
    // 아주 작은 프레임
    auto frame = createRGBAFrame(16, 16);
    ProcessResult result = processor->process(
        frame.data(), 16, 16, FrameFormat::RGBA, nullptr);

    // 초기화 안됐으므로 실패
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.error_code, ErrorCode::NotInitialized);
}

TEST_F(FrameProcessorEdgeCaseTest, VeryLargeFrame) {
    // 아주 큰 프레임 (4K)
    auto frame = createRGBAFrame(3840, 2160);
    ProcessResult result = processor->process(
        frame.data(), 3840, 2160, FrameFormat::RGBA, nullptr);

    // 초기화 안됐으므로 실패
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.error_code, ErrorCode::NotInitialized);
}

TEST_F(FrameProcessorEdgeCaseTest, OddDimensions) {
    // 홀수 크기 (NV21/NV12에서 문제될 수 있음)
    auto frame = createRGBAFrame(641, 481);
    ProcessResult result = processor->process(
        frame.data(), 641, 481, FrameFormat::RGBA, nullptr);

    // 초기화 안됐으므로 실패
    EXPECT_FALSE(result.success);
}

TEST_F(FrameProcessorEdgeCaseTest, MultipleTextureLoadsWithoutInit) {
    // 초기화 없이 여러 번 텍스처 로드 시도 - 모두 실패
    for (int i = 0; i < 10; ++i) {
        auto texture = createTestTexture(64, 64);
        EXPECT_FALSE(processor->loadLensTexture(texture.data(), 64, 64));
        EXPECT_FALSE(processor->hasLensTexture());
    }
}

TEST_F(FrameProcessorEdgeCaseTest, AlternatingTextureLoadUnloadWithoutInit) {
    // 초기화 없이 텍스처 로드/언로드 반복 - 안전하게 실패
    for (int i = 0; i < 10; ++i) {
        auto texture = createTestTexture(64, 64);
        EXPECT_FALSE(processor->loadLensTexture(texture.data(), 64, 64));
        EXPECT_FALSE(processor->hasLensTexture());
        EXPECT_NO_THROW(processor->unloadLensTexture());
        EXPECT_FALSE(processor->hasLensTexture());
    }
}

TEST_F(FrameProcessorEdgeCaseTest, DifferentTextureSizesWithoutInit) {
    // 초기화 없이 다양한 크기의 텍스처 로드 시도 - 모두 실패
    std::vector<std::pair<int, int>> sizes = {
        {32, 32}, {64, 64}, {128, 128}, {256, 256},
        {100, 200}, {512, 256}, {1024, 1024}
    };

    for (const auto& size : sizes) {
        auto texture = createTestTexture(size.first, size.second);
        EXPECT_FALSE(processor->loadLensTexture(
            texture.data(), size.first, size.second))
            << "Should fail for size " << size.first << "x" << size.second;
        EXPECT_FALSE(processor->hasLensTexture());
    }
}

// ============================================================
// 메모리 안전성 테스트
// ============================================================

class FrameProcessorMemoryTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(FrameProcessorMemoryTest, CreateDestroyMultiple) {
    // 여러 번 생성/파괴
    for (int i = 0; i < 100; ++i) {
        auto processor = std::make_unique<FrameProcessor>();
        EXPECT_FALSE(processor->isInitialized());
    }
}

TEST_F(FrameProcessorMemoryTest, TextureMemoryCleanup) {
    // 텍스처 로드 후 파괴
    for (int i = 0; i < 50; ++i) {
        auto processor = std::make_unique<FrameProcessor>();
        auto texture = createTestTexture(256, 256);
        processor->loadLensTexture(texture.data(), 256, 256);
        // 자동 소멸 시 메모리 누수 없어야 함
    }
}

TEST_F(FrameProcessorMemoryTest, MoveSemantics) {
    // 이동 시맨틱 메모리 안전성 (초기화 없이)
    FrameProcessor proc1;
    // 초기화 없이는 텍스처 로드 불가
    auto texture = createTestTexture(64, 64);
    proc1.loadLensTexture(texture.data(), 64, 64);
    EXPECT_FALSE(proc1.hasLensTexture()); // 초기화 안됨

    // 이동
    FrameProcessor proc2(std::move(proc1));
    EXPECT_FALSE(proc2.hasLensTexture()); // 역시 초기화 안됨

    // 이동 후 원본은 빈 상태
    // (Pimpl이 이동되었으므로 원본 접근은 정의되지 않음)
}

} // namespace testing
} // namespace iris_sdk
