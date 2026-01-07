/**
 * @file test_iris_detector.cpp
 * @brief TDD RED Phase: IrisDetector 인터페이스 테스트
 *
 * 이 테스트들은 iris_detector.h 구현 전에 작성됨.
 * 모든 테스트는 현재 실패해야 함 (RED phase).
 */

#include <gtest/gtest.h>
#include <type_traits>
#include <memory>
#include "iris_sdk/iris_detector.h"

namespace iris_sdk {
namespace testing {

// ============================================================
// IrisDetector 인터페이스 테스트
// ============================================================

class IrisDetectorTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ------------------------------------------------------------
// 클래스 특성 테스트
// ------------------------------------------------------------

TEST_F(IrisDetectorTest, IsAbstractClass) {
    // IrisDetector는 추상 클래스여야 함
    EXPECT_TRUE(std::is_abstract<IrisDetector>::value);
}

TEST_F(IrisDetectorTest, HasVirtualDestructor) {
    // 가상 소멸자가 있어야 함 (다형성 삭제 지원)
    EXPECT_TRUE(std::has_virtual_destructor<IrisDetector>::value);
}

TEST_F(IrisDetectorTest, IsNotCopyable) {
    // 복사 금지
    EXPECT_FALSE(std::is_copy_constructible<IrisDetector>::value);
    EXPECT_FALSE(std::is_copy_assignable<IrisDetector>::value);
}

TEST_F(IrisDetectorTest, IsNotMovable) {
    // 이동 금지 (구현체에서 결정)
    EXPECT_FALSE(std::is_move_constructible<IrisDetector>::value);
    EXPECT_FALSE(std::is_move_assignable<IrisDetector>::value);
}

// ------------------------------------------------------------
// Mock 구현체 (테스트용)
// ------------------------------------------------------------

class MockIrisDetector : public IrisDetector {
public:
    MockIrisDetector() : initialized_(false), detector_type_(DetectorType::Unknown) {}

    bool initialize(const std::string& model_path) override {
        model_path_ = model_path;
        initialized_ = !model_path.empty();
        return initialized_;
    }

    IrisResult detect(const uint8_t* frame_data, int width, int height, FrameFormat format) override {
        IrisResult result{};
        if (initialized_ && frame_data != nullptr && width > 0 && height > 0) {
            result.detected = true;
            result.confidence = 0.95f;
            result.left_detected = true;
            result.right_detected = true;
            result.left_iris[0] = {0.3f, 0.4f, 0.0f, 0.9f};
            result.right_iris[0] = {0.7f, 0.4f, 0.0f, 0.9f};
            result.frame_width = width;
            result.frame_height = height;
        }
        return result;
    }

    void release() override {
        initialized_ = false;
        model_path_.clear();
    }

    bool isInitialized() const override {
        return initialized_;
    }

    DetectorType getDetectorType() const override {
        return detector_type_;
    }

    // 테스트용 setter
    void setDetectorType(DetectorType type) { detector_type_ = type; }

private:
    bool initialized_;
    std::string model_path_;
    DetectorType detector_type_;
};

// ------------------------------------------------------------
// Mock 검출기 생성/동작 테스트
// ------------------------------------------------------------

TEST_F(IrisDetectorTest, MockDetectorCreation) {
    auto detector = std::make_unique<MockIrisDetector>();
    ASSERT_NE(detector, nullptr);
    EXPECT_FALSE(detector->isInitialized());
}

TEST_F(IrisDetectorTest, MockDetectorInitialize) {
    auto detector = std::make_unique<MockIrisDetector>();

    // 빈 경로로 초기화 실패
    EXPECT_FALSE(detector->initialize(""));
    EXPECT_FALSE(detector->isInitialized());

    // 유효한 경로로 초기화 성공
    EXPECT_TRUE(detector->initialize("/path/to/model.tflite"));
    EXPECT_TRUE(detector->isInitialized());
}

TEST_F(IrisDetectorTest, MockDetectorDetect) {
    auto detector = std::make_unique<MockIrisDetector>();
    detector->initialize("/path/to/model.tflite");

    // 테스트용 프레임 데이터
    const int width = 640;
    const int height = 480;
    std::vector<uint8_t> frame_data(width * height * 4, 128);  // RGBA

    IrisResult result = detector->detect(frame_data.data(), width, height, FrameFormat::RGBA);

    EXPECT_TRUE(result.detected);
    EXPECT_TRUE(result.left_detected);
    EXPECT_TRUE(result.right_detected);
    EXPECT_FLOAT_EQ(result.confidence, 0.95f);
    EXPECT_EQ(result.frame_width, width);
    EXPECT_EQ(result.frame_height, height);
}

TEST_F(IrisDetectorTest, MockDetectorDetectWithoutInit) {
    auto detector = std::make_unique<MockIrisDetector>();

    // 초기화 없이 검출 시도
    const int width = 640;
    const int height = 480;
    std::vector<uint8_t> frame_data(width * height * 4, 128);

    IrisResult result = detector->detect(frame_data.data(), width, height, FrameFormat::RGBA);

    EXPECT_FALSE(result.detected);
}

TEST_F(IrisDetectorTest, MockDetectorDetectNullFrame) {
    auto detector = std::make_unique<MockIrisDetector>();
    detector->initialize("/path/to/model.tflite");

    IrisResult result = detector->detect(nullptr, 640, 480, FrameFormat::RGBA);

    EXPECT_FALSE(result.detected);
}

TEST_F(IrisDetectorTest, MockDetectorDetectInvalidDimensions) {
    auto detector = std::make_unique<MockIrisDetector>();
    detector->initialize("/path/to/model.tflite");

    std::vector<uint8_t> frame_data(100, 128);

    // 너비 0
    IrisResult result1 = detector->detect(frame_data.data(), 0, 480, FrameFormat::RGBA);
    EXPECT_FALSE(result1.detected);

    // 높이 0
    IrisResult result2 = detector->detect(frame_data.data(), 640, 0, FrameFormat::RGBA);
    EXPECT_FALSE(result2.detected);
}

TEST_F(IrisDetectorTest, MockDetectorRelease) {
    auto detector = std::make_unique<MockIrisDetector>();
    detector->initialize("/path/to/model.tflite");
    EXPECT_TRUE(detector->isInitialized());

    detector->release();
    EXPECT_FALSE(detector->isInitialized());
}

TEST_F(IrisDetectorTest, MockDetectorGetType) {
    auto detector = std::make_unique<MockIrisDetector>();

    EXPECT_EQ(detector->getDetectorType(), DetectorType::Unknown);

    detector->setDetectorType(DetectorType::MediaPipe);
    EXPECT_EQ(detector->getDetectorType(), DetectorType::MediaPipe);
}

// ------------------------------------------------------------
// 다형성 테스트
// ------------------------------------------------------------

TEST_F(IrisDetectorTest, Polymorphism) {
    // 기본 포인터로 파생 클래스 접근
    std::unique_ptr<IrisDetector> detector = std::make_unique<MockIrisDetector>();

    EXPECT_FALSE(detector->isInitialized());
    EXPECT_TRUE(detector->initialize("/path/to/model.tflite"));
    EXPECT_TRUE(detector->isInitialized());

    detector->release();
    EXPECT_FALSE(detector->isInitialized());
}

// ------------------------------------------------------------
// 팩토리 함수 테스트
// ------------------------------------------------------------

TEST_F(IrisDetectorTest, FactoryCreateMediaPipe) {
    auto detector = detail::createDetector(DetectorType::MediaPipe);
    // Phase 1에서는 MediaPipeDetector만 구현
    // 현재는 팩토리 함수 존재 여부만 확인
    // 구현체가 없으면 nullptr 반환 가능
}

TEST_F(IrisDetectorTest, FactoryCreateByString) {
    auto detector = detail::createDetector("mediapipe");
    // 문자열 기반 팩토리 함수 테스트
}

TEST_F(IrisDetectorTest, FactoryCreateUnknown) {
    auto detector = detail::createDetector(DetectorType::Unknown);
    EXPECT_EQ(detector, nullptr);
}

TEST_F(IrisDetectorTest, FactoryCreateInvalidString) {
    auto detector = detail::createDetector("invalid_detector");
    EXPECT_EQ(detector, nullptr);
}

} // namespace testing
} // namespace iris_sdk
