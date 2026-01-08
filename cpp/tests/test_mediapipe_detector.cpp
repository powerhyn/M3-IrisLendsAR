/**
 * @file test_mediapipe_detector.cpp
 * @brief TDD RED Phase: MediaPipeDetector 구현 테스트
 *
 * 이 테스트들은 MediaPipeDetector 구현 전에 작성됨.
 * 모든 테스트는 현재 실패해야 함 (RED phase).
 */

#include <gtest/gtest.h>
#include <type_traits>
#include <memory>
#include "iris_sdk/mediapipe_detector.h"

namespace iris_sdk {
namespace testing {

// ============================================================
// MediaPipeDetector 기본 테스트
// ============================================================

class MediaPipeDetectorTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ------------------------------------------------------------
// 클래스 특성 테스트
// ------------------------------------------------------------

TEST_F(MediaPipeDetectorTest, InheritsFromIrisDetector) {
    // IrisDetector를 상속받아야 함
    EXPECT_TRUE((std::is_base_of<IrisDetector, MediaPipeDetector>::value));
}

TEST_F(MediaPipeDetectorTest, IsNotAbstract) {
    // 구체 클래스여야 함 (인스턴스화 가능)
    EXPECT_FALSE(std::is_abstract<MediaPipeDetector>::value);
}

TEST_F(MediaPipeDetectorTest, HasVirtualDestructor) {
    // 가상 소멸자 필요 (다형성 지원)
    EXPECT_TRUE(std::has_virtual_destructor<MediaPipeDetector>::value);
}

// ------------------------------------------------------------
// 생성 및 소멸 테스트
// ------------------------------------------------------------

TEST_F(MediaPipeDetectorTest, DefaultConstruction) {
    // 기본 생성자로 인스턴스 생성 가능
    MediaPipeDetector detector;
    EXPECT_FALSE(detector.isInitialized());
}

TEST_F(MediaPipeDetectorTest, PointerConstruction) {
    // 포인터로 생성 가능
    auto detector = std::make_unique<MediaPipeDetector>();
    EXPECT_NE(detector, nullptr);
    EXPECT_FALSE(detector->isInitialized());
}

TEST_F(MediaPipeDetectorTest, PolymorphicCreation) {
    // IrisDetector 포인터로 다형성 사용 가능
    std::unique_ptr<IrisDetector> detector = std::make_unique<MediaPipeDetector>();
    EXPECT_NE(detector, nullptr);
    EXPECT_EQ(detector->getDetectorType(), DetectorType::MediaPipe);
}

// ------------------------------------------------------------
// 초기화 테스트
// ------------------------------------------------------------

TEST_F(MediaPipeDetectorTest, InitializeWithInvalidPath) {
    // 존재하지 않는 경로로 초기화 실패
    MediaPipeDetector detector;
    EXPECT_FALSE(detector.initialize("/nonexistent/path/to/models"));
    EXPECT_FALSE(detector.isInitialized());
}

TEST_F(MediaPipeDetectorTest, InitializeWithEmptyPath) {
    // 빈 경로로 초기화 실패
    MediaPipeDetector detector;
    EXPECT_FALSE(detector.initialize(""));
    EXPECT_FALSE(detector.isInitialized());
}

TEST_F(MediaPipeDetectorTest, InitializeWithValidPath) {
    // 유효한 경로로 초기화 (모델 파일 없어도 경로 검증만)
    // 실제 모델 없이 테스트하므로 실패 예상
    MediaPipeDetector detector;
    // 테스트용 임시 디렉토리
    EXPECT_FALSE(detector.initialize("/tmp"));  // 모델 파일이 없으므로 실패
}

TEST_F(MediaPipeDetectorTest, DoubleInitializationFails) {
    // 두 번 초기화 시도하면 두 번째는 실패
    MediaPipeDetector detector;
    // 첫 번째 초기화 (실패하더라도)
    detector.initialize("/tmp");
    // 두 번째 초기화 시도
    EXPECT_FALSE(detector.initialize("/tmp"));
}

// ------------------------------------------------------------
// getDetectorType 테스트
// ------------------------------------------------------------

TEST_F(MediaPipeDetectorTest, GetDetectorTypeReturnsMediaPipe) {
    MediaPipeDetector detector;
    EXPECT_EQ(detector.getDetectorType(), DetectorType::MediaPipe);
}

// ------------------------------------------------------------
// detect 테스트 (초기화 전)
// ------------------------------------------------------------

TEST_F(MediaPipeDetectorTest, DetectWithoutInitialization) {
    // 초기화 없이 detect 호출 시 빈 결과 반환
    MediaPipeDetector detector;

    // 더미 프레임 데이터
    std::vector<uint8_t> dummy_frame(640 * 480 * 3, 128);

    IrisResult result = detector.detect(
        dummy_frame.data(),
        640, 480,
        FrameFormat::RGB
    );

    EXPECT_FALSE(result.detected);
    EXPECT_FLOAT_EQ(result.confidence, 0.0f);
}

TEST_F(MediaPipeDetectorTest, DetectWithNullFrame) {
    // nullptr 프레임으로 detect 호출
    MediaPipeDetector detector;

    IrisResult result = detector.detect(nullptr, 640, 480, FrameFormat::RGB);

    EXPECT_FALSE(result.detected);
}

TEST_F(MediaPipeDetectorTest, DetectWithInvalidDimensions) {
    // 잘못된 크기로 detect 호출
    MediaPipeDetector detector;
    std::vector<uint8_t> dummy_frame(100, 128);

    // 0 크기
    IrisResult result1 = detector.detect(dummy_frame.data(), 0, 480, FrameFormat::RGB);
    EXPECT_FALSE(result1.detected);

    // 음수 크기 (int로 변환 시)
    IrisResult result2 = detector.detect(dummy_frame.data(), 640, 0, FrameFormat::RGB);
    EXPECT_FALSE(result2.detected);
}

// ------------------------------------------------------------
// release 테스트
// ------------------------------------------------------------

TEST_F(MediaPipeDetectorTest, ReleaseBeforeInitialization) {
    // 초기화 전 release 호출해도 안전
    MediaPipeDetector detector;
    EXPECT_NO_THROW(detector.release());
    EXPECT_FALSE(detector.isInitialized());
}

TEST_F(MediaPipeDetectorTest, ReleaseAfterInitialization) {
    // 초기화 후 release 호출
    MediaPipeDetector detector;
    detector.initialize("/tmp");  // 실패하더라도
    EXPECT_NO_THROW(detector.release());
    EXPECT_FALSE(detector.isInitialized());
}

TEST_F(MediaPipeDetectorTest, DoubleRelease) {
    // 두 번 release 호출해도 안전
    MediaPipeDetector detector;
    EXPECT_NO_THROW(detector.release());
    EXPECT_NO_THROW(detector.release());
}

// ------------------------------------------------------------
// MediaPipe 전용 설정 테스트
// ------------------------------------------------------------

TEST_F(MediaPipeDetectorTest, SetMinDetectionConfidence) {
    MediaPipeDetector detector;

    // 유효한 범위 (0.0 ~ 1.0)
    EXPECT_NO_THROW(detector.setMinDetectionConfidence(0.5f));
    EXPECT_NO_THROW(detector.setMinDetectionConfidence(0.0f));
    EXPECT_NO_THROW(detector.setMinDetectionConfidence(1.0f));
}

TEST_F(MediaPipeDetectorTest, SetMinTrackingConfidence) {
    MediaPipeDetector detector;

    EXPECT_NO_THROW(detector.setMinTrackingConfidence(0.5f));
    EXPECT_NO_THROW(detector.setMinTrackingConfidence(0.0f));
    EXPECT_NO_THROW(detector.setMinTrackingConfidence(1.0f));
}

TEST_F(MediaPipeDetectorTest, SetNumFaces) {
    MediaPipeDetector detector;

    // 유효한 범위 (1 이상)
    EXPECT_NO_THROW(detector.setNumFaces(1));
    EXPECT_NO_THROW(detector.setNumFaces(5));
}

// ------------------------------------------------------------
// Factory 통합 테스트 (GREEN 단계에서 활성화)
// ------------------------------------------------------------

// TODO: Factory 구현 후 테스트 활성화
// TEST_F(MediaPipeDetectorTest, FactoryCreatesMediaPipeDetector)
// TEST_F(MediaPipeDetectorTest, FactoryCreatesMediaPipeDetectorByString)

} // namespace testing
} // namespace iris_sdk
