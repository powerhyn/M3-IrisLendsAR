/**
 * @file mediapipe_detector.cpp
 * @brief MediaPipeDetector 구현 (TDD RED Phase - 스텁)
 *
 * 이 파일은 TDD RED Phase를 위한 최소 스텁입니다.
 * 모든 테스트가 실패해야 합니다.
 */

#include "iris_sdk/mediapipe_detector.h"

namespace iris_sdk {

// Pimpl 구현 클래스 (스텁)
class MediaPipeDetector::Impl {
public:
    bool initialized = false;
    float min_detection_confidence = 0.5f;
    float min_tracking_confidence = 0.5f;
    int num_faces = 1;
};

MediaPipeDetector::MediaPipeDetector()
    : impl_(std::make_unique<Impl>()) {
}

MediaPipeDetector::~MediaPipeDetector() = default;

bool MediaPipeDetector::initialize(const std::string& model_path) {
    // RED: 항상 실패 (아직 구현 안됨)
    return false;
}

IrisResult MediaPipeDetector::detect(const uint8_t* frame_data,
                                      int width,
                                      int height,
                                      FrameFormat format) {
    // RED: 빈 결과 반환 (아직 구현 안됨)
    IrisResult result{};
    result.detected = false;
    result.confidence = 0.0f;
    return result;
}

void MediaPipeDetector::release() {
    impl_->initialized = false;
}

bool MediaPipeDetector::isInitialized() const {
    return impl_->initialized;
}

DetectorType MediaPipeDetector::getDetectorType() const {
    return DetectorType::MediaPipe;
}

void MediaPipeDetector::setMinDetectionConfidence(float confidence) {
    impl_->min_detection_confidence = confidence;
}

void MediaPipeDetector::setMinTrackingConfidence(float confidence) {
    impl_->min_tracking_confidence = confidence;
}

void MediaPipeDetector::setNumFaces(int num_faces) {
    impl_->num_faces = num_faces;
}

} // namespace iris_sdk
