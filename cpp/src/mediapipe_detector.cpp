/**
 * @file mediapipe_detector.cpp
 * @brief MediaPipeDetector 구현 (TDD GREEN Phase)
 *
 * MediaPipe Face Mesh + Iris Landmark 모델을 사용한 홍채 검출
 * TensorFlow Lite 런타임으로 모델 추론 수행
 */

#include "iris_sdk/mediapipe_detector.h"
#include <algorithm>  // std::clamp
#include <filesystem>

namespace iris_sdk {

// ============================================================
// Pimpl 구현 클래스
// ============================================================
class MediaPipeDetector::Impl {
public:
    bool initialized = false;
    std::string model_path;

    // MediaPipe 설정
    float min_detection_confidence = 0.5f;
    float min_tracking_confidence = 0.5f;
    int num_faces = 1;

    // 필요한 모델 파일 목록
    static constexpr const char* REQUIRED_MODELS[] = {
        "face_detection_short_range.tflite",
        "face_landmark.tflite",
        "iris_landmark.tflite"
    };

    /**
     * @brief 모델 경로 유효성 검사
     * @param path 모델 디렉토리 경로
     * @return 유효 여부
     */
    bool validateModelPath(const std::string& path) {
        // 빈 경로 검사
        if (path.empty()) {
            return false;
        }

        // 경로 존재 여부 검사
        if (!std::filesystem::exists(path)) {
            return false;
        }

        // 디렉토리 여부 검사
        if (!std::filesystem::is_directory(path)) {
            return false;
        }

        // 필수 모델 파일 존재 여부 검사
        for (const auto& model : REQUIRED_MODELS) {
            std::filesystem::path model_file = std::filesystem::path(path) / model;
            if (!std::filesystem::exists(model_file)) {
                return false;
            }
        }

        return true;
    }
};

// ============================================================
// 생성자/소멸자
// ============================================================

MediaPipeDetector::MediaPipeDetector()
    : impl_(std::make_unique<Impl>()) {
}

MediaPipeDetector::~MediaPipeDetector() = default;

// ============================================================
// IrisDetector 인터페이스 구현
// ============================================================

bool MediaPipeDetector::initialize(const std::string& model_path) {
    // 이미 초기화된 경우 실패
    if (impl_->initialized) {
        return false;
    }

    // 빈 경로 검사
    if (model_path.empty()) {
        return false;
    }

    // 모델 경로 유효성 검사
    if (!impl_->validateModelPath(model_path)) {
        return false;
    }

    // TODO: TensorFlow Lite 모델 로딩
    // - face_detection_short_range.tflite
    // - face_landmark.tflite
    // - iris_landmark.tflite

    impl_->model_path = model_path;
    impl_->initialized = true;

    return true;
}

IrisResult MediaPipeDetector::detect(const uint8_t* frame_data,
                                      int width,
                                      int height,
                                      FrameFormat format) {
    IrisResult result{};
    result.detected = false;
    result.confidence = 0.0f;

    // 초기화 상태 검사
    if (!impl_->initialized) {
        return result;
    }

    // 프레임 데이터 유효성 검사
    if (frame_data == nullptr) {
        return result;
    }

    // 크기 유효성 검사
    if (width <= 0 || height <= 0) {
        return result;
    }

    // TODO: MediaPipe 파이프라인 실행
    // 1. 얼굴 검출 (Face Detection)
    // 2. 얼굴 랜드마크 추출 (Face Mesh)
    // 3. 홍채 랜드마크 추출 (Iris Landmark)
    // 4. 결과 구성

    return result;
}

void MediaPipeDetector::release() {
    // TODO: TensorFlow Lite 인터프리터 해제
    impl_->initialized = false;
    impl_->model_path.clear();
}

bool MediaPipeDetector::isInitialized() const {
    return impl_->initialized;
}

DetectorType MediaPipeDetector::getDetectorType() const {
    return DetectorType::MediaPipe;
}

// ============================================================
// MediaPipe 전용 설정
// ============================================================

void MediaPipeDetector::setMinDetectionConfidence(float confidence) {
    impl_->min_detection_confidence = std::clamp(confidence, 0.0f, 1.0f);
}

void MediaPipeDetector::setMinTrackingConfidence(float confidence) {
    impl_->min_tracking_confidence = std::clamp(confidence, 0.0f, 1.0f);
}

void MediaPipeDetector::setNumFaces(int num_faces) {
    // 최소 1개 이상
    impl_->num_faces = std::max(1, num_faces);
}

} // namespace iris_sdk
