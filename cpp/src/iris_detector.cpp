/**
 * @file iris_detector.cpp
 * @brief IrisDetector 팩토리 함수 구현
 */

#include "iris_sdk/iris_detector.h"
#include <algorithm>
#include <cctype>

namespace iris_sdk {
namespace detail {

std::unique_ptr<IrisDetector> createDetector(DetectorType type) {
    switch (type) {
        case DetectorType::MediaPipe:
            // TODO: Phase 1에서 MediaPipeDetector 구현 후 활성화
            // return std::make_unique<MediaPipeDetector>();
            return nullptr;

        case DetectorType::EyeOnly:
            // TODO: Phase 2에서 EyeOnlyDetector 구현 후 활성화
            return nullptr;

        case DetectorType::Hybrid:
            // TODO: Phase 2에서 HybridDetector 구현 후 활성화
            return nullptr;

        case DetectorType::Unknown:
        default:
            return nullptr;
    }
}

std::unique_ptr<IrisDetector> createDetector(const std::string& type_name) {
    // 소문자로 변환
    std::string lower_name = type_name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (lower_name == "mediapipe") {
        return createDetector(DetectorType::MediaPipe);
    } else if (lower_name == "eyeonly" || lower_name == "eye_only") {
        return createDetector(DetectorType::EyeOnly);
    } else if (lower_name == "hybrid") {
        return createDetector(DetectorType::Hybrid);
    }

    return nullptr;
}

} // namespace detail
} // namespace iris_sdk
