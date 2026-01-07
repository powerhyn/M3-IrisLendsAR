/**
 * @file iris_detector.h
 * @brief 홍채 검출기 추상 인터페이스
 *
 * Strategy 패턴을 사용하여 다양한 검출기 구현체를 교체 가능하게 함.
 */

#ifndef IRIS_SDK_IRIS_DETECTOR_H
#define IRIS_SDK_IRIS_DETECTOR_H

#include <memory>
#include <string>
#include <cstdint>
#include "iris_sdk/types.h"
#include "iris_sdk/export.h"

namespace iris_sdk {

/**
 * @brief 홍채 검출기 추상 인터페이스
 *
 * Strategy 패턴을 사용하여 다양한 검출기 구현체를 교체 가능하게 함.
 * - MediaPipeDetector (Phase 1)
 * - EyeOnlyDetector (Phase 2)
 * - HybridDetector (Phase 2 - 폴백 전략)
 */
class IRIS_SDK_EXPORT IrisDetector {
public:
    /**
     * @brief 가상 소멸자
     */
    virtual ~IrisDetector() = default;

    /**
     * @brief 검출기 초기화
     * @param model_path 모델 파일 경로
     * @return 초기화 성공 여부
     */
    virtual bool initialize(const std::string& model_path) = 0;

    /**
     * @brief 프레임에서 홍채 검출
     * @param frame_data 입력 이미지 데이터 포인터
     * @param width 이미지 너비
     * @param height 이미지 높이
     * @param format 픽셀 포맷
     * @return 검출 결과
     */
    virtual IrisResult detect(const uint8_t* frame_data,
                              int width,
                              int height,
                              FrameFormat format) = 0;

    /**
     * @brief 리소스 해제
     */
    virtual void release() = 0;

    /**
     * @brief 초기화 상태 확인
     * @return 초기화되었으면 true
     */
    virtual bool isInitialized() const = 0;

    /**
     * @brief 검출기 종류 반환
     * @return DetectorType 열거값
     */
    virtual DetectorType getDetectorType() const = 0;

    // 복사 금지
    IrisDetector(const IrisDetector&) = delete;
    IrisDetector& operator=(const IrisDetector&) = delete;

    // 이동 금지
    IrisDetector(IrisDetector&&) = delete;
    IrisDetector& operator=(IrisDetector&&) = delete;

protected:
    /**
     * @brief 기본 생성자 (protected - 직접 인스턴스화 금지)
     */
    IrisDetector() = default;
};

// ============================================================
// 내부 팩토리 함수 (detail 네임스페이스)
// ============================================================
//
// 참고: 외부 API 사용자는 SDKManager::createDetector()를 사용해야 함.
// 아래 함수들은 SDKManager 내부 및 테스트 용도로만 사용됨.
//

namespace detail {

/**
 * @brief 검출기 팩토리 함수 (내부용)
 * @param type 생성할 검출기 종류
 * @return unique_ptr로 관리되는 검출기 인스턴스
 * @note 외부에서는 SDKManager::createDetector() 사용 권장
 */
IRIS_SDK_EXPORT std::unique_ptr<IrisDetector> createDetector(DetectorType type);

/**
 * @brief 문자열로 검출기 생성 (내부용)
 * @param type_name 검출기 이름 ("mediapipe", "eyeonly", "hybrid")
 * @return unique_ptr로 관리되는 검출기 인스턴스
 * @note 외부에서는 SDKManager::createDetector() 사용 권장
 */
IRIS_SDK_EXPORT std::unique_ptr<IrisDetector> createDetector(const std::string& type_name);

} // namespace detail

} // namespace iris_sdk

#endif // IRIS_SDK_IRIS_DETECTOR_H
