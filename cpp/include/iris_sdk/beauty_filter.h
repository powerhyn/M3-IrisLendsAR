/**
 * @file beauty_filter.h
 * @brief IrisLensSDK Beauty Filter C API
 *
 * 뷰티 카메라 필터 효과를 제공하는 C API.
 * 피부 스무딩, 밝기 조절, 소프트 포커스 등의 기능을 지원합니다.
 * POD 구조체와 extern "C" 함수로 FFI 호환성을 보장합니다.
 *
 * @author IrisLensSDK Team
 * @version 1.0.0
 * @copyright Apache 2.0 License
 */

#ifndef IRIS_SDK_BEAUTY_FILTER_H
#define IRIS_SDK_BEAUTY_FILTER_H

#include "export.h"
#include "sdk_api.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// 데이터 구조체
// ============================================================================

/**
 * @brief 뷰티 필터 설정
 *
 * 뷰티 카메라 효과를 위한 파라미터를 포함합니다.
 * POD 타입으로 FFI 호환성을 보장합니다.
 */
typedef struct BeautyFilterConfig {
    /** @brief 필터 활성화 여부 (기본값 true) */
    bool enabled;

    /** @brief 전체 강도 (0.0~1.0, 기본값 0.5) */
    float intensity;

    /** @brief 피부 스무딩 강도 (0.0~1.0, 기본값 0.5)
     *  @note Bilateral Filter 기반 피부 스무딩 */
    float smoothing;

    /** @brief 밝기 조절 (0.0~2.0, 1.0=원본, 기본값 1.05)
     *  @note 1.0 미만은 어둡게, 1.0 초과는 밝게 */
    float brightness;

    /** @brief 소프트 포커스 강도 (0.0~1.0, 기본값 0.3)
     *  @note Gaussian Blur 기반 소프트 글로우 효과 */
    float softFocus;
} BeautyFilterConfig;

// ============================================================================
// 설정 함수
// ============================================================================

/**
 * @brief 기본 뷰티 필터 설정 가져오기
 *
 * 기본값으로 초기화된 BeautyFilterConfig를 반환합니다.
 *
 * @param config 설정 구조체 포인터 (NULL 불가)
 *
 * @code
 * BeautyFilterConfig config;
 * iris_sdk_default_beauty_config(&config);
 * config.smoothing = 0.7f;  // 스무딩 강도 증가
 * iris_sdk_set_beauty_filter(&config);
 * @endcode
 */
IRIS_SDK_EXPORT void iris_sdk_default_beauty_config(BeautyFilterConfig* config);

/**
 * @brief 뷰티 필터 설정 적용
 *
 * 전역 뷰티 필터 설정을 업데이트합니다.
 *
 * @param config 적용할 설정 (NULL 불가)
 * @return IRIS_SDK_OK 성공, IRIS_SDK_NULL_POINTER config가 NULL인 경우
 *
 * @note 설정은 즉시 적용되며, 다음 iris_sdk_apply_beauty_filter() 호출부터 반영됩니다.
 */
IRIS_SDK_EXPORT IrisSdkError iris_sdk_set_beauty_filter(const BeautyFilterConfig* config);

/**
 * @brief 현재 뷰티 필터 설정 가져오기
 *
 * 현재 적용된 뷰티 필터 설정을 반환합니다.
 *
 * @param config 설정을 저장할 구조체 포인터 (NULL 불가)
 * @return IRIS_SDK_OK 성공, IRIS_SDK_NULL_POINTER config가 NULL인 경우
 */
IRIS_SDK_EXPORT IrisSdkError iris_sdk_get_beauty_filter(BeautyFilterConfig* config);

/**
 * @brief 뷰티 필터 활성화 여부 확인
 *
 * @return true 활성화됨, false 비활성화됨
 */
IRIS_SDK_EXPORT bool iris_sdk_is_beauty_filter_enabled(void);

// ============================================================================
// 처리 함수
// ============================================================================

/**
 * @brief 프레임에 뷰티 필터 적용
 *
 * 현재 설정된 뷰티 필터를 프레임에 적용합니다.
 * 프레임 데이터는 in-place로 수정됩니다.
 *
 * @param frame_data 프레임 데이터 (in-place 수정됨)
 * @param width 프레임 너비
 * @param height 프레임 높이
 * @param format 픽셀 포맷
 * @return IRIS_SDK_OK 성공, 그 외 에러 코드
 *
 * @note 지원 포맷: IRIS_FORMAT_RGBA, IRIS_FORMAT_BGRA, IRIS_FORMAT_RGB,
 *       IRIS_FORMAT_BGR, IRIS_FORMAT_NV21, IRIS_FORMAT_NV12
 * @note 필터가 비활성화(enabled=false)된 경우 아무 작업 없이 IRIS_SDK_OK 반환
 *
 * @code
 * // 프레임 처리 예시
 * uint8_t* frame = get_camera_frame();
 * IrisSdkError err = iris_sdk_apply_beauty_filter(frame, 1920, 1080, IRIS_FORMAT_BGRA);
 * if (err != IRIS_SDK_OK) {
 *     printf("Beauty filter failed: %s\n", iris_sdk_error_to_string(err));
 * }
 * @endcode
 */
IRIS_SDK_EXPORT IrisSdkError iris_sdk_apply_beauty_filter(
    uint8_t* frame_data,
    int width,
    int height,
    IrisFrameFormat format);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* IRIS_SDK_BEAUTY_FILTER_H */
