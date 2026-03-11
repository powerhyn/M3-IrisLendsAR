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

/**
 * @brief 확장된 뷰티 필터 설정 (V2)
 *
 * 기존 V1 필드 + 고급 피부 효과 + 얼굴 형태 보정 + 처리 옵션
 * POD 타입으로 FFI 호환성을 보장합니다.
 */
typedef struct BeautyFilterConfigV2 {
    //===== 기본 설정 (V1 호환) =====
    /** @brief 필터 활성화 여부 */
    bool enabled;
    /** @brief 전체 강도 (0.0~1.0, 기본값 0.5) */
    float intensity;

    //===== 피부 효과 =====
    /** @brief 피부 스무딩 (0.0~1.0, 기본값 0.5) */
    float smoothing;
    /** @brief 밝기 (0.5~1.5, 1.0=원본, 기본값 1.0) */
    float brightness;
    /** @brief 소프트 포커스 (0.0~1.0, 기본값 0.3) */
    float softFocus;
    /** @brief 피부톤 화이트닝 (0.0~1.0, 기본값 0.0) */
    float whitening;
    /** @brief 컬러 밸런스 (-1.0~1.0, 기본값 0.0)
     *  @note 음수=쿨톤, 양수=웜톤 */
    float colorBalance;
    /** @brief 주름 제거 (0.0~1.0, 기본값 0.0) */
    float wrinkleRemove;
    /** @brief 피부 품질 개선 (0~1, 기본 0.0, Freq Sep 활성화) */
    float skinQuality;

    //===== 얼굴 형태 보정 =====
    /** @brief 얼굴 슬림화 (0.0~1.0, 기본값 0.0) */
    float slimFace;
    /** @brief 눈 확대 (0.0~1.0, 기본값 0.0) */
    float enlargeEyes;
    /** @brief 턱 축소 (0.0~1.0, 기본값 0.0) */
    float thinChin;

    //===== 처리 옵션 =====
    /** @brief GPU 가속 사용 (기본값 true) */
    bool useGpu;
    /** @brief 얼굴 영역만 처리 (기본값 true) */
    bool roiOnly;
    /** @brief 눈 영역 보호 (기본값 true) */
    bool protectEyes;
    /** @brief 입술 영역 보호 (기본값 true) */
    bool protectLips;
    /** @brief 다운스케일 팩터 (1=원본, 2=1/2, 4=1/4, 기본값 1)
     *  @note 성능과 품질 트레이드오프 조절용 */
    int downscaleFactor;
} BeautyFilterConfigV2;

// ============================================================================
// V1 설정 함수
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

// ============================================================================
// V2 설정 함수
// ============================================================================

/**
 * @brief 기본 V2 뷰티 필터 설정 가져오기
 *
 * 기본값으로 초기화된 BeautyFilterConfigV2를 반환합니다.
 *
 * @param config 설정 구조체 포인터 (NULL 불가)
 */
IRIS_SDK_EXPORT void iris_sdk_default_beauty_config_v2(BeautyFilterConfigV2* config);

/**
 * @brief V2 뷰티 필터 설정 적용
 *
 * @param config 적용할 설정 (NULL 불가)
 * @return IRIS_SDK_OK 성공, IRIS_SDK_INVALID_PARAM 범위 초과
 */
IRIS_SDK_EXPORT IrisSdkError iris_sdk_set_beauty_filter_v2(const BeautyFilterConfigV2* config);

/**
 * @brief V2 현재 뷰티 필터 설정 가져오기
 *
 * @param config 설정을 저장할 구조체 포인터 (NULL 불가)
 * @return IRIS_SDK_OK 성공
 */
IRIS_SDK_EXPORT IrisSdkError iris_sdk_get_beauty_filter_v2(BeautyFilterConfigV2* config);

/**
 * @brief V2 뷰티 필터 적용 (Face Mesh 연동)
 *
 * Face Mesh 정보를 활용하여 ROI 기반 처리 및 얼굴 형태 보정을 수행합니다.
 *
 * @param frame_data 프레임 데이터 (in-place 수정됨)
 * @param width 프레임 너비
 * @param height 프레임 높이
 * @param format 픽셀 포맷
 * @param iris_result Face Mesh 정보 (NULL 가능, NULL이면 V1 동작)
 * @return IRIS_SDK_OK 성공
 */
IRIS_SDK_EXPORT IrisSdkError iris_sdk_apply_beauty_filter_v2(
    uint8_t* frame_data,
    int width,
    int height,
    IrisFrameFormat format,
    const IrisResult* iris_result);

/**
 * @brief GPU 뷰티 필터 사용 가능 여부 확인
 *
 * @return true GPU 사용 가능, false CPU만 사용 가능
 */
IRIS_SDK_EXPORT bool iris_sdk_beauty_gpu_available(void);

/**
 * @brief 현재 GPU 사용 중 여부 확인
 *
 * @return true GPU 사용 중, false CPU 사용 중
 */
IRIS_SDK_EXPORT bool iris_sdk_beauty_using_gpu(void);

// ============================================================================
// 프리셋 및 skinQuality 편의 API
// ============================================================================

/**
 * @brief 뷰티 프리셋 (skinQuality 사전 설정)
 */
typedef enum IrisBeautyPreset {
    IRIS_BEAUTY_PRESET_NATURAL  = 0,   /**< 자연스러운 보정 (skinQuality=0.3) */
    IRIS_BEAUTY_PRESET_MODERATE = 1,   /**< 적절한 보정 (skinQuality=0.5) */
    IRIS_BEAUTY_PRESET_STRONG   = 2,   /**< 강한 보정 (skinQuality=0.8) */
    IRIS_BEAUTY_PRESET_CUSTOM   = 3    /**< 사용자 직접 설정 */
} IrisBeautyPreset;

/**
 * @brief 피부 품질 개선 강도 설정 (Frequency Separation)
 *
 * skinQuality > 0이면 Freq Sep 파이프라인이 활성화됩니다.
 * skinQuality = 0이면 기존 Bilateral 경로로 동작합니다 (하위 호환).
 *
 * @param quality 피부 품질 강도 (0.0~1.0, 0.0=비활성)
 * @return IRIS_SDK_OK 성공, IRIS_SDK_INVALID_PARAM 범위 초과
 */
IRIS_SDK_EXPORT IrisSdkError iris_sdk_set_skin_quality(float quality);

/**
 * @brief 현재 피부 품질 개선 강도 조회
 *
 * @param out_quality 결과를 저장할 포인터 (NULL 불가)
 * @return IRIS_SDK_OK 성공, IRIS_SDK_NULL_POINTER out_quality가 NULL인 경우
 */
IRIS_SDK_EXPORT IrisSdkError iris_sdk_get_skin_quality(float* out_quality);

/**
 * @brief 뷰티 프리셋 적용
 *
 * 프리셋에 따라 skinQuality 값이 자동 설정됩니다.
 * IRIS_BEAUTY_PRESET_CUSTOM 선택 시 skinQuality 값이 변경되지 않습니다.
 *
 * @param preset 적용할 프리셋
 * @return IRIS_SDK_OK 성공, IRIS_SDK_INVALID_PARAM 유효하지 않은 프리셋
 */
IRIS_SDK_EXPORT IrisSdkError iris_sdk_set_beauty_preset(IrisBeautyPreset preset);

#ifdef __cplusplus
}  /* extern "C" */
#endif

// ============================================================================
// C++ Helper Methods (C++ 컴파일 시에만)
// ============================================================================
#ifdef __cplusplus

namespace iris_sdk {

/**
 * @brief BeautyFilterConfigV2 C++ 헬퍼 함수들
 */
struct BeautyFilterConfigV2Helper {
    /**
     * @brief V1 설정으로부터 V2 생성
     */
    static BeautyFilterConfigV2 fromV1(const BeautyFilterConfig& v1) {
        BeautyFilterConfigV2 v2 = {};
        v2.enabled = v1.enabled;
        v2.intensity = v1.intensity;
        v2.smoothing = v1.smoothing;
        v2.brightness = v1.brightness;
        v2.softFocus = v1.softFocus;
        // V2 전용 필드는 기본값
        v2.whitening = 0.0f;
        v2.colorBalance = 0.0f;
        v2.wrinkleRemove = 0.0f;
        v2.skinQuality = 0.0f;
        v2.slimFace = 0.0f;
        v2.enlargeEyes = 0.0f;
        v2.thinChin = 0.0f;
        v2.useGpu = true;
        v2.roiOnly = true;
        v2.protectEyes = true;
        v2.protectLips = true;
        v2.downscaleFactor = 1;
        return v2;
    }

    /**
     * @brief V2 설정을 V1으로 변환 (피부 효과만)
     */
    static BeautyFilterConfig toV1(const BeautyFilterConfigV2& v2) {
        BeautyFilterConfig v1 = {};
        v1.enabled = v2.enabled;
        v1.intensity = v2.intensity;
        v1.smoothing = v2.smoothing;
        v1.brightness = v2.brightness;
        v1.softFocus = v2.softFocus;
        return v1;
    }

    /**
     * @brief 유효성 검증
     */
    static bool isValid(const BeautyFilterConfigV2& cfg) {
        return (cfg.intensity >= 0.0f && cfg.intensity <= 1.0f) &&
               (cfg.smoothing >= 0.0f && cfg.smoothing <= 1.0f) &&
               (cfg.brightness >= 0.5f && cfg.brightness <= 1.5f) &&
               (cfg.softFocus >= 0.0f && cfg.softFocus <= 1.0f) &&
               (cfg.whitening >= 0.0f && cfg.whitening <= 1.0f) &&
               (cfg.colorBalance >= -1.0f && cfg.colorBalance <= 1.0f) &&
               (cfg.wrinkleRemove >= 0.0f && cfg.wrinkleRemove <= 1.0f) &&
               (cfg.skinQuality >= 0.0f && cfg.skinQuality <= 1.0f) &&
               (cfg.slimFace >= 0.0f && cfg.slimFace <= 1.0f) &&
               (cfg.enlargeEyes >= 0.0f && cfg.enlargeEyes <= 1.0f) &&
               (cfg.thinChin >= 0.0f && cfg.thinChin <= 1.0f) &&
               (cfg.downscaleFactor >= 1 && cfg.downscaleFactor <= 4);
    }

    /**
     * @brief 범위 내로 클램핑
     */
    static void clamp(BeautyFilterConfigV2& cfg) {
        auto clampf = [](float v, float lo, float hi) {
            return v < lo ? lo : (v > hi ? hi : v);
        };
        cfg.intensity = clampf(cfg.intensity, 0.0f, 1.0f);
        cfg.smoothing = clampf(cfg.smoothing, 0.0f, 1.0f);
        cfg.brightness = clampf(cfg.brightness, 0.5f, 1.5f);
        cfg.softFocus = clampf(cfg.softFocus, 0.0f, 1.0f);
        cfg.whitening = clampf(cfg.whitening, 0.0f, 1.0f);
        cfg.colorBalance = clampf(cfg.colorBalance, -1.0f, 1.0f);
        cfg.wrinkleRemove = clampf(cfg.wrinkleRemove, 0.0f, 1.0f);
        cfg.skinQuality = clampf(cfg.skinQuality, 0.0f, 1.0f);
        cfg.slimFace = clampf(cfg.slimFace, 0.0f, 1.0f);
        cfg.enlargeEyes = clampf(cfg.enlargeEyes, 0.0f, 1.0f);
        cfg.thinChin = clampf(cfg.thinChin, 0.0f, 1.0f);
        cfg.downscaleFactor = cfg.downscaleFactor < 1 ? 1 :
                              (cfg.downscaleFactor > 4 ? 4 : cfg.downscaleFactor);
    }

    /**
     * @brief 기본값으로 초기화
     */
    static BeautyFilterConfigV2 defaults() {
        BeautyFilterConfigV2 cfg = {};
        cfg.enabled = false;
        cfg.intensity = 0.5f;
        cfg.smoothing = 0.0f;
        cfg.brightness = 1.0f;
        cfg.softFocus = 0.0f;
        cfg.whitening = 0.0f;
        cfg.colorBalance = 0.0f;
        cfg.wrinkleRemove = 0.0f;
        cfg.skinQuality = 0.0f;
        cfg.slimFace = 0.0f;
        cfg.enlargeEyes = 0.0f;
        cfg.thinChin = 0.0f;
        cfg.useGpu = true;
        cfg.roiOnly = true;
        cfg.protectEyes = true;
        cfg.protectLips = true;
        cfg.downscaleFactor = 1;
        return cfg;
    }
};

} // namespace iris_sdk

#endif /* __cplusplus */

#endif /* IRIS_SDK_BEAUTY_FILTER_H */
