/**
 * @file sdk_api.h
 * @brief IrisLensSDK C API
 *
 * C 언어 호환 API로, JNI, Obj-C++, dart:ffi, WASM 등 모든 바인딩 레이어에서 사용합니다.
 * 모든 구조체는 POD(Plain Old Data) 타입으로 FFI 호환성을 보장합니다.
 *
 * @author IrisLensSDK Team
 * @version 1.0.0
 * @copyright Apache 2.0 License
 */

#ifndef IRIS_SDK_API_H
#define IRIS_SDK_API_H

#include "export.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// 에러 코드
// ============================================================================

/**
 * @brief SDK 에러 코드
 *
 * 모든 SDK 함수는 이 열거형을 반환하여 작업 결과를 나타냅니다.
 * C 호환성을 위해 enum class가 아닌 일반 enum을 사용합니다.
 */
typedef enum IrisSdkError {
    /** @brief 성공 */
    IRIS_SDK_OK = 0,

    /* 초기화 에러 (100-199) */
    /** @brief SDK가 초기화되지 않음 */
    IRIS_SDK_NOT_INITIALIZED = 100,
    /** @brief SDK가 이미 초기화됨 */
    IRIS_SDK_ALREADY_INITIALIZED = 101,
    /** @brief 모델 로드 실패 */
    IRIS_SDK_MODEL_LOAD_FAILED = 102,
    /** @brief 잘못된 경로 */
    IRIS_SDK_INVALID_PATH = 103,

    /* 파라미터 에러 (200-299) */
    /** @brief 잘못된 파라미터 */
    IRIS_SDK_INVALID_PARAM = 200,
    /** @brief 널 포인터 */
    IRIS_SDK_NULL_POINTER = 201,
    /** @brief 지원하지 않는 프레임 포맷 */
    IRIS_SDK_INVALID_FORMAT = 202,

    /* 검출 에러 (300-399) */
    /** @brief 검출 실패 */
    IRIS_SDK_DETECTION_FAILED = 300,
    /** @brief 얼굴 미검출 */
    IRIS_SDK_NO_FACE = 301,

    /* 렌더링 에러 (400-499) */
    /** @brief 렌더링 실패 */
    IRIS_SDK_RENDER_FAILED = 400,
    /** @brief 텍스처 미로드 */
    IRIS_SDK_NO_TEXTURE = 401,

    /* 기능 에러 (500-599) */
    /** @brief 지원하지 않는 기능 */
    IRIS_SDK_ERROR_NOT_SUPPORTED = 500,
    /**
     * @brief 초기화되지 않음 (deprecated alias)
     * @deprecated NotInitialized 정본은 IRIS_SDK_NOT_INITIALIZED=100.
     *             이 alias(501)는 ABI 호환을 위해 1.x에서 유지하며 2.0에서 삭제 예정. (W4-A 정정)
     */
    IRIS_SDK_ERROR_NOT_INITIALIZED = 501,

    /** @brief 알 수 없는 에러 */
    IRIS_SDK_UNKNOWN = 999
} IrisSdkError;

// ============================================================================
// 프레임 포맷
// ============================================================================

/**
 * @brief 프레임 픽셀 포맷
 *
 * 입력 이미지의 픽셀 포맷을 지정합니다.
 */
typedef enum IrisFrameFormat {
    IRIS_FORMAT_RGBA = 0,       /**< 32비트 RGBA (각 채널 8비트) */
    IRIS_FORMAT_BGRA = 1,       /**< 32비트 BGRA (각 채널 8비트) */
    IRIS_FORMAT_RGB = 2,        /**< 24비트 RGB (각 채널 8비트) */
    IRIS_FORMAT_BGR = 3,        /**< 24비트 BGR (각 채널 8비트) */
    IRIS_FORMAT_NV21 = 4,       /**< Android 카메라 YUV420sp 포맷 */
    IRIS_FORMAT_NV12 = 5,       /**< iOS 카메라 YUV420sp 포맷 */
    IRIS_FORMAT_GRAY = 6        /**< 8비트 그레이스케일 */
} IrisFrameFormat;

// ============================================================================
// Eye Refiner 정책
// ============================================================================

/**
 * @brief Eye Refiner 실행 정책
 *
 * 2차 눈 정밀화 모델(iris_landmark)의 실행 조건을 지정합니다.
 *
 * @deprecated ④ W4-D에서 삭제 예정 — C++ 미러 EyeRefinerPolicy(types.h)와 함께. 추적
 *   외부화로 Eye Refiner(iris_landmark 2차 추론) 소멸(ADR §6.2). iris_sdk_set_eye_refiner_policy는
 *   이미 no-op(detector 미연결). 호출자 0이라 W4-E deprecation 마킹 후 2.0 삭제(ADR §8.2 패턴).
 */
typedef enum IrisEyeRefinerPolicy {
    IRIS_EYE_REFINER_ALWAYS = 0,        /**< 항상 실행 (HQ 모드) */
    IRIS_EYE_REFINER_CONDITIONAL = 1,   /**< 조건부 실행 (기본값) */
    IRIS_EYE_REFINER_NEVER = 2          /**< 비활성화 (저사양 기기) */
} IrisEyeRefinerPolicy;

// ============================================================================
// 블렌드 모드
// ============================================================================

/**
 * @brief 렌즈 블렌딩 모드
 *
 * 렌즈 렌더링 시 프레임과 텍스처를 합성하는 방식을 지정합니다.
 *
 * P6-W2 §5.4/§5.12: Canonical default = `IRIS_BLEND_LUMINANCE_TINT_LINEAR` (ID=5, TintLinearV2).
 *   활성 ID = {0, 1, 2, 5, 7}. ID 3/4/6은 deprecated이며 셰이더에서 ID 5 fallback.
 *   외부 API 호환성 유지를 위해 enum 값은 보존됨.
 */
typedef enum IrisBlendMode {
    IRIS_BLEND_NORMAL = 0,              /**< 일반 알파 블렌딩 */
    IRIS_BLEND_MULTIPLY = 1,            /**< 곱하기 블렌딩 */
    IRIS_BLEND_SCREEN = 2,              /**< 스크린 (W2: 선형 공간 ScreenLinear) */
    IRIS_BLEND_OVERLAY = 3,             /**< @deprecated TintLinearV2 fallback */
    IRIS_BLEND_LUMINANCE_TINT = 4,      /**< @deprecated TintLinearV2 fallback */
    IRIS_BLEND_LUMINANCE_TINT_LINEAR = 5,/**< 휘도 보존 틴트 — **canonical default** (W2 TintLinearV2) */
    IRIS_BLEND_SOFT_LIGHT = 6,          /**< @deprecated TintLinearV2 fallback */
    IRIS_BLEND_COLOR_REPLACE = 7        /**< 색상 교체 (W2: ColorReplaceLinear, B1 벤치 대기) */
} IrisBlendMode;

// ============================================================================
// 데이터 구조체
// ============================================================================

/**
 * @brief 홍채 랜드마크 좌표
 *
 * 정규화된 좌표 (0.0~1.0) 및 가시성 점수를 포함합니다.
 * POD 타입으로 FFI 호환성을 보장합니다.
 */
typedef struct IrisLandmark {
    float x;            /**< X 좌표 (정규화, 0.0~1.0) */
    float y;            /**< Y 좌표 (정규화, 0.0~1.0) */
    float z;            /**< Z 좌표 (깊이, 정규화) */
    float visibility;   /**< 가시성 점수 (0.0~1.0) */
} IrisLandmark;

/**
 * @brief 사각형 영역
 *
 * 바운딩 박스 표현에 사용됩니다.
 * POD 타입으로 FFI 호환성을 보장합니다.
 */
typedef struct IrisRect {
    float x;            /**< 좌상단 X 좌표 */
    float y;            /**< 좌상단 Y 좌표 */
    float width;        /**< 너비 */
    float height;       /**< 높이 */
} IrisRect;

/**
 * @brief 홍채 검출 결과
 *
 * 양쪽 눈의 홍채 정보 및 얼굴 메타데이터를 포함합니다.
 * POD 타입으로 FFI 호환성을 보장합니다.
 *
 * @note face_mesh 배열은 메모리 효율을 위해 face_mesh_valid가 true일 때만 유효합니다.
 */
typedef struct IrisResult {
    /* 검출 상태 */
    bool detected;              /**< 전체 검출 성공 여부 */
    bool left_detected;         /**< 왼쪽 눈 검출 여부 */
    bool right_detected;        /**< 오른쪽 눈 검출 여부 */
    float confidence;           /**< 전체 신뢰도 (0.0~1.0) */

    /* 왼쪽 눈 홍채 (5개 랜드마크: center + 4 boundary) */
    IrisLandmark left_iris[5];
    float left_radius;          /**< 왼쪽 홍채 반지름 (픽셀) */

    /* 오른쪽 눈 홍채 (5개 랜드마크: center + 4 boundary) */
    IrisLandmark right_iris[5];
    float right_radius;         /**< 오른쪽 홍채 반지름 (픽셀) */

    /* 얼굴 메타데이터 */
    IrisRect face_rect;         /**< 얼굴 바운딩 박스 */
    float face_rotation[3];     /**< 얼굴 회전 [pitch, yaw, roll] (도) */

    /* Face Mesh (478 랜드마크, 디버그/시각화용) */
    IrisLandmark face_mesh[478];    /**< 전체 얼굴 메쉬 랜드마크 */
    bool face_mesh_valid;           /**< face_mesh 데이터 유효 여부 */

    /* 프레임 정보 */
    int64_t timestamp_ms;       /**< 타임스탬프 (밀리초) */
    int32_t frame_width;        /**< 원본 프레임 너비 */
    int32_t frame_height;       /**< 원본 프레임 높이 */

    /* 눈꺼풀 가림 비율 (W3 트랙).
       ④ W4-D: detector 전용 메타 iris_quality_left/right, eye_refiner_used 제거(ADR §6.2,
       C++ 미러 types.h와 동반). eyelid_ratio는 W3용 별도 트랙으로 보존. */
    float eyelid_ratio_left;    /**< 왼쪽 눈꺼풀 가림 비율 (0.0~1.0) */
    float eyelid_ratio_right;   /**< 오른쪽 눈꺼풀 가림 비율 (0.0~1.0) */

    /* P7-W2: iris ROI 실측 평균 luma (srgb²+Rec.709 linear, 0~1, -1=미측정).
       C++ iris_sdk::IrisResult와 동일 레이아웃 유지(sdk_api_v2.cpp reinterpret_cast). */
    float avg_iris_luma_left;   /**< 왼쪽 홍채 ROI 평균 linear luma (-1=미측정) */
    float avg_iris_luma_right;  /**< 오른쪽 홍채 ROI 평균 linear luma (-1=미측정) */
} IrisResult;

/**
 * @brief 렌즈 렌더링 설정
 *
 * 가상 렌즈 오버레이 파라미터를 포함합니다.
 * POD 타입으로 FFI 호환성을 보장합니다.
 */
typedef struct IrisLensConfig {
    float opacity;              /**< 투명도 (0.0~1.0, 기본값 0.7) */
    float scale;                /**< 크기 배율 (기본값 1.0) */
    float offset_x;             /**< X 오프셋 (정규화, 기본값 0.0) */
    float offset_y;             /**< Y 오프셋 (정규화, 기본값 0.0) */
    float rotation;             /**< 회전 각도 (라디안, -PI~PI, 기본값 0.0) */
    IrisBlendMode blend_mode;   /**< 블렌드 모드 (기본값 LUMINANCE_TINT_LINEAR, P6-W2 §5.12) */
    float edge_feather;         /**< 가장자리 페더링 (0.0~1.0, 기본값 0.1) */
    bool apply_left;            /**< 왼쪽 눈 적용 여부 (기본값 true) */
    bool apply_right;           /**< 오른쪽 눈 적용 여부 (기본값 true) */
    bool is_mirror;             /**< 전면 카메라 mirror 여부 (기본값 false) */
} IrisLensConfig;

/**
 * @brief SDK 설정
 *
 * SDK 초기화 시 사용되는 설정 구조체입니다.
 * POD 타입으로 FFI 호환성을 보장합니다.
 */
typedef struct IrisSdkConfig {
    const char* model_path;     /**< 모델 파일 디렉토리 경로 (필수) */
    float min_confidence;       /**< 최소 검출 신뢰도 (0.0~1.0, 기본값 0.5) */
    int max_faces;              /**< 최대 얼굴 수 (기본값 1) */
    bool enable_gpu;            /**< GPU 가속 사용 여부 (기본값 false, 현재 미지원) */
    int num_threads;            /**< 스레드 수 (0=자동, 기본값 0) */
} IrisSdkConfig;

// ============================================================================
// 라이프사이클 함수
// ============================================================================

/**
 * @brief SDK 초기화 (간단 버전)
 *
 * 기본 설정으로 SDK를 초기화합니다.
 *
 * @param model_path 모델 파일 디렉토리 경로 (NULL 불가)
 * @return IRIS_SDK_OK 성공, 그 외 에러 코드
 *
 * @code
 * IrisSdkError err = iris_sdk_init("/path/to/models");
 * if (err != IRIS_SDK_OK) {
 *     printf("Init failed: %s\n", iris_sdk_error_to_string(err));
 * }
 * @endcode
 */
IRIS_SDK_EXPORT IrisSdkError iris_sdk_init(const char* model_path);

/**
 * @brief SDK 초기화 (상세 설정)
 *
 * 사용자 정의 설정으로 SDK를 초기화합니다.
 *
 * @param config SDK 설정 구조체 포인터 (NULL 불가)
 * @return IRIS_SDK_OK 성공, 그 외 에러 코드
 *
 * @code
 * IrisSdkConfig config = {0};
 * config.model_path = "/path/to/models";
 * config.min_confidence = 0.7f;
 * config.max_faces = 1;
 * IrisSdkError err = iris_sdk_init_with_config(&config);
 * @endcode
 */
IRIS_SDK_EXPORT IrisSdkError iris_sdk_init_with_config(const IrisSdkConfig* config);

/**
 * @brief SDK 종료
 *
 * 모든 리소스를 해제하고 초기화 전 상태로 되돌립니다.
 * 여러 번 호출해도 안전합니다.
 */
IRIS_SDK_EXPORT void iris_sdk_destroy(void);

/**
 * @brief SDK 준비 상태 확인
 *
 * @return true SDK가 초기화되어 사용 가능, false 그 외
 */
IRIS_SDK_EXPORT bool iris_sdk_is_ready(void);

// ============================================================================
// 검출/처리 함수
// ============================================================================
// ④ W4-D: 검출 인프라(detector/InferenceThread)를 코어에서 제거하면서
//   iris_sdk_detect / iris_sdk_detect_with_rotation / iris_sdk_process C API가
//   제거되었습니다. 검출(랜드마크)은 외부 추적 글루가 책임지며 주입 경로
//   (iris_set_landmarks)로 코어에 전달됩니다. 렌더는 iris_sdk_render_with_result로.

// ============================================================================
// 렌더링 함수
// ============================================================================

/**
 * @brief 렌즈 텍스처 로드 (파일)
 *
 * 파일에서 렌즈 텍스처 이미지를 로드합니다.
 * 지원 포맷: PNG, JPEG, BMP
 *
 * @param path 텍스처 이미지 파일 경로
 * @return IRIS_SDK_OK 성공, 그 외 에러 코드
 *
 * @deprecated cpu-render/CPU 픽셀 경로는 2.0에서 제거됩니다(ADR-0001 §8.2).
 *   GPU 텍스처 경로(sdk_api_v2)로 이행하세요. 1.x 동안 동작은 유지됩니다.
 */
IRIS_SDK_EXPORT IRIS_SDK_DEPRECATED IrisSdkError iris_sdk_load_texture(const char* path);

/**
 * @brief 렌즈 텍스처 로드 (메모리)
 *
 * 메모리에서 RGBA 텍스처를 로드합니다.
 *
 * @param data RGBA 픽셀 데이터 (4바이트/픽셀)
 * @param width 텍스처 너비
 * @param height 텍스처 높이
 * @return IRIS_SDK_OK 성공, 그 외 에러 코드
 *
 * @deprecated cpu-render/CPU 픽셀 경로는 2.0에서 제거됩니다(ADR-0001 §8.2).
 *   GPU 텍스처 경로(sdk_api_v2)로 이행하세요. 1.x 동안 동작은 유지됩니다.
 */
IRIS_SDK_EXPORT IRIS_SDK_DEPRECATED IrisSdkError iris_sdk_load_texture_from_memory(
    const uint8_t* data,
    int width,
    int height);

/**
 * @brief 렌즈 렌더링
 *
 * 검출된 홍채 위치에 렌즈 텍스처를 오버레이합니다.
 * 프레임 데이터는 in-place로 수정됩니다.
 *
 * @param frame_data 프레임 데이터 (in-place 수정됨)
 * @param width 프레임 너비
 * @param height 프레임 높이
 * @param format 픽셀 포맷
 * @param iris_result 홍채 검출 결과
 * @param config 렌더링 설정
 * @return IRIS_SDK_OK 성공, 그 외 에러 코드
 *
 * @deprecated cpu-render/CPU 픽셀 경로는 2.0에서 제거됩니다(ADR-0001 §8.2).
 *   GPU 텍스처 경로(sdk_api_v2)로 이행하세요. 1.x 동안 동작은 유지됩니다.
 */
IRIS_SDK_EXPORT IRIS_SDK_DEPRECATED IrisSdkError iris_sdk_render_lens(
    uint8_t* frame_data,
    int width,
    int height,
    IrisFrameFormat format,
    const IrisResult* iris_result,
    const IrisLensConfig* config);

// ============================================================================
// 설정 함수
// ============================================================================

/**
 * @brief 런타임 설정 변경
 *
 * 런타임에 SDK 설정을 변경합니다.
 *
 * @param key 설정 키 (예: "min_confidence", "face_tracking")
 * @param value 설정 값 (문자열)
 * @return IRIS_SDK_OK 성공, 그 외 에러 코드
 *
 * @note 지원되는 설정 키:
 *       - "min_confidence": 최소 검출 신뢰도 (0.0~1.0)
 *       - "face_tracking": 얼굴 추적 활성화 ("true" / "false")
 */
IRIS_SDK_EXPORT IrisSdkError iris_sdk_set_config(const char* key, const char* value);

/**
 * @brief 기본 렌즈 설정 가져오기
 *
 * 기본값으로 초기화된 IrisLensConfig를 반환합니다.
 *
 * @param config 설정 구조체 포인터 (NULL 불가)
 */
IRIS_SDK_EXPORT void iris_sdk_default_lens_config(IrisLensConfig* config);

// ============================================================================
// GPU 가속 API
// ============================================================================

/**
 * @brief GPU 가속 사용 여부 설정
 *
 * TFLite GPU Delegate를 사용할지 여부를 설정합니다.
 * 반드시 iris_sdk_init() 호출 전에 설정해야 합니다.
 *
 * @param enable true면 GPU 가속 시도, false면 CPU만 사용
 *
 * @note GPU delegate 사용 조건:
 *       - Android: OpenGL ES 3.1 이상 지원 기기
 *       - iOS: Metal 지원 기기 (현재 미지원)
 * @note init() 후 호출 시 설정이 무시됩니다.
 */
IRIS_SDK_EXPORT void iris_sdk_set_gpu_enabled(bool enable);

/**
 * @brief GPU 가속 사용 가능 여부 확인
 *
 * SDK가 GPU delegate와 함께 빌드되었는지 확인합니다.
 * 컴파일 시점에 결정됩니다.
 *
 * @return true면 GPU delegate 라이브러리가 포함됨, false면 CPU만 사용 가능
 */
IRIS_SDK_EXPORT bool iris_sdk_is_gpu_available(void);

/**
 * @brief 현재 GPU 사용 상태 확인
 *
 * 런타임에 실제로 GPU delegate가 활성화되어 있는지 확인합니다.
 *
 * @return true면 GPU 사용 중, false면 CPU 사용 중
 *
 * @note init() 호출 후에만 정확한 값을 반환합니다.
 */
IRIS_SDK_EXPORT bool iris_sdk_is_using_gpu(void);

// ============================================================================
// 신뢰도 설정 (Confidence Settings)
// ============================================================================

/**
 * @brief 얼굴 검출 최소 신뢰도 설정
 *
 * 얼굴 검출 결과의 최소 신뢰도. 이 값 이하면 검출되지 않은 것으로 처리.
 *
 * @param min_confidence 최소 신뢰도 (0.0 ~ 1.0), 기본값 0.3
 *
 * @note init() 전에 설정하는 것이 권장됩니다.
 */
IRIS_SDK_EXPORT void iris_sdk_set_min_detection_confidence(float min_confidence);

/**
 * @brief 랜드마크 추적 최소 신뢰도 설정
 *
 * 랜드마크 추적 결과의 최소 신뢰도.
 * 이 값 이하면 추적 실패로 간주하고 다시 Face Detection 수행.
 *
 * @param min_confidence 최소 신뢰도 (0.0 ~ 1.0), 기본값 0.5
 *
 * @note init() 전에 설정하는 것이 권장됩니다.
 */
IRIS_SDK_EXPORT void iris_sdk_set_min_tracking_confidence(float min_confidence);

/**
 * @brief 얼굴 존재 최소 신뢰도 설정
 *
 * 추적 모드에서 이전 프레임 결과를 재사용할지 판단하는 임계값.
 * 이전 프레임의 confidence가 이 값 이상이어야 Face Detection을 스킵.
 * 이 값 미만이면 캐시를 무효화하고 다시 Face Detection 수행.
 *
 * @param min_confidence 최소 신뢰도 (0.0 ~ 1.0), 기본값 0.5
 *
 * @note init() 전에 설정하는 것이 권장됩니다.
 */
IRIS_SDK_EXPORT void iris_sdk_set_min_presence_confidence(float min_confidence);

/**
 * @brief InferenceThread 사용 여부 설정 (벤치마크용)
 *
 * 반드시 iris_sdk_init() 호출 전에 설정해야 합니다.
 * false로 설정하면 전용 스레드 없이 직접 호출합니다.
 * GPU 가속은 InferenceThread 사용 시에만 지원됩니다.
 *
 * @param enable true면 InferenceThread 사용 (기본값), false면 직접 호출
 *
 * @note 벤치마크 용도로 사용됩니다.
 */
IRIS_SDK_EXPORT void iris_sdk_set_use_inference_thread(bool enable);

/**
 * @brief 현재 InferenceThread 사용 상태 확인
 *
 * @return true면 InferenceThread 사용, false면 직접 호출
 */
IRIS_SDK_EXPORT bool iris_sdk_is_using_inference_thread(void);

// ============================================================================
// Eye Refiner 설정
// ============================================================================

/**
 * @brief Eye Refiner 정책 설정
 *
 * V2 모델 사용 시 iris_landmark 모델을 2차 홍채 정밀화에 활용하는 정책을 설정합니다.
 * 반드시 iris_sdk_init() 호출 전에 설정해야 합니다.
 *
 * @param policy Eye Refiner 실행 정책
 * @return IRIS_SDK_OK 성공, 그 외 에러 코드
 *
 * @warning NO-OP STUB (③-2 B1 문서화): 현재 구현은 본문이 비어 있어 어떤 정책도
 *          실제로 적용되지 않으며, 그럼에도 IRIS_SDK_OK를 반환한다(조용한 성공).
 *          내부 기본값은 EyeRefinerPolicy::Never이다. 호출자가 정책을 설정해도
 *          검출 동작은 바뀌지 않는다.
 * @todo (④ W4-B2 결정) 실제 연결은 폐기한다 — 추적 외부화로 Eye Refiner 자체가
 *       소멸하므로(ADR §6.2) 연결할 대상이 없다. enum(IrisEyeRefinerPolicy/C++ 미러
 *       EyeRefinerPolicy)과 함께 W4-D 삭제 대상이며, 호출자 0이라 시그니처는 W4-E에서
 *       IRIS_SDK_DEPRECATED 마킹 후 2.0 제거(ADR §8.2 패턴). 그때까지 no-op stub 유지
 *       (반환값 변경도 surface 호환을 위해 보류).
 */
IRIS_SDK_EXPORT IrisSdkError iris_sdk_set_eye_refiner_policy(IrisEyeRefinerPolicy policy);

// ============================================================================
// 정보 함수
// ============================================================================

/**
 * @brief SDK 버전 문자열 반환
 *
 * @return 버전 문자열 (예: "1.0.0"). 정적 문자열로 해제 불필요.
 */
IRIS_SDK_EXPORT const char* iris_sdk_get_version(void);

/**
 * @brief 빌드 정보 문자열 반환
 *
 * @return 빌드 정보 (예: "IrisLensSDK v1.0.0 (Debug, 2024-01-01)"). 정적 문자열로 해제 불필요.
 */
IRIS_SDK_EXPORT const char* iris_sdk_get_build_info(void);

/**
 * @brief 마지막 에러 메시지 반환
 *
 * @return 마지막 에러 상세 메시지. 정적 버퍼로 해제 불필요.
 *         다음 SDK 호출 시 덮어쓸 수 있음.
 */
IRIS_SDK_EXPORT const char* iris_sdk_get_last_error(void);

/**
 * @brief 에러 코드를 문자열로 변환
 *
 * @param error 에러 코드
 * @return 에러 설명 문자열 (예: "IRIS_SDK_OK"). 정적 문자열로 해제 불필요.
 */
IRIS_SDK_EXPORT const char* iris_sdk_error_to_string(IrisSdkError error);

// ============================================================================
// 메모리 관리 함수
// ============================================================================

/**
 * @brief 결과 구조체 초기화
 *
 * IrisResult 구조체를 초기값으로 리셋합니다.
 * SDK에서 할당한 내부 리소스가 있다면 해제합니다.
 *
 * @param result 결과 구조체 포인터 (NULL 가능, NULL이면 무시)
 */
IRIS_SDK_EXPORT void iris_sdk_free_result(IrisResult* result);

// ============================================================================
// 뷰티 필터 V2 GPU API (Phase 2)
// ============================================================================

/**
 * @brief 뷰티 필터 V2 설정 (C API용 POD 구조체)
 *
 * FFI 호환을 위한 POD 타입 구조체입니다.
 * JNI, Obj-C++, dart:ffi, WASM 등 모든 바인딩에서 사용됩니다.
 */
typedef struct IrisBeautyConfigV2 {
    /* 기본 (V1 호환) */
    int enabled;            /**< 필터 활성화 여부 (0=비활성, 1=활성) */
    float intensity;        /**< 전체 강도 (0.0~1.0) */
    float brightness;       /**< 밝기 조절 (0.5~1.5, 1.0=원본) */

    /* 얼굴 형태 보정 */
    float slim_face;        /**< 얼굴 슬림화 (0.0~1.0) */
    float enlarge_eyes;     /**< 눈 확대 (0.0~1.0) */
    float thin_chin;        /**< 턱 축소 (0.0~1.0) */

    /* 처리 옵션 */
    int use_gpu;            /**< GPU 가속 사용 (0=CPU, 1=GPU) */
    int roi_only;           /**< 얼굴 영역만 처리 (0=전체, 1=ROI만) */
    int protect_eyes;       /**< 눈 영역 보호 (0=미보호, 1=보호) */
    int protect_lips;       /**< 입술 영역 보호 (0=미보호, 1=보호) */
    int downscale_factor;   /**< 다운스케일 팩터 (1, 2, 4) */
    int feather_radius;     /**< ROI 페더링 반경 (픽셀) */

    /* 추가 보호 옵션 */
    int protect_nose;          /**< 코 보호 (0=비활성, 1=활성, 기본 0) */
} IrisBeautyConfigV2;

/**
 * @brief 기본 V2 뷰티 필터 설정 반환
 *
 * 기본값으로 초기화된 IrisBeautyConfigV2를 설정합니다.
 *
 * @param config 설정 구조체 포인터 (NULL 불가)
 */
IRIS_SDK_EXPORT void iris_sdk_default_beauty_config_v2_c(IrisBeautyConfigV2* config);

/**
 * @brief 뷰티 필터 V2 적용 (CPU 버퍼)
 *
 * CPU 메모리 버퍼에 V2 뷰티 필터를 적용합니다.
 * 프레임 데이터는 in-place로 수정됩니다.
 *
 * @param frame_data 프레임 데이터 (in-place 수정됨)
 * @param width 프레임 너비
 * @param height 프레임 높이
 * @param format 픽셀 포맷
 * @param config V2 뷰티 필터 설정
 * @param detection 얼굴 검출 결과 (NULL 가능, NULL이면 전체 프레임 처리)
 * @return IRIS_SDK_OK 성공, 그 외 에러 코드
 *
 * @deprecated cpu-render/CPU 픽셀 경로(CPU 픽셀 버퍼)는 2.0에서 제거됩니다(ADR-0001 §8.2).
 *   GPU 텍스처 뷰티 경로(apply_beauty_texture_v2)로 이행하세요. 1.x 동안 동작은 유지됩니다.
 */
IRIS_SDK_EXPORT IRIS_SDK_DEPRECATED IrisSdkError iris_sdk_apply_beauty_v2_c(
    uint8_t* frame_data,
    int width, int height,
    IrisFrameFormat format,
    const IrisBeautyConfigV2* config,
    const IrisResult* detection
);

/**
 * @brief GPU 뷰티 백엔드 초기화
 *
 * OpenGL ES 기반 GPU 뷰티 필터 백엔드를 초기화합니다.
 * Android에서는 EGL 컨텍스트가 현재 스레드에 바인딩되어 있어야 합니다.
 *
 * @return IRIS_SDK_OK 성공, IRIS_SDK_ERROR_NOT_SUPPORTED GPU 미지원
 */
IRIS_SDK_EXPORT IrisSdkError iris_sdk_init_gpu_beauty(void);

/**
 * @brief GPU 뷰티 백엔드 해제
 *
 * GPU 리소스를 해제합니다.
 * OpenGL 컨텍스트가 여전히 유효해야 합니다.
 */
IRIS_SDK_EXPORT void iris_sdk_release_gpu_beauty(void);

/**
 * @brief GPU 뷰티 백엔드 초기화 여부 확인
 *
 * @return 1 초기화됨, 0 미초기화
 */
IRIS_SDK_EXPORT int iris_sdk_is_gpu_beauty_initialized(void);

/**
 * @brief 뷰티 필터 V2 적용 (GPU 텍스처)
 *
 * OpenGL ES 텍스처에 V2 뷰티 필터를 적용합니다.
 * GPU 뷰티 백엔드가 초기화되어 있어야 합니다.
 *
 * @param input_texture 입력 OpenGL ES 텍스처 ID
 * @param output_texture 출력 텍스처 ID 포인터 (SDK가 관리하는 텍스처 반환)
 * @param width 텍스처 너비
 * @param height 텍스처 높이
 * @param config V2 뷰티 필터 설정
 * @param detection 얼굴 검출 결과 (NULL 가능)
 * @return IRIS_SDK_OK 성공, IRIS_SDK_NOT_INITIALIZED GPU 미초기화 (W4-A: 501 alias→100 정본)
 */
IRIS_SDK_EXPORT IrisSdkError iris_sdk_apply_beauty_texture_v2(
    uint32_t input_texture,
    uint32_t* output_texture,
    int width, int height,
    const IrisBeautyConfigV2* config,
    const IrisResult* detection
);

/**
 * @brief P8-W1: landmark-masked skin smoothing 모드 설정 (internal, 벤치/A-B용)
 *
 * LensSimulator에서 검증된 랜드마크 폴리곤 마스크 기반 피부 보정 경로를 토글합니다.
 * 이것이 SDK의 유일한 피부 스무딩 경로입니다(레거시 FreqSep/Bilateral은 P8-W2에서 제거).
 * 활성 시 랜드마크 마스크 기반 스무딩을 적용하며(다른 패스는 불변), 비활성 또는
 * strength 0이면 마스크/블러/필터/저해상도 타깃 생성을 전부 생략합니다(비용 0=스무딩 없음).
 * GL 스레드(GPU 뷰티 백엔드 초기화 스레드)에서 호출하세요.
 *
 * @param enabled 0=off(스무딩 없음), 1=on
 * @param strength 피부 스무딩 강도 (0.0~1.0). 0이면 모드 활성이어도 패스 생략
 */
IRIS_SDK_EXPORT void iris_sdk_set_skin_mask_smoothing(int enabled, float strength);

/**
 * @brief Face Warp 적용 (GPU)
 *
 * GPU에서 얼굴 형태 보정(Face Warp)을 적용합니다.
 * 슬림 페이스, 눈 확대, 턱 축소 등의 효과를 렌더링합니다.
 *
 * @param input_texture 입력 OpenGL ES 텍스처 ID
 * @param output_texture 출력 텍스처 ID 포인터
 * @param width 텍스처 너비
 * @param height 텍스처 높이
 * @param slim_face 얼굴 슬림화 강도 (0.0~1.0)
 * @param thin_chin 턱 축소 강도 (0.0~1.0)
 * @param enlarge_eyes 눈 확대 강도 (0.0~1.0)
 * @param detection 얼굴 검출 결과 (필수, NULL이면 pass-through)
 * @return IRIS_SDK_OK 성공
 */
IRIS_SDK_EXPORT IrisSdkError iris_sdk_apply_face_warp(
    uint32_t input_texture,
    uint32_t* output_texture,
    int width, int height,
    float slim_face,
    float thin_chin,
    float enlarge_eyes,
    const IrisResult* detection
);

/**
 * @brief SDK 관리 텍스처 해제
 *
 * SDK가 내부적으로 관리하는 텍스처를 해제합니다.
 * 외부에서 생성한 텍스처를 전달하면 무시됩니다.
 *
 * @param texture 해제할 텍스처 ID
 * @return IRIS_SDK_OK 성공, IRIS_SDK_INVALID_PARAM 관리 대상 아님
 */
IRIS_SDK_EXPORT IrisSdkError iris_sdk_release_texture(uint32_t texture);

/**
 * @brief 텍스처가 SDK 관리인지 확인
 *
 * @param texture 확인할 텍스처 ID
 * @return 1 SDK 관리, 0 외부 텍스처
 */
IRIS_SDK_EXPORT int iris_sdk_is_texture_managed(uint32_t texture);

// ============================================================================
// GPU 렌즈 렌더링
// ============================================================================

/**
 * @brief GPU 렌즈 렌더러 초기화
 *
 * OpenGL ES 기반 GPU 렌즈 렌더러를 초기화합니다.
 * EGL 컨텍스트가 현재 스레드에 바인딩되어 있어야 합니다.
 */
IRIS_SDK_EXPORT IrisSdkError iris_sdk_init_gpu_lens(void);

/**
 * @brief GPU 렌즈 렌더러 해제
 */
IRIS_SDK_EXPORT void iris_sdk_release_gpu_lens(void);

/**
 * @brief GPU 렌즈 렌더러 초기화 여부 확인
 */
IRIS_SDK_EXPORT int iris_sdk_is_gpu_lens_initialized(void);

/**
 * @brief 렌즈 텍스처 로드 (RGBA 데이터)
 *
 * @param data RGBA 픽셀 데이터
 * @param width 너비
 * @param height 높이
 * @return IRIS_SDK_OK 성공
 */
IRIS_SDK_EXPORT IrisSdkError iris_sdk_load_lens_texture(
    const uint8_t* data, int width, int height);

/**
 * @brief 렌즈 SKU 메타데이터 등록 (P6-W7)
 *
 * lens_meta.json 문자열을 파싱하여 SKU 레지스트리를 구성합니다.
 * 등록된 메타는 GPU 렌즈 렌더러의 림발(limbal) on/off 판정 권위로 사용됩니다.
 * GPU 렌즈 초기화 전/후 어느 시점에 호출해도 무방하며, 가장 최근 등록이 우선합니다.
 *
 * @param lens_meta_json SKU 메타데이터 JSON 문자열 (NULL 불가)
 * @return IRIS_SDK_OK 성공, IRIS_SDK_NULL_POINTER 입력 NULL,
 *         IRIS_SDK_INVALID_FORMAT JSON 파싱 실패
 */
IRIS_SDK_EXPORT IrisSdkError iris_sdk_set_lens_metadata(const char* lens_meta_json);

/**
 * @brief 렌즈 텍스처 로드 (RGBA 데이터 + SKU ID) (P6-W7)
 *
 * iris_sdk_load_lens_texture와 동일하나 sku_id를 함께 전달하여
 * 등록된 SKU 메타(iris_sdk_set_lens_metadata)로 림발 판정을 수행합니다.
 *
 * @param data RGBA 픽셀 데이터
 * @param width 너비
 * @param height 높이
 * @param sku_id SKU 식별자 (NULL이면 빈 문자열로 처리, 메타 미적용)
 * @return IRIS_SDK_OK 성공
 */
IRIS_SDK_EXPORT IrisSdkError iris_sdk_load_lens_texture_with_sku(
    const uint8_t* data, int width, int height, const char* sku_id);

/**
 * @brief 렌즈 텍스처 해제
 */
IRIS_SDK_EXPORT void iris_sdk_unload_lens_texture(void);

/**
 * @brief GPU 렌즈 렌더링 (텍스처)
 *
 * @param input_texture 입력 카메라 프레임 텍스처 ID
 * @param output_texture 출력 텍스처 ID 포인터
 * @param width 프레임 너비
 * @param height 프레임 높이
 * @param detection 홍채 검출 결과 (필수)
 * @param config 렌즈 설정 (NULL이면 기본값)
 * @return IRIS_SDK_OK 성공
 */
IRIS_SDK_EXPORT IrisSdkError iris_sdk_render_lens_texture(
    uint32_t input_texture,
    uint32_t* output_texture,
    int width, int height,
    const IrisResult* detection,
    const IrisLensConfig* config);

/**
 * @brief GPU 렌즈 Sclera Protection 설정
 */
IRIS_SDK_EXPORT void iris_sdk_set_lens_sclera_protect(int enabled);

/**
 * @brief GPU 렌즈 타원 마스크 설정
 */
IRIS_SDK_EXPORT void iris_sdk_set_lens_ellipse_mask(int enabled);

/**
 * @brief GPU 렌즈 각막 하이라이트 설정
 */
IRIS_SDK_EXPORT void iris_sdk_set_lens_highlight(int enabled);

// ============================================================================
// Temporal Stabilizer API (P5-W1)
// ============================================================================

/**
 * @brief Stabilizer 설정 (C API용 POD 구조체)
 */
typedef struct IrisStabilizerConfig {
    float iris_min_cutoff;          /**< 홍채 중심 OneEuro min_cutoff (기본 4.0) */
    float iris_beta;                /**< 홍채 중심 OneEuro beta (기본 15.0) */
    float radius_min_cutoff;        /**< 반지름 OneEuro min_cutoff (기본 4.0) */
    float radius_beta;              /**< 반지름 OneEuro beta (기본 7.5) */
    float eyelid_min_cutoff;        /**< 눈꺼풀 OneEuro min_cutoff (기본 4.0) */
    float eyelid_beta;              /**< 눈꺼풀 OneEuro beta (기본 10.0) */
    float confidence_low_threshold; /**< 이력현상 하한 (기본 0.3) */
    float confidence_high_threshold;/**< 이력현상 상한 (기본 0.6) */
    int confidence_low_frames;      /**< 연속 낮은 confidence 프레임 수 (기본 3) */
    float fade_in_ms;               /**< Fade-in 시간 (기본 100ms) */
    float fade_out_ms;              /**< Fade-out 시간 (기본 200ms) */
    int hold_frames;                /**< Dropout hold 프레임 수 (기본 5) */
    float outlier_radius_multiplier;/**< 아웃라이어 판정 반지름 배수 (기본 4.0 — 코드 정합) */
    int outlier_confirm_frames;     /**< 아웃라이어 확인 프레임 수 (기본 1 — 코드 정합; ≥2라야 단일프레임 reject 활성) */
    float blink_ear_threshold;      /**< 눈깜빡임 EAR 임계값 (기본 0.2) */
} IrisStabilizerConfig;

/**
 * @brief Stabilized 결과 (C API용 POD 구조체)
 */
typedef struct IrisStabilizedResult {
    IrisResult raw;                 /**< 원본 raw 결과 */
    IrisResult stabilized;          /**< 스무딩된 결과 */
    float visibility;               /**< 전체 가시성 (0.0~1.0) */
    int is_held;                    /**< dropout hold 중 (0/1) */
    int64_t last_valid_ms;          /**< 마지막 유효 검출 타임스탬프 */
} IrisStabilizedResult;

/**
 * @brief 기본 Stabilizer 설정 반환
 * @param config 설정 구조체 포인터
 */
IRIS_SDK_EXPORT void iris_sdk_default_stabilizer_config(IrisStabilizerConfig* config);

/**
 * @brief Temporal Stabilizer 생성
 *
 * @param config 설정 (NULL이면 기본값 사용)
 * @return 핸들 (0이면 실패)
 */
IRIS_SDK_EXPORT int64_t iris_sdk_create_stabilizer(const IrisStabilizerConfig* config);

/**
 * @brief Temporal Stabilizer 해제
 * @param handle iris_sdk_create_stabilizer()에서 반환된 핸들
 */
IRIS_SDK_EXPORT void iris_sdk_destroy_stabilizer(int64_t handle);

/**
 * @brief 검출 결과 스무딩
 *
 * @param handle Stabilizer 핸들
 * @param raw 원본 검출 결과
 * @param timestamp_sec 타임스탬프 (초 단위)
 * @param out 스무딩된 결과 출력
 * @return IRIS_SDK_OK 성공
 */
IRIS_SDK_EXPORT IrisSdkError iris_sdk_stabilize(
    int64_t handle,
    const IrisResult* raw,
    double timestamp_sec,
    IrisStabilizedResult* out);

/**
 * @brief Stabilizer 활성화/비활성화
 * @param handle Stabilizer 핸들
 * @param enabled 1=활성, 0=비활성(패스스루)
 */
IRIS_SDK_EXPORT void iris_sdk_stabilizer_set_enabled(int64_t handle, int enabled);

/**
 * @brief Stabilizer 상태 초기화
 * @param handle Stabilizer 핸들
 */
IRIS_SDK_EXPORT void iris_sdk_stabilizer_reset(int64_t handle);

// ============================================================================
// 렌더 결과 적용 API
// ============================================================================
// ④ W4-D: 검출 인프라 제거로 비동기 프레임 제출 API
//   (iris_sdk_submit_frame / _with_rotation / iris_sdk_get_latest_result)가
//   제거되었습니다. 검출 결과는 주입 경로(iris_set_landmarks)로 전달되고,
//   렌더는 아래 iris_sdk_render_with_result로 수행합니다.

/**
 * @brief 기존(주입된) 결과로 렌더링
 *
 * 외부 추적 글루가 산출한 검출 결과를 사용하여 렌더링만 수행합니다.
 * 프레임 데이터는 in-place로 수정됩니다.
 *
 * @param frame_data 프레임 데이터 (in-place 수정됨)
 * @param width 프레임 너비
 * @param height 프레임 높이
 * @param format 픽셀 포맷
 * @param iris_result 홍채 검출 결과
 * @param config 렌더링 설정
 * @return IRIS_SDK_OK 성공
 *
 * @deprecated cpu-render/CPU 픽셀 경로는 2.0에서 제거됩니다(ADR-0001 §8.2).
 *   GPU 텍스처 경로(sdk_api_v2)로 이행하세요. 1.x 동안 동작은 유지됩니다.
 */
IRIS_SDK_EXPORT IRIS_SDK_DEPRECATED IrisSdkError iris_sdk_render_with_result(
    uint8_t* frame_data,
    int width,
    int height,
    IrisFrameFormat format,
    const IrisResult* iris_result,
    const IrisLensConfig* config);

// ============================================================================
// 랜드마크 주입 경계 (ADR-0001 §6 — 추적 외부화 진입점)
// ============================================================================
//
// 추적(코어 밖)이 478점 랜드마크를 코어로 주입한다. 프레임 픽셀은 경계를 넘지 않는다.
// ③-1: 신규 부가 채널로 가동 → ④ W4-B2: **정식 랜드마크 공급 경계로 승격**. generation
// 검증 + §6.1 입력 유효성 가드(num_points 478/NULL/치수/NaN·Inf) 완비, detect 경로와 동일
// 저장소(g_landmark_store, feed_landmark_store) 공유로 writer 직렬화. 완전 단일화(detect
// 경로 제거로 본 함수가 유일 공급자가 됨)는 W4-D(추적 코어 제거)에서.
//
// 좌표 계약(ADR §7): 주입 478점은 회전 보정 완료(upright) 프레임 기준 정규화 좌표
//   [0,1] (x,y) + z. 항상 비미러(센서 원본). 미러는 렌더 단일 책임. 홍채 z 기하 사용 금지.

/**
 * @brief 478점 랜드마크 + upright 프레임 치수 + 타임스탬프 주입.
 *
 * 호출 스레드에서 코어 내부 버퍼로 deep-copy한다(호출자 버퍼 수명은 반환 시 종료).
 * 478점·frame dims·timestamp가 한 세대(generation)에 원자 결속된다(seqlock 더블버퍼).
 *
 * 스레딩 계약(ADR §6.1): 본 함수와 내부 detect 경로(iris_sdk_detect*)의 478점 자동 공급은
 * 동일한 주입 저장소를 공유한다. 두 writer가 서로 다른 스레드에서 동시 진입해도 저장소가
 * writer 측 직렬화를 책임지므로(seqlock generation 규율 보존), torn snapshot이나 generation
 * 홀수 잔류가 발생하지 않는다. 즉 전환기에 detect 가동 중 본 함수를 동시 호출해도 안전하다.
 * (단 동일 세대로 묶이는 것은 한 write 호출의 478점 전부이며, 두 writer 간 어느 쪽 주입이
 *  최신으로 공개되는지는 호출 시점에 따른다 — last-writer-wins.)
 *
 * 입력 유효성(ADR §6.1) — 거부 시 직전 유효 주입(스테일) 유지, generation 불변:
 *   - num_points != 478 → IRIS_SDK_INVALID_PARAM
 *   - pts == NULL 또는 out_generation == NULL → IRIS_SDK_NULL_POINTER
 *   - NaN/Inf 포함, frame_width/height <= 0 → IRIS_SDK_INVALID_PARAM
 *
 * @param pts          num_points×3 (x,y,z) 정규화 좌표 (배열 길이 ≥ num_points×3 필수).
 * @param num_points   점 수 — 478 고정 계약 (그 외 거부).
 * @param frame_width  upright 프레임 너비 (px) — 파생 어댑터 픽셀 환산 기준(§7.0).
 *                     렌더 타깃 치수와 별개(§6.1 — 종횡비 왜곡 봉쇄).
 * @param frame_height upright 프레임 높이 (px).
 * @param timestamp_us 단조 증가 타임스탬프 (µs) — One-Euro dt 산출용.
 * @param out_generation 성공 시 갱신된 주입 세대 번호 출력 (NULL 불가).
 * @return IRIS_SDK_OK 성공, 그 외 에러 코드.
 */
IRIS_SDK_EXPORT IrisSdkError iris_set_landmarks(
    const float* pts,
    int32_t num_points,
    int32_t frame_width,
    int32_t frame_height,
    int64_t timestamp_us,
    uint32_t* out_generation);

/**
 * @brief 현재 주입 세대 번호 조회.
 *
 * 0 = 미주입(주입 이력 없음). 짝수 = 완결 세대. iris_set_landmarks 성공마다 증가한다.
 * (호출자가 슬롯 일관성을 검증하거나 새 주입 도착을 감지하는 용도 — ADR §6.1 노출 필수.)
 *
 * @return 현재 세대 번호.
 */
IRIS_SDK_EXPORT uint32_t iris_get_landmark_generation(void);

/**
 * @brief 가장 최근 완결 세대의 주입 랜드마크에서 파생된 IrisResult를 조회한다.
 *
 * iris_set_landmarks로 주입된 478점에서 코어 어댑터가 유도한 파생 결과
 * (홍채 중심·반경, EAR→visibility, face_rect 등 ADR §6.2)를 out에 채운다.
 * 내부 LandmarkInjectionStore::readDerived()에 위임한다(reader 무락 seqlock 재시도 —
 * ADR §6.1). 주입 채널 전용이며 iris_sdk_get_latest_result(내부 detect 경로)와 별개다.
 *
 * 미주입(generation==0) 시 명시 에러(IRIS_SDK_NO_FACE)를 반환한다 — 검출 실패=주입 부재
 * 일원화(ADR §6.2). out은 미변경으로 남는다(silent OK 금지).
 *
 * @param out 파생 결과 출력 (NULL 불가).
 * @return IRIS_SDK_OK 결과 있음, IRIS_SDK_NO_FACE 미주입(주입 이력 없음),
 *         IRIS_SDK_NULL_POINTER out이 NULL.
 */
IRIS_SDK_EXPORT IrisSdkError iris_get_injected_result(IrisResult* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* IRIS_SDK_API_H */
