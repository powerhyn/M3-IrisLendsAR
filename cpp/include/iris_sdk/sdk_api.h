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
// 블렌드 모드
// ============================================================================

/**
 * @brief 렌즈 블렌딩 모드
 *
 * 렌즈 렌더링 시 프레임과 텍스처를 합성하는 방식을 지정합니다.
 */
typedef enum IrisBlendMode {
    IRIS_BLEND_NORMAL = 0,      /**< 일반 알파 블렌딩 */
    IRIS_BLEND_MULTIPLY = 1,    /**< 곱하기 블렌딩 */
    IRIS_BLEND_SCREEN = 2,      /**< 스크린 블렌딩 */
    IRIS_BLEND_OVERLAY = 3      /**< 오버레이 블렌딩 */
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
    IrisBlendMode blend_mode;   /**< 블렌드 모드 (기본값 NORMAL) */
    float edge_feather;         /**< 가장자리 페더링 (0.0~1.0, 기본값 0.1) */
    bool apply_left;            /**< 왼쪽 눈 적용 여부 (기본값 true) */
    bool apply_right;           /**< 오른쪽 눈 적용 여부 (기본값 true) */
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
// 검출 함수
// ============================================================================

/**
 * @brief 홍채 검출
 *
 * 프레임에서 홍채를 검출합니다. 프레임 데이터는 수정되지 않습니다.
 *
 * @param frame_data 프레임 데이터 (읽기 전용)
 * @param width 프레임 너비
 * @param height 프레임 높이
 * @param format 픽셀 포맷
 * @param result 검출 결과 출력 (NULL 불가)
 * @return IRIS_SDK_OK 성공, 그 외 에러 코드
 *
 * @note 검출 성공 시에도 result->detected가 false일 수 있습니다 (얼굴 없음).
 * @note 회전이 필요한 경우 iris_sdk_detect_with_rotation() 사용
 */
IRIS_SDK_EXPORT IrisSdkError iris_sdk_detect(
    const uint8_t* frame_data,
    int width,
    int height,
    IrisFrameFormat format,
    IrisResult* result);

/**
 * @brief 홍채 검출 (회전 지원)
 *
 * 프레임에서 홍채를 검출합니다. 이미지 회전을 지원합니다.
 * Android/iOS 카메라는 일반적으로 회전된 이미지를 출력하므로
 * 이 함수를 사용하여 회전을 보정합니다.
 *
 * @param frame_data 프레임 데이터 (읽기 전용)
 * @param width 프레임 너비
 * @param height 프레임 높이
 * @param format 픽셀 포맷
 * @param rotation_degrees 이미지 회전 각도 (0, 90, 180, 270)
 *                         카메라 센서 방향에 따른 회전 보정값
 * @param result 검출 결과 출력 (NULL 불가)
 * @return IRIS_SDK_OK 성공, 그 외 에러 코드
 *
 * @note rotation_degrees는 이미지를 정방향으로 만들기 위해 필요한 회전 각도입니다.
 *       예: Android CameraX의 ImageProxy.imageInfo.rotationDegrees 값을 직접 전달
 */
IRIS_SDK_EXPORT IrisSdkError iris_sdk_detect_with_rotation(
    const uint8_t* frame_data,
    int width,
    int height,
    IrisFrameFormat format,
    int rotation_degrees,
    IrisResult* result);

// ============================================================================
// 처리 함수
// ============================================================================

/**
 * @brief 프레임 처리 (검출 + 렌더링)
 *
 * 홍채 검출과 렌즈 렌더링을 한 번에 수행합니다.
 * 프레임 데이터는 in-place로 수정됩니다.
 *
 * @param frame_data 프레임 데이터 (in-place 수정됨)
 * @param width 프레임 너비
 * @param height 프레임 높이
 * @param format 픽셀 포맷
 * @param config 렌더링 설정 (NULL이면 검출만 수행)
 * @param result 검출 결과 출력 (NULL 가능, NULL이면 결과 무시)
 * @return IRIS_SDK_OK 성공, 그 외 에러 코드
 *
 * @note 렌더링을 수행하려면 먼저 iris_sdk_load_texture()로 텍스처를 로드해야 합니다.
 */
IRIS_SDK_EXPORT IrisSdkError iris_sdk_process(
    uint8_t* frame_data,
    int width,
    int height,
    IrisFrameFormat format,
    const IrisLensConfig* config,
    IrisResult* result);

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
 */
IRIS_SDK_EXPORT IrisSdkError iris_sdk_load_texture(const char* path);

/**
 * @brief 렌즈 텍스처 로드 (메모리)
 *
 * 메모리에서 RGBA 텍스처를 로드합니다.
 *
 * @param data RGBA 픽셀 데이터 (4바이트/픽셀)
 * @param width 텍스처 너비
 * @param height 텍스처 높이
 * @return IRIS_SDK_OK 성공, 그 외 에러 코드
 */
IRIS_SDK_EXPORT IrisSdkError iris_sdk_load_texture_from_memory(
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
 */
IRIS_SDK_EXPORT IrisSdkError iris_sdk_render_lens(
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

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* IRIS_SDK_API_H */
