# P1-W4-05: C API 래퍼 구현

**태스크 ID**: P1-W4-05
**상태**: ⏳ 대기
**시작일**: -
**완료일**: -

---

## 1. 계획

### 목표
다양한 언어 바인딩 (JNI, Obj-C++, dart:ffi, WASM)을 위한 extern "C" API를 구현한다.

### 산출물
| 파일 | 설명 |
|------|------|
| `cpp/include/iris_sdk/sdk_api.h` | C API 함수 선언 |
| `cpp/src/sdk_api.cpp` | C API 구현 |

### 검증 기준
- [ ] extern "C" 심볼 export 확인
- [ ] 모든 함수 ABI 안정성 검증
- [ ] 에러 코드 반환 표준화
- [ ] 메모리 관리 규칙 명확화
- [ ] 문서화 (Doxygen)

### 선행 조건
- P1-W4-04: SDKManager 싱글톤 ✅

---

## 2. 분석

### 2.1 C API 설계 원칙

| 원칙 | 설명 |
|------|------|
| ABI 안정성 | 버전 간 바이너리 호환성 유지 |
| POD 데이터 | 모든 구조체는 Plain Old Data |
| 에러 핸들링 | 반환값 또는 out 파라미터로 에러 전달 |
| 메모리 규칙 | 할당자가 해제 책임 (명확한 소유권) |
| Null 안전 | 모든 포인터 파라미터 검증 |

### 2.2 API 함수 목록

| 함수 | 설명 | 카테고리 |
|------|------|----------|
| `iris_sdk_init` | SDK 초기화 | 라이프사이클 |
| `iris_sdk_init_with_config` | 설정으로 초기화 | 라이프사이클 |
| `iris_sdk_destroy` | SDK 해제 | 라이프사이클 |
| `iris_sdk_is_ready` | 상태 확인 | 라이프사이클 |
| `iris_sdk_detect` | 홍채 검출 | 검출 |
| `iris_sdk_process` | 검출+렌더링 | 처리 |
| `iris_sdk_render_lens` | 렌즈 렌더링 | 렌더링 |
| `iris_sdk_load_texture` | 텍스처 로드 | 렌더링 |
| `iris_sdk_set_config` | 설정 변경 | 설정 |
| `iris_sdk_get_version` | 버전 조회 | 정보 |
| `iris_sdk_get_last_error` | 마지막 에러 | 에러 |
| `iris_sdk_free_result` | 검출 결과 메모리 해제 | 메모리 |

### 2.3 헤더 파일 설계

```cpp
/**
 * @file sdk_api.h
 * @brief IrisLensSDK C API
 *
 * 모든 플랫폼 바인딩을 위한 C 인터페이스
 *
 * @note 모든 함수는 스레드 안전하지 않음 (단일 스레드 사용 권장)
 * @note 메모리 관리: SDK가 할당한 메모리는 SDK가 해제
 */

#ifndef IRIS_SDK_API_H
#define IRIS_SDK_API_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * Types
 * ============================================================ */

/**
 * @brief 에러 코드
 */
typedef enum {
    IRIS_SDK_OK = 0,                       ///< 성공

    // 초기화 에러 (100~199)
    IRIS_SDK_ERR_NOT_INITIALIZED = 100,    ///< 초기화 안됨
    IRIS_SDK_ERR_ALREADY_INITIALIZED = 101,///< 이미 초기화됨
    IRIS_SDK_ERR_MODEL_LOAD_FAILED = 102,  ///< 모델 로드 실패
    IRIS_SDK_ERR_INVALID_PATH = 103,       ///< 잘못된 경로

    // 파라미터 에러 (200~299)
    IRIS_SDK_ERR_INVALID_PARAM = 200,      ///< 잘못된 파라미터
    IRIS_SDK_ERR_NULL_POINTER = 201,       ///< NULL 포인터
    IRIS_SDK_ERR_INVALID_FORMAT = 202,     ///< 지원하지 않는 포맷

    // 검출 에러 (300~399)
    IRIS_SDK_ERR_DETECTION_FAILED = 300,   ///< 검출 실패
    IRIS_SDK_ERR_NO_FACE = 301,            ///< 얼굴 없음

    // 렌더링 에러 (400~499)
    IRIS_SDK_ERR_RENDER_FAILED = 400,      ///< 렌더링 실패
    IRIS_SDK_ERR_NO_TEXTURE = 401,         ///< 텍스처 없음

    // 일반 에러
    IRIS_SDK_ERR_UNKNOWN = 999             ///< 알 수 없는 에러
} IrisSdkError;

/**
 * @brief 프레임 포맷
 */
typedef enum {
    IRIS_FORMAT_RGBA = 0,    ///< 32bit RGBA
    IRIS_FORMAT_BGRA = 1,    ///< 32bit BGRA
    IRIS_FORMAT_RGB = 2,     ///< 24bit RGB
    IRIS_FORMAT_BGR = 3,     ///< 24bit BGR
    IRIS_FORMAT_NV21 = 4,    ///< YUV NV21 (Android)
    IRIS_FORMAT_NV12 = 5,    ///< YUV NV12 (iOS)
    IRIS_FORMAT_GRAY = 6     ///< 8bit Grayscale
} IrisFrameFormat;

/**
 * @brief 블렌딩 모드
 */
typedef enum {
    IRIS_BLEND_NORMAL = 0,
    IRIS_BLEND_MULTIPLY = 1,
    IRIS_BLEND_SCREEN = 2,
    IRIS_BLEND_OVERLAY = 3
} IrisBlendMode;

/**
 * @brief 랜드마크 좌표
 */
typedef struct {
    float x;          ///< X 좌표 (정규화 0.0~1.0)
    float y;          ///< Y 좌표 (정규화 0.0~1.0)
    float z;          ///< Z 좌표 (깊이)
    float visibility; ///< 가시성 (0.0~1.0)
} IrisLandmark;

/**
 * @brief 바운딩 박스
 */
typedef struct {
    float x;      ///< 좌상단 X
    float y;      ///< 좌상단 Y
    float width;  ///< 너비
    float height; ///< 높이
} IrisRect;

/**
 * @brief 홍채 검출 결과
 */
typedef struct {
    bool detected;              ///< 검출 성공 여부
    float confidence;           ///< 신뢰도 (0.0~1.0)

    // 왼쪽 눈
    bool left_detected;
    IrisLandmark left_iris[5];  ///< 홍채 랜드마크 (center + 4 boundary)
    IrisRect left_eye_rect;
    float left_radius;

    // 오른쪽 눈
    bool right_detected;
    IrisLandmark right_iris[5];
    IrisRect right_eye_rect;
    float right_radius;

    // 얼굴 정보
    IrisRect face_rect;
    float face_rotation[3];     ///< pitch, yaw, roll

    // 메타데이터
    int64_t timestamp_ms;
    int frame_width;
    int frame_height;
} IrisResult;

/**
 * @brief 렌즈 렌더링 설정
 */
typedef struct {
    float opacity;          ///< 투명도 (0.0~1.0, 기본 0.7)
    float scale;            ///< 크기 배율 (기본 1.0)
    float offset_x;         ///< X 오프셋 (정규화)
    float offset_y;         ///< Y 오프셋 (정규화)
    IrisBlendMode blend_mode; ///< 블렌딩 모드
    float edge_feather;     ///< 경계 페더링 (0.0~1.0)
    bool apply_left;        ///< 왼쪽 눈 적용
    bool apply_right;       ///< 오른쪽 눈 적용
} IrisLensConfig;

/**
 * @brief SDK 설정
 */
typedef struct {
    const char* model_path;     ///< 모델 파일 경로
    float min_confidence;       ///< 최소 신뢰도
    int max_faces;              ///< 최대 얼굴 수
    bool enable_gpu;            ///< GPU 사용 여부
    int num_threads;            ///< 스레드 수 (0=자동)
} IrisSdkConfig;

/* ============================================================
 * Lifecycle Functions
 * ============================================================ */

/**
 * @brief SDK 초기화 (기본 설정)
 * @param model_path 모델 파일 경로
 * @return 에러 코드
 */
IRIS_SDK_EXPORT IrisSdkError iris_sdk_init(const char* model_path);

/**
 * @brief SDK 초기화 (상세 설정)
 * @param config 설정 구조체
 * @return 에러 코드
 */
IRIS_SDK_EXPORT IrisSdkError iris_sdk_init_with_config(const IrisSdkConfig* config);

/**
 * @brief SDK 해제
 */
IRIS_SDK_EXPORT void iris_sdk_destroy(void);

/**
 * @brief SDK 준비 상태 확인
 * @return 준비되었으면 true
 */
IRIS_SDK_EXPORT bool iris_sdk_is_ready(void);

/* ============================================================
 * Detection Functions
 * ============================================================ */

/**
 * @brief 홍채 검출
 * @param frame_data 프레임 데이터 (read-only)
 * @param width 프레임 너비
 * @param height 프레임 높이
 * @param format 픽셀 포맷
 * @param[out] result 검출 결과
 * @return 에러 코드
 */
IRIS_SDK_EXPORT IrisSdkError iris_sdk_detect(
    const uint8_t* frame_data,
    int width,
    int height,
    IrisFrameFormat format,
    IrisResult* result
);

/* ============================================================
 * Processing Functions (Detection + Rendering)
 * ============================================================ */

/**
 * @brief 프레임 처리 (검출 + 렌더링)
 * @param frame_data 프레임 데이터 (in-place 수정됨)
 * @param width 프레임 너비
 * @param height 프레임 높이
 * @param format 픽셀 포맷
 * @param config 렌더링 설정 (NULL이면 검출만)
 * @param[out] result 검출 결과
 * @return 에러 코드
 */
IRIS_SDK_EXPORT IrisSdkError iris_sdk_process(
    uint8_t* frame_data,
    int width,
    int height,
    IrisFrameFormat format,
    const IrisLensConfig* config,
    IrisResult* result
);

/* ============================================================
 * Rendering Functions
 * ============================================================ */

/**
 * @brief 렌즈 텍스처 로드 (파일)
 * @param texture_path 텍스처 이미지 경로
 * @return 에러 코드
 */
IRIS_SDK_EXPORT IrisSdkError iris_sdk_load_texture(const char* texture_path);

/**
 * @brief 렌즈 텍스처 로드 (메모리)
 * @param data RGBA 데이터
 * @param width 너비
 * @param height 높이
 * @return 에러 코드
 */
IRIS_SDK_EXPORT IrisSdkError iris_sdk_load_texture_from_memory(
    const uint8_t* data,
    int width,
    int height
);

/**
 * @brief 렌즈 렌더링만 수행
 * @param frame_data 프레임 데이터 (in-place 수정됨)
 * @param width 프레임 너비
 * @param height 프레임 높이
 * @param format 픽셀 포맷
 * @param iris_result 검출 결과
 * @param config 렌더링 설정
 * @return 에러 코드
 */
IRIS_SDK_EXPORT IrisSdkError iris_sdk_render_lens(
    uint8_t* frame_data,
    int width,
    int height,
    IrisFrameFormat format,
    const IrisResult* iris_result,
    const IrisLensConfig* config
);

/* ============================================================
 * Configuration Functions
 * ============================================================ */

/**
 * @brief 설정 변경
 * @param key 설정 키
 * @param value 설정 값
 * @return 에러 코드
 */
IRIS_SDK_EXPORT IrisSdkError iris_sdk_set_config(
    const char* key,
    const char* value
);

/**
 * @brief 기본 렌즈 설정 생성
 * @param[out] config 설정 구조체
 */
IRIS_SDK_EXPORT void iris_sdk_default_lens_config(IrisLensConfig* config);

/* ============================================================
 * Information Functions
 * ============================================================ */

/**
 * @brief SDK 버전 문자열 반환
 * @return 버전 문자열 (예: "0.1.0")
 */
IRIS_SDK_EXPORT const char* iris_sdk_get_version(void);

/**
 * @brief 빌드 정보 반환
 * @return 빌드 정보 문자열
 */
IRIS_SDK_EXPORT const char* iris_sdk_get_build_info(void);

/**
 * @brief 마지막 에러 메시지 반환
 * @return 에러 메시지 (에러 없으면 NULL)
 */
IRIS_SDK_EXPORT const char* iris_sdk_get_last_error(void);

/**
 * @brief 에러 코드를 문자열로 변환
 * @param error 에러 코드
 * @return 에러 설명 문자열
 */
IRIS_SDK_EXPORT const char* iris_sdk_error_to_string(IrisSdkError error);

/* ============================================================
 * Memory Management Functions
 * ============================================================ */

/**
 * @brief 검출 결과 메모리 해제
 *
 * iris_sdk_detect() 또는 iris_sdk_process()로 반환된 결과의
 * 내부 동적 할당 메모리를 해제한다.
 *
 * @param result 해제할 검출 결과 포인터
 * @note 결과 구조체 자체는 호출자가 관리함
 */
IRIS_SDK_EXPORT void iris_sdk_free_result(IrisResult* result);

#ifdef __cplusplus
}
#endif

#endif /* IRIS_SDK_API_H */
```

### 2.4 구현 예시

```cpp
// sdk_api.cpp

#include "iris_sdk/sdk_api.h"
#include "iris_sdk/sdk_manager.h"
#include "iris_sdk/frame_processor.h"
#include <string>
#include <mutex>

namespace {
    std::unique_ptr<iris_sdk::FrameProcessor> g_processor;
    std::string g_last_error;
    std::mutex g_mutex;

    void setLastError(const std::string& error) {
        g_last_error = error;
    }

    IrisSdkError convertError(iris_sdk::ErrorCode code) {
        switch (code) {
            case iris_sdk::ErrorCode::Success:
                return IRIS_SDK_OK;
            case iris_sdk::ErrorCode::NotInitialized:
                return IRIS_SDK_ERR_NOT_INITIALIZED;
            // ... 기타 변환
            default:
                return IRIS_SDK_ERR_UNKNOWN;
        }
    }
}

extern "C" {

IRIS_SDK_EXPORT IrisSdkError iris_sdk_init(const char* model_path) {
    if (model_path == nullptr) {
        setLastError("model_path is null");
        return IRIS_SDK_ERR_NULL_POINTER;
    }

    std::lock_guard<std::mutex> lock(g_mutex);

    auto& manager = iris_sdk::SDKManager::getInstance();
    if (!manager.initialize(model_path)) {
        setLastError("Failed to initialize SDK");
        return IRIS_SDK_ERR_MODEL_LOAD_FAILED;
    }

    g_processor = manager.createFrameProcessor();
    if (!g_processor) {
        setLastError("Failed to create processor");
        return IRIS_SDK_ERR_UNKNOWN;
    }

    return IRIS_SDK_OK;
}

IRIS_SDK_EXPORT void iris_sdk_destroy(void) {
    std::lock_guard<std::mutex> lock(g_mutex);

    g_processor.reset();
    iris_sdk::SDKManager::getInstance().shutdown();
}

IRIS_SDK_EXPORT IrisSdkError iris_sdk_detect(
    const uint8_t* frame_data,
    int width,
    int height,
    IrisFrameFormat format,
    IrisResult* result) {

    if (!frame_data || !result) {
        return IRIS_SDK_ERR_NULL_POINTER;
    }

    if (!g_processor) {
        return IRIS_SDK_ERR_NOT_INITIALIZED;
    }

    // C++ 결과를 C 구조체로 변환
    auto cpp_result = g_processor->detectOnly(
        frame_data, width, height,
        static_cast<iris_sdk::FrameFormat>(format)
    );

    // 결과 복사
    result->detected = cpp_result.detected;
    result->confidence = cpp_result.confidence;
    // ... 기타 필드 복사

    return IRIS_SDK_OK;
}

IRIS_SDK_EXPORT const char* iris_sdk_get_version(void) {
    return IRIS_SDK_VERSION_STRING;
}

IRIS_SDK_EXPORT const char* iris_sdk_get_last_error(void) {
    return g_last_error.empty() ? nullptr : g_last_error.c_str();
}

IRIS_SDK_EXPORT void iris_sdk_default_lens_config(IrisLensConfig* config) {
    if (config) {
        config->opacity = 0.7f;
        config->scale = 1.0f;
        config->offset_x = 0.0f;
        config->offset_y = 0.0f;
        config->blend_mode = IRIS_BLEND_NORMAL;
        config->edge_feather = 0.1f;
        config->apply_left = true;
        config->apply_right = true;
    }
}

IRIS_SDK_EXPORT void iris_sdk_free_result(IrisResult* result) {
    if (result) {
        // 현재 IrisResult는 고정 크기 배열만 포함하므로
        // 동적 할당 메모리가 없지만, 향후 확장성을 위해 API 제공
        // 구조체 내용을 초기화
        result->detected = false;
        result->left_detected = false;
        result->right_detected = false;
    }
}

} // extern "C"
```

### 2.5 ABI 안정성 가이드라인

| 규칙 | 설명 |
|------|------|
| 구조체 확장 | 새 필드는 끝에만 추가 |
| enum 값 | 기존 값 변경 금지, 새 값은 끝에 추가 |
| 함수 시그니처 | 변경 금지, 새 버전 함수 추가 (v2) |
| 심볼 이름 | 변경 금지 |

---

## 3. 실행 내역

### 3.1 헤더 파일 작성

```bash
# 예정: cpp/include/iris_sdk/sdk_api.h
```

### 3.2 구현 파일 작성

```bash
# 예정: cpp/src/sdk_api.cpp
```

### 3.3 심볼 export 검증

```bash
# 예정: nm -gU libiris_sdk.dylib | grep iris_sdk
```

---

## 4. 검증 결과

### 검증 항목

| 항목 | 결과 | 비고 |
|------|------|------|
| extern "C" 심볼 | ⏳ 대기 | nm 확인 |
| 함수 시그니처 | ⏳ 대기 | |
| 구조체 레이아웃 | ⏳ 대기 | sizeof 검증 |
| 에러 처리 | ⏳ 대기 | |
| 메모리 관리 | ⏳ 대기 | |

### 심볼 export 확인

```bash
# 예정 출력
_iris_sdk_init
_iris_sdk_destroy
_iris_sdk_detect
_iris_sdk_process
_iris_sdk_render_lens
_iris_sdk_get_version
...
```

---

## 5. 이슈 및 학습

### 이슈

| ID | 내용 | 상태 | 해결방안 |
|----|------|------|----------|
| - | - | - | - |

### 결정 사항

| 결정 | 이유 |
|------|------|
| typedef enum 사용 | C 호환성 (C++ enum class 대신) |
| out 파라미터 패턴 | FFI 호환성, 메모리 관리 명확화 |
| 에러 문자열 전역 | 간단한 에러 보고, 멀티스레드 주의 필요 |

### 학습 내용

(실행 후 기록)

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-01-07 | 태스크 문서 생성, C API 설계 완료 |
| 2026-01-07 | 아키텍처 리뷰: iris_sdk_free_result() 메모리 해제 API 추가 |
