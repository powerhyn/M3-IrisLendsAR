# IrisLensSDK 아키텍처 설계서

**버전**: 2.0
**최종 수정**: 2025-01-07
**상태**: 승인됨

---

## 목차

1. [시스템 개요](#1-시스템-개요)
2. [레이어 아키텍처](#2-레이어-아키텍처)
3. [코어 엔진 설계](#3-코어-엔진-설계)
4. [데이터 구조](#4-데이터-구조)
5. [C API 설계](#5-c-api-설계)
6. [데이터 흐름](#6-데이터-흐름)
7. [플랫폼별 바인딩](#7-플랫폼별-바인딩)
8. [프로젝트 구조](#8-프로젝트-구조)
9. [확장성 설계](#9-확장성-설계)
10. [성능 고려사항](#10-성능-고려사항)

---

## 1. 시스템 개요

### 1.1 목적

IrisLensSDK는 실시간 카메라 영상에서 홍채를 검출하고 가상 렌즈를 오버레이하는 AR 피팅 SDK입니다.

### 1.2 설계 원칙

| 원칙 | 설명 |
|------|------|
| **성능 우선** | 30fps+ 실시간 처리를 위한 최적화 |
| **크로스플랫폼** | 단일 코어로 모든 플랫폼 지원 |
| **모듈화** | 컴포넌트 간 느슨한 결합 |
| **확장성** | 모델 교체 가능한 인터페이스 설계 |
| **보안** | 온디바이스 처리, 데이터 외부 전송 금지 |

### 1.3 기술 스택

```
┌─────────────────────────────────────────────────────────┐
│                    Application                           │
│  Android (Kotlin) │ iOS (Swift) │ Flutter │ Web (JS)    │
├─────────────────────────────────────────────────────────┤
│                    Binding Layer                         │
│      JNI      │   Obj-C++   │  dart:ffi  │   WASM      │
├─────────────────────────────────────────────────────────┤
│                      C API                               │
│               extern "C" functions                       │
├─────────────────────────────────────────────────────────┤
│                   C++ Core Engine                        │
│                     (C++17)                              │
├─────────────────────────────────────────────────────────┤
│                  Third Party Libraries                   │
│   MediaPipe    │    OpenCV 4.x    │   TensorFlow Lite   │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 레이어 아키텍처

### 2.1 전체 시스템 다이어그램

```
┌─────────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                           │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐        │
│  │  Android  │ │    iOS    │ │  Flutter  │ │    Web    │        │
│  │    App    │ │    App    │ │    App    │ │    App    │        │
│  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘        │
└────────┼─────────────┼─────────────┼─────────────┼──────────────┘
         │             │             │             │
         ▼             ▼             ▼             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       BINDING LAYER                              │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐        │
│  │    JNI    │ │  Obj-C++  │ │ dart:ffi  │ │   WASM    │        │
│  │  Bridge   │ │  Bridge   │ │  Bridge   │ │  Bridge   │        │
│  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘        │
│        │             │             │             │               │
│        └─────────────┴──────┬──────┴─────────────┘               │
└─────────────────────────────┼───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        C API LAYER                               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  iris_sdk_init()  │  iris_sdk_detect()  │  iris_sdk_*() │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     C++ CORE ENGINE                              │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    SDKManager (Singleton)                   │ │
│  │  - Lifecycle Management                                     │ │
│  │  - Resource Management                                      │ │
│  │  - Configuration                                            │ │
│  └──────────────┬─────────────────┬─────────────────┬─────────┘ │
│                 │                 │                 │           │
│                 ▼                 ▼                 ▼           │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐ │
│  │   IrisDetector   │ │   LensRenderer   │ │  FrameProcessor  │ │
│  │   (Interface)    │ │                  │ │                  │ │
│  ├──────────────────┤ ├──────────────────┤ ├──────────────────┤ │
│  │ MediaPipeDetector│ │ - loadTexture()  │ │ - convertFormat()│ │
│  │ EyeOnlyDetector  │ │ - warpToIris()   │ │ - resize()       │ │
│  │ HybridDetector   │ │ - blend()        │ │ - normalize()    │ │
│  └──────────────────┘ └──────────────────┘ └──────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                   THIRD PARTY LIBRARIES                     │ │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────┐    │ │
│  │  │  MediaPipe │  │  OpenCV    │  │  TensorFlow Lite   │    │ │
│  │  │ Face Mesh  │  │  4.x       │  │  (Custom Models)   │    │ │
│  │  │ Iris Track │  │            │  │                    │    │ │
│  │  └────────────┘  └────────────┘  └────────────────────┘    │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 레이어 책임

| 레이어 | 책임 | 구현 기술 |
|--------|------|----------|
| **Application** | UI, 카메라, 사용자 상호작용 | 플랫폼 네이티브 |
| **Binding** | 플랫폼 ↔ C API 변환 | JNI, Obj-C++, FFI, WASM |
| **C API** | 통일된 외부 인터페이스 | extern "C" |
| **Core Engine** | 비즈니스 로직, 검출, 렌더링 | C++17 |
| **Third Party** | ML 추론, 이미지 처리 | MediaPipe, OpenCV |

---

## 3. 코어 엔진 설계

### 3.1 SDKManager (Singleton + Facade)

```cpp
namespace iris_sdk {

class SDKManager {
public:
    // Singleton 접근
    static SDKManager& getInstance();

    // 라이프사이클
    int init(const SDKConfig& config);
    void destroy();
    bool isInitialized() const;

    // 컴포넌트 접근
    IrisDetector* getDetector();
    LensRenderer* getRenderer();
    FrameProcessor* getProcessor();

    // 설정
    void setConfig(const std::string& key, const std::string& value);
    std::string getConfig(const std::string& key) const;

private:
    SDKManager();
    ~SDKManager();
    SDKManager(const SDKManager&) = delete;
    SDKManager& operator=(const SDKManager&) = delete;

    std::unique_ptr<IrisDetector> detector_;
    std::unique_ptr<LensRenderer> renderer_;
    std::unique_ptr<FrameProcessor> processor_;
    SDKConfig config_;
    bool initialized_ = false;
    std::mutex mutex_;
};

} // namespace iris_sdk
```

**설계 결정**:
- **Singleton**: 전역 리소스 관리, 중복 초기화 방지
- **Facade**: 복잡한 서브시스템을 단순 인터페이스로 제공
- **Thread Safety**: mutex로 동시 접근 보호

### 3.2 IrisDetector (Strategy Pattern)

```cpp
namespace iris_sdk {

// 검출기 타입
enum class DetectorType {
    MEDIAPIPE,      // Phase 1: MediaPipe 기반
    EYE_ONLY,       // Phase 2: 눈 영역 전용
    HYBRID          // Phase 2: MediaPipe + Eye-Only 폴백
};

// 추상 인터페이스
class IrisDetector {
public:
    virtual ~IrisDetector() = default;

    // 라이프사이클
    virtual int init(const std::string& model_path) = 0;
    virtual void release() = 0;

    // 검출
    virtual int detect(const Frame& frame, IrisResult& result) = 0;

    // 메타데이터
    virtual std::string getName() const = 0;
    virtual DetectorType getType() const = 0;

    // Factory Method
    static std::unique_ptr<IrisDetector> create(DetectorType type);
};

// Phase 1 구현체
class MediaPipeDetector : public IrisDetector {
public:
    MediaPipeDetector();
    ~MediaPipeDetector() override;

    int init(const std::string& model_path) override;
    void release() override;
    int detect(const Frame& frame, IrisResult& result) override;
    std::string getName() const override { return "MediaPipe"; }
    DetectorType getType() const override { return DetectorType::MEDIAPIPE; }

private:
    class Impl;  // pImpl 패턴
    std::unique_ptr<Impl> impl_;
};

// Phase 2 구현체 (스텁)
class EyeOnlyDetector : public IrisDetector { /* ... */ };
class HybridDetector : public IrisDetector { /* ... */ };

} // namespace iris_sdk
```

**설계 결정**:
- **Strategy Pattern**: 런타임에 검출기 교체 가능
- **Factory Method**: 타입에 따른 구현체 생성
- **pImpl Pattern**: ABI 안정성, 컴파일 의존성 감소

### 3.3 LensRenderer

```cpp
namespace iris_sdk {

// 블렌딩 모드
enum class BlendMode {
    NORMAL,         // 일반 오버레이
    MULTIPLY,       // 곱하기 블렌드
    SOFT_LIGHT,     // 소프트 라이트
    OVERLAY         // 오버레이
};

class LensRenderer {
public:
    LensRenderer();
    ~LensRenderer();

    // 텍스처 관리
    int loadTexture(const std::string& path);
    int loadTextureFromMemory(const uint8_t* data, int size);
    void unloadTexture();

    // 렌더링
    int render(Frame& frame,
               const IrisResult& iris,
               const LensConfig& config);

    // 설정
    void setBlendMode(BlendMode mode);
    void setOpacity(float opacity);  // 0.0 ~ 1.0
    void setScale(float scale);      // 1.0 = 홍채 크기

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// 렌즈 설정
struct LensConfig {
    float opacity = 0.8f;
    float scale = 1.0f;
    BlendMode blend_mode = BlendMode::NORMAL;
    bool render_left = true;
    bool render_right = true;
};

} // namespace iris_sdk
```

### 3.4 FrameProcessor

```cpp
namespace iris_sdk {

// 픽셀 포맷
enum class PixelFormat {
    RGB,            // 3 channels
    RGBA,           // 4 channels
    BGR,            // OpenCV 기본
    BGRA,           // iOS CVPixelBuffer
    NV21,           // Android Camera1
    YUV_420_888     // Android Camera2
};

class FrameProcessor {
public:
    FrameProcessor();
    ~FrameProcessor();

    // 포맷 변환
    int convert(const Frame& src, Frame& dst, PixelFormat target_format);

    // 전처리
    int resize(const Frame& src, Frame& dst, int target_width, int target_height);
    int normalize(Frame& frame);  // 0-255 → 0.0-1.0

    // 유틸리티
    static int getChannelCount(PixelFormat format);
    static bool isYUVFormat(PixelFormat format);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace iris_sdk
```

---

## 4. 데이터 구조

### 4.1 프레임 데이터

```cpp
namespace iris_sdk {

// 프레임 구조체
struct Frame {
    uint8_t* data;          // 픽셀 데이터 포인터
    int width;              // 프레임 너비
    int height;             // 프레임 높이
    PixelFormat format;     // 픽셀 포맷
    int stride;             // 행당 바이트 수 (0 = width * channels)
    bool owns_data;         // 메모리 소유권

    // 편의 메서드
    int channels() const;
    int size() const;
    Frame clone() const;
    void release();
};

} // namespace iris_sdk
```

### 4.2 홍채 검출 결과

```cpp
namespace iris_sdk {

// 단일 랜드마크
struct IrisLandmark {
    float x;    // 정규화 좌표 (0.0 ~ 1.0)
    float y;    // 정규화 좌표 (0.0 ~ 1.0)
    float z;    // 깊이 (선택적, 기본 0.0)

    // 픽셀 좌표 변환
    int pixelX(int frame_width) const { return static_cast<int>(x * frame_width); }
    int pixelY(int frame_height) const { return static_cast<int>(y * frame_height); }
};

// 단일 눈 결과
struct EyeResult {
    bool detected;              // 검출 성공 여부
    float confidence;           // 신뢰도 (0.0 ~ 1.0)

    // 홍채 랜드마크 (5개: 중심 + 4방향 경계)
    // [0]: center, [1]: left, [2]: top, [3]: right, [4]: bottom
    IrisLandmark iris[5];

    // 눈 영역 바운딩 박스 (정규화)
    float bbox_x;
    float bbox_y;
    float bbox_width;
    float bbox_height;

    // 계산된 홍채 정보
    float iris_center_x() const { return iris[0].x; }
    float iris_center_y() const { return iris[0].y; }
    float iris_radius() const;   // 4방향 경계의 평균 거리
};

// 전체 검출 결과
struct IrisResult {
    bool face_detected;         // 얼굴 검출 여부
    float face_confidence;      // 얼굴 검출 신뢰도

    EyeResult left_eye;         // 왼쪽 눈
    EyeResult right_eye;        // 오른쪽 눈

    int64_t timestamp_ms;       // 검출 타임스탬프
    float processing_time_ms;   // 처리 시간

    // 유효성 검사
    bool isValid() const {
        return face_detected &&
               (left_eye.detected || right_eye.detected);
    }
};

} // namespace iris_sdk
```

### 4.3 SDK 설정

```cpp
namespace iris_sdk {

struct SDKConfig {
    // 경로
    std::string model_path;             // 모델 파일 경로
    std::string lens_texture_path;      // 기본 렌즈 텍스처

    // 검출 설정
    DetectorType detector_type = DetectorType::MEDIAPIPE;
    int max_faces = 1;                  // 최대 검출 얼굴 수
    float min_detection_confidence = 0.5f;
    float min_tracking_confidence = 0.5f;

    // 성능 설정
    bool enable_gpu = false;            // GPU 가속
    int target_fps = 30;                // 목표 FPS
    int input_width = 640;              // 입력 리사이즈 너비 (0 = 원본)

    // 렌더링 설정
    float default_opacity = 0.8f;
    BlendMode default_blend_mode = BlendMode::NORMAL;
};

} // namespace iris_sdk
```

---

## 5. C API 설계

### 5.1 API 헤더

```cpp
// cpp/include/iris_sdk/sdk_api.h

#ifndef IRIS_SDK_API_H
#define IRIS_SDK_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

//=============================================================================
// 버전 정보
//=============================================================================
#define IRIS_SDK_VERSION_MAJOR 1
#define IRIS_SDK_VERSION_MINOR 0
#define IRIS_SDK_VERSION_PATCH 0

//=============================================================================
// 에러 코드
//=============================================================================
typedef enum {
    IRIS_SDK_OK                     = 0,
    IRIS_SDK_ERROR_NOT_INITIALIZED  = -1,
    IRIS_SDK_ERROR_ALREADY_INIT     = -2,
    IRIS_SDK_ERROR_INVALID_PARAM    = -3,
    IRIS_SDK_ERROR_MODEL_LOAD       = -4,
    IRIS_SDK_ERROR_DETECTION        = -5,
    IRIS_SDK_ERROR_RENDER           = -6,
    IRIS_SDK_ERROR_OUT_OF_MEMORY    = -7,
    IRIS_SDK_ERROR_UNSUPPORTED      = -8,
    IRIS_SDK_ERROR_INTERNAL         = -99
} IrisSDKError;

//=============================================================================
// 픽셀 포맷
//=============================================================================
typedef enum {
    IRIS_FORMAT_RGB         = 0,
    IRIS_FORMAT_RGBA        = 1,
    IRIS_FORMAT_BGR         = 2,
    IRIS_FORMAT_BGRA        = 3,
    IRIS_FORMAT_NV21        = 4,
    IRIS_FORMAT_YUV420_888  = 5
} IrisPixelFormat;

//=============================================================================
// 블렌드 모드
//=============================================================================
typedef enum {
    IRIS_BLEND_NORMAL       = 0,
    IRIS_BLEND_MULTIPLY     = 1,
    IRIS_BLEND_SOFT_LIGHT   = 2,
    IRIS_BLEND_OVERLAY      = 3
} IrisBlendMode;

//=============================================================================
// 데이터 구조체
//=============================================================================

// 단일 눈 검출 결과
typedef struct {
    int detected;               // 0 or 1
    float confidence;           // 0.0 ~ 1.0
    float iris_landmarks[10];   // 5 points × (x, y)
    float bbox[4];              // x, y, width, height
} IrisEyeResult;

// 전체 검출 결과
typedef struct {
    int face_detected;
    float face_confidence;
    IrisEyeResult left_eye;
    IrisEyeResult right_eye;
    int64_t timestamp_ms;
    float processing_time_ms;
} IrisDetectionResult;

// SDK 설정
typedef struct {
    const char* model_path;
    const char* detector_type;      // "mediapipe", "eye_only", "hybrid"
    int max_faces;
    float min_detection_confidence;
    float min_tracking_confidence;
    int enable_gpu;
    int input_width;
} IrisSDKConfig;

// 렌즈 설정
typedef struct {
    const char* texture_path;
    float opacity;
    float scale;
    int blend_mode;                 // IrisBlendMode
    int render_left;
    int render_right;
} IrisLensConfig;

//=============================================================================
// SDK 라이프사이클
//=============================================================================

// SDK 초기화
IrisSDKError iris_sdk_init(const IrisSDKConfig* config);

// SDK 해제
IrisSDKError iris_sdk_destroy(void);

// 초기화 상태 확인
int iris_sdk_is_initialized(void);

//=============================================================================
// 홍채 검출
//=============================================================================

// 홍채 검출 수행
IrisSDKError iris_sdk_detect(
    const uint8_t* frame_data,
    int width,
    int height,
    int format,                     // IrisPixelFormat
    IrisDetectionResult* result
);

//=============================================================================
// 렌즈 렌더링
//=============================================================================

// 렌즈 텍스처 로드
IrisSDKError iris_sdk_load_lens_texture(const char* path);

// 렌즈 렌더링 (in-place)
IrisSDKError iris_sdk_render_lens(
    uint8_t* frame_data,            // in-out
    int width,
    int height,
    int format,                     // IrisPixelFormat
    const IrisDetectionResult* iris,
    const IrisLensConfig* config
);

// 렌즈 텍스처 해제
void iris_sdk_unload_lens_texture(void);

//=============================================================================
// 유틸리티
//=============================================================================

// 버전 문자열 반환
const char* iris_sdk_get_version(void);

// 에러 문자열 반환
const char* iris_sdk_error_string(IrisSDKError error);

// 설정 변경
IrisSDKError iris_sdk_set_config(const char* key, const char* value);

// 설정 조회
const char* iris_sdk_get_config(const char* key);

#ifdef __cplusplus
}
#endif

#endif // IRIS_SDK_API_H
```

### 5.2 에러 처리 규칙

| 에러 코드 | 의미 | 대응 |
|-----------|------|------|
| `IRIS_SDK_OK` | 성공 | - |
| `IRIS_SDK_ERROR_NOT_INITIALIZED` | 초기화 안됨 | `iris_sdk_init()` 호출 |
| `IRIS_SDK_ERROR_INVALID_PARAM` | 잘못된 파라미터 | 파라미터 검증 |
| `IRIS_SDK_ERROR_MODEL_LOAD` | 모델 로드 실패 | 경로/파일 확인 |
| `IRIS_SDK_ERROR_DETECTION` | 검출 실패 | 프레임 품질 확인 |
| `IRIS_SDK_ERROR_OUT_OF_MEMORY` | 메모리 부족 | 리소스 해제 |

---

## 6. 데이터 흐름

### 6.1 초기화 시퀀스

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌───────────┐     ┌──────────┐
│   App   │     │ Binding │     │  C API  │     │SDKManager │     │ Detector │
└────┬────┘     └────┬────┘     └────┬────┘     └─────┬─────┘     └────┬─────┘
     │               │               │                │                │
     │ init(config)  │               │                │                │
     │──────────────>│               │                │                │
     │               │ iris_sdk_init │                │                │
     │               │──────────────>│                │                │
     │               │               │ getInstance()  │                │
     │               │               │───────────────>│                │
     │               │               │                │ create(type)   │
     │               │               │                │───────────────>│
     │               │               │                │                │
     │               │               │                │ loadModels()   │
     │               │               │                │<───────────────│
     │               │               │                │                │
     │               │               │<───────────────│                │
     │               │<──────────────│                │                │
     │<──────────────│               │                │                │
     │               │               │                │                │
```

### 6.2 프레임 처리 시퀀스

```
┌────────┐  ┌─────┐  ┌─────────┐  ┌─────────┐  ┌───────────┐  ┌──────────┐  ┌──────────┐
│ Camera │  │ App │  │ Binding │  │  C API  │  │SDKManager │  │ Detector │  │ Renderer │
└───┬────┘  └──┬──┘  └────┬────┘  └────┬────┘  └─────┬─────┘  └────┬─────┘  └────┬─────┘
    │          │          │            │             │             │             │
    │ frame    │          │            │             │             │             │
    │─────────>│          │            │             │             │             │
    │          │          │            │             │             │             │
    │          │ detect() │            │             │             │             │
    │          │─────────>│            │             │             │             │
    │          │          │            │             │             │             │
    │          │          │ sdk_detect │             │             │             │
    │          │          │───────────>│             │             │             │
    │          │          │            │             │             │             │
    │          │          │            │ detect()    │             │             │
    │          │          │            │────────────>│             │             │
    │          │          │            │             │             │             │
    │          │          │            │             │ preprocess  │             │
    │          │          │            │             │────────────>│             │
    │          │          │            │             │             │             │
    │          │          │            │             │ inference   │             │
    │          │          │            │             │────────────>│             │
    │          │          │            │             │             │             │
    │          │          │            │             │<────────────│             │
    │          │          │            │<────────────│             │             │
    │          │          │            │             │             │             │
    │          │          │            │ render()    │             │             │
    │          │          │            │────────────>│             │             │
    │          │          │            │             │             │             │
    │          │          │            │             │ blend       │             │
    │          │          │            │             │────────────────────────────>
    │          │          │            │             │             │             │
    │          │          │            │             │<────────────────────────────
    │          │          │            │<────────────│             │             │
    │          │          │<───────────│             │             │             │
    │          │<─────────│            │             │             │             │
    │          │          │            │             │             │             │
    │ display  │          │            │             │             │             │
    │<─────────│          │            │             │             │             │
```

### 6.3 플랫폼별 프레임 변환

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         ANDROID FRAME FLOW                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Camera2 API                                                                │
│       │                                                                     │
│       ▼                                                                     │
│  ImageReader (YUV_420_888)                                                  │
│       │                                                                     │
│       ▼                                                                     │
│  JNI ByteBuffer                                                             │
│       │                                                                     │
│       ▼                                                                     │
│  FrameProcessor.convert(YUV_420_888 → RGB)                                  │
│       │                                                                     │
│       ▼                                                                     │
│  MediaPipe Inference                                                        │
│       │                                                                     │
│       ▼                                                                     │
│  IrisResult                                                                 │
│       │                                                                     │
│       ▼                                                                     │
│  LensRenderer.render()                                                      │
│       │                                                                     │
│       ▼                                                                     │
│  RGB Frame + Overlay                                                        │
│       │                                                                     │
│       ▼                                                                     │
│  SurfaceView / TextureView                                                  │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│                           iOS FRAME FLOW                                    │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  AVCaptureSession                                                           │
│       │                                                                     │
│       ▼                                                                     │
│  CVPixelBuffer (BGRA)                                                       │
│       │                                                                     │
│       ▼                                                                     │
│  Objective-C++ Bridge                                                       │
│       │                                                                     │
│       ▼                                                                     │
│  FrameProcessor.convert(BGRA → RGB)                                         │
│       │                                                                     │
│       ▼                                                                     │
│  ... (동일한 처리 흐름)                                                      │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. 플랫폼별 바인딩

### 7.1 Android (JNI)

```
┌─────────────────────────────────────┐
│         Android Application         │
│  ┌───────────────────────────────┐  │
│  │     IrisLensSDK.kt            │  │
│  │     (Kotlin API)              │  │
│  └───────────────────────────────┘  │
│                 │                   │
│                 ▼ JNI               │
│  ┌───────────────────────────────┐  │
│  │     iris_jni.cpp              │  │
│  │     (JNI Bridge)              │  │
│  └───────────────────────────────┘  │
│                 │                   │
│                 ▼                   │
│  ┌───────────────────────────────┐  │
│  │     libiris_sdk.so            │  │
│  │     (C++ Core)                │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

**JNI 바인딩 예시:**

```cpp
// android/iris-sdk/src/main/cpp/iris_jni.cpp

extern "C" JNIEXPORT jint JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeDetect(
    JNIEnv* env,
    jobject thiz,
    jbyteArray frame_data,
    jint width,
    jint height,
    jint format,
    jobject result_obj
) {
    jbyte* data = env->GetByteArrayElements(frame_data, nullptr);

    IrisDetectionResult result = {};
    IrisSDKError error = iris_sdk_detect(
        reinterpret_cast<uint8_t*>(data),
        width, height, format, &result
    );

    // result_obj에 결과 복사...

    env->ReleaseByteArrayElements(frame_data, data, JNI_ABORT);
    return static_cast<jint>(error);
}
```

### 7.2 iOS (Objective-C++)

```
┌─────────────────────────────────────┐
│          iOS Application            │
│  ┌───────────────────────────────┐  │
│  │     IrisLensSDK.swift         │  │
│  │     (Swift Interface)         │  │
│  └───────────────────────────────┘  │
│                 │                   │
│                 ▼ Bridge            │
│  ┌───────────────────────────────┐  │
│  │     IrisLensSDK.mm            │  │
│  │     (Objective-C++ Wrapper)   │  │
│  └───────────────────────────────┘  │
│                 │                   │
│                 ▼                   │
│  ┌───────────────────────────────┐  │
│  │     IrisSDK.xcframework       │  │
│  │     (C++ Core)                │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

### 7.3 Flutter (dart:ffi)

```
┌─────────────────────────────────────┐
│         Flutter Application         │
│  ┌───────────────────────────────┐  │
│  │     iris_lens_sdk.dart        │  │
│  │     (Dart API)                │  │
│  └───────────────────────────────┘  │
│                 │                   │
│                 ▼ dart:ffi          │
│  ┌───────────────────────────────┐  │
│  │     Native Library            │  │
│  │     Android: libiris_sdk.so   │  │
│  │     iOS: IrisSDK.xcframework  │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

---

## 8. 프로젝트 구조

```
IrisLensSDK/
├── README.md
├── CLAUDE.md
├── .gitignore
├── .gitattributes
│
├── docs/                           # 문서
│   ├── PROJECT_SPEC.md
│   ├── DECISION_RECORD.md
│   ├── DEVELOPMENT_ROADMAP.md
│   ├── ARCHITECTURE.md             # 본 문서
│   └── workPaper/
│
├── shared/                         # 공유 리소스
│   ├── models/
│   │   ├── face_detection_short_range.tflite
│   │   ├── face_landmark.tflite
│   │   └── iris_landmark.tflite
│   ├── textures/
│   └── test_data/
│
├── cpp/                            # C++ 코어 엔진
│   ├── CMakeLists.txt
│   ├── include/
│   │   ├── iris_sdk/
│   │   │   ├── iris_detector.h
│   │   │   ├── mediapipe_detector.h
│   │   │   ├── lens_renderer.h
│   │   │   ├── frame_processor.h
│   │   │   ├── sdk_manager.h
│   │   │   └── sdk_api.h
│   │   └── iris_sdk.h
│   ├── src/
│   ├── tests/
│   ├── examples/
│   └── third_party/
│
├── python/                         # Python 프로토타이핑
│   ├── requirements.txt
│   ├── iris_sdk/
│   └── examples/
│
├── android/                        # Android
│   ├── iris-sdk/
│   └── demo-app/
│
├── ios/                            # iOS (Phase 2)
│   ├── IrisSDK/
│   └── DemoApp/
│
├── flutter/                        # Flutter (Phase 2)
│   ├── pubspec.yaml
│   └── lib/
│
├── web/                            # WASM (Phase 3)
│
└── scripts/                        # 빌드 스크립트
    ├── build_cpp.sh
    ├── build_android.sh
    └── download_models.sh
```

---

## 9. 확장성 설계

### 9.1 검출기 확장 (Phase 2)

```cpp
// HybridDetector - 폴백 구조
class HybridDetector : public IrisDetector {
public:
    int detect(const Frame& frame, IrisResult& result) override {
        // 1차: MediaPipe 시도
        int ret = mediapipe_->detect(frame, result);
        if (ret == IRIS_SDK_OK && result.isValid()) {
            return IRIS_SDK_OK;
        }

        // 2차: Eye-Only 폴백
        return eye_only_->detect(frame, result);
    }

private:
    std::unique_ptr<MediaPipeDetector> mediapipe_;
    std::unique_ptr<EyeOnlyDetector> eye_only_;
};
```

### 9.2 플러그인 아키텍처

```cpp
// 새 검출기 등록
class DetectorRegistry {
public:
    static void registerDetector(
        const std::string& name,
        std::function<std::unique_ptr<IrisDetector>()> factory
    );

    static std::unique_ptr<IrisDetector> create(const std::string& name);

private:
    static std::map<std::string, DetectorFactory> registry_;
};
```

---

## 10. 성능 고려사항

### 10.1 성능 목표

| 지표 | 목표 | 측정 방법 |
|------|------|----------|
| 프레임 레이트 | 30fps+ | 1초당 처리 프레임 |
| 검출 지연 | 33ms 이하 | 입력→출력 시간 |
| 메모리 사용 | 100MB 이하 | 런타임 RSS |
| SDK 크기 | 20MB 이하 | 라이브러리 파일 |

### 10.2 최적화 전략

| 전략 | 구현 | 효과 |
|------|------|------|
| **입력 리사이즈** | 640px 폭으로 다운스케일 | 추론 시간 감소 |
| **메모리 풀링** | Frame 버퍼 재사용 | 할당/해제 오버헤드 감소 |
| **GPU 가속** | OpenGL ES / Metal | 추론 속도 향상 |
| **NEON/SSE** | 이미지 변환 SIMD | 전처리 속도 향상 |
| **모델 양자화** | INT8 양자화 | 모델 크기/속도 개선 |

### 10.3 메모리 관리

```cpp
// 메모리 풀 예시
class FramePool {
public:
    Frame acquire(int width, int height, PixelFormat format);
    void release(Frame& frame);

private:
    std::vector<Frame> pool_;
    std::mutex mutex_;
};
```

---

## 변경 이력

| 버전 | 날짜 | 변경 내용 |
|------|------|----------|
| 1.0 | 2025-01-07 | 초안 작성 |
| 2.0 | 2025-01-07 | 모노레포 구조 반영, 상세 설계 추가 |
