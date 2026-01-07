# IrisLensSDK 아키텍처 설계

## 1. 시스템 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────┐
│                      Application Layer                       │
│  (Android App / iOS App / Flutter App / Web App)            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Binding Layer                           │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │   JNI   │  │ Obj-C++ │  │dart:ffi │  │  WASM   │        │
│  │(Android)│  │  (iOS)  │  │(Flutter)│  │  (Web)  │        │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        C API Layer                           │
│  iris_sdk_init() | iris_sdk_detect() | iris_sdk_render()    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      C++ Core Engine                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                   SDK Manager                         │   │
│  │  - 초기화/해제 관리                                    │   │
│  │  - 리소스 관리                                        │   │
│  │  - 설정 관리                                          │   │
│  └──────────────────────────────────────────────────────┘   │
│                              │                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │IrisDetector  │    │LensRenderer  │    │FrameProcessor│   │
│  │  (Interface) │    │              │    │              │   │
│  ├──────────────┤    ├──────────────┤    ├──────────────┤   │
│  │MediaPipe     │    │텍스처 로드    │    │전처리        │   │
│  │Detector      │    │홍채 매핑     │    │후처리        │   │
│  ├──────────────┤    │블렌딩       │    │포맷 변환     │   │
│  │EyeOnly       │    │             │    │             │   │
│  │Detector      │    │             │    │             │   │
│  │(Phase 2)     │    │             │    │             │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│                              │                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                 Third Party Libraries                 │   │
│  │  MediaPipe  │  OpenCV  │  TensorFlow Lite            │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 컴포넌트 상세 설계

### 2.1 IrisDetector (추상 인터페이스)

```cpp
// core/include/iris_detector.h

#pragma once
#include <vector>
#include <memory>

namespace iris_sdk {

// 홍채 랜드마크 결과
struct IrisLandmark {
    float x;
    float y;
    float z;  // 깊이 (선택적)
};

struct IrisResult {
    bool detected;
    float confidence;

    // 홍채 중심 및 반경
    float center_x;
    float center_y;
    float radius;

    // 상세 랜드마크 (10개 포인트)
    std::vector<IrisLandmark> landmarks;

    // 눈 영역 바운딩 박스
    float eye_bbox[4];  // x, y, width, height
};

struct FrameData {
    const uint8_t* data;
    int width;
    int height;
    int channels;  // 3 (RGB) or 4 (RGBA)
    int stride;    // bytes per row
};

// 검출기 추상 인터페이스
class IrisDetector {
public:
    virtual ~IrisDetector() = default;

    virtual bool initialize(const std::string& model_path) = 0;
    virtual IrisResult detect(const FrameData& frame) = 0;
    virtual void release() = 0;

    // 팩토리 메서드
    static std::unique_ptr<IrisDetector> create(const std::string& type);
};

}  // namespace iris_sdk
```

### 2.2 MediaPipeDetector

```cpp
// core/include/mediapipe_detector.h

#pragma once
#include "iris_detector.h"

namespace iris_sdk {

class MediaPipeDetector : public IrisDetector {
public:
    MediaPipeDetector();
    ~MediaPipeDetector() override;

    bool initialize(const std::string& model_path) override;
    IrisResult detect(const FrameData& frame) override;
    void release() override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace iris_sdk
```

### 2.3 LensRenderer

```cpp
// core/include/lens_renderer.h

#pragma once
#include "iris_detector.h"
#include <string>

namespace iris_sdk {

struct LensConfig {
    std::string texture_path;
    float opacity = 1.0f;
    float scale = 1.0f;
    bool blend_with_iris = true;
};

class LensRenderer {
public:
    LensRenderer();
    ~LensRenderer();

    bool loadLensTexture(const std::string& path);

    // 프레임에 렌즈 렌더링 (in-place)
    bool render(
        uint8_t* frame_data,
        int width,
        int height,
        int channels,
        const IrisResult& left_eye,
        const IrisResult& right_eye,
        const LensConfig& config
    );

    void release();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace iris_sdk
```

### 2.4 SDK Manager

```cpp
// core/include/sdk_manager.h

#pragma once
#include "iris_detector.h"
#include "lens_renderer.h"
#include <memory>

namespace iris_sdk {

struct SDKConfig {
    std::string model_path;
    std::string detector_type = "mediapipe";  // or "hybrid"
    bool enable_gpu = false;
    int max_faces = 1;
};

class SDKManager {
public:
    static SDKManager& instance();

    bool initialize(const SDKConfig& config);
    bool isInitialized() const;

    IrisDetector* getDetector();
    LensRenderer* getRenderer();

    void release();

private:
    SDKManager();
    ~SDKManager();

    std::unique_ptr<IrisDetector> detector_;
    std::unique_ptr<LensRenderer> renderer_;
    bool initialized_ = false;
};

}  // namespace iris_sdk
```

---

## 3. C API 설계

```cpp
// core/include/sdk_api.h

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// 버전 정보
#define IRIS_SDK_VERSION_MAJOR 1
#define IRIS_SDK_VERSION_MINOR 0
#define IRIS_SDK_VERSION_PATCH 0

// 에러 코드
typedef enum {
    IRIS_SDK_OK = 0,
    IRIS_SDK_ERROR_NOT_INITIALIZED = -1,
    IRIS_SDK_ERROR_INVALID_PARAM = -2,
    IRIS_SDK_ERROR_MODEL_LOAD_FAILED = -3,
    IRIS_SDK_ERROR_DETECTION_FAILED = -4,
    IRIS_SDK_ERROR_RENDER_FAILED = -5,
    IRIS_SDK_ERROR_OUT_OF_MEMORY = -6,
} IrisSDKError;

// 결과 구조체
typedef struct {
    int detected;           // 0 or 1
    float confidence;       // 0.0 ~ 1.0
    float center_x;         // 홍채 중심 X (정규화 0~1)
    float center_y;         // 홍채 중심 Y (정규화 0~1)
    float radius;           // 홍채 반경 (정규화)
    float landmarks[20];    // 10개 포인트 × (x, y)
} IrisSingleResult;

typedef struct {
    IrisSingleResult left_eye;
    IrisSingleResult right_eye;
} IrisDetectionResult;

// 설정 구조체
typedef struct {
    const char* model_path;
    const char* detector_type;  // "mediapipe" or "hybrid"
    int enable_gpu;             // 0 or 1
    int max_faces;
} IrisSDKConfig;

// SDK 초기화/해제
IrisSDKError iris_sdk_init(const IrisSDKConfig* config);
IrisSDKError iris_sdk_destroy(void);
int iris_sdk_is_initialized(void);

// 홍채 검출
IrisSDKError iris_sdk_detect(
    const unsigned char* frame_data,
    int width,
    int height,
    int channels,
    IrisDetectionResult* result
);

// 렌즈 렌더링
IrisSDKError iris_sdk_render_lens(
    unsigned char* frame_data,  // in-out
    int width,
    int height,
    int channels,
    const IrisDetectionResult* iris_result,
    const char* lens_texture_path,
    float opacity
);

// 유틸리티
const char* iris_sdk_get_version(void);
const char* iris_sdk_error_string(IrisSDKError error);

#ifdef __cplusplus
}
#endif
```

---

## 4. 데이터 흐름

### 4.1 검출 파이프라인

```
카메라 프레임 (YUV/RGB)
        │
        ▼
┌───────────────────┐
│  FrameProcessor   │
│  - YUV → RGB 변환  │
│  - 리사이즈        │
│  - 정규화          │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   IrisDetector    │
│  (MediaPipe)      │
│  - Face Detection │
│  - Face Mesh      │
│  - Iris Tracking  │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   IrisResult      │
│  - 검출 여부       │
│  - 신뢰도         │
│  - 랜드마크 좌표   │
└───────────────────┘
```

### 4.2 렌더링 파이프라인

```
IrisResult + 원본 프레임 + 렌즈 텍스처
        │
        ▼
┌───────────────────┐
│   LensRenderer    │
│  1. 홍채 영역 계산 │
│  2. 텍스처 워핑    │
│  3. 알파 블렌딩    │
└───────────────────┘
        │
        ▼
렌즈 합성된 프레임 (출력)
```

---

## 5. 플랫폼별 통합

### 5.1 Android (JNI)

```
┌─────────────────────────────────────┐
│         Android Application         │
│  ┌───────────────────────────────┐  │
│  │     IrisLensSDK.java          │  │
│  │     (Java Wrapper)            │  │
│  └───────────────────────────────┘  │
│                 │                   │
│                 ▼ JNI               │
│  ┌───────────────────────────────┐  │
│  │     iris_sdk_jni.cpp          │  │
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

### 5.2 iOS (Objective-C++)

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
│  │     IrisSDK.framework         │  │
│  │     (C++ Core)                │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

### 5.3 Flutter (dart:ffi)

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
│  │     iOS: IrisSDK.framework    │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

---

## 6. 프로젝트 폴더 구조

```
iris-lens-sdk/
├── CMakeLists.txt                 # 루트 빌드 설정
├── README.md
├── LICENSE
│
├── core/                          # C++ 코어 엔진
│   ├── CMakeLists.txt
│   ├── include/
│   │   ├── iris_detector.h
│   │   ├── mediapipe_detector.h
│   │   ├── eye_only_detector.h    # Phase 2
│   │   ├── hybrid_detector.h      # Phase 2
│   │   ├── lens_renderer.h
│   │   ├── frame_processor.h
│   │   ├── sdk_manager.h
│   │   └── sdk_api.h
│   ├── src/
│   │   ├── iris_detector.cpp
│   │   ├── mediapipe_detector.cpp
│   │   ├── lens_renderer.cpp
│   │   ├── frame_processor.cpp
│   │   ├── sdk_manager.cpp
│   │   └── sdk_api.cpp
│   └── third_party/
│       ├── mediapipe/
│       └── opencv/
│
├── bindings/
│   ├── android/
│   │   ├── CMakeLists.txt
│   │   ├── jni/
│   │   │   └── iris_sdk_jni.cpp
│   │   └── java/
│   │       └── com/irislens/sdk/
│   │           ├── IrisLensSDK.java
│   │           └── IrisResult.java
│   │
│   ├── ios/
│   │   ├── IrisLensSDK.h
│   │   ├── IrisLensSDK.mm
│   │   └── IrisLensSDK.podspec
│   │
│   ├── flutter/
│   │   ├── pubspec.yaml
│   │   ├── lib/
│   │   │   ├── iris_lens_sdk.dart
│   │   │   └── iris_result.dart
│   │   └── src/
│   │       └── ffi_bindings.dart
│   │
│   └── web/
│       ├── CMakeLists.txt
│       └── iris_sdk_wasm.cpp
│
├── models/
│   ├── mediapipe/
│   │   ├── face_detection.tflite
│   │   ├── face_mesh.tflite
│   │   └── iris_landmark.tflite
│   └── custom/                    # Phase 2
│       └── eye_only.tflite
│
├── examples/
│   ├── android-demo/
│   ├── ios-demo/
│   ├── flutter-demo/
│   └── web-demo/
│
├── tests/
│   ├── core_tests/
│   └── integration_tests/
│
├── scripts/
│   ├── build_android.sh
│   ├── build_ios.sh
│   └── build_all.sh
│
└── docs/
    ├── PROJECT_SPEC.md
    ├── DEVELOPMENT_ROADMAP.md
    ├── ARCHITECTURE.md
    └── API_REFERENCE.md
```

---

## 7. 빌드 시스템

### 7.1 루트 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.18)
project(iris_lens_sdk VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_ANDROID "Build for Android" OFF)
option(BUILD_IOS "Build for iOS" OFF)
option(BUILD_WASM "Build for WebAssembly" OFF)
option(BUILD_TESTS "Build tests" ON)

# Core 라이브러리
add_subdirectory(core)

# 플랫폼별 바인딩
if(BUILD_ANDROID)
    add_subdirectory(bindings/android)
elseif(BUILD_IOS)
    add_subdirectory(bindings/ios)
elseif(BUILD_WASM)
    add_subdirectory(bindings/web)
endif()

# 테스트
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
```

### 7.2 Core CMakeLists.txt

```cmake
# core/CMakeLists.txt

# 의존성
find_package(OpenCV REQUIRED)
# MediaPipe는 별도 처리 필요

# 소스 파일
set(CORE_SOURCES
    src/iris_detector.cpp
    src/mediapipe_detector.cpp
    src/lens_renderer.cpp
    src/frame_processor.cpp
    src/sdk_manager.cpp
    src/sdk_api.cpp
)

# 라이브러리 생성
add_library(iris_sdk SHARED ${CORE_SOURCES})

target_include_directories(iris_sdk
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(iris_sdk
    PUBLIC
        ${OpenCV_LIBS}
        # mediapipe libs
)

# 설치
install(TARGETS iris_sdk DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
```
