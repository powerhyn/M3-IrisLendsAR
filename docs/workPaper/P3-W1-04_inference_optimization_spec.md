# P3-W1-04: 추론 최적화 구현 스펙

**상태**: ✅ 완료
**작성일**: 2026-02-09
**담당**: ml-vision
**선행**: P3-W1-03 (양자화 분석)

---

## 1. XNNPACK Delegate 활성화

### 1.1 현황 분석

CMake에서 `IRIS_SDK_HAS_XNNPACK` 매크로를 정의하지만, C++ 코드에서 실제로 XNNPACK delegate를 생성/적용하는 코드가 없습니다.

**CMakeLists.txt** (line 208, 482-483):
```cmake
option(IRIS_SDK_TFLITE_ENABLE_XNNPACK "Enable XNNPACK delegate..." ON)  # 기본 ON
# ...
if(IRIS_SDK_TFLITE_ENABLE_XNNPACK)
    target_compile_definitions(iris_sdk PRIVATE IRIS_SDK_HAS_XNNPACK)
endif()
```

**mediapipe_detector.cpp `loadModel()`** (line 445-529):
- GPU Delegate 적용 코드는 있음 (`IRIS_SDK_GPU_ENABLED` 블록)
- XNNPACK Delegate 적용 코드는 없음

### 1.2 구현 스펙

**수정 대상**: `cpp/src/mediapipe_detector.cpp` - `loadModel()` 메서드

**삽입 위치**: GPU Delegate 블록 이후, `AllocateTensors()` 이전 (line 511-521 사이)

```cpp
// ========================================
// XNNPACK Delegate 적용 (CPU 가속)
// GPU 미사용 시에만 XNNPACK 적용 (GPU와 XNNPACK 동시 사용 불가)
// ========================================
#ifdef IRIS_SDK_HAS_XNNPACK
#if IRIS_SDK_GPU_ENABLED
    bool gpu_applied = (out_gpu_delegate != nullptr && *out_gpu_delegate != nullptr);
    if (!gpu_applied) {
#else
    {
#endif
        // XNNPACK delegate 옵션 설정
        TfLiteXNNPackDelegateOptions xnnpack_opts =
            TfLiteXNNPackDelegateOptionsDefault();
        xnnpack_opts.num_threads = num_threads;

        TfLiteDelegate* xnnpack_delegate =
            TfLiteXNNPackDelegateCreate(&xnnpack_opts);
        if (xnnpack_delegate != nullptr) {
            TfLiteStatus status =
                interpreter->ModifyGraphWithDelegate(xnnpack_delegate);
            if (status == kTfLiteOk) {
                std::fprintf(stderr,
                    "[IrisSDK] XNNPACK delegate activated: %s\n",
                    model_file.c_str());
            } else {
                // XNNPACK 적용 실패 시 기본 CPU 폴백 (에러 아님)
                TfLiteXNNPackDelegateDelete(xnnpack_delegate);
            }
        }
        interpreter->SetNumThreads(num_threads);
    }
#else
    // XNNPACK 미지원 빌드: 기본 CPU
    interpreter->SetNumThreads(num_threads);
#endif  // IRIS_SDK_HAS_XNNPACK
```

**필요 헤더** (mediapipe_detector.cpp 상단에 추가):
```cpp
#ifdef IRIS_SDK_HAS_XNNPACK
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#endif
```

### 1.3 XNNPACK Delegate 수명 관리

현재 GPU delegate는 `Impl` 클래스에서 포인터를 보관하고 `releaseGpuDelegates()`에서 해제합니다.
XNNPACK delegate도 동일한 패턴을 적용해야 합니다.

**`Impl` 클래스에 추가할 필드** (line 156 부근):
```cpp
#ifdef IRIS_SDK_HAS_XNNPACK
    // XNNPACK delegate 포인터 (수동 해제 필요)
    TfLiteDelegate* xnnpack_delegate_face_detection = nullptr;
    TfLiteDelegate* xnnpack_delegate_face_landmark = nullptr;
    TfLiteDelegate* xnnpack_delegate_iris_landmark = nullptr;  // V1 전용
#endif
```

**해제 함수 추가** (`releaseGpuDelegates()` 이후):
```cpp
void releaseXnnpackDelegates() {
#ifdef IRIS_SDK_HAS_XNNPACK
    auto release = [](TfLiteDelegate*& d) {
        if (d != nullptr) {
            TfLiteXNNPackDelegateDelete(d);
            d = nullptr;
        }
    };
    release(xnnpack_delegate_face_detection);
    release(xnnpack_delegate_face_landmark);
    release(xnnpack_delegate_iris_landmark);
#endif
}
```

### 1.4 기대 효과

| 디바이스 | 현재 (CPU) | XNNPACK | 향상 |
|----------|-----------|---------|------|
| Snapdragon 865 | 8-12ms | 5-7ms | ~1.5x |
| Exynos 2100 | 10-15ms | 6-9ms | ~1.6x |
| Tensor G2 | 9-13ms | 5-8ms | ~1.7x |

(Face Landmark V2 추론 시간 기준, XNNPACK은 ARM NEON SIMD 최적화 적용)

---

## 2. V2 전용 모드 전환

### 2.1 현황

코드는 이미 V2 우선 로딩을 지원합니다 (`loadAllModels()` line 580-616):
1. `face_landmark_v2.tflite` 존재 시 V2 로드 (`model_version = 2`)
2. 실패 시 V1으로 폴백 (`model_version = 1`)
3. V2 성공 시 iris_landmark 모델 스킵 (line 653-659)

**결론**: V2 전용 모드는 이미 구현되어 있습니다. `face_landmark_v2.tflite`가 존재하면 자동으로 V2 모드로 동작합니다.

### 2.2 추가 최적화: V1 폴백 비활성화 옵션

프로덕션 빌드에서 V1 모델 파일 자체를 배포에서 제외하면:
- **SDK 크기 절감**: `face_landmark.tflite` (1.2MB) + `iris_landmark.tflite` (2.5MB) = **3.7MB 절감**
- **초기화 시간 단축**: 파일 존재 확인 1회 감소

**구현**: 빌드 시점 결정 (코드 변경 불필요)
```
# Android AAR 빌드 시 V2 전용 모델만 포함
shared/models/
├── face_detection_short_range.tflite  (224KB, 필수)
├── face_landmark_v2.tflite            (2.4MB, 필수)
└── (face_landmark.tflite 제외)
└── (iris_landmark.tflite 제외)
```

### 2.3 V2 모드 검증 강화

현재 V2 모드에서 `getModelVersion()` API가 존재합니다 (line 2851).
프로덕션에서 V2 모드 확인을 위한 로깅 강화:

```cpp
// initialize() 종료 시 로그
std::fprintf(stderr, "[IrisSDK] Model version: V%d, Models loaded: %d, GPU: %s\n",
    impl_->model_version,
    (impl_->model_version == 1) ? 3 : 2,
    impl_->gpu_active ? "active" : "inactive");
```

---

## 3. GPU Delegate Warm-up 전략

### 3.1 문제

TFLite GPU Delegate는 첫 추론 시 GPU 셰이더 컴파일이 발생하여 100-500ms 지연이 생깁니다.
사용자 경험상 카메라 첫 프레임이 눈에 띄게 느려집니다.

### 3.2 구현 스펙

**수정 대상**: `cpp/src/mediapipe_detector.cpp` - `initializeBuffers()` 이후

**삽입 위치**: `initialize()` 메서드 내부, `initializeBuffers()` 호출 후 (line 1953 이후)

```cpp
// GPU Delegate warm-up: 더미 추론으로 셰이더 사전 컴파일
#if IRIS_SDK_GPU_ENABLED
    if (impl_->gpu_active) {
        impl_->warmupGpuDelegates();
    }
#endif
```

**warmupGpuDelegates() 구현**:
```cpp
void warmupGpuDelegates() {
    // 더미 입력으로 각 모델 1회 추론 (셰이더 컴파일 유도)
    std::fprintf(stderr, "[IrisSDK] GPU warm-up starting...\n");

    // Face Detection warm-up
    if (face_detection_interpreter) {
        std::vector<float> dummy_fd(
            FACE_DETECTION_INPUT_WIDTH * FACE_DETECTION_INPUT_HEIGHT *
            FACE_DETECTION_INPUT_CHANNELS, 0.0f);
        float* input = face_detection_interpreter->typed_input_tensor<float>(0);
        if (input) {
            std::memcpy(input, dummy_fd.data(), dummy_fd.size() * sizeof(float));
            face_detection_interpreter->Invoke();
        }
    }

    // Face Landmark warm-up
    if (face_landmark_interpreter) {
        int fl_w = (model_version == 2) ?
            FACE_LANDMARK_V2_INPUT_WIDTH : FACE_LANDMARK_INPUT_WIDTH;
        int fl_h = (model_version == 2) ?
            FACE_LANDMARK_V2_INPUT_HEIGHT : FACE_LANDMARK_INPUT_HEIGHT;
        std::vector<float> dummy_fl(fl_w * fl_h * FACE_LANDMARK_INPUT_CHANNELS, 0.0f);
        float* input = face_landmark_interpreter->typed_input_tensor<float>(0);
        if (input) {
            std::memcpy(input, dummy_fl.data(), dummy_fl.size() * sizeof(float));
            face_landmark_interpreter->Invoke();
        }
    }

    // Iris Landmark warm-up (V1 전용)
    if (model_version == 1 && iris_landmark_interpreter) {
        std::vector<float> dummy_iris(
            IRIS_LANDMARK_INPUT_WIDTH * IRIS_LANDMARK_INPUT_HEIGHT *
            IRIS_LANDMARK_INPUT_CHANNELS, 0.0f);
        float* input = iris_landmark_interpreter->typed_input_tensor<float>(0);
        if (input) {
            std::memcpy(input, dummy_iris.data(), dummy_iris.size() * sizeof(float));
            iris_landmark_interpreter->Invoke();
        }
    }

    std::fprintf(stderr, "[IrisSDK] GPU warm-up completed\n");
}
```

### 3.3 주의사항

- Warm-up은 `initialize()` 내부에서 동기적으로 실행 (UI 스레드 차단 가능)
- Android에서는 `initialize()`가 백그라운드 스레드에서 호출되는지 확인 필요
- Warm-up 소요 시간: GPU 모델 2-3개 기준 약 200-500ms

### 3.4 비동기 Warm-up 대안

```cpp
// 비동기 warm-up (선택적 구현)
std::future<void> warmup_future;

void startAsyncWarmup() {
    warmup_future = std::async(std::launch::async, [this]() {
        warmupGpuDelegates();
    });
}

// 첫 detect() 호출 시 완료 대기
void ensureWarmupComplete() {
    if (warmup_future.valid()) {
        warmup_future.wait();
    }
}
```

---

## 4. ML-Free 피부 세그멘테이션 마스크 설계

### 4.1 개요

Face Landmark V2의 478개 랜드마크를 사용하여 추가 ML 모델 없이
피부 영역 마스크를 GPU 셰이더에서 직접 생성합니다.

### 4.2 피부 영역 랜드마크 인덱스 매핑

#### 기존 코드에서 사용 중인 인덱스

프로젝트에 이미 정의된 랜드마크 인덱스 테이블 (`beauty_roi_manager.cpp`):

| 영역 | 개수 | 인덱스 |
|------|------|--------|
| **Face Oval** | 36 | 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109 |
| **Left Eye** | 16 | 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246 |
| **Right Eye** | 16 | 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398 |
| **Left Eyebrow** | 8 | 70, 63, 105, 66, 107, 55, 65, 52 |
| **Right Eyebrow** | 8 | 300, 293, 334, 296, 336, 285, 295, 282 |
| **Lips (inner)** | 22 | 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78 |
| **Lips (outer)** | 20 | 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185 |
| **Left Cheek** | 8 | 234, 93, 132, 58, 172, 136, 150, 149 |
| **Right Cheek** | 8 | 454, 323, 361, 288, 397, 365, 379, 378 |

#### 추가 정의: 세분화된 피부 영역

```cpp
// 이마 영역 (기존 cpu_beauty_backend.cpp에서 사용 중)
static constexpr int FOREHEAD_TOP[] = {10, 338, 297, 332, 284, 251, 389};
static constexpr int FOREHEAD_BOTTOM[] = {389, 251, 284, 332, 297, 338, 10};

// 코 영역
static constexpr int NOSE_BRIDGE[] = {6, 197, 195, 5, 4, 1, 19};
static constexpr int NOSE_TIP[] = {1, 2, 98, 327};
static constexpr int NOSE_WING_LEFT[] = {129, 49, 131, 134};
static constexpr int NOSE_WING_RIGHT[] = {358, 279, 360, 363};

// 턱 라인 (Face Oval 하단 부분 추출)
static constexpr int JAW_LINE[] = {
    152, 148, 176, 149, 150, 136, 172, 58, 132,  // 좌측 턱
    377, 400, 378, 379, 365, 397, 288, 361, 323   // 우측 턱
};

// 미간
static constexpr int GLABELLA[] = {9, 8, 168, 6, 197, 195, 5, 4};
```

### 4.3 마스크 생성 알고리즘

#### 방법: 다중 다각형 합성 (Polygon Union)

```
피부 마스크 = Face Oval 전체
            - Left Eye 영역
            - Right Eye 영역
            - Left Eyebrow 영역 (선택적)
            - Right Eyebrow 영역 (선택적)
            - Lips 영역 (선택적)
```

#### GPU 셰이더 구현 (OpenGL ES 3.0)

**접근 1: CPU에서 마스크 텍스처 생성 후 업로드**

```
1. Face Oval 36점으로 다각형 내부 채우기 (scanline fill)
2. 눈/입술 영역 제외 (다각형 차집합)
3. 가우시안 블러로 경계 부드럽게
4. GL_TEXTURE_2D로 업로드 → 셰이더에서 샘플링
```

**접근 2: GPU에서 직접 마스크 렌더링 (권장)**

```
1. Face Oval 36점을 삼각형 팬(triangle fan)으로 분할
2. 별도 FBO에 마스크 렌더링 (GL_TRIANGLES)
3. 눈/입술 영역을 검은색으로 덮어쓰기
4. Gaussian blur pass로 경계 스무딩
5. Beauty 셰이더에서 마스크 텍스처 참조
```

#### 삼각형 분할 (Triangulation)

Face Oval 36점을 삼각형 팬으로 분할:

```
중심점 = Face Oval 36점의 무게중심 (centroid)
삼각형[i] = (centroid, oval[i], oval[(i+1) % 36])
총 36개 삼각형
```

### 4.4 GPU 마스크 렌더링 셰이더

#### Vertex Shader (`skin_mask.vert`)

```glsl
#version 300 es
precision mediump float;

// 랜드마크 좌표 (정규화 0-1, 화면 좌표로 변환)
in vec2 a_position;
// 마스크 강도 (1.0 = 피부, 0.0 = 비피부)
in float a_mask_value;

out float v_mask_value;

void main() {
    // 정규화 좌표 → NDC (-1 ~ +1)
    vec2 ndc = a_position * 2.0 - 1.0;
    ndc.y = -ndc.y;  // Y축 반전 (OpenGL ↔ 이미지 좌표)
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_mask_value = a_mask_value;
}
```

#### Fragment Shader (`skin_mask.frag`)

```glsl
#version 300 es
precision mediump float;

in float v_mask_value;
out vec4 fragColor;

void main() {
    fragColor = vec4(v_mask_value, v_mask_value, v_mask_value, 1.0);
}
```

#### Beauty 셰이더에서 마스크 사용 (수정)

```glsl
// 기존 beauty 셰이더에 마스크 텍스처 추가
uniform sampler2D u_skin_mask;
uniform float u_mask_enabled;  // 0.0 = 전체 적용, 1.0 = 마스크 적용

// 마스크 기반 필터 강도 조절
float mask = u_mask_enabled > 0.5
    ? texture(u_skin_mask, v_texCoord).r
    : 1.0;

// 원본과 필터 결과를 마스크로 블렌딩
vec3 final_color = mix(original_color, filtered_color, mask * u_intensity);
```

### 4.5 C++ 인터페이스 설계

```cpp
// cpp/include/iris_sdk/gpu/skin_mask_renderer.h

namespace iris_sdk {

class SkinMaskRenderer {
public:
    SkinMaskRenderer();
    ~SkinMaskRenderer();

    /// 초기화 (셰이더 컴파일, VAO/VBO 생성)
    bool initialize();

    /// 피부 마스크 텍스처 생성/업데이트
    /// @param face_mesh 478개 랜드마크 (정규화 좌표)
    /// @param width 마스크 텍스처 너비
    /// @param height 마스크 텍스처 높이
    /// @return 마스크 텍스처 ID (GLuint)
    uint32_t renderSkinMask(
        const IrisLandmark* face_mesh,
        int width, int height);

    /// 리소스 해제
    void release();

    /// 마스크 영역 설정
    struct MaskConfig {
        bool exclude_eyes = true;       ///< 눈 영역 제외
        bool exclude_eyebrows = false;  ///< 눈썹 제외
        bool exclude_lips = true;       ///< 입술 제외
        float edge_blur_radius = 5.0f;  ///< 경계 블러 반경 (px)
    };

    void setMaskConfig(const MaskConfig& config);

private:
    // Face Oval 삼각형 팬 생성
    void buildTriangleFan(const IrisLandmark* face_mesh,
                          std::vector<float>& vertices);

    // 제외 영역 (눈/입술) 다각형 생성
    void buildExclusionPolygons(const IrisLandmark* face_mesh,
                                std::vector<float>& vertices);

    // GPU 리소스
    uint32_t mask_fbo_ = 0;
    uint32_t mask_texture_ = 0;
    uint32_t mask_program_ = 0;
    uint32_t vao_ = 0;
    uint32_t vbo_ = 0;

    MaskConfig config_;
    int current_width_ = 0;
    int current_height_ = 0;
};

} // namespace iris_sdk
```

### 4.6 GPUBeautyBackend 통합

**수정 대상**: `cpp/src/gpu/gpu_beauty_backend.cpp` - `applyTextureId()`

```cpp
// applyTextureId() 내부, 필터 체인 실행 전에 추가

// 피부 마스크 생성 (Face Mesh 기반)
GLuint skin_mask_tex = 0;
if (skin_mask_renderer_ && detection && detection->face_mesh_valid) {
    skin_mask_tex = skin_mask_renderer_->renderSkinMask(
        detection->face_mesh, width, height);
}

// 각 필터 패스에 마스크 전달
if (config.smoothing > 0.01f) {
    executeSmoothingPass(current_input, current_output->fbo_id,
                         width, height, config,
                         skin_mask_tex);  // 마스크 추가
    // ...
}
```

### 4.7 영역별 뷰티 강도 차별화 (beauty-tuner 연계)

마스크를 단순 이진(0/1)이 아닌 영역별 가중치로 확장 가능:

| 영역 | 마스크 값 | 용도 |
|------|-----------|------|
| 이마 | 0.8 | 스무딩 약하게 (모공 텍스처 유지) |
| 볼 | 1.0 | 스무딩 최대 (피부결 보정 핵심) |
| 코 | 0.6 | 하이라이트 보존 |
| 턱 | 0.9 | 피부톤 균일화 |
| 눈 | 0.0 | 제외 |
| 입술 | 0.0 | 제외 (별도 립 필터) |

```glsl
// 멀티채널 마스크 (R=스무딩, G=화이트닝, B=컬러밸런스)
uniform sampler2D u_skin_mask;

vec3 mask = texture(u_skin_mask, v_texCoord).rgb;
float smoothing_mask = mask.r;
float whitening_mask = mask.g;
float color_mask = mask.b;
```

---

## 5. 구현 우선순위 및 의존성

```
[1] XNNPACK Delegate 활성화
    │  난이도: 낮음 (코드 추가만)
    │  효과: CPU 추론 1.5-2x
    │  담당: cpp-engine
    │
[2] GPU Delegate Warm-up
    │  난이도: 낮음
    │  효과: 첫 프레임 지연 제거
    │  담당: cpp-engine
    │
[3] ML-Free 피부 마스크 (SkinMaskRenderer)
    │  난이도: 중간
    │  효과: 영역별 뷰티 차별화
    │  담당: cpp-engine + ml-vision (설계)
    │  의존: beauty-tuner 셰이더 수정
    │
[4] V2 전용 빌드 (모델 파일 제외)
       난이도: 낮음 (빌드 설정)
       효과: SDK 크기 3.7MB 절감
       담당: android-dev (AAR 빌드)
```

---

## 6. 테스트 방안

### XNNPACK

```cpp
// 기존 test_mediapipe_detector_integration 확장
// XNNPACK 활성/비활성 시 결과 일치 확인
TEST(MediaPipeDetector, XnnpackResultConsistency) {
    // 동일 입력 → XNNPACK on/off 결과 비교
    // 허용 오차: 랜드마크 좌표 ±0.001
}
```

### GPU Warm-up

```
Android 디바이스에서 측정:
1. warm-up 없이 첫 detect() 시간
2. warm-up 후 첫 detect() 시간
3. 차이가 100ms 이상이면 warm-up 효과 확인
```

### 피부 마스크

```
1. 마스크 텍스처 스크린샷으로 시각적 검증
2. Face Oval 내부 = 밝은 영역, 눈/입 = 어두운 영역
3. 경계 부드러움 확인 (aliasing 없어야 함)
```

---

## 7. SkinMaskRenderer C++ 구현 상세 가이드

> Task #21 산출물 - `buildTriangleFan()`, `buildExclusionPolygons()`, Gaussian blur pass, FBO 해상도 전략

### 7.1 Face Oval Triangle Fan - EBO 인덱스 배열

#### 7.1.1 정점 레이아웃

Face Oval 36점으로 삼각형 팬을 구성한다. 중심점(centroid)을 추가하여 총 **37개 정점**, **36개 삼각형**을 생성한다.

```
정점 배열 구조 (VBO):
  index 0:  centroid (무게중심)       mask_value = 1.0
  index 1:  oval[0]  = landmark[10]   mask_value = 1.0
  index 2:  oval[1]  = landmark[338]  mask_value = 1.0
  index 3:  oval[2]  = landmark[297]  mask_value = 1.0
  ...
  index 36: oval[35] = landmark[109]  mask_value = 1.0
```

#### 7.1.2 Centroid 계산

```cpp
void SkinMaskRenderer::buildTriangleFan(
    const IrisLandmark* face_mesh,
    std::vector<float>& vertices,
    std::vector<uint16_t>& indices) {

    // Face Oval 36점 인덱스 (beauty_roi_manager.cpp line 19-24 기준)
    static constexpr int FACE_OVAL_INDICES[36] = {
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
        361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
        176, 149, 150, 136, 172,  58, 132,  93, 234, 127,
        162,  21,  54, 103,  67, 109
    };

    // 1. Centroid 계산
    float cx = 0.0f, cy = 0.0f;
    for (int i = 0; i < 36; ++i) {
        int idx = FACE_OVAL_INDICES[i];
        cx += face_mesh[idx].x;
        cy += face_mesh[idx].y;
    }
    cx /= 36.0f;
    cy /= 36.0f;

    // 2. 정점 배열 생성 (stride: x, y, mask_value = 3 floats)
    vertices.clear();
    vertices.reserve(37 * 3);  // 37 vertices * 3 components

    // index 0: centroid
    vertices.push_back(cx);
    vertices.push_back(cy);
    vertices.push_back(1.0f);  // mask_value

    // index 1~36: oval points
    for (int i = 0; i < 36; ++i) {
        int idx = FACE_OVAL_INDICES[i];
        vertices.push_back(face_mesh[idx].x);
        vertices.push_back(face_mesh[idx].y);
        vertices.push_back(1.0f);  // mask_value
    }

    // 3. EBO 인덱스 배열 (36 triangles * 3 indices = 108)
    indices.clear();
    indices.reserve(36 * 3);

    for (int i = 0; i < 36; ++i) {
        indices.push_back(0);               // centroid
        indices.push_back(i + 1);           // oval[i]
        indices.push_back((i + 1) % 36 + 1); // oval[(i+1) % 36]
    }
}
```

#### 7.1.3 구체적 EBO 인덱스 테이블

```
삼각형  | 인덱스 (centroid, oval[i], oval[i+1])
--------|--------------------------------------
 T0     | 0, 1,  2    → centroid, lm[10],  lm[338]
 T1     | 0, 2,  3    → centroid, lm[338], lm[297]
 T2     | 0, 3,  4    → centroid, lm[297], lm[332]
 T3     | 0, 4,  5    → centroid, lm[332], lm[284]
 T4     | 0, 5,  6    → centroid, lm[284], lm[251]
 T5     | 0, 6,  7    → centroid, lm[251], lm[389]
 T6     | 0, 7,  8    → centroid, lm[389], lm[356]
 T7     | 0, 8,  9    → centroid, lm[356], lm[454]
 T8     | 0, 9,  10   → centroid, lm[454], lm[323]
 T9     | 0, 10, 11   → centroid, lm[323], lm[361]
 T10    | 0, 11, 12   → centroid, lm[361], lm[288]
 T11    | 0, 12, 13   → centroid, lm[288], lm[397]
 T12    | 0, 13, 14   → centroid, lm[397], lm[365]
 T13    | 0, 14, 15   → centroid, lm[365], lm[379]
 T14    | 0, 15, 16   → centroid, lm[379], lm[378]
 T15    | 0, 16, 17   → centroid, lm[378], lm[400]
 T16    | 0, 17, 18   → centroid, lm[400], lm[377]
 T17    | 0, 18, 19   → centroid, lm[377], lm[152]
 T18    | 0, 19, 20   → centroid, lm[152], lm[148]
 T19    | 0, 20, 21   → centroid, lm[148], lm[176]
 T20    | 0, 21, 22   → centroid, lm[176], lm[149]
 T21    | 0, 22, 23   → centroid, lm[149], lm[150]
 T22    | 0, 23, 24   → centroid, lm[150], lm[136]
 T23    | 0, 24, 25   → centroid, lm[136], lm[172]
 T24    | 0, 25, 26   → centroid, lm[172], lm[58]
 T25    | 0, 26, 27   → centroid, lm[58],  lm[132]
 T26    | 0, 27, 28   → centroid, lm[132], lm[93]
 T27    | 0, 28, 29   → centroid, lm[93],  lm[234]
 T28    | 0, 29, 30   → centroid, lm[234], lm[127]
 T29    | 0, 30, 31   → centroid, lm[127], lm[162]
 T30    | 0, 31, 32   → centroid, lm[162], lm[21]
 T31    | 0, 32, 33   → centroid, lm[21],  lm[54]
 T32    | 0, 33, 34   → centroid, lm[54],  lm[103]
 T33    | 0, 34, 35   → centroid, lm[103], lm[67]
 T34    | 0, 35, 36   → centroid, lm[67],  lm[109]
 T35    | 0, 36, 1    → centroid, lm[109], lm[10]  ← 닫힘
```

총 108개 인덱스, `GL_TRIANGLES` 모드로 `glDrawElements()` 호출.

### 7.2 buildExclusionPolygons - 눈/입술 삼각형 분할

제외 영역은 `mask_value = 0.0`으로 **검은색 삼각형**을 Face Oval 위에 덮어쓰기한다. 각 영역도 triangle fan 방식으로 분할한다.

#### 7.2.1 눈 영역 (Left Eye: 16점, Right Eye: 16점)

```cpp
void SkinMaskRenderer::buildExclusionPolygons(
    const IrisLandmark* face_mesh,
    std::vector<float>& vertices,
    std::vector<uint16_t>& indices) {

    // Left Eye 16점 (beauty_roi_manager.cpp line 27-30)
    static constexpr int LEFT_EYE_INDICES[16] = {
        33, 7, 163, 144, 145, 153, 154, 155,
        133, 173, 157, 158, 159, 160, 161, 246
    };

    // Right Eye 16점 (beauty_roi_manager.cpp line 33-36)
    static constexpr int RIGHT_EYE_INDICES[16] = {
        362, 382, 381, 380, 374, 373, 390, 249,
        263, 466, 388, 387, 386, 385, 384, 398
    };

    // Lips 22점 (beauty_roi_manager.cpp line 39-42)
    static constexpr int LIPS_INDICES[22] = {
        61, 146, 91, 181, 84, 17, 314, 405,
        321, 375, 291, 308, 324, 318, 402, 317,
        14, 87, 178, 88, 95, 78
    };

    vertices.clear();
    indices.clear();

    // === Left Eye (16점 → 16 삼각형) ===
    if (config_.exclude_eyes) {
        appendTriangleFan(face_mesh, LEFT_EYE_INDICES, 16,
                          0.0f,  // mask_value = 0 (제외)
                          vertices, indices);

        // === Right Eye (16점 → 16 삼각형) ===
        appendTriangleFan(face_mesh, RIGHT_EYE_INDICES, 16,
                          0.0f, vertices, indices);
    }

    // === Lips (22점 → 22 삼각형) ===
    if (config_.exclude_lips) {
        appendTriangleFan(face_mesh, LIPS_INDICES, 22,
                          0.0f, vertices, indices);
    }
}
```

#### 7.2.2 공통 Triangle Fan 헬퍼

```cpp
void SkinMaskRenderer::appendTriangleFan(
    const IrisLandmark* face_mesh,
    const int* polygon_indices,
    int count,
    float mask_value,
    std::vector<float>& vertices,
    std::vector<uint16_t>& indices) {

    // 1. centroid 계산
    float cx = 0.0f, cy = 0.0f;
    for (int i = 0; i < count; ++i) {
        cx += face_mesh[polygon_indices[i]].x;
        cy += face_mesh[polygon_indices[i]].y;
    }
    cx /= static_cast<float>(count);
    cy /= static_cast<float>(count);

    // 2. 현재 정점 배열 오프셋 (이전 다각형들의 정점 수)
    uint16_t base = static_cast<uint16_t>(vertices.size() / 3);

    // 3. centroid 정점 추가
    vertices.push_back(cx);
    vertices.push_back(cy);
    vertices.push_back(mask_value);

    // 4. 다각형 정점 추가
    for (int i = 0; i < count; ++i) {
        int idx = polygon_indices[i];
        vertices.push_back(face_mesh[idx].x);
        vertices.push_back(face_mesh[idx].y);
        vertices.push_back(mask_value);
    }

    // 5. 삼각형 인덱스 추가
    for (int i = 0; i < count; ++i) {
        indices.push_back(base);                           // centroid
        indices.push_back(base + i + 1);                   // polygon[i]
        indices.push_back(base + (i + 1) % count + 1);    // polygon[(i+1) % N]
    }
}
```

#### 7.2.3 제외 영역 삼각형 테이블

```
영역        | 정점 수 | centroid | 삼각형 수 | EBO 인덱스 수
------------|---------|----------|-----------|-------------
Left Eye    | 16+1=17 | 16점 중심 | 16        | 48
Right Eye   | 16+1=17 | 16점 중심 | 16        | 48
Lips        | 22+1=23 | 22점 중심 | 22        | 66
------------|---------|----------|-----------|-------------
합계         | 57      |          | 54        | 162
```

#### 7.2.4 렌더링 순서

```
Pass 1: 흰색(1.0) Face Oval 삼각형 팬 렌더링 (36 triangles)
Pass 2: 검은색(0.0) Left Eye 삼각형 팬 덮어쓰기 (16 triangles)
Pass 3: 검은색(0.0) Right Eye 삼각형 팬 덮어쓰기 (16 triangles)
Pass 4: 검은색(0.0) Lips 삼각형 팬 덮어쓰기 (22 triangles)
```

실제로는 Pass 1~4를 **단일 draw call**로 통합 가능하다. VBO에 모든 정점(37 + 57 = 94개)을 넣고, EBO에 모든 인덱스(108 + 162 = 270개)를 넣어, **`glDrawElements(GL_TRIANGLES, 270, GL_UNSIGNED_SHORT, 0)`** 한 번으로 처리한다. 정점의 `mask_value` 어트리뷰트가 1.0 또는 0.0이므로 셰이더가 자동으로 올바른 색상을 출력한다.

단, 이 방식은 제외 영역이 Face Oval 위에 정확히 그려져야 하므로 **깊이 테스트 비활성화 + 그리기 순서 보장**이 필요하다. OpenGL ES에서 `glDrawElements` 단일 호출 시 인덱스 순서대로 삼각형이 그려지므로, EBO 배열에서 Face Oval 인덱스를 먼저, 제외 영역 인덱스를 뒤에 배치하면 올바르게 동작한다.

### 7.3 Gaussian Blur Pass - 마스크 경계 스무딩 셰이더

마스크 경계의 하드 에지를 부드럽게 만들기 위해 **Separable Gaussian Blur** 2-pass를 적용한다. 기존 `shader_sources.cpp`의 `GAUSSIAN_BLUR_FRAGMENT`를 재활용하되, 마스크 전용으로 최적화한다.

#### 7.3.1 마스크 전용 Gaussian Blur 셰이더

기존 `GAUSSIAN_BLUR_FRAGMENT`는 RGB 3채널을 처리하지만, 마스크는 R 채널만 필요하다. 성능 최적화를 위해 단일 채널 전용 셰이더를 추가한다.

```glsl
// SKIN_MASK_BLUR_FRAGMENT (shader_sources.cpp에 추가)
#version 310 es
precision mediump float;

uniform sampler2D uTexture;
uniform vec2 uTexelSize;
uniform vec2 uDirection;  // (1,0) = horizontal, (0,1) = vertical
uniform float uBlurRadius;  // 3.0 ~ 8.0 (config.edge_blur_radius에 비례)

in vec2 vTexCoord;
out vec4 fragColor;

// 13-tap Gaussian weights (sigma ~= 3.0, 넓은 블러)
// 더 넓은 커널로 부드러운 경계 생성
const int KERNEL_SIZE = 7;
const float weights[7] = float[](
    0.1964826, 0.1748685, 0.1209854, 0.0651082, 0.0272290, 0.0088503, 0.0022345
);

void main() {
    float result = texture(uTexture, vTexCoord).r * weights[0];

    for (int i = 1; i < KERNEL_SIZE; i++) {
        vec2 offset = uDirection * uTexelSize * float(i) * (uBlurRadius / 5.0);
        result += texture(uTexture, vTexCoord + offset).r * weights[i];
        result += texture(uTexture, vTexCoord - offset).r * weights[i];
    }

    fragColor = vec4(result, result, result, 1.0);
}
```

#### 7.3.2 Gaussian Blur 실행 패턴

기존 `executeSoftFocusPass()` 패턴을 참조하여 2-pass separable blur를 구현한다.

```cpp
void SkinMaskRenderer::executeMaskBlur(int width, int height) {
#if IRIS_SDK_GPU_AVAILABLE
    // Separable Gaussian Blur: Horizontal → Vertical
    // 중간 결과를 저장할 임시 텍스처/FBO 필요

    // === Pass 1: Horizontal Blur ===
    // mask_texture_ → blur_temp_fbo_ (임시 FBO에 수평 블러 결과 저장)
    glBindFramebuffer(GL_FRAMEBUFFER, blur_temp_fbo_);
    glViewport(0, 0, width, height);
    glUseProgram(mask_blur_program_);

    glUniform1i(blur_uniforms_.uTexture, 0);
    glUniform2f(blur_uniforms_.uTexelSize, 1.0f / width, 1.0f / height);
    glUniform2f(blur_uniforms_.uDirection, 1.0f, 0.0f);  // Horizontal
    glUniform1f(blur_uniforms_.uBlurRadius, config_.edge_blur_radius);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, mask_texture_);

    renderMaskQuad();  // fullscreen quad

    // === Pass 2: Vertical Blur ===
    // blur_temp_texture_ → mask_fbo_ (최종 마스크에 수직 블러 결과 저장)
    glBindFramebuffer(GL_FRAMEBUFFER, mask_fbo_);
    glViewport(0, 0, width, height);

    glUniform2f(blur_uniforms_.uDirection, 0.0f, 1.0f);  // Vertical

    glBindTexture(GL_TEXTURE_2D, blur_temp_texture_);

    renderMaskQuad();

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
#endif
}
```

#### 7.3.3 Blur용 Fullscreen Quad

SkinMaskRenderer는 GPUBeautyBackend의 `quad_vao_`와 별도로 자체 quad를 관리한다. GPUBeautyBackend의 패턴(line 241-281)을 그대로 따른다.

```cpp
void SkinMaskRenderer::setupMaskQuad() {
#if IRIS_SDK_GPU_AVAILABLE
    float quad_vertices[] = {
        // Position    // TexCoord
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };

    glGenVertexArrays(1, &blur_quad_vao_);
    glGenBuffers(1, &blur_quad_vbo_);

    glBindVertexArray(blur_quad_vao_);
    glBindBuffer(GL_ARRAY_BUFFER, blur_quad_vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad_vertices), quad_vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                          (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
#endif
}

void SkinMaskRenderer::renderMaskQuad() {
#if IRIS_SDK_GPU_AVAILABLE
    glBindVertexArray(blur_quad_vao_);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
#endif
}
```

### 7.4 FBO 해상도 전략

마스크 텍스처는 원본 카메라 해상도와 동일할 필요가 없다. **1/4 해상도(가로/세로 각 1/2)**를 권장한다.

#### 7.4.1 해상도별 비교 분석

| 해상도 | 카메라 1080x1920 기준 | 텍스처 크기 | 장점 | 단점 |
|--------|----------------------|-------------|------|------|
| Full (1x) | 1080x1920 | 8.3MB | 정밀한 경계 | 메모리 낭비, blur 비용 높음 |
| **1/2 (권장)** | **540x960** | **2.1MB** | **경계 품질 충분, 성능 양호** | **약간의 정밀도 손실** |
| 1/4 | 270x480 | 0.5MB | 최소 메모리/성능 | 경계 뭉개짐 가능 |

#### 7.4.2 해상도 전략 결정 로직

```cpp
// 마스크 해상도 계산 (카메라 해상도의 1/2)
static constexpr float MASK_RESOLUTION_SCALE = 0.5f;

int SkinMaskRenderer::calculateMaskWidth(int camera_width) const {
    int w = static_cast<int>(camera_width * MASK_RESOLUTION_SCALE);
    // 4의 배수로 정렬 (GPU 텍스처 효율)
    return (w + 3) & ~3;
}

int SkinMaskRenderer::calculateMaskHeight(int camera_height) const {
    int h = static_cast<int>(camera_height * MASK_RESOLUTION_SCALE);
    return (h + 3) & ~3;
}
```

#### 7.4.3 FBO/텍스처 생성 및 재사용

카메라 해상도가 변경될 때만 FBO/텍스처를 재생성한다. `texture_pool.cpp` line 315-346 패턴을 따른다.

```cpp
bool SkinMaskRenderer::ensureMaskFBO(int camera_width, int camera_height) {
#if IRIS_SDK_GPU_AVAILABLE
    int mask_w = calculateMaskWidth(camera_width);
    int mask_h = calculateMaskHeight(camera_height);

    // 해상도 변경 없으면 기존 FBO 재사용
    if (mask_w == current_width_ && mask_h == current_height_
        && mask_fbo_ != 0) {
        return true;
    }

    // 기존 리소스 해제
    releaseMaskFBO();

    // === 마스크 텍스처 생성 ===
    glGenTextures(1, &mask_texture_);
    glBindTexture(GL_TEXTURE_2D, mask_texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    // R8 포맷: 단일 채널, 메모리 절약 (RGBA의 1/4)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, mask_w, mask_h, 0,
                 GL_RED, GL_UNSIGNED_BYTE, nullptr);

    // === 마스크 FBO 생성 ===
    glGenFramebuffers(1, &mask_fbo_);
    glBindFramebuffer(GL_FRAMEBUFFER, mask_fbo_);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, mask_texture_, 0);

    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        LOGE("SkinMask FBO incomplete: 0x%x", status);
        releaseMaskFBO();
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        return false;
    }

    // === Blur 임시 텍스처/FBO 생성 (Separable blur 중간 결과) ===
    glGenTextures(1, &blur_temp_texture_);
    glBindTexture(GL_TEXTURE_2D, blur_temp_texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, mask_w, mask_h, 0,
                 GL_RED, GL_UNSIGNED_BYTE, nullptr);

    glGenFramebuffers(1, &blur_temp_fbo_);
    glBindFramebuffer(GL_FRAMEBUFFER, blur_temp_fbo_);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, blur_temp_texture_, 0);

    status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        LOGE("SkinMask blur temp FBO incomplete: 0x%x", status);
        releaseMaskFBO();
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        return false;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    current_width_ = mask_w;
    current_height_ = mask_h;

    LOGI("SkinMask FBO created: mask=%u/%u, blur_temp=%u/%u (%dx%d)",
         mask_texture_, mask_fbo_, blur_temp_texture_, blur_temp_fbo_,
         mask_w, mask_h);
    return true;
#else
    return false;
#endif
}
```

#### 7.4.4 GL_R8 포맷 사용 근거

| 포맷 | 바이트/픽셀 | 540x960 크기 | 비고 |
|------|------------|-------------|------|
| GL_RGBA | 4 | 2.07MB | 과다 (마스크는 단일 채널) |
| **GL_R8** | **1** | **0.52MB** | **권장 - 마스크에 최적** |
| GL_R16F | 2 | 1.04MB | 부동소수점 불필요 |

마스크 FBO 2개(마스크 + blur 임시) 합계: 약 **1.04MB** (GL_R8, 540x960 기준).

### 7.5 renderSkinMask() 전체 파이프라인

```cpp
uint32_t SkinMaskRenderer::renderSkinMask(
    const IrisLandmark* face_mesh,
    int camera_width, int camera_height) {
#if IRIS_SDK_GPU_AVAILABLE
    // 1. FBO 확보 (해상도 변경 시 재생성)
    if (!ensureMaskFBO(camera_width, camera_height)) {
        return 0;
    }

    // 2. 정점/인덱스 데이터 빌드
    std::vector<float> vertices;
    std::vector<uint16_t> indices;

    // Face Oval (흰색 = 1.0)
    buildTriangleFan(face_mesh, vertices, indices);

    // 제외 영역 (검은색 = 0.0) - 기존 vertices/indices에 append
    std::vector<float> excl_vertices;
    std::vector<uint16_t> excl_indices;
    buildExclusionPolygons(face_mesh, excl_vertices, excl_indices);

    // 인덱스 오프셋 조정 후 병합
    uint16_t offset = static_cast<uint16_t>(vertices.size() / 3);
    for (auto& idx : excl_indices) {
        idx += offset;
    }
    vertices.insert(vertices.end(), excl_vertices.begin(), excl_vertices.end());
    indices.insert(indices.end(), excl_indices.begin(), excl_indices.end());

    // 3. VBO/EBO 업데이트 (매 프레임, GL_DYNAMIC_DRAW)
    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER,
                 vertices.size() * sizeof(float),
                 vertices.data(), GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 indices.size() * sizeof(uint16_t),
                 indices.data(), GL_DYNAMIC_DRAW);

    // 4. 마스크 FBO에 렌더링
    glBindFramebuffer(GL_FRAMEBUFFER, mask_fbo_);
    glViewport(0, 0, current_width_, current_height_);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);  // 기본 = 검은색 (피부 아님)
    glClear(GL_COLOR_BUFFER_BIT);

    glDisable(GL_DEPTH_TEST);   // 2D 마스크, 깊이 불필요
    glDisable(GL_BLEND);        // 블렌딩 불필요 (덮어쓰기)
    glUseProgram(mask_program_);

    // position attribute (location = 0): stride 3 floats
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE,
                          3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // mask_value attribute (location = 1): stride 3 floats, offset 2 floats
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE,
                          3 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glDrawElements(GL_TRIANGLES,
                   static_cast<GLsizei>(indices.size()),
                   GL_UNSIGNED_SHORT, 0);

    glBindVertexArray(0);

    // 5. Gaussian blur로 경계 스무딩
    if (config_.edge_blur_radius > 0.1f) {
        executeMaskBlur(current_width_, current_height_);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return mask_texture_;
#else
    return 0;
#endif
}
```

### 7.6 업데이트된 헤더 (skin_mask_renderer.h)

섹션 4.5의 인터페이스를 확장한 최종 헤더:

```cpp
// cpp/include/iris_sdk/gpu/skin_mask_renderer.h

#pragma once

#include "iris_sdk/types.h"
#include <vector>
#include <cstdint>

namespace iris_sdk {

class SkinMaskRenderer {
public:
    SkinMaskRenderer();
    ~SkinMaskRenderer();

    // 복사/이동 금지 (GPU 리소스 소유)
    SkinMaskRenderer(const SkinMaskRenderer&) = delete;
    SkinMaskRenderer& operator=(const SkinMaskRenderer&) = delete;

    /// 초기화 (셰이더 컴파일, VAO/VBO/EBO 생성, blur quad 설정)
    bool initialize();

    /// 피부 마스크 텍스처 생성/업데이트
    /// @param face_mesh 478개 랜드마크 (정규화 좌표 0-1)
    /// @param camera_width 카메라 원본 너비
    /// @param camera_height 카메라 원본 높이
    /// @return 마스크 텍스처 ID (GLuint), 실패 시 0
    uint32_t renderSkinMask(
        const IrisLandmark* face_mesh,
        int camera_width, int camera_height);

    /// 리소스 해제
    void release();

    /// 마스크 영역 설정
    struct MaskConfig {
        bool exclude_eyes = true;
        bool exclude_eyebrows = false;
        bool exclude_lips = true;
        float edge_blur_radius = 5.0f;   ///< Gaussian blur 반경 (px)
        float resolution_scale = 0.5f;   ///< 마스크 해상도 비율 (0.25~1.0)
    };

    void setMaskConfig(const MaskConfig& config);
    const MaskConfig& getMaskConfig() const { return config_; }

private:
    // Face Oval 삼각형 팬 생성 (mask_value = 1.0)
    void buildTriangleFan(const IrisLandmark* face_mesh,
                          std::vector<float>& vertices,
                          std::vector<uint16_t>& indices);

    // 제외 영역 (눈/입술) 다각형 생성 (mask_value = 0.0)
    void buildExclusionPolygons(const IrisLandmark* face_mesh,
                                std::vector<float>& vertices,
                                std::vector<uint16_t>& indices);

    // 공통 triangle fan 빌더
    void appendTriangleFan(const IrisLandmark* face_mesh,
                           const int* polygon_indices, int count,
                           float mask_value,
                           std::vector<float>& vertices,
                           std::vector<uint16_t>& indices);

    // Gaussian blur 실행 (2-pass separable)
    void executeMaskBlur(int width, int height);

    // FBO/텍스처 관리
    bool ensureMaskFBO(int camera_width, int camera_height);
    void releaseMaskFBO();

    // 해상도 계산
    int calculateMaskWidth(int camera_width) const;
    int calculateMaskHeight(int camera_height) const;

    // Blur quad 관련
    void setupMaskQuad();
    void renderMaskQuad();

    // --- GPU 리소스 ---
    // 마스크 렌더링
    uint32_t mask_fbo_ = 0;
    uint32_t mask_texture_ = 0;
    uint32_t mask_program_ = 0;   // skin_mask vert+frag
    uint32_t vao_ = 0;
    uint32_t vbo_ = 0;
    uint32_t ebo_ = 0;            // Element Buffer Object

    // Blur pass
    uint32_t blur_temp_fbo_ = 0;
    uint32_t blur_temp_texture_ = 0;
    uint32_t mask_blur_program_ = 0;  // blur vert+frag
    uint32_t blur_quad_vao_ = 0;
    uint32_t blur_quad_vbo_ = 0;

    // Blur Uniform Locations
    struct BlurUniforms {
        int32_t uTexture = -1;
        int32_t uTexelSize = -1;
        int32_t uDirection = -1;
        int32_t uBlurRadius = -1;
    } blur_uniforms_;

    MaskConfig config_;
    int current_width_ = 0;
    int current_height_ = 0;
    bool initialized_ = false;
};

} // namespace iris_sdk
```

### 7.7 메모리 및 성능 예산

#### 메모리 예산 (카메라 1080x1920, scale=0.5)

| 리소스 | 크기 | 비고 |
|--------|------|------|
| mask_texture_ (GL_R8, 540x960) | 0.52MB | 최종 마스크 |
| blur_temp_texture_ (GL_R8, 540x960) | 0.52MB | Blur 중간 결과 |
| VBO (94 vertices * 3 floats * 4B) | 1.1KB | 매 프레임 갱신 |
| EBO (270 indices * 2B) | 0.5KB | 매 프레임 갱신 |
| Blur quad VBO (6 verts * 4 floats * 4B) | 96B | 고정 |
| **합계** | **~1.05MB** | |

#### 성능 예산 (대상: Snapdragon 855 기준)

| 단계 | 예상 시간 | 비고 |
|------|-----------|------|
| buildTriangleFan + buildExclusion (CPU) | <0.1ms | 단순 산술 |
| VBO/EBO glBufferData (GPU upload) | <0.1ms | 1.6KB 데이터 |
| mask FBO 렌더링 (90 triangles) | <0.2ms | 단순 셰이더, 1/2 해상도 |
| Gaussian blur H-pass | <0.3ms | 13-tap, 1/2 해상도 |
| Gaussian blur V-pass | <0.3ms | 13-tap, 1/2 해상도 |
| **합계** | **<1.0ms** | **프레임 예산 33ms의 3%** |

### 7.8 마스크 셰이더 버전 호환성

현재 코드베이스는 `#version 310 es`를 사용한다 (`shader_sources.cpp` 전체). 마스크 셰이더도 동일하게 310 es를 사용한다. 섹션 4.4의 초안에서는 `#version 300 es`로 작성되었으나, 일관성을 위해 **310 es로 통일**한다.

```glsl
// 수정된 skin_mask.vert
#version 310 es
precision mediump float;

layout(location = 0) in vec2 a_position;
layout(location = 1) in float a_mask_value;

out float v_mask_value;

void main() {
    vec2 ndc = a_position * 2.0 - 1.0;
    ndc.y = -ndc.y;
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_mask_value = a_mask_value;
}

// 수정된 skin_mask.frag
#version 310 es
precision mediump float;

in float v_mask_value;
out vec4 fragColor;

void main() {
    fragColor = vec4(v_mask_value, 0.0, 0.0, 1.0);  // R 채널만 사용
}
```

---

## 변경 이력

| 날짜 | 내용 |
|------|------|
| 2026-02-09 | 초안 작성 - XNNPACK/GPU warm-up/V2 전용/피부 마스크 구현 스펙 |
| 2026-02-09 | 섹션 7 추가 - SkinMaskRenderer C++ 구현 상세 가이드 (EBO, exclusion, blur, FBO 전략) |
| 2026-02-11 | 섹션 1 구현 완료 - XNNPACK Delegate 활성화 (mediapipe_detector.cpp) |
| 2026-02-11 | 섹션 3 구현 완료 - GPU Delegate Warm-up (mediapipe_detector.cpp) |
