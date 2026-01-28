# 010. 뷰티 필터 확장 및 GPU 렌더링 최적화

## 작업 정보

| 항목 | 내용 |
|------|------|
| **작업 ID** | P1-W7-01 ~ P1-W12 |
| **작업명** | 뷰티 필터 확장 및 GPU 렌더링 최적화 |
| **상태** | 📋 계획 완료 |
| **시작일** | 2026-01-27 |
| **예상 기간** | 7주 |
| **의존성** | 008_direct_frame_rendering, 009_performance_optimization |

---

## 1. 개요

### 1.1 목표

IrisLensSDK의 뷰티 필터 기능 확장 및 Android GPU 렌더링 최적화를 통해:
- 고급 뷰티 필터 효과 추가 (피부톤 보정, 얼굴 형태 보정)
- GPU 가속으로 30fps 이상 성능 달성
- Face Mesh 기반 ROI 처리로 효율성 향상

### 1.2 작업 범위

1. **뷰티 필터 확장**
   - 기존 필터 개선 (성능 최적화, 파라미터 튜닝)
   - 새 필터 효과 추가 (피부톤, V-라인, 눈 확대 등)
   - ROI 기반 처리 (얼굴 영역만 필터링)

2. **GPU 렌더링 최적화**
   - OpenGL ES 3.1 기반 (Android)
   - 셰이더 기반 이미지 처리
   - CPU 폴백 지원

### 1.3 현재 상태

| 항목 | 상태 |
|------|------|
| 기본 뷰티 필터 | ✅ 구현됨 (Bilateral Filter + Soft Glow) |
| Face Mesh 데이터 | ✅ 478개 랜드마크 사용 가능 |
| GPU 가속 | ❌ 미구현 (100% CPU 기반) |
| ROI 처리 | ❌ 미구현 (전체 프레임 처리) |

---

## 2. 아키텍처 설계

### 2.1 현재 아키텍처

```
BeautyFilterConfig
├── enabled: bool
├── intensity: float (0.0~1.0)
├── smoothing: float (0.0~1.0)
├── brightness: float (0.0~2.0)
└── softFocus: float (0.0~1.0)
        │
        ▼
BeautyFilter (CPU Only)
├── Bilateral Filter (diameter=5)
├── Gaussian Blur
└── Brightness Adjustment
        │
        ▼
전체 프레임 처리 (비효율적)
```

### 2.2 목표 아키텍처 (피드백 반영 수정)

> **핵심 변경**: 통합 RenderContext + Zero-Copy 파이프라인 + 플랫폼 추상화

```
┌─────────────────────────────────────────────────────────────────┐
│                    BeautyFilterConfigV2 (확장)                   │
├─────────────────────────────────────────────────────────────────┤
│  기본 설정           │  피부 효과          │  얼굴 형태           │
│  • enabled          │  • smoothing       │  • slimFace         │
│  • intensity        │  • whitening       │  • enlargeEyes      │
│  • softFocus        │  • colorBalance    │  • thinChin         │
│  • brightness       │  • wrinkleRemove   │                     │
├─────────────────────┴──────────────────────┴─────────────────────┤
│  처리 옵션                                                       │
│  • useGpu: bool (GPU 가속)                                       │
│  • roiOnly: bool (얼굴 영역만)                                    │
│  • protectEyes: bool (눈 보호)                                   │
│  • protectLips: bool (입술 보호)                                 │
│  • downscaleFactor: int (저사양 기기용 다운스케일, 기본 1)         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              RenderContext (NEW - 통합 렌더링 컨텍스트)           │
├─────────────────────────────────────────────────────────────────┤
│  • Platform abstraction (OpenGL ES / Metal / CPU)               │
│  • TextureHandle (추상화된 텍스처 핸들, not GLuint)               │
│  • Zero-Copy 파이프라인 지원                                     │
│  • DI(Dependency Injection) 기반 설계                           │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  IRenderBackend (플랫폼 추상화 인터페이스)                    ││
│  │       ↑                    ↑                    ↑           ││
│  │  ┌────┴────┐        ┌─────┴─────┐       ┌──────┴──────┐    ││
│  │  │ GLES31  │        │  Metal    │       │    CPU      │    ││
│  │  │ Backend │        │  Backend  │       │  Backend    │    ││
│  │  │(Android)│        │   (iOS)   │       │ (Fallback)  │    ││
│  │  └─────────┘        └───────────┘       └─────────────┘    ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BeautyProcessor (NEW)                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              BeautyROIManager (NEW)                      │   │
│  │  • Face Mesh 478 랜드마크 → ROI 계산                     │   │
│  │  • 피부 마스크 생성 (삼각형 메쉬 기반)                     │   │
│  │  • 눈/입술 보호 마스크 + Soft Feathering                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           IBeautyBackend (Strategy Interface)            │   │
│  │  • apply(TextureHandle input, TextureHandle output)      │   │
│  │  • RenderContext 의존성 주입                              │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LensRenderer (기존)                         │
├─────────────────────────────────────────────────────────────────┤
│  • RenderContext 공유 (Zero-Copy)                               │
│  • BeautyProcessor 출력 텍스처 → LensRenderer 입력              │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Zero-Copy 렌더링 파이프라인

```
Camera Frame
     │
     ▼
RenderContext.uploadTexture() → TextureHandle (A)
     │
     ▼
BeautyProcessor.process(A) → TextureHandle (B)  [GPU 내부 처리]
     │
     ▼
LensRenderer.render(B) → TextureHandle (C)  [Texture 직접 전달]
     │
     ▼
RenderContext.downloadTexture(C) 또는 직접 표시
```

**핵심**: GPU 텍스처 ID가 프레임 전체에서 재사용되어 Context Switching 및 Texture Copy 비용 제거

### 2.3 Face Mesh 랜드마크 활용

MediaPipe Face Mesh 478개 랜드마크 중 주요 영역:

| 영역 | 인덱스 | 용도 |
|------|--------|------|
| **얼굴 윤곽** | 10, 338, 297, 332, 284, 251... (36개) | 얼굴 ROI 마스킹 |
| **왼쪽 눈** | 33, 7, 163, 144, 145, 153... (16개) | 눈 보호 영역 |
| **오른쪽 눈** | 362, 382, 381, 380, 374... (16개) | 눈 보호 영역 |
| **입술** | 61, 146, 91, 181, 84, 17... (22개) | 입술 보호 영역 |
| **볼** | 227, 137, 116, 117... (24개) | 스무딩 집중 영역 |
| **턱** | 152, 148, 176, 149... (13개) | V-라인 보정용 |

기존 정의된 인덱스 (mediapipe_detector.cpp:78-86):
```cpp
constexpr int LEFT_EYE_INDICES[] = {
    33, 7, 163, 144, 145, 153, 154, 155, 133,  // 눈 윤곽
    173, 157, 158, 159, 160, 161, 246          // 눈꺼풀
};
constexpr int RIGHT_EYE_INDICES[] = {
    362, 382, 381, 380, 374, 373, 390, 249, 263,
    466, 388, 387, 386, 385, 384, 398
};
```

---

## 3. 상세 구현 계획

### Phase 1: 기반 구조 리팩토링 (1주)

#### Task 1.1: BeautyFilterConfigV2 정의

**파일**: `cpp/include/iris_sdk/beauty_filter.h`

```cpp
struct BeautyFilterConfigV2 {
    // ===== 기본 설정 (V1 호환) =====
    bool enabled = false;
    float intensity = 0.5f;      // 전체 강도 (0.0~1.0)

    // ===== 피부 효과 =====
    float smoothing = 0.5f;      // 피부 스무딩 (기존)
    float brightness = 1.0f;     // 밝기 (기존, 0.5~1.5)
    float softFocus = 0.3f;      // 소프트 포커스 (기존)
    float whitening = 0.0f;      // 피부톤 화이트닝 (NEW, 0.0~1.0)
    float colorBalance = 0.0f;   // 컬러 밸런스 (NEW, -1.0~1.0)
    float wrinkleRemove = 0.0f;  // 주름 제거 (NEW, 0.0~1.0)

    // ===== 얼굴 형태 보정 =====
    float slimFace = 0.0f;       // 얼굴 슬림화 (NEW, 0.0~1.0)
    float enlargeEyes = 0.0f;    // 눈 확대 (NEW, 0.0~1.0)
    float thinChin = 0.0f;       // 턱 축소 (NEW, 0.0~1.0)

    // ===== 처리 옵션 =====
    bool useGpu = true;          // GPU 가속 사용
    bool roiOnly = true;         // 얼굴 영역만 처리
    bool protectEyes = true;     // 눈 영역 보호
    bool protectLips = true;     // 입술 영역 보호
    int downscaleFactor = 1;     // 다운스케일 (1=원본, 2=1/2, 4=1/4)
};
```

**체크리스트**:
- [ ] 구조체 정의
- [ ] 기본값 설정
- [ ] C API 래퍼 (`iris_sdk_default_beauty_config_v2`)
- [ ] JNI 바인딩 (`BeautyFilterConfigV2.java`)

#### Task 1.2: BeautyROIManager 구현

**새 파일**: `cpp/include/iris_sdk/beauty_roi_manager.h`, `cpp/src/beauty_roi_manager.cpp`

```cpp
namespace iris_sdk {

struct BeautyROI {
    Rect face_rect;                         // 얼굴 바운딩 박스
    std::vector<uint8_t> skin_mask;         // 피부 마스크 (255=적용)
    std::vector<uint8_t> eye_protect_mask;  // 눈 보호 마스크
    std::vector<uint8_t> lip_protect_mask;  // 입술 보호 마스크
    int mask_width, mask_height;
    bool valid = false;
};

class BeautyROIManager {
public:
    // Face Mesh → ROI 계산
    static bool computeROI(
        const IrisLandmark* face_mesh,
        int frame_width, int frame_height,
        const BeautyFilterConfigV2& config,
        BeautyROI& out_roi
    );

    // 피부 영역 마스크 생성
    static void createSkinMask(
        const IrisLandmark* face_mesh,
        int mask_width, int mask_height,
        std::vector<uint8_t>& out_mask
    );

    // 보호 영역 마스크 생성
    static void createProtectionMask(
        const IrisLandmark* face_mesh,
        int mask_width, int mask_height,
        bool include_eyes, bool include_lips,
        std::vector<uint8_t>& out_mask
    );

private:
    // 랜드마크 인덱스 정의
    static constexpr int FACE_OVAL_INDICES[];
    static constexpr int LEFT_EYE_INDICES[];
    static constexpr int RIGHT_EYE_INDICES[];
    static constexpr int LIPS_INDICES[];
};

} // namespace iris_sdk
```

**체크리스트**:
- [ ] 클래스 구조 정의
- [ ] 삼각형 메쉬 기반 마스크 생성 알고리즘
- [ ] 랜드마크 인덱스 테이블
- [ ] 단위 테스트

#### Task 1.3: IBeautyBackend 인터페이스

**새 파일**: `cpp/include/iris_sdk/beauty_backend.h`

```cpp
namespace iris_sdk {

class IBeautyBackend {
public:
    virtual ~IBeautyBackend() = default;

    // 라이프사이클
    virtual bool initialize() = 0;
    virtual void release() = 0;
    virtual bool isInitialized() const = 0;

    // 필터 적용
    virtual IrisSdkError apply(
        uint8_t* frame_data,
        int width, int height,
        IrisFrameFormat format,
        const BeautyFilterConfigV2& config,
        const BeautyROI* roi = nullptr
    ) = 0;

    // 메타데이터
    virtual const char* getName() const = 0;
    virtual bool supportsGpu() const = 0;
};

} // namespace iris_sdk
```

**체크리스트**:
- [ ] 인터페이스 정의
- [ ] CPUBeautyBackend 리팩토링 (기존 BeautyFilter 기반)
- [ ] BeautyProcessor 통합 클래스

---

### Phase 2: CPU 백엔드 개선 (1주)

#### Task 2.1: ROI 기반 처리

**파일**: `cpp/src/cpu_beauty_backend.cpp`

```cpp
IrisSdkError CPUBeautyBackend::apply(..., const BeautyROI* roi) {
    if (roi != nullptr && roi->valid) {
        // ROI 영역만 추출하여 처리
        cv::Rect face_rect(roi->face_rect.x, roi->face_rect.y,
                          roi->face_rect.width, roi->face_rect.height);
        cv::Mat roi_frame = frame(face_rect);

        // 마스크 적용하여 필터링
        applyWithMask(roi_frame, roi->skin_mask, config);

        // 결과를 원본 프레임에 복사
        roi_frame.copyTo(frame(face_rect));
    } else {
        // 전체 프레임 처리 (폴백)
        applyFullFrame(frame, config);
    }
}
```

**예상 성능 개선**: 30-50% (얼굴 영역이 전체의 20-30% 정도)

**ROI 경계 페더링 (Soft Blend)**:

> **중요**: 처리된 영역과 원본 영역의 경계가 뚜렷하게 보이지 않도록 반드시 Soft Feathering 적용

```cpp
void applyFeatheredBlend(const cv::Mat& processed, const cv::Mat& original,
                         const cv::Mat& mask, cv::Mat& output, int feather_radius) {
    // 마스크 가장자리 블러로 Soft Blend 영역 생성
    cv::Mat feathered_mask;
    cv::GaussianBlur(mask, feathered_mask,
                     cv::Size(feather_radius*2+1, feather_radius*2+1), 0);
    feathered_mask.convertTo(feathered_mask, CV_32F, 1.0/255.0);

    // Alpha Blending
    cv::Mat proc_f, orig_f;
    processed.convertTo(proc_f, CV_32F);
    original.convertTo(orig_f, CV_32F);

    std::vector<cv::Mat> channels(3);
    for (int c = 0; c < 3; c++) {
        channels[c] = proc_f.mul(feathered_mask) +
                      orig_f.mul(1.0 - feathered_mask);
    }
    cv::merge(channels, output);
    output.convertTo(output, CV_8U);
}
```

**체크리스트**:
- [ ] ROI 추출 로직
- [ ] 마스크 기반 필터링
- [ ] 경계 처리 (Soft Feathering, feather_radius=10~20)
- [ ] 성능 벤치마크

#### Task 2.2: Bilateral Filter 최적화

현재 문제: Bilateral Filter O(d²) 복잡도

**최적화 방안**:

1. **Fast Guided Filter 직접 구현** (opencv-contrib 의존성 제거)

   > **주의**: `cv::ximgproc::guidedFilter`는 opencv-contrib에 포함되어 있어 AAR 용량이 크게 증가함.
   > 대신 O(1) 복잡도의 Fast Guided Filter를 직접 구현.

   ```cpp
   // Fast Guided Filter 직접 구현 (Box Filter 기반)
   // 참고: "Fast Guided Filter" by Kaiming He (2015)
   class FastGuidedFilter {
   public:
       static void apply(const cv::Mat& I, const cv::Mat& p,
                        cv::Mat& output, int r, float eps) {
           // 1. Box Filter로 mean_I, mean_p, corr_Ip, var_I 계산
           cv::Mat mean_I, mean_p, corr_Ip, var_I;
           cv::boxFilter(I, mean_I, CV_32F, cv::Size(r, r));
           cv::boxFilter(p, mean_p, CV_32F, cv::Size(r, r));
           cv::boxFilter(I.mul(p), corr_Ip, CV_32F, cv::Size(r, r));
           cv::boxFilter(I.mul(I), var_I, CV_32F, cv::Size(r, r));

           // 2. a, b 계산
           cv::Mat cov_Ip = corr_Ip - mean_I.mul(mean_p);
           var_I = var_I - mean_I.mul(mean_I);
           cv::Mat a = cov_Ip / (var_I + eps);
           cv::Mat b = mean_p - a.mul(mean_I);

           // 3. 최종 출력
           cv::Mat mean_a, mean_b;
           cv::boxFilter(a, mean_a, CV_32F, cv::Size(r, r));
           cv::boxFilter(b, mean_b, CV_32F, cv::Size(r, r));
           output = mean_a.mul(I) + mean_b;
       }
   };
   ```

2. **다운샘플링 기법** (저사양 기기용)
   ```cpp
   cv::Mat small;
   cv::resize(input, small, cv::Size(), 0.5, 0.5);
   cv::bilateralFilter(small, filtered_small, d, sigmaColor, sigmaSpace);
   cv::resize(filtered_small, output, input.size());
   ```

**체크리스트**:
- [ ] Fast Guided Filter 직접 구현 (opencv-contrib 의존성 제거)
- [ ] 다운샘플링 비율 튜닝
- [ ] 품질 대비 성능 테스트
- [ ] AAR 용량 확인 (20MB 이하 목표)

#### Task 2.3: 새 필터 효과 (CPU)

**피부톤 보정 (Whitening)**:
```cpp
void applySkinWhitening(cv::Mat& frame, float strength) {
    cv::Mat lab;
    cv::cvtColor(frame, lab, cv::COLOR_BGR2Lab);

    std::vector<cv::Mat> channels;
    cv::split(lab, channels);

    // L 채널 증가 (밝기)
    channels[0] += strength * 15;
    // a 채널 감소 (붉은기 제거)
    channels[1] -= strength * 5;
    // b 채널 감소 (노란기 제거)
    channels[2] -= strength * 5;

    cv::merge(channels, lab);
    cv::cvtColor(lab, frame, cv::COLOR_Lab2BGR);
}
```

**Targeted Smoothing (주름 제거)**:
- 고주파 성분 감지 (Laplacian)
- 고주파 영역에만 선택적 블러

**체크리스트**:
- [ ] 피부톤 보정 구현
- [ ] 주름 제거 구현
- [ ] 파라미터 튜닝

---

### Phase 3: GPU 백엔드 구현 (2주)

#### Task 3.1: 렌더링 컨텍스트 인프라 (플랫폼 추상화)

> **피드백 반영**:
> - 싱글톤 대신 **DI(Dependency Injection)** 방식 채택 (테스트 용이성)
> - `GLuint` 직접 노출 대신 **TextureHandle 추상화** (크로스 플랫폼)
> - 파일명을 `gl_context.h` → `render_context.h`로 일반화

**새 파일**: `cpp/include/iris_sdk/gpu/render_context.h`

```cpp
namespace iris_sdk {

// 플랫폼 독립적 텍스처 핸들
struct TextureHandle {
    void* native_handle;  // GLuint* (GLES), MTLTexture* (Metal), cv::Mat* (CPU)
    int width, height;
    enum class Type { GLES, Metal, CPU } type;
};

// 플랫폼 추상화 인터페이스
class IRenderContext {
public:
    virtual ~IRenderContext() = default;

    virtual bool initialize() = 0;
    virtual void release() = 0;
    virtual bool isInitialized() const = 0;

    // 텍스처 관리
    virtual TextureHandle createTexture(int width, int height) = 0;
    virtual void deleteTexture(TextureHandle& handle) = 0;
    virtual void uploadTexture(TextureHandle& handle, const uint8_t* data,
                               int width, int height, int format) = 0;
    virtual void downloadTexture(const TextureHandle& handle, uint8_t* data) = 0;

    // 컨텍스트 전환
    virtual bool makeCurrent() = 0;
    virtual void doneCurrent() = 0;

    // 버전 정보
    virtual int getMajorVersion() const = 0;
    virtual int getMinorVersion() const = 0;
};

} // namespace iris_sdk
```

**GLES 구현**: `cpp/include/iris_sdk/gpu/gles_render_context.h`

```cpp
#if defined(__ANDROID__) && defined(IRIS_SDK_HAS_GLES)

class GLESRenderContext : public IRenderContext {
public:
    // DI를 위한 생성자 (싱글톤 아님)
    GLESRenderContext();
    ~GLESRenderContext() override;

    bool initialize() override;
    void release() override;
    bool isInitialized() const override;

    TextureHandle createTexture(int width, int height) override;
    void deleteTexture(TextureHandle& handle) override;
    void uploadTexture(TextureHandle& handle, const uint8_t* data,
                       int width, int height, int format) override;
    void downloadTexture(const TextureHandle& handle, uint8_t* data) override;

    bool makeCurrent() override;
    void doneCurrent() override;

    int getMajorVersion() const override;
    int getMinorVersion() const override;

private:
    EGLDisplay display_ = EGL_NO_DISPLAY;
    EGLContext context_ = EGL_NO_CONTEXT;
    EGLSurface surface_ = EGL_NO_SURFACE;

    std::atomic<bool> initialized_{false};
    std::mutex init_mutex_;
};

#endif
```

**CPU Fallback 구현**: `cpp/include/iris_sdk/gpu/cpu_render_context.h`

```cpp
class CPURenderContext : public IRenderContext {
    // cv::Mat 기반 구현 (Desktop, GPU 미지원 기기용)
};
```

**DI 활용 예시**:
```cpp
// 테스트 코드에서 Mock 주입 가능
class BeautyProcessor {
public:
    explicit BeautyProcessor(std::shared_ptr<IRenderContext> context)
        : render_context_(std::move(context)) {}

private:
    std::shared_ptr<IRenderContext> render_context_;
};

// Production
auto gles_ctx = std::make_shared<GLESRenderContext>();
auto processor = BeautyProcessor(gles_ctx);

// Test (Mock)
auto mock_ctx = std::make_shared<MockRenderContext>();
auto processor = BeautyProcessor(mock_ctx);
```

**체크리스트**:
- [ ] IRenderContext 인터페이스 정의
- [ ] TextureHandle 추상화
- [ ] GLESRenderContext 구현 (Android)
- [ ] CPURenderContext 구현 (Fallback)
- [ ] DI 기반 BeautyProcessor 생성자
- [ ] 단위 테스트 (Mock 주입)

**셰이더 관리**: `cpp/include/iris_sdk/gpu/shader_manager.h`

```cpp
class ShaderManager {
public:
    GLuint getProgram(const std::string& name);

    // 캐시에서 로드 (바이너리)
    bool loadFromCache(const std::string& name);

    // 소스에서 컴파일
    bool compileFromSource(const std::string& name,
                           const char* vertexSrc,
                           const char* fragmentSrc);

    // 캐시에 저장
    void saveToCache(const std::string& name);

private:
    std::map<std::string, GLuint> programs_;
    std::string cache_dir_;
};
```

**체크리스트**:
- [ ] EGL 컨텍스트 초기화
- [ ] 지연 초기화 패턴
- [ ] 셰이더 컴파일/캐싱
- [ ] 버전 폴백 (ES 3.1 → 3.0)

#### Task 3.2: 기본 필터 셰이더

**디렉토리**: `cpp/src/gpu/shaders/`

**gaussian_blur.frag** (분리 가능):
```glsl
#version 310 es
precision highp float;

in vec2 v_TexCoord;
out vec4 fragColor;

uniform sampler2D u_InputTexture;
uniform vec2 u_TextureSize;
uniform int u_KernelSize;
uniform float u_Weights[15];
uniform bool u_Horizontal;

void main() {
    vec2 texelSize = 1.0 / u_TextureSize;
    vec4 result = vec4(0.0);

    int halfSize = u_KernelSize / 2;
    for (int i = -halfSize; i <= halfSize; i++) {
        vec2 offset = u_Horizontal
            ? vec2(float(i) * texelSize.x, 0.0)
            : vec2(0.0, float(i) * texelSize.y);
        result += texture(u_InputTexture, v_TexCoord + offset)
                  * u_Weights[i + halfSize];
    }

    fragColor = result;
}
```

**brightness.frag**:
```glsl
#version 310 es
precision highp float;

in vec2 v_TexCoord;
out vec4 fragColor;

uniform sampler2D u_InputTexture;
uniform sampler2D u_MaskTexture;
uniform float u_Brightness;

void main() {
    vec4 color = texture(u_InputTexture, v_TexCoord);
    float mask = texture(u_MaskTexture, v_TexCoord).r;

    vec3 adjusted = color.rgb * u_Brightness;
    fragColor = vec4(mix(color.rgb, adjusted, mask), color.a);
}
```

**체크리스트**:
- [ ] Gaussian Blur (2-pass)
- [ ] Brightness/Contrast
- [ ] Soft Focus
- [ ] 셰이더 테스트

#### Task 3.3: 고급 필터 셰이더

**bilateral_filter.frag**:
```glsl
#version 310 es
precision highp float;

in vec2 v_TexCoord;
out vec4 fragColor;

uniform sampler2D u_InputTexture;
uniform sampler2D u_MaskTexture;
uniform vec2 u_TextureSize;
uniform float u_SigmaSpace;
uniform float u_SigmaColor;
uniform int u_Radius;

void main() {
    vec2 texelSize = 1.0 / u_TextureSize;
    vec4 centerColor = texture(u_InputTexture, v_TexCoord);
    float mask = texture(u_MaskTexture, v_TexCoord).r;

    if (mask < 0.01) {
        fragColor = centerColor;
        return;
    }

    vec4 sum = vec4(0.0);
    float weightSum = 0.0;

    for (int y = -u_Radius; y <= u_Radius; y++) {
        for (int x = -u_Radius; x <= u_Radius; x++) {
            vec2 offset = vec2(float(x), float(y)) * texelSize;
            vec4 sampleColor = texture(u_InputTexture, v_TexCoord + offset);

            // 공간 가중치
            float spatialWeight = exp(-float(x*x + y*y) /
                                     (2.0 * u_SigmaSpace * u_SigmaSpace));

            // 색상 가중치
            vec3 colorDiff = sampleColor.rgb - centerColor.rgb;
            float colorDist = dot(colorDiff, colorDiff);
            float colorWeight = exp(-colorDist /
                                   (2.0 * u_SigmaColor * u_SigmaColor));

            float weight = spatialWeight * colorWeight;
            sum += sampleColor * weight;
            weightSum += weight;
        }
    }

    vec4 filtered = sum / weightSum;
    fragColor = mix(centerColor, filtered, mask);
}
```

**skin_whitening.frag** (LAB 색상 공간):
```glsl
// RGB ↔ LAB 변환 함수 포함
// L 채널 증가, a/b 채널 감소로 화이트닝 효과
```

**체크리스트**:
- [ ] Bilateral Filter
- [ ] Skin Whitening
- [ ] Edge-aware Smoothing
- [ ] 마스크 텍스처 통합

#### Task 3.4: GPUBeautyBackend 통합

**새 파일**: `cpp/include/iris_sdk/gpu/gpu_beauty_backend.h`

```cpp
class GPUBeautyBackend : public IBeautyBackend {
public:
    bool initialize() override;
    void release() override;
    bool isInitialized() const override;

    IrisSdkError apply(
        uint8_t* frame_data,
        int width, int height,
        IrisFrameFormat format,
        const BeautyFilterConfigV2& config,
        const BeautyROI* roi = nullptr
    ) override;

    const char* getName() const override { return "GPUBeautyBackend"; }
    bool supportsGpu() const override { return true; }

private:
    // 텍스처 관리
    GLuint input_texture_ = 0;
    GLuint output_texture_ = 0;
    GLuint mask_texture_ = 0;
    GLuint framebuffer_ = 0;

    // 텍스처 풀
    std::unique_ptr<TexturePool> texture_pool_;
};
```

**체크리스트**:
- [ ] 텍스처 업로드/다운로드
- [ ] FBO 렌더링 파이프라인
- [ ] 텍스처 풀링
- [ ] PBO 비동기 전송 (선택적)

---

### Phase 4: 얼굴 형태 보정 (2주)

#### Task 4.1: Face Warp 엔진 (Grid Mesh 기반)

> **피드백 반영**: 478개 랜드마크만 사용하면 랜드마크 없는 영역(이마, 볼 외곽)에서
> 텍스처 왜곡이 부자연스러움. **Grid Mesh (10x10 or 20x20)** 기반 변형 채택.

**알고리즘**: Grid Mesh + 랜드마크 기반 변형

```cpp
class FaceWarpEngine {
public:
    // Grid Mesh 생성 (예: 20x20 = 400개 정점)
    void buildGridMesh(int grid_width, int grid_height,
                       int frame_width, int frame_height);

    // 랜드마크 움직임에 따라 Grid 정점 변형
    void deformGridByLandmarks(
        const IrisLandmark* face_mesh,
        const BeautyFilterConfigV2& config
    );

    // CPU 와핑 (cv::remap)
    void warpCPU(const cv::Mat& input, cv::Mat& output);

    // GPU 와핑 (Vertex Shader에서 변형)
    void warpGPU(TextureHandle input, TextureHandle output,
                 RenderContext* ctx);

private:
    struct GridVertex {
        float x, y;           // 원본 위치
        float warped_x, warped_y;  // 변형 후 위치
        float tex_u, tex_v;   // 텍스처 좌표
    };

    int grid_w_, grid_h_;
    std::vector<GridVertex> vertices_;
    std::vector<uint16_t> indices_;  // 삼각형 인덱스

    // 랜드마크 영향 범위 계산
    float computeLandmarkInfluence(float vx, float vy,
                                   float lx, float ly, float radius);
};
```

**Grid Mesh 장점**:
1. 모든 영역에 균일한 변형 적용 가능
2. GPU Vertex Shader에서 효율적 처리
3. 400개 정점만 업로드 (478개 랜드마크 vs 400개 Grid)

**구현 방식**:
```cpp
void FaceWarpEngine::deformGridByLandmarks(
    const IrisLandmark* face_mesh,
    const BeautyFilterConfigV2& config) {

    for (auto& vertex : vertices_) {
        float dx = 0, dy = 0;

        // 턱 영역 랜드마크의 영향
        if (config.thinChin > 0) {
            for (int idx : CHIN_INDICES) {
                float influence = computeLandmarkInfluence(
                    vertex.x, vertex.y,
                    face_mesh[idx].x, face_mesh[idx].y,
                    influence_radius_);

                // 턱 중심으로 당기기
                dx += influence * config.thinChin * (chin_center_x - vertex.x) * 0.1f;
                dy += influence * config.thinChin * (chin_center_y - vertex.y) * 0.1f;
            }
        }

        // 눈 확대 영향
        if (config.enlargeEyes > 0) {
            // 눈 중심에서 바깥으로 밀기
            // ...
        }

        vertex.warped_x = vertex.x + dx;
        vertex.warped_y = vertex.y + dy;
    }
}
```

**체크리스트**:
- [ ] Grid Mesh 생성 (20x20 권장)
- [ ] 랜드마크 영향 범위 함수
- [ ] Influence 가중치 튜닝
- [ ] CPU cv::remap 구현
- [ ] GPU Vertex Shader 변형
- [ ] 부드러운 경계 전이 검증

#### Task 4.2: V-라인 보정

턱 영역 랜드마크 (152, 148, 176, 149, 150, 136...) 기반:

```cpp
void computeChinWarpVector(
    const IrisLandmark* face_mesh,
    float thinChin,  // 0.0~1.0
    std::vector<WarpVector>& out_vectors
) {
    // 턱 중심점
    float chin_center_x = face_mesh[152].x;
    float chin_center_y = face_mesh[152].y;

    // 양쪽 턱 포인트를 중심으로 당기기
    for (int idx : CHIN_INDICES) {
        float dx = chin_center_x - face_mesh[idx].x;
        float dy = chin_center_y - face_mesh[idx].y;
        float dist = sqrt(dx*dx + dy*dy);

        // 거리에 반비례하는 변형 강도
        float strength = thinChin * (1.0f - dist / max_dist) * 0.1f;

        out_vectors.push_back({
            .index = idx,
            .dx = dx * strength,
            .dy = dy * strength
        });
    }
}
```

**체크리스트**:
- [ ] 턱 영역 인덱스 정의
- [ ] 변형 벡터 계산
- [ ] 자연스러운 경계 처리
- [ ] 강도 파라미터 튜닝

#### Task 4.3: 눈 확대

눈 영역 스케일 변환:

```cpp
void computeEyeEnlargeVector(
    const IrisLandmark* face_mesh,
    float enlargeEyes,  // 0.0~1.0
    std::vector<WarpVector>& out_vectors
) {
    // 왼쪽 눈 중심
    float left_center_x = 0, left_center_y = 0;
    for (int idx : LEFT_EYE_INDICES) {
        left_center_x += face_mesh[idx].x;
        left_center_y += face_mesh[idx].y;
    }
    left_center_x /= sizeof(LEFT_EYE_INDICES) / sizeof(int);
    left_center_y /= sizeof(LEFT_EYE_INDICES) / sizeof(int);

    // 중심에서 바깥으로 확대
    float scale = 1.0f + enlargeEyes * 0.15f;  // 최대 15% 확대

    for (int idx : LEFT_EYE_REGION_INDICES) {
        float dx = face_mesh[idx].x - left_center_x;
        float dy = face_mesh[idx].y - left_center_y;

        out_vectors.push_back({
            .index = idx,
            .dx = dx * (scale - 1.0f),
            .dy = dy * (scale - 1.0f)
        });
    }

    // 오른쪽 눈도 동일하게 처리
}
```

**체크리스트**:
- [ ] 눈 중심 계산
- [ ] 스케일 변환
- [ ] 부드러운 경계 전이
- [ ] 양쪽 눈 대칭 처리

---

### Phase 5: 통합 및 최적화 (1주)

#### Task 5.1: JNI 바인딩 확장

**파일**: `android/iris-sdk/src/main/cpp/iris_jni.cpp`

```cpp
extern "C" JNIEXPORT void JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeDefaultBeautyConfigV2(
    JNIEnv* env, jobject thiz, jobject config) {
    // Java 객체 → C 구조체
}

extern "C" JNIEXPORT jint JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeSetBeautyFilterV2(
    JNIEnv* env, jobject thiz, jobject config) {
    // V2 설정 적용
}

extern "C" JNIEXPORT jint JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeApplyBeautyFilterV2(
    JNIEnv* env, jobject thiz,
    jbyteArray frameData, jint width, jint height, jint format,
    jobject irisResult) {
    // Face Mesh 연동 필터 적용
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeBeautyGpuAvailable(
    JNIEnv* env, jobject thiz) {
    // GPU 지원 여부
}
```

**새 파일**: `android/iris-sdk/src/main/java/com/irislenssdk/BeautyFilterConfigV2.java`

```java
public class BeautyFilterConfigV2 {
    // 기본 설정
    public boolean enabled = false;
    public float intensity = 0.5f;

    // 피부 효과
    public float smoothing = 0.5f;
    public float brightness = 1.0f;
    public float softFocus = 0.3f;
    public float whitening = 0.0f;
    public float colorBalance = 0.0f;
    public float wrinkleRemove = 0.0f;

    // 얼굴 형태
    public float slimFace = 0.0f;
    public float enlargeEyes = 0.0f;
    public float thinChin = 0.0f;

    // 처리 옵션
    public boolean useGpu = true;
    public boolean roiOnly = true;
    public boolean protectEyes = true;
    public boolean protectLips = true;

    // Builder 패턴
    public static class Builder { ... }
}
```

**체크리스트**:
- [ ] C API → JNI 매핑
- [ ] Java/Kotlin 클래스
- [ ] 유효성 검증
- [ ] 예외 처리

#### Task 5.2: 성능 프로파일링

**측정 항목**:
| 항목 | CPU (현재) | CPU (최적화) | GPU | 목표 |
|------|------------|--------------|-----|------|
| ROI 계산 | N/A | < 1ms | < 0.5ms | ✓ |
| 피부 스무딩 | 25-30ms | 10-15ms | 2-3ms | < 10ms |
| 전체 필터 | 35-43ms | 25-35ms | 6-10ms | < 15ms |
| FPS | ~20fps | ~25fps | 30+ fps | 30fps |

**자동 전환 임계값**:
```cpp
// GPU 초기화 실패 또는 성능 저하 시 CPU 폴백
if (gpu_frame_time > cpu_frame_time * 1.5) {
    switchToCPU();
}
```

**체크리스트**:
- [ ] 프레임 타임 측정
- [ ] GPU/CPU 자동 전환
- [ ] 메모리 모니터링
- [ ] 다양한 기기 테스트

---

## 4. 파일 구조

```
cpp/
├── include/iris_sdk/
│   ├── beauty_filter.h              ← 확장 (V2 config 추가)
│   ├── beauty_processor.h           ← NEW: 통합 프로세서 (DI 기반)
│   ├── beauty_backend.h             ← NEW: 백엔드 인터페이스
│   ├── beauty_roi_manager.h         ← NEW: ROI 관리 + Feathering
│   ├── fast_guided_filter.h         ← NEW: opencv-contrib 대체 구현
│   ├── face_warp_engine.h           ← NEW: Grid Mesh 기반 와핑
│   └── gpu/
│       ├── render_context.h         ← NEW: 플랫폼 추상화 인터페이스
│       ├── texture_handle.h         ← NEW: 텍스처 핸들 추상화
│       ├── gles_render_context.h    ← NEW: OpenGL ES 구현
│       ├── cpu_render_context.h     ← NEW: CPU 폴백 구현
│       ├── shader_manager.h         ← NEW: 셰이더 관리
│       ├── texture_pool.h           ← NEW: 텍스처 풀
│       └── gpu_beauty_backend.h     ← NEW: GPU 백엔드
├── src/
│   ├── beauty_filter.cpp            ← 리팩토링 (레거시 API 유지)
│   ├── beauty_processor.cpp         ← NEW
│   ├── beauty_roi_manager.cpp       ← NEW
│   ├── fast_guided_filter.cpp       ← NEW: Box Filter 기반 구현
│   ├── face_warp_engine.cpp         ← NEW: Grid Mesh 변형
│   ├── cpu_beauty_backend.cpp       ← NEW
│   └── gpu/
│       ├── gles_render_context.cpp  ← NEW
│       ├── cpu_render_context.cpp   ← NEW
│       ├── shader_manager.cpp       ← NEW
│       ├── texture_pool.cpp         ← NEW
│       ├── gpu_beauty_backend.cpp   ← NEW
│       └── shaders/
│           ├── passthrough.vert     ← NEW: 공통 vertex shader
│           ├── warp_mesh.vert       ← NEW: Grid Mesh 변형용
│           ├── gaussian_blur.frag   ← NEW
│           ├── bilateral_filter.frag← NEW
│           ├── skin_whitening.frag  ← NEW
│           ├── brightness.frag      ← NEW
│           ├── soft_focus.frag      ← NEW
│           └── masking.frag         ← NEW
└── tests/
    ├── test_beauty_roi.cpp          ← NEW
    ├── test_fast_guided_filter.cpp  ← NEW
    ├── test_face_warp_engine.cpp    ← NEW
    ├── test_cpu_backend.cpp         ← NEW
    ├── test_gpu_backend.cpp         ← NEW (Android only)
    ├── test_beauty_processor.cpp    ← NEW
    └── mock_render_context.h        ← NEW: DI 테스트용 Mock

android/iris-sdk/
├── src/main/cpp/
│   └── iris_jni.cpp                 ← 확장 (V2 API 추가)
└── src/main/java/com/irislenssdk/
    ├── BeautyFilterConfig.java      ← 기존 (호환성 유지)
    ├── BeautyFilterConfigV2.java    ← NEW
    └── IrisLensSDK.java             ← 확장 (V2 메서드 추가)
```

---

## 5. C API

### 5.1 기존 API (호환성 유지)

```cpp
// V1 API - 그대로 유지
void iris_sdk_default_beauty_config(BeautyFilterConfig* config);
IrisSdkError iris_sdk_set_beauty_filter(const BeautyFilterConfig* config);
IrisSdkError iris_sdk_apply_beauty_filter(uint8_t* frame_data, ...);
```

### 5.2 V2 API (확장)

```cpp
// V2 구조체
typedef struct {
    // 기본 (V1 호환)
    bool enabled;
    float intensity;
    float smoothing;
    float brightness;
    float softFocus;

    // V2 확장: 피부 효과
    float whitening;
    float colorBalance;
    float wrinkleRemove;

    // V2 확장: 얼굴 형태
    float slimFace;
    float enlargeEyes;
    float thinChin;

    // V2 확장: 처리 옵션
    bool useGpu;
    bool roiOnly;
    bool protectEyes;
    bool protectLips;
} IrisBeautyFilterConfigV2;

// V2 API
IRIS_SDK_EXPORT void iris_sdk_default_beauty_config_v2(
    IrisBeautyFilterConfigV2* config);

IRIS_SDK_EXPORT IrisSdkError iris_sdk_set_beauty_filter_v2(
    const IrisBeautyFilterConfigV2* config);

IRIS_SDK_EXPORT IrisSdkError iris_sdk_apply_beauty_filter_v2(
    uint8_t* frame_data,
    int width, int height,
    IrisFrameFormat format,
    const IrisResult* iris_result  // Face Mesh 연동
);

// GPU 상태 확인
IRIS_SDK_EXPORT bool iris_sdk_beauty_gpu_available(void);
IRIS_SDK_EXPORT bool iris_sdk_beauty_using_gpu(void);
```

---

## 6. 리스크 및 완화 방안

### 6.1 렌더링 파이프라인 통합 (Critical - 피드백 반영)

**문제**: BeautyProcessor와 LensRenderer가 별도 GL Context 사용 시 성능 저하
- Context Switching 비용
- Texture Copy (ReadPixels/Upload) 비용
- GPU 가속 이점 상쇄

**완화 방안**:
1. **통합 RenderContext**: 모든 GPU 작업이 동일 컨텍스트 공유
2. **Zero-Copy 파이프라인**: TextureHandle을 통한 직접 전달
3. **Phase 1에서 인터페이스 확정**: LensRenderer와의 통합 설계 선행

```cpp
// Zero-Copy 파이프라인 예시
TextureHandle frame_tex = render_ctx->uploadTexture(camera_frame);
TextureHandle beauty_tex = beauty_processor->process(frame_tex);  // GPU 내부
TextureHandle lens_tex = lens_renderer->render(beauty_tex);       // 직접 전달
render_ctx->downloadTexture(lens_tex, output);  // 최종 1회만 다운로드
```

### 6.2 크로스 플랫폼 확장성 (피드백 반영)

**문제**: GLuint 직접 노출 시 iOS Metal 지원 어려움

**완화 방안**:
1. **TextureHandle 추상화**: `void*` 기반 플랫폼 독립적 핸들
2. **IRenderContext 인터페이스**: 플랫폼별 구현 분리
3. **일반적 파일명**: `render_context.h` (not `gl_context.h`)

### 6.3 OpenCV Contrib 의존성 (피드백 반영)

**문제**: `cv::ximgproc::guidedFilter` 사용 시 AAR 용량 증가 (수십 MB)

**완화 방안**:
1. **Fast Guided Filter 직접 구현**: Box Filter 기반 O(1) 알고리즘
2. **opencv-contrib 의존성 제거**: 코어 OpenCV만 사용
3. **AAR 용량 20MB 이하 유지**

### 6.4 GPU 초기화 오버헤드

**문제**: 이전 테스트에서 GPU 초기화 시 성능 저하 발생

**완화 방안**:
1. **지연 초기화**: 첫 GPU 필터 요청 시에만 초기화
2. **별도 스레드**: 메인 스레드 블로킹 방지
3. **자동 폴백**: 초기화 실패 시 CPU 백엔드로 전환

### 6.5 OpenGL ES 버전 호환성

**문제**: 일부 기기에서 ES 3.1 미지원

**완화 방안**:
1. **런타임 버전 체크**
2. **ES 3.0 폴백 셰이더** (Compute Shader 없이)
3. **기능별 폴백**: 미지원 기능은 CPU로 처리

### 6.6 Face Warp 자연스러움 (피드백 반영)

**문제**: 478개 랜드마크만 사용 시 이마, 볼 외곽에서 왜곡 부자연스러움

**완화 방안**:
1. **Grid Mesh (20x20)**: 균일한 변형 적용
2. **Influence 함수**: 랜드마크 거리 기반 가중치
3. **Vertex Shader 변형**: CPU 부하 감소

### 6.7 메모리 사용량 (피드백 반영)

**문제**: 1080p RGBA 텍스처 = 8MB, Ping-Pong 버퍼링 시 급증

**완화 방안**:
1. **downscaleFactor 옵션**: 저사양 기기용 해상도 감소
2. **텍스처 풀링**: 재사용으로 할당/해제 최소화
3. **30MB 총 추가 메모리 제한**

### 6.8 셰이더 컴파일 시간

**문제**: 첫 실행 시 지연

**완화 방안**:
1. **바이너리 캐싱**: `glGetProgramBinary` / `glProgramBinary`
2. **비동기 컴파일**: 백그라운드에서 미리 컴파일
3. **필수 셰이더만 즉시 컴파일**

### 6.9 Face Mesh 유효성

**문제**: 검출 실패 시 Face Mesh 무효

**완화 방안**:
1. **유효성 검사**: `face_mesh_valid` 플래그 확인
2. **이전 프레임 캐싱**: 100ms 내 유효한 캐시 재사용
3. **전체 프레임 폴백**: 캐시 없으면 전체 처리

---

## 7. 성능 목표

### 7.1 처리 시간 (1080p 기준)

| 효과 | CPU (현재) | CPU (최적화) | GPU | 목표 |
|------|------------|--------------|-----|------|
| ROI 계산 | N/A | < 1ms | < 0.5ms | < 1ms |
| 피부 스무딩 | 25-30ms | 10-15ms | 2-3ms | < 10ms |
| 소프트 포커스 | 8-10ms | 5-7ms | 1-2ms | < 5ms |
| 밝기 조절 | 2-3ms | 2-3ms | < 1ms | < 2ms |
| 피부톤 보정 | N/A | 8-10ms | 2-3ms | < 5ms |
| **전체** | **35-43ms** | **25-35ms** | **6-10ms** | **< 33ms** |

### 7.2 FPS 목표

| 시나리오 | 목표 FPS |
|----------|----------|
| 기본 필터 (GPU) | 30+ fps |
| 고급 필터 (GPU) | 25+ fps |
| 기본 필터 (CPU) | 20+ fps |
| 고급 필터 (CPU) | 15+ fps |

### 7.3 메모리 목표

| 항목 | 목표 |
|------|------|
| GPU 텍스처 | < 10MB |
| CPU 버퍼 | < 20MB |
| 셰이더 프로그램 | < 2MB |
| **총 추가 메모리** | **< 30MB** |

---

## 8. 검증 방법

### 8.1 단위 테스트

```cpp
// test_beauty_roi.cpp
TEST(BeautyROIManager, ComputeROIFromFaceMesh) {
    IrisLandmark face_mesh[478];
    loadTestFaceMesh(face_mesh);

    BeautyROI roi;
    EXPECT_TRUE(BeautyROIManager::computeROI(
        face_mesh, 1920, 1080, config, roi));
    EXPECT_TRUE(roi.valid);
}

TEST(BeautyROIManager, CreateSkinMaskExcludesEyes) {
    // 눈 영역이 피부 마스크에서 제외되는지 확인
}
```

### 8.2 성능 테스트

```cpp
// test_beauty_performance.cpp
TEST(BeautyPerformance, ROIFasterThanFullFrame) {
    // ROI 처리가 전체 프레임보다 30% 이상 빠른지 확인
}

TEST(BeautyPerformance, GPUFasterThanCPU) {
    // GPU가 CPU보다 2배 이상 빠른지 확인 (Android)
}
```

### 8.3 시각적 품질 테스트

```cpp
TEST(BeautyQuality, SmoothingPreservesEdges) {
    // 에지 보존율 80% 이상 확인
}

TEST(BeautyQuality, WhiteningNaturalLook) {
    // 화이트닝 효과 자연스러움 검증 (수동)
}
```

### 8.4 Android 디바이스 테스트

| 기기 | GPU | OpenGL ES | 테스트 항목 |
|------|-----|-----------|------------|
| Samsung Galaxy S21 | Adreno 660 | 3.2 | 전체 기능 |
| Pixel 6 | Mali-G78 | 3.2 | 전체 기능 |
| 저사양 기기 | - | 3.0 | 폴백 동작 |

---

## 9. 일정

| Phase | 작업 | 기간 | 상태 |
|-------|------|------|------|
| **Phase 1** | 기반 구조 리팩토링 | Week 1 | ⏳ |
| **Phase 2** | CPU 백엔드 개선 | Week 2 | ⏳ |
| **Phase 3** | GPU 백엔드 구현 | Week 3-4 | ⏳ |
| **Phase 4** | 얼굴 형태 보정 | Week 5-6 | ⏳ |
| **Phase 5** | 통합 및 최적화 | Week 7 | ⏳ |

---

## 10. 변경 이력

| 날짜 | 버전 | 변경 내용 | 작성자 |
|------|------|----------|--------|
| 2026-01-27 | 1.0 | 초안 작성 | Claude |
| 2026-01-27 | 1.1 | 피드백 반영 수정 | Claude |

### v1.1 변경 사항 (피드백 반영)

**Critical 수정**:
1. **통합 RenderContext 아키텍처**: GPUBeautyBackend + LensRenderer 간 Zero-Copy 파이프라인
2. **플랫폼 추상화**: `GLuint` → `TextureHandle`, `gl_context.h` → `render_context.h`

**중요 수정**:
3. **Fast Guided Filter 직접 구현**: opencv-contrib 의존성 제거 (AAR 용량 최적화)
4. **Grid Mesh 기반 Face Warp**: 478개 랜드마크 → 20x20 Grid로 자연스러운 변형
5. **싱글톤 → DI**: `GLContext::getInstance()` 제거, 의존성 주입 방식 채택

**세부 보완**:
6. **ROI Soft Feathering**: 경계 부자연스러움 방지 코드 추가
7. **downscaleFactor 옵션**: BeautyFilterConfigV2에 저사양 기기용 옵션 추가
8. **Mock 테스트 지원**: `mock_render_context.h` 추가

**피드백 문서**: `docs/conversation/feedback_010_beauty_filter_gpu_optimization.md`
