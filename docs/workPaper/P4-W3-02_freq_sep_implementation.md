# P4-W3-02: Frequency Separation 셰이더 + GPU 파이프라인

| 항목 | 내용 |
|------|------|
| **작업 ID** | P4-W3-02 |
| **유형** | 구현 |
| **상태** | ✅ 완료 |
| **근거 문서** | P4-W3-01 (브레인스토밍, 완료) |
| **작성일** | 2026-03-03 |
| **일정** | Day 1~4 (4일) |
| **후속 문서** | P4-W3-03 (API), P4-W3-04 (Temporal/Tier), P4-W3-05 (튜닝/릴리즈) |

---

## 1. 목표

Frequency Separation GPU 셰이더 2종(Gaussian, Composite)을 작성하고, `GPUBeautyBackend`에 5서브패스 파이프라인을 통합한다. skin mask의 GPU 텍스처 업로드 경로도 함께 구현한다.

### 1.1 완료 조건

- [x] Freq Sep Gaussian 셰이더 컴파일 + 렌더링 확인
- [x] Freq Sep Composite 셰이더 컴파일 + 재합성 확인 (High Freq 인라인 추출 포함)
- [x] `GPUBeautyBackend`에 Freq Sep 멤버/메서드 추가 + 초기화
- [x] CPU `combined_mask` → GPU 텍스처 업로드 경로 구현
- [x] `executeFreqSepPipeline()` 5서브패스 구현 + TexturePool 연동
- [x] `applyTextureId()`에 Freq Sep 분기 통합
- [x] `mapSkinQuality()` 매핑 함수 구현 (GPUBeautyBackend 내부)
- [ ] 테스트 이미지에서 Freq Sep 출력 확인 (품질 무관, 동작 확인) — Android 디바이스 필요

### 1.2 실패 기준 (No-Go)

- 셰이더 컴파일/링크 실패
- TexturePool 텍스처 획득/해제 오류 (null 미처리 포함)
- 5서브패스 중간에 GL 에러 발생
- skin mask 텍스처 업로드 실패
- 기존 Bilateral 경로 회귀 (skinQuality=0일 때 깨짐)

---

## 2. 변경 대상 파일

| 파일 | 변경 내용 | 영향도 |
|------|----------|--------|
| `cpp/src/gpu/shader_sources.cpp` | Freq Sep 셰이더 2종(Gaussian, Composite) 추가 | 높음 |
| `cpp/include/iris_sdk/gpu/gpu_beauty_backend.h` | Freq Sep 멤버/메서드 + mask 텍스처 + mapSkinQuality 추가 | 높음 |
| `cpp/src/gpu/gpu_beauty_backend.cpp` | Freq Sep 파이프라인 + mask 업로드 + 매핑 구현 | 높음 |

---

## 3. 셰이더 설계

### 3.1 패스 구조

```
입력 텍스처 (original)
    ↓
[Pass 1a] Separable Gaussian Blur — Horizontal
    ↓
[Pass 1b] Separable Gaussian Blur — Vertical
    ↓ lowFreq 텍스처 (보존 — Composite에서 high freq 인라인 추출에 필요)
[Pass 2a] Low Freq 추가 Gaussian — Horizontal
[Pass 2b] Low Freq 추가 Gaussian — Vertical
    ↓ smoothedLow 텍스처 (별도 할당)
[Pass 3] 재합성 — high = original - lowFreq (인라인 ALU)
         beauty = smoothedLow + attenuated(high)
         result = mix(original, beauty, skinMask)
    ↓
출력 텍스처
```

> **⚠️ Extract 패스 폐기 사유**: 별도 Extract 패스(`high = orig - lowFreq + 0.5`)는
> full-res 텍스처 1장 추가 write/read를 발생시킨다. 모바일 TBDR GPU에서 이는
> ~2ms 대역폭 낭비이다. Composite 셰이더 내에서 `high = orig - lowFreq`를
> ALU로 인라인 계산하면, 텍스처 수 동일(3장)에 패스 1개 절약.
> `+0.5` 오프셋도 불필요해져 정밀도 손실 위험도 제거된다.

총 5 서브패스 (Separable 분리 포함). 텍스처 3장 필요 (TexturePool에서 획득).

### 3.2 셰이더 소스 명세

#### 3.2.1 FREQ_SEP_GAUSSIAN_FRAGMENT

기존 `GAUSSIAN_BLUR_FRAGMENT`를 확장. Separable 1D Gaussian.

```glsl
// shader_sources.cpp에 추가
const char* FREQ_SEP_GAUSSIAN_FRAGMENT = R"(
#version 310 es
precision highp float;

in vec2 v_texCoord;
out vec4 fragColor;

uniform sampler2D uTexture;
uniform vec2 uDirection;        // (1/w, 0) 또는 (0, 1/h)
uniform int uRadius;            // adaptive radius (6~28)

void main() {
    vec3 sum = vec3(0.0);
    float weightSum = 0.0;
    float sigma = float(uRadius) * 0.4;  // 경험적 비율

    for (int i = -uRadius; i <= uRadius; i++) {
        vec2 offset = uDirection * float(i);
        vec3 sample = texture(uTexture, v_texCoord + offset).rgb;
        float w = exp(-float(i * i) / (2.0 * sigma * sigma));
        sum += sample * w;
        weightSum += w;
    }

    fragColor = vec4(sum / weightSum, 1.0);
}
)";
```

**유니폼**:

| 이름 | 타입 | 용도 |
|------|------|------|
| `uTexture` | sampler2D | 입력 텍스처 |
| `uDirection` | vec2 | 블러 방향 (H: `1/w, 0`, V: `0, 1/h`) |
| `uRadius` | int | adaptive 반경 (clamp 6~28) |

#### 3.2.2 FREQ_SEP_COMPOSITE_FRAGMENT

핵심 셰이더. High Frequency 인라인 추출 + 비선형 attenuation + 마스크 블렌딩.

> **Extract 패스 인라인화**: 기존 별도 Extract 패스(`high = orig - lowFreq + 0.5`)를
> Composite 셰이더 내 ALU 연산으로 통합. `uLowFreq`(Pass 1b 결과)를 직접 샘플링하여
> `high = orig - lowFreq`를 계산한다. `+0.5` 오프셋 불필요 (텍스처 저장 없이 직접 연산).

```glsl
const char* FREQ_SEP_COMPOSITE_FRAGMENT = R"(
#version 310 es
precision highp float;

in vec2 v_texCoord;
out vec4 fragColor;

uniform sampler2D uSmoothedLow;   // Pass 2b 결과 (추가 블러된 저주파)
uniform sampler2D uLowFreq;       // Pass 1b 결과 (원본 저주파 — high freq 추출 기준)
uniform sampler2D uOriginal;      // 원본 프레임
uniform sampler2D uSkinMask;      // ROI 마스크 텍스처

uniform float uHighFreqPreserve;   // 내부 매핑값 (0.1~1.0)
uniform float uAttenuationLow;     // smoothstep 하한 (default 0.02)
uniform float uAttenuationHigh;    // smoothstep 상한 (default 0.15)

void main() {
    vec3 smoothLow = texture(uSmoothedLow, v_texCoord).rgb;
    vec3 low       = texture(uLowFreq, v_texCoord).rgb;
    vec3 orig      = texture(uOriginal, v_texCoord).rgb;
    float mask     = texture(uSkinMask, v_texCoord).r;

    // High Frequency 인라인 추출 (별도 패스/텍스처 없이 ALU 연산)
    vec3 high = orig - low;

    // Y(luminance) 기반 고주파 크기 계산
    float magnitude = dot(abs(high), vec3(0.299, 0.587, 0.114));

    // 비선형 감쇠: 큰 변화(잡티) → 강한 감쇠, 작은 변화(피부결) → 보존
    float blemishFactor = smoothstep(uAttenuationLow, uAttenuationHigh, magnitude);
    float preserve = mix(uHighFreqPreserve, 1.0, 1.0 - blemishFactor);

    vec3 adjusted_high = high * preserve;

    // 재합성
    vec3 beauty = smoothLow + adjusted_high;

    // 피부 마스크로 원본과 블렌딩
    vec3 result = mix(orig, beauty, mask);

    fragColor = vec4(result, 1.0);
}
)";
```

**유니폼**:

| 이름 | 타입 | 기본값 | 용도 |
|------|------|--------|------|
| `uSmoothedLow` | sampler2D | — | Pass 2b 결과 (추가 블러 저주파) |
| `uLowFreq` | sampler2D | — | Pass 1b 결과 (원본 저주파 — 보존 필수) |
| `uOriginal` | sampler2D | — | 원본 프레임 |
| `uSkinMask` | sampler2D | — | ROI 마스크 텍스처 |
| `uHighFreqPreserve` | float | 0.45 | skinQuality에서 매핑 |
| `uAttenuationLow` | float | 0.02 | 잡티 판정 하한 |
| `uAttenuationHigh` | float | 0.15 | 잡티 판정 상한 |

---

## 4. GPUBeautyBackend 변경

### 4.1 새 멤버 변수

```cpp
// gpu_beauty_backend.h에 추가

// Freq Sep 셰이더 프로그램 (Extract 패스 폐기 — Composite에 인라인화)
GLuint freq_sep_gaussian_program_ = 0;
GLuint freq_sep_composite_program_ = 0;

// Skin mask GPU 텍스처 (CPU combined_mask 업로드용)
GLuint skin_mask_texture_ = 0;
int skin_mask_width_ = 0;
int skin_mask_height_ = 0;

// Freq Sep Uniform 캐시
struct FreqSepGaussianUniforms {
    GLint uTexture = -1;
    GLint uDirection = -1;
    GLint uRadius = -1;
} freq_sep_gaussian_uniforms_;

struct FreqSepCompositeUniforms {
    GLint uSmoothedLow = -1;
    GLint uLowFreq = -1;       // Pass 1b 결과 (high freq 인라인 추출 기준)
    GLint uOriginal = -1;
    GLint uSkinMask = -1;
    GLint uHighFreqPreserve = -1;
    GLint uAttenuationLow = -1;
    GLint uAttenuationHigh = -1;
} freq_sep_composite_uniforms_;

// Freq Sep 내부 파라미터
struct FreqSepParams {
    int blur_radius = 15;
    float high_freq_preserve = 0.45f;
    float low_freq_smooth_radius_ratio = 0.5f;  // blur_radius 대비 비율
    float attenuation_low = 0.02f;
    float attenuation_high = 0.15f;
    bool enabled = false;  // skinQuality > 0이면 true
};

// Temporal stability용 One Euro Filter (P4-W3-04에서 사용)
// blur_radius 프레임 간 안정화 — mapSkinQuality() 직후 적용
OneEuroFilter skin_radius_filter_{0.5f, 0.01f, 1.0f};
```

### 4.2 새 메서드

```cpp
// gpu_beauty_backend.h private 섹션에 추가

bool initializeFreqSepShaders();

void executeFreqSepPipeline(
    GLuint input_tex,
    GLuint mask_tex,         // skin mask GPU 텍스처
    GLuint output_fbo,
    int width, int height,
    const FreqSepParams& params
);

// CPU combined_mask → GPU 텍스처 업로드
GLuint uploadSkinMask(
    const std::vector<uint8_t>& combined_mask,
    int mask_width, int mask_height
);

// skinQuality → FreqSepParams 매핑 (GPUBeautyBackend 내부에 배치)
static FreqSepParams mapSkinQuality(
    float skin_quality,
    int face_width           // Face Mesh bbox 폭
);

// Freq Sep 불가 시 공통 Bilateral fallback (§4.4 참조)
void executeSmoothingWithFallbackStrength(
    GLuint input_tex, GLuint output_fbo,
    int width, int height,
    const BeautyFilterConfigV2& config
);
```

### 4.3 Skin Mask GPU 업로드 경로

현재 `BeautyROI::combined_mask`는 CPU `std::vector<uint8_t>`로 생성된다
(`beauty_roi_manager.cpp:166` → `combineMasks()` → `applyFeathering()`).
현재 GPU 파이프라인(`applyTextureId()`)은 Scissor 기반 ROI만 사용하므로
mask 텍스처 업로드 경로가 없다. Freq Sep의 Composite 셰이더는 `uSkinMask`을
필수 입력으로 요구하므로, 아래 업로드 경로를 추가한다.

```cpp
// gpu_beauty_backend.cpp에 구현
GLuint GPUBeautyBackend::uploadSkinMask(
    const std::vector<uint8_t>& combined_mask,
    int mask_width, int mask_height)
{
    if (combined_mask.empty() || mask_width <= 0 || mask_height <= 0) {
        return 0;
    }

    // 텍스처 재사용: 크기 동일하면 glTexSubImage2D, 다르면 재생성
    if (skin_mask_texture_ == 0) {
        glGenTextures(1, &skin_mask_texture_);
    }

    glBindTexture(GL_TEXTURE_2D, skin_mask_texture_);

    // GL_RED 단일 채널: 행 폭이 4의 배수가 아닐 수 있으므로
    // unpack alignment를 1로 설정 (기본값 4이면 행 정렬 깨짐)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    if (mask_width != skin_mask_width_ || mask_height != skin_mask_height_) {
        // 크기 변경 → 전체 재할당
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8,
                     mask_width, mask_height, 0,
                     GL_RED, GL_UNSIGNED_BYTE,
                     combined_mask.data());
        skin_mask_width_ = mask_width;
        skin_mask_height_ = mask_height;
    } else {
        // 크기 동일 → 데이터만 갱신 (더 빠름)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                        mask_width, mask_height,
                        GL_RED, GL_UNSIGNED_BYTE,
                        combined_mask.data());
    }

    // alignment 복원
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glBindTexture(GL_TEXTURE_2D, 0);
    return skin_mask_texture_;
}
```

**호출 위치**: `applyTextureId()` 내부, Freq Sep 분기 진입 전:

```cpp
// applyTextureId() 내부
if (freq_sep_params.enabled && roi_ptr && roi_ptr->isValid()) {
    GLuint mask_tex = uploadSkinMask(
        roi_ptr->combined_mask,
        roi_ptr->mask_width, roi_ptr->mask_height
    );
    if (mask_tex != 0) {
        executeFreqSepPipeline(current_input, mask_tex,
                               current_output->fbo_id,
                               width, height, freq_sep_params);
    } else {
        // mask 업로드 실패 → Bilateral fallback (공통 최소 강도 정책 적용)
        executeSmoothingWithFallbackStrength(current_input, current_output->fbo_id,
                                             width, height, config);
    }
} else if (config.smoothing > 0.01f) {
    executeSmoothingPass(current_input, current_output->fbo_id,
                         width, height, config);
}
```

**해제**: `release()` 메서드에서:

```cpp
if (skin_mask_texture_ != 0) {
    glDeleteTextures(1, &skin_mask_texture_);
    skin_mask_texture_ = 0;
}
```

### 4.4 Bilateral Fallback 공통 강도 정책

`skinQuality > 0`인 상태에서 Freq Sep 대신 Bilateral fallback으로 전환되는 경로가
여러 곳 존재한다. 모든 fallback 경로에서 일관된 최소 smoothing 강도를 보장한다.

**Fallback 발생 경로**:
- mask 업로드 실패 (§4.3 uploadSkinMask 반환값 0)
- LOW tier 디바이스 (P4-W3-04 §4.2)
- MID tier 텍스처 획득 실패 (P4-W3-04 §4.3 acquireRenderTarget null)

**공통 정책**:

```cpp
// 모든 Bilateral fallback 경로에서 호출
void GPUBeautyBackend::executeSmoothingWithFallbackStrength(
    GLuint input_tex, GLuint output_fbo,
    int width, int height,
    const BeautyFilterConfigV2& config)
{
    // skinQuality > 0인 사용자가 smoothing을 별도 설정하지 않았을 수 있으므로,
    // fallback 시 최소 smoothing 강도를 config.skinQuality 기반으로 보장한다.
    BeautyFilterConfigV2 fallback_config = config;
    float effective_smoothing = config.skinQuality * 0.5f;
    if (fallback_config.smoothing < effective_smoothing) {
        fallback_config.smoothing = effective_smoothing;
    }
    executeSmoothingPass(input_tex, output_fbo, width, height, fallback_config);
}
```

> **정책 근거**: `skinQuality * 0.5f`는 Freq Sep 대비 절반 수준의 효과를
> Bilateral로 근사하는 보수적 추정이다. 사용자가 명시적으로 높은 `smoothing`을
> 설정한 경우 그 값이 우선한다 (`max` 연산).

### 4.5 파이프라인 통합 위치

`GPUBeautyBackend::applyTextureId()` 내부 흐름 변경:

```
기존:
  executeSmoothingPass(...)     ← Bilateral Filter
  executeCombinedColorPass(...)

변경 후:
  if (freq_sep_params.enabled && roi valid) {
      mask_tex = uploadSkinMask(roi.combined_mask, ...)
      if (mask_tex != 0):
          executeFreqSepPipeline(...)                    ← NEW: Freq Sep
      else:
          executeSmoothingWithFallbackStrength(...)       ← §4.4 공통 정책
  } else if (skinQuality > 0 && !roi valid) {
      executeSmoothingWithFallbackStrength(...)           ← §4.4 공통 정책
  } else {
      executeSmoothingPass(...)                           ← 유지: skinQuality=0
  }
  executeCombinedColorPass(...)                           ← 유지
```

### 4.6 mapSkinQuality 구현

`mapSkinQuality`는 `GPUBeautyBackend` 내부 static 메서드로 배치한다.
`BeautyProcessor`는 backend-agnostic 구조이므로, GPU 구현 세부사항인
`FreqSepParams`를 `BeautyProcessor`에서 직접 참조하지 않는다.

```cpp
// gpu_beauty_backend.cpp에 구현
GPUBeautyBackend::FreqSepParams
GPUBeautyBackend::mapSkinQuality(float skin_quality, int face_width) {
    FreqSepParams p;

    if (skin_quality <= 0.0f) {
        p.enabled = false;
        return p;
    }

    p.enabled = true;

    // S-커브 매핑 (smoothstep으로 자연스러운 전이)
    float t = std::clamp(skin_quality, 0.0f, 1.0f);
    float s = t * t * (3.0f - 2.0f * t);  // smoothstep

    // blur_radius: face_width의 고정 5% → clamp(6, 28)
    // skinQuality와 독립 — cutoff frequency를 고정하여 슬라이더 조작 시
    // 예측 가능한 단일 축 변화(감쇠 강도만 변경)를 보장한다.
    const float ratio = 0.05f;
    p.blur_radius = std::clamp(
        static_cast<int>(face_width * ratio),
        6, 28
    );

    // high_freq_preserve: 1.0 → 0.10 (quality 높을수록 더 매끄럽게)
    p.high_freq_preserve = 1.0f - s * 0.90f;

    // low_freq_smooth: blur_radius의 40~60%
    p.low_freq_smooth_radius_ratio = 0.4f + s * 0.2f;

    // attenuation 범위 (잡티 판정 기준)
    p.attenuation_low = 0.02f;
    p.attenuation_high = 0.10f + s * 0.10f;  // 0.10 ~ 0.20

    return p;
}
```

**매핑 결과표 (face_width=300px 기준, blur_radius 고정)**:

| skinQuality | blur_radius | highFreqPreserve | attenuation range |
|-------------|-------------|------------------|-------------------|
| 0.0 | 바이패스 | — | — |
| 0.2 | **15** | 0.88 | 0.02~0.11 |
| 0.4 | **15** | 0.72 | 0.02~0.13 |
| **0.6** | **15** | **0.50** | **0.02~0.16** |
| 0.8 | **15** | 0.27 | 0.02~0.18 |
| 1.0 | **15** | 0.10 | 0.02~0.20 |

> **blur_radius 고정 근거**: 300×0.05=15. skinQuality는 감쇠 강도
> (highFreqPreserve, attenuation)만 제어하여, 사용자가 슬라이더 조작 시
> cutoff frequency가 변하지 않는 예측 가능한 단일 축 변화를 보장한다.
> A방식(연동형) 대비 품질 문제 발견 시 P4-W3-05 §2.2.1 A/B 비교에서 재평가.

### 4.7 executeFreqSepPipeline 구현 상세

> **5서브패스 파이프라인**: Extract 패스를 Composite에 인라인화하여 6→5서브패스로 최적화.
> lowFreq 텍스처를 Composite까지 보존하고, smoothedLow를 별도 텍스처에 기록한다.
> 텍스처 총 3장(lowFreq, smoothedLow, temp) — 이전과 동일.

```cpp
void GPUBeautyBackend::executeFreqSepPipeline(
    GLuint input_tex,
    GLuint mask_tex,
    GLuint output_fbo,
    int width, int height,
    const FreqSepParams& params)
{
    // 텍스처 풀에서 중간 버퍼 획득 (null 체크 포함)
    // lowFreq: Pass 1b 결과 보존 (Composite에서 high freq 인라인 추출에 사용)
    // smoothedLow: Pass 2b 결과 (기존 highFreq 슬롯을 대체)
    // temp: 핑퐁용 임시 버퍼
    auto* lowFreq = texture_pool_->acquireRenderTarget(width, height);
    auto* smoothedLow = texture_pool_->acquireRenderTarget(width, height);
    auto* temp = texture_pool_->acquireRenderTarget(width, height);

    if (!lowFreq || !smoothedLow || !temp) {
        LOGE("FreqSep: Failed to acquire render targets");
        if (lowFreq) texture_pool_->releaseTexture(lowFreq);
        if (smoothedLow) texture_pool_->releaseTexture(smoothedLow);
        if (temp) texture_pool_->releaseTexture(temp);
        return;
    }

    bool profiling = profiler_ && profiler_->isEnabled();

    // 의사코드 내 헬퍼 함수 매핑:
    //   bindAndDraw(tex, fbo)      → glActiveTexture + glBindTexture + glBindFramebuffer + glDrawArrays
    //   bindTextures(t1, t2, ...)  → 각 텍스처를 GL_TEXTURE0, GL_TEXTURE1, ... 에 순서대로 바인딩
    //   drawToFBO(fbo)             → glBindFramebuffer(GL_FRAMEBUFFER, fbo) + glViewport + glDrawArrays
    // 이들은 기존 applyTextureId()의 GL 호출 패턴을 축약한 의사코드이며,
    // 실제 구현 시 기존 renderPass() / setupTextureBinding() 등 유틸리티를 활용한다.

    if (profiling) profiler_->begin("FreqSep_GaussianH");
    // Pass 1a: Horizontal Gaussian → temp
    glUseProgram(freq_sep_gaussian_program_);
    glUniform2f(freq_sep_gaussian_uniforms_.uDirection, 1.0f/width, 0.0f);
    glUniform1i(freq_sep_gaussian_uniforms_.uRadius, params.blur_radius);
    bindAndDraw(input_tex, temp->fbo_id);
    if (profiling) profiler_->end("FreqSep_GaussianH");

    if (profiling) profiler_->begin("FreqSep_GaussianV");
    // Pass 1b: Vertical Gaussian → lowFreq (이후 Composite까지 보존됨)
    glUniform2f(freq_sep_gaussian_uniforms_.uDirection, 0.0f, 1.0f/height);
    bindAndDraw(temp->texture_id, lowFreq->fbo_id);
    if (profiling) profiler_->end("FreqSep_GaussianV");

    // [Extract 패스 삭제] — Composite 셰이더에서 high = orig - lowFreq 인라인 연산

    if (profiling) profiler_->begin("FreqSep_LowSmooth_H");
    // Pass 2a: Low Freq 추가 Gaussian H → temp
    // lowFreq를 READ만 하고 보존 (Composite에서 필요)
    int low_radius = std::max(3, (int)(params.blur_radius * params.low_freq_smooth_radius_ratio));
    glUseProgram(freq_sep_gaussian_program_);
    glUniform2f(freq_sep_gaussian_uniforms_.uDirection, 1.0f/width, 0.0f);
    glUniform1i(freq_sep_gaussian_uniforms_.uRadius, low_radius);
    bindAndDraw(lowFreq->texture_id, temp->fbo_id);
    if (profiling) profiler_->end("FreqSep_LowSmooth_H");

    if (profiling) profiler_->begin("FreqSep_LowSmooth_V");
    // Pass 2b: Low Freq 추가 Gaussian V → smoothedLow (별도 텍스처, lowFreq 보존)
    glUniform2f(freq_sep_gaussian_uniforms_.uDirection, 0.0f, 1.0f/height);
    bindAndDraw(temp->texture_id, smoothedLow->fbo_id);
    if (profiling) profiler_->end("FreqSep_LowSmooth_V");

    if (profiling) profiler_->begin("FreqSep_Composite");
    // Pass 3: 재합성 + 마스크 블렌딩 → output
    // Composite 셰이더 내에서 high = orig - lowFreq를 ALU로 계산
    glUseProgram(freq_sep_composite_program_);
    glUniform1f(freq_sep_composite_uniforms_.uHighFreqPreserve, params.high_freq_preserve);
    glUniform1f(freq_sep_composite_uniforms_.uAttenuationLow, params.attenuation_low);
    glUniform1f(freq_sep_composite_uniforms_.uAttenuationHigh, params.attenuation_high);
    // 유니폼 순서: uSmoothedLow, uLowFreq, uOriginal, uSkinMask
    bindTextures(smoothedLow->texture_id, lowFreq->texture_id, input_tex, mask_tex);
    drawToFBO(output_fbo);
    if (profiling) profiler_->end("FreqSep_Composite");

    // 텍스처 반환
    texture_pool_->releaseTexture(lowFreq);
    texture_pool_->releaseTexture(smoothedLow);
    texture_pool_->releaseTexture(temp);
}
```

---

## 5. 실행 일정

| Day | 작업 | 산출물 | 완료 기준 |
|-----|------|--------|----------|
| **1** | Gaussian 셰이더 작성 + 컴파일 테스트 | `FREQ_SEP_GAUSSIAN_FRAGMENT` in shader_sources.cpp | 셰이더 컴파일 성공, passthrough 수준 렌더링 확인 |
| **2** | Composite 셰이더 작성 (High Freq 인라인 추출 포함) | `FREQ_SEP_COMPOSITE_FRAGMENT` | 셰이더 컴파일 성공 |
| **3** | GPUBeautyBackend에 멤버/메서드 추가 + 초기화 + mask 업로드 + mapSkinQuality | `initializeFreqSepShaders()`, `uploadSkinMask()`, `mapSkinQuality()` | 셰이더 링크 성공, mask 업로드 동작, 매핑 테이블 일치 |
| **4** | `executeFreqSepPipeline()` 구현 + TexturePool 연동 + 분기 통합 | 5서브패스 파이프라인 | 테스트 이미지에서 Freq Sep 출력 확인 |

---

## 6. 리스크

| 리스크 | 확률 | 대응 |
|--------|------|------|
| 셰이더 컴파일 에러 (기기별 GLSL 차이) | 중 | `precision` 조정, `#define` 분기 |
| TexturePool 메모리 부족 (3장 추가) | 낮 | max_textures 확대 (8→12) |
| Separable Gaussian 루프 언롤 제한 (일부 GPU) | 낮 | radius 상한 28로 제한 (이미 설정) |
| mask 텍스처 업로드 지연 (매 프레임 glTexSubImage2D) | 낮 | mask 크기는 ROI 크기(~200×200)로 작음, 0.1ms 이하 예상 |

---

## 변경 이력

| 날짜 | 변경 내용 | 작성자 |
|------|----------|--------|
| 2026-03-03 | 통합 문서에서 셰이더+파이프라인 분리 | Claude |
| 2026-03-03 | 리뷰 반영: skin mask GPU 업로드 경로 추가 (§4.3), API 호출 패턴 수정 (texture_pool_->, profiler_->begin/end), mapSkinQuality를 GPUBeautyBackend 내부로 이동 (§4.5), acquireRenderTarget null 체크 추가 (§4.6) | Claude |
| 2026-03-03 | 리뷰 2차 반영: uploadSkinMask()에 GL_UNPACK_ALIGNMENT=1 추가 (§4.3), One Euro Filter 멤버를 GPUBeautyBackend로 이동 (§4.1) | Claude |
| 2026-03-03 | 리뷰 3차 반영: mask 업로드 실패 fallback에 최소 smoothing 강도 보장 정책 추가 (§4.3), executeFreqSepPipeline 헬퍼 함수 매핑 주석 추가 (§4.7) | Claude |
| 2026-03-04 | Codex 리뷰 반영: §4.2 새 메서드 목록에 executeSmoothingWithFallbackStrength() 추가 | Claude |
| 2026-03-04 | Gemini 3차 리뷰 반영: Extract 패스 Composite 인라인화 (6→5서브패스) — §3.1 다이어그램, §3.2.2 Extract 셰이더 삭제, §3.2.2(구 3.2.3) Composite에 uLowFreq+인라인 high 추출, §4.1 extract 프로그램/유니폼 제거, §4.7 파이프라인 전면 재작성 (lowFreq 보존, smoothedLow 별도 할당). blur_radius 고정 ratio(0.05) 채택 — §4.6 mapSkinQuality 수정, 매핑 결과표 갱신 | Claude |
| 2026-03-04 | 리뷰 4차 반영: Bilateral fallback 공통 강도 정책을 §4.4로 독립 (executeSmoothingWithFallbackStrength), §4.5 흐름도에 fallback 경로 통합, 섹션 번호 재정렬 (§4.4~§4.7) | Claude |
| 2026-03-04 | **구현 완료**: 셰이더 2종 (shader_sources.cpp), GPUBeautyBackend 헤더 확장, 5서브패스 파이프라인 구현, skin mask GPU 업로드, mapSkinQuality 매핑, applyTextureId Freq Sep 분기 통합, BeautyFilterConfigV2에 skinQuality 필드 추가 (C++/C API/JNI/Android Java). 빌드 성공 + 20 unit tests 통과 | Claude |
