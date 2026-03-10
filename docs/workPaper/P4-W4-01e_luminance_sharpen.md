# P4-W4-01e: Luminance Sharpen 패스 추가

> **상위 문서**: `P4-W4-01_freqsep_quality_improvement.md`
> **상태**: ✅ 구현 완료
> **난이도**: 중간 | **추가 GPU 비용**: +1~2 패스 (Separable Unsharp Mask)
> **선행 조건**: Step 1~4 완료 후 최종 단계로 적용

---

## 1. 왜 필요한가

### 현재 문제

FreqSep 파이프라인은 다단계 Gaussian blur + 고주파 감쇠를 거치면서 필연적으로 **전반적 선명도 손실**이 발생:

- Gaussian blur가 에지를 포함한 모든 고주파를 감쇠
- 잡티 감쇠 과정에서 미세한 피부 텍스처까지 일부 손실
- Soft Light 합성이 Additive보다 부드럽게 작용하여 선명도 추가 감소
- **눈, 머리카락, 입술** 등 비피부 영역은 mask로 보호되지만, 피부 영역의 질감 손실은 불가피

상용 SDK들은 보정 파이프라인 마지막에 **Sharpen 패스**를 추가하여 perceived sharpness를 복구한다.

### Luminance-only Sharpen의 이점

일반 Unsharp Mask는 RGB 전 채널에 적용하여 **색상 노이즈 증폭** 문제가 있다.
Luminance-only Sharpen은 밝기 채널만 샤프닝:

- **색상 아티팩트 없음**: Chrominance는 그대로 보존
- **지각적 선명도**: 인간 시각은 밝기 변화에 민감 → 밝기만 샤프닝해도 충분
- **노이즈 증폭 최소**: 색상 노이즈는 건드리지 않음
- **연산 효율**: Y 채널 하나만 처리

### Unsharp Mask 원리

```
sharpened = original + amount * (original - blur(original))
         = original + amount * high_frequency
```

우리 파이프라인에서는 이미 Composite 결과의 "beauty" 텍스처가 있으므로,
이 텍스처를 3×3 또는 5×5 작은 Gaussian으로 blur한 뒤 차이를 더한다.

### Council 합의 근거

> "Sharpen은 Council 만장일치로 intensity 0.12~0.18 고정 권장. 슬라이더 노출 불필요, 파이프라인 마지막에 mask 내부에서만 적용" — 만장일치

---

## 2. 아키텍처 선택

### 옵션 A: 별도 Sharpen 패스 (권장)

```
[Pass 3] Composite → beauty_output
[Pass 4] Sharpen(beauty_output) → final_output
```

- Composite 결과를 별도 텍스처에 렌더링
- Sharpen 셰이더가 해당 텍스처를 읽어 샤프닝 적용
- **장점**: 셰이더 분리로 유지보수 용이, 디버깅 쉬움
- **단점**: +1 패스, +1 render target

### 옵션 B: Composite 셰이더 내 Inline Sharpen

- Composite 내에서 beauty 계산 후 주변 텍셀 샘플링으로 blur 계산
- **장점**: 패스 추가 없음
- **단점**: Composite가 이미 4개 텍스처를 사용하므로 추가 샘플링이 과부하
  (ES 3.1 기준 최소 16개 texture unit 보장이지만, bandwidth 병목)

### 옵션 C: Separable Sharpen (2 패스)

```
[Pass 3] Composite → beauty_output
[Pass 4a] Horizontal blur(beauty_output) → blur_temp
[Pass 4b] Sharpen = beauty_output + amount * (beauty_output - blur_temp)
```

→ 사실상 단일 패스 Sharpen과 동일한 결과이므로 **옵션 A 채택**.

**최종 선택: 옵션 A** — 단일 패스 Luminance Sharpen

---

## 3. 변경 대상

### 3.1 LUMINANCE_SHARPEN_FRAGMENT 셰이더 (신규)

**파일**: `cpp/src/gpu/shader_sources.cpp` — 기존 FREQ_SEP_COMPOSITE_FRAGMENT 다음에 추가

```glsl
//=============================================================================
// Luminance Sharpen 프래그먼트 셰이더
// Unsharp Mask on luminance channel only — 색상 아티팩트 방지
//=============================================================================
const char* LUMINANCE_SHARPEN_FRAGMENT = R"glsl(
#version 310 es
precision highp float;

in vec2 vTexCoord;
out vec4 fragColor;

uniform sampler2D uTexture;      // Composite 결과 (beauty)
uniform sampler2D uSkinMask;     // ROI mask
// 참고: uOriginal은 불필요 — 셰이더 본문에서 사용하지 않음 (mask 기반 블렌딩만)
uniform float uSharpenAmount;    // 샤프닝 강도 (0.0~0.5, 기본 0.15)
uniform vec2 uTexelSize;         // (1/width, 1/height)

void main() {
    vec3 center = texture(uTexture, vTexCoord).rgb;
    float mask = texture(uSkinMask, vec2(vTexCoord.x, 1.0 - vTexCoord.y)).r;

    // 5-tap cross blur (중심 가중치 2/6, 이웃 각 1/6)
    // Luminance만 계산하여 성능 최적화
    // Linear-light 공간에서는 Rec.709 계수 사용
    const vec3 LUMA_709 = vec3(0.2126, 0.7152, 0.0722);
    float lumCenter = dot(center, LUMA_709);

    // 4-neighbor (cross) 샘플링 — 대각 생략으로 4 텍스처 읽기
    float lumL = dot(texture(uTexture, vTexCoord - vec2(uTexelSize.x, 0.0)).rgb,
                     LUMA_709);
    float lumR = dot(texture(uTexture, vTexCoord + vec2(uTexelSize.x, 0.0)).rgb,
                     LUMA_709);
    float lumU = dot(texture(uTexture, vTexCoord - vec2(0.0, uTexelSize.y)).rgb,
                     LUMA_709);
    float lumD = dot(texture(uTexture, vTexCoord + vec2(0.0, uTexelSize.y)).rgb,
                     LUMA_709);

    // 간이 blur: (center*2 + neighbors) / 6
    float lumBlur = (lumCenter * 2.0 + lumL + lumR + lumU + lumD) / 6.0;

    // Unsharp Mask: high_freq = center - blur
    float lumSharp = lumCenter + uSharpenAmount * (lumCenter - lumBlur);
    lumSharp = clamp(lumSharp, 0.0, 1.0);

    // Luminance 비율로 RGB 조정 (색상 보존)
    float ratio = (lumCenter > 0.001) ? (lumSharp / lumCenter) : 1.0;
    vec3 sharpened = center * ratio;
    sharpened = clamp(sharpened, 0.0, 1.0);

    // mask 영역에만 sharpen 적용
    vec3 result = mix(center, sharpened, mask);

    fragColor = vec4(result, 1.0);
}
)glsl";
```

**설계 결정**:
- **4-neighbor** (cross) 패턴: 대각 텍셀 생략으로 4 샘플 (8-neighbor는 8 샘플)
- **Luminance 비율 방식**: `ratio = sharpLum / origLum` → RGB에 곱하면 색상 보존
- **mask 적용**: 비피부 영역 (눈, 입술)은 샤프닝에서 제외
- **clamp**: luminance ratio가 극단적일 때 안전장치
- **uTexelSize**: textureSize() 대신 CPU에서 전달 (일부 ES 기기 호환성)

### 3.2 셰이더 소스 선언 (헤더)

**파일**: `cpp/src/gpu/shader_sources.cpp` 상단 또는 별도 선언 위치

```cpp
extern const char* LUMINANCE_SHARPEN_FRAGMENT;
```

### 3.3 gpu_beauty_backend.h — 셰이더 프로그램 + Uniform 추가

**파일**: `cpp/include/iris_sdk/gpu/gpu_beauty_backend.h`

**셰이더 프로그램** (line 406~407 근처):
```cpp
// Freq Sep 셰이더 프로그램
GLuint freq_sep_gaussian_program_ = 0;
GLuint freq_sep_composite_program_ = 0;
GLuint luminance_sharpen_program_ = 0;  // ★ 추가
```

**Uniform 캐시** (FreqSepCompositeUniforms 다음에):
```cpp
// Luminance Sharpen Uniform 캐시
struct LuminanceSharpenUniforms {
    GLint uTexture = -1;
    GLint uSkinMask = -1;
    GLint uSharpenAmount = -1;
    GLint uTexelSize = -1;
} luminance_sharpen_uniforms_;
```

**FreqSepParams 확장** (line 233~240):
```cpp
struct FreqSepParams {
    int blur_radius = 15;
    float high_freq_preserve = 0.45f;
    float low_freq_smooth_radius_ratio = 0.5f;
    float attenuation_low = 0.02f;
    float attenuation_high = 0.15f;
    float edge_weight = 0.5f;      // Step 3
    float chroma_weight = 0.3f;    // Step 3
    float tone_lift = 0.15f;       // Step 4
    float sharpen_amount = 0.15f;  // ★ Step 5 추가
    bool enabled = false;
};
```

### 3.4 gpu_beauty_backend.cpp — initializeShaders()

**파일**: `cpp/src/gpu/gpu_beauty_backend.cpp`

셰이더 컴파일 영역에 추가:

```cpp
// Luminance Sharpen 프로그램 (★ Step 5)
// 기존 FreqSep 셰이더와 동일하게 ShaderManager에 캐시하는 패턴 사용
if (!shader_manager_->createProgram(
        shaders::FULLSCREEN_QUAD_VERTEX,
        shaders::LUMINANCE_SHARPEN_FRAGMENT,
        luminance_sharpen_program_)) {
    LOGE("Failed to create Luminance Sharpen program");
    // 비필수 패스이므로 실패해도 초기화 계속
} else {
    shader_manager_->cacheProgram("luminance_sharpen", luminance_sharpen_program_);
}
```

### 3.5 gpu_beauty_backend.cpp — Uniform 초기화

**파일**: `cpp/src/gpu/gpu_beauty_backend.cpp` (line 345 다음)

```cpp
// Luminance Sharpen Uniforms
if (luminance_sharpen_program_ != 0) {
    luminance_sharpen_uniforms_.uTexture = glGetUniformLocation(luminance_sharpen_program_, "uTexture");
    luminance_sharpen_uniforms_.uSkinMask = glGetUniformLocation(luminance_sharpen_program_, "uSkinMask");
    luminance_sharpen_uniforms_.uSharpenAmount = glGetUniformLocation(luminance_sharpen_program_, "uSharpenAmount");
    luminance_sharpen_uniforms_.uTexelSize = glGetUniformLocation(luminance_sharpen_program_, "uTexelSize");
}
```

### 3.6 gpu_beauty_backend.cpp — executeFreqSepPipelineImpl() 수정

**파일**: `cpp/src/gpu/gpu_beauty_backend.cpp` (line 1138~)

**현재 구조**:
```
Pass 1a → temp
Pass 1b → lowFreq
Pass 2a → temp
Pass 2b → smoothedLow
Pass 3  → output_fbo (Composite)
```

**변경 후 구조**:
```
Pass 1a → temp (blur_w × blur_h)
Pass 1b → lowFreq (blur_w × blur_h)
Pass 2a → temp (blur_w × blur_h)
Pass 2b → smoothedLow (blur_w × blur_h)
Pass 3  → compositeRT (width × height, full-res)  ← 변경: output_fbo → compositeRT
Pass 4  → output_fbo (Sharpen)                     ← 추가
```

**핵심 변경**: Pass 3 (Composite)의 출력 대상을 **full-res intermediate RT**로 변경하고, Pass 4에서 이를 읽어 sharpen 후 output_fbo에 쓴다.

⚠️ **temp 재사용 불가 (MID half-res 경로)**: `temp`은 `blur_w × blur_h` (= `width/res_divisor`)로 할당된다. MID 경로(res_divisor=2)에서는 half-res이므로, Composite 직전에 뷰포트를 full-res로 복원한 상태에서 half-res `temp`에 full-res Composite를 그리면 해상도 불일치가 발생한다. 따라서 **full-res compositeRT를 texture_pool에서 별도 할당**해야 한다.

```cpp
// ① sharpen 활성 여부 판정 (compositeRT 할당보다 먼저 선언)
bool sharpen_enabled = (luminance_sharpen_program_ != 0 && params.sharpen_amount > 0.01f);

// ② Sharpen용 full-res intermediate — temp 재사용 불가 (half-res일 수 있음)
TexturePool::TextureInfo* compositeRT = nullptr;
if (sharpen_enabled) {
    compositeRT = texture_pool_->acquireRenderTarget(width, height);  // full-res
    if (!compositeRT) {
        LOGW("FreqSep: Failed to acquire compositeRT, sharpen disabled");
        sharpen_enabled = false;
    }
}
```

```cpp
// Pass 3: Composite (sharpen 활성이면 compositeRT로, 아니면 output_fbo로)
GLuint composite_output_fbo = sharpen_enabled ? compositeRT->fbo_id : output_fbo;

std::snprintf(tag, sizeof(tag), "FreqSep_Composite%s", cfg.composite_profiler_suffix);
if (profiling) profiler_->begin(tag);

glUseProgram(freq_sep_composite_program_);
// ... (기존 uniform 설정 동일) ...
glBindFramebuffer(GL_FRAMEBUFFER, composite_output_fbo);  // ★ 변경
renderFullscreenQuad();
if (profiling) profiler_->end(tag);

// Pass 4: Luminance Sharpen (선택적)
if (sharpen_enabled) {
    std::snprintf(tag, sizeof(tag), "FreqSep_Sharpen%s", cfg.composite_profiler_suffix);
    if (profiling) profiler_->begin(tag);

    glUseProgram(luminance_sharpen_program_);
    glUniform1i(luminance_sharpen_uniforms_.uTexture, 0);
    glUniform1i(luminance_sharpen_uniforms_.uSkinMask, 1);
    glUniform1f(luminance_sharpen_uniforms_.uSharpenAmount, params.sharpen_amount);
    glUniform2f(luminance_sharpen_uniforms_.uTexelSize, 1.0f / width, 1.0f / height);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, compositeRT->texture_id);  // Composite 결과 (full-res)
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, mask_tex);            // skin mask

    glBindFramebuffer(GL_FRAMEBUFFER, output_fbo);
    renderFullscreenQuad();

    if (profiling) profiler_->end(tag);

    // ③ compositeRT 즉시 반환 — 풀 누수 방지
    texture_pool_->releaseTexture(compositeRT);
}
```

**중간 버퍼**: `temp`은 MID 경로에서 half-res(blur_w × blur_h)이므로 Composite 결과(full-res) 저장에 사용 불가. `compositeRT`를 full-res로 별도 할당하며, Sharpen 완료 후 즉시 반환한다. HIGH 경로(res_divisor=1)에서도 동일 로직으로 통일하여 분기를 줄인다.

### 3.7 mapSkinQuality — sharpen_amount 매핑

**파일**: `cpp/src/gpu/gpu_beauty_backend.cpp` (line 981~1015)

```cpp
// Luminance Sharpen
// Council 권장: 0.12~0.18 고정
// skinQuality가 높을수록 blur가 강하므로 sharpen도 약간 증가
p.sharpen_amount = 0.12f + s * 0.06f;  // 0.12 ~ 0.18
```

**설계 근거**:
- `skinQuality` 0.0: `s=0` → `sharpen_amount = 0.12` (최소 선명도 복구)
- `skinQuality` 1.0: `s=1` → `sharpen_amount = 0.18` (강한 blur 보상)
- Council 권장 범위 0.12~0.18 정확히 반영
- 사용자 UI 노출 없음

### 3.8 셰이더 프로그램 해제

ShaderManager에 캐시하므로 **수동 glDeleteProgram 불필요** — `ShaderManager::releaseAll()`에서 일괄 해제된다. 기존 `freq_sep_gaussian_program_`, `freq_sep_composite_program_`과 동일한 수명 관리 정책.

---

## 4. 성능 분석

### 추가 비용

| 항목 | 비용 |
|------|------|
| 추가 패스 | +1 (Sharpen) |
| 텍스처 샘플링 | 6회 (center + 4 neighbor + mask) |
| ALU | ~20 ops (luminance 변환, ratio 계산) |
| Render target | +1 (compositeRT, full-res) — Sharpen 완료 후 즉시 반환 |

### 파이프라인 총 패스 수

| 구성 | 패스 수 |
|------|--------|
| 현재 (Phase 1 이전) | 5 (Gaussian H/V × 2 + Composite) |
| Phase 1 완료 후 | 6 (+ Sharpen) |

### 예상 프레임 시간

| 기기 등급 | 현재 (5 pass) | Phase 1 후 (6 pass) | 차이 |
|----------|-------------|-------------------|----- |
| HIGH (Adreno 7xx) | ~1.5ms | ~2.0ms | +0.5ms |
| MID (Mali-G78) | ~3.0ms | ~3.8ms | +0.8ms |
| LOW | bilateral only | — | — |

→ 30fps (33ms) 기준으로 충분한 여유 (전체 beauty 파이프라인 ~5ms 이내)

### DeviceTier 분기 (선택적)

MID 기기에서 프레임 시간이 부족하면 (`params`는 `const&`이므로 호출 전에 조정):
```cpp
// mapSkinQuality() 내부 또는 호출측에서 조정
if (device_tier_ == DeviceTier::MID) {
    p.sharpen_amount *= 0.5f;  // 약한 sharpen
    // 또는 p.sharpen_amount = 0.0f로 패스 자체를 스킵
}
```

---

## 5. 테스트 계획

### 5.1 기존 테스트 업데이트

`cpp/tests/test_beauty_config_v2.cpp`:
- `FreqSepParams` 기대값에 `sharpen_amount` 범위 검증 추가

### 5.2 신규 테스트

```cpp
TEST(FreqSepParamsTest, SharpenAmountRange) {
    auto p_low = GPUBeautyBackend::mapSkinQuality(0.2f, 300);
    auto p_high = GPUBeautyBackend::mapSkinQuality(1.0f, 300);

    EXPECT_GE(p_low.sharpen_amount, 0.10f);
    EXPECT_LE(p_low.sharpen_amount, 0.20f);
    EXPECT_GE(p_high.sharpen_amount, 0.15f);
    EXPECT_LE(p_high.sharpen_amount, 0.20f);
}

TEST(FreqSepParamsTest, SharpenDisabledAtZeroQuality) {
    auto p = GPUBeautyBackend::mapSkinQuality(0.0f, 300);
    EXPECT_FALSE(p.enabled);  // 전체 FreqSep 비활성 → sharpen도 비활성
}
```

### 5.3 시각적 검증

| 테스트 항목 | 확인 내용 |
|------------|----------|
| 피부 질감 | 보정 후에도 모공/미세 텍스처가 인지됨 |
| 눈/머리카락 | mask 외부는 sharpen 없음 (원본 유지) |
| 과도한 sharpen | halo/ring 아티팩트 없음 (amount ≤ 0.18) |
| 노이즈 증폭 | 색상 노이즈가 증폭되지 않음 (luminance-only) |
| A/B 비교 | sharpen OFF vs ON 시각 비교 |
| 성능 | 6 패스 상태에서 30fps 유지 확인 |

---

## 6. 완료 기준

- [x] LUMINANCE_SHARPEN_FRAGMENT 셰이더 신규 작성
- [x] `luminance_sharpen_program_` 프로그램 멤버 추가
- [x] `LuminanceSharpenUniforms` 구조체 + 멤버 추가
- [x] `FreqSepParams`에 `sharpen_amount` 필드 추가
- [x] `initializeShaders()`에 sharpen 프로그램 컴파일 추가
- [x] Uniform 초기화 코드 추가
- [x] `executeFreqSepPipelineImpl()`에 Pass 4 (Sharpen) 추가
- [x] Composite 출력을 compositeRT (full-res) 버퍼로 변경 (sharpen 활성 시)
- [x] `mapSkinQuality()`에 sharpen_amount 매핑 추가
- [x] ShaderManager에 cacheProgram 등록 (수명 관리 위임)
- [x] FreqSep 관련 테스트 추가 및 통과 (5개 신규 테스트, 전체 67개 통과)
- [ ] Android 디바이스에서 선명도 복구 시각적 확인
- [ ] 6 패스 파이프라인에서 30fps 유지 성능 확인
