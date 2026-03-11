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
const char* LUMINANCE_SHARPEN_FRAGMENT = R"glsl(
#version 310 es
precision highp float;

in vec2 vTexCoord;
out vec4 fragColor;

uniform sampler2D uTexture;      // Composite 결과 (beauty)
uniform sampler2D uSkinMask;     // ROI mask
uniform float uSharpenAmount;    // 샤프닝 강도 (0.0~0.5, 기본 0.15)
uniform vec2 uTexelSize;         // (1/width, 1/height)

void main() {
    vec3 center = texture(uTexture, vTexCoord).rgb;
    float mask = texture(uSkinMask, vec2(vTexCoord.x, 1.0 - vTexCoord.y)).r;

    const vec3 LUMA_709 = vec3(0.2126, 0.7152, 0.0722);
    float lumCenter = dot(center, LUMA_709);

    // 인접 픽셀의 mask 값 샘플링 (Y-flip 적용)
    float maskL = texture(uSkinMask, vec2(vTexCoord.x - uTexelSize.x, 1.0 - vTexCoord.y)).r;
    float maskR = texture(uSkinMask, vec2(vTexCoord.x + uTexelSize.x, 1.0 - vTexCoord.y)).r;
    float maskU = texture(uSkinMask, vec2(vTexCoord.x, 1.0 - (vTexCoord.y - uTexelSize.y))).r;
    float maskD = texture(uSkinMask, vec2(vTexCoord.x, 1.0 - (vTexCoord.y + uTexelSize.y))).r;

    // 인접 luminance 샘플링
    float rawLumL = dot(texture(uTexture, vTexCoord - vec2(uTexelSize.x, 0.0)).rgb, LUMA_709);
    float rawLumR = dot(texture(uTexture, vTexCoord + vec2(uTexelSize.x, 0.0)).rgb, LUMA_709);
    float rawLumU = dot(texture(uTexture, vTexCoord - vec2(0.0, uTexelSize.y)).rgb, LUMA_709);
    float rawLumD = dot(texture(uTexture, vTexCoord + vec2(0.0, uTexelSize.y)).rgb, LUMA_709);

    // mask가 0인(비피부) 인접 픽셀은 center luminance로 대체하여
    // composite 경계의 합성 에지가 unsharp mask에 반응하지 않도록 함
    float lumL = mix(lumCenter, rawLumL, maskL);
    float lumR = mix(lumCenter, rawLumR, maskR);
    float lumU = mix(lumCenter, rawLumU, maskU);
    float lumD = mix(lumCenter, rawLumD, maskD);

    float lumBlur = (lumCenter * 2.0 + lumL + lumR + lumU + lumD) / 6.0;

    float lumSharp = lumCenter + uSharpenAmount * (lumCenter - lumBlur);
    lumSharp = clamp(lumSharp, 0.0, 1.0);

    float ratio = (lumCenter > 0.001) ? min(lumSharp / lumCenter, 2.0) : 1.0;
    vec3 sharpened = center * ratio;
    sharpened = clamp(sharpened, 0.0, 1.0);

    vec3 result = mix(center, sharpened, mask);

    fragColor = vec4(result, 1.0);
}
)glsl";
```

**설계 결정**:
- **4-neighbor** (cross) 패턴: 대각 텍셀 생략으로 4 샘플
- **Luminance 비율 방식**: `ratio = sharpLum / origLum` → RGB에 곱하면 색상 보존
- **mask 경계 halo 방지**: 인접 픽셀의 mask 값을 추가 샘플링하여 비피부(mask=0) 방향은 `lumCenter`로 대체. Composite가 `mix(orig, beauty, mask)`로 합성한 경계에서 인위적 에지가 unsharp mask에 반응하지 않도록 함
- **ratio 상한**: `min(ratio, 2.0)`으로 극단적 luminance 비율 제한
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

**temp 조기 릴리스 + compositeRT 할당**: `temp`은 Pass 2b 이후 미사용이므로, compositeRT 할당 전에 먼저 릴리스하여 풀 슬롯을 확보한다. 이렇게 하면 최대 동시 RT 수가 3(lowFreq + smoothedLow + compositeRT)으로, 풀 한도(4) 이내에서 안전하게 동작한다. `onMemoryPressure()`로 풀이 축소되어도 sharpen이 탈락하지 않는다.

> **이전 구현의 회귀 (수정됨)**: 초기 구현에서는 temp을 함수 끝에서 릴리스하면서 compositeRT를 추가 할당하여, 동시 RT 수가 4(temp + lowFreq + smoothedLow + compositeRT)로 풀 한도를 꽉 채웠다. `onMemoryPressure()` 후 풀이 축소되면 `acquireRenderTarget` 실패 → sharpen 자동 비활성화라는 기능 회귀가 발생했다.

```cpp
// Pass 2b 완료 후 temp 조기 릴리스 — compositeRT 할당 시 풀 슬롯 확보
texture_pool_->releaseTexture(temp);
temp = nullptr;

// ① sharpen 활성 여부 판정
bool sharpen_enabled = (luminance_sharpen_program_ != 0 && params.sharpen_amount > 0.01f);

// ② full-res compositeRT 할당 (temp 해제 후이므로 풀 슬롯 여유 있음)
TexturePool::TextureInfo* compositeRT = nullptr;
if (sharpen_enabled) {
    compositeRT = texture_pool_->acquireRenderTarget(width, height);
    if (!compositeRT) {
        LOGW("FreqSep: Failed to acquire compositeRT, sharpen disabled");
        sharpen_enabled = false;
    }
}
```

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
| 텍스처 샘플링 | 10회 (center + 4 neighbor + center mask + 4 neighbor mask) |
| ALU | ~30 ops (luminance 변환, mask mix, ratio 계산) |
| Render target | +1 (compositeRT, full-res) — temp 조기 릴리스 후 할당, Sharpen 완료 후 즉시 반환 |

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

### 5.2 매핑 테스트

- `SharpenAmountMidQuality` — skinQuality 0.5에서 sharpen_amount ≈ 0.15 확인
- `SharpenAmountMaxQuality` — skinQuality 1.0에서 sharpen_amount ≈ 0.18 확인
- `SharpenAmountDisabledWhenZero` — skinQuality 0.0에서 enabled=false 확인
- `SharpenAmountRange` — 전 구간에서 0.12~0.18 범위 내 확인
- `SharpenAmountMonotonicallyIncreases` — skinQuality 증가 시 단조 증가

### 5.3 회귀 테스트

- `MaxConcurrentRenderTargetsWithinPoolLimit` — temp 조기 릴리스 후 최대 동시 RT 수가 풀 한도(4) 이내인지 정적 검증. 이전 구현에서 발생한 onMemoryPressure 후 sharpen 탈락 회귀 방지.
- `MaskBoundaryNeighborReplacement` — 비피부(mask=0) 인접 픽셀이 lumCenter로 대체되는 수식 검증
- `MaskBoundaryNoHaloWhenAllNeighborsNonSkin` — 모든 인접이 비피부일 때 high_freq=0 → sharpen 없음 확인
- `FullSkinRegionSharpensNormally` — 피부(mask=1) 인접에서 원래 luminance가 그대로 사용되는지 확인

### 5.4 시각적 검증

| 테스트 항목 | 확인 내용 |
|------------|----------|
| 피부 질감 | 보정 후에도 모공/미세 텍스처가 인지됨 |
| 눈/머리카락 | mask 외부는 sharpen 없음 (원본 유지) |
| mask 경계 | 피부↔비피부 경계에서 halo/ringing 없음 (인접 mask 가중 적용) |
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
- [x] FreqSep 매핑 테스트 추가 및 통과 (5개)
- [x] RT 풀 사용량 회귀 테스트 추가 (temp 조기 릴리스 검증)
- [x] Mask 경계 halo 방지 수식 회귀 테스트 추가 (3개)
- [ ] Android 디바이스에서 선명도 복구 시각적 확인
- [ ] 6 패스 파이프라인에서 30fps 유지 성능 확인
