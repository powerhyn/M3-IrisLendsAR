# P4-W4-01d: 톤커브 미드톤 리프트

> **상위 문서**: `P4-W4-01_freqsep_quality_improvement.md`
> **상태**: ✅ 완료
> **난이도**: 낮음 | **추가 GPU 비용**: ALU only (~3 ops, 패스 추가 없음)
> **선행 조건**: Step 1~3 완료 후 적용

---

## 1. 왜 필요한가

### 현재 문제

FreqSep 파이프라인은 잡티 감쇠 과정에서 피부의 **미드톤 콘트라스트를 감소**시킨다:

- Gaussian blur 기반 low-frequency 추출이 피부의 자연스러운 밝기 변화를 평탄화
- 고주파 감쇠 후 재합성 시 원본 대비 전반적으로 "침침한" 톤 발생
- 특히 **중간 밝기 영역** (피부 톤의 핵심 범위)에서 콘트라스트 손실이 두드러짐
- 상용 SDK들은 보정 후 미드톤을 미세하게 밝혀 "건강한 피부" 느낌을 유지

### 톤커브 미드톤 리프트의 개선

**이차 곡선 (Quadratic Mid-tone Lift)**:

```
f(x) = x + intensity * x * (1 - x)
```

- `intensity = 0`: 변화 없음 (identity, 바이패스)
- `intensity > 0`: 미드톤을 밝게 (0.5 근처에서 최대 효과)
- `x = 0` 또는 `x = 1`일 때: 항상 원래 값 → **하이라이트/섀도우 보존**
- 최대 리프트: `x = 0.5`에서 `0.5 + intensity * 0.25`

특성:
- **하이라이트 안전**: x=1 근처에서 리프트 자연 감소 → 클램핑 없음
- **섀도우 안전**: x=0 근처에서 리프트 자연 감소 → 어두운 영역 침범 없음
- **연산 비용 최소**: 곱셈 2회 + 덧셈 1회 = 3 ALU ops
- **Linear RGB 친화**: 선형 공간에서 물리적으로 정확한 밝기 조정

### Council 합의 근거

> "미드톤 리프트는 주관적 품질 향상의 핵심. intensity 0.12~0.18 고정으로 충분하며, 슬라이더 노출 불필요" — 다수 합의

---

## 2. 변경 대상

### 2.1 FREQ_SEP_COMPOSITE_FRAGMENT 셰이더

**파일**: `cpp/src/gpu/shader_sources.cpp` (line 460~502)

**추가 위치**: Soft Light 합성 이후, mask 블렌딩 이전

**변경 코드**:
```glsl
    // ★ Step 4: 미드톤 리프트 (beauty에만 적용)
    // f(x) = x + intensity * x * (1 - x)
    // 미드톤(0.5 근처)에서 최대 효과, 하이라이트/섀도우에서 0
    beauty = beauty + uToneLift * beauty * (vec3(1.0) - beauty);
```

**핵심 포인트**:
- `beauty` 변수에 적용 (mask 블렌딩 전)
- `orig`에는 적용하지 않음 — 보정된 영역만 톤 리프트
- mask 블렌딩 `mix(orig, beauty, mask)` 이후에는 적용 불가 (비보정 영역까지 영향)

### 2.2 새 Uniform 추가 (셰이더 내)

```glsl
uniform float uToneLift;  // 미드톤 리프트 강도 (0.0~0.3, 기본 0.15)
```

### 2.3 전체 Composite 셰이더 내 위치 (Step 1~4 적용 후)

```glsl
void main() {
    // ... (텍스처 읽기, Linear 변환, 3-신호 감쇠, Soft Light 합성) ...

    // ★ Step 2: Soft Light 합성
    vec3 blend = clamp(vec3(0.5) + adjusted_high, 0.0, 1.0);
    vec3 beauty = (vec3(1.0) - 2.0 * blend) * smoothLow * smoothLow
                + 2.0 * blend * smoothLow;

    // ★ Step 4: 미드톤 리프트 (Soft Light 결과에 적용)
    beauty = beauty + uToneLift * beauty * (vec3(1.0) - beauty);

    // mask 블렌딩
    vec3 result = mix(orig, beauty, mask);

    // ★ Step 1: Linear → sRGB (음수 방어)
    result = pow(max(result, vec3(0.0)), vec3(1.0 / 2.2));

    fragColor = vec4(result, 1.0);
}
```

### 2.4 gpu_beauty_backend.h — Uniform 추가

**파일**: `cpp/include/iris_sdk/gpu/gpu_beauty_backend.h` (line 472~481)

```cpp
struct FreqSepCompositeUniforms {
    GLint uSmoothedLow = -1;
    GLint uLowFreq = -1;
    GLint uOriginal = -1;
    GLint uSkinMask = -1;
    GLint uHighFreqPreserve = -1;
    GLint uAttenuationLow = -1;
    GLint uAttenuationHigh = -1;
    GLint uEdgeWeight = -1;      // Step 3
    GLint uChromaWeight = -1;    // Step 3
    GLint uToneLift = -1;        // ★ Step 4 추가
} freq_sep_composite_uniforms_;
```

### 2.5 gpu_beauty_backend.h — FreqSepParams 확장

**파일**: `cpp/include/iris_sdk/gpu/gpu_beauty_backend.h` (line 233~240)

```cpp
struct FreqSepParams {
    int blur_radius = 15;
    float high_freq_preserve = 0.45f;
    float low_freq_smooth_radius_ratio = 0.5f;
    float attenuation_low = 0.02f;
    float attenuation_high = 0.15f;
    float edge_weight = 0.5f;      // Step 3
    float chroma_weight = 0.3f;    // Step 3
    float tone_lift = 0.15f;       // ★ Step 4 추가
    bool enabled = false;
};
```

### 2.6 gpu_beauty_backend.cpp — Uniform 초기화

**파일**: `cpp/src/gpu/gpu_beauty_backend.cpp` (line 336~345)

```cpp
if (freq_sep_composite_program_ != 0) {
    // ... 기존 uniform 초기화 ...
    freq_sep_composite_uniforms_.uToneLift = glGetUniformLocation(freq_sep_composite_program_, "uToneLift");  // ★ 추가
}
```

### 2.7 gpu_beauty_backend.cpp — Composite 패스에서 Uniform 설정

**파일**: `cpp/src/gpu/gpu_beauty_backend.cpp` (line 1260~1263)

```cpp
glUseProgram(freq_sep_composite_program_);
// ... 기존 uniform 설정 ...
glUniform1f(freq_sep_composite_uniforms_.uToneLift, params.tone_lift);  // ★ 추가
```

### 2.8 mapSkinQuality — tone_lift 매핑

**파일**: `cpp/src/gpu/gpu_beauty_backend.cpp` (line 981~1015)

```cpp
// 톤커브 미드톤 리프트
// Council 권장: 0.12~0.18 고정, skinQuality에 비례하지 않음
// 다만 skinQuality가 매우 낮으면 리프트도 줄여 자연스러움 유지
p.tone_lift = (s > 0.1f) ? 0.15f : s * 1.5f;  // s≤0.1에서 점진적 진입, 이후 0.15 고정
```

**설계 근거**:
- Council 합의: "intensity 0.12~0.18 고정" → 중간값 0.15 채택
- skinQuality가 매우 낮을 때 (≤0.1): 리프트도 비례적으로 줄여 "보정 안 한 것 같은" 자연스러움
- skinQuality 0.1 이상: 고정 0.15 → 사용자 슬라이더에 노출하지 않음
- 별도 UI 없이 내부 파라미터로만 동작

---

## 3. 수학적 검증

### 3.1 Identity (intensity = 0)
```
f(x) = x + 0 * x * (1-x) = x  ✓
```

### 3.2 최대 리프트 (x = 0.5, intensity = 0.15)
```
f(0.5) = 0.5 + 0.15 * 0.5 * 0.5 = 0.5 + 0.0375 = 0.5375
```
→ +7.5% 밝기 증가 (미드톤에서)

### 3.3 하이라이트 안전 (x = 0.9, intensity = 0.15)
```
f(0.9) = 0.9 + 0.15 * 0.9 * 0.1 = 0.9 + 0.0135 = 0.9135
```
→ +1.5% 밝기 증가 (자연스럽게 감쇠) → 1.0 초과 없음

### 3.4 섀도우 안전 (x = 0.1, intensity = 0.15)
```
f(0.1) = 0.1 + 0.15 * 0.1 * 0.9 = 0.1 + 0.0135 = 0.1135
```
→ +1.35% → 어두운 영역 거의 무영향

### 3.5 출력 범위

`intensity ∈ [0, 0.3]`, `x ∈ [0, 1]`일 때:
- `f(x) = x + i*x*(1-x) = x * (1 + i*(1-x))`
- `1 + i*(1-x) ≥ 1` (항상 양수) → `f(x) ≥ 0`
- 최대: `x=0.5, i=0.3` → `f(0.5) = 0.575` < 1.0 ✓
- `x=1.0`: `f(1) = 1 + i*0 = 1.0` ✓

→ **출력 항상 [0, 1]** (intensity ≤ 1.0이면)

---

## 4. 테스트 계획

### 4.1 기존 테스트 업데이트

`cpp/tests/test_beauty_config_v2.cpp`:
- `FreqSepParams` 기대값에 `tone_lift` 범위 검증 추가

### 4.2 신규 테스트

```cpp
TEST(FreqSepParamsTest, ToneLiftFixedValue) {
    auto p = GPUBeautyBackend::mapSkinQuality(0.5f, 300);
    EXPECT_NEAR(p.tone_lift, 0.15f, 0.01f);  // 0.1 이상에서 고정 0.15
}

TEST(FreqSepParamsTest, ToneLiftGradualAtLowQuality) {
    auto p = GPUBeautyBackend::mapSkinQuality(0.05f, 300);
    EXPECT_LT(p.tone_lift, 0.15f);  // 매우 낮은 quality에서 줄어듦
    EXPECT_GE(p.tone_lift, 0.0f);
}
```

### 4.3 시각적 검증

| 테스트 항목 | 확인 내용 |
|------------|----------|
| 미드톤 밝기 | 보정 후 피부가 "침침하지 않음" 확인 |
| 하이라이트 보존 | 밝은 영역이 번지지 않음 (오버클램핑 없음) |
| 섀도우 보존 | 어두운 영역이 과도하게 밝아지지 않음 |
| A/B 비교 | tone_lift=0 vs 0.15 시각 비교 |
| skinQuality 전 범위 | 0.2/0.5/1.0에서 자연스러움 |

---

## 5. 완료 기준

- [x] FREQ_SEP_COMPOSITE_FRAGMENT에 미드톤 리프트 수식 추가
- [x] `uToneLift` uniform 셰이더에 선언
- [x] `FreqSepCompositeUniforms`에 `uToneLift` 멤버 추가
- [x] `FreqSepParams`에 `tone_lift` 필드 추가
- [x] Uniform 초기화 + Composite 패스 설정 추가
- [x] `mapSkinQuality()`에 tone_lift 매핑 추가
- [x] 테스트 추가 및 통과 (5개 테스트, 전체 61개 PASS)
- [ ] Android 디바이스에서 미드톤 개선 시각적 확인
