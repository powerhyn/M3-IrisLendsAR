# P4-W4-01b: Soft Light 합성 전환

> **상위 문서**: `P4-W4-01_freqsep_quality_improvement.md`
> **상태**: ⏳ 대기
> **난이도**: 낮음 | **추가 GPU 비용**: ALU only (패스 추가 없음)
> **선행 조건**: Step 1 (Linear RGB) 완료 후 적용 — Linear 공간에서 Soft Light 수식이 정확

---

## 1. 왜 필요한가

### 현재 문제

현재 Composite 셰이더는 **Additive 합성**을 사용한다:

```glsl
vec3 beauty = smoothLow + adjusted_high;
```

이 방식의 문제점:
- 밝은 영역에서 `smoothLow + high` 값이 1.0을 초과하여 **클램핑 아티팩트** 발생
- 어두운 영역에서 고주파 성분의 영향이 밝은 영역 대비 약하게 작용 → **톤 불균형**
- 피부 전체의 콘트라스트가 평탄해져 "플라스틱" 느낌 유발
- 고주파를 더하는 구조 자체가 원본의 밝기 범위를 보존하지 않음

### Soft Light 합성의 개선

**Soft Light** (Pegtop variant)는 Photoshop 레이어 블렌딩에서 검증된 방식:

```
SoftLight(base, blend) = (1 - 2*blend) * base² + 2 * blend * base
```

- `blend > 0.5`: base를 밝게 (Screen 방향)
- `blend < 0.5`: base를 어둡게 (Multiply 방향)
- `blend = 0.5`: **변화 없음** (identity) — 고주파의 neutral point가 자연스러움

이점:
- **출력 범위 보존**: 결과가 항상 [0, 1] 범위 내 → 클램핑 불필요
- **비선형 톤 보존**: base의 밝기에 비례하여 고주파 영향 스케일링
- **조건 분기 없음**: Pegtop variant는 단일 수식 → GPU에서 Overlay보다 효율적
- **Linear RGB에서 안전**: pow 연산 없이 곱셈/덧셈만으로 구성

### Council 합의 근거

> "Pegtop Soft Light는 8-bit 모바일에서 Overlay 대비 분기가 없고, Linear RGB에서 수학적으로 안정적" — 만장일치

### Overlay vs Soft Light 비교

| 항목 | Overlay | Soft Light (Pegtop) |
|------|---------|---------------------|
| 수식 | `base < 0.5 ? 2*base*blend : 1-2*(1-base)*(1-blend)` | `(1-2*blend)*base²+2*blend*base` |
| 분기 | **있음** (warp divergence 위험) | **없음** |
| 강도 | 더 강함 (콘트라스트 극대화) | 부드러움 (자연스러운 피부 보정) |
| Linear RGB | 분기 threshold가 감마 의존적 | **감마 독립** |
| 8-bit clamp | 오버플로우 가능 | 항상 [0,1] 범위 |

---

## 2. 변경 대상

### 2.1 FREQ_SEP_COMPOSITE_FRAGMENT 셰이더

**파일**: `cpp/src/gpu/shader_sources.cpp` (line 460~502)

**현재 코드** (line 492~495):
```glsl
    vec3 adjusted_high = high * preserve;

    // Re-synthesis
    vec3 beauty = smoothLow + adjusted_high;
```

**변경안**: Additive → Soft Light 합성으로 전환

```glsl
    vec3 adjusted_high = high * preserve;

    // ★ Soft Light 합성 (Pegtop variant)
    // blend = 0.5 + adjusted_high (고주파를 0.5 중심으로 매핑)
    // 고주파가 0이면 blend=0.5 → 변화 없음 (identity)
    // 고주파가 양수면 밝아지고, 음수면 어두워짐
    vec3 blend = vec3(0.5) + adjusted_high;
    blend = clamp(blend, 0.0, 1.0);

    // SoftLight(base, blend) = (1 - 2*blend) * base² + 2 * blend * base
    vec3 beauty = (vec3(1.0) - 2.0 * blend) * smoothLow * smoothLow
                + 2.0 * blend * smoothLow;
```

**핵심 포인트**:
- `adjusted_high`는 부호 있는 값 (양수=밝은 디테일, 음수=어두운 디테일)
- `blend = 0.5 + adjusted_high`로 매핑하면 고주파 0일 때 blend=0.5 → identity
- `clamp`는 극단적 고주파 값에 대한 안전장치 (Linear RGB에서 범위가 넓어질 수 있음)
- `smoothLow`를 base로 사용: 이중 블러된 부드러운 피부톤이 기준

### 2.2 전체 Composite 셰이더 (변경 후 예상 모습)

```glsl
const char* FREQ_SEP_COMPOSITE_FRAGMENT = R"glsl(
#version 310 es
precision highp float;

in vec2 vTexCoord;
out vec4 fragColor;

uniform sampler2D uSmoothedLow;
uniform sampler2D uLowFreq;
uniform sampler2D uOriginal;
uniform sampler2D uSkinMask;

uniform float uHighFreqPreserve;
uniform float uAttenuationLow;
uniform float uAttenuationHigh;

void main() {
    vec3 smoothLow = texture(uSmoothedLow, vTexCoord).rgb;
    vec3 low       = texture(uLowFreq, vTexCoord).rgb;
    vec3 orig      = texture(uOriginal, vTexCoord).rgb;
    float mask     = texture(uSkinMask, vec2(vTexCoord.x, 1.0 - vTexCoord.y)).r;

    // High Frequency inline extraction
    vec3 high = orig - low;

    // Y(luminance) based high-freq magnitude
    float magnitude = dot(abs(high), vec3(0.299, 0.587, 0.114));

    // Non-linear attenuation
    float blemishFactor = smoothstep(uAttenuationLow, uAttenuationHigh, magnitude);
    float preserve = mix(uHighFreqPreserve, 1.0, 1.0 - blemishFactor);

    vec3 adjusted_high = high * preserve;

    // ★ Soft Light 합성 (Pegtop variant)
    vec3 blend = clamp(vec3(0.5) + adjusted_high, 0.0, 1.0);
    vec3 beauty = (vec3(1.0) - 2.0 * blend) * smoothLow * smoothLow
                + 2.0 * blend * smoothLow;

    // Blend with original using skin mask
    vec3 result = mix(orig, beauty, mask);

    fragColor = vec4(result, 1.0);
}
)glsl";
```

### 2.3 C++ 측 변경: 없음

Soft Light 전환은 셰이더 내부 ALU 변경만으로 완료됨:
- **Uniform 추가 없음**: 기존 uniform 그대로 사용
- **FreqSepParams 변경 없음**: 파라미터 구조체 동일
- **gpu_beauty_backend.cpp 변경 없음**: Composite 패스 호출 코드 동일
- **gpu_beauty_backend.h 변경 없음**: 구조체 변경 없음

### 2.4 mapSkinQuality 재튜닝 (선택적)

Soft Light는 Additive 대비 고주파 적용 강도가 부드러워지므로,
`high_freq_preserve` 범위를 미세 조정할 수 있음:

```cpp
// Additive 시절: 1.0 → 0.35 (65% 감쇠)
// Soft Light 전환 후: 1.0 → 0.30 (70% 감쇠) — 블렌딩이 부드러우므로 더 강하게 감쇠 가능
p.high_freq_preserve = 1.0f - s * 0.70f;
```

⚠️ **이 값은 Step 1 (Linear RGB) 적용 후 실측 기반으로 결정** — 두 Step이 동시에 적용되므로 개별 튜닝 불가.

---

## 3. Soft Light 수학적 검증

### 3.1 Identity 조건 (고주파 = 0)

```
adjusted_high = vec3(0.0)
blend = vec3(0.5)
beauty = (1 - 2*0.5) * base² + 2 * 0.5 * base
       = 0 * base² + base
       = base  ✓  (smoothLow 그대로 출력)
```

### 3.2 밝은 디테일 (고주파 > 0)

```
adjusted_high = vec3(0.1)  // 밝은 디테일
blend = vec3(0.6)
beauty = (1 - 1.2) * base² + 1.2 * base
       = -0.2 * base² + 1.2 * base
       = base * (1.2 - 0.2 * base)
```
→ base=0.5일 때: `0.5 * 1.1 = 0.55` (밝아짐)
→ base=0.8일 때: `0.8 * 1.04 = 0.832` (더 밝지만 완만하게)

### 3.3 어두운 디테일 (고주파 < 0)

```
adjusted_high = vec3(-0.1)  // 어두운 디테일 (모공, 주름)
blend = vec3(0.4)
beauty = (1 - 0.8) * base² + 0.8 * base
       = 0.2 * base² + 0.8 * base
       = base * (0.8 + 0.2 * base)
```
→ base=0.5일 때: `0.5 * 0.9 = 0.45` (어두워짐)
→ 항상 0 이상, base 이하 → **안전**

### 3.4 출력 범위 보장

Pegtop Soft Light에서 base ∈ [0,1], blend ∈ [0,1]이면:
- `(1-2b)*a² + 2b*a = a * ((1-2b)*a + 2b) = a * (a + 2b*(1-a))`
- `a ≥ 0`이고 `a + 2b*(1-a) ≥ 0`이므로 결과 ≥ 0
- `a + 2b*(1-a) ≤ 1 + 2(1-a) ≤ 1`일 때... 정확히는 `a*(a+2b-2ab) ≤ 1` (b=1,a=1일때 1)
- ∴ **출력 항상 [0, 1]** → clamp 불필요

---

## 4. 테스트 계획

### 4.1 기존 테스트 업데이트

- `cpp/tests/test_beauty_config_v2.cpp`의 FreqSepParamsTest는 파라미터 매핑 테스트이므로
  Soft Light 전환과 무관 (high_freq_preserve 범위만 재튜닝 시 업데이트)

### 4.2 시각적 검증

| 테스트 항목 | Additive (현재) | Soft Light (변경 후) |
|------------|----------------|---------------------|
| skinQuality 0.2 | 미세 보정 | 동등 이상 자연스러움 |
| skinQuality 0.5 | 잡티 제거 | 톤 보존 개선 확인 |
| skinQuality 1.0 | 플라스틱 경향 | **플라스틱 감소** (핵심) |
| 밝은 피부 하이라이트 | 클램핑 가능 | [0,1] 범위 보장 |
| 어두운 피부 | 톤 불균형 | 비선형 스케일링으로 개선 |

### 4.3 성능 확인

- ALU 연산 차이: `base + high` (1 add) → `(1-2b)*a²+2b*a` (2 mul + 2 mad)
- 차이: ~2~3 ALU ops 추가, **프레임 시간 영향 무시 가능** (texfetch 대비)
- 중간급 기기 (Mali-G78 등)에서 30fps 유지 확인

---

## 5. 완료 기준

- [ ] FREQ_SEP_COMPOSITE_FRAGMENT에서 Additive → Soft Light 전환
- [ ] `blend = clamp(0.5 + adjusted_high, 0, 1)` 매핑 구현
- [ ] high_freq_preserve 재튜닝 (Step 1과 함께)
- [ ] Android 디바이스에서 플라스틱 감소 시각적 확인
- [ ] skinQuality 0.2/0.5/1.0 전 범위 자연스러움 확인
