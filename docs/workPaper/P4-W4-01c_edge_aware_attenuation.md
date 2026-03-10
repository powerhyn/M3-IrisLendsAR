# P4-W4-01c: Edge-aware Attenuation (3-신호 결합)

> **상위 문서**: `P4-W4-01_freqsep_quality_improvement.md`
> **상태**: ✅ 완료
> **난이도**: 중간 | **추가 GPU 비용**: ALU + 추가 텍스처 샘플링 4회 (패스 추가 없음)
> **선행 조건**: Step 1 (Linear RGB), Step 2 (Soft Light) 완료 후 적용

---

## 1. 왜 필요한가

### 현재 문제

현재 Composite 셰이더의 감쇠 로직은 **magnitude 단일 신호**에 의존한다:

```glsl
float magnitude = dot(abs(high), vec3(0.299, 0.587, 0.114));
float blemishFactor = smoothstep(uAttenuationLow, uAttenuationHigh, magnitude);
```

이 방식의 구조적 한계:

| 시나리오 | magnitude | 실제 의도 | 현재 동작 | 문제 |
|---------|-----------|----------|----------|------|
| **잡티** (점, 여드름) | 높음 | 강하게 감쇠 | ✅ 감쇠 | 정상 |
| **피부 결** (미세 텍스처) | 낮음 | 보존 | ✅ 보존 | 정상 |
| **주름** (직선적, 중간 진폭) | 중간 | 보존 | ❌ 부분 감쇠 | **주름 오삭제** |
| **색소침착** (넓고 낮은 진폭) | 낮음 | 감쇠 | ❌ 보존 | **색소침착 미검출** |
| **눈/입술 경계** (강한 에지) | 높음 | 보존 | ❌ 감쇠 | **에지 흐림** |

### 3-신호 결합의 개선

세 가지 독립 신호를 결합하여 잡티/구조물을 정확히 구분:

1. **Magnitude** (기존): 고주파 크기 — 큰 변화 감지
2. **Edge Gradient**: Sobel/중심차분 기반 엣지 강도 — 구조적 경계 보존
3. **Chroma Deviation**: 원본과 블러 간의 색상 차이 — 색소침착 감지

**결합 공식**:
```
blemishScore = magnitude * (1.0 - edgeWeight * edgeStrength) * (1.0 + chromaWeight * chromaDev)
```

- 에지가 강할수록 blemishScore 감소 → **주름/경계 보존**
- 색상 편차가 클수록 blemishScore 증가 → **색소침착 감쇠**
- magnitude는 기존 역할 유지 → **하위 호환성 보장**

### Council 합의 근거

> "magnitude만으로는 주름과 잡티 구별 불가. gradient + chroma 2신호를 추가하면 Phase 1에서 실용적 수준의 edge-awareness 확보 가능" — Codex/Gemini/Claude 만장일치

---

## 2. 변경 대상

### 2.1 FREQ_SEP_COMPOSITE_FRAGMENT 셰이더

**파일**: `cpp/src/gpu/shader_sources.cpp` (line 460~502)

**현재 코드** (line 482~491):
```glsl
    // High Frequency inline extraction (ALU operation, no separate pass/texture)
    vec3 high = orig - low;

    // Y(luminance) based high-freq magnitude
    // ⚠️ Rec.601 계수 — Step 3 구현 시 Rec.709 (0.2126, 0.7152, 0.0722)로 교체
    float magnitude = dot(abs(high), vec3(0.299, 0.587, 0.114));

    // Non-linear attenuation: large changes (blemishes) → strong attenuation
    float blemishFactor = smoothstep(uAttenuationLow, uAttenuationHigh, magnitude);
    float preserve = mix(uHighFreqPreserve, 1.0, 1.0 - blemishFactor);
```

**변경안**: 3-신호 결합으로 확장

```glsl
    // --- 신호 1: Magnitude (기존) ---
    const vec3 LUMA_709 = vec3(0.2126, 0.7152, 0.0722);  // Linear-light Rec.709
    vec3 high = orig - low;
    float magnitude = dot(abs(high), LUMA_709);

    // --- 신호 2: Edge Gradient (중심차분) ---
    // 주변 4 텍셀 샘플링으로 로컬 에지 강도 계산
    vec2 texelSize = vec2(1.0) / vec2(textureSize(uOriginal, 0));
    float lumC = dot(orig, LUMA_709);
    float lumL = dot(texture(uOriginal, vTexCoord - vec2(texelSize.x, 0.0)).rgb,
                     LUMA_709);
    float lumR = dot(texture(uOriginal, vTexCoord + vec2(texelSize.x, 0.0)).rgb,
                     LUMA_709);
    float lumU = dot(texture(uOriginal, vTexCoord - vec2(0.0, texelSize.y)).rgb,
                     LUMA_709);
    float lumD = dot(texture(uOriginal, vTexCoord + vec2(0.0, texelSize.y)).rgb,
                     LUMA_709);
    float gx = lumR - lumL;
    float gy = lumD - lumU;
    float edgeStrength = sqrt(gx * gx + gy * gy);

    // --- 신호 3: Chroma Deviation (YCbCr UV 분리) ---
    // 원본과 블러 간의 색차(Cb/Cr) 성분만 추출
    // 색소침착은 luminance 차이는 작지만 chrominance 차이가 큼
    //
    // ⚠️ 이전 방식 `length(diff) - magnitude`는 잘못된 색차 분리:
    //   diff=(0.1,0.1,0.1) 같은 순수 명도 변화에서도 chromaDev > 0이 됨
    //   (length(0.1,0.1,0.1)=0.173 vs magnitude=dot(0.1,0.1,0.1, luma)=0.1 → 0.073)
    //   neutral contrast를 색소침착으로 오검출하는 문제.
    //
    // 수정: luminance projection을 제거하여 순수 색차 성분만 추출
    vec3 diff = orig - low;
    float lumDiff = dot(diff, vec3(0.2126, 0.7152, 0.0722));  // Linear-light Rec.709
    vec3 chromaDiff = diff - vec3(lumDiff);  // luminance 성분 제거 → 순수 색차
    float chromaDev = length(chromaDiff);

    // --- 3-신호 결합 ---
    float blemishScore = magnitude
                       * (1.0 - uEdgeWeight * clamp(edgeStrength * 5.0, 0.0, 1.0))
                       * (1.0 + uChromaWeight * clamp(chromaDev * 10.0, 0.0, 1.0));

    float blemishFactor = smoothstep(uAttenuationLow, uAttenuationHigh, blemishScore);
    float preserve = mix(uHighFreqPreserve, 1.0, 1.0 - blemishFactor);
```

**설계 결정**:
- `edgeStrength * 5.0`: 에지 값을 [0, 1] 범위로 정규화하는 스케일 팩터 (Linear RGB 기준)
- `chromaDev * 10.0`: 색상 편차를 [0, 1] 범위로 정규화하는 스케일 팩터
- 이 스케일 팩터들은 실측 후 조정 가능 (uniform으로 노출 가능하지만, Phase 1에서는 상수로 고정)
- `clamp(…, 0, 1)`: 정규화된 값의 안전 범위 보장

### 2.2 새 Uniform 추가 (셰이더 내)

```glsl
uniform float uEdgeWeight;     // 에지 보존 강도 (0.0~1.0, 기본 0.5)
uniform float uChromaWeight;   // 색소침착 감지 강도 (0.0~1.0, 기본 0.3)
```

### 2.3 전체 Composite 셰이더 (변경 후 예상 모습)

Step 1 (Linear RGB) + Step 2 (Soft Light) + Step 3 (Edge-aware) 모두 적용된 최종 형태:

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
uniform float uEdgeWeight;       // ★ Step 3 추가
uniform float uChromaWeight;     // ★ Step 3 추가

void main() {
    vec3 smoothLow = texture(uSmoothedLow, vTexCoord).rgb;  // 이미 linear
    vec3 low       = texture(uLowFreq, vTexCoord).rgb;      // 이미 linear
    vec3 orig      = texture(uOriginal, vTexCoord).rgb;
    float mask     = texture(uSkinMask, vec2(vTexCoord.x, 1.0 - vTexCoord.y)).r;

    // ★ Step 1: sRGB → Linear (original만)
    orig = pow(orig, vec3(2.2));

    // --- 신호 1: Magnitude ---
    // Linear-light 공간에서는 Rec.709 계수 사용 (Rec.601 0.299/0.587/0.114는 sRGB 감마용)
    const vec3 LUMA_709 = vec3(0.2126, 0.7152, 0.0722);
    vec3 high = orig - low;
    float magnitude = dot(abs(high), LUMA_709);

    // --- 신호 2: Edge Gradient (★ Step 3) ---
    vec2 texelSize = vec2(1.0) / vec2(textureSize(uOriginal, 0));
    // 주의: 에지 계산도 linear 공간에서 수행 (orig을 이미 linearize했으므로)
    // 직접 uOriginal 텍스처를 다시 읽으면 sRGB이므로, linearize된 orig 기준으로 중심차분
    // → 텍스처 추가 샘플링 필요 (linear 변환 포함)
    vec3 sL = pow(texture(uOriginal, vTexCoord - vec2(texelSize.x, 0.0)).rgb, vec3(2.2));
    vec3 sR = pow(texture(uOriginal, vTexCoord + vec2(texelSize.x, 0.0)).rgb, vec3(2.2));
    vec3 sU = pow(texture(uOriginal, vTexCoord - vec2(0.0, texelSize.y)).rgb, vec3(2.2));
    vec3 sD = pow(texture(uOriginal, vTexCoord + vec2(0.0, texelSize.y)).rgb, vec3(2.2));
    float lumL = dot(sL, LUMA_709);
    float lumR = dot(sR, LUMA_709);
    float lumU = dot(sU, LUMA_709);
    float lumD = dot(sD, LUMA_709);
    float gx = lumR - lumL;
    float gy = lumD - lumU;
    float edgeStrength = sqrt(gx * gx + gy * gy);

    // --- 신호 3: Chroma Deviation (★ Step 3, YCbCr UV 분리) ---
    vec3 diff = orig - low;
    float lumDiff = dot(diff, vec3(0.2126, 0.7152, 0.0722));
    vec3 chromaDiff = diff - vec3(lumDiff);  // luminance 성분 제거
    float chromaDev = length(chromaDiff);

    // --- 3-신호 결합 ---
    float blemishScore = magnitude
                       * (1.0 - uEdgeWeight * clamp(edgeStrength * 5.0, 0.0, 1.0))
                       * (1.0 + uChromaWeight * clamp(chromaDev * 10.0, 0.0, 1.0));

    float blemishFactor = smoothstep(uAttenuationLow, uAttenuationHigh, blemishScore);
    float preserve = mix(uHighFreqPreserve, 1.0, 1.0 - blemishFactor);

    vec3 adjusted_high = high * preserve;

    // ★ Step 2: Soft Light 합성
    vec3 blend = clamp(vec3(0.5) + adjusted_high, 0.0, 1.0);
    vec3 beauty = (vec3(1.0) - 2.0 * blend) * smoothLow * smoothLow
                + 2.0 * blend * smoothLow;

    vec3 result = mix(orig, beauty, mask);

    // ★ Step 1: Linear → sRGB (음수 방어)
    result = pow(max(result, vec3(0.0)), vec3(1.0 / 2.2));

    fragColor = vec4(result, 1.0);
}
)glsl";
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
    GLint uEdgeWeight = -1;      // ★ 추가
    GLint uChromaWeight = -1;    // ★ 추가
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
    float edge_weight = 0.5f;      // ★ 추가: 에지 보존 강도
    float chroma_weight = 0.3f;    // ★ 추가: 색소침착 감지 강도
    bool enabled = false;
};
```

### 2.6 gpu_beauty_backend.cpp — Uniform 초기화

**파일**: `cpp/src/gpu/gpu_beauty_backend.cpp` (line 336~345)

```cpp
if (freq_sep_composite_program_ != 0) {
    freq_sep_composite_uniforms_.uSmoothedLow = glGetUniformLocation(freq_sep_composite_program_, "uSmoothedLow");
    freq_sep_composite_uniforms_.uLowFreq = glGetUniformLocation(freq_sep_composite_program_, "uLowFreq");
    freq_sep_composite_uniforms_.uOriginal = glGetUniformLocation(freq_sep_composite_program_, "uOriginal");
    freq_sep_composite_uniforms_.uSkinMask = glGetUniformLocation(freq_sep_composite_program_, "uSkinMask");
    freq_sep_composite_uniforms_.uHighFreqPreserve = glGetUniformLocation(freq_sep_composite_program_, "uHighFreqPreserve");
    freq_sep_composite_uniforms_.uAttenuationLow = glGetUniformLocation(freq_sep_composite_program_, "uAttenuationLow");
    freq_sep_composite_uniforms_.uAttenuationHigh = glGetUniformLocation(freq_sep_composite_program_, "uAttenuationHigh");
    freq_sep_composite_uniforms_.uEdgeWeight = glGetUniformLocation(freq_sep_composite_program_, "uEdgeWeight");        // ★ 추가
    freq_sep_composite_uniforms_.uChromaWeight = glGetUniformLocation(freq_sep_composite_program_, "uChromaWeight");    // ★ 추가
}
```

### 2.7 gpu_beauty_backend.cpp — Composite 패스에서 Uniform 설정

**파일**: `cpp/src/gpu/gpu_beauty_backend.cpp` (line 1260~1263)

```cpp
glUseProgram(freq_sep_composite_program_);
glUniform1f(freq_sep_composite_uniforms_.uHighFreqPreserve, params.high_freq_preserve);
glUniform1f(freq_sep_composite_uniforms_.uAttenuationLow, params.attenuation_low);
glUniform1f(freq_sep_composite_uniforms_.uAttenuationHigh, params.attenuation_high);
glUniform1f(freq_sep_composite_uniforms_.uEdgeWeight, params.edge_weight);      // ★ 추가
glUniform1f(freq_sep_composite_uniforms_.uChromaWeight, params.chroma_weight);  // ★ 추가
```

### 2.8 mapSkinQuality — edge_weight / chroma_weight 매핑

**파일**: `cpp/src/gpu/gpu_beauty_backend.cpp` (line 981~1015)

```cpp
// Edge-aware attenuation weights
// skinQuality가 높을수록 edge 보존 + chroma 감지 활성화
p.edge_weight = 0.3f + s * 0.4f;     // 0.3 ~ 0.7
p.chroma_weight = 0.2f + s * 0.3f;   // 0.2 ~ 0.5
```

**설계 근거**:
- `edge_weight`: 낮은 보정에서도 기본적인 에지 보존 (0.3), 강한 보정에서 더 적극적 보존 (0.7)
- `chroma_weight`: 색소침착은 강한 보정에서만 적극 감지 (0.2→0.5)
- 두 값 모두 smoothstep `s` 기반으로 자연스러운 전이

⚠️ **이 값들은 실측 기반 최종 조정 필수** — 특히 `edgeStrength * 5.0`과 `chromaDev * 10.0`의 정규화 스케일 팩터와 함께 조정.

---

## 3. 성능 분석

### 추가 비용

| 항목 | 비용 |
|------|------|
| 텍스처 샘플링 | +4회 (상하좌우 이웃 텍셀) |
| pow() 호출 | +4회 (Linear 변환, Step 1 적용 시) |
| ALU | +~15 ops (gradient, chroma, 결합) |
| 패스 추가 | 없음 |

### 최적화 옵션

**옵션 A: pow() 절약** — 이웃 텍셀의 sRGB→Linear 변환을 근사:
```glsl
// pow(x, 2.2) 대신 gamma 2.0 근사 (s*s)
vec3 sL = texture(uOriginal, vTexCoord - vec2(texelSize.x, 0.0)).rgb;
sL = sL * sL;  // gamma 2.0 근사, 오차 ~5%
```
→ 에지 검출 정확도보다 성능이 중요한 경우 허용

**옵션 B: 에지 검출 전용 luminance** — pow 없이 sRGB luminance 사용:
```glsl
// sRGB 공간에서의 에지도 충분히 유효 (상대적 차이가 중요)
float lumL = dot(texture(uOriginal, vTexCoord - vec2(texelSize.x, 0.0)).rgb,
                 vec3(0.299, 0.587, 0.114));
```
→ 에지 **상대적 크기**만 필요하므로 sRGB에서도 유효. **권장 옵션**.

**옵션 C: DeviceTier 분기** — LOW 기기에서 3-신호 비활성화 (`mapSkinQuality()` 내부에서 조정):
```cpp
// mapSkinQuality() 내에서 — params는 const&로 전달되므로 호출 전에 설정
if (device_tier_ == DeviceTier::LOW) {
    p.edge_weight = 0.0f;
    p.chroma_weight = 0.0f;
}
```

### 예상 프레임 시간 영향

| 기기 등급 | 현재 Composite | 3-신호 추가 후 | 차이 |
|----------|---------------|---------------|------|
| HIGH (Adreno 7xx) | ~0.3ms | ~0.5ms | +0.2ms |
| MID (Mali-G78) | ~0.8ms | ~1.2ms | +0.4ms |
| LOW | FreqSep 미사용 | — | — |

→ 30fps (33ms) 기준으로 충분한 여유

---

## 4. 테스트 계획

### 4.1 기존 테스트 업데이트

`cpp/tests/test_beauty_config_v2.cpp`:
- `FreqSepParams` 생성 후 `edge_weight`, `chroma_weight` 범위 검증 추가
- `AttenuationRangeValid` 테스트: 3-신호 weight 범위 추가

### 4.2 신규 테스트

```cpp
TEST(FreqSepParamsTest, EdgeWeightRange) {
    auto p = GPUBeautyBackend::mapSkinQuality(1.0f, 300);
    EXPECT_GE(p.edge_weight, 0.0f);
    EXPECT_LE(p.edge_weight, 1.0f);
}

TEST(FreqSepParamsTest, ChromaWeightRange) {
    auto p = GPUBeautyBackend::mapSkinQuality(1.0f, 300);
    EXPECT_GE(p.chroma_weight, 0.0f);
    EXPECT_LE(p.chroma_weight, 1.0f);
}

TEST(FreqSepParamsTest, EdgeChromaZeroAtLowQuality) {
    auto p = GPUBeautyBackend::mapSkinQuality(0.1f, 300);
    // 낮은 quality에서도 기본 edge 보존 있음
    EXPECT_GE(p.edge_weight, 0.2f);
}
```

### 4.3 시각적 검증

| 테스트 항목 | 확인 내용 |
|------------|----------|
| 주름 보존 | 이마/눈가 주름이 삭제되지 않고 자연스럽게 보존 |
| 잡티 제거 | 여드름/점 등 잡티는 여전히 효과적으로 감쇠 |
| 색소침착 | 넓은 기미/잡티가 기존보다 더 감쇠되는지 확인 |
| 눈/입술 경계 | 에지 흐림이 기존 대비 감소 |
| 성능 | 중간급 기기에서 30fps 유지 |

---

## 5. 완료 기준

- [x] FREQ_SEP_COMPOSITE_FRAGMENT에 에지 gradient + chroma deviation 계산 추가
- [x] `uEdgeWeight`, `uChromaWeight` uniform 셰이더에 선언
- [x] `FreqSepCompositeUniforms`에 2개 멤버 추가
- [x] `FreqSepParams`에 `edge_weight`, `chroma_weight` 필드 추가
- [x] `gpu_beauty_backend.cpp` Uniform 초기화 + Composite 패스 설정 추가
- [x] `mapSkinQuality()`에 edge/chroma weight 매핑 추가
- [x] FreqSep 관련 테스트 추가 및 통과 (4개 신규 테스트, 17개 전체 통과)
- [ ] Android 디바이스에서 주름 보존 + 잡티 감쇠 시각적 확인

---

## 6. 알려진 이슈 (Known Issues)

### ~~KI-1: Edge Gradient의 sRGB/Linear 색공간 불일치~~ [해결됨]

**해결 방법**: 옵션 B 적용 — `sample * sample` (gamma 2.0 근사 linearize)
- 4-neighbor 텍스처 샘플을 `sR * sR`로 근사 linearize 후 Rec.709 계수 적용
- `pow(x, 2.2)` 대비 오차 ~5%, 4×pow 절약 유지
- 3-신호 모두 linear(근사) 공간에서 계산되어 색공간 통일 달성
- LUMA_601 상수 삭제 (더 이상 sRGB 공간 에지 검출 미사용)
