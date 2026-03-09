# P4-W4-01a: Linear RGB 색공간 전환

> **상위 문서**: `P4-W4-01_freqsep_quality_improvement.md`
> **상태**: ✅ 완료
> **난이도**: 낮음 | **추가 GPU 비용**: ALU only (패스 추가 없음)

---

## 1. 왜 필요한가

### 현재 문제

현재 파이프라인은 **sRGB 감마 공간**에서 Gaussian blur를 수행한다.
sRGB는 감마 곡선(~2.2)이 적용된 비선형 공간이므로:

- Gaussian blur의 가중 평균이 **물리적으로 부정확** (어두운 쪽으로 편향)
- 밝은 피부와 어두운 피부 경계에서 **halo**(어두운 띠)가 발생
- 고주파 추출 `high = orig - low` 결과가 감마에 의해 왜곡

### Linear RGB에서의 개선

- 감마를 제거한 **물리적으로 정확한 평균** → halo 감소
- 고주파 성분의 크기가 실제 밝기 차이를 반영 → attenuation 정확도 향상
- **이후 모든 Step(Soft Light, 톤커브 등)의 기준 좌표계가 되므로 가장 먼저 적용**

### Council 합의 근거

> "전체 수식의 기준 좌표계가 바뀌므로 가장 먼저 고정해야 재튜닝 1회로 끝남" — 만장일치

---

## 2. 변경 대상

### 2.1 FREQ_SEP_GAUSSIAN_FRAGMENT 셰이더

**파일**: `cpp/src/gpu/shader_sources.cpp` (line 431~454)

**현재 코드**:
```glsl
// FREQ_SEP_GAUSSIAN_FRAGMENT
void main() {
    vec3 sum = vec3(0.0);
    for (int i = -uRadius; i <= uRadius; i++) {
        vec2 offset = uDirection * float(i);
        vec3 s = texture(uTexture, vTexCoord + offset).rgb;
        sum += s * uWeights[abs(i)];
    }
    fragColor = vec4(sum, 1.0);
}
```

**변경안**: Pass 1a (첫 수평 블러)의 입력만 sRGB→Linear 변환.
중간 패스(1b, 2a, 2b)는 이미 linear이므로 변환 불필요.

```glsl
uniform bool uLinearize;  // Pass 1a에서만 true

void main() {
    vec3 sum = vec3(0.0);
    for (int i = -uRadius; i <= uRadius; i++) {
        vec2 offset = uDirection * float(i);
        vec3 s = texture(uTexture, vTexCoord + offset).rgb;
        if (uLinearize) {
            s = pow(s, vec3(2.2));  // sRGB → Linear
        }
        sum += s * uWeights[abs(i)];
    }
    fragColor = vec4(sum, 1.0);
}
```

**성능 고려**: `if (uLinearize)` 분기는 uniform이므로 GPU에서 warp divergence 없음.
단, `pow(s, vec3(2.2))`가 루프 내부에서 매 텍셀마다 호출되므로 비용이 있음.

**대안 (더 효율적)**: 루프 밖에서 center 텍셀만 변환하는 것은 불가능 (블러 특성상 모든 텍셀 변환 필요).
fast approximation `s * s` (gamma 2.0 근사)를 사용하면 `pow` 비용 절감 가능:
```glsl
// pow(x, 2.2) 대신 정밀한 근사:
vec3 sRGBToLinear(vec3 c) {
    // sRGB 공식 정확 구현 (threshold 0.04045)
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
}
// 또는 간이: s * s (gamma 2.0 근사, 오차 ~5% 이내)
```

### 2.2 FREQ_SEP_COMPOSITE_FRAGMENT 셰이더

**파일**: `cpp/src/gpu/shader_sources.cpp` (line 460~502)

**변경 위치**: `void main()` 시작부와 끝부

```glsl
void main() {
    vec3 smoothLow = texture(uSmoothedLow, vTexCoord).rgb;  // 이미 linear
    vec3 low       = texture(uLowFreq, vTexCoord).rgb;      // 이미 linear
    vec3 orig      = texture(uOriginal, vTexCoord).rgb;
    float mask     = texture(uSkinMask, vec2(vTexCoord.x, 1.0 - vTexCoord.y)).r;

    // ★ 추가: original만 sRGB→Linear 변환
    orig = pow(orig, vec3(2.2));

    // ... (기존 고주파 추출 + 감쇠 로직 동일) ...

    vec3 beauty = smoothLow + adjusted_high;
    vec3 result = mix(orig, beauty, mask);

    // ★ 추가: 최종 출력을 Linear→sRGB 변환 (음수 방어 필수)
    result = pow(max(result, vec3(0.0)), vec3(1.0 / 2.2));

    fragColor = vec4(result, 1.0);
}
```

**핵심 포인트**:
- `smoothLow`, `low`는 Gaussian 셰이더에서 이미 linearize된 상태
- `orig`만 원본 텍스처에서 직접 읽으므로 여기서 변환 필요
- 최종 출력에서 sRGB로 역변환하여 디스플레이 호환

### 2.3 gpu_beauty_backend.h — Uniform 추가

**파일**: `cpp/include/iris_sdk/gpu/gpu_beauty_backend.h` (line 465~470)

```cpp
struct FreqSepGaussianUniforms {
    GLint uTexture = -1;
    GLint uDirection = -1;
    GLint uRadius = -1;
    GLint uWeights = -1;
    GLint uLinearize = -1;  // ★ 추가
} freq_sep_gaussian_uniforms_;
```

### 2.4 gpu_beauty_backend.cpp — Uniform 초기화 + 패스별 설정

**Uniform 초기화** (line 329~333):
```cpp
if (freq_sep_gaussian_program_ != 0) {
    freq_sep_gaussian_uniforms_.uTexture = glGetUniformLocation(freq_sep_gaussian_program_, "uTexture");
    freq_sep_gaussian_uniforms_.uDirection = glGetUniformLocation(freq_sep_gaussian_program_, "uDirection");
    freq_sep_gaussian_uniforms_.uRadius = glGetUniformLocation(freq_sep_gaussian_program_, "uRadius");
    freq_sep_gaussian_uniforms_.uWeights = glGetUniformLocation(freq_sep_gaussian_program_, "uWeights[0]");
    freq_sep_gaussian_uniforms_.uLinearize = glGetUniformLocation(freq_sep_gaussian_program_, "uLinearize");  // ★ 추가
}
```

**Pass 1a에서만 true 설정** (executeFreqSepPipelineImpl, line 1188):
```cpp
// Pass 1a: Horizontal Gaussian → temp
glUseProgram(freq_sep_gaussian_program_);
glUniform1i(freq_sep_gaussian_uniforms_.uLinearize, 1);  // ★ 첫 패스만 linearize
// ... 기존 코드 ...

// Pass 1b: Vertical Gaussian → lowFreq
glUniform1i(freq_sep_gaussian_uniforms_.uLinearize, 0);  // ★ 이후 패스는 이미 linear
// ... 기존 코드 ...
```

### 2.5 mapSkinQuality 재튜닝

**파일**: `cpp/src/gpu/gpu_beauty_backend.cpp` (line 981~1015)

Linear 공간에서는 고주파 magnitude가 sRGB 대비 달라지므로 attenuation 임계값 조정 필요:

- sRGB에서의 밝기 차이 0.04 → Linear에서는 약 `0.04^2.2 ≈ 0.0013` (매우 작아짐)
- **실측 기반으로 재튜닝** 필요 — 고정 공식으로 변환 불가

**예상 조정 방향**:
```cpp
// Linear 공간에서의 attenuation 범위 (sRGB보다 절대값이 작아짐)
p.attenuation_low = 0.005f;   // 기존 0.04 → Linear에서 ~0.005
p.attenuation_high = 0.03f + s * 0.03f;  // 기존 0.15~0.30 → ~0.03~0.06
```

⚠️ **이 값은 실제 테스트 후 재조정 필수** — Linear 변환 후 Android 디바이스에서 skinQuality 0.2/0.5/1.0을 시각적으로 비교하여 최종 확정.

### 2.6 Luminance 계수 교체 (Rec.601 → Rec.709)

**파일**: `cpp/src/gpu/shader_sources.cpp` — FREQ_SEP_COMPOSITE_FRAGMENT 내 magnitude 계산

Linear-light 공간에서는 Rec.601 감마 보정 계수 `(0.299, 0.587, 0.114)` 대신 Rec.709 선형 계수 `(0.2126, 0.7152, 0.0722)`를 사용해야 한다. 현재 코드와 이후 Step 3(edge), Step 5(sharpen)의 모든 luminance 계산에 동일하게 적용.

```glsl
// 변경 전 (sRGB 감마 공간용)
float magnitude = dot(abs(high), vec3(0.299, 0.587, 0.114));

// 변경 후 (Linear-light 공간용)
float magnitude = dot(abs(high), vec3(0.2126, 0.7152, 0.0722));
```

---

## 3. 테스트 계획

### 3.1 기존 테스트 업데이트

`cpp/tests/test_beauty_config_v2.cpp`의 FreqSepParamsTest 기대값 조정:
- attenuation 범위가 바뀌므로 `AttenuationRangeValid`, `LowFreqSmoothRatioInRange` 등

### 3.2 시각적 검증

| 테스트 항목 | 확인 내용 |
|------------|----------|
| skinQuality 0.2 | 미세 보정, 원본과 거의 동일 |
| skinQuality 0.5 | 잡티 제거 + 질감 보존 |
| skinQuality 1.0 | 강한 보정, 플라스틱 아닌지 확인 |
| 밝은 피부 ↔ 어두운 피부 경계 | **halo 감소** (핵심 검증 항목) |
| 머리카락/눈썹 경계 | halo 감소 |

### 3.3 성능 확인

- `pow(x, 2.2)` 추가로 인한 프레임 시간 증가 측정
- 중간급 기기 (Mali-G78 등)에서 30fps 유지 확인

---

## 4. 완료 기준

- [x] FREQ_SEP_GAUSSIAN_FRAGMENT에 `uLinearize` uniform 추가
- [x] FREQ_SEP_COMPOSITE_FRAGMENT에 sRGB↔Linear 변환 추가
- [x] gpu_beauty_backend.h에 `uLinearize` uniform 멤버 추가
- [x] gpu_beauty_backend.cpp에서 Pass 1a에만 linearize=true 설정
- [x] mapSkinQuality의 attenuation 재튜닝
- [x] FreqSep 관련 테스트 통과 (기존 테스트 호환 확인)
- [ ] Android 디바이스에서 halo 감소 시각적 확인 (디바이스 테스트 필요)
