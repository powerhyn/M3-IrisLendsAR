# FreqSep 피부 잡티 보정 품질 분석 리포트

> **프로젝트**: IrisLensSDK — GPU Beauty Filter
> **작성일**: 2026-03-08
> **목적**: 현재 Frequency Separation 구현의 품질을 업계 레퍼런스와 비교 분석하고, 개선 방향을 도출한다.
> **대상 독자**: 개발팀 / AI 리뷰어 (개선 방향에 대한 의견 요청)

---

## 1. 현재 구현 개요

### 1.1 파이프라인 구조 (5-pass GPU)

```
입력 프레임 (sRGB)
    │
    ▼
[Pass 1a] Separable Gaussian Blur (Horizontal) → temp
[Pass 1b] Separable Gaussian Blur (Vertical)   → lowFreq
    │
    ▼
[Pass 2a] Additional Gaussian Blur (H) on lowFreq → temp
[Pass 2b] Additional Gaussian Blur (V) on temp    → smoothedLow
    │
    ▼
[Pass 3] Composite Shader
    ├─ high = original - lowFreq          (고주파 추출)
    ├─ magnitude = luminance(abs(high))   (잡티 크기 측정)
    ├─ blemishFactor = smoothstep(low, high, magnitude)
    ├─ preserve = mix(highFreqPreserve, 1.0, 1.0 - blemishFactor)
    ├─ beauty = smoothedLow + high * preserve  (재합성)
    └─ result = mix(original, beauty, skinMask) (마스크 블렌딩)
    │
    ▼
출력 프레임
```

### 1.2 Composite 셰이더 (GLSL ES 3.1)

```glsl
void main() {
    vec3 smoothLow = texture(uSmoothedLow, vTexCoord).rgb;
    vec3 low       = texture(uLowFreq, vTexCoord).rgb;
    vec3 orig      = texture(uOriginal, vTexCoord).rgb;
    float mask     = texture(uSkinMask, vec2(vTexCoord.x, 1.0 - vTexCoord.y)).r;

    // High Frequency inline extraction
    vec3 high = orig - low;

    // Luminance-based magnitude
    float magnitude = dot(abs(high), vec3(0.299, 0.587, 0.114));

    // Non-linear attenuation
    float blemishFactor = smoothstep(uAttenuationLow, uAttenuationHigh, magnitude);
    float preserve = mix(uHighFreqPreserve, 1.0, 1.0 - blemishFactor);
    vec3 adjusted_high = high * preserve;

    // Re-synthesis (additive)
    vec3 beauty = smoothLow + adjusted_high;

    // Mask blending
    vec3 result = mix(orig, beauty, mask);
    fragColor = vec4(result, 1.0);
}
```

### 1.3 파라미터 매핑 (`mapSkinQuality`)

`skinQuality` (0.0~1.0) 입력을 smoothstep S-curve로 변환 후 FreqSepParams를 생성한다.

```cpp
float t = clamp(skin_quality, 0.0f, 1.0f);
float s = t * t * (3.0f - 2.0f * t);  // smoothstep
```

| 파라미터 | 공식 | 범위 | 역할 |
|---------|------|------|------|
| `blur_radius` | `face_width × (0.03 + s × 0.03)` | 6~28 px | Gaussian 커널 크기 (저주파 추출) |
| `high_freq_preserve` | `1.0 - s × 0.65` | 1.0→0.35 | 고주파 질감 보존율 |
| `low_freq_smooth_ratio` | `0.30 + s × 0.15` | 0.30→0.45 | 2차 블러 비율 (색감 균일화) |
| `attenuation_low` | 고정 `0.04` | — | smoothstep 하한 (이하 = 질감 보존) |
| `attenuation_high` | `0.15 + s × 0.15` | 0.15→0.30 | smoothstep 상한 (이상 = 잡티 제거) |

**skinQuality별 실제 값 (face_width=200 기준)**:

| skinQuality | s | blur_radius | high_freq_preserve | low_freq_ratio | atten_low | atten_high |
|:-----------:|:---:|:-----------:|:------------------:|:--------------:|:---------:|:----------:|
| 0.0 | — | — | — | — | — | — (disabled) |
| 0.2 | 0.104 | 6 | 0.93 | 0.316 | 0.04 | 0.166 |
| 0.5 | 0.500 | 9 | 0.675 | 0.375 | 0.04 | 0.225 |
| 0.7 | 0.784 | 11 | 0.49 | 0.418 | 0.04 | 0.268 |
| 1.0 | 1.000 | 12 | 0.35 | 0.45 | 0.04 | 0.30 |

### 1.4 이론적 기반

| 구현 요소 | 학술적 근거 |
|----------|------------|
| Gaussian blur → low/high 분리 | Durand & Dorsey 2002 — Two-scale base/detail decomposition |
| `high = original - blurred` | 표준 Frequency Separation (푸리에 이론) |
| smoothstep 비선형 감쇠 | YUCIHighPassSkinSmoothing의 tone curve 변형 |
| magnitude 기반 잡티/질감 분류 | Bilateral filter 원리 (Tomasi & Manduchi 1998)의 근사 |
| additive 재합성 | Durand 2002 two-scale recomposition |

---

## 2. 레퍼런스 비교 분석

### 2.1 YUCIHighPassSkinSmoothing (GitHub: YuAo)

**출처**: https://github.com/YuAo/YUCIHighPassSkinSmoothing
**플랫폼**: iOS CoreImage / Metal / GPUImage
**성능**: iPhone 5s에서 640×800 @ 60fps

**파이프라인 (7단계)**:

```
[1] Exposure -1 EV (하이라이트 영역 마스크 정확도 향상)
[2] Green-Blue Overlay Blend → 단일 채널 마스크 생성
[3] High Pass Filter: high = image - gaussianBlur(image) + 0.5
[4] Hard Light ×3 + RGB Curve Boost → 마스크 대비 극대화
[5] RGB Tone Curve 적용: (0,0)→(120/255, 146/255)→(1,1) 미드톤 리프트
[6] CIBlendWithMask: 톤커브 적용본과 원본을 마스크로 블렌딩
[7] CISharpenLuminance: sharpness = amount × 0.6
```

**핵심 특징**:
- G/B 채널의 Overlay 합성으로 **피부 영역 자동 분리** (피부색의 G/B 채널 특성 활용)
- Hard Light 3회 반복으로 마스크 콘트라스트를 극단적으로 강화
- 톤 커브로 미드톤을 올려서 **피부 톤 균일화** (주파수 분리가 아닌 색상 조정 접근)
- 최종 Sharpen 패스로 스무딩 후 **선명도 복구**

### 2.2 Vincent Dedun의 AR Beauty Mode

**출처**: https://medium.com/swlh/how-i-implemented-my-own-augmented-reality-beauty-mode-3bf3b74e5507
**기법**: O(1) Real-Time Bilateral Filter (Yang et al. 2009)

**파이프라인 (4 GPU 패스)**:

```
[1] 5-Slab Bilateral Filter 준비: sRGB→linear RGB, 휘도를 K=5 슬랩으로 분할
[2] Separable Gaussian Blur (H): 5개 슬랩 각각에 수평 블러
[3] Separable Gaussian Blur (V): 5개 슬랩 각각에 수직 블러
[4] Slab Combine + Overlay Blend: bilateral 결과 생성 → inverted high-pass overlay
```

**핵심 특징**:
- **Bilateral Filter**로 저주파 추출 — Gaussian과 달리 에지를 자동 보존
- **CIELAB 기반 스킨 마스크**: Lab a/b 축의 red/yellow 성분으로 피부 감지
  ```glsl
  float skin_mask(vec4 color) {
      vec3 lab = rgb2lab(color.rgb);
      float a = smoothstep(0.45, 0.55, lab.g);   // red 성분
      float b = smoothstep(0.46, 0.54, lab.b);   // yellow 성분
      return min(min(a, b), ...);
  }
  ```
- **Linear RGB 색공간**에서 모든 처리 수행 → 감마 왜곡 없는 정확한 블러
- **Overlay 블렌딩**: `inverted_high_pass` → overlay 합성 (additive와 다른 톤 보존 특성)

### 2.3 Wavelet Decompose (Pat David / PIXLS.US)

**출처**: https://patdavid.net/2014/07/wavelet-decompose-again/
**기법**: À trous wavelet (B3 스플라인 커널)

**파이프라인 (N-스케일 분해)**:

```
입력 이미지
    │
    ├─ Scale 1 (r=2):   가장 미세한 디테일 (모공, 표면 질감)     → 보존
    ├─ Scale 2 (r=4):   약간 더 큰 디테일                         → 보존
    ├─ Scale 3 (r=8):   중간 크기 디테일                          → 보존
    ├─ Scale 4 (r=16):  큰 불규칙성 (잡티, 색소침착)              → bilateral blur 감쇠
    ├─ Scale 5 (r=32):  가장 큰 디테일 (피하 색상 변화)           → bilateral blur 감쇠
    └─ Residual:        색상과 톤 정보만                          → bilateral blur 균일화
    │
    ▼
Grain Merge 블렌딩으로 재합성
```

**핵심 특징**:
- **5+1 스케일 분해**: 주파수 대역별 독립 제어
- Scale 1~3을 손대지 않아 **미세 피부 질감 100% 보존**
- Scale 4~5에만 bilateral blur 적용하여 **잡티만 선택적 제거**
- 물리적으로 "모공은 살리고 여드름만 제거"가 가능

---

## 3. Gap 분석: 우리에게 없는 것

### 3.1 기법별 비교표

| 기법 | YUCI (#4) | Dedun (#7) | Wavelet (#8) | **우리** |
|------|:---------:|:----------:|:------------:|:--------:|
| 저주파 추출 방식 | Gaussian | **Bilateral** | À trous wavelet | Gaussian |
| 색공간 | sRGB | **Linear RGB** | sRGB | sRGB |
| 합성 방식 | **Overlay** | **Overlay** | Grain Merge | **Additive** |
| 마스크 생성 | G/B Overlay 자동 | **CIELAB 자동** | 수동 | 외부 의존 |
| 마스크 후처리 | **Hard Light ×3** | — | — | 없음 |
| 톤 커브 보정 | **미드톤 리프트** | — | — | 없음 |
| 최종 Sharpen | **Luminance Sharpen** | — | — | 없음 |
| 주파수 스케일 수 | 2 | 2 | **5+1** | 2 |
| 스케일별 독립 제어 | ✗ | ✗ | **✓** | ✗ |
| Exposure 전처리 | **-1 EV** | — | — | 없음 |

### 3.2 품질 영향도 분석

**가장 큰 품질 차이를 만드는 요소** (영향도 순):

1. **Gaussian vs Bilateral 저주파 추출** (Dedun)
   - Gaussian은 에지를 무시하고 블러 → 눈, 머리카락, 입술 경계에서 halo 발생
   - Bilateral은 밝기 유사성을 고려하여 에지를 자동 보존
   - **영향**: 피부와 비피부 경계의 자연스러움에 근본적 차이

2. **Additive vs Overlay 합성** (YUCI, Dedun 공통)
   - Additive: `beauty = smoothLow + adjusted_high` — 값이 범위를 벗어날 수 있음
   - Overlay: 어두운 영역은 더 어둡게, 밝은 영역은 더 밝게 하면서 스무딩 — **톤과 콘트라스트 자연스럽게 보존**
   - **영향**: 결과물의 전반적인 톤 자연스러움

3. **최종 Sharpen 패스 부재** (YUCI)
   - 어떤 스무딩이든 약간의 디테일 손실은 불가피
   - Sharpen으로 인지적 선명도를 복구하면 "피부는 매끄럽지만 선명한" 효과
   - **영향**: 사용자가 느끼는 "흐림감" 직접 해소

4. **Linear RGB 처리 부재** (Dedun)
   - sRGB 감마(~2.2)에서 Gaussian blur는 어두운 쪽으로 편향
   - Linear에서 blur하면 물리적으로 정확한 평균 → 피부톤 경계의 halo 감소
   - **영향**: 밝고 어두운 피부톤 경계에서의 부자연스러움

5. **톤 커브 미드톤 리프트 부재** (YUCI)
   - 미드톤을 살짝 올리면 피부의 미세한 색상/밝기 차이가 줄어듦
   - "피부에 광채가 있는" 느낌을 부여
   - **영향**: 피부 톤 균일감

### 3.3 우리의 smoothstep 감쇠 방식의 구조적 한계

현재 방식:
```
magnitude = luminance(abs(original - blurred))
blemishFactor = smoothstep(0.04, 0.15~0.30, magnitude)
```

**문제**: magnitude는 **진폭(amplitude)**만 측정하고 **공간 주파수(spatial frequency)**를 구분하지 못한다.

- 모공: 작은 공간 크기, 작은 진폭 → 보존 ✓
- 여드름: 작은 공간 크기, 큰 진폭 → 제거 ✓
- **넓은 색소침착**: 큰 공간 크기, **작은 진폭** → 보존됨 (제거해야 하지만 못함) ✗
- **선명한 피부 주름**: 작은 공간 크기, 중간 진폭 → **제거됨** (보존해야 하지만 못함) ✗

Wavelet 방식은 공간 주파수를 물리적으로 분리하므로 이 문제가 없다.

---

## 4. 개선 제안

### 4.1 Phase 1: 현재 구조 유지 + 후처리 추가 (5-pass → 6~7 pass)

현재 파이프라인의 구조적 변경 없이 composite 전후에 패스를 추가하여 품질을 개선한다.

#### A. Luminance Sharpen 패스 추가

composite 출력에 Unsharp Mask 기반 선명도 복구를 적용한다.

```glsl
// Unsharp Mask: sharp = original + (original - blur) × amount
vec3 blurred = gaussianBlur(beauty, sigma=1.0);
vec3 sharpened = beauty + (beauty - blurred) * uSharpenAmount;
// uSharpenAmount = skinQuality * 0.4 ~ 0.6
```

- **비용**: +1 GPU 패스 (separable이면 +2)
- **기대 효과**: 스무딩으로 잃어버린 미세 디테일의 인지적 복구. "흐림감" 직접 해소.

#### B. 톤 커브 미드톤 리프트

composite 셰이더 내에 간단한 톤 커브를 추가하여 피부 미드톤을 올린다.

```glsl
// Quadratic mid-tone lift: (0,0) → (0.47, 0.57) → (1,1)
float toneCurve(float x) {
    return x + 0.2 * x * (1.0 - x);  // 간략화된 미드톤 리프트
}
vec3 beauty_toned = vec3(toneCurve(beauty.r), toneCurve(beauty.g), toneCurve(beauty.b));
```

- **비용**: 추가 패스 없음 (composite 셰이더 내 ALU 연산)
- **기대 효과**: 피부 톤 균일감, "광채 피부" 느낌

#### C. Overlay 블렌딩 모드 전환

현재 additive 합성을 Overlay 블렌딩으로 교체한다.

```glsl
// 현재 (additive):
vec3 beauty = smoothLow + adjusted_high;

// 개선 (overlay):
vec3 inverted_high = vec3(0.5) - adjusted_high;
vec3 beauty = overlay(orig, inverted_high);

// overlay 함수:
float overlay(float base, float blend) {
    return base < 0.5
        ? 2.0 * base * blend
        : 1.0 - 2.0 * (1.0 - base) * (1.0 - blend);
}
```

- **비용**: 추가 패스 없음 (composite 셰이더 내 분기)
- **기대 효과**: 톤과 콘트라스트의 자연스러운 보존

#### D. Linear RGB 색공간 처리

모든 블러/합성을 linear RGB에서 수행하고, 최종 출력 시 sRGB로 변환한다.

```glsl
// Pass 1 입력 시: sRGB → linear
vec3 linear = pow(srgb, vec3(2.2));

// Pass 3 출력 시: linear → sRGB
vec3 srgb = pow(linear, vec3(1.0/2.2));
```

- **비용**: 추가 패스 없음 (각 패스의 입출력에 pow 연산 추가)
- **기대 효과**: 밝고 어두운 피부톤 경계에서의 halo 감소

### 4.2 Phase 2: 구조 변경 (성능 영향 큼, 품질 도약)

Phase 1 적용 후에도 품질이 부족한 경우 고려.

#### E. Gaussian → O(1) Bilateral Filter 교체

Yang et al. (CVPR 2009) 기반 real-time bilateral filter로 Pass 1을 교체한다.

- K=5 슬랩, sigma_spatial=3, sigma_range=0.1
- 장점: 에지 보존이 근본적으로 해결
- **비용**: 5 pass → ~12 pass, 텍스처 11개 필요

#### F. 3-Scale Wavelet Decompose

전체 5+1 scale 대신 3-scale + residual로 실용적 타협안을 구현한다.

- Scale 1 (r=2): 미세 질감 → 보존
- Scale 2 (r=8): 중간 결함 → 강한 감쇄
- Scale 3 (r=24): 큰 결함 → 약한 감쇄
- Residual: 톤/색상 → 선택적 균일화
- **비용**: 5 pass → ~10 pass

---

## 5. 의견 요청 사항

이 리포트를 기반으로 다음 질문에 대한 의견을 구합니다:

### Q1. Phase 1 (A~D) 적용 순서와 우선순위

네 가지 개선을 모두 적용할 예정인데, 어떤 순서가 가장 효과적인가?
특히 Overlay 블렌딩(C)과 Linear RGB(D)를 동시에 적용할 때 상호작용 이슈가 있는가?

### Q2. Additive vs Overlay 합성의 실질적 차이

현재 `beauty = smoothedLow + adjusted_high` (additive) 방식에서
Overlay로 전환하면 실제로 어떤 시각적 차이가 예상되는가?
Additive 방식의 장점이 있다면 하이브리드 접근이 더 나은가?

### Q3. smoothstep 감쇠의 구조적 한계 대안

magnitude 기반 smoothstep이 "넓은 색소침착"과 "선명한 주름"을 구분하지 못하는 문제에 대해,
Phase 1 범위 내에서 (wavelet 도입 없이) 완화할 수 있는 방법이 있는가?

### Q4. Sharpen 강도 자동 조절

Sharpen 패스를 추가할 때, `skinQuality`에 비례하여 sharpen 강도를 올리는 것이 맞는가?
아니면 고정값이나 다른 기준이 더 적절한가?

### Q5. Phase 2 진입 기준

Phase 1 완료 후 Phase 2(Bilateral 또는 Wavelet)로 넘어가야 하는 시점을 판단하는 기준은 무엇인가?
모바일 GPU에서 12-pass bilateral의 실시간 처리가 현실적인가?

---

## 6. 참고 문헌

1. Durand & Dorsey (2002). "Fast Bilateral Filtering for the Display of High-Dynamic-Range Images." ACM SIGGRAPH.
2. Tomasi & Manduchi (1998). "Bilateral Filtering for Gray and Color Images." IEEE ICCV.
3. He, Sun & Tang (2010). "Guided Image Filtering." ECCV / IEEE TPAMI 2013.
4. Yang, Tan & Ahuja (2009). "Real-Time O(1) Bilateral Filtering." CVPR.
5. YuAo/YUCIHighPassSkinSmoothing — https://github.com/YuAo/YUCIHighPassSkinSmoothing
6. Vincent Dedun, "How I Implemented My Own AR Beauty Mode" — https://medium.com/swlh/how-i-implemented-my-own-augmented-reality-beauty-mode-3bf3b74e5507
7. Pat David, "Wavelet Decompose for Skin Retouching" — https://patdavid.net/2014/07/wavelet-decompose-again/
8. Velusamy et al. (2020). "FabSoften: Face Beautification via Dynamic Skin Smoothing." CVPR Workshop.
9. CGFR (2024). "Controllable and Gradual Facial Blemishes Retouching via Physics-based Skin Simulation." arXiv:2406.13227.

---

## 부록 A: 전체 파라미터 매핑 코드

```cpp
GPUBeautyBackend::FreqSepParams
GPUBeautyBackend::mapSkinQuality(float skin_quality, int face_width) {
    FreqSepParams p;

    if (skin_quality <= 0.0f) {
        p.enabled = false;
        return p;
    }

    p.enabled = true;

    // S-curve mapping (smoothstep for natural transition)
    float t = std::clamp(skin_quality, 0.0f, 1.0f);
    float s = t * t * (3.0f - 2.0f * t);  // smoothstep

    // blur_radius: 3~6% of face_width, scales with quality → clamp(6, 28)
    const float ratio = 0.03f + s * 0.03f;
    p.blur_radius = std::clamp(
        static_cast<int>(face_width * ratio),
        6, 28
    );

    // high_freq_preserve: 1.0 → 0.35 (최소 35% 질감 보존)
    p.high_freq_preserve = 1.0f - s * 0.65f;

    // low_freq_smooth: 30~45% of blur_radius
    p.low_freq_smooth_radius_ratio = 0.30f + s * 0.15f;

    // attenuation range
    p.attenuation_low = 0.04f;
    p.attenuation_high = 0.15f + s * 0.15f;  // 0.15 ~ 0.30

    return p;
}
```

## 부록 B: FreqSepParams 구조체

```cpp
struct FreqSepParams {
    int blur_radius = 15;
    float high_freq_preserve = 0.45f;
    float low_freq_smooth_radius_ratio = 0.5f;
    float attenuation_low = 0.02f;
    float attenuation_high = 0.15f;
    bool enabled = false;
};
```
