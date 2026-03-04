# P4-W3-01: 피부 보정 고도화 브레인스토밍

| 항목 | 내용 |
|------|------|
| **작업 ID** | P4-W3-01 |
| **유형** | 브레인스토밍 / 기술 탐색 |
| **상태** | ✅ 브레인스토밍 완료 → 구현 워크페이퍼(P4-W3-02) 전환 대기 |
| **작성일** | 2026-03-03 |

---

## 1. 동기 & 현재 상태 평가

### 1.1 사용자 피드백 요약

| 항목 | 평가 | 비고 |
|------|------|------|
| **LUT** | ❌ 실효성 없음 | 거의 필요 없는 수준, 제거/축소 검토 |
| **밝기/색온도 필터** | △ 부차적 | 핵심 가치가 아님 |
| **피부 스무딩** | ⚠️ 핵심이나 부족 | 블러 강하면 얼굴 전체가 흐릿 |
| **경쟁사 대비** | ❌ 격차 존재 | 경쟁사: 선명+매끄러움 동시 달성 |

### 1.2 핵심 격차

```
현재 우리:     블러 ↑ → 피부 매끄러움 ↑ + 얼굴 선명도 ↓↓  (트레이드오프)
경쟁사:        피부 매끄러움 ↑↑ + 얼굴 선명도 유지          (독립 제어)
```

**근본 원인**: 현재 Bilateral Filter는 **공간적 블러**로, 에지를 "보존"하려 하지만 강도를 올리면 결국 에지까지 침범한다. 경쟁사는 **주파수 영역 분리** 접근으로 디테일과 피부 톤을 독립적으로 처리한다.

### 1.3 현재 파이프라인 한계

```
[현재 GPU 파이프라인]
입력 → Bilateral Filter (sigma 조정) → 소프트 포커스 → 출력
       ↑                                 ↑
   여기서 에지 손실 발생              추가 흐림 가중
```

- Bilateral Filter `sigma_color`를 올리면 → 에지 할로(halo) 발생
- `sigma_space`를 올리면 → 전체적으로 흐릿
- 소프트 포커스(Gaussian + Overlay)는 피부 보정이 아니라 "글로우 효과"
- ROI 마스킹은 눈/입술 보호에는 좋지만, 피부 내부의 디테일 보존과는 무관

---

## 2. 경쟁사 기술 분석

### 2.1 경쟁사가 달성하는 결과

| 특성 | 설명 |
|------|------|
| **피부 질감** | 모공, 잡티가 제거되지만 피부 "결"은 살아있음 |
| **얼굴 윤곽** | 코 능선, 턱선, 눈꺼풀 라인이 선명하게 유지 |
| **눈/입술** | 완벽하게 보존, 오히려 더 선명하게 보임 |
| **전체 인상** | "피부가 좋은 사람"처럼 보임 (포토샵 수준) |

### 2.2 경쟁사 추정 기술 스택

```
[추정 파이프라인]
입력 프레임
  ↓
① 피부 영역 정밀 세그멘테이션 (AI 또는 색공간 기반)
  ↓
② Frequency Separation (주파수 분리)
  ├─ Low Frequency: 색상, 톤, 큰 반점
  └─ High Frequency: 텍스처, 모공, 디테일
  ↓
③ Low Freq에만 선택적 스무딩 (잡티/반점 제거)
  ↓
④ High Freq 감쇠 재합성 (모공 약화, 피부결은 유지)
  ↓
⑤ 피부 마스크로 원본과 블렌딩
  ↓
출력 프레임
```

이 방식의 핵심: **에지(윤곽)는 저주파에 포함되지 않으므로 스무딩 대상이 아님**

---

## 3. 기술 후보군

### 3.1 Frequency Separation (주파수 분리) ⭐ 최유력

**원리**:
```
Original = Low Frequency + High Frequency

Low Freq  = GaussianBlur(Original, large_radius)     ← 색상/톤/큰 반점
High Freq = Original - Low Freq                       ← 텍스처/디테일/에지

처리:
  Smoothed Low  = SmartBlur(Low Freq)                 ← 큰 잡티만 제거
  Adjusted High = High Freq × attenuation             ← 미세 텍스처 감쇠

결과 = Smoothed Low + Adjusted High
```

**GPU 셰이더 구현 (4 패스)**:
```glsl
// Pass 1a/1b: Low Frequency 추출 (Separable Gaussian Blur H/V)
// Pass 2: High Frequency 추출 (Original - LowFreq)
// Pass 3a/3b: Low Freq 추가 스무딩 (Separable Gaussian Blur H/V)
// Pass 4: 재합성 (SmoothedLow + HighFreq × attenuation + 마스크 블렌딩)
```

| 항목 | 평가 |
|------|------|
| **결과 품질** | ⭐⭐⭐⭐⭐ 경쟁사 수준 달성 가능 |
| **구현 난이도** | ⭐⭐⭐ 중간 (셰이더 4패스) |
| **성능 비용** | ~6-10ms (GPU) |
| **기존 코드 활용** | TexturePool, RenderContext 재사용 가능 |
| **파라미터** | blur_radius, attenuation, threshold |

**장점**:
- 에지/윤곽이 원리적으로 보존됨 (고주파에 포함)
- attenuation으로 "피부결 유지 정도"를 정밀 제어
- 포토샵 리터칭과 동일한 원리 → 검증된 기법
- Gaussian Blur는 Separable → 매우 빠름

**단점**:
- 패스 수 증가 (현재 1-2패스 → 4패스)
- blur_radius가 크면 텍스처 메모리 증가
- Low Freq 스무딩 방식에 따라 결과 차이 큼

---

### 3.2 Surface Blur (포토샵 스타일)

**원리**:
```
for each pixel P:
  weighted_sum = 0, weight_total = 0
  for each neighbor N in radius:
    color_diff = |P - N|
    if color_diff < threshold:
      w = gaussian(distance) × (1 - color_diff/threshold)
      weighted_sum += N × w
      weight_total += w
  result = weighted_sum / weight_total
```

색상 차이가 임계값 이하인 이웃만 블러에 참여 → 에지는 자동 보존

**GPU 셰이더 구현 (1-2 패스)**:
```glsl
// Single pass Surface Blur
uniform float u_radius;
uniform float u_threshold;  // 색상 차이 임계값

// 임계값 이하 차이만 블러 → 에지 완벽 보존
```

| 항목 | 평가 |
|------|------|
| **결과 품질** | ⭐⭐⭐⭐ 좋음 (에지 보존 우수) |
| **구현 난이도** | ⭐⭐ 쉬움 (1패스) |
| **성능 비용** | ~3-6ms (GPU) |
| **기존 코드 활용** | 기존 Bilateral 셰이더 변형 |
| **파라미터** | radius, threshold |

**장점**:
- 1패스로 구현 가능 → 최소 성능 오버헤드
- 직관적 파라미터 (반경, 임계값)
- 에지 보존 능력이 Bilateral보다 우수

**단점**:
- 큰 반경에서 성능 급감 (O(r²) 샘플링)
- threshold 튜닝이 피부톤에 따라 달라짐
- 미세 텍스처 제어가 Freq Sep보다 제한적

---

### 3.3 멀티스케일 처리 (Laplacian Pyramid)

**원리**:
```
Original → Gaussian Pyramid (다운스케일)
        → Laplacian Pyramid (각 스케일의 디테일)

스케일별 선택적 스무딩:
  Level 0 (원본 해상도): 미세 모공 → 약한 감쇠
  Level 1 (1/2 해상도):  작은 잡티 → 중간 스무딩
  Level 2 (1/4 해상도):  큰 반점  → 강한 스무딩
  Level 3+ (저해상도):   피부톤   → 보존

재합성 → 스케일별 차등 처리된 결과
```

| 항목 | 평가 |
|------|------|
| **결과 품질** | ⭐⭐⭐⭐⭐ 최고 (스케일별 독립 제어) |
| **구현 난이도** | ⭐⭐⭐⭐ 높음 (다중 해상도 관리) |
| **성능 비용** | ~8-15ms (GPU) |
| **기존 코드 활용** | downscaleFactor 로직 일부 재사용 |
| **파라미터** | 스케일별 attenuation 배열 |

**장점**:
- 잡티 크기별 차등 처리 (가장 정밀)
- "모공은 약간 남기고, 여드름은 완전 제거" 가능
- 학술적으로 가장 정교한 접근

**단점**:
- 구현 복잡도 높음
- 텍스처 메모리 사용량 증가
- 성능 비용 최대
- 모바일에서 실시간 처리 도전적

---

### 3.4 AI 피부 세그멘테이션 강화

**원리**: 현재 Face Mesh 기반 ROI 대신, 정밀 피부 영역 마스크 생성

```
[현재]
Face Mesh 478점 → 얼굴 윤곽 폴리곤 → 대략적 피부 마스크
문제: 머리카락, 안경, 그림자 영역도 포함됨

[개선]
방법 A: 색공간 기반 (추가 모델 불필요)
  → YCrCb / HSV에서 피부색 범위 검출
  → Face Mesh 마스크 AND 피부색 마스크
  → 결과: 진짜 피부만 정밀 추출

방법 B: 경량 세그멘테이션 모델 (추가 모델 필요)
  → SkinNet (~500KB, MobileNet 기반)
  → 입력: 다운스케일 프레임 (128×128)
  → 출력: 피부 확률 맵
  → 결과: 가장 정밀한 마스크
```

| 항목 | 방법 A (색공간) | 방법 B (AI 모델) |
|------|----------------|-----------------|
| **정밀도** | ⭐⭐⭐ 양호 | ⭐⭐⭐⭐⭐ 최고 |
| **추가 비용** | 거의 없음 (~1ms) | 모델 ~500KB, ~5ms |
| **조명 강건성** | △ 조명에 민감 | ⭐⭐⭐⭐ 강건 |
| **구현 난이도** | ⭐ 매우 쉬움 | ⭐⭐⭐ 중간 |

**참고**: 이것은 스무딩 기법 자체는 아니고, "어디에 스무딩을 적용할지"를 개선하는 것. 다른 기법과 조합하여 사용.

---

### 3.5 텍스처 복원 블렌딩 (High-Pass Overlay)

**원리**: 스무딩 후 원본의 고주파 텍스처를 일부 다시 합성

```
Smoothed = StrongSmoothing(Original)
HighPass = Original - GaussianBlur(Original)

Result = Smoothed + HighPass × texture_preserve_amount

texture_preserve_amount:
  0.0 = 완전 매끄러움 (인형 같은 느낌)
  0.3 = 피부결은 남기고 잡티만 제거 ← 목표 지점
  1.0 = 원본과 동일 (효과 없음)
```

| 항목 | 평가 |
|------|------|
| **결과 품질** | ⭐⭐⭐ 양호 (단독 사용 시) |
| **구현 난이도** | ⭐ 매우 쉬움 |
| **성능 비용** | ~2-3ms (GPU) |
| **조합 가치** | ⭐⭐⭐⭐⭐ 다른 기법의 마감 단계로 최적 |

**장점**: 기존 어떤 스무딩 기법 위에도 적용 가능한 마감 레이어

---

## 4. 추천 구현 경로

### 4.1 조합 전략 (하이브리드 파이프라인)

```
                        [추천 파이프라인]

입력 프레임
    ↓
① 피부 마스크 정밀화 (색공간 AND Face Mesh)          ~1ms
    ↓
② Frequency Separation                               ~4ms
    ├─ Gaussian Blur → Low Frequency
    └─ Original - Low → High Frequency
    ↓
③ Low Freq: Gaussian Blur (큰 잡티/반점 제거)          ~2ms
    ↓
④ High Freq: 감쇠 재합성 (texture_preserve 파라미터)   ~1ms
    ↓
⑤ 피부 마스크로 원본과 블렌딩                          ~1ms
    ↓
출력 프레임

총 예상 비용: ~9ms (GPU, 원본 해상도) / ~5ms (1/2 다운스케일) → 30fps 유지 가능
```

### 4.2 단계별 구현 계획

#### Step 1: Frequency Separation 코어 (MVP) 🎯

**목표**: 현재 Bilateral을 대체하는 Freq Sep 기반 스무딩
**기간**: 핵심 기능
**구현 범위**:
- Gaussian Blur 셰이더 (Separable, 2패스)
- High Frequency 추출 셰이더
- 재합성 셰이더 (attenuation 파라미터)
- 기존 파이프라인에 통합

**결과물**: `smoothing` 파라미터가 Freq Sep 방식으로 동작

#### Step 2: 피부 마스크 정밀화

**목표**: 색공간 기반 피부 검출 추가
**구현 범위**:
- YCrCb 피부색 범위 검출 셰이더
- 기존 Face Mesh 마스크와 AND 연산
- 경계 페더링 개선

**결과물**: 머리카락/그림자 영역 제외된 정밀 마스크

#### Step 3: 파라미터 재설계

**목표**: 사용자 관점 단순화
**구현 범위**:
- 핵심 파라미터 축소 (아래 5.1 참조)
- LUT 지원 제거 또는 deprecation
- 프리셋 시스템 (자연/보통/강하게)

#### Step 4: 세부 튜닝 & 최적화

**목표**: 성능 최적화 + 품질 미세 조정
**구현 범위**:
- Separable 블러 최적화
- 텍스처 캐싱 전략
- 기기별 성능 프로파일링
- attenuation 커브 최적화

---

## 5. 파라미터 재설계 방향

### 5.1 현재 vs 제안

```
[현재 BeautyFilterConfigV2 — 11개 파라미터]
smoothing, brightness, softFocus, whitening,
colorBalance, wrinkleRemove, slimFace, enlargeEyes,
thinChin, roiOnly, protectEyes, protectLips, ...

[제안 — 핵심 우선순위 재배치]

=== Tier 1: 핵심 (피부 보정) ===
skinQuality     0.0~1.0   ← 단일 파라미터 (내부에서 blurRadius, highFreqPreserve, lowFreqSmooth로 매핑)
                              0.0=처리 없음, 0.6=기본 추천, 1.0=최대 스무딩

=== Tier 2: 보조 (형태 보정) ===
slimFace        0.0~1.0   ← 유지
enlargeEyes     0.0~1.0   ← 유지
thinChin        0.0~1.0   ← 유지

=== Tier 3: 선택적 (톤 보정) ===
brightness      0.5~1.5   ← 유지하되 중요도 낮춤
whitening       0.0~1.0   ← 유지하되 중요도 낮춤
colorBalance    -1.0~1.0  ← 유지하되 중요도 낮춤

=== 제거/통합 후보 ===
softFocus       → skinQuality에 통합 (별도 필요 없음)
wrinkleRemove   → skinQuality에 통합 (Freq Sep이 자동 처리)
LUT 지원        → 제거 (실효성 없음)
```

### 5.2 프리셋 시스템 (추가 제안)

```cpp
enum class BeautyPreset {
    NATURAL,    // skinQuality=0.3
    MODERATE,   // skinQuality=0.5
    STRONG,     // skinQuality=0.8
    CUSTOM      // 사용자 직접 설정 (skinQuality 0.0~1.0)
};
```

---

## 6. LUT & 기존 필터 정리 방향

### 6.1 LUT

| 현재 상태 | 결정 | 이유 |
|-----------|------|------|
| GPU Combined Color Pass에 통합 | **제거 또는 Optional 격하** | 사용자 피드백: 실효성 없음 |
| 3D LUT 텍스처 로딩 | API에서 deprecated 마킹 | 유지보수 비용 대비 가치 낮음 |

### 6.2 소프트 포커스

| 현재 상태 | 결정 | 이유 |
|-----------|------|------|
| 별도 패스 | **skinQuality에 통합** | Freq Sep이 더 나은 결과 제공 |
| Gaussian + Overlay | 대체됨 | "흐릿한 느낌"의 원인 중 하나 |

### 6.3 밝기 / 색온도 / 화이트닝

| 현재 상태 | 결정 | 이유 |
|-----------|------|------|
| Combined Color Pass | **유지하되 Tier 3** | 기능은 유효하나 핵심 가치 아님 |
| 개별 API | 유지 | 이미 구현 완료, 제거 비용 > 유지 비용 |

---

## 7. 성능 예산

### 7.1 tier별 성능 목표 (단일 기준표)

| | High-end | Mid-range | Low-end |
|--|----------|-----------|---------|
| **기기 예시** | Galaxy S24, iPhone 15 | Galaxy A54, iPhone SE3 | 구형 기기 |
| **처리 방식** | Freq Sep (원본 해상도) | Freq Sep (1/2 다운스케일) | Bilateral fallback |
| **뷰티 패스 목표** | ≤ 8ms | ≤ 12ms | ≤ 6ms |
| **검출 (MediaPipe)** | ~12ms | ~12ms | ~12ms |
| **렌즈 렌더링** | ~5ms | ~5ms | ~5ms |
| **기타** | ~3ms | ~3ms | ~3ms |
| **합계** | ~28ms | ~32ms | ~26ms |
| **FPS** | 35fps+ | 30fps | 38fps+ |
| **릴리즈 게이트** | 뷰티 ≤8ms 필수 | 뷰티 ≤12ms 필수 | 뷰티 ≤6ms 필수 |

**참고**: 다운스케일 전략으로 Mid-range에서도 30fps 달성. Low Frequency는 어차피 흐린 정보 → 1/2 해상도로 처리해도 품질 손실 거의 없음.

---

## 8. GPU 셰이더 설계 (개념 스케치)

### 8.1 Frequency Separation 셰이더

```glsl
// === Pass 1a: Horizontal Gaussian Blur ===
// === Pass 1b: Vertical Gaussian Blur ===
// → 결과: lowFreq 텍스처

// === Pass 2: High Frequency 추출 ===
uniform sampler2D u_original;
uniform sampler2D u_lowFreq;

void main() {
    vec3 orig = texture(u_original, v_texCoord).rgb;
    vec3 low  = texture(u_lowFreq, v_texCoord).rgb;
    vec3 high = orig - low + 0.5;  // 0.5 오프셋 (음수 방지)
    fragColor = vec4(high, 1.0);
}

// === Pass 3: Low Freq Gaussian Blur (Separable H/V) ===
// (저주파 추가 스무딩 — 큰 잡티/반점 제거)

// === Pass 4: 재합성 ===
uniform sampler2D u_smoothedLow;
uniform sampler2D u_highFreq;
uniform sampler2D u_original;
uniform sampler2D u_skinMask;
uniform float u_highFreqPreserve;  // 내부 파라미터 (skinQuality에서 매핑됨)

void main() {
    vec3 smoothLow = texture(u_smoothedLow, v_texCoord).rgb;
    vec3 high      = texture(u_highFreq, v_texCoord).rgb - 0.5;
    vec3 orig      = texture(u_original, v_texCoord).rgb;
    float mask     = texture(u_skinMask, v_texCoord).r;

    // 고주파 감쇠 (highFreqPreserve가 높을수록 원본 디테일 유지)
    vec3 adjusted_high = high * u_highFreqPreserve;

    // 재합성
    vec3 beauty = smoothLow + adjusted_high;

    // 피부 마스크로 원본과 블렌딩
    vec3 result = mix(orig, beauty, mask);

    fragColor = vec4(result, 1.0);
}
```

### 8.2 피부색 검출 셰이더 (보조)

> **⚠️ MVP 미포함**: 아래 YCrCb 셰이더는 탐색 단계의 코드 스케치이며,
> P4-W3-02~05 구현 범위에 포함되지 않는다. MVP에서는 Face Mesh 기반
> `combined_mask`만 사용한다 (§10 의사결정 #2 참조). 색공간 기반
> 피부 검출은 post-MVP 품질 gap 확인 시 재평가한다.

```glsl
// YCrCb 색공간에서 피부색 범위 검출 (MVP 미포함 — 참조용 스케치)
uniform sampler2D u_frame;
uniform sampler2D u_faceMeshMask;  // 기존 Face Mesh 마스크

void main() {
    vec3 rgb = texture(u_frame, v_texCoord).rgb;

    // RGB → YCrCb
    float Y  = 0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b;
    float Cr = (rgb.r - Y) * 0.713 + 0.5;
    float Cb = (rgb.b - Y) * 0.564 + 0.5;

    // 피부색 범위 (경험적 값, 조명에 따라 조정 필요)
    float skinProb = step(0.33, Cr) * step(Cr, 0.55)
                   * step(0.23, Cb) * step(Cb, 0.43);

    // Face Mesh 마스크와 AND
    float faceMask = texture(u_faceMeshMask, v_texCoord).r;
    float finalMask = skinProb * faceMask;

    fragColor = vec4(finalMask, finalMask, finalMask, 1.0);
}
```

---

## 9. 리스크 & 고려사항

### 9.1 기술 리스크

| 리스크 | 확률 | 영향 | 대응 |
|--------|------|------|------|
| Freq Sep 성능 부족 (저사양 기기) | 중 | 높음 | 다운스케일 + fallback to Bilateral |
| 피부색 검출 오탐 (조명 변화) | 중 | 중간 | Face Mesh 마스크 우선, 색공간은 보조 |
| 텍스처 메모리 증가 | 낮음 | 중간 | TexturePool 활용 + 다운스케일 |
| API 하위 호환성 | 낮음 | 중간 | V2 Config 확장, V1 호환 유지 |

### 9.2 디바이스 호환성

tier별 처리 방식 및 성능 목표는 **§7.1 단일 기준표** 참조.

---

## 10. 의사결정 상태표

| # | 질문 | 상태 | 결정 | 근거 |
|---|------|------|------|------|
| 1 | Freq Sep vs Surface Blur 단독 | ✅ 결정 완료 | **Freq Sep 확정** (Low Freq 스무딩은 Gaussian Blur) | 텍스처 보존 제어가 Surface Blur 단독보다 우월 (§12, §13.2, §15.2) |
| 2 | AI 피부 세그멘테이션 도입 시점 | ✅ 결정 완료 | **MVP에서는 Face Mesh + eye/lip protection으로 진행**. 색공간 추가는 품질 gap 확인 시 | Face Mesh가 1차 방어선으로 충분, AI 모델은 post-MVP 평가 (§13.1, §13.8) |
| 3 | softFocus 완전 제거 vs 유지 | ✅ 결정 완료 | **기본 비활성 + deprecated 마킹**. API 제거는 다음 메이저 버전에서 | Freq Sep이 상위 대체, 하위 호환성 유지 (§12.6, §13.8) |
| 4 | LUT 제거 시 하위 호환성 | ✅ 결정 완료 | **deprecated 마킹 후 차기 메이저 버전에서 제거** | 즉시 제거보다 안전, 유지 비용 최소 (§6.1, §12.6) |
| 5 | 프리셋 시스템 도입 여부 | ✅ 결정 완료 | **SDK에서 제공** (NATURAL/MODERATE/STRONG/CUSTOM) | 앱단 위임 시 매핑 테이블 파편화 (§5.2, §13.8) |
| 6 | texturePreserve 파라미터 노출 | ✅ 결정 완료 | **내부 매핑으로 숨김**. 외부는 `skinQuality` 단일 노출 | 튜닝 난이도 감소, debug 모드에서만 개별 접근 가능 (§12.5, §13.6, §15.2) |

---

## 11. 추가 코멘트 (비판적/전문가 관점)

> **작성자**: Codex

### 11.1 좋은 점

- 문제 정의가 정확합니다. 사용자 가치의 중심을 `톤 조절`이 아니라 `피부 보정 품질(잡티 억제 + 선명도 유지)`로 둔 판단은 타당합니다.
- LUT를 핵심에서 제외하려는 방향도 맞습니다. 현재 실효성이 낮고 유지보수 비용 대비 기여도가 작습니다.
- Frequency Separation 중심 접근은 기존 블러 기반 한계를 넘기 위한 현실적인 선택입니다.

### 11.2 보완이 필요한 핵심 포인트

1. **성공 기준의 정량화 부족**
   - 현재 문서는 방향은 좋지만, 합격/실패 판정 기준이 부족합니다.
   - 최소 KPI를 명시해야 합니다.
     - 피부 영역 잡티 억제율
     - 비피부 에지 보존율(눈/입술/윤곽)
     - 경쟁사 대비 블라인드 선호도

2. **시간 안정성(Temporal Stability) 항목 누락**
   - 단일 프레임 품질이 좋아도 프레임 간 강도 흔들림이 있으면 실제 사용감이 급격히 나빠집니다.
   - 마스크/저주파 성분에 EMA 안정화가 필요합니다.

3. **피부 마스크 강건성 과소평가**
   - 색공간 기반 마스크는 조명/화이트밸런스/피부톤 다양성에 취약합니다.
   - 초기엔 색공간 보조로 시작하되, 실패 조건(오탐률 임계치)을 미리 정의해야 합니다.

4. **성능 예산 낙관 가능성**
   - 10ms 예산은 중저가 Android에서 메모리 대역폭 병목으로 쉽게 초과될 수 있습니다.
   - 기기 tier별로 pass 수/해상도 스케일을 고정한 운영안이 필요합니다.

5. **파라미터 UX 정리 필요**
   - `skinSmooth`와 `texturePreserve`를 그대로 노출하면 튜닝 난이도가 높습니다.
   - 외부 API는 단일 품질 슬라이더(`skinQuality`) 우선, 내부에서 다중 파라미터로 매핑하는 방식이 안전합니다.

### 11.3 실행 우선순위 제안

1. 품질 KPI와 비교 테스트셋을 먼저 확정
2. MVP는 `선명도 보존형 스무딩` + `시간 안정화`에 집중
3. LUT/SoftFocus는 기본 경로에서 제외(Deprecated 유지)
4. tier별 성능 프로파일(High/Mid/Low) 고정 프리셋 도입

### 11.4 결론

현재 문서는 방향성은 적절하지만, 제품 완성도를 좌우하는 `정량 품질 기준`과 `시간 안정성`, `기기 tier 운영전략`이 부족합니다. 다음 단계는 알고리즘 확장이 아니라 **평가 체계와 운영 전략을 먼저 고정**하는 것이 맞습니다.

---

## 12. 통합 확장 브레인스토밍 (품질 우선, 기존 P4-W3-02 통합)

### 12.1 배경 재확인

- 기존 `필터 + LUT` 경로는 체감 가치가 낮았음
- 밝기/색조보다 사용자 체감 핵심은 `피부 질감 보정 + 얼굴 선명도 유지`
- 블러 중심 접근은 경쟁력 있는 결과(선명+매끈) 달성에 한계가 있음

### 12.2 제품 목표 (What Good Looks Like)

1. 피부 잡티/거친 텍스처는 줄어든다.
2. 눈, 입술, 코 라인, 턱선은 선명하게 유지된다.
3. 블러 필터 느낌이 아니라 자연스럽게 "피부가 좋아 보이는" 결과를 만든다.

### 12.3 실패 기준 (No-Go)

- 얼굴 윤곽/눈썹이 흐려짐
- 코 옆/입 주변 halo 발생
- 프레임 간 flicker
- 조명 변화 시 마스크 오탐으로 비피부 영역 스무딩 누수

### 12.4 MVP 파이프라인

```
입력
  ↓
Face Mesh 기반 피부 마스크
  ↓
Frequency Separation (Low/High)
  ↓
Low 선택적 스무딩
  ↓
High 보존/감쇠 재합성
  ↓
Temporal Stabilization (EMA)  ← ※ 아래 주석 참조
  ↓
출력
```

> **⚠️ Temporal Stabilization 구현 방식 명확화**: 위 다이어그램의
> "Temporal Stabilization (EMA)"는 별도 렌더 패스가 아니라,
> `blur_radius` 등 **스칼라 파라미터**에 One Euro Filter를 적용하는 것이다.
> 픽셀 레벨 프레임 간 EMA(temporal blending)가 아님.
> 구현 상세는 P4-W3-04 §3.1~3.2 참조.

원칙:

- 스무딩은 피부 + 저주파 중심
- 구조 정보(윤곽/눈/입술)는 원본 보존 우선
- 시간축 안정화로 flicker 억제

### 12.5 파라미터 전략 (외부 단순화)

- 외부 API: `skinQuality` 단일 노출
- 내부 매핑: `lowFreqSmoothStrength`, `highFreqPreserve`, `maskFeather`, `temporalAlpha`

### 12.6 운영 정책

- LUT: 기본 경로 제외(Deprecated/Optional)
- Brightness/Whitening/ColorBalance: Tier 3 유지
- SoftFocus: 피부 보정 경로 기본 비활성

### 12.7 검증 프레임워크

정량:
- 피부 영역 고주파 에너지 감소율
- 비피부 에지 보존율
- temporal variance
- ms/frame, FPS (tier별)

정성:
- 경쟁사 블라인드 A/B
- "흐리다/가짜 같다" 피드백 비율
- 피부톤 그룹별 균등성

### 12.8 2주 실행안

1. Week 1: MVP + KPI 계측
2. Week 2: tier 튜닝 + 블라인드 비교 + 프리셋 고정

### 12.9 우선순위 고정

1. 선명도 유지
2. 피부 잡티 보정
3. 시간 안정성
4. 톤/색감 보정

즉, 색감보다 질감 품질이 우선이다.

---

## 13. 추가 코멘트 (구현 관점 심화)

> **작성자**: Claude

### 13.1 11장 코멘트에 대한 응답

**동의하는 부분:**

- **정량 KPI 부족 (11.2.1)**: 맞음. 원본 문서가 방향성 위주로 작성되어 측정 기준이 빠져있었음.
- **시간 안정성 누락 (11.2.2)**: 중요한 지적. 특히 마스크 경계에서의 flicker는 사용자가 즉시 인지하는 결함.
- **단일 슬라이더 API (11.2.5)**: `skinQuality` 하나로 노출하는 방향에 전적으로 동의.

**보완이 필요한 부분:**

1. **피부 마스크 강건성 (11.2.3) — 과소평가는 아님**
   - 원본 3.4에서 이미 "Face Mesh 마스크 AND 색공간 마스크"로 제안함
   - Face Mesh 478점 기반 ROI가 1차 방어선 → 조명에 강건함 (기하학 기반)
   - 색공간은 2차 정밀화 용도 → 실패해도 Face Mesh 마스크 범위를 벗어나지 않음
   - 즉, 오탐 리스크는 "색공간 영역 ∩ Face Mesh 영역"으로 제한되므로, worst case에도 얼굴 밖으로 누수되지 않음
   - **다만** 색공간 오탐 시 "얼굴 내 비피부 영역(눈썹, 눈 그림자)"이 스무딩되는 문제는 있음 → 이건 기존 eye/eyebrow protection mask가 커버

2. **성능 예산 낙관 (11.2.4) — 부분 동의**
   - 원본 7장에서 이미 다운스케일 전략을 언급했고, Low Frequency는 1/2 해상도에서 처리해도 품질 손실이 거의 없음
   - 다운스케일 적용 시 실제 예산: ~5-6ms (원본의 ~10ms 대비 절반)
   - **하지만** 중저가 Android의 텍스처 대역폭 병목은 실재하는 문제 → tier별 고정 프리셋은 필수

### 13.2 12장 코멘트에 대한 응답

**동의하는 부분:**

- 12.2 제품 목표 3가지 → 간결하고 측정 가능. 좋은 정리.
- 12.3 실패 기준 → halo 언급이 특히 중요. Bilateral의 고질적 문제.
- 12.9 우선순위 → "질감 > 색감" 판단 정확.

**보완이 필요한 부분:**

1. **12.4 MVP 파이프라인의 "Low 선택적 스무딩" 미구체화**
   - Low Freq에 어떤 스무딩을 적용할지가 결과의 핵심인데 빠져있음
   - Low Freq에는 이미 에지 정보가 약화되어 있으므로, **단순 Gaussian Blur만으로도 충분**할 가능성이 높음
   - Surface Blur on Low Freq는 과잉 처리 → 에지가 이미 없는 데이터에 에지 보존 블러를 적용하는 셈
   - **제안**: MVP에서는 Low Freq에 Gaussian Blur 적용, 부족하면 Surface Blur로 업그레이드

2. **12.7 "피부 영역 고주파 에너지 감소율" 측정 복잡도**
   - 이론적으로 맞지만 실시간 측정이 어려움
   - 더 실용적인 대안: **SSIM(Structural Similarity) 기반** → 원본 피부 영역 vs 처리 후 피부 영역의 SSIM
   - 또는 단순히: **스무딩 전후 라플라시안 분산(Laplacian variance)** 비교 → 이건 OpenCV 한 줄로 측정 가능

3. **12.8 2주 실행안 — 너무 압축적**
   - Week 1에 "MVP + KPI 계측" 동시 진행은 무리
   - Freq Sep 셰이더만 구현+디버그하는 데 2-3일 소요 예상
   - 현실적 재조정 필요 (아래 13.5에서 제안)

### 13.3 핵심 누락 사항: blur radius의 결정적 중요성

11장, 12장 모두에서 **Frequency Separation의 성패를 가르는 핵심 파라미터**가 빠져있음.

```
blur_radius가 Freq Sep 결과 품질의 ~80%를 결정한다

radius 너무 작음 (5-10px):
  → 잡티가 High Freq에 남음 → 재합성 후에도 잡티 보임
  → 의미 없는 처리

radius 너무 큼 (50px+):
  → Low Freq가 과도하게 뭉개짐 → 색상 번짐, 톤 왜곡
  → 코 옆/눈가에 색 번짐 발생

이상적 범위:
  → 얼굴 바운딩 박스 대비 3-7%
  → 예: 얼굴 폭 300px → radius 9-21px
  → 예: 얼굴 폭 500px → radius 15-35px
```

**핵심**: radius는 고정값이 아니라 **Face Mesh 바운딩 박스에 비례하는 적응형(adaptive) 값**이어야 함. 이건 이미 `BeautyROIManager::computeROI()`에서 face_rect를 계산하고 있으므로 즉시 활용 가능.

### 13.4 핵심 누락 사항: High Freq attenuation 커브 설계

12장에서 "High 보존/감쇠 재합성"을 언급했지만 **감쇠 방식**이 구체화되지 않음.

```
[선형 감쇠] — 단순하지만 결과가 부자연스러움
adjusted_high = high × (1.0 - smoothStrength)
문제: 미세 피부결(약한 고주파)과 잡티(강한 고주파)가 동일 비율로 감쇠

[비선형 감쇠] — 추천 ⭐
// 강한 고주파(잡티)는 강하게 억제, 약한 고주파(피부결)는 보존
float magnitude = length(high);
float attenuation = smoothstep(threshold_low, threshold_high, magnitude);
adjusted_high = high × mix(preserve_amount, 1.0, attenuation);

효과:
  - 작은 변화(피부결, 모공) → magnitude 낮음 → 거의 보존
  - 큰 변화(잡티, 여드름) → magnitude 높음 → 강하게 감쇠
  - "잡티만 지우고 피부결은 살리는" 핵심 메커니즘
```

이 비선형 감쇠가 경쟁사와의 품질 차이를 만드는 **진짜 핵심**. 선형 감쇠만 쓰면 Freq Sep을 도입해도 "밋밋한 피부" 결과가 나옴.

### 13.5 기존 인프라 활용 전략 (누락)

현재 코드베이스에 이미 있는 자산을 활용하면 구현 비용이 크게 줄어드는데, 이 부분이 11-12장에서 다뤄지지 않음.

| 기존 자산 | 활용 방안 |
|-----------|----------|
| **One Euro Filter** (`one_euro_filter.h`) | Temporal Stability에 즉시 적용 — 마스크 경계값, 스무딩 강도의 프레임 간 안정화 |
| **TexturePool** (`texture_pool.h`) | Freq Sep 중간 텍스처(Low, High, Smoothed) 관리 — 이미 Ping-Pong 최적화됨 |
| **GPUProfiler** (`gpu_profiler.h`) | 각 패스 ms 측정 — KPI 계측 인프라 별도 구축 불필요 |
| **ROI 캐싱** (`beauty_roi_manager.h`, 100ms) | 마스크 재계산 방지 — 이미 동작 중 |
| **Separable Blur** (기존 softFocus 구현) | Gaussian Blur의 H/V 분리 처리 — 셰이더 재사용 가능 |
| **Combined Color Pass** | Freq Sep 재합성 패스에 통합 가능 — 추가 패스 없이 합성 |

**특히 One Euro Filter**: 12장에서 "Temporal Stabilization (EMA)"를 별도 단계로 추가했는데, EMA보다 One Euro Filter가 상위 호환이고 이미 구현되어 있음. 반응성과 안정성을 독립적으로 제어 가능.

### 13.6 skinQuality → 내부 매핑 구체화

12.5에서 `skinQuality` 단일 노출을 제안했지만, 내부 매핑 전략이 구체화되지 않음.

```
skinQuality (0.0 ~ 1.0) → 내부 파라미터 매핑

단순 선형 매핑의 문제:
  0.0~0.3 구간: 차이가 거의 안 보임 (사용자 불만)
  0.7~1.0 구간: 급격한 변화 (인형 느낌)

제안: S-커브 매핑 (smoothstep 기반)

skinQuality  →  blurRadius(%)  highFreqPreserve  lowFreqSmooth
    0.0           0%              1.0               0.0
    0.2           3%              0.85              0.2
    0.4           4.5%            0.65              0.4
    0.6           5.5%            0.45              0.6      ← 기본값 추천
    0.8           6.5%            0.25              0.8
    1.0           7%              0.10              1.0

blurRadius: Face Mesh bbox 대비 비율
highFreqPreserve: 1.0=피부결 완전 보존, 0.0=완전 매끄럽게
lowFreqSmooth: 0.0=스무딩 안 함, 1.0=최대 스무딩
```

이 매핑 테이블은 기기 tier별로 달라질 수 있으며, 프리셋 시스템의 기반이 됨.

### 13.7 실행안 현실적 재조정

12.8의 2주안을 구현 난이도 기반으로 재조정:

```
[Week 1: 셰이더 구현 + 파이프라인 통합]
Day 1-2: Separable Gaussian Blur 셰이더 (H/V 2패스)
         → 기존 softFocus의 Gaussian 셰이더 참조/확장
Day 3:   High Freq 추출 + 재합성 셰이더
         → 비선형 attenuation 포함
Day 4:   GPUBeautyBackend에 Freq Sep 모드 통합
         → 기존 smoothing 패스를 대체
Day 5:   adaptive blur radius + skinQuality 매핑
         → Face Mesh bbox 비례 계산

[Week 2: 튜닝 + 검증]
Day 1-2: One Euro Filter 적용 (temporal stability)
         → 마스크 경계 + 스무딩 파라미터 안정화
Day 3:   tier별 프로파일링 (GPUProfiler 활용)
         → High/Mid/Low 프리셋 확정
Day 4:   품질 튜닝 (attenuation 커브, 매핑 테이블)
         → 다양한 피부톤/조명 조건 테스트
Day 5:   A/B 비교 (기존 Bilateral vs Freq Sep)
         → Laplacian variance + 주관 평가
```

### 13.8 열린 질문 10번에 대한 답변 보충

원본 10장의 질문 중 11-12장에서 답변되지 않은 것:

| 질문 | Codex 답변 | 추가 의견 |
|------|-----------|----------|
| **Q1. Freq Sep vs Surface Blur 단독** | 12장에서 Freq Sep으로 확정 | **동의**. Surface Blur 단독은 텍스처 보존 제어가 부족 |
| **Q2. AI 세그멘테이션 도입 시점** | 언급 없음 | **Week 2 이후 평가**. MVP에서 Face Mesh + eye/lip protection으로 충분할 가능성 높음. 색공간 추가는 품질 gap이 확인될 때 |
| **Q3. softFocus 제거 vs 유지** | 12.6에서 "기본 비활성" | **동의하되**, API 제거는 다음 메이저 버전에서. 현재는 deprecated 마킹 |
| **Q5. 프리셋 시스템** | 12.5에서 간접 언급 | **SDK에서 제공이 맞음**. 앱단에 위임하면 매핑 테이블이 파편화됨 |
| **Q6. texturePreserve 노출** | 12.5에서 "내부 매핑" | **동의**. skinQuality 하나로 통합. 전문가 모드(debug용)로만 개별 노출 가능 |

### 13.9 결론

Codex의 11-12장 코멘트는 **제품화 관점에서 빠진 부분을 잘 짚었음**. 특히 정량 KPI, 시간 안정성, tier 전략은 반드시 반영해야 할 사항.

다만 알고리즘 구현 관점에서 **3가지 핵심이 여전히 미구체화** 상태:

1. **blur radius의 적응형 계산** — 결과 품질의 80% 결정
2. **High Freq 비선형 attenuation** — "잡티만 제거 + 피부결 보존"의 핵심 메커니즘
3. **기존 인프라 재활용** — 구현 기간과 안정성에 직접 영향

이 3가지를 MVP 셰이더 설계에 포함시키면, 2주 내 경쟁사 수준의 결과물에 근접할 수 있음.

---

## 14. 추가 코멘트 (13장 검토에 대한 Codex 의견)

> **작성자**: Codex

### 14.1 총평

13장은 단순 아이디어 정리가 아니라, 실제 구현 전환이 가능한 수준까지 내려온 코멘트입니다. 특히 `adaptive blur radius`, `High Freq 비선형 attenuation`, `기존 인프라 재활용`을 동시에 짚은 점이 매우 좋습니다.

### 14.2 강점

1. `adaptive blur radius`를 얼굴 박스 비율로 다룬 점이 정확합니다. 고정 radius 대비 거리/해상도 변화에 안정적입니다.
2. `High Freq 비선형 감쇠`를 핵심 메커니즘으로 본 판단이 타당합니다. 선형 감쇠보다 피부결 보존과 잡티 억제를 분리하기 좋습니다.
3. One Euro Filter, TexturePool, GPUProfiler 같은 기존 자산 재활용 전략이 현실적입니다. 구현 속도와 안정성 모두에 유리합니다.
4. `skinQuality` 단일 노출 + 내부 매핑 전략은 SDK API 복잡도 관리 측면에서 가장 안전합니다.

### 14.3 보완 제안

1. radius 규칙에 `min/max clamp`를 명시해야 합니다.
   - 예: `radius_px = clamp(face_width * ratio, 6, 28)`
2. `length(high)` 기반 감쇠는 색번짐 리스크가 있어, 가능하면 Y(밝기) 기반으로 감쇠량을 계산하는 것이 더 안전합니다.
3. SSIM/Laplacian만으로는 체감 품질을 완전히 대체하기 어려우므로, 블라인드 주관평가를 릴리즈 게이트로 명시하는 것이 좋습니다.
4. 2주 계획은 구현 자체는 가능하지만 디바이스 매트릭스 검증까지 넣으면 타이트하므로, 최소 1주 리스크 버퍼를 문서에 표시하는 것을 권장합니다.

### 14.4 실행 결론

13장은 채택 가치가 높고, 아래 4가지를 추가 반영하면 즉시 개발 기준 문서로 사용 가능합니다.

1. 13.3/13.4를 MVP 완료 조건으로 고정
2. 13.6 매핑 테이블을 3티어 디바이스에서 수치 튜닝
3. 13.7 일정표에 리스크 버퍼 추가
4. KPI + 블라인드 평가를 주간 리포트 항목으로 고정

---

## 15. 최종 응답 (14장 반영 + 합의 사항 확정)

> **작성자**: Claude

### 15.1 14장 보완 제안에 대한 수용

4가지 모두 수용합니다. 아래에 구체화합니다.

#### 15.1.1 radius clamp 규칙 확정

> **⚠️ 브레인스토밍 단계 초안**: 아래 수식(`width × ratio × skinQuality`, 선형 매핑)은
> 탐색 단계에서 작성된 초안이다. **최종 구현 정본**은 P4-W3-02 §4.6의
> `mapSkinQuality()`이며, **고정 ratio=0.05** + S-커브 기반 감쇠 매핑을 사용한다
> (blur_radius는 skinQuality와 독립, 감쇠 강도만 제어). 아래 §15.2 합의표의
> "S-커브 기반 3파라미터 매핑" 항목이 감쇠 파라미터 매핑을 반영한다.

```cpp
// adaptive blur radius 브레인스토밍 초안 (→ 정본: P4-W3-02 §4.6)
const float ratio = 0.05f;  // 얼굴 폭 대비 5% (기본값)
const int min_radius = 6;   // 유효 최소 — 이보다 작으면 효과 없음
const int max_radius = 28;  // 유효 최대 — 이보다 크면 색 번짐

// skinQuality=0 → Freq Sep 바이패스 (원본 그대로 출력)
if (skinQuality <= 0.0f) return;  // 조기 종료, 처리 안 함

int radius_px = std::clamp(
    static_cast<int>(face_rect.width * ratio * skinQuality),
    min_radius,
    max_radius
);
```

- `skinQuality = 0.0` → Freq Sep 전체 바이패스 (처리 없음, 0ms)
- `skinQuality > 0.0` → clamp(6, 28) 범위 내에서 적응형 반경 적용
- clamp는 "유효 처리 범위의 안전 범위"이지, quality=0의 동작이 아님

#### 15.1.2 Y(밝기) 기반 감쇠로 수정

```glsl
// 수정 전 (13.4 원안): RGB length 기반
float magnitude = length(high);  // ← 색번짐 리스크

// 수정 후 (14.3.2 반영): Y(luminance) 기반
float magnitude = dot(abs(high), vec3(0.299, 0.587, 0.114));
// 밝기 변화량만으로 감쇠 판단 → 색 채널 간 차등 감쇠 방지
```

이유: `length(high)`는 RGB 채널 전체의 벡터 크기이므로, 색상 차이가 큰 영역(예: 피부-입술 경계)에서 과도한 감쇠가 발생할 수 있음. Y 기반이면 밝기 변화만 보므로 색번짐이 원천 차단됨.

#### 15.1.3 블라인드 주관평가를 릴리즈 게이트로 확정

```
[릴리즈 게이트 — 필수 통과 조건]

정량 게이트 (자동):
  ✅ Laplacian variance 감소율: 피부 영역 30-60% 감소
  ✅ 비피부 에지 보존율: SSIM > 0.95
  ✅ Temporal variance: 프레임 간 스무딩 강도 변동 < 5%
  ✅ 성능: §7.1 tier별 기준표 준수 (High ≤8ms, Mid ≤12ms, Low ≤6ms)

정성 게이트 (수동, 릴리즈 필수):
  ✅ 내부 블라인드 A/B: 기존 Bilateral 대비 선호도 > 70%
  ✅ "흐리다/가짜 같다" 피드백 비율 < 10%
  ✅ 피부톤 3그룹 (밝음/중간/어두움) 균등 품질 확인
```

정량은 CI에서 자동 검증, 정성은 주간 리뷰에서 수동 판정.

#### 15.1.4 일정 리스크 버퍼 추가

```
[최종 일정 — 2주 + 1주 버퍼]

Week 1: 셰이더 구현 + 파이프라인 통합
  Day 1-2: Separable Gaussian + Freq Sep 코어
  Day 3:   비선형 attenuation (Y 기반) + 재합성
  Day 4:   GPUBeautyBackend 통합
  Day 5:   adaptive radius + skinQuality 매핑

Week 2: 튜닝 + 1차 검증
  Day 1-2: One Euro Filter temporal stability
  Day 3:   tier별 프로파일링 + 프리셋 확정
  Day 4:   품질 튜닝 (attenuation 커브, 매핑 테이블)
  Day 5:   정량 게이트 자동 검증 + 1차 A/B 비교

Week 3 (리스크 버퍼): 디바이스 매트릭스 + 릴리즈 판정
  Day 1-2: 실기기 테스트 (Android 3기기 + iOS 2기기)
  Day 3:   피부톤 그룹별 튜닝
  Day 4:   블라인드 주관평가 + 릴리즈 게이트 판정
  Day 5:   문서화 + 코드 리뷰
```

Week 2까지 완료되면 조기 릴리즈 가능. Week 3는 보험.

### 15.2 합의 사항 총정리

양측(Claude 13장 + Codex 14장) 논의를 통해 확정된 사항:

| 항목 | 확정 내용 |
|------|----------|
| **핵심 기법** | Frequency Separation (Gaussian Blur 기반) |
| **감쇠 방식** | Y(luminance) 기반 비선형 attenuation (smoothstep) |
| **blur radius** | Face Mesh bbox × ratio, clamp(6, 28) |
| **외부 API** | `skinQuality` 단일 파라미터 |
| **내부 매핑** | S-커브 기반 3파라미터 매핑 (blurRadius, highFreqPreserve, lowFreqSmooth) |
| **temporal stability** | One Euro Filter (기존 인프라 활용) |
| **Low Freq 스무딩** | MVP: Gaussian Blur, 필요시 Surface Blur 업그레이드 |
| **MVP 완료 조건** | adaptive radius + 비선형 attenuation 필수 포함 |
| **릴리즈 게이트** | 정량(자동) + 블라인드 주관평가(수동) 이중 게이트 |
| **일정** | 2주 + 1주 리스크 버퍼 = 총 3주 |
| **기존 기능 정리** | LUT deprecated, softFocus/wrinkleRemove → skinQuality 통합, Tier 3 톤 보정 유지 |
| **디바이스 tier + 성능 목표** | §7.1 단일 기준표 참조 (High ≤8ms / Mid ≤12ms / Low ≤6ms) |

### 15.3 다음 단계

이 문서는 브레인스토밍 단계를 완료했으며, 다음 액션은:

1. **구현 워크페이퍼 작성**: `P4-W3-02_freq_sep_implementation.md`
2. **셰이더 프로토타이핑 시작** (Week 1 Day 1)
3. **테스트 이미지셋 준비** (다양한 피부톤/조명 조건)

---

## 변경 이력

| 날짜 | 변경 내용 | 작성자 |
|------|----------|--------|
| 2026-03-03 | 최초 브레인스토밍 문서 작성 | Claude |
| 2026-03-03 | 11장 추가 코멘트 섹션 보강 | Codex |
| 2026-03-03 | 12장 통합 확장 브레인스토밍 추가 | Codex |
| 2026-03-03 | 변경 이력을 문서 마지막으로 재배치, 추가 코멘트 작성자 명시 | Codex |
| 2026-03-03 | 13장 구현 관점 심화 코멘트 추가 (blur radius, attenuation 커브, 인프라 활용, 실행안 재조정) | Claude |
| 2026-03-03 | 14장 Codex 코멘트 추가 (13장 평가 및 보완 제안) | Codex |
| 2026-03-03 | 15장 최종 응답 — 14장 4개 보완 제안 전수 수용, 합의 사항 확정, Y기반 감쇠 수정, 릴리즈 게이트 명시, 3주 일정 확정 | Claude |
| 2026-03-03 | 승격 심사 1차 불합격 5건 수정 — ①radius quality=0 바이패스, ②Low-Freq→Gaussian 통일, ③외부API skinQuality 단일 확정, ④열린질문→결정상태표, ⑤성능목표 tier별 단일표 통합 | Claude |
| 2026-03-03 | 승격 심사 2차 4건 수정 — ①문서상태 '완료'로 통일, ②skinSmooth잔류→skinQuality 전수 교체, ③프리셋 skinQuality 기준으로 수정, ④다음단계 P4-W3-02 단일 경로 확정 | Claude |
| 2026-03-04 | Gemini 리뷰 반영: §8.2 YCrCb 셰이더에 "MVP 미포함" 주석 추가, §12.4 Temporal Stabilization에 "스칼라 파라미터 필터링" 명확화 주석 추가 | Claude |
| 2026-03-04 | Codex 리뷰 반영: §15.1.1 radius 수식에 "브레인스토밍 초안" 주석 추가 — 정본은 P4-W3-02 §4.6 mapSkinQuality() | Claude |
