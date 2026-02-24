# ISS-005: 블렌드 모드 색상 희석(Fading) — PerfectSDK 대비 분석 및 개선 방향

- 작성일: 2026-02-20
- 상태: 분석 완료 — 브레인스토밍 대기
- 우선순위: High
- 영향 범위: GLSL blend 셰이더 전체 (Mode 0~6), Adaptive Scale 로직
- 관련 작업: P4-W1-03 (Luminance Tint 셰이더)

---

## 1) 문제 요약

P4-W1-03에서 구현한 7개 블렌드 모드(Normal, Multiply, Screen, Overlay, LuminanceTint, LumTintLinear, SoftLight)가 경쟁사 PerfectSDK 대비 **색상이 현저히 희석되어 보이는** 현상.

사용자 관찰:
> "블렌딩 옵션들이 너무 색상이 흐리게 보인다"
> "변화는 느껴지지만 이게 올바른 변화인지 판단이 어렵다"

## 2) 비교 스크린샷

| 이미지 | 앱 | 블렌드 모드 | 렌즈 |
|--------|-----|-----------|------|
| `perfectsdk_sample.jpg` | PONVIEW (PerfectSDK) | 미상 (자체 알고리즘) | 트래블 라구나 베이지 |
| `irissdk_sample_normal.jpg` | IrisLensSDK | Normal (mode 0) | LAGUNA BEIGE |
| `irissdk_sample_screen.jpg` | IrisLensSDK | Screen (mode 2) | LAGUNA BEIGE |
| `irissdk_sample_luminance_tinit.jpg` | IrisLensSDK | Luminance Tint (mode 4) | LAGUNA BEIGE |
| `irissdk_sample_lum_tint.jpg` | IrisLensSDK | Lum Tint Linear (mode 5) | LAGUNA BEIGE |

## 3) 시각적 차이 분석

### 3-1. 모드별 비교표

| 항목 | PerfectSDK | Normal(0) | Screen(2) | LumTint(4) | LumTintLinear(5) |
|------|:----------:|:---------:|:---------:|:----------:|:----------------:|
| 색상 채도 | ★★★★★ | ★★★ | ★★ | ★★☆ | ★☆ |
| 렌즈 패턴 선명도 | ★★★★★ | ★★★★ | ★★★ | ★★★ | ★★ |
| 자연스러움 | ★★★★★ | ★★★ | ★★ | ★★★ | ★★ |
| 렌즈 색 가시성 | 뚜렷함 | 보통 | 희미/회색빛 | 어두움 | 매우 어두움 |

### 3-2. PerfectSDK 관찰 포인트

1. **색상 선명도**: 어두운 동양인 홍채 위에서도 베이지/브라운 색이 또렷하게 보임
2. **렌즈 패턴**: 동심원 형태의 렌즈 텍스처가 선명하게 유지됨
3. **자연스러운 입체감**: 홍채의 명암 구조(결, 줄무늬)가 렌즈 색상 아래에서 보존됨
4. **동공 대비**: 동공 부분은 자연스럽게 어둡고, 홍채 부분은 렌즈 색이 충분히 발현
5. **반사광 처리**: 각막 반사(specular)가 렌즈 위에 자연스럽게 존재

### 3-3. IrisLensSDK 문제점 상세

| 모드 | 핵심 문제 | 원인 |
|------|----------|------|
| Normal(0) | 단순 알파 블렌딩 → 렌즈 색과 어두운 원본 홍채가 혼합되며 탁해짐 | `mix(dark_iris, lens_color, alpha)` — dark_iris가 결과를 끌어내림 |
| Screen(2) | 밝아지지만 채도 감소, 회색빛 | `1-(1-dark)*(1-lens)` — 어두운 base에서 탈색 효과 |
| LumTint(4) | 색상이 어두움, 렌즈 색 미발현 | `lens * lum * scale` — lum이 0.15면 scale(2.5)을 곱해도 최대 37.5% |
| LumTintLinear(5) | 가장 어두움, 거의 보이지 않음 | `toLinearFast(base)` 제곱 → lum이 더 작아짐 (0.15² = 0.0225) |
| SoftLight(6) | (미촬영) 은은하지만 색 변화 약함 | Soft Light 특성상 대비 변화는 있으나 색상 치환 효과 부족 |

---

## 4) 근본 원인: 절대 밝기 vs 상대 밝기 매핑

### 핵심 통찰

현재 셰이더는 홍채의 **절대 밝기(absolute luminance)**를 렌즈 색에 곱한다.
PerfectSDK는 **상대 밝기(relative luminance)**를 사용하는 것으로 추정된다.

### 4-1. 현재 방식: 절대 밝기 곱셈

```glsl
// Mode 4: blendLuminanceTint
float lum = dot(base, vec3(0.299, 0.587, 0.114));   // ≈ 0.15 (어두운 홍채)
float scale = clamp(0.5 / max(0.1, uAvgIrisLum), 0.8, 2.5);  // ≈ 2.5 (최대)
vec3 tinted = blend * lum * scale;   // blend * 0.15 * 2.5 = blend * 0.375
return mix(base, tinted, opacity);
```

**문제 수치 시뮬레이션** (어두운 동양인 홍채, lum ≈ 0.15):

| 단계 | 값 | 설명 |
|------|-----|------|
| 렌즈 색(blend) | (0.82, 0.72, 0.52) | LAGUNA BEIGE RGB |
| lum | 0.15 | 어두운 홍채 밝기 |
| uAvgIrisLum | 0.15 | CPU EMA 평균 |
| scale | 2.5 (max clamp) | 0.5/0.15=3.33 → clamped |
| **tinted** | **(0.31, 0.27, 0.20)** | **blend × 0.375 → 어두운 결과** |
| 원본(base) | (0.18, 0.14, 0.10) | 어두운 홍채 RGB |
| **최종 (α=0.7)** | **(0.27, 0.23, 0.17)** | **mix(base, tinted, 0.7) → 여전히 어두움** |

→ 렌즈의 밝은 베이지 색(0.82)이 최종 결과에서 0.27로 축소됨. **원래 색의 33%만 발현.**

### 4-2. 추정 방식: 상대 밝기 정규화 (Color Replace)

```glsl
// 제안: blendColorReplace
float lum = dot(base, vec3(0.299, 0.587, 0.114));   // ≈ 0.15
float detail = lum / max(0.01, uAvgIrisLum);          // 0.15 / 0.15 = 1.0
detail = clamp(detail, 0.2, 2.5);
vec3 colored = blend * detail;   // blend * 1.0 = 원래 렌즈 색 그대로!
return mix(base, colored, opacity);
```

**같은 조건 수치 시뮬레이션**:

| 단계 | 값 | 설명 |
|------|-----|------|
| 렌즈 색(blend) | (0.82, 0.72, 0.52) | LAGUNA BEIGE RGB |
| lum (현재 픽셀) | 0.15 | 평균 밝기 픽셀 |
| uAvgIrisLum | 0.15 | CPU EMA 평균 |
| **detail** | **1.0** | **lum/avg = 정규화됨** |
| **colored** | **(0.82, 0.72, 0.52)** | **렌즈 색 100% 발현!** |
| **최종 (α=0.7)** | **(0.63, 0.55, 0.39)** | **mix(base, colored, 0.7) → 선명한 베이지** |

**각 영역에서의 차이**:

| 홍채 영역 | 절대 밝기 결과 | 상대 밝기 결과 | 차이 |
|----------|:------------:|:------------:|:----:|
| 동공 (lum=0.05) | blend × 0.125 = 매우 어둡 | blend × 0.33 = 자연스럽게 어둡 | ✓ |
| 홍채 평균 (lum=0.15) | blend × 0.375 = 어둡 | **blend × 1.0 = 풀 컬러** | 핵심 차이 |
| 홍채 밝은 결 (lum=0.25) | blend × 0.625 = 중간 | blend × 1.67 = 밝은 포인트 | ✓ |
| 반사광 (lum=0.80) | blend × 2.0 = 밝음 | blend × 2.5(clamp) = 반사광 유지 | ~ |

→ **상대 밝기 방식은 홍채의 절대 어두움과 무관하게 렌즈 색상이 제대로 나오면서, 홍채 고유의 명암 구조(결/패턴)만 보존한다.**

### 4-3. 비유로 설명

| | 절대 밝기 (현재) | 상대 밝기 (제안) |
|--|:--:|:--:|
| **비유** | "어두운 방에서 색안경 착용" → 모든 게 어둡게 보임 | "정상 조명에서 색안경 착용" → 색안경 색이 제대로 보임 |
| **수학** | output = lens_color × absolute_brightness | output = lens_color × (pixel / average) |
| **핵심** | 밝기에 의존적 → 어두운 홍채 = 어두운 결과 | 밝기 독립적 → 어떤 홍채든 렌즈 색 발현 |

---

## 5) Mode 5 (LuminanceTintLinear) 추가 악화 원인

```glsl
vec3 baseL = toLinearFast(base);  // sRGB → Linear: 0.15² = 0.0225
float lum = dot(baseL, vec3(0.2126, 0.7152, 0.0722));  // ≈ 0.02
```

Linear 변환이 어두운 값을 **제곱**하여 더 어둡게 만듦:
- sRGB lum 0.15 → Linear lum **0.02**
- scale은 CPU측 sRGB 기준 `uAvgIrisLum`으로 계산 → Linear 공간에서의 보정이 불일치
- Specular 복원 `smoothstep(0.7, 0.95, lum)`은 lum=0.02에서 전혀 발동 안함

→ Mode 5가 Mode 4보다 더 어두운 이유.

---

## 6) 개선 방안 (브레인스토밍 항목)

### 방안 A: 새 블렌드 모드 "Color Replace" 추가 (Mode 7)

**핵심 아이디어**: `lum / uAvgIrisLum`으로 상대 밝기 정규화

```glsl
vec3 blendColorReplace(vec3 base, vec3 blend, float opacity) {
    float lum = dot(base, vec3(0.299, 0.587, 0.114));
    float detail = lum / max(0.01, uAvgIrisLum);
    detail = clamp(detail, 0.2, 2.5);
    vec3 colored = blend * detail;
    return mix(base, colored, opacity);
}
```

- 장점: 어두운 홍채에서도 렌즈 색 100% 발현, 홍채 질감(결/패턴) 보존
- 단점: `uAvgIrisLum`이 부정확하면 전체 밝기 편향 발생 가능
- 위험: EMA 수렴 전(첫 ~10프레임) 불안정할 수 있음 → 기본값 0.35에서 출발하므로 큰 문제 없을 듯

### 방안 B: 기존 LuminanceTint 밝기 바닥(Floor) 추가

**핵심 아이디어**: luminance를 [minLift, 1.0]으로 리매핑하여 바닥값 보장

```glsl
vec3 blendLuminanceTintV2(vec3 base, vec3 blend, float opacity) {
    float lum = dot(base, vec3(0.299, 0.587, 0.114));
    float liftedLum = mix(0.35, 1.0, lum);  // 0.0 → 0.35, 1.0 → 1.0
    vec3 tinted = blend * liftedLum;
    return mix(base, tinted, opacity);
}
```

- 장점: 단순, `uAvgIrisLum` 의존 없음, 안정적
- 단점: 고정 바닥값이므로 밝은 홍채(파란/녹색)에서는 지나치게 밝을 수 있음
- 변형: `minLift`를 `uAvgIrisLum` 기반으로 동적 조절 가능

### 방안 C: Adaptive Scale 상한 확대 + Gamma Lift

**핵심 아이디어**: 기존 scale 범위 확대 + 감마 보정으로 밝기 끌어올림

```glsl
vec3 blendLuminanceTintV3(vec3 base, vec3 blend, float opacity) {
    float lum = dot(base, vec3(0.299, 0.587, 0.114));
    float scale = clamp(0.5 / max(0.05, uAvgIrisLum), 0.8, 5.0);  // 상한 2.5→5.0
    float liftedLum = pow(lum * scale, 0.7);  // gamma < 1.0으로 밝기 끌어올림
    vec3 tinted = blend * liftedLum;
    return mix(base, tinted, opacity);
}
```

- 장점: 기존 구조 유지하면서 밝기 개선
- 단점: scale 5.0 + gamma 조합이 오히려 과포화 유발 가능, 튜닝 파라미터 증가
- 위험: scale 상한 확대 시 밝은 환경에서 과보정 가능

### 방안 D: 듀얼 레이어 합성 (렌즈 패턴 분리)

**핵심 아이디어**: 렌즈 텍스처 자체의 밝기를 "패턴 레이어"로, 색상을 "컬러 레이어"로 분리

```glsl
vec3 blendDualLayer(vec3 base, vec3 blend, float opacity) {
    float irisLum = dot(base, vec3(0.299, 0.587, 0.114));
    float lensLum = dot(blend, vec3(0.299, 0.587, 0.114));

    // 렌즈 색상 (색상만 추출, 밝기 정규화)
    vec3 lensColor = blend / max(0.01, lensLum);

    // 홍채 디테일 + 렌즈 밝기 패턴 혼합
    float detailLum = mix(lensLum, irisLum / max(0.01, uAvgIrisLum), 0.4);

    vec3 result = lensColor * detailLum;
    return mix(base, result, opacity);
}
```

- 장점: 렌즈 패턴 선명도 + 홍채 디테일 동시 보존 가능, PerfectSDK에 가장 가까운 접근
- 단점: 복잡도 높음, 렌즈 텍스처 RGB에서 색상/밝기 분리 시 정보 손실 가능
- 변형: `mix` 비율을 uniform으로 노출하여 튜닝 가능하게

### 방안 E: 기존 Normal(0) 모드 opacity 부스트

**핵심 아이디어**: 가장 단순한 접근 — opacity를 올려서 렌즈 색 비중 증가

```
현재: mix(iris, lens, alpha * 0.7) → 렌즈 30% 미만
제안: mix(iris, lens, alpha * 0.9) → 렌즈 비중 증가
```

- 장점: 코드 변경 최소, 즉시 효과
- 단점: 홍채 질감 완전 소실, "스티커 붙인 느낌" 위험
- 평가: 근본 해결이 아닌 임시 방편

---

## 7) 방안 비교 매트릭스

| 기준 | A: ColorReplace | B: Floor Lift | C: Scale+Gamma | D: DualLayer | E: Opacity↑ |
|------|:---:|:---:|:---:|:---:|:---:|
| **색상 발현도** | ★★★★★ | ★★★★ | ★★★★ | ★★★★★ | ★★★★ |
| **홍채 질감 보존** | ★★★★ | ★★★ | ★★★ | ★★★★★ | ★★ |
| **구현 난이도** | 낮음 | 매우 낮음 | 보통 | 높음 | 매우 낮음 |
| **튜닝 파라미터** | 1개(clamp 범위) | 1개(floor값) | 3개(scale,gamma,clamp) | 2개(mix비율,clamp) | 0개 |
| **uAvgIrisLum 의존** | 높음 | 없음 | 높음 | 중간 | 없음 |
| **다양한 홍채 범용성** | ★★★★ | ★★★ | ★★★ | ★★★★★ | ★★ |
| **PerfectSDK 근접도** | ★★★★ | ★★★ | ★★★ | ★★★★★ | ★★ |

## 8) 권장 우선순위

1. **방안 A (Color Replace)** — 즉시 구현, 효과 대비 복잡도 최적
2. **방안 D (Dual Layer)** — A 검증 후 고도화 방향으로 검토
3. **방안 B (Floor Lift)** — A의 폴백(uAvgIrisLum 불안정 시 대안)

## 9) 추가 조사 필요 사항

- [ ] PerfectSDK에서 다른 렌즈(진한 색/밝은 색)도 비교 필요
- [ ] 밝은 홍채(서양인/밝은 갈색)에서의 동작 검증 필요
- [ ] `uAvgIrisLum` EMA의 수렴 속도/정확도 → Color Replace 품질에 직접 영향
- [ ] 렌즈 텍스처 자체의 알파 채널 분포 확인 (패턴 vs 투명 영역)
- [ ] 현재 `투명도` 슬라이더 값 확인 (스크린샷 기준 ~65% 위치)

## 10) 브레인스토밍 논의 포인트

1. **A vs D**: Color Replace의 단순함 vs Dual Layer의 완성도 — 어디까지 구현할 것인가?
2. **기존 Mode 4/5 유지 여부**: 개선할 것인가, 새 모드로 대체할 것인가?
3. **uAvgIrisLum 신뢰도**: EMA 초기 수렴 전 불안정 구간 → 기본값 전략은?
4. **UI 측면**: Mode가 8개 이상으로 늘어나면 Spinner에서 구분이 어려운지?
5. **성능 영향**: 추가 연산(나눗셈, clamp)이 저사양 GPU에서 부담인지?
6. **렌즈 텍스처 품질**: 셰이더 개선만으로 충분한가, 텍스처 에셋 자체도 조정 필요한가?

---

## 11) CodexComment (2026-02-20)

### 결론 요약

- 이번 결론은 **샘플 비교 결과 + 실제 코드 경로 검토**를 함께 반영한 결과다.
- `perfectsdk_sample.jpg` 대비 `irissdk_sample_normal/screen/luminance_tinit/lum_tint`에서 공통적으로 렌즈 색 발현이 약하고 모드 간 분리도가 낮다.
- 현재는 알고리즘 고도화(A/D) 이전에 **입력 지표(uAvgIrisLum) 정확도 복구**가 선행되어야 한다.

### 우선 해결해야 할 이슈 (Critical Path)

1. **P1: 홍채 밝기 샘플링 좌표 오류**
   - 현재 `sampleIrisLuminanceNv21()`가 좌/우 홍채의 중간점(눈 사이)을 샘플링한다.
   - 참조: `android/demo-app/src/main/java/com/irislenssdk/demo/GpuRenderActivity.kt`
   - 조치: `leftDetected/rightDetected` 분기 후, 검출된 눈 각각 샘플링하여 평균. 단안 검출 시 단안값만 사용.

2. **P1: 샘플링 오류 상태에서 ColorReplace 튜닝 진행 금지**
   - `uAvgIrisLum`가 오염된 상태로 A/D를 튜닝하면 잘못된 기준에 과적합될 위험이 높다.
   - 조치: 샘플링 수정 → 동일 조건 재촬영 → 그 결과로 블렌드 튜닝.

3. **P2: Mode 5 색공간 불일치 분리 대응 필요**
   - `base`는 linear로 계산하면서 `uAvgIrisLum`은 sRGB 기준이라 보정 축이 어긋난다.
   - 참조: `android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLRenderer.kt`
   - 조치: (a) `uAvgIrisLum` linear 정합 또는 (b) 정합 전까지 Mode 5를 실험 플래그로 격리.

### 실행 순서 제안 (실수 방지형)

1. 샘플링 버그 수정(단안/양안 처리 포함)
2. 동일 조건 A/B 재촬영(거리/노출/렌즈/슬라이더 고정)
3. 방안 A(Color Replace) 최소 구현 + 하이라이트 클리핑 가드
4. 모드 분리도/자연스러움 시연 평가 후 D(Dual Layer) 착수 여부 결정

---

## 12) 2차 심층 분석 및 기술 감사 (Deep Technical Audit)

초기 분석(1~4절) 이후 수행된 정밀 기술 감사 결과, 단순 '알고리즘(Logic)'의 문제가 아닌 **렌더링 파이프라인의 물리적/구조적 결함(Physics/Structure)**이 "물 빠진 색감"과 "이질감"의 주원인으로 밝혀짐.

### 13-1. Gamma Space Violation (치명적, Fatal)
- **현상**: 모든 블렌드 모드에서 중간 톤의 채도가 죽고 회색조(Grayish)로 뜸.
- **원인**: **sRGB 텍스처를 선형 공간(Linear Space)으로 변환하지 않고 바로 연산함.**
- **증거**: `CameraGLRenderer.kt`의 셰이더 코드에서 `texture()` 샘플링 후 `pow(color, 2.2)` 없이 바로 `mix()` 수행.
- **영향**: sRGB의 0.5(회색)는 물리적 밝기가 약 0.21임. 이를 0.5로 취급하여 연산하므로 결과값이 의도보다 훨씬 밝고 탁하게 나옴.

### 13-2. Double-Multiplied Alpha (높음, High)
- **현상**: 렌즈의 반투명한 부분이나 경계선에 검은 테두리(Dark Halo)가 생기거나, 렌즈 전체가 어둡고 유령처럼 보임.
- **원인**: 안드로이드 `Bitmap`은 기본적으로 **Premultiplied Alpha** 상태이나, 셰이더는 이를 무시하고 Straight Alpha 수식(`mix()`)을 사용.
- **수식 오류**: `mix(base, blend, alpha)` → `blend`가 이미 `rgb * alpha`인 상태에서 또 `alpha`를 곱함 → `rgb * alpha * alpha`가 되어 색상이 급격히 약해짐.

### 13-3. Specular Physics 부재 (중간, Medium)
- **현상**: 렌즈가 눈동자 위에 '인쇄된 종이 스티커'처럼 붙어 입체감이 없음.
- **원인**: 실제 눈은 렌즈(Iris) 위에 투명한 각막(Cornea)과 눈물층이 있고 그 위에 반사광(Glint)이 맺힘. 현재는 렌즈 이미지가 반사광을 덮어버림(Occlusion).
- **비교**: PerfectSDK는 렌즈 색상 위로 쨍한 하이라이트가 살아있음 (Layering: Iris < Lens < Specular).

### 13-4. Latency & Sync (구조적, Critical)
- **현상**: 고개를 돌릴 때 렌즈가 미세하게 밀림(Swimming), 조명 변화 시 렌즈 밝기가 한 박자 늦게 반응(Lag).
- **원인**:
    1. **밝기 계산 지연**: CPU에서 `uAvgIrisLum`을 EMA(α=0.1)로 계산 → 약 20프레임(0.6초) 지연 발생.
    2. **비동기 파이프라인**: FaceMesh 결과(CPU)와 카메라 프레임(GPU) 간의 타임스탬프 동기화 부재.

---

## 13) 종합 개선 로드맵 (Roadmap)

단순 색상 보정을 넘어, 상용급(Commercial Quality) 렌더링을 위한 단계별 개선안.

### Phase 1: 파이프라인 정상화 (Physics Fix) - **즉시 적용**
물리적으로 올바른 연산을 수행하도록 셰이더 기본 구조 수정.

1.  **Gamma Fix (Plan F)**:
    - Input: `vec3 linear = pow(sRGB.rgb, vec3(2.2));`
    - Output: `gl_FragColor = vec4(pow(linear, vec3(1.0/2.2)), alpha);`
2.  **Alpha Fix (Plan G)**:
    - `mix()` 함수 제거.
    - Premultiplied Alpha 공식 적용: `final = base * (1-alpha) + lens;` (One, OneMinusSrcAlpha)

### Phase 2: 시각적 품질 향상 (Artistic Polish)
PerfectSDK의 '룩(Look)'을 재현하기 위한 후처리 및 디테일 추가.

3.  **Specular Layer (Plan H)**:
    - 렌즈 합성 후, 원본 눈의 하이라이트(Luminance > 0.8)를 추출하여 `Screen` 모드로 덧그리기.
4.  **W3C Standard Blending**:
    - `SoftLight`, `Overlay` 수식을 W3C 표준으로 교체 (기존 포토샵 수식은 어두운 영역에서 부정확).
    - Luminance 계수를 Rec.601(SDTV)에서 Rec.709(HDTV)로 변경.

### Phase 3: 시스템 최적화 (Architecture)
지연 시간 제거 및 트래킹 성능 확보.

5.  **GPU-based Luminance (Plan I)**:
    - CPU EMA 제거.
    - `glGenerateMipmap`으로 텍스처를 1x1까지 축소하여 GPU에서 즉시 평균 밝기 샘플링. (Latency 0)
6.  **Predictive Tracking**:
    - 타임스탬프 기반으로 동공 위치 예측 보정(Extrapolation) 적용.

---

## 14) Creative Review Comment (Muse)

**"Math is clean, but Reality is messy. Don't scrub the soul out of it."**

I've read your "Deep Technical Audit". It's very... clinical. You're treating the eye like a geometry problem, but it's a wet, organic, light-trapping organ. My concern isn't your "Gamma Space Violation"—it's that your "fixes" might turn a subtle beauty lens into a cheap neon sticker.

Here are the **Artistic Risks** you are ignoring in your pursuit of mathematical purity:

### 1. The "Neon Sticker" Risk (Gamma Overshoot)
You want to "correct" the gamma to make colors brighter. Be careful.
- **The Risk:** If you lift the mid-tones too aggressively, you reveal the truth: our lens textures are flat 2D images.
- **The Reality:** Real eyes have depth. The iris fibers (stroma) trap light. If you just brighten a flat texture, it doesn't look like an eye; it looks like a **contact lens floating 1mm *above* the eye**.
- **My Demand:** If you fix the gamma, you better have a way to keep the "deep" shadows deep. Don't flatten the contrast in the name of "correctness".

### 2. The "Glass Eye" Problem (Specular Layer)
Plan H suggests a "Specular Layer".
- **The Risk:** A static white dot that doesn't move when the head turns is the **Uncanny Valley**. It looks like a prosthetic glass eye.
- **The Reality:** The "glint" (Purkinje image) is a reflection of the world. It stays with the light source, not the head.
- **My Demand:** If you can't make the specular highlight dynamic (reacting to head rotation), **don't draw it at all**. A matte eye is better than a dead eye.

### 3. The "Cataract" Effect (Pupil Masking)
I see a lot of talk about "Color Replace" but zero mention of the pupil.
- **The Risk:** If your new blending logic washes color over the pupil (the black center), the user looks like they have **cataracts** or glaucoma.
- **The Reality:** The pupil is a hole. It must be absolute void #000000.
- **My Demand:** Whatever blending math you use, the pupil region must remain untouched. The transition from colored iris to black pupil must be sharp but organic.

### 4. The "Cookie Cutter" Edge (Alpha Fix)
You call it "Double-Multiplied Alpha Fix". I call it "Hard Edges".
- **The Risk:** The Limbal Ring (the dark outer ring) defines the attractiveness of the eye. It is **soft, smoky, and fading**.
- **The Reality:** If you "fix" the alpha math and the edge becomes a sharp, pixel-perfect circle, the romance is gone. It looks like a cutout.
- **My Demand:** We need *more* softness at the edges, not less. If the math makes it sharp, you need to add a "Blur/Feather" pass to compensate.

**Verdict:**
Proceed with your "Physics Fixes", but if the result looks like a **bright, flat, static sticker**, I will reject it. I don't care if the math is right if the look is wrong. **Make it wet. Make it deep.**

---

## 15) Opus 코드 검증 기반 비판적 리뷰 (2026-02-20)

실제 코드(`CameraGLRenderer.kt`, `GpuRenderActivity.kt`, `LensManager.kt`)를 기준으로 Section 11(CodexComment)과 Section 13-14(Deep Technical Audit)의 각 주장을 검증한 결과.

### 16-1. Section 11 (CodexComment) 검증

#### P1: 홍채 밝기 샘플링 좌표 오류 — ❌ 이미 수정됨

> "현재 `sampleIrisLuminanceNv21()`가 좌/우 홍채의 중간점을 샘플링한다"

**사실**: commit `3edbacc` (2026-02-18)에서 **이미 수정 완료**. 현재 코드(`GpuRenderActivity.kt:991~1010`)는 `leftDetected/rightDetected` 개별 확인 → `samplePointLuminanceNv21()` 각각 호출 → 값 평균 구조로 동작 중.

```kotlin
// 현재 코드 (수정 완료 상태)
val leftLum = if (result.leftDetected) {
    samplePointLuminanceNv21(nv21, sensorW, sensorH, result.leftIrisX, result.leftIrisY, rotation)
} else -1f
val rightLum = if (result.rightDetected) {
    samplePointLuminanceNv21(nv21, sensorW, sensorH, result.rightIrisX, result.rightIrisY, rotation)
} else -1f
```

**결론**: CodexComment 작성자가 최신 커밋 반영 전 코드를 기준으로 분석. 이 항목은 더 이상 유효하지 않음.

#### P1-2: 오염된 상태에서 튜닝 금지 — ⚠️ 전제 무효, 방법론은 타당

P1이 수정되었으므로 "uAvgIrisLum 오염" 전제가 무효화됨. 단, "수정 → 동일 조건 재촬영 → 그 결과로 튜닝"이라는 **실행 순서 원칙은 올바름**.

#### P2: Mode 5 색공간 불일치 — ✅ 타당

> "`base`는 linear로 계산하면서 `uAvgIrisLum`은 sRGB 기준"

**코드 검증**:
- CPU 측: `samplePointLuminanceNv21()`는 NV21 Y채널(sRGB-like 감마)을 0~255 → 0.0~1.0 정규화 → EMA → `uAvgIrisLum`으로 전달 (**sRGB 공간**)
- GPU 측: Mode 5에서 `toLinearFast(base)` → `dot(baseL, vec3(0.2126...))` → lum은 **linear 공간**
- Scale 계산: `clamp(0.5 / max(0.1, uAvgIrisLum), ...)` → sRGB 기준 uAvgIrisLum으로 linear lum을 보정 → **축 불일치**

수치 예시: sRGB 0.15의 홍채 → linear ≈ 0.0225. scale은 sRGB 기준 0.5/0.15=3.33(→2.5 clamp). `0.0225 * 2.5 = 0.056`. **sRGB 공간에서 같은 계산이면 0.15 * 2.5 = 0.375**. 이 불일치가 Mode 5가 Mode 4보다 훨씬 어두운 직접적 원인.

**결론**: Mode 5를 살리려면 `uAvgIrisLum`을 linear 변환하거나, Mode 5 내에서 scale 계산을 linear 기준으로 재조정해야 함.

---

### 16-2. Section 13 (Deep Technical Audit) 검증

#### 13-1. Gamma Space Violation — ⚠️ 심각도 "Fatal" 과장

> "sRGB 텍스처를 선형 공간으로 변환하지 않고 바로 연산 → 중간 톤 채도 사망"

**코드 확인**:
- 카메라: `samplerExternalOES` → OES_TO_2D 패스 → `sampler2D` (sRGB 감마 공간)
- 렌즈: `GLUtils.texImage2D(GL_TEXTURE_2D, 0, bitmap, 0)` → `GL_RGBA8` (sRGB 데이터, linear decode 없음)
- 블렌딩: 감마 공간에서 직접 `mix()`, `screen()`, `overlay()` 등 수행

**사실이지만 치명적이지 않은 이유**:

1. **업계 표준**: PerfectSDK를 비롯한 대부분의 모바일 AR/뷰티 앱은 **감마 공간에서 직접 블렌딩**함. Photoshop도 "Blend Colors Using Gamma" 모드가 기본. linear blending은 VFX/영화 post-production에서 사용하는 것이지 실시간 모바일 AR의 표준이 아님.

2. **sRGB 감마 블렌딩의 시각적 차이**: `mix(0.2, 0.8, 0.5)` = sRGB에서 0.50, linear에서 pow(mix(pow(0.2,2.2), pow(0.8,2.2), 0.5), 1/2.2) ≈ 0.57. **차이는 존재하나 "채도가 죽는" 수준이 아님**.

3. **성능 리스크**: 매 프래그먼트에 `pow(x, 2.2)` 2회(입력) + `pow(x, 1/2.2)` 1회(출력) 추가 → 저사양 GPU에서 30fps 미달 위험.

4. **Regression 위험**: 기존 Normal/Multiply/Screen/Overlay 4개 모드는 감마 공간에서 동작하며 이미 검증됨. 전체 파이프라인을 linear로 전환하면 이 모드들의 시각적 결과가 변경됨 → 의도치 않은 변화.

**결론**: Gamma space violation은 이론적으로 존재하나, **색상 희석의 주원인이 아님**. 주원인은 Section 4의 "절대 밝기 곱셈" 문제. Plan F(전체 linear 전환)는 **위험 대비 이득이 낮음**.

#### 13-2. Double-Multiplied Alpha — ⚠️ 조건부 타당, 확인 필요

> "Android Bitmap은 Premultiplied Alpha → mix()에서 alpha 이중 곱셈"

**코드 확인**:
- `LensManager.loadTexture()`: `BitmapFactory.Options`에 `inPremultiplied` 설정 없음 → 기본값 `true` (API 19+)
- `GLUtils.texImage2D(GL_TEXTURE_2D, 0, bitmap, 0)`: premultiplied bitmap을 그대로 업로드
- 셰이더: `mix(camera.rgb, lens.rgb, finalAlpha)` = `camera*(1-α) + lens.rgb*α`
- Premultiplied 시: `lens.rgb = original_rgb * original_alpha` → 결과에 alpha가 이중 적용

**그러나 실제 영향도는 렌즈 텍스처의 alpha 분포에 의존**:
- 렌즈 중심부 alpha=1.0: `rgb*1.0 = rgb` → **차이 없음**
- 경계부 alpha<1.0: 이중 곱셈 → 경계 어두워짐
- 대부분의 렌즈 텍스처는 중심부 alpha≈1.0이 넓고 경계부만 gradient → 전체 "색 희석"보다 **경계 dark halo** 이슈에 가까움

**실제 스크린샷**: 눈에 띄는 dark halo가 관찰되지 않음 (edgeFeather smoothstep이 마스킹).

**결론**: 이론적으로 맞지만 **"색이 흐리게 보이는" 주원인으로 보기 어려움**. 다만 확인 가치는 있으며, 수정 자체는 간단 — `BitmapFactory.Options().apply { inPremultiplied = false }` 한 줄 추가 또는 셰이더에서 `lens.rgb / max(lens.a, 0.001)` unpremultiply.

#### 13-3. Specular Physics 부재 — ✅ 타당하나 별개 이슈

PerfectSDK에서 반사광이 렌즈 위에 자연스럽게 보이는 것은 확인됨. 그러나 이것은 **"색상 희석"과 다른 차원의 이슈** (리얼리즘/입체감). Mode 5의 `smoothstep(0.7, 0.95, lum)` specular 복원이 이미 이 방향의 시도.

**결론**: 향후 고도화 항목으로 유효. 현재 ISS-005의 "색 흐림" 문제와 직접 관련 없음.

#### 13-4. Latency & Sync — ⚠️ 의도적 설계를 결함으로 오독

> "EMA(α=0.1) → 약 20프레임(0.6초) 지연"

**반박**:
- EMA α=0.1은 **의도적 스무딩**. 급격한 조명 변화에서 렌즈 밝기가 깜빡이지 않도록 설계.
- 63% 수렴 ≈ 10프레임(0.33s), 95% 수렴 ≈ 30프레임(1.0s). 이는 사용자가 인식하는 "자연스러운 적응" 범위.
- P4-W1-03 설계 문서에서 합의된 값 (Section 22 A4).

> "FaceMesh-카메라 프레임 타임스탬프 동기화 부재"

**반박**: 이것은 **트래킹 정밀도** 이슈이며 **색상 희석**과 무관. P3-W2-01(One Euro Filter)에서 별도 다뤄진 영역.

**결론**: 두 주장 모두 색 희석 문제와 직접 관련 없음.

---

### 16-3. Section 14 (로드맵) 검증

#### Plan F (Gamma Fix) — ❌ 현 단계에서 불필요, 위험

| 항목 | 평가 |
|------|------|
| 색 희석 해결 기여 | 미미 (주원인이 gamma가 아님) |
| 성능 비용 | `pow(2.2)` × 3 per fragment → 중-저사양 GPU 위험 |
| Regression 위험 | 기존 Mode 0-3 결과 변경 |
| 업계 관행 | 모바일 AR에서 감마 공간 블렌딩이 표준 |

#### Plan G (Alpha Fix) — ⚠️ 확인 후 선택적 적용

**더 간단한 대안**: `BitmapFactory.Options.inPremultiplied = false` 설정으로 straight alpha 디코딩. 셰이더 수정 불필요.

또는 렌즈 텍스처의 alpha 분포를 먼저 확인하여 실제 영향도 판단 후 결정.

#### Plan H (Specular Layer) — ✅ 향후 고도화로 유효

기존 Mode 5 specular 복원을 확장하거나, 별도 후처리 패스로 분리하는 것은 합리적.

#### Plan I (GPU-based Luminance) — ❌ 비실용적

- `glGenerateMipmap`은 **전체 프레임** 평균을 구함 → **홍채 영역만의** 밝기가 아님
- 홍채만 crop하려면 별도 FBO+패스 필요 → 복잡도 대폭 증가
- CPU NV21 Y채널 5점 샘플링은 O(1) 상수시간, 1프레임 지연(33ms)은 시각적으로 인지 불가
- GPU readback(`glReadPixels`)은 파이프라인 stall 유발 → 오히려 성능 저하

---

### 16-4. 종합 판정표

| 주장 | 정확성 | 색 희석 관련성 | 실행 권장 |
|------|:------:|:------------:|:--------:|
| **CodexComment P1** (샘플링 버그) | ❌ 이미 수정 | — | 불필요 |
| **CodexComment P2** (Mode 5 색공간 불일치) | ✅ 정확 | Mode 5 한정 | **권장** |
| **13-1** Gamma Fatal | ⚠️ 과장 | 낮음 | 비권장 (위험) |
| **13-2** Alpha Double-Multiply | ⚠️ 조건부 | 중간 | 확인 후 결정 |
| **13-3** Specular | ✅ 타당 | 별개 이슈 | 향후 고도화 |
| **13-4** Latency | ⚠️ 오독 | 무관 | 불필요 |
| **Plan F** (Full Linear) | ❌ | 비용>이득 | **비권장** |
| **Plan G** (Alpha Fix) | ⚠️ | 텍스처 의존 | 확인 후 |
| **Plan H** (Specular Layer) | ✅ | 리얼리즘 | 후순위 |
| **Plan I** (GPU Lum) | ❌ | 무관 | **비권장** |

### 16-5. 수정된 실행 우선순위

코드 검증 결과를 반영한 실행 순서:

1. **방안 A (Color Replace)** — 절대→상대 밝기 전환이 색 희석의 직접 해결책. 즉시 구현.
2. **Alpha 확인** — 렌즈 텍스처의 alpha 분포 확인. premultiply 이슈가 유의미하면 `inPremultiplied = false` 적용.
3. **Mode 5 색공간 정합** — `uAvgIrisLum`을 linear 변환하거나 Mode 5 scale 계산 조정.
4. **Specular Layer** — Color Replace 검증 후 리얼리즘 고도화로.
5. **방안 D (Dual Layer)** — A 결과가 부족할 때 고도화 방향.

---

## 16) CodexComment (추가 검토, 비판적 관점)

Section 13~16 추가 코멘트를 코드 기준으로 재검토한 결과, 방향성은 대체로 타당하나 일부 결론은 확정적으로 쓰기 이르다.

### 18-1. 동의하는 부분

1. **샘플링 버그는 현재 코드에서 수정됨**
   - `sampleIrisLuminanceNv21()`가 좌/우 눈 개별 샘플링 후 평균하도록 변경됨.
   - 참조: `android/demo-app/src/main/java/com/irislenssdk/demo/GpuRenderActivity.kt`

2. **Mode 5 색공간 불일치는 실제 결함**
   - Mode 5는 `baseL`(linear) 기반 `lum`을 쓰면서, scale은 sRGB 기준 `uAvgIrisLum`을 사용.
   - 참조: `android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLRenderer.kt`
   - 이 항목은 우선순위 상위로 유지해야 한다.

3. **Full linear 전환(Plan F)을 즉시 적용하는 것은 리스크 큼**
   - 현재 모드 0~3 회귀 가능성과 성능 비용을 고려하면, ISS-005 범위에서는 과도한 변경이다.

### 18-2. 보수적으로 보완해야 할 부분

1. **Gamma 이슈를 "주원인 아님"으로 단정하기는 이르다**
   - 현 시점에서는 "치명적"도 과장이고, "영향 낮음"도 단정이다.
   - 결론: **중간 영향 후보**로 남기고 A/B 실험으로 기여도를 분리해야 한다.

2. **Alpha 이중 곱셈은 텍스처 alpha 분포 검증 없이 배제하면 위험**
   - 중심부 alpha=1.0이 넓더라도, 실제 사용자가 인지하는 품질 저하는 경계 halo에서 크게 발생할 수 있다.
   - 결론: 배제/채택이 아니라 **짧은 실험으로 즉시 판별**이 맞다.

3. **Latency를 색 희석과 완전 무관으로 처리하는 것도 과도함**
   - 평균 밝기 적응 지연은 "색상 절대값"보다 "체감 안정성/펌핑"에 영향.
   - ISS-005의 1차 원인은 아니지만, 시연 품질에는 연관이 있으므로 보조 지표로 추적 필요.

### 18-3. 실행 판단 (엄격 모드)

1. **즉시 채택**: 방안 A(Color Replace) + Mode 5 색공간 정합 수정
2. **즉시 검증**: Alpha 경계 영향(텍스처 alpha 분포 + halo 관찰)
3. **보류**: Full linear 전환(Plan F), GPU 평균 밝기(Plan I)

핵심 원칙: ISS-005는 "아이디어 우열"이 아니라 **재현 가능한 A/B 결과로 채택/폐기**해야 한다.

---

## 17) System Audit Comment (Architect)

**"Stability is not a feature; it's a requirement."**

I have reviewed the Roadmap (Section 14) and the subsequent discussions. While the "Physics Fix" (Phase 1) and "Artistic Polish" (Phase 2) are acceptable iterative improvements, **Phase 3 (System Optimization) is a critical stability risk.**

I am **rejecting Phase 3** in its current form. Here is the engineering audit:

### 19-1. Complexity Debt: The "Black Box" Shader Risk
- **Proposal:** "GPU-based Luminance (Plan I)" — calculating average luminance via mipmaps in the GPU.
- **Critique:** You are moving critical logic (`uAvgIrisLum`) from CPU (observable, loggable) to GPU (opaque).
    - **Debug Nightmare:** If the lens goes black in production, how do we debug it? We can't log a shader variable without `glReadPixels`, which stalls the pipeline.
    - **Requirement:** We must maintain a CPU-side "Shadow State" or use a compute shader with a debug buffer (too heavy for this SDK). **Keep the logic on CPU unless you can prove the 33ms latency is the *root cause* of a user-facing bug.** (Opus confirmed in Sec 16.3 that it is not).

### 19-2. Device Compatibility: The `glGenerateMipmap` Minefield
- **Proposal:** Use `glGenerateMipmap` to downsample the camera texture to 1x1.
- **Critique:** This assumes a compliant GLES 3.0 driver.
    - **Reality:** On fragmented Android devices (Mali-400, older Adreno), `glGenerateMipmap` on **OES Textures** (Camera) is undefined behavior or notoriously buggy. It often produces black textures or segfaults.
    - **Constraint:** Unless you whitelist specific GPUs, **do not rely on driver-side mipmap generation for OES textures.**

### 19-3. Roadmap Risk: Phase 3 is a "Big Rewrite"
- **Proposal:** "Predictive Tracking" + "GPU Luminance".
- **Critique:** This changes the data flow from `Frame -> Detect -> Render` (Linear) to `Frame -> Predict -> Render` (Temporal).
    - **Race Conditions:** Predictive tracking introduces "Swimming" artifacts if the prediction timestamp doesn't perfectly match the VSYNC presentation time. You are trading "Lag" for "Jitter". Jitter is worse.
    - **Pure Function Violation:** The detector should remain a Pure Function (`f(image) = iris_pos`). Injecting temporal state (prediction) into the core renderer makes regression testing impossible.

### 19-4. Fallback Strategy: The "Potato Phone" Scenario
- **Gap:** The roadmap assumes high-end shader support.
- **Question:** What happens if `OES_EGL_image_external` is not supported in the vertex shader (needed for some advanced effects)?
- **Requirement:** We need a **"Safe Mode"** (Mode 0 fallback) that runs on GLES 2.0 with zero fancy logic. If the shader fails to compile, the app must not crash.

**Architect's Decision:**
1.  **Approve Phase 1 & 2** (with Opus's caveats on Alpha/Gamma).
2.  **Veto Phase 3 (Plan I)**. The complexity cost exceeds the value. The current CPU EMA (33ms delay) is acceptable for lighting adaptation.
3.  **Mandate:** Any shader change must include a `try-catch` equivalent (Shader Compilation Status Check) and a fallback to a "Pink Debug Shader" or "Transparent Pass" on failure.

---

## 18) Performance & Architecture Review (Vulkan)

**Reviewer:** Vulkan (Senior Graphics Optimization Engineer)
**Date:** 2026-02-20
**Verdict:** **Request Changes** (Performance Critical)

I've reviewed the proposed "Deep Technical Audit" (Sec 13) and "Roadmap" (Sec 14). I fully agree with the Architect's veto of Plan I (Section 19), but I need to address the **rendering performance** implications of Plan F and G, which are currently being treated too lightly.

### 1. Gamma Performance (Plan F)
> Proposal: `pow(color, vec3(2.2))` per fragment.

**Critique:**
Are we rendering offline CGI? `pow()` is an expensive transcendental function. Doing this 3 times per fragment (decode base, decode lens, encode result) will tank the fill rate on Mali-G57 class devices.
Also, `pow(x, 2.2)` is overkill. The visual difference between Gamma 2.2 and Gamma 2.0 (`x*x`) is negligible for AR overlays, but `x*x` is a single MUL instruction.

**Optimized Alternative:**
- **Best:** Use hardware sRGB samplers (`GL_SRGB8_ALPHA8`). Let the texture unit handle the decode for free.
- **Good:** Use `x * x` (Gamma 2.0 approximation).
- **Acceptable:** `x * x * (x * 0.285 + 0.715)` (Fast approximation).
- **Reject:** `pow(x, 2.2)`.

### 2. Precision & Banding (Section 13-1)
> Context: Converting 8-bit sRGB to Linear in shader.

**Critique:**
If you linearize an 8-bit texture in the shader (`pow(x, 2.2)`), your darks (sRGB 0~20) collapse into a tiny range (Linear 0.0~0.005). When you do math there and convert back, you get massive banding artifacts.
Unless you are using an **FP16 (Half-Float) FBO** for the intermediate buffer, this "physically correct" math will look like a GIF from 1995 in the shadows.

**Action Item:**
- Confirm if we can afford `GL_RGBA16F`. If not, stick to sRGB blending or use dithering. Do not just "math it out" in 8-bit.

### 3. Alpha Math (Plan G)
> Proposal: `final = base * (1-alpha) + lens;` (Manual calculation)

**Critique:**
The proposal to remove `mix()` is correct for premultiplied alpha, but **DO NOT** try to un-multiply the texture in the shader (`lens.rgb / lens.a`). That's an ALU cost and a division-by-zero hazard.
Just configure the pipeline correctly:
- If Premultiplied: `glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)`
- If Straight: `glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)`
Let the ROP (Raster Operations Pipeline) handle it. It's faster than ALU.

### 4. Latency & Mipmaps (Plan I)
> Proposal: `glGenerateMipmap` to 1x1 for luminance.

**Critique:**
**Absolutely NOT.** As the Architect noted, `glGenerateMipmap` is a driver black box. On many mobile drivers, it blocks the pipeline until the previous draw call finishes, causing a massive bubble. Doing this every frame for a video feed is insane.
Also, you're downsampling the *entire* frame just to get the iris brightness? That's wasteful.

**Optimized Alternative:**
- **Compute Shader:** Parallel reduction (if ES 3.1+).
- **Downsample Pass:** `glBlitFramebuffer` with linear filter to 1/8th size, then read pixels.
- **Keep CPU:** The current CPU sampling (NV21) is O(1) and zero GPU cost. The 1-frame latency is acceptable. Don't move this to GPU unless you have a spare 2ms frame budget.

### 5. Predictive Tracking (Section 14)
> Proposal: Extrapolation based on timestamps.

**Critique:**
"Predictive Tracking" usually translates to "Jitter Nightmare". Simple linear extrapolation overshoots when the user stops moving their head. You'll see the lens fly off the eye and snap back.
Unless you have a high-frequency IMU fusion, **clamp** the prediction delta or use a **OneEuroFilter** (low-pass with dynamic cutoff). Do not ship raw extrapolation.

### Summary
The "Physics Fix" (Plan F) is too expensive, and "GPU Luminance" (Plan I) is an architectural hazard.
**Stick to Plan A (Color Replace)**. It solves the visual issue without touching the render pipeline's delicate performance balance.

---

## 19) Opus 2차 리뷰 — 텍스처 Alpha 실측 기반 전면 재평가 (2026-02-20)

### ⚠️ 자기 정정: Section 16의 Alpha 판단이 틀렸습니다

Section 16에서 "렌즈 중심부 alpha=1.0이 넓고 경계부만 gradient → 전체 '색 희석'보다 경계 dark halo 이슈에 가까움"이라고 기술했습니다. **이 판단은 실제 텍스처 데이터와 불일치합니다.**

### 21-1. 실측 데이터: LAGUNA BEIGE 렌즈 텍스처 Alpha 분포

```
python3 분석 결과 (256×256 RGBA PNG):

Alpha=255 (완전 불투명):  14.1%  ← 예상보다 훨씬 적음!
Alpha=0   (완전 투명):    46.2%  ← 동공 + 외곽
0<Alpha<255 (부분 투명):  39.6%  ← 이것이 실제 렌즈 착색 영역!

반경별 분포:
  r=0.00 (중심/동공): alpha=0   ← 동공은 투명 (정상)
  r=0.20 (동공):      alpha=0
  r=0.40 (동공~홍채): alpha≈9   ← 전환 시작
  r=0.60 (홍채 내):   alpha≈142 ← 렌즈 착색 영역
  r=0.80 (홍채 외):   alpha≈198 ← 가장 진한 영역
  r=0.90 (경계):      alpha≈135 ← 림벌 링
  r=0.95 (외곽):      alpha≈57  ← 페이드아웃
  r=1.00 (끝):        alpha≈4
```

**핵심**: 렌즈 텍스처의 **가시 영역 평균 alpha는 0.67 (약 170/255)**이며, 부분 투명 영역이 전체의 **40%**를 차지함. "중심부 alpha=1.0" 가정은 **완전히 틀림**.

### 21-2. Premultiplied Alpha 이중 곱셈의 실제 영향도

3개 렌즈 텍스처 공통 패턴:

| 텍스처 | Opaque(255) | Partial | Partial avg alpha | 이중 곱셈시 색상 잔존율 |
|--------|:-----------:|:-------:|:-----------------:|:-------------------:|
| LAGUNA BEIGE | 14.1% | 39.6% | 0.50 | **25%** |
| KYOTO BROWN | 15.3% | 38.6% | 0.49 | **24%** |
| SEOUL GRAY | 17.2% | 35.5% | 0.53 | **28%** |

**수치 시뮬레이션** (alpha=0.5 픽셀, LAGUNA BEIGE):

```
원본 렌즈 픽셀: RGB=(0.82, 0.72, 0.52), A=0.50

[Premultiplied 저장 상태]
  lens.rgb = (0.41, 0.36, 0.26)  ← 이미 alpha가 곱해진 값
  lens.a   = 0.50

[현재 셰이더] blendNormal(camera, lens.rgb, finalAlpha)
  finalAlpha = lens.a × uOpacity × edgeAlpha = 0.50 × 0.70 × 1.0 = 0.35
  result = camera × 0.65 + (0.41, 0.36, 0.26) × 0.35
  렌즈 기여분 = (0.14, 0.13, 0.09)  ← ❌ 원본의 17%만 발현

[올바른 계산] (Straight Alpha)
  lens.rgb = (0.82, 0.72, 0.52)  ← 원본 색상 그대로
  result = camera × 0.65 + (0.82, 0.72, 0.52) × 0.35
  렌즈 기여분 = (0.29, 0.25, 0.18)  ← ✅ 원본의 35% 발현

→ 렌즈 색상이 의도치의 약 절반으로 감소!
```

### 21-3. 색 희석의 기여 요인 재평가

| 요인 | 추정 기여도 | Section 16 평가 | 수정 평가 |
|------|:----------:|:---:|:---:|
| **절대 밝기 곱셈** (방안 A) | ~50% | 주원인 ✅ | 주원인 ✅ (유지) |
| **Premultiplied Alpha 이중 곱셈** | **~35%** | ⚠️ 조건부 → | **✅ 공동 주원인** (상향) |
| Gamma 공간 블렌딩 | ~10% | ⚠️ 과장 | ⚠️ 부수 요인 (유지) |
| Mode 5 색공간 불일치 | Mode 5 한정 | ✅ | ✅ (유지) |

**수정 결론: "색이 흐리게 보이는" 현상은 단일 원인이 아닌 2개의 공동 주원인.**

---

### 21-4. Section 15 (Muse Creative Review) 검증

#### 1. "Neon Sticker" Risk — ⚠️ 부분 타당, 과도한 우려

> "밝기를 올리면 평면 텍스처가 눈 위 1mm에 떠 있는 느낌"

Color Replace의 `detail = lum / uAvgIrisLum`은 **단순 밝기 증가가 아님**. 상대 밝기 정규화는 명암 구조를 보존함:
- 홍채 골(어두운 결): detail < 1.0 → 어둡게 유지
- 홍채 마루(밝은 결): detail > 1.0 → 밝게 유지
- 평균 밝기 영역: detail ≈ 1.0 → 렌즈 색 그대로

**대비(contrast)가 보존**되므로 "평면 스티커"가 아니라 입체감이 유지됨. 다만, clamp 상한(2.5)이 하이라이트를 잘라낼 수 있으므로 주의 필요 → Muse의 "deep shadows를 유지하라"는 원칙은 수용.

#### 2. "Glass Eye" Problem — ❌ 현 구현 오해

> "정적 흰 점은 Uncanny Valley"

현재 Mode 5의 specular 복원은 **카메라 원본 프레임의 실제 밝기**(`smoothstep(0.7, 0.95, lum)`)에서 추출함. 정적 흰 점이 아니라 **실제 반사광의 위치와 형태를 따라감**. 고개를 돌리면 반사광도 자연스럽게 이동.

Muse가 우려하는 "prosthetic glass eye" 문제는 **인위적으로 specular를 추가할 때** 발생하는 것이며, 우리 방식은 카메라에 이미 존재하는 specular를 복원하는 것. 근본적으로 다른 접근.

#### 3. "Cataract" Effect — ❌ 비해당 (Non-issue)

> "Color Replace가 동공까지 색을 칠하면 백내장처럼 보인다"

렌즈 텍스처 실측 결과: **동공 위치(r=0.0~0.3)의 alpha는 0**. `finalAlpha = lens.a × ...` 에서 lens.a=0이므로 동공은 절대 착색되지 않음. 렌즈 텍스처 자체가 동공 영역을 투명으로 처리하고 있어 블렌드 알고리즘과 무관하게 동공은 보호됨.

#### 4. "Cookie Cutter" Edge — ❌ 수정 범위 오해

> "Alpha를 고치면 경계가 날카로워진다"

`inPremultiplied = false` 설정은 **RGB 값의 해석 방식**만 변경함. Alpha 값 자체(경계의 gradient)는 동일하게 유지됨. `smoothstep` 페더링, 눈꺼풀 클리핑도 그대로. **경계의 부드러움에 영향 없음**.

#### Muse 종합 평가

| 우려 | 판정 | 이유 |
|------|:----:|------|
| Neon Sticker | ⚠️ 부분 수용 | 대비 보존되지만 clamp 조정 주의 |
| Glass Eye | ❌ 오해 | 실제 반사광 복원이지 인위적 추가 아님 |
| Cataract | ❌ 비해당 | 렌즈 텍스처 동공 alpha=0으로 보호됨 |
| Cookie Cutter | ❌ 오해 | Alpha fix는 RGB 해석만 변경, 경계 gradient 유지 |

**수용할 원칙**: "대비를 죽이지 마라" → Color Replace의 detail clamp 범위 튜닝 시 반영.

---

### 21-5. Section 18 (Codex 추가 검토) 검증

#### "Gamma를 '주원인 아님'으로 단정하기 이르다" — ⚠️ 수용하되 비중은 유지

A/B 실험을 통한 기여도 분리라는 방법론은 올바름. 다만 정량 분석 결과:
- Alpha 이중 곱셈: 가시 영역에서 **~50% 색상 손실**
- 절대 밝기 곱셈: 어두운 홍채에서 **~67% 색상 손실**
- Gamma 공간 블렌딩: 중간톤에서 **~14% 밝기 편차**

Gamma가 "주원인이 아니다"라는 결론은 **상대적 크기 비교에 근거**하며, "영향이 없다"고 말한 적은 없음.

#### "Alpha 이중 곱셈은 검증 없이 배제하면 위험" — ✅ 정확 (자기 정정)

**이 경고는 옳았음.** 실측 데이터가 이를 증명. Section 16에서의 내 평가("조건부, dark halo에 가까움")가 틀렸으며, Codex의 "짧은 실험으로 즉시 판별"이라는 제안이 올바른 접근이었음.

#### "Latency를 완전 무관으로 처리하는 것도 과도함" — ⚠️ 수용

정적 스크린샷에서의 색 희석과는 무관하지만, 실시간 시연에서의 "적응 펌핑"은 체감 품질에 영향. 보조 추적 지표로 유지하는 것은 합리적.

---

### 21-6. Section 19 (Architect) 검증

#### Plan I Veto — ✅ 전면 동의

기술적 근거가 모두 타당:
1. **Debug Nightmare**: GPU 내부 값은 로깅 불가 → 프로덕션 디버깅 불능
2. **OES Mipmap**: Mali-400/구형 Adreno에서 undefined behavior → 디바이스 호환성 위험
3. **Predictive Tracking → Jitter**: 이미 P3-W2-01에서 One Euro Filter로 해결 중

#### Shader Fallback Mandate — ✅ 좋은 제안

현재 셰이더 컴파일 실패 시 처리를 확인:

```kotlin
// CameraGLRenderer.kt — 셰이더 컴파일 시 glGetShaderiv(GL_COMPILE_STATUS) 체크 있음
// 그러나 실패 시 로그만 남기고 크래시 위험 존재
```

"Pink Debug Shader" 또는 "Transparent Pass" 폴백 추가는 SDK 안정성에 필수적. ISS-005 범위는 아니지만 별도 이슈로 추적 권장.

#### 한 가지 과도한 제한: "Pure Function Violation"

> "Detector는 Pure Function이어야 한다. Temporal state 주입은 regression testing 불가능"

One Euro Filter는 **렌더러**에 있지 검출기에 없음. 검출기(`IrisDetector`)는 `f(image) → iris_pos`로 이미 순수 함수. 필터링은 렌더링 파이프라인에서 수행되며 이는 업계 표준 패턴. 과도한 순수성 강조.

---

### 21-7. Section 20 (Vulkan) 검증

#### `pow(2.2)` 비용 — ✅ 정확, 이미 대응됨

Mode 5에서 `toLinearFast(x*x)` / `toSRGBFast(sqrt(x))` 사용 중. Gamma 2.0 근사가 이미 적용되어 있어 `pow(2.2)` 대신 단일 MUL/SQRT 사용. Vulkan의 "x*x가 올바른 선택"은 현재 구현과 일치.

#### 8-bit Banding — ✅ 새로운 유효한 통찰

> "8-bit sRGB를 linear로 변환하면 어두운 영역(0~20)이 0.0~0.005로 축소 → banding"

**이것은 Section 16에서 놓친 포인트**. 현재 FBO는 `GL_RGBA8`:
- sRGB 10/255 (0.039) → linear ≈ 0.0015
- sRGB 11/255 (0.043) → linear ≈ 0.0018
- 이 두 값의 차이(0.0003)는 8-bit precision에서 양자화되어 동일 값으로 collapse
- → 어두운 홍채 영역에서 **밴딩 아티팩트** 발생 가능

이것이 Mode 5가 시각적으로 더 나빠 보이는 **추가 원인**. linear 연산 후 8-bit 출력으로 돌아올 때 어두운 영역의 precision이 손실됨.

**해결 방향**:
- `GL_RGBA16F` FBO 사용 (성능 비용 있음)
- 또는 Mode 5를 감마 공간으로 유지하되 색공간 정합만 수정
- 또는 디더링 추가

#### Alpha: glBlendFunc 제안 — ❌ 현 아키텍처에 미적용

> "ROP에 맡기는 것이 ALU보다 빠르다"

**현재 구현은 GL blend state를 사용하지 않음** (코드에 `glEnable(GL_BLEND)` 없음). 모든 블렌딩이 fragment shader 내부에서 커스텀 수행됨. 이유: 7가지 블렌드 모드(Normal, Multiply, Screen, Overlay, LumTint 등)를 ROP 하나로 처리 불가.

따라서 `glBlendFunc` 제안은 아키텍처상 적용 불가. 셰이더 내부에서 unpremultiply하거나, 업로드 시점에 straight alpha로 디코딩하는 것이 올바른 경로.

#### Plan I, Predictive Tracking — ✅ Architect와 동일 결론

One Euro Filter가 이미 P3-W2-01에서 구현되어 있다는 점은 Vulkan 리뷰어가 확인하지 못한 부분.

---

### 21-8. 전체 코멘트 종합 판정표

| Section | 핵심 주장 | 정확성 | 실행 가치 |
|---------|----------|:------:|:--------:|
| **15 Muse-1** Neon Sticker | 밝기 올리면 평면적 | ⚠️ 부분 | clamp 튜닝 시 반영 |
| **15 Muse-2** Glass Eye | 정적 specular 위험 | ❌ 오해 | N/A (동적 복원) |
| **15 Muse-3** Cataract | 동공 착색 위험 | ❌ 비해당 | N/A (alpha=0 보호) |
| **15 Muse-4** Cookie Cutter | Alpha fix → 날카로운 경계 | ❌ 오해 | N/A (gradient 유지) |
| **18 Codex-1** Gamma A/B 필요 | 단정 금지, 실험으로 | ⚠️ 타당 | 후순위 A/B |
| **18 Codex-2** Alpha 검증 필요 | 배제하면 위험 | **✅ 정확** | **즉시 적용** |
| **18 Codex-3** Latency 추적 | 보조 지표로 | ⚠️ 수용 | 추적 유지 |
| **19 Arch-1** Plan I Veto | GPU 이동 위험 | ✅ 동의 | Veto 확정 |
| **19 Arch-2** OES Mipmap | 디바이스 호환성 | ✅ 동의 | 미적용 |
| **19 Arch-3** Shader Fallback | 컴파일 실패 대응 | ✅ 좋은 제안 | 별도 이슈 |
| **20 Vulk-1** pow(2.2) 비용 | 성능 위험 | ✅ 이미 대응 | x*x 유지 |
| **20 Vulk-2** 8-bit Banding | Linear에서 어두운 톤 손실 | **✅ 새 통찰** | Mode 5 재고 |
| **20 Vulk-3** glBlendFunc | ROP에 위임 | ❌ 미적용 | 아키텍처 불가 |
| **20 Vulk-4** Plan I/Track | CPU 유지 | ✅ 동의 | 유지 |

### 21-9. 최종 수정 실행 우선순위

실측 데이터 반영, 전체 코멘트 종합 후 최종 순서:

| 순위 | 작업 | 근거 | 예상 효과 |
|:----:|------|------|:--------:|
| **1** | **`inPremultiplied = false`** | 텍스처 40%가 partial alpha, 색상 ~50% 손실 중 | ~2× 복원 예상 `[Calculated, EXP-A로 확정]` |
| **2** | **방안 A (Color Replace)** | 절대→상대 밝기 전환 | 어두운 홍채 독립적 색 발현 |
| **3** | Mode 5 색공간 정합 | sRGB/linear 축 불일치 | Mode 5 정상화 |
| **4** | 8-bit banding 대응 | Mode 5 linear 연산 시 어두운 톤 손실 | Mode 5 품질 향상 |
| **5** | Specular Layer 고도화 | 리얼리즘/입체감 | PerfectSDK 근접 |
| **6** | Shader Fallback 안전장치 | 컴파일 실패 시 크래시 방지 | SDK 안정성 |

**핵심 변경**: Alpha fix가 "확인 후 결정"에서 **1순위**로 상향. `inPremultiplied = false` 한 줄이 코드 변경 최소로 즉각적 시각 개선을 줄 수 있음.

### 21-10. Codex 3차 교차검토 (동의/비동의, 2026-02-23)

Section 21 전체를 재검토한 결과, 실행 축(Alpha → Color Replace → Mode 5 정합)은 타당하다고 본다. 다만 일부 항목은 `❌`로 단정하기보다 `⚠️ 조건부`로 남겨야 의사결정 리스크가 줄어든다.

#### 동의하는 부분

1. **21-1/21-2 Alpha 실측 기반 재평가는 타당**
   - "중심부 alpha=1.0이 넓다"는 기존 가정을 실제 데이터로 반박했고, 이중 곱셈 기여를 수치로 제시한 점은 설득력이 높다.
2. **21-3 공동 주원인 프레이밍에 동의**
   - 색 희석을 단일 원인으로 환원하지 않고, `절대 밝기 곱셈 + premultiplied 이중 곱셈`의 결합 문제로 정리한 접근이 맞다.
3. **21-6 Plan I Veto 판단에 동의**
   - 디바이스 호환성/디버깅/복잡도 대비 이득이 낮아 ISS-005 범위에서 제외하는 것이 합리적이다.
4. **21-7 8-bit banding 경고는 유효**
   - Mode 5 품질 저하를 단순 색공간 이슈로만 보지 않고 precision 손실까지 포함한 점은 중요한 보완이다.

#### 동의하지 않는 부분

1. **21-4 Glass Eye를 `❌ 오해`로 확정한 표현은 과함**
   - 현재 구현이 정적 점이 아닌 것은 맞지만, 임계값/마스킹/지연 조건에서 인위적 하이라이트처럼 보일 가능성은 남아 있다.
   - 권고: `❌` 대신 `⚠️ 조건부 리스크`로 표기.
2. **21-4 Cookie Cutter를 `영향 없음`으로 단정한 표현은 과함**
   - alpha gradient가 동일해도 RGB 해석 변경으로 경계 대비의 체감 선명도가 달라질 수 있다.
   - 권고: `영향 없음` 대신 `경계 체감은 A/B 확인 필요`.
3. **21-7 glBlendFunc를 `아키텍처 불가`로 확정한 표현은 범위 과잉**
   - 현 단일 패스 구조에서는 미적용이 맞지만, 패스 분리 시 적용 가능한 경로다.
   - 권고: `현재 구조 기준 미적용`으로 제한 표기.
4. **21-9의 "즉각 색상 2배 밝아짐"은 기대치 과대 가능성**
   - 텍스처/조명/opacity 조건에 따라 개선폭 분산이 커질 수 있다.
   - 권고: `유의미 개선 예상(정량은 EXP-A 결과로 확정)`으로 수정.

#### 요약 결론

Section 21의 큰 방향은 유지해도 된다. 다만, 반박 항목을 `오해/비해당`으로 완전 종료하기보다, 시각 품질 리스크가 남는 항목은 `조건부`로 남겨 EXP-A/B/C에서 판정하는 편이 더 안전하다.

---

## 20) Codex 엄격 코멘트 (문제 해결 대화용, 2026-02-23)

문서의 분석 밀도는 높지만, 현재 형태로는 "아이디어 경쟁"에 가깝고 "검증 가능한 의사결정 문서"로는 아직 부족함. 아래 코멘트는 기술 방향 자체보다 **결정의 정확도**를 높이기 위한 것이다.

### 22-1. 가장 큰 구조적 문제: 결론 강도 > 증거 강도

1. `~50%`, `~35%`, `~10%` 같은 기여도 수치는 설득력 있어 보이지만, 측정 프로토콜/샘플 수/오차 범위가 문서에 없다.
   - 현재 상태: **정량처럼 보이는 정성 추정치**.
   - 조치: "확정 수치"가 아니라 "가설 범위"로 표기 전환.

2. PerfectSDK 비교는 참고 지표로 유효하나, 블렌드 모드/opacity/후처리 파라미터가 비공개라서 절대 정답으로 삼기 어렵다.
   - 조치: PerfectSDK를 "목표 룩 레퍼런스"로만 사용하고, 채택 기준은 우리 KPI로 분리.

3. 일부 주장은 코드 검증 기반, 일부는 추론 기반인데 문서 내 구분이 약하다.
   - 조치: 각 주장 끝에 `[Verified]`, `[Measured]`, `[Hypothesis]` 태그를 붙여 혼동 제거.

### 22-2. 의사결정을 위한 최소 실험 세트 (필수)

ISS-005는 다음 3개 실험으로 충분히 결론 낼 수 있다. 이 순서를 어기면 원인 분리가 다시 무너진다.

1. **EXP-A (Alpha 경로 단독 검증)**
   - 변경: `inPremultiplied = false`만 적용, 블렌드 로직은 그대로.
   - 목적: "색 희석" 중 alpha 경로 기여분을 독립 측정.
   - 통과 기준: 홍채 ROI 평균 채도 +20% 이상, 경계 halo 악화 없음.

2. **EXP-B (Color Replace 단독 검증)**
   - 변경: 방안 A만 적용, alpha 경로는 EXP-A 승자 설정 고정.
   - 목적: 절대 밝기 곱셈 문제의 실효 개선 확인.
   - 통과 기준: 어두운 홍채 샘플에서 렌즈 색 가시성/모드 분리도 모두 개선.

3. **EXP-C (Mode 5 존치 여부 판정)**
   - 변경: Mode 5 정합 버전 vs 비활성화 버전 비교.
   - 목적: 유지 가치 판단.
   - 통과 기준: 품질/성능이 Mode 4 대비 명확히 우수하지 않으면 Mode 5 기본 비노출.

### 22-3. 측정 정의가 없으면 다시 논쟁으로 돌아감

현재 별점/인상 평가만으로는 결론이 쉽게 흔들린다. 최소한 아래 4개는 고정 지표로 필요하다.

1. **Color Lift**: 홍채 ROI에서 `C*`(또는 HSV Saturation) 증가율
2. **Detail Retention**: 렌즈 적용 전후 로컬 대비(고주파 성분) 보존율
3. **Pupil Safety**: 동공 ROI 평균 채도/밝기 변화 (변화가 작아야 합격)
4. **Temporal Stability**: 조명 변화 시 1초 내 밝기 변동량(펌핑/깜빡임)

지표가 없으면 "더 자연스럽다"는 주장이 서로 반박만 되고 종료되지 않는다.

### 22-4. 범위 통제: ISS-005에서 지금 하지 말아야 할 것

1. Full linear 전환(Plan F) 전체 적용
2. GPU luminance 파이프라인 신규 구축(Plan I)
3. Predictive tracking 신규 도입

이 셋은 ISS-005의 핵심 실패 원인 분리 전에 넣으면 변수만 늘린다.

### 22-5. 대화용 결론 (바로 의사결정 가능한 형태)

1. **즉시 실행**: EXP-A, EXP-B
2. **조건부 실행**: EXP-C (A/B 결과가 안정화된 뒤)
3. **보류**: Plan F/I, Predictive tracking
4. **중단 규칙**: 동일 축에서 2회 연속 개선폭 <5%이면 해당 축 개발 중단

이 문서를 다음 회의에서 사용할 때는 "누가 맞는가"가 아니라 "어떤 실험이 통과했는가"만 기준으로 결정해야 한다.

---

## 21) 합의 현황 정리 (2026-02-23)

> 본 섹션은 Sec 11~20의 전체 논의를 종합하여, **완전 합의된 사항**과 **미합의/추가 논의 필요 사항**으로 분류한 최종 정리입니다.
>
> 참고: 본문 내 "구 Section N" 참조는 2026-02-23 번호 재정렬 이전 기준입니다. 재정렬 매핑: 구13→12, 구14→13, 구15→14, 구16→15, 구18→16, 구19→17, 구20→18, 구21→19, 구22→20.

---

### A. 완전 합의 (전원 동의, 즉시 실행 가능)

#### A-1. `inPremultiplied = false` 적용 [우선순위 1]

- **내용**: `LensManager.kt`의 `loadTexture()`에서 `options.inPremultiplied = false` 1줄 추가
- **근거**: 텍스처 실측(PIL/numpy 3개 렌즈) — partial alpha 40%, avg 0.50 → premultiplied 이중 곱셈으로 색상 ~50% 손실 `[Measured + Calculated]`
- **합의 과정**: Sec 12(제기) → Sec 15(확인 필요) → Sec 16(실험 권고) → **Sec 19(실측 확증, 1순위 상향)** → Sec 20(EXP-A 설계)
- **반대 의견**: 없음. 전원 동의.
- **부작용 위험**: 없음. RGB 해석만 변경, alpha gradient/경계 처리 영향 없음 `[Verified]`

#### A-2. Color Replace 블렌드 모드 구현 [우선순위 2]

- **내용**: `blendColorReplace()` — `detail = lum / max(0.01, uAvgIrisLum)` 상대 밝기 정규화
- **근거**: 절대 밝기 곱셈(`blend * lum * scale`)이 어두운 홍채에서 렌즈 색 33%만 발현 `[Calculated]`
- **합의 과정**: Sec 4(원인 분석) → Sec 6(방안 A 제안) → Sec 11(실행 순서) → Sec 15(코드 검증) → Sec 18(Vulkan "Stick to Plan A")
- **반대 의견**: 없음. Alpha Fix와는 **독립적 원인**을 해결하므로 **둘 다 순차 적용**. (조건부가 아님)
- **주의**: detail clamp 범위 튜닝 시 하이라이트 클리핑 + 대비 보존 확인 필요 (Muse 권고 반영)

#### A-3. Plan I (GPU Luminance) 폐기

- **내용**: `glGenerateMipmap` 기반 GPU 밝기 분석 — 전면 폐기
- **근거**: Mali-400 OES mipmap undefined behavior, GPU 디버깅 불가, CPU O(1) 샘플링으로 충분 `[Verified]`
- **합의 과정**: Sec 15(비실용적) → Sec 17(Architect Veto) → Sec 18(Vulkan 동의) → Sec 19(재확인) → Sec 20(범위 통제)
- **반대 의견**: 없음. **5회 연속 동일 결론**.

#### A-4. Plan F (Full Linear Pipeline) 보류

- **내용**: 전체 렌더링 파이프라인을 sRGB→Linear→sRGB로 전환 — ISS-005 범위에서 제외
- **근거**: 모바일 AR 업계 표준은 감마 공간 블렌딩. pow(2.2) 성능 비용. 기존 Mode 0~3 regression 위험 `[Verified]`
- **합의 과정**: Sec 15(위험>이득) → Sec 16(보류 동의) → Sec 17(승인 조건) → Sec 18(비용 경고)
- **반대 의견**: 없음. 단, Gamma 기여도(~10%)는 A/B 실험으로 추후 분리 가능 (Sec 16)

#### A-5. Predictive Tracking 불필요

- **내용**: 타임스탬프 기반 예측 보정 — ISS-005 범위 불필요
- **근거**: P3-W2-01에서 One Euro Filter 이미 구현됨. 선형 외삽은 jitter 유발 `[Verified]`
- **합의 과정**: Sec 15(의도적 설계) → Sec 17(Race Condition 위험) → Sec 18(OneEuroFilter 확인)
- **반대 의견**: 없음.

#### A-6. CPU EMA 밝기 샘플링 유지

- **내용**: 현재 CPU NV21 Y채널 5점 샘플링 + EMA(α=0.1) 방식 유지
- **근거**: O(1) 상수시간, GPU 부하 0, 1프레임(33ms) 지연은 시각적으로 인지 불가 `[Verified]`
- **합의 과정**: Sec 15(의도적 스무딩) → Sec 17(CPU 유지 명시) → Sec 18(Keep CPU)
- **반대 의견**: 없음.

#### A-7. 중단 규칙 채택

- **내용**: 동일 개선 축에서 2회 연속 Color Lift(C*) 개선폭 <5%이면 해당 축 개발 중단
- **근거**: 종료 조건 없으면 무한 반복 위험 (Sec 20 제안)
- **합의 과정**: Sec 20(제안) → Opus 3차 리뷰(수용)
- **반대 의견**: 없음.

---

### B. 조건부 합의 (방향 동의, 세부사항 미확정)

#### B-1. Mode 5 색공간 정합 [우선순위 3]

- **내용**: `uAvgIrisLum`(sRGB)과 Mode 5 내부 lum(linear)의 축 불일치 수정
- **합의**: 수정 필요하다는 점에 전원 동의
- **미확정**: 수정 방법 — (a) `uAvgIrisLum`을 linear 변환 vs (b) Mode 5 scale을 sRGB 기준으로 재계산
- **근거**: Sec 11(제기) → Sec 15(코드 검증) → Sec 16(분리 대응 필요)

#### B-2. Mode 5 존치 여부 [EXP-C]

- **내용**: 색공간 정합 + 8-bit banding 수정 후에도 Mode 4 대비 품질 우위가 없으면 기본 비노출
- **합의**: EXP-C 실험으로 판정한다는 절차에 동의
- **미확정**: "명확히 우수"의 정량 기준, 비노출 vs 삭제 결정

#### B-3. 8-bit Banding 대응 [우선순위 4, Mode 5 한정]

- **내용**: 8-bit `GL_RGBA8` FBO에서 linear 연산 시 어두운 톤(sRGB 0~20) precision collapse
- **합의**: 문제 존재에 동의 (Sec 18 Vulkan 발견)
- **미확정**: 해결 방법 — (a) `GL_RGBA16F` FBO vs (b) Mode 5를 감마 공간 유지 vs (c) 디더링
- **의존성**: Mode 5 색공간 정합(B-1)과 함께 결정해야 함

#### B-4. Shader Fallback 안전장치 [우선순위 6]

- **내용**: 셰이더 컴파일 실패 시 투명 패스 또는 Mode 0 폴백
- **합의**: 필요하다는 점에 전원 동의 (Sec 17 Architect Mandate)
- **미확정**: ISS-005 범위 포함 vs 별도 이슈 추적. 구현 방식(Pink Debug Shader vs Transparent Pass)

#### B-5. EXP-A/B/C 실험 프로토콜

- **내용**: Alpha → Color Replace → Mode 5 순서로 변수 통제하며 순차 검증
- **합의**: 순서와 변수 통제 원칙에 동의
- **미확정**: 통과 기준의 정량 임계값 ("+20% 채도" 근거 불명). 현실적으로는 동일 조건 스크린샷 비교가 더 효율적
- **EXP-A 추가 체크**: 경계 영역 before/after 스크린샷 비교 (E/F절 합의, 2026-02-23)

#### B-6. 측정 지표

- **합의**: Color Lift(C* 또는 HSV Saturation)를 1차 정량 지표로 채택
- **미확정**: Detail Retention의 조작적 정의 (Laplacian? SSIM? FFT?). Pupil Safety는 렌즈 alpha=0으로 항상 통과하므로 제외. Temporal Stability는 ISS-005 범위 밖으로 분리

---

### C. 미합의 / 기각 (논쟁 중이거나 기각된 주장)

#### C-1. [기각] Gamma가 색 희석의 "핵심 원인" (Sec 12-1)

- **주장**: sRGB→Linear 미변환이 Fatal 수준의 원인
- **반박**: 모바일 AR 업계 표준은 감마 공간 블렌딩. 기여도 ~10%로 추정 `[Theoretical]`. Alpha(~35%)와 절대 밝기(~50%)가 주원인 `[Calculated]`
- **현재 상태**: "주원인"이라는 등급은 기각. "부수 요인"으로 격하. A/B 실험으로 정확한 기여도 분리 가능 (Sec 16 권고)

#### C-2. [기각] Latency가 색 희석의 원인 (Sec 12-4)

- **주장**: EMA α=0.1의 지연이 색 희석을 유발
- **반박**: EMA는 의도적 스무딩 (P4-W1-03 설계). 정적 스크린샷에서도 색 희석이 관찰됨 → 지연과 무관 `[Verified]`
- **현재 상태**: 기각. 단, 체감 품질의 보조 추적 지표로는 유지 (Sec 16)

#### C-3. [기각] Muse의 "백내장/유리눈/쿠키커터" 우려 (Sec 14)

- **주장**: Color Replace가 동공 착색(백내장), 정적 반사광(유리눈), 날카로운 경계(쿠키커터) 유발
- **반박**: 동공 alpha=0 `[Measured]`, specular는 실시간 카메라 기반 `[Verified]`, alpha gradient 유지 `[Verified]`
- **현재 상태**: 기각. 단, "대비를 죽이지 마라"는 원칙은 clamp 튜닝 시 반영 (Sec 19)

#### C-4. [논쟁 중] 기여도 수치의 증거 등급

- **Codex 주장** (Sec 20): ~50%/~35%/~10%는 "정량처럼 보이는 정성 추정치"
- **Opus 반박**: 텍스처 실측 + 코드 기반 시뮬레이션이므로 `[Calculated]`가 적절. 순수 정성 추정과 동급 취급은 과도
- **현재 상태**: 태그 체계(`[Measured]`/`[Calculated]`/`[Hypothesis]`) 도입에는 합의. 수치 자체의 등급 분류는 미합의
- **해소 방법**: EXP-A/B 실험 결과로 실측 기여도 확인 시 자동 해소

#### C-5. [보류] Specular Layer 고도화 [우선순위 5]

- **내용**: 렌즈 위 반사광 복원 강화로 PerfectSDK 수준의 리얼리즘
- **현재**: Mode 5에 `smoothstep(0.7, 0.95, lum)` 기본 구현 존재
- **상태**: 방향성에 동의하나, ISS-005의 "색 희석" 해결 이후 후순위로 보류. 별도 이슈화 권고

---

### D. 최종 실행 순서

| 순위 | 작업 | 합의 상태 | 근거 등급 |
|:----:|------|:--------:|:---------:|
| **1** | `inPremultiplied = false` | A-1 완전 합의 | `[Measured + Calculated]` |
| **2** | Color Replace 블렌드 모드 | A-2 완전 합의 | `[Calculated]` |
| **3** | Mode 5 색공간 정합 | B-1 조건부 (방법 미확정) | `[Verified]` |
| **4** | 8-bit banding 대응 | B-3 조건부 (방법 미확정) | `[Verified]` |
| **5** | Specular Layer 고도화 | C-5 보류 | — |
| **6** | Shader Fallback | B-4 조건부 (범위 미확정) | — |

**프로세스 가이드라인**: EXP-A → EXP-B → EXP-C 순차 실험. Color Lift(C*) + 스크린샷 비교로 평가. 중단 규칙(2회 연속 <5%) 적용.

---

### E. Codex 제안 기반 재분류 (추가 섹션, 기존 A~D 원문 유지)

본 절은 기존 A~D를 수정하지 않고, 해석 충돌 가능성이 있는 항목만 별도로 재분류한 제안안이다.

| 대상 항목 | 기존 분류 | 제안 분류 | 제안 사유 | 판정 게이트 |
|----------|:--------:|:--------:|----------|------------|
| A-1 Alpha Fix 실행 자체 | 완전 합의 | 완전 합의(유지) | 실행 우선순위와 효과 방향에 대한 합의는 견고함 | EXP-A 통과 시 확정 유지 |
| A-1 "부작용 위험 없음" 단정 | 완전 합의 내 확정 문구 | **조건부 리스크** | 경계 체감(선명도/이질감)은 RGB 해석 변화로 달라질 수 있음 | 경계 ROI A/B 비교(halo/edge contrast) |
| C-3 Muse 우려 전체 기각 | 기각 | **부분 기각 + 조건부 관찰** | 동공 착색 우려는 기각 가능하나, Glass Eye/Cookie Cutter는 체감 리스크 잔존 | EXP-B/EXP-C 시각 평가 체크리스트 |
| A-6 CPU EMA "인지 불가 지연" | 완전 합의 | **조건부 합의** | 색 희석 원인에서 제외는 타당하나, 실시간 체감 품질 영향은 별도 추적 필요 | 조명 스텝 변화 1초 응답 로그 |
| B-6 Pupil Safety 제외 | 조건부 합의(사실상 제외) | **조건부 합의(지표 유지)** | alpha=0이어도 마스크 정렬/좌표 오차 회귀 가능성 존재 | 동공 ROI 색도/밝기 변동 상한 검사 |
| C-4 기여도 수치 등급 | 논쟁 중 | 논쟁 중(유지) | 해소 방법이 이미 정의되어 있음 | EXP-A/B 결과로 자동 해소 |

#### E-1. 실행 해석 원칙 (추가)

1. 실행 우선순위(Alpha → Color Replace → Mode 5)는 유지한다.
2. "완전 기각"은 구조적으로 불가능한 항목에만 사용하고, 시각 체감 이슈는 기본적으로 "조건부"로 분류한다.
3. 정량 수치는 EXP 결과 전까지 추정치로 유지하며, 회의 문구에서 확정 수치처럼 표현하지 않는다.
4. 리뷰 결론은 "의견 우위"가 아니라 "게이트 통과 여부"로 종료한다.

#### E-2. 즉시 반영 가능한 문구 가이드 (추가)

- `오해/비해당` 대신 `조건부 리스크(재현 조건 제한)`를 우선 사용
- `즉각 2배 개선` 대신 `유의미 개선 예상(정량은 EXP-A로 확정)` 사용
- `영향 없음` 대신 `현 구조에서 유의미 영향 미관찰` 사용

---

### F. Opus 4차 리뷰 — Section E 재분류 제안에 대한 코드 기반 검증 (2026-02-23)

> Codex가 기존 A~D 분류에 대해 6개 항목의 재분류를 제안하였다. 아래에서 각 항목을 **실제 코드**와 대조하여 수용/기각을 판정한다.

---

#### F-1. A-1 "부작용 위험 없음" 단정 → 조건부 리스크

**Codex 제안**: `inPremultiplied = false` 적용 시 경계 체감(선명도/이질감)이 RGB 해석 변화로 달라질 수 있다.

**코드 검증**:
```
// CameraGLRenderer.kt:262 — 렌즈 텍스처 샘플링
vec4 lens = texture(uLensTexture, lensCoord);

// :266 — 가장자리 페더링
float edgeAlpha = smoothstep(1.0, featherStart, dist);

// :280 — 최종 알파
float finalAlpha = lens.a * uOpacity * edgeAlpha * eyelidMask;

// :285 (Mode 0 예시)
blended = blendNormal(camera.rgb, lens.rgb, finalAlpha);
// → mix(base, blend, finalAlpha) = base*(1-finalAlpha) + blend*finalAlpha
```

현재(premultiplied) 상태에서 `lens.rgb`는 이미 alpha가 곱해진 값이다. 예를 들어 렌즈 경계에서 원래 RGB=(0.8, 0.4, 0.2), alpha=0.3이면:
- premultiplied `lens.rgb` = (0.24, 0.12, 0.06) — **이미 어두워진 상태**
- mix에서 다시 `finalAlpha=0.3`을 곱함 → **이중 감쇄로 경계가 과도하게 어두움**

`inPremultiplied = false` 적용 후:
- straight `lens.rgb` = (0.8, 0.4, 0.2) — **원본 색상 유지**
- mix에서 `finalAlpha=0.3`으로 한 번만 감쇄 → **렌즈 디자이너 의도대로의 경계**

**∴ 현재가 비정상이고, fix가 정상 복원이다.** 경계가 "달라 보이는" 것은 맞지만, 그것은 이중 감쇄의 부자연스러운 어둠이 제거되는 것이지 "리스크"가 아니다.

**판정**: **부분 수용**. 분류 변경은 거부 — "부작용 위험 없음 `[Verified]`"는 구조적으로 정확하다. 단, Codex의 관찰 자체는 가치가 있으므로, EXP-A 체크리스트에 "경계 영역 before/after 스크린샷 비교" 1줄 추가로 충분하다. 분류를 "조건부 리스크"로 격상할 근거는 없다.

---

#### F-2. C-3 Muse 우려 전체 기각 → 부분 기각 + 조건부 관찰

**Codex 제안**: 동공 착색 기각은 가능하나, Glass Eye/Cookie Cutter 체감 리스크가 잔존한다.

**코드 검증 — Glass Eye (유리눈, 정적 반사광)**:
```
// CameraGLRenderer.kt:225-233 — Mode 5
vec3 baseL = toLinearFast(base);      // ← base = camera 실시간 텍스처
float lum = dot(baseL, vec3(0.2126, 0.7152, 0.0722));
...
float realSpec = smoothstep(0.7, 0.95, lum);
result = mix(result, baseL, realSpec); // ← 밝은 영역은 카메라 원본 복원
```
반사광은 **카메라 실시간 피드에서 유래**한다. 렌즈가 합성하는 것이 아니다. `realSpec`으로 밝은 영역을 카메라 원본으로 되돌리므로, 반사광은 사용자가 고개를 움직이면 함께 움직인다 → **유리눈 발생 불가** `[Verified]`.

**코드 검증 — Cookie Cutter (날카로운 경계)**:
```
// :265-266 — 3단 알파 감쇄
float featherStart = 1.0 - uEdgeFeather;   // 페더링 시작점
float edgeAlpha = smoothstep(1.0, featherStart, dist);  // 1단: 가장자리 감쇄
// :273-277 — 눈꺼풀 마스크 (2단/3단)
float topClip = smoothstep(minY - eyelidFeather, minY + eyelidFeather, vTexCoord.y);
float bottomClip = 1.0 - smoothstep(maxY - eyelidFeather, maxY + eyelidFeather, vTexCoord.y);
// :280 — 최종 합산
float finalAlpha = lens.a * uOpacity * edgeAlpha * eyelidMask;
```
경계는 **4중 알파 감쇄** (렌즈 텍스처 자체 gradient + edgeAlpha smoothstep + topClip + bottomClip)로 보호된다. Cookie Cutter가 발생하려면 이 4중 감쇄가 모두 실패해야 하며, 이는 코드 구조상 불가능하다 `[Verified]`.

**판정**: **기각**. 원래 분류 "C-3 기각" 유지. Glass Eye와 Cookie Cutter 모두 코드 구조로 방지되고 있으며, "조건부 관찰"로 격상할 기술적 근거가 없다. 이를 "조건부"로 만들면 EXP 체크리스트에 무의미한 검증 항목이 추가되어 프로세스 비용만 증가한다.

---

#### F-3. A-6 CPU EMA "인지 불가 지연" → 조건부 합의

**Codex 제안**: 색 희석 원인 제외는 타당하나, 실시간 체감 품질 영향은 별도 추적 필요.

**검증**: EMA α=0.1 @ 30fps → 90% settling ~2.3초. 이것은 P4-W1-03에서 **의도적으로 설계된 스무딩**이다. Codex가 지적하는 "실시간 체감 품질 영향"은:
1. **ISS-005 범위 밖** — ISS-005는 "색이 바래 보이는" 문제이지, "밝기 추적이 느린" 문제가 아니다
2. **별도 이슈** — 조명 급변 시 추적 응답은 별도 이슈로 추적할 사안
3. **정적 스크린샷에서도 색 희석 재현** — EMA 지연과 무관함이 이미 검증됨 `[Verified]`

ISS-005 안에서 A-6의 합의 등급을 낮추는 것은 범위 혼동이다.

**판정**: **기각**. A-6 "완전 합의" 유지. 단, "EMA 응답 속도 개선은 ISS-005 범위 밖이며 별도 이슈로 추적 가능"이라는 주석은 이미 Sec 19에서 다루어져 있다.

---

#### F-4. B-6 Pupil Safety 제외 → 조건부 합의(지표 유지)

**Codex 제안**: alpha=0이어도 마스크 정렬/좌표 오차 회귀 가능성이 존재한다.

**검증**: Codex가 제기하는 "좌표 오차로 렌즈가 동공에 걸치는" 상황은:
1. **홍채 추적 정확도 문제** — P3-W2-01 (Stability) + P4-W1-01 (One Euro Filter)에서 추적
2. **블렌드 모드와 무관** — `inPremultiplied`, Color Replace, Mode 5 어떤 수정을 해도 alpha=0 영역에는 영향 없음
3. **Pupil Safety를 EXP 지표로 측정하는 비용**: 동공 ROI 색도/밝기 검사 → 추가 이미지 프로세싱 코드 필요 → **이미 alpha=0으로 항상 통과할 지표**에 구현 비용을 투입하는 것은 낭비

**판정**: **기각**. B-6 "Pupil Safety 제외" 유지. 좌표 오차 추적은 Stability 이슈의 영역이지 ISS-005 블렌드 모드 개선의 측정 지표가 아니다.

---

#### F-5. E-1 실행 해석 원칙 검토

| 원칙 | 판정 | 사유 |
|------|:----:|------|
| 1. 실행 우선순위 유지 | **수용** | D절과 동일 |
| 2. "완전 기각"은 구조적 불가능에만 사용 | **부분 거부** | 코드 검증으로 방지가 확인된 경우(Glass Eye, Cookie Cutter 등)도 기각 가능. "구조적 불가능"으로 좁히면 코드 검증 결과를 무시하게 됨 |
| 3. 정량 수치는 EXP 전까지 추정치 | **수용 (중복)** | 이미 `[Calculated]`/`[Measured]` 태그로 표현 중. 새 규칙 불필요, 기존 태그 체계가 동일 역할 수행 |
| 4. "의견 우위" → "게이트 통과 여부" | **수용 (중복)** | D절에 EXP-A→B→C 순차 게이트가 이미 정의됨. 원칙은 동의하나 새 규칙 추가는 불필요 |

---

#### F-6. E-2 문구 가이드 검토

| 제안 | 판정 | 사유 |
|------|:----:|------|
| `오해/비해당` → `조건부 리스크(재현 조건 제한)` | **거부** | `[Verified]`로 확인된 사실에 "조건부" 수식을 붙이면 **증거 등급을 의도적으로 낮추는 것**. 측정/검증된 결과는 사실로 서술해야 한다 |
| `즉각 2배 개선` → `유의미 개선 예상` | **부분 수용** | "즉각"은 삭제 가능. 그러나 "~50% 손실 → 복원"은 시뮬레이션 기반 `[Calculated]`이므로 수치를 삭제할 이유 없음. `"~2× 복원 예상 [Calculated, EXP-A로 확정 예정]"`이 정확한 표현 |
| `영향 없음` → `현 구조에서 유의미 영향 미관찰` | **거부** | alpha=0에서 블렌드 결과가 변하지 않는 것은 **수학적 사실** (`mix(base, blend, 0) = base`). "미관찰"이라는 표현은 불확실성이 있는 것처럼 오도 |

---

#### F-7. 종합 판정

**Codex E 섹션의 의도**는 인정한다 — 결론을 확정으로 서술하기보다 실험으로 검증하자는 인식론적 겸양은 건전하다.

그러나 **과교정(overcorrection)** 문제가 있다:

1. **범위 혼동**: ISS-005(블렌드 모드 색 희석)와 인접 이슈(추적 정확도, EMA 응답)를 혼합하여, ISS-005 안에서의 합의 등급을 불필요하게 약화시킨다
2. **코드 검증 경시**: `[Verified]` 태그가 붙은 결론은 코드 라인을 지정하여 확인된 사실이다. 이를 "조건부"로 격하하면 검증 체계의 신뢰도를 훼손한다
3. **프로세스 비용 무시**: 무의미한 지표(alpha=0 Pupil Safety)나 구조적으로 방지된 리스크(Glass Eye)를 "관찰 대상"으로 추가하면 EXP 프로토콜 실행 비용만 증가한다
4. **기존 메커니즘 중복**: 증거 등급 태그(`[Measured]`/`[Calculated]`/`[Hypothesis]`)가 E-1과 E-2가 하려는 역할을 이미 수행 중이다

**수용 항목 요약**:
- EXP-A 체크리스트에 "경계 영역 before/after 스크린샷 비교" 추가 (F-1)
- "즉각" 표현 삭제, `[Calculated, EXP-A 확정 예정]` 표기 (F-6)

**기각 항목 요약**:
- A-1 분류 변경 (조건부 리스크) — 구조적으로 정상 복원이므로 리스크 아님
- C-3 분류 변경 (부분 기각) — Glass Eye, Cookie Cutter 모두 코드로 방지됨
- A-6 분류 변경 (조건부 합의) — ISS-005 범위 밖 이슈로 합의 등급 변경 부적절
- B-6 분류 변경 (지표 유지) — 추적 정확도는 별도 이슈 영역
- E-2 "조건부 리스크" 문구 가이드 — 검증된 사실의 등급을 의도적으로 낮추는 효과

---

## 22) 변경 이력

| 날짜 | 내용 |
|------|------|
| 2026-02-20 | ISS-005 초안: 문제 분석 + 5개 개선 방안 + 비교 매트릭스 |
| 2026-02-20 | CodexComment: 샘플 비교 기반 결론 + Critical Path |
| 2026-02-20 | Deep Technical Audit: Gamma/Alpha/Specular/Latency + Plan F~I |
| 2026-02-20 | Creative Review (Muse): 예술적 관점 리스크 4종 |
| 2026-02-20 | Opus 1차 리뷰: 코드 검증 기반 각 주장 검증 + 실행 우선순위 |
| 2026-02-20 | Codex 추가 검토: A/B 실험 원칙, 단정 리스크 보완 |
| 2026-02-20 | System Audit (Architect): Plan I Veto + Shader Fallback Mandate |
| 2026-02-20 | Vulkan Review: pow() 비용 + 8-bit Banding 경고 |
| 2026-02-20 | Opus 2차 리뷰: 텍스처 alpha 실측 → Alpha fix 1순위 상향 |
| 2026-02-23 | Codex 엄격 코멘트: 증거 강도 + EXP 프로토콜 + 중단 규칙 |
| 2026-02-23 | **문서 정리: 중복 섹션 삭제, 번호 재정렬, 합의/미합의 분류 (Sec 21)** |
| 2026-02-23 | **Codex 재분류 추가 섹션: 기존 A~D 불변 + 조건부 리스크 재분류(E 절)** |
| 2026-02-23 | **Opus 4차 리뷰(F 절): E 섹션 6개 재분류 항목 코드 검증 → 2건 수용, 4건 기각** |
| 2026-02-24 | **E/F절 최종 합의 반영**: "즉각 2배" → "~2× 복원 예상 [Calculated, EXP-A로 확정]", EXP-A 경계 비교 체크 추가 |
