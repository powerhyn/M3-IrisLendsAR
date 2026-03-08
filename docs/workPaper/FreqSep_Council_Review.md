# FreqSep 품질 분석 리포트 — Agent Council 리뷰 결과

> **프로젝트**: IrisLensSDK — GPU Beauty Filter
> **작성일**: 2026-03-08
> **원본 리포트**: `FreqSep_Quality_Analysis_Report.md`
> **리뷰 참여 AI**: Codex (OpenAI o4-mini), Gemini (Google), Claude (Anthropic, 의장)
> **목적**: 원본 리포트 Section 5 "의견 요청 사항" Q1~Q5에 대한 다중 AI 관점의 종합 의견

---

## Q1. Phase 1 (A~D) 적용 순서와 우선순위

### 만장일치 권장 순서: `D → C → B → A`

| 순서 | 개선안 | 이유 |
|:---:|--------|------|
| 1st | **D. Linear RGB 색공간 전환** | 전체 수식의 기준 좌표계가 바뀌므로 가장 먼저 고정해야 재튜닝 1회로 끝남 |
| 2nd | **C. Overlay 합성 전환** | 합성 특성이 크게 바뀌므로 Linear 전환 이후에 맞추는 게 안전 |
| 3rd | **B. 톤커브 미드톤 리프트** | 합성 결과의 전체 톤 밸런스 조정 |
| 4th | **A. Luminance Sharpen** | 최종 출력 선명도 보정은 파이프라인 맨 끝이 제어 용이 |

### C+D 동시 적용 상호작용 이슈 — 있음 (주의 필요)

Overlay 공식은 감마/지각 공간(sRGB) 기준으로 설계된 경우가 많아서, Linear 공간에서 그대로 사용하면 미드톤/하이라이트 반응이 달라진다.

**해법 (두 가지 접근)**:

1. **Luma 분리 적용** (Codex): Overlay는 luma만 지각 공간(sRGB/log)에서 적용하고, 색상(chroma)은 additive/원본 기준 유지
2. **강도 보정** (Codex): Linear Overlay 사용 시 강도를 20~40% 낮추고 pivot(0.5 기준점)을 재튜닝

**Gemini 추가 의견**: Linear 전환 없이 Overlay만 적용하면 "피부가 타거나(Burn)" "그레이 캐스트(Gray cast)"가 생길 위험이 크므로, D를 반드시 C보다 먼저 적용해야 한다.

---

## Q2. Additive vs Overlay 합성의 실질적 차이

### 두 방식 비교

| 항목 | Additive | Overlay |
|------|----------|---------|
| **장점** | 안정적, 프레임간 깜빡임 적음, 색 왜곡 적음, 구현 단순 | 피부 결 대비 살아남, 입체감(Contrast) 보존, "또렷함" |
| **단점** | 평평/플라스틱 느낌, 값이 범위 벗어날 수 있음 | 잡티·노이즈 재강조, 채널별 색 틀어짐 위험, 8-bit 계조 깨짐(Banding) |

### 권장 접근

**Codex — 하이브리드**:
```glsl
vec3 out = mix(additive_out, overlay_luma_out, w);
// w = 0.2 ~ 0.4 범위, skin/edge 마스크 기반 조절
// 베이스는 additive 안정성 유지 + luma만 Overlay로 질감 보강
```

**Gemini — Soft Light 추천**:
- Soft Light는 Overlay의 "약한 버전"으로, 대비 변화가 완만
- 모바일 8-bit 환경에서 계조 깨짐(Banding)이 적고 더 자연스러움
- Overlay보다 안전한 출발점

### 의장 종합

Soft Light + Linear RGB 조합이 가장 실용적인 출발점이다. Soft Light로 시작하여 결과를 평가한 뒤, 더 강한 질감 보강이 필요하면 하이브리드(additive + overlay luma blend)로 확장하는 2단계 접근이 합리적이다.

---

## Q3. smoothstep 감쇠의 구조적 한계 대안

### 문제 재확인

현재 `magnitude = luminance(abs(high))` 단일 신호로는:
- **넓은 색소침착**: 큰 공간 크기, 작은 진폭 → 보존됨 (제거해야 하지만 못함)
- **선명한 피부 주름**: 작은 공간 크기, 중간 진폭 → 제거됨 (보존해야 하지만 못함)

### Phase 1 범위 내 해법: 3신호 결합

두 AI 모두 동일한 방향 — **Edge/Gradient 신호 추가**로 공간 주파수를 근사적으로 구분.

```glsl
// 1. 에지 강도 (Sobel 또는 3x3 에지 커널)
float edge = sobelMagnitude(uOriginal, vTexCoord);

// 2. Chroma 편차 (색소침착 감지용)
vec3 chromaDiff = orig - low;
float chromaDev = length(chromaDiff.gb - chromaDiff.rr);  // G/B 대비 R 편차

// 3. 기존 magnitude
float magnitude = dot(abs(high), vec3(0.299, 0.587, 0.114));

// 결합 감쇠 공식
float baseFactor = smoothstep(uAttenuationLow, uAttenuationHigh, magnitude);
float edgeProtect = 1.0 - kEdge * saturate(edge);           // 에지 강한 곳 → 보존
float chromaBoost = mix(1.0, 1.0 + kChroma, chromaDev);     // chroma 편차 큰 곳 → 제거 강화

float finalAtten = baseFactor * chromaBoost * edgeProtect;
```

**효과**:
- 에지 강도가 높은 곳 (주름, 눈코입 경계) → 감쇠 억제 → **주름 보존**
- 에지 약하고 chroma 편차 큰 곳 (색소침착) → 감쇠 강화 → **색소침착 제거**

**구현 비용**: 추가 GPU 패스 없음. 기존 composite 셰이더 내에서 `textureGather` 또는 인접 텍셀 샘플링으로 구현 가능 (Gemini).

---

## Q4. Sharpen 강도 자동 조절

### 만장일치: `skinQuality` 단순 비례는 비추천

**이유**: 피부 상태가 안 좋을 때(skinQuality 높을 때) sharpen을 올리면 잡티·노이즈가 오히려 더 도드라진다.

### 권장 기준: Smoothing 손실량에 비례

```
sharpen = clamp(s0 + k1 * detailLoss - k2 * noiseLevel, sMin, sMax)
```

| 파라미터 | 비례 방향 | 설명 |
|----------|:---------:|------|
| `detailLoss` (smoothing으로 잃은 고주파량) | **정비례** | 블러를 세게 할수록 sharpen 보상 필요 |
| `noiseLevel` (입력 노이즈 수준) | **반비례** | 노이즈가 높으면 sharpen 억제 |
| `skinQuality` | **보조 계수** (작은 범위) | 직접 구동 파라미터가 아닌 참조용 |

### 추가 세부 조절 (Gemini)

- **마스크 경계** (헤어라인, 얼굴 윤곽): sharpen 강하게 → 경계 선명도 유지
- **이마/볼 중앙** (smoothing 집중 영역): sharpen 억제 → 부드러움 유지

### 실용적 시작점

메트릭 측정 인프라가 부족한 경우, **고정 저강도(0.12~0.18)**로 시작 후 디바이스/조명별 캘리브레이션이 가장 안전하다 (Codex).

---

## Q5. Phase 2 진입 기준

### 진입 판단 2개 게이트

| 게이트 | 기준 | 측정 방법 |
|--------|------|-----------|
| **품질 게이트** | Phase 1 후에도 에지 halo/ghosting, 피부 볼륨감 상실, 색소침착-주름 분리 실패가 목표치 미달 | A/B 비교 테스트, 사용자 설문, SSIM/PSNR 지표 |
| **성능 게이트** | 타겟 중간급 기기에서 30fps 기준 GPU 여유 3~4ms 이상, 10분+ 열 스로틀 안정 | GPU 프로파일링, 서멀 모니터링 |

### 12-pass Bilateral 현실성 평가

**결론: 풀해상도 크로스플랫폼 30fps는 보수적으로 어려움**

| 조건 | 가능성 |
|------|--------|
| 풀해상도(1080p) + 전 기기 | 리스크 매우 큼 (메모리 대역폭 병목) |
| 반해상도 + FP16 + 패스 병합 | 상위 기기에서 가능성 있음 |
| 디바이스 티어별 fallback 적용 | 현실적 |

### 대안 기법 (두 AI 공통 추천)

| 대안 | 패스 수 | 장점 |
|------|:-------:|------|
| **Dual Filtering** (다운샘플링 활용 반복 Gaussian) | 3~4 pass | bilateral 유사 효과, 저비용 |
| **Recursive Bilateral Filter** | 3~4 pass | Yang et al. 변형, 에지 보존 |
| **3-Scale Wavelet** (권장) | ~10 pass | GPU 친화적, 텍스처 피라미드 활용 가능, 스케일별 독립 제어 |

**두 AI 모두 Wavelet 우선 추천**: 핵심 문제가 "스케일 분리 부족"이면 bilateral보다 wavelet이 효율적이고, 텍스처 피라미드를 활용하면 연산량을 획기적으로 줄일 수 있다.

---

## 종합 결론 및 액션 아이템

### 핵심 인사이트

> Phase 1만으로 상용 SDK 수준의 **80% 이상 품질 확보 가능**하다 (Gemini 평가).
> Phase 2가 필요한 경우 **Wavelet > Bilateral** 우선 검토한다.

### Phase 1 실행 계획

| 단계 | 작업 | 예상 난이도 | 추가 GPU 비용 |
|:---:|------|:----------:|:------------:|
| 1 | **Linear RGB 색공간 전환** — 입출력에 pow(2.2) 변환 추가 | 낮음 | ALU only |
| 2 | **Soft Light 합성 전환** — additive → soft light (overlay 대신) | 중간 | ALU only |
| 3 | **Edge-aware Attenuation** — gradient + chroma 신호 추가 | 중간 | ALU + 인접 텍셀 fetch |
| 4 | **톤커브 미드톤 리프트** — composite 셰이더 내 quadratic curve | 낮음 | ALU only |
| 5 | **Luminance Sharpen** — unsharp mask, 고정 저강도(0.15) 시작 | 중간 | +1~2 pass |

### 리뷰어별 차별화된 기여

| AI | 고유 기여 |
|----|-----------|
| **Codex** | 하이브리드 블렌딩 공식, 3신호 결합 감쇠 수식, detailLoss 기반 sharpen 공식 |
| **Gemini** | Soft Light 대안 제안, textureGather 구현 힌트, Dual Filtering/Recursive Bilateral 대안, 마스크 경계별 sharpen 차등 |
| **Claude (의장)** | Soft Light → 하이브리드 2단계 전략, 종합 실행 계획 정리 |
