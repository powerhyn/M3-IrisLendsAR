# P4-W4-01: FreqSep 피부 잡티 보정 품질 개선

## 작업 개요

Agent Council 리뷰 (Codex + Gemini + Claude) 결과를 기반으로,
현재 FreqSep 파이프라인의 품질을 업계 레퍼런스 수준으로 개선한다.

- **원본 분석**: `docs/workPaper/FreqSep_Quality_Analysis_Report.md`
- **리뷰 결과**: `docs/workPaper/FreqSep_Council_Review.md`
- **목표**: 기존 5-pass 구조를 유지하면서 상용 SDK 80%+ 품질 확보 (Phase 1)

## 레퍼런스

| # | 이름 | 핵심 기법 |
|---|------|-----------|
| 4 | YUCIHighPassSkinSmoothing | 톤커브, Sharpen, G/B 마스크, Hard Light 부스트 |
| 7 | Vincent Dedun AR Beauty | O(1) Bilateral, CIELAB 마스크, Linear RGB, Overlay |
| 8 | Wavelet Decompose (Pat David) | 5+1 scale 독립 제어, 미세질감 완전 보존 |

## 작업 상태

- ⏳ 대기 (Phase 1 전체)
- 선행 작업: P4-W3-05 브랜치 머지 완료 후 착수

---

## Phase 1: 현재 파이프라인 개선 (5-pass → 6~7 pass)

적용 순서: Council 합의에 따라 `D → C → B → A` + 추가 항목 E.

### Step 1: Linear RGB 색공간 전환 ⏳

**상태**: ⏳ 대기
**난이도**: 낮음 | **추가 GPU 비용**: ALU only (패스 추가 없음)
**변경 파일**: `cpp/src/gpu/shader_sources.cpp`

**작업 내용**:
- Gaussian blur 셰이더 입력 시 sRGB → Linear 변환 추가
  ```glsl
  vec3 linear = pow(srgb, vec3(2.2));
  ```
- Composite 셰이더 출력 시 Linear → sRGB 변환 추가
  ```glsl
  vec3 srgb = pow(linear, vec3(1.0/2.2));
  ```
- 또는 Pass 1a 입력과 Pass 3 출력에만 적용 (중간 패스는 linear 유지)

**주의사항**:
- Linear 전환 후 기존 파라미터 (blur_radius, attenuation 등) 재튜닝 필요
- 이후 모든 작업이 Linear 기준이므로 **반드시 가장 먼저 적용**

**완료 기준**:
- [ ] Gaussian 셰이더에 sRGB↔Linear 변환 추가
- [ ] 기존 FreqSep 테스트 통과
- [ ] skinQuality 0.2/0.5/1.0에서 시각적 비교 (halo 감소 확인)

---

### Step 2: Soft Light 합성 전환 ⏳

**상태**: ⏳ 대기
**난이도**: 중간 | **추가 GPU 비용**: ALU only
**변경 파일**: `cpp/src/gpu/shader_sources.cpp` (FREQ_SEP_COMPOSITE_FRAGMENT)

**작업 내용**:
- 현재 additive 합성을 Soft Light (Pegtop variant)로 교체
  ```glsl
  // 현재 (additive):
  vec3 beauty = smoothLow + adjusted_high;

  // 변경 (Soft Light — Linear RGB 호환):
  float softLight(float base, float blend) {
      return (1.0 - 2.0 * blend) * base * base + 2.0 * blend * base;
  }
  // blend = 0.5 + adjusted_high (high freq를 0.5 중심으로 재매핑)
  ```
- Overlay 대신 Soft Light를 채택한 이유:
  - 모바일 8-bit 환경에서 banding 위험 감소 (Gemini 제안)
  - Linear RGB에서 조건 분기 없이 동작 (Pegtop variant)
  - 필요 시 하이브리드 (additive + soft light luma blend)로 확장 가능

**주의사항**:
- C+D 동시 적용 시 Overlay 공식이 Linear에서 달라지는 이슈 → Pegtop Soft Light로 회피
- 강도 보정이 필요할 수 있음 (pivot 재튜닝)

**완료 기준**:
- [ ] Composite 셰이더에 Soft Light 합성 구현
- [ ] additive 대비 A/B 비교 (톤/콘트라스트 자연스러움)
- [ ] 기존 FreqSep 테스트 통과 (수치 범위 조정 필요 시 업데이트)

---

### Step 3: Edge-aware Attenuation (3신호 결합) ⏳

**상태**: ⏳ 대기
**난이도**: 중간 | **추가 GPU 비용**: ALU + 인접 텍셀 fetch (패스 추가 없음)
**변경 파일**: `cpp/src/gpu/shader_sources.cpp` (FREQ_SEP_COMPOSITE_FRAGMENT)

**작업 내용**:
- 기존 단일 magnitude 감쇠를 3신호 결합으로 확장
  ```glsl
  // 1. 기존 magnitude (잡티 크기)
  float magnitude = dot(abs(high), vec3(0.299, 0.587, 0.114));

  // 2. 에지 강도 (2-tap 차분, Sobel 대신 경량)
  vec3 dx = textureOffset(uOriginal, vTexCoord, ivec2(1,0)).rgb
          - textureOffset(uOriginal, vTexCoord, ivec2(-1,0)).rgb;
  vec3 dy = textureOffset(uOriginal, vTexCoord, ivec2(0,1)).rgb
          - textureOffset(uOriginal, vTexCoord, ivec2(0,-1)).rgb;
  float edge = length(dx) + length(dy);

  // 3. Chroma 편차 (색소침착 감지)
  vec3 chromaDiff = orig - low;
  float chromaDev = length(chromaDiff.gb - vec2(chromaDiff.r));

  // 결합 감쇠
  float baseFactor = smoothstep(uAttenuationLow, uAttenuationHigh, magnitude);
  float edgeProtect = 1.0 - uEdgeWeight * clamp(edge, 0.0, 1.0);
  float chromaBoost = mix(1.0, 1.0 + uChromaWeight, chromaDev);
  float finalAtten = baseFactor * chromaBoost * edgeProtect;
  ```

**새 uniform 파라미터**:
- `uEdgeWeight` (기본 0.5): 에지 보호 강도. 높을수록 주름/경계 보존
- `uChromaWeight` (기본 0.3): 색소침착 제거 강화 정도

**효과**:
- 에지 강한 곳 (주름, 눈코입 경계) → 감쇠 억제 → 주름 보존
- 에지 약하고 chroma 편차 큰 곳 (색소침착) → 감쇠 강화 → 색소침착 제거

**주의사항**:
- Sobel 대신 2-tap 차분으로 시작 (모바일 저가형 성능 고려)
- `textureOffset`은 GLSL ES 3.0+에서 지원, 현재 셰이더가 310 es이므로 호환

**완료 기준**:
- [ ] 3신호 결합 감쇠 셰이더 구현
- [ ] `uEdgeWeight`, `uChromaWeight` uniform 연결 + `mapSkinQuality` 매핑 추가
- [ ] 색소침착/주름 테스트 이미지로 효과 검증

---

### Step 4: 톤커브 미드톤 리프트 ⏳

**상태**: ⏳ 대기
**난이도**: 낮음 | **추가 GPU 비용**: ALU only
**변경 파일**: `cpp/src/gpu/shader_sources.cpp` (FREQ_SEP_COMPOSITE_FRAGMENT)

**작업 내용**:
- Composite 출력에 간단한 quadratic 미드톤 리프트 적용
  ```glsl
  // Quadratic mid-tone lift: 미드톤(0.5 부근)을 살짝 올림
  // intensity는 skinQuality에 비례하여 0.0~0.15 범위
  float toneCurve(float x, float intensity) {
      return x + intensity * x * (1.0 - x);
  }
  // beauty 결과에 적용 (마스크 블렌딩 전)
  beauty.r = toneCurve(beauty.r, uToneLift);
  beauty.g = toneCurve(beauty.g, uToneLift);
  beauty.b = toneCurve(beauty.b, uToneLift);
  ```

**새 uniform 파라미터**:
- `uToneLift` (기본 0.10): 미드톤 리프트 강도
  - skinQuality 0.2 → 0.04, 0.5 → 0.08, 1.0 → 0.15

**완료 기준**:
- [ ] Composite 셰이더에 toneCurve 함수 추가
- [ ] `uToneLift` uniform 연결 + `mapSkinQuality` 매핑
- [ ] 피부 톤 균일감 시각적 확인

---

### Step 5: Luminance Sharpen 패스 추가 ⏳

**상태**: ⏳ 대기
**난이도**: 중간 | **추가 GPU 비용**: +1~2 GPU 패스
**변경 파일**:
- `cpp/src/gpu/shader_sources.cpp` — 새 셰이더 추가
- `cpp/src/gpu/gpu_beauty_backend.cpp` — 패스 추가
- `cpp/include/iris_sdk/gpu/gpu_beauty_backend.h` — 셰이더 프로그램/uniform 멤버

**작업 내용**:
- Composite 출력에 Unsharp Mask 기반 Luminance Sharpen 적용
  ```glsl
  // Sharpen Fragment Shader
  uniform sampler2D uInput;
  uniform float uSharpenAmount;  // 고정 0.12~0.18로 시작

  void main() {
      vec3 center = texture(uInput, vTexCoord).rgb;

      // 3x3 box blur (또는 5-tap cross)
      vec3 blur = (
          textureOffset(uInput, vTexCoord, ivec2(-1, 0)).rgb +
          textureOffset(uInput, vTexCoord, ivec2( 1, 0)).rgb +
          textureOffset(uInput, vTexCoord, ivec2( 0,-1)).rgb +
          textureOffset(uInput, vTexCoord, ivec2( 0, 1)).rgb
      ) * 0.25;

      // Luminance-only sharpen (색상 왜곡 방지)
      float lumCenter = dot(center, vec3(0.299, 0.587, 0.114));
      float lumBlur = dot(blur, vec3(0.299, 0.587, 0.114));
      float sharpen = (lumCenter - lumBlur) * uSharpenAmount;

      vec3 result = center + vec3(sharpen);
      fragColor = vec4(clamp(result, 0.0, 1.0), 1.0);
  }
  ```

**파라미터 전략** (Council 합의):
- `skinQuality` 단순 비례 **아님**
- 고정 저강도 (0.15)로 시작
- 향후 `detailLoss` 기반 자동 조절로 확장 가능

**완료 기준**:
- [ ] Sharpen 셰이더 작성 + 프로그램 빌드
- [ ] FreqSep 파이프라인 말미에 Sharpen 패스 연결
- [ ] 셰이더 on/off 비교 (선명도 개선 + 노이즈 증폭 없음 확인)
- [ ] 기존 테스트 통과

---

## Phase 2: 구조 변경 (조건부 진입)

### 진입 기준 (두 게이트 모두 충족 시)

| 게이트 | 기준 |
|--------|------|
| 품질 게이트 | Phase 1 후에도 에지 halo, 피부 볼륨감 상실, 색소침착-주름 분리 실패가 목표치 미달 |
| 성능 게이트 | 타겟 중간급 기기에서 30fps 기준 GPU 여유 3~4ms 이상 |

### Phase 2 후보: 3-Scale Wavelet Decompose (우선 추천)

- Scale 1 (r=2): 미세 질감 → 보존
- Scale 2 (r=8): 중간 결함 → 강한 감쇄
- Scale 3 (r=24): 큰 결함 → 약한 감쇄
- Residual: 톤/색상 → 선택적 균일화
- 예상 비용: ~10 GPU 패스

### Phase 2 대안: O(1) Bilateral Filter

- Yang et al. (CVPR 2009) 기반
- K=5 슬랩, sigma_spatial=3, sigma_range=0.1
- 예상 비용: ~12 GPU 패스, 텍스처 11개
- 풀해상도 크로스플랫폼 30fps는 보수적으로 어려움 → 반해상도 + FP16 + 디바이스 티어별 fallback 필수

---

## 변경 이력

| 일자 | 내용 |
|------|------|
| 2026-03-08 | 작업 문서 초안 작성, Phase 1 Step 1~5 정의 |

## 관련 문서

- `docs/workPaper/FreqSep_Quality_Analysis_Report.md` — 품질 분석 리포트
- `docs/workPaper/FreqSep_Council_Review.md` — Agent Council 리뷰 결과
- `docs/workPaper/P4-W3-06_quality_tools_bugfix.md` — Quality Tools 버그픽스
