# P5-W3-05 렌즈 렌더링 자연스러움 — **최종 결정 (R3+R4 합의 반영본)**

> **작성**: Claude Opus 4.7
> **시점**: 2026-04-23 R4 패치 검증 완료 후
> **성격**: Codex(gpt-5.4 xhigh), Gemini(gemini-3-flash), Claude **4라운드**(R1 독립 → R2 교차비판 → R3 Claude 종합 검증 → R4 실측/사용자 피드백 반영 패치 검증) 거친 합의본.
>
> **R4 추가 근거**: 실측 에셋 분석(`15_asset_analysis.md`) + 웹 교차검증(`16_product_crosscheck.md`) + 사용자 실물 관찰 피드백. 6개 Patch를 Codex/Gemini 둘 다 hard veto 없이 승인(`18/19_*_r4_patch.md`).
>
> 모든 쟁점 결론은 **3모델 중 최소 2모델 동의** 또는 **실기기 벤치 이관** 자격을 가진다. 단일 모델 주장만으로 채택된 것은 없다.

---

## 0. 한 줄 결론

W3-04 고정 조명·realSpec·avgIrisLum 하드코드 **제거** + 블렌드 3종(TintLinearV2/Multiply/ScreenLinear) **확정** + 환경 반사 가산 계층 **분리(renderMask hook 포함, 소스는 벤치)** + EyeRenderPacket **도입** + 텍스처 256×256 **표준화**. 5쟁점(I1/I2-잔여/I4/I5-up/I8)은 **실기기 벤치로 최종 확정**. Pupil cutout은 **P6-W1 조건부 트랙**으로 이관 (B2 "중앙 공동 체감" 지표 2/3 이상 시 착수).

---

## 1. 즉시 확정 (R3 합의 — 실기기 검증 없이 채택)

R3에서 Codex·Gemini·Claude가 모두 동의한 항목만 포함. `99_claude_synthesis.md`에서 Codex/Gemini R3 지적으로 **수정된 항목**은 표시.

### 1.1 제거할 것

| ID | 제거 대상 | 파일:라인 | 합의 출처 |
|----|----------|----------|----------|
| D1 | 분석적 노멀 + 고정 조명 `vec3(0.3, 0.4, 1.0)` 블록 | `shader_sources.cpp:1008-1032` | 세 모델 R2 합의, R3 재확인 |
| D2 | `LIMBAL_ENABLED = false` 하드코드 전역 비활성 | `shader_sources.cpp:999-1006` | 세 모델 R2 합의 |
| D3 | `realSpec = smoothstep(0.7, 0.95, lum)` + `mix(result, baseL, realSpec)` | `shader_sources.cpp:865-874` (LTL 내부) | **조건부 폐기**: I1 환경 반사 계층 도입 성공(B2) 확정 후 제거. I1 실패 시 재평가 — Gemini/Claude R3 지적. ✅ **P6-W2 코드 완전 삭제** (커밋 b70490b, archive: `P6-W2_brainstorm/realSpec_archive.md`). 문서상 "확정 폐기" 마킹은 W4 B2 결과 후. |
| D4 | `uAvgIrisLum = 0.35` 하드코드 기본값 | `gpu_lens_renderer.cpp:811-813` 근처 | 세 모델 R2 합의 |
| D5 | "3D Light" 토글 UI + `uHighlightEnabled` uniform | demo + shader | D1 제거 시 자동 무의미 |
| D6 | 블렌드 중 `Overlay`, `LuminanceTint(nonlinear)`, `SoftLight` 3종 | `shader_sources.cpp:979-997` 분기 | ⚠️ **`Normal` 제거 철회** — Codex R3 "B1 벤치 전 제거는 결론 선반영" 지적. Normal은 B1 벤치 전까지 유지. ✅ **P6-W2 적용** (S1 9aee86d로 함수+분기 제거, W2 b70490b로 fallback default를 TintLinearV2로 통일, 25ebf99로 invalid ID 1회 경고). |

### 1.2 추가/변경할 것

| ID | 변경 | 합의 출처 / R3 수정 내역 |
|----|------|----------------------|
| C1 | **블렌드 3종 확정**: `TintLinearV2`(기본값), `Multiply`, `ScreenLinear`. 4번째 슬롯은 §2 B1 벤치 결과로 확정 | ⚠️ **R3 수정**: Codex 원래 R2 입장(3종 확정 + 4번째 슬롯 벤치)으로 되돌림. 99_claude_synthesis의 "4종 확정" 왜곡 수정. ✅ **P6-W2 적용** (커밋 b70490b — 셰이더 함수 + 분기 등록 완료). |
| C2 | `TintLinearV2` = 기존 LuminanceTintLinear에서 `realSpec` 완전 제거한 형태. 수식: `baseL=base*base; lensL=lens*lens; lum=dot(baseL,w); out=sqrt(mix(baseL, lensL*lum*scale, a))`. `scale = clamp(K / uAvgIrisLum, 0.8, ub)`에서 **K(비례 상수)와 ub(scale clamp upper)는 실기기 시각 튜닝값**. W2 적용 시 K=0.85, ub=7.0. | 세 모델 R2/R3 합의. ✅ **P6-W2 적용** (커밋 b70490b — squaring 버그 함께 수정, uAvgIrisLum 이미 linear 가정). 실기기 회귀 fix(K=0.5→0.85, ub=5.0→7.0)는 별도 fix 커밋. **추가 강도는 알고리즘 본질 한계(휘도 보존 구조)로 W5 B1 벤치 재검토 대상**. |
| C3 | `ScreenLinear` 신규 — 수식: `out = sqrt(mix(baseL, 1-(1-baseL)*(1-lensL), a))` | 세 모델 R2/R3 합의. ✅ **P6-W2 적용** (커밋 b70490b — sRGB blendScreen 제거, 옵션 A). |
| C4 | `ColorReplaceLinear` **벤치 대상(B1)** — 수식: `detail = clamp(pow(lum/avgLum, 0.7), 0.75, 1.25); out = sqrt(mix(baseL, lensL*detail, a))` | Codex R2 제안, B1 결과 후 채택 결정. ✅ **P6-W2 적용** (커밋 b70490b — ID 7 정식 분기 활성, §5.7 옵션 B. W5 B1 결과로 채택/제거). |
| C5 | 환경 반사 가산 계층 분리. 기본: `float renderMask = finalAlpha; blended += reflection * fresnel * renderMask;`. **반사 소스는 B2 벤치로 결정**. **renderMask hook**: P6-W1 활성화 시 `#ifdef ENABLE_PUPIL_MATERIAL_RESTORE` 분기로 `renderMask = max(finalAlpha, smoothstep(iris_radius * 1.2, 0.0, dist))`로 확장 가능. 기본 동작은 기존 finalAlpha와 동일, 추가 런타임 비용 0. ⚠️ 이 smoothstep 수식은 **예시, 최종 구현 확정 아님** (Codex R4 단서) | Codex R1/R3 + R4 Patch 4 |
| C6 | 림발 기본 ON + SKU 메타데이터 플래그. 플래그는 **내부 material 모델 필드**(공개 `LensConfig`가 아닌 내부 표현) | ⚠️ **R3 수정**: Codex R3 "공개 API 변경 가능성" 지적 반영. 공개 API는 변경하지 않음 |
| C7 | **블링크 alpha ramp — 시간 범위만 확정, 계수는 실측 튜닝**. down 50~80ms, up은 **B5 벤치로 확정**(아래 §2) | ⚠️ **R3 수정**: Gemini R3 "up 100~120ms 동의한 적 없음" + Codex R3 "α=0.15 계수가 목표 ms와 불일치" 둘 다 반영. up 시간 쟁점 자체를 벤치로 이관 |
| C8 | `EyeRenderPacket` 구조체 도입 (내부 어댑터 레이어) | 세 모델 R2/R3 합의 |
| C9 | **avg_iris_luma 실측 — masked ROI 평균**. 수식: `avg = sum(dot(rgb,w)*mask)/sum(mask)`, `mask = (r < 0.65) AND eyelidMask`. 측정 실패 시 이전값 hold, 3프레임 이상 실패 시 중립 상수 fallback | ⚠️ **R3 수정**: Codex R3 "1샘플은 pupil 중심 검은 동공 읽음 → 정규화 망침" 지적 반영. 중심 샘플 폐기, ROI 평균으로 교체 |
| C10 | 홍채 디테일 재주입 — `detail = clamp(baseLum / blur3x3(baseLum), 0.85, 1.15)`를 **iris inner(r<0.65)에만**, **spec/reflection 계층 계산 전에 합성**, **spec/reflection 영역 제외**. 저조도 gate 임계값은 **B9 벤치로 확정** | ⚠️ **R3 수정**: Codex R3 "spec/reflection 제외 빠짐" + Gemini R3 "gate 0.15 너무 낮을 수 있음" 반영. gate 임계값 확정 벤치 추가 |
| C11 | W2-W3 경계: pupil_center·occlusion·(optional) gaze는 W2 refiner 출력, W3는 EyeRenderPacket으로 수신만. **occlusion은 `visibility + aperture_mask`로 흡수**(별도 필드 X) | ⚠️ **R3 수정**: Codex R3 "스키마 불일치" 지적 반영. occlusion이 별도 필드가 아님을 명시 |

### 1.3 EyeRenderPacket 최종 스키마 (R3 확정)

```cpp
// 내부 구조체. 공개 C API는 변경 없음.
struct EyeRenderPacket {
    // 필수
    glm::vec2 iris_center_norm;
    float     iris_radius_norm;
    // 비대칭 타원 (aperture mask)
    glm::vec2 ellipse_center;
    glm::vec3 ellipse_radii;          // (rxInner, rxOuter, ry)
    float     ellipse_rotation;
    float     eye_top, eye_bottom;    // Y-slab fallback
    float     visibility;             // 0.0~1.0 (occlusion은 이 값 + aperture mask로 흡수)
    uint64_t  timestamp_ms;

    // 선택 (없으면 기능 자동 off)
    std::optional<glm::vec2> pupil_center_norm;       // parallax + 동공 정렬용
    std::optional<glm::vec2> head_pose_yaw_roll;      // env rotation용
    std::optional<glm::vec3> reflection_dir;          // head_pose 대체 가능
    std::optional<float>     avg_iris_luma;           // blend 정규화용 (없으면 C9 self-measure)
    std::optional<float>     eye_depth_mm;            // 스케일 보정용
    std::optional<float>     render_confidence;       // alpha hysteresis용
};
```

**금지**: raw `face_mesh`, detector-specific landmark 인덱스.
**Optional 처리**: 없으면 기능 off. priors로 채우지 않는다 (C9 avg_iris_luma는 self-measure 경로만 예외).

---

### 1.4 텍스처 크기 공식 (R4 Patch 5)

**표준 해상도**: **256×256 RGBA PNG**
- 근거: iris 반경 40~150px × 1.5~2배 영역 = 120~310px. 256 = 1.28× 다운샘플로 자연 (`15_asset_analysis.md §4` 참조)
- GPU 텍스처 메모리: 20 SKU × 262KB ≈ 5.2MB (mipmap 포함 ~7MB)

**공급 방식**:
- 원본(≥1000×1000)은 업체 제공, **사용자 측에서 256×256으로 리사이징하여 SDK에 공급**
- 원본은 아카이빙 용도로만 유지 (4MB/장, 20 SKU = 80MB로 과도)

**HIGH tier 옵션**: 512×512 자동 선택 (고사양 기기)
- 파일 네이밍: `{SKU}_256.png` (표준), `{SKU}_512.png` (HIGH)

---

## 2. 실기기 벤치로 확정 (R3 합의 + R4 메트릭 톤 변경)

**R3 수정 후 5개 벤치**. 기존 4개(B1/B2/B4/B8)에서 **B5(블링크 up ramp)와 B9(저조도 gate) 추가**.

### ⚠️ R4 Patch 1 — 벤치 메트릭 톤 (공통 원칙)

**판정은 정성적 체감 기반 블라인드 비교로 진행**. 숫자 카운트보다 **다양한 환경·조명·동작 커버리지 우선**. 평가자 3명이 각 클립에 대해 (a) 자연스러움 선호도, (b) 언급할 만한 위화감 항목 자유 기술. 다수 의견 기반 판정.

- 사용자 원칙: "수치로 정량적으로 비교하기보단 실제 여러 환경에서 렌더링되는 것을 눈으로 확인하고 체감해야 알 수 있는 부분"
- 아래 각 벤치(B1/B2/B4/B5/B8/B9)의 "실패 카운트", "과반수", "17/30+" 같은 숫자 표현은 **가이드 기준**이며, **다수 의견이 명확하면 더 적은 표본으로도 판정 가능**

### B1 — 블렌드 4번째 슬롯: Normal vs ColorReplaceLinear

- **매트릭스 확장**: Codex R3 지적 반영해 SKU에 **화이트/그래픽 렌즈 추가**. 총 **5 SKU** × 3 홍채 톤 × 2 모드 = **30 클립**
  - 다크브라운 자연 / 헤이즐 자연 / 밝은 그레이·블루 / 불투명 서클 / **화이트·그래픽**(신규)
- **판정**: 3명 블라인드 A/B, 과반수
- **결론 시나리오**:
  - CRL 17/30+ 승리 → **4종 확정** (TintLinearV2 + Multiply + ScreenLinear + ColorReplaceLinear)
  - CRL이 특정 SKU(화이트/그래픽)에서만 승리 → **조건부 채택**(SKU metadata `prefers_crl` 플래그로 자동 선택)
  - Normal 17/30+ 승리 → **3종 유지** (TintLinearV2 + Multiply + ScreenLinear) + Normal
- **소요**: 3~4시간 (SKU 1개 추가 반영)

### B2 — 환경 반사 소스: env map only vs Periphery camera vs OFF

- **프로토타입 3종**:
  1. **OFF** (baseline)
  2. **env-map-only**: 256×128 LDR/RGBM 에셋 + pose 없으면 정적
  3. **periphery-camera**: 화면 상단 1/3 + 좌우 가장자리 샘플링. **`face_region_mask` 구현은 미확정 — 벤치 프로토타입 설명 시 "대략적 중앙 얼굴 영역 제외, 구체 구현은 프로토타입 단계에서 정의"로 표기**
- ⚠️ **R3 수정 (Gemini R3)**: Claude의 "hybrid(env + 상단 crop)"는 프로토타입 복잡도만 늘리므로 **1차 대결에서 제외**. env-only vs periphery-only 선행, 두 프로토타입 모두 OFF 대비 유의미 개선 시 2차로 hybrid 검토.
- **매트릭스**: 4 환경(실내형광/창가측광/야간실내/실외낮) × 2 동작(정면미세/좌우head turn) × 3 프로토타입 = **24 클립**
- **판정 메트릭** (체감 Y/N 체크 + 다수 의견):
  - `가운데 붙은 반짝이` 체감
  - `얼굴 재귀 반사처럼 보임` (Codex 우려)
  - `좌우 눈 불일치` 체감
  - `환경과 무관해 보임` (Gemini 우려)
  - `자연스러움 블라인드 선호도` (양성, 1순위)
  - **`중앙 공동(cavity) 체감`** (R4 Patch 2 신규) — 밝은 렌즈 × 짙은 홍채 조합(엔비_퍼퓸글로우, 클라셋_런웨이그레이 × 아시아 짙은 홍채) 중점 관찰. Y/N. **P6-W1 조건부 트랙 착수 판정 근거 (§4 참조)**
- **결론 시나리오**:
  - env-only 명확 우세 → Codex 안 채택, D3 realSpec 완전 폐기 확정
  - periphery 명확 우세 → Gemini 안 채택, D3 realSpec 완전 폐기 확정
  - 둘 다 OFF 대비 유의미 개선 → 2차 hybrid 검토
  - 둘 다 OFF 대비 차이 미미 → **환경 반사 Phase 6 이월, D3 realSpec 조건부 유지** (C7과 연동)
- **소요**: 5~6시간 (3프로토타입 구현 2.5h + 캡처 1.5h + 평가 1h + 반영 0.5h)

### B4 — 림발 자동 감지 정확도 (수정: fallback only 검증)

- ⚠️ **R3 수정 (Codex/Gemini R3)**: "메타데이터 only 채택은 벤치 없이 확정". 자동 감지는 **fallback only**로 제한. B4는 **자동 감지를 fallback으로 쓸지 말지**만 판정.
- **프로토타입**: 텍스처 로딩 시 CPU 1회 측정. `edge_lum = avg(r∈[0.85, 1.0])`, `center_lum = avg(r<0.3)`, `edge/center < 0.75` → `baked_limbal_detected`
- **테스트**: 림발 내장 5개 + 없는 5개 = **10 SKU**
- **판정 (기준 강화 — Codex R3 반영)**:
  - 정확도 10/10 → **fallback으로 채택** (드문 메타 누락 시 사용)
  - 9/10 이하 → **자동 감지 드롭**, 메타 only. 플래그 누락 텍스처는 경고 로그
- **소요**: 1시간

### B5 — 블링크 up ramp 시간 (신규, I5 잔여)

- ⚠️ **R3 신규**: Gemini R3 "up 100~120ms 부동의", Codex R3 "EMA 계수 검증 안 됨"
- **프로토타입 3안**: down 60ms 고정. up = **60ms / 80ms / 120ms** 3가지
- **매트릭스**: 5 사용자 × 블링크 10회 × 3 up 시간 = **150 이벤트**
- **판정 메트릭**:
  - `팝 느낌`(즉시 off 수준) 낮을수록 좋음
  - `늦게 나타나는 지연`(Gemini 우려) 낮을수록 좋음
- **결론 시나리오**:
  - 60ms 우세 → Gemini 안 채택 (대칭 60/60)
  - 80ms 우세 → 중간 타협
  - 120ms 우세 → Codex 안 채택
- **소요**: 1시간 (블링크 녹화 30분 + 평가 30분)

### B8 — sclera color veto: Codex veto vs Gemini luma-only

- **매트릭스**: 2 SKU(밝은 그레이/다크브라운) × 3 조명(형광/측광/저조도) × 2 방식 = **12 클립**
- **판정**:
  - "흰자 렌즈 번짐" 적은 방식 선택
  - "iris 외곽 어두운 무늬 잘림" 없는 방식 선호
- **결론 시나리오**:
  - 차이 미미 → 더 단순한 Gemini luma-only 채택
  - color-veto 유의미 우위 → Codex 채택 (수식: `geom = smoothstep(0.75, 1.0, dist); veto = smoothstep(0.18, 0.32, sat) * (1.0 - smoothstep(0.45, 0.65, lum))`)
- **소요**: 1시간

### B9 — 홍채 디테일 재주입 저조도 gate 임계값 (신규, C10 잔여)

- ⚠️ **R3 신규**: Codex R1 "확신 없다" + Gemini R3 "0.15 너무 낮을 수 있음"
- **매트릭스**: 3 조도(밝음/보통/저조도) × 3 gate 임계값(0.10 / 0.15 / 0.25) = **9 클립**
- **판정**: 저조도에서 노이즈 증폭 없는 임계값 선택
- **소요**: 1시간

---

## 3. 구현 순서 (R3 반영, S1~S5)

### S1 — 제거 (1~2시간, 단일 커밋)

⚠️ **R3 수정**: `Normal` 제거 철회. D6 범위 축소.

제거 대상:
- D1 고정 조명 + 분석 노멀 라이팅 블록
- D2 `LIMBAL_ENABLED = false` 하드코드
- D3 realSpec 2줄 (조건부 — B2 결과 후 확정. S1에서는 일단 **주석 처리 + 플래그**로 비활성)
- D4 `uAvgIrisLum = 0.35` 하드코드
- D5 "3D Light" 토글 UI
- D6 블렌드 중 **Overlay, LuminanceTint(non-linear), SoftLight 3종만 제거**. Normal은 B1 벤치 전까지 유지.

**커밋 메시지**: `refactor(gpu-lens): P5-W3-05 S1 — 고정조명/realSpec/avgIrisLum 하드코드/블렌드 3종 제거`

### S2 — EyeRenderPacket + 벤치 독립 구현 (4~5시간)

⚠️ **R3 수정 확인 사항**:
- C9 구현을 **1샘플 아닌 masked ROI 평균**으로 (Codex R3 필수 수정)
- C7 ramp 구현 시 시간 범위 구조만 넣고 **EMA 계수는 상수 정의로 유지하되 B5 결과에 따라 재튜닝**
- C10 수식에 **spec/reflection 제외** 조건 추가

작업:
- EyeRenderPacket 구조체 도입(C8)
- 블렌드 3종 재편(C1, C2, C3) — TintLinearV2/Multiply/ScreenLinear
- 환경 반사 가산 계층 **스캐폴드**(C5) — 소스는 B2 후
- 림발 기본 ON + 내부 메타 플래그(C6)
- 블링크 ramp 시간 범위 구조(C7) — 계수는 임시
- avg_iris_luma masked ROI 평균 측정(C9)
- 홍채 디테일 재주입 + spec/reflection 제외(C10) — gate 임계값 B9 후
- W2-W3 경계 명확화(C11) — occlusion 스키마 통합

### S3 — 벤치 실행 (6~8시간)

B1·B2·B4·B5·B8·B9 병렬/순차. 프로토타입 필요한 벤치(B1, B2)부터.

### S4 — 벤치 결과 반영 (2~3시간)

⚠️ **R3 수정**: 99_claude_synthesis.md §3 S4의 "C10(sclera)" 오기 수정 — C10은 detail reinjection, sclera는 B8.

반영 대상:
- B1 → C4(ColorReplaceLinear) 채택/기각/조건부
- B2 → C5(환경 반사 소스) 확정, D3 realSpec 조건부 → 확정 폐기 or 조건부 유지
- B4 → C6(림발 자동감지 fallback) 채택/드롭
- B5 → C7(블링크 up 시간) 확정
- B8 → C7 sclera 관련 분기 (실제로는 별도 수식) — Codex color-veto or Gemini luma-only
- B9 → C10(저조도 gate 임계값) 확정

### S5 — 통합 테스트 (3~4시간)

HIGH/MID/LOW tier 실기기 각 1시간 + 양안 동작 검증 + 블링크 검증 + 환경 반응 검증.

### 예상 총 소요

⚠️ **R3 수정**: Codex R3 "14~20시간은 낙관적" 지적 반영. **실측 기반 상향**: **20~28시간 = 3~4일**. 특히 B2 3프로토타입 구현과 EyeRenderPacket 리팩터가 같은 기간 내 완료된다는 보장 없음. 코드 영향 범위 확인 후 재조정 여지 명시.

---

## 4. Phase 6 이월 (R3 합의)

| 항목 | 근거 |
|------|------|
| 3D Face Geometry 기반 렌즈 렌더링 | 결합도 증가 |
| HDR IBL (Perfect `eye_ibl.hdr`급) | LDR/RGBM env map으로 충분 검증 전까지 불필요 |
| Full PBR (Cook-Torrance) | 렌즈 도메인에 과잉 |
| Neural rendering / harmonization | SDK 크기·지연 예산 파괴 |
| Corneal refraction 실시간 시뮬 | 육안 체감 미미 |
| 속눈썹 전용 세그멘테이션 모델 | 눈꺼풀 타원 feather로 충분 |
| **Pupil cutout 해결 → P6-W1 조건부 트랙** (R4 Patch 3) | **해결 방향**: Option E "렌즈 재질 반투명 복원" — 원본 홍채 건드리지 않고, 렌즈 자체의 재질감(반투명+광택)을 동공 영역에도 옅게 확장. Option A(baseLum 하한)은 홍채 디테일 훼손 리스크로 기각.<br/>**사용자 실물 관찰**: 실제 렌즈 중앙 감도는 재질 특성이며 눈물막과 결합해 자연 투명해짐. W3-05 환경 반사 인프라(C5 renderMask hook)로 **자연 커버 가능성 있음** — 의도적 처리 없이도 해결될 여지.<br/>**착수 조건 (Hard)**: B2 벤치에서 평가자 3명 중 2명 이상 "중앙 공동 체감"이면 P6-W1 착수. 모두 "체감 없음"이면 P6-W1 폐기. 1명만 체감 시 사용자 판단.<br/>**인프라 준비**: C5 renderMask hook (§1.2)이 기본 구조 담당. W3-05는 기존 finalAlpha 동작 유지 |
| Head-pose 기반 env 회전 실제 활용 | W2 refiner 출력 이후 |

---

## 5. R3에서 해소된 Claude 1인 종합의 편향 요약

`99_claude_synthesis.md`에서 다음 편향이 R3로 교정됨:

| # | Claude 종합의 문제 | R3 지적자 | 교정 내용 |
|---|------------------|----------|----------|
| 1 | C1 "4종 확정" 과장 — Codex R2 "3종 + 4번째 벤치"를 무시 | Codex | 3종 확정으로 수정, C4 벤치 대상으로 재분류 |
| 2 | D6 Normal 제거를 B1 벤치 전에 S1에 포함 → 결론 선반영 | Codex | D6에서 Normal 제거, B1 후 결정 |
| 3 | C9 textureLod 1샘플 → pupil 중심 검은 동공 읽음 | Codex | masked ROI 평균으로 교체 |
| 4 | C7 EMA 계수(α=0.15)가 목표 ms와 수학적 불일치 + Gemini up 시간 부동의 | Codex + Gemini | up 시간 B5 벤치 이관, 계수는 튜닝 대상 |
| 5 | C10 spec/reflection 제외 조건 빠짐 + gate 0.15 미확정 | Codex + Gemini | 제외 조건 추가, gate B9 벤치 이관 |
| 6 | C11 occlusion을 언급했지만 EyeRenderPacket 스키마에 없음 | Codex | occlusion = visibility + aperture_mask로 흡수 명시 |
| 7 | D3 realSpec을 "완전 수렴"으로 분류 — I1 벤치 이관 상태에서 위험 | Gemini + Claude 자기 | D3를 **조건부 폐기**로 재분류 |
| 8 | I1 Periphery를 "hybrid 옵션"으로 축소 흡수 — Gemini 고유 안 약화 | Gemini + Claude 자기 | B2에서 env-only vs periphery-only 선행 대결, hybrid는 2차 |
| 9 | S4 "C10(sclera)" 오기 — C10은 detail, sclera는 B8 | Codex | S4 표기 수정 |
| 10 | B1 SKU에 화이트/그래픽 렌즈 누락 — CRL 존재 이유가 이 셀인데 빠짐 | Codex | 5번째 SKU로 추가 |
| 11 | C6 `has_baked_limbal`이 공개 API인지 내부인지 불분명 | Codex | 내부 material 모델 필드로 명시, 공개 API 변경 없음 |
| 12 | 소요 시간 14~20h 낙관적 | Codex | 20~28h로 상향, 코드 영향 파악 후 재조정 여지 명시 |

---

## 6. R3에서 여전히 미결/조건부인 것

| # | 항목 | 상태 |
|---|------|------|
| 1 | D3 realSpec 완전 폐기 | **조건부** — B2 환경 반사 벤치 성공 시 확정 폐기. 실패 시 유지 재평가 |
| 2 | C4 ColorReplaceLinear 채택 | **B1 벤치 대상** |
| 3 | C5 환경 반사 소스 (env map / periphery / hybrid) | **B2 벤치 대상** |
| 4 | C6 림발 자동 감지 fallback | **B4 벤치 대상** |
| 5 | C7 블링크 up ramp 시간 | **B5 벤치 대상** |
| 6 | C8 sclera veto 방식 (color vs luma-only) | **B8 벤치 대상** |
| 7 | C10 저조도 gate 임계값 | **B9 벤치 대상** |
| 8 | Pupil cutout → **P6-W1 조건부 트랙 이관** (R4 Patch 6) | 방향: Option E (렌즈 재질 반투명 복원). 착수 조건: B2 "중앙 공동 체감" 지표 2/3 이상. 결정 근거: 사용자 실측 관찰 + Codex R3 "완벽 아니면 하지 말자" 입장 부합. 상세는 §4 참조 |

---

## 7. 부록 — 전체 응답 파일

| 번호 | 파일 | 성격 |
|------|------|------|
| 01 | `01_brief.md` | R1 공통 브리프 |
| 02 | `02_claude_response.md` | Claude R1 |
| 03 | `03_gemini_response.md` | Gemini R1 |
| 04 | `04_codex_response.md` | Codex R1 (가장 상세, 파일:라인 포함) |
| 05 | `05_r1_summary.md` | R1 결론 요약 (Claude 1인 종합) |
| 06 | `06_r2_issues.md` | R2 쟁점 목록 (Claude 작성) |
| 07 | `07_claude_r2.md` | Claude R2 |
| 08 | `08_codex_r2.md` | Codex R2 |
| 09 | `09_gemini_r2.md` | Gemini R2 |
| 10 | `10_r2_summary.md` | R2 결론 요약 (Claude 1인 종합) |
| 11 | `11_r3_review_request.md` | R3 검토 요청서 |
| 12 | `12_claude_r3.md` | Claude R3 (자기 편향 자기비판) |
| 13 | `13_codex_r3.md` | Codex R3 (가장 엄격한 지적) |
| 14 | `14_gemini_r3.md` | Gemini R3 (실시간성 재강조) |
| 15 | `15_asset_analysis.md` | 렌즈 에셋 20개 실측 분석 (R4 근거) |
| 16 | `16_product_crosscheck.md` | 웹 교차검증 (보정 리스크 명시) |
| 17 | `17_patch_review_request.md` | R4 6개 Patch 검증 요청서 |
| 18 | `18_codex_r4_patch.md` | Codex R4 (5 찬성 + 1 중립, hard veto 없음) |
| 19 | `19_gemini_r4_patch.md` | Gemini R4 (6 찬성, hard veto 없음) |
| — | `99_claude_synthesis.md` | Claude 1인 종합 초안 (역사 자료, R3 전 시점) |
| **99** | `99_final_decision.md` | **이 파일. R3+R4 합의 반영 최종본** |

---

## 8. 실행 통계

- **브레인스토밍 라운드**: 4 (R1 독립 → R2 교차비판 → R3 Claude 종합 검증 → R4 실측/사용자 피드백 패치 검증)
- **R3 Claude 자기비판 지적**: 12개 중 **R3에서 9개 Codex/Gemini가 독립 확인** (자기비판 정확도 75%)
- **R3 신규 지적 (Claude가 자기비판에서 놓친 것)**: 3개 (C1 C4 C9 명시적 지적, C7 계수·C10 수식 세부)
- **R4 실측 교정**: 에셋 분석(15)과 웹 교차검증(16)으로 2개 지점 추가 교정 (B4 재설계, C6 해석 단서)
- **R4 Patch 6개 승인**: Codex/Gemini 모두 hard veto 없음. Claude 작성 Patch 전부 채택
- **최종 결정 성격**: 17개 확정 + 5개 벤치 이관 + 8개 Phase 6 이월 + 1개 조건부(D3) + 1개 조건부 트랙(Pupil cutout → P6-W1)
- **남은 벤치**: 6개(B1~B9 중 B1/B2/B4/B5/B8/B9) = 약 11~15시간
- **구현 착수 가능성**: S1 즉시 착수 가능. S2는 R3+R4 수정 사항 모두 반영 확인 후 시작.

---

## 9. R4로 해소된 사용자 판단 사항

R3 합의본에서 사용자 판단이 필요했던 "Pupil cutout" 쟁점은 R4에서 **P6-W1 조건부 트랙으로 합의 이관** 완료 (사용자 의사 + 세 모델 hard veto 없음).

**합의 내용 요약**:
- W3-05 S2에서 의도적 처리를 추가하지 않음
- 대신 C5 renderMask hook이 P6-W1 활성화를 위한 구조만 준비
- B2 벤치에서 "중앙 공동 체감" 지표로 P6-W1 착수 여부 판정
- 방향은 Gemini의 Option A(baseLum 하한)가 아닌 Option E(렌즈 재질 반투명 복원)
- 어색하면 그때 P6-W1 시작, 자연 커버되면 트랙 폐기

**사용자 판단 남은 항목**: 없음. 99_final_decision.md는 이제 **구현 착수 승인 대기 상태**.
