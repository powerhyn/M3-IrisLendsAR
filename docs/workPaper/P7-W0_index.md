# P7-W0: Phase 7 인덱스 — Phase 6 후속 안정성 + 흰자 빛남 개선

> **상태**: 인사이트 작성 완료 (R1 브레인스토밍 종합, 2026-06-04). 각 W 본문 작성 대기.
> **선행**: Phase 6 develop 머지 완료 (commit `64ba08b`, 2026-06-01)
> **후속**: Phase 8 — 뷰티 (턱 깎기, 얼굴 사이즈 축소) **예약**, P7 범위 미포함
> **브레인스토밍 원문**: `docs/workPaper/P7-W0_brainstorm/{codex,gemini,claude}_p7w0.md` + `synthesis.md`
> **Stage 1 입력**: deep-research `wf_d2eb976b-38b` (107 agents, 25 sources)

---

## 1. 인사이트 ⭐

### 1.1 Phase 7의 정체성

Phase 6 종료 후 **잔여 안정성 + 흰자 빛남 개선 + 데모 cleanup** 단계. 큰 새 기능 도입은 P8(뷰티)에 배치. 본 Phase는:
- **W6에서 발견된 spec 위반 즉시 차단** (0x501)
- **avg_iris_luma 실측 source 연결** (Phase 6 미완 후속)
- **흰자 빛남 1단계 개선** (TintLinearV2 + luma attenuation, 본질 해결책 도입은 Spike 후 판단)
- **데모 UI 정리 + deprecated 제거**
- **MID/LOW tier 회귀 확장**

### 1.2 외부 동향 인사이트 (deep-research 통과 4개)

1. **0x501 = GLSL ES 3.0 spec §8.9 실제 위반** (3-0 vote 7회 통과) — 즉시 textureLod 명시 LOD로 교체.
2. **EXTERNAL_OES → 2D FBO 변환 강제** — 카메라 텍스처에서 통계 직접 측정 불가, 기존 OES→2D 패스 재사용 + ROI 다운샘플 표준.
3. **흰자 빛남 본질 해결책 = Oklab + Laplacian** — 학술 검증되었지만 GPU 예산 미검증 → **단계적 접근**, 1단계는 lightweight luma attenuation.
4. **Phase 8 substrate** — MediaPipe Face Landmarker 478 landmarks + Snap-style per-point Radius/Intensity + Banuba iris/sclera/pupil 분리 API. **P7 범위 제외, P8 진입 시 활용**.

### 1.3 멀티-AI 합의 (R1 8쟁점)

3/3 합의 6개, 2/3 다수결 2개, Hard veto 0건. 매우 건강한 분포. 자세한 표는 `P7-W0_brainstorm/synthesis.md` §1.

### 1.4 W 분할 결정

```
P7-W1 (0x501 spec fix + HIGH tier 회귀) — P0
   ↓ 안정 확인
P7-W2 (avg_iris_luma 측정 패스 + W6 Phase B/C) — P1, W1 후
   ↓
P7-W3 (A 그룹 cleanup) — P1, W1과 병렬 가능 (risk 0)
   ↓
P7-W4 (W5 Phase C 흰자 빛남 1단계 luma attenuation) — P2, W2 데이터 후
   ↓
P7-W5 (MID/LOW tier 통합 회귀) — P2, W4 후

P7-Spike-A (Oklab + Laplacian PoC) — 선택, P7-W4 결과 부족 시
```

### 1.5 소요 추정

| W | 소요 (작업일) | 의존성 |
|---|---:|---|
| P7-W1 | 0.5~1.0d | 없음 |
| P7-W2 | 2.0~3.0d | W1 |
| P7-W3 | 0.5~1.0d | (W1과 병렬) |
| P7-W4 | 2.0~4.0d | W2 권장 |
| P7-W5 | 1.0~2.0d + 기기 확보 | W1~W4 |
| Spike-A | 0.5~1.0d | (선택) |
| **합계** | **6.0~11.0d** | MID/LOW 기기 확보 별도 |

### 1.6 P8 예약 명시

Phase 8 = **뷰티 기능** (턱 깎기, 얼굴 사이즈 축소). P7 범위 밖.
- Substrate: MediaPipe Face Landmarker 478 landmarks (deep-research F4 + Codex 정정)
- 메커니즘: Snap-style per-point Radius + Intensity (<1 = inward warp), MediaPipe FaceMesh anchor + Delaunay triangulation 또는 TPS warp
- 아키텍처 precedent: Banuba iris/sclera/pupil 분리 API → P7 W5 sclera veto 방향성 표준 확인 (별도 검증 X, API surface precedent로만)

---

## 2. P7 W 본문 (각 W별 인사이트 요약)

### P7-W1: 0x501 spec fix + HIGH tier 회귀

> **상태 (2026-06-04)**: 🔄 코드 구현 완료 (commit `38368ec`, branch `feature/P7-W1-0x501-spec-fix`). 9개 `texture()`→`textureLod(uv,0.0)`, 빌드 통과, 적대적 4-agent 검증(편집 정확·등가성 identical·잔존 §8.9 landscape — `P7-W1_*.md` §6.3). **HIGH tier(S23+) 실기기 검증 대기 = DoD 게이트** → 0x501 logcat 사라짐 확인 후 develop 머지.

**목표**: GLSL ES 3.0 spec §8.9 위반 제거.

**대상 코드**: `cpp/src/gpu/shader_sources.cpp:1066-1083` — `if(uDetailReinject==1){ 9 sample texture() }` 패턴.

**수정**:
1. `texture(uCameraTexture, vTexCoord + offset)` 9샘플을 `textureLod(uCameraTexture, vTexCoord + offset, 0.0)`로 교체.
2. (선택) 9샘플을 dynamic branch 밖으로 hoist + 결과만 mix() 적용 — 더 안전하지만 변경량 큼.

**회귀 검증** (HIGH tier S23+):
- DetailReinject ON/OFF 토글 시 검은 화면 없음
- 0x501 잔재 사라짐 확인 (logcat)
- 디테일 재주입 시각 결과 동일 (textureLod의 명시 LOD가 implicit derivative와 동일 시각 결과)

**DoD**:
- 0x501 logcat에서 사라짐
- 6 SKU × 토글 ON/OFF 시각 회귀 없음
- 셰이더 컴파일 warning/error 0

**위험**: textureLod이 implicit derivative와 미세 시각 차이 가능. uniform 토글로 즉시 롤백 가능 보존.

**MID/LOW 회귀는 P7-W5로 분리** (기기 확보 의존).

---

### P7-W2: avg_iris_luma 측정 패스 + W6 Phase B/C

**목표**: W6 fallback 0.1225 대신 실측 source 연결 + 블링크 ramp/저조도 gate 최종 튜닝.

**설계** (R1 3/3 통합):
- 기존 `OES_TO_2D_FRAGMENT_SHADER` + 중간 RGBA FBO 결과 재사용.
- 그 위에 **small ROI(64×64) 다운샘플 패스** 추가 (iris ROI 영역만).
- mipmap level read 또는 작은 reduction shader로 평균 luminance 추출.
- **매 N=5 frame** 주기 측정 + **EMA(`L_t = 0.3·measured_t + 0.7·L_{t-1}`)** 평활 (W6 §5.7 정합).
- 저조도 gate에 hysteresis 추가 (1프레임 지연 oscillation 차단, Gemini 보강).

**LUMA 계수 테스트** (Phase 6 미실행 후속):
- shader vs CPU Rec.709 LUMA 계수 오차 ≤1% 단위 테스트 작성 (W1 §5.2.1 검증).
- 본 W에서 처리, 후속 별도 W로 미루지 않음.

**DoD**:
- avg_iris_luma fallback 0.1225 대신 실측 값 공급 확인
- N=5 EMA로 30fps 유지
- LUMA 계수 테스트 통과
- 블링크 ramp / 저조도 gate 실측 source 기반 토글 비교 (fallback vs 실측 차이 시각 만족 확인)

**위험**: 1프레임 지연이 저조도 gate 진동 유발 가능 → hysteresis 필수.

---

### P7-W3: A 그룹 cleanup 일괄

**목표**: Phase 6 후속 정리. P7-W1과 병렬 가능 (risk 0).

**범위**:
- **W9 데모 UI/KT 동기화**: 블렌드 drop-down 3종으로 축소(TintLinearV2/Multiply/ScreenLinear), 3D Light 버튼 제거, 기본 blendMode TintLinearV2 ID=5 설정.
- **deprecated no-op 제거**: `setLensHighlight`, `setHighlightEnabled` 등.
- **SKU 톤 분류 정정**: "누드 애쉬 로제"는 웜톤 X → 애쉬+누드 (W9 doc §1.4, P6-W0 §1.5, lens_meta.json 메타 업데이트).
- **Phase 6 보존 산출물 위치 명시**: W3/W4/W8 이월 트랙 재개 진입점 cross-link.

**DoD**:
- demo UI에서 사용하지 않는 항목 제거 확인
- 6 SKU × 5축 회귀 (이미 검증된 패턴 그대로 통과해야 함)
- deprecated API 호출 0건 (grep 검증)

**위험 0**. 별도 PR로 분리 가능.

---

### P7-W4: W5 Phase C 흰자 빛남 1단계 — luma attenuation

**목표**: TintLinearV2의 밝은 톤 sclera 영역 과증폭 1차 완화. Oklab/Laplacian PoC는 본 W 결과 부족 시 Spike-A.

**1단계 설계 (luma attenuation 강화)**:
- TintLinearV2의 `out = lens * detail`, `detail = clamp(pow(lum/avgLum, 0.7), 0.75, 1.25)` 수식에서 **밝은 영역(lum > sclera_threshold) attenuation** 추가:
  ```glsl
  // 흰자 영역 추정 (luma 기반)
  float sclera_factor = smoothstep(0.65, 0.85, lum);
  // detail 곱 적용 후 흰자 영역만 attenuation
  out = mix(lens * detail, lens * detail * uScleraAttenuation, sclera_factor);
  ```
- `uScleraAttenuation` uniform 추가 (기본 0.85, 토글 가능).
- W5 Phase A/B luma-only sclera veto와 직교 — 둘 다 적용 가능.

**검증**:
- HIGH tier 6 SKU × ON/OFF 토글
- 특히 SKU 2 (런웨이 그레이), SKU 5 (샤모 그래픽), SKU 6 (누드 애쉬 로제) — W9에서 흰자 빛남 강했던 SKU
- 만족 → 종결. 부족 → P7-Spike-A 진입점 명시.

**DoD**:
- 흰자 빛남 강도 W9 실측 대비 약 50% 이상 감소 (사용자 정성 체감)
- 다른 SKU 자연도 회귀 없음
- 토글 인프라 보존 (정량 비교 가능)

**위험**: sclera_factor 임계값(0.65~0.85)이 SKU별 다를 수 있음 — 메타 플래그로 SKU별 오버라이드 고려.

---

### P7-W5: MID/LOW tier 통합 회귀 (2026 현실 재정의)

**목표**: HIGH tier 외 디바이스 확장.

**Tier 재정의 (2026-06 시장 기준, P6 정의 폐기 — 8년 차이로 시장과 동떨어짐)**:

| Tier | 정의 | 대표 칩 | 대표 기기 |
|---|---|---|---|
| **HIGH** | 직전~당세대 플래그십 (2024~2026) | Snapdragon 8 Gen 3/4 (Adreno 750/800), Apple A18, Dimensity 9400 | Galaxy S24/S25, Pixel 9/10, iPhone 16 |
| **MID-상** (사용자 보유) | 2~3년 전 플래그십 / 현재 보급형 상위 | Snapdragon 8 Gen 2 (Adreno 740), Dimensity 8300, Mali-G68 | **Galaxy S23+ (보유)**, Pixel 8a, Galaxy A55 |
| **MID-하** | 4~5년 전 플래그십 / 저가 신형 상위 | Snapdragon 7 Gen 1/2 (Adreno 644), Dimensity 7300 | Galaxy S20/S21, Galaxy A35, Pixel 7a |
| **LOW** | 5+년 전 보급형 / 예산형 신형 | Snapdragon 695 (Adreno 619), Dimensity 6020, Helio G99 | Galaxy A23, 200~300달러대 신규 |
| ~~레거시 (P6 LOW = Adreno 530 이하)~~ | **P7 범위 제외** — 사용자 분포 <2% | — | — |

**한국 뷰티 앱 도메인 정합**: HIGH + MID-상 합쳐 ~60% 커버 시 매우 안정적. 뷰티 시뮬레이션 사용자는 신형 기기 + 좋은 카메라 비중 높음 (메모리 `low-light-usage-rare` 정합).

**대기 조건**: 추가 기기 1~2대 확보:
- **MID 검증** (사용자 보유 S23+가 이미 MID-상): Galaxy A55 또는 Pixel 8a급 (Adreno 720~730) — 신규 확보 필요
- **LOW 검증** (선택): Galaxy A35 또는 S21급 (Adreno 644/660) — 보유 시 추가, 없으면 P7-W5 완료 후 후속

**범위**:
- 사용자 보유 S23+(MID-상) + 추가 MID 1대 + 추가 LOW 1대(선택)에서 6 SKU × 5축 검증
- P7-W1 0x501 패치 cross-tier 재현/해결 확인
- P7-W2 avg_iris_luma 측정 패스 + EMA + hysteresis 안정성 확인
- P7-W4 흰자 빛남 attenuation cross-tier 시각 일관성 확인

**DoD**:
- HIGH(S23+) + MID 추가 1대에서 6 SKU × 5축 회귀 없음 (LOW는 선택)
- FPS 목표: HIGH 60+, MID 45+, LOW 30+ (2026 기준 상향 조정)
- 메모리 100MB 이하 유지

---

### P7-Spike-A (선택): Oklab + Laplacian PoC

**목표**: GPU 예산 측정 + 시각 이득 평가. P7-W4 결과 부족 시 진입.

**범위**:
- Oklab 변환 셰이더 함수 작성 (sRGB → Oklab → blend → Oklab → sRGB)
- 단일 SKU(SKU 6 누드 애쉬 로제 — 흰자 빛남 가장 강함) × 토글 비교
- Adreno 740 fragment cost 측정 (logcat FPS + GPU ms 추정)
- Multi-scale Laplacian은 1차 후보. 비용 과다 시 single-scale로 축소.

**DoD**:
- GPU 예산 측정 데이터 확보 (사용자 의사결정용)
- 시각 비교 결과 (Oklab vs TintLinearV2 + luma attenuation)
- 채택/롤백 판단 데이터

**Phase 8 진입 전 옵션 — Phase 7.5로 분리할 수도**.

---

## 3. 의존성 그래프

```
P7-W1 (0x501) ─── HIGH tier 안정 ───┬──→ P7-W2 (avg_iris_luma)
                                     │         ↓
P7-W3 (cleanup) [W1과 병렬, risk 0] │     P7-W4 (luma attenuation)
                                     │         ↓
                                     ├──→ P7-W5 (MID/LOW 회귀)
                                     │     (W4 결과 + 기기 확보)
                                     │
                                     └──→ P7-Spike-A (선택)
                                           Oklab/Laplacian PoC
                                           (W4 결과 부족 시)
```

**병렬 가능 포인트**:
- P7-W1 ↔ P7-W3 (risk 0 cleanup)
- P7-W2 + P7-W3 (각각 다른 영역)
- MID/LOW 기기 확보는 P7-W1 시작과 동시에 진행

---

## 4. R1 미결 → 사용자 확정 (2026-06-04)

1. **MID/LOW 기기 확보** — 2026 현실 재정의 채택 (레거시 제외). MID 추가 1대(예: Galaxy A55) 신규 확보 필요. LOW는 선택. 자세한 정의는 §2 P7-W5 본문.
2. **Spike-A 진입 시점** — **P7-W4 결과 후 결정**. luma attenuation으로 흰자 빛남 충분히 완화되면 미진행, 부족하면 즉시 진입.
3. **P7-W1 (0x501) ↔ P7-W3 (cleanup)** — **분리 PR**. W1은 셰이더 수정 + 회귀 검증 위주, W3은 demo UI/메타 정리 위주라 분리가 자연스러움.
4. **Phase 6 이월 트랙 (W3 환경 반사 / W4 B2 / W8 Pupil)** — **Phase 9(또는 7.5)로 분리**. P8(뷰티) 우선. §8 참조.

---

## 5. 메타 (브레인스토밍 프로세스 평가)

### 5.1 합의 분포

8쟁점:
- 3/3 합의: 6개 (75%)
- 2/3 다수결: 2개 (25%)
- Hard veto: 0건
- R2 필요: 없음

P6 평균(합의 71%, 다수결 25%)과 유사 — 안정적 분포.

### 5.2 모델별 강점 (R1 관찰)

- **Codex**: 기존 코드 구조 활용 통찰 (OES→2D 패스 재사용), 1차 출처 정확 인용 (Khronos GLSL ES 3.00 spec PDF, MediaPipe 478 정정)
- **Gemini**: 모바일 GPU 대역폭 한계 강조, 1프레임 지연 oscillation 주의 (EMA + hysteresis 보강), 응답 압축적
- **Claude**: solo dev 시간 + risk 최소 단계적 접근 강조, 메모리 정합 (`solo-dev-bench-method`, `qualitative-device-judgment`)

### 5.3 Stage 1 → Stage 2 검증 효과

- Stage 1 (deep-research)에서 추출된 4개 finding 모두 R1에서 보강·정정 발생:
  - F1: "본질 해결책" 표현 과함 → 단계적 접근으로 정정
  - F2: 즉시 수정 합의 + Adreno 6xx/7xx 일반화 caveat 재확인
  - F3: "신규 패스 설계 필요" → "기존 패스 재사용 + ROI 다운샘플"로 좁힘
  - F4: MediaPipe 468 → 478 landmarks 정정 (Codex)
- 외부 동향 인사이트 + 내부 컨텍스트 교차 검증의 가치 확인.

---

## 6. Phase 6 보존 산출물 (P7 작업 시 cross-link)

- **이월 트랙 보존 (메모리)**:
  - W3 환경 반사 scaffold: `feedback_qualitative_device_judgment`, `w4-env-reflection-deferred`
  - W4 B2 24클립 벤치: `docs/bench/P6-W4/`
  - W8 Pupil material: `P6-W8_pupil_material_conditional.md` 본문 Option E 보존
- **메타 인프라**: `lens_meta.json` 42 SKU (W7), `sku_id` C API + JNI (`c193576`), `LensSkuMetadata` 경량 JSON 파서
- **W5 sclera veto 3-way**: `uScleraVetoMode` (0=legacy / 1=color-veto / 2=luma-only) + A/B/C/D 토글 UI 보존
- **W6 토글 인프라**: `setBlinkRamp` / `setLowLightGate` / `setDetailReinject` 모두 internal API + 토글 UI 보존

---

## 7. 다음 액션

1. **사용자 합의**: §4 미결 4건 결정.
2. **P7-W1 본문 작성**: `docs/workPaper/P7-W1_*.md` — `ar-lens-implement` 호출 전 §5 확정 사항 정리 (W6 §5.7 EMA 공식 등 cross-link).
3. **P7-W3 본문 작성**: cleanup 일괄 (W1과 병렬 진행 시).
4. (선택) MID/LOW 기기 확보 일정 잡기.

---

## 8. Phase 후속 로드맵 (P7 종료 후)

### Phase 8 — 뷰티 (다음 주축, 사용자 확정)
- **범위**: 턱 깎기(jaw narrow), 얼굴 사이즈 축소 (face contour reshape), 후속 추가 검토
- **Substrate** (deep-research F4 + Codex 정정):
  - MediaPipe Face Landmarker **478 landmarks** (P6 기준 468 → 478 정정)
  - Snap Lens Studio Face Liquify 패턴 — per-point **Radius + Intensity** (<1 = inward warp "black hole")
  - Banuba iris/sclera/pupil 분리 API surface precedent (W5 sclera veto 방향성 표준 확인)
- **착수 시점**: P7 종료 후 즉시. P7-W4의 흰자 빛남 1단계가 만족스러우면 Spike-A 생략하고 바로 진입.

### Phase 9 (또는 Phase 7.5) — Phase 6 이월 트랙 재개 (분리 확정, 2026-06-04)
- **분리 사유** (사용자 확정):
  - P8(뷰티)이 한국 뷰티 도메인 비즈니스 임팩트 최우선
  - 환경 반사는 사용자 도메인 판단 "외곽 광택 현실 발생 드묾"으로 우선순위 낮음 (메모리 `w4-env-reflection-deferred`, `feedback_physical_assumption_validation`)
  - W3+W4+W8 묶음 30~40h = P7 전체보다 큰 덩어리, P7 정체성 흐림 방지
  - 이월 산출물(코드 + 벤치 인프라 + 메모리) 모두 보존돼 재개 비용 시간 경과 무관
- **재개 시 작업 묶음**:
  1. W3 Schlick/reflect 기반 환경 반사 재설계 brainstorm
  2. W4 B2 24클립 벤치 (env-map / periphery / OFF)
  3. W8 Pupil material restore (W4 B2 결과 2/3 Y 시 활성)
- **재개 시점**: P8 완료 후 비즈니스 임팩트 재평가 후 결정.
- **보존 산출물 위치**: §6 cross-link 참조.

### 로드맵 요약

```
Phase 6 (완료) ──→ Phase 7 (현재 인덱스) ──→ Phase 8 (뷰티) ──→ Phase 9 (이월 트랙)
   ↓                  ↓                       ↓                  ↓
develop 머지       안정성 + cleanup        substrate 활용     비즈니스 재평가 후
(64ba08b)         + 흰자 빛남 1단계        (FaceMesh 478)     환경 반사 재개
                                          face slimming
```

---

## 변경 이력

| 날짜 | 변경 |
|---|---|
| 2026-06-04 (초안) | R1 브레인스토밍 3/3 합의 + 2/3 다수결 종합 반영. |
| 2026-06-04 (확정) | 사용자 §4 미결 4건 확정. MID/LOW 2026 현실 재정의. Phase 9 분리. §8 후속 로드맵 추가. |
