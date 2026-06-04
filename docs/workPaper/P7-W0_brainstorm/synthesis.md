# P7-W0 브레인스토밍 R1 종합 — Phase 7 전체 그림 합의

**작성일**: 2026-06-04
**참여**: Codex (gpt-5.5 xhigh) / Gemini (gemini-3-flash-preview) / Claude (Opus 4.7)
**라운드**: R1 1회 (P6-W9 §1.8 패턴 — 경량 종합 검증, R2 불필요)
**원문**: `codex_p7w0.md` / `gemini_p7w0.md` / `claude_p7w0.md`
**Stage 1 입력**: deep-research `wf_d2eb976b-38b` (107 agents, 25 sources, 4 핵심 finding)

---

## 1. 합의 + 다수결 결과 (요약 표)

| 쟁점 | 결과 | 분류 | 비고 |
|---|---|---|---|
| Q1. 0x501 처리 | **패치 먼저 + HIGH tier 회귀 묶기** | 3/3 합의 | MID/LOW는 별도 W로 (Codex/Gemini) |
| Q2. W5 Phase C 알고리즘 | **단계적 접근 — sclera/luma attenuation 먼저, Oklab/Laplacian은 spike 후** | 2/3 합의(Claude+Codex) / Gemini는 Laplacian 우선 | 사용자 최종 판단 항목 |
| Q3. avg_iris_luma 측정 | **기존 OES→2D 패스 재사용 위에 small ROI FBO 다운샘플 + N=5 frame + EMA** | 3/3 통합 (Codex의 "기존 패스 재사용" + Claude/Gemini의 "small ROI FBO" 융합) | |
| Q4. W 분할 | **W1: 0x501 / W2: avg_iris_luma / W3: cleanup / W4: 흰자 빛남 / W5: MID/LOW 회귀** | 약한 다수결 (각 모델 약간 다른 순서) | 아래 §3 상세 |
| F1 Oklab/Laplacian | **조건부 수용 + GPU 예산 검증 필수** | 3/3 합의 | "본질 해결책" 표현은 과함 |
| F2 0x501 spec 위반 | **수용 + 즉시 패치** | 3/3 합의 | Adreno 3xx→6xx/7xx 일반화는 보수적 (Claude/Codex/Gemini 동일) |
| F3 EXTERNAL_OES 변환 | **수용 + 기존 2D 패스 재사용으로 좁힘** | 3/3 합의 | 신규 변환 패스 아님 (Codex/Claude 보강), 1프레임 지연 주의 (Gemini) |
| F4 P8 substrate | **P7 범위 제외, 인지만** | 3/3 합의 | Codex 정정: MediaPipe 현재 478 landmarks (deep-research의 468 → 478) |

**Hard veto: 없음.** 모든 큰 방향 합의. 세부 우선순위만 미세 차이.

---

## 2. 쟁점별 상세

### Q1. 0x501 처리

**3/3 합의 사항**:
- `textureLod(uv, 0.0)` 명시 LOD 적용 또는 fetch hoist로 spec 위반 즉시 제거 (W6 detail-reinject 9샘플).
- 패치는 가벼움 — 셰이더 1줄 패턴 교체 수준.
- HIGH tier(S23+) 회귀로 패치 부작용 + 0x501 잔재 사라짐 동시 확인.

**미세 분기**:
- Claude/Codex: MID/LOW 회귀도 같은 W에 묶고 싶어함.
- Gemini: MID/LOW는 별도 W5로 분리.

**채택**: **MID/LOW는 별도 W5** (Codex+Gemini 다수결 + 기기 확보 일정 의존성).

### Q2. W5 Phase C 알고리즘

**2/3 합의 (Claude+Codex)**: 단계적 접근. TintLinearV2 + luma attenuation/sclera mask 강화부터, 부족 시 Oklab/Laplacian PoC.

**Gemini 입장**: Multi-scale Laplacian(sclera attenuation) 우선 적용. Oklab은 별도 PoC 후 결정. 사실상 단계적 접근과 비슷하나 첫 단계가 "Laplacian" vs "luma attenuation".

**채택**: **단계적 접근, 첫 단계는 luma attenuation 강화** (변경량 최소, 위험 최저). Laplacian/Oklab은 **별도 spike** 분리.

**근거**: 
- TintLinearV2 + sclera mask 강도/경계 정밀도 조정이 가장 가벼움 (변경 최소).
- Oklab은 cube root 비용 + 전체 블렌드 재작성 → 변경 큼, GPU 예산 미검증 (3/3 caveat).
- Laplacian은 다중 패스/샘플 비용 — Gemini 자체 반증 "Adreno 6xx 대역폭 한계 가능".

### Q3. avg_iris_luma 측정 패스

**3/3 통합**:
- Codex: 기존 OES→2D 변환 패스 (`OES_TO_2D_FRAGMENT_SHADER` + 중간 RGBA FBO) 재사용 + ROI 통계
- Claude: small ROI FBO 다운샘플 + N=5 EMA
- Gemini: small ROI FBO + N=5 EMA

**채택**: **기존 OES→2D 결과 RGBA FBO 위에 small ROI(64×64) 다운샘플 패스 추가 + N=5 frame + EMA(`L_t = 0.3·measured_t + 0.7·L_{t-1}`, W6 §5.7 정합)**.

**근거**:
- (b) 기존 패스 재사용으로 변환 비용 0 추가 (Codex 보강).
- (a) ROI만 별도 다운샘플로 통계 정밀 (Claude/Gemini).
- (c) compute shader는 GLES 3.1 호환성 + Adreno 드라이버 리스크 (3/3 기각).
- (d) N=5 EMA로 GPU 부담 5×감소 + 진동 억제 (3/3 합의).

**주의 (Gemini)**: 1프레임 지연이 저조도 gate 진동(oscillation) 유발 가능 → EMA 평활 + gate hysteresis 함께 설계.

### Q4. W 분할

**약한 다수결** (각 모델 약간 다른 우선순위 제안):

| W | Claude | Codex | Gemini | **채택** |
|---|---|---|---|---|
| W1 | 0x501 + 회귀 | 0x501 | 0x501 + HIGH 회귀 | **0x501 + HIGH tier 회귀** (3/3) |
| W2 | A 그룹 cleanup | avg_iris_luma | avg_iris_luma | **avg_iris_luma** (2/3) |
| W3 | W5 Phase C | W5 Phase C | A 그룹 cleanup | **A 그룹 cleanup** (Claude+Gemini 우선시 + cleanup risk 0) |
| W4 | avg_iris_luma | A 그룹 cleanup | W5 Phase C | **W5 Phase C 흰자 빛남** (avg_iris_luma 데이터 후) |
| W5 | (옵션) Spike | MID/LOW 회귀 | MID/LOW | **MID/LOW tier 통합 회귀** (Codex+Gemini) |
| 추가 | Spike-A (Oklab) | — | (PoC 후 결정) | **Spike-A 별도 (선택)** |

**최종 채택 분할**:
1. P7-W1: 0x501 spec fix + HIGH tier 회귀
2. P7-W2: avg_iris_luma 측정 패스 + W6 Phase B/C 토글 검증
3. P7-W3: A 그룹 cleanup 일괄 (P7-W1과 병렬 가능, risk 0)
4. P7-W4: W5 Phase C 흰자 빛남 (luma attenuation 강화)
5. P7-W5: MID/LOW tier 통합 회귀
6. P7-Spike-A (선택): Oklab + Laplacian PoC (GPU 예산 측정)

---

## 3. Finding 검토 종합

### F1 Oklab + Multi-scale Laplacian (조건부 수용)

3/3 모델 공통 우려:
- (Claude) "본질 해결책" 과함, sclera mask 강화로 충분할 수 있음
- (Codex) GPU 비용 + 실제 시각 이득 미검증, lightweight luma attenuation이 충분하면 과설계
- (Gemini) Oklab cube root 반복 + Laplacian 텍스처 샘플링 대역폭 한계 가능성

**결론**: P7-W4 1단계는 luma attenuation. Oklab/Laplacian은 Spike-A로 GPU 예산 측정 후 후속 판단.

### F2 0x501 spec 위반 (수용)

3/3 즉시 패치 합의. Codex 보강: "0x501의 유일 원인이라는 보장은 패치 전엔 확정 못 함" — P7-W1 회귀에서 명확화.

### F3 EXTERNAL_OES → 2D 변환 (수용 + 좁힘)

- Claude/Codex: 기존 변환 패스 재사용 → "신규 패스 설계 필요"는 과장.
- Gemini: 1프레임 지연이 W6 gate 진동 유발 가능 → EMA + hysteresis.

**결론**: P7-W2 설계는 기존 패스 재사용 + ROI 다운샘플 + EMA + gate hysteresis 함께.

### F4 Phase 8 substrate (인지만)

- 3/3 P7 범위 제외 동의.
- **Codex 정정**: MediaPipe 현재 Face Landmarker는 **478 landmarks** (deep-research의 468 → 478). P8 설계 시 반영 필요.

---

## 4. 미결 (사용자 최종 판단)

1. **MID/LOW tier 기기 확보 일정** — P7-W5 시작 시점 결정.
2. **Spike-A (Oklab/Laplacian PoC) 진입 시점** — P7-W4 luma attenuation 결과 후 판단? 별도 Phase 7.5?
3. **P7-W3 (A 그룹 cleanup) 와 P7-W1 병렬 진행 여부** — 같은 PR 또는 분리?
4. **Phase 6 이월 트랙(W3/W4/W8 재개) Phase 7 포함 여부** — 채택 안 했지만 별도 W로 둘지 결정 필요.

---

## 5. 메타 (멀티-AI 프로세스)

- **3모델 합의율 매우 높음** — 8개 쟁점 중 6개 3/3, 2개 2/3. Hard veto 0건.
- **3모델 다른 관점 보완 확인**:
  - Codex: 기존 코드 구조 활용 (OES→2D 패스 재사용) + 1차 출처 인용 (MediaPipe 478 정정)
  - Gemini: 모바일 GPU 대역폭 한계 강조 (Laplacian 우려) + 1프레임 지연 oscillation 주의
  - Claude: solo dev 시간 + risk 최소 단계적 접근 강조
- 메모리 `feedback_multi_ai_orchestration_bias` 작동 — Claude 단독 결정 피함.

**R2 불필요**. P7-W0_index.md 작성 진입.
