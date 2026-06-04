# Claude (Opus 4.7) — Phase 7 R1 응답

**작성일**: 2026-06-04
**라운드**: R1 (독립 응답, 모더레이터 자임 금지)
**규칙 준수**: 메모리 `feedback_multi_ai_orchestration_bias` — 평범한 한 모델 의견으로만 제시.

---

## Stage 1 Finding 비판적 검토

### F1. Oklab + Multi-scale Laplacian (흰자 빛남)
- **수용**: 정성적 hue 안정 + Laplacian 다중 스케일 edge-aware는 학술 정합. W3C CSS oklab() 표준화는 강한 보조 신호.
- **유보**: Open question 그대로 — 모바일 GPU sRGB↔Oklab cube root 비용 30fps 예산. Adreno에서 fragment shader당 cube root + 행렬 곱은 보통 0.05~0.15ms/frame 예상하나 실측 필요.
- **부분 반대**: "본질 해결책"이라는 표현은 과함. W5 Phase A/B에서 luma-only가 이미 선택됐고 흰자 빛남은 알고리즘 한계 + sclera mask 강도의 복합. 색공간 교체가 silver bullet은 아닐 가능성.

### F2. 0x501 = spec 위반
- **수용**: 7 claims 3-0 vote 우세. Khronos spec §8.9 명문 + Maister 블로그 + Mozilla 실증 — 매우 강한 증거.
- **수용**: 표준 회피 `textureLod(uv, 0.0)` 명시 LOD는 W6 detail-reinject 9샘플에 직접 적용 가능. 변경량 작음 (셰이더 1줄 패턴 교체).
- **caveat 무게**: Adreno 3xx → 6xx/7xx 일반화 보수적이라는 caveat에 동의하지만, **spec 위반 자체가 해결 사유로 충분**. Adreno 6xx/7xx에서 발현 안 되더라도 향후 드라이버 업데이트나 Mali 디바이스에서 발현 가능성 존재.

### F3. EXTERNAL_OES → 2D 변환 강제
- **수용**: Android 공식 + Khronos extension spec 정합. 다른 해석 여지 없음.
- **보강**: 이미 IrisLensSDK는 OES→2D 변환 패스를 갖고 있음 (`CameraGLRenderer`가 OES texture를 받아 2D FBO로 그림). avg_iris_luma 측정은 그 결과 텍스처 + ROI 영역에서 통계 추출이라 **신규 변환 패스가 필요하다기보다는 기존 패스의 산출물 활용 + ROI 다운샘플** 설계 문제로 좁혀짐.
- **부분 수정 제안**: "변환 패스 신규 설계 필요"는 다소 과장. ROI 영역만 64×64로 추가 다운샘플하는 작은 second-pass면 충분할 가능성.

### F4. Phase 8 substrate (P7 범위 밖)
- **수용**: P8 토대 확정 — FaceMesh 468 vertex + Radius/Intensity. 인지만, 깊이 들어가지 않음.
- **반대**: Banuba "분리 API 노출"이 IrisLensSDK W5 sclera veto 방향성 "업계 표준 검증"으로 일반화되는 건 약함. Banuba는 separate recolor mode를 노출했을 뿐 sclera 보호 알고리즘 자체 검증된 것 아님. API surface precedent로만 활용 (deep-research caveat 그대로).

### Refuted 항목 활용
- if→branchless mix() 평탄화는 Adreno 3xx 사례 — IrisLensSDK Adreno 6xx/7xx에서 동일 효과인지 불확실. 단 dynamic branch 자체를 줄이는 방향은 W6 detail-reinject 외 다른 dynamic branch에도 적용 가능 (cleanup 가치).
- PBO 비동기 readback이 출처 부족으로 refuted됐지만, Khronos GL_PIXEL_PACK_BUFFER spec은 별개 — Q3 검토 시 직접 spec 참고하면 됨.

---

## Q1. 0x501 즉시 수정 vs 디바이스 매트릭스 회귀 우선

**추천**: **(c) 둘 다 같은 W, 패치 먼저 + 회귀로 cross-tier 확인**.

**근거**:
- F2 confidence high + spec 위반 강한 증거 → 패치는 미루지 않을 이유 없음.
- 회귀 검증 단독 W는 결과가 "재현 안 됨"이어도 패치는 여전히 필요 (spec 위반 사실 변화 없음). 회귀 W를 우선시키는 (b)는 선후 관계 어색.
- 패치 후 cross-tier(HIGH 확정 + MID/LOW 별도 확보) 회귀로 (1) Adreno 6xx/7xx 재현 여부 확인 (2) 패치 부작용(시각 차이) 확인을 한 번에 묶기.
- (d) cleanup PR로 패치 분리는 0x501 잔재 추적이 다른 작업과 묶이는 위험.

**위험**: HIGH tier에서 시각 회귀 발생 시 즉시 롤백 가능해야 함 (uniform 토글로 detail-reinject ON/OFF 비교 보존). MID/LOW 미확보 상태라면 회귀 검증 미완으로 P7-W1 종결 못 함 — MID/LOW 확보 시점이 의존.

---

## Q2. W5 Phase C 알고리즘 선택

**추천**: **(a) 단계적 접근 — TintLinearV2 + sclera mask 강화부터 → 효과 부족 시 Oklab/Laplacian**.

**근거**:
- W5 Phase A/B가 sclera veto luma-only로 닫혔지만, **veto 강도와 sclera 경계 마스크 정밀도가 별개 축**. 마스크 강화는 luma-only를 살리면서 흰자 보호 강도만 올리는 작은 변경 — Oklab 도입(중간/큰 변경)보다 risk 작음.
- F1 Oklab은 GPU 예산 미검증. PoC에 sRGB↔Oklab 변환 코드 작성 + 모든 블렌드 함수 재작성 필요 — 변경량 큼.
- 단계 A에서 만족 시 단계 B(Oklab) 생략 가능 — solo dev 시간 절약. 단계 A에서 부족하면 그때 Oklab 도입에 명확한 동기 부여.
- W5 Phase A/B 결정(luma-only)을 잇는 자연스러운 흐름.

**위험**: sclera mask 강화만으로 부족할 가능성 (밝은 톤 SKU에서 알고리즘 본질 한계라면). 단계 B 진입 시 별도 brainstorm 필요.

**보조 제안**: 단계 A 마무리 시 Oklab/Laplacian 미니 PoC를 별도 spike(짧은 실험)로 진행 → GPU 예산 데이터만 확보. P7-W3에 포함하지 않고 P7-W3 후속 spike로 분리.

---

## Q3. avg_iris_luma 측정 패스 설계

**추천**: **(a) 별도 small ROI FBO 다운샘플 (64×64 → mipmap mean) + (d) 매 N frame 측정 + EMA 평활**.

**근거**:
- (a)는 W3-04 Adreno mipmap+dynamic branch 이슈를 가장 잘 회피 — 64×64 작은 FBO에 ROI만 그리고, 별도 dynamic branch 없는 sample 후 mipmap level read는 안전.
- (b) 기존 2D 변환 패스 재사용은 ROI 영역만 통계 뽑기가 어려움 (전체 frame 통계가 됨). 별도 ROI 텍스처가 더 깔끔.
- (c) compute shader는 GLES 3.1 호환성 + Adreno 드라이버 품질 리스크 (P6-W1 §1.8에서 이미 기각된 이력).
- (d) EMA는 W6 §5.7 EMA 공식과 정합 (`L_t = 0.3 · measured_t + 0.7 · L_{t-1}`). 매 5 frame 측정 + EMA = 1초 안에 안정 + GPU 부담 5×감소.

**위험**: 64×64 FBO 추가가 GPU 메모리/상태 변경 비용 — 측정 결과 0.1~0.2ms/measurement 예상. 매 5 frame이면 평균 0.02~0.04ms/frame.

---

## Q4. P7 W 분할 + 의존성 + 소요 추정

### 권장 분할 (4 W + 1 spike)

```
P7-W1: 0x501 수정 + 디바이스 회귀 (Q1 (c) — 패치 + cross-tier 확인)
  ├─ 셰이더 textureLod 명시 LOD (또는 hoist) 패치
  ├─ HIGH tier 회귀 (S23+) — 시각 차이 + 0x501 사라짐 확인
  ├─ MID/LOW tier 회귀 (확보 후)
  └─ 결과 메모리 + W 문서 정리

P7-W2: A 그룹 cleanup 일괄
  ├─ W9 데모 UI/KT 동기화 (블렌드 drop-down 축소, 3D Light 제거, default ID=5)
  ├─ deprecated no-op 제거 (setLensHighlight, setHighlightEnabled)
  ├─ SKU "누드 애쉬 로제" 톤 분류 정정 (메타 + W9 doc + P6-W0)
  ├─ 사용하지 않는 P6 코드 정리 (W3 scaffold OFF 기본 유지 — 보존)
  └─ Phase 6 보존 산출물 위치 명시 (Phase 7+ 재개용)

P7-W3: W5 Phase C 흰자 빛남 (Q2 (a) — sclera mask 강화)
  ├─ sclera mask 강도 + 경계 정밀도 튜닝 (uniform 추가)
  ├─ A/B/C/D 토글에 추가하여 비교 (현재 토글 슬롯에 합치거나 별도)
  ├─ HIGH tier 6 SKU 재검증 (특히 SKU 2 그레이 / SKU 6 누드 애쉬 로제)
  └─ 결과 만족 시 종결, 부족 시 Oklab spike 진입점 명시

P7-W4: W6 Phase B/C avg_iris_luma 실측 source 연결 (Q3 (a)+(d))
  ├─ small ROI FBO 다운샘플 패스 추가
  ├─ EMA 평활 + 매 N frame 측정
  ├─ LUMA 계수 shader vs CPU 오차 ≤1% 테스트
  ├─ 블링크 ramp / 저조도 gate 실측 source 기반 최종 튜닝
  └─ 토글 비교 (fallback 0.1225 vs 실측)

P7-Spike-A (선택): Oklab + Laplacian PoC GPU 예산 측정
  ├─ Oklab 변환 셰이더 함수 작성 (단순)
  ├─ Adreno 740 fragment cost 측정
  └─ 결과만 메모리 + 후속 W 진입점

P7-W5: Phase 6+ 이월 트랙 재개 (선택, 머지 후속)
  ├─ W4 환경 반사 재개 brainstorm (Schlick/reflect 기반)
  ├─ W8 Pupil material 트랙 재검토 (W4 결과 종속)
  └─ Phase 7 후반 또는 별도 Phase
```

### 의존성 그래프

```
P7-W1 (0x501) ──┐
                ├─→ HIGH tier 안정 확인 후 ──→ P7-W3, P7-W4 진입
P7-W2 (cleanup)─┘   (수정 후 회귀 영향 없는 base 확보)

P7-W3 (흰자 빛남) ──→ (만족) Phase 7 종결 후 P8 진입
                  ──→ (부족) Spike-A → 후속 W

P7-W4 (avg_iris_luma) ──→ W6 Phase B/C 완성 → P7-W3 데이터로도 활용

P7-W5 (W4/W8 재개): Phase 7 후반 또는 Phase 7.5
```

### 병렬 가능

- P7-W2 (cleanup) 는 P7-W1 패치 후 언제든. W1 회귀 검증과 병렬 가능.
- P7-W3, P7-W4는 W1 안정 확인 후 병렬 가능.

### 소요 추정

| W | 소요 | 비고 |
|---|---|---|
| P7-W1 | 4~8h | 패치 1h + HIGH 회귀 1h + MID/LOW 회귀 (기기 확보 시간 별도) |
| P7-W2 | 2~4h | 일괄 정리 |
| P7-W3 | 8~12h | sclera mask 강도 튜닝 + 6 SKU × 토글 검증 + 만족도 판정 |
| P7-W4 | 10~16h | FBO 패스 추가 + EMA + LUMA 테스트 + 실기기 재튜닝 |
| Spike-A | 4~6h | PoC + 벤치만 |
| **P7 합계** | **24~46h** | (W5 제외) |

---

## 종합 추천 행동

1. **P7-W1 즉시 착수** — 0x501 패치(가벼움) + HIGH tier 회귀(즉시) → MID/LOW 기기 확보 대기.
2. **P7-W2 병렬 진행** — cleanup은 risk 없으므로 W1 회귀 결과 기다리는 동안 처리.
3. P7-W3 진입 전 사용자 합의 — sclera mask 강화 vs Oklab spike 우선순위.
4. P7-W4는 W1 안정 후 진입 — W3와 병렬 가능.
5. P7-W5(W4/W8 재개)는 Phase 7 후반 별도 brainstorm. 우선 P7-W1~W4 완료 우선.

---

## 미결 (사용자 판단 필요)

- MID/LOW tier 기기 확보 일정 — P7-W1 종결 시점 결정
- Spike-A를 P7 안에 포함할지, 후속 별도 Phase로 둘지
- P7-W5(W4/W8 재개)를 Phase 7에 포함할지 별도 Phase 7.5로 분리할지
