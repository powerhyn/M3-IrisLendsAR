# R2 결론 요약 (Claude 관점의 수렴 상태)

> **작성**: Claude (모더레이터)
> **시점**: R2 3응답 완료 직후, R3 검토 라운드 송신 직전
> **성격**: **Claude 1인 종합**. 세 모델 합의본 아님. R3에서 검증될 대상.

---

## 세 모델 R2 전체 동향

| 모델 | 변동 통계 (10쟁점 기준) | 특징 |
|------|----------------------|------|
| **Claude** | 유지 1 / 철회·수정 9 | Codex 논리 대부분 수용. "Codex R1이 가장 탄탄했다"는 판단. |
| **Codex** | 유지 7 / 수정 2 / 철회 1 | I10(W2 경계)만 철회. I2(4번째 슬롯) 신중화, I5(Claude 시간 수용). 나머지는 근거 보강. |
| **Gemini** | 유지 3 / 수정 4 / 철회 1 (I7) | I1(Periphery 샘플링)·I4(hybrid)·I8(완전 폐기) 고수. I2/I3/I6/I9는 Codex 수용. |

**전체 경향**: 세 모델이 Codex R1 축으로 당겨짐. Claude가 가장 많이 이동, Gemini가 중간, Codex는 자기 입장 고수하되 소폭 수정.

---

## 쟁점별 R2 수렴 상태

### A. 완전 수렴 (7쟁점 — R3에서 재논의 불필요)

| # | 쟁점 | 최종 방향 |
|---|------|----------|
| I3 | LTL realSpec 처리 | **폐기** — 환경 반사 분리 계층이 대체. 세 모델 합의 |
| I5 | 블링크 ramp | **down 50~80ms + up 100~120ms 비대칭** — Claude 시간 + Codex 비대칭 |
| I6 | 입력 계약 | **EyeRenderPacket 구조체** — 세 모델 합의. gaze_vector 드롭 |
| I7 | avg_iris_luma 하드코드 | **ROI 실측** — 세 모델 합의. `textureLod(camera, iris_center, 3.0)` 1샘플 |
| I9 | 홍채 디테일 재주입 | **원본 휘도 샘플링 + 저조도 gate** — Gemini 절차적 철회, 세 모델 합의 |
| I10 | W2-W3 경계 | **렌더러는 refiner 소유하지 않음** — Codex 철회 확정 |
| I2-부분 | 블렌드 3종 확정분 | **TintLinearV2 / Multiply / ScreenLinear** — 세 모델 합의 |

### B. 부분 수렴 / 실기기 벤치 이관 (4쟁점)

| # | 쟁점 | 잔여 대립 | 벤치 플랜 |
|---|------|----------|----------|
| I1 | 환경 반사 소스 | Claude(env+hybrid) / Gemini(Periphery) / Codex(env only) 3분립 | 3 프로토타입 × 4환경 × 2동작 = 24 클립 블라인드 |
| I2-잔여 | 블렌드 4번째 슬롯 | Normal vs ColorReplaceLinear | 4 SKU × 3 홍채 × 2 모드 = 24 클립 블라인드 |
| I4 | 림발 자동 감지 | 메타 only(Codex) vs 메타+자동 hybrid(Claude/Gemini) | 10 SKU (림발 O/X 섞음) 자동 감지 정확도 |
| I8 | sclera color veto | geometry+color-veto(Codex) vs luma-only(Gemini) | 2 SKU × 3 조명 × 2 방식 = 12 클립 |

### C. Claude가 수렴됐다고 봤지만 재검증 필요한 지점

R3 검토 시 세 모델이 다시 뒤집을 가능성 있는 항목들:

- **I2의 3종 확정**: Gemini는 R2에서 자기 하이브리드를 버리고 Codex 4종 수용했지만, 하이브리드의 가치를 더 주장할 수도. 재확인 필요.
- **I3 realSpec 폐기**: I1과 묶음 판단. 만약 R3에서 환경 반사 계층 도입이 벤치 이관된다면, realSpec도 "환경 반사 도입 전까지 임시 유지" 선택지 있음.
- **I9 원본 재주입**: 저조도 noise gate 임계값은 벤치 없이 정확히 정할 수 없다. 세 모델이 "R2에서 수렴"이라 했지만 실제 파라미터는 벤치 필요할 수도.

---

## R2 핵심 결정 사항 17개 (Claude 집계)

**제거 (D1~D6)**:
- D1 고정 조명 + 분석 노멀 라이팅 블록
- D2 `LIMBAL_ENABLED = false` 하드코드
- D3 LTL `realSpec` 2줄
- D4 `uAvgIrisLum = 0.35` 하드코드
- D5 "3D Light" 토글 UI
- D6 블렌드 4종 (Normal/Overlay/LumTint-nonlinear/SoftLight)

**추가/변경 (C1~C11)**:
- C1 블렌드 4종 세트로 재편
- C2 TintLinearV2 수식 (realSpec 제거된 LTL)
- C3 ScreenLinear 수식 신규
- C4 ColorReplaceLinear 수식 신규 (단 §B1 벤치 대상)
- C5 환경 반사 가산 계층 분리 (`blended += reflection * fresnel * finalAlpha`)
- C6 림발 기본 ON + SKU 메타 플래그
- C7 블링크 alpha ramp 비대칭
- C8 EyeRenderPacket 도입
- C9 avg_iris_luma ROI 실측
- C10 홍채 디테일 원본 재주입 + 저조도 gate
- C11 W2-W3 경계 명확화

→ 모두 `99_claude_synthesis.md` §1에 반영됨.

---

## Claude가 R2 작성 과정에서 스스로 내린 판단 (R3에서 재검증 대상)

> "Codex가 R1에서 가장 탄탄한 논리를 제시했고, Claude와 Gemini가 대부분 수렴하는 형태로 정리됨."

이 판단 자체가 Claude의 해석. Codex는 자기 R2에서 "Claude가 Normal 유지한 건 틀렸다" 같은 강한 비판을 했고, Claude가 이를 받아들여 9/10 쟁점에서 입장 변경. 하지만:

- **편향 가능성**: Claude가 "모더레이터" 역할을 자임하면서 Codex의 예리한 비판에 과도하게 동조했을 수 있음.
- **Gemini 입장 약화 가능성**: Gemini R2에서 유지한 3쟁점(I1/I4/I8)에 대해 Claude가 충분한 검토 없이 "벤치 이관"으로 밀어낸 측면.
- **독립적 R3 검증 필요**: Codex와 Gemini가 Claude의 `99_claude_synthesis.md`를 처음 읽고, 자기들의 R2 입장과 Claude 종합이 어긋난 지점에 대해 **정면 비판**해야 진짜 합의가 된다.

---

## R3에 넘길 질문 초안 (이 파일의 후속 작업)

1. **Claude가 "합의됐다"고 분류한 7쟁점**이 진짜 합의인지, 각 모델 관점에서 재검토.
2. **Claude의 "벤치 이관" 결정 4쟁점**이 적절한지, 또는 말로 더 좁힐 수 있는지.
3. **Codex의 CRL 수식, Claude의 환경 반사 hybrid, Gemini의 Periphery 샘플링**에 대한 상호 평가 (R2에서 서로 충분히 비판 못한 부분).
4. **구현 순서 S1~S5** (`99_claude_synthesis.md` §3)가 세 모델 관점에서 합리적인지.

이 질문들은 `11_r3_review_request.md`에서 구체적으로 정리한다.

---

## 문서 한계 명시

이 문서는 **Claude 1인이 R2 3응답을 읽고 본 수렴 상태**의 정리다:
- "완전 수렴" 판단은 Claude가 한 것. 세 모델이 직접 "합의"라고 서로에게 확인한 것이 아님.
- "벤치 이관" 결정도 Claude가 한 것. Codex/Gemini가 벤치 이관에 동의했는지는 R3에서 확인 필요.
- "R2 변동 통계"(9/10, 5/10, 3/10)의 해석도 Claude 관점.

**R3 검토가 이 한계를 해소하는 단계다.** R3 이후의 `99_final_decision.md`가 실제 합의본 자격을 갖는다.
