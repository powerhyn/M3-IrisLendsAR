# P6-W5 B1+B8 벤치 결과 리포트

> **상태**: 🟡 **1차 (형광 환경) 체감 완료 — 잠정 결론**. 저조도 B8 + 흰자 빛남 수식은 후속.
> **명세**: `docs/workPaper/P6-W5_blend_sclera_bench.md` §4.1 / §5.2 / §5.5
> **평가 방식**: 1인 실시간 토글 체감 (`feedback_qualitative_device_judgment`)

---

## 1. 평가 요약

| 항목 | 값 |
|------|-----|
| 디바이스 | Galaxy S23+ (SM-S916N) |
| 평가일 | 2026-05-27 |
| 평가 방식 | 1인 실시간 토글 체감 (A/B/C/D + 스피너) |
| 투명도 | 80% (홍채-렌즈 블렌드 차이 가시성 최적) |
| 환경 | E1 형광 (실내) — **저조도/측광 미실시** |
| SKU (실측) | claset doll-choco(짙은 갈색+패턴), envie chameau-brown(연갈+주황 도트), claset dusty-grape 등 — lens-ar 42종 |
| 빌드 SHA | feature/P6-Works (에셋 교체 + LFS 반영분) |

> ⚠️ 1차는 형광 단일 환경 + 소수 SKU 체감. 통계적 결론 아닌 **방향성 잠정 판정**.

---

## 2. B1 결과 (블렌드: Normal vs CRL, + 기본 TintLinearV2 대조)

토글 체감 (TintLinearV2 vs Normal vs CRL, 우안 홍채 스크린샷 확대 교차 확인):

| SKU | 관찰 |
|-----|------|
| doll-choco (짙은 갈색) | **A(Normal) ≈ C(CRL) 차이 없음** — §1.4 예측 일치 (짙은 렌즈×짙은 홍채 → detail≈1) |
| chameau-brown (연갈+주황 도트) | **선명도 A(Normal) > C(CRL) > T(TintLinearV2)** / **자연스러움 T > C > A** |

- Normal(A): 도트/패턴 경계 가장 또렷 (원본 디자인 충실, 인공적)
- CRL(C): 경계 약간 블러 (Normal보다 자연)
- **TintLinearV2(기본): 가장 은은·자연** — 도트가 홍채에 가장 녹아듦

**판정 (잠정)**: **CRL 4번째 슬롯 불채택 경향.**
- 목적이 "실착용 미리보기(자연스러움)" → 기본 TintLinearV2가 그래픽 렌즈에서도 가장 자연.
- CRL은 Normal보단 자연스러우나 TintLinearV2를 못 넘음. Normal은 패턴 도드라져 부적합.
- → **블렌드 3종(TintLinearV2/Multiply/ScreenLinear) 유지, 4번째 슬롯 비움**이 유력.
- 단 "제품 디자인 정확 표현"이 목적이면 Normal(또렷)도 의미 — 제품 철학 결정 사항.

---

## 3. B8 결과 (sclera: color-veto vs luma-only)

| 환경 | 관찰 |
|------|------|
| E1 형광 | color-veto ↔ luma-only **체감 차이 미미 (비슷)** |
| E2 측광 / E3 저조도 | **미실시** — B8 위험 영역이라 추후 필수 |

**판정 (잠정)**: **luma-only 유력** (§5.5 Occam — 차이 미미 시 더 단순한 수식).
- ⚠️ 단 §5.3 핵심인 **저조도에서 luma-only가 홍채 외곽 깎거나 흰자 못 막는지 미검증**. 저조도 확인 후 확정.

---

## 4. ★ 새 발견 — TintLinearV2 흰자 빛남 (블렌드 수식 한계)

**사용자 관찰**: TintLinearV2가 흰자·눈꺼풀 등 **밝은 영역을 만나면 과증폭되어 빛남** → 어색.

**원인 (수식)**: `tinted = blend * lum * scale` — 원본 휘도(lum)에 비례해 틴트하므로 고휘도 영역에서 과증폭. luminance-tint 계열의 구조적 한계.

**위치**: B1(블렌드 슬롯)·B8(sclera veto)로는 **부분 완화만** 가능. 근본 해결은 **TintLinearV2 수식 자체 개선**(lum 상한/롤오프 등).
→ 메모리 `w5-b1-tintlinearv2-strength`("K=0.85 본질 한계")의 실기기 확인. **별도 후속 과제로 격상**.

---

## 5. 셰이더 반영 (Phase C — 저조도 확인 후 확정)

- [ ] B1: `uBlendMode==7`(CRL) **제거** + 4번째 슬롯 비움 (잠정 — 제품 철학 확정 후)
- [ ] B8: `calcScleraFactor` → **luma-only 단일 수식** (저조도 확인 후, §5.13)
- [ ] `uScleraVetoMode` uniform 제거 (벤치 종료 후)
- [ ] demo A/B/C/D 벤치 버튼 제거 (또는 디버그 플래그 뒤로)
- [ ] 99_final_decision.md §1.1 D6 / §1.2 C4 갱신

## 6. 후속 과제 (우선순위)

1. **TintLinearV2 흰자 빛남 수식 개선** (★ 실착용 품질 핵심) — lum 상한/롤오프. 별도 W.
2. **저조도 B8 재확인** — luma-only 저조도 안전성 (홍채 외곽 깎임). B8 확정 조건.
3. (조건부) B8 luma-only 채택 시 후속 W 저조도 임계 `0.3~0.5` 하향 검토 (§5.12).

---

## 7. 회귀 확인

- [ ] 채택 수식 반영 후 기존 SKU 시각 회귀 없음 (실기기)
- [ ] 4조합 토글 코드 제거 후 빌드/렌더 정상
