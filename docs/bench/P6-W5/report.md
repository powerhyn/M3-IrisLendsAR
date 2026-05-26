# P6-W5 B1+B8 벤치 결과 리포트

> **상태**: ⏳ Phase C 대기 (촬영/평가 후 작성)
> **명세**: `docs/workPaper/P6-W5_blend_sclera_bench.md` §4.1 / §5.2 / §5.5
> **응답 데이터**: `ratings_template.csv` (집계 후)

---

## 1. 촬영 요약

| 항목 | 값 |
|------|-----|
| 디바이스 | ____ |
| 평가일 | ____ |
| take 수 | 9 (예정) |
| 평가 방식 | 1인 실시간 토글 체감 |
| 빌드 SHA | ____ |

---

## 2. B1 결과 (블렌드: Normal vs ColorReplaceLinear)

take별 `B1_blend_winner` 집계 (Normal / CRL / tie):

| Take | SKU | winner | 메모 |
|------|-----|--------|------|
| T1 | S1 다크브라운 | — | — |
| T2 | S2 헤이즐 | — | — |
| T3 | S3 밝은그레이 | — | — |
| T4 | S4 불투명서클 | — | — |
| T5 | S5 화이트그래픽 | — | — |

**판정**: (CRL 채택 4종 / Normal 유지 / 조건부 `prefers_crl` / 차이 미미)

**근거** (정성 경향): ____

---

## 3. B8 결과 (sclera: color-veto vs luma-only)

take별 `B8_sclera_winner` 집계 (color / luma / tie):

| Take | SKU | 조명 | winner | 메모 |
|------|-----|------|--------|------|
| T1 | S1 다크브라운 | E1 형광 | — | — |
| T3 | S3 밝은그레이 | E1 형광 | — | — |
| T6 | S3 밝은그레이 | E3 저조도 | — | — |
| T7 | S3 밝은그레이 | E2 측광 | — | — |
| T8 | S1 다크브라운 | E3 저조도 | — | — |
| T9 | S1 다크브라운 | E2 측광 | — | — |

**판정**: (color-veto 채택 / luma-only 채택)

**근거** (조명별 경향, 특히 저조도 외곽 깎임): ____

---

## 4. 셰이더 반영 (Phase C 작업)

- [ ] B1: `uBlendMode==7` (CRL) — 채택/제거/조건부(`prefers_crl` 메타, §5.11)
- [ ] B8: `calcScleraFactor` 단일 수식 교체 (§5.13)
- [ ] `uScleraVetoMode` uniform 제거 (벤치 종료 후)
- [ ] demo A/B/C/D 벤치 버튼 제거 (또는 디버그 플래그 뒤로)
- [ ] 99_final_decision.md §1.1 D6 / §1.2 C4 갱신

## 5. 조건부 후속 (synthesis §3)

- [ ] B8 color-veto 채택 시 → phase-2 대표 SKU 2종 `0.4/0.6/0.8` 강도 스위프 (§5.14)
- [ ] B8 luma-only 채택 시 → 후속 W 저조도 임계 `0.3~0.5` 하향 검토 (§5.12)

---

## 6. 회귀 확인

- [ ] 채택 수식 반영 후 기존 SKU 시각 회귀 없음 (실기기)
- [ ] 4조합 토글 코드 제거 후 빌드/렌더 정상
