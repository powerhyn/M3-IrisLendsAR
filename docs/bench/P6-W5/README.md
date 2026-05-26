# P6-W5 B1+B8 블렌드/sclera 벤치

> **목적**: B1(블렌드 Normal vs ColorReplaceLinear) + B8(sclera color-veto vs luma-only) 동시 평가.
> **명세**: `docs/workPaper/P6-W5_blend_sclera_bench.md` §5.1~§5.14
> **브레인스토밍 종합**: `docs/workPaper/P6-W5_brainstorm/synthesis.md`

---

## 핵심 설계 — 4조합 동시 캡처

같은 take에서 4조합을 런타임 버튼(A/B/C/D)으로 연속 토글 → 촬영 변수 제거.
해석은 **B1/B8 독립 분리**로 집계 (§1.9 / synthesis §1).

| 라벨 | 블렌드 (B1) | sclera veto (B8) | uniform |
|------|------------|------------------|---------|
| **A** | Normal (ID 0) | color-veto (Codex) | uBlendMode=0, uScleraVetoMode=1 |
| **B** | Normal (ID 0) | luma-only (Gemini) | uBlendMode=0, uScleraVetoMode=2 |
| **C** | ColorReplaceLinear (ID 7) | color-veto (Codex) | uBlendMode=7, uScleraVetoMode=1 |
| **D** | ColorReplaceLinear (ID 7) | luma-only (Gemini) | uBlendMode=7, uScleraVetoMode=2 |

**평가자에게는 A/B/C/D 라벨만 노출** (블라인드). 정답표는 코드/logcat에만 존재.

### B1/B8 독립 집계 방법

같은 take에서 4조합을 토글하며 두 축을 독립 비교:
- **B1 (블렌드)**: A↔C / B↔D 토글 (Normal vs CRL) — 홍채 중앙부 색·디테일 체감
- **B8 (sclera)**: A↔B / C↔D 토글 (color-veto vs luma-only) — 흰자 번짐·외곽 깎임 체감

---

## 산출물

| 파일 | 용도 |
|------|------|
| `checklist.md` | 9 take 트래킹 + 평가 방식 (1인 토글 체감) |
| `ratings_template.csv` | take 단위 판정 시트 (9행, B1/B8 winner) |
| `ratings_legend.md` | 토글 비교 방법 + B1/B8 정성 판정 룰 |
| `recording_guide.md` | §5.6 토글 절차 + (선택) self-blind 후편집 + adb 명령 |
| `report.md` | Phase C 결과 작성용 placeholder |
| `../../../scripts/p6w5_bench_helper.sh` | adb 자동화 (launch/record/pull + 선택 trim/randomize) |

---

## 진행 절차

1. **Phase A 완료** (이 커밋): 셰이더 4조합 토글 + demo A/B/C/D 버튼.
2. **준비** — APK 설치, SKU 6종, **native active 로그 확인 필수 (F-01)**.
3. **토글 평가** — 9 take 각각 실기기에서 A/B/C/D 토글 비교 → `ratings_template.csv` 9행 (1인 체감).
4. **(선택) self-blind** — 애매한 take만 녹화→trim→randomize로 익명 재평가.
5. **집계** — B1/B8 경향 정성 판정 (`ratings_legend.md`).
6. **Phase C 진입** — `report.md` 작성 + 셰이더 최종 수식 정리 (§5.13) + 99 §1.1 D6 / §1.2 C4 갱신.

---

## Phase C 시점 셰이더 정리 (벤치 결과 후)

- **B1 결과** → `uBlendMode==7` (CRL) 채택/제거/조건부(`prefers_crl` 메타, §5.11).
- **B8 결과** → `calcScleraFactor` 단일 수식 교체 (§5.13), `uScleraVetoMode` uniform 제거.
- **조건부 후속** (synthesis §3):
  - B8 color-veto 채택 시 → phase-2 대표 SKU 2종에 `0.4/0.6/0.8` 강도 스위프 (§5.14).
  - B8 luma-only 채택 시 → 후속 W에서 저조도 임계 `0.3~0.5` 하향 검토 (§5.12).

---

## 산출물 git 추적 정책

- `raw/`, `clips/`, `_truth.csv`는 `.gitignore` (대용량 영상 + 정답표 봉인).
- 본 README + checklist + ratings_template + legend + recording_guide + report만 추적.
