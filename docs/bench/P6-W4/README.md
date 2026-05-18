# P6-W4 B2 환경 반사 벤치 — Phase 6 이월 상태

> ⚠️ **2026-05-18 결정**: 이 디렉토리의 산출물은 **현재 미사용 (Phase 6 이월)**.
> Phase 7+ 환경 반사 재개 시점에 즉시 활용 가능하도록 보존.
> 출처: `docs/workPaper/P6-W4_env_reflection_bench.md` §1.17

---

## 보존 사유

W3 §5.8 Fresnel 옵션 C "외곽 강조"의 물리 가정 오류가 W4 Phase A 시각 검증에서 발견됨 (사용자 직관: "외곽 광택 자체가 현실 발생 불가능 케이스"). 24클립 벤치 진행 시 OFF가 가장 자연스러움이 결과로 예측 가능했기에 시간/리소스 절약 위해 미실행.

다만 **벤치 준비물(체크리스트/응답시트/촬영가이드/자동화 스크립트)** 자체는 Phase 7+ 재개 시 옵션 A(Schlick) 또는 B(reflect 기반)로 재검토할 때 그대로 재사용 가능. 그래서 삭제하지 않고 보존.

---

## 보존된 산출물

| 파일 | 용도 | Phase 7+ 재개 시 |
|------|------|------------------|
| `checklist.md` | 24클립 트래킹 (4환경 × 2동작 × 3프로토타입) | 그대로 사용 |
| `ratings_template.csv` | 평가자 3명 블라인드 응답 시트 (72행) | 그대로 사용 |
| `ratings_legend.md` | 메트릭 의미 + 판정 룰 | 그대로 사용 |
| `recording_guide.md` | §5.11 촬영 절차 + adb 명령 | 그대로 사용 |
| `report.md` | Phase C 결과 작성용 placeholder | 그대로 사용 |
| `../../../scripts/p6w4_bench_helper.sh` | adb 자동화 (launch/record/pull/trim/randomize) | 그대로 사용 |

---

## Phase 7+ 재개 절차 (참고)

1. **W3 §5.8 Fresnel 재검토**: brainstorm 재호출 — 옵션 A(Schlick) 또는 B(reflect 기반) 선택
2. **D1 분석 노멀 부분 복귀** 검토 (고정 광원 X)
3. **셰이더 패치 + 재빌드** (W3 §5.8 변경, sampleReflection 시그니처 수정 등)
4. **본 디렉토리 산출물로 24클립 벤치 진행** — `checklist.md`부터 시작
5. **결과 채택** → 99 §1.2 C5 갱신 → 머지

---

## 무시 가능한 변경 (이월 후 잡음 방지)

- `.gitignore`로 `_truth.csv`, `raw/`, `clips/` 제외
- 이월 결정으로 위 파일들은 만들지 않음 (Phase B 미실행)
- 본 README + 위 산출물만 git 추적
