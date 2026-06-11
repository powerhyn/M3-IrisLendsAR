# REFACTOR-1: 리팩토링 감사 리뷰 (1단계)

| 항목 | 내용 |
|---|---|
| 상태 | ✅ 완료 (2026-06-11) |
| 브랜치 | `refactor/p1-audit` (develop d224854 기준) |
| 계획 문서 | `docs/lenssim-handoff/refactoring-audit-and-tracking-migration-plan.md` 1단계 |
| 산출물 | `docs/lenssim-handoff/audit-report.md` |

## 목표

추적 레이어 주입형 전환(2~3단계)에 앞서 현재 코드 품질을 측정하는 읽기 전용 감사.
6개 관점(정확성/스레드·수명/GL·GPU/결합도/플랫폼 패리티/테스트 안전망) 멀티에이전트 적대적 리뷰 +
LensSimulator 함정 13건 대조 + TFLite 결합 지점 전수 조사.

## 수행 내역

- finder 9개(6관점 세분) → finding별 적대적 반박 검증(blocker/major 2표+캐스팅보트, minor 1표) →
  결합도 누락 sweep + 주입지점 seam 적대 검증 → 완전성 비평 → 2라운드 보완 감사 4건 → 모듈 판정 패널 2인(보수/부채)
- 서브에이전트 약 380개. 사용량 한도 2회·수동 중단 2회 발생 → 트랜스크립트 수확 체크포인트로 전량 회수
  (이 패턴은 글로벌 스킬 `safety-workflow-checkpoint`로 문서화)

## 결과 요약

- 원시 findings 141건 → 적대적 검증 확정 **133건** (blocker 2 / major 63 / minor 68, 폐기 8)
- 모듈 판정 (패널 합치): tracking **재작성(외부 교체)**, gpu-render/infra **리팩토링**,
  orchestration/geometry/android-binding/demo-gl **부분 재작성**, cpu-render는 이견(A: 부분 재작성 / B: 폐기) → 2단계 ADR에서 결정
- 종합 권고: 전면 재작성 기각 — "경계 승격형 리팩토링 + tracking 외부 교체".
  선행 조건: 데모 검증 통로 정화 + 골든 비교 인프라 신축 (현재 렌더 픽셀 검증 0건)
- 교체 난이도: 원견적 12~16 사람·일 + 패널 보정 → 약 17~24 사람·일 (1인 4~5주)

## 이슈 및 학습

- Workflow resume 캐시가 prefix-순서 매칭이라 병렬 verify 단계는 중단 시 재실행됨 → 트랜스크립트 수확 체크포인트 패턴으로 전환해 해결
- 한도 사망 에이전트는 null 반환 → null 가드 없으면 finding이 조용히 소실됨 (연속 워크플로에 incomplete 분류 추가)

## 외부 교차 검토 (Codex gpt-5.5 xhigh)

- blocker 2건 + 표본 major 전수를 코드 직접 대조로 **전부 사실 확인** (BlazeFace 건은 MediaPipe 원본 pbtxt 대조 포함)
- 반박 3건 검증 후 수용 2·부분 수용 1, 추가 발견 2건 수용, 거부 0 → 보고서 §11 + §6.5/§9/§10 반영
- 핵심 반영: cpu-render 삭제는 공개 C API 호환성·버전 정책 판단(ADR 결정 항목 추가), JNI CMake OpenCV REQUIRED 제거 범위 추가, 견적 +2~4일 버퍼

## 다음 단계

- 사용자 보고서 검토 → 승인 시 2단계 (ADR 작성 + 골든 베이스라인 구축)
- 필요 시 Gemini 추가 교차 리뷰 후 ADR 확정

## 변경 이력

- 2026-06-11: 감사 완료, audit-report.md 작성 (코드 수정 없음)
- 2026-06-11: Codex 교차 검토 반영 (§11 신설, 정오표·견적 보정)
