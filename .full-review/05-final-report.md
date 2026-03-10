# Comprehensive Code Review Report — P4-W4-01e

## Review Target

P4-W4-01e: Luminance Sharpen 패스 추가
- Branch: `feature/P4-W4-01e` vs `develop`
- Commits: `060b011` (원본), `6164a51` (리뷰 반영), HEAD (문서/테스트 동기화)
- 변경: 5 code files, +170 lines (원본) + 리뷰 반영 수정

## Executive Summary

FreqSep 파이프라인에 Luminance-only Unsharp Mask 패스를 추가하는 변경입니다. 원본 구현(060b011)에서 기능 회귀 2건(RT 풀 드롭아웃, mask 경계 halo)이 발견되어 6164a51에서 수정되었습니다. 현재 브랜치 상태에서 코드는 안정적이며, 문서/테스트 동기화도 완료되었습니다. 남은 기술 부채는 sRGB/Linear 색공간 불일치(Medium)로, 현재 sharpen_amount 범위에서 시각적 영향은 제한적입니다.

---

## 수정 이력

| 커밋 | 내용 | 상태 |
|------|------|------|
| `060b011` | 원본 구현 — Sharpen 패스 추가 | 기능 회귀 2건 포함 |
| `6164a51` | 리뷰 반영 — temp 조기 릴리스 + mask 경계 halo 방지 | 회귀 해결 |
| HEAD | 문서/테스트 동기화 — 헤더 주석 6-subpass, 회귀 테스트 4건, 작업 문서 갱신 | 현재 |

---

## Findings by Priority

### 해결된 이슈 (6164a51에서 수정 완료)

| ID | 이전 심각도 | 설명 | 상태 |
|----|-----------|------|------|
| RT-1 | **P1 (기능 회귀)** | temp 미릴리스로 동시 RT 4개 → onMemoryPressure 후 sharpen 탈락 | ✅ 해결: temp 조기 릴리스 |
| HALO-1 | **P1 (기능 회귀)** | 비피부 인접 픽셀이 blur에 기여 → mask 경계 합성 에지에 halo/ringing | ✅ 해결: 인접 mask 가중 |

### 현재 남은 이슈

#### Medium (P2) — 2건

| ID | 카테고리 | 설명 | 파일 |
|----|----------|------|------|
| F-1 | 셰이더 정확성 | sRGB 공간에서 Linear LUMA_709 계수 사용 — Composite가 gamma 인코딩 후 출력하므로 Sharpen이 sRGB 데이터에 linear 계산 적용. sharpen_amount 0.12~0.18 범위에서 시각적 영향은 제한적이나 파이프라인 색공간 일관성 위반. | `shader_sources.cpp:584` |
| F-2 | 에러 로그 | Sharpen 셰이더 컴파일 실패 시 `LOGE` 사용 + "successfully" 로그 출력 — graceful degradation인데 error 레벨 로그. `luminance_sharpen_program_`은 헤더에서 `= 0` 초기화되어 있으므로 미초기화 문제는 아님. | `gpu_beauty_backend.cpp:292,298` |

#### Low (P3) — 4건

| ID | 카테고리 | 설명 |
|----|----------|------|
| F-3 | 코드 중복 | LUMA_709 상수가 Composite/Sharpen 셰이더에 독립 선언 — 계수 변경 시 동기화 누락 위험 |
| F-4 | 테스트 정밀도 | `SharpenAmountMidQuality` tolerance 0.02f가 넓음, 0.005f로 축소 권장 |
| F-5 | 테스트 경계값 | `SharpenAmountRange` 검증 범위 [0.11, 0.19]가 실제 [0.12, 0.18]보다 느슨 |
| F-6 | 입력 방어 | `sharpen_amount` uniform에 `std::clamp` 방어 코드 없음 (mapSkinQuality만이 유일한 생성 경로이므로 실질적 위험 낮음) |

---

## Findings by Category

| 카테고리 | 건수 | 해결됨 | Medium | Low |
|----------|------|--------|--------|-----|
| 기능 회귀 | 2 | 2 | 0 | 0 |
| 셰이더 정확성 | 1 | 0 | 1 | 0 |
| 에러 처리 | 1 | 0 | 1 | 0 |
| 코드 품질 | 1 | 0 | 0 | 1 |
| 테스트 | 2 | 0 | 0 | 2 |
| 입력 검증 | 1 | 0 | 0 | 1 |
| **Total** | **8** | **2** | **2** | **4** |

---

## 테스트 현황

**총 71개 통과** (기존 62 + 매핑 5 + 회귀 4)

| 테스트 유형 | 건수 | 검증 대상 |
|------------|------|----------|
| 매핑 테스트 | 5 | mapSkinQuality() sharpen_amount 범위/단조성 |
| RT 풀 회귀 | 1 | temp 조기 릴리스 후 최대 동시 RT ≤ 풀 한도 |
| Mask 경계 회귀 | 3 | 비피부 인접→lumCenter 대체, 전비피부→sharpen 없음, 전피부→정상 동작 |

---

## Recommended Action Plan

### 다음 스프린트 (선택적)

| # | 작업 | ID | 노력 |
|---|------|-----|------|
| 1 | Sharpen 셰이더 컴파일 실패 로그 `LOGE`→`LOGW` + "successfully" 메시지 수정 | F-2 | Small |
| 2 | 테스트 tolerance/경계값 정밀화 | F-4, F-5 | Small |
| 3 | `sharpen_amount` uniform `std::clamp` 방어 | F-6 | Small |

### 후속 태스크 (기술 부채)

| # | 작업 | ID | 노력 |
|---|------|-----|------|
| 1 | 파이프라인 sRGB/Linear 색공간 정리 | F-1 | Medium |
| 2 | LUMA_709 상수 셰이더 간 공유 메커니즘 | F-3 | Small |

---

## 긍정적 관찰

1. **기존 아키텍처 패턴 완벽 준수** — 셰이더 선언~mapSkinQuality 6단계 패턴 1:1 일치
2. **Graceful degradation 3단계 방어** — 셰이더 실패 → RT 할당 실패 → 런타임 스킵
3. **리뷰 반영 품질** — 기능 회귀 2건을 정확히 수정, 회귀 테스트로 자동 검증 확보
4. **문서 동기화** — 작업 문서, 헤더 주석 모두 현재 코드와 일치
5. **리소스 수명 관리** — temp 조기 릴리스로 풀 한도 내 안전 동작

---

## Review Metadata

- Review date: 2026-03-10
- Branch state: HEAD (060b011 + 6164a51 + 문서/테스트 동기화)
- Phases completed: Code Quality, Architecture, Security, Performance
- Phases skipped: CI/CD (사용자 요청)
- Framework: C++17 / GLSL ES 3.1
- Tests: 71/71 passed
