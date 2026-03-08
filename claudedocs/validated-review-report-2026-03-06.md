# P4-W3-05 Code Review — Validated Final Report

**Date**: 2026-03-06
**Validation method**: Agent Council (Codex, Gemini) + Claude Chairman
**Original findings**: 51건 → **검증 후 확정**: 10건 YES, 3건 NO, 3건 DEFER + P2 26건 + P3 9건

---

## Validation Summary (P0+P1 검증 결과)

| # | 이슈 | Codex | Gemini | Claude | 합의 |
|---|------|:-----:|:------:|:------:|:----:|
| 1 | addResult() 무제한 메모리 성장 | YES | YES | YES | **YES** |
| 2 | addFrame() Laplacian 2-5ms/frame | NO | NO | NO | **NO** |
| 3 | vector::erase(begin()) O(n) | NO | NO | NO | **NO** |
| 4 | SSIM CV_64F 과도한 정밀도 | YES | YES | YES | **YES** |
| 5 | gridSearch() steps 상한 미검증 | YES | YES | YES | **YES** |
| 6 | catch(...) 무음 예외 삼킴 12곳 | YES | YES | YES | **YES** |
| 7 | DeviceTier enum 중복 정의 | NO | DEFER | DEFER | **DEFER** |
| 8 | JSON 이스케이핑 미처리 | YES | YES | YES | **YES** |
| 9 | 매직넘버/임계값 중복 정의 | YES | YES | DEFER | **YES** |
| 10 | Pimpl 패턴 미적용 | NO | DEFER | NO | **NO** |
| 11 | toGray 4회 중복 호출 | YES | YES | YES | **YES** |
| 12 | Sobel/Laplacian CV_64F | YES | YES | YES | **YES** |
| 13 | recommendPresets 등 미테스트 | YES | YES | YES | **YES** |
| 14 | ReleaseGate 경계값 미테스트 | YES | YES | YES | **YES** |
| 15 | 임계값 도출 근거 부재 | DEFER | YES | DEFER | **DEFER** |
| 16 | TemporalAnalyzer 스레드 안전성 미명시 | YES | YES | YES | **YES** |

---

## 기각 사유 (NO — 3건)

### #2 addFrame() Laplacian 2-5ms (기존 P0)
- TemporalAnalyzer는 품질 측정 도구이며 AR 렌더링 루프에 포함되지 않음
- "프레임 예산 6-15%"는 매 프레임 호출 전제인데, 실제 사용 패턴은 배치 또는 수동 호출
- 3자 전원 NO

### #3 vector::erase(begin()) O(n) (기존 P0)
- kMaxFrames=300 상한이 이미 존재
- 300개 double = 2.4KB 복사, 현대 CPU에서 나노초 단위
- deque로 교체해도 실측 차이 없음, 코드 복잡도만 증가
- 3자 전원 NO

### #10 Pimpl 미적용 (기존 P1)
- 릴리즈 전이라 API가 계속 변경 중
- 지금 Pimpl 적용하면 매 변경마다 impl도 수정 → 개발 속도 저하
- API freeze 시점에 한 번에 적용하는 것이 효율적
- Codex NO, Gemini DEFER, Claude NO

---

## 연기 사유 (DEFER — 3건)

### #7 DeviceTier enum 중복 (기존 P1)
- release_gate.h (네임스페이스 수준)와 gpu_beauty_backend.h (클래스 내부) — 스코프가 달라 ODR 위반 아님
- 의미적 불일치 위험은 있으나, 현재 두 값이 동일하므로 긴급하지 않음
- API 안정화 단계에서 공통 헤더로 통합 예정

### #9 매직넘버/임계값 중복 (기존 P1)
- quality_metrics.cpp에 이미 constexpr 상수 정의됨 (kLaplacianMinReduction 등)
- 다른 파일 리터럴은 ab_compare.cpp의 0.45(이상적 중간값)와 테스트 코드뿐
- 테스트 코드의 리터럴은 의도적일 수 있음 (상수 참조 시 동어반복 테스트가 됨)
- Codex/Gemini YES, Claude DEFER → 다수결 YES이나 긴급도 낮아 DEFER 처리

### #15 임계값 도출 근거 부재 (기존 P1)
- 문서화 이슈이며 코드 결함이 아님
- 릴리즈 전 한 번에 정리하면 충분
- Codex DEFER, Gemini YES, Claude DEFER

---

## Confirmed Findings by Priority

### P0 — Critical (즉시 수정) : 1건

| # | 카테고리 | 위치 | 이슈 | 영향 |
|---|----------|------|------|------|
| 1 | Security | ab_compare.cpp:155 | `addResult()` 무제한 메모리 성장 (CWE-400) | 장시간 QA 세션 시 OOM |

### P1 — High (수정 필요) : 9건

| # | 카테고리 | 위치 | 이슈 | 영향 |
|---|----------|------|------|------|
| 2 | Performance | quality_metrics.cpp:122-136 | SSIM CV_64F 과도한 정밀도 | 메모리 2배 낭비, CV_32F로 충분 |
| 3 | Security | param_tuner.cpp:32-102 | `gridSearch()` steps 상한 미검증 (CWE-400) | 조합 폭발 시 hang |
| 4 | Quality | 12+ 위치 | `catch(...)` 무음 예외 삼킴 (CWE-755) | 데이터 오염, 디버깅 불가 |
| 5 | Quality | ab_compare.cpp:255 | JSON 이스케이핑 미처리 (CWE-116) | malformed JSON 리포트 |
| 6 | Performance | quality_metrics.cpp:343-392 | 중복 toGray 4회 호출 | 불필요한 CPU 낭비 |
| 7 | Performance | quality_metrics.cpp:100,201 | Sobel/Laplacian CV_64F | 모바일에서 64MB 추가 할당 |
| 8 | Testing | test_quality_tuning.cpp | `recommendPresets()`, `generateReport()`, `reset()` 미테스트 | 핵심 로직 미검증 |
| 9 | Testing | test_quality_tuning.cpp | ReleaseGate 경계값 미테스트 | off-by-one 회귀 위험 |
| 10 | Documentation | quality_metrics.h | TemporalAnalyzer 스레드 안전성 미명시 | 멀티스레드 오용 위험 |

### Deferred (API 안정화 시점) : 3건

| # | 이슈 | 시점 |
|---|------|------|
| D1 | DeviceTier enum 공통 헤더 통합 | API freeze 시 |
| D2 | 매직넘버 상수 중앙화 | 릴리즈 전 정리 |
| D3 | 임계값 도출 근거 문서화 | 릴리즈 전 정리 |

### P2 — Medium : 26건

| # | 카테고리 | 이슈 | 비고 |
|---|----------|------|------|
| 11 | Correctness | `detectHalo` zero-gradient baseline 미처리 | 평탄 영역 halo 무시 |
| 12 | Quality | `validateBlurRadiusIndependence` 무의미한 검증 | 동일 입력 3회 = 항상 true |
| 13 | Quality | `computeScore()` 게이트 미통과 시 점수 0 절벽 | "거의 통과" 무시 |
| 14 | Quality | `GateVerdict` switch 반복 3곳 | toString() 필요 |
| 15 | Quality | `SkinToneGroup` 문자열 변환 중복 | skinToneToString 재사용 |
| 16 | Security | 32비트 size_t 곱셈 오버플로우 (CWE-190) | steps 상한으로 해결 가능 |
| 17 | Security | cv::Mat 이미지 크기 상한 미검증 | 대형 이미지 OOM |
| 18 | Security | ReleaseGate NaN/Inf 입력 미검증 | 혼란스러운 리포트 |
| 19 | Security | computeBlurRadius() 음수 face_width | 의미적 오류 |
| 20 | Performance | `generateReport()` 전체 벡터 복사+정렬 | partial_sort 권장 |
| 21 | Performance | Grid search steps^4 조합 폭발 | coarse-to-fine 권장 |
| 22 | Architecture | QualityMetrics-ReleaseGate 통합 편의 메서드 누락 | 수동 값 전사 |
| 23 | Architecture | 4개 모듈 C API 미노출 | 내부 도구 문서화 필요 |
| 24 | Architecture | blur_radius grid search 미사용 혼란 | 문서화 또는 분리 |
| 25 | Best Practice | `noexcept` + `push_back` → bad_alloc 시 terminate | noexcept 재검토 |
| 26 | Best Practice | `std::clamp` vs `max/min` 혼용 | 통일 |
| 27 | Best Practice | `snprintf` + `ostringstream` 혼용 | 통일 |
| 28 | Best Practice | All-static class → namespace 함수 | C++ Core Guidelines C.4 |
| 29 | Testing | NaN/Inf, 300프레임 순환, CONDITIONAL_GO | 엣지 케이스 미검증 |
| 30 | Testing | MID/LOW 디바이스 티어 미테스트 | HIGH만 검증 |
| 31 | Testing | 성능 회귀 테스트 0건 | 시간 제한 필요 |
| 32 | Testing | TEST_F 미사용 → DRY 위반 | 구조체 반복 초기화 |
| 33 | Documentation | 모듈 간 협력 관계 설명 부재 | 데이터 흐름 다이어그램 |
| 34 | Documentation | catch(...) 정책 미명시 | QualityMetrics에만 기재 |
| 35 | Documentation | computeScore() 가중치 근거 없음 | 경험적 판단 기록 |
| 36 | Documentation | 워크 페이퍼 high_freq_preserve 누락 | 4번째 변수 미반영 |

### P3 — Low : 9건

| # | 카테고리 | 이슈 |
|---|----------|------|
| 37 | Correctness | ABCompare halo_improvement b_halo<=0 시 왜곡 |
| 38 | Quality | snprintf 버퍼 크기 하드코딩 |
| 39 | Quality | formatReport 섹션 반복 패턴 |
| 40 | Quality | 빈 마스크와 품질 실패 미구분 |
| 41 | Security | toGray() 얕은 복사 참조 공유 |
| 42 | Security | 테스트 코드 고정 시드 RNG |
| 43 | Best Practice | string_view, structured bindings, constexpr 등 |
| 44 | Testing | 단색 이미지 SSIM 분모 0, null 콜백 |
| 45 | Documentation | 사용 예제, 워크 페이퍼 체크박스 갱신 |

---

## Recommended Action Plan

### Step 1: 안전 가드 (반나절)
- #1 addResult() 메모리 상한 추가
- #3 gridSearch() steps clamp(1,10) + 총 조합 10000 상한
- #4 catch(...) stderr 로그 추가
- #5 JSON escape 유틸리티

### Step 2: 성능 최적화 (1-2일)
- #2 SSIM CV_64F → CV_32F 전환
- #6 toGray 1회 캐시 (evaluateQuantitativeGate 리팩토링)
- #7 Sobel/Laplacian CV_32F 전환

### Step 3: 테스트/문서 보강 (1-2일)
- #8 recommendPresets, generateReport, reset 테스트 추가
- #9 ReleaseGate 경계값 테스트 추가
- #10 TemporalAnalyzer 스레드 안전성 문서화

### Deferred (릴리즈 전)
- D1 DeviceTier 공통 헤더 통합
- D2 매직넘버 상수 중앙화
- D3 임계값 근거 문서화

---

## Review Metadata

- **Original review date**: 2026-03-05
- **Validation date**: 2026-03-06
- **Validation method**: Agent Council (Codex gpt-5.3-codex, Gemini CLI) + Claude Opus 4.6 Chairman
- **Original findings**: 65건 → 중복 제거 51건 → 검증 후 P0+P1 확정 10건
- **Dismissed from P0/P1**: 3건 (#2 Laplacian 비용, #3 vector O(n), #10 Pimpl)
- **Deferred from P1**: 3건 (#7 DeviceTier, #9 매직넘버, #15 임계값 근거)
- **Final total**: P0 1건 + P1 9건 + Deferred 3건 + P2 26건 + P3 9건 = **48건**
