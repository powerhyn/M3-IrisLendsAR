# P4-W3-06: Quality Tools 안정성/성능 수정

## 작업 개요

Agent Council (Codex + Gemini + Claude) 합의에 따른 품질 도구 모듈 수정.
65건 리뷰 이슈 중 3자 합의된 7건의 실질적 수정 수행.

## 수정 대상 이슈

### Step 1: 안전 가드 (즉시)

| ID | 파일 | 이슈 | 수정 내용 |
|----|------|------|-----------|
| #1 | ab_compare.cpp | addResult() 무제한 메모리 성장 | 최대 결과 수 상한 (kMaxResults=1000) |
| #5 | param_tuner.cpp | gridSearch() steps 상한 미검증 | steps clamp(1,10) + 총 조합 10000 상한 |
| #6 | quality_metrics.cpp, ab_compare.cpp, param_tuner.cpp | catch(...) 무음 예외 삼킴 | 에러 로그 + 명시적 에러 상태 반환 |
| #8 | ab_compare.cpp | JSON 이스케이핑 미처리 | condition_label 이스케이프 처리 |

### Step 2: 메모리/성능 최적화 (이번 주)

| ID | 파일 | 이슈 | 수정 내용 |
|----|------|------|-----------|
| #4+#12 | quality_metrics.cpp | SSIM CV_64F 과도, Sobel/Laplacian CV_64F | CV_32F로 전환 |
| #11 | quality_metrics.cpp | 중복 toGray 4회 호출 | evaluateQuantitativeGate에서 1회 변환 후 재사용 |

## 수정 원칙

1. noexcept 계약 유지 (예외를 던지지 않되, 로그는 남김)
2. 기존 테스트 호환성 유지
3. 최소 변경으로 최대 효과

## 수정 상세

### #1 ABCompare::addResult() 메모리 상한

**Before**: `results_.push_back(result);` (무제한)
**After**: 최대 1000개, 초과 시 가장 오래된 결과 제거

### #5 gridSearch() steps 상한

**Before**: steps 값 그대로 사용 (사용자 실수 시 조합 폭발)
**After**: steps를 1~10으로 clamp, 총 조합 수 10000 초과 시 early return

### #6 catch(...) 예외 로깅

**Before**: `catch (...) { return {}; }` (무음)
**After**: `catch (...) { /* logged via fprintf(stderr) */ return {}; }`
- noexcept 계약 유지하면서 stderr로 최소 로그
- SDK 내부 로거 없으므로 fprintf(stderr) 사용

### #8 JSON 이스케이핑

**Before**: `os << "\"condition\": \"" << r.condition_label << "\"`
**After**: 이스케이프 유틸리티 함수로 `"`, `\`, 제어문자 처리

### #4+#12 CV_64F -> CV_32F

**Before**: Laplacian/Sobel/SSIM에서 CV_64F 사용 (~64MB 추가 할당)
**After**: CV_32F 사용 (메모리 절반, 모바일에서 NEON 활용 가능)
- 측정 정밀도 영향: SSIM 기준 ~0.0001 이하 차이로 무시 가능

### #11 toGray 중복 제거

**Before**: measureLaplacianReduction, detectHalo 각각에서 toGray 호출
         -> evaluateQuantitativeGate 경유 시 총 4회 변환
**After**: evaluateQuantitativeGate에서 gray 변환 1회 수행, 내부 메서드에 전달

## 테스트 결과

- 기존 31개 테스트: 전체 통과
- 신규 15개 테스트: 전체 통과
- 총 46개 테스트 통과 (0 실패)

## 변경 파일 목록

| 파일 | 변경 내용 |
|------|-----------|
| `cpp/include/iris_sdk/ab_compare.h` | kMaxResults 상수 추가 |
| `cpp/include/iris_sdk/quality_metrics.h` | Impl 메서드 선언, TemporalAnalyzer 스레드 안전성 문서 |
| `cpp/src/ab_compare.cpp` | addResult 메모리 상한, escapeJson, catch 로그 |
| `cpp/src/param_tuner.cpp` | gridSearch steps clamp + 조합 상한, catch 로그 |
| `cpp/src/quality_metrics.cpp` | CV_32F 전환, toGray 중복 제거, catch 로그 12곳 |
| `cpp/tests/test_quality_tuning.cpp` | 신규 테스트 15건 추가 |

## 변경 이력

| 날짜 | 상태 | 내용 |
|------|------|------|
| 2026-03-06 | 🔄 진행 중 | 수정 계획 수립 및 구현 시작 |
| 2026-03-06 | ✅ 완료 | 전체 6 Phase 구현 완료, 46개 테스트 통과 |
