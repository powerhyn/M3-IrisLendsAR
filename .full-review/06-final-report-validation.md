# 05-final-report 검증 부록

## 목적

`05-final-report.md`의 핵심 주장과 집계 수치가 실제 코드/테스트 상태와 일치하는지 검증한 결과를 기록한다.

## 검증 메타데이터

- 검증 일시: 2026-03-06 (KST)
- 검증 대상 커밋: `0bf8432`
- 검증 대상 문서: `.full-review/05-final-report.md`
- 범위: P4-W3-05 모듈 (`QualityMetrics`, `ABCompare`, `ReleaseGate`, `ParamTuner`)

## 결론 요약

- 결론 1: 핵심 코드 이슈 다수는 사실이다.
- 결론 2: 문서 내부 번호/건수 집계에는 불일치가 존재한다.
- 결론 3: 일부 표현(예: `DeviceTier` ODR 위험)은 기술적으로 과장 가능성이 있다.

## 상세 검증 결과

### 1) 핵심 이슈 사실성 (PASS)

1. `detectHalo` zero-gradient baseline 미처리 이슈는 재현 가능한 사실이다.
   - 근거: `cpp/src/quality_metrics.cpp:328-335`
   - 설명: `original_boundary_gradient <= 1e-12`이면 `gradient_increase_ratio`가 0.0으로 남아 halo가 누락될 수 있다.

2. `ABCompare` halo 개선도 왜곡 이슈는 사실이다.
   - 근거: `cpp/src/ab_compare.cpp:141-143`
   - 설명: `b_halo <= 1e-9`이면 `halo_improvement`가 0.0으로 고정되어 비율 왜곡이 발생한다.

3. `ABCompare::addResult()` 무제한 성장 이슈는 사실이다.
   - 근거: `cpp/src/ab_compare.cpp:155-157`
   - 설명: 상한 없이 `results_.push_back(result)`만 수행한다.

### 2) 문서 정합성 (FAIL)

1. 우선순위 섹션 내 이슈 번호 중복
   - 중복 번호: `18` (Medium), `43` (Low)
   - 근거: `.full-review/05-final-report.md` 본문 표

2. 우선순위별 실제 건수와 메타데이터 총계 불일치
   - 표에 실제 기재된 건수: `P0=3, P1=13, P2=26, P3=13` (합계 55)
   - 문서 메타데이터 표기: `Total findings: 65`
   - 근거: `.full-review/05-final-report.md:16-91`, `:143`

3. 카테고리 집계 표 내부 불일치
   - 여러 행에서 `총 건수`와 `Critical+High+Medium+Low` 합이 다름
   - 예시: `Code Quality`는 `10`으로 표기되었지만 세부 합은 `12`
   - 근거: `.full-review/05-final-report.md:95-107`

### 3) 표현 정확성 (주의)

1. `DeviceTier` 관련 "ODR 위반 위험" 표현은 재검토가 필요하다.
   - 근거:
     - `cpp/include/iris_sdk/release_gate.h:32` (`iris_sdk::DeviceTier`)
     - `cpp/include/iris_sdk/gpu/gpu_beauty_backend.h:259` (`GPUBeautyBackend::DeviceTier`)
   - 설명: 서로 다른 스코프의 enum이므로 동일 심볼 재정의로 인한 직접 ODR 위반으로 단정하기 어렵다.

## 테스트 검증 기록

### 1) 타깃 단위 테스트

- 실행: `./build/bin/test_quality_tuning`
- 결과: `32 tests from 5 test suites`, `PASSED 32 tests`
- 판정: PASS

### 2) 전체 CTest 스모크 확인

- 실행: `ctest --test-dir build/cpp/tests --output-on-failure`
- 결과: `630`개 중 `8`개 실패, 일부 `Skipped`
- 비고: 실패 목록에는 본 리뷰 범위 외 테스트/빌드 타깃 이슈(`*_NOT_BUILT`)가 포함됨

## 재현 커맨드

```bash
# 문서 확인
nl -ba .full-review/05-final-report.md | sed -n '1,260p'

# 핵심 코드 라인 확인
nl -ba cpp/src/quality_metrics.cpp | sed -n '300,360p'
nl -ba cpp/src/ab_compare.cpp | sed -n '120,170p'

# 타깃 테스트
./build/bin/test_quality_tuning

# 전체 테스트 스모크
ctest --test-dir build/cpp/tests --output-on-failure
```

## 상태

- 기존 문서 수정: 없음
- 추가 문서 작성: `.full-review/06-final-report-validation.md`
