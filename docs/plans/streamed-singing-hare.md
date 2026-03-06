# P4-W3-06 Quality Tools Bugfix Implementation Plan

## Overview

Agent Council 합의 기반 10건 이슈 수정 (P0 1건 + P1 9건)
- 참조: `claudedocs/validated-review-report-2026-03-06.md`
- 작업 문서: `docs/workPaper/P4-W3-06_quality_tools_bugfix.md`

## Phase 1: Header Changes (의존성 없음)

### 1-1. `cpp/include/iris_sdk/ab_compare.h`
- private 섹션에 `static constexpr std::size_t kMaxResults = 1000;` 추가

### 1-2. `cpp/include/iris_sdk/quality_metrics.h`
- QualityMetrics private 섹션에 Impl 메서드 선언 추가:
  ```cpp
  double measureLaplacianReductionImpl(const cv::Mat& originalGray,
                                        const cv::Mat& processedGray,
                                        const cv::Mat& mask) const noexcept;
  double detectHaloImpl(const cv::Mat& originalGray,
                        const cv::Mat& processedGray,
                        const cv::Mat& mask) const noexcept;
  ```
- TemporalAnalyzer 클래스 Doxygen에 스레드 안전성 `@note` 추가:
  ```
  @note This class is NOT thread-safe. External synchronization is required
        if accessed from multiple threads. All methods including addFrame()
        modify internal state (reduction_ratios_, frame_count_).
  ```

## Phase 2: Source — Safety Guards

### 2-1. `cpp/src/ab_compare.cpp` — #1 addResult() 메모리 상한
- `#include <cstdio>` 추가
- `addResult()` 내 `results_.push_back()` 후:
  ```cpp
  if (results_.size() > kMaxResults) {
      results_.erase(results_.begin());
  }
  ```

### 2-2. `cpp/src/ab_compare.cpp` — #8 JSON 이스케이핑
- 익명 네임스페이스에 `escapeJson()` 유틸리티 추가:
  ```cpp
  namespace {
  std::string escapeJson(const std::string& s) {
      std::string out;
      out.reserve(s.size());
      for (char c : s) {
          switch (c) {
              case '"':  out += "\\\""; break;
              case '\\': out += "\\\\"; break;
              case '\n': out += "\\n";  break;
              case '\r': out += "\\r";  break;
              case '\t': out += "\\t";  break;
              default:
                  if (static_cast<unsigned char>(c) < 0x20)
                      out += ' ';
                  else
                      out += c;
          }
      }
      return out;
  }
  } // namespace
  ```
- `generateReport()` 내 `condition_label` 출력 시 `escapeJson()` 적용

### 2-3. `cpp/src/param_tuner.cpp` — #5 gridSearch() steps 상한
- `gridSearch()` 시작부에:
  ```cpp
  const int clamped_steps = std::clamp(range.steps, 1, 10);
  const size_t total = static_cast<size_t>(clamped_steps) *
                       static_cast<size_t>(clamped_steps) *
                       static_cast<size_t>(clamped_steps) *
                       static_cast<size_t>(clamped_steps);
  if (total > 10000) {
      fprintf(stderr, "[IrisSDK] gridSearch: combination overflow (%zu), skipped\n", total);
      return best;
  }
  ```
- 4개 내부 루프에서 `range.steps` → `clamped_steps` 교체

## Phase 3: Source — Performance Optimization

### 3-1. `cpp/src/quality_metrics.cpp` — #4+#12 CV_64F → CV_32F
- `computeSSIMChannel()`: `CV_64F` → `CV_32F`, 상수에 `f` 접미사
- `computeMaskedLaplacianVariance()`: `cv::Laplacian(gray, laplacian, CV_64F)` → `CV_32F`
- `computeMaskedGradientMagnitude()`: `cv::Sobel` `CV_64F` → `CV_32F`

### 3-2. `cpp/src/quality_metrics.cpp` — #11 toGray 중복 제거
- `measureLaplacianReductionImpl()`, `detectHaloImpl()` 구현 추가 (gray Mat을 매개변수로 받음)
- 기존 `measureLaplacianReduction()`, `detectHalo()`는 toGray 후 Impl 호출하는 래퍼로 변경
- `evaluateQuantitativeGate()` 내에서:
  ```cpp
  cv::Mat origGray = toGray(original);
  cv::Mat procGray = toGray(processed);
  // Impl 버전 직접 호출
  double laplacian = measureLaplacianReductionImpl(origGray, procGray, mask);
  double halo = detectHaloImpl(origGray, procGray, mask);
  ```

## Phase 4: Source — catch(...) Logging (12곳)

### 4-1. `cpp/src/quality_metrics.cpp` — 10곳
각 `catch (...)` 블록에 `fprintf(stderr, "[IrisSDK] ...")` 추가:
- 라인 61, 107, 163, 187, 209, 247, 295, 338, 389, 416
- 함수명 포함하여 디버깅 가능하도록

### 4-2. `cpp/src/ab_compare.cpp` — 1곳
- 라인 150 `catch(...)` 에 로그 추가

### 4-3. `cpp/src/param_tuner.cpp` — 1곳
- 라인 99 `catch(...)` 에 로그 추가

## Phase 5: Tests

### 5-1. `cpp/tests/test_quality_tuning.cpp` — 누락 테스트 추가 (~14건)

**ParamTuner 테스트** (3건):
- `RecommendPresetsReturnsValid`: recommendPresets() 반환값 검증
- `GenerateReportNonEmpty`: generateReport() 비어있지 않은 문자열 반환 검증
- `GridSearchStepsClamping`: steps=100 입력 시 조합 폭발 없이 정상 반환

**ABCompare 테스트** (2건):
- `ResetClearsResults`: reset() 후 results 비어있음 검증
- `AddResultMemoryCap`: 1001개 추가 시 1000개만 유지

**ReleaseGate 경계값 테스트** (6건):
- `FrameTimeExactBoundary`: frame_time_ms = 33.0 → GO
- `FrameTimeJustOver`: frame_time_ms = 33.01 → NO_GO
- `LaplacianLowerBound`: laplacian = 0.30 → GO
- `LaplacianUpperBound`: laplacian = 0.60 → GO
- `SSIMExactBoundary`: non_skin_ssim = 0.95 → NO_GO (> 0.95 필요)
- `SSIMJustAbove`: non_skin_ssim = 0.951 → GO

**QualityMetrics 테스트** (3건):
- `CV32FPrecisionAdequate`: SSIM 결과가 0~1 범위 내 합리적 값
- `ToGrayCaching`: evaluateQuantitativeGate 정상 동작 확인
- `CatchLogging`: 잘못된 입력 시 크래시 없이 기본값 반환

## Phase 6: Documentation Update

### 6-1. `.full-review/05-final-report.md`
- 수정 완료된 이슈에 해결 상태 표시

### 6-2. `docs/workPaper/P4-W3-06_quality_tools_bugfix.md`
- 변경 이력 업데이트, 상태를 완료로 변경

## Execution Order & Dependencies

```
Phase 1 (Headers) ─────────────┐
                                ├──→ Phase 2 (Safety) ──→ Phase 4 (catch logging)
                                ├──→ Phase 3 (Performance)
                                └──→ Phase 5 (Tests) — Phase 2,3 완료 후
Phase 6 (Docs) — 전체 완료 후
```

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| CV_32F 전환 시 정밀도 변화 | SSIM 기준 ~0.0001 이하 차이, 테스트로 검증 |
| Impl 패턴 API 호환성 | public 메서드는 동일 시그니처 유지 |
| catch 로그가 성능에 영향 | 예외 경로에서만 실행, 정상 경로 무영향 |
| steps clamp가 기존 테스트 깨뜨림 | 기존 테스트는 steps ≤ 10 사용 |

## Verification

1. `cd cpp/cmake-build-debug && cmake --build . --parallel`
2. `ctest` — 기존 31개 + 신규 ~14개 테스트 전체 통과
3. 컴파일 경고 0건 확인
