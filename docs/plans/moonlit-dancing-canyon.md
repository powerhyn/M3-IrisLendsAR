# P4-W3-04-R1: FreqSep 파이프라인 중복 코드 리팩터링

## Context

P4-W3-04에서 MID tier 하이브리드 해상도 파이프라인(`executeFreqSepPipelineHalfRes`)을 구현하면서
기존 `executeFreqSepPipeline`과 ~140줄의 구조적 중복이 발생했다.
두 함수는 5-subpass 구조(GaussianH → GaussianV → LowSmoothH → LowSmoothV → Composite)가 동일하며,
차이점은 **블러 해상도, viewport 관리, GL_LINEAR 필터 설정** 3가지뿐이다.

## 변경 대상 파일

| 파일 | 변경 내용 |
|------|---------|
| `cpp/include/iris_sdk/gpu/gpu_beauty_backend.h` | `FreqSepExecConfig` 구조체 + `executeFreqSepPipelineImpl()` 선언 |
| `cpp/src/gpu/gpu_beauty_backend.cpp` | 공통 구현 추출, 기존 2함수를 래퍼로 변환 |

## 구현 계획

### 1. `FreqSepExecConfig` 구조체 추가 (헤더)

```cpp
/// FreqSep 파이프라인 실행 설정 (full-res / half-res 분기 매개변수화)
struct FreqSepExecConfig {
    int res_divisor;               // 1 = full-res, 2 = half-res
    bool linear_upsample;          // true: composite 입력에 GL_LINEAR 설정
    const char* profiler_suffix;   // "" 또는 "_Half"
};
```

### 2. `executeFreqSepPipelineImpl()` private 메서드 추가 (헤더)

```cpp
bool executeFreqSepPipelineImpl(
    GLuint input_tex, GLuint mask_tex, GLuint output_fbo,
    int width, int height,
    const FreqSepParams& params,
    const FreqSepExecConfig& exec_cfg);
```

### 3. 기존 함수를 래퍼로 변환 (cpp)

```cpp
bool GPUBeautyBackend::executeFreqSepPipeline(...) {
    return executeFreqSepPipelineImpl(input_tex, mask_tex, output_fbo,
        width, height, params, {1, false, ""});
}

bool GPUBeautyBackend::executeFreqSepPipelineHalfRes(...) {
    return executeFreqSepPipelineImpl(input_tex, mask_tex, output_fbo,
        width, height, params, {2, true, "_Half"});
}
```

### 4. `executeFreqSepPipelineImpl()` 구현 핵심 로직

```
res_divisor 기반 계산:
  blur_w = width / res_divisor
  blur_h = height / res_divisor
  blur_radius = (res_divisor == 1) ? params.blur_radius : max(3, params.blur_radius / 2)

텍스처 할당: acquireRenderTarget(blur_w, blur_h) × 3

viewport 관리:
  if (res_divisor > 1) glViewport(0, 0, blur_w, blur_h)  // blur passes
  Composite 전: if (res_divisor > 1) glViewport(0, 0, width, height)  // 복원

GL_LINEAR:
  if (exec_cfg.linear_upsample) → composite 입력 텍스처에 GL_LINEAR 설정

profiler 태그:
  "FreqSep_GaussianH" + exec_cfg.profiler_suffix
```

### 5. 기존 함수 본문 삭제

`executeFreqSepPipeline()` (125줄) → 3줄 래퍼
`executeFreqSepPipelineHalfRes()` (143줄) → 3줄 래퍼

예상 순감소: ~120줄

## 검증

1. `cd cpp/cmake-build-debug && cmake --build . --parallel` — 컴파일 성공
2. `./bin/test_gpu_beauty_backend` — 기존 테스트 통과
3. `./bin/test_device_tier` — FreqSep 매핑 테스트 통과
4. `./bin/test_one_euro_filter` — OEF 테스트 통과
5. 작업 문서 `docs/workPaper/P4-W3-04-R1_freqsep_pipeline_refactor.md` 상태 업데이트
