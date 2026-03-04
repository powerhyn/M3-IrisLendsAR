# Code Review Report: Frequency Separation GPU Pipeline

**Branch**: `feature/P4-W3-02`
**Date**: 2026-03-04
**Reviewer**: Claude Opus 4.6 (automated)
**Scope**: 10 files, +528 / -4 lines
**Feature**: Frequency Separation 기반 고급 피부 스무딩 GPU 파이프라인

---

## Executive Summary

Frequency Separation 파이프라인 구현은 구조적으로 잘 설계되어 있다. Bilateral fallback 전략, 셰이더 uniform 캐싱, 텍스처 풀 재사용, scissor state 관리 등 기존 코드베이스의 패턴을 충실히 따르고 있다. 그러나 GPU 상태 누수, 버퍼 크기 검증 부재, GLSL 동적 루프 성능 우려 등 수정이 필요한 항목들이 발견되었다.

| Severity | Count | Summary |
|----------|-------|---------|
| CRITICAL | 1 | 버퍼 크기 검증 없는 GPU 텍스처 업로드 |
| HIGH | 4 | GL state 누수, 셰이더 동적 분기 성능, 누락된 #if guard, glViewport 미설정 |
| MEDIUM | 6 | mapSkinQuality 정적 함수 일관성, 테스트 부족, 미사용 멤버, cross-platform 동기화 등 |
| LOW | 5 | 문서화, 매직 넘버, 코드 스타일 |

---

## CRITICAL Issues

### C-01: `uploadSkinMask`에서 combined_mask 버퍼 크기 검증 부재

**File**: `cpp/src/gpu/gpu_beauty_backend.cpp` (line ~883-912)
**Severity**: CRITICAL
**Category**: Security / Buffer Overflow

`uploadSkinMask()`가 `combined_mask.data()`를 `glTexImage2D`/`glTexSubImage2D`에 전달하면서, `combined_mask.size() >= mask_width * mask_height` 검증이 없다. `BeautyROI::combined_mask`는 외부에서 채워지는 벡터이므로, 크기가 불일치하면 **out-of-bounds read**가 발생하여 GPU 드라이버 크래시 또는 메모리 손상을 유발할 수 있다.

```cpp
// 현재 코드 (문제)
if (combined_mask.empty() || mask_width <= 0 || mask_height <= 0) {
    return 0;
}
// ...바로 glTexImage2D 호출

// 수정 제안
if (combined_mask.empty() || mask_width <= 0 || mask_height <= 0) {
    return 0;
}
const size_t expected_size = static_cast<size_t>(mask_width) * mask_height;
if (combined_mask.size() < expected_size) {
    LOGE("uploadSkinMask: buffer size mismatch (got %zu, expected %zu)",
         combined_mask.size(), expected_size);
    return 0;
}
```

---

## HIGH Issues

### H-01: `executeFreqSepPipeline` 내부에서 `glViewport` 미설정

**File**: `cpp/src/gpu/gpu_beauty_backend.cpp` (line ~981-1098)
**Severity**: HIGH
**Category**: GL State Correctness

`applyTextureId()`는 1279행에서 `glViewport(0, 0, width, height)`을 설정한 후 scissor를 활성화한다. FreqSep 파이프라인은 scissor를 비활성화한 뒤 중간 렌더 타겟에 그리지만, texture pool에서 획득한 텍스처의 크기가 요청한 `width x height`와 다를 가능성이 있다 (풀이 더 큰 텍스처를 반환하는 경우). 이 경우 `glViewport`가 중간 FBO 크기와 불일치하여 렌더링이 잘리거나 늘어날 수 있다.

```cpp
// 수정 제안: 파이프라인 시작 시 viewport 명시 설정
glViewport(0, 0, width, height);
```

현재 텍스처 풀이 정확히 동일 크기를 반환하는 것이 보장된다면 실제 문제가 아닐 수 있으나, 방어적으로 viewport를 명시 설정하는 것이 안전하다.

### H-02: Gaussian 셰이더의 dynamic loop (`uRadius` uniform)으로 인한 GPU 성능 문제

**File**: `cpp/src/gpu/shader_sources.cpp` (FREQ_SEP_GAUSSIAN_FRAGMENT)
**Severity**: HIGH
**Category**: Performance (GPU)

```glsl
for (int i = -uRadius; i <= uRadius; i++) {
```

GLSL ES 3.1에서 uniform 기반 동적 루프 범위는 **loop unrolling 불가**를 의미한다. `uRadius`가 28이면 루프 57회 반복이며, 각 반복마다 `texture()` + `exp()` 연산이 발생한다. 저사양 모바일 GPU(Adreno 3xx, Mali-T6xx)에서 33ms 프레임 버짓을 초과할 수 있다.

**권장 사항**:
1. 최대 radius에 대한 벤치마크를 수행하고, radius 28이 타겟 디바이스에서 프레임 드롭을 유발하는지 확인
2. 대안으로 2-pass downscale 전략 (1/2 해상도에서 blur 후 upscale) 검토
3. 또는 하드코딩된 radius별 셰이더 변형 (6, 14, 28)을 컴파일 타임에 생성

```glsl
// 대안: compile-time radius macro로 loop unrolling 활성화
#define RADIUS 15
for (int i = -RADIUS; i <= RADIUS; i++) {
```

### H-03: `executeSmoothingWithFallbackStrength`에 `#if IRIS_SDK_GPU_AVAILABLE` guard 누락

**File**: `cpp/src/gpu/gpu_beauty_backend.cpp` (line ~968-979)
**Severity**: HIGH
**Category**: Cross-platform Build

`executeSmoothingWithFallbackStrength()`는 `GLuint` 타입 파라미터를 사용하지만 `#if IRIS_SDK_GPU_AVAILABLE` guard가 없다. GPU 미지원 플랫폼 빌드에서 컴파일 에러 또는 링크 에러가 발생할 수 있다. 같은 파일의 다른 메서드들(`uploadSkinMask`, `executeFreqSepPipeline`)은 guard를 사용하고 있어 일관성이 깨진다.

```cpp
// 수정 제안
void GPUBeautyBackend::executeSmoothingWithFallbackStrength(
    GLuint input_tex, GLuint output_fbo,
    int width, int height,
    const BeautyFilterConfigV2& config) {
#if IRIS_SDK_GPU_AVAILABLE
    BeautyFilterConfigV2 fallback_config = config;
    float effective_smoothing = config.skinQuality * 0.5f;
    if (fallback_config.smoothing < effective_smoothing) {
        fallback_config.smoothing = effective_smoothing;
    }
    executeSmoothingPass(input_tex, output_fbo, width, height, fallback_config);
#else
    (void)input_tex; (void)output_fbo;
    (void)width; (void)height; (void)config;
#endif
}
```

### H-04: `executeFreqSepPipeline` 종료 시 GL_TEXTURE0이 아닌 active texture unit 상태 누수 가능성

**File**: `cpp/src/gpu/gpu_beauty_backend.cpp` (line ~1068-1082)
**Severity**: HIGH
**Category**: GL State Management

cleanup 코드가 texture unit 3에서 0까지 순서대로 바인딩 해제하고 마지막에 `glActiveTexture(GL_TEXTURE0)`을 호출하므로 이 부분은 올바르다. 그러나 **early return 경로** (render target 획득 실패 시)에서는 cleanup이 이루어지지 않은 채 return되므로, `glUseProgram()` 호출 후 early return 시 프로그램 상태가 남을 수 있다. 현재 코드에서 early return은 GL 호출 이전이므로 실제로는 안전하지만, 향후 유지보수 시 위험 요소이다.

```cpp
// 방어적 접근: RAII scope guard 패턴 고려
// 또는 early return 후 상태를 명시적으로 정리
```

---

## MEDIUM Issues

### M-01: `mapSkinQuality`가 public static이지만 FreqSepParams가 public인 이유 불명확

**File**: `cpp/include/iris_sdk/gpu/gpu_beauty_backend.h`
**Severity**: MEDIUM
**Category**: API Design / Encapsulation

`FreqSepParams`와 `mapSkinQuality()`가 public 접근 지정자 아래 선언되어 있다. 이는 내부 구현 세부사항이므로 private 영역으로 이동하거나, 외부 테스트를 위해 의도적으로 public인 경우 주석으로 그 이유를 명시해야 한다.

```cpp
// 수정 제안: private으로 이동하거나 테스트 의도 문서화
/// @internal 테스트 전용 — 외부 사용 금지
static FreqSepParams mapSkinQuality(float skin_quality, int face_width);
```

### M-02: `skin_radius_filter_` 멤버 변수 미사용

**File**: `cpp/include/iris_sdk/gpu/gpu_beauty_backend.h` (line ~430)
**Severity**: MEDIUM
**Category**: Dead Code / Incomplete Implementation

`OneEuroFilter skin_radius_filter_{0.5f, 0.01f, 1.0f};` 가 선언되어 있으나, 이번 diff 어디에서도 사용되지 않는다. 주석에 "P4-W3-04에서 사용"이라 되어 있지만, 미사용 멤버를 미리 추가하는 것은 YAGNI 원칙에 위배되며 불필요한 헤더 의존성(`one_euro_filter.h`)을 도입한다.

```cpp
// 현재: 사용되지 않는 멤버 + 불필요한 include
#include "iris_sdk/one_euro_filter.h"
OneEuroFilter skin_radius_filter_{0.5f, 0.01f, 1.0f};

// 권장: P4-W3-04 구현 시 함께 추가
// 또는 유지할 경우 주석에 이유 명시
```

### M-03: Gaussian 셰이더의 sigma 계산에서 매직 넘버

**File**: `cpp/src/gpu/shader_sources.cpp` (FREQ_SEP_GAUSSIAN_FRAGMENT)
**Severity**: MEDIUM
**Category**: Maintainability

```glsl
float sigma = float(uRadius) * 0.4;
```

`0.4` 매직 넘버는 Gaussian 분포의 유효 범위를 결정하는 중요한 파라미터이다. 일반적으로 sigma = radius / 3.0 (3-sigma rule)이 사용되는데, 여기서는 radius * 0.4로 설정하여 더 좁은 가중치 분포를 만든다. 이는 의도적인 선택일 수 있으나, 근거를 주석으로 남기는 것이 좋다.

```glsl
// 수정 제안: 의도 문서화
// sigma = radius * 0.4 → ~2.5-sigma window
// 3-sigma rule (0.33) 대비 약간 넓은 분포로 edge ringing 최소화
float sigma = float(uRadius) * 0.4;
```

### M-04: 테스트 커버리지 부족

**File**: `cpp/tests/test_beauty_config_v2.cpp` (+1 line)
**Severity**: MEDIUM
**Category**: Testing

새로 추가된 기능 중 테스트 커버리지가 매우 부족하다:

1. `mapSkinQuality()` 함수의 경계값 테스트 없음 (0.0, 0.5, 1.0, 음수, >1.0)
2. `FreqSepParams` 결과값 검증 없음
3. `uploadSkinMask()` 에러 경로 테스트 없음
4. skinQuality와 smoothing 간 상호작용 (fallback 로직) 테스트 없음
5. `isValid()` / `clamp()`에 skinQuality 추가에 대한 테스트 없음

기존 테스트에 default value assertion 1줄만 추가된 상태이다. `mapSkinQuality`는 static public이므로 단위 테스트가 용이하다.

```cpp
// 추가 필요한 테스트 예시
TEST(FreqSepParamsTest, DisabledWhenSkinQualityZero) {
    auto params = GPUBeautyBackend::mapSkinQuality(0.0f, 200);
    EXPECT_FALSE(params.enabled);
}

TEST(FreqSepParamsTest, EnabledWhenSkinQualityPositive) {
    auto params = GPUBeautyBackend::mapSkinQuality(0.5f, 200);
    EXPECT_TRUE(params.enabled);
    EXPECT_GE(params.blur_radius, 6);
    EXPECT_LE(params.blur_radius, 28);
}

TEST(FreqSepParamsTest, RadiusClampedToFaceWidth) {
    auto params_small = GPUBeautyBackend::mapSkinQuality(0.5f, 50);
    EXPECT_EQ(params_small.blur_radius, 6);  // 50 * 0.05 = 2.5 → clamped to 6

    auto params_large = GPUBeautyBackend::mapSkinQuality(0.5f, 800);
    EXPECT_EQ(params_large.blur_radius, 28); // 800 * 0.05 = 40 → clamped to 28
}
```

### M-05: iOS / Flutter / Web 바인딩에 `skinQuality` 필드 미추가

**File**: (bindings/ios, bindings/flutter, bindings/web)
**Severity**: MEDIUM
**Category**: Cross-platform API Consistency

Android JNI 바인딩에는 `skinQuality` 매핑이 완료되었으나, iOS (Obj-C++ bridge), Flutter (dart:ffi), Web (WASM) 바인딩에는 추가되지 않았다. 이 바인딩 파일들은 아직 프로젝트에 존재하지 않을 수 있지만(확인 결과 현재 없음), 향후 생성 시 `skinQuality` 누락 위험이 있다.

**권장**: 작업 문서(P4-W3-02)에 "iOS/Flutter/Web 바인딩 skinQuality 추가 필요" 항목을 기록해 두기.

### M-06: `executeFreqSepPipeline`에서 GL 에러 체크 부재

**File**: `cpp/src/gpu/gpu_beauty_backend.cpp` (line ~981-1098)
**Severity**: MEDIUM
**Category**: Error Handling / Debugging

기존 `executeCombinedColorPass`는 FBO completeness 체크(`glCheckFramebufferStatus`)와 디버그 빌드에서의 `glGetError()` 호출이 있으나, `executeFreqSepPipeline`에는 이 검증이 전혀 없다. 5-subpass 파이프라인이므로 중간 단계에서 발생하는 GL 에러를 진단하기 어렵다.

```cpp
// 수정 제안: 최소한 디버그 빌드에서 에러 체크 추가
#ifndef NDEBUG
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        LOGE("FreqSep pass 1a: GL error 0x%x", err);
    }
#endif
```

---

## LOW Issues

### L-01: `FreqSepParams` 구조체의 기본값에 대한 Doxygen 문서 부족

**File**: `cpp/include/iris_sdk/gpu/gpu_beauty_backend.h`
**Severity**: LOW
**Category**: Documentation

`FreqSepParams` 각 필드의 의미와 유효 범위가 문서화되어 있지 않다. 특히 `attenuation_low`/`attenuation_high`는 smoothstep의 edge 파라미터로, 값의 의미가 비직관적이다.

```cpp
struct FreqSepParams {
    int blur_radius = 15;              ///< Gaussian 블러 반경 (6~28, face_width의 5%)
    float high_freq_preserve = 0.45f;  ///< 고주파 보존율 (0.1~1.0, 낮을수록 스무딩 강함)
    float low_freq_smooth_radius_ratio = 0.5f; ///< 2차 blur radius = blur_radius * ratio
    float attenuation_low = 0.02f;     ///< blemish smoothstep 하한 (luminance 차이)
    float attenuation_high = 0.15f;    ///< blemish smoothstep 상한 (luminance 차이)
    bool enabled = false;              ///< Freq Sep 활성화 여부
};
```

### L-02: Composite 셰이더에서 `uSmoothedLow` 네이밍 혼동

**File**: `cpp/src/gpu/shader_sources.cpp` (FREQ_SEP_COMPOSITE_FRAGMENT)
**Severity**: LOW
**Category**: Readability

`uSmoothedLow` (Pass 2b result)와 `uLowFreq` (Pass 1b result)의 이름만으로 둘의 차이를 이해하기 어렵다. 주석이 코드에 있지만 셰이더 내부에서는 보이지 않는다.

```glsl
// 현재 이름:        제안 이름:
// uSmoothedLow  →  uAdditionalBlurred    (2차 blur 결과)
// uLowFreq      →  uBaselineLow          (1차 blur = high-freq 추출 기준)
```

### L-03: `mapSkinQuality`에서 `face_width == 0` 처리 불명확

**File**: `cpp/src/gpu/gpu_beauty_backend.cpp` (line ~936-965)
**Severity**: LOW
**Category**: Edge Case

`face_width`가 0인 경우(ROI 없이 호출 시) `blur_radius = clamp(0 * 0.05, 6, 28) = 6`으로 최소값이 적용된다. 이는 clamp 덕분에 정상 동작하지만, 명시적으로 face_width <= 0일 때의 의도를 문서화하는 것이 좋다.

### L-04: Java Builder에 `skinQuality` 범위 검증 누락

**File**: `android/.../BeautyFilterConfigV2.java` (Builder 클래스)
**Severity**: LOW
**Category**: API Robustness

`Builder.skinQuality(float)` 메서드에 범위 체크가 없다. 다른 Builder 메서드들도 동일하게 체크가 없으므로 기존 패턴과 일관성은 유지되지만, `clamp()`나 `isValid()` 호출 전에 잘못된 값이 설정될 수 있다.

### L-05: 셰이더 문자열의 `#version 310 es` 일관성

**File**: `cpp/src/gpu/shader_sources.cpp`
**Severity**: LOW
**Category**: Consistency

새 셰이더 2개 모두 `#version 310 es`를 명시하고 있으며, 기존 셰이더들과 동일하다. 일관성 확인 완료.

---

## Positive Observations

이번 구현에서 잘 된 점들을 기록한다.

### P-01: Graceful Degradation 설계

FreqSep 셰이더 컴파일 실패, 텍스처 풀 부족, skin mask 업로드 실패, ROI 미제공 등 모든 실패 경로에서 Bilateral fallback이 작동하도록 설계되었다. 이는 프로덕션 안정성에 매우 중요하다.

### P-02: Scissor State 관리

FreqSep의 multipass Gaussian이 전체 프레임 텍스처에서 작동해야 하므로 ROI scissor를 비활성화하고, 파이프라인 완료 후 복원하는 로직이 올바르게 구현되었다. 최종 마스킹은 Composite 셰이더의 `uSkinMask`가 담당한다.

### P-03: Texture Pool 활용

3개의 중간 렌더 타겟을 TexturePool에서 획득/반환하여 GPU 메모리 할당 오버헤드를 제거했다. 실패 시 모든 타겟을 반환하는 cleanup도 정확하다.

### P-04: GL_UNPACK_ALIGNMENT 처리

single-channel (GL_R8) 텍스처 업로드 시 `glPixelStorei(GL_UNPACK_ALIGNMENT, 1)`을 설정하고 업로드 후 4로 복원하는 것은 올바른 패턴이다.

### P-05: Cross-layer API 일관성 (C++ / C API / JNI / Java)

`skinQuality` 필드가 모든 Android 바인딩 레이어에 걸쳐 일관되게 추가되었다: C++ struct, C API struct, toCppConfigV2/fromCppConfigV2, default 함수, JNI field cache, JNI read/write, Java field + Builder + validation + clamp + toString.

### P-06: Non-linear Attenuation 셰이더 설계

Composite 셰이더의 blemish detection (smoothstep 기반 magnitude 분석)은 피부 텍스처(미세)는 보존하고 잡티(큰 변화)는 감쇠하는 frequency-domain 접근법으로 기술적으로 건전하다.

---

## Summary and Prioritized Action Items

### Must Fix (Before Merge)

| ID | Issue | Effort |
|----|-------|--------|
| C-01 | uploadSkinMask 버퍼 크기 검증 추가 | 5 min |
| H-03 | executeSmoothingWithFallbackStrength에 #if guard 추가 | 3 min |

### Should Fix (Before Release)

| ID | Issue | Effort |
|----|-------|--------|
| H-01 | executeFreqSepPipeline 시작 시 glViewport 명시 설정 | 2 min |
| H-02 | Gaussian shader dynamic loop 성능 벤치마크 실시 | 1-2 hr |
| H-04 | GL state cleanup에 대한 방어적 처리 검토 | 15 min |
| M-04 | mapSkinQuality 및 FreqSep 경로 단위 테스트 작성 | 30 min |
| M-06 | 디버그 빌드 GL error 체크 추가 | 10 min |

### Nice to Have

| ID | Issue | Effort |
|----|-------|--------|
| M-01 | FreqSepParams/mapSkinQuality를 private으로 이동 또는 문서화 | 5 min |
| M-02 | skin_radius_filter_ 미사용 멤버 제거 | 3 min |
| M-03 | 셰이더 sigma 계산 매직 넘버 문서화 | 3 min |
| M-05 | iOS/Flutter/Web 바인딩 추가 필요 사항 작업 문서에 기록 | 5 min |
| L-01 | FreqSepParams Doxygen 문서화 | 5 min |

---

## Appendix: File-by-File Change Summary

### cpp/src/gpu/shader_sources.cpp (+80 lines)
- `FREQ_SEP_GAUSSIAN_FRAGMENT`: Separable 1D Gaussian blur (dynamic radius)
- `FREQ_SEP_COMPOSITE_FRAGMENT`: High-freq extraction + non-linear attenuation + mask blending
- **Verdict**: GLSL 로직 정확, 성능 우려(H-02) 있음

### cpp/include/iris_sdk/gpu/gpu_beauty_backend.h (+72 lines)
- `FreqSepParams` 구조체
- 5개 메서드 선언 (initializeFreqSepShaders, executeFreqSepPipeline, uploadSkinMask, executeSmoothingWithFallbackStrength, mapSkinQuality)
- Uniform 캐시 구조체 2개
- GL 리소스 멤버 3개 (program x2, texture x1, dimension x2)
- OneEuroFilter 멤버 1개 (미사용)
- **Verdict**: 구조 양호, 접근 지정자(M-01)와 미사용 멤버(M-02) 정리 필요

### cpp/src/gpu/gpu_beauty_backend.cpp (+344 lines)
- `initializeFreqSepShaders()`: 셰이더 컴파일 + 캐시
- `cacheUniformLocations()`: FreqSep uniform location 캐싱
- `release()`: 리소스 해제 (skin_mask_texture, program handles)
- `uploadSkinMask()`: GL_R8 single-channel 텍스처 업로드 (C-01 이슈)
- `mapSkinQuality()`: S-curve 매핑 함수
- `executeSmoothingWithFallbackStrength()`: Bilateral fallback (H-03 이슈)
- `executeFreqSepPipeline()`: 5-subpass 파이프라인 (H-01, H-04, M-06 이슈)
- `applyTextureId()` 수정: FreqSep/Bilateral 분기 로직
- **Verdict**: 핵심 구현, 가장 많은 이슈가 집중된 파일

### cpp/include/iris_sdk/beauty_filter.h (+6 lines)
- `skinQuality` 필드 추가 (struct, defaults, isValid, clamp, fromV1, reset)
- **Verdict**: 완벽, 모든 유틸리티 함수에 일관되게 반영

### cpp/include/iris_sdk/sdk_api.h (+1 line)
- `skin_quality` C API 필드
- **Verdict**: OK

### cpp/src/sdk_api_v2.cpp (+3 lines)
- C++ <-> C 변환 양방향 + default 함수
- **Verdict**: OK

### android/.../iris_jni.cpp (+6 lines)
- JNI field cache + validation + read/write
- **Verdict**: OK

### android/.../jni_utils.h (+1 line)
- JNI field ID 선언
- **Verdict**: OK

### android/.../BeautyFilterConfigV2.java (+18 lines)
- Java field + constant + copy ctor + reset + validate + clamp + toString + Builder
- **Verdict**: OK, L-04 참고

### cpp/tests/test_beauty_config_v2.cpp (+1 line)
- Default value assertion
- **Verdict**: 최소한의 테스트만 추가됨 (M-04 참고)
