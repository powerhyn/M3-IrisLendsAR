# Security Audit Report: P4-W3-04 Temporal Stability + Device Tier

**Audit Date**: 2026-03-04
**Auditor**: Security Audit Agent (claude-opus-4-6)
**Scope**: `gpu_beauty_backend.h` / `gpu_beauty_backend.cpp` -- P4-W3-04 변경사항
**Commit**: 9fd5bdc (feature/P4-W3-04)
**Severity Scale**: Critical / High / Medium / Low

---

## Executive Summary

P4-W3-04 커밋에서 3개 주요 기능이 추가되었다: One Euro Filter 기반 temporal stability, GPU 렌더러 문자열 파싱을 통한 DeviceTier 판정, MID tier 하프 해상도 FreqSep 파이프라인. 전체적으로 방어적 코딩이 적용되어 있으나, **`std::stoi` 예외 미처리 (High)**, **static 메서드의 GL 컨텍스트 암묵적 의존 (Medium)**, **One Euro Filter 스테이트 미리셋 (Medium)** 등 주의가 필요한 항목이 식별되었다.

| Severity | Count |
|----------|-------|
| Critical | 0 |
| High | 2 |
| Medium | 5 |
| Low | 3 |
| **Total** | **10** |

---

## Findings

### [H-01] `std::stoi` 예외 미처리 -- SDK 초기화 경로 크래시

**Severity**: High
**File**: `cpp/src/gpu/gpu_beauty_backend.cpp` lines 1173, 1188
**Category**: Exception Safety

**Description**:
`detectDeviceTier()` 내부에서 GPU 렌더러 문자열에서 추출한 숫자 문자열을 `std::stoi(digits)`로 변환한다. `digits` 문자열이 `int` 범위를 초과하거나 예상치 못한 형식일 경우 `std::out_of_range` 또는 `std::invalid_argument` 예외가 발생한다.

이 함수는 `initialize()`에서 호출되며 (line 162), `initialize()`는 try-catch로 감싸져 있지 않다. 결과적으로 예외가 호출 스택을 타고 전파되어 **전체 SDK 초기화가 실패하거나 앱이 크래시**한다.

`digits` 변수는 `std::isdigit` 검증을 통과한 문자만 포함하므로 `std::invalid_argument` 가능성은 낮지만, 극단적 렌더러 문자열(예: "Adreno 99999999999")에서 `std::out_of_range`가 발생할 수 있다.

**Code**:
```cpp
// Line 1173
int num = std::stoi(digits);  // throws on overflow

// Line 1188
int num = std::stoi(digits);  // throws on overflow
```

**Remediation**:
```cpp
// Option A: 범위 검사 후 변환
if (digits.size() <= 5) {  // 5자리 이하만 허용
    int num = std::stoi(digits);
    // ...
}

// Option B: 예외 포착
try {
    int num = std::stoi(digits);
    // ...
} catch (const std::exception&) {
    // fallback to LOW or continue to next check
}

// Option C (권장): strtol 사용으로 예외 회피
long val = std::strtol(digits.c_str(), nullptr, 10);
if (val > 0 && val <= 99999) {
    int num = static_cast<int>(val);
    // ...
}
```

---

### [H-02] `detectDeviceTier()` static 메서드의 GL 컨텍스트 암묵적 의존

**Severity**: High
**File**: `cpp/src/gpu/gpu_beauty_backend.cpp` line 1157
**Category**: API Safety / Undefined Behavior

**Description**:
`detectDeviceTier()`는 `static` 메서드로 선언되어 있어, 인스턴스 없이 `GPUBeautyBackend::detectDeviceTier()`로 호출할 수 있다. 그러나 내부에서 `glGetString(GL_RENDERER)`를 호출하므로 **유효한 GL 컨텍스트가 현재 스레드에 바인딩되어 있어야** 한다. GL 컨텍스트 없이 호출하면 `glGetString`이 `nullptr`을 반환하거나 GPU 드라이버에서 정의되지 않은 동작이 발생한다.

현재 `initialize()` 내부에서만 호출되며 GL 컨텍스트 활성화 이후에 위치하므로 실제 크래시 위험은 낮다. 그러나 public static API이므로 외부에서 잘못된 시점에 호출될 가능성이 존재한다.

**Code**:
```cpp
// Header line 249 -- public static
static DeviceTier detectDeviceTier();

// Implementation line 1157
const char* renderer = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
if (!renderer) return DeviceTier::LOW;  // nullptr 체크는 있으나 GL error 미확인
```

**Remediation**:
1. `static`을 제거하고 인스턴스 메서드 + `private`로 변경하여 외부 호출을 차단한다.
2. 또는 함수 시작에 GL 컨텍스트 유효성 검사를 추가한다:
```cpp
DeviceTier GPUBeautyBackend::detectDeviceTier() {
#if IRIS_SDK_GPU_AVAILABLE
    // GL error state 클리어 후 호출
    glGetError();
    const char* renderer = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
    GLenum err = glGetError();
    if (!renderer || err != GL_NO_ERROR) {
        LOGW("detectDeviceTier: No valid GL context, defaulting to LOW");
        return DeviceTier::LOW;
    }
    // ...
```

---

### [M-01] One Euro Filter 스테이트가 `release()`/`initialize()` 사이클에서 리셋되지 않음

**Severity**: Medium
**File**: `cpp/include/iris_sdk/gpu/gpu_beauty_backend.h` lines 446-448, `cpp/src/gpu/gpu_beauty_backend.cpp` `release()` (line 399)
**Category**: Resource Management / Stale State

**Description**:
`skin_radius_filter_`, `mask_center_x_filter_`, `mask_center_y_filter_` 세 개의 OneEuroFilter는 in-class 초기화로 생성되지만, `release()` 또는 `initialize()`에서 `.reset()`이 호출되지 않는다.

SDK가 `release()` 후 `initialize()`로 재초기화될 경우, 이전 세션의 필터 상태(last_time_, prev_value_)가 남아 있어 첫 프레임에서 비정상적인 temporal 보간이 발생할 수 있다. 예를 들어 이전 세션의 마지막 얼굴 위치가 새 세션의 첫 프레임 얼굴 위치와 크게 다르면, 필터가 큰 delta를 처리하게 되어 ROI 좌표가 순간적으로 왜곡된다.

**Remediation**:
`release()` 또는 `initialize()` 시작부에 필터 리셋을 추가한다:
```cpp
// In release() or at the start of initialize():
skin_radius_filter_.reset();
mask_center_x_filter_.reset();
mask_center_y_filter_.reset();
```

---

### [M-02] `device_tier_`가 `release()` 후 리셋되지 않음

**Severity**: Medium
**File**: `cpp/src/gpu/gpu_beauty_backend.cpp` `release()` function, `cpp/include/iris_sdk/gpu/gpu_beauty_backend.h` line 451
**Category**: Stale State

**Description**:
`device_tier_`는 `initialize()`에서 설정되지만 `release()`에서 초기값으로 리셋되지 않는다. `release()` 후 `initialized_ = false`가 되므로 `applyTextureId`가 호출되면 `IRIS_SDK_ERROR_NOT_INITIALIZED`를 반환하여 직접적 위험은 낮다. 그러나 방어적 코딩 관점에서, `device_tier_` 기본값이 `HIGH`(line 451)이므로 재초기화 전에 부분적으로 접근되는 경로가 있다면 잘못된 tier로 동작할 수 있다.

**Remediation**:
`release()`에 `device_tier_ = DeviceTier::HIGH;` (또는 `LOW` -- 안전한 기본값) 리셋을 추가한다.

---

### [M-03] ROI 포인터를 통한 `face_rect` 직접 수정 -- 사이드 이펙트

**Severity**: Medium
**File**: `cpp/src/gpu/gpu_beauty_backend.cpp` lines 1631-1632
**Category**: API Safety / Side Effect

**Description**:
Temporal stability 로직에서 `roi_ptr->face_rect.x`와 `roi_ptr->face_rect.y`를 필터링된 delta만큼 직접 수정한다:
```cpp
roi_ptr->face_rect.x += dx;
roi_ptr->face_rect.y += dy;
```

`roi_ptr`은 지역 변수 `&roi`를 가리키므로 현재 코드에서는 외부 데이터를 오염시키지 않는다. 그러나 이 패턴은 다음 문제를 내포한다:

1. **Scissor 좌표와 불일치**: Scissor 설정(line 1563-1584)이 temporal filtering 이전에 수행되므로, 필터링 후의 face_rect와 scissor 영역이 불일치한다. Temporal filtering이 face_rect를 이동시키면 skin mask의 중심은 이동하지만 scissor 영역은 원래 위치에 고정되어, ROI 경계에서 필터링 아티팩트가 발생할 수 있다.
2. **upstream에서 참조 전달로 변경 시 파괴적**: 향후 `roi`가 참조로 전달되면 caller의 데이터가 의도치 않게 수정된다.

**Remediation**:
- Temporal-filtered 좌표를 별도 변수에 저장하고 skin mask 업로드에만 사용하거나,
- Scissor 설정을 temporal filtering 이후로 이동시킨다.

---

### [M-04] `executeFreqSepPipelineHalfRes` -- viewport 미복원 시 후속 패스 영향

**Severity**: Medium
**File**: `cpp/src/gpu/gpu_beauty_backend.cpp` lines 1259, 1303
**Category**: GPU State Corruption

**Description**:
`executeFreqSepPipelineHalfRes()`는 line 1259에서 `glViewport(0, 0, half_w, half_h)`로 하프 해상도 viewport를 설정하고, line 1303에서 `glViewport(0, 0, width, height)`로 full-res를 복원한다.

**정상 경로에서는 문제 없다.** 그러나 Pass 1b~Pass 2b 사이에서 텍스처 할당 실패 등으로 early return이 추가될 경우, viewport가 half-res로 남게 되어 후속 렌더링 패스(CombinedColor, SoftFocus)가 잘못된 해상도로 실행된다.

현재 코드에는 중간 early return이 없으므로 실제 발생하지 않지만, 유지보수 과정에서 발생할 수 있는 잠재적 위험이다.

**Remediation**:
RAII 패턴의 viewport 가드를 사용하거나, 함수 끝에 viewport 복원을 보장하는 구조로 변경한다:
```cpp
// RAII viewport guard
struct ViewportGuard {
    int w, h;
    ViewportGuard(int w, int h) : w(w), h(h) {}
    ~ViewportGuard() { glViewport(0, 0, w, h); }
};
// Usage in executeFreqSepPipelineHalfRes:
ViewportGuard vg(width, height);
glViewport(0, 0, half_w, half_h);
```

---

### [M-05] One Euro Filter의 thread safety 미보장

**Severity**: Medium
**File**: `cpp/include/iris_sdk/gpu/gpu_beauty_backend.h` lines 446-448
**Category**: Concurrency

**Description**:
`OneEuroFilter::filter()` 호출은 `applyTextureId()` 내부에서 `mutex_` lock 하에 실행되므로, 현재 코드 경로에서는 thread-safe하다.

그러나 One Euro Filter 멤버는 `private`이 아닌 GPUBeautyBackend의 멤버이고, 헤더의 주석(line 49)에 "Thread-safe: 모든 public 메서드는 mutex로 보호됩니다"라고 명시되어 있다. **Public API만 mutex로 보호되므로**, 향후 internal 메서드에서 lock 없이 필터에 접근하는 코드가 추가되면 data race가 발생할 수 있다.

또한 `OneEuroFilter` 자체에는 동기화 메커니즘이 없으며 `steady_clock::now()` 기반 타임스탬프를 내부에 저장하므로, 멀티스레드 접근 시 상태가 손상된다.

**Remediation**:
현재 구조에서는 즉각적 조치 불필요. 다만 코드 주석으로 다음을 명시할 것:
```cpp
// NOTE: OneEuroFilter members are NOT thread-safe.
// Must be accessed only under mutex_ lock (currently guaranteed by
// applyTextureId's lock_guard).
```

---

### [L-01] GPU 렌더러 문자열 파싱의 취약한 휴리스틱

**Severity**: Low
**File**: `cpp/src/gpu/gpu_beauty_backend.cpp` lines 1162-1207
**Category**: Robustness

**Description**:
`detectDeviceTier()`의 GPU 분류 로직은 문자열 패턴 매칭에 의존한다. 다음 케이스에서 오분류가 발생할 수 있다:

1. **Mali-G710 이상의 3자리 모델**: `Mali-G720`은 `num=720 >= 710`으로 HIGH 판정 (정확). 그러나 `Mali-G78`은 `num=78 >= 70`으로 MID 판정되는데, G78은 사실상 HIGH급이다.
2. **새 GPU 브랜드/모델**: Samsung Xclipse, MediaTek HyperEngine 등 신규 GPU는 `LOW`로 분류된다.
3. **에뮬레이터 GPU**: "Android Emulator" 또는 "SwiftShader" 등은 `LOW`로 분류되어 개발 중 테스트에 영향을 줄 수 있다.

보안 관점에서는 GPU renderer 문자열은 드라이버가 제공하는 값이므로 조작 가능성이 있으나, 이 SDK의 위협 모델에서는 DoS나 기능 저하 정도에 그치므로 Low로 분류한다.

**Remediation**:
- Mali-G78/G79 등을 HIGH에 포함하는 추가 분기를 고려한다.
- 알 수 없는 GPU에 대해 `LOW` 대신 `MID`(보수적 성능, 합리적 품질)를 기본값으로 사용하는 것을 검토한다.
- 에뮬레이터 감지를 추가하고 디버그 빌드에서 tier override API를 제공한다.

---

### [L-02] `half_w` / `half_h` 정수 나눗셈의 홀수 해상도 처리

**Severity**: Low
**File**: `cpp/src/gpu/gpu_beauty_backend.cpp` line 1221-1222
**Category**: Input Validation

**Description**:
```cpp
int half_w = width / 2;
int half_h = height / 2;
```
홀수 해상도(예: 1921x1081)에서 `half_w = 960`, `half_h = 540`이 되어 원본과 정확히 2:1 매핑이 아닌 경우가 발생한다. Composite 패스에서 bilinear upsampling 시 1픽셀 오프셋이 발생할 수 있으나, 시각적 영향은 미미하다.

`width = 1` 또는 `height = 1`인 경우 `half_w = 0` 또는 `half_h = 0`이 되지만, 직후 line 1225의 체크(`if (half_w < 1 || half_h < 1)`)가 이를 방어한다.

**Remediation**:
현재 방어 코드로 충분하다. 필요시 `(width + 1) / 2` 라운드업을 적용할 수 있다.

---

### [L-03] LOGD/LOGW 매크로의 non-GPU 빌드에서 format string 안전성

**Severity**: Low
**File**: `cpp/src/gpu/gpu_beauty_backend.cpp` lines 23-27
**Category**: Code Quality

**Description**:
Desktop 스텁 빌드에서:
```cpp
#define LOGD(...) printf("[GPUBeautyBackend DEBUG] " __VA_ARGS__); printf("\n")
```
이 매크로는 `printf`에 직접 format string을 전달한다. 만약 `__VA_ARGS__`의 첫 인자가 사용자 제어 가능한 문자열이라면 format string vulnerability가 발생할 수 있다. 현재 코드에서는 모든 호출이 상수 리터럴 format string을 사용하므로 실제 취약점은 없다.

또한 매크로가 `do { } while(0)`으로 감싸져 있지 않아, `if-else` 문 내에서 사용 시 dangling-else 문제가 발생할 수 있다:
```cpp
if (condition)
    LOGD("test");  // expands to two statements!
else
    // ... misattached to second printf
```

**Remediation**:
```cpp
#define LOGD(...) do { printf("[GPUBeautyBackend DEBUG] " __VA_ARGS__); printf("\n"); } while(0)
```

---

## Phase 1 Context -- 후속 확인 결과

| Phase 1 이슈 | 상태 | 상세 |
|---|---|---|
| `std::stoi` 예외 | **미해결 [H-01]** | 여전히 try-catch 또는 안전한 변환 없이 사용 |
| `detectDeviceTier()` 호출 위치 | **확인됨** | `initialize()` 내부에서만 1회 호출 (line 162). GL 컨텍스트 활성화 이후 |
| ROI 포인터 직접 수정 | **확인됨 [M-03]** | 지역 변수 기반이라 외부 오염 없음. Scissor 타이밍 불일치 가능성 존재 |
| static 메서드 GL 의존 | **확인됨 [H-02]** | nullptr 체크는 있으나 public static이라 오용 가능 |

---

## Positive Observations

감사 과정에서 다음의 양호한 보안/안정성 패턴이 확인되었다:

1. **방어적 null 체크**: `glGetString` 반환값 nullptr 체크 (line 1158)
2. **텍스처 할당 실패 방어**: `executeFreqSepPipelineHalfRes`에서 3개 렌더타겟 모두 실패 시 정리 및 false 반환 (lines 1235-1241)
3. **최소 해상도 보장**: `half_w < 1 || half_h < 1` 체크로 0 해상도 방지 (line 1225)
4. **blur_radius 하한값**: `std::max(3, ...)` 적용 (lines 1246, 1619)
5. **Mutex 일관성**: 모든 public 메서드에 `lock_guard<mutex>` 적용
6. **Copy/Move 금지**: Rule of Five에 따라 복사/이동 연산자 삭제
7. **GPU 리소스 정리**: `release()`에서 텍스처, VAO/VBO, 펜스, 셰이더 등 체계적 해제
8. **Scissor 교집합 기반 ROI**: 프레임 경계를 넘는 ROI에 대한 안전한 처리 (lines 1568-1575)
9. **Viewport 복원**: HalfRes 파이프라인에서 full-res viewport 명시적 복원 (line 1303)

---

## Remediation Priority

| Priority | Finding | Effort |
|----------|---------|--------|
| 1 (즉시) | [H-01] std::stoi 예외 처리 | 15분 -- strtol 또는 try-catch 추가 |
| 2 (즉시) | [H-02] detectDeviceTier() 접근 제한 | 5분 -- static 제거 + private 변경 |
| 3 (다음 릴리스) | [M-01] One Euro Filter reset | 5분 -- release()에 reset() 추가 |
| 4 (다음 릴리스) | [M-02] device_tier_ reset | 1분 -- release()에 초기화 추가 |
| 5 (다음 릴리스) | [M-03] ROI temporal ordering | 30분 -- Scissor 타이밍 재배치 |
| 6 (다음 릴리스) | [M-04] Viewport RAII guard | 20분 -- ViewportGuard 구현 |
| 7 (백로그) | [M-05] Thread safety 주석 | 5분 -- 주석 추가 |
| 8 (백로그) | [L-01~L-03] Low 이슈들 | 개별 판단 |

---

## Summary

P4-W3-04 변경사항은 전반적으로 안정적인 구현이다. `detectDeviceTier()`의 방어적 null 체크, 텍스처 할당 실패 처리, viewport 복원 등 좋은 패턴이 적용되어 있다.

**즉시 조치가 필요한 항목**은 2건이다:
1. `std::stoi`를 예외가 발생하지 않는 안전한 변환으로 교체
2. `detectDeviceTier()`를 private 인스턴스 메서드로 변경하여 외부 오용 방지

나머지 Medium/Low 이슈는 코드 견고성을 높이는 개선 사항으로, 다음 릴리스 사이클에서 처리하면 충분하다.
