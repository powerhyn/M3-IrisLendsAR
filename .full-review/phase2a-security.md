# Phase 2a: Security Audit -- Vivid Post-Processing Filter

**Auditor**: Security Auditor (DevSecOps)
**Scope**: Vivid post-processing 필터 구현 -- develop 브랜치 변경사항 (+307/-12 lines, 10 files)
**Framework**: C++17 / OpenGL ES 3.1 / Android JNI
**Date**: 2026-03-12

---

## Executive Summary

전반적으로 보안 품질은 양호합니다. 셰이더는 컴파일 타임 상수 문자열로 주입 벡터가 없고, JNI 레이어는 캐시된 필드 ID와 예외 검사 패턴을 일관되게 사용합니다. 그러나 **입력 검증 부재로 인한 NaN/Inf 전파**, **POD 구조체 ABI 비버전링**, **applyTexture(TextureHandle) 경로의 vivid 누락** 등 실질적인 결함이 존재합니다.

| Severity | Count | Summary |
|----------|-------|---------|
| HIGH     | 2     | NaN/Inf 전파, POD ABI 미버전링 |
| MEDIUM   | 3     | applyTexture vivid 누락, GPU 리소스 누수 조건, toCppConfigV2 미클램핑 |
| LOW      | 3     | feather_radius 미변환, integer overflow in frame size calc, reinterpret_cast 안전성 |
| INFO     | 2     | 새 의존성 없음 확인, 셰이더 인젝션 해당 없음 확인 |

---

## Findings

### [S-01] HIGH -- NaN/Inf 파라미터 미검증으로 GPU 파이프라인 오염

**CWE**: CWE-20 (Improper Input Validation)
**Files**:
- `cpp/src/sdk_api_v2.cpp` :: `toCppConfigV2()` (line 52-84)
- `cpp/src/gpu/gpu_beauty_backend.cpp` :: `executeVividPass()` (line 870-890)

**Description**:
`toCppConfigV2()`는 C API 구조체의 float 필드를 그대로 C++ 구조체로 복사합니다. `std::isfinite()` 검증이 전혀 없어, JNI 레이어에서 `Float.NaN`이나 `Float.POSITIVE_INFINITY`가 전달되면 GPU uniform에 NaN이 설정됩니다.

GLSL에서 NaN은 `clamp()`나 비교 연산에서 정의되지 않은 동작을 유발합니다. `VIVID_POSTPROCESS_FRAGMENT`의 `mix(color.rgb, result, uIntensity)`에서 `uIntensity`가 NaN이면 전체 프레임이 검은색 또는 임의의 색으로 렌더링됩니다. 일부 GPU 드라이버에서는 NaN uniform이 셰이더 행 상태(hang state)를 유발할 수 있습니다.

**Attack Vector**: 악의적인 SDK 소비자가 `IrisBeautyConfigV2.vivid_intensity = Float.NaN`을 설정하여 전송.

**Remediation**:
```cpp
// toCppConfigV2() 또는 applyTextureId() 진입점에서:
static bool isFiniteFloat(float v) { return std::isfinite(v); }

// 모든 vivid 파라미터 검증
if (!isFiniteFloat(c_config->vivid_intensity) ||
    !isFiniteFloat(c_config->vivid_saturation) ||
    !isFiniteFloat(c_config->vivid_brightness) ||
    !isFiniteFloat(c_config->vivid_warmth)) {
    // 기본값 사용 또는 IRIS_SDK_INVALID_PARAM 반환
}
```

또는 `BeautyFilterConfigV2Helper::clamp()` 내에서 NaN을 하한값으로 치환하도록 방어적 클램핑을 적용합니다 (NaN < lo 비교는 false이므로, `std::clamp` 대신 삼항 연산자 기반 클램핑 필요).

**현재 코드의 clamp 함수**:
```cpp
auto clampf = [](float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
};
```
이 코드는 NaN에 대해 `hi` 값을 반환합니다 (NaN < lo = false, NaN > hi = false). 이는 의도와 다를 수 있으나 최소한 값 자체는 유한하게 됩니다. 그러나 **clamp가 호출되지 않는 경로가 문제**입니다 -- `toCppConfigV2()`는 clamp를 호출하지 않습니다.

---

### [S-02] HIGH -- POD 구조체 ABI 비버전링: 이전 바이너리와의 레이아웃 불일치

**CWE**: CWE-131 (Incorrect Calculation of Buffer Size)
**Files**:
- `cpp/include/iris_sdk/beauty_filter.h` :: `BeautyFilterConfigV2` (line 63-117)
- `cpp/include/iris_sdk/sdk_api.h` :: `IrisBeautyConfigV2` (line 590-622)

**Description**:
`BeautyFilterConfigV2`와 `IrisBeautyConfigV2`에 vivid 필드 4개가 추가되었으나, 구조체에 버전 필드나 크기 필드가 없습니다. SDK 사용자가 이전 버전의 헤더로 컴파일한 바이너리에서 새 SDK .so를 로드하면:

1. 구조체 끝부분의 vivid 필드가 초기화되지 않은 스택/힙 메모리를 읽습니다
2. `sizeof(IrisBeautyConfigV2)` 불일치로 `memset` 범위가 부족합니다
3. 필드 오프셋 이동은 없지만(뒤에 추가), 구조체 크기 차이로 인한 garbage 값 읽기가 발생합니다

이는 C API를 공유 라이브러리(.so/.dylib)로 배포하는 모든 바인딩(JNI, FFI)에 영향을 미칩니다.

**Impact**: vivid 파라미터에 garbage float 값(수백~수천의 값)이 전달되면, GPU 셰이더에서 `result.r += uWarmth * 0.04`가 극도로 큰 값이 되어 프레임 전체가 백색 클리핑되거나, 드라이버에 따라 GPU hang이 발생할 수 있습니다.

**Remediation**:
```c
typedef struct IrisBeautyConfigV2 {
    uint32_t struct_size;  // sizeof(IrisBeautyConfigV2) -- 버전 식별용
    // ... existing fields ...
} IrisBeautyConfigV2;

// API 함수에서:
if (config->struct_size < offsetof(IrisBeautyConfigV2, vivid_intensity) + sizeof(float)*4) {
    // vivid 필드 접근 불가 -- 기본값(0.0) 사용
}
```

또는 `iris_sdk_default_beauty_config_v2_c()`를 항상 호출하도록 문서화하고, C API 레벨에서 sentinel 값 검증을 추가합니다.

---

### [S-03] MEDIUM -- applyTexture(TextureHandle) 경로에서 vivid 미적용

**CWE**: CWE-684 (Incorrect Provision of Specified Functionality)
**File**: `cpp/src/gpu/gpu_beauty_backend.cpp` :: `applyTexture()` (line 605-625)

**Description**:
`applyTextureId()` (C API 경로)에는 vivid 패스가 구현되어 있으나, `applyTexture(const TextureHandle&, ...)` (C++ 직접 호출 경로)에서는 `config.enabled == false`일 때 즉시 pass-through합니다:

```cpp
if (!config.enabled) {
    output = input;
    return IRIS_SDK_OK;  // vivid 파라미터 무시
}
```

Phase 1 Context에서 지적된 대로, vivid는 beauty `enabled`와 독립적으로 동작해야 합니다. `applyTextureId()`에서는 `needsVivid` 가드가 올바르게 구현되어 있으나 `applyTexture()`에는 없습니다.

**Impact**: C++ 코어를 직접 사용하는 향후 바인딩(iOS Obj-C++, Flutter FFI)에서 vivid가 동작하지 않는 API 불일치.

**보안 관점**: 기능 불일치 자체는 직접적 취약점이 아니나, 사용자가 `applyTexture` 경로에서 beauty를 비활성화하고 vivid만 활성화하면 "아무 효과 없음"으로 fallthrough됩니다. 이는 예상치 못한 동작으로 사용자 혼란을 유발하고, 잘못된 코드 경로 탐색으로 이어질 수 있습니다.

**Remediation**:
```cpp
IrisSdkError GPUBeautyBackend::applyTexture(
    const TextureHandle& input, TextureHandle& output,
    const BeautyFilterConfigV2& config, const BeautyROI* roi) {
    // ...
    bool needsVivid = config.vividIntensity > 0.01f;
    if (!config.enabled && !needsVivid) {
        output = input;
        return IRIS_SDK_OK;
    }
    // vivid-only 경로 처리 추가
}
```

---

### [S-04] MEDIUM -- toCppConfigV2()에서 클램핑 미수행

**CWE**: CWE-20 (Improper Input Validation)
**File**: `cpp/src/sdk_api_v2.cpp` :: `toCppConfigV2()` (line 52-84)

**Description**:
`toCppConfigV2()` 함수는 C API 구조체 값을 그대로 C++ 구조체에 복사할 뿐, 범위 검증이나 클램핑을 수행하지 않습니다. `BeautyFilterConfigV2Helper::clamp()` 함수가 존재하지만 이 변환 함수에서 호출되지 않습니다.

Java의 `Builder.build()`는 `clamp()`를 호출하지만, C API를 직접 사용하는 소비자(Flutter FFI, 서드파티 네이티브 코드)는 이 보호를 받지 못합니다.

**Impact**: `vivid_brightness`에 1.0f (범위 0.0~0.5)가 전달되면 셰이더에서:
```glsl
result += 1.0 * result * (1.0 - result);  // 최대 +0.25 밝기 증가 (중간톤 기준)
```
이는 시각적으로 과도한 밝기이나 crash는 아닙니다. 그러나 `vivid_warmth`에 100.0f가 전달되면:
```glsl
result.r += 100.0 * 0.04;  // +4.0 → 완전 클리핑
result.b -= 100.0 * 0.03;  // -3.0 → 완전 블랙
```
프레임이 붉은색으로 완전히 변형됩니다.

**Remediation**:
```cpp
BeautyFilterConfigV2 toCppConfigV2(const IrisBeautyConfigV2* c_config) {
    BeautyFilterConfigV2 config = {};
    // ... 기존 복사 ...
    iris_sdk::BeautyFilterConfigV2Helper::clamp(config);  // 추가
    return config;
}
```

---

### [S-05] MEDIUM -- GPU 텍스처 누수 조건: vivid-only + ROI 스킵 경로

**CWE**: CWE-401 (Missing Release of Memory after Effective Lifetime)
**File**: `cpp/src/gpu/gpu_beauty_backend.cpp` :: `applyTextureId()` (line ~1762-1770)

**Description**:
ROI Scissor 교집합이 비어있을 때 vivid-only 경로에서:
```cpp
if (needsVivid) {
    executeVividPass(current_input, ping->fbo_id, ...);
    *output_texture = ping->texture_id;
}
```
이후 `previous_output_ping_` / `pong` 업데이트가 조건부로만 이루어집니다. 이 경로를 반복적으로 hit하면 `g_managed_textures`에 텍스처가 누적되지만, 텍스처 풀의 ping/pong 메커니즘이 이전 텍스처를 올바르게 해제하지 않을 수 있습니다.

프레임 단위 반복에서 이 조건이 지속되면 (예: 얼굴이 프레임 밖에 있을 때) 텍스처 메모리가 점진적으로 증가합니다.

**Impact**: 장시간 사용 시 GPU 메모리 고갈. Android에서 EGL_BAD_ALLOC → 앱 crash.

**Remediation**: vivid-only ROI 스킵 경로에서도 `previous_output_ping_` 업데이트를 보장하고, 텍스처 풀 반환 로직을 early-return 경로에도 적용합니다.

---

### [S-06] LOW -- feather_radius 필드가 toCppConfigV2에서 누락

**CWE**: CWE-665 (Improper Initialization)
**File**: `cpp/src/sdk_api_v2.cpp` :: `toCppConfigV2()` (line 52-84)

**Description**:
`IrisBeautyConfigV2`에는 `feather_radius` 필드(line 615)가 있고 `iris_sdk_default_beauty_config_v2_c()`에서 15로 초기화하지만, `toCppConfigV2()`에서 이 값을 `BeautyFilterConfigV2`로 복사하지 않습니다. C++ 구조체 `BeautyFilterConfigV2`에도 해당 필드가 없는 것으로 보입니다.

**Impact**: feather_radius 설정이 C API에서만 존재하고 실제로 사용되지 않을 가능성. 현재 직접적 보안 위협은 없으나, 사용자 혼란 유발.

---

### [S-07] LOW -- calculateExpectedFrameSize에서 integer overflow 가능성

**CWE**: CWE-190 (Integer Overflow)
**File**: `android/.../jni_utils.h` :: `calculateExpectedFrameSize()` (line 531-549)

**Description**:
```cpp
const jsize pixels = width * height;
return pixels * 4;
```
`jsize`는 `int32_t`입니다. width=65536, height=65536이면 `pixels = 65536 * 65536 = 4294967296`으로 int32 overflow가 발생합니다. `pixels * 4`도 overflow됩니다. 결과적으로 `validateFrameBufferSize()`가 잘못된 비교를 수행하여 너무 작은 버퍼를 유효하다고 판단할 수 있습니다.

**Impact**: 극단적인 해상도에서만 트리거됩니다. 실제 카메라 프레임에서는 발생하기 어려우나, 악의적 입력이 가능한 환경에서는 heap overflow로 이어질 수 있습니다.

**Remediation**:
```cpp
inline jsize calculateExpectedFrameSize(jint width, jint height, jint format) noexcept {
    if (width <= 0 || height <= 0) return 0;
    int64_t pixels = static_cast<int64_t>(width) * height;
    if (pixels > INT32_MAX / 4) return 0;  // overflow guard
    // ...
}
```

---

### [S-08] LOW -- reinterpret_cast<const iris_sdk::IrisLandmark*>(detection->face_mesh) 안전성

**CWE**: CWE-704 (Incorrect Type Conversion)
**File**: `cpp/src/sdk_api_v2.cpp` (line 234-235)

**Description**:
```cpp
const iris_sdk::IrisLandmark* face_mesh_ptr =
    reinterpret_cast<const iris_sdk::IrisLandmark*>(detection->face_mesh);
```
C API의 `IrisLandmark` (4 floats: x, y, z, visibility)와 C++ `iris_sdk::IrisLandmark`가 동일한 POD 레이아웃이라는 가정에 의존합니다. 주석에서 이를 명시하고 있으나, `static_assert(sizeof(::IrisLandmark) == sizeof(iris_sdk::IrisLandmark))`가 없어 향후 한쪽만 수정될 경우 정렬/크기 불일치로 메모리 접근 오류가 발생할 수 있습니다.

**Remediation**:
```cpp
static_assert(sizeof(::IrisLandmark) == sizeof(iris_sdk::IrisLandmark),
              "C and C++ IrisLandmark must have identical layout");
static_assert(alignof(::IrisLandmark) == alignof(iris_sdk::IrisLandmark),
              "C and C++ IrisLandmark must have identical alignment");
```

---

### [S-09] INFO -- 신규 의존성 확인

새로운 외부 의존성이 추가되지 않았습니다. vivid 기능은 기존 OpenGL ES 3.1 셰이더 인프라를 재사용하며, 서드파티 라이브러리 추가 없이 구현되었습니다.

---

### [S-10] INFO -- 셰이더 인젝션 해당 없음 확인

`VIVID_POSTPROCESS_FRAGMENT` 셰이더는 C++ 소스의 `R"glsl(...)glsl"` raw string literal로 컴파일 타임에 고정됩니다. 런타임에 셰이더 소스를 조합하거나 외부 입력을 삽입하는 경로가 없습니다. uniform 값만 외부에서 설정되므로 셰이더 인젝션 벡터는 존재하지 않습니다.

---

## Configuration Security: GPU Hang 가능성 분석

**질문**: 악의적 config 값이 GPU hang이나 과도한 리소스 사용을 유발할 수 있는가?

**분석**:

1. **Vivid 셰이더**: 단일 패스, 텍스처 샘플 1회 + ALU 연산. 모든 uniform 값이 유한한 float이면 O(1) 연산으로, 극단적 값이라도 GPU hang은 불가합니다.

2. **Bilateral Filter**: RADIUS=4 (고정 상수), 9x9 커널. `uStrength` 값과 무관하게 반복 횟수가 고정이므로 GPU hang 위험 없습니다.

3. **FreqSep Gaussian**: `uRadius`가 uniform으로 전달되나, CPU에서 `std::clamp(radius, 1, kMaxGaussianRadius)`로 최대 28로 제한됩니다. 셰이더 루프는 `for (int i = -uRadius; i <= uRadius; i++)`이므로 최대 57회 반복 -- hang 위험 없습니다.

4. **NaN uniform**: 일부 GPU 드라이버(특히 Mali-G 계열)에서 NaN이 포함된 프래그먼트 연산이 무한 루프와 유사한 stall을 유발할 수 있다는 보고가 있습니다. 이는 S-01의 NaN 검증으로 완화해야 합니다.

**결론**: 유한한 float 값이 보장되면 GPU hang/DoS 위험은 없습니다. **S-01 수정이 선결 조건**입니다.

---

## JNI Security Analysis

### 필드 접근 안전성

JNI 캐시(`jni_utils.h` line 367-389)에서 vivid 필드 4개의 `jfieldID`를 `JNI_OnLoad` 시점에 캐싱하고, null 여부를 검증합니다 (line 177-178):

```cpp
if (!beautyConfigV2_vividIntensity || !beautyConfigV2_vividSaturation ||
    !beautyConfigV2_vividBrightness || !beautyConfigV2_vividWarmth) {
    LOGE("Failed to get BeautyFilterConfigV2 field IDs");
    return false;
}
```

이 패턴은 올바릅니다. 필드 이름 불일치 시 `JNI_OnLoad`가 실패하여 네이티브 crash를 사전 방지합니다.

### Null Object 처리

`copyBeautyConfigV2FromJava()`에서 `src` jobject가 null인 경우의 가드가 코드 상단에 있어야 합니다. 현재 `env->GetFloatField(null_obj, field_id)`가 호출되면 JNI는 `NoSuchFieldError`를 throw하지 않고 0.0f를 반환하거나, 일부 구현에서는 crash합니다. 호출측(JNI 네이티브 함수)에서 null 체크가 수행되는지 확인이 필요합니다.

---

## Remediation Priority

| ID   | Severity | Fix Effort | Priority |
|------|----------|------------|----------|
| S-01 | HIGH     | 30분       | P0 -- 즉시 |
| S-02 | HIGH     | 2시간      | P1 -- 다음 릴리즈 전 |
| S-04 | MEDIUM   | 15분       | P0 -- S-01과 함께 |
| S-03 | MEDIUM   | 1시간      | P1 -- applyTexture 경로 수정 |
| S-05 | MEDIUM   | 30분       | P1 -- 텍스처 누수 방지 |
| S-08 | LOW      | 5분        | P2 -- static_assert 추가 |
| S-07 | LOW      | 15분       | P2 -- overflow guard 추가 |
| S-06 | LOW      | 10분       | P2 -- 필드 정리 |

---

## Recommended Immediate Actions

1. **S-01 + S-04**: `toCppConfigV2()` 끝에 `BeautyFilterConfigV2Helper::clamp()` 호출 추가. 이것만으로 NaN은 유한값으로, 범위 초과는 정상 범위로 교정됩니다.

2. **S-02**: `IrisBeautyConfigV2`에 `uint32_t struct_size` 필드를 첫 번째 멤버로 추가하고, `iris_sdk_default_beauty_config_v2_c()`에서 `sizeof(IrisBeautyConfigV2)`로 설정. API 함수 진입부에서 크기 검증.

3. **S-03**: `applyTexture()` 진입부에 `needsVivid` 가드 추가하여 `applyTextureId()`와 동일한 비활성화 로직 적용.
