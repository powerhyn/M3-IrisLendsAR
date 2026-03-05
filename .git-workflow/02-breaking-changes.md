# Breaking Changes & Dependency Analysis

**Branch**: `feature/P4-W3-02` targeting `develop`
**Date**: 2026-03-04
**Scope**: 10 files modified, 528 insertions, 4 deletions
**Feature**: Frequency Separation skin smoothing pipeline (`skinQuality` parameter)

---

## 1. Executive Summary

| Category | Risk Level | Verdict |
|----------|-----------|---------|
| C++ ABI (struct layout) | MEDIUM | ABI-breaking for pre-compiled consumers; source-compatible |
| C API ABI (struct layout) | MEDIUM | ABI-breaking for pre-compiled consumers; source-compatible |
| JNI Bridge | LOW | Name-based field access; no ordering dependency |
| Android Java API | LOW | Additive-only; backward compatible |
| iOS/Flutter/Web Bindings | N/A | Not yet implemented (no binding code exists) |
| New External Dependencies | NONE | All dependencies already in project |
| Database/Schema Changes | NONE | No persistence layer affected |
| Configuration Changes | LOW | New field defaults to 0.0 (disabled) |
| GPU Shader Pipeline | LOW | Additive; graceful fallback on failure |
| Backward Compatibility | LOW | Zero-value semantics preserve existing behavior |

**Overall Assessment**: The changes are **source-compatible** and **behaviorally backward-compatible** but constitute an **ABI break** for any consumer that links against pre-compiled `.so`/`.a` binaries containing the old struct layout. In this project's current deployment model (Android AAR rebuilt from source, no pre-distributed native libraries), the practical risk is **LOW**.

---

## 2. Struct Layout Analysis (ABI Impact)

### 2.1 C++ Core Struct: `BeautyFilterConfigV2`

**File**: `cpp/include/iris_sdk/beauty_filter.h`

```
BEFORE (field order):                    AFTER (field order):
bool   enabled          [offset 0]       bool   enabled          [offset 0]
float  intensity         [offset 4]       float  intensity         [offset 4]
float  smoothing         [offset 8]       float  smoothing         [offset 8]
float  brightness        [offset 12]      float  brightness        [offset 12]
float  softFocus         [offset 16]      float  softFocus         [offset 16]
float  whitening         [offset 20]      float  whitening         [offset 20]
float  colorBalance      [offset 24]      float  colorBalance      [offset 24]
float  wrinkleRemove     [offset 28]      float  wrinkleRemove     [offset 28]
float  slimFace          [offset 32]  --> float  skinQuality       [offset 32]  ** NEW **
float  enlargeEyes       [offset 36]      float  slimFace          [offset 36]
float  thinChin          [offset 40]      float  enlargeEyes       [offset 40]
bool   useGpu            [offset 44]      float  thinChin          [offset 44]
bool   roiOnly           [offset 45]      bool   useGpu            [offset 48]
bool   protectEyes       [offset 46]      bool   roiOnly           [offset 49]
bool   protectLips       [offset 47]      bool   protectEyes       [offset 50]
int    downscaleFactor   [offset 48]      bool   protectLips       [offset 51]
                                          int    downscaleFactor   [offset 52]
sizeof BEFORE: 52 bytes                  sizeof AFTER:  56 bytes
```

**Impact**: The `skinQuality` field is inserted BETWEEN `wrinkleRemove` and `slimFace`. This shifts all subsequent fields by 4 bytes. This is:

- **ABI-breaking**: Any pre-compiled object code that accesses `slimFace`, `enlargeEyes`, `thinChin`, `useGpu`, `roiOnly`, `protectEyes`, `protectLips`, or `downscaleFactor` by offset will read incorrect values.
- **Source-compatible**: Recompiling from source resolves all offsets correctly.

### 2.2 C API Struct: `IrisBeautyConfigV2`

**File**: `cpp/include/iris_sdk/sdk_api.h`

```
BEFORE:                                  AFTER:
int   enabled            [offset 0]      int   enabled            [offset 0]
float intensity          [offset 4]      float intensity          [offset 4]
float smoothing          [offset 8]      float smoothing          [offset 8]
float brightness         [offset 12]     float brightness         [offset 12]
float soft_focus         [offset 16]     float soft_focus         [offset 16]
float whitening          [offset 20]     float whitening          [offset 20]
float color_balance      [offset 24]     float color_balance      [offset 24]
float wrinkle_remove     [offset 28]     float wrinkle_remove     [offset 28]
float slim_face          [offset 32] --> float skin_quality       [offset 32]  ** NEW **
float enlarge_eyes       [offset 36]     float slim_face          [offset 36]
float thin_chin          [offset 40]     float enlarge_eyes       [offset 40]
int   use_gpu            [offset 44]     float thin_chin          [offset 44]
int   roi_only           [offset 48]     int   use_gpu            [offset 48]
int   protect_eyes       [offset 52]     int   roi_only           [offset 52]
int   protect_lips       [offset 56]     int   protect_eyes       [offset 56]
int   downscale_factor   [offset 60]     int   protect_lips       [offset 60]
int   feather_radius     [offset 64]     int   downscale_factor   [offset 64]
                                         int   feather_radius     [offset 68]
sizeof BEFORE: 68 bytes                  sizeof AFTER:  72 bytes
```

**Impact**: Same mid-struct insertion pattern. All fields from `slim_face` onward shift by 4 bytes.

**Mitigation in Place**: The C API provides `iris_sdk_default_beauty_config_v2_c()` which callers use for initialization. This function has been correctly updated to set `skin_quality = 0.0f` in the new position. No `memset`/zero-initialization patterns were found in the codebase, confirming all callers use the explicit initializer.

### 2.3 Struct Packing & Alignment

Both structs use natural alignment (no `#pragma pack` directives detected):

- **C API struct** (`IrisBeautyConfigV2`): All members are `int` or `float` (4 bytes each). No padding is introduced by the new field. The struct grows exactly by `sizeof(float)` = 4 bytes. **No alignment issue**.

- **C++ struct** (`BeautyFilterConfigV2`): Contains `bool` (1 byte) and `float`/`int` (4 bytes) members. The `bool` members are grouped together at the end, so padding behavior is unchanged. The `skinQuality` float is inserted between consecutive float fields. **No new padding introduced**.

---

## 3. JNI Bridge Analysis

**Files**: `android/iris-sdk/src/main/cpp/iris_jni.cpp`, `android/iris-sdk/src/main/cpp/jni_utils.h`

### 3.1 Field Access Pattern

The JNI bridge uses **name-based reflection** (`GetFieldID` with string field names), NOT positional/offset-based access:

```cpp
// jni_utils.h - New field ID added
jfieldID beautyConfigV2_skinQuality = nullptr;

// iris_jni.cpp - Name-based lookup
beautyConfigV2_skinQuality = env->GetFieldID(beautyConfigV2Class, "skinQuality", "F");
```

**Impact**: JNI `GetFieldID` resolves field access by name at runtime, making the binding **completely immune to struct layout changes** in the C++ layer. The Java field ordering and C struct field ordering are independent.

### 3.2 Copy Functions

Both `copyBeautyConfigV2FromJava` and `copyBeautyConfigV2ToJava` have been correctly updated:

- `dest.skin_quality = env->GetFloatField(src, g_jniCache.beautyConfigV2_skinQuality);`
- `env->SetFloatField(dest, g_jniCache.beautyConfigV2_skinQuality, src.skin_quality);`

### 3.3 Field Validation

The `JniCache::init` null-check now includes `beautyConfigV2_skinQuality`:

```cpp
if (!beautyConfigV2_enabled || ... || !beautyConfigV2_skinQuality || !beautyConfigV2_slimFace || ...)
```

**Verdict**: JNI bridge is **fully compatible**. No ordering dependency, name-based resolution is correct, validation includes the new field.

---

## 4. Android Java API Analysis

**File**: `android/iris-sdk/src/main/java/com/irislenssdk/BeautyFilterConfigV2.java`

### Changes Applied

| Component | Status | Notes |
|-----------|--------|-------|
| Field declaration | Added | `public float skinQuality;` with Javadoc |
| Default constant | Added | `DEFAULT_SKIN_QUALITY = 0.0f` |
| Copy constructor | Updated | Copies `skinQuality` from `other` |
| `setDefaults()` | Updated | Sets `skinQuality = DEFAULT_SKIN_QUALITY` |
| `isValid()` | Updated | Validates `0.0 <= skinQuality <= 1.0` |
| `clampValues()` | Updated | Clamps `skinQuality` to [0.0, 1.0] |
| `toString()` | Updated | Includes `skinQuality` in output |
| Builder pattern | Added | `.skinQuality(float)` builder method |

### Backward Compatibility

- **Existing code**: Any existing Java code that does NOT set `skinQuality` will get the default value `0.0f` through `setDefaults()` or the Builder pattern. This means `skinQuality = 0.0f` = **feature disabled**, which is the correct backward-compatible behavior.
- **New field is `public`**: Consistent with all other fields in the class.
- **Builder is additive**: The new `.skinQuality()` builder method is optional. Existing builder chains continue to work unchanged.
- **Serialization**: If `BeautyFilterConfigV2` is ever serialized (Intent extras, SharedPreferences), the absence of `skinQuality` in old serialized data would require handling. However, no serialization code was detected in the codebase.

**Verdict**: Java API is **fully backward compatible**.

---

## 5. C API ↔ C++ Core Mapping

**File**: `cpp/src/sdk_api_v2.cpp`

### Conversion Functions

Both `toCppConfigV2` and `fromCppConfigV2` have been updated to map the new field:

```cpp
// C API → C++ Core
config.skinQuality = c_config->skin_quality;

// C++ Core → C API
c_config->skin_quality = cpp_config.skinQuality;
```

### Default Initialization

`iris_sdk_default_beauty_config_v2_c` now includes:
```cpp
config->skin_quality = 0.0f;
```

**Verdict**: Mapping is **complete and consistent**. No field is missed in the conversion layer.

---

## 6. iOS / Flutter / Web Bindings Gap Analysis

### Current State

A thorough search of the repository confirms:
- **iOS**: No Objective-C++ binding code exists yet (only TFLite dependency samples in `docs/mediapipe_sample/ios/`).
- **Flutter**: No `dart:ffi` binding code exists yet.
- **Web**: No WASM/Emscripten binding code exists yet.

Per the `CLAUDE.md` development order:
```
1. core/ - C++ 코어 엔진 먼저 구현     ✅ Active development
2. bindings/android/ - JNI 바인딩       ✅ Active development
3. bindings/ios/ - iOS Framework        ⏳ Future
4. bindings/flutter/ - Flutter Plugin   ⏳ Future
5. bindings/web/ - WASM (선택적)         ⏳ Future
```

### Risk Assessment

| Platform | Risk | Rationale |
|----------|------|-----------|
| iOS | NONE | Binding does not exist yet; will be built against updated C API |
| Flutter | NONE | Binding does not exist yet; will use updated `sdk_api.h` header |
| Web | NONE | Binding does not exist yet; will compile against updated source |

**Verdict**: The absence of `skinQuality` in iOS/Flutter/Web bindings is **NOT a concern** because those bindings have not been implemented. When they are built, they will naturally include the new field from the C API header.

---

## 7. Dependency Analysis

### 7.1 New Includes Added

| File | New Include | Already in Project | Risk |
|------|------------|-------------------|------|
| `gpu_beauty_backend.h` | `iris_sdk/one_euro_filter.h` | Yes (`cpp/include/iris_sdk/one_euro_filter.h`) | NONE |
| `gpu_beauty_backend.h` | `<vector>` | Standard library | NONE |

### 7.2 OneEuroFilter Usage

The `OneEuroFilter` is already used elsewhere in the project (tracking stability). The new usage in `GPUBeautyBackend` is for temporal smoothing of the Frequency Separation radius parameter:

```cpp
// Member variable with inline initialization
OneEuroFilter skin_radius_filter_{0.5f, 0.01f, 1.0f};
```

This is marked as "P4-W3-04에서 사용" (for future use in P4-W3-04). The member is initialized but not yet called from any method in this diff, which means it adds ~12-24 bytes to the `GPUBeautyBackend` object size but has no behavioral impact.

### 7.3 External Dependency Changes

| Category | Change |
|----------|--------|
| New third-party libraries | NONE |
| Version bumps | NONE |
| New system libraries | NONE |
| New build flags | NONE |
| CMakeLists.txt changes | NONE (not in diff) |

**Verdict**: **No new external dependencies**. All includes reference existing project headers or standard library components.

---

## 8. GPU Shader Pipeline Analysis

### 8.1 New Shaders Added

**File**: `cpp/src/gpu/shader_sources.cpp`

| Shader | GLSL Version | Purpose |
|--------|-------------|---------|
| `FREQ_SEP_GAUSSIAN_FRAGMENT` | 310 es | Separable 1D Gaussian blur for frequency decomposition |
| `FREQ_SEP_COMPOSITE_FRAGMENT` | 310 es | High-frequency re-synthesis with mask-based blending |

### 8.2 GLES Compatibility

Both shaders require **OpenGL ES 3.1** (`#version 310 es`). This is consistent with the existing shader set in the project. The minimum Android API level for GLES 3.1 is API 21 (Android 5.0), which should already be the project baseline.

Potential concern: The Gaussian shader uses a **dynamic loop bound** (`for (int i = -uRadius; i <= uRadius; i++)`). On some older Mali and Adreno GPUs, dynamic loop bounds in fragment shaders can cause:
- Shader compilation failure on drivers that require constant loop bounds
- Performance degradation due to branch divergence

Recommendation: Consider adding a compile-time `MAX_RADIUS` guard or unrolling for fixed radius values on low-tier devices.

### 8.3 Graceful Degradation

The shader initialization is correctly non-fatal:

```cpp
if (!initializeFreqSepShaders()) {
    LOGW("Failed to create Freq Sep shaders (non-fatal)");
    // Bilateral fallback is used automatically
}
```

The pipeline execution checks shader availability before attempting FreqSep:

```cpp
if (freq_sep_params.enabled
    && freq_sep_gaussian_program_ != 0 && freq_sep_composite_program_ != 0
    && roi_ptr && roi_ptr->valid && !roi_ptr->combined_mask.empty()) {
    // FreqSep path
} else if (...) {
    // Bilateral fallback
}
```

**Verdict**: Shader pipeline is **robustly designed** with proper fallback. No breaking behavior for existing functionality.

---

## 9. Behavioral Backward Compatibility

### 9.1 Zero-Value Semantics

The `skinQuality` field defaults to `0.0f` across all layers:

| Layer | Default Value | Mechanism |
|-------|--------------|-----------|
| C++ `BeautyFilterConfigV2Helper::createDefault()` | `0.0f` | Explicit assignment |
| C API `iris_sdk_default_beauty_config_v2_c()` | `0.0f` | Explicit assignment |
| Java `BeautyFilterConfigV2.setDefaults()` | `0.0f` | `DEFAULT_SKIN_QUALITY` constant |
| Java primitive default | `0.0f` | Java `float` default |

When `skinQuality = 0.0f`:
- `mapSkinQuality()` returns `FreqSepParams{ enabled = false }`
- The Frequency Separation pipeline is **completely skipped**
- Execution falls through to the existing Bilateral filter path (`config.smoothing > 0.01f`)

**Verdict**: Existing behavior is **100% preserved** when `skinQuality` is not set. The new pipeline only activates on explicit opt-in.

### 9.2 Active Filter Count Change

One behavioral modification affects the texture allocation logic:

```cpp
// BEFORE:
if (config.smoothing > 0.01f) active_filter_count++;

// AFTER:
if (config.skinQuality > 0.0f || config.smoothing > 0.01f) active_filter_count++;
```

This means when `skinQuality > 0.0f` but `smoothing = 0.0f`, a smoothing texture will now be allocated that previously would not have been. This is correct behavior (FreqSep needs output textures) and does not affect cases where `skinQuality = 0.0f`.

### 9.3 Fallback Smoothing Strength

When FreqSep fails (shader compilation failure, texture allocation failure, missing ROI mask), the fallback applies a minimum smoothing strength derived from `skinQuality`:

```cpp
float effective_smoothing = config.skinQuality * 0.5f;
if (fallback_config.smoothing < effective_smoothing) {
    fallback_config.smoothing = effective_smoothing;
}
```

This only triggers when `skinQuality > 0.0f`, so it does not affect existing behavior.

---

## 10. Test Coverage

**File**: `cpp/tests/test_beauty_config_v2.cpp`

### Current Coverage

Only the `DefaultValuesFromHelper` test was updated to verify `skinQuality` defaults to `0.0f`.

### Missing Test Cases

| Test Case | Priority | Description |
|-----------|----------|-------------|
| `mapSkinQuality` edge cases | HIGH | Test `skinQuality = 0.0`, `0.5`, `1.0` with varying `face_width` |
| `mapSkinQuality` boundary values | HIGH | Test `face_width = 0`, negative values, extreme values |
| Validation range check | MEDIUM | Verify `isValid()` rejects `skinQuality < 0` and `> 1.0` |
| Clamping | MEDIUM | Verify `clamp()` brings out-of-range values into [0, 1] |
| C API round-trip | MEDIUM | Verify `toCppConfigV2` + `fromCppConfigV2` preserves `skinQuality` |
| FreqSep fallback behavior | LOW | Test GPU pipeline fallback when shaders unavailable |

---

## 11. Risk Matrix Summary

| # | Risk | Severity | Probability | Mitigation |
|---|------|----------|-------------|------------|
| 1 | ABI break for pre-compiled consumers | HIGH | LOW | Project builds all bindings from source; no distributed `.so` files |
| 2 | Dynamic loop bound in GLES shader | MEDIUM | LOW | Consider `MAX_RADIUS` guard for older GPUs |
| 3 | Unused `skin_radius_filter_` member | LOW | N/A | Intended for P4-W3-04; minimal memory cost |
| 4 | Missing unit tests for `mapSkinQuality` | MEDIUM | HIGH | Should be added before merge |
| 5 | No version bump in C API header | LOW | MEDIUM | Consider adding `IRIS_SDK_API_VERSION` to detect ABI mismatches |

---

## 12. Recommendations

### Must-Do (Before Merge)

1. **Add `mapSkinQuality` unit tests**: The public static method is the only new public API surface and should have dedicated test coverage for edge cases and boundary values.

### Should-Do (Near-Term)

2. **Consider API versioning**: Add a `#define IRIS_SDK_API_VERSION` or `sizeof`-based version check to `sdk_api.h`. This would allow runtime detection of struct layout mismatches if pre-compiled binaries are ever distributed.

3. **Document shader GPU requirements**: Note the GLES 3.1 requirement and dynamic loop bounds in the work paper for device-tier testing (P4-W3-04).

### Nice-to-Have (Future)

4. **Shader loop guard**: Add a `#define MAX_FREQ_SEP_RADIUS 28` compile-time guard to the Gaussian shader to ensure constant loop bounds on all GPU drivers.

5. **Remove unused member or annotate**: The `skin_radius_filter_` member is declared but unused. Either add a `[[maybe_unused]]` attribute or defer its addition to P4-W3-04 to keep the diff minimal.

---

## 13. Conclusion

This changeset is **safe to merge** with the following considerations:

- **No breaking runtime behavior**: The `skinQuality = 0.0f` default ensures complete backward compatibility for all existing clients.
- **ABI technically broken**: Struct layout changed in both C++ and C API layers, but the project's build model (full source rebuild for Android AAR) eliminates practical risk.
- **JNI bridge is immune**: Name-based field resolution means Java/native field ordering is decoupled.
- **Cross-platform bindings**: Not yet implemented, so no gap to close.
- **Graceful degradation**: Shader failures fall back to existing Bilateral filter path transparently.

The primary action item is adding unit tests for the `mapSkinQuality` mapping function before merging.
