# Test Coverage Gap Analysis: Frequency Separation GPU Pipeline (P4-W3-02)

**Branch**: `feature/P4-W3-02`
**Date**: 2026-03-04
**Scope**: skinQuality field, mapSkinQuality(), Freq Sep GPU pipeline, uploadSkinMask(), executeSmoothingWithFallbackStrength()

---

## 1. Current Test Inventory

### 1.1 Test Files in `cpp/tests/` (24 files)

| File | Scope | GPU Required | Status |
|------|-------|-------------|--------|
| `test_beauty_config_v2.cpp` | BeautyFilterConfigV2 defaults, validation, clamping, V1/V2 conversion, C API | No | PASSING (skinQuality default = 0.0 verified) |
| `test_gpu_beauty_backend.cpp` | ShaderManager, TexturePool, GPUBeautyBackend metadata/stubs | No (desktop stub) | PASSING |
| `test_beauty_processor.cpp` | BeautyProcessor + CPUBeautyBackend lifecycle, apply, ROI | No | PASSING |
| `test_sdk_api.cpp` | C API lifecycle, error codes, struct sizes, format enums | No | PASSING |
| `test_types.cpp` | IrisLandmark, Rect, IrisResult, LensConfig, enums | No | PASSING |
| `test_beauty_roi_manager.cpp` | BeautyROIManager face mesh, mask generation | No (OpenCV needed) | PASSING |
| `test_fast_guided_filter.cpp` | FastGuidedFilter CPU processing | No (OpenCV needed) | PASSING |
| `test_new_filter_effects.cpp` | V2 CPU filter effects (guided filter, wrinkle) | No (OpenCV needed) | PASSING |
| `test_iris_detector.cpp` | IrisDetector interface | No | PASSING |
| `test_mediapipe_detector.cpp` | MediaPipe detector unit tests | No (TFLite needed) | PASSING |
| `test_mediapipe_detector_integration.cpp` | MediaPipe integration | Yes (TFLite+OpenCV) | PASSING |
| `test_mediapipe_detector_performance.cpp` | MediaPipe performance | Yes (TFLite+OpenCV) | PASSING |
| `test_lens_renderer.cpp` | Lens rendering logic | No | PASSING |
| `test_lens_renderer_integration.cpp` | Lens rendering integration | Partial | PASSING |
| `test_render_context.cpp` | IRenderContext factory | No | PASSING |
| `test_frame_processor.cpp` | Frame processing pipeline | No | PASSING |
| `test_sdk_manager.cpp` | SDKManager singleton | No | PASSING |
| `test_face_warp_controller.cpp` | Face warp mesh | No | PASSING |
| `test_eye_enlargement.cpp` | Eye enlargement effect | No | PASSING |
| `test_profiler.cpp` | Profiler utility | No | PASSING |
| `test_grid_mesh.cpp` | Grid mesh utility | No | PASSING |
| `test_integration.cpp` | Integration tests | Partial | PASSING |
| `test_camera_demo.cpp` | Camera demo test | Device needed | CONDITIONAL |
| `test_placeholder.cpp` | Placeholder | No | PASSING |

### 1.2 P4-W3-02 Related Coverage (Current)

| New Code Path | Current Test Coverage | Gap |
|---------------|----------------------|-----|
| `skinQuality` field in `BeautyFilterConfigV2` | Default value check in `test_beauty_config_v2.cpp` line 28 | Partial: no validation/clamp tests |
| `skinQuality` in `isValid()` | Covered implicitly by existing range tests | Missing: explicit boundary tests for skinQuality |
| `skinQuality` in `clamp()` | Covered implicitly by existing clamp tests | Missing: explicit skinQuality clamp test |
| `mapSkinQuality()` (public static) | **NONE** | **CRITICAL GAP** |
| `FreqSepParams` struct defaults | **NONE** | Missing |
| `executeFreqSepPipeline()` | **NONE** (requires GPU) | GPU-only, documented |
| `uploadSkinMask()` | **NONE** (requires GPU) | GPU-only, documented |
| `executeSmoothingWithFallbackStrength()` | **NONE** (requires GPU for real path) | Logic testable on desktop stub |
| `initializeFreqSepShaders()` | **NONE** (requires GPU) | GPU-only, documented |
| `skinQuality` in C API round-trip | **NONE** | Testable without GPU |
| `skinQuality` in JNI copy | **NONE** | Requires Android device |
| `skinQuality` in Java validation | **NONE** | Requires Android environment |
| Freq Sep shader compilation | **NONE** | Requires Android GPU |

---

## 2. Prioritized Missing Tests (Desktop-Testable)

### PRIORITY 1 -- CRITICAL (must-add before merge)

#### 2.1 `mapSkinQuality()` Unit Tests

**Rationale**: This is the ONLY new public static method. It is a pure function with no GPU dependency -- fully testable on desktop. All subsequent pipeline behavior depends on its output correctness.

**Test Cases**:

```cpp
// File: cpp/tests/test_gpu_beauty_backend.cpp (append to existing file)

//=============================================================================
// mapSkinQuality 단위 테스트
//=============================================================================

TEST(MapSkinQualityTest, ZeroDisablesFreqSep) {
    auto params = GPUBeautyBackend::mapSkinQuality(0.0f, 300);
    EXPECT_FALSE(params.enabled);
}

TEST(MapSkinQualityTest, NegativeValueDisablesFreqSep) {
    auto params = GPUBeautyBackend::mapSkinQuality(-0.5f, 300);
    EXPECT_FALSE(params.enabled);
}

TEST(MapSkinQualityTest, PositiveValueEnablesFreqSep) {
    auto params = GPUBeautyBackend::mapSkinQuality(0.1f, 300);
    EXPECT_TRUE(params.enabled);
}

TEST(MapSkinQualityTest, BlurRadiusProportionalToFaceWidth) {
    // face_width=300 * 0.05 = 15, clamp(6,28)
    auto params = GPUBeautyBackend::mapSkinQuality(0.5f, 300);
    EXPECT_EQ(params.blur_radius, 15);
}

TEST(MapSkinQualityTest, BlurRadiusClampsToMinimum) {
    // face_width=50 * 0.05 = 2.5 -> clamp to 6
    auto params = GPUBeautyBackend::mapSkinQuality(0.5f, 50);
    EXPECT_EQ(params.blur_radius, 6);
}

TEST(MapSkinQualityTest, BlurRadiusClampsToMaximum) {
    // face_width=1000 * 0.05 = 50 -> clamp to 28
    auto params = GPUBeautyBackend::mapSkinQuality(0.5f, 1000);
    EXPECT_EQ(params.blur_radius, 28);
}

TEST(MapSkinQualityTest, BlurRadiusWithZeroFaceWidth) {
    // face_width=0 * 0.05 = 0 -> clamp to 6
    auto params = GPUBeautyBackend::mapSkinQuality(0.5f, 0);
    EXPECT_EQ(params.blur_radius, 6);
}

TEST(MapSkinQualityTest, BlurRadiusWithNegativeFaceWidth) {
    // face_width=-100 * 0.05 = -5 -> clamp to 6
    auto params = GPUBeautyBackend::mapSkinQuality(0.5f, -100);
    EXPECT_EQ(params.blur_radius, 6);
}

TEST(MapSkinQualityTest, HighFreqPreserveDecreasesWithQuality) {
    auto low = GPUBeautyBackend::mapSkinQuality(0.2f, 300);
    auto mid = GPUBeautyBackend::mapSkinQuality(0.5f, 300);
    auto high = GPUBeautyBackend::mapSkinQuality(0.8f, 300);

    // Higher quality -> lower preserve (smoother skin)
    EXPECT_GT(low.high_freq_preserve, mid.high_freq_preserve);
    EXPECT_GT(mid.high_freq_preserve, high.high_freq_preserve);
}

TEST(MapSkinQualityTest, HighFreqPreserveBoundaries) {
    // skinQuality=0.0001 (barely enabled) -> preserve near 1.0
    auto minimal = GPUBeautyBackend::mapSkinQuality(0.001f, 300);
    EXPECT_NEAR(minimal.high_freq_preserve, 1.0f, 0.05f);

    // skinQuality=1.0 -> preserve = 0.10
    auto maximal = GPUBeautyBackend::mapSkinQuality(1.0f, 300);
    EXPECT_NEAR(maximal.high_freq_preserve, 0.10f, 0.01f);
}

TEST(MapSkinQualityTest, AttenuationRangeIncreasesWithQuality) {
    auto low = GPUBeautyBackend::mapSkinQuality(0.2f, 300);
    auto high = GPUBeautyBackend::mapSkinQuality(0.8f, 300);

    // attenuation_low is constant
    EXPECT_FLOAT_EQ(low.attenuation_low, 0.02f);
    EXPECT_FLOAT_EQ(high.attenuation_low, 0.02f);

    // attenuation_high increases with quality
    EXPECT_LT(low.attenuation_high, high.attenuation_high);
}

TEST(MapSkinQualityTest, AttenuationHighBoundaries) {
    // skinQuality=1.0 -> attenuation_high = 0.10 + 1.0*0.10 = 0.20
    auto maximal = GPUBeautyBackend::mapSkinQuality(1.0f, 300);
    EXPECT_NEAR(maximal.attenuation_high, 0.20f, 0.01f);
}

TEST(MapSkinQualityTest, LowFreqSmoothRatioRange) {
    auto low = GPUBeautyBackend::mapSkinQuality(0.0001f, 300);
    auto high = GPUBeautyBackend::mapSkinQuality(1.0f, 300);

    // Ratio should be in [0.4, 0.6]
    EXPECT_GE(low.low_freq_smooth_radius_ratio, 0.4f);
    EXPECT_LE(low.low_freq_smooth_radius_ratio, 0.6f);
    EXPECT_GE(high.low_freq_smooth_radius_ratio, 0.4f);
    EXPECT_LE(high.low_freq_smooth_radius_ratio, 0.6f);
}

TEST(MapSkinQualityTest, SkinQualityClampedToOne) {
    // Values > 1.0 should be clamped internally by std::clamp
    auto over = GPUBeautyBackend::mapSkinQuality(2.0f, 300);
    auto at_one = GPUBeautyBackend::mapSkinQuality(1.0f, 300);

    EXPECT_TRUE(over.enabled);
    EXPECT_FLOAT_EQ(over.high_freq_preserve, at_one.high_freq_preserve);
    EXPECT_FLOAT_EQ(over.attenuation_high, at_one.attenuation_high);
}

TEST(MapSkinQualityTest, WorkPaperMappingTable_Face300) {
    // Verify mapping table from P4-W3-02 work paper (face_width=300px)

    // skinQuality=0.2 -> highFreqPreserve ~0.88
    auto q02 = GPUBeautyBackend::mapSkinQuality(0.2f, 300);
    EXPECT_EQ(q02.blur_radius, 15);
    EXPECT_NEAR(q02.high_freq_preserve, 0.88f, 0.05f);

    // skinQuality=0.6 -> highFreqPreserve ~0.50
    auto q06 = GPUBeautyBackend::mapSkinQuality(0.6f, 300);
    EXPECT_EQ(q06.blur_radius, 15);
    EXPECT_NEAR(q06.high_freq_preserve, 0.50f, 0.08f);

    // skinQuality=1.0 -> highFreqPreserve ~0.10
    auto q10 = GPUBeautyBackend::mapSkinQuality(1.0f, 300);
    EXPECT_EQ(q10.blur_radius, 15);
    EXPECT_NEAR(q10.high_freq_preserve, 0.10f, 0.01f);
}
```

**Estimated coverage gain**: Covers the entire `mapSkinQuality()` function including:
- Disable path (`skinQuality <= 0`)
- Smoothstep S-curve mapping
- `blur_radius` proportionality and clamping (min=6, max=28)
- `high_freq_preserve` monotonic decrease
- `attenuation` range progression
- Edge cases (negative face_width, zero face_width, over-range skinQuality)
- Work paper mapping table verification

---

### PRIORITY 2 -- HIGH (should-add before merge)

#### 2.2 `skinQuality` Validation and Clamping Tests

**Rationale**: The `isValid()` and `clamp()` functions already include `skinQuality` in their logic, but no explicit tests exercise these paths. Validation is critical for API safety.

```cpp
// File: cpp/tests/test_beauty_config_v2.cpp (append)

TEST(BeautyFilterConfigV2Test, IsValidRejectsSkinQualityOutOfRange) {
    auto config = BeautyFilterConfigV2Helper::defaults();

    config.skinQuality = 1.5f;  // Out of range (max 1.0)
    EXPECT_FALSE(BeautyFilterConfigV2Helper::isValid(config));

    config.skinQuality = -0.1f;  // Out of range (min 0.0)
    EXPECT_FALSE(BeautyFilterConfigV2Helper::isValid(config));
}

TEST(BeautyFilterConfigV2Test, IsValidAcceptsSkinQualityBoundaries) {
    auto config = BeautyFilterConfigV2Helper::defaults();

    config.skinQuality = 0.0f;
    EXPECT_TRUE(BeautyFilterConfigV2Helper::isValid(config));

    config.skinQuality = 1.0f;
    EXPECT_TRUE(BeautyFilterConfigV2Helper::isValid(config));

    config.skinQuality = 0.5f;
    EXPECT_TRUE(BeautyFilterConfigV2Helper::isValid(config));
}

TEST(BeautyFilterConfigV2Test, ClampCorrectsSkinQuality) {
    BeautyFilterConfigV2 config = BeautyFilterConfigV2Helper::defaults();
    config.skinQuality = 2.0f;

    BeautyFilterConfigV2Helper::clamp(config);

    EXPECT_FLOAT_EQ(config.skinQuality, 1.0f);

    config.skinQuality = -0.5f;
    BeautyFilterConfigV2Helper::clamp(config);

    EXPECT_FLOAT_EQ(config.skinQuality, 0.0f);
}
```

#### 2.3 `skinQuality` V1/V2 Conversion Tests

**Rationale**: The `fromV1()` function must set `skinQuality = 0.0f` (V1 has no equivalent). Verify this contract.

```cpp
// File: cpp/tests/test_beauty_config_v2.cpp (append)

TEST(BeautyFilterConfigV2Test, FromV1SetsSkinQualityToZero) {
    BeautyFilterConfig v1 = {};
    v1.enabled = true;
    v1.intensity = 0.7f;
    v1.smoothing = 0.6f;

    auto v2 = BeautyFilterConfigV2Helper::fromV1(v1);

    EXPECT_FLOAT_EQ(v2.skinQuality, 0.0f);
}
```

#### 2.4 `FreqSepParams` Default Values Test

**Rationale**: Verify that the `FreqSepParams` struct initializes with expected defaults.

```cpp
// File: cpp/tests/test_gpu_beauty_backend.cpp (append)

TEST(FreqSepParamsTest, DefaultValues) {
    GPUBeautyBackend::FreqSepParams params;

    EXPECT_EQ(params.blur_radius, 15);
    EXPECT_FLOAT_EQ(params.high_freq_preserve, 0.45f);
    EXPECT_FLOAT_EQ(params.low_freq_smooth_radius_ratio, 0.5f);
    EXPECT_FLOAT_EQ(params.attenuation_low, 0.02f);
    EXPECT_FLOAT_EQ(params.attenuation_high, 0.15f);
    EXPECT_FALSE(params.enabled);
}
```

---

### PRIORITY 3 -- MEDIUM (recommended)

#### 2.5 C API `skinQuality` Round-Trip Test

**Rationale**: Verify the C API `set/get` round-trip preserves the `skinQuality` value through the C-to-C++ conversion layer (`sdk_api_v2.cpp: toCppConfigV2/fromCppConfigV2`).

```cpp
// File: cpp/tests/test_beauty_config_v2.cpp (append to C API section)

TEST(BeautyFilterConfigV2CAPI, SetAndGetPreservesSkinQuality) {
    BeautyFilterConfigV2 set_config = {};
    iris_sdk_default_beauty_config_v2(&set_config);
    set_config.enabled = true;
    set_config.skinQuality = 0.7f;

    IrisSdkError err = iris_sdk_set_beauty_filter_v2(&set_config);
    EXPECT_EQ(err, IRIS_SDK_OK);

    BeautyFilterConfigV2 get_config = {};
    err = iris_sdk_get_beauty_filter_v2(&get_config);
    EXPECT_EQ(err, IRIS_SDK_OK);

    EXPECT_FLOAT_EQ(get_config.skinQuality, 0.7f);
}
```

Note: This test exercises the `BeautyFilterConfigV2Helper::isValid()` path including `skinQuality` range check, as `set_beauty_filter_v2` calls validation internally.

#### 2.6 `skinQuality` C API Validation Rejects Out-of-Range

```cpp
// File: cpp/tests/test_beauty_config_v2.cpp (append)

TEST(BeautyFilterConfigV2CAPI, SetConfigRejectsSkinQualityOutOfRange) {
    BeautyFilterConfigV2 config = {};
    iris_sdk_default_beauty_config_v2(&config);
    config.skinQuality = 1.5f;  // Out of range

    IrisSdkError err = iris_sdk_set_beauty_filter_v2(&config);
    EXPECT_EQ(err, IRIS_SDK_INVALID_PARAM);
}
```

#### 2.7 ABI Size Regression Test

**Rationale**: The breaking change analysis (02-breaking-changes.md) identifies a struct size change (52->56 bytes for C++ struct, 68->72 for C API struct). Adding a size assertion prevents future accidental layout changes.

```cpp
// File: cpp/tests/test_beauty_config_v2.cpp (append)

TEST(BeautyFilterConfigV2Test, StructSizeABI) {
    // ABI guard: detect accidental struct layout changes.
    // Update this value intentionally when adding new fields.
    // C++ struct: 56 bytes after skinQuality addition (P4-W3-02)
    EXPECT_EQ(sizeof(BeautyFilterConfigV2), 56u);
}
```

---

### PRIORITY 4 -- LOW (nice-to-have, future work)

#### 2.8 `executeSmoothingWithFallbackStrength()` Logic Test (Desktop Stub)

**Rationale**: The fallback logic (`max(config.smoothing, skinQuality * 0.5f)`) can be verified at the logic level even in desktop stub mode, though the actual GL calls are no-ops.

**Note**: The current desktop stub for `executeSmoothingWithFallbackStrength` is a no-op (`#else (void)...`). To test the fallback logic without GPU, the fallback strength calculation would need to be extracted into a separate testable function. This is a refactoring recommendation for improved testability, not a blocking item.

---

## 3. Tests That REQUIRE GPU (Android Device Only)

The following code paths cannot be meaningfully tested on desktop because they require a live OpenGL ES 3.1 context. These must be tested on an Android device as documented in the P4-W3-02 work paper.

| Code Path | Why GPU Required | Test Strategy |
|-----------|-----------------|---------------|
| `initializeFreqSepShaders()` | GLSL 310 es compilation + linking | Android instrumented test |
| `FREQ_SEP_GAUSSIAN_FRAGMENT` shader | GPU fragment shader execution | Visual regression test |
| `FREQ_SEP_COMPOSITE_FRAGMENT` shader | Multi-sampler fragment shader (4 texture units) | Visual regression test |
| `executeFreqSepPipeline()` 5 subpasses | FBO rendering, texture ping-pong | Android instrumented test with `GPUProfiler` |
| `uploadSkinMask()` (GL path) | `glTexImage2D` / `glTexSubImage2D` calls | Android instrumented test |
| `applyTextureId()` FreqSep branch | Full pipeline integration with TexturePool | Android end-to-end test |
| Composite shader program null check | `freq_sep_composite_program_ != 0` guard | Implicit in shader init failure path |
| Scissor disable fix | `glDisable(GL_SCISSOR_TEST)` | Visual regression (scissor clipping artifacts) |

**Recommended Android Test Plan** (from P4-W3-02 work paper section 1.1):

1. Load test image on Android device
2. Set `skinQuality = 0.5f` and verify Freq Sep pipeline activates (GPUProfiler traces)
3. Set `skinQuality = 0.0f` and verify Bilateral path activates
4. Force shader compilation failure and verify Bilateral fallback
5. Visual comparison: Freq Sep output vs Bilateral output
6. Performance check: Freq Sep 5-subpass within 33ms frame budget

---

## 4. Risk Assessment for Untested Paths

### 4.1 Risk Matrix

| Untested Path | Severity if Bug | Probability of Bug | Risk Score | Notes |
|---------------|----------------|-------------------|------------|-------|
| `mapSkinQuality()` edge cases | HIGH (wrong params -> visual artifacts) | MEDIUM | **HIGH** | Pure function, easily testable, no excuse |
| `skinQuality` validation boundary | MEDIUM (invalid config accepted) | LOW | MEDIUM | isValid() pattern is proven |
| C API `skinQuality` round-trip | MEDIUM (data loss in conversion) | LOW | MEDIUM | toCpp/fromCpp pattern is mechanical |
| `executeFreqSepPipeline()` null texture | HIGH (crash) | LOW | MEDIUM | Null checks present in code |
| `uploadSkinMask()` buffer mismatch | MEDIUM (GL error, blank mask) | LOW | LOW | Validation present at line 893 |
| `executeSmoothingWithFallbackStrength()` | LOW (slightly different smoothing) | LOW | LOW | Simple max() logic |
| Shader dynamic loop (GPU-specific) | HIGH (shader compile fail) | LOW-MEDIUM | MEDIUM | radius clamped to 28, but some GPUs may reject |
| ABI struct layout | HIGH (data corruption) | LOW (source build) | LOW | No pre-compiled consumers |

### 4.2 Highest Risk: `mapSkinQuality()` Without Tests

This is the single highest-risk untested path because:

1. **It is the control surface**: Every FreqSepParams value flows from this function
2. **It is public static**: Part of the API surface, callable by external test code
3. **It contains math**: Smoothstep S-curve, proportional scaling, clamping -- all error-prone
4. **It has no fallback**: Wrong parameters pass silently to the shader pipeline
5. **It is trivially testable**: Pure function, no dependencies, no setup required

**Recommendation**: Adding `mapSkinQuality()` tests is the single highest-value action for this PR. The test code provided in Section 2.1 covers 15 test cases and can be directly appended to `test_gpu_beauty_backend.cpp`.

### 4.3 Moderate Risk: Missing Fallback Logic Verification

The `executeSmoothingWithFallbackStrength()` function implements the policy:
```cpp
float effective_smoothing = config.skinQuality * 0.5f;
if (fallback_config.smoothing < effective_smoothing) {
    fallback_config.smoothing = effective_smoothing;
}
```

This is not tested anywhere. While the logic is simple, it is exercised in three distinct fallback paths:
- mask upload failure
- FreqSep pipeline failure (texture acquisition)
- skinQuality > 0 but no valid ROI

A bug here would cause inconsistent smoothing behavior across fallback scenarios. However, the risk is mitigated by the fact that all three paths call the same shared function.

---

## 5. Summary and Action Items

### Must-Do (Before Merge)

| # | Action | Files | Est. Effort | Test Count |
|---|--------|-------|-------------|------------|
| 1 | Add `mapSkinQuality()` unit tests | `test_gpu_beauty_backend.cpp` | 30 min | 15 tests |
| 2 | Add `skinQuality` validation/clamp tests | `test_beauty_config_v2.cpp` | 15 min | 4 tests |
| 3 | Add `FreqSepParams` default values test | `test_gpu_beauty_backend.cpp` | 5 min | 1 test |

### Should-Do (Before Merge or Immediately After)

| # | Action | Files | Est. Effort | Test Count |
|---|--------|-------|-------------|------------|
| 4 | Add `skinQuality` C API round-trip test | `test_beauty_config_v2.cpp` | 10 min | 2 tests |
| 5 | Add `skinQuality` V1->V2 conversion test | `test_beauty_config_v2.cpp` | 5 min | 1 test |
| 6 | Add ABI struct size regression test | `test_beauty_config_v2.cpp` | 5 min | 1 test |

### Deferred (Requires Android Device)

| # | Action | Environment | Notes |
|---|--------|-------------|-------|
| 7 | FreqSep pipeline end-to-end test | Android device + GLES 3.1 | See P4-W3-02 work paper section 1.1 |
| 8 | Shader compilation verification | Android device (multiple GPUs) | Mali, Adreno, PowerVR |
| 9 | Visual regression: FreqSep vs Bilateral | Android device + test images | Quality comparison, not functional |
| 10 | Performance validation (5-subpass < 33ms) | Android device + GPUProfiler | P4-W3-05 tuning scope |

### Total Desktop-Testable Coverage Gain

- **Current P4-W3-02 test count**: 1 (skinQuality default check)
- **After Priority 1-3**: 24 additional test cases
- **Coverage of new public API surface**: ~95% of `mapSkinQuality()`, 100% of `skinQuality` validation
- **Remaining untestable**: GPU pipeline execution, shader compilation (Android-only)
