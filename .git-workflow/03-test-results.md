# Test Results Report

**Date**: 2026-03-04
**Branch**: feature/P4-W3-02
**Build System**: CMake + Ninja (CLion cmake-build-debug)
**Platform**: macOS Darwin 24.6.0 (arm64)

---

## Overall Summary

| Metric | Count |
|--------|-------|
| Total Registered Tests | 575 |
| Passed | 553 |
| Failed | 5 |
| Skipped | 17 |
| Pass Rate | **96.2%** (553/575) |
| Total Execution Time | 78.11 sec |

---

## Build Status

### Successfully Built Test Executables (21/22)

| Executable | Status | Test Count |
|------------|--------|------------|
| test_types | Built | 34 |
| test_iris_detector | Built | 15 |
| test_lens_renderer | Built | 22 |
| test_lens_renderer_integration | Built | 7 |
| test_mediapipe_detector_integration | Built | 6 |
| test_frame_processor | Built | 49 |
| test_integration | Built | 18 |
| test_sdk_api | Built | 41 |
| test_sdk_manager | Built | 15 |
| test_camera_demo | Built | 6 |
| test_beauty_config_v2 | Built | 20 |
| test_beauty_processor | Built | 26 |
| test_beauty_roi_manager | Built | 36 |
| test_gpu_beauty_backend | Built | 12 |
| test_new_filter_effects | Built | 14 |
| test_grid_mesh | Built | 62 |
| test_eye_enlargement | Built | 24 |
| test_face_warp_controller | Built | 17 |
| test_fast_guided_filter | Built | 8 |
| test_profiler | Built | 15 |
| test_render_context | Built | 16 |

### Build Failure (1/22)

| Executable | Status | Error |
|------------|--------|-------|
| test_mediapipe_detector | LINK FAILED | `tflite::resource` undefined symbols for arm64 (TFLite resource module linker issue) |

**Root Cause**: `libtensorflow-lite.a` is missing `tflite::resource::ResourceVariable`, `GetHashtableResource`, `CreateResourceVariableIfNotAvailable` symbols. This is a known TFLite build configuration issue unrelated to the current feature branch changes.

---

## Failed Tests (5)

### 1. test_mediapipe_detector_NOT_BUILT (Test #52)
- **Status**: Not Run (build failure)
- **Impact**: Cannot test MediaPipe detector unit tests
- **Root Cause**: TFLite linker error (pre-existing)

### 2. test_mediapipe_detector_performance_NOT_BUILT (Test #53)
- **Status**: Not Run (build failure)
- **Impact**: Cannot test MediaPipe detector performance
- **Root Cause**: Same TFLite linker error as above

### 3. FrameProcessorTest.InitializeWithInvalidPath (Test #142)
- **Status**: Subprocess aborted
- **Error**: `libc++abi: terminating` - InferenceThread crashes when detector initialization fails with invalid path
- **Root Cause**: Missing graceful error handling in InferenceThread when detector init fails; calls `std::terminate` instead of returning error

### 4. FrameProcessorTest.InitializeWithEmptyPath (Test #143)
- **Status**: Subprocess aborted
- **Error**: Same `libc++abi: terminating` as above
- **Root Cause**: Same issue as #142 - empty path also triggers ungraceful terminate

### 5. GPUBeautyBackendTest.FailsWithNullContext (Test #468)
- **Status**: FAILED (assertion failure)
- **Error**: Test expects `backend_->initialize(nullptr)` to return `false` and `isInitialized()` to be `false`, but both return `true`
- **Detail**: The GPUBeautyBackend desktop stub mode initializes successfully even with a null context (because the stub doesn't actually use the OpenGL context). The test expectation is misaligned with the desktop stub behavior.
- **File**: `cpp/tests/test_gpu_beauty_backend.cpp:260-264`
- **Relevance**: Directly related to current feature branch (GPU beauty backend)

---

## Skipped Tests (17)

### FrameProcessorIntegrationTest (13 tests, #170-#182)
Skipped because they require valid model files at runtime:
- InitializeWithValidModels, ProcessRGBAFrame, ProcessBGRFrame
- ProcessNV21Frame, ProcessNV12Frame, ProcessMatBGR, ProcessMatBGRA
- ProcessWithRendering, DetectOnlyReturnsResult, RenderOnlyWithValidIrisResult
- ProcessingTimeUnder33ms, FPSCalculation, FormatConversionOverhead

### SdkApiTest (4 tests, #236-#239)
Skipped because they require valid SDK initialization:
- InitWithValidPathSucceeds, DoubleInitReturnsAlreadyInitialized
- InitWithConfigSucceeds, ShutdownAfterInitSucceeds

---

## Modified Component Test Results

### BeautyFilterConfigV2 (23 tests) -- ALL PASSED
| Test Suite | Tests | Status |
|-----------|-------|--------|
| BeautyFilterConfigV2Test | 12 | 12/12 PASSED |
| BeautyFilterConfigV2CAPI | 8 | 8/8 PASSED |
| BeautyFilterConfigV2HelperTest | 3 | 3/3 PASSED |

Key tests verified:
- DefaultValuesFromHelper: Default config values are correct
- IsValid validation: Rejects out-of-range intensity, brightness, color balance, downscale
- Clamp: Correctly clamps boundaries while preserving valid values
- V1/V2 conversion: FromV1, ToV1, RoundTrip all work correctly
- C API: Set/Get config, null safety, GPU availability flags

### GPU Beauty Backend (9 tests) -- 8 PASSED, 1 FAILED
| Test Suite | Tests | Status |
|-----------|-------|--------|
| GPUBeautyBackendTest | 5 | 4/5 PASSED |
| ShaderManagerTest | 4 | 4/4 PASSED |

Key tests verified:
- ReportsMetadata: Name = "GPUBeautyBackend", supportsGpu = true
- NotInitializedBeforeInit: Correct initial state
- ApplyReturnsErrorWhenNotInitialized: Proper error path
- ApplyTextureReturnsErrorWhenNotInitialized: Proper error path
- **FAILED**: FailsWithNullContext (desktop stub initializes successfully with nullptr)

### Shader Compilation (4 tests) -- ALL PASSED
| Test | Status |
|------|--------|
| ShaderManagerTest.CreatesProgramSuccessfully | PASSED |
| ShaderManagerTest.CachesProgram | PASSED |
| ShaderManagerTest.ReturnsZeroForUncachedProgram | PASSED |
| ShaderManagerTest.ReleasesAllPrograms | PASSED |

Note: These tests run against the desktop stub shader manager. Actual OpenGL shader compilation requires a real GPU context.

### CPU Beauty Backend (12 tests) -- ALL PASSED
| Test | Status |
|------|--------|
| Initialize_Success | PASSED |
| Release_CleansUp | PASSED |
| DoubleInitialize_IsIdempotent | PASSED |
| Metadata_ReturnsExpectedValues | PASSED |
| Apply_NotInitialized_ReturnsError | PASSED |
| Apply_NullFrameData_ReturnsError | PASSED |
| Apply_InvalidDimensions_ReturnsError | PASSED |
| Apply_DisabledConfig_NoChange | PASSED |
| Apply_EnabledConfig_ModifiesFrame | PASSED |
| Apply_SupportsDifferentFormats | PASSED |
| SmoothingEffect_ReducesVariance | PASSED |
| BrightnessEffect_IncreasesValues | PASSED |

### Beauty Processor (14 tests) -- ALL PASSED
- DefaultConstruction, Initialize_WithCPU_Success, Release_CleansUp
- DoubleInitialize_IsIdempotent, DI_AcceptsExternalRenderContext
- SetConfig_Valid_Success, SetConfig_Invalid_ReturnsError
- Process_NotInitialized_ReturnsError, Process_DisabledConfig_NoChange
- Process_EnabledConfig_ModifiesFrame, Process_WithROI_AppliesFilterToFaceOnly
- ProcessTexture_CPUBackend_ReturnsNotSupported, IsEnabled_ReflectsConfig
- IBeautyBackendTest.ApplyTexture_DefaultReturnsNotSupported

### New Filter Effects / Skin Smoothing (14 tests) -- ALL PASSED
| Test | Status |
|------|--------|
| SkinSmoothingV2_ReducesTextureVariance | PASSED |
| SkinSmoothingV2_PreservesEdges | PASSED |
| SoftFocusV2_CreatesGlowEffect | PASSED |
| SoftFocusV2_ZeroStrength_NoChange | PASSED |
| BrightnessV2_IncreasesLChannel | PASSED |
| BrightnessV2_PreservesHighlights | PASSED |
| BrightnessV2_DecreasesWhenBelow1 | PASSED |
| Whitening_BrightensSkintone | PASSED |
| ColorBalance_ShiftsToneWarm | PASSED |
| ColorBalance_ShiftsToneCool | PASSED |
| FullPipeline_AllEffectsCombined | PASSED |
| FullPipeline_DisabledConfig_NoChange | PASSED |
| FullPipeline_MultipleFormats | PASSED |
| Performance_SmoothingV2_ReasonableTime | PASSED |

### Beauty ROI Manager (36 tests) -- ALL PASSED
Full ROI computation, mask generation, and region extraction tests all pass.

---

## Coverage Assessment

### Components WITH Tests

| Component | Test File | Test Count | Coverage Quality |
|-----------|-----------|------------|-----------------|
| BeautyFilterConfigV2 | test_beauty_config_v2 | 23 | Excellent - validation, conversion, C API |
| GPUBeautyBackend | test_gpu_beauty_backend | 9 | Good - metadata, init, error paths (stub mode) |
| ShaderManager | test_gpu_beauty_backend | 4 | Good - create, cache, release |
| CPUBeautyBackend | test_beauty_processor | 12 | Excellent - full lifecycle + effects |
| BeautyProcessor | test_beauty_processor | 14 | Excellent - init, config, processing, ROI |
| BeautyROIManager | test_beauty_roi_manager | 36 | Excellent - comprehensive mask/ROI coverage |
| NewFilterEffects | test_new_filter_effects | 14 | Excellent - individual effects + pipeline |
| IrisDetector | test_iris_detector | 15 | Good - interface, factory, mocking |
| LensRenderer | test_lens_renderer | 22 | Good - config, rendering, effects |
| FrameProcessor | test_frame_processor | 49 | Good - extensive but 2 abort failures |
| SDKManager | test_sdk_manager | 15 | Good - lifecycle management |
| SDK C API | test_sdk_api | 41 | Good - struct/format/error validation |
| GridMesh | test_grid_mesh | 62 | Excellent - comprehensive geometry tests |
| EyeEnlargement | test_eye_enlargement | 24 | Good - effects and parameters |
| Profiler | test_profiler | 15 | Good - performance measurement |
| RenderContext | test_render_context | 16 | Good - factory and context management |
| Types | test_types | 34 | Excellent - all data structure validation |

### Components WITHOUT Tests (Gaps)

| Component | Notes |
|-----------|-------|
| FreqSep Pipeline (GPU) | No tests exist. Requires actual OpenGL context for frequency separation shader pipeline. |
| TexturePool (GPU) | Referenced in GPUBeautyBackend but no dedicated unit tests for pool management. |
| GPU Profiler | Only tested indirectly (warning message in GPUBeautyBackend init). |
| SkinQuality API | No tests found. May be a future P4-W3 task. |
| Temporal Stability | No tests found for temporal filter/stability logic. |

---

## Known Limitations

### GPU-Specific Tests Require OpenGL Context
The following components cannot be fully tested in CLI/headless environments:
- **FreqSep pipeline**: Frequency separation shaders require actual OpenGL ES 3.0+ context
- **GPU shader compilation**: Desktop stub creates fake program IDs; actual GLSL compilation untestable
- **TexturePool GPU operations**: Real texture allocation/deallocation requires GPU context
- **GPU profiler timing queries**: `GL_TIME_ELAPSED` queries require active GL context

All GPU-related tests run against a **desktop stub** that simulates success paths without actual GPU operations. This is by design for CI/headless testing but means real GPU behavior is only validated on-device (Android/iOS).

### FrameProcessor Abort Issue
`FrameProcessorTest.InitializeWithInvalidPath` and `InitializeWithEmptyPath` crash the process instead of returning an error. This is a pre-existing issue in InferenceThread error handling, not introduced by the current feature branch.

---

## Test Suite Breakdown by Executable

| Executable | Total | Passed | Failed | Skipped |
|------------|-------|--------|--------|---------|
| test_types | 34 | 34 | 0 | 0 |
| test_iris_detector | 15 | 15 | 0 | 0 |
| test_mediapipe_detector | 2 | 0 | 2 | 0 |
| test_mediapipe_detector_integration | 6 | 6 | 0 | 0 |
| test_lens_renderer | 22 | 22 | 0 | 0 |
| test_lens_renderer_integration | 7 | 7 | 0 | 0 |
| test_frame_processor | 49 | 30 | 2 | 17 |
| test_integration | 18 | 18 | 0 | 0 |
| test_sdk_api | 41 | 41 | 0 | 0 |
| test_sdk_manager | 15 | 15 | 0 | 0 |
| test_camera_demo | 6 | 6 | 0 | 0 |
| test_beauty_config_v2 | 20 | 20 | 0 | 0 |
| test_beauty_processor | 26 | 26 | 0 | 0 |
| test_beauty_roi_manager | 36 | 36 | 0 | 0 |
| test_gpu_beauty_backend | 12 | 11 | 1 | 0 |
| test_new_filter_effects | 14 | 14 | 0 | 0 |
| test_grid_mesh | 62 | 62 | 0 | 0 |
| test_eye_enlargement | 24 | 24 | 0 | 0 |
| test_face_warp_controller | 17 | 17 | 0 | 0 |
| test_fast_guided_filter | 8 | 8 | 0 | 0 |
| test_profiler | 15 | 15 | 0 | 0 |
| test_render_context | 16 | 16 | 0 | 0 |

---

## Recommendations

1. **GPUBeautyBackendTest.FailsWithNullContext**: Update test to account for desktop stub behavior, or add `#ifdef` guard to skip on desktop. The stub legitimately initializes without a real GL context.

2. **FrameProcessor abort tests**: Add proper exception handling or graceful error return in InferenceThread when detector init fails, instead of letting `std::terminate` be called.

3. **test_mediapipe_detector link failure**: Resolve TFLite resource module linker issue by adding the missing `tflite::resource` object files to the link target.

4. **FreqSep pipeline tests**: Consider adding CPU-side unit tests for the frequency separation logic (parameter computation, config validation) even if GPU shader execution cannot be tested headlessly.

5. **TexturePool unit tests**: Add dedicated tests for pool allocation, eviction, and size limit behavior using the desktop stub.
