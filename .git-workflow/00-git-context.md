# Git Context

## Branch
Current branch: `feature/P4-W3-02`
Target branch: `develop`

## Status
10 modified files, 5 untracked docs:
- android/iris-sdk/src/main/cpp/iris_jni.cpp (6 lines)
- android/iris-sdk/src/main/cpp/jni_utils.h (1 line)
- android/iris-sdk/src/main/java/com/irislenssdk/BeautyFilterConfigV2.java (18 lines)
- cpp/include/iris_sdk/beauty_filter.h (6 lines)
- cpp/include/iris_sdk/gpu/gpu_beauty_backend.h (72 lines)
- cpp/include/iris_sdk/sdk_api.h (1 line)
- cpp/src/gpu/gpu_beauty_backend.cpp (344 lines)
- cpp/src/gpu/shader_sources.cpp (80 lines)
- cpp/src/sdk_api_v2.cpp (3 lines)
- cpp/tests/test_beauty_config_v2.cpp (1 line)

Untracked:
- docs/workPaper/P4-W3-01_advanced_skin_smoothing_brainstorm.md
- docs/workPaper/P4-W3-02_freq_sep_implementation.md
- docs/workPaper/P4-W3-03_skinquality_api_mapping.md
- docs/workPaper/P4-W3-04_temporal_stability_device_tier.md
- docs/workPaper/P4-W3-05_tuning_test_release.md

## Diff Summary
Total: 528 insertions(+), 4 deletions(-)

## Recent Commits
```
a893fae feat(render): 비대칭 타원 Eye Mask — 16점 랜드마크 기반 눈매 클리핑 [P4-W2-02]
0cbf09e feat(render): Sclera Protection + Contact Shadow + Color Replace 증폭 제한 [P4-W2-01]
e8a7428 fix(stability): One Euro Filter 파라미터 튜닝 — 추적 반응성 향상 + 눈 감김 즉시 반응
c6befa1 Merge branch 'feature/P3-W1-03' into develop
a408992 플랜 생성 폴더 변경
```

## Change Description
Implementation of P4-W3-02: Frequency Separation GPU pipeline for advanced skin smoothing.

### Key Changes:
1. **GLSL Shaders** (shader_sources.cpp): Two new shaders - separable Gaussian blur and Composite (high-freq extraction + non-linear attenuation + mask blending)
2. **GPU Pipeline** (gpu_beauty_backend.h/cpp): 5-subpass FreqSep pipeline (GaussH→GaussV→LowSmoothH→LowSmoothV→Composite), skin mask upload, mapSkinQuality() S-curve mapping, Bilateral fallback on failure
3. **API Layer** (beauty_filter.h, sdk_api.h, sdk_api_v2.cpp): New `skinQuality` field (0~1) across C++, C API, JNI, Android Java
4. **Bug Fixes**:
   - Added freq_sep_composite_program_ check (prevent glUseProgram(0))
   - executeFreqSepPipeline returns bool for failure propagation
   - ROI scissor disabled during FreqSep multi-pass (Composite shader handles masking)
5. **Test**: skinQuality default value test added

## Full Diff
(See full diff in tool output - 528 insertions, 4 deletions across 10 files)
