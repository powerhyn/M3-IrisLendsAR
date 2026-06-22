# P8-W2 — 곁가지 뷰티 제거 (레거시 정리 + 2.0 ABI)

> 상태: 🔄 진행 중 (2026-06-20 착수) — A+B 커밋 완료, C/D 잔여
> 브랜치: `feature/p8-beauty` (develop `5bcdd94`에서 분기). A=ea0d7d6, B=912d682. 미푸시.
> 선행: P8 kickoff(`P8_kickoff.md`), P8-W1(skin smoothing 구현 완료)
> 진입 메모리: [[p8-facemesh478-substrate]], [[refactor-p2-adr-golden]]

## 1. 목표 (한 줄)
P8 뷰티 핵심 2개(① 피부 skin smoothing[구현됨] + ② 턱깎기 형태워프[스텁])만 남기고
**곁가지 전부 제거** → 종착 상태 = 뷰티 핵심 의존 config = `enabled` 게이트 + `use_skin_mask` 채널뿐.

## 2. 사용자 결정 (2026-06-20)
1. **제거 범위 = 전체 레거시**: 명시 곁가지 4종(LUT / FreqSep / 색보정 / vivid)에 더해,
   같은 레거시 스무딩 폴백 경로에 묶인 **Bilateral `smoothing` + `soft_focus` + 죽은 `wrinkle_remove`** 도 함께 제거.
   (use_skin_mask OFF 시 레거시 폴백을 남기지 않음 — "깨끗한 분리" 달성)
2. **ABI 전략 = 2.0 완전 제거**: 곁가지 config 필드를 IrisBeautyConfigV2 / BeautyFilterConfigV2에서
   **물리 삭제**(구조체 크기/offset 변경 = ABI break 허용). 근거: 외부 ABI 소비자 없음
   (iOS/Web 빈 스텁, Android SDK+데모는 함께 재빌드). **offsetof 가드 동시 갱신 필수**.

## 3. 보존 vs 제거 (config 필드 기준, IrisBeautyConfigV2 = sdk_api.h:590)
### 보존 (건드리지 않음)
- `enabled`, `intensity`, `brightness`(곁가지 아님 — 단순 밝기, COMBINED_COLOR 잔존),
  `slim_face`/`enlarge_eyes`/`thin_chin`(② 형태워프), `use_gpu`/`roi_only`/`protect_eyes`/
  `protect_lips`/`protect_nose`/`downscale_factor`/`feather_radius`(처리 옵션)
- ① skin smoothing 채널 전체: `use_skin_mask`, `iris_sdk_set_skin_mask_smoothing`,
  셰이더 SKIN_MASK_FILL/SEPARABLE_BLUR/SMOOTH_COMPOSITE, skin_mask_geometry — **불가침**
- ② grid_mesh / face_warp_controller substrate — **불가침**

### 제거 대상 (config 필드)
| 곁가지 | IrisBeautyConfigV2 필드 | BeautyFilterConfigV2(C++/Java) |
|---|---|---|
| ① LUT | (구조체 필드 없음 — apply 함수 인자 `lut_texture_id`/`lut_intensity`) | — |
| ② FreqSep | `skin_quality`, `smooth_intensity`, `pore_reduction` | skinQuality/smoothIntensity/poreReduction |
| ③ 색보정 | `whitening`, `color_balance` | whitening/colorBalance |
| ④ vivid | `vivid_intensity`/`_saturation`/`_brightness`/`_warmth` | vividIntensity/Saturation/Brightness/Warmth |
| 레거시 smoothing(확장) | `smoothing`(Bilateral), `soft_focus`, `wrinkle_remove`(죽은 필드) | smoothing/softFocus/wrinkleRemove |

### 제거 대상 C API 함수
- `iris_sdk_set_skin_color_filter(int)` (sdk_api.h:726)
- `iris_sdk_set_freqsep_debug_mode(int)` (sdk_api.h:720)
- `iris_sdk_apply_beauty_texture_v2`의 LUT 인자 2개 (시그니처 변경 = 2.0)
- 편의 API `iris_sdk_set_skin_quality`/`iris_sdk_get_skin_quality` (beauty_filter.h:298/306)

## 4. 인벤토리 (Explore 매핑, 2026-06-20 develop 기준 — 착수 시 line 재확인)
### 셰이더 (shader_sources.cpp / shader_manager.h)
- **전용 제거**: WHITENING_FRAGMENT(:138), COLOR_BALANCE_FRAGMENT(:200),
  FREQ_SEP_GAUSSIAN_FRAGMENT(:431), FREQ_SEP_COMPOSITE_FRAGMENT(:464),
  LUMINANCE_SHARPEN_FRAGMENT(:661), VIVID_POSTPROCESS_FRAGMENT(:719)
- **발췌 제거(공유 셰이더)**: COMBINED_COLOR_ADJUSTMENT_FRAGMENT(:319,387-390) — LUT 블록 +
  uLutTexture/uLutIntensity/uWhitening/uBalance 유니폼 제거, **brightness 부분 보존**
- soft_focus / smoothing(Bilateral) 전용 셰이더/프로그램(soft_focus_program_/smoothing_program_) 제거

### 결합 주의점 (보존 경로 회귀 위험)
1. COMBINED_COLOR_ADJUSTMENT = brightness(보존) + whitening(③) + LUT(①) 한 셰이더 → brightness만 남기고 발췌.
2. FREQ_SEP_COMPOSITE 안에 uSkinColorFilter(③) 박힘 → ②③ 동반 제거.
3. FreqSep+Bilateral 제거 후 use_skin_mask OFF = 스무딩 없음(의도된 종착). skin mask 경로 게이트 보존.
4. `!config.enabled && !needsVivid` early-return 게이트(sdk_api_v2.cpp:378, gpu:747/2026 등) → `!config.enabled`로 단순화.

### 레이어별 surface 요약 (상세는 인벤토리 보고서 — 본 문서 git 히스토리/세션 기록)
- C++ 헤더: sdk_api.h(590-625, 706-726), beauty_filter.h(78-122, 276-451)
- C++ 구현: sdk_api_v2.cpp(117-229, 359-503), cpu_beauty_backend.cpp(241-303, 722-803),
  gpu_beauty_backend.cpp(다수 — vivid/freqsep/whitening/colorbalance/lut/softfocus/smoothing 패스)
- JNI: iris_jni.cpp(165-195, 463-516, 1466-1538), jni_utils.h(필드ID 캐시)
- Java: IrisLensSDK.java(673-722, 790-803, 1458-1464), BeautyFilterConfigV2.java(필드/빌더/검증/clamp/toString)
- 데모 UI: GpuRenderActivity.kt(128-137, 500-669, 704-710, 763-877), BeautyPresetFactory.kt, CameraGLRenderer.kt/CameraGLView.kt(LUT 래퍼), LutTextureLoader.kt(전체 삭제)
- 부가 인프라(FreqSep 전용): quality_metrics.h, ab_compare.h, param_tuner.h, release_gate.h
- 테스트(stale): test_device_tier.cpp(FreqSepMappingTest), test_beauty_config_v2.cpp(FreqSepParamsTest/whitening/colorBalance/vivid),
  test_quality_tuning.cpp(ab_compare/param_tuner/release_gate FreqSep), test_new_filter_effects.cpp(whitening/colorBalance), test_gpu_beauty_backend.cpp(일부)

## 5. 제거 순서 (plan A→B→C→D) + 게이트
| 단계 | 내용 | 게이트 |
|---|---|---|
| **A** 데모 UI | GpuRenderActivity 버튼/슬라이더/핸들러, CameraGLRenderer/View LUT 래퍼, LutTextureLoader.kt 삭제, BeautyPresetFactory vivid/색보정 제거 | assembleDebug BUILD SUCCESSFUL |
| **B** GPU 패스 + stale 테스트 | gpu_beauty_backend(vivid→FreqSep→LUT→softfocus/smoothing), shader_sources 전용 셰이더 + COMBINED 발췌, 부가 인프라(quality/ab/tuner/gate), stale 테스트 제거 | 데스크톱 빌드 warn0 + ctest 회귀0(pre-existing 3 허용) |
| **C** CPU 색보정 + 골든 | cpu_beauty_backend whitening/colorBalance/softFocus, COMBINED whitening 블록, **beauty.png 4벌 골든 재캡처 + manifest** | 골든 재캡처+manifest, ctest 회귀0 |
| **D** 구조체 ABI 필드 | IrisBeautyConfigV2/BeautyFilterConfigV2 곁가지 필드 물리 삭제(2.0), JNI 필드캐시, Java 미러, **offsetof 가드 갱신**, C API 함수 제거 | 4면 미러 동기 + 빌드 + assembleDebug + offsetof PASS |

## 6. 절대 제약 (불변식 — kickoff §3)
1. 핵심 2개만 + 곁가지 제거. 새 곁가지 추가 금지.
2. grid_mesh substrate 보존(삭제 금지).
3. skin smoothing 독립 채널(use_skin_mask) 보존.
4. ~~ABI no-op~~ → **2.0 물리 제거 결정**(2026-06-20). offsetof 가드 동시 갱신.
5. CPU 색보정 제거 시 beauty.png 골든 재캡처 + manifest.
6. 이식 무관(제거 작업) — LensSimulator 정본은 ② 형태워프 단계에서 적용.

## 7. 게이트 (최종)
- 데스크톱 빌드 신규 warn0
- 골든: beauty.png 4벌 재캡처(W2-C) + manifest, 그 외 ε 일치
- ctest 회귀0 (pre-existing 3 허용: GPUBeautyBackendTest#401 + FreqSepMappingTest#587/588 — 단 FreqSep 테스트는 본 W에서 제거되므로 종료 시 pre-existing 목록 재정의)
- assembleDebug BUILD SUCCESSFUL
- 실기기 육안(밝은 환경, 본인 토글 체감) — 곁가지 제거가 핵심 2개에 무영향인지

## 8. 진행 상태
- [x] 인벤토리 + 스코핑 + 사용자 결정 2건
- [x] W2-A 데모 UI 제거 (assembleDebug SUCCESS)
- [x] W2-B GPU 패스 + stale 테스트 (빌드 신규경고0 + ctest 신규회귀0)
- [x] W2-C CPU 색보정 + 골든 재캡처 (빌드 신규경고0 + ctest 신규회귀0 + beauty.png 4벌 격리 재캡처 + manifest)
- [ ] W2-D 구조체 ABI 필드 (2.0) ← **다음**

## 9. 변경 이력
- 2026-06-20: 착수. 인벤토리(Explore) + 사용자 결정(전체 레거시 제거 / 2.0 ABI) + 본 문서 작성.
- 2026-06-20: **W2-A 완료** (미커밋). 데모 6파일: BeautyPresetFactory.kt(159줄)+LutTextureLoader.kt(346줄) 삭제,
  GpuRenderActivity.kt(~325줄 제거: 프리셋 시스템+곁가지 슬라이더+skinColorFilter+freqSepDebug+initSDK 곁가지리셋,
  btnToggleDebug→overlayView.debugMode 단순토글), CameraGLView/Renderer LUT 플러밍 제거(applyBeautyTextureV2 LUT 인자 0),
  activity_gpu_render.xml(~310줄). 보존: btnP8Skin/btnToggleBeauty/btnProtectNose. assembleDebug SUCCESSFUL.
  export.h 빌드 재생성 diff 되돌림. btnP8Skin 스테일 FreqSep 주석 수정.
- 2026-06-20: **W2-B 완료** (미커밋, cpp-pro 편집 + 메인세션 빌드/ctest 검증). 제거: 부가인프라 4모듈 통째
  (quality_metrics/ab_compare/param_tuner/release_gate .h+.cpp, CMakeLists 등록) — find_referencing_symbols로 비-FreqSep 소비처 0 확인,
  test_release_gate 바이너리 부재 확인. 테스트: test_quality_tuning.cpp 삭제 + FreqSepMappingTest(4) + FreqSepParamsTest(~28) 제거
  (DeviceTierTest 18·config필드 테스트·SkinQualityCAPITest 보존). GPU 백엔드: vivid/FreqSep/LUT/색보정(GPU)/레거시 smoothing·softFocus
  패스·셰이더·멤버 전부 제거(BILATERAL/WHITENING/COLOR_BALANCE/SOFT_FOCUS/FREQ_SEP_GAUSSIAN/FREQ_SEP_COMPOSITE/LUMINANCE_SHARPEN/VIVID),
  COMBINED_COLOR는 brightness-only 발췌, detectDeviceTier/device_tier_ 제거(classifyGpuRenderer 보존).
  C API 봉합(시그니처 D까지 유지): set_freqsep_debug_mode/set_skin_color_filter no-op, apply_beauty_texture_v2 LUT 인자 미전달.
  보존: config 필드 전체(D몫)·skin smoothing(use_skin_mask)·grid_mesh·brightness·CPU 백엔드(C몫)·JNI/Java(D몫).
  **게이트: 데스크톱 빌드 exit0 신규경고0(차집합 공집합) / ctest 591개 590통과, 유일 실패=GPUBeautyBackendTest.FailsWithNullContext(pre-existing 동일, 곁가지무관) — 신규 회귀0.**
  **pre-existing 재정의**: 종료 시점 pre-existing = GPUBeautyBackendTest.FailsWithNullContext 1건만(FreqSepMappingTest #587/588은 제거됨).
  **커밋**: 새 브랜치 `feature/p8-beauty`(develop 5bcdd94 분기)에 슬라이스별 2커밋 — A=ea0d7d6(데모 7파일), B=912d682(cpp 18파일, -5886줄). 미푸시.
  ⚠️ **휴식 지점**: 재개 = W2-C(CPU 색보정 applyWhitening/applyColorBalance/applySoftFocus + COMBINED CPU측 제거 + beauty.png 4벌 골든 재캡처).
- 2026-06-22: **W2-C 편집 완료** (cpp-pro 편집만 — 빌드/ctest/골든 재캡처는 메인세션). CPU 곁가지 EFFECT 제거:
  **cpu_beauty_backend.cpp/.h** — 삭제: applySkinSmoothing(2 오버로드, Bilateral V1)·applySoftFocus(V1)·
  applyWhitening(2 오버로드)·applyColorBalance(색보정)·applySkinSmoothingV2·applySoftFocusV2·applyBrightnessV2·
  applyWrinkleRemoval·overlayBlend·detectSkinTone·createWrinkleRegionMasks(전부 dead 또는 곁가지) + WrinkleRegions 구조체 +
  fast_guided_filter.h include(orphan). applyFullFrame=brightness-only, applyWithROI=brightness-only(ROI추출·feather·applyROIRegion 골격 보존,
  소비처 사라진 protection_mask 준비블록 제거→unused 회피). **생존: applyBrightness**. (find_referencing로 V2 5종+detectSkinTone dead 확인)
  **beauty_filter.cpp(레거시 V1 싱글톤)** — processBeautyEffect에서 effective_smoothing/effective_soft_focus 블록 제거(brightness만 유지),
  자체 applySkinSmoothing/applySoftFocus 정의 제거, MAX_BLUR_KERNEL_SIZE 주석처리(orphan). config 필드 참조(clamp/skinQuality API)는 미접촉(D몫).
  **테스트** — test_new_filter_effects.cpp: SkinSmoothingV2_*/SoftFocusV2_*/Whitening_*/ColorBalance_*/Performance_SmoothingV2_* 제거,
  BrightnessV2_*(3) 보존, FullPipeline_AllEffectsCombined→brightness-only 축소, MultipleFormats→brightness, DisabledConfig 보존, fast_guided_filter.h include 제거.
  test_beauty_processor.cpp(스코프 외였으나 빌드 회귀 방지로 처리): SmoothingEffect_ReducesVariance 제거,
  Apply_EnabledConfig_ModifiesFrame·Process_EnabledConfig_ModifiesFrame·Apply_SupportsDifferentFormats·Process_WithROI를 smoothing→brightness 전환
  (default brightness=1.0라 smoothing 제거 시 isFrameModified 실패하던 2건 수정). BrightnessEffect_IncreasesValues·SetConfig/Clamp/fromV1(config필드 roundtrip) 보존.
  **golden_capture.cpp** — fillBeautyConfig: c.smoothing=0.5/c.skin_quality=0.5 줄 제거, **c.brightness=1.2f**로 변경(곁가지 제거 후 CPU beauty 유일 생존효과=brightness 실측). beauty.png 4벌 재캡처는 메인세션.
  **잔여(D몫)**: config 필드 물리삭제(IrisBeautyConfigV2/BeautyFilterConfigV2)·clamp/reset/skinQuality 편의 C API·JNI/Java·offsetof 가드.
- 2026-06-22: **W2-C 메인세션 검증 완료**. 데스크톱 빌드 exit0, C 변경 파일(cpu_beauty_backend/beauty_filter/test_*) 신규 경고 0(유일 경고 fromCppConfigV2=pre-existing).
  ctest 582개 중 581통과, 유일 실패=GPUBeautyBackendTest.FailsWithNullContext(pre-existing 동일) — 신규 회귀0. **beauty.png 4벌 골든 재캡처**(INJECT_BASELINE 주입):
  격리 검증 PASS — beauty.png 4벌만 변경, JSON 18+render 15 = 33벌 byte-identical, 파일집합 동일. brightness=1.2 ROI 적용 정량확인(변경 5.2%, 변경영역 mean diff 17.3, ×1.2 정합).
  manifest=`cpp/tests/golden/P8-W2-C_RECAPTURE_MANIFEST.md`. test_beauty_processor.cpp 스코프외 변경(smoothing→brightness 4건+SmoothingEffect 제거)은 빌드회귀 방지로 수용(diff 검토 완료).
