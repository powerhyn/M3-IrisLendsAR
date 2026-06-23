# P8-W3 — 피부 화사함(radiance, soft-glow) 이식 = ① skin smoothing 마무리

> 상태: 🔄 진행 중 (2026-06-22 착수)
> 브랜치: `feature/p8-beauty` (W2 곁가지 제거 위에 이어서)
> 성격: **① 마무리** — P8-W1 radiance 잔여 + skin smoothing 실기기 검증 2항을 함께 매듭.
> 출처 스펙: `docs/lenssim-handoff/radiance-skin-tone-handoff-from-lenssimulator.md` (LensSim S23+ 기본 0.40 확정 + 적대리뷰 통과)

## 1. 목표 (한 줄)
LensSim 검증된 soft-glow radiance(윤기/화사함)를 skin smoothing composite에 이식 —
**기존 skin 블러+마스크를 bloom 소스로 재활용**(새 패스 0, 새 fetch 0).

## 2. 핵심 발견 — 핸드오프 재좌표화 (W2 영향)
핸드오프(2026-06-18, W2 이전)는 "FreqSep 블러를 bloom 소스로 재활용"을 전제했으나 **W2-B에서 FreqSep 제거**.
현재 skin smoothing 경로(`SKIN_SMOOTH_COMPOSITE`)가 동일 입력(base/blur/mask)을 이미 보유 →
bloom 소스 = **skin 컬러 블러(uBlurTex)**. 알고리즘·상수 변경 없이 이식 가능.

## 3. 사용자 결정 (2026-06-22)
1. **독립 게이팅**: skin 경로가 `(smoothing>0 OR radiance>0)`이면 실행. smoothing=0이어도 radiance 단독 가능(윤기만).
2. **demo 토글**: `btnP8Radiance` off → 0.40 → 0.60 (기본 0.40 핸드오프 확정, 0.60=§7 cap 권고선).

## 4. 통합 지점 (실측 확정, 2026-06-22)
| 항목 | 위치/방식 |
|---|---|
| 셰이더 | `SKIN_SMOOTH_COMPOSITE_FRAGMENT`(shader_sources.cpp:210-231) line 229(`base = mix(...smoothed...)`) **직후**에 `uniform float uRadiance;` + 핸드오프 §3 step⑥ 블록 삽입(`if(uRadiance>0 && mask>0){...}`). base/blur/mask 그대로 사용. |
| 비용 게이트 | `skinMaskSmoothingActive`(gpu_beauty_backend.cpp:857) `skin_mask_smoothing_strength_ > 0` → `(skin_mask_smoothing_strength_ > 0 \|\| skin_radiance_strength_ > 0)`. (의미=skin mask 경로 활성) |
| 유니폼 | skin composite 유니폼 캐시에 uRadiance location 추가 + renderSkinComposite(1074)에서 glUniform1f(uRadiance, skin_radiance_strength_) |
| 백엔드 멤버 | `float skin_radiance_strength_ = 0.0f` + `void setSkinRadiance(float)` (gpu_beauty_backend.h, setSkinMaskSmoothing 인접) |
| C API | 신규 internal `iris_sdk_set_skin_radiance(float strength)` (sdk_api.h internal 구역 + sdk_api_v2.cpp, set_skin_mask_smoothing 미러) |
| JNI | `nativeSetSkinRadiance(float)` (iris_jni.cpp + jni_utils 불필요 — 필드캐시 아님) |
| Java | `IrisLensSDK.setSkinRadiance(float)` (setSkinMaskSmoothing 미러) |
| 데모 | `btnP8Radiance` 토글(off→0.40→0.60), GpuRenderActivity + CameraGLView.setSkinRadiance 래퍼 (btnP8Skin 패턴) |

## 5. 확정 상수 (핸드오프 §4 — 재논의 금지, LensSim S23+ 검증)
| 상수 | 값 | 의미 |
|---|---|---|
| 기본 `uRadiance` | 0.40 | 데모 기본·자연 톤업 |
| RAD_GLOW | 0.20 | screen-bloom 윤기 |
| RAD_LIFT | 0.08 | 미드톤 화사 |
| RAD_DESAT | 0.10 | 채도감소(맑은 톤, ≤0.16) |
| RAD_KNEE | 0.78 | 하이라이트 보호 |
| RAD_WARM | 0.012 | 웜 바이어스(혈색) |
luma = sRGB 0.299/0.587/0.114. 셰이더 GLSL은 핸드오프 §3 그대로 이식.

## 6. 불변/무영향
- **렌즈 무영향**: radiance는 `mask>0`(눈 제외) 게이팅 + `base = mix(base, even, mask*rad)`. P8-W1 마스크가 눈 contour 제외 확인 + 렌즈 먼저→뷰티 나중 구조 → 렌즈 출력 불변(핸드오프 §5).
- **골든 재캡처 불필요**: radiance = GPU(SKIN_SMOOTH_COMPOSITE) 경로, 데스크톱 beauty.png 골든 = CPU 경로. 무관.
- **비용**: radiance ON이면 composite 내 ~24 ALU 추가(블러 공짜 재활용). skin 경로 OFF(둘 다 0)면 전부 생략.
- **brightness/형태워프/skin smoothing 자체** 무손상.

## 7. 이월 (핸드오프 §7, 저위험 — 실기기 후 판단)
- dark-skin face-relative 캘리브레이션(midW/hiRoll knee가 절대 luma): 기본 0.40 문제없음, 절대-luma 유지.
- radiance 극단(1.0) halo cap: 데모 상한 0.60이라 무관, 정식 승격 시 재검토.

## 8. DoD / 게이트
- [x] 셰이더 uRadiance + step⑥ + 게이트 독립화 + 유니폼 배선 (2026-06-22 구현)
- [x] C API + JNI + Java + demo 토글 (4면) (2026-06-22 구현 — 메인 세션 빌드 검증 대기)
- [x] 데스크톱 빌드 신규 warn0 + ctest 564개 563통과(pre-existing GPUBeautyBackendTest.FailsWithNullContext만, 신규 회귀0)
- [x] assembleDebug BUILD SUCCESSFUL (iris-sdk+demo)
- [x] 🔴 **실기기 육안(S23+ SM-S916N, 2026-06-22 사용자 "다 잘 되네")**: radiance 룩(윤기/화사) + 렌즈 무영향 + skin smoothing 무회귀 + W2 곁가지 무회귀 통과. APK=installDebug 최신(76f4305).
- [ ] 실기기 통과 후 W2+W3 함께 develop 머지 ← **다음 (사용자 확인 대기)**

## 9. 브레인스토밍 생략 사유
P8-W1 선례 — 알고리즘·파라미터·함정이 LensSim 실기기(S23+ 0.40)+적대리뷰로 확정. 본 W 결정=통합 지점만(§3·§4, 코드 실측 기반).

## 변경 이력
- 2026-06-22: 착수. radiance 핸드오프 정독 + P8-W1 substrate 실측 + 통합 지점 확정(§4) + 사용자 결정 2건(독립 게이팅/토글 0.4→0.6). bloom 소스를 FreqSep→skin 블러로 재좌표화(W2 영향). 구현 착수.
- 2026-06-22: 배선 구현 완료(cpp-pro, 편집만 — 빌드/ctest/gradle 미실행, 메인 세션 검증 대기).
  - 셰이더(shader_sources.cpp): `uniform float uRadiance;` 추가 + line 229(스무딩 mix) 직후 radiance step⑥ 블록 삽입(핸드오프 상수 그대로). fragColor 유지.
  - 백엔드(gpu_beauty_backend.h): `skin_radiance_strength_` 멤버 + `setSkinRadiance`/`getSkinRadianceStrength` + SkinUniforms.compositeRadiance.
  - 게이트 독립화(gpu_beauty_backend.cpp:862 skinMaskSmoothingActive): `smoothing>0` → `(smoothing>0 || radiance>0)`. (enabled/detected/face_mesh_valid/program!=0 유지.)
  - 유니폼(gpu_beauty_backend.cpp): cacheUniformLocations에 uRadiance location + renderSkinComposite에 glUniform1f(compositeRadiance, skin_radiance_strength_).
  - C API: sdk_api.h `iris_sdk_set_skin_radiance(float)` 선언 + sdk_api_v2.cpp 구현(set_skin_mask_smoothing 미러, g_gpu_mutex/initialized 가드 동일).
  - JNI: iris_jni.cpp `Java_..._nativeSetSkinRadiance`. Java: IrisLensSDK.setSkinRadiance + nativeSetSkinRadiance 선언.
  - 데모: CameraGLView.setSkinRadiance(queueEvent) 래퍼 + GpuRenderActivity btnP8Radiance(off→0.40→0.60) + layout XML btnP8Radiance 위젯(btnP8Skin 복제).
  - 보존: std::clamp 대신 헤더 기존 수동 clamp 관용구 사용(<algorithm> 미포함 회피). 비-GPU 빌드 영향 없음(GL 호출 전부 #if IRIS_SDK_GPU_AVAILABLE 내부).
- 2026-06-22: **메인세션 게이트 수정 + 검증 완료**. cpp-pro 원안은 게이트가 `enabled_ && (smoothing>0 || radiance>0)`라 enabled_가 OR 밖=radiance도 enabled_ 요구(사용자 "독립" 결정 미충족) → **`(enabled_ && smoothing>0) || radiance>0`로 재구조화**(skinMaskSmoothingActive:862, setSkinRadiance가 strength만 설정 → radiance>0 단독 활성). 데모 onClick 주석도 정정(btnP8Skin off여도 radiance 단독 적용, Beauty 토글만 ON 필요).
  **게이트 검증**: cpp 빌드 exit0 신규경고0(radiance 파일 0, skin_target_* 잔존=pre-existing) / ctest 564개 563통과(유일 실패=GPUBeautyBackendTest.FailsWithNullContext=pre-existing, 신규 회귀0) / assembleDebug(iris-sdk+demo) BUILD SUCCESSFUL. 셰이더 블록 배치·상수·유니폼 캐시(289)+전달(1090)·게이트(865) 독립 검증 통과. 🔴 실기기 육안만 잔여(사용자).
