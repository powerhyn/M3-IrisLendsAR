# 눈꺼풀 클리핑 품질 심화 — 타원 마스크 배선 착수 킥오프

> 이 문서 + §0 필독으로 **대화 맥락 0인 새 세션이 질문 없이 착수**. 작성: 2026-06-30.
> 직전 트랙(버그2 눈 클리핑)의 직접 후속. 1차 슬라이스 = dead 상태인 타원 마스크 배선 살리기.

## 0. 진입 절차
- **필독**: 이 문서 → `docs/workPaper/BUGFIX_resume_and_eye_clipping.md`(특히 "### 3차 수정(최종)" = 현재 클립 정책·불변식의 근거).
- **브랜치/HEAD**: `develop`, HEAD `138e3ca`(버그2 머지, origin/develop 동기화 완료). 새 작업은 새 브랜치 권장: `git checkout -b fix/eyeclip-ellipse`.
- **빌드**: 기존 `android/` gradle + `cpp/cmake-build-debug` 재사용(새 빌드 디렉토리 생성/ TFLite 재다운 **금지**). **1차(A-1)는 KT-only·native 불변** → `cd android && ./gradlew :demo-app:assembleDebug`(빠름). C++ 단계(A-2)만 native 재빌드.
- **실기기**: S23+ = `adb-R3CW20BBFCM-HZeo2S._adb-tls-connect._tcp`(SM-S916N). 화면 절전·잠금이 빨라, 테스트 전 `adb -s <S23> shell wm dismiss-keyguard` + 필요시 `settings put system screen_off_timeout 300000`(끝나면 원복). 렌즈는 새 프로세스마다 미선택 → 하단 렌즈바 탭(`input tap 580 2015` = claset doll choco) 후 `Lens applied` 로그 확인. APK: `android/build/modules/demo-app/outputs/apk/debug/demo-app-debug-b280.apk`.
- **C++ 작업 시**(A-2 contour 이식) `systems-programming:cpp-pro` 동반(프로젝트 규칙).

## 1. 목표 (한 줄) + 왜 지금
눈꺼풀 클립 품질 향상. **1차 = dead인 타원(ellipse) 마스크 토글을 native에 배선**해 실기기에서 ellipse vs Y-slab 육안 비교 → "완전 감음 잔여 띠 / 가로 곡선·안각 미추종"이 개선되는지 데이터 확보 후 다음 결정.
왜 지금: 버그2에서 "Y-slab 먼저"로 미룬 직접 후속. 배선이 사실상 1줄이라 **저비용으로 다음 방향(ellipse로 충분 vs contour 이식 필요)을 가른다**.

## 2. 현재 상태 — 검증된 사실 (2026-06-30 코드 직접 확인)
**타원 토글이 런타임 dead — native 경로는 완성됐는데 데모가 호출만 안 함:**
- 데모 KT 측 dead: `CameraGLRenderer.kt:253` `ellipseMaskEnabled=false`(기본) 선언, `:953-955` setter가 이 로컬 boolean만 set, **읽는 곳 0**(grep 결과 선언+setter 2곳뿐). `CameraGLView.kt:410-414` `setEllipseMask`는 `glRenderer.setEllipseMask(enabled)`만 호출(native 미호출).
- native 경로 완성·정상: `IrisLensSDK.java:1034 setLensEllipseMask(boolean)` → `iris_jni.cpp:2065 nativeSetLensEllipseMask` → `sdk_api_v2.cpp:697 g_gpu_lens->setEllipseMaskEnabled` → `gpu_lens_renderer.cpp:613 use_ellipse_mask_=enabled`(헤더 `:245` 기본 false) → `:1130 uUseEllipseMask` uniform → 셰이더 `asymmetricEllipseMask`. **`setLensEllipseMask` 호출처 = 0**(데모/SDK 어디서도 안 부름).
- 데모 UI 버튼 존재: `GpuRenderActivity.kt:225 btnToggleEllipse`, `:454-459` onClick → `cameraGLView.setEllipseMask(ellipseOn)`. `ellipseOn`은 `setupDebugControls` 내 **지역변수(멤버 아님)**.
- native-call 패턴 레퍼런스(따라 할 것): `CameraGLView.kt:297-299 setReflectionMode` = `queueEvent { IrisLensSDK.setReflectionMode(mode) }`.
- 셰이더 ellipse 한계(이전 조사, 미재확인): `asymmetricEllipseMask`는 `ry=max|Y|`라 반개안 추종 약함(`shader_sources.cpp` ~357-366, fit `gpu_lens_renderer.cpp` ~653-708). → **ellipse로도 부족하면 A-2 contour 이식**.

## 3. 할 일 / 결정

### ⭐ A-1 (이번 착수) — 타원 배선 살리기 (KT-only, native 불변)
1. `CameraGLView.kt:410-414 setEllipseMask`를 reflection 패턴으로 변경:
   `queueEvent { com.irislenssdk.IrisLensSDK.setLensEllipseMask(enabled) }` (현재 `glRenderer.setEllipseMask` 대체).
2. (권고) 컨텍스트 재생성 후 유지: `ellipseOn`을 GpuRenderActivity **멤버로 승격** + 직전 세션이 추가한 `restoreLensRenderState()`(GpuRenderActivity, onGpuInitialized 경유)에 `cameraGLView.setEllipseMask(ellipseOn)` 추가. (`use_ellipse_mask_`는 `releaseGpuLens()`로 리셋되므로 복귀 시 OFF로 돌아감)
3. (정리) dead `CameraGLRenderer.setEllipseMask`/`ellipseMaskEnabled`(:253,:953) 제거.
4. 빌드 → S23+ → **Ellipse 버튼 ON/OFF** 토글로 ellipse vs Y-slab 육안 비교(반쯤 감음 가로곡선·안각, 완전 감음 잔여 띠).

⚠️ **절대 제약(불변식) — 깨지 말 것**:
- 버그2 정책 보존: `gpu_lens_renderer.h blink_ramp_enabled_=false`(클립만으로 가시성), `GpuRenderActivity stabilizerHoldFrames=30`, per-eye held-pose, **public `detected` 의미 불변(render-only)**.
- 미러 = 렌더 X-flip 단일책임. **L/R eye-swap 재도입 금지**.
- **림발은 렌즈 에셋 책임** — 셰이더 절차 darkening 금지([[limbal-in-asset-not-shader]]).

### 후속 진행 후보 (A-1 실기기 결과에 따라 분기)
- **A-2 (ellipse 부족 시) — 눈 contour 폴리곤 클립**: `OverlayView.kt:820 buildEyePath`(16점 레퍼런스, 현재 런타임 dead) → GPU stencil 또는 프래그먼트 point-in-polygon으로 이식. C++(cpp-pro). **주의**: GPU는 좌표 Y-flip(1-y)+미러 X-flip 규약, CPU OverlayView는 다른 공간 → 좌표 규약 정합 필수. GPU측 `LEFT/RIGHT_EYE_CONTOUR` 16점이 이미 fitEyeEllipse 입력으로 들어오므로 새 랜드마크 플러밍 불필요.
  - **연산 비용 (마스크 = 풀스크린 쿼드 × 눈2 × 매프레임)**: Y-slab(~6~10 ALU)≈ellipse(~15~25 ALU) **둘 다 무시 가능 → 성능은 결정요소 아님, 보기로 선택**. contour는 16변 inside판정+페더(에지-거리/sqrt)로 마스크 수식만 ~150~300 ALU(≈20~40배). **단** 렌즈 셰이더는 이미 9-tap 디테일재주입 등 더 무거운 일을 해 contour조차 전체 프래그먼트의 소수 지분일 가능성↑. **필수 완화 = AABB early-out**(눈 바운딩박스 밖 픽셀은 16변 루프 통째 스킵 → 화면 대부분 비용 0); 페더 근사/stencil 별도패스도 대안.
  - **⚖️ tier-adaptive 폴백 = 측정 게이트 후에만(미리 분기 금지)**: 인프라는 이미 있음(`GpuRenderActivity.kt:736 detectGpuTier`/`gpuTier` + RAM 해상도 tier `:830-846`). 그러나 (a) ellipse로 충분하면 contour 자체 불요, (b) contour+AABB면 저티어도 거의 공짜 가능, (c) tier 정의 유동적([[mid-low-tier-2026-redefine]]), (d) 기기별 시각 불일치·QA비용 → **선설계 금지**. 순서: contour+AABB 구현 → **저티어 실기기 FPS/frame-time A/B 실측**(아래 §4) → **실제 프레임 드롭 시에만** `when(gpuTier){HIGH→contour; else→ellipse/Y-slab}` 폴백 추가(그 시점 1줄, 미루는 비용≈0). 개발 중엔 A-1 수동 토글로 A/B, 출시 시점에만 tier 바인딩.
- **A-3 — 완전 감음 잔여 띠**: MediaPipe 눈꺼풀 상/하 랜드마크 잔여 gap이 원인. ellipse/contour로 해소 안 되면 재검토. (eye_opening 기반 alpha fade = blink-ramp는 신호가 0.005~0.02로 작고 노이즈 커 임계값 불안정 → 이번 트랙에서 기각, 근거 `BUGFIX_resume_and_eye_clipping.md §3차`)
- **env png(낮은 우선순위 — 깨진 게 아님)**: `android/demo-app/src/main/assets/env/env_default_256x128.png`는 디스크 유효 PNG(256x128, `file`로 확인)이고 LFS OID 불변(`git lfs status` → `51192df → 51192df`). git M·`diff --stat Bin 5222→129`는 `*.png filter=lfs`(`.gitattributes:14`)의 LFS smudge/포인터 표시 아티팩트(콘텐츠 변화 없음). **별도 조치 불필요**. ⚠️ `git checkout`은 LFS 포인터로 덮을 수 있으니 함부로 하지 말 것(거슬리면 `git lfs checkout`으로만 정리).

### 로드맵 이월 (메모리 기반 — **착수 전 현행 상태 직접 확인 필수**, 스테일 가능)
- demo UI/셰이더 동기화 잔여: blend mode dropdown + KT fallback 셰이더 제거 ([[w9-demo-ui-sync]])
- P7 잔존 §8.9 GLSL spec cleanup 미할당분 ([[p7-residual-glsl-spec-cleanup-unassigned]])
- Phase 9 이월 트랙 W3/W4/W8 ([[phase9-deferred-tracks]])
- W4 환경 반사 재개(Phase7+ 진입점) / TintLinearV2 강도 재검토 ([[w4-env-reflection-deferred]], [[w5-b1-tintlinearv2-strength]])

## 4. 게이트/검증
- 빌드(A-1): `cd android && ./gradlew :demo-app:assembleDebug` → `BUILD SUCCESSFUL`.
- 배선 확인: `grep -rn "setLensEllipseMask" android/demo-app/src` → 호출처 **1+ (현재 0)**.
- 실기기: S23+ 설치 후 **Ellipse 버튼 토글 시 마스크 형상이 실제로 변함**(현재는 안 변함 = dead 증거).
- (A-2 시) **성능 게이트**: contour+AABB 적용 후 데모 HUD `Render FPS` + SDK `gpu_profiler`로 ON/OFF·저티어 기기 frame-time A/B 실측 → **드롭 확인된 경우에만** tier 폴백 추가.

## 5. 워킹트리 주의 — 커밋 금지 의도적 제외분 (정상 잔존, 중단 잔여물 아님)
직전 버그픽스 커밋(`ed6ceab`)에서 **의도적으로 뺀** 3개가 워킹트리에 남아있음:
- `.claude/settings.local.json` (로컬 권한 설정)
- `android/demo-app/src/main/assets/env/env_default_256x128.png` (LFS smudge 아티팩트 — 콘텐츠 불변, 깨짐 아님. §3 env png 항목 참조)
- `docs/plans/bubbly-prancing-boot.md` (이전 P8-W4 UI 플랜, untracked)

## 6. 완료 후
- 작업 문서(이 kickoff 또는 새 `EYECLIP_*` 문서) 갱신 + 분할 커밋.
- 머지: 직전 트랙처럼 `develop`에 `--no-ff` 직접 머지 + push (사용자 기본 방식). 또는 브랜치+PR — 사용자 확인.
- 메모리 갱신: [[bugfix-resume-eye-clipping]](후속 진행 반영), [[demo-ellipse-mask-dead-wiring]](배선 완료 시 dead 해소로 갱신/삭제).
