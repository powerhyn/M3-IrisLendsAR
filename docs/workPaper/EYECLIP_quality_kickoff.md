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

---

## 7. 실행 결과 (2026-06-30 세션)

### A-1 (타원 마스크 native 배선) — ✅ 완료
- `CameraGLView.setEllipseMask` → `IrisLensSDK.setLensEllipseMask(enabled)` 직접 호출로 배선(기존 dead `glRenderer` 경로 대체). `ellipseOn` 멤버 승격 + `restoreLensRenderState()` 복원. dead `CameraGLRenderer.setEllipseMask`/`ellipseMaskEnabled` 제거.
- 검증: 빌드 OK, 호출처 0→1, 실기기 S23+ Ellipse OFF↔ON 토글 동작 확인. **커밋 `dd7bbfb`**.

### squint-freeze 버그 — A-1 검증 중 발견·해결 (예상 밖 핵심 성과)
- **증상**: 눈을 약간 감으면(squint) 렌즈가 화면에 박혀 머리를 못 따라옴. A-1(ellipse)과 무관(실기기 OFF/ON A/B로 확인).
- **사용자 통찰(결정적)**: "클리핑은 렌더인데 왜 트래킹이 끊겨?" → 정확. 눈꺼풀 클리핑(셰이더 Y-slab/ellipse)은 검출과 독립. 클리핑·박힘은 "눈 감음" 공통 원인의 상관일 뿐.
- **진단 경로(측정 게이트로 단계 확정)**:
  - dropout freeze(`temporal_stabilizer.cpp:207-217`, faces=0 시 전역 pose freeze) → numFaces=1로 faces=0 정량 0 됐으나 **체감 박힘 그대로 = 주범 아님**. numFaces=1은 러버밴딩(내부 One-Euro 부활) 부작용만 추가 → 원복.
  - rank1(FaceTracker presence/tracking 0.5→0.3) → 무력. conf가 detected→1.0/0.0 상수(`TasksToIrisResult.kt:120`)라 MediaPipe 내부 score 미관측 + numFaces=2가 tracking 경로 차단. 원복.
  - **진범 = blink-hold**(`temporal_stabilizer.cpp:159-200`): squint로 EAR<0.2면 blink 판정 → 홍채 좌표를 직전 값에 hold → 렌즈가 이전 위치에 박힘. 검출 끊김 아님(faces=1 유지)이라 numFaces로 안 고쳐짐(모순 해소).
- **처방(Option A)**: `blink_ear_threshold 0.2→0.05`. **실효 진입점은 `sdk_api.cpp:782` default config**(헤더 `temporal_stabilizer.h:45` 기본값은 nullptr 경로 전용 — cpp-pro 발견, [[verify-loadbearing-facts]] 사례). squint(EAR~0.1-0.15)엔 hold 미발동 → 홍채 라이브 추적, 거의 완전 감음(EAR<0.05)에만 hold. 눈꺼풀 가림은 렌더 클리핑이 처리.
- **검증**: 실기기 S23+ — squint 박힘 사라짐(렌즈가 눈동자 추적), 완전 깜빡임 튐은 수용 가능(클리핑이 대부분 가림).
- **안전성(cpp-pro 조사)**: blink-hold ≠ "per-eye held-pose" 불변식(후자는 렌더러 드롭아웃 hold `gpu_lens_renderer.cpp:999`, 트리거 `left_detected==false`로 별개 메커니즘). `blink_ramp_enabled_=false`/`detected`(render-only)/미러 X-flip 모두 불변. 완화 안전.
- **테스트**: `test_temporal_stability.cpp`에 squint(EAR~0.1-0.15) 회귀 케이스 추가 — 기존 `BlinkHoldCoordinatesStable`은 완전 blink(EAR~0.02)만 검증해 이 버그가 통과한 갭 메움.

### 곁가지 정리 / 보류
- numFaces, presence/tracking 임계 변경 → 전부 원복(blink 처방으로 불필요).
- Option B(blink-hold 완전 제거)는 깜빡임 튐 악화 위험으로 보류(필요 시 후속).

### 커밋/머지
- A-1: `dd7bbfb`. blink 처방: cpp 3파일(`sdk_api.h`/`sdk_api.cpp`/`temporal_stabilizer.h`) + squint 회귀 테스트. ~~머지 보류~~ → **develop 머지 완료(`4a25d1b`) + origin push**.

## 8. A-2 실행 결과 (2026-07-01~02 세션, 브랜치 `feat/eyeclip-clip-quality`)

### 판단 도구 — MaskDebug 오버레이 (`b8e2639`)
- 셰이더 경계선 방식은 실기기에서 "잘 안 보임"(사용자) → 폐기, **OverlayView CPU 오버레이**로 재구현: 초록=실제 눈꺼풀 16점 폴리곤, 시안=GPU fitEyeEllipse 화면좌표 복제 타원(비대칭+회전, 폴리라인). 기어 패널 MaskEdge 버튼.
- **판정(사용자, S23+)**: "초록이 확실히 타이트" → **A-2 contour 채택** + 방향 확정: S23+급=contour, 하위 티어=폴백(단 tier 바인딩은 실측 후 — kickoff §3 결정 준수).

### A-2 구현 — 16점 contour 폴리곤 마스크 (`fda6fb1`)
- 설계: 병렬 3차원 조사(셰이더/렌더러/표면) → 명세 확정 후 cpp-pro 구현. **fragment point-in-polygon**(IQ sdPolygon, crossing-parity=winding 무관이라 미러 안전), stencil 기각(EGL_STENCIL_SIZE=0).
- 셰이더: `uUseEllipseMask` 0/1/2 확장(기존 Y-slab/ellipse 문자 그대로 보존=폴백), `vec2 uL/REyeContour[16]` + `vec4 AABB×2`(페더 확장, invalid→Y-slab 폴백), **AABB early-out**(눈 밖 픽셀 비교 4회), 페더 ±4px `1-smoothstep(-f,+f,sd)` 정방향(역순 undefined 금지), 순수 ALU(Adreno §8.9 안전).
- 렌더러: `ContourCache/ContourFilters`(per-point OneEuro 4.0/15.0/1.0, 프레임당 ts 1회 공유) — `updateEllipseCache` 규약 복제(face_mesh_valid 게이트, EYELID_HOLD_FRAMES=5, per-eye held-pose 정합). 업로드 시 Y-flip→미러 X-flip→**aspect 사전곱(adjusted 공간, 등방 4px)**.
- API: 기존 `setLensEllipseMask(bool)` 불변(어댑터 mode 0/1). 신규 internal `setLensEyelidMaskMode(int)` — bench_toggles 패턴(C/JNI/Java/KT). 데모 Ellipse 버튼→**3-way 순환(Mask: Y-slab/Ellipse/Contour)** + `restoreLensRenderState` 재적용.
- 검증: ctest **572/572**(stale 단정 `30fbf9a` 수정 포함), 실기기 S23+ 셰이더 컴파일+**uniforms 45/45**, **contour 클립 시각 확인(사용자 "잘 먹네")**, 홈→복귀 Contour 유지.

### 트랙 종결 (2026-07-02) — 후속 트랙 후보로 이월
사용자 결정: 더 급한 이슈 우선으로 EYECLIP 트랙은 여기서 마무리. develop 머지. 아래 2건은 **후속 트랙 후보**로 남긴다:

1. **[후속 후보] A-3 완전 감음 잔여 띠** — contour 모드에서도 잔존 확인(실기기 S23+, 사용자). 수용하고 종결. 원인=MediaPipe 눈꺼풀 상/하 랜드마크가 완전 감음에도 gap을 남김(마스크 형상 무관, 입력 데이터 한계). ⚠️ 재도전 시 주의: alpha fade/blink-ramp는 기각 이력(eye_opening 0.005~0.02 노이즈, BUGFIX §3차) — 다른 접근 필요(예: EAR 기반 contour 폴리곤 강제 붕괴, gap 임계 시 상/하 꺼풀 스냅 등, 미검토).
2. **[후속 후보] A23(저티어) FPS 실측 + tier 폴백** — contour ON/OFF frame-time A/B 미실측(기기 상태로 세션 내 미완). kickoff §3 결정대로 **출시 시점에만** `when(gpuTier){HIGH→contour; else→ellipse/Y-slab}` 1줄 바인딩(실제 드롭 확인 시에만). 개발 중에는 3-way 수동 토글로 충분. AABB early-out이 있어 저티어도 거의 공짜일 가능성 높음(설계 분석).

- 머지: feat/eyeclip-clip-quality → develop `--no-ff` + push.
