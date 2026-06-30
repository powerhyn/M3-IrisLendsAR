# BUGFIX — 백그라운드 복귀 블랙스크린 + 눈 부분 감음 시 렌즈 드롭

> 트랙: 리팩토링 종결(④) 이후 발견된 2개 실사용 이슈 수정.
> 방법: 6-에이전트 병렬조사 + 적대적 교차검증(Workflow) + 직접 코드확인 + **Codex 독립 교차검토** → 사용자 결정 3건 → 구현.

## 이슈

1. **백그라운드 복귀 시 화면이 안 나옴**(블랙/freeze).
2. **눈을 일부 감으면 렌즈가 통째로 사라짐** — 기대: 보이는 눈 영역에 **클리핑**되어 계속 렌더.

## 근본원인 (확정)

### 버그 1 — 복귀 블랙스크린
공통 트리거: `CameraGLView`에 `preserveEGLContextOnPause` 미설정 → 복귀 시 EGL 컨텍스트 파괴·재생성(`onSurfaceCreated` 재실행). 그 위에 4개 복원/안전 결함:

- **1A. 카메라 surface 핸드오프 stale**(전체 블랙): `isGLInitialized` 한 번 true 후 리셋 안 됨 + 옛 `cameraSurface`가 onPause에서 release 안 됨 → 복귀 시 CameraX 새 SurfaceRequest가 새 SurfaceTexture보다 먼저 오면 죽은 옛 surface를 provideSurface.
- **1A-2. stale GL id 오삭제**(추가 블랙 원인, Codex 발견 확증): `recreateIntermediateBuffers`/`createLensFbo`가 컨텍스트 손실 후 **stale ring/lens id를 먼저 glDelete** → 새 컨텍스트의 갓 생성된 텍스처(id 충돌)를 지움.
- **1B. 렌즈 텍스처 미복원**: `onSurfaceCreated`가 native lens는 release+init 하지만 KT `lensImageTextureId`를 0으로 안 내리고 비트맵 재업로드도 안 함 → stale-nonzero가 게이트 통과하나 native `lens_texture_=0` → `NoTextureLoaded`. 재업로드 트리거는 사용자 렌즈 재선택뿐.
- **1C. env_map 텍스처 + native 토글 미복원**: env_map은 GL 텍스처라 컨텍스트 손실 시 소실되나 `!envMapLoaded` 1회 가드로 재로드 안 됨. `releaseGpuLens()` 후 reflection/measured-luma/detail 등 native 멤버도 기본값으로 리셋되나 복원 안 함. (lens_meta는 코어 CPU 상태라 생존.)

### 버그 2 — 눈 부분 감음 렌즈 드롭
원하는 눈꺼풀 클리핑은 **이미 셰이더에 구현됨**(`shader_sources.cpp:461-474`, Y-slab `eyeTop/eyeBottom` 마스크). 문제는 "홍채 5점 전부 `[0,1]`" 검출 게이트가 끊겨 마스크 도달 전에 렌즈가 제거되는 것:

- 한쪽 눈 가림 → `left/right_detected=false` → **per-eye `uApply=0`**(`gpu_lens_renderer.cpp:1004/1012`, shader 606/613)로 그 눈 렌즈 통째 skip ← "한쪽 눈만 사라짐" 주범.
- **eyelid 캐시 갱신이 detected에 종속**(`updateEyelidCache:745/758`) → 검출 끊기면 클리핑 입력까지 stale → 5프레임 hold 후 full-open으로 클리핑 해제.
- `detected` OR early-return(`:885`)이 캐시 갱신(`:897`)보다 앞 + KT `slotDetected` 게이트(`CameraGLRenderer:404`).
- blink ramp(`:919-932`, `eye_opening<0.02`)는 완전 감음 fade-out(의도된 동작).
- **stabilizer 층**(`GpuRenderActivity:980-1007`)이 raw↔슬롯 사이에 있어 양안 모두 false면 최대 5프레임 hold하나, 한쪽만 false면 그 눈 detected 그대로 전파(stabilizer가 못 막음).

> Codex 교차검토 핵심 교정: ① 1A 수정 시 `preserveEGLContextOnPause=true`와 "onPause 무조건 invalidate"를 같이 넣으면 보존 성공 기기에서 영구 멈춤 → "보존=surface 유지 / 컨텍스트 손실 시에만 invalidate" 분기 필수. ② KT `lensImageTextureId`는 native 렌더 미사용 vestigial → 제거가 더 안전. ③ 어느 게이트가 "반쯤 감음"에서 실제 발화하는지 코드만으론 확정 불가 → 계측 우선.

## 사용자 결정

- 버그2 전략: **계측 우선** (실기기 측정 후 타깃 수정).
- 버그1B: **정리 리팩터** (미사용 KT GL 렌즈 텍스처 제거 + native 성공 boolean).
- 버그2 클리핑 품질: **Y-slab 먼저, 실기기 판단** (타원/contour는 후속).

## 구현 (✅ Phase 1 — demo-app Kotlin only, 네이티브 불변)

| # | 변경 | 파일 |
|---|------|------|
| 1A | `preserveEGLContextOnPause=true` + surface 충족을 GL 스레드로 이전(`fulfillPendingSurfaceRequest`), 최신 pending 1건 + `willNotProvideSurface`로 중복 정리 + 취소 리스너, 구 surface는 onSurfaceTextureAvailable에서 release | `CameraGLView.kt` |
| 1A-2 | `markGlHandlesStale()` — onSurfaceCreated 최상단에서 모든 KT GL 핸들(oes/program/quad/ring/lensFbo/...)을 0으로 리셋 + 구 SurfaceTexture release → stale id 오삭제 차단 | `CameraGLRenderer.kt` |
| 1B | KT `lensImageTextureId` 제거 → `nativeLensLoaded` boolean 게이트. `uploadPendingLensTexture`는 native `loadLensTexture`만(성공=OK일 때 true), GPU lens 미초기화면 비트맵 유지·다음 프레임 재시도 | `CameraGLRenderer.kt` |
| 1C | `onGpuInitialized`에서 env_map 매번 재로드(1회 가드 제거) + `restoreLensRenderState()`로 reflection/measured-luma/detail + 현재 선택 렌즈 텍스처 재적용 | `GpuRenderActivity.kt` |
| 2-계측 | `LensEyeDiag` 로그 — stabilize 전/후 per-eye detected + 홍채 중심(매 6프레임). 한쪽 눈 반쯤 감았을 때 stab[해당눈]=false면 검출드롭 경로, true 유지면 native blink-alpha 경로 | `GpuRenderActivity.kt` |

| 1A-3 | **실기기 검증 중 발견한 회귀**: `preserveEGLContextOnPause=true`로 컨텍스트 보존 시 `onSurfaceCreated` 생략 → `onSurfaceChanged`가 ring 버퍼를 **뷰 크기**(1080x2140)로 재생성하는데 `setFrameSize`는 frameWidth 불변(960==960)이라 no-op → ring이 뷰 종횡비에 갇혀 화면 깨짐(상단 블랙+하단 압축). 수정: `onSurfaceChanged`가 알려진 카메라 프레임 크기(frameWidth/frameHeight)로 ring 재생성, 최초엔 뷰 크기 fallback | `CameraGLRenderer.kt` |

빌드: `./gradlew :demo-app:assembleDebug` ✅ BUILD SUCCESSFUL.
APK: `android/build/modules/demo-app/outputs/apk/debug/demo-app-debug-b280.apk`.

### 실기기 검증 (S23+ / SM-S916N)
- 설치·실행 정상, 풀스크린 카메라·양안 검출·크래시 없음.
- **HOME→복귀(보존 컨텍스트 경로) 검증**: 수정 전 상단 블랙+하단 압축 → 수정 후(1A-3) ring 960x720·**풀스크린 정상 복원** 확인.
- 미검증: 컨텍스트 완전 손실 경로(장시간 백그라운드/메모리 압박 → `onSurfaceCreated` 재실행 → `markGlHandlesStale`+카메라 재bind+렌즈/env 복원). "Don't keep activities" 강제 또는 사용자 실사용으로 추가 확인 권장.

## Phase 2 — 버그2 실기기 측정 + 수정 (진행)

### 측정 결과 (S23+ LensEyeDiag, 왼쪽 눈 반쯤 감기)
- `stab[L=false R=true]`(단일 눈 드롭, 홍채 중심 범위밖 ~1.07) 소수 + **`stab[L=false R=false]` + 중심 (0,0)(= 얼굴 전체 미검출 fillNoFace)이 ~6 diag줄(≈1.2s/36프레임) 지배적**.
- 즉 "홍채만 드롭"이 아니라 **반쯤 감으면 MediaPipe가 얼굴 전체를 잃는다**(특히 비정면 포즈).

### 1차 수정 (C++ render-only) — 단독으로는 불충분
gpu_lens_renderer.cpp/.h: per-eye held pose 캐시 + 눈꺼풀 캐시를 face_mesh_valid 기준 갱신 + 캐시 갱신을 early-return 앞으로 + per-eye uApply=검출||held + early-return 완화. (cpp-pro 구현, 빌드·실기기 배포 완료) → **여전히 사라짐**.
- 원인: 이 수정은 `renderToTexture`가 **호출돼야** 실행되는데, 그 앞 KT 게이트 `if (lensEnabled && nativeLensLoaded && slotDetected)`(CameraGLRenderer.kt)가 **slotDetected=false면 호출 자체를 차단** → dead code. 그리고 slotDetected=stabilize 후 detected. tasks stabilizer hold가 **5프레임(~167ms)**뿐(`temporal_stabilizer.h:35`, `sdk_api.cpp:779`)이라 36프레임 dropout을 못 버티고 곧 detected=false → 게이트 닫힘.

### 2차 수정 (핵심) — stabilizer dropout hold 연장
- 검증: `temporal_stabilizer.cpp:211-217` dropout hold는 `result.stabilized = last_valid_iris_`로 **478점 face_mesh + face_mesh_valid=true + detected=true + iris**를 일관되게 hold. hold_frames를 늘리면 그 held 상태가 dropout 내내 슬롯에 유지 → slotDetected=true(게이트 통과) → C++가 face_mesh_valid=true를 받아 눈꺼풀 클립까지 신선 유지. **한 레버로 게이트+클립 동시 해결**.
- 구현(scoped, 코어 default·LEGACY 불변): `nativeCreateStabilizerWithHold(int)` JNI 오버로드(iris_jni.cpp) + `createStabilizer(int holdFrames)`(IrisLensSDK.java) + 데모 `createStabilizer(stabilizerHoldFrames=30)`(GpuRenderActivity.kt). C API는 이미 hold_frames 수용(신규 C++ 코어 변경 없음).
- step2(KT 게이트 `slotDetected`→`detectionHandle!=0` 완화)는 **불채택**: hold 연장으로 detected=true가 유지되면 게이트가 그대로 통과하고, 오히려 "얼굴 진짜 없음" 구간 passthrough를 applyGpuLensRenderer가 렌더실패로 오보고하는 부작용 → 제외.
- 빌드 ✅(libiris_jni.so에 새 심볼 익스포트 확인), S23+ 재배포 완료.
- **상태: 실기기 시각 검증 대기**(장비 화면 off로 이번 세션 미확인). 사용자 재검증 필요: 렌즈 선택 후 왼쪽 눈 반쯤 감아 렌즈 유지/클립 확인.
- **튜닝**: `stabilizerHoldFrames=30`(~1s @30fps). 얼굴 실제로 떠난 뒤 잔상이면 ↓, 더 긴 감음에도 사라지면 ↑.
- **한계(물리적)**: dropout 동안 클립은 **마지막 유효(대개 열린) 눈 형상으로 동결** — 랜드마크가 없어 점점 감기는 형상을 실시간 추종 못 함. "제자리 유지"이지 "실시간 점진 클리핑"은 아님. 진짜 점진 클리핑은 얼굴이 검출 유지돼야 가능(MediaPipe 포즈 강건성 문제).

### 3차 수정 (최종) — blink-ramp(알파 fade) OFF
2차(hold=30) 적용 후에도 "여전히 사라짐". 추가 계측(`[blink] eyeOpening` 로그)으로 확정:
- hold=30은 **정상 작동**(슬롯 `stab[L=true R=true]` + 실제 held 좌표 유지, 위치/클립 유지됨).
- 진짜 범인은 **C++ blink-ramp(render_alpha fade)**: 얼굴 손실 직전 작은 eye_opening이 hold 동안 alpha를 0으로 fade → 위치는 맞는데 투명해져 사라짐.
- **실험(blink-ramp OFF, 가시성=eyelidMask만)**: 실기기 정면 포즈에서 ✅ **반쯤 감음 = 깔끔히 클립**(사용자 확인). 단 완전 감음 시 MediaPipe 눈꺼풀 잔여 gap으로 **얇은 띠 잔존**(경미).
- fade 재활성+임계값 튜닝 시도 → eye_opening 실측이 0.005~0.012(open ~0.016~0.02)로 **작고 노이즈(±0.002~0.003) 커 임계값 안정화 불가**(뜬 눈도 투명해짐). → **fade OFF로 확정**(사용자 동의 + 기술적 타당). 완전 감음 잔여 띠는 MediaPipe 한계로 수용.
- 구현: `gpu_lens_renderer.h` `blink_ramp_enabled_=false`(토글 보존, 재활성 가능), 가시성은 셰이더 Y-slab 클립만. 빌드 ✅ S23+ 배포 ✅(force-stop 후 새 APK 로드 확인, pid 32065, [blink] 로그 제거).

**버그2 최종 동작**: hold=30(위치 유지) + C++ held-pose/eyelid 캐시 분리(per-eye 클립) + blink-ramp OFF(클립만). → 반쯤 감음=클립 유지, 완전 감음=얇은 띠(수용), 얼굴 손실(턴드 포즈)=~1s 제자리 hold.
**잔여 정리(미완)**: `LensEyeDiag`(Kotlin) 계측 로그 제거 + 두 버그 커밋 — 사용자 최종 확인 후.

## (구) 다음 단계 메모 — Phase 2 측정 전 작성, 위 결과로 일부 대체됨

1. 실기기에서 한쪽 눈 반쯤 감고 `adb logcat -s LensEyeDiag` 판독:
   - **stab[해당눈]=false 로 떨어짐** → 검출 드롭 경로. 수정: `extractIris`/`deriveEye`의 strict 5점 AND 완화(중심 valid + radius sane + boundary valid subset, 직전값 hold/clamp) **또는** per-eye 렌더를 검출이 아닌 center+radius(또는 eyelid 캐시) 기준으로 분리. public `detected` 의미 변경은 최후 단계.
   - **stab[해당눈]=true 유지인데 렌즈 사라짐** → native blink-alpha(`kBlinkCloseThreshold=0.02`) 또는 eyelid 캐시 hold 경로. 수정: 임계값 재조율 / eyelid 캐시를 `face_mesh_valid` 기준으로 갱신.
2. 공통: `updateEyelidCache`/`updateEllipseCache`를 `detected` early-return(`gpu_lens_renderer.cpp:885`) **앞**으로 이동(검출 끊겨도 클리핑 입력 갱신). C++ 작업이므로 `systems-programming:cpp-pro` 동반.
3. 계측 로그(`LensEyeDiag`, `lensEyeDiagEnabled`) 제거.

## 미해결/후속 (Codex 발견 latent)

- 데모 `setEllipseMask()`가 native `setLensEllipseMask()`를 호출 안 해 C++ 타원 마스크 토글이 **dead** — 버그2 클리핑 품질 후속 시 우선 배선 필요.
- `onGpuInitialized(success)`의 `success`는 beauty init만 반영(lens init 미포함) — 복원은 `isGpuLensInitialized()` 게이트로 보강됨.
