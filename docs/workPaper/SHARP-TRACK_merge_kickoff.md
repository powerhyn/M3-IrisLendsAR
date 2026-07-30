# SHARP + TRACK-ROT 종결 착수 킥오프 — 뷰티 재판정 → 진단 제거 → develop 머지

> 이 문서 + §0 필독만으로 **대화 맥락 0인 새 세션이 착수**할 수 있게 쓴 것. 작성 2026-07-30.
> 남은 일은 3단계뿐이고, 원인 규명·수정·기기 검증은 **이미 전부 끝났다**. 재조사 금지.

---

## 0. 진입 절차

**필독 (이 순서로)**
1. 이 문서
2. `docs/workPaper/SHARP_ring_fbo_transpose.md` — 선명도 트랙. §7-3 게이트 결과표, §7-4 남은 일
3. `docs/workPaper/TRACK-ROT_landscape_hint_mismatch.md` — 회전 힌트 트랙. §3-2b 판별 실험, §5 남은 일

**브랜치**: `fix/ring-fbo-transpose`, **미머지·미푸시**.

> ⚠️ **`git rev-list --count develop..HEAD` = 11 이다. 9가 아니다.**
> 아래 9개가 이번 두 트랙의 작업분이고, **기저 2커밋이 develop 에 아직 없어 머지에 함께 딸려 온다.**
> §3③ 선행 결정 참조 — 이건 사용자 판단 사항이다.

```
4e23dd1 docs(track): 판별 실험으로 공식 확정 — 가설 B 기각
7896095 fix(track): 적대 검토 반영 — 자동 유도 래치·90배수 검증·캐시 정합·push→pull
9b0b44d fix(demo): 커밋 누락 리소스 3종 추가 — 깨끗한 체크아웃에서 빌드 실패
e2d5852 docs(track): TRACK-ROT 작업 문서
3bcc2a8 fix(track): 가로 화면에서 얼굴 추적 불안정 — MediaPipe 회전 힌트를 화면 회전에서 유도
9e231af docs(sharp): G8 폰 완료 — 태블릿 10/10 + 폰 게이트 전부 통과
6fbce7b docs(sharp): 얼굴 필요 게이트 완료
a65f06b fix(sharp): 링 FBO 전치 리샘플 정본 수정 — 가로 표본 소거 제거
098e0f7 feat(sharp): 선명도 격차 원인 확정 + 진단 경로
--- 여기까지가 이번 두 트랙 (9커밋). 아래는 기저 — develop 에 없음 ---
1579bc7 feat(single-stream): Step1~2 — FaceTracker RGBA 진입점 + GL 분석 패스(**미배선 WIP**)
0cb2c08 feat(demo): 태블릿 가로 + 선명도 개선 1차 (DEMO-LAND 트랙)
```

기저 2커밋 규모: `git diff --stat develop..HEAD` = 18파일 3386+/143- 인데 그중 9커밋분은
`git diff --stat 1579bc7..HEAD` = 9파일 1129+/35- 뿐이다. **머지 diff 의 약 2/3 이 기저 2커밋**이다.
특히 `0cb2c08` 은 `.claude/settings.local.json`(+153) 과 `CLAUDE.md` 까지 건드린다.

**빌드·설치**
```bash
cd /Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/android && ./gradlew :demo-app:assembleDebug
# APK 산출 경로 (build/outputs 아님 — 주의)
#   android/build/modules/demo-app/outputs/apk/debug/demo-app-debug-b314.apk  (~145MB)
ADB=~/Library/Android/sdk/platform-tools/adb
$ADB install -r android/build/modules/demo-app/outputs/apk/debug/demo-app-debug-b314.apk
```
- ⚠️ **항상 재빌드부터 하고 설치할 것.** `versionCode` 가 314 로 고정(`android/demo-app/build.gradle.kts`)이라 APK 파일명이 늘 `demo-app-debug-b314.apk` 다 — **디스크 APK 가 최신인지 알 신호가 없다.**
- ⚠️ 무선 ADB라 **설치가 1~3분** 걸린다. 포그라운드로 돌리면 타임아웃 나므로 `run_in_background` 로 돌리고 `dumpsys package com.irislenssdk.demo | grep lastUpdateTime` 로 완료를 폴링할 것.
- ⚠️ **작성 시점(2026-07-30 11:40) 기준 두 기기 모두 adb 연결이 끊겨 있다.** `adb devices` 가 비면 사용자에게 무선 디버깅 재연결을 요청할 것.
- 기기: 태블릿 `SM-X920`(Tab S10 Ultra, 주 검증기) / 폰 `SM-S916N`(S23+, 대조).
  둘 다 붙어 있으면 `adb -s <serial>` 로 명시할 것 — serial 은 매 연결마다 바뀌므로 `adb devices -l` 로 확인.
- 측정 전 `$ADB shell svc power stayon true` (절전으로 화면 꺼지면 `screencap` 이 전부 검은 이미지).

**C++ 작업 시**: `cpp/` 하위를 건드리면 `systems-programming:cpp-pro` 에이전트와 함께 (CLAUDE.md 규칙). 단 **3단계 중 cpp 를 건드리는 건 1단계에서 재튜닝이 필요하다고 판정났을 때뿐**이다.

---

## 1. 목표

두 트랙(SHARP 선명도 / TRACK-ROT 회전 힌트)을 **develop 머지까지 종결**한다. 순서:

**① 뷰티 튜닝 재판정 → ② 진단 표면 제거 → ③ 머지**

①이 먼저인 이유: 진단 표면(A/B 킬스위치)이 살아 있어야 "수정 전/후"를 같은 빌드에서 대조할 수 있다. 제거 후에는 대조 수단이 사라진다.

---

## 2. 현재 상태 — 검증된 사실 (2026-07-30 코드 직접 확인)

### 2-1. 두 수정의 요지 (재조사 불필요)

| 트랙 | 원인 | 수정 | 검증 |
|---|---|---|---|
| **SHARP** | 카메라 버퍼는 1080×1920 세로형인데 링/렌즈 FBO를 `SurfaceRequest.resolution`(센서 좌표 1920×1080)으로 잡아 **전치 리샘플** → 장면 가로축 표본 43% 소거 | `ringW/ringH` 를 링 공간 단일 진리원으로 승격, 회전 90/270이면 치수 교환. 회전 정본 생산자를 `SurfaceRequest.setTransformationInfoListener` 로 교체 | 태블릿 게이트 10/10 + 폰 G8. 태블릿 2.35배·폰 2.09배 개선 |
| **TRACK-ROT** | `targetRotation=ROTATION_0` 핀 때문에 `imageInfo.rotationDegrees` 가 기기 자세와 무관한 상수(270) → 기기를 90° 돌려 들면 MediaPipe 가 누운 얼굴을 봄 | 힌트에만 화면 회전을 더함: `hint = (buffer + screenRotation) % 360`. 좌표 변환 경로 무변경 | screenRot 90·270 두 지점 블라인드 A/B 로 공식 확정(가설 B 기각) |

### 2-2. 뷰티 튜닝 상수 (①의 대상)

```
cpp/include/iris_sdk/warp/jaw_warp_geometry.h:132   constexpr float kMaxDispRatio = 0.032f;
cpp/include/iris_sdk/warp/jaw_warp_geometry.h:134   constexpr float kSigmaRatio   = 0.13f;
cpp/src/warp/jaw_warp_geometry.cpp:133,146          max_disp = face_width_px * kMaxDispRatio * strength
cpp/src/warp/jaw_warp_geometry.cpp:170              out.sigma_px = face_width_px * kSigmaRatio
```

**왜 재판정이 필요한가**: 이 상수들은 워프가 도는 **링 공간**에서 `face_width_px` 에 비례한다. SHARP 수정 전 태블릿 링 공간은 **비등방 3.16:1** 이었으므로, 등방 가우시안이 실제 장면에서는 타원이었다. 수정 후 등방이 되면서 워프 거동이 바뀐다 — **수치 회귀가 아니라 정상화**다. S23+ 세로(원래 등방)에서 검증된 값이라면 오히려 같은 조건으로 되돌아온 것이라 재튜닝이 불필요할 수 있다. 그래서 "확인"이지 "수정"이 아니다.

### 2-3. 제거 대상 vs 보존 대상 — ⚠️ 여기서 헷갈리면 수정이 통째로 날아간다

**보존 (= 이번 수정 본체. 절대 지우지 말 것)**

| 위치 | 내용 |
|---|---|
| `CameraGLRenderer.kt` | `ringW`/`ringH` 필드, `ringDimsFor()`, `recreateIntermediateBuffers` 치수 교환, `renderToScreen` 의 `srcTexW/srcTexH`, `applyGpuLensRenderer`/`applyGpuBeautyFilter` 의 `ringW/ringH` 사용, `markGlHandlesStale` 의 ringW/ringH 리셋, `setFrameRotation` 멱등 재생성 |
| `CameraGLView.kt` | `fulfillPendingSurfaceRequest` 의 `setTransformationInfoListener` → `glRenderer.setFrameRotation(info.rotationDegrees)` |
| `FaceTracker.kt` | `detectionRotationOffset` 필드, `setDetectionRotationOffset()`, `processingOptions()` 의 힌트 산식·캐시 순서 — **전부 수정 본체다** |
| `GpuRenderActivity.kt` | `detRotOffset` 필드, `pushScreenRotation()` 의 오프셋 유도, `processFrameTasks` 의 매 프레임 pull(`:1420`), `ensureFaceTracker` 의 적용(`:1445`) |

**제거 (= 진단 전용)**

| 파일 | 라인(2026-07-30 기준) | 내용 |
|---|---|---|
| `GpuRenderActivity.kt` | 113–120 | `ACTION_*` 상수 7개 |
| | 267 | `detRotAuto` 필드 |
| | 279–282 | `blindOn` / `blindArms` / `blindIdx` / `blindLabel` |
| | 1573–1574 | HUD 의 `◀ 팔 X ▶` / `det:N(auto)` 표시 |
| | 1753 | `pushScreenRotation` 의 `detRotAuto &&` 조건 → `if (detRotOffset != deg)` 로 |
| | 1951 / 2162 | `registerSharpnessDiagReceiver()` / `unregisterSharpnessDiagReceiver()` 호출 |
| | 1968–2080 | 리시버 필드·등록·분기 전체 + `applyDetRot()` + 해제 함수 |
| `CameraGLRenderer.kt` | 324 | `dumpRingDir` |
| | 354–355 | `ringLegacyTranspose` / `ringRecreatePending` |
| | 637–643 | `onDrawFrame` 의 `ringRecreatePending` 블록 (선행 주석 포함) |
| | 663–666 | `onDrawFrame` 의 덤프 블록 |
| | 1462 / 1474 / 1487~ | `requestRingDump()` / `setRingLegacyTranspose()` / `dumpRingSlot()` |
| `CameraGLView.kt` | 312 / 319 | `requestRingDump()` / `setRingLegacyTranspose()` |

**⚠️ 킬스위치 제거 시 조건식 2곳 — 기계적으로 지우면 동작이 뒤집힌다**

```kotlin
// CameraGLRenderer.kt:1159 (renderToScreen)
//   전: val isRotated = ringLegacyTranspose && (frameRotation == 90 || frameRotation == 270)
//   후: isRotated 는 항상 false → 변수와 분기를 통째로 제거하고
//       val texWidth = srcTexW; val texHeight = srcTexH  로 단순화

// CameraGLRenderer.kt:1416 (ringDimsFor)
//   전: val rotated = !ringLegacyTranspose && (frameRotation == 90 || frameRotation == 270)
//   후: val rotated = (frameRotation == 90 || frameRotation == 270)     // ← ! 제거 주의
```

⚠️ **라인 번호만 기계적으로 지우면 고아 주석이 남는다.** 각 항목에 딸린 설명 주석도 함께 지울 것 —
`GpuRenderActivity.kt` 263–266(`detRotAuto` KDoc) / 269–278(블라인드 설계 주석) / 1955–1967(리시버 섹션 헤더),
`CameraGLRenderer.kt` 314–323(`dumpRingDir` 헤더) / 348–353(킬스위치 KDoc).
라인 번호는 2026-07-30 기준이며 **편집하는 순간 어긋난다 — 위에서부터가 아니라 아래에서부터 지우거나 심볼로 찾을 것.**

**선택(안 해도 머지 가능)**: `resolveCoordinateSpace`(호출자 0) / `createLensFbo`+`lensFboId`(도달 불가) dead 정리.
이미 경고 주석이 붙어 있다. SHARP §7-4-6 은 이를 "트랙 종결 시"로 적었으나 **머지를 막지 않는다 — 이 킥오프가 우선**이다.

---

## 3. 할 일

### ① 뷰티 튜닝 재판정 — 육안

1. 태블릿 **가로**(평소 자세), 자동회전 ON. 얼굴을 프레임에 넣는다.
2. 뷰티 ON: 좌측 패널 핸들(가로 레이아웃에서 화면 좌측 세로 보라색 탭) → `뷰티` 탭 → `Beauty: OFF` 버튼 탭.
   - 좌표는 해상도·레이아웃에 따라 다르니 **탭 전에 `screencap` 으로 위치를 확인**할 것.
3. 턱 V라인 워프와 스킨 마스크를 본다. 판정 기준: **턱선 워프가 과하거나 부족하지 않은가 / 좌우 대칭인가 / 마스크 경계가 튀지 않는가.**
4. 대조가 필요하면 킬스위치로 수정 전 상태와 A/B:
   ```bash
   $ADB shell am broadcast -a com.irislenssdk.demo.SET_RING_SWAP --ez legacy true   # 구 동작(비등방)
   $ADB shell am broadcast -a com.irislenssdk.demo.SET_RING_SWAP --ez legacy false  # 수정(등방)
   ```
5. **S23+ 세로에서도** 한 번 볼 것 — 이 상수의 원래 검증 환경이다. 폰이 세로에서 종전과 같아 보이면 "정상화" 판정의 근거가 된다.

**판정 → 조치**
- 자연스러움 → **재튜닝 불필요.** workPaper SHARP §7-4-3 을 완료로 표시하고 ②로.
- 과하거나 부족 → 상수 조정이 필요. `cpp/` 작업이므로 `systems-programming:cpp-pro` 와 함께. **단, 이건 별도 트랙으로 분리하는 것을 우선 검토**할 것 — 머지를 막을 사안이 아니고, 튜닝은 반복 육안 판정이 필요해 길어진다.

> ⚠️ 이 판정은 **정량 지표로 하지 말 것.** 같은 이유로 실패한 전례가 있다 — TRACK-ROT §3-3 참조(정지 상태 마커 흔들림 지표는 오프셋 간 차이보다 내부 편차가 커서 판별력이 없었고, 1라운드 결과를 믿었다가 3라운드에서 뒤집혔다).

### ② 진단 표면 제거

§2-3 표대로 제거한다. 순서 권장: `CameraGLView` → `CameraGLRenderer` → `GpuRenderActivity`(참조하는 쪽부터 지워야 컴파일 오류로 누락을 잡는다).

**제거 후 게이트** (전부 통과해야 함):
```bash
cd android && ./gradlew :demo-app:assembleDebug     # 기대: BUILD SUCCESSFUL

# ⚠️ 심볼을 빠짐없이 열거할 것. 짝으로 남으면 컴파일도 통과하고 게이트도 통과한다
#    (예: requestRingDump 와 dumpRingDir 를 함께 남기면 아무도 못 잡는다).
#    android/ 통째 재귀는 build/(145MB APK 포함)까지 훑으므로 제외한다.
grep -rn --include='*.kt' \
  -e ringLegacyTranspose -e ringRecreatePending -e dumpRingDir -e dumpRingSlot \
  -e requestRingDump -e setRingLegacyTranspose \
  -e detRotAuto -e blindOn -e blindArms -e blindIdx -e blindLabel -e applyDetRot \
  -e SharpnessDiagReceiver -e ACTION_DUMP_RING -e ACTION_SET_UPSCALE \
  -e ACTION_SET_RING_SWAP -e ACTION_SET_DET_ROT -e ACTION_DET_BLIND \
  android/ | grep -v '/build/'
                                                    # 기대: 결과 0건
```
실기기 재확인 (태블릿 가로):
```bash
$ADB logcat -d -v time CameraGLRenderer:D GpuRenderActivity:I FaceTracker:I "*:S" \
  | grep -E "Intermediate ring|검출 힌트"
```
- 기대 1: 콜드스타트 **마지막** `Intermediate ring buffers created:` 가 `1080x1920 ... swapped=true`
- 기대 2: `검출 힌트 회전 → 0 (버퍼 270 + 오프셋 90)` (가로 기준)
- 기대 3: 렌즈 켠 상태에서 홍채 정합 유지, Render FPS 60 전후

### ③ develop 머지

> ⛔ **선행 결정 2건 — 사용자에게 반드시 물을 것. 임의로 진행 금지.**

**결정 1 — 기저 2커밋을 어떻게 할 것인가 (blocker)**

`git merge fix/ring-fbo-transpose` 를 그대로 치면 **11커밋이 들어간다**. 이번 두 트랙(9커밋) 외에:

| 커밋 | 정체 | 상태 |
|---|---|---|
| `1579bc7` | 단일 스트림 Step1~2 | **미배선 WIP**. §6이 "재개 후보"로 적은 바로 그 트랙 |
| `0cb2c08` | DEMO-LAND 태블릿 가로 1차 | 별개 트랙. `.claude/settings.local.json`·`CLAUDE.md` 포함 |

선택지:
- **(A) 11커밋 통째 머지** — 기저 두 트랙도 develop 에 올린다는 뜻. DEMO-LAND 는 이번 SHARP G4/G6 에서 렌즈·마커 정합과 `rot:±` 부호가 확인됐으므로 근거가 있다. 단일 스트림 WIP 는 미배선이라 실동작 영향이 없다.
- **(B) 9커밋만** — `git checkout -b <새브랜치> develop && git cherry-pick 0cb2c08..HEAD` 는 결국 같은 11개다. 9개만 떼려면 `1579bc7`·`0cb2c08` 를 뺀 cherry-pick 이 필요한데, 두 트랙 코드가 같은 파일을 크게 건드려 **충돌이 확실하다.** 권장하지 않는다.
- **(C) 기저를 먼저 별도 머지** — DEMO-LAND / 단일스트림을 각자 검증·머지한 뒤 이 브랜치를 리베이스.

**권고: (A)**. (B)는 충돌 비용이 크고, (C)는 기저 트랙 재검증이 필요해 길어진다. 다만 **`.claude/settings.local.json` 과 `CLAUDE.md` 변경이 develop 에 실리는 것**은 사용자가 알고 승인해야 한다.

**결정 2 — SHARP / TRACK-ROT 브랜치 분리 여부**

두 트랙은 원인·수정 지점·검증이 전부 독립이라 분리가 깔끔하지만, 함께 머지할 거면 굳이 쪼갤 필요 없다. **미결.**

**머지 실행 (결정 후)**
```bash
git checkout develop
git merge --no-ff fix/ring-fbo-transpose      # 커밋 메시지: 접두사 영문 + 본문 한글 (CLAUDE.md)
git push origin develop
```
- ⚠️ **`develop` 이 `origin/develop` 보다 이미 3커밋 앞서 있다** (`c38f76f`, `c5d9f37`, `48f26f0`).
  `git push` 하면 이 3개도 함께 올라간다 — 의도한 것인지 확인할 것.
- 이 저장소의 내부 통합 머지는 **PR 생략, `--no-ff` 직접 머지**가 관례다.
- 머지 후 두 workPaper 의 상태 줄에서 "미머지·미푸시" 를 갱신하고 머지 커밋 해시를 변경 이력에 남길 것.

---

## 4. ⚠️ 절대 제약 (불변식)

1. **`screenRotation == 0` 에서 종전과 픽셀 동일**이어야 한다. 폰 세로·태블릿 세로 무회귀의 근거다. 회전 유도식을 건드릴 때 이 성질이 깨지는지 반드시 확인.
2. **좌표 변환 경로(`TasksToIrisResult` / `CoordMapper` / native 렌즈 좌표)는 건드리지 않는다.** MediaPipe 가 힌트와 무관하게 원본 센서 공간으로 재투영한다는 계약(`FaceTracker.kt` `onRawResult` KDoc) 위에 두 수정이 서 있다. 실측으로 확인됐다.
3. **렌즈·뷰티에 넘기는 텍스처 치수는 반드시 `ringW/ringH`.** 어긋나면 native 패스가 링을 재전치 리샘플해 SHARP 수정이 통째로 무효화되는데, **렌즈 기하는 UV 불변이라 육안으로는 멀쩡하다.** 선명도 측정으로만 잡힌다.
4. **`ensureAnalysisTarget`(`CameraGLRenderer.kt`)은 손대지 않는다.** 같은 전치 버그가 잠복해 있으나 현재 미배선이다. 배선 시 **치수와 회전을 반드시 함께** 바꿔야 하고 반쪽만 바꾸면 렌즈가 3.16:1 타원이 된다. 경고 주석이 코드에 있다.
5. **`recreateIntermediateBuffers` 에 '치수 동일 시 조기 return' 멱등 가드를 넣지 말 것.** `markGlHandlesStale` 후 같은 치수로 재진입하므로 조기 return 하면 링이 영영 안 만들어져 블랙스크린이 된다.

---

## 5. 워킹트리 주의 — 커밋 금지 의도적 제외분 (정상 잔존)

```
 M .gitignore                                  ← 사용자가 docs/ar-report, docs/fm-lens-images 추가. 손대지 말 것
?? .claude/settings.local.json.doctor-backup   ← 도구 백업
?? docs/plans/bubbly-prancing-boot.md          ← 이 트랙 무관
?? scripts/__pycache__/                        ← 파이썬 캐시
```
중단 잔여물이 아니라 **정상 상태**다. 커밋하지 말 것.
(이 킥오프 문서와 두 workPaper 는 **커밋돼 있다** — `git ls-files docs/workPaper/` 로 확인 가능.)

**측정 산출물 위치 (gitignore 됨 — 디스크에만 존재)**
`docs/ar-report/sharp-measure/` — `axis_metrics.py`(축 분리 2차차분), `measure_mag.py`, `ring_vs_fmlens.py`, `gates/`(게이트 캡처).
`.gitignore` 에 `docs/ar-report` 가 들어 있어 git 에 없다. **수치는 두 workPaper 에 전부 전사돼 있으므로 사라져도 재현 가능**하지만, 재측정하려면 이 경로를 먼저 확인할 것.

---

## 6. 완료 후

- 두 workPaper(`SHARP_*`, `TRACK-ROT_*`) 상태 줄·변경 이력 갱신 (CLAUDE.md 작업 규칙).
- 메모리 `MEMORY.md` 의 두 트랙 항목을 종결 상태로 갱신.
- 이 킥오프 문서는 종결 시 삭제하거나 "완료" 표시.

**다음 단계 후보** (이번 범위 밖, 두 workPaper 에 상세):
- 태블릿 분석 스트림 **512×288** (폰 640×360) — 면적 1.56배 차이로 랜드마크 정밀도에 불리. 회전과 무관한 별개 요인이며 **회전 건이 해결된 지금 1순위 후보**.
- `screenRotation 180`(거꾸로 세로) 외삽 — 90·270 확정으로 가법성 성립, 위험 낮음.
- 자동회전 **잠금** 시 회전 수정 미동작 — 윈도우 회전을 읽으므로 잠그면 engage 안 함. 잠금 상태는 수정 전과 동일 동작이라 회귀가 아니라 미개선.
- 단일 스트림 트랙(`feature/single-stream-camera` 원래 목표) 재개.
