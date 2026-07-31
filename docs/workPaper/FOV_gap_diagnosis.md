## 원인 확정

기여도 순.

### 1. (주원인·확정) 우리는 4:3 소스를 가로 창에 Cover로 깔아 **가능한 최소 배율**에 고정돼 있다

- 프리뷰·분석 둘 다 4:3 강제: `GpuRenderActivity.kt:1179` `AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY` (프리뷰 1440x1080 `:1140`, 분석 960x720 `:1155`).
- 최종 blit의 Cover 분기: `CameraGLRenderer.kt:840-846`
  ```kotlin
  val (screenScaleX, screenScaleY) = if (texAspect > viewAspect) (texAspect/viewAspect) to 1.0f
                                     else 1.0f to (viewAspect/texAspect)
  ```
  이 값은 정점에 곱해진다(`:98` `pos *= uScale`) → 균일 배율 **S = max(viewW/texW, viewH/texH)**.
- texAspect(1.333) < viewAspect(2960/1848 = 1.602) → **항상 else = width-bound**. 즉 **S = viewWidth / 1440 = 2.0556 px/텍셀**로 못박히고, 창 높이는 배율에 아예 안 들어간다.
- 물리적으로 4:3은 센서 active array 전체(= 이 카메라의 최대 화각)다. 그 최대 화각을 화면 폭에 통째로 우겨넣으니 배율이 최소가 된다. "얼굴 위 천장이 넓다"와 "얼굴이 작다"는 같은 현상의 두 얼굴이다(세로 가시율 1848/(1080×2.0556) = 83.2%).

> 주의(미확정): 절대값 2.0556은 회전 축 교환 두 곳(`CameraGLRenderer.kt:821-827` isRotated, `:831-836` rotSwap)이 상쇄될 때의 값이다. 아직 `rot:±` 육안 확정이 안 됐으므로(`GpuRenderActivity.kt:1013-1024` btnRotSign) 절대 배율은 미확인. **다만 "width-bound라 S ∝ viewWidth"와 "Z를 곱하면 정확히 Z배"라는 상대 관계는 네 가지 회전 조합 전부에서 성립**하므로 처방은 영향받지 않는다.

### 2. (FM 측·부분 확인) FM은 화면비에 가까운 스트림을 골라 height-bound로 들어간다

- FM은 가로에서 **풀스크린이 아니다**. `FmFittingCameraActivity.java:1112-1127 changeOrientation(true)`가 런타임에 프리뷰 박스를 `width = 화면폭 − dp(110) − dp(100)`, `height = 화면높이`로 줄인다(XML의 match_parent는 이 런타임 대입으로 덮인다). density 2.0이면 **2540x1848**. 좌 110dp 브랜드 패널은 `@color/white` 불투명이다. → 과제문 전제 "FM은 카메라 풀스크린 + 반투명 리스트"는 **사실이 아니다**.
- 표시 규칙은 우리와 동일한 Cover: `InterVisionEffect210.java:390-401` `scale = Math.max(pw/bufferW, ph/bufferH)` + `gravity = 17(CENTER)`.
- 다만 카메라 크기를 **화면 종횡비에 맞춰** 고른다: `InterVisionEffect210.java:495-510` targetAspect = 뷰(없으면 DisplayMetrics) long/short → `selectBestSupportedSize(..., tol 0.12, longCap 2160)`. 4:3(1.333)은 이 창에서 탈락한다.
- 결과적으로 FM은 **스트림 종횡비 > 박스 종횡비(1.374)** 라 height-bound가 되고, 화면 배율(센서 수평 화각 W당 화면 px) = `A × 1848` (A = 스트림 종횡비). 박스 폭은 배율에 안 들어간다.
- **미확인**: FM이 이 기기에서 실제로 고른 A. 실측 +10.8%를 역산하면 A ≈ 1.775 ≈ 16:9다. 그런데 `selectBestSupportedSize`의 tol 0.12는 1.602 기준 16:9(1.778)를 탈락시키므로, 폴백 경로를 탔거나 ST 경로(Camera1, 우선순위 1920x1080)였을 가능성이 있다. FM은 이걸 스스로 로그로 찍는다 → `adb logcat -s CamPick` 한 번이면 확정된다.

### 3. (기여 0 — 통념 반박) 좌우 패널은 배율에 영향이 **없다**

- `activity_gpu_render.xml:21-30` cameraGLView / overlayView는 CoordinatorLayout 직속 **match_parent 풀스크린**. `landLeftPanel`(300dp) / `landRightPanel`(180dp)는 `:96-112`의 형제 오버레이(`#A6000000`, 알파 65%).
- `enterLandscapeLayout()`(`GpuRenderActivity.kt:1582~`)은 tabLayout/tabContentContainer/rvLenses만 재부모화하고 cameraGLView를 한 번도 건드리지 않는다.
- 오히려 **GL 뷰를 패널만큼 좁히면 배율이 떨어진다**(아래 표). 지금 구조가 배율을 지켜주고 있다.
- 패널의 실제 손해는 두 가지뿐: (a) 그려놓고 못 보는 가로 화소, (b) 이미지가 창 중심에 그려져(셰이더에 translate 없음, `:92-109`) 우패널만 열린 기본 상태에서 얼굴이 가시영역 중심에서 **약 180px 우측 편위**.

### 4. (기여 0 — 확인) 해상도 변경(c38f76f)이 FOV를 못 바꾼 이유

960x720과 1440x1080 모두 4:3 → 같은 else 분기, 화면상 전체 프레임 세로 길이 2220px 동일. 바뀐 건 S(3.083 → 2.056 px/텍셀 = 선명도)뿐. 사용자 관찰과 정확히 일치.

### 5. (기여 0, 역효과) edge-to-edge

`WindowCompat.setDecorFitsSystemWindows` 등 옵트인 호출 grep 0건 → 지금은 시스템바만큼 인셋돼 있다. 풀스크린화해도 width-bound라 **배율 +0.0%**, 세로 가시만 78.9%→83.2%로 늘어 **천장이 더 보인다**.

---

## 정량

실측(사용자 제공, 스크린샷 파일은 이 세션에서 접근 불가 → 재검증 못 함):

| 지표 | 우리 | FM | 비 |
|---|---|---|---|
| 안경 렌즈 지름 | 185px | 205px | **+10.8%** |
| 얼굴 세로 | 755px | 805px | **+6.6%** |

두 지표가 4%p 어긋난다. 등방 crop이라면 같아야 하므로 둘 중 하나에 측정/자세 오차가 있다. **그래서 목표 배율을 1.11로 못박지 말고 1.06~1.11 구간을 실기기에서 스윕해 육안 확정할 것을 권한다.** (강체 경계인 렌즈 지름 쪽이 신뢰도가 높지만, 얼굴 세로가 이론과 안 맞는 이유는 미확인.)

요인별 기여:

| 요인 | 배율 기여 | 근거 |
|---|---|---|
| 우리 4:3 → width-bound Cover | **전부** (S = 2960/1440 고정) | `CameraGLRenderer.kt:840-846`, 코드 확정 |
| FM 스트림 종횡비 A ≈ 1.775 (height-bound, A×1848) | 위 격차의 반대편 항 | **추정** — A는 CamPick 로그로만 확정 |
| 좌우 패널 | 0% | 코드 확정 |
| 해상도 960→1440 | 0% | 코드 확정 |
| edge-to-edge | 0% (세로 가시만 +4.3%p, 역효과) | 코드 확정 |

레이아웃 변형별 배율(창 2960 기준, 소스 1440x1080):

| GL 뷰 폭 | 바인딩 축 | S | 배율 변화 |
|---|---|---|---|
| 2960 (현재) | W | 2.0556 | 기준 |
| 2600 (우패널 제외) | W | 1.8056 | **−12.2%** |
| 2000 (양 패널 제외) | H | 1.6222 | **−21.1%** |
| 렌더 zoom Z=1.11 | W | 2.2819 | **+11.0%** |

Z=1.11 적용 시 가시 FOV: 가로 100% → **90.1%**, 세로(4:3 프레임 대비) 83.2% → **75.0%**. (세로 75.0%는 4:3에서 16:9를 뽑을 때의 크롭 75%와 동일 — FM 모델과 정합.)

---

## 처방 비교

| 처방 | 효과 | 렌즈정합 위험 | 규모 | 검증결과 |
|---|---|---|---|---|
| **B. 렌더 최종 blit 등방 zoom** (`uScale × Z`) | **정확히 ×Z (+11.0% @ Z=1.11)** | **없음** (링 FBO·랜드마크·렌즈 좌표계 전부 불변) | S (4파일, 실질 1줄+주입) | **반증 실패 (high)** — 메커니즘·산술 모두 확인. 보조 수치 3건만 정정 |
| A. CameraX `setZoomRatio(Z)` | ×Z, 단 **maxZoomRatio ≥ Z일 때만** | 없음 (Preview/Analysis 동일 request) | S (1줄+onResume) | **부분 반증** — 전면 카메라 zoom range **미확인**(1.0이면 조용히 no-op). 복귀 시 줌 리셋 보고됨. 분석 FOV도 10% 축소 |
| C. 원안 레이아웃 개편(풀스크린+반투명+좌패널 닫힘) | **+0.0%** | 없음 | 0 | **반증** — 전제 3건이 이미 충족(패널 이미 오버레이·`#A6000000` 반투명·`leftPanelOpen=false`), edge-to-edge는 역효과 |
| C′. GL 뷰 가로 오버스캔(음수 마진) | +11% + 재중심 | 오버레이 마진 미동기 시 최대 326px | M | **반증** — 호출 시점에 `parent.width==0`이라 콜드 스타트 no-op. 확대 효과는 B와 수학적 중복 |
| GL 뷰를 패널 사이로 좁히기 | **−12~−21%** | 없음 | S | **반증** — Cover 특성상 역효과 |

---

## 권고

**단일 실행안: B — 최종 blit 등방 zoom(`displayZoom`), 가로에서만, 스윕 버튼으로 값 확정 후 상수화.**

이유:
- 기기 의존이 없다. A는 전면 카메라 `maxZoomRatio`가 1.0이면 코드가 조용히 아무 것도 안 한다(미확인 리스크). B는 무조건 동작한다.
- **캡처는 full 4:3을 유지**하므로 FaceLandmarker가 화면 밖까지 계속 본다. A는 분석 스트림도 함께 잘려 근접 시 얼굴 잘림 검출 실패가 10% 앞당겨진다 — CLAUDE.md "얼굴 전체 인식 필수" 한계에 직접 걸린다.
- 렌즈 정합이 구조적으로 안전하다(검증 확인): 링 FBO는 `CameraGLRenderer.kt:559` `glUniform2f(uScaleLocation, 1.0f, 1.0f)`로 프레임 1:1, 렌즈 합성도 `:716-717` frameWidth/Height 공간, 렌즈 uniform은 전부 정규화(`gpu_lens_renderer.cpp:1072-1073`, uFrameAspect는 검출 프레임 기준 `:1163`). 배경+렌즈+뷰티가 **같은 정점 변환 하나**를 탄다. C++/JNI/SDK 무변경.
- 되돌리기가 상수 1개. 기본값 1.0f는 현행과 비트 동일이라 폰 세로 무회귀.

대가: 텍셀 밀도 2.0556 → 2.2819(약 11% 물러짐). c38f76f 이전(3.083)보다는 여전히 크게 낫다. 이게 육안으로 거슬리면 그때 A(센서 크롭)로 갈아타면 된다 — 둘은 배타이므로 **절대 동시 적용 금지**(+23% 과확대).

### 코드 변경 위치

1. `android/.../camera/gpu/CameraGLRenderer.kt`
   - `:225` 근처(screenRotation 필드군)에 `private var displayZoom: Float = 1.0f`
   - `:1040 setScreenRotation` 옆에 `fun setDisplayZoom(z: Float) { displayZoom = z.coerceIn(1.0f, 1.5f) }`
   - **핵심 1줄** `:851` `GLES31.glUniform2f(scaleLocation, scaleX, scaleY)` → `GLES31.glUniform2f(scaleLocation, scaleX * displayZoom, scaleY * displayZoom)`
     (등방이라 `:849-850` rotSwap 축 교환과 무관 — 회전 경로 간섭 0)
2. `android/.../camera/gpu/CameraGLView.kt` — `:258-262 setScreenRotation`과 동일 패턴으로 `fun setDisplayZoom(z: Float) { queueEvent { glRenderer.setDisplayZoom(z) } }`
3. `android/.../camera/OverlayView.kt` — `:227` 근처에 `var displayZoom: Float = 1.0f (set → invalidate)`, `:640-642` `computeScreenTransform(...)` 호출 **닫는 괄호 뒤에** `* displayZoom`. `offsetX/offsetY`(`:649-650`)는 이 값에서 파생되므로 자동 정합.
   ※ GPU 데모는 `GpuRenderActivity.kt:384 overlayView.showLens = false`라 렌즈는 GL 단독. 여기를 놓쳐도 어긋나는 건 디버그 마커(기본 OFF)뿐이다.
4. `android/.../GpuRenderActivity.kt`
   - 필드: `private var landDisplayZoom = 1.11f` — **상수(const)가 아니라 var**로. 스윕 버튼이 이 값을 갱신하고, 회전 재주입이 스윕값을 덮지 않게.
   - `pushScreenRotation()` 옆에 `private fun pushDisplayZoom() { val z = if (landscapeLayoutApplied) landDisplayZoom else 1.0f; cameraGLView.setDisplayZoom(z); overlayView.displayZoom = z }`
   - `enterLandscapeLayout()` 끝(`:1619` Log 직전)과 `enterPortraitLayout()` 끝에서 각각 호출. 세로 원복 누락 시 폰 세로가 11% 확대된 채 고착되므로 필수.
   - 스윕 버튼: `activity_gpu_render.xml:425-436 btnRotSign` 패턴 복제(`btnZoom`, 텍스트 `zoom:1.00`), 탭마다 `{1.00, 1.06, 1.08, 1.11, 1.15}` 순환 → `landDisplayZoom` 갱신 후 `pushDisplayZoom()`. **육안 확정 후 값 고정하고 버튼 제거**(btnRotSign과 동일 수명 정책).

### 요청하신 "좌 패널 기본 닫힘"과의 관계

**이미 구현돼 있다** — `GpuRenderActivity.kt:1510` `private var leftPanelOpen = false`, 주석 `:1508-1509`에 "좌 패널(기능설정)은 기본 닫힘 — 카메라 표시 영역을 최대한 넓게 확보한다(사용자 요청)". 추가 작업 없음.

남은 선택지는 두 개뿐이고, **둘 다 배율에는 영향이 0**이다(가림·구도만 개선):
- 지금은 세션 내 상태 유지라 한 번 열면 회전을 넘어 열린 채 남는다. 가로 진입마다 강제로 닫으려면 `enterLandscapeLayout()`의 `setLeftPanelOpen(leftPanelOpen, animate = false)` → `setLeftPanelOpen(false, animate = false)` 한 줄.
- 우패널 180dp(`activity_gpu_render.xml:107` + `GpuRenderActivity.kt:90 RIGHT_PANEL_DP` **이중 정의 — 반드시 동시 수정**)를 줄이면 가림 화소가 준다.

즉 **확대는 B가, 가림/구도는 패널 조정이 담당**하며 서로 간섭하지 않는다. 한 커밋에 묶어도 안전하다.

---

## 하지 말 것

1. **edge-to-edge 전환** — 배율 +0.0%인데 세로 가시가 78.9%→83.2%로 늘어 "천장이 넓다"는 불만을 **악화**시킨다. 인셋 대응 부수 작업만 붙는다.
2. **GL 뷰를 패널 사이(2600 또는 2000px)로 좁히기** — Cover의 max()에서 좁아지는 축 항이 빠져 배율이 −12.2%~−21.1%. 직관과 반대다.
3. **C′ 오버스캔(음수 마진)** — 지정 호출 시점에 부모 폭이 0이라 콜드 스타트에서 통째로 no-op이 되고, 확대 효과 자체가 B와 수학적 중복이다. 여기에 EGL 버퍼 +7MB, 낭비 화소 9.9%, SurfaceView 창밖 배치 리스크가 얹힌다. 가치 있는 건 "가시영역 중심 재정렬"뿐인데 그건 별건으로 분리 가능.
4. **A(setZoomRatio)를 B와 동시 적용** — +23% 과확대.
5. **A를 1차로 채택** — 전면 카메라 zoom range 미확인이라 조용히 무효화될 수 있고, 복귀 시 리셋 대응(`onResume` 재적용)과 `ListenableFuture` 실패 로깅까지 붙어야 한다. B가 안 통할 때의 후보로 남긴다.
6. **"FM은 카메라 풀스크린"을 전제로 한 설계** — `FmFittingCameraActivity.java:1119-1123`이 반증. FM도 좌 110dp / 우 100dp를 차지하며 좌패널은 불투명 흰색이다.
7. **Z=1.11 하드코딩 후 바로 커밋** — 실측 두 지표가 1.066/1.108로 갈린다. 스윕 확정 전까지 매직넘버 확정 금지.

---

## 실기기 확인 항목 (Galaxy Tab S10 Ultra, 가로)

**순서 중요: rot 부호를 먼저 고정한 뒤 zoom을 판단한다.** 회전 부호가 어긋나면 절대 배율이 33% 달라져 Z 판단이 오염된다.

1. **`rot:±` 확정** — btnRotSign을 토글해 영상이 정립인 쪽으로 고정. 미결 과제였던 항목.
2. **로그 3줄 캡처** — `adb logcat | grep -E "Providing surface|Analysis resolution|Frame size set|Screen rotation|onSurfaceChanged"`. 여기서 frameW/H(정말 1440x1080인지), rotK, viewW/viewH(시스템바 인셋 후 실제 창 크기)가 한 번에 확정된다. 이 값들이 위 계산의 유일한 미확인 입력이다.
3. **FM 정답값 확보** — FM 피팅 화면 진입 상태로 `adb logcat -c; adb logcat -s CamPick:V ProcSize:V`. `>>> chosen(by aspect+cap): WxH`가 찍힌다. 이걸로 A(FM 스트림 종횡비)가 추정에서 실측으로 바뀌고 목표 Z가 확정된다.
4. **zoom 스윕** — btnZoom으로 1.00 → 1.06 → 1.08 → 1.11 → 1.15를 돌리며 FM과 같은 자리에서 육안 비교. 확인할 것:
   - 안경 렌즈 지름이 FM과 같아 보이는 지점
   - 얼굴 위 천장 여백이 FM과 같아 보이는 지점 (두 지점이 다르면 실측 4%p 불일치의 정체가 드러난다)
5. **렌즈-홍채 정합** — 각 Z에서 렌즈가 홍채에 붙어 있는지. 이론상 불변이지만 육안 1회 확인. 디버그 메시(showFaceMesh) 켜서 마커도 같이 붙는지 확인(오버레이 동기화 검증).
6. **근접 케이스** — 카메라에 얼굴을 가까이 가져갔을 때 이마/턱이 화면 밖으로 나가도 **추적이 끊기지 않는지**(B는 캡처 FOV를 유지하므로 끊기면 안 된다). 여기서 끊기면 다른 원인.
7. **회전 왕복** — 가로↔세로 3회. 세로에서 배율이 원래대로(확대 없음) 돌아오는지, 가로 복귀 시 스윕값이 유지되는지.
8. **백그라운드 복귀** — 홈 → 재진입. B는 렌더러 필드라 리셋되지 않아야 정상. 리셋되면 `pushDisplayZoom()` 재주입 지점 추가.
9. **패널 개폐** — 좌패널을 열었다 닫을 때 얼굴 위치가 어떻게 느껴지는지(현재 우패널만 열린 상태에서 약 180px 우측 편위 추정). 거슬리면 별건으로 재중심 처방 검토.
10. **프레임레이트** — HUD fps가 30 아래로 안 떨어지는지(이론상 blit 프래그먼트 비용 불변이라 무영향).