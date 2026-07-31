# DEMO-LAND — 태블릿 가로 레이아웃 + 회전 파이프라인

**상태**: 🔄 구현 완료 · 실기기(태블릿) 검증 대기
**작성일**: 2026-07-22
**브랜치**: develop (워킹트리, 미커밋)

---

## 1. 목표

데모 앱을 태블릿 가로에서 쓸 수 있게 만든다.

- 폰은 **세로 고정 유지**(기존 동작 무회귀), sw600dp 태블릿만 가로 허용
- 가로에서 하단 BottomSheet 한 덩어리를 **좌(기능 설정) / 우(렌즈 선택)** 로 분리
- 양쪽 패널은 **각각 접기 가능**, 둘 다 접으면 카메라 풀스크린
- 가로에서 **카메라 영상이 눕지 않도록** 회전 파이프라인까지 정리
- 디버그 오버레이(Mesh/Iris/MaskEdge)도 가로에서 얼굴에 정확히 붙게

사용자 결정 사항:
- 렌즈 리스트 = **1열 가로형 아이템**(썸네일+이름, 행 높이 72dp)
- 개발/벤치 패널(기어 ⚙) = **지금대로 카메라 위 오버레이 유지**
- 디버그 오버레이 회전 보정 = **이번 범위에 포함**

---

## 2. 사전 조사에서 확정된 제약

| 사실 | 근거 | 영향 |
|---|---|---|
| `configChanges="orientation\|screenSize"` 때문에 회전해도 액티비티가 재생성되지 않는다 | AndroidManifest.xml | **`layout-sw600dp-land/` 대체 리소스가 적용되지 않는다** → 재부모화로 전환 |
| 실제 픽셀 회전은 `stMatrix` 한 곳이 담당하고, 그 upright는 *기기 natural orientation 기준* | CameraGLRenderer.kt (`onDrawFrame`의 `getTransformMatrix`, `renderOESToRgba`) | 화면만 돌면 영상이 눕는 근본 원인 |
| `bottomSheet`/`dragHandle`은 Kotlin 코드 참조가 0건 | grep | 가로에서 시트를 통째로 숨겨도 안전 |
| `findViewById` 52건이 전부 `lateinit` 비-null 대입 | GpuRenderActivity.initViews / setupDebugControls | 레이아웃 재인플레이트 방식은 재바인딩 누락 리스크가 큼 |
| `onSaveInstanceState`가 없다 | GpuRenderActivity | 재생성 경로를 택하면 슬라이더·렌즈·벤치 플래그가 전부 리셋 |

→ **재생성 없는 런타임 재부모화 + 최종 blit 회전** 채택.

---

## 3. 구현 내역

### 3-1. 회전 파이프라인

핵심 원칙: **추적·합성 좌표계는 회전과 무관하게 고정하고, 화면 방향 보정은 최종 blit 한 곳만 담당한다.**

- `GpuRenderActivity.bindCameraUseCases` — `Preview`/`ImageAnalysis`에 `setTargetRotation(Surface.ROTATION_0)` **핀**.
  기본값은 use case 생성 시점의 display rotation이라, 세로 고정을 풀면 회전마다
  `rotationDegrees`와 `IrisResult.frameWidth/Height`가 480×640↔640×480으로 뒤집힌다.
  링 FBO는 센서 치수 그대로라 랜드마크 upright 공간과의 계약이 깨져 **렌즈가 어긋난다**.
  ROTATION_0 핀으로 이 축을 아예 상수화했다.
- `CameraGLRenderer.VERTEX_SHADER` — `uniform int uRotate` 추가. **정점만 90°×k 회전**하며
  texCoord·`uSTMatrix`·`uMirror` 경로는 건드리지 않는다(미러 순서 함정 회피).
- `CameraGLRenderer.renderToScreen` — 회전 후 종횡비로 cover 스케일을 계산하고,
  셰이더가 `[scale → rotate]` 순서이므로 `rotSwap`일 때 **uScale 축을 교환**해 넘긴다.
  (90° 회전은 축 교환이라 별도 종횡비 보정 없이 정확히 맞는다.)
- `renderOESToRgba` — `uRotate=0` **명시**. `passthroughProgram`과 `oesToRgbProgram`이
  같은 버텍스 셰이더 소스를 공유하므로 미설정 시 FBO 패스가 오염된다.
- `OverlayView.onDraw` — 캔버스를 `translate+rotate`로 돌리고, 본문을 `drawContent()`로 분리
  (중간 `return`이 있어 save/restore 짝이 깨지지 않도록). 좌표 계산 25곳을
  **논리 캔버스**(`vw`/`vh`)로 전환해 GL의 COVER 결과와 픽셀 단위로 일치시켰다.
- 회전 주입 경로: `pushScreenRotation()` → `onCreate` / `onConfigurationChanged` / `DisplayManager.DisplayListener`.
  **DisplayListener가 필수인 이유**: 180° 회전은 orientation/screenSize가 안 바뀌어
  `onConfigurationChanged`가 오지 않는다.

**세로에서는 `screenRotation=0` → 회전 항등 + aspect 스왑 없음 → 수정 전과 픽셀 동일.**

### 3-2. 가로 레이아웃 (재부모화)

`applyOrientationLayout(landscape)` → `enterLandscapeLayout()` / `enterPortraitLayout()`.

| | 세로 | 가로 |
|---|---|---|
| `tabLayout` + `tabContentContainer` | bottomSheet 안 (280dp 고정) | `landLeftPanel` (300dp, weight로 전체 높이) |
| `rvLenses` | bottomSheet peek (가로 스크롤, 120dp) | `landRightPanel` (180dp, 세로 리스트) |
| 렌즈 아이템 | `item_lens.xml` (썸네일+이름 세로) | `item_lens_land.xml` (썸네일+이름 행) |
| `bottomSheet` | VISIBLE | GONE |

- 뷰 **인스턴스를 그대로 옮기므로** 슬라이더 값·리스너·렌즈 선택·GL 상태가 회전을 넘어 유지된다.
- `LensAdapter.getItemViewType`이 레이아웃 리소스 id를 그대로 반환 → 레이아웃 전환 시
  기존 ViewHolder가 재사용되지 않고 새로 inflate된다(RecycledViewPool 수동 정리 불필요).
- 세로 복귀는 `dragHandle` 다음 index 1/2/3에 원본 LayoutParams로 재삽입.

### 3-3. 접이식 패널

- 핸들 버튼 2개(`btnLandLeftHandle` / `btnLandRightHandle`) → `translationX` 슬라이드 (180ms).
- 패널이 열리면 겹치는 오버레이도 함께 밀린다: 좌 → `statusOverlay`(HUD), 우 → `devPanelContainer`+`btnGearToggle`.
- **리뷰에서 잡힌 결함(수정 완료)**: 슬라이드 애니메이션 도중 회전하면 살아 있는
  `ViewPropertyAnimator`가 `translationX=0` 대입을 덮어써 HUD·기어 버튼이 세로 화면에서
  밀린 채 고착됐다. `moveX`/`slidePanel`/`enterPortraitLayout`에서 `animate().cancel()` 선행으로 해결.

### 3-4. 태블릿 한정 가로 허용

- `res/values/bools.xml` → `allow_landscape=false` (폰)
- `res/values-sw600dp/bools.xml` → `allow_landscape=true` (태블릿)
- `onCreate`에서 `setContentView` **직전**에 `requestedOrientation` 결정 (초기 1프레임 깜빡임 방지)
- AndroidManifest: `screenOrientation="portrait"` 제거, `configChanges`는 **유지**하고
  `smallestScreenSize|screenLayout|keyboardHidden` 추가. `MediaPipeBenchmarkActivity`는 그대로 세로 고정.

---

## 4. 변경 파일

**신규**
- `android/demo-app/src/main/res/values/bools.xml`
- `android/demo-app/src/main/res/values-sw600dp/bools.xml`
- `android/demo-app/src/main/res/layout/item_lens_land.xml`

**수정**
- `android/demo-app/src/main/AndroidManifest.xml`
- `android/demo-app/src/main/res/layout/activity_gpu_render.xml` (좌/우 패널·핸들·`btnRotSign` 추가 — 기존 노드 이동/삭제 없음)
- `android/demo-app/src/main/java/com/irislenssdk/demo/GpuRenderActivity.kt`
- `android/demo-app/src/main/java/com/irislenssdk/demo/camera/OverlayView.kt`
- `android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLRenderer.kt`
- `android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLView.kt`
- `android/demo-app/src/main/java/com/irislenssdk/demo/lens/LensAdapter.kt`

빌드: `:demo-app:assembleDebug` 통과.

---

## 5. 미결 — 실기기에서 확정할 것

### 5-1. 회전 부호 (필수)

`uRotate = k` vs `(4-k)%4` 중 어느 쪽인지는 리포 내 근거만으로 확정 불가.
기어 ⚙ 패널의 **`rot:+` / `rot:-` 토글**로 A/B 한다 (GL과 OverlayView를 동시에 뒤집으므로 정합 유지).
확정되면 `CameraGLRenderer.screenRotationInverted` 기본값을 상수화하고 버튼을 제거한다.

### 5-2. 검증 체크리스트

**폰(세로) — 회귀 확인**
- [ ] 회전해도 세로 고정 유지
- [ ] 카메라 영상 방향·미러링이 작업 전과 동일
- [ ] 바텀시트 peek → 확장, 렌즈/뷰티 탭 전환 정상
- [ ] 슬라이더 4종 반응, 렌즈 착용 위치 정확
- [ ] 기어 → dev 패널 토글, w6 행 가로 스크롤 정상

**태블릿(가로) — 신규**
- [ ] **가로에서 영상이 눕지 않는다** (최우선 / `rot:±` 토글로 부호 확정)
- [ ] 얼굴이 화면 밖으로 튀지 않는다 (cover 스왑 검증)
- [ ] 좌우 미러링 방향이 세로와 동일
- [ ] **렌즈가 눈동자에 정확히 붙는다** → targetRotation 핀 검증
- [ ] Mesh/Iris 오버레이 마커가 얼굴에 정확히 붙는다 → OverlayView 회전 검증
- [ ] 좌패널 300dp에 슬라이더가 잘리지 않고 들어간다
- [ ] 우패널 렌즈 리스트 스크롤·선택 반영
- [ ] 좌/우 핸들 접기 → 둘 다 접으면 풀스크린
- [ ] **180° 회전**(가로↔가로 뒤집기)에서도 영상 정상 → DisplayListener 검증
- [ ] **가로 상태로 콜드 스타트** → 영상·렌즈 정상
- [ ] 회전 왕복 3회 후 렌즈 선택/슬라이더 값 유지, 블랙스크린 없음
- [ ] 태블릿 세로에서는 기존 바텀시트 UI 정상

---

## 6. 알려진 한계 / 후속 후보

1. **노치·시스템바 인셋 미처리** — `themes.xml`이 `shortEdges`라 가로에서 컷아웃이
   좌/우 패널 가장자리를 물 수 있다. 실기기 확인 후 필요하면 패널에만 인셋 패딩 적용.
2. **핸들 터치 타깃 28dp** — 제스처 백 영역과 가까워 실사용에서 좁게 느껴지면 폭 상향.
3. **태블릿 세로는 기존 바텀시트 유지** — 좌우 패널로 갈지는 미결.
4. **`rot:±` 토글은 임시 벤치 UI** — 부호 확정 후 제거 대상.

---

## 7. 변경 이력

| 날짜 | 내용 |
|---|---|
| 2026-07-22 | 조사(4트랙 병렬+적대검증) → 구현 → 리뷰(5렌즈, 지적 19건 중 3건 확인/16건 반박) → 애니메이터 취소 결함 수정 → 빌드 통과 |
