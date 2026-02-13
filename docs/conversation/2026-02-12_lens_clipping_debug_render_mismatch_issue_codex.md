# 렌즈 클리핑 미적용 + 디버그/실렌더 불일치 이슈 리포트

- 작성일: 2026-02-12
- 작성자: Codex
- 대상: `GpuRenderActivity` GPU 렌더 경로

---

## 1. 이슈 요약

현재 GPU 렌더 경로에서 아래 2가지가 동시에 관찰됩니다.

1. 렌즈 눈꺼풀 클리핑이 기대대로 동작하지 않음.
2. 개발 확인용 오버레이(메쉬/디버그)와 실제 렌즈 렌더 결과가 서로 다른 위치/크기/타이밍으로 보임.

---

## 2. 재현 관찰 포인트

1. 렌즈 적용 상태에서 눈꺼풀 위쪽/아래쪽 경계 밖으로 렌즈가 보이는 프레임이 있음.
2. `Debug/Mesh`를 켜면 오버레이 표시(포인트/원/메쉬)가 실제 렌즈 중심/반경과 체계적으로 어긋남.
3. 얼굴 거리/회전/검출 흔들림 구간에서 오버레이와 실제 렌즈 차이가 더 커짐.

---

## 3. 원인 분석

### A. 렌즈 클리핑 미적용(또는 불안정) 원인

1. GLSL `smoothstep` 호출 순서가 사양 위반 형태로 작성됨.
파일: `android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLRenderer.kt:210`
현재 코드:
```glsl
float bottomClip = smoothstep(maxY + eyelidFeather, maxY - eyelidFeather, vTexCoord.y);
```
`smoothstep(edge0, edge1, x)`에서 `edge0 < edge1`이 보장되어야 하는데, 현재는 역순(`edge0 > edge1`)입니다. 이 경우 결과는 구현/드라이버 의존(Undefined)이라 클리핑이 깨지거나 무시될 수 있습니다.

2. FaceMesh 비유효 시 클리핑이 전체 허용으로 즉시 폴백됨.
파일: `android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLRenderer.kt:702`
FaceMesh가 없는 프레임은 상하 경계를 `0.0~1.0`으로 넣어 사실상 클리핑을 끕니다.

3. GL 렌더용 `IrisResult`는 `detected=true`일 때만 갱신됨.
파일: `android/demo-app/src/main/java/com/irislenssdk/demo/GpuRenderActivity.kt:798`
검출 실패/저신뢰 프레임에서 GL쪽은 이전 스냅샷을 계속 사용하므로, 클리핑 경계가 현재 프레임과 어긋난 채 유지될 수 있습니다.

### B. 디버그/메쉬 vs 실제 렌즈 불일치 원인

1. 화면 매핑 정책 자체가 다름.
실렌더:
파일: `android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLRenderer.kt:859`
`renderToScreen()`가 `Cover`(화면 꽉 채움 + crop) 스케일링을 사용합니다.

디버그 오버레이:
파일: `android/demo-app/src/main/java/com/irislenssdk/demo/GpuRenderActivity.kt:250`
파일: `android/demo-app/src/main/java/com/irislenssdk/demo/camera/OverlayView.kt:309`
`overlayView.gpuMode = true`이며 `computeScreenTransform()`에서 `fit(min)` 매핑을 사용합니다.

즉, 한쪽은 `cover`, 한쪽은 `fit`이라 동일 좌표를 그려도 화면 위치가 일치할 수 없습니다.

2. 입력 데이터 정책이 다름(원천 데이터 불일치).
실렌더(GPU 렌즈):
파일: `android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLRenderer.kt:630`
`result.leftIrisX/rightIrisX` 등 원시값 기반, 별도 스무딩 없음.

디버그 오버레이:
파일: `android/demo-app/src/main/java/com/irislenssdk/demo/camera/OverlayView.kt:375`
One Euro Filter + 유지 타임아웃(렌즈 2초, 디버그 1초) 기반의 캐시값을 사용합니다.

데이터 소스와 시간 정책이 달라 동작이 다르게 보이는 것이 구조적으로 정상인 상태입니다.

3. 좌표 갱신 타이밍도 다름.
GL 업데이트는 조건부(`detected=true`)이고, 오버레이는 매 프레임 `uiIrisResult.copyFrom(...)` 후 UI 스레드 반영됩니다.
파일: `android/demo-app/src/main/java/com/irislenssdk/demo/GpuRenderActivity.kt:798`
파일: `android/demo-app/src/main/java/com/irislenssdk/demo/GpuRenderActivity.kt:820`

---

## 4. 수정 우선순위 제안

### P0. 셰이더 클리핑 수식 수정

`bottomClip`을 역순 `smoothstep` 대신 정상 순서 식으로 변경:
```glsl
float bottomClip = 1.0 - smoothstep(maxY - eyelidFeather, maxY + eyelidFeather, vTexCoord.y);
```
또는 동일 의미로 `edge0 < edge1`을 만족하는 식으로 재작성.

### P0. 오버레이/실렌더 매핑 정책 통일

둘 중 하나로 통일 필요:

1. GL 출력을 `fit`으로 되돌림.
2. Overlay를 `cover` 변환으로 바꿈.

현재는 `cover` vs `fit` 불일치가 가장 큰 시각적 오차 원인입니다.

### P1. GL/Overlay 데이터 계약 통일

1. GL용 `IrisResult`도 매 프레임 갱신(검출 실패 프레임 포함)해 stale 스냅샷 사용 최소화.
2. 디버그 표시 모드를 분리:
- `Raw(GL 입력과 동일)`
- `Smoothed(UI 가시화용)`

기본값은 문제 추적을 위해 `Raw` 권장.

### P1. 클리핑 폴백 정책 명확화

FaceMesh 비유효 시 즉시 클리핑 OFF 대신:

1. 이전 유효 eye-top/bottom N프레임 유지, 또는
2. 렌즈 alpha를 단계적으로 감쇠.

깜빡임과 경계 튐을 줄일 수 있습니다.

---

## 5. 검증 체크리스트

1. 클리핑 검증:
- 눈꺼풀 경계 외부 픽셀에서 렌즈 alpha가 0에 수렴하는지 확인.

2. 정합성 검증:
- 동일 프레임에서 오버레이 원 중심/반경과 실렌더 렌즈 중심/반경이 일치하는지 확인.

3. 모드 검증:
- 전면 미러 ON/OFF, 회전 0/90/270, 얼굴 근/원거리 모두에서 동일 정책 유지 확인.

4. 실패 프레임 검증:
- `detected=false`, `faceMeshValid=false` 연속 구간에서 렌즈/디버그 동작이 정책대로 안정 동작하는지 확인.

---

## 6. 결론

현재 이슈는 단일 버그가 아니라 `클리핑 셰이더 수식 문제 + 화면 매핑 정책 불일치 + 데이터 파이프라인 불일치`가 결합된 상태입니다.
우선 `P0(클리핑 식 + 매핑 통일)`를 먼저 처리하면 체감 문제 대부분이 즉시 줄어들고, 이후 `P1(데이터 계약/폴백)`로 안정화를 마무리하는 것이 가장 효율적입니다.

---

## ClaudeCodeComment (2026-02-12)

Codex 리포트의 코드 경로 추적과 원인 분류는 정확합니다. 실제 코드를 검증한 결과 아래와 같이 보강/수정합니다.

### 1) 3-A-1. smoothstep 역순 — 확인됨, 핵심 버그

`CameraGLRenderer.kt:210`:
```glsl
float bottomClip = smoothstep(maxY + eyelidFeather, maxY - eyelidFeather, vTexCoord.y);
```

- `edge0 = maxY + feather`, `edge1 = maxY - feather`이므로 `edge0 > edge1`.
- GLSL 사양(OpenGL ES 3.1 §8.3): `edge0 >= edge1`이면 결과가 **undefined**입니다.
- 실제 대부분의 모바일 GPU(Adreno/Mali)에서는 0.0을 반환하는 것으로 관찰되며, 이 경우 `eyelidMask = topClip * 0.0 = 0`이 되어 **렌즈 전체가 안 보이거나**, 또는 역으로 모든 값이 1.0을 반환하여 **클리핑이 완전 무효화**됩니다.
- Codex 제안 수정안 `1.0 - smoothstep(maxY - f, maxY + f, y)`는 정확합니다.
- 참고: `topClip`(line 209)은 `smoothstep(minY - f, minY + f, y)`로 `edge0 < edge1` 정상 순서.

### 2) 3-A-2. FaceMesh 비유효 시 즉시 폴백 — 확인됨, 정책적 판단 필요

`CameraGLRenderer.kt:701-706`:
```kotlin
// faceMesh가 없으면 클리핑 비활성화 (전체 영역 허용)
GLES31.glUniform1f(uLeftEyeTopLocation, 0.0f)
GLES31.glUniform1f(uLeftEyeBottomLocation, 1.0f)
```

- 이렇게 하면 `minY=0.0, maxY=1.0` → `topClip≈1.0, bottomClip≈1.0(정상일 때)` → 클리핑 OFF.
- **문제**: 검출 프레임과 미검출 프레임이 교차되면 렌즈가 깜빡거림 (클리핑 ON/OFF 번갈아).
- **그러나**: 현재 `bottomClip`이 undefined 상태이므로 이 폴백의 실효성도 undefined입니다. smoothstep 수정이 선행되어야 폴백 정책이 의미를 가집니다.

### 3) 3-B-1. Cover vs Fit 매핑 불일치 — 확인됨, 가장 큰 시각적 오차 원인

코드 상태:
- **GL 렌더**: `renderToScreen()` (line 872-879)에서 **Cover 모드** 사용 (`max(scale)` 방식)
- **OverlayView**: `gpuMode = true` → `computeScreenTransform(..., fitMode=true)` → **Fit 모드** 사용 (`min(scale)` 방식)

전형적인 portrait 환경(텍스처 9:16, 뷰 9:20)에서:
- Cover: `scaleY = 1.0, scaleX = texAspect/viewAspect ≈ 1.12` → 좌우 crop
- Fit: `scaleFactor = min(viewW/imgW, viewH/imgH)` → 상하 레터박스

결과: 동일 좌표에서 오버레이와 실렌더의 위치가 **수평/수직 모두** 어긋남. 얼굴이 중앙에서 벗어날수록 차이가 극대화됨.

**수정 방향**: OverlayView의 `gpuMode`를 Cover 매핑으로 전환해야 합니다. 현재 `computeScreenTransform`에 이미 `fitMode=false`(fill-center = max) 경로가 있으므로, `gpuMode = true`일 때 `fitMode = false`를 사용하면 Cover와 일치합니다. 즉 **로직이 반대로 연결되어 있는 상태**입니다.

### 4) 3-B-2. 데이터 소스 차이 — 확인됨, 의도된 설계

- GL 렌즈: `result.leftIrisX/Y` 원시값 직접 사용 (스무딩 없음)
- OverlayView: One Euro Filter 적용 + 유지 타임아웃

이 차이는 의도된 것이며 (GL 렌즈는 최소 지연, 디버그 오버레이는 안정적 시각화), 매핑 정책이 통일되면 "위치 차이"가 "미세한 시간차"로 축소됩니다. 별도 수정 불필요.

### 5) 추가 발견: 렌즈 셰이더 좌표계와 눈꺼풀 좌표계 불일치 가능성

렌즈 셰이더 내에서:
- **홍채 중심**: `adjustedCenter`는 `vTexCoord.x * aspectRatio` 보정이 적용된 좌표계
- **눈꺼풀 경계**: `eyeTop/eyeBottom`은 `vTexCoord.y`와 직접 비교 (aspect 보정 없음)

Y축만 사용하므로 aspect ratio 차이가 직접적 문제는 아니지만, 렌즈 중심의 Y와 눈꺼풀 경계의 Y가 동일 좌표계인지 검증이 필요합니다. `applyLensOverlay()`에서 홍채 Y는 `1.0 - rawY`(Y flip)을 거쳐 uniform으로 전달되고, 눈꺼풀 Y도 `1.0 - meshY`(같은 flip)를 거치므로 **좌표계는 일치**합니다.

### 보강 액션 플랜

#### P0-A. smoothstep 역순 수정 (즉시)
```glsl
// 현재 (undefined behavior):
float bottomClip = smoothstep(maxY + eyelidFeather, maxY - eyelidFeather, vTexCoord.y);
// 수정:
float bottomClip = 1.0 - smoothstep(maxY - eyelidFeather, maxY + eyelidFeather, vTexCoord.y);
```

#### P0-B. OverlayView Cover 매핑 전환 (즉시)
`OverlayView.kt`의 `onDraw()` 호출에서 `gpuMode=true`일 때 **fitMode를 false로 전환**:
```kotlin
// 현재 (잘못된 연결):
val scaleFactor = computeScreenTransform(imageWidth, imageHeight, width, height, gpuMode)
// 수정 — gpuMode일 때 cover(fill) 매핑 사용:
val scaleFactor = computeScreenTransform(imageWidth, imageHeight, width, height, !gpuMode)
```
또는 `gpuMode`의 의미를 재정의하여 혼란을 줄이는 리팩토링 권장.

#### P1-A. GL IrisResult stale 데이터 방지
`GpuRenderActivity.kt:798`에서 `detected=false`일 때도 slot을 갱신하되, `detected` 플래그를 false로 전달하여 GL 쪽에서 렌즈 페이드아웃 등의 정책을 적용할 수 있게 처리.

#### P1-B. 클리핑 폴백 개선
`smoothstep` 수정 후, FaceMesh 비유효 프레임에서 이전 유효 경계를 N프레임(예: 5프레임) 유지하는 temporal hold 로직 추가 권장.

---

## CodexReviewComment (2026-02-12)

ClaudeCodeComment의 기술 진단은 전체적으로 타당합니다. 특히 `P0` 2개(`smoothstep` 역순 수정, `Cover/Fit` 매핑 통일)는 우선 적용 대상이 맞습니다.  
다만 실제 코드 반영 시 아래 3가지를 함께 보완하는 것을 권장합니다.

### 1) P1-A는 Detection Slot만으로는 불충분

- 현재 stale 렌즈의 직접 원인은 `GpuRenderActivity`에서 GL 렌더러 입력(`setIrisResult`)을 `detected=true`일 때만 갱신하는 구조입니다.
- 따라서 `updateDetectionSlot`만 매 프레임 갱신해도, 렌즈 경로의 `irisResult`가 과거 스냅샷을 계속 참조하면 문제는 남습니다.
- 조치: `cameraGLView.setIrisResult(...)`도 매 프레임 호출하고, 미검출 프레임은 `detected=false` 상태를 명시 전달해야 합니다.

### 2) `!gpuMode` 패치는 빠르지만 의미 혼동 리스크가 큼

- 단기적으로는 `computeScreenTransform(..., !gpuMode)`로 정합성이 맞아질 수 있습니다.
- 그러나 변수명(`gpuMode`)과 실제 동작(`cover/fill`)이 역전되어 유지보수 시 재발 위험이 큽니다.
- 조치: `screenMappingMode`(예: `FIT`, `COVER`) 같은 명시적 정책 변수로 분리하는 리팩토링을 권장합니다.

### 3) GPU별 동작 서술은 관측 근거 수준으로 낮추는 것이 안전

- `smoothstep(edge0 > edge1)`은 사양상 undefined라는 사실만으로 수정 근거가 충분합니다.
- 특정 GPU에서의 반환 경향(예: 0.0/1.0)은 로그/재현 데이터가 첨부되지 않으면 단정 표현 대신 "관측 가능성이 있음" 수준으로 기술하는 편이 문서 신뢰도에 유리합니다.

## 반영 우선순위 제안 (최종)

1. `P0`: `smoothstep` 역순 수정.
2. `P0`: Overlay/실렌더 `Cover` 매핑 통일.
3. `P1`: GL `IrisResult` 매 프레임 갱신(미검출 프레임 포함).
4. `P1`: FaceMesh 미유효 시 클리핑 temporal hold 적용.
5. `P2`: 매핑 정책 enum화(`FIT/COVER`)로 구조 고정.
