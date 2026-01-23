# ISS-002: C++ Demo vs Android Demo 좌표 처리 분석

## 문서 정보
- **작성일**: 2025-01-22
- **상태**: ✅ 수정 완료 (검증 대기)
- **관련 파일**:
  - `cpp/src/mediapipe_detector.cpp` (핵심 버그 수정)
  - `cpp/examples/camera_demo.cpp`
  - `android/demo-app/src/main/java/com/irislenssdk/demo/camera/OverlayView.kt`
  - `android/demo-app/src/main/java/com/irislenssdk/demo/camera/CameraManager.kt`
  - `android/demo-app/src/main/java/com/irislenssdk/demo/MainActivity.kt`

---

## 1. 문제 상황

C++ demo는 정상 동작하나 Android demo에서 좌표가 맞지 않는 문제 발생.

---

## 2. 좌표 처리 흐름 비교

### 2.1 C++ Demo (단순 구조)

```
카메라 1280x720
    ↓ cv::flip(frame, 1) - 미러링
반전된 프레임
    ↓ SDK 검출
정규화 좌표 (0.0~1.0)
    ↓ x * frame.cols, y * frame.rows
픽셀 좌표
    ↓ cv::circle() 등
화면 렌더링 (화면 = 프레임 크기)
```

**핵심**: 프레임 크기 = 윈도우 크기이므로 스케일링 불필요

### 2.2 Android Demo (복잡 구조)

```
카메라 640x480 (원본, 가로)
    ↓ rotationDegrees (90°)
회전 후 480x640 (SDK 처리, 세로)
    ↓ SDK 검출
정규화 좌표 (0.0~1.0)
    ↓ x * imageWidth * scaleFactor + offsetX
화면 좌표 (1080x1920 등)
    ↓ if(isMirror) cx = width - cx
최종 미러링된 좌표
```

**핵심**: 프레임 크기 ≠ 화면 크기이므로 스케일링 필수

---

## 3. Android에서 스케일링이 필요한 이유

### 3.1 다양한 해상도

| 구분 | 크기 예시 |
|------|----------|
| 카메라 해상도 | 480x640, 720x1280 |
| 화면 해상도 | 1080x1920, 1440x3200 |
| PreviewView | 전체 화면 또는 일부 |
| OverlayView | PreviewView와 동일 |

### 3.2 PreviewView FILL_CENTER 모드

```kotlin
// PreviewView가 카메라 이미지를 확대하여 화면 채움
val scaleFactor = max(viewWidth / imageWidth, viewHeight / imageHeight)
// 예: max(1080/480, 1920/640) = max(2.25, 3.0) = 3.0
```

### 3.3 좌표 변환 공식

```kotlin
// 1. 스케일 팩터 계산
val scaleFactor = max(width.toFloat() / imageWidth, height.toFloat() / imageHeight)

// 2. 스케일된 이미지 크기
val scaledImageWidth = imageWidth * scaleFactor
val scaledImageHeight = imageHeight * scaleFactor

// 3. 중앙 정렬 오프셋
val offsetX = (width - scaledImageWidth) / 2f
val offsetY = (height - scaledImageHeight) / 2f

// 4. 정규화 좌표 → 화면 좌표
var cx = normalizedX * imageWidth * scaleFactor + offsetX
val cy = normalizedY * imageHeight * scaleFactor + offsetY

// 5. 미러링 (전면 카메라)
if (isMirror) {
    cx = width - cx
}
```

---

## 4. 발견된 버그

### 4.1 `drawDebugInfo()`에서 faceRect 좌표 처리 오류

**위치**: `OverlayView.kt` Line 527-531

**현재 코드 (버그)**:
```kotlin
var left = result.faceRectX * scaleFactor + offsetX  // ❌ imageWidth 누락
val top = result.faceRectY * scaleFactor + offsetY
var right = (result.faceRectX + result.faceRectWidth) * scaleFactor + offsetX
val bottom = (result.faceRectY + result.faceRectHeight) * scaleFactor + offsetY
```

**수정 코드**:
```kotlin
var left = result.faceRectX * imageWidth * scaleFactor + offsetX  // ✅ 정규화 좌표 처리
val top = result.faceRectY * imageHeight * scaleFactor + offsetY
var right = (result.faceRectX + result.faceRectWidth) * imageWidth * scaleFactor + offsetX
val bottom = (result.faceRectY + result.faceRectHeight) * imageHeight * scaleFactor + offsetY
```

**원인**: `faceRectX/Y`는 정규화 좌표(0.0~1.0)인데 `imageWidth`를 곱하지 않음

**참조 - C++ Demo의 올바른 처리** (`camera_demo.cpp` Line 633-638):
```cpp
cv::Rect face_rect(
    static_cast<int>(iris.face_rect.x * frame.cols),      // 정규화 × 프레임너비
    static_cast<int>(iris.face_rect.y * frame.rows),
    static_cast<int>(iris.face_rect.width * frame.cols),
    static_cast<int>(iris.face_rect.height * frame.rows)
);
```

### 4.2 `drawFaceRect()`는 올바르게 구현됨

**위치**: `OverlayView.kt` Line 478-484

```kotlin
// ✅ 올바른 구현
var left = result.faceRectX * imageWidth * scaleFactor + offsetX
val top = result.faceRectY * imageHeight * scaleFactor + offsetY
var right = (result.faceRectX + result.faceRectWidth) * imageWidth * scaleFactor + offsetX
val bottom = (result.faceRectY + result.faceRectHeight) * imageHeight * scaleFactor + offsetY
```

### 4.3 🔴 **C++ SDK: Face Mesh 좌표 변환 버그 (핵심 버그)**

**위치**: `cpp/src/mediapipe_detector.cpp` Line 2285-2287

**증상**:
- Face Mesh가 얼굴 위치를 제대로 추적하지 못함
- 좌표가 항상 0.3~0.6 범위에 머무름
- 화면 끝으로 이동해도 좌표 범위가 좁음

**문제 코드 (버그)**:
```cpp
float img_aspect_ratio = static_cast<float>(width) / height;
float crop_scale_x = actual_face_crop.width;
float crop_scale_y = actual_face_crop.width * img_aspect_ratio;  // ❌ 버그!
```

**버그 원인**:
`crop_scale_y = actual_face_crop.width * img_aspect_ratio` 공식은 **정사각형 픽셀 크롭**에서만 정확함.

`cropFaceRegion()`에서 경계 클램핑으로 비정사각형 크롭이 생성되면:
- `roi_width ≠ roi_height`
- `actual_face_crop.width * img_aspect_ratio ≠ actual_face_crop.height`
- Y 좌표 스케일이 잘못됨

**수학적 증명**:
```
정사각형 픽셀 크롭 S×S:
- actual_face_crop.width = S / img_width
- actual_face_crop.height = S / img_height
- actual_face_crop.width * img_aspect_ratio
  = (S / img_width) * (img_width / img_height)
  = S / img_height
  = actual_face_crop.height  ✓ (정사각형일 때만 성립)

비정사각형 크롭 W×H (W≠H):
- actual_face_crop.width = W / img_width
- actual_face_crop.height = H / img_height
- actual_face_crop.width * img_aspect_ratio
  = (W / img_width) * (img_width / img_height)
  = W / img_height
  ≠ H / img_height  ✗ (잘못된 값!)
```

**피드백 루프 문제**:
Face Mesh 좌표가 잘못 변환 → `prev_face_rect` 잘못 계산 → 다음 프레임 추적 영역 잘못됨 → 연쇄적 오류 증폭

**수정 코드**:
```cpp
// ISS-002 수정: actual_face_crop.height 직접 사용
float crop_scale_x = actual_face_crop.width;
float crop_scale_y = actual_face_crop.height;  // ✅ 정확한 값
```

---

## 5. 좌표 처리 비교 요약

| 항목 | C++ Demo | Android Demo | 비고 |
|------|----------|--------------|------|
| 미러링 | 프레임 먼저 `cv::flip` | 좌표 변환 후 X 반전 | 둘 다 올바름 |
| 스케일링 | 없음 (프레임=화면) | FILL_CENTER (scaleFactor + offset) | Android 특성 |
| 홍채 좌표 | `x * frame.cols` | `x * imageWidth * scaleFactor + offsetX` | ✅ 올바름 |
| 반지름 | 그대로 사용 (픽셀) | `radius * scaleFactor` | ✅ 올바름 |
| faceRect | `x * frame.cols` | **drawDebugInfo에서 버그** | ❌ 수정 필요 |
| Face Mesh | `x * frame.cols` | `x * imageWidth * scaleFactor + offsetX` | ✅ 올바름 |

---

## 6. 수정 계획

### 6.1 즉시 수정

- [x] `OverlayView.kt` `drawDebugInfo()` 함수의 faceRect 좌표 처리 수정 ✅
- [x] `mediapipe_detector.cpp` crop_scale_y 버그 수정 ✅

### 6.2 검증

- [ ] Debug 모드에서 faceRect 표시 확인
- [ ] 홍채 마커와 faceRect 위치 일치 확인
- [ ] Face Mesh가 얼굴 전체 범위(0~1)로 추적되는지 확인
- [ ] 화면 끝으로 이동 시 좌표 범위 확인

---

## 7. 수정된 코드

### OverlayView.kt Line 526-532

**Before (버그)**:
```kotlin
// 얼굴 바운딩 박스 (faceRectX/Y는 픽셀 좌표)
if (result.faceRectWidth > 0 && result.faceRectHeight > 0) {
    var left = result.faceRectX * scaleFactor + offsetX
    val top = result.faceRectY * scaleFactor + offsetY
    var right = (result.faceRectX + result.faceRectWidth) * scaleFactor + offsetX
    val bottom = (result.faceRectY + result.faceRectHeight) * scaleFactor + offsetY
```

**After (수정됨)**:
```kotlin
// 얼굴 바운딩 박스 (faceRectX/Y는 정규화 좌표 0.0~1.0)
// ISS-002 수정: imageWidth/Height 곱셈 추가 (정규화 좌표 → 화면 좌표)
if (result.faceRectWidth > 0 && result.faceRectHeight > 0) {
    var left = result.faceRectX * imageWidth * scaleFactor + offsetX
    val top = result.faceRectY * imageHeight * scaleFactor + offsetY
    var right = (result.faceRectX + result.faceRectWidth) * imageWidth * scaleFactor + offsetX
    val bottom = (result.faceRectY + result.faceRectHeight) * imageHeight * scaleFactor + offsetY
```

### mediapipe_detector.cpp Line 2285-2296

**Before (버그)**:
```cpp
// Aspect ratio 보정: 픽셀 기준 정사각형 crop을 비정사각형 정규화 좌표로 변환할 때
// Y 스케일이 X와 다르게 되는 문제를 보정
float img_aspect_ratio = static_cast<float>(width) / height;
float crop_scale_x = actual_face_crop.width;
float crop_scale_y = actual_face_crop.width * img_aspect_ratio;
```

**After (수정됨)**:
```cpp
// ISS-002 수정: crop_scale_y를 actual_face_crop.height 직접 사용
// Face Mesh 모델 출력 좌표 (local_x, local_y)는 256x256 정사각형 입력 기준 0-1 범위
// 이를 원본 이미지의 정규화 좌표로 변환:
//   final_x = actual_face_crop.x + local_x * actual_face_crop.width
//   final_y = actual_face_crop.y + local_y * actual_face_crop.height
float crop_scale_x = actual_face_crop.width;
float crop_scale_y = actual_face_crop.height;  // ISS-002: 직접 height 사용
```

---

## 8. 변경 이력

| 날짜 | 변경 내용 | 작성자 |
|------|----------|--------|
| 2025-01-22 | 문서 초안 작성 | Claude |
| 2025-01-22 | `drawDebugInfo()` faceRect 좌표 버그 수정 완료 | Claude |
| 2025-01-22 | 🔴 **핵심 버그 발견**: C++ SDK crop_scale_y 계산 오류 | Claude |
| 2025-01-22 | `mediapipe_detector.cpp` crop_scale_y 버그 수정 완료 | Claude |
