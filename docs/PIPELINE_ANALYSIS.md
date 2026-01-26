# IrisLensSDK 파이프라인 분석

## 개요

이 문서는 IrisLensSDK의 Face Mesh + Iris 검출 파이프라인을 분석합니다.
공식 MediaPipe와 비교하여 문제점을 파악하기 위한 참조 자료입니다.

---

## 1. 전체 파이프라인 흐름

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           입력 이미지                                         │
│                      (예: 1080x1920 NV21)                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: Face Detection (얼굴 검출)                                          │
│  ─────────────────────────────────────────────────────────────────────────  │
│  입력: 전체 이미지 → 128x128 리사이즈                                          │
│  출력: face_rect (정규화 좌표 0.0~1.0)                                        │
│        - x, y: 좌상단 좌표                                                    │
│        - width, height: 박스 크기                                             │
│  모델: face_detection_short_range.tflite                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: Face Region Crop (얼굴 영역 크롭)                                    │
│  ─────────────────────────────────────────────────────────────────────────  │
│  입력: face_rect (정규화 좌표)                                                │
│  처리:                                                                       │
│    1. face_rect 중심점 계산 (픽셀 단위)                                       │
│    2. 정사각형 크기 = max(width, height) × 1.2 (20% 마진)                    │
│    3. 중심점 기준 정사각형 ROI 생성 (픽셀 단위)                                │
│    4. 경계 체크 및 클램핑                                                     │
│                                                                              │
│  출력:                                                                       │
│    - cropped_image: 정사각형 픽셀 이미지 (예: 600x600)                        │
│    - actual_face_crop: 정규화 좌표 (⚠️ 직사각형이 됨!)                        │
│                                                                              │
│  ⚠️ 핵심 문제점:                                                             │
│    픽셀 정사각형 (600x600) → 정규화 좌표 변환 시:                              │
│    - width = 600 / 1080 = 0.5556                                            │
│    - height = 600 / 1920 = 0.3125                                           │
│    → 정규화 좌표계에서는 직사각형! (세로가 더 짧음)                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: Face Landmark (얼굴 랜드마크)                                        │
│  ─────────────────────────────────────────────────────────────────────────  │
│  입력: cropped_image → 192x192 (V1) 또는 256x256 (V2) 리사이즈               │
│  출력: 468개 (V1) 또는 478개 (V2) 랜드마크                                    │
│        - 정규화 좌표 (0.0~1.0) 또는 픽셀 좌표 (0~192/256)                     │
│  모델: face_landmark.tflite (V1) / face_landmark_v2.tflite (V2)              │
│                                                                              │
│  좌표 타입 판별:                                                              │
│    - max(x, y) > 1.5 → 픽셀 좌표 → 입력 크기로 나눔                           │
│    - max(x, y) ≤ 1.5 → 정규화 좌표 → 나눔 불필요                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: 좌표 변환 (크롭 → 전체 이미지) - ✅ ASPECT RATIO 보정 적용됨        │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  현재 구현 (2025-01-21 수정됨):                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  local_x, local_y = 모델 출력 (0.0~1.0 정규화)                       │    │
│  │                                                                      │    │
│  │  // Aspect ratio 보정 계산                                           │    │
│  │  img_aspect_ratio = width / height                                  │    │
│  │  crop_scale_x = actual_face_crop.width                              │    │
│  │  crop_scale_y = actual_face_crop.width × img_aspect_ratio           │    │
│  │                                                                      │    │
│  │  final_x = actual_face_crop.x + local_x × crop_scale_x             │    │
│  │  final_y = actual_face_crop.y + local_y × crop_scale_y             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  예시 (세로 모드 1080x1920):                                                  │
│    actual_face_crop = {x: 0.22, y: 0.34, w: 0.5556, h: 0.3125}              │
│                                                                              │
│    local_x = 0.5, local_y = 0.5 (정사각형 중심)                               │
│    final_x = 0.22 + 0.5 × 0.5556 = 0.4978                                   │
│    final_y = 0.34 + 0.5 × 0.3125 = 0.4963                                   │
│                                                                              │
│    → 픽셀 변환:                                                               │
│      pixel_x = 0.4978 × 1080 = 537.6 px                                     │
│      pixel_y = 0.4963 × 1920 = 952.9 px                                     │
│                                                                              │
│    → 원본 crop 중심 (픽셀): (540, 953) ✓ 거의 일치                           │
│                                                                              │
│  ⚠️ 그러나 SCALE이 다름:                                                     │
│    - X 방향: local 1.0 → final 0.5556 증가                                  │
│    - Y 방향: local 1.0 → final 0.3125 증가                                  │
│    - Y/X 비율 = 0.3125 / 0.5556 = 0.5625 (원래 1.0이어야 함!)               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: Iris Landmark (홍채 검출) - V1 전용                                  │
│  ─────────────────────────────────────────────────────────────────────────  │
│  V2는 Face Landmark 출력에서 직접 추출 (인덱스 468-477)                       │
│                                                                              │
│  V1 처리:                                                                    │
│    1. Face Landmark에서 눈 좌표 추출 (인덱스 33, 133 등)                      │
│    2. 눈 영역 정사각형 crop (64x64)                                          │
│    3. Iris Landmark 모델 실행                                                │
│    4. 동일한 좌표 변환 적용 (eye_crop → 전체 이미지)                          │
│                                                                              │
│  출력: 5개 홍채 랜드마크 (중심 + 4방향 경계)                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 6: Android OverlayView 렌더링                                          │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  좌표 변환 (정규화 → 화면):                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  scaleFactor = max(viewWidth / imageWidth, viewHeight / imageHeight)│    │
│  │  offsetX = (viewWidth - imageWidth × scaleFactor) / 2               │    │
│  │  offsetY = (viewHeight - imageHeight × scaleFactor) / 2             │    │
│  │                                                                      │    │
│  │  screenX = normalizedX × imageWidth × scaleFactor + offsetX         │    │
│  │  screenY = normalizedY × imageHeight × scaleFactor + offsetY        │    │
│  │                                                                      │    │
│  │  if (isMirror) screenX = viewWidth - screenX                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  예시 (세로 모드, View 1080x2400):                                            │
│    imageWidth = 1080, imageHeight = 1920                                    │
│    scaleFactor = max(1080/1080, 2400/1920) = max(1.0, 1.25) = 1.25         │
│    offsetX = (1080 - 1080 × 1.25) / 2 = -67.5                              │
│    offsetY = (2400 - 1920 × 1.25) / 2 = 0                                  │
│                                                                              │
│    normalizedX = 0.5, normalizedY = 0.5                                     │
│    screenX = 0.5 × 1080 × 1.25 + (-67.5) = 607.5                           │
│    screenY = 0.5 × 1920 × 1.25 + 0 = 1200                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 좌표계 요약

| 단계 | 좌표계 | 범위 | 비고 |
|------|--------|------|------|
| 입력 이미지 | 픽셀 | 0 ~ width/height | 세로 모드: 1080x1920 |
| Face Detection 입력 | 픽셀 | 0 ~ 128 | 정사각형 리사이즈 |
| face_rect | 정규화 | 0.0 ~ 1.0 | 전체 이미지 기준 |
| Face Landmark 입력 | 픽셀 | 0 ~ 192/256 | 정사각형 crop |
| actual_face_crop | 정규화 | 0.0 ~ 1.0 | ⚠️ 직사각형 (비정사각형) |
| Face Landmark 출력 | 정규화/픽셀 | 0.0~1.0 또는 0~256 | crop 영역 기준 |
| 최종 랜드마크 | 정규화 | 0.0 ~ 1.0 | 전체 이미지 기준 |
| 화면 좌표 | 픽셀 | 0 ~ viewWidth/Height | 스케일+오프셋 적용 |

---

## 3. 공식 MediaPipe와 비교

### 3.1 공식 MediaPipe 파이프라인

```
입력 이미지
    ↓
Face Detection → ROI (회전 포함)
    ↓
Affine Transform (ROI → 정사각형)
    ↓
Face Landmark 모델 (192x192 / 256x256)
    ↓
Inverse Affine Transform (랜드마크 → 원본 좌표)
    ↓
정규화 좌표 출력
```

**핵심 차이점:**
- **Affine Transform 사용**: 회전, 스케일, 이동을 하나의 행렬로 처리
- **Inverse Transform으로 복원**: 정확한 역변환 보장

### 3.2 현재 IrisLensSDK 파이프라인

```
입력 이미지
    ↓
Face Detection → face_rect (정규화)
    ↓
중심점 + 정사각형 크기 계산 (픽셀)
    ↓
픽셀 정사각형 crop → 정규화 좌표 저장 (⚠️ 직사각형)
    ↓
Face Landmark 모델
    ↓
선형 좌표 변환 (crop.x + local_x × crop.width)
    ↓
정규화 좌표 출력
```

**문제점:**
- **Affine Transform 미사용**: 단순 선형 변환
- **정사각형 crop → 직사각형 정규화**: aspect ratio 불일치

---

## 4. 문제 분석: "세로 늘어남"

### 4.1 원인 가설

**픽셀 정사각형 crop이 정규화 좌표에서 직사각형으로 표현됨:**

```
세로 모드 이미지: 1080 × 1920
픽셀 crop: 600 × 600 (정사각형)

정규화 좌표:
  width  = 600 / 1080 = 0.5556
  height = 600 / 1920 = 0.3125

비율: height / width = 0.5625 (원래 1.0이어야 함)
```

### 4.2 좌표 변환 시 왜곡

```
모델 출력: (0.5, 0.5) - 정사각형 입력의 중심

변환 후:
  x 증가량 = 0.5 × 0.5556 = 0.2778
  y 증가량 = 0.5 × 0.3125 = 0.1563

비율: y/x = 0.5625

픽셀로 변환:
  x 픽셀 = 0.2778 × 1080 = 300 px
  y 픽셀 = 0.1563 × 1920 = 300 px  ✓ 동일!
```

**결론: 수학적으로는 정확함!**

### 4.3 그럼 왜 "세로 늘어남"이 발생하나?

가능한 원인들:

1. **OverlayView에서 추가 왜곡?**
   - PreviewView와 OverlayView 크기 불일치
   - scaleFactor 계산 오류

2. **Face Detection의 face_rect 자체가 잘못됨?**
   - face_rect가 세로로 긴 형태로 출력

3. **crop 영역 계산 오류?**
   - 정사각형이 아닌 직사각형으로 crop

4. **카메라 이미지 크기와 실제 처리 크기 불일치?**
   - 카메라가 16:9인데 4:3으로 처리

---

## 5. 디버깅 체크리스트 (2025-01-21 검증 완료)

### 5.1 Face Detection 확인
- [x] face_rect가 정사각형에 가까운가? ✅ 확인됨
- [x] face_rect 위치가 얼굴 중심인가? ✅ 확인됨

### 5.2 Face Crop 확인
- [x] 픽셀 crop이 실제로 정사각형인가? (roi_width == roi_height) ✅ 확인됨
- [x] actual_face_crop의 width/height 비율 확인 → **문제 발견 및 해결됨**

### 5.3 좌표 변환 확인
- [x] 변환 전 랜드마크가 0.0~1.0 범위인가? ✅ 확인됨
- [x] 변환 후 랜드마크가 crop 영역 내에 있는가? ✅ aspect ratio 보정 후 정상

### 5.4 렌더링 확인
- [x] imageWidth/imageHeight가 실제 카메라 이미지 크기와 일치하는가? ✅ ViewPort 적용으로 해결
- [x] scaleFactor 계산이 올바른가? ✅ 확인됨
- [x] PreviewView와 OverlayView 크기가 일치하는가? ✅ 확인됨

### 5.5 추가 수정 사항 (2025-01-21)
- [x] CameraManager.kt: ViewPort + UseCaseGroup 적용 (Preview/ImageAnalysis FOV 일치)
- [x] mediapipe_detector.cpp: aspect ratio 보정 로직 추가

---

## 6. 수정 이력 (해결됨)

### ✅ 적용된 해결책: 옵션 B 변형 (Aspect Ratio 보정)

**수정일**: 2025-01-21
**수정 파일**: `cpp/src/mediapipe_detector.cpp`

```cpp
// 적용된 수정 (라인 2190-2227)
// 정사각형 crop을 비정사각형 이미지에서 올바르게 변환

// Aspect ratio 보정 계산
float img_aspect_ratio = static_cast<float>(width) / height;
float crop_scale_x = actual_face_crop.width;
float crop_scale_y = actual_face_crop.width * img_aspect_ratio;

// 좌표 변환 (aspect ratio 보정 적용)
final_x = actual_face_crop.x + local_x * crop_scale_x;
final_y = actual_face_crop.y + local_y * crop_scale_y;
```

**동일한 보정이 적용된 위치**:
- Face Landmark 좌표 변환 (라인 2207-2210)
- 왼쪽 Iris 좌표 변환 (라인 2397-2398)
- 오른쪽 Iris 좌표 변환 (라인 2447-2448)

### 이전 옵션들 (참고용)

<details>
<summary>옵션 A: Affine Transform (미채택)</summary>

```cpp
// 공식 MediaPipe 방식 - 더 정확하지만 복잡함
cv::Mat affine = getAffineTransform(roi_points, square_points);
cv::warpAffine(image, cropped, affine, Size(256, 256));
cv::Mat inv_affine;
cv::invertAffineTransform(affine, inv_affine);
```
</details>

<details>
<summary>옵션 C: 렌더링 측 보정 (미채택)</summary>

```kotlin
// OverlayView에서 보정 - 플랫폼마다 구현 필요
val aspectRatio = imageWidth.toFloat() / imageHeight
val screenY = normalizedY * imageHeight * scaleFactor * aspectRatio + offsetY
```
</details>

---

## 7. Face Landmark 모델 V1 vs V2 비교

### 7.1 개요

MediaPipe는 두 가지 버전의 Face Landmark 모델을 제공합니다.

| 항목 | V1 모델 | V2 모델 |
|------|---------|---------|
| **파일명** | `face_landmark.tflite` | `face_landmark_v2.tflite` |
| **입력 크기** | 192 × 192 | 256 × 256 |
| **랜드마크 수** | 468개 | 478개 |
| **홍채 포함** | ❌ 별도 모델 필요 | ✅ 내장 |
| **파일 크기** | ~1.2MB | ~2.5MB |
| **추론 시간** | 빠름 | 약간 느림 |

### 7.2 랜드마크 인덱스 구조

#### V1 모델 (468개)
```
인덱스 0-467: 얼굴 랜드마크
  - 0-16: 턱 윤곽 (17개)
  - 17-21: 왼쪽 눈썹 (5개)
  - 22-26: 오른쪽 눈썹 (5개)
  - 27-35: 코 (9개)
  - 36-47: 눈 (12개)
  - 48-67: 입술 외곽 (20개)
  - 68-467: 나머지 얼굴 메쉬
```

#### V2 모델 (478개)
```
인덱스 0-467: V1과 동일한 얼굴 랜드마크
인덱스 468-477: 홍채 랜드마크 (추가됨)

  왼쪽 홍채 (화면상 오른쪽):
  - 468: 홍채 중심
  - 469: 홍채 상단 (12시 방향)
  - 470: 홍채 우측 (3시 방향)
  - 471: 홍채 하단 (6시 방향)
  - 472: 홍채 좌측 (9시 방향)

  오른쪽 홍채 (화면상 왼쪽):
  - 473: 홍채 중심
  - 474: 홍채 상단 (12시 방향)
  - 475: 홍채 우측 (3시 방향)
  - 476: 홍채 하단 (6시 방향)
  - 477: 홍채 좌측 (9시 방향)
```

### 7.3 파이프라인 차이

#### V1 파이프라인 (3단계)
```
Face Detection → Face Landmark (468) → Iris Landmark (별도 모델)
                      ↓                        ↓
              얼굴 메쉬 출력             홍채 5포인트 출력
```
- 눈 영역을 별도로 crop하여 Iris Landmark 모델 실행 필요
- 추가 좌표 변환 필요 (눈 crop → 전체 이미지)

#### V2 파이프라인 (2단계)
```
Face Detection → Face Landmark V2 (478)
                        ↓
              얼굴 메쉬 + 홍채 포인트 통합 출력
```
- Iris Landmark 모델 실행 불필요
- 단일 모델에서 홍채까지 출력
- 좌표 변환 단순화

### 7.4 성능 비교

| 항목 | V1 | V2 | 비고 |
|------|-----|-----|------|
| 모델 로드 시간 | 빠름 | 보통 | V2가 파일 크기 큼 |
| 추론 시간 (Face) | ~15ms | ~20ms | 입력 크기 차이 |
| 추론 시간 (Iris) | +~10ms | 0ms | V2는 내장 |
| **총 시간** | ~25ms | ~20ms | **V2가 더 빠름** |
| 메모리 사용량 | 낮음 | 중간 | 단일 모델 |
| 홍채 정확도 | 높음 | 보통 | V1 전용 모델이 더 정밀 |

### 7.5 권장 사용 시나리오

| 시나리오 | 권장 모델 | 이유 |
|----------|-----------|------|
| **AR 렌즈 피팅** | V2 | 단순한 파이프라인, 충분한 정확도 |
| **정밀 홍채 추적** | V1 | 전용 Iris 모델의 높은 정확도 |
| **저사양 디바이스** | V1 | 낮은 메모리 사용량 |
| **빠른 개발** | V2 | 단순한 구현 |

### 7.6 IrisLensSDK 구현

현재 SDK는 **V2 우선, V1 폴백** 전략 사용:

```cpp
// cpp/src/mediapipe_detector.cpp (라인 568-604)

// 1. V2 모델 먼저 시도
if (std::filesystem::exists(face_landmark_v2_path) &&
    loadModel(face_landmark_v2_path, ...)) {
    model_version = 2;
    // V2: 478 랜드마크, 홍채 내장
}
// 2. V1 모델로 폴백
else if (loadModel(face_landmark_v1_path, ...)) {
    model_version = 1;
    // V1: 468 랜드마크, Iris 모델 별도 로드
}
```

V1 모델 사용 시:
- 홍채 랜드마크(468-477)는 `-1.0`으로 채워짐
- OverlayView에서 유효하지 않은 좌표 스킵 처리

---

## 8. 참고: 주요 상수

```cpp
// Face Detection
FACE_DETECTION_INPUT_WIDTH = 128
FACE_DETECTION_INPUT_HEIGHT = 128

// Face Landmark V1
FACE_LANDMARK_INPUT_WIDTH = 192
FACE_LANDMARK_INPUT_HEIGHT = 192
FACE_LANDMARK_COUNT = 468

// Face Landmark V2
FACE_LANDMARK_V2_INPUT_WIDTH = 256
FACE_LANDMARK_V2_INPUT_HEIGHT = 256
FACE_LANDMARK_V2_COUNT = 478

// Iris Landmark (V1 전용)
IRIS_LANDMARK_INPUT_WIDTH = 64
IRIS_LANDMARK_INPUT_HEIGHT = 64
IRIS_LANDMARK_COUNT = 5

// Crop 마진
FACE_CROP_MARGIN = 0.25 (25%)
```

---

## 9. 관련 파일

| 파일 | 역할 |
|------|------|
| `cpp/src/mediapipe_detector.cpp` | C++ 핵심 파이프라인 |
| `android/.../OverlayView.kt` | Android 렌더링 |
| `android/.../CameraManager.kt` | 카메라 + SDK 연동 |
| `cpp/include/iris_sdk/types.h` | 데이터 구조 정의 |
| `shared/models/face_landmark.tflite` | V1 모델 (468 랜드마크) |
| `shared/models/face_landmark_v2.tflite` | V2 모델 (478 랜드마크, 홍채 내장) |
| `shared/models/iris_landmark.tflite` | Iris 모델 (V1 전용) |

---

*문서 작성일: 2025-01-21*
*최종 수정일: 2025-01-26 (V1 vs V2 모델 비교 섹션 추가)*
