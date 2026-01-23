# ISS-001: Face Mesh 세로 늘어남 (Aspect Ratio 왜곡) 수정

## 작업 정보

| 항목 | 내용 |
|------|------|
| 이슈 ID | ISS-001 |
| 상태 | ✅ 완료 |
| 우선순위 | P0 (Critical) |
| 관련 문서 | claudedocs/03_ISSUE_RESOLUTION_INSIGHTS.md |
| 시작일 | 2026-01-21 |
| 완료일 | 2026-01-21 |

---

## 문제 증상

### 문제 A: Face Mesh 세로 늘어남 (Aspect Ratio 왜곡)
1. **Face Mesh 세로 늘어남**: 얼굴 위에 그려지는 Face Mesh가 세로로 ~1.78배 길게 표시
2. **홍채 타원형 변형**: 정상적으로 원형이어야 할 홍채 검출 결과가 타원형으로 왜곡
3. **발생 환경**: 16:9 비율 카메라 (1920x1080 등) 사용 시

### 문제 B: Face Rect 좌표 고정 (얼굴 추적 불가)
1. **Face Rect 고정**: 얼굴 인식 rect가 화면상 고정 위치에 그려지고 얼굴을 따라 이동하지 않음
2. **Face Mesh에도 영향**: Face Rect를 기반으로 Face Mesh가 계산되므로 Mesh도 영향 받음
3. **1프레임 지연**: result.face_rect가 이전 프레임의 값을 사용하여 발생

---

## 근본 원인 분석

### 원인 A: 좌표 변환 로직 오류 (세로 늘어남)
- **위치**: `cpp/src/mediapipe_detector.cpp`
- **문제**: 픽셀 기준 정사각형 crop을 비정사각형 정규화 좌표로 저장할 때 aspect ratio 미적용

**상세 분석:**
```
원본 이미지: 1920x1080 (16:9)
픽셀 정사각형 crop: 600x600
정규화 후: width=0.3125 (600/1920), height=0.5556 (600/1080)
Y축이 X축의 1.78배로 스케일링되어 세로 늘어남 발생
```

**문제 코드 (라인 2207-2210):**
```cpp
impl_->face_landmarks_buffer[i * 3 + 0] = actual_face_crop.x + local_x * actual_face_crop.width;
impl_->face_landmarks_buffer[i * 3 + 1] = actual_face_crop.y + local_y * actual_face_crop.height;
// height이 width와 다른 비율로 적용되어 왜곡 발생
```

### 원인 B: result.face_rect 업데이트 순서 오류 (좌표 고정)
- **위치**: `cpp/src/mediapipe_detector.cpp`
- **문제**: `result.face_rect`가 Face Mesh 처리 **이전**에 설정되어 **이전 프레임**의 값을 사용

**코드 흐름 분석:**
```cpp
// 라인 1994: Face Mesh 처리 전에 result.face_rect 할당
result.face_rect = impl_->prev_face_rect;  // ⚠️ 이전 프레임 값!

// ... Face Mesh 처리 ...

// 라인 2546-2549: Face Mesh 처리 후에 prev_face_rect 업데이트
impl_->prev_face_rect.x = min_x;
impl_->prev_face_rect.y = min_y;
impl_->prev_face_rect.width = max_x - min_x;
impl_->prev_face_rect.height = max_y - min_y;

// result.face_rect는 업데이트되지 않음! → 이전 프레임 값 유지
```

**결과:**
- `result.face_rect`는 항상 1프레임 지연된 값을 가짐
- 얼굴이 빠르게 움직이면 rect가 뒤따라오지 못함
- 정지 상태에서도 이전 프레임 위치에 고정됨

---

## PerfectLib 참고 인사이트

### 1. FaceAlignMotionSmoother의 SetFrameInfo()
PerfectLib은 프레임 정보를 명시적으로 설정하는 메서드를 제공:
```java
class FaceAlignMotionSmoother {
    void SetFrameInfo(int width, int height);
}
```

### 2. FaceAlignData 구조체
좌표 데이터와 함께 프레임 메타데이터를 묶어서 관리:
```java
class UIMakeupLiveFaceAlignData {
    int frameWidth;
    int frameHeight;
    // 좌표 데이터...
}
```

### 3. ApplyRotateCorrect() 분리
회전 보정을 별도 단계로 분리하여 좌표 변환 정확도 향상:
```java
void ApplyRotateCorrect();  // 명시적 호출로 회전 보정 적용
```

### 4. 좌표 업데이트 시점 (문제 B 관련)
PerfectLib은 각 프레임마다 좌표를 명시적으로 업데이트:
```java
// 검출 완료 후 즉시 결과 반영
FaceAlignData result = detector.detect(frame);
// result에 현재 프레임의 좌표가 즉시 포함됨 (지연 없음)
```

현재 IrisLensSDK 문제:
```cpp
result.face_rect = impl_->prev_face_rect;  // 이전 프레임 값 사용
// ... Face Mesh 처리 후 prev_face_rect만 업데이트, result는 그대로
```

### 권장 개선 방향
```cpp
// CoordinateTransformer 클래스로 분리
class CoordinateTransformer {
    void setFrameInfo(int width, int height);
    Point2f cropToNormalized(const Point2f& point, const Rect& crop);
    void applyRotationCorrection(float* landmarks, int count, int rotation);
};
```

---

## 수정 계획

### 수정 파일
`cpp/src/mediapipe_detector.cpp`

### 수정 1: Face Landmark 좌표 변환 (라인 ~2190)

**추가할 코드:**
```cpp
// Aspect ratio 보정
float img_aspect_ratio = static_cast<float>(width) / height;
float crop_scale_x = actual_face_crop.width;
float crop_scale_y = actual_face_crop.width * img_aspect_ratio;
```

**변경할 코드 (라인 2207-2210):**
```cpp
// 수정 전
impl_->face_landmarks_buffer[i * 3 + 0] = actual_face_crop.x + local_x * actual_face_crop.width;
impl_->face_landmarks_buffer[i * 3 + 1] = actual_face_crop.y + local_y * actual_face_crop.height;

// 수정 후
impl_->face_landmarks_buffer[i * 3 + 0] = actual_face_crop.x + local_x * crop_scale_x;
impl_->face_landmarks_buffer[i * 3 + 1] = actual_face_crop.y + local_y * crop_scale_y;
```

### 수정 2: 왼쪽 Iris 좌표 변환 (라인 ~2388)

**추가할 코드:**
```cpp
float eye_crop_scale_x = left_eye_crop.width;
float eye_crop_scale_y = left_eye_crop.width * (static_cast<float>(width) / height);
```

**변경할 코드 (라인 2397-2398):**
```cpp
// 수정 전
result.left_iris[i].x = left_eye_crop.x + local_x * left_eye_crop.width;
result.left_iris[i].y = left_eye_crop.y + local_y * left_eye_crop.height;

// 수정 후
result.left_iris[i].x = left_eye_crop.x + local_x * eye_crop_scale_x;
result.left_iris[i].y = left_eye_crop.y + local_y * eye_crop_scale_y;
```

### 수정 3: 오른쪽 Iris 좌표 변환 (라인 ~2433)

**추가할 코드:**
```cpp
float eye_crop_scale_x = right_eye_crop.width;
float eye_crop_scale_y = right_eye_crop.width * (static_cast<float>(width) / height);
```

**변경할 코드 (라인 2447-2448):**
```cpp
// 수정 전
result.right_iris[i].x = right_eye_crop.x + local_x * right_eye_crop.width;
result.right_iris[i].y = right_eye_crop.y + local_y * right_eye_crop.height;

// 수정 후
result.right_iris[i].x = right_eye_crop.x + local_x * eye_crop_scale_x;
result.right_iris[i].y = right_eye_crop.y + local_y * eye_crop_scale_y;
```

### 수정 4: result.face_rect 업데이트 순서 변경 (라인 ~2550)

**문제 위치**: 라인 1994
```cpp
// 현재 (문제): Face Mesh 처리 전에 설정
result.face_rect = impl_->prev_face_rect;
```

**해결 방법**: Face Mesh 처리 후에 result.face_rect 업데이트

**변경할 코드 (라인 2546-2555 부근):**
```cpp
// Face Mesh 기반 새 face_rect 저장 (다음 프레임 추적용)
impl_->prev_face_rect.x = min_x;
impl_->prev_face_rect.y = min_y;
impl_->prev_face_rect.width = max_x - min_x;
impl_->prev_face_rect.height = max_y - min_y;

// ✅ 추가: 현재 프레임 result에도 반영
result.face_rect = impl_->prev_face_rect;
```

**PerfectLib 인사이트 적용:**
- PerfectLib의 SetFrameInfo() 패턴처럼 각 프레임마다 좌표를 명시적으로 업데이트
- 지연 없이 현재 프레임의 검출 결과를 즉시 반영

---

## 수정 요약

| 문제 | 위치 | 라인 | 작업 |
|------|------|------|------|
| A (세로 늘어남) | Face Landmark | ~2190 | `crop_scale_x/y` 변수 추가 |
| A (세로 늘어남) | Face Landmark | 2207-2210 | 스케일 변수 사용으로 변경 |
| A (세로 늘어남) | Left Iris | ~2388 | `eye_crop_scale_x/y` 변수 추가 |
| A (세로 늘어남) | Left Iris | 2397-2398 | 스케일 변수 사용으로 변경 |
| A (세로 늘어남) | Right Iris | ~2433 | `eye_crop_scale_x/y` 변수 추가 |
| A (세로 늘어남) | Right Iris | 2447-2448 | 스케일 변수 사용으로 변경 |
| B (좌표 고정) | result.face_rect | ~2550 | Face Mesh 처리 후 result.face_rect 업데이트 추가 |

---

## 검증 방법

### 빌드
```bash
cd /Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/android
./gradlew :demo-app:assembleDebug
```

### 기능 검증 (문제 A: 세로 늘어남)
1. **Face Mesh 비율 확인**: 얼굴 형태와 동일한 비율로 Mesh 표시
2. **홍채 원형 확인**: 홍채가 타원이 아닌 원형으로 검출

### 기능 검증 (문제 B: 좌표 고정)
3. **Face Rect 추적**: 얼굴을 좌우/상하로 이동 시 Face Rect가 즉시 따라오는지 확인
4. **지연 없음 확인**: 빠르게 얼굴을 움직여도 Face Rect가 1프레임 이상 지연되지 않는지
5. **Face Mesh 추적**: Face Rect와 함께 Face Mesh도 얼굴을 정확히 추적하는지

### 다양한 해상도 테스트
- 16:9 (1920x1080, 1280x720)
- 4:3 (640x480)
- 정사각형 (1080x1080)

---

## 향후 개선 방향

### Phase 2 고려사항
1. **CoordinateTransformer 클래스 분리**: 좌표 변환 로직을 별도 클래스로 추출
2. **프레임 메타데이터 명시적 관리**: PerfectLib의 SetFrameInfo() 패턴 도입
3. **회전 보정 분리**: ApplyRotateCorrect() 패턴 적용

---

## 변경 이력

| 날짜 | 작업자 | 내용 |
|------|--------|------|
| 2026-01-21 | Claude | 작업 계획 문서 작성 |
| 2026-01-21 | Claude | 수정 1-3 (aspect ratio 보정) - 이미 적용됨 확인 |
| 2026-01-21 | Claude | 수정 4 (result.face_rect 업데이트) - 라인 ~2562에 추가 완료 |
| 2026-01-21 | Claude | Mac 테스트 검증 완료 - android_screenshot.jpg로 테스트 시 Face Mesh/Face Rect 정상 표시 확인 |

---

## 추가 검증 결과 (2026-01-21)

### Mac 버전 테스트 결과
- **테스트 이미지**: `shared/test_data/android_screenshot.jpg` (1080x2140)
- **결과**: Face Mesh, Face Rect 모두 정상적으로 얼굴에 맞게 표시됨
- **결론**: C++ SDK 코어 로직은 정상 동작

### Android 스크린샷 분석
| 파일 | 표시 항목 | 비율 상태 |
|------|----------|----------|
| 1000011307.jpg | Face Mesh + Contour | 정상으로 보임 |
| 1000011321.jpg | Face Rect만 | 정상으로 보임 |
| 1000011323.jpg | Face Rect만 | 정상으로 보임 |

### 추가 조사 필요 사항
- 라이브 프리뷰에서 특정 조건에서만 문제 발생 여부 확인
- 특정 카메라 해상도(예: 4:3 vs 16:9)에서 차이 확인
- Android 렌더링 레이어(OverlayView.kt) 추가 분석 필요시 진행
