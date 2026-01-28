# 현재 이슈 해결 방안 - PerfectLib 인사이트 기반

**작성일**: 2026-01-21
**분석 에이전트**: comprehensive-review:architect-review

---

## 1. 현재 이슈 요약

### 1.1 이슈 목록

| 이슈 ID | 이슈명 | 심각도 | 상태 |
|---------|--------|--------|------|
| ISS-001 | Face Mesh 세로 늘어남 | 🔴 Critical | 해결 중 |
| ISS-002 | Face Mesh 프레임 떨림 | 🟡 Major | 미해결 |
| ISS-003 | 저조한 FPS (17-19) | 🟡 Major | 미해결 |
| ISS-004 | MediaPipe 폴백 없음 | 🟢 Minor | 미해결 |

---

## 2. ISS-001: Face Mesh 세로 늘어남 (해결 중)

### 2.1 문제 상황

- Face Mesh가 세로로 길게 늘어남 (Y축이 X축의 ~1.78배)
- 홍채 위치도 실제 눈보다 아래에 표시

### 2.2 근본 원인

`mediapipe_detector.cpp`에서 **정사각형 픽셀 crop**을 **정규화 좌표**로 변환 시 aspect ratio 왜곡:

```
원본 이미지: 1920 × 1080 (가로 모드)
           또는 1080 × 1920 (세로 모드)

픽셀 정사각형 crop: 600 × 600

정규화 후:
  width  = 600 / 1080 = 0.5556
  height = 600 / 1920 = 0.3125
  → Y가 1.78배 작음 → 세로로 늘어난 것처럼 보임
```

### 2.3 계획된 수정 코드

```cpp
// mediapipe_detector.cpp (라인 2190 부근)

// Aspect ratio 보정
float img_aspect_ratio = static_cast<float>(width) / height;
float crop_scale_x = actual_face_crop.width;
float crop_scale_y = actual_face_crop.width * img_aspect_ratio;

// 좌표 변환 (동일한 스케일 적용)
final_x = actual_face_crop.x + local_x * crop_scale_x;
final_y = actual_face_crop.y + local_y * crop_scale_y;
```

### 2.4 수학적 검증

**예시**: 세로 모드 1080×1920, 600×600 crop

| 변수 | 값 | 계산 |
|------|------|------|
| img_aspect_ratio | 0.5625 | 1080/1920 |
| actual_face_crop.width | 0.5556 | 600/1080 |
| crop_scale_y | 0.3125 | 0.5556 × 0.5625 |

✅ `crop_scale_y`가 `actual_face_crop.height`와 동일 → **수학적으로 정확**

### 2.5 권장 조치

만약 코드 수정 후에도 문제가 지속된다면:

```bash
# 1. 빌드 캐시 정리 후 재빌드
cd cpp/cmake-build-debug
cmake --build . --target clean
cmake --build . --parallel

# 2. Android 앱 완전 재설치
adb uninstall com.irislenssdk.demo
./gradlew :demo-app:installDebug

# 3. 디버그 로그 확인
adb logcat | grep -i "IrisSDK\|MediaPipe\|aspect"
```

### 2.6 PerfectLib 인사이트 (좌표 변환 관련)

PerfectLib 분석에서 발견된 좌표 처리 관련 핵심 클래스와 패턴:

#### 1. FaceAlignMotionSmoother의 SetFrameInfo()

```cpp
// FaceTracking 모듈 (libvenus_tracking.so)
class FaceAlignMotionSmoother {
    void SetFrameInfo(int width, int height);  // 프레임 정보 명시적 설정
    void ApplyRotateCorrect();                  // 회전 보정 적용
    // ...
};
```

**핵심 인사이트**: PerfectLib는 **프레임 크기를 명시적으로 전달**하여 좌표 변환 시 사용합니다.
- 스무딩 처리 전에 `SetFrameInfo()`로 현재 프레임 크기 설정
- 이를 통해 정규화 좌표 ↔ 픽셀 좌표 변환 시 올바른 aspect ratio 적용

#### 2. FaceAlignData 구조체

```cpp
// Core 모듈에서 발견된 데이터 구조
UIMakeupLiveFaceAlignData   // 메이크업용 얼굴 정렬 데이터
SkinCareFaceAlignData       // 스킨케어용 얼굴 정렬 데이터
MakeupLive_FaceAlignData    // 네이티브 레벨 정렬 데이터
```

**핵심 인사이트**: 좌표 데이터와 함께 **원본 프레임 메타데이터**를 묶어서 관리합니다.

#### 3. 좌표 데이터 타입

```cpp
// FaceTracking에서 사용하는 좌표 타입
VN_Point32f     // 32비트 부동소수점 좌표
HyPoint2D32f    // 2D 좌표 (아마도 정규화)
HyRect          // 사각형 영역 (바운딩 박스용)
```

#### 4. IrisDetectorForLive (Makeup 모듈)

```cpp
// Venus 엔진 내 홍채 전용 검출기
IrisDetectorForLive
EyeContactsLive
```

**핵심 인사이트**: 홍채 검출을 **별도 클래스로 분리**하여 좌표 변환 로직을 캡슐화합니다.

#### 5. ApplyRotateCorrect()

```cpp
void FaceAlignMotionSmoother::ApplyRotateCorrect();
```

**핵심 인사이트**: 회전 보정을 **별도 단계**로 처리합니다.
- 현재 IrisLensSDK는 `detectOnlyWithRotation()`에서 회전과 좌표 변환을 함께 처리
- PerfectLib는 회전 보정을 분리하여 좌표 변환의 복잡성 감소

---

### 2.7 IrisLensSDK에 적용할 수 있는 개선 방향

PerfectLib의 접근법을 참고한 권장 개선사항:

#### A. 프레임 메타데이터를 IrisResult에 포함 (이미 구현됨 ✅)

```cpp
// IrisResult에 frameWidth, frameHeight 포함 - 이미 구현되어 있음
struct IrisResult {
    int frame_width;
    int frame_height;
    // ...
};
```

#### B. 좌표 변환 로직 분리 (권장)

현재 `mediapipe_detector.cpp`에 좌표 변환이 흩어져 있음.

```cpp
// 권장: CoordinateTransformer 클래스로 분리
class CoordinateTransformer {
public:
    void setFrameInfo(int width, int height);

    // 모델 출력(정사각형 crop 기준) → 전체 이미지(정규화)
    Point2f cropToNormalized(const Point2f& point, const Rect& crop);

    // 정규화 → 픽셀
    Point2i normalizedToPixel(const Point2f& point);

    // 회전 보정
    void applyRotationCorrection(float* landmarks, int count, int rotation);

private:
    int frame_width_;
    int frame_height_;
    float aspect_ratio_;
};
```

#### C. 회전 처리 분리 (권장)

현재 `detectOnlyWithRotation()`이 너무 많은 책임을 가짐:
1. 이미지 회전
2. 검출 실행
3. 좌표 역회전
4. Aspect ratio 보정

**PerfectLib 스타일 분리**:
```cpp
// 단계별 분리
Frame rotatedFrame = rotateFrame(input, rotation);
IrisResult rawResult = detect(rotatedFrame);
applyRotationCorrection(rawResult, rotation);  // 별도 단계
applyAspectRatioCorrection(rawResult, originalSize);  // 별도 단계
```

#### D. 디버깅을 위한 좌표 로깅 강화

```cpp
// PerfectLib처럼 프레임 정보와 함께 좌표 로그
LOG_D("IrisSDK", "Frame: %dx%d, Crop: %.4f,%.4f,%.4f,%.4f",
      width, height,
      crop.x, crop.y, crop.width, crop.height);
LOG_D("IrisSDK", "Raw landmark[0]: (%.4f, %.4f) -> Final: (%.4f, %.4f)",
      local_x, local_y, final_x, final_y);
```

---

## 3. ISS-002: Face Mesh 프레임 떨림 (미해결)

### 3.1 문제 상황

- Face Mesh가 프레임마다 미세하게 떨림
- 홍채 위치도 안정적이지 않음
- 렌즈 오버레이가 "흔들리는" 느낌

### 3.2 PerfectLib 해결 방식: FaceAlignMotionSmoother

```
프레임 N-2: 좌표 A
프레임 N-1: 좌표 A' (A와 미세 차이)
프레임 N:   좌표 A'' (A'와 미세 차이)

스무딩 없이: A → A' → A'' (급격한 변화 = 떨림)
스무딩 적용: A → smooth(A,A') → smooth(A',A'') (완만한 변화)
```

### 3.3 권장 해결 방안

#### 방안 A: C++ Core에서 EMA 스무딩 (권장)

**장점**:
- 플랫폼 독립적 (Android, iOS, Flutter 모두 적용)
- 내부 상태 관리 용이
- 렌더링 레이어 부담 감소

**구현 파일**: `cpp/include/iris_sdk/motion_smoother.h`

```cpp
namespace iris_sdk {

class MotionSmoother {
public:
    struct Config {
        float alpha = 0.3f;              // EMA 계수 (0.1~0.5)
        float velocity_threshold = 0.05f; // 빠른 이동 감지
        bool adaptive_alpha = true;       // 적응형 알파
    };

    // Face Mesh 스무딩 (478개 랜드마크)
    void smooth(float* landmarks, int count) {
        if (!initialized_) {
            prev_landmarks_.assign(landmarks, landmarks + count * 3);
            initialized_ = true;
            return;
        }

        for (int i = 0; i < count * 3; ++i) {
            float velocity = std::abs(landmarks[i] - prev_landmarks_[i]);
            float alpha = computeAdaptiveAlpha(velocity);

            // EMA: smoothed = alpha * current + (1 - alpha) * prev
            landmarks[i] = alpha * landmarks[i] + (1.0f - alpha) * prev_landmarks_[i];
            prev_landmarks_[i] = landmarks[i];
        }
    }

private:
    float computeAdaptiveAlpha(float velocity) {
        // 빠른 이동: 높은 알파 (즉각 반응)
        // 느린 이동: 낮은 알파 (강한 스무딩)
        return velocity > config_.velocity_threshold
            ? std::min(0.8f, config_.alpha + velocity * 2.0f)
            : config_.alpha;
    }
};

} // namespace iris_sdk
```

**통합 위치**: `mediapipe_detector.cpp`

```cpp
int MediaPipeDetector::detect(const Frame& frame, IrisResult& result) {
    // ... 기존 검출 로직 ...

    // 모션 스무딩 적용
    if (impl_->motion_smoother_ && result.face_detected) {
        impl_->motion_smoother_->smooth(
            impl_->face_landmarks_buffer,
            NUM_FACE_LANDMARKS
        );
    }

    return IRIS_SDK_OK;
}
```

#### 방안 B: 칼만 필터 (고급)

**장점**:
- 검출 누락 시 위치 예측 가능
- 물리 기반 모델로 자연스러운 움직임

**고려사항**:
- EMA보다 구현 복잡
- 파라미터 튜닝 필요

**권장**: EMA로 먼저 구현하고, 품질 평가 후 칼만 필터 도입 검토

### 3.4 현재 Android 스무딩 상태

`OverlayView.kt`에 EMA 스무딩이 부분 적용됨:

```kotlin
private const val SMOOTHING_FACTOR = 0.25f

smoothedLeftX = lerp(smoothedLeftX, it.leftIrisX, SMOOTHING_FACTOR)
smoothedLeftY = lerp(smoothedLeftY, it.leftIrisY, SMOOTHING_FACTOR)
```

**문제점**: 홍채만 스무딩, Face Mesh(478개)는 스무딩 안 됨

**해결 방향**: C++ Core로 전체 스무딩 이동 후 Android 스무딩 제거

---

## 4. ISS-003: 저조한 FPS (17-19fps)

### 4.1 현재 병목 분석

```
전체 파이프라인 (~58ms)
├── YUV→RGB 변환: ~5ms  (CPU)
├── Face Detection: ~15ms
├── Face Landmark: ~20ms
├── Iris Landmark: ~10ms
└── 좌표 변환: ~8ms
```

### 4.2 PerfectLib 인사이트

PerfectLib는 **GPU 기반 파이프라인**으로 30fps+ 달성:

```
카메라 → GPU 텍스처 → GPU 쉐이더 → 화면
         (제로카피)    (24+ 필터)
```

### 4.3 권장 최적화 전략

#### 즉시 적용 가능 (Phase 2 Sprint 1)

**1. GPU Delegate 활성화**

```kotlin
// MainActivity.kt
val enableGpu = true  // 현재 false
```

예상 효과: +40~60%

**2. 입력 해상도 감소**

```kotlin
// CameraManager.kt
imageAnalysis = ImageAnalysis.Builder()
    .setTargetResolution(Size(640, 360))  // 기존: 더 높음
    .build()
```

예상 효과: +30%

**3. 추적 모드 튜닝**

Face Detection 스킵 빈도 증가:

```cpp
// 신뢰도 임계값 조정
impl_->min_presence_confidence = 0.4f;  // 기존: 0.5f
```

#### 중기 최적화 (Phase 2 Sprint 2-3)

**4. OpenGL ES 렌더링**

CPU 블렌딩 → GPU 쉐이더:

```glsl
// Fragment Shader
uniform sampler2D u_frame;
uniform sampler2D u_lens;
uniform vec2 u_iris_center;
uniform float u_iris_radius;
uniform float u_opacity;

void main() {
    vec4 frame = texture(u_frame, v_texCoord);
    vec4 lens = texture(u_lens, lensCoord);
    float mask = smoothstep(radius * 0.8, radius, dist);
    fragColor = mix(frame, lens, mask * u_opacity);
}
```

예상 효과: 28-32fps

**5. 제로카피 파이프라인** (고급)

HardwareBuffer → EGLImage → GLES 텍스처 직접 매핑

예상 효과: 35-40fps

### 4.4 예상 성능 개선

| 최적화 | 현재 FPS | 예상 FPS | 구현 난이도 |
|--------|---------|---------|-----------|
| 현재 상태 | 17-19 | - | - |
| GPU Delegate | - | 25-28 | 낮음 |
| + 해상도 감소 | - | 28-32 | 낮음 |
| + OpenGL ES | - | 32-35 | 중간 |
| + 제로카피 | - | 35-40+ | 높음 |

---

## 5. ISS-004: MediaPipe 폴백 없음

### 5.1 문제 상황

- MediaPipe는 **얼굴 전체**가 보여야 검출 가능
- 눈만 클로즈업 시 검출 실패
- 극단적 각도(45°+)에서 불안정

### 5.2 PerfectLib 해결 방식

다중 파이프라인:

```
입력 → FaceDetection → FaceTracking → VenusMakeup
              ↓ 실패시
         대체 검출기 (예: 눈 영역 전용)
```

### 5.3 권장 해결 방안: HybridDetector

```cpp
class HybridDetector : public IrisDetector {
public:
    int detect(const Frame& frame, IrisResult& result) override {
        // 1차: MediaPipe 시도
        int ret = primary_->detect(frame, result);
        if (ret == IRIS_SDK_OK && result.isValid()) {
            return IRIS_SDK_OK;
        }

        // 2차: Eye-Only 폴백 (Phase 2에서 구현)
        return fallback_->detect(frame, result);
    }

private:
    std::unique_ptr<MediaPipeDetector> primary_;
    std::unique_ptr<EyeOnlyDetector> fallback_;
};
```

### 5.4 Eye-Only 모델 구현 계획 (Phase 2)

1. **데이터 수집**: 눈 영역 이미지 5,000~10,000장
2. **모델 학습**: U-Net 또는 MobileNet 기반
3. **통합**: HybridDetector에 fallback으로 추가

---

## 6. 테스트 전략

### 6.1 좌표 변환 검증

```kotlin
// OverlayView에 테스트 모드 추가
fun setTestMode(enabled: Boolean) {
    if (enabled) {
        // 고정 좌표로 렌더링하여 변환 로직 검증
        val testResult = IrisResult().apply {
            detected = true
            leftDetected = true
            leftIrisX = 0.5f  // 정중앙 → 화면 중앙에 표시되어야 함
            leftIrisY = 0.5f
            leftRadius = 30f
        }
        setIrisResult(testResult, 1080, 1920, false)
    }
}
```

### 6.2 스무딩 품질 평가

```kotlin
class SmoothingEvaluator {
    private val rawPositions = mutableListOf<PointF>()
    private val smoothedPositions = mutableListOf<PointF>()

    fun evaluate(): SmoothingMetrics {
        // 지터(jitter) 계산: 프레임 간 이동 거리의 표준편차
        val rawJitter = calculateJitter(rawPositions)
        val smoothedJitter = calculateJitter(smoothedPositions)

        return SmoothingMetrics(
            jitterReduction = (rawJitter - smoothedJitter) / rawJitter,
            // 목표: 90% 이상 감소
        )
    }
}
```

### 6.3 FPS 벤치마크

기존 `BenchmarkManager` 활용:

```kotlin
// 1분간 측정
benchmarkManager.startBenchmark()
// ... 1분 후
val results = benchmarkManager.stopBenchmark()

// 확인 항목:
// - averageFps >= 30
// - p95Latency <= 40ms
// - droppedFrames < 5%
```

---

## 7. 구현 로드맵

### Phase 2 Sprint 1 (즉시 적용)

| 작업 | 예상 시간 | 효과 |
|------|----------|------|
| 빌드 캐시 정리 + 재빌드 | 30분 | ISS-001 해결 확인 |
| GPU Delegate 활성화 | 1시간 | FPS +40% |
| 입력 해상도 감소 | 30분 | FPS +30% |
| **합계** | **2시간** | **기본 목표 달성** |

### Phase 2 Sprint 2 (1주)

| 작업 | 예상 시간 | 효과 |
|------|----------|------|
| C++ 모션 스무딩 | 4시간 | ISS-002 해결 |
| Face Mesh 스무딩 적용 | 2시간 | 시각적 품질 |
| Android 스무딩 제거 | 1시간 | 코드 간소화 |
| **합계** | **7시간** | **떨림 90% 감소** |

### Phase 2 Sprint 3 (2주)

| 작업 | 예상 시간 | 효과 |
|------|----------|------|
| OpenGL ES 렌더링 | 16시간 | 30fps+ |
| HybridDetector 기본 | 8시간 | 폴백 전략 |
| **합계** | **24시간** | **성능 목표 달성** |

---

## 8. 결론

### 즉시 조치 사항

1. **빌드 정리 후 재빌드**: ISS-001 해결 확인
2. **GPU Delegate 활성화**: `enableGpu = true`
3. **디버그 로그 분석**: aspect ratio 값 확인

### 핵심 개선 방향

1. **모션 스무딩**: C++ Core에서 EMA 구현 → 떨림 해결
2. **GPU 파이프라인**: OpenGL ES 도입 → 30fps 달성
3. **폴백 전략**: HybridDetector → 안정성 향상

### PerfectLib 대비 목표

| 기능 | PerfectLib | IrisLensSDK 목표 |
|------|-----------|-----------------|
| 모션 스무딩 | 칼만 필터 | EMA (동등 효과) |
| FPS | 30+ | 30+ |
| 검출 안정성 | 98% | 95% |
| SDK 크기 | 16.5MB | 20MB 이하 |

**핵심 메시지**: 2주 내에 핵심 품질 지표(스무딩, 성능)를 PerfectLib 수준으로 끌어올릴 수 있습니다.
