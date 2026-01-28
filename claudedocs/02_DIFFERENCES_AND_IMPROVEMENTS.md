# PerfectLib vs IrisLensSDK 차이점 및 개선방안

**작성일**: 2026-01-21
**분석 에이전트**: system-architect, c4-architecture, architect-review

---

## 1. 핵심 기술 격차 분석

### 1.1 격차 요약표

| 영역 | PerfectLib | IrisLensSDK | 격차 수준 | 영향도 |
|------|-----------|-------------|----------|--------|
| **모션 스무딩** | FaceAlignMotionSmoother | 없음 (Android EMA만) | 🔴 심각 | 사용자 경험 |
| **GPU 렌더링** | 24+ OpenGL ES 필터 | CPU OpenCV | 🔴 심각 | 성능 |
| **얼굴 추적** | 106+ 전용 랜드마크 | MediaPipe 478 | 🟡 보통 | 품질 |
| **멀티스레딩** | Camera/Track/Render 분리 | InferenceThread만 | 🟡 보통 | 성능 |
| **메모리 관리** | LRU 캐싱 + 메모리 풀 | 기본 스마트 포인터 | 🟡 보통 | 안정성 |
| **추론 엔진** | MNN (1.8MB) | TFLite | 🟢 양호 | 크기 |
| **폴백 전략** | 다중 파이프라인 | 단일 MediaPipe | 🟡 보통 | 안정성 |

### 1.2 가장 심각한 격차: 모션 스무딩

**PerfectLib FaceAlignMotionSmoother의 역할**:
```
프레임 N-2: ●●●●○●●●●●
프레임 N-1: ●●●●●○●●●●
프레임 N:   ●●●●●●○●●●  <- 급격한 변화 억제
```

**IrisLensSDK 현재 상태**:
```
프레임 N-2: ●●●○●●●●●●
프레임 N-1: ●●●●●○●●●●  <- 떨림!
프레임 N:   ●●○●●●●●●●  <- 불규칙!
```

**결과**: 렌즈 오버레이가 불안정하게 흔들리는 사용자 경험 저하

### 1.3 두 번째 격차: GPU 파이프라인

**PerfectLib (제로카피)**:
```
카메라 → GPU 텍스처 → GPU 쉐이더 → 화면
         (직접 매핑)    (24+ 필터)
```

**IrisLensSDK (CPU 병목)**:
```
카메라 → YUV변환(CPU) → RGB(CPU) → OpenCV(CPU) → 화면
          ↑              ↑           ↑
       병목 1          병목 2       병목 3
```

**결과**: 17-19fps (목표 30fps 미달)

---

## 2. 아키텍처 차이점

### 2.1 레이어 구조 비교

**PerfectLib (기능별 레이어드)**:
```
┌─────────────────────────────────────────┐
│ ProductHandler (비즈니스 로직)           │
├─────────────────────────────────────────┤
│ HandlerCore (데이터 관리)                │
├─────────────────────────────────────────┤
│ Makeup (Venus 렌더링 엔진)               │
├─────────────────────────────────────────┤
│ Core (SDK 인프라, MNN)                   │
├─────────────────────────────────────────┤
│ FaceTracking (얼굴 추적)                 │
└─────────────────────────────────────────┘
```

**IrisLensSDK (플랫폼별 레이어드)**:
```
┌─────────────────────────────────────────┐
│ Application Layer                        │
├──────┬──────┬──────┬─────────────────────┤
│ JNI  │Obj-C │ FFI  │ WASM                │
├──────┴──────┴──────┴─────────────────────┤
│ C API Layer                              │
├─────────────────────────────────────────┤
│ C++ Core Engine                          │
│ (SDKManager, IrisDetector, LensRenderer) │
├─────────────────────────────────────────┤
│ Third Party (MediaPipe, OpenCV, TFLite)  │
└─────────────────────────────────────────┘
```

**핵심 차이**:
- PerfectLib: 기능 중심 분리 → Android 최적화
- IrisLensSDK: 플랫폼 중심 분리 → 크로스플랫폼 우수

### 2.2 설계 패턴 비교

| 패턴 | PerfectLib | IrisLensSDK |
|------|-----------|-------------|
| Strategy | - | ✅ IrisDetector |
| Singleton | ✅ 추정 | ✅ SDKManager |
| pImpl | ❓ | ✅ ABI 안정성 |
| Factory | ✅ VtoApplier | ✅ create() |
| Handler | ✅ 8종 Handler | - |
| DAO | ✅ SQLite 추상화 | - |
| Pipeline | ✅ GPU 파이프라인 | 🔄 기본 수준 |

### 2.3 확장성 비교

| 확장 시나리오 | PerfectLib | IrisLensSDK |
|--------------|-----------|-------------|
| 새 검출기 추가 | 중간 (내부 구조 복잡) | 우수 (Strategy 패턴) |
| 새 렌더링 효과 | 우수 (Venus 엔진) | 보통 (직접 구현 필요) |
| 새 플랫폼 | 어려움 (Android 특화) | 매우 우수 (C API) |
| 새 비즈니스 기능 | 매우 우수 (Handler) | 어려움 (레이어 없음) |

---

## 3. 개선 방안

### 3.1 우선순위별 개선 로드맵

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        개선 로드맵                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Phase 2 - Sprint 1 (2주)     Sprint 2 (2주)      Sprint 3 (3주)         │
│  ┌─────────────────┐        ┌─────────────────┐  ┌─────────────────┐    │
│  │ P0: 모션 스무딩  │───────>│ P1: GPU 렌더링  │─>│ P1: Hybrid     │    │
│  │ (EMA+적응형)    │        │ (OpenGL ES)     │  │     Detector   │    │
│  └─────────────────┘        └─────────────────┘  └─────────────────┘    │
│          │                         │                     │               │
│  ┌─────────────────┐        ┌─────────────────┐  ┌─────────────────┐    │
│  │ P0: 메모리 풀   │───────>│ P2: 적응형 FPS │─>│ P2: 칼만 필터  │    │
│  │ (FramePool)    │        │                 │  │                 │    │
│  └─────────────────┘        └─────────────────┘  └─────────────────┘    │
│                                                                          │
│  목표: 30fps 안정, 떨림 제거, 검출 안정성                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 P0: 모션 스무딩 구현

**구현 위치**: `cpp/src/motion_smoother.cpp` (신규)

```cpp
namespace iris_sdk {

class MotionSmoother {
public:
    struct Config {
        float alpha = 0.3f;              // EMA 계수
        float velocity_threshold = 0.05f; // 빠른 이동 감지
        bool adaptive_alpha = true;       // 적응형 알파
    };

    void smooth(float* landmarks, int count);
    void smoothIris(float* iris_landmarks);
    void reset();

private:
    // 적응형 EMA: 느린 이동=강한 스무딩, 빠른 이동=약한 스무딩
    float computeAdaptiveAlpha(float velocity) {
        if (velocity > config_.velocity_threshold) {
            return std::min(0.8f, config_.alpha + velocity * 2.0f);
        }
        return config_.alpha;
    }
};

} // namespace iris_sdk
```

**통합 위치**: `mediapipe_detector.cpp`의 `detect()` 메서드 끝

```cpp
if (impl_->motion_smoother_ && result.face_detected) {
    impl_->motion_smoother_->update(result);
}
```

**예상 효과**: 떨림 90% 감소

### 3.3 P1: GPU 렌더링 파이프라인

**Fragment Shader (렌즈 블렌딩)**:

```glsl
#version 300 es
precision highp float;

uniform sampler2D u_frame;      // 카메라 프레임
uniform sampler2D u_lens;       // 렌즈 텍스처
uniform vec2 u_iris_center_l;   // 왼쪽 홍채 중심
uniform vec2 u_iris_center_r;   // 오른쪽 홍채 중심
uniform float u_iris_radius_l;  // 왼쪽 홍채 반경
uniform float u_iris_radius_r;  // 오른쪽 홍채 반경
uniform float u_opacity;        // 불투명도

float calculateMask(vec2 coord, vec2 center, float radius) {
    float dist = distance(coord, center);
    return 1.0 - smoothstep(radius * 0.8, radius, dist);
}

void main() {
    vec4 frameColor = texture(u_frame, v_texCoord);

    // 왼쪽/오른쪽 눈 처리
    float maskL = calculateMask(v_texCoord, u_iris_center_l, u_iris_radius_l);
    float maskR = calculateMask(v_texCoord, u_iris_center_r, u_iris_radius_r);

    // 블렌딩
    // ...
}
```

**예상 성능 개선**:

| 최적화 단계 | 현재 FPS | 예상 FPS |
|------------|---------|---------|
| 현재 상태 | 17-19 | - |
| SIMD + 해상도 적응 | - | 22-25 |
| OpenGL ES | - | 28-32 |
| 제로카피 | - | 35-40+ |

### 3.4 P1: HybridDetector 폴백 전략

```cpp
class HybridDetector : public IrisDetector {
public:
    int detect(const Frame& frame, IrisResult& result) override {
        // 1차: MediaPipe 시도
        int ret = primary_->detect(frame, result);
        if (ret == IRIS_SDK_OK && result.isValid()) {
            primary_success_++;
            return IRIS_SDK_OK;
        }

        // 2차: Eye-Only 폴백
        fallback_success_++;
        return fallback_->detect(frame, result);
    }

    float getFallbackRate() const {
        return total_frames_ > 0
            ? float(fallback_success_) / total_frames_
            : 0.0f;
    }

private:
    std::unique_ptr<MediaPipeDetector> primary_;
    std::unique_ptr<EyeOnlyDetector> fallback_;
};
```

### 3.5 아키텍처 개선안

**현재 → 개선 후 비교**:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   현재 아키텍처                                          │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  카메라 ───> detect() ───> render() ───> 화면                           │
│              (동기)        (동기, CPU)                                   │
│                                                                         │
│  문제점:                                                                 │
│  1. 동기식 → 병렬화 불가                                                 │
│  2. 단일 검출기 → 폴백 없음                                              │
│  3. CPU 렌더링 → 성능 병목                                               │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
                                ↓
┌────────────────────────────────────────────────────────────────────────┐
│                   개선 아키텍처                                          │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────┐    ┌───────────┐    ┌──────────────┐                    │
│  │  Camera   │───>│ FramePool │───>│  Detection   │                    │
│  │  Thread   │    │ (Triple)  │    │   Thread     │                    │
│  └───────────┘    └───────────┘    └───────┬──────┘                    │
│                                            │                            │
│                      ┌─────────────────────┴─────────┐                  │
│                      │       HybridDetector          │                  │
│                      │   ┌────────┐  ┌────────────┐  │                  │
│                      │   │MediaPipe│  │ Eye-Only  │  │                  │
│                      │   │ Primary │  │ Fallback  │  │                  │
│                      │   └────────┘  └────────────┘  │                  │
│                      └───────────────────────────────┘                  │
│                                            │                            │
│                              ┌─────────────▼─────────────┐              │
│                              │     Motion Smoother       │              │
│                              │   (EMA + Kalman Filter)   │              │
│                              └─────────────┬─────────────┘              │
│                                            │                            │
│  ┌───────────┐    ┌───────────┐    ┌──────▼──────────────┐              │
│  │  Display  │<───│  Render   │<───│    GPU Pipeline     │              │
│  │           │    │  Thread   │    │  (OpenGL ES/Metal)  │              │
│  └───────────┘    └───────────┘    └─────────────────────┘              │
│                                                                         │
│  개선점:                                                                 │
│  1. Triple-Buffering → Camera/Detection/Render 병렬화                   │
│  2. HybridDetector → 폴백 전략                                          │
│  3. MotionSmoother → 떨림 제거                                          │
│  4. GPU Pipeline → 30fps+ 달성                                          │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. API 개선 권고

### 4.1 비동기 API 추가 (권장)

```cpp
// 현재: 동기식만
IrisSdkError iris_sdk_detect(...);

// 추가: 비동기 콜백
typedef void (*IrisDetectCallback)(
    IrisSdkError error,
    const IrisResult* result,
    void* user_data
);

IrisSdkError iris_sdk_detect_async(
    const uint8_t* frame_data,
    int width, int height, int format,
    IrisDetectCallback callback,
    void* user_data
);
```

### 4.2 성능 메트릭 API 확장

```cpp
typedef struct {
    float detection_time_ms;
    float render_time_ms;
    float total_time_ms;
    float average_fps;
    int frames_processed;
    int frames_dropped;
} IrisSdkMetrics;

IrisSdkError iris_sdk_get_metrics(IrisSdkMetrics* metrics);
```

### 4.3 에러 컨텍스트 강화

```cpp
typedef struct {
    IrisSdkError code;
    const char* message;
    const char* file;    // 디버그 빌드용
    int line;
} IrisSdkErrorContext;

IrisSdkErrorContext iris_sdk_get_error_context(void);
```

---

## 5. 구현 일정 및 효과

### 5.1 Phase 2 일정

| 작업 | 예상 기간 | 담당 | 효과 |
|------|----------|------|------|
| 모션 스무딩 (EMA) | 3일 | C++ | 떨림 90% 감소 |
| 메모리 풀 | 5일 | C++ | GC 스파이크 제거 |
| OpenGL ES 렌더링 | 2주 | Android | 30fps+ 달성 |
| HybridDetector | 1주 | C++ | 검출 안정성 향상 |
| 적응형 FPS | 3일 | C++ | 부하 분산 |
| 칼만 필터 | 5일 | C++ | 예측 기반 안정화 |

### 5.2 기대 효과

| 지표 | 현재 | Phase 2 목표 | PerfectLib 수준 |
|------|------|-------------|----------------|
| FPS | 17-19 | 30+ | 30+ |
| 떨림 | 심함 | 미미 | 없음 |
| 검출 안정성 | 80% | 95% | 98% |
| 메모리 | 변동 | 안정 | 안정 |

---

## 6. 결론

### PerfectLib에서 배울 점

1. **모션 스무딩은 필수**: 사용자 경험의 핵심
2. **GPU 파이프라인**: 30fps 달성의 열쇠
3. **폴백 전략**: 단일 모델 의존 위험
4. **멀티스레딩**: 파이프라인 병렬화

### IrisLensSDK의 차별화 포인트

1. **클린 아키텍처**: 확장성과 테스트 용이성
2. **크로스플랫폼**: C API 기반 통일
3. **경량화**: 목표 20MB 이하
4. **오픈소스 친화**: 라이선스 비용 우위

### 최종 권고

> **전략**: PerfectLib의 모든 기능을 모방하지 말고,
> **핵심 품질(스무딩, 성능)**을 확보한 후
> **경량화와 크로스플랫폼**으로 차별화할 것.
