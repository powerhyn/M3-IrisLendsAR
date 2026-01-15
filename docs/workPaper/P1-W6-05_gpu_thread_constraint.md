# P1-W6-05: GPU Delegate 스레드 제약 이슈

## 상태: ✅ 완료 (InferenceThread로 해결)

**작성일**: 2026-01-14
**작성자**: Claude Code
**관련 작업**: GPU 가속 런타임 활성화

---

## 요약

TFLite GPU delegate가 빌드 및 초기화는 성공하지만, 런타임에서 스레드 제약으로 인해 실패함.
현재 CPU 폴백으로 동작 중이며, GPU 가속을 위해서는 추가 아키텍처 수정 필요.

---

## 구현 완료 항목

### 1. GPU API 스택 추가 ✅

| 레이어 | 파일 | 추가 내용 |
|--------|------|-----------|
| C API | `cpp/include/iris_sdk/sdk_api.h` | `iris_sdk_set_gpu_enabled()`, `iris_sdk_is_gpu_available()`, `iris_sdk_is_using_gpu()` |
| C++ 구현 | `cpp/src/sdk_api.cpp` | GPU 상태 관리 및 FrameProcessor 연동 |
| FrameProcessor | `cpp/src/frame_processor.cpp` | GPU 설정 전달 로직 (초기화 전 호출) |
| JNI | `android/iris-sdk/src/main/cpp/iris_jni.cpp` | `nativeSetGpuEnabled()`, `nativeIsGpuAvailable()`, `nativeIsUsingGpu()` |
| Java SDK | `android/iris-sdk/src/main/java/com/irislenssdk/IrisLensSDK.java` | `setGpuEnabled()`, `isGpuAvailable()`, `isUsingGpu()` |

### 2. GPU Delegate 초기화 성공 ✅

```
01-14 11:58:11.807 I tflite: Created 1 GPU delegate kernels.  ← Face Detection
01-14 11:58:11.843 I tflite: Created 1 GPU delegate kernels.  ← Face Landmark
01-14 11:58:11.909 I tflite: Created 1 GPU delegate kernels.  ← Iris Landmark
01-14 11:58:11.911 I IrisLensSDK-Demo: GPU Active: true
```

---

## 발생한 문제

### TFLite GPU Delegate 스레드 제약

**에러 메시지:**
```
E tflite: TfLiteGpuDelegate Invoke: GpuDelegate must run on the same thread where it was initialized.
E tflite: Node number 164 (TfLiteGpuDelegateV2) failed to invoke.
```

**원인 분석:**
- TFLite GPU delegate (OpenGL ES 기반)는 **초기화된 스레드에서만 inference 실행 가능**
- 현재 Demo App 구조:
  - SDK 초기화: 메인 UI 스레드 (Thread 19061)
  - 프레임 검출: CameraX ImageAnalysis 스레드 (Thread 19205)
- 스레드 불일치로 GPU delegate `Invoke()` 실패

**영향:**
- GPU 초기화는 성공하지만 실제 inference 실행 불가
- 모든 프레임에서 검출 실패 (`detected=0`)

---

## 현재 해결책 (CPU 폴백)

**Demo App 수정** (`MainActivity.kt`):
```kotlin
// GPU 가속 활성화 요청 (init 전에 호출!)
// TODO: GPU delegate는 초기화 스레드에서만 실행 가능
//       CameraX 콜백 스레드와 다르므로 현재 비활성화
//       정식 수정: SDK 초기화를 inference 스레드에서 수행 필요
val gpuAvailable = IrisLensSDK.isGpuAvailable()
Log.i(TAG, "GPU available: $gpuAvailable (disabled due to thread constraint)")
```

**결과:**
- CPU 모드로 정상 동작 (`detected=1, confidence=0.89`)
- GPU 가속 미적용 상태

---

## 향후 해결 방안

### 방안 1: 전용 Inference 스레드 (권장)

SDK 내부에 전용 inference 스레드를 생성하여 모든 TFLite 연산을 해당 스레드에서 수행.

```
┌─────────────────────────────────────────────────────────┐
│ Demo App                                                │
│  ┌──────────────┐    ┌──────────────┐                  │
│  │ Main Thread  │    │ CameraX      │                  │
│  │ (UI)         │    │ Thread       │                  │
│  └──────────────┘    └──────┬───────┘                  │
│                             │ Frame                     │
│                             ▼                           │
│  ┌──────────────────────────────────────────────────┐  │
│  │ IrisLensSDK                                      │  │
│  │  ┌─────────────────────────────────────────────┐ │  │
│  │  │ Inference Thread (GPU delegate 소유)        │ │  │
│  │  │  - SDK 초기화                               │ │  │
│  │  │  - TFLite GPU delegate 생성                │ │  │
│  │  │  - 모든 detect() 호출 처리                 │ │  │
│  │  └─────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**장점:**
- SDK 사용자가 스레드 관리할 필요 없음
- 기존 API 유지 가능

**단점:**
- 프레임 큐잉으로 인한 지연 가능성
- 구현 복잡도 증가

### 방안 2: NNAPI Delegate 사용

TFLite NNAPI delegate는 스레드 제약이 없음.

```cpp
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"

// NNAPI delegate 생성
tflite::StatefulNnApiDelegate::Options options;
options.execution_preference =
    tflite::StatefulNnApiDelegate::Options::kSustainedSpeed;
auto delegate = new tflite::StatefulNnApiDelegate(options);
interpreter->ModifyGraphWithDelegate(delegate);
```

**장점:**
- 스레드 제약 없음
- Android 8.1+ 기본 지원
- 하드웨어 가속 (NPU, DSP, GPU 자동 선택)

**단점:**
- 디바이스/모델 호환성 이슈 가능
- 성능이 GPU delegate보다 낮을 수 있음

### 방안 3: 앱 레벨 스레드 관리

Demo App에서 SDK 초기화와 검출을 동일 스레드에서 수행.

```kotlin
// 전용 스레드 생성
private val inferenceThread = HandlerThread("InferenceThread").apply { start() }
private val inferenceHandler = Handler(inferenceThread.looper)

// SDK 초기화를 inference 스레드에서 수행
inferenceHandler.post {
    IrisLensSDK.setGpuEnabled(true)
    IrisLensSDK.init(context)
}

// CameraX 프레임도 inference 스레드로 전달
imageAnalysis.setAnalyzer(inferenceExecutor) { image ->
    inferenceHandler.post {
        IrisLensSDK.detectWithRotation(...)
    }
}
```

**장점:**
- SDK 수정 불필요
- 빠른 적용 가능

**단점:**
- 앱 개발자가 스레드 관리 필요
- 프레임 드롭 가능성

---

## 권장 구현 순서

1. **단기 (현재)**: CPU 모드로 기능 검증 완료
2. **중기**: NNAPI delegate 테스트 및 성능 비교
3. **장기**: SDK 내부 inference 스레드 구현 (방안 1)

---

## 관련 참고 자료

- [TFLite GPU Delegate Guide](https://www.tensorflow.org/lite/performance/gpu)
- [TFLite NNAPI Delegate](https://www.tensorflow.org/lite/android/delegates/nnapi)
- [OpenGL ES Thread Safety](https://www.khronos.org/opengl/wiki/OpenGL_and_multithreading)

---

---

## 해결 구현: InferenceThread (방안 1 적용)

**구현일**: 2026-01-14

### 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│ 앱 (CameraX 스레드)                                              │
│   │                                                             │
│   │ detectWithRotation(frame)                                   │
│   ▼                                                             │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ IrisLensSDK (JNI Layer)                                     │ │
│ │   │                                                         │ │
│ │   │ detectSync() - 동기 호출                                 │ │
│ │   ▼                                                         │ │
│ │ ┌─────────────────────────────────────────────────────────┐ │ │
│ │ │ InferenceThread (전용 스레드)                           │ │ │
│ │ │   ├─ TFLite 인터프리터 소유                             │ │ │
│ │ │   ├─ GPU Delegate 소유 (동일 스레드에서 Init + Invoke)  │ │ │
│ │ │   ├─ 프레임 큐에서 입력 수신                            │ │ │
│ │ │   └─ 결과 큐로 출력 전송                                │ │ │
│ │ └─────────────────────────────────────────────────────────┘ │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 구현 파일

| 파일 | 변경 |
|------|------|
| `cpp/include/iris_sdk/inference_thread.h` | **신규** - InferenceThread 클래스 선언 |
| `cpp/src/inference_thread.cpp` | **신규** - InferenceThread 구현 |
| `cpp/CMakeLists.txt` | inference_thread.cpp 빌드 추가 |
| `cpp/src/frame_processor.cpp` | InferenceThread 사용으로 변경 |
| `android/demo-app/.../MainActivity.kt` | GPU 활성화 코드 복원 |

### 검증 로그

```
01-14 20:16:29.981 I IrisLensSDK-Demo: GPU available: true
01-14 20:16:29.981 I IrisLensSDK-Demo: GPU acceleration requested
01-14 20:16:30.026 I tflite: Created TensorFlow Lite delegate for GPU.
01-14 20:16:30.042 I tflite: Initialized OpenGL-based API.
01-14 20:16:30.062 I tflite: Created 1 GPU delegate kernels.  ← Face Detection
01-14 20:16:30.135 I tflite: Created 1 GPU delegate kernels.  ← Face Landmark
01-14 20:16:30.181 I tflite: Created 1 GPU delegate kernels.  ← Iris Landmark
01-14 20:16:30.182 I IrisLensSDK-Demo: GPU Active: true  ← 성공!
```

**스레드 에러 없음** - 이전 `GpuDelegate must run on the same thread` 에러 해결됨

---

## 변경 이력

| 날짜 | 내용 | 작성자 |
|------|------|--------|
| 2026-01-14 | 초기 작성 - GPU 스레드 이슈 문서화 | Claude Code |
| 2026-01-14 | InferenceThread 구현으로 이슈 해결 | Claude Code |
