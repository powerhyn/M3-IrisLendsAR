# P1-W5-06: TFLite Android 통합 및 Face Mesh 시각화

**태스크 ID**: P1-W5-06
**상태**: ✅ 완료
**시작일**: 2025-01-12
**완료일**: 2025-01-13

---

## 1. 개요

### 배경
- Android에서 TFLite C++ API를 사용하여 MediaPipe Face Mesh 모델 실행 필요
- Google Maven AAR은 C API만 제공 → C++ API (`tflite::Interpreter`) 사용 불가
- 해결책: Bazel로 TFLite C++ 라이브러리 직접 빌드

### 목표
1. Bazel로 `libtensorflowlite.so` (C++ API 포함) 빌드
2. Android CMake에서 Pre-built 라이브러리 연동
3. Face Mesh 478개 랜드마크 데이터를 Android 앱에서 시각화

### 결과
- ✅ TFLite C++ 빌드 성공 (arm64-v8a)
- ✅ 홍채 검출 동작 확인 (confidence=0.94)
- ✅ Face Mesh 시각화 구현 완료

---

## 2. TFLite Bazel 빌드

### 2.1 환경 설정

> **중요**: TensorFlow v2.14.1은 NDK 19-21만 지원하여 최신 NDK와 호환되지 않습니다.
> TensorFlow v2.16.1과 NDK 25.1을 사용해야 합니다.

```bash
# 환경변수 (NDK 25.1 필수!)
export ANDROID_NDK_HOME=~/Library/Android/sdk/ndk/25.1.8937393
export ANDROID_HOME=~/Library/Android/sdk

# TensorFlow v2.16.1 클론 (v2.14.1 아님!)
cd /tmp
git clone --depth 1 --branch v2.16.1 https://github.com/tensorflow/tensorflow.git tensorflow-v2.16.1
cd tensorflow-v2.16.1
```

### 2.2 TensorFlow 설정

```bash
# configure 스크립트 실행
./configure

# 설정 값:
# - Python path: 기본값 Enter
# - Android SDK: /Users/[username]/Library/Android/sdk
# - Android NDK: /Users/[username]/Library/Android/sdk/ndk/25.1.8937393
# - Android API level: 24 (기본값)
```

### 2.3 CoreFoundation Workaround (macOS 필수)

> **문제**: macOS에서 Android 크로스 컴파일 시 Abseil의 cctz가 CoreFoundation 프레임워크를 링크하려고 시도하여 빌드 실패
>
> **에러 메시지**:
> ```
> ld.lld: error: unknown argument '-framework'
> ld.lld: error: cannot open CoreFoundation: No such file or directory
> ```

**해결 방법**: Abseil BUILD.bazel에서 CoreFoundation 링크 옵션 제거

```bash
# Bazel 캐시에서 Abseil BUILD.bazel 파일 찾기
ABSEIL_BUILD=$(find /private/var/tmp/_bazel_$(whoami) -name "BUILD.bazel" -path "*com_google_absl*cctz*" 2>/dev/null | head -1)

# CoreFoundation 링크 옵션 제거
sed -i '' 's/\["-Wl,-framework,CoreFoundation"\]/[]/g' "$ABSEIL_BUILD"

# 또는 직접 편집
# @platforms//os:osx: ["-Wl,-framework,CoreFoundation"], → @platforms//os:osx: [],
# @platforms//os:ios: ["-Wl,-framework,CoreFoundation"], → @platforms//os:ios: [],
```

**수정 전**:
```bazel
linkopts = select({
    "@platforms//os:osx": ["-Wl,-framework,CoreFoundation"],
    "@platforms//os:ios": ["-Wl,-framework,CoreFoundation"],
    "//conditions:default": [],
}),
```

**수정 후**:
```bazel
linkopts = select({
    "@platforms//os:osx": [],
    "@platforms//os:ios": [],
    "//conditions:default": [],
}),
```

### 2.4 CPU 라이브러리 빌드

```bash
# bazelisk 사용 권장 (bazel 버전 자동 관리)
cd /tmp/tensorflow-v2.16.1

bazelisk build -c opt \
  --config=android_arm64 \
  //tensorflow/lite:libtensorflowlite.so

# 출력 위치
# bazel-bin/tensorflow/lite/libtensorflowlite.so (~3.5MB)
```

### 2.5 GPU Delegate 빌드

```bash
# GPU delegate 빌드 (OpenGL ES 3.1 기반)
bazelisk build -c opt \
  --config=android_arm64 \
  //tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so

# 출력 위치
# bazel-bin/tensorflow/lite/delegates/gpu/libtensorflowlite_gpu_delegate.so (~16MB)
```

### 2.6 아티팩트 배치

```
cpp/third_party/tflite/
├── android/
│   └── arm64-v8a/
│       ├── libtensorflowlite.so              # CPU 라이브러리 (~3.5MB)
│       └── libtensorflowlite_gpu_delegate.so # GPU delegate (~16MB)
├── include/
│   └── tensorflow/
│       └── lite/
│           ├── interpreter.h
│           ├── model.h
│           ├── delegates/
│           │   └── gpu/
│           │       └── delegate.h            # GPU delegate 헤더
│           └── ...
└── flatbuffers/
    └── include/
        └── flatbuffers/
            └── flatbuffers.h
```

### 2.7 CMake 설정

**파일**: `cpp/CMakeLists.txt`

```cmake
# TFLite Pre-built 경로
set(TFLITE_PREBUILT_DIR "${CMAKE_SOURCE_DIR}/third_party/tflite")

# CPU 라이브러리 임포트
add_library(tensorflowlite SHARED IMPORTED)
set_target_properties(tensorflowlite PROPERTIES
    IMPORTED_LOCATION "${TFLITE_PREBUILT_DIR}/android/${ANDROID_ABI}/libtensorflowlite.so"
)

# 헤더 경로
target_include_directories(iris_sdk PRIVATE
    "${TFLITE_PREBUILT_DIR}/include"
    "${TFLITE_PREBUILT_DIR}/flatbuffers/include"
)

# 기본 링크
target_link_libraries(iris_sdk PRIVATE tensorflowlite)

# GPU Delegate (옵션)
set(TFLITE_GPU_DELEGATE_LIB "${TFLITE_PREBUILT_DIR}/android/${ANDROID_ABI}/libtensorflowlite_gpu_delegate.so")
if(EXISTS "${TFLITE_GPU_DELEGATE_LIB}")
    message(STATUS "Found pre-built TFLite GPU Delegate for Android ${ANDROID_ABI}")

    add_library(tflite_gpu_delegate SHARED IMPORTED)
    set_target_properties(tflite_gpu_delegate PROPERTIES
        IMPORTED_LOCATION "${TFLITE_GPU_DELEGATE_LIB}"
    )

    target_link_libraries(iris_sdk PRIVATE tflite_gpu_delegate)
    target_compile_definitions(iris_sdk PRIVATE IRIS_SDK_HAS_GPU_DELEGATE)

    # EGL, GLESv2 링크 (GPU delegate 필수 의존성)
    find_library(EGL_LIB EGL)
    find_library(GLES_LIB GLESv2)
    if(EGL_LIB AND GLES_LIB)
        target_link_libraries(iris_sdk PRIVATE ${EGL_LIB} ${GLES_LIB})
    endif()
endif()
```

---

## 3. Face Mesh 데이터 파이프라인

### 3.1 데이터 흐름

```
C++ (MediaPipeDetector)
    ↓ IrisResult.face_mesh[478] (Point3D 배열)
JNI (iris_jni.cpp)
    ↓ copyResultToJava() - float[1434]로 변환
Java (IrisResult.java)
    ↓ faceMesh float[] 필드
Kotlin (OverlayView.kt)
    ↓ Canvas에 그리기
화면 표시
```

### 3.2 C++ 구조체

**파일**: `cpp/include/iris_sdk/frame_processor.h`

```cpp
struct IrisResult {
    // ... 기존 필드 ...

    // Face Mesh (478개 랜드마크)
    bool face_mesh_valid = false;
    Point3D face_mesh[478];  // x, y, z 정규화 좌표 (0.0~1.0)
};
```

### 3.3 Java 클래스

**파일**: `android/iris-sdk/src/main/java/com/irislenssdk/IrisResult.java`

```java
public class IrisResult {
    // Face Mesh
    public static final int FACE_MESH_LANDMARK_COUNT = 478;
    public boolean faceMeshValid;
    public float[] faceMesh;  // 478 * 3 = 1434 floats

    public void reset() {
        faceMeshValid = false;
        if (faceMesh == null) {
            faceMesh = new float[FACE_MESH_LANDMARK_COUNT * 3];
        }
    }
}
```

### 3.4 JNI 필드 ID 캐시

**파일**: `android/iris-sdk/src/main/cpp/jni_utils.h`

```cpp
struct JniCache {
    // ... 기존 필드 ...

    // Face Mesh 필드
    jfieldID irisResult_faceMeshValid = nullptr;
    jfieldID irisResult_faceMesh = nullptr;
};
```

### 3.5 JNI 데이터 복사

**파일**: `android/iris-sdk/src/main/cpp/iris_jni.cpp`

```cpp
void copyResultToJava(JNIEnv* env, const iris::IrisResult& src, jobject dest) {
    // ... 기존 필드 복사 ...

    // Face Mesh 복사
    env->SetBooleanField(dest, g_jniCache.irisResult_faceMeshValid,
                         src.face_mesh_valid);

    if (src.face_mesh_valid) {
        jfloatArray faceMeshArray = static_cast<jfloatArray>(
            env->GetObjectField(dest, g_jniCache.irisResult_faceMesh));

        if (faceMeshArray) {
            constexpr int LANDMARK_COUNT = 478;
            constexpr int ARRAY_SIZE = LANDMARK_COUNT * 3;
            float tempBuffer[ARRAY_SIZE];

            for (int i = 0; i < LANDMARK_COUNT; ++i) {
                tempBuffer[i * 3] = src.face_mesh[i].x;
                tempBuffer[i * 3 + 1] = src.face_mesh[i].y;
                tempBuffer[i * 3 + 2] = src.face_mesh[i].z;
            }

            env->SetFloatArrayRegion(faceMeshArray, 0, ARRAY_SIZE, tempBuffer);
        }
    }
}
```

### 3.6 Kotlin 시각화

**파일**: `android/demo-app/src/main/java/com/irislenssdk/demo/camera/OverlayView.kt`

```kotlin
class OverlayView : View {
    var showFaceMesh: Boolean = false

    override fun onDraw(canvas: Canvas) {
        // ... 홍채 그리기 ...

        // Face Mesh 표시
        if (showFaceMesh && result.faceMeshValid && result.faceMesh != null) {
            drawFaceMesh(canvas, result, scaleX, scaleY)
        }
    }

    private fun drawFaceMesh(canvas: Canvas, result: IrisResult,
                             scaleX: Float, scaleY: Float) {
        val mesh = result.faceMesh ?: return

        // 478개 랜드마크 점 그리기
        for (i in 0 until 478) {
            val x = mesh[i * 3]
            val y = mesh[i * 3 + 1]
            var screenX = x * imageWidth * scaleX
            val screenY = y * imageHeight * scaleY

            if (isMirror) screenX = width - screenX

            canvas.drawCircle(screenX, screenY, 2f, meshPointPaint)
        }

        // 윤곽선 연결
        drawFaceContour(canvas, mesh, scaleX, scaleY)
        drawEyeContours(canvas, mesh, scaleX, scaleY)
        drawLipsContour(canvas, mesh, scaleX, scaleY)
    }
}
```

---

## 4. MediaPipe 랜드마크 인덱스

### 4.1 얼굴 윤곽 (Face Oval)

```kotlin
val faceOvalIndices = intArrayOf(
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10
)
```

### 4.2 눈 윤곽

```kotlin
// 왼쪽 눈 (화면 기준 오른쪽)
val leftEyeIndices = intArrayOf(
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387,
    386, 385, 384, 398, 362
)

// 오른쪽 눈 (화면 기준 왼쪽)
val rightEyeIndices = intArrayOf(
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158,
    159, 160, 161, 246, 33
)
```

### 4.3 입술 윤곽

```kotlin
val outerLipsIndices = intArrayOf(
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409,
    270, 269, 267, 0, 37, 39, 40, 185, 61
)
```

---

## 5. 카메라 회전 처리

### 5.1 문제
- CameraX 전면 카메라: 270° 회전된 이미지 제공
- 회전 미처리 시 검출 좌표가 이마/코에 표시됨

### 5.2 해결

**파일**: `android/demo-app/src/main/java/com/irislenssdk/demo/camera/FrameAnalyzer.kt`

```kotlin
private fun rotateImage(bitmap: Bitmap, degrees: Int): Bitmap {
    val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
    return Bitmap.createBitmap(bitmap, 0, 0,
                               bitmap.width, bitmap.height, matrix, true)
}

// 회전 적용
val rotatedBitmap = rotateImage(yuvBitmap, rotationDegrees)
```

### 5.3 미러링 (전면 카메라)

```kotlin
// OverlayView.kt
if (isMirror) {
    screenX = width - screenX  // X좌표 반전
}
```

---

## 6. 현재 상태

### 6.1 동작 확인

| 항목 | 상태 | 비고 |
|------|------|------|
| TFLite 로드 | ✅ | arm64-v8a |
| 모델 로드 | ✅ | face_detection, face_landmark, iris_landmark |
| 얼굴 검출 | ✅ | confidence=0.94 |
| 홍채 검출 | ✅ | 양쪽 눈 검출 |
| Face Mesh | ✅ | 478개 랜드마크 표시 |
| 회전 보정 | ✅ | 270° 회전 처리 |
| GPU Delegate 빌드 | ✅ | TF v2.16.1 + NDK 25.1 |
| GPU Delegate CMake 통합 | ✅ | `IRIS_SDK_HAS_GPU_DELEGATE` 매크로 |

### 6.2 성능 (CPU 모드)

| 지표 | 측정값 | 목표 |
|------|--------|------|
| FPS | ~15-20 | 30+ |
| 검출 지연 | ~50-60ms | ≤33ms |
| 메모리 | ~80MB | ≤100MB |

### 6.3 알려진 이슈

| 이슈 | 원인 | 해결 방안 |
|------|------|----------|
| 낮은 FPS | CPU 전용 추론 | GPU delegate 추가 ✅ |
| 가끔 프레임 드랍 | 메인 스레드 부하 | 백그라운드 처리 최적화 |
| TF v2.14.1 빌드 실패 | NDK 27 미지원 | TF v2.16.1 + NDK 25.1 사용 ✅ |
| CoreFoundation 링커 에러 | macOS 크로스컴파일 시 Abseil 이슈 | BUILD.bazel 수정 ✅ |

---

## 7. GPU Delegate 통합 (P1-W6-03)

### 7.1 GPU Delegate 사용법

**파일**: `cpp/src/mediapipe_detector.cpp` (예시)

```cpp
#ifdef IRIS_SDK_HAS_GPU_DELEGATE
#include "tensorflow/lite/delegates/gpu/delegate.h"

// GPU delegate 생성 및 적용
TfLiteGpuDelegateOptionsV2 gpu_options = TfLiteGpuDelegateOptionsV2Default();
gpu_options.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
gpu_options.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;

TfLiteDelegate* gpu_delegate = TfLiteGpuDelegateV2Create(&gpu_options);
if (interpreter_->ModifyGraphWithDelegate(gpu_delegate) == kTfLiteOk) {
    LOG_INFO("GPU delegate enabled successfully");
} else {
    LOG_WARN("GPU delegate failed, falling back to CPU");
    TfLiteGpuDelegateV2Delete(gpu_delegate);
}
#endif
```

### 7.2 GPU Delegate 옵션

| 옵션 | 설명 | 권장값 |
|------|------|--------|
| `inference_preference` | 추론 최적화 방향 | `SUSTAINED_SPEED` |
| `inference_priority1` | 첫 번째 우선순위 | `MIN_LATENCY` |
| `is_precision_loss_allowed` | FP16 허용 여부 | `true` (성능 향상) |

### 7.3 빌드 트러블슈팅

#### NDK 버전 호환성

| TensorFlow 버전 | 지원 NDK | 권장 |
|-----------------|----------|------|
| v2.14.1 | NDK 19-21 | ❌ 권장하지 않음 |
| v2.15.x | NDK 21-25 | ⚠️ |
| **v2.16.1** | **NDK 25** | **✅ 권장** |

#### CoreFoundation 에러 (macOS)

macOS에서 Android 크로스 컴파일 시 발생:
```
ld.lld: error: unknown argument '-framework'
ld.lld: error: cannot open CoreFoundation
```

**원인**: Abseil cctz의 `@platforms//os:osx` 감지가 호스트(macOS)를 기준으로 함

**해결**: Section 2.3 참조

### 7.4 성능 벤치마크 결과

| 모드 | FPS | Avg Latency | Memory |
|------|-----|-------------|--------|
| CPU Only | 21-26 | 10-17ms | 141-202MB |
| GPU Delegate | TBD | TBD | TBD |

> GPU 벤치마크는 P1-W6-03 태스크에서 측정 예정

---

## 8. 빌드 명령어 요약

### Android 빌드

```bash
cd /Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/android

# SDK 빌드
./gradlew :iris-sdk:assembleDebug

# Demo 앱 빌드
./gradlew :demo-app:assembleDebug

# APK 설치
adb install -r build/modules/demo-app/outputs/apk/debug/demo-app-debug.apk
```

### 로그 확인

```bash
# SDK 로그
adb logcat | grep -E "(IrisSDK|IrisLensSDK)"

# 검출 결과
adb logcat | grep "detected="
```

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2025-01-12 | TFLite Bazel 빌드 시작 |
| 2025-01-12 | CMake 연동 완료, 검출 동작 확인 |
| 2025-01-12 | 90° 회전 이슈 해결 |
| 2025-01-13 | Face Mesh 시각화 구현 완료 |
| 2025-01-13 | 문서 작성 |
| 2025-01-13 | TF v2.16.1 + NDK 25.1 빌드로 변경 |
| 2025-01-13 | CoreFoundation workaround 문서화 |
| 2025-01-13 | GPU delegate 빌드 및 CMake 통합 완료 |
