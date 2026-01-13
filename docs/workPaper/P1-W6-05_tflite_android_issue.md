# P1-W6-05: TFLite Android 통합 이슈 분석

**작성일**: 2026-01-13
**상태**: 🔍 분석 중
**심각도**: 🔴 Critical - 핵심 기능 미작동

---

## 1. 우리의 목표

### 최종 목표
**실시간 카메라 영상에서 홍채를 추적하여 가상 렌즈를 오버레이하는 AR 피팅 SDK**

### Phase 1 목표
- 30fps 이상 실시간 홍채 검출
- Android 플랫폼 지원 (arm64-v8a, armeabi-v7a)
- C++ 코어 + JNI 바인딩 아키텍처
- MediaPipe Face Mesh + Iris Tracking 기반 검출

### 핵심 성공 지표
| 지표 | 목표값 | 현재 상태 |
|------|--------|----------|
| 얼굴 검출 | 작동 | ❌ 미작동 |
| 홍채 검출 | 작동 | ❌ 미작동 |
| 프레임레이트 | 30fps+ | ⏳ 측정 불가 |

---

## 2. 목표를 이루기 위해 진행한 작업

### 2.1 아키텍처 설계 (완료)
```
┌─────────────────────────────────────────────────────┐
│                 Android Demo App                     │
│              (CameraX + UI Layer)                    │
└─────────────────────┬───────────────────────────────┘
                      │ JNI
┌─────────────────────▼───────────────────────────────┐
│                  IrisLensSDK                         │
│    ┌─────────────────────────────────────────────┐  │
│    │              Java API Layer                  │  │
│    │  - IrisLensSDK.java (싱글톤)                 │  │
│    │  - IrisResult.java (결과 구조체)            │  │
│    └─────────────────────┬───────────────────────┘  │
│                          │ native calls              │
│    ┌─────────────────────▼───────────────────────┐  │
│    │              JNI Bridge                      │  │
│    │  - iris_jni.cpp                             │  │
│    └─────────────────────┬───────────────────────┘  │
│                          │                           │
│    ┌─────────────────────▼───────────────────────┐  │
│    │           C++ Core Engine                    │  │
│    │  - sdk_api.cpp (C API 래퍼)                 │  │
│    │  - FrameProcessor (프레임 처리)             │  │
│    │  - MediaPipeDetector (TFLite 기반 검출)     │  │
│    │  - LensRenderer (렌즈 렌더링)               │  │
│    └─────────────────────┬───────────────────────┘  │
│                          │                           │
│    ┌─────────────────────▼───────────────────────┐  │
│    │            Third Party                       │  │
│    │  - TensorFlow Lite (모델 추론)              │  │
│    │  - OpenCV (이미지 처리)                     │  │
│    └─────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### 2.2 C++ 코어 개발 (완료)
| 컴포넌트 | 파일 | 상태 | 설명 |
|----------|------|------|------|
| IrisDetector | iris_detector.h/cpp | ✅ | 검출기 인터페이스 |
| MediaPipeDetector | mediapipe_detector.h/cpp | ✅ | TFLite 기반 구현체 |
| FrameProcessor | frame_processor.h/cpp | ✅ | 프레임 처리 파이프라인 |
| SDKManager | sdk_manager.h/cpp | ✅ | 싱글톤 관리자 |
| SDK API | sdk_api.h/cpp | ✅ | C API 래퍼 |
| LensRenderer | lens_renderer.h/cpp | ✅ | 렌즈 오버레이 |

### 2.3 JNI 바인딩 (완료)
| 함수 | 상태 | 설명 |
|------|------|------|
| nativeInit | ✅ | SDK 초기화 |
| nativeDestroy | ✅ | SDK 해제 |
| nativeDetect | ✅ | 기본 검출 |
| nativeDetectWithRotation | ✅ | 회전 보정 검출 |
| nativeRenderLens | ✅ | 렌즈 렌더링 |

### 2.4 Android 통합 (부분 완료)
| 항목 | 상태 | 설명 |
|------|------|------|
| Demo App UI | ✅ | CameraX 프리뷰 + 오버레이 |
| FrameAnalyzer | ✅ | YUV→NV21 변환, rotation 전달 |
| 모델 에셋 | ✅ | .tflite 파일 assets 폴더에 배치 |
| OpenCV Android | ✅ | third_party에 SDK 배치 |
| TFLite Android | ❌ | **빌드 실패** |

### 2.5 데스크톱 빌드 검증 (성공)
- CLion cmake-build-debug에서 TFLite + OpenCV 정상 빌드
- 데스크톱 테스트에서 얼굴/홍채 검출 정상 작동 확인

---

## 3. 현재 문제상황

### 3.1 증상
```
V  nativeDetectWithRotation called: 1088x1088, format=4, rotation=270
V  Detection with rotation completed: detected=0, confidence=0.00
```
- SDK 초기화: ✅ 성공
- 모델 로딩: ✅ 성공 (로그상)
- 이미지 전달: ✅ 정상
- **얼굴 검출: ❌ 항상 detected=0**

### 3.2 근본 원인
```cpp
// mediapipe_detector.cpp:1744
#if defined(IRIS_SDK_HAS_TFLITE) && defined(IRIS_SDK_HAS_OPENCV)
    // 실제 검출 로직 (TFLite 추론)
#else
    // 이 블록이 실행됨 - 항상 detected=false 반환
#endif
```

**`IRIS_SDK_HAS_TFLITE`가 Android 빌드에서 정의되지 않음**

### 3.3 TFLite가 비활성화된 이유

#### CMakeLists.txt (android/iris-sdk/src/main/cpp/)
```cmake
# 80-81줄
set(IRIS_SDK_FETCH_TFLITE OFF CACHE BOOL "" FORCE)
set(IRIS_SDK_USE_SYSTEM_TFLITE OFF CACHE BOOL "" FORCE)
```

의도적으로 비활성화한 이유:
1. FetchContent로 TFLite 빌드 시 10-30분 소요
2. 400MB+ 다운로드 필요
3. Android NDK와 호환성 문제 예상

### 3.4 FetchContent 빌드 시도 결과
```
IRIS_SDK_FETCH_TFLITE=ON 설정 후 빌드 시도
→ Eigen 라이브러리의 컴파일러 플래그 호환성 오류
→ -lpthread 링크 오류
→ 빌드 실패
```

---

## 4. 해결 방안들

### 방안 A: TFLite Pre-built 라이브러리 사용
**개요**: 이미 빌드된 TFLite Android 라이브러리를 다운로드하여 링크

**출처 옵션**:
1. [cuongvng/TF-Lite-Cpp-API-for-Android](https://github.com/cuongvng/TF-Lite-Cpp-API-for-Android)
2. TensorFlow Lite AAR에서 추출 (C API만 제공)
3. Bazel로 직접 빌드

**장점**:
- 기존 C++ 코드 100% 유지
- 아키텍처 변경 없음
- JNI, rotation 등 모든 작업 그대로 활용

**단점**:
- Pre-built 버전과 헤더 호환성 확인 필요
- TFLite 버전 고정 (업데이트 어려움)
- C++ API vs C API 차이 확인 필요

**예상 작업량**: 1-2일

---

### 방안 B: MediaPipe Android SDK (Java) 사용
**개요**: Google의 MediaPipe Tasks Vision AAR을 Java 계층에서 사용

```kotlin
// Gradle 의존성
implementation("com.google.mediapipe:tasks-vision:0.10.14")
```

**장점**:
- Google 공식 지원, 안정성 보장
- TFLite 내장, 별도 통합 불필요
- 최신 모델 자동 업데이트

**단점**:
- **기존 C++ MediaPipeDetector 코드 폐기**
- Java에서 검출 → JNI로 결과 전달 → C++ 렌더링 (역방향 흐름)
- 아키텍처 재설계 필요
- Phase 1 C++ 코어 작업의 상당 부분 무의미화

**예상 작업량**: 3-5일

---

### 방안 C: TFLite C API로 코드 수정
**개요**: C++ API (`tflite::Interpreter`) 대신 C API (`TfLiteInterpreter`)로 변경

**현재 코드**:
```cpp
std::unique_ptr<tflite::Interpreter> interpreter;
tflite::InterpreterBuilder builder(*model, resolver);
interpreter->Invoke();
```

**변경 후**:
```c
TfLiteInterpreter* interpreter;
TfLiteInterpreterCreate(model, options);
TfLiteInterpreterInvoke(interpreter);
```

**장점**:
- TensorFlow Lite AAR에서 C API 라이브러리 추출 가능
- 공식 지원 API 사용
- 기존 아키텍처 유지

**단점**:
- MediaPipeDetector 전체 리팩토링 필요 (500+ 줄)
- C API는 C++ API보다 verbose
- 에러 처리 방식 변경

**예상 작업량**: 5-7일 (RAII 래퍼 작성 + 에러 처리 전면 재작성 포함)

---

### 방안 D: Bazel로 TFLite 직접 빌드
**개요**: Google의 공식 빌드 시스템으로 Android용 TFLite 빌드

```bash
bazel build -c opt --config=android_arm64 \
  //tensorflow/lite:libtensorflowlite.so
```

**장점**:
- 공식 빌드 방법
- 최신 버전 사용 가능
- 커스텀 옵션 가능

**단점**:
- Bazel 설치 및 학습 필요
- 빌드 환경 복잡 (Java, Python, Bazel 버전 호환)
- 빌드 시간 30분-1시간+
- CI/CD 파이프라인 복잡화

**예상 작업량**: 2-3일 (환경 설정 포함)

---

## 5. 해결 방안과 기존 개발의 차이점

### 5.1 방안별 영향도 비교

| 기존 작업 | 방안 A (Pre-built) | 방안 B (Java SDK) | 방안 C (C API) | 방안 D (Bazel) |
|-----------|-------------------|-------------------|----------------|----------------|
| C++ IrisDetector 인터페이스 | ✅ 유지 | ⚠️ 축소 | ✅ 유지 | ✅ 유지 |
| MediaPipeDetector (1700줄) | ✅ 유지 | ❌ 폐기 | 🔄 리팩토링 | ✅ 유지 |
| FrameProcessor | ✅ 유지 | ⚠️ 수정 | ✅ 유지 | ✅ 유지 |
| JNI 바인딩 | ✅ 유지 | 🔄 역방향 수정 | ✅ 유지 | ✅ 유지 |
| rotation 지원 | ✅ 유지 | 🔄 Java로 이동 | ✅ 유지 | ✅ 유지 |
| 모델 파일 (.tflite) | ✅ 유지 | ⚠️ MediaPipe 모델 | ✅ 유지 | ✅ 유지 |
| 크로스플랫폼 가능성 | ✅ 높음 | ❌ Android 전용 | ✅ 높음 | ✅ 높음 |

### 5.2 아키텍처 변화

#### 현재 (방안 A, C, D)
```
Camera → JNI → C++ FrameProcessor → C++ MediaPipeDetector → TFLite
                                                              ↓
                                            C++ LensRenderer ← IrisResult
                                                              ↓
                                                         JNI → Java UI
```

#### 방안 B (Java SDK)
```
Camera → Java MediaPipe FaceLandmarker → Java IrisResult
                                              ↓
         Java UI ← JNI ← C++ LensRenderer ← JNI (결과 전달)
```

### 5.3 코드 재사용률

| 방안 | 재사용률 | 폐기 코드 | 신규 코드 |
|------|----------|----------|----------|
| A (Pre-built) | ~95% | 없음 | CMake 설정 |
| B (Java SDK) | ~40% | MediaPipeDetector, 일부 FrameProcessor | Java 검출기 래퍼 |
| C (C API) | ~60-70% | 없음 | RAII 래퍼 + MediaPipeDetector 전면 리팩토링 |
| D (Bazel) | ~95% | 없음 | Bazel 빌드 설정 |

---

## 6. 전문가 에이전트 평가 결과

### 6.1 평가 참여 전문가
- **C++ 시스템 프로그래밍 전문가** (systems-programming:cpp-pro)
- **ML 엔지니어** (machine-learning-ops:ml-engineer)
- **MLOps 엔지니어** (machine-learning-ops:mlops-engineer)

### 6.2 전문가별 권장 순서

| 전문가 | 1순위 | 2순위 | 3순위 | 4순위 | 핵심 관점 |
|--------|-------|-------|-------|-------|----------|
| **C++ 전문가** | A | D | B | C | RAII 패턴 변환 복잡도 |
| **ML 엔지니어** | B | A | C | D | 모델 호환성, 정확도 |
| **MLOps 엔지니어** | A | C | D | B | CI/CD, 크로스플랫폼 |

### 6.3 전문가별 상세 평가

---

#### 📘 C++ 시스템 프로그래밍 전문가 평가

**평가 요청 항목**:
1. C++ API vs C API 리팩토링 기술적 타당성
2. Pre-built 라이브러리 헤더/버전 호환성 이슈
3. CMake FetchContent 실패 원인 분석
4. 권장 순서에 대한 C++ 관점 평가

**주요 평가 내용**:

| 항목 | 평가 결과 |
|------|----------|
| **방안 A** | ✅ **권장** - 즉시 적용 가능, ABI 안정성, Google 공식 지원 |
| **방안 B** | ⚠️ 중간 - JNI 오버헤드 있으나 안정적 |
| **방안 C** | ❌ **비권장** - RAII 재구현 부담 과다, 작업량 과소평가됨 |
| **방안 D** | ⚠️ 중간 - 공식 방법이나 빌드 시간 리스크 |

**핵심 기술 분석**:

| 측면 | C++ API | C API |
|------|---------|-------|
| 메모리 관리 | 자동 (unique_ptr) | 수동 (malloc/free) |
| 예외 안전성 | 보장됨 | 직접 구현 필요 |
| 타입 안전성 | 컴파일 타임 체크 | void* 캐스팅 다수 |
| 에러 처리 | 예외/optional | 반환 코드 체크 |

**결론**: "방안 C의 코드 재사용률 80%는 낙관적. 실제로는 60-70% 수준이며, RAII 래퍼 작성 + 에러 처리 로직 전면 재작성 필요"

---

#### 📗 ML 엔지니어 평가

**평가 요청 항목**:
1. Pre-built TFLite와 .tflite 모델 파일 호환성
2. MediaPipe Java SDK 장단점 (모델 업데이트, 정확도)
3. TFLite C API vs C++ API 성능/기능 차이
4. 권장 순서에 대한 ML 관점 평가

**주요 평가 내용**:

| 항목 | 평가 결과 |
|------|----------|
| **방안 A** | ⚠️ 중간 - TFLite 버전과 모델 호환성 확인 필요 |
| **방안 B** | ✅ **강력 권장** - Google 공식 지원, 최적화된 추론, 모델 자동 업데이트 |
| **방안 C** | ⚠️ 중간 - 성능 동일하나 빌드 복잡도 감소 |
| **방안 D** | ❌ 비권장 - 실제로 매우 어려움, 환경 설정 복잡 |

**모델 호환성 분석**:

| 리스크 요소 | 심각도 | 설명 |
|------------|--------|------|
| Op 버전 불일치 | 중간 | MediaPipe 모델은 특정 TFLite 버전에 맞춰 빌드됨 |
| Custom Op 미지원 | 낮음 | 공식 모델은 표준 Op만 사용 |
| Delegate 호환성 | 중간 | XNNPACK/NNAPI 버전에 따라 가속 불가 가능 |

**MediaPipe Java SDK 장점**:
- Google이 직접 유지보수, 버그 수정 보장
- 내부적으로 GPU delegate, NNAPI 자동 선택
- V2 모델 기본 사용 → 홍채 랜드마크(468-477) 기본 포함
- Gradle 의존성 한 줄로 완료

**결론**: "현재 상황에서 방안 B가 가장 현실적인 선택. 데스크톱은 C++ TFLite 유지, 모바일은 플랫폼 네이티브 SDK 사용이 프로덕션 전략"

---

#### 📙 MLOps 엔지니어 평가

**평가 요청 항목**:
1. 각 방안의 CI/CD 통합 용이성
2. TFLite/MediaPipe 버전 업데이트 전략
3. iOS, Flutter 확장 시 각 방안의 영향
4. 2-3년 관점 장기 유지보수 분석

**주요 평가 내용**:

| 항목 | 평가 결과 |
|------|----------|
| **방안 A** | ✅ **최적** - 빌드 1-2분, 캐싱 효과 높음 |
| **방안 B** | ❌ **경고** - 크로스플랫폼 포기 = 유지보수 비용 4배 |
| **방안 C** | ⚠️ 양호 - A 실패 시 대안 |
| **방안 D** | ❌ **비권장** - 빌드 10-30분, CI/CD 복잡화 |

**CI/CD 통합 분석**:

| 방안 | 빌드 시간 | 캐싱 효과 | CI/CD 복잡도 |
|------|-----------|-----------|-------------|
| A | 1-2분 | 높음 | 낮음 |
| B | 2-3분 | 높음 | 낮음 |
| C | 3-5분 | 중간 | 중간 |
| D | 10-30분 | 낮음 | **높음** |

**크로스플랫폼 확장 영향**:

| 방안 | Android | iOS | Flutter | Web | 통합 복잡도 |
|------|---------|-----|---------|-----|------------|
| A | C++ 공유 | C++ 공유 | dart:ffi | WASM | 낮음 |
| B | Java 전용 | 별도 구현 | 별도 구현 | 별도 구현 | **매우 높음** |
| C | C API | C API | C API | 제한적 | 중간 |
| D | C++ 공유 | C++ 공유 | dart:ffi | WASM | 중간 |

**장기 유지보수 리스크 (2-3년)**:

| 방안 | 버전 호환성 | 인력 의존성 | 확장 비용 |
|------|------------|------------|----------|
| A | 중간 (수동) | 낮음 | 낮음 |
| B | 낮음 (자동) | **높음 (4코드)** | **매우 높음** |
| C | 중간 | 중간 | 낮음 |
| D | 높음 | 높음 (전문가) | 중간 |

**결론**: "크로스플랫폼 SDK 프로젝트에서 플랫폼별 분리는 기술 부채의 가장 큰 원천. 방안 B는 단기적으로 쉬워 보이지만, 2-3년 관점에서 유지보수 비용이 기하급수적으로 증가"

---

### 6.5 의견 충돌 분석

| 쟁점 | C++/MLOps 의견 | ML 엔지니어 의견 |
|------|---------------|-----------------|
| **방안 B 평가** | ❌ 크로스플랫폼 포기 | ✅ 빠른 출시, 안정성 |
| **방안 C 평가** | ❌ 리팩토링 과다 | ⚠️ 대안으로 적절 |
| **우선순위** | 크로스플랫폼 유지 | 동작하는 코드 우선 |

**충돌 원인**: 프로젝트 목표 해석 차이
- 크로스플랫폼 필수 → A 우선
- Android 동작 우선 → B 우선

---

## 7. 총평

### 7.1 수정된 권장 방안: **A (Pre-built) > C (C API) > D (Bazel) > B (Java SDK)**

#### 방안 A를 최우선으로 권장하는 이유:
1. **기존 작업 최대 보존**: C++ 코어, JNI, rotation 등 모든 작업 유지
2. **최소 변경**: CMake 설정만 수정
3. **빠른 적용**: 1-2일 내 검증 가능
4. **크로스플랫폼 유지**: iOS, Flutter, Web 확장 가능성 유지

#### 방안 B를 비권장하는 이유 (MLOps 관점 강화):
1. **기존 작업 50% 이상 폐기**
2. **아키텍처 역전**: C++→Java 대신 Java→C++ 흐름
3. **크로스플랫폼 포기**: 4개 코드베이스 유지 = **유지보수 비용 4배**
4. **Phase 1 목표 변경**: "C++ 코어 + JNI" 아키텍처 포기
5. **장기 기술 부채**: 2-3년 후 유지보수 비용 기하급수적 증가

#### 권장 순서 변경 근거 (D → C 순서 변경):
- **Bazel + CMake 혼합**: CI/CD 파이프라인 2개 관리 필요
- **빌드 캐시 전략 복잡화**: Bazel과 CMake 캐시 분리
- **C API는 NDK 호환성 우수**: 단일 `.so` 링크로 충분

### 7.2 리스크 분석 (전문가 의견 반영)

| 방안 | 기술 리스크 | 일정 리스크 | 유지보수 리스크 | CI/CD 복잡도 |
|------|------------|------------|----------------|-------------|
| A | 중 (버전 호환) | 낮음 | 중 (업데이트) | 낮음 |
| B | 낮음 | 중 (재설계) | **높음 (4코드)** | 낮음 |
| C | 중 (리팩토링) | **높음** | 낮음 | 중간 |
| D | 높음 (환경) | 높음 | 낮음 | **높음** |

### 7.3 다음 단계 제안 (수정)

1. **방안 A 먼저 시도** (1-2일)
   - TensorFlow Lite AAR에서 pre-built `.so` 추출
   - C++ API 헤더 버전 일치 확인
   - `nm -D`로 심볼 존재 확인
   - 간단한 inference 테스트

```bash
# AAR에서 .so 추출
unzip tensorflow-lite-2.14.0.aar -d tflite_aar
cp tflite_aar/jni/arm64-v8a/libtensorflowlite_jni.so third_party/tflite/
```

2. **실패 시 방안 C로 전환** (5-7일) - ⚠️ 작업량 상향 조정
   - AAR에서 C API 라이브러리 추출
   - RAII 래퍼 클래스 작성
   - MediaPipeDetector를 C API로 전면 리팩토링

3. **그래도 실패 시 방안 D** (1-2주)
   - Bazel 환경 설정 (학습 곡선 포함)
   - TensorFlow 전체 리포지토리 필요 (~2GB)
   - 직접 빌드

4. **최후 수단으로 방안 B** (3-5일)
   - ⚠️ 크로스플랫폼 포기 결정 필요
   - 아키텍처 재설계
   - Java SDK 통합

### 7.4 결론

> **현재 "No face detected" 문제는 TFLite 라이브러리 부재로 인한 것이며,
> 기존 C++ 코드 자체는 정상입니다 (데스크톱에서 검증됨).**
>
> **방안 A (Pre-built 라이브러리)로 시작하여 기존 작업을 최대한 보존하면서
> 문제를 해결하는 것을 권장합니다.**

### 7.5 의사결정 가이드

| 상황 | 권장 방안 | 이유 |
|------|----------|------|
| 크로스플랫폼 필수 | A > C > D > B | 코드 공유, 장기 유지보수 |
| Android 출시 급함 | B > A | 1-2일 내 동작 가능 |
| 최대 성능 필요 | A + 커스텀 최적화 | C++ 수준 제어 |
| 팀에 Bazel 전문가 있음 | D > A | 공식 빌드 방법 |

---

## 8. 하이브리드 아키텍처 제안 (ML 엔지니어)

크로스플랫폼 목표를 유지하면서 플랫폼별 최적화를 달성하는 대안:

```
┌──────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
└──────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│     Android     │  │       iOS       │  │    Desktop      │
│                 │  │                 │  │                 │
│ MediaPipe Tasks │  │ MediaPipe Tasks │  │   C++ TFLite    │
│   (Java SDK)    │  │   (Swift SDK)   │  │   (기존 코드)   │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              ▼
                ┌─────────────────────────┐
                │  Common Data Structures │
                │   (IrisResult, Lens)    │
                │       Shared Logic      │
                └─────────────────────────┘
```

**이 접근법의 장단점**:
- ✅ 각 플랫폼의 공식 SDK 사용 → 안정성
- ✅ 데이터 구조와 렌더링 로직은 공유
- ❌ 검출 로직 플랫폼별 분리 → 동작 차이 가능
- ❌ 유지보수 복잡도 증가

**적용 시점**: 방안 A~D 모두 실패 시 고려

---

## 변경 이력

| 날짜 | 버전 | 변경 내용 |
|------|------|----------|
| 2026-01-13 | 1.0 | 초안 작성 |
| 2026-01-13 | 2.0 | 전문가 에이전트 평가 결과 반영 |
|            |     | - 섹션 6 추가: 전문가별 상세 평가 |
|            |     | - 권장 순서 수정: A > D > C > B → A > C > D > B |
|            |     | - 방안 C 작업량 조정: 2-3일 → 5-7일 |
|            |     | - 방안 C 재사용률 수정: 80% → 60-70% |
|            |     | - 섹션 8 추가: 하이브리드 아키텍처 제안 |
