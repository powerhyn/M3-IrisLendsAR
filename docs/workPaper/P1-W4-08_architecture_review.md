# P1-W4-08: 아키텍처 리뷰

**태스크 ID**: P1-W4-08
**상태**: ✅ 완료
**시작일**: 2026-01-12
**완료일**: 2026-01-12

---

## 1. 개요

### 목적
Phase 1 Week 3-4 코어 엔진 완료 후, Week 5-6 Android 바인딩 구현 전 아키텍처 검증

### 리뷰 범위
- `cpp/include/iris_sdk/` 헤더 파일 8개
- `cpp/src/` 구현 파일 7개
- 핵심 컴포넌트: SDKManager, IrisDetector, MediaPipeDetector, LensRenderer, FrameProcessor, C API

---

## 2. 아키텍처 평가 요약

| 영역 | 평점 | 상태 |
|------|------|------|
| 레이어 분리 | A | 우수 - JNI 래핑에 최적화됨 |
| 인터페이스 설계 | A | 우수 - Strategy 패턴 적절히 적용 |
| JNI 준비 상태 | A | 우수 - POD 타입, 에러 코드 설계 적절 |
| 메모리 관리 | B+ | 양호 - 스마트 포인터 일관적 사용, 개선 여지 존재 |
| 확장성 | A | 우수 - 플랫폼 확장 기반 마련됨 |
| 성능 아키텍처 | B+ | 양호 - 파이프라인 구조 적절, GPU 가속 미완성 |

---

## 3. 상세 평가

### 3.1 레이어 분리 (평점: A)

```
┌─────────────────────────────────────┐
│         C API Layer                  │  ← sdk_api.h (extern "C")
│   iris_sdk_init/detect/render_lens  │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│        SDKManager (Singleton)        │  ← 팩토리/라이프사이클 관리
│   getInstance/initialize/shutdown   │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│      Core Components Layer           │
│   FrameProcessor → IrisDetector     │
│                 → LensRenderer      │
└─────────────────────────────────────┘
```

**강점**:
- C API 레이어가 완전히 분리되어 JNI 래핑에 최적화됨
- SDKManager가 팩토리 메서드를 통해 컴포넌트 생성을 중앙 관리
- Pimpl 패턴으로 구현 세부사항 은닉 (ABI 안정성 확보)

**개선 포인트**:
- `sdk_api.cpp`의 전역 변수(`g_processor`, `g_mutex`)는 단일 인스턴스 제약을 강제함
- 멀티 인스턴스가 필요한 경우 핸들 기반 API로 확장 고려

---

### 3.2 인터페이스 설계 (평점: A)

**Strategy 패턴 적용**:
```cpp
// iris_detector.h
class IrisDetector {
public:
    virtual bool initialize(const std::string& model_path) = 0;
    virtual IrisResult detect(...) = 0;
    virtual void release() = 0;
    virtual DetectorType getDetectorType() const = 0;
};

// 구현체
class MediaPipeDetector : public IrisDetector { ... };
// Phase 2 예정
// class EyeOnlyDetector : public IrisDetector { ... };
// class HybridDetector : public IrisDetector { ... };
```

**강점**:
- 추상 인터페이스(`IrisDetector`)와 구현체(`MediaPipeDetector`) 분리
- `DetectorType` 열거형으로 런타임 타입 식별 가능
- 팩토리 함수(`detail::createDetector`)로 생성 로직 캡슐화

**개선 포인트**:
- `detail` 네임스페이스 팩토리는 테스트용으로 적절하나, 외부 노출 시 주의 필요

---

### 3.3 JNI 바인딩 준비 상태 (평점: A)

**C API 설계 분석** (`sdk_api.h`):

| 항목 | 상태 | 설명 |
|------|------|------|
| POD 타입 사용 | ✅ | `IrisLandmark`, `IrisRect`, `IrisResult` 모두 POD |
| 고정 크기 배열 | ✅ | `face_mesh[478]`, `left_iris[5]` 등 |
| 에러 반환 방식 | ✅ | `IrisSdkError` 열거형 (int 기반) |
| 문자열 처리 | ✅ | `const char*` 입력, 정적 문자열 반환 |
| 콜백 지원 | ⚠️ | 현재 미구현 (비동기 처리 시 필요) |

**JNI 래핑 용이성**:
```java
// 예상되는 JNI 네이티브 메서드 시그니처
public native int iris_sdk_init(String modelPath);
public native int iris_sdk_detect(byte[] frameData, int width, int height, int format, IrisResult result);
```

**강점**:
- 모든 구조체가 POD 타입으로 `memcpy` 기반 마샬링 가능
- 에러 코드가 정수형으로 JNI `jint` 직접 반환 가능
- 프레임 포맷에 `NV21` 포함 (Android 카메라 기본 포맷)

---

### 3.4 메모리 관리 (평점: B+)

**스마트 포인터 사용**:
```cpp
// SDKManager - Pimpl
std::unique_ptr<Impl> impl_;

// FrameProcessor 생성
std::unique_ptr<FrameProcessor> createFrameProcessor();

// 전역 프로세서 (sdk_api.cpp)
std::unique_ptr<iris_sdk::FrameProcessor> g_processor;
```

**강점**:
- `unique_ptr` 일관적 사용으로 RAII 패턴 준수
- 복사/이동 시맨틱 명시적 관리 (`delete`/`noexcept`)
- `LensRenderer`, `FrameProcessor`에서 이동 지원

**감점 요인**:

#### 1. IrisResult 구조체 크기 문제
```cpp
struct IrisResult {
    // ...
    IrisLandmark face_mesh[478];  // 478 * 16 bytes = ~7.6KB
    // ...
};
```
- `face_mesh[478]` 배열이 **약 7.6KB** 크기
- JNI로 **매 프레임마다 전체 복사** 시 성능 부담
- 현재 `face_mesh_valid` 플래그로 선택적 전달 가능하지만, **기본적으로 비활성화 옵션이 C API에 없음**

#### 2. 전역 Mutex 경합 가능성
```cpp
// sdk_api.cpp
static std::mutex g_mutex;
static std::unique_ptr<FrameProcessor> g_processor;

IrisSdkError iris_sdk_detect(...) {
    std::lock_guard<std::mutex> lock(g_mutex);  // 모든 호출에서 락
    // ...
}
```
- **모든 C API 호출**이 단일 mutex로 보호됨
- 30fps (33ms 간격) 호출 시 경합 발생 가능
- 인스턴스별 mutex나 lock-free 패턴이 더 효율적

**개선 권장사항**:

| 우선순위 | 항목 | 현재 상태 |
|---------|------|----------|
| 높음 | C API에 `include_face_mesh` 플래그 추가 | 미구현 |
| 중간 | 인스턴스별 mutex 또는 핸들 기반 API | 미구현 |
| 낮음 | 메모리 풀링 (프레임 버퍼 재사용) | 부분 구현 |

---

### 3.5 확장성 (평점: A)

**Phase 2 확장 준비 상태**:

| 플랫폼 | 확장 난이도 | 주요 작업 |
|--------|------------|----------|
| Android | 낮음 | JNI 래퍼 + AAR 패키징 |
| iOS | 낮음 | Obj-C++ 브리지 + xcframework |
| Flutter | 중간 | dart:ffi 바인딩 + 플러그인 구조 |
| Web | 중간-높음 | Emscripten 빌드 + WASM 최적화 |

**아키텍처 확장 포인트**:
1. **새 검출기 추가**: `IrisDetector` 상속 → `createDetector` 팩토리 수정
2. **새 렌더링 효과**: `LensConfig`에 필드 추가 (POD 호환 유지)
3. **새 프레임 포맷**: `FrameFormat` 열거형 확장 + 변환 로직 추가

---

### 3.6 성능 고려사항 (평점: B+)

**실시간 처리(30fps+) 아키텍처 적합성**:

| 항목 | 평가 | 상세 |
|------|------|------|
| 파이프라인 구조 | ✅ | `FrameProcessor`가 검출+렌더링 통합 |
| 버퍼 관리 | ✅ | Pimpl 내부 버퍼 재사용 |
| 포맷 변환 | ⚠️ | `convert_time_ms` 추적 중, NV21→RGB 변환 비용 존재 |
| 스레드 안전성 | ⚠️ | 전역 mutex 사용 (경합 발생 가능) |
| GPU 가속 | ⏳ | `enable_gpu` 플래그 존재, 현재 미지원 |

**성능 메트릭 추적**:
```cpp
struct ProcessResult {
    float processing_time_ms;   // 총 처리 시간
    float detection_time_ms;    // 검출 시간
    float render_time_ms;       // 렌더링 시간
    float convert_time_ms;      // 포맷 변환 시간
};
```

---

## 4. Android 바인딩(Week 5-6) 구현 주의사항

### 4.1 JNI 레이어 설계 권장

```
┌─────────────────────────────────────┐
│  Android App (Kotlin/Java)          │
│  CameraX / Camera2 API              │
└─────────────────┬───────────────────┘
                  │ byte[], ImageProxy
┌─────────────────▼───────────────────┐
│  IrisSDK.kt (Kotlin Wrapper)        │
│  suspend fun detect(): IrisResult   │
└─────────────────┬───────────────────┘
                  │ JNI call
┌─────────────────▼───────────────────┐
│  iris_sdk_jni.cpp (JNI Bridge)      │
│  Java_com_iris_sdk_IrisSDK_detect   │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│  C API (sdk_api.h)                  │
│  iris_sdk_detect()                  │
└─────────────────────────────────────┘
```

### 4.2 핵심 구현 주의사항

1. **NV21 포맷 처리**: Android 카메라 기본 포맷. C API에서 `IRIS_FORMAT_NV21` 지원됨

2. **IrisResult 크기 문제**:
   - `face_mesh[478]` 배열로 인해 약 8KB 크기
   - JNI로 매 프레임 전체 복사는 비효율적
   - **권장**: `face_mesh_valid=false`로 디버그 모드가 아닐 때 스킵

3. **전역 mutex 경합**:
   - `sdk_api.cpp`의 `g_mutex`가 모든 C API 호출에서 사용됨
   - 고빈도 호출(30fps) 시 경합 발생 가능
   - **대안**: 인스턴스별 mutex 또는 lock-free 패턴 고려

4. **모델 파일 경로**:
   - Android에서는 assets 폴더 파일 직접 접근 불가
   - APK에서 추출 후 내부 저장소 경로 전달 필요
   ```kotlin
   val modelPath = "${context.filesDir}/models/"
   copyAssetsToInternal("models", modelPath)
   IrisSDK.init(modelPath)
   ```

5. **카메라 방향**:
   - 전면/후면 카메라에 따른 미러링 처리
   - 디바이스 회전에 따른 이미지 회전 처리 필요

---

## 5. 개선 권장사항

### 5.1 단기 개선 (Android 바인딩 전)

| 우선순위 | 항목 | 상세 |
|---------|------|------|
| 높음 | face_mesh 선택적 전달 | C API에 `include_face_mesh` 플래그 추가 |
| 중간 | 비동기 콜백 API | 고빈도 처리용 `iris_sdk_process_async` 추가 고려 |
| 낮음 | 핸들 기반 API | 멀티 인스턴스 지원 시 `IrisHandle` 도입 |

### 5.2 중기 개선 (Phase 2)

1. **GPU 가속 활성화**:
   - TFLite GPU 델리게이트 통합
   - OpenGL/Vulkan 기반 렌더링

2. **비동기 파이프라인**:
   - 검출과 렌더링 분리된 스레드 처리
   - 프레임 드롭 전략 도입

3. **핫스왑 검출기**:
   - 런타임에 `MediaPipe` ↔ `EyeOnly` 전환
   - 얼굴 전체 미검출 시 `EyeOnly`로 폴백

---

## 6. Phase 2 확장 고려사항

### 6.1 플랫폼별 확장 전략

**iOS (Obj-C++ 브리지)**:
```objc
// IrisSDK.mm
@implementation IrisSDK
- (BOOL)initWithModelPath:(NSString*)path {
    return iris_sdk_init([path UTF8String]) == IRIS_SDK_OK;
}
@end
```

**Flutter (dart:ffi)**:
```dart
final DynamicLibrary _lib = Platform.isAndroid
    ? DynamicLibrary.open('libiris_sdk.so')
    : DynamicLibrary.open('IrisSDK.framework/IrisSDK');
```

**Web (WASM)**:
- Emscripten 빌드로 `.wasm` 생성
- WebGL 렌더링 통합 필요
- 모델 파일 fetch API로 다운로드

### 6.2 공통 고려사항

- **모델 버전 관리**: 모델 파일 버전과 SDK 버전 호환성 체크
- **에러 로깅**: 플랫폼별 로그 시스템 연동 (Android Logcat, iOS os_log)
- **프로파일링**: 플랫폼별 성능 측정 도구 연동

---

## 7. 결론

현재 아키텍처는 **Android JNI 바인딩 구현에 충분히 준비**되어 있으며, Week 5-6 마일스톤 달성에 큰 장애물이 없을 것으로 판단됩니다.

**핵심 액션 아이템**:
1. **즉시 적용**: `face_mesh` 선택적 전달 옵션 추가 (JNI 효율성)
2. **Android 구현 시**: NV21 변환 최적화, 카메라 방향 처리 주의
3. **Phase 2 준비**: GPU 델리게이트 통합, 비동기 파이프라인 설계

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-01-12 | 아키텍처 리뷰 수행 및 문서화 |
