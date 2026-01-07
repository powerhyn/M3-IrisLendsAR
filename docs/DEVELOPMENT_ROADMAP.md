# IrisLensSDK 개발 로드맵

## 전체 개발 단계 개요

```
Phase 1: MVP (MediaPipe 기반)
    ↓
Phase 2: 하이브리드 시스템 (커스텀 모델 추가)
    ↓
Phase 3: 최적화 및 확장
```

---

## Phase 1: MVP 개발 (예상 6-8주)

### Week 1-2: 환경 설정 및 기반 구축

#### 목표
- [ ] 개발 환경 구성
- [ ] 프로젝트 구조 생성
- [ ] 의존성 라이브러리 빌드

#### 상세 작업

**1.1 개발 환경**
```
필요 도구:
├── CMake 3.18+
├── C++17 호환 컴파일러
│   ├── macOS: Xcode Command Line Tools
│   ├── Linux: GCC 9+ / Clang 10+
│   └── Windows: MSVC 2019+
├── Android NDK r21+
├── Xcode 13+ (iOS 빌드용)
└── Python 3.8+ (MediaPipe 빌드 스크립트용)
```

**1.2 MediaPipe 빌드**
```bash
# Bazel 설치 (MediaPipe 빌드에 필요)
# MediaPipe 소스 클론
git clone https://github.com/google/mediapipe.git

# 필요한 그래프만 추출하여 C++ 라이브러리화
# - face_mesh
# - iris_tracking
```

**1.3 OpenCV 빌드**
```bash
# 최소 모듈만 포함하여 빌드 (SDK 크기 최소화)
cmake -DBUILD_LIST=core,imgproc,imgcodecs ...
```

**1.4 프로젝트 구조 생성**
```
iris-lens-sdk/
├── CMakeLists.txt
├── core/
│   ├── include/
│   └── src/
├── bindings/
├── examples/
├── models/
└── docs/
```

---

### Week 3-4: 코어 엔진 개발

#### 목표
- [ ] 홍채 검출 인터페이스 구현
- [ ] MediaPipe 기반 검출기 구현
- [ ] 렌즈 렌더링 로직 구현
- [ ] C API 래퍼 작성

#### 상세 작업

**3.1 인터페이스 정의**
```cpp
// IrisDetector 추상 인터페이스
// - detect() 메서드
// - IrisResult 구조체
```

**3.2 MediaPipe 통합**
```cpp
// MediaPipeIrisDetector 구현
// - MediaPipe 그래프 로드
// - 프레임 처리
// - 결과 변환
```

**3.3 렌즈 렌더링**
```cpp
// LensRenderer 구현
// - 텍스처 로드
// - 홍채 영역에 매핑
// - 블렌딩 처리
```

**3.4 C API**
```cpp
// sdk_api.h / sdk_api.cpp
// - iris_sdk_init()
// - iris_sdk_detect()
// - iris_sdk_render_lens()
// - iris_sdk_destroy()
```

#### 검증 기준
- [ ] 단위 테스트 통과
- [ ] 샘플 이미지로 검출 확인
- [ ] 렌즈 오버레이 시각적 확인

---

### Week 5-6: Android 바인딩 및 데모

#### 목표
- [ ] JNI 바인딩 작성
- [ ] Android 데모 앱 제작
- [ ] 실시간 카메라 연동
- [ ] 성능 측정 및 최적화

#### 상세 작업

**5.1 JNI 바인딩**
```java
// IrisLensSDK.java
public class IrisLensSDK {
    public native int init(String modelPath);
    public native IrisResult detect(byte[] frame, int w, int h);
    public native void renderLens(...);
    public native void destroy();
}
```

**5.2 NDK 빌드 설정**
```cmake
# Android.cmake
# ABI별 빌드 (arm64-v8a, armeabi-v7a, x86_64)
```

**5.3 데모 앱**
```
android-demo/
├── CameraX 연동
├── 프리뷰 화면
├── 렌즈 선택 UI
└── FPS 표시
```

#### 검증 기준
- [ ] 실제 디바이스에서 30fps 이상
- [ ] 검출 지연 33ms 이하
- [ ] 메모리 누수 없음

---

### Week 7-8: iOS 및 Flutter 바인딩

#### 목표
- [ ] iOS Framework 생성
- [ ] Flutter Plugin 생성
- [ ] 각 플랫폼 데모 앱

#### iOS 작업

**7.1 Objective-C++ 래퍼**
```objc
// IrisLensSDK.h / IrisLensSDK.mm
@interface IrisLensSDK : NSObject
- (BOOL)initWithModelPath:(NSString *)path;
- (IrisResult *)detectWithPixelBuffer:(CVPixelBufferRef)buffer;
@end
```

**7.2 XCFramework 생성**
```bash
# 시뮬레이터 + 디바이스 통합 프레임워크
xcodebuild -create-xcframework ...
```

#### Flutter 작업

**7.3 dart:ffi 바인딩**
```dart
// iris_lens_sdk.dart
class IrisLensSDK {
  late DynamicLibrary _lib;
  // FFI 함수 바인딩
}
```

**7.4 Flutter Plugin 구조**
```
iris_lens_sdk/
├── pubspec.yaml
├── lib/iris_lens_sdk.dart
├── android/ (AAR 포함)
└── ios/ (Framework 포함)
```

#### 검증 기준
- [ ] iOS 실기기 동작 확인
- [ ] Flutter 양 플랫폼 동작 확인
- [ ] API 일관성 확인

---

## Phase 2: 하이브리드 시스템 (예상 4-6주)

### 목표
- Eye-Only 커스텀 모델 개발
- 하이브리드 검출 시스템 구축
- MediaPipe 한계 상황 대응

### 작업 내용

**데이터 수집**
```
필요 데이터:
├── 눈 클로즈업 이미지 5,000~10,000장
├── 다양한 각도/조명/인종
└── 홍채 경계 라벨링
```

**모델 학습**
```
아키텍처 후보:
├── U-Net 기반 세그멘테이션
├── HRNet (고해상도)
└── MobileNet 기반 경량 모델
```

**하이브리드 시스템**
```cpp
class HybridIrisDetector : public IrisDetector {
    MediaPipeDetector mediapipe_;
    EyeOnlyDetector eyeOnly_;

    IrisResult detect(Frame frame) {
        auto result = mediapipe_.detect(frame);
        if (result.confidence > threshold) {
            return result;
        }
        return eyeOnly_.detect(frame);
    }
};
```

---

## Phase 3: 최적화 및 확장 (지속적)

### 성능 최적화
- [ ] 모델 양자화 (INT8)
- [ ] GPU 가속 (OpenGL ES / Metal)
- [ ] 멀티스레드 파이프라인

### 기능 확장
- [ ] 다중 렌즈 스타일
- [ ] 렌즈 색상 커스터마이징
- [ ] 눈 깜빡임 처리
- [ ] 조명 반사 시뮬레이션

### 플랫폼 확장
- [ ] Web WASM 지원
- [ ] Desktop (Windows/macOS/Linux)

---

## 마일스톤 요약

| 마일스톤 | 완료 기준 | 예상 시점 |
|----------|----------|----------|
| M1: 환경 구축 | MediaPipe + OpenCV 빌드 성공 | Week 2 |
| M2: 코어 완성 | 이미지 기반 검출 + 렌더링 동작 | Week 4 |
| M3: Android MVP | 실기기 30fps 달성 | Week 6 |
| M4: 크로스플랫폼 | iOS + Flutter 동작 | Week 8 |
| M5: 하이브리드 | Eye-Only 모델 통합 | Phase 2 |

---

## 리스크 및 대응

| 리스크 | 영향 | 대응 방안 |
|--------|------|----------|
| MediaPipe 빌드 복잡성 | 일정 지연 | 사전 빌드된 바이너리 활용 검토 |
| 성능 미달 | 품질 저하 | GPU 가속 우선 적용 |
| 모바일 메모리 제한 | 크래시 | 모델 경량화, 메모리 풀링 |
| Eye-Only 모델 정확도 | 기능 제한 | 데이터 추가 수집, 앙상블 |
