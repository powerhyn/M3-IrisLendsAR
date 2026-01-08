# P1-W3-03: MediaPipeDetector 구현

**태스크 ID**: P1-W3-03
**상태**: ✅ 완료 (Phase 1 - TDD 기반 인터페이스 구현)
**시작일**: 2026-01-08
**완료일**: 2026-01-08

---

## 1. 계획

### 목표
MediaPipe Face Mesh 및 Iris 모델을 사용하여 IrisDetector 인터페이스의 첫 번째 구현체를 완성한다.

### 산출물
| 파일 | 설명 | 상태 |
|------|------|------|
| `cpp/include/iris_sdk/mediapipe_detector.h` | MediaPipeDetector 클래스 선언 | ✅ |
| `cpp/src/mediapipe_detector.cpp` | MediaPipeDetector 구현 | ✅ |
| `cpp/tests/test_mediapipe_detector.cpp` | 단위 테스트 (20개) | ✅ |
| `shared/models/*.tflite` | TFLite 모델 파일 | ⏳ Phase 2 |

### 검증 기준
- [x] 클래스 구조 및 인터페이스 정의
- [x] 초기화 검증 로직 (경로, 모델 파일)
- [x] 입력 검증 (null, 크기)
- [x] 상태 관리 (중복 초기화 방지)
- [ ] 정적 이미지에서 홍채 검출 성공 (TFLite 필요)
- [ ] 검출 정확도 95% 이상 (TFLite 필요)
- [ ] 검출 시간 33ms 이하 (TFLite 필요)

### 선행 조건
- P1-W3-02: 데이터 구조 정의 ✅
- OpenCV 4.13.0 설치 ✅

---

## 2. 분석

### 2.1 MediaPipe 모델 구성

**필요한 모델 파일**:
| 모델 | 용도 | 크기 |
|------|------|------|
| `face_detection_short_range.tflite` | 얼굴 검출 (근거리) | ~200KB |
| `face_landmark.tflite` | 얼굴 468 랜드마크 | ~2MB |
| `iris_landmark.tflite` | 홍채 5 랜드마크 | ~100KB |

**추론 파이프라인**:
```
Input Frame
    │
    ▼
Face Detection (BlazeFace)
    │ 얼굴 바운딩 박스
    ▼
Face Landmark (468 points)
    │ 눈 영역 추출
    ▼
Iris Landmark (5 points × 2 eyes)
    │
    ▼
IrisResult 구조체
```

### 2.2 클래스 설계 (구현됨)

```cpp
#pragma once

#include "iris_sdk/iris_detector.h"
#include <memory>

namespace iris_sdk {

class IRIS_SDK_EXPORT MediaPipeDetector : public IrisDetector {
public:
    MediaPipeDetector();
    ~MediaPipeDetector() override;

    // 복사/이동 금지 (Pimpl 사용)
    MediaPipeDetector(const MediaPipeDetector&) = delete;
    MediaPipeDetector& operator=(const MediaPipeDetector&) = delete;
    MediaPipeDetector(MediaPipeDetector&&) = delete;
    MediaPipeDetector& operator=(MediaPipeDetector&&) = delete;

    // IrisDetector 인터페이스 구현
    bool initialize(const std::string& model_path) override;
    IrisResult detect(const uint8_t* frame_data,
                      int width, int height,
                      FrameFormat format) override;
    void release() override;
    bool isInitialized() const override;
    DetectorType getDetectorType() const override;

    // MediaPipe 전용 설정
    void setMinDetectionConfidence(float confidence);
    void setMinTrackingConfidence(float confidence);
    void setNumFaces(int num_faces);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace iris_sdk
```

### 2.3 의존성

| 라이브러리 | 용도 | 상태 |
|------------|------|------|
| OpenCV 4.13.0 | 이미지 전처리 | ✅ 설치됨 |
| TensorFlow Lite | 모델 추론 | ⏳ Phase 2 |
| XNNPACK | CPU 가속 | ⏳ Optional |
| GPU Delegate | GPU 가속 | ⏳ Optional |

---

## 3. 실행 내역

### 3.1 TDD 워크플로우 적용

#### 🔴 RED Phase (2026-01-08)
- 20개 테스트 케이스 작성
- 최소 스텁 구현 (모든 기능 실패 반환)
- 커밋: `a123ed9`

**테스트 케이스 목록**:
| 카테고리 | 테스트 | 설명 |
|----------|--------|------|
| 클래스 특성 | InheritsFromIrisDetector | IrisDetector 상속 확인 |
| | IsNotAbstract | 인스턴스화 가능 |
| | HasVirtualDestructor | 다형성 안전 소멸 |
| 생성/소멸 | DefaultConstruction | 기본 생성자 |
| | PointerConstruction | unique_ptr 생성 |
| | PolymorphicCreation | 다형성 생성 |
| 초기화 | InitializeWithInvalidPath | 존재하지 않는 경로 |
| | InitializeWithEmptyPath | 빈 경로 |
| | InitializeWithValidPath | 모델 없는 유효 경로 |
| | DoubleInitializationFails | 중복 초기화 방지 |
| 타입 | GetDetectorTypeReturnsMediaPipe | DetectorType::MediaPipe |
| 검출 | DetectWithoutInitialization | 미초기화 시 빈 결과 |
| | DetectWithNullFrame | null 프레임 처리 |
| | DetectWithInvalidDimensions | 잘못된 크기 처리 |
| 해제 | ReleaseBeforeInitialization | 초기화 전 해제 안전 |
| | ReleaseAfterInitialization | 정상 해제 |
| | DoubleRelease | 중복 해제 안전 |
| 설정 | SetMinDetectionConfidence | 신뢰도 설정 |
| | SetMinTrackingConfidence | 추적 신뢰도 설정 |
| | SetNumFaces | 얼굴 수 설정 |

#### 🟢 GREEN Phase (2026-01-08)
- 경로 검증 로직 추가 (std::filesystem)
- 상태 관리 구현 (중복 초기화 방지)
- 입력 검증 강화 (nullptr, 크기)
- 설정 값 클램핑 (0.0~1.0, 1 이상)
- 커밋: `3697b3f`

#### 🔄 REFACTOR Phase (2026-01-08)
- std::clamp 적용 (C++17)
- 불필요한 헤더 제거
- 커밋: `81bc708`

### 3.2 Git 커밋 히스토리

```
81bc708 refactor(mediapipe): MediaPipeDetector TDD REFACTOR - 코드 품질 개선
3697b3f feat(mediapipe): MediaPipeDetector TDD GREEN 단계 - 최소 구현
a123ed9 test(mediapipe): MediaPipeDetector TDD RED 단계 - 테스트 및 스텁 작성
```

### 3.3 빌드 및 테스트

```bash
# 빌드
cd cpp/cmake-build-debug
cmake --build . --target test_mediapipe_detector

# 테스트 실행
./bin/test_mediapipe_detector
# 결과: 20 tests PASSED

# 전체 테스트
ctest --output-on-failure
# 결과: 71 tests passed, 0 tests failed
```

---

## 4. 검증 결과

### 테스트 결과

| 테스트 스위트 | 테스트 수 | 결과 |
|--------------|----------|------|
| IrisLandmarkTest | 6 | ✅ PASSED |
| RectTest | 5 | ✅ PASSED |
| IrisResultTest | 10 | ✅ PASSED |
| LensConfigTest | 4 | ✅ PASSED |
| BlendModeTest | 1 | ✅ PASSED |
| FrameFormatTest | 1 | ✅ PASSED |
| ErrorCodeTest | 6 | ✅ PASSED |
| DetectorTypeTest | 1 | ✅ PASSED |
| IrisDetectorTest | 17 | ✅ PASSED |
| MediaPipeDetectorTest | 20 | ✅ PASSED |
| **총계** | **71** | **100% PASSED** |

### 구현 검증

| 항목 | 결과 | 비고 |
|------|------|------|
| 클래스 구조 | ✅ 완료 | Pimpl 패턴 적용 |
| 경로 검증 | ✅ 완료 | std::filesystem 사용 |
| 상태 관리 | ✅ 완료 | 중복 초기화 방지 |
| 입력 검증 | ✅ 완료 | nullptr, 크기 검사 |
| 설정 범위 | ✅ 완료 | std::clamp 적용 |
| 모델 로드 | ⏳ 대기 | TFLite 필요 |
| 추론 실행 | ⏳ 대기 | TFLite 필요 |

---

## 5. 이슈 및 학습

### 이슈

| ID | 내용 | 상태 | 해결방안 |
|----|------|------|----------|
| #1 | TensorFlow Lite Homebrew 미지원 | 📌 오픈 | CMake FetchContent 또는 수동 빌드 |
| #2 | CLion 테스트 트리 표시 안됨 | 📌 오픈 | 터미널 기반 테스트로 진행 |
| #3 | gtest_discover_tests 타임아웃 | ✅ 해결 | DISCOVERY_TIMEOUT 60 추가 |

### 결정 사항

| 결정 | 이유 |
|------|------|
| TDD 워크플로우 적용 | 안정적인 인터페이스 설계 보장 |
| std::filesystem 사용 | C++17 표준, 크로스플랫폼 호환 |
| Pimpl 패턴 유지 | 컴파일 의존성 분리, ABI 안정성 |
| 모델 검증 지연 | TFLite 없이 인터페이스 먼저 확정 |

### 학습 내용

1. **TDD RED-GREEN-REFACTOR**: 인터페이스 설계 품질 향상에 효과적
2. **std::clamp (C++17)**: min/max 중첩보다 가독성 우수
3. **gtest_discover_tests**: 빌드 시 타임아웃 설정 필요
4. **Homebrew TFLite**: 미지원, 별도 빌드 필요

---

## 6. 다음 단계

### Phase 2 작업
1. ~~TensorFlow Lite 빌드 및 통합~~ ✅ CMake FetchContent 설정 완료
2. ~~모델 파일 다운로드 스크립트 작성~~ ✅ 3개 모델 다운로드 완료
3. 실제 추론 파이프라인 구현 ⏳
4. 성능 측정 및 최적화 ⏳

### 완료된 리소스
| 리소스 | 상태 | 경로/비고 |
|--------|------|----------|
| TensorFlow Lite | ✅ 설정됨 | CMake FetchContent (-DIRIS_SDK_FETCH_TFLITE=ON) |
| face_detection_short_range.tflite | ✅ 225KB | `shared/models/` |
| face_landmark.tflite | ✅ 1.2MB | `shared/models/` |
| iris_landmark.tflite | ✅ 2.5MB | `shared/models/` |
| 테스트용 얼굴 이미지 | ⏳ 대기 | |

### 다음 구현 항목
1. TFLite 인터프리터 초기화 (`MediaPipeDetector::Impl`)
2. 모델 로딩 로직 구현
3. 추론 파이프라인 구현 (Face Detection → Face Mesh → Iris)
4. IrisResult 구조체에 결과 매핑

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-01-07 | 태스크 문서 생성, 구현 설계 완료 |
| 2026-01-08 | TDD 기반 Phase 1 구현 완료 (71개 테스트 통과) |
| 2026-01-08 | TFLite FetchContent 설정, MediaPipe 모델 3개 다운로드 완료 |
