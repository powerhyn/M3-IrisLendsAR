# P1-W3-03: MediaPipeDetector 구현

**태스크 ID**: P1-W3-03
**상태**: ⏳ 대기
**시작일**: -
**완료일**: -

---

## 1. 계획

### 목표
MediaPipe Face Mesh 및 Iris 모델을 사용하여 IrisDetector 인터페이스의 첫 번째 구현체를 완성한다.

### 산출물
| 파일 | 설명 |
|------|------|
| `cpp/include/iris_sdk/mediapipe_detector.h` | MediaPipeDetector 클래스 선언 |
| `cpp/src/mediapipe_detector.cpp` | MediaPipeDetector 구현 |
| `shared/models/*.tflite` | 필요한 TFLite 모델 파일 |

### 검증 기준
- [ ] 초기화 성공 (모델 로드)
- [ ] 정적 이미지에서 홍채 검출 성공
- [ ] 검출 정확도 95% 이상 (정상 조건)
- [ ] 검출 시간 33ms 이하 (Desktop)
- [ ] 메모리 사용량 합리적 (100MB 이하)

### 선행 조건
- P1-W3-02: 데이터 구조 정의 ✅

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

### 2.2 클래스 설계

```cpp
#pragma once

#include "iris_sdk/iris_detector.h"
#include <memory>

// Forward declarations
namespace tflite {
    class FlatBufferModel;
    class Interpreter;
}

namespace iris_sdk {

/**
 * @brief MediaPipe 기반 홍채 검출기
 *
 * TensorFlow Lite를 사용하여 MediaPipe의 Face Mesh 및
 * Iris Landmark 모델을 실행
 */
class IRIS_SDK_EXPORT MediaPipeDetector : public IrisDetector {
public:
    MediaPipeDetector();
    ~MediaPipeDetector() override;

    // IrisDetector 인터페이스 구현
    bool initialize(const std::string& model_path) override;
    IrisResult detect(const cv::Mat& frame) override;
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

### 2.3 핵심 구현 로직

**1. 모델 로드**
```cpp
bool MediaPipeDetector::Impl::loadModels(const std::string& model_dir) {
    // Face detection 모델
    face_detection_model_ = tflite::FlatBufferModel::BuildFromFile(
        (model_dir + "/face_detection_short_range.tflite").c_str());

    // Face landmark 모델
    face_landmark_model_ = tflite::FlatBufferModel::BuildFromFile(
        (model_dir + "/face_landmark.tflite").c_str());

    // Iris landmark 모델
    iris_landmark_model_ = tflite::FlatBufferModel::BuildFromFile(
        (model_dir + "/iris_landmark.tflite").c_str());

    // 인터프리터 생성 및 텐서 할당
    // ...
}
```

**2. 얼굴 검출**
```cpp
std::vector<Rect> detectFaces(const cv::Mat& frame) {
    // 입력 전처리 (128x128 리사이즈, 정규화)
    cv::Mat input;
    cv::resize(frame, input, cv::Size(128, 128));
    input.convertTo(input, CV_32FC3, 1.0/255.0);

    // 추론 실행
    face_detection_interpreter_->Invoke();

    // 결과 파싱 (바운딩 박스 + 신뢰도)
    // ...
}
```

**3. 홍채 랜드마크 추출**
```cpp
void extractIrisLandmarks(const cv::Mat& eye_region,
                          IrisLandmark landmarks[5]) {
    // Eye region 전처리 (64x64)
    cv::Mat input;
    cv::resize(eye_region, input, cv::Size(64, 64));

    // Iris 모델 추론
    iris_interpreter_->Invoke();

    // 5개 랜드마크 추출:
    // [0]: 홍채 중심
    // [1]: 상단
    // [2]: 하단
    // [3]: 좌측
    // [4]: 우측
}
```

### 2.4 성능 최적화 전략

| 전략 | 설명 | 예상 효과 |
|------|------|----------|
| GPU Delegate | TFLite GPU 가속 | 2-3x 속도 향상 |
| XNNPACK Delegate | CPU SIMD 최적화 | 1.5-2x 속도 향상 |
| 입력 크기 조정 | 해상도 동적 조절 | 품질/속도 트레이드오프 |
| 얼굴 추적 | 검출 스킵 (추적 모드) | 프레임 당 연산 감소 |

### 2.5 의존성

| 라이브러리 | 용도 | 필수 |
|------------|------|------|
| TensorFlow Lite | 모델 추론 | Yes |
| OpenCV | 이미지 전처리 | Yes |
| XNNPACK | CPU 가속 | Optional |
| GPU Delegate | GPU 가속 | Optional |

---

## 3. 실행 내역

### 3.1 모델 파일 다운로드

```bash
# 예정: MediaPipe 모델 다운로드 스크립트 실행
# ./scripts/download_models.sh
```

### 3.2 헤더 파일 작성

```bash
# 예정: cpp/include/iris_sdk/mediapipe_detector.h
```

### 3.3 구현 파일 작성

```bash
# 예정: cpp/src/mediapipe_detector.cpp
```

### 3.4 빌드 및 테스트

```bash
# 예정: cmake --build . && ctest
```

---

## 4. 검증 결과

### 검증 항목

| 항목 | 결과 | 비고 |
|------|------|------|
| 모델 로드 | ⏳ 대기 | 3개 모델 |
| 얼굴 검출 | ⏳ 대기 | |
| 랜드마크 추출 | ⏳ 대기 | 468점 |
| 홍채 검출 | ⏳ 대기 | 5점 × 2눈 |
| 검출 시간 | ⏳ 대기 | 목표: 33ms 이하 |
| 정확도 | ⏳ 대기 | 목표: 95%+ |

### 테스트 이미지

| 이미지 | 조건 | 예상 결과 |
|--------|------|----------|
| frontal_face.jpg | 정면 | 검출 성공 |
| side_face_30.jpg | 30도 측면 | 검출 성공 |
| side_face_45.jpg | 45도 측면 | 검출 성공 (불안정) |
| closeup_eyes.jpg | 눈 클로즈업 | 검출 실패 (MediaPipe 한계) |
| multiple_faces.jpg | 다중 얼굴 | 첫 번째 얼굴만 |

---

## 5. 이슈 및 학습

### 이슈

| ID | 내용 | 상태 | 해결방안 |
|----|------|------|----------|
| - | - | - | - |

### 결정 사항

| 결정 | 이유 |
|------|------|
| TFLite 직접 사용 | MediaPipe C++ Bazel 빌드 복잡성 회피 |
| Pimpl 패턴 | 컴파일 의존성 분리, 바이너리 호환성 |
| 단일 얼굴 처리 | Phase 1 범위, 성능 우선 |

### 학습 내용

(실행 후 기록)

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-01-07 | 태스크 문서 생성, 구현 설계 완료 |
