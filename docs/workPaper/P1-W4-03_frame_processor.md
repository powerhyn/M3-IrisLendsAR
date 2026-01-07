# P1-W4-03: FrameProcessor 파이프라인

**태스크 ID**: P1-W4-03
**상태**: ⏳ 대기
**시작일**: -
**완료일**: -

---

## 1. 계획

### 목표
검출과 렌더링을 통합하는 프레임 처리 파이프라인을 구현한다. 단일 호출로 전체 처리가 가능하도록 한다.

### 산출물
| 파일 | 설명 |
|------|------|
| `cpp/include/iris_sdk/frame_processor.h` | FrameProcessor 클래스 선언 |
| `cpp/src/frame_processor.cpp` | FrameProcessor 구현 |

### 검증 기준
- [ ] 단일 호출로 검출 + 렌더링 처리
- [ ] 다양한 프레임 포맷 지원 (RGBA, BGR, NV21, NV12)
- [ ] 파이프라인 처리 시간 33ms 이하
- [ ] 메모리 효율적 처리 (불필요한 복사 최소화)

### 선행 조건
- P1-W3-04: 검출기 단위 테스트 ✅
- P1-W4-02: 블렌딩 알고리즘 구현 ✅

---

## 2. 분석

### 2.1 파이프라인 구조

```
Input Frame (다양한 포맷)
    │
    ▼
┌─────────────────────────────────────────┐
│            FrameProcessor               │
│                                         │
│  ┌──────────────┐                       │
│  │ Format Conv. │ NV21/RGBA → BGR       │
│  └──────────────┘                       │
│         │                               │
│         ▼                               │
│  ┌──────────────┐                       │
│  │ IrisDetector │ 홍채 검출             │
│  └──────────────┘                       │
│         │                               │
│         ▼                               │
│  ┌──────────────┐                       │
│  │LensRenderer  │ 렌즈 오버레이         │
│  └──────────────┘                       │
│         │                               │
│         ▼                               │
│  ┌──────────────┐                       │
│  │ Format Conv. │ BGR → 원본 포맷       │
│  └──────────────┘                       │
└─────────────────────────────────────────┘
    │
    ▼
Output Frame + IrisResult
```

### 2.2 클래스 설계

```cpp
#pragma once

#include "iris_sdk/types.h"
#include "iris_sdk/iris_detector.h"
#include "iris_sdk/lens_renderer.h"
#include "iris_sdk/export.h"
#include <memory>
#include <functional>

namespace iris_sdk {

/**
 * @brief 프레임 처리 결과
 */
struct ProcessResult {
    bool success;           ///< 처리 성공 여부
    IrisResult iris_result; ///< 홍채 검출 결과
    ErrorCode error_code;   ///< 에러 코드 (실패 시)
    float processing_time_ms; ///< 처리 시간 (밀리초)
};

/**
 * @brief 프레임 처리 콜백
 */
using ProcessCallback = std::function<void(const ProcessResult&)>;

/**
 * @brief 프레임 처리 파이프라인
 *
 * 검출과 렌더링을 통합하여 단일 인터페이스로 제공
 */
class IRIS_SDK_EXPORT FrameProcessor {
public:
    FrameProcessor();
    ~FrameProcessor();

    /**
     * @brief 프로세서 초기화
     * @param model_path 모델 파일 경로
     * @param detector_type 검출기 종류
     * @return 성공 여부
     */
    bool initialize(const std::string& model_path,
                    DetectorType detector_type = DetectorType::MediaPipe);

    /**
     * @brief 프레임 처리 (동기)
     * @param frame_data 프레임 데이터
     * @param width 프레임 너비
     * @param height 프레임 높이
     * @param format 픽셀 포맷
     * @param config 렌더링 설정 (nullptr이면 검출만)
     * @return 처리 결과
     */
    ProcessResult process(uint8_t* frame_data,
                          int width,
                          int height,
                          FrameFormat format,
                          const LensConfig* config = nullptr);

    /**
     * @brief cv::Mat 프레임 처리 (동기)
     */
    ProcessResult process(cv::Mat& frame,
                          const LensConfig* config = nullptr);

    /**
     * @brief 프레임 처리 (비동기)
     * @param callback 완료 콜백
     */
    void processAsync(uint8_t* frame_data,
                      int width,
                      int height,
                      FrameFormat format,
                      const LensConfig* config,
                      ProcessCallback callback);

    /**
     * @brief 검출만 수행
     */
    IrisResult detectOnly(const uint8_t* frame_data,
                          int width,
                          int height,
                          FrameFormat format);

    /**
     * @brief 렌더링만 수행
     */
    bool renderOnly(uint8_t* frame_data,
                    int width,
                    int height,
                    FrameFormat format,
                    const IrisResult& iris_result,
                    const LensConfig& config);

    /**
     * @brief 렌즈 텍스처 로드
     */
    bool loadLensTexture(const std::string& texture_path);

    /**
     * @brief 렌즈 텍스처 로드 (메모리)
     */
    bool loadLensTexture(const uint8_t* data, int width, int height);

    /**
     * @brief 설정 변경
     */
    void setDetectionConfig(float min_confidence, bool track_faces = true);

    /**
     * @brief 리소스 해제
     */
    void release();

    /**
     * @brief 초기화 상태 확인
     */
    bool isInitialized() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace iris_sdk
```

### 2.3 프레임 포맷 변환

**2.3.1 NV21 → BGR (Android Camera)**
```cpp
cv::Mat convertNV21toBGR(const uint8_t* nv21_data, int width, int height) {
    cv::Mat nv21(height + height / 2, width, CV_8UC1, const_cast<uint8_t*>(nv21_data));
    cv::Mat bgr;
    cv::cvtColor(nv21, bgr, cv::COLOR_YUV2BGR_NV21);
    return bgr;
}
```

**2.3.2 NV12 → BGR (iOS Camera)**
```cpp
cv::Mat convertNV12toBGR(const uint8_t* nv12_data, int width, int height) {
    // NV12: Y plane 후 UV interleaved (U first, V second)
    // NV21과 UV 순서만 다름
    cv::Mat nv12(height + height / 2, width, CV_8UC1, const_cast<uint8_t*>(nv12_data));
    cv::Mat bgr;
    cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_NV12);
    return bgr;
}
```

**2.3.3 RGBA → BGR (일반적인 모바일)**
```cpp
cv::Mat convertRGBAtoBGR(const uint8_t* rgba_data, int width, int height) {
    cv::Mat rgba(height, width, CV_8UC4, const_cast<uint8_t*>(rgba_data));
    cv::Mat bgr;
    cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
    return bgr;
}
```

**2.3.4 BGR → 원본 포맷**
```cpp
void convertBGRtoFormat(const cv::Mat& bgr, uint8_t* output_data,
                        int width, int height, FrameFormat format) {
    cv::Mat output;
    switch (format) {
        case FrameFormat::RGBA:
            cv::cvtColor(bgr, output, cv::COLOR_BGR2RGBA);
            break;
        case FrameFormat::NV21:
            cv::cvtColor(bgr, output, cv::COLOR_BGR2YUV_YV12);
            // NV21 재배열 필요 (UV plane swap)
            rearrangeYV12toNV21(output, output_data, width, height);
            return;
        case FrameFormat::NV12:
            cv::cvtColor(bgr, output, cv::COLOR_BGR2YUV_I420);
            // NV12 재배열 필요 (UV interleave)
            rearrangeI420toNV12(output, output_data, width, height);
            return;
        case FrameFormat::BGR:
            output = bgr;
            break;
        // ... 기타 포맷
    }

    if (output.isContinuous()) {
        std::memcpy(output_data, output.data, output.total() * output.elemSize());
    }
}

// I420 → NV12 재배열 헬퍼
void rearrangeI420toNV12(const cv::Mat& i420, uint8_t* nv12_data, int width, int height) {
    const int y_size = width * height;
    const int uv_size = y_size / 4;

    // Y plane 복사
    std::memcpy(nv12_data, i420.data, y_size);

    // U, V → UV interleaved
    const uint8_t* u_plane = i420.data + y_size;
    const uint8_t* v_plane = i420.data + y_size + uv_size;
    uint8_t* uv_plane = nv12_data + y_size;

    for (int i = 0; i < uv_size; ++i) {
        uv_plane[2 * i] = u_plane[i];      // U
        uv_plane[2 * i + 1] = v_plane[i];  // V
    }
}
```

### 2.4 파이프라인 구현

```cpp
ProcessResult FrameProcessor::Impl::process(uint8_t* frame_data,
                                            int width, int height,
                                            FrameFormat format,
                                            const LensConfig* config) {
    ProcessResult result;
    auto start_time = std::chrono::high_resolution_clock::now();

    // 1. 포맷 변환 (필요시)
    cv::Mat frame = convertToWorkingFormat(frame_data, width, height, format);
    if (frame.empty()) {
        result.success = false;
        result.error_code = ErrorCode::FrameFormatUnsupported;
        return result;
    }

    // 2. 홍채 검출
    result.iris_result = detector_->detect(frame);

    // 3. 렌더링 (설정이 있고 검출 성공 시)
    if (config != nullptr && result.iris_result.detected && renderer_->hasTexture()) {
        renderer_->render(frame, result.iris_result, *config);
    }

    // 4. 원본 포맷으로 변환 (필요시)
    if (format != FrameFormat::BGR && config != nullptr) {
        convertFromWorkingFormat(frame, frame_data, width, height, format);
    }

    // 5. 결과 설정
    auto end_time = std::chrono::high_resolution_clock::now();
    result.success = true;
    result.error_code = ErrorCode::Success;
    result.processing_time_ms = std::chrono::duration<float, std::milli>(
        end_time - start_time).count();

    return result;
}
```

### 2.5 메모리 최적화

**2.5.1 Zero-copy 처리 (BGR 입력)**
```cpp
// BGR 입력은 복사 없이 직접 처리
cv::Mat frame(height, width, CV_8UC3, frame_data);  // 데이터 공유
```

**2.5.2 버퍼 재사용**
```cpp
class FrameProcessor::Impl {
    // 버퍼 풀 (재사용)
    cv::Mat work_buffer_;
    cv::Mat temp_buffer_;

    cv::Mat& getWorkBuffer(int width, int height, int type) {
        if (work_buffer_.rows != height || work_buffer_.cols != width ||
            work_buffer_.type() != type) {
            work_buffer_.create(height, width, type);
        }
        return work_buffer_;
    }
};
```

### 2.6 성능 측정 포인트

| 단계 | 예상 시간 | 측정 방법 |
|------|----------|----------|
| 포맷 변환 | 2-3ms | RGBA→BGR |
| 검출 | 20-25ms | MediaPipe 추론 |
| 렌더링 | 3-5ms | 알파 블렌딩 |
| 포맷 복원 | 2-3ms | BGR→RGBA |
| **총합** | **~33ms** | **30fps 목표** |

---

## 3. 실행 내역

### 3.1 헤더 파일 작성

```bash
# 예정: cpp/include/iris_sdk/frame_processor.h
```

### 3.2 구현 파일 작성

```bash
# 예정: cpp/src/frame_processor.cpp
```

### 3.3 포맷 변환 테스트

```bash
# 예정: cpp/tests/test_frame_processor.cpp
```

### 3.4 성능 측정

```bash
# 예정: 각 단계별 시간 측정 및 최적화
```

---

## 4. 검증 결과

### 검증 항목

| 항목 | 결과 | 비고 |
|------|------|------|
| 통합 처리 | ⏳ 대기 | 검출 + 렌더링 |
| BGR 포맷 | ⏳ 대기 | Zero-copy |
| RGBA 포맷 | ⏳ 대기 | 변환 필요 |
| NV21 포맷 | ⏳ 대기 | Android |
| NV12 포맷 | ⏳ 대기 | iOS |
| 처리 시간 | ⏳ 대기 | 목표: 33ms |
| 메모리 사용 | ⏳ 대기 | 버퍼 재사용 |

### 성능 측정 결과

| 단계 | 실측 시간 | 목표 | 결과 |
|------|----------|------|------|
| 포맷 변환 | - | 3ms | ⏳ |
| 검출 | - | 25ms | ⏳ |
| 렌더링 | - | 5ms | ⏳ |
| 포맷 복원 | - | 3ms | ⏳ |
| **총합** | - | 33ms | ⏳ |

---

## 5. 이슈 및 학습

### 이슈

| ID | 내용 | 상태 | 해결방안 |
|----|------|------|----------|
| - | - | - | - |

### 결정 사항

| 결정 | 이유 |
|------|------|
| BGR 작업 포맷 | OpenCV 기본 포맷, 변환 최소화 |
| 버퍼 재사용 | 메모리 할당 오버헤드 감소 |
| 동기/비동기 분리 | 사용 사례별 유연성 |

### 학습 내용

(실행 후 기록)

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-01-07 | 태스크 문서 생성, 파이프라인 설계 완료 |
| 2026-01-07 | 아키텍처 리뷰: NV12 포맷 변환 로직 추가 (iOS 지원) |
