# P1-W4-01: LensRenderer 기본 구현

**태스크 ID**: P1-W4-01
**상태**: ⏳ 대기
**시작일**: -
**완료일**: -

---

## 1. 계획

### 목표
검출된 홍채 위치에 가상 렌즈 텍스처를 오버레이하는 렌더러를 구현한다.

### 산출물
| 파일 | 설명 |
|------|------|
| `cpp/include/iris_sdk/lens_renderer.h` | LensRenderer 클래스 선언 |
| `cpp/src/lens_renderer.cpp` | LensRenderer 구현 |
| `shared/textures/sample_lens.png` | 샘플 렌즈 텍스처 |

### 검증 기준
- [ ] 텍스처 로딩 성공
- [ ] 정적 이미지에 렌즈 오버레이 성공
- [ ] 양눈 독립 렌더링 지원
- [ ] 렌더링 시간 10ms 이하
- [ ] 시각적 품질 확인

### 선행 조건
- P1-W3-02: 데이터 구조 정의 ✅
- P1-W3-03: MediaPipeDetector 구현 (병렬 가능)

---

## 2. 분석

### 2.1 렌더링 파이프라인

```
IrisResult (검출 결과)
    │
    ▼
┌─────────────────┐
│ 좌표 변환       │ 정규화 좌표 → 픽셀 좌표
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 마스크 생성     │ 원형 마스크 (홍채 크기)
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 텍스처 변환     │ 크기 조절, 위치 맞춤
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 블렌딩          │ 알파 블렌딩 + 마스크 적용
└─────────────────┘
    │
    ▼
Output Frame (렌즈 적용된 이미지)
```

### 2.2 클래스 설계

```cpp
#pragma once

#include "iris_sdk/types.h"
#include "iris_sdk/export.h"
#include <memory>
#include <string>

namespace cv { class Mat; }

namespace iris_sdk {

/**
 * @brief 가상 렌즈 렌더러
 *
 * 검출된 홍채 위치에 렌즈 텍스처를 오버레이
 */
class IRIS_SDK_EXPORT LensRenderer {
public:
    LensRenderer();
    ~LensRenderer();

    /**
     * @brief 렌더러 초기화
     * @return 성공 여부
     */
    bool initialize();

    /**
     * @brief 파일에서 렌즈 텍스처 로드
     * @param texture_path 텍스처 이미지 경로
     * @return 성공 여부
     */
    bool loadTexture(const std::string& texture_path);

    /**
     * @brief 메모리에서 렌즈 텍스처 로드
     * @param data RGBA 데이터
     * @param width 너비
     * @param height 높이
     * @return 성공 여부
     */
    bool loadTexture(const uint8_t* data, int width, int height);

    /**
     * @brief 프레임에 렌즈 렌더링
     * @param frame 입출력 이미지 (in-place 수정)
     * @param iris_result 홍채 검출 결과
     * @param config 렌더링 설정
     * @return 성공 여부
     */
    bool render(cv::Mat& frame,
                const IrisResult& iris_result,
                const LensConfig& config);

    /**
     * @brief 왼쪽 눈에만 렌더링
     */
    bool renderLeftEye(cv::Mat& frame,
                       const IrisResult& iris_result,
                       const LensConfig& config);

    /**
     * @brief 오른쪽 눈에만 렌더링
     */
    bool renderRightEye(cv::Mat& frame,
                        const IrisResult& iris_result,
                        const LensConfig& config);

    /**
     * @brief 텍스처 언로드
     */
    void unloadTexture();

    /**
     * @brief 리소스 해제
     */
    void release();

    /**
     * @brief 텍스처 로드 상태 확인
     */
    bool hasTexture() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace iris_sdk
```

### 2.3 핵심 렌더링 로직

**2.3.1 좌표 변환**
```cpp
cv::Point2f normalizedToPixel(const IrisLandmark& landmark,
                              int frame_width, int frame_height) {
    return cv::Point2f(
        landmark.x * frame_width,
        landmark.y * frame_height
    );
}
```

**2.3.2 홍채 영역 계산**
```cpp
struct IrisRegion {
    cv::Point2f center;
    float radius;
    float angle;  // 회전 각도 (눈 기울기)
};

IrisRegion calculateIrisRegion(const IrisLandmark iris[5],
                                int frame_width, int frame_height) {
    IrisRegion region;

    // 중심점 (landmark[0])
    region.center = normalizedToPixel(iris[0], frame_width, frame_height);

    // 반지름 (landmark[1,2,3,4]의 평균 거리)
    float sum_radius = 0;
    for (int i = 1; i <= 4; ++i) {
        cv::Point2f pt = normalizedToPixel(iris[i], frame_width, frame_height);
        sum_radius += cv::norm(pt - region.center);
    }
    region.radius = sum_radius / 4.0f;

    // 회전 각도 (수평 랜드마크로 계산)
    cv::Point2f left = normalizedToPixel(iris[3], frame_width, frame_height);
    cv::Point2f right = normalizedToPixel(iris[4], frame_width, frame_height);
    region.angle = std::atan2(right.y - left.y, right.x - left.x) * 180.0f / CV_PI;

    return region;
}
```

**2.3.3 텍스처 변환**
```cpp
cv::Mat transformTexture(const cv::Mat& texture,
                         const IrisRegion& region,
                         const LensConfig& config) {
    // 목표 크기 계산
    int target_size = static_cast<int>(region.radius * 2 * config.scale);

    // 리사이즈
    cv::Mat resized;
    cv::resize(texture, resized, cv::Size(target_size, target_size),
               0, 0, cv::INTER_LINEAR);

    // 회전 적용
    cv::Mat rotated;
    cv::Point2f center(resized.cols / 2.0f, resized.rows / 2.0f);
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, region.angle, 1.0);
    cv::warpAffine(resized, rotated, rot_mat, resized.size(),
                   cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);

    return rotated;
}
```

**2.3.4 알파 블렌딩**
```cpp
void alphaBlend(cv::Mat& dst, const cv::Mat& src, const cv::Mat& mask,
                cv::Point2i position, float opacity) {
    // ROI 계산 (경계 처리)
    cv::Rect src_rect(0, 0, src.cols, src.rows);
    cv::Rect dst_rect(position.x - src.cols/2, position.y - src.rows/2,
                      src.cols, src.rows);

    // 프레임 경계 클리핑
    cv::Rect frame_rect(0, 0, dst.cols, dst.rows);
    cv::Rect valid_dst = dst_rect & frame_rect;
    if (valid_dst.empty()) return;

    // 소스 영역 조정
    cv::Rect valid_src(
        valid_dst.x - dst_rect.x,
        valid_dst.y - dst_rect.y,
        valid_dst.width,
        valid_dst.height
    );

    // 블렌딩
    cv::Mat dst_roi = dst(valid_dst);
    cv::Mat src_roi = src(valid_src);
    cv::Mat mask_roi = mask(valid_src);

    for (int y = 0; y < dst_roi.rows; ++y) {
        for (int x = 0; x < dst_roi.cols; ++x) {
            float alpha = mask_roi.at<float>(y, x) * opacity;
            if (alpha > 0.01f) {
                cv::Vec3b& d = dst_roi.at<cv::Vec3b>(y, x);
                cv::Vec3b s = src_roi.at<cv::Vec3b>(y, x);
                d = cv::Vec3b(
                    static_cast<uint8_t>(d[0] * (1 - alpha) + s[0] * alpha),
                    static_cast<uint8_t>(d[1] * (1 - alpha) + s[1] * alpha),
                    static_cast<uint8_t>(d[2] * (1 - alpha) + s[2] * alpha)
                );
            }
        }
    }
}
```

### 2.4 텍스처 요구사항

| 항목 | 요구사항 |
|------|----------|
| 포맷 | PNG (알파 채널 포함) |
| 크기 | 256x256 ~ 512x512 권장 |
| 형태 | 원형, 중심 정렬 |
| 알파 | 경계 그라데이션 권장 |

### 2.5 성능 최적화

| 전략 | 설명 |
|------|------|
| 텍스처 캐싱 | 미리 변환된 텍스처 저장 |
| ROI 처리 | 눈 영역만 렌더링 |
| SIMD 블렌딩 | OpenCV 최적화 활용 |
| 룩업 테이블 | 알파 계산 미리 수행 |

---

## 3. 실행 내역

### 3.1 헤더 파일 작성

```bash
# 예정: cpp/include/iris_sdk/lens_renderer.h
```

### 3.2 구현 파일 작성

```bash
# 예정: cpp/src/lens_renderer.cpp
```

### 3.3 샘플 텍스처 생성

```bash
# 예정: shared/textures/sample_lens.png
```

### 3.4 테스트 작성

```bash
# 예정: cpp/tests/test_lens_renderer.cpp
```

---

## 4. 검증 결과

### 검증 항목

| 항목 | 결과 | 비고 |
|------|------|------|
| 텍스처 로딩 | ⏳ 대기 | PNG, JPEG |
| 좌표 변환 | ⏳ 대기 | |
| 렌더링 정확성 | ⏳ 대기 | |
| 양눈 렌더링 | ⏳ 대기 | |
| 렌더링 시간 | ⏳ 대기 | 목표: 10ms |
| 시각적 품질 | ⏳ 대기 | |

---

## 5. 이슈 및 학습

### 이슈

| ID | 내용 | 상태 | 해결방안 |
|----|------|------|----------|
| - | - | - | - |

### 결정 사항

| 결정 | 이유 |
|------|------|
| in-place 렌더링 | 메모리 복사 최소화 |
| Pimpl 패턴 | OpenCV 헤더 의존성 분리 |
| 원형 마스크 | 홍채 형태에 가장 적합 |

### 학습 내용

(실행 후 기록)

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-01-07 | 태스크 문서 생성, 렌더링 파이프라인 설계 완료 |
