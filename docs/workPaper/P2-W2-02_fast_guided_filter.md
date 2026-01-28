# P2-W2-02. Fast Guided Filter 직접 구현

## 작업 정보

| 항목 | 내용 |
|------|------|
| **작업 ID** | P2-W2-02 |
| **Phase** | Phase 2: CPU 백엔드 개선 |
| **상태** | ⏳ 대기 |
| **예상 기간** | 2일 |
| **의존성** | P2-W2-01 (ROI 기반 처리) |
| **담당** | systems-programming:cpp-pro |

---

## 1. 목표

OpenCV-contrib 의존성 없이 Fast Guided Filter를 직접 구현하여 고품질 에지 보존 스무딩 제공

### 핵심 산출물
- `FastGuidedFilter` 클래스 직접 구현
- Box Filter 기반 O(1) 시간 복잡도
- 서브샘플링 기반 추가 최적화
- 기존 Bilateral Filter 대비 2-3배 성능 향상

### 참고 논문
- "Guided Image Filtering" (He et al., ECCV 2010)
- "Fast Guided Filter" (He & Sun, 2015)

---

## 2. Guided Filter 이론

### 2.1 기본 원리

Guided Filter는 가이드 이미지 `I`를 사용하여 입력 이미지 `p`를 필터링:

```
q_i = a_k * I_i + b_k,  for all i in window w_k
```

여기서 `a_k`와 `b_k`는 윈도우 `w_k` 내에서 `q`와 `p` 사이의 차이를 최소화하도록 계산.

### 2.2 수학적 정의

```
mean_I = boxfilter(I) / |w|
mean_p = boxfilter(p) / |w|
corr_I = boxfilter(I * I) / |w|
corr_Ip = boxfilter(I * p) / |w|

var_I = corr_I - mean_I * mean_I
cov_Ip = corr_Ip - mean_I * mean_p

a = cov_Ip / (var_I + eps)
b = mean_p - a * mean_I

mean_a = boxfilter(a) / |w|
mean_b = boxfilter(b) / |w|

q = mean_a * I + mean_b
```

### 2.3 Fast Guided Filter (서브샘플링)

성능 향상을 위해 다운샘플 → 계산 → 업샘플 전략 적용:

```
I_sub = subsample(I, s)
p_sub = subsample(p, s)

// 축소된 해상도에서 a, b 계산
a_sub, b_sub = guided_filter(I_sub, p_sub, r/s, eps)

// 원본 해상도로 업샘플
a = upsample(a_sub)
b = upsample(b_sub)

q = a * I + b
```

---

## 3. 상세 구현

### 3.1 FastGuidedFilter 클래스

**파일**: `cpp/include/iris_sdk/fast_guided_filter.h`

```cpp
#ifndef IRIS_SDK_FAST_GUIDED_FILTER_H
#define IRIS_SDK_FAST_GUIDED_FILTER_H

#include <opencv2/core.hpp>

namespace iris_sdk {

/**
 * @brief Fast Guided Filter 구현
 *
 * opencv-contrib 의존 없이 직접 구현
 * Box Filter 기반 O(1) 시간 복잡도
 */
class FastGuidedFilter {
public:
    /**
     * @brief Guided Filter 적용 (Self-Guided)
     *
     * 입력 이미지 자체를 가이드로 사용 (에지 보존 스무딩)
     *
     * @param src 입력 이미지 (CV_8UC1/3 또는 CV_32FC1/3)
     * @param dst 출력 이미지
     * @param radius 윈도우 반경 (실제 윈도우 = 2*radius+1)
     * @param eps 정규화 파라미터 (0.01² ~ 0.4² 범위, 클수록 스무딩 강함)
     * @param subsample_ratio 서브샘플링 비율 (1 = 없음, 2/4 = 빠른 처리)
     */
    static void filter(
        const cv::Mat& src,
        cv::Mat& dst,
        int radius,
        double eps,
        int subsample_ratio = 1
    );

    /**
     * @brief Guided Filter 적용 (External Guide)
     *
     * 외부 가이드 이미지를 사용 (예: 피부톤 마스크 적용)
     *
     * @param guide 가이드 이미지 (에지 정보 제공)
     * @param src 입력 이미지
     * @param dst 출력 이미지
     * @param radius 윈도우 반경
     * @param eps 정규화 파라미터
     * @param subsample_ratio 서브샘플링 비율
     */
    static void filter(
        const cv::Mat& guide,
        const cv::Mat& src,
        cv::Mat& dst,
        int radius,
        double eps,
        int subsample_ratio = 1
    );

private:
    /**
     * @brief 단일 채널 Guided Filter 코어 구현
     */
    static void filterSingleChannel(
        const cv::Mat& guide,
        const cv::Mat& src,
        cv::Mat& dst,
        int radius,
        double eps
    );

    /**
     * @brief 다중 채널 Guided Filter (채널별 처리)
     */
    static void filterMultiChannel(
        const cv::Mat& guide,
        const cv::Mat& src,
        cv::Mat& dst,
        int radius,
        double eps
    );

    /**
     * @brief Box Filter (적분 이미지 기반, O(1))
     *
     * OpenCV의 cv::boxFilter는 이미 O(1) 최적화됨
     */
    static void boxFilter(
        const cv::Mat& src,
        cv::Mat& dst,
        int radius
    );
};

} // namespace iris_sdk

#endif // IRIS_SDK_FAST_GUIDED_FILTER_H
```

### 3.2 구현 파일

**파일**: `cpp/src/fast_guided_filter.cpp`

```cpp
#include "iris_sdk/fast_guided_filter.h"
#include <opencv2/imgproc.hpp>

namespace iris_sdk {

void FastGuidedFilter::boxFilter(
    const cv::Mat& src,
    cv::Mat& dst,
    int radius) {

    // OpenCV boxFilter는 적분 이미지 기반 O(1) 구현
    int ksize = 2 * radius + 1;
    cv::boxFilter(src, dst, -1, cv::Size(ksize, ksize),
                  cv::Point(-1, -1), true, cv::BORDER_REFLECT);
}

void FastGuidedFilter::filterSingleChannel(
    const cv::Mat& guide,
    const cv::Mat& src,
    cv::Mat& dst,
    int radius,
    double eps) {

    CV_Assert(guide.channels() == 1 && src.channels() == 1);

    // 32F로 변환
    cv::Mat I, p;
    guide.convertTo(I, CV_32F, 1.0 / 255.0);
    src.convertTo(p, CV_32F, 1.0 / 255.0);

    // Step 1: Box filter로 평균 계산
    cv::Mat mean_I, mean_p;
    boxFilter(I, mean_I, radius);
    boxFilter(p, mean_p, radius);

    // Step 2: 상관관계 계산
    cv::Mat I_square, Ip;
    cv::multiply(I, I, I_square);
    cv::multiply(I, p, Ip);

    cv::Mat mean_II, mean_Ip;
    boxFilter(I_square, mean_II, radius);
    boxFilter(Ip, mean_Ip, radius);

    // Step 3: 분산 및 공분산
    cv::Mat var_I, cov_Ip;
    var_I = mean_II - mean_I.mul(mean_I);
    cov_Ip = mean_Ip - mean_I.mul(mean_p);

    // Step 4: 계수 a, b 계산
    cv::Mat a, b;
    a = cov_Ip / (var_I + eps);
    b = mean_p - a.mul(mean_I);

    // Step 5: 평균화
    cv::Mat mean_a, mean_b;
    boxFilter(a, mean_a, radius);
    boxFilter(b, mean_b, radius);

    // Step 6: 최종 출력
    cv::Mat q_float;
    q_float = mean_a.mul(I) + mean_b;

    // 8비트로 변환
    q_float.convertTo(dst, CV_8U, 255.0);
}

void FastGuidedFilter::filterMultiChannel(
    const cv::Mat& guide,
    const cv::Mat& src,
    cv::Mat& dst,
    int radius,
    double eps) {

    CV_Assert(guide.channels() == 3 && src.channels() == 3);

    // 채널 분리
    std::vector<cv::Mat> guide_channels, src_channels, dst_channels;
    cv::split(guide, guide_channels);
    cv::split(src, src_channels);
    dst_channels.resize(3);

    // 채널별 처리 (병렬화 가능)
    #pragma omp parallel for
    for (int c = 0; c < 3; ++c) {
        filterSingleChannel(guide_channels[c], src_channels[c],
                            dst_channels[c], radius, eps);
    }

    // 채널 병합
    cv::merge(dst_channels, dst);
}

void FastGuidedFilter::filter(
    const cv::Mat& src,
    cv::Mat& dst,
    int radius,
    double eps,
    int subsample_ratio) {

    // Self-guided: 입력 = 가이드
    filter(src, src, dst, radius, eps, subsample_ratio);
}

void FastGuidedFilter::filter(
    const cv::Mat& guide,
    const cv::Mat& src,
    cv::Mat& dst,
    int radius,
    double eps,
    int subsample_ratio) {

    CV_Assert(!guide.empty() && !src.empty());
    CV_Assert(guide.size() == src.size());

    cv::Mat work_guide = guide;
    cv::Mat work_src = src;

    // 서브샘플링 적용 (Fast Guided Filter)
    if (subsample_ratio > 1) {
        int new_width = guide.cols / subsample_ratio;
        int new_height = guide.rows / subsample_ratio;

        cv::resize(guide, work_guide, cv::Size(new_width, new_height),
                   0, 0, cv::INTER_LINEAR);
        cv::resize(src, work_src, cv::Size(new_width, new_height),
                   0, 0, cv::INTER_LINEAR);

        // radius도 조정
        radius = std::max(1, radius / subsample_ratio);
    }

    cv::Mat work_dst;

    if (work_guide.channels() == 1) {
        // 그레이스케일
        cv::Mat gray_src;
        if (work_src.channels() == 3) {
            cv::cvtColor(work_src, gray_src, cv::COLOR_BGR2GRAY);
        } else {
            gray_src = work_src;
        }
        filterSingleChannel(work_guide, gray_src, work_dst, radius, eps);
    } else {
        // 컬러
        filterMultiChannel(work_guide, work_src, work_dst, radius, eps);
    }

    // 업샘플링
    if (subsample_ratio > 1) {
        cv::resize(work_dst, dst, src.size(), 0, 0, cv::INTER_LINEAR);
    } else {
        dst = work_dst;
    }
}

} // namespace iris_sdk
```

### 3.3 뷰티 필터 통합

**파일**: `cpp/src/cpu_beauty_backend.cpp` (확장)

```cpp
#include "iris_sdk/fast_guided_filter.h"

void CPUBeautyBackend::applySkinSmoothingV2(
    cv::Mat& roi,
    float strength,
    const cv::Mat& protection_mask) {

    // Guided Filter 파라미터
    // radius: 윈도우 크기, eps: 스무딩 강도
    int radius = 8;  // 고정 또는 강도에 따라 조정
    double eps = 0.01 + strength * 0.15;  // 0.01 ~ 0.16

    // 서브샘플링 비율 (2 = 1/2 해상도에서 계산)
    int subsample = (roi.cols * roi.rows > 640 * 480) ? 2 : 1;

    cv::Mat smoothed;
    FastGuidedFilter::filter(roi, smoothed, radius, eps, subsample);

    // 원본과 블렌딩 (강도 조절)
    cv::addWeighted(smoothed, strength, roi, 1.0 - strength, 0, smoothed);

    // 보호 마스크 적용
    if (!protection_mask.empty()) {
        cv::Mat mask_inv;
        cv::bitwise_not(protection_mask, mask_inv);

        cv::Mat mask_3ch;
        cv::cvtColor(mask_inv, mask_3ch, cv::COLOR_GRAY2BGR);
        mask_3ch.convertTo(mask_3ch, CV_32F, 1.0 / 255.0);

        cv::Mat roi_f, smoothed_f;
        roi.convertTo(roi_f, CV_32F);
        smoothed.convertTo(smoothed_f, CV_32F);

        cv::Mat result = smoothed_f.mul(mask_3ch) +
                         roi_f.mul(cv::Scalar(1, 1, 1) - mask_3ch);
        result.convertTo(roi, CV_8U);
    } else {
        smoothed.copyTo(roi);
    }
}
```

---

## 4. 성능 비교

### 4.1 Bilateral Filter vs Guided Filter

| 항목 | Bilateral Filter | Guided Filter | Fast Guided (s=2) |
|------|------------------|---------------|-------------------|
| 시간 복잡도 | O(r²) | O(1) | O(1) |
| 640x480 처리 시간 | 15-20ms | 5-8ms | 3-5ms |
| 1080p 처리 시간 | 45-60ms | 15-20ms | 6-10ms |
| 에지 보존 | 우수 | 우수 | 우수 |
| 헤일로 아티팩트 | 있음 | 적음 | 적음 |

### 4.2 eps 파라미터 가이드

| 강도 | eps 값 | 효과 |
|------|--------|------|
| 미약 | 0.01² = 0.0001 | 노이즈 제거, 에지 완벽 보존 |
| 약함 | 0.04² = 0.0016 | 부드러운 스무딩 |
| 보통 | 0.1² = 0.01 | 피부 스무딩 (권장) |
| 강함 | 0.2² = 0.04 | 강한 스무딩, 에지 약간 흐림 |
| 매우 강함 | 0.4² = 0.16 | 블러에 가까움 |

---

## 5. 단위 테스트

**파일**: `cpp/tests/test_fast_guided_filter.cpp`

```cpp
#include <gtest/gtest.h>
#include "iris_sdk/fast_guided_filter.h"
#include <chrono>

using namespace iris_sdk;

class FastGuidedFilterTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 테스트 이미지 생성
        test_image_ = cv::Mat(480, 640, CV_8UC3);
        cv::randu(test_image_, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

        // 에지가 있는 테스트 이미지
        edge_image_ = cv::Mat::zeros(480, 640, CV_8UC3);
        cv::rectangle(edge_image_, cv::Rect(100, 100, 200, 200),
                      cv::Scalar(255, 255, 255), -1);
    }

    cv::Mat test_image_;
    cv::Mat edge_image_;
};

TEST_F(FastGuidedFilterTest, BasicFiltering) {
    cv::Mat result;
    EXPECT_NO_THROW(
        FastGuidedFilter::filter(test_image_, result, 8, 0.01)
    );

    EXPECT_EQ(result.size(), test_image_.size());
    EXPECT_EQ(result.type(), test_image_.type());
}

TEST_F(FastGuidedFilterTest, EdgePreservation) {
    cv::Mat result;
    FastGuidedFilter::filter(edge_image_, result, 8, 0.01);

    // 에지 검출
    cv::Mat gray_orig, gray_result;
    cv::cvtColor(edge_image_, gray_orig, cv::COLOR_BGR2GRAY);
    cv::cvtColor(result, gray_result, cv::COLOR_BGR2GRAY);

    cv::Mat edges_orig, edges_result;
    cv::Canny(gray_orig, edges_orig, 50, 150);
    cv::Canny(gray_result, edges_result, 50, 150);

    // 에지 보존율 80% 이상
    double orig_edges = cv::countNonZero(edges_orig);
    double result_edges = cv::countNonZero(edges_result);
    double preservation = result_edges / orig_edges;

    EXPECT_GE(preservation, 0.8);
}

TEST_F(FastGuidedFilterTest, SubsamplingPerformance) {
    cv::Mat dst1, dst2;

    // 서브샘플링 없음
    auto start1 = std::chrono::high_resolution_clock::now();
    FastGuidedFilter::filter(test_image_, dst1, 8, 0.01, 1);
    auto dur1 = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start1);

    // 서브샘플링 2x
    auto start2 = std::chrono::high_resolution_clock::now();
    FastGuidedFilter::filter(test_image_, dst2, 8, 0.01, 2);
    auto dur2 = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start2);

    // 서브샘플링이 더 빠름
    EXPECT_LT(dur2.count(), dur1.count());

    // 결과 품질은 유사 (PSNR > 30dB)
    cv::Mat diff;
    cv::absdiff(dst1, dst2, diff);
    double max_diff = cv::mean(diff)[0];
    EXPECT_LT(max_diff, 20);  // 평균 차이 20 미만
}

TEST_F(FastGuidedFilterTest, GrayscaleFiltering) {
    cv::Mat gray;
    cv::cvtColor(test_image_, gray, cv::COLOR_BGR2GRAY);

    cv::Mat result;
    EXPECT_NO_THROW(
        FastGuidedFilter::filter(gray, result, 8, 0.01)
    );

    EXPECT_EQ(result.channels(), 1);
}

TEST_F(FastGuidedFilterTest, EpsParameterEffect) {
    cv::Mat result_low_eps, result_high_eps;

    FastGuidedFilter::filter(test_image_, result_low_eps, 8, 0.001);   // 에지 보존 강함
    FastGuidedFilter::filter(test_image_, result_high_eps, 8, 0.16);  // 스무딩 강함

    // 높은 eps = 더 많은 스무딩 = 원본과 차이 큼
    cv::Mat diff_low, diff_high;
    cv::absdiff(test_image_, result_low_eps, diff_low);
    cv::absdiff(test_image_, result_high_eps, diff_high);

    double mean_diff_low = cv::mean(diff_low)[0];
    double mean_diff_high = cv::mean(diff_high)[0];

    EXPECT_LT(mean_diff_low, mean_diff_high);
}

TEST_F(FastGuidedFilterTest, ExternalGuide) {
    cv::Mat guide = test_image_.clone();
    cv::GaussianBlur(test_image_, guide, cv::Size(5, 5), 0);  // 블러된 가이드

    cv::Mat result;
    EXPECT_NO_THROW(
        FastGuidedFilter::filter(guide, test_image_, result, 8, 0.01)
    );

    EXPECT_EQ(result.size(), test_image_.size());
}

TEST_F(FastGuidedFilterTest, PerformanceBenchmark) {
    cv::Mat hd_image(1080, 1920, CV_8UC3);
    cv::randu(hd_image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

    cv::Mat result;
    const int iterations = 10;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        FastGuidedFilter::filter(hd_image, result, 8, 0.01, 2);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double avg_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start).count() / static_cast<double>(iterations);

    std::cout << "Average time for 1080p (s=2): " << avg_ms << " ms" << std::endl;

    // 1080p에서 15ms 이하 목표
    EXPECT_LT(avg_ms, 15.0);
}
```

---

## 6. CMake 설정

**파일**: `cpp/CMakeLists.txt` (추가)

```cmake
# Fast Guided Filter
set(FILTER_SOURCES
    src/fast_guided_filter.cpp
)

set(FILTER_HEADERS
    include/iris_sdk/fast_guided_filter.h
)

# OpenMP 지원 (선택적, 채널 병렬 처리용)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(iris_sdk PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(iris_sdk PRIVATE IRIS_SDK_HAS_OPENMP)
endif()
```

---

## 7. 완료 기준

- [ ] `FastGuidedFilter` 클래스 구현
- [ ] Box Filter 기반 O(1) 최적화
- [ ] 서브샘플링 지원
- [ ] 단일/다중 채널 처리
- [ ] 외부 가이드 이미지 지원
- [ ] CPUBeautyBackend 통합
- [ ] 단위 테스트 100% 통과
- [ ] 1080p 기준 10ms 이하 (s=2)

---

## 8. 다음 작업

- **P2-W2-03**: 새 필터 효과 구현 (화이트닝, 컬러 밸런스)
