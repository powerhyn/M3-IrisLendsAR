/**
 * @file fast_guided_filter.cpp
 * @brief Fast Guided Filter implementation
 *
 * Implementation based on:
 * - "Guided Image Filtering" (He et al., ECCV 2010)
 * - "Fast Guided Filter" (He & Sun, 2015)
 */

#include "iris_sdk/fast_guided_filter.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace iris_sdk {

namespace {

/**
 * @brief Convert image to 32F if needed
 */
cv::Mat ensureFloat(const cv::Mat& src) {
    if (src.depth() == CV_32F) {
        return src;
    }
    cv::Mat dst;
    src.convertTo(dst, CV_32F, 1.0 / 255.0);
    return dst;
}

/**
 * @brief Convert back to original type
 */
void convertBack(const cv::Mat& src, cv::Mat& dst, int original_type) {
    if (src.type() == original_type) {
        src.copyTo(dst);
        return;
    }

    int depth = CV_MAT_DEPTH(original_type);
    if (depth == CV_8U) {
        src.convertTo(dst, original_type, 255.0);
    } else if (depth == CV_16U) {
        src.convertTo(dst, original_type, 65535.0);
    } else {
        src.convertTo(dst, original_type);
    }
}

/**
 * @brief O(1) box filter using OpenCV's optimized implementation
 *
 * OpenCV's boxFilter uses integral images for O(1) complexity
 */
void boxFilter(const cv::Mat& src, cv::Mat& dst, int radius) {
    int ksize = 2 * radius + 1;
    cv::boxFilter(src, dst, CV_32F, cv::Size(ksize, ksize), cv::Point(-1, -1), true, cv::BORDER_REFLECT_101);
}

} // anonymous namespace

void FastGuidedFilter::filter(
    const cv::Mat& src,
    cv::Mat& dst,
    int radius,
    double eps,
    int subsample_ratio
) {
    // Self-guided: use input as guide
    filter(src, src, dst, radius, eps, subsample_ratio);
}

void FastGuidedFilter::filter(
    const cv::Mat& guide,
    const cv::Mat& src,
    cv::Mat& dst,
    int radius,
    double eps,
    int subsample_ratio
) {
    CV_Assert(!guide.empty() && !src.empty());
    CV_Assert(guide.size() == src.size());
    CV_Assert(radius > 0);
    CV_Assert(eps > 0);
    CV_Assert(subsample_ratio >= 1);

    if (subsample_ratio > 1) {
        filterWithSubsampling(guide, src, dst, radius, eps, subsample_ratio);
    } else {
        // Convert to float for processing
        cv::Mat guide_f = ensureFloat(guide);
        cv::Mat src_f = ensureFloat(src);

        cv::Mat result;
        if (src_f.channels() == 1) {
            cv::Mat guide_gray;
            if (guide_f.channels() > 1) {
                cv::cvtColor(guide_f, guide_gray, cv::COLOR_BGR2GRAY);
            } else {
                guide_gray = guide_f;
            }
            filterSingleChannel(guide_gray, src_f, result, radius, eps);
        } else {
            filterMultiChannel(guide_f, src_f, result, radius, eps);
        }

        convertBack(result, dst, src.type());
    }
}

void FastGuidedFilter::filterSingleChannel(
    const cv::Mat& guide,
    const cv::Mat& src,
    cv::Mat& dst,
    int radius,
    double eps
) {
    CV_Assert(guide.channels() == 1);
    CV_Assert(src.channels() == 1);

    cv::Mat I = guide;
    cv::Mat p = src;

    // Step 1: Compute means
    cv::Mat mean_I, mean_p;
    boxFilter(I, mean_I, radius);
    boxFilter(p, mean_p, radius);

    // Step 2: Compute correlations
    cv::Mat II = I.mul(I);
    cv::Mat Ip = I.mul(p);

    cv::Mat mean_II, mean_Ip;
    boxFilter(II, mean_II, radius);
    boxFilter(Ip, mean_Ip, radius);

    // Step 3: Compute variance and covariance
    cv::Mat var_I = mean_II - mean_I.mul(mean_I);
    cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);

    // Step 4: Compute linear coefficients a and b
    cv::Mat a = cov_Ip / (var_I + eps);
    cv::Mat b = mean_p - a.mul(mean_I);

    // Step 5: Compute mean of a and b
    cv::Mat mean_a, mean_b;
    boxFilter(a, mean_a, radius);
    boxFilter(b, mean_b, radius);

    // Step 6: Compute output
    dst = mean_a.mul(I) + mean_b;
}

void FastGuidedFilter::filterMultiChannel(
    const cv::Mat& guide,
    const cv::Mat& src,
    cv::Mat& dst,
    int radius,
    double eps
) {
    CV_Assert(src.channels() == 3);

    // Prepare guide image (grayscale for simplicity)
    cv::Mat guide_gray;
    if (guide.channels() > 1) {
        cv::cvtColor(guide, guide_gray, cv::COLOR_BGR2GRAY);
    } else {
        guide_gray = guide;
    }

    // Split source channels
    std::vector<cv::Mat> src_channels(3);
    cv::split(src, src_channels);

    // Process each channel
    std::vector<cv::Mat> dst_channels(3);

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < 3; ++i) {
        filterSingleChannel(guide_gray, src_channels[i], dst_channels[i], radius, eps);
    }

    // Merge channels
    cv::merge(dst_channels, dst);
}

void FastGuidedFilter::filterWithSubsampling(
    const cv::Mat& guide,
    const cv::Mat& src,
    cv::Mat& dst,
    int radius,
    double eps,
    int subsample_ratio
) {
    // Convert to float
    cv::Mat guide_f = ensureFloat(guide);
    cv::Mat src_f = ensureFloat(src);

    // Downsample guide and source
    cv::Mat guide_sub = downsample(guide_f, subsample_ratio);
    cv::Mat src_sub = downsample(src_f, subsample_ratio);

    // Adjust radius for subsampled resolution
    int radius_sub = std::max(1, radius / subsample_ratio);

    // Prepare guide (grayscale)
    cv::Mat guide_sub_gray;
    if (guide_sub.channels() > 1) {
        cv::cvtColor(guide_sub, guide_sub_gray, cv::COLOR_BGR2GRAY);
    } else {
        guide_sub_gray = guide_sub;
    }

    // Compute coefficients a and b at low resolution
    cv::Mat mean_I, mean_p;
    boxFilter(guide_sub_gray, mean_I, radius_sub);

    cv::Mat II = guide_sub_gray.mul(guide_sub_gray);
    cv::Mat mean_II;
    boxFilter(II, mean_II, radius_sub);

    cv::Mat var_I = mean_II - mean_I.mul(mean_I);

    // Process each channel
    std::vector<cv::Mat> src_channels, dst_channels;
    if (src_sub.channels() > 1) {
        cv::split(src_sub, src_channels);
    } else {
        src_channels.push_back(src_sub);
    }
    dst_channels.resize(src_channels.size());

    // Prepare full resolution guide
    cv::Mat guide_f_gray;
    if (guide_f.channels() > 1) {
        cv::cvtColor(guide_f, guide_f_gray, cv::COLOR_BGR2GRAY);
    } else {
        guide_f_gray = guide_f;
    }

    // Prepare full resolution source channels
    std::vector<cv::Mat> src_f_channels;
    if (src_f.channels() > 1) {
        cv::split(src_f, src_f_channels);
    } else {
        src_f_channels.push_back(src_f);
    }

    cv::Size original_size = src_f.size();

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int c = 0; c < static_cast<int>(src_channels.size()); ++c) {
        cv::Mat mean_p_c;
        boxFilter(src_channels[c], mean_p_c, radius_sub);

        cv::Mat Ip = guide_sub_gray.mul(src_channels[c]);
        cv::Mat mean_Ip;
        boxFilter(Ip, mean_Ip, radius_sub);

        cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p_c);

        // Compute a and b at low resolution
        cv::Mat a_sub = cov_Ip / (var_I + eps);
        cv::Mat b_sub = mean_p_c - a_sub.mul(mean_I);

        // Upsample a and b to original resolution
        cv::Mat a_up = upsample(a_sub, original_size);
        cv::Mat b_up = upsample(b_sub, original_size);

        // Apply at full resolution: q = a * I + b
        dst_channels[c] = a_up.mul(guide_f_gray) + b_up;
    }

    // Merge and convert back
    cv::Mat result;
    if (dst_channels.size() > 1) {
        cv::merge(dst_channels, result);
    } else {
        result = dst_channels[0];
    }

    convertBack(result, dst, src.type());
}

cv::Mat FastGuidedFilter::downsample(const cv::Mat& src, int ratio) {
    if (ratio <= 1) {
        return src.clone();
    }

    cv::Mat dst;
    cv::Size new_size(
        std::max(1, src.cols / ratio),
        std::max(1, src.rows / ratio)
    );
    cv::resize(src, dst, new_size, 0, 0, cv::INTER_AREA);
    return dst;
}

cv::Mat FastGuidedFilter::upsample(const cv::Mat& src, cv::Size target_size) {
    if (src.size() == target_size) {
        return src.clone();
    }

    cv::Mat dst;
    cv::resize(src, dst, target_size, 0, 0, cv::INTER_LINEAR);
    return dst;
}

double FastGuidedFilter::getRecommendedEps(int level) {
    // eps values corresponding to different smoothing levels
    // Based on typical use cases for skin smoothing
    static const double eps_values[] = {
        0.0001,  // 0: minimal - noise removal only
        0.0016,  // 1: light - subtle smoothing
        0.01,    // 2: medium - recommended for skin
        0.04,    // 3: strong - visible smoothing
        0.16     // 4: very strong - blur-like
    };

    level = std::clamp(level, 0, 4);
    return eps_values[level];
}

int FastGuidedFilter::getRecommendedSubsampleRatio(int width, int height, double target_ms) {
    // Empirical estimation based on typical performance
    // Box filter is O(1) per pixel, but total pixels affect performance

    int total_pixels = width * height;

    // Rough estimates (can be tuned based on actual benchmarks)
    // 720p (~1M pixels): ratio=1 ~10ms
    // 1080p (~2M pixels): ratio=1 ~20ms, ratio=2 ~8ms
    // 4K (~8M pixels): ratio=1 ~80ms, ratio=2 ~20ms, ratio=4 ~8ms

    if (total_pixels < 500000) {  // < 720p
        return 1;
    } else if (total_pixels < 1500000) {  // 720p - 1080p
        return target_ms < 10 ? 2 : 1;
    } else if (total_pixels < 4000000) {  // 1080p - 2K
        return target_ms < 10 ? 2 : (target_ms < 20 ? 1 : 1);
    } else {  // >= 4K
        return target_ms < 10 ? 4 : (target_ms < 20 ? 2 : 1);
    }
}

} // namespace iris_sdk
