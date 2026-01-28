/**
 * @file fast_guided_filter.h
 * @brief Fast Guided Filter implementation for edge-preserving smoothing
 *
 * Based on:
 * - "Guided Image Filtering" (He et al., ECCV 2010)
 * - "Fast Guided Filter" (He & Sun, 2015)
 *
 * Features:
 * - O(1) time complexity using box filter
 * - Subsampling optimization for faster processing
 * - Self-guided and external guide modes
 * - Multi-channel support with parallel processing
 */

#pragma once

#include <opencv2/core.hpp>

namespace iris_sdk {

/**
 * @brief Fast Guided Filter for edge-preserving smoothing
 *
 * The guided filter is a linear filter that computes the output as a weighted
 * sum of the input pixels, where the weights depend on the guidance image.
 *
 * Mathematical formulation:
 *   q_i = a_k * I_i + b_k, for all i in window w_k
 *
 * where a and b are linear coefficients computed as:
 *   a_k = cov(I, p) / (var(I) + eps)
 *   b_k = mean(p) - a_k * mean(I)
 *
 * The eps parameter controls the degree of smoothing:
 * - Small eps (0.0001): noise removal, perfect edge preservation
 * - Medium eps (0.01): skin smoothing (recommended for beauty filter)
 * - Large eps (0.16): strong blur effect
 */
class FastGuidedFilter {
public:
    /**
     * @brief Apply self-guided filter (input image is also the guide)
     *
     * @param src Input image (CV_8UC1, CV_8UC3, CV_32FC1, or CV_32FC3)
     * @param dst Output filtered image (same type as src)
     * @param radius Filter radius (kernel size = 2*radius + 1)
     * @param eps Regularization parameter (controls smoothness vs edge preservation)
     * @param subsample_ratio Subsampling ratio for fast mode (1 = no subsampling)
     *
     * @note For 1080p images, use subsample_ratio=2 or 4 for real-time performance
     */
    static void filter(
        const cv::Mat& src,
        cv::Mat& dst,
        int radius,
        double eps,
        int subsample_ratio = 1
    );

    /**
     * @brief Apply guided filter with external guide image
     *
     * @param guide Guide image (determines edge structure)
     * @param src Input image to be filtered
     * @param dst Output filtered image
     * @param radius Filter radius
     * @param eps Regularization parameter
     * @param subsample_ratio Subsampling ratio for fast mode
     *
     * @note Guide and src must have the same size
     * @note For skin smoothing, use the original image as guide to preserve structure
     */
    static void filter(
        const cv::Mat& guide,
        const cv::Mat& src,
        cv::Mat& dst,
        int radius,
        double eps,
        int subsample_ratio = 1
    );

    /**
     * @brief Get recommended eps value for different smoothing levels
     *
     * @param level Smoothing level: 0=minimal, 1=light, 2=medium, 3=strong, 4=very_strong
     * @return Recommended eps value
     */
    static double getRecommendedEps(int level);

    /**
     * @brief Get recommended subsample ratio based on image size
     *
     * @param width Image width
     * @param height Image height
     * @param target_ms Target processing time in milliseconds
     * @return Recommended subsample ratio
     */
    static int getRecommendedSubsampleRatio(int width, int height, double target_ms = 15.0);

private:
    /**
     * @brief Filter single channel image
     */
    static void filterSingleChannel(
        const cv::Mat& guide,
        const cv::Mat& src,
        cv::Mat& dst,
        int radius,
        double eps
    );

    /**
     * @brief Filter multi-channel image (parallel processing per channel)
     */
    static void filterMultiChannel(
        const cv::Mat& guide,
        const cv::Mat& src,
        cv::Mat& dst,
        int radius,
        double eps
    );

    /**
     * @brief Apply subsampled guided filter for fast processing
     */
    static void filterWithSubsampling(
        const cv::Mat& guide,
        const cv::Mat& src,
        cv::Mat& dst,
        int radius,
        double eps,
        int subsample_ratio
    );

    /**
     * @brief Downsample image by ratio
     */
    static cv::Mat downsample(const cv::Mat& src, int ratio);

    /**
     * @brief Upsample image to target size
     */
    static cv::Mat upsample(const cv::Mat& src, cv::Size target_size);
};

} // namespace iris_sdk
