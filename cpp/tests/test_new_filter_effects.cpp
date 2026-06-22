/**
 * @file test_new_filter_effects.cpp
 * @brief CPUBeautyBackend brightness 생존 테스트 + apply() 파이프라인 스모크.
 *
 * P8-W2-C: 곁가지 CPU 효과(skin smoothing / soft focus / whitening / color balance /
 *   V2 Guided 계열 / wrinkle removal)가 전부 제거되어, 해당 효과를 검증하던
 *   케이스(SkinSmoothingV2_* / SoftFocusV2_* / Whitening_* / ColorBalance_* /
 *   Performance_SmoothingV2_*)도 함께 삭제했다.
 *
 * 남은 테스트:
 * - BrightnessV2_* : apply() → applyBrightness 생존 검증
 * - FullPipeline_AllEffectsCombined : brightness-only 스모크
 * - FullPipeline_DisabledConfig_NoChange : enabled=false no-op
 * - FullPipeline_MultipleFormats : 포맷 변환 경로(brightness 적용)
 */

#include <gtest/gtest.h>
#include "iris_sdk/cpu_beauty_backend.h"
#include "iris_sdk/types.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cmath>
#include <numeric>
#include <vector>

// Use iris_sdk namespace for types, but be explicit where needed
using namespace iris_sdk;

//=============================================================================
// Test Fixture
//=============================================================================

class NewFilterEffectsTest : public ::testing::Test {
protected:
    void SetUp() override {
        createTestImages();
        createFakeFaceMesh();
    }

    void createTestImages() {
        // Create test face-like image (480x640)
        test_image_ = cv::Mat(480, 640, CV_8UC3);

        // Fill with skin-like color (warm beige tone)
        test_image_.setTo(cv::Scalar(140, 170, 200));  // BGR

        // Add some texture/noise to simulate skin
        cv::Mat noise(480, 640, CV_8UC3);
        cv::randn(noise, cv::Scalar(0, 0, 0), cv::Scalar(10, 10, 10));
        cv::add(test_image_, noise, test_image_);

        // Add brighter regions (highlights)
        cv::rectangle(test_image_, cv::Rect(200, 100, 100, 50),
                     cv::Scalar(200, 220, 240), -1);

        // Add darker regions (shadows)
        cv::rectangle(test_image_, cv::Rect(300, 300, 100, 50),
                     cv::Scalar(80, 100, 120), -1);

        // Create protection mask (simulating eyes region)
        protection_mask_ = cv::Mat::zeros(480, 640, CV_8UC1);
        cv::rectangle(protection_mask_, cv::Rect(150, 180, 80, 40), cv::Scalar(255), -1);
        cv::rectangle(protection_mask_, cv::Rect(350, 180, 80, 40), cv::Scalar(255), -1);
    }

    void createFakeFaceMesh() {
        // Create fake face mesh landmarks (478 points)
        // Just enough to test wrinkle region creation
        fake_landmarks_.resize(478);

        // Set all landmarks to default positions in center area
        for (int i = 0; i < 478; ++i) {
            fake_landmarks_[i].x = 0.5f + (i % 20 - 10) * 0.01f;
            fake_landmarks_[i].y = 0.5f + (i / 20 - 12) * 0.01f;
            fake_landmarks_[i].z = 0.0f;
            fake_landmarks_[i].visibility = 1.0f;
        }

        // Set specific landmarks for wrinkle regions
        // Forehead landmarks (10, 338, 297, 332, 284, 251, 389)
        fake_landmarks_[10] = {0.5f, 0.15f, 0.0f, 1.0f};
        fake_landmarks_[338] = {0.55f, 0.15f, 0.0f, 1.0f};
        fake_landmarks_[297] = {0.45f, 0.15f, 0.0f, 1.0f};
        fake_landmarks_[332] = {0.6f, 0.18f, 0.0f, 1.0f};
        fake_landmarks_[284] = {0.4f, 0.18f, 0.0f, 1.0f};
        fake_landmarks_[251] = {0.35f, 0.2f, 0.0f, 1.0f};
        fake_landmarks_[389] = {0.65f, 0.2f, 0.0f, 1.0f};

        // Frown lines landmarks (9, 8, 168, 6, 197, 195, 5, 4)
        fake_landmarks_[9] = {0.5f, 0.22f, 0.0f, 1.0f};
        fake_landmarks_[8] = {0.48f, 0.24f, 0.0f, 1.0f};
        fake_landmarks_[168] = {0.52f, 0.24f, 0.0f, 1.0f};
        fake_landmarks_[6] = {0.5f, 0.26f, 0.0f, 1.0f};
        fake_landmarks_[197] = {0.48f, 0.28f, 0.0f, 1.0f};
        fake_landmarks_[195] = {0.52f, 0.28f, 0.0f, 1.0f};
        fake_landmarks_[5] = {0.5f, 0.3f, 0.0f, 1.0f};
        fake_landmarks_[4] = {0.5f, 0.32f, 0.0f, 1.0f};

        // Crow feet landmarks - left (130, 247, 30, 29, 27, 28, 56, 190)
        fake_landmarks_[130] = {0.3f, 0.35f, 0.0f, 1.0f};
        fake_landmarks_[247] = {0.25f, 0.33f, 0.0f, 1.0f};
        fake_landmarks_[30] = {0.22f, 0.35f, 0.0f, 1.0f};
        fake_landmarks_[29] = {0.2f, 0.37f, 0.0f, 1.0f};
        fake_landmarks_[27] = {0.22f, 0.39f, 0.0f, 1.0f};
        fake_landmarks_[28] = {0.25f, 0.41f, 0.0f, 1.0f};
        fake_landmarks_[56] = {0.28f, 0.4f, 0.0f, 1.0f};
        fake_landmarks_[190] = {0.3f, 0.38f, 0.0f, 1.0f};

        // Crow feet landmarks - right (359, 467, 260, 259, 257, 258, 286, 414)
        fake_landmarks_[359] = {0.7f, 0.35f, 0.0f, 1.0f};
        fake_landmarks_[467] = {0.75f, 0.33f, 0.0f, 1.0f};
        fake_landmarks_[260] = {0.78f, 0.35f, 0.0f, 1.0f};
        fake_landmarks_[259] = {0.8f, 0.37f, 0.0f, 1.0f};
        fake_landmarks_[257] = {0.78f, 0.39f, 0.0f, 1.0f};
        fake_landmarks_[258] = {0.75f, 0.41f, 0.0f, 1.0f};
        fake_landmarks_[286] = {0.72f, 0.4f, 0.0f, 1.0f};
        fake_landmarks_[414] = {0.7f, 0.38f, 0.0f, 1.0f};
    }

    double calculateMeanBrightness(const cv::Mat& image) {
        cv::Mat gray;
        if (image.channels() == 3) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image;
        }
        return cv::mean(gray)[0];
    }

    double calculateVariance(const cv::Mat& image) {
        cv::Mat gray;
        if (image.channels() == 3) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image;
        }

        cv::Mat mean_mat, stddev_mat;
        cv::meanStdDev(gray, mean_mat, stddev_mat);
        double stddev = stddev_mat.at<double>(0);
        return stddev * stddev;
    }

    double calculateLabChannel(const cv::Mat& image, int channel) {
        cv::Mat lab;
        cv::cvtColor(image, lab, cv::COLOR_BGR2Lab);

        std::vector<cv::Mat> channels;
        cv::split(lab, channels);

        return cv::mean(channels[channel])[0];
    }

    cv::Mat test_image_;
    cv::Mat protection_mask_;
    std::vector<iris_sdk::IrisLandmark> fake_landmarks_;
};

//=============================================================================
// Brightness Tests (apply() → applyBrightness 생존 검증)
// P8-W2-C: SkinSmoothingV2_* / SoftFocusV2_* 케이스 제거(곁가지 효과 삭제).
//   이 테스트들은 이름은 V2였으나 실제로는 apply()→applyFullFrame→V1 함수
//   (applySkinSmoothing/applySoftFocus, 삭제됨)를 경유했다.
//=============================================================================

TEST_F(NewFilterEffectsTest, BrightnessV2_IncreasesLChannel) {
    CPUBeautyBackend backend;
    backend.initialize();

    cv::Mat original = test_image_.clone();
    cv::Mat processed = test_image_.clone();

    double original_L = calculateLabChannel(original, 0);

    BeautyFilterConfigV2 config{};
    config.enabled = true;
    config.intensity = 1.0f;
    config.smoothing = 0.0f;
    config.brightness = 1.3f;  // 30% brighter
    config.softFocus = 0.0f;
    config.whitening = 0.0f;
    config.colorBalance = 0.0f;
    config.useGpu = false;
    config.roiOnly = false;

    backend.apply(processed.data, processed.cols, processed.rows,
                  IRIS_FORMAT_BGR, config, nullptr);

    double processed_L = calculateLabChannel(processed, 0);

    // L channel should increase
    EXPECT_GT(processed_L, original_L);
}

TEST_F(NewFilterEffectsTest, BrightnessV2_PreservesHighlights) {
    CPUBeautyBackend backend;
    backend.initialize();

    // Create image with highlight region
    cv::Mat bright_image = cv::Mat(100, 100, CV_8UC3, cv::Scalar(180, 200, 220));
    // Add very bright region (highlights)
    cv::rectangle(bright_image, cv::Rect(30, 30, 40, 40),
                 cv::Scalar(240, 250, 255), -1);

    cv::Mat processed = bright_image.clone();

    // Get original highlight values
    cv::Mat highlight_roi_orig = bright_image(cv::Rect(40, 40, 20, 20));
    double orig_highlight_L = calculateLabChannel(highlight_roi_orig, 0);

    BeautyFilterConfigV2 config{};
    config.enabled = true;
    config.intensity = 1.0f;
    config.smoothing = 0.0f;
    config.brightness = 1.4f;  // 40% brighter
    config.softFocus = 0.0f;
    config.whitening = 0.0f;
    config.colorBalance = 0.0f;
    config.useGpu = false;
    config.roiOnly = false;

    backend.apply(processed.data, processed.cols, processed.rows,
                  IRIS_FORMAT_BGR, config, nullptr);

    // Get processed highlight values
    cv::Mat highlight_roi_proc = processed(cv::Rect(40, 40, 20, 20));
    double proc_highlight_L = calculateLabChannel(highlight_roi_proc, 0);

    // Highlight protection: bright areas should not clip to 255
    // The increase should be smaller than linear scaling would suggest
    double linear_increase = orig_highlight_L * 1.4;
    double actual_increase = proc_highlight_L;

    // With highlight protection, actual increase should be less than linear
    // (or at most clamped to 255)
    EXPECT_LE(actual_increase, 255.0);
    EXPECT_GE(actual_increase, orig_highlight_L);  // Should still increase
}

TEST_F(NewFilterEffectsTest, BrightnessV2_DecreasesWhenBelow1) {
    CPUBeautyBackend backend;
    backend.initialize();

    cv::Mat original = test_image_.clone();
    cv::Mat processed = test_image_.clone();

    double original_L = calculateLabChannel(original, 0);

    BeautyFilterConfigV2 config{};
    config.enabled = true;
    config.intensity = 1.0f;
    config.smoothing = 0.0f;
    config.brightness = 0.7f;  // 30% darker
    config.softFocus = 0.0f;
    config.whitening = 0.0f;
    config.colorBalance = 0.0f;
    config.useGpu = false;
    config.roiOnly = false;

    backend.apply(processed.data, processed.cols, processed.rows,
                  IRIS_FORMAT_BGR, config, nullptr);

    double processed_L = calculateLabChannel(processed, 0);

    // L channel should decrease
    EXPECT_LT(processed_L, original_L);
}

// P8-W2-C: Whitening_* / ColorBalance_* 케이스 제거(곁가지 색보정 효과 삭제).
//   Wrinkle Removal 테스트는 원래 코멘트-only(applyWrinkleRemoval가 dead였음)였고
//   해당 함수 삭제로 코멘트 블록도 제거.

//=============================================================================
// Integration Tests
//=============================================================================

TEST_F(NewFilterEffectsTest, FullPipeline_AllEffectsCombined) {
    CPUBeautyBackend backend;
    backend.initialize();

    cv::Mat original = test_image_.clone();
    cv::Mat processed = test_image_.clone();

    BeautyFilterConfigV2 config{};
    // P8-W2-C: 곁가지 효과(smoothing/softFocus/whitening/colorBalance) 삭제 후
    //   CPU 파이프라인 생존 효과는 brightness 뿐 → brightness-only 스모크 테스트로 축소.
    config.enabled = true;
    config.intensity = 0.7f;
    config.smoothing = 0.0f;
    config.brightness = 1.1f;
    config.softFocus = 0.0f;
    config.whitening = 0.0f;
    config.colorBalance = 0.0f;
    config.wrinkleRemove = 0.0f;
    config.useGpu = false;
    config.roiOnly = false;
    config.protectEyes = false;
    config.protectLips = false;
    config.downscaleFactor = 1;

    IrisSdkError result = backend.apply(
        processed.data, processed.cols, processed.rows,
        IRIS_FORMAT_BGR, config, nullptr);

    EXPECT_EQ(result, IRIS_SDK_OK);

    // Image should be modified
    double diff = cv::norm(original, processed);
    EXPECT_GT(diff, 0.0);

    // Image should still be valid (no NaN or extreme values)
    cv::Mat gray;
    cv::cvtColor(processed, gray, cv::COLOR_BGR2GRAY);
    double min_val, max_val;
    cv::minMaxLoc(gray, &min_val, &max_val);

    EXPECT_GE(min_val, 0.0);
    EXPECT_LE(max_val, 255.0);
}

TEST_F(NewFilterEffectsTest, FullPipeline_DisabledConfig_NoChange) {
    CPUBeautyBackend backend;
    backend.initialize();

    cv::Mat original = test_image_.clone();
    cv::Mat processed = test_image_.clone();

    BeautyFilterConfigV2 config{};
    config.enabled = false;  // Disabled

    IrisSdkError result = backend.apply(
        processed.data, processed.cols, processed.rows,
        IRIS_FORMAT_BGR, config, nullptr);

    EXPECT_EQ(result, IRIS_SDK_OK);

    // No change when disabled
    double diff = cv::norm(original, processed);
    EXPECT_DOUBLE_EQ(diff, 0.0);
}

TEST_F(NewFilterEffectsTest, FullPipeline_MultipleFormats) {
    CPUBeautyBackend backend;
    backend.initialize();

    // P8-W2-C: smoothing(삭제됨) 대신 생존 효과 brightness로 포맷 변환 경로를 검증.
    BeautyFilterConfigV2 config{};
    config.enabled = true;
    config.intensity = 1.0f;
    config.smoothing = 0.0f;
    config.brightness = 1.1f;
    config.useGpu = false;
    config.roiOnly = false;

    // Test RGBA format
    cv::Mat rgba;
    cv::cvtColor(test_image_, rgba, cv::COLOR_BGR2RGBA);
    cv::Mat rgba_copy = rgba.clone();

    IrisSdkError result_rgba = backend.apply(
        rgba_copy.data, rgba_copy.cols, rgba_copy.rows,
        IRIS_FORMAT_RGBA, config, nullptr);

    EXPECT_EQ(result_rgba, IRIS_SDK_OK);

    // Test RGB format
    cv::Mat rgb;
    cv::cvtColor(test_image_, rgb, cv::COLOR_BGR2RGB);
    cv::Mat rgb_copy = rgb.clone();

    IrisSdkError result_rgb = backend.apply(
        rgb_copy.data, rgb_copy.cols, rgb_copy.rows,
        IRIS_FORMAT_RGB, config, nullptr);

    EXPECT_EQ(result_rgb, IRIS_SDK_OK);
}

// P8-W2-C: Performance_SmoothingV2_ReasonableTime 제거
//   (삭제된 Bilateral skin smoothing 효과의 성능 측정이라 의미 소멸).

