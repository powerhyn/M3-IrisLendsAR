/**
 * @file test_temporal_stabilizer.cpp
 * @brief TemporalStabilizer 단위 테스트
 *
 * W1-01: Smoothing (지터 감소, 빠른 움직임 추종)
 * W1-02: Confidence Hysteresis, Visibility Fade, Dropout Hold
 * W1-03: Outlier Rejection, Blink Detection
 */

#include <gtest/gtest.h>
#include "iris_sdk/temporal_stabilizer.h"

#include <cmath>
#include <numeric>
#include <vector>

using namespace iris_sdk;

// ============================================================================
// Helper
// ============================================================================

static IrisResult makeResult(bool detected, float confidence,
                             float lx, float ly, float lr,
                             float rx, float ry, float rr,
                             int64_t ts_ms = 0) {
    IrisResult r{};
    r.detected = detected;
    r.left_detected = detected;
    r.right_detected = detected;
    r.confidence = confidence;
    r.left_iris[0] = {lx, ly, 0.0f, 1.0f};
    r.left_radius = lr;
    r.right_iris[0] = {rx, ry, 0.0f, 1.0f};
    r.right_radius = rr;
    r.face_mesh_valid = false;
    r.timestamp_ms = ts_ms;
    r.frame_width = 640;
    r.frame_height = 480;
    return r;
}

/// Set face_mesh EAR landmarks for blink testing.
/// Left eye: 33, 160, 158, 133, 153, 144
/// Right eye: 362, 385, 387, 263, 373, 380
static void setEARLandmarks(IrisResult& r, bool blinking) {
    r.face_mesh_valid = true;

    // Left eye horizontal: 33 and 133
    r.face_mesh[33]  = {0.3f, 0.4f, 0.0f, 1.0f};
    r.face_mesh[133] = {0.4f, 0.4f, 0.0f, 1.0f};

    // Right eye horizontal: 362 and 263
    r.face_mesh[362] = {0.6f, 0.4f, 0.0f, 1.0f};
    r.face_mesh[263] = {0.7f, 0.4f, 0.0f, 1.0f};

    // EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    // p2=160, p6=144, p3=158, p5=153 for left eye
    // Keep p2/p6 and p3/p5 at same x to control vertical distance cleanly.
    float mid = 0.4f;
    float cx_left = 0.35f;
    float cx_right = 0.65f;

    if (blinking) {
        // Tiny vertical gap -> EAR ~ 0.002 / 0.1 = 0.02 (well below 0.2)
        float offset = 0.001f;
        r.face_mesh[160] = {cx_left, mid + offset, 0.0f, 1.0f};
        r.face_mesh[144] = {cx_left, mid - offset, 0.0f, 1.0f};
        r.face_mesh[158] = {cx_left, mid + offset, 0.0f, 1.0f};
        r.face_mesh[153] = {cx_left, mid - offset, 0.0f, 1.0f};

        r.face_mesh[385] = {cx_right, mid + offset, 0.0f, 1.0f};
        r.face_mesh[380] = {cx_right, mid - offset, 0.0f, 1.0f};
        r.face_mesh[387] = {cx_right, mid + offset, 0.0f, 1.0f};
        r.face_mesh[373] = {cx_right, mid - offset, 0.0f, 1.0f};
    } else {
        // Normal open eye -> EAR ~ 0.03 / 0.1 = 0.3
        float offset = 0.015f;
        r.face_mesh[160] = {cx_left, mid + offset, 0.0f, 1.0f};
        r.face_mesh[144] = {cx_left, mid - offset, 0.0f, 1.0f};
        r.face_mesh[158] = {cx_left, mid + offset, 0.0f, 1.0f};
        r.face_mesh[153] = {cx_left, mid - offset, 0.0f, 1.0f};

        r.face_mesh[385] = {cx_right, mid + offset, 0.0f, 1.0f};
        r.face_mesh[380] = {cx_right, mid - offset, 0.0f, 1.0f};
        r.face_mesh[387] = {cx_right, mid + offset, 0.0f, 1.0f};
        r.face_mesh[373] = {cx_right, mid - offset, 0.0f, 1.0f};
    }
}

/// Feed N identical stable frames to bring stabilizer into steady tracking state.
static void warmUp(TemporalStabilizer& stab, int n = 10) {
    for (int i = 0; i < n; ++i) {
        double t = 0.1 + i * 0.033;
        auto r = makeResult(true, 0.9f, 0.5f, 0.5f, 10.0f,
                            0.7f, 0.5f, 10.0f,
                            static_cast<int64_t>(t * 1000));
        stab.stabilize(r, t);
    }
}

// ============================================================================
// 1. Construction & Defaults
// ============================================================================

TEST(TemporalStabilizerTest, DefaultConstruction) {
    TemporalStabilizer stab;
    EXPECT_TRUE(stab.isEnabled());

    const auto& cfg = stab.config();
    EXPECT_NEAR(cfg.iris_min_cutoff, 4.0f, 1e-3);
    EXPECT_NEAR(cfg.iris_beta, 15.0f, 1e-3);
    EXPECT_NEAR(cfg.confidence_low_threshold, 0.3f, 1e-3);
    EXPECT_NEAR(cfg.confidence_high_threshold, 0.6f, 1e-3);
    EXPECT_EQ(cfg.confidence_low_frames, 3);
    EXPECT_NEAR(cfg.fade_in_ms, 100.0f, 1e-3);
    EXPECT_NEAR(cfg.fade_out_ms, 200.0f, 1e-3);
    EXPECT_EQ(cfg.hold_frames, 5);
    EXPECT_NEAR(cfg.blink_ear_threshold, 0.2f, 1e-3);
}

TEST(TemporalStabilizerTest, CustomConfig) {
    StabilizerConfig cfg;
    cfg.hold_frames = 10;
    cfg.fade_in_ms = 50.0f;
    TemporalStabilizer stab(cfg);

    EXPECT_EQ(stab.config().hold_frames, 10);
    EXPECT_NEAR(stab.config().fade_in_ms, 50.0f, 1e-3);
}

// ============================================================================
// 2. Enable / Disable
// ============================================================================

TEST(TemporalStabilizerTest, DisabledPassthrough) {
    TemporalStabilizer stab;
    stab.setEnabled(false);
    EXPECT_FALSE(stab.isEnabled());

    auto raw = makeResult(true, 0.9f, 0.5f, 0.5f, 10.0f,
                          0.7f, 0.5f, 10.0f, 100);
    auto out = stab.stabilize(raw, 0.1);

    // When disabled, stabilized should equal raw
    EXPECT_NEAR(out.stabilized.left_iris[0].x, 0.5f, 1e-3);
    EXPECT_NEAR(out.stabilized.left_iris[0].y, 0.5f, 1e-3);
    EXPECT_NEAR(out.stabilized.right_iris[0].x, 0.7f, 1e-3);
    EXPECT_NEAR(out.visibility, 1.0f, 1e-3);
}

TEST(TemporalStabilizerTest, ReEnableAfterDisable) {
    TemporalStabilizer stab;
    stab.setEnabled(false);
    stab.setEnabled(true);
    EXPECT_TRUE(stab.isEnabled());
}

// ============================================================================
// 3. Reset
// ============================================================================

TEST(TemporalStabilizerTest, ResetClearsState) {
    TemporalStabilizer stab;
    warmUp(stab, 10);

    stab.reset();

    // After reset, first detection should behave like fresh start
    // Feed a no-detection frame: should not hold (no prior valid data)
    auto r = makeResult(false, 0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 0.0f, 0.0f, 1000);
    auto out = stab.stabilize(r, 1.0);
    EXPECT_FALSE(out.is_held);
    EXPECT_FALSE(out.stabilized.detected);
}

// ============================================================================
// 4. Smoothing (W1-01)
// ============================================================================

TEST(TemporalStabilizerTest, JitterReduction) {
    TemporalStabilizer stab;

    // Feed stable frames with small noise
    std::vector<float> raw_x, stabilized_x;
    const float base_x = 0.5f;

    for (int i = 0; i < 30; ++i) {
        float noise = (i % 2 == 0) ? 0.005f : -0.005f;  // jitter +/-0.005
        float x = base_x + noise;
        double t = 0.1 + i * 0.033;
        auto r = makeResult(true, 0.9f, x, 0.5f, 10.0f,
                            0.7f, 0.5f, 10.0f,
                            static_cast<int64_t>(t * 1000));
        auto out = stab.stabilize(r, t);

        if (i >= 5) {  // skip warm-up transient
            raw_x.push_back(x);
            stabilized_x.push_back(out.stabilized.left_iris[0].x);
        }
    }

    // Compute variance of raw vs stabilized
    auto variance = [](const std::vector<float>& v) {
        float mean = std::accumulate(v.begin(), v.end(), 0.0f) /
                     static_cast<float>(v.size());
        float var = 0.0f;
        for (float x : v) var += (x - mean) * (x - mean);
        return var / static_cast<float>(v.size());
    };

    float raw_var = variance(raw_x);
    float stab_var = variance(stabilized_x);

    // Stabilized should have less variance than raw
    EXPECT_LT(stab_var, raw_var);
}

TEST(TemporalStabilizerTest, FastMovementTracking) {
    TemporalStabilizer stab;
    warmUp(stab, 5);

    // Move from 0.5 to 0.8 rapidly
    double t = 0.5;
    for (int i = 0; i < 10; ++i) {
        t += 0.033;
        auto r = makeResult(true, 0.9f, 0.8f, 0.5f, 10.0f,
                            0.9f, 0.5f, 10.0f,
                            static_cast<int64_t>(t * 1000));
        stab.stabilize(r, t);
    }

    // After several frames at the new position, stabilized should converge
    t += 0.033;
    auto r = makeResult(true, 0.9f, 0.8f, 0.5f, 10.0f,
                        0.9f, 0.5f, 10.0f,
                        static_cast<int64_t>(t * 1000));
    auto out = stab.stabilize(r, t);

    // Should be close to 0.8 (within 0.05 tolerance for filter lag)
    EXPECT_NEAR(out.stabilized.left_iris[0].x, 0.8f, 0.05f);
}

// ============================================================================
// 5. Confidence Hysteresis (W1-02)
// ============================================================================

TEST(TemporalStabilizerTest, LowConfidenceThreeFramesLosesTracking) {
    TemporalStabilizer stab;
    warmUp(stab, 10);

    // Feed 3 low-confidence frames (below 0.3 threshold)
    double t = 1.0;
    StabilizedResult out;
    for (int i = 0; i < 3; ++i) {
        t += 0.033;
        auto r = makeResult(true, 0.1f, 0.5f, 0.5f, 10.0f,
                            0.7f, 0.5f, 10.0f,
                            static_cast<int64_t>(t * 1000));
        out = stab.stabilize(r, t);
    }

    // After 3 low-confidence frames, tracking should be lost
    // Next frame should reflect loss (is_held or not detected in stabilized)
    t += 0.033;
    auto r = makeResult(true, 0.1f, 0.5f, 0.5f, 10.0f,
                        0.7f, 0.5f, 10.0f,
                        static_cast<int64_t>(t * 1000));
    out = stab.stabilize(r, t);

    // Should be in hold or lost state
    // Visibility should be fading
    EXPECT_LT(out.visibility, 1.0f);
}

TEST(TemporalStabilizerTest, HighConfidenceResetsLowCount) {
    TemporalStabilizer stab;
    warmUp(stab, 10);

    double t = 1.0;
    // Feed 2 low-confidence frames (not enough to lose tracking)
    for (int i = 0; i < 2; ++i) {
        t += 0.033;
        auto r = makeResult(true, 0.1f, 0.5f, 0.5f, 10.0f,
                            0.7f, 0.5f, 10.0f,
                            static_cast<int64_t>(t * 1000));
        stab.stabilize(r, t);
    }

    // High confidence frame resets the count
    t += 0.033;
    auto r = makeResult(true, 0.9f, 0.5f, 0.5f, 10.0f,
                        0.7f, 0.5f, 10.0f,
                        static_cast<int64_t>(t * 1000));
    auto out = stab.stabilize(r, t);
    EXPECT_NEAR(out.visibility, 1.0f, 0.1f);

    // Now 2 more low-confidence frames should NOT lose tracking
    for (int i = 0; i < 2; ++i) {
        t += 0.033;
        r = makeResult(true, 0.1f, 0.5f, 0.5f, 10.0f,
                       0.7f, 0.5f, 10.0f,
                       static_cast<int64_t>(t * 1000));
        out = stab.stabilize(r, t);
    }

    // Should still be tracking (visibility near 1)
    EXPECT_GT(out.visibility, 0.5f);
}

TEST(TemporalStabilizerTest, NeedHighConfidenceToReEnterTracking) {
    TemporalStabilizer stab;
    warmUp(stab, 10);

    // Lose tracking: 4 low-confidence frames
    double t = 1.0;
    for (int i = 0; i < 4; ++i) {
        t += 0.033;
        auto r = makeResult(true, 0.1f, 0.5f, 0.5f, 10.0f,
                            0.7f, 0.5f, 10.0f,
                            static_cast<int64_t>(t * 1000));
        stab.stabilize(r, t);
    }

    // Wait for hold to expire
    for (int i = 0; i < 10; ++i) {
        t += 0.033;
        auto r = makeResult(false, 0.0f, 0.0f, 0.0f, 0.0f,
                            0.0f, 0.0f, 0.0f,
                            static_cast<int64_t>(t * 1000));
        stab.stabilize(r, t);
    }

    // Medium confidence (0.4) should NOT re-enter tracking
    t += 0.033;
    auto r = makeResult(true, 0.4f, 0.5f, 0.5f, 10.0f,
                        0.7f, 0.5f, 10.0f,
                        static_cast<int64_t>(t * 1000));
    auto out = stab.stabilize(r, t);
    EXPECT_LT(out.visibility, 0.5f);

    // High confidence (>=0.6) should re-enter tracking
    t += 0.033;
    r = makeResult(true, 0.7f, 0.5f, 0.5f, 10.0f,
                   0.7f, 0.5f, 10.0f,
                   static_cast<int64_t>(t * 1000));
    out = stab.stabilize(r, t);

    // At fade_start_time, elapsed=0 so visibility=0. Need another frame.
    t += 0.033;
    r = makeResult(true, 0.7f, 0.5f, 0.5f, 10.0f,
                   0.7f, 0.5f, 10.0f,
                   static_cast<int64_t>(t * 1000));
    out = stab.stabilize(r, t);

    // Visibility should start climbing (fade-in, 33ms / 100ms ≈ 0.33)
    EXPECT_GT(out.visibility, 0.0f);
}

// ============================================================================
// 6. Visibility Fade (W1-02)
// ============================================================================

TEST(TemporalStabilizerTest, FadeInFromZero) {
    TemporalStabilizer stab;

    // First detection: visibility starts fading in from 0
    double t = 0.0;
    auto r = makeResult(true, 0.9f, 0.5f, 0.5f, 10.0f,
                        0.7f, 0.5f, 10.0f, 0);
    auto out = stab.stabilize(r, t);

    // Immediately after first detection, visibility should be low or partial
    // (fade_in_ms = 100ms, so at t=0 it starts)
    float initial_vis = out.visibility;

    // After 50ms (~half fade-in), should be partial
    t = 0.05;
    r = makeResult(true, 0.9f, 0.5f, 0.5f, 10.0f,
                   0.7f, 0.5f, 10.0f, 50);
    out = stab.stabilize(r, t);
    EXPECT_GT(out.visibility, initial_vis);
    EXPECT_LT(out.visibility, 1.0f);
}

TEST(TemporalStabilizerTest, FadeInCompletes) {
    TemporalStabilizer stab;

    // Feed frames over 150ms (> fade_in_ms=100ms)
    double t = 0.0;
    StabilizedResult out;
    for (int i = 0; i < 10; ++i) {
        t = i * 0.020;  // 20ms intervals, total 180ms
        auto r = makeResult(true, 0.9f, 0.5f, 0.5f, 10.0f,
                            0.7f, 0.5f, 10.0f,
                            static_cast<int64_t>(t * 1000));
        out = stab.stabilize(r, t);
    }

    // After 180ms, fade-in (100ms) should be complete
    EXPECT_NEAR(out.visibility, 1.0f, 0.05f);
}

TEST(TemporalStabilizerTest, FadeOutOnDetectionLoss) {
    TemporalStabilizer stab;
    warmUp(stab, 10);

    // Lose detection
    double t = 1.0;
    auto r = makeResult(false, 0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 0.0f, 0.0f, 1000);
    auto out = stab.stabilize(r, t);

    // Should start fading out but not immediately zero
    float vis_at_loss = out.visibility;

    // After 100ms (half of fade_out_ms=200ms)
    t = 1.1;
    r = makeResult(false, 0.0f, 0.0f, 0.0f, 0.0f,
                   0.0f, 0.0f, 0.0f, 1100);
    out = stab.stabilize(r, t);
    EXPECT_LT(out.visibility, vis_at_loss);

    // After 300ms (> fade_out_ms=200ms), should be near zero
    t = 1.3;
    r = makeResult(false, 0.0f, 0.0f, 0.0f, 0.0f,
                   0.0f, 0.0f, 0.0f, 1300);
    out = stab.stabilize(r, t);
    EXPECT_NEAR(out.visibility, 0.0f, 0.05f);
}

// ============================================================================
// 7. Dropout Hold (W1-02)
// ============================================================================

TEST(TemporalStabilizerTest, HoldLastValidOnDropout) {
    TemporalStabilizer stab;
    warmUp(stab, 10);

    // Last valid position at (0.5, 0.5)
    double t = 1.0;

    // Drop detection for 3 frames (within hold_frames=5)
    for (int i = 1; i <= 3; ++i) {
        t += 0.033;
        auto r = makeResult(false, 0.0f, 0.0f, 0.0f, 0.0f,
                            0.0f, 0.0f, 0.0f,
                            static_cast<int64_t>(t * 1000));
        auto out = stab.stabilize(r, t);

        EXPECT_TRUE(out.is_held) << "Frame " << i << " should be held";
        // Held coordinates should be near last valid
        EXPECT_NEAR(out.stabilized.left_iris[0].x, 0.5f, 0.05f);
    }
}

TEST(TemporalStabilizerTest, HoldExpiresAfterMaxFrames) {
    TemporalStabilizer stab;
    warmUp(stab, 10);

    double t = 1.0;

    // Drop detection for more than hold_frames (5) frames
    StabilizedResult out;
    for (int i = 0; i < 8; ++i) {
        t += 0.033;
        auto r = makeResult(false, 0.0f, 0.0f, 0.0f, 0.0f,
                            0.0f, 0.0f, 0.0f,
                            static_cast<int64_t>(t * 1000));
        out = stab.stabilize(r, t);
    }

    // After hold expires, is_held should be false and detection lost
    EXPECT_FALSE(out.is_held);
    EXPECT_FALSE(out.stabilized.detected);
}

// ============================================================================
// 8. Outlier Rejection (W1-03)
// ============================================================================

TEST(TemporalStabilizerTest, SingleFrameSpikeRejected) {
    // Outlier threshold = radius * multiplier (2.0).
    // Use small radius (0.02) so threshold = 0.04 in normalized coords.
    // Warm up at (0.5, 0.5), then spike to (0.6, 0.6) which is dist=0.14 > 0.04.
    TemporalStabilizer stab;

    double t = 0.0;
    for (int i = 0; i < 15; ++i) {
        t += 0.033;
        auto r = makeResult(true, 0.9f, 0.5f, 0.5f, 0.02f,
                            0.7f, 0.5f, 0.02f,
                            static_cast<int64_t>(t * 1000));
        stab.stabilize(r, t);
    }

    // Single-frame spike: jump of ~0.14 > threshold 0.04
    t += 0.033;
    auto spike = makeResult(true, 0.9f, 0.6f, 0.6f, 0.02f,
                            0.8f, 0.6f, 0.02f,
                            static_cast<int64_t>(t * 1000));
    auto out = stab.stabilize(spike, t);

    // Should reject spike and hold near previous position
    EXPECT_NEAR(out.stabilized.left_iris[0].x, 0.5f, 0.02f);

    // Return to normal position
    t += 0.033;
    auto normal = makeResult(true, 0.9f, 0.5f, 0.5f, 0.02f,
                             0.7f, 0.5f, 0.02f,
                             static_cast<int64_t>(t * 1000));
    out = stab.stabilize(normal, t);
    EXPECT_NEAR(out.stabilized.left_iris[0].x, 0.5f, 0.02f);
}

TEST(TemporalStabilizerTest, ConsecutiveSpikesAccepted) {
    // Use small radius so outlier detection fires
    TemporalStabilizer stab;

    double t = 0.0;
    for (int i = 0; i < 15; ++i) {
        t += 0.033;
        auto r = makeResult(true, 0.9f, 0.5f, 0.5f, 0.02f,
                            0.7f, 0.5f, 0.02f,
                            static_cast<int64_t>(t * 1000));
        stab.stabilize(r, t);
    }

    // Two consecutive frames at new position (outlier_confirm_frames=2)
    // Jump of ~0.14 > threshold 0.04
    for (int i = 0; i < 2; ++i) {
        t += 0.033;
        auto r = makeResult(true, 0.9f, 0.6f, 0.6f, 0.02f,
                            0.8f, 0.6f, 0.02f,
                            static_cast<int64_t>(t * 1000));
        stab.stabilize(r, t);
    }

    // Feed more frames at new position so filter converges
    for (int i = 0; i < 10; ++i) {
        t += 0.033;
        auto r = makeResult(true, 0.9f, 0.6f, 0.6f, 0.02f,
                            0.8f, 0.6f, 0.02f,
                            static_cast<int64_t>(t * 1000));
        stab.stabilize(r, t);
    }

    t += 0.033;
    auto r = makeResult(true, 0.9f, 0.6f, 0.6f, 0.02f,
                        0.8f, 0.6f, 0.02f,
                        static_cast<int64_t>(t * 1000));
    auto out = stab.stabilize(r, t);

    // Should have accepted the new position and converged
    EXPECT_NEAR(out.stabilized.left_iris[0].x, 0.6f, 0.05f);
}

// ============================================================================
// 9. Blink Detection (W1-03)
// ============================================================================

TEST(TemporalStabilizerTest, BlinkHoldsIrisCoordinates) {
    TemporalStabilizer stab;

    // Warm up with open eyes and face mesh
    double t = 0.0;
    for (int i = 0; i < 15; ++i) {
        t += 0.033;
        auto r = makeResult(true, 0.9f, 0.5f, 0.5f, 10.0f,
                            0.7f, 0.5f, 10.0f,
                            static_cast<int64_t>(t * 1000));
        setEARLandmarks(r, false);  // eyes open
        stab.stabilize(r, t);
    }

    // Blink frame: EAR below threshold
    t += 0.033;
    auto blink = makeResult(true, 0.9f, 0.5f, 0.8f, 10.0f,
                            0.7f, 0.8f, 10.0f,
                            static_cast<int64_t>(t * 1000));
    setEARLandmarks(blink, true);  // eyes closed
    auto out = stab.stabilize(blink, t);

    // During blink, iris coordinates should be held near last valid (0.5, 0.5)
    // not the blink-frame raw value (0.5, 0.8)
    EXPECT_NEAR(out.stabilized.left_iris[0].y, 0.5f, 0.1f);
}

TEST(TemporalStabilizerTest, EyeOpenResumesNormalProcessing) {
    TemporalStabilizer stab;

    // Warm up
    double t = 0.0;
    for (int i = 0; i < 15; ++i) {
        t += 0.033;
        auto r = makeResult(true, 0.9f, 0.5f, 0.5f, 10.0f,
                            0.7f, 0.5f, 10.0f,
                            static_cast<int64_t>(t * 1000));
        setEARLandmarks(r, false);
        stab.stabilize(r, t);
    }

    // Blink for 3 frames
    for (int i = 0; i < 3; ++i) {
        t += 0.033;
        auto r = makeResult(true, 0.9f, 0.5f, 0.8f, 10.0f,
                            0.7f, 0.8f, 10.0f,
                            static_cast<int64_t>(t * 1000));
        setEARLandmarks(r, true);
        stab.stabilize(r, t);
    }

    // Open eyes at new position
    for (int i = 0; i < 10; ++i) {
        t += 0.033;
        auto r = makeResult(true, 0.9f, 0.55f, 0.5f, 10.0f,
                            0.75f, 0.5f, 10.0f,
                            static_cast<int64_t>(t * 1000));
        setEARLandmarks(r, false);
        stab.stabilize(r, t);
    }

    // Should resume tracking the new position
    t += 0.033;
    auto r = makeResult(true, 0.9f, 0.55f, 0.5f, 10.0f,
                        0.75f, 0.5f, 10.0f,
                        static_cast<int64_t>(t * 1000));
    setEARLandmarks(r, false);
    auto out = stab.stabilize(r, t);

    EXPECT_NEAR(out.stabilized.left_iris[0].x, 0.55f, 0.05f);
}

// ============================================================================
// Additional edge cases
// ============================================================================

TEST(TemporalStabilizerTest, SetConfigResetsState) {
    TemporalStabilizer stab;
    warmUp(stab, 10);

    StabilizerConfig new_cfg;
    new_cfg.hold_frames = 8;
    stab.setConfig(new_cfg);

    EXPECT_EQ(stab.config().hold_frames, 8);

    // State should be reset: no hold data
    auto r = makeResult(false, 0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 0.0f, 0.0f, 2000);
    auto out = stab.stabilize(r, 2.0);
    EXPECT_FALSE(out.is_held);
}

TEST(TemporalStabilizerTest, RawFieldPreserved) {
    TemporalStabilizer stab;

    auto r = makeResult(true, 0.85f, 0.3f, 0.4f, 12.0f,
                        0.6f, 0.4f, 11.0f, 100);
    auto out = stab.stabilize(r, 0.1);

    // Raw field should always contain the original input
    EXPECT_NEAR(out.raw.left_iris[0].x, 0.3f, 1e-5);
    EXPECT_NEAR(out.raw.left_iris[0].y, 0.4f, 1e-5);
    EXPECT_NEAR(out.raw.confidence, 0.85f, 1e-5);
    EXPECT_TRUE(out.raw.detected);
}

TEST(TemporalStabilizerTest, TimestampProgression) {
    TemporalStabilizer stab;
    warmUp(stab, 5);

    double t = 0.5;
    auto r = makeResult(true, 0.9f, 0.5f, 0.5f, 10.0f,
                        0.7f, 0.5f, 10.0f, 500);
    auto out = stab.stabilize(r, t);

    EXPECT_GE(out.last_valid_ms, 0);
}
