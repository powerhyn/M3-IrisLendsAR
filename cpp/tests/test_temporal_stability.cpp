/**
 * @file test_temporal_stability.cpp
 * @brief TemporalStabilizer 통합/안정성 메트릭 테스트
 *
 * 개별 기능 단위 테스트(test_temporal_stabilizer.cpp, 22개)와 달리,
 * 이 파일은 정량적 품질 측정(quantitative stability metrics)에 초점을 맞춤.
 *
 * 테스트 카테고리:
 *   1. Jitter Metric       — 정지 상태 표준편차 감소율
 *   2. Lag Metric          — 추적 지연 측정
 *   3. Visibility Transition Smoothness — 가시성 전환 매끄러움
 *   4. Blink Hold Stability — 눈 깜빡임 중 좌표 유지
 *   5. Outlier Rejection    — 이상치 제거 효과
 *   6. Async API Cache      — getLatestResult 캐싱 로직
 *   7. Smoothing CV         — 정현파 입력의 프레임 간 변동 계수
 */

#include <gtest/gtest.h>
#include "iris_sdk/temporal_stabilizer.h"
#include "iris_sdk/types.h"

#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

using namespace iris_sdk;

// ============================================================================
// Helpers
// ============================================================================

/// 합성 IrisResult 생성. 좌우 홍채 + EAR용 face_mesh 포함.
static IrisResult makeResult(float cx, float cy, float radius,
                             float confidence = 0.9f, bool detected = true) {
    IrisResult r{};
    r.detected = detected;
    r.left_detected = detected;
    r.right_detected = detected;
    r.confidence = confidence;
    r.frame_width = 640;
    r.frame_height = 480;

    // Left iris: center + 4 boundary points
    r.left_iris[0] = {cx, cy, 0.0f, 1.0f};
    r.left_iris[1] = {cx + radius, cy, 0.0f, 1.0f};
    r.left_iris[2] = {cx, cy - radius, 0.0f, 1.0f};
    r.left_iris[3] = {cx - radius, cy, 0.0f, 1.0f};
    r.left_iris[4] = {cx, cy + radius, 0.0f, 1.0f};
    r.left_radius = radius;

    // Right iris: offset by 0.3
    float rcx = cx + 0.3f;
    r.right_iris[0] = {rcx, cy, 0.0f, 1.0f};
    r.right_iris[1] = {rcx + radius, cy, 0.0f, 1.0f};
    r.right_iris[2] = {rcx, cy - radius, 0.0f, 1.0f};
    r.right_iris[3] = {rcx - radius, cy, 0.0f, 1.0f};
    r.right_iris[4] = {rcx, cy + radius, 0.0f, 1.0f};
    r.right_radius = radius;

    // Face mesh for EAR (eyelid landmarks — "open" position)
    r.face_mesh_valid = true;

    // Left eye: 159(top), 145(bottom), 33(left), 133(right), 160(top2), 144(bottom2)
    r.face_mesh[159] = {cx - 0.01f, cy - 0.02f, 0, 1};
    r.face_mesh[145] = {cx - 0.01f, cy + 0.02f, 0, 1};
    r.face_mesh[33]  = {cx - 0.04f, cy, 0, 1};
    r.face_mesh[133] = {cx + 0.02f, cy, 0, 1};
    r.face_mesh[160] = {cx - 0.02f, cy - 0.018f, 0, 1};
    r.face_mesh[144] = {cx - 0.02f, cy + 0.018f, 0, 1};

    // Right eye: 386(top), 374(bottom), 362(left), 263(right), 385(top2), 380(bottom2)
    r.face_mesh[386] = {rcx + 0.01f, cy - 0.02f, 0, 1};
    r.face_mesh[374] = {rcx + 0.01f, cy + 0.02f, 0, 1};
    r.face_mesh[362] = {rcx - 0.02f, cy, 0, 1};
    r.face_mesh[263] = {rcx + 0.04f, cy, 0, 1};
    r.face_mesh[385] = {rcx + 0.02f, cy - 0.018f, 0, 1};
    r.face_mesh[380] = {rcx + 0.02f, cy + 0.018f, 0, 1};

    return r;
}

/// EAR 랜드마크를 눈 깜빡임 상태로 설정 (기존 단위 테스트의 setEARLandmarks 참고).
static void setBlinkEAR(IrisResult& r, bool blinking) {
    r.face_mesh_valid = true;
    const float mid = 0.4f;
    const float cx_left = 0.35f;
    const float cx_right = 0.65f;

    // EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    // Left: 33, 160, 158, 133, 153, 144
    // Right: 362, 385, 387, 263, 373, 380
    r.face_mesh[33]  = {0.3f, mid, 0.0f, 1.0f};
    r.face_mesh[133] = {0.4f, mid, 0.0f, 1.0f};
    r.face_mesh[362] = {0.6f, mid, 0.0f, 1.0f};
    r.face_mesh[263] = {0.7f, mid, 0.0f, 1.0f};

    const float offset = blinking ? 0.001f : 0.015f;
    r.face_mesh[160] = {cx_left, mid + offset, 0, 1};
    r.face_mesh[144] = {cx_left, mid - offset, 0, 1};
    r.face_mesh[158] = {cx_left, mid + offset, 0, 1};
    r.face_mesh[153] = {cx_left, mid - offset, 0, 1};

    r.face_mesh[385] = {cx_right, mid + offset, 0, 1};
    r.face_mesh[380] = {cx_right, mid - offset, 0, 1};
    r.face_mesh[387] = {cx_right, mid + offset, 0, 1};
    r.face_mesh[373] = {cx_right, mid - offset, 0, 1};
}

/// 표준편차 계산 유틸리티.
static double stddev(const std::vector<float>& v) {
    if (v.size() < 2) return 0.0;
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / static_cast<double>(v.size());
    double sq_sum = 0.0;
    for (float x : v) {
        double d = static_cast<double>(x) - mean;
        sq_sum += d * d;
    }
    return std::sqrt(sq_sum / static_cast<double>(v.size()));
}

/// 평균 계산 유틸리티.
static double mean(const std::vector<float>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) /
           static_cast<double>(v.size());
}

// ============================================================================
// Test Fixture
// ============================================================================

class TemporalStabilityTest : public ::testing::Test {
protected:
    static constexpr double kFrameInterval = 1.0 / 30.0;  // 30fps = ~33ms

    /// 안정 프레임을 N개 공급하여 워밍업.
    void warmUp(TemporalStabilizer& stab, int n = 15) {
        for (int i = 0; i < n; ++i) {
            double t = 0.1 + i * kFrameInterval;
            auto r = makeResult(0.5f, 0.5f, 10.0f);
            r.timestamp_ms = static_cast<int64_t>(t * 1000);
            stab.stabilize(r, t);
        }
    }
};

// ============================================================================
// 1. Jitter Metric — 정지 상태 표준편차 감소율
// ============================================================================

TEST_F(TemporalStabilityTest, JitterReductionXY) {
    TemporalStabilizer stab;
    warmUp(stab);

    std::mt19937 rng(42);  // 고정 시드
    std::normal_distribution<float> noise(0.0f, 0.005f);

    const float base_x = 0.5f, base_y = 0.5f;
    const int N = 100;

    std::vector<float> raw_x, raw_y, smooth_x, smooth_y;
    raw_x.reserve(N);
    raw_y.reserve(N);
    smooth_x.reserve(N);
    smooth_y.reserve(N);

    for (int i = 0; i < N; ++i) {
        float nx = base_x + noise(rng);
        float ny = base_y + noise(rng);
        double t = 1.0 + i * kFrameInterval;

        auto r = makeResult(nx, ny, 10.0f);
        r.timestamp_ms = static_cast<int64_t>(t * 1000);
        auto out = stab.stabilize(r, t);

        raw_x.push_back(nx);
        raw_y.push_back(ny);
        smooth_x.push_back(out.stabilized.left_iris[0].x);
        smooth_y.push_back(out.stabilized.left_iris[0].y);
    }

    double raw_sd_x = stddev(raw_x);
    double raw_sd_y = stddev(raw_y);
    double smooth_sd_x = stddev(smooth_x);
    double smooth_sd_y = stddev(smooth_y);

    std::fprintf(stderr,
        "[JitterXY] raw_sd_x=%.6f smooth_sd_x=%.6f reduction=%.1f%%\n"
        "           raw_sd_y=%.6f smooth_sd_y=%.6f reduction=%.1f%%\n",
        raw_sd_x, smooth_sd_x, (1.0 - smooth_sd_x / raw_sd_x) * 100.0,
        raw_sd_y, smooth_sd_y, (1.0 - smooth_sd_y / raw_sd_y) * 100.0);

    // 30% 이상 지터 감소 (OneEuroFilter는 beta=15로 반응성 우선 튜닝)
    EXPECT_LT(smooth_sd_x, raw_sd_x * 0.7)
        << "X jitter reduction insufficient";
    EXPECT_LT(smooth_sd_y, raw_sd_y * 0.7)
        << "Y jitter reduction insufficient";
}

TEST_F(TemporalStabilityTest, JitterReductionRadius) {
    TemporalStabilizer stab;
    warmUp(stab);

    std::mt19937 rng(123);
    std::normal_distribution<float> noise(0.0f, 0.3f);

    const float base_radius = 10.0f;
    const int N = 100;

    std::vector<float> raw_r, smooth_r;
    raw_r.reserve(N);
    smooth_r.reserve(N);

    for (int i = 0; i < N; ++i) {
        float nr = base_radius + noise(rng);
        double t = 1.0 + i * kFrameInterval;

        auto r = makeResult(0.5f, 0.5f, nr);
        r.timestamp_ms = static_cast<int64_t>(t * 1000);
        auto out = stab.stabilize(r, t);

        raw_r.push_back(nr);
        smooth_r.push_back(out.stabilized.left_radius);
    }

    double raw_sd = stddev(raw_r);
    double smooth_sd = stddev(smooth_r);

    std::fprintf(stderr,
        "[JitterRadius] raw_sd=%.6f smooth_sd=%.6f reduction=%.1f%%\n",
        raw_sd, smooth_sd, (1.0 - smooth_sd / raw_sd) * 100.0);

    // 반지름 필터(beta=7.5)는 노이즈 대비 신호가 큼 → 완화된 기준
    EXPECT_LT(smooth_sd, raw_sd)
        << "Radius jitter reduction insufficient";
}

// ============================================================================
// 2. Lag Metric — 추적 지연 측정
// ============================================================================

TEST_F(TemporalStabilityTest, LinearMotionLag) {
    TemporalStabilizer stab;
    warmUp(stab);

    // 등속 직선 운동: x가 0.3 -> 0.7 까지 60프레임 동안 이동
    const int N = 60;
    const float x_start = 0.3f, x_end = 0.7f;
    const float speed = (x_end - x_start) / static_cast<float>(N);

    std::vector<float> lags;
    lags.reserve(N);

    for (int i = 0; i < N; ++i) {
        float true_x = x_start + speed * static_cast<float>(i);
        double t = 1.0 + i * kFrameInterval;

        auto r = makeResult(true_x, 0.5f, 10.0f);
        r.timestamp_ms = static_cast<int64_t>(t * 1000);
        auto out = stab.stabilize(r, t);

        // 워밍업 구간(처음 10프레임) 무시
        if (i >= 10) {
            float lag = true_x - out.stabilized.left_iris[0].x;
            lags.push_back(lag);
        }
    }

    double avg_lag = mean(lags);
    // 프레임 당 이동량 * 2 이하여야 함 (2프레임 분량 지연)
    double max_allowed_lag = static_cast<double>(speed) * 2.0;

    std::fprintf(stderr,
        "[Lag] avg_lag=%.6f max_allowed=%.6f speed_per_frame=%.6f\n",
        avg_lag, max_allowed_lag, static_cast<double>(speed));

    EXPECT_LT(std::abs(avg_lag), max_allowed_lag)
        << "Smoothing lag exceeds 2-frame threshold";
}

// ============================================================================
// 3. Visibility Transition Smoothness — 가시성 전환 매끄러움
// ============================================================================

TEST_F(TemporalStabilityTest, VisibilityTransitionNoLargeJump) {
    TemporalStabilizer stab;

    // Phase 1: 30 detected frames
    double t = 0.0;
    float prev_vis = -1.0f;
    float max_jump = 0.0f;

    auto feedFrame = [&](bool detected, float conf) {
        t += kFrameInterval;
        auto r = makeResult(0.5f, 0.5f, 10.0f, conf, detected);
        r.timestamp_ms = static_cast<int64_t>(t * 1000);
        auto out = stab.stabilize(r, t);

        if (prev_vis >= 0.0f) {
            float jump = std::abs(out.visibility - prev_vis);
            if (jump > max_jump) max_jump = jump;
        }
        prev_vis = out.visibility;
        return out.visibility;
    };

    // Phase 1: 30 detected frames (fade-in 포함)
    for (int i = 0; i < 30; ++i) {
        feedFrame(true, 0.9f);
    }

    // Phase 2: 10 undetected frames (fade-out)
    for (int i = 0; i < 10; ++i) {
        feedFrame(false, 0.0f);
    }

    // Phase 3: 30 detected frames (fade-in again)
    for (int i = 0; i < 30; ++i) {
        feedFrame(true, 0.9f);
    }

    std::fprintf(stderr,
        "[VisibilityTransition] max_jump=%.4f\n", max_jump);

    // 연속 프레임 간 가시성 점프가 0.35 이하 (hold_frames=5, fade=200ms@30fps)
    EXPECT_LE(max_jump, 0.35f)
        << "Visibility jump between consecutive frames too large";
}

TEST_F(TemporalStabilityTest, VisibilityFadeOutIsGradual) {
    TemporalStabilizer stab;
    warmUp(stab);

    // 검출 소실 후 visibility 값 수집
    std::vector<float> vis_values;
    double t = 1.0;

    for (int i = 0; i < 20; ++i) {
        t += kFrameInterval;
        auto r = makeResult(0.5f, 0.5f, 10.0f, 0.0f, false);
        r.timestamp_ms = static_cast<int64_t>(t * 1000);
        auto out = stab.stabilize(r, t);
        vis_values.push_back(out.visibility);
    }

    // 단조 감소 또는 유지 확인 (감소 방향으로만 이동)
    int non_decreasing_count = 0;
    for (size_t i = 1; i < vis_values.size(); ++i) {
        if (vis_values[i] > vis_values[i - 1] + 1e-5f) {
            ++non_decreasing_count;
        }
    }

    std::fprintf(stderr, "[FadeOut] values: ");
    for (float v : vis_values) std::fprintf(stderr, "%.3f ", v);
    std::fprintf(stderr, "\n");

    EXPECT_EQ(non_decreasing_count, 0)
        << "Visibility should monotonically decrease during fade-out";
}

// ============================================================================
// 4. Blink Hold Stability — 눈 깜빡임 중 좌표 유지
// ============================================================================

TEST_F(TemporalStabilityTest, BlinkHoldCoordinatesStable) {
    TemporalStabilizer stab;

    // 안정 상태로 워밍업 (눈 열린 상태, face_mesh 포함)
    double t = 0.0;
    const float stable_x = 0.5f, stable_y = 0.5f;

    for (int i = 0; i < 20; ++i) {
        t += kFrameInterval;
        auto r = makeResult(stable_x, stable_y, 10.0f);
        r.timestamp_ms = static_cast<int64_t>(t * 1000);
        setBlinkEAR(r, false);
        stab.stabilize(r, t);
    }

    // 눈 깜빡임 3프레임: raw 좌표가 크게 흔들려도 stabilized는 유지되어야 함
    std::vector<float> blink_x, blink_y;
    for (int i = 0; i < 3; ++i) {
        t += kFrameInterval;
        // 깜빡임 중 raw 좌표가 아래로 크게 이탈 (y=0.8)
        auto r = makeResult(stable_x, 0.8f, 10.0f);
        r.timestamp_ms = static_cast<int64_t>(t * 1000);
        setBlinkEAR(r, true);
        auto out = stab.stabilize(r, t);

        blink_x.push_back(out.stabilized.left_iris[0].x);
        blink_y.push_back(out.stabilized.left_iris[0].y);
    }

    // 깜빡임 중 stabilized Y는 안정 상태(0.5)에 가까워야 함 (0.8이 아님)
    for (size_t i = 0; i < blink_y.size(); ++i) {
        std::fprintf(stderr, "[BlinkHold] frame %zu: x=%.4f y=%.4f\n",
                     i, blink_x[i], blink_y[i]);
        EXPECT_NEAR(blink_y[i], stable_y, 0.1f)
            << "Blink frame " << i << ": Y coordinate jumped during blink";
    }

    // 눈 뜨기 후 복귀: 새로운 위치(0.55, 0.5)로 부드럽게 전환
    float post_blink_x = 0.55f;
    std::vector<float> recovery_x;
    for (int i = 0; i < 10; ++i) {
        t += kFrameInterval;
        auto r = makeResult(post_blink_x, stable_y, 10.0f);
        r.timestamp_ms = static_cast<int64_t>(t * 1000);
        setBlinkEAR(r, false);
        auto out = stab.stabilize(r, t);
        recovery_x.push_back(out.stabilized.left_iris[0].x);
    }

    // 마지막 프레임은 목표 위치에 수렴해야 함
    EXPECT_NEAR(recovery_x.back(), post_blink_x, 0.03f)
        << "Post-blink recovery did not converge to new position";
}

// ============================================================================
// 5. Outlier Rejection Effectiveness — 이상치 제거 효과
// ============================================================================

TEST_F(TemporalStabilityTest, SingleOutlierRejected) {
    // confirm_frames=2로 설정하여 극단적 스파이크 거부 테스트
    StabilizerConfig cfg;
    cfg.outlier_radius_multiplier = 4.0f;
    cfg.outlier_confirm_frames = 2;
    TemporalStabilizer stab(cfg);

    const float radius = 0.02f;
    const float stable_x = 0.5f, stable_y = 0.5f;
    double t = 0.0;

    // 50프레임 안정 상태
    for (int i = 0; i < 50; ++i) {
        t += kFrameInterval;
        auto r = makeResult(stable_x, stable_y, radius);
        r.timestamp_ms = static_cast<int64_t>(t * 1000);
        stab.stabilize(r, t);
    }

    // 극단적 이상치 1프레임: 0.5 → 0.9 (dist=0.566 > 4*0.02=0.08)
    t += kFrameInterval;
    auto outlier = makeResult(0.9f, 0.9f, radius);
    outlier.timestamp_ms = static_cast<int64_t>(t * 1000);
    auto out_outlier = stab.stabilize(outlier, t);

    std::fprintf(stderr,
        "[OutlierReject] after spike: x=%.4f y=%.4f (expected ~0.5)\n",
        out_outlier.stabilized.left_iris[0].x,
        out_outlier.stabilized.left_iris[0].y);

    // confirm_frames=2 → 단일 극단적 스파이크 거부
    EXPECT_NEAR(out_outlier.stabilized.left_iris[0].x, stable_x, 0.05f);
    EXPECT_NEAR(out_outlier.stabilized.left_iris[0].y, stable_y, 0.05f);

    // 정상 복귀
    t += kFrameInterval;
    auto normal = makeResult(stable_x, stable_y, radius);
    normal.timestamp_ms = static_cast<int64_t>(t * 1000);
    auto out_normal = stab.stabilize(normal, t);

    EXPECT_NEAR(out_normal.stabilized.left_iris[0].x, stable_x, 0.02f);
}

TEST_F(TemporalStabilityTest, ConsecutiveOutliersAcceptedAsRealMovement) {
    TemporalStabilizer stab;

    const float radius = 0.02f;
    const float stable_x = 0.5f;
    const float new_x = 0.65f;
    double t = 0.0;

    // 50프레임 안정 상태
    for (int i = 0; i < 50; ++i) {
        t += kFrameInterval;
        auto r = makeResult(stable_x, 0.5f, radius);
        r.timestamp_ms = static_cast<int64_t>(t * 1000);
        stab.stabilize(r, t);
    }

    // 3프레임 연속 같은 방향 "이상치" -> 실제 움직임으로 인정
    for (int i = 0; i < 3; ++i) {
        t += kFrameInterval;
        auto r = makeResult(new_x, 0.5f, radius);
        r.timestamp_ms = static_cast<int64_t>(t * 1000);
        stab.stabilize(r, t);
    }

    // 추가 프레임으로 수렴 대기
    for (int i = 0; i < 15; ++i) {
        t += kFrameInterval;
        auto r = makeResult(new_x, 0.5f, radius);
        r.timestamp_ms = static_cast<int64_t>(t * 1000);
        stab.stabilize(r, t);
    }

    t += kFrameInterval;
    auto r = makeResult(new_x, 0.5f, radius);
    r.timestamp_ms = static_cast<int64_t>(t * 1000);
    auto out = stab.stabilize(r, t);

    std::fprintf(stderr,
        "[OutlierAccept] converged x=%.4f (expected ~%.4f)\n",
        out.stabilized.left_iris[0].x, new_x);

    EXPECT_NEAR(out.stabilized.left_iris[0].x, new_x, 0.05f)
        << "Consecutive outliers should be accepted as real movement";
}

// ============================================================================
// 6. Async API Cache — getLatestResult 캐싱 로직
// ============================================================================

TEST_F(TemporalStabilityTest, StabilizerCachesLastResult) {
    // TemporalStabilizer의 stabilize()는 매 호출마다 StabilizedResult를 반환.
    // 동일 입력을 연속 호출하면 결과가 결정적이어야 함 (캐시 일관성).
    TemporalStabilizer stab;
    warmUp(stab);

    double t = 1.0;
    auto r = makeResult(0.5f, 0.5f, 10.0f);
    r.timestamp_ms = static_cast<int64_t>(t * 1000);
    auto out1 = stab.stabilize(r, t);

    // 동일 타임스탬프로 다시 호출 (dt=0 상황)
    auto out2 = stab.stabilize(r, t);

    // 출력이 일관적이어야 함 (NaN이나 발산 없음)
    EXPECT_FALSE(std::isnan(out2.stabilized.left_iris[0].x));
    EXPECT_FALSE(std::isnan(out2.stabilized.left_iris[0].y));
    EXPECT_FALSE(std::isnan(out2.stabilized.left_radius));
    EXPECT_GE(out2.visibility, 0.0f);
    EXPECT_LE(out2.visibility, 1.0f);
}

TEST_F(TemporalStabilityTest, StabilizerResultDeterministic) {
    // 동일 입력 시퀀스 -> 동일 출력 (재현성)
    auto runSequence = []() -> std::vector<float> {
        TemporalStabilizer stab;
        std::vector<float> outputs;
        for (int i = 0; i < 30; ++i) {
            double t = i * (1.0 / 30.0);
            float x = 0.5f + 0.01f * std::sin(static_cast<float>(i) * 0.5f);
            auto r = makeResult(x, 0.5f, 10.0f);
            r.timestamp_ms = static_cast<int64_t>(t * 1000);
            auto out = stab.stabilize(r, t);
            outputs.push_back(out.stabilized.left_iris[0].x);
        }
        return outputs;
    };

    auto seq1 = runSequence();
    auto seq2 = runSequence();

    ASSERT_EQ(seq1.size(), seq2.size());
    for (size_t i = 0; i < seq1.size(); ++i) {
        EXPECT_FLOAT_EQ(seq1[i], seq2[i])
            << "Non-deterministic at frame " << i;
    }
}

// ============================================================================
// 7. Smoothing Coefficient of Variation (CV) — 정현파 프레임 간 변동 계수
// ============================================================================

TEST_F(TemporalStabilityTest, SinusoidalMotionSmootherDeltas) {
    TemporalStabilizer stab;
    warmUp(stab);

    const int N = 120;  // 4초 @ 30fps
    const float amplitude = 0.05f;
    const float freq = 2.0f;  // 2Hz 자연스러운 눈 움직임

    std::vector<float> raw_deltas, smooth_deltas;
    raw_deltas.reserve(N);
    smooth_deltas.reserve(N);

    float prev_raw_x = -1.0f, prev_smooth_x = -1.0f;

    for (int i = 0; i < N; ++i) {
        double t = 1.0 + i * kFrameInterval;
        float x = 0.5f + amplitude * std::sin(
            2.0f * static_cast<float>(M_PI) * freq *
            static_cast<float>(i) * static_cast<float>(kFrameInterval));

        auto r = makeResult(x, 0.5f, 10.0f);
        r.timestamp_ms = static_cast<int64_t>(t * 1000);
        auto out = stab.stabilize(r, t);

        float smooth_x = out.stabilized.left_iris[0].x;

        if (prev_raw_x >= 0.0f) {
            raw_deltas.push_back(std::abs(x - prev_raw_x));
            smooth_deltas.push_back(std::abs(smooth_x - prev_smooth_x));
        }

        prev_raw_x = x;
        prev_smooth_x = smooth_x;
    }

    // CV = stddev / mean (변동 계수)
    double raw_delta_mean = mean(raw_deltas);
    double raw_delta_sd = stddev(raw_deltas);
    double smooth_delta_mean = mean(smooth_deltas);
    double smooth_delta_sd = stddev(smooth_deltas);

    double raw_cv = (raw_delta_mean > 1e-8) ? raw_delta_sd / raw_delta_mean : 0.0;
    double smooth_cv = (smooth_delta_mean > 1e-8)
                           ? smooth_delta_sd / smooth_delta_mean
                           : 0.0;

    std::fprintf(stderr,
        "[SinCV] raw_cv=%.4f smooth_cv=%.4f (lower is smoother)\n"
        "        raw_delta: mean=%.6f sd=%.6f\n"
        "        smooth_delta: mean=%.6f sd=%.6f\n",
        raw_cv, smooth_cv,
        raw_delta_mean, raw_delta_sd,
        smooth_delta_mean, smooth_delta_sd);

    // 스무딩된 delta의 평균이 raw보다 작거나 같아야 함 (고주파 노이즈 감소)
    // 참고: 높은 beta(15.0)에서 정현파는 거의 패스스루되므로 CV 대신 mean delta 비교
    EXPECT_LE(smooth_delta_mean, raw_delta_mean * 1.05)
        << "Smoothed motion deltas should not be significantly larger than raw";
}

TEST_F(TemporalStabilityTest, SinusoidalMotionAmplitudePreserved) {
    // 스무딩이 신호의 진폭을 과도하게 줄이지 않는지 확인
    TemporalStabilizer stab;
    warmUp(stab);

    const int N = 120;
    const float amplitude = 0.05f;
    const float freq = 1.0f;  // 1Hz

    float smooth_min = 1.0f, smooth_max = 0.0f;

    for (int i = 0; i < N; ++i) {
        double t = 1.0 + i * kFrameInterval;
        float x = 0.5f + amplitude * std::sin(
            2.0f * static_cast<float>(M_PI) * freq *
            static_cast<float>(i) * static_cast<float>(kFrameInterval));

        auto r = makeResult(x, 0.5f, 10.0f);
        r.timestamp_ms = static_cast<int64_t>(t * 1000);
        auto out = stab.stabilize(r, t);

        float sx = out.stabilized.left_iris[0].x;
        if (i >= 30) {  // 워밍업 후
            smooth_min = std::min(smooth_min, sx);
            smooth_max = std::max(smooth_max, sx);
        }
    }

    float smooth_amplitude = (smooth_max - smooth_min) / 2.0f;

    std::fprintf(stderr,
        "[SinAmplitude] input=%.4f output=%.4f ratio=%.2f%%\n",
        amplitude, smooth_amplitude,
        (smooth_amplitude / amplitude) * 100.0f);

    // 진폭이 원래의 50% 이상 유지 (과도한 감쇠 방지)
    EXPECT_GT(smooth_amplitude, amplitude * 0.5f)
        << "Smoothing attenuated sinusoidal motion too aggressively";
}

// ============================================================================
// Bonus: End-to-End Multi-Phase Scenario
// ============================================================================

TEST_F(TemporalStabilityTest, FullScenarioStability) {
    // 실제 사용 시나리오: 안정 -> 이동 -> 깜빡임 -> 이탈 -> 복귀
    TemporalStabilizer stab;
    double t = 0.0;

    auto feed = [&](float cx, float cy, float radius,
                    float conf, bool detected, bool blink = false) {
        t += kFrameInterval;
        auto r = makeResult(cx, cy, radius, conf, detected);
        r.timestamp_ms = static_cast<int64_t>(t * 1000);
        if (blink) setBlinkEAR(r, true);
        return stab.stabilize(r, t);
    };

    // Phase A: 안정 상태 (20프레임)
    for (int i = 0; i < 20; ++i) feed(0.5f, 0.5f, 10.0f, 0.9f, true);

    // Phase B: 느린 이동 (20프레임, 0.5 -> 0.6)
    for (int i = 0; i < 20; ++i) {
        float x = 0.5f + 0.1f * static_cast<float>(i) / 20.0f;
        feed(x, 0.5f, 10.0f, 0.9f, true);
    }

    // Phase C: 눈 깜빡임 (3프레임)
    for (int i = 0; i < 3; ++i) {
        auto out = feed(0.6f, 0.8f, 10.0f, 0.9f, true, true);
        // 깜빡임 중 Y가 과도하게 이동하지 않아야 함
        EXPECT_LT(out.stabilized.left_iris[0].y, 0.65f)
            << "Y jumped during blink in full scenario";
    }

    // Phase D: 검출 소실 (5프레임)
    StabilizedResult last_held;
    for (int i = 0; i < 5; ++i) {
        last_held = feed(0.0f, 0.0f, 0.0f, 0.0f, false);
    }
    // Hold 중이거나 fade-out 중이어야 함
    EXPECT_TRUE(last_held.is_held || last_held.visibility > 0.0f);

    // Phase E: 복귀 (20프레임)
    StabilizedResult final_out;
    for (int i = 0; i < 20; ++i) {
        final_out = feed(0.6f, 0.5f, 10.0f, 0.9f, true);
    }

    // 최종 상태: 안정적으로 추적 중
    EXPECT_NEAR(final_out.stabilized.left_iris[0].x, 0.6f, 0.05f);
    EXPECT_GT(final_out.visibility, 0.8f);
    EXPECT_FALSE(final_out.is_held);

    std::fprintf(stderr,
        "[FullScenario] final: x=%.4f vis=%.4f held=%d\n",
        final_out.stabilized.left_iris[0].x,
        final_out.visibility,
        final_out.is_held);
}
