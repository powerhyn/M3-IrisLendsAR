#pragma once

#include "types.h"
#include "one_euro_filter.h"

#include <array>
#include <cstdint>
#include <cmath>

namespace iris_sdk {

// ============================================================
// Configuration
// ============================================================

struct StabilizerConfig {
    // Smoothing (OneEuroFilter parameters, tuned from Android demo)
    float iris_min_cutoff = 4.0f;
    float iris_beta = 15.0f;
    float radius_min_cutoff = 4.0f;
    float radius_beta = 7.5f;
    float eyelid_min_cutoff = 4.0f;
    float eyelid_beta = 10.0f;

    // Confidence hysteresis
    float confidence_low_threshold = 0.3f;
    float confidence_high_threshold = 0.6f;
    int confidence_low_frames = 3;

    // Visibility fade
    float fade_in_ms = 100.0f;
    float fade_out_ms = 200.0f;

    // Dropout hold
    int hold_frames = 5;

    // Outlier rejection
    float outlier_radius_multiplier = 4.0f;   // 정규화 단위 기준: 반지름 4배 이상 이동 시 의심
    int outlier_confirm_frames = 1;            // 1프레임만 확인 (지연 최소화)

    // Blink detection (Eye Aspect Ratio)
    // EYECLIP(실기기 검증): 0.2 → 0.05. squint(약간 감음, EAR~0.1-0.15)에서 blink-hold가
    // 발동해 홍채 좌표가 고정(렌즈 박힘)되던 것을, 거의 완전 감음(EAR<0.05)에만 hold하도록.
    // squint 동안 홍채를 계속 추적 → 렌즈가 눈동자를 따라감. 눈꺼풀 가림은 렌더 클리핑이 처리.
    // ※ 데모 실효 진입점은 sdk_api.cpp default config(이 헤더 기본값은 nullptr 경로 전용).
    float blink_ear_threshold = 0.05f;
};

// ============================================================
// Output
// ============================================================

struct StabilizedResult {
    IrisResult raw;
    IrisResult stabilized;
    float visibility;        // 0.0~1.0, renderer uses as opacity
    bool is_held;            // true during dropout hold
    int64_t last_valid_ms;   // timestamp of last valid detection
};

// ============================================================
// TemporalStabilizer
// ============================================================

class TemporalStabilizer {
public:
    explicit TemporalStabilizer(const StabilizerConfig& config = {});

    /// Process a raw detection result and produce a stabilized output.
    StabilizedResult stabilize(const IrisResult& raw, double timestamp_sec);

    /// Reset all internal state (filters, hysteresis, hold counters).
    void reset();

    /// Enable/disable stabilization. When disabled, raw result is passed through.
    void setEnabled(bool enabled);
    bool isEnabled() const { return enabled_; }

    /// Update configuration. Calls reset() internally.
    void setConfig(const StabilizerConfig& config);
    const StabilizerConfig& config() const { return config_; }

private:
    // --- Per-eye state ---
    struct EyeState {
        // Iris center + radius filters (signal-specific parameters)
        OneEuroFilter center_x;
        OneEuroFilter center_y;
        OneEuroFilter radius;

        // Eyelid landmark filters (top and bottom)
        OneEuroFilter eyelid_top_x;
        OneEuroFilter eyelid_top_y;
        OneEuroFilter eyelid_bottom_x;
        OneEuroFilter eyelid_bottom_y;

        // Outlier rejection
        float prev_center_x = 0.0f;
        float prev_center_y = 0.0f;
        float prev_radius = 0.0f;
        int consecutive_outlier_frames = 0;
        bool has_previous = false;

        // Blink
        bool is_blinking = false;

        void resetFilters();
        void resetState();
    };

    // --- Hysteresis state ---
    enum class TrackingState { Tracking, Lost };

    // --- Methods ---
    void initFilters();

    bool isOutlier(const EyeState& state, float cx, float cy, float ref_radius) const;
    float computeEAR(const IrisResult& result, bool left_eye) const;
    void smoothEye(EyeState& state, IrisLandmark iris[5], float& radius,
                   int eyelid_top_idx, int eyelid_bottom_idx,
                   IrisLandmark face_mesh[478], bool mesh_valid,
                   double timestamp_sec);

    float updateVisibility(bool detected, double timestamp_sec);

    // --- Data ---
    StabilizerConfig config_;
    bool enabled_ = true;

    EyeState left_eye_;
    EyeState right_eye_;

    // Confidence hysteresis
    TrackingState tracking_state_ = TrackingState::Lost;
    int low_confidence_count_ = 0;

    // Visibility fade
    float visibility_ = 0.0f;
    double fade_start_time_ = -1.0;
    bool fading_in_ = false;
    bool fading_out_ = false;

    // Dropout hold
    int frames_since_valid_ = 0;
    bool has_any_valid_ = false;

    // Last valid iris results for hold/blink
    IrisResult last_valid_iris_{};
};

}  // namespace iris_sdk
