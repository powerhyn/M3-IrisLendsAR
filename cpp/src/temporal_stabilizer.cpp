#include "iris_sdk/temporal_stabilizer.h"

#include <algorithm>

namespace iris_sdk {

// ============================================================
// MediaPipe face mesh landmark indices for EAR computation
// ============================================================

// Left eye EAR landmarks: p1=33, p2=160, p3=158, p4=133, p5=153, p6=144
static constexpr int kLeftEAR[] = {33, 160, 158, 133, 153, 144};
// Right eye EAR landmarks: p1=362, p2=385, p3=387, p4=263, p5=373, p6=380
static constexpr int kRightEAR[] = {362, 385, 387, 263, 373, 380};

// Eyelid key landmark indices
static constexpr int kLeftEyelidTop = 159;
static constexpr int kLeftEyelidBottom = 145;
static constexpr int kRightEyelidTop = 386;
static constexpr int kRightEyelidBottom = 374;

// ============================================================
// EyeState helpers
// ============================================================

void TemporalStabilizer::EyeState::resetFilters() {
    center_x.reset();
    center_y.reset();
    radius.reset();
    eyelid_top_x.reset();
    eyelid_top_y.reset();
    eyelid_bottom_x.reset();
    eyelid_bottom_y.reset();
}

void TemporalStabilizer::EyeState::resetState() {
    resetFilters();
    prev_center_x = 0.0f;
    prev_center_y = 0.0f;
    prev_radius = 0.0f;
    consecutive_outlier_frames = 0;
    has_previous = false;
    is_blinking = false;
}

// ============================================================
// TemporalStabilizer
// ============================================================

TemporalStabilizer::TemporalStabilizer(const StabilizerConfig& config)
    : config_(config) {
    initFilters();
    last_valid_iris_ = IrisResult{};
}

void TemporalStabilizer::initFilters() {
    // Left eye
    left_eye_.center_x = OneEuroFilter(config_.iris_min_cutoff, config_.iris_beta);
    left_eye_.center_y = OneEuroFilter(config_.iris_min_cutoff, config_.iris_beta);
    left_eye_.radius = OneEuroFilter(config_.radius_min_cutoff, config_.radius_beta);
    left_eye_.eyelid_top_x = OneEuroFilter(config_.eyelid_min_cutoff, config_.eyelid_beta);
    left_eye_.eyelid_top_y = OneEuroFilter(config_.eyelid_min_cutoff, config_.eyelid_beta);
    left_eye_.eyelid_bottom_x = OneEuroFilter(config_.eyelid_min_cutoff, config_.eyelid_beta);
    left_eye_.eyelid_bottom_y = OneEuroFilter(config_.eyelid_min_cutoff, config_.eyelid_beta);

    // Right eye
    right_eye_.center_x = OneEuroFilter(config_.iris_min_cutoff, config_.iris_beta);
    right_eye_.center_y = OneEuroFilter(config_.iris_min_cutoff, config_.iris_beta);
    right_eye_.radius = OneEuroFilter(config_.radius_min_cutoff, config_.radius_beta);
    right_eye_.eyelid_top_x = OneEuroFilter(config_.eyelid_min_cutoff, config_.eyelid_beta);
    right_eye_.eyelid_top_y = OneEuroFilter(config_.eyelid_min_cutoff, config_.eyelid_beta);
    right_eye_.eyelid_bottom_x = OneEuroFilter(config_.eyelid_min_cutoff, config_.eyelid_beta);
    right_eye_.eyelid_bottom_y = OneEuroFilter(config_.eyelid_min_cutoff, config_.eyelid_beta);
}

void TemporalStabilizer::reset() {
    left_eye_.resetState();
    right_eye_.resetState();

    tracking_state_ = TrackingState::Lost;
    low_confidence_count_ = 0;

    visibility_ = 0.0f;
    fade_start_time_ = -1.0;
    fading_in_ = false;
    fading_out_ = false;

    frames_since_valid_ = 0;
    has_any_valid_ = false;
    last_valid_iris_ = IrisResult{};
}

void TemporalStabilizer::setEnabled(bool enabled) {
    enabled_ = enabled;
    if (!enabled) {
        reset();
    }
}

void TemporalStabilizer::setConfig(const StabilizerConfig& config) {
    config_ = config;
    reset();
    initFilters();
}

// ============================================================
// Core: stabilize()
// ============================================================

StabilizedResult TemporalStabilizer::stabilize(const IrisResult& raw, double timestamp_sec) {
    StabilizedResult result{};
    result.raw = raw;
    result.stabilized = raw;  // start from raw, then overwrite smoothed fields
    result.is_held = false;
    result.last_valid_ms = has_any_valid_ ? last_valid_iris_.timestamp_ms : 0;

    // Passthrough mode
    if (!enabled_) {
        result.visibility = raw.detected ? 1.0f : 0.0f;
        return result;
    }

    bool valid_detection = raw.detected;

    // --- Confidence hysteresis ---
    if (valid_detection) {
        switch (tracking_state_) {
        case TrackingState::Tracking:
            if (raw.confidence < config_.confidence_low_threshold) {
                ++low_confidence_count_;
                if (low_confidence_count_ >= config_.confidence_low_frames) {
                    tracking_state_ = TrackingState::Lost;
                    valid_detection = false;
                }
            } else {
                low_confidence_count_ = 0;
            }
            break;

        case TrackingState::Lost:
            if (raw.confidence >= config_.confidence_high_threshold) {
                tracking_state_ = TrackingState::Tracking;
                low_confidence_count_ = 0;
            } else {
                // Not confident enough to re-enter tracking
                valid_detection = false;
            }
            break;
        }
    } else {
        // No detection at all
        if (tracking_state_ == TrackingState::Tracking) {
            tracking_state_ = TrackingState::Lost;
        }
        low_confidence_count_ = 0;
    }

    // --- Blink detection ---
    bool left_blink = false;
    bool right_blink = false;
    if (raw.face_mesh_valid) {
        float left_ear = computeEAR(raw, true);
        float right_ear = computeEAR(raw, false);
        left_blink = left_ear < config_.blink_ear_threshold;
        right_blink = right_ear < config_.blink_ear_threshold;
        left_eye_.is_blinking = left_blink;
        right_eye_.is_blinking = right_blink;
    }

    // --- Process valid detection ---
    if (valid_detection) {
        frames_since_valid_ = 0;

        // Smooth left eye (skip iris smoothing during blink, hold last valid)
        if (raw.left_detected && !left_blink) {
            smoothEye(left_eye_, result.stabilized.left_iris, result.stabilized.left_radius,
                      kLeftEyelidTop, kLeftEyelidBottom,
                      result.stabilized.face_mesh, raw.face_mesh_valid,
                      timestamp_sec);
        } else if (left_blink && has_any_valid_ && last_valid_iris_.left_detected) {
            // During blink: hold last valid iris position
            std::copy(std::begin(last_valid_iris_.left_iris),
                      std::end(last_valid_iris_.left_iris),
                      std::begin(result.stabilized.left_iris));
            result.stabilized.left_radius = last_valid_iris_.left_radius;
        }

        // Smooth right eye
        if (raw.right_detected && !right_blink) {
            smoothEye(right_eye_, result.stabilized.right_iris, result.stabilized.right_radius,
                      kRightEyelidTop, kRightEyelidBottom,
                      result.stabilized.face_mesh, raw.face_mesh_valid,
                      timestamp_sec);
        } else if (right_blink && has_any_valid_ && last_valid_iris_.right_detected) {
            std::copy(std::begin(last_valid_iris_.right_iris),
                      std::end(last_valid_iris_.right_iris),
                      std::begin(result.stabilized.right_iris));
            result.stabilized.right_radius = last_valid_iris_.right_radius;
        }

        // Update last valid
        last_valid_iris_ = result.stabilized;
        has_any_valid_ = true;
        result.last_valid_ms = raw.timestamp_ms;

    } else {
        // --- Dropout hold ---
        ++frames_since_valid_;

        if (has_any_valid_ && frames_since_valid_ <= config_.hold_frames) {
            // Hold last valid coordinates
            result.stabilized = last_valid_iris_;
            result.stabilized.timestamp_ms = raw.timestamp_ms;
            result.stabilized.frame_width = raw.frame_width;
            result.stabilized.frame_height = raw.frame_height;
            result.is_held = true;
        } else {
            // Hold expired or no valid data ever seen
            result.stabilized.detected = false;
            result.stabilized.left_detected = false;
            result.stabilized.right_detected = false;
        }
    }

    // --- Visibility fade ---
    result.visibility = updateVisibility(valid_detection, timestamp_sec);

    return result;
}

// ============================================================
// Smoothing per eye
// ============================================================

void TemporalStabilizer::smoothEye(EyeState& state,
                                    IrisLandmark iris[5], float& radius,
                                    int eyelid_top_idx, int eyelid_bottom_idx,
                                    IrisLandmark face_mesh[478], bool mesh_valid,
                                    double timestamp_sec) {
    float cx = iris[0].x;
    float cy = iris[0].y;
    float raw_radius = radius;  // Save raw radius before filtering

    // --- Outlier rejection ---
    // ref_radius를 정규화 단위로 근사: boundary 랜드마크(iris[1])와 center(iris[0])의 거리
    float norm_radius = 0.0f;
    {
        float bdx = iris[1].x - iris[0].x;
        float bdy = iris[1].y - iris[0].y;
        norm_radius = std::sqrt(bdx * bdx + bdy * bdy);
    }
    if (state.has_previous) {
        if (isOutlier(state, cx, cy, norm_radius)) {
            ++state.consecutive_outlier_frames;
            if (state.consecutive_outlier_frames < config_.outlier_confirm_frames) {
                // Single-frame outlier: reject center, keep previous position
                // (radius는 OneEuroFilter가 스무딩하므로 되돌리지 않음)
                iris[0].x = state.prev_center_x;
                iris[0].y = state.prev_center_y;
                cx = state.prev_center_x;
                cy = state.prev_center_y;
            } else {
                // Consecutive outliers confirmed as real movement, accept and reset
                state.consecutive_outlier_frames = 0;
            }
        } else {
            state.consecutive_outlier_frames = 0;
        }
    }

    // --- Apply OneEuroFilter to iris center and radius ---
    iris[0].x = state.center_x.filter(iris[0].x, timestamp_sec);
    iris[0].y = state.center_y.filter(iris[0].y, timestamp_sec);
    radius = state.radius.filter(radius, timestamp_sec);

    // Store raw (pre-filter) values for next frame's outlier check
    // prev_radius는 정규화 단위로 저장 (isOutlier에서 정규화 좌표와 비교)
    state.prev_center_x = cx;
    state.prev_center_y = cy;
    state.prev_radius = norm_radius;
    state.has_previous = true;

    // --- Smooth eyelid landmarks ---
    if (mesh_valid) {
        face_mesh[eyelid_top_idx].x =
            state.eyelid_top_x.filter(face_mesh[eyelid_top_idx].x, timestamp_sec);
        face_mesh[eyelid_top_idx].y =
            state.eyelid_top_y.filter(face_mesh[eyelid_top_idx].y, timestamp_sec);
        face_mesh[eyelid_bottom_idx].x =
            state.eyelid_bottom_x.filter(face_mesh[eyelid_bottom_idx].x, timestamp_sec);
        face_mesh[eyelid_bottom_idx].y =
            state.eyelid_bottom_y.filter(face_mesh[eyelid_bottom_idx].y, timestamp_sec);
    }
}

// ============================================================
// Outlier detection
// ============================================================

bool TemporalStabilizer::isOutlier(const EyeState& state,
                                    float cx, float cy,
                                    float ref_radius) const {
    float dx = cx - state.prev_center_x;
    float dy = cy - state.prev_center_y;
    float dist = std::sqrt(dx * dx + dy * dy);
    float threshold = config_.outlier_radius_multiplier * ref_radius;
    return dist > threshold;
}

// ============================================================
// Blink detection (Eye Aspect Ratio)
// ============================================================

float TemporalStabilizer::computeEAR(const IrisResult& result, bool left_eye) const {
    if (!result.face_mesh_valid) {
        return 1.0f;  // Assume open if no mesh data
    }

    const int* idx = left_eye ? kLeftEAR : kRightEAR;
    const auto& m = result.face_mesh;

    // EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    auto dist = [&](int a, int b) -> float {
        float dx = m[a].x - m[b].x;
        float dy = m[a].y - m[b].y;
        return std::sqrt(dx * dx + dy * dy);
    };

    float vertical1 = dist(idx[1], idx[5]);  // |p2-p6|
    float vertical2 = dist(idx[2], idx[4]);  // |p3-p5|
    float horizontal = dist(idx[0], idx[3]); // |p1-p4|

    if (horizontal < 1e-6f) {
        return 0.0f;
    }

    return (vertical1 + vertical2) / (2.0f * horizontal);
}

// ============================================================
// Visibility fade
// ============================================================

float TemporalStabilizer::updateVisibility(bool detected, double timestamp_sec) {
    if (detected) {
        if (!fading_in_) {
            // Start or resume fade-in from current visibility level.
            // Back-calculate start time so fade continues smoothly from current value.
            fading_in_ = true;
            fading_out_ = false;
            if (config_.fade_in_ms > 0.0f) {
                float already_elapsed_ms = visibility_ * config_.fade_in_ms;
                fade_start_time_ = timestamp_sec - already_elapsed_ms / 1000.0;
            } else {
                fade_start_time_ = timestamp_sec;
            }
        }

        if (fading_in_) {
            float elapsed_ms = static_cast<float>((timestamp_sec - fade_start_time_) * 1000.0);
            if (config_.fade_in_ms > 0.0f) {
                visibility_ = std::min(1.0f, elapsed_ms / config_.fade_in_ms);
            } else {
                visibility_ = 1.0f;
            }
            if (visibility_ >= 1.0f) {
                fading_in_ = false;
                visibility_ = 1.0f;
            }
        }
    } else {
        if (!fading_out_ && visibility_ > 0.0f) {
            // Start fade-out from current visibility level.
            // Back-calculate start time to continue smoothly.
            fading_out_ = true;
            fading_in_ = false;
            if (config_.fade_out_ms > 0.0f) {
                float already_elapsed_ms = (1.0f - visibility_) * config_.fade_out_ms;
                fade_start_time_ = timestamp_sec - already_elapsed_ms / 1000.0;
            } else {
                fade_start_time_ = timestamp_sec;
            }
        }

        if (fading_out_) {
            float elapsed_ms = static_cast<float>((timestamp_sec - fade_start_time_) * 1000.0);
            if (config_.fade_out_ms > 0.0f) {
                visibility_ = std::max(0.0f, 1.0f - elapsed_ms / config_.fade_out_ms);
            } else {
                visibility_ = 0.0f;
            }
            if (visibility_ <= 0.0f) {
                fading_out_ = false;
                visibility_ = 0.0f;
                // Reset filters when fully faded out
                left_eye_.resetFilters();
                right_eye_.resetFilters();
            }
        }
    }

    return visibility_;
}

}  // namespace iris_sdk
