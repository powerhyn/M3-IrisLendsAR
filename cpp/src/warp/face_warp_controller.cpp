/**
 * @file face_warp_controller.cpp
 * @brief Implementation of FaceWarpController for slim face, V-line, and eye enlargement effects
 *
 * P2-W4-02: Slim Face / V-Line Effect Implementation
 * P2-W4-03: Eye Enlargement Effect Implementation
 */

#include "iris_sdk/warp/face_warp_controller.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace iris_sdk {
namespace warp {

// =============================================================================
// Public API
// =============================================================================

bool FaceWarpController::applyWarp(GridMesh& mesh,
                                   const IrisLandmark* face_mesh,
                                   const WarpConfig& config,
                                   int landmark_count) {
    // Validate input
    if (!mesh.isInitialized()) {
        return false;
    }

    if (face_mesh == nullptr) {
        return false;
    }

    // ③-2 B3: 사용 최대 인덱스가 LEFT_IRIS_CENTER(473, ④ §7.3 canonical 후)이므로
    // 그보다 짧은 배열은 기존에 out-of-bounds 읽기(UB)였다 — 명시 거부로 전환
    // (kMinWarpLandmarkCount=474 = max index+1 불변, 정상 478 호출자도 불변).
    if (landmark_count < kMinWarpLandmarkCount) {
        return false;
    }

    // Clamp configuration values
    WarpConfig safe_config = config.clamped();

    // Early exit if no effects are active
    if (!safe_config.hasActiveEffect()) {
        return true;
    }

    // Reset all existing displacements
    mesh.resetDisplacements();

    // Register additional control points for face warp effects
    // These landmarks may not be in the default ControlLandmarks set
    registerWarpControlPoints(mesh, face_mesh);

    // Calculate face metrics for displacement scaling
    float face_width = calculateFaceWidth(face_mesh);
    float center_x = calculateFaceCenterX(face_mesh);

    // Validate face metrics
    if (face_width <= 0.0f || face_width > 1.0f) {
        // Invalid face width, skip warp
        return false;
    }

    // Apply effects in order
    if (safe_config.slimFace > 0.0f) {
        applySlimFace(mesh, face_mesh, safe_config.slimFace, face_width, center_x);
    }

    if (safe_config.thinChin > 0.0f) {
        applyThinChin(mesh, face_mesh, safe_config.thinChin, face_width, center_x);
    }

    if (safe_config.enlargeEyes > 0.0f) {
        applyEnlargeEyes(mesh, face_mesh, safe_config.enlargeEyes);
    }

    // Interpolate displacements from control points to all vertices
    mesh.interpolateDisplacements();

    // Compute final vertex positions
    mesh.computeFinalPositions();

    return true;
}

float FaceWarpController::calculateFaceWidth(const IrisLandmark* face_mesh) {
    if (face_mesh == nullptr) {
        return 0.0f;
    }

    // Use outer cheek landmarks for width estimation
    // Left side: index 234 (leftmost cheek point)
    // Right side: index 454 (rightmost cheek point)
    constexpr int LEFT_OUTER = 234;
    constexpr int RIGHT_OUTER = 454;

    float left_x = face_mesh[LEFT_OUTER].x;
    float right_x = face_mesh[RIGHT_OUTER].x;

    return std::abs(right_x - left_x);
}

float FaceWarpController::calculateFaceCenterX(const IrisLandmark* face_mesh) {
    if (face_mesh == nullptr) {
        return 0.5f;
    }

    // Use nose tip as center reference
    return face_mesh[NOSE_TIP_INDEX].x;
}

// =============================================================================
// Effect Implementations
// =============================================================================

void FaceWarpController::applySlimFace(GridMesh& mesh,
                                        const IrisLandmark* face_mesh,
                                        float strength,
                                        float face_width,
                                        float center_x) {
    // Calculate maximum displacement based on face width
    const float max_dx = MAX_SLIM_FACE_DX * face_width * strength;

    // Find Y range of cheek landmarks for gradient calculation
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();

    for (int idx : LEFT_CHEEK_INDICES) {
        min_y = std::min(min_y, face_mesh[idx].y);
        max_y = std::max(max_y, face_mesh[idx].y);
    }
    for (int idx : RIGHT_CHEEK_INDICES) {
        min_y = std::min(min_y, face_mesh[idx].y);
        max_y = std::max(max_y, face_mesh[idx].y);
    }

    // Process left cheek landmarks (move rightward, toward center)
    for (int idx : LEFT_CHEEK_INDICES) {
        const auto& lm = face_mesh[idx];

        // Calculate Y-based weight (more effect at cheek center)
        float y_weight = calculateYWeight(lm.y, min_y, max_y);

        // Calculate distance from center for falloff
        float dist_from_center = std::abs(lm.x - center_x);
        float dist_weight = calculateDistanceFalloff(dist_from_center, face_width * 0.5f);

        // Combined weight
        float weight = y_weight * (1.0f - dist_weight * 0.3f);

        // Displacement toward center (positive = right)
        float dx = max_dx * weight;

        mesh.setControlPointDisplacement(idx, dx, 0.0f);
    }

    // Process right cheek landmarks (move leftward, toward center)
    for (int idx : RIGHT_CHEEK_INDICES) {
        const auto& lm = face_mesh[idx];

        // Calculate Y-based weight
        float y_weight = calculateYWeight(lm.y, min_y, max_y);

        // Calculate distance from center for falloff
        float dist_from_center = std::abs(lm.x - center_x);
        float dist_weight = calculateDistanceFalloff(dist_from_center, face_width * 0.5f);

        // Combined weight
        float weight = y_weight * (1.0f - dist_weight * 0.3f);

        // Displacement toward center (negative = left)
        float dx = -max_dx * weight;

        mesh.setControlPointDisplacement(idx, dx, 0.0f);
    }
}

void FaceWarpController::applyThinChin(GridMesh& mesh,
                                        const IrisLandmark* face_mesh,
                                        float strength,
                                        float face_width,
                                        float center_x) {
    // Calculate maximum displacements
    const float max_dx = MAX_VLINE_DX * face_width * strength;
    const float max_dy = MAX_VLINE_DY * strength;

    // Find Y range for gradient calculation (jaw and chin region)
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();

    // Collect all jaw and chin landmarks for Y range
    for (int idx : LEFT_JAW_INDICES) {
        min_y = std::min(min_y, face_mesh[idx].y);
        max_y = std::max(max_y, face_mesh[idx].y);
    }
    for (int idx : RIGHT_JAW_INDICES) {
        min_y = std::min(min_y, face_mesh[idx].y);
        max_y = std::max(max_y, face_mesh[idx].y);
    }
    for (int idx : CHIN_CENTER_INDICES) {
        min_y = std::min(min_y, face_mesh[idx].y);
        max_y = std::max(max_y, face_mesh[idx].y);
    }

    // Process left jaw landmarks (move right and up)
    for (int idx : LEFT_JAW_INDICES) {
        const auto& lm = face_mesh[idx];

        // Y-based weight: more displacement at lower positions
        float y_normalized = (max_y > min_y) ? (lm.y - min_y) / (max_y - min_y) : 0.5f;
        float y_weight = y_normalized;  // Linear: lower = more weight

        // X displacement toward center
        float dx = max_dx * y_weight;

        // Y displacement upward (negative = up in image coords)
        float dy = -max_dy * y_weight;

        mesh.setControlPointDisplacement(idx, dx, dy);
    }

    // Process right jaw landmarks (move left and up)
    for (int idx : RIGHT_JAW_INDICES) {
        const auto& lm = face_mesh[idx];

        // Y-based weight
        float y_normalized = (max_y > min_y) ? (lm.y - min_y) / (max_y - min_y) : 0.5f;
        float y_weight = y_normalized;

        // X displacement toward center
        float dx = -max_dx * y_weight;

        // Y displacement upward
        float dy = -max_dy * y_weight;

        mesh.setControlPointDisplacement(idx, dx, dy);
    }

    // Process chin center landmarks (move up only, no X displacement)
    for (int idx : CHIN_CENTER_INDICES) {
        const auto& lm = face_mesh[idx];

        // Y-based weight
        float y_normalized = (max_y > min_y) ? (lm.y - min_y) / (max_y - min_y) : 0.5f;
        float y_weight = y_normalized;

        // X displacement: slight movement toward center based on position
        float x_offset = lm.x - center_x;
        float dx = -x_offset * max_dx * 0.5f * y_weight;

        // Y displacement upward
        float dy = -max_dy * y_weight;

        mesh.setControlPointDisplacement(idx, dx, dy);
    }
}

void FaceWarpController::applyEnlargeEyes(GridMesh& mesh,
                                           const IrisLandmark* face_mesh,
                                           float strength) {
    // Apply eye enlargement to both eyes
    // Each eye uses radial expansion from iris center

    // Left eye enlargement
    applyEyeEnlargementSingle(mesh, face_mesh,
                               LEFT_IRIS_CENTER,
                               LEFT_EYE_CONTOUR,
                               LEFT_EYEBROW,
                               strength);

    // Right eye enlargement
    applyEyeEnlargementSingle(mesh, face_mesh,
                               RIGHT_IRIS_CENTER,
                               RIGHT_EYE_CONTOUR,
                               RIGHT_EYEBROW,
                               strength);
}

template<std::size_t ContourSize, std::size_t EyebrowSize>
void FaceWarpController::applyEyeEnlargementSingle(
    GridMesh& mesh,
    const IrisLandmark* face_mesh,
    int iris_center_idx,
    const std::array<int, ContourSize>& eye_contour,
    const std::array<int, EyebrowSize>& eyebrow_indices,
    float strength) {

    // Get eye center coordinates
    float center_x = face_mesh[iris_center_idx].x;
    float center_y = face_mesh[iris_center_idx].y;

    // Calculate eye radius from contour
    float eye_radius = calculateEyeRadius(face_mesh, iris_center_idx, eye_contour);

    if (eye_radius < 1e-6f) {
        return;  // Invalid eye radius
    }

    // Calculate scale factor based on strength
    float scale_factor = strength * MAX_EYE_ENLARGE_SCALE;

    // Apply radial expansion to eye contour landmarks
    for (int idx : eye_contour) {
        float lm_x = face_mesh[idx].x;
        float lm_y = face_mesh[idx].y;

        // Vector from center to landmark
        float vec_x = lm_x - center_x;
        float vec_y = lm_y - center_y;
        float dist = std::sqrt(vec_x * vec_x + vec_y * vec_y);

        if (dist < 1e-6f) {
            continue;  // Skip points at center
        }

        // Normalize direction vector
        float norm_x = vec_x / dist;
        float norm_y = vec_y / dist;

        // Calculate displacement magnitude
        // Points closer to eye radius get full expansion
        // Inner points get slightly less to preserve iris shape
        float dist_ratio = dist / eye_radius;
        float expansion_weight = 1.0f;
        if (dist_ratio < 0.5f) {
            // Inner region: reduced expansion to preserve iris
            expansion_weight = 0.5f + dist_ratio;
        }

        float displacement = dist * scale_factor * expansion_weight;

        // Apply displacement (outward from center)
        float dx = norm_x * displacement;
        float dy = norm_y * displacement;

        mesh.setControlPointDisplacement(idx, dx, dy);
    }

    // Apply subtle eyebrow lift proportional to eye expansion
    float eyebrow_lift = eye_radius * scale_factor * EYEBROW_LIFT_RATIO;

    for (int idx : eyebrow_indices) {
        // Eyebrow moves upward (negative Y in image coordinates)
        mesh.setControlPointDisplacement(idx, 0.0f, -eyebrow_lift);
    }
}

template<std::size_t ContourSize>
float FaceWarpController::calculateEyeRadius(
    const IrisLandmark* face_mesh,
    int iris_center_idx,
    const std::array<int, ContourSize>& eye_contour) {

    float center_x = face_mesh[iris_center_idx].x;
    float center_y = face_mesh[iris_center_idx].y;

    float max_dist = 0.0f;

    for (int idx : eye_contour) {
        float dx = face_mesh[idx].x - center_x;
        float dy = face_mesh[idx].y - center_y;
        float dist = std::sqrt(dx * dx + dy * dy);
        max_dist = std::max(max_dist, dist);
    }

    return max_dist;
}

// Explicit template instantiations
template void FaceWarpController::applyEyeEnlargementSingle<16, 10>(
    GridMesh&, const IrisLandmark*, int,
    const std::array<int, 16>&, const std::array<int, 10>&, float);

template float FaceWarpController::calculateEyeRadius<16>(
    const IrisLandmark*, int, const std::array<int, 16>&);

// =============================================================================
// Helper Functions
// =============================================================================

float FaceWarpController::calculateYWeight(float y, float min_y, float max_y) {
    if (max_y <= min_y) {
        return 1.0f;
    }

    // Normalize Y to 0-1 range within the region
    float normalized = (y - min_y) / (max_y - min_y);

    // Use bell curve centered at 0.5 for maximum effect at center
    // This creates a smooth gradient with less effect at top and bottom
    float distance_from_center = std::abs(normalized - 0.5f);

    // Cosine falloff: 1.0 at center, ~0.5 at edges
    float weight = 0.5f + 0.5f * std::cos(distance_from_center * 3.14159f);

    return weight;
}

float FaceWarpController::calculateDistanceFalloff(float distance, float max_distance) {
    if (max_distance <= 0.0f) {
        return 0.0f;
    }

    // Clamp distance to valid range
    float normalized = std::min(distance / max_distance, 1.0f);

    // Smooth cosine falloff: 1.0 at distance=0, 0.0 at max_distance
    float weight = 0.5f + 0.5f * std::cos(normalized * 3.14159f);

    return weight;
}

void FaceWarpController::registerWarpControlPoints(GridMesh& mesh, const IrisLandmark* face_mesh) {
    // Collect all landmark indices used by face warp effects
    std::vector<int> warp_landmarks;
    warp_landmarks.reserve(
        LEFT_CHEEK_INDICES.size() +
        RIGHT_CHEEK_INDICES.size() +
        CHIN_CENTER_INDICES.size() +
        LEFT_JAW_INDICES.size() +
        RIGHT_JAW_INDICES.size() +
        LEFT_EYE_CONTOUR.size() +
        RIGHT_EYE_CONTOUR.size() +
        LEFT_EYEBROW.size() +
        RIGHT_EYEBROW.size() +
        2  // Iris centers
    );

    // Left cheek
    for (int idx : LEFT_CHEEK_INDICES) {
        warp_landmarks.push_back(idx);
    }

    // Right cheek
    for (int idx : RIGHT_CHEEK_INDICES) {
        warp_landmarks.push_back(idx);
    }

    // Chin center
    for (int idx : CHIN_CENTER_INDICES) {
        warp_landmarks.push_back(idx);
    }

    // Left jaw
    for (int idx : LEFT_JAW_INDICES) {
        warp_landmarks.push_back(idx);
    }

    // Right jaw
    for (int idx : RIGHT_JAW_INDICES) {
        warp_landmarks.push_back(idx);
    }

    // Eye landmarks for eye enlargement effect
    // Left eye contour
    for (int idx : LEFT_EYE_CONTOUR) {
        warp_landmarks.push_back(idx);
    }

    // Right eye contour
    for (int idx : RIGHT_EYE_CONTOUR) {
        warp_landmarks.push_back(idx);
    }

    // Left eyebrow
    for (int idx : LEFT_EYEBROW) {
        warp_landmarks.push_back(idx);
    }

    // Right eyebrow
    for (int idx : RIGHT_EYEBROW) {
        warp_landmarks.push_back(idx);
    }

    // Iris centers (reference points)
    warp_landmarks.push_back(LEFT_IRIS_CENTER);
    warp_landmarks.push_back(RIGHT_IRIS_CENTER);

    // Register these as additional control points
    mesh.addControlPoints(face_mesh, warp_landmarks.data(), static_cast<int>(warp_landmarks.size()));
}

} // namespace warp
} // namespace iris_sdk
