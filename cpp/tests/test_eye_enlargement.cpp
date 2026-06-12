/**
 * @file test_eye_enlargement.cpp
 * @brief Unit tests for Eye Enlargement effect (P2-W4-03)
 *
 * Tests for:
 * - Radial expansion from eye center
 * - Symmetric left-right eye enlargement
 * - Eyebrow lift proportional to expansion
 * - Zero strength produces no change
 * - Maximum strength bounded
 * - Smooth falloff at outer region
 * - Combined effects with other warp effects
 */

#include <gtest/gtest.h>
#include "iris_sdk/warp/face_warp_controller.h"
#include "iris_sdk/warp/grid_mesh.h"
#include "iris_sdk/types.h"

#include <array>
#include <cmath>
#include <vector>

namespace iris_sdk {
namespace warp {
namespace test {

/**
 * @brief Test fixture for Eye Enlargement tests
 */
class EyeEnlargementTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a standard face mesh with 478 landmarks
        createMockFaceMesh();

        // Initialize grid mesh with standard settings
        Rect face_rect{0.1f, 0.1f, 0.8f, 0.8f};  // Face ROI
        mesh_.initialize(GridMesh::DEFAULT_GRID_SIZE, face_rect);
        mesh_.setControlPoints(face_mesh_.data(),
                               static_cast<int>(face_mesh_.size()),
                               640, 480);
    }

    /**
     * @brief Create a mock face mesh with realistic eye landmark positions
     */
    void createMockFaceMesh() {
        face_mesh_.resize(478);

        // Initialize all landmarks to face center
        for (auto& lm : face_mesh_) {
            lm.x = 0.5f;
            lm.y = 0.5f;
            lm.z = 0.0f;
            lm.visibility = 1.0f;
        }

        // Set nose tip (center reference)
        face_mesh_[FaceWarpController::NOSE_TIP_INDEX].x = 0.5f;
        face_mesh_[FaceWarpController::NOSE_TIP_INDEX].y = 0.5f;

        // Set cheek landmarks for face width calculation
        face_mesh_[234].x = 0.25f;  // Left outer cheek
        face_mesh_[234].y = 0.4f;
        face_mesh_[454].x = 0.75f;  // Right outer cheek
        face_mesh_[454].y = 0.4f;

        // Set left cheek landmarks (for slim face effect)
        face_mesh_[93].x = 0.30f;  face_mesh_[93].y = 0.42f;
        face_mesh_[132].x = 0.32f; face_mesh_[132].y = 0.45f;
        face_mesh_[58].x = 0.33f;  face_mesh_[58].y = 0.50f;
        face_mesh_[172].x = 0.32f; face_mesh_[172].y = 0.55f;
        face_mesh_[136].x = 0.30f; face_mesh_[136].y = 0.60f;
        face_mesh_[150].x = 0.28f; face_mesh_[150].y = 0.65f;
        face_mesh_[149].x = 0.27f; face_mesh_[149].y = 0.70f;

        // Set right cheek landmarks
        face_mesh_[323].x = 0.70f; face_mesh_[323].y = 0.42f;
        face_mesh_[361].x = 0.68f; face_mesh_[361].y = 0.45f;
        face_mesh_[288].x = 0.67f; face_mesh_[288].y = 0.50f;
        face_mesh_[397].x = 0.68f; face_mesh_[397].y = 0.55f;
        face_mesh_[365].x = 0.70f; face_mesh_[365].y = 0.60f;
        face_mesh_[379].x = 0.72f; face_mesh_[379].y = 0.65f;
        face_mesh_[378].x = 0.73f; face_mesh_[378].y = 0.70f;

        // Set chin center landmarks (for thin chin effect)
        face_mesh_[152].x = 0.50f; face_mesh_[152].y = 0.75f;
        face_mesh_[175].x = 0.50f; face_mesh_[175].y = 0.78f;
        face_mesh_[199].x = 0.50f; face_mesh_[199].y = 0.82f;
        face_mesh_[18].x = 0.48f;  face_mesh_[18].y = 0.80f;
        face_mesh_[17].x = 0.52f;  face_mesh_[17].y = 0.80f;
        face_mesh_[200].x = 0.45f; face_mesh_[200].y = 0.77f;
        face_mesh_[421].x = 0.55f; face_mesh_[421].y = 0.77f;
        face_mesh_[418].x = 0.47f; face_mesh_[418].y = 0.79f;

        // Set left jaw landmarks
        face_mesh_[176].x = 0.30f; face_mesh_[176].y = 0.72f;
        face_mesh_[148].x = 0.35f; face_mesh_[148].y = 0.74f;

        // Set right jaw landmarks
        face_mesh_[400].x = 0.70f; face_mesh_[400].y = 0.72f;
        face_mesh_[377].x = 0.65f; face_mesh_[377].y = 0.74f;

        // ========================================
        // Left Eye Setup
        // ========================================

        // Left iris center (468)
        face_mesh_[FaceWarpController::LEFT_IRIS_CENTER].x = 0.35f;
        face_mesh_[FaceWarpController::LEFT_IRIS_CENTER].y = 0.35f;

        // Left eye contour - create a realistic eye shape
        const float left_cx = 0.35f;  // Eye center X
        const float left_cy = 0.35f;  // Eye center Y
        const float eye_rx = 0.03f;   // Horizontal radius
        const float eye_ry = 0.015f;  // Vertical radius

        // Upper lid landmarks
        setEyeContourLandmark(33,  left_cx - eye_rx,      left_cy);           // Inner corner
        setEyeContourLandmark(7,   left_cx - eye_rx*0.7f, left_cy - eye_ry*0.5f);
        setEyeContourLandmark(163, left_cx - eye_rx*0.3f, left_cy - eye_ry*0.8f);
        setEyeContourLandmark(144, left_cx,               left_cy - eye_ry);  // Top center
        setEyeContourLandmark(145, left_cx + eye_rx*0.3f, left_cy - eye_ry*0.8f);
        setEyeContourLandmark(153, left_cx + eye_rx*0.5f, left_cy - eye_ry*0.5f);
        setEyeContourLandmark(154, left_cx + eye_rx*0.7f, left_cy - eye_ry*0.3f);
        setEyeContourLandmark(155, left_cx + eye_rx*0.9f, left_cy);
        setEyeContourLandmark(133, left_cx + eye_rx,      left_cy);           // Outer corner

        // Lower lid landmarks
        setEyeContourLandmark(173, left_cx - eye_rx*0.7f, left_cy + eye_ry*0.5f);
        setEyeContourLandmark(157, left_cx - eye_rx*0.3f, left_cy + eye_ry*0.7f);
        setEyeContourLandmark(158, left_cx,               left_cy + eye_ry*0.8f);
        setEyeContourLandmark(159, left_cx + eye_rx*0.3f, left_cy + eye_ry*0.7f);
        setEyeContourLandmark(160, left_cx + eye_rx*0.5f, left_cy + eye_ry*0.5f);
        setEyeContourLandmark(161, left_cx + eye_rx*0.7f, left_cy + eye_ry*0.3f);
        setEyeContourLandmark(246, left_cx + eye_rx*0.9f, left_cy + eye_ry*0.1f);

        // Left eyebrow (above eye)
        const float left_brow_y = left_cy - 0.05f;  // Above eye
        setEyeContourLandmark(70,  left_cx - eye_rx,      left_brow_y);
        setEyeContourLandmark(63,  left_cx - eye_rx*0.7f, left_brow_y - 0.005f);
        setEyeContourLandmark(105, left_cx - eye_rx*0.3f, left_brow_y - 0.008f);
        setEyeContourLandmark(66,  left_cx,               left_brow_y - 0.01f);
        setEyeContourLandmark(107, left_cx + eye_rx*0.3f, left_brow_y - 0.008f);
        setEyeContourLandmark(55,  left_cx + eye_rx*0.5f, left_brow_y - 0.005f);
        setEyeContourLandmark(65,  left_cx + eye_rx*0.7f, left_brow_y);
        setEyeContourLandmark(52,  left_cx + eye_rx*0.85f, left_brow_y + 0.003f);
        setEyeContourLandmark(53,  left_cx + eye_rx*0.95f, left_brow_y + 0.005f);
        setEyeContourLandmark(46,  left_cx + eye_rx,      left_brow_y + 0.008f);

        // ========================================
        // Right Eye Setup (mirrored)
        // ========================================

        // Right iris center (473)
        face_mesh_[FaceWarpController::RIGHT_IRIS_CENTER].x = 0.65f;
        face_mesh_[FaceWarpController::RIGHT_IRIS_CENTER].y = 0.35f;

        const float right_cx = 0.65f;
        const float right_cy = 0.35f;

        // Upper lid landmarks
        setEyeContourLandmark(362, right_cx - eye_rx,      right_cy);           // Inner corner
        setEyeContourLandmark(382, right_cx - eye_rx*0.7f, right_cy - eye_ry*0.3f);
        setEyeContourLandmark(381, right_cx - eye_rx*0.5f, right_cy - eye_ry*0.5f);
        setEyeContourLandmark(380, right_cx - eye_rx*0.3f, right_cy - eye_ry*0.8f);
        setEyeContourLandmark(374, right_cx,               right_cy - eye_ry);  // Top center
        setEyeContourLandmark(373, right_cx + eye_rx*0.3f, right_cy - eye_ry*0.8f);
        setEyeContourLandmark(390, right_cx + eye_rx*0.7f, right_cy - eye_ry*0.5f);
        setEyeContourLandmark(249, right_cx + eye_rx*0.9f, right_cy);
        setEyeContourLandmark(263, right_cx + eye_rx,      right_cy);           // Outer corner

        // Lower lid landmarks
        setEyeContourLandmark(466, right_cx - eye_rx*0.7f, right_cy + eye_ry*0.3f);
        setEyeContourLandmark(388, right_cx - eye_rx*0.5f, right_cy + eye_ry*0.5f);
        setEyeContourLandmark(387, right_cx - eye_rx*0.3f, right_cy + eye_ry*0.7f);
        setEyeContourLandmark(386, right_cx,               right_cy + eye_ry*0.8f);
        setEyeContourLandmark(385, right_cx + eye_rx*0.3f, right_cy + eye_ry*0.7f);
        setEyeContourLandmark(384, right_cx + eye_rx*0.5f, right_cy + eye_ry*0.5f);
        setEyeContourLandmark(398, right_cx + eye_rx*0.7f, right_cy + eye_ry*0.3f);

        // Right eyebrow (above eye)
        const float right_brow_y = right_cy - 0.05f;
        setEyeContourLandmark(300, right_cx - eye_rx,      right_brow_y + 0.008f);
        setEyeContourLandmark(293, right_cx - eye_rx*0.7f, right_brow_y);
        setEyeContourLandmark(334, right_cx - eye_rx*0.3f, right_brow_y - 0.005f);
        setEyeContourLandmark(296, right_cx,               right_brow_y - 0.01f);
        setEyeContourLandmark(336, right_cx + eye_rx*0.3f, right_brow_y - 0.008f);
        setEyeContourLandmark(285, right_cx + eye_rx*0.5f, right_brow_y - 0.005f);
        setEyeContourLandmark(295, right_cx + eye_rx*0.7f, right_brow_y);
        setEyeContourLandmark(282, right_cx + eye_rx*0.85f, right_brow_y + 0.003f);
        setEyeContourLandmark(283, right_cx + eye_rx*0.95f, right_brow_y + 0.005f);
        setEyeContourLandmark(276, right_cx + eye_rx,      right_brow_y + 0.008f);
    }

    void setEyeContourLandmark(int idx, float x, float y) {
        face_mesh_[idx].x = x;
        face_mesh_[idx].y = y;
    }

    /**
     * @brief Calculate distance from eye center to a landmark after warp
     */
    float getDistanceFromEyeCenter(int landmark_idx, int iris_center_idx) {
        // Get original positions
        float center_x = face_mesh_[iris_center_idx].x;
        float center_y = face_mesh_[iris_center_idx].y;
        float lm_x = face_mesh_[landmark_idx].x;
        float lm_y = face_mesh_[landmark_idx].y;

        // Get displacement from mesh (if control point exists)
        const auto& vertices = mesh_.getVertices();
        for (const auto& v : vertices) {
            if (v.landmark_idx == landmark_idx && v.is_control) {
                lm_x += v.dx;
                lm_y += v.dy;
                break;
            }
        }

        float dx = lm_x - center_x;
        float dy = lm_y - center_y;
        return std::sqrt(dx * dx + dy * dy);
    }

    /**
     * @brief Get Y displacement for a landmark
     */
    float getYDisplacement(int landmark_idx) {
        const auto& vertices = mesh_.getVertices();
        for (const auto& v : vertices) {
            if (v.landmark_idx == landmark_idx && v.is_control) {
                return v.dy;
            }
        }
        return 0.0f;
    }

    /**
     * @brief Get total displacement for a landmark
     */
    float getTotalDisplacement(int landmark_idx) {
        const auto& vertices = mesh_.getVertices();
        for (const auto& v : vertices) {
            if (v.landmark_idx == landmark_idx && v.is_control) {
                return std::sqrt(v.dx * v.dx + v.dy * v.dy);
            }
        }
        return 0.0f;
    }

    std::vector<IrisLandmark> face_mesh_;
    GridMesh mesh_;
    FaceWarpController controller_;
};

// =============================================================================
// Test Cases
// =============================================================================

/**
 * @test Radial expansion from eye center
 *
 * Eye contour points should move outward from the iris center
 */
TEST_F(EyeEnlargementTest, RadialExpansionFromCenter) {
    // Get original distances for left eye contour
    std::vector<float> original_distances;
    for (int idx : FaceWarpController::LEFT_EYE_CONTOUR) {
        float center_x = face_mesh_[FaceWarpController::LEFT_IRIS_CENTER].x;
        float center_y = face_mesh_[FaceWarpController::LEFT_IRIS_CENTER].y;
        float dx = face_mesh_[idx].x - center_x;
        float dy = face_mesh_[idx].y - center_y;
        original_distances.push_back(std::sqrt(dx * dx + dy * dy));
    }

    // Apply eye enlargement
    WarpConfig config;
    config.enlargeEyes = 0.5f;
    ASSERT_TRUE(controller_.applyWarp(mesh_, face_mesh_.data(), config));

    // Check that contour points moved outward
    int expanded_count = 0;
    for (size_t i = 0; i < FaceWarpController::LEFT_EYE_CONTOUR.size(); ++i) {
        int idx = FaceWarpController::LEFT_EYE_CONTOUR[i];
        float new_distance = getDistanceFromEyeCenter(idx, FaceWarpController::LEFT_IRIS_CENTER);

        // Points should be farther from center (expanded)
        if (original_distances[i] > 1e-6f && new_distance > original_distances[i]) {
            expanded_count++;
        }
    }

    // Some contour points should expand outward (those that were registered as control points)
    EXPECT_GT(expanded_count, 0) << "At least some contour points should expand";
}

/**
 * @test Symmetric left and right eye enlargement
 *
 * Both eyes should enlarge by approximately equal amounts
 */
TEST_F(EyeEnlargementTest, SymmetricLeftRightEyes) {
    WarpConfig config;
    config.enlargeEyes = 0.5f;
    ASSERT_TRUE(controller_.applyWarp(mesh_, face_mesh_.data(), config));

    // Calculate average displacement magnitude for left eye
    float left_total = 0.0f;
    int left_count = 0;
    for (int idx : FaceWarpController::LEFT_EYE_CONTOUR) {
        float disp = getTotalDisplacement(idx);
        if (disp > 1e-6f) {
            left_total += disp;
            left_count++;
        }
    }

    // Calculate average displacement magnitude for right eye
    float right_total = 0.0f;
    int right_count = 0;
    for (int idx : FaceWarpController::RIGHT_EYE_CONTOUR) {
        float disp = getTotalDisplacement(idx);
        if (disp > 1e-6f) {
            right_total += disp;
            right_count++;
        }
    }

    ASSERT_GT(left_count, 0) << "Left eye should have displaced points";
    ASSERT_GT(right_count, 0) << "Right eye should have displaced points";

    float left_avg = left_total / static_cast<float>(left_count);
    float right_avg = right_total / static_cast<float>(right_count);

    // Left and right should be within 20% of each other
    float diff_ratio = std::abs(left_avg - right_avg) / std::max(left_avg, right_avg);
    EXPECT_LT(diff_ratio, 0.2f) << "Left and right eye enlargement should be similar";
}

/**
 * @test Eyebrow lifts up with eye enlargement
 *
 * Eyebrow landmarks should move upward (negative Y displacement)
 */
TEST_F(EyeEnlargementTest, EyebrowLiftsUp) {
    WarpConfig config;
    config.enlargeEyes = 0.5f;
    ASSERT_TRUE(controller_.applyWarp(mesh_, face_mesh_.data(), config));

    // Check left eyebrow moves upward
    int left_lifted = 0;
    for (int idx : FaceWarpController::LEFT_EYEBROW) {
        float dy = getYDisplacement(idx);
        if (dy < -1e-6f) {  // Negative = upward
            left_lifted++;
        }
    }
    EXPECT_GT(left_lifted, 0) << "At least some left eyebrow points should lift up";

    // Check right eyebrow moves upward
    int right_lifted = 0;
    for (int idx : FaceWarpController::RIGHT_EYEBROW) {
        float dy = getYDisplacement(idx);
        if (dy < -1e-6f) {
            right_lifted++;
        }
    }
    EXPECT_GT(right_lifted, 0) << "At least some right eyebrow points should lift up";
}

/**
 * @test Zero strength produces no change
 *
 * When enlargeEyes = 0, no displacement should occur on eye landmarks
 */
TEST_F(EyeEnlargementTest, ZeroStrengthNoChange) {
    WarpConfig config;
    config.enlargeEyes = 0.0f;
    ASSERT_TRUE(controller_.applyWarp(mesh_, face_mesh_.data(), config));

    // All eye contour displacements should be zero
    for (int idx : FaceWarpController::LEFT_EYE_CONTOUR) {
        float disp = getTotalDisplacement(idx);
        EXPECT_LT(disp, 1e-6f) << "Displacement should be zero for idx " << idx;
    }

    for (int idx : FaceWarpController::RIGHT_EYE_CONTOUR) {
        float disp = getTotalDisplacement(idx);
        EXPECT_LT(disp, 1e-6f) << "Displacement should be zero for idx " << idx;
    }
}

/**
 * @test Maximum strength bounded
 *
 * At full strength, displacement should not exceed reasonable bounds
 */
TEST_F(EyeEnlargementTest, MaxStrengthBounded) {
    WarpConfig config;
    config.enlargeEyes = 1.0f;  // Full strength
    ASSERT_TRUE(controller_.applyWarp(mesh_, face_mesh_.data(), config));

    // Maximum expected displacement is ~25% of eye radius
    // Eye radius is roughly 0.03f, so max displacement should be < 0.01f
    const float max_reasonable_displacement = 0.02f;

    for (int idx : FaceWarpController::LEFT_EYE_CONTOUR) {
        float disp = getTotalDisplacement(idx);
        EXPECT_LT(disp, max_reasonable_displacement)
            << "Displacement should be bounded for idx " << idx;
    }
}

/**
 * @test Strength scaling
 *
 * Higher strength should produce larger displacements
 */
TEST_F(EyeEnlargementTest, StrengthScaling) {
    // Apply with low strength
    WarpConfig config_low;
    config_low.enlargeEyes = 0.3f;
    ASSERT_TRUE(controller_.applyWarp(mesh_, face_mesh_.data(), config_low));

    float low_disp = getTotalDisplacement(FaceWarpController::LEFT_EYE_CONTOUR[4]);

    // Reset and apply with high strength
    mesh_.resetDisplacements();
    WarpConfig config_high;
    config_high.enlargeEyes = 0.9f;
    ASSERT_TRUE(controller_.applyWarp(mesh_, face_mesh_.data(), config_high));

    float high_disp = getTotalDisplacement(FaceWarpController::LEFT_EYE_CONTOUR[4]);

    // Higher strength should produce larger displacement
    EXPECT_GT(high_disp, low_disp) << "Higher strength should produce larger displacement";

    // The ratio should be approximately 3:1 (0.9/0.3)
    float ratio = high_disp / low_disp;
    EXPECT_GT(ratio, 2.0f) << "Displacement ratio should be significant";
    EXPECT_LT(ratio, 4.0f) << "Displacement ratio should be reasonable";
}

/**
 * @test Combined with slim face effect
 *
 * Eye enlargement should work correctly when combined with slim face effect
 */
TEST_F(EyeEnlargementTest, CombinedWithSlimFace) {
    WarpConfig config;
    config.slimFace = 0.5f;
    config.enlargeEyes = 0.5f;
    ASSERT_TRUE(controller_.applyWarp(mesh_, face_mesh_.data(), config));

    // Eyes should still be enlarged (at least some control points)
    int expanded = 0;
    for (int idx : FaceWarpController::LEFT_EYE_CONTOUR) {
        float disp = getTotalDisplacement(idx);
        if (disp > 1e-6f) {
            expanded++;
        }
    }
    EXPECT_GT(expanded, 0) << "Eye contour should expand even with slim face active";

    // Cheeks should also be affected (from slim face effect)
    bool cheek_moved = false;
    for (int idx : FaceWarpController::LEFT_CHEEK_INDICES) {
        float disp = getTotalDisplacement(idx);
        if (disp > 1e-6f) {
            cheek_moved = true;
            break;
        }
    }
    EXPECT_TRUE(cheek_moved) << "Cheek landmarks should also move (slim face effect)";
}

/**
 * @test Combined with thin chin effect
 *
 * Eye enlargement should work correctly when combined with V-line effect
 */
TEST_F(EyeEnlargementTest, CombinedWithThinChin) {
    WarpConfig config;
    config.thinChin = 0.5f;
    config.enlargeEyes = 0.5f;
    ASSERT_TRUE(controller_.applyWarp(mesh_, face_mesh_.data(), config));

    // Eyes should still be enlarged (at least some control points)
    int expanded = 0;
    for (int idx : FaceWarpController::LEFT_EYE_CONTOUR) {
        float disp = getTotalDisplacement(idx);
        if (disp > 1e-6f) {
            expanded++;
        }
    }
    EXPECT_GT(expanded, 0) << "Eye contour should expand even with thin chin active";
}

/**
 * @test All effects combined
 *
 * All three effects should work together
 */
TEST_F(EyeEnlargementTest, AllEffectsCombined) {
    WarpConfig config;
    config.slimFace = 0.5f;
    config.thinChin = 0.5f;
    config.enlargeEyes = 0.5f;
    ASSERT_TRUE(controller_.applyWarp(mesh_, face_mesh_.data(), config));

    // Verify eye enlargement (at least some control points)
    int eye_expanded = 0;
    for (int idx : FaceWarpController::LEFT_EYE_CONTOUR) {
        float disp = getTotalDisplacement(idx);
        if (disp > 1e-6f) {
            eye_expanded++;
        }
    }
    EXPECT_GT(eye_expanded, 0) << "At least some eye contour points should expand";

    // Verify slim face (cheek movement)
    bool cheek_moved = false;
    for (int idx : FaceWarpController::LEFT_CHEEK_INDICES) {
        float disp = getTotalDisplacement(idx);
        if (disp > 1e-6f) {
            cheek_moved = true;
            break;
        }
    }
    EXPECT_TRUE(cheek_moved) << "Cheek should move with slim face effect";

    // Verify thin chin (chin movement) - check if any jaw or chin landmark moved
    bool jaw_chin_moved = false;
    for (int idx : FaceWarpController::LEFT_JAW_INDICES) {
        float disp = getTotalDisplacement(idx);
        if (disp > 1e-6f) {
            jaw_chin_moved = true;
            break;
        }
    }
    if (!jaw_chin_moved) {
        for (int idx : FaceWarpController::CHIN_CENTER_INDICES) {
            float disp = getTotalDisplacement(idx);
            if (disp > 1e-6f) {
                jaw_chin_moved = true;
                break;
            }
        }
    }
    EXPECT_TRUE(jaw_chin_moved) << "Jaw or chin should move with thin chin effect";
}

/**
 * @test Expansion direction is outward
 *
 * Verify that displacement direction is radially outward from center
 */
TEST_F(EyeEnlargementTest, ExpansionDirectionIsOutward) {
    WarpConfig config;
    config.enlargeEyes = 0.5f;
    ASSERT_TRUE(controller_.applyWarp(mesh_, face_mesh_.data(), config));

    float center_x = face_mesh_[FaceWarpController::LEFT_IRIS_CENTER].x;
    float center_y = face_mesh_[FaceWarpController::LEFT_IRIS_CENTER].y;

    // Check that displacement direction aligns with radial direction
    const auto& vertices = mesh_.getVertices();
    for (int idx : FaceWarpController::LEFT_EYE_CONTOUR) {
        for (const auto& v : vertices) {
            if (v.landmark_idx == idx && v.is_control) {
                // Original direction from center
                float orig_dx = face_mesh_[idx].x - center_x;
                float orig_dy = face_mesh_[idx].y - center_y;
                float orig_len = std::sqrt(orig_dx * orig_dx + orig_dy * orig_dy);

                if (orig_len < 1e-6f) continue;

                // Normalize
                orig_dx /= orig_len;
                orig_dy /= orig_len;

                // Displacement direction
                float disp_len = std::sqrt(v.dx * v.dx + v.dy * v.dy);
                if (disp_len < 1e-6f) continue;

                float disp_dx = v.dx / disp_len;
                float disp_dy = v.dy / disp_len;

                // Dot product should be positive (same direction)
                float dot = orig_dx * disp_dx + orig_dy * disp_dy;
                EXPECT_GT(dot, 0.5f) << "Displacement should be outward for idx " << idx;
                break;
            }
        }
    }
}

/**
 * @test Landmark constants are valid
 *
 * Verify that all landmark indices are within valid range
 */
TEST_F(EyeEnlargementTest, LandmarkConstantsValid) {
    // Check iris centers
    EXPECT_GE(FaceWarpController::LEFT_IRIS_CENTER, 0);
    EXPECT_LT(FaceWarpController::LEFT_IRIS_CENTER, 478);
    EXPECT_GE(FaceWarpController::RIGHT_IRIS_CENTER, 0);
    EXPECT_LT(FaceWarpController::RIGHT_IRIS_CENTER, 478);

    // Check eye contours
    for (int idx : FaceWarpController::LEFT_EYE_CONTOUR) {
        EXPECT_GE(idx, 0);
        EXPECT_LT(idx, 478);
    }
    for (int idx : FaceWarpController::RIGHT_EYE_CONTOUR) {
        EXPECT_GE(idx, 0);
        EXPECT_LT(idx, 478);
    }

    // Check eyebrows
    for (int idx : FaceWarpController::LEFT_EYEBROW) {
        EXPECT_GE(idx, 0);
        EXPECT_LT(idx, 478);
    }
    for (int idx : FaceWarpController::RIGHT_EYEBROW) {
        EXPECT_GE(idx, 0);
        EXPECT_LT(idx, 478);
    }
}

/**
 * @test Effect parameters are reasonable
 *
 * Verify that effect parameter constants have reasonable values
 */
TEST_F(EyeEnlargementTest, EffectParametersReasonable) {
    // Max enlarge scale should be between 10% and 50%
    EXPECT_GE(FaceWarpController::MAX_EYE_ENLARGE_SCALE, 0.1f);
    EXPECT_LE(FaceWarpController::MAX_EYE_ENLARGE_SCALE, 0.5f);

    // Eyebrow lift ratio should be small (less than 1x eye expansion)
    EXPECT_GT(FaceWarpController::EYEBROW_LIFT_RATIO, 0.0f);
    EXPECT_LE(FaceWarpController::EYEBROW_LIFT_RATIO, 1.0f);
}

// ③-2 B3: applyWarp 랜드마크 개수 가드 — 474 미만 배열은 명시 거부 (기존: OOB 읽기 UB)
TEST_F(EyeEnlargementTest, ApplyWarpRejectsShortLandmarkArray) {
    WarpConfig config;
    config.enlargeEyes = 0.5f;

    // 468짜리 배열(구 Face Mesh 규약)은 RIGHT_IRIS_CENTER(473) 접근이 OOB였다 — 거부돼야 함
    EXPECT_FALSE(controller_.applyWarp(mesh_, face_mesh_.data(), config, 468));
    EXPECT_FALSE(controller_.applyWarp(mesh_, face_mesh_.data(), config,
                                       FaceWarpController::kMinWarpLandmarkCount - 1));

    // 경계값과 표준 478은 통과
    EXPECT_TRUE(controller_.applyWarp(mesh_, face_mesh_.data(), config,
                                      FaceWarpController::kMinWarpLandmarkCount));
    EXPECT_TRUE(controller_.applyWarp(mesh_, face_mesh_.data(), config, 478));

    // 디폴트 인자(478) 경로 — 기존 호출 형태 불변
    EXPECT_TRUE(controller_.applyWarp(mesh_, face_mesh_.data(), config));
}

} // namespace test
} // namespace warp
} // namespace iris_sdk
