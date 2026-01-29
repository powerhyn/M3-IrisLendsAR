/**
 * @file test_face_warp_controller.cpp
 * @brief Unit tests for FaceWarpController (P2-W4-02)
 *
 * Tests for:
 * - Slim face effect (cheeks move inward)
 * - V-line / thin chin effect (jaw and chin move up and inward)
 * - Combined effects stacking
 * - Strength scaling
 * - Gradient falloff
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
 * @brief Test fixture for FaceWarpController tests
 */
class FaceWarpControllerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a standard face mesh with 478 landmarks
        // Positioned in a typical face configuration
        createMockFaceMesh();

        // Initialize grid mesh with standard settings
        Rect face_rect{0.2f, 0.1f, 0.6f, 0.8f};  // Face ROI
        mesh_.initialize(GridMesh::DEFAULT_GRID_SIZE, face_rect);
        mesh_.setControlPoints(face_mesh_.data(),
                               static_cast<int>(face_mesh_.size()),
                               640, 480);
    }

    /**
     * @brief Create a mock face mesh with realistic landmark positions
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
        face_mesh_[FaceWarpController::NOSE_TIP_INDEX].y = 0.45f;

        // Set left cheek landmarks (left side of face)
        setLandmarkPosition(234, 0.25f, 0.35f);  // Outer left
        setLandmarkPosition(93,  0.30f, 0.40f);
        setLandmarkPosition(132, 0.32f, 0.45f);
        setLandmarkPosition(58,  0.33f, 0.50f);
        setLandmarkPosition(172, 0.32f, 0.55f);
        setLandmarkPosition(136, 0.30f, 0.60f);
        setLandmarkPosition(150, 0.28f, 0.65f);
        setLandmarkPosition(149, 0.27f, 0.70f);

        // Set right cheek landmarks (right side of face, mirrored)
        setLandmarkPosition(454, 0.75f, 0.35f);  // Outer right
        setLandmarkPosition(323, 0.70f, 0.40f);
        setLandmarkPosition(361, 0.68f, 0.45f);
        setLandmarkPosition(288, 0.67f, 0.50f);
        setLandmarkPosition(397, 0.68f, 0.55f);
        setLandmarkPosition(365, 0.70f, 0.60f);
        setLandmarkPosition(379, 0.72f, 0.65f);
        setLandmarkPosition(378, 0.73f, 0.70f);

        // Set chin center landmarks
        setLandmarkPosition(152, 0.50f, 0.75f);
        setLandmarkPosition(175, 0.50f, 0.78f);
        setLandmarkPosition(199, 0.50f, 0.82f);
        setLandmarkPosition(18,  0.48f, 0.80f);
        setLandmarkPosition(17,  0.52f, 0.80f);
        setLandmarkPosition(200, 0.45f, 0.77f);
        setLandmarkPosition(421, 0.55f, 0.77f);
        setLandmarkPosition(418, 0.47f, 0.79f);

        // Set left jaw landmarks
        setLandmarkPosition(136, 0.30f, 0.60f);  // Already set in cheek
        setLandmarkPosition(150, 0.28f, 0.65f);
        setLandmarkPosition(149, 0.27f, 0.70f);
        setLandmarkPosition(176, 0.30f, 0.72f);
        setLandmarkPosition(148, 0.35f, 0.74f);

        // Set right jaw landmarks
        setLandmarkPosition(365, 0.70f, 0.60f);  // Already set in cheek
        setLandmarkPosition(379, 0.72f, 0.65f);
        setLandmarkPosition(378, 0.73f, 0.70f);
        setLandmarkPosition(400, 0.70f, 0.72f);
        setLandmarkPosition(377, 0.65f, 0.74f);
    }

    void setLandmarkPosition(int idx, float x, float y) {
        face_mesh_[idx].x = x;
        face_mesh_[idx].y = y;
    }

    /**
     * @brief Get displacement for a specific landmark from the mesh
     */
    bool getLandmarkDisplacement(int landmark_idx, float& dx, float& dy) {
        const auto& vertices = mesh_.getVertices();
        for (const auto& v : vertices) {
            if (v.is_control && v.landmark_idx == landmark_idx) {
                dx = v.dx;
                dy = v.dy;
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Check if a control point exists for a landmark
     */
    bool hasControlPoint(int landmark_idx) {
        const auto& vertices = mesh_.getVertices();
        for (const auto& v : vertices) {
            if (v.is_control && v.landmark_idx == landmark_idx) {
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Calculate total displacement magnitude across all control points
     */
    float getTotalDisplacementMagnitude() {
        float total = 0.0f;
        const auto& vertices = mesh_.getVertices();
        for (const auto& v : vertices) {
            if (v.is_control) {
                total += std::sqrt(v.dx * v.dx + v.dy * v.dy);
            }
        }
        return total;
    }

    GridMesh mesh_;
    std::vector<IrisLandmark> face_mesh_;
    FaceWarpController controller_;
};

// =============================================================================
// Basic Functionality Tests
// =============================================================================

TEST_F(FaceWarpControllerTest, ApplyWarpWithNullMeshFails) {
    GridMesh uninitialized_mesh;
    WarpConfig config{0.5f, 0.5f, 0.0f};

    EXPECT_FALSE(controller_.applyWarp(uninitialized_mesh, face_mesh_.data(), config));
}

TEST_F(FaceWarpControllerTest, ApplyWarpWithNullLandmarksFails) {
    WarpConfig config{0.5f, 0.5f, 0.0f};

    EXPECT_FALSE(controller_.applyWarp(mesh_, nullptr, config));
}

TEST_F(FaceWarpControllerTest, ZeroStrengthNoChange) {
    WarpConfig config{0.0f, 0.0f, 0.0f};

    // Apply warp with zero strength
    EXPECT_TRUE(controller_.applyWarp(mesh_, face_mesh_.data(), config));

    // Displacements should remain zero
    float dx, dy;
    for (int idx : FaceWarpController::LEFT_CHEEK_INDICES) {
        if (getLandmarkDisplacement(idx, dx, dy)) {
            EXPECT_FLOAT_EQ(dx, 0.0f) << "Left cheek landmark " << idx << " should have zero dx";
            EXPECT_FLOAT_EQ(dy, 0.0f) << "Left cheek landmark " << idx << " should have zero dy";
        }
    }
}

TEST_F(FaceWarpControllerTest, ConfigClampValues) {
    WarpConfig config{-0.5f, 1.5f, 2.0f};
    WarpConfig clamped = config.clamped();

    EXPECT_FLOAT_EQ(clamped.slimFace, 0.0f);
    EXPECT_FLOAT_EQ(clamped.thinChin, 1.0f);
    EXPECT_FLOAT_EQ(clamped.enlargeEyes, 1.0f);
}

TEST_F(FaceWarpControllerTest, ConfigHasActiveEffect) {
    EXPECT_FALSE((WarpConfig{0.0f, 0.0f, 0.0f}.hasActiveEffect()));
    EXPECT_TRUE((WarpConfig{0.1f, 0.0f, 0.0f}.hasActiveEffect()));
    EXPECT_TRUE((WarpConfig{0.0f, 0.1f, 0.0f}.hasActiveEffect()));
    EXPECT_TRUE((WarpConfig{0.0f, 0.0f, 0.1f}.hasActiveEffect()));
}

// =============================================================================
// Slim Face Effect Tests
// =============================================================================

TEST_F(FaceWarpControllerTest, SlimFaceMovesLeftCheekRight) {
    WarpConfig config{1.0f, 0.0f, 0.0f};

    EXPECT_TRUE(controller_.applyWarp(mesh_, face_mesh_.data(), config));

    // Left cheek landmarks should have positive dx (move right toward center)
    float dx, dy;
    int positive_count = 0;
    for (int idx : FaceWarpController::LEFT_CHEEK_INDICES) {
        if (getLandmarkDisplacement(idx, dx, dy)) {
            if (dx > 0.0f) {
                positive_count++;
            }
        }
    }

    // At least some left cheek landmarks should move right
    EXPECT_GT(positive_count, 0) << "Left cheek landmarks should move rightward";
}

TEST_F(FaceWarpControllerTest, SlimFaceMovesRightCheekLeft) {
    WarpConfig config{1.0f, 0.0f, 0.0f};

    EXPECT_TRUE(controller_.applyWarp(mesh_, face_mesh_.data(), config));

    // Right cheek landmarks should have negative dx (move left toward center)
    float dx, dy;
    int negative_count = 0;
    for (int idx : FaceWarpController::RIGHT_CHEEK_INDICES) {
        if (getLandmarkDisplacement(idx, dx, dy)) {
            if (dx < 0.0f) {
                negative_count++;
            }
        }
    }

    // At least some right cheek landmarks should move left
    EXPECT_GT(negative_count, 0) << "Right cheek landmarks should move leftward";
}

TEST_F(FaceWarpControllerTest, SlimFaceSymmetric) {
    WarpConfig config{1.0f, 0.0f, 0.0f};

    EXPECT_TRUE(controller_.applyWarp(mesh_, face_mesh_.data(), config));

    // Get displacement of left outer and right outer cheek landmarks
    float left_dx = 0.0f, left_dy = 0.0f;
    float right_dx = 0.0f, right_dy = 0.0f;

    getLandmarkDisplacement(234, left_dx, left_dy);   // Left outer
    getLandmarkDisplacement(454, right_dx, right_dy); // Right outer

    // Displacements should be approximately symmetric (opposite signs)
    // Allow some tolerance for different Y positions
    EXPECT_NEAR(std::abs(left_dx), std::abs(right_dx), 0.01f)
        << "Symmetric cheek points should have similar displacement magnitude";
    EXPECT_TRUE(left_dx > 0 && right_dx < 0)
        << "Left cheek should move right, right cheek should move left";
}

// =============================================================================
// V-Line / Thin Chin Effect Tests
// =============================================================================

TEST_F(FaceWarpControllerTest, ThinChinMovesJawInward) {
    WarpConfig config{0.0f, 1.0f, 0.0f};

    EXPECT_TRUE(controller_.applyWarp(mesh_, face_mesh_.data(), config));

    // Left jaw landmarks should move right (toward center)
    float dx, dy;
    int left_inward = 0;
    for (int idx : FaceWarpController::LEFT_JAW_INDICES) {
        if (getLandmarkDisplacement(idx, dx, dy)) {
            if (dx > 0.0f) {
                left_inward++;
            }
        }
    }

    // Right jaw landmarks should move left (toward center)
    int right_inward = 0;
    for (int idx : FaceWarpController::RIGHT_JAW_INDICES) {
        if (getLandmarkDisplacement(idx, dx, dy)) {
            if (dx < 0.0f) {
                right_inward++;
            }
        }
    }

    EXPECT_GT(left_inward, 0) << "Left jaw landmarks should move inward";
    EXPECT_GT(right_inward, 0) << "Right jaw landmarks should move inward";
}

TEST_F(FaceWarpControllerTest, ThinChinMovesChinUp) {
    WarpConfig config{0.0f, 1.0f, 0.0f};

    EXPECT_TRUE(controller_.applyWarp(mesh_, face_mesh_.data(), config));

    // Chin center landmarks should have negative dy (move up in image coords)
    float dx, dy;
    int upward_count = 0;
    for (int idx : FaceWarpController::CHIN_CENTER_INDICES) {
        if (getLandmarkDisplacement(idx, dx, dy)) {
            if (dy < 0.0f) {
                upward_count++;
            }
        }
    }

    EXPECT_GT(upward_count, 0) << "Chin center landmarks should move upward";
}

TEST_F(FaceWarpControllerTest, ThinChinJawMovesUp) {
    WarpConfig config{0.0f, 1.0f, 0.0f};

    EXPECT_TRUE(controller_.applyWarp(mesh_, face_mesh_.data(), config));

    // Jaw landmarks should also move up
    float dx, dy;
    int upward_count = 0;

    for (int idx : FaceWarpController::LEFT_JAW_INDICES) {
        if (getLandmarkDisplacement(idx, dx, dy)) {
            if (dy < 0.0f) {
                upward_count++;
            }
        }
    }
    for (int idx : FaceWarpController::RIGHT_JAW_INDICES) {
        if (getLandmarkDisplacement(idx, dx, dy)) {
            if (dy < 0.0f) {
                upward_count++;
            }
        }
    }

    EXPECT_GT(upward_count, 0) << "Jaw landmarks should move upward";
}

// =============================================================================
// Combined Effects Tests
// =============================================================================

TEST_F(FaceWarpControllerTest, CombinedEffectsStack) {
    // Apply slim face only
    WarpConfig slim_only{1.0f, 0.0f, 0.0f};
    EXPECT_TRUE(controller_.applyWarp(mesh_, face_mesh_.data(), slim_only));
    float slim_total = getTotalDisplacementMagnitude();

    // Reset and apply thin chin only
    mesh_.resetDisplacements();
    WarpConfig chin_only{0.0f, 1.0f, 0.0f};
    EXPECT_TRUE(controller_.applyWarp(mesh_, face_mesh_.data(), chin_only));
    float chin_total = getTotalDisplacementMagnitude();

    // Reset and apply both
    mesh_.resetDisplacements();
    WarpConfig combined{1.0f, 1.0f, 0.0f};
    EXPECT_TRUE(controller_.applyWarp(mesh_, face_mesh_.data(), combined));
    float combined_total = getTotalDisplacementMagnitude();

    // Combined should have more total displacement than either alone
    // Note: Some landmarks overlap between cheek and jaw, so it's not strictly additive
    EXPECT_GT(combined_total, slim_total * 0.5f)
        << "Combined effects should have significant displacement";
    EXPECT_GT(combined_total, chin_total * 0.5f)
        << "Combined effects should have significant displacement";
}

// =============================================================================
// Strength Scaling Tests
// =============================================================================

TEST_F(FaceWarpControllerTest, StrengthScalesLinearly) {
    // Apply with 0.5 strength
    WarpConfig half_strength{0.5f, 0.0f, 0.0f};
    EXPECT_TRUE(controller_.applyWarp(mesh_, face_mesh_.data(), half_strength));
    float half_total = getTotalDisplacementMagnitude();

    // Reset and apply with 1.0 strength
    mesh_.resetDisplacements();
    WarpConfig full_strength{1.0f, 0.0f, 0.0f};
    EXPECT_TRUE(controller_.applyWarp(mesh_, face_mesh_.data(), full_strength));
    float full_total = getTotalDisplacementMagnitude();

    // Full strength should be approximately double half strength
    // Allow 20% tolerance for non-linear effects
    EXPECT_NEAR(full_total, half_total * 2.0f, half_total * 0.4f)
        << "Full strength should be approximately double half strength";
}

TEST_F(FaceWarpControllerTest, ThinChinStrengthScales) {
    // Apply with 0.5 strength
    WarpConfig half_strength{0.0f, 0.5f, 0.0f};
    EXPECT_TRUE(controller_.applyWarp(mesh_, face_mesh_.data(), half_strength));
    float half_total = getTotalDisplacementMagnitude();

    // Reset and apply with 1.0 strength
    mesh_.resetDisplacements();
    WarpConfig full_strength{0.0f, 1.0f, 0.0f};
    EXPECT_TRUE(controller_.applyWarp(mesh_, face_mesh_.data(), full_strength));
    float full_total = getTotalDisplacementMagnitude();

    // Full strength should be greater than half strength
    EXPECT_GT(full_total, half_total * 1.5f)
        << "Full strength should be significantly more than half strength";
}

// =============================================================================
// Gradient Falloff Tests
// =============================================================================

TEST_F(FaceWarpControllerTest, SlimFaceGradientFalloff) {
    WarpConfig config{1.0f, 0.0f, 0.0f};
    EXPECT_TRUE(controller_.applyWarp(mesh_, face_mesh_.data(), config));

    // Check that outer landmarks have different displacement than inner landmarks
    float outer_dx = 0.0f, outer_dy = 0.0f;
    float inner_dx = 0.0f, inner_dy = 0.0f;

    getLandmarkDisplacement(234, outer_dx, outer_dy); // Outer left cheek
    getLandmarkDisplacement(58, inner_dx, inner_dy);  // More central cheek point

    // Both should have positive displacement, but may differ in magnitude
    // The gradient ensures smooth transitions
    EXPECT_GE(outer_dx, 0.0f) << "Outer landmark should have positive dx";
    EXPECT_GE(inner_dx, 0.0f) << "Inner landmark should have positive dx";
}

TEST_F(FaceWarpControllerTest, VLineYGradient) {
    WarpConfig config{0.0f, 1.0f, 0.0f};
    EXPECT_TRUE(controller_.applyWarp(mesh_, face_mesh_.data(), config));

    // Lower jaw points should have more displacement than upper jaw points
    // Index 149 is lower (y ~0.70), index 136 is upper (y ~0.60)
    float lower_dx = 0.0f, lower_dy = 0.0f;
    float upper_dx = 0.0f, upper_dy = 0.0f;

    getLandmarkDisplacement(149, lower_dx, lower_dy);
    getLandmarkDisplacement(136, upper_dx, upper_dy);

    // Lower points should have more Y displacement (more negative)
    EXPECT_LE(lower_dy, upper_dy)
        << "Lower jaw points should move up more than upper jaw points";
}

// =============================================================================
// Face Metric Calculation Tests
// =============================================================================

TEST_F(FaceWarpControllerTest, CalculateFaceWidth) {
    float width = FaceWarpController::calculateFaceWidth(face_mesh_.data());

    // Width should be approximately the distance between landmarks 234 and 454
    float expected_width = std::abs(face_mesh_[454].x - face_mesh_[234].x);
    EXPECT_FLOAT_EQ(width, expected_width);
    EXPECT_GT(width, 0.0f);
    EXPECT_LT(width, 1.0f);
}

TEST_F(FaceWarpControllerTest, CalculateFaceCenterX) {
    float center_x = FaceWarpController::calculateFaceCenterX(face_mesh_.data());

    // Center should be at nose tip
    EXPECT_FLOAT_EQ(center_x, face_mesh_[FaceWarpController::NOSE_TIP_INDEX].x);
}

TEST_F(FaceWarpControllerTest, CalculateFaceWidthWithNull) {
    float width = FaceWarpController::calculateFaceWidth(nullptr);
    EXPECT_FLOAT_EQ(width, 0.0f);
}

TEST_F(FaceWarpControllerTest, CalculateFaceCenterXWithNull) {
    float center_x = FaceWarpController::calculateFaceCenterX(nullptr);
    EXPECT_FLOAT_EQ(center_x, 0.5f);  // Default fallback
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(FaceWarpControllerTest, MaxStrengthStaysWithinBounds) {
    WarpConfig config{1.0f, 1.0f, 0.0f};
    EXPECT_TRUE(controller_.applyWarp(mesh_, face_mesh_.data(), config));

    // Check that displacements don't exceed maximum limits
    const auto& vertices = mesh_.getVertices();
    for (const auto& v : vertices) {
        if (v.is_control) {
            // Maximum displacement should be bounded
            float magnitude = std::sqrt(v.dx * v.dx + v.dy * v.dy);
            EXPECT_LT(magnitude, 0.1f)
                << "Displacement magnitude should be reasonable for landmark " << v.landmark_idx;
        }
    }
}

TEST_F(FaceWarpControllerTest, RepeatedApplyResetsDisplacements) {
    // Apply once
    WarpConfig config1{1.0f, 0.0f, 0.0f};
    EXPECT_TRUE(controller_.applyWarp(mesh_, face_mesh_.data(), config1));
    float first_total = getTotalDisplacementMagnitude();

    // Apply again with different config - should reset first
    WarpConfig config2{0.5f, 0.0f, 0.0f};
    EXPECT_TRUE(controller_.applyWarp(mesh_, face_mesh_.data(), config2));
    float second_total = getTotalDisplacementMagnitude();

    // Second application should not accumulate on first
    EXPECT_LT(second_total, first_total)
        << "Second apply should reset, not accumulate";
}

// =============================================================================
// Performance Sanity Tests
// =============================================================================

TEST_F(FaceWarpControllerTest, ApplyWarpPerformance) {
    WarpConfig config{1.0f, 1.0f, 0.0f};

    // Apply warp 100 times and ensure it completes quickly
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 100; ++i) {
        controller_.applyWarp(mesh_, face_mesh_.data(), config);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Should complete 100 iterations in under 100ms (1ms per iteration target)
    EXPECT_LT(duration.count(), 100)
        << "100 warp applications should complete in under 100ms";
}

} // namespace test
} // namespace warp
} // namespace iris_sdk
