/**
 * @file face_warp_controller.h
 * @brief Face warp effects controller for slim face and V-line effects
 *
 * P2-W4-02: Slim Face / V-Line Effect Implementation
 *
 * This class applies face warping effects using GridMesh displacement control.
 * Effects include:
 * - Slim Face: Moves cheek landmarks inward toward face center
 * - Thin Chin / V-Line: Moves jaw and chin landmarks inward and upward
 * - Enlarge Eyes: (Placeholder for P2-W4-03)
 */

#ifndef IRIS_SDK_WARP_FACE_WARP_CONTROLLER_H
#define IRIS_SDK_WARP_FACE_WARP_CONTROLLER_H

#include "iris_sdk/export.h"
#include "iris_sdk/types.h"
#include "iris_sdk/warp/grid_mesh.h"

#include <array>
#include <cmath>

namespace iris_sdk {
namespace warp {

/**
 * @brief Face warp effect configuration
 *
 * All values are normalized to 0.0 ~ 1.0 range where:
 * - 0.0 = No effect
 * - 1.0 = Maximum effect intensity
 */
struct WarpConfig {
    float slimFace = 0.0f;     ///< Slim face effect (cheeks move inward)
    float thinChin = 0.0f;     ///< V-line effect (chin/jaw move inward and up)
    float enlargeEyes = 0.0f;  ///< Eye enlargement (placeholder for P2-W4-03)

    /**
     * @brief Check if any warp effect is active
     * @return true if at least one effect has non-zero strength
     */
    bool hasActiveEffect() const {
        return slimFace > 0.0f || thinChin > 0.0f || enlargeEyes > 0.0f;
    }

    /**
     * @brief Clamp all values to valid range [0.0, 1.0]
     * @return Clamped configuration
     */
    WarpConfig clamped() const {
        WarpConfig result;
        result.slimFace = std::clamp(slimFace, 0.0f, 1.0f);
        result.thinChin = std::clamp(thinChin, 0.0f, 1.0f);
        result.enlargeEyes = std::clamp(enlargeEyes, 0.0f, 1.0f);
        return result;
    }
};

/**
 * @brief Controller for face warp effects using GridMesh displacement
 *
 * This class applies various face warping effects by setting displacements
 * on control points of a GridMesh. The GridMesh then interpolates these
 * displacements to all vertices using RBF interpolation.
 *
 * MediaPipe Face Mesh uses 478 landmarks. Key landmarks for face warp:
 * - Left Cheek: 234, 93, 132, 58, 172, 136, 150, 149
 * - Right Cheek: 454, 323, 361, 288, 397, 365, 379, 378
 * - Chin Center: 152, 175, 199, 18, 17, 200, 421, 418
 * - Left Jaw: 136, 150, 149, 176, 148
 * - Right Jaw: 365, 379, 378, 400, 377
 * - Nose Tip (reference): 4
 *
 * @note Thread-safe: This class is stateless and can be used from multiple threads
 */
class IRIS_SDK_EXPORT FaceWarpController {
public:
    // =========================================================================
    // Landmark Index Constants (MediaPipe 478 Face Mesh)
    // =========================================================================

    /// Left cheek landmark indices (outer contour)
    static constexpr std::array<int, 8> LEFT_CHEEK_INDICES = {234, 93, 132, 58, 172, 136, 150, 149};

    /// Right cheek landmark indices (outer contour)
    static constexpr std::array<int, 8> RIGHT_CHEEK_INDICES = {454, 323, 361, 288, 397, 365, 379, 378};

    /// Chin center landmark indices
    static constexpr std::array<int, 8> CHIN_CENTER_INDICES = {152, 175, 199, 18, 17, 200, 421, 418};

    /// Left jaw landmark indices
    static constexpr std::array<int, 5> LEFT_JAW_INDICES = {136, 150, 149, 176, 148};

    /// Right jaw landmark indices
    static constexpr std::array<int, 5> RIGHT_JAW_INDICES = {365, 379, 378, 400, 377};

    /// Nose tip landmark index (face center reference)
    static constexpr int NOSE_TIP_INDEX = 4;

    // =========================================================================
    // Effect Parameters
    // =========================================================================

    /// Maximum X displacement ratio for slim face effect (3% of face width)
    static constexpr float MAX_SLIM_FACE_DX = 0.03f;

    /// Maximum X displacement ratio for V-line effect (2% of face width)
    static constexpr float MAX_VLINE_DX = 0.02f;

    /// Maximum Y displacement ratio for V-line effect (2.5% upward)
    static constexpr float MAX_VLINE_DY = 0.025f;

    // =========================================================================
    // Public API
    // =========================================================================

    FaceWarpController() = default;
    ~FaceWarpController() = default;

    // Non-copyable, non-movable (stateless, no need)
    FaceWarpController(const FaceWarpController&) = delete;
    FaceWarpController& operator=(const FaceWarpController&) = delete;

    /**
     * @brief Apply warp effects to mesh based on landmarks and configuration
     *
     * This method:
     * 1. Resets all mesh displacements
     * 2. Applies slim face effect (if slimFace > 0)
     * 3. Applies thin chin / V-line effect (if thinChin > 0)
     * 4. Applies eye enlargement effect (if enlargeEyes > 0) - placeholder
     * 5. Interpolates displacements via RBF
     * 6. Computes final vertex positions
     *
     * @param mesh GridMesh instance with control points set
     * @param face_mesh Face mesh landmarks array (478 points, normalized 0-1)
     * @param config Warp configuration with effect strengths
     * @return true if warp was applied successfully, false otherwise
     */
    bool applyWarp(GridMesh& mesh,
                   const IrisLandmark* face_mesh,
                   const WarpConfig& config);

    /**
     * @brief Calculate face width from landmarks (for displacement scaling)
     *
     * Uses left-right cheek outer landmarks to estimate face width.
     *
     * @param face_mesh Face mesh landmarks array
     * @return Estimated face width in normalized coordinates
     */
    static float calculateFaceWidth(const IrisLandmark* face_mesh);

    /**
     * @brief Calculate face center X coordinate (reference point)
     *
     * Uses nose tip landmark as face center reference.
     *
     * @param face_mesh Face mesh landmarks array
     * @return Face center X coordinate in normalized space
     */
    static float calculateFaceCenterX(const IrisLandmark* face_mesh);

private:
    /**
     * @brief Apply slim face effect (cheeks move toward center)
     *
     * Left cheek landmarks move rightward, right cheek landmarks move leftward.
     * Uses Y-position weight: more effect at cheek center, less at top/bottom.
     * Uses distance-based falloff for natural gradient.
     *
     * @param mesh GridMesh to modify
     * @param face_mesh Face mesh landmarks
     * @param strength Effect strength (0.0 ~ 1.0)
     * @param face_width Face width for displacement scaling
     * @param center_x Face center X coordinate
     */
    void applySlimFace(GridMesh& mesh,
                       const IrisLandmark* face_mesh,
                       float strength,
                       float face_width,
                       float center_x);

    /**
     * @brief Apply thin chin / V-line effect
     *
     * Jaw landmarks move inward AND upward.
     * Chin center landmarks move upward.
     * Uses Y-position weight: more displacement at lower positions.
     *
     * @param mesh GridMesh to modify
     * @param face_mesh Face mesh landmarks
     * @param strength Effect strength (0.0 ~ 1.0)
     * @param face_width Face width for displacement scaling
     * @param center_x Face center X coordinate
     */
    void applyThinChin(GridMesh& mesh,
                       const IrisLandmark* face_mesh,
                       float strength,
                       float face_width,
                       float center_x);

    /**
     * @brief Apply eye enlargement effect (placeholder for P2-W4-03)
     *
     * @param mesh GridMesh to modify
     * @param face_mesh Face mesh landmarks
     * @param strength Effect strength (0.0 ~ 1.0)
     */
    void applyEnlargeEyes(GridMesh& mesh,
                          const IrisLandmark* face_mesh,
                          float strength);

    /**
     * @brief Calculate Y-position weight for smooth gradient effect
     *
     * Weight is higher near the center of the effect region and
     * falls off toward the edges for natural-looking transitions.
     *
     * @param y Current Y coordinate
     * @param min_y Minimum Y of effect region
     * @param max_y Maximum Y of effect region
     * @return Weight factor (0.0 ~ 1.0)
     */
    static float calculateYWeight(float y, float min_y, float max_y);

    /**
     * @brief Calculate distance-based falloff weight
     *
     * Uses cosine falloff for smooth transitions.
     *
     * @param distance Distance from reference point
     * @param max_distance Maximum distance for full falloff
     * @return Weight factor (0.0 ~ 1.0)
     */
    static float calculateDistanceFalloff(float distance, float max_distance);

    /**
     * @brief Register additional control points for face warp effects
     *
     * Adds cheek, jaw, and chin landmarks as control points in the mesh
     * if they are not already registered.
     *
     * @param mesh GridMesh to modify
     * @param face_mesh Face mesh landmarks
     */
    void registerWarpControlPoints(GridMesh& mesh, const IrisLandmark* face_mesh);
};

} // namespace warp
} // namespace iris_sdk

#endif // IRIS_SDK_WARP_FACE_WARP_CONTROLLER_H
