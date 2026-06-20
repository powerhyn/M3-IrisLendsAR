/**
 * @file eye_render_packet_adapter.cpp
 * @brief IrisResult → EyeRenderPacket 변환 구현 (P6-W1)
 */

#include "iris_sdk/gpu/eye_render_packet_adapter.h"

#include <algorithm>

#include "iris_sdk/gpu/gpu_lens_renderer.h"

namespace iris_sdk::gpu {

namespace {

// gpu_lens_renderer.cpp의 파일-local 상수와 동일값.
// 해당 상수는 소스 파일 내 static이라 외부 참조 불가 → 동일 인덱스 재선언.
// 수식 아님(MediaPipe FaceMesh 표준 인덱스) — "신규 수식 금지" 원칙 불위배.
// ④ §7.3 canonical: kLeft*=피험자 좌안(386/374그룹), kRight*=피험자 우안(159/145그룹).
constexpr int kLeftUpperEyelid[]  = {386, 385, 384};
constexpr int kLeftLowerEyelid[]  = {374, 373, 380};
constexpr int kRightUpperEyelid[] = {159, 160, 161};
constexpr int kRightLowerEyelid[] = {145, 144, 153};
constexpr int kEyelidIndexCount   = 3;

inline float clamp01(float v) {
    return std::max(0.0f, std::min(1.0f, v));
}

} // namespace

EyeRenderPacket adaptIrisResult(const IrisResult& result,
                                EyeSide side,
                                int frame_width,
                                int frame_height) {
    (void)frame_height; // 현재 normalize는 frame_width 기준 (렌더러와 같은 규약)

    EyeRenderPacket packet{};
    packet.timestamp_ms = static_cast<uint64_t>(result.timestamp_ms);

    const bool is_left = (side == EyeSide::Left);
    const bool detected = is_left ? result.left_detected : result.right_detected;

    if (!detected) {
        // 미검출: visibility=0, 나머지는 기본/0. timestamp만 전달. (adapter.h 계약)
        packet.visibility = 0.0f;
        return packet;
    }

    // --- iris center / radius (정규화) ---
    // IrisLandmark.x/y는 이미 [0,1] 정규화. radius는 픽셀 → frame_width 정규화.
    const IrisLandmark& iris_center = is_left ? result.left_iris[0] : result.right_iris[0];
    const float radius_px = is_left ? result.left_radius : result.right_radius;
    const float inv_w = (frame_width > 0) ? (1.0f / static_cast<float>(frame_width)) : 0.0f;

    packet.iris_center_norm = Vec2{iris_center.x, iris_center.y};
    packet.iris_radius_norm = radius_px * inv_w;

    // --- ellipse (aperture mask) ---
    if (result.face_mesh_valid) {
        float cx = 0.0f, cy = 0.0f;
        float rx_inner = 0.0f, rx_outer = 0.0f, ry = 0.0f, rot = 0.0f;
        GPULensRenderer::fitEyeEllipse(result.face_mesh, is_left,
                                       cx, cy, rx_inner, rx_outer, ry, rot);
        packet.ellipse_center   = Vec2{cx, cy};
        packet.ellipse_radii    = Vec3{rx_inner, rx_outer, ry};
        packet.ellipse_rotation = rot;
    } else {
        // face_mesh 없음 → 홍채 원을 aperture로 fallback
        packet.ellipse_center   = packet.iris_center_norm;
        const float r = packet.iris_radius_norm;
        packet.ellipse_radii    = Vec3{r, r, r};
        packet.ellipse_rotation = 0.0f;
    }

    // --- eye_top / eye_bottom (Y-slab fallback) ---
    if (result.face_mesh_valid) {
        const int* upper = is_left ? kLeftUpperEyelid  : kRightUpperEyelid;
        const int* lower = is_left ? kLeftLowerEyelid  : kRightLowerEyelid;
        packet.eye_top    = GPULensRenderer::medianLandmarkY(result.face_mesh, upper, kEyelidIndexCount);
        packet.eye_bottom = GPULensRenderer::medianLandmarkY(result.face_mesh, lower, kEyelidIndexCount);
    } else {
        // face_mesh 없으면 홍채 중심 기준 ±radius 대칭 slab으로 fallback
        packet.eye_top    = packet.iris_center_norm.y - packet.iris_radius_norm;
        packet.eye_bottom = packet.iris_center_norm.y + packet.iris_radius_norm;
    }

    // --- visibility: confidence · (1 - eyelid_ratio) 조합, clamp[0,1] ---
    const float eyelid_ratio = is_left ? result.eyelid_ratio_left : result.eyelid_ratio_right;
    packet.visibility = clamp01(result.confidence * (1.0f - clamp01(eyelid_ratio)));

    // --- render_confidence: W1 §5.12 — stabilizer 미연결이면 visibility 값을 그대로 ---
    packet.render_confidence = packet.visibility;

    // --- avg_iris_luma: P7-W2 §5.5 — detector 실측값(eye별). 미측정이면 nullopt 유지
    //     → 기존 consumer fallback chain(0.1225) 작동. 측정값이면 packet에 설정. ---
    //     >0 가드: -1 sentinel + memset(0)/zero-init(0)을 모두 미측정 처리 (실측은 clamp상 항상 ≥0.01).
    const float measured_luma = is_left ? result.avg_iris_luma_left
                                        : result.avg_iris_luma_right;
    if (measured_luma > 0.0f) {
        packet.avg_iris_luma = measured_luma;
    }

    // pupil_center_norm / head_pose_yaw_roll / reflection_dir / eye_depth_mm:
    // W1 §5.11에 따라 스키마 예약만, nullopt 유지.

    return packet;
}

} // namespace iris_sdk::gpu
