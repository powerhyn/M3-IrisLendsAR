/**
 * @file eye_render_packet.h
 * @brief 렌더러 내부 계약 구조체 (P6-W1)
 *
 * IrisResult(검출기 산출물)와 렌더러 사이의 경계면. 공개 C API는 변경되지 않고
 * 내부 어댑터(eye_render_packet_adapter)를 경유한다. 후속 W(W2~W8)가 의존하는
 * 최소 고정 계약이므로 필드 추가는 "신규 기능 활성화 시점"에만 optional로 확장한다.
 */

#ifndef IRIS_SDK_GPU_EYE_RENDER_PACKET_H
#define IRIS_SDK_GPU_EYE_RENDER_PACKET_H

#include <cstdint>
#include <optional>

namespace iris_sdk::gpu {

// 외부 벡터 의존성(glm 등) 없이 프로젝트 내부 POD로 유지 — types.h 스타일 일관.
struct Vec2 { float x = 0.0f; float y = 0.0f; };
struct Vec3 { float x = 0.0f; float y = 0.0f; float z = 0.0f; };

// 렌더러 입력 계약. occlusion은 별도 필드가 아니라
// visibility(0~1) + aperture mask(ellipse_*)의 조합으로 표현한다. (W1 §5.4 확정)
struct EyeRenderPacket {
    // === 필수 ===
    Vec2  iris_center_norm{};                 ///< 홍채 중심 (정규화 좌표)
    float iris_radius_norm{0.0f};             ///< 홍채 반경 (frame_width 기준 정규화)

    Vec2  ellipse_center{};                   ///< aperture 타원 중심
    Vec3  ellipse_radii{};                    ///< (rxInner=x, rxOuter=y, ry=z)
    float ellipse_rotation{0.0f};             ///< 타원 회전 (rad)

    float eye_top{0.0f};                      ///< Y-slab fallback 상단
    float eye_bottom{0.0f};                   ///< Y-slab fallback 하단

    float    visibility{0.0f};                ///< 0=미검출/완전가림, 1=완전 가시
    uint64_t timestamp_ms{0};

    // === 선택 (없으면 기능 자동 off, priors로 채우지 않음. W1 §5.11 예약) ===
    std::optional<Vec2>  pupil_center_norm;   ///< W8 parallax / 동공 정렬
    std::optional<Vec2>  head_pose_yaw_roll;  ///< W4 env rotation
    std::optional<Vec3>  reflection_dir;      ///< head_pose 대체 경로
    std::optional<float> avg_iris_luma;       ///< W2 blend 정규화 (없으면 self-measure)
    std::optional<float> eye_depth_mm;        ///< 스케일 보정
    std::optional<float> render_confidence;   ///< alpha hysteresis (W1 §5.12: stabilizer.visibility 재활용)
};

} // namespace iris_sdk::gpu

#endif // IRIS_SDK_GPU_EYE_RENDER_PACKET_H
