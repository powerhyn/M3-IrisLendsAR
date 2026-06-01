/**
 * @file eye_render_packet_adapter.h
 * @brief IrisResult → EyeRenderPacket 변환 레이어 (P6-W1)
 *
 * 검출기 산출물을 렌더러 내부 계약(EyeRenderPacket)으로 변환한다.
 * face_mesh 원시 인덱스/수식은 이 레이어에서만 건드리고, 렌더러는
 * packet만 본다. 신규 수식 도입 금지 — 기존 GPULensRenderer static
 * 유틸(fitEyeEllipse, medianLandmarkY)을 재활용한다.
 */

#ifndef IRIS_SDK_GPU_EYE_RENDER_PACKET_ADAPTER_H
#define IRIS_SDK_GPU_EYE_RENDER_PACKET_ADAPTER_H

#include "iris_sdk/gpu/eye_render_packet.h"
#include "iris_sdk/types.h"

namespace iris_sdk::gpu {

/// 한쪽 눈 식별자
enum class EyeSide { Left, Right };

/**
 * @brief IrisResult + 프레임 정보에서 한쪽 눈의 EyeRenderPacket 구성.
 *
 * - iris_center_norm / iris_radius_norm: IrisLandmark[0] 및 *_radius (픽셀→normalize).
 * - ellipse_*: GPULensRenderer::fitEyeEllipse() 결과 재사용. face_mesh_valid=false이면
 *   원형 fallback (rxInner=rxOuter=ry=iris_radius_norm, rotation=0).
 * - eye_top / eye_bottom: GPULensRenderer::medianLandmarkY() 재사용 (기존 눈꺼풀 Y-slab).
 * - visibility: 미검출 시 0. 검출 시 confidence * (1 - eyelid_ratio_side) clamp[0,1].
 * - timestamp_ms: result.timestamp_ms 그대로 전달.
 * - render_confidence(optional): W1 §5.12 — adapter에는 stabilizer가 전달되지 않으므로
 *   packet.visibility를 그대로 담아둔다. 후속 W에서 stabilizer 연결 시 치환.
 * - 나머지 optional 필드: std::nullopt (W1 §5.11 스키마 예약만).
 *
 * 미검출 케이스(left/right_detected=false): visibility=0, 다른 필드는 기본/0값이지만
 * timestamp_ms는 여전히 전달한다.
 */
EyeRenderPacket adaptIrisResult(const IrisResult& result,
                                EyeSide side,
                                int frame_width,
                                int frame_height);

} // namespace iris_sdk::gpu

#endif // IRIS_SDK_GPU_EYE_RENDER_PACKET_ADAPTER_H
