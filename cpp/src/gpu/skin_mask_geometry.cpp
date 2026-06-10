/**
 * @file skin_mask_geometry.cpp
 * @brief 피부 마스크 이마 확장 순수 함수 구현 (P8-W1)
 *
 * 출처: LensSimulator BeautyGeometry.kt#extendForehead (40-61줄) — 값 변경 없이 이식.
 */

#include "iris_sdk/gpu/skin_mask_geometry.h"

#include <cmath>

namespace iris_sdk {
namespace skin_mask {

void extendForehead(float* ovalPx, std::size_t count, float factor) {
    if (ovalPx == nullptr || factor <= 0.0f || count == 0) {
        return;
    }

    const float chinX = ovalPx[kOvalChin * 2];
    const float chinY = ovalPx[kOvalChin * 2 + 1];

    // 얼굴 세로축 단위 벡터 (턱→이마)
    float upX = ovalPx[kOvalForehead * 2] - chinX;
    float upY = ovalPx[kOvalForehead * 2 + 1] - chinY;
    const float len = std::sqrt(upX * upX + upY * upY);
    if (len < 1.0f) {
        return;  // 퇴화 입력 방어
    }
    upX /= len;
    upY /= len;

    // 피벗 = 광대 라인 중점 (이보다 위쪽 점만 확장)
    const float pivotX = (ovalPx[kOvalCheekLeft * 2] + ovalPx[kOvalCheekRight * 2]) * 0.5f;
    const float pivotY = (ovalPx[kOvalCheekLeft * 2 + 1] + ovalPx[kOvalCheekRight * 2 + 1]) * 0.5f;

    for (std::size_t i = 0; i < count; ++i) {
        // 세로축 투영 t>0 (피벗 위쪽)만 축 방향으로 늘림
        const float t = (ovalPx[i * 2] - pivotX) * upX + (ovalPx[i * 2 + 1] - pivotY) * upY;
        if (t > 0.0f) {
            ovalPx[i * 2] += upX * t * factor;
            ovalPx[i * 2 + 1] += upY * t * factor;
        }
    }
}

}  // namespace skin_mask
}  // namespace iris_sdk
