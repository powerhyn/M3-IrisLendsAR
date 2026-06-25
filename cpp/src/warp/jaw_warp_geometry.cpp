/**
 * @file jaw_warp_geometry.cpp
 * @brief 턱 V라인 슬림 워프 CPU 제어점/파라미터 산출 구현 (P8-W4-A)
 *
 * 핸드오프 §1/§3/§4를 전부 픽셀 공간으로 이식. 값 변경 없음(LensSim S23+ 검증본).
 * 출처: docs/lenssim-handoff/jaw-vline-warp-handoff-from-lenssimulator.md.
 */

#include "iris_sdk/warp/jaw_warp_geometry.h"

#include <cmath>

namespace iris_sdk {
namespace jaw_warp {

namespace {

/// 2D 벡터 (픽셀 공간 내부 계산 전용 — POD).
struct Vec2 {
    float x;
    float y;
};

/// 랜드마크(정규화)를 픽셀 공간으로 환산.
inline Vec2 toPixel(const IrisLandmark& lm, int image_width, int image_height) {
    return Vec2{lm.x * static_cast<float>(image_width),
                lm.y * static_cast<float>(image_height)};
}

inline float dot(const Vec2& a, const Vec2& b) {
    return a.x * b.x + a.y * b.y;
}

inline float length(const Vec2& v) {
    return std::sqrt(v.x * v.x + v.y * v.y);
}

inline float distance(const Vec2& a, const Vec2& b) {
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

/// out을 "워프 비활성" 상태로 초기화한다(count=0, sigma=0, bounds=0).
inline void disableWarp(JawWarpParams& out) {
    out.count = 0;
    out.sigma_px = 0.0f;
    out.bounds_min_x = 0.0f;
    out.bounds_min_y = 0.0f;
    out.bounds_max_x = 0.0f;
    out.bounds_max_y = 0.0f;
    // cx/cy/dx/dy는 비활성(count=0)에서 의미 없으나, 잔여값 노출을 피해 0으로 정리.
    for (int i = 0; i < JawWarpParams::kMaxControlPoints; ++i) {
        out.cx[i] = 0.0f;
        out.cy[i] = 0.0f;
        out.dx[i] = 0.0f;
        out.dy[i] = 0.0f;
    }
}

}  // namespace

bool computeJawWarp(const IrisLandmark* face_mesh,
                    int image_width,
                    int image_height,
                    float strength,
                    JawWarpParams& out) {
    // strength≤0 또는 null/퇴화 차원 → 비활성.
    if (face_mesh == nullptr || strength <= 0.0f ||
        image_width <= 0 || image_height <= 0) {
        disableWarp(out);
        return false;
    }

    // --- 얼굴 폭: 유클리드 픽셀거리(454↔234) — roll-robust (audit #3) ---
    const Vec2 cheekL = toPixel(face_mesh[kLandmarkCheekLeft], image_width, image_height);
    const Vec2 cheekR = toPixel(face_mesh[kLandmarkCheekRight], image_width, image_height);
    const float face_width_px = distance(cheekL, cheekR);

    // --- 세로축: 이마(10) → 턱끝(152) ---
    const Vec2 axisOrigin = toPixel(face_mesh[kLandmarkForehead], image_width, image_height);
    const Vec2 chin = toPixel(face_mesh[kLandmarkChin], image_width, image_height);
    const Vec2 axis = Vec2{chin.x - axisOrigin.x, chin.y - axisOrigin.y};
    const float axis_len_px = length(axis);

    // --- 퇴화 방어 ---
    if (axis_len_px < kMinAxisLenPx || face_width_px < kMinFaceWidthPx) {
        disableWarp(out);
        return false;
    }

    const Vec2 axisHat = Vec2{axis.x / axis_len_px, axis.y / axis_len_px};
    const float max_disp_px = face_width_px * kMaxDispRatio * strength;

    // --- 제어점 14개: 좌 7 + 우 7 (배열 슬롯 0~6=좌, 7~13=우) ---
    float min_x = 0.0f, min_y = 0.0f, max_x = 0.0f, max_y = 0.0f;
    for (int side = 0; side < 2; ++side) {
        const int* indices = (side == 0) ? kLeftSideControl : kRightSideControl;
        for (int i = 0; i < kSideControlPoints; ++i) {
            const int slot = side * kSideControlPoints + i;
            const Vec2 c = toPixel(face_mesh[indices[i]], image_width, image_height);

            // 세로축에 수직인 성분: perp = v − (v·axisHat)·axisHat.
            const Vec2 v = Vec2{c.x - axisOrigin.x, c.y - axisOrigin.y};
            const float along = dot(v, axisHat);
            const Vec2 perp = Vec2{v.x - along * axisHat.x, v.y - along * axisHat.y};
            const float perp_len = length(perp);

            Vec2 disp = Vec2{0.0f, 0.0f};
            if (perp_len >= 1e-6f) {
                // inward = −normalize(perp) (안쪽 = 세로축 방향).
                const float scale = -(max_disp_px * kTaper[i]) / perp_len;
                disp = Vec2{perp.x * scale, perp.y * scale};
            }

            out.cx[slot] = c.x;
            out.cy[slot] = c.y;
            out.dx[slot] = disp.x;
            out.dy[slot] = disp.y;

            if (slot == 0) {
                min_x = max_x = c.x;
                min_y = max_y = c.y;
            } else {
                if (c.x < min_x) min_x = c.x;
                if (c.x > max_x) max_x = c.x;
                if (c.y < min_y) min_y = c.y;
                if (c.y > max_y) max_y = c.y;
            }
        }
    }

    // --- σ + 바운딩박스(±3σ 마진) ---
    out.sigma_px = face_width_px * kSigmaRatio;
    const float margin = kBoundsSigmaMargin * out.sigma_px;
    out.bounds_min_x = min_x - margin;
    out.bounds_min_y = min_y - margin;
    out.bounds_max_x = max_x + margin;
    out.bounds_max_y = max_y + margin;
    out.count = kControlPointCount;

    return true;
}

void evaluateWarpDisplacement(const JawWarpParams& p,
                              float px,
                              float py,
                              float& out_dx,
                              float& out_dy) {
    out_dx = 0.0f;
    out_dy = 0.0f;

    // 워프 비활성 또는 바운딩박스(+3σ) 밖이면 early-out (셰이더 §3과 동일).
    if (p.sigma_px <= 0.0f || p.count <= 0) {
        return;
    }
    if (px < p.bounds_min_x || px > p.bounds_max_x ||
        py < p.bounds_min_y || py > p.bounds_max_y) {
        return;
    }

    const float inv2s2 = 0.5f / (p.sigma_px * p.sigma_px);
    float sum_dx = 0.0f;
    float sum_dy = 0.0f;
    const int count = (p.count < JawWarpParams::kMaxControlPoints)
                          ? p.count
                          : JawWarpParams::kMaxControlPoints;
    for (int i = 0; i < count; ++i) {
        const float ddx = px - p.cx[i];
        const float ddy = py - p.cy[i];
        const float w = std::exp(-(ddx * ddx + ddy * ddy) * inv2s2);
        sum_dx += p.dx[i] * w;
        sum_dy += p.dy[i] * w;
    }
    out_dx = sum_dx;
    out_dy = sum_dy;
}

}  // namespace jaw_warp
}  // namespace iris_sdk
