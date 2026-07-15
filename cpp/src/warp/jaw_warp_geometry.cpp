/**
 * @file jaw_warp_geometry.cpp
 * @brief V라인 + 내부 축소 워프 CPU 제어점/파라미터 산출 구현 (P8-W4 / P8-W4B)
 *
 * jaw 그룹(7점/측) + upper 그룹(1점/측)은 LensSim S23+ 검증본 값 무변경.
 * P8-W4B 확장: interior 그룹(4점/측)을 interior_strength로 조건부 패킹 → face-small lite.
 * 전부 픽셀 공간. interior_strength=0 이면 jaw 경로는 기존과 수치 동일(회귀 불변).
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
                    float jaw_strength,
                    float interior_strength,
                    JawWarpParams& out,
                    int interior_taper_preset) {
    // 두 strength 모두 ≤0 또는 null/퇴화 차원 → 비활성(기존 strength≤0 동작 보존).
    if (face_mesh == nullptr || image_width <= 0 || image_height <= 0 ||
        (jaw_strength <= 0.0f && interior_strength <= 0.0f)) {
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

    // 그룹 하나(좌 per_side + 우 per_side)를 슬롯 n부터 연속 패킹하고 n을 전진시킨다.
    // perp-inward 수식은 모든 그룹 공통 — max_disp/taper만 그룹별로 다르다.
    int n = 0;
    auto packGroup = [&](const int* idxL, const int* idxR, const float* taper,
                         int per_side, float max_disp) {
        for (int side = 0; side < 2; ++side) {
            const int* indices = (side == 0) ? idxL : idxR;
            for (int i = 0; i < per_side; ++i) {
                const Vec2 c = toPixel(face_mesh[indices[i]], image_width, image_height);

                // 세로축에 수직인 성분: perp = v − (v·axisHat)·axisHat.
                const Vec2 v = Vec2{c.x - axisOrigin.x, c.y - axisOrigin.y};
                const float along = dot(v, axisHat);
                const Vec2 perp = Vec2{v.x - along * axisHat.x, v.y - along * axisHat.y};
                const float perp_len = length(perp);

                Vec2 disp = Vec2{0.0f, 0.0f};
                if (perp_len >= 1e-6f) {
                    // inward = −normalize(perp) (안쪽 = 세로축 방향).
                    const float scale = -(max_disp * taper[i]) / perp_len;
                    disp = Vec2{perp.x * scale, perp.y * scale};
                }

                out.cx[n] = c.x;
                out.cy[n] = c.y;
                out.dx[n] = disp.x;
                out.dy[n] = disp.y;
                ++n;
            }
        }
    };

    // 조건부 패킹: jaw + upper (jaw_strength) → interior (interior_strength).
    // jaw 그룹은 항상 슬롯 0..13(좌 0..6, 우 7..13)에 먼저 놓여 좌우 대칭 검증을 보존한다.
    if (jaw_strength > 0.0f) {
        const float max_disp_jaw = face_width_px * kMaxDispRatio * jaw_strength;
        packGroup(kLeftSideControl, kRightSideControl, kTaper,
                  kSideControlPoints, max_disp_jaw);   // 슬롯 0..13
        packGroup(kLeftUpperControl, kRightUpperControl, kUpperTaper,
                  kUpperPerSide, max_disp_jaw);        // 슬롯 14..15
    }
    if (interior_strength > 0.0f) {
        // 벤치 임시(P8-W4B): interior taper 프리셋 행 선택. 범위 밖 인덱스는 프리셋 0으로
        // 폴백해 기존 동작(비트 동일)으로 안전 수렴한다. jaw/upper 그룹과 무관.
        const int taper_preset =
            (interior_taper_preset >= 0 && interior_taper_preset < kInteriorTaperPresetCount)
                ? interior_taper_preset
                : 0;
        const float max_disp_interior = face_width_px * kMaxDispRatio * interior_strength;
        packGroup(kLeftInteriorControl, kRightInteriorControl,
                  kInteriorTaperPresets[taper_preset],
                  kInteriorPerSide, max_disp_interior);  // 슬롯 16..23 (jaw 생략 시 0..7)
    }

    // 미사용 슬롯 정리(잔여값 노출 회피 — 셰이더/evaluate는 count까지만 읽음).
    for (int i = n; i < JawWarpParams::kMaxControlPoints; ++i) {
        out.cx[i] = 0.0f;
        out.cy[i] = 0.0f;
        out.dx[i] = 0.0f;
        out.dy[i] = 0.0f;
    }

    // --- σ + 바운딩박스(±3σ 마진, 패킹된 점만으로 산출) ---
    float min_x = out.cx[0], max_x = out.cx[0];
    float min_y = out.cy[0], max_y = out.cy[0];
    for (int i = 1; i < n; ++i) {
        if (out.cx[i] < min_x) min_x = out.cx[i];
        if (out.cx[i] > max_x) max_x = out.cx[i];
        if (out.cy[i] < min_y) min_y = out.cy[i];
        if (out.cy[i] > max_y) max_y = out.cy[i];
    }

    out.sigma_px = face_width_px * kSigmaRatio;
    const float margin = kBoundsSigmaMargin * out.sigma_px;
    out.bounds_min_x = min_x - margin;
    out.bounds_min_y = min_y - margin;
    out.bounds_max_x = max_x + margin;
    out.bounds_max_y = max_y + margin;
    out.count = n;

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
