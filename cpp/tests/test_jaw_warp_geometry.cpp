/**
 * @file test_jaw_warp_geometry.cpp
 * @brief P8-W4-A 턱 V라인 워프 CPU 제어점 산출 단위 테스트 (GoogleTest 7종)
 *
 * 핸드오프 §5 5종 + 추가 2종(좌우 대칭 / roll 등변):
 *   1. DirectionInward               — 각 제어점 변위가 세로축(안쪽)을 향함
 *   2. ZeroStrengthDisabled          — strength=0 → false, count=0, sigma=0
 *   3. EyeHeightDisplacementSmall    — 눈 높이 변위 < 최대 제어점 변위의 5% (렌즈 정합)
 *   4. BoundsContainControlPointsPlus3Sigma — bbox가 14점 포함 + ±3σ, 박스 밖 ≈ 0
 *   5. DegenerateGuard               — 퇴화 입력(축<1px / 폭<8px) → false, sigma=0
 *   6. MirrorSymmetry                — 좌/우 대응 제어점 변위가 X-미러 대칭
 *   7. RollRobustness                — roll 회전해도 face_width(유클리드) 일관(audit #3)
 *
 * 합성 타원 얼굴(실랜드마크 불필요): 필요한 인덱스만 정규화 타원 위에 배치한다.
 *   - 세로축: 이마(10) 상단 ↔ 턱끝(152) 하단 (수직 대칭축).
 *   - 얼굴 폭: 좌 광대(454)/우 광대(234) — 측면.
 *   - 제어점 14: 좌측 7점은 대칭축 왼쪽, 우측 7점은 같은 높이의 X-미러 위치.
 *   - 눈꼬리(33/263)는 눈 높이(광대보다 약간 위)에 배치 — 변위 게이트용.
 */

#include "iris_sdk/warp/jaw_warp_geometry.h"
#include "iris_sdk/types.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace {

using iris_sdk::IrisLandmark;
using iris_sdk::jaw_warp::JawWarpParams;
using iris_sdk::jaw_warp::computeJawWarp;
using iris_sdk::jaw_warp::evaluateWarpDisplacement;
using iris_sdk::jaw_warp::kControlPointCount;
using iris_sdk::jaw_warp::kSideControlPoints;
using iris_sdk::jaw_warp::kLeftSideControl;
using iris_sdk::jaw_warp::kRightSideControl;
using iris_sdk::jaw_warp::kLandmarkForehead;
using iris_sdk::jaw_warp::kLandmarkChin;
using iris_sdk::jaw_warp::kLandmarkCheekLeft;
using iris_sdk::jaw_warp::kLandmarkCheekRight;

constexpr int kFaceMeshCount = 478;
constexpr int kImageW = 1000;
constexpr int kImageH = 1000;
constexpr float kPi = 3.14159265358979323846f;

// 눈꼬리(피험자 좌/우 눈 바깥) 인덱스 — 눈 높이 변위 게이트용.
constexpr int kLandmarkEyeLeft = 263;   ///< 피험자 좌안 바깥
constexpr int kLandmarkEyeRight = 33;   ///< 피험자 우안 바깥

// 합성 타원 파라미터(정규화 좌표 0~1, image 1000 기준 픽셀과 동일 스케일).
// 중심 (0.5, 0.5), 가로 반경 0.20, 세로 반경 0.35.
constexpr float kCenterX = 0.5f;
constexpr float kCenterY = 0.5f;
constexpr float kRadiusX = 0.20f;
constexpr float kRadiusY = 0.35f;

/// 한 점을 정규화 IrisLandmark로 설정(z/visibility는 기하 무관).
void setLm(std::vector<IrisLandmark>& mesh, int idx, float nx, float ny) {
    mesh[idx].x = nx;
    mesh[idx].y = ny;
    mesh[idx].z = 0.0f;
    mesh[idx].visibility = 1.0f;
}

/**
 * 합성 얼굴 생성(정규화). 모든 478점을 중심으로 초기화한 뒤 필요한 인덱스만 배치.
 * @param roll_deg 세로축을 중심에 대해 시계방향 roll 회전(픽셀 종횡비 1:1이라 등방).
 */
std::vector<IrisLandmark> makeSyntheticFace(float roll_deg = 0.0f) {
    std::vector<IrisLandmark> mesh(kFaceMeshCount);
    for (int i = 0; i < kFaceMeshCount; ++i) {
        setLm(mesh, i, kCenterX, kCenterY);
    }

    const float c = std::cos(roll_deg * kPi / 180.0f);
    const float s = std::sin(roll_deg * kPi / 180.0f);
    // (dx,dy)를 중심 기준으로 회전해 정규화 좌표로.
    auto place = [&](int idx, float dx, float dy) {
        const float rx = dx * c - dy * s;
        const float ry = dx * s + dy * c;
        setLm(mesh, idx, kCenterX + rx, kCenterY + ry);
    };

    // 세로축: 이마(상단 y 작음) ↔ 턱끝(하단 y 큼).
    place(kLandmarkForehead, 0.0f, -kRadiusY);
    place(kLandmarkChin, 0.0f, kRadiusY);

    // 광대(얼굴 폭): 중심 높이, 좌우 끝.
    // kLandmarkCheekLeft(454)=피험자 좌(여기선 대칭축 오른쪽 +x),
    // kLandmarkCheekRight(234)=피험자 우(왼쪽 −x). 거리만 쓰므로 부호는 일관성만 필요.
    place(kLandmarkCheekLeft, +kRadiusX, 0.0f);
    place(kLandmarkCheekRight, -kRadiusX, 0.0f);

    // 눈꼬리: 광대보다 약간 위(눈 높이), 좌우 안쪽.
    place(kLandmarkEyeLeft, +kRadiusX * 0.55f, -kRadiusY * 0.18f);
    place(kLandmarkEyeRight, -kRadiusX * 0.55f, -kRadiusY * 0.18f);

    // 제어점 14: 볼 상부(약간 위, 측면) → 턱끝 옆(아래, 중앙 쪽).
    // 인덱스 0..6: y는 중심 위(-0.10·Ry)에서 턱끝 옆(+0.85·Ry)까지 균등,
    //             x는 측면(0.95·Rx)에서 턱끝 옆 안쪽(0.30·Rx)까지 좁혀짐.
    for (int i = 0; i < kSideControlPoints; ++i) {
        const float t = static_cast<float>(i) / (kSideControlPoints - 1);  // 0..1
        const float yOff = (-0.10f + 0.95f * t) * kRadiusY;
        const float xMag = (0.95f - 0.65f * t) * kRadiusX;
        place(kLeftSideControl[i], +xMag, yOff);   // 대칭축 오른쪽
        place(kRightSideControl[i], -xMag, yOff);  // 대칭축 왼쪽 (X-미러)
    }

    return mesh;
}

/// 픽셀 변환된 제어점에서 세로축까지의 수직 거리(미러/회전 무관 검증용).
float perpDistanceToAxis(float px, float py,
                         float originX, float originY,
                         float axisHatX, float axisHatY) {
    const float vx = px - originX;
    const float vy = py - originY;
    const float along = vx * axisHatX + vy * axisHatY;
    const float perpX = vx - along * axisHatX;
    const float perpY = vy - along * axisHatY;
    return std::sqrt(perpX * perpX + perpY * perpY);
}

// 실제 face_mesh 스냅샷 (cpp/tests/golden/baseline/face_closeup__rot0.result.json, frame 960x720).
// 눈높이 변위 게이트(렌즈 정합)는 **실제 얼굴 geometry**로 검증한다 — 합성 타원은 눈-볼
// 수직 분리를 현실적으로 못 살려(최상단 볼 제어점이 눈에 ~0.5σ까지 붙어 누출 과대) 본
// 제약엔 부적합. 실제 얼굴은 볼 제어점이 눈에서 ~1.75σ 떨어져 변위 <5%. [real-data-first]
constexpr int kRealFrameW = 960;
constexpr int kRealFrameH = 720;

std::vector<IrisLandmark> makeRealFace() {
    struct Lm { int idx; float x; float y; };
    static const Lm kReal[] = {
        {10, 0.517359f, 0.413989f},  {152, 0.486712f, 0.945630f},
        {454, 0.688720f, 0.657440f}, {234, 0.361343f, 0.627335f},
        {323, 0.686197f, 0.708543f}, {361, 0.680689f, 0.763564f},
        {288, 0.668082f, 0.821564f}, {397, 0.648326f, 0.866099f},
        {365, 0.624812f, 0.895797f}, {379, 0.596615f, 0.918726f},
        {378, 0.572829f, 0.932322f}, {93, 0.359015f, 0.676693f},
        {132, 0.358829f, 0.730125f}, {58, 0.363728f, 0.787819f},
        {172, 0.373540f, 0.832579f}, {136, 0.386602f, 0.864130f},
        {150, 0.404779f, 0.890875f}, {149, 0.420177f, 0.908766f},
        {33, 0.392351f, 0.574315f},  {263, 0.631697f, 0.589395f},
        {133, 0.461127f, 0.586457f}, {362, 0.559653f, 0.592914f},
        {468, 0.423615f, 0.574366f}, {473, 0.588524f, 0.585086f},
    };
    std::vector<IrisLandmark> mesh(kFaceMeshCount);
    for (int i = 0; i < kFaceMeshCount; ++i) setLm(mesh, i, kCenterX, kCenterY);
    for (const auto& lm : kReal) setLm(mesh, lm.idx, lm.x, lm.y);
    return mesh;
}

// ① 각 제어점 변위가 세로축 안쪽을 향한다 — 변위 적용 후 perp 거리가 줄어든다.
TEST(JawWarpGeometry, DirectionInward) {
    auto mesh = makeSyntheticFace();
    JawWarpParams p{};
    ASSERT_TRUE(computeJawWarp(mesh.data(), kImageW, kImageH, 1.0f, p));
    ASSERT_EQ(kControlPointCount, p.count);

    // 세로축(픽셀): origin=10, hat=normalize(152−10).
    const float ox = mesh[kLandmarkForehead].x * kImageW;
    const float oy = mesh[kLandmarkForehead].y * kImageH;
    const float chx = mesh[kLandmarkChin].x * kImageW;
    const float chy = mesh[kLandmarkChin].y * kImageH;
    const float axLen = std::sqrt((chx - ox) * (chx - ox) + (chy - oy) * (chy - oy));
    const float hx = (chx - ox) / axLen;
    const float hy = (chy - oy) / axLen;

    for (int i = 0; i < p.count; ++i) {
        const float before = perpDistanceToAxis(p.cx[i], p.cy[i], ox, oy, hx, hy);
        const float after = perpDistanceToAxis(p.cx[i] + p.dx[i], p.cy[i] + p.dy[i],
                                               ox, oy, hx, hy);
        EXPECT_LT(after, before) << "control point " << i << " did not move inward";
        // 변위 자체도 안쪽(축으로 향하는 단위벡터와 양의 내적).
        const float perpX = (p.cx[i] - ox) -
            ((p.cx[i] - ox) * hx + (p.cy[i] - oy) * hy) * hx;
        const float perpY = (p.cy[i] - oy) -
            ((p.cx[i] - ox) * hx + (p.cy[i] - oy) * hy) * hy;
        const float perpLen = std::sqrt(perpX * perpX + perpY * perpY);
        // inward 단위벡터 = −perp/|perp|. dot(disp, inward) > 0.
        const float dotInward = -(p.dx[i] * perpX + p.dy[i] * perpY) / perpLen;
        EXPECT_GT(dotInward, 0.0f) << "control point " << i << " displacement not inward";
    }
}

// ② strength=0 → 워프 비활성.
TEST(JawWarpGeometry, ZeroStrengthDisabled) {
    auto mesh = makeSyntheticFace();
    JawWarpParams p{};
    EXPECT_FALSE(computeJawWarp(mesh.data(), kImageW, kImageH, 0.0f, p));
    EXPECT_EQ(0, p.count);
    EXPECT_FLOAT_EQ(0.0f, p.sigma_px);
    // 음수 강도도 동일.
    EXPECT_FALSE(computeJawWarp(mesh.data(), kImageW, kImageH, -0.5f, p));
    EXPECT_EQ(0, p.count);
    EXPECT_FLOAT_EQ(0.0f, p.sigma_px);
}

// ③ 눈 높이 변위 < 최대 제어점 변위 크기의 5% (렌즈 정합 게이트) — 실제 얼굴 geometry.
TEST(JawWarpGeometry, EyeHeightDisplacementSmall) {
    auto mesh = makeRealFace();
    JawWarpParams p{};
    ASSERT_TRUE(computeJawWarp(mesh.data(), kRealFrameW, kRealFrameH, 1.0f, p));

    // 최대 제어점 변위 크기.
    float maxDisp = 0.0f;
    for (int i = 0; i < p.count; ++i) {
        const float m = std::sqrt(p.dx[i] * p.dx[i] + p.dy[i] * p.dy[i]);
        if (m > maxDisp) maxDisp = m;
    }
    ASSERT_GT(maxDisp, 0.0f);

    auto evalAt = [&](int idx) {
        const float ex = mesh[idx].x * kRealFrameW;
        const float ey = mesh[idx].y * kRealFrameH;
        float odx = 0.0f, ody = 0.0f;
        evaluateWarpDisplacement(p, ex, ey, odx, ody);
        return std::sqrt(odx * odx + ody * ody);
    };

    // 눈꼬리(33/263) + 홍채중심(468/473) 전부 < 5% — 렌즈/홍채 영역 무변위.
    for (int idx : {33, 263, 468, 473}) {
        EXPECT_LT(evalAt(idx), maxDisp * 0.05f)
            << "landmark " << idx << " eye-height displacement exceeds 5%";
    }
}

// ④ bounds가 14 제어점 bbox를 포함하고 ±3σ 마진 — 박스 밖 점은 변위 ≈ 0.
TEST(JawWarpGeometry, BoundsContainControlPointsPlus3Sigma) {
    auto mesh = makeSyntheticFace();
    JawWarpParams p{};
    ASSERT_TRUE(computeJawWarp(mesh.data(), kImageW, kImageH, 1.0f, p));

    // 제어점 bbox 재계산.
    float minX = p.cx[0], maxX = p.cx[0], minY = p.cy[0], maxY = p.cy[0];
    for (int i = 1; i < p.count; ++i) {
        minX = std::min(minX, p.cx[i]);
        maxX = std::max(maxX, p.cx[i]);
        minY = std::min(minY, p.cy[i]);
        maxY = std::max(maxY, p.cy[i]);
    }
    const float margin = 3.0f * p.sigma_px;
    EXPECT_NEAR(p.bounds_min_x, minX - margin, 1e-2f);
    EXPECT_NEAR(p.bounds_min_y, minY - margin, 1e-2f);
    EXPECT_NEAR(p.bounds_max_x, maxX + margin, 1e-2f);
    EXPECT_NEAR(p.bounds_max_y, maxY + margin, 1e-2f);

    // 바운딩박스 안에 모든 제어점이 들어간다.
    for (int i = 0; i < p.count; ++i) {
        EXPECT_GE(p.cx[i], p.bounds_min_x);
        EXPECT_LE(p.cx[i], p.bounds_max_x);
        EXPECT_GE(p.cy[i], p.bounds_min_y);
        EXPECT_LE(p.cy[i], p.bounds_max_y);
    }

    // 박스 한참 밖(min에서 1px 더 밖)은 early-out으로 변위 정확히 0.
    float odx = 1.0f, ody = 1.0f;
    evaluateWarpDisplacement(p, p.bounds_min_x - 1.0f, p.bounds_min_y - 1.0f, odx, ody);
    EXPECT_FLOAT_EQ(0.0f, odx);
    EXPECT_FLOAT_EQ(0.0f, ody);
}

// ⑤ 퇴화 입력(모든 랜드마크를 한 점에 모음 → 축<1px, 폭<8px) → 비활성.
TEST(JawWarpGeometry, DegenerateGuard) {
    std::vector<IrisLandmark> mesh(kFaceMeshCount);
    for (int i = 0; i < kFaceMeshCount; ++i) {
        setLm(mesh, i, 0.5f, 0.5f);  // 전부 한 점
    }
    JawWarpParams p{};
    EXPECT_FALSE(computeJawWarp(mesh.data(), kImageW, kImageH, 1.0f, p));
    EXPECT_EQ(0, p.count);
    EXPECT_FLOAT_EQ(0.0f, p.sigma_px);
}

// ⑥ 좌우 대칭 합성 얼굴 → 좌/우 대응 제어점 변위가 X-미러 대칭.
//    (대칭축이 수직이므로 dx 부호 반대, dy 동일, 크기 같음.)
TEST(JawWarpGeometry, MirrorSymmetry) {
    auto mesh = makeSyntheticFace();
    JawWarpParams p{};
    ASSERT_TRUE(computeJawWarp(mesh.data(), kImageW, kImageH, 1.0f, p));

    // 슬롯 0..6 = 좌측, 7..13 = 우측 (동일 i끼리 대응).
    for (int i = 0; i < kSideControlPoints; ++i) {
        const int l = i;
        const int r = kSideControlPoints + i;
        EXPECT_NEAR(p.dx[l], -p.dx[r], 1e-2f) << "pair " << i << " dx not mirrored";
        EXPECT_NEAR(p.dy[l], p.dy[r], 1e-2f) << "pair " << i << " dy differs";
        const float magL = std::sqrt(p.dx[l] * p.dx[l] + p.dy[l] * p.dy[l]);
        const float magR = std::sqrt(p.dx[r] * p.dx[r] + p.dy[r] * p.dy[r]);
        EXPECT_NEAR(magL, magR, 1e-2f) << "pair " << i << " magnitude differs";
    }
}

// ⑦ roll 회전해도 face_width(유클리드)가 비회전 대비 ε 내 동일 + 워프 크기 일관.
//    (|Δx|였다면 cosθ만큼 줄었을 것 — audit #3 수정 게이트.)
TEST(JawWarpGeometry, RollRobustness) {
    auto base = makeSyntheticFace(0.0f);
    auto rolled = makeSyntheticFace(25.0f);

    JawWarpParams pb{}, pr{};
    ASSERT_TRUE(computeJawWarp(base.data(), kImageW, kImageH, 1.0f, pb));
    ASSERT_TRUE(computeJawWarp(rolled.data(), kImageW, kImageH, 1.0f, pr));

    // face_width 직접 비교(유클리드라 회전 불변, 픽셀 1:1 종횡비).
    auto faceWidth = [&](const std::vector<IrisLandmark>& m) {
        const float lx = m[kLandmarkCheekLeft].x * kImageW;
        const float ly = m[kLandmarkCheekLeft].y * kImageH;
        const float rx = m[kLandmarkCheekRight].x * kImageW;
        const float ry = m[kLandmarkCheekRight].y * kImageH;
        return std::sqrt((lx - rx) * (lx - rx) + (ly - ry) * (ly - ry));
    };
    const float wBase = faceWidth(base);
    const float wRolled = faceWidth(rolled);
    EXPECT_NEAR(wBase, wRolled, 1e-1f) << "face_width changed under roll";

    // σ는 face_width 비례 → 회전 후에도 동일.
    EXPECT_NEAR(pb.sigma_px, pr.sigma_px, 1e-1f);

    // 워프 변위 크기 집합이 회전 후에도 일관(같은 제어점은 같은 크기, 방향만 회전).
    for (int i = 0; i < kControlPointCount; ++i) {
        const float magB = std::sqrt(pb.dx[i] * pb.dx[i] + pb.dy[i] * pb.dy[i]);
        const float magR = std::sqrt(pr.dx[i] * pr.dx[i] + pr.dy[i] * pr.dy[i]);
        EXPECT_NEAR(magB, magR, 1e-1f) << "control point " << i << " disp magnitude changed under roll";
    }
}

}  // namespace
