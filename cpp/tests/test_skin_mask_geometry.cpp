/**
 * @file test_skin_mask_geometry.cpp
 * @brief P8-W1 피부 마스크 이마 확장 순수 함수 테스트
 *
 * 원본 LensSimulator BeautyGeometryTest.kt (155-179줄) 5케이스 포팅:
 *   ① 이마 y 축 방향 확장  ② 이마 x 불변  ③ 턱 불변
 *   ④ 광대 거의 불변  ⑤ 전 구간 단조성 (위쪽으로만 이동)
 *
 * 합성 타원: 중심 (200, 300), 가로 반경 100 / 세로 반경 200.
 * 배열 인덱스 0 = 이마(상단, y 작음), 18 = 턱끝(하단), 8/28 = 좌우 광대.
 */

#include "iris_sdk/gpu/skin_mask_geometry.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>

namespace {

using iris_sdk::skin_mask::extendForehead;
using iris_sdk::skin_mask::kFaceOvalCount;
using iris_sdk::skin_mask::kOvalForehead;
using iris_sdk::skin_mask::kOvalChin;
using iris_sdk::skin_mask::kOvalCheekLeft;
using iris_sdk::skin_mask::kOvalCheekRight;

constexpr float kFactor = 0.35f;

// 원본 syntheticOval(): theta = i*10도, x = 200 + 100·sin, y = 300 - 200·cos.
std::array<float, kFaceOvalCount * 2> syntheticOval() {
    std::array<float, kFaceOvalCount * 2> out{};
    for (int i = 0; i < kFaceOvalCount; ++i) {
        const double theta = i * 10.0 * 3.14159265358979323846 / 180.0;
        out[i * 2] = static_cast<float>(200.0 + 100.0 * std::sin(theta));
        out[i * 2 + 1] = static_cast<float>(300.0 - 200.0 * std::cos(theta));
    }
    return out;
}

// ① 이마(상단)는 축 방향(위)으로 t×factor 이동 — 합성 타원은 세로축이 수직이라 y만 변함.
TEST(SkinMaskGeometry, ForeheadYExtendsAlongAxis) {
    auto oval = syntheticOval();
    const float pivotY = (oval[kOvalCheekLeft * 2 + 1] + oval[kOvalCheekRight * 2 + 1]) * 0.5f;
    const float foreheadYBefore = oval[kOvalForehead * 2 + 1];

    extendForehead(oval.data(), kFaceOvalCount, kFactor);

    const float expectedForeheadY = foreheadYBefore - (pivotY - foreheadYBefore) * kFactor;
    EXPECT_NEAR(expectedForeheadY, oval[kOvalForehead * 2 + 1], 1e-2f);
}

// ② 이마 x 불변 (가로 폭은 늘리지 않음 — 축 방향 변위만).
TEST(SkinMaskGeometry, ForeheadXUnchanged) {
    auto oval = syntheticOval();
    extendForehead(oval.data(), kFaceOvalCount, kFactor);
    EXPECT_NEAR(200.0f, oval[kOvalForehead * 2], 1e-2f);
}

// ③ 턱(피벗 아래)은 완전 불변.
TEST(SkinMaskGeometry, ChinUnchanged) {
    auto before = syntheticOval();
    auto oval = syntheticOval();
    extendForehead(oval.data(), kFaceOvalCount, kFactor);
    EXPECT_NEAR(before[kOvalChin * 2], oval[kOvalChin * 2], 1e-3f);
    EXPECT_NEAR(before[kOvalChin * 2 + 1], oval[kOvalChin * 2 + 1], 1e-3f);
}

// ④ 광대(피벗 높이)는 거의 불변 (t ≈ 0).
TEST(SkinMaskGeometry, CheekNearlyUnchanged) {
    auto before = syntheticOval();
    auto oval = syntheticOval();
    extendForehead(oval.data(), kFaceOvalCount, kFactor);
    EXPECT_NEAR(before[kOvalCheekLeft * 2 + 1], oval[kOvalCheekLeft * 2 + 1], 0.5f);
}

// ⑤ 전 구간 단조성: 위쪽 점들이 항상 기존보다 같거나 위(y 작음)로만 이동.
TEST(SkinMaskGeometry, MonotonicUpwardOnly) {
    auto original = syntheticOval();
    auto oval = syntheticOval();
    extendForehead(oval.data(), kFaceOvalCount, kFactor);
    for (int i = 0; i < kFaceOvalCount; ++i) {
        EXPECT_LE(oval[i * 2 + 1], original[i * 2 + 1] + 1e-3f) << "point " << i << " moved down";
    }
}

// factor <= 0 은 no-op (비용 게이팅 진입점).
TEST(SkinMaskGeometry, NonPositiveFactorNoOp) {
    auto before = syntheticOval();
    auto oval = syntheticOval();
    extendForehead(oval.data(), kFaceOvalCount, 0.0f);
    for (int i = 0; i < kFaceOvalCount * 2; ++i) {
        EXPECT_FLOAT_EQ(before[i], oval[i]);
    }
}

}  // namespace
