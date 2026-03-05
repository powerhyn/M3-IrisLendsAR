/**
 * @file test_one_euro_filter.cpp
 * @brief OneEuroFilter 단위 테스트 (P4-W3-04: T2, T6)
 *
 * - T2: OneEuroFilter 수렴, 리셋, 첫 프레임 패스스루, 파라미터 민감도
 * - T6: OneEuroFilter reset() 후 상태 초기화 검증
 *
 * Header-only 구현이므로 iris_sdk 라이브러리 링크 불필요.
 */

#include <gtest/gtest.h>
#include "iris_sdk/one_euro_filter.h"

#include <cmath>

namespace iris_sdk {
namespace test {

// ============================================================================
// LowPassFilter 테스트
// ============================================================================

TEST(LowPassFilterTest, FirstSamplePassthrough) {
    LowPassFilter lpf;
    // 첫 번째 샘플은 alpha에 관계없이 그대로 반환
    float result = lpf.filter(42.0f, 0.5f);
    EXPECT_FLOAT_EQ(result, 42.0f);
    EXPECT_TRUE(lpf.isInitialized());
}

TEST(LowPassFilterTest, BlendsPreviousValue) {
    LowPassFilter lpf;
    lpf.filter(10.0f, 0.5f);  // 첫 번째: 10.0

    // 두 번째: alpha=0.5 → 0.5*20 + 0.5*10 = 15.0
    float result = lpf.filter(20.0f, 0.5f);
    EXPECT_FLOAT_EQ(result, 15.0f);
}

// ============================================================================
// OneEuroFilter 테스트
// ============================================================================

class OneEuroFilterTest : public ::testing::Test {
protected:
    // 명시적 타임스탬프를 사용하여 재현 가능한 테스트 보장
    static constexpr double kDt = 1.0 / 60.0;  // 60fps
};

TEST_F(OneEuroFilterTest, FirstSamplePassthrough) {
    OneEuroFilter filter(1.0f, 0.007f, 1.0f);
    float result = filter.filter(100.0f, 0.0);
    EXPECT_FLOAT_EQ(result, 100.0f);
}

TEST_F(OneEuroFilterTest, ConvergesToStableInput) {
    // 고정값 50.0을 100회 반복하면 필터 출력이 50.0에 수렴해야 함
    OneEuroFilter filter(1.0f, 0.007f, 1.0f);
    constexpr float kTarget = 50.0f;

    double t = 0.0;
    float last = 0.0f;
    for (int i = 0; i < 100; ++i) {
        last = filter.filter(kTarget, t);
        t += kDt;
    }
    EXPECT_NEAR(last, kTarget, 0.1f);
}

TEST_F(OneEuroFilterTest, RespondsToLargeChange) {
    // beta가 클수록 큰 변화에 빠르게 추종
    OneEuroFilter filter(1.0f, 1.0f, 1.0f);  // 높은 beta
    double t = 0.0;

    // 안정 구간: 0.0
    for (int i = 0; i < 30; ++i) {
        filter.filter(0.0f, t);
        t += kDt;
    }

    // 큰 변화: 0 → 100
    float result = 0.0f;
    for (int i = 0; i < 10; ++i) {
        result = filter.filter(100.0f, t);
        t += kDt;
    }

    // 높은 beta로 인해 10프레임 내에 100에 가까이 접근해야 함
    EXPECT_GT(result, 80.0f);
}

TEST_F(OneEuroFilterTest, SmoothsSmallJitter) {
    // 작은 지터를 가진 입력 → 출력 분산이 입력 분산보다 작아야 함
    OneEuroFilter filter(0.5f, 0.01f, 1.0f);  // 낮은 min_cutoff = 강한 스무딩

    constexpr float kCenter = 50.0f;
    constexpr float kJitter = 0.5f;  // +/- 0.5 지터
    double t = 0.0;

    // 워밍업
    for (int i = 0; i < 20; ++i) {
        float noise = (i % 2 == 0) ? kJitter : -kJitter;
        filter.filter(kCenter + noise, t);
        t += kDt;
    }

    // 측정 구간
    float sum_sq_diff = 0.0f;
    constexpr int kSamples = 60;
    for (int i = 0; i < kSamples; ++i) {
        float noise = (i % 2 == 0) ? kJitter : -kJitter;
        float result = filter.filter(kCenter + noise, t);
        float diff = result - kCenter;
        sum_sq_diff += diff * diff;
        t += kDt;
    }

    float output_variance = sum_sq_diff / kSamples;
    float input_variance = kJitter * kJitter;  // 0.25

    // 출력 분산이 입력 분산보다 작아야 함 (스무딩 효과)
    EXPECT_LT(output_variance, input_variance);
}

TEST_F(OneEuroFilterTest, ResetClearsState) {
    // T6: reset() 후 첫 번째 값이 패스스루되어야 함
    OneEuroFilter filter(1.0f, 0.007f, 1.0f);

    // 필터에 데이터 주입
    double t = 0.0;
    for (int i = 0; i < 20; ++i) {
        filter.filter(100.0f, t);
        t += kDt;
    }

    // 리셋
    filter.reset();

    // 리셋 후 새 값은 패스스루
    float result = filter.filter(42.0f, t + 1.0);
    EXPECT_FLOAT_EQ(result, 42.0f);
}

TEST_F(OneEuroFilterTest, TimestampOverload) {
    // 명시적 타임스탬프 오버로드가 동작하는지 확인
    OneEuroFilter filter(1.0f, 0.007f, 1.0f);

    float r1 = filter.filter(10.0f, 0.0);
    EXPECT_FLOAT_EQ(r1, 10.0f);

    float r2 = filter.filter(10.0f, kDt);
    // 동일 값이므로 10.0에 매우 가까워야 함
    EXPECT_NEAR(r2, 10.0f, 0.01f);
}

TEST_F(OneEuroFilterTest, ZeroDtFallback) {
    // dt=0일 때 기본 60fps(1/60) 사용 확인 (0 나누기 방지)
    OneEuroFilter filter(1.0f, 0.007f, 1.0f);
    filter.filter(10.0f, 1.0);

    // 동일 타임스탬프 → dt=0 → 기본값 사용
    float result = filter.filter(20.0f, 1.0);
    // 크래시 없이 유한한 값이 반환되어야 함
    EXPECT_TRUE(std::isfinite(result));
    // 10과 20 사이 값이어야 함
    EXPECT_GT(result, 10.0f);
    EXPECT_LT(result, 20.0f);
}

TEST_F(OneEuroFilterTest, ParameterSettersWork) {
    OneEuroFilter filter(1.0f, 0.007f, 1.0f);

    filter.setMinCutoff(2.0f);
    EXPECT_FLOAT_EQ(filter.minCutoff(), 2.0f);

    filter.setBeta(0.1f);
    EXPECT_FLOAT_EQ(filter.beta(), 0.1f);

    filter.setDCutoff(2.0f);
    EXPECT_FLOAT_EQ(filter.dCutoff(), 2.0f);

    // 파라미터 변경 후에도 필터가 정상 동작
    float result = filter.filter(50.0f, 0.0);
    EXPECT_TRUE(std::isfinite(result));
}

// ============================================================================
// OneEuroFilter2D 테스트
// ============================================================================

TEST(OneEuroFilter2DTest, FiltersXYIndependently) {
    OneEuroFilter2D filter(1.0f, 0.007f, 1.0f);

    float x = 10.0f, y = 20.0f;
    filter.filter(x, y, 0.0);
    // 첫 번째 값은 패스스루
    EXPECT_FLOAT_EQ(x, 10.0f);
    EXPECT_FLOAT_EQ(y, 20.0f);

    // X만 변화, Y는 그대로
    x = 15.0f;
    y = 20.0f;
    filter.filter(x, y, 1.0 / 60.0);

    // X는 10~15 사이, Y는 20에 가까움
    EXPECT_GT(x, 10.0f);
    EXPECT_LT(x, 15.0f);
    EXPECT_NEAR(y, 20.0f, 0.01f);
}

// ============================================================================
// IrisOneEuroFilter 테스트
// ============================================================================

TEST(IrisOneEuroFilterTest, FiltersXYRadius) {
    IrisOneEuroFilter filter(1.5f, 0.05f);

    float x = 100.0f, y = 200.0f, r = 30.0f;
    filter.filter(x, y, r, 0.0);
    // 첫 번째 값 패스스루
    EXPECT_FLOAT_EQ(x, 100.0f);
    EXPECT_FLOAT_EQ(y, 200.0f);
    EXPECT_FLOAT_EQ(r, 30.0f);

    // 두 번째 값: 약간 변화
    x = 102.0f;
    y = 198.0f;
    r = 31.0f;
    filter.filter(x, y, r, 1.0 / 60.0);

    // 스무딩되어 원래 값과 새 값 사이
    EXPECT_GT(x, 100.0f);
    EXPECT_LT(x, 102.0f);
    EXPECT_GT(y, 198.0f);
    EXPECT_LT(y, 200.0f);
    EXPECT_GT(r, 30.0f);
    EXPECT_LT(r, 31.0f);
}

TEST(IrisOneEuroFilterTest, DisablePassesThrough) {
    IrisOneEuroFilter filter(1.5f, 0.05f);
    filter.setEnabled(false);
    EXPECT_FALSE(filter.isEnabled());

    // 첫 번째 값
    float x = 100.0f, y = 200.0f, r = 30.0f;
    filter.filter(x, y, r, 0.0);

    // 두 번째 값: disabled → 원래 값 그대로
    x = 150.0f;
    y = 250.0f;
    r = 40.0f;
    filter.filter(x, y, r, 1.0 / 60.0);

    EXPECT_FLOAT_EQ(x, 150.0f);
    EXPECT_FLOAT_EQ(y, 250.0f);
    EXPECT_FLOAT_EQ(r, 40.0f);
}

}  // namespace test
}  // namespace iris_sdk
