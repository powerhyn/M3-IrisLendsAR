/**
 * @file test_iris_luma_measure.cpp
 * @brief P7-W2 §5.5: iris ROI 평균 luma 측정의 색공간(srgb²+Rec.709) 정합 테스트.
 *
 * 측정 producer(MediaPipeDetector::Impl::calculateIrisLuma)는 익명 클래스의 private
 * 메서드라 직접 링크 불가. 따라서 동일한 픽셀 수식을 mirroring한 reference 구현을
 * 셰이더-equivalent reference와 비교하여 색공간 계약(§5.5)을 ≤1% 오차로 검증한다.
 *
 * 셰이더 정합(shader_sources.cpp):
 *   - toLinearFast(srgb) = srgb*srgb  (정확 sRGB 곡선 아님, fast 근사. :851)
 *   - LUMA_709 = dot(linear, vec3(0.2126, 0.7152, 0.0722))  (:849)
 *
 * 측정 producer가 위와 동일해야 uAvgIrisLum(블렌드 정규화 분모)이 정합한다.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr float kR = 0.2126f;
constexpr float kG = 0.7152f;
constexpr float kB = 0.0722f;
constexpr float kInv255 = 1.0f / 255.0f;

// 셰이더 toLinearFast + LUMA_709 등가 (단일 픽셀, 0~255 입력).
// 이 함수가 "정답(reference)"이며 producer 측정이 이를 평균해야 한다.
float shaderEquivalentLuma(uint8_t r, uint8_t g, uint8_t b) {
    const float sr = static_cast<float>(r) * kInv255;
    const float sg = static_cast<float>(g) * kInv255;
    const float sb = static_cast<float>(b) * kInv255;
    return kR * (sr * sr) + kG * (sg * sg) + kB * (sb * sb);
}

/**
 * @brief producer(calculateIrisLuma)와 동일한 ROI 평균 수식 mirror.
 *
 * mediapipe_detector.cpp Impl::calculateIrisLuma의 픽셀 루프와 동일:
 *   각 픽셀 srgb=v/255, linear=srgb*srgb, luma=dot(linear, Rec.709), 원형마스크 평균.
 * 입력은 flat RGB(uint8) 버퍼(폭*높이*3). 중심/반경은 픽셀 단위(이미 0.65 적용된 r).
 */
float referenceIrisLuma(const std::vector<uint8_t>& rgb, int w, int h,
                        float cx, float cy, float r_px) {
    const float r_sq = r_px * r_px;
    const int x0 = std::max(0, static_cast<int>(std::floor(cx - r_px)));
    const int y0 = std::max(0, static_cast<int>(std::floor(cy - r_px)));
    const int x1 = std::min(w - 1, static_cast<int>(std::ceil(cx + r_px)));
    const int y1 = std::min(h - 1, static_cast<int>(std::ceil(cy + r_px)));

    double sum = 0.0;
    int count = 0;
    for (int y = y0; y <= y1; ++y) {
        const float dy = static_cast<float>(y) - cy;
        for (int x = x0; x <= x1; ++x) {
            const float dx = static_cast<float>(x) - cx;
            if (dx * dx + dy * dy > r_sq) continue;
            const int idx = (y * w + x) * 3;
            const float sr = rgb[idx + 0] * kInv255;
            const float sg = rgb[idx + 1] * kInv255;
            const float sb = rgb[idx + 2] * kInv255;
            sum += static_cast<double>(kR * sr * sr + kG * sg * sg + kB * sb * sb);
            ++count;
        }
    }
    if (count == 0) return -1.0f;
    const float avg = static_cast<float>(sum / count);
    return std::max(0.01f, std::min(0.81f, avg));
}

// 단색 RGB 버퍼 생성.
std::vector<uint8_t> solidBuffer(int w, int h, uint8_t r, uint8_t g, uint8_t b) {
    std::vector<uint8_t> buf(static_cast<size_t>(w) * h * 3);
    for (size_t i = 0; i < buf.size(); i += 3) {
        buf[i + 0] = r;
        buf[i + 1] = g;
        buf[i + 2] = b;
    }
    return buf;
}

// 단색 ROI에서는 평균=단일픽셀 luma여야 한다(clamp 범위 내). ≤1% 검증.
void expectSolidLumaWithin1pct(uint8_t r, uint8_t g, uint8_t b) {
    constexpr int W = 64, H = 64;
    const auto buf = solidBuffer(W, H, r, g, b);
    const float measured = referenceIrisLuma(buf, W, H, 32.0f, 32.0f, 20.0f);
    float expected = shaderEquivalentLuma(r, g, b);
    expected = std::max(0.01f, std::min(0.81f, expected));  // 동일 clamp
    const float tol = std::max(1e-4f, expected * 0.01f);    // ≤1% 상대오차
    EXPECT_NEAR(measured, expected, tol)
        << "rgb=(" << int(r) << "," << int(g) << "," << int(b) << ")";
}

}  // namespace

// 색공간 정합: sRGB 평균이 아니라 linear(srgb²) 평균이어야 한다.
TEST(IrisLumaMeasure, BlackIsClampedLow) {
    expectSolidLumaWithin1pct(0, 0, 0);  // linear 0 → clamp 0.01
}

TEST(IrisLumaMeasure, WhiteIsClampedHigh) {
    expectSolidLumaWithin1pct(255, 255, 255);  // linear 1.0 → clamp 0.81
}

TEST(IrisLumaMeasure, MidGray) {
    // sRGB 0.5(127) → linear 0.25. sRGB 평균(0.5)과 명확히 구분되어야 함.
    const auto buf = solidBuffer(64, 64, 127, 127, 127);
    const float measured = referenceIrisLuma(buf, 64, 64, 32.0f, 32.0f, 20.0f);
    const float expected_linear = std::pow(127.0f / 255.0f, 2.0f);  // ≈0.248
    EXPECT_NEAR(measured, expected_linear, expected_linear * 0.01f);
    // sRGB 평균(0.498)이 아님을 확인 — squaring 누락 회귀 방지.
    EXPECT_LT(measured, 0.40f);
}

TEST(IrisLumaMeasure, PureChannelsMatchRec709) {
    expectSolidLumaWithin1pct(255, 0, 0);  // R: 0.2126
    expectSolidLumaWithin1pct(0, 255, 0);  // G: 0.7152
    expectSolidLumaWithin1pct(0, 0, 255);  // B: 0.0722 → clamp 0.0722
}

// Rec.709 계수 순서(R<B 가중) 정합: G가 가장 밝고 B가 가장 어두워야 한다.
TEST(IrisLumaMeasure, ChannelOrderingRec709) {
    const auto rbuf = solidBuffer(64, 64, 255, 0, 0);
    const auto gbuf = solidBuffer(64, 64, 0, 255, 0);
    const auto bbuf = solidBuffer(64, 64, 0, 0, 255);
    const float lr = referenceIrisLuma(rbuf, 64, 64, 32, 32, 20);
    const float lg = referenceIrisLuma(gbuf, 64, 64, 32, 32, 20);
    const float lb = referenceIrisLuma(bbuf, 64, 64, 32, 32, 20);
    EXPECT_GT(lg, lr);
    EXPECT_GT(lr, lb);
}

// 64×64 synthetic 그라디언트 ROI: 원형마스크 평균이 brute-force 기대값과 ≤1% 일치.
TEST(IrisLumaMeasure, GradientRoiMatchesBruteForce) {
    constexpr int W = 64, H = 64;
    std::vector<uint8_t> buf(static_cast<size_t>(W) * H * 3);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const uint8_t v = static_cast<uint8_t>((x * 4) & 0xFF);  // 가로 그라디언트
            const int idx = (y * W + x) * 3;
            buf[idx + 0] = v;
            buf[idx + 1] = static_cast<uint8_t>(255 - v);
            buf[idx + 2] = 64;
        }
    }
    const float cx = 32.0f, cy = 32.0f, r = 20.0f;
    const float measured = referenceIrisLuma(buf, W, H, cx, cy, r);

    // brute-force: 동일 원형마스크에서 shaderEquivalentLuma 평균.
    const float r_sq = r * r;
    double sum = 0.0;
    int count = 0;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const float dx = x - cx, dy = y - cy;
            if (dx * dx + dy * dy > r_sq) continue;
            const int idx = (y * W + x) * 3;
            sum += shaderEquivalentLuma(buf[idx], buf[idx + 1], buf[idx + 2]);
            ++count;
        }
    }
    float expected = static_cast<float>(sum / count);
    expected = std::max(0.01f, std::min(0.81f, expected));
    EXPECT_NEAR(measured, expected, std::max(1e-4f, expected * 0.01f));
}

// ROI 픽셀이 하나도 없으면(반경 0) -1 sentinel.
TEST(IrisLumaMeasure, EmptyRoiReturnsSentinel) {
    const auto buf = solidBuffer(64, 64, 100, 100, 100);
    // 중심이 이미지 밖 + 작은 반경 → ROI 교집합 없음.
    const float measured = referenceIrisLuma(buf, 64, 64, -100.0f, -100.0f, 1.0f);
    EXPECT_FLOAT_EQ(measured, -1.0f);
}
