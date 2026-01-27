/**
 * @file one_euro_filter.h
 * @brief One-Euro Filter 구현 - 지터링 방지 및 스무딩
 *
 * One-Euro Filter는 적응형 저역 통과 필터로:
 * - 느린 움직임: 강한 스무딩 (지터링 제거)
 * - 빠른 움직임: 약한 스무딩 (빠른 반응)
 *
 * 참고: https://cristal.univ-lille.fr/~casiez/1euro/
 */

#ifndef IRIS_SDK_ONE_EURO_FILTER_H
#define IRIS_SDK_ONE_EURO_FILTER_H

#include <cmath>
#include <chrono>

namespace iris_sdk {

/**
 * @brief 저역 통과 필터 (Low-Pass Filter)
 */
class LowPassFilter {
public:
    LowPassFilter(float alpha = 1.0f)
        : alpha_(alpha), initialized_(false), prev_value_(0.0f) {}

    void reset() {
        initialized_ = false;
    }

    float filter(float value, float alpha) {
        alpha_ = alpha;
        if (!initialized_) {
            initialized_ = true;
            prev_value_ = value;
            return value;
        }
        float filtered = alpha * value + (1.0f - alpha) * prev_value_;
        prev_value_ = filtered;
        return filtered;
    }

    float lastValue() const { return prev_value_; }
    bool isInitialized() const { return initialized_; }

private:
    float alpha_;
    bool initialized_;
    float prev_value_;
};

/**
 * @brief One-Euro Filter
 *
 * 적응형 필터로 움직임 속도에 따라 스무딩 강도를 조절합니다.
 */
class OneEuroFilter {
public:
    /**
     * @brief 생성자
     * @param min_cutoff 최소 컷오프 주파수 (낮을수록 더 스무딩, 기본 1.0)
     * @param beta 속도 계수 (높을수록 빠른 움직임에 더 반응, 기본 0.007)
     * @param d_cutoff 미분 필터 컷오프 주파수 (기본 1.0)
     */
    OneEuroFilter(float min_cutoff = 1.0f, float beta = 0.007f, float d_cutoff = 1.0f)
        : min_cutoff_(min_cutoff)
        , beta_(beta)
        , d_cutoff_(d_cutoff)
        , last_time_(-1.0)
        , x_filter_(computeAlpha(d_cutoff))
        , dx_filter_(computeAlpha(d_cutoff)) {}

    /**
     * @brief 필터 리셋
     */
    void reset() {
        last_time_ = -1.0;
        x_filter_.reset();
        dx_filter_.reset();
    }

    /**
     * @brief 값 필터링 (타임스탬프 자동 계산)
     * @param value 입력 값
     * @return 필터링된 값
     */
    float filter(float value) {
        auto now = std::chrono::steady_clock::now();
        double timestamp = std::chrono::duration<double>(now.time_since_epoch()).count();
        return filter(value, timestamp);
    }

    /**
     * @brief 값 필터링
     * @param value 입력 값
     * @param timestamp 타임스탬프 (초 단위)
     * @return 필터링된 값
     */
    float filter(float value, double timestamp) {
        // 첫 번째 샘플
        if (last_time_ < 0.0) {
            last_time_ = timestamp;
            dx_filter_.filter(0.0f, computeAlpha(d_cutoff_));
            return x_filter_.filter(value, computeAlpha(min_cutoff_));
        }

        // 시간 간격 계산
        double dt = timestamp - last_time_;
        if (dt <= 0.0) {
            dt = 1.0 / 60.0;  // 기본값 60fps
        }
        last_time_ = timestamp;

        // 속도 추정 (미분)
        float dx = (value - x_filter_.lastValue()) / static_cast<float>(dt);
        float filtered_dx = dx_filter_.filter(dx, computeAlpha(d_cutoff_, dt));

        // 적응형 컷오프 계산
        float cutoff = min_cutoff_ + beta_ * std::abs(filtered_dx);

        // 메인 필터 적용
        return x_filter_.filter(value, computeAlpha(cutoff, dt));
    }

    // 파라미터 설정
    void setMinCutoff(float value) { min_cutoff_ = value; }
    void setBeta(float value) { beta_ = value; }
    void setDCutoff(float value) { d_cutoff_ = value; }

    // 파라미터 조회
    float minCutoff() const { return min_cutoff_; }
    float beta() const { return beta_; }
    float dCutoff() const { return d_cutoff_; }

private:
    static float computeAlpha(float cutoff, double dt = 1.0 / 60.0) {
        float tau = 1.0f / (2.0f * static_cast<float>(M_PI) * cutoff);
        return 1.0f / (1.0f + tau / static_cast<float>(dt));
    }

    float min_cutoff_;
    float beta_;
    float d_cutoff_;
    double last_time_;
    LowPassFilter x_filter_;
    LowPassFilter dx_filter_;
};

/**
 * @brief 2D 좌표용 One-Euro Filter
 */
class OneEuroFilter2D {
public:
    OneEuroFilter2D(float min_cutoff = 1.0f, float beta = 0.007f, float d_cutoff = 1.0f)
        : x_filter_(min_cutoff, beta, d_cutoff)
        , y_filter_(min_cutoff, beta, d_cutoff) {}

    void reset() {
        x_filter_.reset();
        y_filter_.reset();
    }

    void filter(float& x, float& y) {
        x = x_filter_.filter(x);
        y = y_filter_.filter(y);
    }

    void filter(float& x, float& y, double timestamp) {
        x = x_filter_.filter(x, timestamp);
        y = y_filter_.filter(y, timestamp);
    }

    void setMinCutoff(float value) {
        x_filter_.setMinCutoff(value);
        y_filter_.setMinCutoff(value);
    }

    void setBeta(float value) {
        x_filter_.setBeta(value);
        y_filter_.setBeta(value);
    }

private:
    OneEuroFilter x_filter_;
    OneEuroFilter y_filter_;
};

/**
 * @brief 홍채 좌표용 One-Euro Filter (x, y, radius)
 */
class IrisOneEuroFilter {
public:
    /**
     * @brief 생성자
     * @param min_cutoff 최소 컷오프 (기본 1.5 - 약간의 스무딩)
     * @param beta 속도 계수 (기본 0.05 - 빠른 반응)
     */
    IrisOneEuroFilter(float min_cutoff = 1.5f, float beta = 0.05f)
        : x_filter_(min_cutoff, beta)
        , y_filter_(min_cutoff, beta)
        , radius_filter_(min_cutoff * 0.5f, beta * 0.5f)  // radius는 더 안정적으로
        , enabled_(true) {}

    void reset() {
        x_filter_.reset();
        y_filter_.reset();
        radius_filter_.reset();
    }

    void setEnabled(bool enabled) { enabled_ = enabled; }
    bool isEnabled() const { return enabled_; }

    /**
     * @brief 홍채 좌표 필터링
     * @param x 중심 x 좌표 (in/out)
     * @param y 중심 y 좌표 (in/out)
     * @param radius 반지름 (in/out)
     */
    void filter(float& x, float& y, float& radius) {
        if (!enabled_) return;
        x = x_filter_.filter(x);
        y = y_filter_.filter(y);
        radius = radius_filter_.filter(radius);
    }

    void filter(float& x, float& y, float& radius, double timestamp) {
        if (!enabled_) return;
        x = x_filter_.filter(x, timestamp);
        y = y_filter_.filter(y, timestamp);
        radius = radius_filter_.filter(radius, timestamp);
    }

    // 파라미터 조정
    void setMinCutoff(float value) {
        x_filter_.setMinCutoff(value);
        y_filter_.setMinCutoff(value);
        radius_filter_.setMinCutoff(value * 0.5f);
    }

    void setBeta(float value) {
        x_filter_.setBeta(value);
        y_filter_.setBeta(value);
        radius_filter_.setBeta(value * 0.5f);
    }

private:
    OneEuroFilter x_filter_;
    OneEuroFilter y_filter_;
    OneEuroFilter radius_filter_;
    bool enabled_;
};

}  // namespace iris_sdk

#endif  // IRIS_SDK_ONE_EURO_FILTER_H
