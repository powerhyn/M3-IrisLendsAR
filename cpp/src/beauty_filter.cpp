/**
 * @file beauty_filter.cpp
 * @brief BeautyFilter 구현 - 뷰티 카메라 필터 효과
 *
 * 피부 스무딩, 밝기 조절, 소프트 포커스 등의 뷰티 필터 효과 구현.
 * OpenCV 기반 이미지 처리로 실시간 적용 가능.
 */

#include "iris_sdk/beauty_filter.h"

#include <algorithm>
#include <cmath>
#include <mutex>

// OpenCV 헤더 (조건부 컴파일)
#ifdef IRIS_SDK_HAS_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#endif

// ============================================================================
// 상수 정의
// ============================================================================
namespace {
    /// 최소 프레임 크기
    constexpr int MIN_FRAME_SIZE = 32;

    /// 최대 프레임 크기 (메모리 보호)
    constexpr int MAX_FRAME_SIZE = 8192;

    /// Bilateral Filter 최대 diameter (현재 고정값 사용으로 미사용)
    // constexpr int MAX_BILATERAL_DIAMETER = 15;

    /// Gaussian Blur 최대 커널 크기
    constexpr int MAX_BLUR_KERNEL_SIZE = 31;

    /// 기본 설정값
    constexpr bool DEFAULT_ENABLED = true;
    constexpr float DEFAULT_INTENSITY = 0.5f;
    constexpr float DEFAULT_SMOOTHING = 0.5f;
    constexpr float DEFAULT_BRIGHTNESS = 1.05f;
    constexpr float DEFAULT_SOFT_FOCUS = 0.3f;
}

// ============================================================================
// BeautyFilter 싱글톤 클래스
// ============================================================================

/**
 * @brief BeautyFilter 싱글톤 구현
 *
 * 전역 뷰티 필터 상태 관리 및 이미지 처리 담당.
 * 스레드 안전성을 위해 뮤텍스 사용.
 */
class BeautyFilter {
public:
    /**
     * @brief 싱글톤 인스턴스 반환
     */
    static BeautyFilter& getInstance() {
        static BeautyFilter instance;
        return instance;
    }

    /**
     * @brief 기본 설정 반환
     */
    void getDefaultConfig(BeautyFilterConfig* config) const {
        if (config == nullptr) {
            return;
        }

        config->enabled = DEFAULT_ENABLED;
        config->intensity = DEFAULT_INTENSITY;
        config->smoothing = DEFAULT_SMOOTHING;
        config->brightness = DEFAULT_BRIGHTNESS;
        config->softFocus = DEFAULT_SOFT_FOCUS;
    }

    /**
     * @brief 설정 적용
     */
    IrisSdkError setConfig(const BeautyFilterConfig* config) {
        if (config == nullptr) {
            return IRIS_SDK_NULL_POINTER;
        }

        std::lock_guard<std::mutex> lock(config_mutex_);

        // 유효 범위로 클램핑
        config_.enabled = config->enabled;
        config_.intensity = std::clamp(config->intensity, 0.0f, 1.0f);
        config_.smoothing = std::clamp(config->smoothing, 0.0f, 1.0f);
        config_.brightness = std::clamp(config->brightness, 0.0f, 2.0f);
        config_.softFocus = std::clamp(config->softFocus, 0.0f, 1.0f);

        return IRIS_SDK_OK;
    }

    /**
     * @brief 현재 설정 반환
     */
    IrisSdkError getConfig(BeautyFilterConfig* config) const {
        if (config == nullptr) {
            return IRIS_SDK_NULL_POINTER;
        }

        std::lock_guard<std::mutex> lock(config_mutex_);
        *config = config_;

        return IRIS_SDK_OK;
    }

    /**
     * @brief 필터 활성화 여부 확인
     */
    bool isEnabled() const {
        std::lock_guard<std::mutex> lock(config_mutex_);
        return config_.enabled;
    }

    /**
     * @brief 프레임에 뷰티 필터 적용
     */
    IrisSdkError applyFilter(uint8_t* frame_data, int width, int height,
                             IrisFrameFormat format) {
        // 파라미터 검증
        if (frame_data == nullptr) {
            return IRIS_SDK_NULL_POINTER;
        }

        if (width < MIN_FRAME_SIZE || width > MAX_FRAME_SIZE ||
            height < MIN_FRAME_SIZE || height > MAX_FRAME_SIZE) {
            return IRIS_SDK_INVALID_PARAM;
        }

        // 현재 설정 복사 (뮤텍스 범위 최소화)
        BeautyFilterConfig current_config;
        {
            std::lock_guard<std::mutex> lock(config_mutex_);
            current_config = config_;
        }

        // 필터 비활성화시 조기 반환
        if (!current_config.enabled) {
            return IRIS_SDK_OK;
        }

        // 전체 강도가 0이면 스킵
        if (current_config.intensity < 0.01f) {
            return IRIS_SDK_OK;
        }

#ifdef IRIS_SDK_HAS_OPENCV
        return applyFilterOpenCV(frame_data, width, height, format, current_config);
#else
        // OpenCV 없으면 그냥 성공 반환 (효과 없음)
        (void)format;
        return IRIS_SDK_OK;
#endif
    }

private:
    BeautyFilter() {
        // 기본 설정으로 초기화
        getDefaultConfig(&config_);
    }

    ~BeautyFilter() = default;

    // 복사/이동 금지
    BeautyFilter(const BeautyFilter&) = delete;
    BeautyFilter& operator=(const BeautyFilter&) = delete;
    BeautyFilter(BeautyFilter&&) = delete;
    BeautyFilter& operator=(BeautyFilter&&) = delete;

    BeautyFilterConfig config_;
    mutable std::mutex config_mutex_;

#ifdef IRIS_SDK_HAS_OPENCV
    // 사전 할당 버퍼 (메모리 재사용)
    cv::Mat work_buffer_;
    cv::Mat smooth_buffer_;
    cv::Mat glow_buffer_;
    std::mutex buffer_mutex_;

    /**
     * @brief OpenCV 기반 필터 적용
     */
    IrisSdkError applyFilterOpenCV(uint8_t* frame_data, int width, int height,
                                   IrisFrameFormat format,
                                   const BeautyFilterConfig& config) {
        std::lock_guard<std::mutex> buffer_lock(buffer_mutex_);

        cv::Mat frame;
        bool needs_conversion = false;

        // 포맷에 따라 Mat 생성 및 BGR로 변환
        switch (format) {
            case IRIS_FORMAT_BGRA:
                frame = cv::Mat(height, width, CV_8UC4, frame_data);
                cv::cvtColor(frame, work_buffer_, cv::COLOR_BGRA2BGR);
                needs_conversion = true;
                break;

            case IRIS_FORMAT_RGBA:
                frame = cv::Mat(height, width, CV_8UC4, frame_data);
                cv::cvtColor(frame, work_buffer_, cv::COLOR_RGBA2BGR);
                needs_conversion = true;
                break;

            case IRIS_FORMAT_BGR:
                work_buffer_ = cv::Mat(height, width, CV_8UC3, frame_data).clone();
                break;

            case IRIS_FORMAT_RGB:
                frame = cv::Mat(height, width, CV_8UC3, frame_data);
                cv::cvtColor(frame, work_buffer_, cv::COLOR_RGB2BGR);
                needs_conversion = true;
                break;

            case IRIS_FORMAT_NV21:
            case IRIS_FORMAT_NV12: {
                // YUV420sp to BGR 변환
                int yuv_height = height + height / 2;
                cv::Mat yuv(yuv_height, width, CV_8UC1, frame_data);
                int conversion_code = (format == IRIS_FORMAT_NV21) ?
                                      cv::COLOR_YUV2BGR_NV21 : cv::COLOR_YUV2BGR_NV12;
                cv::cvtColor(yuv, work_buffer_, conversion_code);
                needs_conversion = true;
                break;
            }

            case IRIS_FORMAT_GRAY:
                // 그레이스케일은 지원하지 않음 (컬러 필터이므로)
                return IRIS_SDK_INVALID_FORMAT;

            default:
                return IRIS_SDK_INVALID_FORMAT;
        }

        // 실제 필터 처리
        processBeautyEffect(work_buffer_, config);

        // 원본 포맷으로 다시 변환
        if (needs_conversion) {
            switch (format) {
                case IRIS_FORMAT_BGRA:
                    cv::cvtColor(work_buffer_, frame, cv::COLOR_BGR2BGRA);
                    break;

                case IRIS_FORMAT_RGBA:
                    cv::cvtColor(work_buffer_, frame, cv::COLOR_BGR2RGBA);
                    break;

                case IRIS_FORMAT_RGB:
                    cv::cvtColor(work_buffer_, frame, cv::COLOR_BGR2RGB);
                    break;

                case IRIS_FORMAT_NV21:
                case IRIS_FORMAT_NV12: {
                    // BGR to YUV420sp 변환
                    cv::Mat yuv_out;
                    int conversion_code = (format == IRIS_FORMAT_NV21) ?
                                          cv::COLOR_BGR2YUV_I420 : cv::COLOR_BGR2YUV_I420;
                    cv::cvtColor(work_buffer_, yuv_out, conversion_code);

                    // YUV420 (I420) to NV21/NV12 변환
                    // I420: YYYYYYYY UUUU VVVV
                    // NV21: YYYYYYYY VUVU
                    // NV12: YYYYYYYY UVUV
                    int y_size = width * height;
                    int uv_size = y_size / 4;

                    // Y plane 복사
                    std::memcpy(frame_data, yuv_out.data, y_size);

                    // UV plane 인터리빙
                    uint8_t* u_plane = yuv_out.data + y_size;
                    uint8_t* v_plane = u_plane + uv_size;
                    uint8_t* uv_dst = frame_data + y_size;

                    if (format == IRIS_FORMAT_NV21) {
                        // NV21: V first, then U
                        for (int i = 0; i < uv_size; ++i) {
                            uv_dst[i * 2] = v_plane[i];
                            uv_dst[i * 2 + 1] = u_plane[i];
                        }
                    } else {
                        // NV12: U first, then V
                        for (int i = 0; i < uv_size; ++i) {
                            uv_dst[i * 2] = u_plane[i];
                            uv_dst[i * 2 + 1] = v_plane[i];
                        }
                    }
                    break;
                }

                default:
                    break;
            }
        } else {
            // BGR 포맷인 경우 직접 복사
            std::memcpy(frame_data, work_buffer_.data,
                        static_cast<size_t>(width) * height * 3);
        }

        return IRIS_SDK_OK;
    }

    /**
     * @brief 뷰티 효과 처리 (BGR 이미지)
     */
    void processBeautyEffect(cv::Mat& image, const BeautyFilterConfig& config) {
        // 유효 강도 계산
        float effective_smoothing = config.smoothing * config.intensity;
        float effective_soft_focus = config.softFocus * config.intensity;
        float effective_brightness = 1.0f + (config.brightness - 1.0f) * config.intensity;

        // 1. 피부 스무딩 (Bilateral Filter)
        if (effective_smoothing > 0.01f) {
            applySkinSmoothing(image, effective_smoothing);
        }

        // 2. 소프트 포커스 (Gaussian Blur 블렌딩)
        if (effective_soft_focus > 0.01f) {
            applySoftFocus(image, effective_soft_focus);
        }

        // 3. 밝기 조절
        if (std::abs(effective_brightness - 1.0f) > 0.01f) {
            applyBrightness(image, effective_brightness);
        }
    }

    /**
     * @brief 피부 스무딩 적용 (Bilateral Filter - 성능 최적화 버전)
     *
     * Bilateral Filter는 에지를 보존하면서 노이즈를 제거하여
     * 피부를 자연스럽게 스무딩합니다.
     *
     * 성능 최적화: diameter 5 고정, sigma 값 감소
     */
    void applySkinSmoothing(cv::Mat& image, float strength) {
        // 성능 최적화: diameter 5 고정 (기존: 5~15)
        // bilateral filter는 O(d^2)이므로 diameter가 클수록 급격히 느려짐
        constexpr int diameter = 5;

        // sigma 값: strength에 따라 조절 (기존보다 감소)
        double sigma_color = 30.0 + strength * 50.0;  // 30~80 (기존: 50~150)
        double sigma_space = 30.0 + strength * 50.0;

        cv::bilateralFilter(image, smooth_buffer_, diameter, sigma_color, sigma_space);

        // 원본과 블렌딩하여 자연스럽게
        float blend_alpha = strength * 0.7f;  // 최대 70%까지만 적용
        cv::addWeighted(smooth_buffer_, blend_alpha, image, 1.0f - blend_alpha, 0, image);
    }

    /**
     * @brief 소프트 포커스 적용 (Gaussian Blur 블렌딩)
     *
     * 원본 이미지에 블러된 이미지를 블렌딩하여
     * 소프트한 글로우 효과를 만듭니다.
     */
    void applySoftFocus(cv::Mat& image, float strength) {
        // 커널 크기: strength에 따라 5~31 (홀수)
        int kernel_size = static_cast<int>(5 + strength * 26);
        kernel_size = std::min(kernel_size, MAX_BLUR_KERNEL_SIZE);
        if (kernel_size % 2 == 0) {
            kernel_size += 1;
        }

        cv::GaussianBlur(image, glow_buffer_, cv::Size(kernel_size, kernel_size), 0);

        // 소프트 글로우 블렌딩 (스크린 블렌딩과 유사한 효과)
        float blend_alpha = strength * 0.4f;  // 최대 40%까지만 적용

        // 간단한 블렌딩 (addWeighted)
        cv::addWeighted(glow_buffer_, blend_alpha, image, 1.0f - blend_alpha, 0, image);
    }

    /**
     * @brief 밝기 조절
     */
    void applyBrightness(cv::Mat& image, float brightness) {
        // brightness를 스케일 팩터로 사용
        // 1.0 = 원본, 1.05 = 5% 밝게, 0.95 = 5% 어둡게
        image.convertTo(image, -1, brightness, 0);
    }
#endif // IRIS_SDK_HAS_OPENCV
};

// ============================================================================
// C API 구현
// ============================================================================

extern "C" {

IRIS_SDK_EXPORT void iris_sdk_default_beauty_config(BeautyFilterConfig* config) {
    BeautyFilter::getInstance().getDefaultConfig(config);
}

IRIS_SDK_EXPORT IrisSdkError iris_sdk_set_beauty_filter(const BeautyFilterConfig* config) {
    return BeautyFilter::getInstance().setConfig(config);
}

IRIS_SDK_EXPORT IrisSdkError iris_sdk_get_beauty_filter(BeautyFilterConfig* config) {
    return BeautyFilter::getInstance().getConfig(config);
}

IRIS_SDK_EXPORT bool iris_sdk_is_beauty_filter_enabled(void) {
    return BeautyFilter::getInstance().isEnabled();
}

IRIS_SDK_EXPORT IrisSdkError iris_sdk_apply_beauty_filter(
    uint8_t* frame_data,
    int width,
    int height,
    IrisFrameFormat format) {
    return BeautyFilter::getInstance().applyFilter(frame_data, width, height, format);
}

}  /* extern "C" */
