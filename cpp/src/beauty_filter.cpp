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

    /// Gaussian Blur 최대 커널 크기 (P8-W2-C: applySoftFocus 제거로 미사용)
    // constexpr int MAX_BLUR_KERNEL_SIZE = 31;

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
        // P8-W2-C: 곁가지 효과(피부 스무딩 Bilateral / 소프트 포커스 Gaussian) 제거.
        //   레거시 BeautyFilter(V1)의 생존 효과는 밝기 조절 뿐이다.
        //   config.smoothing/softFocus 필드는 D단계 전까지 보존(여기서는 미사용).
        float effective_brightness = 1.0f + (config.brightness - 1.0f) * config.intensity;

        // 밝기 조절
        if (std::abs(effective_brightness - 1.0f) > 0.01f) {
            applyBrightness(image, effective_brightness);
        }
    }

    // P8-W2-C: applySkinSmoothing(Bilateral)/applySoftFocus(Gaussian) 정의 제거.
    //   곁가지 효과로 분류되어 삭제. 레거시 BeautyFilter(V1) 생존 효과는 밝기 뿐.
    //   (smooth_buffer_/glow_buffer_ 멤버 + MAX_BLUR_KERNEL_SIZE는 호출처가 사라져
    //    orphan 상태가 되나, 멤버 변수는 -Wunused-private-field 대상이 아니라 보존.
    //    MAX_BLUR_KERNEL_SIZE는 아래 상수 정의에서 주석 처리하여 미사용 경고 회피.)

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

// ============================================================================
// V2 C API 구현
// ============================================================================

/// V2 설정 저장소 (싱글톤 패턴)
namespace {
    BeautyFilterConfigV2 g_config_v2 = {};
    std::mutex g_config_v2_mutex;
    bool g_config_v2_initialized = false;

    void ensureV2ConfigInitialized() {
        if (!g_config_v2_initialized) {
            g_config_v2 = iris_sdk::BeautyFilterConfigV2Helper::defaults();
            g_config_v2_initialized = true;
        }
    }
}

IRIS_SDK_EXPORT void iris_sdk_default_beauty_config_v2(BeautyFilterConfigV2* config) {
    if (config == nullptr) {
        return;
    }
    *config = iris_sdk::BeautyFilterConfigV2Helper::defaults();
}

IRIS_SDK_EXPORT IrisSdkError iris_sdk_set_beauty_filter_v2(const BeautyFilterConfigV2* config) {
    if (config == nullptr) {
        return IRIS_SDK_NULL_POINTER;
    }

    if (!iris_sdk::BeautyFilterConfigV2Helper::isValid(*config)) {
        return IRIS_SDK_INVALID_PARAM;
    }

    std::lock_guard<std::mutex> lock(g_config_v2_mutex);
    g_config_v2 = *config;
    g_config_v2_initialized = true;

    // V1 API와 동기화 (기본 효과만)
    BeautyFilterConfig v1 = iris_sdk::BeautyFilterConfigV2Helper::toV1(*config);
    BeautyFilter::getInstance().setConfig(&v1);

    return IRIS_SDK_OK;
}

IRIS_SDK_EXPORT IrisSdkError iris_sdk_get_beauty_filter_v2(BeautyFilterConfigV2* config) {
    if (config == nullptr) {
        return IRIS_SDK_NULL_POINTER;
    }

    std::lock_guard<std::mutex> lock(g_config_v2_mutex);
    ensureV2ConfigInitialized();
    *config = g_config_v2;

    return IRIS_SDK_OK;
}

IRIS_SDK_EXPORT IrisSdkError iris_sdk_apply_beauty_filter_v2(
    uint8_t* frame_data,
    int width,
    int height,
    IrisFrameFormat format,
    const IrisResult* iris_result) {

    // 현재 구현: V1 필터 적용 (V2 고급 기능은 BeautyProcessor에서 처리 예정)
    // Face Mesh 연동 및 ROI 기반 처리는 P2-W1-03, P2-W1-04에서 구현

    (void)iris_result;  // 향후 BeautyROIManager에서 사용

    std::lock_guard<std::mutex> lock(g_config_v2_mutex);
    ensureV2ConfigInitialized();

    if (!g_config_v2.enabled) {
        return IRIS_SDK_OK;  // 비활성화 시 패스
    }

    // V1 필터 적용 (기본 피부 효과)
    return BeautyFilter::getInstance().applyFilter(frame_data, width, height, format);
}

IRIS_SDK_EXPORT bool iris_sdk_beauty_gpu_available(void) {
    // GPU 지원 여부 확인 (P2-W3-01에서 구현 예정)
    // 현재는 항상 false 반환 (CPU 전용)
#if defined(__ANDROID__) && defined(IRIS_SDK_HAS_GLES)
    return true;  // Android에서 OpenGL ES 사용 가능
#else
    return false;  // Desktop은 CPU 전용
#endif
}

IRIS_SDK_EXPORT bool iris_sdk_beauty_using_gpu(void) {
    std::lock_guard<std::mutex> lock(g_config_v2_mutex);
    ensureV2ConfigInitialized();

    // GPU 사용 설정 && GPU 사용 가능 여부 확인
    return g_config_v2.useGpu && iris_sdk_beauty_gpu_available();
}

// ============================================================================
// skinQuality 편의 API
// ============================================================================

IRIS_SDK_EXPORT IrisSdkError iris_sdk_set_skin_quality(float quality) {
    if (quality < 0.0f || quality > 1.0f) {
        return IRIS_SDK_INVALID_PARAM;
    }
    ensureV2ConfigInitialized();
    std::lock_guard<std::mutex> lock(g_config_v2_mutex);
    g_config_v2.skinQuality = quality;
    return IRIS_SDK_OK;
}

IRIS_SDK_EXPORT IrisSdkError iris_sdk_get_skin_quality(float* out_quality) {
    if (out_quality == nullptr) {
        return IRIS_SDK_NULL_POINTER;
    }
    ensureV2ConfigInitialized();
    std::lock_guard<std::mutex> lock(g_config_v2_mutex);
    *out_quality = g_config_v2.skinQuality;
    return IRIS_SDK_OK;
}

IRIS_SDK_EXPORT IrisSdkError iris_sdk_set_beauty_preset(IrisBeautyPreset preset) {
    float quality = 0.0f;
    switch (preset) {
        case IRIS_BEAUTY_PRESET_NATURAL:  quality = 0.3f; break;
        case IRIS_BEAUTY_PRESET_MODERATE: quality = 0.5f; break;
        case IRIS_BEAUTY_PRESET_STRONG:   quality = 0.8f; break;
        case IRIS_BEAUTY_PRESET_CUSTOM:
            ensureV2ConfigInitialized();
            {
                std::lock_guard<std::mutex> lock(g_config_v2_mutex);
                g_config_v2.smoothing = 0.0f;
                g_config_v2.softFocus = 0.0f;
            }
            return IRIS_SDK_OK;  // skinQuality 유지, smoothing/softFocus 초기화
        default:
            return IRIS_SDK_INVALID_PARAM;
    }
    ensureV2ConfigInitialized();
    std::lock_guard<std::mutex> lock(g_config_v2_mutex);
    g_config_v2.skinQuality = quality;
    return IRIS_SDK_OK;
}

}  /* extern "C" */
