/**
 * @file lens_renderer.cpp
 * @brief LensRenderer 구현 - 가상 렌즈 렌더링
 *
 * 검출된 홍채 위치에 렌즈 텍스처를 오버레이하는 렌더러 구현.
 * ROI 기반 처리와 다양한 블렌드 모드 지원.
 */

#include "iris_sdk/lens_renderer.h"
#include <chrono>
#include <cmath>
#include <algorithm>

// OpenCV 헤더 (조건부 컴파일)
#ifdef IRIS_SDK_HAS_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#endif

namespace iris_sdk {

// ============================================================
// 상수 정의
// ============================================================
namespace {
    /// 최소 홍채 반지름 (픽셀) - 이보다 작으면 렌더링 스킵
    constexpr float MIN_IRIS_RADIUS = 5.0f;

    /// 최대 홍채 반지름 (픽셀) - 이보다 크면 비정상으로 간주
    constexpr float MAX_IRIS_RADIUS = 200.0f;

    /// 페더링에 사용할 최소 마스크 크기
    constexpr int MIN_MASK_SIZE = 8;

    /// 최대 텍스처 크기 (픽셀) - 메모리 보호
    constexpr int MAX_TEXTURE_SIZE = 4096;

    /// PI 상수
    constexpr float PI = 3.14159265358979323846f;
}

// ============================================================
// 홍채 영역 정보 구조체
// ============================================================
#ifdef IRIS_SDK_HAS_OPENCV
namespace {

/**
 * @brief 홍채 영역 정보
 */
struct IrisRegion {
    cv::Point2f center;     ///< 홍채 중심 (픽셀 좌표)
    float radius;           ///< 홍채 반지름 (픽셀)
    float angle;            ///< 회전 각도 (도)
    bool valid;             ///< 유효성 플래그
};

/**
 * @brief 정규화 좌표를 픽셀 좌표로 변환
 * @param landmark 정규화된 랜드마크
 * @param frame_width 프레임 너비
 * @param frame_height 프레임 높이
 * @return 픽셀 좌표
 */
inline cv::Point2f normalizedToPixel(const IrisLandmark& landmark,
                                     int frame_width, int frame_height) {
    return cv::Point2f(
        landmark.x * static_cast<float>(frame_width),
        landmark.y * static_cast<float>(frame_height)
    );
}

/**
 * @brief 홍채 랜드마크에서 영역 정보 계산
 * @param iris 홍채 랜드마크 배열 (5개: center + 4 boundary)
 * @param frame_width 프레임 너비
 * @param frame_height 프레임 높이
 * @return 홍채 영역 정보
 *
 * @note 방어적 프로그래밍: 이 함수는 IrisResult의 left_iris/right_iris 배열에서
 *       호출되며, 해당 배열은 항상 5개의 요소를 갖도록 types.h에서 정의됨.
 *       C 스타일 배열 파라미터는 크기를 강제하지 않으므로 호출자가
 *       정확히 5개 요소의 배열을 전달해야 함.
 */
IrisRegion calculateIrisRegion(const IrisLandmark iris[5],
                               int frame_width, int frame_height) {
    IrisRegion region;
    region.valid = false;

    // frame_width/height 유효성 검사 (방어적 프로그래밍)
    if (frame_width <= 0 || frame_height <= 0) {
        return region;
    }

    // 중심점 계산 (landmark[0])
    region.center = normalizedToPixel(iris[0], frame_width, frame_height);

    // 중심점 유효성 검사
    if (region.center.x < 0 || region.center.x >= frame_width ||
        region.center.y < 0 || region.center.y >= frame_height) {
        return region;
    }

    // 반지름 계산 (landmark[1,2,3,4]의 평균 거리)
    // landmark[1]: 상단, landmark[2]: 하단, landmark[3]: 좌측, landmark[4]: 우측
    float sum_radius = 0.0f;
    int valid_points = 0;

    for (int i = 1; i <= 4; ++i) {
        cv::Point2f pt = normalizedToPixel(iris[i], frame_width, frame_height);
        float dist = cv::norm(pt - region.center);
        if (dist > 0.0f) {
            sum_radius += dist;
            ++valid_points;
        }
    }

    if (valid_points < 2) {
        return region;
    }

    region.radius = sum_radius / static_cast<float>(valid_points);

    // 반지름 유효성 검사
    if (region.radius < MIN_IRIS_RADIUS || region.radius > MAX_IRIS_RADIUS) {
        return region;
    }

    // 회전 각도 계산 (좌우 랜드마크로 계산)
    // landmark[3]: 좌측, landmark[4]: 우측
    cv::Point2f left_pt = normalizedToPixel(iris[3], frame_width, frame_height);
    cv::Point2f right_pt = normalizedToPixel(iris[4], frame_width, frame_height);

    float dx = right_pt.x - left_pt.x;
    float dy = right_pt.y - left_pt.y;

    if (std::abs(dx) > 0.001f || std::abs(dy) > 0.001f) {
        region.angle = std::atan2(dy, dx) * 180.0f / PI;
    } else {
        region.angle = 0.0f;
    }

    region.valid = true;
    return region;
}

} // anonymous namespace
#endif // IRIS_SDK_HAS_OPENCV

// ============================================================
// Pimpl 구현 클래스
// ============================================================
class LensRenderer::Impl {
public:
    bool initialized = false;
    bool texture_loaded = false;
    double last_render_time_ms = 0.0;

#ifdef IRIS_SDK_HAS_OPENCV
    // 텍스처 데이터
    cv::Mat texture_bgra;       ///< BGRA 텍스처 (알파 포함)
    cv::Mat texture_bgr;        ///< BGR 텍스처 (알파 없는 버전)
    cv::Mat texture_alpha;      ///< 알파 채널 (float, 0~1)

    // 사전 할당 버퍼 (메모리 재사용)
    cv::Mat transformed_texture;    ///< 변환된 텍스처
    cv::Mat transformed_alpha;      ///< 변환된 알파
    cv::Mat feather_mask;           ///< 페더링 마스크
    cv::Mat blend_buffer;           ///< 블렌딩 버퍼

    /**
     * @brief 텍스처를 타겟 크기와 회전으로 변환
     * @param target_size 목표 크기 (정사각형)
     * @param angle 회전 각도 (도)
     * @param out_texture 출력 텍스처 (BGR)
     * @param out_alpha 출력 알파 (float)
     * @return 성공 여부
     */
    bool transformTexture(int target_size, float angle,
                          cv::Mat& out_texture, cv::Mat& out_alpha) {
        if (!texture_loaded || target_size < MIN_MASK_SIZE) {
            return false;
        }

        // 텍스처 유효성 체크
        if (texture_bgr.empty() || texture_alpha.empty()) {
            return false;
        }

        // 리사이즈
        cv::resize(texture_bgr, out_texture,
                   cv::Size(target_size, target_size),
                   0, 0, cv::INTER_LINEAR);
        cv::resize(texture_alpha, out_alpha,
                   cv::Size(target_size, target_size),
                   0, 0, cv::INTER_LINEAR);

        // 회전이 필요한 경우
        if (std::abs(angle) > 0.5f) {
            cv::Point2f center(target_size / 2.0f, target_size / 2.0f);
            cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);

            cv::Mat rotated_texture, rotated_alpha;
            cv::warpAffine(out_texture, rotated_texture, rot_mat,
                           cv::Size(target_size, target_size),
                           cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                           cv::Scalar(0, 0, 0));
            cv::warpAffine(out_alpha, rotated_alpha, rot_mat,
                           cv::Size(target_size, target_size),
                           cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                           cv::Scalar(0));

            out_texture = rotated_texture;
            out_alpha = rotated_alpha;
        }

        return true;
    }

    /**
     * @brief 원형 페더링 마스크 생성
     * @param size 마스크 크기
     * @param feather_amount 페더링 정도 (0~1)
     * @param out_mask 출력 마스크 (float, 0~1)
     */
    void createFeatherMask(int size, float feather_amount, cv::Mat& out_mask) {
        out_mask.create(size, size, CV_32FC1);

        float center = size / 2.0f;
        float outer_radius = center;
        float inner_radius = outer_radius * (1.0f - feather_amount);

        if (inner_radius < 1.0f) {
            inner_radius = 1.0f;
        }

        float feather_width = outer_radius - inner_radius;
        if (feather_width < 1.0f) {
            feather_width = 1.0f;
        }

        for (int y = 0; y < size; ++y) {
            float* row = out_mask.ptr<float>(y);
            for (int x = 0; x < size; ++x) {
                float dx = static_cast<float>(x) - center;
                float dy = static_cast<float>(y) - center;
                float dist = std::sqrt(dx * dx + dy * dy);

                if (dist <= inner_radius) {
                    row[x] = 1.0f;
                } else if (dist >= outer_radius) {
                    row[x] = 0.0f;
                } else {
                    // 부드러운 페이드 (코사인 보간)
                    float t = (dist - inner_radius) / feather_width;
                    row[x] = 0.5f * (1.0f + std::cos(t * PI));
                }
            }
        }
    }

    /**
     * @brief 알파 블렌딩 수행 (Normal 모드)
     * @param dst 대상 프레임
     * @param src 소스 텍스처
     * @param alpha 알파 마스크 (float)
     * @param position 중심 위치
     * @param opacity 전체 투명도
     */
    void alphaBlendNormal(cv::Mat& dst, const cv::Mat& src,
                          const cv::Mat& alpha, cv::Point2i position,
                          float opacity) {
        int half_w = src.cols / 2;
        int half_h = src.rows / 2;

        // ROI 계산
        int dst_x = position.x - half_w;
        int dst_y = position.y - half_h;

        // 소스 시작 오프셋 (경계 클리핑)
        int src_x = 0, src_y = 0;

        if (dst_x < 0) {
            src_x = -dst_x;
            dst_x = 0;
        }
        if (dst_y < 0) {
            src_y = -dst_y;
            dst_y = 0;
        }

        // 유효한 너비/높이 계산
        int valid_w = std::min(src.cols - src_x, dst.cols - dst_x);
        int valid_h = std::min(src.rows - src_y, dst.rows - dst_y);

        if (valid_w <= 0 || valid_h <= 0) {
            return;
        }

        // ROI 생성
        cv::Rect src_rect(src_x, src_y, valid_w, valid_h);
        cv::Rect dst_rect(dst_x, dst_y, valid_w, valid_h);

        cv::Mat dst_roi = dst(dst_rect);
        cv::Mat src_roi = src(src_rect);
        cv::Mat alpha_roi = alpha(src_rect);

        // 채널 수에 따른 처리
        int channels = dst.channels();

        for (int y = 0; y < valid_h; ++y) {
            const float* alpha_row = alpha_roi.ptr<float>(y);
            const uint8_t* src_row = src_roi.ptr<uint8_t>(y);
            uint8_t* dst_row = dst_roi.ptr<uint8_t>(y);

            for (int x = 0; x < valid_w; ++x) {
                float a = alpha_row[x] * opacity;

                if (a > 0.01f) {
                    float inv_a = 1.0f - a;

                    for (int c = 0; c < std::min(channels, 3); ++c) {
                        int idx = x * channels + c;
                        dst_row[idx] = static_cast<uint8_t>(
                            dst_row[idx] * inv_a + src_row[x * 3 + c] * a
                        );
                    }
                }
            }
        }
    }

    /**
     * @brief Multiply 블렌드 모드
     */
    void alphaBlendMultiply(cv::Mat& dst, const cv::Mat& src,
                            const cv::Mat& alpha, cv::Point2i position,
                            float opacity) {
        int half_w = src.cols / 2;
        int half_h = src.rows / 2;

        int dst_x = position.x - half_w;
        int dst_y = position.y - half_h;
        int src_x = 0, src_y = 0;

        if (dst_x < 0) { src_x = -dst_x; dst_x = 0; }
        if (dst_y < 0) { src_y = -dst_y; dst_y = 0; }

        int valid_w = std::min(src.cols - src_x, dst.cols - dst_x);
        int valid_h = std::min(src.rows - src_y, dst.rows - dst_y);

        if (valid_w <= 0 || valid_h <= 0) return;

        cv::Rect src_rect(src_x, src_y, valid_w, valid_h);
        cv::Rect dst_rect(dst_x, dst_y, valid_w, valid_h);

        cv::Mat dst_roi = dst(dst_rect);
        cv::Mat src_roi = src(src_rect);
        cv::Mat alpha_roi = alpha(src_rect);

        int channels = dst.channels();

        for (int y = 0; y < valid_h; ++y) {
            const float* alpha_row = alpha_roi.ptr<float>(y);
            const uint8_t* src_row = src_roi.ptr<uint8_t>(y);
            uint8_t* dst_row = dst_roi.ptr<uint8_t>(y);

            for (int x = 0; x < valid_w; ++x) {
                float a = alpha_row[x] * opacity;

                if (a > 0.01f) {
                    for (int c = 0; c < std::min(channels, 3); ++c) {
                        int idx = x * channels + c;
                        // Multiply: result = src * dst / 255
                        float blended = (src_row[x * 3 + c] * dst_row[idx]) / 255.0f;
                        dst_row[idx] = static_cast<uint8_t>(
                            dst_row[idx] * (1.0f - a) + blended * a
                        );
                    }
                }
            }
        }
    }

    /**
     * @brief Screen 블렌드 모드
     */
    void alphaBlendScreen(cv::Mat& dst, const cv::Mat& src,
                          const cv::Mat& alpha, cv::Point2i position,
                          float opacity) {
        int half_w = src.cols / 2;
        int half_h = src.rows / 2;

        int dst_x = position.x - half_w;
        int dst_y = position.y - half_h;
        int src_x = 0, src_y = 0;

        if (dst_x < 0) { src_x = -dst_x; dst_x = 0; }
        if (dst_y < 0) { src_y = -dst_y; dst_y = 0; }

        int valid_w = std::min(src.cols - src_x, dst.cols - dst_x);
        int valid_h = std::min(src.rows - src_y, dst.rows - dst_y);

        if (valid_w <= 0 || valid_h <= 0) return;

        cv::Rect src_rect(src_x, src_y, valid_w, valid_h);
        cv::Rect dst_rect(dst_x, dst_y, valid_w, valid_h);

        cv::Mat dst_roi = dst(dst_rect);
        cv::Mat src_roi = src(src_rect);
        cv::Mat alpha_roi = alpha(src_rect);

        int channels = dst.channels();

        for (int y = 0; y < valid_h; ++y) {
            const float* alpha_row = alpha_roi.ptr<float>(y);
            const uint8_t* src_row = src_roi.ptr<uint8_t>(y);
            uint8_t* dst_row = dst_roi.ptr<uint8_t>(y);

            for (int x = 0; x < valid_w; ++x) {
                float a = alpha_row[x] * opacity;

                if (a > 0.01f) {
                    for (int c = 0; c < std::min(channels, 3); ++c) {
                        int idx = x * channels + c;
                        // Screen: result = 255 - (255 - src) * (255 - dst) / 255
                        float s = src_row[x * 3 + c];
                        float d = dst_row[idx];
                        float blended = 255.0f - ((255.0f - s) * (255.0f - d)) / 255.0f;
                        dst_row[idx] = static_cast<uint8_t>(
                            d * (1.0f - a) + blended * a
                        );
                    }
                }
            }
        }
    }

    /**
     * @brief Overlay 블렌드 모드
     */
    void alphaBlendOverlay(cv::Mat& dst, const cv::Mat& src,
                           const cv::Mat& alpha, cv::Point2i position,
                           float opacity) {
        int half_w = src.cols / 2;
        int half_h = src.rows / 2;

        int dst_x = position.x - half_w;
        int dst_y = position.y - half_h;
        int src_x = 0, src_y = 0;

        if (dst_x < 0) { src_x = -dst_x; dst_x = 0; }
        if (dst_y < 0) { src_y = -dst_y; dst_y = 0; }

        int valid_w = std::min(src.cols - src_x, dst.cols - dst_x);
        int valid_h = std::min(src.rows - src_y, dst.rows - dst_y);

        if (valid_w <= 0 || valid_h <= 0) return;

        cv::Rect src_rect(src_x, src_y, valid_w, valid_h);
        cv::Rect dst_rect(dst_x, dst_y, valid_w, valid_h);

        cv::Mat dst_roi = dst(dst_rect);
        cv::Mat src_roi = src(src_rect);
        cv::Mat alpha_roi = alpha(src_rect);

        int channels = dst.channels();

        for (int y = 0; y < valid_h; ++y) {
            const float* alpha_row = alpha_roi.ptr<float>(y);
            const uint8_t* src_row = src_roi.ptr<uint8_t>(y);
            uint8_t* dst_row = dst_roi.ptr<uint8_t>(y);

            for (int x = 0; x < valid_w; ++x) {
                float a = alpha_row[x] * opacity;

                if (a > 0.01f) {
                    for (int c = 0; c < std::min(channels, 3); ++c) {
                        int idx = x * channels + c;
                        float s = src_row[x * 3 + c] / 255.0f;
                        float d = dst_row[idx] / 255.0f;

                        // Overlay: if d < 0.5: 2*s*d, else: 1 - 2*(1-s)*(1-d)
                        float blended;
                        if (d < 0.5f) {
                            blended = 2.0f * s * d;
                        } else {
                            blended = 1.0f - 2.0f * (1.0f - s) * (1.0f - d);
                        }
                        blended = std::clamp(blended, 0.0f, 1.0f) * 255.0f;

                        dst_row[idx] = static_cast<uint8_t>(
                            dst_row[idx] * (1.0f - a) + blended * a
                        );
                    }
                }
            }
        }
    }

    /**
     * @brief 블렌드 모드에 따라 적절한 블렌딩 수행
     */
    void applyBlend(cv::Mat& dst, const cv::Mat& src,
                    const cv::Mat& alpha, cv::Point2i position,
                    float opacity, BlendMode mode) {
        switch (mode) {
            case BlendMode::Multiply:
                alphaBlendMultiply(dst, src, alpha, position, opacity);
                break;
            case BlendMode::Screen:
                alphaBlendScreen(dst, src, alpha, position, opacity);
                break;
            case BlendMode::Overlay:
                alphaBlendOverlay(dst, src, alpha, position, opacity);
                break;
            case BlendMode::Normal:
            default:
                alphaBlendNormal(dst, src, alpha, position, opacity);
                break;
        }
    }

    /**
     * @brief 단일 눈에 렌즈 렌더링
     * @param frame 대상 프레임
     * @param region 홍채 영역 정보
     * @param config 렌더링 설정
     * @return 성공 여부
     */
    bool renderEye(cv::Mat& frame, const IrisRegion& region,
                   const LensConfig& config) {
        if (!region.valid || !texture_loaded) {
            return false;
        }

        // 목표 크기 계산 (스케일 적용)
        int target_size = static_cast<int>(region.radius * 2.0f * config.scale);
        if (target_size < MIN_MASK_SIZE) {
            return false;
        }

        // 텍스처 변환
        if (!transformTexture(target_size, region.angle,
                              transformed_texture, transformed_alpha)) {
            return false;
        }

        // 페더링 마스크 생성
        createFeatherMask(target_size, config.edge_feather, feather_mask);

        // 최종 알파 마스크 = 텍스처 알파 * 페더 마스크
        cv::Mat final_alpha;
        cv::multiply(transformed_alpha, feather_mask, final_alpha);

        // 중심 위치 계산 (오프셋 적용)
        cv::Point2i center(
            static_cast<int>(region.center.x + config.offset_x * frame.cols),
            static_cast<int>(region.center.y + config.offset_y * frame.rows)
        );

        // 블렌딩 적용
        applyBlend(frame, transformed_texture, final_alpha,
                   center, config.opacity, config.blend_mode);

        return true;
    }
#endif // IRIS_SDK_HAS_OPENCV
};

// ============================================================
// LensRenderer 공개 인터페이스 구현
// ============================================================

LensRenderer::LensRenderer()
    : impl_(std::make_unique<Impl>()) {
}

LensRenderer::~LensRenderer() {
    release();
}

LensRenderer::LensRenderer(LensRenderer&&) noexcept = default;
LensRenderer& LensRenderer::operator=(LensRenderer&&) noexcept = default;

bool LensRenderer::initialize() {
    if (impl_->initialized) {
        return true;
    }

#ifdef IRIS_SDK_HAS_OPENCV
    impl_->initialized = true;
    return true;
#else
    // OpenCV 없이는 동작 불가
    return false;
#endif
}

void LensRenderer::release() {
#ifdef IRIS_SDK_HAS_OPENCV
    impl_->texture_bgra.release();
    impl_->texture_bgr.release();
    impl_->texture_alpha.release();
    impl_->transformed_texture.release();
    impl_->transformed_alpha.release();
    impl_->feather_mask.release();
    impl_->blend_buffer.release();
#endif

    impl_->texture_loaded = false;
    impl_->initialized = false;
}

bool LensRenderer::isInitialized() const noexcept {
    return impl_ && impl_->initialized;
}

bool LensRenderer::loadTexture(const std::string& texture_path) {
    if (!impl_ || !impl_->initialized) {
        return false;
    }

#ifdef IRIS_SDK_HAS_OPENCV
    // 알파 채널 포함하여 읽기
    cv::Mat loaded = cv::imread(texture_path, cv::IMREAD_UNCHANGED);
    if (loaded.empty()) {
        return false;
    }

    // 텍스처 크기 검증 (메모리 보호)
    if (loaded.cols > MAX_TEXTURE_SIZE || loaded.rows > MAX_TEXTURE_SIZE) {
        return false;
    }

    // 채널 수에 따라 처리
    int channels = loaded.channels();

    if (channels == 4) {
        // BGRA
        impl_->texture_bgra = loaded;
        cv::cvtColor(loaded, impl_->texture_bgr, cv::COLOR_BGRA2BGR);

        // 알파 채널 추출 및 정규화
        std::vector<cv::Mat> bgra_channels;
        cv::split(impl_->texture_bgra, bgra_channels);
        bgra_channels[3].convertTo(impl_->texture_alpha, CV_32FC1, 1.0 / 255.0);

    } else if (channels == 3) {
        // BGR (알파 없음)
        impl_->texture_bgr = loaded;
        cv::cvtColor(loaded, impl_->texture_bgra, cv::COLOR_BGR2BGRA);

        // 완전 불투명 알파
        impl_->texture_alpha = cv::Mat::ones(loaded.size(), CV_32FC1);

    } else if (channels == 1) {
        // Grayscale
        cv::cvtColor(loaded, impl_->texture_bgr, cv::COLOR_GRAY2BGR);
        cv::cvtColor(impl_->texture_bgr, impl_->texture_bgra, cv::COLOR_BGR2BGRA);
        impl_->texture_alpha = cv::Mat::ones(loaded.size(), CV_32FC1);

    } else {
        return false;
    }

    impl_->texture_loaded = true;
    return true;
#else
    (void)texture_path;
    return false;
#endif
}

bool LensRenderer::loadTexture(const uint8_t* data, int width, int height) {
    if (!impl_ || !impl_->initialized || data == nullptr || width <= 0 || height <= 0) {
        return false;
    }

    // 텍스처 크기 검증 (메모리 보호)
    if (width > MAX_TEXTURE_SIZE || height > MAX_TEXTURE_SIZE) {
        return false;
    }

#ifdef IRIS_SDK_HAS_OPENCV
    // RGBA 데이터에서 Mat 생성
    cv::Mat rgba(height, width, CV_8UC4, const_cast<uint8_t*>(data));

    // BGRA로 변환
    cv::cvtColor(rgba, impl_->texture_bgra, cv::COLOR_RGBA2BGRA);
    cv::cvtColor(impl_->texture_bgra, impl_->texture_bgr, cv::COLOR_BGRA2BGR);

    // 알파 채널 추출
    std::vector<cv::Mat> bgra_channels;
    cv::split(impl_->texture_bgra, bgra_channels);
    bgra_channels[3].convertTo(impl_->texture_alpha, CV_32FC1, 1.0 / 255.0);

    impl_->texture_loaded = true;
    return true;
#else
    (void)data;
    (void)width;
    (void)height;
    return false;
#endif
}

void LensRenderer::unloadTexture() {
#ifdef IRIS_SDK_HAS_OPENCV
    impl_->texture_bgra.release();
    impl_->texture_bgr.release();
    impl_->texture_alpha.release();
#endif
    impl_->texture_loaded = false;
}

bool LensRenderer::hasTexture() const noexcept {
    return impl_ && impl_->texture_loaded;
}

bool LensRenderer::render(cv::Mat& frame,
                          const IrisResult& iris_result,
                          const LensConfig& config) {
#ifdef IRIS_SDK_HAS_OPENCV
    if (!impl_ || !impl_->initialized || !impl_->texture_loaded) {
        return false;
    }

    // 프레임 유효성 검사
    if (frame.empty() || frame.depth() != CV_8U ||
        (frame.channels() != 3 && frame.channels() != 4)) {
        return false;
    }

    if (!iris_result.detected) {
        return false;
    }

    // frame_width/height 유효성 검사
    if (iris_result.frame_width <= 0 || iris_result.frame_height <= 0) {
        return false;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    bool any_rendered = false;

    // 왼쪽 눈 렌더링
    if (config.apply_left && iris_result.left_detected) {
        IrisRegion left_region = calculateIrisRegion(
            iris_result.left_iris,
            iris_result.frame_width,
            iris_result.frame_height
        );

        if (impl_->renderEye(frame, left_region, config)) {
            any_rendered = true;
        }
    }

    // 오른쪽 눈 렌더링
    if (config.apply_right && iris_result.right_detected) {
        IrisRegion right_region = calculateIrisRegion(
            iris_result.right_iris,
            iris_result.frame_width,
            iris_result.frame_height
        );

        if (impl_->renderEye(frame, right_region, config)) {
            any_rendered = true;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    impl_->last_render_time_ms = std::chrono::duration<double, std::milli>(
        end_time - start_time
    ).count();

    return any_rendered;
#else
    (void)frame;
    (void)iris_result;
    (void)config;
    return false;
#endif
}

bool LensRenderer::renderLeftEye(cv::Mat& frame,
                                 const IrisResult& iris_result,
                                 const LensConfig& config) {
#ifdef IRIS_SDK_HAS_OPENCV
    if (!impl_ || !impl_->initialized || !impl_->texture_loaded) {
        return false;
    }

    // 프레임 유효성 검사
    if (frame.empty() || frame.depth() != CV_8U ||
        (frame.channels() != 3 && frame.channels() != 4)) {
        return false;
    }

    if (!iris_result.detected || !iris_result.left_detected) {
        return false;
    }

    // frame_width/height 유효성 검사
    if (iris_result.frame_width <= 0 || iris_result.frame_height <= 0) {
        return false;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    IrisRegion left_region = calculateIrisRegion(
        iris_result.left_iris,
        iris_result.frame_width,
        iris_result.frame_height
    );

    bool result = impl_->renderEye(frame, left_region, config);

    auto end_time = std::chrono::high_resolution_clock::now();
    impl_->last_render_time_ms = std::chrono::duration<double, std::milli>(
        end_time - start_time
    ).count();

    return result;
#else
    (void)frame;
    (void)iris_result;
    (void)config;
    return false;
#endif
}

bool LensRenderer::renderRightEye(cv::Mat& frame,
                                  const IrisResult& iris_result,
                                  const LensConfig& config) {
#ifdef IRIS_SDK_HAS_OPENCV
    if (!impl_ || !impl_->initialized || !impl_->texture_loaded) {
        return false;
    }

    // 프레임 유효성 검사
    if (frame.empty() || frame.depth() != CV_8U ||
        (frame.channels() != 3 && frame.channels() != 4)) {
        return false;
    }

    if (!iris_result.detected || !iris_result.right_detected) {
        return false;
    }

    // frame_width/height 유효성 검사
    if (iris_result.frame_width <= 0 || iris_result.frame_height <= 0) {
        return false;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    IrisRegion right_region = calculateIrisRegion(
        iris_result.right_iris,
        iris_result.frame_width,
        iris_result.frame_height
    );

    bool result = impl_->renderEye(frame, right_region, config);

    auto end_time = std::chrono::high_resolution_clock::now();
    impl_->last_render_time_ms = std::chrono::duration<double, std::milli>(
        end_time - start_time
    ).count();

    return result;
#else
    (void)frame;
    (void)iris_result;
    (void)config;
    return false;
#endif
}

double LensRenderer::getLastRenderTimeMs() const noexcept {
    return impl_ ? impl_->last_render_time_ms : 0.0;
}

} // namespace iris_sdk
