/**
 * @file cpu_beauty_backend.cpp
 * @brief CPUBeautyBackend 구현
 */

#include "iris_sdk/cpu_beauty_backend.h"
#include "iris_sdk/beauty_roi_manager.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>

namespace iris_sdk {

//=============================================================================
// 생성자/소멸자
//=============================================================================

CPUBeautyBackend::CPUBeautyBackend() = default;

CPUBeautyBackend::~CPUBeautyBackend() {
    release();
}

//=============================================================================
// IBeautyBackend 라이프사이클
//=============================================================================

bool CPUBeautyBackend::initialize(IRenderContext* /*render_context*/) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (initialized_) return true;

    // CPU 백엔드는 특별한 초기화 필요 없음
    initialized_ = true;
    return true;
}

void CPUBeautyBackend::release() {
    std::lock_guard<std::mutex> lock(mutex_);

    work_buffer_.release();
    smooth_buffer_.release();
    roi_buffer_.release();
    initialized_ = false;
}

bool CPUBeautyBackend::isInitialized() const {
    return initialized_;
}

//=============================================================================
// 포맷 변환 유틸리티
//=============================================================================

cv::Mat CPUBeautyBackend::convertToBGR(const uint8_t* data, int width, int height,
                                        IrisFrameFormat format) {
    cv::Mat result;

    switch (format) {
        case IRIS_FORMAT_RGBA: {
            cv::Mat rgba(height, width, CV_8UC4, const_cast<uint8_t*>(data));
            cv::cvtColor(rgba, result, cv::COLOR_RGBA2BGR);
            break;
        }
        case IRIS_FORMAT_BGRA: {
            cv::Mat bgra(height, width, CV_8UC4, const_cast<uint8_t*>(data));
            cv::cvtColor(bgra, result, cv::COLOR_BGRA2BGR);
            break;
        }
        case IRIS_FORMAT_RGB: {
            cv::Mat rgb(height, width, CV_8UC3, const_cast<uint8_t*>(data));
            cv::cvtColor(rgb, result, cv::COLOR_RGB2BGR);
            break;
        }
        case IRIS_FORMAT_BGR: {
            result = cv::Mat(height, width, CV_8UC3, const_cast<uint8_t*>(data)).clone();
            break;
        }
        default:
            // 지원하지 않는 포맷
            break;
    }

    return result;
}

void CPUBeautyBackend::convertFromBGR(const cv::Mat& bgr, uint8_t* data,
                                       int width, int height, IrisFrameFormat format) {
    switch (format) {
        case IRIS_FORMAT_RGBA: {
            cv::Mat rgba(height, width, CV_8UC4, data);
            cv::cvtColor(bgr, rgba, cv::COLOR_BGR2RGBA);
            break;
        }
        case IRIS_FORMAT_BGRA: {
            cv::Mat bgra(height, width, CV_8UC4, data);
            cv::cvtColor(bgr, bgra, cv::COLOR_BGR2BGRA);
            break;
        }
        case IRIS_FORMAT_RGB: {
            cv::Mat rgb(height, width, CV_8UC3, data);
            cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
            break;
        }
        case IRIS_FORMAT_BGR: {
            cv::Mat dst(height, width, CV_8UC3, data);
            bgr.copyTo(dst);
            break;
        }
        default:
            break;
    }
}

//=============================================================================
// 필터 효과 함수
//=============================================================================

void CPUBeautyBackend::applySkinSmoothing(cv::Mat& frame, float strength) {
    if (strength <= 0.0f) return;

    // Bilateral Filter 파라미터 (강도에 따라 조정)
    int d = static_cast<int>(5 + strength * 10);
    double sigma_color = 20.0 + strength * 50.0;
    double sigma_space = 20.0 + strength * 50.0;

    // Bilateral Filter는 계산 비용이 높으므로 다운스케일 적용 가능
    cv::bilateralFilter(frame.clone(), frame, d, sigma_color, sigma_space);
}

void CPUBeautyBackend::applySkinSmoothing(cv::Mat& frame, float strength,
                                           const cv::Mat& protection_mask) {
    if (strength <= 0.0f) return;

    // 기존 Bilateral Filter 적용
    cv::Mat smoothed;
    int d = static_cast<int>(5 + strength * 10);
    double sigma_color = 20.0 + strength * 50.0;
    double sigma_space = 20.0 + strength * 50.0;
    cv::bilateralFilter(frame, smoothed, d, sigma_color, sigma_space);

    // 보호 마스크 적용 (눈/입술 보호)
    if (!protection_mask.empty() &&
        protection_mask.cols == frame.cols &&
        protection_mask.rows == frame.rows) {
        cv::Mat mask_inv;
        cv::bitwise_not(protection_mask, mask_inv);

        cv::Mat mask_3ch;
        cv::cvtColor(mask_inv, mask_3ch, cv::COLOR_GRAY2BGR);
        mask_3ch.convertTo(mask_3ch, CV_32F, 1.0 / 255.0);

        cv::Mat frame_f, smoothed_f;
        frame.convertTo(frame_f, CV_32F);
        smoothed.convertTo(smoothed_f, CV_32F);

        // 보호 영역은 원본, 나머지는 스무딩
        cv::Mat result = smoothed_f.mul(mask_3ch) +
                         frame_f.mul(cv::Scalar(1.0, 1.0, 1.0) - mask_3ch);
        result.convertTo(frame, CV_8U);
    } else {
        smoothed.copyTo(frame);
    }
}

void CPUBeautyBackend::applySoftFocus(cv::Mat& frame, float strength) {
    if (strength <= 0.0f) return;

    // Gaussian Blur로 소프트 이미지 생성
    cv::Mat soft;
    int blur_size = static_cast<int>(5 + strength * 20) | 1;  // 홀수로 보장
    cv::GaussianBlur(frame, soft, cv::Size(blur_size, blur_size), 0);

    // 원본과 블렌딩
    cv::addWeighted(frame, 1.0 - strength * 0.5, soft, strength * 0.5, 0, frame);
}

void CPUBeautyBackend::applyBrightness(cv::Mat& frame, float brightness) {
    if (std::abs(brightness - 1.0f) < 0.01f) return;

    // 밝기 조정 (1.0 = 원본, >1 = 밝게, <1 = 어둡게)
    frame.convertTo(frame, -1, brightness, 0);
}

void CPUBeautyBackend::applyWhitening(cv::Mat& frame, float strength) {
    if (strength <= 0.0f) return;

    // LAB 색공간으로 변환
    cv::Mat lab;
    cv::cvtColor(frame, lab, cv::COLOR_BGR2Lab);

    // L 채널 (밝기) 증가
    std::vector<cv::Mat> lab_channels;
    cv::split(lab, lab_channels);

    // L 채널에 증가분 적용 (0~255 범위 유지)
    float increment = strength * 20.0f;
    lab_channels[0].convertTo(lab_channels[0], -1, 1.0, increment);

    cv::merge(lab_channels, lab);
    cv::cvtColor(lab, frame, cv::COLOR_Lab2BGR);
}

void CPUBeautyBackend::applyWhitening(cv::Mat& frame, float strength,
                                       const cv::Mat& protection_mask) {
    if (strength <= 0.0f) return;

    // 화이트닝 적용된 이미지 생성
    cv::Mat whitened = frame.clone();

    cv::Mat lab;
    cv::cvtColor(whitened, lab, cv::COLOR_BGR2Lab);

    std::vector<cv::Mat> lab_channels;
    cv::split(lab, lab_channels);

    float increment = strength * 20.0f;
    lab_channels[0].convertTo(lab_channels[0], -1, 1.0, increment);

    cv::merge(lab_channels, lab);
    cv::cvtColor(lab, whitened, cv::COLOR_Lab2BGR);

    // 보호 마스크 적용
    if (!protection_mask.empty() &&
        protection_mask.cols == frame.cols &&
        protection_mask.rows == frame.rows) {
        cv::Mat mask_inv;
        cv::bitwise_not(protection_mask, mask_inv);

        cv::Mat mask_3ch;
        cv::cvtColor(mask_inv, mask_3ch, cv::COLOR_GRAY2BGR);
        mask_3ch.convertTo(mask_3ch, CV_32F, 1.0 / 255.0);

        cv::Mat frame_f, whitened_f;
        frame.convertTo(frame_f, CV_32F);
        whitened.convertTo(whitened_f, CV_32F);

        cv::Mat result = whitened_f.mul(mask_3ch) +
                         frame_f.mul(cv::Scalar(1.0, 1.0, 1.0) - mask_3ch);
        result.convertTo(frame, CV_8U);
    } else {
        whitened.copyTo(frame);
    }
}

void CPUBeautyBackend::applyColorBalance(cv::Mat& frame, float balance) {
    if (std::abs(balance) < 0.01f) return;

    // 색상 밸런스 조정 (-1 = 쿨톤, +1 = 웜톤)
    std::vector<cv::Mat> channels;
    cv::split(frame, channels);

    if (balance > 0) {
        // 웜톤: R 증가, B 감소
        channels[2].convertTo(channels[2], -1, 1.0, balance * 15);
        channels[0].convertTo(channels[0], -1, 1.0, -balance * 10);
    } else {
        // 쿨톤: B 증가, R 감소
        channels[0].convertTo(channels[0], -1, 1.0, -balance * 15);
        channels[2].convertTo(channels[2], -1, 1.0, balance * 10);
    }

    cv::merge(channels, frame);
}

//=============================================================================
// 필터 적용
//=============================================================================

IrisSdkError CPUBeautyBackend::apply(
    uint8_t* frame_data,
    int width, int height,
    IrisFrameFormat format,
    const BeautyFilterConfigV2& config,
    const BeautyROI* roi) {

    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_) {
        return IRIS_SDK_ERROR_NOT_INITIALIZED;
    }

    if (!frame_data || width <= 0 || height <= 0) {
        return IRIS_SDK_INVALID_PARAM;
    }

    if (!config.enabled) {
        return IRIS_SDK_OK;  // 비활성화 상태
    }

    // BGR로 변환
    cv::Mat frame = convertToBGR(frame_data, width, height, format);
    if (frame.empty()) {
        return IRIS_SDK_INVALID_FORMAT;
    }

    // ROI 기반 또는 전체 프레임 처리
    IrisSdkError result;
    if (roi && roi->isValid() && config.roiOnly) {
        result = applyWithROI(frame, config, *roi);
    } else {
        result = applyFullFrame(frame, config);
    }

    if (result != IRIS_SDK_OK) {
        return result;
    }

    // 원본 포맷으로 변환
    convertFromBGR(frame, frame_data, width, height, format);

    return IRIS_SDK_OK;
}

IrisSdkError CPUBeautyBackend::applyFullFrame(
    cv::Mat& frame,
    const BeautyFilterConfigV2& config) {

    // 각 효과 순차 적용
    applySkinSmoothing(frame, config.smoothing * config.intensity);
    applySoftFocus(frame, config.softFocus * config.intensity);
    applyBrightness(frame, config.brightness);
    applyWhitening(frame, config.whitening * config.intensity);
    applyColorBalance(frame, config.colorBalance);

    return IRIS_SDK_OK;
}

IrisSdkError CPUBeautyBackend::applyWithROI(
    cv::Mat& frame,
    const BeautyFilterConfigV2& config,
    const BeautyROI& roi) {

    // 1. ROI 영역 추출 (패딩 포함)
    const int PADDING = 20;  // 블러 경계 아티팩트 방지
    cv::Mat roi_region;
    cv::Rect actual_rect;

    if (!BeautyROIManager::extractROIRegion(frame, roi, roi_region, actual_rect, PADDING)) {
        // 폴백: 기본 방식으로 처리
        int roi_x = static_cast<int>(roi.face_rect.x * frame.cols);
        int roi_y = static_cast<int>(roi.face_rect.y * frame.rows);
        int roi_w = static_cast<int>(roi.face_rect.width * frame.cols);
        int roi_h = static_cast<int>(roi.face_rect.height * frame.rows);

        roi_x = std::clamp(roi_x, 0, frame.cols - 1);
        roi_y = std::clamp(roi_y, 0, frame.rows - 1);
        roi_w = std::clamp(roi_w, 1, frame.cols - roi_x);
        roi_h = std::clamp(roi_h, 1, frame.rows - roi_y);

        actual_rect = cv::Rect(roi_x, roi_y, roi_w, roi_h);
        roi_region = frame(actual_rect).clone();
    }

    // 2. 보호 마스크 준비 (ROI 영역 크기로 리사이즈)
    cv::Mat protection_mask;
    if (!roi.eye_protect_mask.empty() || !roi.lip_protect_mask.empty()) {
        // combined_mask에서 보호 영역 추출 (반전된 값)
        // combined_mask = skin * (1 - eye) * (1 - eyebrow) * (1 - lip)
        // 보호 마스크 = eye | eyebrow | lip
        size_t mask_size = static_cast<size_t>(roi.mask_width) * roi.mask_height;
        std::vector<uint8_t> protect_combined(mask_size, 0);

        for (size_t i = 0; i < mask_size; ++i) {
            uint8_t eye_val = (i < roi.eye_protect_mask.size()) ? roi.eye_protect_mask[i] : 0;
            uint8_t eyebrow_val = (i < roi.eyebrow_protect_mask.size()) ? roi.eyebrow_protect_mask[i] : 0;
            uint8_t lip_val = (i < roi.lip_protect_mask.size()) ? roi.lip_protect_mask[i] : 0;
            protect_combined[i] = std::max({eye_val, eyebrow_val, lip_val});
        }

        cv::Mat protect_mat(roi.mask_height, roi.mask_width, CV_8UC1, protect_combined.data());
        cv::resize(protect_mat, protection_mask,
                   cv::Size(actual_rect.width, actual_rect.height));
    }

    // 3. ROI 영역에 필터 적용 (보호 마스크 사용)
    if (config.smoothing > 0.01f) {
        if (!protection_mask.empty()) {
            applySkinSmoothing(roi_region, config.smoothing * config.intensity, protection_mask);
        } else {
            applySkinSmoothing(roi_region, config.smoothing * config.intensity);
        }
    }

    if (config.softFocus > 0.01f) {
        applySoftFocus(roi_region, config.softFocus * config.intensity);
    }

    if (std::abs(config.brightness - 1.0f) > 0.01f) {
        applyBrightness(roi_region, config.brightness);
    }

    if (config.whitening > 0.01f) {
        if (!protection_mask.empty()) {
            applyWhitening(roi_region, config.whitening * config.intensity, protection_mask);
        } else {
            applyWhitening(roi_region, config.whitening * config.intensity);
        }
    }

    if (std::abs(config.colorBalance) > 0.01f) {
        applyColorBalance(roi_region, config.colorBalance);
    }

    // 4. 페더링 마스크 생성
    cv::Mat feather_mask;
    if (!roi.combined_mask.empty()) {
        // combined_mask에서 보호 영역이 이미 제거된 마스크 사용
        feather_mask = BeautyROIManager::createFeatherMask(
            roi, 15,
            roi.combined_mask,
            std::vector<uint8_t>()  // 이미 combined에 포함됨
        );

        // 실제 ROI 크기로 리사이즈
        if (!feather_mask.empty() &&
            (feather_mask.cols != actual_rect.width ||
             feather_mask.rows != actual_rect.height)) {
            cv::resize(feather_mask, feather_mask,
                       cv::Size(actual_rect.width, actual_rect.height));
        }
    }

    // 5. 원본에 합성
    BeautyROIManager::applyROIRegion(frame, roi_region, actual_rect, feather_mask);

    return IRIS_SDK_OK;
}

} // namespace iris_sdk
