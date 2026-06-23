/**
 * @file cpu_beauty_backend.cpp
 * @brief CPUBeautyBackend 구현
 */

#include "iris_sdk/cpu_beauty_backend.h"
#include "iris_sdk/beauty_roi_manager.h"
// P8-W2-C: fast_guided_filter.h include 제거 — V2(Guided) 효과 전부 삭제로 orphan.
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>

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
        case IRIS_FORMAT_NV21: {
            // NV21 (YUV420sp): Y plane + interleaved VU plane
            cv::Mat nv21(height + height / 2, width, CV_8UC1, const_cast<uint8_t*>(data));
            cv::cvtColor(nv21, result, cv::COLOR_YUV2BGR_NV21);
            break;
        }
        case IRIS_FORMAT_NV12: {
            // NV12 (YUV420sp): Y plane + interleaved UV plane
            cv::Mat nv12(height + height / 2, width, CV_8UC1, const_cast<uint8_t*>(data));
            cv::cvtColor(nv12, result, cv::COLOR_YUV2BGR_NV12);
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
        case IRIS_FORMAT_NV21: {
            // BGR -> I420 변환 후 VU 인터리브로 재배열
            cv::Mat yuv_i420;
            cv::cvtColor(bgr, yuv_i420, cv::COLOR_BGR2YUV_I420);

            // Y 평면 복사
            int y_size = width * height;
            int uv_size = y_size / 4;
            std::memcpy(data, yuv_i420.data, y_size);

            // U, V 평면을 VU 인터리브로 재배열 (NV21: VUVU...)
            const uint8_t* u_plane = yuv_i420.data + y_size;
            const uint8_t* v_plane = u_plane + uv_size;
            uint8_t* vu_plane = data + y_size;

            for (int i = 0; i < uv_size; i++) {
                vu_plane[2*i] = v_plane[i];     // V
                vu_plane[2*i + 1] = u_plane[i]; // U
            }
            break;
        }
        case IRIS_FORMAT_NV12: {
            // BGR -> I420 변환 후 UV 인터리브로 재배열
            cv::Mat yuv_i420;
            cv::cvtColor(bgr, yuv_i420, cv::COLOR_BGR2YUV_I420);

            // Y 평면 복사
            int y_size = width * height;
            int uv_size = y_size / 4;
            std::memcpy(data, yuv_i420.data, y_size);

            // U, V 평면을 UV 인터리브로 재배열 (NV12: UVUV...)
            const uint8_t* u_plane = yuv_i420.data + y_size;
            const uint8_t* v_plane = u_plane + uv_size;
            uint8_t* uv_plane = data + y_size;

            for (int i = 0; i < uv_size; i++) {
                uv_plane[2*i] = u_plane[i];     // U
                uv_plane[2*i + 1] = v_plane[i]; // V
            }
            break;
        }
        default:
            break;
    }
}

//=============================================================================
// 필터 효과 함수
//=============================================================================
//
// P8-W2-C: 곁가지 CPU 색보정/스무딩 EFFECT 제거.
//   - applySkinSmoothing(Bilateral, V1) / applySoftFocus(V1)
//   - applyWhitening(2 오버로드) / applyColorBalance (색보정)
//   - V2(Guided) 계열 전부 (applySkinSmoothingV2 / applySoftFocusV2 /
//     applyBrightnessV2 / applyWrinkleRemoval / overlayBlend / detectSkinTone /
//     createWrinkleRegionMasks) — 호출처 없는 dead code
// 생존: applyBrightness(골든/apply 경로가 사용하는 유일한 CPU beauty 효과).
// ① skin smoothing(GPU use_skin_mask) / ② 형태워프는 본 정리와 무관(미접촉).

void CPUBeautyBackend::applyBrightness(cv::Mat& frame, float brightness) {
    if (std::abs(brightness - 1.0f) < 0.01f) return;

    // 밝기 조정 (1.0 = 원본, >1 = 밝게, <1 = 어둡게)
    frame.convertTo(frame, -1, brightness, 0);
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
        return IRIS_SDK_NOT_INITIALIZED;
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

    // P8-W2-C: 곁가지 색보정/스무딩 제거 후 CPU 전체프레임 경로의 생존 효과는
    // brightness 뿐이다. (skin smoothing은 GPU use_skin_mask 채널 책임)
    applyBrightness(frame, config.brightness);

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

    // 2. ROI 영역에 필터 적용
    //    P8-W2-C: 곁가지 색보정/스무딩(applySkinSmoothing/applySoftFocus/
    //    applyWhitening/applyColorBalance) 제거 → 생존 효과는 brightness 뿐.
    //    보호 마스크 준비 블록도 소비처(스무딩/화이트닝)가 사라져 함께 제거.
    //    (ROI 추출·feather·applyROIRegion 골격은 유지)
    if (std::abs(config.brightness - 1.0f) > 0.01f) {
        applyBrightness(roi_region, config.brightness);
    }

    // 3. 페더링 마스크 생성
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

    // 4. 원본에 합성
    BeautyROIManager::applyROIRegion(frame, roi_region, actual_rect, feather_mask);

    return IRIS_SDK_OK;
}

} // namespace iris_sdk
