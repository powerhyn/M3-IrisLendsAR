/**
 * @file cpu_beauty_backend.cpp
 * @brief CPUBeautyBackend 구현
 */

#include "iris_sdk/cpu_beauty_backend.h"
#include "iris_sdk/beauty_roi_manager.h"
#include "iris_sdk/fast_guided_filter.h"
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
// V2 필터 효과 함수 (Guided Filter 기반)
//=============================================================================

void CPUBeautyBackend::applySkinSmoothingV2(cv::Mat& frame, float strength,
                                             const cv::Mat& protection_mask) {
    if (strength <= 0.0f) return;

    // FastGuidedFilter 파라미터
    // eps: 스무딩 정도 (높을수록 더 스무딩)
    // radius: 필터 반경
    double eps = 0.01 + strength * 0.15;  // 0.01 ~ 0.16
    int radius = static_cast<int>(4 + strength * 8);  // 4 ~ 12

    // 서브샘플링 비율 (큰 이미지에서 성능 최적화)
    int subsample = FastGuidedFilter::getRecommendedSubsampleRatio(
        frame.cols, frame.rows, 15.0);

    // Guided Filter 적용
    cv::Mat smoothed;
    FastGuidedFilter::filter(frame, smoothed, radius, eps, subsample);

    // 보호 마스크 적용
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

void CPUBeautyBackend::applySoftFocusV2(cv::Mat& frame, float strength) {
    if (strength <= 0.0f) return;

    // 1단계: Guided Filter로 에지 보존 스무딩
    double eps = FastGuidedFilter::getRecommendedEps(2);  // medium smoothing
    int radius = static_cast<int>(6 + strength * 10);
    int subsample = FastGuidedFilter::getRecommendedSubsampleRatio(
        frame.cols, frame.rows, 15.0);

    cv::Mat guided_smooth;
    FastGuidedFilter::filter(frame, guided_smooth, radius, eps, subsample);

    // 2단계: 오버레이 블렌딩 (하이라이트 강조)
    cv::Mat overlay_result;
    float overlay_opacity = strength * 0.3f;  // 은은한 오버레이 효과
    overlayBlend(frame, guided_smooth, overlay_result, overlay_opacity);

    // 3단계: 가우시안 글로우 추가
    int blur_size = static_cast<int>(11 + strength * 20) | 1;  // 홀수 보장
    cv::Mat glow;
    cv::GaussianBlur(overlay_result, glow, cv::Size(blur_size, blur_size), 0);

    // 글로우와 블렌딩 (낮은 불투명도로 부드러운 효과)
    float glow_opacity = strength * 0.25f;
    cv::addWeighted(overlay_result, 1.0 - glow_opacity, glow, glow_opacity, 0, frame);
}

void CPUBeautyBackend::applyBrightnessV2(cv::Mat& frame, float brightness) {
    if (std::abs(brightness - 1.0f) < 0.01f) return;

    // LAB 색상 공간에서 L 채널 조정 (하이라이트 보호)
    cv::Mat lab;
    cv::cvtColor(frame, lab, cv::COLOR_BGR2Lab);

    std::vector<cv::Mat> lab_channels;
    cv::split(lab, lab_channels);

    cv::Mat L_float;
    lab_channels[0].convertTo(L_float, CV_32F);

    if (brightness > 1.0f) {
        // 밝게: 비선형 증가 (하이라이트 보호)
        // L = L + (255 - L) * (factor - 1) * 0.5
        // 이렇게 하면 밝은 영역(255에 가까운)은 덜 변하고,
        // 어두운 영역은 더 많이 밝아짐
        float factor = brightness;
        cv::Mat headroom = 255.0f - L_float;  // 255까지 남은 여유
        cv::Mat adjustment = headroom * ((factor - 1.0f) * 0.5f);
        L_float = L_float + adjustment;
    } else {
        // 어둡게: 선형 감소 (간단한 스케일링)
        L_float = L_float * brightness;
    }

    // 범위 클리핑 (0~255)
    cv::min(L_float, 255.0f, L_float);
    cv::max(L_float, 0.0f, L_float);

    L_float.convertTo(lab_channels[0], CV_8U);
    cv::merge(lab_channels, lab);
    cv::cvtColor(lab, frame, cv::COLOR_Lab2BGR);
}

void CPUBeautyBackend::applyWrinkleRemoval(cv::Mat& frame, float strength,
                                            const IrisLandmark* face_mesh,
                                            int landmark_count,
                                            int offset_x, int offset_y) {
    if (strength <= 0.0f || face_mesh == nullptr || landmark_count < 468) return;

    // 주름 영역 마스크 생성
    WrinkleRegions regions = createWrinkleRegionMasks(
        face_mesh, landmark_count,
        frame.cols, frame.rows,
        offset_x, offset_y);

    if (regions.combined.empty()) return;

    // 강한 스무딩 적용 (주름 제거용)
    double eps = 0.04 + strength * 0.12;  // 0.04 ~ 0.16 (더 강한 스무딩)
    int radius = static_cast<int>(8 + strength * 12);  // 8 ~ 20
    int subsample = FastGuidedFilter::getRecommendedSubsampleRatio(
        frame.cols, frame.rows, 15.0);

    cv::Mat heavily_smoothed;
    FastGuidedFilter::filter(frame, heavily_smoothed, radius, eps, subsample);

    // 주름 영역 마스크 기반 선택적 블렌딩
    cv::Mat mask_3ch;
    cv::cvtColor(regions.combined, mask_3ch, cv::COLOR_GRAY2BGR);
    mask_3ch.convertTo(mask_3ch, CV_32F, 1.0 / 255.0);

    // 강도에 따른 블렌딩 조정
    mask_3ch = mask_3ch * strength;

    cv::Mat frame_f, smoothed_f;
    frame.convertTo(frame_f, CV_32F);
    heavily_smoothed.convertTo(smoothed_f, CV_32F);

    // 마스크 영역은 스무딩, 나머지는 원본
    cv::Mat result = smoothed_f.mul(mask_3ch) +
                     frame_f.mul(cv::Scalar(1.0, 1.0, 1.0) - mask_3ch);
    result.convertTo(frame, CV_8U);
}

//=============================================================================
// 헬퍼 함수
//=============================================================================

cv::Mat CPUBeautyBackend::detectSkinTone(const cv::Mat& A_channel,
                                          const cv::Mat& B_channel) {
    // LAB 색상 공간에서 피부톤 감지
    // 일반적인 피부톤 범위:
    // A 채널: 130~175 (약간 붉은 기미)
    // B 채널: 130~200 (약간 노란 기미)

    cv::Mat A_float, B_float;
    A_channel.convertTo(A_float, CV_32F);
    B_channel.convertTo(B_float, CV_32F);

    // A 채널 범위 체크 (130 ~ 175)
    cv::Mat A_lower, A_upper, A_mask;
    cv::compare(A_float, 130.0, A_lower, cv::CMP_GE);
    cv::compare(A_float, 175.0, A_upper, cv::CMP_LE);
    cv::bitwise_and(A_lower, A_upper, A_mask);

    // B 채널 범위 체크 (130 ~ 200)
    cv::Mat B_lower, B_upper, B_mask;
    cv::compare(B_float, 130.0, B_lower, cv::CMP_GE);
    cv::compare(B_float, 200.0, B_upper, cv::CMP_LE);
    cv::bitwise_and(B_lower, B_upper, B_mask);

    // 두 마스크 결합
    cv::Mat skin_mask;
    cv::bitwise_and(A_mask, B_mask, skin_mask);

    // 노이즈 제거를 위한 모폴로지 연산
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(skin_mask, skin_mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(skin_mask, skin_mask, cv::MORPH_CLOSE, kernel);

    return skin_mask;
}

void CPUBeautyBackend::overlayBlend(const cv::Mat& base, const cv::Mat& blend,
                                     cv::Mat& result, float opacity) {
    // 포토샵 스타일 오버레이 블렌드:
    // if base < 128: 2 * base * blend / 255
    // else: 255 - 2 * (255 - base) * (255 - blend) / 255

    cv::Mat base_f, blend_f;
    base.convertTo(base_f, CV_32F, 1.0 / 255.0);
    blend.convertTo(blend_f, CV_32F, 1.0 / 255.0);

    cv::Mat overlay_result = cv::Mat::zeros(base_f.size(), base_f.type());

    // 각 픽셀에 대해 오버레이 계산
    for (int y = 0; y < base_f.rows; ++y) {
        const float* base_row = base_f.ptr<float>(y);
        const float* blend_row = blend_f.ptr<float>(y);
        float* out_row = overlay_result.ptr<float>(y);

        for (int x = 0; x < base_f.cols * base_f.channels(); ++x) {
            float b = base_row[x];
            float l = blend_row[x];

            // 오버레이 공식
            float overlay_val;
            if (b < 0.5f) {
                overlay_val = 2.0f * b * l;
            } else {
                overlay_val = 1.0f - 2.0f * (1.0f - b) * (1.0f - l);
            }

            out_row[x] = overlay_val;
        }
    }

    // 원본과 오버레이 결과 블렌딩
    cv::Mat blended;
    cv::addWeighted(base_f, 1.0 - opacity, overlay_result, opacity, 0, blended);

    // 8비트로 변환
    blended.convertTo(result, CV_8U, 255.0);
}

WrinkleRegions CPUBeautyBackend::createWrinkleRegionMasks(
    const IrisLandmark* face_mesh, int landmark_count,
    int frame_width, int frame_height,
    int offset_x, int offset_y) {

    WrinkleRegions regions;

    if (face_mesh == nullptr || landmark_count < 468) {
        return regions;
    }

    // 각 마스크 초기화
    regions.forehead_mask = cv::Mat::zeros(frame_height, frame_width, CV_8UC1);
    regions.crow_feet_mask = cv::Mat::zeros(frame_height, frame_width, CV_8UC1);
    regions.frown_lines_mask = cv::Mat::zeros(frame_height, frame_width, CV_8UC1);
    regions.combined = cv::Mat::zeros(frame_height, frame_width, CV_8UC1);

    // 좌표 변환 람다 함수
    auto toPixel = [&](int idx) -> cv::Point {
        if (idx >= landmark_count) return cv::Point(-1, -1);
        int px = static_cast<int>(face_mesh[idx].x * frame_width) + offset_x;
        int py = static_cast<int>(face_mesh[idx].y * frame_height) + offset_y;
        px = std::clamp(px, 0, frame_width - 1);
        py = std::clamp(py, 0, frame_height - 1);
        return cv::Point(px, py);
    };

    // 1. 이마 영역 마스크 (MediaPipe 랜드마크 인덱스)
    // 이마 상단: 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288
    // 이마 하단: 54, 103, 67, 109, 10, 338, 297, 332
    {
        std::vector<cv::Point> forehead_pts;
        // 이마 위쪽 라인
        int forehead_top[] = {10, 338, 297, 332, 284, 251, 389};
        for (int idx : forehead_top) {
            cv::Point pt = toPixel(idx);
            if (pt.x >= 0) forehead_pts.push_back(pt);
        }

        // 이마 아래쪽 라인 (역순)
        int forehead_bottom[] = {389, 251, 284, 332, 297, 338, 10};
        for (int idx : forehead_bottom) {
            cv::Point pt = toPixel(idx);
            if (pt.x >= 0) {
                // 약간 아래로 이동
                pt.y += static_cast<int>(frame_height * 0.03);
                forehead_pts.push_back(pt);
            }
        }

        if (forehead_pts.size() >= 3) {
            std::vector<std::vector<cv::Point>> contours = {forehead_pts};
            cv::fillPoly(regions.forehead_mask, contours, cv::Scalar(255));
        }
    }

    // 2. 눈가 주름 (까마귀 발) 영역 마스크
    // 왼쪽 눈 바깥: 130, 247, 30, 29, 27, 28, 56, 190
    // 오른쪽 눈 바깥: 359, 467, 260, 259, 257, 258, 286, 414
    {
        // 왼쪽 눈가
        std::vector<cv::Point> left_crow_pts;
        int left_crow[] = {130, 247, 30, 29, 27, 28, 56, 190};
        for (int idx : left_crow) {
            cv::Point pt = toPixel(idx);
            if (pt.x >= 0) left_crow_pts.push_back(pt);
        }

        if (left_crow_pts.size() >= 3) {
            std::vector<std::vector<cv::Point>> contours = {left_crow_pts};
            cv::fillPoly(regions.crow_feet_mask, contours, cv::Scalar(255));
        }

        // 오른쪽 눈가
        std::vector<cv::Point> right_crow_pts;
        int right_crow[] = {359, 467, 260, 259, 257, 258, 286, 414};
        for (int idx : right_crow) {
            cv::Point pt = toPixel(idx);
            if (pt.x >= 0) right_crow_pts.push_back(pt);
        }

        if (right_crow_pts.size() >= 3) {
            std::vector<std::vector<cv::Point>> contours = {right_crow_pts};
            cv::fillPoly(regions.crow_feet_mask, contours, cv::Scalar(255));
        }
    }

    // 3. 미간 주름 영역 마스크
    // 미간: 9, 8, 168, 6, 197, 195, 5, 4
    {
        std::vector<cv::Point> frown_pts;
        int frown[] = {9, 8, 168, 6, 197, 195, 5, 4};
        for (int idx : frown) {
            cv::Point pt = toPixel(idx);
            if (pt.x >= 0) frown_pts.push_back(pt);
        }

        if (frown_pts.size() >= 3) {
            std::vector<std::vector<cv::Point>> contours = {frown_pts};
            cv::fillPoly(regions.frown_lines_mask, contours, cv::Scalar(255));
        }
    }

    // 모든 마스크 결합
    cv::bitwise_or(regions.forehead_mask, regions.crow_feet_mask, regions.combined);
    cv::bitwise_or(regions.combined, regions.frown_lines_mask, regions.combined);

    // 부드러운 경계를 위한 블러
    cv::GaussianBlur(regions.combined, regions.combined, cv::Size(15, 15), 0);

    return regions;
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
