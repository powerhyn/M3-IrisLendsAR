/**
 * @file beauty_roi_manager.cpp
 * @brief BeautyROIManager 구현
 */

#include "iris_sdk/beauty_roi_manager.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>

namespace iris_sdk {

//=============================================================================
// Face Mesh 랜드마크 인덱스 테이블
// MediaPipe Face Mesh 기준 (478 랜드마크)
//=============================================================================

// 얼굴 윤곽 인덱스 (36개) - 시계 방향으로 정렬
const int BeautyROIManager::FACE_OVAL_INDICES[36] = {
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109
};

// 왼쪽 눈 인덱스 (16개)
const int BeautyROIManager::LEFT_EYE_INDICES[16] = {
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    173, 157, 158, 159, 160, 161, 246
};

// 오른쪽 눈 인덱스 (16개)
const int BeautyROIManager::RIGHT_EYE_INDICES[16] = {
    362, 382, 381, 380, 374, 373, 390, 249, 263,
    466, 388, 387, 386, 385, 384, 398
};

// 입술 인덱스 (22개)
const int BeautyROIManager::LIPS_INDICES[22] = {
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78
};

// 왼쪽 눈썹 인덱스 (8개)
const int BeautyROIManager::LEFT_EYEBROW_INDICES[8] = {
    70, 63, 105, 66, 107, 55, 65, 52
};

// 오른쪽 눈썹 인덱스 (8개)
const int BeautyROIManager::RIGHT_EYEBROW_INDICES[8] = {
    300, 293, 334, 296, 336, 285, 295, 282
};

// 외곽 입술 인덱스 (20개) - MediaPipe Face Mesh
const int BeautyROIManager::LIP_OUTER_INDICES[20] = {
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    409, 270, 269, 267, 0, 37, 39, 40, 185
};

//=============================================================================
// 유틸리티 함수
//=============================================================================

std::pair<int, int> BeautyROIManager::landmarkToMaskCoord(
    const IrisLandmark& lm,
    int mask_width, int mask_height,
    const Rect& face_rect) {

    // 정규화 좌표를 프레임 좌표로 변환 후 face_rect 기준으로 마스크 좌표로 변환
    // lm.x, lm.y는 0~1 정규화 좌표
    // face_rect는 프레임 좌표 (정규화)

    // 마스크 좌표로 스케일 (face_rect 내 상대 위치)
    int mx = static_cast<int>((lm.x - face_rect.x) * mask_width / face_rect.width);
    int my = static_cast<int>((lm.y - face_rect.y) * mask_height / face_rect.height);

    return {mx, my};
}

//=============================================================================
// ROI 계산
//=============================================================================

bool BeautyROIManager::computeROI(
    const IrisLandmark* face_mesh,
    int landmark_count,
    int frame_width, int frame_height,
    const BeautyFilterConfigV2& config,
    BeautyROI& out_roi) {

    // 입력 검증
    if (!face_mesh || landmark_count < FACE_MESH_LANDMARK_COUNT) {
        return false;
    }

    if (frame_width <= 0 || frame_height <= 0) {
        return false;
    }

    // 1. 얼굴 바운딩 박스 계산 (정규화 좌표)
    float min_x = 1.0f, min_y = 1.0f, max_x = 0.0f, max_y = 0.0f;
    for (int i = 0; i < FACE_OVAL_COUNT; i++) {
        int idx = FACE_OVAL_INDICES[i];
        if (idx >= landmark_count) continue;

        const auto& lm = face_mesh[idx];
        min_x = std::min(min_x, lm.x);
        min_y = std::min(min_y, lm.y);
        max_x = std::max(max_x, lm.x);
        max_y = std::max(max_y, lm.y);
    }

    // 패딩 추가 (10%)
    float padding_x = (max_x - min_x) * 0.1f;
    float padding_y = (max_y - min_y) * 0.1f;
    min_x = std::max(0.0f, min_x - padding_x);
    min_y = std::max(0.0f, min_y - padding_y);
    max_x = std::min(1.0f, max_x + padding_x);
    max_y = std::min(1.0f, max_y + padding_y);

    // 정규화 좌표 → 픽셀 좌표
    out_roi.face_rect.x = min_x;
    out_roi.face_rect.y = min_y;
    out_roi.face_rect.width = max_x - min_x;
    out_roi.face_rect.height = max_y - min_y;

    // 2. 마스크 크기 결정 (ROI 크기 또는 최대 256x256)
    int roi_pixel_width = static_cast<int>(out_roi.face_rect.width * frame_width);
    int roi_pixel_height = static_cast<int>(out_roi.face_rect.height * frame_height);

    out_roi.mask_width = std::min(roi_pixel_width, 256);
    out_roi.mask_height = std::min(roi_pixel_height, 256);

    // 최소 크기 보장
    out_roi.mask_width = std::max(out_roi.mask_width, 64);
    out_roi.mask_height = std::max(out_roi.mask_height, 64);

    // 3. 피부 마스크 생성
    createSkinMask(face_mesh, out_roi.mask_width, out_roi.mask_height,
                   frame_width, frame_height, out_roi.skin_mask);

    // 4. 보호 영역 마스크
    if (config.protectEyes) {
        createEyeProtectionMask(face_mesh, out_roi.mask_width, out_roi.mask_height,
                                frame_width, frame_height, out_roi.eye_protect_mask);
    } else {
        out_roi.eye_protect_mask.assign(
            static_cast<size_t>(out_roi.mask_width) * out_roi.mask_height, 0);
    }

    // 눈썹은 항상 보호 (피부 블러링 시 눈썹 흐려짐 방지)
    createEyebrowProtectionMask(face_mesh, out_roi.mask_width, out_roi.mask_height,
                                frame_width, frame_height, out_roi.eyebrow_protect_mask);

    if (config.protectLips) {
        createLipProtectionMask(face_mesh, out_roi.mask_width, out_roi.mask_height,
                                frame_width, frame_height, out_roi.lip_protect_mask);
    } else {
        out_roi.lip_protect_mask.assign(
            static_cast<size_t>(out_roi.mask_width) * out_roi.mask_height, 0);
    }

    // 5. 마스크 합성 (눈썹 보호 포함)
    combineMasks(out_roi.skin_mask, out_roi.eye_protect_mask,
                 out_roi.eyebrow_protect_mask, out_roi.lip_protect_mask,
                 out_roi.combined_mask);

    // 6. 페더링 적용
    applyFeathering(out_roi.combined_mask, out_roi.mask_width,
                    out_roi.mask_height, 15);

    // 타임스탬프 설정
    auto now = std::chrono::steady_clock::now();
    out_roi.timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();

    out_roi.valid = true;
    return true;
}

//=============================================================================
// 마스크 생성 함수들
//=============================================================================

void BeautyROIManager::createSkinMask(
    const IrisLandmark* face_mesh,
    int mask_width, int mask_height,
    int frame_width, int frame_height,
    std::vector<uint8_t>& out_mask) {

    out_mask.assign(static_cast<size_t>(mask_width) * mask_height, 0);

    // 얼굴 윤곽 좌표 수집 (정규화 → 마스크 좌표)
    std::vector<cv::Point> contour;
    contour.reserve(FACE_OVAL_COUNT);

    for (int i = 0; i < FACE_OVAL_COUNT; i++) {
        const auto& lm = face_mesh[FACE_OVAL_INDICES[i]];
        int x = static_cast<int>(lm.x * mask_width);  // 전면카메라 미러링 보정
        int y = static_cast<int>(lm.y * mask_height);

        // 범위 제한
        x = std::clamp(x, 0, mask_width - 1);
        y = std::clamp(y, 0, mask_height - 1);

        contour.emplace_back(x, y);
    }

    // "피부 판정" 대신 얼굴 안쪽 finish 영역에 가깝게 마스크를 단순화한다.
    // 외곽 턱선/헤어라인까지 꽉 채우면 foundation가 아니라 cutout처럼 보이기 쉬워서,
    // face oval을 중심 방향으로 한 번 더 축소한 내부 contour를 사용한다.
    cv::Point2f contour_center(0.0f, 0.0f);
    for (const auto& pt : contour) {
        contour_center.x += static_cast<float>(pt.x);
        contour_center.y += static_cast<float>(pt.y);
    }
    contour_center.x /= static_cast<float>(contour.size());
    contour_center.y /= static_cast<float>(contour.size());

    std::vector<cv::Point> inner_contour;
    inner_contour.reserve(contour.size());
    for (const auto& pt : contour) {
        const bool upper_half = static_cast<float>(pt.y) < contour_center.y;
        const float scale_x = upper_half ? 0.90f : 0.94f;
        const float scale_y = upper_half ? 0.84f : 0.96f;
        const float inner_x = contour_center.x + (static_cast<float>(pt.x) - contour_center.x) * scale_x;
        const float inner_y = contour_center.y + (static_cast<float>(pt.y) - contour_center.y) * scale_y;
        inner_contour.emplace_back(
            std::clamp(static_cast<int>(std::round(inner_x)), 0, mask_width - 1),
            std::clamp(static_cast<int>(std::round(inner_y)), 0, mask_height - 1)
        );
    }

    // OpenCV로 다각형 채우기
    cv::Mat mask_mat(mask_height, mask_width, CV_8UC1, out_mask.data());
    std::vector<std::vector<cv::Point>> contours = {inner_contour};
    cv::fillPoly(mask_mat, contours, cv::Scalar(255));
}

void BeautyROIManager::createEyeProtectionMask(
    const IrisLandmark* face_mesh,
    int mask_width, int mask_height,
    int frame_width, int frame_height,
    std::vector<uint8_t>& out_mask) {

    out_mask.assign(static_cast<size_t>(mask_width) * mask_height, 0);
    cv::Mat mask_mat(mask_height, mask_width, CV_8UC1, out_mask.data());

    // 왼쪽 눈
    std::vector<cv::Point> left_eye;
    left_eye.reserve(LEFT_EYE_COUNT);
    for (int i = 0; i < LEFT_EYE_COUNT; i++) {
        const auto& lm = face_mesh[LEFT_EYE_INDICES[i]];
        int x = std::clamp(static_cast<int>(lm.x * mask_width), 0, mask_width - 1);
        int y = std::clamp(static_cast<int>(lm.y * mask_height), 0, mask_height - 1);
        left_eye.emplace_back(x, y);
    }
    std::vector<std::vector<cv::Point>> left_contours = {left_eye};
    cv::fillPoly(mask_mat, left_contours, cv::Scalar(255));

    // 오른쪽 눈
    std::vector<cv::Point> right_eye;
    right_eye.reserve(RIGHT_EYE_COUNT);
    for (int i = 0; i < RIGHT_EYE_COUNT; i++) {
        const auto& lm = face_mesh[RIGHT_EYE_INDICES[i]];
        int x = std::clamp(static_cast<int>(lm.x * mask_width), 0, mask_width - 1);
        int y = std::clamp(static_cast<int>(lm.y * mask_height), 0, mask_height - 1);
        right_eye.emplace_back(x, y);
    }
    std::vector<std::vector<cv::Point>> right_contours = {right_eye};
    cv::fillPoly(mask_mat, right_contours, cv::Scalar(255));

    // 눈 영역 약간 확장 (자연스러운 보호)
    cv::dilate(mask_mat, mask_mat,
               cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 5)));
}

void BeautyROIManager::createEyebrowProtectionMask(
    const IrisLandmark* face_mesh,
    int mask_width, int mask_height,
    int frame_width, int frame_height,
    std::vector<uint8_t>& out_mask) {

    out_mask.assign(static_cast<size_t>(mask_width) * mask_height, 0);
    cv::Mat mask_mat(mask_height, mask_width, CV_8UC1, out_mask.data());

    // 왼쪽 눈썹
    std::vector<cv::Point> left_eyebrow;
    left_eyebrow.reserve(LEFT_EYEBROW_COUNT);
    for (int i = 0; i < LEFT_EYEBROW_COUNT; i++) {
        const auto& lm = face_mesh[LEFT_EYEBROW_INDICES[i]];
        int x = std::clamp(static_cast<int>(lm.x * mask_width), 0, mask_width - 1);
        int y = std::clamp(static_cast<int>(lm.y * mask_height), 0, mask_height - 1);
        left_eyebrow.emplace_back(x, y);
    }
    std::vector<std::vector<cv::Point>> left_contours = {left_eyebrow};
    cv::fillPoly(mask_mat, left_contours, cv::Scalar(255));

    // 오른쪽 눈썹
    std::vector<cv::Point> right_eyebrow;
    right_eyebrow.reserve(RIGHT_EYEBROW_COUNT);
    for (int i = 0; i < RIGHT_EYEBROW_COUNT; i++) {
        const auto& lm = face_mesh[RIGHT_EYEBROW_INDICES[i]];
        int x = std::clamp(static_cast<int>(lm.x * mask_width), 0, mask_width - 1);
        int y = std::clamp(static_cast<int>(lm.y * mask_height), 0, mask_height - 1);
        right_eyebrow.emplace_back(x, y);
    }
    std::vector<std::vector<cv::Point>> right_contours = {right_eyebrow};
    cv::fillPoly(mask_mat, right_contours, cv::Scalar(255));

    // 눈썹 영역 약간 확장 (자연스러운 경계)
    cv::dilate(mask_mat, mask_mat,
               cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 3)));
}

void BeautyROIManager::createLipProtectionMask(
    const IrisLandmark* face_mesh,
    int mask_width, int mask_height,
    int frame_width, int frame_height,
    std::vector<uint8_t>& out_mask) {

    out_mask.assign(static_cast<size_t>(mask_width) * mask_height, 0);
    cv::Mat mask_mat(mask_height, mask_width, CV_8UC1, out_mask.data());

    // 입술
    std::vector<cv::Point> lips;
    lips.reserve(LIPS_COUNT);
    for (int i = 0; i < LIPS_COUNT; i++) {
        const auto& lm = face_mesh[LIPS_INDICES[i]];
        int x = std::clamp(static_cast<int>(lm.x * mask_width), 0, mask_width - 1);
        int y = std::clamp(static_cast<int>(lm.y * mask_height), 0, mask_height - 1);
        lips.emplace_back(x, y);
    }
    std::vector<std::vector<cv::Point>> contours = {lips};
    cv::fillPoly(mask_mat, contours, cv::Scalar(255));

    // 입술 영역 약간 확장
    cv::dilate(mask_mat, mask_mat,
               cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
}

//=============================================================================
// 마스크 합성 및 페더링
//=============================================================================

void BeautyROIManager::combineMasks(
    const std::vector<uint8_t>& skin_mask,
    const std::vector<uint8_t>& eye_protect_mask,
    const std::vector<uint8_t>& eyebrow_protect_mask,
    const std::vector<uint8_t>& lip_protect_mask,
    std::vector<uint8_t>& out_combined) {

    size_t size = skin_mask.size();
    out_combined.resize(size);

    // 빈 마스크 처리
    bool has_eye = !eye_protect_mask.empty() && eye_protect_mask.size() == size;
    bool has_eyebrow = !eyebrow_protect_mask.empty() && eyebrow_protect_mask.size() == size;
    bool has_lip = !lip_protect_mask.empty() && lip_protect_mask.size() == size;

    for (size_t i = 0; i < size; i++) {
        float skin = skin_mask[i] / 255.0f;
        float eye = has_eye ? eye_protect_mask[i] / 255.0f : 0.0f;
        float eyebrow = has_eyebrow ? eyebrow_protect_mask[i] / 255.0f : 0.0f;
        float lip = has_lip ? lip_protect_mask[i] / 255.0f : 0.0f;

        // combined = skin * (1 - eye) * (1 - eyebrow) * (1 - lip)
        // 보호 영역에서는 0이 되어 필터가 적용되지 않음
        float combined = skin * (1.0f - eye) * (1.0f - eyebrow) * (1.0f - lip);
        out_combined[i] = static_cast<uint8_t>(std::clamp(combined * 255.0f, 0.0f, 255.0f));
    }
}

void BeautyROIManager::applyFeathering(
    std::vector<uint8_t>& mask,
    int width, int height,
    int feather_radius) {

    if (feather_radius <= 0) return;
    if (mask.empty()) return;

    cv::Mat mask_mat(height, width, CV_8UC1, mask.data());

    // Gaussian Blur로 경계 부드럽게
    int kernel_size = feather_radius * 2 + 1;
    cv::GaussianBlur(mask_mat, mask_mat, cv::Size(kernel_size, kernel_size), 0);
}

//=============================================================================
// P2-W2-01: ROI 기반 처리 및 페더링 통합
//=============================================================================

bool BeautyROIManager::extractROIRegion(
    const cv::Mat& full_frame,
    const BeautyROI& roi,
    cv::Mat& out_roi_region,
    cv::Rect& out_actual_rect,
    int padding) {

    if (full_frame.empty() || !roi.valid) {
        return false;
    }

    // 정규화 좌표를 픽셀 좌표로 변환
    int roi_x = static_cast<int>(roi.face_rect.x * full_frame.cols);
    int roi_y = static_cast<int>(roi.face_rect.y * full_frame.rows);
    int roi_w = static_cast<int>(roi.face_rect.width * full_frame.cols);
    int roi_h = static_cast<int>(roi.face_rect.height * full_frame.rows);

    // 패딩 적용된 ROI 계산
    int padded_x = std::max(0, roi_x - padding);
    int padded_y = std::max(0, roi_y - padding);
    int padded_right = std::min(full_frame.cols, roi_x + roi_w + padding);
    int padded_bottom = std::min(full_frame.rows, roi_y + roi_h + padding);

    out_actual_rect = cv::Rect(
        padded_x, padded_y,
        padded_right - padded_x,
        padded_bottom - padded_y
    );

    // ROI 추출
    out_roi_region = full_frame(out_actual_rect).clone();

    return true;
}

void BeautyROIManager::applyROIRegion(
    cv::Mat& full_frame,
    const cv::Mat& roi_region,
    const cv::Rect& actual_rect,
    const cv::Mat& feather_mask) {

    if (full_frame.empty() || roi_region.empty()) {
        return;
    }

    cv::Mat roi_target = full_frame(actual_rect);

    if (feather_mask.empty()) {
        // 직접 복사
        roi_region.copyTo(roi_target);
    } else {
        // 페더링 마스크로 블렌딩
        cv::Mat mask_resized;
        if (feather_mask.cols != actual_rect.width ||
            feather_mask.rows != actual_rect.height) {
            cv::resize(feather_mask, mask_resized,
                       cv::Size(actual_rect.width, actual_rect.height));
        } else {
            mask_resized = feather_mask;
        }

        cv::Mat mask_3ch;
        cv::cvtColor(mask_resized, mask_3ch, cv::COLOR_GRAY2BGR);
        mask_3ch.convertTo(mask_3ch, CV_32F, 1.0 / 255.0);

        cv::Mat roi_f, target_f;
        roi_region.convertTo(roi_f, CV_32F);
        roi_target.convertTo(target_f, CV_32F);

        // 블렌딩: result = roi * mask + original * (1 - mask)
        cv::Mat blended = roi_f.mul(mask_3ch) +
                          target_f.mul(cv::Scalar(1.0, 1.0, 1.0) - mask_3ch);
        blended.convertTo(roi_target, CV_8U);
    }
}

cv::Mat BeautyROIManager::createFeatherMask(
    const BeautyROI& roi,
    int feather_radius,
    const std::vector<uint8_t>& skin_mask,
    const std::vector<uint8_t>& protection_mask) {

    if (skin_mask.empty() || roi.mask_width <= 0 || roi.mask_height <= 0) {
        return cv::Mat();
    }

    // 1. 기본 피부 마스크에서 시작
    cv::Mat combined_mask(roi.mask_height, roi.mask_width, CV_8UC1,
                          const_cast<uint8_t*>(skin_mask.data()));
    combined_mask = combined_mask.clone();  // 원본 수정 방지

    // 2. 보호 영역 제외 (눈/입술)
    if (!protection_mask.empty() && protection_mask.size() == skin_mask.size()) {
        cv::Mat protect_mat(roi.mask_height, roi.mask_width, CV_8UC1,
                            const_cast<uint8_t*>(protection_mask.data()));
        combined_mask.setTo(0, protect_mat);
    }

    // 3. 가우시안 블러로 소프트 페더링
    cv::Mat feathered;
    int blur_size = feather_radius * 2 + 1;
    cv::GaussianBlur(combined_mask, feathered, cv::Size(blur_size, blur_size), 0);

    return feathered;
}

bool BeautyROIManager::createProtectionMasks(
    const IrisLandmark* face_mesh,
    int mask_width, int mask_height,
    const BeautyFilterConfigV2& config,
    ProtectionMasks& out_masks) {

    if (!face_mesh || mask_width <= 0 || mask_height <= 0) {
        return false;
    }

    out_masks.width = mask_width;
    out_masks.height = mask_height;
    size_t mask_size = static_cast<size_t>(mask_width) * mask_height;
    out_masks.combined.assign(mask_size, 0);
    out_masks.valid = false;

    cv::Mat combined_mat(mask_height, mask_width, CV_8UC1, out_masks.combined.data());

    // 눈 보호
    if (config.protectEyes) {
        createEyeMasks(face_mesh, mask_width, mask_height, 1.3f,
                       out_masks.left_eye, out_masks.right_eye);

        if (!out_masks.left_eye.empty()) {
            cv::Mat left_mat(mask_height, mask_width, CV_8UC1, out_masks.left_eye.data());
            combined_mat |= left_mat;
        }
        if (!out_masks.right_eye.empty()) {
            cv::Mat right_mat(mask_height, mask_width, CV_8UC1, out_masks.right_eye.data());
            combined_mat |= right_mat;
        }
    }

    // 입술 보호
    if (config.protectLips) {
        createLipMask(face_mesh, mask_width, mask_height, 1.2f, out_masks.lips);

        if (!out_masks.lips.empty()) {
            cv::Mat lip_mat(mask_height, mask_width, CV_8UC1, out_masks.lips.data());
            combined_mat |= lip_mat;
        }
    }

    out_masks.valid = true;
    return true;
}

void BeautyROIManager::createEyeMasks(
    const IrisLandmark* face_mesh,
    int mask_width, int mask_height,
    float expansion_ratio,
    std::vector<uint8_t>& out_left,
    std::vector<uint8_t>& out_right) {

    size_t mask_size = static_cast<size_t>(mask_width) * mask_height;
    out_left.assign(mask_size, 0);
    out_right.assign(mask_size, 0);

    cv::Mat left_mat(mask_height, mask_width, CV_8UC1, out_left.data());
    cv::Mat right_mat(mask_height, mask_width, CV_8UC1, out_right.data());

    // 왼쪽 눈 - 랜드마크에서 바운딩 박스 계산
    float left_min_x = 1.0f, left_min_y = 1.0f, left_max_x = 0.0f, left_max_y = 0.0f;
    for (int i = 0; i < LEFT_EYE_COUNT; ++i) {
        int idx = LEFT_EYE_INDICES[i];
        left_min_x = std::min(left_min_x, face_mesh[idx].x);
        left_min_y = std::min(left_min_y, face_mesh[idx].y);
        left_max_x = std::max(left_max_x, face_mesh[idx].x);
        left_max_y = std::max(left_max_y, face_mesh[idx].y);
    }

    // 마스크 좌표로 변환 및 확장
    int left_cx = static_cast<int>((left_min_x + left_max_x) / 2 * mask_width);
    int left_cy = static_cast<int>((left_min_y + left_max_y) / 2 * mask_height);
    int left_rx = static_cast<int>((left_max_x - left_min_x) / 2 * mask_width * expansion_ratio);
    int left_ry = static_cast<int>((left_max_y - left_min_y) / 2 * mask_height * expansion_ratio);

    // 최소 크기 보장
    left_rx = std::max(left_rx, 5);
    left_ry = std::max(left_ry, 3);

    // 타원으로 마스크 생성
    cv::ellipse(left_mat, cv::Point(left_cx, left_cy),
                cv::Size(left_rx, left_ry), 0, 0, 360, cv::Scalar(255), -1);

    // 오른쪽 눈 - 동일 로직
    float right_min_x = 1.0f, right_min_y = 1.0f, right_max_x = 0.0f, right_max_y = 0.0f;
    for (int i = 0; i < RIGHT_EYE_COUNT; ++i) {
        int idx = RIGHT_EYE_INDICES[i];
        right_min_x = std::min(right_min_x, face_mesh[idx].x);
        right_min_y = std::min(right_min_y, face_mesh[idx].y);
        right_max_x = std::max(right_max_x, face_mesh[idx].x);
        right_max_y = std::max(right_max_y, face_mesh[idx].y);
    }

    int right_cx = static_cast<int>((right_min_x + right_max_x) / 2 * mask_width);
    int right_cy = static_cast<int>((right_min_y + right_max_y) / 2 * mask_height);
    int right_rx = static_cast<int>((right_max_x - right_min_x) / 2 * mask_width * expansion_ratio);
    int right_ry = static_cast<int>((right_max_y - right_min_y) / 2 * mask_height * expansion_ratio);

    right_rx = std::max(right_rx, 5);
    right_ry = std::max(right_ry, 3);

    cv::ellipse(right_mat, cv::Point(right_cx, right_cy),
                cv::Size(right_rx, right_ry), 0, 0, 360, cv::Scalar(255), -1);
}

void BeautyROIManager::createLipMask(
    const IrisLandmark* face_mesh,
    int mask_width, int mask_height,
    float expansion_ratio,
    std::vector<uint8_t>& out_lips) {

    size_t mask_size = static_cast<size_t>(mask_width) * mask_height;
    out_lips.assign(mask_size, 0);

    cv::Mat lip_mat(mask_height, mask_width, CV_8UC1, out_lips.data());

    // 외곽 입술 랜드마크에서 바운딩 박스 계산
    float min_x = 1.0f, min_y = 1.0f, max_x = 0.0f, max_y = 0.0f;
    for (int i = 0; i < LIP_OUTER_COUNT; ++i) {
        int idx = LIP_OUTER_INDICES[i];
        min_x = std::min(min_x, face_mesh[idx].x);
        min_y = std::min(min_y, face_mesh[idx].y);
        max_x = std::max(max_x, face_mesh[idx].x);
        max_y = std::max(max_y, face_mesh[idx].y);
    }

    // 마스크 좌표로 변환 및 확장
    int cx = static_cast<int>((min_x + max_x) / 2 * mask_width);
    int cy = static_cast<int>((min_y + max_y) / 2 * mask_height);
    int rx = static_cast<int>((max_x - min_x) / 2 * mask_width * expansion_ratio);
    int ry = static_cast<int>((max_y - min_y) / 2 * mask_height * expansion_ratio);

    // 최소 크기 보장
    rx = std::max(rx, 10);
    ry = std::max(ry, 5);

    // 타원으로 마스크 생성 (입술은 수평으로 더 넓음)
    cv::ellipse(lip_mat, cv::Point(cx, cy),
                cv::Size(rx, ry), 0, 0, 360, cv::Scalar(255), -1);
}

} // namespace iris_sdk
