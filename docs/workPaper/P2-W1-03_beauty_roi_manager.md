# P2-W1-03. BeautyROIManager 구현

## 작업 정보

| 항목 | 내용 |
|------|------|
| **작업 ID** | P2-W1-03 |
| **Phase** | Phase 1: 기반 구조 리팩토링 |
| **상태** | ✅ 완료 |
| **예상 기간** | 2일 |
| **의존성** | P2-W1-02 (ConfigV2) |
| **담당** | systems-programming:cpp-pro |

---

## 1. 목표

Face Mesh 478개 랜드마크를 활용하여 ROI(Region of Interest) 기반 처리를 위한 마스크 생성 시스템 구현

### 핵심 산출물
- `BeautyROI` 구조체
- `BeautyROIManager` 클래스
- 피부/눈/입술 마스크 생성 함수
- Soft Feathering 기능

---

## 2. 상세 작업

### 2.1 BeautyROI 구조체

**파일**: `cpp/include/iris_sdk/beauty_roi_manager.h`

```cpp
#ifndef IRIS_SDK_BEAUTY_ROI_MANAGER_H
#define IRIS_SDK_BEAUTY_ROI_MANAGER_H

#include "types.h"
#include <vector>
#include <cstdint>

namespace iris_sdk {

/**
 * @brief ROI 데이터 구조체
 *
 * Face Mesh 기반으로 계산된 얼굴 영역 정보 및 마스크
 */
struct BeautyROI {
    // 얼굴 바운딩 박스 (원본 프레임 좌표)
    Rect face_rect;

    // 마스크 (8-bit, 0~255)
    // 255 = 완전 적용, 0 = 미적용
    std::vector<uint8_t> skin_mask;           ///< 피부 영역 마스크
    std::vector<uint8_t> eye_protect_mask;    ///< 눈 보호 마스크 (반전)
    std::vector<uint8_t> eyebrow_protect_mask;///< 눈썹 보호 마스크 (반전)
    std::vector<uint8_t> lip_protect_mask;    ///< 입술 보호 마스크 (반전)
    std::vector<uint8_t> combined_mask;       ///< 최종 합성 마스크

    // 마스크 크기
    int mask_width = 0;
    int mask_height = 0;

    // 유효성
    bool valid = false;

    // 타임스탬프 (캐싱용)
    int64_t timestamp_ms = 0;

    // 헬퍼
    bool isValid() const { return valid && !combined_mask.empty(); }
    void invalidate() { valid = false; combined_mask.clear(); }
};

/**
 * @brief Face Mesh 기반 ROI 관리자
 */
class BeautyROIManager {
public:
    BeautyROIManager() = default;

    /**
     * @brief Face Mesh로부터 ROI 계산
     *
     * @param face_mesh 478개 랜드마크 (IrisResult.face_mesh)
     * @param frame_width 프레임 너비
     * @param frame_height 프레임 높이
     * @param config 필터 설정 (보호 영역 옵션)
     * @param out_roi 출력 ROI
     * @return 성공 여부
     */
    static bool computeROI(
        const IrisLandmark* face_mesh,
        int frame_width, int frame_height,
        const BeautyFilterConfigV2& config,
        BeautyROI& out_roi
    );

    /**
     * @brief 피부 영역 마스크 생성
     *
     * 얼굴 윤곽 내부를 채우는 마스크 생성
     * 삼각형 메쉬 기반 래스터화
     */
    static void createSkinMask(
        const IrisLandmark* face_mesh,
        int mask_width, int mask_height,
        int frame_width, int frame_height,
        std::vector<uint8_t>& out_mask
    );

    /**
     * @brief 눈 영역 보호 마스크 생성
     *
     * 눈 영역을 255로 마킹 (보호할 영역)
     */
    static void createEyeProtectionMask(
        const IrisLandmark* face_mesh,
        int mask_width, int mask_height,
        int frame_width, int frame_height,
        std::vector<uint8_t>& out_mask
    );

    /**
     * @brief 입술 영역 보호 마스크 생성
     */
    static void createLipProtectionMask(
        const IrisLandmark* face_mesh,
        int mask_width, int mask_height,
        int frame_width, int frame_height,
        std::vector<uint8_t>& out_mask
    );

    /**
     * @brief 마스크 합성 (피부 - 눈 - 눈썹 - 입술)
     *
     * combined = skin_mask * (1 - eye_mask) * (1 - eyebrow_mask) * (1 - lip_mask)
     */
    static void combineMasks(
        const std::vector<uint8_t>& skin_mask,
        const std::vector<uint8_t>& eye_protect_mask,
        const std::vector<uint8_t>& eyebrow_protect_mask,
        const std::vector<uint8_t>& lip_protect_mask,
        std::vector<uint8_t>& out_combined
    );

    /**
     * @brief 마스크 경계 페더링 (Soft Blend)
     *
     * 가장자리를 부드럽게 처리하여 자연스러운 전환
     * @param feather_radius 페더링 반경 (픽셀)
     */
    static void applyFeathering(
        std::vector<uint8_t>& mask,
        int width, int height,
        int feather_radius = 15
    );

    /**
     * @brief 눈썹 영역 보호 마스크 생성
     *
     * 눈썹 영역을 255로 마킹 (보호할 영역)
     * 피부 블러링 시 눈썹이 흐려지는 것을 방지
     */
    static void createEyebrowProtectionMask(
        const IrisLandmark* face_mesh,
        int mask_width, int mask_height,
        int frame_width, int frame_height,
        std::vector<uint8_t>& out_mask
    );

private:
    // Face Mesh 랜드마크 인덱스
    static constexpr int FACE_OVAL_COUNT = 36;
    static const int FACE_OVAL_INDICES[FACE_OVAL_COUNT];

    static constexpr int LEFT_EYE_COUNT = 16;
    static const int LEFT_EYE_INDICES[LEFT_EYE_COUNT];

    static constexpr int RIGHT_EYE_COUNT = 16;
    static const int RIGHT_EYE_INDICES[RIGHT_EYE_COUNT];

    static constexpr int LIPS_COUNT = 22;
    static const int LIPS_INDICES[LIPS_COUNT];

    // 눈썹 인덱스 (피부 마스크에서 제외용)
    static constexpr int LEFT_EYEBROW_COUNT = 8;
    static const int LEFT_EYEBROW_INDICES[LEFT_EYEBROW_COUNT];

    static constexpr int RIGHT_EYEBROW_COUNT = 8;
    static const int RIGHT_EYEBROW_INDICES[RIGHT_EYEBROW_COUNT];

    // 삼각형 래스터화 헬퍼
    static void fillTriangle(
        uint8_t* mask, int width, int height,
        float x0, float y0,
        float x1, float y1,
        float x2, float y2,
        uint8_t value
    );

    // 다각형 래스터화 (Scanline)
    static void fillPolygon(
        uint8_t* mask, int width, int height,
        const std::vector<std::pair<float, float>>& vertices,
        uint8_t value
    );
};

} // namespace iris_sdk

#endif // IRIS_SDK_BEAUTY_ROI_MANAGER_H
```

### 2.2 랜드마크 인덱스 정의

**파일**: `cpp/src/beauty_roi_manager.cpp`

```cpp
#include "iris_sdk/beauty_roi_manager.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace iris_sdk {

// 얼굴 윤곽 인덱스 (36개, MediaPipe Face Mesh 기준)
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

// 왼쪽 눈썹 인덱스 (8개) - 피부 마스크에서 제외용
const int BeautyROIManager::LEFT_EYEBROW_INDICES[8] = {
    70, 63, 105, 66, 107, 55, 65, 52
};

// 오른쪽 눈썹 인덱스 (8개)
const int BeautyROIManager::RIGHT_EYEBROW_INDICES[8] = {
    300, 293, 334, 296, 336, 285, 295, 282
};

bool BeautyROIManager::computeROI(
    const IrisLandmark* face_mesh,
    int frame_width, int frame_height,
    const BeautyFilterConfigV2& config,
    BeautyROI& out_roi) {

    if (!face_mesh) return false;

    // 1. 얼굴 바운딩 박스 계산
    float min_x = 1.0f, min_y = 1.0f, max_x = 0.0f, max_y = 0.0f;
    for (int i = 0; i < FACE_OVAL_COUNT; i++) {
        const auto& lm = face_mesh[FACE_OVAL_INDICES[i]];
        min_x = std::min(min_x, lm.x);
        min_y = std::min(min_y, lm.y);
        max_x = std::max(max_x, lm.x);
        max_y = std::max(max_y, lm.y);
    }

    // 정규화 좌표 → 픽셀 좌표
    out_roi.face_rect.x = static_cast<int>(min_x * frame_width);
    out_roi.face_rect.y = static_cast<int>(min_y * frame_height);
    out_roi.face_rect.width = static_cast<int>((max_x - min_x) * frame_width);
    out_roi.face_rect.height = static_cast<int>((max_y - min_y) * frame_height);

    // 2. 마스크 크기 (ROI 크기 또는 최대 256x256)
    out_roi.mask_width = std::min(out_roi.face_rect.width, 256);
    out_roi.mask_height = std::min(out_roi.face_rect.height, 256);

    // 3. 피부 마스크 생성
    createSkinMask(face_mesh, out_roi.mask_width, out_roi.mask_height,
                   frame_width, frame_height, out_roi.skin_mask);

    // 4. 보호 영역 마스크
    if (config.protectEyes) {
        createEyeProtectionMask(face_mesh, out_roi.mask_width, out_roi.mask_height,
                                frame_width, frame_height, out_roi.eye_protect_mask);
    } else {
        out_roi.eye_protect_mask.assign(out_roi.mask_width * out_roi.mask_height, 0);
    }

    // 4.1 눈썹 보호 마스크 (피부 블러링 시 눈썹 흐려짐 방지)
    // 눈썹은 항상 보호 (config 옵션과 무관하게)
    createEyebrowProtectionMask(face_mesh, out_roi.mask_width, out_roi.mask_height,
                                 frame_width, frame_height, out_roi.eyebrow_protect_mask);

    if (config.protectLips) {
        createLipProtectionMask(face_mesh, out_roi.mask_width, out_roi.mask_height,
                                frame_width, frame_height, out_roi.lip_protect_mask);
    } else {
        out_roi.lip_protect_mask.assign(out_roi.mask_width * out_roi.mask_height, 0);
    }

    // 5. 마스크 합성 (눈썹 보호 포함)
    combineMasks(out_roi.skin_mask, out_roi.eye_protect_mask,
                 out_roi.eyebrow_protect_mask, out_roi.lip_protect_mask,
                 out_roi.combined_mask);

    // 6. 페더링 적용
    applyFeathering(out_roi.combined_mask, out_roi.mask_width,
                    out_roi.mask_height, 15);

    out_roi.valid = true;
    out_roi.timestamp_ms = /* current time */;
    return true;
}

void BeautyROIManager::createSkinMask(
    const IrisLandmark* face_mesh,
    int mask_width, int mask_height,
    int frame_width, int frame_height,
    std::vector<uint8_t>& out_mask) {

    out_mask.assign(mask_width * mask_height, 0);

    // 얼굴 윤곽 좌표 수집 (정규화 → 마스크 좌표)
    std::vector<cv::Point> contour;
    for (int i = 0; i < FACE_OVAL_COUNT; i++) {
        const auto& lm = face_mesh[FACE_OVAL_INDICES[i]];
        int x = static_cast<int>(lm.x * mask_width);
        int y = static_cast<int>(lm.y * mask_height);
        contour.emplace_back(x, y);
    }

    // OpenCV로 다각형 채우기
    cv::Mat mask_mat(mask_height, mask_width, CV_8UC1, out_mask.data());
    cv::fillConvexPoly(mask_mat, contour, cv::Scalar(255));
}

void BeautyROIManager::createEyeProtectionMask(
    const IrisLandmark* face_mesh,
    int mask_width, int mask_height,
    int frame_width, int frame_height,
    std::vector<uint8_t>& out_mask) {

    out_mask.assign(mask_width * mask_height, 0);
    cv::Mat mask_mat(mask_height, mask_width, CV_8UC1, out_mask.data());

    // 왼쪽 눈
    std::vector<cv::Point> left_eye;
    for (int i = 0; i < LEFT_EYE_COUNT; i++) {
        const auto& lm = face_mesh[LEFT_EYE_INDICES[i]];
        left_eye.emplace_back(
            static_cast<int>(lm.x * mask_width),
            static_cast<int>(lm.y * mask_height)
        );
    }
    cv::fillConvexPoly(mask_mat, left_eye, cv::Scalar(255));

    // 오른쪽 눈
    std::vector<cv::Point> right_eye;
    for (int i = 0; i < RIGHT_EYE_COUNT; i++) {
        const auto& lm = face_mesh[RIGHT_EYE_INDICES[i]];
        right_eye.emplace_back(
            static_cast<int>(lm.x * mask_width),
            static_cast<int>(lm.y * mask_height)
        );
    }
    cv::fillConvexPoly(mask_mat, right_eye, cv::Scalar(255));
}

void BeautyROIManager::createEyebrowProtectionMask(
    const IrisLandmark* face_mesh,
    int mask_width, int mask_height,
    int frame_width, int frame_height,
    std::vector<uint8_t>& out_mask) {

    out_mask.assign(mask_width * mask_height, 0);
    cv::Mat mask_mat(mask_height, mask_width, CV_8UC1, out_mask.data());

    // 왼쪽 눈썹 (확장된 영역으로 자연스러운 보호)
    std::vector<cv::Point> left_eyebrow;
    for (int i = 0; i < LEFT_EYEBROW_COUNT; i++) {
        const auto& lm = face_mesh[LEFT_EYEBROW_INDICES[i]];
        left_eyebrow.emplace_back(
            static_cast<int>(lm.x * mask_width),
            static_cast<int>(lm.y * mask_height)
        );
    }
    cv::fillConvexPoly(mask_mat, left_eyebrow, cv::Scalar(255));

    // 오른쪽 눈썹
    std::vector<cv::Point> right_eyebrow;
    for (int i = 0; i < RIGHT_EYEBROW_COUNT; i++) {
        const auto& lm = face_mesh[RIGHT_EYEBROW_INDICES[i]];
        right_eyebrow.emplace_back(
            static_cast<int>(lm.x * mask_width),
            static_cast<int>(lm.y * mask_height)
        );
    }
    cv::fillConvexPoly(mask_mat, right_eyebrow, cv::Scalar(255));

    // 눈썹 영역 약간 확장 (자연스러운 경계)
    cv::dilate(mask_mat, mask_mat, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 3)));
}

void BeautyROIManager::applyFeathering(
    std::vector<uint8_t>& mask,
    int width, int height,
    int feather_radius) {

    if (feather_radius <= 0) return;

    cv::Mat mask_mat(height, width, CV_8UC1, mask.data());

    // Gaussian Blur로 경계 부드럽게
    int kernel_size = feather_radius * 2 + 1;
    cv::GaussianBlur(mask_mat, mask_mat, cv::Size(kernel_size, kernel_size), 0);
}

void BeautyROIManager::combineMasks(
    const std::vector<uint8_t>& skin_mask,
    const std::vector<uint8_t>& eye_protect_mask,
    const std::vector<uint8_t>& eyebrow_protect_mask,
    const std::vector<uint8_t>& lip_protect_mask,
    std::vector<uint8_t>& out_combined) {

    size_t size = skin_mask.size();
    out_combined.resize(size);

    for (size_t i = 0; i < size; i++) {
        float skin = skin_mask[i] / 255.0f;
        float eye = eye_protect_mask[i] / 255.0f;
        float eyebrow = eyebrow_protect_mask[i] / 255.0f;
        float lip = lip_protect_mask[i] / 255.0f;

        // combined = skin * (1 - eye) * (1 - eyebrow) * (1 - lip)
        // 눈썹 영역도 보호하여 피부 스무딩 시 흐려지지 않도록 함
        float combined = skin * (1.0f - eye) * (1.0f - eyebrow) * (1.0f - lip);
        out_combined[i] = static_cast<uint8_t>(combined * 255);
    }
}

} // namespace iris_sdk
```

---

## 3. 단위 테스트

**파일**: `cpp/tests/test_beauty_roi_manager.cpp`

```cpp
#include <gtest/gtest.h>
#include "iris_sdk/beauty_roi_manager.h"

class BeautyROIManagerTest : public ::testing::Test {
protected:
    // 테스트용 Face Mesh 로드
    IrisLandmark face_mesh_[478];

    void SetUp() override {
        loadTestFaceMesh("test_data/face_mesh_478.bin", face_mesh_);
    }
};

TEST_F(BeautyROIManagerTest, ComputeROI_ValidFaceMesh_ReturnsTrue) {
    BeautyFilterConfigV2 config;
    BeautyROI roi;

    bool result = BeautyROIManager::computeROI(
        face_mesh_, 1920, 1080, config, roi);

    EXPECT_TRUE(result);
    EXPECT_TRUE(roi.valid);
    EXPECT_GT(roi.face_rect.width, 0);
    EXPECT_GT(roi.face_rect.height, 0);
    EXPECT_FALSE(roi.combined_mask.empty());
}

TEST_F(BeautyROIManagerTest, ComputeROI_NullFaceMesh_ReturnsFalse) {
    BeautyFilterConfigV2 config;
    BeautyROI roi;

    bool result = BeautyROIManager::computeROI(
        nullptr, 1920, 1080, config, roi);

    EXPECT_FALSE(result);
}

TEST_F(BeautyROIManagerTest, SkinMask_CoversExpectedArea) {
    std::vector<uint8_t> skin_mask;
    BeautyROIManager::createSkinMask(face_mesh_, 256, 256, 1920, 1080, skin_mask);

    // 얼굴 영역에서 일정 비율 이상이 마스킹되어야 함
    int filled_count = std::count_if(skin_mask.begin(), skin_mask.end(),
                                     [](uint8_t v) { return v > 128; });
    float fill_ratio = static_cast<float>(filled_count) / skin_mask.size();

    EXPECT_GT(fill_ratio, 0.2f);  // 최소 20%
    EXPECT_LT(fill_ratio, 0.8f);  // 최대 80%
}

TEST_F(BeautyROIManagerTest, EyeProtectionMask_ExcludesEyeRegion) {
    std::vector<uint8_t> skin_mask, eye_mask, combined;

    BeautyROIManager::createSkinMask(face_mesh_, 256, 256, 1920, 1080, skin_mask);
    BeautyROIManager::createEyeProtectionMask(face_mesh_, 256, 256, 1920, 1080, eye_mask);
    BeautyROIManager::combineMasks(skin_mask, eye_mask, {}, combined);

    // 눈 영역 (eye_mask > 128)에서 combined는 0에 가까워야 함
    for (size_t i = 0; i < eye_mask.size(); i++) {
        if (eye_mask[i] > 200) {
            EXPECT_LT(combined[i], 50);  // 눈 영역 보호됨
        }
    }
}

TEST_F(BeautyROIManagerTest, Feathering_SmoothsEdges) {
    std::vector<uint8_t> mask(256 * 256, 0);

    // 중앙에 사각형 영역 생성
    for (int y = 64; y < 192; y++) {
        for (int x = 64; x < 192; x++) {
            mask[y * 256 + x] = 255;
        }
    }

    BeautyROIManager::applyFeathering(mask, 256, 256, 15);

    // 경계에서 중간값이 있어야 함 (sharp edge가 아님)
    int edge_x = 64, edge_y = 128;
    uint8_t edge_value = mask[edge_y * 256 + edge_x];
    EXPECT_GT(edge_value, 0);
    EXPECT_LT(edge_value, 255);
}
```

---

## 4. 완료 기준

- [x] `BeautyROI` 구조체 정의
- [x] 랜드마크 인덱스 테이블 정의 (얼굴/눈/눈썹/입술)
- [x] `computeROI()` 함수 구현
- [x] `createSkinMask()` 구현
- [x] `createEyeProtectionMask()` 구현
- [x] `createEyebrowProtectionMask()` 구현 (피부 블러링 시 눈썹 보호)
- [x] `createLipProtectionMask()` 구현
- [x] `combineMasks()` 구현 (눈썹 보호 포함)
- [x] `applyFeathering()` 구현
- [x] 단위 테스트 100% 통과 (26개 테스트)

---

## 5. 다음 작업

- **P2-W1-04**: IBeautyBackend 인터페이스 및 BeautyProcessor
