/**
 * @file test_beauty_roi_manager.cpp
 * @brief BeautyROIManager 단위 테스트
 */

#include <gtest/gtest.h>
#include "iris_sdk/beauty_roi_manager.h"
#include <cmath>
#include <algorithm>

#ifdef IRIS_SDK_HAS_OPENCV
#include <opencv2/imgproc.hpp>
#endif

namespace iris_sdk {
namespace testing {

//=============================================================================
// 테스트 헬퍼
//=============================================================================

/**
 * @brief 테스트용 Face Mesh 데이터 생성
 *
 * 얼굴 중앙에 타원형 얼굴을 가정한 랜드마크 생성
 */
class FaceMeshGenerator {
public:
    static void generateCenteredFace(IrisLandmark* landmarks, int count = 478) {
        // 기본값: 중앙에 위치한 정규화 좌표
        for (int i = 0; i < count; i++) {
            // 랜덤 분포 대신 구조화된 위치 사용
            float angle = static_cast<float>(i) / count * 2.0f * 3.14159f;
            float radius = 0.2f + 0.1f * std::sin(angle * 3);

            landmarks[i].x = 0.5f + radius * std::cos(angle) * 0.5f;
            landmarks[i].y = 0.5f + radius * std::sin(angle) * 0.6f;
            landmarks[i].z = 0.0f;
            landmarks[i].visibility = 1.0f;
        }

        // 얼굴 윤곽 랜드마크 명시적 설정
        setFaceOvalLandmarks(landmarks);
        setEyeLandmarks(landmarks);
        setEyebrowLandmarks(landmarks);
        setLipLandmarks(landmarks);
    }

private:
    static void setFaceOvalLandmarks(IrisLandmark* landmarks) {
        // 얼굴 윤곽: 중앙(0.5, 0.5) 기준 타원형
        const int indices[] = {
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
            361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
            176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
            162, 21, 54, 103, 67, 109
        };
        int count = sizeof(indices) / sizeof(indices[0]);

        for (int i = 0; i < count; i++) {
            float angle = static_cast<float>(i) / count * 2.0f * 3.14159f;
            landmarks[indices[i]].x = 0.5f + 0.2f * std::cos(angle);
            landmarks[indices[i]].y = 0.5f + 0.25f * std::sin(angle);
            landmarks[indices[i]].visibility = 1.0f;
        }
    }

    static void setEyeLandmarks(IrisLandmark* landmarks) {
        // BeautyROIManager.LEFT_EYE_INDICES와 동일한 33그룹을 채운다.
        // ④ §7.3 canonical: 33그룹은 피험자 우안(canonical RIGHT). 눈 보호 마스크는
        //   양쪽을 union으로 마킹하므로 left/right 라벨은 동작 무관(아래 변수명은 소스
        //   상수명 LEFT_EYE_INDICES와의 대응 유지를 위해 보존). 좌표는 화면 좌측 배치.
        const int left_indices[] = {  // = LEFT_EYE_INDICES (33그룹 = 피험자 우안)
            33, 7, 163, 144, 145, 153, 154, 155, 133,
            173, 157, 158, 159, 160, 161, 246
        };
        for (int i = 0; i < 16; i++) {
            float angle = static_cast<float>(i) / 16 * 2.0f * 3.14159f;
            landmarks[left_indices[i]].x = 0.35f + 0.03f * std::cos(angle);
            landmarks[left_indices[i]].y = 0.4f + 0.015f * std::sin(angle);
            landmarks[left_indices[i]].visibility = 1.0f;
        }

        // BeautyROIManager.RIGHT_EYE_INDICES와 동일한 362그룹을 채운다.
        // ④ §7.3 canonical: 362그룹은 피험자 좌안(canonical LEFT). 좌표는 화면 우측 배치.
        const int right_indices[] = {  // = RIGHT_EYE_INDICES (362그룹 = 피험자 좌안)
            362, 382, 381, 380, 374, 373, 390, 249, 263,
            466, 388, 387, 386, 385, 384, 398
        };
        for (int i = 0; i < 16; i++) {
            float angle = static_cast<float>(i) / 16 * 2.0f * 3.14159f;
            landmarks[right_indices[i]].x = 0.65f + 0.03f * std::cos(angle);
            landmarks[right_indices[i]].y = 0.4f + 0.015f * std::sin(angle);
            landmarks[right_indices[i]].visibility = 1.0f;
        }
    }

    static void setEyebrowLandmarks(IrisLandmark* landmarks) {
        // 왼쪽 눈썹
        const int left_indices[] = {70, 63, 105, 66, 107, 55, 65, 52};
        for (int i = 0; i < 8; i++) {
            float t = static_cast<float>(i) / 7;
            landmarks[left_indices[i]].x = 0.28f + t * 0.14f;
            landmarks[left_indices[i]].y = 0.32f - 0.02f * std::sin(t * 3.14159f);
            landmarks[left_indices[i]].visibility = 1.0f;
        }

        // 오른쪽 눈썹
        const int right_indices[] = {300, 293, 334, 296, 336, 285, 295, 282};
        for (int i = 0; i < 8; i++) {
            float t = static_cast<float>(i) / 7;
            landmarks[right_indices[i]].x = 0.58f + t * 0.14f;
            landmarks[right_indices[i]].y = 0.32f - 0.02f * std::sin(t * 3.14159f);
            landmarks[right_indices[i]].visibility = 1.0f;
        }
    }

    static void setLipLandmarks(IrisLandmark* landmarks) {
        // 입술
        const int indices[] = {
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78
        };
        int count = sizeof(indices) / sizeof(indices[0]);

        for (int i = 0; i < count; i++) {
            float angle = static_cast<float>(i) / count * 2.0f * 3.14159f;
            landmarks[indices[i]].x = 0.5f + 0.06f * std::cos(angle);
            landmarks[indices[i]].y = 0.65f + 0.025f * std::sin(angle);
            landmarks[indices[i]].visibility = 1.0f;
        }
    }
};

//=============================================================================
// BeautyROI 기본 테스트
//=============================================================================

TEST(BeautyROITest, DefaultConstruction) {
    BeautyROI roi;

    EXPECT_FALSE(roi.valid);
    EXPECT_EQ(roi.mask_width, 0);
    EXPECT_EQ(roi.mask_height, 0);
    EXPECT_TRUE(roi.combined_mask.empty());
    EXPECT_FALSE(roi.isValid());
}

TEST(BeautyROITest, InvalidateClears) {
    BeautyROI roi;
    roi.valid = true;
    roi.combined_mask.resize(100, 128);

    roi.invalidate();

    EXPECT_FALSE(roi.valid);
    EXPECT_TRUE(roi.combined_mask.empty());
    EXPECT_FALSE(roi.isValid());
}

TEST(BeautyROITest, IsValidChecks) {
    BeautyROI roi;

    // valid = false, mask empty
    EXPECT_FALSE(roi.isValid());

    // valid = true, mask empty
    roi.valid = true;
    EXPECT_FALSE(roi.isValid());

    // valid = true, mask not empty
    roi.combined_mask.resize(100, 128);
    EXPECT_TRUE(roi.isValid());

    // valid = false, mask not empty
    roi.valid = false;
    EXPECT_FALSE(roi.isValid());
}

//=============================================================================
// ComputeROI 테스트
//=============================================================================

class BeautyROIManagerTest : public ::testing::Test {
protected:
    IrisLandmark face_mesh_[478];
    BeautyFilterConfigV2 config_;

    void SetUp() override {
        FaceMeshGenerator::generateCenteredFace(face_mesh_, 478);
        config_ = BeautyFilterConfigV2Helper::defaults();
        config_.protectEyes = true;
        config_.protectLips = true;
    }
};

TEST_F(BeautyROIManagerTest, ComputeROI_ValidInput_ReturnsTrue) {
    BeautyROI roi;

    bool result = BeautyROIManager::computeROI(
        face_mesh_, 478, 1920, 1080, config_, roi);

    EXPECT_TRUE(result);
    EXPECT_TRUE(roi.valid);
    EXPECT_TRUE(roi.isValid());
}

TEST_F(BeautyROIManagerTest, ComputeROI_ValidInput_HasFaceRect) {
    BeautyROI roi;

    BeautyROIManager::computeROI(face_mesh_, 478, 1920, 1080, config_, roi);

    // face_rect는 정규화 좌표 (0~1)
    EXPECT_GE(roi.face_rect.x, 0.0f);
    EXPECT_LE(roi.face_rect.x, 1.0f);
    EXPECT_GE(roi.face_rect.y, 0.0f);
    EXPECT_LE(roi.face_rect.y, 1.0f);
    EXPECT_GT(roi.face_rect.width, 0.0f);
    EXPECT_GT(roi.face_rect.height, 0.0f);
}

TEST_F(BeautyROIManagerTest, ComputeROI_ValidInput_HasMasks) {
    BeautyROI roi;

    BeautyROIManager::computeROI(face_mesh_, 478, 1920, 1080, config_, roi);

    size_t expected_size = static_cast<size_t>(roi.mask_width) * roi.mask_height;

    EXPECT_EQ(roi.skin_mask.size(), expected_size);
    EXPECT_EQ(roi.eye_protect_mask.size(), expected_size);
    EXPECT_EQ(roi.eyebrow_protect_mask.size(), expected_size);
    EXPECT_EQ(roi.lip_protect_mask.size(), expected_size);
    EXPECT_EQ(roi.combined_mask.size(), expected_size);
}

TEST_F(BeautyROIManagerTest, ComputeROI_ValidInput_HasTimestamp) {
    BeautyROI roi;

    BeautyROIManager::computeROI(face_mesh_, 478, 1920, 1080, config_, roi);

    EXPECT_GT(roi.timestamp_ms, 0);
}

TEST_F(BeautyROIManagerTest, ComputeROI_NullFaceMesh_ReturnsFalse) {
    BeautyROI roi;

    bool result = BeautyROIManager::computeROI(
        nullptr, 478, 1920, 1080, config_, roi);

    EXPECT_FALSE(result);
    EXPECT_FALSE(roi.valid);
}

TEST_F(BeautyROIManagerTest, ComputeROI_InsufficientLandmarks_ReturnsFalse) {
    BeautyROI roi;

    // 478개 미만
    bool result = BeautyROIManager::computeROI(
        face_mesh_, 100, 1920, 1080, config_, roi);

    EXPECT_FALSE(result);
    EXPECT_FALSE(roi.valid);
}

TEST_F(BeautyROIManagerTest, ComputeROI_InvalidFrameSize_ReturnsFalse) {
    BeautyROI roi;

    EXPECT_FALSE(BeautyROIManager::computeROI(
        face_mesh_, 478, 0, 1080, config_, roi));

    EXPECT_FALSE(BeautyROIManager::computeROI(
        face_mesh_, 478, 1920, 0, config_, roi));

    EXPECT_FALSE(BeautyROIManager::computeROI(
        face_mesh_, 478, -1, 1080, config_, roi));
}

TEST_F(BeautyROIManagerTest, ComputeROI_MaskSizeCapped) {
    BeautyROI roi;

    BeautyROIManager::computeROI(face_mesh_, 478, 1920, 1080, config_, roi);

    // 마스크 크기는 최대 256x256
    EXPECT_LE(roi.mask_width, 256);
    EXPECT_LE(roi.mask_height, 256);

    // 최소 64x64
    EXPECT_GE(roi.mask_width, 64);
    EXPECT_GE(roi.mask_height, 64);
}

//=============================================================================
// 마스크 생성 테스트
//=============================================================================

TEST_F(BeautyROIManagerTest, SkinMask_CoversExpectedArea) {
    std::vector<uint8_t> skin_mask;

    BeautyROIManager::createSkinMask(face_mesh_, 256, 256, 1920, 1080, skin_mask);

    EXPECT_EQ(skin_mask.size(), 256u * 256u);

    // 얼굴 영역에서 일정 비율 이상이 마스킹되어야 함
    int filled_count = static_cast<int>(std::count_if(
        skin_mask.begin(), skin_mask.end(),
        [](uint8_t v) { return v > 128; }));
    float fill_ratio = static_cast<float>(filled_count) / skin_mask.size();

    // 테스트 얼굴은 중앙에 위치하므로 20~80% 사이여야 함
    EXPECT_GT(fill_ratio, 0.1f);
    EXPECT_LT(fill_ratio, 0.9f);
}

TEST_F(BeautyROIManagerTest, EyeProtectionMask_MarksEyeRegions) {
    std::vector<uint8_t> eye_mask;

    BeautyROIManager::createEyeProtectionMask(
        face_mesh_, 256, 256, 1920, 1080, eye_mask);

    EXPECT_EQ(eye_mask.size(), 256u * 256u);

    // 눈 영역이 마킹되어야 함 (255)
    int marked_count = static_cast<int>(std::count_if(
        eye_mask.begin(), eye_mask.end(),
        [](uint8_t v) { return v > 200; }));

    // 눈은 작은 영역이므로 1~10% 정도
    float mark_ratio = static_cast<float>(marked_count) / eye_mask.size();
    EXPECT_GT(mark_ratio, 0.001f);
    EXPECT_LT(mark_ratio, 0.15f);
}

TEST_F(BeautyROIManagerTest, EyebrowProtectionMask_MarksEyebrowRegions) {
    std::vector<uint8_t> eyebrow_mask;

    BeautyROIManager::createEyebrowProtectionMask(
        face_mesh_, 256, 256, 1920, 1080, eyebrow_mask);

    EXPECT_EQ(eyebrow_mask.size(), 256u * 256u);

    // 눈썹 영역이 마킹되어야 함
    int marked_count = static_cast<int>(std::count_if(
        eyebrow_mask.begin(), eyebrow_mask.end(),
        [](uint8_t v) { return v > 200; }));

    // 눈썹은 작은 영역
    float mark_ratio = static_cast<float>(marked_count) / eyebrow_mask.size();
    EXPECT_GT(mark_ratio, 0.001f);
    EXPECT_LT(mark_ratio, 0.1f);
}

TEST_F(BeautyROIManagerTest, LipProtectionMask_MarksLipRegion) {
    std::vector<uint8_t> lip_mask;

    BeautyROIManager::createLipProtectionMask(
        face_mesh_, 256, 256, 1920, 1080, lip_mask);

    EXPECT_EQ(lip_mask.size(), 256u * 256u);

    // 입술 영역이 마킹되어야 함
    int marked_count = static_cast<int>(std::count_if(
        lip_mask.begin(), lip_mask.end(),
        [](uint8_t v) { return v > 200; }));

    // 입술은 작은 영역
    float mark_ratio = static_cast<float>(marked_count) / lip_mask.size();
    EXPECT_GT(mark_ratio, 0.001f);
    EXPECT_LT(mark_ratio, 0.1f);
}

//=============================================================================
// 마스크 합성 테스트
//=============================================================================

TEST(CombineMasksTest, EmptyInputs_ReturnsEmpty) {
    std::vector<uint8_t> combined;

    BeautyROIManager::combineMasks({}, {}, {}, {}, combined);

    EXPECT_TRUE(combined.empty());
}

TEST(CombineMasksTest, SkinOnly_PreservesValues) {
    std::vector<uint8_t> skin = {0, 128, 255};
    std::vector<uint8_t> empty = {};
    std::vector<uint8_t> combined;

    BeautyROIManager::combineMasks(skin, empty, empty, empty, combined);

    EXPECT_EQ(combined.size(), 3u);
    EXPECT_EQ(combined[0], 0);
    EXPECT_EQ(combined[1], 128);
    EXPECT_EQ(combined[2], 255);
}

TEST(CombineMasksTest, EyeProtection_SubtractsFromSkin) {
    std::vector<uint8_t> skin = {255, 255, 255};
    std::vector<uint8_t> eye = {0, 128, 255};
    std::vector<uint8_t> empty = {};
    std::vector<uint8_t> combined;

    BeautyROIManager::combineMasks(skin, eye, empty, empty, combined);

    // skin * (1 - eye) = 255 * (1 - 0) = 255
    EXPECT_EQ(combined[0], 255);
    // skin * (1 - eye) = 255 * (1 - 0.5) ≈ 127
    EXPECT_NEAR(combined[1], 127, 5);
    // skin * (1 - eye) = 255 * (1 - 1) = 0
    EXPECT_EQ(combined[2], 0);
}

TEST(CombineMasksTest, AllProtections_Combined) {
    std::vector<uint8_t> skin = {255, 255, 255, 255};
    std::vector<uint8_t> eye = {255, 0, 0, 0};
    std::vector<uint8_t> eyebrow = {0, 255, 0, 0};
    std::vector<uint8_t> lip = {0, 0, 255, 0};
    std::vector<uint8_t> combined;

    BeautyROIManager::combineMasks(skin, eye, eyebrow, lip, combined);

    EXPECT_EQ(combined[0], 0);   // 눈 영역
    EXPECT_EQ(combined[1], 0);   // 눈썹 영역
    EXPECT_EQ(combined[2], 0);   // 입술 영역
    EXPECT_EQ(combined[3], 255); // 피부 영역 (보호 없음)
}

//=============================================================================
// 페더링 테스트
//=============================================================================

TEST(FeatheringTest, ZeroRadius_NoChange) {
    std::vector<uint8_t> mask = {0, 255, 0, 255};
    auto original = mask;

    BeautyROIManager::applyFeathering(mask, 2, 2, 0);

    EXPECT_EQ(mask, original);
}

TEST(FeatheringTest, EmptyMask_NoChange) {
    std::vector<uint8_t> mask;

    BeautyROIManager::applyFeathering(mask, 0, 0, 15);

    EXPECT_TRUE(mask.empty());
}

TEST(FeatheringTest, ValidMask_SmoothsEdges) {
    // 256x256 마스크, 중앙에 사각형
    const int size = 256;
    std::vector<uint8_t> mask(size * size, 0);

    // 중앙 64-192 영역을 255로 채움
    for (int y = 64; y < 192; y++) {
        for (int x = 64; x < 192; x++) {
            mask[y * size + x] = 255;
        }
    }

    BeautyROIManager::applyFeathering(mask, size, size, 15);

    // 경계에서 중간값이 있어야 함 (sharp edge가 아님)
    int edge_x = 64, edge_y = 128;
    uint8_t edge_value = mask[edge_y * size + edge_x];

    // 페더링 후 경계 값은 0과 255 사이
    EXPECT_GT(edge_value, 0);
    EXPECT_LT(edge_value, 255);
}

//=============================================================================
// 보호 영역 옵션 테스트
//=============================================================================

TEST_F(BeautyROIManagerTest, ProtectEyesDisabled_EyeMaskEmpty) {
    config_.protectEyes = false;
    BeautyROI roi;

    BeautyROIManager::computeROI(face_mesh_, 478, 1920, 1080, config_, roi);

    // 눈 보호가 비활성화되면 모든 값이 0
    bool all_zero = std::all_of(
        roi.eye_protect_mask.begin(), roi.eye_protect_mask.end(),
        [](uint8_t v) { return v == 0; });

    EXPECT_TRUE(all_zero);
}

TEST_F(BeautyROIManagerTest, ProtectLipsDisabled_LipMaskEmpty) {
    config_.protectLips = false;
    BeautyROI roi;

    BeautyROIManager::computeROI(face_mesh_, 478, 1920, 1080, config_, roi);

    // 입술 보호가 비활성화되면 모든 값이 0
    bool all_zero = std::all_of(
        roi.lip_protect_mask.begin(), roi.lip_protect_mask.end(),
        [](uint8_t v) { return v == 0; });

    EXPECT_TRUE(all_zero);
}

TEST_F(BeautyROIManagerTest, EyebrowAlwaysProtected) {
    // 눈썹은 항상 보호됨 (config와 무관)
    config_.protectEyes = false;
    config_.protectLips = false;
    BeautyROI roi;

    BeautyROIManager::computeROI(face_mesh_, 478, 1920, 1080, config_, roi);

    // 눈썹 마스크는 일부 영역이 마킹되어 있음
    bool has_marked = std::any_of(
        roi.eyebrow_protect_mask.begin(), roi.eyebrow_protect_mask.end(),
        [](uint8_t v) { return v > 200; });

    EXPECT_TRUE(has_marked);
}

//=============================================================================
// 상수 확인 테스트
//=============================================================================

TEST(BeautyROIManagerConstantsTest, LandmarkCounts) {
    EXPECT_EQ(BeautyROIManager::FACE_MESH_LANDMARK_COUNT, 478);
    EXPECT_EQ(BeautyROIManager::FACE_OVAL_COUNT, 36);
    EXPECT_EQ(BeautyROIManager::LEFT_EYE_COUNT, 16);
    EXPECT_EQ(BeautyROIManager::RIGHT_EYE_COUNT, 16);
    EXPECT_EQ(BeautyROIManager::LIPS_COUNT, 22);
    EXPECT_EQ(BeautyROIManager::LEFT_EYEBROW_COUNT, 8);
    EXPECT_EQ(BeautyROIManager::RIGHT_EYEBROW_COUNT, 8);
    EXPECT_EQ(BeautyROIManager::LIP_OUTER_COUNT, 20);
}

//=============================================================================
// P2-W2-01: 새로 추가된 메서드 테스트
//=============================================================================

// ProtectionMasks 테스트
TEST(ProtectionMasksTest, DefaultConstruction) {
    ProtectionMasks masks;

    EXPECT_TRUE(masks.left_eye.empty());
    EXPECT_TRUE(masks.right_eye.empty());
    EXPECT_TRUE(masks.lips.empty());
    EXPECT_TRUE(masks.combined.empty());
    EXPECT_EQ(masks.width, 0);
    EXPECT_EQ(masks.height, 0);
    EXPECT_FALSE(masks.valid);
}

// createProtectionMasks 테스트
TEST_F(BeautyROIManagerTest, CreateProtectionMasks_ValidInput_ReturnsTrue) {
    ProtectionMasks masks;

    bool result = BeautyROIManager::createProtectionMasks(
        face_mesh_, 256, 256, config_, masks);

    EXPECT_TRUE(result);
    EXPECT_TRUE(masks.valid);
    EXPECT_EQ(masks.width, 256);
    EXPECT_EQ(masks.height, 256);
}

TEST_F(BeautyROIManagerTest, CreateProtectionMasks_ProtectEyes_HasEyeMasks) {
    config_.protectEyes = true;
    config_.protectLips = false;
    ProtectionMasks masks;

    BeautyROIManager::createProtectionMasks(face_mesh_, 256, 256, config_, masks);

    EXPECT_FALSE(masks.left_eye.empty());
    EXPECT_FALSE(masks.right_eye.empty());
    EXPECT_TRUE(masks.lips.empty());

    // combined 마스크에 눈 영역이 포함됨
    int non_zero = static_cast<int>(std::count_if(
        masks.combined.begin(), masks.combined.end(),
        [](uint8_t v) { return v > 0; }));
    EXPECT_GT(non_zero, 0);
}

TEST_F(BeautyROIManagerTest, CreateProtectionMasks_ProtectLips_HasLipMask) {
    config_.protectEyes = false;
    config_.protectLips = true;
    ProtectionMasks masks;

    BeautyROIManager::createProtectionMasks(face_mesh_, 256, 256, config_, masks);

    EXPECT_TRUE(masks.left_eye.empty());
    EXPECT_TRUE(masks.right_eye.empty());
    EXPECT_FALSE(masks.lips.empty());

    int non_zero = static_cast<int>(std::count_if(
        masks.combined.begin(), masks.combined.end(),
        [](uint8_t v) { return v > 0; }));
    EXPECT_GT(non_zero, 0);
}

TEST_F(BeautyROIManagerTest, CreateProtectionMasks_NullFaceMesh_ReturnsFalse) {
    ProtectionMasks masks;

    bool result = BeautyROIManager::createProtectionMasks(
        nullptr, 256, 256, config_, masks);

    EXPECT_FALSE(result);
    EXPECT_FALSE(masks.valid);
}

TEST_F(BeautyROIManagerTest, CreateProtectionMasks_InvalidSize_ReturnsFalse) {
    ProtectionMasks masks;

    EXPECT_FALSE(BeautyROIManager::createProtectionMasks(
        face_mesh_, 0, 256, config_, masks));

    EXPECT_FALSE(BeautyROIManager::createProtectionMasks(
        face_mesh_, 256, 0, config_, masks));

    EXPECT_FALSE(BeautyROIManager::createProtectionMasks(
        face_mesh_, -1, 256, config_, masks));
}

// createEyeMasks 테스트
TEST_F(BeautyROIManagerTest, CreateEyeMasks_GeneratesEllipseMasks) {
    std::vector<uint8_t> left, right;

    BeautyROIManager::createEyeMasks(face_mesh_, 256, 256, 1.3f, left, right);

    EXPECT_EQ(left.size(), 256u * 256u);
    EXPECT_EQ(right.size(), 256u * 256u);

    // 눈 영역이 마킹되어 있어야 함
    int left_marked = static_cast<int>(std::count_if(
        left.begin(), left.end(), [](uint8_t v) { return v > 200; }));
    int right_marked = static_cast<int>(std::count_if(
        right.begin(), right.end(), [](uint8_t v) { return v > 200; }));

    EXPECT_GT(left_marked, 0);
    EXPECT_GT(right_marked, 0);
}

TEST_F(BeautyROIManagerTest, CreateEyeMasks_ExpansionRatioWorks) {
    std::vector<uint8_t> small_left, small_right;
    std::vector<uint8_t> large_left, large_right;

    BeautyROIManager::createEyeMasks(face_mesh_, 256, 256, 1.0f, small_left, small_right);
    BeautyROIManager::createEyeMasks(face_mesh_, 256, 256, 1.5f, large_left, large_right);

    // 확장 비율이 클수록 더 많은 픽셀이 마킹됨
    int small_count = static_cast<int>(std::count_if(
        small_left.begin(), small_left.end(), [](uint8_t v) { return v > 200; }));
    int large_count = static_cast<int>(std::count_if(
        large_left.begin(), large_left.end(), [](uint8_t v) { return v > 200; }));

    EXPECT_GT(large_count, small_count);
}

// createLipMask 테스트
TEST_F(BeautyROIManagerTest, CreateLipMask_GeneratesEllipseMask) {
    std::vector<uint8_t> lips;

    BeautyROIManager::createLipMask(face_mesh_, 256, 256, 1.2f, lips);

    EXPECT_EQ(lips.size(), 256u * 256u);

    int marked = static_cast<int>(std::count_if(
        lips.begin(), lips.end(), [](uint8_t v) { return v > 200; }));

    EXPECT_GT(marked, 0);
    // 입술 영역은 전체의 작은 부분
    float ratio = static_cast<float>(marked) / lips.size();
    EXPECT_LT(ratio, 0.15f);
}

#ifdef IRIS_SDK_HAS_OPENCV
//=============================================================================
// OpenCV 전용 테스트
//=============================================================================

TEST_F(BeautyROIManagerTest, ExtractROIRegion_ValidInput_ExtractsCorrectly) {
    // 테스트용 프레임 생성
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(100, 100, 100));

    BeautyROI roi;
    BeautyROIManager::computeROI(face_mesh_, 478, 640, 480, config_, roi);

    cv::Mat roi_region;
    cv::Rect actual_rect;

    bool result = BeautyROIManager::extractROIRegion(
        frame, roi, roi_region, actual_rect, 10);

    EXPECT_TRUE(result);
    EXPECT_FALSE(roi_region.empty());
    EXPECT_GT(actual_rect.width, 0);
    EXPECT_GT(actual_rect.height, 0);
}

TEST_F(BeautyROIManagerTest, ExtractROIRegion_EmptyFrame_ReturnsFalse) {
    cv::Mat empty_frame;
    BeautyROI roi;
    roi.valid = true;

    cv::Mat roi_region;
    cv::Rect actual_rect;

    bool result = BeautyROIManager::extractROIRegion(
        empty_frame, roi, roi_region, actual_rect, 10);

    EXPECT_FALSE(result);
}

TEST_F(BeautyROIManagerTest, ExtractROIRegion_InvalidROI_ReturnsFalse) {
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(100, 100, 100));
    BeautyROI roi;  // valid = false

    cv::Mat roi_region;
    cv::Rect actual_rect;

    bool result = BeautyROIManager::extractROIRegion(
        frame, roi, roi_region, actual_rect, 10);

    EXPECT_FALSE(result);
}

TEST_F(BeautyROIManagerTest, ExtractROIRegion_PaddingApplied) {
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(100, 100, 100));

    BeautyROI roi;
    roi.valid = true;
    roi.face_rect.x = 0.25f;  // 160
    roi.face_rect.y = 0.25f;  // 120
    roi.face_rect.width = 0.5f;  // 320
    roi.face_rect.height = 0.5f;  // 240

    cv::Mat roi_region;
    cv::Rect actual_rect;
    int padding = 20;

    BeautyROIManager::extractROIRegion(frame, roi, roi_region, actual_rect, padding);

    // 패딩 적용 확인
    EXPECT_LE(actual_rect.x, 160 - padding + 1);
    EXPECT_LE(actual_rect.y, 120 - padding + 1);
}

TEST_F(BeautyROIManagerTest, ApplyROIRegion_WithFeatherMask_Blends) {
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::Rect actual_rect(100, 100, 200, 200);

    // 처리된 ROI (밝은 색)
    cv::Mat roi_region(200, 200, CV_8UC3, cv::Scalar(200, 200, 200));

    // 페더 마스크 (중앙은 255, 가장자리는 0)
    cv::Mat feather_mask(200, 200, CV_8UC1, cv::Scalar(0));
    cv::circle(feather_mask, cv::Point(100, 100), 50, cv::Scalar(255), -1);
    cv::GaussianBlur(feather_mask, feather_mask, cv::Size(31, 31), 0);

    cv::Mat original = frame.clone();

    BeautyROIManager::applyROIRegion(frame, roi_region, actual_rect, feather_mask);

    // 중앙은 변경됨
    cv::Vec3b center_val = frame.at<cv::Vec3b>(200, 200);
    EXPECT_GT(center_val[0], 100);  // 블렌딩된 값

    // 마스크 외부는 원본 유지
    cv::Vec3b edge_val = frame.at<cv::Vec3b>(100, 100);
    EXPECT_EQ(edge_val[0], 100);
}

TEST_F(BeautyROIManagerTest, ApplyROIRegion_EmptyMask_DirectCopy) {
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::Rect actual_rect(100, 100, 200, 200);

    cv::Mat roi_region(200, 200, CV_8UC3, cv::Scalar(200, 200, 200));
    cv::Mat empty_mask;

    BeautyROIManager::applyROIRegion(frame, roi_region, actual_rect, empty_mask);

    // ROI 영역이 직접 복사됨
    cv::Vec3b val = frame.at<cv::Vec3b>(150, 150);
    EXPECT_EQ(val[0], 200);
    EXPECT_EQ(val[1], 200);
    EXPECT_EQ(val[2], 200);
}

TEST_F(BeautyROIManagerTest, CreateFeatherMask_ValidInput_ReturnsValidMask) {
    BeautyROI roi;
    BeautyROIManager::computeROI(face_mesh_, 478, 640, 480, config_, roi);

    cv::Mat feather = BeautyROIManager::createFeatherMask(
        roi, 15, roi.combined_mask, std::vector<uint8_t>());

    EXPECT_FALSE(feather.empty());
    EXPECT_EQ(feather.rows, roi.mask_height);
    EXPECT_EQ(feather.cols, roi.mask_width);
}

TEST_F(BeautyROIManagerTest, CreateFeatherMask_WithProtection_ExcludesRegions) {
    BeautyROI roi;
    BeautyROIManager::computeROI(face_mesh_, 478, 640, 480, config_, roi);

    // 보호 마스크가 있는 경우
    cv::Mat feather_with = BeautyROIManager::createFeatherMask(
        roi, 15, roi.skin_mask, roi.eye_protect_mask);

    // 보호 마스크가 없는 경우
    cv::Mat feather_without = BeautyROIManager::createFeatherMask(
        roi, 15, roi.skin_mask, std::vector<uint8_t>());

    // 보호 영역이 있으면 non-zero 픽셀이 더 적어야 함
    int count_with = cv::countNonZero(feather_with);
    int count_without = cv::countNonZero(feather_without);

    EXPECT_LE(count_with, count_without);
}

TEST_F(BeautyROIManagerTest, CreateFeatherMask_EmptySkin_ReturnsEmpty) {
    BeautyROI roi;
    roi.mask_width = 0;
    roi.mask_height = 0;

    cv::Mat feather = BeautyROIManager::createFeatherMask(
        roi, 15, std::vector<uint8_t>(), std::vector<uint8_t>());

    EXPECT_TRUE(feather.empty());
}

#endif // IRIS_SDK_HAS_OPENCV

} // namespace testing
} // namespace iris_sdk
