/**
 * @file beauty_roi_manager.h
 * @brief Face Mesh 기반 ROI(Region of Interest) 관리자
 *
 * MediaPipe Face Mesh의 478개 랜드마크를 활용하여
 * 뷰티 필터 적용을 위한 마스크 영역을 생성합니다.
 */

#ifndef IRIS_SDK_BEAUTY_ROI_MANAGER_H
#define IRIS_SDK_BEAUTY_ROI_MANAGER_H

#include "types.h"
#include "beauty_filter.h"
#include <vector>
#include <cstdint>
#include <algorithm>

#ifdef IRIS_SDK_HAS_OPENCV
#include <opencv2/core.hpp>
#endif

namespace iris_sdk {

/**
 * @brief 보호 영역 마스크 구조체
 *
 * 눈/입술 등 필터를 적용하지 않을 영역의 마스크들
 */
struct ProtectionMasks {
    std::vector<uint8_t> left_eye;   ///< 왼쪽 눈 보호 마스크
    std::vector<uint8_t> right_eye;  ///< 오른쪽 눈 보호 마스크
    std::vector<uint8_t> lips;       ///< 입술 보호 마스크
    std::vector<uint8_t> combined;   ///< 통합 보호 마스크

    int width = 0;   ///< 마스크 너비
    int height = 0;  ///< 마스크 높이
    bool valid = false;
};

/**
 * @brief ROI 데이터 구조체
 *
 * Face Mesh 기반으로 계산된 얼굴 영역 정보 및 마스크
 */
struct BeautyROI {
    /// 얼굴 바운딩 박스 (원본 프레임 좌표)
    Rect face_rect;

    /// 마스크 (8-bit, 0~255)
    /// 255 = 완전 적용, 0 = 미적용
    std::vector<uint8_t> skin_mask;            ///< 피부 영역 마스크
    std::vector<uint8_t> eye_protect_mask;     ///< 눈 보호 마스크 (반전)
    std::vector<uint8_t> eyebrow_protect_mask; ///< 눈썹 보호 마스크 (반전)
    std::vector<uint8_t> lip_protect_mask;     ///< 입술 보호 마스크 (반전)
    std::vector<uint8_t> combined_mask;        ///< 최종 합성 마스크

    /// 마스크 크기
    int mask_width = 0;
    int mask_height = 0;

    /// 유효성
    bool valid = false;

    /// 타임스탬프 (캐싱용)
    int64_t timestamp_ms = 0;

    /// 유효성 확인
    bool isValid() const { return valid && !combined_mask.empty(); }

    /// 무효화
    void invalidate() {
        valid = false;
        combined_mask.clear();
    }
};

/**
 * @brief 정규화된 face_rect로부터 픽셀 좌표 ROI를 계산 (20% 마진 포함)
 *
 * sdk_api_v2.cpp(CPU/GPU), gpu_beauty_backend.cpp(applyTextureId)에서
 * 동일하게 사용되는 ROI 확장 로직의 공통 헬퍼.
 *
 * @param norm_x 정규화된 face rect x (0.0~1.0)
 * @param norm_y 정규화된 face rect y (0.0~1.0)
 * @param norm_w 정규화된 face rect width (0.0~1.0)
 * @param norm_h 정규화된 face rect height (0.0~1.0)
 * @param frame_width 프레임 너비 (px)
 * @param frame_height 프레임 높이 (px)
 * @return 확장된 face_rect (픽셀 좌표, Rect)
 */
inline Rect computeExpandedFaceRect(float norm_x, float norm_y,
                                    float norm_w, float norm_h,
                                    int frame_width, int frame_height) {
    int face_x = static_cast<int>(norm_x * frame_width);
    int face_y = static_cast<int>(norm_y * frame_height);
    int face_w = static_cast<int>(norm_w * frame_width);
    int face_h = static_cast<int>(norm_h * frame_height);

    // 20% 마진 확장
    int margin_x = face_w / 5;
    int margin_y = face_h / 5;
    face_x = std::max(0, face_x - margin_x);
    face_y = std::max(0, face_y - margin_y);
    face_w = std::min(frame_width - face_x, face_w + 2 * margin_x);
    face_h = std::min(frame_height - face_y, face_h + 2 * margin_y);

    return Rect{
        static_cast<float>(face_x),
        static_cast<float>(face_y),
        static_cast<float>(face_w),
        static_cast<float>(face_h)
    };
}

/**
 * @brief Face Mesh 기반 ROI 관리자
 *
 * Face Mesh 478개 랜드마크로부터 피부/눈/눈썹/입술 영역을 추출하고
 * 뷰티 필터 적용을 위한 마스크를 생성합니다.
 */
class BeautyROIManager {
public:
    BeautyROIManager() = default;

    /**
     * @brief Face Mesh로부터 ROI 계산
     *
     * @param face_mesh 478개 랜드마크 (IrisResult.face_mesh)
     * @param landmark_count 랜드마크 개수 (보통 478)
     * @param frame_width 프레임 너비
     * @param frame_height 프레임 높이
     * @param config 필터 설정 (보호 영역 옵션)
     * @param out_roi 출력 ROI
     * @return 성공 여부
     */
    static bool computeROI(
        const IrisLandmark* face_mesh,
        int landmark_count,
        int frame_width, int frame_height,
        const BeautyFilterConfigV2& config,
        BeautyROI& out_roi
    );

    /**
     * @brief 피부 영역 마스크 생성
     *
     * 얼굴 윤곽 내부를 채우는 마스크 생성
     * OpenCV fillPoly 기반 래스터화
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

#ifdef IRIS_SDK_HAS_OPENCV
    //=========================================================================
    // P2-W2-01: ROI 기반 처리 및 페더링 통합 (OpenCV 전용)
    //=========================================================================

    /**
     * @brief ROI 영역 추출 (패딩 포함)
     *
     * @param full_frame 전체 프레임 (cv::Mat)
     * @param roi ROI 정보
     * @param out_roi_region 출력 ROI 영역 (cv::Mat)
     * @param out_actual_rect 실제 추출된 영역 (cv::Rect)
     * @param padding 패딩 픽셀
     * @return 성공 여부
     */
    static bool extractROIRegion(
        const cv::Mat& full_frame,
        const BeautyROI& roi,
        cv::Mat& out_roi_region,
        cv::Rect& out_actual_rect,
        int padding = 20
    );

    /**
     * @brief ROI 영역 합성 (페더링 마스크 적용)
     *
     * @param full_frame 전체 프레임 (in-place 수정)
     * @param roi_region 처리된 ROI 영역
     * @param actual_rect ROI가 위치하는 영역
     * @param feather_mask 페더링 마스크 (선택적)
     */
    static void applyROIRegion(
        cv::Mat& full_frame,
        const cv::Mat& roi_region,
        const cv::Rect& actual_rect,
        const cv::Mat& feather_mask
    );

    /**
     * @brief 소프트 페더링 마스크 생성
     *
     * @param roi ROI 정보
     * @param feather_radius 페더링 반경
     * @param skin_mask 피부 마스크
     * @param protection_mask 보호 마스크 (눈/입술)
     * @return 페더링이 적용된 마스크 (cv::Mat)
     */
    static cv::Mat createFeatherMask(
        const BeautyROI& roi,
        int feather_radius,
        const std::vector<uint8_t>& skin_mask,
        const std::vector<uint8_t>& protection_mask
    );
#endif // IRIS_SDK_HAS_OPENCV

    /**
     * @brief 보호 영역 마스크 생성
     *
     * @param face_mesh 478개 랜드마크
     * @param mask_width 마스크 너비
     * @param mask_height 마스크 높이
     * @param config 필터 설정 (protectEyes, protectLips)
     * @param out_masks 출력 마스크들
     * @return 성공 여부
     */
    static bool createProtectionMasks(
        const IrisLandmark* face_mesh,
        int mask_width, int mask_height,
        const BeautyFilterConfigV2& config,
        ProtectionMasks& out_masks
    );

    /**
     * @brief 눈 영역 타원 마스크 생성 (확장 비율 적용)
     *
     * @param face_mesh 478개 랜드마크
     * @param mask_width 마스크 너비
     * @param mask_height 마스크 높이
     * @param expansion_ratio 확장 비율 (1.2 = 20% 확장)
     * @param out_left 왼쪽 눈 마스크
     * @param out_right 오른쪽 눈 마스크
     */
    static void createEyeMasks(
        const IrisLandmark* face_mesh,
        int mask_width, int mask_height,
        float expansion_ratio,
        std::vector<uint8_t>& out_left,
        std::vector<uint8_t>& out_right
    );

    /**
     * @brief 입술 영역 마스크 생성
     *
     * @param face_mesh 478개 랜드마크
     * @param mask_width 마스크 너비
     * @param mask_height 마스크 높이
     * @param expansion_ratio 확장 비율
     * @param out_lips 입술 마스크
     */
    static void createLipMask(
        const IrisLandmark* face_mesh,
        int mask_width, int mask_height,
        float expansion_ratio,
        std::vector<uint8_t>& out_lips
    );

    // 랜드마크 인덱스 상수
    static constexpr int FACE_MESH_LANDMARK_COUNT = 478;
    static constexpr int FACE_OVAL_COUNT = 36;
    static constexpr int LEFT_EYE_COUNT = 16;
    static constexpr int RIGHT_EYE_COUNT = 16;
    static constexpr int LIPS_COUNT = 22;
    static constexpr int LEFT_EYEBROW_COUNT = 8;
    static constexpr int RIGHT_EYEBROW_COUNT = 8;
    static constexpr int LIP_OUTER_COUNT = 20;  ///< 외곽 입술 랜드마크 수

private:
    // Face Mesh 랜드마크 인덱스 테이블
    static const int FACE_OVAL_INDICES[FACE_OVAL_COUNT];
    static const int LEFT_EYE_INDICES[LEFT_EYE_COUNT];
    static const int RIGHT_EYE_INDICES[RIGHT_EYE_COUNT];
    static const int LIPS_INDICES[LIPS_COUNT];
    static const int LEFT_EYEBROW_INDICES[LEFT_EYEBROW_COUNT];
    static const int RIGHT_EYEBROW_INDICES[RIGHT_EYEBROW_COUNT];
    static const int LIP_OUTER_INDICES[LIP_OUTER_COUNT];  ///< 외곽 입술 인덱스

    /**
     * @brief 랜드마크 좌표를 마스크 좌표로 변환
     * @param lm 랜드마크 (정규화 0~1)
     * @param mask_width 마스크 너비
     * @param mask_height 마스크 높이
     * @param face_rect 얼굴 영역
     * @return 마스크 좌표
     */
    static std::pair<int, int> landmarkToMaskCoord(
        const IrisLandmark& lm,
        int mask_width, int mask_height,
        const Rect& face_rect
    );
};

} // namespace iris_sdk

#endif // IRIS_SDK_BEAUTY_ROI_MANAGER_H
