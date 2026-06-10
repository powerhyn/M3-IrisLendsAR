/**
 * @file skin_mask_geometry.h
 * @brief 피부 마스크 폴리곤용 랜드마크 인덱스 + 이마 확장 순수 함수 (P8-W1)
 *
 * LensSimulator에서 검증 완료된 피부 보정(skin smoothing) 이식 스펙.
 * GL에 의존하지 않는 순수 기하 연산만 모아 단위 테스트 가능하게 분리한다.
 *
 * 출처: LensSimulator `internal/LandmarkIndices.kt` + `internal/math/BeautyGeometry.kt`
 * (S23+ 실기기 검증 완료, docs/skin-smoothing-handoff-from-lenssimulator.md 참조)
 */

#ifndef IRIS_SDK_SKIN_MASK_GEOMETRY_H
#define IRIS_SDK_SKIN_MASK_GEOMETRY_H

#include <array>
#include <cstddef>

namespace iris_sdk {
namespace skin_mask {

/// MediaPipe FaceLandmarker 478점 규약 (LandmarkIndices.kt 그대로)
constexpr int kFaceOvalCount = 36;
constexpr int kBrowCount = 10;     ///< 한쪽 눈썹
constexpr int kLipsCount = 20;
constexpr int kEyeContourCount = 16; ///< 한쪽 눈 윤곽

/**
 * 얼굴 외곽(FACEMESH_FACE_OVAL) 36점 — 순서 체인:
 * 이마(10) → 피험자 좌측면 → 턱끝(152) → 피험자 우측면 → 닫힘.
 */
constexpr std::array<int, kFaceOvalCount> kFaceOval = {
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
};

/// FACE_OVAL 배열 내 주요 포인트 위치 (랜드마크 번호가 아니라 배열 인덱스)
constexpr int kOvalForehead = 0;     ///< 10 — 얼굴 세로축 상단
constexpr int kOvalCheekLeft = 8;    ///< 454 — 피험자 좌측 광대 (얼굴 폭 측정)
constexpr int kOvalChin = 18;        ///< 152 — 얼굴 세로축 하단
constexpr int kOvalCheekRight = 28;  ///< 234 — 피험자 우측 광대

/// 눈썹 폴리곤 (피부 마스크 제외 영역) — 아래 폴리라인 + 위 폴리라인 역순
constexpr std::array<int, kBrowCount> kRightBrow = {46, 53, 52, 65, 55, 107, 66, 105, 63, 70};
constexpr std::array<int, kBrowCount> kLeftBrow = {276, 283, 282, 295, 285, 336, 296, 334, 293, 300};

/// 입술 외곽 링 20점 (FACEMESH_LIPS 외곽 체인)
constexpr std::array<int, kLipsCount> kLipsOuter = {
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
};

/// 피험자 우안 눈 윤곽 16점
constexpr std::array<int, kEyeContourCount> kRightEyeContour = {
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
};
/// 피험자 좌안 눈 윤곽 16점
constexpr std::array<int, kEyeContourCount> kLeftEyeContour = {
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
};

/// 이마 확장 비율 — 마스크 전용 (P8-W1 §5 확정)
constexpr float kForeheadExtend = 0.35f;

/**
 * @brief 피부 마스크용 얼굴 윤곽 이마 확장 (in-place, 픽셀 좌표)
 *
 * MediaPipe 얼굴 메시 상단 경계(랜드마크 10)는 헤어라인보다 한참 아래(이마 중간)라
 * 마스크가 이마 상부를 덮지 못한다. 광대 라인(454·234 중점)을 피벗으로, 그보다 위쪽
 * 점들을 얼굴 세로축(턱152→이마10) 방향으로 factor 비율만큼 늘린다. 가로 폭은 늘리지
 * 않으며(축 방향 변위만), 턱/볼 라인은 변하지 않는다. 헤어라인 침범분은 컴포지트
 * 셰이더의 에지 가드가 보호한다.
 *
 * @param ovalPx FACE_OVAL 36점의 픽셀 좌표 [x0,y0,...] (최소 72 floats)
 * @param count  점 개수 (= 36)
 * @param factor 확장 비율 (0이면 no-op). P8-W1 기본 0.35
 *
 * 출처: LensSimulator BeautyGeometry.kt#extendForehead (단위 테스트 동반).
 * 주의: 마스크 전용 변형 — 다른 용도(워프 제어점 등)는 원본 윤곽을 쓸 것.
 */
void extendForehead(float* ovalPx, std::size_t count, float factor);

}  // namespace skin_mask
}  // namespace iris_sdk

#endif  // IRIS_SDK_SKIN_MASK_GEOMETRY_H
