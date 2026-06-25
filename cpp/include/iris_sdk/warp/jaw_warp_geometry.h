/**
 * @file jaw_warp_geometry.h
 * @brief 턱 V라인 슬림 워프용 CPU 제어점/파라미터 산출 (P8-W4-A)
 *
 * LensSimulator에서 S23+ 실기기 검증 완료된 "턱선 V라인 워프"의 코어 기하 계산.
 * 출력 픽셀 p의 소스 좌표를 인버스 워프로 구하는 비정규 RBF 모델:
 *
 *   src(p) = p − Σᵢ dᵢ · exp(−|p − cᵢ|² / 2σ²)
 *
 * (cᵢ = 얼굴 측면 윤곽 제어점, dᵢ = 세로축 방향 "안쪽" 변위, 전부 디스플레이 픽셀 공간.)
 *
 * 본 모듈은 그 중 **CPU 측 제어점·변위·σ·바운딩박스 산출**과 검증용 CPU RBF 평가만
 * 담당한다. GPU 셰이더/백엔드 결선은 후속 W4-B의 몫이다. GL/OpenCV 의존이 전혀 없는
 * 순수 기하 모듈이라 데스크톱에서 단위 테스트 가능하다.
 *
 * 설계 불변식 (위반 시 렌즈가 어긋난다):
 *   - **전부 픽셀 공간 계산**(정규화 비등방 σ 회피, audit #1c/#4).
 *   - **face_width = 유클리드 픽셀거리**(454↔234) — roll-robust(audit #3 수정).
 *   - **눈 높이 변위 ≈ 0**: 워프는 렌즈 합성보다 앞 단계라 눈 높이 픽셀이 움직이면
 *     홍채와 렌즈가 어긋난다. 광대 정점(454/234)·턱끝(152)을 제어점에서 제외하고
 *     볼 상부 테이퍼를 0.2로 낮춰 이 조건을 지킨다(단위 테스트 게이트).
 *
 * 출처: docs/lenssim-handoff/jaw-vline-warp-handoff-from-lenssimulator.md (§1 파라미터,
 *       §3/§4 GLSL, §5 테스트). 정본 코드 BeautyGeometry.kt#computeJawWarp.
 *
 * 🔴 주의: grid_mesh.{h,cpp} / face_warp_controller.{h,cpp}는 미사용 dead substrate이며
 *         본 모듈과 무관하다(완전 신규). 그쪽은 건드리지 않는다.
 */

#ifndef IRIS_SDK_WARP_JAW_WARP_GEOMETRY_H
#define IRIS_SDK_WARP_JAW_WARP_GEOMETRY_H

#include "../types.h"  // iris_sdk::IrisLandmark (정규화 .x/.y)

namespace iris_sdk {
namespace jaw_warp {

// ============================================================
// 확정 파라미터 (핸드오프 §1 — LensSim S23+ 검증, 재논의 금지)
// ============================================================

/// 제어점 한쪽(측면) 개수 — 볼 상부→턱끝 옆 7점.
constexpr int kSideControlPoints = 7;

/// 전체 제어점 수 (좌 7 + 우 7 = 14).
constexpr int kControlPointCount = kSideControlPoints * 2;

/// 피험자 좌측 측면 7점 (FACE_OVAL 체인, 볼 상부 → 턱끝 옆 순서).
constexpr int kLeftSideControl[kSideControlPoints] = {323, 361, 288, 397, 365, 379, 378};
/// 피험자 우측 측면 7점 (좌측과 대칭, 동일 순서).
constexpr int kRightSideControl[kSideControlPoints] = {93, 132, 58, 172, 136, 150, 149};

/// 세로축 상단(이마)·하단(턱끝) 랜드마크 — 세로축 정의용(제어점 아님).
constexpr int kLandmarkForehead = 10;
constexpr int kLandmarkChin = 152;
/// 얼굴 폭(광대 간) 측정용 랜드마크 — 유클리드 거리(제어점 아님).
constexpr int kLandmarkCheekLeft = 454;
constexpr int kLandmarkCheekRight = 234;

/// 테이퍼 (볼 상부→턱끝 옆, 좌/우 각각 인덱스 0~6) — 하악체 1.0 최대.
constexpr float kTaper[kSideControlPoints] = {0.2f, 0.45f, 0.65f, 0.85f, 1.0f, 0.85f, 0.6f};

/// 최대 변위 = 얼굴 폭 × 이 값 × strength.
constexpr float kMaxDispRatio = 0.032f;
/// 가우시안 σ = 얼굴 폭 × 이 값.
constexpr float kSigmaRatio = 0.13f;
/// 바운딩박스 마진 = 이 값 × σ (박스 밖 가중치 ≈ e⁻⁴·⁵).
constexpr float kBoundsSigmaMargin = 3.0f;

/// 퇴화 방어 임계: 세로축 길이가 이보다 짧으면 워프 비활성.
constexpr float kMinAxisLenPx = 1.0f;
/// 퇴화 방어 임계: 얼굴 폭이 이보다 좁으면 워프 비활성.
constexpr float kMinFaceWidthPx = 8.0f;

// ============================================================
// 자료구조
// ============================================================

/**
 * @brief 턱 V라인 워프 파라미터 (셰이더 유니폼 산출 결과, 전부 디스플레이 픽셀)
 *
 * GLSL uWarp[14]/uWarpSigma/uWarpBounds에 그대로 대응한다. POD — 할당 0(배열 재사용).
 */
struct JawWarpParams {
    static constexpr int kMaxControlPoints = 14;

    float cx[kMaxControlPoints];  ///< 제어점 X (픽셀)
    float cy[kMaxControlPoints];  ///< 제어점 Y (픽셀)
    float dx[kMaxControlPoints];  ///< 변위 X (픽셀, 안쪽=세로축 방향)
    float dy[kMaxControlPoints];  ///< 변위 Y (픽셀, 안쪽=세로축 방향)
    int   count;                  ///< 활성 제어점 수 (보통 14, 비활성/퇴화 시 0)
    float sigma_px;               ///< 가우시안 σ (픽셀). 0이면 워프 비활성
    float bounds_min_x;           ///< 제어점 bbox + 3σ 마진 (픽셀)
    float bounds_min_y;
    float bounds_max_x;
    float bounds_max_y;
};

// ============================================================
// API
// ============================================================

/**
 * @brief 478점 얼굴 메시에서 턱 V라인 워프 파라미터를 픽셀 공간으로 산출
 *
 * 핸드오프 §1 알고리즘을 전부 픽셀 공간(정규화 .x/.y × image_width/height)으로 계산한다.
 *   1. face_width_px = 유클리드(454↔234) — roll-robust.
 *   2. 세로축: origin = 10(이마), axis_hat = normalize(152 − 10), axis_len_px = 길이.
 *   3. 퇴화 방어: axis_len_px < 1.0 또는 face_width_px < 8.0 → false(비활성).
 *   4. 제어점 14개 각각: v = c − origin; perp = v − (v·axis_hat)axis_hat;
 *      inward = −normalize(perp); disp = inward × (face_width_px × 0.032 × strength × taper).
 *      |perp| < 1e-6(축 위)이면 변위 0.
 *   5. σ = face_width_px × 0.13. bounds = 14점 cx/cy min/max ± 3σ.
 *
 * @param face_mesh    478 정규화(0~1) 랜드마크 (upright 공간 — ADR §7.1)
 * @param image_width  픽셀 환산용 프레임 너비 (픽셀)
 * @param image_height 픽셀 환산용 프레임 높이 (픽셀)
 * @param strength     워프 강도 0~1 (slim/chin 매핑은 호출부 책임 — 본 함수는 1개 받음)
 * @param out          [출력] 워프 파라미터. 비활성 시 count=0, sigma_px=0
 * @return true=워프 활성, false=비활성(strength≤0 또는 퇴화 입력 또는 null)
 *
 * @note out은 false 반환 시에도 count=0/sigma_px=0으로 안전하게 채워진다.
 * @warning face_mesh는 최소 478원소여야 한다(인덱스 454까지 접근). null이면 false.
 */
bool computeJawWarp(const IrisLandmark* face_mesh,
                    int image_width,
                    int image_height,
                    float strength,
                    JawWarpParams& out);

/**
 * @brief 출력 픽셀 (px,py)에서의 워프 변위 합 (CPU RBF 평가 — 테스트/검증용)
 *
 * 셰이더 §3과 동일 수식: Σᵢ dᵢ · exp(−|p − cᵢ|² / 2σ²). **비정규 RBF 합**(정규화 안 함).
 * 바운딩박스(+3σ) 밖이거나 sigma_px ≤ 0이면 (0,0)을 반환한다(early-out).
 *
 * 소스 좌표는 인버스 워프이므로 호출자가 (px − out_dx, py − out_dy)로 구한다.
 *
 * @param p       워프 파라미터 (computeJawWarp 산출)
 * @param px      평가 픽셀 X
 * @param py      평가 픽셀 Y
 * @param out_dx  [출력] 변위 X 합 (픽셀)
 * @param out_dy  [출력] 변위 Y 합 (픽셀)
 */
void evaluateWarpDisplacement(const JawWarpParams& p,
                              float px,
                              float py,
                              float& out_dx,
                              float& out_dy);

}  // namespace jaw_warp
}  // namespace iris_sdk

#endif  // IRIS_SDK_WARP_JAW_WARP_GEOMETRY_H
