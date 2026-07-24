/**
 * @file jaw_warp_geometry.h
 * @brief 얼굴 V라인 + 내부 축소 워프용 CPU 제어점/파라미터 산출 (P8-W4 / P8-W4B)
 *
 * LensSimulator에서 S23+ 실기기 검증 완료된 "턱선 V라인 워프"를 코어로,
 * "내부 축소(face-small lite)" 2노브 모델로 확장한 기하 계산.
 * 출력 픽셀 p의 소스 좌표를 인버스 워프로 구하는 비정규 RBF 모델:
 *
 *   src(p) = p − Σᵢ dᵢ · exp(−|p − cᵢ|² / 2σ²)
 *
 * (cᵢ = 얼굴 윤곽/내부 제어점, dᵢ = 세로축 방향 "안쪽" 변위, 전부 디스플레이 픽셀 공간.)
 *
 * ── 2노브 모델 ──────────────────────────────────────────────
 *   • jaw_strength (= slimFace): 하악 윤곽 V라인. 제어점 = 측면 7점/측(kTaper) +
 *     상방 실루엣 1점/측(광대 정점 454/234, kUpperTaper — V라인을 "더 위에서부터 시작").
 *   • interior_strength (= thinChin, 의미 재정의): 얼굴 내부 축소. 제어점 = 볼 중앙·
 *     볼 하부(팔자 옆)·입꼬리·콧볼 4점/측(kInteriorTaper)을 세로축 방향으로 좁혀
 *     face-small 느낌. 더 이상 "턱끝만 축소"가 아니다.
 *   두 노브는 독립 max_disp(각자 strength × 0.032 × face_width)로 스케일되며,
 *   strength ≤ 0 인 그룹은 아예 패킹에서 제외한다(조건부 패킹 → count 가변).
 *
 * 본 모듈은 **CPU 측 제어점·변위·σ·바운딩박스 산출**과 검증용 CPU RBF 평가만
 * 담당한다. GPU 셰이더/백엔드 결선은 gpu_beauty_backend.cpp 의 몫이다. GL/OpenCV
 * 의존이 전혀 없는 순수 기하 모듈이라 데스크톱에서 단위 테스트 가능하다.
 *
 * 설계 불변식 (위반 시 렌즈가 어긋난다):
 *   - **전부 픽셀 공간 계산**(정규화 비등방 σ 회피, audit #1c/#4).
 *   - **face_width = 유클리드 픽셀거리**(454↔234) — roll-robust(audit #3 수정).
 *   - **홍채/눈 영역 변위 최소화**: 실제 렌더 순서는 **렌즈 합성 → 뷰티/워프**다
 *     (CameraGLRenderer.kt:474 렌즈 → :491 뷰티, gpu_beauty_backend.cpp:1420-1435 에서
 *     워프가 skin/brightness 적용된 '렌즈 합성본' current_input 을 인버스 리샘플). 워프가
 *     렌즈 뒤라 홍채를 움직여도 그 위에 얹힌 렌즈 픽셀이 함께 워프되어 **정합은 자동 보존**된다.
 *     따라서 이 게이트의 근거는 "렌즈 어긋남 방지"가 아니라 ① **눈 영역 비등방 왜곡 방지**
 *     (눈이 세로로 찌그러지면 정합과 무관하게 눈 모양 자체가 부자연스러움) + ② **눈 고정 미학**
 *     (렌즈 SDK 제품 결정: "눈은 건드리지 않는다")이다. **홍채 중심(468/473) 변위 < 최대 변위의
 *     5% (하드 게이트)**, 눈꼬리(33/263)는 < 12% 로 완화. 게이트값은 kUpperTaper(상방 0.05)·
 *     kInteriorTaper[0](볼 중앙 0.08)로 지킨다. 최종 판정은 실기기 눈 형태/렌즈 정합.
 *     (근거 정정 이력: FMLENS 리버스 비교 감사 — 기존 "워프=렌즈 앞" 서술이 실제 렌더 순서와
 *      반대였음. 게이트 값은 눈 고정 미학상 그대로 유지하며 동작 코드 변경은 없다.)
 *   - **jaw/upper 7점(측면) + 1점(상방)·kTaper·kMaxDispRatio·kSigmaRatio 등 기존 값은
 *     S23+ 실기기 검증 고정값이라 수정 금지.** interior_strength=0 이면 jaw 경로는
 *     기존과 수치 동일(jaw 그룹 14점 cx/cy/dx/dy 무변경).
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

// ── jaw 그룹: 하악 측면 윤곽 (jaw_strength 스케일, S23+ 검증 고정) ──

/// 제어점 한쪽(측면) 개수 — 볼 상부→턱끝 옆 7점.
constexpr int kSideControlPoints = 7;

/// 피험자 좌측 측면 7점 (FACE_OVAL 체인, 볼 상부 → 턱끝 옆 순서).
constexpr int kLeftSideControl[kSideControlPoints] = {323, 361, 288, 397, 365, 379, 378};
/// 피험자 우측 측면 7점 (좌측과 대칭, 동일 순서).
constexpr int kRightSideControl[kSideControlPoints] = {93, 132, 58, 172, 136, 150, 149};

/// 테이퍼 (볼 상부→턱끝 옆, 좌/우 각각 인덱스 0~6) — 하악체 1.0 최대.
constexpr float kTaper[kSideControlPoints] = {0.2f, 0.45f, 0.65f, 0.85f, 1.0f, 0.85f, 0.6f};

// ── upper 그룹: 상방 실루엣 (jaw_strength 스케일, V라인을 더 위에서 시작) ──

/// 상방 실루엣 제어점 한쪽 개수 — 광대 정점 1점.
constexpr int kUpperPerSide = 1;
/// 피험자 좌/우 광대 정점(454/234) — face_width 측정 랜드마크와 동일 인덱스 재사용.
constexpr int kLeftUpperControl[kUpperPerSide] = {454};
constexpr int kRightUpperControl[kUpperPerSide] = {234};
/// 상방 테이퍼 — 아주 약함(눈 높이 변위 누출 방지: 홍채 5% / 눈꼬리 12% 게이트).
constexpr float kUpperTaper[kUpperPerSide] = {0.05f};

// ── interior 그룹: 얼굴 내부 축소 face-small (interior_strength 스케일) ──

/// 내부 축소 제어점 한쪽 개수 — 볼 중앙·볼 하부·입꼬리·콧볼 4점.
constexpr int kInteriorPerSide = 4;
/// 피험자 좌측 내부 4점 (순서: 볼 중앙, 볼 하부(팔자 옆), 입꼬리, 콧볼).
constexpr int kLeftInteriorControl[kInteriorPerSide] = {280, 425, 291, 358};
/// 피험자 우측 내부 4점 (좌측 대칭, 동일 순서).
constexpr int kRightInteriorControl[kInteriorPerSide] = {50, 205, 61, 129};
/// 내부 테이퍼 확정값(볼 중앙, 볼 하부, 입꼬리, 콧볼) = **프리셋 0(단일 진실)**.
/// [0]=0.08: 초기값 0.12는 풀강도에서 홍채 중심(468) 변위 5.02%로 5% 게이트 초과 →
/// 0.08로 낮춰 4.73%(margin 0.27%p) 통과. 나머지 3개는 눈에서 충분히 멀어 미조정.
/// 아래 kInteriorTaperPresets[0] 이 이 배열을 그대로 참조한다(중복 리터럴 없음).
constexpr float kInteriorTaper[kInteriorPerSide] = {0.08f, 0.35f, 0.30f, 0.20f};

/// 내부 테이퍼 프리셋 개수 (P8-W4B 벤치 임시).
constexpr int kInteriorTaperPresetCount = 4;

/// 내부 테이퍼 프리셋 테이블 (각 행 = 볼 중앙, 볼 하부, 입꼬리, 콧볼) — **벤치 임시**.
///
/// 목적: 여러 평가자가 실기기에서 내부 4점 비율 조합을 런타임 토글로 A/B 비교하기 위한
/// 임시 테이블. 다수 의견 수집 후 승자 행을 kInteriorTaper 로 고정하고 이 테이블·프리셋
/// API(iris_sdk_set_interior_taper_preset)는 제거한다. 프리셋 선택 상태는 backend 멤버가
/// 아니라 sdk 레벨 std::atomic<int>(sdk_api_v2.cpp)로 보관 → GL 컨텍스트 재생성에도 유지되며,
/// gpu_beauty_backend 가 매 프레임 읽어 computeJawWarp(..., interior_taper_preset)로 전달한다.
///
/// 행 순서: 0=기본(kInteriorTaper 참조 — 단일 진실), 1=볼 강조, 2=입·코 강조, 3=약하게.
/// ⚠️ 각 행 [0]=볼 중앙은 눈(홍채)에 가장 가까워 **전 프리셋 0.08 고정**한다(홍채 <5% 하드
///    게이트가 볼 중앙만 제약). 프리셋 간 가변은 나머지 3점(볼 하부·입꼬리·콧볼)에 한정한다.
constexpr float kInteriorTaperPresets[kInteriorTaperPresetCount][kInteriorPerSide] = {
    {kInteriorTaper[0], kInteriorTaper[1], kInteriorTaper[2], kInteriorTaper[3]},  // 0=기본(단일 진실)
    {0.08f, 0.45f, 0.25f, 0.15f},  // 1=볼 강조   (볼 하부 ↑, 입꼬리·콧볼 ↓)
    {0.08f, 0.28f, 0.38f, 0.28f},  // 2=입·코 강조 (입꼬리·콧볼 ↑, 볼 하부 ↓)
    {0.08f, 0.25f, 0.25f, 0.18f},  // 3=약하게    (볼 중앙 제외 전반 하향)
};

/// 전체 제어점 수 = (jaw 7 + upper 1 + interior 4) × 2 = 24 (조건부 패킹 시 상한).
constexpr int kControlPointCount = (kSideControlPoints + kUpperPerSide + kInteriorPerSide) * 2;

/// 세로축 상단(이마)·하단(턱끝) 랜드마크 — 세로축 정의용(제어점 아님).
constexpr int kLandmarkForehead = 10;
constexpr int kLandmarkChin = 152;
/// 얼굴 폭(광대 간) 측정용 랜드마크 — 유클리드 거리(제어점 아님).
constexpr int kLandmarkCheekLeft = 454;
constexpr int kLandmarkCheekRight = 234;

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
 * @brief V라인 + 내부 축소 워프 파라미터 (셰이더 유니폼 산출 결과, 전부 디스플레이 픽셀)
 *
 * GLSL uWarp[24]/uWarpSigma/uWarpBounds에 그대로 대응한다. POD — 할당 0(배열 재사용).
 * 조건부 패킹이라 count는 8(interior만)·16(jaw+upper만)·24(둘 다) 중 하나(비활성 시 0).
 */
struct JawWarpParams {
    static constexpr int kMaxControlPoints = 24;

    float cx[kMaxControlPoints];  ///< 제어점 X (픽셀)
    float cy[kMaxControlPoints];  ///< 제어점 Y (픽셀)
    float dx[kMaxControlPoints];  ///< 변위 X (픽셀, 안쪽=세로축 방향)
    float dy[kMaxControlPoints];  ///< 변위 Y (픽셀, 안쪽=세로축 방향)
    int   count;                  ///< 활성 제어점 수 (8/16/24, 비활성/퇴화 시 0)
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
 * @brief 478점 얼굴 메시에서 V라인 + 내부 축소 워프 파라미터를 픽셀 공간으로 산출 (2노브)
 *
 * 전부 픽셀 공간(정규화 .x/.y × image_width/height)으로 계산한다.
 *   1. face_width_px = 유클리드(454↔234) — roll-robust.
 *   2. 세로축: origin = 10(이마), axis_hat = normalize(152 − 10), axis_len_px = 길이.
 *   3. 퇴화 방어: axis_len_px < 1.0 또는 face_width_px < 8.0 → false(비활성).
 *   4. 조건부 그룹 패킹 (strength ≤ 0 인 그룹은 생략, 슬롯 0부터 연속 패킹):
 *      • jaw_strength > 0: jaw 7점/측(kTaper) + upper 1점/측(kUpperTaper),
 *        max_disp = face_width_px × 0.032 × jaw_strength.
 *      • interior_strength > 0: interior 4점/측(kInteriorTaper),
 *        max_disp = face_width_px × 0.032 × interior_strength.
 *      각 점: v = c − origin; perp = v − (v·axis_hat)axis_hat;
 *      inward = −normalize(perp); disp = inward × (max_disp × taper). |perp|<1e-6이면 0.
 *   5. σ = face_width_px × 0.13. bounds = 패킹된 점 cx/cy min/max ± 3σ.
 *
 * interior_strength=0 이면 jaw 그룹 14점 cx/cy/dx/dy는 기존과 수치 동일(회귀 불변).
 *
 * @param face_mesh        478 정규화(0~1) 랜드마크 (upright 공간 — ADR §7.1)
 * @param image_width      픽셀 환산용 프레임 너비 (픽셀)
 * @param image_height     픽셀 환산용 프레임 높이 (픽셀)
 * @param jaw_strength     V라인 강도 0~1 (= slimFace). jaw + upper 그룹 스케일.
 * @param interior_strength 내부 축소 강도 0~1 (= thinChin). interior 그룹 스케일.
 * @param out              [출력] 워프 파라미터. 비활성 시 count=0, sigma_px=0
 * @param interior_taper_preset [벤치 임시] interior 그룹 taper 프리셋 행 인덱스
 *        (kInteriorTaperPresets, 0=기본). **기본 0 = 기존 동작 비트 동일**. 범위
 *        밖([0,kInteriorTaperPresetCount) 밖: 예 −1·99)은 프리셋 0으로 폴백한다.
 *        jaw/upper 그룹과는 무관(jaw 경로 수치 불변).
 * @return true=워프 활성, false=비활성(두 strength ≤ 0 또는 퇴화 입력 또는 null)
 *
 * @note out은 false 반환 시에도 count=0/sigma_px=0으로 안전하게 채워진다.
 * @warning face_mesh는 최소 478원소여야 한다(인덱스 454까지 접근). null이면 false.
 */
bool computeJawWarp(const IrisLandmark* face_mesh,
                    int image_width,
                    int image_height,
                    float jaw_strength,
                    float interior_strength,
                    JawWarpParams& out,
                    int interior_taper_preset = 0);

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
