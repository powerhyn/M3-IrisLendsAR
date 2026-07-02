/**
 * @file types.h
 * @brief IrisLensSDK 핵심 데이터 타입 정의
 *
 * 모든 플랫폼 바인딩에서 사용되는 기본 데이터 구조체 정의.
 * POD (Plain Old Data) 타입으로 FFI 호환성 보장.
 */

#ifndef IRIS_SDK_TYPES_H
#define IRIS_SDK_TYPES_H

#include <cstdint>

namespace iris_sdk {

// ============================================================
// 열거형 정의
// ============================================================

/**
 * @brief 블렌드 모드 열거형
 * 렌즈 렌더링 시 사용할 블렌딩 방식
 */
// P6-W2 §5.4/§5.12: 블렌드 모드 ID 매핑.
//   - 활성 ID: 0(Normal) / 1(Multiply) / 2(ScreenLinear) / 5(TintLinearV2, default) / 7(ColorReplaceLinear).
//   - Deprecated ID: 3/4/6 → 셰이더에서 TintLinearV2 fallback. enum 값은 외부 호환성 유지를 위해 보존.
//   - W2 시점: ID 7 ColorReplaceLinear는 활성이지만 W5 B1 벤치 결과에 따라 채택/제거 결정.
enum class BlendMode : int {
    Normal = 0,             ///< 일반 알파 블렌딩
    Multiply = 1,           ///< 곱하기 블렌딩
    Screen = 2,             ///< 스크린 블렌딩 (W2: 선형 공간 ScreenLinear로 내부 구현 교체)
    Overlay = 3,            ///< @deprecated (P5-W3-05 S1 D6) 셰이더에서 TintLinearV2 fallback
    LuminanceTint = 4,      ///< @deprecated (P5-W3-05 S1 D6) 셰이더에서 TintLinearV2 fallback
    LuminanceTintLinear = 5,///< 휘도 보존 틴트 (선형 색공간) — **canonical default** (W2 TintLinearV2)
    SoftLight = 6,          ///< @deprecated (P5-W3-05 S1 D6) 셰이더에서 TintLinearV2 fallback
    ColorReplace = 7        ///< 색상 교체 블렌딩 (W2: 선형 공간 ColorReplaceLinear, B1 벤치 대기)
};

/**
 * @brief 프레임 포맷 열거형
 * 입력 이미지의 픽셀 포맷
 */
enum class FrameFormat : int {
    RGBA = 0,       ///< 32비트 RGBA
    BGRA = 1,       ///< 32비트 BGRA
    RGB = 2,        ///< 24비트 RGB
    BGR = 3,        ///< 24비트 BGR
    NV21 = 4,       ///< Android 카메라 YUV 포맷
    NV12 = 5,       ///< iOS 카메라 YUV 포맷
    Grayscale = 6   ///< 8비트 그레이스케일
};

/**
 * @brief 에러 코드 열거형
 * SDK 작업 결과 상태
 */
enum class ErrorCode : int {
    // 성공
    Success = 0,

    // 100번대: 초기화 에러
    NotInitialized = 100,       ///< SDK 초기화되지 않음
    AlreadyInitialized = 101,   ///< SDK 이미 초기화됨
    ModelLoadFailed = 102,      ///< 모델 로드 실패
    InvalidPath = 103,          ///< 잘못된 경로

    // 200번대: 파라미터 에러
    InvalidParameter = 200,         ///< 잘못된 파라미터
    NullPointer = 201,              ///< 널 포인터
    FrameFormatUnsupported = 202,   ///< 지원하지 않는 프레임 포맷

    // 300번대: 검출 에러
    DetectionFailed = 300,      ///< 검출 실패
    NoFaceDetected = 301,       ///< 얼굴 미검출

    // 400번대: 렌더링 에러
    RenderFailed = 400,         ///< 렌더링 실패
    NoTextureLoaded = 401,      ///< 텍스처 미로드

    // 일반 에러
    Unknown = 999               ///< 알 수 없는 에러
};

// ④ W4-D: enum class DetectorType / EyeRefinerPolicy 제거.
//   검출 인프라(iris_detector·mediapipe_detector·inference_thread) 물리 제거에 동반.
//   추적 외부화로 코어는 검출/추적·Eye Refiner를 보유하지 않는다(ADR §3/§6.2).
//   C 미러 IrisEyeRefinerPolicy(sdk_api.h) + no-op stub iris_sdk_set_eye_refiner_policy는
//   ABI 호환을 위해 W4-E deprecation까지 보존된다.

// ============================================================
// 기본 데이터 구조체
// ============================================================

/**
 * @brief 홍채/메시 랜드마크 좌표
 * 정규화된 좌표 (0.0~1.0) 및 가시성 점수
 * POD 타입 - FFI 호환
 *
 * 좌표 계약 (ADR-0001 §7 — 명문 확정):
 *   - x/y는 회전 보정 완료(upright) 프레임 기준 정규화 좌표 [0,1]이다.
 *     회전 책임은 공급자(추적 글루)에 있다 — 코어는 upright 공간만 받는다(§7.1).
 *   - 주입 478점은 항상 비미러(센서 원본 upright). 전면 카메라 미러는 렌더 단계
 *     단일 책임이며 호출자별 분기를 금지한다(§7.4). L/R 의미는 미러 여부와 무관.
 *   - 거리·반경 계산은 반드시 픽셀 좌표 변환 후 수행한다(정규화 그대로 쓰면 종횡비
 *     왜곡으로 가짜 타원 발생, §7.0). frame_width/height가 픽셀 환산 기준이다.
 */
struct IrisLandmark {
    float x;            ///< X 좌표 (정규화, 0.0~1.0, upright 공간 — §7.1)
    float y;            ///< Y 좌표 (정규화, 0.0~1.0, upright 공간 — §7.1)
    float z;            ///< Z 좌표 (깊이). 홍채 z(인덱스 468~477)는 기하 사용 금지(§7.0/§7.2).
                        ///<   비홍채 z는 비기하(깊이 순서) 용도만 허용, 거리·반경 계산 금지(§7.2).
    float visibility;   ///< 가시성 점수 (0.0~1.0)
};

/**
 * @brief 사각형 영역
 * 바운딩 박스 표현용
 * POD 타입 - FFI 호환
 */
struct Rect {
    float x;        ///< 좌상단 X 좌표
    float y;        ///< 좌상단 Y 좌표
    float width;    ///< 너비
    float height;   ///< 높이
};

/**
 * @brief 홍채 검출 결과
 * 양쪽 눈의 홍채 정보 및 얼굴 메타데이터
 * POD 타입 - FFI 호환
 *
 * left/right 명명 계약 (ADR-0001 §7.3 — ④ canonical relabeling 적용 후):
 *   - 인덱스가 정본이고 라벨은 보조 표기다. left_iris ← face_mesh 인덱스 473그룹
 *     {473,474,475,476,477}=피험자 좌안(canonical LEFT_IRIS), right_iris ← 468그룹
 *     {468,469,470,471,472}=피험자 우안(canonical RIGHT_IRIS). 명명 정본 = LandmarkIndices.kt.
 *   - left=피험자 좌안 / right=피험자 우안으로 MediaPipe canonical 해부학 명명에 정합됐다(§7.3).
 *     비미러(센서 원본 upright)에서 피험자 좌안은 화면 우측에 보인다.
 *   - 화면 기준 게이트(데모 applyLeft 등)는 screen_left/screen_right로 명시 분리하고
 *     해부학 라벨과 혼용을 금지한다(§7.3).
 *
 * 홍채 경계 순서 (ADR §7.0 — boundary 인덱스 순서):
 *   right(469/474) → top(470/475) → left(471/476) → bottom(472/477) (이미지 좌표 기준).
 *   patlevin IrisIndex의 LEFT/RIGHT 라벨은 반대이므로 신뢰 금지.
 */
struct IrisResult {
    // 검출 상태
    bool detected;          ///< 전체 검출 성공 여부
    bool left_detected;     ///< 왼쪽 눈 검출 여부
    bool right_detected;    ///< 오른쪽 눈 검출 여부
    float confidence;       ///< 전체 신뢰도 (0.0~1.0)
                            ///<   detector 경로: face_confidence * eye_factor (측정값).
                            ///<   주입 경로(§6.2): MediaPipe Tasks가 score 미노출 → 측정값 부재.
                            ///<   게이팅을 visibility(EAR 파생)로 일원화하기 위해 detected 시 게이트
                            ///<   통과 상수 1.0(곱셈 항등원), 미검출 시 0.0으로 고정한다. presence
                            ///<   게이트는 detected/visibility가 담당(deriveIrisResult 참조).

    // 피험자 좌안 홍채 (5개 랜드마크: center + 4 boundary). center=인덱스 473 (§7.3 canonical)
    IrisLandmark left_iris[5];
    float left_radius;      ///< 피험자 좌안 홍채 반지름 (픽셀 — 중심↔경계 평균 거리, 픽셀 환산 §7.0)

    // 피험자 우안 홍채 (5개 랜드마크: center + 4 boundary). center=인덱스 468 (§7.3 canonical)
    IrisLandmark right_iris[5];
    float right_radius;     ///< 오른쪽 홍채 반지름 (픽셀 — 픽셀 환산 §7.0)

    // 얼굴 메타데이터
    Rect face_rect;             ///< 얼굴 바운딩 박스
    float face_rotation[3];     ///< 얼굴 회전 [pitch, yaw, roll] (도)

    // Face Mesh (478 랜드마크, 디버그/시각화용)
    static constexpr int FACE_MESH_LANDMARK_COUNT = 478;
    IrisLandmark face_mesh[478];    ///< 전체 얼굴 메쉬 랜드마크
    bool face_mesh_valid;           ///< face_mesh 데이터 유효 여부

    // 프레임 정보
    int64_t timestamp_ms;   ///< 타임스탬프 (밀리초)
    int32_t frame_width;    ///< 원본 프레임 너비
    int32_t frame_height;   ///< 원본 프레임 높이

    // 눈꺼풀 가림 비율 (W3 트랙)
    // ④ W4-D: detector 전용 메타 iris_quality_*/eye_refiner_used 제거(ADR §6.2).
    //   eyelid_ratio_*는 W3용 별도 트랙, avg_iris_luma_*(아래)는 P7-W2 활성 필드 — 보존.
    float eyelid_ratio_left;    ///< 왼쪽 눈꺼풀 가림 비율 (0.0~1.0, 향후 W3용)
    float eyelid_ratio_right;   ///< 오른쪽 눈꺼풀 가림 비율 (0.0~1.0, 향후 W3용)

    // P7-W2 §5.5: iris ROI 실측 평균 luma (srgb²+Rec.709 linear, 0~1).
    // 미측정/미검출 시 -1.0f sentinel. C IrisResult(sdk_api.h)와 reinterpret_cast로
    // 교환되므로(sdk_api_v2.cpp) 양쪽 동일 위치(struct 끝)에 동일 타입으로 추가.
    // default -1: detect() 밖(frame_processor 등) stack 생성 시 garbage 방지(완전성).
    // is_trivially_copyable/standard_layout 불변(test_types 통과), memset(0) 경로는 어댑터 >0 가드.
    float avg_iris_luma_left = -1.0f;   ///< 왼쪽 홍채 ROI 평균 linear luma (-1=미측정)
    float avg_iris_luma_right = -1.0f;  ///< 오른쪽 홍채 ROI 평균 linear luma (-1=미측정)
};

/**
 * @brief 렌즈 렌더링 설정
 * 가상 렌즈 오버레이 파라미터
 * POD 타입 - FFI 호환 (기본값은 생성 시 설정)
 */
struct LensConfig {
    float opacity = 0.7f;       ///< 투명도 (0.0~1.0)
    float scale = 1.3f;         ///< 크기 배율 — 실기기 튜닝(989fdac/6f596a2) canonical 승격 (P6-W2 §5.12 패턴)
    float offset_x = 0.0f;      ///< X 오프셋 (정규화)
    float offset_y = 0.0f;      ///< Y 오프셋 (정규화)
    float rotation = 0.0f;      ///< 회전 각도 (라디안, -PI~PI)
    BlendMode blend_mode = BlendMode::LuminanceTintLinear;  ///< 블렌드 모드 — P6-W2 §5.12 canonical default (ID=5, TintLinearV2)
    float edge_feather = 0.15f; ///< 가장자리 페더링 (0.0~1.0) — 실기기 튜닝(989fdac/6f596a2) canonical 승격
    bool apply_left = true;     ///< 왼쪽 눈 적용 여부
    bool apply_right = true;    ///< 오른쪽 눈 적용 여부
    bool is_mirror = false;     ///< 전면 카메라 mirror (X-flip + 좌우 swap)
};

} // namespace iris_sdk

#endif // IRIS_SDK_TYPES_H
