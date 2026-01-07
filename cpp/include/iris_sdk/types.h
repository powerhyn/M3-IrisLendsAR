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
enum class BlendMode : int {
    Normal = 0,     ///< 일반 알파 블렌딩
    Multiply = 1,   ///< 곱하기 블렌딩
    Screen = 2,     ///< 스크린 블렌딩
    Overlay = 3     ///< 오버레이 블렌딩
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

/**
 * @brief 검출기 타입 열거형
 * 홍채 검출에 사용할 검출기 종류
 */
enum class DetectorType : int {
    Unknown = 0,    ///< 알 수 없음
    MediaPipe = 1,  ///< MediaPipe Face Mesh + Iris
    EyeOnly = 2,    ///< 눈 영역 전용 커스텀 모델
    Hybrid = 3      ///< MediaPipe + EyeOnly 하이브리드
};

// ============================================================
// 기본 데이터 구조체
// ============================================================

/**
 * @brief 홍채 랜드마크 좌표
 * 정규화된 좌표 (0.0~1.0) 및 가시성 점수
 * POD 타입 - FFI 호환
 */
struct IrisLandmark {
    float x;            ///< X 좌표 (정규화, 0.0~1.0)
    float y;            ///< Y 좌표 (정규화, 0.0~1.0)
    float z;            ///< Z 좌표 (깊이, 정규화)
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
 */
struct IrisResult {
    // 검출 상태
    bool detected;          ///< 전체 검출 성공 여부
    bool left_detected;     ///< 왼쪽 눈 검출 여부
    bool right_detected;    ///< 오른쪽 눈 검출 여부
    float confidence;       ///< 전체 신뢰도 (0.0~1.0)

    // 왼쪽 눈 홍채 (5개 랜드마크: center + 4 boundary)
    IrisLandmark left_iris[5];
    float left_radius;      ///< 왼쪽 홍채 반지름 (픽셀)

    // 오른쪽 눈 홍채 (5개 랜드마크: center + 4 boundary)
    IrisLandmark right_iris[5];
    float right_radius;     ///< 오른쪽 홍채 반지름 (픽셀)

    // 얼굴 메타데이터
    Rect face_rect;             ///< 얼굴 바운딩 박스
    float face_rotation[3];     ///< 얼굴 회전 [pitch, yaw, roll] (도)

    // 프레임 정보
    int64_t timestamp_ms;   ///< 타임스탬프 (밀리초)
    int32_t frame_width;    ///< 원본 프레임 너비
    int32_t frame_height;   ///< 원본 프레임 높이
};

/**
 * @brief 렌즈 렌더링 설정
 * 가상 렌즈 오버레이 파라미터
 * POD 타입 - FFI 호환 (기본값은 생성 시 설정)
 */
struct LensConfig {
    float opacity = 0.7f;       ///< 투명도 (0.0~1.0)
    float scale = 1.0f;         ///< 크기 배율
    float offset_x = 0.0f;      ///< X 오프셋 (정규화)
    float offset_y = 0.0f;      ///< Y 오프셋 (정규화)
    BlendMode blend_mode = BlendMode::Normal;   ///< 블렌드 모드
    float edge_feather = 0.1f;  ///< 가장자리 페더링 (0.0~1.0)
    bool apply_left = true;     ///< 왼쪽 눈 적용 여부
    bool apply_right = true;    ///< 오른쪽 눈 적용 여부
};

} // namespace iris_sdk

#endif // IRIS_SDK_TYPES_H
