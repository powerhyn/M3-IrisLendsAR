# P1-W3-02: 데이터 구조 정의

**태스크 ID**: P1-W3-02
**상태**: ✅ 완료
**시작일**: 2026-01-07
**완료일**: 2026-01-07

---

## 1. 계획

### 목표
SDK 전체에서 사용되는 핵심 데이터 구조 (IrisResult, IrisLandmark, LensConfig, FrameFormat 등)를 정의한다.

### 산출물
| 파일 | 설명 |
|------|------|
| `cpp/include/iris_sdk/types.h` | 모든 데이터 구조 정의 |
| `cpp/include/iris_sdk/error_codes.h` | 에러 코드 열거형 |

### 검증 기준
- [x] 모든 구조체 컴파일 성공
- [x] 메모리 레이아웃 검증 (C ABI 호환)
- [x] POD 타입 검증 (trivially copyable)
- [x] 기본값 초기화 지원
- [x] Doxygen 문서화 완료

### 선행 조건
- P1-W3-01: IrisDetector 인터페이스 정의 (병렬 가능)

---

## 2. 분석

### 2.1 데이터 구조 목록

| 구조체 | 용도 | C API 호환 |
|--------|------|------------|
| IrisLandmark | 단일 랜드마크 좌표 | Yes |
| IrisResult | 홍채 검출 결과 | Yes |
| LensConfig | 렌즈 렌더링 설정 | Yes |
| FrameInfo | 프레임 메타데이터 | Yes |
| Rect | 바운딩 박스 | Yes |

### 2.2 IrisLandmark 구조체

```cpp
/**
 * @brief 단일 랜드마크 좌표
 *
 * 좌표는 정규화된 값 (0.0 ~ 1.0) 또는 픽셀 좌표로 사용 가능
 */
struct IrisLandmark {
    float x;        ///< X 좌표 (정규화: 0.0~1.0, 픽셀: 실제 좌표)
    float y;        ///< Y 좌표 (정규화: 0.0~1.0, 픽셀: 실제 좌표)
    float z;        ///< Z 좌표 (깊이, 선택적)
    float visibility; ///< 가시성 신뢰도 (0.0~1.0)
};
```

### 2.3 IrisResult 구조체

```cpp
/**
 * @brief 홍채 검출 결과
 *
 * 좌/우 눈의 홍채 랜드마크 및 관련 정보를 포함
 */
struct IrisResult {
    // 검출 상태
    bool detected;              ///< 검출 성공 여부
    float confidence;           ///< 전체 신뢰도 (0.0~1.0)

    // 왼쪽 눈 홍채 (5 points: center + 4 boundary)
    bool left_detected;         ///< 왼쪽 눈 검출 여부
    IrisLandmark left_iris[5];  ///< 왼쪽 홍채 랜드마크
    Rect left_eye_rect;         ///< 왼쪽 눈 바운딩 박스
    float left_iris_radius;     ///< 왼쪽 홍채 반지름 (픽셀)

    // 오른쪽 눈 홍채 (5 points: center + 4 boundary)
    bool right_detected;        ///< 오른쪽 눈 검출 여부
    IrisLandmark right_iris[5]; ///< 오른쪽 홍채 랜드마크
    Rect right_eye_rect;        ///< 오른쪽 눈 바운딩 박스
    float right_iris_radius;    ///< 오른쪽 홍채 반지름 (픽셀)

    // 얼굴 정보
    Rect face_rect;             ///< 얼굴 바운딩 박스
    float face_rotation[3];     ///< 얼굴 회전 (pitch, yaw, roll)

    // 메타데이터
    int64_t timestamp_ms;       ///< 타임스탬프 (밀리초)
    int frame_width;            ///< 원본 프레임 너비
    int frame_height;           ///< 원본 프레임 높이
};
```

### 2.4 LensConfig 구조체

```cpp
/**
 * @brief 렌즈 렌더링 설정
 */
struct LensConfig {
    // 렌즈 텍스처
    const char* texture_path;   ///< 텍스처 파일 경로
    const uint8_t* texture_data; ///< 텍스처 데이터 (메모리)
    int texture_width;          ///< 텍스처 너비
    int texture_height;         ///< 텍스처 높이

    // 렌더링 옵션
    float opacity;              ///< 투명도 (0.0~1.0, 기본 0.7)
    float scale;                ///< 크기 배율 (기본 1.0)
    float offset_x;             ///< X 오프셋 (정규화, 기본 0.0)
    float offset_y;             ///< Y 오프셋 (정규화, 기본 0.0)

    // 블렌딩 옵션
    int blend_mode;             ///< 블렌딩 모드 (0: Normal, 1: Multiply, 2: Screen)
    bool edge_smoothing;        ///< 경계 스무딩 활성화
    float edge_feather;         ///< 경계 페더링 정도 (0.0~1.0)

    // 양눈 독립 설정
    bool apply_left;            ///< 왼쪽 눈 적용 여부
    bool apply_right;           ///< 오른쪽 눈 적용 여부
};
```

### 2.5 FrameFormat 열거형

```cpp
/**
 * @brief 프레임 픽셀 포맷
 */
enum class FrameFormat {
    Unknown = 0,
    RGBA,       ///< 32bit RGBA (모바일 일반)
    BGRA,       ///< 32bit BGRA (Windows)
    RGB,        ///< 24bit RGB
    BGR,        ///< 24bit BGR (OpenCV 기본)
    NV21,       ///< YUV NV21 (Android Camera)
    NV12,       ///< YUV NV12 (iOS Camera)
    YUV420P,    ///< YUV 4:2:0 Planar
    GRAY        ///< 8bit Grayscale
};
```

### 2.6 ErrorCode 열거형

```cpp
/**
 * @brief SDK 에러 코드
 */
enum class ErrorCode {
    Success = 0,

    // 초기화 에러 (100~199)
    NotInitialized = 100,
    AlreadyInitialized = 101,
    ModelLoadFailed = 102,
    InvalidModelPath = 103,

    // 검출 에러 (200~299)
    DetectionFailed = 200,
    InvalidFrame = 201,
    FrameFormatUnsupported = 202,
    NoFaceDetected = 203,

    // 렌더링 에러 (300~399)
    RenderFailed = 300,
    TextureLoadFailed = 301,
    InvalidTexture = 302,

    // 일반 에러 (900~999)
    Unknown = 999
};
```

### 2.7 메모리 레이아웃 고려사항

**C ABI 호환성 규칙**:
1. POD (Plain Old Data) 타입 유지
2. 가상 함수 테이블 없음
3. 표준 레이아웃 (standard layout)
4. 패딩 최소화 (데이터 정렬 고려)

```cpp
// 크기 검증 (static_assert 사용)
static_assert(sizeof(IrisLandmark) == 16, "IrisLandmark size mismatch");
static_assert(sizeof(Rect) == 16, "Rect size mismatch");
static_assert(std::is_trivially_copyable_v<IrisResult>, "IrisResult must be POD");
```

---

## 3. 실행 내역

### 3.1 TDD RED Phase - 테스트 먼저 작성

```bash
# cpp/tests/test_types.cpp 생성 (34개 테스트 케이스)
# 테스트 작성 후 빌드 → 예상대로 컴파일 실패 (타입 미정의)
```

**작성된 테스트 케이스**:
- IrisLandmarkTest: 6개 (기본생성, 값할당, 집합초기화, POD검증, 크기검증, 경계값)
- RectTest: 5개 (기본생성, 값할당, 집합초기화, POD검증, 크기검증)
- IrisResultTest: 10개 (기본생성, 좌/우눈 검출, 양눈검출, 얼굴정보, 회전, 타임스탬프, 프레임크기, 배열, POD검증)
- LensConfigTest: 4개 (기본값, 커스텀값, 경계값, POD검증)
- BlendModeTest: 1개 (열거값)
- FrameFormatTest: 1개 (열거값)
- ErrorCodeTest: 6개 (성공, 초기화에러, 파라미터에러, 검출에러, 렌더링에러, 일반에러)
- DetectorTypeTest: 1개 (열거값)

### 3.2 TDD GREEN Phase - types.h 구현

```bash
# cpp/include/iris_sdk/types.h 구현
# 모든 34개 테스트 통과
```

**구현된 타입**:
- `enum class BlendMode` - 블렌드 모드 (Normal, Multiply, Screen, Overlay)
- `enum class FrameFormat` - 프레임 포맷 (RGBA, BGRA, RGB, BGR, NV21, NV12, Grayscale)
- `enum class ErrorCode` - 에러 코드 (100번대: 초기화, 200번대: 파라미터, 300번대: 검출, 400번대: 렌더링)
- `enum class DetectorType` - 검출기 타입 (Unknown, MediaPipe, EyeOnly, Hybrid)
- `struct IrisLandmark` - 홍채 랜드마크 좌표 (x, y, z, visibility - 16바이트)
- `struct Rect` - 사각형 영역 (x, y, width, height - 16바이트)
- `struct IrisResult` - 홍채 검출 결과
- `struct LensConfig` - 렌즈 설정 (기본값 포함)

### 3.3 TDD REFACTOR Phase

리팩터링 불필요 - 코드가 이미 깔끔하고 모든 테스트 통과

### 3.4 CMakeLists.txt 업데이트

```cmake
# cpp/tests/CMakeLists.txt에 GoogleTest 설정 및 테스트 타겟 추가
add_executable(test_types test_types.cpp)
target_link_libraries(test_types PRIVATE GTest::gtest GTest::gtest_main)
```

---

## 4. 검증 결과

### 검증 항목

| 항목 | 결과 | 비고 |
|------|------|------|
| 구조체 컴파일 | ✅ 통과 | 모든 타입 컴파일 성공 |
| POD 타입 검증 | ✅ 통과 | `is_trivially_copyable`, `is_standard_layout` |
| 메모리 정렬 | ✅ 통과 | IrisLandmark=16B, Rect=16B |
| 기본값 초기화 | ✅ 통과 | LensConfig 기본값 테스트 |
| Doxygen 문서화 | ✅ 통과 | 모든 타입/필드 주석 완료 |

### 테스트 결과

```
[==========] 34 tests from 8 test suites ran. (0 ms total)
[  PASSED  ] 34 tests.
```

---

## 5. 이슈 및 학습

### 이슈

| ID | 내용 | 상태 | 해결방안 |
|----|------|------|----------|
| - | - | - | - |

### 결정 사항

| 결정 | 이유 |
|------|------|
| 5-point 홍채 랜드마크 | MediaPipe 출력과 일치, 중심점 + 경계 4점 |
| 정규화 좌표 기본 | 해상도 독립성, 플랫폼 간 호환성 |
| C-style 배열 사용 | C ABI 호환성, FFI 단순화 |
| timestamp_ms int64_t | 밀리초 정밀도, 오버플로우 방지 |

### 학습 내용

1. **TDD RED-GREEN-REFACTOR 사이클**: 테스트 먼저 작성 후 구현하니 요구사항이 명확해지고, 구현 완료 시점도 확실해짐
2. **GoogleTest FetchContent**: CMake의 FetchContent로 GoogleTest를 자동 다운로드하여 의존성 관리 간소화
3. **POD 타입 검증**: `std::is_trivially_copyable`, `std::is_standard_layout` 타입 트레이트로 FFI 호환성 보장
4. **기본값 멤버 초기화**: C++11 이상에서 구조체 멤버에 직접 기본값 할당 가능 (LensConfig 활용)
5. **enum class 명시적 값**: FFI 호환을 위해 열거형에 명시적 정수값 지정 필수

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-01-07 | 태스크 문서 생성, 데이터 구조 설계 완료 |
| 2026-01-07 | TDD 기반 구현 완료 (34개 테스트 통과) |
