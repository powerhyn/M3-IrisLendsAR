# P1-W3-01: IrisDetector 인터페이스 정의

**태스크 ID**: P1-W3-01
**상태**: ✅ 완료
**시작일**: 2026-01-07
**완료일**: 2026-01-07

---

## 1. 계획

### 목표
홍채 검출기의 추상 인터페이스를 정의하여 Strategy 패턴 기반의 검출기 교체가 가능하도록 한다.

### 산출물
| 파일 | 설명 |
|------|------|
| `cpp/include/iris_sdk/iris_detector.h` | IrisDetector 추상 클래스 정의 |
| `cpp/src/iris_detector.cpp` | 기본 구현 (팩토리 함수 등) |

### 검증 기준
- [x] 헤더 파일 컴파일 성공
- [x] 가상 소멸자 정의
- [x] 순수 가상 함수 정의 (initialize, detect, release)
- [x] 팩토리 패턴 지원
- [x] Doxygen 스타일 문서화

### 선행 조건
- P1-W2-04: Android 크로스컴파일 테스트 ✅ 완료

---

## 2. 분석

### 2.1 인터페이스 설계 원칙

**Strategy 패턴 적용**
```
IrisDetector (추상 인터페이스)
├── MediaPipeDetector (Phase 1)
├── EyeOnlyDetector (Phase 2)
└── HybridDetector (Phase 2 - 폴백 전략)
```

**핵심 고려사항**:
1. **플랫폼 독립성**: OpenCV `cv::Mat` 또는 raw pointer 기반 입력
2. **메모리 관리**: RAII 패턴, unique_ptr 반환
3. **스레드 안전성**: 단일 인스턴스 단일 스레드 처리 가정
4. **확장성**: 새 검출기 추가 시 인터페이스 변경 불필요

### 2.2 메서드 설계

| 메서드 | 반환값 | 설명 |
|--------|--------|------|
| `initialize(model_path)` | `bool` | 모델 로드 및 초기화 |
| `detect(frame)` | `IrisResult` | 프레임에서 홍채 검출 |
| `release()` | `void` | 리소스 해제 |
| `isInitialized()` | `bool` | 초기화 상태 확인 |
| `getDetectorType()` | `DetectorType` | 검출기 종류 반환 |

### 2.3 예상 인터페이스 코드

```cpp
#pragma once

#include <memory>
#include <string>
#include "iris_sdk/types.h"
#include "iris_sdk/export.h"

// Forward declarations
namespace cv { class Mat; }

namespace iris_sdk {

/**
 * @brief 검출기 종류 열거형
 */
enum class DetectorType {
    Unknown = 0,
    MediaPipe,    ///< MediaPipe 기반 (Phase 1)
    EyeOnly,      ///< Eye-Only 커스텀 모델 (Phase 2)
    Hybrid        ///< 하이브리드 (Phase 2)
};

/**
 * @brief 홍채 검출기 추상 인터페이스
 *
 * Strategy 패턴을 사용하여 다양한 검출기 구현체를 교체 가능하게 함.
 */
class IRIS_SDK_EXPORT IrisDetector {
public:
    virtual ~IrisDetector() = default;

    /**
     * @brief 검출기 초기화
     * @param model_path 모델 파일 경로
     * @return 초기화 성공 여부
     */
    virtual bool initialize(const std::string& model_path) = 0;

    /**
     * @brief 프레임에서 홍채 검출
     * @param frame 입력 이미지 (BGR 또는 RGB)
     * @return 검출 결과
     */
    virtual IrisResult detect(const cv::Mat& frame) = 0;

    /**
     * @brief 리소스 해제
     */
    virtual void release() = 0;

    /**
     * @brief 초기화 상태 확인
     * @return 초기화되었으면 true
     */
    virtual bool isInitialized() const = 0;

    /**
     * @brief 검출기 종류 반환
     * @return DetectorType 열거값
     */
    virtual DetectorType getDetectorType() const = 0;

    // 복사/이동 금지 (각 구현체에서 결정)
    IrisDetector(const IrisDetector&) = delete;
    IrisDetector& operator=(const IrisDetector&) = delete;

protected:
    IrisDetector() = default;
};

// ============================================================
// 내부 팩토리 함수 (detail 네임스페이스)
// ============================================================
//
// 참고: 외부 API 사용자는 SDKManager::createDetector()를 사용해야 함.
// 아래 함수들은 SDKManager 내부 및 테스트 용도로만 사용됨.
//

namespace detail {

/**
 * @brief 검출기 팩토리 함수 (내부용)
 * @param type 생성할 검출기 종류
 * @return unique_ptr로 관리되는 검출기 인스턴스
 * @note 외부에서는 SDKManager::createDetector() 사용 권장
 */
std::unique_ptr<IrisDetector> createDetector(DetectorType type);

/**
 * @brief 문자열로 검출기 생성 (내부용)
 * @param type_name 검출기 이름 ("mediapipe", "eyeonly", "hybrid")
 * @return unique_ptr로 관리되는 검출기 인스턴스
 * @note 외부에서는 SDKManager::createDetector() 사용 권장
 */
std::unique_ptr<IrisDetector> createDetector(const std::string& type_name);

} // namespace detail

} // namespace iris_sdk
```

### 2.4 의존성 분석

| 의존성 | 필수 | 용도 |
|--------|------|------|
| OpenCV | No | 내부 구현체에서만 사용 (인터페이스는 raw pointer) |
| types.h | Yes | IrisResult, FrameFormat, DetectorType |
| export.h | Yes | DLL export 매크로 |

> **변경 사항**: 인터페이스에서 cv::Mat 대신 raw pointer를 사용하여 OpenCV 의존성 제거. MediaPipeDetector 등 구현체 내부에서만 OpenCV 사용.

---

## 3. 실행 내역

### 3.1 TDD RED Phase - 테스트 먼저 작성

```bash
# cpp/tests/test_iris_detector.cpp 생성 (17개 테스트 케이스)
# 테스트 작성 후 빌드 → 예상대로 컴파일 실패 (IrisDetector 미정의)
```

**작성된 테스트 케이스**:
- 클래스 특성 테스트: 4개 (추상클래스, 가상소멸자, 복사금지, 이동금지)
- Mock 검출기 테스트: 8개 (생성, 초기화, 검출, 미초기화검출, 널프레임, 잘못된크기, 해제, 타입)
- 다형성 테스트: 1개 (기본포인터로 파생클래스 접근)
- 팩토리 함수 테스트: 4개 (MediaPipe생성, 문자열생성, Unknown생성, 잘못된문자열)

### 3.2 TDD GREEN Phase - iris_detector.h 구현

```bash
# cpp/include/iris_sdk/iris_detector.h 구현
# cpp/src/iris_detector.cpp 생성 (팩토리 함수)
# 모든 17개 테스트 통과
```

**구현된 인터페이스**:
- `class IrisDetector` - 추상 검출기 인터페이스
  - `virtual ~IrisDetector() = default` - 가상 소멸자
  - `virtual bool initialize(const std::string& model_path) = 0` - 초기화
  - `virtual IrisResult detect(const uint8_t*, int, int, FrameFormat) = 0` - 검출
  - `virtual void release() = 0` - 해제
  - `virtual bool isInitialized() const = 0` - 상태 확인
  - `virtual DetectorType getDetectorType() const = 0` - 타입 반환
  - 복사/이동 금지 (`= delete`)
  - protected 기본 생성자

**팩토리 함수** (detail:: 네임스페이스):
- `createDetector(DetectorType type)` - 열거형 기반 생성
- `createDetector(const std::string& type_name)` - 문자열 기반 생성

### 3.3 CMakeLists.txt 업데이트

```cmake
# cpp/CMakeLists.txt - 소스 파일 추가
set(IRIS_SDK_SOURCES
    src/iris_detector.cpp
)

# cpp/tests/CMakeLists.txt - 테스트 타겟 추가
add_executable(test_iris_detector test_iris_detector.cpp)
target_link_libraries(test_iris_detector PRIVATE GTest::gtest GTest::gtest_main iris_sdk)
```

### 3.4 설계 변경 사항

**cv::Mat 대신 raw pointer 사용**:
- 원래 설계: `detect(const cv::Mat& frame)`
- 변경된 설계: `detect(const uint8_t* frame_data, int width, int height, FrameFormat format)`
- 이유: OpenCV 의존성 없이 테스트 가능, FFI 호환성 향상

---

## 4. 검증 결과

### 검증 항목

| 항목 | 결과 | 비고 |
|------|------|------|
| 헤더 컴파일 | ✅ 통과 | iris_detector.h 컴파일 성공 |
| 가상 소멸자 | ✅ 통과 | `has_virtual_destructor` 테스트 |
| 순수 가상 함수 | ✅ 통과 | `is_abstract` 테스트 |
| 팩토리 함수 | ✅ 통과 | Unknown/Invalid → nullptr 반환 |
| Doxygen 문서화 | ✅ 통과 | 모든 메서드 주석 완료 |

### 테스트 결과

```
[==========] 17 tests from 1 test suite ran. (0 ms total)
[  PASSED  ] 17 tests.
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
| cv::Mat 직접 사용 | OpenCV가 필수 의존성이므로 raw pointer 래핑 불필요 |
| unique_ptr 반환 | 명확한 소유권 이전, 메모리 누수 방지 |
| 가상 소멸자 default | 구현체에서 필요시 오버라이드 |
| 팩토리 함수 detail:: 이동 | SDKManager가 공식 진입점, 내부 함수 분리로 API 명확화 |

### 학습 내용

1. **TDD로 인터페이스 설계 검증**: 테스트 먼저 작성하니 인터페이스 사용성이 명확해지고, Mock 구현체로 설계 검증 가능
2. **추상 클래스 테스트 전략**: `std::is_abstract`, `std::has_virtual_destructor` 등 타입 트레이트로 클래스 특성 검증
3. **cv::Mat vs raw pointer**: OpenCV 의존성 없이 테스트하려면 raw pointer 인터페이스가 유리, 내부에서 cv::Mat 변환 가능
4. **복사/이동 금지**: 리소스 관리 객체는 `= delete`로 복사/이동 금지하여 안전성 확보
5. **protected 생성자**: 추상 클래스의 직접 인스턴스화 방지, 팩토리 패턴과 결합

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-01-07 | 태스크 문서 생성, 분석 및 설계 완료 |
| 2026-01-07 | 아키텍처 리뷰: 팩토리 함수를 detail:: 네임스페이스로 이동 (API 명확화) |
| 2026-01-07 | TDD 기반 구현 완료 (17개 테스트 통과), detect() 시그니처 변경 (cv::Mat → raw pointer) |
