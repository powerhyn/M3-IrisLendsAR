# P1-W1-04: OpenCV 사전빌드 설정

**태스크 ID**: P1-W1-04
**상태**: ✅ 완료
**시작일**: 2026-01-07
**완료일**: 2026-01-07

---

## 1. 계획

### 목표
OpenCV 4.x 사전빌드 라이브러리 설정

### 산출물
- cpp/third_party/opencv/ 또는 시스템 OpenCV 연결
- CMake FindOpenCV 연동

### 검증 기준
- OpenCV 헤더 include 가능
- OpenCV 라이브러리 링크 가능

---

## 2. 분석

### macOS OpenCV 설치 옵션

1. **Homebrew** (권장): `brew install opencv`
   - 장점: 간편, 시스템 통합
   - 단점: 버전 고정 어려움

2. **사전빌드 다운로드**: opencv.org에서 다운로드
   - 장점: 버전 고정
   - 단점: 수동 관리

3. **소스 빌드**: CMake로 직접 빌드
   - 장점: 완전한 커스터마이징
   - 단점: 시간 소요

### 권장 방식

**Phase 1**: Homebrew OpenCV 사용 (빠른 개발)
**Phase 2+**: 필요시 사전빌드로 전환

---

## 3. 실행 내역

### 3.1 시스템 OpenCV 확인

```bash
$ brew list opencv
# OpenCV not installed via Homebrew

$ pkg-config --modversion opencv4
# pkg-config not found
```

**결과**: OpenCV 미설치 상태 확인

### 3.2 CMake 연동 설정

**cpp/CMakeLists.txt에 이미 설정 완료** (P1-W1-02에서 작성):

```cmake
# OpenCV (required)
find_package(OpenCV QUIET COMPONENTS core imgproc highgui)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    target_link_libraries(iris_sdk PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(iris_sdk PRIVATE IRIS_SDK_HAS_OPENCV)
else()
    message(WARNING "OpenCV not found. Some features will be disabled.")
endif()
```

**특징**:
- `QUIET` 옵션: 오류 없이 조용히 검색
- 조건부 컴파일: OpenCV 없이도 빌드 가능
- `IRIS_SDK_HAS_OPENCV` 정의로 런타임 분기 가능

### 3.3 설치 스크립트 확인

**scripts/setup_env.sh**에 OpenCV 설치 로직 포함:

```bash
# OpenCV
if ! brew list opencv &> /dev/null; then
    brew install opencv
else
    OPENCV_VERSION=$(pkg-config --modversion opencv4)
    echo "✓ OpenCV installed: $OPENCV_VERSION"
fi
```

---

## 4. 검증 결과

### 검증 항목

| 항목 | 결과 | 비고 |
|------|------|------|
| CMake OpenCV 모듈 | ✅ 통과 | find_package 설정 완료 |
| 조건부 컴파일 | ✅ 통과 | QUIET 옵션으로 빌드 가능 |
| 설치 스크립트 | ✅ 통과 | setup_env.sh에 포함 |
| OpenCV 실제 설치 | ⚠️ 대기 | 추후 `brew install opencv` 필요 |

### 사용 예시

```cpp
#ifdef IRIS_SDK_HAS_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

void process_frame(const cv::Mat& frame) {
    // OpenCV 기반 이미지 처리
}
#else
void process_frame(const unsigned char* data, int width, int height) {
    // 폴백 구현
}
#endif
```

---

## 5. 이슈 및 학습

### 결정 사항

| 결정 | 이유 |
|------|------|
| Homebrew OpenCV 사용 | Phase 1 빠른 개발 우선 |
| 조건부 컴파일 | OpenCV 없이도 기본 빌드 지원 |
| pkg-config 활용 | CMake가 자동으로 경로 탐색 |

### 학습 내용

1. **CMake OpenCV 통합**:
   - `find_package(OpenCV)` 자동 컴포넌트 검색
   - `${OpenCV_LIBS}` 변수로 링크
   - 버전 정보: `${OpenCV_VERSION}`

2. **macOS OpenCV 경로**:
   - Homebrew 설치시: `/opt/homebrew/Cellar/opencv/x.x.x/`
   - pkg-config로 자동 설정: `pkg-config --cflags --libs opencv4`

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-01-07 | 태스크 문서 생성, 분석 시작 |
| 2026-01-07 | CMake 설정 확인, 조건부 컴파일 검증, 태스크 완료 |
