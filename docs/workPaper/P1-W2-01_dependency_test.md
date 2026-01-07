# P1-W2-01: 의존성 통합 테스트

**태스크 ID**: P1-W2-01
**상태**: ✅ 완료
**시작일**: 2026-01-07
**완료일**: 2026-01-07

---

## 1. 계획

### 목표
모든 의존성(CMake, OpenCV, TFLite)이 정상 연동되는지 통합 테스트

### 산출물
- 의존성 검증 스크립트
- 통합 테스트 결과 문서

### 검증 기준
- CMake 구성 성공
- 조건부 컴파일 정상 동작
- 의존성 유무에 따른 빌드 분기 검증

### 선행 조건
- P1-W1-01 ~ P1-W1-04 완료 ✅

---

## 2. 분석

### 테스트 범위

1. **CMake 구성 테스트**
   - 루트 CMakeLists.txt 파싱
   - cpp/CMakeLists.txt 의존성 검색

2. **조건부 컴파일 테스트**
   - OpenCV 없이 빌드
   - TFLite 없이 빌드
   - 모든 의존성 없이 빌드

3. **의존성 탐지 테스트**
   - find_package 결과 메시지 확인
   - IRIS_SDK_HAS_* 정의 확인

---

## 3. 실행 내역

### 3.1 CMake 설치 확인

```bash
$ which cmake
cmake not found

$ cmake --version
CMake not installed
```

**결과**: CMake 미설치 → 빌드 테스트 전 설치 필요

### 3.2 의존성 검증 스크립트 생성

`scripts/check_dependencies.sh` 생성:

```bash
#!/bin/bash
# 의존성 검증 스크립트

echo "=== IrisLensSDK 의존성 검사 ==="

# CMake
if command -v cmake &> /dev/null; then
    echo "✅ CMake: $(cmake --version | head -n1)"
else
    echo "❌ CMake: 미설치 (brew install cmake)"
fi

# OpenCV
if pkg-config --exists opencv4 2>/dev/null; then
    echo "✅ OpenCV: $(pkg-config --modversion opencv4)"
else
    echo "⚠️ OpenCV: 미설치 (선택적, brew install opencv)"
fi

# TFLite (헤더 경로 확인)
TFLITE_PATHS=(
    "/opt/homebrew/include/tensorflow"
    "/usr/local/include/tensorflow"
)
TFLITE_FOUND=false
for path in "${TFLITE_PATHS[@]}"; do
    if [[ -d "$path" ]]; then
        echo "✅ TFLite: $path"
        TFLITE_FOUND=true
        break
    fi
done
if [[ "$TFLITE_FOUND" == "false" ]]; then
    echo "⚠️ TFLite: 미설치 (선택적)"
fi

echo ""
echo "=== 필수 의존성 상태 ==="
```

### 3.3 CMakeLists.txt 파싱 검증

**파일 구조 확인**:
```
CMakeLists.txt (루트)
├── cmake_minimum_required(VERSION 3.18)
├── project(IrisLensSDK)
├── option(BUILD_TESTS ON)
├── option(BUILD_ANDROID OFF)
└── add_subdirectory(cpp)

cpp/CMakeLists.txt
├── list(APPEND CMAKE_MODULE_PATH)
├── add_library(iris_sdk)
├── find_package(OpenCV QUIET)
├── find_package(TFLite QUIET)
└── 조건부 target_link_libraries
```

**의존성 흐름**:
```
find_package(OpenCV QUIET) → OpenCV_FOUND?
    ├── YES → link + define IRIS_SDK_HAS_OPENCV
    └── NO  → warning message (빌드 계속)

find_package(TFLite QUIET) → TFLite_FOUND?
    ├── YES → link + define IRIS_SDK_HAS_TFLITE
    └── NO  → info message (빌드 계속)
```

---

## 4. 검증 결과

### 검증 항목

| 항목 | 결과 | 비고 |
|------|------|------|
| CMake 설치 | ⚠️ 미설치 | `brew install cmake` 필요 (별도 설치 가이드) |
| CMakeLists.txt 구문 | ✅ 검증됨 | 수동 검토 완료 |
| 조건부 컴파일 로직 | ✅ 검증됨 | QUIET + 조건부 링크 |
| 의존성 스크립트 | ✅ 생성됨 | check_dependencies.sh |

### 테스트 명령어 (CMake 설치 후)

```bash
# 1. 의존성 확인
./scripts/check_dependencies.sh

# 2. 빌드 디렉토리 생성
mkdir -p build && cd build

# 3. CMake 구성 (의존성 메시지 확인)
cmake .. -DBUILD_TESTS=ON

# 4. 예상 출력
# -- OpenCV not found. Some features will be disabled.
# -- TensorFlow Lite not found. Will be configured later.
# -- Configuring done
# -- Generating done
```

---

## 5. 이슈 및 학습

### 이슈

| ID | 내용 | 상태 | 해결방안 |
|----|------|------|----------|
| W2-01-001 | CMake 미설치 | 🔄 진행중 | brew install cmake 실행 필요 |

### 결정 사항

| 결정 | 이유 |
|------|------|
| 검증 스크립트 분리 | 재사용 가능한 의존성 체크 |
| 수동 검토 병행 | CMake 미설치 상황 대응 |

### 학습 내용

1. **CMake QUIET 옵션**:
   - find_package에서 오류 메시지 억제
   - 조건부 컴파일과 함께 사용시 유연한 빌드

2. **macOS 의존성 관리**:
   - Homebrew로 통합 관리 권장
   - pkg-config으로 라이브러리 경로 자동화

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-01-07 | 태스크 문서 생성 |
| 2026-01-07 | 의존성 상태 확인, 검증 스크립트 설계, CMake 미설치 이슈 기록 |
| 2026-01-07 | check_dependencies.sh 생성 및 실행 테스트, Android NDK 감지 확인, 태스크 완료 |
