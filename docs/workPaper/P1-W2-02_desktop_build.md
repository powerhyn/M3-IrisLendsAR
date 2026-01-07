# P1-W2-02: Desktop 빌드 검증

**태스크 ID**: P1-W2-02
**상태**: ✅ 완료
**시작일**: 2026-01-07
**완료일**: 2026-01-07

---

## 1. 계획

### 목표
macOS에서 Desktop 빌드 성공 및 라이브러리 생성 확인

### 산출물
- build/ 디렉토리 내 빌드 아티팩트
- libiris_sdk 라이브러리 파일

### 검증 기준
- CMake 구성 성공
- make/ninja 빌드 성공
- 라이브러리 파일 생성 확인

### 선행 조건
- CMake 설치 필요 ⚠️
- P1-W2-01 완료 ✅

---

## 2. 분석

### 빌드 환경

| 항목 | 현재 상태 |
|------|----------|
| OS | macOS Darwin 23.6.0 |
| Compiler | Apple Clang 16.0.0.16000026 |
| CMake | ✅ 4.2.1 |
| OpenCV | ⚠️ 미설치 (선택적) |
| TFLite | ⚠️ 미설치 (선택적) |

### 빌드 설정

```cmake
# 기본 빌드 (의존성 없이)
cmake .. -DBUILD_TESTS=ON -DBUILD_EXAMPLES=OFF

# 전체 빌드 (의존성 포함)
cmake .. -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON
```

---

## 3. 실행 내역

### 3.1 CMake 설치 확인

```bash
$ which cmake
/opt/homebrew/bin/cmake

$ cmake --version
cmake version 4.2.1
```

### 3.2 CMake 구성

```bash
$ mkdir -p build && cd build
$ cmake .. -DBUILD_TESTS=ON -DBUILD_EXAMPLES=OFF
```

**출력 결과:**
```
-- The CXX compiler identification is AppleClang 16.0.0.16000026
-- Setting build type to 'Release' as none was specified.
-- Platform: macOS
CMake Warning: OpenCV not found. Some features will be disabled.
-- TensorFlow Lite not found. Will be configured later.

-- === IrisLensSDK Configuration ===
--   Version:        0.1.0
--   Platform:       macos
--   Build Type:     Release
--   C++ Standard:   17
--   Shared Libs:    ON
--   Tests:          ON
--   Examples:       OFF
-- ================================

-- Configuring done (4.5s)
-- Generating done (0.0s)
```

### 3.3 빌드 실행

```bash
$ cmake --build . --parallel
```

**출력 결과:**
```
[ 25%] Building CXX object cpp/CMakeFiles/iris_sdk.dir/src/placeholder.cpp.o
[ 50%] Linking CXX shared library ../lib/libiris_sdk.dylib
[ 50%] Built target iris_sdk
[ 75%] Building CXX object cpp/tests/CMakeFiles/test_placeholder.dir/test_placeholder.cpp.o
[100%] Linking CXX executable ../../bin/test_placeholder
[100%] Built target test_placeholder
```

### 3.4 결과물 확인

```bash
$ ls -la lib/
libiris_sdk.0.1.0.dylib  (16,840 bytes)
libiris_sdk.0.dylib -> libiris_sdk.0.1.0.dylib
libiris_sdk.dylib -> libiris_sdk.0.dylib

$ file lib/libiris_sdk.dylib
Mach-O 64-bit dynamically linked shared library arm64

$ ./bin/test_placeholder
IrisLensSDK test placeholder
```

---

## 4. 검증 결과

### 검증 항목

| 항목 | 결과 | 비고 |
|------|------|------|
| CMake 설치 | ✅ 통과 | v4.2.1 |
| CMake 구성 | ✅ 통과 | 4.5초 |
| 빌드 실행 | ✅ 통과 | 100% 완료 |
| 라이브러리 생성 | ✅ 통과 | libiris_sdk.dylib (arm64) |
| 테스트 실행 | ✅ 통과 | test_placeholder 성공 |

### 실제 결과물

```
build/
├── CMakeCache.txt
├── CMakeFiles/
├── lib/
│   ├── libiris_sdk.0.1.0.dylib  (16.8KB)
│   ├── libiris_sdk.0.dylib -> libiris_sdk.0.1.0.dylib
│   └── libiris_sdk.dylib -> libiris_sdk.0.dylib
├── bin/
│   └── test_placeholder
└── Makefile
```

---

## 5. 이슈 및 학습

### 이슈

| ID | 내용 | 상태 | 해결방안 |
|----|------|------|----------|
| W2-02-001 | CMake 미설치 | ✅ 해결됨 | brew install cmake |

### 결정 사항

| 결정 | 이유 |
|------|------|
| 의존성 없이 기본 빌드 | 최소 요구사항으로 검증 |
| 조건부 기능 비활성화 | OpenCV/TFLite 없이 빌드 가능 |
| Release 빌드 기본값 | 성능 최적화된 바이너리 생성 |

### 학습 내용

1. **CMake 4.x 호환성**: 최신 CMake 4.2.1에서 정상 작동
2. **Mach-O arm64**: Apple Silicon (M1/M2/M3) 네이티브 빌드 확인
3. **심볼릭 링크**: SO 버전 관리 패턴 적용 (0.1.0 → 0 → latest)

### 빌드 명령어 요약

```bash
# 개발 환경 설정 (최초 1회)
./scripts/setup_env.sh

# 의존성 확인
./scripts/check_dependencies.sh

# 빌드
mkdir -p build && cd build
cmake .. -DBUILD_TESTS=ON
cmake --build . --parallel

# 테스트 실행
./bin/test_placeholder
```

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-01-07 | 태스크 문서 생성, 빌드 절차 문서화 |
| 2026-01-07 | CMake 설치 후 빌드 성공, 검증 완료 |
