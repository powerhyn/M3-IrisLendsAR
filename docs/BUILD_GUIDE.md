# IrisLensSDK 빌드 가이드

**최종 업데이트**: 2026-01-08

이 문서는 프로젝트 빌드의 일관성을 유지하기 위한 가이드라인입니다.

---

## 0. Claude Code 작업 시 주의사항

> **대용량 작업 정책**: TensorFlow git clone (~400MB) 등 대용량 다운로드가 필요한 경우, Claude가 직접 실행하지 않고 사용자에게 요청합니다. 사용자가 직접 실행 후 결과를 알려주세요.

**대용량 작업 예시**:
- TensorFlow 소스 다운로드 (FetchContent)
- 초기 CMake 구성 (의존성 다운로드 포함)
- 클린 빌드 후 재구성

**Claude가 직접 수행 가능한 작업**:
- 기존 빌드 디렉토리에서 증분 빌드
- 테스트 실행
- 파일 편집 및 문서 업데이트

---

## 1. 빌드 환경 개요

### 1.1 디렉토리 구조

```
IrisLensSDK/
├── cpp/                          # C++ 코어 소스
│   ├── CMakeLists.txt           # 메인 CMake 설정
│   ├── cmake-build-debug/       # ⭐ CLion 빌드 디렉토리 (기본)
│   ├── build/                   # 터미널 빌드 디렉토리 (대체)
│   ├── include/
│   ├── src/
│   └── tests/
├── shared/
│   ├── models/                  # TFLite 모델 파일
│   └── test_data/               # 테스트 이미지
└── docs/
    └── BUILD_GUIDE.md           # 이 문서
```

### 1.2 빌드 도구

| 도구 | 버전 | 비고 |
|------|------|------|
| CMake | 4.1+ | FetchContent 사용 |
| Ninja | 1.11+ | CLion 기본 제너레이터 |
| Make | - | 터미널 빌드 대체용 |
| Clang | 16.0+ | Apple Silicon 네이티브 |

---

## 2. 빌드 디렉토리 정책

### 2.1 기본 원칙

> **⚠️ 핵심 규칙: 기존 빌드 디렉토리를 유지하고 재사용하세요.**

TensorFlow Lite 소스 다운로드에 약 400MB, 빌드에 10분 이상 소요되므로:
- 새 빌드 디렉토리 생성을 최소화
- CLion의 `cmake-build-debug` 디렉토리를 표준으로 사용
- 터미널 빌드 시에도 이 디렉토리 재사용 권장

### 2.2 권장 빌드 디렉토리

| 환경 | 디렉토리 | 제너레이터 | 비고 |
|------|----------|------------|------|
| CLion | `cpp/cmake-build-debug` | Ninja | **기본 권장** |
| 터미널 | `cpp/cmake-build-debug` | Ninja | CLion 디렉토리 재사용 |
| CI/CD | `cpp/build` | Unix Makefiles | 격리된 환경 |

### 2.3 제너레이터 충돌 방지

**문제**: 동일 디렉토리에서 다른 제너레이터(Ninja vs Make) 사용 시 에러 발생

```
CMake Error: Error: generator : Ninja
Does not match the generator used previously: Unix Makefiles
```

**해결책**:
1. 기존 제너레이터 유지 (권장)
2. 또는 빌드 디렉토리 삭제 후 재생성 (TFLite 재다운로드 필요)

---

## 3. CMake 설정 옵션

### 3.1 주요 옵션

```cmake
# TFLite 관련 (대용량 의존성)
-DIRIS_SDK_FETCH_TFLITE=ON|OFF   # TFLite FetchContent 사용 여부
-DTFLITE_FOUND=ON                # TFLite 사전 빌드 감지 (자동)

# 빌드 대상
-DBUILD_TESTS=ON                 # 테스트 빌드 (기본: ON)
-DBUILD_EXAMPLES=ON              # 예제 빌드

# 플랫폼
-DBUILD_ANDROID=ON               # Android NDK 빌드
-DBUILD_IOS=ON                   # iOS 빌드
```

### 3.2 TFLite 재다운로드 방지

기존 빌드가 있을 때 CMake 재구성 시:

```bash
cd cpp/cmake-build-debug
cmake .. -DIRIS_SDK_FETCH_TFLITE=OFF
```

### 3.3 캐시 변수 확인

```bash
# CMakeCache.txt에서 현재 설정 확인
grep -E "TFLITE|OPENCV|BUILD_" cpp/cmake-build-debug/CMakeCache.txt
```

---

## 4. 개발 시나리오별 빌드

### 4.1 CLion에서 개발 (권장)

1. CLion에서 `cpp/` 폴더 열기
2. CMake 프로필 자동 구성 (cmake-build-debug)
3. 빌드 타겟 선택 후 실행

```
# CLion에서 자동 설정되는 값
Build directory: cmake-build-debug
Generator: Ninja
Build type: Debug
```

### 4.2 터미널에서 테스트 실행

```bash
# CLion 빌드 디렉토리 재사용
cd cpp/cmake-build-debug

# 특정 테스트만 빌드
cmake --build . --target test_mediapipe_detector_integration

# 테스트 실행
./bin/test_mediapipe_detector_integration

# 또는 ctest로 전체 테스트
ctest --output-on-failure
```

### 4.3 클린 빌드 (주의)

```bash
# ⚠️ TFLite 재다운로드 필요 - 시간 소요 큼
cd cpp
rm -rf cmake-build-debug
mkdir cmake-build-debug && cd cmake-build-debug
cmake .. -G Ninja -DBUILD_TESTS=ON
cmake --build . --parallel
```

### 4.4 CI/CD 빌드

```bash
# 격리된 환경에서 새 빌드
cd cpp
mkdir -p build && cd build
cmake .. -DBUILD_TESTS=ON
make -j$(nproc)
ctest --output-on-failure
```

---

## 5. 의존성 관리

### 5.1 FetchContent 의존성

| 의존성 | 크기 | 빌드 시간 | 캐시 위치 |
|--------|------|-----------|-----------|
| TensorFlow Lite | ~400MB | 5-10분 | `_deps/tensorflow-*` |
| GoogleTest | ~10MB | 30초 | `_deps/googletest-*` |
| Abseil | ~50MB | 2분 | `_deps/abseil-*` |
| FlatBuffers | ~5MB | 30초 | `_deps/flatbuffers-*` |

### 5.2 사전 빌드된 라이브러리 확인

```bash
# TFLite 라이브러리 존재 여부
ls -la cpp/cmake-build-debug/lib/libtensorflow-lite.a

# Abseil 라이브러리들
ls cpp/cmake-build-debug/lib/libabsl_*.a | wc -l  # 약 85개
```

### 5.3 의존성 문제 해결

**TFLite 다운로드 실패 시**:
```bash
# 네트워크 재시도
cd cpp/cmake-build-debug/_deps
rm -rf tensorflow-*
cd ..
cmake ..
```

**사전 빌드된 라이브러리 사용**:
```bash
# 이미 libtensorflow-lite.a가 있으면 FetchContent 비활성화
cmake .. -DIRIS_SDK_FETCH_TFLITE=OFF
```

---

## 6. 테스트 컴파일 정의

### 6.1 조건부 컴파일

테스트 코드는 의존성 유무에 따라 조건부로 컴파일됩니다:

```cpp
#if defined(IRIS_SDK_HAS_TFLITE) && defined(IRIS_SDK_HAS_OPENCV)
// 실제 TFLite + OpenCV 통합 테스트
#else
TEST(..., SkippedWithoutDependencies) {
    GTEST_SKIP() << "TFLite or OpenCV not available";
}
#endif
```

### 6.2 테스트 활성화 상태 확인

```bash
# CMake 구성 메시지에서 확인
cmake .. 2>&1 | grep -E "TFLite|OpenCV"

# 예상 출력:
# -- OpenCV: FOUND
# -- Integration test: TFLite enabled (pre-built library found)
```

---

## 7. 트러블슈팅

### 7.1 "generator does not match" 에러

```bash
# 해결: 캐시 삭제 후 동일 제너레이터로 재구성
rm cpp/cmake-build-debug/CMakeCache.txt
rm -rf cpp/cmake-build-debug/CMakeFiles
cmake .. -G Ninja
```

### 7.2 TFLite git clone 실패

```
fatal: fetch-pack: invalid index-pack output
```

해결:
1. 네트워크 상태 확인
2. 기존 `_deps/tensorflow-*` 삭제 후 재시도
3. 또는 `-DIRIS_SDK_FETCH_TFLITE=OFF`로 사전 빌드 사용

### 7.3 테스트가 SKIP되는 경우

```
[ SKIPPED ] TFLite or OpenCV not available
```

해결:
1. `tests/CMakeLists.txt`에서 `IRIS_SDK_HAS_TFLITE` 정의 확인
2. `lib/libtensorflow-lite.a` 존재 여부 확인
3. OpenCV 설치 여부 확인: `brew install opencv`

### 7.4 링크 에러 (undefined symbols)

TFLite 의존성 라이브러리 누락 시:

```bash
# tests/CMakeLists.txt에서 의존성 자동 수집 코드 확인
# TFLITE_DEP_LIBS, ABSEIL_LIBS 변수가 올바르게 설정되었는지 확인
```

---

## 8. 버전 관리 정책

### 8.1 .gitignore 설정

```gitignore
# 빌드 디렉토리
cpp/build/
cpp/cmake-build-*/

# 의존성 캐시 (재다운로드 필요)
cpp/*/_deps/

# IDE 설정
.idea/
*.xcworkspace/
```

### 8.2 커밋하지 않는 파일

- `cmake-build-debug/` 전체
- `_deps/` 의존성 소스
- `lib/` 빌드된 라이브러리
- `bin/` 실행 파일

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-01-08 | 문서 생성 - 빌드 일관성 가이드 |
| 2026-01-08 | TFLite 빌드 이슈 및 해결책 추가 |
