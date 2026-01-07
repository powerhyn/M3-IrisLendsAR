# P1-W2-04: Android 크로스컴파일 테스트

**태스크 ID**: P1-W2-04
**상태**: ✅ 완료
**시작일**: 2026-01-07
**완료일**: 2026-01-07

---

## 1. 계획

### 목표
Android arm64-v8a 타겟으로 크로스컴파일 테스트

### 산출물
- libiris_sdk.so (Android arm64-v8a)
- 빌드 로그 및 결과

### 검증 기준
- CMake 구성 성공
- 크로스컴파일 빌드 성공
- .so 파일 생성 확인
- file 명령으로 아키텍처 검증

### 선행 조건
- CMake 설치 필요 ⚠️
- P1-W2-03 완료 ✅

---

## 2. 분석

### 크로스컴파일 환경

| 항목 | 값 |
|------|-----|
| Host | macOS Darwin 23.6.0 (x86_64/arm64) |
| Target | Android arm64-v8a |
| NDK | 27.0.12077973 |
| Min API | 24 |
| STL | c++_shared |

### 빌드 명령어

```bash
./scripts/build_android.sh arm64-v8a Release
```

---

## 3. 실행 내역

### 3.1 빌드 스크립트 실행

```bash
$ ./scripts/build_android.sh arm64-v8a Release
```

### 3.2 빌드 출력

```
======================================
IrisLensSDK Android Build
======================================

Using NDK: /Users/seokhahyeon/Library/Android/sdk/ndk/27.0.12077973
ABI: arm64-v8a
Build Type: Release
Min SDK: 24

======================================
Building for arm64-v8a
======================================

-- The CXX compiler identification is Clang 18.0.1
-- Platform: Linux
CMake Warning: OpenCV not found. Some features will be disabled.
-- TensorFlow Lite not found. Will be configured later.

-- === IrisLensSDK Configuration ===
--   Version:        0.1.0
--   Platform:       linux
--   Build Type:     Release
--   C++ Standard:   17
--   Shared Libs:    ON
--   Tests:          OFF
--   Examples:       OFF
-- ================================

-- Configuring done (1.2s)
-- Generating done (0.0s)

[ 50%] Building CXX object cpp/CMakeFiles/iris_sdk.dir/src/placeholder.cpp.o
[100%] Linking CXX shared library ../lib/libiris_sdk.so
[100%] Built target iris_sdk

✅ Build successful for arm64-v8a
```

### 3.3 결과물 검증

```bash
$ ls -la build/android-arm64-v8a/lib/libiris_sdk.so
-rwxr-xr-x  1 seokhahyeon  staff  6512  libiris_sdk.so

$ file build/android-arm64-v8a/lib/libiris_sdk.so
ELF 64-bit LSB shared object, ARM aarch64, version 1 (SYSV),
dynamically linked, BuildID[sha1]=dc6255e34473fd57...,
with debug_info, not stripped
```

---

## 4. 검증 결과

### 검증 항목

| 항목 | 결과 | 비고 |
|------|------|------|
| CMake 설치 | ✅ 통과 | v4.2.1 |
| 빌드 스크립트 | ✅ 통과 | scripts/build_android.sh |
| NDK toolchain | ✅ 통과 | NDK 27.0 + Clang 18.0.1 |
| 크로스컴파일 | ✅ 통과 | arm64-v8a 빌드 성공 |
| 바이너리 검증 | ✅ 통과 | ELF 64-bit ARM aarch64 |

### 실제 결과물

```
build/android-arm64-v8a/
├── CMakeCache.txt
├── CMakeFiles/
├── lib/
│   └── libiris_sdk.so     (6.5KB)
└── Makefile
```

### 바이너리 비교

| 플랫폼 | 파일명 | 크기 | 포맷 |
|--------|--------|------|------|
| macOS | libiris_sdk.dylib | 16.8KB | Mach-O arm64 |
| Android | libiris_sdk.so | 6.5KB | ELF aarch64 |

---

## 5. 이슈 및 학습

### 이슈

| ID | 내용 | 상태 | 해결방안 |
|----|------|------|----------|
| W2-04-001 | CMake 미설치 | ✅ 해결됨 | brew install cmake |
| W2-04-002 | NDK deprecation 경고 | ℹ️ 무시 | NDK toolchain의 CMake 버전 경고, 기능 문제 없음 |

### 결정 사항

| 결정 | 이유 |
|------|------|
| arm64-v8a 우선 테스트 | 최신 기기 타겟, 주력 ABI |
| Release 빌드 | 실제 배포 시나리오 검증 |
| NDK 27.0 사용 | 최신 Clang 18.0.1 컴파일러 |

### 학습 내용

1. **크로스컴파일 검증**:
   - `file` 명령으로 바이너리 아키텍처 확인
   - ELF 포맷: Android/Linux 공유 라이브러리
   - Mach-O 포맷: macOS/iOS 공유 라이브러리

2. **Android 라이브러리 배포**:
   - .so 파일을 jniLibs/{ABI}/ 디렉토리에 배치
   - AAR 패키징시 자동 포함

3. **CMake 4.x + NDK 호환성**:
   - NDK toolchain의 오래된 CMake 버전 호환성 경고 발생
   - 실제 빌드에는 영향 없음

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-01-07 | 태스크 문서 생성, 테스트 시나리오 준비 |
| 2026-01-07 | 크로스컴파일 성공, ELF arm64 바이너리 검증 완료 |
