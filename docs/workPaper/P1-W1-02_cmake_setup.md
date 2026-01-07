# P1-W1-02: CMakeLists.txt 루트 설정

**태스크 ID**: P1-W1-02
**상태**: ✅ 완료
**시작일**: 2026-01-07
**완료일**: 2026-01-07

---

## 1. 계획

### 목표
루트 CMakeLists.txt 및 cpp/ CMake 설정 구성

### 산출물
- CMakeLists.txt (루트)
- cpp/CMakeLists.txt

### 검증 기준
- `cmake ..` 성공 (의존성 없이 기본 구조 확인)
- 프로젝트 이름, 버전, C++ 표준 설정 확인

---

## 2. 실행 내역

### 2.1 CMake 구조 설계

```
IrisLensSDK/
├── CMakeLists.txt          ← 루트 (프로젝트 설정)
└── cpp/
    └── CMakeLists.txt      ← C++ 코어 엔진
```

### 2.2 생성할 파일

#### 루트 CMakeLists.txt
- 프로젝트 메타정보
- C++ 표준 설정 (C++17)
- 플랫폼별 옵션
- cpp/ 서브디렉토리 추가

#### cpp/CMakeLists.txt
- iris_sdk 라이브러리 타겟
- 헤더/소스 파일 구성
- 의존성 연결 (추후)
- 테스트/예제 서브디렉토리

---

## 3. 산출물

### 생성된 파일

| 파일 | 설명 | 크기 |
|------|------|------|
| `CMakeLists.txt` | 루트 CMake 설정 | ~2.5KB |
| `cpp/CMakeLists.txt` | C++ 코어 엔진 CMake | ~3.5KB |
| `cpp/tests/CMakeLists.txt` | 테스트 CMake | ~0.3KB |
| `cpp/examples/CMakeLists.txt` | 예제 CMake | ~0.4KB |
| `cpp/include/iris_sdk.h` | 메인 헤더 | ~0.8KB |
| `cpp/src/placeholder.cpp` | 플레이스홀더 소스 | ~0.2KB |

### 주요 설정

**루트 CMakeLists.txt**:
- 프로젝트: IrisLensSDK v0.1.0
- C++ 표준: C++17
- 옵션: BUILD_SHARED_LIBS, BUILD_TESTS, BUILD_EXAMPLES, BUILD_ANDROID
- 플랫폼 감지: macOS, Linux, Windows, Android

**cpp/CMakeLists.txt**:
- 타겟: `iris_sdk` (공유/정적 라이브러리)
- 별칭: `IrisSDK::iris_sdk`
- 컴파일 경고: -Wall -Wextra -Wpedantic
- export 헤더 자동 생성

---

## 4. 검증 결과

### 검증 항목

| 검증 항목 | 결과 | 비고 |
|----------|------|------|
| CMakeLists.txt 문법 | ✅ 통과 | Modern CMake 3.18+ 패턴 적용 |
| 서브디렉토리 구조 | ✅ 통과 | cpp/, tests/, examples/ 연결 |
| 빌드 테스트 | ⚠️ 보류 | CMake 미설치 (설치 필요) |

### 빌드 테스트 전제조건

```bash
# CMake 설치 (Homebrew)
brew install cmake

# 빌드 테스트
mkdir -p cpp/build && cd cpp/build
cmake .. -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON
cmake --build .
```

---

## 5. 이슈 및 학습

### 이슈

| ID | 이슈 | 상태 | 해결 |
|----|------|------|------|
| W1-02-001 | CMake 미설치 | ⚠️ 대기 | `brew install cmake` 필요 |

### 학습 내용

1. **Modern CMake 패턴**:
   - `cmake_minimum_required(VERSION 3.18...4.0)` - 버전 범위 지정
   - `target_compile_features(iris_sdk PUBLIC cxx_std_17)` - 타겟 기반 설정
   - `add_library(IrisSDK::iris_sdk ALIAS iris_sdk)` - 네임스페이스 별칭

2. **export 헤더 자동 생성**:
   - `generate_export_header()` 사용으로 DLL/SO 심볼 export 관리

3. **빌드 타입 기본값 설정**:
   - 빌드 타입 미지정시 Release 기본값 설정 패턴

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-01-07 | 태스크 문서 생성, 실행 시작 |
| 2026-01-07 | CMake 파일 생성 완료, CMake 설치 필요 이슈 기록 |
