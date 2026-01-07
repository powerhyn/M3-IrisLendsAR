# P1-W2-03: Android NDK 환경 설정

**태스크 ID**: P1-W2-03
**상태**: ✅ 완료
**시작일**: 2026-01-07
**완료일**: 2026-01-07

---

## 1. 계획

### 목표
Android NDK 크로스컴파일 환경 구성

### 산출물
- Android toolchain 설정
- CMake Android 빌드 옵션
- 빌드 스크립트 (scripts/build_android.sh)

### 검증 기준
- NDK toolchain 경로 확인
- CMake Android toolchain 연동

### 선행 조건
- Android NDK 설치 ✅

---

## 2. 분석

### 현재 NDK 상태

```bash
ANDROID_NDK_HOME=/Users/seokhahyeon/Library/Android/sdk/ndk/26.3.11579264
```

### Android 빌드 설정

**타겟 ABI**:
| ABI | 설명 | 우선순위 |
|-----|------|----------|
| arm64-v8a | 64-bit ARM (대부분 현대 기기) | 1순위 |
| armeabi-v7a | 32-bit ARM (구형 기기) | 2순위 |
| x86_64 | 64-bit x86 (에뮬레이터) | 3순위 |
| x86 | 32-bit x86 (구형 에뮬레이터) | 선택적 |

**최소 API 레벨**: 24 (Android 7.0 Nougat)

---

## 3. 실행 내역

### 3.1 NDK 경로 확인

```bash
$ ls ~/Library/Android/sdk/ndk/
26.3.11579264
27.0.12077973
```

**감지된 NDK**:
- 버전 26.3.11579264 ✅
- 버전 27.0.12077973 ✅
- 사용 예정: 27.0.12077973 (최신)

**Toolchain 경로**:
```
~/Library/Android/sdk/ndk/27.0.12077973/build/cmake/android.toolchain.cmake
```

### 3.2 Android 빌드 스크립트 생성

**생성된 파일**: `scripts/build_android.sh`

**주요 기능**:
```bash
# 사용법
./scripts/build_android.sh [ABI] [BUILD_TYPE]

# ABI 옵션
arm64-v8a     # 64-bit ARM (기본값)
armeabi-v7a   # 32-bit ARM
x86_64        # 64-bit x86 (에뮬레이터)
all           # 전체 ABI 빌드

# 빌드 타입
Release       # 최적화 빌드 (기본값)
Debug         # 디버그 빌드
```

**스크립트 특징**:
- NDK 자동 감지 (ANDROID_NDK_HOME, 기본 경로)
- 최신 NDK 버전 자동 선택
- 병렬 빌드 지원
- ABI별 빌드 디렉토리 분리

### 3.3 CMake toolchain 설정

**CMake 옵션**:
```cmake
-DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake
-DANDROID_ABI=arm64-v8a
-DANDROID_PLATFORM=android-24
-DANDROID_STL=c++_shared
-DBUILD_ANDROID=ON
```

**설정 설명**:
| 옵션 | 값 | 설명 |
|------|-----|------|
| ANDROID_ABI | arm64-v8a | 타겟 아키텍처 |
| ANDROID_PLATFORM | android-24 | 최소 API 레벨 (Android 7.0) |
| ANDROID_STL | c++_shared | 공유 C++ 라이브러리 |
| BUILD_ANDROID | ON | Android 빌드 플래그 |

---

## 4. 검증 결과

### 검증 항목

| 항목 | 결과 | 비고 |
|------|------|------|
| NDK 설치 | ✅ 확인 | 26.3, 27.0 버전 |
| Toolchain 경로 | ✅ 확인 | android.toolchain.cmake |
| 빌드 스크립트 | ✅ 생성 | scripts/build_android.sh |
| 환경변수 | ⚠️ 선택적 | ANDROID_NDK_HOME 미설정 (자동 감지) |

### 빌드 명령어

```bash
# 단일 ABI 빌드
./scripts/build_android.sh arm64-v8a Release

# 전체 ABI 빌드
./scripts/build_android.sh all Release
```

### 결과물 위치

```
build/
├── android-arm64-v8a/
│   └── cpp/
│       └── libiris_sdk.so
├── android-armeabi-v7a/
│   └── cpp/
│       └── libiris_sdk.so
└── android-x86_64/
    └── cpp/
        └── libiris_sdk.so
```

---

## 5. 이슈 및 학습

### 결정 사항

| 결정 | 이유 |
|------|------|
| 최소 API 24 | Android 7.0 Nougat (2016), 현재 점유율 충분 |
| c++_shared STL | 다른 라이브러리와 호환성 우선 |
| 자동 NDK 감지 | 환경 설정 부담 최소화 |

### 학습 내용

1. **Android NDK CMake 통합**:
   - `android.toolchain.cmake` 사용
   - ANDROID_* 변수로 타겟 설정
   - ABI별 빌드 분리 권장

2. **STL 선택**:
   - `c++_shared`: 공유 libc++, 다른 네이티브 라이브러리와 호환
   - `c++_static`: 정적 링크, 크기 최적화
   - `none`: STL 미사용

3. **ABI 우선순위**:
   - arm64-v8a: 주력 (최신 기기)
   - armeabi-v7a: 레거시 지원
   - x86_64: 에뮬레이터/Chromebook

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-01-07 | 태스크 문서 생성 |
| 2026-01-07 | NDK 환경 확인, build_android.sh 생성, 태스크 완료 |
