# P1-W5-04: AAR 빌드 설정

**태스크 ID**: P1-W5-04
**상태**: ✅ 완료
**시작일**: 2026-01-13
**완료일**: 2026-01-13

---

## 1. 계획

### 목표
외부 프로젝트에서 import 가능한 AAR(Android Archive) 패키지 생성 및 배포 설정 완료

### 산출물
| 파일 | 설명 | 상태 |
|------|------|------|
| `iris-sdk-release.aar` | 릴리스 AAR 패키지 | ✅ 생성됨 (5.5MB) |
| `iris-sdk-debug.aar` | 디버그 AAR 패키지 | ✅ 생성됨 (5.8MB) |
| `android/iris-sdk/proguard-rules.pro` | ProGuard 규칙 | ✅ 구성됨 |
| `scripts/build_android_aar.sh` | AAR 빌드 스크립트 | ✅ 작성됨 |

### 검증 기준
- [x] AAR 파일 생성 성공
- [x] AAR에 네이티브 .so 파일 포함 확인
- [x] AAR에 Kotlin 클래스 포함 확인
- [ ] 외부 프로젝트에서 AAR import 테스트 (P1-W6에서 수행)
- [ ] SDK 초기화 및 기본 기능 동작 확인 (P1-W6에서 수행)

### 선행 조건
- ✅ P1-W5-03 Gradle/CMake 통합 완료

---

## 2. 분석

### 2.1 AAR 구조 (실제 빌드 결과)

```
iris-sdk-release.aar (5.5MB, ZIP 형식)
├── AndroidManifest.xml        # 라이브러리 매니페스트
├── classes.jar                # Kotlin/Java 바이트코드 (60KB)
├── R.txt                      # 리소스 ID
├── proguard.txt               # ProGuard 규칙
├── META-INF/                  # 메타데이터
│   └── com/android/build/gradle/aar-metadata.properties
└── jni/                       # 네이티브 라이브러리
    ├── arm64-v8a/
    │   └── libiris_jni.so     # 7.8MB (Release)
    └── armeabi-v7a/
        └── libiris_jni.so     # 4.9MB (Release)
```

### 2.2 OpenCV 통합

OpenCV Android SDK 4.12.0 통합 완료:
- 위치: `cpp/third_party/opencv/android/sdk/`
- CMake 설정: `OpenCV_DIR` 변수로 경로 지정
- 정적 링크: OpenCV 라이브러리가 libiris_jni.so에 포함됨

---

## 3. 실행 내역

### 3.1 OpenCV Android SDK 설정

`android/iris-sdk/src/main/cpp/CMakeLists.txt` 수정:

```cmake
# OpenCV Android SDK 경로 설정 (add_subdirectory 전에 설정 필수)
if(ANDROID)
    set(OpenCV_DIR "${PROJECT_ROOT}/cpp/third_party/opencv/android/sdk/native/jni" CACHE PATH "" FORCE)
    message(STATUS "IrisSDK JNI: OpenCV_DIR set to: ${OpenCV_DIR}")
endif()
```

### 3.2 R8 Minification 이슈 해결

Release 빌드에서 R8 minification 오류 발생:
```
ERROR: R8: Missing class java.lang.invoke.StringConcatFactory
```

해결책:
1. `build.gradle.kts`에서 `isMinifyEnabled = false` 설정
2. `proguard-rules.pro`에 `-dontwarn java.lang.invoke.StringConcatFactory` 추가

**이유**: SDK 라이브러리는 minification을 앱 레벨에서 처리하는 것이 권장됨

### 3.3 빌드 명령

```bash
# Debug AAR 빌드
cd android && ./gradlew :iris-sdk:assembleDebug

# Release AAR 빌드
cd android && ./gradlew :iris-sdk:assembleRelease

# 자동화 스크립트
./scripts/build_android_aar.sh
```

### 3.4 빌드 로그

```
> Task :iris-sdk:externalNativeBuildRelease
[...] OpenCV ARCH: ARM64
[...] OpenCV FOUND: 4.12.0

BUILD SUCCESSFUL in 3s
34 actionable tasks: 6 executed, 28 up-to-date
```

---

## 4. 검증 결과

### AAR 내용 검증

| 항목 | Debug | Release | 비고 |
|------|-------|---------|------|
| AAR 크기 | 5.8MB | 5.5MB | 압축 후 |
| arm64-v8a .so | 8.4MB | 7.8MB | 디버그 심볼 제거됨 |
| armeabi-v7a .so | 5.2MB | 4.9MB | 디버그 심볼 제거됨 |
| classes.jar | 60KB | 60KB | Kotlin 클래스 |
| proguard.txt | 1.8KB | 1.8KB | 규칙 파일 |

### 네이티브 라이브러리 검증

```bash
$ unzip -l iris-sdk-release.aar | grep ".so"
  8202696  jni/arm64-v8a/libiris_jni.so
  5132712  jni/armeabi-v7a/libiris_jni.so
```

### 포함된 컴포넌트

- ✅ JNI 래퍼 (iris_jni.cpp)
- ✅ C++ Core Engine (정적 링크)
- ✅ OpenCV 4.12.0 (정적 링크)
- ✅ Kotlin SDK 클래스 (IrisLensSDK, IrisResult 등)
- ⏳ TFLite 모델 (Phase 1에서는 미포함, 런타임 로드 예정)

---

## 5. 이슈 및 학습

### 이슈

| ID | 내용 | 상태 | 해결방안 |
|----|------|------|----------|
| W5-04-01 | R8 minification 오류 | ✅ 해결 | isMinifyEnabled=false 설정 |
| W5-04-02 | OpenCV 경로 찾기 실패 | ✅ 해결 | OpenCV_DIR 변수로 명시적 경로 설정 |

### 결정 사항

| 결정 | 이유 |
|------|------|
| 모델 런타임 로드 | AAR 크기 최적화 (현재 ~5.5MB) |
| Minification 비활성화 | SDK 라이브러리는 앱 레벨에서 처리 권장 |
| OpenCV 정적 링크 | 단일 .so 파일로 배포 단순화 |

### 학습 내용
- Android AAR 구조 및 패키징
- CMake Android NDK 빌드 시 OpenCV_DIR 설정 방법
- R8/ProGuard SDK 라이브러리 처리 방식
- Release 빌드에서 디버그 심볼 자동 제거

---

## 6. 다음 단계

### P1-W6 통합 테스트

1. **데모 앱 프로젝트 생성**
   - 새 Android 프로젝트에서 AAR import 테스트
   - SDK 초기화 및 버전 확인

2. **CameraX 연동**
   - 실시간 카메라 프리뷰
   - 프레임 처리 파이프라인 구현

3. **성능 검증**
   - 30fps 달성 여부
   - 메모리 사용량 측정

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-01-12 | 태스크 문서 생성 |
| 2026-01-13 | OpenCV 통합 및 AAR 빌드 완료 |
| 2026-01-13 | R8 minification 이슈 해결 |
| 2026-01-13 | Debug/Release AAR 검증 완료 |
