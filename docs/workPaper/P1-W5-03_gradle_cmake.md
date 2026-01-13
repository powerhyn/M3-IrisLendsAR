# P1-W5-03: Gradle/CMake 통합

**태스크 ID**: P1-W5-03
**상태**: ✅ 완료
**시작일**: 2026-01-13
**완료일**: 2026-01-13

---

## 1. 계획

### 목표
Android Gradle 빌드 시스템과 CMake 네이티브 빌드를 통합하여 단일 `./gradlew assembleDebug` 명령으로 전체 SDK 빌드 가능하게 설정

### 산출물
| 파일 | 설명 | 상태 |
|------|------|------|
| `android/settings.gradle.kts` | 프로젝트 설정 | ✅ |
| `android/build.gradle.kts` | 루트 빌드 스크립트 | ✅ |
| `android/iris-sdk/build.gradle.kts` | 라이브러리 모듈 빌드 | ✅ |
| `android/iris-sdk/src/main/cpp/CMakeLists.txt` | JNI CMake 설정 (업데이트) | ✅ |
| `android/gradle.properties` | Gradle 속성 | ✅ |
| `android/demo-app/build.gradle.kts` | 데모 앱 빌드 | ✅ |
| `android/iris-sdk/src/main/AndroidManifest.xml` | SDK 매니페스트 | ✅ |
| `android/demo-app/src/main/AndroidManifest.xml` | 앱 매니페스트 | ✅ |
| `android/iris-sdk/proguard-rules.pro` | ProGuard 규칙 | ✅ |
| `android/iris-sdk/consumer-rules.pro` | 소비자 ProGuard 규칙 | ✅ |

### 검증 기준
- [ ] `./gradlew assembleDebug` 성공 (P1-W5-04에서 검증)
- [ ] `./gradlew assembleRelease` 성공 (P1-W5-04에서 검증)
- [ ] `libiris_jni.so` 생성 (arm64-v8a, armeabi-v7a) (P1-W5-04에서 검증)
- [ ] 코어 iris_sdk 라이브러리 링크 성공 (P1-W5-04에서 검증)

### 선행 조건
- P1-W5-02 Kotlin SDK 클래스 완료 ✅
- P1-W2-04 Android 크로스컴파일 검증 완료 ✅

---

## 2. 분석

### 2.1 프로젝트 구조 (생성됨)

```
android/
├── settings.gradle.kts          ✅ 생성
├── build.gradle.kts              ✅ 생성
├── gradle.properties             ✅ 생성
├── gradlew                       ✅ 생성
├── gradlew.bat                   ✅ 생성
├── local.properties.template     ✅ 생성
├── .gitignore                    ✅ 생성
│
├── gradle/
│   └── wrapper/
│       └── gradle-wrapper.properties  ✅ 생성
│
├── iris-sdk/                     # AAR 라이브러리 모듈
│   ├── build.gradle.kts          ✅ 생성
│   ├── proguard-rules.pro        ✅ 생성
│   ├── consumer-rules.pro        ✅ 생성
│   └── src/
│       └── main/
│           ├── AndroidManifest.xml    ✅ 생성
│           ├── java/com/irislenssdk/  (기존)
│           └── cpp/
│               └── CMakeLists.txt     ✅ 업데이트
│
└── demo-app/                     # 데모 앱 모듈
    ├── build.gradle.kts          ✅ 생성
    ├── proguard-rules.pro        ✅ 생성
    └── src/main/
        ├── AndroidManifest.xml   ✅ 생성
        ├── java/com/irislenssdk/demo/
        │   └── MainActivity.kt   ✅ 생성
        └── res/
            ├── layout/activity_main.xml  ✅ 생성
            ├── values/strings.xml        ✅ 생성
            ├── values/colors.xml         ✅ 생성
            ├── values/themes.xml         ✅ 생성
            └── drawable/                 ✅ 생성
```

### 2.2 Gradle 버전 및 플러그인

```kotlin
// 실제 구현된 버전 (JDK 21 호환을 위해 업데이트됨)
Gradle: 8.7
AGP: 8.5.0
Kotlin: 2.0.0
CMake: 3.22.1
NDK: 26.1.10909125 (자동 설치)
Min SDK: 24
Target/Compile SDK: 34
Java: 17/21
```

### 2.3 CMake 통합 전략

**선택: 소스 빌드 방식 (서브디렉토리)**
- 장점: 단일 빌드 시스템, ABI별 최적화, 자동 의존성 관리
- 코어 C++ 코드를 `add_subdirectory()`로 포함
- TFLite FetchContent는 Android에서 비활성화 (빌드 시간 최적화)

### 2.4 ABI 지원 범위

| ABI | 지원 | 설명 |
|-----|------|------|
| arm64-v8a | ✅ | 64비트 ARM (권장) |
| armeabi-v7a | ✅ | 32비트 ARM (레거시) |
| x86_64 | ❌ | 미포함 (필요시 추가 가능) |
| x86 | ❌ | 미지원 |

---

## 3. 실행 내역

### 3.1 settings.gradle.kts

```kotlin
// android/settings.gradle.kts
pluginManagement {
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "IrisLensSDK"
include(":iris-sdk")
include(":demo-app")
```

### 3.2 iris-sdk/build.gradle.kts 주요 설정

```kotlin
android {
    namespace = "com.irislenssdk"
    compileSdk = 34

    defaultConfig {
        minSdk = 24
        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a")
        }
        externalNativeBuild {
            cmake {
                arguments += listOf(
                    "-DANDROID_STL=c++_shared",
                    "-DANDROID_ARM_NEON=TRUE",
                    "-DBUILD_TESTS=OFF",
                    "-DBUILD_EXAMPLES=OFF",
                    "-DBUILD_SHARED_LIBS=OFF"
                )
                cppFlags += listOf("-std=c++17", "-frtti", "-fexceptions")
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}
```

### 3.3 CMakeLists.txt 업데이트 사항

1. CMake 버전을 3.22.1로 업데이트 (Android SDK 번들 버전)
2. 공유 리소스 경로 추가 (`IRIS_SDK_SHARED_DIR`)
3. Android TFLite FetchContent 비활성화
4. Assets 디렉토리 자동 생성
5. NDK 버전 정보 출력 추가

### 3.4 demo-app 구현

- CameraX 기반 카메라 프리뷰 준비
- Material Design 3 테마
- 권한 처리 로직 구현
- SDK 초기화 플레이스홀더

---

## 4. 검증 결과

### 검증 항목

| 항목 | 결과 | 비고 |
|------|------|------|
| 모든 Gradle 파일 생성 | ✅ | Kotlin DSL |
| 모든 AndroidManifest.xml 생성 | ✅ | SDK + Demo |
| ProGuard 규칙 생성 | ✅ | JNI 메서드 보존 |
| CMakeLists.txt 업데이트 | ✅ | Android 최적화 |
| Demo app UI 생성 | ✅ | 기본 레이아웃 |
| Gradle wrapper 실행 | ✅ | 8.7 다운로드 성공 |
| NDK 자동 설치 | ✅ | 26.1.10909125 |
| CMake 구성 (arm64-v8a) | ✅ | 설정 완료 |
| CMake 구성 (armeabi-v7a) | ✅ | 설정 완료 |
| 코어 SDK 컴파일 | ⚠️ | OpenCV 의존성 필요 (P1-W5-04) |

### 빌드 테스트 로그

```
✅ Gradle 8.7 다운로드 완료
✅ NDK 26.1.10909125 자동 설치
✅ CMake arm64-v8a 구성 성공
✅ CMake armeabi-v7a 구성 성공
⚠️ frame_processor.cpp: cv::Mat 불완전 타입 에러
   → OpenCV Android SDK 설치 필요 (P1-W5-04 범위)
```

> **다음 단계**: P1-W5-04에서 OpenCV Android SDK 통합 후 AAR 빌드 완료

---

## 5. 이슈 및 학습

### 이슈
| ID | 내용 | 상태 | 해결방안 |
|----|------|------|----------|
| 1 | gradle-wrapper.jar 바이너리 필요 | ✅ 해결 | 사용자가 수동 다운로드 |
| 2 | JDK 21 + AGP 8.2.0 호환성 | ✅ 해결 | AGP 8.5.0, Gradle 8.7로 업그레이드 |
| 3 | OpenCV 의존성 누락 | ⏳ 진행중 | P1-W5-04에서 Android OpenCV SDK 설치 |

### 결정 사항
| 결정 | 이유 |
|------|------|
| 소스 빌드 방식 | ABI별 최적화, 단일 빌드 시스템, 자동 의존성 |
| c++_shared STL | 여러 네이티브 라이브러리 공유 가능 |
| API 24 최소 | Android 7.0+, 64비트 지원, CameraX 호환 |
| TFLite FetchContent 비활성화 | 빌드 시간 최적화 (10분+ 절약) |
| AGP 8.5.0 + Gradle 8.7 | JDK 21 호환, 최신 기능 지원 |
| Kotlin 2.0.0 | 최신 안정 버전, AGP 8.5.0 호환 |

### 학습 내용
- Android Gradle CMake 통합 패턴
- NDK ABI 필터링 및 최적화 설정
- Kotlin DSL for Gradle 사용법
- ProGuard JNI 메서드 보존 규칙

---

## 6. 다음 단계

### P1-W5-04: AAR 빌드 및 테스트
1. `gradle wrapper` 실행으로 wrapper JAR 생성
2. `./gradlew assembleDebug` 빌드 검증
3. 생성된 AAR 패키지 검증
4. 네이티브 라이브러리 (.so) 확인

### 빌드 실행 방법

```bash
# 1. Android Studio에서 android/ 폴더를 프로젝트로 열기
#    - Android Studio가 자동으로 gradle wrapper 생성

# 2. 또는 터미널에서 (Gradle이 시스템에 설치된 경우)
cd android
gradle wrapper
./gradlew assembleDebug

# 3. local.properties 설정
cp local.properties.template local.properties
# SDK/NDK 경로 수정
```

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-01-12 | 태스크 문서 생성 |
| 2026-01-13 | Gradle/CMake 파일 구현 완료 |
| 2026-01-13 | Demo app 기본 구조 생성 |
| 2026-01-13 | 태스크 완료 |
