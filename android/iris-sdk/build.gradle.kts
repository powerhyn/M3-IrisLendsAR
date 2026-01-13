// IrisLensSDK Android - iris-sdk Module build.gradle.kts
//
// SDK 라이브러리 모듈 빌드 설정
// - JNI/CMake 네이티브 라이브러리 통합
// - AAR 패키징 설정
//
// @version 1.0.0

plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

// =============================================================================
// 버전 정보 (gradle.properties에서 가져옴)
// =============================================================================
val sdkVersionName: String by project
val sdkVersionCode: String by project

android {
    namespace = "com.irislenssdk"
    compileSdk = 34

    defaultConfig {
        minSdk = 24

        // AAR 버전 정보
        buildConfigField("String", "SDK_VERSION", "\"${findProperty("IRIS_SDK_VERSION_NAME") ?: "1.0.0"}\"")
        buildConfigField("int", "SDK_VERSION_CODE", "${findProperty("IRIS_SDK_VERSION_CODE") ?: "1"}")

        // NDK 빌드 설정
        ndk {
            // 지원 ABI (ARM64, ARMv7)
            // x86, x86_64는 에뮬레이터 전용으로 선택적 추가 가능
            abiFilters += listOf("arm64-v8a", "armeabi-v7a")
        }

        // CMake 설정
        externalNativeBuild {
            cmake {
                // CMake 빌드 인자
                arguments += listOf(
                    "-DANDROID_STL=c++_shared",
                    "-DANDROID_ARM_NEON=TRUE",
                    "-DBUILD_TESTS=OFF",
                    "-DBUILD_EXAMPLES=OFF",
                    "-DBUILD_SHARED_LIBS=OFF"
                )

                // C/C++ 컴파일 플래그
                cppFlags += listOf(
                    "-std=c++17",
                    "-frtti",
                    "-fexceptions",
                    "-Wall",
                    "-Wextra"
                )
            }
        }

        // 테스트 러너
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // 소비자 ProGuard 규칙
        consumerProguardFiles("consumer-rules.pro")
    }

    // =============================================================================
    // 빌드 타입 설정
    // =============================================================================
    buildTypes {
        debug {
            isMinifyEnabled = false

            // 디버그 빌드 CMake 옵션
            externalNativeBuild {
                cmake {
                    arguments += "-DCMAKE_BUILD_TYPE=Debug"
                    cppFlags += listOf("-O0", "-g", "-DDEBUG")
                }
            }
        }

        release {
            isMinifyEnabled = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )

            // 릴리스 빌드 CMake 옵션
            externalNativeBuild {
                cmake {
                    arguments += "-DCMAKE_BUILD_TYPE=Release"
                    cppFlags += listOf(
                        "-O3",
                        "-DNDEBUG",
                        "-ffunction-sections",
                        "-fdata-sections"
                    )
                }
            }
        }
    }

    // =============================================================================
    // CMake 네이티브 빌드 설정
    // =============================================================================
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    // =============================================================================
    // Java/Kotlin 설정
    // =============================================================================
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
        // Kotlin 추가 옵션
        freeCompilerArgs += listOf(
            "-Xjvm-default=all",
            "-opt-in=kotlin.RequiresOptIn"
        )
    }

    // =============================================================================
    // 빌드 기능
    // =============================================================================
    buildFeatures {
        buildConfig = true
        // AIDL (필요 시 활성화)
        // aidl = true
    }

    // =============================================================================
    // 소스 세트 설정
    // =============================================================================
    sourceSets {
        getByName("main") {
            // JNI 라이브러리 위치 (빌드 후 자동 생성)
            jniLibs.srcDirs("src/main/jniLibs")
        }
    }

    // =============================================================================
    // 패키징 옵션
    // =============================================================================
    packaging {
        // 네이티브 라이브러리 압축 (비압축 권장 - 더 빠른 로딩)
        jniLibs {
            useLegacyPackaging = false
            // 중복 라이브러리 제외
            excludes += listOf(
                "**/libc++_shared.so"  // 앱에서 제공해야 함
            )
        }

        resources {
            excludes += listOf(
                "/META-INF/{AL2.0,LGPL2.1}",
                "/META-INF/DEPENDENCIES",
                "/META-INF/LICENSE*",
                "/META-INF/NOTICE*"
            )
        }
    }

    // =============================================================================
    // Lint 설정
    // =============================================================================
    lint {
        abortOnError = false
        warningsAsErrors = false
        checkReleaseBuilds = true
    }

    // =============================================================================
    // 게시 설정 (AAR 빌드)
    // =============================================================================
    publishing {
        singleVariant("release") {
            withSourcesJar()
            withJavadocJar()
        }
    }
}

// =============================================================================
// 의존성
// =============================================================================
dependencies {
    // AndroidX Core
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.annotation:annotation:1.7.1")

    // Kotlin 코루틴 (비동기 처리)
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")

    // 테스트
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
}

// =============================================================================
// 빌드 정보 태스크
// =============================================================================
tasks.register("printNativeBuildInfo") {
    doLast {
        println("""
            |===========================================
            | IrisLensSDK Native Build Info
            |===========================================
            | SDK Version: ${findProperty("IRIS_SDK_VERSION_NAME") ?: "1.0.0"}
            | Namespace: com.irislenssdk
            | Min SDK: 24
            | Target SDK: 34
            | ABIs: arm64-v8a, armeabi-v7a
            | CMake Version: 3.22.1
            | C++ Standard: C++17
            |===========================================
        """.trimMargin())
    }
}
