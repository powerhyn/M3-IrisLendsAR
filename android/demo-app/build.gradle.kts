// IrisLensSDK Android - Demo App build.gradle.kts
//
// 데모 애플리케이션 빌드 설정
// - IrisLensSDK 통합 데모
// - CameraX 기반 카메라 프리뷰
//
// @version 1.0.0

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.irislenssdk.demo"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.irislenssdk.demo"
        minSdk = 24
        targetSdk = 34
        versionCode = 284
        versionName = "1.0.1"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // NDK ABI 필터 (SDK와 동일)
        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a")
        }
    }

    buildTypes {
        debug {
            isDebuggable = true
            isMinifyEnabled = false
        }

        release {
            isMinifyEnabled = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        viewBinding = true
        buildConfig = true
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }

    // APK 파일명에 빌드 번호 포함
    applicationVariants.all {
        val variant = this
        variant.outputs.all {
            val output = this as com.android.build.gradle.internal.api.BaseVariantOutputImpl
            val versionCode = variant.versionCode
            val buildType = variant.buildType.name
            output.outputFileName = "demo-app-${buildType}-b${versionCode}.apk"
        }
    }
}

dependencies {
    // IrisLensSDK (로컬 모듈)
    implementation(project(":iris-sdk"))

    // MediaPipe Tasks Vision (Face Landmarker): W4-C에서 iris-sdk로 이관 — api()로 전이 제공.
    //   데모(AbMeasure A/B 하니스·MediaPipeBenchmarkActivity)는 transitive로 계속 사용.
    //   버전 고정(0.10.35)은 iris-sdk/build.gradle.kts에서 단일 관리(ADR-0001 §5).

    // AndroidX Core
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("androidx.activity:activity-ktx:1.8.2")
    implementation("androidx.fragment:fragment-ktx:1.6.2")

    // Material Design
    implementation("com.google.android.material:material:1.11.0")

    // ConstraintLayout
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")

    // CameraX (카메라 프리뷰 및 분석)
    // ④ W4-E: 1.3.1 → 1.4.2 — camera-core 1.3.1의 libimage_processing_util_jni.so가
    // 4KB(2**12) 정렬이라 16KB 페이지 미준수. 1.4.2는 16KB(2**14) 정렬(objdump 확인).
    val cameraxVersion = "1.4.2"
    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
    implementation("androidx.camera:camera-view:$cameraxVersion")

    // Kotlin 코루틴
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")

    // Lifecycle
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.7.0")

    // 테스트
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
}
