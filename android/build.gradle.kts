// IrisLensSDK Android - Root build.gradle.kts
//
// 루트 프로젝트 빌드 설정
// - 공통 플러그인 버전 관리
// - 전역 빌드 설정
//
// @version 1.0.0

plugins {
    id("com.android.application") version "8.5.0" apply false
    id("com.android.library") version "8.5.0" apply false
    id("org.jetbrains.kotlin.android") version "2.0.0" apply false
}

// =============================================================================
// 전역 프로젝트 설정
// =============================================================================

// 모든 서브프로젝트에 공통 설정 적용
subprojects {
    // 빌드 출력 디렉토리 설정
    layout.buildDirectory.set(file("${rootProject.layout.buildDirectory.get()}/modules/${project.name}"))
}

// =============================================================================
// Clean 태스크
// =============================================================================

tasks.register("clean", Delete::class) {
    delete(layout.buildDirectory)
}

// =============================================================================
// 빌드 정보 태스크
// =============================================================================

tasks.register("printBuildInfo") {
    doLast {
        println("""
            |===========================================
            | IrisLensSDK Android Build Info
            |===========================================
            | Root Project: ${rootProject.name}
            | Gradle Version: ${gradle.gradleVersion}
            | Android Gradle Plugin: 8.5.0
            | Kotlin Version: 2.0.0
            | Java Version: ${JavaVersion.current()}
            |===========================================
        """.trimMargin())
    }
}
