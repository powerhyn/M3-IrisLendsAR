// IrisLensSDK Android - settings.gradle.kts
//
// Gradle 빌드 설정 파일
// - 플러그인 저장소 설정
// - 의존성 해결 모드 설정
// - 프로젝트 모듈 정의
//
// @version 1.0.0

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

// SDK 라이브러리 모듈
include(":iris-sdk")

// 데모 애플리케이션 모듈
include(":demo-app")
