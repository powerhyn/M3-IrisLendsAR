# CMakeLists.txt for IrisLensSDK JNI Bindings
#
# Android NDK 크로스 컴파일 설정 (ARM64, ARMv7)
#
# 사용법:
#   - Android Studio/Gradle에서 자동으로 호출됨
#   - 직접 빌드: cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
#                     -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24 ..
#
# @author IrisLensSDK Team
# @version 1.0.0

cmake_minimum_required(VERSION 3.22.1)
project(iris_jni VERSION 1.0.0 LANGUAGES CXX)

# ==============================================================================
# C++ 표준 설정
# ==============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ==============================================================================
# 빌드 타입 기본값
# ==============================================================================

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "IrisSDK JNI: Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "IrisSDK JNI: Android ABI: ${ANDROID_ABI}")
message(STATUS "IrisSDK JNI: Android Platform: ${ANDROID_PLATFORM}")

# ==============================================================================
# 경로 설정
# ==============================================================================

# 프로젝트 루트 (android/iris-sdk/src/main/cpp 기준으로 5단계 상위)
get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../.." ABSOLUTE)

# C++ 코어 라이브러리 경로
set(IRIS_SDK_CORE_DIR "${PROJECT_ROOT}/cpp")
set(IRIS_SDK_INCLUDE_DIR "${IRIS_SDK_CORE_DIR}/include")

# 공유 리소스 경로 (모델 파일 등)
set(IRIS_SDK_SHARED_DIR "${PROJECT_ROOT}/shared")

message(STATUS "IrisSDK JNI: Project root: ${PROJECT_ROOT}")
message(STATUS "IrisSDK JNI: Core include dir: ${IRIS_SDK_INCLUDE_DIR}")
message(STATUS "IrisSDK JNI: Shared dir: ${IRIS_SDK_SHARED_DIR}")

# ==============================================================================
# iris_sdk 코어 라이브러리 찾기
# ==============================================================================

# 코어 라이브러리가 이미 빌드되어 있는 경우
find_library(IRIS_SDK_CORE_LIB
    NAMES iris_sdk
    PATHS
        "${IRIS_SDK_CORE_DIR}/build/android/${ANDROID_ABI}"
        "${IRIS_SDK_CORE_DIR}/build/${CMAKE_BUILD_TYPE}"
        "${IRIS_SDK_CORE_DIR}/cmake-build-debug"
        "${IRIS_SDK_CORE_DIR}/cmake-build-release"
    NO_DEFAULT_PATH
)

# 빌드된 라이브러리가 없으면 코어를 서브디렉토리로 추가
if(NOT IRIS_SDK_CORE_LIB)
    message(STATUS "IrisSDK JNI: Core library not found, adding as subdirectory")

    # Android 빌드용 옵션 설정
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

    # Android에서는 TFLite FetchContent 비활성화 (빌드 시간 최적화)
    # TFLite는 사전 빌드된 AAR로 제공되거나 별도로 통합
    set(IRIS_SDK_FETCH_TFLITE OFF CACHE BOOL "" FORCE)
    set(IRIS_SDK_USE_SYSTEM_TFLITE OFF CACHE BOOL "" FORCE)

    add_subdirectory("${IRIS_SDK_CORE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/iris_sdk_core")
    set(IRIS_SDK_CORE_TARGET iris_sdk)
else()
    message(STATUS "IrisSDK JNI: Found core library: ${IRIS_SDK_CORE_LIB}")

    # 찾은 라이브러리를 imported target으로 설정
    add_library(iris_sdk STATIC IMPORTED)
    set_target_properties(iris_sdk PROPERTIES
        IMPORTED_LOCATION "${IRIS_SDK_CORE_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${IRIS_SDK_INCLUDE_DIR}"
    )
    set(IRIS_SDK_CORE_TARGET iris_sdk)
endif()

# ==============================================================================
# JNI 소스 파일
# ==============================================================================

set(JNI_SOURCES
    iris_jni.cpp
)

set(JNI_HEADERS
    jni_utils.h
)

# ==============================================================================
# JNI 공유 라이브러리 생성
# ==============================================================================

add_library(iris_jni SHARED
    ${JNI_SOURCES}
    ${JNI_HEADERS}
)

# ==============================================================================
# 타겟 설정
# ==============================================================================

target_include_directories(iris_jni
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${IRIS_SDK_INCLUDE_DIR}
)

# ==============================================================================
# 링크 라이브러리
# ==============================================================================

# Android 시스템 라이브러리 찾기
find_library(ANDROID_LOG_LIB log)
find_library(ANDROID_LIB android)

target_link_libraries(iris_jni
    PRIVATE
        ${IRIS_SDK_CORE_TARGET}
        ${ANDROID_LOG_LIB}
)

# android 라이브러리가 있으면 링크 (jnigraphics 등 사용 시 필요)
if(ANDROID_LIB)
    target_link_libraries(iris_jni PRIVATE ${ANDROID_LIB})
endif()

# ==============================================================================
# 컴파일 옵션
# ==============================================================================

target_compile_options(iris_jni
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Werror=return-type
        -fvisibility=hidden
        -fvisibility-inlines-hidden
)

# Release 빌드 최적화
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(iris_jni
        PRIVATE
            -O3
            -DNDEBUG
            -ffunction-sections
            -fdata-sections
    )

    # 링커 최적화
    target_link_options(iris_jni
        PRIVATE
            -Wl,--gc-sections
            -Wl,--strip-all
    )
endif()

# Debug 빌드 설정
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(iris_jni
        PRIVATE
            -O0
            -g
    )
endif()

# ==============================================================================
# JNI 심볼 내보내기
# ==============================================================================

# JNI 함수들의 심볼이 숨겨지지 않도록 설정
set_target_properties(iris_jni PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# ==============================================================================
# 설치 설정 (optional)
# ==============================================================================

# Android Gradle Plugin이 사용할 위치로 라이브러리 복사
# install(TARGETS iris_jni
#     LIBRARY DESTINATION lib/${ANDROID_ABI}
# )

# ==============================================================================
# 정보 출력
# ==============================================================================

message(STATUS "")
message(STATUS "=== IrisSDK JNI Build Configuration ===")
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  Android ABI:    ${ANDROID_ABI}")
message(STATUS "  Platform:       ${ANDROID_PLATFORM}")
message(STATUS "  C++ Standard:   ${CMAKE_CXX_STANDARD}")
message(STATUS "  Core library:   ${IRIS_SDK_CORE_TARGET}")
message(STATUS "  Project root:   ${PROJECT_ROOT}")
message(STATUS "  NDK version:    ${ANDROID_NDK_REVISION}")
message(STATUS "========================================")
message(STATUS "")

# ==============================================================================
# 에셋 복사 (모델 파일 등)
# ==============================================================================

# 모델 파일을 assets 디렉토리로 복사하는 커스텀 타겟
# Gradle에서 preBuild 태스크로 실행 권장
set(ASSETS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../assets")

# assets 디렉토리가 없으면 생성
file(MAKE_DIRECTORY "${ASSETS_DIR}")

message(STATUS "IrisSDK JNI: Assets directory: ${ASSETS_DIR}")
