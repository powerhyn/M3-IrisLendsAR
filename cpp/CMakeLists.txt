# IrisLensSDK - C++ Core Engine CMakeLists.txt

# =============================================================================
# Standalone Build Support (CLion에서 cpp 폴더만 열 때)
# =============================================================================
# 독립 실행 시에만 필수 설정 추가 (서브프로젝트로 사용될 때는 스킵)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.18)

    # 오래된 서드파티 라이브러리 호환성 (FP16, XNNPACK 등)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "Minimum CMake policy version for legacy deps")

    project(IrisSDK_Core
        VERSION 0.1.0
        DESCRIPTION "IrisLensSDK C++ Core Engine"
        LANGUAGES CXX
    )

    # 독립 빌드 옵션
    option(BUILD_TESTS "Build unit tests" ON)
    option(BUILD_EXAMPLES "Build examples" ON)

    # C++ 표준 설정
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)

    # 출력 디렉토리
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

    # 컴파일 명령어 내보내기 (IDE 지원)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    message(STATUS "Building IrisSDK Core as standalone project")
endif()

# =============================================================================
# CMake Module Path
# =============================================================================
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# =============================================================================
# Source Files
# =============================================================================
set(IRIS_SDK_SOURCES
    # Core implementation
    src/iris_detector.cpp
    src/mediapipe_detector.cpp
    # src/lens_renderer.cpp
    # src/frame_processor.cpp
    # src/sdk_manager.cpp
    # src/sdk_api.cpp
)

set(IRIS_SDK_HEADERS
    # Public headers (to be added)
    # include/iris_sdk/iris_detector.h
    # include/iris_sdk/mediapipe_detector.h
    # include/iris_sdk/lens_renderer.h
    # include/iris_sdk/frame_processor.h
    # include/iris_sdk/sdk_manager.h
    # include/iris_sdk/sdk_api.h
    # include/iris_sdk.h
)

# =============================================================================
# Library Target
# =============================================================================
# Create placeholder source for initial build test
file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/src/placeholder.cpp
"// Placeholder source file for initial build test
// This will be replaced with actual implementation
namespace iris_sdk {
    const char* get_version() { return \"0.1.0\"; }
}
")

add_library(iris_sdk
    src/placeholder.cpp
    ${IRIS_SDK_SOURCES}
)

# Create alias for consistent naming
add_library(IrisSDK::iris_sdk ALIAS iris_sdk)

# =============================================================================
# Include Directories
# =============================================================================
target_include_directories(iris_sdk
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# =============================================================================
# Compile Features
# =============================================================================
target_compile_features(iris_sdk PUBLIC cxx_std_17)

# =============================================================================
# Compile Definitions
# =============================================================================
target_compile_definitions(iris_sdk
    PRIVATE
        IRIS_SDK_VERSION="${PROJECT_VERSION}"
        $<$<CONFIG:Debug>:IRIS_SDK_DEBUG>
)

# Platform-specific definitions
if(APPLE)
    target_compile_definitions(iris_sdk PRIVATE IRIS_SDK_PLATFORM_MACOS)
elseif(ANDROID)
    target_compile_definitions(iris_sdk PRIVATE IRIS_SDK_PLATFORM_ANDROID)
elseif(UNIX)
    target_compile_definitions(iris_sdk PRIVATE IRIS_SDK_PLATFORM_LINUX)
elseif(WIN32)
    target_compile_definitions(iris_sdk PRIVATE IRIS_SDK_PLATFORM_WINDOWS)
endif()

# =============================================================================
# Dependencies
# =============================================================================

# OpenCV (required)
find_package(OpenCV QUIET COMPONENTS core imgproc highgui)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    target_link_libraries(iris_sdk PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(iris_sdk PRIVATE IRIS_SDK_HAS_OPENCV)
else()
    message(WARNING "OpenCV not found. Some features will be disabled.")
endif()

# =============================================================================
# TensorFlow Lite (FetchContent with fallback to find_package)
# =============================================================================
#
# 빌드 옵션:
#   -DIRIS_SDK_USE_SYSTEM_TFLITE=ON   : 시스템 설치된 TFLite 우선 사용 (기본값)
#   -DIRIS_SDK_TFLITE_ENABLE_XNNPACK=ON : XNNPACK CPU 가속 활성화 (기본값)
#   -DIRIS_SDK_FETCH_TFLITE=ON        : FetchContent로 TFLite 다운로드 (기본값 OFF, 느림)
#   -DTFLITE_VERSION="2.16.1"         : TensorFlow Lite 버전 지정
#
# 참고: FetchContent는 TensorFlow 전체를 다운로드하므로 초기 빌드에 10-30분 소요됩니다.
#       빠른 빌드를 위해 시스템에 TFLite를 설치하는 것을 권장합니다:
#       - macOS: brew install tensorflow-lite
#       - Ubuntu: apt install libtensorflow-lite-dev
# =============================================================================

option(IRIS_SDK_USE_SYSTEM_TFLITE "Use system-installed TensorFlow Lite if available" ON)
option(IRIS_SDK_TFLITE_ENABLE_XNNPACK "Enable XNNPACK delegate for TFLite acceleration" ON)
option(IRIS_SDK_FETCH_TFLITE "Download and build TFLite via FetchContent (slow, use as fallback)" OFF)

set(TFLITE_FOUND FALSE)

# 1. 먼저 시스템에 설치된 TFLite 확인 (옵션이 켜져있을 때)
if(IRIS_SDK_USE_SYSTEM_TFLITE)
    # pkg-config로 TFLite 찾기 시도
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(TFLITE_PKG QUIET tensorflowlite)
        if(TFLITE_PKG_FOUND)
            message(STATUS "TensorFlow Lite found via pkg-config: ${TFLITE_PKG_VERSION}")
            target_include_directories(iris_sdk PRIVATE ${TFLITE_PKG_INCLUDE_DIRS})
            target_link_libraries(iris_sdk PRIVATE ${TFLITE_PKG_LIBRARIES})
            target_compile_definitions(iris_sdk PRIVATE IRIS_SDK_HAS_TFLITE)
            set(TFLITE_FOUND TRUE)
        endif()
    endif()

    # pkg-config 실패 시 직접 find_package 시도
    if(NOT TFLITE_FOUND)
        find_package(TFLite QUIET)
        if(TFLite_FOUND)
            message(STATUS "Using system-installed TensorFlow Lite")
            target_link_libraries(iris_sdk PRIVATE TFLite::TFLite)
            target_compile_definitions(iris_sdk PRIVATE IRIS_SDK_HAS_TFLITE)
            set(TFLITE_FOUND TRUE)
        endif()
    endif()

    # Homebrew 경로에서 직접 찾기 (macOS)
    if(NOT TFLITE_FOUND AND APPLE)
        # Homebrew Intel/ARM 경로 확인
        set(HOMEBREW_PATHS
            /opt/homebrew       # Apple Silicon
            /usr/local          # Intel
        )

        foreach(BREW_PREFIX ${HOMEBREW_PATHS})
            if(EXISTS "${BREW_PREFIX}/include/tensorflow/lite")
                set(TFLITE_INCLUDE_DIR "${BREW_PREFIX}/include")
                find_library(TFLITE_LIBRARY
                    NAMES tensorflowlite tensorflow-lite
                    PATHS "${BREW_PREFIX}/lib"
                    NO_DEFAULT_PATH
                )

                if(TFLITE_LIBRARY)
                    message(STATUS "TensorFlow Lite found in Homebrew: ${BREW_PREFIX}")
                    target_include_directories(iris_sdk PRIVATE ${TFLITE_INCLUDE_DIR})
                    target_link_libraries(iris_sdk PRIVATE ${TFLITE_LIBRARY})
                    target_compile_definitions(iris_sdk PRIVATE IRIS_SDK_HAS_TFLITE)
                    set(TFLITE_FOUND TRUE)
                    break()
                endif()
            endif()
        endforeach()
    endif()
endif()

# 2. 시스템에 없고 IRIS_SDK_FETCH_TFLITE=ON이면 FetchContent로 다운로드 및 빌드
if(NOT TFLITE_FOUND AND IRIS_SDK_FETCH_TFLITE)
    message(STATUS "============================================================")
    message(STATUS "TensorFlow Lite not found. Downloading via FetchContent...")
    message(STATUS "WARNING: This will download ~400MB and take 10-30 minutes!")
    message(STATUS "Consider installing TFLite system-wide for faster builds.")
    message(STATUS "============================================================")

    include(FetchContent)

    # TensorFlow Lite 버전 설정
    set(TFLITE_VERSION "2.16.1" CACHE STRING "TensorFlow Lite version to download")

    # TFLite 빌드 옵션 설정 (빌드 시간 최적화)
    set(TFLITE_ENABLE_RUY ON CACHE BOOL "Enable RUY matrix multiplication library" FORCE)
    set(TFLITE_ENABLE_RESOURCE OFF CACHE BOOL "Disable resource support" FORCE)
    set(TFLITE_ENABLE_NNAPI OFF CACHE BOOL "Disable NNAPI (Android only)" FORCE)
    set(TFLITE_ENABLE_GPU OFF CACHE BOOL "Disable GPU delegate" FORCE)
    set(TFLITE_ENABLE_MMAP ON CACHE BOOL "Enable memory-mapped model loading" FORCE)

    # XNNPACK 설정 (CPU 가속)
    if(IRIS_SDK_TFLITE_ENABLE_XNNPACK)
        set(TFLITE_ENABLE_XNNPACK ON CACHE BOOL "Enable XNNPACK delegate" FORCE)
        message(STATUS "XNNPACK delegate enabled for TFLite acceleration")
    else()
        set(TFLITE_ENABLE_XNNPACK OFF CACHE BOOL "Disable XNNPACK delegate" FORCE)
    endif()

    # 불필요한 기능 비활성화 (빌드 시간 단축)
    set(TFLITE_ENABLE_INSTALL OFF CACHE BOOL "Disable TFLite install targets" FORCE)
    set(TFLITE_KERNEL_TEST OFF CACHE BOOL "Disable kernel tests" FORCE)
    set(TENSORFLOW_SOURCE_DIR "" CACHE PATH "TensorFlow source directory (auto-downloaded)")

    # macOS 전용 설정
    if(APPLE)
        set(TFLITE_ENABLE_METAL OFF CACHE BOOL "Disable Metal delegate (use XNNPACK instead)" FORCE)
    endif()

    # FetchContent로 TensorFlow Lite 다운로드
    FetchContent_Declare(
        tensorflow
        GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
        GIT_TAG        v${TFLITE_VERSION}
        GIT_SHALLOW    TRUE
        GIT_PROGRESS   TRUE
        SOURCE_SUBDIR  tensorflow/lite
    )

    # TFLite 빌드 전 캐시 설정 (재빌드 방지)
    set(FETCHCONTENT_QUIET OFF)

    # 병렬 다운로드 최적화
    if(NOT DEFINED FETCHCONTENT_UPDATES_DISCONNECTED)
        set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
    endif()

    # TFLite 다운로드 및 MakeAvailable (최신 CMake 권장 방식)
    FetchContent_MakeAvailable(tensorflow)

    # TFLite 라이브러리 링크
    if(TARGET tensorflow-lite)
        message(STATUS "TensorFlow Lite v${TFLITE_VERSION} configured successfully")
        target_link_libraries(iris_sdk PRIVATE tensorflow-lite)
        target_compile_definitions(iris_sdk PRIVATE IRIS_SDK_HAS_TFLITE)

        # XNNPACK이 활성화된 경우 정의 추가
        if(IRIS_SDK_TFLITE_ENABLE_XNNPACK)
            target_compile_definitions(iris_sdk PRIVATE IRIS_SDK_HAS_XNNPACK)
        endif()

        # TFLite include 디렉토리 추가
        target_include_directories(iris_sdk PRIVATE
            ${tensorflow_SOURCE_DIR}
        )

        set(TFLITE_FOUND TRUE)
    else()
        message(WARNING "Failed to configure TensorFlow Lite target.")
    endif()
endif()

# TFLite 상태 요약
if(TFLITE_FOUND)
    message(STATUS "TensorFlow Lite: ENABLED")
    if(IRIS_SDK_TFLITE_ENABLE_XNNPACK)
        message(STATUS "  - XNNPACK acceleration: ENABLED")
    endif()
else()
    message(STATUS "TensorFlow Lite: DISABLED")
    message(STATUS "  To enable TFLite, either:")
    message(STATUS "    1. Install TFLite system-wide (recommended)")
    message(STATUS "    2. Set -DIRIS_SDK_FETCH_TFLITE=ON (slow, downloads ~400MB)")
endif()

# =============================================================================
# Compiler Warnings
# =============================================================================
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(iris_sdk PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
    )
elseif(MSVC)
    target_compile_options(iris_sdk PRIVATE /W4)
endif()

# =============================================================================
# Library Properties
# =============================================================================
set_target_properties(iris_sdk PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME "iris_sdk"
    DEBUG_POSTFIX "d"
)

# =============================================================================
# Export Symbols (for shared library)
# =============================================================================
include(GenerateExportHeader)
generate_export_header(iris_sdk
    EXPORT_FILE_NAME ${CMAKE_CURRENT_SOURCE_DIR}/include/iris_sdk/export.h
)

# =============================================================================
# Tests
# =============================================================================
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# =============================================================================
# Examples
# =============================================================================
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# =============================================================================
# Installation (optional)
# =============================================================================
install(TARGETS iris_sdk
    EXPORT IrisSDKTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)
