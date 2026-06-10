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
    src/lens_renderer.cpp
    src/lens_sku_metadata.cpp
    src/frame_processor.cpp
    src/inference_thread.cpp
    src/sdk_manager.cpp
    src/sdk_api.cpp
    src/sdk_api_v2.cpp
    src/beauty_filter.cpp
    src/beauty_roi_manager.cpp
    src/beauty_processor.cpp
    src/cpu_beauty_backend.cpp
    src/fast_guided_filter.cpp
    # Profiling and optimization
    src/profiler.cpp
    src/buffer_pool.cpp
    # GPU module
    src/gpu/render_context.cpp
    src/gpu/cpu_render_context.cpp
    src/gpu/shader_manager.cpp
    src/gpu/texture_pool.cpp
    src/gpu/gpu_beauty_backend.cpp
    src/gpu/gpu_lens_renderer.cpp
    src/gpu/eye_render_packet_adapter.cpp
    src/gpu/shader_sources.cpp
    src/gpu/skin_mask_geometry.cpp
    src/gpu/gpu_profiler.cpp
    # Warp module
    src/warp/grid_mesh.cpp
    src/warp/face_warp_controller.cpp
    # Quality metrics
    src/quality_metrics.cpp
    # A/B comparison framework
    src/ab_compare.cpp
    # Release gate
    src/release_gate.cpp
    # Parameter tuning
    src/param_tuner.cpp
    # Temporal stabilizer (P5-W1)
    src/temporal_stabilizer.cpp
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
    include/iris_sdk/beauty_roi_manager.h
    # Lens SKU metadata + baked limbal auto-detect (P6-W7)
    include/iris_sdk/lens_sku_metadata.h
    # Profiling and optimization headers
    include/iris_sdk/profiler.h
    include/iris_sdk/buffer_pool.h
    # GPU module headers
    include/iris_sdk/gpu/texture_handle.h
    include/iris_sdk/gpu/render_context.h
    include/iris_sdk/gpu/cpu_render_context.h
    include/iris_sdk/gpu/shader_manager.h
    include/iris_sdk/gpu/texture_pool.h
    include/iris_sdk/gpu/gpu_beauty_backend.h
    include/iris_sdk/gpu/gpu_profiler.h
    include/iris_sdk/gpu/eye_render_packet.h
    include/iris_sdk/gpu/eye_render_packet_adapter.h
    # Warp module headers
    include/iris_sdk/warp/grid_mesh.h
    include/iris_sdk/warp/face_warp_controller.h
    # Quality metrics headers
    include/iris_sdk/quality_metrics.h
    # A/B comparison framework headers
    include/iris_sdk/ab_compare.h
    # Release gate headers
    include/iris_sdk/release_gate.h
    # Parameter tuning headers
    include/iris_sdk/param_tuner.h
    # Temporal stabilizer headers (P5-W1)
    include/iris_sdk/temporal_stabilizer.h
)

# =============================================================================
# GPU Module (Android GLES)
# =============================================================================
# Android 빌드 시 GLESRenderContext 추가
if(ANDROID)
    list(APPEND IRIS_SDK_SOURCES src/gpu/gles_render_context.cpp)
    list(APPEND IRIS_SDK_HEADERS include/iris_sdk/gpu/gles_render_context.h)
endif()

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
    # GPU 모듈을 위한 GLES 라이브러리 링크
    find_library(EGL_LIBRARY EGL)
    find_library(GLESV3_LIBRARY GLESv3)
    if(EGL_LIBRARY AND GLESV3_LIBRARY)
        target_link_libraries(iris_sdk PRIVATE ${EGL_LIBRARY} ${GLESV3_LIBRARY})
        target_compile_definitions(iris_sdk PRIVATE IRIS_SDK_HAS_GLES)
        message(STATUS "OpenGL ES 3.x support enabled for GPU module")
    endif()
elseif(UNIX)
    target_compile_definitions(iris_sdk PRIVATE IRIS_SDK_PLATFORM_LINUX)
elseif(WIN32)
    target_compile_definitions(iris_sdk PRIVATE IRIS_SDK_PLATFORM_WINDOWS)
endif()

# =============================================================================
# Dependencies
# =============================================================================

# OpenCV (required)
find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs highgui)
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

# 0. Android: Pre-built TFLite 라이브러리 사용 (Bazel 빌드 결과물)
if(ANDROID AND NOT TFLITE_FOUND)
    set(TFLITE_PREBUILT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tflite")
    set(TFLITE_PREBUILT_LIB "${TFLITE_PREBUILT_DIR}/android/${ANDROID_ABI}/libtensorflowlite.so")
    set(TFLITE_PREBUILT_INCLUDE "${TFLITE_PREBUILT_DIR}/include")

    if(EXISTS "${TFLITE_PREBUILT_LIB}" AND EXISTS "${TFLITE_PREBUILT_INCLUDE}/tensorflow/lite/interpreter.h")
        message(STATUS "Found pre-built TensorFlow Lite for Android ${ANDROID_ABI}")
        message(STATUS "  - Library: ${TFLITE_PREBUILT_LIB}")
        message(STATUS "  - Headers: ${TFLITE_PREBUILT_INCLUDE}")

        # TFLite imported target 생성
        add_library(tflite_prebuilt SHARED IMPORTED)
        set_target_properties(tflite_prebuilt PROPERTIES
            IMPORTED_LOCATION "${TFLITE_PREBUILT_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${TFLITE_PREBUILT_INCLUDE}"
        )

        target_link_libraries(iris_sdk PRIVATE tflite_prebuilt)
        target_include_directories(iris_sdk PRIVATE "${TFLITE_PREBUILT_INCLUDE}")
        target_compile_definitions(iris_sdk PRIVATE IRIS_SDK_HAS_TFLITE)
        set(TFLITE_FOUND TRUE CACHE BOOL "TensorFlow Lite found and configured" FORCE)

        # GPU Delegate 라이브러리 확인 및 링크
        set(TFLITE_GPU_DELEGATE_LIB "${TFLITE_PREBUILT_DIR}/android/${ANDROID_ABI}/libtensorflowlite_gpu_delegate.so")
        if(EXISTS "${TFLITE_GPU_DELEGATE_LIB}")
            message(STATUS "Found pre-built TFLite GPU Delegate for Android ${ANDROID_ABI}")
            message(STATUS "  - GPU Library: ${TFLITE_GPU_DELEGATE_LIB}")

            add_library(tflite_gpu_delegate SHARED IMPORTED)
            set_target_properties(tflite_gpu_delegate PROPERTIES
                IMPORTED_LOCATION "${TFLITE_GPU_DELEGATE_LIB}"
            )

            target_link_libraries(iris_sdk PRIVATE tflite_gpu_delegate)
            target_compile_definitions(iris_sdk PRIVATE IRIS_SDK_HAS_GPU_DELEGATE)

            # EGL, GLESv2 링크 (Android GPU delegate 필수 의존성)
            find_library(EGL_LIB EGL)
            find_library(GLES_LIB GLESv2)
            if(EGL_LIB AND GLES_LIB)
                target_link_libraries(iris_sdk PRIVATE ${EGL_LIB} ${GLES_LIB})
                message(STATUS "  - Linked EGL and GLESv2 for GPU delegate")
            endif()
        else()
            message(STATUS "GPU Delegate not found (CPU only mode)")
            message(STATUS "  Expected: ${TFLITE_GPU_DELEGATE_LIB}")
        endif()
    else()
        message(STATUS "Pre-built TFLite not found for Android ${ANDROID_ABI}")
        message(STATUS "  Expected: ${TFLITE_PREBUILT_LIB}")
    endif()
endif()

# 1. 먼저 시스템에 설치된 TFLite 확인 (옵션이 켜져있을 때)
if(NOT TFLITE_FOUND AND IRIS_SDK_USE_SYSTEM_TFLITE)
    # pkg-config로 TFLite 찾기 시도
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(TFLITE_PKG QUIET tensorflowlite)
        if(TFLITE_PKG_FOUND)
            message(STATUS "TensorFlow Lite found via pkg-config: ${TFLITE_PKG_VERSION}")
            target_include_directories(iris_sdk PRIVATE ${TFLITE_PKG_INCLUDE_DIRS})
            target_link_libraries(iris_sdk PRIVATE ${TFLITE_PKG_LIBRARIES})
            target_compile_definitions(iris_sdk PRIVATE IRIS_SDK_HAS_TFLITE)
            set(TFLITE_FOUND TRUE CACHE BOOL "TensorFlow Lite found and configured" FORCE)
        endif()
    endif()

    # pkg-config 실패 시 직접 find_package 시도
    if(NOT TFLITE_FOUND)
        find_package(TFLite QUIET)
        if(TFLite_FOUND)
            message(STATUS "Using system-installed TensorFlow Lite")
            target_link_libraries(iris_sdk PRIVATE TFLite::TFLite)
            target_compile_definitions(iris_sdk PRIVATE IRIS_SDK_HAS_TFLITE)
            set(TFLITE_FOUND TRUE CACHE BOOL "TensorFlow Lite found and configured" FORCE)
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
                    set(TFLITE_FOUND TRUE CACHE BOOL "TensorFlow Lite found and configured" FORCE)
                    break()
                endif()
            endif()
        endforeach()
    endif()
endif()

# 1.5 빌드 디렉토리에 사전 빌드된 라이브러리 확인 (FetchContent로 이미 빌드된 경우)
if(NOT TFLITE_FOUND)
    if(EXISTS "${CMAKE_BINARY_DIR}/lib/libtensorflow-lite.a" OR TARGET tensorflow-lite)
        message(STATUS "Found pre-built TensorFlow Lite in build directory")

        # tensorflow-lite 타겟이 있으면 사용
        if(TARGET tensorflow-lite)
            target_link_libraries(iris_sdk PRIVATE tensorflow-lite)
            message(STATUS "  - Using tensorflow-lite target")
        else()
            # 타겟이 없으면 라이브러리 파일 직접 링크
            # 주의: libfft2d_fft4f2d.a와 libfft2d_fftsg.a는 중복 심볼 충돌이 있음
            #       libfft2d_fftsg*.a만 사용 (TFLite 표준 구성)
            file(GLOB TFLITE_DEP_LIBS
                "${CMAKE_BINARY_DIR}/lib/libtensorflow-lite.a"
                "${CMAKE_BINARY_DIR}/lib/libflatbuffers.a"
                "${CMAKE_BINARY_DIR}/lib/libfft2d_fftsg.a"
                "${CMAKE_BINARY_DIR}/lib/libfft2d_fftsg2d.a"
                "${CMAKE_BINARY_DIR}/lib/libfft2d_fftsg3d.a"
                "${CMAKE_BINARY_DIR}/lib/libfft2d_alloc.a"
                "${CMAKE_BINARY_DIR}/lib/libfft2d_shrtdct.a"
                "${CMAKE_BINARY_DIR}/lib/libfarmhash.a"
                "${CMAKE_BINARY_DIR}/lib/libruy*.a"
                "${CMAKE_BINARY_DIR}/lib/libcpuinfo*.a"
                "${CMAKE_BINARY_DIR}/lib/libXNNPACK.a"
                "${CMAKE_BINARY_DIR}/lib/libpthreadpool.a"
            )
            file(GLOB ABSEIL_LIBS "${CMAKE_BINARY_DIR}/lib/libabsl_*.a")

            target_link_libraries(iris_sdk PRIVATE ${TFLITE_DEP_LIBS} ${ABSEIL_LIBS})
            list(LENGTH TFLITE_DEP_LIBS DEP_COUNT)
            list(LENGTH ABSEIL_LIBS ABSEIL_COUNT)
            message(STATUS "  - Linked ${DEP_COUNT} TFLite libraries")
            message(STATUS "  - Linked ${ABSEIL_COUNT} Abseil libraries")
        endif()

        # TFLite include 디렉토리 추가 (여러 위치 검색)
        set(TFLITE_INCLUDE_FOUND FALSE)

        # 1. 현재 빌드 디렉토리
        if(EXISTS "${CMAKE_BINARY_DIR}/_deps/tensorflow-src")
            target_include_directories(iris_sdk PRIVATE "${CMAKE_BINARY_DIR}/_deps/tensorflow-src")
            set(TFLITE_INCLUDE_FOUND TRUE)
            message(STATUS "  - Added TFLite include: ${CMAKE_BINARY_DIR}/_deps/tensorflow-src")
        endif()

        # 2. 다른 빌드 디렉토리 검색 (기존 빌드에서 재사용)
        if(NOT TFLITE_INCLUDE_FOUND)
            foreach(ALT_BUILD_DIR "build_tflite" "build")
                set(ALT_TFLITE_SRC "${CMAKE_SOURCE_DIR}/${ALT_BUILD_DIR}/_deps/tensorflow-src")
                if(EXISTS "${ALT_TFLITE_SRC}/tensorflow/lite/interpreter.h")
                    target_include_directories(iris_sdk PRIVATE "${ALT_TFLITE_SRC}")
                    set(TFLITE_INCLUDE_FOUND TRUE)
                    message(STATUS "  - Added TFLite include (alt): ${ALT_TFLITE_SRC}")
                    break()
                endif()
            endforeach()
        endif()

        if(NOT TFLITE_INCLUDE_FOUND)
            message(WARNING "TFLite libraries found but headers missing. TFLite features disabled.")
        else()
            # TFLite 의존성 헤더 경로 추가
            # FlatBuffers
            if(EXISTS "${CMAKE_BINARY_DIR}/flatbuffers/include")
                target_include_directories(iris_sdk PRIVATE "${CMAKE_BINARY_DIR}/flatbuffers/include")
                message(STATUS "  - Added FlatBuffers include")
            endif()

            # Abseil
            if(EXISTS "${CMAKE_BINARY_DIR}/abseil-cpp")
                target_include_directories(iris_sdk PRIVATE "${CMAKE_BINARY_DIR}/abseil-cpp")
                message(STATUS "  - Added Abseil include")
            endif()

            # Eigen (일부 TFLite 기능에 필요)
            if(EXISTS "${CMAKE_BINARY_DIR}/eigen")
                target_include_directories(iris_sdk PRIVATE "${CMAKE_BINARY_DIR}/eigen")
                message(STATUS "  - Added Eigen include")
            endif()

            # RUY
            if(EXISTS "${CMAKE_BINARY_DIR}/ruy")
                target_include_directories(iris_sdk PRIVATE "${CMAKE_BINARY_DIR}/ruy")
                message(STATUS "  - Added RUY include")
            endif()

            target_compile_definitions(iris_sdk PRIVATE IRIS_SDK_HAS_TFLITE)
            set(TFLITE_FOUND TRUE CACHE BOOL "TensorFlow Lite found and configured" FORCE)
        endif()
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
    set(TFLITE_ENABLE_RESOURCE ON CACHE BOOL "Enable resource support" FORCE)
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

        set(TFLITE_FOUND TRUE CACHE BOOL "TensorFlow Lite found and configured" FORCE)
    else()
        message(WARNING "Failed to configure TensorFlow Lite target.")
    endif()
endif()

# TFLite 상태 요약
if(TFLITE_FOUND)
    set(TFLITE_FOUND TRUE CACHE BOOL "TensorFlow Lite found and configured" FORCE)
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
# P6-W3 §5.10 hand-off / P6-W4 §1.15: W8 Pupil material renderMask hook 옵션.
# =============================================================================
# OFF default — 프로덕션 바이너리에서 hook dead code 제거 + 셰이더 #ifdef 비활성.
# debug 벤치 시: cmake .. -DIRIS_SDK_RENDER_MASK_HOOK_ENABLED=ON
# TODO(P6-W4): 셰이더 측 #define RENDER_MASK_HOOK_ENABLED prepend 미구현.
#              현 단계는 C++ 빌드 매크로만 정의 — W8 활성 시 buildLensOverlayFragment()
#              dynamic prepend 추가 (spec 5번 ⚠️ 안전 경로).
option(IRIS_SDK_RENDER_MASK_HOOK_ENABLED "Enable W8 Pupil material renderMask hook (debug bench only)" OFF)
if(IRIS_SDK_RENDER_MASK_HOOK_ENABLED)
    target_compile_definitions(iris_sdk PRIVATE IRIS_SDK_RENDER_MASK_HOOK_ENABLED=1)
    message(STATUS "P6-W4: RENDER_MASK_HOOK_ENABLED debug build (shader prepend TODO)")
endif()

# =============================================================================
# Tests
# =============================================================================
if(BUILD_TESTS)
    target_compile_definitions(iris_sdk PRIVATE IRIS_SDK_TEST_HOOKS)
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
