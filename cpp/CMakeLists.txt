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
    # ④ W4-D: iris_detector.cpp / mediapipe_detector.cpp / inference_thread.cpp 제거
    #   (검출 인프라 물리 삭제 — 추적 외부화).
    src/lens_renderer.cpp
    src/lens_sku_metadata.cpp
    src/frame_processor.cpp
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
    # Jaw V-line warp control-point geometry (P8-W4-A — pure CPU, no GL)
    src/warp/jaw_warp_geometry.cpp
    # (P8-W2 제거) FreqSep 전용 부가 인프라: quality_metrics / ab_compare /
    #             release_gate / param_tuner — 곁가지 뷰티 제거로 dead.
    # Temporal stabilizer (P5-W1)
    src/temporal_stabilizer.cpp
    # Landmark injection boundary (ADR-0001 §6 — 추적 외부화 진입점)
    src/landmark_injection.cpp
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
    # Jaw V-line warp control-point geometry (P8-W4-A)
    include/iris_sdk/warp/jaw_warp_geometry.h
    # (P8-W2 제거) FreqSep 전용 부가 인프라 헤더 — 곁가지 뷰티 제거로 dead.
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
# TensorFlow Lite
# =============================================================================
# ④ W4-D: 검출 인프라(mediapipe_detector/inference_thread) 물리 제거로 TFLite 소비처가
#   0이 되었다. find_package/FetchContent/Android prebuilt/link 블록 전체를 삭제한다.
#   (third_party/tflite/ 디렉토리 산출물은 별도 git rm으로 정리.)

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
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "internal" EXCLUDE  # W4-A §6.4: internal/ 벤치 토글 헤더는 공개 패키징 표면이 아님 (JNI 전용 경계)
)
