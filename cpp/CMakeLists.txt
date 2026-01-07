# IrisLensSDK - C++ Core Engine CMakeLists.txt

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
    # src/mediapipe_detector.cpp
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

# TensorFlow Lite (optional, for MediaPipe model inference)
find_package(TFLite QUIET)
if(TFLite_FOUND)
    message(STATUS "TensorFlow Lite found")
    target_link_libraries(iris_sdk PRIVATE TFLite::TFLite)
    target_compile_definitions(iris_sdk PRIVATE IRIS_SDK_HAS_TFLITE)
else()
    message(STATUS "TensorFlow Lite not found. Will be configured later.")
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
