# IrisLensSDK - Root CMakeLists.txt
# Real-time iris tracking and virtual lens overlay AR SDK

cmake_minimum_required(VERSION 3.18...4.0)

project(IrisLensSDK
    VERSION 0.1.0
    DESCRIPTION "Real-time iris tracking and virtual lens overlay AR SDK"
    LANGUAGES CXX
)

# =============================================================================
# Build Options
# =============================================================================
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_ANDROID "Build for Android (cross-compile)" OFF)

# =============================================================================
# Global Settings
# =============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Position independent code (required for shared libraries)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# =============================================================================
# Output Directories
# =============================================================================
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# =============================================================================
# Platform Detection
# =============================================================================
if(APPLE)
    message(STATUS "Platform: macOS")
    set(IRIS_SDK_PLATFORM "macos")
elseif(UNIX AND NOT APPLE)
    message(STATUS "Platform: Linux")
    set(IRIS_SDK_PLATFORM "linux")
elseif(WIN32)
    message(STATUS "Platform: Windows")
    set(IRIS_SDK_PLATFORM "windows")
elseif(ANDROID)
    message(STATUS "Platform: Android")
    set(IRIS_SDK_PLATFORM "android")
endif()

# =============================================================================
# Subdirectories
# =============================================================================
add_subdirectory(cpp)

# =============================================================================
# Project Info
# =============================================================================
message(STATUS "")
message(STATUS "=== IrisLensSDK Configuration ===")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  Platform:       ${IRIS_SDK_PLATFORM}")
message(STATUS "  Build Type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard:   ${CMAKE_CXX_STANDARD}")
message(STATUS "  Shared Libs:    ${BUILD_SHARED_LIBS}")
message(STATUS "  Tests:          ${BUILD_TESTS}")
message(STATUS "  Examples:       ${BUILD_EXAMPLES}")
message(STATUS "================================")
message(STATUS "")
