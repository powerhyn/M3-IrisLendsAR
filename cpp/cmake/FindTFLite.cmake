# FindTFLite.cmake
# Find TensorFlow Lite library
#
# This module defines:
#   TFLite_FOUND - System has TensorFlow Lite
#   TFLite_INCLUDE_DIRS - TensorFlow Lite include directories
#   TFLite_LIBRARIES - Libraries needed to use TensorFlow Lite
#   TFLite::TFLite - Imported target for TensorFlow Lite

# Try to find TensorFlow Lite in common locations

# Check for environment variable
if(DEFINED ENV{TFLITE_ROOT})
    set(TFLITE_ROOT_DIR $ENV{TFLITE_ROOT})
endif()

# Search paths
set(TFLITE_SEARCH_PATHS
    ${TFLITE_ROOT_DIR}
    ${CMAKE_SOURCE_DIR}/cpp/third_party/tflite
    /usr/local
    /usr
    /opt/local
    /opt/homebrew
)

# Find include directory
find_path(TFLite_INCLUDE_DIR
    NAMES tensorflow/lite/interpreter.h
    PATHS ${TFLITE_SEARCH_PATHS}
    PATH_SUFFIXES include
)

# Find library
find_library(TFLite_LIBRARY
    NAMES tensorflowlite tensorflow-lite
    PATHS ${TFLITE_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64
)

# Handle standard find_package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TFLite
    REQUIRED_VARS TFLite_LIBRARY TFLite_INCLUDE_DIR
)

if(TFLite_FOUND)
    set(TFLite_INCLUDE_DIRS ${TFLite_INCLUDE_DIR})
    set(TFLite_LIBRARIES ${TFLite_LIBRARY})

    # Create imported target
    if(NOT TARGET TFLite::TFLite)
        add_library(TFLite::TFLite UNKNOWN IMPORTED)
        set_target_properties(TFLite::TFLite PROPERTIES
            IMPORTED_LOCATION "${TFLite_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${TFLite_INCLUDE_DIR}"
        )
    endif()
endif()

mark_as_advanced(TFLite_INCLUDE_DIR TFLite_LIBRARY)
