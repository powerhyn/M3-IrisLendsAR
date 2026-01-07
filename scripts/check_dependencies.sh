#!/bin/bash
# IrisLensSDK - Dependency Checker Script
# Usage: ./scripts/check_dependencies.sh

echo "======================================"
echo "IrisLensSDK Dependency Check"
echo "======================================"
echo ""

MISSING_REQUIRED=0
MISSING_OPTIONAL=0

# CMake (Required)
echo "=== Required Dependencies ==="
if command -v cmake &> /dev/null; then
    CMAKE_VER=$(cmake --version | head -n1)
    echo "✅ CMake: $CMAKE_VER"
else
    echo "❌ CMake: NOT INSTALLED"
    echo "   Install: brew install cmake"
    MISSING_REQUIRED=1
fi

# C++ Compiler
if command -v clang++ &> /dev/null; then
    CLANG_VER=$(clang++ --version | head -n1)
    echo "✅ Clang++: $CLANG_VER"
elif command -v g++ &> /dev/null; then
    GXX_VER=$(g++ --version | head -n1)
    echo "✅ g++: $GXX_VER"
else
    echo "❌ C++ Compiler: NOT FOUND"
    MISSING_REQUIRED=1
fi

echo ""
echo "=== Optional Dependencies ==="

# OpenCV
if command -v pkg-config &> /dev/null && pkg-config --exists opencv4 2>/dev/null; then
    OPENCV_VER=$(pkg-config --modversion opencv4)
    echo "✅ OpenCV: $OPENCV_VER"
elif brew list opencv &> /dev/null 2>&1; then
    echo "✅ OpenCV: installed via Homebrew"
else
    echo "⚠️  OpenCV: NOT INSTALLED (optional)"
    echo "   Install: brew install opencv"
    MISSING_OPTIONAL=1
fi

# TensorFlow Lite
TFLITE_PATHS=(
    "/opt/homebrew/include/tensorflow"
    "/usr/local/include/tensorflow"
    "$HOME/.local/include/tensorflow"
)
TFLITE_FOUND=false
for path in "${TFLITE_PATHS[@]}"; do
    if [[ -d "$path" ]]; then
        echo "✅ TFLite: Found at $path"
        TFLITE_FOUND=true
        break
    fi
done
if [[ "$TFLITE_FOUND" == "false" ]]; then
    echo "⚠️  TFLite: NOT INSTALLED (optional)"
    echo "   Will be configured separately"
    MISSING_OPTIONAL=1
fi

# Android NDK (for cross-compilation)
echo ""
echo "=== Android Development (Optional) ==="
if [[ -n "$ANDROID_NDK_HOME" ]] && [[ -d "$ANDROID_NDK_HOME" ]]; then
    echo "✅ Android NDK: $ANDROID_NDK_HOME"
elif [[ -n "$ANDROID_HOME" ]] && [[ -d "$ANDROID_HOME/ndk" ]]; then
    NDK_VERSION=$(ls "$ANDROID_HOME/ndk" 2>/dev/null | head -n1)
    if [[ -n "$NDK_VERSION" ]]; then
        echo "✅ Android NDK: $ANDROID_HOME/ndk/$NDK_VERSION"
    else
        echo "⚠️  Android NDK: NOT INSTALLED"
    fi
else
    echo "⚠️  Android NDK: NOT CONFIGURED"
    echo "   Set ANDROID_NDK_HOME environment variable"
fi

# Summary
echo ""
echo "======================================"
echo "Summary"
echo "======================================"

if [[ $MISSING_REQUIRED -eq 0 ]]; then
    echo "✅ All required dependencies installed"
    echo ""
    echo "Next steps:"
    echo "  mkdir -p build && cd build"
    echo "  cmake .. -DBUILD_TESTS=ON"
    echo "  cmake --build ."
    exit 0
else
    echo "❌ Missing required dependencies"
    echo ""
    echo "Run the following to install:"
    echo "  ./scripts/setup_env.sh"
    exit 1
fi
