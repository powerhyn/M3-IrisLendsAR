#!/bin/bash
# IrisLensSDK - Android Build Script
# Usage: ./scripts/build_android.sh [ABI] [BUILD_TYPE]
#   ABI: arm64-v8a (default), armeabi-v7a, x86_64, x86, all
#   BUILD_TYPE: Release (default), Debug

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "======================================"
echo "IrisLensSDK Android Build"
echo "======================================"
echo ""

# =============================================================================
# Configuration
# =============================================================================

# Find Android NDK
if [[ -n "$ANDROID_NDK_HOME" ]]; then
    NDK_PATH="$ANDROID_NDK_HOME"
elif [[ -d "$HOME/Library/Android/sdk/ndk" ]]; then
    # Find latest NDK version
    NDK_PATH=$(ls -d "$HOME/Library/Android/sdk/ndk"/* 2>/dev/null | sort -V | tail -n1)
elif [[ -d "$ANDROID_HOME/ndk" ]]; then
    NDK_PATH=$(ls -d "$ANDROID_HOME/ndk"/* 2>/dev/null | sort -V | tail -n1)
else
    echo -e "${RED}Error: Android NDK not found${NC}"
    echo "Please set ANDROID_NDK_HOME or install NDK via Android Studio"
    exit 1
fi

echo "Using NDK: $NDK_PATH"

# CMake toolchain
TOOLCHAIN="$NDK_PATH/build/cmake/android.toolchain.cmake"
if [[ ! -f "$TOOLCHAIN" ]]; then
    echo -e "${RED}Error: CMake toolchain not found at $TOOLCHAIN${NC}"
    exit 1
fi

# Parameters
ABI="${1:-arm64-v8a}"
BUILD_TYPE="${2:-Release}"
MIN_SDK_VERSION=24

echo "ABI: $ABI"
echo "Build Type: $BUILD_TYPE"
echo "Min SDK: $MIN_SDK_VERSION"
echo ""

# =============================================================================
# Build Functions
# =============================================================================

build_for_abi() {
    local abi=$1
    local build_dir="$PROJECT_ROOT/build/android-$abi"

    echo ""
    echo "======================================"
    echo "Building for $abi"
    echo "======================================"

    mkdir -p "$build_dir"
    cd "$build_dir"

    cmake "$PROJECT_ROOT" \
        -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
        -DANDROID_ABI="$abi" \
        -DANDROID_PLATFORM="android-$MIN_SDK_VERSION" \
        -DANDROID_STL=c++_shared \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_ANDROID=ON

    cmake --build . --parallel

    echo -e "${GREEN}✅ Build successful for $abi${NC}"

    # Show output
    echo ""
    echo "Output files:"
    find . -name "*.so" -type f | head -10
}

# =============================================================================
# Main
# =============================================================================

# Check CMake
if ! command -v cmake &> /dev/null; then
    echo -e "${RED}Error: CMake not found${NC}"
    echo "Install with: brew install cmake"
    exit 1
fi

# Build
if [[ "$ABI" == "all" ]]; then
    for abi in arm64-v8a armeabi-v7a x86_64; do
        build_for_abi "$abi"
    done
else
    build_for_abi "$ABI"
fi

echo ""
echo "======================================"
echo "Build Complete!"
echo "======================================"
echo ""
echo "Output directories:"
ls -d "$PROJECT_ROOT/build/android-"* 2>/dev/null || echo "No build directories found"
echo ""
