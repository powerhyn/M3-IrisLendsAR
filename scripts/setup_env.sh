#!/bin/bash
# IrisLensSDK - Development Environment Setup Script
# Usage: ./scripts/setup_env.sh

set -e

echo "======================================"
echo "IrisLensSDK Development Environment Setup"
echo "======================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
fi

echo "Detected OS: $OS"
echo ""

# =============================================================================
# macOS Setup (Homebrew)
# =============================================================================
if [[ "$OS" == "macos" ]]; then
    echo "Setting up for macOS..."
    echo ""

    # Check Homebrew
    if ! command -v brew &> /dev/null; then
        echo -e "${RED}Homebrew not found. Please install Homebrew first:${NC}"
        echo '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
        exit 1
    fi
    echo -e "${GREEN}✓ Homebrew found${NC}"

    # CMake
    echo ""
    echo "Checking CMake..."
    if ! command -v cmake &> /dev/null; then
        echo -e "${YELLOW}Installing CMake...${NC}"
        brew install cmake
    else
        CMAKE_VERSION=$(cmake --version | head -n1)
        echo -e "${GREEN}✓ CMake installed: $CMAKE_VERSION${NC}"
    fi

    # OpenCV
    echo ""
    echo "Checking OpenCV..."
    if ! brew list opencv &> /dev/null; then
        echo -e "${YELLOW}Installing OpenCV...${NC}"
        brew install opencv
    else
        OPENCV_VERSION=$(pkg-config --modversion opencv4 2>/dev/null || echo "version unknown")
        echo -e "${GREEN}✓ OpenCV installed: $OPENCV_VERSION${NC}"
    fi

    # TensorFlow Lite (via tensorflow)
    echo ""
    echo "Checking TensorFlow..."
    # TFLite는 별도 설치 필요 - 나중에 처리
    echo -e "${YELLOW}Note: TensorFlow Lite will be configured separately${NC}"

    # Android NDK (optional)
    echo ""
    echo "Checking Android NDK..."
    if [[ -z "$ANDROID_NDK_HOME" ]]; then
        echo -e "${YELLOW}ANDROID_NDK_HOME not set. Android build will not be available.${NC}"
        echo "To install: brew install android-ndk"
    else
        echo -e "${GREEN}✓ Android NDK found: $ANDROID_NDK_HOME${NC}"
    fi
fi

# =============================================================================
# Linux Setup (apt)
# =============================================================================
if [[ "$OS" == "linux" ]]; then
    echo "Setting up for Linux..."
    echo ""

    # Check if running as root or with sudo
    SUDO=""
    if [[ $EUID -ne 0 ]]; then
        SUDO="sudo"
    fi

    echo "Installing dependencies..."
    $SUDO apt-get update
    $SUDO apt-get install -y \
        cmake \
        build-essential \
        libopencv-dev \
        pkg-config

    echo -e "${GREEN}✓ Dependencies installed${NC}"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. cd cpp/build"
echo "2. cmake .. -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON"
echo "3. cmake --build ."
echo ""
echo "For Android development:"
echo "1. Install Android NDK"
echo "2. Set ANDROID_NDK_HOME environment variable"
echo "3. Use BUILD_ANDROID=ON flag"
echo ""
