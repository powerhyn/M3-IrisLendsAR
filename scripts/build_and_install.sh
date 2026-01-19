#!/bin/bash
#
# IrisLensSDK - Build and Install Script
#
# 네이티브 라이브러리 재빌드 + 앱 설치를 한 번에 수행
# versionCode를 자동으로 증가시켜 빌드 구분 가능
#
# Usage:
#   ./scripts/build_and_install.sh          # 빌드 + 설치
#   ./scripts/build_and_install.sh --no-increment  # versionCode 증가 없이 빌드
#   ./scripts/build_and_install.sh --clean  # 전체 클린 빌드
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ANDROID_DIR="$PROJECT_ROOT/android"
BUILD_GRADLE="$ANDROID_DIR/demo-app/build.gradle.kts"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 옵션 파싱
INCREMENT_VERSION=true
CLEAN_BUILD=false

for arg in "$@"; do
    case $arg in
        --no-increment)
            INCREMENT_VERSION=false
            ;;
        --clean)
            CLEAN_BUILD=true
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --no-increment  versionCode 증가 없이 빌드"
            echo "  --clean         전체 클린 빌드 (build 폴더 삭제)"
            echo "  --help, -h      도움말 표시"
            exit 0
            ;;
    esac
done

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}  IrisLensSDK Build & Install${NC}"
echo -e "${BLUE}=====================================${NC}"
echo ""

# 1. versionCode 증가
if [ "$INCREMENT_VERSION" = true ]; then
    echo -e "${YELLOW}[1/5] versionCode 증가 중...${NC}"

    # 현재 versionCode 추출
    CURRENT_VERSION=$(grep "versionCode = " "$BUILD_GRADLE" | head -1 | sed 's/.*versionCode = \([0-9]*\).*/\1/')
    NEW_VERSION=$((CURRENT_VERSION + 1))

    # versionCode 업데이트
    sed -i '' "s/versionCode = $CURRENT_VERSION/versionCode = $NEW_VERSION/" "$BUILD_GRADLE"

    echo -e "  ${GREEN}versionCode: $CURRENT_VERSION → $NEW_VERSION${NC}"
else
    echo -e "${YELLOW}[1/5] versionCode 유지${NC}"
    NEW_VERSION=$(grep "versionCode = " "$BUILD_GRADLE" | head -1 | sed 's/.*versionCode = \([0-9]*\).*/\1/')
    echo -e "  ${GREEN}versionCode: $NEW_VERSION${NC}"
fi

# 2. 네이티브 빌드 캐시 삭제 (C++ 재빌드 보장)
echo -e "${YELLOW}[2/5] 네이티브 빌드 캐시 삭제 중...${NC}"

# CMake 캐시 삭제
rm -rf "$ANDROID_DIR/iris-sdk/.cxx"
echo -e "  ${GREEN}iris-sdk/.cxx 삭제 완료${NC}"

# prebuild된 jniLibs 삭제 (있으면 CMake 빌드가 무시됨)
if [ -d "$ANDROID_DIR/iris-sdk/src/main/jniLibs" ]; then
    rm -rf "$ANDROID_DIR/iris-sdk/src/main/jniLibs"
    echo -e "  ${GREEN}jniLibs 삭제 완료 (CMake 빌드 우선)${NC}"
fi

# 3. 클린 빌드 (기본 활성화)
echo -e "${YELLOW}[3/5] 전체 클린 빌드 중...${NC}"
cd "$ANDROID_DIR"
./gradlew clean
echo -e "  ${GREEN}클린 완료${NC}"

# 4. 빌드
echo -e "${YELLOW}[4/5] 빌드 중... (네이티브 + 앱)${NC}"
cd "$ANDROID_DIR"
./gradlew :demo-app:assembleDebug

APK_PATH="$ANDROID_DIR/build/modules/demo-app/outputs/apk/debug/demo-app-debug-b${NEW_VERSION}.apk"

if [ -f "$APK_PATH" ]; then
    echo -e "  ${GREEN}빌드 완료: demo-app-debug-b${NEW_VERSION}.apk${NC}"
else
    echo -e "  ${RED}빌드 실패: APK 파일을 찾을 수 없습니다${NC}"
    exit 1
fi

# 5. 설치
echo -e "${YELLOW}[5/5] 앱 설치 중...${NC}"

# 기존 앱 삭제 (실패해도 계속 진행)
adb uninstall com.irislenssdk.demo 2>/dev/null || true

# 새 앱 설치
if adb install "$APK_PATH"; then
    echo -e "  ${GREEN}설치 완료!${NC}"
else
    echo -e "  ${RED}설치 실패: 디바이스 연결을 확인하세요${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}  빌드 완료!${NC}"
echo -e "${GREEN}=====================================${NC}"
echo -e "  APK: demo-app-debug-b${NEW_VERSION}.apk"
echo -e "  모드: $(grep 'useInferenceThread = ' "$ANDROID_DIR/demo-app/src/main/java/com/irislenssdk/demo/MainActivity.kt" | head -1 | grep -o 'true\|false' | head -1)"
echo ""
echo -e "${BLUE}벤치마크 파일명 예상:${NC}"
MODE=$(grep 'useInferenceThread = ' "$ANDROID_DIR/demo-app/src/main/java/com/irislenssdk/demo/MainActivity.kt" | head -1 | grep -o 'true\|false' | head -1)
if [ "$MODE" = "true" ]; then
    echo -e "  benchmark_thread_b${NEW_VERSION}_*.csv"
else
    echo -e "  benchmark_direct_b${NEW_VERSION}_*.csv"
fi
