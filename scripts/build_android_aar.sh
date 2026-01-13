#!/bin/bash
#
# IrisLensSDK Android AAR 빌드 스크립트
#
# 사용법:
#   ./scripts/build_android_aar.sh          # Debug + Release 빌드
#   ./scripts/build_android_aar.sh debug    # Debug 빌드만
#   ./scripts/build_android_aar.sh release  # Release 빌드만
#   ./scripts/build_android_aar.sh clean    # 빌드 디렉토리 정리
#
# 환경 변수:
#   ANDROID_SDK_ROOT - Android SDK 경로 (필수)
#   ANDROID_NDK_ROOT - Android NDK 경로 (옵션, SDK에서 자동 탐색)
#
# @version 1.0.0

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 스크립트 경로
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ANDROID_DIR="$PROJECT_ROOT/android"

# 기본값
BUILD_TYPE="${1:-all}"
PARALLEL_JOBS=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)

# =============================================================================
# 유틸리티 함수
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "$1 명령어를 찾을 수 없습니다. 설치가 필요합니다."
        return 1
    fi
    return 0
}

# =============================================================================
# 환경 검증
# =============================================================================

verify_environment() {
    log_info "환경 검증 중..."

    # Java 버전 확인
    if ! check_command java; then
        log_error "Java가 설치되지 않았습니다. JDK 17 이상이 필요합니다."
        exit 1
    fi

    JAVA_VERSION=$(java -version 2>&1 | head -n 1 | cut -d'"' -f2 | cut -d'.' -f1)
    if [ "$JAVA_VERSION" -lt 17 ]; then
        log_error "Java 17 이상이 필요합니다. 현재 버전: $JAVA_VERSION"
        exit 1
    fi
    log_info "Java 버전: $JAVA_VERSION"

    # Android SDK 경로 확인
    if [ -z "$ANDROID_SDK_ROOT" ] && [ -z "$ANDROID_HOME" ]; then
        # 일반적인 위치에서 탐색
        if [ -d "$HOME/Library/Android/sdk" ]; then
            export ANDROID_SDK_ROOT="$HOME/Library/Android/sdk"
        elif [ -d "$HOME/Android/Sdk" ]; then
            export ANDROID_SDK_ROOT="$HOME/Android/Sdk"
        else
            log_error "ANDROID_SDK_ROOT 환경 변수를 설정해주세요."
            exit 1
        fi
    fi
    ANDROID_SDK_ROOT="${ANDROID_SDK_ROOT:-$ANDROID_HOME}"
    log_info "Android SDK: $ANDROID_SDK_ROOT"

    # NDK 확인
    if [ -z "$ANDROID_NDK_ROOT" ]; then
        # SDK 내에서 NDK 탐색
        if [ -d "$ANDROID_SDK_ROOT/ndk" ]; then
            ANDROID_NDK_ROOT=$(ls -d "$ANDROID_SDK_ROOT/ndk"/*/ 2>/dev/null | sort -V | tail -n 1)
        fi
    fi
    if [ -n "$ANDROID_NDK_ROOT" ]; then
        log_info "Android NDK: $ANDROID_NDK_ROOT"
    else
        log_warning "NDK 경로를 찾을 수 없습니다. Gradle이 자동으로 다운로드합니다."
    fi

    # local.properties 생성
    if [ ! -f "$ANDROID_DIR/local.properties" ]; then
        log_info "local.properties 생성 중..."
        cat > "$ANDROID_DIR/local.properties" << EOF
# 자동 생성됨 - $(date)
sdk.dir=$ANDROID_SDK_ROOT
EOF
        if [ -n "$ANDROID_NDK_ROOT" ]; then
            echo "ndk.dir=$ANDROID_NDK_ROOT" >> "$ANDROID_DIR/local.properties"
        fi
    fi

    log_success "환경 검증 완료"
}

# =============================================================================
# Gradle Wrapper 확인
# =============================================================================

ensure_gradle_wrapper() {
    log_info "Gradle Wrapper 확인 중..."

    cd "$ANDROID_DIR"

    if [ ! -f "gradlew" ]; then
        log_error "gradlew 파일이 없습니다."
        exit 1
    fi

    # 실행 권한 부여
    chmod +x gradlew

    # gradle-wrapper.jar 확인
    if [ ! -f "gradle/wrapper/gradle-wrapper.jar" ]; then
        log_warning "gradle-wrapper.jar가 없습니다. Gradle 설치가 필요합니다."
        if check_command gradle; then
            log_info "시스템 Gradle로 wrapper 생성 중..."
            gradle wrapper --gradle-version=8.4
        else
            log_error "Gradle이 설치되지 않았습니다. 먼저 설치해주세요."
            log_info "설치 방법: brew install gradle (macOS)"
            exit 1
        fi
    fi

    log_success "Gradle Wrapper 준비 완료"
}

# =============================================================================
# 빌드 함수
# =============================================================================

build_debug() {
    log_info "Debug AAR 빌드 중..."
    cd "$ANDROID_DIR"

    ./gradlew :iris-sdk:assembleDebug \
        --parallel \
        --max-workers=$PARALLEL_JOBS \
        --build-cache

    if [ $? -eq 0 ]; then
        log_success "Debug 빌드 완료"
        log_info "출력: $ANDROID_DIR/iris-sdk/build/outputs/aar/"
    else
        log_error "Debug 빌드 실패"
        return 1
    fi
}

build_release() {
    log_info "Release AAR 빌드 중..."
    cd "$ANDROID_DIR"

    ./gradlew :iris-sdk:assembleRelease \
        --parallel \
        --max-workers=$PARALLEL_JOBS \
        --build-cache

    if [ $? -eq 0 ]; then
        log_success "Release 빌드 완료"
        log_info "출력: $ANDROID_DIR/iris-sdk/build/outputs/aar/"
    else
        log_error "Release 빌드 실패"
        return 1
    fi
}

build_all() {
    log_info "전체 빌드 (Debug + Release) 시작..."

    build_debug
    build_release

    log_success "전체 빌드 완료"
}

clean_build() {
    log_info "빌드 디렉토리 정리 중..."
    cd "$ANDROID_DIR"

    ./gradlew clean

    # 추가 정리
    rm -rf "$ANDROID_DIR/.gradle"
    rm -rf "$ANDROID_DIR/iris-sdk/build"
    rm -rf "$ANDROID_DIR/iris-sdk/.cxx"
    rm -rf "$ANDROID_DIR/demo-app/build"

    log_success "정리 완료"
}

# =============================================================================
# 빌드 결과 요약
# =============================================================================

show_build_summary() {
    echo ""
    echo "=============================================="
    echo "          IrisLensSDK 빌드 결과"
    echo "=============================================="

    AAR_DIR="$ANDROID_DIR/iris-sdk/build/outputs/aar"

    if [ -d "$AAR_DIR" ]; then
        echo ""
        echo "생성된 AAR 파일:"
        ls -la "$AAR_DIR"/*.aar 2>/dev/null || echo "  (없음)"

        echo ""
        echo "네이티브 라이브러리:"
        for abi in arm64-v8a armeabi-v7a; do
            SO_FILE="$ANDROID_DIR/iris-sdk/build/intermediates/merged_native_libs/release/out/lib/$abi/libiris_jni.so"
            if [ -f "$SO_FILE" ]; then
                SIZE=$(ls -lh "$SO_FILE" | awk '{print $5}')
                echo "  - $abi: $SIZE"
            else
                # Debug 빌드에서 찾기
                SO_FILE="$ANDROID_DIR/iris-sdk/build/intermediates/merged_native_libs/debug/out/lib/$abi/libiris_jni.so"
                if [ -f "$SO_FILE" ]; then
                    SIZE=$(ls -lh "$SO_FILE" | awk '{print $5}')
                    echo "  - $abi: $SIZE (debug)"
                fi
            fi
        done
    else
        echo "AAR 빌드 결과가 없습니다."
    fi

    echo ""
    echo "=============================================="
}

# =============================================================================
# 메인 실행
# =============================================================================

main() {
    echo ""
    echo "=============================================="
    echo "     IrisLensSDK Android AAR 빌드"
    echo "=============================================="
    echo ""

    verify_environment
    ensure_gradle_wrapper

    case "$BUILD_TYPE" in
        debug)
            build_debug
            ;;
        release)
            build_release
            ;;
        all)
            build_all
            ;;
        clean)
            clean_build
            exit 0
            ;;
        *)
            log_error "알 수 없는 빌드 타입: $BUILD_TYPE"
            echo "사용법: $0 [debug|release|all|clean]"
            exit 1
            ;;
    esac

    show_build_summary
}

# 스크립트 실행
main "$@"
