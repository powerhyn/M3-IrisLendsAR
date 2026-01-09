# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

IrisLensSDK는 실시간 카메라 영상에서 홍채를 추적하여 가상 렌즈를 오버레이하는 AR 피팅 SDK입니다.

**핵심 목표**: 30fps 이상 실시간 처리, 크로스플랫폼 지원 (Android, iOS, Flutter, Web)

## 빌드 명령어

### CLion 개발 환경 (권장)
```bash
# CLion은 cpp/cmake-build-debug 디렉토리를 사용
# Ninja 제너레이터가 기본으로 설정됨
# TFLite 등 대용량 의존성이 사전 빌드되어 있음

# CLion 빌드 디렉토리에서 직접 테스트 실행
cd cpp/cmake-build-debug
cmake --build . --target test_mediapipe_detector_integration
./bin/test_mediapipe_detector_integration
```

### 터미널 빌드 (CLion 빌드 디렉토리 재사용)
```bash
# ⚠️ 중요: 새 빌드 디렉토리 생성 금지!
# 기존 cmake-build-debug 디렉토리를 재사용해야 TFLite 재다운로드 방지

cd cpp/cmake-build-debug
cmake .. -DIRIS_SDK_FETCH_TFLITE=OFF  # TFLite 재다운로드 방지
cmake --build . --parallel
```

### 클린 빌드 (주의 필요)
```bash
# ⚠️ 클린 빌드 시 TFLite 재다운로드 필요 (~400MB, 10분+)
# 가급적 기존 빌드 디렉토리 유지 권장

mkdir build && cd build
cmake -DBUILD_TESTS=ON ..
make
```

### Android/iOS 빌드
```bash
./scripts/build_android.sh
./scripts/build_ios.sh
./scripts/build_all.sh
```

### 테스트 실행
```bash
cd cpp/cmake-build-debug && ctest
```

> **빌드 일관성 가이드**: 자세한 빌드 설정은 `docs/BUILD_GUIDE.md` 참조

## 아키텍처

### 레이어 구조

```
Application Layer (앱)
    ↓
Binding Layer (JNI/Obj-C++/dart:ffi/WASM)
    ↓
C API Layer (extern "C" 함수들)
    ↓
C++ Core Engine (핵심 로직)
    ↓
Third Party (MediaPipe, OpenCV, TFLite)
```

### 핵심 컴포넌트

| 컴포넌트 | 역할 | 파일 |
|----------|------|------|
| `IrisDetector` | 홍채 검출 추상 인터페이스 | `core/include/iris_detector.h` |
| `MediaPipeDetector` | MediaPipe 기반 구현체 | `core/include/mediapipe_detector.h` |
| `LensRenderer` | 가상 렌즈 렌더링 | `core/include/lens_renderer.h` |
| `SDKManager` | 싱글톤 관리자 | `core/include/sdk_manager.h` |
| `sdk_api.h` | C API 래퍼 (바인딩용) | `core/include/sdk_api.h` |

### 검출기 인터페이스 패턴

모델 교체를 위해 Strategy 패턴 사용:
```
IrisDetector (인터페이스)
├── MediaPipeDetector (Phase 1)
├── EyeOnlyDetector (Phase 2)
└── HybridDetector (Phase 2 - MediaPipe 실패시 EyeOnly 폴백)
```

## 플랫폼별 바인딩

| 플랫폼 | 바인딩 | 결과물 | 위치 |
|--------|--------|--------|------|
| Android | JNI | libiris_sdk.so → AAR | `bindings/android/` |
| iOS | Objective-C++ | IrisSDK.xcframework | `bindings/ios/` |
| Flutter | dart:ffi | Plugin | `bindings/flutter/` |
| Web | Emscripten | WASM + JS | `bindings/web/` |

## C API 사용 규칙

모든 바인딩은 `sdk_api.h`의 C API를 통해 코어 호출:
- `iris_sdk_init()` - 초기화
- `iris_sdk_detect()` - 홍채 검출
- `iris_sdk_render_lens()` - 렌즈 오버레이
- `iris_sdk_destroy()` - 해제

## 성능 기준

| 지표 | 목표 |
|------|------|
| 프레임 레이트 | 30fps+ |
| 검출 지연 | 33ms 이하 |
| 메모리 | 100MB 이하 |
| SDK 크기 | 20MB 이하 |

## MediaPipe 한계점 (Phase 1)

- 얼굴 전체 인식 필수 (눈만 클로즈업시 실패)
- 극단적 각도(45°+)에서 불안정
- Phase 2에서 Eye-Only 커스텀 모델로 보완 예정

## 개발 순서

1. **core/** - C++ 코어 엔진 먼저 구현
2. **bindings/android/** - JNI 바인딩 (테스트 용이)
3. **bindings/ios/** - iOS Framework
4. **bindings/flutter/** - Flutter Plugin
5. **bindings/web/** - WASM (선택적)

## 의존성

- MediaPipe: Apache 2.0 (Face Mesh + Iris Tracking)
- OpenCV 4.x: Apache 2.0 (이미지 처리)
- TensorFlow Lite 2.x: Apache 2.0 (커스텀 모델용)
- CMake 3.18+, C++17

## 주의사항

- 카메라 데이터는 온디바이스 처리만 (외부 전송 금지)
- 처리 완료 후 메모리 즉시 해제
- 모델 파일 암호화 옵션 제공 필요

## 작업 규칙 (필수)

### 1. 작업 문서화
**모든 작업 완료 후 반드시 `docs/workPaper/`의 해당 작업 문서를 업데이트해야 합니다.**

- 작업 상태 변경 (⏳ 대기 → 🔄 진행 중 → ✅ 완료)
- 실행 내역 기록
- 이슈 및 학습 내용 추가
- 변경 이력 업데이트

### 2. C++ 작업 시 전문 에이전트 사용
**`cpp/` 폴더 하위에서 C++ 작업을 수행할 때는 반드시 `systems-programming:cpp-pro` 에이전트와 함께 작업합니다.**

```
cpp/
├── include/    ← C++ 헤더
├── src/        ← C++ 구현
└── tests/      ← C++ 테스트
```

이 에이전트는 다음을 제공합니다:
- 모던 C++ (C++17/20) 패턴 및 관용구
- RAII, 스마트 포인터, STL 알고리즘
- 메모리 안전성 및 성능 최적화
- 템플릿 및 이동 시맨틱

## 작업 히스토리 (Work Papers)

모든 개발 작업은 `docs/workPaper/` 폴더에 문서화됩니다.

**문서 명명 규칙**: `NNN_작업명.md` (예: `001_project_setup.md`)

**문서 내용**:
- 작업 목표 및 범위
- 수행한 작업 내역
- 의사결정 사항 및 이유
- 이슈 및 해결 방안
- 다음 단계

**현재 작업 문서**:
- `000_implementation_plan.md` - 전체 구현 계획서 (승인됨)
- `001_project_setup.md` - 프로젝트 초기 구조 설정
