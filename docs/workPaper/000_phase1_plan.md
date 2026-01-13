    # Phase 1: MVP 개발 마스터 계획서

**작성일**: 2025-01-07
**최종 수정**: 2026-01-12
**상태**: 진행중 🔄

---

## 진척도 현황

### 전체 진행률: 69% (18/26 태스크 완료)

```
Week 1-2: 환경 설정     [██████████] 100% (8/8 완료) ✅
Week 3-4: 코어 엔진     [██████████] 100% (10/10 완료) ✅
Week 5-6: Android       [░░░░░░░░░░] 0% (0/8)
```

### 마일스톤 상태

| 마일스톤 | 상태 | 완료일 |
|----------|------|--------|
| M1: 환경 구축 | ✅ 완료 (8/8) | 2026-01-07 |
| M2: 코어 완성 | ✅ 완료 (10/10) | 2026-01-12 |
| M3: 데스크톱 실시간 | ✅ 완료 | 2026-01-12 |
| M4: Android MVP | ⏳ 대기 | - |

### 현재 워크플로우

- **활성 워크플로우**: [001_phase1_workflow.md](001_phase1_workflow.md)
  - 26개 태스크, 4개 마일스톤
  - 의존성 다이어그램 및 품질 게이트 포함
- **완료된 태스크**:
  - M1 전체 (P1-W1-01~04, P1-W2-01~04) - 환경 구축
  - M2 전체 (P1-W3-01~04, P1-W4-01~06) - 코어 엔진
  - M3 달성 (웹캠 실시간 30fps+ 확인)
- **다음 단계**: M4 Android MVP (Week 5-6)

---

## 이슈 트래킹

### 진행중 이슈

| ID | 제목 | 심각도 | 담당 | 상태 |
|----|------|--------|------|------|
| - | - | - | - | - |

### 해결된 이슈

| ID | 제목 | 해결일 | 해결 방법 |
|----|------|--------|----------|
| ISSUE-001 | CMake 미설치 | 2026-01-07 | brew install cmake |

### 이슈 상세

**ISSUE-001: CMake 미설치** ✅ 해결됨
- **심각도**: 🟡 중간 (빌드 테스트 블로커)
- **영향**: P1-W2-02, P1-W2-04 진행 불가
- **해결 방안**: `brew install cmake` 실행
- **해결 결과**: CMake 4.2.1 설치 완료, 모든 빌드 성공

---

## 기반 지식

### 학습한 내용

> 개발 중 학습한 기술적 내용 기록

1. **MediaPipe → TFLite 전환**: MediaPipe C++는 Bazel 빌드 필요, TFLite로 직접 모델 추론 가능
2. **CMake 조건부 컴파일**: `find_package(... QUIET)` + 조건부 링크로 유연한 빌드
3. **Android NDK CMake**: `android.toolchain.cmake`으로 쉬운 크로스컴파일
4. **Face Mesh 인덱스 체계**: 좌측 홍채 468-472, 우측 홍채 473-477 (MediaPipe 표준)
5. **GPU 가속 렌더링**: OpenCV UMat 활용 시 T-API GPU 자동 선택으로 성능 향상
6. **프레임 포맷 처리**: RGBA/BGR/Grayscale 통합 처리 파이프라인 설계
7. **홍채 이탈 감지**: 프레임 경계 체크로 부분 가시 상태 안전 처리
8. **알파 블렌딩 최적화**: 사전 계산 알파와 행 단위 처리로 렌더링 효율화

### 의사결정 로그

> 이 Phase에서 내린 기술적 결정들

| 날짜 | 결정 | 이유 | 영향 |
|------|------|------|------|
| 2026-01-07 | TFLite 직접 사용 | MediaPipe Bazel 복잡성 회피 | FindTFLite.cmake 생성 필요 |
| 2026-01-07 | 조건부 컴파일 채택 | 의존성 없이도 빌드 가능 | 개발 환경 유연성 증가 |
| 2026-01-07 | API 24 최소 타겟 | Android 7.0+, 현재 점유율 충분 | 레거시 제외 |
| 2026-01-10 | Strategy 패턴 검출기 | 모델 교체 유연성 확보 | EyeOnly/Hybrid 검출기 확장 용이 |
| 2026-01-10 | 싱글톤 SDKManager | 전역 상태 관리 단순화 | 멀티 인스턴스 불가 (트레이드오프) |
| 2026-01-11 | OpenCV UMat 사용 | GPU/CPU 자동 선택 | 하드웨어 의존 없이 최적화 |
| 2026-01-11 | extern "C" API | 플랫폼 바인딩 호환성 | 맹글링 방지, ABI 안정성 |
| 2026-01-12 | 추적 모드 구현 | 30fps 성능 달성 | face_rect 기반 빠른 재검출 |
| 2026-01-12 | 정규화 좌표계 | 해상도 독립성 확보 | 모든 해상도 동일 처리 |

---

## 1. Phase 1 개요

### 1.1 목적
실시간 카메라 영상에서 홍채를 추적하여 가상 렌즈를 오버레이하는 AR 피팅 SDK 개발

### 1.2 핵심 요구사항
| 항목 | 목표 |
|------|------|
| 프레임 레이트 | 30fps 이상 |
| 검출 지연 | 33ms 이하 |
| 메모리 사용 | 100MB 이하 |
| SDK 크기 | 20MB 이하 |
| 검출 정확도 | 95% 이상 (정상 조건) |

### 1.3 타겟 플랫폼
- Phase 1: Desktop (macOS/Linux) → Android
- Phase 2: iOS, Flutter
- Phase 3: Web (WASM)

---

## 2. 확정된 의사결정

| 결정 항목 | 선택 | 이유 |
|----------|------|------|
| MediaPipe 통합 | 사전 빌드 바이너리 → 소스 빌드 전환 | 빠른 시작 + 추후 커스터마이징 |
| 테스트 우선순위 | Desktop 먼저 | 코어 로직 검증 용이 |
| Phase 1 범위 | Desktop + Android만 | MVP 집중 |
| 프로젝트 구조 | 모노레포 스타일 | 플랫폼별 독립성 + 중앙 관리 |

---

## 3. 프로젝트 구조

```
IrisLensSDK/
├── README.md                       ← 프로젝트 소개
├── CLAUDE.md                       ← AI 어시스턴트 가이드
├── .gitignore
├── .gitattributes                  ← Git LFS 설정
│
├── docs/                           ← 📚 전체 문서
│   ├── PROJECT_SPEC.md             ← 프로젝트 명세서
│   ├── DECISION_RECORD.md          ← 의사결정 기록
│   ├── DEVELOPMENT_ROADMAP.md      ← 개발 로드맵
│   ├── ARCHITECTURE.md             ← 아키텍처 문서
│   └── workPaper/                  ← 작업 히스토리
│       ├── 000_phase1_plan.md      ← Phase 1 마스터 (본 문서)
│       ├── 000_phase2_plan.md      ← Phase 2 마스터 (예정)
│       └── 00N_xxx.md              ← 작업 로그
│
├── shared/                         ← 🔗 공유 리소스
│   ├── models/                     ← ML 모델 파일 (.tflite)
│   │   ├── face_detection_short_range.tflite
│   │   ├── face_landmark.tflite
│   │   └── iris_landmark.tflite
│   ├── textures/                   ← 렌즈 텍스처 이미지
│   └── test_data/                  ← 테스트용 이미지/영상
│
├── cpp/                            ← ⚙️ C++ 코어 엔진
│   ├── CMakeLists.txt
│   ├── include/                    ← 공개 헤더
│   │   ├── iris_sdk/
│   │   │   ├── iris_detector.h
│   │   │   ├── mediapipe_detector.h
│   │   │   ├── lens_renderer.h
│   │   │   ├── frame_processor.h
│   │   │   ├── sdk_manager.h
│   │   │   └── sdk_api.h           ← C API (바인딩용)
│   │   └── iris_sdk.h              ← 통합 헤더
│   ├── src/                        ← 구현 파일
│   │   ├── iris_detector.cpp
│   │   ├── mediapipe_detector.cpp
│   │   ├── lens_renderer.cpp
│   │   ├── frame_processor.cpp
│   │   ├── sdk_manager.cpp
│   │   └── sdk_api.cpp
│   ├── tests/                      ← 단위/통합 테스트
│   │   ├── CMakeLists.txt
│   │   ├── test_iris_detector.cpp
│   │   ├── test_lens_renderer.cpp
│   │   └── test_integration.cpp
│   ├── examples/                   ← 데스크톱 예제
│   │   ├── CMakeLists.txt
│   │   ├── image_demo.cpp          ← 이미지 기반 데모
│   │   └── camera_demo.cpp         ← 웹캠 기반 데모
│   ├── third_party/                ← 사전빌드 의존성
│   │   ├── mediapipe/
│   │   └── opencv/
│   └── build/                      ← 빌드 결과물 (gitignore)
│
├── python/                         ← 🐍 Python 프로토타이핑
│   ├── requirements.txt
│   ├── setup.py
│   ├── iris_sdk/                   ← ctypes 바인딩
│   │   ├── __init__.py
│   │   └── bindings.py
│   ├── examples/
│   │   └── demo.py
│   └── tests/
│
├── android/                        ← 🤖 Android
│   ├── settings.gradle.kts
│   ├── build.gradle.kts
│   ├── iris-sdk/                   ← AAR 라이브러리 모듈
│   │   ├── build.gradle.kts
│   │   ├── src/main/
│   │   │   ├── cpp/                ← JNI 코드
│   │   │   │   ├── CMakeLists.txt
│   │   │   │   └── iris_jni.cpp
│   │   │   ├── java/com/irislenssdk/
│   │   │   │   ├── IrisLensSDK.kt
│   │   │   │   └── IrisResult.kt
│   │   │   └── AndroidManifest.xml
│   │   └── proguard-rules.pro
│   └── demo-app/                   ← 데모 앱
│       ├── build.gradle.kts
│       └── src/main/
│
├── ios/                            ← 🍎 iOS (Phase 2)
│   ├── IrisSDK/
│   └── DemoApp/
│
├── flutter/                        ← 🦋 Flutter Plugin (Phase 2)
│   ├── pubspec.yaml
│   ├── lib/
│   ├── android/
│   └── ios/
│
├── web/                            ← 🌐 WASM (Phase 3)
│   ├── package.json
│   └── src/
│
└── scripts/                        ← 🔧 빌드/배포 스크립트
    ├── build_cpp.sh                ← C++ 빌드 (Desktop)
    ├── build_android.sh            ← Android 빌드
    ├── build_ios.sh                ← iOS 빌드
    ├── sync_libs.sh                ← 빌드 결과물 → 플랫폼 복사
    ├── download_models.sh          ← 모델 파일 다운로드
    └── setup_env.sh                ← 개발 환경 설정
```

---

## 4. Phase 1 상세 계획

### 4.1 Week 1-2: 환경 설정 및 프로젝트 구조

#### Week 1: 기반 구축

| 작업 | 산출물 | 검증 기준 |
|------|--------|----------|
| 프로젝트 디렉토리 구조 생성 | 위 구조대로 폴더 생성 | 구조 확인 |
| Git 설정 | `.gitignore`, `.gitattributes` | LFS 동작 확인 |
| 문서 업데이트 | CLAUDE.md 업데이트 | 구조 반영 확인 |
| CMake 루트 설정 | `cpp/CMakeLists.txt` | cmake 실행 성공 |

#### Week 2: 의존성 통합

| 작업 | 산출물 | 검증 기준 |
|------|--------|----------|
| OpenCV 설치/빌드 | `cpp/third_party/opencv/` | 링크 성공 |
| MediaPipe 바이너리 다운로드 | `cpp/third_party/mediapipe/` | 헤더 포함 성공 |
| ML 모델 다운로드 | `shared/models/*.tflite` | 파일 존재 확인 |
| 빌드 테스트 | 빈 라이브러리 빌드 | `libiris_sdk.a` 생성 |

### 4.2 Week 3-4: 코어 엔진 개발

#### Week 3: 인터페이스 및 검출기

| 작업 | 산출물 | 검증 기준 |
|------|--------|----------|
| IrisDetector 인터페이스 | `iris_detector.h/cpp` | 컴파일 성공 |
| IrisResult 구조체 | 10개 랜드마크 + 신뢰도 | 타입 정의 완료 |
| MediaPipeDetector 구현 | `mediapipe_detector.h/cpp` | MediaPipe 호출 성공 |
| 이미지 테스트 | `test_iris_detector.cpp` | 샘플 이미지 검출 성공 |

#### Week 4: 렌더링 및 API

| 작업 | 산출물 | 검증 기준 |
|------|--------|----------|
| LensRenderer 구현 | `lens_renderer.h/cpp` | 렌즈 오버레이 성공 |
| FrameProcessor 구현 | `frame_processor.h/cpp` | YUV→RGB 변환 |
| SDKManager 구현 | `sdk_manager.h/cpp` | 싱글톤 동작 |
| C API 래퍼 | `sdk_api.h/cpp` | extern "C" 함수 동작 |
| 데스크톱 데모 | `examples/image_demo.cpp` | 이미지 처리 성공 |
| 웹캠 데모 | `examples/camera_demo.cpp` | 실시간 처리 확인 |

### 4.3 Week 5-6: Android 바인딩

#### Week 5: JNI 바인딩

| 작업 | 산출물 | 검증 기준 |
|------|--------|----------|
| Android 프로젝트 설정 | `android/` 구조 | Gradle sync 성공 |
| JNI 바인딩 작성 | `iris_jni.cpp` | 네이티브 함수 호출 성공 |
| Kotlin API 작성 | `IrisLensSDK.kt` | API 사용 가능 |
| AAR 빌드 | `iris-sdk-release.aar` | AAR 생성 성공 |

#### Week 6: 데모 앱 및 성능 검증

| 작업 | 산출물 | 검증 기준 |
|------|--------|----------|
| CameraX 연동 | 데모 앱 카메라 프리뷰 | 카메라 동작 |
| SDK 통합 | 실시간 홍채 검출 | 검출 결과 표시 |
| 렌즈 오버레이 | AR 렌즈 표시 | 시각적 확인 |
| 성능 측정 | FPS 카운터, 메모리 프로파일 | **30fps+, 100MB 이하** |

---

## 5. 아키텍처 다이어그램

```
┌─────────────────────────────────────────────────────────────┐
│                      Application Layer                       │
│  (Android App / iOS App / Flutter App / Web App)            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Binding Layer                           │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │   JNI   │  │ Obj-C++ │  │dart:ffi │  │  WASM   │        │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        C API Layer                           │
│  iris_sdk_init() │ iris_sdk_detect() │ iris_sdk_render()    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      C++ Core Engine                         │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                    SDKManager (Singleton)               │ │
│  └────────────────────────────────────────────────────────┘ │
│           │                    │                    │        │
│           ▼                    ▼                    ▼        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │IrisDetector  │    │LensRenderer  │    │FrameProcessor│   │
│  │  (Interface) │    │              │    │              │   │
│  ├──────────────┤    └──────────────┘    └──────────────┘   │
│  │MediaPipe     │                                           │
│  │Detector      │                                           │
│  ├──────────────┤                                           │
│  │EyeOnly       │  (Phase 2)                                │
│  │Detector      │                                           │
│  ├──────────────┤                                           │
│  │Hybrid        │  (Phase 2)                                │
│  │Detector      │                                           │
│  └──────────────┘                                           │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                 Third Party Libraries                 │   │
│  │  MediaPipe  │  OpenCV 4.x  │  TensorFlow Lite        │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. 핵심 인터페이스 설계

### 6.1 IrisResult 구조체

```cpp
struct IrisLandmark {
    float x;           // 0.0 ~ 1.0 (normalized)
    float y;           // 0.0 ~ 1.0 (normalized)
    float z;           // depth (optional)
};

struct IrisResult {
    bool detected;                    // 검출 성공 여부
    float confidence;                 // 신뢰도 (0.0 ~ 1.0)

    // 왼쪽 눈 홍채 (5 points: center + 4 boundary)
    IrisLandmark left_iris[5];

    // 오른쪽 눈 홍채 (5 points: center + 4 boundary)
    IrisLandmark right_iris[5];

    // 눈 영역 바운딩 박스 (렌더링 참조용)
    float left_eye_rect[4];   // x, y, width, height
    float right_eye_rect[4];
};
```

### 6.2 C API

```cpp
extern "C" {
    // 초기화/해제
    int iris_sdk_init(const char* model_path);
    void iris_sdk_destroy();

    // 홍채 검출
    int iris_sdk_detect(
        const uint8_t* frame_data,
        int width,
        int height,
        int format,              // RGBA, BGR, NV21 등
        IrisResult* result
    );

    // 렌즈 렌더링
    int iris_sdk_render_lens(
        uint8_t* frame_data,     // in-place 수정
        int width,
        int height,
        const IrisResult* iris,
        const char* lens_texture_path,
        float opacity
    );

    // 설정
    void iris_sdk_set_config(const char* key, const char* value);
    const char* iris_sdk_get_version();
}
```

---

## 7. 마일스톤 및 검증 기준

| 마일스톤 | 완료 기준 | 검증 방법 |
|----------|----------|----------|
| **M1: 환경 구축** | CMake 빌드 성공, 의존성 해결 | `cmake --build .` 성공 |
| **M2: 코어 동작** | 샘플 이미지 홍채 검출 + 렌즈 오버레이 | 이미지 데모 실행 |
| **M3: 데스크톱 실시간** | 웹캠 30fps 실시간 처리 | 카메라 데모 실행 |
| **M4: Android MVP** | 실기기 30fps+, 지연 33ms 이하 | 데모 앱 프로파일링 |

---

## 8. 리스크 및 대응

| 리스크 | 확률 | 영향 | 대응 방안 |
|--------|------|------|----------|
| MediaPipe 사전빌드 호환성 | 중간 | 일정 지연 | 여러 버전 테스트, 필요시 소스 빌드 |
| OpenCV 빌드 복잡성 | 낮음 | 일정 지연 | Homebrew/apt 패키지 활용 |
| 성능 미달 (30fps 미만) | 중간 | 품질 저하 | 해상도 조절, GPU 가속 적용 |
| 모바일 메모리 제한 | 중간 | 크래시 | 메모리 풀링, 모델 경량화 |
| MediaPipe 한계 (눈 클로즈업) | 높음 | 기능 제한 | Phase 2 Eye-Only 모델 |

---

## 9. 워크플로우 사용법

### Phase 작업 시작 시

```bash
# 1. 워크플로우 생성
/sc:workflow 000_phase1_plan.md --strategy systematic

# 2. 생성된 워크플로우로 태스크 분리
# 3. 태스크별 작업 진행
# 4. 완료 시 본 문서의 진척도 업데이트
```

### 진척도 업데이트 규칙

- 마일스톤 완료 시: 상태를 ✅ 완료로 변경, 완료일 기입
- 이슈 발생 시: 이슈 트래킹 테이블에 추가
- 학습 내용: 기반 지식 섹션에 기록
- 의사결정: 의사결정 로그에 기록

---

## 작업 로그

### 관련 작업 문서

| 문서 | 내용 | 상태 |
|------|------|------|
| [001_phase1_workflow.md](001_phase1_workflow.md) | Phase 1 상세 워크플로우 | ✅ 완료 |
| [P1-W3-01_iris_detector_interface.md](P1-W3-01_iris_detector_interface.md) | IrisDetector 인터페이스 | ✅ 완료 |
| [P1-W3-02_data_structures.md](P1-W3-02_data_structures.md) | 데이터 구조 정의 | ✅ 완료 |
| [P1-W3-03_mediapipe_detector.md](P1-W3-03_mediapipe_detector.md) | MediaPipeDetector 구현 | ✅ 완료 |
| [P1-W3-04_detector_unit_test.md](P1-W3-04_detector_unit_test.md) | 검출기 단위 테스트 | ✅ 완료 |
| [P1-W4-01_lens_renderer.md](P1-W4-01_lens_renderer.md) | LensRenderer 기본 구현 | ✅ 완료 |
| [P1-W4-02_blending_algorithm.md](P1-W4-02_blending_algorithm.md) | 블렌딩 알고리즘 | ✅ 완료 |
| [P1-W4-03_frame_processor.md](P1-W4-03_frame_processor.md) | FrameProcessor 파이프라인 | ✅ 완료 |
| [P1-W4-04_sdk_manager.md](P1-W4-04_sdk_manager.md) | SDKManager 싱글톤 | ✅ 완료 |
| [P1-W4-05_c_api_wrapper.md](P1-W4-05_c_api_wrapper.md) | C API 래퍼 | ✅ 완료 |
| [P1-W4-06_integration_test.md](P1-W4-06_integration_test.md) | 코어 통합 테스트 | ✅ 완료 |
| [P1-W4-07_webcam_demo.md](P1-W4-07_webcam_demo.md) | 웹캠 데모 애플리케이션 | ✅ 완료 |

---

## 변경 이력

| 버전 | 날짜 | 변경 내용 |
|------|------|----------|
| v1 | 2025-01-07 | 초안 작성 |
| v2 | 2025-01-07 | 모노레포 구조로 변경, Phase 1 범위 조정 |
| v3 | 2026-01-07 | Phase 마스터 문서로 전환, 진행 관리 섹션 추가 |
| v4 | 2026-01-12 | W3-W4 코어 엔진 완료 (M2, M3 달성), 진척률 69% 업데이트 |
