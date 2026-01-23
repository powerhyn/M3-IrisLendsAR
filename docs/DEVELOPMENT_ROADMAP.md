# IrisLensSDK 개발 로드맵

**버전**: 2.0
**최종 수정**: 2026-01-07

---

## 전체 개발 단계 개요

```
Phase 1: MVP (MediaPipe 기반)
├── Desktop 코어 엔진
└── Android 바인딩
    ↓
Phase 2: 크로스플랫폼 확장
├── iOS 바인딩
├── Flutter Plugin
└── 하이브리드 검출 시스템
    ↓
Phase 3: 최적화 및 Web 확장
├── Web WASM 지원
├── 성능 최적화
└── 기능 확장
```

---

## 프로젝트 구조

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
│   ├── DEVELOPMENT_ROADMAP.md      ← 개발 로드맵 (본 문서)
│   ├── ARCHITECTURE.md             ← 아키텍처 문서
│   └── workPaper/                  ← 작업 히스토리
│       ├── 000_phase1_plan.md      ← Phase 1 마스터 계획
│       ├── 000_phase2_plan.md      ← Phase 2 마스터 계획 (예정)
│       ├── 000_phase3_plan.md      ← Phase 3 마스터 계획 (예정)
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

## Phase 1: MVP 개발

**목표**: Desktop 코어 엔진 + Android 바인딩
**상세 계획**: [workPaper/000_phase1_plan.md](workPaper/000_phase1_plan.md)

### 범위

| 구분 | 포함 | 제외 |
|------|------|------|
| 플랫폼 | Desktop, Android | iOS, Flutter, Web |
| 검출기 | MediaPipe | Eye-Only, Hybrid |
| 기능 | 홍채 검출, 렌즈 오버레이 | 다중 렌즈, 색상 커스텀 |

### 마일스톤

| 마일스톤 | 완료 기준 | 상태 |
|----------|----------|------|
| M1: 환경 구축 | MediaPipe + OpenCV 빌드 성공 | ✅ 완료 |
| M2: 코어 완성 | 이미지 기반 검출 + 렌더링 동작 | ✅ 완료 |
| M3: 데스크톱 실시간 | 웹캠 30fps 실시간 처리 | ✅ 완료 |
| M4: Android MVP | 실기기 30fps 달성 | 🔄 진행 중 |

### 최근 진행 상황 (2025-01-21 업데이트)

#### 완료된 작업
- ✅ C++ 코어 엔진 구현 (MediaPipeDetector)
- ✅ Android JNI 바인딩 구현
- ✅ Android 데모 앱 기본 구현
- ✅ 실시간 홍채 검출 동작 확인 (17-19 FPS)

#### 진행 중인 작업
- 🔄 **Face Mesh 좌표계 문제 해결**
  - ✅ CameraX ViewPort 적용 (Preview/ImageAnalysis FOV 일치)
  - ✅ C++ SDK aspect ratio 보정 로직 추가
  - 🔄 최종 테스트 및 검증

#### 관련 문서
- 좌표계 문제 분석: `docs/PIPELINE_ANALYSIS.md`
- 아키텍처 변경: `docs/ARCHITECTURE.md` 섹션 6.4

---

## Phase 2: 크로스플랫폼 확장

**목표**: iOS/Flutter 지원 + 하이브리드 검출
**상세 계획**: [workPaper/000_phase2_plan.md](workPaper/000_phase2_plan.md) (예정)

### 범위

| 구분 | 포함 |
|------|------|
| 플랫폼 | iOS, Flutter |
| 검출기 | Eye-Only 모델, Hybrid 시스템 |
| 기능 | MediaPipe 한계 보완 |

### 주요 작업

1. **iOS Framework 생성**
   - Objective-C++ 래퍼
   - XCFramework 빌드
   - AVCaptureSession 연동

2. **Flutter Plugin**
   - dart:ffi 바인딩
   - 플랫폼 채널
   - 크로스플랫폼 데모 앱

3. **Eye-Only 모델**
   - 데이터 수집 (5,000~10,000장)
   - 모델 학습 (U-Net/MobileNet 기반)
   - 하이브리드 시스템 구축

---

## Phase 3: 최적화 및 Web 확장

**목표**: Web WASM + 성능 최적화 + 기능 확장
**상세 계획**: [workPaper/000_phase3_plan.md](workPaper/000_phase3_plan.md) (예정)

### 범위

| 구분 | 포함 |
|------|------|
| 플랫폼 | Web (WASM) |
| 최적화 | GPU 가속, 모델 양자화, 멀티스레드 |
| 기능 | 다중 렌즈, 색상 커스텀, 눈 깜빡임 처리 |

### 주요 작업

1. **Web WASM**
   - Emscripten 빌드
   - JavaScript API
   - WebGL 렌더링

2. **성능 최적화**
   - 모델 양자화 (INT8)
   - GPU 가속 (OpenGL ES / Metal)
   - 멀티스레드 파이프라인

3. **기능 확장**
   - 다중 렌즈 스타일
   - 렌즈 색상 커스터마이징
   - 눈 깜빡임 감지 및 처리
   - 조명 반사 시뮬레이션

---

## 성능 목표

| 지표 | 목표 | 측정 방법 |
|------|------|----------|
| 프레임 레이트 | 30fps 이상 | 1초당 처리 프레임 수 |
| 검출 지연 | 33ms 이하 | 입력→출력 시간 |
| 메모리 사용 | 100MB 이하 | 런타임 메모리 |
| SDK 크기 | 20MB 이하 | 라이브러리 파일 크기 |
| 검출 정확도 | 95% 이상 | 정상 조건 기준 |

---

## 문서 체계

```
docs/
├── PROJECT_SPEC.md         ← 제품 명세 (WHAT)
├── DECISION_RECORD.md      ← 의사결정 (WHY)
├── ARCHITECTURE.md         ← 시스템 설계 (HOW)
├── DEVELOPMENT_ROADMAP.md  ← 전체 로드맵 (본 문서)
│
└── workPaper/
    ├── 000_phase1_plan.md  ← Phase 1 마스터 (진행 관리)
    ├── 000_phase2_plan.md  ← Phase 2 마스터 (예정)
    ├── 000_phase3_plan.md  ← Phase 3 마스터 (예정)
    └── 00N_xxx.md          ← 작업 로그
```

### Phase별 000 문서 역할

각 `000_phaseN_plan.md`는 해당 Phase의 마스터 문서로:
- **진척도 추적**: 완료/진행중/대기 상태
- **워크플로우 기반**: 세부 태스크 분리 및 실행
- **이슈 트래킹**: 발생한 문제 및 해결 방안
- **기반 지식**: 학습한 내용 및 결정 사항

---

## 변경 이력

| 버전 | 날짜 | 변경 내용 |
|------|------|----------|
| 1.0 | 2025-01-07 | 초안 작성 |
| 2.0 | 2025-01-07 | 프로젝트 구조 업데이트, 문서 체계 정립 |
| 2.1 | 2025-01-21 | Phase 1 마일스톤 진행 상황 업데이트, 최근 작업 내역 추가 |
