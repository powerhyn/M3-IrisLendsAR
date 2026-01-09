# P1-W4-07: 웹캠 데모 (Camera Demo)

**작성일**: 2026-01-09
**상태**: ✅ 완료
**우선순위**: 높음
**예상 복잡도**: 중간

---

## 1. 개요

### 1.1 목적
macOS 데스크톱 환경에서 웹캠을 통한 실시간 홍채 검출 및 렌즈 오버레이 데모 구현.
SDKManager 없이 FrameProcessor를 직접 활용하여 품질 검증 및 고도화 진행.

### 1.2 배경
- P1-W4-03 FrameProcessor 파이프라인 완료
- SDKManager는 C API 래퍼 용도로, 데스크톱 데모에는 불필요
- Demo-First 접근으로 실제 사용 패턴 파악 후 API 설계

### 1.3 범위
```
포함:
├── 웹캠 실시간 캡처 (OpenCV VideoCapture)
├── FrameProcessor 통합 (검출 + 렌더링)
├── 렌즈 텍스처 로딩 및 오버레이
├── 실시간 FPS 표시
├── 키보드 컨트롤 (렌즈 선택, 설정 변경)
└── 품질 검증 및 피드백 수집

제외:
├── SDKManager 구현 (나중에 Android 바인딩 시 추가)
├── C API 래퍼 (데모에 불필요)
└── 멀티스레드 처리 (Phase 2)
```

---

## 2. 기술 설계

### 2.1 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                     camera_demo.cpp                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  OpenCV     │    │  Frame      │    │  OpenCV     │     │
│  │  VideoCapture│ ──▶│  Processor  │ ──▶│  imshow     │     │
│  │  (웹캠)     │    │  (SDK Core) │    │  (디스플레이)│     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │             │
│         │                  │                  │             │
│         ▼                  ▼                  ▼             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  실시간 루프                         │   │
│  │  1. 프레임 캡처 (30fps)                             │   │
│  │  2. 포맷 변환 (BGR → 내부 처리)                     │   │
│  │  3. 홍채 검출 (MediaPipeDetector)                   │   │
│  │  4. 렌즈 렌더링 (LensRenderer)                      │   │
│  │  5. 결과 표시 + FPS 오버레이                        │   │
│  │  6. 키보드 입력 처리                                │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 컴포넌트 의존성

```
camera_demo.cpp
├── iris_sdk/frame_processor.h
│   ├── iris_sdk/mediapipe_detector.h
│   └── iris_sdk/lens_renderer.h
├── iris_sdk/types.h
└── OpenCV (VideoCapture, imshow, putText)
```

### 2.3 주요 클래스/함수

```cpp
// 데모 앱 구조
class CameraDemo {
public:
    // 초기화
    bool initialize(const std::string& model_path);

    // 메인 루프
    void run();

    // 정리
    void shutdown();

private:
    // 프레임 처리
    void processFrame(cv::Mat& frame);

    // UI 오버레이
    void drawDebugInfo(cv::Mat& frame, const ProcessResult& result);
    void drawFPS(cv::Mat& frame);
    void drawLandmarks(cv::Mat& frame, const IrisResult& iris);

    // 입력 처리
    bool handleKeyInput(int key);

    // 렌즈 관리
    void loadLensTextures(const std::string& texture_dir);
    void cycleLens();

private:
    cv::VideoCapture camera_;
    FrameProcessor processor_;

    // 렌즈 설정
    std::vector<std::string> lens_paths_;
    size_t current_lens_index_ = 0;
    LensConfig lens_config_;

    // 디버그 설정
    bool show_landmarks_ = true;
    bool show_fps_ = true;
    bool show_debug_info_ = false;

    // FPS 계산
    double fps_ = 0.0;
    std::chrono::steady_clock::time_point last_time_;
};
```

### 2.4 키보드 컨트롤

| 키 | 기능 | 설명 |
|----|------|------|
| `ESC` / `Q` | 종료 | 데모 종료 |
| `SPACE` | 렌즈 전환 | 다음 렌즈 텍스처로 변경 |
| `L` | 랜드마크 토글 | 검출된 랜드마크 표시/숨김 |
| `F` | FPS 토글 | FPS 표시/숨김 |
| `D` | 디버그 토글 | 상세 디버그 정보 표시 |
| `+` / `-` | 투명도 조절 | 렌즈 투명도 증가/감소 |
| `[` / `]` | 크기 조절 | 렌즈 크기 증가/감소 |
| `S` | 스크린샷 | 현재 프레임 저장 |
| `R` | 리셋 | 설정 초기화 |

---

## 3. 구현 계획

### 3.1 단계별 작업

#### Phase 1: 기본 웹캠 루프 (예상: 1시간)
```
├── OpenCV VideoCapture 설정
├── 기본 윈도우 생성 및 표시
├── 프레임 캡처 루프
└── ESC 종료 처리
```

#### Phase 2: FrameProcessor 통합 (예상: 2시간)
```
├── FrameProcessor 초기화
├── 모델 로딩
├── 프레임별 process() 호출
├── 검출 결과 확인
└── 에러 처리
```

#### Phase 3: 렌즈 렌더링 (예상: 1시간)
```
├── 렌즈 텍스처 로딩
├── LensConfig 설정
├── 렌더링 적용
└── 텍스처 전환 기능
```

#### Phase 4: UI/UX 개선 (예상: 2시간)
```
├── FPS 오버레이
├── 랜드마크 시각화
├── 디버그 정보 표시
├── 키보드 컨트롤 구현
└── 스크린샷 기능
```

#### Phase 5: 품질 검증 (예상: 2시간)
```
├── 성능 측정 (30fps 달성 여부)
├── 검출 정확도 확인
├── 렌더링 품질 평가
├── 엣지 케이스 테스트
└── 개선점 문서화
```

### 3.2 예상 파일 구조

```
cpp/examples/
├── CMakeLists.txt           # 빌드 설정 (수정)
├── camera_demo.cpp          # 웹캠 데모 (신규)
├── image_demo.cpp           # 이미지 데모 (나중에)
└── common/                  # 공통 유틸리티 (선택)
    ├── demo_utils.h
    └── demo_utils.cpp
```

---

## 4. 성공 기준

### 4.1 기능 요구사항
| 항목 | 기준 | 검증 방법 |
|------|------|----------|
| 웹캠 캡처 | 정상 동작 | 화면 표시 확인 |
| 홍채 검출 | 실시간 검출 | 랜드마크 표시 |
| 렌즈 오버레이 | 정확한 위치 | 시각적 확인 |
| 키보드 컨트롤 | 모든 키 동작 | 각 키 테스트 |

### 4.2 성능 요구사항
| 지표 | 목표 | 측정 방법 |
|------|------|----------|
| FPS | 30fps 이상 | 화면 FPS 표시 |
| 지연 | 33ms 이하 | ProcessResult 시간 |
| 메모리 | 200MB 이하 | Activity Monitor |

### 4.3 품질 체크리스트
- [ ] 정면 얼굴에서 안정적 검출
- [ ] 좌/우 회전 (30도 내) 정상 동작
- [ ] 조명 변화에 적응
- [ ] 렌즈 오버레이 자연스러움
- [ ] 눈 깜빡임 시 안정성
- [ ] 다양한 거리에서 동작

---

## 5. 의존성

### 5.1 필수 의존성
- FrameProcessor (완료)
- MediaPipeDetector (완료)
- LensRenderer (완료)
- OpenCV 4.x (설치됨)
- TFLite 모델 (설치됨)

### 5.2 테스트 환경
- macOS (M3 Pro)
- 내장 FaceTime HD 카메라 또는 외장 웹캠
- 해상도: 640x480 또는 1280x720

---

## 6. 리스크 및 대응

| 리스크 | 확률 | 영향 | 대응 |
|--------|------|------|------|
| 웹캠 권한 문제 (macOS) | 낮음 | 블로커 | 시스템 환경설정에서 권한 부여 |
| 30fps 미달 | 중간 | 품질 | 해상도 낮춤, 최적화 |
| 검출 불안정 | 중간 | UX | 스무딩, 칼만 필터 적용 |
| 렌즈 위치 부정확 | 중간 | 품질 | 좌표 보정 로직 추가 |

---

## 7. 다음 단계 (데모 완료 후)

### 7.1 품질 고도화 (Demo 기반)
```
├── 검출 스무딩 (칼만 필터)
├── 렌더링 품질 개선
├── 엣지 케이스 처리
└── 성능 최적화
```

### 7.2 SDKManager 추가 (Android 이식 시)
```
├── C API 래퍼 설계
├── SDKManager 싱글톤 구현
├── 초기화/해제 관리
└── 에러 처리 표준화
```

### 7.3 Android 포팅
```
├── JNI 바인딩
├── Camera2/CameraX 연동
├── 성능 최적화 (GPU)
└── 데모 앱
```

---

## 8. 참고 자료

### 8.1 관련 문서
- [P1-W4-03_frame_processor.md](P1-W4-03_frame_processor.md) - FrameProcessor 구현
- [P1-W4-01_lens_renderer.md](P1-W4-01_lens_renderer.md) - LensRenderer 구현
- [P1-W3-03_mediapipe_detector.md](P1-W3-03_mediapipe_detector.md) - MediaPipeDetector 구현
- [000_phase1_plan.md](000_phase1_plan.md) - Phase 1 마스터 계획

### 8.2 코드 참조
```cpp
// FrameProcessor 사용 예시 (from frame_processor.h)
FrameProcessor processor;
processor.initialize("models/");
processor.loadLensTexture("lens.png");

LensConfig config;
config.opacity = 0.8f;

ProcessResult result = processor.process(frame_data, width, height,
                                         FrameFormat::BGR, &config);
```

---

## 9. 실행 내역

### 작업 로그
| 날짜 | 작업 | 결과 | 비고 |
|------|------|------|------|
| 2026-01-09 | 계획 문서 작성 | ✅ 완료 | - |
| 2026-01-09 | Phase 1: 기본 웹캠 루프 | ✅ 완료 | OpenCV VideoCapture 사용 |
| 2026-01-09 | Phase 2: FrameProcessor 통합 | ✅ 완료 | process() 메서드 호출 |
| 2026-01-09 | Phase 3: 렌즈 렌더링 | ✅ 완료 | 텍스처 순환, 렌즈 설정 조절 |
| 2026-01-09 | Phase 4: UI/UX 개선 | ✅ 완료 | FPS, 랜드마크, 디버그 정보 |
| - | Phase 5: 품질 검증 | ⏳ 대기 | 빌드 후 실행 테스트 필요 |

---

## 10. 구현 결과

### 10.1 생성된 파일
- `cpp/examples/camera_demo.cpp` - 웹캠 데모 애플리케이션
- `cpp/examples/CMakeLists.txt` - camera_demo 타겟 추가

### 10.2 구현된 기능

#### CameraDemo 클래스
```cpp
class CameraDemo {
public:
    bool initialize(const std::string& model_path);  // 초기화
    int loadLensTextures(const std::vector<std::string>& paths);  // 텍스처 로드
    void run();  // 메인 루프
    void shutdown();  // 정리
};
```

#### 키보드 컨트롤 (전체 구현)
| 키 | 기능 | 상태 |
|----|------|------|
| ESC/Q | 종료 | ✅ |
| SPACE | 렌즈 텍스처 변경 | ✅ |
| E | 렌즈 효과 ON/OFF | ✅ |
| L | 랜드마크 표시 토글 | ✅ |
| F | FPS 표시 토글 | ✅ |
| D | 디버그 정보 토글 | ✅ |
| +/- | 투명도 조절 | ✅ |
| [/] | 크기 조절 | ✅ |
| S | 스크린샷 저장 | ✅ |
| R | 설정 초기화 | ✅ |
| H | 도움말 출력 | ✅ |

#### 디버그 오버레이
- FPS 카운터 (녹색)
- 처리 시간 (Total, Detection, Render)
- 검출 상태 (Detected/Not Detected, L/R)
- 신뢰도 점수
- 현재 렌즈 설정 (Opacity, Scale)
- 홍채 랜드마크 시각화 (중심점, 경계점, 반지름 원)
- 얼굴 바운딩 박스

### 10.3 빌드 설정
```cmake
# CMakeLists.txt 업데이트
- OpenCV 의존성 (core, imgproc, highgui, videoio)
- C++17 필수 (std::filesystem)
- macOS 프레임워크 (AVFoundation, CoreMedia, CoreVideo)
- 빌드 후 모델 파일 복사
```

### 10.4 실행 방법
```bash
# 빌드
cd cpp/cmake-build-debug
cmake ..
cmake --build . --target camera_demo

# 실행
./bin/camera_demo [model_path]
# 또는
./bin/camera_demo  # 자동 경로 탐색
```

---

## 변경 이력

| 버전 | 날짜 | 변경 내용 |
|------|------|----------|
| v1.0 | 2026-01-09 | 초안 작성 |
| v1.1 | 2026-01-09 | camera_demo.cpp 구현 완료, CMakeLists.txt 업데이트 |
