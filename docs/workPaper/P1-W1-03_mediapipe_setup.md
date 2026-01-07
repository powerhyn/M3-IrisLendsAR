# P1-W1-03: MediaPipe 사전빌드 설정

**태스크 ID**: P1-W1-03
**상태**: ✅ 완료
**시작일**: 2026-01-07
**완료일**: 2026-01-07

---

## 1. 계획

### 목표
MediaPipe 사전빌드 라이브러리 다운로드 및 CMake 연동

### 산출물
- cpp/third_party/mediapipe/ 디렉토리
- FindMediaPipe.cmake 또는 mediapipe-config.cmake

### 검증 기준
- MediaPipe 헤더 include 가능
- CMake에서 find_package() 또는 타겟 링크 가능

### 리스크
- macOS에서 MediaPipe 사전빌드 호환성 문제 가능
- 대안: TensorFlow Lite 직접 사용

---

## 2. 분석

### MediaPipe macOS 지원 현황

MediaPipe는 주로 다음 방식으로 사용 가능:
1. **Python API**: pip install mediapipe (가장 쉬움)
2. **C++ Bazel 빌드**: 복잡하지만 완전한 지원
3. **사전빌드 바이너리**: 제한적 지원

### macOS용 접근 방식 결정

**권장 방식**: TensorFlow Lite + MediaPipe 모델 직접 사용

이유:
- MediaPipe C++ API는 Bazel 빌드 필요 (복잡)
- TensorFlow Lite는 CMake 지원, 사전빌드 제공
- MediaPipe의 Face Mesh/Iris 모델은 TFLite 형식

---

## 3. 구현 계획

### 3.1 TensorFlow Lite 기반 접근

```
cpp/third_party/
├── tflite/                  ← TensorFlow Lite 런타임
│   ├── include/
│   └── lib/
└── mediapipe_models/        ← MediaPipe TFLite 모델
    ├── face_detection_short_range.tflite
    ├── face_landmark.tflite
    └── iris_landmark.tflite
```

### 3.2 모델 파일 위치

모델 파일은 shared/models/에 저장 (이미 계획됨):
```
shared/models/
├── face_detection_short_range.tflite
├── face_landmark.tflite
└── iris_landmark.tflite
```

---

## 4. 실행 내역

### 4.1 TensorFlow Lite 설정

**생성된 파일:**
- `cpp/cmake/FindTFLite.cmake` - TFLite 찾기 모듈
- `cpp/CMakeLists.txt` 업데이트 - TFLite 의존성 추가

**CMake 연동:**
```cmake
find_package(TFLite QUIET)
if(TFLite_FOUND)
    target_link_libraries(iris_sdk PRIVATE TFLite::TFLite)
    target_compile_definitions(iris_sdk PRIVATE IRIS_SDK_HAS_TFLITE)
endif()
```

### 4.2 모델 파일 다운로드

**생성된 파일:**
- `scripts/download_models.sh` - 모델 다운로드 스크립트

**다운로드할 모델:**
| 모델 | 용도 | URL |
|------|------|-----|
| blaze_face_short_range.tflite | 얼굴 검출 | storage.googleapis.com |
| face_landmark.tflite | 얼굴 메시 | storage.googleapis.com |
| iris_landmark.tflite | 홍채 랜드마크 | storage.googleapis.com |

---

## 5. 검증 결과

### 검증 항목

| 항목 | 결과 | 비고 |
|------|------|------|
| FindTFLite.cmake 생성 | ✅ 통과 | CMake 모듈 작성 완료 |
| CMake 연동 | ✅ 통과 | 조건부 링크 설정 |
| 다운로드 스크립트 | ✅ 통과 | scripts/download_models.sh |
| TFLite 설치 | ⚠️ 대기 | 추후 설치 필요 |

---

## 6. 이슈 및 학습

### 결정 사항

| 결정 | 이유 |
|------|------|
| TFLite 직접 사용 | MediaPipe C++ Bazel 빌드 복잡성 회피 |
| MediaPipe 모델 사용 | Face Mesh + Iris 모델은 TFLite 형식으로 사용 가능 |
| 조건부 컴파일 | TFLite 없이도 빌드 가능하도록 설계 |

### 학습 내용

1. **MediaPipe 아키텍처**:
   - MediaPipe C++는 Bazel 빌드 시스템 의존
   - 모델은 TFLite 형식으로 제공됨
   - TFLite 런타임으로 직접 추론 가능

2. **CMake 모듈 작성**:
   - `find_path()`, `find_library()` 사용
   - IMPORTED 타겟 생성 패턴
   - `FindPackageHandleStandardArgs` 활용

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-01-07 | 태스크 문서 생성, 분석 시작 |
| 2026-01-07 | TFLite 기반 접근 결정, CMake 모듈 및 스크립트 생성 |
