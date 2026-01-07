# P1-W1-01: 프로젝트 디렉토리 구조 생성

**태스크 ID**: P1-W1-01
**상태**: ✅ 완료
**시작일**: 2026-01-07
**완료일**: 2026-01-07

---

## 1. 계획

### 목표
전체 프로젝트 디렉토리 구조를 000_phase1_plan.md 명세에 맞게 생성

### 산출물
- cpp/ (C++ 코어 엔진)
- shared/ (공유 리소스)
- android/ (Android 바인딩)
- python/ (프로토타이핑)
- ios/ (Phase 2 예약)
- flutter/ (Phase 2 예약)
- web/ (Phase 3 예약)
- scripts/ (빌드 스크립트)

### 검증 기준
- 모든 디렉토리 존재
- .gitkeep 파일 배치 (빈 디렉토리 추적용)
- 기존 파일과 충돌 없음

---

## 2. 실행 내역

### 2.1 생성할 디렉토리 목록

```
IrisLensSDK/
├── cpp/
│   ├── include/iris_sdk/
│   ├── src/
│   ├── tests/
│   ├── examples/
│   ├── third_party/
│   │   ├── mediapipe/
│   │   └── opencv/
│   └── build/              (gitignore)
├── shared/
│   ├── models/
│   ├── textures/
│   └── test_data/
├── android/
│   ├── iris-sdk/
│   │   └── src/main/
│   │       ├── cpp/
│   │       └── java/com/irislenssdk/
│   └── demo-app/
│       └── src/main/
├── python/
│   ├── iris_sdk/
│   ├── examples/
│   └── tests/
├── ios/
│   ├── IrisSDK/
│   └── DemoApp/
├── flutter/
│   ├── lib/
│   ├── android/
│   └── ios/
├── web/
│   └── src/
└── scripts/
```

### 2.2 실행 명령

```bash
# 디렉토리 생성
mkdir -p cpp/include/iris_sdk cpp/src cpp/tests cpp/examples cpp/third_party/mediapipe cpp/third_party/opencv cpp/build
mkdir -p shared/models shared/textures shared/test_data
mkdir -p android/iris-sdk/src/main/cpp android/iris-sdk/src/main/java/com/irislenssdk android/demo-app/src/main
mkdir -p python/iris_sdk python/examples python/tests
mkdir -p ios/IrisSDK ios/DemoApp
mkdir -p flutter/lib flutter/android flutter/ios
mkdir -p web/src scripts

# .gitkeep 파일 생성 (빈 디렉토리 Git 추적용)
touch cpp/include/iris_sdk/.gitkeep cpp/src/.gitkeep cpp/tests/.gitkeep ...
```

---

## 3. 산출물

### 생성된 디렉토리 (43개)

| 카테고리 | 디렉토리 | 용도 |
|----------|----------|------|
| **cpp** | cpp/include/iris_sdk | 공개 헤더 |
| | cpp/src | 구현 파일 |
| | cpp/tests | 단위/통합 테스트 |
| | cpp/examples | 데스크톱 예제 |
| | cpp/third_party/mediapipe | MediaPipe 의존성 |
| | cpp/third_party/opencv | OpenCV 의존성 |
| | cpp/build | 빌드 결과물 (gitignore) |
| **shared** | shared/models | ML 모델 파일 |
| | shared/textures | 렌즈 텍스처 |
| | shared/test_data | 테스트용 데이터 |
| **android** | android/iris-sdk/src/main/cpp | JNI 코드 |
| | android/iris-sdk/src/main/java/com/irislenssdk | Kotlin SDK |
| | android/demo-app/src/main | 데모 앱 |
| **python** | python/iris_sdk | ctypes 바인딩 |
| | python/examples | 예제 |
| | python/tests | 테스트 |
| **ios** | ios/IrisSDK | iOS Framework (Phase 2) |
| | ios/DemoApp | iOS 데모 앱 (Phase 2) |
| **flutter** | flutter/lib, android, ios | Flutter Plugin (Phase 2) |
| **web** | web/src | WASM (Phase 3) |
| **scripts** | scripts/ | 빌드 스크립트 |

---

## 4. 검증 결과

### 검증 명령
```bash
find . -type d -name ".git" -prune -o -type d -print | grep -v "\.git" | wc -l
# 결과: 43개 디렉토리
```

### 검증 항목

| 검증 항목 | 결과 | 비고 |
|----------|------|------|
| 모든 디렉토리 존재 | ✅ 통과 | 43개 디렉토리 생성 확인 |
| .gitkeep 파일 배치 | ✅ 통과 | 빈 디렉토리에 .gitkeep 추가 |
| 기존 파일 충돌 없음 | ✅ 통과 | docs/, CLAUDE.md 등 유지 |
| cpp/build gitignore | ✅ 통과 | .gitignore에 build/ 패턴 존재 |

---

## 5. 이슈 및 학습

### 이슈
- 없음

### 학습 내용
- `.gitkeep` 파일은 Git이 빈 디렉토리를 추적하도록 하는 관례적 방법
- `.gitignore`에 `build/` 패턴이 있어 `cpp/build/`도 자동 무시됨

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-01-07 | 태스크 문서 생성, 실행 시작 |
| 2026-01-07 | 디렉토리 생성 완료, 검증 통과, 태스크 완료 |
