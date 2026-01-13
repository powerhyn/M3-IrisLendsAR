# P1-W5-05: 모델 에셋 통합

**태스크 ID**: P1-W5-05
**상태**: ✅ 완료
**시작일**: 2026-01-13
**완료일**: 2026-01-13

---

## 1. 계획

### 목표
Android 데모 앱에서 TFLite 모델 파일을 assets에 포함하여 SDK가 완전히 초기화될 수 있도록 함

### 산출물
| 파일 | 설명 |
|------|------|
| `assets/models/*.tflite` | TFLite 모델 파일 3종 |
| `build.gradle.kts` 수정 | C++ STL 정적 링크 설정 |

### 검증 기준
- [x] 모델 파일 assets에 포함
- [x] 앱 시작 시 모델 추출 성공
- [x] SDK Ready: true 상태 달성
- [x] 네이티브 라이브러리 로딩 성공

### 선행 조건
- P1-W5-04 AAR 빌드 완료 ✅
- P1-W6-02 CameraX 연동 완료 ✅

---

## 2. 분석

### 2.1 문제 상황

데모 앱 테스트 시 SDK가 초기화되지 않는 문제 발생:
- `IrisLensSDK.isReady()` = false
- 모델 파일이 assets에 없어 SDK 초기화 실패

### 2.2 필요한 모델 파일

| 모델 파일 | 크기 | 용도 |
|----------|------|------|
| `face_detection_short_range.tflite` | 224KB | 얼굴 검출 |
| `face_landmark.tflite` | 1.2MB | 얼굴 랜드마크 |
| `iris_landmark.tflite` | 2.5MB | 홍채 랜드마크 |

### 2.3 모델 로딩 흐름

```
IrisLensSDK.init(context)
    └─ extractModelsFromAssets(context)
        └─ assets/models/*.tflite → files/iris_models/
            └─ nativeInit(modelDir)
                └─ SDKManager::init(path)
                    └─ MediaPipeDetector::initialize()
```

---

## 3. 실행 내역

### 3.1 모델 파일 복사

```bash
# assets/models 디렉토리 생성 및 모델 복사
mkdir -p android/demo-app/src/main/assets/models
cp shared/models/face_detection_short_range.tflite \
   shared/models/face_landmark.tflite \
   shared/models/iris_landmark.tflite \
   android/demo-app/src/main/assets/models/
```

### 3.2 C++ STL 설정 수정

**문제**: `libc++_shared.so` 로딩 실패
```
dlopen failed: library "libc++_shared.so" not found
```

**원인**:
- `-DANDROID_STL=c++_shared` 설정 (공유 런타임)
- `excludes += "libc++_shared.so"` (번들 제외)

**해결**: 정적 C++ 런타임으로 변경

```kotlin
// iris-sdk/build.gradle.kts
arguments += listOf(
    "-DANDROID_STL=c++_static",  // 변경: c++_shared → c++_static
    ...
)
```

### 3.3 빌드 및 테스트

```bash
./gradlew :iris-sdk:clean :demo-app:clean
./gradlew :demo-app:assembleDebug
adb install -r demo-app-debug.apk
```

---

## 4. 검증 결과

### 검증 항목

| 항목 | 결과 | 비고 |
|------|------|------|
| 네이티브 라이브러리 로딩 | ✅ | `Native library loaded successfully` |
| 모델 추출 | ✅ | 3개 파일 모두 추출 |
| SDK 초기화 | ✅ | `SDK Ready: true` |
| 버전 정보 | ✅ | `SDK Version: 1.0.0` |

### 로그 확인

```
D IrisLensSDK-Demo: Initializing IrisLensSDK...
I IrisLensSDK: Native library loaded successfully
D IrisLensSDK-Demo: Library loaded: true
D IrisLensSDK: Extracted model: face_detection_short_range.tflite
D IrisLensSDK: Extracted model: face_landmark.tflite
D IrisLensSDK: Extracted model: iris_landmark.tflite
I IrisLensSDK-Demo: SDK Version: 1.0.0
I IrisLensSDK-Demo: SDK Ready: true
D IrisLensSDK-Demo: IrisLensSDK initialized successfully
```

---

## 5. 이슈 및 학습

### 이슈

| ID | 내용 | 상태 | 해결방안 |
|----|------|------|----------|
| 1 | libc++_shared.so 누락 | ✅ 해결 | c++_static으로 변경 |

### 결정 사항

| 결정 | 이유 |
|------|------|
| 정적 C++ 런타임 사용 | SDK 라이브러리로서 런타임 의존성 최소화 |
| 모델 파일 demo-app에 포함 | SDK AAR은 가볍게, 앱에서 모델 제공 |

### 학습 내용

- Android NDK STL 옵션: `c++_shared` vs `c++_static`
  - `c++_shared`: 여러 라이브러리가 C++ 런타임 공유 (충돌 방지)
  - `c++_static`: 각 라이브러리에 C++ 런타임 내장 (의존성 없음)
- SDK 라이브러리는 정적 링크 권장
- `IrisLensSDK.extractModelsFromAssets()` 자동 추출 동작 확인

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-01-13 | 태스크 문서 생성 및 완료 |
