# P5-W2: Cascaded Eye Refinement (2-Stage 눈 정밀화)

## 작업 개요
- **Phase**: P5 (경쟁사 대비 품질 갭 해소)
- **기간**: P5-W1 완료 후
- **상태**: 🔄 구현 완료 (실기기 검증 대기)
- **선행 조건**: P5-W1 (스무딩이 동작해야 정밀화 효과 체감 가능)
- **근거**: Perfect Corp의 `second_step_eye_model` (228KB) 방식. Codex 권장 우선순위 2위.

## 목표

현재 V2 모델의 내장 홍채 좌표(478점 중 468~477)는 heuristic 보정으로 보완하고 있으나, 이는 robust한 정밀화가 아니다. 경량 Eye Refiner 모델(150~300KB)을 추가하여 홍채 중심/반지름의 정확도를 높인다.

### 현재 V2 홍채 추출의 한계

```cpp
// mediapipe_detector.cpp:1831, 1927 부근
// V2에서 추출한 홍채 좌표를 eye center 방향으로 "당기는" heuristic
// → 안정적이지 않고, 극단적 각도에서 부정확
```

### Perfect Corp 참조 구조

```
1차: YMK_Venus (1.3MB) → 106+ 얼굴 랜드마크
2차: second_step_eye_model (228KB) → 눈 ROI에서 홍채 정밀 보정
```

---

## W2-01: Eye Refiner 모델 선정 + TFLite 변환

### 상태: ✅ 완료

### 작업 내용

1. **모델 후보 조사**

   | 후보 | 크기 | 입력 | 출력 | 출처 |
   |------|------|------|------|------|
   | MediaPipe iris_landmark.tflite | 2.5MB | 64×64 | 71점 | 이미 V1에서 사용 |
   | 경량 iris regressor (INT8) | 150~300KB | 32×32 or 48×48 | 5점(중심+4경계) + 품질점수 | 자체 학습 or 공개 모델 |
   | EyeSeg lite | ~200KB | 64×64 | 홍채 마스크 + 중심 | 공개 논문 기반 |

2. **모델 선택 기준**
   - 크기: 300KB 이하 (APK 부담 최소)
   - 추론 시간: 5ms 이하 (CPU, ARM NEON)
   - 출력: 최소 홍채 중심(x,y) + 반지름 + 품질 점수
   - 보너스: 눈꺼풀 오클루전 비율 (W3에서 활용)

3. **실용적 첫 접근: V1 iris_landmark를 V2에서 재활용**
   - V2에서 Face Landmark로 눈 ROI를 잡고
   - 기존 iris_landmark.tflite (64×64)를 2차로 실행
   - **주의**: V2 초기화 경로에서 `iris_landmark_model.reset()` / `iris_landmark_interpreter.reset()`이 명시적으로 호출됨 (`mediapipe_detector.cpp:768-771`). 따라서 "조건부 활성화" 수준이 아니라, **V2 경로에서 iris_landmark 모델의 추가 로딩과 interpreter 수명주기 관리가 새로 필요**
   - 구현 범위: V2 초기화 시 iris_landmark 모델을 해제하지 않는 분기 추가 + Eye Refiner 모드 플래그
   - 크기 부담: 2.5MB (이미 V1에 포함)
   - 향후 경량 모델로 교체 가능

### 수정 대상 파일

| 파일 | 수정 내용 |
|------|----------|
| `shared/models/` | 경량 모델 추가 (또는 기존 iris_landmark 재활용) |
| `cpp/CMakeLists.txt` | 모델 경로 설정 |

### 결정 필요 사항

- [ ] V1 iris_landmark 재활용 vs 경량 커스텀 모델 학습
- [ ] Perfect Corp처럼 눈꺼풀 오클루전 출력을 모델에 포함할지

---

## W2-02: Eye Refiner 추론 파이프라인 통합

### 상태: ✅ 완료

### 작업 내용

1. **눈 ROI 크롭**
   ```
   V2 Face Landmark에서 눈 주변 랜드마크 추출:
     좌눈: landmarks[33, 133, 159, 145, ...] → bounding box → 20% 확장
     우눈: landmarks[362, 263, 386, 374, ...] → bounding box → 20% 확장
   크롭된 이미지를 Eye Refiner 입력 크기로 리사이즈
   ```

2. **추론 + 결과 병합**
   ```
   Eye Refiner 출력:
     - refined_iris_center (x, y) — ROI 상대 좌표
     - refined_iris_radius
     - quality_score (0~1)
     - eyelid_occlusion_ratio (0~1) — optional, W3용
   
   ROI 상대 좌표 → 전체 프레임 정규화 좌표로 역변환
   기존 V2 coarse 좌표를 refined 좌표로 교체
   ```

3. **IrisResult 확장**
   ```cpp
   struct IrisResult {
       // 기존 필드들...
       float iris_quality_left;    // Eye Refiner 품질 점수
       float iris_quality_right;
       float eyelid_ratio_left;    // 눈꺼풀 가림 비율 (W3용)
       float eyelid_ratio_right;
   };
   ```

### 수정 대상 파일 (전 레이어 일괄 반영 필요)

| 레이어 | 파일 | 수정 내용 |
|--------|------|----------|
| C++ Core | `cpp/include/iris_sdk/types.h` | IrisResult에 `iris_quality_left/right`, `eyelid_ratio_left/right` 추가 |
| C++ Core | `cpp/src/mediapipe_detector.cpp` | V2 경로에 Eye Refiner 추론 추가, 신규 필드 채우기 |
| C++ Core | `cpp/include/iris_sdk/mediapipe_detector.h` | Eye Refiner 관련 설정 |
| C API | `cpp/include/iris_sdk/sdk_api.h` | `IrisResultC` 구조체에 신규 필드 추가 |
| C API | `cpp/src/sdk_api.cpp` | C++ → C 변환에 신규 필드 반영 |
| JNI | `android/iris-sdk/src/main/cpp/iris_jni.cpp` | JNI field cache에 신규 필드 추가 |
| Java | `android/iris-sdk/src/main/java/com/irislenssdk/IrisResult.java` | 신규 필드 추가 |
| Kotlin | `android/iris-sdk/src/main/java/com/irislenssdk/kotlin/` | Kotlin wrapper 업데이트 |

---

## W2-03: 조건부 실행 정책

### 상태: ✅ 완료

### 배경

Codex 권장: "yes to a second-step eye model, no to making it an expensive always-on default before you have gating."

### 작업 내용

1. **실행 조건**
   ```
   항상 실행:
     - HQ 모드 (사진 촬영, 녹화)
   
   조건부 실행:
     - confidence < 0.7 (불확실할 때 정밀화 필요)
     - 아웃라이어 감지 후 재검증용
     - iris_radius가 작음 (원거리 = 정밀도 낮음)
   
   스킵:
     - confidence > 0.9 + 적절한 크기 → coarse로 충분
   
   ⚠️ 참고: "pose 변화 > threshold"는 현재 구현 불가.
   face_rotation[0..2]이 모두 0.0f로 하드코딩됨 (mediapipe_detector.cpp:2972-2974).
   pose 기반 gating은 face_rotation 구현 이후에 추가.
   ```

2. **API**
   ```cpp
   enum class EyeRefinerPolicy {
       ALWAYS,      // 항상 실행 (HQ)
       CONDITIONAL, // 조건부 (기본값)
       NEVER        // 비활성화 (저사양)
   };
   ```

### 수정 대상 파일

| 파일 | 수정 내용 |
|------|----------|
| `cpp/src/mediapipe_detector.cpp` | 조건부 실행 로직 |
| `cpp/include/iris_sdk/sdk_api.h` | 정책 설정 C API |

---

## W2-04: V2 Heuristic 대비 정밀도 벤치마크

### 상태: ⏳ 실기기 검증 시 수행

### 작업 내용

1. **테스트 이미지 세트 구성**
   - 정면/15°/30°/45° 각도별 이미지
   - 눈 크기 다양 (근거리/원거리)
   - 눈 일부 가림 (머리카락, 안경 프레임)

2. **정밀도 측정**
   - 홍채 중심 오차 (GT 대비 픽셀 거리)
   - 반지름 오차 (GT 대비 비율)
   - 검출 실패율 (false negative)
   - V2 heuristic vs Eye Refiner 비교

3. **속도 측정**
   - V2 only: 추론 시간
   - V2 + Eye Refiner: 추론 시간
   - 추가 지연이 5ms 이내인지 확인

### 수정 대상 파일

| 파일 | 수정 내용 |
|------|----------|
| `cpp/tests/test_eye_refiner_benchmark.cpp` (신규) | 정밀도/속도 벤치마크 |

---

## 예상 효과

- **홍채 중심 정확도**: 30%+ 개선 (특히 비스듬한 각도)
- **반지름 안정성**: heuristic 보정 의존 제거
- **렌즈 피팅 품질**: 위치 오류가 가장 눈에 띄는 결함이므로 체감 효과 큼
- **추가 지연**: 5ms 이내 (조건부 실행 시 평균 2~3ms)

---

## 실행 내역 (2026-04-10)

### W2-01: 결정 사항
- **V1 iris_landmark.tflite 재활용** 선택 (2.5MB, 이미 V1에 포함)
- V2 초기화 시 `iris_landmark_model.reset()` 대신 조건부 유지
- `EyeRefinerPolicy` enum으로 ALWAYS/CONDITIONAL/NEVER 제어

### W2-02: 구현 완료
- `EyeRefinerPolicy` enum 추가 (`types.h`)
- `IrisResult`에 5개 필드 추가: `iris_quality_left/right`, `eyelid_ratio_left/right`, `eye_refiner_used`
- V2 검출 흐름에 Eye Refiner 삽입 (`mediapipe_detector.cpp`)
  - `shouldRunEyeRefiner()`: 조건부 실행 판단 (confidence < 0.7 or radius < 8px)
  - `runEyeRefiner()`: extractEyeRegionMediaPipe + runIrisLandmark 재활용
  - 좌우 눈 각각 독립 정밀화 + 품질 점수 산출
- 전 레이어 반영: C API (`sdk_api.h/cpp`), JNI (`iris_jni.cpp`, `jni_utils.h`), Java (`IrisResult.java`), Kotlin (`IrisResultKt.kt`)

### W2-03: 구현 완료
- `EyeRefinerPolicy::Conditional` 기본값
- 실행 조건: confidence < 0.7f || iris_radius < 8.0px
- pose 기반 gating은 face_rotation 구현 이후 추가 예정

### W2-04: 빌드 검증
- C++ 컴파일 성공 (cmake-build-debug)
- TFLite 링커 에러는 기존 이슈 (test_mediapipe_detector 타겟, 우리 코드 무관)
- 벤치마크 테스트는 실기기에서 수행 예정

### 변경 파일 목록

| 파일 | 변경 내용 |
|------|----------|
| `cpp/include/iris_sdk/types.h` | EyeRefinerPolicy enum + IrisResult 5개 필드 |
| `cpp/include/iris_sdk/sdk_api.h` | IrisEyeRefinerPolicy C enum + IrisResult C 필드 + API 함수 |
| `cpp/include/iris_sdk/mediapipe_detector.h` | setEyeRefinerPolicy/getEyeRefinerPolicy |
| `cpp/src/mediapipe_detector.cpp` | Eye Refiner 파이프라인 전체 구현 |
| `cpp/src/sdk_api.cpp` | C↔C++ 변환에 신규 필드 반영 |
| `android/.../jni_utils.h` | JniCache Eye Refiner 필드 ID |
| `android/.../iris_jni.cpp` | JNI 캐시 초기화 + 양방향 복사 |
| `android/.../IrisResult.java` | Java 필드 5개 + reset/copyFrom/toString |
| `android/.../IrisResultKt.kt` | Kotlin data class + fromJava 변환 |
