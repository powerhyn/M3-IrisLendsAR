# IrisLensSDK 성능 및 정확도 개선 제안서

**작성일**: 2025-01-26
**작성자**: Claude (AI Assistant)
**대상**: IrisLensSDK 개발팀
**관련 문서**: 
- `docs/PERFORMANCE_REPORT.md`
- `cpp/src/mediapipe_detector.cpp`

---

## 1. 개요

본 문서는 IrisLensSDK의 핵심 목표인 **'실시간성(30fps+)'**과 **'정밀한 홍채 추적'**을 달성하기 위한 구체적인 기술 제안을 담고 있습니다. 현재 성능 리포트 분석 결과(21-26fps)를 바탕으로, 즉시 적용 가능한 최적화 방안과 중장기적인 아키텍처 개선안을 제시합니다.

---

## 2. 반응 속도(FPS) 개선 설계

### A. 스마트 트래킹 (Smart Tracking) 모드 구현
매 프레임 `Face Detection`을 수행하는 것은 자원 낭비입니다. 이전 프레임의 정보를 활용하여 검출 단계를 생략합니다.

*   **현재 흐름**: `Face Detection` (10ms) → `Face Landmark` (15ms) → `Iris Landmark` (5ms) = **Total 30ms**
*   **개선 흐름**:
    1.  **Frame 0**: 전체 파이프라인 수행. 얼굴 ROI 확보.
    2.  **Frame 1~N**: `Face Landmark` 모델에 이전 프레임 ROI(확장된 영역)를 직접 입력.
        - `Face Detection` 생략 (**10ms 절약**)
        - `Face Landmark`의 `Presence Confidence`가 임계값(0.5) 미만이면 다시 Detection 수행.
*   **구현 포인트**:
    - `MediaPipeDetector::Impl` 내에 `prev_face_rect`와 `tracking_state` 관리 로직 강화.
    - ROI 계산 시 얼굴 움직임을 고려하여 20~30% 마진(Padding) 추가.

### B. GPU Delegate 활성화 (Android)
CPU 추론만으로는 한계가 있습니다. NPU/GPU 가속을 필수로 활성화해야 합니다.

*   **설계**:
    - **JNI 레이어**: `nativeInit` 시점에 Java의 `EGLContext`를 네이티브로 전달하지 않아도, TFLite GPU Delegate는 내부적으로 `EGLContext`를 생성할 수 있습니다. 단, `OpenCL` 또는 `OpenGL ES 3.1` 지원 여부를 명확히 체크해야 합니다.
    - **C++ 레이어**: `TfLiteGpuDelegateV2Create` 옵션 튜닝.
        - `inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY`
        - `inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED` (발열 제어)

---

## 3. 홍채 검출 정확도 향상 설계

### C. 단일 모델(Single Model) 전략: Face Mesh V2 활용
현재 V1 파이프라인은 3개의 모델을 순차 실행합니다. 이를 V2 모델 하나로 통합하여 속도와 정확도를 동시에 잡습니다.

*   **배경**: `Face Landmark V2` (Attention Mesh) 모델은 홍채 랜드마크(468~477번 인덱스)를 내장하고 있습니다.
*   **전략**:
    - `Iris Landmark` 전용 모델(64x64 입력) 실행을 **제거**.
    - `Face Landmark V2`의 출력 랜드마크에서 직접 홍채 좌표 추출.
    - **이점**: 모델 로딩 시간 감소, 메모리 절약(약 3MB), 추론 시간 단축(5ms 절약).

### D. 지터링(Jittering) 방지: One-Euro Filter 적용
검출된 홍채 좌표가 미세하게 떨리는 현상을 방지하여 사용자 경험을 개선합니다.

*   **알고리즘**: `One-Euro Filter` (저속 이동 시 지터링 제거, 고속 이동 시 반응성 유지)
*   **설계**:
    - `IrisFilter` 클래스 신설.
    - 각 랜드마크(x, y, r)에 대해 독립적인 필터 적용.
    - 파라미터 튜닝: `min_cutoff` (떨림 제거 강도), `beta` (반응 속도).

---

## 4. 실행 로드맵 (Action Plan)

### Phase 1: 기반 최적화 (현재 진행 중)
- [ ] `008` 뷰티 필터 및 직접 렌더링 구현 (성능 병목 해소)
- [ ] `sdk_api.cpp` 누락된 API 구현

### Phase 2: 트래킹 및 가속 (우선순위 높음)
- [ ] **Smart Tracking** 로직 적용 (`Face Detection` 스킵)
- [ ] **GPU Delegate** 옵션 튜닝 및 활성화 검증

### Phase 3: 모델 고도화 (정확도)
- [ ] **Face Mesh V2** 단일 모델 체제로 전환 (`Iris Landmark` 모델 제거)
- [ ] **One-Euro Filter** 적용하여 렌더링 안정화

---

## 5. 결론

제안된 **'Smart Tracking'**과 **'V2 단일 모델 전략'**을 적용하면, 이론적으로 프레임당 처리 시간을 **30ms → 15ms** 수준으로 단축하여 안정적인 **45~60fps** 달성이 가능할 것으로 예상됩니다. 이는 상용 AR 앱 수준의 퍼포먼스입니다.

---

## 6. 구현 상태 분석 (2025-01-26 업데이트)

**작성자**: Claude (Implementation Review)

### 6.1 제안별 현재 구현 상태

| 제안 | 상태 | 코드 위치 | 비고 |
|------|------|-----------|------|
| A. Smart Tracking | ✅ **구현됨** | `mediapipe_detector.cpp:2003-2038` | `skip_face_detection` 로직 동작 중 |
| B. GPU Delegate | ✅ **인프라 완료** | `CMakeLists.txt:185-208`, `mediapipe_detector.cpp:460` | 실제 활성화 검증 필요 |
| C. Face Mesh V2 | ⏳ **미구현** | - | V1 파이프라인 사용 중 |
| D. One-Euro Filter | ⏳ **미구현** | - | 지터링 방지 없음 |

### 6.2 상세 분석

#### A. Smart Tracking ✅
```cpp
// mediapipe_detector.cpp:2003-2021
bool skip_face_detection = false;
const bool tracking_valid = impl_->use_tracking
    && impl_->prev_confidence >= impl_->min_presence_confidence
    && impl_->prev_face_rect.width > 0;

if (tracking_valid) {
    face_rect = impl_->prev_face_rect;  // 이전 ROI 재사용
    skip_face_detection = true;          // Face Detection 스킵!
}
```
- `prev_face_rect` 활용한 Face Detection 스킵 ✅
- `min_presence_confidence = 0.5f` 임계값 ✅
- `use_tracking = true` 기본값 ✅

#### B. GPU Delegate ✅ (검증 필요)
```cpp
// CMakeLists.txt:196-197
target_link_libraries(iris_sdk PRIVATE tflite_gpu_delegate)
target_compile_definitions(iris_sdk PRIVATE IRIS_SDK_HAS_GPU_DELEGATE)
```
- Pre-built GPU Delegate 라이브러리 연결 ✅
- EGL, GLESv2 의존성 설정 ✅
- **⚠️ 실제 활성화 여부 디바이스 로그 확인 필요**

#### C. Face Mesh V2 ⏳
- 현재 V1 파이프라인: `Face Detection → Face Mesh V1 → Iris Landmark`
- V2 전환 시: `Face Detection → Face Mesh V2 (홍채 내장)`
- **예상 효과**: 5ms 절약 + 메모리 3MB 절약

#### D. One-Euro Filter ⏳
- 현재 지터링 방지 필터 없음
- 홍채 좌표 미세 떨림 발생 가능
- **구현 필요**: `IrisFilter` 클래스 신설

### 6.3 실행 로드맵 현황 업데이트

#### Phase 1: 기반 최적화 ✅ **완료**
- [x] `008` 뷰티 필터 및 직접 렌더링 구현
- [x] `sdk_api.cpp` 누락된 API 구현
- [x] JNI NV21→RGBA 고속 변환 구현

#### Phase 2: 트래킹 및 가속 🔄 **진행 필요**
- [x] **Smart Tracking** 로직 (이미 구현됨)
- [ ] **GPU Delegate** 활성화 검증 및 최적화

#### Phase 3: 모델 고도화 ⏳ **대기**
- [ ] **Face Mesh V2** 단일 모델 체제로 전환
- [ ] **One-Euro Filter** 적용

### 6.4 즉시 조치 권장 사항

1. **GPU Delegate 검증** (우선순위 🔴)
   - `IrisLensSDK.isUsingGpu()` 호출하여 실제 GPU 모드 확인
   - 로그: `nativeIsUsingGpu()` 반환값 확인
   - GPU 미활성화 시 원인 분석 (OpenGL ES 버전, EGLContext 등)

2. **성능 프로파일링**
   - 현재 FPS 측정 (뷰티 필터 ON/OFF)
   - Face Detection / Face Mesh / Iris Landmark 각 단계별 시간 측정

3. **V2 모델 전환 검토** (GPU 최적화 후)
   - Face Mesh V2 모델 파일 확보
   - Iris Landmark 모델 제거 테스트
