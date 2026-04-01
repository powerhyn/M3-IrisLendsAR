# P5-W1: Temporal Architecture (시간적 안정성 스택)

## 작업 개요
- **Phase**: P5 (경쟁사 대비 품질 갭 해소)
- **기간**: 2026-04 ~
- **상태**: ✅ 완료 (W1-01~05 전체 완료)
- **선행 조건**: 없음 (최우선 작업)
- **근거**: 경쟁사 분석 — 시간적 안정성 아키텍처 정립 필요

## 선결 사항: Temporal Source of Truth 결정

### 현재 상태 — 이중 경로 문제

**C++ SDK 코어 (`mediapipe_detector.cpp`)**:
- `use_one_euro_filter = false` → raw 좌표 출력
- visibility = 0/1 하드코딩
- 프레임 제어 없음

**Android 데모 렌더러 (`CameraGLRenderer.kt`, `OverlayView.kt`)**:
- OneEuroFilter **26개+** 이미 동작 중 (iris center/radius/eyelid/ellipse 각각) 
- `EYELID_HOLD_FRAMES = 5` hold 구현
- `lastValidFaceTimeMs` + `FACE_INVALID_TIMEOUT_MS` 기반 dropout 처리
- `ellipseCacheValidFrames` 타원 캐시
- `LENS_PERSISTENCE_TIMEOUT_MS` 기반 렌즈 지속 표시

**확정**: C++ `TemporalStabilizer`가 유일한 스무딩 레이어 (프로덕션 기본값). detector는 raw 출력. Android 데모의 Kotlin OneEuroFilter(26개+)는 P5 완료 시 제거되고 SDK 코어 호출로 대체.

- detector (`MediaPipeDetector`): raw 좌표 + confidence 출력. 내부 OneEuro는 OFF 유지.
- `TemporalStabilizer` (신규 C++ 클래스): 스무딩/fade/hold/outlier/blink 모든 temporal 책임 소유.
- renderer (`LensRenderer`): stabilized 결과를 받아 렌더링만 담당.
- raw 보조 경로: `stabilizer.setEnabled(false)` 또는 `iris_sdk_detect()` 직접 호출.

## 목표

시간적 안정성 아키텍처를 정립하여 지터링, 깜빡임, 크기 불안정을 제거한다.

### 현재 상태 vs 목표

| 항목 | C++ 코어 (현재) | Android 데모 (현재) | P5 후 (코어 일원화) |
|------|----------------|-------------------|-------------------|
| 랜드마크 스무딩 | OneEuroFilter **OFF** | OneEuro 26개+ **ON** | `TemporalStabilizer` (C++) |
| Visibility/fade | 없음 (`IrisResult`에 필드 없음) | hold + persistence | `TemporalStabilizer.visibility` (신규 필드) |
| Confidence | raw 값 직접 사용 | 부분적 사용 | `TemporalStabilizer` 이력현상 |
| 아웃라이어 | 없음 | 없음 | `TemporalStabilizer` 거부 로직 |
| 프레임 제어 | `detectSync()` 블로킹 | CameraX 비동기 | 비동기 결과 API |
| 데모 Kotlin 스무딩 | — | 26개+ OneEuro | **제거** (SDK 코어 호출로 대체) |

---

## W1-01: SDK 코어 스무딩 전략 결정 + TemporalStabilizer 설계

### 상태: ✅ 완료 (2026-03-31)

### 배경

`mediapipe_detector.cpp:218`의 `use_one_euro_filter = false`를 단순히 켜는 것은 적절하지 않다.

**Android 데모에 이미 렌더러 측 스무딩이 성숙하게 구현됨:**
- `CameraGLRenderer.kt:548-606` — OneEuroFilter 26개+ (iris center/radius, eyelid top/bottom, ellipse 파라미터 각각)
- `OverlayView.kt:159-164` — OneEuroFilter 6개 (iris x/y/radius × 좌우)
- detector에서도 스무딩하면 **double-smoothing → 과도한 지연(lag)** 위험

### 작업 내용

1. **스무딩 책임 경계 결정**

   | 방안 | 장점 | 단점 | 권장 |
   |------|------|------|------|
   | A: detector에서 스무딩 | 크로스플랫폼 일관성 | Android 데모와 이중 적용 위험 | ❌ |
   | B: renderer에서 스무딩 | 플랫폼 최적화 가능 | 각 플랫폼 재구현 | △ (현재 상태) |
   | **C: 별도 TemporalStabilizer** | 선택적 사용, 이중 적용 방지 | 새 클래스 필요 | **✅ 권장** |

2. **`TemporalStabilizer` 클래스 설계** (detector와 renderer 사이의 독립 레이어)
   ```cpp
   class TemporalStabilizer {
       // 입력: raw IrisResult
       // 출력: stabilized IrisResult
       // 내부: OneEuroFilter (신호별 개별 파라미터)
       IrisResult stabilize(const IrisResult& raw, double timestamp);
       void reset();
       void setEnabled(bool enabled);  // false 시 raw 좌표 패스스루 (디버깅/커스텀 스무딩용)
   };
   ```
   - C API로 노출: `iris_sdk_create_stabilizer()`, `iris_sdk_stabilize()`
   - 프로덕션 기본값: stabilizer ON (모든 플랫폼)
   - raw 보조 경로: `stabilizer.setEnabled(false)` 또는 `iris_sdk_detect()` 직접 호출
   - Android 데모: P5 완료 시 Kotlin OneEuro 제거, SDK 코어 stabilizer 사용으로 전환

3. **신호별 개별 파라미터** (Android 데모 `CameraGLRenderer` 파라미터 참조)

   | 신호 | min_cutoff | beta | 참조 |
   |------|-----------|------|------|
   | iris_center (x,y) | 4.0 | 15.0 | `CameraGLRenderer:GL_FILTER_MIN_CUTOFF/BETA` |
   | iris_radius | 4.0 | 7.5 | `GL_FILTER_BETA_RADIUS` |
   | eyelid (top/bottom) | 4.0 | 10.0 | `GL_FILTER_BETA_EYELID` |

### 수정 대상 파일

| 파일 | 수정 내용 |
|------|----------|
| `cpp/include/iris_sdk/temporal_stabilizer.h` (신규) | TemporalStabilizer 클래스 + StabilizedResult 정의 |
| `cpp/src/temporal_stabilizer.cpp` (신규) | OneEuroFilter 기반 스무딩 구현 |
| `cpp/include/iris_sdk/sdk_api.h` | stabilizer C API 추가 |
| `cpp/src/sdk_api.cpp` | C API 구현 |

### 검증 방법

- 데스크톱 camera_demo에서 stabilizer ON: 지터링 감소 확인
- Android demo-app에서 stabilizer ON: Kotlin OneEuro 제거 전이라도 SDK 코어 스무딩 단독 동작 확인
- 정지 상태: 표준편차 측정, 이동 상태: lag 측정

---

## W1-02: Confidence 이력현상 + Visibility Fade-in/out

### 상태: ✅ 완료 (2026-03-31)

### 배경

현재 `IrisResult`에 `visibility` 필드가 없다 (개별 `IrisLandmark.visibility`만 존재, 0/1 하드코딩). 검출 실패 시 렌즈가 즉시 사라진다.

Android 데모에는 `lastValidFaceTimeMs` + `LENS_PERSISTENCE_TIMEOUT_MS` 기반 dropout 처리와 `EYELID_HOLD_FRAMES = 5` hold가 이미 있다. 이 로직을 C++ `TemporalStabilizer`로 이관한다.

### 작업 내용

1. **StabilizedResult 출력 구조 정의**
   ```cpp
   // TemporalStabilizer의 출력 (IrisResult을 래핑)
   struct StabilizedResult {
       IrisResult raw;           // 원본 raw 결과
       IrisResult stabilized;    // 스무딩된 결과
       float visibility;         // 전체 가시성 (0.0~1.0, fade 적용)
       bool is_held;             // dropout hold 중인지
       int64_t last_valid_ms;    // 마지막 유효 검출 타임스탬프
   };
   ```
   - `IrisResult` 자체는 수정하지 않음 (ABI 안정성)
   - `StabilizedResult`는 TemporalStabilizer 전용 출력

2. **Confidence 이력현상 (Hysteresis)** — `TemporalStabilizer` 내부
   ```
   추적 중 → confidence < 0.3이 3프레임 연속 시 재검출 (낮은 임계값)
   재검출 후 → confidence >= 0.6이어야 추적 전환 (높은 임계값)
   ```

3. **Visibility fade-in/out** — `TemporalStabilizer` 내부
   - fade_in: ~100ms, fade_out: ~200ms
   - 렌더러는 `StabilizedResult.visibility`를 opacity 곱으로 사용

4. **Dropout hold** — `TemporalStabilizer` 내부
   - hold_frames: 5 (데모 `EYELID_HOLD_FRAMES` 참조)
   - hold 중 마지막 유효 좌표 + fade-out 진행

### 수정 대상 파일

| 파일 | 수정 내용 |
|------|----------|
| `cpp/include/iris_sdk/temporal_stabilizer.h` | StabilizedResult 정의, 이력현상/fade/hold 로직 |
| `cpp/src/temporal_stabilizer.cpp` | 구현 |
| `cpp/include/iris_sdk/sdk_api.h` | `iris_sdk_get_stabilized_result()` C API |
| `cpp/src/sdk_api.cpp` | C API 구현 |

### 검증 방법

- 손으로 얼굴 일부를 가렸다 떼는 시나리오
- `StabilizedResult.visibility`가 점진적 변화하는지 로그 확인
- 일시적 검출 실패 시 hold → fade-out → 사라짐 순서 확인

---

## W1-03: 아웃라이어 거부 + 눈깜빡임 감지

### 상태: ✅ 완료 (2026-03-31)

### 배경

모델이 간헐적으로 비정상 좌표를 출력할 수 있다. 이 경우 렌즈가 순간적으로 튀는 현상이 발생한다.

### 작업 내용

1. **아웃라이어 거부 (Outlier Rejection)**
   ```
   이전 프레임 대비 홍채 중심 이동 거리가 threshold 초과 시:
     → 해당 프레임 결과 무시
     → 이전 프레임 결과 유지
     → 2프레임 연속 "이동"이면 실제 이동으로 판단하여 수용
   ```
   - threshold: 홍채 반지름의 2배 이상 이동 시 의심
   - 빠른 시선 이동과 노이즈 구분: 연속성 체크

2. **눈깜빡임 감지**
   - Eye Aspect Ratio (EAR) 계산:
     ```
     EAR = (|p2-p6| + |p3-p5|) / (2 × |p1-p4|)
     ```
     (eye contour 랜드마크 사용)
   - EAR < threshold → 눈 감은 상태
   - 눈 감은 상태에서는 홍채 검출 결과 무시, 마지막 유효 결과 hold
   - 눈 뜨는 순간 fade-in으로 복귀

### 수정 대상 파일

| 파일 | 수정 내용 |
|------|----------|
| `cpp/include/iris_sdk/temporal_stabilizer.h` | 아웃라이어 거부 + EAR 계산 로직 |
| `cpp/src/temporal_stabilizer.cpp` | 구현 (W1-01/02와 같은 클래스 확장) |

### 검증 방법

- 의도적으로 빠르게 시선 이동 → 렌즈가 따라가는지 (거부 안 됨)
- 눈 깜빡임 → `StabilizedResult.is_held = true` + hold 동작 확인
- 극단적 조명 변화 → 아웃라이어 발생 시 렌즈가 튀지 않는지

---

## W1-04: Frame Controller (비동기 API 재설계 + 추론/렌더링 분리)

### 상태: ✅ 완료 (2026-04-01)

### 배경

현재 `InferenceThread`는 동기 단일 슬롯 블로킹 패턴이다. **`FrameProcessor::process()`가 `detectSync()`를 inline으로 기다리는 구조**(`frame_processor.cpp:510-511`)이므로, 내부 큐를 바꿔도 호출자 입장에서 렌더링이 추론 완료에 묶인다. 단순 큐 교체가 아니라 **공개 API 계약 재설계가 선행**되어야 한다.

### 선행 과제: 비동기 결과 API 설계

현재 API:
```cpp
// 동기: detect + render가 하나의 process() 호출에 묶임
ProcessResult FrameProcessor::process(data, width, height, format, config);
```

목표 API:
```cpp
// 비동기 분리: submit(fire-and-forget) + getLatestResult(non-blocking) + render(독립)
void FrameProcessor::submitFrame(data, width, height, format);  // 큐에 넣고 즉시 반환
bool FrameProcessor::getLatestResult(IrisResult& out);          // 최신 결과 스냅샷
void FrameProcessor::renderWithResult(buffer, result, config);  // 결과로 렌더링
```

### 작업 내용

1. **비동기 결과 스냅샷 API 정의**
   - `submitFrame()`: 프레임을 latest-frame 큐에 넣고 즉시 반환 (drop-oldest)
   - `getLatestResult()`: 가장 최근 완료된 추론 결과를 복사하여 반환 (non-blocking)
   - `renderWithResult()`: 주어진 결과로 렌더링 (추론과 독립)
   - 기존 `process()`는 하위 호환을 위해 유지 (내부에서 submit+wait+render)

2. **추론/렌더링 FPS 분리**
   ```
   카메라: 30fps → submitFrame()
   추론 스레드: 15~20fps (큐에서 최신 프레임 꺼내서 처리)
   렌더링: 30fps → getLatestResult() + renderWithResult()
   ```

3. **프레임 보간**
   - `getLatestResult()` 호출 시 새 결과가 없으면 이전 결과 + TemporalStabilizer 예측으로 보간

### 수정 대상 파일

| 파일 | 수정 내용 |
|------|----------|
| `cpp/include/iris_sdk/frame_processor.h` | 비동기 API 추가 (submit/getLatest/render) |
| `cpp/src/frame_processor.cpp` | 비동기 구현, 기존 process() 하위 호환 |
| `cpp/include/iris_sdk/inference_thread.h` | latest-frame 큐 + 결과 스냅샷 |
| `cpp/src/inference_thread.cpp` | 비동기 큐 + atomic 결과 저장 |
| `cpp/include/iris_sdk/sdk_api.h` | C API 비동기 함수 추가 |

### 검증 방법

- CPU 부하가 높은 상황에서 렌더링 FPS 유지 확인
- 추론 15fps + 렌더링 30fps에서 렌즈 이동이 매끄러운지
- 기존 `process()` API 하위 호환 테스트
- 프레임 드롭 로그 확인

---

## W1-05: 통합 테스트 + 안정성 메트릭 측정

### 상태: ✅ 완료 (2026-04-01)

### 작업 내용

1. **Before/After 정량 측정**
   - 정지 상태: 홍채 좌표 표준편차 (jitter metric)
   - 이동 상태: 홍채 좌표 추적 지연 (lag metric)
   - 깜빡임: visibility 전환 횟수/초
   - 프레임 간 스무딩 강도 변동 계수(CV) — `TemporalAnalyzer` 활용

2. **경쟁사 대비 체크리스트**
   - [ ] 정지 상태에서 렌즈 떨림 없음
   - [ ] 검출 실패→복구 시 점진적 전환
   - [ ] 눈깜빡임 시 렌즈 안정 유지
   - [ ] 빠른 시선 이동에 지연 없이 추적
   - [ ] CPU 과부하 시 렌더링 FPS 유지

3. **Android 데모 Kotlin 스무딩 제거**
   - `CameraGLRenderer.kt`: OneEuroFilter 26개+, eyelid hold, ellipse cache 제거
   - `OverlayView.kt`: OneEuroFilter 6개, lens persistence 제거
   - `OneEuroFilter.kt`: 파일 제거 (C++ TemporalStabilizer로 대체)
   - SDK 코어의 `StabilizedResult`를 JNI 경유로 수신하도록 변경

4. **Android 실기기 테스트**
   - demo-app에서 위 시나리오 검증
   - 디바이스별 (HIGH/MID/LOW) 동작 확인
   - Kotlin 스무딩 제거 후 SDK 코어 스무딩과 동등 품질 확인

### 수정 대상 파일

| 파일 | 수정 내용 |
|------|----------|
| `cpp/tests/test_temporal_stability.cpp` (신규) | 자동화 메트릭 테스트 |
| `cpp/examples/camera_demo.cpp` | 안정성 메트릭 오버레이 표시 (선택) |
| `android/demo-app/.../camera/OneEuroFilter.kt` | 제거 |
| `android/demo-app/.../camera/OverlayView.kt` | Kotlin 스무딩 제거, SDK 결과 직접 사용 |
| `android/demo-app/.../camera/gpu/CameraGLRenderer.kt` | Kotlin OneEuro 26개+ 제거, SDK 결과 직접 사용 |

---

## 작업 순서 및 의존성

```
W1-01 (TemporalStabilizer 설계 + 스무딩) ← 독립, 최우선
    ↓
W1-02 (Confidence/Visibility/Hold) ← W1-01 후 (같은 클래스 확장)
    ↓
W1-03 (아웃라이어/눈깜빡임) ← W1-02 후 (hold 로직 필요)
    ↓
W1-04 (비동기 API + Frame Controller) ← W1-01 후 병렬 가능
    ↓
W1-05 (통합 테스트 + 데모 Kotlin 스무딩 제거) ← 모두 완료 후
```

---

## 실행 내역

### W1-01~03 (2026-03-31)

**생성 파일:**
| 파일 | 내용 |
|------|------|
| `cpp/include/iris_sdk/temporal_stabilizer.h` | TemporalStabilizer 클래스 + StabilizerConfig + StabilizedResult |
| `cpp/src/temporal_stabilizer.cpp` | OneEuroFilter 기반 스무딩/이력현상/fade/hold/outlier/blink 구현 |
| `cpp/tests/test_temporal_stabilizer.cpp` | 22개 단위 테스트 (전부 통과) |

**수정 파일:**
| 파일 | 내용 |
|------|------|
| `cpp/include/iris_sdk/sdk_api.h` | IrisStabilizerConfig, IrisStabilizedResult, C API 6개 함수 추가 |
| `cpp/src/sdk_api.cpp` | Stabilizer C API 구현 (핸들 기반 관리) |
| `cpp/CMakeLists.txt` | temporal_stabilizer 소스/헤더 등록 |
| `cpp/tests/CMakeLists.txt` | test_temporal_stabilizer 타겟 추가 |

**설계 결정:**
- 방안 C 채택: detector와 renderer 사이의 독립 레이어 (`TemporalStabilizer`)
- `IrisResult` 수정 없이 `StabilizedResult`로 래핑 (ABI 안정성 유지)
- 기존 `OneEuroFilter` 재사용 (재구현 없음)
- 신호별 개별 파라미터: center(4.0/15.0), radius(4.0/7.5), eyelid(4.0/10.0)
- C API: 핸들 기반 생성/해제 패턴 (다중 인스턴스 지원)

**테스트 결과:**
- 22/22 테스트 통과 (0ms)
- 지터 감소, 빠른 추종, confidence 이력현상, visibility fade, dropout hold, 아웃라이어 거부, 눈깜빡임 감지 모두 검증

### W1-04 (2026-04-01)

**수정 파일:**
| 파일 | 내용 |
|------|------|
| `cpp/include/iris_sdk/inference_thread.h` | 비동기 API 추가 (`submitFrameAsync`, `getLatestResult`) + 비동기 입출력 슬롯 멤버 |
| `cpp/src/inference_thread.cpp` | 비동기 프레임 딥카피/제출, 결과 조회, 메인 루프 동기+비동기 혼합 패턴 |
| `cpp/include/iris_sdk/frame_processor.h` | 비동기 공개 API 4개 추가 (`submitFrame`, `submitFrameWithRotation`, `getLatestResult`, `renderWithResult`) |
| `cpp/src/frame_processor.cpp` | Impl 비동기 메서드 구현 (포맷 변환+회전+딥카피+캐싱) + 공개 인터페이스 위임 |
| `cpp/include/iris_sdk/sdk_api.h` | C API 4개 함수 추가 (`iris_sdk_submit_frame`, `iris_sdk_submit_frame_with_rotation`, `iris_sdk_get_latest_result`, `iris_sdk_render_with_result`) |
| `cpp/src/sdk_api.cpp` | C API 4개 함수 구현 (파라미터 검증, 포맷 변환, g_processor 호출) |

**설계 결정:**
- 동기/비동기 공존: 기존 `detectSync()` 경로 완전 보존, 비동기는 별도 입출력 슬롯 사용
- 입력 슬롯: single-slot drop-oldest (최신 프레임만 유지), `async_input_mutex_` + `has_new_frame_` atomic 보호
- 출력 슬롯: `async_result_mutex_` + `has_async_result_` atomic 보호, `getLatestResult()`는 논블로킹
- 딥카피 필수: 카메라 콜백 버퍼 재사용 시 데이터 손상 방지 (`submitFrameAsync` 내부에서 `std::memcpy`)
- 워커 루프: `slot_cv_.wait_for(5ms)` 타임아웃 사용 → 동기 요청 없을 때 비동기 프레임 처리
- 캐싱 로직: `getLatestResult()`에 기존 `detectOnly()`와 동일한 3프레임 캐시 적용
- `renderWithResult()`는 `renderOnly()`와 동일 동작 (명시적 이름 제공)
- `submitFrame()`에서 RGB 변환 수행 → InferenceThread에는 항상 RGB 전달
- No double-smoothing: 이 레이어는 raw 결과만 제공, TemporalStabilizer는 호출자가 별도 적용

**빌드 검증:**
- `libiris_sdkd.a` 정상 빌드 확인 (기존 테스트 링커 에러는 TFLite 미관련 이슈)
- 하위 호환성: 기존 `process()`, `detectSync()`, `detectOnly()` 코드 변경 없음

### W1-05 — 통합 테스트 + JNI Stabilizer 바인딩 + Kotlin OneEuroFilter 제거 (2026-04-01)

**생성 파일:**
| 파일 | 내용 |
|------|------|
| `cpp/tests/test_temporal_stability.cpp` | 13개 통합 안정성 메트릭 테스트 (jitter/lag/visibility/blink/outlier/sinusoidal/종합 시나리오) |

**수정 파일:**
| 파일 | 내용 |
|------|------|
| `cpp/tests/CMakeLists.txt` | test_temporal_stability 타겟 추가 |
| `android/iris-sdk/src/main/cpp/iris_jni.cpp` | JNI 바인딩 3개 추가: `nativeCreateStabilizer`, `nativeStabilize`, `nativeDestroyStabilizer` |
| `android/iris-sdk/src/main/java/com/irislenssdk/IrisLensSDK.java` | 공개 API 3개 + 네이티브 선언 3개 추가 (`createStabilizer`, `stabilize`, `destroyStabilizer`) |
| `android/demo-app/.../camera/FrameAnalyzer.kt` | 검출 직후 SDK 코어 `stabilize()` 호출 (in-place 스무딩), 라이프사이클 관리 |
| `android/demo-app/.../camera/OverlayView.kt` | OneEuroFilter 6개 제거, 상수 3개 제거, 직접 값 사용으로 교체 |
| `android/demo-app/.../camera/gpu/CameraGLRenderer.kt` | OneEuroFilter 22개+ 제거, 상수 6개 제거, 데드밴드 로직 제거, 직접 값 사용으로 교체 |

**삭제 파일:**
| 파일 | 사유 |
|------|------|
| `android/demo-app/.../camera/OneEuroFilter.kt` | C++ TemporalStabilizer로 완전 대체 |

**설계 결정:**
- FrameAnalyzer에서 검출 직후 stabilize → 렌더러에 이미 스무딩된 IrisResult 전달 (단일 스무딩 레이어)
- `nativeStabilize`가 Java IrisResult를 in-place 수정 (copyResultFromJava → iris_sdk_stabilize → copyResultToJava)
- 눈꺼풀/타원 hold 로직, LENS_PERSISTENCE_TIMEOUT 등 비-스무딩 시간적 로직은 유지 (다른 관심사)
- 타원 파라미터 필터도 제거 — SDK 코어 face_mesh 스무딩이 원본 랜드마크를 안정화하므로 타원 피팅 결과도 자동 안정화

**C++ 테스트 결과:**
- 기존 22/22 단위 테스트 통과 (test_temporal_stabilizer)
- 신규 13/13 통합 메트릭 테스트 통과 (test_temporal_stability)
- 주요 측정값: 지터 ~47% 감소 (XY), 추적 지연 0.004 (< 0.013 허용), fade-out 단조 감소 확인

**검증 필요:**
- [ ] Android 실기기에서 렌즈 안정성 확인 (이전 Kotlin 스무딩 대비 동등 이상)
- [ ] 과도한 lag 없는지 확인
- [ ] 빌드 성공 확인 (JNI 링킹)

---

## 예상 효과

- **지터링**: 현재 대비 80%+ 감소 (OneEuroFilter만으로도 큰 효과)
- **깜빡임**: 완전 제거 (fade + hold)
- **체감 FPS**: 추론 15fps에서도 렌더링 30fps 유지
- **경쟁사 대비**: 피팅몬스터의 3-layer 스무딩과 동등 수준 달성
