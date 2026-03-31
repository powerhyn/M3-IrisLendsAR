# P5-Preview: 경쟁사 분석 기반 품질 개선 로드맵

## Phase 개요
- **Phase**: P5 (경쟁사 대비 품질 갭 해소 — Competitive Quality Parity)
- **기간**: 2026-04 ~
- **상태**: 📋 계획 수립 완료
- **근거**: 피팅몬스터(InterVision210) + Perfect Corp(PerfectLib) SDK 역분석
- **목표**: 렌즈 피팅 품질을 경쟁사 수준으로 끌어올리기 위한 핵심 3개 축 개선

---

## 1. 분석 인사이트 요약

### 1.1 경쟁사 SDK 분석 결과

| 항목 | 피팅몬스터 (InterVision210) | Perfect Corp (PerfectLib) | IrisLensSDK (현재) |
|------|---------------------------|--------------------------|-------------------|
| **ML 프레임워크** | TFLite (via MediaPipe Graph) | MNN (FP16) | TFLite (직접 호출) |
| **스무딩** | 3-layer (Landmark×2 + Visibility) | FaceAlignMotionSmoother | C++ Core: OneEuroFilter(**OFF**) / Android Demo: OneEuro 26개+hold+cache(**ON**) |
| **렌즈 렌더링** | 3D FaceGeometry | PBR (Normal Map + IBL) | C++ Core: 2D 오버레이 / Android Demo: GPU 셰이더(타원+눈꺼풀+8블렌드) |
| **프레임 제어** | FlowLimiter + PacketThinner | Triple Buffering | 없음 |
| **눈 정밀화** | IrisLandmarkLeftAndRightGpu | second_step_eye_model (228KB) | V2 내장 홍채 (heuristic 보정) |
| **피부 분할** | 랜드마크 기반 (selfie_seg 미사용) | 경량 파싱 모델 추정 (~413KB) | 16MB 6-class seg (과도) |
| **SDK 크기** | ~33MB | ~18MB | ~9MB (추정) |

### 1.2 핵심 발견사항

**Claude + Codex(GPT-5.4) 공통 합의:**

1. **C++ 코어의 스무딩은 꺼져있으나, Android 데모에는 성숙한 렌더러 측 스무딩이 이미 존재** — `mediapipe_detector.cpp`의 `use_one_euro_filter = false`이지만, `CameraGLRenderer.kt`에 OneEuroFilter 26개+, 눈꺼풀 hold, 타원 캐시가 동작 중. P5에서 C++ `TemporalStabilizer`로 일원화하고 데모 Kotlin 스무딩은 제거한다.
2. **안정성은 시스템이지 단일 필터가 아니다** — 3-layer 아키텍처 필요 (프레임 제어 / 상태 관리 / 기하 필터링). C++ `TemporalStabilizer`가 이 모든 책임을 소유한다.
3. **C++ LensRenderer는 2D 오버레이이지만, Android 데모 CameraGLRenderer는 이미 GPU 렌즈 셰이더 보유** — 타원 마스크, 눈꺼풀 클리핑, 8종 블렌드 모드, sclera protection이 Kotlin/GLES로 구현됨. **W3는 신규 구현이 아니라 데모→SDK 코어 이관이 핵심.**

### 1.3 P5 핵심 설계 결정 (확정)

> **Temporal 정책**: SDK 코어 C++ `TemporalStabilizer`가 유일한 스무딩 레이어이며 프로덕션 기본값이다. raw 좌표는 `stabilizer.setEnabled(false)` 또는 별도 API로 접근 가능한 보조 경로로 유지한다. Android 데모의 Kotlin OneEuroFilter(26개+)는 P5 완료 시 제거되고 SDK 코어 호출로 대체된다.

이 결정의 파생:
- **W1**: `TemporalStabilizer`가 confidence/visibility/outlier/blink 모두 소유
- **W3**: 데모 Kotlin 렌더링 중 스무딩은 제거, GPU 셰이더만 C++ 포팅
- **W2**: IrisResult 확장 시 전 레이어(C++ types.h → sdk_api.h → iris_jni.cpp → IrisResult.java) 일괄 반영
4. **16MB 세그멘테이션은 과도하다** — 교집합(min) 방식이라 랜드마크 마스크가 이미 잘 잡으면 차이 미미. Codex 권장: 200~800KB 경량 교체.
5. **2-Stage 눈 정밀화가 효과적** — Perfect Corp의 228KB eye refiner 방식. 현재 V2의 heuristic 보정보다 robust.

### 1.3 우선순위 결정 근거

| 순위 | 개선 항목 | 사용자 체감 영향 | 구현 난이도 | 근거 |
|------|----------|----------------|-----------|------|
| **1** | Temporal Architecture | 🔴 매우 높음 (지터/깜빡임 제거) | 🟡 중간 | 양사 모두 다중 스무딩 적용. 가장 빠른 체감 개선 |
| **2** | Cascaded Eye Refiner | 🔴 높음 (렌즈 위치 정확도) | 🟡 중간 | Perfect Corp의 핵심 차별화. 위치 오류는 가장 눈에 띄는 결함 |
| **3** | GPU Lens Surface Rendering | 🟡 높음 (렌즈 사실감) | 🟡 중간~높음 | "2D 스티커"→"실제 컨택트 렌즈" 전환점 |
| **4** | 경량 Segmenter 교체 | 🟢 낮음 (뷰티 경계만) | 🟢 낮음 | 현재 16MB→200~800KB. 뷰티 품질이지 렌즈 품질 아님 |

---

## 2. P5 전체 아키텍처

### 2.1 현재 파이프라인 (As-Is)

```
카메라 프레임 (동기)
    ↓
[FrameProcessor] 포맷 변환
    ↓
[MediaPipeDetector] TFLite 3-stage 추론
    │  ├── Face Detection (128×128)
    │  ├── Face Landmark (192×192 or 256×256)
    │  └── Iris Landmark (64×64, V1) / 내장 (V2)
    ↓
RAW 좌표 그대로 출력 (스무딩 없음)
    ↓
[LensRenderer] 2D 사각형 텍스처 오버레이
    ↓
출력
```

**문제점 (C++ SDK 코어 경로):**
- C++ detector의 스무딩 비활성 → raw 좌표 출력
- 검출 실패 시 즉시 사라짐 → 깜빡임
- `FrameProcessor::process()`가 `detectSync()` 블로킹 대기 → 추론 지연이 렌더링 지연으로 직결
- C++ LensRenderer가 평면 2D 오버레이 → "붙은 느낌"
- V2 heuristic iris 보정 → 불안정

**참고 (Android 데모 경로):**
- `CameraGLRenderer.kt`에 이미 OneEuroFilter 26개+ / 눈꺼풀 클리핑 / 타원 마스크 / hold frames / sclera protection 구현됨
- `OverlayView.kt`에도 OneEuroFilter 6개 + lens persistence + fade 구현됨
- **이 데모 구현은 P5의 레퍼런스이자 이관 대상** — 이중 구현을 방지해야 함

### 2.2 목표 파이프라인 (To-Be)

```
카메라 프레임
    ↓
[Frame Controller] ← P5-W1
    ├── Latest-frame 큐 (drop-oldest)
    ├── 추론/렌더링 FPS 분리
    └── 과부하 시 프레임 드롭
    ↓
[MediaPipeDetector] TFLite 추론 (비동기)
    ↓
[Eye Refiner] ← P5-W2
    ├── 눈 ROI 크롭 → 경량 모델 (150~300KB)
    ├── 홍채 중심/반지름 정밀화
    └── 품질 점수 + 눈꺼풀 오클루전 큐 출력
    ↓
[Temporal Stability Stack] ← P5-W1
    ├── Layer 1: Confidence 이력현상 + Visibility fade
    ├── Layer 2: 아웃라이어 거부 + 눈깜빡임 감지
    └── Layer 3: OneEuroFilter (iris center/radius/pose 개별 튜닝)
    ↓
[GPU Lens Renderer] ← P5-W3
    ├── 타원/극좌표 홍채 메시 (평면 사각형 탈피)
    ├── 눈꺼풀 오클루전 (eye contour 랜드마크)
    ├── 각막 하이라이트 + 림발 다크닝
    └── 선택적: Normal Map + IBL
    ↓
출력 (안정적, 사실적)
```

### 2.3 작업 구조

```
P5-W1: Temporal Architecture (C++ TemporalStabilizer 일원화)
  ├── W1-01: TemporalStabilizer 설계 + OneEuroFilter 스무딩
  ├── W1-02: Confidence 이력현상 + Visibility fade-in/out
  ├── W1-03: 아웃라이어 거부 + 눈깜빡임 감지
  ├── W1-04: 비동기 결과 API 설계 + Frame Controller
  └── W1-05: 통합 테스트 + Android 데모 Kotlin 스무딩 제거

P5-W2: Cascaded Eye Refinement (2-Stage 눈 정밀화)
  ├── W2-01: Eye Refiner 모델 선정 (V2에서 iris_landmark 재로딩 포함)
  ├── W2-02: Eye Refiner 추론 파이프라인 통합
  ├── W2-03: 조건부 실행 정책 (confidence/iris_radius/outlier)
  └── W2-04: V2 heuristic 대비 정밀도 벤치마크

P5-W3: GPU Lens Surface Rendering (데모→SDK 코어 이관 + 확장)
  ├── W3-00: 데모 CameraGLRenderer 기능 분석 + C++ 포팅 범위 결정
  ├── W3-01: 데모 타원 마스크/눈꺼풀 클리핑 → C++ 포팅
  ├── W3-02: 데모 8종 블렌드 → C++ GPU 셰이더 이관
  ├── W3-03: 각막 하이라이트 + 림발 다크닝 (신규)
  └── W3-04: 선택적 Normal Map + IBL (신규)

P5-W4: Lightweight Segmenter (경량 세그멘터 교체) — 선택적
  ├── W4-01: 200~800KB face-skin parser 모델 선정
  ├── W4-02: ROI 내 실행 + 매 N프레임 정책
  └── W4-03: 16MB 모델 제거 / 옵션화
```

---

## 3. 경쟁사 대비 갭 해소 예상

### P5 완료 후 비교

| 항목 | 피팅몬스터 | Perfect Corp | IrisLensSDK (P5 후) |
|------|-----------|-------------|-------------------|
| **스무딩** | 3-layer | FaceAlignMotion | 3-layer (W1) |
| **눈 정밀화** | MediaPipe Iris | 2nd-step eye model | Eye Refiner (W2) |
| **렌즈 표면** | 3D FaceGeometry | PBR (Normal+IBL) | GPU 메시 + 오클루전 (W3) |
| **프레임 제어** | FlowLimiter | Triple Buffer | Frame Controller (W1) |
| **세그멘터** | 244KB (미사용) | ~413KB (추정) | 200~800KB (W4) |

### 남는 갭 (P5로 해소 불가, Phase 6+ 검토)

- **풀 3D Face Geometry 렌더링** — 고개 회전 시 원근 변형
- **포즈 추정 모델** — 별도 머리 회전 추정
- **자체 학습 모델** — MediaPipe 의존 탈피 (Eye-Only detector)
- **iOS/Flutter/Web 플랫폼** — 크로스 플랫폼 배포

---

## 4. 참고 문서

| 문서 | 위치 |
|------|------|
| 피팅몬스터 분석 레포트 | `docs/reports/ANALYSIS_fittingMonster_vs_IrisLensSDK.md` |
| Perfect Corp 분석 레포트 | `docs/reports/ANALYSIS_perfectCorp_SDK.md` |
| Android 데모 GPU 렌더러 | `android/demo-app/.../gpu/CameraGLRenderer.kt` (W3 이관 레퍼런스) |
| Android 데모 OverlayView | `android/demo-app/.../camera/OverlayView.kt` (W1 스무딩 레퍼런스) |
| 피팅몬스터 SDK 원본 | `docs/fittingMonster/` |
| Perfect Corp SDK 원본 | `docs/perfect/` |
| Codex 분석 원문 | P5-Preview 부록 참조 |

---

## 부록: Codex (GPT-5.4) 원문 요약

> "The fastest path to competitor-grade quality is better temporal architecture, a small cascaded eye refiner, and a shader-based lens surface model."

> "Stability is a system, not a single filter constant."

> "Full face geometry can wait. Full PBR can wait. Lid occlusion plus corneal specular cannot."

> "Replace the 16MB model with a small quantized face-skin/parts parser in roughly the 200KB to 800KB class. Run it on a face ROI, not the full frame, and not necessarily every frame."

> "Yes to a second-step eye model, no to making it an expensive always-on default before you have gating."
