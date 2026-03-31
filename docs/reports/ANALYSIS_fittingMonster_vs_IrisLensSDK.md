# 피팅몬스터 SDK vs IrisLensSDK 비교 분석 레포트

> **작성일**: 2026-03-30
> **목적**: 피팅몬스터(InterVision210) SDK의 구현 방식을 역분석하고, IrisLensSDK와의 아키텍처·기능·성능 차이를 비교

---

## 1. 개요

| 항목 | 피팅몬스터 (InterVision210) | IrisLensSDK |
|------|---------------------------|-------------|
| **버전** | v2.1.0 (빌드 1640) | Phase 1 MVP (개발 중) |
| **패키지명** | `com.google.mediapipe.apps.aar_irisgeo_210` | `com.irislenssdk` |
| **기반 기술** | MediaPipe Graph Framework 직접 사용 | TFLite 직접 추론 + 자체 C++ 파이프라인 |
| **플랫폼** | iOS + Android (출시됨) | Android (구현 중), iOS/Flutter/Web (계획) |
| **라이선스** | ECDSA P-256 서명 검증 | 없음 (오픈 SDK) |

---

## 2. 아키텍처 비교

### 2.1 파이프라인 구조

#### 피팅몬스터: MediaPipe Graph 기반
```
카메라 입력
    ↓
[FlowLimiterCalculator] ← 프레임 속도 제어
    ↓
[FaceLandmarkFrontGpu] ← 얼굴 검출 + 468점 랜드마크 (GPU)
    ↓
[IrisLandmarkLeftAndRightGpu] ← 좌/우 홍채 분리 검출 (GPU)
    ↓
[LandmarksSmoothingCalculator] ← 시간적 스무딩
    ↓
[FaceEffectGpuTensor210] ← 커스텀 이펙트 텐서 처리
    ↓
[FaceGeometryEffectRenderer] ← 3D 지오메트리 렌더링
    ↓
[FaceBeauty] → [FaceBeautyChange] ← 뷰티 필터 (선택적)
    ↓
출력 프레임
```

- **특징**: MediaPipe의 Calculator Graph 프레임워크를 그대로 활용
- **그래프 파일**: `.binarypb` 형식으로 사전 컴파일된 그래프 3개
- **커스텀 Calculator**: `FaceEffectGpuTensor210`, `FaceCustom210`, `FaceBeauty` 등 자체 구현

#### IrisLensSDK: 자체 C++ 파이프라인
```
카메라 입력
    ↓
[FrameProcessor] ← 포맷 변환/회전 보정
    ↓
[MediaPipeDetector] ← 3단계 TFLite 직접 추론
    │  ├── Stage 1: face_detection_short_range (128x128)
    │  ├── Stage 2: face_landmark / face_landmark_v2 (192x192 or 256x256)
    │  └── Stage 3: iris_landmark (64x64, V1 전용)
    ↓
[OneEuroFilter] ← 적응형 저역 통과 필터 (지터 방지)
    ↓
[BeautyProcessor] ← DI 기반 뷰티 파이프라인
    │  ├── CPU: FastGuidedFilter / Bilateral
    │  └── GPU: FreqSep 6-subpass 셰이더
    ↓
[LensRenderer] ← 알파 블렌딩 렌즈 오버레이 (8종 블렌드 모드)
    ↓
출력 프레임
```

- **특징**: MediaPipe 프레임워크 없이 TFLite 모델만 직접 호출
- **장점**: 의존성 최소화, 파이프라인 완전 제어, 커스터마이징 자유도 높음
- **단점**: 모든 전/후처리를 직접 구현해야 함

### 2.2 핵심 아키텍처 차이

| 관점 | 피팅몬스터 | IrisLensSDK |
|------|-----------|-------------|
| **프레임워크 의존도** | MediaPipe Graph Runtime 전체 포함 | TFLite Runtime만 포함 |
| **그래프 실행** | `binarypb` 그래프를 MediaPipe 런타임이 실행 | C++ 코드에서 순차적 TFLite 호출 |
| **GPU 추론** | MediaPipe 내장 GPU Calculator | XNNPACK(CPU) + GPU Delegate(선택적) |
| **랜드마크 스무딩** | MediaPipe `LandmarksSmoothingCalculator` | 자체 구현 `OneEuroFilter` |
| **설계 패턴** | Graph-based (데이터 플로우) | Strategy + Pimpl + DI + Factory |
| **확장성** | Calculator 추가로 확장 | 인터페이스 구현체 교체로 확장 |

---

## 3. 모델 비교

### 3.1 공통 모델

| 모델 | 피팅몬스터 | IrisLensSDK | 비고 |
|------|-----------|-------------|------|
| `face_detection_short_range.tflite` | ✅ 224KB | ✅ 224KB | 동일 (BlazeFace) |
| `face_landmark.tflite` | ✅ 1.2MB | ✅ 1.2MB | 동일 (468점) |
| `face_landmark_with_attention.tflite` | ✅ 2.4MB | — | 피팅몬스터만 사용 |
| `iris_landmark.tflite` | ✅ 2.5MB | ✅ 2.5MB | 동일 (71점) |
| `selfie_segmentation.tflite` | ✅ 244KB | — | 피팅몬스터만 포함 |

### 3.2 IrisLensSDK 추가 모델

| 모델 | 크기 | 용도 |
|------|------|------|
| `face_landmark_v2.tflite` | 2.4MB | 478점 (홍채 10점 내장, iris_landmark 불필요) |
| `selfie_multiclass_256x256.tflite` | 16MB | 6-class 세그멘테이션 (Phase D 실험) |

### 3.3 모델 전략 분석

**피팅몬스터**:
- `face_landmark_with_attention` 사용 → Attention 메커니즘으로 정밀도 향상
- `selfie_segmentation` (244KB) → 경량 2-class 분할 (인물/배경)
- 모든 모델이 MediaPipe 원본 그대로

**IrisLensSDK**:
- V2 모델 지원 → 478점에 홍채 내장, 추론 횟수 감소 (3→2단계)
- `selfie_multiclass_256x256` (16MB) → 6-class 세그멘테이션 (face-skin 분리)
- V1/V2 자동 전환 지원

---

## 4. 뷰티 필터 비교

### 4.1 피팅몬스터 뷰티 시스템

**파라미터 4개 (단순)**:
```
BT_radius   (0~255)  - 블러 반경 (기본: 10)
BT_epsilon  (0~1.0)  - 스무딩 계수 (기본: 0.05)
BT_down     (1~4)    - 다운샘플링 레벨 (기본: 2)
BT_strength (0~1.0)  - 효과 강도 (기본: 0.7)
```

**구현**:
- `FaceBeauty` Calculator: Bilateral Filter 기반 전처리
- `FaceBeautyChange` Calculator: 후처리 보정
- `geo_gpu_po_210.binarypb` 그래프에서만 활성화
- 전체 프레임에 적용 (ROI 분리 없음으로 추정)

### 4.2 IrisLensSDK 뷰티 시스템

**파라미터 20개+ (정밀)**:
```
[피부 효과]
  smoothing, brightness, softFocus, whitening, colorBalance
  wrinkleRemove, skinQuality, smoothIntensity, poreReduction

[얼굴 형태]
  slimFace, enlargeEyes, thinChin

[포스트 프로세싱 - Vivid]
  vividIntensity, vividSaturation, vividBrightness, vividWarmth

[처리 옵션]
  useGpu, roiOnly, protectEyes, protectLips, protectNose, downscaleFactor
```

**구현**:
- **CPU 경로**: FastGuidedFilter (He et al., O(1)) + Bilateral Filter
- **GPU 경로**: Frequency Separation 6-subpass 셰이더 파이프라인
  1. Gaussian Blur H (sRGB→Linear)
  2. Gaussian Blur V → Low Frequency
  3. High Frequency 추출 + Composite (edge/chroma/tone 보존)
  4. Luminance Sharpen (피부 텍스처 복원)
  5. Masking (피부 마스크 적용)
  6. Combined Color (brightness + colorBalance + whitening + 3D LUT)
- **ROI 관리**: 478점 Face Mesh 기반 영역 분리
  - 피부 마스크: 36개 랜드마크 fillPoly
  - 눈/눈썹/입술/코 보호 마스크: 개별 랜드마크 기반
  - 합성: `skin × (1-eye) × (1-eyebrow) × (1-lip) × (1-nose)`
  - 페더링: Gaussian blur 경계 처리
- **DeviceTier 분기**: HIGH/MID/LOW 디바이스별 최적 파이프라인 선택
- **얼굴 변형**: GridMesh + RBF 보간 (슬림페이스, V라인, 눈 확대)

### 4.3 뷰티 비교 요약

| 항목 | 피팅몬스터 | IrisLensSDK |
|------|-----------|-------------|
| **조절 파라미터** | 4개 | 20개+ |
| **스무딩 알고리즘** | Bilateral Filter | FastGuidedFilter + FreqSep |
| **ROI 분리** | 미확인 (전체 프레임 추정) | ✅ 478점 기반 정밀 ROI |
| **눈/입술 보호** | 미확인 | ✅ 개별 마스크 |
| **얼굴 변형** | 없음 | ✅ 슬림페이스/V라인/눈확대 |
| **GPU 파이프라인** | MediaPipe Calculator 내 처리 | GLES 3.1 셰이더 (6-subpass) |
| **디바이스 적응** | 없음 (단일 경로) | ✅ HIGH/MID/LOW 분기 |
| **품질 메트릭** | 없음 | ✅ Laplacian/SSIM/Halo |
| **색상 보정** | 없음 | ✅ LAB 화이트닝, Vivid |

---

## 5. 렌더링 비교

### 5.1 피팅몬스터

- **이펙트 시스템**: PNG 텍스처 오버레이 (effect type + PNG 파일)
- **투명도**: alpha (0~255)
- **렌더러**: `FaceGeometryEffectRenderer` — 3D Face Geometry 기반
- **렌즈 피팅**: `FaceCustomLensFit210` — 커스텀 렌즈 맞춤
- **홍채 깊이**: `eyes_dp_length_mm` 출력 (물리적 깊이 추정)
- **블렌드 모드**: 미확인 (단일 방식 추정)

### 5.2 IrisLensSDK

- **이펙트 시스템**: 텍스처 파일/메모리 로드
- **투명도**: alpha (0.0~1.0)
- **렌더러**: `LensRenderer` — Pimpl 패턴, OpenCV 기반
- **블렌드 모드 8종**: Normal, Multiply, Screen, Overlay, LuminanceTint, LuminanceTintLinear, SoftLight, ColorReplace
- **양쪽/단눈**: 개별 렌더링 지원
- **ROI 최적화**: 홍채 주변 영역만 처리

### 5.3 렌더링 비교 요약

| 항목 | 피팅몬스터 | IrisLensSDK |
|------|-----------|-------------|
| **3D Geometry** | ✅ Face Mesh 3D 렌더링 | ❌ 2D 투영 기반 |
| **블렌드 모드** | 1종 (추정) | 8종 |
| **홍채 깊이 추정** | ✅ mm 단위 출력 | ❌ (없음) |
| **렌즈 피팅** | 전용 Calculator | 홍채 좌표 + 스케일링 |
| **멀티페이스** | ✅ num_faces 설정 | ✅ 지원 |

---

## 6. 플랫폼 지원 비교

### 6.1 Android

| 항목 | 피팅몬스터 | IrisLensSDK |
|------|-----------|-------------|
| **Min SDK** | 21 | 24 |
| **Target SDK** | 34 | 34 |
| **ABI** | arm64-v8a + armeabi-v7a | arm64-v8a |
| **JNI 라이브러리** | lib210_jni.so (9.8MB) + libopencv_java4.so (17MB) | libiris_jni.so + 자체 OpenCV 링크 |
| **카메라 API** | Camera1 + Camera2 (듀얼) | CameraX |
| **Java 클래스** | 339개 (MediaPipe 포함) | ~10개 (경량) |
| **총 크기** | ~44MB | 추정 ~25MB |

### 6.2 iOS

| 항목 | 피팅몬스터 | IrisLensSDK |
|------|-----------|-------------|
| **Min iOS** | 12.0 | — (미구현) |
| **프레임워크** | InterVision210.framework (63.5MB) | 계획: XCFramework |
| **렌더링** | Metal + OpenGL ES | 계획: Metal |
| **API** | Objective-C | 계획: Objective-C++ |

### 6.3 멀티플랫폼

| 플랫폼 | 피팅몬스터 | IrisLensSDK |
|--------|-----------|-------------|
| Android | ✅ 출시 | 🔄 구현 중 |
| iOS | ✅ 출시 | ⏳ Phase 2 |
| Flutter | ❌ | ⏳ Phase 2 |
| Web | ❌ | ⏳ Phase 3 |
| Desktop | ❌ | ✅ 데모 완료 |

---

## 7. API 설계 비교

### 7.1 피팅몬스터 — View 기반 (블랙박스)

```java
// Android 초기화 — Activity + FrameLayout 전달, 내부에서 카메라+렌더링 전부 관리
InterVisionEffect210 effect = new InterVisionEffect210(
    activity, frameLayout, license, facking, graphType,
    btRadius, btEpsilon, btDown, btStrength);

// 이펙트 설정
effect.setEffect(effectType, pngFilename);
effect.setAlpha(128);

// 라이프사이클
effect.onCusResume();
effect.onCusPause();
effect.onClose();
```

```objc
// iOS 초기화
InterVisionEffect210 *effect = [[InterVisionEffect210 alloc]
    initName:license f_View:view f_Facking:1 f_sGType:@"geo_gpu_210"];

// 프레임 처리 (수동 전달)
[effect processVideoFrame:pixelBuffer timestamp:timestamp];
```

**특징**:
- 카메라, 검출, 렌더링이 SDK 내부에서 일괄 처리
- 개발자는 View만 전달하고 이펙트만 제어
- 내부 구현 접근 불가 (블랙박스)

### 7.2 IrisLensSDK — 모듈식 C API

```c
// 초기화
iris_sdk_init(model_dir);

// 검출 (개별 호출 가능)
IrisResultC result;
iris_sdk_detect(image_data, width, height, format, &result);

// 렌더링 (개별 호출 가능)
iris_sdk_render_lens(image_data, width, height, &result, &lens_config);

// 뷰티 (개별 호출 가능)
iris_sdk_beauty_apply_v2(image_data, width, height, &beauty_config);

// 해제
iris_sdk_destroy();
```

**특징**:
- 검출, 렌더링, 뷰티를 독립적으로 호출 가능
- C API로 모든 바인딩 레이어에서 동일한 인터페이스
- 카메라 관리는 앱 책임 (SDK는 프레임 처리만)
- 세밀한 제어 가능 (검출만, 렌더링만 등)

### 7.3 API 설계 비교 요약

| 항목 | 피팅몬스터 | IrisLensSDK |
|------|-----------|-------------|
| **API 스타일** | High-level (View 전달) | Low-level (프레임 전달) |
| **카메라 관리** | SDK 내부 | 앱 책임 |
| **모듈 분리** | 불가 (일체형) | ✅ 검출/렌더링/뷰티 독립 |
| **크로스 플랫폼 API** | iOS/Android 별도 | C API 단일 인터페이스 |
| **커스터마이징** | 제한적 (파라미터만) | 높음 (파이프라인 제어) |
| **학습 곡선** | 낮음 (간단) | 중간 (유연하지만 복잡) |

---

## 8. 성능 비교

| 지표 | 피팅몬스터 (추정) | IrisLensSDK (측정) |
|------|-------------------|-------------------|
| **FPS (Android)** | 30+ (GPU Calculator) | 15-20 (CPU), 30+ 목표 |
| **추론 단계** | 3단계 (모두 GPU) | 2단계 (V2) / 3단계 (V1), CPU |
| **GPU 활용** | ✅ 전면 (MediaPipe GPU) | 🔄 뷰티만 GPU, 추론은 CPU |
| **스무딩** | MediaPipe Calculator | OneEuroFilter (적응형) |
| **메모리 관리** | MediaPipe 내부 관리 | BufferPool + TexturePool |
| **프레임 제어** | FlowLimiter + PacketThinner | 없음 (호출자 책임) |

---

## 9. 강점/약점 분석

### 9.1 피팅몬스터 강점 (IrisLensSDK에 적용 가능)

| # | 강점 | 적용 가능성 |
|---|------|------------|
| 1 | **FlowLimiter 패턴** — 입력 프레임 속도 제어로 과부하 방지 | ✅ FrameProcessor에 프레임 스로틀링 추가 가능 |
| 2 | **듀얼 그래프 선택** — 용도별 최적화된 파이프라인 선택 | ✅ 이미 Strategy 패턴으로 지원 가능 |
| 3 | **홍채 깊이 추정** — `eyes_dp_length_mm` 물리적 거리 출력 | ⚠️ iris_landmark 모델 출력에서 추출 가능, 구현 검토 필요 |
| 4 | **좌/우 홍채 분리 처리** — 각 눈을 독립적으로 검출 후 병합 | ✅ 이미 V1에서 유사하게 구현 |
| 5 | **3D Face Geometry 렌더링** — 실제 3D 메시 기반 렌즈 피팅 | ⚠️ 현재 2D 투영 방식, Phase 2에서 검토 |
| 6 | **Attention 모델 옵션** — 정밀도가 필요할 때 Attention 모델 선택 | ✅ V2 모델로 이미 대응 |

### 9.2 IrisLensSDK 강점 (피팅몬스터 대비)

| # | 강점 | 상세 |
|---|------|------|
| 1 | **MediaPipe 프레임워크 비의존** | TFLite만 사용하여 라이브러리 크기 절감, 업데이트 자유도 높음 |
| 2 | **V2 모델 지원** | 478점 + 홍채 내장으로 추론 1단계 절감 |
| 3 | **정밀한 뷰티 시스템** | 20+ 파라미터, ROI 분리, FreqSep GPU 파이프라인 |
| 4 | **얼굴 변형 기능** | GridMesh + RBF 보간 (슬림페이스, V라인, 눈확대) |
| 5 | **8종 블렌드 모드** | 다양한 렌즈 표현 가능 |
| 6 | **품질 자동화** | Laplacian/SSIM/Halo 메트릭, ReleaseGate, A/B 테스트 |
| 7 | **디바이스 적응** | HIGH/MID/LOW 자동 분기 |
| 8 | **모듈식 C API** | 크로스플랫폼 단일 인터페이스, 유연한 파이프라인 제어 |
| 9 | **눈/입술/코 보호 마스크** | 뷰티 적용 시 자연스러움 보장 |
| 10 | **Temporal Stability 측정** | 프레임 간 변동 계수(CV) 자동 측정 |

### 9.3 피팅몬스터 약점

| # | 약점 | 영향 |
|---|------|------|
| 1 | **MediaPipe 종속** | 프레임워크 업데이트에 SDK 전체 영향 |
| 2 | **블랙박스 API** | 내부 파이프라인 커스터마이징 불가 |
| 3 | **뷰티 파라미터 제한** | 4개만으로는 세밀한 조정 불가 |
| 4 | **얼굴 변형 없음** | 슬림페이스 등 미지원 |
| 5 | **OpenCV 전체 동봉** | `libopencv_java4.so` (17MB) — 불필요한 모듈 포함 |
| 6 | **라이선스 강제** | ECDSA 검증 실패 시 SDK 전체 동작 불가 |
| 7 | **Flutter/Web 미지원** | iOS/Android만 대응 |

---

## 10. 크기 비교

### 10.1 라이브러리 크기 (Android arm64-v8a)

| 컴포넌트 | 피팅몬스터 | IrisLensSDK (예상) |
|----------|-----------|-------------------|
| JNI 네이티브 | 9.8 MB | ~5 MB (추정) |
| OpenCV | 17 MB (전체) | ~3 MB (core+imgproc만) |
| TFLite 모델 | 6.6 MB (5개) | 3.9 MB (3개, V2 기준) |
| 그래프/설정 | 8 KB (.binarypb) | 0 (코드 내장) |
| **합계** | **~33 MB** | **~12 MB (추정)** |

### 10.2 모델 크기

| 구성 | 피팅몬스터 | IrisLensSDK V1 | IrisLensSDK V2 |
|------|-----------|---------------|----------------|
| Face Detection | 224 KB | 224 KB | 224 KB |
| Face Landmark | 1.2 + 2.4 MB | 1.2 MB | 2.4 MB |
| Iris Landmark | 2.5 MB | 2.5 MB | — (V2에 내장) |
| Segmentation | 244 KB | 16 MB (선택) | 16 MB (선택) |
| **합계** | **6.6 MB** | **3.9 MB** / **19.9 MB** | **2.6 MB** / **18.6 MB** |

---

## 11. 기술 스택 비교 요약

```
┌─────────────────────────────────────────────────────────────────┐
│                    피팅몬스터 (InterVision210)                     │
├─────────────────────────────────────────────────────────────────┤
│  App Layer    │  Java/Kotlin (Android) │ Obj-C (iOS)           │
│  Binding      │  JNI (339 classes)     │ Native Framework      │
│  Pipeline     │  ▓▓▓ MediaPipe Graph Runtime ▓▓▓               │
│  ML           │  TFLite (via MediaPipe Calculator)             │
│  GPU          │  MediaPipe GPU Calculator + OpenGL ES          │
│  Beauty       │  Bilateral Filter (FaceBeauty Calculator)      │
│  Image        │  OpenCV 4.x (전체 동봉)                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         IrisLensSDK                              │
├─────────────────────────────────────────────────────────────────┤
│  App Layer    │  Java/Kotlin │ (iOS/Flutter/Web 계획)           │
│  Binding      │  JNI (~10 classes) │ C API (extern "C")        │
│  Pipeline     │  ▓▓▓ 자체 C++ 파이프라인 ▓▓▓                    │
│  ML           │  TFLite 직접 호출 (XNNPACK/GPU Delegate)        │
│  GPU          │  GLES 3.1 셰이더 (FreqSep 6-subpass)           │
│  Beauty       │  FastGuidedFilter + FreqSep + ROI + FaceWarp   │
│  Image        │  OpenCV 4.x (core+imgproc만)                   │
│  QA           │  Laplacian/SSIM/Halo + ReleaseGate             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 12. 결론 및 권장사항

### 12.1 핵심 결론

1. **아키텍처 방향은 IrisLensSDK가 더 유연**: MediaPipe 프레임워크 비의존으로 경량화 + 완전한 파이프라인 제어 확보. 피팅몬스터는 MediaPipe에 종속되어 커스터마이징이 제한적.

2. **뷰티 시스템은 IrisLensSDK가 월등히 앞섬**: 20+ 파라미터, ROI 기반 보호 마스크, FreqSep GPU 파이프라인, 품질 자동화 시스템. 피팅몬스터는 4파라미터 Bilateral Filter로 기본적 수준.

3. **실시간 성능은 피팅몬스터가 현재 앞섬**: MediaPipe의 GPU Calculator가 추론과 렌더링을 모두 GPU에서 처리하는 반면, IrisLensSDK는 추론이 아직 CPU 기반 (15-20fps).

4. **SDK 크기는 IrisLensSDK가 유리**: ~12MB 예상 vs 피팅몬스터 ~33MB. OpenCV 경량 링크 + TFLite 직접 사용 효과.

### 12.2 피팅몬스터에서 배울 점

| 우선순위 | 항목 | 적용 방안 |
|---------|------|----------|
| **높음** | FlowLimiter 패턴 | FrameProcessor에 프레임 스로틀링 + 드롭 정책 추가 |
| **높음** | GPU 추론 파이프라인 | GPU Delegate 안정화 (Phase 1-W6 진행 중) |
| **중간** | 홍채 깊이 추정 | iris_landmark 출력에서 `eyes_dp_length_mm` 추출 로직 추가 |
| **중간** | 3D Face Geometry | Phase 2에서 3D 메시 기반 렌즈 피팅 검토 |
| **낮음** | Attention 모델 옵션 | V2 모델로 이미 커버, 별도 Attention 모델 불필요 |

### 12.3 IrisLensSDK가 유지해야 할 차별점

1. **모듈식 C API** — 크로스플랫폼 단일 인터페이스의 전략적 가치
2. **정밀 뷰티 시스템** — ROI + FreqSep + FaceWarp는 경쟁력 있는 차별 요소
3. **품질 자동화** — ReleaseGate + QualityMetrics는 프로덕션 안정성의 핵심
4. **디바이스 적응** — HIGH/MID/LOW 자동 분기로 다양한 디바이스 대응
5. **경량 아키텍처** — MediaPipe 비의존으로 크기 절감 + 업데이트 자유도

---

## 부록: 파일 참조

### 피팅몬스터 SDK 분석 대상
- `docs/fittingMonster/InterVision210.framework/Headers/*.h`
- `docs/fittingMonster/aar_irisgeo_210/classes.jar` (339 클래스)
- `docs/fittingMonster/aar_irisgeo_210/jni/arm64-v8a/lib210_jni.so` (757 JNI 심볼)
- `docs/fittingMonster/*/assets/*.tflite` (5 모델)
- `docs/fittingMonster/*/assets/*.binarypb` (3 그래프)

### IrisLensSDK 분석 대상
- `cpp/include/iris_sdk/` (37 헤더)
- `cpp/src/` (30 구현 파일)
- `cpp/tests/` (28 테스트)
- `android/iris-sdk/src/main/cpp/iris_jni.cpp` (68KB)
- `android/iris-sdk/src/main/java/com/irislenssdk/` (Java/Kotlin API)
