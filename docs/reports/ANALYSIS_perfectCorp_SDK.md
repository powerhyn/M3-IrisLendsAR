# Perfect Corp SDK 분석 레포트

> **작성일**: 2026-03-30
> **목적**: Perfect Corp(퍼펙트) SDK의 구현 방식 역분석 및 IrisLensSDK/피팅몬스터와 비교

---

## 1. 개요

| 항목 | Perfect Corp SDK |
|------|-----------------|
| **제품명** | PerfectLib (YMK = YouCam Makeup 계열) |
| **주요 기능** | 풀페이스 메이크업 AR + 컬러렌즈 + 헤어/액세서리 |
| **추론 엔진** | MNN (Alibaba, FP16 양자화) |
| **렌더링 엔진** | Venus (자체 GPU 엔진, 24+ 블렌드 모드) |
| **플랫폼** | Android (Flutter PlatformView 연동) |
| **비즈니스 레이어** | SKU/상품 관리 시스템 내장 (VTO) |

---

## 2. 모델 구조

### 2.1 ML 모델 (MNN 기반)

| 모델 | 크기 | 용도 | 날짜 |
|------|------|------|------|
| `YMK_Venus_20210709_fp16_alignmodel.mnn` | 1.3MB | 얼굴 정렬 + 메이크업 좌표 매핑 | 2021.07 |
| `YMK_Davinci_20200512_fp16.mnn` | 885KB | 레거시 얼굴 모델 | 2020.05 |
| `second_step_eye_model_20220609_fp16.mnn` | 228KB | 눈/홍채 영역 정밀 보정 (2차) | 2022.06 |
| `second_step_mouth_model_20240814_fp16.mnn` | 104KB | 입 영역 정밀 보정 (2차) | 2024.08 |
| `faceart_20240812.model` | 413KB | 페이스 아트/스타일 | 2024.08 |
| `eyebrow_20210311.model` | 148KB | 눈썹 전용 검출 | 2021.03 |
| `pose_20210601.mtnet` | 118KB | 머리 포즈/회전 추정 | 2021.06 |

### 2.2 렌더링 에셋

| 파일 | 크기 | 용도 |
|------|------|------|
| `eye_ibl.hdr` | 1.8MB | Image-Based Lighting — 눈 표면 반사광 시뮬레이션 |
| `eye_normal.png` | 569KB (512×512) | Normal Map — 눈/홍채 3D 곡률 표현 |

### 2.3 핵심 포인트: 2-Stage 추론

```
1차: 얼굴 전체 → YMK_Venus (106+ 랜드마크)
2차: 눈 영역 크롭 → second_step_eye_model (홍채 정밀 보정)
     입 영역 크롭 → second_step_mouth_model (입술 정밀 보정)
```

**피팅몬스터와 차이**: MediaPipe 모델 대신 자체 학습 모델 사용 (MNN 프레임워크)

---

## 3. 파이프라인 아키텍처

```
카메라 프레임 (NV21, 1280×960)
    ↓
[YMK_Venus] 얼굴 정렬 + 106+ 랜드마크
    ↓
[second_step_eye_model] 눈/홍채 정밀 보정 ← ⭐ 2차 정밀화
[second_step_mouth_model] 입 정밀 보정
    ↓
[FaceAlignMotionSmoother] 시간적 스무딩 ← ⭐ 핵심 안정화
    ↓
[Venus GPU Engine] 메이크업 렌더링
    ├── Normal Mapping (eye_normal.png)
    ├── IBL 반사 (eye_ibl.hdr)
    ├── 24+ 블렌드 모드
    └── 멀티패스 합성
    ↓
출력 프레임 (RGBA)
```

### 스레딩 모델

```
카메라 스레드 → 프레임 캡처
    ↓ (Triple Buffering)
검출 스레드 → 얼굴/눈/입 추론 + 스무딩
    ↓
렌더링 스레드 → Venus GPU 셰이더 실행
    ↓
디스플레이
```

---

## 4. 렌더링 시스템

### 4.1 Venus 엔진 특징

- **PBR (Physically Based Rendering)**: IBL + Normal Map으로 물리 기반 렌더링
- **24+ 블렌드 모드**: 메이크업 종류별 최적 블렌딩
- **sRGB 색공간 보정**: 메이크업 색상 정확도 보장

### 4.2 눈(렌즈) 렌더링 방식

```
eye_normal.png → 홍채/안구 3D 곡률 정의
    +
eye_ibl.hdr → 환경 반사광 시뮬레이션
    ↓
Normal Mapping + IBL = 사실적인 3D 렌즈 효과
    ↓
카메라 프레임과 합성
```

**IrisLensSDK와 핵심 차이**: 퍼펙트는 **PBR 렌더링**으로 렌즈가 실제 눈 위에 올라간 것처럼 보임. IrisLensSDK는 2D 텍스처 오버레이.

### 4.3 메이크업 블렌드 모드 (이펙트별)

| 이펙트 | 블렌드 방식 |
|--------|------------|
| 립스틱 | Multiply (색상 깊이감) |
| 파운데이션 | Screen (피부톤 보정) |
| 블러셔 | Soft Light / Overlay |
| 아이라이너 | Multiply (선명한 라인) |
| 아이섀도 | Overlay (색상 + 질감) |
| 하이라이터 | Screen / Addition (광택) |
| 컨투어 | Multiply (음영) |

---

## 5. 지원 이펙트 (18종)

```
[눈]  Eyeliner, Eyelashes, Eyeshadow, Mascara, Eyebrow, EyeContact(컬러렌즈)
[얼굴] Foundation, Blush, SkinSmooth, Contour, Highlighter, Bronzer, Concealer
[입]  Lipstick, Lipliner
[기타] Eyewear, Eyewear3D, Hairdye, Earrings, Background
```

각 이펙트별 **강도 조절** 가능: `getIntensities()` → `[min, current, max]`

---

## 6. SKU/상품 관리 시스템 (VTO)

### 6.1 상품 계층 구조

```
Effect (메이크업 카테고리)
    └── Product (브랜드/상품)
         └── SKU (색상/제품 변형)
              ├── WearingStyle (적용 스타일: 연하게/보통/진하게)
              ├── Palette (색상 조합)
              └── Pattern (질감: 매트/글로시/메탈릭)
```

### 6.2 서버 동기화

```kotlin
SkuHandler.syncServer()   // 상품 카탈로그 다운로드
LookHandler.syncServer()  // 프리셋 룩 다운로드
```

---

## 7. Flutter 연동 패턴

```
Flutter Widget (makeupCam_view)
    ↕ MethodChannel("PerfectSDKMakeupCamView/{viewId}")
PlatformView (MakeupCamView.kt)
    ↕ Native Camera + Venus Engine
```

**주요 Method Call**:
- `startCamera` / `switchCamera` / `takePicture`
- `apply(product, sku, palette, pattern, wearingStyle)`
- `applyLook(lookGuid)`
- `getIntensities` / `setIntensities`
- `clear` / `clearAllEffects`
- `dispose`

---

## 8. 3사 SDK 비교

### 8.1 아키텍처 비교

| 항목 | Perfect Corp | 피팅몬스터 | IrisLensSDK |
|------|-------------|-----------|-------------|
| **ML 프레임워크** | MNN (FP16) | TFLite (via MediaPipe) | TFLite (직접 호출) |
| **추론 구조** | 2-Stage (전체→부분 정밀화) | 3-Stage (Detection→Landmark→Iris) | 2~3-Stage (V2/V1) |
| **렌더링 엔진** | Venus (자체 GPU, PBR) | MediaPipe Calculator | OpenCV + GLES 셰이더 |
| **스무딩** | FaceAlignMotionSmoother | LandmarksSmoothingCalc × 3 | OneEuroFilter (비활성) |
| **파이프라인 구조** | 자체 엔진 | MediaPipe Graph | 자체 C++ |
| **프레임워크 종속** | MNN + Venus | MediaPipe Runtime | TFLite만 |

### 8.2 렌즈/눈 렌더링 비교

| 항목 | Perfect Corp | 피팅몬스터 | IrisLensSDK |
|------|-------------|-----------|-------------|
| **렌더링 방식** | PBR (Normal Map + IBL) | 3D Face Geometry | 2D 텍스처 오버레이 |
| **반사광** | ✅ HDR 환경맵 반사 | ❌ | ❌ |
| **Normal Mapping** | ✅ 512×512 노말맵 | ❌ | ❌ |
| **3D 곡률** | ✅ 안구 곡면 시뮬레이션 | ✅ Face Mesh 기반 | ❌ (평면) |
| **블렌드 모드** | 24+ | 미확인 (1종 추정) | 8종 |
| **깊이 정보** | 포즈 모델로 추정 | eyes_dp_length_mm | 미사용 (z 무시) |

### 8.3 안정화 비교

| 항목 | Perfect Corp | 피팅몬스터 | IrisLensSDK |
|------|-------------|-----------|-------------|
| **랜드마크 스무딩** | FaceAlignMotionSmoother | × 2 Calculator | OneEuroFilter (OFF) |
| **신뢰도 스무딩** | 자체 구현 추정 | VisibilitySmoothingCalc | 없음 (0/1) |
| **프레임 제어** | Triple Buffering | FlowLimiter + Thinner | 없음 |
| **2차 정밀화** | ✅ 눈/입 전용 모델 | ❌ | ❌ |
| **포즈 보정** | ✅ pose_model | ❌ | ❌ |

### 8.4 비즈니스 레이어 비교

| 항목 | Perfect Corp | 피팅몬스터 | IrisLensSDK |
|------|-------------|-----------|-------------|
| **상품 관리** | ✅ 풀 SKU 시스템 | ❌ | ❌ |
| **서버 동기화** | ✅ 온라인 카탈로그 | ❌ | ❌ |
| **프리셋 룩** | ✅ Look 시스템 | ❌ | ❌ |
| **라이선스** | userId 기반 | ECDSA P-256 | 없음 |
| **다중 이펙트** | ✅ 18종 동시 적용 | 1종 선택 | 렌즈만 |

### 8.5 SDK 크기 비교

| 항목 | Perfect Corp | 피팅몬스터 | IrisLensSDK |
|------|-------------|-----------|-------------|
| **네이티브 라이브러리** | ~11MB | ~27MB | ~5MB (추정) |
| **ML 모델** | 4.5MB | 6.6MB | 3.9MB (V1) |
| **렌더링 에셋** | 2.4MB (IBL+Normal) | 0 | 0 |
| **합계** | **~18MB** | **~33MB** | **~9MB (추정)** |

---

## 9. Perfect Corp에서 배울 점

### 9.1 IrisLensSDK에 적용 가능한 기술

| 우선순위 | 기술 | 효과 | 난이도 |
|---------|------|------|--------|
| **1** | **2차 정밀화 (Second-Step Eye Model)** | 홍채 검출 정밀도 향상 | 🟡 커스텀 모델 필요 |
| **2** | **Normal Mapping for 렌즈** | 렌즈가 3D 곡면처럼 보임 | 🟡 셰이더 추가 |
| **3** | **IBL 반사** | 렌즈에 현실적 반사광 | 🟡 HDR 에셋 + 셰이더 |
| **4** | **Triple Buffering** | 추론/렌더링 분리로 FPS 향상 | 🟢 아키텍처 변경 |
| **5** | **포즈 추정 모델** | 고개 회전 시 렌즈 변형 보정 | 🔴 모델 학습 필요 |

### 9.2 적용하지 않아도 될 것

| 기술 | 이유 |
|------|------|
| MNN 프레임워크 전환 | TFLite로 충분, 전환 비용 대비 이점 적음 |
| SKU/상품 관리 시스템 | IrisLensSDK는 렌즈 피팅 SDK, 비즈니스 레이어는 앱 몫 |
| 18종 메이크업 이펙트 | 범위 초과, 렌즈에 집중 |
| Venus 엔진급 렌더러 | 과도한 투자, GLES 셰이더로 충분 |

---

## 10. 결론

Perfect Corp SDK는 **메이크업 AR의 엔터프라이즈 레퍼런스**입니다. 피팅몬스터가 MediaPipe 기반 "래핑" 접근이라면, 퍼펙트는 **자체 ML 모델 + 자체 렌더링 엔진**의 완전 독자 스택입니다.

IrisLensSDK 관점에서 가장 주목할 점은:

1. **2-Stage 추론**: 전체 얼굴 → 눈 영역 정밀화로 홍채 정확도를 높이는 방식
2. **PBR 렌즈 렌더링**: Normal Map + IBL로 렌즈가 실제 눈 위에 있는 것처럼 보이게 하는 기술
3. **FaceAlignMotionSmoother**: 시간적 안정성의 중요성 (지터 제거)

이 세 가지는 IrisLensSDK의 렌즈 품질을 한 단계 올릴 수 있는 핵심 기술입니다.

---

## 부록: 분석 대상 파일

### Kotlin 래퍼 (6개)
- `docs/perfect/perfectlibwrapper/MakeupCamView.kt`
- `docs/perfect/perfectlibwrapper/MakeupCamViewFactory.kt`
- `docs/perfect/perfectlibwrapper/MakeupCamViewPlugin.kt`
- `docs/perfect/perfectlibwrapper/PerfectLibHandler.kt`
- `docs/perfect/perfectlibwrapper/PerfectLookHandler.kt`
- `docs/perfect/perfectlibwrapper/PerfectSkuHandler.kt`

### ML 모델 (7개, 총 3.2MB)
- `docs/perfect/model/*.mnn` (4개, MNN FP16)
- `docs/perfect/model/*.model` (2개, Proprietary)
- `docs/perfect/model/*.mtnet` (1개, 포즈)

### 렌더링 에셋 (2개, 총 2.4MB)
- `docs/perfect/model/eye_ibl.hdr` (IBL 환경맵)
- `docs/perfect/model/eye_normal.png` (노말맵)
