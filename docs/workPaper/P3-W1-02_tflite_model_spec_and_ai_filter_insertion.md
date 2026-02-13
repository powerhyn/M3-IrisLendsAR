# P3-W1-02: TFLite 모델 3종 입출력 스펙 및 AI 필터 삽입 포인트 설계

**상태**: ✅ 완료
**작성일**: 2026-02-09
**담당**: ml-vision

---

## 1. TFLite 모델 입출력 스펙

### 1.1 Face Detection (BlazeFace Short Range)

| 항목 | 값 |
|------|------|
| **파일** | `face_detection_short_range.tflite` |
| **크기** | 229,746 bytes (224.4 KB) |
| **용도** | 얼굴 검출 및 바운딩 박스 추출 |

#### 입력 텐서

| 속성 | 값 |
|------|------|
| 형태 | `[1, 128, 128, 3]` |
| 데이터 타입 | `float32` |
| 채널 | RGB (HWC 포맷) |
| 정규화 | `[0, 1]` (pixel / 255.0) |
| 전처리 | Letterbox (aspect ratio 보존, 검은색 패딩) |

#### 출력 텐서

| 출력 | 인덱스 | 형태 | 설명 |
|------|--------|------|------|
| Output 0 (regressors) | 175 | `[1, 896, 16]` | 바운딩 박스 오프셋 + 6 키포인트 |
| Output 1 (classificators) | 174 | `[1, 896, 1]` | 신뢰도 점수 (raw logits, sigmoid 적용 필요) |

#### 앵커 구조

```
Feature Map Level 1: 16x16 grid, 2 anchors/cell, stride=8  → 512개
Feature Map Level 2:  8x8  grid, 6 anchors/cell, stride=16 → 384개
총 앵커: 896개
```

#### 출력 디코딩

- **Output 0** 각 앵커별 16개 값:
  - `[0]` yc_offset: Y 중심 오프셋 (입력 크기 기준)
  - `[1]` xc_offset: X 중심 오프셋
  - `[2]` h_scale: 높이 스케일
  - `[3]` w_scale: 너비 스케일
  - `[4-15]` 6개 키포인트 (각 x,y 쌍)
- **좌표 변환**: `actual = anchor_center + offset / 128.0`
- **Letterbox 역변환** 필요 (패딩 보정)

---

### 1.2 Face Landmark V1

| 항목 | 값 |
|------|------|
| **파일** | `face_landmark.tflite` |
| **크기** | 1,242,398 bytes (1.2 MB) |
| **용도** | 얼굴 468개 랜드마크 검출 |

#### 입력 텐서

| 속성 | 값 |
|------|------|
| 형태 | `[1, 192, 192, 3]` |
| 데이터 타입 | `float32` |
| 채널 | RGB (HWC 포맷) |
| 정규화 | `[0, 1]` (pixel / 255.0) |
| 전처리 | Face Detection 바운딩 박스 기반 크롭 후 리사이즈 |

#### 출력 텐서

| 출력 | 인덱스 | 형태 | 설명 |
|------|--------|------|------|
| Output 0 | 213 | `[1, 1, 1, 1404]` (=468*3) | 468개 랜드마크 (x, y, z) |
| Output 1 | 210 | `[1, 1, 1, 1]` | Face presence score |

#### 좌표 해석

- 출력은 픽셀 좌표(0-192) 또는 정규화 좌표(0-1) (모델에 따라 다름)
- 자동 감지 로직으로 판별 후 정규화 적용
- 크롭 영역 → 전체 이미지 좌표로 역변환 필요

---

### 1.3 Face Landmark V2 (현재 기본)

| 항목 | 값 |
|------|------|
| **파일** | `face_landmark_v2.tflite` |
| **크기** | 2,553,590 bytes (2.4 MB) |
| **용도** | 얼굴 478개 랜드마크 + 홍채 검출 (통합) |

#### 입력 텐서

| 속성 | 값 |
|------|------|
| 형태 | `[1, 256, 256, 3]` |
| 데이터 타입 | `float32` |
| 채널 | RGB (HWC 포맷) |
| 정규화 | `[0, 1]` (pixel / 255.0) |
| 전처리 | 중앙 정사각형 크롭 (landscape) 또는 직접 리사이즈 |

#### 출력 텐서

| 출력 | 인덱스 | 형태 | 설명 |
|------|--------|------|------|
| Output 0 | 473 | `[1, 1, 1, 1434]` (=478*3) | 478개 랜드마크 (x, y, z) |
| Output 1 | 472 | `[1, 1, 1, 1]` | Face presence score |
| Output 2 | 475 | `[1, 1]` | 추가 신뢰도 |

#### 랜드마크 구조

```
인덱스   0-467: 얼굴 랜드마크 (Face Mesh 468점)
인덱스 468-472: 왼쪽 홍채 (중심 + 4 경계점)
인덱스 473-477: 오른쪽 홍채 (중심 + 4 경계점)
```

#### 홍채 경계점 순서

- `468/473`: 홍채 중심
- `469/474`: 상단
- `470/475`: 하단
- `471/476`: 좌측
- `472/477`: 우측

---

### 1.4 Iris Landmark (V1 전용)

| 항목 | 값 |
|------|------|
| **파일** | `iris_landmark.tflite` |
| **크기** | 2,640,568 bytes (2.5 MB) |
| **용도** | 눈 영역 내 홍채 정밀 검출 (V1 파이프라인에서만 사용) |

#### 입력 텐서

| 속성 | 값 |
|------|------|
| 형태 | `[1, 64, 64, 3]` |
| 데이터 타입 | `float32` |
| 채널 | RGB (HWC 포맷) |
| 정규화 | `[0, 1]` (pixel / 255.0) |
| 전처리 | 눈 영역 ROI 크롭 (IRIS_ROI_SCALE=2.3x 확대) |

#### 출력 텐서

| 출력 | 인덱스 | 형태 | 설명 |
|------|--------|------|------|
| Output 0 | 384 | `[1, 213]` (=71*3) | 71개 랜드마크 (눈 윤곽 + 홍채) |
| Output 1 | 385 | `[1, 15]` (=5*3) | 홍채 전용 5개 랜드마크 (중심 + 경계 4점) |

#### 출력 랜드마크 구조

```
인덱스  0-67: 눈 윤곽 (Eye Contour) - 68개
인덱스 68-72: 홍채 (Iris) - 5개
  - 68: 홍채 중심
  - 69: 상단 경계
  - 70: 하단 경계
  - 71: 좌측 경계
  - 72: 우측 경계
```

- 출력은 64x64 픽셀 좌표 → 정규화 변환 필요 (`/ 64.0`)
- 눈 ROI → 전체 이미지 좌표 역변환 필요

---

## 2. 추론 파이프라인 흐름도

```
카메라 프레임 (RGBA/NV21)
    │
    ▼
┌─────────────────────────────┐
│ 1. preprocessImage          │
│    RGB 변환 → Letterbox     │
│    128x128 float [0,1]      │
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│ 2. Face Detection           │
│    BlazeFace Short Range    │
│    → 바운딩 박스 + 신뢰도    │
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│ 3. Face Landmark (V2)       │
│    중앙 크롭 → 256x256      │
│    → 478개 랜드마크 (홍채 포함) │
└────────────┬────────────────┘
             ▼
   ┌─────────┴─────────┐
   │ V2: 직접 추출      │ V1: Iris Landmark
   │ idx 468-477       │ 64x64 입력
   └─────────┬─────────┘
             ▼
┌─────────────────────────────┐
│ 4. GPU Beauty Pipeline      │  ← AI 필터 삽입 지점
│    Smoothing → Color →      │
│    SoftFocus                │
└─────────────────────────────┘
```

---

## 3. GPU 파이프라인 AI 필터 삽입 포인트 설계

### 3.1 현재 GPU Beauty 파이프라인

```
입력 텍스처 (카메라 프레임)
    │
    ▼ [Pass 1]
    Smoothing (Bilateral Filter) ─── Ping → Pong
    │
    ▼ [Pass 2]
    Combined Color (Brightness + ColorBalance + Whitening) ─── Pong → Ping
    │
    ▼ [Pass 3]
    Soft Focus (Gaussian Blur blend) ─── Ping → Pong
    │
    ▼
    출력 텍스처
```

### 3.2 제안: AI 필터 삽입 지점 3곳

#### 삽입 지점 A: 전처리 단계 (Face Detection 이전)

```
카메라 프레임 → [AI 전처리] → Face Detection → ...
```

- **용도**: 저조도 향상, 노이즈 제거, 화이트밸런스 자동 보정
- **모델 후보**: Low-Light Enhancement Net (< 3ms)
- **장점**: 검출 정확도 향상
- **단점**: 전체 파이프라인 지연 증가
- **구현 위치**: `MediaPipeDetector::detect()` 내부, `preprocessImage()` 호출 전

#### 삽입 지점 B: 랜드마크 후, GPU Beauty 전 (권장)

```
Face Landmark → [AI 세그멘테이션] → GPU Beauty Pipeline
```

- **용도**: AI 피부 세그멘테이션 마스크 생성
- **모델 후보**: 아래 3.3 참조
- **장점**: 세그멘테이션 마스크를 GPU 셰이더에 텍스처로 전달 가능
- **단점**: 추가 텍스처 업로드 필요
- **구현 위치**: `GPUBeautyBackend::applyTextureId()` 내부, 필터 체인 실행 전
- **데이터 흐름**:
  1. Face Landmark 출력 (478개 좌표) → 피부 영역 ROI 계산
  2. ROI 크롭 → AI 세그멘테이션 모델 → 마스크 텍스처
  3. 마스크 텍스처를 각 GPU 셰이더에 uniform으로 전달

#### 삽입 지점 C: GPU 셰이더 체인 내부

```
Smoothing → [AI Style Transfer] → Combined Color → Soft Focus
```

- **용도**: AI 스타일 전이, 뉴럴 필터
- **모델 후보**: 경량 스타일 전이 모델 (< 10ms)
- **장점**: 기존 Ping-Pong 버퍼 재사용 가능
- **단점**: GPU→CPU→GPU 전환 오버헤드 (TFLite CPU 추론 시)
- **대안**: TFLite GPU Delegate로 GPU 텍스처 직접 입력 (Android)
- **구현 위치**: `applyTextureId()` 필터 체인 중간에 새로운 패스 추가

### 3.3 경량 AI 피부 세그멘테이션 모델 후보

#### 후보 1: MediaPipe Selfie Segmentation

| 항목 | 값 |
|------|------|
| 입력 | 256x256x3 RGB float |
| 출력 | 256x256x1 마스크 (0-1) |
| 크기 | ~200KB (landscape), ~1MB (general) |
| 추론 시간 | 2-4ms (Snapdragon 865+, GPU Delegate) |
| 장점 | MediaPipe 생태계 호환, 검증됨 |
| 단점 | 전신 세그멘테이션 (피부만 분리 불가) |

#### 후보 2: Face Parsing (BiSeNet 경량화)

| 항목 | 값 |
|------|------|
| 입력 | 128x128x3 또는 256x256x3 |
| 출력 | HxWx19 클래스별 마스크 |
| 크기 | ~2-5MB |
| 추론 시간 | 3-8ms (INT8 양자화 시) |
| 장점 | 피부/입술/눈/눈썹 등 개별 세그멘테이션 |
| 단점 | 커스텀 학습 및 변환 필요 |

#### 후보 3: 랜드마크 기반 피부 마스크 (ML-free)

| 항목 | 값 |
|------|------|
| 입력 | Face Landmark 478점 |
| 출력 | GPU에서 직접 마스크 렌더링 |
| 크기 | 0 (추가 모델 없음) |
| 추론 시간 | <1ms |
| 장점 | 추가 모델 불필요, 가장 빠름 |
| 단점 | 정밀도 제한, 머리카락/귀 경계 부정확 |

#### 권장 전략: 단계적 접근

1. **Phase 1 (즉시)**: 랜드마크 기반 피부 마스크 (모델 추가 없이 구현 가능)
   - Face Mesh 478점에서 피부 영역 삼각형 메시 생성
   - GPU에서 마스크 텍스처로 렌더링
   - `BeautyROIManager`가 이미 이 방향으로 설계됨

2. **Phase 2 (후속)**: MediaPipe Selfie Segmentation 통합
   - TFLite GPU Delegate로 추론
   - Face Landmark와 텍스처 공유 가능

3. **Phase 3 (고급)**: Face Parsing 모델
   - 피부/입술/눈 개별 마스킹
   - 영역별 차별화된 뷰티 필터 적용

---

## 4. 구현 인터페이스 설계 (초안)

### 4.1 AI 필터 추상 인터페이스

```cpp
// 제안: cpp/include/iris_sdk/ai_filter.h

class AIFilter {
public:
    virtual ~AIFilter() = default;

    // 초기화 (모델 로드)
    virtual bool initialize(const std::string& model_path,
                            int num_threads = 2) = 0;

    // 마스크 생성 (CPU 버퍼)
    virtual bool generateMask(const uint8_t* input_data,
                              int width, int height,
                              float* output_mask) = 0;

    // GPU 텍스처 입력 (Android GPU Delegate 사용 시)
    virtual bool generateMaskFromTexture(uint32_t input_texture,
                                         uint32_t* output_mask_texture,
                                         int width, int height) = 0;

    virtual void release() = 0;
    virtual bool isInitialized() const = 0;
};
```

### 4.2 GPU 파이프라인 통합 지점

```cpp
// GPUBeautyBackend::applyTextureId() 내 삽입 위치

// [기존] 필터 체인 실행 전
// [추가] AI 세그멘테이션 마스크 생성
GLuint skin_mask_texture = 0;
if (ai_filter_ && ai_filter_->isInitialized()) {
    ai_filter_->generateMaskFromTexture(input_tex_id, &skin_mask_texture,
                                         width, height);
}

// [기존] Smoothing pass에 마스크 전달
if (config.smoothing > 0.01f) {
    executeSmoothingPass(current_input, current_output->fbo_id,
                         width, height, config,
                         skin_mask_texture);  // 추가 파라미터
}
```

---

## 5. 성능 예산 분석

### 현재 파이프라인 타이밍 (목표: 33ms/프레임)

| 단계 | 소요 시간 (추정) |
|------|------|
| 전처리 (RGB 변환 + Letterbox) | 1-2ms |
| Face Detection | 3-5ms |
| Face Landmark V2 | 5-8ms |
| GPU Beauty (3 pass) | 2-4ms |
| 렌더링 (렌즈 오버레이) | 1-2ms |
| **합계** | **12-21ms** |
| **여유분** | **12-21ms** |

### AI 필터 추가 시 예산

| AI 필터 종류 | 추가 소요 시간 | 잔여 여유 |
|------|------|------|
| 랜드마크 기반 마스크 (ML-free) | <1ms | 11-20ms |
| Selfie Segmentation (GPU) | 2-4ms | 8-17ms |
| Face Parsing (INT8) | 3-8ms | 4-13ms |
| 스타일 전이 | 8-15ms | -3~6ms (위험) |

**결론**: 5ms 이내의 AI 필터는 30fps 유지에 안전합니다.

---

## 6. 다음 단계

1. [ ] 랜드마크 기반 피부 마스크 프로토타입 구현 (GPU 셰이더)
2. [ ] MediaPipe Selfie Segmentation TFLite 모델 테스트
3. [ ] AI 필터 인터페이스(`AIFilter`) C++ 구현
4. [ ] GPU Delegate 통합 테스트 (Android)
5. [ ] 모델 양자화 분석 (Task #6 연계)

---

## 변경 이력

| 날짜 | 내용 |
|------|------|
| 2026-02-09 | 초안 작성 - 모델 3종 입출력 스펙 문서화, AI 필터 삽입 포인트 설계 |
