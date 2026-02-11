# P3-W1-03: 모델 양자화 (FP16→INT8) 트레이드오프 분석

**상태**: ✅ 완료
**작성일**: 2026-02-09
**담당**: ml-vision
**선행 태스크**: P3-W1-02 (TFLite 모델 스펙 문서화)

---

## 1. 현재 모델 양자화 현황

### 1.1 텐서 타입 분포 (FlatBuffer 파싱 결과)

| 모델 | 파일 크기 | FLOAT32 | FLOAT16 | INT32 | 양자화 수준 |
|------|-----------|---------|---------|-------|------------|
| face_detection_short_range | 224 KB | 115 | 74 | 11 | 혼합 정밀도 (가중치 FP16) |
| face_landmark V1 | 1.2 MB | 91 | 106 | 3 | FP16 가중치 우세 |
| face_landmark V2 | 2.4 MB | 1 | 199 | 0 | 거의 완전 FP16 |
| iris_landmark | 2.5 MB | 199 | 0 | 1 | **완전 FP32 (양자화 미적용)** |

### 1.2 완전한 입출력 텐서 형태 (FlatBuffer 파싱 검증)

#### Face Detection
- **입력**: `[1, 128, 128, 3]` FLOAT32
- **출력 0**: `[1, 896, 16]` (인덱스 175) - regressors
- **출력 1**: `[1, 896, 1]` (인덱스 174) - classificators

#### Face Landmark V1
- **입력**: `[1, 192, 192, 3]` FLOAT32
- **출력 0**: `[1, 1, 1, 1404]` (인덱스 213) - 468 * 3 랜드마크
- **출력 1**: `[1, 1, 1, 1]` (인덱스 210) - face presence score

#### Face Landmark V2
- **입력**: `[1, 256, 256, 3]` FLOAT32
- **출력 0**: `[1, 1, 1, 1434]` (인덱스 473) - 478 * 3 랜드마크
- **출력 1**: `[1, 1, 1, 1]` (인덱스 472) - face presence score
- **출력 2**: `[1, 1]` (인덱스 475) - 추가 신뢰도 (blendshapes 관련 추정)

#### Iris Landmark
- **입력**: `[1, 64, 64, 3]` FLOAT32
- **출력 0**: `[1, 213]` (인덱스 384) - 71 * 3 (눈 윤곽 + 홍채)
- **출력 1**: `[1, 15]` (인덱스 385) - 5 * 3 홍채 전용 랜드마크

---

## 2. INT8 양자화 트레이드오프 분석

### 2.1 양자화 이론적 배경

| 정밀도 | 비트 | 크기 배수 | 추론 속도 | 정확도 영향 |
|--------|------|-----------|-----------|------------|
| FP32 | 32 | 1.0x (기준) | 기준 | 기준 |
| FP16 | 16 | 0.5x | 1.5-2x | 무시 가능 (<0.1%) |
| INT8 (PTQ) | 8 | 0.25x | 2-4x | 1-3% 하락 |
| INT8 (QAT) | 8 | 0.25x | 2-4x | <1% 하락 |
| INT4 | 4 | 0.125x | 3-6x | 3-8% 하락 |

PTQ = Post-Training Quantization (학습 후 양자화)
QAT = Quantization-Aware Training (양자화 인식 학습)

### 2.2 모델별 INT8 양자화 가능성 평가

#### Face Detection (BlazeFace) - 양자화 권장도: **중**

| 항목 | 분석 |
|------|------|
| 현재 상태 | 이미 FP16 혼합 정밀도 |
| INT8 기대 효과 | 크기: 224KB → ~112KB, 속도: ~1.5x |
| 정확도 위험 | 앵커 기반 검출은 양자화에 비교적 강건 |
| 권장 | INT8 dynamic range quantization 적용 가능 |
| 주의 | 이미 충분히 빠르므로 (3-5ms) 우선순위 낮음 |

#### Face Landmark V2 - 양자화 권장도: **낮음**

| 항목 | 분석 |
|------|------|
| 현재 상태 | 거의 완전 FP16 (199/200 텐서) |
| INT8 기대 효과 | 크기: 2.4MB → ~1.2MB, 속도: ~1.3x |
| 정확도 위험 | **높음** - 랜드마크 좌표는 서브픽셀 정밀도 필요 |
| 리스크 | 478개 좌표의 미세 오차 → 홍채 추적 떨림 |
| 권장 | FP16 유지, INT8 지양 |
| 근거 | Google이 이미 FP16으로 배포 (추가 양자화는 비권장) |

#### Iris Landmark - 양자화 권장도: **높음 (최우선)**

| 항목 | 분석 |
|------|------|
| 현재 상태 | **완전 FP32** (양자화 미적용) |
| INT8 기대 효과 | 크기: 2.5MB → ~650KB, 속도: ~2-3x |
| 정확도 위험 | 중간 - 64x64 입력이므로 양자화 오차 상대적으로 큼 |
| 권장 | FP16 양자화 우선, INT8은 PTQ 후 정확도 검증 필요 |
| V2 사용 시 | **불필요** (V2 모델에 홍채 내장, 이 모델 사용 안 함) |

### 2.3 양자화 우선순위 종합

```
우선순위 1: iris_landmark FP32 → FP16
  - 효과: 크기 50% 감소, 속도 1.5-2x 향상
  - 위험: 매우 낮음 (기본적인 최적화)
  - 비고: V2 모델 사용 시 이 모델 자체가 불필요

우선순위 2: V2 파이프라인으로 전환 (양자화 대안)
  - 효과: 모델 3개 → 2개, 전체 추론 시간 감소
  - 위험: 낮음 (이미 코드에 V2 지원 구현됨)
  - 비고: 가장 효과적인 "양자화" 전략

우선순위 3: face_detection INT8 동적 양자화
  - 효과: 크기 추가 50% 감소, 속도 소폭 향상
  - 위험: 낮음 (앵커 검출은 양자화 강건)
  - 비고: 이미 충분히 빠르므로 선택적
```

---

## 3. Android GPU Delegate 활용 방안

### 3.1 현재 구현 상태

코드 분석 결과 (`mediapipe_detector.cpp:460-530`):
- `IRIS_SDK_GPU_ENABLED` 매크로로 GPU Delegate 조건부 컴파일
- `TfLiteGpuDelegateV2Create()` / `TfLiteGpuDelegateV2Delete()` 사용
- 모델별 개별 GPU delegate 관리 (3개)
- GPU 실패 시 CPU 자동 폴백 구현됨

### 3.2 GPU Delegate 최적화 제안

#### A. 현재 이슈

```cpp
// 현재: 모델별 개별 GPU delegate (메모리 3배)
TfLiteDelegate* gpu_delegate_face_detection;
TfLiteDelegate* gpu_delegate_face_landmark;
TfLiteDelegate* gpu_delegate_iris_landmark;
```

#### B. 최적화 방안

| 방안 | 설명 | 기대 효과 |
|------|------|-----------|
| Shared GPU Context | 모델간 GPU 컨텍스트 공유 | GPU 메모리 30-40% 절감 |
| Warm-up on Init | 초기화 시 더미 추론 실행 | 첫 프레임 지연 제거 |
| SSBO Backend | OpenGL SSBO 기반 추론 | CPU↔GPU 전송 최소화 |
| Serialized Model | 사전 컴파일된 GPU 모델 캐시 | 초기화 시간 50% 단축 |

#### C. GPU Delegate + Beauty Pipeline 통합

```
현재:
  Camera Frame → [CPU] TFLite 추론 → [GPU] Beauty Shader
                   ↑                     ↑
                   CPU 영역              GPU 영역
                   (데이터 전송 오버헤드)

개선안:
  Camera Frame [GPU Texture]
       ↓
  [GPU] TFLite Delegate (GPU에서 추론)
       ↓
  [GPU] Beauty Shader (텍스처 직접 전달)
       ↓
  Screen
```

- **핵심**: GPU Delegate의 `TfLiteGpuDelegateOptionsV2`에서
  `is_precision_loss_allowed = true` + `inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER` 설정
- **요구 조건**: 모든 연산이 GPU에서 실행 가능해야 함 (일부 ops 미지원 시 CPU 폴백)

### 3.3 NNAPI Delegate 대안

| 항목 | GPU Delegate | NNAPI Delegate |
|------|-------------|----------------|
| 지원 범위 | OpenGL ES 3.1+ | Android 8.1+ |
| 가속기 | GPU | NPU/DSP/GPU (SoC 의존) |
| 성능 | 일관적 | SoC별 차이 큼 |
| 호환성 | 대부분 ops 지원 | ops 지원 제한적 |
| 권장 | 범용 | Qualcomm/Samsung 최적화 시 |

---

## 4. 추론 최적화 전략 종합

### 4.1 즉시 적용 가능 (코드 변경만)

| 최적화 | 기대 효과 | 구현 난이도 |
|--------|-----------|------------|
| V2 전용 모드 강제 | 모델 로드 1개 감소, 추론 1회 감소 | 낮음 |
| 입력 버퍼 사전 할당 | 메모리 할당 오버헤드 제거 | 이미 구현됨 |
| 스레드 수 자동 조정 | SoC별 최적 스레드 수 | 낮음 |
| Face Detection 스킵 (추적 시) | 33ms/프레임 절약 | 이미 구현됨 |

### 4.2 중기 (1-2주 소요)

| 최적화 | 기대 효과 | 구현 난이도 |
|--------|-----------|------------|
| iris_landmark FP16 양자화 | 크기 50% 감소, 속도 1.5x | 낮음 |
| GPU Delegate 컨텍스트 공유 | GPU 메모리 30% 절감 | 중간 |
| Serialized GPU Model 캐시 | 초기화 50% 단축 | 중간 |
| XNNPACK Delegate 활성화 | CPU 추론 1.5-2x 향상 | 낮음 |

### 4.3 장기 (모델 재학습 필요)

| 최적화 | 기대 효과 | 구현 난이도 |
|--------|-----------|------------|
| QAT INT8 양자화 | 크기 75% 감소, 속도 3x | 높음 |
| Knowledge Distillation | 경량 모델 학습 | 높음 |
| 커스텀 경량 모델 | 최적화된 아키텍처 | 매우 높음 |

### 4.4 XNNPACK Delegate 활성화 (즉시 적용 권장)

현재 코드에서 XNNPACK이 명시적으로 활성화되지 않았습니다.
TFLite 2.x에서 XNNPACK은 기본 비활성이지만, 활성화 시 CPU 추론이 1.5-2x 빨라집니다.

```cpp
// 제안: loadModel()에 XNNPACK delegate 추가
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

TfLiteXNNPackDelegateOptions xnnpack_opts = TfLiteXNNPackDelegateOptionsDefault();
xnnpack_opts.num_threads = num_threads;
auto* xnnpack_delegate = TfLiteXNNPackDelegateCreate(&xnnpack_opts);
interpreter->ModifyGraphWithDelegate(xnnpack_delegate);
```

---

## 5. 뷰티 필터용 피부톤 분류 모델 프로토타입 가능성

### 5.1 피부톤 분류의 필요성

현재 뷰티 필터는 모든 피부톤에 동일한 파라미터를 적용합니다.
피부톤별 최적화된 파라미터를 적용하면 자연스러운 결과를 얻을 수 있습니다.

### 5.2 접근 방식

#### 방법 A: ML-Free (Face Landmark 기반)

```
Face Landmark 출력 → 피부 영역 ROI 추출
    → 평균 RGB/HSV 계산
    → 룩업 테이블로 피부톤 분류
```

- **장점**: 추가 모델 불필요, <1ms
- **단점**: 조명 변화에 민감
- **분류**: Fitzpatrick Scale I-VI (6단계)

#### 방법 B: 경량 분류 모델 (TFLite)

```
Face Landmark ROI → 32x32 크롭 → MobileNetV3 Tiny
    → 피부톤 6클래스 분류
```

- **모델 크기**: ~100KB (양자화 시)
- **추론 시간**: <1ms
- **장점**: 조명 불변성, 높은 정확도
- **단점**: 학습 데이터 필요

### 5.3 권장

**Phase 1에서는 방법 A (ML-Free)를 구현하고**, 정확도가 부족할 경우 방법 B를 후속 구현합니다.

```cpp
// 제안: 피부톤 추정 유틸리티
enum class SkinTone { VeryLight, Light, Medium, Olive, Brown, Dark };

SkinTone estimateSkinTone(const uint8_t* frame_data,
                          int width, int height,
                          const float* face_landmarks,
                          int landmark_count);
```

---

## 6. 결론 및 권장 액션 아이템

### 핵심 발견사항

1. **iris_landmark 모델이 유일한 FP32 모델** → FP16 양자화 즉시 적용 가능
2. **V2 파이프라인 전환이 가장 효과적인 최적화** → iris_landmark 모델 자체가 불필요
3. **Face Landmark V2는 이미 FP16** → 추가 양자화 비권장 (정밀도 손실 위험)
4. **XNNPACK Delegate 미활성** → 즉시 활성화로 CPU 추론 1.5-2x 향상 기대

### 우선순위 액션

| 순위 | 액션 | 난이도 | 기대 효과 |
|------|------|--------|-----------|
| 1 | V2 전용 모드 확인 및 V1 폴백 제거 검토 | 낮음 | 추론 1회 제거, 메모리 2.5MB 절감 |
| 2 | XNNPACK Delegate 활성화 | 낮음 | CPU 추론 1.5-2x |
| 3 | GPU Delegate warm-up 구현 | 낮음 | 첫 프레임 지연 제거 |
| 4 | GPU 컨텍스트 공유 | 중간 | GPU 메모리 30% 절감 |
| 5 | 피부톤 분류 ML-Free 프로토타입 | 중간 | 뷰티 필터 품질 향상 |

---

## 변경 이력

| 날짜 | 내용 |
|------|------|
| 2026-02-09 | 초안 작성 - 양자화 현황 분석, GPU Delegate 방안, 추론 최적화 전략 |
