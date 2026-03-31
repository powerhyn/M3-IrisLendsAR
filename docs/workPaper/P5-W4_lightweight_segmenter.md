# P5-W4: Lightweight Segmenter (경량 세그멘터 교체)

## 작업 개요
- **Phase**: P5 (경쟁사 대비 품질 갭 해소)
- **기간**: W1~W3와 독립 가능 (우선순위 낮음)
- **상태**: ⏳ 대기 (선택적)
- **선행 조건**: 없음
- **근거**: 현재 16MB 세그멘터는 체감 효과 대비 비용이 과도. Codex 권장: 200~800KB 경량 교체.

## 목표

16MB `selfie_multiclass_256x256.tflite`를 200~800KB급 경량 face-skin parser로 교체하여 APK 크기를 줄이고, 실행 정책을 최적화한다.

### 현재 문제점

| 항목 | 현재 상태 | 문제 |
|------|----------|------|
| 모델 크기 | 16MB | APK에 과도한 부담 |
| 입력 크기 | 256×256 (전체 프레임 letterbox) | 불필요하게 넓은 영역 |
| 실행 빈도 | 매 프레임 (direct detector 모드에서만) | CPU 부하. InferenceThread 모드에서는 비활성 |
| 체감 효과 | 극히 미미 | min 교집합 → 랜드마크가 이미 충분 |
| GPU 경로 | direct detector(CPU) 모드에서만 동작 | InferenceThread 모드 비활성 제약 (`frame_processor.cpp:103`) |

### 실제 테스트 결과

뷰티 필터를 최대 강도로 올려도 on/off 차이가 **육안으로 구분 불가**했음. 이유:
- `combined_mask[i] = min(seg_mask[i], landmark_mask[i])` → 랜드마크 마스크가 이미 잘 잡으면 세그멘테이션이 깎아낼 게 없음
- 차이가 나는 건 머리카락/이마 경계 정도이며, 스무딩 0.3~0.5에서는 미미

---

## W4-01: 경량 Face-Skin Parser 모델 선정

### 상태: ⏳ 대기

### 모델 후보

| 후보 | 크기 | 입력 | 클래스 | 출처 |
|------|------|------|--------|------|
| BiSeNet-lite (INT8) | ~300KB | 128×128 | skin/non-skin (2-class) | 공개 |
| Face Parsing Lite | ~500KB | 128×128 | 5-class (skin/eye/lip/eyebrow/bg) | 자체 학습 |
| MediaPipe selfie_segmentation | 244KB | 256×256 | 2-class (person/bg) | Google (피팅몬스터 동봉) |
| faceart 스타일 (Perfect Corp 참고) | ~413KB | 128×128 | face regions | 자체 학습 |

### 선택 기준

1. **크기**: 800KB 이하 (현재 16MB의 1/20)
2. **속도**: 3ms 이하 (face ROI 128×128 기준, CPU)
3. **정밀도**: 피부/비피부 경계에서 현재 16MB 모델의 80%+ 정확도
4. **클래스**: 최소 skin/non-skin. 5-class면 보호 마스크 자동화 가능

### 결정 필요 사항

- [ ] 2-class (skin/non-skin) vs 5-class (skin/eye/lip/eyebrow/bg)
- [ ] 공개 모델 vs 자체 학습
- [ ] face ROI 내 실행 vs 전체 프레임 (ROI 권장)

---

## W4-02: ROI 내 실행 + 매 N프레임 정책

### 상태: ⏳ 대기

### 작업 내용

1. **Face ROI 내에서만 실행**
   ```
   현재: 전체 프레임 → 256×256 리사이즈 → 추론
   변경: face_rect ROI 크롭 → 128×128 리사이즈 → 추론
   
   효과: 입력 해상도 절반 + 불필요 영역 제거 → 속도 2~4× 향상
   ```

2. **매 N프레임 실행**
   ```
   피부 마스크는 매 프레임 크게 변하지 않음
   → 3~5프레임마다 한 번 실행
   → 사이 프레임은 이전 마스크 재사용
   → 얼굴 이동 시 ROI 좌표 보정으로 정렬
   ```

3. **GPU 경로 호환**
   ```
   현재: segmentation이 CPU 전용 직접 호출 경로에서만 동작
   변경: GPU beauty backend에서도 별도 스레드로 실행 가능하도록
   ```

### 수정 대상 파일

| 파일 | 수정 내용 |
|------|----------|
| `cpp/src/mediapipe_detector.cpp` | ROI 크롭 추론, N프레임 정책 |
| `cpp/src/gpu/gpu_beauty_backend.cpp` | 세그멘테이션 마스크 캐싱 |

---

## W4-03: 16MB 모델 제거 / 옵션화

### 상태: ⏳ 대기

### 작업 내용

1. **기본 모델 교체**
   - `selfie_multiclass_256x256.tflite` (16MB) → 경량 모델 (300~800KB)
   - CMakeLists.txt에서 모델 경로 변경
   - Android assets에서 교체

2. **16MB 모델은 옵션으로 유지**
   ```cpp
   // 빌드 옵션으로 HQ 세그멘터 포함 여부 결정
   option(IRIS_SDK_HQ_SEGMENTER "Include 16MB HQ segmentation model" OFF)
   ```

3. **API 정리**
   ```cpp
   // 세그멘터 품질 선택
   enum class SegmenterQuality {
       LITE,    // 경량 (기본, 300~800KB)
       HQ,      // 고품질 (16MB, 옵션)
       OFF      // 비활성
   };
   ```

### 수정 대상 파일

| 파일 | 수정 내용 |
|------|----------|
| `cpp/CMakeLists.txt` | 모델 경로 + 빌드 옵션 |
| `cpp/src/mediapipe_detector.cpp` | 모델 로딩 분기 |
| `cpp/include/iris_sdk/sdk_api.h` | SegmenterQuality API |
| `shared/models/` | 경량 모델 추가, 16MB 모델 옵션화 |

---

## 예상 효과

| 항목 | 현재 | W4 후 |
|------|------|-------|
| 세그멘터 크기 | 16MB | 300~800KB |
| 추론 시간 | ~15ms (256×256) | ~3ms (128×128 ROI) |
| 실행 빈도 | 매 프레임 | 3~5프레임마다 |
| APK 크기 절감 | — | ~15MB 감소 |
| 체감 품질 | 미미 (교집합 방식) | 동등 (경량이라도 충분) |

---

## 비고

이 작업은 **렌즈 품질과 직접 관련 없음**. 뷰티 필터의 경계 품질 개선이며, W1~W3가 렌즈 품질에 직접적인 영향을 준다. Codex 의견: "That matters for beauty quality, but it is not the first thing I would fund if the goal is closing the lens-quality gap fastest."

리소스가 제한적이면 W1~W3를 먼저 완료한 후 W4를 진행한다.
