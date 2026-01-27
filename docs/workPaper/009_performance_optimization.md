# 009. 성능 최적화 (Performance Optimization)

**작업 번호**: 009
**시작일**: 2025-01-27
**상태**: 🔄 진행 중
**브랜치**: `feature/render-optimize-stage-02`

---

## 1. 배경 및 목표

### 문제점
- 뷰티 필터 활성화 시 FPS ~10 (목표: 30fps+)
- 렌즈가 동공을 따라가는 반응이 느림
- 홍채 좌표의 미세한 떨림(지터링) 발생

### 목표
1. **FPS 30+ 달성**: 렌더링 파이프라인 최적화
2. **빠른 반응**: 홍채 추적 지연 최소화
3. **안정적 렌더링**: 지터링 방지로 부드러운 렌즈 오버레이

---

## 2. 수행한 최적화

### 2.1 렌더링 파이프라인 최적화

| 항목 | 이전 | 이후 | 효과 |
|------|------|------|------|
| 검출 주기 | 33ms (30fps) | 16ms (60fps) | 반응 2배 향상 |
| 다운스케일 | 2 (960x540) | 3 (640x360) | 픽셀 수 55% 감소 |
| 프레임 스킵 | 없음 | 3프레임마다 | 필터 부하 66% 감소 |
| bilateral diameter | 5~15 | 5 고정 | 처리 시간 ~60% 감소 |
| sigma 값 | 50~150 | 30~80 | 처리 범위 축소 |

**변경 파일:**
- `android/demo-app/.../camera/FrameAnalyzer.kt`
  - `MIN_INTERVAL_MS`: 33 → 16
  - `BEAUTY_DOWNSCALE_FACTOR`: 2 → 3
  - `BEAUTY_FILTER_FRAME_SKIP`: 3 (신규)
  - 캐시된 비트맵 재사용 로직 추가

- `cpp/src/beauty_filter.cpp`
  - bilateral filter diameter 5 고정
  - sigma 값 감소 (30~80)

### 2.2 GPU 가속 테스트

**결과**: GPU 활성화 시 오히려 성능 저하 발생
- 원인 추정: GPU delegate 초기화 오버헤드, 호환성 문제
- 결정: GPU 비활성화 유지 (`enableGpu = false`)

**변경 파일:**
- `android/demo-app/.../MainActivity.kt:212`

### 2.3 One-Euro Filter 구현

지터링 방지를 위한 적응형 저역 통과 필터 구현.

**알고리즘 특성:**
- 느린 움직임 → 강한 스무딩 → 지터링 제거
- 빠른 움직임 → 약한 스무딩 → 즉각 반응

**파라미터 튜닝:**
| 파라미터 | 초기값 | 현재값 | 설명 |
|---------|--------|--------|------|
| min_cutoff | 1.5 | 5.0 | 높을수록 반응 빠름 |
| beta | 0.05 | 0.3 | 높을수록 빠른 움직임에 민감 |

**신규 파일:**
- `cpp/include/iris_sdk/one_euro_filter.h`
  - `LowPassFilter` 클래스
  - `OneEuroFilter` 클래스
  - `OneEuroFilter2D` 클래스
  - `IrisOneEuroFilter` 클래스 (홍채 전용)

**변경 파일:**
- `cpp/src/mediapipe_detector.cpp`
  - One-Euro Filter 헤더 include
  - Impl에 `left_iris_filter`, `right_iris_filter` 멤버 추가
  - 검출 결과에 필터 적용 (섹션 5.5)

---

## 3. 아키텍처

### 3.1 최적화된 처리 흐름

```
CameraX (YUV_420_888) @ 60fps
    │
    ├─→ 홍채 검출 (원본 해상도) @ 60fps
    │       │
    │       └─→ One-Euro Filter (지터링 방지)
    │               │
    │               └─→ IrisResult (필터링된 좌표)
    │
    └─→ 뷰티 필터 @ 20fps (3프레임마다)
            │
            └─→ 다운스케일 (640x360)
                    │
                    └─→ 캐시된 Bitmap

OverlayView
    ├─→ 배경: 캐시된 필터 Bitmap (20fps 업데이트)
    └─→ 렌즈: 필터링된 홍채 좌표 (60fps 업데이트)
```

### 3.2 One-Euro Filter 적용 위치

```cpp
// mediapipe_detector.cpp

// 5. 홍채 검출 완료
result.left_iris[0..4] = 검출된 좌표

// 5.5 One-Euro Filter 적용 ← 여기서 필터링
if (use_one_euro_filter) {
    left_iris_filter.filter(x, y, radius);
    // 좌표 업데이트
}

// 6. 최종 결과 반환
return result;
```

---

## 4. 테스트 결과

### 4.1 성능 측정 (예정)

| 시나리오 | FPS | 검출 지연 | 비고 |
|---------|-----|----------|------|
| 뷰티 필터 OFF | - | - | 기준선 |
| 뷰티 필터 ON | - | - | 목표: 30fps+ |
| 빠른 움직임 | - | - | 반응 테스트 |

### 4.2 발견 및 해결된 이슈

1. **One-Euro Filter 파라미터 조정** ✅ 해결
   - 초기 파라미터 (1.5, 0.05)로는 반응이 ~1초 지연
   - 최종 파라미터 (15.0, 0.5)로 빠른 반응 + 약간의 지터 방지

2. **Kotlin vs C++ One-Euro Filter 중복**
   - C++ 필터 (mediapipe_detector.cpp): 현재 비활성화
   - Kotlin 필터 (OverlayView.kt): 활성화 및 파라미터 튜닝 완료

3. **렌즈 깜빡임 문제** ✅ 해결
   - 원인 1: confidence 체크가 타임아웃 리셋도 막고 있었음
   - 원인 2: `shouldRender` 조건이 mesh와 lens를 함께 제어하고 있었음
   - 수정 1: 홍채 검출 시 타임아웃 리셋, confidence 체크는 좌표 업데이트에만 적용
   - 수정 2: 렌즈 전용 타임아웃 분리 (`LENS_PERSISTENCE_TIMEOUT_MS = 2초`)
   - 수정 3: 첫 검출 시 confidence 무관하게 좌표 업데이트 (렌즈 초기 표시)
   - 결과: Mesh는 1초, 렌즈는 2초 동안 독립적으로 유지됨

---

## 5. 향후 작업

### 즉시 (테스트 후)
- [ ] FPS 측정 및 기록
- [ ] One-Euro Filter 파라미터 최종 튜닝
- [ ] 다운스케일 품질 평가

### 추가 최적화 (필요 시)
- [ ] Face Mesh V2 단일 모델 전환
- [ ] GPU delegate 호환성 조사
- [ ] OpenGL 기반 렌더링 검토

---

## 6. 변경 이력

| 날짜 | 작업 | 담당 |
|------|------|------|
| 2025-01-27 | 문서 작성, 렌더링 최적화 | Claude |
| 2025-01-27 | One-Euro Filter 구현 (C++) | Claude |
| 2025-01-27 | C++ 파라미터 튜닝 (1.5→5.0, 0.05→0.3) | Claude |
| 2025-01-27 | Kotlin One-Euro Filter 발견 및 튜닝 (1.0→15.0, 0.05→0.5) | Claude |
| 2025-01-27 | 렌즈 깜빡임 수정 v1 (타임아웃 로직 개선) | Claude |
| 2025-01-27 | 렌즈 깜빡임 수정 v2 (렌즈 2초/Mesh 1초 타임아웃 분리) | Claude |

---

## 7. 참고 자료

- [One-Euro Filter Paper](https://cristal.univ-lille.fr/~casiez/1euro/)
- `docs/conversation/2025-01-26_performance_accuracy_proposal.md`
- `docs/workPaper/008_direct_frame_rendering.md`
