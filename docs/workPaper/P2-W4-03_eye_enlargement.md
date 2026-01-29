# P2-W4-03. Eye Enlargement 효과 구현

## 작업 정보

| 항목 | 내용 |
|------|------|
| **작업 ID** | P2-W4-03 |
| **Phase** | Phase 4: Face Warp 구현 |
| **상태** | ✅ 완료 |
| **완료일** | 2026-01-29 |
| **예상 기간** | 2일 |
| **실제 기간** | 1일 |
| **의존성** | P2-W4-01 (Grid Mesh), P2-W4-02 (FaceWarpController) |
| **담당** | systems-programming:cpp-pro |

---

## 1. 목표

눈 영역 랜드마크 기반 자연스러운 눈 확대 효과 구현

### 핵심 산출물
- [x] 눈 중심(홍채) 기준 방사형 확대
- [x] 눈꺼풀/눈썹 영역 자연스러운 변형
- [x] 좌우 대칭 처리
- [x] 단위 테스트 (12개 테스트 케이스)

---

## 2. 구현 내용

### 2.1 MediaPipe 478 랜드마크 인덱스

```cpp
// 헤더에 정의된 상수들 (face_warp_controller.h)

// 홍채 중심
static constexpr int LEFT_IRIS_CENTER = 468;
static constexpr int RIGHT_IRIS_CENTER = 473;

// 왼쪽 눈 윤곽 (16개 랜드마크)
static constexpr std::array<int, 16> LEFT_EYE_CONTOUR = {
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    173, 157, 158, 159, 160, 161, 246
};

// 오른쪽 눈 윤곽 (16개 랜드마크)
static constexpr std::array<int, 16> RIGHT_EYE_CONTOUR = {
    362, 382, 381, 380, 374, 373, 390, 249, 263,
    466, 388, 387, 386, 385, 384, 398
};

// 눈썹 (각 10개 랜드마크)
static constexpr std::array<int, 10> LEFT_EYEBROW = {
    70, 63, 105, 66, 107, 55, 65, 52, 53, 46
};
static constexpr std::array<int, 10> RIGHT_EYEBROW = {
    300, 293, 334, 296, 336, 285, 295, 282, 283, 276
};
```

### 2.2 효과 파라미터

```cpp
// 최대 확대 비율: 25%
static constexpr float MAX_EYE_ENLARGE_SCALE = 0.25f;

// 눈썹 리프트 비율 (눈 확대의 30%)
static constexpr float EYEBROW_LIFT_RATIO = 0.3f;
```

### 2.3 알고리즘

```
1. 눈 중심 좌표 획득 (홍채 중심 랜드마크)
2. 눈 반지름 계산 (윤곽 점들의 최대 거리)
3. 방사형 확대 적용:
   - 각 윤곽 점에서 중심 방향 벡터 계산
   - 거리 비율에 따른 가중치 적용 (내부: 감소, 외부: 전체)
   - 확대 변위 = 거리 × scale_factor × weight
4. 눈썹 리프트:
   - 눈 확대에 비례하여 상향 이동
   - 이동량 = eye_radius × scale_factor × 0.3
```

---

## 3. 변경된 파일

### 3.1 헤더 파일
- **파일**: `cpp/include/iris_sdk/warp/face_warp_controller.h`
- **변경 내용**:
  - 눈 랜드마크 상수 추가 (IRIS_CENTER, EYE_CONTOUR, EYEBROW)
  - MAX_EYE_ENLARGE_SCALE, EYEBROW_LIFT_RATIO 상수 추가
  - `applyEyeEnlargementSingle` 템플릿 메서드 선언
  - `calculateEyeRadius` 정적 템플릿 메서드 선언

### 3.2 구현 파일
- **파일**: `cpp/src/warp/face_warp_controller.cpp`
- **변경 내용**:
  - `applyEnlargeEyes()` 구현 (placeholder 대체)
  - `applyEyeEnlargementSingle()` 템플릿 메서드 구현
  - `calculateEyeRadius()` 템플릿 메서드 구현
  - `registerWarpControlPoints()` 업데이트 (눈 랜드마크 추가)

### 3.3 테스트 파일
- **파일**: `cpp/tests/test_eye_enlargement.cpp` (신규)
- **테스트 케이스** (12개):
  1. RadialExpansionFromCenter - 방사형 확대 동작
  2. SymmetricLeftRightEyes - 좌우 대칭
  3. EyebrowLiftsUp - 눈썹 상향 이동
  4. ZeroStrengthNoChange - 강도 0일 때 무변화
  5. MaxStrengthBounded - 최대 강도 제한
  6. StrengthScaling - 강도 스케일링
  7. CombinedWithSlimFace - 슬림페이스 효과와 결합
  8. CombinedWithThinChin - V라인 효과와 결합
  9. AllEffectsCombined - 모든 효과 결합
  10. ExpansionDirectionIsOutward - 확장 방향 검증
  11. LandmarkConstantsValid - 랜드마크 상수 유효성
  12. EffectParametersReasonable - 파라미터 범위 검증

### 3.4 CMakeLists.txt
- **파일**: `cpp/tests/CMakeLists.txt`
- **변경 내용**: test_eye_enlargement 타겟 추가

---

## 4. 테스트 결과

```
[==========] Running 12 tests from 1 test suite.
[----------] 12 tests from EyeEnlargementTest
[ RUN      ] EyeEnlargementTest.RadialExpansionFromCenter
[       OK ] EyeEnlargementTest.RadialExpansionFromCenter (1 ms)
[ RUN      ] EyeEnlargementTest.SymmetricLeftRightEyes
[       OK ] EyeEnlargementTest.SymmetricLeftRightEyes (1 ms)
[ RUN      ] EyeEnlargementTest.EyebrowLiftsUp
[       OK ] EyeEnlargementTest.EyebrowLiftsUp (0 ms)
[ RUN      ] EyeEnlargementTest.ZeroStrengthNoChange
[       OK ] EyeEnlargementTest.ZeroStrengthNoChange (0 ms)
[ RUN      ] EyeEnlargementTest.MaxStrengthBounded
[       OK ] EyeEnlargementTest.MaxStrengthBounded (1 ms)
[ RUN      ] EyeEnlargementTest.StrengthScaling
[       OK ] EyeEnlargementTest.StrengthScaling (1 ms)
[ RUN      ] EyeEnlargementTest.CombinedWithSlimFace
[       OK ] EyeEnlargementTest.CombinedWithSlimFace (2 ms)
[ RUN      ] EyeEnlargementTest.CombinedWithThinChin
[       OK ] EyeEnlargementTest.CombinedWithThinChin (0 ms)
[ RUN      ] EyeEnlargementTest.AllEffectsCombined
[       OK ] EyeEnlargementTest.AllEffectsCombined (0 ms)
[ RUN      ] EyeEnlargementTest.ExpansionDirectionIsOutward
[       OK ] EyeEnlargementTest.ExpansionDirectionIsOutward (1 ms)
[ RUN      ] EyeEnlargementTest.LandmarkConstantsValid
[       OK ] EyeEnlargementTest.LandmarkConstantsValid (0 ms)
[ RUN      ] EyeEnlargementTest.EffectParametersReasonable
[       OK ] EyeEnlargementTest.EffectParametersReasonable (0 ms)
[==========] 12 tests from 1 test suite ran. (12 ms total)
[  PASSED  ] 12 tests.
```

기존 FaceWarpController 테스트 (23개)도 모두 통과.

---

## 5. 완료 기준 체크리스트

- [x] 눈 중심 기준 방사형 확대 구현
- [x] 비선형 확대 함수 (inner region weight)
- [x] 외곽 영역 부드러운 감쇠 (RBF 보간)
- [x] 눈썹 영역 연동 조정 (상향 리프트)
- [ ] GPU 셰이더 대안 구현 (선택적, 미구현)
- [x] 좌우 대칭 검증
- [x] 단위 테스트 통과 (12/12)

---

## 6. 사용법

```cpp
#include "iris_sdk/warp/face_warp_controller.h"
#include "iris_sdk/warp/grid_mesh.h"

// GridMesh 초기화
iris_sdk::warp::GridMesh mesh;
mesh.initialize(20, face_rect);
mesh.setControlPoints(face_landmarks, width, height);

// FaceWarpController로 효과 적용
iris_sdk::warp::FaceWarpController controller;
iris_sdk::warp::WarpConfig config;
config.slimFace = 0.5f;     // 슬림페이스
config.thinChin = 0.5f;     // V라인
config.enlargeEyes = 0.5f;  // 눈 확대

controller.applyWarp(mesh, face_landmarks, config);

// mesh의 final positions 사용하여 렌더링
```

---

## 7. 다음 작업

- **P2-W5-01**: JNI 바인딩 및 Android 통합
- **선택적**: GPU 셰이더 기반 눈 확대 최적화
