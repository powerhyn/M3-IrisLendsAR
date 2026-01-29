# P2-W4-02. Slim Face / V-Line 효과 구현

## 작업 정보

| 항목 | 내용 |
|------|------|
| **작업 ID** | P2-W4-02 |
| **Phase** | Phase 4: Face Warp 구현 |
| **상태** | ✅ 완료 |
| **예상 기간** | 2일 |
| **완료일** | 2026-01-29 |
| **의존성** | P2-W4-01 (Grid Mesh) |
| **담당** | systems-programming:cpp-pro |

---

## 1. 목표

얼굴 외곽 랜드마크 기반 자연스러운 슬림 효과 및 V-라인 구현

### 핵심 산출물
- slimFace: 전체 얼굴 너비 축소 (볼 내측 이동)
- thinChin: 턱선 V-라인 효과 (턱 내측+상단 이동)
- 자연스러운 그라데이션 변위 (RBF 보간)

---

## 2. 랜드마크 분석

### 2.1 얼굴 외곽 랜드마크 (Face Oval)

```
          10 (이마 중앙)
         /  \
       338  109
      /        \
    297         67
    |           |
   332         103
    |           |
   284         132
    \           /
    251       172
      \       /
       389  136
        \   /
         152 (턱 끝)
```

### 2.2 슬림 효과 영역

| 영역 | 랜드마크 | 변위 방향 |
|------|----------|-----------|
| 좌측 볼 | 234, 93, 132, 58, 172, 136, 150, 149 | 우측 (내측) |
| 우측 볼 | 454, 323, 361, 288, 397, 365, 379, 378 | 좌측 (내측) |
| 좌측 턱 | 136, 150, 149, 176, 148 | 우측+상단 |
| 우측 턱 | 365, 379, 378, 400, 377 | 좌측+상단 |
| 턱 끝 | 152, 175, 199, 18, 17, 200, 421, 418 | 상단 |

---

## 3. 구현 상세

### 3.1 FaceWarpController 클래스

**파일**: `cpp/include/iris_sdk/warp/face_warp_controller.h`

```cpp
namespace iris_sdk {
namespace warp {

struct WarpConfig {
    float slimFace = 0.0f;     ///< Slim face effect (0.0 ~ 1.0)
    float thinChin = 0.0f;     ///< V-line effect (0.0 ~ 1.0)
    float enlargeEyes = 0.0f;  ///< Eye enlargement (placeholder for P2-W4-03)

    bool hasActiveEffect() const;
    WarpConfig clamped() const;
};

class FaceWarpController {
public:
    // Landmark index constants
    static constexpr std::array<int, 8> LEFT_CHEEK_INDICES;
    static constexpr std::array<int, 8> RIGHT_CHEEK_INDICES;
    static constexpr std::array<int, 8> CHIN_CENTER_INDICES;
    static constexpr std::array<int, 5> LEFT_JAW_INDICES;
    static constexpr std::array<int, 5> RIGHT_JAW_INDICES;
    static constexpr int NOSE_TIP_INDEX = 4;

    // Effect parameters
    static constexpr float MAX_SLIM_FACE_DX = 0.03f;  // 3% of face width
    static constexpr float MAX_VLINE_DX = 0.02f;      // 2% of face width
    static constexpr float MAX_VLINE_DY = 0.025f;     // 2.5% upward

    bool applyWarp(GridMesh& mesh, const IrisLandmark* face_mesh, const WarpConfig& config);
    static float calculateFaceWidth(const IrisLandmark* face_mesh);
    static float calculateFaceCenterX(const IrisLandmark* face_mesh);

private:
    void applySlimFace(GridMesh& mesh, const IrisLandmark* face_mesh,
                       float strength, float face_width, float center_x);
    void applyThinChin(GridMesh& mesh, const IrisLandmark* face_mesh,
                       float strength, float face_width, float center_x);
    void applyEnlargeEyes(GridMesh& mesh, const IrisLandmark* face_mesh, float strength);
    void registerWarpControlPoints(GridMesh& mesh, const IrisLandmark* face_mesh);

    static float calculateYWeight(float y, float min_y, float max_y);
    static float calculateDistanceFalloff(float distance, float max_distance);
};

} // namespace warp
} // namespace iris_sdk
```

### 3.2 GridMesh 확장

**추가된 메서드**: `GridMesh::addControlPoints()`

Face Warp 효과를 위해 기존 `ControlLandmarks`에 정의되지 않은 랜드마크를
추가 컨트롤 포인트로 등록하는 기능 추가.

```cpp
int GridMesh::addControlPoints(const IrisLandmark* face_mesh,
                               const int* landmark_indices,
                               int count);
```

---

## 4. 알고리즘 상세

### 4.1 Slim Face 알고리즘

1. **얼굴 중심 계산**: 코 끝 (index 4) 기준
2. **얼굴 너비 계산**: 좌우 외곽 볼 (234, 454) 거리
3. **좌측 볼**: 우측(내측)으로 이동
   - 최대 변위: face_width * 3% * strength
   - Y 위치 가중치: 볼 중앙 최대, 상하단 감소 (bell curve)
   - 중심 거리 가중치: 중심에서 멀수록 감소 (cosine falloff)
4. **우측 볼**: 좌측(내측)으로 이동 (대칭)

### 4.2 V-Line / Thin Chin 알고리즘

1. **턱 영역 Y 범위 계산**: 모든 턱/턱선 랜드마크의 min/max Y
2. **좌측 턱선**: 우측+상단 이동
   - X 변위: face_width * 2% * strength * y_weight
   - Y 변위: -2.5% * strength * y_weight (상단 이동)
   - y_weight: 아래쪽(y 큰 값)일수록 더 많이 이동
3. **우측 턱선**: 좌측+상단 이동 (대칭)
4. **턱 중심**: 상단으로만 이동 + 약간 내측 이동

### 4.3 RBF 보간

컨트롤 포인트의 변위가 설정되면 GridMesh의 `interpolateDisplacements()`가
Gaussian RBF 함수를 사용하여 모든 정점의 변위를 보간합니다.

```cpp
// Gaussian RBF 가중치 함수
weight = exp(-distance^2 / (2 * sigma^2))
```

---

## 5. 단위 테스트

**파일**: `cpp/tests/test_face_warp_controller.cpp`

### 5.1 테스트 케이스 (23개)

| 테스트 | 설명 | 결과 |
|--------|------|------|
| ApplyWarpWithNullMeshFails | 초기화 안된 mesh 실패 | ✅ |
| ApplyWarpWithNullLandmarksFails | null 랜드마크 실패 | ✅ |
| ZeroStrengthNoChange | 강도 0이면 변위 없음 | ✅ |
| ConfigClampValues | 설정값 클램핑 확인 | ✅ |
| ConfigHasActiveEffect | 활성 효과 체크 | ✅ |
| SlimFaceMovesLeftCheekRight | 좌측 볼 우측 이동 | ✅ |
| SlimFaceMovesRightCheekLeft | 우측 볼 좌측 이동 | ✅ |
| SlimFaceSymmetric | 좌우 대칭 변위 | ✅ |
| ThinChinMovesJawInward | 턱선 내측 이동 | ✅ |
| ThinChinMovesChinUp | 턱 상단 이동 | ✅ |
| ThinChinJawMovesUp | 턱선 상단 이동 | ✅ |
| CombinedEffectsStack | 복합 효과 적용 | ✅ |
| StrengthScalesLinearly | 강도 선형 스케일 | ✅ |
| ThinChinStrengthScales | V-라인 강도 스케일 | ✅ |
| SlimFaceGradientFalloff | 슬림 그라데이션 | ✅ |
| VLineYGradient | V-라인 Y 그라데이션 | ✅ |
| CalculateFaceWidth | 얼굴 너비 계산 | ✅ |
| CalculateFaceCenterX | 얼굴 중심 계산 | ✅ |
| CalculateFaceWidthWithNull | null 처리 | ✅ |
| CalculateFaceCenterXWithNull | null 처리 | ✅ |
| MaxStrengthStaysWithinBounds | 최대 강도 범위 확인 | ✅ |
| RepeatedApplyResetsDisplacements | 반복 적용시 리셋 | ✅ |
| ApplyWarpPerformance | 성능 테스트 (100회 74ms) | ✅ |

---

## 6. 완료 기준

- [x] Slim Face 효과 구현
- [x] V-Line (Thin Chin) 효과 구현
- [x] 자연스러운 그라데이션 변위 (Y-weight, distance falloff)
- [x] GridMesh 컨트롤 포인트 확장 (addControlPoints)
- [x] 단위 테스트 23개 100% 통과

---

## 7. 변경된 파일

### 신규 파일
- `cpp/include/iris_sdk/warp/face_warp_controller.h`: FaceWarpController 헤더
- `cpp/src/warp/face_warp_controller.cpp`: 구현 파일
- `cpp/tests/test_face_warp_controller.cpp`: 단위 테스트 (23개)

### 수정된 파일
- `cpp/include/iris_sdk/warp/grid_mesh.h`: addControlPoints() 선언 추가
- `cpp/src/warp/grid_mesh.cpp`: addControlPoints() 구현 추가
- `cpp/CMakeLists.txt`: face_warp_controller 소스 추가
- `cpp/tests/CMakeLists.txt`: test_face_warp_controller 추가

---

## 8. 다음 작업

- **P2-W4-03**: Eye Enlargement 효과 구현

---

## 9. 실행 내역

### 2026-01-29: 구현 완료

**구현 내용**:
1. FaceWarpController 클래스 구현
   - WarpConfig 구조체 (slimFace, thinChin, enlargeEyes)
   - 랜드마크 인덱스 상수 정의
   - applyWarp() 메인 함수
   - applySlimFace(), applyThinChin() 효과 함수
   - registerWarpControlPoints() 컨트롤 포인트 등록
   - 헬퍼 함수 (calculateYWeight, calculateDistanceFalloff)

2. GridMesh 확장
   - addControlPoints() 메서드 추가 (Face Warp용 추가 컨트롤 포인트 등록)

3. 단위 테스트 23개 작성 및 통과

**빌드 및 테스트 결과**:
```
[==========] Running 23 tests from 1 test suite.
[  PASSED  ] 23 tests.
```

**성능**:
- 100회 warp 적용: 74ms (평균 0.74ms/회)
- 30fps 실시간 처리에 적합

**기존 테스트 영향**:
- test_grid_mesh: 31개 테스트 모두 통과 (회귀 없음)
