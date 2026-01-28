# P2-W4-02. Slim Face / V-Line 효과 구현

## 작업 정보

| 항목 | 내용 |
|------|------|
| **작업 ID** | P2-W4-02 |
| **Phase** | Phase 4: Face Warp 구현 |
| **상태** | ⏳ 대기 |
| **예상 기간** | 2일 |
| **의존성** | P2-W4-01 (Grid Mesh) |
| **담당** | systems-programming:cpp-pro |

---

## 1. 목표

얼굴 외곽 랜드마크 기반 자연스러운 슬림 효과 및 V-라인 구현

### 핵심 산출물
- slimFace: 전체 얼굴 너비 축소
- thinChin: 턱선 V-라인 효과
- 자연스러운 그라데이션 변위

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
| 좌측 볼 | 234, 93, 132, 58, 172 | 우측 (내측) |
| 우측 볼 | 454, 323, 361, 288, 397 | 좌측 (내측) |
| 좌측 턱 | 136, 150, 149, 176, 148 | 우측+상단 |
| 우측 턱 | 365, 379, 378, 400, 377 | 좌측+상단 |
| 턱 끝 | 152, 175, 199, 18, 17 | 상단 |

---

## 3. 상세 구현

### 3.1 FaceWarpController 클래스

**파일**: `cpp/include/iris_sdk/warp/face_warp_controller.h`

```cpp
#ifndef IRIS_SDK_FACE_WARP_CONTROLLER_H
#define IRIS_SDK_FACE_WARP_CONTROLLER_H

#include "iris_sdk/warp/grid_mesh.h"
#include "iris_sdk/types.h"

namespace iris_sdk {

/**
 * @brief 얼굴 변형 효과 컨트롤러
 *
 * slimFace, thinChin, enlargeEyes 등 변형 효과 계산
 */
class FaceWarpController {
public:
    struct WarpConfig {
        float slimFace = 0.0f;      ///< 얼굴 슬림 (0.0 ~ 1.0)
        float thinChin = 0.0f;      ///< V-라인 (0.0 ~ 1.0)
        float enlargeEyes = 0.0f;   ///< 눈 확대 (0.0 ~ 1.0)
    };

    /**
     * @brief 변형 효과 적용
     *
     * @param mesh Grid Mesh (변위 설정됨)
     * @param face_mesh 478개 랜드마크
     * @param config 변형 설정
     */
    void applyWarp(
        GridMesh& mesh,
        const IrisLandmark* face_mesh,
        const WarpConfig& config
    );

private:
    // 슬림 효과 적용
    void applySlimFace(
        GridMesh& mesh,
        const IrisLandmark* face_mesh,
        float strength
    );

    // V-라인 효과 적용
    void applyThinChin(
        GridMesh& mesh,
        const IrisLandmark* face_mesh,
        float strength
    );

    // 눈 확대 효과 적용
    void applyEnlargeEyes(
        GridMesh& mesh,
        const IrisLandmark* face_mesh,
        float strength
    );

    // 변위 계산 유틸리티
    void calculateCheekDisplacement(
        const IrisLandmark* face_mesh,
        float strength,
        std::vector<std::pair<int, std::pair<float, float>>>& displacements
    );

    void calculateChinDisplacement(
        const IrisLandmark* face_mesh,
        float strength,
        std::vector<std::pair<int, std::pair<float, float>>>& displacements
    );
};

} // namespace iris_sdk

#endif // IRIS_SDK_FACE_WARP_CONTROLLER_H
```

### 3.2 Slim Face 구현

**파일**: `cpp/src/warp/face_warp_controller.cpp`

```cpp
#include "iris_sdk/warp/face_warp_controller.h"
#include <cmath>

namespace iris_sdk {

// 볼 영역 랜드마크
const int LEFT_CHEEK_INDICES[] = {234, 93, 132, 58, 172, 136, 150, 149};
const int LEFT_CHEEK_COUNT = 8;

const int RIGHT_CHEEK_INDICES[] = {454, 323, 361, 288, 397, 365, 379, 378};
const int RIGHT_CHEEK_COUNT = 8;

// 턱 영역 랜드마크
const int CHIN_INDICES[] = {152, 175, 199, 18, 17, 200, 421, 418};
const int CHIN_COUNT = 8;

const int LEFT_JAW_INDICES[] = {136, 150, 149, 176, 148, 152};
const int LEFT_JAW_COUNT = 6;

const int RIGHT_JAW_INDICES[] = {365, 379, 378, 400, 377, 152};
const int RIGHT_JAW_COUNT = 6;

void FaceWarpController::applyWarp(
    GridMesh& mesh,
    const IrisLandmark* face_mesh,
    const WarpConfig& config) {

    mesh.resetDisplacements();

    if (config.slimFace > 0.01f) {
        applySlimFace(mesh, face_mesh, config.slimFace);
    }

    if (config.thinChin > 0.01f) {
        applyThinChin(mesh, face_mesh, config.thinChin);
    }

    if (config.enlargeEyes > 0.01f) {
        applyEnlargeEyes(mesh, face_mesh, config.enlargeEyes);
    }

    // 전체 그리드 변위 보간
    mesh.interpolateDisplacements();
    mesh.computeFinalPositions();
}

void FaceWarpController::applySlimFace(
    GridMesh& mesh,
    const IrisLandmark* face_mesh,
    float strength) {

    // 얼굴 중심 계산 (코 끝 기준)
    float center_x = face_mesh[4].x;  // 코 끝
    float center_y = face_mesh[4].y;

    // 최대 변위량 (정규화 좌표)
    float max_displacement = 0.03f * strength;  // 최대 3%

    // 좌측 볼: 우측(내측)으로 이동
    for (int i = 0; i < LEFT_CHEEK_COUNT; ++i) {
        int idx = LEFT_CHEEK_INDICES[i];
        float lm_x = face_mesh[idx].x;
        float lm_y = face_mesh[idx].y;

        // 중심으로부터의 거리에 따른 변위
        float dist_from_center = center_x - lm_x;
        float normalized_dist = std::min(1.0f, dist_from_center * 4.0f);

        // Y 위치에 따른 가중치 (볼 중앙이 가장 많이)
        float y_weight = 1.0f - std::abs(lm_y - 0.5f) * 2.0f;
        y_weight = std::max(0.3f, y_weight);

        float dx = max_displacement * normalized_dist * y_weight;
        mesh.setControlPointDisplacement(idx, dx, 0.0f);
    }

    // 우측 볼: 좌측(내측)으로 이동
    for (int i = 0; i < RIGHT_CHEEK_COUNT; ++i) {
        int idx = RIGHT_CHEEK_INDICES[i];
        float lm_x = face_mesh[idx].x;
        float lm_y = face_mesh[idx].y;

        float dist_from_center = lm_x - center_x;
        float normalized_dist = std::min(1.0f, dist_from_center * 4.0f);

        float y_weight = 1.0f - std::abs(lm_y - 0.5f) * 2.0f;
        y_weight = std::max(0.3f, y_weight);

        float dx = -max_displacement * normalized_dist * y_weight;
        mesh.setControlPointDisplacement(idx, dx, 0.0f);
    }
}

void FaceWarpController::applyThinChin(
    GridMesh& mesh,
    const IrisLandmark* face_mesh,
    float strength) {

    // 턱 끝 위치
    float chin_x = face_mesh[152].x;
    float chin_y = face_mesh[152].y;

    // 최대 변위량
    float max_x_displacement = 0.02f * strength;  // 좌우
    float max_y_displacement = 0.025f * strength; // 상단

    // 좌측 턱: 우측+상단
    for (int i = 0; i < LEFT_JAW_COUNT; ++i) {
        int idx = LEFT_JAW_INDICES[i];
        float lm_x = face_mesh[idx].x;
        float lm_y = face_mesh[idx].y;

        // 턱 끝에 가까울수록 상단 이동 더 많이
        float y_factor = (lm_y - 0.5f) * 2.0f;  // 아래쪽일수록 큰 값
        y_factor = std::max(0.0f, y_factor);

        // 중심으로부터 거리
        float x_factor = (chin_x - lm_x) * 3.0f;
        x_factor = std::min(1.0f, std::max(0.0f, x_factor));

        float dx = max_x_displacement * x_factor;
        float dy = -max_y_displacement * y_factor;

        mesh.setControlPointDisplacement(idx, dx, dy);
    }

    // 우측 턱: 좌측+상단
    for (int i = 0; i < RIGHT_JAW_COUNT; ++i) {
        int idx = RIGHT_JAW_INDICES[i];
        float lm_x = face_mesh[idx].x;
        float lm_y = face_mesh[idx].y;

        float y_factor = (lm_y - 0.5f) * 2.0f;
        y_factor = std::max(0.0f, y_factor);

        float x_factor = (lm_x - chin_x) * 3.0f;
        x_factor = std::min(1.0f, std::max(0.0f, x_factor));

        float dx = -max_x_displacement * x_factor;
        float dy = -max_y_displacement * y_factor;

        mesh.setControlPointDisplacement(idx, dx, dy);
    }

    // 턱 끝: 상단으로만 이동
    for (int i = 0; i < CHIN_COUNT; ++i) {
        int idx = CHIN_INDICES[i];
        float lm_y = face_mesh[idx].y;

        float y_factor = (lm_y - 0.5f) * 2.0f;
        y_factor = std::max(0.0f, y_factor);

        float dy = -max_y_displacement * y_factor * 0.7f;
        mesh.setControlPointDisplacement(idx, 0.0f, dy);
    }
}

} // namespace iris_sdk
```

---

## 4. 시각화 및 디버깅

### 4.1 변위 벡터 시각화

```cpp
void debugVisualize(
    cv::Mat& frame,
    const GridMesh& mesh,
    const IrisLandmark* face_mesh) {

    // 컨트롤 포인트 변위 화살표
    for (const auto& v : mesh.getVertices()) {
        if (!v.is_control) continue;

        int x1 = static_cast<int>(v.x * frame.cols);
        int y1 = static_cast<int>(v.y * frame.rows);
        int x2 = static_cast<int>((v.x + v.dx * 10) * frame.cols);
        int y2 = static_cast<int>((v.y + v.dy * 10) * frame.rows);

        cv::arrowedLine(frame, cv::Point(x1, y1), cv::Point(x2, y2),
                        cv::Scalar(0, 255, 0), 2);
    }
}
```

---

## 5. 단위 테스트

**파일**: `cpp/tests/test_face_warp.cpp`

```cpp
TEST(FaceWarp, SlimFaceMovesInward) {
    GridMesh mesh;
    Rect face_rect{0.2f, 0.1f, 0.6f, 0.8f};
    mesh.initialize(20, face_rect);

    // Mock face mesh
    IrisLandmark face_mesh[478];
    // ... 설정 ...

    mesh.setControlPoints(face_mesh, 1920, 1080);

    FaceWarpController controller;
    FaceWarpController::WarpConfig config;
    config.slimFace = 0.8f;

    controller.applyWarp(mesh, face_mesh, config);

    // 좌측 볼 정점이 우측으로 이동했는지 확인
    // 우측 볼 정점이 좌측으로 이동했는지 확인
}

TEST(FaceWarp, ThinChinMovesUp) {
    // 턱 정점이 상단으로 이동했는지 확인
}

TEST(FaceWarp, CombinedEffects) {
    // slimFace + thinChin 동시 적용
}
```

---

## 6. 완료 기준

- [ ] Slim Face 효과 구현
- [ ] V-Line (Thin Chin) 효과 구현
- [ ] 자연스러운 그라데이션 변위
- [ ] 디버그 시각화
- [ ] 단위 테스트 통과

---

## 7. 다음 작업

- **P2-W4-03**: Eye Enlargement 효과 구현
