# P2-W4-03. Eye Enlargement 효과 구현

## 작업 정보

| 항목 | 내용 |
|------|------|
| **작업 ID** | P2-W4-03 |
| **Phase** | Phase 4: Face Warp 구현 |
| **상태** | ⏳ 대기 |
| **예상 기간** | 2일 |
| **의존성** | P2-W4-01 (Grid Mesh) |
| **담당** | systems-programming:cpp-pro |

---

## 1. 목표

눈 영역 랜드마크 기반 자연스러운 눈 확대 효과 구현

### 핵심 산출물
- 눈 중심 기준 방사형 확대
- 눈꺼풀/눈썹 영역 자연스러운 변형
- 좌우 대칭 처리

---

## 2. 랜드마크 분석

### 2.1 눈 영역 랜드마크

```
왼쪽 눈 (LEFT_EYE):
        105 (눈썹 상단)
         │
   70───33───────133───243
   │     │         │     │
   63──159───145──153───154
         │
        168 (눈 아래)

오른쪽 눈 (RIGHT_EYE):
        334 (눈썹 상단)
         │
  300──362───────263───466
   │     │         │     │
  293──386───374──380───381
         │
        397 (눈 아래)
```

### 2.2 눈 확대 전략

```
확대 전:                확대 후:
   ┌───────┐            ┌─────────┐
   │  eye  │   ───►     │   eye   │
   └───────┘            └─────────┘

방사형 확대: 중심에서 외곽으로 밀어냄
- 눈 내부: 확대 비율 적용
- 눈 외곽: 부드럽게 감소하는 변위
- 눈썹 영역: 미세한 상향 조정
```

---

## 3. 상세 구현

### 3.1 Eye Enlargement 구현

**파일**: `cpp/src/warp/face_warp_controller.cpp` (확장)

```cpp
// 눈 영역 랜드마크 인덱스
namespace {

// 왼쪽 눈
const int LEFT_EYE_CENTER = 468;  // 홍채 중심
const int LEFT_EYE_CONTOUR[] = {
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    173, 157, 158, 159, 160, 161, 246
};
const int LEFT_EYE_CONTOUR_COUNT = 16;

// 왼쪽 눈 주변 (확장 영역)
const int LEFT_EYE_OUTER[] = {
    70, 63, 105, 66, 107, 55, 65, 52, 53, 46
};
const int LEFT_EYE_OUTER_COUNT = 10;

// 왼쪽 눈썹
const int LEFT_EYEBROW[] = {
    70, 63, 105, 66, 107, 55, 65, 52, 53, 46
};
const int LEFT_EYEBROW_COUNT = 10;

// 오른쪽 눈
const int RIGHT_EYE_CENTER = 473;  // 홍채 중심
const int RIGHT_EYE_CONTOUR[] = {
    362, 382, 381, 380, 374, 373, 390, 249, 263,
    466, 388, 387, 386, 385, 384, 398
};
const int RIGHT_EYE_CONTOUR_COUNT = 16;

// 오른쪽 눈 주변
const int RIGHT_EYE_OUTER[] = {
    300, 293, 334, 296, 336, 285, 295, 282, 283, 276
};
const int RIGHT_EYE_OUTER_COUNT = 10;

} // namespace

void FaceWarpController::applyEnlargeEyes(
    GridMesh& mesh,
    const IrisLandmark* face_mesh,
    float strength) {

    // 최대 확대 비율 (중심에서)
    float max_scale = 1.0f + strength * 0.25f;  // 최대 25% 확대

    // 왼쪽 눈 처리
    applyEyeEnlargement(
        mesh, face_mesh,
        LEFT_EYE_CENTER,
        LEFT_EYE_CONTOUR, LEFT_EYE_CONTOUR_COUNT,
        LEFT_EYE_OUTER, LEFT_EYE_OUTER_COUNT,
        LEFT_EYEBROW, LEFT_EYEBROW_COUNT,
        max_scale
    );

    // 오른쪽 눈 처리
    applyEyeEnlargement(
        mesh, face_mesh,
        RIGHT_EYE_CENTER,
        RIGHT_EYE_CONTOUR, RIGHT_EYE_CONTOUR_COUNT,
        RIGHT_EYE_OUTER, RIGHT_EYE_OUTER_COUNT,
        nullptr, 0,  // 오른쪽 눈썹은 별도 정의 필요
        max_scale
    );
}

void FaceWarpController::applyEyeEnlargement(
    GridMesh& mesh,
    const IrisLandmark* face_mesh,
    int center_idx,
    const int* contour_indices, int contour_count,
    const int* outer_indices, int outer_count,
    const int* eyebrow_indices, int eyebrow_count,
    float scale) {

    // 눈 중심 좌표
    float center_x = face_mesh[center_idx].x;
    float center_y = face_mesh[center_idx].y;

    // 눈 크기 추정 (외곽 점들로부터)
    float max_dist = 0.0f;
    for (int i = 0; i < contour_count; ++i) {
        int idx = contour_indices[i];
        float dx = face_mesh[idx].x - center_x;
        float dy = face_mesh[idx].y - center_y;
        float dist = std::sqrt(dx * dx + dy * dy);
        max_dist = std::max(max_dist, dist);
    }

    float eye_radius = max_dist * 1.2f;  // 약간 여유 있게

    // 눈 윤곽 랜드마크: 방사형 확대
    for (int i = 0; i < contour_count; ++i) {
        int idx = contour_indices[i];
        float lm_x = face_mesh[idx].x;
        float lm_y = face_mesh[idx].y;

        // 중심으로부터의 벡터
        float vec_x = lm_x - center_x;
        float vec_y = lm_y - center_y;
        float dist = std::sqrt(vec_x * vec_x + vec_y * vec_y);

        if (dist < 1e-6f) continue;

        // 정규화
        float norm_x = vec_x / dist;
        float norm_y = vec_y / dist;

        // 확대 변위 (중심에서 멀어지는 방향)
        float displacement = dist * (scale - 1.0f);

        float dx = norm_x * displacement;
        float dy = norm_y * displacement;

        mesh.setControlPointDisplacement(idx, dx, dy);
    }

    // 외곽 영역: 감쇠된 변위
    for (int i = 0; i < outer_count; ++i) {
        int idx = outer_indices[i];
        float lm_x = face_mesh[idx].x;
        float lm_y = face_mesh[idx].y;

        float vec_x = lm_x - center_x;
        float vec_y = lm_y - center_y;
        float dist = std::sqrt(vec_x * vec_x + vec_y * vec_y);

        if (dist < 1e-6f) continue;

        // 감쇠 계수 (거리에 따라 감소)
        float attenuation = 1.0f - std::min(1.0f, dist / (eye_radius * 2.0f));
        attenuation = attenuation * attenuation;  // 더 부드럽게

        float norm_x = vec_x / dist;
        float norm_y = vec_y / dist;

        float displacement = dist * (scale - 1.0f) * attenuation * 0.5f;

        float dx = norm_x * displacement;
        float dy = norm_y * displacement;

        mesh.setControlPointDisplacement(idx, dx, dy);
    }

    // 눈썹: 약간 상향 (눈 확대에 맞춰)
    if (eyebrow_indices && eyebrow_count > 0) {
        float eyebrow_lift = (scale - 1.0f) * 0.3f * eye_radius;

        for (int i = 0; i < eyebrow_count; ++i) {
            int idx = eyebrow_indices[i];

            // 현재 변위에 추가
            float current_dx = 0.0f, current_dy = 0.0f;
            // (기존 변위 가져오기 필요)

            mesh.setControlPointDisplacement(idx, current_dx, current_dy - eyebrow_lift);
        }
    }
}
```

### 3.2 비선형 확대 (Magnification)

더 자연스러운 결과를 위한 비선형 확대 함수:

```cpp
/**
 * @brief 비선형 확대 함수
 *
 * 중심에서 멀어질수록 확대 비율이 감소
 *
 * @param dist 중심으로부터 거리
 * @param radius 영향 반경
 * @param strength 최대 강도
 * @return 확대 비율 (1.0 = 변화 없음)
 */
float computeMagnification(float dist, float radius, float strength) {
    if (dist > radius * 2.0f) {
        return 1.0f;  // 영향 없음
    }

    float normalized = dist / radius;

    if (normalized < 1.0f) {
        // 눈 내부: 전체 확대
        return 1.0f + strength;
    } else {
        // 외곽: 부드럽게 감소
        float t = (normalized - 1.0f);  // 0 ~ 1
        float ease = 1.0f - t * t;      // ease-out
        return 1.0f + strength * ease;
    }
}
```

### 3.3 GPU 셰이더 기반 확대 (대안)

메시 변형 대신 셰이더로 구현하는 방법:

**파일**: `cpp/src/gpu/shaders/eye_magnify.frag`

```glsl
#version 310 es
precision highp float;

in vec2 v_TexCoord;
out vec4 fragColor;

uniform sampler2D u_Texture;
uniform vec2 u_LeftEyeCenter;   // 왼쪽 눈 중심 (정규화 좌표)
uniform vec2 u_RightEyeCenter;  // 오른쪽 눈 중심
uniform float u_EyeRadius;      // 눈 반경 (정규화)
uniform float u_Strength;       // 확대 강도

vec2 applyMagnification(vec2 uv, vec2 center, float radius, float strength) {
    vec2 delta = uv - center;
    float dist = length(delta);

    if (dist > radius * 2.0) {
        return uv;
    }

    float normalized = dist / radius;
    float magnification;

    if (normalized < 1.0) {
        // 눈 내부: 축소된 좌표로 매핑 (확대 효과)
        magnification = 1.0 / (1.0 + strength);
    } else {
        // 외곽: 부드럽게 전환
        float t = normalized - 1.0;
        float ease = 1.0 - t * t;
        magnification = 1.0 / (1.0 + strength * ease);
    }

    return center + delta * magnification;
}

void main() {
    vec2 uv = v_TexCoord;

    // 왼쪽 눈 확대
    uv = applyMagnification(uv, u_LeftEyeCenter, u_EyeRadius, u_Strength);

    // 오른쪽 눈 확대
    uv = applyMagnification(uv, u_RightEyeCenter, u_EyeRadius, u_Strength);

    fragColor = texture(u_Texture, uv);
}
```

---

## 4. 통합 테스트

### 4.1 전체 Face Warp 파이프라인

```cpp
void processFrame(
    cv::Mat& frame,
    const IrisLandmark* face_mesh,
    const BeautyFilterConfigV2& config) {

    // 1. Grid Mesh 초기화
    GridMesh mesh;
    Rect face_rect = computeFaceBoundingBox(face_mesh);
    mesh.initialize(20, face_rect);
    mesh.setControlPoints(face_mesh, frame.cols, frame.rows);

    // 2. Face Warp 효과 적용
    FaceWarpController warp_controller;
    FaceWarpController::WarpConfig warp_config;
    warp_config.slimFace = config.slimFace;
    warp_config.thinChin = config.thinChin;
    warp_config.enlargeEyes = config.enlargeEyes;

    warp_controller.applyWarp(mesh, face_mesh, warp_config);

    // 3. GPU 렌더링
    GPUMeshRenderer renderer;
    // ...
}
```

---

## 5. 단위 테스트

**파일**: `cpp/tests/test_eye_enlargement.cpp`

```cpp
TEST(EyeEnlargement, CenteredExpansion) {
    GridMesh mesh;
    Rect face_rect{0.0f, 0.0f, 1.0f, 1.0f};
    mesh.initialize(20, face_rect);

    IrisLandmark face_mesh[478];
    // 눈 중심을 (0.3, 0.4)로 설정
    face_mesh[468].x = 0.3f;
    face_mesh[468].y = 0.4f;
    // 눈 윤곽 설정...

    mesh.setControlPoints(face_mesh, 100, 100);

    FaceWarpController controller;
    FaceWarpController::WarpConfig config;
    config.enlargeEyes = 0.5f;

    controller.applyWarp(mesh, face_mesh, config);

    // 눈 윤곽 점들이 중심에서 멀어졌는지 확인
    const auto& vertices = mesh.getVertices();
    for (const auto& v : vertices) {
        if (v.landmark_idx == LEFT_EYE_CONTOUR[0]) {
            float orig_dist = std::sqrt(
                (0.3f - face_mesh[v.landmark_idx].x) *
                (0.3f - face_mesh[v.landmark_idx].x) +
                (0.4f - face_mesh[v.landmark_idx].y) *
                (0.4f - face_mesh[v.landmark_idx].y)
            );
            float new_dist = std::sqrt(
                (0.3f - (v.x + v.dx)) * (0.3f - (v.x + v.dx)) +
                (0.4f - (v.y + v.dy)) * (0.4f - (v.y + v.dy))
            );

            EXPECT_GT(new_dist, orig_dist);  // 확대됨
            break;
        }
    }
}

TEST(EyeEnlargement, SymmetricLeftRight) {
    // 좌우 대칭 확인
}

TEST(EyeEnlargement, SmoothFalloff) {
    // 외곽으로 갈수록 변위 감소 확인
}
```

---

## 6. 완료 기준

- [ ] 눈 중심 기준 방사형 확대 구현
- [ ] 비선형 확대 함수
- [ ] 외곽 영역 부드러운 감쇠
- [ ] 눈썹 영역 연동 조정
- [ ] GPU 셰이더 대안 구현 (선택적)
- [ ] 좌우 대칭 검증
- [ ] 단위 테스트 통과

---

## 7. 다음 작업

- **P2-W5-01**: JNI 바인딩 및 Android 통합
