      # P2-W4-01. Face Warp - Grid Mesh 기반 구현

## 작업 정보

| 항목 | 내용 |
|------|------|
| **작업 ID** | P2-W4-01 |
| **Phase** | Phase 4: Face Warp 구현 |
| **상태** | ✅ 완료 |
| **예상 기간** | 3일 |
| **의존성** | P2-W3-01, P2-W3-02 |
| **담당** | systems-programming:cpp-pro |

---

## 1. 목표

Face Mesh 478 랜드마크 기반 Grid Mesh 변형으로 자연스러운 얼굴 워핑 구현

### 핵심 산출물
- Grid Mesh 생성 시스템 (20x20 기본)
- 랜드마크 → Grid 컨트롤 포인트 매핑
- GPU 기반 메시 렌더링
- V-line, 눈 확대 효과 기반 인프라

### 왜 Grid Mesh인가?

| 방식 | 장점 | 단점 |
|------|------|------|
| 478 삼각형 직접 | 정밀한 제어 | 불연속적, 아티팩트 발생 |
| Grid Mesh (20x20) | 자연스러운 변형, GPU 효율적 | 약간 덜 정밀 |
| Thin Plate Spline | 매우 자연스러움 | 계산 비용 높음 |

**결론**: Grid Mesh는 품질과 성능의 최적 균형점

---

## 2. 아키텍처

### 2.1 Face Warp 파이프라인

```
Face Mesh 478 랜드마크
    │
    ▼
┌─────────────────────────────────┐
│     GridMeshGenerator           │
│  - 20x20 균일 그리드 생성        │
│  - 랜드마크 → 컨트롤 포인트 매핑  │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│     WarpController              │
│  - slimFace, enlargeEyes 적용   │
│  - 컨트롤 포인트 변위 계산       │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│     MeshInterpolator            │
│  - RBF/Bilinear 보간            │
│  - 전체 그리드 정점 변위 계산    │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│     GPUMeshRenderer             │
│  - 텍스처 매핑 + 메시 렌더링     │
│  - 삼각형 스트립 최적화          │
└─────────────────────────────────┘
                │
                ▼
         워핑된 이미지
```

### 2.2 Grid Mesh 구조

```
(0,0)─────────────────────────(1,0)
  │    │    │    │    │    │    │
  ├────┼────┼────┼────┼────┼────┤
  │    │    │ CP │    │    │    │  CP = Control Point (랜드마크 기반)
  ├────┼────┼────┼────┼────┼────┤
  │    │ CP │    │ CP │    │    │
  ├────┼────┼────┼────┼────┼────┤
  │    │    │    │    │    │    │
  ├────┼────┼────┼────┼────┼────┤
  │    │    │ CP │    │    │    │
  ├────┼────┼────┼────┼────┼────┤
(0,1)─────────────────────────(1,1)

Grid Size: 20x20 = 400 vertices
Triangles: 19x19x2 = 722 triangles
```

---

## 2.3 좌표계 정의 (Coordinate System Specification)

> **중요**: 파이프라인 전체에서 좌표계 일관성을 유지해야 함

### 좌표계 종류

| 좌표계 | 범위 | 사용 위치 | 설명 |
|--------|------|-----------|------|
| **정규화 좌표** | 0.0 ~ 1.0 | CPU (GridMesh) | 이미지 기준 상대 좌표 |
| **픽셀 좌표** | 0 ~ width/height | Face Mesh 입력 | 실제 픽셀 위치 |
| **NDC 좌표** | -1.0 ~ 1.0 | GPU (셰이더) | OpenGL 정규화 장치 좌표 |

### 좌표 변환 흐름

```
Face Mesh (픽셀 좌표)
    │
    │  ÷ (width, height)
    ▼
GridMesh CPU (정규화 좌표 0~1)
    │
    │  GridVertex { x, y, dx, dy }
    │  모든 값은 정규화 좌표 (0~1)
    ▼
getVertexBuffer() 출력
    │
    │  [x+dx, y+dy, u, v, ...]
    │  여전히 정규화 좌표
    ▼
mesh_warp.vert 셰이더
    │
    │  ndc = a_Position * 2.0 - 1.0
    │  정규화(0~1) → NDC(-1~1) 변환
    ▼
gl_Position (NDC 좌표)
```

### GridVertex 필드 상세

```cpp
struct GridVertex {
    float x, y;         ///< 원본 위치 (정규화 좌표 0~1, 이미지 기준)
    float u, v;         ///< 텍스처 좌표 (정규화 0~1, 얼굴 ROI 기준)
    float dx, dy;       ///< 변위량 (정규화 좌표 단위, 양수=우측/아래)
    bool is_control;    ///< 컨트롤 포인트 여부
    int landmark_idx;   ///< 매핑된 랜드마크 인덱스 (-1 if none)
};

// 예시: 얼굴 중심이 (0.5, 0.5)에 있고, V-line 효과로 턱을 2% 위로 올리려면
// 해당 랜드마크의 dy = -0.02 (음수 = 위쪽 이동)
```

### 셰이더에서의 변환

```glsl
// mesh_warp.vert
void main() {
    // a_Position은 정규화 좌표 (0~1)로 전달됨
    // GPU에서 NDC로 변환
    vec2 ndc = a_Position * 2.0 - 1.0;  // [0,1] → [-1,1]
    ndc.y = -ndc.y;  // OpenGL Y축 반전 (상단이 +1)

    gl_Position = vec4(ndc, 0.0, 1.0);
    v_TexCoord = a_TexCoord;  // 텍스처 좌표는 변환 없이 전달
}
```

### 주의사항

1. **변위(dx, dy) 단위**: 반드시 정규화 좌표(0~1) 기준으로 계산
   - 픽셀 단위로 계산 후 `/width`, `/height`로 변환 필요

2. **Y축 방향**:
   - CPU(이미지): Y 증가 = 아래쪽
   - OpenGL NDC: Y 증가 = 위쪽
   - 셰이더에서 `ndc.y = -ndc.y`로 보정

3. **ROI 내 좌표 vs 전체 이미지 좌표**:
   - `u, v`: 얼굴 ROI 내부의 정규화 좌표 (텍스처 샘플링용)
   - `x, y`: 전체 이미지 기준 정규화 좌표 (렌더링 위치용)

---

## 3. 상세 구현

### 3.1 GridMesh 클래스

**파일**: `cpp/include/iris_sdk/warp/grid_mesh.h`

```cpp
#ifndef IRIS_SDK_GRID_MESH_H
#define IRIS_SDK_GRID_MESH_H

#include "iris_sdk/types.h"
#include <vector>

namespace iris_sdk {

/**
 * @brief Grid Mesh 정점
 */
struct GridVertex {
    float x, y;         ///< 원본 위치 (정규화 0~1)
    float u, v;         ///< 텍스처 좌표
    float dx, dy;       ///< 변위량
    bool is_control;    ///< 컨트롤 포인트 여부
    int landmark_idx;   ///< 매핑된 랜드마크 인덱스 (-1 if none)
};

/**
 * @brief Face Warp용 Grid Mesh
 *
 * 얼굴 영역에 균일 그리드 생성 및 변형 관리
 */
class GridMesh {
public:
    /**
     * @brief 그리드 초기화
     *
     * @param grid_size 그리드 크기 (20 = 20x20)
     * @param face_rect 얼굴 바운딩 박스 (정규화 좌표)
     */
    void initialize(int grid_size, const Rect& face_rect);

    /**
     * @brief Face Mesh 랜드마크로 컨트롤 포인트 설정
     *
     * @param face_mesh 478개 랜드마크
     * @param width 프레임 너비
     * @param height 프레임 높이
     */
    void setControlPoints(
        const IrisLandmark* face_mesh,
        int width, int height
    );

    /**
     * @brief 변위 초기화 (모든 정점 dx=dy=0)
     */
    void resetDisplacements();

    /**
     * @brief 컨트롤 포인트 변위 설정
     *
     * @param landmark_idx 랜드마크 인덱스
     * @param dx X 변위 (정규화)
     * @param dy Y 변위 (정규화)
     */
    void setControlPointDisplacement(int landmark_idx, float dx, float dy);

    /**
     * @brief 전체 그리드 변위 보간 (RBF)
     *
     * 컨트롤 포인트 변위를 기반으로 전체 그리드 정점 변위 계산
     */
    void interpolateDisplacements();

    /**
     * @brief 최종 위치 계산 (원본 + 변위)
     */
    void computeFinalPositions();

    //=== Getters ===
    const std::vector<GridVertex>& getVertices() const { return vertices_; }
    const std::vector<uint16_t>& getIndices() const { return indices_; }
    int getGridSize() const { return grid_size_; }
    int getVertexCount() const { return static_cast<int>(vertices_.size()); }
    int getIndexCount() const { return static_cast<int>(indices_.size()); }

    /**
     * @brief GPU 버퍼용 정점 데이터 (position + texcoord)
     *
     * @return [x, y, u, v, x, y, u, v, ...]
     */
    std::vector<float> getVertexBuffer() const;

private:
    // 랜드마크 → 그리드 최근접 정점 찾기
    int findNearestVertex(float x, float y) const;

    // RBF (Radial Basis Function) 보간
    float rbfWeight(float dist, float sigma) const;

    std::vector<GridVertex> vertices_;
    std::vector<uint16_t> indices_;
    std::vector<int> control_vertex_indices_;  // 컨트롤 포인트 정점 인덱스

    int grid_size_ = 0;
    Rect face_rect_;
    bool initialized_ = false;
};

} // namespace iris_sdk

#endif // IRIS_SDK_GRID_MESH_H
```

### 3.2 GridMesh 구현

**파일**: `cpp/src/warp/grid_mesh.cpp`

```cpp
#include "iris_sdk/warp/grid_mesh.h"
#include <cmath>
#include <algorithm>

namespace iris_sdk {

// 컨트롤 포인트로 사용할 주요 랜드마크 인덱스
// 얼굴 외곽, 눈, 코, 입 등 특징점
const int CONTROL_LANDMARK_INDICES[] = {
    // 얼굴 외곽 (10개)
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    // 눈 (8개)
    33, 133, 362, 263, 159, 145, 386, 374,
    // 코 (4개)
    1, 4, 168, 6,
    // 입 (6개)
    61, 291, 0, 17, 78, 308,
    // 턱 (4개)
    152, 377, 400, 378
};
const int CONTROL_LANDMARK_COUNT = 32;

void GridMesh::initialize(int grid_size, const Rect& face_rect) {
    grid_size_ = grid_size;
    face_rect_ = face_rect;

    vertices_.clear();
    indices_.clear();
    control_vertex_indices_.clear();

    // 균일 그리드 정점 생성
    float step = 1.0f / (grid_size - 1);

    for (int y = 0; y < grid_size; ++y) {
        for (int x = 0; x < grid_size; ++x) {
            GridVertex v;

            // 얼굴 영역 내 정규화 좌표
            v.u = x * step;
            v.v = y * step;

            // 실제 이미지 좌표로 변환
            v.x = face_rect.x + v.u * face_rect.width;
            v.y = face_rect.y + v.v * face_rect.height;

            v.dx = 0.0f;
            v.dy = 0.0f;
            v.is_control = false;
            v.landmark_idx = -1;

            vertices_.push_back(v);
        }
    }

    // 삼각형 인덱스 생성 (Triangle Strip)
    for (int y = 0; y < grid_size - 1; ++y) {
        for (int x = 0; x < grid_size - 1; ++x) {
            int i0 = y * grid_size + x;
            int i1 = i0 + 1;
            int i2 = i0 + grid_size;
            int i3 = i2 + 1;

            // 두 개의 삼각형
            indices_.push_back(i0);
            indices_.push_back(i2);
            indices_.push_back(i1);

            indices_.push_back(i1);
            indices_.push_back(i2);
            indices_.push_back(i3);
        }
    }

    initialized_ = true;
}

void GridMesh::setControlPoints(
    const IrisLandmark* face_mesh,
    int width, int height) {

    if (!initialized_) return;

    control_vertex_indices_.clear();

    for (int i = 0; i < CONTROL_LANDMARK_COUNT; ++i) {
        int lm_idx = CONTROL_LANDMARK_INDICES[i];

        // 랜드마크 좌표 (정규화 → 이미지)
        float lm_x = face_mesh[lm_idx].x;
        float lm_y = face_mesh[lm_idx].y;

        // 얼굴 영역 내 정규화 좌표
        float local_x = (lm_x - face_rect_.x) / face_rect_.width;
        float local_y = (lm_y - face_rect_.y) / face_rect_.height;

        // 범위 체크
        if (local_x < 0 || local_x > 1 || local_y < 0 || local_y > 1) {
            continue;
        }

        // 최근접 그리드 정점 찾기
        int vertex_idx = findNearestVertex(local_x, local_y);
        if (vertex_idx >= 0) {
            vertices_[vertex_idx].is_control = true;
            vertices_[vertex_idx].landmark_idx = lm_idx;

            // 정확한 랜드마크 위치로 업데이트
            vertices_[vertex_idx].x = lm_x;
            vertices_[vertex_idx].y = lm_y;

            control_vertex_indices_.push_back(vertex_idx);
        }
    }
}

int GridMesh::findNearestVertex(float x, float y) const {
    int nearest = -1;
    float min_dist = std::numeric_limits<float>::max();

    float step = 1.0f / (grid_size_ - 1);

    for (size_t i = 0; i < vertices_.size(); ++i) {
        float vx = (i % grid_size_) * step;
        float vy = (i / grid_size_) * step;

        float dx = vx - x;
        float dy = vy - y;
        float dist = dx * dx + dy * dy;

        if (dist < min_dist) {
            min_dist = dist;
            nearest = static_cast<int>(i);
        }
    }

    return nearest;
}

void GridMesh::resetDisplacements() {
    for (auto& v : vertices_) {
        v.dx = 0.0f;
        v.dy = 0.0f;
    }
}

void GridMesh::setControlPointDisplacement(int landmark_idx, float dx, float dy) {
    for (int vi : control_vertex_indices_) {
        if (vertices_[vi].landmark_idx == landmark_idx) {
            vertices_[vi].dx = dx;
            vertices_[vi].dy = dy;
            break;
        }
    }
}

float GridMesh::rbfWeight(float dist, float sigma) const {
    // Gaussian RBF
    return std::exp(-(dist * dist) / (2.0f * sigma * sigma));
}

void GridMesh::interpolateDisplacements() {
    if (control_vertex_indices_.empty()) return;

    float sigma = 0.15f;  // 영향 범위 (정규화 좌표)

    // 각 정점에 대해 RBF 보간
    for (size_t i = 0; i < vertices_.size(); ++i) {
        if (vertices_[i].is_control) {
            continue;  // 컨트롤 포인트는 이미 변위 설정됨
        }

        float sum_weight = 0.0f;
        float sum_dx = 0.0f;
        float sum_dy = 0.0f;

        for (int ci : control_vertex_indices_) {
            const auto& cp = vertices_[ci];

            float dx = vertices_[i].x - cp.x;
            float dy = vertices_[i].y - cp.y;
            float dist = std::sqrt(dx * dx + dy * dy);

            float weight = rbfWeight(dist, sigma);

            sum_dx += cp.dx * weight;
            sum_dy += cp.dy * weight;
            sum_weight += weight;
        }

        if (sum_weight > 1e-6f) {
            vertices_[i].dx = sum_dx / sum_weight;
            vertices_[i].dy = sum_dy / sum_weight;
        }
    }
}

void GridMesh::computeFinalPositions() {
    // dx, dy가 이미 정규화 좌표 기준이므로 그대로 적용
    // GPU 렌더링 시 최종 위치 = (x + dx, y + dy)
}

std::vector<float> GridMesh::getVertexBuffer() const {
    std::vector<float> buffer;
    buffer.reserve(vertices_.size() * 4);  // x, y, u, v

    for (const auto& v : vertices_) {
        // 최종 위치 (변위 적용)
        buffer.push_back(v.x + v.dx);
        buffer.push_back(v.y + v.dy);
        // 텍스처 좌표 (원본 위치 기준)
        buffer.push_back(v.u);
        buffer.push_back(v.v);
    }

    return buffer;
}

} // namespace iris_sdk
```

### 3.3 GPUMeshRenderer

**파일**: `cpp/include/iris_sdk/warp/gpu_mesh_renderer.h`

```cpp
#ifndef IRIS_SDK_GPU_MESH_RENDERER_H
#define IRIS_SDK_GPU_MESH_RENDERER_H

#if defined(__ANDROID__) && defined(IRIS_SDK_HAS_GLES)

#include "iris_sdk/warp/grid_mesh.h"
#include "iris_sdk/gpu/gles_render_context.h"
#include <GLES3/gl31.h>

namespace iris_sdk {

/**
 * @brief GPU 기반 메시 렌더링
 *
 * Grid Mesh를 텍스처와 함께 렌더링하여 워핑 효과 생성
 */
class GPUMeshRenderer {
public:
    GPUMeshRenderer();
    ~GPUMeshRenderer();

    bool initialize(GLESRenderContext* context);
    void release();

    /**
     * @brief 메시 렌더링
     *
     * @param input_tex 입력 텍스처
     * @param output_fbo 출력 FBO
     * @param mesh 변형된 Grid Mesh
     * @param width 출력 너비
     * @param height 출력 높이
     */
    void render(
        GLuint input_tex,
        GLuint output_fbo,
        const GridMesh& mesh,
        int width, int height
    );

    /**
     * @brief 메시 버퍼 업데이트
     *
     * 매 프레임 변경된 정점 데이터 업로드
     */
    void updateMeshBuffer(const GridMesh& mesh);

private:
    bool initializeShader();
    void setupBuffers(int max_vertices);

    GLESRenderContext* context_ = nullptr;

    GLuint program_ = 0;
    GLuint vao_ = 0;
    GLuint vbo_ = 0;
    GLuint ebo_ = 0;

    int max_vertices_ = 0;
    int current_index_count_ = 0;

    bool initialized_ = false;
};

} // namespace iris_sdk

#endif // __ANDROID__ && IRIS_SDK_HAS_GLES

#endif // IRIS_SDK_GPU_MESH_RENDERER_H
```

### 3.4 메시 렌더링 셰이더

**파일**: `cpp/src/gpu/shaders/mesh_warp.vert`

```glsl
#version 310 es
precision highp float;

layout(location = 0) in vec2 a_Position;  // 변형된 위치
layout(location = 1) in vec2 a_TexCoord;  // 원본 텍스처 좌표

out vec2 v_TexCoord;

void main() {
    // 정규화 좌표 → NDC (-1 ~ 1)
    vec2 ndc = a_Position * 2.0 - 1.0;
    ndc.y = -ndc.y;  // Y축 반전 (OpenGL 좌표계)

    gl_Position = vec4(ndc, 0.0, 1.0);
    v_TexCoord = a_TexCoord;
}
```

**파일**: `cpp/src/gpu/shaders/mesh_warp.frag`

```glsl
#version 310 es
precision highp float;

in vec2 v_TexCoord;
out vec4 fragColor;

uniform sampler2D u_Texture;

void main() {
    fragColor = texture(u_Texture, v_TexCoord);
}
```

### 3.5 GPUMeshRenderer 구현

**파일**: `cpp/src/warp/gpu_mesh_renderer.cpp`

```cpp
#if defined(__ANDROID__) && defined(IRIS_SDK_HAS_GLES)

#include "iris_sdk/warp/gpu_mesh_renderer.h"
#include "iris_sdk/gpu/shader_manager.h"

namespace iris_sdk {

namespace {
const char* MESH_WARP_VERTEX = R"(
#version 310 es
precision highp float;

layout(location = 0) in vec2 a_Position;
layout(location = 1) in vec2 a_TexCoord;

out vec2 v_TexCoord;

void main() {
    vec2 ndc = a_Position * 2.0 - 1.0;
    ndc.y = -ndc.y;
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_TexCoord = a_TexCoord;
}
)";

const char* MESH_WARP_FRAGMENT = R"(
#version 310 es
precision highp float;

in vec2 v_TexCoord;
out vec4 fragColor;

uniform sampler2D u_Texture;

void main() {
    fragColor = texture(u_Texture, v_TexCoord);
}
)";
}

GPUMeshRenderer::GPUMeshRenderer() = default;

GPUMeshRenderer::~GPUMeshRenderer() {
    release();
}

bool GPUMeshRenderer::initialize(GLESRenderContext* context) {
    context_ = context;

    if (!context_->makeCurrent()) {
        return false;
    }

    if (!initializeShader()) {
        return false;
    }

    // 최대 30x30 그리드 지원
    setupBuffers(30 * 30);

    initialized_ = true;
    return true;
}

bool GPUMeshRenderer::initializeShader() {
    GLuint vertex_shader, fragment_shader;

    if (!ShaderManager::compileShader(GL_VERTEX_SHADER, MESH_WARP_VERTEX, vertex_shader)) {
        return false;
    }

    if (!ShaderManager::compileShader(GL_FRAGMENT_SHADER, MESH_WARP_FRAGMENT, fragment_shader)) {
        glDeleteShader(vertex_shader);
        return false;
    }

    if (!ShaderManager::linkProgram(vertex_shader, fragment_shader, program_)) {
        glDeleteShader(vertex_shader);
        glDeleteShader(fragment_shader);
        return false;
    }

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    return true;
}

void GPUMeshRenderer::setupBuffers(int max_vertices) {
    max_vertices_ = max_vertices;

    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_);
    glGenBuffers(1, &ebo_);

    glBindVertexArray(vao_);

    // VBO: position (2) + texcoord (2) = 4 floats per vertex
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, max_vertices * 4 * sizeof(float),
                 nullptr, GL_DYNAMIC_DRAW);

    // Position
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // TexCoord
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                          (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // EBO: 최대 인덱스 수 = (grid-1)^2 * 6
    int max_indices = (30 - 1) * (30 - 1) * 6;
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, max_indices * sizeof(uint16_t),
                 nullptr, GL_DYNAMIC_DRAW);

    glBindVertexArray(0);
}

void GPUMeshRenderer::updateMeshBuffer(const GridMesh& mesh) {
    if (!initialized_) return;

    context_->makeCurrent();

    std::vector<float> vertex_data = mesh.getVertexBuffer();
    const auto& indices = mesh.getIndices();

    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferSubData(GL_ARRAY_BUFFER, 0,
                    vertex_data.size() * sizeof(float), vertex_data.data());

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0,
                    indices.size() * sizeof(uint16_t), indices.data());

    current_index_count_ = static_cast<int>(indices.size());
}

void GPUMeshRenderer::render(
    GLuint input_tex,
    GLuint output_fbo,
    const GridMesh& mesh,
    int width, int height) {

    if (!initialized_) return;

    context_->makeCurrent();

    // 메시 버퍼 업데이트
    updateMeshBuffer(mesh);

    // 렌더링
    glBindFramebuffer(GL_FRAMEBUFFER, output_fbo);
    glViewport(0, 0, width, height);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(program_);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, input_tex);
    glUniform1i(glGetUniformLocation(program_, "u_Texture"), 0);

    glBindVertexArray(vao_);
    glDrawElements(GL_TRIANGLES, current_index_count_, GL_UNSIGNED_SHORT, 0);
    glBindVertexArray(0);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void GPUMeshRenderer::release() {
    if (!initialized_) return;

    if (context_) {
        context_->makeCurrent();
    }

    if (vao_) {
        glDeleteVertexArrays(1, &vao_);
        vao_ = 0;
    }
    if (vbo_) {
        glDeleteBuffers(1, &vbo_);
        vbo_ = 0;
    }
    if (ebo_) {
        glDeleteBuffers(1, &ebo_);
        ebo_ = 0;
    }
    if (program_) {
        glDeleteProgram(program_);
        program_ = 0;
    }

    initialized_ = false;
}

} // namespace iris_sdk

#endif // __ANDROID__ && IRIS_SDK_HAS_GLES
```

---

## 4. 단위 테스트

**파일**: `cpp/tests/test_grid_mesh.cpp`

```cpp
#include <gtest/gtest.h>
#include "iris_sdk/warp/grid_mesh.h"

using namespace iris_sdk;

TEST(GridMesh, InitializesCorrectly) {
    GridMesh mesh;
    Rect face_rect{0.2f, 0.1f, 0.6f, 0.8f};

    mesh.initialize(20, face_rect);

    EXPECT_EQ(mesh.getGridSize(), 20);
    EXPECT_EQ(mesh.getVertexCount(), 400);  // 20x20
    EXPECT_EQ(mesh.getIndexCount(), 19 * 19 * 6);  // (20-1)^2 * 6
}

TEST(GridMesh, ControlPointsSetCorrectly) {
    GridMesh mesh;
    Rect face_rect{0.2f, 0.1f, 0.6f, 0.8f};
    mesh.initialize(20, face_rect);

    // Mock face mesh
    IrisLandmark face_mesh[478];
    for (int i = 0; i < 478; ++i) {
        face_mesh[i].x = 0.5f;
        face_mesh[i].y = 0.5f;
    }

    mesh.setControlPoints(face_mesh, 1920, 1080);

    // 컨트롤 포인트가 설정되었는지 확인
    const auto& vertices = mesh.getVertices();
    int control_count = 0;
    for (const auto& v : vertices) {
        if (v.is_control) control_count++;
    }

    EXPECT_GT(control_count, 0);
}

TEST(GridMesh, DisplacementInterpolation) {
    GridMesh mesh;
    Rect face_rect{0.0f, 0.0f, 1.0f, 1.0f};
    mesh.initialize(5, face_rect);

    // 중앙 정점에 변위 설정
    IrisLandmark face_mesh[478];
    face_mesh[10].x = 0.5f;
    face_mesh[10].y = 0.5f;

    mesh.setControlPoints(face_mesh, 100, 100);
    mesh.setControlPointDisplacement(10, 0.1f, 0.05f);
    mesh.interpolateDisplacements();

    // 인접 정점도 변위가 있어야 함 (RBF 보간)
    const auto& vertices = mesh.getVertices();
    bool found_interpolated = false;
    for (const auto& v : vertices) {
        if (!v.is_control && (v.dx != 0.0f || v.dy != 0.0f)) {
            found_interpolated = true;
            break;
        }
    }

    EXPECT_TRUE(found_interpolated);
}

TEST(GridMesh, VertexBufferFormat) {
    GridMesh mesh;
    Rect face_rect{0.0f, 0.0f, 1.0f, 1.0f};
    mesh.initialize(3, face_rect);

    auto buffer = mesh.getVertexBuffer();

    // 3x3 = 9 vertices, 4 floats each
    EXPECT_EQ(buffer.size(), 9 * 4);
}
```

---

## 5. 완료 기준

- [x] GridMesh 클래스 구현 (2026-01-28)
- [x] 좌표계 명세 문서화 (정규화/픽셀/NDC 변환 규칙)
- [x] 랜드마크 → 컨트롤 포인트 매핑
- [x] RBF 보간 알고리즘
- [ ] GPUMeshRenderer 구현 (Android GLES 환경 필요)
- [ ] 메시 렌더링 셰이더 (좌표 변환 포함)
- [x] 단위 테스트 통과 (31개 테스트, 100% 통과)

---

## 6. 실행 내역

### 2026-01-28: GridMesh CPU 구현 완료

**구현된 파일**:
1. `cpp/include/iris_sdk/warp/grid_mesh.h` - GridMesh 클래스 헤더
2. `cpp/src/warp/grid_mesh.cpp` - GridMesh 구현
3. `cpp/tests/test_grid_mesh.cpp` - 31개 단위 테스트

**주요 기능**:
- `GridVertex` 구조체: x, y, u, v, dx, dy, is_control, landmark_idx
- `ControlLandmarks` 구조체: 32개 컨트롤 랜드마크 인덱스 (얼굴 외곽 10개, 눈 8개, 코 4개, 입 6개, 턱 4개)
- `GridMesh::initialize()`: 균일 그리드 생성 (기본 20x20)
- `GridMesh::setControlPoints()`: Face Mesh 랜드마크 기반 컨트롤 포인트 설정
- `GridMesh::interpolateDisplacements()`: Gaussian RBF 보간 알고리즘
- `GridMesh::getVertexBuffer()`: GPU용 정점 버퍼 [x+dx, y+dy, u, v, ...]

**테스트 결과**:
- 31개 테스트 모두 통과
- 테스트 스위트: GridMeshInitTest, GridMeshGridTest, GridMeshControlPointTest, GridMeshDisplacementTest, GridMeshBufferTest, ControlLandmarksTest, GridMeshRbfTest, GridMeshEdgeCaseTest

**GPUMeshRenderer 상태**:
- Desktop에서는 OpenGL ES 미지원으로 구현 생략
- Android 빌드 시 구현 예정 (P2-W5 JNI 통합 후)

---

## 7. 다음 작업

- **P2-W4-02**: Slim Face / V-Line 효과 구현
- **P2-W4-03**: Eye Enlargement 효과 구현
