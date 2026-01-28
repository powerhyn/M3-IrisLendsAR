/**
 * @file grid_mesh.cpp
 * @brief Face Warp용 Grid Mesh 클래스 구현
 */

#include "iris_sdk/warp/grid_mesh.h"
#include <cmath>
#include <algorithm>
#include <limits>

namespace iris_sdk {
namespace warp {

// ControlLandmarks 정적 멤버 정의 (C++17 이전 호환)
constexpr int ControlLandmarks::FACE_CONTOUR[];
constexpr int ControlLandmarks::EYES[];
constexpr int ControlLandmarks::NOSE[];
constexpr int ControlLandmarks::MOUTH[];
constexpr int ControlLandmarks::JAW[];

std::vector<int> ControlLandmarks::getAllIndices() {
    std::vector<int> indices;
    indices.reserve(TOTAL_COUNT);

    // 얼굴 외곽
    for (int i = 0; i < FACE_CONTOUR_COUNT; ++i) {
        indices.push_back(FACE_CONTOUR[i]);
    }
    // 눈
    for (int i = 0; i < EYES_COUNT; ++i) {
        indices.push_back(EYES[i]);
    }
    // 코
    for (int i = 0; i < NOSE_COUNT; ++i) {
        indices.push_back(NOSE[i]);
    }
    // 입
    for (int i = 0; i < MOUTH_COUNT; ++i) {
        indices.push_back(MOUTH[i]);
    }
    // 턱
    for (int i = 0; i < JAW_COUNT; ++i) {
        indices.push_back(JAW[i]);
    }

    return indices;
}

// =============================================================================
// GridMesh 구현
// =============================================================================

GridMesh::GridMesh()
    : grid_size_(0)
    , rbf_sigma_(DEFAULT_RBF_SIGMA)
    , initialized_(false) {
    face_rect_ = {0.0f, 0.0f, 0.0f, 0.0f};
}

GridMesh::~GridMesh() = default;

bool GridMesh::initialize(int grid_size, const Rect& face_rect) {
    if (grid_size < 2 || grid_size > 100) {
        return false;
    }

    // ROI 유효성 검사
    if (face_rect.width <= 0.0f || face_rect.height <= 0.0f) {
        return false;
    }

    grid_size_ = grid_size;
    face_rect_ = face_rect;

    // 그리드 생성
    createGridVertices();
    createTriangleIndices();

    // 랜드마크-정점 매핑 테이블 초기화
    landmark_to_vertex_.clear();
    landmark_to_vertex_.resize(468, -1);  // MediaPipe Face Mesh = 468 landmarks

    initialized_ = true;
    return true;
}

void GridMesh::createGridVertices() {
    const int vertex_count = (grid_size_ + 1) * (grid_size_ + 1);
    vertices_.clear();
    vertices_.reserve(vertex_count);

    // ROI 경계
    const float roi_left = face_rect_.x;
    const float roi_top = face_rect_.y;
    const float roi_width = face_rect_.width;
    const float roi_height = face_rect_.height;

    // 균일 그리드 생성
    for (int row = 0; row <= grid_size_; ++row) {
        for (int col = 0; col <= grid_size_; ++col) {
            GridVertex vertex;

            // 텍스처 좌표 (ROI 내부 정규화 0~1)
            vertex.u = static_cast<float>(col) / grid_size_;
            vertex.v = static_cast<float>(row) / grid_size_;

            // 이미지 좌표 (전체 이미지 기준 정규화)
            vertex.x = roi_left + vertex.u * roi_width;
            vertex.y = roi_top + vertex.v * roi_height;

            // 변위 초기화
            vertex.dx = 0.0f;
            vertex.dy = 0.0f;

            // 컨트롤 포인트는 나중에 설정
            vertex.is_control = false;
            vertex.landmark_idx = -1;

            vertices_.push_back(vertex);
        }
    }
}

void GridMesh::createTriangleIndices() {
    // 삼각형 개수: grid_size * grid_size * 2
    const int triangle_count = grid_size_ * grid_size_ * 2;
    indices_.clear();
    indices_.reserve(triangle_count * 3);

    const int cols = grid_size_ + 1;

    for (int row = 0; row < grid_size_; ++row) {
        for (int col = 0; col < grid_size_; ++col) {
            // 사각형의 4개 정점 인덱스
            const uint16_t top_left = static_cast<uint16_t>(row * cols + col);
            const uint16_t top_right = top_left + 1;
            const uint16_t bottom_left = static_cast<uint16_t>((row + 1) * cols + col);
            const uint16_t bottom_right = bottom_left + 1;

            // 첫 번째 삼각형 (왼쪽 위)
            indices_.push_back(top_left);
            indices_.push_back(bottom_left);
            indices_.push_back(top_right);

            // 두 번째 삼각형 (오른쪽 아래)
            indices_.push_back(top_right);
            indices_.push_back(bottom_left);
            indices_.push_back(bottom_right);
        }
    }
}

bool GridMesh::setControlPoints(const IrisLandmark* face_mesh,
                                 int landmark_count,
                                 int image_width,
                                 int image_height) {
    if (!initialized_ || face_mesh == nullptr || landmark_count < 468) {
        return false;
    }

    if (image_width <= 0 || image_height <= 0) {
        return false;
    }

    // 컨트롤 랜드마크 인덱스 가져오기
    auto control_indices = ControlLandmarks::getAllIndices();

    // 이전 컨트롤 포인트 초기화
    for (auto& vertex : vertices_) {
        vertex.is_control = false;
        vertex.landmark_idx = -1;
    }
    std::fill(landmark_to_vertex_.begin(), landmark_to_vertex_.end(), -1);

    // 각 컨트롤 랜드마크에 대해 최근접 정점 찾기
    for (int lm_idx : control_indices) {
        if (lm_idx < 0 || lm_idx >= landmark_count) {
            continue;
        }

        const IrisLandmark& lm = face_mesh[lm_idx];

        // 랜드마크 좌표 (이미 정규화되어 있다고 가정)
        // MediaPipe 랜드마크는 0~1 정규화 좌표
        float lm_x = lm.x;
        float lm_y = lm.y;

        // ROI 내부에 있는지 확인
        if (lm_x < face_rect_.x || lm_x > face_rect_.x + face_rect_.width ||
            lm_y < face_rect_.y || lm_y > face_rect_.y + face_rect_.height) {
            continue;
        }

        // 최근접 정점 찾기
        int nearest_idx = findNearestVertex(lm_x, lm_y);
        if (nearest_idx >= 0 && nearest_idx < static_cast<int>(vertices_.size())) {
            GridVertex& vertex = vertices_[nearest_idx];
            vertex.is_control = true;
            vertex.landmark_idx = lm_idx;

            // 정점 위치를 랜드마크 위치로 업데이트 (정밀도 향상)
            vertex.x = lm_x;
            vertex.y = lm_y;

            // 매핑 테이블 업데이트
            landmark_to_vertex_[lm_idx] = nearest_idx;
        }
    }

    return true;
}

int GridMesh::findNearestVertex(float lm_x, float lm_y) const {
    int nearest_idx = -1;
    float min_dist_sq = std::numeric_limits<float>::max();

    for (int i = 0; i < static_cast<int>(vertices_.size()); ++i) {
        const GridVertex& v = vertices_[i];
        float dx = v.x - lm_x;
        float dy = v.y - lm_y;
        float dist_sq = dx * dx + dy * dy;

        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            nearest_idx = i;
        }
    }

    return nearest_idx;
}

void GridMesh::resetDisplacements() {
    for (auto& vertex : vertices_) {
        vertex.dx = 0.0f;
        vertex.dy = 0.0f;
    }
}

bool GridMesh::setControlPointDisplacement(int landmark_idx, float dx, float dy) {
    if (landmark_idx < 0 || landmark_idx >= static_cast<int>(landmark_to_vertex_.size())) {
        return false;
    }

    int vertex_idx = landmark_to_vertex_[landmark_idx];
    if (vertex_idx < 0) {
        return false;  // 이 랜드마크는 컨트롤 포인트가 아님
    }

    vertices_[vertex_idx].dx = dx;
    vertices_[vertex_idx].dy = dy;

    return true;
}

void GridMesh::interpolateDisplacements() {
    // 컨트롤 포인트 수집
    std::vector<int> control_indices;
    for (int i = 0; i < static_cast<int>(vertices_.size()); ++i) {
        if (vertices_[i].is_control) {
            control_indices.push_back(i);
        }
    }

    if (control_indices.empty()) {
        return;  // 컨트롤 포인트 없음
    }

    // 비-컨트롤 정점에 대해 RBF 보간
    for (int i = 0; i < static_cast<int>(vertices_.size()); ++i) {
        GridVertex& vertex = vertices_[i];

        if (vertex.is_control) {
            continue;  // 컨트롤 포인트는 이미 변위가 설정됨
        }

        float weight_sum = 0.0f;
        float dx_sum = 0.0f;
        float dy_sum = 0.0f;

        // 모든 컨트롤 포인트로부터 가중 평균
        for (int ctrl_idx : control_indices) {
            const GridVertex& ctrl = vertices_[ctrl_idx];

            // 거리 계산
            float dist_x = vertex.x - ctrl.x;
            float dist_y = vertex.y - ctrl.y;
            float distance = std::sqrt(dist_x * dist_x + dist_y * dist_y);

            // RBF 가중치
            float weight = gaussianRbf(distance);

            weight_sum += weight;
            dx_sum += weight * ctrl.dx;
            dy_sum += weight * ctrl.dy;
        }

        // 가중 평균 적용
        if (weight_sum > 1e-6f) {
            vertex.dx = dx_sum / weight_sum;
            vertex.dy = dy_sum / weight_sum;
        } else {
            vertex.dx = 0.0f;
            vertex.dy = 0.0f;
        }
    }
}

float GridMesh::gaussianRbf(float distance) const {
    // Gaussian RBF: exp(-distance^2 / (2 * sigma^2))
    const float sigma_sq_2 = 2.0f * rbf_sigma_ * rbf_sigma_;
    return std::exp(-(distance * distance) / sigma_sq_2);
}

void GridMesh::computeFinalPositions() {
    const int vertex_count = static_cast<int>(vertices_.size());
    final_positions_.clear();
    final_positions_.reserve(vertex_count * 4);  // x, y, u, v per vertex

    for (const auto& vertex : vertices_) {
        // 최종 위치 = 원본 + 변위
        final_positions_.push_back(vertex.x + vertex.dx);
        final_positions_.push_back(vertex.y + vertex.dy);
        final_positions_.push_back(vertex.u);
        final_positions_.push_back(vertex.v);
    }
}

std::vector<float> GridMesh::getVertexBuffer() const {
    if (!final_positions_.empty()) {
        return final_positions_;
    }

    // final_positions_가 비어있으면 현재 상태로 버퍼 생성
    std::vector<float> buffer;
    buffer.reserve(vertices_.size() * 4);

    for (const auto& vertex : vertices_) {
        buffer.push_back(vertex.x + vertex.dx);
        buffer.push_back(vertex.y + vertex.dy);
        buffer.push_back(vertex.u);
        buffer.push_back(vertex.v);
    }

    return buffer;
}

const std::vector<uint16_t>& GridMesh::getIndices() const {
    return indices_;
}

} // namespace warp
} // namespace iris_sdk
