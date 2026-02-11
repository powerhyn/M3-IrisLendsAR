/**
 * @file test_grid_mesh.cpp
 * @brief GridMesh 클래스 단위 테스트
 */

#include <gtest/gtest.h>
#include "iris_sdk/warp/grid_mesh.h"
#include <chrono>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace iris_sdk;
using namespace iris_sdk::warp;

// =============================================================================
// 초기화 테스트
// =============================================================================

class GridMeshInitTest : public ::testing::Test {
protected:
    GridMesh mesh;
    Rect face_rect{0.2f, 0.1f, 0.6f, 0.8f};  // 얼굴 ROI
};

TEST_F(GridMeshInitTest, InitializeWithDefaultGridSize) {
    EXPECT_TRUE(mesh.initialize(GridMesh::DEFAULT_GRID_SIZE, face_rect));
    EXPECT_TRUE(mesh.isInitialized());
    EXPECT_EQ(mesh.getGridSize(), GridMesh::DEFAULT_GRID_SIZE);
}

TEST_F(GridMeshInitTest, InitializeWithCustomGridSize) {
    EXPECT_TRUE(mesh.initialize(10, face_rect));
    EXPECT_EQ(mesh.getGridSize(), 10);

    // 정점 개수 확인: (grid_size + 1) ^ 2
    EXPECT_EQ(mesh.getVertexCount(), 11 * 11);
}

TEST_F(GridMeshInitTest, InitializeWithInvalidGridSize) {
    EXPECT_FALSE(mesh.initialize(0, face_rect));
    EXPECT_FALSE(mesh.initialize(1, face_rect));  // 최소 2 이상
    EXPECT_FALSE(mesh.initialize(101, face_rect));  // 최대 100
    EXPECT_FALSE(mesh.isInitialized());
}

TEST_F(GridMeshInitTest, InitializeWithInvalidRect) {
    Rect invalid_rect{0.0f, 0.0f, 0.0f, 0.0f};
    EXPECT_FALSE(mesh.initialize(20, invalid_rect));

    Rect negative_rect{0.0f, 0.0f, -0.5f, 0.5f};
    EXPECT_FALSE(mesh.initialize(20, negative_rect));
}

TEST_F(GridMeshInitTest, FaceRectIsStored) {
    mesh.initialize(20, face_rect);
    const Rect& stored = mesh.getFaceRect();
    EXPECT_FLOAT_EQ(stored.x, face_rect.x);
    EXPECT_FLOAT_EQ(stored.y, face_rect.y);
    EXPECT_FLOAT_EQ(stored.width, face_rect.width);
    EXPECT_FLOAT_EQ(stored.height, face_rect.height);
}

// =============================================================================
// 그리드 생성 테스트
// =============================================================================

class GridMeshGridTest : public ::testing::Test {
protected:
    void SetUp() override {
        face_rect = {0.2f, 0.1f, 0.6f, 0.8f};
        mesh.initialize(4, face_rect);  // 5x5 정점 = 25개
    }

    GridMesh mesh;
    Rect face_rect;
};

TEST_F(GridMeshGridTest, CorrectVertexCount) {
    // 4x4 그리드 = 5x5 정점
    EXPECT_EQ(mesh.getVertexCount(), 25);
}

TEST_F(GridMeshGridTest, CorrectTriangleCount) {
    // 4x4 그리드 = 16 셀 = 32 삼각형
    EXPECT_EQ(mesh.getTriangleCount(), 32);
}

TEST_F(GridMeshGridTest, CorrectIndexCount) {
    // 32 삼각형 * 3 정점 = 96 인덱스
    EXPECT_EQ(mesh.getIndices().size(), 96u);
}

TEST_F(GridMeshGridTest, VertexCoordinatesInRange) {
    const auto& vertices = mesh.getVertices();

    for (const auto& v : vertices) {
        // X 좌표: face_rect.x ~ face_rect.x + face_rect.width
        EXPECT_GE(v.x, face_rect.x - 1e-6f);
        EXPECT_LE(v.x, face_rect.x + face_rect.width + 1e-6f);

        // Y 좌표: face_rect.y ~ face_rect.y + face_rect.height
        EXPECT_GE(v.y, face_rect.y - 1e-6f);
        EXPECT_LE(v.y, face_rect.y + face_rect.height + 1e-6f);

        // 텍스처 좌표: 0 ~ 1
        EXPECT_GE(v.u, 0.0f - 1e-6f);
        EXPECT_LE(v.u, 1.0f + 1e-6f);
        EXPECT_GE(v.v, 0.0f - 1e-6f);
        EXPECT_LE(v.v, 1.0f + 1e-6f);
    }
}

TEST_F(GridMeshGridTest, CornerVertices) {
    const auto& vertices = mesh.getVertices();

    // 왼쪽 위 (첫 번째 정점)
    EXPECT_FLOAT_EQ(vertices[0].x, face_rect.x);
    EXPECT_FLOAT_EQ(vertices[0].y, face_rect.y);
    EXPECT_FLOAT_EQ(vertices[0].u, 0.0f);
    EXPECT_FLOAT_EQ(vertices[0].v, 0.0f);

    // 오른쪽 위 (인덱스 4)
    EXPECT_FLOAT_EQ(vertices[4].x, face_rect.x + face_rect.width);
    EXPECT_FLOAT_EQ(vertices[4].y, face_rect.y);
    EXPECT_FLOAT_EQ(vertices[4].u, 1.0f);
    EXPECT_FLOAT_EQ(vertices[4].v, 0.0f);

    // 왼쪽 아래 (인덱스 20)
    EXPECT_FLOAT_EQ(vertices[20].x, face_rect.x);
    EXPECT_FLOAT_EQ(vertices[20].y, face_rect.y + face_rect.height);
    EXPECT_FLOAT_EQ(vertices[20].u, 0.0f);
    EXPECT_FLOAT_EQ(vertices[20].v, 1.0f);

    // 오른쪽 아래 (인덱스 24)
    EXPECT_FLOAT_EQ(vertices[24].x, face_rect.x + face_rect.width);
    EXPECT_FLOAT_EQ(vertices[24].y, face_rect.y + face_rect.height);
    EXPECT_FLOAT_EQ(vertices[24].u, 1.0f);
    EXPECT_FLOAT_EQ(vertices[24].v, 1.0f);
}

// =============================================================================
// 컨트롤 포인트 테스트
// =============================================================================

class GridMeshControlPointTest : public ::testing::Test {
protected:
    void SetUp() override {
        face_rect = {0.0f, 0.0f, 1.0f, 1.0f};  // 전체 이미지
        mesh.initialize(20, face_rect);

        // 가상 랜드마크 생성 (468개)
        landmarks.resize(468);
        for (int i = 0; i < 468; ++i) {
            landmarks[i].x = static_cast<float>(i % 22) / 21.0f;
            landmarks[i].y = static_cast<float>(i / 22) / 21.0f;
            landmarks[i].z = 0.0f;
            landmarks[i].visibility = 1.0f;
        }
    }

    GridMesh mesh;
    Rect face_rect;
    std::vector<IrisLandmark> landmarks;
};

TEST_F(GridMeshControlPointTest, SetControlPointsSuccess) {
    EXPECT_TRUE(mesh.setControlPoints(landmarks.data(), 468, 1920, 1080));
}

TEST_F(GridMeshControlPointTest, SetControlPointsWithNullptr) {
    EXPECT_FALSE(mesh.setControlPoints(nullptr, 468, 1920, 1080));
}

TEST_F(GridMeshControlPointTest, SetControlPointsWithInsufficientLandmarks) {
    EXPECT_FALSE(mesh.setControlPoints(landmarks.data(), 100, 1920, 1080));
}

TEST_F(GridMeshControlPointTest, ControlPointsAreMarked) {
    mesh.setControlPoints(landmarks.data(), 468, 1920, 1080);

    const auto& vertices = mesh.getVertices();
    int control_count = 0;

    for (const auto& v : vertices) {
        if (v.is_control) {
            control_count++;
            EXPECT_GE(v.landmark_idx, 0);
        }
    }

    // 최소 1개 이상의 컨트롤 포인트가 있어야 함
    EXPECT_GT(control_count, 0);
}

// =============================================================================
// 변위 테스트
// =============================================================================

class GridMeshDisplacementTest : public ::testing::Test {
protected:
    void SetUp() override {
        face_rect = {0.0f, 0.0f, 1.0f, 1.0f};
        mesh.initialize(10, face_rect);

        // 랜드마크 생성 및 컨트롤 포인트 설정
        landmarks.resize(468);
        for (int i = 0; i < 468; ++i) {
            landmarks[i].x = static_cast<float>(i % 22) / 21.0f;
            landmarks[i].y = static_cast<float>(i / 22) / 21.0f;
            landmarks[i].z = 0.0f;
            landmarks[i].visibility = 1.0f;
        }
        mesh.setControlPoints(landmarks.data(), 468, 1920, 1080);
    }

    GridMesh mesh;
    Rect face_rect;
    std::vector<IrisLandmark> landmarks;
};

TEST_F(GridMeshDisplacementTest, ResetDisplacements) {
    // 변위 설정 후 리셋
    mesh.setControlPointDisplacement(10, 0.1f, 0.2f);
    mesh.resetDisplacements();

    const auto& vertices = mesh.getVertices();
    for (const auto& v : vertices) {
        EXPECT_FLOAT_EQ(v.dx, 0.0f);
        EXPECT_FLOAT_EQ(v.dy, 0.0f);
    }
}

TEST_F(GridMeshDisplacementTest, SetControlPointDisplacement) {
    // ControlLandmarks 인덱스 중 하나 (얼굴 외곽 첫 번째)
    int lm_idx = 10;
    EXPECT_TRUE(mesh.setControlPointDisplacement(lm_idx, 0.05f, -0.03f));
}

TEST_F(GridMeshDisplacementTest, SetDisplacementForNonControlPoint) {
    // 컨트롤 포인트가 아닌 랜드마크 인덱스
    EXPECT_FALSE(mesh.setControlPointDisplacement(999, 0.1f, 0.1f));
}

TEST_F(GridMeshDisplacementTest, InterpolateDisplacements) {
    // 컨트롤 포인트에 변위 설정
    mesh.setControlPointDisplacement(10, 0.1f, 0.1f);

    // 보간 실행
    mesh.interpolateDisplacements();

    // 컨트롤 포인트의 변위는 유지되어야 함
    const auto& vertices = mesh.getVertices();
    for (const auto& v : vertices) {
        if (v.landmark_idx == 10) {
            EXPECT_FLOAT_EQ(v.dx, 0.1f);
            EXPECT_FLOAT_EQ(v.dy, 0.1f);
        }
    }
}

TEST_F(GridMeshDisplacementTest, NearbyVerticesHaveNonZeroDisplacement) {
    // 컨트롤 포인트에 변위 설정
    mesh.setControlPointDisplacement(10, 0.2f, 0.2f);
    mesh.interpolateDisplacements();

    // 최소 일부 정점은 0이 아닌 변위를 가져야 함
    const auto& vertices = mesh.getVertices();
    int nonzero_count = 0;

    for (const auto& v : vertices) {
        if (std::abs(v.dx) > 1e-6f || std::abs(v.dy) > 1e-6f) {
            nonzero_count++;
        }
    }

    EXPECT_GT(nonzero_count, 1);  // 컨트롤 포인트 외에도 영향 받은 정점 존재
}

// =============================================================================
// 정점 버퍼 테스트
// =============================================================================

class GridMeshBufferTest : public ::testing::Test {
protected:
    void SetUp() override {
        face_rect = {0.1f, 0.1f, 0.8f, 0.8f};
        mesh.initialize(5, face_rect);  // 6x6 = 36 정점
    }

    GridMesh mesh;
    Rect face_rect;
};

TEST_F(GridMeshBufferTest, VertexBufferSize) {
    auto buffer = mesh.getVertexBuffer();

    // 36 정점 * 4 floats (x, y, u, v)
    EXPECT_EQ(buffer.size(), 36u * 4u);
}

TEST_F(GridMeshBufferTest, VertexBufferFormat) {
    auto buffer = mesh.getVertexBuffer();
    const auto& vertices = mesh.getVertices();

    for (int i = 0; i < mesh.getVertexCount(); ++i) {
        const auto& v = vertices[i];
        int offset = i * 4;

        // x + dx
        EXPECT_FLOAT_EQ(buffer[offset + 0], v.x + v.dx);
        // y + dy
        EXPECT_FLOAT_EQ(buffer[offset + 1], v.y + v.dy);
        // u
        EXPECT_FLOAT_EQ(buffer[offset + 2], v.u);
        // v
        EXPECT_FLOAT_EQ(buffer[offset + 3], v.v);
    }
}

TEST_F(GridMeshBufferTest, ComputeFinalPositions) {
    // 변위 직접 설정 (테스트용)
    // 정상적으로는 setControlPointDisplacement + interpolateDisplacements 사용

    mesh.computeFinalPositions();
    auto buffer = mesh.getVertexBuffer();

    // 버퍼가 유효한지 확인
    EXPECT_EQ(buffer.size(), 36u * 4u);
}

TEST_F(GridMeshBufferTest, IndexBufferValid) {
    const auto& indices = mesh.getIndices();

    // 모든 인덱스가 유효한 범위인지 확인
    for (uint16_t idx : indices) {
        EXPECT_LT(idx, static_cast<uint16_t>(mesh.getVertexCount()));
    }
}

// =============================================================================
// ControlLandmarks 테스트
// =============================================================================

TEST(ControlLandmarksTest, TotalCount) {
    EXPECT_EQ(ControlLandmarks::TOTAL_COUNT, 32);
}

TEST(ControlLandmarksTest, GetAllIndices) {
    auto indices = ControlLandmarks::getAllIndices();

    EXPECT_EQ(indices.size(), static_cast<size_t>(ControlLandmarks::TOTAL_COUNT));

    // 모든 인덱스가 MediaPipe 범위 내에 있는지 확인
    for (int idx : indices) {
        EXPECT_GE(idx, 0);
        EXPECT_LT(idx, 468);
    }
}

TEST(ControlLandmarksTest, NoDuplicates) {
    auto indices = ControlLandmarks::getAllIndices();
    std::sort(indices.begin(), indices.end());

    auto last = std::unique(indices.begin(), indices.end());
    EXPECT_EQ(last, indices.end());  // 중복 없음
}

// =============================================================================
// RBF 파라미터 테스트
// =============================================================================

TEST(GridMeshRbfTest, DefaultSigma) {
    GridMesh mesh;
    EXPECT_FLOAT_EQ(mesh.getRbfSigma(), GridMesh::DEFAULT_RBF_SIGMA);
}

TEST(GridMeshRbfTest, SetSigma) {
    GridMesh mesh;
    mesh.setRbfSigma(0.25f);
    EXPECT_FLOAT_EQ(mesh.getRbfSigma(), 0.25f);
}

// =============================================================================
// LOD (Level of Detail) 테스트
// =============================================================================

class GridMeshLODTest : public ::testing::Test {
protected:
    GridMesh mesh;
};

TEST_F(GridMeshLODTest, SelectLODForLargeFace) {
    // 얼굴이 화면의 48% (0.8 * 0.6) 차지 → High
    Rect large_face{0.1f, 0.2f, 0.8f, 0.6f};
    EXPECT_EQ(GridMesh::selectLOD(large_face), MeshLOD::High);
}

TEST_F(GridMeshLODTest, SelectLODForMediumFace) {
    // 얼굴이 화면의 20% (0.5 * 0.4) 차지 → Medium
    Rect medium_face{0.25f, 0.3f, 0.5f, 0.4f};
    EXPECT_EQ(GridMesh::selectLOD(medium_face), MeshLOD::Medium);
}

TEST_F(GridMeshLODTest, SelectLODForSmallFace) {
    // 얼굴이 화면의 6% (0.2 * 0.3) 차지 → Low
    Rect small_face{0.4f, 0.35f, 0.2f, 0.3f};
    EXPECT_EQ(GridMesh::selectLOD(small_face), MeshLOD::Low);
}

TEST_F(GridMeshLODTest, SelectLODAtHighThreshold) {
    // 정확히 30% → High
    Rect threshold_face{0.0f, 0.0f, 0.6f, 0.5f};
    EXPECT_EQ(GridMesh::selectLOD(threshold_face), MeshLOD::High);
}

TEST_F(GridMeshLODTest, SelectLODAtMediumThreshold) {
    // 정확히 15% → Medium
    Rect threshold_face{0.0f, 0.0f, 0.5f, 0.3f};
    EXPECT_EQ(GridMesh::selectLOD(threshold_face), MeshLOD::Medium);
}

TEST_F(GridMeshLODTest, SelectLODBelowMediumThreshold) {
    // 14% → Low
    Rect below_face{0.0f, 0.0f, 0.35f, 0.4f};
    EXPECT_EQ(GridMesh::selectLOD(below_face), MeshLOD::Low);
}

TEST_F(GridMeshLODTest, InitializeWithLODAutoSelect) {
    // 큰 얼굴 → High (20x20)
    Rect large_face{0.1f, 0.1f, 0.8f, 0.8f};
    EXPECT_TRUE(mesh.initializeWithLOD(large_face));
    EXPECT_EQ(mesh.getCurrentLOD(), MeshLOD::High);
    EXPECT_EQ(mesh.getGridSize(), 20);
    EXPECT_EQ(mesh.getVertexCount(), 21 * 21);
}

TEST_F(GridMeshLODTest, InitializeWithLODMedium) {
    Rect medium_face{0.2f, 0.2f, 0.5f, 0.4f};
    EXPECT_TRUE(mesh.initializeWithLOD(medium_face));
    EXPECT_EQ(mesh.getCurrentLOD(), MeshLOD::Medium);
    EXPECT_EQ(mesh.getGridSize(), 14);
    EXPECT_EQ(mesh.getVertexCount(), 15 * 15);
}

TEST_F(GridMeshLODTest, InitializeWithLODLow) {
    Rect small_face{0.3f, 0.3f, 0.2f, 0.3f};
    EXPECT_TRUE(mesh.initializeWithLOD(small_face));
    EXPECT_EQ(mesh.getCurrentLOD(), MeshLOD::Low);
    EXPECT_EQ(mesh.getGridSize(), 8);
    EXPECT_EQ(mesh.getVertexCount(), 9 * 9);
}

TEST_F(GridMeshLODTest, InitializeWithExplicitLOD) {
    Rect face_rect{0.1f, 0.1f, 0.6f, 0.8f};

    // 명시적으로 Low LOD 지정 (얼굴 크기와 무관)
    EXPECT_TRUE(mesh.initializeWithLOD(MeshLOD::Low, face_rect));
    EXPECT_EQ(mesh.getCurrentLOD(), MeshLOD::Low);
    EXPECT_EQ(mesh.getGridSize(), 8);
}

TEST_F(GridMeshLODTest, LODGridSizesAreCorrect) {
    EXPECT_EQ(GridMesh::LOD_GRID_SIZES[0], 8);   // Low
    EXPECT_EQ(GridMesh::LOD_GRID_SIZES[1], 14);  // Medium
    EXPECT_EQ(GridMesh::LOD_GRID_SIZES[2], 20);  // High
}

TEST_F(GridMeshLODTest, LODTriangleCountScaling) {
    Rect face_rect{0.1f, 0.1f, 0.8f, 0.8f};

    // Low LOD
    GridMesh mesh_low;
    mesh_low.initializeWithLOD(MeshLOD::Low, face_rect);
    int low_triangles = mesh_low.getTriangleCount();

    // Medium LOD
    GridMesh mesh_med;
    mesh_med.initializeWithLOD(MeshLOD::Medium, face_rect);
    int med_triangles = mesh_med.getTriangleCount();

    // High LOD
    GridMesh mesh_high;
    mesh_high.initializeWithLOD(MeshLOD::High, face_rect);
    int high_triangles = mesh_high.getTriangleCount();

    // 삼각형 수가 LOD 레벨에 따라 증가해야 함
    EXPECT_LT(low_triangles, med_triangles);
    EXPECT_LT(med_triangles, high_triangles);

    // 구체적인 값 검증
    EXPECT_EQ(low_triangles, 8 * 8 * 2);    // 128
    EXPECT_EQ(med_triangles, 14 * 14 * 2);  // 392
    EXPECT_EQ(high_triangles, 20 * 20 * 2); // 800
}

TEST_F(GridMeshLODTest, LODWithInvalidRectFails) {
    Rect invalid_rect{0.0f, 0.0f, 0.0f, 0.0f};
    EXPECT_FALSE(mesh.initializeWithLOD(invalid_rect));
}

// =============================================================================
// 성능 최적화 테스트
// =============================================================================

class GridMeshPerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        face_rect = {0.0f, 0.0f, 1.0f, 1.0f};

        // 랜드마크 생성 (468개)
        landmarks.resize(468);
        for (int i = 0; i < 468; ++i) {
            landmarks[i].x = static_cast<float>(i % 22) / 21.0f;
            landmarks[i].y = static_cast<float>(i / 22) / 21.0f;
            landmarks[i].z = 0.0f;
            landmarks[i].visibility = 1.0f;
        }
    }

    Rect face_rect;
    std::vector<IrisLandmark> landmarks;
};

TEST_F(GridMeshPerformanceTest, LowLODFasterThanHigh) {
    const int iterations = 200;

    // High LOD 성능 측정
    GridMesh mesh_high;
    mesh_high.initializeWithLOD(MeshLOD::High, face_rect);
    mesh_high.setControlPoints(landmarks.data(), 468, 1920, 1080);

    auto start_high = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        mesh_high.resetDisplacements();
        mesh_high.setControlPointDisplacement(10, 0.1f, 0.1f);
        mesh_high.interpolateDisplacements();
        mesh_high.computeFinalPositions();
    }
    auto end_high = std::chrono::high_resolution_clock::now();
    auto high_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_high - start_high).count();

    // Low LOD 성능 측정
    GridMesh mesh_low;
    mesh_low.initializeWithLOD(MeshLOD::Low, face_rect);
    mesh_low.setControlPoints(landmarks.data(), 468, 1920, 1080);

    auto start_low = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        mesh_low.resetDisplacements();
        mesh_low.setControlPointDisplacement(10, 0.1f, 0.1f);
        mesh_low.interpolateDisplacements();
        mesh_low.computeFinalPositions();
    }
    auto end_low = std::chrono::high_resolution_clock::now();
    auto low_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_low - start_low).count();

    // Low LOD는 High LOD보다 빨라야 함
    EXPECT_LT(low_ms, high_ms) << "Low LOD (" << low_ms << "us) should be faster than High LOD (" << high_ms << "us)";
}

TEST_F(GridMeshPerformanceTest, OptimizedInterpolationSkipsZeroDisplacement) {
    GridMesh mesh;
    mesh.initialize(20, face_rect);
    mesh.setControlPoints(landmarks.data(), 468, 1920, 1080);

    // 변위 없이 보간 실행 → 활성 컨트롤 포인트 0개이므로 빨라야 함
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        mesh.resetDisplacements();
        mesh.interpolateDisplacements();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // 1000번 보간이 50ms 이내 (변위 없는 경우 즉시 반환)
    EXPECT_LT(duration.count(), 50)
        << "Zero displacement interpolation should be fast: " << duration.count() << "ms";
}

TEST_F(GridMeshPerformanceTest, FindNearestVertexOptimized) {
    GridMesh mesh;
    mesh.initialize(20, face_rect);

    // O(1) 최근접 탐색이 빠른지 검증
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; ++i) {
        float x = static_cast<float>(i % 100) / 100.0f;
        float y = static_cast<float>(i / 100) / 100.0f;
        // setControlPoints 내부에서 findNearestVertex 호출
        // 직접 호출할 수 없으므로 setControlPoints 전체 측정
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // setControlPoints 전체 성능 검증
    auto start2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        mesh.setControlPoints(landmarks.data(), 468, 1920, 1080);
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);

    // 100번 setControlPoints가 100ms 이내
    EXPECT_LT(duration2.count(), 100)
        << "100x setControlPoints should complete in < 100ms: " << duration2.count() << "ms";
}

// =============================================================================
// LOD + 변위 보간 정확도 테스트
// =============================================================================

class GridMeshLODAccuracyTest : public ::testing::Test {
protected:
    void SetUp() override {
        face_rect = {0.0f, 0.0f, 1.0f, 1.0f};

        landmarks.resize(468);
        for (int i = 0; i < 468; ++i) {
            landmarks[i].x = static_cast<float>(i % 22) / 21.0f;
            landmarks[i].y = static_cast<float>(i / 22) / 21.0f;
            landmarks[i].z = 0.0f;
            landmarks[i].visibility = 1.0f;
        }
    }

    Rect face_rect;
    std::vector<IrisLandmark> landmarks;
};

TEST_F(GridMeshLODAccuracyTest, LowLODStillProducesReasonableResults) {
    GridMesh mesh;
    mesh.initializeWithLOD(MeshLOD::Low, face_rect);
    mesh.setControlPoints(landmarks.data(), 468, 1920, 1080);

    // 컨트롤 포인트에 변위 설정
    mesh.setControlPointDisplacement(10, 0.1f, 0.1f);
    mesh.interpolateDisplacements();
    mesh.computeFinalPositions();

    // 정점 버퍼가 올바른 크기
    auto buffer = mesh.getVertexBuffer();
    EXPECT_EQ(buffer.size(), static_cast<size_t>(mesh.getVertexCount()) * 4u);

    // 비-컨트롤 정점에도 보간된 변위가 있어야 함
    const auto& vertices = mesh.getVertices();
    int nonzero_count = 0;
    for (const auto& v : vertices) {
        if (!v.is_control && (std::abs(v.dx) > 1e-6f || std::abs(v.dy) > 1e-6f)) {
            nonzero_count++;
        }
    }
    EXPECT_GT(nonzero_count, 0) << "Interpolation should affect nearby vertices even at Low LOD";
}

TEST_F(GridMeshLODAccuracyTest, AllLODsProduceValidIndexBuffers) {
    for (int lod_int = 0; lod_int <= 2; ++lod_int) {
        MeshLOD lod = static_cast<MeshLOD>(lod_int);
        GridMesh mesh;
        mesh.initializeWithLOD(lod, face_rect);

        const auto& indices = mesh.getIndices();

        // 모든 인덱스가 유효한 범위
        for (uint16_t idx : indices) {
            EXPECT_LT(idx, static_cast<uint16_t>(mesh.getVertexCount()))
                << "Index out of range at LOD " << lod_int;
        }
    }
}

// =============================================================================
// 엣지 케이스 테스트
// =============================================================================

TEST(GridMeshEdgeCaseTest, MinimalGrid) {
    GridMesh mesh;
    Rect rect{0.0f, 0.0f, 1.0f, 1.0f};

    EXPECT_TRUE(mesh.initialize(2, rect));  // 최소 그리드
    EXPECT_EQ(mesh.getVertexCount(), 9);    // 3x3
    EXPECT_EQ(mesh.getTriangleCount(), 8);  // 2x2x2
}

TEST(GridMeshEdgeCaseTest, LargeGrid) {
    GridMesh mesh;
    Rect rect{0.0f, 0.0f, 1.0f, 1.0f};

    EXPECT_TRUE(mesh.initialize(100, rect));  // 최대 그리드
    EXPECT_EQ(mesh.getVertexCount(), 101 * 101);
    EXPECT_EQ(mesh.getTriangleCount(), 100 * 100 * 2);
}

TEST(GridMeshEdgeCaseTest, SmallFaceRect) {
    GridMesh mesh;
    Rect small_rect{0.4f, 0.4f, 0.2f, 0.2f};

    EXPECT_TRUE(mesh.initialize(10, small_rect));

    const auto& vertices = mesh.getVertices();
    for (const auto& v : vertices) {
        EXPECT_GE(v.x, 0.4f - 1e-6f);
        EXPECT_LE(v.x, 0.6f + 1e-6f);
        EXPECT_GE(v.y, 0.4f - 1e-6f);
        EXPECT_LE(v.y, 0.6f + 1e-6f);
    }
}

// =============================================================================
// 메인 함수
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
