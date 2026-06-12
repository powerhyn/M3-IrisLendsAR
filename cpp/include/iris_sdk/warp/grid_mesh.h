/**
 * @file grid_mesh.h
 * @brief Face Warp용 Grid Mesh 클래스
 *
 * 얼굴 워핑을 위한 균일 그리드 메시를 생성하고 관리합니다.
 * MediaPipe 얼굴 랜드마크를 컨트롤 포인트로 사용하여
 * RBF(Radial Basis Function) 보간으로 변위를 계산합니다.
 */

#ifndef IRIS_SDK_WARP_GRID_MESH_H
#define IRIS_SDK_WARP_GRID_MESH_H

#include <vector>
#include <cstdint>
#include <array>
#include "../types.h"
#include "../export.h"

namespace iris_sdk {
namespace warp {

/**
 * @brief 그리드 정점 구조체
 *
 * 각 정점은 원본 위치, 텍스처 좌표, 변위량을 포함합니다.
 */
struct GridVertex {
    float x;                ///< 원본 X 좌표 (이미지 기준 정규화 0~1)
    float y;                ///< 원본 Y 좌표 (이미지 기준 정규화 0~1)
    float u;                ///< 텍스처 U 좌표 (ROI 내부 정규화 0~1)
    float v;                ///< 텍스처 V 좌표 (ROI 내부 정규화 0~1)
    float dx;               ///< X 변위량 (정규화 좌표 단위)
    float dy;               ///< Y 변위량 (정규화 좌표 단위)
    bool is_control;        ///< 컨트롤 포인트 여부
    int landmark_idx;       ///< 매핑된 랜드마크 인덱스 (-1 if none)

    GridVertex()
        : x(0.0f), y(0.0f), u(0.0f), v(0.0f)
        , dx(0.0f), dy(0.0f)
        , is_control(false), landmark_idx(-1) {}
};

/**
 * @brief 컨트롤 포인트용 랜드마크 인덱스
 *
 * MediaPipe Face Mesh의 468개 랜드마크 중 얼굴 형태 변형에
 * 중요한 32개 포인트를 선택합니다.
 */
struct ControlLandmarks {
    /// 얼굴 외곽 (10개) - 좌/우 윤곽선
    static constexpr int FACE_CONTOUR[] = {
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323
    };
    static constexpr int FACE_CONTOUR_COUNT = 10;

    /// 눈 (8개) - 좌/우 눈 외곽
    static constexpr int EYES[] = {
        33, 133, 362, 263, 159, 145, 386, 374
    };
    static constexpr int EYES_COUNT = 8;

    /// 코 (4개) - 코 끝, 콧볼
    static constexpr int NOSE[] = {
        1, 4, 168, 6
    };
    static constexpr int NOSE_COUNT = 4;

    /// 입 (6개) - 입술 외곽
    static constexpr int MOUTH[] = {
        61, 291, 0, 17, 78, 308
    };
    static constexpr int MOUTH_COUNT = 6;

    /// 턱 (4개) - 턱선
    static constexpr int JAW[] = {
        152, 377, 400, 378
    };
    static constexpr int JAW_COUNT = 4;

    /// 전체 컨트롤 포인트 수
    static constexpr int TOTAL_COUNT =
        FACE_CONTOUR_COUNT + EYES_COUNT + NOSE_COUNT + MOUTH_COUNT + JAW_COUNT;

    /**
     * @brief 모든 컨트롤 랜드마크 인덱스 반환
     * @return 랜드마크 인덱스 벡터
     */
    static std::vector<int> getAllIndices();
};

/**
 * @brief LOD(Level of Detail) 레벨 열거형
 *
 * 얼굴 크기(화면 비율)에 따라 적절한 메쉬 밀도를 선택합니다.
 * - High: 얼굴이 화면의 30% 이상 차지 (근거리 / 상세)
 * - Medium: 얼굴이 화면의 15~30% (중간 거리)
 * - Low: 얼굴이 화면의 15% 미만 (원거리 / 전신)
 */
enum class MeshLOD : int {
    Low    = 0,  ///< 8x8 그리드 (81 정점, 128 삼각형)
    Medium = 1,  ///< 14x14 그리드 (225 정점, 392 삼각형)
    High   = 2   ///< 20x20 그리드 (441 정점, 800 삼각형)
};

/**
 * @brief Face Warp용 Grid Mesh 클래스
 *
 * 균일 그리드를 생성하고 얼굴 랜드마크 기반 변형을 지원합니다.
 * GPU 렌더링을 위한 정점/인덱스 버퍼를 제공합니다.
 * LOD(Level of Detail)를 지원하여 얼굴 크기에 따라 메쉬 밀도를 자동 조절합니다.
 */
class IRIS_SDK_EXPORT GridMesh {
public:
    /// 기본 그리드 크기 (20x20)
    static constexpr int DEFAULT_GRID_SIZE = 20;

    /// RBF 보간 파라미터 (Gaussian 함수의 sigma)
    static constexpr float DEFAULT_RBF_SIGMA = 0.15f;

    /// LOD별 그리드 크기
    static constexpr std::array<int, 3> LOD_GRID_SIZES = {8, 14, 20};

    /// LOD 전환 임계값 (얼굴 면적 비율)
    static constexpr float LOD_THRESHOLD_HIGH = 0.30f;    ///< 30% 이상 → High
    static constexpr float LOD_THRESHOLD_MEDIUM = 0.15f;  ///< 15% 이상 → Medium

    GridMesh();
    ~GridMesh();

    // 복사/이동 금지 (대용량 버퍼)
    GridMesh(const GridMesh&) = delete;
    GridMesh& operator=(const GridMesh&) = delete;
    GridMesh(GridMesh&&) = default;
    GridMesh& operator=(GridMesh&&) = default;

    /**
     * @brief 그리드 메시 초기화
     * @param grid_size 그리드 분할 수 (grid_size x grid_size 정점)
     * @param face_rect 얼굴 ROI 영역 (이미지 기준 정규화 좌표)
     * @return 초기화 성공 여부
     */
    bool initialize(int grid_size, const Rect& face_rect);

    /**
     * @brief LOD 기반 그리드 메시 초기화
     *
     * 얼굴 ROI의 화면 비율에 따라 적절한 LOD를 자동 선택합니다.
     *
     * @param face_rect 얼굴 ROI 영역 (이미지 기준 정규화 좌표)
     * @return 초기화 성공 여부
     */
    bool initializeWithLOD(const Rect& face_rect);

    /**
     * @brief 특정 LOD 레벨로 그리드 메시 초기화
     * @param lod LOD 레벨
     * @param face_rect 얼굴 ROI 영역
     * @return 초기화 성공 여부
     */
    bool initializeWithLOD(MeshLOD lod, const Rect& face_rect);

    /**
     * @brief 얼굴 ROI의 화면 비율에서 적절한 LOD 레벨 결정
     * @param face_rect 얼굴 ROI 영역 (정규화 좌표)
     * @return 적절한 LOD 레벨
     */
    static MeshLOD selectLOD(const Rect& face_rect);

    /**
     * @brief 현재 LOD 레벨 반환
     */
    MeshLOD getCurrentLOD() const { return current_lod_; }

    /**
     * @brief 얼굴 랜드마크로 컨트롤 포인트 설정
     * @param face_mesh MediaPipe Face Mesh 랜드마크 배열 (프로젝트 표준 478개 — 단,
     *        내부 매핑 테이블은 0~467만 수용하며 468~477 등록은 현재 탈락함. grid_mesh.cpp 참고)
     * @param landmark_count 랜드마크 개수
     * @param image_width 원본 이미지 너비 (픽셀)
     * @param image_height 원본 이미지 높이 (픽셀)
     * @return 설정 성공 여부
     */
    bool setControlPoints(const IrisLandmark* face_mesh,
                          int landmark_count,
                          int image_width,
                          int image_height);

    /**
     * @brief 모든 변위 초기화
     */
    void resetDisplacements();

    /**
     * @brief 특정 컨트롤 포인트의 변위 설정
     * @param landmark_idx MediaPipe 랜드마크 인덱스
     * @param dx X 변위량 (정규화 좌표)
     * @param dy Y 변위량 (정규화 좌표)
     * @return 설정 성공 여부 (컨트롤 포인트가 아니면 false)
     */
    bool setControlPointDisplacement(int landmark_idx, float dx, float dy);

    /**
     * @brief 추가 컨트롤 포인트 등록 (Face Warp 효과용)
     *
     * 기존 setControlPoints에서 등록되지 않은 랜드마크를 추가로
     * 컨트롤 포인트로 등록합니다. 이미 등록된 랜드마크는 무시됩니다.
     *
     * @param face_mesh MediaPipe Face Mesh 랜드마크 배열
     * @param landmark_indices 추가할 랜드마크 인덱스 배열
     * @param count 인덱스 배열 크기
     * @return 추가된 컨트롤 포인트 수
     */
    int addControlPoints(const IrisLandmark* face_mesh,
                         const int* landmark_indices,
                         int count);

    /**
     * @brief RBF 보간으로 모든 정점의 변위 계산
     *
     * 컨트롤 포인트의 변위를 기반으로 Gaussian RBF 함수를 사용하여
     * 나머지 정점들의 변위를 보간합니다.
     */
    void interpolateDisplacements();

    /**
     * @brief 최종 위치 계산 (원본 + 변위)
     *
     * 각 정점의 최종 위치는 원본 좌표에 변위를 더한 값입니다.
     * 이 함수는 GPU 버퍼 생성 전에 호출해야 합니다.
     */
    void computeFinalPositions();

    /**
     * @brief GPU용 정점 버퍼 생성
     * @return 정점 데이터 [x+dx, y+dy, u, v, ...] (4 floats per vertex)
     *
     * GPU에서 NDC 변환: ndc = position * 2.0 - 1.0
     */
    std::vector<float> getVertexBuffer() const;

    /**
     * @brief GPU용 인덱스 버퍼 반환
     * @return 삼각형 인덱스 (Triangle List)
     */
    const std::vector<uint16_t>& getIndices() const;

    /**
     * @brief 그리드 크기 반환
     */
    int getGridSize() const { return grid_size_; }

    /**
     * @brief 정점 개수 반환
     */
    int getVertexCount() const { return static_cast<int>(vertices_.size()); }

    /**
     * @brief 삼각형 개수 반환
     */
    int getTriangleCount() const { return static_cast<int>(indices_.size() / 3); }

    /**
     * @brief 얼굴 ROI 반환
     */
    const Rect& getFaceRect() const { return face_rect_; }

    /**
     * @brief 정점 배열 접근 (읽기 전용)
     */
    const std::vector<GridVertex>& getVertices() const { return vertices_; }

    /**
     * @brief 초기화 여부 확인
     */
    bool isInitialized() const { return initialized_; }

    /**
     * @brief RBF Sigma 값 설정
     * @param sigma Gaussian 함수의 표준편차 (0.05 ~ 0.5 권장)
     */
    void setRbfSigma(float sigma) { rbf_sigma_ = sigma; }

    /**
     * @brief 현재 RBF Sigma 값 반환
     */
    float getRbfSigma() const { return rbf_sigma_; }

private:
    /**
     * @brief 균일 그리드 정점 생성
     */
    void createGridVertices();

    /**
     * @brief 삼각형 인덱스 생성 (Triangle List)
     */
    void createTriangleIndices();

    /**
     * @brief 랜드마크와 가장 가까운 정점 찾기
     * @param lm_x 랜드마크 X 좌표 (정규화)
     * @param lm_y 랜드마크 Y 좌표 (정규화)
     * @return 가장 가까운 정점의 인덱스
     */
    int findNearestVertex(float lm_x, float lm_y) const;

    /**
     * @brief Gaussian RBF 함수
     * @param distance 두 점 사이 거리
     * @return RBF 가중치 (0~1)
     */
    float gaussianRbf(float distance) const;

    std::vector<GridVertex> vertices_;      ///< 그리드 정점 배열
    std::vector<uint16_t> indices_;         ///< 삼각형 인덱스 배열
    std::vector<float> final_positions_;    ///< 최종 위치 버퍼

    Rect face_rect_;                        ///< 얼굴 ROI 영역
    int grid_size_;                         ///< 그리드 분할 수
    float rbf_sigma_;                       ///< RBF Sigma 파라미터
    bool initialized_;                      ///< 초기화 완료 여부
    MeshLOD current_lod_;                   ///< 현재 LOD 레벨

    /// 랜드마크 인덱스 -> 정점 인덱스 매핑
    std::vector<int> landmark_to_vertex_;
};

} // namespace warp
} // namespace iris_sdk

#endif // IRIS_SDK_WARP_GRID_MESH_H
