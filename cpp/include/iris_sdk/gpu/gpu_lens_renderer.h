/**
 * @file gpu_lens_renderer.h
 * @brief GPU 기반 렌즈 렌더러 (OpenGL ES 3.1)
 *
 * Android 데모 CameraGLRenderer.kt의 검증된 GPU 렌즈 파이프라인을
 * SDK C++ 코어로 포팅. 전 플랫폼에서 동일한 렌즈 품질 보장.
 *
 * 기능:
 * - 비대칭 타원 마스크 (fitEyeEllipse)
 * - 눈꺼풀 클리핑 (Y-slab + 타원 듀얼 모드)
 * - 8종 블렌드 모드 (GPU 셰이더)
 * - Sclera Protection (기하학적 + 색상 기반)
 * - Contact Shadow (눈꺼풀 아래 그림자)
 * - OneEuroFilter 스무딩 (타원 + 눈꺼풀 파라미터)
 * - 눈꺼풀 홀드 프레임 (5프레임 캐시)
 */

#ifndef IRIS_SDK_GPU_LENS_RENDERER_H
#define IRIS_SDK_GPU_LENS_RENDERER_H

#include "iris_sdk/gpu/eye_render_packet.h"
#include "iris_sdk/gpu/shader_manager.h"
#include "iris_sdk/gpu/texture_pool.h"
#include "iris_sdk/one_euro_filter.h"
#include "iris_sdk/types.h"

#include <memory>
#include <mutex>
#include <array>

namespace iris_sdk {

class IRenderContext;

#if IRIS_SDK_GPU_AVAILABLE
class GLESRenderContext;
#endif

/**
 * @brief GPU 기반 렌즈 렌더러 (OpenGL ES 3.1)
 *
 * Android 데모 CameraGLRenderer.kt의 검증된 GPU 렌즈 파이프라인을
 * SDK C++ 코어로 포팅. 전 플랫폼에서 동일한 렌즈 품질 보장.
 *
 * 기능:
 * - 비대칭 타원 마스크 (fitEyeEllipse)
 * - 눈꺼풀 클리핑 (Y-slab + 타원 듀얼 모드)
 * - 8종 블렌드 모드 (GPU 셰이더)
 * - Sclera Protection (기하학적 + 색상 기반)
 * - Contact Shadow (눈꺼풀 아래 그림자)
 * - OneEuroFilter 스무딩 (타원 + 눈꺼풀 파라미터)
 * - 눈꺼풀 홀드 프레임 (5프레임 캐시)
 */
class GPULensRenderer {
public:
    GPULensRenderer();
    ~GPULensRenderer();

    GPULensRenderer(const GPULensRenderer&) = delete;
    GPULensRenderer& operator=(const GPULensRenderer&) = delete;
    GPULensRenderer(GPULensRenderer&&) = delete;
    GPULensRenderer& operator=(GPULensRenderer&&) = delete;

    // ========================================
    // 라이프사이클
    // ========================================

    bool initialize(IRenderContext* render_context);
    void release();
    bool isInitialized() const;

    // ========================================
    // 텍스처 관리
    // ========================================

    /// 렌즈 텍스처 업로드 (RGBA 데이터)
    bool loadLensTexture(const uint8_t* data, int width, int height);

    /// 렌즈 텍스처 해제
    void unloadLensTexture();

    /// 텍스처 로드 여부
    bool hasLensTexture() const;

    // ========================================
    // 렌더링
    // ========================================

    /**
     * @brief 텍스처 ID 기반 렌즈 렌더링 (Zero-Copy)
     *
     * @param input_texture 입력 카메라 프레임 텍스처 ID
     * @param output_texture 출력 텍스처 ID (생성됨)
     * @param width 프레임 너비
     * @param height 프레임 높이
     * @param iris_result 홍채 검출 결과 (face_mesh 포함)
     * @param config 렌즈 설정
     * @return 에러 코드
     */
    ErrorCode renderToTexture(
        uint32_t input_texture,
        uint32_t* output_texture,
        int width, int height,
        const IrisResult& iris_result,
        const LensConfig& config);

    // ========================================
    // 설정
    // ========================================

    void setScleraProtectEnabled(bool enabled);
    void setContactShadowEnabled(bool enabled);
    void setContactShadowIntensity(float intensity);
    void setEllipseMaskEnabled(bool enabled);
    /**
     * @deprecated P5-W3-05 S1에서 고정 조명 하이라이트 폐기. C5 환경 반사 계층이 대체.
     * 호환성 위해 선언은 유지하지만 no-op. 호출부는 제거 권장.
     */
    [[deprecated("P5-W3-05 S1: 고정 조명 하이라이트 폐기. C5 환경 반사 계층이 대체.")]]
    void setHighlightEnabled(bool enabled);

    // ========================================
    // 타원 피팅 (공개 유틸리티)
    // ========================================

    /**
     * @brief Face Mesh 랜드마크에서 비대칭 타원 피팅
     *
     * @param face_mesh 478점 Face Mesh (IrisLandmark[478])
     * @param is_left_eye true=왼쪽 눈
     * @param out_cx, out_cy 타원 중심 (정규화 좌표)
     * @param out_rx_inner 내안각 반경
     * @param out_rx_outer 외안각 반경
     * @param out_ry Y 반경
     * @param out_rotation 회전 각도 (라디안)
     */
    static void fitEyeEllipse(const IrisLandmark* face_mesh, bool is_left_eye,
                               float& out_cx, float& out_cy,
                               float& out_rx_inner, float& out_rx_outer,
                               float& out_ry, float& out_rotation);

    /**
     * @brief 눈꺼풀 Y 좌표 추출 (중앙값)
     */
    static float medianLandmarkY(const IrisLandmark* face_mesh, const int* indices, int count);

private:
    bool initializeShaders();
    void setupFullscreenQuad();
    void renderFullscreenQuad();
    void updateEyelidCache(const IrisResult& iris_result);
    void updateEllipseCache(const IrisResult& iris_result);

    /// W1 §5.3 fallback chain (실측 source 미연결 버전).
    /// 1) packet.avg_iris_luma → 2) hold(직전 유효값) < kAvgLumaMaxHoldFrames → 3) fallback 상수.
    /// 실측은 W6 (비동기 readback + 노출 정규화)에서 packet에 채울 예정.
    /// 반환은 clamp [kAvgLumaClampMin, kAvgLumaClampMax].
    float updateAvgIrisLuma(const gpu::EyeRenderPacket& left,
                            const gpu::EyeRenderPacket& right);

    // 렌더 컨텍스트 (외부 소유)
#if IRIS_SDK_GPU_AVAILABLE
    GLESRenderContext* render_context_ = nullptr;
#else
    void* render_context_ = nullptr;
#endif

    std::unique_ptr<ShaderManager> shader_manager_;
    std::unique_ptr<TexturePool> texture_pool_;

    // 풀스크린 쿼드
    GLuint quad_vao_ = 0;
    GLuint quad_vbo_ = 0;

    // 셰이더 프로그램
    GLuint lens_program_ = 0;

    // 렌즈 텍스처
    GLuint lens_texture_ = 0;
    int lens_texture_width_ = 0;
    int lens_texture_height_ = 0;

    // 설정
    // 기본값은 Kotlin 데모 참조 구현(CameraGLRenderer.kt 558~562)과 일치
    bool sclera_protect_ = true;
    bool contact_shadow_ = false;
    float shadow_intensity_ = 0.15f;
    bool use_ellipse_mask_ = false;
    // P5-W3-05 S1 D5: highlight_enabled_ 멤버 제거 (uniform/기능 모두 폐기)

    // ========================================
    // Uniform Location 캐시
    // ========================================
    struct LensUniforms {
        GLint uCameraTexture = -1;
        GLint uLensTexture = -1;

        // 홍채 파라미터
        GLint uLeftIrisCenter = -1;
        GLint uLeftIrisRadius = -1;
        GLint uRightIrisCenter = -1;
        GLint uRightIrisRadius = -1;

        // 렌즈 설정
        GLint uOpacity = -1;
        GLint uLensScale = -1;
        GLint uEdgeFeather = -1;
        GLint uBlendMode = -1;
        GLint uApplyLeft = -1;
        GLint uApplyRight = -1;
        GLint uFrameAspect = -1;

        // 눈꺼풀 클리핑
        GLint uLeftEyeTop = -1;
        GLint uLeftEyeBottom = -1;
        GLint uRightEyeTop = -1;
        GLint uRightEyeBottom = -1;
        GLint uEyelidFeather = -1;

        // 기능 플래그
        GLint uScleraProtect = -1;
        GLint uContactShadow = -1;
        GLint uShadowIntensity = -1;
        GLint uMaxDetail = -1;

        // 비대칭 타원
        GLint uUseEllipseMask = -1;
        GLint uLeftEyeEllipseCenter = -1;
        GLint uLeftEyeEllipseRadii = -1;
        GLint uLeftEyeEllipseRot = -1;
        GLint uRightEyeEllipseCenter = -1;
        GLint uRightEyeEllipseRadii = -1;
        GLint uRightEyeEllipseRot = -1;

        // 기타
        GLint uAvgIrisLum = -1;
        GLint uDetH = -1;
        // P5-W3-05 S1 D5: uHighlightEnabled 멤버 제거
    } lens_uniforms_;

    void cacheLensUniforms();

    // ========================================
    // OneEuroFilter 스무딩
    // ========================================
    static constexpr float FILTER_MIN_CUTOFF = 4.0f;
    static constexpr float FILTER_BETA = 15.0f;
    static constexpr float FILTER_BETA_EYELID = 12.0f;
    static constexpr float FILTER_D_CUTOFF = 1.0f;

    // 눈꺼풀 필터 (좌/우 x 상/하 = 4개)
    OneEuroFilter eyelid_top_filters_[2]{
        {FILTER_MIN_CUTOFF, FILTER_BETA_EYELID, FILTER_D_CUTOFF},
        {FILTER_MIN_CUTOFF, FILTER_BETA_EYELID, FILTER_D_CUTOFF}
    };
    OneEuroFilter eyelid_bottom_filters_[2]{
        {FILTER_MIN_CUTOFF, FILTER_BETA_EYELID, FILTER_D_CUTOFF},
        {FILTER_MIN_CUTOFF, FILTER_BETA_EYELID, FILTER_D_CUTOFF}
    };

    // 타원 필터 (좌/우 x 6파라미터 = 12개)
    struct EllipseFilters {
        OneEuroFilter cx{FILTER_MIN_CUTOFF, FILTER_BETA, FILTER_D_CUTOFF};
        OneEuroFilter cy{FILTER_MIN_CUTOFF, FILTER_BETA, FILTER_D_CUTOFF};
        OneEuroFilter rx_inner{FILTER_MIN_CUTOFF, FILTER_BETA, FILTER_D_CUTOFF};
        OneEuroFilter rx_outer{FILTER_MIN_CUTOFF, FILTER_BETA, FILTER_D_CUTOFF};
        OneEuroFilter ry{FILTER_MIN_CUTOFF, FILTER_BETA, FILTER_D_CUTOFF};
        OneEuroFilter rotation{FILTER_MIN_CUTOFF, FILTER_BETA, FILTER_D_CUTOFF};
    };
    EllipseFilters ellipse_filters_[2];  // [0]=left, [1]=right

    // ========================================
    // 눈꺼풀 캐시 (홀드 프레임)
    // ========================================
    static constexpr int EYELID_HOLD_FRAMES = 5;

    struct EyelidCache {
        float top = 0.0f;
        float bottom = 1.0f;
        int valid_frames = 0;
    };
    EyelidCache eyelid_cache_[2];  // [0]=left, [1]=right

    struct EllipseCache {
        float cx = 0.0f, cy = 0.0f;
        float rx_inner = 0.0f, rx_outer = 0.0f;
        float ry = 0.0f, rotation = 0.0f;
        int valid_frames = 0;
    };
    EllipseCache ellipse_cache_[2];

    // 이전 출력 텍스처 추적
    GLuint previous_output_texture_ = 0;

    // ========================================
    // P6-W1: avg_iris_luma fallback chain (실측 source는 W6 이관)
    // ========================================
    // 실측(self-measure)은 Android 카메라 텍스처(EXTERNAL_OES) + 임시 FBO attach +
    // glReadPixels 조합이 GL state 오염을 일으켜 검은 화면 회귀 발생 → revert.
    // 정식 측정 경로는 W6에서 비동기 PBO readback or detector CPU 버퍼 활용.
    // 현 단계는 packet.avg_iris_luma(미연결) → hold → fallback 상수 3단만 동작.
    static constexpr float kAvgLumaFallback      = 0.35f;
    static constexpr int   kAvgLumaMaxHoldFrames = 3;
    static constexpr float kAvgLumaClampMin      = 0.1f;
    static constexpr float kAvgLumaClampMax      = 0.9f;

    float current_avg_luma_    = kAvgLumaFallback;
    int   avg_luma_hold_count_ = 0;
    bool  avg_luma_has_valid_  = false;

    // P6-W2 §5.9: invalid blend ID(3/4/6/etc.) 1회 경고 (debug 빌드 한정).
    //   유효 ID = {0, 1, 2, 5, 7}. 그 외는 셰이더에서 TintLinearV2 fallback.
    bool invalid_blend_warned_ = false;

    bool initialized_ = false;
    mutable std::mutex mutex_;
};

} // namespace iris_sdk

#endif // IRIS_SDK_GPU_LENS_RENDERER_H
