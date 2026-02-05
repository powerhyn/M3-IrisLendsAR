/**
 * @file gpu_beauty_backend.h
 * @brief GPU 기반 뷰티 필터 백엔드 (OpenGL ES 3.1)
 *
 * 셰이더 기반 실시간 이미지 처리를 수행합니다.
 * LensRenderer와 GL 컨텍스트를 공유하여 Zero-Copy 텍스처 처리를 지원합니다.
 */

#ifndef IRIS_SDK_GPU_BEAUTY_BACKEND_H
#define IRIS_SDK_GPU_BEAUTY_BACKEND_H

#include "iris_sdk/beauty_backend.h"
#include "iris_sdk/gpu/shader_manager.h"
#include "iris_sdk/gpu/texture_pool.h"
#include "iris_sdk/gpu/gpu_profiler.h"

#include <memory>
#include <mutex>

// Forward declarations
namespace iris_sdk {
class IRenderContext;
}

#if IRIS_SDK_GPU_AVAILABLE
// Forward declaration for GLES context
namespace iris_sdk {
class GLESRenderContext;
}
#endif

namespace iris_sdk {

/**
 * @brief GPU 기반 뷰티 필터 백엔드
 *
 * OpenGL ES 3.1 셰이더를 사용하여 실시간 뷰티 필터를 적용합니다.
 *
 * 지원 기능:
 * - 피부 스무딩 (Bilateral Filter)
 * - 피부톤 화이트닝
 * - 컬러 밸런스
 * - 소프트 포커스
 * - 밝기 조정
 * - ROI 마스킹
 *
 * Thread-safe: 모든 public 메서드는 mutex로 보호됩니다.
 */
class GPUBeautyBackend : public IBeautyBackend {
public:
    GPUBeautyBackend();
    ~GPUBeautyBackend() override;

    // 복사/이동 금지
    GPUBeautyBackend(const GPUBeautyBackend&) = delete;
    GPUBeautyBackend& operator=(const GPUBeautyBackend&) = delete;
    GPUBeautyBackend(GPUBeautyBackend&&) = delete;
    GPUBeautyBackend& operator=(GPUBeautyBackend&&) = delete;

    //=========================================================================
    // IBeautyBackend 구현
    //=========================================================================

    /**
     * @brief 백엔드 초기화
     *
     * @param render_context GLESRenderContext 포인터 (필수)
     * @return 성공 시 true
     */
    bool initialize(IRenderContext* render_context) override;

    /**
     * @brief 리소스 해제
     */
    void release() override;

    /**
     * @brief 초기화 여부 확인
     */
    bool isInitialized() const override;

    /**
     * @brief 필터 적용 (CPU 버퍼, in-place)
     *
     * CPU 버퍼를 GPU 텍스처로 업로드하고 처리 후 다시 다운로드합니다.
     * 성능이 중요한 경우 applyTexture()를 사용하세요.
     */
    IrisSdkError apply(
        uint8_t* frame_data,
        int width, int height,
        IrisFrameFormat format,
        const BeautyFilterConfigV2& config,
        const BeautyROI* roi = nullptr
    ) override;

    /**
     * @brief 필터 적용 (텍스처, Zero-Copy)
     *
     * GPU 텍스처를 직접 처리하여 최대 성능을 달성합니다.
     */
    IrisSdkError applyTexture(
        const TextureHandle& input,
        TextureHandle& output,
        const BeautyFilterConfigV2& config,
        const BeautyROI* roi = nullptr
    ) override;

    /**
     * @brief 백엔드 이름
     */
    const char* getName() const override { return "GPUBeautyBackend"; }

    /**
     * @brief GPU 지원 여부
     */
    bool supportsGpu() const override { return true; }

    /**
     * @brief 텍스처 처리 지원 여부
     */
    bool supportsTextureProcessing() const override { return true; }

    //=========================================================================
    // GPU 전용 메서드
    //=========================================================================

    /**
     * @brief 텍스처 풀 접근 (메모리 관리용)
     */
    TexturePool* getTexturePool() { return texture_pool_.get(); }
    const TexturePool* getTexturePool() const { return texture_pool_.get(); }

    /**
     * @brief 셰이더 매니저 접근
     */
    ShaderManager* getShaderManager() { return shader_manager_.get(); }
    const ShaderManager* getShaderManager() const { return shader_manager_.get(); }

    /**
     * @brief 메모리 압력 처리 (Android onTrimMemory 연동)
     */
    void onMemoryPressure(int level);

    /**
     * @brief GPU 프로파일러 활성화/비활성화
     * @param enabled true: 활성화, false: 비활성화
     */
    void setProfilingEnabled(bool enabled);

    /**
     * @brief GPU 프로파일러 활성화 여부
     */
    bool isProfilingEnabled() const;

    /**
     * @brief GPU 성능 리포트 생성
     * @return 포맷된 성능 리포트 문자열
     */
    std::string getProfilingReport() const;

    /**
     * @brief GPU 프로파일러 접근
     */
    GPUProfiler* getProfiler() { return profiler_.get(); }
    const GPUProfiler* getProfiler() const { return profiler_.get(); }

    //=========================================================================
    // V2 API - 텍스처 ID 기반 (C API 호환)
    //=========================================================================

    /**
     * @brief 텍스처 ID 기반 뷰티 필터 적용 (C API용)
     *
     * @param input_texture 입력 OpenGL 텍스처 ID
     * @param output_texture 출력 텍스처 ID (새로 생성됨)
     * @param width 텍스처 너비
     * @param height 텍스처 높이
     * @param config 필터 설정
     * @param detection 얼굴 검출 결과 (ROI 생성용)
     * @return 에러 코드
     */
    IrisSdkError applyTextureId(
        uint32_t input_texture,
        uint32_t* output_texture,
        int width, int height,
        const BeautyFilterConfigV2& config,
        const IrisResult* detection
    );

    /**
     * @brief Face Warp 적용 (얼굴 형태 변형)
     *
     * @param input_texture 입력 텍스처 ID
     * @param output_texture 출력 텍스처 ID
     * @param width 텍스처 너비
     * @param height 텍스처 높이
     * @param slim_face 갸름한 얼굴 강도 (0.0~1.0)
     * @param thin_chin 턱 축소 강도 (0.0~1.0)
     * @param enlarge_eyes 눈 확대 강도 (0.0~1.0)
     * @param detection 얼굴 검출 결과 (랜드마크 필요)
     * @return 에러 코드
     */
    IrisSdkError applyFaceWarp(
        uint32_t input_texture,
        uint32_t* output_texture,
        int width, int height,
        float slim_face,
        float thin_chin,
        float enlarge_eyes,
        const IrisResult* detection
    );

    /**
     * @brief SDK가 관리하는 텍스처 해제
     *
     * @param texture 해제할 텍스처 ID
     */
    void releaseTexture(uint32_t texture);

private:
    //=========================================================================
    // 초기화 헬퍼
    //=========================================================================

    /// 셰이더 프로그램 초기화
    bool initializeShaders();

    /// 풀스크린 쿼드 VAO/VBO 설정
    void setupFullscreenQuad();

    /// 풀스크린 쿼드 렌더링
    void renderFullscreenQuad();

    //=========================================================================
    // 필터 패스
    //=========================================================================

    /// 스무딩 패스 (Bilateral Filter)
    void executeSmoothingPass(GLuint input_tex, GLuint output_fbo,
                              int width, int height,
                              const BeautyFilterConfigV2& config);

    /// 화이트닝 패스
    void executeWhiteningPass(GLuint input_tex, GLuint output_fbo,
                              int width, int height,
                              float whitening);

    /// 컬러 밸런스 패스
    void executeColorBalancePass(GLuint input_tex, GLuint output_fbo,
                                 int width, int height,
                                 float balance);

    /// 소프트 포커스 패스
    void executeSoftFocusPass(GLuint input_tex, GLuint output_fbo,
                              int width, int height,
                              float strength);

    /// 밝기 패스
    void executeBrightnessPass(GLuint input_tex, GLuint output_fbo,
                               int width, int height,
                               float brightness);

    /// 마스킹 패스 (ROI, 눈/입술 보호)
    void applyMasking(GLuint filtered_tex, GLuint original_tex,
                      GLuint mask_tex, GLuint output_fbo,
                      int width, int height);

    /// 통합 Color Adjustment 패스 (Brightness + ColorBalance + Whitening)
    /// 3개 패스를 1개로 병합하여 성능 최적화
    void executeCombinedColorPass(GLuint input_tex, GLuint output_fbo,
                                  int width, int height,
                                  float brightness, float balance, float whitening);

    //=========================================================================
    // 멤버 변수
    //=========================================================================

#if IRIS_SDK_GPU_AVAILABLE
    GLESRenderContext* render_context_ = nullptr;  // 외부 소유, 해제하지 않음
#else
    void* render_context_ = nullptr;
#endif

    std::unique_ptr<ShaderManager> shader_manager_;
    std::unique_ptr<TexturePool> texture_pool_;
    std::unique_ptr<GPUProfiler> profiler_;  // GPU 성능 프로파일러

    // 풀스크린 쿼드 VAO/VBO
    GLuint quad_vao_ = 0;
    GLuint quad_vbo_ = 0;

    // 셰이더 프로그램 ID
    GLuint passthrough_program_ = 0;
    GLuint smoothing_program_ = 0;
    GLuint whitening_program_ = 0;
    GLuint color_balance_program_ = 0;
    GLuint soft_focus_program_ = 0;
    GLuint brightness_program_ = 0;
    GLuint masking_program_ = 0;
    GLuint combined_color_program_ = 0;  // 통합 Color Adjustment (최적화)

    //=========================================================================
    // Uniform Location 캐시 (성능 최적화)
    //=========================================================================

    /// 셰이더별 Uniform Location 캐시 구조체
    struct UniformLocations {
        // 공통
        GLint uTexture = -1;

        // Smoothing (Bilateral Filter)
        GLint uTexelSize = -1;
        GLint uStrength = -1;

        // Brightness
        GLint uBrightness = -1;

        // Whitening
        GLint uWhiteningStrength = -1;

        // Color Balance
        GLint uBalance = -1;

        // Soft Focus
        GLint uSoftFocusTexelSize = -1;
        GLint uSoftFocusStrength = -1;

        // Masking
        GLint uFiltered = -1;
        GLint uOriginal = -1;
        GLint uMask = -1;

        // Combined Color Adjustment (통합 필터)
        GLint uCombinedBrightness = -1;
        GLint uCombinedBalance = -1;
        GLint uCombinedWhitening = -1;
    };

    /// 프로그램별 Uniform Location 캐시
    UniformLocations smoothing_uniforms_;
    UniformLocations whitening_uniforms_;
    UniformLocations color_balance_uniforms_;
    UniformLocations soft_focus_uniforms_;
    UniformLocations brightness_uniforms_;
    UniformLocations masking_uniforms_;
    UniformLocations combined_color_uniforms_;  // 통합 Color Adjustment

    /// Uniform Location 캐싱 (초기화 시 호출)
    void cacheUniformLocations();

    // 이전 출력 텍스처 추적 (텍스처 풀 관리용)
    TexturePool::TextureInfo* previous_output_ping_ = nullptr;
    TexturePool::TextureInfo* previous_output_pong_ = nullptr;
    GLuint previous_output_texture_ = 0;

    bool initialized_ = false;
    mutable std::mutex mutex_;
};

} // namespace iris_sdk

#endif // IRIS_SDK_GPU_BEAUTY_BACKEND_H
