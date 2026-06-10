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
#include "iris_sdk/one_euro_filter.h"

#include <array>
#include <memory>
#include <mutex>
#include <vector>

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
 * @brief GPU 기반 뷰티 필터 백엔드 (OpenGL ES 3.1)
 *
 * 실시간 뷰티 필터를 GPU 셰이더로 적용합니다.
 *
 * **파이프라인 구조**:
 * - Frequency Separation (skinQuality > 0): 6-subpass GPU 파이프라인 (Sharpen 패스 포함)
 *   - DeviceTier::HIGH → full-res, MID → hybrid half-res blur
 * - Bilateral Filter (skinQuality == 0 또는 FreqSep 실패 시 fallback)
 * - Combined Color Pass (brightness + balance + whitening + LUT)
 *
 * **Temporal Stability** (P4-W3-04):
 * - One Euro Filter로 blur_radius 및 face_rect center jitter 억제
 * - DeviceTier 기반 half-res 분기 (GPU 렌더러 문자열 파싱)
 *
 * **Thread Safety**: 모든 public 메서드는 mutex_로 보호됩니다.
 *
 * @see FreqSepParams, DeviceTier, OneEuroFilter
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
        const IrisResult* detection,
        uint32_t lut_texture_id = 0,
        float lut_intensity = 0.0f
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

    //=========================================================================
    // Frequency Separation
    //=========================================================================

    /// Freq Sep 내부 파라미터
    struct FreqSepParams {
        int blur_radius = 15;
        float high_freq_preserve = 0.45f;
        float low_freq_smooth_radius_ratio = 0.5f;
        float attenuation_low = 0.02f;
        float attenuation_high = 0.15f;
        float edge_weight = 0.5f;      // 에지 보존 강도
        float chroma_weight = 0.3f;    // 색소침착 감지 강도
        float tone_lift = 0.15f;      // 미드톤 리프트 강도
        float sharpen_amount = 0.15f; // Luminance sharpen 강도
        float texture_blend_floor = 0.38f; // Composite textureBlend 하한 (기본 0.38)
        bool enabled = false;
    };

    /// skinQuality → FreqSepParams 매핑 (레거시 호환)
    static FreqSepParams mapSkinQuality(float skin_quality, int face_width);

    /// smoothIntensity/poreReduction 2축 → FreqSepParams 매핑
    static FreqSepParams mapSmoothingAndPore(float smooth_intensity, float pore_reduction, int face_width);

    /**
     * @brief GPU 디바이스 성능 등급
     *
     * GL_RENDERER 문자열을 파싱하여 결정됩니다 (detectDeviceTier()).
     *
     * 파이프라인 동작 차이:
     * - HIGH: FreqSep full-res 6-subpass (blur + composite + sharpen, 모두 원본 해상도)
     * - MID:  FreqSep hybrid half-res (blur는 1/2 해상도, composite는 full-res)
     * - LOW:  FreqSep 비활성 → Bilateral fallback
     *
     * @note Mali 분류 비대칭: Adreno는 100 단위 시리즈(6xx/7xx),
     *       Mali-G는 2자리 vs 3자리(G7x/G710+)로 분류 기준이 다름.
     *       Mali-G78은 MID, Mali-G710은 HIGH로 분류됨.
     */
    enum class DeviceTier {
        HIGH,   ///< Adreno 7xx, Mali-G710+, Apple GPU, Desktop GPU
        MID,    ///< Adreno 6xx, Mali-G7x (G71~G78), PowerVR
        LOW     ///< 기타 저사양 GPU
    };

    /// GPU 렌더러 문자열 기반 디바이스 등급 분류 (GL 컨텍스트 불필요, 단위 테스트용)
    static DeviceTier classifyGpuRenderer(const std::string& renderer_str);

private:
    //=========================================================================
    // 초기화 헬퍼
    //=========================================================================

    /// GPU 렌더러 문자열 기반 디바이스 등급 감지 (GL 컨텍스트 활성 상태에서만 호출)
    DeviceTier detectDeviceTier();

    /// Temporal filter 일괄 리셋 (release/얼굴 추적 끊김 시 공통 호출)
    void resetTemporalFilters();

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
                                  float brightness, float balance, float whitening,
                                  GLuint lut_texture = 0, float lut_intensity = 0.0f);

    /// Frequency Separation Gaussian blur 셰이더 초기화
    bool initializeFreqSepShaders();

    /// FreqSep 파이프라인 실행 설정 (full-res / half-res 분기 매개변수화)
    struct FreqSepExecConfig {
        int res_divisor;                       ///< 1 = full-res, 2 = half-res
        bool linear_upsample;                  ///< true: composite 입력에 GL_LINEAR 설정
        const char* blur_profiler_suffix;      ///< "" 또는 "_Half"
        const char* composite_profiler_suffix; ///< "" 또는 "_Full"
    };

    /// Frequency Separation 6서브패스 파이프라인 (full-res, sharpen 포함)
    /// @return true: 파이프라인 정상 완료, false: 텍스처 할당 실패 등 (호출자가 fallback 처리)
    bool executeFreqSepPipeline(
        GLuint input_tex,
        GLuint mask_tex,
        GLuint output_fbo,
        int width, int height,
        const FreqSepParams& params);

    /// Frequency Separation MID 디바이스 하프 해상도 파이프라인
    /// blur 패스는 half-res, composite 패스는 full-res로 실행
    /// @return true: 파이프라인 정상 완료
    bool executeFreqSepPipelineHalfRes(
        GLuint input_tex,
        GLuint mask_tex,
        GLuint output_fbo,
        int width, int height,
        const FreqSepParams& params);

    /// FreqSep 공통 구현 (full-res / half-res 통합)
    bool executeFreqSepPipelineImpl(
        GLuint input_tex, GLuint mask_tex, GLuint output_fbo,
        int width, int height,
        const FreqSepParams& params,
        const FreqSepExecConfig& exec_cfg);

    /// CPU combined_mask → GPU 텍스처 업로드
    GLuint uploadSkinMask(
        const std::vector<uint8_t>& combined_mask,
        int mask_width, int mask_height);

    /// Vivid 포스트프로세싱 패스 (전체 프레임, ROI 무관)
    void executeVividPass(GLuint input_tex, GLuint output_fbo,
                          int width, int height,
                          float intensity, float saturation,
                          float brightness, float warmth);

    /// Freq Sep 불가 시 Bilateral fallback (공통 최소 강도 정책)
    void executeSmoothingWithFallbackStrength(
        GLuint input_tex, GLuint output_fbo,
        int width, int height,
        const BeautyFilterConfigV2& config);

    //=========================================================================
    // P8-W1: landmark-masked skin smoothing (LensSimulator 이식)
    //=========================================================================

    /// 새 모드 활성 + 강도>0 + face_mesh 유효 여부 판정 (비용 게이팅 진입점).
    bool skinMaskSmoothingActive(const IrisResult* detection) const;

    /// 1/4 해상도 5타깃(RGBA8×3 + R8×2) 지연 생성. 실패 시 1회 로그 후 비활성.
    /// 원본 ensureBeautyTargets/createBeautyTargets 패턴 (Renderer.kt 455-506).
    bool ensureSkinTargets(int width, int height);
    void destroySkinTargets();

    /// detection->face_mesh → 픽셀 공간 One-Euro 필터 → 이마 확장(마스크 전용) →
    /// 팬 정점 NDC 채우기. 모드 활성 시만 호출 (필터링 비용 0 게이팅).
    void prepareSkinFans(const IrisResult* detection, int width, int height, double frame_ts);

    /// 패스 1-2: 입력 1/4 다운샘플 → 컬러 블러 H/V → 마스크 팬 → 마스크 블러 H/V.
    void renderSkinBasePasses(GLuint input_tex, int width, int height);

    /// 분리형 가우시안 1방향 (저해상도 타깃 한정).
    void skinBlurPass(GLuint src_tex, GLuint dst_fbo, float dir_x, float dir_y, float offset_scale);

    /// 패스 3: 풀해상도 에지 가드 컴포지트 (output_fbo에 기록).
    void renderSkinComposite(GLuint base_tex, GLuint output_fbo,
                             int width, int height, float strength);

    /// landmark-masked smoothing 셰이더 초기화 (non-fatal).
    bool initializeSkinSmoothingShaders();

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

    // Freq Sep 셰이더 프로그램
    GLuint freq_sep_gaussian_program_ = 0;
    GLuint freq_sep_composite_program_ = 0;
    GLuint luminance_sharpen_program_ = 0;
    GLuint vivid_program_ = 0;

    // Skin mask GPU 텍스처
    GLuint skin_mask_texture_ = 0;
    int skin_mask_width_ = 0;
    int skin_mask_height_ = 0;

    //=========================================================================
    // P8-W1: landmark-masked skin smoothing 상태
    //=========================================================================

    // 5타깃 인덱스: [0]=다운샘플 RGBA, [1]=블러 중간 RGBA, [2]=컬러 블러 RGBA,
    //              [3]=마스크 R8, [4]=마스크 블러 R8 (원본 BT_LOW..BT_MASK_BLUR)
    static constexpr int kSkinTargetCount = 5;
    static constexpr int kSkinLow = 0;
    static constexpr int kSkinTmp = 1;
    static constexpr int kSkinBlur = 2;
    static constexpr int kSkinMask = 3;       // 이 인덱스 이상은 R8 단일 채널
    static constexpr int kSkinMaskBlur = 4;

    // 팬 정점 카운트 [무게중심+N+첫점반복]: 외곽38/눈썹12×2/입술22/눈18×2 (원본 동일)
    static constexpr std::array<int, 6> kSkinFanCounts = {38, 12, 12, 22, 18, 18};
    // 총 정점 수 = Σ kSkinFanCounts = 120 → 240 floats
    static constexpr int kSkinFanFloats = 240;
    // 폴리곤별 점 수: 외곽36/눈썹10×2/입술20/눈16×2 (= 108점, 216 floats)
    static constexpr int kSkinPointCount = 108;
    static constexpr float kSkinColorBlurScale = 1.6f; // 컬러 블러 offsetScale (P8-W1 §5)
    static constexpr float kSkinMaskBlurScale = 1.0f;  // 마스크 블러 offsetScale

    GLuint skin_mask_fill_program_ = 0;
    GLuint skin_blur_program_ = 0;
    GLuint skin_composite_program_ = 0;

    GLuint skin_target_tex_[kSkinTargetCount] = {0, 0, 0, 0, 0};
    GLuint skin_target_fbo_[kSkinTargetCount] = {0, 0, 0, 0, 0};
    int skin_low_w_ = 0;
    int skin_low_h_ = 0;
    bool skin_targets_ready_ = false;
    bool skin_targets_failed_ = false;  // 생성 실패 — 매 프레임 재시도 방지

    // 모드 토글 (internal API)
    bool skin_mask_smoothing_enabled_ = false;
    float skin_mask_smoothing_strength_ = 0.0f;

    // 팬 정점 버퍼 (NDC, position만) + 픽셀 좌표 작업 버퍼 (프레임당 할당 금지)
    std::array<float, kSkinFanFloats> skin_fan_{};
    std::array<float, kSkinPointCount * 2> skin_oval_px_{};  // 외곽 픽셀 (이마 확장용)

    // 픽셀 공간 One-Euro 필터 — 점별 x/y 2축 (모드 활성 시만 필터링·재획득 시 reset).
    // P8-W1 §5: min_cutoff 0.5 / beta 0.007 / d_cutoff 1.0 (원본 FaceTracker ADR-0002).
    std::array<OneEuroFilter, kSkinPointCount * 2> skin_landmark_filters_;
    bool skin_filters_active_ = false;  // 직전 프레임에 필터가 동작했는지 (재획득 reset 판정)

    // Skin smoothing uniform 캐시
    struct SkinUniforms {
        GLint maskFillValue = -1;
        GLint blurTexture = -1;
        GLint blurDirection = -1;
        GLint blurOffsetScale = -1;
        GLint compositeTexture = -1;
        GLint compositeBlurTex = -1;
        GLint compositeMaskTex = -1;
        GLint compositeSkin = -1;
    } skin_uniforms_;

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

        // Combined Color + LUT
        GLint uCombinedLutTexture = -1;
        GLint uCombinedLutIntensity = -1;
    };

    /// 프로그램별 Uniform Location 캐시
    UniformLocations smoothing_uniforms_;
    UniformLocations whitening_uniforms_;
    UniformLocations color_balance_uniforms_;
    UniformLocations soft_focus_uniforms_;
    UniformLocations brightness_uniforms_;
    UniformLocations masking_uniforms_;
    UniformLocations combined_color_uniforms_;  // 통합 Color Adjustment

    // Freq Sep Gaussian Uniform 캐시
    struct FreqSepGaussianUniforms {
        GLint uTexture = -1;
        GLint uDirection = -1;
        GLint uRadius = -1;
        GLint uWeights = -1;
        GLint uLinearize = -1;  // sRGB→Linear 변환 플래그
    } freq_sep_gaussian_uniforms_;

    // Freq Sep Composite Uniform 캐시
    struct FreqSepCompositeUniforms {
        GLint uSmoothedLow = -1;
        GLint uLowFreq = -1;
        GLint uOriginal = -1;
        GLint uSkinMask = -1;
        GLint uHighFreqPreserve = -1;
        GLint uAttenuationLow = -1;
        GLint uAttenuationHigh = -1;
        GLint uEdgeWeight = -1;
        GLint uChromaWeight = -1;
        GLint uToneLift = -1;
        GLint uTextureBlendFloor = -1;
        GLint uDebugMode = -1;
        GLint uSkinColorFilter = -1;
    } freq_sep_composite_uniforms_;

    // Luminance Sharpen Uniform 캐시
    struct LuminanceSharpenUniforms {
        GLint uTexture = -1;
        GLint uSkinMask = -1;
        GLint uSharpenAmount = -1;
        GLint uTexelSize = -1;
    } luminance_sharpen_uniforms_;

    // Vivid Postprocess Uniform 캐시
    struct VividUniforms {
        GLint uTexture = -1;
        GLint uIntensity = -1;
        GLint uSaturation = -1;
        GLint uBrightness = -1;
        GLint uWarmth = -1;
    } vivid_uniforms_;

    // Temporal stability용 One Euro Filter (P4-W3-04)
    // 모든 필터는 mutex_ lock 하에서만 접근 (applyTextureId → public → lock_guard)
    //
    // 파라미터 선택 근거:
    //   skin_radius: min_cutoff=0.5 (강한 스무딩 — radius 변화가 급격하면 블러 플리커 발생)
    //                beta=0.01 (느린 추종 — radius는 급변할 이유가 없음)
    //   face_rect center: min_cutoff=1.0 (적당한 스무딩 — 자연스러운 이동 허용)
    //                     beta=0.02 (빠른 머리 움직임에 약간 반응)
    OneEuroFilter skin_radius_filter_{0.5f, 0.01f, 1.0f};
    OneEuroFilter mask_center_x_filter_{1.0f, 0.02f, 1.0f};  ///< face_rect 중심 X 안정화
    OneEuroFilter mask_center_y_filter_{1.0f, 0.02f, 1.0f};  ///< face_rect 중심 Y 안정화

    // 디바이스 성능 등급 (P4-W3-04)
    DeviceTier device_tier_ = DeviceTier::HIGH;

    // FreqSep 디버그 모드 (0=off, 1=magnitude, 2=compression, 3=mask)
    int freqsep_debug_mode_ = 0;
    bool skin_color_filter_ = false;
public:
    void setFreqSepDebugMode(int mode) { freqsep_debug_mode_ = mode; }
    int getFreqSepDebugMode() const { return freqsep_debug_mode_; }
    void setSkinColorFilter(bool enabled) { skin_color_filter_ = enabled; }
    bool getSkinColorFilter() const { return skin_color_filter_; }

    /// P8-W1: landmark-masked skin smoothing 모드 토글 (internal/벤치용).
    /// 활성 시 기존 FreqSep/Bilateral 스무딩을 대체한다 (다른 패스는 불변).
    /// strength=0 또는 enabled=false면 마스크/블러/필터/타깃 전부 생략 (비용 0).
    void setSkinMaskSmoothing(bool enabled, float strength) {
        skin_mask_smoothing_enabled_ = enabled;
        skin_mask_smoothing_strength_ = (strength < 0.0f) ? 0.0f : (strength > 1.0f ? 1.0f : strength);
    }
    bool getSkinMaskSmoothingEnabled() const { return skin_mask_smoothing_enabled_; }
    float getSkinMaskSmoothingStrength() const { return skin_mask_smoothing_strength_; }
private:

    /// Uniform Location 캐싱 (초기화 시 호출)
    void cacheUniformLocations();

    // 이전 출력 텍스처 추적 (텍스처 풀 관리용)
    TexturePool::TextureInfo* previous_output_ping_ = nullptr;
    TexturePool::TextureInfo* previous_output_pong_ = nullptr;
    GLuint previous_output_texture_ = 0;

    // GPU 동기화 펜스 (glFinish 대체)
#if IRIS_SDK_GPU_AVAILABLE
    GLsync previous_fence_ = nullptr;
#endif

    // Neutral 1x1x1 identity 3D LUT (sampler3D fallback용)
    GLuint neutral_lut_texture_ = 0;

    bool initialized_ = false;
    mutable std::mutex mutex_;
};

} // namespace iris_sdk

#endif // IRIS_SDK_GPU_BEAUTY_BACKEND_H
