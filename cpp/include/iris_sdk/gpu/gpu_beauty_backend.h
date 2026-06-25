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
#include "iris_sdk/warp/jaw_warp_geometry.h"  // P8-W4: JawWarpParams (executeWarpPass)

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
 * **파이프라인 구조** (P8-W2 곁가지 제거 후):
 * - ① landmark-masked skin smoothing (use_skin_mask 채널)
 * - ② Combined Color Pass (brightness 잔존)
 *   (FreqSep/Bilateral/whitening/colorBalance/softFocus/LUT/vivid 곁가지 제거)
 *
 * **Temporal Stability**:
 * - One Euro Filter로 face_rect center jitter 억제 (ROI scissor 경계 안정화)
 *
 * **Thread Safety**: 모든 public 메서드는 mutex_로 보호됩니다.
 *
 * @see DeviceTier, OneEuroFilter
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
    // GPU 디바이스 성능 등급 분류 (단위 테스트 + 분류 유틸 한정)
    //=========================================================================

    /**
     * @brief GPU 디바이스 성능 등급
     *
     * GL_RENDERER 문자열을 파싱하여 결정됩니다 (classifyGpuRenderer()).
     *
     * @note (P8-W2) FreqSep half-res 분기는 곁가지 제거로 사라졌고,
     *       현재는 classifyGpuRenderer() 분류 유틸 + 단위 테스트에서만 사용됩니다.
     *       Mali 분류 비대칭: Adreno는 100 단위 시리즈(6xx/7xx),
     *       Mali-G는 2자리 vs 3자리(G7x/G710+)로 분류 기준이 다름.
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

    /// Temporal filter 일괄 리셋 (release/얼굴 추적 끊김 시 공통 호출)
    void resetTemporalFilters();

    /// [B2 idx20] 이전 프레임이 이월한 ping/pong 텍스처와 GPU fence를 정리한다.
    /// applyTextureId 진입 시(필터 활성/비활성 모두)와 조기 반환 경로에서 공통 호출.
    void releasePreviousFrameResources();

    /// 셰이더 프로그램 초기화
    bool initializeShaders();

    /// 풀스크린 쿼드 VAO/VBO 설정
    void setupFullscreenQuad();

    /// 풀스크린 쿼드 렌더링
    void renderFullscreenQuad();

    //=========================================================================
    // 필터 패스
    //=========================================================================

    /// 밝기 패스
    void executeBrightnessPass(GLuint input_tex, GLuint output_fbo,
                               int width, int height,
                               float brightness);

    /// 마스킹 패스 (ROI, 눈/입술 보호)
    void applyMasking(GLuint filtered_tex, GLuint original_tex,
                      GLuint mask_tex, GLuint output_fbo,
                      int width, int height);

    /// 통합 Color Adjustment 패스 (Brightness)
    /// (P8-W2) whitening/LUT 곁가지 제거 — brightness만 잔존.
    void executeCombinedColorPass(GLuint input_tex, GLuint output_fbo,
                                  int width, int height,
                                  float brightness);

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
    // P8-W4: 턱 V라인 워프 (fragment-direct 비정규 RBF)
    //=========================================================================

    /// 턱 V라인 워프 패스 (풀스크린, 인버스 워프 리샘플).
    /// warp_program_ 사용. params(cx/cy/dx/dy/count/sigma/bounds)는 렌더 텍스처 픽셀 공간으로
    /// 이미 정렬된 상태여야 한다(executeWarpPass는 좌표 변환을 하지 않는다 — 호출부 책임).
    /// params.sigma_px==0 또는 count==0이면 입력을 그대로 복사한다(방어적 패스스루).
    void executeWarpPass(GLuint input_tex, GLuint output_fbo,
                         int width, int height,
                         const iris_sdk::jaw_warp::JawWarpParams& params);

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
    // (P8-W2 제거) 곁가지 프로그램: smoothing(Bilateral)/whitening/color_balance/
    //             soft_focus/freq_sep_gaussian/freq_sep_composite/luminance_sharpen/vivid.
    GLuint passthrough_program_ = 0;
    GLuint brightness_program_ = 0;
    GLuint masking_program_ = 0;
    GLuint combined_color_program_ = 0;  // 통합 Color Adjustment (brightness 잔존)
    GLuint warp_program_ = 0;            // P8-W4: 턱 V라인 워프 (fragment-direct RBF)

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
    // P8-W3: skin 화사함(soft-glow radiance) 강도. skin mask 경로(blur/mask) 공유.
    // smoothing=0이어도 radiance>0이면 skin 경로가 활성화되어 radiance만 단독 적용된다.
    float skin_radiance_strength_ = 0.0f;

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
        GLint compositeRadiance = -1;  // P8-W3: uRadiance
    } skin_uniforms_;

    // [B2 idx18] passthrough 셰이더 uTexture location 캐시
    // (renderSkinBasePasses에서 매 프레임 glGetUniformLocation 호출 제거)
    // maybe_unused: 비-GPU(desktop) 빌드에서는 GL 경로가 컴파일되지 않아 미사용.
    [[maybe_unused]] GLint passthrough_u_texture_ = -1;

    //=========================================================================
    // Uniform Location 캐시 (성능 최적화)
    //=========================================================================

    /// 셰이더별 Uniform Location 캐시 구조체
    /// (P8-W2) 곁가지 제거: Smoothing/Whitening/ColorBalance/SoftFocus/Combined-Whitening/
    ///         Combined-LUT 멤버 삭제. Brightness/Masking/Combined-Brightness 잔존.
    struct UniformLocations {
        // 공통
        GLint uTexture = -1;

        // Brightness
        GLint uBrightness = -1;

        // Masking
        GLint uFiltered = -1;
        GLint uOriginal = -1;
        GLint uMask = -1;

        // Combined Color Adjustment (brightness 잔존)
        GLint uCombinedBrightness = -1;
    };

    /// 프로그램별 Uniform Location 캐시
    UniformLocations brightness_uniforms_;
    UniformLocations masking_uniforms_;
    UniformLocations combined_color_uniforms_;  // 통합 Color Adjustment (brightness)

    /// P8-W4: 턱 V라인 워프 셰이더 uniform location 캐시
    struct WarpUniforms {
        GLint uTexture = -1;
        GLint uWarp = -1;        // vec4[14] 배열 — glUniform4fv(uWarp, count, ...)
        GLint uWarpCount = -1;
        GLint uWarpSigma = -1;
        GLint uWarpBounds = -1;
        GLint uViewportPx = -1;
    } warp_uniforms_;

    // Temporal stability용 One Euro Filter (P4-W3-04)
    // 모든 필터는 mutex_ lock 하에서만 접근 (applyTextureId → public → lock_guard)
    // (P8-W2) skin_radius_filter_(FreqSep blur_radius 안정화)는 곁가지 제거로 dead.
    //   face_rect center 필터는 ROI scissor 경계 안정화에 잔존.
    OneEuroFilter mask_center_x_filter_{1.0f, 0.02f, 1.0f};  ///< face_rect 중심 X 안정화
    OneEuroFilter mask_center_y_filter_{1.0f, 0.02f, 1.0f};  ///< face_rect 중심 Y 안정화

    // (P8-W2 제거) freqsep_debug_mode_ / skin_color_filter_ 필드 + set/get 4종은
    //             FreqSep·색보정 곁가지 제거로 dead. (C API 시그니처는 D단계까지 no-op로 유지)
public:
    /// P8-W1: landmark-masked skin smoothing 모드 토글 (internal/벤치용).
    /// 활성 시 스무딩을 적용한다 (다른 패스는 불변).
    /// strength=0 또는 enabled=false면 마스크/블러/필터/타깃 전부 생략 (비용 0).
    void setSkinMaskSmoothing(bool enabled, float strength) {
        skin_mask_smoothing_enabled_ = enabled;
        skin_mask_smoothing_strength_ = (strength < 0.0f) ? 0.0f : (strength > 1.0f ? 1.0f : strength);
    }
    bool getSkinMaskSmoothingEnabled() const { return skin_mask_smoothing_enabled_; }
    float getSkinMaskSmoothingStrength() const { return skin_mask_smoothing_strength_; }

    /// P8-W3: skin 화사함(soft-glow radiance) 강도 설정 (internal/벤치용).
    /// skin mask 경로(blur/mask)를 공유한다. smoothing=0이어도 radiance>0이면
    /// skin 경로가 활성화되어 radiance만 단독으로 적용된다(윤기/화사 단독 가능).
    /// strength=0이면 radiance 블록은 생략된다(skin 경로 자체는 smoothing 조건에 따름).
    void setSkinRadiance(float strength) {
        // setSkinMaskSmoothing와 동일한 수동 clamp 관용구 (헤더에 <algorithm> 미포함).
        skin_radiance_strength_ = (strength < 0.0f) ? 0.0f : (strength > 1.0f ? 1.0f : strength);
    }
    float getSkinRadianceStrength() const { return skin_radiance_strength_; }
private:

    /// Uniform Location 캐싱 (초기화 시 호출)
    void cacheUniformLocations();

    // 이전 출력 텍스처 추적 (텍스처 풀 관리용)
    TexturePool::TextureInfo* previous_output_ping_ = nullptr;
    TexturePool::TextureInfo* previous_output_pong_ = nullptr;

    // [B2 idx2] applyTexture(TextureHandle) 출력 핸들이 가리킬 안정적 GLuint 저장소.
    // 풀 내부 TextureInfo 멤버 주소를 직접 노출하면 trim()이 unique_ptr을 파괴할 때
    // 댕글링 포인터가 되므로, 백엔드 수명에 묶인 멤버 버퍼의 주소를 노출한다.
    // maybe_unused: 비-GPU(desktop) 빌드에서는 GL 경로가 컴파일되지 않아 미사용.
    [[maybe_unused]] GLuint applytexture_output_id_ = 0;

    // GPU 동기화 펜스 (glFinish 대체)
#if IRIS_SDK_GPU_AVAILABLE
    GLsync previous_fence_ = nullptr;
#endif

    // (P8-W2 제거) neutral_lut_texture_ (sampler3D LUT fallback)는 LUT 곁가지 제거로 dead.

    bool initialized_ = false;
    mutable std::mutex mutex_;
};

} // namespace iris_sdk

#endif // IRIS_SDK_GPU_BEAUTY_BACKEND_H
