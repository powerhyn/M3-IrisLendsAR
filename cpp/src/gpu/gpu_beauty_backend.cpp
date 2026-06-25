/**
 * @file gpu_beauty_backend.cpp
 * @brief GPUBeautyBackend 구현
 */

#include "iris_sdk/gpu/gpu_beauty_backend.h"
#include "iris_sdk/gpu/render_context.h"
#include "iris_sdk/gpu/skin_mask_geometry.h"
#include "iris_sdk/beauty_roi_manager.h"
#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <string>

#if IRIS_SDK_GPU_AVAILABLE
#include "iris_sdk/gpu/gles_render_context.h"
#include <android/log.h>
#define LOG_TAG "GPUBeautyBackend"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#include <cstdio>
#define LOGD(...) do { printf("[GPUBeautyBackend DEBUG] " __VA_ARGS__); printf("\n"); } while(0)
#define LOGI(...) do { printf("[GPUBeautyBackend INFO] " __VA_ARGS__); printf("\n"); } while(0)
#define LOGW(...) do { printf("[GPUBeautyBackend WARN] " __VA_ARGS__); printf("\n"); } while(0)
#define LOGE(...) do { printf("[GPUBeautyBackend ERROR] " __VA_ARGS__); printf("\n"); } while(0)
#endif

namespace iris_sdk {

#if IRIS_SDK_GPU_AVAILABLE
// (P8-W2 제거) computeGaussianWeights / computeFallbackSmoothing 는 FreqSep 전용 → dead.

static BeautyFilterConfigV2 buildEffectiveConfig(const BeautyFilterConfigV2& config) {
    BeautyFilterConfigV2 effective = config;

    if (!config.enabled) {
        // beauty 비활성 경로. 잔존 패스가 읽는 값만 중립화하면 충분.
        // (P8-W2) smoothing/softFocus/whitening/colorBalance 등 곁가지 필드는
        // 더 이상 읽히지 않으므로 brightness만 중립값으로 둔다(필드 자체는 D단계까지 보존).
        effective.brightness = 1.0f;
        return effective;
    }

    const float master = std::clamp(config.intensity, 0.0f, 1.0f);

    // (P8-W2) 잔존 패스가 읽는 brightness만 마스터 강도로 스케일.
    effective.brightness = 1.0f + (config.brightness - 1.0f) * master;

    return effective;
}

// P8-W4: computeJawWarp 출력(원본 비미러·top-down 이미지 픽셀 공간)을 렌더 텍스처
// 픽셀 공간(전면 미러·Y-flip 적용)으로 정렬한다. prepareSkinFans 좌표 정합과 동형:
//   prepareSkinFans: 제어점 px = (1 - x)·W (미러), py = y·H, NDC ny = 1 - 2y (Y-flip).
//   ⇒ 렌더 텍스처 px = (1 - x)·W = W − cx,  py(vTexCoord) = (1 - y)·H = H − cy.
// 변위는 선형변환 (x→W−x, y→H−y) 하에서 dx'=−dx, dy'=−dy. σ는 등방·거리불변이라 유지.
// bbox는 X·Y가 뒤집히므로 min/max를 교차해 다시 정렬한다.
// ⚠️ 이 변환이 prepareSkinFans 와 어긋나면 워프가 엉뚱한 위치/렌즈 불일치를 일으킨다.
static iris_sdk::jaw_warp::JawWarpParams alignWarpToRenderSpace(
    const iris_sdk::jaw_warp::JawWarpParams& in, int width, int height) {
    iris_sdk::jaw_warp::JawWarpParams out = in;
    const float fw = static_cast<float>(width);
    const float fh = static_cast<float>(height);
    const int count = (in.count < iris_sdk::jaw_warp::JawWarpParams::kMaxControlPoints)
                          ? in.count
                          : iris_sdk::jaw_warp::JawWarpParams::kMaxControlPoints;
    for (int i = 0; i < count; ++i) {
        out.cx[i] = fw - in.cx[i];   // 미러 X (prepareSkinFans mx = (1-x)·W)
        out.cy[i] = fh - in.cy[i];   // Y-flip (vTexCoord.y = 1 - y)
        out.dx[i] = -in.dx[i];       // 미러가 X 부호 반전
        out.dy[i] = -in.dy[i];       // Y-flip이 Y 부호 반전
    }
    // bbox: X·Y가 각각 뒤집히므로 min↔max 교차.
    out.bounds_min_x = fw - in.bounds_max_x;
    out.bounds_max_x = fw - in.bounds_min_x;
    out.bounds_min_y = fh - in.bounds_max_y;
    out.bounds_max_y = fh - in.bounds_min_y;
    // sigma_px / count 는 유지.
    return out;
}
#endif

// 셰이더 소스 extern 선언
// (P8-W2 제거) Bilateral/Whitening/ColorBalance/SoftFocus/FreqSep/Vivid 곁가지 셰이더 extern 삭제.
namespace shaders {
extern const char* FULLSCREEN_QUAD_VERTEX;
extern const char* PASSTHROUGH_FRAGMENT;
// (dead 정리) BRIGHTNESS_FRAGMENT / MASKING_FRAGMENT extern 제거 — 호출 0.
//             brightness는 COMBINED_COLOR_ADJUSTMENT가 담당, masking 패스는 미사용.
extern const char* COMBINED_COLOR_ADJUSTMENT_FRAGMENT;
// P8-W1: landmark-masked skin smoothing
extern const char* SKIN_MASK_FILL_VERTEX;
extern const char* SKIN_MASK_FILL_FRAGMENT;
extern const char* SKIN_SEPARABLE_BLUR_FRAGMENT;
extern const char* SKIN_SMOOTH_COMPOSITE_FRAGMENT;
// P8-W4: 턱 V라인 워프
extern const char* WARP_FRAGMENT;
}

GPUBeautyBackend::GPUBeautyBackend() = default;

GPUBeautyBackend::~GPUBeautyBackend() {
    release();
}

bool GPUBeautyBackend::initialize(IRenderContext* render_context) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (initialized_) {
        LOGW("GPUBeautyBackend already initialized");
        return true;
    }

#if IRIS_SDK_GPU_AVAILABLE
    if (render_context) {
        // GLESRenderContext로 다운캐스트
        render_context_ = dynamic_cast<GLESRenderContext*>(render_context);
        if (!render_context_) {
            LOGE("RenderContext is not GLESRenderContext");
            return false;
        }

        // GL 컨텍스트 활성화
        if (!render_context_->makeCurrent()) {
            LOGE("Failed to make GL context current");
            return false;
        }
    } else {
        // render_context가 null일 경우, 현재 스레드의 EGL 컨텍스트 사용 (Android GLSurfaceView)
        render_context_ = nullptr;
        LOGI("Using current thread's EGL context (GLSurfaceView mode)");
    }
#else
    render_context_ = nullptr;
    LOGI("Running in desktop stub mode");
#endif

    // 셰이더 매니저 초기화
    shader_manager_ = std::make_unique<ShaderManager>();
    if (!initializeShaders()) {
        LOGE("Failed to initialize shaders");
        shader_manager_.reset();
        return false;
    }

    // 텍스처 풀 초기화 (최대 8개, 1920x1080)
    texture_pool_ = std::make_unique<TexturePool>();
    if (!texture_pool_->initialize(8, 1920, 1080)) {
        LOGE("Failed to initialize texture pool");
        shader_manager_->releaseAll();
        shader_manager_.reset();
        texture_pool_.reset();
        return false;
    }

    // GPU 프로파일러 초기화
    profiler_ = std::make_unique<GPUProfiler>();
    if (profiler_->initialize()) {
        profiler_->setEnabled(false);  // 기본 비활성화 (성능 영향)
        LOGI("GPU profiler initialized (disabled by default)");
    } else {
        LOGW("GPU profiler not supported on this device");
    }

    // 풀스크린 쿼드 설정
    setupFullscreenQuad();

    // Uniform Location 캐싱 (성능 최적화)
    cacheUniformLocations();

    // (P8-W2 제거) Neutral 1x1x1 identity 3D LUT 생성 + 디바이스 티어 감지(FreqSep half-res
    //             분기용)는 LUT/FreqSep 곁가지 제거로 dead.

    // P8-W1 §5: landmark One-Euro 픽셀 공간 파라미터 (min_cutoff 0.5 / beta 0.007 / d_cutoff 1.0).
    // 기본 생성자는 min_cutoff=1.0 이므로 0.5로 명시 설정.
    for (auto& f : skin_landmark_filters_) {
        f.setMinCutoff(0.5f);
        f.setBeta(0.007f);
        f.setDCutoff(1.0f);
    }

    initialized_ = true;
    LOGI("GPUBeautyBackend initialized successfully");
    return true;
}

bool GPUBeautyBackend::initializeShaders() {
    // 패스스루 (디버그/테스트용)
    if (!shader_manager_->createProgram(
            shaders::FULLSCREEN_QUAD_VERTEX,
            shaders::PASSTHROUGH_FRAGMENT,
            passthrough_program_)) {
        LOGE("Failed to create passthrough program");
        return false;
    }
    shader_manager_->cacheProgram("passthrough", passthrough_program_);

    // (P8-W2 제거) 곁가지 프로그램: smoothing(Bilateral)/whitening/color_balance/soft_focus.
    // (dead 정리) brightness/masking 프로그램 컴파일 제거 — 호출 0.
    //             brightness는 아래 combined_color 패스가 담당, masking 패스는 미사용.

    // 통합 Color Adjustment (Brightness + ColorBalance + Whitening)
    if (!shader_manager_->createProgram(
            shaders::FULLSCREEN_QUAD_VERTEX,
            shaders::COMBINED_COLOR_ADJUSTMENT_FRAGMENT,
            combined_color_program_)) {
        LOGE("Failed to create combined_color program");
        return false;
    }
    shader_manager_->cacheProgram("combined_color", combined_color_program_);

    // (P8-W2 제거) Vivid 포스트프로세싱 / Frequency Separation 셰이더 곁가지 제거.

    // P8-W1: landmark-masked skin smoothing 셰이더 (non-fatal — OFF 시 영향 없음)
    if (!initializeSkinSmoothingShaders()) {
        LOGW("Failed to create skin-mask smoothing shaders (non-fatal)");
    }

    // P8-W4: 턱 V라인 워프 셰이더 (non-fatal — OFF 시 영향 없음, gate가 program!=0 확인)
    if (!shader_manager_->createProgram(
            shaders::FULLSCREEN_QUAD_VERTEX,
            shaders::WARP_FRAGMENT,
            warp_program_)) {
        LOGW("Failed to create warp program (non-fatal)");
        warp_program_ = 0;
    } else {
        shader_manager_->cacheProgram("warp", warp_program_);
    }

    LOGI("All %zu shader programs created successfully",
         shader_manager_->getCachedProgramCount());
    return true;
}

bool GPUBeautyBackend::initializeSkinSmoothingShaders() {
    // 마스크 채움 (전용 VS: position만, UV 없음)
    if (!shader_manager_->createProgram(
            shaders::SKIN_MASK_FILL_VERTEX,
            shaders::SKIN_MASK_FILL_FRAGMENT,
            skin_mask_fill_program_)) {
        LOGE("Failed to create skin_mask_fill program");
        return false;
    }
    shader_manager_->cacheProgram("skin_mask_fill", skin_mask_fill_program_);

    // 분리형 가우시안 블러 (풀스크린 쿼드 VS 재사용)
    if (!shader_manager_->createProgram(
            shaders::FULLSCREEN_QUAD_VERTEX,
            shaders::SKIN_SEPARABLE_BLUR_FRAGMENT,
            skin_blur_program_)) {
        LOGE("Failed to create skin_blur program");
        return false;
    }
    shader_manager_->cacheProgram("skin_blur", skin_blur_program_);

    // 에지 가드 컴포지트
    if (!shader_manager_->createProgram(
            shaders::FULLSCREEN_QUAD_VERTEX,
            shaders::SKIN_SMOOTH_COMPOSITE_FRAGMENT,
            skin_composite_program_)) {
        LOGE("Failed to create skin_composite program");
        return false;
    }
    shader_manager_->cacheProgram("skin_composite", skin_composite_program_);

    LOGI("Skin-mask smoothing shaders created successfully");
    return true;
}

// (P8-W2 제거) initializeFreqSepShaders() — FreqSep 곁가지 전체 제거.

void GPUBeautyBackend::cacheUniformLocations() {
#if IRIS_SDK_GPU_AVAILABLE
    // Passthrough Uniforms (B2 idx18: renderSkinBasePasses 매 프레임 조회 제거)
    if (passthrough_program_ != 0) {
        passthrough_u_texture_ = glGetUniformLocation(passthrough_program_, "uTexture");
    }

    // (P8-W2 제거) Smoothing/Whitening/ColorBalance/SoftFocus/Vivid/FreqSep/Sharpen 유니폼 캐시.
    // (dead 정리) brightness/masking 유니폼 캐시 제거 — 해당 프로그램/패스가 dead.

    // Combined Color Adjustment Uniforms (brightness 잔존)
    combined_color_uniforms_.uTexture = glGetUniformLocation(combined_color_program_, "uTexture");
    combined_color_uniforms_.uCombinedBrightness = glGetUniformLocation(combined_color_program_, "uBrightness");

    // P8-W4: 턱 V라인 워프 Uniforms
    if (warp_program_ != 0) {
        warp_uniforms_.uTexture = glGetUniformLocation(warp_program_, "uTexture");
        warp_uniforms_.uWarp = glGetUniformLocation(warp_program_, "uWarp");
        warp_uniforms_.uWarpCount = glGetUniformLocation(warp_program_, "uWarpCount");
        warp_uniforms_.uWarpSigma = glGetUniformLocation(warp_program_, "uWarpSigma");
        warp_uniforms_.uWarpBounds = glGetUniformLocation(warp_program_, "uWarpBounds");
        warp_uniforms_.uViewportPx = glGetUniformLocation(warp_program_, "uViewportPx");
    }

    // P8-W1: Skin-mask smoothing Uniforms
    if (skin_mask_fill_program_ != 0) {
        skin_uniforms_.maskFillValue = glGetUniformLocation(skin_mask_fill_program_, "uValue");
    }
    if (skin_blur_program_ != 0) {
        skin_uniforms_.blurTexture = glGetUniformLocation(skin_blur_program_, "uTexture");
        skin_uniforms_.blurDirection = glGetUniformLocation(skin_blur_program_, "uDirection");
        skin_uniforms_.blurOffsetScale = glGetUniformLocation(skin_blur_program_, "uOffsetScale");
    }
    if (skin_composite_program_ != 0) {
        skin_uniforms_.compositeTexture = glGetUniformLocation(skin_composite_program_, "uTexture");
        skin_uniforms_.compositeBlurTex = glGetUniformLocation(skin_composite_program_, "uBlurTex");
        skin_uniforms_.compositeMaskTex = glGetUniformLocation(skin_composite_program_, "uSkinMaskTex");
        skin_uniforms_.compositeSkin = glGetUniformLocation(skin_composite_program_, "uSkin");
        skin_uniforms_.compositeRadiance = glGetUniformLocation(skin_composite_program_, "uRadiance");  // P8-W3
    }

    LOGI("Uniform locations cached successfully");
#endif
}

void GPUBeautyBackend::setupFullscreenQuad() {
#if IRIS_SDK_GPU_AVAILABLE
    // 풀스크린 쿼드 정점 데이터
    // position (x, y), texcoord (u, v)
    float quad_vertices[] = {
        // Position    // TexCoord
        -1.0f,  1.0f,  0.0f, 1.0f,  // 좌상
        -1.0f, -1.0f,  0.0f, 0.0f,  // 좌하
         1.0f, -1.0f,  1.0f, 0.0f,  // 우하

        -1.0f,  1.0f,  0.0f, 1.0f,  // 좌상
         1.0f, -1.0f,  1.0f, 0.0f,  // 우하
         1.0f,  1.0f,  1.0f, 1.0f   // 우상
    };

    glGenVertexArrays(1, &quad_vao_);
    glGenBuffers(1, &quad_vbo_);

    glBindVertexArray(quad_vao_);
    glBindBuffer(GL_ARRAY_BUFFER, quad_vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad_vertices), quad_vertices, GL_STATIC_DRAW);

    // position attribute (location = 0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // texcoord attribute (location = 1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                          (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    LOGI("Fullscreen quad VAO=%u, VBO=%u", quad_vao_, quad_vbo_);
#else
    quad_vao_ = 1;
    quad_vbo_ = 1;
    LOGI("Fullscreen quad created (stub)");
#endif
}

void GPUBeautyBackend::renderFullscreenQuad() {
#if IRIS_SDK_GPU_AVAILABLE
    glBindVertexArray(quad_vao_);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
#endif
}

void GPUBeautyBackend::release() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_) {
        return;
    }

#if IRIS_SDK_GPU_AVAILABLE
    if (render_context_) {
        render_context_->makeCurrent();
    }

    // 쿼드 VAO/VBO 해제
    if (quad_vao_ != 0) {
        glDeleteVertexArrays(1, &quad_vao_);
        quad_vao_ = 0;
    }
    if (quad_vbo_ != 0) {
        glDeleteBuffers(1, &quad_vbo_);
        quad_vbo_ = 0;
    }
#endif

#if IRIS_SDK_GPU_AVAILABLE
    // GPU 펜스 정리
    if (previous_fence_ != nullptr) {
        glDeleteSync(previous_fence_);
        previous_fence_ = nullptr;
    }

    // (P8-W2 제거) neutral LUT / FreqSep skin mask 텍스처 해제 — 곁가지 제거로 dead.

    // P8-W1: landmark-masked smoothing 타깃 해제
    destroySkinTargets();
    skin_targets_failed_ = false;
#endif

    // 셰이더 해제
    if (shader_manager_) {
        shader_manager_->releaseAll();
        shader_manager_.reset();
    }

    // 텍스처 풀 해제
    if (texture_pool_) {
        texture_pool_->release();
        texture_pool_.reset();
    }

    // 프로파일러 해제
    if (profiler_) {
        profiler_->release();
        profiler_.reset();
    }

    render_context_ = nullptr;
    initialized_ = false;

    resetTemporalFilters();

    // 프로그램 ID 초기화 ((P8-W2) 곁가지 프로그램 제거 — 잔존 2종 + skin 3종)
    // (dead 정리) brightness/masking program 리셋 제거 — 컴파일 자체가 사라짐.
    passthrough_program_ = 0;
    combined_color_program_ = 0;
    warp_program_ = 0;  // P8-W4
    skin_mask_fill_program_ = 0;
    skin_blur_program_ = 0;
    skin_composite_program_ = 0;

    LOGI("GPUBeautyBackend released");
}

void GPUBeautyBackend::resetTemporalFilters() {
    // (P8-W2) skin_radius_filter_(FreqSep blur_radius)는 dead — face_rect center만 리셋.
    mask_center_x_filter_.reset();
    mask_center_y_filter_.reset();
}

void GPUBeautyBackend::releasePreviousFrameResources() {
    // [B2 idx20] 이전 프레임이 이월한 출력 ping/pong과 GPU fence를 정리한다.
    // fence는 GL 객체이므로 GPU 가드로 분리하고, 풀 텍스처 반환은 GL 무관 로직이다.
#if IRIS_SDK_GPU_AVAILABLE
    if (previous_output_ping_ != nullptr || previous_output_pong_ != nullptr) {
        if (previous_fence_ != nullptr) {
            // 펜스가 시그널될 때까지 대기 (최대 16ms = 1프레임)
            GLenum waitResult = glClientWaitSync(previous_fence_, GL_SYNC_FLUSH_COMMANDS_BIT, 16000000);
            if (waitResult == GL_TIMEOUT_EXPIRED) {
                LOGW("GPU fence wait timeout - previous frame still rendering");
            }
            glDeleteSync(previous_fence_);
            previous_fence_ = nullptr;
        }
    }
#endif
    if (texture_pool_) {
        if (previous_output_ping_ != nullptr) {
            texture_pool_->releaseTexture(previous_output_ping_);
        }
        if (previous_output_pong_ != nullptr) {
            texture_pool_->releaseTexture(previous_output_pong_);
        }
    }
    previous_output_ping_ = nullptr;
    previous_output_pong_ = nullptr;
}

bool GPUBeautyBackend::isInitialized() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return initialized_;
}

IrisSdkError GPUBeautyBackend::apply(
    uint8_t* frame_data,
    int width, int height,
    IrisFrameFormat format,
    const BeautyFilterConfigV2& config,
    const BeautyROI* roi) {

    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_) {
        return IRIS_SDK_NOT_INITIALIZED;
    }

    if (!frame_data || width <= 0 || height <= 0) {
        return IRIS_SDK_INVALID_PARAM;
    }

    if (!config.enabled) {
        return IRIS_SDK_OK;  // 비활성화 시 아무 작업 없음
    }

    // CPU 버퍼 경로는 미지원 — CPUBeautyBackend를 사용해야 함
    LOGW("GPUBeautyBackend::apply() CPU buffer path not supported. Use CPUBeautyBackend instead.");

    (void)format;
    (void)roi;

    return IRIS_SDK_ERROR_NOT_SUPPORTED;
}

IrisSdkError GPUBeautyBackend::applyTexture(
    const TextureHandle& input,
    TextureHandle& output,
    const BeautyFilterConfigV2& config,
    const BeautyROI* roi) {

    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_) {
        return IRIS_SDK_NOT_INITIALIZED;
    }

    if (!input.isValid() || input.type != TextureHandle::Type::OpenGLES) {
        LOGE("Invalid input texture");
        return IRIS_SDK_INVALID_PARAM;
    }

    // (P8-W2) vivid 곁가지 제거 — beauty enabled 게이트로 단순화.
    if (!config.enabled) {
        output = input;
        return IRIS_SDK_OK;
    }

#if IRIS_SDK_GPU_AVAILABLE
    // GLSurfaceView 모드에서는 이미 EGL 컨텍스트가 바인딩되어 있음
    if (render_context_) {
        render_context_->makeCurrent();
    }

    int width = input.width;
    int height = input.height;

    // native_handle는 GLuint* 타입
    GLuint input_tex = *static_cast<GLuint*>(input.native_handle);
    BeautyFilterConfigV2 effective_config = buildEffectiveConfig(config);

    // [B2 idx2] 이전 프레임의 출력 ping/pong을 먼저 반환한다.
    // 이전 구현은 acquirePingPongPair만 하고 release/이월이 전혀 없어 4회 호출 후
    // 풀(최대 8개)이 영구 고갈되었다(applyTextureId와 동일한 이월 반환 패턴 적용).
    if (previous_output_ping_ != nullptr) {
        texture_pool_->releaseTexture(previous_output_ping_);
        previous_output_ping_ = nullptr;
    }
    if (previous_output_pong_ != nullptr) {
        texture_pool_->releaseTexture(previous_output_pong_);
        previous_output_pong_ = nullptr;
    }

    // [B2 idx5] portrait/고해상도 입력에서도 silent 실패하지 않도록 상한 보장.
    texture_pool_->ensureCapacity(width, height);

    // Ping-Pong 버퍼 획득
    TexturePool::TextureInfo* ping = nullptr;
    TexturePool::TextureInfo* pong = nullptr;
    if (!texture_pool_->acquirePingPongPair(width, height, ping, pong)) {
        LOGE("Failed to acquire ping-pong buffers");
        return IRIS_SDK_UNKNOWN;
    }

    GLuint current_input = input_tex;
    TexturePool::TextureInfo* current_output = ping;

    // 뷰포트 설정
    glViewport(0, 0, width, height);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);

    // 필터 체인 실행 (최적화됨 + 프로파일링)
    bool profiling = profiler_ && profiler_->isEnabled();

    // (P8-W2) 곁가지 제거: Bilateral 스무딩 / SoftFocus / Vivid / whitening·colorBalance.
    // 통합 Color Adjustment (brightness 잔존)
    bool needsBrightness = std::abs(effective_config.brightness - 1.0f) > 0.01f;
    if (needsBrightness) {
        if (profiling) profiler_->begin("CombinedColor");
        executeCombinedColorPass(current_input, current_output->fbo_id,
                                 width, height,
                                 effective_config.brightness);
        if (profiling) profiler_->end("CombinedColor");
        current_input = current_output->texture_id;
        current_output = (current_output == ping) ? pong : ping;
    }

    // 출력 텍스처 핸들 설정
    if (current_input == input_tex) {
        // 아무 패스도 실행되지 않음 — 입력을 그대로 반환.
        // 이 경우 ping/pong은 이번 프레임에 쓰이지 않았으므로 즉시 반환한다.
        output = input;
        texture_pool_->releaseTexture(ping);
        texture_pool_->releaseTexture(pong);
        ping = nullptr;
        pong = nullptr;
    } else {
        // [B2 idx2] 풀 내부 TextureInfo 멤버 주소 노출(댕글링 위험) 대신
        // 백엔드 수명에 묶인 멤버 버퍼에 GLuint 값을 복사하고 그 주소를 노출한다.
        applytexture_output_id_ = current_input;
        output.native_handle = &applytexture_output_id_;
        output.type = TextureHandle::Type::OpenGLES;
        output.width = width;
        output.height = height;
        output.format = input.format;
        // 출력 텍스처가 살아있도록 ping/pong을 다음 프레임에 반환(이월).
        previous_output_ping_ = ping;
        previous_output_pong_ = pong;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // 프레임 종료 처리 (프로파일링 결과 수집)
    if (profiling) {
        profiler_->frameEnd();
    }

    (void)roi;  // ROI 마스킹은 후속 작업에서 구현
#else
    // Desktop 스텁
    output = input;
    (void)roi;
#endif

    return IRIS_SDK_OK;
}

// (P8-W2 제거) executeSmoothingPass(Bilateral) / executeWhiteningPass /
//             executeColorBalancePass / executeSoftFocusPass — 곁가지 패스 전체 제거.

// (dead 정리) executeBrightnessPass / applyMasking 제거 — 호출 0.
//             brightness는 executeCombinedColorPass(COMBINED_COLOR_ADJUSTMENT)가 담당,
//             masking 패스는 어디서도 호출되지 않음.
// (P8-W2 제거) executeVividPass — Vivid 곁가지 패스 제거.

void GPUBeautyBackend::executeCombinedColorPass(
    GLuint input_tex, GLuint output_fbo,
    int width, int height,
    float brightness) {

#if IRIS_SDK_GPU_AVAILABLE
    glBindFramebuffer(GL_FRAMEBUFFER, output_fbo);

#ifndef NDEBUG
    LOGI("CombinedColor: program=%u, fbo=%u, input=%u, brightness=%.2f",
         combined_color_program_, output_fbo, input_tex, brightness);

    // FBO Completeness 체크 (디버깅)
    GLenum fboStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (fboStatus != GL_FRAMEBUFFER_COMPLETE) {
        LOGE("CombinedColor: FBO incomplete! status=0x%x", fboStatus);
        return;
    }

    // 텍스처 유효성 확인
    GLboolean isValidTex = glIsTexture(input_tex);
    LOGI("CombinedColor: input_tex=%u valid=%d", input_tex, isValidTex);
#endif

    glUseProgram(combined_color_program_);

#ifndef NDEBUG
    // 프로그램 링크 상태 확인
    GLint linkStatus;
    glGetProgramiv(combined_color_program_, GL_LINK_STATUS, &linkStatus);
    if (linkStatus != GL_TRUE) {
        LOGE("CombinedColor: program link failed!");
        return;
    }
#endif

    // 캐시된 Uniform Location 사용 ((P8-W2) balance/whitening/LUT 곁가지 제거 — brightness 잔존)
    glUniform1i(combined_color_uniforms_.uTexture, 0);
    glUniform1f(combined_color_uniforms_.uCombinedBrightness, brightness);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, input_tex);

    renderFullscreenQuad();

#ifndef NDEBUG
    // 렌더링 후 에러 체크
    GLenum glErr = glGetError();
    if (glErr != GL_NO_ERROR) {
        LOGE("CombinedColor: GL error after render: 0x%x", glErr);
    }
#endif

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
#else
    (void)input_tex;
    (void)output_fbo;
    (void)width;
    (void)height;
    (void)brightness;
#endif
}

// (P8-W2 제거) uploadSkinMask / mapSkinQuality / mapSmoothingAndPore /
//             executeSmoothingWithFallbackStrength / executeFreqSepPipeline —
//             FreqSep 곁가지 전체 제거.

// =============================================================================
// Device Tier 분류 (classifyGpuRenderer — 단위 테스트 + 분류 유틸 한정으로 보존)
// =============================================================================

GPUBeautyBackend::DeviceTier
GPUBeautyBackend::classifyGpuRenderer(const std::string& gpu) {
    if (gpu.empty()) return DeviceTier::LOW;

    // Adreno GPU (e.g. "Adreno (TM) 750", "Adreno 640")
    if (gpu.find("Adreno") != std::string::npos) {
        auto pos = gpu.find("Adreno") + 6; // "Adreno" 이후
        // 비숫자 문자 스킵 (공백, "(TM)" 등)
        while (pos < gpu.size() && !std::isdigit(static_cast<unsigned char>(gpu[pos]))) ++pos;
        // 첫 번째 연속 숫자 블록만 수집
        std::string digits;
        while (pos < gpu.size() && std::isdigit(static_cast<unsigned char>(gpu[pos]))) {
            digits += gpu[pos++];
        }
        if (!digits.empty()) {
            long val = std::strtol(digits.c_str(), nullptr, 10);
            if (val > 0 && val <= 99999) {
                int num = static_cast<int>(val);
                if (num >= 700) return DeviceTier::HIGH;
                if (num >= 600) return DeviceTier::MID;
                return DeviceTier::LOW;
            }
        }
        // "Adreno" 키워드가 있지만 숫자가 없으면 LOW
        return DeviceTier::LOW;
    }

    // Mali GPU
    // NOTE: Adreno와 분류 기준이 비대칭적임.
    //   Adreno: 100 단위 시리즈 (6xx=MID, 7xx=HIGH)
    //   Mali-G: 2자리 vs 3자리 모델 번호 (G7x=MID, G710+=HIGH)
    //   예) Mali-G78(MID) vs Mali-G710(HIGH) — G710은 Valhall 아키텍처 전환 세대
    if (gpu.find("Mali-G") != std::string::npos) {
        auto pos = gpu.find("Mali-G") + 6;
        std::string digits;
        for (size_t i = pos; i < gpu.size() && std::isdigit(static_cast<unsigned char>(gpu[i])); ++i) {
            digits += gpu[i];
        }
        if (!digits.empty()) {
            long val = std::strtol(digits.c_str(), nullptr, 10);
            if (val > 0 && val <= 99999) {
                int num = static_cast<int>(val);
                if (num >= 710) return DeviceTier::HIGH;  // Valhall+: G710, G715, G720
                if (num >= 70) return DeviceTier::MID;    // Bifrost/Valhall: G71~G78
                return DeviceTier::LOW;
            }
        }
        // "Mali-G" 키워드가 있지만 숫자가 없으면 LOW
        return DeviceTier::LOW;
    }

    // Apple GPU → HIGH
    if (gpu.find("Apple") != std::string::npos) return DeviceTier::HIGH;
    // PowerVR → MID
    if (gpu.find("PowerVR") != std::string::npos) return DeviceTier::MID;

    // Desktop GPU → HIGH
    if (gpu.find("NVIDIA") != std::string::npos ||
        gpu.find("AMD") != std::string::npos ||
        gpu.find("Intel") != std::string::npos) {
        return DeviceTier::HIGH;
    }

    return DeviceTier::LOW;
}

// (P8-W2 제거) detectDeviceTier() / executeFreqSepPipelineHalfRes() /
//             executeFreqSepPipelineImpl() — FreqSep half-res 분기 + 공통 구현 제거.

void GPUBeautyBackend::onMemoryPressure(int level) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (texture_pool_) {
        texture_pool_->onMemoryPressure(level);
    }
}

void GPUBeautyBackend::setProfilingEnabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (profiler_) {
        profiler_->setEnabled(enabled);
        LOGI("GPU profiling %s", enabled ? "enabled" : "disabled");
    }
}

bool GPUBeautyBackend::isProfilingEnabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return profiler_ && profiler_->isEnabled();
}

std::string GPUBeautyBackend::getProfilingReport() const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (profiler_) {
        return profiler_->generateReport();
    }
    return "GPU profiler not available";
}

//=============================================================================
// P8-W1: landmark-masked skin smoothing (LensSimulator 이식)
//=============================================================================

// P8-W3: skin mask 경로(smoothing + radiance) 활성 여부.
// 독립 게이팅 — (smoothing 모드 enabled & strength>0) OR (radiance>0) 중 하나라도 켜지면
// 마스크/블러 base 패스 + composite를 돌린다. enabled_ 토글은 **smoothing만** 게이트하고,
// radiance는 strength>0으로 독립 활성(setSkinRadiance가 strength만 설정 — enabled 무관).
// (smoothing off & radiance>0이면 composite의 uSkin=0으로 스무딩 mix가 no-op, radiance 블록만 적용.)
bool GPUBeautyBackend::skinMaskSmoothingActive(const IrisResult* detection) const {
    return ((skin_mask_smoothing_enabled_ && skin_mask_smoothing_strength_ > 0.0f)
            || skin_radiance_strength_ > 0.0f)
        && detection != nullptr
        && detection->detected
        && detection->face_mesh_valid
        && skin_mask_fill_program_ != 0
        && skin_blur_program_ != 0
        && skin_composite_program_ != 0;
}

bool GPUBeautyBackend::ensureSkinTargets(int width, int height) {
#if IRIS_SDK_GPU_AVAILABLE
    if (skin_targets_failed_) return false;

    const int target_w = std::max(1, width / 4);
    const int target_h = std::max(1, height / 4);
    // 동일 크기면 재생성 생략 (원본 createBeautyTargets 크기 가드)
    if (skin_targets_ready_ && skin_low_w_ == target_w && skin_low_h_ == target_h) {
        return true;
    }
    destroySkinTargets();
    skin_low_w_ = target_w;
    skin_low_h_ = target_h;

    glGenTextures(kSkinTargetCount, skin_target_tex_);
    glGenFramebuffers(kSkinTargetCount, skin_target_fbo_);
    for (int i = 0; i < kSkinTargetCount; ++i) {
        const bool r8 = (i >= kSkinMask);  // 마스크 계열은 단일 채널 (원본과 동일)
        glBindTexture(GL_TEXTURE_2D, skin_target_tex_[i]);
        glTexImage2D(GL_TEXTURE_2D, 0,
                     r8 ? GL_R8 : GL_RGBA8, skin_low_w_, skin_low_h_, 0,
                     r8 ? GL_RED : GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindFramebuffer(GL_FRAMEBUFFER, skin_target_fbo_[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, skin_target_tex_[i], 0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            LOGE("Skin smoothing FBO incomplete (index=%d) — disabling mode", i);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            destroySkinTargets();
            skin_targets_failed_ = true;  // 매 프레임 재시도 방지 (원본 ensureBeautyTargets)
            return false;
        }
    }
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    skin_targets_ready_ = true;
    return true;
#else
    (void)width; (void)height;
    return false;
#endif
}

void GPUBeautyBackend::destroySkinTargets() {
#if IRIS_SDK_GPU_AVAILABLE
    if (skin_target_tex_[0] != 0 || skin_target_fbo_[0] != 0) {
        glDeleteFramebuffers(kSkinTargetCount, skin_target_fbo_);
        glDeleteTextures(kSkinTargetCount, skin_target_tex_);
    }
    for (int i = 0; i < kSkinTargetCount; ++i) {
        skin_target_tex_[i] = 0;
        skin_target_fbo_[i] = 0;
    }
    skin_low_w_ = 0;
    skin_low_h_ = 0;
    skin_targets_ready_ = false;
#endif
}

void GPUBeautyBackend::prepareSkinFans(const IrisResult* detection,
                                       int width, int height, double frame_ts) {
    // 좌표 정합 (P8-W1 §6): face_mesh 정규 좌표(MediaPipe 0..1, 원본 미러링 전 기준)를
    // 입력 텍스처 공간에 맞춘다. 입력 텍스처는 전면 카메라라 이미 X 미러링되어 있으므로
    // 기존 FreqSep ROI 경로와 동일하게 mx = 1 - x 로 X를 뒤집는다(applyTextureId 1672-1689).
    // Y는 GL 풀스크린 쿼드 규약상 입력 텍스처의 vTexCoord.y = 1 - (이미지 y)로 나타나므로
    // 팬 NDC는 ny = 2*(1 - y) - 1 = 1 - 2y. 컴포지트는 base/blur/mask를 모두 vTexCoord로
    // 샘플하므로 셋이 동일 공간에서 일치한다 (마스크 Y-flip 별도 불필요).
    const float fw = static_cast<float>(width);
    const float fh = static_cast<float>(height);

    // ── 점별 픽셀 좌표 추출 + One-Euro 필터링 (모드 활성 시만, 재획득 시 reset) ──
    if (!skin_filters_active_) {
        for (auto& f : skin_landmark_filters_) f.reset();  // false→true 전환: 글라이드 방지
    }
    skin_filters_active_ = true;

    // 폴리곤 순서: 외곽36 / 우눈썹10 / 좌눈썹10 / 입술20 / 우눈16 / 좌눈16
    // (skin_oval_px_ 0..71 은 외곽 — 이마 확장 대상, 나머지는 팬에서 직접 사용)
    std::array<float, kSkinPointCount * 2> pts{};  // 필터링된 픽셀 좌표 (mirrored-x, image-y)

    auto fillIndices = [&](const int* idx, int n, int point_offset) {
        for (int i = 0; i < n; ++i) {
            const IrisLandmark& lm = detection->face_mesh[idx[i]];
            const float mx = (1.0f - lm.x) * fw;  // 미러 보정 후 픽셀 X
            const float py = lm.y * fh;            // 이미지 Y (픽셀)
            const int fi = (point_offset + i) * 2;
            pts[fi]     = skin_landmark_filters_[fi].filter(mx, frame_ts);
            pts[fi + 1] = skin_landmark_filters_[fi + 1].filter(py, frame_ts);
        }
    };

    int po = 0;
    fillIndices(skin_mask::kFaceOval.data(), skin_mask::kFaceOvalCount, po); po += skin_mask::kFaceOvalCount;
    fillIndices(skin_mask::kRightBrow.data(), skin_mask::kBrowCount, po); po += skin_mask::kBrowCount;
    fillIndices(skin_mask::kLeftBrow.data(), skin_mask::kBrowCount, po); po += skin_mask::kBrowCount;
    fillIndices(skin_mask::kLipsOuter.data(), skin_mask::kLipsCount, po); po += skin_mask::kLipsCount;
    fillIndices(skin_mask::kRightEyeContour.data(), skin_mask::kEyeContourCount, po); po += skin_mask::kEyeContourCount;
    fillIndices(skin_mask::kLeftEyeContour.data(), skin_mask::kEyeContourCount, po);

    // 외곽 36점만 이마 확장 사본에 복사 후 확장 (마스크 전용 — P8-W1 §5)
    for (int i = 0; i < skin_mask::kFaceOvalCount * 2; ++i) {
        skin_oval_px_[i] = pts[i];
    }
    skin_mask::extendForehead(skin_oval_px_.data(), skin_mask::kFaceOvalCount,
                              skin_mask::kForeheadExtend);

    // ── 팬 정점(NDC) 채우기: [무게중심, p0..pN-1, p0] ──
    // px(미러 X, 이미지 Y) → NDC: nx = 2*(px/W) - 1, ny = 1 - 2*(py/H)
    int out = 0;
    auto fanFromPx = [&](const float* src_px, int n) {
        float sumX = 0.0f, sumY = 0.0f;
        for (int p = 0; p < n; ++p) {
            const float nx = 2.0f * (src_px[p * 2] / fw) - 1.0f;
            const float ny = 1.0f - 2.0f * (src_px[p * 2 + 1] / fh);
            skin_fan_[out + (p + 1) * 2]     = nx;
            skin_fan_[out + (p + 1) * 2 + 1] = ny;
            sumX += nx;
            sumY += ny;
        }
        skin_fan_[out] = sumX / static_cast<float>(n);          // 무게중심
        skin_fan_[out + 1] = sumY / static_cast<float>(n);
        skin_fan_[out + (n + 1) * 2]     = skin_fan_[out + 2];   // 첫 점 반복 (닫기)
        skin_fan_[out + (n + 1) * 2 + 1] = skin_fan_[out + 3];
        out += (n + 2) * 2;
    };

    // 외곽은 이마 확장 사본, 나머지는 필터링 결과 (pts)에서 오프셋 슬라이스
    fanFromPx(skin_oval_px_.data(), skin_mask::kFaceOvalCount);
    fanFromPx(pts.data() + skin_mask::kFaceOvalCount * 2, skin_mask::kBrowCount);
    fanFromPx(pts.data() + (skin_mask::kFaceOvalCount + skin_mask::kBrowCount) * 2, skin_mask::kBrowCount);
    fanFromPx(pts.data() + (skin_mask::kFaceOvalCount + 2 * skin_mask::kBrowCount) * 2, skin_mask::kLipsCount);
    fanFromPx(pts.data() + (skin_mask::kFaceOvalCount + 2 * skin_mask::kBrowCount + skin_mask::kLipsCount) * 2, skin_mask::kEyeContourCount);
    fanFromPx(pts.data() + (skin_mask::kFaceOvalCount + 2 * skin_mask::kBrowCount + skin_mask::kLipsCount + skin_mask::kEyeContourCount) * 2, skin_mask::kEyeContourCount);
}

void GPUBeautyBackend::skinBlurPass(GLuint src_tex, GLuint dst_fbo,
                                    float dir_x, float dir_y, float offset_scale) {
#if IRIS_SDK_GPU_AVAILABLE
    glBindFramebuffer(GL_FRAMEBUFFER, dst_fbo);
    glViewport(0, 0, skin_low_w_, skin_low_h_);
    glUseProgram(skin_blur_program_);
    glUniform1i(skin_uniforms_.blurTexture, 0);
    glUniform2f(skin_uniforms_.blurDirection, dir_x, dir_y);
    glUniform1f(skin_uniforms_.blurOffsetScale, offset_scale);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, src_tex);
    renderFullscreenQuad();
#else
    (void)src_tex; (void)dst_fbo; (void)dir_x; (void)dir_y; (void)offset_scale;
#endif
}

void GPUBeautyBackend::renderSkinBasePasses(GLuint input_tex, int width, int height) {
#if IRIS_SDK_GPU_AVAILABLE
    (void)width; (void)height;
    glDisable(GL_BLEND);
    glDisable(GL_SCISSOR_TEST);

    // ① 입력 1/4 다운샘플 (passthrough — LINEAR 필터로 자동 박스 다운샘플)
    glBindFramebuffer(GL_FRAMEBUFFER, skin_target_fbo_[kSkinLow]);
    glViewport(0, 0, skin_low_w_, skin_low_h_);
    glUseProgram(passthrough_program_);
    glUniform1i(passthrough_u_texture_, 0);  // B2 idx18: 캐시된 location 사용
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, input_tex);
    renderFullscreenQuad();

    // ② 컬러 블러 H/V (offsetScale 1.6 — P8-W1 §5)
    skinBlurPass(skin_target_tex_[kSkinLow], skin_target_fbo_[kSkinTmp],
                 1.0f / static_cast<float>(skin_low_w_), 0.0f, kSkinColorBlurScale);
    skinBlurPass(skin_target_tex_[kSkinTmp], skin_target_fbo_[kSkinBlur],
                 0.0f, 1.0f / static_cast<float>(skin_low_h_), kSkinColorBlurScale);

    // ③ 피부 마스크: 외곽 팬 1.0 → 눈썹×2/입술/눈×2 팬 0.0 덮어쓰기 (블렌드 없음, 순서 중요)
    glBindFramebuffer(GL_FRAMEBUFFER, skin_target_fbo_[kSkinMask]);
    glViewport(0, 0, skin_low_w_, skin_low_h_);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(skin_mask_fill_program_);
    // ES 3.x: client-side 정점 배열은 기본 VAO(0)에서만 허용 — quad_vao_ 바인딩 금지.
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glEnableVertexAttribArray(0);
    int first = 0;
    for (std::size_t fan = 0; fan < kSkinFanCounts.size(); ++fan) {
        glUniform1f(skin_uniforms_.maskFillValue, (fan == 0) ? 1.0f : 0.0f);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, skin_fan_.data() + first * 2);
        glDrawArrays(GL_TRIANGLE_FAN, 0, kSkinFanCounts[fan]);
        first += kSkinFanCounts[fan];
    }
    glDisableVertexAttribArray(0);

    // ④ 마스크 블러 H/V (페더링 — offsetScale 1.0)
    skinBlurPass(skin_target_tex_[kSkinMask], skin_target_fbo_[kSkinTmp],
                 1.0f / static_cast<float>(skin_low_w_), 0.0f, kSkinMaskBlurScale);
    skinBlurPass(skin_target_tex_[kSkinTmp], skin_target_fbo_[kSkinMaskBlur],
                 0.0f, 1.0f / static_cast<float>(skin_low_h_), kSkinMaskBlurScale);
#else
    (void)input_tex; (void)width; (void)height;
#endif
}

void GPUBeautyBackend::renderSkinComposite(GLuint base_tex, GLuint output_fbo,
                                           int width, int height, float strength) {
#if IRIS_SDK_GPU_AVAILABLE
    glBindFramebuffer(GL_FRAMEBUFFER, output_fbo);
    glViewport(0, 0, width, height);
    glUseProgram(skin_composite_program_);
    glUniform1i(skin_uniforms_.compositeTexture, 0);
    glUniform1i(skin_uniforms_.compositeBlurTex, 1);
    glUniform1i(skin_uniforms_.compositeMaskTex, 2);
    glUniform1f(skin_uniforms_.compositeSkin, strength);
    glUniform1f(skin_uniforms_.compositeRadiance, skin_radiance_strength_);  // P8-W3
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, base_tex);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, skin_target_tex_[kSkinBlur]);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, skin_target_tex_[kSkinMaskBlur]);
    renderFullscreenQuad();
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
#else
    (void)base_tex; (void)output_fbo; (void)width; (void)height; (void)strength;
#endif
}

// =============================================================================
// P8-W4: 턱 V라인 워프 패스 (fragment-direct 비정규 RBF 인버스 워프)
// =============================================================================

void GPUBeautyBackend::executeWarpPass(
    GLuint input_tex, GLuint output_fbo,
    int width, int height,
    const iris_sdk::jaw_warp::JawWarpParams& params) {
#if IRIS_SDK_GPU_AVAILABLE
    if (warp_program_ == 0) {
        // 셰이더 미생성(non-fatal 실패) — 호출부 gate가 막지만 방어적으로 무동작.
        return;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, output_fbo);
    glViewport(0, 0, width, height);
    glUseProgram(warp_program_);

    // 셰이더 §4: uWarp[14] = [cx, cy, dx, dy] (렌더 텍스처 픽셀 공간 — 호출부가 이미 정렬).
    // params는 cx/cy/dx/dy를 SoA로 보관하므로 vec4 AoS 배열로 패킹한 뒤 한 번에 업로드.
    const int count = (params.count < iris_sdk::jaw_warp::JawWarpParams::kMaxControlPoints)
                          ? params.count
                          : iris_sdk::jaw_warp::JawWarpParams::kMaxControlPoints;
    // sigma_px==0 또는 count==0이면 셰이더 분기가 패스스루(입력 그대로 복사) — 방어적.
    float warp_data[iris_sdk::jaw_warp::JawWarpParams::kMaxControlPoints * 4] = {0.0f};
    for (int i = 0; i < count; ++i) {
        warp_data[i * 4 + 0] = params.cx[i];
        warp_data[i * 4 + 1] = params.cy[i];
        warp_data[i * 4 + 2] = params.dx[i];
        warp_data[i * 4 + 3] = params.dy[i];
    }
    // glUniform4fv count: 0이어도 안전(아무것도 안 올림). 셰이더 루프는 uWarpCount로 제한.
    glUniform4fv(warp_uniforms_.uWarp, count, warp_data);
    glUniform1i(warp_uniforms_.uWarpCount, count);
    glUniform1f(warp_uniforms_.uWarpSigma, params.sigma_px);
    glUniform4f(warp_uniforms_.uWarpBounds,
                params.bounds_min_x, params.bounds_min_y,
                params.bounds_max_x, params.bounds_max_y);
    glUniform2f(warp_uniforms_.uViewportPx,
                static_cast<float>(width), static_cast<float>(height));

    glUniform1i(warp_uniforms_.uTexture, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, input_tex);

    renderFullscreenQuad();

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
#else
    (void)input_tex; (void)output_fbo; (void)width; (void)height; (void)params;
#endif
}

//=============================================================================
// V2 API - 텍스처 ID 기반 (C API 호환)
//=============================================================================

IrisSdkError GPUBeautyBackend::applyTextureId(
    uint32_t input_texture,
    uint32_t* output_texture,
    int width, int height,
    const BeautyFilterConfigV2& config,
    const IrisResult* detection,
    uint32_t lut_texture_id,
    float lut_intensity) {

    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_) {
        return IRIS_SDK_NOT_INITIALIZED;
    }

    if (input_texture == 0 || !output_texture) {
        return IRIS_SDK_INVALID_PARAM;
    }

    if (width <= 0 || height <= 0) {
        return IRIS_SDK_INVALID_PARAM;
    }

    // (P8-W2) vivid 곁가지 제거 — beauty enabled 게이트로 단순화.
    if (!config.enabled) {
        // [B2 idx20-(3)] 필터를 끄는 프레임에서도 이전 프레임이 이월한 풀 텍스처
        // 2장과 fence를 즉시 정리한다. 이전 구현은 이 조기 반환이 정리 블록보다
        // 앞서 있어, 필터 비활성 동안 풀 텍스처 2장이 in_use로, fence 1개가 미삭제로
        // 다음 활성화 시점까지 잔류했다(상한 고정이라 누수는 아니나 점유 낭비).
        // 헬퍼가 fence(GL 객체)를 삭제하므로 자체 컨텍스트 모드에서는 GL 컨텍스트를
        // 먼저 current로 만들어야 한다(GLSurfaceView 모드는 이미 current).
#if IRIS_SDK_GPU_AVAILABLE
        if (render_context_) {
            render_context_->makeCurrent();
        }
#endif
        releasePreviousFrameResources();
        *output_texture = input_texture;
        return IRIS_SDK_OK;
    }

#if IRIS_SDK_GPU_AVAILABLE
    // GLSurfaceView 모드에서는 이미 EGL 컨텍스트가 바인딩되어 있음
    if (render_context_) {
        render_context_->makeCurrent();
    }

    // 입력 텍스처 ID (B2 idx11: 미사용 input_handle 데드코드 제거)
    GLuint input_tex_id = static_cast<GLuint>(input_texture);

    // ROI 생성 (detection이 있는 경우)
    BeautyROI roi;
    BeautyROI* roi_ptr = nullptr;

    if (config.enabled && detection && detection->detected && config.roiOnly) {
        roi.face_rect = computeExpandedFaceRect(
            detection->face_rect.x, detection->face_rect.y,
            detection->face_rect.width, detection->face_rect.height,
            width, height);
        roi.mask_width = static_cast<int>(roi.face_rect.width);
        roi.mask_height = static_cast<int>(roi.face_rect.height);
        roi.valid = true;
        roi.timestamp_ms = detection->timestamp_ms;

        if (detection->face_mesh_valid) {
            // 전면 카메라 미러링 보정: 입력 텍스처는 이미 X 미러링됨,
            // 하지만 face_mesh 좌표는 원본(미러링 전) 기준 → X를 뒤집어야 함
            std::array<IrisLandmark, 478> mirrored_mesh;
            for (int i = 0; i < 478; i++) {
                mirrored_mesh[i] = detection->face_mesh[i];
                mirrored_mesh[i].x = 1.0f - mirrored_mesh[i].x;
            }

            if (BeautyROIManager::computeROI(
                    mirrored_mesh.data(), 478,
                    width, height, config, roi)) {
                // computeROI outputs normalized (0~1) face_rect → convert to pixel
                roi.face_rect.x *= width;
                roi.face_rect.y *= height;
                roi.face_rect.width *= width;
                roi.face_rect.height *= height;
            }
            // 실패 시 line 871-888의 픽셀 좌표 ROI를 그대로 사용
        }
        roi_ptr = &roi;
    }

    // 이전 프레임의 출력 텍스처 반환 + GPU fence 정리 (B2 idx20: 헬퍼로 일원화)
    releasePreviousFrameResources();

    BeautyFilterConfigV2 effective_config = buildEffectiveConfig(config);

    // (P8-W2) LUT 곁가지 제거 — 시그니처는 D단계까지 유지, 전달값은 무시.
    (void)lut_texture_id;
    (void)lut_intensity;

    // P8-W1: landmark-masked smoothing 모드 — 활성 시 스무딩을 적용한다.
    // 강도 0 / 모드 OFF / face_mesh 무효면 false → 마스크·블러·필터·타깃 전부 생략(비용 0).
    // (P8-W2) skin mask가 OFF면 스무딩 없음 = 의도된 종착(레거시 FreqSep/Bilateral 폴백 제거).
    const bool use_skin_mask = config.enabled && skinMaskSmoothingActive(detection);

    // P8-W4: 턱 V라인 워프 게이트. slim_face>0 또는 thin_chin>0 + face_mesh 유효 +
    //   warp 셰이더 생성 성공일 때만 활성. computeJawWarp가 추가로 strength≤0/퇴화를 거른다.
    const bool needs_warp = config.enabled
        && (effective_config.slimFace > 0.0f || effective_config.thinChin > 0.0f)
        && detection && detection->detected && detection->face_mesh_valid
        && warp_program_ != 0;
    // slim/chin 합성(단순): 둘 중 큰 값이 메인 강도 (브레인스토밍 §5-3).
    const float jawStrength = needs_warp
        ? std::clamp(std::max(effective_config.slimFace, effective_config.thinChin), 0.0f, 1.0f)
        : 0.0f;

    // 활성 필터 수에 따라 동적으로 텍스처 할당
    // (P8-W2) 잔존 패스: ① skin mask smoothing + ② brightness(통합 Color). + (P8-W4) ③ warp.
    int active_filter_count = 0;
    if (use_skin_mask) {
        active_filter_count++;
    }
    bool needsBrightness = std::abs(effective_config.brightness - 1.0f) > 0.01f;
    if (needsBrightness) active_filter_count++;
    if (needs_warp) active_filter_count++;

    // 필터 0개: 패스스루 (텍스처 할당 불필요)
    if (active_filter_count == 0) {
        *output_texture = input_tex_id;
        return IRIS_SDK_OK;
    }

    // 필터 1개: 단일 텍스처, 2개+: ping-pong 버퍼
    TexturePool::TextureInfo* ping = nullptr;
    TexturePool::TextureInfo* pong = nullptr;

    // [B2 idx5] portrait/고해상도 입력에서도 풀 상한이 부족해 silent 실패하지
    // 않도록 acquire 전에 상한을 입력 크기까지 보장한다 (GL 무관, lazy 할당).
    texture_pool_->ensureCapacity(width, height);

    if (active_filter_count == 1) {
        ping = texture_pool_->acquireRenderTarget(width, height);
        if (!ping) {
            LOGE("Failed to acquire render target");
            return IRIS_SDK_UNKNOWN;
        }
    } else {
        if (!texture_pool_->acquirePingPongPair(width, height, ping, pong)) {
            LOGE("Failed to acquire ping-pong buffers");
            return IRIS_SDK_UNKNOWN;
        }
    }

    GLuint current_input = input_tex_id;
    TexturePool::TextureInfo* current_output = ping;

    // 뷰포트 설정
    glViewport(0, 0, width, height);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);

    // ROI passthrough: scissor 활성화 전에 출력 FBO를 원본으로 채움
    // → scissor 외부 픽셀이 stale 데이터가 되는 것을 방지
    // (P8-W2) brightness=1.0 중립값으로 원본 그대로 채운다(balance/whitening/LUT 인자 제거).
    if (roi_ptr && roi_ptr->valid) {
        executeCombinedColorPass(input_tex_id, ping->fbo_id, width, height, 1.0f);
        if (pong) {
            executeCombinedColorPass(input_tex_id, pong->fbo_id, width, height, 1.0f);
        }
    }

    // Temporal stability: One Euro Filter for face_rect center (P4-W3-04)
    // 효과 범위: scissor 영역 안정화 (face_rect jitter에 의한 scissor 경계 흔들림 방지)
    // 제한사항: FreqSep 마스크 내용에는 영향 없음 (마스크는 computeROI()에서 face mesh
    //   랜드마크 기반으로 생성되고, composite 셰이더에서 UV 직접 샘플링)
    // TODO(P4-W3-04-R2): 진짜 마스크 안정화가 필요하면 uSkinMask 샘플링에
    //   프레임별 UV offset 도입 또는 computeROI() 이전 단계에서 안정화 적용 검토
    // 동일 프레임 내 모든 OEF가 같은 타임스탬프를 공유하도록 캡처
    auto frame_now = std::chrono::steady_clock::now();
    double frame_ts = std::chrono::duration<double>(frame_now.time_since_epoch()).count();

    if (roi_ptr && roi_ptr->valid) {
        float cx = roi_ptr->face_rect.x + roi_ptr->face_rect.width * 0.5f;
        float cy = roi_ptr->face_rect.y + roi_ptr->face_rect.height * 0.5f;
        float stable_cx = mask_center_x_filter_.filter(cx, frame_ts);
        float stable_cy = mask_center_y_filter_.filter(cy, frame_ts);
        float dx = stable_cx - cx;
        float dy = stable_cy - cy;
        roi_ptr->face_rect.x += dx;
        roi_ptr->face_rect.y += dy;
    } else {
        // 얼굴 추적 끊김 → 필터 리셋 (재획득 시 이전 상태 잔류 방지)
        resetTemporalFilters();
    }

    // ROI glScissor 설정 (교집합 기반 — 프레임 경계를 넘는 ROI에도 안전)
    bool scissor_active = false;
    if (roi_ptr && roi_ptr->valid) {
        // top-left 기준 ROI 사각형
        int rect_x = static_cast<int>(std::floor(roi_ptr->face_rect.x));
        int rect_y = static_cast<int>(std::floor(roi_ptr->face_rect.y));
        int rect_w = static_cast<int>(std::ceil(roi_ptr->face_rect.width));
        int rect_h = static_cast<int>(std::ceil(roi_ptr->face_rect.height));

        // 프레임 영역 [0, width) × [0, height) 과의 교집합 (Intersection)
        int intersect_l = std::max(0, rect_x);
        int intersect_t = std::max(0, rect_y);
        int intersect_r = std::min(width,  rect_x + rect_w);
        int intersect_b = std::min(height, rect_y + rect_h);

        int sw = intersect_r - intersect_l;
        int sh = intersect_b - intersect_t;

        if (sw > 0 && sh > 0) {
            // GL bottom-left 좌표계로 변환
            int sx = intersect_l;
            int sy = height - intersect_b;

            glEnable(GL_SCISSOR_TEST);
            glScissor(sx, sy, sw, sh);
            scissor_active = true;

            LOGD("ROI Scissor: (%d, %d, %d, %d) from face_rect(%.1f, %.1f, %.1f, %.1f)",
                 sx, sy, sw, sh, roi_ptr->face_rect.x, roi_ptr->face_rect.y,
                 roi_ptr->face_rect.width, roi_ptr->face_rect.height);
        } else {
            // roiOnly 정책: 교집합이 비어있으면 beauty 필터 스킵
            // (P8-W2) vivid 곁가지 제거 — 입력을 그대로 반환.
            LOGW("ROI Scissor skipped: intersection empty (rect=%d,%d,%d,%d frame=%dx%d)",
                 rect_x, rect_y, rect_w, rect_h, width, height);
            *output_texture = current_input;
            if (ping) { previous_output_ping_ = ping; }
            if (pong) { previous_output_pong_ = pong; }
            previous_fence_ = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            return IRIS_SDK_OK;
        }
    }

    // 필터 체인 실행 (최적화됨 + 프로파일링)
    bool profiling = profiler_ && profiler_->isEnabled();

    // 1. 스무딩 (P8-W1): landmark-masked skin smoothing (① 핵심).
    //    (P8-W2) 곁가지 제거 — skin mask OFF면 스무딩 없음(의도된 종착, FreqSep/Bilateral 폴백 제거).
    //    실패(타깃 생성 실패 등) 시 안전하게 스무딩만 생략 (다른 패스는 정상).
    if (use_skin_mask) {
        if (profiling) profiler_->begin("SkinMaskSmoothing");
        // 마스크는 폴리곤이 영역을 정의하므로 ROI scissor를 끄고 전체 프레임에서 동작
        if (scissor_active) glDisable(GL_SCISSOR_TEST);
        if (ensureSkinTargets(width, height)) {
            prepareSkinFans(detection, width, height, frame_ts);
            renderSkinBasePasses(current_input, width, height);
            glViewport(0, 0, width, height);  // base 패스가 1/4 뷰포트로 바꿈 → 복원
            renderSkinComposite(current_input, current_output->fbo_id,
                                width, height, skin_mask_smoothing_strength_);
            current_input = current_output->texture_id;
            if (pong) current_output = (current_output == ping) ? pong : ping;
        }
        if (scissor_active) glEnable(GL_SCISSOR_TEST);
        if (profiling) profiler_->end("SkinMaskSmoothing");
    } else {
        // 스킨 모드가 이 프레임에 동작하지 않음 → 다음 활성 프레임에 필터 reset 하도록 표시
        // (얼굴 재획득 시 묵은 필터 상태로 인한 마스크 경계 글라이드 방지 — 원본 FaceTracker)
        skin_filters_active_ = false;
    }

    // 2. 통합 Color Adjustment ((P8-W2) brightness 잔존, balance/whitening/LUT 곁가지 제거)
    if (needsBrightness) {
        if (profiling) profiler_->begin("CombinedColor");
        executeCombinedColorPass(current_input, current_output->fbo_id,
                                 width, height,
                                 effective_config.brightness);
        if (profiling) profiler_->end("CombinedColor");
        current_input = current_output->texture_id;
        if (pong) current_output = (current_output == ping) ? pong : ping;
    }

    // (P8-W2 제거) 소프트 포커스 / Vivid 포스트프로세싱 곁가지 제거.

    // 3. P8-W4: 턱 V라인 워프 (skin/brightness 이후 마지막 패스).
    //    워프장은 턱(프레임 가장자리)까지 닿으므로 ROI scissor를 끄고 전체 프레임에서 동작
    //    (skin mask 1345 패턴과 동일). computeJawWarp가 strength≤0/퇴화면 false → 패스 생략.
    if (needs_warp) {
        if (profiling) profiler_->begin("JawWarp");
        iris_sdk::jaw_warp::JawWarpParams wp{};
        // computeJawWarp는 원본(비미러) face_mesh를 받아 이미지 픽셀 공간 제어점을 낸다.
        // detection->face_mesh 는 iris_sdk::IrisLandmark[478] — computeJawWarp 인자 타입과
        // 동일하므로 캐스팅 불필요(prepareSkinFans 와 동일하게 직접 접근).
        if (iris_sdk::jaw_warp::computeJawWarp(detection->face_mesh, width, height,
                                               jawStrength, wp)
            && wp.sigma_px > 0.0f && wp.count > 0) {
            // 🔴 좌표 정합: 이미지 공간 → 렌더 텍스처 공간(미러·Y-flip). prepareSkinFans 동형.
            iris_sdk::jaw_warp::JawWarpParams wp_render =
                alignWarpToRenderSpace(wp, width, height);
            if (scissor_active) glDisable(GL_SCISSOR_TEST);
            executeWarpPass(current_input, current_output->fbo_id,
                            width, height, wp_render);
            if (scissor_active) glEnable(GL_SCISSOR_TEST);
            current_input = current_output->texture_id;
            if (pong) current_output = (current_output == ping) ? pong : ping;
        }
        if (profiling) profiler_->end("JawWarp");
    }

    // ROI Scissor 해제
    if (scissor_active) {
        glDisable(GL_SCISSOR_TEST);
    }

    *output_texture = current_input;

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // GPU 펜스 삽입: 현재 프레임 커맨드가 완료될 때 시그널됨
    previous_fence_ = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);

    // Ping-Pong 버퍼를 다음 프레임에서 반환하도록 저장
    // (현재 프레임에서 즉시 반환하면 출력 텍스처가 사라짐)
    previous_output_ping_ = ping;
    previous_output_pong_ = pong;

    // 프레임 종료 처리 (프로파일링 결과 수집)
    if (profiling) {
        profiler_->frameEnd();
    }
#else
    *output_texture = input_texture;
    (void)detection;
    (void)lut_texture_id;
    (void)lut_intensity;
#endif

    return IRIS_SDK_OK;
}

IrisSdkError GPUBeautyBackend::applyFaceWarp(
    uint32_t input_texture,
    uint32_t* output_texture,
    int width, int height,
    float slim_face,
    float thin_chin,
    float enlarge_eyes,
    const IrisResult* detection) {

    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_) {
        return IRIS_SDK_NOT_INITIALIZED;
    }

    if (input_texture == 0 || !output_texture) {
        return IRIS_SDK_INVALID_PARAM;
    }

    // 모든 값이 0이거나 검출 결과가 없으면 패스스루
    if (!detection || !detection->detected ||
        (slim_face <= 0.0f && thin_chin <= 0.0f && enlarge_eyes <= 0.0f)) {
        *output_texture = input_texture;
        return IRIS_SDK_OK;
    }

    // P8-W4: 턱 V라인 워프의 정본 경로는 config 기반 applyTextureId() 내 warp 패스다
    //   (gate=slimFace>0‖thinChin>0, skin→brightness→warp 순서, 텍스처 풀 deferred-release).
    //   이 standalone 진입점은 풀 수명 관리(previous_output_*)와 패스 체인을 우회하므로
    //   별도 정식화를 보류한다. 호출자는 iris_sdk_set_beauty_filter_v2(slim_face/thin_chin)
    //   + iris_sdk_apply_beauty_filter_v2 경로를 사용해야 한다.
    LOGW("GPUBeautyBackend::applyFaceWarp() standalone path not supported. "
         "Use config-based applyTextureId (slim_face/thin_chin) — P8-W4-B.");

    (void)input_texture;
    (void)output_texture;
    (void)width;
    (void)height;
    (void)slim_face;
    (void)thin_chin;
    (void)enlarge_eyes;

    return IRIS_SDK_ERROR_NOT_SUPPORTED;
}

void GPUBeautyBackend::releaseTexture(uint32_t texture) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_ || texture == 0) {
        return;
    }

#if IRIS_SDK_GPU_AVAILABLE
    if (render_context_) {
        render_context_->makeCurrent();
    }

    GLuint tex_id = static_cast<GLuint>(texture);

    // 풀 관리 텍스처는 풀에 반환, 외부 텍스처만 직접 삭제
    if (texture_pool_ && texture_pool_->releaseTextureById(tex_id)) {
        LOGI("Released pool-managed texture %u", tex_id);
    } else {
        glDeleteTextures(1, &tex_id);
        LOGI("Released external texture %u", tex_id);
    }
#else
    (void)texture;
#endif
}

} // namespace iris_sdk
