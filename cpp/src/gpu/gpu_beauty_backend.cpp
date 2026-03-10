/**
 * @file gpu_beauty_backend.cpp
 * @brief GPUBeautyBackend 구현
 */

#include "iris_sdk/gpu/gpu_beauty_backend.h"
#include "iris_sdk/gpu/render_context.h"
#include "iris_sdk/beauty_roi_manager.h"
#include <algorithm>
#include <array>
#include <cctype>
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
// Gaussian half-kernel: center + kMaxGaussianRadius sides = 29 entries
constexpr int kMaxGaussianRadius = 28;

// CPU-side Gaussian weight precomputation (symmetric half-kernel)
// weights[i] = exp(-i*i / (2*sigma*sigma)), normalized so full kernel sums to 1.0
static void computeGaussianWeights(int radius, float weights[kMaxGaussianRadius + 1]) {
    radius = std::clamp(radius, 1, kMaxGaussianRadius);
    float sigma = radius * 0.4f;
    float sum = 0.0f;
    for (int i = 0; i <= radius; i++) {
        weights[i] = std::exp(-(float)(i * i) / (2.0f * sigma * sigma));
        sum += weights[i] * (i == 0 ? 1.0f : 2.0f); // center once, sides twice
    }
    for (int i = 0; i <= radius; i++) {
        weights[i] /= sum;
    }
    // Zero out unused entries
    for (int i = radius + 1; i <= kMaxGaussianRadius; i++) {
        weights[i] = 0.0f;
    }
}
#endif

// 셰이더 소스 extern 선언
namespace shaders {
extern const char* FULLSCREEN_QUAD_VERTEX;
extern const char* PASSTHROUGH_FRAGMENT;
extern const char* BRIGHTNESS_FRAGMENT;
extern const char* BILATERAL_FILTER_FRAGMENT;
extern const char* WHITENING_FRAGMENT;
extern const char* COLOR_BALANCE_FRAGMENT;
extern const char* SOFT_FOCUS_FRAGMENT;
extern const char* MASKING_FRAGMENT;
extern const char* COMBINED_COLOR_ADJUSTMENT_FRAGMENT;
extern const char* FREQ_SEP_GAUSSIAN_FRAGMENT;
extern const char* FREQ_SEP_COMPOSITE_FRAGMENT;
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

#if IRIS_SDK_GPU_AVAILABLE
    // Neutral 1x1x1 identity 3D LUT 생성 (sampler3D fallback)
    // LUT OFF 시 glBindTexture(GL_TEXTURE_3D, 0) 대신 바인딩하여
    // 드라이버 의존적 불안정을 방지
    {
        glGenTextures(1, &neutral_lut_texture_);
        glBindTexture(GL_TEXTURE_3D, neutral_lut_texture_);
        // Identity: RGB 그대로 반환 (R=1, G=1, B=1, A=1)
        const uint8_t identity_data[4] = {255, 255, 255, 255};
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA8, 1, 1, 1, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, identity_data);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_3D, 0);
        LOGI("Neutral 1x1x1 identity LUT created: id=%u", neutral_lut_texture_);
    }
#endif

    // 디바이스 성능 등급 감지 (P4-W3-04)
    device_tier_ = detectDeviceTier();
    LOGI("Device tier detected: %s",
         device_tier_ == DeviceTier::HIGH ? "HIGH" :
         device_tier_ == DeviceTier::MID ? "MID" : "LOW");

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

    // 스무딩 (Bilateral Filter)
    if (!shader_manager_->createProgram(
            shaders::FULLSCREEN_QUAD_VERTEX,
            shaders::BILATERAL_FILTER_FRAGMENT,
            smoothing_program_)) {
        LOGE("Failed to create smoothing program");
        return false;
    }
    shader_manager_->cacheProgram("smoothing", smoothing_program_);

    // 화이트닝
    if (!shader_manager_->createProgram(
            shaders::FULLSCREEN_QUAD_VERTEX,
            shaders::WHITENING_FRAGMENT,
            whitening_program_)) {
        LOGE("Failed to create whitening program");
        return false;
    }
    shader_manager_->cacheProgram("whitening", whitening_program_);

    // 컬러 밸런스
    if (!shader_manager_->createProgram(
            shaders::FULLSCREEN_QUAD_VERTEX,
            shaders::COLOR_BALANCE_FRAGMENT,
            color_balance_program_)) {
        LOGE("Failed to create color_balance program");
        return false;
    }
    shader_manager_->cacheProgram("color_balance", color_balance_program_);

    // 소프트 포커스
    if (!shader_manager_->createProgram(
            shaders::FULLSCREEN_QUAD_VERTEX,
            shaders::SOFT_FOCUS_FRAGMENT,
            soft_focus_program_)) {
        LOGE("Failed to create soft_focus program");
        return false;
    }
    shader_manager_->cacheProgram("soft_focus", soft_focus_program_);

    // 밝기
    if (!shader_manager_->createProgram(
            shaders::FULLSCREEN_QUAD_VERTEX,
            shaders::BRIGHTNESS_FRAGMENT,
            brightness_program_)) {
        LOGE("Failed to create brightness program");
        return false;
    }
    shader_manager_->cacheProgram("brightness", brightness_program_);

    // 마스킹
    if (!shader_manager_->createProgram(
            shaders::FULLSCREEN_QUAD_VERTEX,
            shaders::MASKING_FRAGMENT,
            masking_program_)) {
        LOGE("Failed to create masking program");
        return false;
    }
    shader_manager_->cacheProgram("masking", masking_program_);

    // 통합 Color Adjustment (Brightness + ColorBalance + Whitening)
    if (!shader_manager_->createProgram(
            shaders::FULLSCREEN_QUAD_VERTEX,
            shaders::COMBINED_COLOR_ADJUSTMENT_FRAGMENT,
            combined_color_program_)) {
        LOGE("Failed to create combined_color program");
        return false;
    }
    shader_manager_->cacheProgram("combined_color", combined_color_program_);

    // Frequency Separation 셰이더
    if (!initializeFreqSepShaders()) {
        LOGW("Failed to create Freq Sep shaders (non-fatal)");
        // Non-fatal: Freq Sep은 선택적 기능, Bilateral fallback 사용
    }

    LOGI("All %zu shader programs created successfully",
         shader_manager_->getCachedProgramCount());
    return true;
}

bool GPUBeautyBackend::initializeFreqSepShaders() {
    // Freq Sep Gaussian
    if (!shader_manager_->createProgram(
            shaders::FULLSCREEN_QUAD_VERTEX,
            shaders::FREQ_SEP_GAUSSIAN_FRAGMENT,
            freq_sep_gaussian_program_)) {
        LOGE("Failed to create freq_sep_gaussian program");
        return false;
    }
    shader_manager_->cacheProgram("freq_sep_gaussian", freq_sep_gaussian_program_);

    // Freq Sep Composite
    if (!shader_manager_->createProgram(
            shaders::FULLSCREEN_QUAD_VERTEX,
            shaders::FREQ_SEP_COMPOSITE_FRAGMENT,
            freq_sep_composite_program_)) {
        LOGE("Failed to create freq_sep_composite program");
        return false;
    }
    shader_manager_->cacheProgram("freq_sep_composite", freq_sep_composite_program_);

    LOGI("Freq Sep shader programs created successfully");
    return true;
}

void GPUBeautyBackend::cacheUniformLocations() {
#if IRIS_SDK_GPU_AVAILABLE
    // Smoothing (Bilateral Filter) Uniforms
    smoothing_uniforms_.uTexture = glGetUniformLocation(smoothing_program_, "uTexture");
    smoothing_uniforms_.uTexelSize = glGetUniformLocation(smoothing_program_, "uTexelSize");
    smoothing_uniforms_.uStrength = glGetUniformLocation(smoothing_program_, "uStrength");

    // Whitening Uniforms
    whitening_uniforms_.uTexture = glGetUniformLocation(whitening_program_, "uTexture");
    whitening_uniforms_.uWhiteningStrength = glGetUniformLocation(whitening_program_, "uStrength");

    // Color Balance Uniforms
    color_balance_uniforms_.uTexture = glGetUniformLocation(color_balance_program_, "uTexture");
    color_balance_uniforms_.uBalance = glGetUniformLocation(color_balance_program_, "uBalance");

    // Soft Focus Uniforms
    soft_focus_uniforms_.uTexture = glGetUniformLocation(soft_focus_program_, "uTexture");
    soft_focus_uniforms_.uSoftFocusTexelSize = glGetUniformLocation(soft_focus_program_, "uTexelSize");
    soft_focus_uniforms_.uSoftFocusStrength = glGetUniformLocation(soft_focus_program_, "uStrength");

    // Brightness Uniforms
    brightness_uniforms_.uTexture = glGetUniformLocation(brightness_program_, "uTexture");
    brightness_uniforms_.uBrightness = glGetUniformLocation(brightness_program_, "uBrightness");

    // Masking Uniforms
    masking_uniforms_.uFiltered = glGetUniformLocation(masking_program_, "uFiltered");
    masking_uniforms_.uOriginal = glGetUniformLocation(masking_program_, "uOriginal");
    masking_uniforms_.uMask = glGetUniformLocation(masking_program_, "uMask");

    // Combined Color Adjustment Uniforms
    combined_color_uniforms_.uTexture = glGetUniformLocation(combined_color_program_, "uTexture");
    combined_color_uniforms_.uCombinedBrightness = glGetUniformLocation(combined_color_program_, "uBrightness");
    combined_color_uniforms_.uCombinedBalance = glGetUniformLocation(combined_color_program_, "uBalance");
    combined_color_uniforms_.uCombinedWhitening = glGetUniformLocation(combined_color_program_, "uWhitening");
    combined_color_uniforms_.uCombinedLutTexture = glGetUniformLocation(combined_color_program_, "uLutTexture");
    combined_color_uniforms_.uCombinedLutIntensity = glGetUniformLocation(combined_color_program_, "uLutIntensity");

    // Freq Sep Gaussian Uniforms
    if (freq_sep_gaussian_program_ != 0) {
        freq_sep_gaussian_uniforms_.uTexture = glGetUniformLocation(freq_sep_gaussian_program_, "uTexture");
        freq_sep_gaussian_uniforms_.uDirection = glGetUniformLocation(freq_sep_gaussian_program_, "uDirection");
        freq_sep_gaussian_uniforms_.uRadius = glGetUniformLocation(freq_sep_gaussian_program_, "uRadius");
        freq_sep_gaussian_uniforms_.uWeights = glGetUniformLocation(freq_sep_gaussian_program_, "uWeights[0]");
        freq_sep_gaussian_uniforms_.uLinearize = glGetUniformLocation(freq_sep_gaussian_program_, "uLinearize");
    }

    // Freq Sep Composite Uniforms
    if (freq_sep_composite_program_ != 0) {
        freq_sep_composite_uniforms_.uSmoothedLow = glGetUniformLocation(freq_sep_composite_program_, "uSmoothedLow");
        freq_sep_composite_uniforms_.uLowFreq = glGetUniformLocation(freq_sep_composite_program_, "uLowFreq");
        freq_sep_composite_uniforms_.uOriginal = glGetUniformLocation(freq_sep_composite_program_, "uOriginal");
        freq_sep_composite_uniforms_.uSkinMask = glGetUniformLocation(freq_sep_composite_program_, "uSkinMask");
        freq_sep_composite_uniforms_.uHighFreqPreserve = glGetUniformLocation(freq_sep_composite_program_, "uHighFreqPreserve");
        freq_sep_composite_uniforms_.uAttenuationLow = glGetUniformLocation(freq_sep_composite_program_, "uAttenuationLow");
        freq_sep_composite_uniforms_.uAttenuationHigh = glGetUniformLocation(freq_sep_composite_program_, "uAttenuationHigh");
        freq_sep_composite_uniforms_.uEdgeWeight = glGetUniformLocation(freq_sep_composite_program_, "uEdgeWeight");
        freq_sep_composite_uniforms_.uChromaWeight = glGetUniformLocation(freq_sep_composite_program_, "uChromaWeight");
        freq_sep_composite_uniforms_.uToneLift = glGetUniformLocation(freq_sep_composite_program_, "uToneLift");
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

    // Neutral LUT 텍스처 해제
    if (neutral_lut_texture_ != 0) {
        glDeleteTextures(1, &neutral_lut_texture_);
        neutral_lut_texture_ = 0;
    }
#endif

    // Freq Sep skin mask 텍스처 해제
#if IRIS_SDK_GPU_AVAILABLE
    if (skin_mask_texture_ != 0) {
        glDeleteTextures(1, &skin_mask_texture_);
        skin_mask_texture_ = 0;
    }
    skin_mask_width_ = 0;
    skin_mask_height_ = 0;
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

    // 프로그램 ID 초기화
    passthrough_program_ = 0;
    smoothing_program_ = 0;
    whitening_program_ = 0;
    color_balance_program_ = 0;
    soft_focus_program_ = 0;
    brightness_program_ = 0;
    masking_program_ = 0;
    combined_color_program_ = 0;
    freq_sep_gaussian_program_ = 0;
    freq_sep_composite_program_ = 0;

    LOGI("GPUBeautyBackend released");
}

void GPUBeautyBackend::resetTemporalFilters() {
    skin_radius_filter_.reset();
    mask_center_x_filter_.reset();
    mask_center_y_filter_.reset();
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
        return IRIS_SDK_ERROR_NOT_INITIALIZED;
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
        return IRIS_SDK_ERROR_NOT_INITIALIZED;
    }

    if (!input.isValid() || input.type != TextureHandle::Type::OpenGLES) {
        LOGE("Invalid input texture");
        return IRIS_SDK_INVALID_PARAM;
    }

    if (!config.enabled) {
        // 비활성화: 입력을 그대로 출력으로
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

    // 1. 스무딩 (Bilateral Filter) - 단독 패스
    if (config.smoothing > 0.01f) {
        if (profiling) profiler_->begin("Smoothing");
        executeSmoothingPass(current_input, current_output->fbo_id,
                             width, height, config);
        if (profiling) profiler_->end("Smoothing");
        current_input = current_output->texture_id;
        current_output = (current_output == ping) ? pong : ping;
    }

    // 2. 통합 Color Adjustment (Brightness + ColorBalance + Whitening)
    //    기존 3개 패스를 1개로 병합하여 FBO 전환 오버헤드 감소
    bool needsBrightness = std::abs(config.brightness - 1.0f) > 0.01f;
    bool needsBalance = std::abs(config.colorBalance) > 0.01f;
    bool needsWhitening = config.whitening > 0.01f;

    if (needsBrightness || needsBalance || needsWhitening) {
        if (profiling) profiler_->begin("CombinedColor");
        executeCombinedColorPass(current_input, current_output->fbo_id,
                                 width, height,
                                 config.brightness,
                                 config.colorBalance,
                                 config.whitening);
        if (profiling) profiler_->end("CombinedColor");
        current_input = current_output->texture_id;
        current_output = (current_output == ping) ? pong : ping;
    }

    // 3. 소프트 포커스 - 단독 패스 (blur 필요)
    if (config.softFocus > 0.01f) {
        if (profiling) profiler_->begin("SoftFocus");
        executeSoftFocusPass(current_input, current_output->fbo_id,
                             width, height, config.softFocus);
        if (profiling) profiler_->end("SoftFocus");
        current_input = current_output->texture_id;
    }

    // 출력 텍스처 핸들 설정
    // 주의: 이 텍스처는 풀에서 관리되므로 사용 후 반환 필요
    // current_input이 가리키는 텍스처는 풀의 TextureInfo가 소유하므로
    // native_handle은 해당 TextureInfo의 texture_id 주소를 사용
    TexturePool::TextureInfo* result_info = (current_input == ping->texture_id) ? ping : pong;
    output.native_handle = &result_info->texture_id;
    output.type = TextureHandle::Type::OpenGLES;
    output.width = width;
    output.height = height;
    output.format = input.format;

    // 텍스처 반환 (현재 output으로 사용 중인 것 제외)
    // 실제 구현에서는 호출자가 output 사용 후 반환해야 함

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

void GPUBeautyBackend::executeSmoothingPass(
    GLuint input_tex, GLuint output_fbo,
    int width, int height,
    const BeautyFilterConfigV2& config) {

#if IRIS_SDK_GPU_AVAILABLE
    glBindFramebuffer(GL_FRAMEBUFFER, output_fbo);
    glUseProgram(smoothing_program_);

    // 캐시된 Uniform Location 사용 (성능 최적화)
    glUniform1i(smoothing_uniforms_.uTexture, 0);
    glUniform2f(smoothing_uniforms_.uTexelSize, 1.0f / width, 1.0f / height);
    glUniform1f(smoothing_uniforms_.uStrength, config.smoothing);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, input_tex);

    renderFullscreenQuad();

    glBindTexture(GL_TEXTURE_2D, 0);
#else
    (void)input_tex;
    (void)output_fbo;
    (void)width;
    (void)height;
    (void)config;
#endif
}

void GPUBeautyBackend::executeWhiteningPass(
    GLuint input_tex, GLuint output_fbo,
    int width, int height,
    float whitening) {

#if IRIS_SDK_GPU_AVAILABLE
    glBindFramebuffer(GL_FRAMEBUFFER, output_fbo);
    glUseProgram(whitening_program_);

    // 캐시된 Uniform Location 사용
    glUniform1i(whitening_uniforms_.uTexture, 0);
    glUniform1f(whitening_uniforms_.uWhiteningStrength, whitening);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, input_tex);

    renderFullscreenQuad();

    glBindTexture(GL_TEXTURE_2D, 0);
#else
    (void)input_tex;
    (void)output_fbo;
    (void)width;
    (void)height;
    (void)whitening;
#endif
}

void GPUBeautyBackend::executeColorBalancePass(
    GLuint input_tex, GLuint output_fbo,
    int width, int height,
    float balance) {

#if IRIS_SDK_GPU_AVAILABLE
    glBindFramebuffer(GL_FRAMEBUFFER, output_fbo);
    glUseProgram(color_balance_program_);

    // 캐시된 Uniform Location 사용
    glUniform1i(color_balance_uniforms_.uTexture, 0);
    glUniform1f(color_balance_uniforms_.uBalance, balance);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, input_tex);

    renderFullscreenQuad();

    glBindTexture(GL_TEXTURE_2D, 0);
#else
    (void)input_tex;
    (void)output_fbo;
    (void)width;
    (void)height;
    (void)balance;
#endif
}

void GPUBeautyBackend::executeSoftFocusPass(
    GLuint input_tex, GLuint output_fbo,
    int width, int height,
    float strength) {

#if IRIS_SDK_GPU_AVAILABLE
    glBindFramebuffer(GL_FRAMEBUFFER, output_fbo);
    glUseProgram(soft_focus_program_);

    // 캐시된 Uniform Location 사용
    glUniform1i(soft_focus_uniforms_.uTexture, 0);
    glUniform2f(soft_focus_uniforms_.uSoftFocusTexelSize, 1.0f / width, 1.0f / height);
    glUniform1f(soft_focus_uniforms_.uSoftFocusStrength, strength);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, input_tex);

    renderFullscreenQuad();

    glBindTexture(GL_TEXTURE_2D, 0);
#else
    (void)input_tex;
    (void)output_fbo;
    (void)width;
    (void)height;
    (void)strength;
#endif
}

void GPUBeautyBackend::executeBrightnessPass(
    GLuint input_tex, GLuint output_fbo,
    int width, int height,
    float brightness) {

#if IRIS_SDK_GPU_AVAILABLE
    glBindFramebuffer(GL_FRAMEBUFFER, output_fbo);
    glUseProgram(brightness_program_);

    // 캐시된 Uniform Location 사용
    glUniform1i(brightness_uniforms_.uTexture, 0);
    glUniform1f(brightness_uniforms_.uBrightness, brightness);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, input_tex);

    renderFullscreenQuad();

    glBindTexture(GL_TEXTURE_2D, 0);
#else
    (void)input_tex;
    (void)output_fbo;
    (void)width;
    (void)height;
    (void)brightness;
#endif
}

void GPUBeautyBackend::applyMasking(
    GLuint filtered_tex, GLuint original_tex,
    GLuint mask_tex, GLuint output_fbo,
    int width, int height) {

#if IRIS_SDK_GPU_AVAILABLE
    glBindFramebuffer(GL_FRAMEBUFFER, output_fbo);
    glUseProgram(masking_program_);

    // 캐시된 Uniform Location 사용
    glUniform1i(masking_uniforms_.uFiltered, 0);
    glUniform1i(masking_uniforms_.uOriginal, 1);
    glUniform1i(masking_uniforms_.uMask, 2);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, filtered_tex);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, original_tex);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, mask_tex);

    renderFullscreenQuad();

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
#else
    (void)filtered_tex;
    (void)original_tex;
    (void)mask_tex;
    (void)output_fbo;
    (void)width;
    (void)height;
#endif
}

void GPUBeautyBackend::executeCombinedColorPass(
    GLuint input_tex, GLuint output_fbo,
    int width, int height,
    float brightness, float balance, float whitening,
    GLuint lut_texture, float lut_intensity) {

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

    // 캐시된 Uniform Location 사용
    glUniform1i(combined_color_uniforms_.uTexture, 0);
    glUniform1f(combined_color_uniforms_.uCombinedBrightness, brightness);
    glUniform1f(combined_color_uniforms_.uCombinedBalance, balance);
    glUniform1f(combined_color_uniforms_.uCombinedWhitening, whitening);

    // LUT: C++ controls activation - if no texture, force intensity to 0
    float effective_lut_intensity = (lut_texture != 0) ? lut_intensity : 0.0f;
    glUniform1f(combined_color_uniforms_.uCombinedLutIntensity, effective_lut_intensity);

    // P0-FIX: sampler2D(unit0) / sampler3D(unit1) 충돌 방지
    // uCombinedLutTexture는 LUT 활성 여부와 관계없이 항상 TEXTURE1에 바인딩.
    // LUT 비활성 시 uniform 기본값(0)이 TEXTURE0을 가리키면
    // sampler2D와 sampler3D가 동일 유닛을 공유 → GL_INVALID_OPERATION.
    glUniform1i(combined_color_uniforms_.uCombinedLutTexture, 1);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, input_tex);

    // TEXTURE1: LUT 텍스처 또는 neutral identity fallback
    glActiveTexture(GL_TEXTURE1);
    if (lut_texture != 0 && lut_intensity > 0.01f) {
        glBindTexture(GL_TEXTURE_3D, lut_texture);
    } else {
        // Neutral 1x1x1 identity LUT로 sampler3D 경로 안정화
        // glBindTexture(GL_TEXTURE_3D, 0) 대신 사용하여 드라이버 호환성 확보
        glBindTexture(GL_TEXTURE_3D, neutral_lut_texture_);
    }

    renderFullscreenQuad();

    // Cleanup: TEXTURE1 해제
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, 0);

#ifndef NDEBUG
    // 렌더링 후 에러 체크
    GLenum glErr = glGetError();
    if (glErr != GL_NO_ERROR) {
        LOGE("CombinedColor: GL error after render: 0x%x", glErr);
    }
#endif

    glBindTexture(GL_TEXTURE_2D, 0);
#else
    (void)input_tex;
    (void)output_fbo;
    (void)width;
    (void)height;
    (void)brightness;
    (void)balance;
    (void)whitening;
    (void)lut_texture;
    (void)lut_intensity;
#endif
}

GLuint GPUBeautyBackend::uploadSkinMask(
    const std::vector<uint8_t>& combined_mask,
    int mask_width, int mask_height) {

#if IRIS_SDK_GPU_AVAILABLE
    if (combined_mask.empty() || mask_width <= 0 || mask_height <= 0) {
        return 0;
    }

    const size_t expected_size = static_cast<size_t>(mask_width) * mask_height;
    if (combined_mask.size() < expected_size) {
        LOGE("uploadSkinMask: buffer size mismatch (got %zu, expected %zu)",
             combined_mask.size(), expected_size);
        return 0;
    }

    if (skin_mask_texture_ == 0) {
        glGenTextures(1, &skin_mask_texture_);
    }

    glBindTexture(GL_TEXTURE_2D, skin_mask_texture_);

    // GL_RED single channel: row width may not be 4-byte aligned
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    if (mask_width != skin_mask_width_ || mask_height != skin_mask_height_) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8,
                     mask_width, mask_height, 0,
                     GL_RED, GL_UNSIGNED_BYTE,
                     combined_mask.data());
        skin_mask_width_ = mask_width;
        skin_mask_height_ = mask_height;
    } else {
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                        mask_width, mask_height,
                        GL_RED, GL_UNSIGNED_BYTE,
                        combined_mask.data());
    }

    // Restore alignment
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glBindTexture(GL_TEXTURE_2D, 0);
    return skin_mask_texture_;
#else
    (void)combined_mask;
    (void)mask_width;
    (void)mask_height;
    return 0;
#endif
}

GPUBeautyBackend::FreqSepParams
GPUBeautyBackend::mapSkinQuality(float skin_quality, int face_width) {
    FreqSepParams p;

    if (skin_quality <= 0.0f) {
        p.enabled = false;
        return p;
    }

    p.enabled = true;

    // S-curve mapping (smoothstep for natural transition)
    float t = std::clamp(skin_quality, 0.0f, 1.0f);
    float s = t * t * (3.0f - 2.0f * t);  // smoothstep

    // blur_radius: face_width 대비 3~6%
    const float ratio = 0.03f + s * 0.03f;
    p.blur_radius = std::clamp(
        static_cast<int>(face_width * ratio),
        6, 28
    );

    // high_freq_preserve: 질감 보존 비율 (1.0=원본 → 0.30=70% 감쇠)
    constexpr float kMaxHighFreqAttenuation = 0.70f;
    p.high_freq_preserve = 1.0f - s * kMaxHighFreqAttenuation;

    // low_freq_smooth: 이중 블러 반경 비율
    p.low_freq_smooth_radius_ratio = 0.30f + s * 0.15f;

    // attenuation: 3-신호 감쇠 강도
    p.attenuation_low = 0.005f;
    p.attenuation_high = 0.03f + s * 0.03f;

    // Edge-aware attenuation weights (3-signal combination)
    p.edge_weight = 0.3f + s * 0.4f;     // 0.3 ~ 0.7
    p.chroma_weight = 0.2f + s * 0.3f;   // 0.2 ~ 0.5

    // Mid-tone lift: Council recommended 0.12~0.18 fixed → center 0.15
    // Very low skinQuality (≤0.1): gradual ramp for natural look
    // Use raw t (not smoothstep s) so threshold matches slider value 0.1
    // At t=0.1: t*1.5 = 0.15 = fixed value (continuous by design)
    p.tone_lift = (t > 0.1f) ? 0.15f : t * 1.5f;

    return p;
}

void GPUBeautyBackend::executeSmoothingWithFallbackStrength(
    GLuint input_tex, GLuint output_fbo,
    int width, int height,
    const BeautyFilterConfigV2& config) {

#if IRIS_SDK_GPU_AVAILABLE
    BeautyFilterConfigV2 fallback_config = config;
    float effective_smoothing = config.skinQuality * 0.5f;
    if (fallback_config.smoothing < effective_smoothing) {
        fallback_config.smoothing = effective_smoothing;
    }
    executeSmoothingPass(input_tex, output_fbo, width, height, fallback_config);
#else
    (void)input_tex; (void)output_fbo;
    (void)width; (void)height; (void)config;
#endif
}

bool GPUBeautyBackend::executeFreqSepPipeline(
    GLuint input_tex, GLuint mask_tex, GLuint output_fbo,
    int width, int height, const FreqSepParams& params) {
    return executeFreqSepPipelineImpl(input_tex, mask_tex, output_fbo,
        width, height, params, {1, false, "", ""});
}

// =============================================================================
// Device Tier 감지 (P4-W3-04)
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

GPUBeautyBackend::DeviceTier GPUBeautyBackend::detectDeviceTier() {
#if IRIS_SDK_GPU_AVAILABLE
    const char* renderer = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
    if (!renderer) return DeviceTier::LOW;
    LOGI("GPU Renderer: %s", renderer);
    return classifyGpuRenderer(std::string(renderer));
#else
    return DeviceTier::HIGH;
#endif
}

// =============================================================================
// MID 디바이스 하프 해상도 FreqSep 파이프라인 (P4-W3-04)
// =============================================================================

bool GPUBeautyBackend::executeFreqSepPipelineHalfRes(
    GLuint input_tex, GLuint mask_tex, GLuint output_fbo,
    int width, int height, const FreqSepParams& params) {
    return executeFreqSepPipelineImpl(input_tex, mask_tex, output_fbo,
        width, height, params, {2, true, "_Half", "_Full"});
}

// =============================================================================
// FreqSep 공통 구현 (full-res / half-res 통합)
// =============================================================================

bool GPUBeautyBackend::executeFreqSepPipelineImpl(
    GLuint input_tex, GLuint mask_tex, GLuint output_fbo,
    int width, int height,
    const FreqSepParams& params,
    const FreqSepExecConfig& cfg) {
#if IRIS_SDK_GPU_AVAILABLE
    const int blur_w = width / cfg.res_divisor;
    const int blur_h = height / cfg.res_divisor;

    // 최소 해상도 보장 (0 나누기 방지)
    if (blur_w < 1 || blur_h < 1) {
        LOGW("FreqSep: resolution too small (%dx%d, divisor=%d)", width, height, cfg.res_divisor);
        return false;
    }

    // Acquire intermediate buffers from texture pool
    auto* lowFreq = texture_pool_->acquireRenderTarget(blur_w, blur_h);
    auto* smoothedLow = texture_pool_->acquireRenderTarget(blur_w, blur_h);
    auto* temp = texture_pool_->acquireRenderTarget(blur_w, blur_h);

    if (!lowFreq || !smoothedLow || !temp) {
        LOGE("FreqSep: Failed to acquire render targets");
        if (lowFreq) texture_pool_->releaseTexture(lowFreq);
        if (smoothedLow) texture_pool_->releaseTexture(smoothedLow);
        if (temp) texture_pool_->releaseTexture(temp);
        return false;
    }

    bool profiling = profiler_ && profiler_->isEnabled();

    // blur_radius: half-res일 때 물리적 blur 범위 보존을 위해 축소
    const int blur_radius = (cfg.res_divisor == 1)
        ? params.blur_radius
        : std::max(3, params.blur_radius / cfg.res_divisor);

    // Precompute Gaussian weights (Pass 1a/1b)
    float weights[kMaxGaussianRadius + 1];
    computeGaussianWeights(blur_radius, weights);

    // half-res일 때 blur 패스 viewport 축소
    if (cfg.res_divisor > 1) {
        glViewport(0, 0, blur_w, blur_h);
    }

    // Profiler 태그 버퍼 (고정 크기, 스택 할당)
    char tag[48];

    // Pass 1a: Horizontal Gaussian → temp
    std::snprintf(tag, sizeof(tag), "FreqSep_GaussianH%s", cfg.blur_profiler_suffix);
    if (profiling) profiler_->begin(tag);
    glUseProgram(freq_sep_gaussian_program_);
    glUniform1i(freq_sep_gaussian_uniforms_.uTexture, 0);
    glUniform2f(freq_sep_gaussian_uniforms_.uDirection, 1.0f / blur_w, 0.0f);
    glUniform1i(freq_sep_gaussian_uniforms_.uRadius, blur_radius);
    glUniform1fv(freq_sep_gaussian_uniforms_.uWeights, 29, weights);
    glUniform1i(freq_sep_gaussian_uniforms_.uLinearize, 1);  // Pass 1a: sRGB→Linear 변환 활성화
    glBindFramebuffer(GL_FRAMEBUFFER, temp->fbo_id);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, input_tex);
    renderFullscreenQuad();
    if (profiling) profiler_->end(tag);

    // Pass 1b: Vertical Gaussian → lowFreq
    std::snprintf(tag, sizeof(tag), "FreqSep_GaussianV%s", cfg.blur_profiler_suffix);
    if (profiling) profiler_->begin(tag);
    glUniform1i(freq_sep_gaussian_uniforms_.uLinearize, 0);  // Pass 1b 이후: linearize 비활성화
    glUniform2f(freq_sep_gaussian_uniforms_.uDirection, 0.0f, 1.0f / blur_h);
    glBindFramebuffer(GL_FRAMEBUFFER, lowFreq->fbo_id);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, temp->texture_id);
    renderFullscreenQuad();
    if (profiling) profiler_->end(tag);

    // Precompute Gaussian weights for low_radius (Pass 2a/2b)
    int low_radius = std::max(3,
        static_cast<int>(blur_radius * params.low_freq_smooth_radius_ratio));
    computeGaussianWeights(low_radius, weights);

    // Pass 2a: Low Freq additional Gaussian H → temp
    std::snprintf(tag, sizeof(tag), "FreqSep_LowSmoothH%s", cfg.blur_profiler_suffix);
    if (profiling) profiler_->begin(tag);
    glUseProgram(freq_sep_gaussian_program_);
    glUniform1i(freq_sep_gaussian_uniforms_.uLinearize, 0);  // Pass 2a/2b: 이미 Linear — 변환 불필요
    glUniform1i(freq_sep_gaussian_uniforms_.uTexture, 0);
    glUniform2f(freq_sep_gaussian_uniforms_.uDirection, 1.0f / blur_w, 0.0f);
    glUniform1i(freq_sep_gaussian_uniforms_.uRadius, low_radius);
    glUniform1fv(freq_sep_gaussian_uniforms_.uWeights, 29, weights);
    glBindFramebuffer(GL_FRAMEBUFFER, temp->fbo_id);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, lowFreq->texture_id);
    renderFullscreenQuad();
    if (profiling) profiler_->end(tag);

    // Pass 2b: Low Freq additional Gaussian V → smoothedLow
    std::snprintf(tag, sizeof(tag), "FreqSep_LowSmoothV%s", cfg.blur_profiler_suffix);
    if (profiling) profiler_->begin(tag);
    glUniform2f(freq_sep_gaussian_uniforms_.uDirection, 0.0f, 1.0f / blur_h);
    glBindFramebuffer(GL_FRAMEBUFFER, smoothedLow->fbo_id);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, temp->texture_id);
    renderFullscreenQuad();
    if (profiling) profiler_->end(tag);

    // Composite 전 full-res viewport 복원
    if (cfg.res_divisor > 1) {
        glViewport(0, 0, width, height);
    }

    // Pass 3: Composite — re-synthesis + mask blending → output
    std::snprintf(tag, sizeof(tag), "FreqSep_Composite%s", cfg.composite_profiler_suffix);
    if (profiling) profiler_->begin(tag);

    // GL_LINEAR 업샘플링 (half-res → full-res composite 입력)
    if (cfg.linear_upsample) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, smoothedLow->texture_id);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, lowFreq->texture_id);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }

    glUseProgram(freq_sep_composite_program_);
    glUniform1f(freq_sep_composite_uniforms_.uHighFreqPreserve, params.high_freq_preserve);
    glUniform1f(freq_sep_composite_uniforms_.uAttenuationLow, params.attenuation_low);
    glUniform1f(freq_sep_composite_uniforms_.uAttenuationHigh, params.attenuation_high);
    glUniform1f(freq_sep_composite_uniforms_.uEdgeWeight, params.edge_weight);
    glUniform1f(freq_sep_composite_uniforms_.uChromaWeight, params.chroma_weight);
    glUniform1f(freq_sep_composite_uniforms_.uToneLift, params.tone_lift);

    glUniform1i(freq_sep_composite_uniforms_.uSmoothedLow, 0);
    glUniform1i(freq_sep_composite_uniforms_.uLowFreq, 1);
    glUniform1i(freq_sep_composite_uniforms_.uOriginal, 2);
    glUniform1i(freq_sep_composite_uniforms_.uSkinMask, 3);

    if (!cfg.linear_upsample) {
        // full-res: 텍스처 바인딩 필요
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, smoothedLow->texture_id);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, lowFreq->texture_id);
    }
    // linear_upsample 경로에서는 unit0/unit1 이미 바인딩됨

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, input_tex);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, mask_tex);

    glBindFramebuffer(GL_FRAMEBUFFER, output_fbo);
    renderFullscreenQuad();

    // Cleanup texture bindings
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
    if (profiling) profiler_->end(tag);

    // Release textures back to pool
    texture_pool_->releaseTexture(lowFreq);
    texture_pool_->releaseTexture(smoothedLow);
    texture_pool_->releaseTexture(temp);
    return true;
#else
    (void)input_tex; (void)mask_tex; (void)output_fbo;
    (void)width; (void)height; (void)params; (void)cfg;
    return false;
#endif
}

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
        return IRIS_SDK_ERROR_NOT_INITIALIZED;
    }

    if (input_texture == 0 || !output_texture) {
        return IRIS_SDK_INVALID_PARAM;
    }

    if (width <= 0 || height <= 0) {
        return IRIS_SDK_INVALID_PARAM;
    }

    if (!config.enabled) {
        *output_texture = input_texture;
        return IRIS_SDK_OK;
    }

#if IRIS_SDK_GPU_AVAILABLE
    // GLSurfaceView 모드에서는 이미 EGL 컨텍스트가 바인딩되어 있음
    if (render_context_) {
        render_context_->makeCurrent();
    }

    // TextureHandle 생성 (입력)
    GLuint input_tex_id = static_cast<GLuint>(input_texture);
    TextureHandle input_handle;
    input_handle.native_handle = &input_tex_id;
    input_handle.type = TextureHandle::Type::OpenGLES;
    input_handle.width = width;
    input_handle.height = height;
    input_handle.format = TextureFormat::RGBA8;

    // ROI 생성 (detection이 있는 경우)
    BeautyROI roi;
    BeautyROI* roi_ptr = nullptr;

    if (detection && detection->detected && config.roiOnly) {
        int face_x = static_cast<int>(detection->face_rect.x * width);
        int face_y = static_cast<int>(detection->face_rect.y * height);
        int face_w = static_cast<int>(detection->face_rect.width * width);
        int face_h = static_cast<int>(detection->face_rect.height * height);

        int margin_x = face_w / 5;
        int margin_y = face_h / 5;
        face_x = std::max(0, face_x - margin_x);
        face_y = std::max(0, face_y - margin_y);
        face_w = std::min(width - face_x, face_w + 2 * margin_x);
        face_h = std::min(height - face_y, face_h + 2 * margin_y);

        roi.face_rect = Rect{
            static_cast<float>(face_x),
            static_cast<float>(face_y),
            static_cast<float>(face_w),
            static_cast<float>(face_h)
        };
        roi.mask_width = face_w;
        roi.mask_height = face_h;
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

    // 이전 프레임의 출력 텍스처 반환 (텍스처 풀 관리)
    // GPU 동기화: glFenceSync로 이전 프레임 렌더링 완료 대기 (non-blocking)
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
    if (previous_output_ping_ != nullptr) {
        texture_pool_->releaseTexture(previous_output_ping_);
        previous_output_ping_ = nullptr;
    }
    if (previous_output_pong_ != nullptr) {
        texture_pool_->releaseTexture(previous_output_pong_);
        previous_output_pong_ = nullptr;
    }

    // 활성 필터 수에 따라 동적으로 텍스처 할당
    int active_filter_count = 0;
    if (config.skinQuality > 0.0f || config.smoothing > 0.01f) active_filter_count++;
    bool needsBrightness = std::abs(config.brightness - 1.0f) > 0.01f;
    bool needsBalance = std::abs(config.colorBalance) > 0.01f;
    bool needsWhitening = config.whitening > 0.01f;
    bool needsLut = (lut_texture_id != 0 && lut_intensity > 0.01f);
    if (needsBrightness || needsBalance || needsWhitening || needsLut) active_filter_count++;
    if (config.softFocus > 0.01f) active_filter_count++;

    // 필터 0개: 패스스루 (텍스처 할당 불필요)
    if (active_filter_count == 0) {
        *output_texture = input_tex_id;
        return IRIS_SDK_OK;
    }

    // 필터 1개: 단일 텍스처, 2개+: ping-pong 버퍼
    TexturePool::TextureInfo* ping = nullptr;
    TexturePool::TextureInfo* pong = nullptr;

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
    if (roi_ptr && roi_ptr->valid) {
        executeCombinedColorPass(input_tex_id, ping->fbo_id,
                                 width, height,
                                 1.0f, 0.0f, 0.0f, 0, 0.0f);
        if (pong) {
            executeCombinedColorPass(input_tex_id, pong->fbo_id,
                                     width, height,
                                     1.0f, 0.0f, 0.0f, 0, 0.0f);
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
            // roiOnly 정책: 교집합이 비어있으면 필터 처리를 건너뛰고
            // pre-fill된 원본(passthrough)을 그대로 반환
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

    // 1. 스무딩: Freq Sep (skinQuality > 0) 또는 Bilateral (기존)
    // Compute Freq Sep params from skinQuality
    int face_w = 0;
    if (roi_ptr && roi_ptr->valid) {
        face_w = static_cast<int>(roi_ptr->face_rect.width);
    }
    FreqSepParams freq_sep_params = mapSkinQuality(config.skinQuality, face_w);

    // Temporal stability: One Euro Filter for blur_radius (P4-W3-04)
    // frame_ts는 위에서 캡처된 동일 프레임 타임스탬프
    if (freq_sep_params.enabled) {
        float raw_radius = static_cast<float>(freq_sep_params.blur_radius);
        float filtered_radius = skin_radius_filter_.filter(raw_radius, frame_ts);
        freq_sep_params.blur_radius = static_cast<int>(std::round(filtered_radius));
        freq_sep_params.blur_radius = std::max(3, freq_sep_params.blur_radius);
    }

    // Bilateral fallback 헬퍼 (FreqSep 실패 시 공통 경로)
    auto runBilateralFallback = [&]() {
        if (profiling) profiler_->begin("Smoothing_Fallback");
        executeSmoothingWithFallbackStrength(current_input, current_output->fbo_id,
                                              width, height, config);
        if (profiling) profiler_->end("Smoothing_Fallback");
        current_input = current_output->texture_id;
        if (pong) current_output = (current_output == ping) ? pong : ping;
    };

    if (freq_sep_params.enabled
        && freq_sep_gaussian_program_ != 0 && freq_sep_composite_program_ != 0
        && roi_ptr && roi_ptr->valid && !roi_ptr->combined_mask.empty()) {

        // LOW tier: FreqSep 전체 건너뛰기 → Bilateral fallback
        if (device_tier_ == DeviceTier::LOW) {
            runBilateralFallback();
        } else {
            // HIGH 또는 MID: skin mask 업로드 후 파이프라인 실행
            GLuint mask_tex = uploadSkinMask(
                roi_ptr->combined_mask,
                roi_ptr->mask_width, roi_ptr->mask_height);
            if (mask_tex != 0) {
                // FreqSep 멀티패스 Gaussian은 전체 프레임 중간 텍스처가 필요하므로
                // ROI scissor를 비활성화 (Composite 셰이더의 uSkinMask가 ROI 마스킹 담당)
                if (scissor_active) {
                    glDisable(GL_SCISSOR_TEST);
                }

                bool freq_sep_ok = false;
                if (device_tier_ == DeviceTier::MID) {
                    // MID: blur half-res, composite full-res (하이브리드 해상도)
                    freq_sep_ok = executeFreqSepPipelineHalfRes(
                        current_input, mask_tex,
                        current_output->fbo_id,
                        width, height, freq_sep_params);
                } else {
                    // HIGH: full resolution
                    freq_sep_ok = executeFreqSepPipeline(
                        current_input, mask_tex,
                        current_output->fbo_id,
                        width, height, freq_sep_params);
                }

                // Scissor 복원
                if (scissor_active) {
                    glEnable(GL_SCISSOR_TEST);
                }
                if (freq_sep_ok) {
                    current_input = current_output->texture_id;
                    if (pong) current_output = (current_output == ping) ? pong : ping;
                } else {
                    runBilateralFallback();
                }
            } else {
                // mask upload failed → Bilateral fallback
                runBilateralFallback();
            }
        }
    } else if (freq_sep_params.enabled) {
        // skinQuality > 0 but FreqSep cannot run (shader not compiled / no valid ROI/mask)
        // → Bilateral fallback with minimum strength
        runBilateralFallback();
    } else if (config.smoothing > 0.01f) {
        // Original Bilateral path (skinQuality = 0)
        if (profiling) profiler_->begin("Smoothing");
        executeSmoothingPass(current_input, current_output->fbo_id,
                             width, height, config);
        if (profiling) profiler_->end("Smoothing");
        current_input = current_output->texture_id;
        if (pong) current_output = (current_output == ping) ? pong : ping;
    }

    // 2. 통합 Color Adjustment (Brightness + ColorBalance + Whitening + LUT)
    //    기존 3개 패스를 1개로 병합하여 FBO 전환 오버헤드 감소
    if (needsBrightness || needsBalance || needsWhitening || needsLut) {
        if (profiling) profiler_->begin("CombinedColor");
        executeCombinedColorPass(current_input, current_output->fbo_id,
                                 width, height,
                                 config.brightness,
                                 config.colorBalance,
                                 config.whitening,
                                 static_cast<GLuint>(lut_texture_id),
                                 lut_intensity);
        if (profiling) profiler_->end("CombinedColor");
        current_input = current_output->texture_id;
        if (pong) current_output = (current_output == ping) ? pong : ping;
    }

    // 3. 소프트 포커스 - 단독 패스 (blur 필요)
    if (config.softFocus > 0.01f) {
        if (profiling) profiler_->begin("SoftFocus");
        executeSoftFocusPass(current_input, current_output->fbo_id,
                             width, height, config.softFocus);
        if (profiling) profiler_->end("SoftFocus");
        current_input = current_output->texture_id;
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
    previous_output_texture_ = current_input;

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
        return IRIS_SDK_ERROR_NOT_INITIALIZED;
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

    // Face Warp 미구현 — P4에서 Face Mesh 기반 메시 워핑으로 구현 예정
    LOGW("GPUBeautyBackend::applyFaceWarp() not supported. Will be implemented in P4.");

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
