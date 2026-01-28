/**
 * @file gpu_beauty_backend.cpp
 * @brief GPUBeautyBackend 구현
 */

#include "iris_sdk/gpu/gpu_beauty_backend.h"
#include "iris_sdk/gpu/render_context.h"

#if IRIS_SDK_GPU_AVAILABLE
#include "iris_sdk/gpu/gles_render_context.h"
#include <android/log.h>
#define LOG_TAG "GPUBeautyBackend"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#include <cstdio>
#define LOGI(...) printf("[GPUBeautyBackend INFO] " __VA_ARGS__); printf("\n")
#define LOGW(...) printf("[GPUBeautyBackend WARN] " __VA_ARGS__); printf("\n")
#define LOGE(...) printf("[GPUBeautyBackend ERROR] " __VA_ARGS__); printf("\n")
#endif

namespace iris_sdk {

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

    if (!render_context) {
        LOGE("RenderContext is null");
        return false;
    }

#if IRIS_SDK_GPU_AVAILABLE
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

    // 풀스크린 쿼드 설정
    setupFullscreenQuad();

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

    LOGI("All %zu shader programs created successfully",
         shader_manager_->getCachedProgramCount());
    return true;
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

    render_context_ = nullptr;
    initialized_ = false;

    // 프로그램 ID 초기화
    passthrough_program_ = 0;
    smoothing_program_ = 0;
    whitening_program_ = 0;
    color_balance_program_ = 0;
    soft_focus_program_ = 0;
    brightness_program_ = 0;
    masking_program_ = 0;

    LOGI("GPUBeautyBackend released");
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

    // CPU 버퍼 처리는 텍스처 업로드/다운로드가 필요하여 성능이 낮음
    // 실제 구현에서는 텍스처로 업로드 → 처리 → 다운로드
    // 여기서는 기본 프레임워크만 구현

#if IRIS_SDK_GPU_AVAILABLE
    render_context_->makeCurrent();

    // TODO: 텍스처 업로드 → applyTexture 호출 → 다운로드
    // 현재는 stub 구현

    LOGW("CPU buffer processing not yet implemented, use applyTexture for GPU processing");
#endif

    (void)format;
    (void)roi;

    return IRIS_SDK_OK;
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
    render_context_->makeCurrent();

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

    // 필터 체인 실행
    // 1. 스무딩 (Bilateral Filter)
    if (config.smoothing > 0.01f) {
        executeSmoothingPass(current_input, current_output->fbo_id,
                             width, height, config);
        current_input = current_output->texture_id;
        current_output = (current_output == ping) ? pong : ping;
    }

    // 2. 화이트닝
    if (config.whitening > 0.01f) {
        executeWhiteningPass(current_input, current_output->fbo_id,
                             width, height, config.whitening);
        current_input = current_output->texture_id;
        current_output = (current_output == ping) ? pong : ping;
    }

    // 3. 컬러 밸런스
    if (std::abs(config.colorBalance) > 0.01f) {
        executeColorBalancePass(current_input, current_output->fbo_id,
                                width, height, config.colorBalance);
        current_input = current_output->texture_id;
        current_output = (current_output == ping) ? pong : ping;
    }

    // 4. 소프트 포커스
    if (config.softFocus > 0.01f) {
        executeSoftFocusPass(current_input, current_output->fbo_id,
                             width, height, config.softFocus);
        current_input = current_output->texture_id;
        current_output = (current_output == ping) ? pong : ping;
    }

    // 5. 밝기
    if (std::abs(config.brightness - 1.0f) > 0.01f) {
        executeBrightnessPass(current_input, current_output->fbo_id,
                              width, height, config.brightness);
        current_input = current_output->texture_id;
    }

    // 출력 텍스처 핸들 설정
    // 주의: 이 텍스처는 풀에서 관리되므로 사용 후 반환 필요
    output.native_handle = new GLuint(current_input);
    output.type = TextureHandle::Type::OpenGLES;
    output.width = width;
    output.height = height;
    output.format = input.format;

    // 텍스처 반환 (현재 output으로 사용 중인 것 제외)
    // 실제 구현에서는 호출자가 output 사용 후 반환해야 함

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

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

    // Uniforms 설정
    glUniform1i(glGetUniformLocation(smoothing_program_, "uTexture"), 0);
    glUniform2f(glGetUniformLocation(smoothing_program_, "uTexelSize"),
                1.0f / width, 1.0f / height);
    glUniform1f(glGetUniformLocation(smoothing_program_, "uStrength"), config.smoothing);

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

    glUniform1i(glGetUniformLocation(whitening_program_, "uTexture"), 0);
    glUniform1f(glGetUniformLocation(whitening_program_, "uStrength"), whitening);

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

    glUniform1i(glGetUniformLocation(color_balance_program_, "uTexture"), 0);
    glUniform1f(glGetUniformLocation(color_balance_program_, "uBalance"), balance);

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

    glUniform1i(glGetUniformLocation(soft_focus_program_, "uTexture"), 0);
    glUniform2f(glGetUniformLocation(soft_focus_program_, "uTexelSize"),
                1.0f / width, 1.0f / height);
    glUniform1f(glGetUniformLocation(soft_focus_program_, "uStrength"), strength);

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

    glUniform1i(glGetUniformLocation(brightness_program_, "uTexture"), 0);
    glUniform1f(glGetUniformLocation(brightness_program_, "uBrightness"), brightness);

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

    glUniform1i(glGetUniformLocation(masking_program_, "uFiltered"), 0);
    glUniform1i(glGetUniformLocation(masking_program_, "uOriginal"), 1);
    glUniform1i(glGetUniformLocation(masking_program_, "uMask"), 2);

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

void GPUBeautyBackend::onMemoryPressure(int level) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (texture_pool_) {
        texture_pool_->onMemoryPressure(level);
    }
}

} // namespace iris_sdk
