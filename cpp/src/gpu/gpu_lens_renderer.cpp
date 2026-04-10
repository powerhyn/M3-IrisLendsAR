/**
 * @file gpu_lens_renderer.cpp
 * @brief GPULensRenderer 구현
 *
 * Android 데모 CameraGLRenderer.kt의 GPU 렌즈 파이프라인을
 * C++ SDK 코어로 포팅한 구현체.
 */

#include "iris_sdk/gpu/gpu_lens_renderer.h"
#include "iris_sdk/gpu/render_context.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

#if IRIS_SDK_GPU_AVAILABLE
#include "iris_sdk/gpu/gles_render_context.h"
#include <android/log.h>
#define LOG_TAG "GPULensRenderer"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#include <cstdio>
#define LOGD(...) do { printf("[GPULensRenderer DEBUG] " __VA_ARGS__); printf("\n"); } while(0)
#define LOGI(...) do { printf("[GPULensRenderer INFO] " __VA_ARGS__); printf("\n"); } while(0)
#define LOGW(...) do { printf("[GPULensRenderer WARN] " __VA_ARGS__); printf("\n"); } while(0)
#define LOGE(...) do { printf("[GPULensRenderer ERROR] " __VA_ARGS__); printf("\n"); } while(0)
#endif

namespace iris_sdk {

// 셰이더 소스 extern 선언
namespace shaders {
extern const char* FULLSCREEN_QUAD_VERTEX;
extern const char* LENS_OVERLAY_VERTEX;
extern const char* LENS_OVERLAY_FRAGMENT;
}

// ============================================================================
// Face Mesh 랜드마크 인덱스 상수
// ============================================================================

// 왼쪽 눈 16개 윤곽 인덱스
static constexpr int LEFT_EYE_CONTOUR[] = {
    33, 246, 161, 160, 159, 158, 157, 173,
    133, 155, 154, 153, 145, 144, 163, 7
};
static constexpr int LEFT_EYE_CONTOUR_COUNT = 16;

// 오른쪽 눈 16개 윤곽 인덱스
static constexpr int RIGHT_EYE_CONTOUR[] = {
    263, 466, 388, 387, 386, 385, 384, 398,
    362, 382, 381, 380, 374, 373, 390, 249
};
static constexpr int RIGHT_EYE_CONTOUR_COUNT = 16;

// 내안각/외안각 인덱스
static constexpr int LEFT_INNER_CORNER = 33;
static constexpr int LEFT_OUTER_CORNER = 133;
static constexpr int RIGHT_INNER_CORNER = 263;
static constexpr int RIGHT_OUTER_CORNER = 362;

// 눈꺼풀 Y 좌표 인덱스
static constexpr int LEFT_UPPER_EYELID[] = {159, 160, 161};
static constexpr int LEFT_LOWER_EYELID[] = {145, 144, 153};
static constexpr int RIGHT_UPPER_EYELID[] = {386, 385, 384};
static constexpr int RIGHT_LOWER_EYELID[] = {374, 373, 380};
static constexpr int EYELID_INDEX_COUNT = 3;

// ============================================================================
// 생성자/소멸자
// ============================================================================

GPULensRenderer::GPULensRenderer() = default;

GPULensRenderer::~GPULensRenderer() {
    release();
}

// ============================================================================
// 라이프사이클
// ============================================================================

bool GPULensRenderer::initialize(IRenderContext* render_context) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (initialized_) {
        LOGW("GPULensRenderer already initialized");
        return true;
    }

#if IRIS_SDK_GPU_AVAILABLE
    if (render_context) {
        render_context_ = dynamic_cast<GLESRenderContext*>(render_context);
        if (!render_context_) {
            LOGE("RenderContext is not GLESRenderContext");
            return false;
        }

        if (!render_context_->makeCurrent()) {
            LOGE("Failed to make GL context current");
            return false;
        }
    } else {
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
        LOGE("Failed to initialize lens shaders");
        shader_manager_.reset();
        return false;
    }

    // 텍스처 풀 초기화 (최대 4개, 1920x1080)
    texture_pool_ = std::make_unique<TexturePool>();
    if (!texture_pool_->initialize(4, 1920, 1080)) {
        LOGE("Failed to initialize texture pool");
        shader_manager_->releaseAll();
        shader_manager_.reset();
        texture_pool_.reset();
        return false;
    }

    // 풀스크린 쿼드 설정
    setupFullscreenQuad();

    // Uniform Location 캐싱
    cacheLensUniforms();

    initialized_ = true;
    LOGI("GPULensRenderer initialized successfully");
    return true;
}

void GPULensRenderer::release() {
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

    // 렌즈 텍스처 해제
    if (lens_texture_ != 0) {
        glDeleteTextures(1, &lens_texture_);
        lens_texture_ = 0;
    }
    lens_texture_width_ = 0;
    lens_texture_height_ = 0;
#endif

    // 셰이더/텍스처 풀 해제
    if (shader_manager_) {
        shader_manager_->releaseAll();
        shader_manager_.reset();
    }
    if (texture_pool_) {
        texture_pool_->release();
        texture_pool_.reset();
    }

    lens_program_ = 0;
    previous_output_texture_ = 0;

    // 필터 리셋
    for (int i = 0; i < 2; ++i) {
        eyelid_top_filters_[i].reset();
        eyelid_bottom_filters_[i].reset();
        ellipse_filters_[i].cx.reset();
        ellipse_filters_[i].cy.reset();
        ellipse_filters_[i].rx_inner.reset();
        ellipse_filters_[i].rx_outer.reset();
        ellipse_filters_[i].ry.reset();
        ellipse_filters_[i].rotation.reset();
        eyelid_cache_[i] = EyelidCache{};
        ellipse_cache_[i] = EllipseCache{};
    }

    initialized_ = false;
    LOGI("GPULensRenderer released");
}

bool GPULensRenderer::isInitialized() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return initialized_;
}

// ============================================================================
// 텍스처 관리
// ============================================================================

bool GPULensRenderer::loadLensTexture(const uint8_t* data, int width, int height) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_) {
        LOGE("loadLensTexture: not initialized");
        return false;
    }
    if (!data || width <= 0 || height <= 0) {
        LOGE("loadLensTexture: invalid parameters");
        return false;
    }

#if IRIS_SDK_GPU_AVAILABLE
    if (render_context_) {
        render_context_->makeCurrent();
    }

    // 기존 텍스처가 있으면 해제
    if (lens_texture_ != 0) {
        glDeleteTextures(1, &lens_texture_);
        lens_texture_ = 0;
    }

    glGenTextures(1, &lens_texture_);
    glBindTexture(GL_TEXTURE_2D, lens_texture_);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, data);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glBindTexture(GL_TEXTURE_2D, 0);

    lens_texture_width_ = width;
    lens_texture_height_ = height;

    LOGI("Lens texture loaded: %dx%d, id=%u", width, height, lens_texture_);
    return true;
#else
    lens_texture_ = 1; // stub
    lens_texture_width_ = width;
    lens_texture_height_ = height;
    LOGI("Lens texture loaded (stub): %dx%d", width, height);
    return true;
#endif
}

void GPULensRenderer::unloadLensTexture() {
    std::lock_guard<std::mutex> lock(mutex_);

#if IRIS_SDK_GPU_AVAILABLE
    if (lens_texture_ != 0) {
        if (render_context_) {
            render_context_->makeCurrent();
        }
        glDeleteTextures(1, &lens_texture_);
        lens_texture_ = 0;
    }
#else
    lens_texture_ = 0;
#endif
    lens_texture_width_ = 0;
    lens_texture_height_ = 0;
    LOGI("Lens texture unloaded");
}

bool GPULensRenderer::hasLensTexture() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return lens_texture_ != 0;
}

// ============================================================================
// 셰이더 초기화
// ============================================================================

bool GPULensRenderer::initializeShaders() {
    // 렌즈 오버레이 셰이더
    if (!shader_manager_->createProgram(
            shaders::LENS_OVERLAY_VERTEX,
            shaders::LENS_OVERLAY_FRAGMENT,
            lens_program_)) {
        LOGE("Failed to create lens overlay program");
        return false;
    }
    shader_manager_->cacheProgram("lens_overlay", lens_program_);
    LOGI("Lens overlay shader compiled: program=%u", lens_program_);
    return true;
}

void GPULensRenderer::cacheLensUniforms() {
#if IRIS_SDK_GPU_AVAILABLE
    if (lens_program_ == 0) return;

    lens_uniforms_.uCameraTexture = glGetUniformLocation(lens_program_, "uCameraTexture");
    lens_uniforms_.uLensTexture = glGetUniformLocation(lens_program_, "uLensTexture");

    lens_uniforms_.uLeftIrisCenter = glGetUniformLocation(lens_program_, "uLeftIrisCenter");
    lens_uniforms_.uLeftIrisRadius = glGetUniformLocation(lens_program_, "uLeftIrisRadius");
    lens_uniforms_.uRightIrisCenter = glGetUniformLocation(lens_program_, "uRightIrisCenter");
    lens_uniforms_.uRightIrisRadius = glGetUniformLocation(lens_program_, "uRightIrisRadius");

    lens_uniforms_.uOpacity = glGetUniformLocation(lens_program_, "uOpacity");
    lens_uniforms_.uLensScale = glGetUniformLocation(lens_program_, "uLensScale");
    lens_uniforms_.uEdgeFeather = glGetUniformLocation(lens_program_, "uEdgeFeather");
    lens_uniforms_.uBlendMode = glGetUniformLocation(lens_program_, "uBlendMode");
    lens_uniforms_.uApplyLeft = glGetUniformLocation(lens_program_, "uApplyLeft");
    lens_uniforms_.uApplyRight = glGetUniformLocation(lens_program_, "uApplyRight");
    lens_uniforms_.uFrameAspect = glGetUniformLocation(lens_program_, "uFrameAspect");

    lens_uniforms_.uLeftEyeTop = glGetUniformLocation(lens_program_, "uLeftEyeTop");
    lens_uniforms_.uLeftEyeBottom = glGetUniformLocation(lens_program_, "uLeftEyeBottom");
    lens_uniforms_.uRightEyeTop = glGetUniformLocation(lens_program_, "uRightEyeTop");
    lens_uniforms_.uRightEyeBottom = glGetUniformLocation(lens_program_, "uRightEyeBottom");
    lens_uniforms_.uEyelidFeather = glGetUniformLocation(lens_program_, "uEyelidFeather");

    lens_uniforms_.uScleraProtect = glGetUniformLocation(lens_program_, "uScleraProtect");
    lens_uniforms_.uContactShadow = glGetUniformLocation(lens_program_, "uContactShadow");
    lens_uniforms_.uShadowIntensity = glGetUniformLocation(lens_program_, "uShadowIntensity");
    lens_uniforms_.uMaxDetail = glGetUniformLocation(lens_program_, "uMaxDetail");

    lens_uniforms_.uUseEllipseMask = glGetUniformLocation(lens_program_, "uUseEllipseMask");
    lens_uniforms_.uLeftEyeEllipseCenter = glGetUniformLocation(lens_program_, "uLeftEyeEllipseCenter");
    lens_uniforms_.uLeftEyeEllipseRadii = glGetUniformLocation(lens_program_, "uLeftEyeEllipseRadii");
    lens_uniforms_.uLeftEyeEllipseRot = glGetUniformLocation(lens_program_, "uLeftEyeEllipseRot");
    lens_uniforms_.uRightEyeEllipseCenter = glGetUniformLocation(lens_program_, "uRightEyeEllipseCenter");
    lens_uniforms_.uRightEyeEllipseRadii = glGetUniformLocation(lens_program_, "uRightEyeEllipseRadii");
    lens_uniforms_.uRightEyeEllipseRot = glGetUniformLocation(lens_program_, "uRightEyeEllipseRot");

    lens_uniforms_.uAvgIrisLum = glGetUniformLocation(lens_program_, "uAvgIrisLum");
    lens_uniforms_.uDetH = glGetUniformLocation(lens_program_, "uDetH");

    LOGI("Lens uniforms cached (%d locations)",
         (lens_uniforms_.uCameraTexture != -1 ? 1 : 0) +
         (lens_uniforms_.uLensTexture != -1 ? 1 : 0) +
         (lens_uniforms_.uLeftIrisCenter != -1 ? 1 : 0));
#endif
}

// ============================================================================
// 풀스크린 쿼드
// ============================================================================

void GPULensRenderer::setupFullscreenQuad() {
#if IRIS_SDK_GPU_AVAILABLE
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

void GPULensRenderer::renderFullscreenQuad() {
#if IRIS_SDK_GPU_AVAILABLE
    glBindVertexArray(quad_vao_);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
#endif
}

// ============================================================================
// 설정
// ============================================================================

void GPULensRenderer::setScleraProtectEnabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    sclera_protect_ = enabled;
}

void GPULensRenderer::setContactShadowEnabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    contact_shadow_ = enabled;
}

void GPULensRenderer::setContactShadowIntensity(float intensity) {
    std::lock_guard<std::mutex> lock(mutex_);
    shadow_intensity_ = std::clamp(intensity, 0.0f, 1.0f);
}

void GPULensRenderer::setEllipseMaskEnabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    use_ellipse_mask_ = enabled;
}

// ============================================================================
// 타원 피팅 (Static)
// ============================================================================

void GPULensRenderer::fitEyeEllipse(
    const IrisLandmark* face_mesh, bool is_left_eye,
    float& out_cx, float& out_cy,
    float& out_rx_inner, float& out_rx_outer,
    float& out_ry, float& out_rotation)
{
    const int* contour = is_left_eye ? LEFT_EYE_CONTOUR : RIGHT_EYE_CONTOUR;
    const int contour_count = is_left_eye ? LEFT_EYE_CONTOUR_COUNT : RIGHT_EYE_CONTOUR_COUNT;

    int inner_idx = is_left_eye ? LEFT_INNER_CORNER : RIGHT_INNER_CORNER;
    int outer_idx = is_left_eye ? LEFT_OUTER_CORNER : RIGHT_OUTER_CORNER;

    // Step 1: center = mean of 16 contour points
    float sum_x = 0.0f, sum_y = 0.0f;
    for (int i = 0; i < contour_count; ++i) {
        sum_x += face_mesh[contour[i]].x;
        sum_y += face_mesh[contour[i]].y;
    }
    out_cx = sum_x / static_cast<float>(contour_count);
    out_cy = sum_y / static_cast<float>(contour_count);

    // Step 2: rotation = atan2(outerY - innerY, outerX - innerX)
    float inner_x = face_mesh[inner_idx].x;
    float inner_y = face_mesh[inner_idx].y;
    float outer_x = face_mesh[outer_idx].x;
    float outer_y = face_mesh[outer_idx].y;

    out_rotation = std::atan2(outer_y - inner_y, outer_x - inner_x);

    // Step 3: rx_inner / rx_outer
    float dx_inner = inner_x - out_cx;
    float dy_inner = inner_y - out_cy;
    float dist_inner = std::sqrt(dx_inner * dx_inner + dy_inner * dy_inner);

    float dx_outer = outer_x - out_cx;
    float dy_outer = outer_y - out_cy;
    float dist_outer = std::sqrt(dx_outer * dx_outer + dy_outer * dy_outer);

    out_rx_inner = dist_inner * 0.85f;
    out_rx_outer = dist_outer * 1.0f;

    // Step 4: ry = max abs Y in rotated frame across all 16 points
    float cos_r = std::cos(-out_rotation);
    float sin_r = std::sin(-out_rotation);
    float max_ry = 0.0f;

    for (int i = 0; i < contour_count; ++i) {
        float px = face_mesh[contour[i]].x - out_cx;
        float py = face_mesh[contour[i]].y - out_cy;
        // 회전 적용
        float rotated_y = -px * sin_r + py * cos_r;
        max_ry = std::max(max_ry, std::abs(rotated_y));
    }

    out_ry = max_ry;
}

float GPULensRenderer::medianLandmarkY(
    const IrisLandmark* face_mesh, const int* indices, int count)
{
    if (count <= 0) return 0.0f;

    // 작은 배열이므로 정렬 후 중앙값
    std::array<float, 8> values{};
    int n = std::min(count, static_cast<int>(values.size()));
    for (int i = 0; i < n; ++i) {
        values[i] = face_mesh[indices[i]].y;
    }
    std::sort(values.begin(), values.begin() + n);

    if (n % 2 == 0) {
        return (values[n / 2 - 1] + values[n / 2]) * 0.5f;
    }
    return values[n / 2];
}

// ============================================================================
// 눈꺼풀/타원 캐시 업데이트
// ============================================================================

void GPULensRenderer::updateEyelidCache(const IrisResult& iris_result) {
    if (!iris_result.face_mesh_valid) {
        // 검출 실패: 홀드 프레임 감소
        for (int i = 0; i < 2; ++i) {
            if (eyelid_cache_[i].valid_frames > 0) {
                eyelid_cache_[i].valid_frames--;
            }
        }
        return;
    }

    // 왼쪽 눈
    if (iris_result.left_detected) {
        float top_raw = medianLandmarkY(iris_result.face_mesh,
                                         LEFT_UPPER_EYELID, EYELID_INDEX_COUNT);
        float bot_raw = medianLandmarkY(iris_result.face_mesh,
                                         LEFT_LOWER_EYELID, EYELID_INDEX_COUNT);
        eyelid_cache_[0].top = eyelid_top_filters_[0].filter(top_raw);
        eyelid_cache_[0].bottom = eyelid_bottom_filters_[0].filter(bot_raw);
        eyelid_cache_[0].valid_frames = EYELID_HOLD_FRAMES;
    } else if (eyelid_cache_[0].valid_frames > 0) {
        eyelid_cache_[0].valid_frames--;
    }

    // 오른쪽 눈
    if (iris_result.right_detected) {
        float top_raw = medianLandmarkY(iris_result.face_mesh,
                                         RIGHT_UPPER_EYELID, EYELID_INDEX_COUNT);
        float bot_raw = medianLandmarkY(iris_result.face_mesh,
                                         RIGHT_LOWER_EYELID, EYELID_INDEX_COUNT);
        eyelid_cache_[1].top = eyelid_top_filters_[1].filter(top_raw);
        eyelid_cache_[1].bottom = eyelid_bottom_filters_[1].filter(bot_raw);
        eyelid_cache_[1].valid_frames = EYELID_HOLD_FRAMES;
    } else if (eyelid_cache_[1].valid_frames > 0) {
        eyelid_cache_[1].valid_frames--;
    }
}

void GPULensRenderer::updateEllipseCache(const IrisResult& iris_result) {
    if (!iris_result.face_mesh_valid) {
        for (int i = 0; i < 2; ++i) {
            if (ellipse_cache_[i].valid_frames > 0) {
                ellipse_cache_[i].valid_frames--;
            }
        }
        return;
    }

    for (int eye = 0; eye < 2; ++eye) {
        bool detected = (eye == 0) ? iris_result.left_detected : iris_result.right_detected;
        if (!detected) {
            if (ellipse_cache_[eye].valid_frames > 0) {
                ellipse_cache_[eye].valid_frames--;
            }
            continue;
        }

        float cx, cy, rx_inner, rx_outer, ry, rotation;
        fitEyeEllipse(iris_result.face_mesh, (eye == 0),
                       cx, cy, rx_inner, rx_outer, ry, rotation);

        // OneEuroFilter 스무딩
        ellipse_cache_[eye].cx = ellipse_filters_[eye].cx.filter(cx);
        ellipse_cache_[eye].cy = ellipse_filters_[eye].cy.filter(cy);
        ellipse_cache_[eye].rx_inner = ellipse_filters_[eye].rx_inner.filter(rx_inner);
        ellipse_cache_[eye].rx_outer = ellipse_filters_[eye].rx_outer.filter(rx_outer);
        ellipse_cache_[eye].ry = ellipse_filters_[eye].ry.filter(ry);
        ellipse_cache_[eye].rotation = ellipse_filters_[eye].rotation.filter(rotation);
        ellipse_cache_[eye].valid_frames = EYELID_HOLD_FRAMES;
    }
}

// ============================================================================
// 렌더링
// ============================================================================

ErrorCode GPULensRenderer::renderToTexture(
    uint32_t input_texture,
    uint32_t* output_texture,
    int width, int height,
    const IrisResult& iris_result,
    const LensConfig& config)
{
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_) {
        return ErrorCode::NotInitialized;
    }
    if (!output_texture) {
        return ErrorCode::NullPointer;
    }
    if (lens_texture_ == 0) {
        return ErrorCode::NoTextureLoaded;
    }
    if (!iris_result.detected) {
        // 검출 실패 시 입력을 그대로 반환
        *output_texture = input_texture;
        return ErrorCode::Success;
    }

#if IRIS_SDK_GPU_AVAILABLE
    if (render_context_) {
        render_context_->makeCurrent();
    }

    // 캐시 업데이트
    updateEyelidCache(iris_result);
    updateEllipseCache(iris_result);

    // 출력 텍스처 획득
    auto* output_info = texture_pool_->acquireRenderTarget(width, height);
    if (!output_info) {
        LOGE("Failed to acquire render target %dx%d", width, height);
        return IRIS_SDK_RENDER_FAILED;
    }

    // 이전 출력 텍스처 반환
    if (previous_output_texture_ != 0) {
        texture_pool_->releaseTextureById(previous_output_texture_);
    }
    previous_output_texture_ = output_info->texture_id;

    // FBO 바인딩
    glBindFramebuffer(GL_FRAMEBUFFER, output_info->fbo_id);
    glViewport(0, 0, width, height);

    // 셰이더 프로그램 활성화
    glUseProgram(lens_program_);

    // 텍스처 바인딩: unit 0 = 카메라, unit 1 = 렌즈
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(input_texture));
    glUniform1i(lens_uniforms_.uCameraTexture, 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, lens_texture_);
    glUniform1i(lens_uniforms_.uLensTexture, 1);

    // 홍채 파라미터 설정
    if (iris_result.left_detected && config.apply_left) {
        glUniform2f(lens_uniforms_.uLeftIrisCenter,
                    iris_result.left_iris[0].x,
                    iris_result.left_iris[0].y);
        glUniform1f(lens_uniforms_.uLeftIrisRadius, iris_result.left_radius);
        glUniform1i(lens_uniforms_.uApplyLeft, 1);
    } else {
        glUniform1i(lens_uniforms_.uApplyLeft, 0);
    }

    if (iris_result.right_detected && config.apply_right) {
        glUniform2f(lens_uniforms_.uRightIrisCenter,
                    iris_result.right_iris[0].x,
                    iris_result.right_iris[0].y);
        glUniform1f(lens_uniforms_.uRightIrisRadius, iris_result.right_radius);
        glUniform1i(lens_uniforms_.uApplyRight, 1);
    } else {
        glUniform1i(lens_uniforms_.uApplyRight, 0);
    }

    // 렌즈 설정
    glUniform1f(lens_uniforms_.uOpacity, config.opacity);
    glUniform1f(lens_uniforms_.uLensScale, config.scale);
    glUniform1f(lens_uniforms_.uEdgeFeather, config.edge_feather);
    glUniform1i(lens_uniforms_.uBlendMode, static_cast<int>(config.blend_mode));
    glUniform1f(lens_uniforms_.uFrameAspect,
                static_cast<float>(width) / static_cast<float>(height));

    // 눈꺼풀 클리핑 (캐시 사용)
    if (eyelid_cache_[0].valid_frames > 0) {
        glUniform1f(lens_uniforms_.uLeftEyeTop, eyelid_cache_[0].top);
        glUniform1f(lens_uniforms_.uLeftEyeBottom, eyelid_cache_[0].bottom);
    } else {
        glUniform1f(lens_uniforms_.uLeftEyeTop, 0.0f);
        glUniform1f(lens_uniforms_.uLeftEyeBottom, 1.0f);
    }
    if (eyelid_cache_[1].valid_frames > 0) {
        glUniform1f(lens_uniforms_.uRightEyeTop, eyelid_cache_[1].top);
        glUniform1f(lens_uniforms_.uRightEyeBottom, eyelid_cache_[1].bottom);
    } else {
        glUniform1f(lens_uniforms_.uRightEyeTop, 0.0f);
        glUniform1f(lens_uniforms_.uRightEyeBottom, 1.0f);
    }
    glUniform1f(lens_uniforms_.uEyelidFeather, 0.008f);

    // 기능 플래그
    glUniform1i(lens_uniforms_.uScleraProtect, sclera_protect_ ? 1 : 0);
    glUniform1i(lens_uniforms_.uContactShadow, contact_shadow_ ? 1 : 0);
    glUniform1f(lens_uniforms_.uShadowIntensity, shadow_intensity_);
    glUniform1i(lens_uniforms_.uMaxDetail, 1);

    // 비대칭 타원 마스크
    glUniform1i(lens_uniforms_.uUseEllipseMask, use_ellipse_mask_ ? 1 : 0);

    if (use_ellipse_mask_) {
        // 왼쪽 눈 타원
        if (ellipse_cache_[0].valid_frames > 0) {
            glUniform2f(lens_uniforms_.uLeftEyeEllipseCenter,
                        ellipse_cache_[0].cx, ellipse_cache_[0].cy);
            glUniform3f(lens_uniforms_.uLeftEyeEllipseRadii,
                        ellipse_cache_[0].rx_inner,
                        ellipse_cache_[0].rx_outer,
                        ellipse_cache_[0].ry);
            glUniform1f(lens_uniforms_.uLeftEyeEllipseRot, ellipse_cache_[0].rotation);
        }
        // 오른쪽 눈 타원
        if (ellipse_cache_[1].valid_frames > 0) {
            glUniform2f(lens_uniforms_.uRightEyeEllipseCenter,
                        ellipse_cache_[1].cx, ellipse_cache_[1].cy);
            glUniform3f(lens_uniforms_.uRightEyeEllipseRadii,
                        ellipse_cache_[1].rx_inner,
                        ellipse_cache_[1].rx_outer,
                        ellipse_cache_[1].ry);
            glUniform1f(lens_uniforms_.uRightEyeEllipseRot, ellipse_cache_[1].rotation);
        }
    }

    // 평균 홍채 휘도 (Sclera Protection용 — 근사값)
    glUniform1f(lens_uniforms_.uAvgIrisLum, 0.3f);

    // 검출 높이 (정규화)
    float det_h = static_cast<float>(iris_result.frame_height) /
                  static_cast<float>(std::max(iris_result.frame_height, 1));
    glUniform1f(lens_uniforms_.uDetH, det_h);

    // 풀스크린 쿼드 렌더링
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    renderFullscreenQuad();

    // 언바인딩
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glUseProgram(0);
    glActiveTexture(GL_TEXTURE0);

    *output_texture = output_info->texture_id;
    return ErrorCode::Success;

#else
    // Desktop stub: 입력을 그대로 반환
    *output_texture = input_texture;
    LOGD("renderToTexture: stub mode, passthrough");
    return ErrorCode::Success;
#endif
}

} // namespace iris_sdk
