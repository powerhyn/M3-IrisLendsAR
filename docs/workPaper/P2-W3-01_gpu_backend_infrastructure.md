# P2-W3-01. GPU 백엔드 인프라 (OpenGL ES)

## 작업 정보

| 항목 | 내용 |
|------|------|
| **작업 ID** | P2-W3-01 |
| **Phase** | Phase 3: GPU 백엔드 구현 |
| **상태** | ⏳ 대기 |
| **예상 기간** | 3일 |
| **의존성** | P2-W1-01 (RenderContext), P2-W1-04 (BeautyProcessor) |
| **담당** | systems-programming:cpp-pro |

---

## 1. 목표

Android OpenGL ES 3.1 기반 GPU 뷰티 필터 백엔드 인프라 구축

### 핵심 산출물
- `GPUBeautyBackend` 클래스
- 셰이더 컴파일/관리 시스템
- 프레임버퍼 및 텍스처 관리
- LensRenderer와 GL 컨텍스트 공유

---

## 2. 아키텍처

### 2.1 GPU 파이프라인 개요

```
Input (TextureHandle from Camera/LensRenderer)
    │
    ▼
┌─────────────────────────────────────────┐
│         GPUBeautyBackend                │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │    ShaderManager                │    │
│  │    - Compile/Cache Shaders      │    │
│  │    - Uniform Management         │    │
│  └─────────────────────────────────┘    │
│                │                        │
│  ┌─────────────▼─────────────────────┐  │
│  │    FilterChain                    │  │
│  │    - Smoothing Pass               │  │
│  │    - Whitening Pass               │  │
│  │    - Color Balance Pass           │  │
│  │    - Soft Focus Pass              │  │
│  └─────────────────────────────────┬─┘  │
│                                    │    │
│  ┌─────────────────────────────────▼─┐  │
│  │    TexturePool                    │  │
│  │    - Intermediate FBOs            │  │
│  │    - Ping-Pong Buffers            │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
    │
    ▼
Output (TextureHandle to LensRenderer/Display)
```

### 2.2 LensRenderer 연동

```
┌──────────────────────────────────────────────────────┐
│                  Shared IRenderContext               │
│                  (GLESRenderContext)                 │
└───────────────────┬──────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌───────────────┐       ┌───────────────────┐
│ LensRenderer  │       │ GPUBeautyBackend  │
│ (렌즈 오버레이)│◀─────▶│ (뷰티 필터)       │
└───────┬───────┘       └─────────┬─────────┘
        │                         │
        │   TextureHandle 공유    │
        └─────────────────────────┘
```

---

## 3. 상세 구현

### 3.1 GPUBeautyBackend 클래스

**파일**: `cpp/include/iris_sdk/gpu/gpu_beauty_backend.h`

```cpp
#ifndef IRIS_SDK_GPU_BEAUTY_BACKEND_H
#define IRIS_SDK_GPU_BEAUTY_BACKEND_H

#if defined(__ANDROID__) && defined(IRIS_SDK_HAS_GLES)

#include "iris_sdk/beauty_backend.h"
#include "iris_sdk/gpu/shader_manager.h"
#include "iris_sdk/gpu/texture_pool.h"
#include "iris_sdk/gpu/gles_render_context.h"
#include <GLES3/gl31.h>

namespace iris_sdk {

/**
 * @brief GPU 기반 뷰티 필터 백엔드 (OpenGL ES 3.1)
 *
 * 셰이더 기반 실시간 이미지 처리
 */
class GPUBeautyBackend : public IBeautyBackend {
public:
    GPUBeautyBackend();
    ~GPUBeautyBackend() override;

    // 복사/이동 금지
    GPUBeautyBackend(const GPUBeautyBackend&) = delete;
    GPUBeautyBackend& operator=(const GPUBeautyBackend&) = delete;

    //=== IBeautyBackend 구현 ===
    bool initialize(IRenderContext* render_context) override;
    void release() override;
    bool isInitialized() const override;

    // CPU 버퍼 처리 (텍스처 업로드 필요)
    IrisSdkError apply(
        uint8_t* frame_data,
        int width, int height,
        IrisFrameFormat format,
        const BeautyFilterConfigV2& config,
        const BeautyROI* roi = nullptr
    ) override;

    // 텍스처 직접 처리 (Zero-Copy)
    IrisSdkError applyTexture(
        const TextureHandle& input,
        TextureHandle& output,
        const BeautyFilterConfigV2& config,
        const BeautyROI* roi = nullptr
    ) override;

    const char* getName() const override { return "GPUBeautyBackend"; }
    bool supportsGpu() const override { return true; }
    bool supportsTextureProcessing() const override { return true; }

    //=== GPU 전용 ===
    /**
     * @brief 마스크 텍스처 업로드
     *
     * Face Mesh 기반 마스크를 GPU 텍스처로 전송
     */
    bool uploadMaskTexture(const cv::Mat& mask, GLuint& out_texture);

private:
    // 셰이더 초기화
    bool initializeShaders();

    // 필터 패스 실행
    void executeSmoothingPass(GLuint input_tex, GLuint output_fbo,
                              int width, int height,
                              const BeautyFilterConfigV2& config);

    void executeWhiteningPass(GLuint input_tex, GLuint output_fbo,
                              int width, int height,
                              float whitening);

    void executeColorBalancePass(GLuint input_tex, GLuint output_fbo,
                                 int width, int height,
                                 float balance);

    void executeSoftFocusPass(GLuint input_tex, GLuint output_fbo,
                              int width, int height,
                              float strength);

    void executeBrightnessPass(GLuint input_tex, GLuint output_fbo,
                               int width, int height,
                               float brightness);

    // 마스킹 적용 (ROI, 눈/입술 보호)
    void applyMasking(GLuint filtered_tex, GLuint original_tex,
                      GLuint mask_tex, GLuint output_fbo,
                      int width, int height);

    // 풀스크린 쿼드 렌더링
    void renderFullscreenQuad();
    void setupFullscreenQuad();

    GLESRenderContext* render_context_ = nullptr;  // 외부 소유
    std::unique_ptr<ShaderManager> shader_manager_;
    std::unique_ptr<TexturePool> texture_pool_;

    // 풀스크린 쿼드 VAO/VBO
    GLuint quad_vao_ = 0;
    GLuint quad_vbo_ = 0;

    // 셰이더 프로그램 ID
    GLuint smoothing_program_ = 0;
    GLuint whitening_program_ = 0;
    GLuint color_balance_program_ = 0;
    GLuint soft_focus_program_ = 0;
    GLuint brightness_program_ = 0;
    GLuint masking_program_ = 0;

    bool initialized_ = false;
    std::mutex mutex_;
};

} // namespace iris_sdk

#endif // __ANDROID__ && IRIS_SDK_HAS_GLES

#endif // IRIS_SDK_GPU_BEAUTY_BACKEND_H
```

### 3.2 ShaderManager 클래스

**파일**: `cpp/include/iris_sdk/gpu/shader_manager.h`

```cpp
#ifndef IRIS_SDK_SHADER_MANAGER_H
#define IRIS_SDK_SHADER_MANAGER_H

#if defined(__ANDROID__) && defined(IRIS_SDK_HAS_GLES)

#include <GLES3/gl31.h>
#include <string>
#include <unordered_map>
#include <mutex>

namespace iris_sdk {

/**
 * @brief 셰이더 컴파일 및 캐싱 관리
 */
class ShaderManager {
public:
    ShaderManager();
    ~ShaderManager();

    /**
     * @brief 셰이더 프로그램 생성
     *
     * @param vertex_source 버텍스 셰이더 소스
     * @param fragment_source 프래그먼트 셰이더 소스
     * @param out_program 출력 프로그램 ID
     * @return 성공 여부
     */
    bool createProgram(
        const char* vertex_source,
        const char* fragment_source,
        GLuint& out_program
    );

    /**
     * @brief 캐시된 프로그램 조회
     */
    GLuint getProgram(const std::string& name) const;

    /**
     * @brief 프로그램 캐시 등록
     */
    void cacheProgram(const std::string& name, GLuint program);

    /**
     * @brief 모든 셰이더 해제
     */
    void releaseAll();

    //=== 유틸리티 ===
    static bool compileShader(GLenum type, const char* source, GLuint& out_shader);
    static bool linkProgram(GLuint vertex, GLuint fragment, GLuint& out_program);
    static std::string getShaderLog(GLuint shader);
    static std::string getProgramLog(GLuint program);

private:
    std::unordered_map<std::string, GLuint> program_cache_;
    mutable std::mutex mutex_;
};

//=== 내장 셰이더 소스 ===
namespace shaders {

// 공통 버텍스 셰이더
extern const char* FULLSCREEN_QUAD_VERTEX;

// 필터 프래그먼트 셰이더
extern const char* BILATERAL_FILTER_FRAGMENT;
extern const char* WHITENING_FRAGMENT;
extern const char* COLOR_BALANCE_FRAGMENT;
extern const char* SOFT_FOCUS_FRAGMENT;
extern const char* BRIGHTNESS_FRAGMENT;
extern const char* MASKING_FRAGMENT;
extern const char* GAUSSIAN_BLUR_FRAGMENT;

} // namespace shaders

} // namespace iris_sdk

#endif // __ANDROID__ && IRIS_SDK_HAS_GLES

#endif // IRIS_SDK_SHADER_MANAGER_H
```

### 3.3 TexturePool 클래스

**파일**: `cpp/include/iris_sdk/gpu/texture_pool.h`

```cpp
#ifndef IRIS_SDK_TEXTURE_POOL_H
#define IRIS_SDK_TEXTURE_POOL_H

#if defined(__ANDROID__) && defined(IRIS_SDK_HAS_GLES)

#include <GLES3/gl31.h>
#include <vector>
#include <queue>
#include <mutex>

namespace iris_sdk {

/**
 * @brief GPU 텍스처 및 프레임버퍼 풀
 *
 * 재사용을 통한 메모리 효율화 및 할당 오버헤드 감소
 */
class TexturePool {
public:
    struct TextureInfo {
        GLuint texture_id = 0;
        GLuint fbo_id = 0;      // 연결된 FBO (렌더 타겟용)
        int width = 0;
        int height = 0;
        GLenum format = GL_RGBA;
        bool in_use = false;
    };

    TexturePool();
    ~TexturePool();

    /**
     * @brief 풀 초기화
     *
     * @param max_textures 최대 텍스처 수
     * @param max_width 최대 너비
     * @param max_height 최대 높이
     */
    bool initialize(int max_textures, int max_width, int max_height);

    void release();

    /**
     * @brief 텍스처 획득 (렌더 타겟용)
     *
     * @param width 요청 너비
     * @param height 요청 높이
     * @return 텍스처 정보 (nullptr if failed)
     */
    TextureInfo* acquireRenderTarget(int width, int height);

    /**
     * @brief 텍스처 반환
     */
    void releaseTexture(TextureInfo* info);

    /**
     * @brief Ping-Pong 버퍼 획득 (필터 체이닝용)
     */
    bool acquirePingPongPair(int width, int height,
                             TextureInfo*& ping, TextureInfo*& pong);

    /**
     * @brief 모든 텍스처 반환
     */
    void releaseAll();

    //=== 메모리 관리 (P2-W5-02 최적화 단계에서 중요) ===

    /**
     * @brief 미사용 텍스처 정리
     *
     * 일정 시간 이상 사용되지 않은 텍스처를 해제하여 메모리 절약
     * @param max_idle_ms 최대 유휴 시간 (ms), 기본 5000ms
     * @return 해제된 텍스처 수
     */
    int trim(int64_t max_idle_ms = 5000);

    /**
     * @brief 풀 크기 동적 조정
     *
     * 저사양 기기에서 메모리 부족(OOM) 방지를 위해
     * 런타임에 최대 텍스처 수를 조정
     *
     * @param new_max_textures 새로운 최대 텍스처 수 (1-16)
     */
    void resizePool(int new_max_textures);

    /**
     * @brief 메모리 압력 콜백 설정
     *
     * Android onTrimMemory() 이벤트와 연동하여 자동 정리
     */
    using MemoryPressureCallback = std::function<void(int level)>;
    void setMemoryPressureCallback(MemoryPressureCallback callback);

    /**
     * @brief 현재 메모리 사용량 조회
     *
     * @return 사용 중인 텍스처 메모리 (바이트)
     */
    size_t getUsedMemory() const;

    /**
     * @brief 풀 상태 조회
     */
    struct PoolStats {
        int total_textures;
        int in_use;
        int available;
        size_t total_memory_bytes;
    };
    PoolStats getStats() const;

private:
    TextureInfo* createTexture(int width, int height);
    TextureInfo* findAvailable(int width, int height);

    std::vector<std::unique_ptr<TextureInfo>> textures_;
    int max_textures_ = 0;
    int max_width_ = 0;
    int max_height_ = 0;
    std::mutex mutex_;
    bool initialized_ = false;

    // 메모리 관리 관련
    std::unordered_map<GLuint, int64_t> last_used_time_;  // 텍스처 ID → 마지막 사용 시각
    MemoryPressureCallback memory_pressure_callback_;
};

} // namespace iris_sdk

#endif // __ANDROID__ && IRIS_SDK_HAS_GLES

#endif // IRIS_SDK_TEXTURE_POOL_H
```

### 3.4 구현 - GPUBeautyBackend

**파일**: `cpp/src/gpu/gpu_beauty_backend.cpp`

```cpp
#if defined(__ANDROID__) && defined(IRIS_SDK_HAS_GLES)

#include "iris_sdk/gpu/gpu_beauty_backend.h"
#include "iris_sdk/gpu/shader_sources.h"
#include <android/log.h>

#define LOG_TAG "GPUBeautyBackend"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace iris_sdk {

GPUBeautyBackend::GPUBeautyBackend() = default;

GPUBeautyBackend::~GPUBeautyBackend() {
    release();
}

bool GPUBeautyBackend::initialize(IRenderContext* render_context) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (initialized_) return true;

    // RenderContext 캐스팅 (GLES 전용)
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

    // 셰이더 매니저 초기화
    shader_manager_ = std::make_unique<ShaderManager>();
    if (!initializeShaders()) {
        LOGE("Failed to initialize shaders");
        return false;
    }

    // 텍스처 풀 초기화 (최대 8개, 1920x1080)
    texture_pool_ = std::make_unique<TexturePool>();
    if (!texture_pool_->initialize(8, 1920, 1080)) {
        LOGE("Failed to initialize texture pool");
        return false;
    }

    // 풀스크린 쿼드 설정
    setupFullscreenQuad();

    initialized_ = true;
    LOGI("GPUBeautyBackend initialized successfully");
    return true;
}

bool GPUBeautyBackend::initializeShaders() {
    // 스무딩 (Bilateral/Guided Filter)
    if (!shader_manager_->createProgram(
            shaders::FULLSCREEN_QUAD_VERTEX,
            shaders::BILATERAL_FILTER_FRAGMENT,
            smoothing_program_)) {
        return false;
    }
    shader_manager_->cacheProgram("smoothing", smoothing_program_);

    // 화이트닝
    if (!shader_manager_->createProgram(
            shaders::FULLSCREEN_QUAD_VERTEX,
            shaders::WHITENING_FRAGMENT,
            whitening_program_)) {
        return false;
    }
    shader_manager_->cacheProgram("whitening", whitening_program_);

    // 컬러 밸런스
    if (!shader_manager_->createProgram(
            shaders::FULLSCREEN_QUAD_VERTEX,
            shaders::COLOR_BALANCE_FRAGMENT,
            color_balance_program_)) {
        return false;
    }
    shader_manager_->cacheProgram("color_balance", color_balance_program_);

    // 소프트 포커스
    if (!shader_manager_->createProgram(
            shaders::FULLSCREEN_QUAD_VERTEX,
            shaders::SOFT_FOCUS_FRAGMENT,
            soft_focus_program_)) {
        return false;
    }
    shader_manager_->cacheProgram("soft_focus", soft_focus_program_);

    // 밝기
    if (!shader_manager_->createProgram(
            shaders::FULLSCREEN_QUAD_VERTEX,
            shaders::BRIGHTNESS_FRAGMENT,
            brightness_program_)) {
        return false;
    }
    shader_manager_->cacheProgram("brightness", brightness_program_);

    // 마스킹 (ROI 블렌딩)
    if (!shader_manager_->createProgram(
            shaders::FULLSCREEN_QUAD_VERTEX,
            shaders::MASKING_FRAGMENT,
            masking_program_)) {
        return false;
    }
    shader_manager_->cacheProgram("masking", masking_program_);

    return true;
}

void GPUBeautyBackend::setupFullscreenQuad() {
    // 풀스크린 쿼드 정점 데이터
    // position (x, y), texcoord (u, v)
    float quad_vertices[] = {
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

    // position attribute
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // texcoord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                          (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

void GPUBeautyBackend::renderFullscreenQuad() {
    glBindVertexArray(quad_vao_);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
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
        return IRIS_SDK_ERROR_INVALID_PARAM;
    }

    if (!config.enabled) {
        // 비활성화: 입력을 그대로 출력으로
        output = input;
        return IRIS_SDK_OK;
    }

    render_context_->makeCurrent();

    int width = input.width;
    int height = input.height;
    GLuint input_tex = *static_cast<GLuint*>(input.native_handle);

    // Ping-Pong 버퍼 획득
    TexturePool::TextureInfo* ping = nullptr;
    TexturePool::TextureInfo* pong = nullptr;
    if (!texture_pool_->acquirePingPongPair(width, height, ping, pong)) {
        return IRIS_SDK_ERROR_INTERNAL;
    }

    GLuint current_input = input_tex;
    TexturePool::TextureInfo* current_output = ping;

    // 필터 체인 실행
    // 1. 스무딩
    if (config.smoothing > 0.01f) {
        executeSmoothingPass(current_input, current_output->fbo_id,
                             width, height, config);
        current_input = current_output->texture_id;
        std::swap(current_output, (current_output == ping) ? pong : ping);
    }

    // 2. 화이트닝
    if (config.whitening > 0.01f) {
        executeWhiteningPass(current_input, current_output->fbo_id,
                             width, height, config.whitening);
        current_input = current_output->texture_id;
        std::swap(current_output, (current_output == ping) ? pong : ping);
    }

    // 3. 컬러 밸런스
    if (std::abs(config.colorBalance) > 0.01f) {
        executeColorBalancePass(current_input, current_output->fbo_id,
                                width, height, config.colorBalance);
        current_input = current_output->texture_id;
        std::swap(current_output, (current_output == ping) ? pong : ping);
    }

    // 4. 소프트 포커스
    if (config.softFocus > 0.01f) {
        executeSoftFocusPass(current_input, current_output->fbo_id,
                             width, height, config.softFocus);
        current_input = current_output->texture_id;
        std::swap(current_output, (current_output == ping) ? pong : ping);
    }

    // 5. 밝기
    if (std::abs(config.brightness - 1.0f) > 0.01f) {
        executeBrightnessPass(current_input, current_output->fbo_id,
                              width, height, config.brightness);
        current_input = current_output->texture_id;
    }

    // 출력 텍스처 핸들 설정
    output.native_handle = new GLuint(current_input);
    output.type = TextureHandle::Type::OpenGLES;
    output.width = width;
    output.height = height;

    // 텍스처 반환 (current_input은 output으로 사용 중이므로 제외)
    // 주의: 호출자가 output 사용 후 반환해야 함

    return IRIS_SDK_OK;
}

void GPUBeautyBackend::release() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_) return;

    if (render_context_) {
        render_context_->makeCurrent();
    }

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

    // 쿼드 VAO/VBO 해제
    if (quad_vao_) {
        glDeleteVertexArrays(1, &quad_vao_);
        quad_vao_ = 0;
    }
    if (quad_vbo_) {
        glDeleteBuffers(1, &quad_vbo_);
        quad_vbo_ = 0;
    }

    render_context_ = nullptr;
    initialized_ = false;

    LOGI("GPUBeautyBackend released");
}

} // namespace iris_sdk

#endif // __ANDROID__ && IRIS_SDK_HAS_GLES
```

### 3.5 TexturePool 메모리 관리 구현

**파일**: `cpp/src/gpu/texture_pool.cpp` (메모리 관리 부분)

```cpp
int TexturePool::trim(int64_t max_idle_ms) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    int released_count = 0;

    auto it = textures_.begin();
    while (it != textures_.end()) {
        TextureInfo* info = it->get();

        // 사용 중인 텍스처는 건너뜀
        if (info->in_use) {
            ++it;
            continue;
        }

        // 유휴 시간 확인
        auto last_used_it = last_used_time_.find(info->texture_id);
        if (last_used_it != last_used_time_.end()) {
            int64_t idle_time = now - last_used_it->second;

            if (idle_time > max_idle_ms) {
                // GL 리소스 해제
                if (info->fbo_id) glDeleteFramebuffers(1, &info->fbo_id);
                if (info->texture_id) glDeleteTextures(1, &info->texture_id);

                last_used_time_.erase(last_used_it);
                it = textures_.erase(it);
                released_count++;
                continue;
            }
        }
        ++it;
    }

    LOGI("TexturePool::trim() released %d textures", released_count);
    return released_count;
}

void TexturePool::resizePool(int new_max_textures) {
    std::lock_guard<std::mutex> lock(mutex_);

    new_max_textures = std::clamp(new_max_textures, 1, 16);

    // 현재 텍스처 수가 새 최대값보다 크면 미사용 텍스처 해제
    while (static_cast<int>(textures_.size()) > new_max_textures) {
        // 미사용 텍스처 찾기
        auto it = std::find_if(textures_.begin(), textures_.end(),
            [](const auto& info) { return !info->in_use; });

        if (it == textures_.end()) break;  // 모든 텍스처가 사용 중

        TextureInfo* info = it->get();
        if (info->fbo_id) glDeleteFramebuffers(1, &info->fbo_id);
        if (info->texture_id) glDeleteTextures(1, &info->texture_id);
        textures_.erase(it);
    }

    max_textures_ = new_max_textures;
    LOGI("TexturePool resized to %d textures", max_textures_);
}

size_t TexturePool::getUsedMemory() const {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t total = 0;
    for (const auto& info : textures_) {
        // RGBA = 4 bytes per pixel
        total += info->width * info->height * 4;
    }
    return total;
}

TexturePool::PoolStats TexturePool::getStats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    PoolStats stats = {};
    stats.total_textures = static_cast<int>(textures_.size());

    for (const auto& info : textures_) {
        if (info->in_use) {
            stats.in_use++;
        } else {
            stats.available++;
        }
        stats.total_memory_bytes += info->width * info->height * 4;
    }

    return stats;
}
```

### 3.6 Android 메모리 압력 연동

**파일**: JNI에서 Android onTrimMemory 이벤트 연동

```cpp
// JNI 콜백 (Android Activity/Fragment에서 호출)
extern "C" JNIEXPORT void JNICALL
Java_com_example_irissdk_IrisSDK_onTrimMemory(
    JNIEnv* env, jobject thiz, jint level) {

    auto* backend = SDKManager::getInstance().getGPUBeautyBackend();
    if (!backend) return;

    auto* texture_pool = backend->getTexturePool();
    if (!texture_pool) return;

    // Android ComponentCallbacks2 상수
    // TRIM_MEMORY_RUNNING_LOW = 10
    // TRIM_MEMORY_RUNNING_CRITICAL = 15
    // TRIM_MEMORY_UI_HIDDEN = 20
    // TRIM_MEMORY_BACKGROUND = 40
    // TRIM_MEMORY_MODERATE = 60
    // TRIM_MEMORY_COMPLETE = 80

    if (level >= 60) {
        // MODERATE 이상: 공격적 정리
        texture_pool->trim(0);  // 모든 미사용 텍스처 즉시 해제
        texture_pool->resizePool(4);  // 최소 풀 크기
    } else if (level >= 40) {
        // BACKGROUND: 적당한 정리
        texture_pool->trim(1000);  // 1초 이상 유휴 텍스처 해제
    } else if (level >= 10) {
        // RUNNING_LOW: 가벼운 정리
        texture_pool->trim(5000);  // 5초 이상 유휴 텍스처 해제
    }
}
```

---

## 4. CMake 설정

**파일**: `cpp/CMakeLists.txt` (확장)

```cmake
# GPU 모듈 (Android 전용)
if(ANDROID AND IRIS_SDK_HAS_GLES)
    set(GPU_BEAUTY_SOURCES
        src/gpu/gpu_beauty_backend.cpp
        src/gpu/shader_manager.cpp
        src/gpu/texture_pool.cpp
        src/gpu/shader_sources.cpp
    )

    set(GPU_BEAUTY_HEADERS
        include/iris_sdk/gpu/gpu_beauty_backend.h
        include/iris_sdk/gpu/shader_manager.h
        include/iris_sdk/gpu/texture_pool.h
    )

    # 셰이더 소스 파일 빌드 타임 포함
    set(SHADER_FILES
        src/gpu/shaders/fullscreen_quad.vert
        src/gpu/shaders/bilateral_filter.frag
        src/gpu/shaders/whitening.frag
        src/gpu/shaders/color_balance.frag
        src/gpu/shaders/soft_focus.frag
        src/gpu/shaders/brightness.frag
        src/gpu/shaders/masking.frag
    )

    # 셰이더 → C++ 문자열 변환 (빌드 타임)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shader_sources.cpp
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ConvertShaders.cmake
        DEPENDS ${SHADER_FILES}
        COMMENT "Converting shader sources to C++ strings"
    )

    list(APPEND GPU_BEAUTY_SOURCES ${CMAKE_CURRENT_BINARY_DIR}/shader_sources.cpp)
endif()
```

---

## 5. 단위 테스트

**파일**: `cpp/tests/test_gpu_beauty_backend.cpp`

```cpp
#if defined(__ANDROID__) && defined(IRIS_SDK_HAS_GLES)

#include <gtest/gtest.h>
#include "iris_sdk/gpu/gpu_beauty_backend.h"
#include "iris_sdk/gpu/gles_render_context.h"

class GPUBeautyBackendTest : public ::testing::Test {
protected:
    void SetUp() override {
        render_context_ = std::make_unique<GLESRenderContext>();
        ASSERT_TRUE(render_context_->initialize());

        backend_ = std::make_unique<GPUBeautyBackend>();
    }

    void TearDown() override {
        if (backend_) backend_->release();
        if (render_context_) render_context_->release();
    }

    std::unique_ptr<GLESRenderContext> render_context_;
    std::unique_ptr<GPUBeautyBackend> backend_;
};

TEST_F(GPUBeautyBackendTest, InitializesSuccessfully) {
    ASSERT_TRUE(backend_->initialize(render_context_.get()));
    EXPECT_TRUE(backend_->isInitialized());
    EXPECT_TRUE(backend_->supportsGpu());
    EXPECT_TRUE(backend_->supportsTextureProcessing());
}

TEST_F(GPUBeautyBackendTest, ProcessesTextureWithoutROI) {
    ASSERT_TRUE(backend_->initialize(render_context_.get()));

    // 입력 텍스처 생성
    TextureHandle input = render_context_->createTexture(640, 480, GL_RGBA);
    ASSERT_TRUE(input.isValid());

    // 테스트 데이터 업로드
    std::vector<uint8_t> test_data(640 * 480 * 4, 128);
    render_context_->uploadTexture(input, test_data.data(), 640, 480, GL_RGBA);

    // 필터 적용
    BeautyFilterConfigV2 config;
    config.enabled = true;
    config.smoothing = 0.5f;
    config.whitening = 0.3f;

    TextureHandle output;
    IrisSdkError err = backend_->applyTexture(input, output, config, nullptr);

    EXPECT_EQ(err, IRIS_SDK_OK);
    EXPECT_TRUE(output.isValid());
    EXPECT_EQ(output.width, 640);
    EXPECT_EQ(output.height, 480);

    // 정리
    render_context_->deleteTexture(input);
}

TEST_F(GPUBeautyBackendTest, DisabledConfigPassesThrough) {
    ASSERT_TRUE(backend_->initialize(render_context_.get()));

    TextureHandle input = render_context_->createTexture(640, 480, GL_RGBA);

    BeautyFilterConfigV2 config;
    config.enabled = false;

    TextureHandle output;
    IrisSdkError err = backend_->applyTexture(input, output, config, nullptr);

    EXPECT_EQ(err, IRIS_SDK_OK);
    // 비활성화 시 입력 = 출력
    EXPECT_EQ(output.native_handle, input.native_handle);

    render_context_->deleteTexture(input);
}

TEST_F(GPUBeautyBackendTest, PerformanceBenchmark) {
    ASSERT_TRUE(backend_->initialize(render_context_.get()));

    TextureHandle input = render_context_->createTexture(1920, 1080, GL_RGBA);
    std::vector<uint8_t> test_data(1920 * 1080 * 4, 128);
    render_context_->uploadTexture(input, test_data.data(), 1920, 1080, GL_RGBA);

    BeautyFilterConfigV2 config;
    config.enabled = true;
    config.smoothing = 0.8f;
    config.whitening = 0.5f;
    config.colorBalance = 0.2f;
    config.softFocus = 0.3f;
    config.brightness = 1.1f;

    const int iterations = 100;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        TextureHandle output;
        backend_->applyTexture(input, output, config, nullptr);
        glFinish();  // GPU 완료 대기
    }

    auto end = std::chrono::high_resolution_clock::now();
    double avg_ms = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start).count() / (iterations * 1000.0);

    std::cout << "Average GPU processing time (1080p): " << avg_ms << " ms" << std::endl;

    // 10ms 이하 목표
    EXPECT_LT(avg_ms, 10.0);

    render_context_->deleteTexture(input);
}

#endif // __ANDROID__ && IRIS_SDK_HAS_GLES
```

---

## 6. 완료 기준

- [ ] `GPUBeautyBackend` 클래스 구현
- [ ] `ShaderManager` 구현
- [ ] `TexturePool` 구현
- [ ] LensRenderer와 RenderContext 공유
- [ ] Ping-Pong 버퍼링
- [ ] 기본 필터 패스 프레임워크
- [ ] TexturePool 메모리 관리 기능
  - [ ] `trim()` 미사용 텍스처 정리
  - [ ] `resizePool()` 동적 풀 크기 조정
  - [ ] `getStats()` 풀 상태 모니터링
- [ ] Android onTrimMemory 연동
- [ ] 단위 테스트 통과
- [ ] 1080p 기준 10ms 이하 목표

---

## 7. 다음 작업

- **P2-W3-02**: 기본 필터 셰이더 구현
