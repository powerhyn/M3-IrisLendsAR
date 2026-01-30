/**
 * @file gles_render_context.cpp
 * @brief Android OpenGL ES 렌더링 컨텍스트 구현
 */

#if defined(__ANDROID__) || defined(IRIS_SDK_FORCE_GLES)

#include "iris_sdk/gpu/gles_render_context.h"

#include <android/log.h>
#include <cstring>

// EGL_OPENGL_ES3_BIT_KHR may not be defined in older NDK headers
#ifndef EGL_OPENGL_ES3_BIT_KHR
#define EGL_OPENGL_ES3_BIT_KHR 0x0040
#endif

#define LOG_TAG "GLESRenderContext"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)

namespace iris_sdk {

GLESRenderContext::GLESRenderContext() = default;

GLESRenderContext::~GLESRenderContext() {
    release();
}

bool GLESRenderContext::initialize() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (initialized_.load()) {
        return true;
    }

    if (!initEGL()) {
        LOGE("Failed to initialize EGL");
        return false;
    }

    if (!checkGLESVersion()) {
        LOGE("OpenGL ES version check failed");
        releaseEGL();
        return false;
    }

    LOGI("GLESRenderContext initialized: OpenGL ES %d.%d", major_version_, minor_version_);

    initialized_.store(true);
    context_lost_.store(false);
    return true;
}

void GLESRenderContext::release() {
    std::lock_guard<std::mutex> lock(mutex_);

    // 모든 텍스처 삭제
    if (display_ != EGL_NO_DISPLAY && context_ != EGL_NO_CONTEXT) {
        eglMakeCurrent(display_, surface_, surface_, context_);

        for (auto& pair : textures_) {
            glDeleteTextures(1, &pair.second.gl_id);
        }
    }
    textures_.clear();
    next_texture_id_ = 1;

    releaseEGL();
    initialized_.store(false);
}

bool GLESRenderContext::isInitialized() const {
    return initialized_.load() && !context_lost_.load();
}

bool GLESRenderContext::initEGL() {
    // 1. EGL Display 획득
    display_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (display_ == EGL_NO_DISPLAY) {
        LOGE("eglGetDisplay failed");
        return false;
    }

    // 2. EGL 초기화
    EGLint major, minor;
    if (!eglInitialize(display_, &major, &minor)) {
        LOGE("eglInitialize failed");
        return false;
    }
    LOGI("EGL version: %d.%d", major, minor);

    // 3. Config 선택 (OpenGL ES 3.1 또는 3.0)
    EGLint config_attribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_DEPTH_SIZE, 0,
        EGL_STENCIL_SIZE, 0,
        EGL_NONE
    };

    EGLint num_configs;
    if (!eglChooseConfig(display_, config_attribs, &config_, 1, &num_configs) ||
        num_configs == 0) {
        LOGE("eglChooseConfig failed");
        return false;
    }

    // 4. PBuffer Surface 생성 (오프스크린 렌더링용)
    EGLint surface_attribs[] = {
        EGL_WIDTH, 1,
        EGL_HEIGHT, 1,
        EGL_NONE
    };
    surface_ = eglCreatePbufferSurface(display_, config_, surface_attribs);
    if (surface_ == EGL_NO_SURFACE) {
        LOGE("eglCreatePbufferSurface failed");
        return false;
    }

    // 5. OpenGL ES 3.1 Context 생성 (실패 시 3.0 폴백)
    EGLint context_attribs_31[] = {
        EGL_CONTEXT_MAJOR_VERSION, 3,
        EGL_CONTEXT_MINOR_VERSION, 1,
        EGL_NONE
    };

    context_ = eglCreateContext(display_, config_, EGL_NO_CONTEXT, context_attribs_31);
    if (context_ == EGL_NO_CONTEXT) {
        LOGW("OpenGL ES 3.1 context creation failed, trying 3.0");

        EGLint context_attribs_30[] = {
            EGL_CONTEXT_MAJOR_VERSION, 3,
            EGL_CONTEXT_MINOR_VERSION, 0,
            EGL_NONE
        };
        context_ = eglCreateContext(display_, config_, EGL_NO_CONTEXT, context_attribs_30);

        if (context_ == EGL_NO_CONTEXT) {
            LOGE("OpenGL ES 3.0 context creation also failed");
            eglDestroySurface(display_, surface_);
            surface_ = EGL_NO_SURFACE;
            return false;
        }
    }

    // 6. Context 활성화
    if (!eglMakeCurrent(display_, surface_, surface_, context_)) {
        LOGE("eglMakeCurrent failed");
        eglDestroyContext(display_, context_);
        eglDestroySurface(display_, surface_);
        context_ = EGL_NO_CONTEXT;
        surface_ = EGL_NO_SURFACE;
        return false;
    }

    return true;
}

void GLESRenderContext::releaseEGL() {
    if (display_ != EGL_NO_DISPLAY) {
        eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);

        if (context_ != EGL_NO_CONTEXT) {
            eglDestroyContext(display_, context_);
            context_ = EGL_NO_CONTEXT;
        }

        if (surface_ != EGL_NO_SURFACE) {
            eglDestroySurface(display_, surface_);
            surface_ = EGL_NO_SURFACE;
        }

        eglTerminate(display_);
        display_ = EGL_NO_DISPLAY;
    }
}

bool GLESRenderContext::checkGLESVersion() {
    const char* version = reinterpret_cast<const char*>(glGetString(GL_VERSION));
    if (version == nullptr) {
        return false;
    }

    LOGI("OpenGL ES version string: %s", version);

    // "OpenGL ES 3.1 ..." 형식에서 버전 파싱
    if (std::sscanf(version, "OpenGL ES %d.%d", &major_version_, &minor_version_) != 2) {
        // 다른 형식 시도
        major_version_ = 3;
        minor_version_ = 0;
    }

    // 최소 OpenGL ES 3.0 필요
    return (major_version_ > 3) || (major_version_ == 3 && minor_version_ >= 0);
}

TextureHandle GLESRenderContext::createTexture(int width, int height,
                                                TextureFormat format) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (context_lost_.load() || !initialized_.load()) {
        return TextureHandle{};
    }

    if (width <= 0 || height <= 0) {
        return TextureHandle{};
    }

    // GL 텍스처 생성
    GLuint gl_id = 0;
    glGenTextures(1, &gl_id);
    if (gl_id == 0 || !checkGLError("glGenTextures")) {
        return TextureHandle{};
    }

    glBindTexture(GL_TEXTURE_2D, gl_id);

    // 텍스처 파라미터 설정
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // 빈 텍스처 할당
    GLenum internal_format = toGLInternalFormat(format);
    GLenum gl_format = toGLFormat(format);
    GLenum gl_type = toGLType(format);

    glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0,
                 gl_format, gl_type, nullptr);

    if (!checkGLError("glTexImage2D")) {
        glDeleteTextures(1, &gl_id);
        return TextureHandle{};
    }

    glBindTexture(GL_TEXTURE_2D, 0);

    // 텍스처 ID 발급
    uint64_t id = next_texture_id_++;

    // 핸들 생성
    TextureHandle handle;
    handle.native_handle = reinterpret_cast<void*>(static_cast<uintptr_t>(gl_id));
    handle.type = TextureHandle::Type::OpenGLES;
    handle.width = width;
    handle.height = height;
    handle.format = format;
    handle.id = id;

    // 추적 저장
    textures_[id] = TextureInfo{gl_id, width, height, format};

    return handle;
}

void GLESRenderContext::deleteTexture(TextureHandle& handle) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!handle.isValid() || handle.type != TextureHandle::Type::OpenGLES) {
        return;
    }

    auto it = textures_.find(handle.id);
    if (it != textures_.end()) {
        if (!context_lost_.load()) {
            glDeleteTextures(1, &it->second.gl_id);
        }
        textures_.erase(it);
    }

    handle.invalidate();
}

bool GLESRenderContext::uploadTexture(TextureHandle& handle, const uint8_t* data,
                                        int width, int height, TextureFormat format) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (context_lost_.load() || !initialized_.load()) {
        return false;
    }

    if (!handle.isValid() || handle.type != TextureHandle::Type::OpenGLES) {
        return false;
    }

    if (data == nullptr || width <= 0 || height <= 0) {
        return false;
    }

    auto it = textures_.find(handle.id);
    if (it == textures_.end()) {
        return false;
    }

    GLuint gl_id = it->second.gl_id;
    glBindTexture(GL_TEXTURE_2D, gl_id);

    GLenum internal_format = toGLInternalFormat(format);
    GLenum gl_format = toGLFormat(format);
    GLenum gl_type = toGLType(format);

    // 크기가 다르면 재할당, 같으면 subImage 사용
    if (it->second.width != width || it->second.height != height) {
        glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0,
                     gl_format, gl_type, data);
        it->second.width = width;
        it->second.height = height;
        handle.width = width;
        handle.height = height;
    } else {
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                        gl_format, gl_type, data);
    }

    glBindTexture(GL_TEXTURE_2D, 0);

    return checkGLError("uploadTexture");
}

bool GLESRenderContext::downloadTexture(const TextureHandle& handle, uint8_t* data,
                                          size_t max_size) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (context_lost_.load() || !initialized_.load()) {
        return false;
    }

    if (!handle.isValid() || handle.type != TextureHandle::Type::OpenGLES) {
        return false;
    }

    if (data == nullptr || max_size == 0) {
        return false;
    }

    auto it = textures_.find(handle.id);
    if (it == textures_.end()) {
        return false;
    }

    size_t required_size = handle.memorySize();
    if (max_size < required_size) {
        return false;
    }

    // FBO를 사용하여 glReadPixels로 읽기
    GLuint fbo = 0;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, it->second.gl_id, 0);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        LOGE("Framebuffer incomplete for texture download");
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDeleteFramebuffers(1, &fbo);
        return false;
    }

    GLenum gl_format = toGLFormat(handle.format);
    GLenum gl_type = toGLType(handle.format);

    glReadPixels(0, 0, handle.width, handle.height, gl_format, gl_type, data);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);

    return checkGLError("downloadTexture");
}

bool GLESRenderContext::makeCurrent() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (context_lost_.load() || display_ == EGL_NO_DISPLAY) {
        return false;
    }

    return eglMakeCurrent(display_, surface_, surface_, context_) == EGL_TRUE;
}

void GLESRenderContext::doneCurrent() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (display_ != EGL_NO_DISPLAY) {
        eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    }
}

void GLESRenderContext::onSurfaceCreated() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (context_lost_.load()) {
        LOGI("Recovering from context loss");

        if (initEGL()) {
            context_lost_.store(false);
            initialized_.store(true);
            LOGI("Context recovery successful");
        } else {
            LOGE("Context recovery failed");
        }
    }
}

void GLESRenderContext::onSurfaceDestroyed() {
    std::lock_guard<std::mutex> lock(mutex_);

    LOGI("Surface destroyed, marking context as lost");

    // Context 손실 플래그 설정
    context_lost_.store(true);

    // 콜백 호출 (앱에서 TextureHandle 무효화 처리)
    if (context_lost_callback_) {
        context_lost_callback_();
    }

    // 모든 텍스처 핸들 무효화 (GL 리소스는 이미 무효)
    textures_.clear();

    // EGL 리소스 해제
    releaseEGL();

    initialized_.store(false);
}

GLuint GLESRenderContext::getGLTextureId(const TextureHandle& handle) const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!handle.isValid() || handle.type != TextureHandle::Type::OpenGLES) {
        return 0;
    }

    auto it = textures_.find(handle.id);
    if (it == textures_.end()) {
        return 0;
    }

    return it->second.gl_id;
}

bool GLESRenderContext::isExtensionSupported(const char* extension_name) const {
    if (extension_name == nullptr) {
        return false;
    }

    const char* extensions = reinterpret_cast<const char*>(glGetString(GL_EXTENSIONS));
    if (extensions == nullptr) {
        return false;
    }

    return std::strstr(extensions, extension_name) != nullptr;
}

bool GLESRenderContext::checkGLError(const char* operation) const {
    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        LOGE("GL error after %s: 0x%x", operation, error);
        return false;
    }
    return true;
}

bool GLESRenderContext::dumpTexture(const TextureHandle& handle,
                                     const std::string& file_path) {
    // Android에서는 별도 이미지 라이브러리 필요
    // 기본 구현은 downloadTexture + 외부 저장으로 대체 가능
    (void)handle;
    (void)file_path;
    LOGW("dumpTexture not implemented on Android");
    return false;
}

size_t GLESRenderContext::getTextureCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return textures_.size();
}

size_t GLESRenderContext::getTextureMemoryUsage() const {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t total = 0;
    for (const auto& pair : textures_) {
        const TextureInfo& info = pair.second;
        int bpp = 4;  // 기본 RGBA8
        switch (info.format) {
            case TextureFormat::RGBA8:   bpp = 4; break;
            case TextureFormat::RGB8:    bpp = 3; break;
            case TextureFormat::R8:      bpp = 1; break;
            case TextureFormat::RGBA16F: bpp = 8; break;
            default: break;
        }
        total += static_cast<size_t>(info.width) * info.height * bpp;
    }
    return total;
}

GLenum GLESRenderContext::toGLFormat(TextureFormat format) const {
    switch (format) {
        case TextureFormat::RGBA8:
        case TextureFormat::RGBA16F:
            return GL_RGBA;
        case TextureFormat::RGB8:
            return GL_RGB;
        case TextureFormat::R8:
            return GL_RED;
        default:
            return GL_RGBA;
    }
}

GLenum GLESRenderContext::toGLInternalFormat(TextureFormat format) const {
    switch (format) {
        case TextureFormat::RGBA8:   return GL_RGBA8;
        case TextureFormat::RGB8:    return GL_RGB8;
        case TextureFormat::R8:      return GL_R8;
        case TextureFormat::RGBA16F: return GL_RGBA16F;
        default:                     return GL_RGBA8;
    }
}

GLenum GLESRenderContext::toGLType(TextureFormat format) const {
    switch (format) {
        case TextureFormat::RGBA8:
        case TextureFormat::RGB8:
        case TextureFormat::R8:
            return GL_UNSIGNED_BYTE;
        case TextureFormat::RGBA16F:
            return GL_HALF_FLOAT;
        default:
            return GL_UNSIGNED_BYTE;
    }
}

} // namespace iris_sdk

#endif // __ANDROID__ || IRIS_SDK_FORCE_GLES
