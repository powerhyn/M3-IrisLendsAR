/**
 * @file shader_manager.cpp
 * @brief ShaderManager 구현
 */

#include "iris_sdk/gpu/shader_manager.h"

#if IRIS_SDK_GPU_AVAILABLE
#include <android/log.h>
#define LOG_TAG "ShaderManager"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#include <cstdio>
#define LOGI(...) printf("[ShaderManager INFO] " __VA_ARGS__); printf("\n")
#define LOGW(...) printf("[ShaderManager WARN] " __VA_ARGS__); printf("\n")
#define LOGE(...) printf("[ShaderManager ERROR] " __VA_ARGS__); printf("\n")
#endif

namespace iris_sdk {

ShaderManager::ShaderManager() = default;

ShaderManager::~ShaderManager() {
    releaseAll();
}

bool ShaderManager::createProgram(const char* vertex_source,
                                   const char* fragment_source,
                                   GLuint& out_program) {
#if IRIS_SDK_GPU_AVAILABLE
    std::lock_guard<std::mutex> lock(mutex_);

    out_program = 0;

    if (!vertex_source || !fragment_source) {
        LOGE("Shader source is null");
        return false;
    }

    // 버텍스 셰이더 컴파일
    GLuint vertex_shader = 0;
    if (!compileShader(GL_VERTEX_SHADER, vertex_source, vertex_shader)) {
        LOGE("Failed to compile vertex shader");
        return false;
    }

    // 프래그먼트 셰이더 컴파일
    GLuint fragment_shader = 0;
    if (!compileShader(GL_FRAGMENT_SHADER, fragment_source, fragment_shader)) {
        LOGE("Failed to compile fragment shader");
        glDeleteShader(vertex_shader);
        return false;
    }

    // 프로그램 링크
    if (!linkProgram(vertex_shader, fragment_shader, out_program)) {
        LOGE("Failed to link shader program");
        glDeleteShader(vertex_shader);
        glDeleteShader(fragment_shader);
        return false;
    }

    // 셰이더는 프로그램에 링크되면 삭제 가능
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    LOGI("Shader program created: %u", out_program);
    return true;
#else
    // Desktop 스텁: 항상 성공 반환
    (void)vertex_source;
    (void)fragment_source;
    static GLuint stub_program_counter = 1;
    out_program = stub_program_counter++;
    LOGI("Shader program created (stub): %u", out_program);
    return true;
#endif
}

GLuint ShaderManager::getProgram(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = program_cache_.find(name);
    if (it != program_cache_.end()) {
        return it->second;
    }
    return 0;
}

void ShaderManager::cacheProgram(const std::string& name, GLuint program) {
    std::lock_guard<std::mutex> lock(mutex_);

    // 기존 프로그램이 있으면 삭제
    auto it = program_cache_.find(name);
    if (it != program_cache_.end() && it->second != program) {
#if IRIS_SDK_GPU_AVAILABLE
        glDeleteProgram(it->second);
#endif
        LOGW("Replacing cached program '%s': %u -> %u",
             name.c_str(), it->second, program);
    }

    program_cache_[name] = program;
    LOGI("Program cached: '%s' = %u", name.c_str(), program);
}

void ShaderManager::releaseAll() {
    std::lock_guard<std::mutex> lock(mutex_);

#if IRIS_SDK_GPU_AVAILABLE
    for (const auto& pair : program_cache_) {
        if (pair.second != 0) {
            glDeleteProgram(pair.second);
        }
    }
#endif

    size_t count = program_cache_.size();
    program_cache_.clear();

    LOGI("Released %zu shader programs", count);
}

size_t ShaderManager::getCachedProgramCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return program_cache_.size();
}

bool ShaderManager::compileShader(GLenum type, const char* source, GLuint& out_shader) {
#if IRIS_SDK_GPU_AVAILABLE
    out_shader = glCreateShader(type);
    if (out_shader == 0) {
        LOGE("Failed to create shader object");
        return false;
    }

    glShaderSource(out_shader, 1, &source, nullptr);
    glCompileShader(out_shader);

    GLint compiled = 0;
    glGetShaderiv(out_shader, GL_COMPILE_STATUS, &compiled);

    if (!compiled) {
        std::string log = getShaderLog(out_shader);
        LOGE("Shader compilation failed:\n%s", log.c_str());
        glDeleteShader(out_shader);
        out_shader = 0;
        return false;
    }

    return true;
#else
    // Desktop 스텁
    (void)type;
    (void)source;
    static GLuint stub_shader_counter = 1;
    out_shader = stub_shader_counter++;
    return true;
#endif
}

bool ShaderManager::linkProgram(GLuint vertex_shader,
                                 GLuint fragment_shader,
                                 GLuint& out_program) {
#if IRIS_SDK_GPU_AVAILABLE
    out_program = glCreateProgram();
    if (out_program == 0) {
        LOGE("Failed to create program object");
        return false;
    }

    glAttachShader(out_program, vertex_shader);
    glAttachShader(out_program, fragment_shader);
    glLinkProgram(out_program);

    GLint linked = 0;
    glGetProgramiv(out_program, GL_LINK_STATUS, &linked);

    if (!linked) {
        std::string log = getProgramLog(out_program);
        LOGE("Program link failed:\n%s", log.c_str());
        glDeleteProgram(out_program);
        out_program = 0;
        return false;
    }

    return true;
#else
    // Desktop 스텁
    (void)vertex_shader;
    (void)fragment_shader;
    static GLuint stub_program_counter = 100;
    out_program = stub_program_counter++;
    return true;
#endif
}

std::string ShaderManager::getShaderLog(GLuint shader) {
#if IRIS_SDK_GPU_AVAILABLE
    GLint log_length = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);

    if (log_length > 1) {
        std::string log(static_cast<size_t>(log_length), '\0');
        glGetShaderInfoLog(shader, log_length, nullptr, &log[0]);
        return log;
    }
    return "";
#else
    (void)shader;
    return "";
#endif
}

std::string ShaderManager::getProgramLog(GLuint program) {
#if IRIS_SDK_GPU_AVAILABLE
    GLint log_length = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_length);

    if (log_length > 1) {
        std::string log(static_cast<size_t>(log_length), '\0');
        glGetProgramInfoLog(program, log_length, nullptr, &log[0]);
        return log;
    }
    return "";
#else
    (void)program;
    return "";
#endif
}

bool ShaderManager::checkGLError(const char* operation) {
#if IRIS_SDK_GPU_AVAILABLE
    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        const char* error_str = "Unknown";
        switch (error) {
            case GL_INVALID_ENUM:      error_str = "GL_INVALID_ENUM"; break;
            case GL_INVALID_VALUE:     error_str = "GL_INVALID_VALUE"; break;
            case GL_INVALID_OPERATION: error_str = "GL_INVALID_OPERATION"; break;
            case GL_OUT_OF_MEMORY:     error_str = "GL_OUT_OF_MEMORY"; break;
            case GL_INVALID_FRAMEBUFFER_OPERATION:
                error_str = "GL_INVALID_FRAMEBUFFER_OPERATION"; break;
        }
        LOGE("GL Error after '%s': %s (0x%x)", operation, error_str, error);
        return false;
    }
    return true;
#else
    (void)operation;
    return true;
#endif
}

} // namespace iris_sdk
