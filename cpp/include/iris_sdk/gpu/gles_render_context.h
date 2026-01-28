/**
 * @file gles_render_context.h
 * @brief Android OpenGL ES 렌더링 컨텍스트 구현
 *
 * EGL Context 관리, 텍스처 생성/삭제, Context Loss 처리 등을 담당.
 * Android 플랫폼 전용 (조건부 컴파일).
 */

#ifndef IRIS_SDK_GLES_RENDER_CONTEXT_H
#define IRIS_SDK_GLES_RENDER_CONTEXT_H

// Android OpenGL ES 환경에서만 컴파일
#if defined(__ANDROID__) || defined(IRIS_SDK_FORCE_GLES)

#include "render_context.h"

#include <EGL/egl.h>
#include <GLES3/gl31.h>

#include <atomic>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace iris_sdk {

/**
 * @brief Android OpenGL ES 렌더링 컨텍스트
 *
 * OpenGL ES 3.1을 기본으로 하며, 3.0 폴백 지원.
 * EGL PBuffer Surface를 사용하여 오프스크린 렌더링 가능.
 */
class GLESRenderContext : public IRenderContext {
public:
    GLESRenderContext();
    ~GLESRenderContext() override;

    // 복사/이동 금지
    GLESRenderContext(const GLESRenderContext&) = delete;
    GLESRenderContext& operator=(const GLESRenderContext&) = delete;
    GLESRenderContext(GLESRenderContext&&) = delete;
    GLESRenderContext& operator=(GLESRenderContext&&) = delete;

    //=========================================================================
    // IRenderContext 구현
    //=========================================================================

    bool initialize() override;
    void release() override;
    bool isInitialized() const override;

    TextureHandle createTexture(int width, int height,
                                 TextureFormat format = TextureFormat::RGBA8) override;
    void deleteTexture(TextureHandle& handle) override;

    bool uploadTexture(TextureHandle& handle, const uint8_t* data,
                        int width, int height, TextureFormat format) override;
    bool downloadTexture(const TextureHandle& handle, uint8_t* data,
                          size_t max_size) override;

    bool makeCurrent() override;
    void doneCurrent() override;

    const char* getName() const override { return "GLESRenderContext"; }
    int getMajorVersion() const override { return major_version_; }
    int getMinorVersion() const override { return minor_version_; }
    bool supportsGpu() const override { return true; }

    //=========================================================================
    // Android 라이프사이클 (Context Loss 대응)
    //=========================================================================

    void onSurfaceCreated() override;
    void onSurfaceDestroyed() override;
    bool isContextLost() const override { return context_lost_.load(); }

    /**
     * @brief Context Loss 콜백 타입
     *
     * Context가 손실될 때 호출되어 앱에서 TextureHandle 무효화 처리 가능.
     */
    using ContextLostCallback = std::function<void()>;

    /**
     * @brief Context Loss 콜백 등록
     * @param callback Context 손실 시 호출될 함수
     */
    void setContextLostCallback(ContextLostCallback callback) {
        std::lock_guard<std::mutex> lock(mutex_);
        context_lost_callback_ = std::move(callback);
    }

    //=========================================================================
    // OpenGL ES 전용 메서드
    //=========================================================================

    /**
     * @brief TextureHandle에서 OpenGL 텍스처 ID 추출
     * @param handle 대상 핸들
     * @return GLuint 텍스처 ID (무효 시 0)
     */
    GLuint getGLTextureId(const TextureHandle& handle) const;

    /**
     * @brief OpenGL ES 확장 지원 여부 확인
     * @param extension_name 확장 이름 (예: "GL_EXT_disjoint_timer_query")
     * @return 지원 시 true
     */
    bool isExtensionSupported(const char* extension_name) const;

    /**
     * @brief OpenGL 에러 체크 및 로깅 (디버그용)
     * @param operation 현재 작업 이름
     * @return 에러 발생 시 false
     */
    bool checkGLError(const char* operation) const;

    //=========================================================================
    // 디버깅 도구
    //=========================================================================

    bool dumpTexture(const TextureHandle& handle,
                      const std::string& file_path) override;

    /**
     * @brief 할당된 텍스처 수 조회
     */
    size_t getTextureCount() const;

    /**
     * @brief 할당된 텍스처 총 메모리 (바이트)
     */
    size_t getTextureMemoryUsage() const;

private:
    //=========================================================================
    // EGL 관리
    //=========================================================================

    bool initEGL();
    void releaseEGL();
    bool checkGLESVersion();

    /**
     * @brief TextureFormat → OpenGL 포맷 변환
     */
    GLenum toGLFormat(TextureFormat format) const;
    GLenum toGLInternalFormat(TextureFormat format) const;
    GLenum toGLType(TextureFormat format) const;

    //=========================================================================
    // 멤버 변수
    //=========================================================================

    // EGL 핸들
    EGLDisplay display_ = EGL_NO_DISPLAY;
    EGLContext context_ = EGL_NO_CONTEXT;
    EGLSurface surface_ = EGL_NO_SURFACE;
    EGLConfig config_ = nullptr;

    // 버전 정보
    int major_version_ = 0;
    int minor_version_ = 0;

    // 상태 플래그
    std::atomic<bool> initialized_{false};
    std::atomic<bool> context_lost_{false};

    // Context Loss 콜백
    ContextLostCallback context_lost_callback_;

    // 텍스처 추적 (Context Loss 시 일괄 무효화용)
    struct TextureInfo {
        GLuint gl_id;
        int width;
        int height;
        TextureFormat format;
    };
    std::unordered_map<uint64_t, TextureInfo> textures_;
    uint64_t next_texture_id_ = 1;

    // 동기화
    mutable std::mutex mutex_;
};

} // namespace iris_sdk

#endif // __ANDROID__ || IRIS_SDK_FORCE_GLES

#endif // IRIS_SDK_GLES_RENDER_CONTEXT_H
