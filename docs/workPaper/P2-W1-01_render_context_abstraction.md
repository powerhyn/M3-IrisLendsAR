# P2-W1-01. 렌더링 컨텍스트 추상화 및 플랫폼 인터페이스

## 작업 정보

| 항목 | 내용 |
|------|------|
| **작업 ID** | P2-W1-01 |
| **Phase** | Phase 1: 기반 구조 리팩토링 |
| **상태** | ⏳ 대기 |
| **예상 기간** | 2일 |
| **의존성** | 없음 (Phase 1 첫 작업) |
| **담당** | systems-programming:cpp-pro |

---

## 1. 목표

GPU 렌더링과 뷰티 필터를 위한 플랫폼 독립적 렌더링 컨텍스트 추상화 레이어 구현

### 핵심 산출물
- `IRenderContext` 인터페이스
- `TextureHandle` 추상화 구조체
- `GLESRenderContext` (Android OpenGL ES 구현)
- `CPURenderContext` (CPU 폴백 구현)

---

## 2. 상세 작업

### 2.1 TextureHandle 추상화

**파일**: `cpp/include/iris_sdk/gpu/texture_handle.h`

```cpp
#ifndef IRIS_SDK_TEXTURE_HANDLE_H
#define IRIS_SDK_TEXTURE_HANDLE_H

namespace iris_sdk {

/**
 * @brief 플랫폼 독립적 텍스처 핸들
 *
 * GLuint (OpenGL ES), MTLTexture* (Metal), cv::Mat* (CPU) 등을
 * 통합 관리하기 위한 추상화 구조체
 */
struct TextureHandle {
    enum class Type {
        Invalid = 0,
        OpenGLES,   // Android
        Metal,      // iOS (향후)
        CPU         // Fallback
    };

    void* native_handle = nullptr;  // 플랫폼별 핸들
    Type type = Type::Invalid;
    int width = 0;
    int height = 0;
    int format = 0;  // 플랫폼별 포맷 코드

    bool isValid() const { return native_handle != nullptr && type != Type::Invalid; }
    void invalidate() { native_handle = nullptr; type = Type::Invalid; }
};

} // namespace iris_sdk

#endif // IRIS_SDK_TEXTURE_HANDLE_H
```

**체크리스트**:
- [ ] 구조체 정의
- [ ] 헬퍼 메서드 (isValid, invalidate)
- [ ] 포맷 상수 정의

### 2.2 IRenderContext 인터페이스

**파일**: `cpp/include/iris_sdk/gpu/render_context.h`

```cpp
#ifndef IRIS_SDK_RENDER_CONTEXT_H
#define IRIS_SDK_RENDER_CONTEXT_H

#include "texture_handle.h"
#include <cstdint>
#include <memory>

namespace iris_sdk {

/**
 * @brief 렌더링 컨텍스트 추상화 인터페이스
 *
 * OpenGL ES, Metal, CPU 백엔드를 통합하는 플랫폼 독립적 인터페이스.
 * DI(Dependency Injection) 패턴으로 테스트 용이성 확보.
 */
class IRenderContext {
public:
    virtual ~IRenderContext() = default;

    //=== 라이프사이클 ===
    virtual bool initialize() = 0;
    virtual void release() = 0;
    virtual bool isInitialized() const = 0;

    //=== 텍스처 관리 ===
    /**
     * @brief 텍스처 생성
     * @param width 너비
     * @param height 높이
     * @param format 픽셀 포맷 (RGBA, RGB 등)
     * @return 생성된 텍스처 핸들
     */
    virtual TextureHandle createTexture(int width, int height, int format = 0) = 0;

    /**
     * @brief 텍스처 삭제
     */
    virtual void deleteTexture(TextureHandle& handle) = 0;

    /**
     * @brief CPU 데이터 → GPU 텍스처 업로드
     */
    virtual bool uploadTexture(TextureHandle& handle, const uint8_t* data,
                               int width, int height, int format) = 0;

    /**
     * @brief GPU 텍스처 → CPU 데이터 다운로드
     */
    virtual bool downloadTexture(const TextureHandle& handle, uint8_t* data,
                                 int max_size) = 0;

    //=== 컨텍스트 전환 ===
    virtual bool makeCurrent() = 0;
    virtual void doneCurrent() = 0;

    //=== Android 라이프사이클 (Context Loss 대응) ===
    /**
     * @brief Surface 생성 시 호출 (Activity onResume 등)
     *
     * GL Context가 재생성된 후 리소스를 복구합니다.
     */
    virtual void onSurfaceCreated() {}

    /**
     * @brief Surface 파괴 시 호출 (Activity onPause/onStop 등)
     *
     * GL Context 손실 전에 리소스를 정리합니다.
     * 이 시점 이후 모든 TextureHandle은 무효화됩니다.
     */
    virtual void onSurfaceDestroyed() {}

    /**
     * @brief Context Loss 상태 확인
     * @return true면 모든 GPU 리소스가 무효화된 상태
     */
    virtual bool isContextLost() const { return false; }

    //=== 메타데이터 ===
    virtual const char* getName() const = 0;
    virtual int getMajorVersion() const = 0;
    virtual int getMinorVersion() const = 0;
    virtual bool supportsGpu() const = 0;

    //=== 팩토리 ===
    /**
     * @brief 플랫폼에 적합한 RenderContext 생성
     * @param prefer_gpu GPU 선호 여부
     * @return 생성된 컨텍스트 (소유권 이전)
     */
    static std::unique_ptr<IRenderContext> create(bool prefer_gpu = true);
};

} // namespace iris_sdk

#endif // IRIS_SDK_RENDER_CONTEXT_H
```

**체크리스트**:
- [ ] 인터페이스 정의
- [ ] 라이프사이클 메서드
- [ ] 텍스처 관리 메서드
- [ ] 팩토리 메서드

### 2.3 GLESRenderContext 구현 (Android)

**파일**: `cpp/include/iris_sdk/gpu/gles_render_context.h`

```cpp
#ifndef IRIS_SDK_GLES_RENDER_CONTEXT_H
#define IRIS_SDK_GLES_RENDER_CONTEXT_H

#if defined(__ANDROID__) && defined(IRIS_SDK_HAS_GLES)

#include "render_context.h"
#include <EGL/egl.h>
#include <GLES3/gl31.h>
#include <mutex>
#include <atomic>

namespace iris_sdk {

class GLESRenderContext : public IRenderContext {
public:
    GLESRenderContext();
    ~GLESRenderContext() override;

    // 복사/이동 금지
    GLESRenderContext(const GLESRenderContext&) = delete;
    GLESRenderContext& operator=(const GLESRenderContext&) = delete;

    //=== IRenderContext 구현 ===
    bool initialize() override;
    void release() override;
    bool isInitialized() const override;

    TextureHandle createTexture(int width, int height, int format) override;
    void deleteTexture(TextureHandle& handle) override;
    bool uploadTexture(TextureHandle& handle, const uint8_t* data,
                       int width, int height, int format) override;
    bool downloadTexture(const TextureHandle& handle, uint8_t* data,
                         int max_size) override;

    bool makeCurrent() override;
    void doneCurrent() override;

    const char* getName() const override { return "GLESRenderContext"; }
    int getMajorVersion() const override { return major_version_; }
    int getMinorVersion() const override { return minor_version_; }
    bool supportsGpu() const override { return true; }

    //=== GLES 전용 ===
    GLuint getGLTextureId(const TextureHandle& handle) const;

    //=== Android 라이프사이클 (Context Loss 대응) ===
    void onSurfaceCreated() override;
    void onSurfaceDestroyed() override;
    bool isContextLost() const override { return context_lost_.load(); }

    /**
     * @brief Context Loss 발생 시 호출할 콜백 등록
     *
     * 앱에서 TextureHandle 무효화 처리를 위해 사용
     */
    using ContextLostCallback = std::function<void()>;
    void setContextLostCallback(ContextLostCallback callback) {
        context_lost_callback_ = std::move(callback);
    }

private:
    bool initEGL();
    void releaseEGL();
    bool checkGLESVersion();

    EGLDisplay display_ = EGL_NO_DISPLAY;
    EGLContext context_ = EGL_NO_CONTEXT;
    EGLSurface surface_ = EGL_NO_SURFACE;
    EGLConfig config_ = nullptr;

    int major_version_ = 0;
    int minor_version_ = 0;

    std::atomic<bool> initialized_{false};
    std::atomic<bool> context_lost_{false};
    ContextLostCallback context_lost_callback_;
    std::mutex mutex_;
};

} // namespace iris_sdk

#endif // __ANDROID__ && IRIS_SDK_HAS_GLES

#endif // IRIS_SDK_GLES_RENDER_CONTEXT_H
```

**구현 파일**: `cpp/src/gpu/gles_render_context.cpp`

**핵심 구현 사항**:
1. EGL 초기화 (EGLDisplay, EGLContext, PBuffer Surface)
2. OpenGL ES 3.1 버전 체크 (3.0 폴백 지원)
3. glGenTextures / glDeleteTextures
4. glTexImage2D / glReadPixels

**체크리스트**:
- [ ] EGL 초기화 코드
- [ ] 텍스처 생성/삭제
- [ ] 업로드 (glTexImage2D)
- [ ] 다운로드 (glReadPixels)
- [ ] 버전 체크 및 폴백
- [ ] Context Loss 처리

### 2.3.1 Context Loss 처리 구현

**파일**: `cpp/src/gpu/gles_render_context.cpp` (추가)

```cpp
void GLESRenderContext::onSurfaceDestroyed() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Context 손실 플래그 설정
    context_lost_.store(true);

    // 콜백 호출 (앱에서 TextureHandle 무효화 처리)
    if (context_lost_callback_) {
        context_lost_callback_();
    }

    // EGL 리소스 해제 (이미 무효화된 상태이지만 명시적 정리)
    releaseEGL();

    initialized_.store(false);
}

void GLESRenderContext::onSurfaceCreated() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Context 복구
    if (context_lost_.load()) {
        if (initEGL()) {
            context_lost_.store(false);
            initialized_.store(true);
        }
    }
}

// 텍스처 생성 시 Context Loss 체크
TextureHandle GLESRenderContext::createTexture(int width, int height, int format) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (context_lost_.load()) {
        // Context 손실 상태 → 빈 핸들 반환
        return TextureHandle{};
    }

    // ... 기존 구현 ...
}
```

**Android Activity/Fragment 연동 예시**:

```java
// Java/Kotlin에서 라이프사이클 이벤트 전달
@Override
protected void onPause() {
    super.onPause();
    IrisSDKNative.onSurfaceDestroyed();
}

@Override
protected void onResume() {
    super.onResume();
    IrisSDKNative.onSurfaceCreated();
}
```

### 2.4 CPURenderContext 구현 (Fallback)

**파일**: `cpp/include/iris_sdk/gpu/cpu_render_context.h`

```cpp
#ifndef IRIS_SDK_CPU_RENDER_CONTEXT_H
#define IRIS_SDK_CPU_RENDER_CONTEXT_H

#include "render_context.h"
#include <opencv2/core.hpp>
#include <map>
#include <mutex>

namespace iris_sdk {

/**
 * @brief CPU 기반 렌더링 컨텍스트 (GPU 미지원 환경용)
 *
 * cv::Mat를 TextureHandle로 래핑하여 동일한 인터페이스 제공
 */
class CPURenderContext : public IRenderContext {
public:
    CPURenderContext();
    ~CPURenderContext() override;

    //=== IRenderContext 구현 ===
    bool initialize() override;
    void release() override;
    bool isInitialized() const override;

    TextureHandle createTexture(int width, int height, int format) override;
    void deleteTexture(TextureHandle& handle) override;
    bool uploadTexture(TextureHandle& handle, const uint8_t* data,
                       int width, int height, int format) override;
    bool downloadTexture(const TextureHandle& handle, uint8_t* data,
                         int max_size) override;

    bool makeCurrent() override { return true; }  // CPU는 컨텍스트 전환 불필요
    void doneCurrent() override {}

    const char* getName() const override { return "CPURenderContext"; }
    int getMajorVersion() const override { return 1; }
    int getMinorVersion() const override { return 0; }
    bool supportsGpu() const override { return false; }

    //=== CPU 전용 ===
    cv::Mat* getCvMat(const TextureHandle& handle) const;

private:
    std::map<void*, std::unique_ptr<cv::Mat>> textures_;
    std::mutex mutex_;
    bool initialized_ = false;
    int next_id_ = 1;
};

} // namespace iris_sdk

#endif // IRIS_SDK_CPU_RENDER_CONTEXT_H
```

**체크리스트**:
- [ ] cv::Mat 기반 텍스처 관리
- [ ] 핸들 ID 발급 및 맵 관리
- [ ] 메모리 누수 방지

### 2.5 팩토리 메서드 구현

**파일**: `cpp/src/gpu/render_context.cpp`

```cpp
#include "iris_sdk/gpu/render_context.h"
#include "iris_sdk/gpu/cpu_render_context.h"

#if defined(__ANDROID__) && defined(IRIS_SDK_HAS_GLES)
#include "iris_sdk/gpu/gles_render_context.h"
#endif

namespace iris_sdk {

std::unique_ptr<IRenderContext> IRenderContext::create(bool prefer_gpu) {
#if defined(__ANDROID__) && defined(IRIS_SDK_HAS_GLES)
    if (prefer_gpu) {
        auto gles_ctx = std::make_unique<GLESRenderContext>();
        if (gles_ctx->initialize()) {
            return gles_ctx;
        }
        // GPU 초기화 실패 → CPU 폴백
    }
#endif
    // CPU 컨텍스트 (Desktop, GPU 미지원 기기)
    auto cpu_ctx = std::make_unique<CPURenderContext>();
    cpu_ctx->initialize();
    return cpu_ctx;
}

} // namespace iris_sdk
```

**체크리스트**:
- [ ] 플랫폼별 조건 컴파일
- [ ] GPU 초기화 실패 시 CPU 폴백
- [ ] Desktop 환경 지원

---

## 3. 단위 테스트

### 3.1 MockRenderContext

**파일**: `cpp/tests/mock_render_context.h`

```cpp
#include "iris_sdk/gpu/render_context.h"
#include <gmock/gmock.h>

namespace iris_sdk {
namespace testing {

class MockRenderContext : public IRenderContext {
public:
    MOCK_METHOD(bool, initialize, (), (override));
    MOCK_METHOD(void, release, (), (override));
    MOCK_METHOD(bool, isInitialized, (), (const, override));
    MOCK_METHOD(TextureHandle, createTexture, (int, int, int), (override));
    MOCK_METHOD(void, deleteTexture, (TextureHandle&), (override));
    // ... 나머지 메서드
};

} // namespace testing
} // namespace iris_sdk
```

### 3.2 테스트 케이스

**파일**: `cpp/tests/test_render_context.cpp`

```cpp
TEST(RenderContextFactory, CreatesCPUContextOnDesktop) {
    auto ctx = IRenderContext::create(false);
    ASSERT_NE(ctx, nullptr);
    EXPECT_FALSE(ctx->supportsGpu());
    EXPECT_STREQ(ctx->getName(), "CPURenderContext");
}

TEST(CPURenderContext, CreateAndDeleteTexture) {
    CPURenderContext ctx;
    ASSERT_TRUE(ctx.initialize());

    auto tex = ctx.createTexture(640, 480, 0);
    EXPECT_TRUE(tex.isValid());
    EXPECT_EQ(tex.width, 640);
    EXPECT_EQ(tex.height, 480);

    ctx.deleteTexture(tex);
    EXPECT_FALSE(tex.isValid());
}

TEST(CPURenderContext, UploadDownloadRoundTrip) {
    CPURenderContext ctx;
    ctx.initialize();

    std::vector<uint8_t> input(640 * 480 * 4, 128);  // RGBA
    auto tex = ctx.createTexture(640, 480, 0);

    ASSERT_TRUE(ctx.uploadTexture(tex, input.data(), 640, 480, 0));

    std::vector<uint8_t> output(640 * 480 * 4);
    ASSERT_TRUE(ctx.downloadTexture(tex, output.data(), output.size()));

    EXPECT_EQ(input, output);

    ctx.deleteTexture(tex);
}
```

**체크리스트**:
- [ ] Mock 클래스 작성
- [ ] 팩토리 테스트
- [ ] 텍스처 생성/삭제 테스트
- [ ] 업로드/다운로드 Round-trip 테스트

---

## 4. CMake 설정

**파일**: `cpp/CMakeLists.txt` (추가)

```cmake
# GPU 모듈
set(GPU_SOURCES
    src/gpu/render_context.cpp
    src/gpu/cpu_render_context.cpp
)

set(GPU_HEADERS
    include/iris_sdk/gpu/texture_handle.h
    include/iris_sdk/gpu/render_context.h
    include/iris_sdk/gpu/cpu_render_context.h
)

if(ANDROID AND IRIS_SDK_HAS_GLES)
    list(APPEND GPU_SOURCES src/gpu/gles_render_context.cpp)
    list(APPEND GPU_HEADERS include/iris_sdk/gpu/gles_render_context.h)

    # EGL 및 GLES 라이브러리 링크
    find_library(EGL_LIBRARY EGL)
    find_library(GLESV3_LIBRARY GLESv3)
    target_link_libraries(iris_sdk PRIVATE ${EGL_LIBRARY} ${GLESV3_LIBRARY})
endif()
```

---

## 5. 완료 기준

- [ ] `IRenderContext` 인터페이스 정의 완료
- [ ] `TextureHandle` 추상화 구조체 정의
- [ ] `CPURenderContext` 구현 및 테스트 통과
- [ ] `GLESRenderContext` 구현 (Android 빌드 성공)
- [ ] **Context Loss 처리** (onSurfaceCreated/Destroyed) 구현
- [ ] 팩토리 메서드 동작 확인
- [ ] 단위 테스트 100% 통과
- [ ] Desktop 빌드 성공

---

## 6. 다음 작업

- **P2-W1-02**: BeautyFilterConfigV2 정의
- **P2-W1-03**: BeautyROIManager 구현
