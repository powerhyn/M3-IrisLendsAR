/**
 * @file render_context.cpp
 * @brief IRenderContext 팩토리 메서드 구현
 */

#include "iris_sdk/gpu/render_context.h"
#include "iris_sdk/gpu/cpu_render_context.h"

#if defined(__ANDROID__) || defined(IRIS_SDK_FORCE_GLES)
#include "iris_sdk/gpu/gles_render_context.h"
#endif

namespace iris_sdk {

std::unique_ptr<IRenderContext> IRenderContext::create(bool prefer_gpu) {
#if defined(__ANDROID__) || defined(IRIS_SDK_FORCE_GLES)
    if (prefer_gpu) {
        auto gles_ctx = std::make_unique<GLESRenderContext>();
        if (gles_ctx->initialize()) {
            return gles_ctx;
        }
        // GPU 초기화 실패 → CPU 폴백
    }
#else
    (void)prefer_gpu;  // 미사용 경고 방지
#endif

    // CPU 컨텍스트 (Desktop 또는 GPU 미지원 기기)
    auto cpu_ctx = std::make_unique<CPURenderContext>();
    cpu_ctx->initialize();
    return cpu_ctx;
}

} // namespace iris_sdk
