/**
 * @file sdk_api_v2.cpp
 * @brief IrisLensSDK V2 C API 구현
 *
 * Beauty Filter V2, Face Warp, GPU 텍스처 처리 등 Phase 2 기능의 C API 구현.
 *
 * @author IrisLensSDK Team
 * @version 2.0.0
 * @copyright Apache 2.0 License
 */

#include "iris_sdk/sdk_api.h"
#include "iris_sdk/beauty_filter.h"
#include "iris_sdk/cpu_beauty_backend.h"
#include "iris_sdk/beauty_roi_manager.h"

#ifdef IRIS_SDK_HAS_GLES
#include "iris_sdk/gpu/gpu_beauty_backend.h"
#include "iris_sdk/gpu/gpu_lens_renderer.h"
#include "iris_sdk/gpu/texture_handle.h"
#endif

#include <memory>
#include <mutex>
#include <set>
#include <cstring>
#include <algorithm>

namespace {

// ============================================================================
// 전역 상태
// ============================================================================

std::mutex g_gpu_mutex;

#ifdef IRIS_SDK_HAS_GLES
std::unique_ptr<iris_sdk::GPUBeautyBackend> g_gpu_beauty;
std::unique_ptr<iris_sdk::GPULensRenderer> g_gpu_lens;
std::set<uint32_t> g_managed_textures;
#endif

// CPU 백엔드는 항상 사용 가능
std::unique_ptr<iris_sdk::CPUBeautyBackend> g_cpu_beauty;
std::mutex g_cpu_mutex;

// ============================================================================
// 헬퍼 함수
// ============================================================================

/**
 * @brief C API 구조체를 C++ 구조체로 변환
 */
BeautyFilterConfigV2 toCppConfigV2(const IrisBeautyConfigV2* c_config) {
    BeautyFilterConfigV2 config = {};

    if (!c_config) {
        // 기본값 반환
        iris_sdk_default_beauty_config_v2(&config);
        return config;
    }

    config.enabled = c_config->enabled != 0;
    config.intensity = c_config->intensity;
    config.smoothing = c_config->smoothing;
    config.brightness = c_config->brightness;
    config.softFocus = c_config->soft_focus;
    config.whitening = c_config->whitening;
    config.colorBalance = c_config->color_balance;
    config.wrinkleRemove = c_config->wrinkle_remove;
    config.skinQuality = c_config->skin_quality;
    config.smoothIntensity = c_config->smooth_intensity;
    config.poreReduction = c_config->pore_reduction;
    config.slimFace = c_config->slim_face;
    config.enlargeEyes = c_config->enlarge_eyes;
    config.thinChin = c_config->thin_chin;
    config.useGpu = c_config->use_gpu != 0;
    config.roiOnly = c_config->roi_only != 0;
    config.protectEyes = c_config->protect_eyes != 0;
    config.protectLips = c_config->protect_lips != 0;
    config.downscaleFactor = c_config->downscale_factor;
    config.vividIntensity = c_config->vivid_intensity;
    config.vividSaturation = c_config->vivid_saturation;
    config.vividBrightness = c_config->vivid_brightness;
    config.vividWarmth = c_config->vivid_warmth;
    config.protectNose = c_config->protect_nose != 0;

    // NaN/Inf/범위초과 방어 — 모든 C API 진입점에서 정규화
    iris_sdk::BeautyFilterConfigV2Helper::clamp(config);

    return config;
}

/**
 * @brief C++ 구조체를 C API 구조체로 변환
 */
void fromCppConfigV2(const BeautyFilterConfigV2& cpp_config, IrisBeautyConfigV2* c_config) {
    if (!c_config) return;

    c_config->enabled = cpp_config.enabled ? 1 : 0;
    c_config->intensity = cpp_config.intensity;
    c_config->smoothing = cpp_config.smoothing;
    c_config->brightness = cpp_config.brightness;
    c_config->soft_focus = cpp_config.softFocus;
    c_config->whitening = cpp_config.whitening;
    c_config->color_balance = cpp_config.colorBalance;
    c_config->wrinkle_remove = cpp_config.wrinkleRemove;
    c_config->skin_quality = cpp_config.skinQuality;
    c_config->smooth_intensity = cpp_config.smoothIntensity;
    c_config->pore_reduction = cpp_config.poreReduction;
    c_config->slim_face = cpp_config.slimFace;
    c_config->enlarge_eyes = cpp_config.enlargeEyes;
    c_config->thin_chin = cpp_config.thinChin;
    c_config->use_gpu = cpp_config.useGpu ? 1 : 0;
    c_config->roi_only = cpp_config.roiOnly ? 1 : 0;
    c_config->protect_eyes = cpp_config.protectEyes ? 1 : 0;
    c_config->protect_lips = cpp_config.protectLips ? 1 : 0;
    c_config->downscale_factor = cpp_config.downscaleFactor;
    c_config->vivid_intensity = cpp_config.vividIntensity;
    c_config->vivid_saturation = cpp_config.vividSaturation;
    c_config->vivid_brightness = cpp_config.vividBrightness;
    c_config->vivid_warmth = cpp_config.vividWarmth;
    c_config->protect_nose = cpp_config.protectNose ? 1 : 0;
}

/**
 * @brief CPU 백엔드 초기화 확인 및 초기화
 */
bool ensureCpuBackend() {
    std::lock_guard<std::mutex> lock(g_cpu_mutex);

    if (!g_cpu_beauty) {
        g_cpu_beauty = std::make_unique<iris_sdk::CPUBeautyBackend>();
        if (!g_cpu_beauty->initialize()) {
            g_cpu_beauty.reset();
            return false;
        }
    }

    return g_cpu_beauty && g_cpu_beauty->isInitialized();
}

} // anonymous namespace

// ============================================================================
// C API 구현
// ============================================================================

extern "C" {

void iris_sdk_default_beauty_config_v2_c(IrisBeautyConfigV2* config) {
    if (!config) return;

    std::memset(config, 0, sizeof(IrisBeautyConfigV2));

    config->enabled = 1;
    config->intensity = 0.5f;
    config->smoothing = 0.0f;
    config->brightness = 1.0f;
    config->soft_focus = 0.0f;
    config->whitening = 0.0f;
    config->color_balance = 0.0f;
    config->wrinkle_remove = 0.0f;
    config->skin_quality = 0.0f;
    config->smooth_intensity = 0.0f;
    config->pore_reduction = 0.0f;
    config->slim_face = 0.0f;
    config->enlarge_eyes = 0.0f;
    config->thin_chin = 0.0f;
    config->use_gpu = 1;
    config->roi_only = 1;
    config->protect_eyes = 1;
    config->protect_lips = 1;
    config->downscale_factor = 1;
    config->feather_radius = 15;
    config->vivid_intensity = 0.0f;
    config->vivid_saturation = 0.0f;
    config->vivid_brightness = 0.0f;
    config->vivid_warmth = 0.0f;
    config->protect_nose = 0;
}

IrisSdkError iris_sdk_apply_beauty_v2_c(
    uint8_t* frame_data,
    int width, int height,
    IrisFrameFormat format,
    const IrisBeautyConfigV2* config,
    const IrisResult* detection) {

    // 파라미터 검증
    if (!frame_data) {
        return IRIS_SDK_NULL_POINTER;
    }

    if (width <= 0 || height <= 0) {
        return IRIS_SDK_INVALID_PARAM;
    }

    if (!config) {
        return IRIS_SDK_INVALID_PARAM;
    }

    // 비활성화 상태면 패스스루
    if (!config->enabled) {
        return IRIS_SDK_OK;
    }

    // CPU 백엔드 사용
    if (!ensureCpuBackend()) {
        return IRIS_SDK_ERROR_NOT_INITIALIZED;
    }

    // C++ 설정으로 변환
    BeautyFilterConfigV2 cpp_config = toCppConfigV2(config);

    // ROI 생성 (detection이 있는 경우)
    iris_sdk::BeautyROI roi;

    if (detection && detection->detected && config->roi_only) {
        roi.face_rect = iris_sdk::computeExpandedFaceRect(
            detection->face_rect.x, detection->face_rect.y,
            detection->face_rect.width, detection->face_rect.height,
            width, height);
        roi.mask_width = static_cast<int>(roi.face_rect.width);
        roi.mask_height = static_cast<int>(roi.face_rect.height);
        roi.valid = true;
        roi.timestamp_ms = detection->timestamp_ms;

        // ROI 기반 마스크 생성 (Face Mesh 사용)
        if (detection->face_mesh_valid) {
            // C API IrisLandmark와 C++ iris_sdk::IrisLandmark는 동일한 레이아웃 (POD)
            const iris_sdk::IrisLandmark* face_mesh_ptr =
                reinterpret_cast<const iris_sdk::IrisLandmark*>(detection->face_mesh);
            iris_sdk::BeautyROIManager::computeROI(
                face_mesh_ptr,
                478,
                width, height,
                cpp_config,
                roi
            );
        }
    }

    // CPU 백엔드로 처리
    std::lock_guard<std::mutex> lock(g_cpu_mutex);
    return g_cpu_beauty->apply(
        frame_data,
        width, height,
        format,
        cpp_config,
        roi.valid ? &roi : nullptr
    );
}

IrisSdkError iris_sdk_init_gpu_beauty(void) {
#ifdef IRIS_SDK_HAS_GLES
    std::lock_guard<std::mutex> lock(g_gpu_mutex);

    if (g_gpu_beauty && g_gpu_beauty->isInitialized()) {
        return IRIS_SDK_OK;  // 이미 초기화됨
    }

    g_gpu_beauty = std::make_unique<iris_sdk::GPUBeautyBackend>();
    if (!g_gpu_beauty->initialize(nullptr)) {
        g_gpu_beauty.reset();
        return IRIS_SDK_ERROR_NOT_INITIALIZED;
    }

    return IRIS_SDK_OK;
#else
    return IRIS_SDK_ERROR_NOT_SUPPORTED;
#endif
}

void iris_sdk_release_gpu_beauty(void) {
#ifdef IRIS_SDK_HAS_GLES
    std::lock_guard<std::mutex> lock(g_gpu_mutex);

    // GPU 백엔드 해제 (TexturePool 경유 GL 리소스 해제)
    if (g_gpu_beauty) {
        g_gpu_beauty->release();
    }

    // 추적 set 정리 (GL 리소스는 이미 해제됨, 개별 glDeleteTextures 금지)
    g_managed_textures.clear();

    // 객체 소멸
    if (g_gpu_beauty) {
        g_gpu_beauty.reset();
    }
#endif
}

int iris_sdk_is_gpu_beauty_initialized(void) {
#ifdef IRIS_SDK_HAS_GLES
    std::lock_guard<std::mutex> lock(g_gpu_mutex);
    return (g_gpu_beauty && g_gpu_beauty->isInitialized()) ? 1 : 0;
#else
    return 0;
#endif
}

IrisSdkError iris_sdk_apply_beauty_texture_v2(
    uint32_t input_texture,
    uint32_t* output_texture,
    int width, int height,
    const IrisBeautyConfigV2* config,
    const IrisResult* detection,
    uint32_t lut_texture_id,
    float lut_intensity) {

#ifdef IRIS_SDK_HAS_GLES
    std::lock_guard<std::mutex> lock(g_gpu_mutex);

    if (!g_gpu_beauty || !g_gpu_beauty->isInitialized()) {
        return IRIS_SDK_ERROR_NOT_INITIALIZED;
    }

    if (!config || !output_texture) {
        return IRIS_SDK_INVALID_PARAM;
    }

    if (width <= 0 || height <= 0) {
        return IRIS_SDK_INVALID_PARAM;
    }

    // 비활성화 상태면 입력 텍스처 그대로 반환 (vivid는 enabled와 독립)
    bool needsVivid = config->vivid_intensity > 0.01f;
    if (!config->enabled && !needsVivid) {
        *output_texture = input_texture;
        return IRIS_SDK_OK;
    }

    // C++ 설정으로 변환
    BeautyFilterConfigV2 cpp_config = toCppConfigV2(config);

    // ROI 생성 (detection이 있는 경우)
    iris_sdk::BeautyROI roi;
    if (detection && detection->detected && config->roi_only) {
        roi.face_rect = iris_sdk::computeExpandedFaceRect(
            detection->face_rect.x, detection->face_rect.y,
            detection->face_rect.width, detection->face_rect.height,
            width, height);
        roi.mask_width = static_cast<int>(roi.face_rect.width);
        roi.mask_height = static_cast<int>(roi.face_rect.height);
        roi.valid = true;
        roi.timestamp_ms = detection->timestamp_ms;

        if (detection->face_mesh_valid) {
            const iris_sdk::IrisLandmark* face_mesh_ptr =
                reinterpret_cast<const iris_sdk::IrisLandmark*>(detection->face_mesh);
            iris_sdk::BeautyROIManager::computeROI(
                face_mesh_ptr, 478,
                width, height,
                cpp_config, roi
            );
        }
    }

    // GPU 뷰티 필터 적용 (텍스처 ID 기반)
    uint32_t result_texture = 0;
    IrisSdkError err = g_gpu_beauty->applyTextureId(
        input_texture,
        &result_texture,
        width, height,
        cpp_config,
        reinterpret_cast<const iris_sdk::IrisResult*>(detection),
        lut_texture_id,
        lut_intensity
    );

    if (err == IRIS_SDK_OK && result_texture != 0) {
        *output_texture = result_texture;
        // 관리 텍스처로 등록 (passthrough 시 입력 텍스처는 등록하지 않음)
        if (result_texture != input_texture) {
            g_managed_textures.insert(result_texture);
        }
    } else {
        *output_texture = input_texture;  // 실패 시 입력 텍스처 반환
    }

    return err;
#else
    if (output_texture) {
        *output_texture = input_texture;
    }
    return IRIS_SDK_ERROR_NOT_SUPPORTED;
#endif
}

void iris_sdk_set_freqsep_debug_mode(int mode) {
#ifdef IRIS_SDK_HAS_GLES
    std::lock_guard<std::mutex> lock(g_gpu_mutex);
    if (g_gpu_beauty && g_gpu_beauty->isInitialized()) {
        g_gpu_beauty->setFreqSepDebugMode(mode);
    }
#else
    (void)mode;
#endif
}

void iris_sdk_set_skin_color_filter(int enabled) {
#ifdef IRIS_SDK_HAS_GLES
    std::lock_guard<std::mutex> lock(g_gpu_mutex);
    if (g_gpu_beauty && g_gpu_beauty->isInitialized()) {
        g_gpu_beauty->setSkinColorFilter(enabled != 0);
    }
#else
    (void)enabled;
#endif
}

IrisSdkError iris_sdk_apply_face_warp(
    uint32_t input_texture,
    uint32_t* output_texture,
    int width, int height,
    float slim_face,
    float thin_chin,
    float enlarge_eyes,
    const IrisResult* detection) {

#ifdef IRIS_SDK_HAS_GLES
    std::lock_guard<std::mutex> lock(g_gpu_mutex);

    if (!output_texture) {
        return IRIS_SDK_INVALID_PARAM;
    }

    // 검출 결과 없거나 모든 값이 0이면 패스스루
    if (!detection || !detection->detected ||
        (slim_face <= 0.0f && thin_chin <= 0.0f && enlarge_eyes <= 0.0f)) {
        *output_texture = input_texture;
        return IRIS_SDK_OK;
    }

    if (!g_gpu_beauty || !g_gpu_beauty->isInitialized()) {
        *output_texture = input_texture;
        return IRIS_SDK_ERROR_NOT_INITIALIZED;
    }

    // Face Warp 적용
    uint32_t result_texture = 0;
    IrisSdkError err = g_gpu_beauty->applyFaceWarp(
        input_texture,
        &result_texture,
        width, height,
        slim_face,
        thin_chin,
        enlarge_eyes,
        reinterpret_cast<const iris_sdk::IrisResult*>(detection)
    );

    if (err == IRIS_SDK_OK && result_texture != 0) {
        *output_texture = result_texture;
        g_managed_textures.insert(result_texture);
    } else {
        *output_texture = input_texture;
    }

    return err;
#else
    if (output_texture) {
        *output_texture = input_texture;
    }
    return IRIS_SDK_ERROR_NOT_SUPPORTED;
#endif
}

IrisSdkError iris_sdk_release_texture(uint32_t texture) {
#ifdef IRIS_SDK_HAS_GLES
    std::lock_guard<std::mutex> lock(g_gpu_mutex);

    if (texture == 0) {
        return IRIS_SDK_INVALID_PARAM;
    }

    auto it = g_managed_textures.find(texture);
    if (it == g_managed_textures.end()) {
        // 관리 대상이 아닌 텍스처
        return IRIS_SDK_INVALID_PARAM;
    }

    // GPU 백엔드를 통해 텍스처 해제
    if (g_gpu_beauty) {
        g_gpu_beauty->releaseTexture(texture);
    }

    g_managed_textures.erase(it);
    return IRIS_SDK_OK;
#else
    return IRIS_SDK_ERROR_NOT_SUPPORTED;
#endif
}

int iris_sdk_is_texture_managed(uint32_t texture) {
#ifdef IRIS_SDK_HAS_GLES
    std::lock_guard<std::mutex> lock(g_gpu_mutex);
    return (g_managed_textures.count(texture) > 0) ? 1 : 0;
#else
    return 0;
#endif
}

// ============================================================================
// GPU 렌즈 렌더링 C API
// ============================================================================

IrisSdkError iris_sdk_init_gpu_lens(void) {
#ifdef IRIS_SDK_HAS_GLES
    std::lock_guard<std::mutex> lock(g_gpu_mutex);

    if (g_gpu_lens && g_gpu_lens->isInitialized()) {
        return IRIS_SDK_OK;
    }

    g_gpu_lens = std::make_unique<iris_sdk::GPULensRenderer>();
    if (!g_gpu_lens->initialize(nullptr)) {
        g_gpu_lens.reset();
        return IRIS_SDK_ERROR_NOT_INITIALIZED;
    }

    return IRIS_SDK_OK;
#else
    return IRIS_SDK_ERROR_NOT_SUPPORTED;
#endif
}

void iris_sdk_release_gpu_lens(void) {
#ifdef IRIS_SDK_HAS_GLES
    std::lock_guard<std::mutex> lock(g_gpu_mutex);
    if (g_gpu_lens) {
        g_gpu_lens->release();
        g_gpu_lens.reset();
    }
#endif
}

int iris_sdk_is_gpu_lens_initialized(void) {
#ifdef IRIS_SDK_HAS_GLES
    std::lock_guard<std::mutex> lock(g_gpu_mutex);
    return (g_gpu_lens && g_gpu_lens->isInitialized()) ? 1 : 0;
#else
    return 0;
#endif
}

IrisSdkError iris_sdk_load_lens_texture(const uint8_t* data, int width, int height) {
#ifdef IRIS_SDK_HAS_GLES
    std::lock_guard<std::mutex> lock(g_gpu_mutex);
    if (!g_gpu_lens || !g_gpu_lens->isInitialized()) {
        return IRIS_SDK_ERROR_NOT_INITIALIZED;
    }
    if (!data || width <= 0 || height <= 0) {
        return IRIS_SDK_INVALID_PARAM;
    }
    return g_gpu_lens->loadLensTexture(data, width, height) ? IRIS_SDK_OK : IRIS_SDK_RENDER_FAILED;
#else
    return IRIS_SDK_ERROR_NOT_SUPPORTED;
#endif
}

void iris_sdk_unload_lens_texture(void) {
#ifdef IRIS_SDK_HAS_GLES
    std::lock_guard<std::mutex> lock(g_gpu_mutex);
    if (g_gpu_lens) {
        g_gpu_lens->unloadLensTexture();
    }
#endif
}

IrisSdkError iris_sdk_render_lens_texture(
    uint32_t input_texture,
    uint32_t* output_texture,
    int width, int height,
    const IrisResult* detection,
    const IrisLensConfig* config)
{
#ifdef IRIS_SDK_HAS_GLES
    std::lock_guard<std::mutex> lock(g_gpu_mutex);

    if (!g_gpu_lens || !g_gpu_lens->isInitialized()) {
        return IRIS_SDK_ERROR_NOT_INITIALIZED;
    }
    if (!output_texture || !detection) {
        return IRIS_SDK_NULL_POINTER;
    }

    // C IrisResult → C++ iris_sdk::IrisResult 변환 (동일 POD 레이아웃)
    const iris_sdk::IrisResult* cpp_result =
        reinterpret_cast<const iris_sdk::IrisResult*>(detection);

    // C IrisLensConfig → C++ iris_sdk::LensConfig 변환
    iris_sdk::LensConfig cpp_config;
    if (config) {
        cpp_config.opacity = config->opacity;
        cpp_config.scale = config->scale;
        cpp_config.offset_x = config->offset_x;
        cpp_config.offset_y = config->offset_y;
        cpp_config.rotation = config->rotation;
        cpp_config.blend_mode = static_cast<iris_sdk::BlendMode>(config->blend_mode);
        cpp_config.edge_feather = config->edge_feather;
        cpp_config.apply_left = config->apply_left;
        cpp_config.apply_right = config->apply_right;
        cpp_config.is_mirror = config->is_mirror;
    }

    iris_sdk::ErrorCode err = g_gpu_lens->renderToTexture(
        input_texture, output_texture,
        width, height, *cpp_result, cpp_config);

    // C++ ErrorCode → C IrisSdkError 변환
    switch (err) {
        case iris_sdk::ErrorCode::Success: return IRIS_SDK_OK;
        case iris_sdk::ErrorCode::NotInitialized: return IRIS_SDK_ERROR_NOT_INITIALIZED;
        case iris_sdk::ErrorCode::NullPointer: return IRIS_SDK_NULL_POINTER;
        case iris_sdk::ErrorCode::NoTextureLoaded: return IRIS_SDK_NO_TEXTURE;
        default: return IRIS_SDK_RENDER_FAILED;
    }
#else
    return IRIS_SDK_ERROR_NOT_SUPPORTED;
#endif
}

void iris_sdk_set_lens_sclera_protect(int enabled) {
#ifdef IRIS_SDK_HAS_GLES
    std::lock_guard<std::mutex> lock(g_gpu_mutex);
    if (g_gpu_lens) {
        g_gpu_lens->setScleraProtectEnabled(enabled != 0);
    }
#else
    (void)enabled;
#endif
}

void iris_sdk_set_lens_ellipse_mask(int enabled) {
#ifdef IRIS_SDK_HAS_GLES
    std::lock_guard<std::mutex> lock(g_gpu_mutex);
    if (g_gpu_lens) {
        g_gpu_lens->setEllipseMaskEnabled(enabled != 0);
    }
#else
    (void)enabled;
#endif
}

void iris_sdk_set_lens_highlight(int enabled) {
#ifdef IRIS_SDK_HAS_GLES
    std::lock_guard<std::mutex> lock(g_gpu_mutex);
    if (g_gpu_lens) {
        g_gpu_lens->setHighlightEnabled(enabled != 0);
    }
#else
    (void)enabled;
#endif
}

} // extern "C"
