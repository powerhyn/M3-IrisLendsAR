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
#include "iris_sdk/internal/bench_toggles.h"  // W4-A §6.4: internal 벤치 토글 선언↔정의 시그니처 일치 강제
#include "iris_sdk/beauty_filter.h"
#include "iris_sdk/cpu_beauty_backend.h"
#include "iris_sdk/beauty_roi_manager.h"
#include "iris_sdk/lens_sku_metadata.h"

#ifdef IRIS_SDK_HAS_GLES
#include "iris_sdk/gpu/gpu_beauty_backend.h"
#include "iris_sdk/gpu/gpu_lens_renderer.h"
#include "iris_sdk/gpu/texture_handle.h"
#endif

#include <memory>
#include <mutex>
#include <set>
#include <cstring>
#include <cstddef>
#include <algorithm>

namespace {

// ============================================================================
// 전역 상태
// ============================================================================

std::mutex g_gpu_mutex;

// ③-2 B3 (감사 finding): sizeof 단일 가드는 '동일 크기를 유지하는 필드 순서 교환·타입
// 치환'을 잡지 못한다 — 필드별 offset 일치를 컴파일 타임에 강제한다. 한쪽 정의에만
// 필드를 추가/이동/치환하면 아래에서 즉시 실패한다 (types.h / sdk_api.h 수동 복제 규율의
// 기계 가드. 단일 정의 공유로의 통합은 ④ 표면 정리에서).
// GLES 가드 밖에 둔다 — macOS 데스크톱 빌드에서도 레이아웃 드리프트를 잡기 위함.
static_assert(sizeof(::IrisLandmark) == sizeof(iris_sdk::IrisLandmark),
              "C/C++ IrisLandmark layout must match for reinterpret_cast");
static_assert(offsetof(::IrisLandmark, z) == offsetof(iris_sdk::IrisLandmark, z) &&
              offsetof(::IrisLandmark, visibility) == offsetof(iris_sdk::IrisLandmark, visibility),
              "C/C++ IrisLandmark field offsets must match");
#define IRIS_SDK_ASSERT_RESULT_FIELD(f)                                              \
    static_assert(offsetof(::IrisResult, f) == offsetof(iris_sdk::IrisResult, f),    \
                  "C/C++ IrisResult field offset mismatch: " #f)
IRIS_SDK_ASSERT_RESULT_FIELD(detected);
IRIS_SDK_ASSERT_RESULT_FIELD(left_detected);
IRIS_SDK_ASSERT_RESULT_FIELD(right_detected);
IRIS_SDK_ASSERT_RESULT_FIELD(confidence);
IRIS_SDK_ASSERT_RESULT_FIELD(left_iris);
IRIS_SDK_ASSERT_RESULT_FIELD(left_radius);
IRIS_SDK_ASSERT_RESULT_FIELD(right_iris);
IRIS_SDK_ASSERT_RESULT_FIELD(right_radius);
IRIS_SDK_ASSERT_RESULT_FIELD(face_rect);
IRIS_SDK_ASSERT_RESULT_FIELD(face_rotation);
IRIS_SDK_ASSERT_RESULT_FIELD(face_mesh);
IRIS_SDK_ASSERT_RESULT_FIELD(face_mesh_valid);
IRIS_SDK_ASSERT_RESULT_FIELD(timestamp_ms);
IRIS_SDK_ASSERT_RESULT_FIELD(frame_width);
IRIS_SDK_ASSERT_RESULT_FIELD(frame_height);
// ④ W4-D: iris_quality_left/right, eye_refiner_used 필드 제거(detector 전용 메타).
IRIS_SDK_ASSERT_RESULT_FIELD(eyelid_ratio_left);
IRIS_SDK_ASSERT_RESULT_FIELD(eyelid_ratio_right);
IRIS_SDK_ASSERT_RESULT_FIELD(avg_iris_luma_left);
IRIS_SDK_ASSERT_RESULT_FIELD(avg_iris_luma_right);
#undef IRIS_SDK_ASSERT_RESULT_FIELD

// P7-W2: 이 파일은 C IrisResult ↔ C++ iris_sdk::IrisResult를 reinterpret_cast로
// 교환한다(렌더/ROI 경로, GLES 블록 내). 두 구조체 레이아웃이 어긋나면 UB.
// ④ W4-B2: sizeof 가드를 GLES 밖으로 이동 — 기존엔 #ifdef 안이라 non-GLES(데스크톱)
// 빌드에서 죽어, offsetof가 못 잡는 trailing-padding 드리프트가 미검출됐다. 레이아웃
// 일치는 컴파일타임 불변이라 위 offsetof 가드와 동일하게 양쪽 빌드 모두에서 검사한다.
static_assert(sizeof(::IrisResult) == sizeof(iris_sdk::IrisResult),
              "C/C++ IrisResult layout must match for reinterpret_cast (P7-W2 field add)");

#ifdef IRIS_SDK_HAS_GLES
std::unique_ptr<iris_sdk::GPUBeautyBackend> g_gpu_beauty;
std::unique_ptr<iris_sdk::GPULensRenderer> g_gpu_lens;
// P6-W7: SKU 메타 레지스트리. g_gpu_lens->setSkuRegistry()에 raw 포인터를 넘기므로
// 수명이 g_gpu_lens와 동일한 파일 스코프 전역으로 보관해야 dangling을 방지한다.
// (로컬 unique_ptr 금지 — g_gpu_mutex로 보호.)
std::unique_ptr<iris_sdk::LensSkuRegistry> g_sku_registry;
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
    config.brightness = c_config->brightness;
    config.slimFace = c_config->slim_face;
    config.enlargeEyes = c_config->enlarge_eyes;
    config.thinChin = c_config->thin_chin;
    config.useGpu = c_config->use_gpu != 0;
    config.roiOnly = c_config->roi_only != 0;
    config.protectEyes = c_config->protect_eyes != 0;
    config.protectLips = c_config->protect_lips != 0;
    config.downscaleFactor = c_config->downscale_factor;
    config.protectNose = c_config->protect_nose != 0;

    // NaN/Inf/범위초과 방어 — 모든 C API 진입점에서 정규화
    iris_sdk::BeautyFilterConfigV2Helper::clamp(config);

    return config;
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
    config->brightness = 1.0f;
    config->slim_face = 0.0f;
    config->enlarge_eyes = 0.0f;
    config->thin_chin = 0.0f;
    config->use_gpu = 1;
    config->roi_only = 1;
    config->protect_eyes = 1;
    config->protect_lips = 1;
    config->downscale_factor = 1;
    config->feather_radius = 15;
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
        return IRIS_SDK_NOT_INITIALIZED;
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
        return IRIS_SDK_NOT_INITIALIZED;
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
    const IrisResult* detection) {

#ifdef IRIS_SDK_HAS_GLES
    std::lock_guard<std::mutex> lock(g_gpu_mutex);

    if (!g_gpu_beauty || !g_gpu_beauty->isInitialized()) {
        return IRIS_SDK_NOT_INITIALIZED;
    }

    if (!config || !output_texture) {
        return IRIS_SDK_INVALID_PARAM;
    }

    if (width <= 0 || height <= 0) {
        return IRIS_SDK_INVALID_PARAM;
    }

    // 비활성화 상태면 입력 텍스처 그대로 반환
    if (!config->enabled) {
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
        reinterpret_cast<const iris_sdk::IrisResult*>(detection)
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

void iris_sdk_set_skin_mask_smoothing(int enabled, float strength) {
#ifdef IRIS_SDK_HAS_GLES
    std::lock_guard<std::mutex> lock(g_gpu_mutex);
    if (g_gpu_beauty && g_gpu_beauty->isInitialized()) {
        g_gpu_beauty->setSkinMaskSmoothing(enabled != 0, strength);
    }
#else
    (void)enabled;
    (void)strength;
#endif
}

void iris_sdk_set_skin_radiance(float strength) {
#ifdef IRIS_SDK_HAS_GLES
    std::lock_guard<std::mutex> lock(g_gpu_mutex);
    if (g_gpu_beauty && g_gpu_beauty->isInitialized()) {
        g_gpu_beauty->setSkinRadiance(strength);
    }
#else
    (void)strength;
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
        return IRIS_SDK_NOT_INITIALIZED;
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
        return IRIS_SDK_NOT_INITIALIZED;
    }

    // P6-W7: 메타가 init 이전에 등록된 경우 새 렌더러에 다시 연결.
    if (g_sku_registry) {
        g_gpu_lens->setSkuRegistry(g_sku_registry.get());
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
        return IRIS_SDK_NOT_INITIALIZED;
    }
    if (!data || width <= 0 || height <= 0) {
        return IRIS_SDK_INVALID_PARAM;
    }
    return g_gpu_lens->loadLensTexture(data, width, height) ? IRIS_SDK_OK : IRIS_SDK_RENDER_FAILED;
#else
    return IRIS_SDK_ERROR_NOT_SUPPORTED;
#endif
}

IrisSdkError iris_sdk_set_lens_metadata(const char* lens_meta_json) {
#ifdef IRIS_SDK_HAS_GLES
    if (!lens_meta_json) {
        return IRIS_SDK_NULL_POINTER;
    }

    std::lock_guard<std::mutex> lock(g_gpu_mutex);

    auto reg = std::make_unique<iris_sdk::LensSkuRegistry>();
    if (!reg->loadFromJson(lens_meta_json)) {
        return IRIS_SDK_INVALID_FORMAT;
    }

    g_sku_registry = std::move(reg);

    // 이미 GPU 렌즈가 초기화돼 있으면 즉시 연결. 미초기화 시에는
    // iris_sdk_init_gpu_lens()에서 g_sku_registry를 연결한다.
    if (g_gpu_lens) {
        g_gpu_lens->setSkuRegistry(g_sku_registry.get());
    }

    return IRIS_SDK_OK;
#else
    (void)lens_meta_json;
    return IRIS_SDK_ERROR_NOT_SUPPORTED;
#endif
}

IrisSdkError iris_sdk_load_lens_texture_with_sku(
    const uint8_t* data, int width, int height, const char* sku_id) {
#ifdef IRIS_SDK_HAS_GLES
    std::lock_guard<std::mutex> lock(g_gpu_mutex);
    if (!g_gpu_lens || !g_gpu_lens->isInitialized()) {
        return IRIS_SDK_NOT_INITIALIZED;
    }
    if (!data || width <= 0 || height <= 0) {
        return IRIS_SDK_INVALID_PARAM;
    }
    const std::string sku = sku_id ? std::string(sku_id) : std::string();
    return g_gpu_lens->loadLensTexture(data, width, height, sku) ? IRIS_SDK_OK : IRIS_SDK_RENDER_FAILED;
#else
    (void)data;
    (void)width;
    (void)height;
    (void)sku_id;
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
        return IRIS_SDK_NOT_INITIALIZED;
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
        // NotInitialized 정본은 IRIS_SDK_NOT_INITIALIZED=100 (굳은 ABI 계약). v1 convert_error_code와 동일 매핑. (W4-A 정정)
        case iris_sdk::ErrorCode::NotInitialized: return IRIS_SDK_NOT_INITIALIZED;
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

// P6-W5 §5.9: B1/B8 4조합 벤치용 sclera veto 수식 토글 internal C API.
// JNI 파일에서 forward declare 후 호출. 공개 sdk_api.h 미노출.
void iris_sdk_set_lens_sclera_veto_mode(int mode) {
#ifdef IRIS_SDK_HAS_GLES
    std::lock_guard<std::mutex> lock(g_gpu_mutex);
    if (g_gpu_lens) {
        g_gpu_lens->setScleraVetoMode(mode);
    }
#else
    (void)mode;
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

// EYECLIP A-2: 눈꺼풀 마스크 모드 토글 internal C API (0=Y-slab, 1=ellipse, 2=contour).
// JNI 파일에서 bench_toggles.h로 선언·호출. 공개 sdk_api.h 미노출.
void iris_sdk_set_lens_eyelid_mask_mode(int mode) {
#ifdef IRIS_SDK_HAS_GLES
    std::lock_guard<std::mutex> lock(g_gpu_mutex);
    if (g_gpu_lens) {
        g_gpu_lens->setEyelidMaskMode(mode);
    }
#else
    (void)mode;
#endif
}

// P5-W3-05 S1 D5: iris_sdk_set_lens_highlight는 no-op으로 축소.
// 고정 조명 하이라이트 기능 폐기. 공개 C API 호환성을 위해 심볼은 유지.
// C5 환경 반사 가산 계층(B2 결과 후)이 대체 역할 수행.
void iris_sdk_set_lens_highlight(int enabled) {
    (void)enabled;  // no-op
}

// ============================================================================
// P6-W4 §5.7/§5.11: 환경 반사 internal C API.
// 공개 sdk_api.h에는 노출하지 않음 (W4 Phase 벤치 전용 internal 경로).
// JNI 파일에서 forward declare 후 직접 호출. SDK 외부 surface 변화 없음.
// W5~W6 정식 활성 시점에 sdk_api.h로 승격 검토.
// ============================================================================

IRIS_SDK_EXPORT IrisSdkError iris_sdk_load_env_map(const uint8_t* data, int width, int height) {
#ifdef IRIS_SDK_HAS_GLES
    std::lock_guard<std::mutex> lock(g_gpu_mutex);
    if (!g_gpu_lens || !g_gpu_lens->isInitialized()) {
        return IRIS_SDK_NOT_INITIALIZED;
    }
    if (!data || width <= 0 || height <= 0) {
        return IRIS_SDK_INVALID_PARAM;
    }
    return g_gpu_lens->loadEnvMap(data, width, height) ? IRIS_SDK_OK : IRIS_SDK_RENDER_FAILED;
#else
    (void)data; (void)width; (void)height;
    return IRIS_SDK_ERROR_NOT_SUPPORTED;
#endif
}

IRIS_SDK_EXPORT void iris_sdk_unload_env_map(void) {
#ifdef IRIS_SDK_HAS_GLES
    std::lock_guard<std::mutex> lock(g_gpu_mutex);
    if (g_gpu_lens) {
        g_gpu_lens->unloadEnvMap();
    }
#endif
}

IRIS_SDK_EXPORT void iris_sdk_set_reflection_mode(int mode) {
#ifdef IRIS_SDK_HAS_GLES
    std::lock_guard<std::mutex> lock(g_gpu_mutex);
    if (g_gpu_lens) {
        g_gpu_lens->setReflectionMode(mode);
    }
#else
    (void)mode;
#endif
}

IRIS_SDK_EXPORT void iris_sdk_set_reflection_intensity(float intensity) {
#ifdef IRIS_SDK_HAS_GLES
    std::lock_guard<std::mutex> lock(g_gpu_mutex);
    if (g_gpu_lens) {
        g_gpu_lens->setReflectionIntensity(intensity);
    }
#else
    (void)intensity;
#endif
}

// ============================================================================
// P6-W6: 블링크 ramp(B5) / 저조도 gate(B9) / 디테일 재주입(C10) 벤치 토글 internal C API.
// JNI 파일에서 forward declare 후 호출. 공개 sdk_api.h 미노출 (벤치 전용 internal 경로).
// ============================================================================

void iris_sdk_set_lens_blink_up_ms(float ms) {
#ifdef IRIS_SDK_HAS_GLES
    std::lock_guard<std::mutex> lock(g_gpu_mutex);
    if (g_gpu_lens) {
        g_gpu_lens->setBlinkUpMs(ms);
    }
#else
    (void)ms;
#endif
}

void iris_sdk_set_lens_gate_threshold(float threshold) {
#ifdef IRIS_SDK_HAS_GLES
    std::lock_guard<std::mutex> lock(g_gpu_mutex);
    if (g_gpu_lens) {
        g_gpu_lens->setGateThreshold(threshold);
    }
#else
    (void)threshold;
#endif
}

void iris_sdk_set_lens_detail_reinject(int enabled) {
#ifdef IRIS_SDK_HAS_GLES
    std::lock_guard<std::mutex> lock(g_gpu_mutex);
    if (g_gpu_lens) {
        g_gpu_lens->setDetailReinject(enabled != 0);
    }
#else
    (void)enabled;
#endif
}

// P7-W2 §5.6: avg_iris_luma 실측↔fallback A/B 토글 internal C API.
// JNI 파일에서 forward declare 후 호출. 공개 sdk_api.h 미노출(detail_reinject와 동일 패턴).
void iris_sdk_set_use_measured_luma(int enabled) {
#ifdef IRIS_SDK_HAS_GLES
    std::lock_guard<std::mutex> lock(g_gpu_mutex);
    if (g_gpu_lens) {
        g_gpu_lens->setUseMeasuredLuma(enabled != 0);
    }
#else
    (void)enabled;
#endif
}

// P7-W4 §5.8: TintLinearV2 흰자 빛남 cap internal C API (유효 틴트 배율 상한).
// JNI 파일에서 forward declare 후 호출. 공개 sdk_api.h 미노출(gate_threshold와 동일 패턴).
void iris_sdk_set_lens_sclera_tint_max(float cap) {
#ifdef IRIS_SDK_HAS_GLES
    std::lock_guard<std::mutex> lock(g_gpu_mutex);
    if (g_gpu_lens) {
        g_gpu_lens->setScleraTintMax(cap);
    }
#else
    (void)cap;
#endif
}

// NLR-W2 R5: 흰자 페이드 시작점(홍채 반경 단위) 라이브 튜닝 internal C API.
// JNI 파일에서 forward declare 후 호출. 공개 sdk_api.h 미노출(sclera_tint_max와 동일 패턴).
void iris_sdk_set_lens_fade_start(float v) {
#ifdef IRIS_SDK_HAS_GLES
    std::lock_guard<std::mutex> lock(g_gpu_mutex);
    if (g_gpu_lens) {
        g_gpu_lens->setLensFadeStart(v);
    }
#else
    (void)v;
#endif
}

} // extern "C"
