/**
 * @file sdk_api.cpp
 * @brief IrisLensSDK C API 구현
 *
 * C++ SDK 핵심 기능을 C API로 래핑하여 다양한 바인딩 레이어에서 사용할 수 있도록 합니다.
 *
 * @author IrisLensSDK Team
 * @version 1.0.0
 */

#include "iris_sdk/sdk_api.h"
#include "iris_sdk/sdk_manager.h"
#include "iris_sdk/frame_processor.h"
#include "iris_sdk/temporal_stabilizer.h"
#include "iris_sdk/types.h"

#include <cstring>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

// Android 로그 매크로
#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "IrisSDK-API"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#define LOGI(...) ((void)0)
#define LOGW(...) ((void)0)
#define LOGE(...) ((void)0)
#endif

// ============================================================================
// 내부 전역 변수 (익명 네임스페이스)
// ============================================================================

namespace {

/// 전역 FrameProcessor 인스턴스 (싱글 스레드 사용 기준)
std::unique_ptr<iris_sdk::FrameProcessor> g_processor;

/// 마지막 에러 메시지 버퍼
thread_local char g_last_error[512] = {0};

/// 전역 상태 뮤텍스
std::mutex g_mutex;

/// GPU 가속 요청 플래그 (init 전에 설정)
bool g_gpu_request = false;

/// InferenceThread 사용 여부 요청 플래그 (init 전에 설정)
bool g_use_inference_thread_request = true;  // 기본값: InferenceThread 사용

/// Temporal Stabilizer 인스턴스 관리
std::unordered_map<int64_t, std::unique_ptr<iris_sdk::TemporalStabilizer>> g_stabilizers;
int64_t g_next_stabilizer_handle = 1;

/**
 * @brief 마지막 에러 메시지 설정
 */
void set_last_error(const char* message) {
    if (message) {
        std::strncpy(g_last_error, message, sizeof(g_last_error) - 1);
        g_last_error[sizeof(g_last_error) - 1] = '\0';
    } else {
        g_last_error[0] = '\0';
    }
}

/**
 * @brief C++ ErrorCode를 C IrisSdkError로 변환
 */
IrisSdkError convert_error_code(iris_sdk::ErrorCode code) {
    switch (code) {
        case iris_sdk::ErrorCode::Success:
            return IRIS_SDK_OK;
        case iris_sdk::ErrorCode::NotInitialized:
            return IRIS_SDK_NOT_INITIALIZED;
        case iris_sdk::ErrorCode::AlreadyInitialized:
            return IRIS_SDK_ALREADY_INITIALIZED;
        case iris_sdk::ErrorCode::ModelLoadFailed:
            return IRIS_SDK_MODEL_LOAD_FAILED;
        case iris_sdk::ErrorCode::InvalidPath:
            return IRIS_SDK_INVALID_PATH;
        case iris_sdk::ErrorCode::InvalidParameter:
            return IRIS_SDK_INVALID_PARAM;
        case iris_sdk::ErrorCode::NullPointer:
            return IRIS_SDK_NULL_POINTER;
        case iris_sdk::ErrorCode::FrameFormatUnsupported:
            return IRIS_SDK_INVALID_FORMAT;
        case iris_sdk::ErrorCode::DetectionFailed:
            return IRIS_SDK_DETECTION_FAILED;
        case iris_sdk::ErrorCode::NoFaceDetected:
            return IRIS_SDK_NO_FACE;
        case iris_sdk::ErrorCode::RenderFailed:
            return IRIS_SDK_RENDER_FAILED;
        case iris_sdk::ErrorCode::NoTextureLoaded:
            return IRIS_SDK_NO_TEXTURE;
        default:
            return IRIS_SDK_UNKNOWN;
    }
}

/**
 * @brief C IrisFrameFormat을 C++ FrameFormat으로 변환
 */
iris_sdk::FrameFormat convert_frame_format(IrisFrameFormat format) {
    switch (format) {
        case IRIS_FORMAT_RGBA:
            return iris_sdk::FrameFormat::RGBA;
        case IRIS_FORMAT_BGRA:
            return iris_sdk::FrameFormat::BGRA;
        case IRIS_FORMAT_RGB:
            return iris_sdk::FrameFormat::RGB;
        case IRIS_FORMAT_BGR:
            return iris_sdk::FrameFormat::BGR;
        case IRIS_FORMAT_NV21:
            return iris_sdk::FrameFormat::NV21;
        case IRIS_FORMAT_NV12:
            return iris_sdk::FrameFormat::NV12;
        case IRIS_FORMAT_GRAY:
            return iris_sdk::FrameFormat::Grayscale;
        default:
            return iris_sdk::FrameFormat::RGBA;
    }
}

/**
 * @brief C IrisBlendMode를 C++ BlendMode로 변환
 */
iris_sdk::BlendMode convert_blend_mode(IrisBlendMode mode) {
    switch (mode) {
        case IRIS_BLEND_NORMAL:
            return iris_sdk::BlendMode::Normal;
        case IRIS_BLEND_MULTIPLY:
            return iris_sdk::BlendMode::Multiply;
        case IRIS_BLEND_SCREEN:
            return iris_sdk::BlendMode::Screen;
        case IRIS_BLEND_OVERLAY:
            return iris_sdk::BlendMode::Overlay;
        case IRIS_BLEND_LUMINANCE_TINT:
            return iris_sdk::BlendMode::LuminanceTint;
        case IRIS_BLEND_LUMINANCE_TINT_LINEAR:
            return iris_sdk::BlendMode::LuminanceTintLinear;
        case IRIS_BLEND_SOFT_LIGHT:
            return iris_sdk::BlendMode::SoftLight;
        case IRIS_BLEND_COLOR_REPLACE:
            return iris_sdk::BlendMode::ColorReplace;
        default:
            // P6-W2 §5.9: invalid blend ID는 TintLinearV2(ID=5)로 fallback.
            // 셰이더 측 §5.9 경고와 정합 (외부 API path도 동일 fallback 보장).
            LOGW("[IrisSDK] Unknown blend mode: %d, falling back to LuminanceTintLinear (ID=5)",
                 static_cast<int>(mode));
            return iris_sdk::BlendMode::LuminanceTintLinear;
    }
}

/**
 * @brief C IrisLensConfig를 C++ LensConfig로 변환
 */
iris_sdk::LensConfig convert_to_cpp_lens_config(const IrisLensConfig* c_config) {
    iris_sdk::LensConfig cpp_config;
    cpp_config.opacity = c_config->opacity;
    cpp_config.scale = c_config->scale;
    cpp_config.offset_x = c_config->offset_x;
    cpp_config.offset_y = c_config->offset_y;
    cpp_config.rotation = c_config->rotation;
    cpp_config.blend_mode = convert_blend_mode(c_config->blend_mode);
    cpp_config.edge_feather = c_config->edge_feather;
    cpp_config.apply_left = c_config->apply_left;
    cpp_config.apply_right = c_config->apply_right;
    return cpp_config;
}

/**
 * @brief C++ IrisLandmark를 C IrisLandmark로 복사
 */
void copy_landmark(const iris_sdk::IrisLandmark& src, IrisLandmark& dst) {
    dst.x = src.x;
    dst.y = src.y;
    dst.z = src.z;
    dst.visibility = src.visibility;
}

/**
 * @brief C++ IrisResult를 C IrisResult로 변환
 */
void convert_to_c_iris_result(const iris_sdk::IrisResult& cpp_result, IrisResult* c_result) {
    // 검출 상태
    c_result->detected = cpp_result.detected;
    c_result->left_detected = cpp_result.left_detected;
    c_result->right_detected = cpp_result.right_detected;
    c_result->confidence = cpp_result.confidence;

    // 왼쪽 홍채
    for (int i = 0; i < 5; ++i) {
        copy_landmark(cpp_result.left_iris[i], c_result->left_iris[i]);
    }
    c_result->left_radius = cpp_result.left_radius;

    // 오른쪽 홍채
    for (int i = 0; i < 5; ++i) {
        copy_landmark(cpp_result.right_iris[i], c_result->right_iris[i]);
    }
    c_result->right_radius = cpp_result.right_radius;

    // 얼굴 메타데이터
    c_result->face_rect.x = cpp_result.face_rect.x;
    c_result->face_rect.y = cpp_result.face_rect.y;
    c_result->face_rect.width = cpp_result.face_rect.width;
    c_result->face_rect.height = cpp_result.face_rect.height;

    c_result->face_rotation[0] = cpp_result.face_rotation[0];
    c_result->face_rotation[1] = cpp_result.face_rotation[1];
    c_result->face_rotation[2] = cpp_result.face_rotation[2];

    // Face Mesh
    c_result->face_mesh_valid = cpp_result.face_mesh_valid;
    if (cpp_result.face_mesh_valid) {
        for (int i = 0; i < 478; ++i) {
            copy_landmark(cpp_result.face_mesh[i], c_result->face_mesh[i]);
        }
    }

    // 프레임 정보
    c_result->timestamp_ms = cpp_result.timestamp_ms;
    c_result->frame_width = cpp_result.frame_width;
    c_result->frame_height = cpp_result.frame_height;

    // Eye Refiner 메타데이터
    c_result->iris_quality_left = cpp_result.iris_quality_left;
    c_result->iris_quality_right = cpp_result.iris_quality_right;
    c_result->eyelid_ratio_left = cpp_result.eyelid_ratio_left;
    c_result->eyelid_ratio_right = cpp_result.eyelid_ratio_right;
    c_result->eye_refiner_used = cpp_result.eye_refiner_used;

    // P7-W2: iris ROI 실측 luma 전달 (없으면 -1 sentinel).
    c_result->avg_iris_luma_left = cpp_result.avg_iris_luma_left;
    c_result->avg_iris_luma_right = cpp_result.avg_iris_luma_right;
}

/**
 * @brief C IrisResult를 C++ IrisResult로 변환
 */
iris_sdk::IrisResult convert_to_cpp_iris_result(const IrisResult* c_result) {
    iris_sdk::IrisResult cpp_result;

    // 검출 상태
    cpp_result.detected = c_result->detected;
    cpp_result.left_detected = c_result->left_detected;
    cpp_result.right_detected = c_result->right_detected;
    cpp_result.confidence = c_result->confidence;

    // 왼쪽 홍채
    for (int i = 0; i < 5; ++i) {
        cpp_result.left_iris[i].x = c_result->left_iris[i].x;
        cpp_result.left_iris[i].y = c_result->left_iris[i].y;
        cpp_result.left_iris[i].z = c_result->left_iris[i].z;
        cpp_result.left_iris[i].visibility = c_result->left_iris[i].visibility;
    }
    cpp_result.left_radius = c_result->left_radius;

    // 오른쪽 홍채
    for (int i = 0; i < 5; ++i) {
        cpp_result.right_iris[i].x = c_result->right_iris[i].x;
        cpp_result.right_iris[i].y = c_result->right_iris[i].y;
        cpp_result.right_iris[i].z = c_result->right_iris[i].z;
        cpp_result.right_iris[i].visibility = c_result->right_iris[i].visibility;
    }
    cpp_result.right_radius = c_result->right_radius;

    // 얼굴 메타데이터
    cpp_result.face_rect.x = c_result->face_rect.x;
    cpp_result.face_rect.y = c_result->face_rect.y;
    cpp_result.face_rect.width = c_result->face_rect.width;
    cpp_result.face_rect.height = c_result->face_rect.height;

    cpp_result.face_rotation[0] = c_result->face_rotation[0];
    cpp_result.face_rotation[1] = c_result->face_rotation[1];
    cpp_result.face_rotation[2] = c_result->face_rotation[2];

    // Face Mesh
    cpp_result.face_mesh_valid = c_result->face_mesh_valid;
    if (c_result->face_mesh_valid) {
        for (int i = 0; i < 478; ++i) {
            cpp_result.face_mesh[i].x = c_result->face_mesh[i].x;
            cpp_result.face_mesh[i].y = c_result->face_mesh[i].y;
            cpp_result.face_mesh[i].z = c_result->face_mesh[i].z;
            cpp_result.face_mesh[i].visibility = c_result->face_mesh[i].visibility;
        }
    }

    // 프레임 정보
    cpp_result.timestamp_ms = c_result->timestamp_ms;
    cpp_result.frame_width = c_result->frame_width;
    cpp_result.frame_height = c_result->frame_height;

    // Eye Refiner 메타데이터
    cpp_result.iris_quality_left = c_result->iris_quality_left;
    cpp_result.iris_quality_right = c_result->iris_quality_right;
    cpp_result.eyelid_ratio_left = c_result->eyelid_ratio_left;
    cpp_result.eyelid_ratio_right = c_result->eyelid_ratio_right;
    cpp_result.eye_refiner_used = c_result->eye_refiner_used;

    // P7-W2: iris ROI 실측 luma 전달 (없으면 -1 sentinel).
    cpp_result.avg_iris_luma_left = c_result->avg_iris_luma_left;
    cpp_result.avg_iris_luma_right = c_result->avg_iris_luma_right;

    return cpp_result;
}

}  // anonymous namespace

// ============================================================================
// 라이프사이클 함수 구현
// ============================================================================

extern "C" {

IrisSdkError iris_sdk_init(const char* model_path) {
    if (!model_path) {
        set_last_error("model_path is null");
        return IRIS_SDK_NULL_POINTER;
    }

    std::lock_guard<std::mutex> lock(g_mutex);

    auto& manager = iris_sdk::SDKManager::getInstance();

    if (manager.isReady()) {
        set_last_error("SDK is already initialized");
        return IRIS_SDK_ALREADY_INITIALIZED;
    }

    if (!manager.initialize(std::string(model_path))) {
        set_last_error("Failed to initialize SDK with model path");
        return IRIS_SDK_MODEL_LOAD_FAILED;
    }

    // FrameProcessor 직접 생성 (createFrameProcessor()는 내부에서 initialize()를 호출하므로 사용 안함)
    g_processor = std::make_unique<iris_sdk::FrameProcessor>();
    if (!g_processor) {
        manager.shutdown();
        set_last_error("Failed to create FrameProcessor - internal error");
        return IRIS_SDK_UNKNOWN;
    }

    // InferenceThread 사용 여부 설정 적용 (init 전에 setUseInferenceThread() 호출한 경우)
    LOGI("iris_sdk_init: applying g_use_inference_thread_request=%s",
         g_use_inference_thread_request ? "true" : "false");
    g_processor->setUseInferenceThread(g_use_inference_thread_request);

    // GPU 가속 설정 적용 (init 전에 setGpuEnabled() 호출한 경우)
    if (g_gpu_request) {
        g_processor->setGpuEnabled(true);
    }

    LOGI("iris_sdk_init: calling g_processor->initialize() with useInferenceThread=%s",
         g_processor->isUsingInferenceThread() ? "true" : "false");

    if (!g_processor->initialize(model_path)) {
        g_processor.reset();
        manager.shutdown();
        set_last_error("Failed to initialize FrameProcessor");
        return IRIS_SDK_MODEL_LOAD_FAILED;
    }

    set_last_error(nullptr);
    return IRIS_SDK_OK;
}

IrisSdkError iris_sdk_init_with_config(const IrisSdkConfig* config) {
    if (!config) {
        set_last_error("config is null");
        return IRIS_SDK_NULL_POINTER;
    }

    if (!config->model_path) {
        set_last_error("config->model_path is null");
        return IRIS_SDK_NULL_POINTER;
    }

    std::lock_guard<std::mutex> lock(g_mutex);

    auto& manager = iris_sdk::SDKManager::getInstance();

    if (manager.isReady()) {
        set_last_error("SDK is already initialized");
        return IRIS_SDK_ALREADY_INITIALIZED;
    }

    // SDKConfig 변환
    iris_sdk::SDKConfig cpp_config;
    cpp_config.model_path = config->model_path;
    cpp_config.min_detection_confidence = config->min_confidence > 0.0f ? config->min_confidence : 0.5f;
    cpp_config.max_faces = config->max_faces > 0 ? config->max_faces : 1;
    cpp_config.enable_gpu = config->enable_gpu;
    cpp_config.num_threads = config->num_threads;

    if (!manager.initialize(cpp_config)) {
        set_last_error("Failed to initialize SDK with config");
        return IRIS_SDK_MODEL_LOAD_FAILED;
    }

    // FrameProcessor 직접 생성 (createFrameProcessor()는 내부에서 initialize()를 호출하므로 사용 안함)
    g_processor = std::make_unique<iris_sdk::FrameProcessor>();
    if (!g_processor) {
        manager.shutdown();
        set_last_error("Failed to create FrameProcessor - internal error");
        return IRIS_SDK_UNKNOWN;
    }

    // InferenceThread 사용 여부 설정 적용
    LOGI("iris_sdk_init_with_config: applying g_use_inference_thread_request=%s",
         g_use_inference_thread_request ? "true" : "false");
    g_processor->setUseInferenceThread(g_use_inference_thread_request);

    // GPU 가속 설정 적용
    if (g_gpu_request) {
        g_processor->setGpuEnabled(true);
    }

    LOGI("iris_sdk_init_with_config: calling g_processor->initialize() with useInferenceThread=%s",
         g_processor->isUsingInferenceThread() ? "true" : "false");

    if (!g_processor->initialize(config->model_path)) {
        g_processor.reset();
        manager.shutdown();
        set_last_error("Failed to initialize FrameProcessor");
        return IRIS_SDK_MODEL_LOAD_FAILED;
    }

    // 신뢰도 설정
    if (config->min_confidence > 0.0f) {
        g_processor->setMinConfidence(config->min_confidence);
    }

    set_last_error(nullptr);
    return IRIS_SDK_OK;
}

void iris_sdk_destroy(void) {
    std::lock_guard<std::mutex> lock(g_mutex);

    g_processor.reset();
    iris_sdk::SDKManager::getInstance().shutdown();
    set_last_error(nullptr);
}

bool iris_sdk_is_ready(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return iris_sdk::SDKManager::getInstance().isReady() && g_processor && g_processor->isInitialized();
}

// ============================================================================
// 검출 함수 구현
// ============================================================================

IrisSdkError iris_sdk_detect(
    const uint8_t* frame_data,
    int width,
    int height,
    IrisFrameFormat format,
    IrisResult* result) {

    if (!frame_data) {
        set_last_error("frame_data is null");
        return IRIS_SDK_NULL_POINTER;
    }

    if (!result) {
        set_last_error("result is null");
        return IRIS_SDK_NULL_POINTER;
    }

    if (width <= 0 || height <= 0) {
        set_last_error("Invalid frame dimensions");
        return IRIS_SDK_INVALID_PARAM;
    }

    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_processor || !g_processor->isInitialized()) {
        set_last_error("SDK not initialized");
        return IRIS_SDK_NOT_INITIALIZED;
    }

    iris_sdk::FrameFormat cpp_format = convert_frame_format(format);
    iris_sdk::IrisResult cpp_result = g_processor->detectOnly(frame_data, width, height, cpp_format);

    convert_to_c_iris_result(cpp_result, result);

    if (!cpp_result.detected) {
        // 검출은 성공했지만 얼굴이 없음
        set_last_error(nullptr);
        return IRIS_SDK_OK;
    }

    set_last_error(nullptr);
    return IRIS_SDK_OK;
}

IrisSdkError iris_sdk_detect_with_rotation(
    const uint8_t* frame_data,
    int width,
    int height,
    IrisFrameFormat format,
    int rotation_degrees,
    IrisResult* result) {

    if (!frame_data) {
        set_last_error("frame_data is null");
        return IRIS_SDK_NULL_POINTER;
    }

    if (!result) {
        set_last_error("result is null");
        return IRIS_SDK_NULL_POINTER;
    }

    if (width <= 0 || height <= 0) {
        set_last_error("Invalid frame dimensions");
        return IRIS_SDK_INVALID_PARAM;
    }

    // 회전 각도 정규화 (0, 90, 180, 270)
    int normalized_rotation = ((rotation_degrees % 360) + 360) % 360;
    if (normalized_rotation != 0 && normalized_rotation != 90 &&
        normalized_rotation != 180 && normalized_rotation != 270) {
        set_last_error("Invalid rotation degrees (must be 0, 90, 180, or 270)");
        return IRIS_SDK_INVALID_PARAM;
    }

    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_processor || !g_processor->isInitialized()) {
        set_last_error("SDK not initialized");
        return IRIS_SDK_NOT_INITIALIZED;
    }

    iris_sdk::FrameFormat cpp_format = convert_frame_format(format);

    // 회전이 필요 없는 경우 기존 로직 사용
    if (normalized_rotation == 0) {
        iris_sdk::IrisResult cpp_result = g_processor->detectOnly(frame_data, width, height, cpp_format);
        convert_to_c_iris_result(cpp_result, result);
        set_last_error(nullptr);
        return IRIS_SDK_OK;
    }

    // 회전이 필요한 경우: 회전 파라미터를 포함하여 검출
    iris_sdk::IrisResult cpp_result = g_processor->detectOnlyWithRotation(
        frame_data, width, height, cpp_format, normalized_rotation);

    convert_to_c_iris_result(cpp_result, result);

    set_last_error(nullptr);
    return IRIS_SDK_OK;
}

// ============================================================================
// 처리 함수 구현
// ============================================================================

IrisSdkError iris_sdk_process(
    uint8_t* frame_data,
    int width,
    int height,
    IrisFrameFormat format,
    const IrisLensConfig* config,
    IrisResult* result) {

    if (!frame_data) {
        set_last_error("frame_data is null");
        return IRIS_SDK_NULL_POINTER;
    }

    if (width <= 0 || height <= 0) {
        set_last_error("Invalid frame dimensions");
        return IRIS_SDK_INVALID_PARAM;
    }

    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_processor || !g_processor->isInitialized()) {
        set_last_error("SDK not initialized");
        return IRIS_SDK_NOT_INITIALIZED;
    }

    iris_sdk::FrameFormat cpp_format = convert_frame_format(format);

    // LensConfig 변환 (nullptr이면 검출만 수행)
    const iris_sdk::LensConfig* cpp_config_ptr = nullptr;
    iris_sdk::LensConfig cpp_config;
    if (config) {
        cpp_config = convert_to_cpp_lens_config(config);
        cpp_config_ptr = &cpp_config;
    }

    iris_sdk::ProcessResult process_result = g_processor->process(
        frame_data, width, height, cpp_format, cpp_config_ptr);

    if (result) {
        convert_to_c_iris_result(process_result.iris_result, result);
    }

    if (!process_result.success) {
        set_last_error("Processing failed");
        return convert_error_code(process_result.error_code);
    }

    set_last_error(nullptr);
    return IRIS_SDK_OK;
}

// ============================================================================
// 렌더링 함수 구현
// ============================================================================

IrisSdkError iris_sdk_load_texture(const char* path) {
    if (!path) {
        set_last_error("path is null");
        return IRIS_SDK_NULL_POINTER;
    }

    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_processor || !g_processor->isInitialized()) {
        set_last_error("SDK not initialized");
        return IRIS_SDK_NOT_INITIALIZED;
    }

    if (!g_processor->loadLensTexture(std::string(path))) {
        set_last_error("Failed to load texture from file");
        return IRIS_SDK_INVALID_PATH;
    }

    set_last_error(nullptr);
    return IRIS_SDK_OK;
}

IrisSdkError iris_sdk_load_texture_from_memory(
    const uint8_t* data,
    int width,
    int height) {

    if (!data) {
        set_last_error("data is null");
        return IRIS_SDK_NULL_POINTER;
    }

    if (width <= 0 || height <= 0) {
        set_last_error("Invalid texture dimensions");
        return IRIS_SDK_INVALID_PARAM;
    }

    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_processor || !g_processor->isInitialized()) {
        set_last_error("SDK not initialized");
        return IRIS_SDK_NOT_INITIALIZED;
    }

    if (!g_processor->loadLensTexture(data, width, height)) {
        set_last_error("Failed to load texture from memory");
        return IRIS_SDK_INVALID_PARAM;
    }

    set_last_error(nullptr);
    return IRIS_SDK_OK;
}

IrisSdkError iris_sdk_render_lens(
    uint8_t* frame_data,
    int width,
    int height,
    IrisFrameFormat format,
    const IrisResult* iris_result,
    const IrisLensConfig* config) {

    if (!frame_data) {
        set_last_error("frame_data is null");
        return IRIS_SDK_NULL_POINTER;
    }

    if (!iris_result) {
        set_last_error("iris_result is null");
        return IRIS_SDK_NULL_POINTER;
    }

    if (!config) {
        set_last_error("config is null");
        return IRIS_SDK_NULL_POINTER;
    }

    if (width <= 0 || height <= 0) {
        set_last_error("Invalid frame dimensions");
        return IRIS_SDK_INVALID_PARAM;
    }

    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_processor || !g_processor->isInitialized()) {
        set_last_error("SDK not initialized");
        return IRIS_SDK_NOT_INITIALIZED;
    }

    if (!g_processor->hasLensTexture()) {
        set_last_error("No texture loaded");
        return IRIS_SDK_NO_TEXTURE;
    }

    iris_sdk::FrameFormat cpp_format = convert_frame_format(format);
    iris_sdk::IrisResult cpp_iris_result = convert_to_cpp_iris_result(iris_result);
    iris_sdk::LensConfig cpp_config = convert_to_cpp_lens_config(config);

    if (!g_processor->renderOnly(frame_data, width, height, cpp_format, cpp_iris_result, cpp_config)) {
        set_last_error("Rendering failed");
        return IRIS_SDK_RENDER_FAILED;
    }

    set_last_error(nullptr);
    return IRIS_SDK_OK;
}

// ============================================================================
// 설정 함수 구현
// ============================================================================

IrisSdkError iris_sdk_set_config(const char* key, const char* value) {
    if (!key || !value) {
        set_last_error("key or value is null");
        return IRIS_SDK_NULL_POINTER;
    }

    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_processor || !g_processor->isInitialized()) {
        set_last_error("SDK not initialized");
        return IRIS_SDK_NOT_INITIALIZED;
    }

    std::string key_str(key);
    std::string value_str(value);

    if (key_str == "min_confidence") {
        try {
            float confidence = std::stof(value_str);
            if (confidence < 0.0f || confidence > 1.0f) {
                set_last_error("min_confidence must be between 0.0 and 1.0");
                return IRIS_SDK_INVALID_PARAM;
            }
            g_processor->setMinConfidence(confidence);
        } catch (const std::invalid_argument&) {
            set_last_error("Invalid confidence value format");
            return IRIS_SDK_INVALID_PARAM;
        } catch (const std::out_of_range&) {
            set_last_error("Confidence value out of range");
            return IRIS_SDK_INVALID_PARAM;
        }
    } else if (key_str == "face_tracking") {
        bool enable = (value_str == "true" || value_str == "1" || value_str == "yes");
        g_processor->setFaceTracking(enable);
    } else {
        set_last_error("Unknown config key");
        return IRIS_SDK_INVALID_PARAM;
    }

    set_last_error(nullptr);
    return IRIS_SDK_OK;
}

void iris_sdk_default_lens_config(IrisLensConfig* config) {
    if (!config) {
        return;
    }

    config->opacity = 0.7f;
    config->scale = 1.0f;
    config->offset_x = 0.0f;
    config->offset_y = 0.0f;
    config->rotation = 0.0f;
    config->blend_mode = IRIS_BLEND_LUMINANCE_TINT_LINEAR;  // P6-W2 §5.12 canonical default
    config->edge_feather = 0.1f;
    config->apply_left = true;
    config->apply_right = true;
    config->is_mirror = false;
}

// ============================================================================
// 정보 함수 구현
// ============================================================================

const char* iris_sdk_get_version(void) {
    return iris_sdk::SDKManager::getVersion();
}

const char* iris_sdk_get_build_info(void) {
    return iris_sdk::SDKManager::getBuildInfo();
}

const char* iris_sdk_get_last_error(void) {
    return g_last_error;
}

const char* iris_sdk_error_to_string(IrisSdkError error) {
    switch (error) {
        case IRIS_SDK_OK:
            return "IRIS_SDK_OK";
        case IRIS_SDK_NOT_INITIALIZED:
            return "IRIS_SDK_NOT_INITIALIZED";
        case IRIS_SDK_ALREADY_INITIALIZED:
            return "IRIS_SDK_ALREADY_INITIALIZED";
        case IRIS_SDK_MODEL_LOAD_FAILED:
            return "IRIS_SDK_MODEL_LOAD_FAILED";
        case IRIS_SDK_INVALID_PATH:
            return "IRIS_SDK_INVALID_PATH";
        case IRIS_SDK_INVALID_PARAM:
            return "IRIS_SDK_INVALID_PARAM";
        case IRIS_SDK_NULL_POINTER:
            return "IRIS_SDK_NULL_POINTER";
        case IRIS_SDK_INVALID_FORMAT:
            return "IRIS_SDK_INVALID_FORMAT";
        case IRIS_SDK_DETECTION_FAILED:
            return "IRIS_SDK_DETECTION_FAILED";
        case IRIS_SDK_NO_FACE:
            return "IRIS_SDK_NO_FACE";
        case IRIS_SDK_RENDER_FAILED:
            return "IRIS_SDK_RENDER_FAILED";
        case IRIS_SDK_NO_TEXTURE:
            return "IRIS_SDK_NO_TEXTURE";
        case IRIS_SDK_UNKNOWN:
        default:
            return "IRIS_SDK_UNKNOWN";
    }
}

// ============================================================================
// 메모리 관리 함수 구현
// ============================================================================

void iris_sdk_free_result(IrisResult* result) {
    if (!result) {
        return;
    }

    // 현재 IrisResult는 순수 POD 타입이므로 동적 할당이 없습니다.
    // 구조체를 0으로 초기화합니다.
    std::memset(result, 0, sizeof(IrisResult));
}

// ============================================================================
// GPU 가속 함수 구현
// ============================================================================

void iris_sdk_set_gpu_enabled(bool enable) {
    std::lock_guard<std::mutex> lock(g_mutex);

    // 이미 초기화된 경우 경고
    if (g_processor && g_processor->isInitialized()) {
        set_last_error("setGpuEnabled() must be called before init()");
        return;
    }

    g_gpu_request = enable;

    // FrameProcessor가 이미 생성되어 있으면 설정 전달
    if (g_processor) {
        g_processor->setGpuEnabled(enable);
    }
}

bool iris_sdk_is_gpu_available(void) {
#if defined(IRIS_SDK_HAS_GPU_DELEGATE) && defined(__ANDROID__)
    return true;
#else
    return false;
#endif
}

bool iris_sdk_is_using_gpu(void) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_processor || !g_processor->isInitialized()) {
        return false;
    }

    return g_processor->isUsingGpu();
}

void iris_sdk_set_min_detection_confidence(float min_confidence) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_processor) {
        g_processor->setMinDetectionConfidence(min_confidence);
    }
}

void iris_sdk_set_min_tracking_confidence(float min_confidence) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_processor) {
        g_processor->setMinTrackingConfidence(min_confidence);
    }
}

void iris_sdk_set_min_presence_confidence(float min_confidence) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_processor) {
        g_processor->setMinPresenceConfidence(min_confidence);
    }
}

void iris_sdk_set_use_inference_thread(bool enable) {
    std::lock_guard<std::mutex> lock(g_mutex);

    LOGI("iris_sdk_set_use_inference_thread: %s (current g_use_inference_thread_request=%s)",
         enable ? "true" : "false",
         g_use_inference_thread_request ? "true" : "false");

    // 이미 초기화된 경우 경고
    if (g_processor && g_processor->isInitialized()) {
        set_last_error("setUseInferenceThread() must be called before init()");
        LOGW("setUseInferenceThread called after init - ignored");
        return;
    }

    // 전역 플래그 설정 (init()에서 적용됨)
    g_use_inference_thread_request = enable;
    LOGI("g_use_inference_thread_request set to: %s", enable ? "true" : "false");
}

bool iris_sdk_is_using_inference_thread(void) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_processor || !g_processor->isInitialized()) {
        return g_use_inference_thread_request;  // 설정된 요청 값 반환
    }

    return g_processor->isUsingInferenceThread();
}

// ============================================================================
// Temporal Stabilizer API 구현
// ============================================================================

void iris_sdk_default_stabilizer_config(IrisStabilizerConfig* config) {
    if (!config) return;
    config->iris_min_cutoff = 4.0f;
    config->iris_beta = 15.0f;
    config->radius_min_cutoff = 4.0f;
    config->radius_beta = 7.5f;
    config->eyelid_min_cutoff = 4.0f;
    config->eyelid_beta = 10.0f;
    config->confidence_low_threshold = 0.3f;
    config->confidence_high_threshold = 0.6f;
    config->confidence_low_frames = 3;
    config->fade_in_ms = 100.0f;
    config->fade_out_ms = 200.0f;
    config->hold_frames = 5;
    config->outlier_radius_multiplier = 4.0f;
    config->outlier_confirm_frames = 1;
    config->blink_ear_threshold = 0.2f;
}

int64_t iris_sdk_create_stabilizer(const IrisStabilizerConfig* config) {
    std::lock_guard<std::mutex> lock(g_mutex);

    iris_sdk::StabilizerConfig cpp_config;
    if (config) {
        cpp_config.iris_min_cutoff = config->iris_min_cutoff;
        cpp_config.iris_beta = config->iris_beta;
        cpp_config.radius_min_cutoff = config->radius_min_cutoff;
        cpp_config.radius_beta = config->radius_beta;
        cpp_config.eyelid_min_cutoff = config->eyelid_min_cutoff;
        cpp_config.eyelid_beta = config->eyelid_beta;
        cpp_config.confidence_low_threshold = config->confidence_low_threshold;
        cpp_config.confidence_high_threshold = config->confidence_high_threshold;
        cpp_config.confidence_low_frames = config->confidence_low_frames;
        cpp_config.fade_in_ms = config->fade_in_ms;
        cpp_config.fade_out_ms = config->fade_out_ms;
        cpp_config.hold_frames = config->hold_frames;
        cpp_config.outlier_radius_multiplier = config->outlier_radius_multiplier;
        cpp_config.outlier_confirm_frames = config->outlier_confirm_frames;
        cpp_config.blink_ear_threshold = config->blink_ear_threshold;
    }

    int64_t handle = g_next_stabilizer_handle++;
    g_stabilizers[handle] = std::make_unique<iris_sdk::TemporalStabilizer>(cpp_config);
    return handle;
}

void iris_sdk_destroy_stabilizer(int64_t handle) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_stabilizers.erase(handle);
}

IrisSdkError iris_sdk_stabilize(
    int64_t handle,
    const IrisResult* raw,
    double timestamp_sec,
    IrisStabilizedResult* out) {

    if (!raw || !out) {
        set_last_error("null pointer");
        return IRIS_SDK_NULL_POINTER;
    }

    std::lock_guard<std::mutex> lock(g_mutex);

    auto it = g_stabilizers.find(handle);
    if (it == g_stabilizers.end()) {
        set_last_error("invalid stabilizer handle");
        return IRIS_SDK_INVALID_PARAM;
    }

    // C IrisResult → C++ IrisResult
    iris_sdk::IrisResult cpp_raw = convert_to_cpp_iris_result(raw);

    // Stabilize
    iris_sdk::StabilizedResult cpp_result = it->second->stabilize(cpp_raw, timestamp_sec);

    // Convert output
    convert_to_c_iris_result(cpp_result.raw, &out->raw);
    convert_to_c_iris_result(cpp_result.stabilized, &out->stabilized);
    out->visibility = cpp_result.visibility;
    out->is_held = cpp_result.is_held ? 1 : 0;
    out->last_valid_ms = cpp_result.last_valid_ms;

    set_last_error(nullptr);
    return IRIS_SDK_OK;
}

void iris_sdk_stabilizer_set_enabled(int64_t handle, int enabled) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_stabilizers.find(handle);
    if (it != g_stabilizers.end()) {
        it->second->setEnabled(enabled != 0);
    }
}

void iris_sdk_stabilizer_reset(int64_t handle) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_stabilizers.find(handle);
    if (it != g_stabilizers.end()) {
        it->second->reset();
    }
}

// ============================================================================
// Async Frame API (P5-W1-04)
// ============================================================================

IrisSdkError iris_sdk_submit_frame(
    const uint8_t* frame_data,
    int width,
    int height,
    IrisFrameFormat format) {

    if (frame_data == nullptr) {
        set_last_error("frame_data is null");
        return IRIS_SDK_NULL_POINTER;
    }

    if (width <= 0 || height <= 0) {
        set_last_error("Invalid frame dimensions");
        return IRIS_SDK_INVALID_PARAM;
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_processor || !g_processor->isInitialized()) {
        set_last_error("SDK not initialized");
        return IRIS_SDK_NOT_INITIALIZED;
    }

    iris_sdk::FrameFormat cpp_format = convert_frame_format(format);
    g_processor->submitFrame(frame_data, width, height, cpp_format);

    return IRIS_SDK_OK;
}

IrisSdkError iris_sdk_submit_frame_with_rotation(
    const uint8_t* frame_data,
    int width,
    int height,
    IrisFrameFormat format,
    int rotation_degrees) {

    if (frame_data == nullptr) {
        set_last_error("frame_data is null");
        return IRIS_SDK_NULL_POINTER;
    }

    if (width <= 0 || height <= 0) {
        set_last_error("Invalid frame dimensions");
        return IRIS_SDK_INVALID_PARAM;
    }

    // 회전 각도 정규화
    int normalized_rotation = ((rotation_degrees % 360) + 360) % 360;
    if (normalized_rotation != 0 && normalized_rotation != 90 &&
        normalized_rotation != 180 && normalized_rotation != 270) {
        set_last_error("Invalid rotation degrees (must be 0, 90, 180, or 270)");
        return IRIS_SDK_INVALID_PARAM;
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_processor || !g_processor->isInitialized()) {
        set_last_error("SDK not initialized");
        return IRIS_SDK_NOT_INITIALIZED;
    }

    iris_sdk::FrameFormat cpp_format = convert_frame_format(format);
    g_processor->submitFrameWithRotation(frame_data, width, height,
                                          cpp_format, normalized_rotation);

    return IRIS_SDK_OK;
}

IrisSdkError iris_sdk_get_latest_result(IrisResult* result) {
    if (result == nullptr) {
        set_last_error("result is null");
        return IRIS_SDK_NULL_POINTER;
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_processor || !g_processor->isInitialized()) {
        set_last_error("SDK not initialized");
        return IRIS_SDK_NOT_INITIALIZED;
    }

    iris_sdk::IrisResult cpp_result;
    if (!g_processor->getLatestResult(cpp_result)) {
        // 아직 결과 없음
        std::memset(result, 0, sizeof(IrisResult));
        result->detected = false;
        return IRIS_SDK_NO_FACE;
    }

    convert_to_c_iris_result(cpp_result, result);
    return IRIS_SDK_OK;
}

IrisSdkError iris_sdk_render_with_result(
    uint8_t* frame_data,
    int width,
    int height,
    IrisFrameFormat format,
    const IrisResult* iris_result,
    const IrisLensConfig* config) {

    if (frame_data == nullptr) {
        set_last_error("frame_data is null");
        return IRIS_SDK_NULL_POINTER;
    }

    if (iris_result == nullptr || config == nullptr) {
        set_last_error("iris_result or config is null");
        return IRIS_SDK_NULL_POINTER;
    }

    if (width <= 0 || height <= 0) {
        set_last_error("Invalid frame dimensions");
        return IRIS_SDK_INVALID_PARAM;
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_processor || !g_processor->isInitialized()) {
        set_last_error("SDK not initialized");
        return IRIS_SDK_NOT_INITIALIZED;
    }

    if (!g_processor->hasLensTexture()) {
        set_last_error("No lens texture loaded");
        return IRIS_SDK_NO_TEXTURE;
    }

    iris_sdk::FrameFormat cpp_format = convert_frame_format(format);
    iris_sdk::IrisResult cpp_iris_result = convert_to_cpp_iris_result(iris_result);
    iris_sdk::LensConfig cpp_config = convert_to_cpp_lens_config(config);

    bool success = g_processor->renderWithResult(
        frame_data, width, height, cpp_format, cpp_iris_result, cpp_config);

    if (!success) {
        set_last_error("Render with result failed");
        return IRIS_SDK_RENDER_FAILED;
    }

    return IRIS_SDK_OK;
}

IrisSdkError iris_sdk_set_eye_refiner_policy(IrisEyeRefinerPolicy policy) {
    // This needs to access the SDK manager's detector
    // For now, store and apply on next init
    // TODO: Implement via SDKManager
    return IRIS_SDK_OK;
}

}  // extern "C"

// ============================================================================
// 테스트 전용 hook (테스트 빌드에서만 노출)
// ============================================================================

#ifdef IRIS_SDK_TEST_HOOKS
extern "C" int iris_sdk_test_convert_blend_mode(int raw_mode) {
    return static_cast<int>(convert_blend_mode(static_cast<IrisBlendMode>(raw_mode)));
}
#endif
