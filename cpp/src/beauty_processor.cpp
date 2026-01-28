/**
 * @file beauty_processor.cpp
 * @brief BeautyProcessor 구현
 */

#include "iris_sdk/beauty_processor.h"
#include "iris_sdk/cpu_beauty_backend.h"
#include <chrono>

// GPU 백엔드 (Android GLES 빌드 시에만)
#if defined(__ANDROID__) && defined(IRIS_SDK_HAS_GLES)
#include "iris_sdk/gpu/gpu_beauty_backend.h"
#endif

namespace iris_sdk {

//=============================================================================
// 생성자/소멸자
//=============================================================================

BeautyProcessor::BeautyProcessor(std::shared_ptr<IRenderContext> render_context)
    : render_context_(std::move(render_context))
    , config_(BeautyFilterConfigV2Helper::defaults()) {
}

BeautyProcessor::~BeautyProcessor() {
    release();
}

//=============================================================================
// 라이프사이클
//=============================================================================

bool BeautyProcessor::initialize(bool prefer_gpu) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (initialized_) return true;

    // RenderContext가 없으면 생성
    if (!render_context_) {
        render_context_ = IRenderContext::create(prefer_gpu);
    }

    if (render_context_ && !render_context_->isInitialized()) {
        render_context_->initialize();
    }

    // 백엔드 선택
    if (!selectBackend(prefer_gpu)) {
        return false;
    }

    initialized_ = true;
    return true;
}

void BeautyProcessor::release() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (backend_) {
        backend_->release();
        backend_.reset();
    }

    // RenderContext는 공유되므로 해제하지 않음

    cached_roi_.invalidate();
    cached_roi_timestamp_ = 0;
    initialized_ = false;
}

bool BeautyProcessor::isInitialized() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return initialized_;
}

//=============================================================================
// 백엔드 선택
//=============================================================================

bool BeautyProcessor::selectBackend(bool prefer_gpu) {
#if defined(__ANDROID__) && defined(IRIS_SDK_HAS_GLES)
    if (prefer_gpu && render_context_ && render_context_->supportsGpu()) {
        auto gpu_backend = std::make_unique<GPUBeautyBackend>();
        if (gpu_backend->initialize(render_context_.get())) {
            backend_ = std::move(gpu_backend);
            return true;
        }
        // GPU 실패 → CPU 폴백
    }
#else
    (void)prefer_gpu;  // 미사용 경고 방지
#endif

    // CPU 백엔드
    backend_ = std::make_unique<CPUBeautyBackend>();
    return backend_->initialize(nullptr);
}

//=============================================================================
// 설정
//=============================================================================

IrisSdkError BeautyProcessor::setConfig(const BeautyFilterConfigV2& config) {
    std::lock_guard<std::mutex> lock(mutex_);

    // 유효성 검사
    if (!BeautyFilterConfigV2Helper::isValid(config)) {
        return IRIS_SDK_INVALID_PARAM;
    }

    config_ = config;
    return IRIS_SDK_OK;
}

IrisSdkError BeautyProcessor::getConfig(BeautyFilterConfigV2& out_config) const {
    std::lock_guard<std::mutex> lock(mutex_);
    out_config = config_;
    return IRIS_SDK_OK;
}

//=============================================================================
// 필터 적용
//=============================================================================

IrisSdkError BeautyProcessor::process(
    uint8_t* frame_data,
    int width, int height,
    IrisFrameFormat format,
    const IrisResult* iris_result) {

    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_ || !backend_) {
        return IRIS_SDK_ERROR_NOT_INITIALIZED;
    }

    if (!config_.enabled) {
        return IRIS_SDK_OK;  // 비활성화 상태
    }

    // ROI 계산
    BeautyROI* roi_ptr = nullptr;
    BeautyROI computed_roi;

    if (config_.roiOnly && iris_result && iris_result->face_mesh_valid) {
        // mutex가 이미 잠겨 있으므로 직접 ROI 계산 호출
        int64_t now = getCurrentTimeMs();
        if (cached_roi_.valid && (now - cached_roi_timestamp_) < ROI_CACHE_TIMEOUT_MS) {
            roi_ptr = &cached_roi_;
        } else {
            if (BeautyROIManager::computeROI(
                    iris_result->face_mesh,
                    IrisResult::FACE_MESH_LANDMARK_COUNT,
                    width, height, config_, computed_roi)) {
                cached_roi_ = computed_roi;
                cached_roi_timestamp_ = now;
                roi_ptr = &cached_roi_;
            }
        }
    }

    // 백엔드에 처리 위임
    return backend_->apply(frame_data, width, height, format, config_, roi_ptr);
}

IrisSdkError BeautyProcessor::processTexture(
    const TextureHandle& input,
    TextureHandle& output,
    const IrisResult* iris_result) {

    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_ || !backend_) {
        return IRIS_SDK_ERROR_NOT_INITIALIZED;
    }

    if (!config_.enabled) {
        return IRIS_SDK_OK;
    }

    if (!backend_->supportsTextureProcessing()) {
        return IRIS_SDK_ERROR_NOT_SUPPORTED;
    }

    // ROI 계산
    BeautyROI* roi_ptr = nullptr;
    BeautyROI computed_roi;

    if (config_.roiOnly && iris_result && iris_result->face_mesh_valid) {
        int64_t now = getCurrentTimeMs();
        if (cached_roi_.valid && (now - cached_roi_timestamp_) < ROI_CACHE_TIMEOUT_MS) {
            roi_ptr = &cached_roi_;
        } else {
            if (BeautyROIManager::computeROI(
                    iris_result->face_mesh,
                    IrisResult::FACE_MESH_LANDMARK_COUNT,
                    input.width, input.height, config_, computed_roi)) {
                cached_roi_ = computed_roi;
                cached_roi_timestamp_ = now;
                roi_ptr = &cached_roi_;
            }
        }
    }

    return backend_->applyTexture(input, output, config_, roi_ptr);
}

//=============================================================================
// 상태 조회
//=============================================================================

bool BeautyProcessor::isUsingGpu() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return backend_ && backend_->supportsGpu();
}

bool BeautyProcessor::isEnabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_.enabled;
}

const char* BeautyProcessor::getBackendName() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return backend_ ? backend_->getName() : "None";
}

//=============================================================================
// 유틸리티
//=============================================================================

bool BeautyProcessor::computeROI(const IrisResult* iris_result, int width, int height,
                                  BeautyROI& out_roi) {
    // 캐시 확인 (100ms 이내)
    int64_t now = getCurrentTimeMs();
    if (cached_roi_.valid && (now - cached_roi_timestamp_) < ROI_CACHE_TIMEOUT_MS) {
        out_roi = cached_roi_;
        return true;
    }

    // 새로 계산
    if (BeautyROIManager::computeROI(
            iris_result->face_mesh,
            IrisResult::FACE_MESH_LANDMARK_COUNT,
            width, height, config_, out_roi)) {
        cached_roi_ = out_roi;
        cached_roi_timestamp_ = now;
        return true;
    }

    return false;
}

int64_t BeautyProcessor::getCurrentTimeMs() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();
}

} // namespace iris_sdk
