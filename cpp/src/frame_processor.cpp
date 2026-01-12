/**
 * @file frame_processor.cpp
 * @brief 프레임 처리 파이프라인 구현
 *
 * 검출과 렌더링을 통합하는 파이프라인 구현.
 * 다양한 프레임 포맷 변환 및 버퍼 재사용을 통한 최적화.
 */

#include "iris_sdk/frame_processor.h"
#include "iris_sdk/mediapipe_detector.h"
#include "iris_sdk/lens_renderer.h"

#include <chrono>
#include <deque>
#include <numeric>

#ifdef IRIS_SDK_HAS_OPENCV
#include <opencv2/opencv.hpp>
#endif

namespace iris_sdk {

// ============================================================================
// Impl 클래스 정의
// ============================================================================

class FrameProcessor::Impl {
public:
    Impl() = default;
    ~Impl() { release(); }

    // 복사/이동
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&) = default;
    Impl& operator=(Impl&&) = default;

    // 초기화
    bool initialize(const std::string& model_path, DetectorType detector_type);
    void release();
    bool isInitialized() const noexcept { return initialized_; }

    // 텍스처
    bool loadLensTexture(const std::string& texture_path);
    bool loadLensTexture(const uint8_t* data, int width, int height);
    void unloadLensTexture();
    bool hasLensTexture() const noexcept;

    // 처리
    ProcessResult process(uint8_t* frame_data, int width, int height,
                          FrameFormat format, const LensConfig* config);
    ProcessResult process(cv::Mat& frame, const LensConfig* config);
    IrisResult detectOnly(const uint8_t* frame_data, int width, int height,
                          FrameFormat format);
    bool renderOnly(uint8_t* frame_data, int width, int height,
                    FrameFormat format, const IrisResult& iris_result,
                    const LensConfig& config);

    // 설정
    void setMinConfidence(float min_confidence);
    void setFaceTracking(bool enable);

    // 통계
    double getLastProcessingTimeMs() const noexcept { return last_processing_time_ms_; }
    double getAverageFPS() const noexcept;

    // 디버그용 Face Mesh 접근
    int getFaceLandmarkCount() const {
        return detector_ ? detector_->getFaceLandmarkCount() : 0;
    }
    bool getFaceLandmarks(float* out_landmarks) const {
        return detector_ ? detector_->getFaceLandmarks(out_landmarks) : false;
    }
    int getModelVersion() const {
        return detector_ ? detector_->getModelVersion() : 0;
    }

private:
    // 포맷 변환
    bool convertToWorkingFormat(const uint8_t* input, int width, int height,
                                FrameFormat format, cv::Mat& output);
    bool convertFromWorkingFormat(const cv::Mat& input, uint8_t* output,
                                  int width, int height, FrameFormat format);

    // NV21/NV12 변환 헬퍼
    void convertNV21toBGR(const uint8_t* nv21, int width, int height, cv::Mat& bgr);
    void convertNV12toBGR(const uint8_t* nv12, int width, int height, cv::Mat& bgr);
    void convertBGRtoNV21(const cv::Mat& bgr, uint8_t* nv21, int width, int height);
    void convertBGRtoNV12(const cv::Mat& bgr, uint8_t* nv12, int width, int height);

    // FPS 계산
    void recordProcessingTime(double time_ms);

    // 멤버 변수
    bool initialized_ = false;
    std::unique_ptr<MediaPipeDetector> detector_;
    std::unique_ptr<LensRenderer> renderer_;
    DetectorType detector_type_ = DetectorType::MediaPipe;

    // 설정
    float min_confidence_ = 0.5f;
    bool face_tracking_ = true;

    // 버퍼 재사용 (메모리 할당 최소화)
    cv::Mat work_buffer_;
    cv::Mat rgb_buffer_;

    // 통계
    double last_processing_time_ms_ = 0.0;
    std::deque<double> processing_times_;
    static constexpr size_t MAX_FPS_SAMPLES = 30;
};

// ============================================================================
// Impl 구현 - 초기화
// ============================================================================

bool FrameProcessor::Impl::initialize(const std::string& model_path,
                                       DetectorType detector_type) {
#ifndef IRIS_SDK_HAS_OPENCV
    return false;
#else
    if (initialized_) {
        release();
    }

    detector_type_ = detector_type;

    // 검출기 생성 및 초기화
    switch (detector_type) {
        case DetectorType::MediaPipe:
            detector_ = std::make_unique<MediaPipeDetector>();
            break;
        case DetectorType::EyeOnly:
        case DetectorType::Hybrid:
            // Phase 2에서 구현 예정
            return false;
        default:
            return false;
    }

    if (!detector_->initialize(model_path)) {
        detector_.reset();
        return false;
    }

    // 렌더러 생성 및 초기화
    renderer_ = std::make_unique<LensRenderer>();
    if (!renderer_->initialize()) {
        detector_->release();
        detector_.reset();
        renderer_.reset();
        return false;
    }

    initialized_ = true;
    return true;
#endif
}

void FrameProcessor::Impl::release() {
    if (renderer_) {
        renderer_->release();
        renderer_.reset();
    }
    if (detector_) {
        detector_->release();
        detector_.reset();
    }
    work_buffer_.release();
    rgb_buffer_.release();
    processing_times_.clear();
    initialized_ = false;
}

// ============================================================================
// Impl 구현 - 텍스처
// ============================================================================

bool FrameProcessor::Impl::loadLensTexture(const std::string& texture_path) {
    if (!initialized_ || !renderer_) return false;
    return renderer_->loadTexture(texture_path);
}

bool FrameProcessor::Impl::loadLensTexture(const uint8_t* data, int width, int height) {
    if (!initialized_ || !renderer_) return false;
    return renderer_->loadTexture(data, width, height);
}

void FrameProcessor::Impl::unloadLensTexture() {
    if (renderer_) {
        renderer_->unloadTexture();
    }
}

bool FrameProcessor::Impl::hasLensTexture() const noexcept {
    return renderer_ && renderer_->hasTexture();
}

// ============================================================================
// Impl 구현 - 포맷 변환
// ============================================================================

bool FrameProcessor::Impl::convertToWorkingFormat(const uint8_t* input,
                                                   int width, int height,
                                                   FrameFormat format,
                                                   cv::Mat& output) {
#ifndef IRIS_SDK_HAS_OPENCV
    return false;
#else
    switch (format) {
        case FrameFormat::BGR: {
            // BGR은 직접 사용 (zero-copy)
            output = cv::Mat(height, width, CV_8UC3,
                             const_cast<uint8_t*>(input));
            return true;
        }
        case FrameFormat::RGB: {
            cv::Mat rgb(height, width, CV_8UC3, const_cast<uint8_t*>(input));
            cv::cvtColor(rgb, output, cv::COLOR_RGB2BGR);
            return true;
        }
        case FrameFormat::RGBA: {
            cv::Mat rgba(height, width, CV_8UC4, const_cast<uint8_t*>(input));
            cv::cvtColor(rgba, output, cv::COLOR_RGBA2BGR);
            return true;
        }
        case FrameFormat::BGRA: {
            cv::Mat bgra(height, width, CV_8UC4, const_cast<uint8_t*>(input));
            cv::cvtColor(bgra, output, cv::COLOR_BGRA2BGR);
            return true;
        }
        case FrameFormat::NV21: {
            convertNV21toBGR(input, width, height, output);
            return true;
        }
        case FrameFormat::NV12: {
            convertNV12toBGR(input, width, height, output);
            return true;
        }
        case FrameFormat::Grayscale: {
            cv::Mat gray(height, width, CV_8UC1, const_cast<uint8_t*>(input));
            cv::cvtColor(gray, output, cv::COLOR_GRAY2BGR);
            return true;
        }
        default:
            return false;
    }
#endif
}

bool FrameProcessor::Impl::convertFromWorkingFormat(const cv::Mat& input,
                                                     uint8_t* output,
                                                     int width, int height,
                                                     FrameFormat format) {
#ifndef IRIS_SDK_HAS_OPENCV
    return false;
#else
    if (input.empty() || input.cols != width || input.rows != height) {
        return false;
    }

    switch (format) {
        case FrameFormat::BGR: {
            // BGR은 이미 작업 포맷이므로 직접 복사
            if (input.isContinuous()) {
                std::memcpy(output, input.data, width * height * 3);
            } else {
                for (int y = 0; y < height; ++y) {
                    std::memcpy(output + y * width * 3,
                               input.ptr(y), width * 3);
                }
            }
            return true;
        }
        case FrameFormat::RGB: {
            cv::Mat rgb;
            cv::cvtColor(input, rgb, cv::COLOR_BGR2RGB);
            std::memcpy(output, rgb.data, width * height * 3);
            return true;
        }
        case FrameFormat::RGBA: {
            cv::Mat rgba;
            cv::cvtColor(input, rgba, cv::COLOR_BGR2RGBA);
            std::memcpy(output, rgba.data, width * height * 4);
            return true;
        }
        case FrameFormat::BGRA: {
            cv::Mat bgra;
            cv::cvtColor(input, bgra, cv::COLOR_BGR2BGRA);
            std::memcpy(output, bgra.data, width * height * 4);
            return true;
        }
        case FrameFormat::NV21: {
            convertBGRtoNV21(input, output, width, height);
            return true;
        }
        case FrameFormat::NV12: {
            convertBGRtoNV12(input, output, width, height);
            return true;
        }
        case FrameFormat::Grayscale: {
            cv::Mat gray;
            cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
            std::memcpy(output, gray.data, width * height);
            return true;
        }
        default:
            return false;
    }
#endif
}

void FrameProcessor::Impl::convertNV21toBGR(const uint8_t* nv21,
                                             int width, int height,
                                             cv::Mat& bgr) {
#ifdef IRIS_SDK_HAS_OPENCV
    // NV21: Y plane 후 VU interleaved
    cv::Mat nv21_mat(height + height / 2, width, CV_8UC1,
                     const_cast<uint8_t*>(nv21));
    cv::cvtColor(nv21_mat, bgr, cv::COLOR_YUV2BGR_NV21);
#endif
}

void FrameProcessor::Impl::convertNV12toBGR(const uint8_t* nv12,
                                             int width, int height,
                                             cv::Mat& bgr) {
#ifdef IRIS_SDK_HAS_OPENCV
    // NV12: Y plane 후 UV interleaved
    cv::Mat nv12_mat(height + height / 2, width, CV_8UC1,
                     const_cast<uint8_t*>(nv12));
    cv::cvtColor(nv12_mat, bgr, cv::COLOR_YUV2BGR_NV12);
#endif
}

void FrameProcessor::Impl::convertBGRtoNV21(const cv::Mat& bgr,
                                             uint8_t* nv21,
                                             int width, int height) {
#ifdef IRIS_SDK_HAS_OPENCV
    // BGR → YUV_YV12 (Y, V, U planar)
    cv::Mat yuv_yv12;
    cv::cvtColor(bgr, yuv_yv12, cv::COLOR_BGR2YUV_YV12);

    const int y_size = width * height;
    const int uv_size = y_size / 4;

    // Y plane 복사
    std::memcpy(nv21, yuv_yv12.data, y_size);

    // V, U → VU interleaved (NV21)
    const uint8_t* v_plane = yuv_yv12.data + y_size;
    const uint8_t* u_plane = yuv_yv12.data + y_size + uv_size;
    uint8_t* vu_plane = nv21 + y_size;

    for (int i = 0; i < uv_size; ++i) {
        vu_plane[2 * i] = v_plane[i];      // V
        vu_plane[2 * i + 1] = u_plane[i];  // U
    }
#endif
}

void FrameProcessor::Impl::convertBGRtoNV12(const cv::Mat& bgr,
                                             uint8_t* nv12,
                                             int width, int height) {
#ifdef IRIS_SDK_HAS_OPENCV
    // BGR → YUV_I420 (Y, U, V planar)
    cv::Mat yuv_i420;
    cv::cvtColor(bgr, yuv_i420, cv::COLOR_BGR2YUV_I420);

    const int y_size = width * height;
    const int uv_size = y_size / 4;

    // Y plane 복사
    std::memcpy(nv12, yuv_i420.data, y_size);

    // U, V → UV interleaved (NV12)
    const uint8_t* u_plane = yuv_i420.data + y_size;
    const uint8_t* v_plane = yuv_i420.data + y_size + uv_size;
    uint8_t* uv_plane = nv12 + y_size;

    for (int i = 0; i < uv_size; ++i) {
        uv_plane[2 * i] = u_plane[i];      // U
        uv_plane[2 * i + 1] = v_plane[i];  // V
    }
#endif
}

// ============================================================================
// Impl 구현 - 처리
// ============================================================================

ProcessResult FrameProcessor::Impl::process(uint8_t* frame_data,
                                             int width, int height,
                                             FrameFormat format,
                                             const LensConfig* config) {
    ProcessResult result;
    result.success = false;

#ifndef IRIS_SDK_HAS_OPENCV
    result.error_code = ErrorCode::NotInitialized;
    return result;
#else
    if (!initialized_) {
        result.error_code = ErrorCode::NotInitialized;
        return result;
    }

    if (frame_data == nullptr || width <= 0 || height <= 0) {
        result.error_code = ErrorCode::InvalidParameter;
        return result;
    }

    auto total_start = std::chrono::high_resolution_clock::now();

    // 1. 포맷 변환 (작업 버퍼로)
    auto convert_start = std::chrono::high_resolution_clock::now();
    if (!convertToWorkingFormat(frame_data, width, height, format, work_buffer_)) {
        result.error_code = ErrorCode::FrameFormatUnsupported;
        return result;
    }
    auto convert_end = std::chrono::high_resolution_clock::now();
    result.convert_time_ms = std::chrono::duration<float, std::milli>(
        convert_end - convert_start).count();

    // 2. 검출을 위해 RGB로 변환
    auto detect_start = std::chrono::high_resolution_clock::now();
    cv::cvtColor(work_buffer_, rgb_buffer_, cv::COLOR_BGR2RGB);

    // 검출 수행
    result.iris_result = detector_->detect(
        rgb_buffer_.data, rgb_buffer_.cols, rgb_buffer_.rows, FrameFormat::RGB);
    auto detect_end = std::chrono::high_resolution_clock::now();
    result.detection_time_ms = std::chrono::duration<float, std::milli>(
        detect_end - detect_start).count();

    // 3. 렌더링 (설정이 있고 검출 성공 시)
    if (config != nullptr && result.iris_result.detected && hasLensTexture()) {
        auto render_start = std::chrono::high_resolution_clock::now();
        renderer_->render(work_buffer_, result.iris_result, *config);
        auto render_end = std::chrono::high_resolution_clock::now();
        result.render_time_ms = std::chrono::duration<float, std::milli>(
            render_end - render_start).count();

        // 4. 원본 포맷으로 변환 (BGR이 아닌 경우)
        if (format != FrameFormat::BGR) {
            auto convert2_start = std::chrono::high_resolution_clock::now();
            convertFromWorkingFormat(work_buffer_, frame_data, width, height, format);
            auto convert2_end = std::chrono::high_resolution_clock::now();
            result.convert_time_ms += std::chrono::duration<float, std::milli>(
                convert2_end - convert2_start).count();
        }
    }

    // 5. 결과 설정
    auto total_end = std::chrono::high_resolution_clock::now();
    result.processing_time_ms = std::chrono::duration<float, std::milli>(
        total_end - total_start).count();
    result.success = true;
    result.error_code = ErrorCode::Success;

    // FPS 계산용 기록
    recordProcessingTime(result.processing_time_ms);
    last_processing_time_ms_ = result.processing_time_ms;

    return result;
#endif
}

ProcessResult FrameProcessor::Impl::process(cv::Mat& frame,
                                             const LensConfig* config) {
    ProcessResult result;
    result.success = false;

#ifndef IRIS_SDK_HAS_OPENCV
    result.error_code = ErrorCode::NotInitialized;
    return result;
#else
    if (!initialized_) {
        result.error_code = ErrorCode::NotInitialized;
        return result;
    }

    if (frame.empty()) {
        result.error_code = ErrorCode::InvalidParameter;
        return result;
    }

    auto total_start = std::chrono::high_resolution_clock::now();

    // BGR/BGRA를 작업 버퍼로 변환
    auto convert_start = std::chrono::high_resolution_clock::now();
    if (frame.channels() == 4) {
        cv::cvtColor(frame, work_buffer_, cv::COLOR_BGRA2BGR);
    } else if (frame.channels() == 3) {
        work_buffer_ = frame;  // BGR은 직접 사용
    } else if (frame.channels() == 1) {
        cv::cvtColor(frame, work_buffer_, cv::COLOR_GRAY2BGR);
    } else {
        result.error_code = ErrorCode::FrameFormatUnsupported;
        return result;
    }
    auto convert_end = std::chrono::high_resolution_clock::now();
    result.convert_time_ms = std::chrono::duration<float, std::milli>(
        convert_end - convert_start).count();

    // 검출을 위해 RGB로 변환
    auto detect_start = std::chrono::high_resolution_clock::now();
    cv::cvtColor(work_buffer_, rgb_buffer_, cv::COLOR_BGR2RGB);

    // 검출 수행
    result.iris_result = detector_->detect(
        rgb_buffer_.data, rgb_buffer_.cols, rgb_buffer_.rows, FrameFormat::RGB);
    auto detect_end = std::chrono::high_resolution_clock::now();
    result.detection_time_ms = std::chrono::duration<float, std::milli>(
        detect_end - detect_start).count();

    // 렌더링 (설정이 있고 검출 성공 시)
    if (config != nullptr && result.iris_result.detected && hasLensTexture()) {
        auto render_start = std::chrono::high_resolution_clock::now();

        // BGR 프레임에 직접 렌더링
        if (frame.channels() == 3 && work_buffer_.data == frame.data) {
            renderer_->render(frame, result.iris_result, *config);
        } else {
            renderer_->render(work_buffer_, result.iris_result, *config);

            // 결과를 원본 프레임으로 복사
            if (frame.channels() == 4) {
                cv::cvtColor(work_buffer_, frame, cv::COLOR_BGR2BGRA);
            } else if (frame.channels() == 1) {
                cv::cvtColor(work_buffer_, frame, cv::COLOR_BGR2GRAY);
            } else {
                work_buffer_.copyTo(frame);
            }
        }

        auto render_end = std::chrono::high_resolution_clock::now();
        result.render_time_ms = std::chrono::duration<float, std::milli>(
            render_end - render_start).count();
    }

    // 결과 설정
    auto total_end = std::chrono::high_resolution_clock::now();
    result.processing_time_ms = std::chrono::duration<float, std::milli>(
        total_end - total_start).count();
    result.success = true;
    result.error_code = ErrorCode::Success;

    // FPS 계산용 기록
    recordProcessingTime(result.processing_time_ms);
    last_processing_time_ms_ = result.processing_time_ms;

    return result;
#endif
}

IrisResult FrameProcessor::Impl::detectOnly(const uint8_t* frame_data,
                                             int width, int height,
                                             FrameFormat format) {
    IrisResult result;
    result.detected = false;

#ifdef IRIS_SDK_HAS_OPENCV
    if (!initialized_ || frame_data == nullptr || width <= 0 || height <= 0) {
        return result;
    }

    // 포맷 변환
    if (!convertToWorkingFormat(frame_data, width, height, format, work_buffer_)) {
        return result;
    }

    // RGB로 변환 후 검출
    cv::cvtColor(work_buffer_, rgb_buffer_, cv::COLOR_BGR2RGB);
    result = detector_->detect(rgb_buffer_.data, rgb_buffer_.cols,
                                rgb_buffer_.rows, FrameFormat::RGB);
#endif

    return result;
}

bool FrameProcessor::Impl::renderOnly(uint8_t* frame_data,
                                       int width, int height,
                                       FrameFormat format,
                                       const IrisResult& iris_result,
                                       const LensConfig& config) {
#ifndef IRIS_SDK_HAS_OPENCV
    return false;
#else
    if (!initialized_ || !hasLensTexture() || !iris_result.detected) {
        return false;
    }

    if (frame_data == nullptr || width <= 0 || height <= 0) {
        return false;
    }

    // 포맷 변환
    if (!convertToWorkingFormat(frame_data, width, height, format, work_buffer_)) {
        return false;
    }

    // 렌더링
    if (!renderer_->render(work_buffer_, iris_result, config)) {
        return false;
    }

    // 원본 포맷으로 변환
    return convertFromWorkingFormat(work_buffer_, frame_data, width, height, format);
#endif
}

// ============================================================================
// Impl 구현 - 설정
// ============================================================================

void FrameProcessor::Impl::setMinConfidence(float min_confidence) {
    min_confidence_ = std::clamp(min_confidence, 0.0f, 1.0f);
    // detector에 전달
    if (detector_) {
        detector_->setMinDetectionConfidence(min_confidence_);
    }
}

void FrameProcessor::Impl::setFaceTracking(bool enable) {
    face_tracking_ = enable;
    // TODO: detector에 전달
}

// ============================================================================
// Impl 구현 - 통계
// ============================================================================

void FrameProcessor::Impl::recordProcessingTime(double time_ms) {
    processing_times_.push_back(time_ms);
    if (processing_times_.size() > MAX_FPS_SAMPLES) {
        processing_times_.pop_front();
    }
}

double FrameProcessor::Impl::getAverageFPS() const noexcept {
    if (processing_times_.empty()) {
        return 0.0;
    }

    double avg_time = std::accumulate(processing_times_.begin(),
                                       processing_times_.end(), 0.0) /
                      processing_times_.size();

    if (avg_time <= 0.0) {
        return 0.0;
    }

    return 1000.0 / avg_time;
}

// ============================================================================
// FrameProcessor 공개 인터페이스
// ============================================================================

FrameProcessor::FrameProcessor() : impl_(std::make_unique<Impl>()) {}

FrameProcessor::~FrameProcessor() = default;

FrameProcessor::FrameProcessor(FrameProcessor&&) noexcept = default;

FrameProcessor& FrameProcessor::operator=(FrameProcessor&&) noexcept = default;

bool FrameProcessor::initialize(const std::string& model_path,
                                 DetectorType detector_type) {
    return impl_ ? impl_->initialize(model_path, detector_type) : false;
}

void FrameProcessor::release() {
    if (impl_) impl_->release();
}

bool FrameProcessor::isInitialized() const noexcept {
    return impl_ ? impl_->isInitialized() : false;
}

bool FrameProcessor::loadLensTexture(const std::string& texture_path) {
    return impl_ ? impl_->loadLensTexture(texture_path) : false;
}

bool FrameProcessor::loadLensTexture(const uint8_t* data, int width, int height) {
    return impl_ ? impl_->loadLensTexture(data, width, height) : false;
}

void FrameProcessor::unloadLensTexture() {
    if (impl_) impl_->unloadLensTexture();
}

bool FrameProcessor::hasLensTexture() const noexcept {
    return impl_ ? impl_->hasLensTexture() : false;
}

ProcessResult FrameProcessor::process(uint8_t* frame_data,
                                       int width, int height,
                                       FrameFormat format,
                                       const LensConfig* config) {
    if (!impl_) {
        ProcessResult result;
        result.success = false;
        result.error_code = ErrorCode::NotInitialized;
        return result;
    }
    return impl_->process(frame_data, width, height, format, config);
}

ProcessResult FrameProcessor::process(cv::Mat& frame, const LensConfig* config) {
    if (!impl_) {
        ProcessResult result;
        result.success = false;
        result.error_code = ErrorCode::NotInitialized;
        return result;
    }
    return impl_->process(frame, config);
}

IrisResult FrameProcessor::detectOnly(const uint8_t* frame_data,
                                       int width, int height,
                                       FrameFormat format) {
    if (!impl_) {
        IrisResult result;
        result.detected = false;
        return result;
    }
    return impl_->detectOnly(frame_data, width, height, format);
}

bool FrameProcessor::renderOnly(uint8_t* frame_data,
                                 int width, int height,
                                 FrameFormat format,
                                 const IrisResult& iris_result,
                                 const LensConfig& config) {
    return impl_ ? impl_->renderOnly(frame_data, width, height, format,
                                      iris_result, config) : false;
}

void FrameProcessor::setMinConfidence(float min_confidence) {
    if (impl_) impl_->setMinConfidence(min_confidence);
}

void FrameProcessor::setFaceTracking(bool enable) {
    if (impl_) impl_->setFaceTracking(enable);
}

double FrameProcessor::getLastProcessingTimeMs() const noexcept {
    return impl_ ? impl_->getLastProcessingTimeMs() : 0.0;
}

double FrameProcessor::getAverageFPS() const noexcept {
    return impl_ ? impl_->getAverageFPS() : 0.0;
}

int FrameProcessor::getFaceLandmarkCount() const {
    return impl_ ? impl_->getFaceLandmarkCount() : 0;
}

bool FrameProcessor::getFaceLandmarks(float* out_landmarks) const {
    return impl_ ? impl_->getFaceLandmarks(out_landmarks) : false;
}

int FrameProcessor::getModelVersion() const {
    return impl_ ? impl_->getModelVersion() : 0;
}

} // namespace iris_sdk
