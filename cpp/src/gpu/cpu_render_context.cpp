/**
 * @file cpu_render_context.cpp
 * @brief CPU 기반 렌더링 컨텍스트 구현
 */

#include "iris_sdk/gpu/cpu_render_context.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cstring>

namespace iris_sdk {

CPURenderContext::CPURenderContext() = default;

CPURenderContext::~CPURenderContext() {
    release();
}

bool CPURenderContext::initialize() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (initialized_) {
        return true;
    }

    // CPU 컨텍스트는 특별한 초기화 불필요
    initialized_ = true;
    return true;
}

void CPURenderContext::release() {
    std::lock_guard<std::mutex> lock(mutex_);

    // 모든 텍스처 해제
    textures_.clear();
    next_texture_id_ = 1;
    initialized_ = false;
}

bool CPURenderContext::isInitialized() const {
    return initialized_;
}

TextureHandle CPURenderContext::createTexture(int width, int height,
                                               TextureFormat format) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_ || width <= 0 || height <= 0) {
        return TextureHandle{};
    }

    // cv::Mat 생성
    int cv_type = toCvType(format);
    if (cv_type < 0) {
        return TextureHandle{};
    }

    auto mat = std::make_unique<cv::Mat>(height, width, cv_type, cv::Scalar::all(0));

    // 텍스처 ID 발급
    uint64_t id = next_texture_id_++;

    // 핸들 생성
    TextureHandle handle;
    handle.native_handle = mat.get();
    handle.type = TextureHandle::Type::CPU;
    handle.width = width;
    handle.height = height;
    handle.format = format;
    handle.id = id;

    // 저장
    textures_[id] = TextureData{std::move(mat), format};

    return handle;
}

void CPURenderContext::deleteTexture(TextureHandle& handle) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!handle.isValid() || handle.type != TextureHandle::Type::CPU) {
        return;
    }

    auto it = textures_.find(handle.id);
    if (it != textures_.end()) {
        textures_.erase(it);
    }

    handle.invalidate();
}

bool CPURenderContext::uploadTexture(TextureHandle& handle, const uint8_t* data,
                                      int width, int height, TextureFormat format) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!handle.isValid() || handle.type != TextureHandle::Type::CPU) {
        return false;
    }

    if (data == nullptr || width <= 0 || height <= 0) {
        return false;
    }

    auto it = textures_.find(handle.id);
    if (it == textures_.end()) {
        return false;
    }

    cv::Mat* mat = it->second.mat.get();

    // 크기가 다르면 재할당
    if (mat->cols != width || mat->rows != height) {
        int cv_type = toCvType(format);
        if (cv_type < 0) {
            return false;
        }
        mat->create(height, width, cv_type);
        handle.width = width;
        handle.height = height;
    }

    // 포맷이 다르면 변환 필요 (간단하게 동일 포맷만 지원)
    if (format != handle.format) {
        // TODO: 포맷 변환 지원
        return false;
    }

    // 데이터 복사
    size_t data_size = static_cast<size_t>(width) * height * handle.bytesPerPixel();
    if (mat->isContinuous()) {
        std::memcpy(mat->data, data, data_size);
    } else {
        // 연속 메모리가 아닌 경우 행별 복사
        int row_bytes = width * handle.bytesPerPixel();
        for (int y = 0; y < height; ++y) {
            std::memcpy(mat->ptr(y), data + y * row_bytes, row_bytes);
        }
    }

    return true;
}

bool CPURenderContext::downloadTexture(const TextureHandle& handle, uint8_t* data,
                                         size_t max_size) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!handle.isValid() || handle.type != TextureHandle::Type::CPU) {
        return false;
    }

    if (data == nullptr || max_size == 0) {
        return false;
    }

    auto it = textures_.find(handle.id);
    if (it == textures_.end()) {
        return false;
    }

    const cv::Mat* mat = it->second.mat.get();
    size_t data_size = mat->total() * mat->elemSize();

    if (max_size < data_size) {
        return false;
    }

    if (mat->isContinuous()) {
        std::memcpy(data, mat->data, data_size);
    } else {
        int row_bytes = mat->cols * static_cast<int>(mat->elemSize());
        for (int y = 0; y < mat->rows; ++y) {
            std::memcpy(data + y * row_bytes, mat->ptr(y), row_bytes);
        }
    }

    return true;
}

cv::Mat* CPURenderContext::getCvMat(const TextureHandle& handle) const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!handle.isValid() || handle.type != TextureHandle::Type::CPU) {
        return nullptr;
    }

    auto it = textures_.find(handle.id);
    if (it == textures_.end()) {
        return nullptr;
    }

    return it->second.mat.get();
}

const cv::Mat& CPURenderContext::getCvMatRef(const TextureHandle& handle) const {
    cv::Mat* mat = getCvMat(handle);
    if (mat == nullptr) {
        return empty_mat_;
    }
    return *mat;
}

bool CPURenderContext::dumpTexture(const TextureHandle& handle,
                                    const std::string& file_path) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!handle.isValid() || handle.type != TextureHandle::Type::CPU) {
        return false;
    }

    auto it = textures_.find(handle.id);
    if (it == textures_.end()) {
        return false;
    }

    const cv::Mat* mat = it->second.mat.get();

    // RGBA → BGR 변환 (imwrite는 BGR 사용)
    cv::Mat bgr_mat;
    if (mat->channels() == 4) {
        cv::cvtColor(*mat, bgr_mat, cv::COLOR_RGBA2BGR);
    } else if (mat->channels() == 3) {
        cv::cvtColor(*mat, bgr_mat, cv::COLOR_RGB2BGR);
    } else {
        bgr_mat = *mat;
    }

    return cv::imwrite(file_path, bgr_mat);
}

size_t CPURenderContext::getTextureCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return textures_.size();
}

size_t CPURenderContext::getTextureMemoryUsage() const {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t total = 0;
    for (const auto& pair : textures_) {
        const cv::Mat* mat = pair.second.mat.get();
        total += mat->total() * mat->elemSize();
    }
    return total;
}

int CPURenderContext::toCvType(TextureFormat format) const {
    switch (format) {
        case TextureFormat::RGBA8:   return CV_8UC4;
        case TextureFormat::RGB8:    return CV_8UC3;
        case TextureFormat::R8:      return CV_8UC1;
        case TextureFormat::RGBA16F: return CV_32FC4;  // 근사 (CV_16F 대신)
        default:                     return -1;
    }
}

TextureFormat CPURenderContext::fromCvType(int cv_type) const {
    switch (cv_type) {
        case CV_8UC4:  return TextureFormat::RGBA8;
        case CV_8UC3:  return TextureFormat::RGB8;
        case CV_8UC1:  return TextureFormat::R8;
        case CV_32FC4: return TextureFormat::RGBA16F;
        default:       return TextureFormat::Unknown;
    }
}

} // namespace iris_sdk
