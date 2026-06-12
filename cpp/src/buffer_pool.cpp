/**
 * @file buffer_pool.cpp
 * @brief Implementation of BufferPool for cv::Mat reuse
 */

#include "iris_sdk/buffer_pool.h"

#include <chrono>
#include <algorithm>
#include <mutex>

// ============================================================================
// ③-2 B1 주석: BufferPool / ScopedBuffer는 현재 실제 파이프라인에서 미사용이다.
// (카메라→추론→렌더 핸드오프는 InferenceThread의 딥카피 슬롯을 사용한다.)
// 유일한 참조처는 단위 테스트(test_profiler.cpp)뿐이다.
// 표면 정리(파일 삭제 등)는 ④ 단계에서 다룬다. 여기서는 move-assign의 잠재
// ABBA 데드락만 수리한다. — refactor/p32-core-quality
// ============================================================================

namespace iris_sdk {

BufferPool::BufferPool() = default;

BufferPool::~BufferPool() {
    release();
}

BufferPool::BufferPool(BufferPool&& other) noexcept {
    std::lock_guard<std::mutex> lock(other.mutex_);
    width_ = other.width_;
    height_ = other.height_;
    type_ = other.type_;
    max_size_ = other.max_size_;
    initialized_ = other.initialized_;
    buffers_ = std::move(other.buffers_);
    acquire_count_ = other.acquire_count_;
    release_count_ = other.release_count_;
    allocation_count_ = other.allocation_count_;

    other.initialized_ = false;
}

BufferPool& BufferPool::operator=(BufferPool&& other) noexcept {
    if (this != &other) {
        release();

        // ③-2 B1: std::scoped_lock으로 두 뮤텍스를 deadlock-free 순서로 동시 잠근다.
        // 기존엔 other.mutex_ → mutex_ 고정 순서로 잠가, 두 풀을 서로 반대 방향으로
        // 동시에 move-assign하면 ABBA 데드락이 이론상 가능했다.
        std::scoped_lock lock(other.mutex_, mutex_);

        width_ = other.width_;
        height_ = other.height_;
        type_ = other.type_;
        max_size_ = other.max_size_;
        initialized_ = other.initialized_;
        buffers_ = std::move(other.buffers_);
        acquire_count_ = other.acquire_count_;
        release_count_ = other.release_count_;
        allocation_count_ = other.allocation_count_;

        other.initialized_ = false;
    }
    return *this;
}

bool BufferPool::initialize(int width, int height, int type,
                            size_t initial_size, size_t max_size) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (initialized_) {
        return false; // Already initialized
    }

    if (width <= 0 || height <= 0) {
        return false;
    }

    width_ = width;
    height_ = height;
    type_ = type;
    max_size_ = max_size;

    // Pre-allocate initial buffers
    buffers_.reserve(initial_size);
    for (size_t i = 0; i < initial_size; ++i) {
        buffers_.push_back(createBuffer());
        allocation_count_++;
    }

    initialized_ = true;
    return true;
}

void BufferPool::release() {
    std::lock_guard<std::mutex> lock(mutex_);

    buffers_.clear();
    initialized_ = false;
    acquire_count_ = 0;
    release_count_ = 0;
    allocation_count_ = 0;
}

bool BufferPool::isInitialized() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return initialized_;
}

cv::Mat BufferPool::acquire() {
    cv::Mat buffer;
    tryAcquire(buffer);
    return buffer;
}

bool BufferPool::tryAcquire(cv::Mat& buffer) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_) {
        buffer = cv::Mat();
        return false;
    }

    acquire_count_++;

    // Find an available buffer
    for (auto& entry : buffers_) {
        if (!entry.in_use) {
            entry.in_use = true;
            entry.last_used_time = currentTimeMs();
            buffer = entry.buffer;
            return true;
        }
    }

    // No available buffer, try to allocate new one
    if (max_size_ == 0 || buffers_.size() < max_size_) {
        buffers_.push_back(createBuffer());
        allocation_count_++;

        auto& entry = buffers_.back();
        entry.in_use = true;
        entry.last_used_time = currentTimeMs();
        buffer = entry.buffer;
        return true;
    }

    // Pool exhausted
    buffer = cv::Mat();
    return false;
}

void BufferPool::release(cv::Mat& buffer) {
    if (buffer.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_) {
        buffer = cv::Mat();
        return;
    }

    release_count_++;

    // Find and mark as available
    for (auto& entry : buffers_) {
        // Compare by data pointer to identify the buffer
        if (entry.buffer.data == buffer.data) {
            entry.in_use = false;
            entry.last_used_time = currentTimeMs();
            buffer = cv::Mat(); // Clear caller's reference
            return;
        }
    }

    // Buffer not from this pool - just clear reference
    buffer = cv::Mat();
}

BufferPoolStats BufferPool::getStats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    BufferPoolStats stats;
    stats.total_buffers = buffers_.size();
    stats.acquire_count = acquire_count_;
    stats.release_count = release_count_;
    stats.allocation_count = allocation_count_;

    for (const auto& entry : buffers_) {
        if (entry.in_use) {
            stats.in_use_buffers++;
        } else {
            stats.available_buffers++;
        }
    }

    // Estimate memory usage
    if (!buffers_.empty() && !buffers_[0].buffer.empty()) {
        size_t bytes_per_buffer = buffers_[0].buffer.total() *
                                  buffers_[0].buffer.elemSize();
        stats.total_memory_bytes = bytes_per_buffer * stats.total_buffers;
    }

    return stats;
}

void BufferPool::resize(size_t new_size) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_) {
        return;
    }

    max_size_ = new_size;

    // If shrinking, remove unused buffers
    if (new_size > 0 && buffers_.size() > new_size) {
        // Keep only the first new_size buffers, preferring in-use ones
        std::stable_partition(buffers_.begin(), buffers_.end(),
                              [](const BufferEntry& e) { return e.in_use; });

        if (buffers_.size() > new_size) {
            buffers_.resize(new_size);
        }
    }
}

void BufferPool::trim(size_t keep_count) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_) {
        return;
    }

    // Count available buffers
    size_t available = 0;
    for (const auto& entry : buffers_) {
        if (!entry.in_use) {
            available++;
        }
    }

    if (available <= keep_count) {
        return;
    }

    // Remove oldest unused buffers
    // Sort by last_used_time (oldest first) among unused
    std::vector<size_t> unused_indices;
    for (size_t i = 0; i < buffers_.size(); ++i) {
        if (!buffers_[i].in_use) {
            unused_indices.push_back(i);
        }
    }

    std::sort(unused_indices.begin(), unused_indices.end(),
              [this](size_t a, size_t b) {
                  return buffers_[a].last_used_time < buffers_[b].last_used_time;
              });

    // Mark for removal (remove oldest, keep keep_count)
    size_t to_remove = unused_indices.size() - keep_count;
    for (size_t i = 0; i < to_remove; ++i) {
        buffers_[unused_indices[i]].buffer = cv::Mat(); // Release memory
    }

    // Compact the vector
    buffers_.erase(
        std::remove_if(buffers_.begin(), buffers_.end(),
                       [](const BufferEntry& e) { return e.buffer.empty(); }),
        buffers_.end());
}

void BufferPool::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    buffers_.clear();
}

BufferPool::BufferEntry BufferPool::createBuffer() {
    BufferEntry entry;
    entry.buffer = cv::Mat(height_, width_, type_);
    entry.in_use = false;
    entry.last_used_time = currentTimeMs();
    return entry;
}

uint64_t BufferPool::currentTimeMs() {
    using namespace std::chrono;
    return static_cast<uint64_t>(
        duration_cast<milliseconds>(
            steady_clock::now().time_since_epoch()
        ).count()
    );
}

bool BufferPool::isCompatible(const cv::Mat& buffer) const {
    return buffer.rows == height_ &&
           buffer.cols == width_ &&
           buffer.type() == type_;
}

// ScopedBuffer implementation

ScopedBuffer::ScopedBuffer(BufferPool& pool)
    : pool_(pool)
    , buffer_(pool.acquire()) {
}

ScopedBuffer::~ScopedBuffer() {
    if (!buffer_.empty()) {
        pool_.release(buffer_);
    }
}

} // namespace iris_sdk
