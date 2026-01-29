/**
 * @file buffer_pool.h
 * @brief Thread-safe buffer pool for cv::Mat reuse
 *
 * Reduces memory allocation overhead by pooling cv::Mat buffers.
 * Particularly useful in real-time video processing pipelines.
 */

#ifndef IRIS_SDK_BUFFER_POOL_H
#define IRIS_SDK_BUFFER_POOL_H

#include "export.h"

#include <opencv2/core.hpp>

#include <mutex>
#include <vector>
#include <memory>
#include <cstdint>
#include <functional>

namespace iris_sdk {

/**
 * @brief Statistics for buffer pool monitoring
 */
struct IRIS_SDK_EXPORT BufferPoolStats {
    size_t total_buffers = 0;      ///< Total buffers in pool
    size_t available_buffers = 0;   ///< Available (not in use) buffers
    size_t in_use_buffers = 0;      ///< Buffers currently acquired
    size_t total_memory_bytes = 0;  ///< Estimated total memory usage
    size_t acquire_count = 0;       ///< Total acquire() calls
    size_t release_count = 0;       ///< Total release() calls
    size_t allocation_count = 0;    ///< New allocations (pool miss)
};

/**
 * @brief Thread-safe pool for cv::Mat buffer reuse
 *
 * Usage:
 * @code
 * BufferPool pool;
 * pool.initialize(1920, 1080, CV_8UC4, 5); // Pool of 5 RGBA buffers
 *
 * // Acquire buffer from pool
 * cv::Mat buffer = pool.acquire();
 * // ... use buffer ...
 *
 * // Return to pool when done
 * pool.release(buffer);
 * @endcode
 *
 * The pool automatically creates new buffers if all are in use,
 * up to a configurable maximum.
 */
class IRIS_SDK_EXPORT BufferPool {
public:
    BufferPool();
    ~BufferPool();

    // Non-copyable, movable
    BufferPool(const BufferPool&) = delete;
    BufferPool& operator=(const BufferPool&) = delete;
    BufferPool(BufferPool&& other) noexcept;
    BufferPool& operator=(BufferPool&& other) noexcept;

    /**
     * @brief Initialize the buffer pool
     * @param width Buffer width in pixels
     * @param height Buffer height in pixels
     * @param type OpenCV type (e.g., CV_8UC3, CV_8UC4)
     * @param initial_size Initial number of buffers to pre-allocate
     * @param max_size Maximum pool size (0 = unlimited)
     * @return true if initialization succeeded
     */
    bool initialize(int width, int height, int type,
                    size_t initial_size = 3, size_t max_size = 10);

    /**
     * @brief Release all pool resources
     */
    void release();

    /**
     * @brief Check if pool is initialized
     * @return true if pool is ready for use
     */
    bool isInitialized() const;

    /**
     * @brief Acquire a buffer from the pool
     * @return cv::Mat buffer, or empty Mat if pool is exhausted
     *
     * If all buffers are in use and pool is at max capacity,
     * returns an empty cv::Mat. Always check with empty() or
     * use tryAcquire() for explicit error handling.
     */
    cv::Mat acquire();

    /**
     * @brief Try to acquire a buffer with explicit success/failure
     * @param[out] buffer Output buffer if successful
     * @return true if buffer was acquired, false otherwise
     */
    bool tryAcquire(cv::Mat& buffer);

    /**
     * @brief Return a buffer to the pool
     * @param buffer Buffer to return (will be set to empty after release)
     *
     * If the buffer doesn't belong to this pool (different size/type),
     * it will be silently ignored. The buffer reference is cleared
     * after release to prevent use-after-release.
     */
    void release(cv::Mat& buffer);

    /**
     * @brief Get current pool statistics
     * @return BufferPoolStats structure
     */
    BufferPoolStats getStats() const;

    /**
     * @brief Resize the pool
     * @param new_size New number of buffers
     *
     * If shrinking, in-use buffers will be released when returned.
     */
    void resize(size_t new_size);

    /**
     * @brief Remove unused buffers to free memory
     * @param keep_count Number of available buffers to keep
     */
    void trim(size_t keep_count = 1);

    /**
     * @brief Clear all buffers (both available and mark in-use as orphaned)
     */
    void clear();

    /**
     * @brief Get buffer dimensions
     * @return Width in pixels, or 0 if not initialized
     */
    int getWidth() const { return width_; }

    /**
     * @brief Get buffer height
     * @return Height in pixels, or 0 if not initialized
     */
    int getHeight() const { return height_; }

    /**
     * @brief Get buffer OpenCV type
     * @return CV type constant
     */
    int getType() const { return type_; }

private:
    /**
     * @brief Internal buffer tracking
     */
    struct BufferEntry {
        cv::Mat buffer;
        bool in_use = false;
        uint64_t last_used_time = 0;
    };

    /**
     * @brief Create a new buffer
     * @return New BufferEntry
     */
    BufferEntry createBuffer();

    /**
     * @brief Get current time in milliseconds (for LRU tracking)
     */
    static uint64_t currentTimeMs();

    /**
     * @brief Check if a buffer matches pool specifications
     */
    bool isCompatible(const cv::Mat& buffer) const;

    // Pool configuration
    int width_ = 0;
    int height_ = 0;
    int type_ = 0;
    size_t max_size_ = 10;
    bool initialized_ = false;

    // Buffer storage
    std::vector<BufferEntry> buffers_;

    // Statistics
    mutable size_t acquire_count_ = 0;
    mutable size_t release_count_ = 0;
    mutable size_t allocation_count_ = 0;

    // Synchronization
    mutable std::mutex mutex_;
};

/**
 * @brief RAII wrapper for automatic buffer release
 *
 * Usage:
 * @code
 * BufferPool pool;
 * // ... initialize ...
 *
 * {
 *     ScopedBuffer scoped(pool);
 *     if (scoped) {
 *         cv::Mat& buffer = scoped.get();
 *         // ... use buffer ...
 *     }
 * } // buffer automatically released
 * @endcode
 */
class IRIS_SDK_EXPORT ScopedBuffer {
public:
    /**
     * @brief Construct and acquire buffer from pool
     * @param pool Pool to acquire from
     */
    explicit ScopedBuffer(BufferPool& pool);

    /**
     * @brief Destruct and release buffer back to pool
     */
    ~ScopedBuffer();

    // Non-copyable, non-movable
    ScopedBuffer(const ScopedBuffer&) = delete;
    ScopedBuffer& operator=(const ScopedBuffer&) = delete;
    ScopedBuffer(ScopedBuffer&&) = delete;
    ScopedBuffer& operator=(ScopedBuffer&&) = delete;

    /**
     * @brief Check if buffer was successfully acquired
     * @return true if buffer is valid
     */
    explicit operator bool() const { return !buffer_.empty(); }

    /**
     * @brief Get the acquired buffer
     * @return Reference to cv::Mat buffer
     */
    cv::Mat& get() { return buffer_; }

    /**
     * @brief Get the acquired buffer (const)
     * @return Const reference to cv::Mat buffer
     */
    const cv::Mat& get() const { return buffer_; }

private:
    BufferPool& pool_;
    cv::Mat buffer_;
};

} // namespace iris_sdk

#endif // IRIS_SDK_BUFFER_POOL_H
