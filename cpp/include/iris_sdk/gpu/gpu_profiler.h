/**
 * @file gpu_profiler.h
 * @brief GPU performance profiling using GL_EXT_disjoint_timer_query
 *
 * Provides GPU timing measurements for OpenGL ES environments.
 * Safely handles devices that don't support the timer query extension.
 */

#ifndef IRIS_SDK_GPU_PROFILER_H
#define IRIS_SDK_GPU_PROFILER_H

#include "../export.h"

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <cstdint>
#include <vector>

// Only enable GPU profiling on Android with GLES
#if defined(__ANDROID__) && defined(IRIS_SDK_HAS_GLES)

#include <GLES3/gl31.h>
#include <GLES2/gl2ext.h>

namespace iris_sdk {

/**
 * @brief GPU timing measurement data
 */
struct IRIS_SDK_EXPORT GPUMeasurement {
    double min_ms = 1e9;        ///< Minimum GPU time in milliseconds
    double max_ms = 0.0;        ///< Maximum GPU time in milliseconds
    double total_ms = 0.0;      ///< Total accumulated GPU time in milliseconds
    uint64_t count = 0;         ///< Number of measurements

    /**
     * @brief Calculate average GPU time in milliseconds
     * @return Average time or 0.0 if no measurements
     */
    double avg() const {
        return count > 0 ? total_ms / static_cast<double>(count) : 0.0;
    }

    /**
     * @brief Reset all measurement values
     */
    void reset() {
        min_ms = 1e9;
        max_ms = 0.0;
        total_ms = 0.0;
        count = 0;
    }
};

/**
 * @brief GPU profiler using GL_EXT_disjoint_timer_query
 *
 * Usage:
 * @code
 * GPUProfiler profiler;
 * if (profiler.initialize()) {
 *     profiler.begin("shader_pass");
 *     // ... GPU work ...
 *     profiler.end("shader_pass");
 *
 *     // At frame end, check for disjoint and collect results
 *     profiler.frameEnd();
 *
 *     // Get timing (may be from previous frame due to async nature)
 *     double time_ms = profiler.getResult("shader_pass");
 * }
 * @endcode
 *
 * @note GPU timing is asynchronous. Results from end() may not be
 *       available until the next frame or later.
 */
class IRIS_SDK_EXPORT GPUProfiler {
public:
    GPUProfiler();
    ~GPUProfiler();

    // Non-copyable, non-movable
    GPUProfiler(const GPUProfiler&) = delete;
    GPUProfiler& operator=(const GPUProfiler&) = delete;
    GPUProfiler(GPUProfiler&&) = delete;
    GPUProfiler& operator=(GPUProfiler&&) = delete;

    /**
     * @brief Initialize the GPU profiler
     * @return true if timer query extension is supported, false otherwise
     *
     * If the extension is not supported, all timing calls become no-ops.
     */
    bool initialize();

    /**
     * @brief Release GPU resources
     */
    void release();

    /**
     * @brief Check if GL_EXT_disjoint_timer_query is supported
     * @return true if extension is available and profiler is initialized
     */
    bool checkExtensionSupport() const;

    /**
     * @brief Check if profiler is initialized and ready
     * @return true if profiler can record GPU times
     */
    bool isInitialized() const { return initialized_ && extension_supported_; }

    /**
     * @brief Begin GPU timing for a named section
     * @param name Section name
     *
     * Safe to call even if extension is not supported (becomes no-op).
     */
    void begin(const std::string& name);

    /**
     * @brief End GPU timing for a named section
     * @param name Section name (must match corresponding begin() call)
     *
     * Safe to call even if extension is not supported (becomes no-op).
     */
    void end(const std::string& name);

    /**
     * @brief Get the last recorded GPU time for a section
     * @param name Section name
     * @return GPU time in milliseconds, or -1.0 if not available
     *
     * Due to the asynchronous nature of GPU queries, this may return
     * the result from a previous frame.
     */
    double getResult(const std::string& name) const;

    /**
     * @brief Get accumulated measurement data for a section
     * @param name Section name
     * @return GPUMeasurement with statistics
     */
    GPUMeasurement getMeasurement(const std::string& name) const;

    /**
     * @brief Call at the end of each frame to check disjoint and collect results
     *
     * This method:
     * 1. Checks GPU_DISJOINT_EXT flag (if set, timing data is invalid)
     * 2. Collects available query results
     * 3. Manages query object recycling
     */
    void frameEnd();

    /**
     * @brief Reset all measurements
     */
    void reset();

    /**
     * @brief Generate a formatted report of GPU timings
     * @return Human-readable report string
     */
    std::string generateReport() const;

    /**
     * @brief Enable or disable GPU profiling
     * @param enabled true to enable, false to disable
     */
    void setEnabled(bool enabled) { enabled_ = enabled; }

    /**
     * @brief Check if GPU profiling is enabled
     * @return true if enabled
     */
    bool isEnabled() const { return enabled_ && extension_supported_; }

private:
    /**
     * @brief Query object wrapper with state tracking
     */
    struct QueryInfo {
        GLuint query_id = 0;
        std::string section_name;
        bool in_progress = false;
        bool result_available = false;
    };

    /**
     * @brief Acquire a query object from the pool
     * @return Query ID, or 0 if failed
     */
    GLuint acquireQuery();

    /**
     * @brief Return a query object to the pool
     * @param query_id Query to return
     */
    void releaseQuery(GLuint query_id);

    /**
     * @brief Try to collect result from a pending query
     * @param info Query info
     * @return true if result was collected
     */
    bool tryCollectResult(QueryInfo& info);

    // Extension support flag
    bool extension_supported_ = false;
    bool initialized_ = false;
    bool enabled_ = true;

    // Query object pool (reuse to avoid frequent allocation)
    std::vector<GLuint> query_pool_;
    size_t pool_index_ = 0;
    static constexpr size_t QUERY_POOL_SIZE = 32;

    // [B2 idx21] 아직 결과를 수집하지 못한 in-flight 쿼리 id 집합.
    // acquireQuery가 이 집합에 있는 쿼리를 건너뛰어, GPU가 밀려 결과 수집 전에
    // 같은 쿼리를 glBeginQueryEXT로 재시작해 이전 측정을 덮어쓰는 것을 막는다.
    std::unordered_set<GLuint> in_flight_queries_;

    // Active queries (currently in-flight)
    std::unordered_map<std::string, QueryInfo> active_queries_;

    // Measurement results
    std::unordered_map<std::string, GPUMeasurement> measurements_;
    std::unordered_map<std::string, double> last_results_;

    // Synchronization
    mutable std::mutex mutex_;

    // Extension function pointers
    PFNGLGENQUERIESEXTPROC glGenQueriesEXT_ = nullptr;
    PFNGLDELETEQUERIESEXTPROC glDeleteQueriesEXT_ = nullptr;
    PFNGLBEGINQUERYEXTPROC glBeginQueryEXT_ = nullptr;
    PFNGLENDQUERYEXTPROC glEndQueryEXT_ = nullptr;
    PFNGLGETQUERYOBJECTUI64VEXTPROC glGetQueryObjectui64vEXT_ = nullptr;
    PFNGLGETQUERYOBJECTIVEXTPROC glGetQueryObjectivEXT_ = nullptr;
};

} // namespace iris_sdk

#else // !(__ANDROID__ && IRIS_SDK_HAS_GLES)

// Stub implementation for non-Android/GLES platforms
namespace iris_sdk {

struct IRIS_SDK_EXPORT GPUMeasurement {
    double min_ms = 0.0;
    double max_ms = 0.0;
    double total_ms = 0.0;
    uint64_t count = 0;
    double avg() const { return 0.0; }
    void reset() {}
};

class IRIS_SDK_EXPORT GPUProfiler {
public:
    GPUProfiler() = default;
    ~GPUProfiler() = default;

    bool initialize() { return false; }
    void release() {}
    bool checkExtensionSupport() const { return false; }
    bool isInitialized() const { return false; }

    void begin(const std::string&) {}
    void end(const std::string&) {}
    double getResult(const std::string&) const { return -1.0; }
    GPUMeasurement getMeasurement(const std::string&) const { return GPUMeasurement{}; }

    void frameEnd() {}
    void reset() {}
    std::string generateReport() const { return "GPU profiling not available on this platform"; }

    void setEnabled(bool) {}
    bool isEnabled() const { return false; }
};

} // namespace iris_sdk

#endif // __ANDROID__ && IRIS_SDK_HAS_GLES

#endif // IRIS_SDK_GPU_PROFILER_H
