/**
 * @file profiler.h
 * @brief Performance profiling utilities for IrisLensSDK
 *
 * Thread-safe CPU profiler with RAII support for measuring
 * execution times across different code sections.
 */

#ifndef IRIS_SDK_PROFILER_H
#define IRIS_SDK_PROFILER_H

#include "export.h"

#include <chrono>
#include <string>
#include <unordered_map>
#include <mutex>
#include <cstdint>

namespace iris_sdk {

/**
 * @brief Performance measurement data for a profiled section
 */
struct IRIS_SDK_EXPORT Measurement {
    double min_ms = std::numeric_limits<double>::max();  ///< Minimum time in milliseconds
    double max_ms = 0.0;                                   ///< Maximum time in milliseconds
    double total_ms = 0.0;                                 ///< Total accumulated time in milliseconds
    uint64_t count = 0;                                    ///< Number of measurements

    /**
     * @brief Calculate average time in milliseconds
     * @return Average time or 0.0 if no measurements
     */
    double avg() const {
        return count > 0 ? total_ms / static_cast<double>(count) : 0.0;
    }

    /**
     * @brief Reset all measurement values
     */
    void reset() {
        min_ms = std::numeric_limits<double>::max();
        max_ms = 0.0;
        total_ms = 0.0;
        count = 0;
    }
};

/**
 * @brief Thread-safe CPU profiler singleton
 *
 * Usage:
 * @code
 * // Manual timing
 * Profiler::getInstance().begin("section_name");
 * // ... code to measure ...
 * Profiler::getInstance().end("section_name");
 *
 * // RAII timing (recommended)
 * {
 *     ProfileScope scope("section_name");
 *     // ... code to measure ...
 * } // automatically records time when scope ends
 *
 * // Get report
 * std::string report = Profiler::getInstance().generateReport();
 * @endcode
 */
class IRIS_SDK_EXPORT Profiler {
public:
    /**
     * @brief Get singleton instance
     * @return Reference to the global Profiler instance
     */
    static Profiler& getInstance();

    /**
     * @brief Begin timing a named section
     * @param name Section name (must match corresponding end() call)
     */
    void begin(const std::string& name);

    /**
     * @brief End timing a named section and record measurement
     * @param name Section name (must match corresponding begin() call)
     */
    void end(const std::string& name);

    /**
     * @brief Get measurement data for a section
     * @param name Section name
     * @return Measurement data, or empty Measurement if not found
     */
    Measurement getMeasurement(const std::string& name) const;

    /**
     * @brief Reset all measurements
     */
    void reset();

    /**
     * @brief Enable or disable profiling
     * @param enabled True to enable, false to disable
     *
     * When disabled, begin()/end() calls are no-ops for minimal overhead.
     */
    void setEnabled(bool enabled);

    /**
     * @brief Check if profiling is enabled
     * @return True if profiling is enabled
     */
    bool isEnabled() const;

    /**
     * @brief Generate a formatted report of all measurements
     * @return Human-readable report string
     *
     * Report format:
     * @code
     * === Performance Report ===
     * section_name: avg=5.23ms min=4.10ms max=6.50ms count=100
     * other_section: avg=1.05ms min=0.90ms max=1.20ms count=50
     * @endcode
     */
    std::string generateReport() const;

    // Delete copy/move operations for singleton
    Profiler(const Profiler&) = delete;
    Profiler& operator=(const Profiler&) = delete;
    Profiler(Profiler&&) = delete;
    Profiler& operator=(Profiler&&) = delete;

private:
    Profiler() = default;
    ~Profiler() = default;

    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;

    mutable std::mutex mutex_;
    std::unordered_map<std::string, Measurement> measurements_;
    std::unordered_map<std::string, TimePoint> start_times_;
    bool enabled_ = true;
};

/**
 * @brief RAII helper for automatic profiling scope
 *
 * Records execution time from construction to destruction.
 *
 * Usage:
 * @code
 * void myFunction() {
 *     ProfileScope scope("myFunction");
 *     // ... function body ...
 * } // time automatically recorded
 * @endcode
 */
class IRIS_SDK_EXPORT ProfileScope {
public:
    /**
     * @brief Construct and begin timing
     * @param name Section name for the profiler
     */
    explicit ProfileScope(const std::string& name);

    /**
     * @brief Destruct and end timing
     */
    ~ProfileScope();

    // Non-copyable, non-movable
    ProfileScope(const ProfileScope&) = delete;
    ProfileScope& operator=(const ProfileScope&) = delete;
    ProfileScope(ProfileScope&&) = delete;
    ProfileScope& operator=(ProfileScope&&) = delete;

private:
    std::string name_;
};

} // namespace iris_sdk

/**
 * @brief Convenience macro for profiling a scope with a custom name
 * @param name The name to use for the profiled section
 */
#define PROFILE_SCOPE(name) \
    iris_sdk::ProfileScope _profile_scope_##__LINE__(name)

/**
 * @brief Convenience macro for profiling the current function
 *
 * Uses __FUNCTION__ or __func__ as the section name.
 */
#define PROFILE_FUNCTION() \
    iris_sdk::ProfileScope _profile_scope_func_(__func__)

#endif // IRIS_SDK_PROFILER_H
