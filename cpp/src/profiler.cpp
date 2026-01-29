/**
 * @file profiler.cpp
 * @brief Implementation of the Profiler class
 */

#include "iris_sdk/profiler.h"

#include <sstream>
#include <iomanip>
#include <algorithm>
#include <vector>

namespace iris_sdk {

Profiler& Profiler::getInstance() {
    static Profiler instance;
    return instance;
}

void Profiler::begin(const std::string& name) {
    if (!enabled_) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    start_times_[name] = Clock::now();
}

void Profiler::end(const std::string& name) {
    if (!enabled_) {
        return;
    }

    auto end_time = Clock::now();

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = start_times_.find(name);
    if (it == start_times_.end()) {
        // begin() was not called for this name
        return;
    }

    auto duration = std::chrono::duration<double, std::milli>(end_time - it->second);
    double elapsed_ms = duration.count();

    // Update measurement
    Measurement& m = measurements_[name];
    m.total_ms += elapsed_ms;
    m.count++;
    m.min_ms = std::min(m.min_ms, elapsed_ms);
    m.max_ms = std::max(m.max_ms, elapsed_ms);

    // Remove start time
    start_times_.erase(it);
}

Measurement Profiler::getMeasurement(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = measurements_.find(name);
    if (it == measurements_.end()) {
        return Measurement{};
    }
    return it->second;
}

void Profiler::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    measurements_.clear();
    start_times_.clear();
}

void Profiler::setEnabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    enabled_ = enabled;
}

bool Profiler::isEnabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return enabled_;
}

std::string Profiler::generateReport() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::ostringstream oss;
    oss << "=== Performance Report ===" << std::endl;

    if (measurements_.empty()) {
        oss << "(No measurements recorded)" << std::endl;
        return oss.str();
    }

    // Sort by name for consistent output
    std::vector<std::pair<std::string, Measurement>> sorted_measurements(
        measurements_.begin(), measurements_.end());
    std::sort(sorted_measurements.begin(), sorted_measurements.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    oss << std::fixed << std::setprecision(2);

    for (const auto& [name, m] : sorted_measurements) {
        double avg_ms = m.count > 0 ? m.total_ms / static_cast<double>(m.count) : 0.0;
        double min_display = (m.min_ms == std::numeric_limits<double>::max()) ? 0.0 : m.min_ms;

        oss << name << ": "
            << "avg=" << avg_ms << "ms "
            << "min=" << min_display << "ms "
            << "max=" << m.max_ms << "ms "
            << "count=" << m.count
            << std::endl;
    }

    return oss.str();
}

// ProfileScope implementation

ProfileScope::ProfileScope(const std::string& name)
    : name_(name) {
    Profiler::getInstance().begin(name_);
}

ProfileScope::~ProfileScope() {
    Profiler::getInstance().end(name_);
}

} // namespace iris_sdk
