/**
 * @file gpu_profiler.cpp
 * @brief Implementation of GPU profiler using GL_EXT_disjoint_timer_query
 */

#include "iris_sdk/gpu/gpu_profiler.h"

// Only compile implementation on Android with GLES
#if defined(__ANDROID__) && defined(IRIS_SDK_HAS_GLES)

#include <EGL/egl.h>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cstring>

namespace iris_sdk {

namespace {

/**
 * @brief Check if OpenGL extension is supported
 */
bool isExtensionSupported(const char* extension) {
    const char* extensions = reinterpret_cast<const char*>(glGetString(GL_EXTENSIONS));
    if (!extensions) {
        return false;
    }

    const char* start = extensions;
    const char* end;

    while ((end = strchr(start, ' ')) != nullptr) {
        size_t len = end - start;
        if (strncmp(start, extension, len) == 0 && extension[len] == '\0') {
            return true;
        }
        start = end + 1;
    }

    // Check last extension (no trailing space)
    return strcmp(start, extension) == 0;
}

} // anonymous namespace

GPUProfiler::GPUProfiler() = default;

GPUProfiler::~GPUProfiler() {
    release();
}

bool GPUProfiler::initialize() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (initialized_) {
        return extension_supported_;
    }

    // Check for disjoint timer query extension
    extension_supported_ = isExtensionSupported("GL_EXT_disjoint_timer_query");

    if (!extension_supported_) {
        initialized_ = true;
        return false;
    }

    // Load extension functions
    glGenQueriesEXT_ = reinterpret_cast<PFNGLGENQUERIESEXTPROC>(
        eglGetProcAddress("glGenQueriesEXT"));
    glDeleteQueriesEXT_ = reinterpret_cast<PFNGLDELETEQUERIESEXTPROC>(
        eglGetProcAddress("glDeleteQueriesEXT"));
    glBeginQueryEXT_ = reinterpret_cast<PFNGLBEGINQUERYEXTPROC>(
        eglGetProcAddress("glBeginQueryEXT"));
    glEndQueryEXT_ = reinterpret_cast<PFNGLENDQUERYEXTPROC>(
        eglGetProcAddress("glEndQueryEXT"));
    glGetQueryObjectui64vEXT_ = reinterpret_cast<PFNGLGETQUERYOBJECTUI64VEXTPROC>(
        eglGetProcAddress("glGetQueryObjectui64vEXT"));
    glGetQueryObjectivEXT_ = reinterpret_cast<PFNGLGETQUERYOBJECTIVEXTPROC>(
        eglGetProcAddress("glGetQueryObjectivEXT"));

    // Verify all function pointers were loaded
    if (!glGenQueriesEXT_ || !glDeleteQueriesEXT_ ||
        !glBeginQueryEXT_ || !glEndQueryEXT_ ||
        !glGetQueryObjectui64vEXT_ || !glGetQueryObjectivEXT_) {
        extension_supported_ = false;
        initialized_ = true;
        return false;
    }

    // Pre-allocate query objects
    query_pool_.resize(QUERY_POOL_SIZE, 0);
    glGenQueriesEXT_(static_cast<GLsizei>(QUERY_POOL_SIZE), query_pool_.data());

    // Check for GL errors
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        extension_supported_ = false;
        query_pool_.clear();
        initialized_ = true;
        return false;
    }

    pool_index_ = 0;
    initialized_ = true;

    return true;
}

void GPUProfiler::release() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_) {
        return;
    }

    if (extension_supported_ && glDeleteQueriesEXT_ && !query_pool_.empty()) {
        glDeleteQueriesEXT_(static_cast<GLsizei>(query_pool_.size()), query_pool_.data());
    }

    query_pool_.clear();
    active_queries_.clear();
    in_flight_queries_.clear();  // B2 idx21
    measurements_.clear();
    last_results_.clear();
    pool_index_ = 0;
    initialized_ = false;
}

bool GPUProfiler::checkExtensionSupport() const {
    return extension_supported_;
}

void GPUProfiler::begin(const std::string& name) {
    if (!isEnabled()) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Check if there's already an active query for this name
    auto it = active_queries_.find(name);
    if (it != active_queries_.end()) {
        if (it->second.in_progress) {
            // Query already in progress, skip
            return;
        }
        // [B2 idx21] 같은 이름의 미수집(end 완료·결과 미수집) 쿼리가 남아 있으면,
        // 새 QueryInfo로 덮어쓰기 전에 이전 쿼리를 in-flight에서 회수한다.
        // (이전 구현은 그냥 덮어써, 해당 query_id가 영구히 in-flight로 남았다.)
        releaseQuery(it->second.query_id);
    }

    GLuint query = acquireQuery();
    if (query == 0) {
        return;
    }

    QueryInfo info;
    info.query_id = query;
    info.section_name = name;
    info.in_progress = true;
    info.result_available = false;

    active_queries_[name] = info;

    glBeginQueryEXT_(GL_TIME_ELAPSED_EXT, query);
}

void GPUProfiler::end(const std::string& name) {
    if (!isEnabled()) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = active_queries_.find(name);
    if (it == active_queries_.end() || !it->second.in_progress) {
        return;
    }

    glEndQueryEXT_(GL_TIME_ELAPSED_EXT);
    it->second.in_progress = false;
}

double GPUProfiler::getResult(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = last_results_.find(name);
    if (it == last_results_.end()) {
        return -1.0;
    }
    return it->second;
}

GPUMeasurement GPUProfiler::getMeasurement(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = measurements_.find(name);
    if (it == measurements_.end()) {
        return GPUMeasurement{};
    }
    return it->second;
}

void GPUProfiler::frameEnd() {
    if (!isEnabled()) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Check for GPU disjoint condition
    // If GPU was reset/disrupted, timing data is invalid
    GLint disjoint = 0;
    glGetIntegerv(GL_GPU_DISJOINT_EXT, &disjoint);

    if (disjoint) {
        // Discard all pending queries as data is invalid.
        // [B2 idx21] in_progress 여부와 무관하게 모든 활성 쿼리를 in-flight에서
        // 회수해 다음 프레임에 재사용 가능하게 한다(추적이 active_queries_.clear()로
        // 사라지므로, 회수하지 않으면 해당 쿼리가 영구히 in-flight로 남아 풀이 고갈됨).
        for (auto& [name, info] : active_queries_) {
            releaseQuery(info.query_id);
        }
        active_queries_.clear();
        return;
    }

    // Collect results from completed queries
    std::vector<std::string> completed;

    for (auto& [name, info] : active_queries_) {
        if (info.in_progress) {
            continue; // Query still running
        }

        if (tryCollectResult(info)) {
            completed.push_back(name);
        }
    }

    // Remove completed queries and return them to the pool
    for (const auto& name : completed) {
        auto it = active_queries_.find(name);
        if (it != active_queries_.end()) {
            releaseQuery(it->second.query_id);
            active_queries_.erase(it);
        }
    }
}

void GPUProfiler::reset() {
    std::lock_guard<std::mutex> lock(mutex_);

    measurements_.clear();
    last_results_.clear();
}

std::string GPUProfiler::generateReport() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::ostringstream oss;
    oss << "=== GPU Performance Report ===" << std::endl;

    if (!extension_supported_) {
        oss << "(GPU timing not supported on this device)" << std::endl;
        return oss.str();
    }

    if (measurements_.empty()) {
        oss << "(No GPU measurements recorded)" << std::endl;
        return oss.str();
    }

    // Sort by name
    std::vector<std::pair<std::string, GPUMeasurement>> sorted(
        measurements_.begin(), measurements_.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    oss << std::fixed << std::setprecision(2);

    for (const auto& [name, m] : sorted) {
        double min_display = (m.min_ms >= 1e9) ? 0.0 : m.min_ms;

        oss << name << ": "
            << "avg=" << m.avg() << "ms "
            << "min=" << min_display << "ms "
            << "max=" << m.max_ms << "ms "
            << "count=" << m.count
            << std::endl;
    }

    return oss.str();
}

GLuint GPUProfiler::acquireQuery() {
    if (query_pool_.empty()) {
        return 0;
    }

    // [B2 idx21] in-flight(미수집) 쿼리는 건너뛰는 라운드로빈.
    // 풀 전체를 한 바퀴 돌아도 가용 쿼리가 없으면 0을 반환해 측정을 스킵한다.
    // (이전 구현은 가용성 검사 없이 라운드로빈으로 반환해, GPU가 밀린 구간에서
    //  결과 수집 전 같은 쿼리를 재시작하며 직전 측정을 유실시켰다.)
    const size_t pool_size = query_pool_.size();
    for (size_t i = 0; i < pool_size; ++i) {
        GLuint query = query_pool_[pool_index_];
        pool_index_ = (pool_index_ + 1) % pool_size;
        if (in_flight_queries_.find(query) == in_flight_queries_.end()) {
            in_flight_queries_.insert(query);
            return query;
        }
    }
    return 0;  // 모든 쿼리가 in-flight — 이번 구간 측정 스킵
}

void GPUProfiler::releaseQuery(GLuint query_id) {
    // [B2 idx21] 결과를 수집했거나 폐기한 쿼리를 in-flight 집합에서 제거해
    // 다음 acquireQuery가 재사용할 수 있게 한다 (객체 자체는 풀에 남아 재활용).
    in_flight_queries_.erase(query_id);
}

bool GPUProfiler::tryCollectResult(QueryInfo& info) {
    GLint available = 0;
    glGetQueryObjectivEXT_(info.query_id, GL_QUERY_RESULT_AVAILABLE_EXT, &available);

    if (!available) {
        return false;
    }

    GLuint64 elapsed_ns = 0;
    glGetQueryObjectui64vEXT_(info.query_id, GL_QUERY_RESULT_EXT, &elapsed_ns);

    double elapsed_ms = static_cast<double>(elapsed_ns) / 1e6;

    // Update measurement
    GPUMeasurement& m = measurements_[info.section_name];
    m.total_ms += elapsed_ms;
    m.count++;
    m.min_ms = std::min(m.min_ms, elapsed_ms);
    m.max_ms = std::max(m.max_ms, elapsed_ms);

    // Store last result
    last_results_[info.section_name] = elapsed_ms;

    return true;
}

} // namespace iris_sdk

#endif // __ANDROID__ && IRIS_SDK_HAS_GLES
