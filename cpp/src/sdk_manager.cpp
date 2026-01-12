/**
 * @file sdk_manager.cpp
 * @brief SDK 관리자 싱글톤 구현
 */

#include "iris_sdk/sdk_manager.h"
#include "iris_sdk/frame_processor.h"
#include "iris_sdk/iris_detector.h"
#include "iris_sdk/lens_renderer.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <filesystem>
#include <mutex>

namespace iris_sdk {

// ============================================================
// 버전 정보
// ============================================================

namespace {
    constexpr const char* SDK_VERSION = "1.0.0";
    constexpr const char* SDK_BUILD_TYPE =
#ifdef NDEBUG
        "Release";
#else
        "Debug";
#endif

    constexpr size_t LOG_BUFFER_SIZE = 4096;
    constexpr const char TRUNCATION_MARKER[] = "...[TRUNCATED]";
}  // namespace

// ============================================================
// SDKManager::Impl 구현
// ============================================================

class SDKManager::Impl {
public:
    Impl()
        : state_(SDKState::Uninitialized)
        , log_level_(LogLevel::Info) {}

    ~Impl() {
        shutdown();
    }

    // ========================================
    // 라이프사이클
    // ========================================

    bool initialize(const SDKConfig& config) {
        std::lock_guard<std::mutex> lock(mutex_);

        // 이미 초기화된 경우
        if (state_ != SDKState::Uninitialized) {
            logInternal(LogLevel::Warning, "SDKManager",
                       "Already initialized (state=%d)", static_cast<int>(state_.load()));
            return state_ == SDKState::Ready;
        }

        state_ = SDKState::Initializing;

        // 로깅 설정 먼저 적용 (이후 로그 출력을 위해)
        log_level_ = config.log_level;
        if (config.log_callback) {
            log_callback_ = config.log_callback;
        }

        // 모델 경로 검증
        if (config.model_path.empty()) {
            logInternal(LogLevel::Error, "SDKManager", "Model path is empty");
            state_ = SDKState::Error;
            return false;
        }

        if (!validateModelPath(config.model_path)) {
            logInternal(LogLevel::Error, "SDKManager",
                       "Invalid model path: %s", config.model_path.c_str());
            state_ = SDKState::Error;
            return false;
        }

        // 설정 저장
        config_ = config;

        state_ = SDKState::Ready;
        logInternal(LogLevel::Info, "SDKManager",
                   "SDK initialized successfully v%s (%s)",
                   SDK_VERSION, SDK_BUILD_TYPE);

        return true;
    }

    bool initialize(const std::string& model_path) {
        SDKConfig config;
        config.model_path = model_path;
        return initialize(config);
    }

    void shutdown() {
        std::lock_guard<std::mutex> lock(mutex_);

        if (state_ == SDKState::Uninitialized) {
            return;  // 이미 종료됨
        }

        logInternal(LogLevel::Info, "SDKManager", "Shutting down SDK...");

        // 설정 초기화
        config_ = SDKConfig{};
        log_callback_ = nullptr;
        log_level_ = LogLevel::Info;

        state_ = SDKState::Uninitialized;
    }

    SDKState getState() const noexcept {
        return state_.load();
    }

    // ========================================
    // 팩토리 메서드
    // ========================================

    std::unique_ptr<FrameProcessor> createFrameProcessor() {
        // 스레드 안전: 뮤텍스로 config_ 접근 보호
        std::lock_guard<std::mutex> lock(mutex_);

        if (!isReadyUnlocked()) {
            logInternalUnlocked(LogLevel::Error, "SDKManager",
                               "SDK not initialized, cannot create FrameProcessor");
            return nullptr;
        }

        // config_를 안전하게 사용 (lock 보유 중)
        auto processor = std::make_unique<FrameProcessor>();
        if (!processor->initialize(config_.model_path, config_.detector_type)) {
            logInternalUnlocked(LogLevel::Error, "SDKManager",
                               "Failed to initialize FrameProcessor");
            return nullptr;
        }

        logInternalUnlocked(LogLevel::Debug, "SDKManager", "FrameProcessor created");
        return processor;
    }

    std::unique_ptr<IrisDetector> createDetector(DetectorType type) {
        // 스레드 안전: 뮤텍스로 config_ 접근 보호
        std::lock_guard<std::mutex> lock(mutex_);

        if (!isReadyUnlocked()) {
            logInternalUnlocked(LogLevel::Error, "SDKManager",
                               "SDK not initialized, cannot create IrisDetector");
            return nullptr;
        }

        auto detector = detail::createDetector(type);
        if (!detector) {
            logInternalUnlocked(LogLevel::Error, "SDKManager",
                               "Failed to create IrisDetector (type=%d)",
                               static_cast<int>(type));
            return nullptr;
        }

        if (!detector->initialize(config_.model_path)) {
            logInternalUnlocked(LogLevel::Error, "SDKManager",
                               "Failed to initialize IrisDetector");
            return nullptr;
        }

        logInternalUnlocked(LogLevel::Debug, "SDKManager",
                           "IrisDetector created (type=%d)", static_cast<int>(type));
        return detector;
    }

    std::unique_ptr<LensRenderer> createRenderer() {
        // 스레드 안전: 뮤텍스로 config_ 접근 보호
        std::lock_guard<std::mutex> lock(mutex_);

        if (!isReadyUnlocked()) {
            logInternalUnlocked(LogLevel::Error, "SDKManager",
                               "SDK not initialized, cannot create LensRenderer");
            return nullptr;
        }

        auto renderer = std::make_unique<LensRenderer>();
        if (!renderer->initialize()) {
            logInternalUnlocked(LogLevel::Error, "SDKManager",
                               "Failed to initialize LensRenderer");
            return nullptr;
        }

        logInternalUnlocked(LogLevel::Debug, "SDKManager", "LensRenderer created");
        return renderer;
    }

    // ========================================
    // 설정 접근
    // ========================================

    SDKConfig getConfig() const {
        // 스레드 안전: 복사본 반환으로 데이터 레이스 방지
        std::lock_guard<std::mutex> lock(mutex_);
        return config_;
    }

    void setLogLevel(LogLevel level) {
        log_level_ = level;
    }

    void setLogCallback(LogCallback callback) {
        std::lock_guard<std::mutex> lock(log_mutex_);
        log_callback_ = callback;
    }

    // ========================================
    // 로깅
    // ========================================

    void log(LogLevel level, const char* tag, const char* format, ...) {
        va_list args;
        va_start(args, format);
        logV(level, tag, format, args);
        va_end(args);
    }

    void logV(LogLevel level, const char* tag, const char* format, va_list args) {
        // 레벨 필터링
        if (level < log_level_.load()) {
            return;
        }

        // 메시지 포맷
        char buffer[LOG_BUFFER_SIZE];
        vsnprintf(buffer, sizeof(buffer), format, args);

        // 콜백 호출 또는 기본 출력
        std::lock_guard<std::mutex> lock(log_mutex_);
        if (log_callback_) {
            log_callback_(level, tag, buffer);
        } else {
            outputToStderr(level, tag, buffer);
        }
    }

    // ========================================
    // 내부 헬퍼
    // ========================================

private:
    // 뮤텍스 없이 호출 (호출자가 lock 보유)
    bool isReadyUnlocked() const noexcept {
        return state_.load() == SDKState::Ready;
    }

    // 뮤텍스 보유 상태에서 로깅 (mutex_ 재획득 방지)
    void logInternalUnlocked(LogLevel level, const char* tag, const char* format, ...) {
        if (level < log_level_.load()) {
            return;
        }

        va_list args;
        va_start(args, format);

        char buffer[LOG_BUFFER_SIZE];
        int written = vsnprintf(buffer, sizeof(buffer), format, args);
        va_end(args);

        // 메시지가 잘렸으면 표시
        if (written >= static_cast<int>(LOG_BUFFER_SIZE)) {
            constexpr size_t marker_len = sizeof(TRUNCATION_MARKER);
            if (LOG_BUFFER_SIZE > marker_len) {
                memcpy(buffer + LOG_BUFFER_SIZE - marker_len,
                       TRUNCATION_MARKER, marker_len);
            }
        }

        // 로깅 뮤텍스로 콜백 보호
        std::lock_guard<std::mutex> log_lock(log_mutex_);
        if (log_callback_) {
            log_callback_(level, tag, buffer);
        } else {
            outputToStderr(level, tag, buffer);
        }
    }

    static bool validateModelPath(const std::string& path) {
        try {
            // 디렉토리 존재 확인
            if (!std::filesystem::exists(path)) {
                return false;
            }

            // 디렉토리인지 확인
            if (!std::filesystem::is_directory(path)) {
                // 파일 경로일 수도 있음 (단일 모델 파일)
                return std::filesystem::is_regular_file(path);
            }

            return true;
        } catch (const std::filesystem::filesystem_error&) {
            return false;
        }
    }

    void logInternal(LogLevel level, const char* tag, const char* format, ...) {
        va_list args;
        va_start(args, format);
        logV(level, tag, format, args);
        va_end(args);
    }

    void outputToStderr(LogLevel level, const char* tag, const char* message) {
        const char* level_str = "?";
        switch (level) {
            case LogLevel::Verbose: level_str = "V"; break;
            case LogLevel::Debug:   level_str = "D"; break;
            case LogLevel::Info:    level_str = "I"; break;
            case LogLevel::Warning: level_str = "W"; break;
            case LogLevel::Error:   level_str = "E"; break;
            default: break;
        }
        fprintf(stderr, "[%s] %s: %s\n", level_str, tag, message);
    }

    // ========================================
    // 멤버 변수
    // ========================================

    mutable std::mutex mutex_;              ///< 라이프사이클 뮤텍스 (const 메서드 지원)
    mutable std::mutex log_mutex_;          ///< 로깅 뮤텍스 (const 메서드 지원)
    std::atomic<SDKState> state_;           ///< SDK 상태 (원자적)
    SDKConfig config_;                      ///< 설정
    std::atomic<LogLevel> log_level_;       ///< 로그 레벨 (원자적)
    LogCallback log_callback_;              ///< 로그 콜백
};

// ============================================================
// SDKManager 공개 인터페이스 구현
// ============================================================

SDKManager& SDKManager::getInstance() {
    // Meyer's Singleton: C++11 이상에서 스레드 안전
    static SDKManager instance;
    return instance;
}

SDKManager::SDKManager()
    : impl_(std::make_unique<Impl>()) {}

SDKManager::~SDKManager() = default;

bool SDKManager::initialize(const SDKConfig& config) {
    return impl_->initialize(config);
}

bool SDKManager::initialize(const std::string& model_path) {
    return impl_->initialize(model_path);
}

void SDKManager::shutdown() {
    impl_->shutdown();
}

SDKState SDKManager::getState() const noexcept {
    return impl_->getState();
}

std::unique_ptr<FrameProcessor> SDKManager::createFrameProcessor() {
    return impl_->createFrameProcessor();
}

std::unique_ptr<IrisDetector> SDKManager::createDetector(DetectorType type) {
    return impl_->createDetector(type);
}

std::unique_ptr<LensRenderer> SDKManager::createRenderer() {
    return impl_->createRenderer();
}

SDKConfig SDKManager::getConfig() const {
    return impl_->getConfig();
}

void SDKManager::setLogLevel(LogLevel level) {
    impl_->setLogLevel(level);
}

void SDKManager::setLogCallback(LogCallback callback) {
    impl_->setLogCallback(callback);
}

void SDKManager::log(LogLevel level, const char* tag, const char* format, ...) {
    va_list args;
    va_start(args, format);
    impl_->logV(level, tag, format, args);
    va_end(args);
}

void SDKManager::logV(LogLevel level, const char* tag,
                      const char* format, va_list args) {
    impl_->logV(level, tag, format, args);
}

const char* SDKManager::getVersion() noexcept {
    return SDK_VERSION;
}

const char* SDKManager::getBuildInfo() noexcept {
    // C++11 Magic Statics: 스레드 안전한 정적 초기화
    static const std::string build_info = []() {
        char buffer[256];
        snprintf(buffer, sizeof(buffer),
                 "IrisLensSDK v%s (%s, %s %s)",
                 SDK_VERSION, SDK_BUILD_TYPE, __DATE__, __TIME__);
        return std::string(buffer);
    }();
    return build_info.c_str();
}

} // namespace iris_sdk
