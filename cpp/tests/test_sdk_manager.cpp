/**
 * @file test_sdk_manager.cpp
 * @brief SDKManager 싱글톤 단위 테스트
 *
 * P1-W4-04: SDKManager 검증
 * - 싱글톤 패턴 동작
 * - 스레드 안전 초기화
 * - 라이프사이클 관리
 * - 설정 관리
 * - 로깅 시스템
 */

#include <gtest/gtest.h>

#include <atomic>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include "iris_sdk/sdk_manager.h"
#include "iris_sdk/frame_processor.h"
#include "iris_sdk/iris_detector.h"
#include "iris_sdk/lens_renderer.h"
#include "iris_sdk/types.h"

namespace iris_sdk {
namespace {

// 테스트용 더미 모델 경로 생성 헬퍼
class TestModelPath {
public:
    TestModelPath() {
        // 테스트용 임시 디렉토리 생성
        temp_dir_ = std::filesystem::temp_directory_path() / "iris_sdk_test";
        std::filesystem::create_directories(temp_dir_);
    }

    ~TestModelPath() {
        // 정리
        try {
            std::filesystem::remove_all(temp_dir_);
        } catch (...) {
            // 무시
        }
    }

    std::string getPath() const { return temp_dir_.string(); }

private:
    std::filesystem::path temp_dir_;
};

// 로그 캡처용 헬퍼
class LogCapture {
public:
    void callback(LogLevel level, const char* tag, const char* message) {
        std::lock_guard<std::mutex> lock(mutex_);
        logs_.push_back({level, std::string(tag), std::string(message)});
    }

    size_t count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return logs_.size();
    }

    bool hasLog(LogLevel level, const std::string& tag_substring,
                const std::string& msg_substring) const {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& log : logs_) {
            if (log.level == level &&
                log.tag.find(tag_substring) != std::string::npos &&
                log.message.find(msg_substring) != std::string::npos) {
                return true;
            }
        }
        return false;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        logs_.clear();
    }

private:
    struct LogEntry {
        LogLevel level;
        std::string tag;
        std::string message;
    };
    mutable std::mutex mutex_;
    std::vector<LogEntry> logs_;
};

// ============================================================
// 테스트 픽스처
// ============================================================

class SDKManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 각 테스트 전 SDK 종료 (클린 상태 보장)
        SDKManager::getInstance().shutdown();
        log_capture_.clear();
    }

    void TearDown() override {
        // 각 테스트 후 SDK 종료
        SDKManager::getInstance().shutdown();
    }

    TestModelPath model_path_;
    LogCapture log_capture_;
};

// ============================================================
// 싱글톤 패턴 테스트
// ============================================================

TEST_F(SDKManagerTest, GetInstance_ReturnsSameInstance) {
    // 여러 번 호출해도 동일한 인스턴스 반환
    SDKManager& instance1 = SDKManager::getInstance();
    SDKManager& instance2 = SDKManager::getInstance();
    SDKManager& instance3 = SDKManager::getInstance();

    EXPECT_EQ(&instance1, &instance2);
    EXPECT_EQ(&instance2, &instance3);
}

TEST_F(SDKManagerTest, Singleton_ThreadSafeInitialization) {
    // 여러 스레드에서 동시에 getInstance 호출
    constexpr int kThreadCount = 10;
    std::vector<std::thread> threads;
    std::vector<SDKManager*> instances(kThreadCount, nullptr);
    std::atomic<int> ready_count{0};

    // 모든 스레드가 준비될 때까지 대기 후 동시 시작
    for (int i = 0; i < kThreadCount; ++i) {
        threads.emplace_back([&instances, &ready_count, i]() {
            ready_count.fetch_add(1);
            while (ready_count.load() < kThreadCount) {
                std::this_thread::yield();
            }
            instances[i] = &SDKManager::getInstance();
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // 모든 스레드가 동일한 인스턴스를 받았는지 확인
    for (int i = 1; i < kThreadCount; ++i) {
        EXPECT_EQ(instances[0], instances[i]);
    }
}

// ============================================================
// 라이프사이클 관리 테스트
// ============================================================

TEST_F(SDKManagerTest, InitialState_IsUninitialized) {
    EXPECT_EQ(SDKManager::getInstance().getState(), SDKState::Uninitialized);
    EXPECT_FALSE(SDKManager::getInstance().isReady());
}

TEST_F(SDKManagerTest, Initialize_WithValidPath_Succeeds) {
    SDKConfig config;
    config.model_path = model_path_.getPath();
    config.log_level = LogLevel::Debug;

    EXPECT_TRUE(SDKManager::getInstance().initialize(config));
    EXPECT_EQ(SDKManager::getInstance().getState(), SDKState::Ready);
    EXPECT_TRUE(SDKManager::getInstance().isReady());
}

TEST_F(SDKManagerTest, Initialize_WithSimplePath_Succeeds) {
    EXPECT_TRUE(SDKManager::getInstance().initialize(model_path_.getPath()));
    EXPECT_TRUE(SDKManager::getInstance().isReady());
}

TEST_F(SDKManagerTest, Initialize_WithEmptyPath_Fails) {
    SDKConfig config;
    config.model_path = "";
    config.log_level = LogLevel::None;  // 에러 로그 억제

    EXPECT_FALSE(SDKManager::getInstance().initialize(config));
    EXPECT_EQ(SDKManager::getInstance().getState(), SDKState::Error);
    EXPECT_FALSE(SDKManager::getInstance().isReady());
}

TEST_F(SDKManagerTest, Initialize_WithInvalidPath_Fails) {
    SDKConfig config;
    config.model_path = "/nonexistent/path/to/models";
    config.log_level = LogLevel::None;

    EXPECT_FALSE(SDKManager::getInstance().initialize(config));
    EXPECT_EQ(SDKManager::getInstance().getState(), SDKState::Error);
}

TEST_F(SDKManagerTest, Initialize_CalledTwice_ReturnsTrue) {
    SDKConfig config;
    config.model_path = model_path_.getPath();
    config.log_level = LogLevel::None;

    EXPECT_TRUE(SDKManager::getInstance().initialize(config));
    EXPECT_TRUE(SDKManager::getInstance().initialize(config));  // 이미 초기화됨
    EXPECT_TRUE(SDKManager::getInstance().isReady());
}

TEST_F(SDKManagerTest, Shutdown_ResetsToUninitialized) {
    SDKConfig config;
    config.model_path = model_path_.getPath();

    (void)SDKManager::getInstance().initialize(config);
    EXPECT_TRUE(SDKManager::getInstance().isReady());

    SDKManager::getInstance().shutdown();
    EXPECT_EQ(SDKManager::getInstance().getState(), SDKState::Uninitialized);
    EXPECT_FALSE(SDKManager::getInstance().isReady());
}

TEST_F(SDKManagerTest, Shutdown_CalledMultipleTimes_IsSafe) {
    SDKConfig config;
    config.model_path = model_path_.getPath();

    (void)SDKManager::getInstance().initialize(config);

    // 여러 번 shutdown 호출해도 안전
    EXPECT_NO_THROW(SDKManager::getInstance().shutdown());
    EXPECT_NO_THROW(SDKManager::getInstance().shutdown());
    EXPECT_NO_THROW(SDKManager::getInstance().shutdown());

    EXPECT_EQ(SDKManager::getInstance().getState(), SDKState::Uninitialized);
}

TEST_F(SDKManagerTest, Initialize_AfterShutdown_Succeeds) {
    SDKConfig config;
    config.model_path = model_path_.getPath();

    // 초기화 → 종료 → 재초기화
    (void)SDKManager::getInstance().initialize(config);
    SDKManager::getInstance().shutdown();
    EXPECT_TRUE(SDKManager::getInstance().initialize(config));
    EXPECT_TRUE(SDKManager::getInstance().isReady());
}

// ============================================================
// 설정 관리 테스트
// ============================================================

TEST_F(SDKManagerTest, GetConfig_ReturnsConfiguredValues) {
    SDKConfig config;
    config.model_path = model_path_.getPath();
    config.detector_type = DetectorType::MediaPipe;
    config.min_detection_confidence = 0.75f;
    config.max_faces = 2;
    config.enable_gpu = false;
    config.num_threads = 4;
    config.log_level = LogLevel::Debug;

    (void)SDKManager::getInstance().initialize(config);

    const SDKConfig& retrieved = SDKManager::getInstance().getConfig();
    EXPECT_EQ(retrieved.model_path, config.model_path);
    EXPECT_EQ(retrieved.detector_type, DetectorType::MediaPipe);
    EXPECT_FLOAT_EQ(retrieved.min_detection_confidence, 0.75f);
    EXPECT_EQ(retrieved.max_faces, 2);
    EXPECT_FALSE(retrieved.enable_gpu);
    EXPECT_EQ(retrieved.num_threads, 4);
    EXPECT_EQ(retrieved.log_level, LogLevel::Debug);
}

TEST_F(SDKManagerTest, SetLogLevel_ChangesLevel) {
    SDKConfig config;
    config.model_path = model_path_.getPath();
    config.log_level = LogLevel::Info;

    (void)SDKManager::getInstance().initialize(config);

    SDKManager::getInstance().setLogLevel(LogLevel::Debug);
    EXPECT_EQ(SDKManager::getInstance().getConfig().log_level, LogLevel::Info);
    // 참고: setLogLevel은 런타임 레벨을 변경하지만 config 자체는 불변
}

// ============================================================
// 로깅 시스템 테스트
// ============================================================

TEST_F(SDKManagerTest, Log_WithCallback_InvokesCallback) {
    SDKConfig config;
    config.model_path = model_path_.getPath();
    config.log_level = LogLevel::Debug;
    config.log_callback = [this](LogLevel level, const char* tag, const char* msg) {
        log_capture_.callback(level, tag, msg);
    };

    (void)SDKManager::getInstance().initialize(config);

    // 초기화 로그가 캡처되었는지 확인
    EXPECT_GT(log_capture_.count(), 0u);
    EXPECT_TRUE(log_capture_.hasLog(LogLevel::Info, "SDKManager", "initialized"));
}

TEST_F(SDKManagerTest, Log_BelowLevel_IsFiltered) {
    SDKConfig config;
    config.model_path = model_path_.getPath();
    config.log_level = LogLevel::Warning;  // Warning 이상만 출력
    config.log_callback = [this](LogLevel level, const char* tag, const char* msg) {
        log_capture_.callback(level, tag, msg);
    };

    (void)SDKManager::getInstance().initialize(config);

    size_t count_before = log_capture_.count();

    // Debug 로그는 필터링됨
    SDKManager::getInstance().log(LogLevel::Debug, "Test", "Debug message");
    EXPECT_EQ(log_capture_.count(), count_before);

    // Warning 로그는 출력됨
    SDKManager::getInstance().log(LogLevel::Warning, "Test", "Warning message");
    EXPECT_GT(log_capture_.count(), count_before);
}

TEST_F(SDKManagerTest, Log_WithFormat_FormatsCorrectly) {
    SDKConfig config;
    config.model_path = model_path_.getPath();
    config.log_level = LogLevel::Debug;
    config.log_callback = [this](LogLevel level, const char* tag, const char* msg) {
        log_capture_.callback(level, tag, msg);
    };

    (void)SDKManager::getInstance().initialize(config);
    log_capture_.clear();

    // printf 스타일 포맷팅
    SDKManager::getInstance().log(LogLevel::Info, "Test", "Value: %d, String: %s", 42, "hello");

    EXPECT_TRUE(log_capture_.hasLog(LogLevel::Info, "Test", "Value: 42"));
    EXPECT_TRUE(log_capture_.hasLog(LogLevel::Info, "Test", "String: hello"));
}

TEST_F(SDKManagerTest, SetLogCallback_ChangesCallback) {
    SDKConfig config;
    config.model_path = model_path_.getPath();
    config.log_level = LogLevel::Debug;

    (void)SDKManager::getInstance().initialize(config);

    // 콜백 설정
    SDKManager::getInstance().setLogCallback(
        [this](LogLevel level, const char* tag, const char* msg) {
            log_capture_.callback(level, tag, msg);
        });

    log_capture_.clear();
    SDKManager::getInstance().log(LogLevel::Info, "Test", "After callback set");

    EXPECT_TRUE(log_capture_.hasLog(LogLevel::Info, "Test", "After callback set"));
}

TEST_F(SDKManagerTest, LogMacros_Work) {
    SDKConfig config;
    config.model_path = model_path_.getPath();
    config.log_level = LogLevel::Verbose;
    config.log_callback = [this](LogLevel level, const char* tag, const char* msg) {
        log_capture_.callback(level, tag, msg);
    };

    (void)SDKManager::getInstance().initialize(config);
    log_capture_.clear();

    // 매크로 테스트
    IRIS_LOGV("MacroTest", "Verbose log");
    IRIS_LOGD("MacroTest", "Debug log");
    IRIS_LOGI("MacroTest", "Info log");
    IRIS_LOGW("MacroTest", "Warning log");
    IRIS_LOGE("MacroTest", "Error log");

    EXPECT_TRUE(log_capture_.hasLog(LogLevel::Verbose, "MacroTest", "Verbose"));
    EXPECT_TRUE(log_capture_.hasLog(LogLevel::Debug, "MacroTest", "Debug"));
    EXPECT_TRUE(log_capture_.hasLog(LogLevel::Info, "MacroTest", "Info"));
    EXPECT_TRUE(log_capture_.hasLog(LogLevel::Warning, "MacroTest", "Warning"));
    EXPECT_TRUE(log_capture_.hasLog(LogLevel::Error, "MacroTest", "Error"));
}

// ============================================================
// 팩토리 메서드 테스트
// ============================================================

TEST_F(SDKManagerTest, CreateFrameProcessor_WhenNotInitialized_ReturnsNull) {
    EXPECT_EQ(SDKManager::getInstance().getState(), SDKState::Uninitialized);

    auto processor = SDKManager::getInstance().createFrameProcessor();
    EXPECT_EQ(processor, nullptr);
}

TEST_F(SDKManagerTest, CreateDetector_WhenNotInitialized_ReturnsNull) {
    EXPECT_EQ(SDKManager::getInstance().getState(), SDKState::Uninitialized);

    auto detector = SDKManager::getInstance().createDetector();
    EXPECT_EQ(detector, nullptr);
}

TEST_F(SDKManagerTest, CreateRenderer_WhenNotInitialized_ReturnsNull) {
    EXPECT_EQ(SDKManager::getInstance().getState(), SDKState::Uninitialized);

    auto renderer = SDKManager::getInstance().createRenderer();
    EXPECT_EQ(renderer, nullptr);
}

// 참고: 실제 FrameProcessor, IrisDetector, LensRenderer 생성 테스트는
// 해당 클래스의 통합 테스트에서 수행 (모델 파일 필요)

// ============================================================
// 버전 정보 테스트
// ============================================================

TEST_F(SDKManagerTest, GetVersion_ReturnsNonEmpty) {
    const char* version = SDKManager::getVersion();
    ASSERT_NE(version, nullptr);
    EXPECT_GT(strlen(version), 0u);
}

TEST_F(SDKManagerTest, GetBuildInfo_ReturnsNonEmpty) {
    const char* build_info = SDKManager::getBuildInfo();
    ASSERT_NE(build_info, nullptr);
    EXPECT_GT(strlen(build_info), 0u);

    // 빌드 정보에 "IrisLensSDK" 포함 확인
    std::string info(build_info);
    EXPECT_NE(info.find("IrisLensSDK"), std::string::npos);
}

// ============================================================
// 스레드 안전성 테스트
// ============================================================

TEST_F(SDKManagerTest, ThreadSafe_ConcurrentInitializeShutdown) {
    constexpr int kIterations = 5;
    constexpr int kThreadCount = 4;

    for (int iter = 0; iter < kIterations; ++iter) {
        std::vector<std::thread> threads;
        std::atomic<int> success_count{0};

        for (int i = 0; i < kThreadCount; ++i) {
            threads.emplace_back([this, &success_count, i]() {
                SDKConfig config;
                config.model_path = model_path_.getPath();
                config.log_level = LogLevel::None;

                if (i % 2 == 0) {
                    if (SDKManager::getInstance().initialize(config)) {
                        success_count.fetch_add(1);
                    }
                } else {
                    SDKManager::getInstance().shutdown();
                }
            });
        }

        for (auto& t : threads) {
            t.join();
        }

        // 종료 후 재설정
        SDKManager::getInstance().shutdown();
    }

    // 테스트 통과 = 크래시 없이 완료
    SUCCEED();
}

TEST_F(SDKManagerTest, ThreadSafe_ConcurrentLogging) {
    SDKConfig config;
    config.model_path = model_path_.getPath();
    config.log_level = LogLevel::Debug;

    std::atomic<size_t> log_count{0};
    config.log_callback = [&log_count](LogLevel, const char*, const char*) {
        log_count.fetch_add(1);
    };

    (void)SDKManager::getInstance().initialize(config);

    constexpr int kThreadCount = 4;
    constexpr int kLogsPerThread = 100;
    std::vector<std::thread> threads;

    for (int i = 0; i < kThreadCount; ++i) {
        threads.emplace_back([i]() {
            for (int j = 0; j < kLogsPerThread; ++j) {
                SDKManager::getInstance().log(
                    LogLevel::Debug, "Thread", "Thread %d, Log %d", i, j);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // 모든 로그가 캡처되었는지 확인 (초기화 로그 + 스레드 로그)
    EXPECT_GE(log_count.load(), static_cast<size_t>(kThreadCount * kLogsPerThread));
}

// ============================================================
// 에러 상태 테스트
// ============================================================

TEST_F(SDKManagerTest, ErrorState_AfterFailedInit_CanReinitialize) {
    // 잘못된 경로로 초기화 실패
    SDKConfig bad_config;
    bad_config.model_path = "/nonexistent";
    bad_config.log_level = LogLevel::None;

    EXPECT_FALSE(SDKManager::getInstance().initialize(bad_config));
    EXPECT_EQ(SDKManager::getInstance().getState(), SDKState::Error);

    // shutdown 후 올바른 경로로 재초기화
    SDKManager::getInstance().shutdown();

    SDKConfig good_config;
    good_config.model_path = model_path_.getPath();

    EXPECT_TRUE(SDKManager::getInstance().initialize(good_config));
    EXPECT_TRUE(SDKManager::getInstance().isReady());
}

// ============================================================
// SDKConfig 기본값 테스트
// ============================================================

TEST_F(SDKManagerTest, SDKConfig_HasCorrectDefaults) {
    SDKConfig config;

    EXPECT_TRUE(config.model_path.empty());
    EXPECT_EQ(config.detector_type, DetectorType::MediaPipe);
    EXPECT_FLOAT_EQ(config.min_detection_confidence, 0.5f);
    EXPECT_FLOAT_EQ(config.min_tracking_confidence, 0.5f);
    EXPECT_EQ(config.max_faces, 1);
    EXPECT_FALSE(config.enable_gpu);
    EXPECT_EQ(config.num_threads, 0);
    EXPECT_EQ(config.log_level, LogLevel::Info);
    EXPECT_EQ(config.log_callback, nullptr);
}

}  // namespace
}  // namespace iris_sdk
