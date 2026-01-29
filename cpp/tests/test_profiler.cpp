/**
 * @file test_profiler.cpp
 * @brief Unit tests for Profiler and BufferPool classes
 */

#include <gtest/gtest.h>

#include "iris_sdk/profiler.h"
#include "iris_sdk/buffer_pool.h"

#include <thread>
#include <chrono>
#include <vector>
#include <atomic>
#include <string>

namespace iris_sdk {
namespace {

// ============================================================================
// Profiler Tests
// ============================================================================

class ProfilerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Reset profiler state before each test
        Profiler::getInstance().reset();
        Profiler::getInstance().setEnabled(true);
    }

    void TearDown() override {
        Profiler::getInstance().reset();
    }
};

TEST_F(ProfilerTest, SingletonInstance) {
    // Verify singleton returns same instance
    Profiler& p1 = Profiler::getInstance();
    Profiler& p2 = Profiler::getInstance();
    EXPECT_EQ(&p1, &p2);
}

TEST_F(ProfilerTest, BasicBeginEnd) {
    Profiler& profiler = Profiler::getInstance();

    profiler.begin("test_section");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    profiler.end("test_section");

    Measurement m = profiler.getMeasurement("test_section");
    EXPECT_EQ(m.count, 1);
    EXPECT_GE(m.total_ms, 10.0);  // At least 10ms
    EXPECT_GE(m.avg(), 10.0);
}

TEST_F(ProfilerTest, MultipleMeasurements) {
    Profiler& profiler = Profiler::getInstance();

    constexpr int NUM_ITERATIONS = 5;
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        profiler.begin("repeated_section");
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        profiler.end("repeated_section");
    }

    Measurement m = profiler.getMeasurement("repeated_section");
    EXPECT_EQ(m.count, NUM_ITERATIONS);
    EXPECT_GE(m.total_ms, 25.0);  // At least 5ms * 5
    EXPECT_GE(m.min_ms, 5.0);
    EXPECT_GE(m.max_ms, m.min_ms);
}

TEST_F(ProfilerTest, MeasurementStatistics) {
    Measurement m;
    m.min_ms = 1.0;
    m.max_ms = 10.0;
    m.total_ms = 30.0;
    m.count = 5;

    EXPECT_DOUBLE_EQ(m.avg(), 6.0);  // 30 / 5 = 6

    m.reset();
    EXPECT_EQ(m.count, 0);
    EXPECT_DOUBLE_EQ(m.total_ms, 0.0);
    EXPECT_DOUBLE_EQ(m.avg(), 0.0);  // 0 / 0 should return 0
}

TEST_F(ProfilerTest, NonExistentMeasurement) {
    Profiler& profiler = Profiler::getInstance();

    Measurement m = profiler.getMeasurement("non_existent");
    EXPECT_EQ(m.count, 0);
    EXPECT_DOUBLE_EQ(m.total_ms, 0.0);
}

TEST_F(ProfilerTest, EndWithoutBegin) {
    Profiler& profiler = Profiler::getInstance();

    // Should not crash
    profiler.end("never_started");

    Measurement m = profiler.getMeasurement("never_started");
    EXPECT_EQ(m.count, 0);
}

TEST_F(ProfilerTest, DisabledProfiling) {
    Profiler& profiler = Profiler::getInstance();

    profiler.setEnabled(false);
    EXPECT_FALSE(profiler.isEnabled());

    profiler.begin("disabled_section");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    profiler.end("disabled_section");

    Measurement m = profiler.getMeasurement("disabled_section");
    EXPECT_EQ(m.count, 0);  // Should not record when disabled

    profiler.setEnabled(true);
    EXPECT_TRUE(profiler.isEnabled());
}

TEST_F(ProfilerTest, Reset) {
    Profiler& profiler = Profiler::getInstance();

    profiler.begin("reset_test");
    profiler.end("reset_test");

    EXPECT_EQ(profiler.getMeasurement("reset_test").count, 1);

    profiler.reset();

    EXPECT_EQ(profiler.getMeasurement("reset_test").count, 0);
}

TEST_F(ProfilerTest, GenerateReport) {
    Profiler& profiler = Profiler::getInstance();

    profiler.begin("section_a");
    profiler.end("section_a");

    profiler.begin("section_b");
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    profiler.end("section_b");

    std::string report = profiler.generateReport();

    // Report should contain section names
    EXPECT_NE(report.find("section_a"), std::string::npos);
    EXPECT_NE(report.find("section_b"), std::string::npos);
    EXPECT_NE(report.find("avg="), std::string::npos);
    EXPECT_NE(report.find("min="), std::string::npos);
    EXPECT_NE(report.find("max="), std::string::npos);
    EXPECT_NE(report.find("count="), std::string::npos);
}

TEST_F(ProfilerTest, GenerateReportEmpty) {
    Profiler& profiler = Profiler::getInstance();

    std::string report = profiler.generateReport();

    EXPECT_NE(report.find("No measurements"), std::string::npos);
}

// ============================================================================
// ProfileScope RAII Tests
// ============================================================================

TEST_F(ProfilerTest, ProfileScopeBasic) {
    Profiler& profiler = Profiler::getInstance();

    {
        ProfileScope scope("scope_test");
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    Measurement m = profiler.getMeasurement("scope_test");
    EXPECT_EQ(m.count, 1);
    EXPECT_GE(m.total_ms, 10.0);
}

TEST_F(ProfilerTest, ProfileScopeNested) {
    Profiler& profiler = Profiler::getInstance();

    {
        ProfileScope outer("outer");
        {
            ProfileScope inner("inner");
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    Measurement outer_m = profiler.getMeasurement("outer");
    Measurement inner_m = profiler.getMeasurement("inner");

    EXPECT_EQ(outer_m.count, 1);
    EXPECT_EQ(inner_m.count, 1);
    EXPECT_GE(outer_m.total_ms, 15.0);  // outer >= inner + extra
    EXPECT_GE(inner_m.total_ms, 10.0);
}

// ============================================================================
// PROFILE_SCOPE and PROFILE_FUNCTION Macro Tests
// ============================================================================

void testFunction() {
    PROFILE_FUNCTION();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
}

TEST_F(ProfilerTest, ProfileFunctionMacro) {
    Profiler& profiler = Profiler::getInstance();

    testFunction();
    testFunction();

    // The function name should be recorded
    Measurement m = profiler.getMeasurement("testFunction");
    EXPECT_EQ(m.count, 2);
    EXPECT_GE(m.total_ms, 10.0);
}

TEST_F(ProfilerTest, ProfileScopeMacro) {
    Profiler& profiler = Profiler::getInstance();

    {
        PROFILE_SCOPE("macro_scope");
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    Measurement m = profiler.getMeasurement("macro_scope");
    EXPECT_EQ(m.count, 1);
    EXPECT_GE(m.total_ms, 5.0);
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

TEST_F(ProfilerTest, ThreadSafety) {
    Profiler& profiler = Profiler::getInstance();

    constexpr int NUM_THREADS = 4;
    constexpr int NUM_ITERATIONS = 100;
    std::atomic<int> completed{0};

    auto worker = [&](int thread_id) {
        std::string section = "thread_" + std::to_string(thread_id);
        for (int i = 0; i < NUM_ITERATIONS; ++i) {
            profiler.begin(section);
            // Small delay to simulate work
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            profiler.end(section);
        }
        completed++;
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back(worker, i);
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(completed.load(), NUM_THREADS);

    // Verify each thread's measurements
    for (int i = 0; i < NUM_THREADS; ++i) {
        std::string section = "thread_" + std::to_string(i);
        Measurement m = profiler.getMeasurement(section);
        EXPECT_EQ(m.count, NUM_ITERATIONS);
    }
}

// ============================================================================
// BufferPool Tests
// ============================================================================

class BufferPoolTest : public ::testing::Test {
protected:
    static constexpr int TEST_WIDTH = 640;
    static constexpr int TEST_HEIGHT = 480;
    static constexpr int TEST_TYPE = CV_8UC3;
};

TEST_F(BufferPoolTest, Initialization) {
    BufferPool pool;

    EXPECT_FALSE(pool.isInitialized());

    EXPECT_TRUE(pool.initialize(TEST_WIDTH, TEST_HEIGHT, TEST_TYPE, 3, 5));
    EXPECT_TRUE(pool.isInitialized());

    EXPECT_EQ(pool.getWidth(), TEST_WIDTH);
    EXPECT_EQ(pool.getHeight(), TEST_HEIGHT);
    EXPECT_EQ(pool.getType(), TEST_TYPE);

    BufferPoolStats stats = pool.getStats();
    EXPECT_EQ(stats.total_buffers, 3);
    EXPECT_EQ(stats.available_buffers, 3);
    EXPECT_EQ(stats.in_use_buffers, 0);
}

TEST_F(BufferPoolTest, InitializationInvalidParams) {
    BufferPool pool;

    EXPECT_FALSE(pool.initialize(0, TEST_HEIGHT, TEST_TYPE));
    EXPECT_FALSE(pool.initialize(TEST_WIDTH, 0, TEST_TYPE));
    EXPECT_FALSE(pool.initialize(-1, -1, TEST_TYPE));
}

TEST_F(BufferPoolTest, AcquireRelease) {
    BufferPool pool;
    pool.initialize(TEST_WIDTH, TEST_HEIGHT, TEST_TYPE, 2, 4);

    // Acquire first buffer
    cv::Mat buffer1 = pool.acquire();
    EXPECT_FALSE(buffer1.empty());
    EXPECT_EQ(buffer1.cols, TEST_WIDTH);
    EXPECT_EQ(buffer1.rows, TEST_HEIGHT);
    EXPECT_EQ(buffer1.type(), TEST_TYPE);

    BufferPoolStats stats = pool.getStats();
    EXPECT_EQ(stats.in_use_buffers, 1);
    EXPECT_EQ(stats.available_buffers, 1);

    // Acquire second buffer
    cv::Mat buffer2 = pool.acquire();
    EXPECT_FALSE(buffer2.empty());
    EXPECT_NE(buffer1.data, buffer2.data);  // Different buffers

    stats = pool.getStats();
    EXPECT_EQ(stats.in_use_buffers, 2);
    EXPECT_EQ(stats.available_buffers, 0);

    // Release first buffer
    pool.release(buffer1);
    EXPECT_TRUE(buffer1.empty());  // Should be cleared

    stats = pool.getStats();
    EXPECT_EQ(stats.in_use_buffers, 1);
    EXPECT_EQ(stats.available_buffers, 1);
}

TEST_F(BufferPoolTest, TryAcquire) {
    BufferPool pool;
    pool.initialize(TEST_WIDTH, TEST_HEIGHT, TEST_TYPE, 1, 2);

    cv::Mat buffer1, buffer2, buffer3;

    EXPECT_TRUE(pool.tryAcquire(buffer1));
    EXPECT_TRUE(pool.tryAcquire(buffer2));  // Will allocate new
    EXPECT_FALSE(pool.tryAcquire(buffer3)); // Pool exhausted

    EXPECT_FALSE(buffer1.empty());
    EXPECT_FALSE(buffer2.empty());
    EXPECT_TRUE(buffer3.empty());
}

TEST_F(BufferPoolTest, PoolGrowth) {
    BufferPool pool;
    pool.initialize(TEST_WIDTH, TEST_HEIGHT, TEST_TYPE, 1, 5);

    std::vector<cv::Mat> buffers;
    for (int i = 0; i < 5; ++i) {
        buffers.push_back(pool.acquire());
        EXPECT_FALSE(buffers.back().empty());
    }

    BufferPoolStats stats = pool.getStats();
    EXPECT_EQ(stats.total_buffers, 5);
    EXPECT_EQ(stats.in_use_buffers, 5);
    EXPECT_EQ(stats.allocation_count, 5);

    // Should fail now
    cv::Mat extra = pool.acquire();
    EXPECT_TRUE(extra.empty());
}

TEST_F(BufferPoolTest, UnlimitedPool) {
    BufferPool pool;
    pool.initialize(TEST_WIDTH, TEST_HEIGHT, TEST_TYPE, 1, 0);  // 0 = unlimited

    std::vector<cv::Mat> buffers;
    for (int i = 0; i < 20; ++i) {
        buffers.push_back(pool.acquire());
        EXPECT_FALSE(buffers.back().empty());
    }

    BufferPoolStats stats = pool.getStats();
    EXPECT_EQ(stats.total_buffers, 20);
}

TEST_F(BufferPoolTest, ReleaseInvalidBuffer) {
    BufferPool pool;
    pool.initialize(TEST_WIDTH, TEST_HEIGHT, TEST_TYPE, 2, 4);

    // Create buffer not from pool
    cv::Mat external_buffer(TEST_HEIGHT, TEST_WIDTH, TEST_TYPE);

    // Should not crash, just ignore
    pool.release(external_buffer);
    EXPECT_TRUE(external_buffer.empty());  // Should still be cleared

    BufferPoolStats stats = pool.getStats();
    EXPECT_EQ(stats.in_use_buffers, 0);
}

TEST_F(BufferPoolTest, ReleaseEmptyBuffer) {
    BufferPool pool;
    pool.initialize(TEST_WIDTH, TEST_HEIGHT, TEST_TYPE, 2, 4);

    cv::Mat empty_buffer;
    pool.release(empty_buffer);  // Should not crash
}

TEST_F(BufferPoolTest, GetStatsMemory) {
    BufferPool pool;
    pool.initialize(TEST_WIDTH, TEST_HEIGHT, TEST_TYPE, 3, 5);

    BufferPoolStats stats = pool.getStats();

    size_t expected_per_buffer = TEST_WIDTH * TEST_HEIGHT * 3;  // CV_8UC3
    EXPECT_EQ(stats.total_memory_bytes, expected_per_buffer * 3);
}

TEST_F(BufferPoolTest, Trim) {
    BufferPool pool;
    pool.initialize(TEST_WIDTH, TEST_HEIGHT, TEST_TYPE, 5, 10);

    // Acquire and release to mark as used
    std::vector<cv::Mat> buffers;
    for (int i = 0; i < 5; ++i) {
        buffers.push_back(pool.acquire());
    }
    for (auto& b : buffers) {
        pool.release(b);
    }

    BufferPoolStats stats = pool.getStats();
    EXPECT_EQ(stats.total_buffers, 5);
    EXPECT_EQ(stats.available_buffers, 5);

    // Trim to keep only 2
    pool.trim(2);

    stats = pool.getStats();
    EXPECT_EQ(stats.total_buffers, 2);
}

TEST_F(BufferPoolTest, Clear) {
    BufferPool pool;
    pool.initialize(TEST_WIDTH, TEST_HEIGHT, TEST_TYPE, 3, 5);

    pool.acquire();
    pool.acquire();

    pool.clear();

    BufferPoolStats stats = pool.getStats();
    EXPECT_EQ(stats.total_buffers, 0);
}

TEST_F(BufferPoolTest, Release) {
    BufferPool pool;
    pool.initialize(TEST_WIDTH, TEST_HEIGHT, TEST_TYPE, 3, 5);

    cv::Mat buffer = pool.acquire();
    EXPECT_TRUE(pool.isInitialized());

    pool.release();

    EXPECT_FALSE(pool.isInitialized());
    EXPECT_TRUE(pool.acquire().empty());
}

TEST_F(BufferPoolTest, MoveConstruction) {
    BufferPool pool1;
    pool1.initialize(TEST_WIDTH, TEST_HEIGHT, TEST_TYPE, 3, 5);

    cv::Mat buffer = pool1.acquire();

    BufferPool pool2(std::move(pool1));

    EXPECT_TRUE(pool2.isInitialized());
    EXPECT_FALSE(pool1.isInitialized());  // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(pool2.getWidth(), TEST_WIDTH);
}

// ============================================================================
// ScopedBuffer Tests
// ============================================================================

TEST_F(BufferPoolTest, ScopedBufferBasic) {
    BufferPool pool;
    pool.initialize(TEST_WIDTH, TEST_HEIGHT, TEST_TYPE, 2, 4);

    {
        ScopedBuffer scoped(pool);
        EXPECT_TRUE(static_cast<bool>(scoped));
        EXPECT_FALSE(scoped.get().empty());
        EXPECT_EQ(scoped.get().cols, TEST_WIDTH);

        BufferPoolStats stats = pool.getStats();
        EXPECT_EQ(stats.in_use_buffers, 1);
    }

    // After scope ends, buffer should be released
    BufferPoolStats stats = pool.getStats();
    EXPECT_EQ(stats.in_use_buffers, 0);
    EXPECT_EQ(stats.available_buffers, 2);
}

TEST_F(BufferPoolTest, ScopedBufferExhaustedPool) {
    BufferPool pool;
    pool.initialize(TEST_WIDTH, TEST_HEIGHT, TEST_TYPE, 1, 1);

    cv::Mat held = pool.acquire();  // Take the only buffer

    ScopedBuffer scoped(pool);
    EXPECT_FALSE(static_cast<bool>(scoped));  // Should fail
    EXPECT_TRUE(scoped.get().empty());
}

// ============================================================================
// BufferPool Thread Safety Tests
// ============================================================================

TEST_F(BufferPoolTest, ThreadSafety) {
    BufferPool pool;
    pool.initialize(TEST_WIDTH, TEST_HEIGHT, TEST_TYPE, 4, 20);

    constexpr int NUM_THREADS = 4;
    constexpr int NUM_ITERATIONS = 50;
    std::atomic<int> success_count{0};

    auto worker = [&]() {
        for (int i = 0; i < NUM_ITERATIONS; ++i) {
            cv::Mat buffer = pool.acquire();
            if (!buffer.empty()) {
                // Simulate some work
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                pool.release(buffer);
                success_count++;
            }
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back(worker);
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(success_count.load(), NUM_THREADS * NUM_ITERATIONS);

    BufferPoolStats stats = pool.getStats();
    EXPECT_EQ(stats.in_use_buffers, 0);  // All released
}

} // namespace
} // namespace iris_sdk
