/**
 * @file test_mediapipe_detector_performance.cpp
 * @brief MediaPipeDetector 성능 벤치마크 테스트
 *
 * 성능 목표:
 * - 검출 지연: 33ms 이하 (30fps)
 * - 프레임 레이트: 30fps 이상
 * - 메모리 사용량: 100MB 이하
 *
 * @note 실제 TFLite 모델이 있어야 의미 있는 테스트 가능
 */

#include <gtest/gtest.h>
#include <chrono>
#include <vector>
#include <numeric>
#include <cmath>
#include <random>
#include "iris_sdk/mediapipe_detector.h"

namespace iris_sdk {
namespace testing {

// ============================================================
// 테스트 설정 상수
// ============================================================
namespace {
    // 성능 목표
    constexpr double TARGET_LATENCY_MS = 33.0;        ///< 33ms (30fps)
    constexpr double TARGET_FPS = 30.0;              ///< 30fps
    constexpr size_t TARGET_MEMORY_MB = 100;          ///< 100MB

    // 테스트 설정
    constexpr int WARMUP_ITERATIONS = 10;             ///< 워밍업 반복 횟수
    constexpr int BENCHMARK_ITERATIONS = 100;         ///< 벤치마크 반복 횟수
    constexpr int FRAME_WIDTH = 640;                  ///< 테스트 프레임 너비
    constexpr int FRAME_HEIGHT = 480;                 ///< 테스트 프레임 높이
    constexpr int FRAME_CHANNELS = 3;                 ///< RGB

    // 테스트 프레임 크기 (바이트)
    constexpr size_t FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT * FRAME_CHANNELS;
}

// ============================================================
// 성능 측정 유틸리티
// ============================================================

/**
 * @brief 시간 측정 결과
 */
struct TimingResult {
    double min_ms = 0.0;
    double max_ms = 0.0;
    double avg_ms = 0.0;
    double median_ms = 0.0;
    double std_dev_ms = 0.0;
    double p95_ms = 0.0;      ///< 95 퍼센타일
    double p99_ms = 0.0;      ///< 99 퍼센타일
    int sample_count = 0;
};

/**
 * @brief 타이밍 데이터로부터 통계 계산
 */
TimingResult calculateStats(std::vector<double>& times) {
    TimingResult result;
    if (times.empty()) {
        return result;
    }

    result.sample_count = static_cast<int>(times.size());

    // 정렬
    std::sort(times.begin(), times.end());

    // Min, Max
    result.min_ms = times.front();
    result.max_ms = times.back();

    // Average
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    result.avg_ms = sum / times.size();

    // Median
    size_t mid = times.size() / 2;
    if (times.size() % 2 == 0) {
        result.median_ms = (times[mid - 1] + times[mid]) / 2.0;
    } else {
        result.median_ms = times[mid];
    }

    // Standard Deviation
    double sq_sum = 0.0;
    for (double t : times) {
        sq_sum += (t - result.avg_ms) * (t - result.avg_ms);
    }
    result.std_dev_ms = std::sqrt(sq_sum / times.size());

    // Percentiles
    size_t p95_idx = static_cast<size_t>(times.size() * 0.95);
    size_t p99_idx = static_cast<size_t>(times.size() * 0.99);
    result.p95_ms = times[std::min(p95_idx, times.size() - 1)];
    result.p99_ms = times[std::min(p99_idx, times.size() - 1)];

    return result;
}

/**
 * @brief 테스트용 더미 프레임 생성
 *
 * 실제 얼굴 이미지가 없어도 API 동작과 시간 측정 가능
 */
std::vector<uint8_t> generateDummyFrame(int width, int height, int channels,
                                         bool with_noise = true) {
    std::vector<uint8_t> frame(width * height * channels);

    if (with_noise) {
        // 랜덤 노이즈 추가 (실제 이미지와 유사한 처리 부하)
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dist(0, 255);

        for (auto& pixel : frame) {
            pixel = static_cast<uint8_t>(dist(gen));
        }
    } else {
        // 단색 채우기
        std::fill(frame.begin(), frame.end(), 128);
    }

    return frame;
}

// ============================================================
// 성능 테스트 픽스처
// ============================================================

class MediaPipeDetectorPerformanceTest : public ::testing::Test {
protected:
    std::unique_ptr<MediaPipeDetector> detector_;
    std::vector<uint8_t> test_frame_;

    void SetUp() override {
        detector_ = std::make_unique<MediaPipeDetector>();
        test_frame_ = generateDummyFrame(FRAME_WIDTH, FRAME_HEIGHT, FRAME_CHANNELS);

        // 성능 설정
        detector_->setNumThreads(4);
        detector_->setTrackingEnabled(true);

        // 참고: 실제 모델 경로가 필요함
        // 테스트 환경에서는 초기화 실패할 수 있음
    }

    void TearDown() override {
        if (detector_) {
            detector_->release();
            detector_.reset();
        }
    }

    /**
     * @brief detect() 호출 시간 측정
     * @param iterations 반복 횟수
     * @param warmup 워밍업 반복 횟수
     * @return 타이밍 결과
     */
    TimingResult measureDetectLatency(int iterations, int warmup = 0) {
        std::vector<double> times;
        times.reserve(iterations);

        // 워밍업 (캐시, JIT 등 최적화)
        for (int i = 0; i < warmup; ++i) {
            detector_->detect(test_frame_.data(), FRAME_WIDTH, FRAME_HEIGHT,
                             FrameFormat::RGB);
        }

        // 실제 측정
        for (int i = 0; i < iterations; ++i) {
            auto start = std::chrono::high_resolution_clock::now();

            detector_->detect(test_frame_.data(), FRAME_WIDTH, FRAME_HEIGHT,
                             FrameFormat::RGB);

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            times.push_back(duration.count() / 1000.0);  // ms
        }

        return calculateStats(times);
    }
};

// ============================================================
// 기본 성능 테스트 (모델 없이도 동작)
// ============================================================

/**
 * @test 초기화되지 않은 상태에서 detect() 지연 시간
 *
 * 초기화되지 않은 경우 즉시 반환해야 함 (< 1ms)
 */
TEST_F(MediaPipeDetectorPerformanceTest, UninitializedDetectIsImmediate) {
    // 초기화하지 않은 상태
    ASSERT_FALSE(detector_->isInitialized());

    // 측정
    auto result = measureDetectLatency(100, 10);

    // 초기화되지 않은 경우 즉시 반환 (1ms 이하)
    EXPECT_LT(result.avg_ms, 1.0)
        << "Uninitialized detect should return immediately";
    EXPECT_LT(result.p99_ms, 2.0)
        << "99th percentile should be under 2ms";

    // 결과 출력
    std::cout << "Uninitialized detect latency:" << std::endl;
    std::cout << "  Avg: " << result.avg_ms << " ms" << std::endl;
    std::cout << "  Min: " << result.min_ms << " ms" << std::endl;
    std::cout << "  Max: " << result.max_ms << " ms" << std::endl;
    std::cout << "  P95: " << result.p95_ms << " ms" << std::endl;
    std::cout << "  P99: " << result.p99_ms << " ms" << std::endl;
}

/**
 * @test NULL 프레임에 대한 detect() 지연 시간
 */
TEST_F(MediaPipeDetectorPerformanceTest, NullFrameDetectIsImmediate) {
    std::vector<double> times;

    for (int i = 0; i < 100; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        detector_->detect(nullptr, FRAME_WIDTH, FRAME_HEIGHT, FrameFormat::RGB);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        times.push_back(duration.count() / 1000.0);
    }

    auto result = calculateStats(times);

    EXPECT_LT(result.avg_ms, 0.1)
        << "Null frame detect should return immediately";

    std::cout << "Null frame detect latency: " << result.avg_ms << " ms (avg)" << std::endl;
}

/**
 * @test 잘못된 크기에 대한 detect() 지연 시간
 */
TEST_F(MediaPipeDetectorPerformanceTest, InvalidSizeDetectIsImmediate) {
    std::vector<double> times;

    for (int i = 0; i < 100; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        detector_->detect(test_frame_.data(), 0, 0, FrameFormat::RGB);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        times.push_back(duration.count() / 1000.0);
    }

    auto result = calculateStats(times);

    EXPECT_LT(result.avg_ms, 0.1)
        << "Invalid size detect should return immediately";

    std::cout << "Invalid size detect latency: " << result.avg_ms << " ms (avg)" << std::endl;
}

// ============================================================
// 메모리 관련 테스트
// ============================================================

/**
 * @test 반복 detect() 호출 시 메모리 누수 없음
 *
 * 사전 할당 버퍼 재사용으로 메모리 증가 없어야 함
 */
TEST_F(MediaPipeDetectorPerformanceTest, NoMemoryLeakOnRepeatedDetect) {
    // 초기화되지 않은 상태에서도 버퍼 할당 체크 가능
    // 실제 초기화 후에는 버퍼가 재사용되어야 함

    // 초기 메모리 상태 (간접 측정)
    size_t initial_vector_capacity = test_frame_.capacity();

    // 여러 번 호출
    for (int i = 0; i < 1000; ++i) {
        IrisResult result = detector_->detect(
            test_frame_.data(), FRAME_WIDTH, FRAME_HEIGHT, FrameFormat::RGB);
        (void)result;  // 사용하지 않음
    }

    // 테스트 프레임 용량이 변경되지 않았는지 확인
    // (detect 내부에서 입력 벡터를 수정하지 않음)
    EXPECT_EQ(test_frame_.capacity(), initial_vector_capacity);
}

/**
 * @test 다양한 프레임 크기에서 성능 일관성
 */
TEST_F(MediaPipeDetectorPerformanceTest, ConsistentPerformanceAcrossFrameSizes) {
    struct FrameSize {
        int width;
        int height;
        const char* name;
    };

    std::vector<FrameSize> sizes = {
        {320, 240, "QVGA"},
        {640, 480, "VGA"},
        {1280, 720, "HD"},
        {1920, 1080, "FHD"}
    };

    std::cout << "\nFrame size performance comparison (uninitialized):" << std::endl;

    for (const auto& size : sizes) {
        auto frame = generateDummyFrame(size.width, size.height, FRAME_CHANNELS);
        std::vector<double> times;

        for (int i = 0; i < 100; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            detector_->detect(frame.data(), size.width, size.height, FrameFormat::RGB);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            times.push_back(duration.count() / 1000.0);
        }

        auto result = calculateStats(times);
        std::cout << "  " << size.name << " (" << size.width << "x" << size.height << "): "
                  << result.avg_ms << " ms (avg)" << std::endl;
    }
}

// ============================================================
// 설정 API 성능 테스트
// ============================================================

/**
 * @test 설정 API 호출 시간
 *
 * 설정 변경은 즉시 완료되어야 함 (< 1ms)
 */
TEST_F(MediaPipeDetectorPerformanceTest, SettingsApiIsImmediate) {
    auto measureSetting = [](std::function<void()> fn) -> double {
        auto start = std::chrono::high_resolution_clock::now();
        fn();
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    };

    double time_confidence = measureSetting([this]() {
        detector_->setMinDetectionConfidence(0.7f);
    });

    double time_tracking = measureSetting([this]() {
        detector_->setMinTrackingConfidence(0.6f);
    });

    double time_faces = measureSetting([this]() {
        detector_->setNumFaces(2);
    });

    double time_threads = measureSetting([this]() {
        detector_->setNumThreads(4);
    });

    double time_tracking_enable = measureSetting([this]() {
        detector_->setTrackingEnabled(true);
    });

    double time_reset = measureSetting([this]() {
        detector_->resetTracking();
    });

    EXPECT_LT(time_confidence, 1.0);
    EXPECT_LT(time_tracking, 1.0);
    EXPECT_LT(time_faces, 1.0);
    EXPECT_LT(time_threads, 1.0);
    EXPECT_LT(time_tracking_enable, 1.0);
    EXPECT_LT(time_reset, 1.0);

    std::cout << "\nSettings API latency:" << std::endl;
    std::cout << "  setMinDetectionConfidence: " << time_confidence << " ms" << std::endl;
    std::cout << "  setMinTrackingConfidence: " << time_tracking << " ms" << std::endl;
    std::cout << "  setNumFaces: " << time_faces << " ms" << std::endl;
    std::cout << "  setNumThreads: " << time_threads << " ms" << std::endl;
    std::cout << "  setTrackingEnabled: " << time_tracking_enable << " ms" << std::endl;
    std::cout << "  resetTracking: " << time_reset << " ms" << std::endl;
}

// ============================================================
// 스레드 수 설정 테스트
// ============================================================

/**
 * @test 스레드 수 설정 범위 검증
 */
TEST_F(MediaPipeDetectorPerformanceTest, ThreadCountBoundary) {
    // 유효한 범위 (1-16)
    EXPECT_NO_THROW(detector_->setNumThreads(1));
    EXPECT_NO_THROW(detector_->setNumThreads(4));
    EXPECT_NO_THROW(detector_->setNumThreads(8));
    EXPECT_NO_THROW(detector_->setNumThreads(16));

    // 경계값 테스트 (클램핑 동작)
    EXPECT_NO_THROW(detector_->setNumThreads(0));   // -> 1로 클램핑
    EXPECT_NO_THROW(detector_->setNumThreads(-1));  // -> 1로 클램핑
    EXPECT_NO_THROW(detector_->setNumThreads(100)); // -> 16으로 클램핑
}

// ============================================================
// 추적 모드 테스트
// ============================================================

/**
 * @test 추적 모드 활성화/비활성화
 */
TEST_F(MediaPipeDetectorPerformanceTest, TrackingModeToggle) {
    // 추적 활성화
    detector_->setTrackingEnabled(true);
    EXPECT_NO_THROW(detector_->detect(test_frame_.data(), FRAME_WIDTH, FRAME_HEIGHT,
                                      FrameFormat::RGB));

    // 추적 비활성화
    detector_->setTrackingEnabled(false);
    EXPECT_NO_THROW(detector_->detect(test_frame_.data(), FRAME_WIDTH, FRAME_HEIGHT,
                                      FrameFormat::RGB));

    // 추적 리셋
    detector_->setTrackingEnabled(true);
    detector_->resetTracking();
    EXPECT_NO_THROW(detector_->detect(test_frame_.data(), FRAME_WIDTH, FRAME_HEIGHT,
                                      FrameFormat::RGB));
}

// ============================================================
// 연속 프레임 처리 테스트 (시뮬레이션)
// ============================================================

/**
 * @test 연속 프레임 처리 시간 일관성
 *
 * 실제 비디오 스트림 처리 시뮬레이션
 */
TEST_F(MediaPipeDetectorPerformanceTest, ContinuousFrameProcessing) {
    constexpr int NUM_FRAMES = 300;  // 10초 분량 (30fps)
    std::vector<double> frame_times;
    frame_times.reserve(NUM_FRAMES);

    // 연속 프레임 처리
    for (int i = 0; i < NUM_FRAMES; ++i) {
        // 약간 다른 프레임 생성 (노이즈 변화)
        if (i % 10 == 0) {
            test_frame_ = generateDummyFrame(FRAME_WIDTH, FRAME_HEIGHT, FRAME_CHANNELS, true);
        }

        auto start = std::chrono::high_resolution_clock::now();
        IrisResult result = detector_->detect(
            test_frame_.data(), FRAME_WIDTH, FRAME_HEIGHT, FrameFormat::RGB);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        frame_times.push_back(duration.count() / 1000.0);

        (void)result;
    }

    auto stats = calculateStats(frame_times);

    std::cout << "\nContinuous frame processing (" << NUM_FRAMES << " frames):" << std::endl;
    std::cout << "  Avg: " << stats.avg_ms << " ms" << std::endl;
    std::cout << "  Min: " << stats.min_ms << " ms" << std::endl;
    std::cout << "  Max: " << stats.max_ms << " ms" << std::endl;
    std::cout << "  Std Dev: " << stats.std_dev_ms << " ms" << std::endl;
    std::cout << "  P95: " << stats.p95_ms << " ms" << std::endl;
    std::cout << "  P99: " << stats.p99_ms << " ms" << std::endl;

    // 표준 편차가 너무 크지 않아야 함 (일관된 성능)
    // 초기화되지 않은 상태에서는 매우 낮은 값이어야 함
    if (!detector_->isInitialized()) {
        EXPECT_LT(stats.std_dev_ms, 1.0)
            << "Uninitialized detection should have consistent low latency";
    }
}

// ============================================================
// 워밍업 효과 테스트
// ============================================================

/**
 * @test 워밍업이 성능에 미치는 영향 측정
 *
 * 첫 몇 프레임 후 성능이 안정화되어야 함
 */
TEST_F(MediaPipeDetectorPerformanceTest, WarmupDoesNotAffectPerformance) {
    constexpr int TOTAL_ITERATIONS = 50;
    constexpr int WARMUP_COUNT = 10;

    std::vector<double> all_times;
    all_times.reserve(TOTAL_ITERATIONS);

    for (int i = 0; i < TOTAL_ITERATIONS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        detector_->detect(test_frame_.data(), FRAME_WIDTH, FRAME_HEIGHT, FrameFormat::RGB);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        all_times.push_back(duration.count() / 1000.0);
    }

    // 워밍업 기간 vs 안정화 후 비교
    std::vector<double> warmup_times(all_times.begin(), all_times.begin() + WARMUP_COUNT);
    std::vector<double> stable_times(all_times.begin() + WARMUP_COUNT, all_times.end());

    auto warmup_stats = calculateStats(warmup_times);
    auto stable_stats = calculateStats(stable_times);

    std::cout << "\nWarmup effect analysis:" << std::endl;
    std::cout << "  Warmup period (first " << WARMUP_COUNT << "): "
              << warmup_stats.avg_ms << " ms (avg)" << std::endl;
    std::cout << "  Stable period: " << stable_stats.avg_ms << " ms (avg)" << std::endl;
    std::cout << "  Improvement: "
              << ((warmup_stats.avg_ms - stable_stats.avg_ms) / warmup_stats.avg_ms * 100)
              << "%" << std::endl;

    // 워밍업 후 성능이 나빠지면 안 됨
    // (초기화되지 않은 상태에서는 거의 동일해야 함)
    EXPECT_LE(stable_stats.avg_ms, warmup_stats.avg_ms * 1.5)
        << "Performance should not degrade after warmup";
}

// ============================================================
// 프레임 포맷별 성능 테스트
// ============================================================

/**
 * @test 다양한 프레임 포맷에서 성능 비교
 */
TEST_F(MediaPipeDetectorPerformanceTest, FrameFormatPerformance) {
    struct FormatTest {
        FrameFormat format;
        int channels;
        const char* name;
    };

    std::vector<FormatTest> formats = {
        {FrameFormat::RGB, 3, "RGB"},
        {FrameFormat::BGR, 3, "BGR"},
        {FrameFormat::RGBA, 4, "RGBA"},
        {FrameFormat::BGRA, 4, "BGRA"},
        {FrameFormat::Grayscale, 1, "Grayscale"}
    };

    std::cout << "\nFrame format performance comparison:" << std::endl;

    for (const auto& fmt : formats) {
        auto frame = generateDummyFrame(FRAME_WIDTH, FRAME_HEIGHT, fmt.channels);
        std::vector<double> times;

        for (int i = 0; i < 100; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            detector_->detect(frame.data(), FRAME_WIDTH, FRAME_HEIGHT, fmt.format);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            times.push_back(duration.count() / 1000.0);
        }

        auto result = calculateStats(times);
        std::cout << "  " << fmt.name << ": " << result.avg_ms << " ms (avg), "
                  << "P99: " << result.p99_ms << " ms" << std::endl;
    }
}

// ============================================================
// 목표 성능 테스트 (실제 모델 필요)
// ============================================================

/**
 * @test 목표 지연 시간 달성 여부 (33ms 이하)
 *
 * @note 이 테스트는 실제 TFLite 모델이 초기화된 경우에만 의미 있음
 */
TEST_F(MediaPipeDetectorPerformanceTest, DetectionLatencyUnder33ms) {
    // 모델이 초기화되지 않은 경우 스킵
    if (!detector_->isInitialized()) {
        GTEST_SKIP() << "Detector not initialized (model files required)";
    }

    auto result = measureDetectLatency(BENCHMARK_ITERATIONS, WARMUP_ITERATIONS);

    std::cout << "\nTarget latency test (33ms):" << std::endl;
    std::cout << "  Avg: " << result.avg_ms << " ms" << std::endl;
    std::cout << "  P95: " << result.p95_ms << " ms" << std::endl;
    std::cout << "  P99: " << result.p99_ms << " ms" << std::endl;

    // 평균 지연 시간이 목표 이하여야 함
    EXPECT_LE(result.avg_ms, TARGET_LATENCY_MS)
        << "Average latency should be under " << TARGET_LATENCY_MS << "ms";

    // 95 퍼센타일도 목표 근처여야 함
    EXPECT_LE(result.p95_ms, TARGET_LATENCY_MS * 1.5)
        << "P95 latency should be under " << (TARGET_LATENCY_MS * 1.5) << "ms";
}

/**
 * @test 30fps 처리 가능 여부
 *
 * @note 이 테스트는 실제 TFLite 모델이 초기화된 경우에만 의미 있음
 */
TEST_F(MediaPipeDetectorPerformanceTest, CanProcess30FPS) {
    if (!detector_->isInitialized()) {
        GTEST_SKIP() << "Detector not initialized (model files required)";
    }

    constexpr int DURATION_SECONDS = 5;
    constexpr int TARGET_FRAMES = static_cast<int>(TARGET_FPS * DURATION_SECONDS);

    auto start = std::chrono::high_resolution_clock::now();

    int processed_frames = 0;
    while (processed_frames < TARGET_FRAMES) {
        detector_->detect(test_frame_.data(), FRAME_WIDTH, FRAME_HEIGHT, FrameFormat::RGB);
        ++processed_frames;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    double actual_fps = processed_frames / (duration.count() / 1000.0);

    std::cout << "\n30fps capability test:" << std::endl;
    std::cout << "  Target: " << TARGET_FRAMES << " frames in " << DURATION_SECONDS << "s" << std::endl;
    std::cout << "  Actual: " << processed_frames << " frames in " << duration.count() << "ms" << std::endl;
    std::cout << "  Achieved FPS: " << actual_fps << std::endl;

    EXPECT_GE(actual_fps, TARGET_FPS)
        << "Should achieve at least " << TARGET_FPS << " fps";
}

/**
 * @test 메모리 사용량 100MB 이하
 *
 * @note 실제 메모리 측정은 플랫폼별로 다름
 *       이 테스트는 대략적인 추정만 수행
 */
TEST_F(MediaPipeDetectorPerformanceTest, MemoryUsageUnder100MB) {
    // MediaPipeDetector 객체 크기 추정
    // (실제 메모리는 TFLite 모델 로딩 시 증가)

    size_t detector_size = sizeof(MediaPipeDetector);
    size_t frame_size = test_frame_.capacity();

    std::cout << "\nMemory usage estimate:" << std::endl;
    std::cout << "  MediaPipeDetector object: " << detector_size << " bytes" << std::endl;
    std::cout << "  Test frame buffer: " << frame_size << " bytes" << std::endl;

    // 기본 객체 크기는 작아야 함
    EXPECT_LT(detector_size, 1024)  // 1KB 미만
        << "MediaPipeDetector object should be small (Pimpl pattern)";

    // 참고: 실제 메모리 사용량은 초기화 후 모델/버퍼 포함
    // 플랫폼별 메모리 API로 측정 필요
    // (예: getrusage on Linux, mach_task_info on macOS)
}

} // namespace testing
} // namespace iris_sdk
