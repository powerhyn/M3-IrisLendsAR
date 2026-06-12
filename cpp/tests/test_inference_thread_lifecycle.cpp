/**
 * @file test_inference_thread_lifecycle.cpp
 * @brief InferenceThread 수명/스레드 안전성 테스트 (③-2 B1)
 *
 * 목적: 추적 레이어 분리/이동에 앞서 start/stop/소멸의 수명 경로를 회귀로 고정한다.
 *
 * 설계 원칙(결정적, 실모델 불필요):
 *  - InferenceThread는 내부에서 MediaPipeDetector를 하드코딩 생성하므로 mock 주입이
 *    불가능하다. 대신 "존재하지 않는 모델 경로"를 주면 MediaPipeDetector::initialize가
 *    std::filesystem::exists 단계에서 즉시 false를 반환하여 초기화 실패가 결정적으로
 *    재현된다. 이 init-fail 경로가 본 배치(B1)에서 수리한 결함들이 가장 잘 드러나는
 *    지점이다(joinable 미회수 → std::terminate, Starting 중 stop → join 영구 행 등).
 *  - 각 위험 동작은 별도 스레드에서 실행하고 std::future::wait_for로 "행(hang)"을
 *    감지하는 타임아웃 가드를 둔다. 가드가 걸리면 결함(데드락/행)으로 FAIL 처리한다.
 *
 * 주의: 이 테스트는 모델/TFLite/OpenCV에 의존하지 않는다.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <future>
#include <thread>
#include <vector>

#include "iris_sdk/inference_thread.h"
#include "iris_sdk/types.h"

using namespace iris_sdk;
using namespace std::chrono_literals;

namespace {

// 절대 존재하지 않는 모델 경로 → MediaPipeDetector::initialize가 즉시 실패.
constexpr const char* kBadModelPath = "/nonexistent/iris_model_dir_zzz_p32_b1";

// 주어진 작업을 별도 스레드에서 실행하고, deadline 내 완료되지 않으면 false 반환.
// (테스트가 데드락/행에 걸렸을 때 전체 스위트가 멈추는 것을 막는 가드)
template <typename Fn>
bool runWithGuard(Fn&& fn, std::chrono::milliseconds deadline) {
    auto fut = std::async(std::launch::async, std::forward<Fn>(fn));
    return fut.wait_for(deadline) == std::future_status::ready;
}

}  // namespace

// ① 초기화 실패 후 소멸 시 std::terminate 없음.
//    (joinable 스레드를 stop()이 상태 무관하게 join하는지 검증)
TEST(InferenceThreadLifecycle, InitFailThenDestroyNoTerminate) {
    bool completed = runWithGuard([] {
        InferenceThread thread;
        bool started = thread.start(kBadModelPath, /*gpu_enabled=*/false);
        // 잘못된 경로이므로 start는 실패해야 한다.
        EXPECT_FALSE(started);
        EXPECT_FALSE(thread.isRunning());
        // 스코프 종료 → 소멸자 → stop() → joinable 스레드 join.
        // 수리 전이라면 여기서 std::terminate(프로세스 abort)가 발생했다.
    }, 5s);

    ASSERT_TRUE(completed) << "초기화 실패 후 소멸이 시한 내 완료되지 않음(행/데드락 의심)";
}

// ② 초기화 실패 후 stop()/재start() 안전(이중 stop 포함 무해성).
TEST(InferenceThreadLifecycle, InitFailThenStopAndRestartSafe) {
    bool completed = runWithGuard([] {
        InferenceThread thread;

        EXPECT_FALSE(thread.start(kBadModelPath, false));

        // 명시적 stop() — 이미 정지 상태여도 무해해야 한다.
        thread.stop();
        EXPECT_FALSE(thread.isRunning());

        // 이중 stop — 두 번째 호출도 안전(joinable 아님).
        thread.stop();
        EXPECT_FALSE(thread.isRunning());

        // 같은 객체로 재start() — joinable 스레드에 move-대입되어 terminate되면 안 됨.
        EXPECT_FALSE(thread.start(kBadModelPath, false));
        EXPECT_FALSE(thread.isRunning());

        thread.stop();
    }, 5s);

    ASSERT_TRUE(completed) << "재시작/이중 stop 경로가 시한 내 완료되지 않음(행/데드락 의심)";
}

// ③ Starting 직후 stop() 경합 반복 — Stopping 덮어쓰기로 인한 join 영구 행이 없는지.
//    start()가 내부적으로 초기화 완료까지 대기하므로, 여러 start/stop 사이클을
//    빠르게 반복해 상태 전이 경합을 흔들어도 어떤 사이클도 행에 빠지지 않아야 한다.
//
//    NOTE(③-2 B1): start()/stop() 제어 경로가 lifecycle_mutex_로 직렬화되도록
//    수리되었으므로, 한 스레드에서 start() 중인 동안 다른 스레드가 stop()을
//    호출하는 본 패턴은 이제 합법(데이터 레이스 없음)이다. 따라서 회귀로 유지한다.
TEST(InferenceThreadLifecycle, RepeatedStartStopRaceNoHang) {
    bool completed = runWithGuard([] {
        for (int i = 0; i < 20; ++i) {
            InferenceThread thread;
            // start는 별도 스레드에서 시작시키고, 메인에서 곧바로 stop을 호출해
            // Starting 윈도우와 stop을 겹치게 한다.
            std::thread starter([&thread] {
                thread.start(kBadModelPath, false);
            });
            // 약간의 지터를 주어 Starting 구간과 stop이 다양한 타이밍에 겹치게 함.
            if (i % 2 == 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(i * 50));
            }
            thread.stop();
            if (starter.joinable()) {
                starter.join();
            }
            // 최종적으로 항상 정지 상태여야 한다.
            EXPECT_FALSE(thread.isRunning());
        }
    }, 15s);

    ASSERT_TRUE(completed)
        << "Starting/stop 경합 반복이 시한 내 완료되지 않음(Stopping 덮어쓰기 → join 행 의심)";
}

// ③-b 이중 stop() 동시 호출 — lifecycle_mutex_가 joinable()/join()을 직렬화해
//      이중 join(UB)/데이터 레이스 없이 안전한지 회귀로 고정한다.
//      (important 1 수리: thread_/제어 경로를 lifecycle_mutex_로 보호)
//
//      여러 스레드가 같은 InferenceThread에 동시에 stop()을 호출한다. 수리 전이라면
//      두 스레드가 동시에 joinable()을 통과해 같은 thread_에 두 번 join() →
//      std::terminate(UB)에 빠질 수 있었다. 직렬화되면 정확히 한 번만 join하고
//      나머지 호출은 이미 Stopped인 thread_에 무해하게 통과해야 한다.
TEST(InferenceThreadLifecycle, ConcurrentDoubleStopIsSafe) {
    bool completed = runWithGuard([] {
        for (int i = 0; i < 20; ++i) {
            InferenceThread thread;
            // 실제로 워커 스레드가 (잠깐) 살아있도록 start를 시도한다.
            // bad path라 start는 곧 실패하지만, 워커 thread_는 join 대상으로 남는다.
            thread.start(kBadModelPath, false);

            constexpr int kStoppers = 4;
            std::vector<std::thread> stoppers;
            stoppers.reserve(kStoppers);
            for (int t = 0; t < kStoppers; ++t) {
                stoppers.emplace_back([&thread] {
                    // 같은 thread_에 대한 동시 stop — lifecycle_mutex_로 직렬화되어야 함.
                    thread.stop();
                });
            }
            for (auto& s : stoppers) {
                if (s.joinable()) {
                    s.join();
                }
            }
            // 동시 stop 이후 항상 정지 상태이고, 추가 stop도 무해해야 한다.
            EXPECT_FALSE(thread.isRunning());
            thread.stop();
            EXPECT_FALSE(thread.isRunning());
        }
    }, 15s);

    ASSERT_TRUE(completed)
        << "동시 이중 stop()이 시한 내 완료되지 않음(이중 join/데드락 의심)";
}

// ④ detectSync 호출이 상태기계를 영구 고착시키지 않는지(반복 호출 일관성).
//    실모델이 없으므로 5초 추론 타임아웃을 직접 유발할 수는 없다. 대신 init 실패로
//    워커가 없는 상태에서 detectSync를 반복 호출해도 매번 일관되게 비검출을 반환하고
//    상태가 깨지지 않음을 확인한다. (타임아웃 후 ResultReady 영구 고착 결함의
//    모델-비의존 대용 검증.)
TEST(InferenceThreadLifecycle, DetectSyncRepeatedCallsRemainConsistent) {
    bool completed = runWithGuard([] {
        InferenceThread thread;
        EXPECT_FALSE(thread.start(kBadModelPath, false));

        // 더미 RGB 프레임 (8x8x3).
        const int w = 8, h = 8;
        std::vector<uint8_t> frame(static_cast<size_t>(w) * h * 3, 128);

        for (int i = 0; i < 5; ++i) {
            IrisResult r = thread.detectSync(
                frame.data(), w, h, static_cast<int>(FrameFormat::RGB));
            // 워커가 없으므로 비검출. 핵심은 stuck 없이 매번 즉시 반환된다는 것.
            EXPECT_FALSE(r.detected);
        }
        EXPECT_FALSE(thread.isRunning());

        thread.stop();
    }, 5s);

    ASSERT_TRUE(completed)
        << "detectSync 반복 호출이 시한 내 완료되지 않음(상태 고착/행 의심)";
}

// 추가: nullptr/잘못된 입력에 대한 detectSync 방어(비검출 반환, 크래시 없음).
TEST(InferenceThreadLifecycle, DetectSyncRejectsInvalidInput) {
    bool completed = runWithGuard([] {
        InferenceThread thread;
        thread.start(kBadModelPath, false);

        IrisResult r1 = thread.detectSync(nullptr, 8, 8,
                                          static_cast<int>(FrameFormat::RGB));
        EXPECT_FALSE(r1.detected);

        std::vector<uint8_t> frame(8 * 8 * 3, 0);
        IrisResult r2 = thread.detectSync(frame.data(), 0, 8,
                                          static_cast<int>(FrameFormat::RGB));
        EXPECT_FALSE(r2.detected);

        thread.stop();
    }, 5s);

    ASSERT_TRUE(completed) << "잘못된 입력 처리가 시한 내 완료되지 않음";
}
