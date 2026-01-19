/**
 * IrisLensSDK Android - Benchmark Manager
 *
 * 성능 및 메모리 벤치마크 통합 관리
 * - 자동 샘플링
 * - 리포트 생성
 * - CSV 내보내기
 *
 * @version 1.0.0
 */
package com.irislenssdk.demo.benchmark

import android.content.Context
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.util.Log
import com.irislenssdk.demo.BuildConfig
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * 벤치마크 결과
 */
data class BenchmarkResult(
    val testDurationMs: Long,
    val performance: PerformanceStats,
    val memory: MemoryInfo,
    val deviceInfo: DeviceInfo,
    val timestamp: Long = System.currentTimeMillis()
)

/**
 * 디바이스 정보
 */
data class DeviceInfo(
    val manufacturer: String = Build.MANUFACTURER,
    val model: String = Build.MODEL,
    val device: String = Build.DEVICE,
    val androidVersion: String = Build.VERSION.RELEASE,
    val sdkVersion: Int = Build.VERSION.SDK_INT,
    val cpuAbi: String = Build.SUPPORTED_ABIS.firstOrNull() ?: "unknown"
)

/**
 * 벤치마크 상태
 */
enum class BenchmarkState {
    IDLE,
    RUNNING,
    PAUSED,
    COMPLETED
}

/**
 * 벤치마크 콜백
 */
interface BenchmarkCallback {
    fun onBenchmarkStarted()
    fun onBenchmarkProgress(elapsedMs: Long, stats: PerformanceStats, memory: MemoryInfo)
    fun onBenchmarkCompleted(result: BenchmarkResult)
    fun onBenchmarkError(error: String)
}

/**
 * 벤치마크 관리자
 */
class BenchmarkManager(private val context: Context) {

    companion object {
        private const val TAG = "BenchmarkManager"
        private const val SAMPLE_INTERVAL_MS = 1000L  // 1초마다 샘플링
    }

    // 트래커들
    val performanceTracker = PerformanceTracker()
    val memoryMonitor = MemoryMonitor(context)

    // 상태
    private var state = BenchmarkState.IDLE
    private var startTimeMs = 0L
    private var targetDurationMs = 0L

    // 핸들러
    private val handler = Handler(Looper.getMainLooper())
    private var sampleRunnable: Runnable? = null

    // 콜백
    private var callback: BenchmarkCallback? = null

    // 샘플 데이터
    private val performanceSamples = mutableListOf<PerformanceStats>()
    private val memorySamples = mutableListOf<MemoryInfo>()

    // SDK 모드 정보
    private var inferenceThreadEnabled = true  // 기본값: InferenceThread 사용
    private var customModeTag: String? = null  // 커스텀 모드 태그 (MediaPipe 등)

    /**
     * 벤치마크 시작
     *
     * @param durationMs 테스트 시간 (밀리초)
     * @param callback 콜백
     */
    fun start(durationMs: Long, callback: BenchmarkCallback) {
        if (state == BenchmarkState.RUNNING) {
            callback.onBenchmarkError("Benchmark already running")
            return
        }

        this.callback = callback
        this.targetDurationMs = durationMs
        this.startTimeMs = System.currentTimeMillis()
        this.state = BenchmarkState.RUNNING

        // 초기화
        performanceTracker.reset()
        memoryMonitor.clearHistory()
        performanceSamples.clear()
        memorySamples.clear()

        Log.i(TAG, "Benchmark started: duration=${durationMs}ms")
        callback.onBenchmarkStarted()

        // 샘플링 시작
        startSampling()
    }

    /**
     * 벤치마크 중지
     */
    fun stop() {
        if (state != BenchmarkState.RUNNING) return

        stopSampling()
        state = BenchmarkState.COMPLETED

        val result = createResult()
        Log.i(TAG, "Benchmark completed: ${result.testDurationMs}ms")
        callback?.onBenchmarkCompleted(result)
    }

    /**
     * 벤치마크 일시정지
     */
    fun pause() {
        if (state != BenchmarkState.RUNNING) return
        stopSampling()
        state = BenchmarkState.PAUSED
    }

    /**
     * 벤치마크 재개
     */
    fun resume() {
        if (state != BenchmarkState.PAUSED) return
        state = BenchmarkState.RUNNING
        startSampling()
    }

    /**
     * 프레임 처리 완료 이벤트
     */
    fun onFrameProcessed(latencyMs: Long) {
        if (state != BenchmarkState.RUNNING) return
        performanceTracker.onFrameProcessed(latencyMs)
    }

    /**
     * 현재 상태 반환
     */
    fun getState(): BenchmarkState = state

    /**
     * GPU 모드 설정
     */
    fun setGpuEnabled(enabled: Boolean) {
        performanceTracker.gpuEnabled = enabled
    }

    /**
     * InferenceThread 모드 설정
     */
    fun setInferenceThreadEnabled(enabled: Boolean) {
        inferenceThreadEnabled = enabled
    }

    /**
     * 커스텀 모드 태그 설정 (MediaPipe 등 외부 SDK 벤치마크용)
     *
     * @param tag 파일명에 사용할 모드 태그 (예: "mediapipe_cpu", "mediapipe_gpu")
     *            null이면 기본 thread/direct 로직 사용
     */
    fun setCustomModeTag(tag: String?) {
        customModeTag = tag
    }

    /**
     * 샘플링 시작
     */
    private fun startSampling() {
        sampleRunnable = object : Runnable {
            override fun run() {
                if (state != BenchmarkState.RUNNING) return

                val elapsedMs = System.currentTimeMillis() - startTimeMs

                // 샘플 수집
                val perfStats = performanceTracker.getStats()
                val memInfo = memoryMonitor.getMemoryInfo()

                performanceSamples.add(perfStats)
                memorySamples.add(memInfo)

                // 콜백
                callback?.onBenchmarkProgress(elapsedMs, perfStats, memInfo)

                // 시간 초과 체크
                if (elapsedMs >= targetDurationMs) {
                    stop()
                } else {
                    handler.postDelayed(this, SAMPLE_INTERVAL_MS)
                }
            }
        }

        handler.post(sampleRunnable!!)
    }

    /**
     * 샘플링 중지
     */
    private fun stopSampling() {
        sampleRunnable?.let { handler.removeCallbacks(it) }
        sampleRunnable = null
    }

    /**
     * 결과 생성
     */
    private fun createResult(): BenchmarkResult {
        return BenchmarkResult(
            testDurationMs = System.currentTimeMillis() - startTimeMs,
            performance = performanceTracker.getStats(),
            memory = memoryMonitor.getMemoryInfo(),
            deviceInfo = DeviceInfo()
        )
    }

    /**
     * CSV 파일로 내보내기
     *
     * 파일명 형식: benchmark_[mode]_b[buildNumber]_[timestamp].csv
     * - mode: "thread" (InferenceThread 사용) 또는 "direct" (직접 호출)
     * - buildNumber: 앱 versionCode (빌드마다 증가시켜 구분)
     */
    fun exportToCsv(): File? {
        if (performanceSamples.isEmpty()) return null

        val dateFormat = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault())
        // 커스텀 모드 태그가 있으면 사용, 없으면 기본 thread/direct 로직 사용
        val modeTag = customModeTag ?: if (inferenceThreadEnabled) "thread" else "direct"
        val buildNumber = BuildConfig.VERSION_CODE
        val fileName = "benchmark_${modeTag}_b${buildNumber}_${dateFormat.format(Date())}.csv"
        val file = File(context.getExternalFilesDir(null), fileName)

        try {
            file.bufferedWriter().use { writer ->
                // 헤더 (빌드 정보 컬럼 추가)
                writer.write("timestamp,fps,avgLatencyMs,minLatencyMs,maxLatencyMs,p95LatencyMs,")
                writer.write("p99LatencyMs,totalFrames,droppedFrames,dropRate,")
                writer.write("totalPssMB,nativeHeapMB,jvmUsedMB,gpuEnabled,inferenceThread,buildNumber,appVersion")
                writer.newLine()

                // 데이터
                val minSize = minOf(performanceSamples.size, memorySamples.size)
                for (i in 0 until minSize) {
                    val perf = performanceSamples[i]
                    val mem = memorySamples[i]

                    writer.write("${mem.timestamp},")
                    writer.write("${perf.fps},${perf.avgLatencyMs},${perf.minLatencyMs},")
                    writer.write("${perf.maxLatencyMs},${perf.p95LatencyMs},${perf.p99LatencyMs},")
                    writer.write("${perf.totalFrames},${perf.droppedFrames},${perf.dropRate},")
                    writer.write("${mem.totalPssMB},${mem.nativeHeapMB},${mem.jvmUsedMB},")
                    writer.write("${perf.gpuEnabled},${inferenceThreadEnabled},")
                    writer.write("${BuildConfig.VERSION_CODE},${BuildConfig.VERSION_NAME}")
                    writer.newLine()
                }
            }

            Log.i(TAG, "Exported to: ${file.absolutePath} (mode: $modeTag)")
            return file

        } catch (e: Exception) {
            Log.e(TAG, "Export failed", e)
            return null
        }
    }

    /**
     * 마크다운 리포트 생성
     */
    fun generateMarkdownReport(): String {
        val stats = performanceTracker.getStats()
        val memInfo = memoryMonitor.getMemoryInfo()
        val deviceInfo = DeviceInfo()
        val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())

        return buildString {
            appendLine("# IrisLensSDK Performance Report")
            appendLine()
            appendLine("## Test Environment")
            appendLine("- **Date**: ${dateFormat.format(Date())}")
            appendLine("- **App Version**: ${BuildConfig.VERSION_NAME} (build ${BuildConfig.VERSION_CODE})")
            appendLine("- **Device**: ${deviceInfo.manufacturer} ${deviceInfo.model}")
            appendLine("- **Android Version**: ${deviceInfo.androidVersion} (SDK ${deviceInfo.sdkVersion})")
            appendLine("- **CPU ABI**: ${deviceInfo.cpuAbi}")
            appendLine("- **GPU Mode**: ${if (stats.gpuEnabled) "Enabled" else "CPU Only"}")
            appendLine("- **Inference Mode**: ${if (inferenceThreadEnabled) "InferenceThread" else "Direct Call"}")
            appendLine()
            appendLine("## Performance Results")
            appendLine()
            appendLine("### FPS")
            appendLine("| Metric | Value | Target | Status |")
            appendLine("|--------|-------|--------|--------|")
            appendLine("| Current FPS | %.1f | ≥30 | ${if (stats.fps >= 30) "✅" else "❌"} |".format(stats.fps))
            appendLine()
            appendLine("### Latency")
            appendLine("| Metric | Value | Target | Status |")
            appendLine("|--------|-------|--------|--------|")
            appendLine("| Average | %.1f ms | ≤33ms | ${if (stats.avgLatencyMs <= 33) "✅" else "❌"} |".format(stats.avgLatencyMs))
            appendLine("| Min | ${stats.minLatencyMs} ms | - | - |")
            appendLine("| Max | ${stats.maxLatencyMs} ms | - | - |")
            appendLine("| P95 | ${stats.p95LatencyMs} ms | ≤50ms | ${if (stats.p95LatencyMs <= 50) "✅" else "❌"} |")
            appendLine("| P99 | ${stats.p99LatencyMs} ms | ≤100ms | ${if (stats.p99LatencyMs <= 100) "✅" else "❌"} |")
            appendLine()
            appendLine("### Frames")
            appendLine("| Metric | Value |")
            appendLine("|--------|-------|")
            appendLine("| Total Frames | ${stats.totalFrames} |")
            appendLine("| Dropped Frames | ${stats.droppedFrames} |")
            appendLine("| Drop Rate | %.2f%% |".format(stats.dropRate * 100))
            appendLine()
            appendLine("## Memory Usage")
            appendLine("| Metric | Value | Target | Status |")
            appendLine("|--------|-------|--------|--------|")
            appendLine("| Total PSS | ${memInfo.totalPssMB} MB | ≤100MB | ${if (memInfo.totalPssMB <= 100) "✅" else "❌"} |")
            appendLine("| Native Heap | ${memInfo.nativeHeapMB} MB | - | - |")
            appendLine("| JVM Heap | ${memInfo.jvmUsedMB}/${memInfo.jvmMaxMB} MB | - | - |")
            appendLine()
            appendLine("## Summary")
            val passCount = listOf(
                stats.fps >= 30,
                stats.avgLatencyMs <= 33,
                memInfo.totalPssMB <= 100
            ).count { it }
            appendLine("- **Pass Rate**: $passCount/3")
            if (passCount == 3) {
                appendLine("- **Status**: ✅ All targets met!")
            } else {
                appendLine("- **Status**: ⚠️ Some targets not met")
                if (stats.fps < 30) appendLine("  - FPS below target: Consider enabling GPU delegate")
                if (stats.avgLatencyMs > 33) appendLine("  - Latency above target: Optimize inference pipeline")
                if (memInfo.totalPssMB > 100) appendLine("  - Memory above target: Check for leaks")
            }
        }
    }

    /**
     * 리소스 해제
     */
    fun release() {
        stopSampling()
        performanceSamples.clear()
        memorySamples.clear()
        callback = null
    }
}
