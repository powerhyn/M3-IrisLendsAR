/**
 * MediaPipe Face Landmarker - Performance Tracker
 *
 * 실시간 성능 지표 추적
 * - FPS (1초 윈도우 기반)
 * - 검출 지연 시간 통계
 * - 프레임 드랍 감지
 */
package com.google.mediapipe.examples.facelandmarker.benchmark

import java.util.LinkedList

/**
 * 성능 통계 데이터 클래스
 */
data class PerformanceStats(
    val fps: Float,
    val avgLatencyMs: Float,
    val minLatencyMs: Long,
    val maxLatencyMs: Long,
    val p95LatencyMs: Long,
    val p99LatencyMs: Long,
    val totalFrames: Long,
    val droppedFrames: Long,
    val dropRate: Float,
    val gpuEnabled: Boolean = false
)

/**
 * 실시간 성능 추적기
 *
 * 스레드 안전하게 FPS, 지연시간, 프레임 드랍을 추적
 */
class PerformanceTracker {

    companion object {
        private const val WINDOW_SIZE_MS = 1000L  // 1초 윈도우
        private const val TARGET_FRAME_TIME_MS = 33L  // 30fps 목표
        private const val LATENCY_HISTORY_SIZE = 1000  // 백분위 계산용
    }

    // FPS 계산용 타임스탬프
    private val frameTimestamps = LinkedList<Long>()

    // 지연 시간 히스토리 (백분위 계산용)
    private val latencyHistory = LinkedList<Long>()

    // 통계
    private var totalFrames = 0L
    private var droppedFrames = 0L
    private var minLatency = Long.MAX_VALUE
    private var maxLatency = 0L
    private var sumLatency = 0L

    // GPU 모드
    var gpuEnabled: Boolean = false

    // 동기화 객체
    private val lock = Any()

    /**
     * 프레임 처리 완료 시 호출
     *
     * @param latencyMs 처리 지연 시간 (밀리초)
     */
    fun onFrameProcessed(latencyMs: Long) {
        val now = System.currentTimeMillis()

        synchronized(lock) {
            // FPS 계산을 위한 타임스탬프 추가
            frameTimestamps.add(now)

            // 1초 이전 타임스탬프 제거
            while (frameTimestamps.isNotEmpty() &&
                   now - frameTimestamps.first() > WINDOW_SIZE_MS) {
                frameTimestamps.removeFirst()
            }

            // 지연 시간 히스토리 추가
            latencyHistory.add(latencyMs)
            if (latencyHistory.size > LATENCY_HISTORY_SIZE) {
                latencyHistory.removeFirst()
            }

            // 통계 업데이트
            totalFrames++
            sumLatency += latencyMs

            if (latencyMs < minLatency) minLatency = latencyMs
            if (latencyMs > maxLatency) maxLatency = latencyMs

            // 프레임 드랍 감지 (33ms 초과)
            if (latencyMs > TARGET_FRAME_TIME_MS) {
                droppedFrames++
            }
        }
    }

    /**
     * 현재 FPS 반환
     */
    fun getCurrentFps(): Float {
        synchronized(lock) {
            return frameTimestamps.size.toFloat()
        }
    }

    /**
     * 평균 지연 시간 반환
     */
    fun getAverageLatency(): Float {
        synchronized(lock) {
            return if (totalFrames > 0) sumLatency.toFloat() / totalFrames else 0f
        }
    }

    /**
     * 백분위 지연 시간 계산
     *
     * @param percentile 백분위 (0.0 ~ 1.0)
     */
    fun getPercentileLatency(percentile: Float): Long {
        synchronized(lock) {
            if (latencyHistory.isEmpty()) return 0L

            val sorted = latencyHistory.sorted()
            val index = ((sorted.size - 1) * percentile).toInt()
            return sorted[index]
        }
    }

    /**
     * 전체 성능 통계 반환
     */
    fun getStats(): PerformanceStats {
        synchronized(lock) {
            return PerformanceStats(
                fps = getCurrentFps(),
                avgLatencyMs = getAverageLatency(),
                minLatencyMs = if (minLatency == Long.MAX_VALUE) 0L else minLatency,
                maxLatencyMs = maxLatency,
                p95LatencyMs = getPercentileLatency(0.95f),
                p99LatencyMs = getPercentileLatency(0.99f),
                totalFrames = totalFrames,
                droppedFrames = droppedFrames,
                dropRate = if (totalFrames > 0) droppedFrames.toFloat() / totalFrames else 0f,
                gpuEnabled = gpuEnabled
            )
        }
    }

    /**
     * 통계 리셋
     */
    fun reset() {
        synchronized(lock) {
            frameTimestamps.clear()
            latencyHistory.clear()
            totalFrames = 0
            droppedFrames = 0
            minLatency = Long.MAX_VALUE
            maxLatency = 0
            sumLatency = 0
        }
    }

    /**
     * 성능 상태 문자열 반환
     */
    fun getStatusString(): String {
        val stats = getStats()
        return buildString {
            append("FPS: %.1f".format(stats.fps))
            if (stats.fps < 30) append(" ⚠️")
            append(" | Latency: %.0fms".format(stats.avgLatencyMs))
            if (stats.avgLatencyMs > TARGET_FRAME_TIME_MS) append(" ⚠️")
            append(" | Drop: %.1f%%".format(stats.dropRate * 100))
            if (gpuEnabled) append(" | GPU")
        }
    }
}
