/**
 * IrisLensSDK - StabilityLogger
 *
 * Gate 1 안정성 검증을 위한 프레임별 CSV 로거.
 * raw/filtered 홍채 좌표, 눈꺼풀 경계, temporal hold,
 * 렌더링 소요 시간을 22 컬럼 CSV로 기록한다.
 *
 * 디버그 빌드에서만 동작하며, adb pull로 추출 가능.
 *
 * @see docs/workPaper/P4-W1-02_stability_baseline.md
 */
package com.irislenssdk.demo.util

import android.content.Context
import android.os.Environment
import android.util.Log
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class StabilityLogger(
    private val context: Context,
    private val gpuTier: String = "UNKNOWN",
    private val gpuRenderer: String = ""
) {
    companion object {
        private const val TAG = "StabilityLogger"
        private const val MAX_FRAMES = 900  // 30fps * 30sec

        private val CSV_HEADER = listOf(
            "frame_id",
            "timestamp_ms",
            "face_detected",
            "raw_left_cx",
            "raw_left_cy",
            "raw_left_r",
            "flt_left_cx",
            "flt_left_cy",
            "flt_left_r",
            "raw_right_cx",
            "raw_right_cy",
            "raw_right_r",
            "flt_right_cx",
            "flt_right_cy",
            "flt_right_r",
            "eyelid_lt",
            "eyelid_lb",
            "eyelid_rt",
            "eyelid_rb",
            "hold_active",
            "hold_remaining",
            "render_time_us"
        ).joinToString(",")
    }

    private var writer: BufferedWriter? = null
    private var frameCount = 0
    private var isLogging = false
    private var currentFile: File? = null

    val isActive: Boolean get() = isLogging

    /**
     * 로깅 시작. 새 CSV 파일을 생성한다.
     *
     * @return 생성된 파일 경로, 실패 시 null
     */
    fun start(): String? {
        if (isLogging) {
            Log.w(TAG, "Already logging, stop first")
            return null
        }

        val dir = context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)
            ?: run {
                Log.e(TAG, "External files dir not available")
                return null
            }

        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val file = File(dir, "iris_stability_log_$timestamp.csv")

        return try {
            writer = BufferedWriter(FileWriter(file))

            // 메타데이터 헤더 (# 주석)
            writer?.write("# gpu_tier=$gpuTier\n")
            writer?.write("# gpu_renderer=$gpuRenderer\n")
            writer?.write("# start_time=$timestamp\n")

            // CSV 헤더
            writer?.write(CSV_HEADER)
            writer?.newLine()

            frameCount = 0
            isLogging = true
            currentFile = file

            Log.d(TAG, "Logging started: ${file.absolutePath}")
            file.absolutePath
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start logging", e)
            try { writer?.close() } catch (_: Exception) {}
            writer = null
            null
        }
    }

    /**
     * 프레임 데이터 기록.
     * GL 스레드에서 호출되므로 최소한의 처리만 수행한다.
     * @Synchronized: stop()과의 크로스 스레드 경합 방지
     */
    @Synchronized
    fun logFrame(
        faceDetected: Boolean,
        rawLeftCx: Float,
        rawLeftCy: Float,
        rawLeftR: Float,
        filteredLeftCx: Float,
        filteredLeftCy: Float,
        filteredLeftR: Float,
        rawRightCx: Float,
        rawRightCy: Float,
        rawRightR: Float,
        filteredRightCx: Float,
        filteredRightCy: Float,
        filteredRightR: Float,
        eyelidLt: Float,
        eyelidLb: Float,
        eyelidRt: Float,
        eyelidRb: Float,
        holdActive: Boolean,
        holdRemaining: Int,
        renderTimeUs: Long
    ) {
        if (!isLogging) return
        if (frameCount >= MAX_FRAMES) {
            stop()
            return
        }

        val timestampMs = System.nanoTime() / 1_000_000L

        try {
            writer?.write(buildString {
                append(frameCount)
                append(',').append(timestampMs)
                append(',').append(if (faceDetected) 1 else 0)
                append(',').append(rawLeftCx)
                append(',').append(rawLeftCy)
                append(',').append(rawLeftR)
                append(',').append(filteredLeftCx)
                append(',').append(filteredLeftCy)
                append(',').append(filteredLeftR)
                append(',').append(rawRightCx)
                append(',').append(rawRightCy)
                append(',').append(rawRightR)
                append(',').append(filteredRightCx)
                append(',').append(filteredRightCy)
                append(',').append(filteredRightR)
                append(',').append(eyelidLt)
                append(',').append(eyelidLb)
                append(',').append(eyelidRt)
                append(',').append(eyelidRb)
                append(',').append(if (holdActive) 1 else 0)
                append(',').append(holdRemaining)
                append(',').append(renderTimeUs)
            })
            writer?.newLine()
            frameCount++
        } catch (e: Exception) {
            Log.e(TAG, "Failed to write frame $frameCount", e)
            stop()
        }
    }

    /**
     * 로깅 종료. 파일을 플러시/닫는다.
     * @Synchronized: logFrame()과의 크로스 스레드 경합 방지
     *
     * @return 기록된 프레임 수
     */
    @Synchronized
    fun stop(): Int {
        if (!isLogging) return 0

        val recorded = frameCount
        try {
            writer?.flush()
            writer?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to close writer", e)
        }

        writer = null
        isLogging = false

        Log.d(TAG, "Logging stopped: $recorded frames → ${currentFile?.name}")
        return recorded
    }

    /**
     * 현재 기록된 프레임 수
     */
    fun getFrameCount(): Int = frameCount

    /**
     * 현재 로그 파일 경로
     */
    fun getCurrentFilePath(): String? = currentFile?.absolutePath
}
