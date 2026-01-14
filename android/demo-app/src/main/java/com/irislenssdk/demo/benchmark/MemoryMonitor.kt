/**
 * IrisLensSDK Android - Memory Monitor
 *
 * 앱 메모리 사용량 모니터링
 * - JVM Heap 메모리
 * - Native Heap 메모리
 * - 전체 PSS (Proportional Set Size)
 *
 * @version 1.0.0
 */
package com.irislenssdk.demo.benchmark

import android.app.ActivityManager
import android.content.Context
import android.os.Debug

/**
 * 메모리 정보 데이터 클래스
 */
data class MemoryInfo(
    val jvmUsedMB: Long,
    val jvmMaxMB: Long,
    val jvmFreePercent: Float,
    val nativeHeapMB: Long,
    val nativeHeapAllocatedMB: Long,
    val totalPssMB: Int,
    val timestamp: Long = System.currentTimeMillis()
) {
    /**
     * 메모리 상태가 정상인지 확인 (100MB 이하)
     */
    val isHealthy: Boolean
        get() = totalPssMB < 100

    /**
     * 상태 문자열
     */
    fun getStatusString(): String {
        return buildString {
            append("PSS: ${totalPssMB}MB")
            if (!isHealthy) append(" ⚠️")
            append(" | Native: ${nativeHeapMB}MB")
            append(" | JVM: ${jvmUsedMB}/${jvmMaxMB}MB")
        }
    }
}

/**
 * 메모리 모니터
 *
 * Android 메모리 사용량을 실시간으로 모니터링
 */
class MemoryMonitor(private val context: Context) {

    companion object {
        private const val BYTES_TO_MB = 1024L * 1024L
        private const val KB_TO_MB = 1024
    }

    private val activityManager: ActivityManager by lazy {
        context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
    }

    // 메모리 히스토리 (누수 감지용)
    private val memoryHistory = mutableListOf<MemoryInfo>()
    private val maxHistorySize = 60  // 60개 샘플 유지

    /**
     * 현재 메모리 정보 반환
     */
    fun getMemoryInfo(): MemoryInfo {
        val runtime = Runtime.getRuntime()

        // JVM 메모리
        val jvmTotal = runtime.totalMemory()
        val jvmFree = runtime.freeMemory()
        val jvmMax = runtime.maxMemory()
        val jvmUsed = jvmTotal - jvmFree

        val jvmUsedMB = jvmUsed / BYTES_TO_MB
        val jvmMaxMB = jvmMax / BYTES_TO_MB
        val jvmFreePercent = (jvmFree.toFloat() / jvmTotal) * 100f

        // Native 메모리
        val nativeHeapSize = Debug.getNativeHeapSize() / BYTES_TO_MB
        val nativeHeapAllocated = Debug.getNativeHeapAllocatedSize() / BYTES_TO_MB

        // 전체 PSS (Proportional Set Size)
        val processInfo = Debug.MemoryInfo()
        Debug.getMemoryInfo(processInfo)
        val totalPssMB = processInfo.totalPss / KB_TO_MB

        val info = MemoryInfo(
            jvmUsedMB = jvmUsedMB,
            jvmMaxMB = jvmMaxMB,
            jvmFreePercent = jvmFreePercent,
            nativeHeapMB = nativeHeapSize,
            nativeHeapAllocatedMB = nativeHeapAllocated,
            totalPssMB = totalPssMB
        )

        // 히스토리에 추가
        synchronized(memoryHistory) {
            memoryHistory.add(info)
            if (memoryHistory.size > maxHistorySize) {
                memoryHistory.removeAt(0)
            }
        }

        return info
    }

    /**
     * 상세 메모리 정보 반환 (디버그용)
     */
    fun getDetailedMemoryInfo(): Map<String, Any> {
        val memInfo = Debug.MemoryInfo()
        Debug.getMemoryInfo(memInfo)

        return mapOf(
            "dalvikPss" to memInfo.dalvikPss / KB_TO_MB,
            "nativePss" to memInfo.nativePss / KB_TO_MB,
            "otherPss" to memInfo.otherPss / KB_TO_MB,
            "totalPss" to memInfo.totalPss / KB_TO_MB,
            "dalvikPrivateDirty" to memInfo.dalvikPrivateDirty / KB_TO_MB,
            "nativePrivateDirty" to memInfo.nativePrivateDirty / KB_TO_MB,
            "otherPrivateDirty" to memInfo.otherPrivateDirty / KB_TO_MB,
            "totalPrivateDirty" to memInfo.totalPrivateDirty / KB_TO_MB
        )
    }

    /**
     * 메모리 누수 감지
     *
     * 최근 샘플들의 메모리 증가 추세를 분석
     *
     * @return 누수가 의심되면 true
     */
    fun detectMemoryLeak(): Boolean {
        synchronized(memoryHistory) {
            if (memoryHistory.size < 10) return false

            // 최근 10개 샘플의 PSS 추세 분석
            val recentSamples = memoryHistory.takeLast(10)
            val firstHalf = recentSamples.take(5).map { it.totalPssMB }.average()
            val secondHalf = recentSamples.takeLast(5).map { it.totalPssMB }.average()

            // 10% 이상 증가하면 누수 의심
            val increaseRate = (secondHalf - firstHalf) / firstHalf
            return increaseRate > 0.1
        }
    }

    /**
     * 메모리 사용률 반환 (0.0 ~ 1.0)
     */
    fun getMemoryUsageRatio(): Float {
        val info = getMemoryInfo()
        return info.totalPssMB / 100f  // 100MB 기준
    }

    /**
     * 시스템 메모리 정보
     */
    fun getSystemMemoryInfo(): Map<String, Long> {
        val memInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)

        return mapOf(
            "availableMB" to memInfo.availMem / BYTES_TO_MB,
            "totalMB" to memInfo.totalMem / BYTES_TO_MB,
            "threshold" to memInfo.threshold / BYTES_TO_MB,
            "lowMemory" to if (memInfo.lowMemory) 1L else 0L
        )
    }

    /**
     * 히스토리 초기화
     */
    fun clearHistory() {
        synchronized(memoryHistory) {
            memoryHistory.clear()
        }
    }

    /**
     * GC 실행 요청
     */
    fun requestGC() {
        System.gc()
    }
}
