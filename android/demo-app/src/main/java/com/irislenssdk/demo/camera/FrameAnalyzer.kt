/**
 * IrisLensSDK Android - FrameAnalyzer
 *
 * CameraX ImageAnalysis 기반 프레임 분석기
 * - YUV_420_888 → NV21 변환
 * - IrisLensSDK 홍채 검출 호출
 * - FPS 계산
 *
 * @version 1.0.0
 */
package com.irislenssdk.demo.camera

import android.util.Log
import androidx.camera.core.ImageProxy
import com.irislenssdk.IrisLensSDK
import com.irislenssdk.IrisResult

/**
 * 프레임 분석 결과 콜백
 *
 * @property result 홍채 검출 결과
 * @property processingTimeMs 처리 시간 (밀리초)
 * @property fps 현재 FPS
 */
data class AnalysisResult(
    val result: IrisResult,
    val processingTimeMs: Long,
    val fps: Float
)

/**
 * 프레임 분석기
 *
 * @param onResult 분석 결과 콜백 (메인 스레드에서 호출)
 */
class FrameAnalyzer(
    private val onResult: (AnalysisResult) -> Unit
) {
    companion object {
        private const val TAG = "FrameAnalyzer"

        // 분석 간격 (약 30fps)
        private const val MIN_INTERVAL_MS = 33L

        // FPS 계산 윈도우
        private const val FPS_WINDOW_SIZE = 10
    }

    // NV21 버퍼 (재사용)
    private var nv21Buffer: ByteArray? = null

    // 결과 객체 (재사용)
    private val irisResult = IrisResult()

    // 타이밍
    private var lastAnalysisTime = 0L
    private val processingTimes = ArrayDeque<Long>(FPS_WINDOW_SIZE)

    // 프레임 카운터
    private var frameCount = 0L
    private var lastFpsUpdateTime = 0L
    private var currentFps = 0f

    /**
     * 프레임 분석 수행
     *
     * @param imageProxy CameraX ImageProxy
     */
    fun analyze(imageProxy: ImageProxy) {
        val currentTime = System.currentTimeMillis()

        // 프레임 스킵 (30fps 제한)
        if (currentTime - lastAnalysisTime < MIN_INTERVAL_MS) {
            imageProxy.close()
            return
        }

        try {
            val width = imageProxy.width
            val height = imageProxy.height

            // FPS 계산 (1초마다 업데이트) - SDK 상태와 무관하게 동작
            frameCount++
            if (currentTime - lastFpsUpdateTime >= 1000) {
                currentFps = frameCount * 1000f / (currentTime - lastFpsUpdateTime)
                frameCount = 0
                lastFpsUpdateTime = currentTime
            }

            // SDK 준비 상태 확인
            if (!IrisLensSDK.isReady()) {
                // SDK가 준비되지 않아도 FPS 업데이트 및 빈 결과 전달
                irisResult.reset()
                onResult(
                    AnalysisResult(
                        result = irisResult,
                        processingTimeMs = 0,
                        fps = currentFps
                    )
                )
                lastAnalysisTime = currentTime
                imageProxy.close()
                return
            }

            // YUV_420_888 → NV21 변환
            val nv21 = imageProxyToNV21(imageProxy)

            // 이미지 회전 정보 (카메라 센서 방향)
            val rotationDegrees = imageProxy.imageInfo.rotationDegrees

            // 홍채 검출 (회전 보정 적용)
            val startTime = System.nanoTime()

            irisResult.reset()
            val error = IrisLensSDK.detectWithRotation(
                nv21, width, height, IrisLensSDK.FORMAT_NV21, rotationDegrees, irisResult
            )

            val processingTimeMs = (System.nanoTime() - startTime) / 1_000_000

            // 처리 시간 기록
            processingTimes.addLast(processingTimeMs)
            if (processingTimes.size > FPS_WINDOW_SIZE) {
                processingTimes.removeFirst()
            }

            // 결과 전달
            if (error == IrisLensSDK.OK || error == IrisLensSDK.NO_FACE) {
                onResult(
                    AnalysisResult(
                        result = irisResult,
                        processingTimeMs = processingTimeMs,
                        fps = currentFps
                    )
                )
            } else {
                Log.w(TAG, "Detection error: ${IrisLensSDK.errorToString(error)}")
                // 에러가 발생해도 빈 결과 전달하여 UI 업데이트
                irisResult.reset()
                onResult(
                    AnalysisResult(
                        result = irisResult,
                        processingTimeMs = processingTimeMs,
                        fps = currentFps
                    )
                )
            }

            lastAnalysisTime = currentTime

        } catch (e: Exception) {
            Log.e(TAG, "Analysis failed", e)
        } finally {
            imageProxy.close()
        }
    }

    /**
     * ImageProxy를 NV21 바이트 배열로 변환
     *
     * YUV_420_888 → NV21 변환
     */
    private fun imageProxyToNV21(imageProxy: ImageProxy): ByteArray {
        val width = imageProxy.width
        val height = imageProxy.height
        val ySize = width * height
        val uvSize = width * height / 2
        val totalSize = ySize + uvSize

        // 버퍼 재사용 또는 생성
        val buffer = nv21Buffer?.takeIf { it.size == totalSize }
            ?: ByteArray(totalSize).also { nv21Buffer = it }

        val planes = imageProxy.planes

        // Y 평면 복사
        val yPlane = planes[0]
        val yBuffer = yPlane.buffer
        val yRowStride = yPlane.rowStride

        if (yRowStride == width) {
            // 연속 메모리 - 직접 복사
            yBuffer.get(buffer, 0, ySize)
        } else {
            // 행별 복사 (패딩 처리)
            var yOffset = 0
            for (row in 0 until height) {
                yBuffer.position(row * yRowStride)
                yBuffer.get(buffer, yOffset, width)
                yOffset += width
            }
        }

        // UV 평면 처리
        val uPlane = planes[1]
        val vPlane = planes[2]
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer
        val uvPixelStride = uPlane.pixelStride
        val uvRowStride = uPlane.rowStride

        var uvOffset = ySize

        if (uvPixelStride == 2 && uvRowStride == width) {
            // 인터리브된 UV (대부분의 기기)
            // VU 순서로 이미 되어 있는 경우
            vBuffer.get(buffer, uvOffset, uvSize - 1)
        } else {
            // 평면형 UV - 수동 인터리브
            for (row in 0 until height / 2) {
                for (col in 0 until width / 2) {
                    val uvIndex = row * uvRowStride + col * uvPixelStride
                    buffer[uvOffset++] = vBuffer.get(uvIndex)
                    buffer[uvOffset++] = uBuffer.get(uvIndex)
                }
            }
        }

        return buffer
    }

    /**
     * 평균 처리 시간 반환
     */
    fun getAverageProcessingTimeMs(): Float {
        if (processingTimes.isEmpty()) return 0f
        return processingTimes.average().toFloat()
    }

    /**
     * 리소스 해제
     */
    fun release() {
        nv21Buffer = null
        processingTimes.clear()
    }
}
