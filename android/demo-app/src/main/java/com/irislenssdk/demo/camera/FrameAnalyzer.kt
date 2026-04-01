/**
 * IrisLensSDK Android - FrameAnalyzer
 *
 * CameraX ImageAnalysis 기반 프레임 분석기
 * - YUV_420_888 → NV21 변환
 * - IrisLensSDK 홍채 검출 호출
 * - 뷰티 필터 적용 및 Bitmap 변환
 * - FPS 계산
 *
 * @version 1.1.0
 */
package com.irislenssdk.demo.camera

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.util.Log
import androidx.camera.core.ImageProxy
import com.irislenssdk.BeautyFilterConfigV2
import com.irislenssdk.IrisLensSDK
import com.irislenssdk.IrisResult
import java.io.ByteArrayOutputStream

/**
 * 프레임 분석 결과 콜백
 *
 * @property result 홍채 검출 결과
 * @property processingTimeMs 처리 시간 (밀리초)
 * @property fps 현재 FPS
 * @property filteredFrame 뷰티 필터가 적용된 프레임 (Bitmap, null이면 필터 미적용)
 */
data class AnalysisResult(
    val result: IrisResult,
    val processingTimeMs: Long,
    val fps: Float,
    val filteredFrame: Bitmap? = null
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

        // 분석 간격 (약 60fps 목표)
        private const val MIN_INTERVAL_MS = 16L

        // FPS 계산 윈도우
        private const val FPS_WINDOW_SIZE = 10

        // 뷰티 필터 렌더링용 다운스케일 비율 (1 = 원본, 2 = 1/2, 4 = 1/4)
        // 높을수록 빠르지만 품질 저하
        private const val BEAUTY_DOWNSCALE_FACTOR = 2  // 품질 개선: 3 → 2

        // 프레임 스킵 (필터 적용 빈도: 1 = 매 프레임, 2 = 2프레임마다, 3 = 3프레임마다)
        // 60fps 검출 기준 → 2 = 30fps 필터, 3 = 20fps 필터
        private const val BEAUTY_FILTER_FRAME_SKIP = 2  // 부드러움 개선: 3 → 2
    }

    // Temporal Stabilizer (SDK 코어)
    private var stabilizerHandle: Long = 0

    // NV21 버퍼 (재사용)
    private var nv21Buffer: ByteArray? = null

    // 결과 객체 (재사용)
    private val irisResult = IrisResult()

    // 뷰티 필터 활성화 플래그
    var beautyFilterEnabled: Boolean = false

    // V2 뷰티 필터 설정 (외부에서 설정)
    var beautyConfigV2: BeautyFilterConfigV2? = null

    // 프레임 스킵 카운터 (뷰티 필터 적용 빈도 조절)
    private var beautyFrameCounter = 0
    private var cachedFilteredBitmap: Bitmap? = null

    // JNI 변환용 RGBA Bitmap (재사용)
    private var rgbaBitmap: Bitmap? = null
    private var rgbaBitmapWidth: Int = 0
    private var rgbaBitmapHeight: Int = 0

    // Java 폴백용 버퍼 (JNI 실패 시)
    private var jpegOutputStream: ByteArrayOutputStream? = null

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

            // Temporal Stabilizer 적용 (SDK 코어 스무딩)
            // 검출 실패 프레임도 반드시 전달해야 hold/fade-out이 동작함
            if (error == IrisLensSDK.OK || error == IrisLensSDK.NO_FACE) {
                if (stabilizerHandle == 0L) {
                    stabilizerHandle = IrisLensSDK.createStabilizer()
                }
                if (stabilizerHandle != 0L) {
                    val timestampSec = System.nanoTime() / 1_000_000_000.0
                    IrisLensSDK.stabilize(stabilizerHandle, irisResult, timestampSec)
                }
            }

            // 처리 시간 기록
            processingTimes.addLast(processingTimeMs)
            if (processingTimes.size > FPS_WINDOW_SIZE) {
                processingTimes.removeFirst()
            }

            // 뷰티 필터 적용 (활성화된 경우, 프레임 스킵 적용)
            var filteredBitmap: Bitmap? = null
            val configV2 = beautyConfigV2
            if (beautyFilterEnabled && configV2 != null && configV2.enabled) {
                beautyFrameCounter++

                // 프레임 스킵: N프레임마다 필터 적용, 나머지는 캐시 사용
                if (beautyFrameCounter % BEAUTY_FILTER_FRAME_SKIP == 0) {
                    // V2 API 시도, 실패 시 V1 폴백
                    var filterError = IrisLensSDK.applyBeautyFilterV2(
                        nv21, width, height, IrisLensSDK.FORMAT_NV21,
                        configV2, irisResult
                    )

                    // V2 실패 시 V1으로 폴백
                    if (filterError != IrisLensSDK.OK) {
                        Log.w(TAG, "Beauty V2 failed ($filterError), falling back to V1")
                        filterError = IrisLensSDK.applyBeautyFilter(
                            nv21, width, height, IrisLensSDK.FORMAT_NV21
                        )
                    }

                    if (filterError == IrisLensSDK.OK) {
                        // NV21 → Bitmap 변환 후 캐시
                        cachedFilteredBitmap?.recycle()
                        cachedFilteredBitmap = nv21ToBitmap(nv21, width, height, rotationDegrees)
                        filteredBitmap = cachedFilteredBitmap
                    } else {
                        Log.w(TAG, "Beauty filter error: ${IrisLensSDK.errorToString(filterError)}")
                        // 필터 실패해도 원본 프레임 표시
                        cachedFilteredBitmap?.recycle()
                        cachedFilteredBitmap = nv21ToBitmap(nv21, width, height, rotationDegrees)
                        filteredBitmap = cachedFilteredBitmap
                    }
                } else {
                    // 캐시된 비트맵 재사용 (프레임 스킵)
                    filteredBitmap = cachedFilteredBitmap
                }
            } else {
                // 필터 비활성화 시 캐시 정리
                cachedFilteredBitmap?.recycle()
                cachedFilteredBitmap = null
                beautyFrameCounter = 0
            }

            // 결과 전달
            if (error == IrisLensSDK.OK || error == IrisLensSDK.NO_FACE) {
                onResult(
                    AnalysisResult(
                        result = irisResult,
                        processingTimeMs = processingTimeMs,
                        fps = currentFps,
                        filteredFrame = filteredBitmap
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
                        fps = currentFps,
                        filteredFrame = filteredBitmap
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
     * NV21 바이트 배열을 Bitmap으로 변환 (JNI 고속 변환)
     *
     * OpenCV를 사용한 직접 색공간 변환으로 Java JPEG 방식 대비 10배 이상 빠름.
     * - Java (YuvImage → JPEG → Bitmap): 50-100ms
     * - JNI (OpenCV cvtColor): 5-10ms
     *
     * @param nv21 NV21 포맷 바이트 배열
     * @param width 프레임 너비
     * @param height 프레임 높이
     * @param rotation 회전 각도 (0, 90, 180, 270)
     * @return 변환된 Bitmap
     */
    private fun nv21ToBitmap(nv21: ByteArray, width: Int, height: Int, rotation: Int): Bitmap {
        // RGBA Bitmap 재사용 또는 생성
        val targetWidth = width / BEAUTY_DOWNSCALE_FACTOR
        val targetHeight = height / BEAUTY_DOWNSCALE_FACTOR

        // JNI 변환은 원본 크기 필요, 이후 다운스케일 적용
        val bitmap = getOrCreateRgbaBitmap(width, height)

        // JNI 고속 변환 시도
        val error = IrisLensSDK.nv21ToRgba(nv21, width, height, bitmap)

        val resultBitmap = if (error == IrisLensSDK.OK) {
            // JNI 변환 성공 - 다운스케일 적용
            if (BEAUTY_DOWNSCALE_FACTOR > 1) {
                Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)
            } else {
                // 복사본 반환 (원본 버퍼는 재사용)
                bitmap.copy(Bitmap.Config.ARGB_8888, false)
            }
        } else {
            // JNI 실패 시 Java 폴백
            Log.w(TAG, "JNI conversion failed ($error), falling back to Java JPEG method")
            nv21ToBitmapFallback(nv21, width, height)
        }

        // 회전 적용 (카메라 센서 방향 보정)
        return if (rotation != 0) {
            val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
            val rotatedBitmap = Bitmap.createBitmap(
                resultBitmap, 0, 0, resultBitmap.width, resultBitmap.height, matrix, true
            )
            if (rotatedBitmap != resultBitmap) {
                resultBitmap.recycle()
            }
            rotatedBitmap
        } else {
            resultBitmap
        }
    }

    /**
     * RGBA Bitmap 버퍼 재사용 또는 생성
     */
    private fun getOrCreateRgbaBitmap(width: Int, height: Int): Bitmap {
        val existing = rgbaBitmap
        if (existing != null && rgbaBitmapWidth == width && rgbaBitmapHeight == height && !existing.isRecycled) {
            return existing
        }

        // 기존 버퍼 해제
        existing?.recycle()

        // 새 버퍼 생성 (ARGB_8888 필수)
        val newBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        rgbaBitmap = newBitmap
        rgbaBitmapWidth = width
        rgbaBitmapHeight = height
        return newBitmap
    }

    /**
     * Java 기반 NV21 → Bitmap 변환 (폴백용)
     *
     * JNI 변환 실패 시 사용하는 기존 방식.
     */
    private fun nv21ToBitmapFallback(nv21: ByteArray, width: Int, height: Int): Bitmap {
        // YuvImage로 JPEG 압축
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)

        // 출력 스트림 재사용 또는 생성
        val outputStream = jpegOutputStream?.also { it.reset() }
            ?: ByteArrayOutputStream().also { jpegOutputStream = it }

        // JPEG 압축 (품질 80%로 속도 우선)
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 80, outputStream)
        val jpegData = outputStream.toByteArray()

        // JPEG → Bitmap 디코딩 (다운스케일 적용)
        val options = BitmapFactory.Options().apply {
            inMutable = false
            inSampleSize = BEAUTY_DOWNSCALE_FACTOR
        }
        return BitmapFactory.decodeByteArray(jpegData, 0, jpegData.size, options)
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
        if (stabilizerHandle != 0L) {
            IrisLensSDK.destroyStabilizer(stabilizerHandle)
            stabilizerHandle = 0
        }
        nv21Buffer = null
        jpegOutputStream = null
        rgbaBitmap?.recycle()
        rgbaBitmap = null
        rgbaBitmapWidth = 0
        rgbaBitmapHeight = 0
        processingTimes.clear()
    }
}
