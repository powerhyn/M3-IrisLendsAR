/**
 * IrisLensSDK Android - CameraManager
 *
 * CameraX 카메라 라이프사이클 관리
 * - Preview Use Case
 * - ImageAnalysis Use Case
 * - 전면/후면 카메라 전환
 *
 * @version 1.0.0
 */
package com.irislenssdk.demo.camera

import android.content.Context
import android.util.Log
import android.util.Size
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * CameraX 기반 카메라 관리 클래스
 *
 * @param context Android Context
 * @param lifecycleOwner 라이프사이클 소유자 (Activity/Fragment)
 */
class CameraManager(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner
) {
    companion object {
        private const val TAG = "CameraManager"

        // 프리뷰 해상도 (16:9)
        private val PREVIEW_SIZE = Size(1280, 720)

        // 분석용 해상도 (성능 최적화)
        private val ANALYSIS_SIZE = Size(640, 480)
    }

    // CameraX 컴포넌트
    private var cameraProvider: ProcessCameraProvider? = null
    private var imageAnalysis: ImageAnalysis? = null
    private var preview: Preview? = null
    private var camera: Camera? = null

    // 분석 실행자 (백그라운드 스레드)
    private val analysisExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    // 현재 카메라 방향
    var lensFacing: Int = CameraSelector.LENS_FACING_FRONT
        private set

    // 카메라 상태
    var isRunning: Boolean = false
        private set

    // 이미지 크기 (분석용)
    var imageWidth: Int = ANALYSIS_SIZE.width
        private set
    var imageHeight: Int = ANALYSIS_SIZE.height
        private set

    /**
     * 전면 카메라 사용 중인지 확인
     */
    val isFrontCamera: Boolean
        get() = lensFacing == CameraSelector.LENS_FACING_FRONT

    /**
     * 카메라 시작
     *
     * @param previewView 프리뷰 표시 뷰
     * @param onFrameAnalyzed 프레임 분석 콜백
     */
    fun startCamera(
        previewView: PreviewView,
        onFrameAnalyzed: (ImageProxy) -> Unit
    ) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)

        cameraProviderFuture.addListener({
            try {
                cameraProvider = cameraProviderFuture.get()
                bindCameraUseCases(previewView, onFrameAnalyzed)
                isRunning = true
                Log.d(TAG, "Camera started successfully")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to start camera", e)
            }
        }, ContextCompat.getMainExecutor(context))
    }

    /**
     * CameraX Use Cases 바인딩
     */
    private fun bindCameraUseCases(
        previewView: PreviewView,
        onFrameAnalyzed: (ImageProxy) -> Unit
    ) {
        val cameraProvider = cameraProvider ?: run {
            Log.e(TAG, "CameraProvider is null")
            return
        }

        // 카메라 선택
        val cameraSelector = CameraSelector.Builder()
            .requireLensFacing(lensFacing)
            .build()

        // 프리뷰 설정
        preview = Preview.Builder()
            .setTargetResolution(PREVIEW_SIZE)
            .build()
            .also { preview ->
                preview.setSurfaceProvider(previewView.surfaceProvider)
            }

        // 이미지 분석 설정
        imageAnalysis = ImageAnalysis.Builder()
            .setTargetResolution(ANALYSIS_SIZE)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
            .build()
            .also { analysis ->
                analysis.setAnalyzer(analysisExecutor) { imageProxy ->
                    // 이미지 크기 업데이트
                    imageWidth = imageProxy.width
                    imageHeight = imageProxy.height

                    // 콜백 호출
                    onFrameAnalyzed(imageProxy)
                }
            }

        try {
            // 기존 바인딩 해제
            cameraProvider.unbindAll()

            // 새 바인딩
            camera = cameraProvider.bindToLifecycle(
                lifecycleOwner,
                cameraSelector,
                preview,
                imageAnalysis
            )

            Log.d(TAG, "Camera use cases bound: ${if (isFrontCamera) "Front" else "Back"}")

        } catch (e: Exception) {
            Log.e(TAG, "Camera binding failed", e)
        }
    }

    /**
     * 카메라 전환 (전면 ↔ 후면)
     *
     * @param previewView 프리뷰 표시 뷰
     * @param onFrameAnalyzed 프레임 분석 콜백
     */
    fun switchCamera(
        previewView: PreviewView,
        onFrameAnalyzed: (ImageProxy) -> Unit
    ) {
        lensFacing = if (lensFacing == CameraSelector.LENS_FACING_FRONT) {
            CameraSelector.LENS_FACING_BACK
        } else {
            CameraSelector.LENS_FACING_FRONT
        }

        Log.d(TAG, "Switching to ${if (isFrontCamera) "front" else "back"} camera")
        bindCameraUseCases(previewView, onFrameAnalyzed)
    }

    /**
     * 카메라 중지 및 리소스 해제
     */
    fun release() {
        try {
            cameraProvider?.unbindAll()
            analysisExecutor.shutdown()
            isRunning = false
            Log.d(TAG, "Camera released")
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing camera", e)
        }
    }
}
