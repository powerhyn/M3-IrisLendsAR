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
import androidx.camera.core.AspectRatio
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

        // Preview와 ImageAnalysis가 같은 비율을 사용해야 FOV(시야각)가 일치함
        // 16:9 비율 사용 (대부분의 기기에서 지원)
        // NOTE: setTargetResolution()은 "요청"일 뿐, 기기가 지원하지 않으면 다른 해상도 반환
        // setTargetAspectRatio()를 사용하면 비율만 고정하고 해상도는 기기가 최적으로 선택
        private const val TARGET_ASPECT_RATIO = AspectRatio.RATIO_16_9
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

    // 이미지 크기 (분석용) - 실제 값은 카메라 시작 시 imageProxy에서 동적으로 설정됨
    var imageWidth: Int = 640
        private set
    var imageHeight: Int = 360
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

        // 프리뷰 설정 (비율만 고정, 해상도는 기기가 최적으로 선택)
        preview = Preview.Builder()
            .setTargetAspectRatio(TARGET_ASPECT_RATIO)
            .build()
            .also { preview ->
                preview.setSurfaceProvider(previewView.surfaceProvider)
            }

        // 이미지 분석 설정 (Preview와 동일한 비율로 FOV 일치)
        imageAnalysis = ImageAnalysis.Builder()
            .setTargetAspectRatio(TARGET_ASPECT_RATIO)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
            .build()
            .also { analysis ->
                analysis.setAnalyzer(analysisExecutor) { imageProxy ->
                    // 이미지 크기 업데이트 (디스플레이 방향 기준)
                    // 회전이 90° 또는 270°인 경우 가로/세로 교환
                    val rotationDegrees = imageProxy.imageInfo.rotationDegrees
                    if (rotationDegrees == 90 || rotationDegrees == 270) {
                        imageWidth = imageProxy.height
                        imageHeight = imageProxy.width
                    } else {
                        imageWidth = imageProxy.width
                        imageHeight = imageProxy.height
                    }

                    // 콜백 호출
                    onFrameAnalyzed(imageProxy)
                }
            }

        try {
            // 기존 바인딩 해제
            cameraProvider.unbindAll()

            // ISS-001 수정: ViewPort 사용하지 않음
            // ViewPort는 레이아웃 완료 전에 생성되면 잘못된 aspect ratio를 가질 수 있음
            // Preview와 ImageAnalysis가 같은 TARGET_ASPECT_RATIO(16:9)를 사용하고,
            // OverlayView가 동일한 fillCenter 스케일링을 적용하므로 좌표가 일치함
            camera = cameraProvider.bindToLifecycle(
                lifecycleOwner,
                cameraSelector,
                preview,
                imageAnalysis
            )

            Log.d(TAG, "Camera use cases bound (without ViewPort): ${if (isFrontCamera) "Front" else "Back"}")

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
