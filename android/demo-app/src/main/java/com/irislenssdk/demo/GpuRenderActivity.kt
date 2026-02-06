/**
 * IrisLensSDK Android - GPU Render Activity
 *
 * GPU 기반 렌더링 테스트용 Activity
 * - GLSurfaceView + CameraX 연동
 * - GPU Beauty Filter (OpenGL ES 3.1)
 * - MediaPipe 추론은 CPU (하이브리드 아키텍처)
 *
 * @version 1.0.0
 */
package com.irislenssdk.demo

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.irislenssdk.BeautyFilterConfigV2
import com.irislenssdk.IrisLensSDK
import com.irislenssdk.IrisResult
import com.irislenssdk.demo.camera.gpu.CameraGLView
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * GPU 렌더링 테스트 Activity
 *
 * 하이브리드 아키텍처:
 * - MediaPipe (CPU): 홍채 추적 (FrameAnalyzer)
 * - GPU: 뷰티 필터 렌더링 (CameraGLView)
 */
class GpuRenderActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "GpuRenderActivity"
        private const val REQUEST_CAMERA_PERMISSION = 1001
    }

    // UI
    private lateinit var cameraGLView: CameraGLView
    private lateinit var tvFps: TextView
    private lateinit var tvGpuStatus: TextView
    private lateinit var btnToggleBeauty: Button
    private lateinit var seekSmoothing: SeekBar
    private lateinit var seekBrightness: SeekBar
    private lateinit var seekWhitening: SeekBar
    private lateinit var seekColorBalance: SeekBar
    private lateinit var seekSoftFocus: SeekBar

    // 카메라
    private var cameraProvider: ProcessCameraProvider? = null
    private var lensFacing = CameraSelector.LENS_FACING_FRONT
    private val analysisExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    // 뷰티 설정
    private var beautyConfig = BeautyFilterConfigV2()
    private var beautyEnabled = true

    // 홍채 검출
    private val irisResult = IrisResult()

    // NV21 버퍼 (재사용)
    private var nv21Buffer: ByteArray? = null

    // 카메라 회전 (한 번만 설정)
    private var lastRotation: Int = -1

    // FPS 계산
    private var frameCount = 0
    private var lastFpsTime = System.currentTimeMillis()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_gpu_render)

        initViews()
        initSDK()

        if (hasCameraPermission()) {
            startCamera()
        } else {
            requestCameraPermission()
        }
    }

    private fun initViews() {
        cameraGLView = findViewById(R.id.cameraGLView)
        tvFps = findViewById(R.id.tvFps)
        tvGpuStatus = findViewById(R.id.tvGpuStatus)
        btnToggleBeauty = findViewById(R.id.btnToggleBeauty)
        seekSmoothing = findViewById(R.id.seekSmoothing)
        seekBrightness = findViewById(R.id.seekBrightness)
        seekWhitening = findViewById(R.id.seekWhitening)
        seekColorBalance = findViewById(R.id.seekColorBalance)
        seekSoftFocus = findViewById(R.id.seekSoftFocus)

        // GPU 초기화 콜백 설정
        cameraGLView.onGpuInitialized = { success ->
            runOnUiThread {
                tvGpuStatus.text = "GPU: Available (init: $success)"
                Log.d(TAG, "GPU initialized: $success")
            }
        }

        // 뷰티 토글
        btnToggleBeauty.setOnClickListener {
            beautyEnabled = !beautyEnabled
            beautyConfig.enabled = beautyEnabled
            cameraGLView.setBeautyEnabled(beautyEnabled)
            btnToggleBeauty.text = if (beautyEnabled) "Beauty: ON" else "Beauty: OFF"
        }

        // Smoothing 슬라이더
        seekSmoothing.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                beautyConfig.smoothing = progress / 100f
                cameraGLView.setBeautyConfig(beautyConfig)
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        // Brightness 슬라이더
        seekBrightness.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                // 0-100 → 0.5-1.5
                beautyConfig.brightness = 0.5f + progress / 100f
                cameraGLView.setBeautyConfig(beautyConfig)
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        // Whitening 슬라이더
        seekWhitening.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                // 0-100 → 0.0-1.0
                beautyConfig.whitening = progress / 100f
                cameraGLView.setBeautyConfig(beautyConfig)
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        // Color Balance 슬라이더
        seekColorBalance.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                // 0-100 → -1.0-1.0 (50 = 0.0 중립)
                beautyConfig.colorBalance = (progress - 50) / 50f
                cameraGLView.setBeautyConfig(beautyConfig)
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        // Soft Focus 슬라이더
        seekSoftFocus.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                // 0-100 → 0.0-1.0
                beautyConfig.softFocus = progress / 100f
                cameraGLView.setBeautyConfig(beautyConfig)
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        // 초기값
        seekSmoothing.progress = 50
        seekBrightness.progress = 50
        seekWhitening.progress = 0
        seekColorBalance.progress = 50  // 중립 (0.0)
        seekSoftFocus.progress = 30     // 기본값 0.3f

        // 초기 버튼 상태 표시 (ON = 뷰티 활성화)
        btnToggleBeauty.text = if (beautyEnabled) "Beauty: ON" else "Beauty: OFF"
    }

    private fun initSDK() {
        // SDK 초기화 (이미 초기화된 경우 무시)
        val result = IrisLensSDK.init(this)
        if (result != IrisLensSDK.OK && result != IrisLensSDK.ALREADY_INITIALIZED) {
            Log.e(TAG, "SDK init failed: ${IrisLensSDK.errorToString(result)}")
            Toast.makeText(this, "SDK init failed", Toast.LENGTH_SHORT).show()
            return
        }

        // GPU 가용성 확인
        val gpuAvailable = IrisLensSDK.isGpuAvailable()
        val gpuInitialized = IrisLensSDK.isGpuBeautyInitialized()
        tvGpuStatus.text = "GPU: ${if (gpuAvailable) "Available" else "N/A"} (init: $gpuInitialized)"

        // 기본 뷰티 설정
        beautyConfig = IrisLensSDK.getDefaultBeautyConfigV2()
        beautyConfig.enabled = true
        beautyConfig.smoothing = 0.5f
        beautyConfig.brightness = 1.0f
        beautyConfig.whitening = 0.0f
        beautyConfig.colorBalance = 0.0f
        beautyConfig.softFocus = 0.3f

        // GLView에 초기 뷰티 설정 전달
        cameraGLView.setBeautyConfig(beautyConfig)
        cameraGLView.setBeautyEnabled(true)
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCameraUseCases() {
        val cameraProvider = cameraProvider ?: return

        // 카메라 선택 (전면)
        val cameraSelector = CameraSelector.Builder()
            .requireLensFacing(lensFacing)
            .build()

        // Preview → GLSurfaceView
        val preview = Preview.Builder()
            .build()
            .apply {
                setSurfaceProvider(cameraGLView.getSurfaceProvider())
            }

        // ImageAnalysis (MediaPipe 추론용)
        val imageAnalysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
            .build()
            .apply {
                setAnalyzer(analysisExecutor) { imageProxy ->
                    processFrame(imageProxy)
                }
            }

        try {
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalysis
            )

            // 미러링 설정 (전면 카메라)
            cameraGLView.setMirror(lensFacing == CameraSelector.LENS_FACING_FRONT)

            Log.d(TAG, "Camera bound successfully")

        } catch (e: Exception) {
            Log.e(TAG, "Camera binding failed", e)
        }
    }

    /**
     * 프레임 처리 (MediaPipe 추론 - CPU)
     */
    private fun processFrame(imageProxy: androidx.camera.core.ImageProxy) {
        try {
            // 카메라 회전 정보 전달 (한 번만)
            val rotation = imageProxy.imageInfo.rotationDegrees
            if (rotation != lastRotation) {
                lastRotation = rotation
                cameraGLView.setFrameRotation(rotation)
                Log.d(TAG, "Camera rotation: $rotation")
            }

            // YUV → NV21 변환
            val nv21 = yuvToNv21(imageProxy)

            // 홍채 검출 (CPU)
            val detectResult = IrisLensSDK.detectWithRotation(
                nv21,
                imageProxy.width,
                imageProxy.height,
                IrisLensSDK.FORMAT_NV21,
                imageProxy.imageInfo.rotationDegrees,
                irisResult
            )

            // GPU 렌더러에 검출 결과 전달
            if (detectResult == IrisLensSDK.OK && irisResult.detected) {
                cameraGLView.setIrisResult(irisResult)
            }

            // FPS 계산
            updateFps()

        } finally {
            imageProxy.close()
        }
    }

    /**
     * YUV_420_888 → NV21 변환
     */
    private fun yuvToNv21(imageProxy: androidx.camera.core.ImageProxy): ByteArray {
        val yPlane = imageProxy.planes[0]
        val uPlane = imageProxy.planes[1]
        val vPlane = imageProxy.planes[2]

        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21Size = imageProxy.width * imageProxy.height * 3 / 2

        // 버퍼 재사용
        if (nv21Buffer == null || nv21Buffer!!.size != nv21Size) {
            nv21Buffer = ByteArray(nv21Size)
        }
        val nv21 = nv21Buffer!!

        // Y plane 복사
        yBuffer.get(nv21, 0, ySize)

        // UV interleaved (NV21: VUVU...)
        val uvOffset = imageProxy.width * imageProxy.height
        val pixelStride = uPlane.pixelStride

        if (pixelStride == 2) {
            // 이미 interleaved (대부분의 기기)
            vBuffer.get(nv21, uvOffset, vSize.coerceAtMost(nv21Size - uvOffset))
        } else {
            // Planar → interleaved 변환
            var uvIndex = uvOffset
            for (i in 0 until uSize) {
                if (uvIndex < nv21Size) nv21[uvIndex++] = vBuffer.get(i)
                if (uvIndex < nv21Size) nv21[uvIndex++] = uBuffer.get(i)
            }
        }

        return nv21
    }

    private fun updateFps() {
        frameCount++
        val currentTime = System.currentTimeMillis()
        if (currentTime - lastFpsTime >= 1000) {
            val fps = frameCount
            frameCount = 0
            lastFpsTime = currentTime

            runOnUiThread {
                tvFps.text = "FPS: $fps"
            }
        }
    }

    //=========================================================================
    // 권한 처리
    //=========================================================================

    private fun hasCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this, Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun requestCameraPermission() {
        ActivityCompat.requestPermissions(
            this,
            arrayOf(Manifest.permission.CAMERA),
            REQUEST_CAMERA_PERMISSION
        )
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startCamera()
            } else {
                Toast.makeText(this, "Camera permission required", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    //=========================================================================
    // 라이프사이클
    //=========================================================================

    override fun onResume() {
        super.onResume()
        cameraGLView.onResume()
    }

    override fun onPause() {
        super.onPause()
        cameraGLView.onPause()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraGLView.release()
        analysisExecutor.shutdown()
        IrisLensSDK.releaseGpuBeauty()
    }
}
