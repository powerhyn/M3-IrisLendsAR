/**
 * MediaPipe SDK 직접 벤치마크 액티비티
 *
 * MediaPipe Tasks Vision의 FaceLandmarker를 직접 사용하여
 * IrisLensSDK(TFLite 기반)와 성능 비교를 위한 테스트
 */
package com.irislenssdk.demo

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import com.irislenssdk.demo.benchmark.BenchmarkCallback
import com.irislenssdk.demo.benchmark.BenchmarkManager
import com.irislenssdk.demo.benchmark.BenchmarkResult
import com.irislenssdk.demo.benchmark.MemoryInfo
import com.irislenssdk.demo.benchmark.PerformanceStats
import com.irislenssdk.demo.databinding.ActivityMediapipeBenchmarkBinding
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MediaPipeBenchmarkActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MediaPipeBenchmark"
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)

        // MediaPipe Face Landmarker 모델
        private const val FACE_LANDMARKER_MODEL = "face_landmarker.task"
    }

    private lateinit var binding: ActivityMediapipeBenchmarkBinding

    // MediaPipe FaceLandmarker
    private var faceLandmarker: FaceLandmarker? = null
    private var useGpu = false

    // Camera
    private var cameraProvider: ProcessCameraProvider? = null
    private lateinit var cameraExecutor: ExecutorService
    private var imageWidth = 0
    private var imageHeight = 0

    // Benchmark
    private lateinit var benchmarkManager: BenchmarkManager
    private var isBenchmarkRunning = false
    private val benchmarkDurationMs = 60_000L

    // FPS calculation
    private var frameCount = 0
    private var lastFpsTime = 0L
    private var currentFps = 0f

    // UI Handler
    private val mainHandler = Handler(Looper.getMainLooper())

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        if (permissions.all { it.value }) {
            initializeMediaPipe()
            startCamera()
        } else {
            Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMediapipeBenchmarkBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()
        benchmarkManager = BenchmarkManager(this)

        setupUI()

        if (allPermissionsGranted()) {
            initializeMediaPipe()
            startCamera()
        } else {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        benchmarkManager.release()
        cameraExecutor.shutdown()
        faceLandmarker?.close()
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
    }

    private fun setupUI() {
        binding.statusText.text = "Initializing MediaPipe..."
        binding.fpsText.text = "FPS: --"

        // GPU 토글 버튼
        binding.gpuToggleButton.setOnClickListener {
            useGpu = !useGpu
            binding.gpuToggleButton.text = if (useGpu) "GPU: ON" else "GPU: OFF"

            // MediaPipe 재초기화
            faceLandmarker?.close()
            initializeMediaPipe()

            Toast.makeText(this, "MediaPipe ${if (useGpu) "GPU" else "CPU"} mode", Toast.LENGTH_SHORT).show()
        }
        binding.gpuToggleButton.text = if (useGpu) "GPU: ON" else "GPU: OFF"

        // 벤치마크 버튼
        binding.benchmarkButton.setOnClickListener {
            toggleBenchmark()
        }

        // 뒤로가기 버튼
        binding.backButton.setOnClickListener {
            finish()
        }
    }

    private fun initializeMediaPipe() {
        try {
            Log.d(TAG, "Initializing MediaPipe FaceLandmarker (GPU: $useGpu)...")

            val baseOptionsBuilder = BaseOptions.builder()
                .setModelAssetPath(FACE_LANDMARKER_MODEL)

            if (useGpu) {
                baseOptionsBuilder.setDelegate(Delegate.GPU)
            } else {
                baseOptionsBuilder.setDelegate(Delegate.CPU)
            }

            val options = FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(baseOptionsBuilder.build())
                .setRunningMode(RunningMode.IMAGE)  // 동기 모드로 정확한 시간 측정
                .setNumFaces(1)
                .setMinFaceDetectionConfidence(0.5f)
                .setMinFacePresenceConfidence(0.5f)
                .setMinTrackingConfidence(0.5f)
                .setOutputFaceBlendshapes(false)
                .setOutputFacialTransformationMatrixes(false)
                .build()

            faceLandmarker = FaceLandmarker.createFromOptions(this, options)

            val mode = if (useGpu) "GPU" else "CPU"
            updateStatus("MediaPipe Ready ($mode)")
            Log.i(TAG, "MediaPipe FaceLandmarker initialized ($mode)")

        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize MediaPipe", e)
            updateStatus("MediaPipe Error: ${e.message}")
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(binding.cameraPreview.surfaceProvider)
                }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        processFrame(imageProxy)
                    }
                }

            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

            try {
                cameraProvider?.unbindAll()
                cameraProvider?.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer
                )
                Log.d(TAG, "Camera started")
            } catch (e: Exception) {
                Log.e(TAG, "Camera bind failed", e)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun processFrame(imageProxy: ImageProxy) {
        val landmarker = faceLandmarker
        if (landmarker == null) {
            imageProxy.close()
            return
        }

        imageWidth = imageProxy.width
        imageHeight = imageProxy.height

        val startTime = SystemClock.elapsedRealtime()

        try {
            // ImageProxy를 Bitmap으로 변환
            val bitmap = imageProxyToBitmap(imageProxy)

            // MediaPipe 이미지로 변환
            val mpImage = BitmapImageBuilder(bitmap).build()

            // Face Landmark 검출
            val result = landmarker.detect(mpImage)

            val processingTime = SystemClock.elapsedRealtime() - startTime

            // FPS 계산
            frameCount++
            val now = SystemClock.elapsedRealtime()
            if (now - lastFpsTime >= 1000) {
                currentFps = frameCount * 1000f / (now - lastFpsTime)
                frameCount = 0
                lastFpsTime = now
            }

            // UI 업데이트
            mainHandler.post {
                onFrameProcessed(result, processingTime)
            }

            // 벤치마크 데이터 수집
            if (isBenchmarkRunning) {
                benchmarkManager.onFrameProcessed(processingTime)
            }

        } catch (e: Exception) {
            Log.e(TAG, "Frame processing error", e)
        } finally {
            imageProxy.close()
        }
    }

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
        val yBuffer = imageProxy.planes[0].buffer
        val uBuffer = imageProxy.planes[1].buffer
        val vBuffer = imageProxy.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = android.graphics.YuvImage(
            nv21, android.graphics.ImageFormat.NV21,
            imageProxy.width, imageProxy.height, null
        )

        val out = java.io.ByteArrayOutputStream()
        yuvImage.compressToJpeg(
            android.graphics.Rect(0, 0, imageProxy.width, imageProxy.height),
            100, out
        )

        val imageBytes = out.toByteArray()
        val bitmap = android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)

        // 회전 적용
        val matrix = Matrix()
        matrix.postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())

        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun onFrameProcessed(result: FaceLandmarkerResult, processingTimeMs: Long) {
        // FPS 텍스트
        val fpsText = if (isBenchmarkRunning) {
            val stats = benchmarkManager.performanceTracker.getStats()
            "FPS: %.1f | Lat: %.0fms".format(stats.fps, stats.avgLatencyMs)
        } else {
            "FPS: %.1f".format(currentFps)
        }
        binding.fpsText.text = fpsText

        // 상태 텍스트
        val facesDetected = result.faceLandmarks().size
        val landmarkCount = if (facesDetected > 0) result.faceLandmarks()[0].size else 0

        val status = if (isBenchmarkRunning) {
            val mem = benchmarkManager.memoryMonitor.getMemoryInfo()
            if (facesDetected > 0) {
                "Face: $landmarkCount pts | ${processingTimeMs}ms | PSS: ${mem.totalPssMB}MB"
            } else {
                "No Face | ${processingTimeMs}ms | PSS: ${mem.totalPssMB}MB"
            }
        } else {
            if (facesDetected > 0) {
                "Face detected: $landmarkCount landmarks (${processingTimeMs}ms)"
            } else {
                "No face detected (${processingTimeMs}ms)"
            }
        }
        binding.statusText.text = status
    }

    private fun toggleBenchmark() {
        if (isBenchmarkRunning) {
            benchmarkManager.stop()
        } else {
            Log.i(TAG, "Starting MediaPipe benchmark for ${benchmarkDurationMs}ms")

            // MediaPipe용 벤치마크 설정
            val modeTag = if (useGpu) "mediapipe_gpu" else "mediapipe_cpu"
            benchmarkManager.setCustomModeTag(modeTag)
            benchmarkManager.setGpuEnabled(useGpu)

            benchmarkManager.start(benchmarkDurationMs, object : BenchmarkCallback {
                override fun onBenchmarkStarted() {
                    runOnUiThread {
                        isBenchmarkRunning = true
                        binding.benchmarkButton.alpha = 0.5f
                        Toast.makeText(
                            this@MediaPipeBenchmarkActivity,
                            "MediaPipe Benchmark started (${benchmarkDurationMs / 1000}s)",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                }

                override fun onBenchmarkProgress(
                    elapsedMs: Long,
                    stats: PerformanceStats,
                    memory: MemoryInfo
                ) {
                    Log.d(TAG, "Benchmark: ${elapsedMs/1000}s | " +
                            "FPS: %.1f | Lat: %.1fms | PSS: ${memory.totalPssMB}MB"
                                .format(stats.fps, stats.avgLatencyMs))
                }

                override fun onBenchmarkCompleted(result: BenchmarkResult) {
                    runOnUiThread {
                        isBenchmarkRunning = false
                        binding.benchmarkButton.alpha = 1.0f

                        Log.i(TAG, "=== MediaPipe Benchmark Results ===")
                        Log.i(TAG, "Mode: ${if (useGpu) "GPU" else "CPU"}")
                        Log.i(TAG, "Duration: ${result.testDurationMs}ms")
                        Log.i(TAG, "FPS: ${result.performance.fps}")
                        Log.i(TAG, "Avg Latency: ${result.performance.avgLatencyMs}ms")
                        Log.i(TAG, "P95 Latency: ${result.performance.p95LatencyMs}ms")
                        Log.i(TAG, "Drop Rate: ${result.performance.dropRate * 100}%")
                        Log.i(TAG, "Memory PSS: ${result.memory.totalPssMB}MB")

                        // CSV 내보내기
                        val csvFile = benchmarkManager.exportToCsv()
                        csvFile?.let {
                            Log.i(TAG, "CSV exported to: ${it.absolutePath}")
                        }

                        Toast.makeText(
                            this@MediaPipeBenchmarkActivity,
                            "MediaPipe ${if (useGpu) "GPU" else "CPU"}: FPS=%.1f, Lat=%.0fms"
                                .format(result.performance.fps, result.performance.avgLatencyMs),
                            Toast.LENGTH_LONG
                        ).show()
                    }
                }

                override fun onBenchmarkError(error: String) {
                    runOnUiThread {
                        isBenchmarkRunning = false
                        binding.benchmarkButton.alpha = 1.0f
                        Toast.makeText(this@MediaPipeBenchmarkActivity, "Error: $error", Toast.LENGTH_SHORT).show()
                    }
                }
            })
        }
    }

    private fun updateStatus(status: String) {
        runOnUiThread {
            binding.statusText.text = status
        }
    }
}
