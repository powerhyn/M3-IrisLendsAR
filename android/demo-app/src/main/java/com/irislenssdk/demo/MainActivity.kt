/**
 * IrisLensSDK Android - Demo MainActivity
 *
 * SDK 기능 데모를 위한 메인 액티비티
 * - CameraX 카메라 프리뷰
 * - 실시간 홍채 검출
 * - 렌즈 오버레이 표시
 * - 렌즈 선택 및 설정 조절
 *
 * @version 1.1.0
 */
package com.irislenssdk.demo

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import com.irislenssdk.BeautyFilterConfig
import com.irislenssdk.BeautyFilterConfigV2
import com.irislenssdk.IrisLensSDK
import com.irislenssdk.IrisResult
import com.irislenssdk.LensConfig
import com.irislenssdk.demo.benchmark.BenchmarkCallback
import com.irislenssdk.demo.benchmark.BenchmarkManager
import com.irislenssdk.demo.benchmark.BenchmarkResult
import com.irislenssdk.demo.benchmark.MemoryInfo
import com.irislenssdk.demo.benchmark.PerformanceStats
import com.irislenssdk.demo.camera.AnalysisResult
import com.irislenssdk.demo.camera.CameraManager
import com.irislenssdk.demo.camera.FrameAnalyzer
import com.irislenssdk.demo.databinding.ActivityMainBinding
import com.irislenssdk.demo.lens.LensAdapter
import com.irislenssdk.demo.lens.LensData
import com.irislenssdk.demo.lens.LensManager
import com.irislenssdk.demo.lens.NoLens
import kotlinx.coroutines.launch

/**
 * 데모 앱 메인 액티비티
 *
 * IrisLensSDK를 사용한 실시간 AR 렌즈 피팅 데모
 */
class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "IrisLensSDK-Demo"
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }

    // View Binding
    private lateinit var binding: ActivityMainBinding

    // SDK 초기화 상태
    private var isSDKInitialized: Boolean = false

    // 렌즈 설정
    private val lensConfig = LensConfig()

    // 뷰티 필터 설정 (V1 - 하위 호환)
    private val beautyConfig = BeautyFilterConfig()

    // 뷰티 필터 V2 설정 (확장 기능)
    private val beautyConfigV2 = BeautyFilterConfigV2()

    // 렌즈 관리자
    private lateinit var lensManager: LensManager

    // 렌즈 어댑터
    private lateinit var lensAdapter: LensAdapter

    // CameraX 관리자
    private var cameraManager: CameraManager? = null

    // 프레임 분석기
    private var frameAnalyzer: FrameAnalyzer? = null

    // UI 업데이트 핸들러
    private val mainHandler = Handler(Looper.getMainLooper())

    // 마지막 검출 결과
    private var lastIrisResult: IrisResult? = null

    // 벤치마크 관리자
    private lateinit var benchmarkManager: BenchmarkManager
    private var isBenchmarkRunning = false

    // 벤치마크 설정
    private val benchmarkDurationMs = 60_000L  // 1분 벤치마크

    // 권한 요청 런처
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val allGranted = permissions.all { it.value }
        if (allGranted) {
            Log.d(TAG, "All permissions granted")
            initializeSDK()
            startCamera()
        } else {
            Log.e(TAG, "Permissions denied")
            Toast.makeText(
                this,
                getString(R.string.permission_camera_rationale),
                Toast.LENGTH_LONG
            ).show()
            finish()
        }
    }

    // ==========================================================================
    // Lifecycle
    // ==========================================================================

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // View Binding 초기화
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // 벤치마크 관리자 초기화
        benchmarkManager = BenchmarkManager(this)

        // 렌즈 관리자 초기화
        lensManager = LensManager(this)

        // UI 설정
        setupUI()
        setupLensSelector()
        setupSliders()
        setupButtons()

        // 권한 확인 및 요청
        if (allPermissionsGranted()) {
            initializeSDK()
            startCamera()
        } else {
            requestPermissions()
        }
    }

    override fun onResume() {
        super.onResume()
        // 카메라 재시작 (필요시)
        if (isSDKInitialized && cameraManager?.isRunning == false) {
            startCamera()
        }
    }

    override fun onPause() {
        super.onPause()
        // 카메라 일시정지
    }

    override fun onDestroy() {
        super.onDestroy()
        // 리소스 해제
        benchmarkManager.release()
        lensManager.release()
        releaseCamera()
        releaseSDK()
    }

    // ==========================================================================
    // Permission Handling
    // ==========================================================================

    /**
     * 모든 필수 권한이 부여되었는지 확인
     */
    private fun allPermissionsGranted(): Boolean {
        return REQUIRED_PERMISSIONS.all { permission ->
            ContextCompat.checkSelfPermission(this, permission) ==
                    PackageManager.PERMISSION_GRANTED
        }
    }

    /**
     * 권한 요청
     */
    private fun requestPermissions() {
        requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
    }

    // ==========================================================================
    // SDK Initialization
    // ==========================================================================

    /**
     * IrisLensSDK 초기화
     */
    private fun initializeSDK() {
        try {
            Log.d(TAG, "Initializing IrisLensSDK...")
            Log.d(TAG, "Library loaded: ${IrisLensSDK.isLibraryLoaded()}")

            // ===== 벤치마크 테스트: 직접 호출 모드 (InferenceThread 우회) =====
            // false: InferenceThread 없이 직접 MediaPipeDetector 호출 (CPU only)
            // true: InferenceThread 사용 (기본값, GPU 지원)
            val useInferenceThread = true  // 벤치마크 테스트용: true로 설정 (Thread 모드)
            IrisLensSDK.setUseInferenceThread(useInferenceThread)
            Log.i(TAG, "InferenceThread mode: ${if (useInferenceThread) "enabled" else "disabled (direct call)"}")

            // GPU 가속 활성화 요청 (init 전에 호출!)
            // InferenceThread를 통해 전용 스레드에서 GPU delegate 초기화/실행
            // NOTE: 직접 호출 모드에서는 GPU 지원 안됨 (스레드 제약)
            val enableGpu = false  // GPU delegate 호환성 문제로 비활성화
            val gpuAvailable = IrisLensSDK.isGpuAvailable() && useInferenceThread && enableGpu
            Log.i(TAG, "GPU available: $gpuAvailable (enableGpu=$enableGpu)")
            if (gpuAvailable) {
                IrisLensSDK.setGpuEnabled(true)
                Log.i(TAG, "GPU acceleration requested")
            }

            // SDK 초기화 (정적 메서드 사용)
            val error = IrisLensSDK.init(this)

            if (error == IrisLensSDK.OK) {
                isSDKInitialized = true
                val version = IrisLensSDK.getVersion()
                val isReady = IrisLensSDK.isReady()
                val gpuActive = IrisLensSDK.isUsingGpu()
                val usingInferenceThread = IrisLensSDK.isUsingInferenceThread()
                Log.i(TAG, "SDK Version: $version")
                Log.i(TAG, "SDK Ready: $isReady")
                Log.i(TAG, "GPU Active: $gpuActive")
                Log.i(TAG, "InferenceThread Active: $usingInferenceThread")
                Log.d(TAG, "IrisLensSDK initialized successfully")

                val gpuStatus = if (gpuActive) "GPU" else "CPU"
                val threadMode = if (usingInferenceThread) "Thread" else "Direct"
                if (isReady) {
                    updateStatus("SDK Ready ($gpuStatus/$threadMode)\n$version")
                } else {
                    updateStatus("SDK Loaded ($gpuStatus/$threadMode)\n$version")
                }

                // 뷰티 필터 기본 설정 적용
                initializeBeautyFilter()
            } else {
                val errorStr = IrisLensSDK.errorToString(error)
                val lastError = IrisLensSDK.getLastError()
                Log.e(TAG, "SDK init error: $errorStr")
                Log.e(TAG, "Last error: $lastError")
                isSDKInitialized = false

                // SDK 초기화 실패해도 카메라는 동작하도록 함
                updateStatus("SDK Error: $errorStr\nCamera preview only")
            }

        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize IrisLensSDK", e)
            isSDKInitialized = false
            updateStatus("SDK Exception: ${e.message}\nCamera preview only")
        }
    }

    /**
     * SDK 리소스 해제
     */
    private fun releaseSDK() {
        try {
            if (isSDKInitialized) {
                IrisLensSDK.destroy()
                isSDKInitialized = false
                Log.d(TAG, "IrisLensSDK released")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing SDK", e)
        }
    }

    /**
     * 뷰티 필터 초기화
     *
     * 기본 설정으로 뷰티 필터를 활성화합니다.
     * 현재 아키텍처에서는 OpenCV CPU 기반으로 동작합니다.
     * 실제 프레임 렌더링은 Stage 2 (OpenGL)에서 구현 예정입니다.
     */
    private fun initializeBeautyFilter() {
        try {
            // V1 설정 (하위 호환)
            beautyConfig.enabled = false
            beautyConfig.intensity = 0.5f
            beautyConfig.smoothing = 0.5f
            beautyConfig.brightness = 1.05f
            beautyConfig.softFocus = 0.3f

            // V2 설정 (확장 기능)
            beautyConfigV2.enabled = false  // 기본 비활성화
            beautyConfigV2.intensity = 0.5f
            beautyConfigV2.smoothing = 0.5f
            beautyConfigV2.brightness = 1.05f
            beautyConfigV2.softFocus = 0.3f
            beautyConfigV2.whitening = 0.0f
            beautyConfigV2.colorBalance = 0.0f
            beautyConfigV2.skinQuality = 0.0f   // 잡티 보정 (Freq Sep)
            // Face Warp 기본값
            beautyConfigV2.slimFace = 0.0f
            beautyConfigV2.enlargeEyes = 0.0f
            beautyConfigV2.thinChin = 0.0f
            // 처리 옵션
            beautyConfigV2.roiOnly = true
            beautyConfigV2.protectEyes = true
            beautyConfigV2.protectLips = true

            // V1 API로 초기화 (하위 호환)
            val error = IrisLensSDK.setBeautyFilter(beautyConfig)
            if (error == IrisLensSDK.OK) {
                Log.i(TAG, "Beauty filter V2 initialized: enabled=${beautyConfigV2.enabled}")
            } else {
                Log.w(TAG, "Beauty filter init failed: ${IrisLensSDK.errorToString(error)}")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Beauty filter init exception", e)
        }
    }

    // ==========================================================================
    // Camera Setup
    // ==========================================================================

    /**
     * CameraX 카메라 시작
     */
    private fun startCamera() {
        Log.d(TAG, "Starting camera...")

        // CameraManager 생성
        cameraManager = CameraManager(this, this)

        // FrameAnalyzer 생성 (결과 콜백은 백그라운드 스레드에서 호출됨)
        frameAnalyzer = FrameAnalyzer { result ->
            // UI 업데이트는 메인 스레드에서
            mainHandler.post {
                onFrameAnalyzed(result)
            }
        }.apply {
            // 뷰티 필터 초기 상태 동기화
            beautyFilterEnabled = this@MainActivity.beautyConfigV2.enabled
            beautyConfigV2 = this@MainActivity.beautyConfigV2
        }

        // OverlayView 초기 상태 동기화
        binding.overlayView.showFilteredFrame = beautyConfigV2.enabled

        // 카메라 시작
        cameraManager?.startCamera(binding.cameraPreview) { imageProxy ->
            frameAnalyzer?.analyze(imageProxy)
        }

        updateStatus("Camera starting...")
    }

    /**
     * 카메라 리소스 해제
     */
    private fun releaseCamera() {
        frameAnalyzer?.release()
        frameAnalyzer = null

        cameraManager?.release()
        cameraManager = null
    }

    /**
     * 카메라 전환 (전면 ↔ 후면)
     */
    private fun switchCamera() {
        val cm = cameraManager ?: return
        val fa = frameAnalyzer ?: return

        Log.d(TAG, "Switching camera...")

        cm.switchCamera(binding.cameraPreview) { imageProxy ->
            fa.analyze(imageProxy)
        }

        Toast.makeText(
            this,
            "Camera: ${if (cm.isFrontCamera) "Front" else "Back"}",
            Toast.LENGTH_SHORT
        ).show()
    }

    /**
     * 프레임 분석 결과 처리
     */
    private fun onFrameAnalyzed(result: AnalysisResult) {
        // 벤치마크 데이터 수집
        if (isBenchmarkRunning) {
            benchmarkManager.onFrameProcessed(result.processingTimeMs)
        }

        // FPS 업데이트
        val fpsText = if (isBenchmarkRunning) {
            val stats = benchmarkManager.performanceTracker.getStats()
            "FPS: %.1f | Lat: %.0fms".format(stats.fps, stats.avgLatencyMs)
        } else {
            "FPS: %.1f".format(result.fps)
        }
        binding.fpsText.text = fpsText

        // 검출 결과 저장
        lastIrisResult = result.result

        // 오버레이 뷰 업데이트
        val cm = cameraManager ?: return

        // ===== 좌표 디버그 로그 =====
        if (result.result.detected) {
            Log.d(TAG, """
                === 좌표 디버그 ===
                CameraManager: ${cm.imageWidth} x ${cm.imageHeight}
                IrisResult.frame: ${result.result.frameWidth} x ${result.result.frameHeight}
                SDK 좌표 (정규화 0~1):
                  - leftIrisX: ${result.result.leftIrisX}
                  - leftIrisY: ${result.result.leftIrisY}
                  - faceMesh[0]: x=${result.result.faceMesh?.getOrNull(0)}, y=${result.result.faceMesh?.getOrNull(1)}
                  - faceMesh[1]: x=${result.result.faceMesh?.getOrNull(3)}, y=${result.result.faceMesh?.getOrNull(4)}
                OverlayView: ${binding.overlayView.width} x ${binding.overlayView.height}
                isMirror: ${cm.isFrontCamera}
            """.trimIndent())
        }

        // SDK가 반환하는 frameWidth/Height 사용 (회전 후 이미지 크기)
        // CameraManager의 값과 일치해야 하지만, SDK 값이 더 정확함

        // DEBUG: Face Rect 값 출력
        if (result.result.faceRectWidth > 0) {
            Log.d(TAG, """
                [Face Rect Debug]
                frameSize: ${result.result.frameWidth} x ${result.result.frameHeight}
                faceRect: x=${result.result.faceRectX}, y=${result.result.faceRectY}, w=${result.result.faceRectWidth}, h=${result.result.faceRectHeight}
                overlayView: ${binding.overlayView.width} x ${binding.overlayView.height}
            """.trimIndent())
        }

        // 필터된 프레임과 함께 오버레이 뷰 업데이트
        binding.overlayView.setIrisResult(
            result.result,
            result.result.frameWidth,
            result.result.frameHeight,
            cm.isFrontCamera,
            result.filteredFrame  // 뷰티 필터가 적용된 프레임
        )
        binding.overlayView.setLensConfig(lensConfig)

        // 상태 텍스트 업데이트
        val status = when {
            !isSDKInitialized -> "SDK Not Ready (Model loading...)"
            !IrisLensSDK.isReady() -> "SDK Initializing..."
            result.result.detected -> {
                val leftStr = if (result.result.leftDetected) "L" else "-"
                val rightStr = if (result.result.rightDetected) "R" else "-"
                if (isBenchmarkRunning) {
                    val mem = benchmarkManager.memoryMonitor.getMemoryInfo()
                    "Tracking: $leftStr $rightStr | PSS: ${mem.totalPssMB}MB"
                } else {
                    "Tracking: $leftStr $rightStr (${result.processingTimeMs}ms)"
                }
            }
            else -> getString(R.string.status_no_face)
        }
        binding.statusText.text = status
    }

    // ==========================================================================
    // UI Setup
    // ==========================================================================

    /**
     * UI 컴포넌트 설정
     */
    private fun setupUI() {
        // 상태 텍스트 초기화
        binding.statusText.text = getString(R.string.status_initializing)
        binding.fpsText.text = "FPS: --"

        // 디버그 모드 (설정 버튼 롱클릭으로 토글 - Debug + Face Mesh)
        binding.settingsButton.setOnLongClickListener {
            binding.overlayView.debugMode = !binding.overlayView.debugMode
            binding.overlayView.showFaceMesh = binding.overlayView.debugMode
            Toast.makeText(
                this,
                "Debug mode: ${if (binding.overlayView.debugMode) "ON" else "OFF"}",
                Toast.LENGTH_SHORT
            ).show()
            true
        }
    }

    /**
     * 렌즈 선택기 설정 (RecyclerView 기반)
     */
    private fun setupLensSelector() {
        // 어댑터 초기화
        lensAdapter = LensAdapter { lens ->
            onLensSelected(lens)
        }

        // RecyclerView 설정
        binding.lensRecyclerView.apply {
            layoutManager = LinearLayoutManager(
                this@MainActivity,
                LinearLayoutManager.HORIZONTAL,
                false
            )
            adapter = lensAdapter
            setHasFixedSize(false)
        }

        // 렌즈 목록 로드
        lifecycleScope.launch {
            val lenses = lensManager.loadLensesFromAssets()
            lensAdapter.submitList(lenses)
            Log.i(TAG, "Loaded ${lenses.size} lenses")

            // 기본 선택: "없음"
            lensManager.clearLens()
        }

        // 렌즈 변경 리스너
        lensManager.onLensChangedListener = { lens ->
            if (lens != null && lens.id != NoLens.ID) {
                // 렌즈 텍스처 설정
                binding.overlayView.setLensTexture(lens.texture)
                binding.overlayView.showLens = true
            } else {
                // 렌즈 없음
                binding.overlayView.setLensTexture(null)
                binding.overlayView.showLens = false
            }
        }
    }

    /**
     * 렌즈 선택 처리
     */
    private fun onLensSelected(lens: LensData) {
        Log.d(TAG, "Lens selected: ${lens.name} (${lens.id})")

        // LensManager를 통해 선택
        lensManager.selectLens(lens)

        // 사용자 피드백
        if (lens.id != NoLens.ID) {
            Toast.makeText(
                this,
                "${lens.name} 선택됨",
                Toast.LENGTH_SHORT
            ).show()
        }
    }

    /**
     * 슬라이더 설정
     */
    private fun setupSliders() {
        // 초기값 설정 (LensConfig 기본값과 동일하게 유지)
        lensConfig.opacity = 0.4f  // 40%
        lensConfig.scale = 0.9f    // 90%

        // 슬라이더 초기 위치 설정
        binding.opacitySlider.value = lensConfig.opacity
        binding.scaleSlider.value = lensConfig.scale

        // 투명도 슬라이더
        binding.opacitySlider.addOnChangeListener { _, value, fromUser ->
            if (fromUser) {
                lensConfig.opacity = value
                binding.opacityValue.text = "${(value * 100).toInt()}%"
                applyLensSettings()
            }
        }

        // 크기 슬라이더
        binding.scaleSlider.addOnChangeListener { _, value, fromUser ->
            if (fromUser) {
                lensConfig.scale = value
                binding.scaleValue.text = "${(value * 100).toInt()}%"
                applyLensSettings()
            }
        }

        // 초기값 표시
        binding.opacityValue.text = "${(lensConfig.opacity * 100).toInt()}%"
        binding.scaleValue.text = "${(lensConfig.scale * 100).toInt()}%"
    }

    /**
     * 버튼 설정
     */
    private fun setupButtons() {
        // 캡처 버튼
        binding.captureButton.setOnClickListener {
            captureImage()
        }

        // 카메라 전환 버튼
        binding.switchCameraButton.setOnClickListener {
            switchCamera()
        }

        // 벤치마크 버튼 (기존 갤러리 버튼)
        binding.galleryButton.setOnClickListener {
            toggleBenchmark()
        }

        // 설정 버튼
        binding.settingsButton.setOnClickListener {
            openSettings()
        }

        // GPU 모드 버튼
        binding.btnGpuMode.setOnClickListener {
            openGpuRenderActivity()
        }
    }

    /**
     * GPU 렌더링 테스트 Activity 열기
     */
    private fun openGpuRenderActivity() {
        val intent = android.content.Intent(this, GpuRenderActivity::class.java)
        startActivity(intent)
    }

    // ==========================================================================
    // Actions
    // ==========================================================================

    /**
     * 렌즈 설정 적용
     */
    private fun applyLensSettings() {
        // OverlayView에 설정 업데이트
        binding.overlayView.setLensConfig(lensConfig)

        Log.d(TAG, "Applying lens settings: opacity=${lensConfig.opacity}, scale=${lensConfig.scale}")
    }

    /**
     * 이미지 캡처
     */
    private fun captureImage() {
        Log.d(TAG, "Capturing image...")

        // TODO: 구현
        // - 현재 프레임 캡처
        // - 렌즈 오버레이 적용
        // - 갤러리에 저장

        Toast.makeText(this, "Capture (TODO)", Toast.LENGTH_SHORT).show()
    }

    /**
     * 벤치마크 시작/종료 토글
     */
    private fun toggleBenchmark() {
        if (isBenchmarkRunning) {
            // 벤치마크 종료
            benchmarkManager.stop()
        } else {
            // 벤치마크 시작
            Log.i(TAG, "Starting benchmark for ${benchmarkDurationMs}ms")

            // 모드 정보 설정 (CSV/리포트에 기록)
            val usingInferenceThread = IrisLensSDK.isUsingInferenceThread()
            benchmarkManager.setInferenceThreadEnabled(usingInferenceThread)
            benchmarkManager.setGpuEnabled(IrisLensSDK.isUsingGpu())
            Log.i(TAG, "Benchmark mode: inferenceThread=$usingInferenceThread, gpu=${IrisLensSDK.isUsingGpu()}")

            benchmarkManager.start(benchmarkDurationMs, object : BenchmarkCallback {
                override fun onBenchmarkStarted() {
                    runOnUiThread {
                        isBenchmarkRunning = true
                        binding.galleryButton.alpha = 0.5f  // 시각적 피드백
                        Toast.makeText(
                            this@MainActivity,
                            "Benchmark started (${benchmarkDurationMs / 1000}s)",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                }

                override fun onBenchmarkProgress(
                    elapsedMs: Long,
                    stats: PerformanceStats,
                    memory: MemoryInfo
                ) {
                    // 진행 상황 로깅 (매 초)
                    Log.d(TAG, "Benchmark: ${elapsedMs/1000}s | " +
                            "FPS: %.1f | Lat: %.1fms | PSS: ${memory.totalPssMB}MB"
                                .format(stats.fps, stats.avgLatencyMs))
                }

                override fun onBenchmarkCompleted(result: BenchmarkResult) {
                    runOnUiThread {
                        isBenchmarkRunning = false
                        binding.galleryButton.alpha = 1.0f

                        // 결과 로깅
                        Log.i(TAG, "Benchmark completed!")
                        Log.i(TAG, "Duration: ${result.testDurationMs}ms")
                        Log.i(TAG, "FPS: ${result.performance.fps}")
                        Log.i(TAG, "Avg Latency: ${result.performance.avgLatencyMs}ms")
                        Log.i(TAG, "P95 Latency: ${result.performance.p95LatencyMs}ms")
                        Log.i(TAG, "Drop Rate: ${result.performance.dropRate * 100}%")
                        Log.i(TAG, "Memory PSS: ${result.memory.totalPssMB}MB")

                        // 마크다운 리포트 생성
                        val report = benchmarkManager.generateMarkdownReport()
                        Log.i(TAG, "\n$report")

                        // CSV 내보내기
                        val csvFile = benchmarkManager.exportToCsv()
                        csvFile?.let {
                            Log.i(TAG, "CSV exported to: ${it.absolutePath}")
                        }

                        // 결과 요약 토스트
                        val passCount = listOf(
                            result.performance.fps >= 30,
                            result.performance.avgLatencyMs <= 33,
                            result.memory.totalPssMB <= 100
                        ).count { it }

                        Toast.makeText(
                            this@MainActivity,
                            "Benchmark done! FPS: %.1f, Lat: %.0fms, Mem: ${result.memory.totalPssMB}MB ($passCount/3 passed)"
                                .format(result.performance.fps, result.performance.avgLatencyMs),
                            Toast.LENGTH_LONG
                        ).show()
                    }
                }

                override fun onBenchmarkError(error: String) {
                    runOnUiThread {
                        isBenchmarkRunning = false
                        binding.galleryButton.alpha = 1.0f
                        Toast.makeText(this@MainActivity, "Benchmark error: $error", Toast.LENGTH_SHORT).show()
                    }
                }
            })
        }
    }

    /**
     * 설정 다이얼로그 열기
     */
    private fun openSettings() {
        // 빌드 정보 가져오기
        val versionName = packageManager.getPackageInfo(packageName, 0).versionName
        val versionCode = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.P) {
            packageManager.getPackageInfo(packageName, 0).longVersionCode
        } else {
            @Suppress("DEPRECATION")
            packageManager.getPackageInfo(packageName, 0).versionCode.toLong()
        }

        val beautyEnabled = IrisLensSDK.isBeautyFilterEnabled()
        val options = arrayOf(
            "Face Mesh 표시: ${if (binding.overlayView.showFaceMesh) "ON" else "OFF"}",
            "Face Rect 표시: ${if (binding.overlayView.showFaceRect) "ON" else "OFF"}",
            "Debug 모드: ${if (binding.overlayView.debugMode) "ON" else "OFF"}",
            "렌즈 표시: ${if (binding.overlayView.showLens) "ON" else "OFF"}",
            "뷰티 필터: ${if (beautyEnabled) "ON" else "OFF"}",
            "뷰티 필터 설정...",
            "신뢰도 설정..."
        )

        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("디스플레이 설정 (v$versionName build $versionCode)")
            .setItems(options) { _, which ->
                when (which) {
                    0 -> {
                        binding.overlayView.showFaceMesh = !binding.overlayView.showFaceMesh
                        Toast.makeText(
                            this,
                            "Face Mesh: ${if (binding.overlayView.showFaceMesh) "ON" else "OFF"}",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                    1 -> {
                        binding.overlayView.showFaceRect = !binding.overlayView.showFaceRect
                        Toast.makeText(
                            this,
                            "Face Rect: ${if (binding.overlayView.showFaceRect) "ON" else "OFF"}",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                    2 -> {
                        binding.overlayView.debugMode = !binding.overlayView.debugMode
                        Toast.makeText(
                            this,
                            "Debug mode: ${if (binding.overlayView.debugMode) "ON" else "OFF"}",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                    3 -> {
                        binding.overlayView.showLens = !binding.overlayView.showLens
                        Toast.makeText(
                            this,
                            "렌즈 표시: ${if (binding.overlayView.showLens) "ON" else "OFF"}",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                    4 -> toggleBeautyFilter()
                    5 -> openBeautyFilterSettings()
                    6 -> openConfidenceSettings()
                }
            }
            .setNegativeButton("닫기", null)
            .show()
    }

    /**
     * 뷰티 필터 활성화 토글
     *
     * 필터 ON 시:
     * - PreviewView 숨김 (원본 피드)
     * - OverlayView에서 필터된 프레임을 배경으로 표시
     *
     * 필터 OFF 시:
     * - PreviewView 표시 (원본 피드)
     * - OverlayView는 렌즈만 오버레이
     */
    private fun toggleBeautyFilter() {
        beautyConfigV2.enabled = !beautyConfigV2.enabled
        beautyConfig.enabled = beautyConfigV2.enabled  // V1 동기화
        val error = IrisLensSDK.setBeautyFilter(beautyConfig)

        if (error == IrisLensSDK.OK) {
            // FrameAnalyzer에 필터 활성화 상태 전달
            frameAnalyzer?.beautyFilterEnabled = beautyConfigV2.enabled

            // OverlayView 설정
            binding.overlayView.showFilteredFrame = beautyConfigV2.enabled

            // PreviewView 가시성 제어
            // 필터 ON: PreviewView 숨김 → OverlayView가 필터된 프레임 배경 표시
            // 필터 OFF: PreviewView 표시 → 원본 카메라 피드 직접 표시
            binding.cameraPreview.visibility = if (beautyConfigV2.enabled) {
                View.INVISIBLE
            } else {
                View.VISIBLE
            }

            Toast.makeText(
                this,
                "뷰티 필터 V2: ${if (beautyConfigV2.enabled) "ON" else "OFF"}",
                Toast.LENGTH_SHORT
            ).show()
            Log.d(TAG, "Beauty filter V2 toggled: ${beautyConfigV2.enabled}, previewVisible=${!beautyConfigV2.enabled}")
        } else {
            Toast.makeText(
                this,
                "뷰티 필터 설정 실패",
                Toast.LENGTH_SHORT
            ).show()
            Log.e(TAG, "Beauty filter toggle failed: ${IrisLensSDK.errorToString(error)}")
        }
    }

    /**
     * 뷰티 필터 V2 설정 다이얼로그
     */
    private fun openBeautyFilterSettings() {
        val scrollView = android.widget.ScrollView(this)
        val layout = android.widget.LinearLayout(this).apply {
            orientation = android.widget.LinearLayout.VERTICAL
            setPadding(48, 32, 48, 16)
        }
        scrollView.addView(layout)

        // 헬퍼 함수: 슬라이더 추가
        fun addSlider(
            label: String,
            value: Float,
            max: Int = 100,
            suffix: String = "%",
            onChanged: (Float) -> Unit
        ): android.widget.SeekBar {
            val textView = android.widget.TextView(this).apply {
                text = "$label: ${String.format("%.0f", value * 100)}$suffix"
            }
            val seekBar = android.widget.SeekBar(this).apply {
                this.max = max
                progress = (value * 100).toInt()
                setOnSeekBarChangeListener(object : android.widget.SeekBar.OnSeekBarChangeListener {
                    override fun onProgressChanged(seekBar: android.widget.SeekBar?, progress: Int, fromUser: Boolean) {
                        val newValue = progress / 100f
                        onChanged(newValue)
                        textView.text = "$label: ${progress}$suffix"
                    }
                    override fun onStartTrackingTouch(seekBar: android.widget.SeekBar?) {}
                    override fun onStopTrackingTouch(seekBar: android.widget.SeekBar?) {}
                })
            }
            layout.addView(textView)
            layout.addView(seekBar)
            layout.addView(android.widget.Space(this).apply {
                layoutParams = android.widget.LinearLayout.LayoutParams(
                    android.widget.LinearLayout.LayoutParams.MATCH_PARENT, 12
                )
            })
            return seekBar
        }

        // 섹션 제목 헬퍼
        fun addSectionTitle(title: String) {
            layout.addView(android.widget.TextView(this).apply {
                text = title
                setTextColor(android.graphics.Color.parseColor("#2196F3"))
                textSize = 14f
                setPadding(0, 16, 0, 8)
            })
        }

        // ========== 피부 효과 섹션 ==========
        addSectionTitle("📸 피부 효과")

        addSlider("전체 강도 (Intensity)", beautyConfigV2.intensity) { v ->
            beautyConfigV2.intensity = v
            beautyConfig.intensity = v
        }

        addSlider("피부 스무딩 (Smoothing)", beautyConfigV2.smoothing) { v ->
            beautyConfigV2.smoothing = v
            beautyConfig.smoothing = v
        }

        addSlider("밝기 (Brightness)", beautyConfigV2.brightness, max = 150) { v ->
            beautyConfigV2.brightness = v
            beautyConfig.brightness = v
        }

        addSlider("소프트 포커스 (Soft Focus)", beautyConfigV2.softFocus) { v ->
            beautyConfigV2.softFocus = v
            beautyConfig.softFocus = v
        }

        addSlider("화이트닝 (Whitening)", beautyConfigV2.whitening) { v ->
            beautyConfigV2.whitening = v
        }

        addSlider("잡티 보정 (Skin Quality)", beautyConfigV2.skinQuality) { v ->
            beautyConfigV2.skinQuality = v
        }

        // ========== Face Warp 섹션 ==========
        addSectionTitle("✨ 얼굴 보정 (Face Warp)")

        addSlider("갸름한 얼굴 (Slim Face)", beautyConfigV2.slimFace) { v ->
            beautyConfigV2.slimFace = v
        }

        addSlider("V라인 (Thin Chin)", beautyConfigV2.thinChin) { v ->
            beautyConfigV2.thinChin = v
        }

        addSlider("눈 확대 (Enlarge Eyes)", beautyConfigV2.enlargeEyes) { v ->
            beautyConfigV2.enlargeEyes = v
        }

        // 설명 추가
        val description = android.widget.TextView(this).apply {
            text = """
                |
                |📸 피부 효과: CPU 기반 실시간 적용
                |✨ 얼굴 보정: GPU 파이프라인 필요 (추후 지원)
                |
                |• ROI 기반 처리로 성능 최적화
                |• 눈/입술 영역 자동 보호
            """.trimMargin()
            setTextColor(android.graphics.Color.GRAY)
            textSize = 11f
        }
        layout.addView(description)

        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("뷰티 필터 V2 설정")
            .setView(scrollView)
            .setPositiveButton("확인", null)
            .setNeutralButton("기본값 복원") { _, _ ->
                beautyConfigV2.setDefaults()
                beautyConfig.intensity = BeautyFilterConfig.DEFAULT_INTENSITY
                beautyConfig.smoothing = BeautyFilterConfig.DEFAULT_SMOOTHING
                beautyConfig.brightness = BeautyFilterConfig.DEFAULT_BRIGHTNESS
                beautyConfig.softFocus = BeautyFilterConfig.DEFAULT_SOFT_FOCUS
                IrisLensSDK.setBeautyFilter(beautyConfig)
                Toast.makeText(this, "기본값으로 복원됨", Toast.LENGTH_SHORT).show()
            }
            .show()
    }

    // 신뢰도 설정값 저장
    private var detectionConfidence = 0.3f
    private var trackingConfidence = 0.5f
    private var presenceConfidence = 0.5f

    /**
     * 신뢰도 설정 다이얼로그 열기
     */
    private fun openConfidenceSettings() {
        val layout = android.widget.LinearLayout(this).apply {
            orientation = android.widget.LinearLayout.VERTICAL
            setPadding(48, 32, 48, 16)
        }

        // Detection Confidence 슬라이더
        val detectionLabel = android.widget.TextView(this).apply {
            text = "검출 신뢰도 (Detection): ${String.format("%.1f", detectionConfidence)}"
        }
        val detectionSeekBar = android.widget.SeekBar(this).apply {
            max = 100
            progress = (detectionConfidence * 100).toInt()
            setOnSeekBarChangeListener(object : android.widget.SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(seekBar: android.widget.SeekBar?, progress: Int, fromUser: Boolean) {
                    detectionConfidence = progress / 100f
                    detectionLabel.text = "검출 신뢰도 (Detection): ${String.format("%.1f", detectionConfidence)}"
                }
                override fun onStartTrackingTouch(seekBar: android.widget.SeekBar?) {}
                override fun onStopTrackingTouch(seekBar: android.widget.SeekBar?) {
                    IrisLensSDK.setMinDetectionConfidence(detectionConfidence)
                    Log.d(TAG, "Detection confidence: $detectionConfidence")
                }
            })
        }

        // Tracking Confidence 슬라이더
        val trackingLabel = android.widget.TextView(this).apply {
            text = "추적 신뢰도 (Tracking): ${String.format("%.1f", trackingConfidence)}"
        }
        val trackingSeekBar = android.widget.SeekBar(this).apply {
            max = 100
            progress = (trackingConfidence * 100).toInt()
            setOnSeekBarChangeListener(object : android.widget.SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(seekBar: android.widget.SeekBar?, progress: Int, fromUser: Boolean) {
                    trackingConfidence = progress / 100f
                    trackingLabel.text = "추적 신뢰도 (Tracking): ${String.format("%.1f", trackingConfidence)}"
                }
                override fun onStartTrackingTouch(seekBar: android.widget.SeekBar?) {}
                override fun onStopTrackingTouch(seekBar: android.widget.SeekBar?) {
                    IrisLensSDK.setMinTrackingConfidence(trackingConfidence)
                    Log.d(TAG, "Tracking confidence: $trackingConfidence")
                }
            })
        }

        // Presence Confidence 슬라이더
        val presenceLabel = android.widget.TextView(this).apply {
            text = "존재 신뢰도 (Presence): ${String.format("%.1f", presenceConfidence)}"
        }
        val presenceSeekBar = android.widget.SeekBar(this).apply {
            max = 100
            progress = (presenceConfidence * 100).toInt()
            setOnSeekBarChangeListener(object : android.widget.SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(seekBar: android.widget.SeekBar?, progress: Int, fromUser: Boolean) {
                    presenceConfidence = progress / 100f
                    presenceLabel.text = "존재 신뢰도 (Presence): ${String.format("%.1f", presenceConfidence)}"
                }
                override fun onStartTrackingTouch(seekBar: android.widget.SeekBar?) {}
                override fun onStopTrackingTouch(seekBar: android.widget.SeekBar?) {
                    IrisLensSDK.setMinPresenceConfidence(presenceConfidence)
                    Log.d(TAG, "Presence confidence: $presenceConfidence")
                }
            })
        }

        // 레이아웃에 추가
        layout.addView(detectionLabel)
        layout.addView(detectionSeekBar)
        layout.addView(android.widget.Space(this).apply {
            layoutParams = android.widget.LinearLayout.LayoutParams(
                android.widget.LinearLayout.LayoutParams.MATCH_PARENT, 24
            )
        })
        layout.addView(trackingLabel)
        layout.addView(trackingSeekBar)
        layout.addView(android.widget.Space(this).apply {
            layoutParams = android.widget.LinearLayout.LayoutParams(
                android.widget.LinearLayout.LayoutParams.MATCH_PARENT, 24
            )
        })
        layout.addView(presenceLabel)
        layout.addView(presenceSeekBar)

        // 설명 추가
        val description = android.widget.TextView(this).apply {
            text = "\n• Detection: 얼굴 검출 민감도\n• Tracking: 랜드마크 추적 신뢰도\n• Presence: 추적 캐시 재사용 임계값"
            setTextColor(android.graphics.Color.GRAY)
            textSize = 12f
        }
        layout.addView(description)

        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("신뢰도 설정")
            .setView(layout)
            .setPositiveButton("확인", null)
            .setNeutralButton("기본값 복원") { _, _ ->
                detectionConfidence = 0.3f
                trackingConfidence = 0.5f
                presenceConfidence = 0.5f
                IrisLensSDK.setMinDetectionConfidence(detectionConfidence)
                IrisLensSDK.setMinTrackingConfidence(trackingConfidence)
                IrisLensSDK.setMinPresenceConfidence(presenceConfidence)
                Toast.makeText(this, "기본값으로 복원됨", Toast.LENGTH_SHORT).show()
            }
            .show()
    }

    /**
     * Face Mesh 표시 토글 (Long press debug mode에서 제공)
     */
    private fun toggleFaceMesh() {
        binding.overlayView.showFaceMesh = !binding.overlayView.showFaceMesh

        Toast.makeText(
            this,
            "Face Mesh: ${if (binding.overlayView.showFaceMesh) "ON" else "OFF"}",
            Toast.LENGTH_SHORT
        ).show()

        Log.d(TAG, "Face Mesh: ${binding.overlayView.showFaceMesh}")
    }

    // ==========================================================================
    // Helper Methods
    // ==========================================================================

    /**
     * 상태 업데이트
     */
    private fun updateStatus(status: String) {
        runOnUiThread {
            binding.statusText.text = status
        }
    }
}


