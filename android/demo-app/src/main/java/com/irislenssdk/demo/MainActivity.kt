/**
 * IrisLensSDK Android - Demo MainActivity
 *
 * SDK 기능 데모를 위한 메인 액티비티
 * - CameraX 카메라 프리뷰
 * - 실시간 홍채 검출
 * - 렌즈 오버레이 표시
 * - 렌즈 선택 및 설정 조절
 *
 * @version 1.0.0
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
import com.irislenssdk.IrisLensSDK
import com.irislenssdk.IrisResult
import com.irislenssdk.LensConfig
import com.irislenssdk.demo.camera.AnalysisResult
import com.irislenssdk.demo.camera.CameraManager
import com.irislenssdk.demo.camera.FrameAnalyzer
import com.irislenssdk.demo.databinding.ActivityMainBinding

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

    // 현재 선택된 렌즈
    private var currentLensId: String = "off"

    // 렌즈 설정
    private val lensConfig = LensConfig()

    // 렌즈 뷰 맵
    private val lensViews = mutableMapOf<String, View>()

    // CameraX 관리자
    private var cameraManager: CameraManager? = null

    // 프레임 분석기
    private var frameAnalyzer: FrameAnalyzer? = null

    // UI 업데이트 핸들러
    private val mainHandler = Handler(Looper.getMainLooper())

    // 마지막 검출 결과
    private var lastIrisResult: IrisResult? = null

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

            // SDK 초기화 (정적 메서드 사용)
            val error = IrisLensSDK.init(this)

            if (error == IrisLensSDK.OK) {
                isSDKInitialized = true
                Log.i(TAG, "SDK Version: ${IrisLensSDK.getVersion()}")
                Log.d(TAG, "IrisLensSDK initialized successfully")

                updateStatus("SDK Ready\n${IrisLensSDK.getVersion()}")
            } else {
                Log.e(TAG, "SDK init error: ${IrisLensSDK.errorToString(error)}")
                isSDKInitialized = false
                updateStatus("SDK Init Error: ${IrisLensSDK.errorToString(error)}")
            }

        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize IrisLensSDK", e)
            isSDKInitialized = false
            Toast.makeText(
                this,
                "${getString(R.string.error_sdk_init)}: ${e.message}",
                Toast.LENGTH_LONG
            ).show()
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
        }

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
        // FPS 업데이트
        binding.fpsText.text = "FPS: %.1f".format(result.fps)

        // 검출 결과 저장
        lastIrisResult = result.result

        // 오버레이 뷰 업데이트
        val cm = cameraManager ?: return
        binding.overlayView.setIrisResult(
            result.result,
            cm.imageWidth,
            cm.imageHeight,
            cm.isFrontCamera
        )
        binding.overlayView.setLensConfig(lensConfig)

        // 상태 텍스트 업데이트
        val status = when {
            result.result.detected -> {
                val leftStr = if (result.result.leftDetected) "L" else "-"
                val rightStr = if (result.result.rightDetected) "R" else "-"
                "Tracking: $leftStr $rightStr (${result.processingTimeMs}ms)"
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

        // 디버그 모드 (설정 버튼 롱클릭으로 토글)
        binding.settingsButton.setOnLongClickListener {
            binding.overlayView.debugMode = !binding.overlayView.debugMode
            Toast.makeText(
                this,
                "Debug mode: ${if (binding.overlayView.debugMode) "ON" else "OFF"}",
                Toast.LENGTH_SHORT
            ).show()
            true
        }
    }

    /**
     * 렌즈 선택기 설정
     */
    private fun setupLensSelector() {
        // 렌즈 뷰 맵 초기화
        lensViews["blue"] = binding.lensBlue
        lensViews["green"] = binding.lensGreen
        lensViews["brown"] = binding.lensBrown
        lensViews["gray"] = binding.lensGray
        lensViews["off"] = binding.lensOff

        // 각 렌즈에 클릭 리스너 설정
        lensViews.forEach { (lensId, view) ->
            view.setOnClickListener {
                selectLens(lensId)
            }
        }

        // 초기 선택 (Off)
        selectLens("off")
    }

    /**
     * 렌즈 선택
     */
    private fun selectLens(lensId: String) {
        // 이전 선택 해제
        lensViews[currentLensId]?.isSelected = false

        // 새 선택 적용
        currentLensId = lensId
        lensViews[currentLensId]?.isSelected = true

        Log.d(TAG, "Selected lens: $lensId")

        // SDK에 렌즈 적용
        applyLensSettings()

        // 사용자 피드백
        if (lensId != "off") {
            Toast.makeText(
                this,
                "${lensId.replaceFirstChar { it.uppercase() }} lens selected",
                Toast.LENGTH_SHORT
            ).show()
        }
    }

    /**
     * 슬라이더 설정
     */
    private fun setupSliders() {
        // 초기값 설정
        lensConfig.opacity = 0.8f
        lensConfig.scale = 1.0f

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

        // 갤러리 버튼
        binding.galleryButton.setOnClickListener {
            openGallery()
        }

        // 설정 버튼
        binding.settingsButton.setOnClickListener {
            openSettings()
        }
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

        Log.d(TAG, "Applying lens: $currentLensId, opacity: ${lensConfig.opacity}, scale: ${lensConfig.scale}")
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
     * 갤러리 열기
     */
    private fun openGallery() {
        Log.d(TAG, "Opening gallery...")

        // TODO: 구현
        // - 저장된 이미지 목록 표시

        Toast.makeText(this, "Gallery (TODO)", Toast.LENGTH_SHORT).show()
    }

    /**
     * 설정 화면 열기
     */
    private fun openSettings() {
        Log.d(TAG, "Opening settings...")

        // TODO: 구현
        // - 설정 다이얼로그 또는 화면

        Toast.makeText(this, "Settings (TODO)", Toast.LENGTH_SHORT).show()
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
