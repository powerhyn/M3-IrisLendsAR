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
import android.util.Log
import android.view.View
import android.widget.LinearLayout
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.google.android.material.slider.Slider
import com.irislenssdk.IrisLensSDK
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

    // 현재 카메라 (true = 전면, false = 후면)
    private var isFrontCamera: Boolean = true

    // 렌즈 설정 값
    private var lensOpacity: Float = 0.8f
    private var lensScale: Float = 1.0f

    // 렌즈 뷰 맵
    private val lensViews = mutableMapOf<String, View>()

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
    }

    override fun onPause() {
        super.onPause()
        // 카메라 일시정지 (필요시)
    }

    override fun onDestroy() {
        super.onDestroy()
        // SDK 리소스 해제
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
            IrisLensSDK.init(this)
            isSDKInitialized = true

            // SDK 버전 로그
            Log.i(TAG, "SDK Version: ${IrisLensSDK.getVersion()}")

            // TODO: P1-W6-02에서 실제 초기화 구현
            // - 모델 로드
            // - 카메라 파이프라인 설정

            Log.d(TAG, "IrisLensSDK initialized successfully")

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

        // TODO: P1-W6-02에서 CameraX 통합 구현
        // - PreviewView 설정
        // - ImageAnalysis 콜백 설정
        // - 실시간 홍채 검출 파이프라인 구성

        binding.statusText.text = "Camera ready\nSDK: ${IrisLensSDK.getVersion()}"
    }

    /**
     * 카메라 전환 (전면 ↔ 후면)
     */
    private fun switchCamera() {
        isFrontCamera = !isFrontCamera
        Log.d(TAG, "Switching to ${if (isFrontCamera) "front" else "back"} camera")

        // TODO: P1-W6-02에서 구현
        // CameraX 카메라 전환 로직

        Toast.makeText(
            this,
            "Camera: ${if (isFrontCamera) "Front" else "Back"}",
            Toast.LENGTH_SHORT
        ).show()
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
        // 투명도 슬라이더
        binding.opacitySlider.addOnChangeListener { _, value, fromUser ->
            if (fromUser) {
                lensOpacity = value
                binding.opacityValue.text = "${(value * 100).toInt()}%"
                applyLensSettings()
            }
        }

        // 크기 슬라이더
        binding.scaleSlider.addOnChangeListener { _, value, fromUser ->
            if (fromUser) {
                lensScale = value
                binding.scaleValue.text = "${(value * 100).toInt()}%"
                applyLensSettings()
            }
        }

        // 초기값 표시
        binding.opacityValue.text = "${(lensOpacity * 100).toInt()}%"
        binding.scaleValue.text = "${(lensScale * 100).toInt()}%"
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
        // TODO: P1-W6-02에서 SDK 연동
        // irisSDK?.setLens(currentLensId)
        // irisSDK?.setOpacity(lensOpacity)
        // irisSDK?.setScale(lensScale)

        Log.d(TAG, "Applying lens: $currentLensId, opacity: $lensOpacity, scale: $lensScale")
    }

    /**
     * 이미지 캡처
     */
    private fun captureImage() {
        Log.d(TAG, "Capturing image...")

        // TODO: P1-W6-02에서 구현
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
    // FPS Update
    // ==========================================================================

    /**
     * FPS 업데이트 (P1-W6-02에서 호출)
     */
    fun updateFPS(fps: Float) {
        runOnUiThread {
            binding.fpsText.text = "FPS: %.1f".format(fps)
        }
    }

    /**
     * 상태 업데이트
     */
    fun updateStatus(status: String) {
        runOnUiThread {
            binding.statusText.text = status
        }
    }
}
