/**
 * IrisLensSDK Android - Demo MainActivity
 *
 * SDK 기능 데모를 위한 메인 액티비티
 * - CameraX 카메라 프리뷰
 * - 실시간 홍채 검출
 * - 렌즈 오버레이 표시
 *
 * @version 1.0.0
 */
package com.irislenssdk.demo

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
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

    // SDK 인스턴스
    private var irisSDK: IrisLensSDK? = null

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
                "Camera permission is required for AR lens fitting",
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

            irisSDK = IrisLensSDK.getInstance(this)

            // SDK 버전 로그
            Log.i(TAG, "SDK Version: ${IrisLensSDK.getVersion()}")

            // TODO: P1-W6에서 실제 초기화 구현
            // irisSDK?.initialize(modelPath)

            Log.d(TAG, "IrisLensSDK initialized successfully")

        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize IrisLensSDK", e)
            Toast.makeText(
                this,
                "Failed to initialize SDK: ${e.message}",
                Toast.LENGTH_LONG
            ).show()
        }
    }

    /**
     * SDK 리소스 해제
     */
    private fun releaseSDK() {
        try {
            irisSDK?.destroy()
            irisSDK = null
            Log.d(TAG, "IrisLensSDK released")
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

        binding.statusText.text = "Camera ready\nSDK Version: ${IrisLensSDK.getVersion()}"
    }

    // ==========================================================================
    // UI Setup
    // ==========================================================================

    /**
     * UI 컴포넌트 설정
     */
    private fun setupUI() {
        // 상태 텍스트 초기화
        binding.statusText.text = "Initializing..."

        // TODO: P1-W6-01에서 전체 UI 구현
        // - 렌즈 선택 버튼
        // - 캡처 버튼
        // - 설정 메뉴
    }
}
