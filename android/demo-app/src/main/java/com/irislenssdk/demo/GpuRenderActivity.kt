/**
 * IrisLensSDK Android - GPU Render Activity
 *
 * GPU 기반 렌더링 테스트용 Activity
 * - GLSurfaceView + CameraX 연동
 * - GPU Beauty Filter (OpenGL ES 3.1)
 * - GPU Lens Overlay (셰이더 기반)
 * - MediaPipe 추론은 CPU (하이브리드 아키텍처)
 *
 * @version 2.0.0
 */
package com.irislenssdk.demo

import android.Manifest
import android.app.ActivityManager
import android.content.Context
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.tabs.TabLayout
import com.irislenssdk.BeautyFilterConfigV2
import com.irislenssdk.IrisLensSDK
import com.irislenssdk.IrisResult
import com.irislenssdk.LensConfig
import com.irislenssdk.demo.beauty.BeautyPreset
import com.irislenssdk.demo.beauty.BeautyPresetFactory
import com.irislenssdk.demo.beauty.LutTextureLoader
import com.irislenssdk.demo.camera.OverlayView
import com.irislenssdk.demo.camera.gpu.CameraGLView
import com.irislenssdk.demo.lens.LensAdapter
import com.irislenssdk.demo.lens.LensData
import com.irislenssdk.demo.lens.LensManager
import com.irislenssdk.demo.lens.NoLens
import kotlinx.coroutines.launch
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * GPU 렌더링 테스트 Activity
 *
 * 하이브리드 아키텍처:
 * - MediaPipe (CPU): 홍채 추적 (FrameAnalyzer)
 * - GPU: 렌즈 오버레이 + 뷰티 필터 렌더링 (CameraGLView)
 */
class GpuRenderActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "GpuRenderActivity"
        private const val REQUEST_CAMERA_PERMISSION = 1001
    }

    // UI
    private lateinit var cameraGLView: CameraGLView
    private lateinit var overlayView: OverlayView
    private lateinit var tvFps: TextView
    private lateinit var tvGpuFps: TextView
    private lateinit var tvGpuStatus: TextView
    private lateinit var tvLensStatus: TextView
    private lateinit var tabLayout: TabLayout
    private lateinit var lensTabContent: LinearLayout
    private lateinit var beautyTabContent: ScrollView

    // 디버그 버튼
    private lateinit var btnToggleMesh: Button
    private lateinit var btnToggleDebug: Button
    private lateinit var btnToggleIris: Button

    // 렌즈 탭 UI
    private lateinit var rvLenses: RecyclerView
    private lateinit var seekLensOpacity: SeekBar
    private lateinit var seekLensScale: SeekBar
    private lateinit var seekLensFeather: SeekBar
    private lateinit var rgBlendMode: RadioGroup

    // 뷰티 탭 UI
    private lateinit var btnToggleBeauty: Button
    private lateinit var btnPresetNatural: Button
    private lateinit var btnPresetStudio: Button
    private lateinit var btnPresetGlamour: Button
    private lateinit var btnPresetCustom: Button
    private lateinit var seekSmoothing: SeekBar
    private lateinit var seekBrightness: SeekBar
    private lateinit var seekWhitening: SeekBar
    private lateinit var seekColorBalance: SeekBar
    private lateinit var seekSoftFocus: SeekBar
    private lateinit var btnToggleLut: Button
    private lateinit var lutIntensityPanel: LinearLayout
    private lateinit var seekLutIntensity: SeekBar

    // 카메라
    private var cameraProvider: ProcessCameraProvider? = null
    private var lensFacing = CameraSelector.LENS_FACING_FRONT
    private val analysisExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    // 렌즈 관리
    private lateinit var lensManager: LensManager
    private lateinit var lensAdapter: LensAdapter
    private var lensConfig = LensConfig()

    // 뷰티 설정
    private var beautyConfig = BeautyFilterConfigV2()
    private var beautyEnabled = true
    private var currentPreset = BeautyPreset.CUSTOM
    private var isUpdatingSliders = false
    private var lutEnabled = false

    // 홍채 검출 (스레드별 불변 스냅샷 사용)
    private val irisResult = IrisResult()       // Analyzer 스레드 전용 (JNI 결과 수신)
    private val glIrisResult = IrisResult()     // GL 스레드 전달용 스냅샷
    private val uiIrisResult = IrisResult()     // UI 스레드 전달용 스냅샷

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
        initLensManager()
        initSDK()

        if (hasCameraPermission()) {
            startCamera()
        } else {
            requestCameraPermission()
        }
    }

    private fun initViews() {
        // Camera view
        cameraGLView = findViewById(R.id.cameraGLView)
        overlayView = findViewById(R.id.overlayView)
        tvFps = findViewById(R.id.tvFps)
        tvGpuFps = findViewById(R.id.tvGpuFps)
        tvGpuStatus = findViewById(R.id.tvGpuStatus)
        tvLensStatus = findViewById(R.id.tvLensStatus)

        // Tab layout
        tabLayout = findViewById(R.id.tabLayout)
        lensTabContent = findViewById(R.id.lensTabContent)
        beautyTabContent = findViewById(R.id.beautyTabContent)

        // 렌즈 탭 UI
        rvLenses = findViewById(R.id.rvLenses)
        seekLensOpacity = findViewById(R.id.seekLensOpacity)
        seekLensScale = findViewById(R.id.seekLensScale)
        seekLensFeather = findViewById(R.id.seekLensFeather)
        rgBlendMode = findViewById(R.id.rgBlendMode)

        // 뷰티 탭 UI
        btnToggleBeauty = findViewById(R.id.btnToggleBeauty)
        btnPresetNatural = findViewById(R.id.btnPresetNatural)
        btnPresetStudio = findViewById(R.id.btnPresetStudio)
        btnPresetGlamour = findViewById(R.id.btnPresetGlamour)
        btnPresetCustom = findViewById(R.id.btnPresetCustom)
        seekSmoothing = findViewById(R.id.seekSmoothing)
        seekBrightness = findViewById(R.id.seekBrightness)
        seekWhitening = findViewById(R.id.seekWhitening)
        seekColorBalance = findViewById(R.id.seekColorBalance)
        seekSoftFocus = findViewById(R.id.seekSoftFocus)
        btnToggleLut = findViewById(R.id.btnToggleLut)
        lutIntensityPanel = findViewById(R.id.lutIntensityPanel)
        seekLutIntensity = findViewById(R.id.seekLutIntensity)

        // GPU 초기화 콜백 설정
        cameraGLView.onGpuInitialized = { success ->
            runOnUiThread {
                tvGpuStatus.text = "GPU: Available (init: $success)"
                Log.d(TAG, "GPU initialized: $success")
            }
        }

        // GPU FPS 콜백 설정
        cameraGLView.onGpuFpsUpdated = { fps ->
            runOnUiThread {
                tvGpuFps.text = "Render FPS: $fps"
            }
        }

        // Tab 리스너
        tabLayout.addOnTabSelectedListener(object : TabLayout.OnTabSelectedListener {
            override fun onTabSelected(tab: TabLayout.Tab?) {
                when (tab?.position) {
                    0 -> {
                        lensTabContent.visibility = View.VISIBLE
                        beautyTabContent.visibility = View.GONE
                    }
                    1 -> {
                        lensTabContent.visibility = View.GONE
                        beautyTabContent.visibility = View.VISIBLE
                    }
                }
            }
            override fun onTabUnselected(tab: TabLayout.Tab?) {}
            override fun onTabReselected(tab: TabLayout.Tab?) {}
        })

        setupLensControls()
        setupBeautyControls()
        setupLutControls()
        setupDebugControls()

        // OverlayView 초기 설정: 렌즈는 GPU에서 렌더링하므로 OverlayView에서는 비활성화
        overlayView.showLens = false
        overlayView.showFaceMesh = false
        overlayView.debugMode = false
        overlayView.showFaceRect = false
        overlayView.gpuMode = true  // GPU 모드: fit 기반 매핑 (GL 출력과 동일)
    }

    private fun initLensManager() {
        lensManager = LensManager(this)

        // 렌즈 어댑터 설정
        lensAdapter = LensAdapter { lens ->
            onLensSelected(lens)
        }

        rvLenses.apply {
            layoutManager = LinearLayoutManager(this@GpuRenderActivity, LinearLayoutManager.HORIZONTAL, false)
            adapter = lensAdapter
        }

        // 렌즈 목록 로드
        lifecycleScope.launch {
            val lenses = lensManager.loadLensesFromAssets()
            lensAdapter.submitList(lenses)
            Log.d(TAG, "Loaded ${lenses.size} lenses")
        }

        // 렌즈 변경 리스너
        lensManager.onLensChangedListener = { lens ->
            runOnUiThread {
                tvLensStatus.text = "Lens: ${lens?.name ?: "없음"}"
            }
        }
    }

    private fun onLensSelected(lens: LensData) {
        lensManager.selectLens(lens)

        if (lens.id == NoLens.ID) {
            // 렌즈 제거
            cameraGLView.setLensTexture(null)
            cameraGLView.setLensEnabled(false)
            Log.d(TAG, "Lens cleared")
        } else {
            // 렌즈 적용
            val texture = lensManager.getTexture(lens)
            if (texture != null) {
                cameraGLView.setLensTexture(texture)
                cameraGLView.setLensConfig(lensConfig)
                cameraGLView.setLensEnabled(true)
                Log.d(TAG, "Lens applied: ${lens.name}")
            } else {
                Log.e(TAG, "Failed to load lens texture: ${lens.name}")
            }
        }
    }

    private fun setupLensControls() {
        // 초기 렌즈 설정
        lensConfig.setDefaults()

        // Opacity 슬라이더 (0~100 → 0.0~1.0)
        seekLensOpacity.progress = (lensConfig.opacity * 100).toInt()
        seekLensOpacity.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                lensConfig.opacity = progress / 100f
                cameraGLView.setLensConfig(lensConfig)
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        // Scale 슬라이더 (0~100 → 0.8~1.8, progress 50 = scale 1.3)
        seekLensScale.progress = ((lensConfig.scale - 0.8f) / 1.0f * 100).toInt().coerceIn(0, 100)
        seekLensScale.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                lensConfig.scale = 0.8f + progress / 100f * 1.0f  // 0.8 ~ 1.8
                cameraGLView.setLensConfig(lensConfig)
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        // Edge Feather 슬라이더 (0~100 → 0.0~1.0)
        seekLensFeather.progress = (lensConfig.edgeFeather * 100).toInt()
        seekLensFeather.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                lensConfig.edgeFeather = progress / 100f
                cameraGLView.setLensConfig(lensConfig)
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        // 블렌드 모드 선택
        rgBlendMode.setOnCheckedChangeListener { _, checkedId ->
            lensConfig.blendMode = when (checkedId) {
                R.id.rbBlendNormal -> LensConfig.BLEND_NORMAL
                R.id.rbBlendMultiply -> LensConfig.BLEND_MULTIPLY
                R.id.rbBlendScreen -> LensConfig.BLEND_SCREEN
                R.id.rbBlendOverlay -> LensConfig.BLEND_OVERLAY
                else -> LensConfig.BLEND_NORMAL
            }
            cameraGLView.setLensConfig(lensConfig)
            Log.d(TAG, "Blend mode changed: ${lensConfig.getBlendModeName()}")
        }
    }

    private fun setupBeautyControls() {
        // 뷰티 토글
        btnToggleBeauty.setOnClickListener {
            beautyEnabled = !beautyEnabled
            beautyConfig.enabled = beautyEnabled
            cameraGLView.setBeautyEnabled(beautyEnabled)
            btnToggleBeauty.text = if (beautyEnabled) "Beauty: ON" else "Beauty: OFF"
        }

        // 프리셋 버튼 리스너
        btnPresetNatural.setOnClickListener { applyPreset(BeautyPreset.NATURAL) }
        btnPresetStudio.setOnClickListener { applyPreset(BeautyPreset.STUDIO) }
        btnPresetGlamour.setOnClickListener { applyPreset(BeautyPreset.GLAMOUR) }
        btnPresetCustom.setOnClickListener { applyPreset(BeautyPreset.CUSTOM) }

        // Smoothing 슬라이더
        seekSmoothing.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                beautyConfig.smoothing = progress / 100f
                if (fromUser) onSliderManualChange()
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
                if (fromUser) onSliderManualChange()
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
                if (fromUser) onSliderManualChange()
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
                if (fromUser) onSliderManualChange()
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
                if (fromUser) onSliderManualChange()
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
        updatePresetButtonHighlight()
    }

    /**
     * 프리셋을 적용하고 슬라이더 값을 동기화합니다.
     */
    private fun applyPreset(preset: BeautyPreset) {
        currentPreset = preset
        beautyConfig = BeautyPresetFactory.createPreset(preset, beautyConfig)
        BeautyPresetFactory.sanitizeConfig(beautyConfig)

        // 뷰티 활성화
        beautyEnabled = true
        beautyConfig.enabled = true
        cameraGLView.setBeautyEnabled(true)
        btnToggleBeauty.text = "Beauty: ON"

        // 슬라이더 동기화
        syncSlidersToConfig()

        // GPU에 설정 전달
        cameraGLView.setBeautyConfig(beautyConfig)

        // 버튼 하이라이트 업데이트
        updatePresetButtonHighlight()

        Log.d(TAG, "Preset applied: ${preset.label}, config=$beautyConfig")
    }

    /**
     * beautyConfig 값에 맞게 슬라이더 위치를 업데이트합니다.
     */
    private fun syncSlidersToConfig() {
        isUpdatingSliders = true

        seekSmoothing.progress = (beautyConfig.smoothing * 100).toInt()
        // brightness: 0.5~1.5 → 0~100
        seekBrightness.progress = ((beautyConfig.brightness - 0.5f) * 100).toInt()
        seekWhitening.progress = (beautyConfig.whitening * 100).toInt()
        // colorBalance: -1.0~1.0 → 0~100 (50 = 중립)
        seekColorBalance.progress = ((beautyConfig.colorBalance * 50) + 50).toInt()
        seekSoftFocus.progress = (beautyConfig.softFocus * 100).toInt()

        isUpdatingSliders = false
    }

    /**
     * 슬라이더를 수동 조작하면 Custom 프리셋으로 전환합니다.
     */
    private fun onSliderManualChange() {
        if (!isUpdatingSliders && currentPreset != BeautyPreset.CUSTOM) {
            currentPreset = BeautyPreset.CUSTOM
            updatePresetButtonHighlight()
        }
    }

    /**
     * 현재 선택된 프리셋 버튼의 시각적 하이라이트를 업데이트합니다.
     */
    private fun updatePresetButtonHighlight() {
        // 모든 프리셋 버튼 알파 조정 (선택 = 1.0, 미선택 = 0.5)
        btnPresetNatural.alpha = if (currentPreset == BeautyPreset.NATURAL) 1.0f else 0.5f
        btnPresetStudio.alpha = if (currentPreset == BeautyPreset.STUDIO) 1.0f else 0.5f
        btnPresetGlamour.alpha = if (currentPreset == BeautyPreset.GLAMOUR) 1.0f else 0.5f
        btnPresetCustom.alpha = if (currentPreset == BeautyPreset.CUSTOM) 1.0f else 0.5f
    }

    /**
     * LUT 필터 컨트롤 설정
     *
     * Identity LUT를 프로그래밍적으로 생성하여 테스트합니다.
     * Identity LUT는 색상을 변경하지 않으므로 on/off 차이가 없어야 합니다.
     * 실제 컬러 그레이딩 LUT PNG를 assets/luts/에 추가하면 효과가 나타납니다.
     */
    private fun setupLutControls() {
        btnToggleLut.setOnClickListener {
            lutEnabled = !lutEnabled

            if (lutEnabled) {
                // GL 스레드에서 Identity LUT 3D 텍스처 생성 및 적용
                cameraGLView.queueEvent {
                    val identityBitmap = LutTextureLoader.generateIdentityLutBitmap()
                    val textureId = LutTextureLoader.createLut3dTexture(identityBitmap)
                    identityBitmap.recycle()

                    if (textureId != 0) {
                        cameraGLView.setLut3dTexture(textureId)
                        cameraGLView.setLutEnabled(true)
                        Log.d(TAG, "LUT filter enabled with identity LUT (textureId=$textureId)")
                    } else {
                        Log.e(TAG, "Failed to create identity LUT texture")
                    }
                }

                lutIntensityPanel.visibility = View.VISIBLE
                btnToggleLut.text = "LUT Filter: ON (Identity)"
            } else {
                cameraGLView.setLutEnabled(false)
                lutIntensityPanel.visibility = View.GONE
                btnToggleLut.text = "LUT Filter: OFF"
            }
        }

        seekLutIntensity.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                cameraGLView.setLutIntensity(progress / 100f)
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
    }

    private fun setupDebugControls() {
        btnToggleMesh = findViewById(R.id.btnToggleMesh)
        btnToggleDebug = findViewById(R.id.btnToggleDebug)
        btnToggleIris = findViewById(R.id.btnToggleIris)

        // 버튼 상태 업데이트 헬퍼
        fun updateButtonColors() {
            btnToggleMesh.setTextColor(
                if (overlayView.showFaceMesh) 0xFF00FF00.toInt() else 0xFFAAAAAA.toInt()
            )
            btnToggleDebug.setTextColor(
                if (overlayView.debugMode) 0xFF00FF00.toInt() else 0xFFAAAAAA.toInt()
            )
            btnToggleIris.setTextColor(
                if (overlayView.showFaceRect) 0xFF00FF00.toInt() else 0xFFAAAAAA.toInt()
            )
        }

        btnToggleMesh.setOnClickListener {
            overlayView.showFaceMesh = !overlayView.showFaceMesh
            updateButtonColors()
            Toast.makeText(this, "Face Mesh: ${if (overlayView.showFaceMesh) "ON" else "OFF"}", Toast.LENGTH_SHORT).show()
        }

        btnToggleDebug.setOnClickListener {
            overlayView.debugMode = !overlayView.debugMode
            updateButtonColors()
            Toast.makeText(this, "Debug: ${if (overlayView.debugMode) "ON" else "OFF"}", Toast.LENGTH_SHORT).show()
        }

        btnToggleIris.setOnClickListener {
            overlayView.showFaceRect = !overlayView.showFaceRect
            updateButtonColors()
            Toast.makeText(this, "Face Rect: ${if (overlayView.showFaceRect) "ON" else "OFF"}", Toast.LENGTH_SHORT).show()
        }

        updateButtonColors()
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

    /**
     * 디바이스 성능에 따라 최적 카메라 해상도를 결정합니다.
     *
     * - 저사양 (RAM <= 3GB): 480x640
     * - 중간 사양 (RAM <= 6GB): 720x1280
     * - 고사양 (RAM > 6GB): 1080x1920
     */
    private fun selectOptimalResolution(): Size {
        val activityManager = getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)
        val totalRamMb = memInfo.totalMem / (1024 * 1024)

        val resolution = when {
            totalRamMb <= 3072 -> Size(480, 640)
            totalRamMb <= 6144 -> Size(720, 1280)
            else -> Size(1080, 1920)
        }

        Log.d(TAG, "Device RAM: ${totalRamMb}MB -> resolution: ${resolution.width}x${resolution.height}")
        return resolution
    }

    private fun bindCameraUseCases() {
        val cameraProvider = cameraProvider ?: return

        // 카메라 선택 (전면)
        val cameraSelector = CameraSelector.Builder()
            .requireLensFacing(lensFacing)
            .build()

        // 디바이스 성능 기반 해상도 선택
        val targetResolution = selectOptimalResolution()
        val resolutionSelector = ResolutionSelector.Builder()
            .setResolutionStrategy(
                ResolutionStrategy(
                    targetResolution,
                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER
                )
            )
            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
            .build()

        // Preview → GLSurfaceView
        val preview = Preview.Builder()
            .setResolutionSelector(resolutionSelector)
            .build()
            .apply {
                setSurfaceProvider(cameraGLView.getSurfaceProvider())
            }

        // ImageAnalysis (MediaPipe 추론용)
        val imageAnalysis = ImageAnalysis.Builder()
            .setResolutionSelector(resolutionSelector)
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

            // GPU 렌더러에 검출 결과 전달 (깊은 복사 스냅샷)
            if (detectResult == IrisLensSDK.OK && irisResult.detected) {
                glIrisResult.copyFrom(irisResult)
                cameraGLView.setIrisResult(glIrisResult)

                // Detection Slot 업데이트 (lock-free → GL 스레드에서 읽음)
                IrisLensSDK.updateDetectionSlot(irisResult)
            }

            // OverlayView에도 검출 결과 전달 (디버그 시각화용, 별도 스냅샷)
            // SDK는 회전 후 좌표를 반환하므로 회전 후 프레임 크기를 전달해야 함
            // (imageProxy.width/height는 회전 전 센서 크기 → 매쉬가 늘어나는 원인)
            val isRotated = (rotation == 90 || rotation == 270)
            val overlayFrameW = if (irisResult.frameWidth > 0) {
                irisResult.frameWidth
            } else {
                if (isRotated) imageProxy.height else imageProxy.width
            }
            val overlayFrameH = if (irisResult.frameHeight > 0) {
                irisResult.frameHeight
            } else {
                if (isRotated) imageProxy.width else imageProxy.height
            }
            uiIrisResult.copyFrom(irisResult)
            runOnUiThread {
                overlayView.setIrisResult(
                    uiIrisResult,
                    overlayFrameW,
                    overlayFrameH,
                    lensFacing == CameraSelector.LENS_FACING_FRONT
                )
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
                tvFps.text = "Detect FPS: $fps"
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
        lensManager.release()
        analysisExecutor.shutdown()
        IrisLensSDK.releaseDetectionSlot()
        IrisLensSDK.releaseGpuBeauty()
    }
}
