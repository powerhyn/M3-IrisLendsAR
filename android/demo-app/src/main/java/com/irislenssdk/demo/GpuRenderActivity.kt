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
import android.opengl.GLES31
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.Spinner
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
import com.irislenssdk.demo.camera.OverlayView
import com.irislenssdk.demo.camera.gpu.CameraGLView
import com.irislenssdk.demo.lens.LensAdapter
import com.irislenssdk.demo.lens.LensData
import com.irislenssdk.demo.lens.LensManager
import com.irislenssdk.demo.lens.NoLens
import com.irislenssdk.demo.util.StabilityLogger
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
    private lateinit var lensTabContent: ScrollView
    private lateinit var beautyTabContent: ScrollView

    // 디버그 버튼
    private lateinit var btnToggleMesh: Button
    private lateinit var btnToggleDebug: Button
    private lateinit var btnToggleIris: Button
    private lateinit var btnToggleLog: Button

    // 렌즈 탭 UI
    private lateinit var rvLenses: RecyclerView
    private lateinit var seekLensOpacity: SeekBar
    private lateinit var seekLensScale: SeekBar
    private lateinit var seekLensFeather: SeekBar
    private lateinit var spinnerBlendMode: Spinner
    private lateinit var btnToggleSclera: Button
    private lateinit var btnToggleShadow: Button
    private lateinit var btnToggleEllipse: Button
    private lateinit var seekMaxDetail: SeekBar
    private lateinit var tvMaxDetailValue: TextView

    // 뷰티 탭 UI
    private lateinit var btnToggleBeauty: Button
    private lateinit var btnPresetNaturalGlow: Button
    private lateinit var btnPresetSpring: Button
    private lateinit var btnPresetStudio: Button
    private lateinit var btnPresetGoldenHour: Button
    private lateinit var btnPresetVividPop: Button
    private lateinit var btnPresetCustom: Button
    private lateinit var seekSkinQuality: SeekBar
    private lateinit var seekSmoothIntensity: SeekBar
    private lateinit var seekPoreReduction: SeekBar
    private lateinit var btnProtectNose: Button
    private lateinit var seekVividIntensity: SeekBar
    private lateinit var seekVividSaturation: SeekBar
    private lateinit var seekVividBrightness: SeekBar
    private lateinit var seekVividWarmth: SeekBar

    // 카메라
    private var cameraProvider: ProcessCameraProvider? = null
    private var lensFacing = CameraSelector.LENS_FACING_FRONT
    private val analysisExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    // 렌즈 관리
    private lateinit var lensManager: LensManager
    private lateinit var lensAdapter: LensAdapter
    private var lensConfig = LensConfig()

    // 뷰티 설정
    private var beautyConfig = BeautyPresetFactory.createCustomPreset()
    private var beautyEnabled = true
    private var currentPreset = BeautyPreset.CUSTOM
    private var isUpdatingSliders = false

    // 홍채 검출 (스레드별 불변 스냅샷 사용)
    private val irisResult = IrisResult()       // Analyzer 스레드 전용 (JNI 결과 수신)
    private val glIrisResult = IrisResult()     // GL 스레드 전달용 스냅샷
    private val uiIrisResult = IrisResult()     // UI 스레드 전달용 스냅샷

    // NV21 버퍼 (재사용)
    private var nv21Buffer: ByteArray? = null

    // 카메라 회전 (한 번만 설정)
    private var lastRotation: Int = -1

    // StabilityLogger (P4-W1-02)
    private var stabilityLogger: StabilityLogger? = null
    @Volatile private var gpuTier: String = "UNKNOWN"
    @Volatile private var gpuRendererName: String = ""

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
        spinnerBlendMode = findViewById(R.id.spinnerBlendMode)
        btnToggleSclera = findViewById(R.id.btnToggleSclera)
        btnToggleShadow = findViewById(R.id.btnToggleShadow)
        btnToggleEllipse = findViewById(R.id.btnToggleEllipse)
        seekMaxDetail = findViewById(R.id.seekMaxDetail)
        tvMaxDetailValue = findViewById(R.id.tvMaxDetailValue)

        // 뷰티 탭 UI
        btnToggleBeauty = findViewById(R.id.btnToggleBeauty)
        btnPresetNaturalGlow = findViewById(R.id.btnPresetNaturalGlow)
        btnPresetSpring = findViewById(R.id.btnPresetSpring)
        btnPresetStudio = findViewById(R.id.btnPresetStudio)
        btnPresetGoldenHour = findViewById(R.id.btnPresetGoldenHour)
        btnPresetVividPop = findViewById(R.id.btnPresetVividPop)
        btnPresetCustom = findViewById(R.id.btnPresetCustom)
        seekSkinQuality = findViewById(R.id.seekSkinQuality)
        seekSmoothIntensity = findViewById(R.id.seekSmoothIntensity)
        seekPoreReduction = findViewById(R.id.seekPoreReduction)
        btnProtectNose = findViewById(R.id.btnProtectNose)
        seekVividIntensity = findViewById(R.id.seekVividIntensity)
        seekVividSaturation = findViewById(R.id.seekVividSaturation)
        seekVividBrightness = findViewById(R.id.seekVividBrightness)
        seekVividWarmth = findViewById(R.id.seekVividWarmth)

        // GPU 초기화 콜백 설정
        cameraGLView.onGpuInitialized = { success ->
            // GPU tier 판별 (GL 컨텍스트 활성 상태)
            detectGpuTier()
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
        setupDebugControls()

        // OverlayView 초기 설정: 렌즈는 GPU에서 렌더링하므로 OverlayView에서는 비활성화
        overlayView.showLens = false
        overlayView.showFaceMesh = false
        overlayView.debugMode = false
        overlayView.showFaceRect = false
        overlayView.screenMappingMode = OverlayView.ScreenMappingMode.COVER  // GL Cover 출력과 동일한 매핑
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

        // 블렌드 모드 선택 (Mode 5 비노출: ISS-005 B-2 존치 판정)
        val blendModeEntries = arrayOf(
            "Normal" to 0, "Multiply" to 1, "Screen" to 2, "Overlay" to 3,
            "Luminance Tint" to 4, "Soft Light" to 6, "Color Replace" to 7
        )
        spinnerBlendMode.adapter = ArrayAdapter(
            this, android.R.layout.simple_spinner_dropdown_item, blendModeEntries.map { it.first }.toTypedArray()
        )
        spinnerBlendMode.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                lensConfig.blendMode = blendModeEntries[position].second
                cameraGLView.setLensConfig(lensConfig)
                Log.d(TAG, "Blend mode changed: ${lensConfig.getBlendModeName()}")
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        // Sclera Protection 토글 (P4-W2-01, 기본 ON)
        var scleraOn = true
        btnToggleSclera.setOnClickListener {
            scleraOn = !scleraOn
            cameraGLView.setScleraProtect(scleraOn)
            btnToggleSclera.text = if (scleraOn) "Sclera: ON" else "Sclera: OFF"
            btnToggleSclera.setBackgroundColor(if (scleraOn) 0x4400CC00.toInt() else 0x44FF0000.toInt())
        }

        // Contact Shadow 토글 (P4-W2-01, 기본 OFF)
        var shadowOn = false
        btnToggleShadow.setOnClickListener {
            shadowOn = !shadowOn
            cameraGLView.setContactShadow(shadowOn)
            btnToggleShadow.text = if (shadowOn) "Shadow: ON" else "Shadow: OFF"
            btnToggleShadow.setBackgroundColor(if (shadowOn) 0x4400CC00.toInt() else 0x44FF0000.toInt())
        }

        // 비대칭 타원 Eye Mask 토글 (P4-W2-02, 기본 OFF)
        var ellipseOn = false
        btnToggleEllipse.setOnClickListener {
            ellipseOn = !ellipseOn
            cameraGLView.setEllipseMask(ellipseOn)
            btnToggleEllipse.text = if (ellipseOn) "Ellipse: ON" else "Ellipse: OFF"
            btnToggleEllipse.setBackgroundColor(if (ellipseOn) 0x4400CC00.toInt() else 0x44FF0000.toInt())
        }

        // 홍채 밝기 보정 슬라이더 (P4-W2-01, 0.8~1.4 / 0.1 스텝 / 기본 1.2)
        seekMaxDetail.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                val value = 0.8f + progress * 0.1f
                tvMaxDetailValue.text = String.format("%.1f", value)
                cameraGLView.setMaxDetail(value)
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
    }

    private fun setupBeautyControls() {
        // 뷰티 토글
        btnToggleBeauty.setOnClickListener {
            beautyEnabled = !beautyEnabled
            beautyConfig.enabled = beautyEnabled
            cameraGLView.setBeautyEnabled(beautyEnabled)
            btnToggleBeauty.text = if (beautyEnabled) "Beauty: ON" else "Beauty: OFF"
        }

        // Vivid 프리셋 버튼 리스너
        btnPresetNaturalGlow.setOnClickListener { applyPreset(BeautyPreset.NATURAL_GLOW) }
        btnPresetSpring.setOnClickListener { applyPreset(BeautyPreset.SPRING) }
        btnPresetStudio.setOnClickListener { applyPreset(BeautyPreset.STUDIO) }
        btnPresetGoldenHour.setOnClickListener { applyPreset(BeautyPreset.GOLDEN_HOUR) }
        btnPresetVividPop.setOnClickListener { applyPreset(BeautyPreset.VIVID_POP) }
        btnPresetCustom.setOnClickListener { applyPreset(BeautyPreset.CUSTOM) }

        // SkinQuality (잡티 보정) 슬라이더
        seekSkinQuality.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                beautyConfig.skinQuality = progress / 100f
                // 레거시 모드로 전환: 2축 값을 클리어하여 stale 값 방지
                beautyConfig.smoothIntensity = 0f
                beautyConfig.poreReduction = 0f
                if (fromUser) onSliderManualChange()
                cameraGLView.setBeautyConfig(beautyConfig)
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        // SmoothIntensity (매끈하게) 슬라이더
        seekSmoothIntensity.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                beautyConfig.smoothIntensity = progress / 100f
                // 2축 모드 진입 시 레거시 skinQuality 클리어
                if (progress > 0) beautyConfig.skinQuality = 0f
                if (fromUser) onSliderManualChange()
                cameraGLView.setBeautyConfig(beautyConfig)
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        // PoreReduction (모공) 슬라이더
        seekPoreReduction.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                beautyConfig.poreReduction = progress / 100f
                // 2축 모드 진입 시 레거시 skinQuality 클리어
                if (progress > 0) beautyConfig.skinQuality = 0f
                if (fromUser) onSliderManualChange()
                cameraGLView.setBeautyConfig(beautyConfig)
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        // 코 보호 토글
        btnProtectNose.setOnClickListener {
            beautyConfig.protectNose = !beautyConfig.protectNose
            btnProtectNose.text = if (beautyConfig.protectNose) "코보호: ON" else "코보호: OFF"
            btnProtectNose.setTextColor(if (beautyConfig.protectNose) 0xFF00FF00.toInt() else 0xFFAAAAAA.toInt())
            cameraGLView.setBeautyConfig(beautyConfig)
        }

        // Vivid Intensity (0-100 → 0.0-1.0)
        seekVividIntensity.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                beautyConfig.vividIntensity = progress / 100f
                if (fromUser) onSliderManualChange()
                cameraGLView.setBeautyConfig(beautyConfig)
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        // Vivid Saturation (0-100 → 0.0-1.0)
        seekVividSaturation.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                beautyConfig.vividSaturation = progress / 100f
                if (fromUser) onSliderManualChange()
                cameraGLView.setBeautyConfig(beautyConfig)
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        // Vivid Brightness (0-100 → 0.0-0.5)
        seekVividBrightness.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                beautyConfig.vividBrightness = progress / 200f
                if (fromUser) onSliderManualChange()
                cameraGLView.setBeautyConfig(beautyConfig)
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        // Vivid Warmth (0-100 → 0.0-1.0)
        seekVividWarmth.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                beautyConfig.vividWarmth = progress / 100f
                if (fromUser) onSliderManualChange()
                cameraGLView.setBeautyConfig(beautyConfig)
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        // 초기 프리셋 적용 (Custom = 잡티보정만, vivid OFF)
        syncSlidersToConfig()

        btnToggleBeauty.text = if (beautyEnabled) "Beauty: ON" else "Beauty: OFF"
        updatePresetButtonHighlight()
    }

    private fun applyPreset(preset: BeautyPreset) {
        currentPreset = preset
        beautyConfig = BeautyPresetFactory.createPreset(preset, beautyConfig)
        BeautyPresetFactory.sanitizeConfig(beautyConfig)

        beautyEnabled = true
        beautyConfig.enabled = true
        cameraGLView.setBeautyEnabled(true)
        btnToggleBeauty.text = "Beauty: ON"

        syncSlidersToConfig()
        cameraGLView.setBeautyConfig(beautyConfig)
        updatePresetButtonHighlight()

        Log.d(TAG, "Preset applied: ${preset.label}, config=$beautyConfig")
    }

    private fun syncSlidersToConfig() {
        isUpdatingSliders = true

        seekSkinQuality.progress = (beautyConfig.skinQuality * 100).toInt()
        seekSmoothIntensity.progress = (beautyConfig.smoothIntensity * 100).toInt()
        seekPoreReduction.progress = (beautyConfig.poreReduction * 100).toInt()
        seekVividIntensity.progress = (beautyConfig.vividIntensity * 100).toInt()
        seekVividSaturation.progress = (beautyConfig.vividSaturation * 100).toInt()
        seekVividBrightness.progress = (beautyConfig.vividBrightness * 200).toInt()
        seekVividWarmth.progress = (beautyConfig.vividWarmth * 100).toInt()

        isUpdatingSliders = false
    }

    private fun onSliderManualChange() {
        if (!isUpdatingSliders && currentPreset != BeautyPreset.CUSTOM) {
            currentPreset = BeautyPreset.CUSTOM
            updatePresetButtonHighlight()
        }
    }

    private fun updatePresetButtonHighlight() {
        val buttons = mapOf(
            BeautyPreset.NATURAL_GLOW to btnPresetNaturalGlow,
            BeautyPreset.SPRING to btnPresetSpring,
            BeautyPreset.STUDIO to btnPresetStudio,
            BeautyPreset.GOLDEN_HOUR to btnPresetGoldenHour,
            BeautyPreset.VIVID_POP to btnPresetVividPop,
            BeautyPreset.CUSTOM to btnPresetCustom
        )
        buttons.forEach { (preset, btn) ->
            btn.alpha = if (currentPreset == preset) 1.0f else 0.5f
        }
    }
    private fun setupDebugControls() {
        btnToggleMesh = findViewById(R.id.btnToggleMesh)
        btnToggleDebug = findViewById(R.id.btnToggleDebug)
        btnToggleIris = findViewById(R.id.btnToggleIris)
        btnToggleLog = findViewById(R.id.btnToggleLog)

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
            btnToggleLog.setTextColor(
                if (stabilityLogger?.isActive == true) 0xFFFF4444.toInt() else 0xFFAAAAAA.toInt()
            )
        }

        btnToggleMesh.setOnClickListener {
            overlayView.showFaceMesh = !overlayView.showFaceMesh
            updateButtonColors()
            Toast.makeText(this, "Face Mesh: ${if (overlayView.showFaceMesh) "ON" else "OFF"}", Toast.LENGTH_SHORT).show()
        }

        var freqSepDebugMode = 0
        val debugModeNames = arrayOf("OFF", "Magnitude", "MicroBand", "EdgeProt", "EffectStr", "Compression×3", "Mask")
        btnToggleDebug.setOnClickListener {
            freqSepDebugMode = (freqSepDebugMode + 1) % 7
            cameraGLView.queueEvent {
                com.irislenssdk.IrisLensSDK.setFreqSepDebugMode(freqSepDebugMode)
            }
            overlayView.debugMode = freqSepDebugMode > 0
            updateButtonColors()
            Toast.makeText(this, "FreqSep Debug: ${debugModeNames[freqSepDebugMode]}", Toast.LENGTH_SHORT).show()
        }

        btnToggleIris.setOnClickListener {
            overlayView.showFaceRect = !overlayView.showFaceRect
            updateButtonColors()
            Toast.makeText(this, "Face Rect: ${if (overlayView.showFaceRect) "ON" else "OFF"}", Toast.LENGTH_SHORT).show()
        }

        // StabilityLogger 토글 (P4-W1-02)
        btnToggleLog.setOnClickListener {
            val logger = stabilityLogger
            if (logger == null || !logger.isActive) {
                startStabilityLog()
            } else {
                stopStabilityLog()
            }
            updateButtonColors()
        }

        updateButtonColors()
    }

    // === StabilityLogger (P4-W1-02) ===

    private fun detectGpuTier() {
        cameraGLView.queueEvent {
            gpuRendererName = GLES31.glGetString(GLES31.GL_RENDERER) ?: ""
            // "(TM)" 등 상표 표기 제거 후 매칭 (e.g. "Adreno (TM) 740" → "Adreno 740")
            val normalized = gpuRendererName.replace(Regex("\\s*\\(TM\\)\\s*", RegexOption.IGNORE_CASE), " ").trim()
            gpuTier = when {
                normalized.contains("Adreno 7", ignoreCase = true) -> "HIGH"
                normalized.contains("Adreno 6", ignoreCase = true) -> "MID"
                normalized.contains("Mali-G7", ignoreCase = true) -> "MID"
                normalized.contains("Mali-G5", ignoreCase = true) -> "LOW"
                else -> "MID"
            }
            Log.d(TAG, "GPU tier: $gpuTier ($gpuRendererName)")
        }
    }

    private fun startStabilityLog() {
        val logger = StabilityLogger(applicationContext, gpuTier, gpuRendererName)
        val path = logger.start()
        if (path != null) {
            stabilityLogger = logger

            // M-1 fix: queueEvent로 GL 스레드에서 콜백 연결
            cameraGLView.setStabilityFrameCallback { faceDetected,
                rawLCx, rawLCy, rawLR, fltLCx, fltLCy, fltLR,
                rawRCx, rawRCy, rawRR, fltRCx, fltRCy, fltRR,
                eLt, eLb, eRt, eRb,
                holdActive, holdRemaining, renderTimeUs ->
                logger.logFrame(
                    faceDetected,
                    rawLCx, rawLCy, rawLR, fltLCx, fltLCy, fltLR,
                    rawRCx, rawRCy, rawRR, fltRCx, fltRCy, fltRR,
                    eLt, eLb, eRt, eRb,
                    holdActive, holdRemaining, renderTimeUs
                )
            }
            cameraGLView.setStabilityLogEnabled(true)

            Toast.makeText(this, "Logging started", Toast.LENGTH_SHORT).show()
            Log.d(TAG, "StabilityLog started: $path")
        } else {
            Toast.makeText(this, "Log start failed", Toast.LENGTH_SHORT).show()
        }
    }

    private fun stopStabilityLog() {
        // M-2 fix: GL 스레드에서 먼저 비활성화 후 콜백 해제, 그 다음 writer 정리
        cameraGLView.setStabilityLogEnabled(false)
        cameraGLView.setStabilityFrameCallback(null)

        val frames = stabilityLogger?.stop() ?: 0
        val filePath = stabilityLogger?.getCurrentFilePath()

        Toast.makeText(this, "Logged $frames frames → ${filePath?.substringAfterLast('/')}", Toast.LENGTH_LONG).show()
        Log.d(TAG, "StabilityLog stopped: $frames frames → $filePath")
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
        beautyConfig.smoothing = 0.0f
        beautyConfig.brightness = 1.0f
        beautyConfig.whitening = 0.0f
        beautyConfig.colorBalance = 0.0f
        beautyConfig.softFocus = 0.0f
        beautyConfig.roiOnly = true  // ROI 전용 모드: 얼굴 피부에만 보정 적용

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

            // GPU 렌더러에 검출 결과 전달 (매 프레임, 미검출 포함)
            // detected=false 프레임도 전달하여 stale 스냅샷 방지
            // → GL 쪽에서 렌즈 페이드아웃/클리핑 폴백 정책 적용 가능
            glIrisResult.copyFrom(irisResult)
            cameraGLView.setIrisResult(glIrisResult)

            // P4-W1-03: 홍채 밝기 샘플링 → EMA (Luminance Tint 블렌드용)
            val rawLum = sampleIrisLuminanceNv21(
                nv21, imageProxy.width, imageProxy.height, irisResult, rotation
            )
            cameraGLView.setRawIrisLuminance(rawLum)

            // Detection Slot 업데이트 (lock-free → GL 스레드에서 읽음)
            if (detectResult == IrisLensSDK.OK) {
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

    /**
     * NV21 Y채널에서 홍채 영역 평균 밝기 샘플링
     *
     * 각 눈을 개별 샘플링 후 밝기값을 평균합니다.
     * 좌표를 평균하면 두 눈 사이(피부/배경)를 샘플링하게 되므로,
     * 반드시 개별 샘플링 → 값 평균 순서를 따릅니다.
     *
     * @return 0.0~1.0 밝기 (미검출 시 -1f)
     */
    private fun sampleIrisLuminanceNv21(
        nv21: ByteArray, sensorW: Int, sensorH: Int,
        result: IrisResult, rotation: Int
    ): Float {
        if (!result.detected || result.frameWidth <= 0 || result.frameHeight <= 0) return -1f

        val leftLum = if (result.leftDetected) {
            samplePointLuminanceNv21(nv21, sensorW, sensorH, result.leftIrisX, result.leftIrisY, rotation)
        } else -1f

        val rightLum = if (result.rightDetected) {
            samplePointLuminanceNv21(nv21, sensorW, sensorH, result.rightIrisX, result.rightIrisY, rotation)
        } else -1f

        return when {
            leftLum >= 0f && rightLum >= 0f -> (leftLum + rightLum) / 2f
            leftLum >= 0f -> leftLum
            rightLum >= 0f -> rightLum
            else -> -1f
        }
    }

    /**
     * NV21 Y채널에서 단일 홍채 중심의 5점 크로스 샘플링
     */
    private fun samplePointLuminanceNv21(
        nv21: ByteArray, sensorW: Int, sensorH: Int,
        nx: Float, ny: Float, rotation: Int
    ): Float {
        // 검출 좌표(회전 후) → 센서 좌표(회전 전) 역변환
        val (sx, sy) = when (rotation) {
            90 -> Pair(ny, 1f - nx)
            180 -> Pair(1f - nx, 1f - ny)
            270 -> Pair(1f - ny, nx)
            else -> Pair(nx, ny)
        }

        val cx = (sx * sensorW).toInt().coerceIn(0, sensorW - 1)
        val cy = (sy * sensorH).toInt().coerceIn(0, sensorH - 1)

        // Y채널 5점 크로스 샘플링 (center + 4방향)
        val r = 3.coerceAtMost(minOf(cx, cy, sensorW - 1 - cx, sensorH - 1 - cy))
        val offsets = intArrayOf(0, 0, 0, -r, 0, r, -r, 0, r, 0) // (dx,dy) 쌍
        var sum = 0
        for (i in offsets.indices step 2) {
            val px = cx + offsets[i]
            val py = cy + offsets[i + 1]
            sum += (nv21[py * sensorW + px].toInt() and 0xFF)
        }
        return (sum / 5f) / 255f
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
        cameraGLView.resetTemporalState()  // P4-W1-03: resume jump 방지
    }

    override fun onPause() {
        super.onPause()
        cameraGLView.onPause()
    }

    override fun onDestroy() {
        // StabilityLogger 정리 (H-1: 파일 누수 + 데이터 손실 방지)
        if (stabilityLogger?.isActive == true) {
            stopStabilityLog()
        }
        super.onDestroy()
        cameraGLView.release()
        lensManager.release()
        analysisExecutor.shutdown()
        IrisLensSDK.releaseDetectionSlot()
        IrisLensSDK.releaseGpuBeauty()
    }
}
