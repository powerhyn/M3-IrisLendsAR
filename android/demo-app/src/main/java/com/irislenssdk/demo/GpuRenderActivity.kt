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
import android.graphics.BitmapFactory
import android.opengl.GLES31
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.KeyEvent
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
    private lateinit var btnToggleHighlight: Button

    // P6-W5 §5.9: B1/B8 4조합 블라인드 벤치 토글 (A/B/C/D)
    private lateinit var btnBenchA: Button
    private lateinit var btnBenchB: Button
    private lateinit var btnBenchC: Button
    private lateinit var btnBenchD: Button
    // P6-W6: 블링크 ramp(B5) / 저조도 gate(B9) / 디테일 재주입(C10) 벤치 토글
    private lateinit var btnW6Blink: Button
    private lateinit var btnW6Gate: Button
    private lateinit var btnW6Detail: Button
    private lateinit var btnW7Measured: Button   // P7-W2: avg_iris_luma fallback↔실측 A/B
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

    // Temporal Stabilizer (SDK 코어 스무딩)
    private var stabilizerHandle: Long = 0

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
        btnToggleHighlight = findViewById(R.id.btnToggleHighlight)

        // P6-W5 §5.9: 4조합 블라인드 벤치 버튼 (A/B/C/D)
        btnBenchA = findViewById(R.id.btnBenchA)
        btnBenchB = findViewById(R.id.btnBenchB)
        btnBenchC = findViewById(R.id.btnBenchC)
        btnBenchD = findViewById(R.id.btnBenchD)
        btnW6Blink = findViewById(R.id.btnW6Blink)
        btnW6Gate = findViewById(R.id.btnW6Gate)
        btnW6Detail = findViewById(R.id.btnW6Detail)
        btnW7Measured = findViewById(R.id.btnW7Measured)
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
            // P6-W4 §5.7: GPU lens init 완료 후 env_map 로드 (NOT_INITIALIZED 회피).
            if (success && !envMapLoaded) {
                loadEnvMapAsset()
                envMapLoaded = true
            }
            // P6-W7: GPU lens init 완료 후 lens_meta.json 등록 (즉시 주입 보장).
            if (success && !lensMetaLoaded) {
                loadLensMetadataAsset()
                lensMetaLoaded = true
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
                cameraGLView.setLensTexture(texture, lens.id)  // P6-W7: sku_id 전달
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
        // P6-W1 검증용으로 Mode 5 (Luminance Tint Linear) 임시 노출.
        // W1이 S1에서 비운 uAvgIrisLum 주입을 실측으로 복구했고, LTL이 그 값을
        // 직접 쓰는 유일한 모드라 시각 확인이 여기서만 가능하다. ISS-005 B-2 존치
        // 판정은 W2 블렌드 3종 확정 단계에서 재검토 예정.
        val blendModeEntries = arrayOf(
            "Normal" to 0, "Multiply" to 1, "Screen" to 2, "Overlay" to 3,
            "Luminance Tint" to 4, "Luminance Tint Linear" to 5,
            "Soft Light" to 6, "Color Replace" to 7
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
        // P6-W2 §5.12: SDK LensConfig 기본값(LuminanceTintLinear=5)과 spinner 초기 위치 동기화.
        // 기본값 변경 시 demo가 SDK surface와 일관되게 시작.
        val defaultIdx = blendModeEntries.indexOfFirst { it.second == lensConfig.blendMode }
        if (defaultIdx >= 0) spinnerBlendMode.setSelection(defaultIdx)

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

        // Normal Map 라이팅 토글 (P5-W3-04, 기본 OFF)
        var highlightOn = false
        btnToggleHighlight.setOnClickListener {
            highlightOn = !highlightOn
            cameraGLView.setHighlight(highlightOn)
            btnToggleHighlight.text = if (highlightOn) "3D Light: ON" else "3D Light: OFF"
            btnToggleHighlight.setBackgroundColor(if (highlightOn) 0x4400CC00.toInt() else 0x44FF0000.toInt())
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

        // P6-W5 §5.9: B1/B8 4조합 블라인드 벤치 토글 (A/B/C/D)
        listOf(btnBenchA, btnBenchB, btnBenchC, btnBenchD).forEachIndexed { idx, btn ->
            btn.setOnClickListener { applyBenchCombo(idx) }
        }

        // P6-W6: B5 블링크 up ramp / B9 저조도 gate / C10 디테일 재주입 벤치 토글
        btnW6Blink.setOnClickListener {
            w6BlinkIdx = (w6BlinkIdx + 1) % w6BlinkUpSweep.size
            val ms = w6BlinkUpSweep[w6BlinkIdx]
            cameraGLView.setBlinkUpMs(ms)
            btnW6Blink.text = "up${ms.toInt()}"
            Log.i(TAG, "P6-W6 B5 blink up → ${ms.toInt()}ms")
        }
        btnW6Gate.setOnClickListener {
            w6GateIdx = (w6GateIdx + 1) % w6GateSweep.size
            val t = w6GateSweep[w6GateIdx]
            cameraGLView.setGateThreshold(t)
            btnW6Gate.text = String.format("g%.2f", t)
            Log.i(TAG, "P6-W6 B9 gate → $t")
        }
        btnW6Detail.setOnClickListener {
            w6DetailOn = !w6DetailOn
            cameraGLView.setDetailReinject(w6DetailOn)
            btnW6Detail.text = if (w6DetailOn) "C10:on" else "C10:off"
            Log.i(TAG, "P6-W6 C10 detail → ${if (w6DetailOn) "on" else "off"}")
        }
        // P7-W2 §5.6: avg_iris_luma fallback(lum:fb) ↔ 실측(lum:meas) A/B 토글.
        btnW7Measured.setOnClickListener {
            w7MeasuredOn = !w7MeasuredOn
            cameraGLView.setUseMeasuredLuma(w7MeasuredOn)
            btnW7Measured.text = if (w7MeasuredOn) "lum:meas" else "lum:fb"
            Log.i(TAG, "P7-W2 measured luma → ${if (w7MeasuredOn) "on" else "off"}")
        }
    }

    //=========================================================================
    // P6-W5 §5.9: B1/B8 4조합 블라인드 벤치 (A/B/C/D)
    // 정답표는 코드/로그에만 존재. 평가자에게는 라벨만 노출.
    //=========================================================================

    private data class BenchCombo(val label: String, val blendMode: Int, val vetoMode: Int, val desc: String)

    private val benchCombos = listOf(
        BenchCombo("A", 0, 1, "Normal + color-veto(Codex)"),
        BenchCombo("B", 0, 2, "Normal + luma-only(Gemini)"),
        BenchCombo("C", 7, 1, "CRL + color-veto(Codex)"),
        BenchCombo("D", 7, 2, "CRL + luma-only(Gemini)"),
    )
    private var currentBenchIdx = -1

    // P6-W6 §5.3/§5.7: 벤치 토글 sweep 상태 (기본값=중간값, 코어 기본과 일치).
    private val w6BlinkUpSweep = floatArrayOf(60f, 80f, 120f)
    private var w6BlinkIdx = 1   // 기본 80ms
    private val w6GateSweep = floatArrayOf(0.10f, 0.15f, 0.25f)
    private var w6GateIdx = 0    // 기본 0.10 (저조도 드묾 — C10 디테일 항상 ON)
    private var w6DetailOn = true
    private var w7MeasuredOn = true   // P7-W2 §5.6: 실기기 검증 후 기본 실측 ON (SDK default와 일치). 토글로 fallback 비교.

    private fun applyBenchCombo(idx: Int) {
        val combo = benchCombos[idx]
        lensConfig.blendMode = combo.blendMode
        cameraGLView.setLensConfig(lensConfig)
        cameraGLView.setScleraVetoMode(combo.vetoMode)
        // blendMode 0/7은 spinner index와 1:1 매핑 (Normal=0, ColorReplace=7)
        spinnerBlendMode.setSelection(combo.blendMode)
        currentBenchIdx = idx
        updateBenchButtonHighlight()
        Toast.makeText(this, "Bench ${combo.label}", Toast.LENGTH_SHORT).show()
        Log.i(TAG, "P6-W5 bench → ${combo.label} (${combo.desc})")
    }

    private fun updateBenchButtonHighlight() {
        val buttons = listOf(btnBenchA, btnBenchB, btnBenchC, btnBenchD)
        buttons.forEachIndexed { idx, btn ->
            btn.setBackgroundColor(if (idx == currentBenchIdx) 0xCC2196F3.toInt() else 0x66555555.toInt())
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

        // 피부색 필터 토글 (실험용, backend-local)
        var skinColorFilterEnabled = false
        val btnSkinColorFilter = findViewById<Button>(R.id.btnSkinColorFilter)
        btnSkinColorFilter.setOnClickListener {
            skinColorFilterEnabled = !skinColorFilterEnabled
            cameraGLView.queueEvent {
                com.irislenssdk.IrisLensSDK.setSkinColorFilter(skinColorFilterEnabled)
            }
            btnSkinColorFilter.text = if (skinColorFilterEnabled) "피부색필터: ON" else "피부색필터: OFF"
            btnSkinColorFilter.setTextColor(if (skinColorFilterEnabled) 0xFF00FF00.toInt() else 0xFFAAAAAA.toInt())
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
        val debugModeNames = arrayOf("OFF", "Magnitude", "MicroBand", "EdgeProt", "EffectStr", "Compression×3", "Mask", "SkinColor")
        btnToggleDebug.setOnClickListener {
            freqSepDebugMode = (freqSepDebugMode + 1) % 8
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

            // Temporal Stabilizer 적용 (검출 실패 포함 — hold/fade-out 동작 필요)
            if (detectResult == IrisLensSDK.OK || detectResult == IrisLensSDK.NO_FACE) {
                if (stabilizerHandle == 0L) {
                    stabilizerHandle = IrisLensSDK.createStabilizer()
                }
                if (stabilizerHandle != 0L) {
                    val timestampSec = System.nanoTime() / 1_000_000_000.0
                    IrisLensSDK.stabilize(stabilizerHandle, irisResult, timestampSec)
                }
            }

            // GPU 렌더러에 스무딩된 결과 전달 (매 프레임, 미검출 포함)
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
        // P6-W4 env_map 로드는 onGpuInitialized 콜백에서 처리 (GPU lens init 완료 보장).
    }

    //=========================================================================
    // P6-W4: 환경 반사 벤치 (env_map 로드 + 3 프로토타입 토글)
    //=========================================================================

    private var envMapLoaded = false
    private var lensMetaLoaded = false  // P6-W7: lens_meta.json 1회 등록 가드
    private var reflectionMode = 0  // 0=OFF, 1=EnvMap, 2=Periphery

    // P6-W4 Phase A 보완: intensity sweep (W3 §5.7 기본 0.3, clamp 0~5 확장).
    private val intensitySweep = floatArrayOf(0.3f, 1.0f, 2.0f, 3.0f)
    private var intensitySweepIdx = 0

    /** P6-W4 §5.7: assets/env/env_default_256x128.png 로드 + GL 스레드 디스패치. */
    private fun loadEnvMapAsset() {
        try {
            val bitmap = BitmapFactory.decodeStream(assets.open("env/env_default_256x128.png"))
            val w = bitmap.width
            val h = bitmap.height
            val rgb = ByteArray(w * h * 3)
            var idx = 0
            for (y in 0 until h) {
                for (x in 0 until w) {
                    val pixel = bitmap.getPixel(x, y)
                    rgb[idx++] = ((pixel shr 16) and 0xFF).toByte()
                    rgb[idx++] = ((pixel shr 8) and 0xFF).toByte()
                    rgb[idx++] = (pixel and 0xFF).toByte()
                }
            }
            cameraGLView.setEnvMap(rgb, w, h)
            Log.i(TAG, "P6-W4 env_map asset loaded: ${w}x${h}")
        } catch (e: Exception) {
            Log.e(TAG, "P6-W4 env_map load failed: ${e.message}")
        }
    }

    /**
     * P6-W7: assets/lens_meta.json 로드 + 코어 등록.
     *
     * 림발 등 SKU별 렌즈 메타를 코어에 1회 등록한다. assets가 없거나
     * 등록 실패해도 앱은 계속 동작한다(로그만 남김).
     */
    private fun loadLensMetadataAsset() {
        try {
            val json = assets.open("lens_meta.json").bufferedReader().use { it.readText() }
            val result = IrisLensSDK.setLensMetadata(json)
            Log.i(TAG, "P6-W7 lens_meta.json registered: result=$result")
        } catch (e: Exception) {
            Log.e(TAG, "P6-W7 lens_meta.json load failed: ${e.message}")
        }
    }

    /**
     * P6-W4 §5.11: VOLUME_UP 키로 반사 모드 순환 (OFF → EnvMap → Periphery).
     * P6-W4 Phase A 보완: VOLUME_DOWN 키로 intensity sweep (0.3 → 1.0 → 2.0 → 3.0).
     */
    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        when (keyCode) {
            KeyEvent.KEYCODE_VOLUME_UP -> {
                reflectionMode = (reflectionMode + 1) % 3
                cameraGLView.setReflectionMode(reflectionMode)
                val modeName = arrayOf("OFF", "EnvMap", "Periphery")[reflectionMode]
                Toast.makeText(this, "Reflection: $modeName", Toast.LENGTH_SHORT).show()
                Log.i(TAG, "P6-W4 reflection mode → $modeName ($reflectionMode)")
                return true
            }
            KeyEvent.KEYCODE_VOLUME_DOWN -> {
                intensitySweepIdx = (intensitySweepIdx + 1) % intensitySweep.size
                val newIntensity = intensitySweep[intensitySweepIdx]
                cameraGLView.setReflectionIntensity(newIntensity)
                Toast.makeText(this, "Intensity: $newIntensity", Toast.LENGTH_SHORT).show()
                Log.i(TAG, "P6-W4 reflection intensity → $newIntensity")
                return true
            }
        }
        return super.onKeyDown(keyCode, event)
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
        if (stabilizerHandle != 0L) {
            IrisLensSDK.destroyStabilizer(stabilizerHandle)
            stabilizerHandle = 0
        }
        cameraGLView.release()
        lensManager.release()
        analysisExecutor.shutdown()
        IrisLensSDK.releaseDetectionSlot()
        IrisLensSDK.releaseGpuBeauty()
    }
}
