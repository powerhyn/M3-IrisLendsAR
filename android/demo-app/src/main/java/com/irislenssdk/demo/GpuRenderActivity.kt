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
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import com.irislenssdk.BeautyFilterConfigV2
import com.irislenssdk.IrisLensSDK
import com.irislenssdk.IrisResult
import com.irislenssdk.LensConfig
import com.irislenssdk.tracking.FaceTracker
import com.irislenssdk.tracking.TasksToIrisResult
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
 * - MediaPipe Tasks (CPU/GPU delegate): 홍채 추적 (FaceTracker, processFrameTasks)
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
    private lateinit var btnP8Skin: Button       // P8-W1: landmark-masked skin smoothing
    private lateinit var btnP8Radiance: Button   // P8-W3: skin soft-glow radiance(화사함)
    private lateinit var btnP8Slim: Button        // P8-W4: 턱 V라인 슬림(slim_face)
    private lateinit var seekMaxDetail: SeekBar
    private lateinit var tvMaxDetailValue: TextView

    // 뷰티 탭 UI
    private lateinit var btnToggleBeauty: Button
    private lateinit var btnProtectNose: Button

    // 카메라
    private var cameraProvider: ProcessCameraProvider? = null
    private var lensFacing = CameraSelector.LENS_FACING_FRONT
    private val analysisExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    // 렌즈 관리
    private lateinit var lensManager: LensManager
    private lateinit var lensAdapter: LensAdapter
    private var lensConfig = LensConfig()

    // 뷰티 설정
    private var beautyConfig = com.irislenssdk.BeautyFilterConfigV2.Builder().enabled(true).intensity(1.0f).build()
    private var beautyEnabled = true

    //=========================================================================
    // 추적: MediaPipe Tasks 단일 경로 (W4-D — LEGACY 자체 검출 경로 제거)
    //=========================================================================

    private var faceTracker: FaceTracker? = null      // 분석 스레드 전용 (생성·detect·close 동일 스레드)
    private val tasksIrisResult = IrisResult()        // 분석 스레드 전용 (TASKS 변환 수신)

    /** TASKS stabilizer 핸들 — 분석 스레드 전용 (코어 stabilize 단일 적용). */
    private var tasksStabilizerHandle: Long = 0
    @Volatile private var tasksUsingGpu = false
    @Volatile private var lastTasksInferMs = 0f
    private var tasksHudCounter = 0                   // 분석 스레드 전용

    private lateinit var btnFrameSync: Button
    @Volatile private var frameSyncEnabled = false  // frame-sync 킬스위치 (트래킹 지연 핸드오프 §3-b)
    private lateinit var tvAbHud: TextView

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

        // P6-W5 §5.9: 4조합 블라인드 벤치 버튼 (A/B/C/D)
        btnBenchA = findViewById(R.id.btnBenchA)
        btnBenchB = findViewById(R.id.btnBenchB)
        btnBenchC = findViewById(R.id.btnBenchC)
        btnBenchD = findViewById(R.id.btnBenchD)
        btnW6Blink = findViewById(R.id.btnW6Blink)
        btnW6Gate = findViewById(R.id.btnW6Gate)
        btnW6Detail = findViewById(R.id.btnW6Detail)
        btnW7Measured = findViewById(R.id.btnW7Measured)
        btnP8Skin = findViewById(R.id.btnP8Skin)
        btnP8Radiance = findViewById(R.id.btnP8Radiance)
        btnP8Slim = findViewById(R.id.btnP8Slim)
        seekMaxDetail = findViewById(R.id.seekMaxDetail)
        tvMaxDetailValue = findViewById(R.id.tvMaxDetailValue)

        // W4-D: frame-sync 킬스위치 + TASKS HUD (추적 공급자 토글/듀얼 A/B 측정 제거)
        btnFrameSync = findViewById(R.id.btnFrameSync)
        tvAbHud = findViewById(R.id.tvAbHud)

        // 뷰티 탭 UI
        btnToggleBeauty = findViewById(R.id.btnToggleBeauty)
        btnProtectNose = findViewById(R.id.btnProtectNose)

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
        setupTrackingAbControls()

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
        // P8-W1: landmark-masked skin smoothing — off → 0.5 → 1.0 사이클.
        // 뷰티 토글 ON 상태에서만 시각 효과. off면 스무딩 없음(레거시 FreqSep/Bilateral 제거됨 — P8-W2).
        btnP8Skin.setOnClickListener {
            p8SkinIdx = (p8SkinIdx + 1) % p8SkinSweep.size
            val s = p8SkinSweep[p8SkinIdx]
            cameraGLView.setSkinMaskSmoothing(s > 0f, s)
            btnP8Skin.text = if (s > 0f) String.format("skin:%.1f", s) else "skin:off"
            Log.i(TAG, "P8-W1 skin mask smoothing → strength $s")
        }
        // P8-W3: skin 화사함(soft-glow radiance) — off → 0.40 → 0.60 사이클.
        // skin 경로(블러+마스크)를 공유하되 게이트 독립 — radiance>0이면 btnP8Skin off여도 단독 적용.
        // (Beauty 토글 ON 필요 — radiance는 뷰티 효과.)
        btnP8Radiance.setOnClickListener {
            p8RadianceIdx = (p8RadianceIdx + 1) % p8RadianceSweep.size
            val s = p8RadianceSweep[p8RadianceIdx]
            cameraGLView.setSkinRadiance(s)
            btnP8Radiance.text = if (s > 0f) String.format("rad:%.2f", s) else "rad:off"
            Log.i(TAG, "P8-W3 skin radiance → strength $s")
        }
        // P8-W4: 턱 V라인 슬림 — off → 0.25 → 0.50 (slim_face config 필드, GPU 워프 패스).
        // Beauty 토글 ON + 얼굴 검출 필요. 눈높이 변위≈0이라 렌즈 무영향(W4-A 실데이터 검증).
        btnP8Slim.setOnClickListener {
            p8SlimIdx = (p8SlimIdx + 1) % p8SlimSweep.size
            val s = p8SlimSweep[p8SlimIdx]
            beautyConfig.slimFace = s
            cameraGLView.setBeautyConfig(beautyConfig)
            btnP8Slim.text = if (s > 0f) String.format("slim:%.2f", s) else "slim:off"
            Log.i(TAG, "P8-W4 slim face → $s")
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
    private val p8SkinSweep = floatArrayOf(0f, 0.5f, 1.0f)   // P8-W1: off → 0.5 → 1.0 사이클
    private var p8SkinIdx = 0         // 기본 off (SDK default와 일치 — FreqSep 경로 무회귀)
    private val p8RadianceSweep = floatArrayOf(0f, 0.40f, 0.60f) // P8-W3: off → 0.40 → 0.60 (핸드오프 기본 0.40)
    private var p8RadianceIdx = 0     // 기본 off (SDK default와 일치)
    private val p8SlimSweep = floatArrayOf(0f, 0.25f, 0.5f) // P8-W4: off → 0.25 → 0.50 (핸드오프 데모 기본 0.25)
    private var p8SlimIdx = 0          // 기본 off (SDK default와 일치)

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

        // 코 보호 토글
        btnProtectNose.setOnClickListener {
            beautyConfig.protectNose = !beautyConfig.protectNose
            btnProtectNose.text = if (beautyConfig.protectNose) "코보호: ON" else "코보호: OFF"
            btnProtectNose.setTextColor(if (beautyConfig.protectNose) 0xFF00FF00.toInt() else 0xFFAAAAAA.toInt())
            cameraGLView.setBeautyConfig(beautyConfig)
        }

        btnToggleBeauty.text = if (beautyEnabled) "Beauty: ON" else "Beauty: OFF"
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
                if (overlayView.showRawIris) 0xFF00FF00.toInt() else 0xFFAAAAAA.toInt()
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

        btnToggleDebug.setOnClickListener {
            overlayView.debugMode = !overlayView.debugMode
            updateButtonColors()
            Toast.makeText(this, "Debug: ${if (overlayView.debugMode) "ON" else "OFF"}", Toast.LENGTH_SHORT).show()
        }

        // 진단(트래킹 지연 핸드오프 §2): raw(stabilize 이전, 마젠타) + 필터(녹색) 홍채 중심 오버레이.
        // 움직임 중 마젠타가 화면 눈보다 늦으면 = 픽셀-랜드마크 프레임 불일치(필터 무관) → frame-sync 필요.
        btnToggleIris.setOnClickListener {
            overlayView.showRawIris = !overlayView.showRawIris
            updateButtonColors()
            Toast.makeText(this, "Raw 홍채 오버레이(진단): ${if (overlayView.showRawIris) "ON (마젠타=raw, 녹색=필터)" else "OFF"}", Toast.LENGTH_SHORT).show()
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
        beautyConfig.brightness = 1.0f
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

        // ImageAnalysis (MediaPipe Tasks 추론용)
        // W4-D: TASKS 단일 경로. Tasks는 YUV 직접 입력 불가(함정 #2) → RGBA_8888 직접 스트림.
        val imageAnalysis = ImageAnalysis.Builder()
            .setResolutionSelector(resolutionSelector)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
            .apply {
                setAnalyzer(analysisExecutor) { imageProxy ->
                    processFrameTasks(imageProxy)
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

    //=========================================================================
    // TASKS 공급자 경로 + frame-sync 제어 (W4-D — LEGACY 제거 후 단일 경로)
    //=========================================================================

    /**
     * TASKS 공급자 프레임 처리 (§4.1) — 분석 스레드 전용.
     *
     * FaceTracker(MediaPipe Tasks, RGBA_8888 직접 입력)가 detect 후
     * [onTasksRawResult]를 같은 스레드에서 동기 호출한다.
     * imageProxy의 close는 FaceTracker.analyze가 책임진다.
     */
    private fun processFrameTasks(imageProxy: androidx.camera.core.ImageProxy) {
        val rotation = imageProxy.imageInfo.rotationDegrees
        if (rotation != lastRotation) {
            lastRotation = rotation
            cameraGLView.setFrameRotation(rotation)
            Log.d(TAG, "Camera rotation (TASKS): $rotation")
        }
        ensureFaceTracker().analyze(imageProxy)
    }

    /** 분석 스레드 전용 — FaceTracker는 생성 스레드에서만 detect/close (스레드 친화성). */
    private fun ensureFaceTracker(): FaceTracker {
        faceTracker?.let { return it }
        val tracker = FaceTracker(
            context = applicationContext,
            preferGpu = true,
            // ⚠️ mirror=false 필수: 변환 계약은 비미러(센서 원본 upright) 공간 —
            // ADR §7.4 '미러는 렌더 단일 책임'. LEGACY 검출 결과와 동일 공간이어야
            // 같은 DetectionSlot/렌더 경로에서 A/B가 정합한다.
            mirror = false,
            // 내장 One-Euro 경로(onSnapshot)는 소비하지 않는다 — 이중 필터 금지
            // (§5 주의 3 '우회'). 스무딩은 LEGACY와 동일하게 코어 stabilize 단일 적용.
            onSnapshot = { },
            onInferenceStats = { ms, gpu ->
                lastTasksInferMs = ms
                tasksUsingGpu = gpu
            },
            onError = { msg -> Log.w(TAG, "③-3 FaceTracker: $msg") },
        )
        tracker.onRawResult = ::onTasksRawResult
        faceTracker = tracker
        return tracker
    }

    /**
     * TASKS 원시 478점 → IrisResult 변환 → 코어 stabilize → DetectionSlot (§4.1).
     *
     * LEGACY processFrame과 단계별 1:1 대응 — 동일 stabilize·동일 슬롯 채널·동일
     * 슬롯 갱신 정책(stabilize 후 결과를 매 프레임 무조건 갱신 — 미검출 hold/fade
     * 포함, LEGACY 실동작과 동일)·동일 GL/Overlay 전달 정책.
     * 비교 변인은 추적기뿐이다. 분석 스레드 동기 실행.
     */
    private fun onTasksRawResult(
        result: FaceLandmarkerResult,
        rotation: Int,
        srcWidth: Int,
        srcHeight: Int,
        rgba: java.nio.ByteBuffer,
        rowStride: Int,
        frameTimestampNs: Long,
    ) {
        val faces = result.faceLandmarks()
        // numFaces=2(MP 내부 스무딩 우회, FaceTracker)면 배경 얼굴/포스터가 섞일 수 있어
        // 전경(최대 bbox) 얼굴을 고른다 — MediaPipe는 faces[0]가 주 피사체라고 보장하지 않는다.
        // 단일 얼굴이면 bbox 계산 없이 그대로 사용.
        val lm = when (faces.size) {
            0 -> null
            1 -> faces[0]
            else -> faces.maxByOrNull { f ->
                var minX = Float.MAX_VALUE
                var maxX = -Float.MAX_VALUE
                var minY = Float.MAX_VALUE
                var maxY = -Float.MAX_VALUE
                for (p in f) {
                    val x = p.x()
                    val y = p.y()
                    if (x < minX) minX = x
                    if (x > maxX) maxX = x
                    if (y < minY) minY = y
                    if (y > maxY) maxY = y
                }
                (maxX - minX) * (maxY - minY)
            }
        }
        // ④ W4-B4: convert+luma를 SDK 단일 진입(convertWithLuma)으로 일원화 — 같은 전경 얼굴(lm)로
        // 변환·홍채 luma 측정(P7-W2). 측정 비용=눈당 디스크 스캔(분석 스레드, 검출 지연과 분리).
        // 동작 동일(이전 convert+fillIrisLuma 2단계와 같은 측정·같은 얼굴).
        val converted = lm != null && TasksToIrisResult.convertWithLuma(
            lm, rotation, srcWidth, srcHeight, result.timestampMs(), rgba, rowStride, tasksIrisResult
        )
        if (!converted) {
            TasksToIrisResult.fillNoFace(
                rotation, srcWidth, srcHeight, result.timestampMs(), tasksIrisResult
            )
        }

        // 진단(트래킹 지연 핸드오프 §2): stabilize 이전 raw 홍채 사본 — OverlayView 마젠타 오버레이용.
        // 변환만 거친(필터 없음) 좌표라 '필터 이전부터 늦는가' 확인의 정본.
        val rawDiagSnapshot = IrisResult().also { it.copyFrom(tasksIrisResult) }

        // Temporal Stabilizer 적용 (검출 실패 포함 — hold/fade-out 동작 필요, LEGACY 동일)
        // TASKS 전용 핸들 — 같은 코어 stabilize, LEGACY 핸들 수명 불간섭 (§5-5)
        if (tasksStabilizerHandle == 0L) {
            tasksStabilizerHandle = IrisLensSDK.createStabilizer()
        }
        if (tasksStabilizerHandle != 0L) {
            val timestampSec = System.nanoTime() / 1_000_000_000.0
            IrisLensSDK.stabilize(tasksStabilizerHandle, tasksIrisResult, timestampSec)
        }

        // GPU 렌더러에 스무딩된 결과 전달 (매 프레임, 미검출 포함 — 새 복사본, LEGACY 동일)
        val glSnapshot = IrisResult().also { it.copyFrom(tasksIrisResult) }
        cameraGLView.setIrisResult(glSnapshot)

        // P4-W1-03 패리티: 홍채 5점 크로스 휘도 (LEGACY sampleIrisLuminanceNv21 대응)
        val rawLum = if (lm != null) {
            TasksToIrisResult.sampleCrossLuma(
                rgba, rowStride, srcWidth, srcHeight, lm, tasksIrisResult
            )
        } else {
            -1f
        }
        cameraGLView.setRawIrisLuminance(rawLum)

        // Detection Slot 업데이트 — LEGACY와 동일 채널 공유 (렌더 경로 완전 동일).
        // LEGACY processFrame은 detectWithRotation이 미검출(detected=false)에도 항상
        // IRIS_SDK_OK를 반환하므로 `if (detectResult == OK)` 게이트가 매 프레임 참이 되어
        // stabilize 후 결과(hold/fade-out 궤적·detected=false 포함)를 무조건 슬롯에 넣는다
        // (nativeUpdateDetectionSlot도 detected 무관 무조건 복사). TASKS도 동일하게
        // stabilize 후 결과를 무조건 갱신해야 검출 손실 구간에서 슬롯 정본(getActiveDetectionSlot
        // 소비 — 네이티브 렌더/뷰티 + GL 렌즈 게이트)이 양 모드 동일하게 게이트된다 (plan §4.1 비교 변인=추적기뿐).
        // fillNoFace 프레임(detected=false)도 그대로 들어가야 LEGACY 미검출 동작과 일치.
        // W4-B3: 분석 프레임 센서 ns(frameTimestampNs)를 좌표와 한 슬롯에 원자 결속 — 별도 ts 채널 폐기.
        IrisLensSDK.updateDetectionSlot(tasksIrisResult, frameTimestampNs)

        // OverlayView 전달 (LEGACY와 동일 정책 — 변환 결과의 upright frame dims 사용)
        val uiSnapshot = IrisResult().also { it.copyFrom(tasksIrisResult) }
        val frameW = uiSnapshot.frameWidth
        val frameH = uiSnapshot.frameHeight
        runOnUiThread {
            overlayView.setRawIris(rawDiagSnapshot)
            overlayView.setIrisResult(
                uiSnapshot, frameW, frameH,
                lensFacing == CameraSelector.LENS_FACING_FRONT
            )
        }

        // HUD 1줄 (30프레임 스로틀)
        if (++tasksHudCounter >= 30) {
            tasksHudCounter = 0
            val hud = String.format(
                java.util.Locale.US, "TRK:TASKS(%s) infer≈%.0fms",
                if (tasksUsingGpu) "gpu" else "cpu", lastTasksInferMs
            )
            runOnUiThread { tvAbHud.text = hud }
        }

        updateFps()
    }

    /** W4-D UI: frame-sync 킬스위치 (TASKS 단일 경로 — 공급자 토글/듀얼 측정 제거) */
    private fun setupTrackingAbControls() {
        tvAbHud.text = "TRK:TASKS"

        // frame-sync 킬스위치 (트래킹 지연 핸드오프 §3-b): 렌더가 '랜드마크가 계산된 프레임'을
        // 그린다. ON이면 렌즈-눈 어긋남 제거, 대신 거울 지연 +2~3프레임.
        btnFrameSync.setOnClickListener {
            setFrameSyncEnabled(!frameSyncEnabled)
        }
    }

    private fun setFrameSyncEnabled(enabled: Boolean) {
        frameSyncEnabled = enabled
        cameraGLView.setFrameSyncEnabled(enabled)
        btnFrameSync.text = if (enabled) "fsync:on" else "fsync:off"
        btnFrameSync.setBackgroundColor(
            if (enabled) 0xCC2196F3.toInt() else 0x66555555.toInt()
        )
        Toast.makeText(
            this,
            "Frame-sync: ${if (enabled) "ON (렌즈 정합↑, 거울 지연↑)" else "OFF (현행)"}",
            Toast.LENGTH_SHORT
        ).show()
        Log.i(TAG, "frame-sync → $enabled")
    }

    private fun updateFps() {
        frameCount++
        val currentTime = System.currentTimeMillis()
        if (currentTime - lastFpsTime >= 1000) {
            val fps = frameCount
            frameCount = 0
            lastFpsTime = currentTime

            runOnUiThread {
                // SDK 렌즈 실패 시 명시 표시 (무음 폴백 제거 — 검증 통로가 거짓말하지 않게)
                val lensFailure = cameraGLView.getSdkLensFailure()
                tvFps.text = if (lensFailure != null) {
                    "Detect FPS: $fps  ⚠ SDK 렌즈 실패: $lensFailure"
                } else {
                    "Detect FPS: $fps"
                }
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
        cameraGLView.release()
        lensManager.release()
        // 분석 스레드 소유 자원 정리 — GPU delegate 스레드 친화성상 같은 스레드에서
        // close해야 한다. shutdown은 큐 잔여 작업 완료 후 종료된다.
        runCatching {
            analysisExecutor.execute {
                faceTracker?.close()
                faceTracker = null
                if (tasksStabilizerHandle != 0L) {
                    IrisLensSDK.destroyStabilizer(tasksStabilizerHandle)
                    tasksStabilizerHandle = 0L
                }
            }
        }
        analysisExecutor.shutdown()
        IrisLensSDK.releaseDetectionSlot()
        IrisLensSDK.releaseGpuBeauty()
    }
}
