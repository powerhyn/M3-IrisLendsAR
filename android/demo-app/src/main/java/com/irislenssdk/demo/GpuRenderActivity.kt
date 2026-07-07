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
import android.widget.ImageButton
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
    private lateinit var btnToggleMaskEdge: Button

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
    private lateinit var btnW4Cap: Button        // P7-W4 §5.8: 흰자 빛남 cap sweep
    private lateinit var btnImgMode: Button      // NLR 트래킹 A/B: MediaPipe IMAGE 모드 (내부 스무딩·ROI 우회)
    private lateinit var btnStabFast: Button     // NLR 트래킹 A/B: stabilizer near-raw 프리셋
    private lateinit var seekMaxDetail: SeekBar
    private lateinit var tvMaxDetailValue: TextView

    // 개발자 패널 토글 (기어 ⚙ — 벤치/디버그 패널 전체 표시/숨김)
    private lateinit var btnGearToggle: ImageButton
    private lateinit var devPanelContainer: View

    // 뷰티 탭 UI — 토글 + 단계형 슬라이더 (P8 흩어진 sweep 버튼 통합)
    private lateinit var btnToggleBeauty: Button
    private lateinit var seekSlim: SeekBar          // P8-W4: 턱 V라인 슬림
    private lateinit var tvSlimValue: TextView
    private lateinit var seekSkin: SeekBar          // P8-W1: 피부 스무딩
    private lateinit var tvSkinValue: TextView
    private lateinit var seekRadiance: SeekBar      // P8-W3: 화사함 radiance
    private lateinit var tvRadianceValue: TextView

    // 뷰티 슬라이더 단계값 (기존 sweep 사전지정값 기반 + 가짓수 2배 디테일화).
    // progress = 단계 인덱스. 기존 sweep 값을 모두 포함하고 그 사이를 보간했다.
    private val slimSteps = floatArrayOf(0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f)   // 0.1 간격, 기본 0.2(idx 2)
    private val skinSteps = floatArrayOf(0f, 0.25f, 0.5f, 0.75f, 1.0f)       // 기존 [0, 0.5, 1.0]
    private val radianceSteps = floatArrayOf(0f, 0.2f, 0.4f, 0.5f, 0.6f)     // 기존 [0, 0.40, 0.60]

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

    // 눈 일부 감김 시 얼굴 전체 dropout(MediaPipe 0 faces, 실측 ~1s)을 버티도록 stabilizer
    // dropout hold 연장(코어 기본 5프레임 → 이 값). hold 동안 마지막 유효 프레임(face_mesh+
    // iris+detected=true)이 슬롯에 유지되어 렌즈가 제자리+눈꺼풀 클립으로 유지된다.
    // 실기기 육안 튜닝 포인트 — 너무 크면 얼굴이 실제로 프레임을 떠난 뒤 렌즈 잔상.
    private val stabilizerHoldFrames = 30
    @Volatile private var tasksUsingGpu = false
    @Volatile private var lastTasksInferMs = 0f
    private var tasksHudCounter = 0                   // 분석 스레드 전용

    private lateinit var btnFrameSync: Button
    @Volatile private var frameSyncEnabled = true   // frame-sync 킬스위치 (기본 ON). 트래킹 지연 핸드오프 §3-b
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
        btnW4Cap = findViewById(R.id.btnW4Cap)
        btnImgMode = findViewById(R.id.btnImgMode)
        btnStabFast = findViewById(R.id.btnStabFast)
        seekMaxDetail = findViewById(R.id.seekMaxDetail)
        tvMaxDetailValue = findViewById(R.id.tvMaxDetailValue)

        // 개발자 패널 토글 (기어)
        btnGearToggle = findViewById(R.id.btnGearToggle)
        devPanelContainer = findViewById(R.id.devPanelContainer)

        // W4-D: frame-sync 킬스위치 + TASKS HUD (추적 공급자 토글/듀얼 A/B 측정 제거)
        btnFrameSync = findViewById(R.id.btnFrameSync)
        tvAbHud = findViewById(R.id.tvAbHud)

        // 뷰티 탭 UI — 토글 + 슬라이더
        btnToggleBeauty = findViewById(R.id.btnToggleBeauty)
        seekSlim = findViewById(R.id.seekSlim)
        tvSlimValue = findViewById(R.id.tvSlimValue)
        seekSkin = findViewById(R.id.seekSkin)
        tvSkinValue = findViewById(R.id.tvSkinValue)
        seekRadiance = findViewById(R.id.seekRadiance)
        tvRadianceValue = findViewById(R.id.tvRadianceValue)

        // GPU 초기화 콜백 설정
        cameraGLView.onGpuInitialized = { success ->
            // GPU tier 판별 (GL 컨텍스트 활성 상태)
            detectGpuTier()
            runOnUiThread {
                tvGpuStatus.text = "GPU: Available (init: $success)"
                Log.d(TAG, "GPU initialized: $success")
            }
            if (success) {
                // P6-W4 §5.7: env_map은 GL 텍스처라 EGL 컨텍스트 (재)생성마다 소실된다
                //   → 1회 가드 없이 매번 재로드해야 백그라운드 복귀 후 반사가 동작한다.
                //   (lens_meta는 코어 CPU 상태 g_sku_registry라 컨텍스트와 무관하게 생존 → 1회 가드 유지.)
                loadEnvMapAsset()
                if (!lensMetaLoaded) {
                    loadLensMetadataAsset()
                    lensMetaLoaded = true
                }
                // 뷰티·렌즈·렌더 토글 복원: GL 컨텍스트 (재)생성 후 native 렌더러 상태가 기본값으로
                // 리셋되므로 현재 UI 상태를 재적용한다. queueEvent 기반 set*는 GL init 전엔 무효화되므로
                // 여기서 재적용해야 초기/복귀 시 피부·렌즈·반사가 실제로 반영된다(슬라이더/렌즈 재선택 없이).
                runOnUiThread {
                    applyBeautyFromSliders()
                    cameraGLView.setFrameSyncEnabled(frameSyncEnabled)  // GL 재생성 후 fsync 복원
                    restoreLensRenderState()
                }
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

        // 블렌드 모드 선택 — 활성 ID {0,1,2,5,7}만 노출 (micro-cleanup, sdk_api.h IrisBlendMode 정합).
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

        // Sclera Protection 토글 (P4-W2-01, 기본 ON) — NLR-W1: 복원 가능하도록 필드 승격
        btnToggleSclera.setOnClickListener {
            scleraProtectOn = !scleraProtectOn
            cameraGLView.setScleraProtect(scleraProtectOn)
            btnToggleSclera.text = if (scleraProtectOn) "Sclera: ON" else "Sclera: OFF"
            btnToggleSclera.setBackgroundColor(if (scleraProtectOn) 0x4400CC00.toInt() else 0x44FF0000.toInt())
        }

        // Contact Shadow 토글 (P4-W2-01, 기본 OFF) — NLR-W1: 복원 가능하도록 필드 승격
        btnToggleShadow.setOnClickListener {
            contactShadowOn = !contactShadowOn
            cameraGLView.setContactShadow(contactShadowOn)
            btnToggleShadow.text = if (contactShadowOn) "Shadow: ON" else "Shadow: OFF"
            btnToggleShadow.setBackgroundColor(if (contactShadowOn) 0x4400CC00.toInt() else 0x44FF0000.toInt())
        }

        // EYECLIP A-2: 눈꺼풀 마스크 모드 3-way 순환 (Y-slab → Ellipse → Contour)
        btnToggleEllipse.setOnClickListener {
            maskMode = (maskMode + 1) % 3
            cameraGLView.setEyelidMaskMode(maskMode)
            btnToggleEllipse.text = when (maskMode) {
                1 -> "Mask: Ellipse"
                2 -> "Mask: Contour"
                else -> "Mask: Y-slab"
            }
            btnToggleEllipse.setBackgroundColor(when (maskMode) {
                1 -> 0x4400CC00.toInt()
                2 -> 0x440066FF.toInt()
                else -> 0x44FF0000.toInt()
            })
        }


        // 홍채 밝기 보정 슬라이더 (P4-W2-01, 0.8~1.4 / 0.1 스텝 / 기본 1.2)
        // NLR-W2 R5: dead 슬라이더(구 setMaxDetail — 소비처 없음) → 흰자 페이드 시작점 재배선.
        // C(기하 디버그) 켠 채 드래그하면 빨강 창이 움직임 → 빛나는 링을 덮게 맞춘 뒤 D로 결과 확인.
        seekMaxDetail.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                val value = 0.8f + progress * 0.1f
                fadeStartV = value
                tvMaxDetailValue.text = String.format("f%.1f", value)
                cameraGLView.setLensFadeStart(value)
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
        // NLR 트래킹 A/B: MediaPipe RunningMode.IMAGE 토글 — 내부 스무딩·ROI 추적 우회
        // (mediapipe-internal-smoothing-bypass 핸드오프). 전환 시 landmarker는 분석 스레드에서
        // 자동 재생성(필터 리셋 포함) — 1~2프레임 렌즈 드롭은 정상.
        btnImgMode.setOnClickListener {
            imgModeOn = !imgModeOn
            faceTracker?.setImageMode(imgModeOn)
            btnImgMode.text = if (imgModeOn) "img:on" else "img:off"
            btnImgMode.setBackgroundColor(if (imgModeOn) 0xCC2196F3.toInt() else 0x66555555.toInt())
            Log.i(TAG, "NLR tracking mode → ${if (imgModeOn) "IMAGE(스무딩·ROI 우회)" else "VIDEO"}")
        }
        // NLR 트래킹 A/B: stabilizer near-raw 프리셋 토글 — 분석 스레드에서 재생성 (플래그 방식).
        btnStabFast.setOnClickListener {
            stabFastOn = !stabFastOn
            stabRecreateRequested = true
            btnStabFast.text = if (stabFastOn) "stab:fast" else "stab:norm"
            btnStabFast.setBackgroundColor(if (stabFastOn) 0xCC2196F3.toInt() else 0x66555555.toInt())
            Log.i(TAG, "NLR stabilizer → ${if (stabFastOn) "FAST(near-raw 3.0/200)" else "NORM(4.0/15)"}")
        }
        // P7-W4 §5.8: TintLinearV2 흰자 빛남 cap sweep (1.275/1.5/2.0/OFF).
        btnW4Cap.setOnClickListener {
            w4CapIdx = (w4CapIdx + 1) % w4CapSweep.size
            val cap = w4CapSweep[w4CapIdx]
            cameraGLView.setScleraTintMax(cap)
            btnW4Cap.text = if (cap >= 1.0e5f) "cap:off" else String.format("cap%.2f", cap)
            Log.i(TAG, "P7-W4 sclera tint cap → ${if (cap >= 1.0e5f) "off" else "$cap"}")
        }
        // (P8 통합) skin/radiance/slim sweep 버튼 → 뷰티 탭 슬라이더로 이전 (setupBeautyControls).
    }

    //=========================================================================
    // P6-W5 §5.9: B1/B8 4조합 블라인드 벤치 (A/B/C/D)
    // 정답표는 코드/로그에만 존재. 평가자에게는 라벨만 노출.
    //=========================================================================

    private data class BenchCombo(val label: String, val blendMode: Int, val vetoMode: Int, val desc: String)

    // NLR-W2 수식 후보 블라인드 벤치 (codex_r1.md §5): veto는 전 조합 legacy 0 고정(§5.10 변수 분리),
    // lum:meas 고정 권장. 정답표는 여기에만 — 평가자에겐 A/B/C/D 라벨만.
    private val benchCombos = listOf(
        BenchCombo("A", 5, 0, "current TintLinearV2 (control)"),
        BenchCombo("B", 3, 0, "R6: TintLinear 고정 scale K=3.4 (장면 적응만 제거, lum 곱셈 유지)"),
        BenchCombo("C", 4, 0, "R5: 기하 디버그 (초록<0.95/빨강 0.95-1.15/파랑>1.15)"),
        BenchCombo("D", 6, 0, "E-v3: V2 수식 + 흰자 조기 페이드만"),
    )
    private var currentBenchIdx = -1

    // P6-W6 §5.3/§5.7: 벤치 토글 sweep 상태 (기본값=중간값, 코어 기본과 일치).
    private val w6BlinkUpSweep = floatArrayOf(60f, 80f, 120f)
    private var w6BlinkIdx = 1   // 기본 80ms
    private val w6GateSweep = floatArrayOf(0.10f, 0.15f, 0.25f)
    private var w6GateIdx = 0    // 기본 0.10 (저조도 드묾 — C10 디테일 항상 ON)
    private var w6DetailOn = true
    private var w7MeasuredOn = true   // P7-W2 §5.6: 실기기 검증 후 기본 실측 ON (SDK default와 일치). 토글로 fallback 비교.

    // P7-W4 §5.8 → NLR-W2 R4 재조정: 유효 구간(0.95~1.15)으로 sweep 교체. 마지막 = OFF 센티널(1e6).
    // D(V2+페이드) + cap 조합 = 후보 H 라이브 튜닝 (cap이 렌즈 내 밝은 픽셀 과증폭 상한 역할).
    private val w4CapSweep = floatArrayOf(0.95f, 1.05f, 1.15f, 1.0e6f)
    private var w4CapIdx = 3   // 시작 = OFF (코어 기본 1.275는 사실상 무효 구간이라 OFF와 동일 취급)

    // NLR-W1 복원 갭 보수: 컨텍스트 재생성 시 native 기본값으로 리셋되는 상태들의 UI 측 진실값.
    // (기존 지역 변수라 restoreLensRenderState가 복원 불가했던 구조 결함 — 필드 승격)
    private var scleraProtectOn = true    // 코어 기본 ON과 일치
    private var contactShadowOn = false   // 코어 기본 OFF와 일치
    private var currentVetoMode = 0       // P6-W5 B8 미판정 — legacy 0 유지 (§5.10)
    private var fadeStartV = 0.95f        // NLR-W2 R5: 흰자 페이드 시작점 (코어 기본 0.95와 일치)
    private var imgModeOn = false         // NLR 트래킹 A/B: IMAGE 모드 (기본 VIDEO — 트래커 재생성 시 재적용)

    // NLR 트래킹 A/B: stabilizer near-raw 프리셋 (3.0/200 — LensSim 등가). 핸들은 분석 스레드
    // 전용이라 UI에서 직접 destroy 금지 — 재생성 요청 플래그만 세우고 분석 경로에서 처리.
    @Volatile private var stabFastOn = false
    @Volatile private var stabRecreateRequested = false
    private var maskMode = 0      // EYECLIP A-2: 눈꺼풀 마스크 모드 0=Y-slab, 1=ellipse, 2=contour. 컨텍스트 재생성 후 restoreLensRenderState로 복원.
    // (P8 통합) p8Skin/Radiance/Slim sweep 상태 제거 — 뷰티 탭 슬라이더가 연속값을 직접 보유.

    // 활성 blend ID {0,1,2,5,7} + NLR-W2 벤치 임시 슬롯 {3,4,6}.
    // ⚠️ 3/4/6은 벤치 기간 한정 재배선(KM/OkShift/Pivot — codex_r1.md §5) — develop 머지 금지,
    //   채택 시 W6에서 정식 ID 부여 후 원래 "deprecated → ID5 fallback"으로 복원.
    // 스피너 position ≠ blend ID이므로 선택/복원은 반드시 값 기반 역조회(indexOfFirst)로.
    private val blendModeEntries = arrayOf(
        "Normal" to 0, "Multiply" to 1, "Screen Linear" to 2,
        "Lum Tint Linear" to 5, "Color Replace" to 7,
        "Quot†" to 3, "Quot+F†" to 4, "V2+Fade†" to 6
    )

    private fun applyBenchCombo(idx: Int) {
        val combo = benchCombos[idx]
        lensConfig.blendMode = combo.blendMode
        cameraGLView.setLensConfig(lensConfig)
        cameraGLView.setScleraVetoMode(combo.vetoMode)
        currentVetoMode = combo.vetoMode  // NLR-W1: 컨텍스트 재생성 복원용 진실값
        // 축소 스피너(5종)에서 position ≠ blend ID — 값 기반 역조회.
        // (구 setSelection(blendMode)은 ID7이 어댑터 범위 초과 → IndexOutOfBounds 크래시)
        val benchIdx = blendModeEntries.indexOfFirst { it.second == combo.blendMode }
        if (benchIdx >= 0) spinnerBlendMode.setSelection(benchIdx)
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
        // 뷰티 마스터 토글 — OFF면 슬라이더도 비활성(조작해도 효과 없어 오인 방지).
        btnToggleBeauty.setOnClickListener {
            beautyEnabled = !beautyEnabled
            beautyConfig.enabled = beautyEnabled
            cameraGLView.setBeautyEnabled(beautyEnabled)
            btnToggleBeauty.text = if (beautyEnabled) "Beauty: ON" else "Beauty: OFF"
            setBeautySlidersEnabled(beautyEnabled)
        }
        btnToggleBeauty.text = if (beautyEnabled) "Beauty: ON" else "Beauty: OFF"

        // === 뷰티 단계형 슬라이더 (기존 sweep 사전지정값 → 가짓수 2배 디테일) ===
        // 각 SeekBar progress = 단계 인덱스. max·progress 먼저 → 리스너 나중(spurious 콜백 회피).
        // skin/radiance는 기본값을 최대로 둔다. 리스너 등록 전 progress 설정은 콜백을 부르지
        // 않으므로, 초기 효과는 아래에서 명시 적용한다.

        // 턱 V라인 슬림 (slimSteps, config 경로 = slim_face GPU 워프). 기본 0.2(20%).
        seekSlim.max = slimSteps.lastIndex
        val slimDefaultIdx = slimSteps.indexOfFirst { it >= 0.2f }.coerceAtLeast(0)
        seekSlim.progress = slimDefaultIdx
        beautyConfig.slimFace = slimSteps[slimDefaultIdx]
        tvSlimValue.text = stepLabel(slimSteps[slimDefaultIdx], "%.2f")
        seekSlim.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                val s = slimSteps[progress]
                beautyConfig.slimFace = s
                tvSlimValue.text = stepLabel(s, "%.2f")
                cameraGLView.setBeautyConfig(beautyConfig)
                Log.i(TAG, "P8-W4 slim face → $s")
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        // 피부 스무딩 (skinSteps, 직접 메서드 경로). 기본 최대(1.0).
        seekSkin.max = skinSteps.lastIndex
        seekSkin.progress = skinSteps.lastIndex
        tvSkinValue.text = stepLabel(skinSteps.last(), "%.2f")
        seekSkin.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                val s = skinSteps[progress]
                tvSkinValue.text = stepLabel(s, "%.2f")
                cameraGLView.setSkinMaskSmoothing(s > 0f, s)
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        // 화사함 radiance (radianceSteps, 직접 메서드 경로). 기본 최대(0.60).
        seekRadiance.max = radianceSteps.lastIndex
        seekRadiance.progress = radianceSteps.lastIndex
        tvRadianceValue.text = stepLabel(radianceSteps.last(), "%.2f")
        seekRadiance.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                val s = radianceSteps[progress]
                tvRadianceValue.text = stepLabel(s, "%.2f")
                cameraGLView.setSkinRadiance(s)
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        // 초기 효과(피부·화사함 최대 등)는 GL 컨텍스트 생성 이후에야 유효하므로
        // onGpuInitialized 콜백의 applyBeautyFromSliders()에서 적용한다(여기 queueEvent는 무효).

        // 초기 슬라이더 활성 상태를 뷰티 토글과 동기화.
        setBeautySlidersEnabled(beautyEnabled)
    }

    /** 현재 뷰티 슬라이더 상태를 GL에 적용 (GL 컨텍스트 생성/재생성 후 호출 — 초기·복원 보장). */
    private fun applyBeautyFromSliders() {
        cameraGLView.setBeautyEnabled(beautyEnabled)
        val skin = skinSteps[seekSkin.progress]
        cameraGLView.setSkinMaskSmoothing(skin > 0f, skin)
        cameraGLView.setSkinRadiance(radianceSteps[seekRadiance.progress])
        beautyConfig.slimFace = slimSteps[seekSlim.progress]
        cameraGLView.setBeautyConfig(beautyConfig)
    }

    /**
     * EGL 컨텍스트 (재)생성 후 native 렌즈 렌더 상태 복원.
     *
     * onSurfaceCreated가 releaseGpuLens()+initGpuLens()로 GPULensRenderer를 새 컨텍스트에 재생성하면
     * reflection/measured-luma/detail/렌즈 텍스처 등 native 멤버가 기본값/미로드로 돌아간다.
     * 백그라운드 복귀 시 렌즈 소멸·반사 미동작을 막기 위해 현재 UI 상태를 재주입한다.
     */
    private fun restoreLensRenderState() {
        // 반사 모드/강도 (native GPULensRenderer 멤버 — 컨텍스트 재생성 시 기본값으로 리셋됨)
        cameraGLView.setReflectionMode(reflectionMode)
        cameraGLView.setReflectionIntensity(intensitySweep[intensitySweepIdx])
        // 렌더 품질 토글(현재 UI 상태)
        cameraGLView.setUseMeasuredLuma(w7MeasuredOn)
        cameraGLView.setDetailReinject(w6DetailOn)
        // EYECLIP A-2: eyelid_mask_mode_는 releaseGpuLens()로 리셋 → 복귀 시 현재 UI 상태 재적용
        //   (누락 시 백그라운드 복귀 후 Y-slab로 강등됨).
        cameraGLView.setEyelidMaskMode(maskMode)
        // NLR-W1: 기존 미복원 갭 일괄 보수 — 아래 native 상태들도 컨텍스트 재생성 시 기본값으로
        //   리셋되나 복원 대상에서 빠져 있었음 (벤치 중 백그라운드 복귀 시 판정 조건이 조용히 어긋남).
        cameraGLView.setBlinkUpMs(w6BlinkUpSweep[w6BlinkIdx])
        cameraGLView.setGateThreshold(w6GateSweep[w6GateIdx])
        cameraGLView.setScleraVetoMode(currentVetoMode)
        cameraGLView.setScleraProtect(scleraProtectOn)
        cameraGLView.setContactShadow(contactShadowOn)
        // P7-W4 §5.8: 흰자 빛남 cap — 현재 sweep 값 재주입
        cameraGLView.setScleraTintMax(w4CapSweep[w4CapIdx])
        // NLR-W2 R5: 흰자 페이드 시작점 — 현재 슬라이더 값 재주입
        cameraGLView.setLensFadeStart(fadeStartV)
        // 현재 선택 렌즈 텍스처 재업로드 (stale native texture는 onSurfaceCreated에서 이미 해제됨).
        if (::lensManager.isInitialized) {
            lensManager.currentLens?.let { lens ->
                if (lens.id != NoLens.ID) {
                    lensManager.getTexture(lens)?.let { bmp ->
                        cameraGLView.setLensTexture(bmp, lens.id)
                        cameraGLView.setLensConfig(lensConfig)
                        cameraGLView.setLensEnabled(true)
                    }
                }
            }
        }
    }

    /** off-aware 단계 라벨 (값 0이면 "off"). */
    private fun stepLabel(v: Float, fmt: String): String =
        if (v > 0f) String.format(fmt, v) else "off"

    /** 뷰티 슬라이더 일괄 활성/비활성 (Beauty OFF면 조작해도 효과 없어 오인 방지). */
    private fun setBeautySlidersEnabled(enabled: Boolean) {
        seekSlim.isEnabled = enabled
        seekSkin.isEnabled = enabled
        seekRadiance.isEnabled = enabled
    }

    private fun setupDebugControls() {
        btnToggleMesh = findViewById(R.id.btnToggleMesh)
        btnToggleDebug = findViewById(R.id.btnToggleDebug)
        btnToggleIris = findViewById(R.id.btnToggleIris)
        btnToggleLog = findViewById(R.id.btnToggleLog)
        btnToggleMaskEdge = findViewById(R.id.btnToggleMaskEdge)

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
            btnToggleMaskEdge.setTextColor(
                if (overlayView.showMaskDebug) 0xFF00FFFF.toInt() else 0xFFAAAAAA.toInt()
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

        // EYECLIP: eyelidMask 형상 디버그 오버레이 토글 (초록=눈꺼풀 16점 / 시안=ellipse fit; A-2 판단용).
        //   OverlayView(CPU 캔버스)에 뚜렷한 선으로 그림 — ellipse가 실제 눈꺼풀을 얼마나 따라가는지 비교.
        btnToggleMaskEdge.setOnClickListener {
            overlayView.showMaskDebug = !overlayView.showMaskDebug
            overlayView.invalidate()
            updateButtonColors()
            Toast.makeText(this, "MaskDebug: ${if (overlayView.showMaskDebug) "ON (초록=눈꺼풀 / 시안=ellipse)" else "OFF"}", Toast.LENGTH_SHORT).show()
        }

        updateButtonColors()

        // 기어(⚙) → 개발/벤치 패널 전체 표시/숨김 (기본 숨김 — 평소 카메라 화면을 가리지 않음).
        btnGearToggle.setOnClickListener {
            devPanelContainer.visibility =
                if (devPanelContainer.visibility == View.VISIBLE) View.GONE else View.VISIBLE
        }
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
        tracker.setImageMode(imgModeOn) // 트래커 재생성 시 현재 A/B 상태 재적용
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
        // (성능 정리) showRawIris OFF(기본)면 매 프레임 478×3 arraycopy가 버려지므로 토글 ON일 때만 사본 생성.
        val rawDiagSnapshot = if (overlayView.showRawIris) {
            IrisResult().also { it.copyFrom(tasksIrisResult) }
        } else null

        // Temporal Stabilizer 적용 (검출 실패 포함 — hold/fade-out 동작 필요, LEGACY 동일)
        // TASKS 전용 핸들 — 같은 코어 stabilize, LEGACY 핸들 수명 불간섭 (§5-5)
        // NLR A/B: 프리셋 전환 요청 시 분석 스레드(여기)에서 재생성 — 핸들 스레드 안전.
        if (stabRecreateRequested) {
            if (tasksStabilizerHandle != 0L) {
                IrisLensSDK.destroyStabilizer(tasksStabilizerHandle)
                tasksStabilizerHandle = 0L
            }
            stabRecreateRequested = false
        }
        if (tasksStabilizerHandle == 0L) {
            // hold 연장본으로 생성 — 눈 일부 감김 시 얼굴 dropout 동안 렌즈 유지(stabilizerHoldFrames 주석 참조).
            tasksStabilizerHandle = if (stabFastOn) {
                IrisLensSDK.createStabilizerFast(stabilizerHoldFrames)
            } else {
                IrisLensSDK.createStabilizer(stabilizerHoldFrames)
            }
        }
        if (tasksStabilizerHandle != 0L) {
            val timestampSec = System.nanoTime() / 1_000_000_000.0
            IrisLensSDK.stabilize(tasksStabilizerHandle, tasksIrisResult, timestampSec)
        }

        // GPU 렌더러에 스무딩된 결과 전달 (매 프레임, 미검출 포함 — 새 복사본, LEGACY 동일)
        val glSnapshot = IrisResult().also { it.copyFrom(tasksIrisResult) }
        cameraGLView.setIrisResult(glSnapshot)

        // (성능 정리) P4-W1-03 KT측 avgIrisLum EMA 체인 제거 — sampleCrossLuma(매 프레임 눈 픽셀
        // 샘플) → setRawIrisLuminance → avgIrisLum 은 읽는 곳이 없는 dead. 렌즈 휘도 적응은 SDK
        // measured-luma(roiLumaLinear, P7-W2 기본 ON)가 담당하므로 KT 샘플링 제거해도 시각 무변화.

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
            rawDiagSnapshot?.let { overlayView.setRawIris(it) }
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
        // 기본 활성화: 버튼 상태 즉시 초기화 (실제 GL 적용은 onGpuInitialized에서 — queueEvent GL-init 타이밍).
        btnFrameSync.text = if (frameSyncEnabled) "fsync:on" else "fsync:off"
        btnFrameSync.setBackgroundColor(
            if (frameSyncEnabled) 0xCC2196F3.toInt() else 0x66555555.toInt()
        )
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
        // P6-W4 env_map 로드는 onGpuInitialized 콜백에서 처리 (GPU lens init 완료 보장).
    }

    //=========================================================================
    // P6-W4: 환경 반사 벤치 (env_map 로드 + 3 프로토타입 토글)
    //=========================================================================

    // env_map은 GL 텍스처라 컨텍스트 (재)생성마다 재로드(onGpuInitialized) — 1회 가드 없음.
    private var lensMetaLoaded = false  // P6-W7: lens_meta.json 1회 등록 가드(코어 CPU 상태라 생존)
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
