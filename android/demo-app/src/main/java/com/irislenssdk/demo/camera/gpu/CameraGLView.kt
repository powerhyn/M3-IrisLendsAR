/**
 * IrisLensSDK Android - CameraGLView
 *
 * GLSurfaceView 기반 카메라 뷰
 * - CameraX와 OpenGL 연동
 * - GPU 기반 실시간 렌더링
 *
 * @version 1.0.0
 */
package com.irislenssdk.demo.camera.gpu

import android.content.Context
import android.graphics.Bitmap
import android.graphics.SurfaceTexture
import android.opengl.GLSurfaceView
import android.util.AttributeSet
import android.util.Log
import android.view.Surface
import androidx.camera.core.Preview
import androidx.camera.core.SurfaceRequest
import com.irislenssdk.BeautyFilterConfigV2
import com.irislenssdk.IrisResult
import com.irislenssdk.LensConfig
import java.util.concurrent.Executors

/**
 * GPU 렌더링 기반 카메라 뷰
 *
 * PreviewView 대신 사용하여 GPU 가속 렌더링 지원
 */
class CameraGLView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : GLSurfaceView(context, attrs) {

    companion object {
        private const val TAG = "CameraGLView"
    }

    // GL 렌더러
    private val glRenderer = CameraGLRenderer()

    // SurfaceTexture (카메라 연결용)
    private var cameraSurfaceTexture: SurfaceTexture? = null
    private var cameraSurface: Surface? = null

    // GPU 초기화 콜백
    var onGpuInitialized: ((Boolean) -> Unit)? = null
        set(value) {
            field = value
            glRenderer.onGpuInitialized = value
        }

    // GPU FPS 업데이트 콜백
    var onGpuFpsUpdated: ((Int) -> Unit)? = null
        set(value) {
            field = value
            glRenderer.onGpuFpsUpdated = value
        }

    // StabilityLogger 콜백 (P4-W1-02)
    // M-1 fix: GL 스레드에서만 접근하도록 queueEvent 사용
    fun setStabilityFrameCallback(callback: ((
        faceDetected: Boolean,
        rawLeftCx: Float, rawLeftCy: Float, rawLeftR: Float,
        filteredLeftCx: Float, filteredLeftCy: Float, filteredLeftR: Float,
        rawRightCx: Float, rawRightCy: Float, rawRightR: Float,
        filteredRightCx: Float, filteredRightCy: Float, filteredRightR: Float,
        eyelidLt: Float, eyelidLb: Float, eyelidRt: Float, eyelidRb: Float,
        holdActive: Boolean, holdRemaining: Int,
        renderTimeUs: Long
    ) -> Unit)?) {
        queueEvent {
            glRenderer.onStabilityFrame = callback
        }
    }

    fun setStabilityLogEnabled(enabled: Boolean) {
        queueEvent {
            glRenderer.stabilityLogEnabled = enabled
        }
    }

    // (P4-W1-03 정리) setRawIrisLuminance/avgIrisLum EMA 제거 — 읽는 곳 0의 dead 측정.
    // 렌즈 휘도 적응은 SDK measured-luma(roiLumaLinear, P7-W2) 경로가 담당.

    // W4-B3: setLandmarkFrameTimestamp 제거 — 분석 프레임 센서 ns는 이제 IrisLensSDK.updateDetectionSlot(
    // result, frameTsNs)로 렌즈 좌표와 한 슬롯에 원자 결속된다. 별도 ts 사이드채널 폐기로 frame-sync
    // 배경/렌즈 1프레임 스큐 제거.

    /** frame-sync 킬스위치 토글 (UI 스레드 → GL 스레드에서 강등 상태 리셋). */
    fun setFrameSyncEnabled(enabled: Boolean) {
        queueEvent {
            glRenderer.setFrameSyncEnabled(enabled)
        }
    }


    // 펜딩 SurfaceRequest (최신 1건) — GL 컨텍스트 (재)생성 후 새 surface로 충족한다.
    // GL 스레드(fulfill)·메인 스레드(getSurfaceProvider)·surfaceExecutor(취소 리스너)가 접근 → 잠금 보호.
    @Volatile private var pendingSurfaceRequest: SurfaceRequest? = null
    private val surfaceLock = Any()

    // Executor for surface release
    private val surfaceExecutor = Executors.newSingleThreadExecutor()

    init {
        // OpenGL ES 3.1 설정
        setEGLContextClientVersion(3)

        // 투명 배경
        setEGLConfigChooser(8, 8, 8, 8, 16, 0)
        holder.setFormat(android.graphics.PixelFormat.TRANSLUCENT)

        // 백그라운드 복귀 시 EGL 컨텍스트 파괴를 가능한 기기에서 회피(주 완화책).
        // 단 보존은 드라이버 의존이라 보장되지 않으므로, 컨텍스트 손실 복원 경로
        // (아래 surface 핸드오프 + CameraGLRenderer.onSurfaceCreated의 stale 핸들 리셋 +
        //  GpuRenderActivity.onGpuInitialized의 렌즈/env_map 재적용)와 반드시 함께 동작한다.
        preserveEGLContextOnPause = true

        // 렌더러 설정
        setRenderer(glRenderer)
        renderMode = RENDERMODE_CONTINUOUSLY

        // SurfaceTexture 콜백 — EGL 컨텍스트가 (재)생성될 때마다 새 SurfaceTexture가 만들어진다.
        // 컨텍스트가 보존되면 호출되지 않는다. 구 컨텍스트 surface를 정리·교체하고,
        // 대기 중인 CameraX 요청을 새 surface로 충족한다.
        glRenderer.onSurfaceTextureAvailable = { surfaceTexture ->
            Log.d(TAG, "SurfaceTexture available from GL")
            // 구 컨텍스트의 surface는 stale — 정리. 컨텍스트 재생성 시점엔 CameraX가 재bind 과정에서
            // 구 request를 이미 취소했으므로 release가 BufferQueue를 abandon시킬 위험이 없다.
            cameraSurface?.let { old -> runCatching { old.release() } }
            cameraSurfaceTexture = surfaceTexture
            cameraSurface = Surface(surfaceTexture)
            fulfillPendingSurfaceRequest()
        }
    }

    /**
     * CameraX Preview SurfaceProvider 반환
     *
     * CameraX Preview.setSurfaceProvider()에 전달하여 카메라 연결
     */
    fun getSurfaceProvider(): Preview.SurfaceProvider {
        return Preview.SurfaceProvider { request ->
            Log.d(TAG, "SurfaceProvider received request: ${request.resolution}")
            synchronized(surfaceLock) {
                // 한 request엔 provideSurface/willNotProvideSurface 정확히 1회만 허용 —
                // 직전 미충족 pending이 있으면 willNotProvideSurface로 정리한다(재bind 시 중복 방지).
                pendingSurfaceRequest?.let { old ->
                    if (old !== request) runCatching { old.willNotProvideSurface() }
                }
                pendingSurfaceRequest = request
            }
            // CameraX가 이 request를 취소하면(재bind/언바인드 등) 참조를 정리한다.
            request.addRequestCancellationListener(surfaceExecutor) {
                synchronized(surfaceLock) {
                    if (pendingSurfaceRequest === request) pendingSurfaceRequest = null
                }
            }
            // 충족은 GL 스레드에서만 — '현재 유효한' cameraSurface(컨텍스트 보존 또는 재생성 후)로만
            // 제공해 구(파괴된 컨텍스트의) stale surface 제공을 원천 차단한다. GL 스레드가 일시정지
            // 상태(백그라운드)면 복귀 후 onSurfaceCreated→fulfill 또는 이 큐 이벤트가 실행되어 충족된다.
            queueEvent { fulfillPendingSurfaceRequest() }
        }
    }

    /**
     * 대기 중인 CameraX SurfaceRequest를 현재 유효한 cameraSurface로 충족한다.
     *
     * GL 스레드(onSurfaceTextureAvailable / getSurfaceProvider의 queueEvent)에서 호출된다.
     * pendingSurfaceRequest·cameraSurface 캡처는 잠금으로 원자화하고, 실제 provideSurface는
     * 잠금 밖에서 수행한다.
     */
    private fun fulfillPendingSurfaceRequest() {
        val request: SurfaceRequest
        val surface: Surface
        synchronized(surfaceLock) {
            request = pendingSurfaceRequest ?: return
            surface = cameraSurface ?: return
            pendingSurfaceRequest = null
        }

        val resolution = request.resolution
        cameraSurfaceTexture?.setDefaultBufferSize(resolution.width, resolution.height)
        // GL 스레드에서 프레임 크기 설정 (FBO 재생성 포함)
        queueEvent {
            glRenderer.setFrameSize(resolution.width, resolution.height)
        }

        Log.d(TAG, "Providing surface: ${resolution.width}x${resolution.height}")
        runCatching {
            request.provideSurface(surface, surfaceExecutor) { result ->
                Log.d(TAG, "Surface result: ${result.resultCode}")
            }
        }.onFailure { Log.e(TAG, "provideSurface failed: ${it.message}") }
    }

    //=========================================================================
    // 설정 API
    //=========================================================================

    /**
     * 뷰티 필터 설정
     */
    fun setBeautyConfig(config: BeautyFilterConfigV2) {
        queueEvent {
            glRenderer.setBeautyConfig(config)
        }
    }

    /**
     * 뷰티 필터 활성화/비활성화
     */
    fun setBeautyEnabled(enabled: Boolean) {
        queueEvent {
            glRenderer.setBeautyEnabled(enabled)
        }
    }

    /**
     * 미러링 설정 (전면 카메라)
     */
    fun setMirror(mirror: Boolean) {
        queueEvent {
            glRenderer.setMirror(mirror)
        }
    }

    /**
     * 홍채 검출 결과 설정 (렌즈 오버레이용)
     *
     * 주의: 호출자는 이후 변경하지 않을 인스턴스(프레임별 새 복사본)를 넘겨야 한다 —
     * 공유 가변 인스턴스 재사용은 GL 스레드 torn read를 유발한다 (감사 finding).
     */
    fun setIrisResult(result: IrisResult?) {
        queueEvent {
            glRenderer.setIrisResult(result)
        }
    }

    /** SDK 렌즈 렌더 실패 사유 (null = 정상) — HUD 표시용, 임의 스레드에서 읽기 가능. */
    fun getSdkLensFailure(): String? = glRenderer.sdkLensFailure

    /**
     * 카메라 회전 설정 (0, 90, 180, 270)
     */
    fun setFrameRotation(rotation: Int) {
        queueEvent {
            glRenderer.setFrameRotation(rotation)
        }
    }

    /**
     * 렌즈 설정
     */
    fun setLensConfig(config: LensConfig) {
        queueEvent {
            glRenderer.setLensConfig(config)
        }
    }

    /**
     * 렌즈 활성화/비활성화
     */
    fun setLensEnabled(enabled: Boolean) {
        queueEvent {
            glRenderer.setLensEnabled(enabled)
        }
    }

    /**
     * 렌즈 텍스처 설정 (비트맵)
     *
     * @param skuId P6-W7: 렌즈 SKU id (메타 연동, 빈 문자열이면 미적용)
     */
    fun setLensTexture(bitmap: Bitmap?, skuId: String = "") {
        queueEvent {
            glRenderer.setLensTexture(bitmap, skuId)
        }
    }

    /**
     * P6-W4 §5.7: 환경 반사 env_map 텍스처 로드 (RGB 8bit).
     * GL 스레드에서 native loadEnvMap 호출.
     */
    fun setEnvMap(rgbData: ByteArray, width: Int, height: Int) {
        queueEvent {
            val result = com.irislenssdk.IrisLensSDK.loadEnvMap(rgbData, width, height)
            android.util.Log.i("CameraGLView", "EnvMap load: ${width}x${height}, result=$result")
        }
    }

    /**
     * P6-W4 §5.11: 환경 반사 모드 토글 (0=OFF, 1=EnvMap, 2=Periphery).
     */
    fun setReflectionMode(mode: Int) {
        queueEvent {
            com.irislenssdk.IrisLensSDK.setReflectionMode(mode)
        }
    }

    /**
     * P6-W4 §5.7: 환경 반사 강도 (0.0~1.0, W3 기본 0.3).
     */
    fun setReflectionIntensity(intensity: Float) {
        queueEvent {
            com.irislenssdk.IrisLensSDK.setReflectionIntensity(intensity)
        }
    }

    /**
     * P6-W5 §5.9: sclera veto 수식 토글 (B1/B8 4조합 벤치용).
     * @param mode 0=legacy, 1=color-veto(Codex), 2=luma-only(Gemini)
     */
    fun setScleraVetoMode(mode: Int) {
        queueEvent {
            com.irislenssdk.IrisLensSDK.setScleraVetoMode(mode)
        }
    }

    /**
     * P6-W6 B5: 블링크 up ramp 시간 토글 (60/80/120ms).
     */
    fun setBlinkUpMs(ms: Float) {
        queueEvent {
            com.irislenssdk.IrisLensSDK.setBlinkUpMs(ms)
        }
    }

    /**
     * P6-W6 B9: 저조도 디테일 gate 임계값 토글 (0.10/0.15/0.25).
     */
    fun setGateThreshold(threshold: Float) {
        queueEvent {
            com.irislenssdk.IrisLensSDK.setGateThreshold(threshold)
        }
    }

    /**
     * P6-W6 C10: 홍채 디테일 재주입 on/off.
     */
    fun setDetailReinject(enabled: Boolean) {
        queueEvent {
            com.irislenssdk.IrisLensSDK.setDetailReinject(enabled)
        }
    }

    /**
     * P7-W2 §5.6: avg_iris_luma fallback(false) ↔ 실측(true) A/B 토글.
     * 실측 ON 시 SDK가 detector 측정값으로 uAvgIrisLum을 구동(블렌드 정규화 + gate).
     */
    fun setUseMeasuredLuma(enabled: Boolean) {
        queueEvent {
            com.irislenssdk.IrisLensSDK.setUseMeasuredLuma(enabled)
        }
    }

    /**
     * P8-W1: landmark-masked skin smoothing 토글 (FreqSep 대체 A/B).
     * 뷰티 활성(beautyEnabled) 상태에서만 시각 효과 발생.
     */
    fun setSkinMaskSmoothing(enabled: Boolean, strength: Float) {
        queueEvent {
            com.irislenssdk.IrisLensSDK.setSkinMaskSmoothing(enabled, strength)
        }
    }

    /**
     * P8-W3: 피부 화사함(soft-glow radiance) 토글.
     * skin smoothing 경로(블러+마스크)를 공유한다 — smoothing=0이어도 radiance 단독 적용.
     * 뷰티 활성(beautyEnabled) 상태에서만 시각 효과 발생.
     */
    fun setSkinRadiance(strength: Float) {
        queueEvent {
            com.irislenssdk.IrisLensSDK.setSkinRadiance(strength)
        }
    }

    /**
     * Sclera Protection 활성화/비활성화 (P4-W2-01)
     */
    fun setScleraProtect(enabled: Boolean) {
        queueEvent {
            glRenderer.setScleraProtect(enabled)
        }
    }

    /**
     * Contact Shadow 활성화/비활성화 (P4-W2-01)
     */
    fun setContactShadow(enabled: Boolean, intensity: Float = 0.15f) {
        queueEvent {
            glRenderer.setContactShadow(enabled, intensity)
        }
    }

    /**
     * Color Replace 홍채 밝기 보정 상한 설정 (P4-W2-01)
     */
    fun setMaxDetail(value: Float) {
        queueEvent {
            glRenderer.setMaxDetail(value)
        }
    }

    /**
     * 비대칭 타원 Eye Mask 활성화/비활성화 (P4-W2-02 → EYECLIP A-1 배선).
     * native GPULensRenderer(use_ellipse_mask_)를 직접 구동 — OFF 시 기존 Y-slab 마스킹.
     */
    fun setEllipseMask(enabled: Boolean) {
        queueEvent {
            com.irislenssdk.IrisLensSDK.setLensEllipseMask(enabled)
        }
    }

    /**
     * EYECLIP A-2: 눈꺼풀 마스크 모드 (0=Y-slab, 1=ellipse, 2=contour). GL 스레드 마샬링.
     */
    fun setEyelidMaskMode(mode: Int) {
        queueEvent {
            com.irislenssdk.IrisLensSDK.setLensEyelidMaskMode(mode)
        }
    }

    /**
     * 리소스 해제
     */
    fun release() {
        queueEvent {
            glRenderer.release()
        }

        cameraSurface?.release()
        cameraSurface = null
        cameraSurfaceTexture = null

        surfaceExecutor.shutdown()

        Log.d(TAG, "CameraGLView released")
    }

    override fun onDetachedFromWindow() {
        super.onDetachedFromWindow()
        release()
    }
}
