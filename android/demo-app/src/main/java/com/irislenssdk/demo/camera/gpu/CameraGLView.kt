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
import android.graphics.SurfaceTexture
import android.opengl.GLSurfaceView
import android.util.AttributeSet
import android.util.Log
import android.view.Surface
import androidx.camera.core.Preview
import androidx.camera.core.SurfaceRequest
import com.irislenssdk.BeautyFilterConfigV2
import com.irislenssdk.IrisResult
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

    // 초기화 상태
    private var isGLInitialized = false

    // GPU 초기화 콜백
    var onGpuInitialized: ((Boolean) -> Unit)? = null
        set(value) {
            field = value
            glRenderer.onGpuInitialized = value
        }

    // 펜딩 SurfaceRequest (GL 초기화 전 Preview.setSurfaceProvider 호출 시)
    private var pendingSurfaceRequest: SurfaceRequest? = null

    // Executor for surface release
    private val surfaceExecutor = Executors.newSingleThreadExecutor()

    init {
        // OpenGL ES 3.1 설정
        setEGLContextClientVersion(3)

        // 투명 배경
        setEGLConfigChooser(8, 8, 8, 8, 16, 0)
        holder.setFormat(android.graphics.PixelFormat.TRANSLUCENT)

        // 렌더러 설정
        setRenderer(glRenderer)
        renderMode = RENDERMODE_CONTINUOUSLY

        // SurfaceTexture 콜백 설정
        glRenderer.onSurfaceTextureAvailable = { surfaceTexture ->
            Log.d(TAG, "SurfaceTexture available from GL")
            cameraSurfaceTexture = surfaceTexture
            cameraSurface = Surface(surfaceTexture)
            isGLInitialized = true

            // 펜딩 요청 처리
            pendingSurfaceRequest?.let { request ->
                provideSurfaceToRequest(request)
                pendingSurfaceRequest = null
            }
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

            if (isGLInitialized && cameraSurface != null) {
                provideSurfaceToRequest(request)
            } else {
                // GL 초기화 전이면 펜딩
                Log.d(TAG, "GL not initialized, pending request")
                pendingSurfaceRequest = request
            }
        }
    }

    /**
     * SurfaceRequest에 Surface 제공
     */
    private fun provideSurfaceToRequest(request: SurfaceRequest) {
        val surface = cameraSurface ?: run {
            Log.e(TAG, "Surface is null")
            return
        }

        // SurfaceTexture 크기 설정
        val resolution = request.resolution
        cameraSurfaceTexture?.setDefaultBufferSize(resolution.width, resolution.height)

        // GL 스레드에서 프레임 크기 설정 (FBO 재생성 포함)
        queueEvent {
            glRenderer.setFrameSize(resolution.width, resolution.height)
        }

        Log.d(TAG, "Providing surface: ${resolution.width}x${resolution.height}")

        request.provideSurface(surface, surfaceExecutor) { result ->
            Log.d(TAG, "Surface result: ${result.resultCode}")
        }
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
     */
    fun setIrisResult(result: IrisResult?) {
        queueEvent {
            glRenderer.setIrisResult(result)
        }
    }

    /**
     * 카메라 회전 설정 (0, 90, 180, 270)
     */
    fun setFrameRotation(rotation: Int) {
        queueEvent {
            glRenderer.setFrameRotation(rotation)
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
