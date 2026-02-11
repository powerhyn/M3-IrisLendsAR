/**
 * IrisLensSDK Android - CameraGLRenderer
 *
 * OpenGL ES 3.1 기반 카메라 렌더러
 * - SurfaceTexture → OpenGL 텍스처 변환
 * - GPU Beauty Backend 연동
 * - GPU 기반 렌즈 오버레이
 *
 * 하이브리드 아키텍처:
 * - MediaPipe 추론: CPU (기존 FrameAnalyzer)
 * - 렌더링: GPU (이 클래스)
 *
 * @version 1.0.0
 */
package com.irislenssdk.demo.camera.gpu

import android.graphics.Bitmap
import android.graphics.SurfaceTexture
import android.opengl.GLES11Ext
import android.opengl.GLES31
import android.opengl.GLSurfaceView
import android.opengl.GLUtils
import android.opengl.Matrix
import android.util.Log
import com.irislenssdk.BeautyFilterConfigV2
import com.irislenssdk.IrisLensSDK
import com.irislenssdk.IrisResult
import com.irislenssdk.LensConfig
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

/**
 * GPU 기반 카메라 렌더러
 *
 * CameraX SurfaceTexture → OpenGL 텍스처 → GPU Beauty Filter → 화면 출력
 */
class CameraGLRenderer : GLSurfaceView.Renderer {

    companion object {
        private const val TAG = "CameraGLRenderer"

        // 풀스크린 쿼드 좌표 (NDC + 텍스처 좌표)
        // SurfaceTexture.getTransformMatrix()가 필요한 변환을 포함하므로
        // 텍스처 좌표는 표준 좌표 사용
        private val FULLSCREEN_QUAD = floatArrayOf(
            // Position (x, y)    // TexCoord (s, t)
            -1.0f, -1.0f,         0.0f, 0.0f,
             1.0f, -1.0f,         1.0f, 0.0f,
            -1.0f,  1.0f,         0.0f, 1.0f,
             1.0f,  1.0f,         1.0f, 1.0f
        )

        // OES 텍스처 → 2D 텍스처 변환 버텍스 셰이더
        private const val VERTEX_SHADER = """
            #version 310 es
            layout(location = 0) in vec2 aPosition;
            layout(location = 1) in vec2 aTexCoord;

            uniform mat4 uSTMatrix;  // SurfaceTexture 변환 행렬
            uniform int uMirror;     // 미러링 (전면 카메라)
            uniform int uFlipY;      // Y축 뒤집기
            uniform vec2 uScale;     // Aspect ratio 보정 스케일

            out vec2 vTexCoord;

            void main() {
                vec2 pos = aPosition;
                // Y축 뒤집기 (상하 반전)
                if (uFlipY == 1) {
                    pos.y = -pos.y;
                }
                // Aspect ratio 보정 스케일 적용
                pos *= uScale;
                gl_Position = vec4(pos, 0.0, 1.0);

                // SurfaceTexture 변환 적용
                vec4 texCoord = uSTMatrix * vec4(aTexCoord, 0.0, 1.0);

                // 미러링 (전면 카메라)
                if (uMirror == 1) {
                    texCoord.x = 1.0 - texCoord.x;
                }

                vTexCoord = texCoord.xy;
            }
        """

        // OES 텍스처 샘플링 프래그먼트 셰이더
        private const val OES_TO_2D_FRAGMENT_SHADER = """
            #version 310 es
            #extension GL_OES_EGL_image_external_essl3 : require
            precision highp float;

            uniform samplerExternalOES uOESTexture;

            in vec2 vTexCoord;
            out vec4 fragColor;

            void main() {
                fragColor = texture(uOESTexture, vTexCoord);
            }
        """

        // 패스스루 프래그먼트 셰이더 (2D 텍스처)
        private const val PASSTHROUGH_FRAGMENT_SHADER = """
            #version 310 es
            precision highp float;

            uniform sampler2D uTexture;

            in vec2 vTexCoord;
            out vec4 fragColor;

            void main() {
                fragColor = texture(uTexture, vTexCoord);
            }
        """

        // 렌즈 오버레이 프래그먼트 셰이더
        private const val LENS_OVERLAY_FRAGMENT_SHADER = """
            #version 310 es
            precision highp float;

            uniform sampler2D uCameraTexture;
            uniform sampler2D uLensTexture;

            // 왼쪽 눈 파라미터
            uniform vec2 uLeftIrisCenter;   // 정규화된 좌표 (0~1)
            uniform float uLeftIrisRadius;  // 정규화된 반경

            // 오른쪽 눈 파라미터
            uniform vec2 uRightIrisCenter;  // 정규화된 좌표 (0~1)
            uniform float uRightIrisRadius; // 정규화된 반경

            // 렌즈 설정
            uniform float uOpacity;         // 투명도 (0~1)
            uniform float uLensScale;       // 크기 배율 (uScale은 vertex shader에서 사용됨)
            uniform float uEdgeFeather;     // 가장자리 페더링
            uniform int uBlendMode;         // 블렌드 모드 (0=Normal, 1=Multiply, 2=Screen, 3=Overlay)
            uniform int uApplyLeft;         // 왼쪽 눈 적용 여부
            uniform int uApplyRight;        // 오른쪽 눈 적용 여부
            uniform float uFrameAspect;     // 프레임 비율 (width / height)

            // 눈꺼풀 클리핑용 (정규화 좌표 0~1)
            uniform float uLeftEyeTop;      // 왼쪽 눈 상단 Y
            uniform float uLeftEyeBottom;   // 왼쪽 눈 하단 Y
            uniform float uRightEyeTop;     // 오른쪽 눈 상단 Y
            uniform float uRightEyeBottom;  // 오른쪽 눈 하단 Y

            in vec2 vTexCoord;
            out vec4 fragColor;

            // 블렌드 함수들
            vec3 blendNormal(vec3 base, vec3 blend, float opacity) {
                return mix(base, blend, opacity);
            }

            vec3 blendMultiply(vec3 base, vec3 blend, float opacity) {
                return mix(base, base * blend, opacity);
            }

            vec3 blendScreen(vec3 base, vec3 blend, float opacity) {
                return mix(base, 1.0 - (1.0 - base) * (1.0 - blend), opacity);
            }

            vec3 blendOverlay(vec3 base, vec3 blend, float opacity) {
                vec3 result;
                for (int i = 0; i < 3; i++) {
                    if (base[i] < 0.5) {
                        result[i] = 2.0 * base[i] * blend[i];
                    } else {
                        result[i] = 1.0 - 2.0 * (1.0 - base[i]) * (1.0 - blend[i]);
                    }
                }
                return mix(base, result, opacity);
            }

            // 렌즈 합성 함수 (눈꺼풀 클리핑 포함)
            vec4 applyLens(vec4 camera, vec2 irisCenter, float irisRadius, float aspectRatio, float eyeTop, float eyeBottom) {
                if (irisRadius <= 0.0) return camera;

                // 원형 유지를 위한 좌표 보정 (x를 aspectRatio로 스케일)
                vec2 adjustedCoord = vec2(vTexCoord.x * aspectRatio, vTexCoord.y);
                vec2 adjustedCenter = vec2(irisCenter.x * aspectRatio, irisCenter.y);

                // 홍채 중심으로부터의 거리 계산 (스케일 적용)
                float scaledRadius = irisRadius * uLensScale;
                float dist = distance(adjustedCoord, adjustedCenter) / scaledRadius;

                if (dist >= 1.0) return camera;

                // 렌즈 텍스처 좌표 계산 (홍채 영역을 렌즈 전체에 매핑)
                vec2 lensCoord = (adjustedCoord - adjustedCenter) / scaledRadius * 0.5 + 0.5;

                // 렌즈 텍스처 샘플링
                vec4 lens = texture(uLensTexture, lensCoord);

                // 가장자리 페더링 (부드러운 경계)
                float featherStart = 1.0 - uEdgeFeather;
                float edgeAlpha = smoothstep(1.0, featherStart, dist);

                // 눈꺼풀 클리핑 (Y축 뒤집힘 고려: top < bottom after flip)
                float eyelidFeather = 0.015;  // 눈꺼풀 경계 페더링
                float minY = min(eyeTop, eyeBottom);
                float maxY = max(eyeTop, eyeBottom);
                float topClip = smoothstep(minY - eyelidFeather, minY + eyelidFeather, vTexCoord.y);
                float bottomClip = smoothstep(maxY + eyelidFeather, maxY - eyelidFeather, vTexCoord.y);
                float eyelidMask = topClip * bottomClip;

                // 최종 알파 계산 (눈꺼풀 마스크 적용)
                float finalAlpha = lens.a * uOpacity * edgeAlpha * eyelidMask;

                // 블렌드 모드에 따른 합성
                vec3 blended;
                if (uBlendMode == 0) {
                    blended = blendNormal(camera.rgb, lens.rgb, finalAlpha);
                } else if (uBlendMode == 1) {
                    blended = blendMultiply(camera.rgb, lens.rgb, finalAlpha);
                } else if (uBlendMode == 2) {
                    blended = blendScreen(camera.rgb, lens.rgb, finalAlpha);
                } else {
                    blended = blendOverlay(camera.rgb, lens.rgb, finalAlpha);
                }

                return vec4(blended, camera.a);
            }

            void main() {
                vec4 camera = texture(uCameraTexture, vTexCoord);
                vec4 result = camera;

                // 화면 비율 보정 (프레임 width/height)
                float aspectRatio = uFrameAspect;

                // 왼쪽 눈 렌즈 적용
                if (uApplyLeft == 1 && uLeftIrisRadius > 0.0) {
                    result = applyLens(result, uLeftIrisCenter, uLeftIrisRadius, aspectRatio, uLeftEyeTop, uLeftEyeBottom);
                }

                // 오른쪽 눈 렌즈 적용
                if (uApplyRight == 1 && uRightIrisRadius > 0.0) {
                    result = applyLens(result, uRightIrisCenter, uRightIrisRadius, aspectRatio, uRightEyeTop, uRightEyeBottom);
                }

                fragColor = result;
            }
        """
    }

    // SurfaceTexture (카메라 출력)
    private var surfaceTexture: SurfaceTexture? = null
    private var oesTextureId: Int = 0
    private val stMatrix = FloatArray(16)

    // 셰이더 프로그램
    private var oesToRgbProgram: Int = 0
    private var passthroughProgram: Int = 0
    private var lensProgram: Int = 0

    // Uniform locations (OES → RGBA)
    private var uSTMatrixLocation: Int = -1
    private var uMirrorLocation: Int = -1
    private var uFlipYLocation: Int = -1
    private var uScaleLocation: Int = -1
    private var uOESTextureLocation: Int = -1
    private var uTextureLocation: Int = -1

    // Uniform locations (렌즈 셰이더)
    private var uLensCameraTextureLocation: Int = -1
    private var uLensTextureLocation: Int = -1
    private var uLeftIrisCenterLocation: Int = -1
    private var uLeftIrisRadiusLocation: Int = -1
    private var uRightIrisCenterLocation: Int = -1
    private var uRightIrisRadiusLocation: Int = -1
    private var uLensOpacityLocation: Int = -1
    private var uLensScaleLocation: Int = -1
    private var uLensEdgeFeatherLocation: Int = -1
    private var uLensBlendModeLocation: Int = -1
    private var uApplyLeftLocation: Int = -1
    private var uApplyRightLocation: Int = -1
    private var uFrameAspectLocation: Int = -1
    private var uLeftEyeTopLocation: Int = -1
    private var uLeftEyeBottomLocation: Int = -1
    private var uRightEyeTopLocation: Int = -1
    private var uRightEyeBottomLocation: Int = -1

    // 풀스크린 쿼드 VAO/VBO
    private var quadVao: Int = 0
    private var quadVbo: Int = 0

    // 중간 텍스처/FBO (OES → RGBA 변환용)
    private var rgbaTextureId: Int = 0
    private var rgbaFboId: Int = 0

    // 뷰티 필터 출력 텍스처
    private var beautyOutputTextureId: Int = 0

    // 렌즈 FBO/텍스처
    private var lensFboId: Int = 0
    private var lensOutputTextureId: Int = 0

    // 렌즈 텍스처 (렌즈 이미지)
    private var lensImageTextureId: Int = 0
    private var pendingLensBitmap: Bitmap? = null

    // LUT 필터 (C++ Combined Color Pass로 통합 - 3D 텍스처만 관리)
    private var lut3dTextureId: Int = 0
    private var lutEnabled: Boolean = false
    private var lutIntensity: Float = 1.0f
    private var pendingLut3dTextureId: Int = -1  // -1 = no pending

    // 화면 크기
    private var viewWidth: Int = 0
    private var viewHeight: Int = 0

    // 카메라 프레임 크기
    private var frameWidth: Int = 0
    private var frameHeight: Int = 0
    private var frameRotation: Int = 0  // 카메라 회전 각도 (0, 90, 180, 270)

    // 상태
    private var isInitialized: Boolean = false
    private var isMirror: Boolean = true

    // 뷰티 필터 설정
    private var beautyConfig: BeautyFilterConfigV2 = BeautyFilterConfigV2()
    private var beautyEnabled: Boolean = false

    // 홍채 검출 결과 (렌즈 오버레이용)
    private var irisResult: IrisResult? = null

    // 렌즈 설정
    private var lensConfig: LensConfig = LensConfig()
    private var lensEnabled: Boolean = false

    // GPU FPS 측정
    private var gpuFrameCount = 0
    private var lastGpuFpsTime = System.nanoTime()
    private var currentGpuFps = 0

    // 콜백
    var onSurfaceTextureAvailable: ((SurfaceTexture) -> Unit)? = null
    var onGpuInitialized: ((Boolean) -> Unit)? = null
    var onGpuFpsUpdated: ((Int) -> Unit)? = null

    //=========================================================================
    // GLSurfaceView.Renderer 구현
    //=========================================================================

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        Log.d(TAG, "onSurfaceCreated")

        // OpenGL ES 버전 확인
        val version = GLES31.glGetString(GLES31.GL_VERSION)
        Log.d(TAG, "OpenGL ES version: $version")

        // 배경색 설정
        GLES31.glClearColor(0.0f, 0.0f, 0.0f, 1.0f)

        // 셰이더 프로그램 생성
        oesToRgbProgram = createProgram(VERTEX_SHADER, OES_TO_2D_FRAGMENT_SHADER)
        passthroughProgram = createProgram(VERTEX_SHADER, PASSTHROUGH_FRAGMENT_SHADER)
        lensProgram = createProgram(VERTEX_SHADER, LENS_OVERLAY_FRAGMENT_SHADER)

        // Uniform locations 캐시 (OES → RGBA)
        uSTMatrixLocation = GLES31.glGetUniformLocation(oesToRgbProgram, "uSTMatrix")
        uMirrorLocation = GLES31.glGetUniformLocation(oesToRgbProgram, "uMirror")
        uFlipYLocation = GLES31.glGetUniformLocation(oesToRgbProgram, "uFlipY")
        uScaleLocation = GLES31.glGetUniformLocation(oesToRgbProgram, "uScale")
        uOESTextureLocation = GLES31.glGetUniformLocation(oesToRgbProgram, "uOESTexture")
        uTextureLocation = GLES31.glGetUniformLocation(passthroughProgram, "uTexture")

        // Uniform locations 캐시 (렌즈 셰이더)
        uLensCameraTextureLocation = GLES31.glGetUniformLocation(lensProgram, "uCameraTexture")
        uLensTextureLocation = GLES31.glGetUniformLocation(lensProgram, "uLensTexture")
        uLeftIrisCenterLocation = GLES31.glGetUniformLocation(lensProgram, "uLeftIrisCenter")
        uLeftIrisRadiusLocation = GLES31.glGetUniformLocation(lensProgram, "uLeftIrisRadius")
        uRightIrisCenterLocation = GLES31.glGetUniformLocation(lensProgram, "uRightIrisCenter")
        uRightIrisRadiusLocation = GLES31.glGetUniformLocation(lensProgram, "uRightIrisRadius")
        uLensOpacityLocation = GLES31.glGetUniformLocation(lensProgram, "uOpacity")
        uLensScaleLocation = GLES31.glGetUniformLocation(lensProgram, "uLensScale")
        uLensEdgeFeatherLocation = GLES31.glGetUniformLocation(lensProgram, "uEdgeFeather")
        uLensBlendModeLocation = GLES31.glGetUniformLocation(lensProgram, "uBlendMode")
        uApplyLeftLocation = GLES31.glGetUniformLocation(lensProgram, "uApplyLeft")
        uApplyRightLocation = GLES31.glGetUniformLocation(lensProgram, "uApplyRight")
        uFrameAspectLocation = GLES31.glGetUniformLocation(lensProgram, "uFrameAspect")
        uLeftEyeTopLocation = GLES31.glGetUniformLocation(lensProgram, "uLeftEyeTop")
        uLeftEyeBottomLocation = GLES31.glGetUniformLocation(lensProgram, "uLeftEyeBottom")
        uRightEyeTopLocation = GLES31.glGetUniformLocation(lensProgram, "uRightEyeTop")
        uRightEyeBottomLocation = GLES31.glGetUniformLocation(lensProgram, "uRightEyeBottom")

        // 풀스크린 쿼드 설정
        setupFullscreenQuad()

        // OES 텍스처 생성 (카메라 입력)
        oesTextureId = createOESTexture()

        // SurfaceTexture 생성 및 콜백 호출
        surfaceTexture = SurfaceTexture(oesTextureId).apply {
            setOnFrameAvailableListener { /* 새 프레임 대기 */ }
        }

        // GPU Beauty Backend 초기화
        val gpuInitResult = IrisLensSDK.initGpuBeauty()
        val gpuSuccess = (gpuInitResult == IrisLensSDK.OK || gpuInitResult == IrisLensSDK.ALREADY_INITIALIZED)
        Log.d(TAG, "GPU Beauty Backend init: $gpuInitResult (success: $gpuSuccess)")

        isInitialized = true

        // 콜백으로 SurfaceTexture 전달 (카메라 연결용)
        surfaceTexture?.let { onSurfaceTextureAvailable?.invoke(it) }

        // GPU 초기화 결과 콜백
        onGpuInitialized?.invoke(gpuSuccess)
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        Log.d(TAG, "onSurfaceChanged: ${width}x${height}")

        viewWidth = width
        viewHeight = height

        GLES31.glViewport(0, 0, width, height)

        // 중간 텍스처/FBO 재생성 (크기 변경)
        recreateIntermediateBuffers(width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        if (!isInitialized) return

        // SurfaceTexture 업데이트
        surfaceTexture?.updateTexImage()
        surfaceTexture?.getTransformMatrix(stMatrix)

        // 펜딩 렌즈 텍스처 업로드
        uploadPendingLensTexture()

        // 화면 클리어
        GLES31.glClear(GLES31.GL_COLOR_BUFFER_BIT)

        // 1단계: OES 텍스처 → RGBA 텍스처 변환
        renderOESToRgba()

        // 2단계: 렌즈 오버레이 (홍채 위치에 렌즈 합성)
        var currentTexture = rgbaTextureId
        if (lensEnabled && lensImageTextureId != 0 && irisResult?.detected == true) {
            currentTexture = renderLensOverlay(currentTexture)
        }

        // 3단계: 펜딩 LUT 3D 텍스처 적용 (beauty 호출 전 준비)
        uploadPendingLut3dTexture()

        // 4단계: GPU Beauty + LUT 통합 적용 (C++ Combined Color Pass에서 LUT 포함)
        val beautyApplied = beautyEnabled && beautyConfig.enabled
        var outputTexture = if (beautyApplied) {
            applyGpuBeautyFilter(currentTexture)
        } else {
            currentTexture
        }

        // 5단계: 화면에 렌더링
        renderToScreen(outputTexture, beautyApplied)

        // GPU FPS 측정
        updateGpuFps()

        checkGlError("onDrawFrame")
    }

    //=========================================================================
    // GPU FPS 측정
    //=========================================================================

    /**
     * GPU 렌더링 FPS 측정 (GL 스레드에서 호출)
     */
    private fun updateGpuFps() {
        gpuFrameCount++
        val now = System.nanoTime()
        val elapsed = now - lastGpuFpsTime
        if (elapsed >= 1_000_000_000L) {  // 1초 경과
            currentGpuFps = gpuFrameCount
            gpuFrameCount = 0
            lastGpuFpsTime = now
            onGpuFpsUpdated?.invoke(currentGpuFps)
        }
    }

    /**
     * 현재 GPU FPS 반환
     */
    fun getGpuFps(): Int = currentGpuFps

    //=========================================================================
    // 렌더링 단계
    //=========================================================================

    /**
     * OES 텍스처를 RGBA 2D 텍스처로 변환
     */
    private fun renderOESToRgba() {
        // FBO 바인딩
        GLES31.glBindFramebuffer(GLES31.GL_FRAMEBUFFER, rgbaFboId)
        // 프레임 크기 사용 (FBO 텍스처 크기와 일치)
        val fboWidth = if (frameWidth > 0) frameWidth else viewWidth
        val fboHeight = if (frameHeight > 0) frameHeight else viewHeight
        GLES31.glViewport(0, 0, fboWidth, fboHeight)

        // OES → RGBA 셰이더 사용
        GLES31.glUseProgram(oesToRgbProgram)

        // SurfaceTexture 변환 행렬 설정
        GLES31.glUniformMatrix4fv(uSTMatrixLocation, 1, false, stMatrix, 0)
        GLES31.glUniform1i(uMirrorLocation, if (isMirror) 1 else 0)
        GLES31.glUniform1i(uFlipYLocation, 1)  // Y축 뒤집기 활성화
        GLES31.glUniform2f(uScaleLocation, 1.0f, 1.0f)  // FBO에는 전체 프레임 캡처

        // OES 텍스처 바인딩
        GLES31.glActiveTexture(GLES31.GL_TEXTURE0)
        GLES31.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, oesTextureId)
        GLES31.glUniform1i(uOESTextureLocation, 0)

        // 풀스크린 쿼드 렌더링
        renderFullscreenQuad()

        // FBO 언바인딩
        GLES31.glBindFramebuffer(GLES31.GL_FRAMEBUFFER, 0)
    }

    /**
     * 펜딩 렌즈 텍스처 업로드 (GL 스레드에서 실행)
     */
    private fun uploadPendingLensTexture() {
        val bitmap = pendingLensBitmap ?: return
        pendingLensBitmap = null

        // 기존 텍스처 삭제
        if (lensImageTextureId != 0) {
            GLES31.glDeleteTextures(1, intArrayOf(lensImageTextureId), 0)
        }

        // 새 텍스처 생성
        val textures = IntArray(1)
        GLES31.glGenTextures(1, textures, 0)
        lensImageTextureId = textures[0]

        GLES31.glBindTexture(GLES31.GL_TEXTURE_2D, lensImageTextureId)
        GLES31.glTexParameteri(GLES31.GL_TEXTURE_2D, GLES31.GL_TEXTURE_MIN_FILTER, GLES31.GL_LINEAR)
        GLES31.glTexParameteri(GLES31.GL_TEXTURE_2D, GLES31.GL_TEXTURE_MAG_FILTER, GLES31.GL_LINEAR)
        GLES31.glTexParameteri(GLES31.GL_TEXTURE_2D, GLES31.GL_TEXTURE_WRAP_S, GLES31.GL_CLAMP_TO_EDGE)
        GLES31.glTexParameteri(GLES31.GL_TEXTURE_2D, GLES31.GL_TEXTURE_WRAP_T, GLES31.GL_CLAMP_TO_EDGE)

        // 비트맵 업로드
        GLUtils.texImage2D(GLES31.GL_TEXTURE_2D, 0, bitmap, 0)

        GLES31.glBindTexture(GLES31.GL_TEXTURE_2D, 0)

        Log.d(TAG, "Lens texture uploaded: ${bitmap.width}x${bitmap.height}, id=$lensImageTextureId")
    }

    /**
     * 렌즈 오버레이 렌더링
     *
     * @param inputTexture 입력 텍스처 (카메라/뷰티 필터 출력)
     * @return 출력 텍스처 ID
     */
    private fun renderLensOverlay(inputTexture: Int): Int {
        val result = irisResult ?: return inputTexture

        // 렌즈 FBO가 없으면 생성
        if (lensFboId == 0 || lensOutputTextureId == 0) {
            createLensFbo()
            // FBO 생성 실패 시 입력 텍스처 반환
            if (lensFboId == 0 || lensOutputTextureId == 0) {
                Log.w(TAG, "Lens FBO creation failed, passing through")
                return inputTexture
            }
        }

        // FBO 바인딩
        GLES31.glBindFramebuffer(GLES31.GL_FRAMEBUFFER, lensFboId)

        val fboWidth = if (frameWidth > 0) frameWidth else viewWidth
        val fboHeight = if (frameHeight > 0) frameHeight else viewHeight
        GLES31.glViewport(0, 0, fboWidth, fboHeight)

        // 렌즈 셰이더 사용
        GLES31.glUseProgram(lensProgram)

        // 변환 행렬 설정 (단위 행렬 - 이미 변환 완료된 텍스처)
        val identityMatrix = FloatArray(16)
        Matrix.setIdentityM(identityMatrix, 0)
        val stLocation = GLES31.glGetUniformLocation(lensProgram, "uSTMatrix")
        val mirrorLocation = GLES31.glGetUniformLocation(lensProgram, "uMirror")
        val flipYLocation = GLES31.glGetUniformLocation(lensProgram, "uFlipY")
        val scaleLocation = GLES31.glGetUniformLocation(lensProgram, "uScale")
        GLES31.glUniformMatrix4fv(stLocation, 1, false, identityMatrix, 0)
        GLES31.glUniform1i(mirrorLocation, 0)
        GLES31.glUniform1i(flipYLocation, 0)
        GLES31.glUniform2f(scaleLocation, 1.0f, 1.0f)

        // 카메라 텍스처 바인딩 (unit 0)
        GLES31.glActiveTexture(GLES31.GL_TEXTURE0)
        GLES31.glBindTexture(GLES31.GL_TEXTURE_2D, inputTexture)
        GLES31.glUniform1i(uLensCameraTextureLocation, 0)

        // 렌즈 텍스처 바인딩 (unit 1)
        GLES31.glActiveTexture(GLES31.GL_TEXTURE1)
        GLES31.glBindTexture(GLES31.GL_TEXTURE_2D, lensImageTextureId)
        GLES31.glUniform1i(uLensTextureLocation, 1)

        // 홍채 위치/크기 설정 (IrisResult는 이미 정규화된 좌표 0~1)
        // leftRadius/rightRadius는 픽셀 단위이므로 정규화 필요
        val normalizedLeftRadius = result.leftRadius / fboWidth.toFloat()
        val normalizedRightRadius = result.rightRadius / fboWidth.toFloat()

        // 좌표 변환: renderOESToRgba()에서 적용한 변환과 동일하게 적용
        // 1. Y축 뒤집기 (uFlipY=1 적용됨)
        // 2. 미러링 (전면 카메라, uMirror=1 적용됨)
        var leftX = result.leftIrisX
        var leftY = result.leftIrisY
        var rightX = result.rightIrisX
        var rightY = result.rightIrisY

        // Y축 뒤집기
        leftY = 1.0f - leftY
        rightY = 1.0f - rightY

        // 미러링 (전면 카메라)
        if (isMirror) {
            leftX = 1.0f - leftX
            rightX = 1.0f - rightX
            // 미러링 시 좌/우 눈도 교환
            val tempX = leftX
            val tempY = leftY
            val tempRadius = normalizedLeftRadius
            leftX = rightX
            leftY = rightY
            rightX = tempX
            rightY = tempY
        }

        GLES31.glUniform2f(uLeftIrisCenterLocation, leftX, leftY)
        GLES31.glUniform1f(uLeftIrisRadiusLocation, if (isMirror) normalizedRightRadius else normalizedLeftRadius)
        GLES31.glUniform2f(uRightIrisCenterLocation, rightX, rightY)
        GLES31.glUniform1f(uRightIrisRadiusLocation, if (isMirror) normalizedLeftRadius else normalizedRightRadius)

        // 렌즈 설정 전달
        GLES31.glUniform1f(uLensOpacityLocation, lensConfig.opacity)
        GLES31.glUniform1f(uLensScaleLocation, lensConfig.scale)
        GLES31.glUniform1f(uLensEdgeFeatherLocation, lensConfig.edgeFeather)
        GLES31.glUniform1i(uLensBlendModeLocation, lensConfig.blendMode)
        GLES31.glUniform1i(uApplyLeftLocation, if (lensConfig.applyLeft) 1 else 0)
        GLES31.glUniform1i(uApplyRightLocation, if (lensConfig.applyRight) 1 else 0)

        // 프레임 비율 전달 (원형 렌즈를 위한 aspect ratio 보정)
        val frameAspect = fboWidth.toFloat() / fboHeight.toFloat()
        GLES31.glUniform1f(uFrameAspectLocation, frameAspect)

        // 눈꺼풀 클리핑 좌표 추출 (MediaPipe Face Mesh 랜드마크)
        // 랜드마크 인덱스: 왼쪽 눈 상단(159), 하단(145), 오른쪽 눈 상단(386), 하단(374)
        val faceMesh = result.faceMesh
        if (result.faceMeshValid && faceMesh != null) {
            // Y 좌표 추출 (인덱스 * 3 + 1 = y 좌표)
            var leftEyeTop = faceMesh[159 * 3 + 1]
            var leftEyeBottom = faceMesh[145 * 3 + 1]
            var rightEyeTop = faceMesh[386 * 3 + 1]
            var rightEyeBottom = faceMesh[374 * 3 + 1]

            // Y축 뒤집기 적용 (renderOESToRgba와 동일)
            leftEyeTop = 1.0f - leftEyeTop
            leftEyeBottom = 1.0f - leftEyeBottom
            rightEyeTop = 1.0f - rightEyeTop
            rightEyeBottom = 1.0f - rightEyeBottom

            // 미러링 시 좌/우 교환
            if (isMirror) {
                val tempTop = leftEyeTop
                val tempBottom = leftEyeBottom
                leftEyeTop = rightEyeTop
                leftEyeBottom = rightEyeBottom
                rightEyeTop = tempTop
                rightEyeBottom = tempBottom
            }

            GLES31.glUniform1f(uLeftEyeTopLocation, leftEyeTop)
            GLES31.glUniform1f(uLeftEyeBottomLocation, leftEyeBottom)
            GLES31.glUniform1f(uRightEyeTopLocation, rightEyeTop)
            GLES31.glUniform1f(uRightEyeBottomLocation, rightEyeBottom)
        } else {
            // faceMesh가 없으면 클리핑 비활성화 (전체 영역 허용)
            GLES31.glUniform1f(uLeftEyeTopLocation, 0.0f)
            GLES31.glUniform1f(uLeftEyeBottomLocation, 1.0f)
            GLES31.glUniform1f(uRightEyeTopLocation, 0.0f)
            GLES31.glUniform1f(uRightEyeBottomLocation, 1.0f)
        }

        // 풀스크린 쿼드 렌더링
        renderFullscreenQuad()

        // FBO 언바인딩
        GLES31.glBindFramebuffer(GLES31.GL_FRAMEBUFFER, 0)

        return lensOutputTextureId
    }

    /**
     * 렌즈 FBO 생성
     */
    private fun createLensFbo() {
        val width = if (frameWidth > 0) frameWidth else viewWidth
        val height = if (frameHeight > 0) frameHeight else viewHeight

        // 기존 버퍼 삭제
        if (lensOutputTextureId != 0) {
            GLES31.glDeleteTextures(1, intArrayOf(lensOutputTextureId), 0)
        }
        if (lensFboId != 0) {
            GLES31.glDeleteFramebuffers(1, intArrayOf(lensFboId), 0)
        }

        // 텍스처 생성
        val textures = IntArray(1)
        GLES31.glGenTextures(1, textures, 0)
        lensOutputTextureId = textures[0]

        GLES31.glBindTexture(GLES31.GL_TEXTURE_2D, lensOutputTextureId)
        GLES31.glTexImage2D(
            GLES31.GL_TEXTURE_2D, 0, GLES31.GL_RGBA,
            width, height, 0,
            GLES31.GL_RGBA, GLES31.GL_UNSIGNED_BYTE, null
        )
        GLES31.glTexParameteri(GLES31.GL_TEXTURE_2D, GLES31.GL_TEXTURE_MIN_FILTER, GLES31.GL_LINEAR)
        GLES31.glTexParameteri(GLES31.GL_TEXTURE_2D, GLES31.GL_TEXTURE_MAG_FILTER, GLES31.GL_LINEAR)
        GLES31.glTexParameteri(GLES31.GL_TEXTURE_2D, GLES31.GL_TEXTURE_WRAP_S, GLES31.GL_CLAMP_TO_EDGE)
        GLES31.glTexParameteri(GLES31.GL_TEXTURE_2D, GLES31.GL_TEXTURE_WRAP_T, GLES31.GL_CLAMP_TO_EDGE)

        // FBO 생성
        val fbos = IntArray(1)
        GLES31.glGenFramebuffers(1, fbos, 0)
        lensFboId = fbos[0]

        GLES31.glBindFramebuffer(GLES31.GL_FRAMEBUFFER, lensFboId)
        GLES31.glFramebufferTexture2D(
            GLES31.GL_FRAMEBUFFER, GLES31.GL_COLOR_ATTACHMENT0,
            GLES31.GL_TEXTURE_2D, lensOutputTextureId, 0
        )

        val status = GLES31.glCheckFramebufferStatus(GLES31.GL_FRAMEBUFFER)
        if (status != GLES31.GL_FRAMEBUFFER_COMPLETE) {
            Log.e(TAG, "Lens FBO is not complete: $status")
        }

        GLES31.glBindFramebuffer(GLES31.GL_FRAMEBUFFER, 0)

        Log.d(TAG, "Lens FBO created: ${width}x${height}")
    }

    /**
     * GPU 뷰티 필터 적용
     *
     * @param inputTexture 입력 텍스처 ID
     * @return 출력 텍스처 ID
     */
    private fun applyGpuBeautyFilter(inputTexture: Int): Int {
        // 프레임 크기 사용
        val texWidth = if (frameWidth > 0) frameWidth else viewWidth
        val texHeight = if (frameHeight > 0) frameHeight else viewHeight

        // LUT 파라미터 결정 (C++ Combined Color Pass에서 통합 처리)
        val lutTextureId = if (lutEnabled && lut3dTextureId != 0) lut3dTextureId else 0
        val lutIntensityVal = if (lutTextureId != 0) lutIntensity else 0.0f

        // 디버그: 뷰티+LUT 설정 확인
        Log.d(TAG, "Beauty filter call: enabled=${beautyConfig.enabled}, smoothing=${beautyConfig.smoothing}, brightness=${beautyConfig.brightness}, lut=$lutTextureId, lutIntensity=$lutIntensityVal")

        // GPU Beauty Backend 호출 (JNI) - LUT 통합
        val outputTexture = IrisLensSDK.applyBeautyFilterTextureV2(
            inputTexture,
            texWidth,
            texHeight,
            beautyConfig,
            lutTextureId,
            lutIntensityVal
        )

        // 디버그: 결과 확인
        Log.d(TAG, "Beauty filter result: input=$inputTexture, output=$outputTexture, size=${texWidth}x${texHeight}")

        return if (outputTexture != 0 && outputTexture != inputTexture) {
            // NOTE: 텍스처 해제는 C++ TexturePool에서 관리함
            // Android에서 releaseTexture() 호출하면 이중 해제 발생 → 검은 화면 원인
            // 이전 코드: IrisLensSDK.releaseTexture(beautyOutputTextureId) - 제거됨
            beautyOutputTextureId = outputTexture
            beautyOutputTextureId
        } else {
            // 필터 실패 또는 pass-through 시 원본 반환
            Log.w(TAG, "Beauty filter pass-through: output=$outputTexture (same as input or 0)")
            inputTexture
        }
    }

    /**
     * 펜딩 LUT 3D 텍스처 적용 (GL 스레드에서 실행)
     */
    private fun uploadPendingLut3dTexture() {
        val newTextureId = pendingLut3dTextureId
        if (newTextureId == -1) return
        pendingLut3dTextureId = -1

        // 기존 3D 텍스처 삭제
        if (lut3dTextureId != 0) {
            GLES31.glDeleteTextures(1, intArrayOf(lut3dTextureId), 0)
        }

        lut3dTextureId = newTextureId
        Log.d(TAG, "LUT 3D texture set: id=$lut3dTextureId")
    }

    /**
     * 화면에 텍스처 렌더링
     * @param textureId 렌더링할 텍스처 ID
     * @param beautyApplied 뷰티 필터 적용 여부 (테스트용 틴트)
     */
    private fun renderToScreen(textureId: Int, beautyApplied: Boolean = false) {
        // 기본 프레임버퍼 바인딩
        GLES31.glBindFramebuffer(GLES31.GL_FRAMEBUFFER, 0)
        GLES31.glViewport(0, 0, viewWidth, viewHeight)

        // 패스스루 셰이더 사용
        GLES31.glUseProgram(passthroughProgram)

        // 변환 행렬 설정 (단위 행렬)
        val identityMatrix = FloatArray(16)
        Matrix.setIdentityM(identityMatrix, 0)
        val stLocation = GLES31.glGetUniformLocation(passthroughProgram, "uSTMatrix")
        val mirrorLocation = GLES31.glGetUniformLocation(passthroughProgram, "uMirror")
        val flipYLocation = GLES31.glGetUniformLocation(passthroughProgram, "uFlipY")
        val scaleLocation = GLES31.glGetUniformLocation(passthroughProgram, "uScale")
        GLES31.glUniformMatrix4fv(stLocation, 1, false, identityMatrix, 0)
        GLES31.glUniform1i(mirrorLocation, 0)  // 이미 미러링 적용됨
        GLES31.glUniform1i(flipYLocation, 0)   // 이미 Y축 뒤집기 적용됨

        // Aspect ratio 보정 스케일 계산 (Cover 모드 - 화면 꽉 채우기)
        // 회전 고려: 90도 또는 270도 회전 시 width/height 교환
        val isRotated = (frameRotation == 90 || frameRotation == 270)
        val texWidth = if (frameWidth > 0) {
            if (isRotated) frameHeight else frameWidth
        } else viewWidth
        val texHeight = if (frameHeight > 0) {
            if (isRotated) frameWidth else frameHeight
        } else viewHeight

        val texAspect = texWidth.toFloat() / texHeight.toFloat()
        val viewAspect = viewWidth.toFloat() / viewHeight.toFloat()

        val (scaleX, scaleY) = if (texAspect > viewAspect) {
            // 텍스처가 더 넓음 → 높이 맞추고 좌우 확장
            1.0f to (viewAspect / texAspect)
        } else {
            // 텍스처가 더 좁음 → 너비 맞추고 상하 확장
            (texAspect / viewAspect) to 1.0f
        }
        GLES31.glUniform2f(scaleLocation, scaleX, scaleY)

        // 텍스처 바인딩
        GLES31.glActiveTexture(GLES31.GL_TEXTURE0)
        GLES31.glBindTexture(GLES31.GL_TEXTURE_2D, textureId)
        GLES31.glUniform1i(uTextureLocation, 0)

        // 풀스크린 쿼드 렌더링
        renderFullscreenQuad()
    }

    //=========================================================================
    // 설정 API
    //=========================================================================

    /**
     * 뷰티 필터 설정
     */
    fun setBeautyConfig(config: BeautyFilterConfigV2) {
        this.beautyConfig = config
        this.beautyEnabled = config.enabled
    }

    /**
     * 뷰티 필터 활성화/비활성화
     */
    fun setBeautyEnabled(enabled: Boolean) {
        this.beautyEnabled = enabled
        this.beautyConfig.enabled = enabled  // JNI에 전달되는 config도 업데이트
    }

    /**
     * 미러링 설정 (전면 카메라)
     */
    fun setMirror(mirror: Boolean) {
        this.isMirror = mirror
    }

    /**
     * 홍채 검출 결과 설정 (렌즈 오버레이용)
     */
    fun setIrisResult(result: IrisResult?) {
        this.irisResult = result
    }

    /**
     * 렌즈 설정
     */
    fun setLensConfig(config: LensConfig) {
        this.lensConfig = LensConfig(config)
    }

    /**
     * 렌즈 활성화/비활성화
     */
    fun setLensEnabled(enabled: Boolean) {
        this.lensEnabled = enabled
    }

    /**
     * 렌즈 텍스처 설정 (비트맵)
     *
     * GL 스레드가 아닌 곳에서 호출해도 안전 (펜딩 처리)
     */
    fun setLensTexture(bitmap: Bitmap?) {
        if (bitmap == null) {
            // 렌즈 제거
            pendingLensBitmap = null
            lensEnabled = false
        } else {
            // 새 렌즈 설정 (GL 스레드에서 업로드)
            pendingLensBitmap = bitmap
            lensEnabled = true
        }
    }

    /**
     * 렌즈 활성화 여부 반환
     */
    fun isLensEnabled(): Boolean = lensEnabled

    /**
     * LUT 3D 텍스처 설정 (GL 스레드에서 호출)
     *
     * @param textureId LutTextureLoader에서 생성한 3D 텍스처 ID (0이면 비활성화)
     */
    fun setLut3dTexture(textureId: Int) {
        if (textureId == 0) {
            lutEnabled = false
            pendingLut3dTextureId = 0
        } else {
            pendingLut3dTextureId = textureId
            lutEnabled = true
        }
    }

    /**
     * LUT 필터 활성화/비활성화
     */
    fun setLutEnabled(enabled: Boolean) {
        this.lutEnabled = enabled
    }

    /**
     * LUT 필터 강도 설정 (0.0 ~ 1.0)
     */
    fun setLutIntensity(intensity: Float) {
        this.lutIntensity = intensity.coerceIn(0.0f, 1.0f)
    }

    /**
     * LUT 활성화 여부 반환
     */
    fun isLutEnabled(): Boolean = lutEnabled

    /**
     * 카메라 프레임 크기 설정
     */
    fun setFrameSize(width: Int, height: Int) {
        if (this.frameWidth != width || this.frameHeight != height) {
            this.frameWidth = width
            this.frameHeight = height
            Log.d(TAG, "Frame size set: ${width}x${height}")
            // FBO를 프레임 크기로 재생성 필요
            if (isInitialized && width > 0 && height > 0) {
                recreateIntermediateBuffers(width, height)
            }
        }
    }

    /**
     * 카메라 회전 설정 (0, 90, 180, 270)
     */
    fun setFrameRotation(rotation: Int) {
        this.frameRotation = rotation
        Log.d(TAG, "Frame rotation set: $rotation")
    }

    /**
     * SurfaceTexture 반환 (카메라 연결용)
     */
    fun getSurfaceTexture(): SurfaceTexture? = surfaceTexture

    //=========================================================================
    // OpenGL 헬퍼
    //=========================================================================

    /**
     * OES 텍스처 생성
     */
    private fun createOESTexture(): Int {
        val textures = IntArray(1)
        GLES31.glGenTextures(1, textures, 0)
        val textureId = textures[0]

        GLES31.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId)
        GLES31.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES31.GL_TEXTURE_MIN_FILTER, GLES31.GL_LINEAR)
        GLES31.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES31.GL_TEXTURE_MAG_FILTER, GLES31.GL_LINEAR)
        GLES31.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES31.GL_TEXTURE_WRAP_S, GLES31.GL_CLAMP_TO_EDGE)
        GLES31.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES31.GL_TEXTURE_WRAP_T, GLES31.GL_CLAMP_TO_EDGE)

        return textureId
    }

    /**
     * 중간 버퍼 (RGBA 텍스처 + FBO) 생성
     */
    private fun recreateIntermediateBuffers(width: Int, height: Int) {
        // 기존 버퍼 삭제
        if (rgbaTextureId != 0) {
            GLES31.glDeleteTextures(1, intArrayOf(rgbaTextureId), 0)
        }
        if (rgbaFboId != 0) {
            GLES31.glDeleteFramebuffers(1, intArrayOf(rgbaFboId), 0)
        }

        // RGBA 텍스처 생성
        val textures = IntArray(1)
        GLES31.glGenTextures(1, textures, 0)
        rgbaTextureId = textures[0]

        GLES31.glBindTexture(GLES31.GL_TEXTURE_2D, rgbaTextureId)
        GLES31.glTexImage2D(
            GLES31.GL_TEXTURE_2D, 0, GLES31.GL_RGBA,
            width, height, 0,
            GLES31.GL_RGBA, GLES31.GL_UNSIGNED_BYTE, null
        )
        GLES31.glTexParameteri(GLES31.GL_TEXTURE_2D, GLES31.GL_TEXTURE_MIN_FILTER, GLES31.GL_LINEAR)
        GLES31.glTexParameteri(GLES31.GL_TEXTURE_2D, GLES31.GL_TEXTURE_MAG_FILTER, GLES31.GL_LINEAR)
        GLES31.glTexParameteri(GLES31.GL_TEXTURE_2D, GLES31.GL_TEXTURE_WRAP_S, GLES31.GL_CLAMP_TO_EDGE)
        GLES31.glTexParameteri(GLES31.GL_TEXTURE_2D, GLES31.GL_TEXTURE_WRAP_T, GLES31.GL_CLAMP_TO_EDGE)

        // FBO 생성
        val fbos = IntArray(1)
        GLES31.glGenFramebuffers(1, fbos, 0)
        rgbaFboId = fbos[0]

        GLES31.glBindFramebuffer(GLES31.GL_FRAMEBUFFER, rgbaFboId)
        GLES31.glFramebufferTexture2D(
            GLES31.GL_FRAMEBUFFER, GLES31.GL_COLOR_ATTACHMENT0,
            GLES31.GL_TEXTURE_2D, rgbaTextureId, 0
        )

        // FBO 상태 확인
        val status = GLES31.glCheckFramebufferStatus(GLES31.GL_FRAMEBUFFER)
        if (status != GLES31.GL_FRAMEBUFFER_COMPLETE) {
            Log.e(TAG, "FBO is not complete: $status")
        }

        GLES31.glBindFramebuffer(GLES31.GL_FRAMEBUFFER, 0)

        // 렌즈 FBO도 재생성
        if (lensFboId != 0) {
            createLensFbo()
        }

        Log.d(TAG, "Intermediate buffers created: ${width}x${height}")
    }

    /**
     * 풀스크린 쿼드 VAO/VBO 설정
     */
    private fun setupFullscreenQuad() {
        // VAO 생성
        val vaos = IntArray(1)
        GLES31.glGenVertexArrays(1, vaos, 0)
        quadVao = vaos[0]

        // VBO 생성
        val vbos = IntArray(1)
        GLES31.glGenBuffers(1, vbos, 0)
        quadVbo = vbos[0]

        // 버텍스 데이터 업로드
        val vertexBuffer: FloatBuffer = ByteBuffer
            .allocateDirect(FULLSCREEN_QUAD.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .put(FULLSCREEN_QUAD)
        vertexBuffer.position(0)

        GLES31.glBindVertexArray(quadVao)
        GLES31.glBindBuffer(GLES31.GL_ARRAY_BUFFER, quadVbo)
        GLES31.glBufferData(
            GLES31.GL_ARRAY_BUFFER,
            FULLSCREEN_QUAD.size * 4,
            vertexBuffer,
            GLES31.GL_STATIC_DRAW
        )

        // 위치 속성 (location 0)
        GLES31.glEnableVertexAttribArray(0)
        GLES31.glVertexAttribPointer(0, 2, GLES31.GL_FLOAT, false, 16, 0)

        // 텍스처 좌표 속성 (location 1)
        GLES31.glEnableVertexAttribArray(1)
        GLES31.glVertexAttribPointer(1, 2, GLES31.GL_FLOAT, false, 16, 8)

        GLES31.glBindVertexArray(0)
    }

    /**
     * 풀스크린 쿼드 렌더링
     */
    private fun renderFullscreenQuad() {
        GLES31.glBindVertexArray(quadVao)
        GLES31.glDrawArrays(GLES31.GL_TRIANGLE_STRIP, 0, 4)
        GLES31.glBindVertexArray(0)
    }

    /**
     * 셰이더 프로그램 생성
     */
    private fun createProgram(vertexSource: String, fragmentSource: String): Int {
        val vertexShader = compileShader(GLES31.GL_VERTEX_SHADER, vertexSource)
        val fragmentShader = compileShader(GLES31.GL_FRAGMENT_SHADER, fragmentSource)

        val program = GLES31.glCreateProgram()
        GLES31.glAttachShader(program, vertexShader)
        GLES31.glAttachShader(program, fragmentShader)
        GLES31.glLinkProgram(program)

        // 링크 상태 확인
        val linkStatus = IntArray(1)
        GLES31.glGetProgramiv(program, GLES31.GL_LINK_STATUS, linkStatus, 0)
        if (linkStatus[0] != GLES31.GL_TRUE) {
            val log = GLES31.glGetProgramInfoLog(program)
            Log.e(TAG, "Program link failed: $log")
            GLES31.glDeleteProgram(program)
            return 0
        }

        // 셰이더 삭제 (프로그램에 링크됨)
        GLES31.glDeleteShader(vertexShader)
        GLES31.glDeleteShader(fragmentShader)

        return program
    }

    /**
     * 셰이더 컴파일
     */
    private fun compileShader(type: Int, source: String): Int {
        val shader = GLES31.glCreateShader(type)
        GLES31.glShaderSource(shader, source)
        GLES31.glCompileShader(shader)

        // 컴파일 상태 확인
        val compileStatus = IntArray(1)
        GLES31.glGetShaderiv(shader, GLES31.GL_COMPILE_STATUS, compileStatus, 0)
        if (compileStatus[0] != GLES31.GL_TRUE) {
            val log = GLES31.glGetShaderInfoLog(shader)
            val typeName = if (type == GLES31.GL_VERTEX_SHADER) "vertex" else "fragment"
            Log.e(TAG, "$typeName shader compile failed: $log")
            GLES31.glDeleteShader(shader)
            return 0
        }

        return shader
    }

    /**
     * OpenGL 에러 체크
     */
    private fun checkGlError(op: String) {
        var error: Int
        while (GLES31.glGetError().also { error = it } != GLES31.GL_NO_ERROR) {
            Log.e(TAG, "$op: glError 0x${Integer.toHexString(error)}")
        }
    }

    /**
     * 리소스 해제
     */
    fun release() {
        if (beautyOutputTextureId != 0) {
            IrisLensSDK.releaseTexture(beautyOutputTextureId)
        }

        if (oesTextureId != 0) {
            GLES31.glDeleteTextures(1, intArrayOf(oesTextureId), 0)
        }
        if (rgbaTextureId != 0) {
            GLES31.glDeleteTextures(1, intArrayOf(rgbaTextureId), 0)
        }
        if (rgbaFboId != 0) {
            GLES31.glDeleteFramebuffers(1, intArrayOf(rgbaFboId), 0)
        }
        if (quadVao != 0) {
            GLES31.glDeleteVertexArrays(1, intArrayOf(quadVao), 0)
        }
        if (quadVbo != 0) {
            GLES31.glDeleteBuffers(1, intArrayOf(quadVbo), 0)
        }
        if (oesToRgbProgram != 0) {
            GLES31.glDeleteProgram(oesToRgbProgram)
        }
        if (passthroughProgram != 0) {
            GLES31.glDeleteProgram(passthroughProgram)
        }

        // 렌즈 관련 리소스 해제
        if (lensProgram != 0) {
            GLES31.glDeleteProgram(lensProgram)
        }
        if (lensImageTextureId != 0) {
            GLES31.glDeleteTextures(1, intArrayOf(lensImageTextureId), 0)
        }
        if (lensOutputTextureId != 0) {
            GLES31.glDeleteTextures(1, intArrayOf(lensOutputTextureId), 0)
        }
        if (lensFboId != 0) {
            GLES31.glDeleteFramebuffers(1, intArrayOf(lensFboId), 0)
        }
        pendingLensBitmap = null

        // LUT 3D 텍스처 해제 (LUT 셰이더/FBO는 C++ 통합으로 제거됨)
        if (lut3dTextureId != 0) {
            GLES31.glDeleteTextures(1, intArrayOf(lut3dTextureId), 0)
        }

        surfaceTexture?.release()
        surfaceTexture = null

        isInitialized = false
        Log.d(TAG, "Resources released")
    }
}
