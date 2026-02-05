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

import android.graphics.SurfaceTexture
import android.opengl.GLES11Ext
import android.opengl.GLES31
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import android.util.Log
import com.irislenssdk.BeautyFilterConfigV2
import com.irislenssdk.IrisLensSDK
import com.irislenssdk.IrisResult
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
    }

    // SurfaceTexture (카메라 출력)
    private var surfaceTexture: SurfaceTexture? = null
    private var oesTextureId: Int = 0
    private val stMatrix = FloatArray(16)

    // 셰이더 프로그램
    private var oesToRgbProgram: Int = 0
    private var passthroughProgram: Int = 0

    // Uniform locations
    private var uSTMatrixLocation: Int = -1
    private var uMirrorLocation: Int = -1
    private var uFlipYLocation: Int = -1
    private var uScaleLocation: Int = -1
    private var uOESTextureLocation: Int = -1
    private var uTextureLocation: Int = -1

    // 풀스크린 쿼드 VAO/VBO
    private var quadVao: Int = 0
    private var quadVbo: Int = 0

    // 중간 텍스처/FBO (OES → RGBA 변환용)
    private var rgbaTextureId: Int = 0
    private var rgbaFboId: Int = 0

    // 뷰티 필터 출력 텍스처
    private var beautyOutputTextureId: Int = 0

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

    // 콜백
    var onSurfaceTextureAvailable: ((SurfaceTexture) -> Unit)? = null
    var onGpuInitialized: ((Boolean) -> Unit)? = null

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

        // Uniform locations 캐시
        uSTMatrixLocation = GLES31.glGetUniformLocation(oesToRgbProgram, "uSTMatrix")
        uMirrorLocation = GLES31.glGetUniformLocation(oesToRgbProgram, "uMirror")
        uFlipYLocation = GLES31.glGetUniformLocation(oesToRgbProgram, "uFlipY")
        uScaleLocation = GLES31.glGetUniformLocation(oesToRgbProgram, "uScale")
        uOESTextureLocation = GLES31.glGetUniformLocation(oesToRgbProgram, "uOESTexture")
        uTextureLocation = GLES31.glGetUniformLocation(passthroughProgram, "uTexture")

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

        // 화면 클리어
        GLES31.glClear(GLES31.GL_COLOR_BUFFER_BIT)

        // 1단계: OES 텍스처 → RGBA 텍스처 변환
        renderOESToRgba()

        // 2단계: GPU Beauty 필터 적용
        val inputTexture = rgbaTextureId
        val beautyApplied = beautyEnabled && beautyConfig.enabled
        val outputTexture = if (beautyApplied) {
            applyGpuBeautyFilter(inputTexture)
        } else {
            inputTexture
        }

        // 3단계: 화면에 렌더링 (뷰티 적용 여부 전달 - 테스트용 틴트)
        renderToScreen(outputTexture, beautyApplied)

        checkGlError("onDrawFrame")
    }

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
     * GPU 뷰티 필터 적용
     *
     * @param inputTexture 입력 텍스처 ID
     * @return 출력 텍스처 ID
     */
    private fun applyGpuBeautyFilter(inputTexture: Int): Int {
        // 프레임 크기 사용
        val texWidth = if (frameWidth > 0) frameWidth else viewWidth
        val texHeight = if (frameHeight > 0) frameHeight else viewHeight

        // 디버그: 뷰티 설정 확인
        Log.d(TAG, "Beauty filter call: enabled=${beautyConfig.enabled}, smoothing=${beautyConfig.smoothing}, brightness=${beautyConfig.brightness}")

        // GPU Beauty Backend 호출 (JNI)
        val outputTexture = IrisLensSDK.applyBeautyFilterTextureV2(
            inputTexture,
            texWidth,
            texHeight,
            beautyConfig
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

        surfaceTexture?.release()
        surfaceTexture = null

        isInitialized = false
        Log.d(TAG, "Resources released")
    }
}
