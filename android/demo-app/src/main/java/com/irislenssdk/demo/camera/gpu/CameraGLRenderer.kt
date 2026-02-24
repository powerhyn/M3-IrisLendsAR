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
import com.irislenssdk.demo.camera.OneEuroFilter
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

        // FaceMesh 비유효 시 이전 눈꺼풀 클리핑 경계를 유지할 프레임 수
        private const val EYELID_HOLD_FRAMES = 5

        // One Euro Filter 파라미터 (GL 렌즈 경로용)
        // minCutoff: 정지 시 최소 컷오프 주파수. 낮을수록 스무딩 강함.
        //   15.0 → α≈0.61 (pass-through), 1.5 → α≈0.14 (효과적 스무딩)
        private const val GL_FILTER_MIN_CUTOFF = 3.0f    // 정지 시 스무딩 + 이동 초반 반응성 균형
        private const val GL_FILTER_BETA = 7.0f          // 이동 시 필터 즉시 해제 수준
        private const val GL_FILTER_BETA_RADIUS = 3.0f   // 반경: 거리 변화 빠른 추적
        private const val GL_FILTER_BETA_EYELID = 5.0f   // 눈꺼풀: 깜빡임 즉시 반응
        private const val GL_FILTER_D_CUTOFF = 1.0f

        // 반경 데드밴드 (정규화 좌표 기준, detH=1920 시 ~0.5px)
        private const val RADIUS_DEADBAND = 0.0003f

        // 얼굴 미검출 시 avgIrisLum 유지 → 기본값 리셋 타임아웃 (P4-W1-03)
        private const val FACE_INVALID_TIMEOUT_MS = 2000L

        // 눈꺼풀 경계 페더링 범위 (픽셀 기반 동적 계산)
        private const val EYELID_FEATHER_MIN_PX = 2.0f
        private const val EYELID_FEATHER_MAX_PX = 6.0f

        // 다중 랜드마크: 상/하 눈꺼풀 인덱스
        private val LEFT_UPPER_EYELID_INDICES = intArrayOf(159, 160, 161)
        private val LEFT_LOWER_EYELID_INDICES = intArrayOf(145, 144, 153)
        private val RIGHT_UPPER_EYELID_INDICES = intArrayOf(386, 385, 384)
        private val RIGHT_LOWER_EYELID_INDICES = intArrayOf(374, 373, 380)

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
            uniform int uBlendMode;         // 블렌드 모드 (0-7: Normal/Multiply/Screen/Overlay/LumTint/LumTintLinear/SoftLight/ColorReplace)
            uniform int uApplyLeft;         // 왼쪽 눈 적용 여부
            uniform int uApplyRight;        // 오른쪽 눈 적용 여부
            uniform float uFrameAspect;     // 프레임 비율 (width / height)

            // 눈꺼풀 클리핑용 (정규화 좌표 0~1)
            uniform float uLeftEyeTop;      // 왼쪽 눈 상단 Y
            uniform float uLeftEyeBottom;   // 왼쪽 눈 하단 Y
            uniform float uRightEyeTop;     // 오른쪽 눈 상단 Y
            uniform float uRightEyeBottom;  // 오른쪽 눈 하단 Y
            uniform float uEyelidFeather;   // 눈꺼풀 경계 페더링 (동적, 픽셀 기반)
            uniform float uAvgIrisLum;      // 홍채 평균 밝기 (CPU EMA, 0.0~1.0)

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

            // Fast linearization helpers (pow(2.2) 대비 ~3-5x 빠름)
            vec3 toLinearFast(vec3 srgb) { return srgb * srgb; }
            vec3 toSRGBFast(vec3 linear) { return sqrt(max(linear, vec3(0.0))); }

            // Mode 4: Luminance-preserving color tint (sRGB 근사)
            vec3 blendLuminanceTint(vec3 base, vec3 blend, float opacity) {
                float lum = dot(base, vec3(0.299, 0.587, 0.114));
                float scale = clamp(0.5 / max(0.1, uAvgIrisLum), 0.8, 2.5);
                vec3 tinted = blend * lum * scale;
                return mix(base, tinted, opacity);
            }

            // Mode 5: Luminance-preserving color tint (fast linear space + specular 복원)
            vec3 blendLuminanceTintLinear(vec3 base, vec3 blend, float opacity) {
                vec3 baseL = toLinearFast(base);
                float lum = dot(baseL, vec3(0.2126, 0.7152, 0.0722));
                float avgLumLinear = uAvgIrisLum * uAvgIrisLum;  // ISS-005 EXP-C: sRGB→linear 근사
                float scale = clamp(0.5 / max(0.01, avgLumLinear), 0.8, 5.0);
                vec3 tinted = toLinearFast(blend) * lum * scale;
                vec3 result = mix(baseL, tinted, opacity);
                float realSpec = smoothstep(0.7, 0.95, lum);
                result = mix(result, baseL, realSpec);
                return toSRGBFast(result);
            }

            // Mode 6: Photoshop Soft Light
            vec3 blendSoftLight(vec3 base, vec3 blend, float opacity) {
                vec3 lo = base - (1.0 - 2.0 * blend) * base * (1.0 - base);
                vec3 hi = base + (2.0 * blend - 1.0) * (sqrt(base) - base);
                vec3 result = mix(lo, hi, step(vec3(0.5), blend));
                return mix(base, result, opacity);
            }

            // Mode 7: Color Replace (상대 밝기 정규화)
            vec3 blendColorReplace(vec3 base, vec3 blend, float opacity) {
                float lum = dot(base, vec3(0.299, 0.587, 0.114));
                float detail = lum / max(0.01, uAvgIrisLum);
                detail = clamp(detail, 0.2, 2.5);
                vec3 colored = blend * detail;
                return mix(base, colored, opacity);
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
                float eyelidFeather = uEyelidFeather;
                float minY = min(eyeTop, eyeBottom);
                float maxY = max(eyeTop, eyeBottom);
                // topClip: minY 아래쪽에서 1.0 (눈 안쪽), minY 위쪽에서 0.0 (눈꺼풀 밖)
                float topClip = smoothstep(minY - eyelidFeather, minY + eyelidFeather, vTexCoord.y);
                // bottomClip: maxY 위쪽에서 1.0 (눈 안쪽), maxY 아래쪽에서 0.0 (눈꺼풀 밖)
                // 주의: smoothstep(edge0, edge1, x)는 edge0 < edge1 필수 (GLSL spec)
                float bottomClip = 1.0 - smoothstep(maxY - eyelidFeather, maxY + eyelidFeather, vTexCoord.y);
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
                } else if (uBlendMode == 3) {
                    blended = blendOverlay(camera.rgb, lens.rgb, finalAlpha);
                } else if (uBlendMode == 4) {
                    blended = blendLuminanceTint(camera.rgb, lens.rgb, finalAlpha);
                } else if (uBlendMode == 5) {
                    blended = blendLuminanceTintLinear(camera.rgb, lens.rgb, finalAlpha);
                } else if (uBlendMode == 6) {
                    blended = blendSoftLight(camera.rgb, lens.rgb, finalAlpha);
                } else if (uBlendMode == 7) {
                    blended = blendColorReplace(camera.rgb, lens.rgb, finalAlpha);
                } else {
                    blended = blendNormal(camera.rgb, lens.rgb, finalAlpha);
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
    private var uEyelidFeatherLocation: Int = -1
    private var uAvgIrisLumLocation: Int = -1

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

    // === One Euro Filter: 홍채 중심/반경 안정화 (GL 경로) ===
    private val glLeftXFilter = OneEuroFilter(GL_FILTER_MIN_CUTOFF, GL_FILTER_BETA, GL_FILTER_D_CUTOFF)
    private val glLeftYFilter = OneEuroFilter(GL_FILTER_MIN_CUTOFF, GL_FILTER_BETA, GL_FILTER_D_CUTOFF)
    private val glLeftRadiusFilter = OneEuroFilter(GL_FILTER_MIN_CUTOFF, GL_FILTER_BETA_RADIUS, GL_FILTER_D_CUTOFF)
    private val glRightXFilter = OneEuroFilter(GL_FILTER_MIN_CUTOFF, GL_FILTER_BETA, GL_FILTER_D_CUTOFF)
    private val glRightYFilter = OneEuroFilter(GL_FILTER_MIN_CUTOFF, GL_FILTER_BETA, GL_FILTER_D_CUTOFF)
    private val glRightRadiusFilter = OneEuroFilter(GL_FILTER_MIN_CUTOFF, GL_FILTER_BETA_RADIUS, GL_FILTER_D_CUTOFF)

    // === One Euro Filter: 눈꺼풀 경계 안정화 ===
    private val glLeftEyeTopFilter = OneEuroFilter(GL_FILTER_MIN_CUTOFF, GL_FILTER_BETA_EYELID, GL_FILTER_D_CUTOFF)
    private val glLeftEyeBottomFilter = OneEuroFilter(GL_FILTER_MIN_CUTOFF, GL_FILTER_BETA_EYELID, GL_FILTER_D_CUTOFF)
    private val glRightEyeTopFilter = OneEuroFilter(GL_FILTER_MIN_CUTOFF, GL_FILTER_BETA_EYELID, GL_FILTER_D_CUTOFF)
    private val glRightEyeBottomFilter = OneEuroFilter(GL_FILTER_MIN_CUTOFF, GL_FILTER_BETA_EYELID, GL_FILTER_D_CUTOFF)

    // 필터링된 반경 (데드밴드 적용용)
    private var lastFilteredLeftRadius: Float = 0f
    private var lastFilteredRightRadius: Float = 0f

    // 눈꺼풀 클리핑 temporal hold (FaceMesh 비유효 시 이전 값 유지)
    private var cachedLeftEyeTop: Float = 0.0f
    private var cachedLeftEyeBottom: Float = 1.0f
    private var cachedRightEyeTop: Float = 0.0f
    private var cachedRightEyeBottom: Float = 1.0f
    private var eyelidCacheValidFrames: Int = 0  // 캐시 유효 잔여 프레임 수

    // === Adaptive Iris Luminance: EMA (P4-W1-03) ===
    private var avgIrisLum = 0.35f           // EMA 평균 (어두운 홍채 기본값, 한국인 평균 근사)
    private var lastValidFaceTimeMs = 0L

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

    // StabilityLogger 콜백 (P4-W1-02: 안정성 측정)
    var stabilityLogEnabled: Boolean = false
    var onStabilityFrame: ((
        faceDetected: Boolean,
        rawLeftCx: Float, rawLeftCy: Float, rawLeftR: Float,
        filteredLeftCx: Float, filteredLeftCy: Float, filteredLeftR: Float,
        rawRightCx: Float, rawRightCy: Float, rawRightR: Float,
        filteredRightCx: Float, filteredRightCy: Float, filteredRightR: Float,
        eyelidLt: Float, eyelidLb: Float, eyelidRt: Float, eyelidRb: Float,
        holdActive: Boolean, holdRemaining: Int,
        renderTimeUs: Long
    ) -> Unit)? = null

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
        uEyelidFeatherLocation = GLES31.glGetUniformLocation(lensProgram, "uEyelidFeather")
        uAvgIrisLumLocation = GLES31.glGetUniformLocation(lensProgram, "uAvgIrisLum")

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
        } else if (stabilityLogEnabled && lensEnabled && lensImageTextureId != 0) {
            // 렌즈 파이프라인 활성 상태에서 검출 실패 시에만 기록
            // (렌즈 미선택/텍스처 미준비 시에는 기록하지 않음)
            onStabilityFrame?.invoke(
                irisResult?.detected ?: false,
                0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 0f, 0f,
                0f, 1f, 0f, 1f,
                eyelidCacheValidFrames > 0, eyelidCacheValidFrames, 0L
            )
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
        GLES31.glTexParameteri(GLES31.GL_TEXTURE_2D, GLES31.GL_TEXTURE_MAG_FILTER, GLES31.GL_LINEAR)
        GLES31.glTexParameteri(GLES31.GL_TEXTURE_2D, GLES31.GL_TEXTURE_WRAP_S, GLES31.GL_CLAMP_TO_EDGE)
        GLES31.glTexParameteri(GLES31.GL_TEXTURE_2D, GLES31.GL_TEXTURE_WRAP_T, GLES31.GL_CLAMP_TO_EDGE)

        // 비트맵 업로드 + mipmap 생성 (축소 시 shimmer/aliasing 방지)
        GLUtils.texImage2D(GLES31.GL_TEXTURE_2D, 0, bitmap, 0)
        GLES31.glGenerateMipmap(GLES31.GL_TEXTURE_2D)
        GLES31.glTexParameteri(GLES31.GL_TEXTURE_2D, GLES31.GL_TEXTURE_MIN_FILTER, GLES31.GL_LINEAR_MIPMAP_LINEAR)

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
        val renderStartNs = if (stabilityLogEnabled) System.nanoTime() else 0L
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
        // 좌표 계약: result.frameWidth/Height가 검출기의 좌표 기준 (회전 적용 후)
        val (detW, detH) = resolveCoordinateSpace(result)
        val detHf = detH.toFloat()
        // ISS-004 Fix-B: 셰이더의 adjusted 좌표계(높이 기준)에 맞춰 detH로 정규화
        val normalizedLeftRadius = result.leftRadius / detHf
        val normalizedRightRadius = result.rightRadius / detHf

        // 좌표 변환: renderOESToRgba()에서 적용한 변환과 동일하게 적용
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
            val tempX = leftX; val tempY = leftY
            leftX = rightX; leftY = rightY
            rightX = tempX; rightY = tempY
        }

        // === One Euro Filter: 홍채 중심/반경 안정화 ===
        val now = System.currentTimeMillis()
        val rawLeftRadius = if (isMirror) normalizedRightRadius else normalizedLeftRadius
        val rawRightRadius = if (isMirror) normalizedLeftRadius else normalizedRightRadius

        val filteredLeftX = glLeftXFilter.filter(leftX, now)
        val filteredLeftY = glLeftYFilter.filter(leftY, now)
        var filteredLeftR = glLeftRadiusFilter.filter(rawLeftRadius, now)
        val filteredRightX = glRightXFilter.filter(rightX, now)
        val filteredRightY = glRightYFilter.filter(rightY, now)
        var filteredRightR = glRightRadiusFilter.filter(rawRightRadius, now)

        // 반경 데드밴드: 변화량이 임계값 미만이면 이전 값 유지
        if (kotlin.math.abs(filteredLeftR - lastFilteredLeftRadius) < RADIUS_DEADBAND && lastFilteredLeftRadius > 0f) {
            filteredLeftR = lastFilteredLeftRadius
        } else {
            lastFilteredLeftRadius = filteredLeftR
        }
        if (kotlin.math.abs(filteredRightR - lastFilteredRightRadius) < RADIUS_DEADBAND && lastFilteredRightRadius > 0f) {
            filteredRightR = lastFilteredRightRadius
        } else {
            lastFilteredRightRadius = filteredRightR
        }

        GLES31.glUniform2f(uLeftIrisCenterLocation, filteredLeftX, filteredLeftY)
        GLES31.glUniform1f(uLeftIrisRadiusLocation, filteredLeftR)
        GLES31.glUniform2f(uRightIrisCenterLocation, filteredRightX, filteredRightY)
        GLES31.glUniform1f(uRightIrisRadiusLocation, filteredRightR)

        // 렌즈 설정 전달
        GLES31.glUniform1f(uLensOpacityLocation, lensConfig.opacity)
        GLES31.glUniform1f(uLensScaleLocation, lensConfig.scale)
        GLES31.glUniform1f(uLensEdgeFeatherLocation, lensConfig.edgeFeather)
        GLES31.glUniform1i(uLensBlendModeLocation, lensConfig.blendMode)
        GLES31.glUniform1i(uApplyLeftLocation, if (lensConfig.applyLeft) 1 else 0)
        GLES31.glUniform1i(uApplyRightLocation, if (lensConfig.applyRight) 1 else 0)
        GLES31.glUniform1f(uAvgIrisLumLocation, avgIrisLum)

        // 프레임 비율 전달
        val frameAspect = detW.toFloat() / detHf
        GLES31.glUniform1f(uFrameAspectLocation, frameAspect)

        // === 동적 eyelidFeather 계산 (픽셀 기반) ===
        val featherPx = EYELID_FEATHER_MIN_PX.coerceAtLeast(
            EYELID_FEATHER_MAX_PX.coerceAtMost(4.0f)
        )
        val eyelidFeatherNorm = featherPx / detHf
        GLES31.glUniform1f(uEyelidFeatherLocation, eyelidFeatherNorm)

        // === 눈꺼풀 클리핑: 다중 랜드마크 + One Euro Filter ===
        // StabilityLogger용: 필터 후 눈꺼풀 값 보존
        var logEyelidLt = 0.0f
        var logEyelidLb = 1.0f
        var logEyelidRt = 0.0f
        var logEyelidRb = 1.0f
        var logHoldActive = false
        var logHoldRemaining = 0

        val faceMesh = result.faceMesh
        if (result.faceMeshValid && faceMesh != null) {
            // 다중 랜드마크에서 median Y 추출 (노이즈 내성 향상)
            var leftEyeTop = medianLandmarkY(faceMesh, LEFT_UPPER_EYELID_INDICES)
            var leftEyeBottom = medianLandmarkY(faceMesh, LEFT_LOWER_EYELID_INDICES)
            var rightEyeTop = medianLandmarkY(faceMesh, RIGHT_UPPER_EYELID_INDICES)
            var rightEyeBottom = medianLandmarkY(faceMesh, RIGHT_LOWER_EYELID_INDICES)

            // Y축 뒤집기 적용
            leftEyeTop = 1.0f - leftEyeTop
            leftEyeBottom = 1.0f - leftEyeBottom
            rightEyeTop = 1.0f - rightEyeTop
            rightEyeBottom = 1.0f - rightEyeBottom

            // 미러링 시 좌/우 교환
            if (isMirror) {
                val tT = leftEyeTop; val tB = leftEyeBottom
                leftEyeTop = rightEyeTop; leftEyeBottom = rightEyeBottom
                rightEyeTop = tT; rightEyeBottom = tB
            }

            // One Euro Filter 적용 (눈꺼풀 경계 안정화)
            leftEyeTop = glLeftEyeTopFilter.filter(leftEyeTop, now)
            leftEyeBottom = glLeftEyeBottomFilter.filter(leftEyeBottom, now)
            rightEyeTop = glRightEyeTopFilter.filter(rightEyeTop, now)
            rightEyeBottom = glRightEyeBottomFilter.filter(rightEyeBottom, now)

            // 캐시 갱신 (temporal hold용)
            cachedLeftEyeTop = leftEyeTop
            cachedLeftEyeBottom = leftEyeBottom
            cachedRightEyeTop = rightEyeTop
            cachedRightEyeBottom = rightEyeBottom
            eyelidCacheValidFrames = EYELID_HOLD_FRAMES

            GLES31.glUniform1f(uLeftEyeTopLocation, leftEyeTop)
            GLES31.glUniform1f(uLeftEyeBottomLocation, leftEyeBottom)
            GLES31.glUniform1f(uRightEyeTopLocation, rightEyeTop)
            GLES31.glUniform1f(uRightEyeBottomLocation, rightEyeBottom)

            logEyelidLt = leftEyeTop
            logEyelidLb = leftEyeBottom
            logEyelidRt = rightEyeTop
            logEyelidRb = rightEyeBottom
        } else if (eyelidCacheValidFrames > 0) {
            // FaceMesh 비유효: temporal hold + 점진적 감쇠(fade)
            eyelidCacheValidFrames--
            // 감쇠 비율: 잔여 프레임 / 전체 → 0에 가까워질수록 클리핑 해제
            val fadeAlpha = eyelidCacheValidFrames.toFloat() / EYELID_HOLD_FRAMES.toFloat()
            // 캐시 값 → 전체 허용(0.0/1.0) 방향으로 lerp
            GLES31.glUniform1f(uLeftEyeTopLocation, lerp(0.0f, cachedLeftEyeTop, fadeAlpha))
            GLES31.glUniform1f(uLeftEyeBottomLocation, lerp(1.0f, cachedLeftEyeBottom, fadeAlpha))
            GLES31.glUniform1f(uRightEyeTopLocation, lerp(0.0f, cachedRightEyeTop, fadeAlpha))
            GLES31.glUniform1f(uRightEyeBottomLocation, lerp(1.0f, cachedRightEyeBottom, fadeAlpha))

            logEyelidLt = lerp(0.0f, cachedLeftEyeTop, fadeAlpha)
            logEyelidLb = lerp(1.0f, cachedLeftEyeBottom, fadeAlpha)
            logEyelidRt = lerp(0.0f, cachedRightEyeTop, fadeAlpha)
            logEyelidRb = lerp(1.0f, cachedRightEyeBottom, fadeAlpha)
            logHoldActive = true
            logHoldRemaining = eyelidCacheValidFrames
        } else {
            // 캐시 소진: 클리핑 비활성화 (전체 영역 허용)
            GLES31.glUniform1f(uLeftEyeTopLocation, 0.0f)
            GLES31.glUniform1f(uLeftEyeBottomLocation, 1.0f)
            GLES31.glUniform1f(uRightEyeTopLocation, 0.0f)
            GLES31.glUniform1f(uRightEyeBottomLocation, 1.0f)
        }

        // 풀스크린 쿼드 렌더링
        renderFullscreenQuad()

        // FBO 언바인딩
        GLES31.glBindFramebuffer(GLES31.GL_FRAMEBUFFER, 0)

        // === StabilityLogger 콜백 (P4-W1-02) ===
        if (stabilityLogEnabled) {
            val renderTimeUs = (System.nanoTime() - renderStartNs) / 1000L
            onStabilityFrame?.invoke(
                result.detected,
                leftX, leftY, rawLeftRadius,
                filteredLeftX, filteredLeftY, filteredLeftR,
                rightX, rightY, rawRightRadius,
                filteredRightX, filteredRightY, filteredRightR,
                logEyelidLt, logEyelidLb, logEyelidRt, logEyelidRb,
                logHoldActive, logHoldRemaining,
                renderTimeUs
            )
        }

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

        // Detection Slot에서 최신 검출 결과 포인터 취득 (lock-free)
        val detectionHandle = IrisLensSDK.getDetectionSlotPtr()

        // 디버그: 뷰티+LUT 설정 확인
        Log.d(TAG, "Beauty filter call: enabled=${beautyConfig.enabled}, smoothing=${beautyConfig.smoothing}, brightness=${beautyConfig.brightness}, lut=$lutTextureId, lutIntensity=$lutIntensityVal, detHandle=$detectionHandle")

        // GPU Beauty Backend 호출 (JNI) - LUT 통합 + Detection Handle
        val outputTexture = IrisLensSDK.applyBeautyFilterTextureV2(
            inputTexture,
            texWidth,
            texHeight,
            beautyConfig,
            detectionHandle,
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

        // Aspect ratio 보정 스케일 계산 (Cover 모드 - 화면 꽉 채우기, 넘치는 부분 crop)
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

        // Cover 모드: 화면을 꽉 채우고 넘치는 부분은 GL viewport에 의해 자동 crop
        val (scaleX, scaleY) = if (texAspect > viewAspect) {
            // 텍스처가 더 넓음 → 높이 채우고 좌우 넘침 (crop)
            (texAspect / viewAspect) to 1.0f
        } else {
            // 텍스처가 더 좁음 → 너비 채우고 상하 넘침 (crop)
            1.0f to (viewAspect / texAspect)
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
     * 검출 좌표계와 렌더 좌표계를 통일합니다.
     *
     * 좌표 계약:
     * - 1순위: result.frameWidth/frameHeight (검출기가 회전 적용 후 기록)
     * - 2순위: frameWidth/frameHeight (카메라 프레임 크기, 회전 보정 적용)
     * - 3순위: viewWidth/viewHeight (폴백)
     *
     * @return (detW, detH) 렌즈 계산에 사용할 좌표 기준 크기
     */
    private fun resolveCoordinateSpace(result: IrisResult): Pair<Int, Int> {
        // 1순위: 검출 결과의 프레임 크기 (이미 회전 적용됨)
        if (result.frameWidth > 0 && result.frameHeight > 0) {
            return Pair(result.frameWidth, result.frameHeight)
        }
        // 2순위: 카메라 프레임 크기 (회전 보정)
        val isRotated = (frameRotation == 90 || frameRotation == 270)
        val w = if (frameWidth > 0) {
            if (isRotated) frameHeight else frameWidth
        } else viewWidth
        val h = if (frameHeight > 0) {
            if (isRotated) frameWidth else frameHeight
        } else viewHeight
        return Pair(w, h)
    }

    /**
     * FaceMesh 랜드마크 인덱스 배열에서 Y 좌표의 중앙값(median) 추출.
     * 단일 점 대비 노이즈 내성 향상.
     */
    private fun medianLandmarkY(mesh: FloatArray, indices: IntArray): Float {
        val ys = FloatArray(indices.size) { mesh[indices[it] * 3 + 1] }
        ys.sort()
        return if (ys.size % 2 == 1) {
            ys[ys.size / 2]
        } else {
            (ys[ys.size / 2 - 1] + ys[ys.size / 2]) / 2f
        }
    }

    /**
     * 선형 보간 (a → b, t=0이면 a, t=1이면 b)
     */
    private fun lerp(a: Float, b: Float, t: Float): Float {
        return a + (b - a) * t
    }

    /**
     * 홍채 검출 결과 설정 (렌즈 오버레이용)
     */
    fun setIrisResult(result: IrisResult?) {
        this.irisResult = result
    }

    /**
     * 홍채 평균 밝기 업데이트 (EMA α=0.1)
     *
     * CPU 측에서 NV21 Y채널 샘플링 후 호출.
     * @param rawLuminance 0.0~1.0 범위의 원시 밝기 (미검출 시 음수)
     */
    fun updateAvgIrisLum(rawLuminance: Float) {
        val currentTimeMs = System.currentTimeMillis()
        if (rawLuminance >= 0f) {
            lastValidFaceTimeMs = currentTimeMs
            avgIrisLum = avgIrisLum * 0.9f + rawLuminance * 0.1f
            avgIrisLum = avgIrisLum.coerceIn(0.05f, 0.95f)
        } else {
            // Hold: 미검출 시 마지막 유효값 유지, 타임아웃 시 기본값 리셋
            if (currentTimeMs - lastValidFaceTimeMs > FACE_INVALID_TIMEOUT_MS) {
                avgIrisLum = 0.35f
            }
        }
    }

    /**
     * Temporal 상태 리셋 (onResume 시 호출)
     *
     * resume 후 dt 기반 연산의 cold-start 폭주 방지.
     */
    fun resetTemporalState() {
        lastValidFaceTimeMs = 0L
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
        // pending 슬롯에 아직 업로드되지 않은 텍스처가 있으면 누수 방지를 위해 즉시 삭제
        val oldPending = pendingLut3dTextureId
        if (oldPending > 0 && oldPending != textureId) {
            GLES31.glDeleteTextures(1, intArrayOf(oldPending), 0)
            Log.d(TAG, "Deleted overwritten pending LUT texture: id=$oldPending")
        }

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
        // beautyOutputTextureId는 TexturePool 소유 → releaseGpuBeauty()에서 일괄 해제
        // 여기서 releaseTexture() 호출하면 이중 해제 발생
        beautyOutputTextureId = 0

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
