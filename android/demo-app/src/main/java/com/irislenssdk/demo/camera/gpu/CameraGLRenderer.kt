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
import android.opengl.Matrix
import android.util.Log
import com.irislenssdk.BeautyFilterConfigV2
import com.irislenssdk.IrisLensSDK
import com.irislenssdk.IrisResult
import com.irislenssdk.demo.tracking.math.FrameRingSelector
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

        // One Euro Filter 파라미터 (타원/눈꺼풀 파생 값 전용 — SDK 코어가 커버하지 않는 Kotlin 계산 값)
        private const val GL_FILTER_MIN_CUTOFF = 4.0f
        private const val GL_FILTER_BETA = 15.0f
        private const val GL_FILTER_BETA_EYELID = 12.0f
        private const val GL_FILTER_D_CUTOFF = 1.0f


        // 눈꺼풀 경계 페더링 범위 (픽셀 기반 동적 계산)
        private const val EYELID_FEATHER_MIN_PX = 2.0f
        private const val EYELID_FEATHER_MAX_PX = 6.0f

        // 다중 랜드마크: 상/하 눈꺼풀 인덱스
        private val LEFT_UPPER_EYELID_INDICES = intArrayOf(159, 160, 161)
        private val LEFT_LOWER_EYELID_INDICES = intArrayOf(145, 144, 153)
        private val RIGHT_UPPER_EYELID_INDICES = intArrayOf(386, 385, 384)
        private val RIGHT_LOWER_EYELID_INDICES = intArrayOf(374, 373, 380)

        // 16점 눈 윤곽 랜드마크 (P4-W2-02: 비대칭 타원 Eye Mask)

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
        private const val VERTEX_SHADER = """#version 310 es
            layout(location = 0) in vec2 aPosition;
            layout(location = 1) in vec2 aTexCoord;

            uniform mat4 uSTMatrix;  // SurfaceTexture 변환 행렬
            uniform int uMirror;     // 미러링 (전면 카메라)
            uniform int uFlipY;      // Y축 뒤집기
            uniform vec2 uScale;     // Aspect ratio 보정 스케일
            uniform int uRotate;     // 화면 회전 90도 배수 (0..3) — 최종 blit 전용, FBO 패스는 항상 0

            out vec2 vTexCoord;

            void main() {
                vec2 pos = aPosition;
                // Y축 뒤집기 (상하 반전)
                if (uFlipY == 1) {
                    pos.y = -pos.y;
                }
                // Aspect ratio 보정 스케일 적용
                pos *= uScale;
                // 화면 회전 (정점만 회전 — texCoord/미러 경로는 불변).
                // 90도 배수는 축 교환이라 uScale을 회전 후 기준으로 미리 교환해 두면
                // 별도 종횡비 보정 없이 정확히 맞는다 (renderToScreen 참조).
                if (uRotate == 1) {
                    pos = vec2(-pos.y, pos.x);
                } else if (uRotate == 2) {
                    pos = vec2(-pos.x, -pos.y);
                } else if (uRotate == 3) {
                    pos = vec2(pos.y, -pos.x);
                }
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
        private const val OES_TO_2D_FRAGMENT_SHADER = """#version 310 es
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
        // 최종 blit — 업스케일 품질 담당.
        //
        // 소스(1920x1080)를 태블릿 창(2960x1848)에 그리면 1.54배 확대가 되는데, 기본 GL_LINEAR
        // (bilinear)는 이 배율에서 디테일을 뭉갠다(기본 카메라 앱·참조앱 대비 흐림의 직접 원인).
        //   uUpscaleMode 0 = bilinear (종전과 동일)
        //   uUpscaleMode 1 = Catmull-Rom bicubic — 하드웨어 bilinear 4탭 조합으로 16탭 bicubic을
        //       근사(Sigg&Hadwiger). 엣지를 살려 확대해 선명도가 오른다.
        //   uUpscaleMode 2 = bicubic + 언샤프 마스크 — 확대로 잃은 고주파를 되살린다(과하면 링잉).
        private const val PASSTHROUGH_FRAGMENT_SHADER = """#version 310 es
            precision highp float;

            uniform sampler2D uTexture;
            uniform vec2 uTexSize;       // 소스 텍스처 픽셀 크기 (bicubic 좌표 계산용)
            uniform int uUpscaleMode;    // 0=bilinear, 1=bicubic, 2=bicubic+sharpen
            uniform float uSharpen;      // 언샤프 강도 (mode 2에서만)

            in vec2 vTexCoord;
            out vec4 fragColor;

            // Catmull-Rom bicubic: 4번의 하드웨어 bilinear fetch로 16탭 bicubic과 동등한 결과.
            vec4 textureBicubic(vec2 uv) {
                vec2 texelSize = 1.0 / uTexSize;
                vec2 coord = uv * uTexSize - 0.5;
                vec2 f = fract(coord);
                coord = floor(coord);

                vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
                vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
                vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
                vec2 w3 = f * f * (-0.5 + 0.5 * f);

                vec2 s0 = w0 + w1;
                vec2 s1 = w2 + w3;
                vec2 f0 = w1 / s0;
                vec2 f1 = w3 / s1;

                vec2 t0 = (coord - 1.0 + f0) * texelSize;
                vec2 t1 = (coord + 1.0 + f1) * texelSize;

                return texture(uTexture, vec2(t0.x, t0.y)) * s0.x * s0.y
                     + texture(uTexture, vec2(t1.x, t0.y)) * s1.x * s0.y
                     + texture(uTexture, vec2(t0.x, t1.y)) * s0.x * s1.y
                     + texture(uTexture, vec2(t1.x, t1.y)) * s1.x * s1.y;
            }

            void main() {
                if (uUpscaleMode == 0) {
                    fragColor = texture(uTexture, vTexCoord);
                    return;
                }

                vec4 c = textureBicubic(vTexCoord);

                if (uUpscaleMode == 2 && uSharpen > 0.0) {
                    // 언샤프 마스크: 소스 텍셀 기준 4-이웃 평균을 저주파로 보고 차분을 되돌린다.
                    vec2 t = 1.0 / uTexSize;
                    vec3 lo = (texture(uTexture, vTexCoord + vec2( t.x, 0.0)).rgb
                             + texture(uTexture, vTexCoord + vec2(-t.x, 0.0)).rgb
                             + texture(uTexture, vTexCoord + vec2(0.0,  t.y)).rgb
                             + texture(uTexture, vTexCoord + vec2(0.0, -t.y)).rgb) * 0.25;
                    c.rgb = clamp(c.rgb + (c.rgb - lo) * uSharpen, 0.0, 1.0);
                }

                fragColor = c;
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

    // Uniform locations (OES → RGBA)
    private var uSTMatrixLocation: Int = -1
    private var uMirrorLocation: Int = -1
    private var uFlipYLocation: Int = -1
    private var uScaleLocation: Int = -1
    private var uRotateLocation: Int = -1
    private var uOESTextureLocation: Int = -1
    private var uTextureLocation: Int = -1


    // 풀스크린 쿼드 VAO/VBO
    private var quadVao: Int = 0
    private var quadVbo: Int = 0

    // 중간 텍스처/FBO (OES → RGBA 변환용) — frame-sync 링버퍼로 확장.
    // frame-sync(트래킹 지연 핸드오프 §3-b): OES→RGBA 변환 출력을 RGBA 프레임 링버퍼
    // [ringSize]장에 센서 ts와 함께 보관하고, 렌더는 '랜드마크가 계산된 프레임'(분석측
    // frameTimestampNs와 |Δ| 최소, FrameRingSelector)을 골라 그 위에 렌즈를 합성한다.
    // frame-sync OFF(기본)면 항상 최신 슬롯 = 현행 단일버퍼 동작과 비트 동일.
    private val ringSize = 4
    private val ringTex = IntArray(ringSize)
    private val ringFbo = IntArray(ringSize)
    private val ringTsNs = LongArray(ringSize)
    private var ringWrite = 0       // 이번 프레임이 쓸 슬롯
    private var ringCount = 0       // 유효 슬롯 수 (워밍업 중 ringSize보다 작을 수 있음)
    @Volatile private var frameSyncActive: Boolean = false   // 킬스위치 상태 (setFrameSyncEnabled로만 변경)
    // W4-B3: 검출 슬롯 단일 스냅샷 메타 수신용 재사용 배열 [0]=frameTsNs, [1]=detected?1:0 (GL 스레드 전용)
    private val detSlotMeta = LongArray(2)
    // 클럭 도메인 검증 (LensSim selectCamSource 정본 가드 2종):
    private var clockDomainStreak = 0          // 연속 |Δ|>1s 카운트
    private var clockDomainChecked = false     // 검증 완료(일치/불일치 무관) — 이후 재평가 영구 중단
    private var clockDomainMismatch = false    // 불일치 확정 → 영구 최신 슬롯 폴백
    private var clockDomainLastTs = 0L         // 마지막 평가한 스냅샷 ts — '스냅샷당 1표'(같은 ts 재집계 금지)
    private var frameSyncLogCounter = 0

    // 뷰티 필터 출력 텍스처
    private var beautyOutputTextureId: Int = 0

    // 렌즈 FBO/텍스처
    private var lensFboId: Int = 0
    private var lensOutputTextureId: Int = 0

    // 렌즈 텍스처 상태
    // (KT fallback 셰이더 제거 후) KT측 GL 렌즈 텍스처는 native 렌더에 쓰이지 않는다.
    // 렌즈는 native GPULensRenderer(lens_texture_)로만 업로드하며, 렌더 게이트는 native 로드 성공
    // 여부(nativeLensLoaded)로 판정한다. EGL 컨텍스트 재생성 시 native lens_texture_가 해제되므로
    // onSurfaceCreated에서 false로 리셋되고, GpuRenderActivity가 현재 렌즈를 재업로드한다.
    private var nativeLensLoaded: Boolean = false
    private var pendingLensBitmap: Bitmap? = null
    private var pendingLensSkuId: String = ""  // P6-W7: 렌즈 SKU id (메타 연동)

    // 화면 크기
    private var viewWidth: Int = 0
    private var viewHeight: Int = 0

    // 카메라 프레임 크기
    private var frameWidth: Int = 0
    private var frameHeight: Int = 0
    private var frameRotation: Int = 0  // 카메라 회전 각도 (0, 90, 180, 270)

    // 화면(디스플레이) 회전 각도 (0, 90, 180, 270). natural orientation 기준.
    // 세로 고정 폰에서는 항상 0이라 회전 경로 전체가 항등(=수정 전과 픽셀 동일)이다.
    private var screenRotation: Int = 0
    // 회전 부호 A/B (실기기 육안 확정용). true면 보정 방향을 반대로 적용한다.
    private var screenRotationInverted: Boolean = false

    // FOV 확대(displayZoom): 최종 blit에만 곱하는 등방 배율.
    //
    // 문제: 카메라는 4:3(=센서 최대 화각)인데 태블릿 가로 창은 약 16:10이라, Cover 계산이 항상
    //   width-bound(else 분기)로 떨어져 배율이 S = viewWidth/texWidth 최소값에 고정된다.
    //   → 얼굴 위 천장이 넓게 잡히고 피사체가 작아 보인다(참조앱 FMLens 대비 약 -10%).
    // 처방: 캡처·랜드마크·렌즈 좌표계는 그대로 두고 **최종 blit 정점 스케일에만** 등방 배율을 곱해
    //   화면 표시만 확대한다. 링 FBO는 1:1(uScale=1,1)이고 렌즈/뷰티 합성도 그 공간에서 끝나므로
    //   배경·렌즈·뷰티가 같은 변환 하나를 함께 타 정합이 자동 보존된다(캡처 FOV 유지 → 추적도 무영향).
    // 등방이라 rotSwap 축 교환과 무관하다. 기본 1.0f = 현행과 비트 동일(폰 세로 무회귀).
    private var displayZoom: Float = 1.0f

    // 업스케일 품질 (최종 blit). 0=bilinear(종전) / 1=bicubic / 2=bicubic+언샤프.
    // 1.54배 확대 구간에서 bilinear이 디테일을 뭉개는 것을 보정한다. 실기기 육안 확정 후 고정 예정.
    private var upscaleMode: Int = 1
    private var sharpenAmount: Float = 0.35f

    /** 최종 blit에 적용할 90도 배수 회전량 (0..3). */
    private fun screenRotationQuadrant(): Int {
        val k = ((screenRotation / 90) % 4 + 4) % 4
        return if (screenRotationInverted) (4 - k) % 4 else k
    }

    // 상태
    private var isInitialized: Boolean = false
    private var isMirror: Boolean = true

    // 뷰티 필터 설정
    private var beautyConfig: BeautyFilterConfigV2 = BeautyFilterConfigV2()
    private var beautyEnabled: Boolean = false

    // 홍채 검출 결과 (렌즈 오버레이용)
    private var irisResult: IrisResult? = null

    // SDK 렌즈 렌더 실패 상태 (KT 폴백 제거 — 실패는 무음 폴백 대신 명시 신호)
    // null = 정상. 비-null = 마지막 실패 사유 (HUD 표시용, UI 스레드에서 읽음)
    @Volatile var sdkLensFailure: String? = null
        private set
    private var sdkLensFailureLogFrames: Int = 0

    // 검출 결과 수신 시각 (렌더 시점 랜드마크 age 정량화용 — 프레임-랜드마크 시차 로깅)
    @Volatile private var resultReceivedAtMs: Long = 0L
    private var landmarkAgeLogCounter: Int = 0

    // 눈꺼풀 클리핑 temporal hold (FaceMesh 비유효 시 이전 값 유지)
    private var cachedLeftEyeTop: Float = 0.0f
    private var cachedLeftEyeBottom: Float = 1.0f
    private var cachedRightEyeTop: Float = 0.0f
    private var cachedRightEyeBottom: Float = 1.0f

    // (P4-W1-03 정리) avgIrisLum/lastValidFaceTimeMs 제거 — dead 측정. SDK measured-luma가 담당.

    // 렌즈 설정
    private var lensConfig: LensConfig = LensConfig()
    private var lensEnabled: Boolean = false
    private var sdkLensLoggedOnce: Boolean = false

    // Feature flags (P4-W2-01: Sclera Protection + Contact Shadow)
    private var scleraProtectEnabled: Boolean = true   // 기본 ON
    private var contactShadowEnabled: Boolean = false   // 기본 OFF
    private var shadowIntensity: Float = 0.15f          // 기본 강도
    private var maxDetailValue: Float = 1.2f             // 홍채 밝기 보정 상한

    // === One Euro Filter: 눈꺼풀 경계 안정화 (Kotlin 파생 값 — SDK 코어 미커버) ===
    private val glLeftEyeTopFilter = OneEuroFilter(GL_FILTER_MIN_CUTOFF, GL_FILTER_BETA_EYELID, GL_FILTER_D_CUTOFF)
    private val glLeftEyeBottomFilter = OneEuroFilter(GL_FILTER_MIN_CUTOFF, GL_FILTER_BETA_EYELID, GL_FILTER_D_CUTOFF)
    private val glRightEyeTopFilter = OneEuroFilter(GL_FILTER_MIN_CUTOFF, GL_FILTER_BETA_EYELID, GL_FILTER_D_CUTOFF)
    private val glRightEyeBottomFilter = OneEuroFilter(GL_FILTER_MIN_CUTOFF, GL_FILTER_BETA_EYELID, GL_FILTER_D_CUTOFF)

    // === One Euro Filter: 타원 파라미터 안정화 (Kotlin 파생 값 — SDK 코어 미커버) ===
    private val glLeftEllipseCxFilter = OneEuroFilter(GL_FILTER_MIN_CUTOFF, GL_FILTER_BETA_EYELID, GL_FILTER_D_CUTOFF)
    private val glLeftEllipseCyFilter = OneEuroFilter(GL_FILTER_MIN_CUTOFF, GL_FILTER_BETA_EYELID, GL_FILTER_D_CUTOFF)
    private val glLeftEllipseRxIFilter = OneEuroFilter(GL_FILTER_MIN_CUTOFF, GL_FILTER_BETA, GL_FILTER_D_CUTOFF)
    private val glLeftEllipseRxOFilter = OneEuroFilter(GL_FILTER_MIN_CUTOFF, GL_FILTER_BETA, GL_FILTER_D_CUTOFF)
    private val glLeftEllipseRyFilter = OneEuroFilter(GL_FILTER_MIN_CUTOFF, GL_FILTER_BETA, GL_FILTER_D_CUTOFF)
    private val glLeftEllipseRotFilter = OneEuroFilter(GL_FILTER_MIN_CUTOFF, GL_FILTER_BETA, GL_FILTER_D_CUTOFF)
    private val glRightEllipseCxFilter = OneEuroFilter(GL_FILTER_MIN_CUTOFF, GL_FILTER_BETA_EYELID, GL_FILTER_D_CUTOFF)
    private val glRightEllipseCyFilter = OneEuroFilter(GL_FILTER_MIN_CUTOFF, GL_FILTER_BETA_EYELID, GL_FILTER_D_CUTOFF)
    private val glRightEllipseRxIFilter = OneEuroFilter(GL_FILTER_MIN_CUTOFF, GL_FILTER_BETA, GL_FILTER_D_CUTOFF)
    private val glRightEllipseRxOFilter = OneEuroFilter(GL_FILTER_MIN_CUTOFF, GL_FILTER_BETA, GL_FILTER_D_CUTOFF)
    private val glRightEllipseRyFilter = OneEuroFilter(GL_FILTER_MIN_CUTOFF, GL_FILTER_BETA, GL_FILTER_D_CUTOFF)
    private val glRightEllipseRotFilter = OneEuroFilter(GL_FILTER_MIN_CUTOFF, GL_FILTER_BETA, GL_FILTER_D_CUTOFF)

    // P4-W2-02: 타원 파라미터 캐시 (temporal hold, eyelid 캐시와 동일 패턴)

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

    /**
     * EGL 컨텍스트 손실 후 stale GL 핸들 일괄 무효화.
     *
     * 구 컨텍스트의 GL 객체는 이미 파괴됐으므로 glDelete를 호출하면 안 된다(대상이 없거나,
     * 더 위험하게는 새 컨텍스트에서 같은 id로 갓 생성된 객체를 오삭제). 단순히 핸들을 0으로
     * 리셋해, 이후 recreateIntermediateBuffers()/createLensFbo()의 조건부 'delete-before-recreate'가
     * 무해한 no-op이 되도록 한다. nativeLensLoaded도 함께 내려 native lens 재업로드 전까지 게이트를 닫는다.
     */
    private fun markGlHandlesStale() {
        oesTextureId = 0
        oesToRgbProgram = 0
        passthroughProgram = 0
        quadVao = 0
        quadVbo = 0
        beautyOutputTextureId = 0
        lensFboId = 0
        lensOutputTextureId = 0
        nativeLensLoaded = false
        for (i in 0 until ringSize) {
            ringTex[i] = 0
            ringFbo[i] = 0
        }
        ringWrite = 0
        ringCount = 0
        // 구 SurfaceTexture는 파괴된 컨텍스트의 OES에 묶였던 stale — release 후 새로 만든다.
        surfaceTexture?.let { runCatching { it.release() } }
        surfaceTexture = null
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        Log.d(TAG, "onSurfaceCreated")

        // EGL 컨텍스트 (재)생성: 구 컨텍스트의 모든 GL 핸들은 무효다. 0으로 리셋하지 않으면
        // 이후 recreateIntermediateBuffers/createLensFbo의 'delete-before-recreate'가 stale id를
        // 삭제하다가 새 컨텍스트의 갓 생성된 텍스처(id 충돌)를 지워버려 블랙스크린을 유발한다.
        markGlHandlesStale()

        // OpenGL ES 버전 확인
        val version = GLES31.glGetString(GLES31.GL_VERSION)
        Log.d(TAG, "OpenGL ES version: $version")

        // 배경색 설정
        GLES31.glClearColor(0.0f, 0.0f, 0.0f, 1.0f)

        // 셰이더 프로그램 생성
        oesToRgbProgram = createProgram(VERTEX_SHADER, OES_TO_2D_FRAGMENT_SHADER)
        passthroughProgram = createProgram(VERTEX_SHADER, PASSTHROUGH_FRAGMENT_SHADER)

        // Uniform locations 캐시 (OES → RGBA)
        uSTMatrixLocation = GLES31.glGetUniformLocation(oesToRgbProgram, "uSTMatrix")
        uMirrorLocation = GLES31.glGetUniformLocation(oesToRgbProgram, "uMirror")
        uFlipYLocation = GLES31.glGetUniformLocation(oesToRgbProgram, "uFlipY")
        uScaleLocation = GLES31.glGetUniformLocation(oesToRgbProgram, "uScale")
        uRotateLocation = GLES31.glGetUniformLocation(oesToRgbProgram, "uRotate")
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
        // 권한 모달 → 카메라 재시작 등으로 EGL context가 재생성될 때 stale GL 핸들이 남아
        // 다음 frame부터 glError 0x501이 발생하는 회귀를 차단하기 위해 init 직전 명시적 release.
        IrisLensSDK.releaseGpuBeauty()
        val gpuInitResult = IrisLensSDK.initGpuBeauty()
        val gpuSuccess = (gpuInitResult == IrisLensSDK.OK || gpuInitResult == IrisLensSDK.ALREADY_INITIALIZED)
        Log.d(TAG, "GPU Beauty Backend init: $gpuInitResult (success: $gpuSuccess)")

        // GPU Lens Renderer 초기화 (위와 동일 사유)
        IrisLensSDK.releaseGpuLens()
        val gpuLensResult = IrisLensSDK.initGpuLens()
        Log.d(TAG, "GPU Lens Renderer init: $gpuLensResult")

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

        // 중간 ring/FBO는 '카메라 프레임 크기' 기준이다(뷰 크기 아님 — 디스플레이는 Cover 스케일).
        // 카메라 프레임 크기를 이미 알면(setFrameSize 후/복귀) 그 크기로 재생성하고, 최초엔 뷰 크기로
        // 임시 생성한 뒤 카메라 연결 시 setFrameSize가 교정한다.
        // ※ 백그라운드 복귀 시 컨텍스트 보존(preserveEGLContextOnPause)으로 onSurfaceCreated는 생략되나
        //   window surface 재생성으로 onSurfaceChanged는 호출된다. 이때 ring을 뷰 크기(예: 1080x2140)로
        //   덮으면, setFrameSize는 frameWidth 불변(640==640)이라 no-op → ring이 뷰 종횡비에 갇혀
        //   화면이 깨진다(상단 블랙 + 하단 압축). 따라서 알려진 프레임 크기를 우선한다.
        if (frameWidth > 0 && frameHeight > 0) {
            recreateIntermediateBuffers(frameWidth, frameHeight)
        } else {
            recreateIntermediateBuffers(width, height)
        }
    }

    override fun onDrawFrame(gl: GL10?) {
        if (!isInitialized) return

        // SurfaceTexture 업데이트
        surfaceTexture?.updateTexImage()
        surfaceTexture?.getTransformMatrix(stMatrix)

        // frame-sync: 이 프레임의 센서 타임스탬프(ns) — 링 슬롯 태그 + 랜드마크 매칭 키.
        // 분석측 imageInfo.timestamp와 동일 클럭(클럭 게이트 실기기 검증 완료, 06-15).
        val frameTsNs = surfaceTexture?.timestamp ?: 0L

        // 펜딩 렌즈 텍스처 업로드
        uploadPendingLensTexture()

        // 화면 클리어
        GLES31.glClear(GLES31.GL_COLOR_BUFFER_BIT)

        // 1단계: OES → RGBA 변환을 이번 프레임의 링 슬롯에 렌더하고 센서 ts로 태그.
        val writtenIdx = ringWrite
        renderOESToRgba(ringFbo[writtenIdx])
        ringTsNs[writtenIdx] = frameTsNs
        if (ringCount < ringSize) ringCount++
        ringWrite = (ringWrite + 1) % ringSize

        // W4-B3: 검출 슬롯을 단일 스냅샷으로 취득 — 렌즈 좌표 포인터·센서 ts·detected를
        // 모두 같은 슬롯에서 읽어 배경(ts)과 렌즈(좌표)가 서로 다른 프레임이 되는 스큐를 원천 제거.
        // (active index 1회 read 보장 — getDetectionSlotPtr 다중 호출 race 대체)
        val detectionHandle = IrisLensSDK.getActiveDetectionSlot(detSlotMeta)
        val slotTsNs = if (detectionHandle != 0L) detSlotMeta[0] else 0L
        val slotDetected = detectionHandle != 0L && detSlotMeta[1] != 0L

        // frame-sync: 슬롯과 동반된 센서 ts와 |Δ| 최소인 슬롯 선택 (OFF/강등 시 최신 슬롯).
        val sourceIdx = selectFrameSyncSlot(writtenIdx, slotTsNs)

        // 2단계: 렌즈 오버레이 (홍채 위치에 렌즈 합성). 게이트는 슬롯 detected(좌표와 동일 스냅샷).
        var currentTexture = ringTex[sourceIdx]
        if (lensEnabled && nativeLensLoaded && slotDetected) {
            currentTexture = applyGpuLensRenderer(currentTexture, detectionHandle)
        } else if (stabilityLogEnabled && lensEnabled && nativeLensLoaded) {
            // 렌즈 파이프라인 활성 상태에서 검출 실패 시에만 기록
            // (렌즈 미선택/텍스처 미준비 시에는 기록하지 않음)
            onStabilityFrame?.invoke(
                irisResult?.detected ?: false,
                0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 0f, 0f,
                0f, 1f, 0f, 1f,
                false, 0, 0L  // KT 폴백 셰이더 제거로 eyelid hold 캐시 없음
            )
        }

        // 4단계: GPU Beauty 적용
        val beautyApplied = beautyEnabled && beautyConfig.enabled
        var outputTexture = if (beautyApplied) {
            applyGpuBeautyFilter(currentTexture, detectionHandle)
        } else {
            currentTexture
        }

        // 5단계: 화면에 렌더링
        renderToScreen(outputTexture, beautyApplied)

        // 프레임-랜드마크 시차 정량화 (감사: Preview/ImageAnalysis 별도 스트림 — 구조 개선은 ④ 주입 설계에서)
        if (++landmarkAgeLogCounter >= 120) {
            landmarkAgeLogCounter = 0
            val receivedAt = resultReceivedAtMs
            if (receivedAt > 0L) {
                val ageMs = android.os.SystemClock.elapsedRealtime() - receivedAt
                Log.i(TAG, "Landmark age at render: ${ageMs}ms (검출 결과 수신→렌더 시차)")
            }
        }

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
    private fun renderOESToRgba(targetFbo: Int) {
        // FBO 바인딩 (frame-sync: 이번 프레임의 링 슬롯 FBO)
        GLES31.glBindFramebuffer(GLES31.GL_FRAMEBUFFER, targetFbo)
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
        // 화면 회전은 최종 blit 전용 — 링 FBO(렌즈/뷰티 합성 좌표계)는 항상 회전 0을 유지해야
        // 랜드마크 upright 공간과의 계약이 깨지지 않는다. (VERTEX_SHADER 소스 공유 → 명시 필수)
        GLES31.glUniform1i(uRotateLocation, 0)

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
     * frame-sync 슬롯 선택 (트래킹 지연 핸드오프 §3-b). OFF/강등/랜드마크 미수신이면 최신 슬롯
     * (= 현행 동작). ON이면 분석측 랜드마크 프레임 ts와 |Δ| 최소 슬롯(FrameRingSelector).
     * 클럭 도메인 불일치가 연속 확정되면 영구 강등(최신 슬롯)한다.
     */
    private fun selectFrameSyncSlot(latestIdx: Int, snapTs: Long): Int {
        if (!frameSyncActive) return latestIdx
        if (snapTs <= 0L) return latestIdx  // 아직 랜드마크 없음 — 최신, 평가 안 함

        // 클럭 도메인 검증 (LensSim 정본 가드): '검증 미완 && 새 스냅샷 ts'일 때만 1표 집계.
        // 같은 랜드마크 ts가 여러 렌더 프레임에 재사용돼도(렌더 fps > 분석 cadence) 재집계 안 함 →
        // 콜드스타트 묵은 ts 1개로 streak가 누적돼 오강등되는 버그 차단. |Δ|≤1s 1회 관측 시 영구 확정.
        if (!clockDomainChecked && snapTs != clockDomainLastTs) {
            clockDomainLastTs = snapTs
            val minDelta = FrameRingSelector.minAbsDeltaNs(ringTsNs, ringCount, snapTs)
            clockDomainStreak = FrameRingSelector.updateClockDomainStreak(clockDomainStreak, minDelta)
            if (clockDomainStreak == 0) {
                clockDomainChecked = true  // |Δ|≤1s — 같은 센서 클럭 확정, 재검사 불필요
            } else if (clockDomainStreak >= FrameRingSelector.CLOCK_DOMAIN_CONFIRM_STREAK) {
                clockDomainChecked = true
                clockDomainMismatch = true
                Log.w(TAG, "frame-sync: 클럭 도메인 불일치 확정(연속 ${clockDomainStreak}회 |Δ|>1s) — 최신 프레임 영구 강등")
            }
        }

        if (clockDomainMismatch) return latestIdx

        val sel = FrameRingSelector.select(ringTsNs, ringCount, latestIdx, snapTs)
        if (++frameSyncLogCounter >= 120) {
            frameSyncLogCounter = 0
            val md = FrameRingSelector.minAbsDeltaNs(ringTsNs, ringCount, snapTs)
            Log.i(TAG, "frame-sync: minΔ=${"%.1f".format(md / 1e6)}ms slot=$sel count=$ringCount latest=$latestIdx")
        }
        return if (sel < 0) latestIdx else sel
    }

    // W4-B3: setLandmarkFrameTimestamp(별도 volatile ts 사이드채널)는 제거됨. 분석 프레임 센서 ns는
    // 이제 updateDetectionSlot(result, frameTsNs)로 렌즈 좌표와 한 슬롯에 원자 결속되고, GL 스레드는
    // getActiveDetectionSlot 단일 스냅샷으로 ts·좌표·detected를 함께 읽어 1프레임 스큐를 원천 제거한다.

    /** 킬스위치 — frame-sync ON 시 클럭 도메인 검증/강등 상태를 리셋해 재시도를 허용한다. */
    fun setFrameSyncEnabled(enabled: Boolean) {
        if (enabled && !frameSyncActive) {
            clockDomainStreak = 0
            clockDomainChecked = false
            clockDomainMismatch = false
            clockDomainLastTs = 0L
        }
        frameSyncActive = enabled
    }

    /**
     * 펜딩 렌즈 텍스처 업로드 (GL 스레드에서 실행)
     */
    private fun uploadPendingLensTexture() {
        val bitmap = pendingLensBitmap ?: return
        // GPU lens 미초기화면 이번 프레임은 보류하고 비트맵을 유지해 다음 프레임 재시도한다
        // (컨텍스트 재생성 직후 initGpuLens 완료 전 호출되어도 렌즈가 영구 소실되지 않게).
        if (!IrisLensSDK.isGpuLensInitialized()) {
            return
        }
        pendingLensBitmap = null
        val skuId = pendingLensSkuId  // P6-W7: 스레드 안전하게 로컬 캡처

        // KT GL 텍스처 업로드는 제거(KT fallback 셰이더 폐지 후 native 렌더에 미사용 — 게이트/삭제
        // 위험만 만들던 vestigial 경로). native GPULensRenderer로만 RGBA 업로드하고, 로드 성공
        // 여부를 렌더 게이트(nativeLensLoaded)로 사용한다.
        val rgbaBytes = ByteArray(bitmap.width * bitmap.height * 4)
        bitmap.copyPixelsToBuffer(java.nio.ByteBuffer.wrap(rgbaBytes))
        val loadResult = IrisLensSDK.loadLensTexture(rgbaBytes, bitmap.width, bitmap.height, skuId)
        nativeLensLoaded = (loadResult == IrisLensSDK.OK)
        Log.d(TAG, "SDK lens texture loaded: ${bitmap.width}x${bitmap.height}, sku=$skuId, result=$loadResult, ok=$nativeLensLoaded")
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
     * SDK GPULensRenderer를 통한 렌즈 렌더링
     *
     * SDK C++ 코어의 GPULensRenderer를 호출하여 렌즈를 합성합니다.
     * 실패 시 무음 폴백 없이 렌즈 미적용 + sdkLensFailure 신호로 명시 처리합니다
     * (감사 finding: KT fallback 셰이더 blendMode 의미 불일치 + 프레임 단위 무음 폴백 제거).
     */
    private fun applyGpuLensRenderer(inputTexture: Int, detectionHandle: Long): Int {
        if (!IrisLensSDK.isGpuLensInitialized()) {
            reportSdkLensFailure("GPU Lens 미초기화")
            return inputTexture
        }

        val texWidth = if (frameWidth > 0) frameWidth else viewWidth
        val texHeight = if (frameHeight > 0) frameHeight else viewHeight

        // detectionHandle은 onDrawFrame의 단일 슬롯 스냅샷(getActiveDetectionSlot)에서 전달됨 (W4-B3)

        lensConfig.isMirror = isMirror

        val outputTexture = IrisLensSDK.renderLensTexture(
            inputTexture,
            texWidth,
            texHeight,
            detectionHandle,
            lensConfig
        )

        return if (outputTexture != 0 && outputTexture != inputTexture) {
            if (!sdkLensLoggedOnce) {
                Log.i(TAG, "SDK C++ GPULensRenderer active (out=$outputTexture, ${texWidth}x${texHeight})")
                sdkLensLoggedOnce = true
            }
            sdkLensFailure = null
            outputTexture
        } else {
            // 무음 폴백 금지: 렌즈 미적용으로 명시 실패 (검증 통로가 거짓말하지 않게)
            reportSdkLensFailure("렌즈 렌더 실패 (output=$outputTexture)")
            inputTexture
        }
    }

    /** SDK 렌즈 실패를 기록한다 — HUD 신호 설정 + 스로틀 로그(60프레임당 1회). */
    private fun reportSdkLensFailure(reason: String) {
        sdkLensFailure = reason
        if (sdkLensFailureLogFrames <= 0) {
            Log.e(TAG, "SDK lens render FAILED — 렌즈 미적용 (무음 폴백 제거됨): $reason")
            sdkLensFailureLogFrames = 60
        } else {
            sdkLensFailureLogFrames--
        }
    }

    /**
     * GPU 뷰티 필터 적용
     *
     * @param inputTexture 입력 텍스처 ID
     * @return 출력 텍스처 ID
     */
    private fun applyGpuBeautyFilter(inputTexture: Int, detectionHandle: Long): Int {
        // 프레임 크기 사용
        val texWidth = if (frameWidth > 0) frameWidth else viewWidth
        val texHeight = if (frameHeight > 0) frameHeight else viewHeight

        // detectionHandle은 onDrawFrame의 단일 슬롯 스냅샷(getActiveDetectionSlot)에서 전달됨 (W4-B3, lock-free)

        // (성능 정리) 매 프레임 뷰티 설정 Log.d 제거 — 활성 프레임마다 문자열 보간 비용이었음.

        // GPU Beauty Backend 호출 (JNI) - Detection Handle
        // (P8-W2-D: LUT 곁가지 시그니처 제거됨)
        val outputTexture = IrisLensSDK.applyBeautyFilterTextureV2(
            inputTexture,
            texWidth,
            texHeight,
            beautyConfig,
            detectionHandle
        )

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
        val rotateLocation = GLES31.glGetUniformLocation(passthroughProgram, "uRotate")
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

        // 링 FBO 내용은 '기기 natural orientation 기준 upright'다(stMatrix + targetRotation=ROTATION_0 핀).
        // 화면이 회전해 있으면 그만큼 최종 blit에서 되돌려야 사용자 눈에 정립으로 보인다.
        val rotK = screenRotationQuadrant()
        val rotSwap = (rotK == 1 || rotK == 3)

        val texAspectUpright = texWidth.toFloat() / texHeight.toFloat()
        // 회전 후 화면에서 콘텐츠가 실제로 갖는 종횡비
        val texAspect = if (rotSwap) 1.0f / texAspectUpright else texAspectUpright
        val viewAspect = viewWidth.toFloat() / viewHeight.toFloat()

        // Cover 모드: 화면을 꽉 채우고 넘치는 부분은 GL viewport에 의해 자동 crop
        val (screenScaleX, screenScaleY) = if (texAspect > viewAspect) {
            // 텍스처가 더 넓음 → 높이 채우고 좌우 넘침 (crop)
            (texAspect / viewAspect) to 1.0f
        } else {
            // 텍스처가 더 좁음 → 너비 채우고 상하 넘침 (crop)
            1.0f to (viewAspect / texAspect)
        }
        // 셰이더는 [scale → rotate] 순서다. 90도 회전은 축 교환이므로 화면 기준 배율을
        // 축만 바꿔 넘기면 회전 후 정확히 (screenScaleX, screenScaleY)가 된다.
        val (scaleX, scaleY) =
            if (rotSwap) screenScaleY to screenScaleX else screenScaleX to screenScaleY
        // displayZoom: 등방이라 축 교환(rotSwap)과 무관하게 양축에 동일 배율.
        GLES31.glUniform2f(scaleLocation, scaleX * displayZoom, scaleY * displayZoom)

        // 업스케일 품질: 소스 텍셀 크기 + 모드/샤프닝 강도 주입.
        // texWidth/texHeight는 회전 보정 전(소스 텍스처 실제 픽셀)이어야 bicubic 좌표가 맞는다.
        val srcW = if (frameWidth > 0) frameWidth else viewWidth
        val srcH = if (frameHeight > 0) frameHeight else viewHeight
        GLES31.glUniform2f(
            GLES31.glGetUniformLocation(passthroughProgram, "uTexSize"),
            srcW.toFloat(), srcH.toFloat()
        )
        GLES31.glUniform1i(
            GLES31.glGetUniformLocation(passthroughProgram, "uUpscaleMode"), upscaleMode
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(passthroughProgram, "uSharpen"), sharpenAmount
        )
        GLES31.glUniform1i(rotateLocation, rotK)

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
        this.resultReceivedAtMs = android.os.SystemClock.elapsedRealtime()
    }

    // (P4-W1-03 정리) updateAvgIrisLum/avgIrisLum EMA + resetTemporalState 제거 —
    // avgIrisLum은 읽는 곳 0의 dead. resetTemporalState는 dead가 된 lastValidFaceTimeMs 전용이었음
    // (블링크·눈꺼풀 EMA는 자체 dt 관리라 무관).

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
     * Sclera Protection 활성화/비활성화 (P4-W2-01)
     */
    fun setScleraProtect(enabled: Boolean) {
        this.scleraProtectEnabled = enabled
    }

    /**
     * Contact Shadow 활성화/비활성화 (P4-W2-01)
     */
    fun setContactShadow(enabled: Boolean, intensity: Float = 0.15f) {
        this.contactShadowEnabled = enabled
        this.shadowIntensity = intensity.coerceIn(0.0f, 0.25f)
    }

    /**
     * Color Replace 홍채 밝기 보정 상한 설정 (P4-W2-01)
     */
    fun setMaxDetail(value: Float) {
        this.maxDetailValue = value.coerceIn(0.5f, 1.5f)
    }

    /**
     * 렌즈 텍스처 설정 (비트맵)
     *
     * GL 스레드가 아닌 곳에서 호출해도 안전 (펜딩 처리)
     */
    fun setLensTexture(bitmap: Bitmap?, skuId: String = "") {
        if (bitmap == null) {
            // 렌즈 제거 — 게이트를 닫는다(native 텍스처는 다음 렌즈 선택 시 덮어쓰기됨).
            pendingLensBitmap = null
            pendingLensSkuId = ""  // P6-W7
            lensEnabled = false
            nativeLensLoaded = false
        } else {
            // 새 렌즈 설정 (GL 스레드에서 업로드)
            pendingLensBitmap = bitmap
            pendingLensSkuId = skuId  // P6-W7
            lensEnabled = true
        }
    }

    /**
     * 렌즈 활성화 여부 반환
     */
    fun isLensEnabled(): Boolean = lensEnabled

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
     * 화면(디스플레이) 회전 설정 (0, 90, 180, 270).
     *
     * 최종 blit에만 반영되며 링 FBO·랜드마크 좌표계는 건드리지 않는다
     * (렌즈/뷰티 합성은 회전 영향 0). 세로 고정에서는 항상 0.
     */
    fun setScreenRotation(rotation: Int) {
        if (this.screenRotation != rotation) {
            this.screenRotation = rotation
            Log.d(TAG, "Screen rotation set: $rotation (quadrant=${screenRotationQuadrant()})")
        }
    }

    /** 회전 부호 A/B 토글 (실기기 육안 확정용). */
    fun setScreenRotationInverted(inverted: Boolean) {
        this.screenRotationInverted = inverted
        Log.i(TAG, "Screen rotation sign inverted → $inverted (quadrant=${screenRotationQuadrant()})")
    }

    /**
     * 최종 blit 등방 확대 배율 설정 (FOV 확대). 1.0 = 무확대(현행 동일).
     * 캡처·랜드마크·렌즈 좌표계는 불변이라 렌즈 정합·추적에 영향 없다.
     */
    fun setDisplayZoom(zoom: Float) {
        val z = zoom.coerceIn(1.0f, 1.5f)
        if (this.displayZoom != z) {
            this.displayZoom = z
            Log.i(TAG, "Display zoom set: $z")
        }
    }

    /** 업스케일 품질 모드 (0=bilinear, 1=bicubic, 2=bicubic+언샤프). */
    fun setUpscaleMode(mode: Int, sharpen: Float = 0.35f) {
        upscaleMode = mode.coerceIn(0, 2)
        sharpenAmount = sharpen.coerceIn(0.0f, 1.5f)
        Log.i(TAG, "Upscale mode → $upscaleMode (sharpen=$sharpenAmount)")
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
        // 기존 링버퍼 삭제
        deleteRingBuffers()

        // frame-sync RGBA 링버퍼 ringSize장 생성 (각 = 2D 텍스처 + FBO)
        GLES31.glGenTextures(ringSize, ringTex, 0)
        GLES31.glGenFramebuffers(ringSize, ringFbo, 0)
        for (i in 0 until ringSize) {
            GLES31.glBindTexture(GLES31.GL_TEXTURE_2D, ringTex[i])
            GLES31.glTexImage2D(
                GLES31.GL_TEXTURE_2D, 0, GLES31.GL_RGBA,
                width, height, 0,
                GLES31.GL_RGBA, GLES31.GL_UNSIGNED_BYTE, null
            )
            GLES31.glTexParameteri(GLES31.GL_TEXTURE_2D, GLES31.GL_TEXTURE_MIN_FILTER, GLES31.GL_LINEAR)
            GLES31.glTexParameteri(GLES31.GL_TEXTURE_2D, GLES31.GL_TEXTURE_MAG_FILTER, GLES31.GL_LINEAR)
            GLES31.glTexParameteri(GLES31.GL_TEXTURE_2D, GLES31.GL_TEXTURE_WRAP_S, GLES31.GL_CLAMP_TO_EDGE)
            GLES31.glTexParameteri(GLES31.GL_TEXTURE_2D, GLES31.GL_TEXTURE_WRAP_T, GLES31.GL_CLAMP_TO_EDGE)

            GLES31.glBindFramebuffer(GLES31.GL_FRAMEBUFFER, ringFbo[i])
            GLES31.glFramebufferTexture2D(
                GLES31.GL_FRAMEBUFFER, GLES31.GL_COLOR_ATTACHMENT0,
                GLES31.GL_TEXTURE_2D, ringTex[i], 0
            )
            val status = GLES31.glCheckFramebufferStatus(GLES31.GL_FRAMEBUFFER)
            if (status != GLES31.GL_FRAMEBUFFER_COMPLETE) {
                Log.e(TAG, "Ring FBO[$i] is not complete: $status")
            }
            ringTsNs[i] = 0L
        }
        GLES31.glBindFramebuffer(GLES31.GL_FRAMEBUFFER, 0)

        // 링 상태 리셋 (크기 변경 시 묵은 프레임/타임스탬프 폐기)
        ringWrite = 0
        ringCount = 0
        clockDomainStreak = 0
        clockDomainChecked = false
        clockDomainMismatch = false
        clockDomainLastTs = 0L

        // 렌즈 FBO도 재생성
        if (lensFboId != 0) {
            createLensFbo()
        }

        Log.d(TAG, "Intermediate ring buffers created: ${width}x${height} x$ringSize")
    }

    /** frame-sync 링버퍼(텍스처 + FBO) 일괄 해제. */
    private fun deleteRingBuffers() {
        for (i in 0 until ringSize) {
            if (ringTex[i] != 0) {
                GLES31.glDeleteTextures(1, intArrayOf(ringTex[i]), 0)
                ringTex[i] = 0
            }
            if (ringFbo[i] != 0) {
                GLES31.glDeleteFramebuffers(1, intArrayOf(ringFbo[i]), 0)
                ringFbo[i] = 0
            }
        }
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
        // SDK GPU 렌즈 렌더러 해제
        IrisLensSDK.releaseGpuLens()

        // beautyOutputTextureId는 TexturePool 소유 → releaseGpuBeauty()에서 일괄 해제
        // 여기서 releaseTexture() 호출하면 이중 해제 발생
        beautyOutputTextureId = 0

        if (oesTextureId != 0) {
            GLES31.glDeleteTextures(1, intArrayOf(oesTextureId), 0)
        }
        // frame-sync 링버퍼 해제
        deleteRingBuffers()
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

        if (lensOutputTextureId != 0) {
            GLES31.glDeleteTextures(1, intArrayOf(lensOutputTextureId), 0)
        }
        if (lensFboId != 0) {
            GLES31.glDeleteFramebuffers(1, intArrayOf(lensFboId), 0)
        }
        pendingLensBitmap = null
        nativeLensLoaded = false

        surfaceTexture?.release()
        surfaceTexture = null

        isInitialized = false
        Log.d(TAG, "Resources released")
    }
}
