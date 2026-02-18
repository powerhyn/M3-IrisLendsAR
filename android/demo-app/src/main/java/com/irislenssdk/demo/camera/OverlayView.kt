/**
 * IrisLensSDK Android - OverlayView
 *
 * 홍채 검출 결과를 시각화하고 렌즈 텍스처를 오버레이하는 커스텀 뷰
 * - 홍채 위치 마커
 * - 렌즈 텍스처 오버레이
 * - 디버그 정보 표시
 *
 * @version 1.1.0
 */
package com.irislenssdk.demo.camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.DashPathEffect
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.graphics.RectF
import android.util.AttributeSet
import android.util.Log
import android.view.View
import com.irislenssdk.IrisResult
import com.irislenssdk.LensConfig
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

/**
 * 홍채 오버레이 뷰
 *
 * 카메라 프리뷰 위에 오버레이되어 홍채 검출 결과를 표시하고 렌즈 텍스처를 렌더링
 */
class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    companion object {
        private const val TAG = "OverlayView"

        // 색상
        private const val COLOR_IRIS_CIRCLE = 0xFF00FF00.toInt()  // Green
        private const val COLOR_IRIS_CENTER = 0xFFFF0000.toInt()  // Red
        private const val COLOR_FACE_RECT = 0xFF00FFFF.toInt()    // Cyan
        private const val COLOR_DEBUG_TEXT = 0xFFFFFFFF.toInt()   // White
        private const val COLOR_DEBUG_BG = 0x80000000.toInt()     // Black 50%
        private const val COLOR_FACE_MESH_LINE = 0x8000FF00.toInt()  // Green 50%
        private const val COLOR_FACE_MESH_POINT = 0xFFFFFF00.toInt() // Yellow

        // 디버그 원 크기
        private const val CENTER_DOT_RADIUS = 6f
        private const val MESH_POINT_RADIUS = 2f
        private const val MESH_LINE_WIDTH = 1f
        private const val IRIS_POINT_RADIUS = 5f  // 홍채 포인트는 더 크게

        // 홍채 랜드마크 색상
        private const val COLOR_IRIS_LANDMARK = 0xFFFF00FF.toInt()  // Magenta

        // MediaPipe Face Mesh with Iris 랜드마크 인덱스
        // 왼쪽 홍채 (화면상 오른쪽): 중심 468, 경계 469-472
        private const val LEFT_IRIS_CENTER = 468
        private val LEFT_IRIS_POINTS = intArrayOf(469, 470, 471, 472)
        // 오른쪽 홍채 (화면상 왼쪽): 중심 473, 경계 474-477
        private const val RIGHT_IRIS_CENTER = 473
        private val RIGHT_IRIS_POINTS = intArrayOf(474, 475, 476, 477)

        // 렌즈 렌더링 설정
        // 홍채 반지름 대비 렌즈 크기 배율 (1.0 = 홍채 크기와 동일)
        private const val LENS_SCALE_FACTOR = 1.0f

        // 스무딩 설정 (부드러운 추적)
        // 0.0 = 변화 없음, 1.0 = 즉시 반영
        // 낮은 값 = 더 부드러운 추적, 높은 값 = 즉각 반응
        // ISS-002: 0.25 → 0.7로 증가 (빠른 반응)
        private const val SMOOTHING_FACTOR = 0.7f

        // 렌즈 크기 양자화 단위 (픽셀) - 자글거림 방지
        private const val LENS_SIZE_QUANTIZATION_STEP = 2f

        // Radius 변화 최소 임계값 (픽셀) - 미세한 변화 무시
        // 이 값 이하의 radius 변화는 노이즈로 간주하여 무시
        private const val RADIUS_CHANGE_THRESHOLD = 0.5f

        // 스케일된 렌즈 비트맵 캐시 무효화 임계값 (픽셀 단위)
        // 이 값 이상 크기가 변할 때만 새 비트맵 생성 (꿀렁거림 방지)
        private const val LENS_CACHE_THRESHOLD = 4

        // 최소 렌더링 신뢰도 임계값 (False Positive 방지)
        // 이 값 미만의 신뢰도를 가진 검출 결과는 렌더링하지 않음
        // 허공/천장 감지 문제 해결을 위해 추가
        private const val MIN_RENDER_CONFIDENCE = 0.5f

        // 검출 실패 시 렌즈 유지 시간 (밀리초)
        // 이 시간 동안 얼굴 인식이 실패해도 마지막 유효한 위치에 렌즈 유지
        // 깜빡임 방지를 위한 임계값
        private const val DETECTION_TIMEOUT_MS = 1000L  // 1초 (mesh, debug info 등)
        private const val LENS_PERSISTENCE_TIMEOUT_MS = 2000L  // 2초 (렌즈 전용 - 더 긴 유지)

        // One Euro Filter 파라미터
        // minCutoff: 정지 시 최소 컷오프 주파수. 낮을수록 스무딩 강함.
        //   15.0 → α≈0.61 (pass-through), 1.5 → α≈0.14 (효과적 스무딩)
        // beta: 높을수록 이동 시 필터가 빨리 풀림 (빠른 추적)
        private const val ONE_EURO_MIN_CUTOFF = 3.0f   // 정지 시 스무딩 + 이동 초반 반응성 균형
        private const val ONE_EURO_BETA = 7.0f         // 이동 시 필터 즉시 해제 수준
        private const val ONE_EURO_D_CUTOFF = 1.0f     // 미분 컷오프 주파수

        // 눈 윤곽 랜드마크 인덱스 (MediaPipe Face Mesh 468개 기준)
        // 왼쪽 눈 (화면상 오른쪽) - 시계방향 순서
        private val LEFT_EYE_CONTOUR_INDICES = intArrayOf(
            33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7
        )
        // 오른쪽 눈 (화면상 왼쪽) - 시계방향 순서
        private val RIGHT_EYE_CONTOUR_INDICES = intArrayOf(
            362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382
        )
    }

    // 검출 결과
    private var irisResult: IrisResult? = null

    // 렌즈 설정
    private var lensConfig: LensConfig = LensConfig()

    // 렌즈 텍스처
    private var lensTexture: Bitmap? = null

    // 이미지 크기 (분석 이미지)
    private var imageWidth: Int = 640
    private var imageHeight: Int = 480

    // 미러링 (전면 카메라)
    private var isMirror: Boolean = true

    // 디버그 모드
    var debugMode: Boolean = false

    /**
     * 화면 매핑 정책.
     * - FIT: 이미지를 뷰 안에 맞춤 (레터박스 가능). CPU 모드에서 PreviewView FILL_CENTER와 사용.
     * - COVER: 이미지가 뷰를 완전히 채움 (넘치는 부분 crop). GPU 모드의 GL 출력과 동일.
     */
    enum class ScreenMappingMode { FIT, COVER }

    var screenMappingMode: ScreenMappingMode = ScreenMappingMode.FIT

    // 하위 호환: 기존 gpuMode 사용처 지원
    @Deprecated("screenMappingMode를 직접 사용하세요", ReplaceWith("screenMappingMode"))
    var gpuMode: Boolean
        get() = screenMappingMode == ScreenMappingMode.COVER
        set(value) { screenMappingMode = if (value) ScreenMappingMode.COVER else ScreenMappingMode.FIT }

    // === One Euro Filter를 사용한 스무딩 (깜빡임/흔들거림 방지) ===
    private val leftXFilter = OneEuroFilter(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_D_CUTOFF)
    private val leftYFilter = OneEuroFilter(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_D_CUTOFF)
    private val leftRadiusFilter = OneEuroFilter(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA * 0.5f, ONE_EURO_D_CUTOFF)
    private val rightXFilter = OneEuroFilter(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_D_CUTOFF)
    private val rightYFilter = OneEuroFilter(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_D_CUTOFF)
    private val rightRadiusFilter = OneEuroFilter(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA * 0.5f, ONE_EURO_D_CUTOFF)

    // 필터링된 결과값
    private var filteredLeftX: Float = 0f
    private var filteredLeftY: Float = 0f
    private var filteredLeftRadius: Float = 0f
    private var filteredRightX: Float = 0f
    private var filteredRightY: Float = 0f
    private var filteredRightRadius: Float = 0f
    private var lastTimestamp: Long = 0L

    // 각 눈이 한 번이라도 검출되었는지 추적 (깜빡임 방지)
    // 한 번 검출되면 이후 검출 실패 시에도 마지막 위치에 렌즈 유지
    private var hasLeftEverDetected: Boolean = false
    private var hasRightEverDetected: Boolean = false

    // 마지막 유효한 검출 시간 (렌즈 유지 타임아웃용)
    private var lastValidDetectionTime: Long = 0L

    // 마지막 유효한 검출 결과 캐시 (렌더링 유지용)
    // 참조가 아닌 값 복사 (IrisResult가 재사용되므로)
    private var cachedDetected: Boolean = false
    private var cachedConfidence: Float = 0f
    private var cachedLeftDetected: Boolean = false
    private var cachedRightDetected: Boolean = false
    private var cachedFaceMeshValid: Boolean = false
    private var cachedFaceMesh: FloatArray? = null

    // 눈 영역 클리핑용 Path
    private val leftEyePath = Path()
    private val rightEyePath = Path()
    private var eyeClippingEnabled: Boolean = true  // 눈 영역 클리핑 활성화 여부

    // === 스케일된 렌즈 비트맵 캐시 (꿀렁거림 방지) ===
    // 매 프레임 새로 스케일링하지 않고 캐시된 비트맵 재사용
    private var cachedLeftLensBitmap: Bitmap? = null
    private var cachedRightLensBitmap: Bitmap? = null
    private var cachedLeftLensSize: Int = 0
    private var cachedRightLensSize: Int = 0

    // Face Mesh 표시 모드
    var showFaceMesh: Boolean = false

    // 얼굴 검출 영역 표시 (Face Detection 결과)
    var showFaceRect: Boolean = false

    // 렌즈 표시 여부
    var showLens: Boolean = true

    // === 필터된 프레임 렌더링 ===
    // 뷰티 필터가 적용된 프레임을 배경으로 표시
    private var filteredFrame: Bitmap? = null
    var showFilteredFrame: Boolean = false

    // Paint 객체들 (재사용)
    private val irisPaint = Paint().apply {
        color = COLOR_IRIS_CIRCLE
        style = Paint.Style.STROKE
        strokeWidth = 4f
        isAntiAlias = true
    }

    private val centerPaint = Paint().apply {
        color = COLOR_IRIS_CENTER
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val faceRectPaint = Paint().apply {
        color = COLOR_FACE_RECT
        style = Paint.Style.STROKE
        strokeWidth = 2f
        isAntiAlias = true
    }

    private val debugTextPaint = Paint().apply {
        color = COLOR_DEBUG_TEXT
        textSize = 32f
        isAntiAlias = true
    }

    private val debugBgPaint = Paint().apply {
        color = COLOR_DEBUG_BG
        style = Paint.Style.FILL
    }

    private val meshLinePaint = Paint().apply {
        color = COLOR_FACE_MESH_LINE
        style = Paint.Style.STROKE
        strokeWidth = MESH_LINE_WIDTH
        isAntiAlias = true
    }

    private val meshPointPaint = Paint().apply {
        color = COLOR_FACE_MESH_POINT
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    // ISS-004 Fix-A: Raw 홍채 반경 원 (파란색 점선 - 실제 홍채 경계)
    private val rawIrisPaint = Paint().apply {
        color = 0xFF4488FF.toInt()  // 밝은 파란색
        style = Paint.Style.STROKE
        strokeWidth = 2f
        pathEffect = DashPathEffect(floatArrayOf(8f, 6f), 0f)
        isAntiAlias = true
    }

    // 홍채 랜드마크용 Paint (마젠타 - 눈에 잘 띄도록)
    private val irisLandmarkPaint = Paint().apply {
        color = COLOR_IRIS_LANDMARK
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    // 렌즈 렌더링용 Paint
    private val lensPaint = Paint().apply {
        isAntiAlias = true
        isFilterBitmap = true
        isDither = true
    }

    // 렌즈 블렌딩용 Paint (Multiply 모드)
    private val lensBlendPaint = Paint().apply {
        isAntiAlias = true
        isFilterBitmap = true
        isDither = true
        xfermode = PorterDuffXfermode(PorterDuff.Mode.MULTIPLY)
    }

    // 필터된 프레임 렌더링용 Paint
    private val framePaint = Paint().apply {
        isAntiAlias = true
        isFilterBitmap = true
    }

    // 임시 RectF (재사용)
    private val tempRect = RectF()
    private val lensDestRect = RectF()

    // 변환 매트릭스 (재사용)
    private val lensMatrix = Matrix()

    /**
     * 화면 변환 스케일 팩터를 계산합니다.
     *
     * @param imageW 분석 이미지 너비
     * @param imageH 분석 이미지 높이
     * @param viewW 뷰 너비
     * @param viewH 뷰 높이
     * @param fitMode true=fit (GL 출력과 동일), false=fill-center (PreviewView와 동일)
     * @return 스케일 팩터
     */
    private fun computeScreenTransform(
        imageW: Int, imageH: Int,
        viewW: Int, viewH: Int,
        fitMode: Boolean
    ): Float {
        return if (fitMode) {
            // fit: 이미지가 뷰 안에 맞춤 (레터박스 가능)
            kotlin.math.min(viewW.toFloat() / imageW, viewH.toFloat() / imageH)
        } else {
            // fill-center: 이미지가 뷰를 완전히 채움 (넘치는 부분 잘림)
            max(viewW.toFloat() / imageW, viewH.toFloat() / imageH)
        }
    }

    /**
     * 홍채 검출 결과 설정 (One Euro Filter 적용)
     *
     * One Euro Filter는 속도에 따라 적응적으로 스무딩 강도를 조절:
     * - 느린 움직임: 강한 스무딩 (떨림 제거)
     * - 빠른 움직임: 약한 스무딩 (반응성 유지)
     *
     * @param result 검출 결과
     * @param width 분석 이미지 너비
     * @param height 분석 이미지 높이
     * @param mirror 미러링 여부 (전면 카메라)
     */
    fun setIrisResult(result: IrisResult?, width: Int, height: Int, mirror: Boolean) {
        this.irisResult = result
        this.imageWidth = width
        this.imageHeight = height
        this.isMirror = mirror

        val currentTime = System.currentTimeMillis()

        // One Euro Filter를 사용한 스무딩
        result?.let {
            // 깜빡임 방지: 홍채가 검출되면 타임아웃 리셋 (confidence와 무관)
            // confidence 체크는 좌표 업데이트에만 적용
            val hasAnyIrisDetection = it.detected && (it.leftDetected || it.rightDetected)

            if (hasAnyIrisDetection) {
                // 홍채 검출됨 - 타임아웃 리셋 (렌즈 유지)
                lastValidDetectionTime = currentTime
                lastTimestamp = currentTime

                // 값 복사 (IrisResult가 재사용되어 다음 프레임에서 reset()되므로)
                cachedDetected = it.detected
                cachedConfidence = it.confidence
                cachedLeftDetected = it.leftDetected
                cachedRightDetected = it.rightDetected
                cachedFaceMeshValid = it.faceMeshValid

                // FaceMesh 복사 (클리핑용)
                it.faceMesh?.let { mesh ->
                    if (cachedFaceMesh == null || cachedFaceMesh!!.size != mesh.size) {
                        cachedFaceMesh = mesh.copyOf()
                    } else {
                        System.arraycopy(mesh, 0, cachedFaceMesh!!, 0, mesh.size)
                    }
                }

                // 좌표 업데이트는 충분한 confidence가 있을 때만
                // (낮은 confidence에서는 좌표가 부정확할 수 있음)
                // 단, 첫 검출(radius=0)에서는 confidence 무관하게 업데이트 (렌즈 표시 위해)
                val hasGoodConfidence = it.confidence >= MIN_RENDER_CONFIDENCE

                // 왼쪽 눈 필터링
                if (it.leftDetected) {
                    hasLeftEverDetected = true
                    // 첫 검출이거나 confidence가 충분하면 업데이트
                    if (hasGoodConfidence || filteredLeftRadius == 0f) {
                        filteredLeftX = leftXFilter.filter(it.leftIrisX, currentTime)
                        filteredLeftY = leftYFilter.filter(it.leftIrisY, currentTime)
                        filteredLeftRadius = leftRadiusFilter.filter(it.leftRadius, currentTime)
                    }
                    // confidence 낮으면 마지막 필터링 값 유지 (깜빡임 방지)
                }

                // 오른쪽 눈 필터링
                if (it.rightDetected) {
                    hasRightEverDetected = true
                    // 첫 검출이거나 confidence가 충분하면 업데이트
                    if (hasGoodConfidence || filteredRightRadius == 0f) {
                        filteredRightX = rightXFilter.filter(it.rightIrisX, currentTime)
                        filteredRightY = rightYFilter.filter(it.rightIrisY, currentTime)
                        filteredRightRadius = rightRadiusFilter.filter(it.rightRadius, currentTime)
                    }
                    // confidence 낮으면 마지막 필터링 값 유지 (깜빡임 방지)
                }
            }
            // 홍채 검출 실패 시에도 필터 상태 유지 (타임아웃 전까지 마지막 위치에 렌즈 유지)
        }
        // result가 null이어도 필터 상태 유지 (타임아웃 전까지 마지막 위치에 렌즈 유지)

        invalidate()
    }

    /**
     * 필터된 프레임 설정
     *
     * 뷰티 필터가 적용된 프레임을 배경으로 표시하기 위해 설정합니다.
     * 이전 프레임은 자동으로 recycled됩니다.
     *
     * @param bitmap 필터된 프레임 (null이면 표시 안함)
     */
    fun setFilteredFrame(bitmap: Bitmap?) {
        // 이전 비트맵 해제 (새 비트맵과 다른 경우에만)
        val oldFrame = filteredFrame
        if (oldFrame != null && oldFrame != bitmap && !oldFrame.isRecycled) {
            oldFrame.recycle()
        }
        filteredFrame = bitmap
    }

    /**
     * 홍채 검출 결과와 필터된 프레임을 함께 설정
     *
     * @param result 검출 결과
     * @param width 분석 이미지 너비
     * @param height 분석 이미지 높이
     * @param mirror 미러링 여부 (전면 카메라)
     * @param filtered 필터된 프레임 (null이면 필터 미적용)
     */
    fun setIrisResult(result: IrisResult?, width: Int, height: Int, mirror: Boolean, filtered: Bitmap?) {
        setFilteredFrame(filtered)
        setIrisResult(result, width, height, mirror)
    }

    /**
     * 눈 영역 클리핑 활성화/비활성화
     */
    fun setEyeClippingEnabled(enabled: Boolean) {
        eyeClippingEnabled = enabled
        invalidate()
    }

    /**
     * 렌즈 설정 업데이트
     */
    fun setLensConfig(config: LensConfig) {
        this.lensConfig = config
        invalidate()
    }

    /**
     * 렌즈 텍스처 설정
     *
     * 텍스처 변경 시 캐시된 스케일 비트맵 무효화
     */
    fun setLensTexture(texture: Bitmap?) {
        this.lensTexture = texture

        // 텍스처 변경 시 캐시 무효화
        cachedLeftLensBitmap?.recycle()
        cachedLeftLensBitmap = null
        cachedLeftLensSize = 0
        cachedRightLensBitmap?.recycle()
        cachedRightLensBitmap = null
        cachedRightLensSize = 0

        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        // 필터된 프레임 배경 렌더링 (뷰티 필터 활성화 시)
        if (showFilteredFrame) {
            filteredFrame?.let { frame ->
                if (!frame.isRecycled) {
                    drawFilteredFrame(canvas, frame)
                }
            }
        }

        val currentTime = System.currentTimeMillis()
        val timeSinceLastValid = currentTime - lastValidDetectionTime

        // 캐시된 값 기반으로 유효한 검출 확인
        val hasValidDetection = cachedDetected && cachedConfidence >= MIN_RENDER_CONFIDENCE

        // [중요] 렌즈는 별도의 긴 타임아웃 사용 (깜빡임 방지)
        // 렌즈 렌더링 조건: 유효한 필터 좌표가 있고 렌즈 타임아웃 내
        val shouldRenderLens = (hasLeftEverDetected || hasRightEverDetected) &&
            (filteredLeftRadius > 0 || filteredRightRadius > 0) &&
            timeSinceLastValid < LENS_PERSISTENCE_TIMEOUT_MS

        // Mesh/Debug 렌더링 조건 (기존 로직 유지)
        val shouldRenderMeshAndDebug = hasValidDetection ||
            (timeSinceLastValid < DETECTION_TIMEOUT_MS && (hasLeftEverDetected || hasRightEverDetected))

        // 렌즈도 Mesh도 렌더링할 것이 없으면 리턴
        if (!shouldRenderLens && !shouldRenderMeshAndDebug) return

        // 타임아웃 상태 로깅 (디버깅용)
        if (!hasValidDetection && timeSinceLastValid < LENS_PERSISTENCE_TIMEOUT_MS) {
            Log.d(TAG, "Lens persistence mode (${timeSinceLastValid}ms since last valid detection)")
        }

        // DEBUG: 좌표 변환 값 로깅 (ISS-001 디버깅)
        Log.d(TAG, "=== ISS-001 DEBUG ===")
        Log.d(TAG, "SDK imageSize: ${imageWidth}x${imageHeight}")
        Log.d(TAG, "View size: ${width}x${height}")
        Log.d(TAG, "imageAspect: ${imageWidth.toFloat()/imageHeight}, viewAspect: ${width.toFloat()/height}")

        // 좌표 변환 계산
        // FIT: min() — 이미지가 뷰 안에 맞춤 (레터박스)
        // COVER: max() — 이미지가 뷰를 완전히 채움 (넘치는 부분 crop, GL Cover 출력과 동일)
        val useFitMode = screenMappingMode == ScreenMappingMode.FIT
        val scaleFactor = computeScreenTransform(
            imageWidth, imageHeight, width, height, useFitMode
        )

        // 스케일된 이미지 크기
        val scaledImageWidth = imageWidth * scaleFactor
        val scaledImageHeight = imageHeight * scaleFactor

        // 이미지를 뷰 중앙에 배치하기 위한 오프셋
        val offsetX = (width - scaledImageWidth) / 2f
        val offsetY = (height - scaledImageHeight) / 2f

        // DEBUG: 변환 파라미터 로깅 (ISS-001 디버깅)
        Log.d(TAG, "scaleFactor: $scaleFactor, scaledImage: ${scaledImageWidth}x${scaledImageHeight}")
        Log.d(TAG, "offset: ($offsetX, $offsetY)")
        if (cachedFaceMeshValid && cachedFaceMesh != null) {
            // 첫 번째 랜드마크 좌표 확인 (코 끝 - 인덱스 1)
            val mesh = cachedFaceMesh!!
            val x0 = mesh[1 * 3]
            val y0 = mesh[1 * 3 + 1]
            Log.d(TAG, "Landmark[1] normalized: ($x0, $y0)")
            val screenX = x0 * imageWidth * scaleFactor + offsetX
            val screenY = y0 * imageHeight * scaleFactor + offsetY
            Log.d(TAG, "Landmark[1] screen: ($screenX, $screenY)")
        }
        Log.d(TAG, "=====================")

        // 렌즈 텍스처 렌더링 (별도의 긴 타임아웃 적용)
        // 깜빡임 방지: 검출 실패해도 3초간 마지막 위치에 렌즈 유지
        if (shouldRenderLens && showLens && lensTexture != null) {
            // 눈 영역 클리핑을 위한 Path 생성 (캐시된 FaceMesh 사용)
            val mesh = cachedFaceMesh
            val canClip = eyeClippingEnabled && mesh != null && cachedFaceMeshValid

            if (hasLeftEverDetected && lensConfig.applyLeft && filteredLeftRadius > 0) {
                if (canClip) {
                    // 왼쪽 눈 영역으로 클리핑하여 렌즈 렌더링
                    buildEyePath(leftEyePath, mesh!!, LEFT_EYE_CONTOUR_INDICES, scaleFactor, offsetX, offsetY)
                    canvas.save()
                    canvas.clipPath(leftEyePath)
                }
                drawLensTexture(
                    canvas,
                    filteredLeftX,
                    filteredLeftY,
                    filteredLeftRadius,
                    scaleFactor,
                    offsetX,
                    offsetY,
                    isLeft = true
                )
                if (canClip) {
                    canvas.restore()
                }
            }

            if (hasRightEverDetected && lensConfig.applyRight && filteredRightRadius > 0) {
                if (canClip) {
                    // 오른쪽 눈 영역으로 클리핑하여 렌즈 렌더링
                    buildEyePath(rightEyePath, mesh!!, RIGHT_EYE_CONTOUR_INDICES, scaleFactor, offsetX, offsetY)
                    canvas.save()
                    canvas.clipPath(rightEyePath)
                }
                drawLensTexture(
                    canvas,
                    filteredRightX,
                    filteredRightY,
                    filteredRightRadius,
                    scaleFactor,
                    offsetX,
                    offsetY,
                    isLeft = false
                )
                if (canClip) {
                    canvas.restore()
                }
            }
        }

        // 디버그 모드에서만 홍채 마커 표시 (필터링된 값 사용, 일반 타임아웃 적용)
        if (shouldRenderMeshAndDebug && debugMode) {
            if (cachedLeftDetected && lensConfig.applyLeft) {
                drawIrisMarker(
                    canvas,
                    filteredLeftX,
                    filteredLeftY,
                    filteredLeftRadius,
                    scaleFactor,
                    offsetX,
                    offsetY,
                    "L"
                )
            }

            if (cachedRightDetected && lensConfig.applyRight) {
                drawIrisMarker(
                    canvas,
                    filteredRightX,
                    filteredRightY,
                    filteredRightRadius,
                    scaleFactor,
                    offsetX,
                    offsetY,
                    "R"
                )
            }
        }

        // Face Mesh 표시 (충분한 신뢰도로 얼굴 감지 시에만, 캐시된 값 사용)
        // Mesh는 일반 타임아웃 적용 (렌즈보다 빠르게 사라짐)
        if (shouldRenderMeshAndDebug && showFaceMesh && cachedFaceMeshValid && cachedFaceMesh != null
            && cachedConfidence >= MIN_RENDER_CONFIDENCE) {
            drawFaceMeshCached(canvas, scaleFactor, offsetX, offsetY)
        }

        // 얼굴 검출 영역 및 디버그 정보는 원본 결과 필요 시 표시
        // (캐시에 faceRect 정보 없으므로 현재 결과 사용)
        val result = irisResult
        if (result != null) {
            if (showFaceRect) {
                drawFaceRect(canvas, result, scaleFactor, offsetX, offsetY)
            }
            if (debugMode) {
                drawDebugInfo(canvas, result, scaleFactor, offsetX, offsetY)
            }
        }
    }

    /**
     * 필터된 프레임 배경 그리기
     *
     * PreviewView의 FILL_CENTER와 동일한 스케일링을 적용하여
     * 뷰티 필터가 적용된 프레임을 배경으로 렌더링합니다.
     *
     * @param canvas 캔버스
     * @param frame 필터된 프레임 비트맵
     */
    private fun drawFilteredFrame(canvas: Canvas, frame: Bitmap) {
        // fillCenter 스케일링 계산 (PreviewView와 동일)
        val scaleFactor = max(width.toFloat() / frame.width, height.toFloat() / frame.height)
        val scaledWidth = frame.width * scaleFactor
        val scaledHeight = frame.height * scaleFactor
        val offsetX = (width - scaledWidth) / 2f
        val offsetY = (height - scaledHeight) / 2f

        canvas.save()

        // 미러링 적용 (전면 카메라)
        if (isMirror) {
            canvas.scale(-1f, 1f, width / 2f, height / 2f)
        }

        // 프레임 그리기
        tempRect.set(offsetX, offsetY, offsetX + scaledWidth, offsetY + scaledHeight)
        canvas.drawBitmap(frame, null, tempRect, framePaint)

        canvas.restore()
    }

    /**
     * 캐시된 스케일 비트맵 가져오기 또는 생성
     *
     * 스케일 변화가 임계값 미만이면 캐시된 비트맵을 재사용하여 꿀렁거림 방지.
     * 크기가 의미있게 변할 때만 새 비트맵을 생성.
     *
     * @param targetSize 목표 렌즈 크기 (반지름, 픽셀)
     * @param isLeft 왼쪽 눈 여부 (true: 왼쪽, false: 오른쪽)
     * @return 스케일된 비트맵 (없으면 null)
     */
    private fun getScaledLensBitmap(targetSize: Int, isLeft: Boolean): Bitmap? {
        val texture = lensTexture ?: return null

        val cachedBitmap = if (isLeft) cachedLeftLensBitmap else cachedRightLensBitmap
        val cachedSize = if (isLeft) cachedLeftLensSize else cachedRightLensSize

        // 캐시 유효성 검사: 크기 차이가 임계값 미만이면 캐시 재사용
        if (cachedBitmap != null && !cachedBitmap.isRecycled
            && abs(targetSize - cachedSize) < LENS_CACHE_THRESHOLD) {
            return cachedBitmap
        }

        // 새 스케일 비트맵 생성
        val diameter = (targetSize * 2).coerceAtLeast(1)
        val newBitmap = Bitmap.createScaledBitmap(texture, diameter, diameter, true)

        // 캐시 업데이트
        if (isLeft) {
            cachedLeftLensBitmap?.recycle()  // 이전 비트맵 해제
            cachedLeftLensBitmap = newBitmap
            cachedLeftLensSize = targetSize
        } else {
            cachedRightLensBitmap?.recycle()
            cachedRightLensBitmap = newBitmap
            cachedRightLensSize = targetSize
        }

        Log.d(TAG, "Created new scaled lens bitmap: size=$diameter, isLeft=$isLeft")
        return newBitmap
    }

    /**
     * 렌즈 텍스처 그리기 (캐시된 스케일 비트맵 사용)
     *
     * 매 프레임 새로 스케일링하지 않고 캐시된 비트맵을 재사용하여
     * 스케일 꿀렁거림 방지.
     *
     * @param isLeft 왼쪽 눈 여부 (캐시 구분용)
     */
    private fun drawLensTexture(
        canvas: Canvas,
        normalizedX: Float,
        normalizedY: Float,
        radius: Float,
        scaleFactor: Float,
        offsetX: Float,
        offsetY: Float,
        isLeft: Boolean
    ) {
        // 정규화 좌표 → 화면 좌표 변환 (MediaPipe 공식 예제 방식)
        var cx = normalizedX * imageWidth * scaleFactor + offsetX
        val cy = normalizedY * imageHeight * scaleFactor + offsetY

        // 렌즈 크기 계산 (홍채 반지름 * 배율 * 사용자 스케일)
        // radius는 픽셀 단위이므로 scaleFactor로 스케일
        val rawLensSize = radius * scaleFactor * LENS_SCALE_FACTOR * lensConfig.scale

        // 정수로 양자화하여 캐시 키로 사용
        val lensSize = rawLensSize.toInt().coerceAtLeast(1)

        // DEBUG: 실제 렌더링 크기 로깅
        Log.d(TAG, "Lens render size: radius=$radius, scaleFactor=$scaleFactor, lensSize=$lensSize (diameter=${lensSize*2})")

        // 미러링 (전면 카메라)
        if (isMirror) {
            cx = width - cx
        }

        // 캐시된 스케일 비트맵 가져오기 (없거나 크기 변경 시 새로 생성)
        val scaledBitmap = getScaledLensBitmap(lensSize, isLeft) ?: return

        // 투명도 설정
        lensPaint.alpha = (255 * lensConfig.opacity).toInt()

        // 캐시된 비트맵 그리기 (Matrix 스케일링 없음 - 이미 스케일됨)
        // 비트맵 중심을 cx, cy에 맞추기 위해 lensSize만큼 오프셋
        canvas.drawBitmap(
            scaledBitmap,
            cx - lensSize,
            cy - lensSize,
            lensPaint
        )
    }

    /**
     * 눈 영역 Path 생성 (클리핑용)
     *
     * Face Mesh 랜드마크에서 눈 윤곽을 추출하여 Path로 변환.
     * 이 Path를 clipPath()에 사용하여 눈꺼풀 위로 렌즈가 보이지 않도록 처리.
     *
     * @param path 결과를 저장할 Path 객체
     * @param mesh Face Mesh 랜드마크 배열
     * @param indices 눈 윤곽 랜드마크 인덱스 배열
     * @param scaleFactor 스케일 팩터
     * @param offsetX X 오프셋
     * @param offsetY Y 오프셋
     */
    private fun buildEyePath(
        path: Path,
        mesh: FloatArray,
        indices: IntArray,
        scaleFactor: Float,
        offsetX: Float,
        offsetY: Float
    ) {
        path.reset()

        if (indices.isEmpty()) return

        var firstX = 0f
        var firstY = 0f

        for ((i, idx) in indices.withIndex()) {
            // 랜드마크 좌표 추출 (정규화 좌표 0.0~1.0)
            val x = mesh[idx * 3].coerceIn(0f, 1f)
            val y = mesh[idx * 3 + 1].coerceIn(0f, 1f)

            // 화면 좌표로 변환
            var screenX = x * imageWidth * scaleFactor + offsetX
            val screenY = y * imageHeight * scaleFactor + offsetY

            // 미러링 (전면 카메라)
            if (isMirror) {
                screenX = width - screenX
            }

            if (i == 0) {
                path.moveTo(screenX, screenY)
                firstX = screenX
                firstY = screenY
            } else {
                path.lineTo(screenX, screenY)
            }
        }

        // Path 닫기
        path.close()
    }

    /**
     * 홍채 마커 그리기 (디버그용)
     */
    private fun drawIrisMarker(
        canvas: Canvas,
        normalizedX: Float,
        normalizedY: Float,
        radius: Float,
        scaleFactor: Float,
        offsetX: Float,
        offsetY: Float,
        label: String
    ) {
        // 정규화 좌표 → 화면 좌표 변환 (MediaPipe 공식 예제 방식)
        var cx = normalizedX * imageWidth * scaleFactor + offsetX
        val cy = normalizedY * imageHeight * scaleFactor + offsetY

        // ISS-004 Fix-A: Raw 홍채 반경과 Effective 렌즈 반경 분리 표시
        val rawR = radius * scaleFactor                       // 실제 홍채 크기
        val effectiveR = radius * scaleFactor * lensConfig.scale  // 렌즈 적용 크기

        // 미러링 (전면 카메라)
        if (isMirror) {
            cx = width - cx
        }

        // 1) Raw 홍채 반경 원 (파란색 점선 - 실제 홍채 경계)
        canvas.drawCircle(cx, cy, rawR, rawIrisPaint)

        // 2) Effective 렌즈 반경 원 (녹색 실선 - 렌즈 적용 영역)
        canvas.drawCircle(cx, cy, effectiveR, irisPaint)

        // 중심점 그리기
        canvas.drawCircle(cx, cy, CENTER_DOT_RADIUS, centerPaint)

        // 디버그 모드: 라벨 + 반경 정보 표시
        debugTextPaint.textSize = 24f
        val debugLabel = "%s rawR=%.0f effR=%.0f".format(label, rawR, effectiveR)
        canvas.drawText(debugLabel, cx + effectiveR + 10, cy, debugTextPaint)
    }

    /**
     * 얼굴 검출 영역 그리기 (Face Detection crop 영역)
     */
    private fun drawFaceRect(
        canvas: Canvas,
        result: IrisResult,
        scaleFactor: Float,
        offsetX: Float,
        offsetY: Float
    ) {
        // faceRectX/Y/Width/Height는 정규화 좌표 (0.0 ~ 1.0)
        if (result.faceRectWidth > 0 && result.faceRectHeight > 0) {
            // 정규화 좌표 → 화면 좌표 변환
            var left = result.faceRectX * imageWidth * scaleFactor + offsetX
            val top = result.faceRectY * imageHeight * scaleFactor + offsetY
            var right = (result.faceRectX + result.faceRectWidth) * imageWidth * scaleFactor + offsetX
            val bottom = (result.faceRectY + result.faceRectHeight) * imageHeight * scaleFactor + offsetY

            // 미러링 (전면 카메라)
            if (isMirror) {
                val tempLeft = width - right
                right = width - left
                left = tempLeft
            }

            tempRect.set(left, top, right, bottom)

            // 두꺼운 노란색 사각형으로 표시
            faceRectPaint.color = 0xFFFFFF00.toInt()  // Yellow
            faceRectPaint.strokeWidth = 4f
            canvas.drawRect(tempRect, faceRectPaint)

            // crop 영역 정보 텍스트 표시
            debugTextPaint.textSize = 24f
            debugTextPaint.color = 0xFFFFFF00.toInt()
            val infoText = "Face: (%.2f,%.2f) %.2fx%.2f".format(
                result.faceRectX, result.faceRectY,
                result.faceRectWidth, result.faceRectHeight
            )
            canvas.drawText(infoText, left, top - 10, debugTextPaint)

            // 색상 복원
            faceRectPaint.color = COLOR_FACE_RECT
            faceRectPaint.strokeWidth = 2f
            debugTextPaint.color = COLOR_DEBUG_TEXT
        }
    }

    /**
     * 디버그 정보 그리기
     */
    private fun drawDebugInfo(
        canvas: Canvas,
        result: IrisResult,
        scaleFactor: Float,
        offsetX: Float,
        offsetY: Float
    ) {
        // 얼굴 바운딩 박스 (faceRectX/Y는 정규화 좌표 0.0~1.0)
        // ISS-002 수정: imageWidth/Height 곱셈 추가 (정규화 좌표 → 화면 좌표)
        if (result.faceRectWidth > 0 && result.faceRectHeight > 0) {
            var left = result.faceRectX * imageWidth * scaleFactor + offsetX
            val top = result.faceRectY * imageHeight * scaleFactor + offsetY
            var right = (result.faceRectX + result.faceRectWidth) * imageWidth * scaleFactor + offsetX
            val bottom = (result.faceRectY + result.faceRectHeight) * imageHeight * scaleFactor + offsetY

            if (isMirror) {
                val tempLeft = width - right
                right = width - left
                left = tempLeft
            }

            tempRect.set(left, top, right, bottom)
            canvas.drawRect(tempRect, faceRectPaint)
        }

        // 모델 버전 판별 (랜드마크 수로 구분)
        val meshSize = result.faceMesh?.size?.div(3) ?: 0
        val modelVersion = when {
            meshSize >= 478 -> "V2 (478)"
            meshSize >= 468 -> "V1 (468)"
            else -> "N/A ($meshSize)"
        }

        // 디버그 텍스트 (ISS-004 Fix-A: rawR + effectiveR 동시 표시)
        val debugInfo = buildString {
            append("Model: $modelVersion\n")
            append("Confidence: %.2f\n".format(result.confidence))
            append("Left: (%.3f, %.3f) rawR=%.1f effR=%.1f\n".format(
                result.leftIrisX, result.leftIrisY,
                result.leftRadius, result.leftRadius * lensConfig.scale))
            append("Right: (%.3f, %.3f) rawR=%.1f effR=%.1f\n".format(
                result.rightIrisX, result.rightIrisY,
                result.rightRadius, result.rightRadius * lensConfig.scale))
            append("Face: P=%.1f Y=%.1f R=%.1f\n".format(
                result.facePitch, result.faceYaw, result.faceRoll))
            append("Lens: %.0f%% opacity, x%.1f scale".format(
                lensConfig.opacity * 100, lensConfig.scale))
        }

        // 배경
        debugTextPaint.textSize = 28f
        val textLines = debugInfo.split("\n")
        val lineHeight = debugTextPaint.fontSpacing
        val bgHeight = lineHeight * textLines.size + 20

        tempRect.set(10f, height - bgHeight - 10, 450f, height - 10f)
        canvas.drawRect(tempRect, debugBgPaint)

        // 텍스트
        var y = height - bgHeight + lineHeight
        for (line in textLines) {
            canvas.drawText(line, 20f, y, debugTextPaint)
            y += lineHeight
        }
    }

    /**
     * Face Mesh 그리기 (캐시된 FaceMesh 사용)
     */
    private fun drawFaceMeshCached(
        canvas: Canvas,
        scaleFactor: Float,
        offsetX: Float,
        offsetY: Float
    ) {
        val mesh = cachedFaceMesh ?: return
        val landmarkCount = IrisResult.FACE_MESH_LANDMARK_COUNT

        // 모든 랜드마크 점 그리기
        for (i in 0 until landmarkCount) {
            val x = mesh[i * 3].coerceIn(0f, 1f)
            val y = mesh[i * 3 + 1].coerceIn(0f, 1f)

            var screenX = x * imageWidth * scaleFactor + offsetX
            val screenY = y * imageHeight * scaleFactor + offsetY

            if (isMirror) {
                screenX = width - screenX
            }

            canvas.drawCircle(screenX, screenY, MESH_POINT_RADIUS, meshPointPaint)
        }

        // 주요 연결선 그리기
        drawFaceContour(canvas, mesh, scaleFactor, offsetX, offsetY)
        drawEyeContours(canvas, mesh, scaleFactor, offsetX, offsetY)
        drawLipsContour(canvas, mesh, scaleFactor, offsetX, offsetY)
        drawIrisLandmarks(canvas, mesh, scaleFactor, offsetX, offsetY)
    }

    /**
     * Face Mesh 그리기 (IrisResult 사용 - 레거시)
     */
    private fun drawFaceMesh(
        canvas: Canvas,
        result: IrisResult,
        scaleFactor: Float,
        offsetX: Float,
        offsetY: Float
    ) {
        val mesh = result.faceMesh ?: return
        val landmarkCount = IrisResult.FACE_MESH_LANDMARK_COUNT

        // 모든 랜드마크 점 그리기
        for (i in 0 until landmarkCount) {
            // 정규화된 좌표 (0.0 ~ 1.0)에 클램핑 적용 (방어적 처리)
            val x = mesh[i * 3].coerceIn(0f, 1f)
            val y = mesh[i * 3 + 1].coerceIn(0f, 1f)

            // 화면 좌표로 변환 (MediaPipe 공식 예제 방식)
            var screenX = x * imageWidth * scaleFactor + offsetX
            val screenY = y * imageHeight * scaleFactor + offsetY

            // 미러링 (전면 카메라)
            if (isMirror) {
                screenX = width - screenX
            }

            canvas.drawCircle(screenX, screenY, MESH_POINT_RADIUS, meshPointPaint)
        }

        // 주요 연결선 그리기 (얼굴 윤곽, 눈, 입술, 눈썹)
        drawFaceContour(canvas, mesh, scaleFactor, offsetX, offsetY)
        drawEyeContours(canvas, mesh, scaleFactor, offsetX, offsetY)
        drawLipsContour(canvas, mesh, scaleFactor, offsetX, offsetY)

        // 홍채 랜드마크 그리기 (478개 랜드마크 모델인 경우)
        drawIrisLandmarks(canvas, mesh, scaleFactor, offsetX, offsetY)
    }

    /**
     * 얼굴 윤곽선 그리기
     */
    private fun drawFaceContour(
        canvas: Canvas,
        mesh: FloatArray,
        scaleFactor: Float,
        offsetX: Float,
        offsetY: Float
    ) {
        // 얼굴 윤곽 인덱스 (MediaPipe Face Mesh 기준)
        val faceOvalIndices = intArrayOf(
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10
        )
        drawConnectedLandmarks(canvas, mesh, faceOvalIndices, scaleFactor, offsetX, offsetY)
    }

    /**
     * 눈 윤곽선 그리기
     */
    private fun drawEyeContours(
        canvas: Canvas,
        mesh: FloatArray,
        scaleFactor: Float,
        offsetX: Float,
        offsetY: Float
    ) {
        // 왼쪽 눈 (화면 기준 오른쪽)
        val leftEyeIndices = intArrayOf(
            362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387,
            386, 385, 384, 398, 362
        )
        drawConnectedLandmarks(canvas, mesh, leftEyeIndices, scaleFactor, offsetX, offsetY)

        // 오른쪽 눈 (화면 기준 왼쪽)
        val rightEyeIndices = intArrayOf(
            33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158,
            159, 160, 161, 246, 33
        )
        drawConnectedLandmarks(canvas, mesh, rightEyeIndices, scaleFactor, offsetX, offsetY)
    }

    /**
     * 입술 윤곽선 그리기
     */
    private fun drawLipsContour(
        canvas: Canvas,
        mesh: FloatArray,
        scaleFactor: Float,
        offsetX: Float,
        offsetY: Float
    ) {
        // 외곽 입술
        val outerLipsIndices = intArrayOf(
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409,
            270, 269, 267, 0, 37, 39, 40, 185, 61
        )
        drawConnectedLandmarks(canvas, mesh, outerLipsIndices, scaleFactor, offsetX, offsetY)
    }

    /**
     * 홍채 랜드마크 그리기 (4포인트 + 중심)
     *
     * MediaPipe Face Mesh with Iris (478개 랜드마크)에서:
     * - 왼쪽 홍채: 중심 468, 경계 469-472
     * - 오른쪽 홍채: 중심 473, 경계 474-477
     */
    private fun drawIrisLandmarks(
        canvas: Canvas,
        mesh: FloatArray,
        scaleFactor: Float,
        offsetX: Float,
        offsetY: Float
    ) {
        // 478개 랜드마크가 있는지 확인 (홍채 포함 모델)
        val meshSize = mesh.size / 3
        if (meshSize < 478) {
            Log.d(TAG, "Mesh has only $meshSize landmarks, iris landmarks (478) not available")
            return
        }

        // 왼쪽 홍채 중심 그리기
        drawSingleIrisPoint(canvas, mesh, LEFT_IRIS_CENTER, scaleFactor, offsetX, offsetY, isCenter = true)

        // 왼쪽 홍채 4포인트 그리기
        for (idx in LEFT_IRIS_POINTS) {
            drawSingleIrisPoint(canvas, mesh, idx, scaleFactor, offsetX, offsetY, isCenter = false)
        }

        // 오른쪽 홍채 중심 그리기
        drawSingleIrisPoint(canvas, mesh, RIGHT_IRIS_CENTER, scaleFactor, offsetX, offsetY, isCenter = true)

        // 오른쪽 홍채 4포인트 그리기
        for (idx in RIGHT_IRIS_POINTS) {
            drawSingleIrisPoint(canvas, mesh, idx, scaleFactor, offsetX, offsetY, isCenter = false)
        }
    }

    /**
     * 단일 홍채 포인트 그리기
     *
     * V1 모델(468개)은 홍채 랜드마크가 없어서 -1로 채워짐
     * 이 경우 그리지 않음
     */
    private fun drawSingleIrisPoint(
        canvas: Canvas,
        mesh: FloatArray,
        index: Int,
        scaleFactor: Float,
        offsetX: Float,
        offsetY: Float,
        isCenter: Boolean
    ) {
        val rawX = mesh[index * 3]
        val rawY = mesh[index * 3 + 1]

        // V1 모델 사용 시 홍채 랜드마크는 -1로 채워짐 - 스킵
        if (rawX < 0f || rawY < 0f || rawX > 1f || rawY > 1f) {
            return
        }

        var screenX = rawX * imageWidth * scaleFactor + offsetX
        val screenY = rawY * imageHeight * scaleFactor + offsetY

        if (isMirror) {
            screenX = width - screenX
        }

        // 중심은 더 크게, 경계는 작게
        val radius = if (isCenter) IRIS_POINT_RADIUS * 1.5f else IRIS_POINT_RADIUS
        canvas.drawCircle(screenX, screenY, radius, irisLandmarkPaint)
    }

    /**
     * 연결된 랜드마크 그리기
     */
    private fun drawConnectedLandmarks(
        canvas: Canvas,
        mesh: FloatArray,
        indices: IntArray,
        scaleFactor: Float,
        offsetX: Float,
        offsetY: Float
    ) {
        if (indices.size < 2) return

        for (i in 0 until indices.size - 1) {
            val idx1 = indices[i]
            val idx2 = indices[i + 1]

            // 정규화된 좌표에 클램핑 적용 (방어적 처리)
            val x1 = mesh[idx1 * 3].coerceIn(0f, 1f)
            val y1 = mesh[idx1 * 3 + 1].coerceIn(0f, 1f)
            val x2 = mesh[idx2 * 3].coerceIn(0f, 1f)
            val y2 = mesh[idx2 * 3 + 1].coerceIn(0f, 1f)

            // 화면 좌표로 변환 (MediaPipe 공식 예제 방식)
            var screenX1 = x1 * imageWidth * scaleFactor + offsetX
            val screenY1 = y1 * imageHeight * scaleFactor + offsetY
            var screenX2 = x2 * imageWidth * scaleFactor + offsetX
            val screenY2 = y2 * imageHeight * scaleFactor + offsetY

            if (isMirror) {
                screenX1 = width - screenX1
                screenX2 = width - screenX2
            }

            canvas.drawLine(screenX1, screenY1, screenX2, screenY2, meshLinePaint)
        }
    }

    /**
     * 뷰가 윈도우에서 분리될 때 캐시된 비트맵 해제
     */
    override fun onDetachedFromWindow() {
        super.onDetachedFromWindow()

        // 캐시된 스케일 비트맵 해제
        cachedLeftLensBitmap?.recycle()
        cachedLeftLensBitmap = null
        cachedRightLensBitmap?.recycle()
        cachedRightLensBitmap = null

        // 필터된 프레임 해제
        filteredFrame?.recycle()
        filteredFrame = null

        Log.d(TAG, "Cached lens bitmaps released")
    }
}

