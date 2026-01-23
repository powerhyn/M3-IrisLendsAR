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

        // One Euro Filter 파라미터
        private const val ONE_EURO_MIN_CUTOFF = 1.0f   // 최소 컷오프 주파수 (낮을수록 부드러움)
        private const val ONE_EURO_BETA = 0.007f       // 속도 계수 (높을수록 빠른 움직임에 민감)
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

    // 임시 RectF (재사용)
    private val tempRect = RectF()
    private val lensDestRect = RectF()

    // 변환 매트릭스 (재사용)
    private val lensMatrix = Matrix()

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
            if (it.detected) {
                // 왼쪽 눈 필터링
                if (it.leftDetected) {
                    hasLeftEverDetected = true
                    filteredLeftX = leftXFilter.filter(it.leftIrisX, currentTime)
                    filteredLeftY = leftYFilter.filter(it.leftIrisY, currentTime)
                    filteredLeftRadius = leftRadiusFilter.filter(it.leftRadius, currentTime)
                }
                // 검출 실패 시 마지막 필터링 값 유지 (깜빡임 방지)

                // 오른쪽 눈 필터링
                if (it.rightDetected) {
                    hasRightEverDetected = true
                    filteredRightX = rightXFilter.filter(it.rightIrisX, currentTime)
                    filteredRightY = rightYFilter.filter(it.rightIrisY, currentTime)
                    filteredRightRadius = rightRadiusFilter.filter(it.rightRadius, currentTime)
                }
                // 검출 실패 시 마지막 필터링 값 유지 (깜빡임 방지)

                lastTimestamp = currentTime
            }
        }
        // result가 null이어도 필터 상태 유지 (마지막 위치에 렌즈 유지)

        invalidate()
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

        val result = irisResult ?: return
        if (!result.detected) return

        // 신뢰도 검증 (False Positive 방지)
        // 낮은 신뢰도의 검출 결과는 허공/천장 오인식일 가능성이 높음
        if (result.confidence < MIN_RENDER_CONFIDENCE) return

        // DEBUG: 좌표 변환 값 로깅 (ISS-001 디버깅)
        Log.d(TAG, "=== ISS-001 DEBUG ===")
        Log.d(TAG, "SDK imageSize: ${imageWidth}x${imageHeight}")
        Log.d(TAG, "SDK frameSize: ${result.frameWidth}x${result.frameHeight}")
        Log.d(TAG, "View size: ${width}x${height}")
        Log.d(TAG, "imageAspect: ${imageWidth.toFloat()/imageHeight}, viewAspect: ${width.toFloat()/height}")

        // 좌표 변환 계산 (MediaPipe 공식 예제 방식)
        // PreviewView가 FILL_CENTER 모드이므로:
        // 1. max() 사용: 이미지가 뷰를 완전히 채움 (넘치는 부분 잘림)
        // 2. offset 계산: 중앙 정렬 (잘리는 부분이 양쪽에 균등하게 분배)
        val scaleFactor = max(width.toFloat() / imageWidth, height.toFloat() / imageHeight)

        // 스케일된 이미지 크기
        val scaledImageWidth = imageWidth * scaleFactor
        val scaledImageHeight = imageHeight * scaleFactor

        // 이미지를 뷰 중앙에 배치하기 위한 오프셋
        val offsetX = (width - scaledImageWidth) / 2f
        val offsetY = (height - scaledImageHeight) / 2f

        // DEBUG: 변환 파라미터 로깅 (ISS-001 디버깅)
        Log.d(TAG, "scaleFactor: $scaleFactor, scaledImage: ${scaledImageWidth}x${scaledImageHeight}")
        Log.d(TAG, "offset: ($offsetX, $offsetY)")
        if (result.faceMeshValid && result.faceMesh != null) {
            // 첫 번째 랜드마크 좌표 확인 (코 끝 - 인덱스 1)
            val mesh = result.faceMesh!!
            val x0 = mesh[1 * 3]
            val y0 = mesh[1 * 3 + 1]
            Log.d(TAG, "Landmark[1] normalized: ($x0, $y0)")
            val screenX = x0 * imageWidth * scaleFactor + offsetX
            val screenY = y0 * imageHeight * scaleFactor + offsetY
            Log.d(TAG, "Landmark[1] screen: ($screenX, $screenY)")
        }
        Log.d(TAG, "=====================")

        // 렌즈 텍스처 렌더링 (스무딩된 값 사용)
        // 렌즈 텍스처 렌더링 (One Euro Filter 적용된 값 사용)
        // 깜빡임 방지: 한 번 검출된 눈은 검출 실패 시에도 마지막 위치에 렌즈 유지
        if (showLens && lensTexture != null) {
            // 눈 영역 클리핑을 위한 Path 생성
            val mesh = result.faceMesh
            val canClipLeft = eyeClippingEnabled && mesh != null && result.faceMeshValid
            val canClipRight = canClipLeft

            if (hasLeftEverDetected && lensConfig.applyLeft && filteredLeftRadius > 0) {
                if (canClipLeft) {
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
                if (canClipLeft) {
                    canvas.restore()
                }
            }

            if (hasRightEverDetected && lensConfig.applyRight && filteredRightRadius > 0) {
                if (canClipRight) {
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
                if (canClipRight) {
                    canvas.restore()
                }
            }
        }

        // 디버그 모드에서만 홍채 마커 표시 (필터링된 값 사용)
        if (debugMode) {
            if (result.leftDetected && lensConfig.applyLeft) {
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

            if (result.rightDetected && lensConfig.applyRight) {
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

        // Face Mesh 표시 (충분한 신뢰도로 얼굴 감지 시에만)
        // 이중 검증: onDraw() 시작의 신뢰도 검증 + 여기서의 추가 검증 (방어적 프로그래밍)
        if (showFaceMesh && result.faceMeshValid && result.faceMesh != null
            && result.confidence >= MIN_RENDER_CONFIDENCE) {
            drawFaceMesh(canvas, result, scaleFactor, offsetX, offsetY)
        }

        // 얼굴 검출 영역 표시 (Face Detection 결과)
        if (showFaceRect) {
            drawFaceRect(canvas, result, scaleFactor, offsetX, offsetY)
        }

        // 디버그 모드: 얼굴 영역 및 정보 표시
        if (debugMode) {
            drawDebugInfo(canvas, result, scaleFactor, offsetX, offsetY)
        }
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
        val r = radius * scaleFactor * lensConfig.scale

        // 미러링 (전면 카메라)
        if (isMirror) {
            cx = width - cx
        }

        // 홍채 원 그리기
        canvas.drawCircle(cx, cy, r, irisPaint)

        // 중심점 그리기
        canvas.drawCircle(cx, cy, CENTER_DOT_RADIUS, centerPaint)

        // 디버그 모드: 라벨 표시
        debugTextPaint.textSize = 24f
        canvas.drawText(label, cx + r + 10, cy, debugTextPaint)
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

        // 디버그 텍스트
        val debugInfo = buildString {
            append("Confidence: %.2f\n".format(result.confidence))
            append("Left: (%.3f, %.3f) r=%.1f\n".format(
                result.leftIrisX, result.leftIrisY, result.leftRadius))
            append("Right: (%.3f, %.3f) r=%.1f\n".format(
                result.rightIrisX, result.rightIrisY, result.rightRadius))
            append("Face: P=%.1f Y=%.1f R=%.1f\n".format(
                result.facePitch, result.faceYaw, result.faceRoll))
            append("Lens: %.0f%% opacity, %.0f%% scale".format(
                lensConfig.opacity * 100, lensConfig.scale * 100))
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
     * Face Mesh 그리기
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

        Log.d(TAG, "Cached lens bitmaps released")
    }
}

/**
 * One Euro Filter - 적응형 노이즈 필터링
 *
 * 느린 움직임에는 강한 스무딩, 빠른 움직임에는 빠른 반응을 제공하는 필터.
 * 깜빡임과 흔들거림을 효과적으로 제거하면서 반응성 유지.
 *
 * @param minCutoff 최소 컷오프 주파수 (낮을수록 부드러움)
 * @param beta 속도 계수 (높을수록 빠른 움직임에 민감)
 * @param dCutoff 미분 컷오프 주파수
 *
 * 참조: https://cristal.univ-lille.fr/~casiez/1euro/
 */
class OneEuroFilter(
    private val minCutoff: Float = 1.0f,
    private val beta: Float = 0.007f,
    private val dCutoff: Float = 1.0f
) {
    private var x: Float = 0f
    private var dx: Float = 0f
    private var lastTime: Long = 0L
    private var initialized: Boolean = false

    /**
     * 새로운 값을 필터링
     * @param value 입력 값
     * @param timestamp 타임스탬프 (밀리초)
     * @return 필터링된 값
     */
    fun filter(value: Float, timestamp: Long): Float {
        if (!initialized) {
            x = value
            dx = 0f
            lastTime = timestamp
            initialized = true
            return value
        }

        // 시간 간격 계산 (초 단위)
        val dt = ((timestamp - lastTime).coerceAtLeast(1L)) / 1000f
        lastTime = timestamp

        // 속도 추정 (미분 필터링)
        val edx = (value - x) / dt
        dx = lowPassFilter(edx, dx, alpha(dCutoff, dt))

        // 적응형 컷오프 주파수 계산
        val cutoff = minCutoff + beta * abs(dx)

        // 위치 필터링
        x = lowPassFilter(value, x, alpha(cutoff, dt))

        return x
    }

    /**
     * 필터 초기화 (검출 실패 후 재검출 시)
     */
    fun reset() {
        initialized = false
    }

    /**
     * 현재 필터링된 값 반환
     */
    fun getValue(): Float = x

    /**
     * 저역 통과 필터
     */
    private fun lowPassFilter(x: Float, prevX: Float, alpha: Float): Float {
        return alpha * x + (1f - alpha) * prevX
    }

    /**
     * 알파 값 계산 (컷오프 주파수 기반)
     */
    private fun alpha(cutoff: Float, dt: Float): Float {
        val tau = 1f / (2f * Math.PI.toFloat() * cutoff)
        return 1f / (1f + tau / dt)
    }
}
