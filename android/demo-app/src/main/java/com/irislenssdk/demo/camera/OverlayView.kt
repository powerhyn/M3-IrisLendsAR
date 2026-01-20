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
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import com.irislenssdk.IrisResult
import com.irislenssdk.LensConfig
import kotlin.math.max

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
        private const val SMOOTHING_FACTOR = 0.25f

        // 렌즈 크기 양자화 단위 (픽셀) - 자글거림 방지
        private const val LENS_SIZE_QUANTIZATION_STEP = 2f

        // Radius 변화 최소 임계값 (픽셀) - 미세한 변화 무시
        // 이 값 이하의 radius 변화는 노이즈로 간주하여 무시
        private const val RADIUS_CHANGE_THRESHOLD = 0.5f

        // 최소 렌더링 신뢰도 임계값 (False Positive 방지)
        // 이 값 미만의 신뢰도를 가진 검출 결과는 렌더링하지 않음
        // 허공/천장 감지 문제 해결을 위해 추가
        private const val MIN_RENDER_CONFIDENCE = 0.5f
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

    // === 스무딩용 변수 (부드러운 추적) ===
    private var smoothedLeftX: Float = 0f
    private var smoothedLeftY: Float = 0f
    private var smoothedLeftRadius: Float = 0f
    private var smoothedRightX: Float = 0f
    private var smoothedRightY: Float = 0f
    private var smoothedRightRadius: Float = 0f
    private var isFirstFrame: Boolean = true

    // Face Mesh 표시 모드
    var showFaceMesh: Boolean = false

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
     * 홍채 검출 결과 설정 (스무딩 적용)
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

        // 스무딩 적용 (부드러운 추적)
        result?.let {
            if (it.detected) {
                if (isFirstFrame) {
                    // 첫 프레임: 즉시 반영
                    smoothedLeftX = it.leftIrisX
                    smoothedLeftY = it.leftIrisY
                    smoothedLeftRadius = it.leftRadius
                    smoothedRightX = it.rightIrisX
                    smoothedRightY = it.rightIrisY
                    smoothedRightRadius = it.rightRadius
                    isFirstFrame = false
                } else {
                    // 이후 프레임: EMA(지수이동평균) 스무딩
                    if (it.leftDetected) {
                        smoothedLeftX = lerp(smoothedLeftX, it.leftIrisX, SMOOTHING_FACTOR)
                        smoothedLeftY = lerp(smoothedLeftY, it.leftIrisY, SMOOTHING_FACTOR)
                        // Radius는 임계값 이상 변화시에만 업데이트 (자글거림 방지)
                        if (kotlin.math.abs(it.leftRadius - smoothedLeftRadius) > RADIUS_CHANGE_THRESHOLD) {
                            smoothedLeftRadius = lerp(smoothedLeftRadius, it.leftRadius, SMOOTHING_FACTOR)
                        }
                    }
                    if (it.rightDetected) {
                        smoothedRightX = lerp(smoothedRightX, it.rightIrisX, SMOOTHING_FACTOR)
                        smoothedRightY = lerp(smoothedRightY, it.rightIrisY, SMOOTHING_FACTOR)
                        // Radius는 임계값 이상 변화시에만 업데이트 (자글거림 방지)
                        if (kotlin.math.abs(it.rightRadius - smoothedRightRadius) > RADIUS_CHANGE_THRESHOLD) {
                            smoothedRightRadius = lerp(smoothedRightRadius, it.rightRadius, SMOOTHING_FACTOR)
                        }
                    }
                }
            }
        } ?: run {
            // 검출 실패 시 스무딩 초기화
            isFirstFrame = true
        }

        invalidate()
    }

    /**
     * 선형 보간 (Linear Interpolation)
     */
    private fun lerp(start: Float, end: Float, factor: Float): Float {
        return start + (end - start) * factor
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
     */
    fun setLensTexture(texture: Bitmap?) {
        this.lensTexture = texture
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val result = irisResult ?: return
        if (!result.detected) return

        // 신뢰도 검증 (False Positive 방지)
        // 낮은 신뢰도의 검출 결과는 허공/천장 오인식일 가능성이 높음
        if (result.confidence < MIN_RENDER_CONFIDENCE) return

        // 좌표 변환 계산 (MediaPipe 공식 예제 방식)
        // PreviewView가 FILL_START 모드이므로 max 사용하여 이미지가 뷰를 채우도록 함
        val scaleFactor = max(width.toFloat() / imageWidth, height.toFloat() / imageHeight)

        // 스케일된 이미지 크기
        val scaledImageWidth = imageWidth * scaleFactor
        val scaledImageHeight = imageHeight * scaleFactor

        // 이미지를 뷰 중앙에 배치하기 위한 오프셋
        val offsetX = (width - scaledImageWidth) / 2f
        val offsetY = (height - scaledImageHeight) / 2f

        // 렌즈 텍스처 렌더링 (스무딩된 값 사용)
        if (showLens && lensTexture != null) {
            if (result.leftDetected && lensConfig.applyLeft) {
                drawLensTexture(
                    canvas,
                    smoothedLeftX,
                    smoothedLeftY,
                    smoothedLeftRadius,
                    scaleFactor,
                    offsetX,
                    offsetY
                )
            }

            if (result.rightDetected && lensConfig.applyRight) {
                drawLensTexture(
                    canvas,
                    smoothedRightX,
                    smoothedRightY,
                    smoothedRightRadius,
                    scaleFactor,
                    offsetX,
                    offsetY
                )
            }
        }

        // 디버그 모드에서만 홍채 마커 표시 (스무딩된 값 사용)
        if (debugMode) {
            if (result.leftDetected && lensConfig.applyLeft) {
                drawIrisMarker(
                    canvas,
                    smoothedLeftX,
                    smoothedLeftY,
                    smoothedLeftRadius,
                    scaleFactor,
                    offsetX,
                    offsetY,
                    "L"
                )
            }

            if (result.rightDetected && lensConfig.applyRight) {
                drawIrisMarker(
                    canvas,
                    smoothedRightX,
                    smoothedRightY,
                    smoothedRightRadius,
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

        // 디버그 모드: 얼굴 영역 및 정보 표시
        if (debugMode) {
            drawDebugInfo(canvas, result, scaleFactor, offsetX, offsetY)
        }
    }

    /**
     * 렌즈 텍스처 그리기
     */
    private fun drawLensTexture(
        canvas: Canvas,
        normalizedX: Float,
        normalizedY: Float,
        radius: Float,
        scaleFactor: Float,
        offsetX: Float,
        offsetY: Float
    ) {
        val texture = lensTexture ?: return

        // 정규화 좌표 → 화면 좌표 변환 (MediaPipe 공식 예제 방식)
        var cx = normalizedX * imageWidth * scaleFactor + offsetX
        val cy = normalizedY * imageHeight * scaleFactor + offsetY

        // 렌즈 크기 계산 (홍채 반지름 * 배율 * 사용자 스케일)
        // radius는 픽셀 단위이므로 scaleFactor로 스케일
        val rawLensSize = radius * scaleFactor * LENS_SCALE_FACTOR * lensConfig.scale

        // 양자화 적용 (미세한 크기 변화로 인한 자글거림 방지)
        // 2픽셀 단위로 반올림하여 매 프레임 동일한 스케일 유지
        val lensSize = ((rawLensSize / LENS_SIZE_QUANTIZATION_STEP + 0.5f).toInt() * LENS_SIZE_QUANTIZATION_STEP)

        // 미러링 (전면 카메라)
        if (isMirror) {
            cx = width - cx
        }

        // 렌즈 위치 (중심점 기준)
        lensDestRect.set(
            cx - lensSize,
            cy - lensSize,
            cx + lensSize,
            cy + lensSize
        )

        // 투명도 설정
        lensPaint.alpha = (255 * lensConfig.opacity).toInt()

        // 매트릭스 설정 (텍스처 → 화면)
        lensMatrix.reset()
        lensMatrix.setRectToRect(
            RectF(0f, 0f, texture.width.toFloat(), texture.height.toFloat()),
            lensDestRect,
            Matrix.ScaleToFit.FILL
        )

        // 렌즈 텍스처 그리기
        canvas.drawBitmap(texture, lensMatrix, lensPaint)
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
     * 디버그 정보 그리기
     */
    private fun drawDebugInfo(
        canvas: Canvas,
        result: IrisResult,
        scaleFactor: Float,
        offsetX: Float,
        offsetY: Float
    ) {
        // 얼굴 바운딩 박스 (faceRectX/Y는 픽셀 좌표)
        if (result.faceRectWidth > 0 && result.faceRectHeight > 0) {
            var left = result.faceRectX * scaleFactor + offsetX
            val top = result.faceRectY * scaleFactor + offsetY
            var right = (result.faceRectX + result.faceRectWidth) * scaleFactor + offsetX
            val bottom = (result.faceRectY + result.faceRectHeight) * scaleFactor + offsetY

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
            val x = mesh[i * 3]      // 정규화된 x (0.0 ~ 1.0)
            val y = mesh[i * 3 + 1]  // 정규화된 y (0.0 ~ 1.0)

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

            val x1 = mesh[idx1 * 3]
            val y1 = mesh[idx1 * 3 + 1]
            val x2 = mesh[idx2 * 3]
            val y2 = mesh[idx2 * 3 + 1]

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
}
