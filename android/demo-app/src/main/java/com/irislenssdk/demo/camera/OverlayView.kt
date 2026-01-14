/**
 * IrisLensSDK Android - OverlayView
 *
 * 홍채 검출 결과를 시각화하는 커스텀 뷰
 * - 홍채 위치 마커
 * - 렌즈 텍스처 오버레이 (향후 구현)
 * - 디버그 정보 표시
 *
 * @version 1.0.0
 */
package com.irislenssdk.demo.camera

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import com.irislenssdk.IrisResult
import com.irislenssdk.LensConfig

/**
 * 홍채 오버레이 뷰
 *
 * 카메라 프리뷰 위에 오버레이되어 홍채 검출 결과를 표시
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
    }

    // 검출 결과
    private var irisResult: IrisResult? = null

    // 렌즈 설정
    private var lensConfig: LensConfig = LensConfig()

    // 이미지 크기 (분석 이미지)
    private var imageWidth: Int = 640
    private var imageHeight: Int = 480

    // 미러링 (전면 카메라)
    private var isMirror: Boolean = true

    // 디버그 모드
    var debugMode: Boolean = false

    // Face Mesh 표시 모드
    var showFaceMesh: Boolean = false

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

    // 임시 RectF (재사용)
    private val tempRect = RectF()

    /**
     * 홍채 검출 결과 설정
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
        invalidate()
    }

    /**
     * 렌즈 설정 업데이트
     */
    fun setLensConfig(config: LensConfig) {
        this.lensConfig = config
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val result = irisResult ?: return
        if (!result.detected) return

        // 좌표 변환 스케일
        val scaleX = width.toFloat() / imageWidth
        val scaleY = height.toFloat() / imageHeight

        // 왼쪽 홍채 그리기
        if (result.leftDetected && lensConfig.applyLeft) {
            drawIris(
                canvas,
                result.leftIrisX,
                result.leftIrisY,
                result.leftRadius,
                scaleX,
                scaleY,
                "L"
            )
        }

        // 오른쪽 홍채 그리기
        if (result.rightDetected && lensConfig.applyRight) {
            drawIris(
                canvas,
                result.rightIrisX,
                result.rightIrisY,
                result.rightRadius,
                scaleX,
                scaleY,
                "R"
            )
        }

        // Face Mesh 표시
        if (showFaceMesh && result.faceMeshValid && result.faceMesh != null) {
            drawFaceMesh(canvas, result, scaleX, scaleY)
        }

        // 디버그 모드: 얼굴 영역 및 정보 표시
        if (debugMode) {
            drawDebugInfo(canvas, result, scaleX, scaleY)
        }
    }

    /**
     * 홍채 원 그리기
     */
    private fun drawIris(
        canvas: Canvas,
        normalizedX: Float,
        normalizedY: Float,
        radius: Float,
        scaleX: Float,
        scaleY: Float,
        label: String
    ) {
        // 정규화 좌표 → 화면 좌표 변환
        var cx = normalizedX * imageWidth * scaleX
        val cy = normalizedY * imageHeight * scaleY
        val r = radius * scaleX * lensConfig.scale

        // 미러링 (전면 카메라)
        if (isMirror) {
            cx = width - cx
        }

        // 홍채 원 그리기
        canvas.drawCircle(cx, cy, r, irisPaint)

        // 중심점 그리기
        canvas.drawCircle(cx, cy, CENTER_DOT_RADIUS, centerPaint)

        // 디버그 모드: 라벨 표시
        if (debugMode) {
            debugTextPaint.textSize = 24f
            canvas.drawText(label, cx + r + 10, cy, debugTextPaint)
        }
    }

    /**
     * 디버그 정보 그리기
     */
    private fun drawDebugInfo(
        canvas: Canvas,
        result: IrisResult,
        scaleX: Float,
        scaleY: Float
    ) {
        // 얼굴 바운딩 박스
        if (result.faceRectWidth > 0 && result.faceRectHeight > 0) {
            var left = result.faceRectX * scaleX
            val top = result.faceRectY * scaleY
            var right = (result.faceRectX + result.faceRectWidth) * scaleX
            val bottom = (result.faceRectY + result.faceRectHeight) * scaleY

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
            append("Face: P=%.1f Y=%.1f R=%.1f".format(
                result.facePitch, result.faceYaw, result.faceRoll))
        }

        // 배경
        debugTextPaint.textSize = 28f
        val textLines = debugInfo.split("\n")
        val lineHeight = debugTextPaint.fontSpacing
        val bgHeight = lineHeight * textLines.size + 20

        tempRect.set(10f, height - bgHeight - 10, 400f, height - 10f)
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
        scaleX: Float,
        scaleY: Float
    ) {
        val mesh = result.faceMesh ?: return
        val landmarkCount = IrisResult.FACE_MESH_LANDMARK_COUNT

        // 모든 랜드마크 점 그리기
        for (i in 0 until landmarkCount) {
            val x = mesh[i * 3]      // 정규화된 x (0.0 ~ 1.0)
            val y = mesh[i * 3 + 1]  // 정규화된 y (0.0 ~ 1.0)

            // 화면 좌표로 변환
            var screenX = x * imageWidth * scaleX
            val screenY = y * imageHeight * scaleY

            // 미러링 (전면 카메라)
            if (isMirror) {
                screenX = width - screenX
            }

            canvas.drawCircle(screenX, screenY, MESH_POINT_RADIUS, meshPointPaint)
        }

        // 주요 연결선 그리기 (얼굴 윤곽, 눈, 입술, 눈썹)
        drawFaceContour(canvas, mesh, scaleX, scaleY)
        drawEyeContours(canvas, mesh, scaleX, scaleY)
        drawLipsContour(canvas, mesh, scaleX, scaleY)
    }

    /**
     * 얼굴 윤곽선 그리기
     */
    private fun drawFaceContour(
        canvas: Canvas,
        mesh: FloatArray,
        scaleX: Float,
        scaleY: Float
    ) {
        // 얼굴 윤곽 인덱스 (MediaPipe Face Mesh 기준)
        val faceOvalIndices = intArrayOf(
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10
        )
        drawConnectedLandmarks(canvas, mesh, faceOvalIndices, scaleX, scaleY)
    }

    /**
     * 눈 윤곽선 그리기
     */
    private fun drawEyeContours(
        canvas: Canvas,
        mesh: FloatArray,
        scaleX: Float,
        scaleY: Float
    ) {
        // 왼쪽 눈 (화면 기준 오른쪽)
        val leftEyeIndices = intArrayOf(
            362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387,
            386, 385, 384, 398, 362
        )
        drawConnectedLandmarks(canvas, mesh, leftEyeIndices, scaleX, scaleY)

        // 오른쪽 눈 (화면 기준 왼쪽)
        val rightEyeIndices = intArrayOf(
            33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158,
            159, 160, 161, 246, 33
        )
        drawConnectedLandmarks(canvas, mesh, rightEyeIndices, scaleX, scaleY)
    }

    /**
     * 입술 윤곽선 그리기
     */
    private fun drawLipsContour(
        canvas: Canvas,
        mesh: FloatArray,
        scaleX: Float,
        scaleY: Float
    ) {
        // 외곽 입술
        val outerLipsIndices = intArrayOf(
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409,
            270, 269, 267, 0, 37, 39, 40, 185, 61
        )
        drawConnectedLandmarks(canvas, mesh, outerLipsIndices, scaleX, scaleY)
    }

    /**
     * 연결된 랜드마크 그리기
     */
    private fun drawConnectedLandmarks(
        canvas: Canvas,
        mesh: FloatArray,
        indices: IntArray,
        scaleX: Float,
        scaleY: Float
    ) {
        if (indices.size < 2) return

        for (i in 0 until indices.size - 1) {
            val idx1 = indices[i]
            val idx2 = indices[i + 1]

            val x1 = mesh[idx1 * 3]
            val y1 = mesh[idx1 * 3 + 1]
            val x2 = mesh[idx2 * 3]
            val y2 = mesh[idx2 * 3 + 1]

            var screenX1 = x1 * imageWidth * scaleX
            val screenY1 = y1 * imageHeight * scaleY
            var screenX2 = x2 * imageWidth * scaleX
            val screenY2 = y2 * imageHeight * scaleY

            if (isMirror) {
                screenX1 = width - screenX1
                screenX2 = width - screenX2
            }

            canvas.drawLine(screenX1, screenY1, screenX2, screenY2, meshLinePaint)
        }
    }
}
