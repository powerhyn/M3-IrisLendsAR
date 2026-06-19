package com.irislenssdk.tracking

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.irislenssdk.IrisResult
import com.irislenssdk.tracking.math.CoordMapper
import com.irislenssdk.tracking.math.IrisGeometry
import java.nio.ByteBuffer
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * MediaPipe Tasks 478점 → SDK [IrisResult] 변환 계약 (REFACTOR-3-3 plan §4.1).
 *
 * TASKS 공급자 경로와 듀얼 비교 모드(§4.2)가 공유하는 변환 정본이다.
 * 변환된 IrisResult는 LEGACY 경로와 **동일한 DetectionSlot 채널**로 들어가므로,
 * 렌더 경로가 완전히 같고 비교 변인은 추적기뿐이다.
 *
 * 변환 계약 (plan §4.1 / ADR-0001 §7):
 *  1. 좌표: Tasks 출력은 rotationDegrees를 줘도 **원본(미회전) 정규화 좌표**로
 *     재투영되어 반환된다 (함정 #13, LandmarkProjection). 여기서
 *     [CoordMapper.sensorToUpright]로 upright 정규화 공간으로 변환한다.
 *     **미러 적용 금지** (ADR §7.4 — 주입 좌표는 항상 비미러, 미러는 렌더 단일 책임).
 *  2. 홍채: 중심 468/473, 경계 469~472/474~477 (§7.0 순서 right→top→left→bottom).
 *     ④ 좌표 canonical relabeling(ADR §7.3) 적용: IrisResult.left* ← 피험자 좌안
 *     473그룹, right* ← 피험자 우안 468그룹. [LandmarkIndices] 해부학 라벨
 *     (RIGHT_IRIS_CENTER=468)·코어 landmark_injection.h(kLeftIris=473그룹)와 정합.
 *     (W4-C까지는 LEGACY detector 동작 불변 위해 반전 매핑이었으나 ④에서 코어·글루
 *     동시 정정 + 골든 재기준선 완료 — migration checker new.left==old.right 증명.)
 *  3. radius: 중심→경계 4점 평균 **픽셀** 거리 — LEGACY calculateIrisRadius
 *     (mediapipe_detector.cpp:2240)와 동일 수식. 픽셀 환산은 upright 프레임 치수
 *     기준 ([IrisGeometry.radiusPx] — 함정 #5: 정규화 공간 직접 거리 금지).
 *  4. eyelid: EAR=(|p2−p6|+|p3−p5|)/(2|p1−p4|) — 코어 ③-1 어댑터
 *     computeEyeAspectRatio(landmark_injection.h kLeftEAR/kRightEAR)와 동일
 *     수식·동일 좌표공간(upright 정규화).
 *  5. confidence: Tasks는 per-face confidence를 노출하지 않는다 → **1.0 고정**
 *     (ADR §6.2 — '검출 실패 = 주입 부재'로 일원화, 게이팅은 visibility/EAR로).
 *  6. timestamp: 호출자가 전달한 SystemClock 단조 ms.
 *
 * 스레드: 분석 executor 스레드 전용 (내부 재사용 버퍼 — 동시 호출 금지).
 */
object TasksToIrisResult {

    /** IrisResult.left* 필드 소스 — 피험자 좌안 473그룹 (canonical, ADR §7.3; 코어 kLeftIris 정합) */
    private val RESULT_LEFT_IRIS = intArrayOf(473, 474, 475, 476, 477)

    /** IrisResult.right* 필드 소스 — 피험자 우안 468그룹 (canonical) */
    private val RESULT_RIGHT_IRIS = intArrayOf(468, 469, 470, 471, 472)

    /** EAR 6점 — 피험자 좌안 362그룹 (canonical, landmark_injection.h kLeftEAR 정합) */
    private val RESULT_LEFT_EAR = intArrayOf(362, 385, 387, 263, 373, 380)

    /** EAR 6점 — 피험자 우안 33그룹 (canonical, kRightEAR 정합) */
    private val RESULT_RIGHT_EAR = intArrayOf(33, 160, 158, 133, 153, 144)

    // 분석 스레드 전용 재사용 버퍼 (프레임당 힙 할당 회피)
    private val pt = FloatArray(2)
    private val eyeTmp = FloatArray(4) // [cx, cy, z, radiusPx]

    /**
     * Tasks 478점 한 얼굴을 [out]에 변환한다.
     *
     * @param landmarks Tasks 원본(센서/미회전) 정규화 좌표 478점
     * @param rotationDegrees 센서 버퍼를 시계방향으로 돌리면 upright가 되는 각도
     * @param sensorWidth/sensorHeight 분석 버퍼(센서) 크기 px — 90/270이면 내부에서 스왑
     * @param timestampMs SystemClock 단조 타임스탬프 (ms)
     * @return 변환 성공(478점 충족) 여부. false면 [out]은 noFace 상태로 채워진다.
     */
    fun convert(
        landmarks: List<NormalizedLandmark>,
        rotationDegrees: Int,
        sensorWidth: Int,
        sensorHeight: Int,
        timestampMs: Long,
        out: IrisResult,
    ): Boolean {
        if (landmarks.size < LandmarkIndices.LANDMARK_COUNT) {
            fillNoFace(rotationDegrees, sensorWidth, sensorHeight, timestampMs, out)
            return false
        }
        out.reset()

        // upright 프레임 치수 — 90/270 회전 시 스왑은 공급자 책임 (ADR §7.1)
        val swap = isSwap(rotationDegrees)
        val upW = if (swap) sensorHeight else sensorWidth
        val upH = if (swap) sensorWidth else sensorHeight
        val upWf = upW.toFloat()
        val upHf = upH.toFloat()

        // ① 478점 전체: 센서 → upright 정규화 (함정 #13). z는 Tasks 표준 스케일 그대로 —
        //    비기하 용도 한정 (ADR §7.2; 홍채 z 468~477 기하 사용 금지는 radius가
        //    x/y 픽셀 거리만 쓰므로 자동 충족). 미러 미적용 (ADR §7.4).
        val mesh = out.faceMesh
        for (i in 0 until LandmarkIndices.LANDMARK_COUNT) {
            val l = landmarks[i]
            CoordMapper.sensorToUpright(l.x(), l.y(), rotationDegrees, pt)
            mesh[i * 3] = pt[0]
            mesh[i * 3 + 1] = pt[1]
            mesh[i * 3 + 2] = l.z()
        }
        out.faceMeshValid = true

        // ② 홍채 중심·반경 — 검출 판정은 LEGACY extractIris 재현 (5점 모두 [0,1] 범위)
        out.leftDetected = deriveEye(mesh, RESULT_LEFT_IRIS, upWf, upHf, eyeTmp)
        out.leftIrisX = eyeTmp[0]
        out.leftIrisY = eyeTmp[1]
        out.leftIrisZ = eyeTmp[2]
        out.leftRadius = eyeTmp[3]

        out.rightDetected = deriveEye(mesh, RESULT_RIGHT_IRIS, upWf, upHf, eyeTmp)
        out.rightIrisX = eyeTmp[0]
        out.rightIrisY = eyeTmp[1]
        out.rightIrisZ = eyeTmp[2]
        out.rightRadius = eyeTmp[3]

        out.detected = out.leftDetected || out.rightDetected

        // ③ confidence — Tasks 미제공 → 1.0 고정 (ADR §6.2, 코어 임계 게이트 통과 상수)
        out.confidence = 1.0f

        // ④ eyelid — EAR, 코어 ③-1 어댑터와 동일 수식·공간 (upright 정규화)
        out.eyelidRatioLeft = ear(mesh, RESULT_LEFT_EAR)
        out.eyelidRatioRight = ear(mesh, RESULT_RIGHT_EAR)

        // ⑤ face_rect — 478점 메시 바운딩 박스, [0,1] 범위 점만 집계.
        //    LEGACY detector·③-1 deriveIrisResult와 동일하게 **정규화 좌표**로 채운다
        //    (IrisResult.java의 '픽셀' 주석은 문서 드리프트 — 감사 finding).
        var minX = Float.MAX_VALUE
        var minY = Float.MAX_VALUE
        var maxX = -Float.MAX_VALUE
        var maxY = -Float.MAX_VALUE
        for (i in 0 until LandmarkIndices.LANDMARK_COUNT) {
            val x = mesh[i * 3]
            val y = mesh[i * 3 + 1]
            if (x < 0f || x > 1f || y < 0f || y > 1f) continue
            if (x < minX) minX = x
            if (y < minY) minY = y
            if (x > maxX) maxX = x
            if (y > maxY) maxY = y
        }
        if (minX <= maxX && minY <= maxY) {
            out.faceRectX = minX
            out.faceRectY = minY
            out.faceRectWidth = maxX - minX
            out.faceRectHeight = maxY - minY
        }

        // ⑥ 얼굴 자세 — Tasks transformation matrix 비활성(지연 절약, FaceTracker 동일
        //    설정)이라 미산출 0 유지. 렌더 미소비, OverlayView 표기용 필드일 뿐.

        out.timestampMs = timestampMs
        out.frameWidth = upW
        out.frameHeight = upH
        return true
    }

    /** 미검출 프레임 — LEGACY NO_FACE와 동일하게 stabilize(hold/fade-out) 입력이 된다. */
    fun fillNoFace(
        rotationDegrees: Int,
        sensorWidth: Int,
        sensorHeight: Int,
        timestampMs: Long,
        out: IrisResult,
    ) {
        out.reset()
        val swap = isSwap(rotationDegrees)
        out.frameWidth = if (swap) sensorHeight else sensorWidth
        out.frameHeight = if (swap) sensorWidth else sensorHeight
        out.timestampMs = timestampMs
    }

    /**
     * P7-W2 패리티: 홍채 ROI 평균 linear luma → [IrisResult.avgIrisLumaLeft]/Right.
     *
     * LEGACY detector calculateIrisLuma(mediapipe_detector.cpp:2273)와 동일 수식:
     * srgb=v/255, linear=srgb² (toLinearFast), luma=dot(linear, Rec.709),
     * ROI=중심 ±0.65·radius 원형 마스크 전수 평균, clamp [0.01, 0.81], 무샘플 시 -1.
     * 센서 좌표계 RGBA 버퍼에서 직접 측정한다 — 디스크는 회전 불변이라 LEGACY의
     * upright RGB 측정과 동치다.
     *
     * 비용: 눈당 디스크 bbox 전수 스캔 (반경 ~40px 기준 약 8천 픽셀, 분석 스레드).
     * [rgba]는 절대 인덱스 get만 사용한다 (position/limit 불변).
     */
    fun fillIrisLuma(
        rgba: ByteBuffer,
        rowStride: Int,
        sensorWidth: Int,
        sensorHeight: Int,
        landmarks: List<NormalizedLandmark>,
        out: IrisResult,
    ) {
        if (out.leftDetected) {
            out.avgIrisLumaLeft =
                roiLumaLinear(rgba, rowStride, sensorWidth, sensorHeight, landmarks, RESULT_LEFT_IRIS)
        }
        if (out.rightDetected) {
            out.avgIrisLumaRight =
                roiLumaLinear(rgba, rowStride, sensorWidth, sensorHeight, landmarks, RESULT_RIGHT_IRIS)
        }
    }

    /**
     * ④ W4-B4: 478점 한 얼굴을 변환하고 홍채 ROI 평균 luma까지 채운 **완전한** [IrisResult]를 만든다.
     *
     * [convert] + [fillIrisLuma]를 SDK 내부에서 일원화한 단일 진입점이다. 이로써 SDK 단독 소비자가
     * 두 단계를 따로 호출하지 않아도 avg_iris_luma(P7-W2 렌즈 색 적응, default ON)가 채워진 결과를
     * 얻는다 — 측정 orchestration을 호출자(데모)에 의존하던 조용한 퇴화를 차단한다.
     *
     * [landmarks]는 convert·luma 측정에 **동일 얼굴**이 쓰이도록 호출자가 같은 한 얼굴을 전달한다
     * (다얼굴 선택은 호출자 책임 — W4-C Option A 데모 오케스트레이션 유지). 검출 실패 시 [out]은
     * noFace 상태(luma -1)로 채워진다.
     *
     * @return 변환 성공(478점 충족) 여부.
     */
    fun convertWithLuma(
        landmarks: List<NormalizedLandmark>,
        rotationDegrees: Int,
        sensorWidth: Int,
        sensorHeight: Int,
        timestampMs: Long,
        rgba: ByteBuffer,
        rowStride: Int,
        out: IrisResult,
    ): Boolean {
        val ok = convert(landmarks, rotationDegrees, sensorWidth, sensorHeight, timestampMs, out)
        if (ok) {
            fillIrisLuma(rgba, rowStride, sensorWidth, sensorHeight, landmarks, out)
        }
        return ok
    }

    /**
     * P4-W1-03 패리티: 홍채 중심 5점 크로스 Rec.601 휘도 (0..1, 미검출 -1).
     *
     * LEGACY sampleIrisLuminanceNv21/samplePointLuminanceNv21(GpuRenderActivity)와
     * 동일 의미 — 눈별 개별 샘플 후 **값** 평균 (좌표 평균 금지: 두 눈 사이 피부 오염).
     * NV21 Y채널 대신 RGBA에서 Y′=Rec.601(R′G′B′)로 계산한다 (동일 색공간 근사).
     */
    fun sampleCrossLuma(
        rgba: ByteBuffer,
        rowStride: Int,
        sensorWidth: Int,
        sensorHeight: Int,
        landmarks: List<NormalizedLandmark>,
        result: IrisResult,
    ): Float {
        val left = if (result.leftDetected) {
            crossLuma601(rgba, rowStride, sensorWidth, sensorHeight, landmarks[RESULT_LEFT_IRIS[0]])
        } else -1f
        val right = if (result.rightDetected) {
            crossLuma601(rgba, rowStride, sensorWidth, sensorHeight, landmarks[RESULT_RIGHT_IRIS[0]])
        } else -1f
        return when {
            left >= 0f && right >= 0f -> (left + right) / 2f
            left >= 0f -> left
            right >= 0f -> right
            else -> -1f
        }
    }

    // ───────────────────────── 내부 ─────────────────────────

    private fun isSwap(rotationDegrees: Int): Boolean {
        val r = ((rotationDegrees % 360) + 360) % 360
        return r == 90 || r == 270
    }

    /**
     * upright 메시에서 한쪽 눈 중심·반경 파생.
     * @return 5점 모두 [0,1] 범위 여부 (LEGACY extractIris 검출 판정 재현)
     * @param eyeOut [cx, cy, z, radiusPx]
     */
    private fun deriveEye(
        mesh: FloatArray,
        idx5: IntArray,
        upW: Float,
        upH: Float,
        eyeOut: FloatArray,
    ): Boolean {
        var inRange = true
        for (k in idx5.indices) {
            val x = mesh[idx5[k] * 3]
            val y = mesh[idx5[k] * 3 + 1]
            if (x < 0f || x > 1f || y < 0f || y > 1f) inRange = false
        }
        val c = idx5[0]
        val cx = mesh[c * 3]
        val cy = mesh[c * 3 + 1]
        // 반경: 중심→경계 4점 평균 픽셀 거리 — LEGACY calculateIrisRadius 동일 수식.
        // 픽셀 환산 후 거리 (함정 #5 — 정규화 좌표 직접 거리는 종횡비 왜곡).
        var sum = 0f
        for (k in 1..4) {
            val b = idx5[k]
            sum += IrisGeometry.radiusPx(
                (mesh[b * 3] - cx) * upW,
                (mesh[b * 3 + 1] - cy) * upH,
            )
        }
        eyeOut[0] = cx
        eyeOut[1] = cy
        eyeOut[2] = mesh[c * 3 + 2]
        eyeOut[3] = sum / 4f
        return inRange
    }

    /** EAR=(|p2−p6|+|p3−p5|)/(2|p1−p4|) — 코어 computeEyeAspectRatio 동일 (정규화 공간) */
    private fun ear(mesh: FloatArray, idx: IntArray): Float {
        fun dist(a: Int, b: Int): Float {
            val dx = mesh[idx[a] * 3] - mesh[idx[b] * 3]
            val dy = mesh[idx[a] * 3 + 1] - mesh[idx[b] * 3 + 1]
            return hypot(dx, dy)
        }
        val horizontal = dist(0, 3)
        if (horizontal < 1e-6f) return 0f
        return (dist(1, 5) + dist(2, 4)) / (2f * horizontal)
    }

    private fun roiLumaLinear(
        rgba: ByteBuffer,
        rowStride: Int,
        width: Int,
        height: Int,
        landmarks: List<NormalizedLandmark>,
        idx5: IntArray,
    ): Float {
        val c = landmarks[idx5[0]]
        val cxPx = c.x() * width
        val cyPx = c.y() * height
        // 센서 픽셀 공간 반경 (중심→경계 4점 평균 — 측정 디스크용)
        var radius = 0f
        for (k in 1..4) {
            val b = landmarks[idx5[k]]
            radius += IrisGeometry.radiusPx((b.x() - c.x()) * width, (b.y() - c.y()) * height)
        }
        radius /= 4f
        val r = radius * 0.65f // P7-W2 §5.2 디스크 비율
        if (r <= 0f) return -1f
        val r2 = r * r
        val x0 = max(0, floor(cxPx - r).toInt())
        val y0 = max(0, floor(cyPx - r).toInt())
        val x1 = min(width - 1, ceil(cxPx + r).toInt())
        val y1 = min(height - 1, ceil(cyPx + r).toInt())
        if (x1 < x0 || y1 < y0) return -1f

        val limit = rgba.limit()
        var sum = 0.0
        var count = 0
        for (y in y0..y1) {
            val dy = y - cyPx
            val rowBase = y * rowStride
            for (x in x0..x1) {
                val dx = x - cxPx
                if (dx * dx + dy * dy > r2) continue // 원형 마스크 밖
                val off = rowBase + x * 4
                if (off < 0 || off + 2 >= limit) continue // 버퍼 경계 방어
                val sr = (rgba.get(off).toInt() and 0xFF) * INV_255
                val sg = (rgba.get(off + 1).toInt() and 0xFF) * INV_255
                val sb = (rgba.get(off + 2).toInt() and 0xFF) * INV_255
                // linear = srgb² (toLinearFast), Rec.709 — sRGB 평균 금지 (P7-W2 §5.5)
                sum += (0.2126f * sr * sr + 0.7152f * sg * sg + 0.0722f * sb * sb).toDouble()
                count++
            }
        }
        if (count == 0) return -1f
        return (sum / count).toFloat().coerceIn(0.01f, 0.81f) // consumer clamp 동일 범위
    }

    /** 단일 중심점 5점 크로스(중심+상하좌우 r=3px) Rec.601 휘도 — LEGACY 크로스 샘플 재현 */
    private fun crossLuma601(
        rgba: ByteBuffer,
        rowStride: Int,
        width: Int,
        height: Int,
        center: NormalizedLandmark,
    ): Float {
        val cx = (center.x() * width).toInt().coerceIn(0, width - 1)
        val cy = (center.y() * height).toInt().coerceIn(0, height - 1)
        val r = 3.coerceAtMost(minOf(cx, cy, width - 1 - cx, height - 1 - cy))
        val limit = rgba.limit()
        var sum = 0f
        var count = 0
        // (dx,dy) 쌍: center + 4방향 — LEGACY offsets와 동일
        val offsets = intArrayOf(0, 0, 0, -r, 0, r, -r, 0, r, 0)
        for (i in offsets.indices step 2) {
            val off = (cy + offsets[i + 1]) * rowStride + (cx + offsets[i]) * 4
            if (off < 0 || off + 2 >= limit) continue
            val rr = (rgba.get(off).toInt() and 0xFF).toFloat()
            val gg = (rgba.get(off + 1).toInt() and 0xFF).toFloat()
            val bb = (rgba.get(off + 2).toInt() and 0xFF).toFloat()
            sum += 0.299f * rr + 0.587f * gg + 0.114f * bb
            count++
        }
        if (count == 0) return -1f
        return (sum / count) / 255f
    }

    private const val INV_255 = 1f / 255f
}
