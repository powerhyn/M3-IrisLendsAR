package com.irislenssdk.demo.tracking

import android.content.Context
import android.os.SystemClock
import android.util.Log
import com.google.mediapipe.framework.image.ByteBufferImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.irislenssdk.IrisResult
import com.irislenssdk.tracking.EmulatorDetector
import com.irislenssdk.tracking.TasksToIrisResult
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import kotlin.math.abs
import kotlin.math.hypot

/**
 * ③-3 §4.2 듀얼 비교 모드 (T1 정밀 판정) — 같은 NV21 프레임을
 * ① 자체 detectWithRotation(stabilize 전 raw) ② Tasks **IMAGE 모드** 동기 detect
 * 로 양쪽 실행해 좌표 계약 위반 패턴을 측정한다.
 *
 * - 토글(공급자 교체)과 별개의 측정 전용 채널 — LEGACY 라이브 경로에서만 가동
 *   (NV21 입력 전제). 기본 OFF, OFF 시 비용 0.
 * - 입력 공정성 (plan §5 주의 2): 양쪽 모두 같은 NV21 픽셀을 소비한다. Tasks 쪽
 *   NV21→RGBA 변환 비용은 detect와 분리해 `convMs`로 별도 표기한다 (Bitmap 경유
 *   금지 — ADR-0002, ByteBufferImageBuilder 직접 사용).
 * - **rot0(실기기 자연 경로) 중심 판정** (ADR §10 주의): 합성 회전 입력은 기존
 *   추적기 기준 자체가 오염(rot0≡rot180 퇴화)이라 T1에 사용 금지. CSV에 rot를
 *   기록만 한다 — 판정은 실기기 자연 rotation 고정 상태에서.
 * - 페어링: 양쪽 모두 [TasksToIrisResult]의 코어 레거시 시맨틱(left*←468그룹)으로
 *   채워지므로 left↔left 비교가 같은 눈 비교다 (인위적 L/R 스왑 없음).
 *
 * 출력:
 *  - logcat `AB_METRIC` 태그 CSV형 구조화 라인 (헤더 1회 + 프레임당 1줄)
 *  - [WINDOW]프레임(기본 300) 누적 요약 + 종료(flush) 시 요약
 *  - HUD 1줄 콜백 ([onHud], 15프레임 스로틀)
 *
 * 스레드: 분석 executor 스레드 전용 — IMAGE 모드도 GPU delegate 스레드 친화성
 * 규약(§5 주의 1)을 따른다 (생성/​detect/close 동일 스레드).
 */
internal class AbMeasure(
    private val context: Context,
    private val onHud: (String) -> Unit,
) {
    private var landmarker: FaceLandmarker? = null
    private var usingGpu = false
    private var cpuForced = false
    private var creationFailed = false

    private var cachedRotation = -1
    private var cachedOptions: ImageProcessingOptions? = null

    // NV21→RGBA 재사용 버퍼 (측정 모드 동안만 점유)
    private var rgbaBytes: ByteArray? = null
    private var rgbaBuffer: ByteBuffer? = null

    private val tasksResult = IrisResult()
    private var headerLogged = false

    // ── 누적 (분석 스레드 전용, WINDOW 단위 리셋) ──
    private var frames = 0
    private var legacyDetN = 0
    private var tasksDetN = 0
    private var bothEyeN = 0 // 같은 눈 양쪽 검출 인스턴스 수 (눈 단위)
    private var sumErrRatio = 0.0
    private var maxErrRatio = 0.0
    private var sumDx = 0.0 // 정규화 dx 합 (계통 오프셋)
    private var sumDy = 0.0
    private var posDx = 0
    private var negDx = 0
    private var posDy = 0
    private var negDy = 0
    private var mirrorXFlags = 0
    private var mirrorYFlags = 0
    private var swapFlags = 0
    private var sumRadRatio = 0.0
    private var radRatioN = 0
    private var sumLegacyMs = 0.0
    private var sumTasksMs = 0.0
    private var sumConvMs = 0.0

    /**
     * 한 프레임 측정. [legacyRaw]는 detectWithRotation 직후·stabilize 전 raw 결과.
     * 분석 스레드 동기 실행 — 측정 중에는 라이브 FPS가 떨어진다 (측정 모드 전제,
     * CameraX KEEP_ONLY_LATEST가 백프레셔를 흡수).
     */
    fun measureFrame(
        nv21: ByteArray,
        sensorWidth: Int,
        sensorHeight: Int,
        rotation: Int,
        legacyRaw: IrisResult,
        legacyMs: Float,
    ) {
        val lm = ensureLandmarker() ?: return

        val tConv0 = SystemClock.elapsedRealtimeNanos()
        val buffer = nv21ToRgba(nv21, sensorWidth, sensorHeight)
        val convMs = (SystemClock.elapsedRealtimeNanos() - tConv0) / 1e6f

        val mpImage: MPImage =
            ByteBufferImageBuilder(buffer, sensorWidth, sensorHeight, MPImage.IMAGE_FORMAT_RGBA).build()
        val ts = SystemClock.uptimeMillis()
        val tDet0 = SystemClock.elapsedRealtimeNanos()
        val result = try {
            lm.detect(mpImage, processingOptions(rotation))
        } catch (e: RuntimeException) {
            handleDetectError(e)
            return
        }
        val tasksMs = (SystemClock.elapsedRealtimeNanos() - tDet0) / 1e6f

        val faces = result.faceLandmarks()
        val converted = faces.isNotEmpty() &&
            TasksToIrisResult.convert(faces[0], rotation, sensorWidth, sensorHeight, ts, tasksResult)
        if (!converted) {
            TasksToIrisResult.fillNoFace(rotation, sensorWidth, sensorHeight, ts, tasksResult)
        }

        accumulate(rotation, legacyRaw, tasksResult, legacyMs, tasksMs, convMs)
    }

    /** 측정 종료 — 잔여 누적분 요약 후 리셋. 분석 스레드에서 호출. */
    fun flush() {
        summarize("flush")
        resetWindow()
    }

    /** 분석 스레드에서 호출 — landmarker를 생성한 스레드에서 닫는다 (스레드 친화성). */
    fun close() {
        flush()
        try {
            landmarker?.close()
        } catch (_: RuntimeException) {
        }
        landmarker = null
    }

    // ───────────────────────── 내부 ─────────────────────────

    private fun ensureLandmarker(): FaceLandmarker? {
        if (landmarker != null || creationFailed) return landmarker
        val wantGpu = !cpuForced && !EmulatorDetector.isEmulator
        landmarker = try {
            createLandmarker(wantGpu).also { usingGpu = wantGpu }
        } catch (e: RuntimeException) {
            if (wantGpu) {
                Log.w(TAG, "summary,warn,GPU delegate 초기화 실패 — CPU 폴백: ${e.message}")
                cpuForced = true
                try {
                    createLandmarker(false).also { usingGpu = false }
                } catch (e2: RuntimeException) {
                    creationFailed = true
                    Log.e(TAG, "summary,error,FaceLandmarker 생성 실패(CPU): ${e2.message}")
                    null
                }
            } else {
                creationFailed = true
                Log.e(TAG, "summary,error,FaceLandmarker 생성 실패: ${e.message}")
                null
            }
        }
        return landmarker
    }

    private fun createLandmarker(useGpu: Boolean): FaceLandmarker {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath(MODEL_ASSET_PATH)
            .setDelegate(if (useGpu) Delegate.GPU else Delegate.CPU)
            .build()
        val options = FaceLandmarker.FaceLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setRunningMode(RunningMode.IMAGE) // §4.2: 동기 IMAGE 모드 — 같은 프레임 즉시 비교
            .setNumFaces(1)
            .setMinFaceDetectionConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .setMinFacePresenceConfidence(0.5f)
            .setOutputFaceBlendshapes(false)
            .setOutputFacialTransformationMatrixes(false)
            .build()
        return FaceLandmarker.createFromOptions(context, options)
    }

    private fun processingOptions(rotation: Int): ImageProcessingOptions {
        if (rotation != cachedRotation || cachedOptions == null) {
            cachedRotation = rotation
            cachedOptions = ImageProcessingOptions.builder().setRotationDegrees(rotation).build()
        }
        return cachedOptions!!
    }

    private fun handleDetectError(e: RuntimeException) {
        if (usingGpu && !cpuForced) {
            // GPU 런타임 오류 → 다음 프레임에 CPU 재생성
            cpuForced = true
            try {
                landmarker?.close()
            } catch (_: RuntimeException) {
            }
            landmarker = null
            Log.w(TAG, "summary,warn,GPU 추론 오류 — CPU 재생성: ${e.message}")
        } else {
            Log.e(TAG, "summary,error,Tasks 추론 오류: ${e.message}")
        }
    }

    private fun accumulate(
        rotation: Int,
        legacy: IrisResult,
        tasks: IrisResult,
        legacyMs: Float,
        tasksMs: Float,
        convMs: Float,
    ) {
        if (!headerLogged) {
            headerLogged = true
            Log.i(
                TAG,
                "hdr,frame,rot,legDet,taskDet," +
                    "errLpx,errLr,errRpx,errRr,radRatioL,radRatioR," +
                    "dxL,dyL,dxR,dyR,mirX,mirY,swp,legacyMs,tasksMs,convMs,delegate",
            )
        }
        frames++
        if (legacy.detected) legacyDetN++
        if (tasks.detected) tasksDetN++

        // upright 픽셀 환산 기준 — LEGACY 결과의 frame dims (양쪽 동일 upright 공간 계약)
        val w = (if (legacy.frameWidth > 0) legacy.frameWidth else tasks.frameWidth).toFloat()
        val h = (if (legacy.frameHeight > 0) legacy.frameHeight else tasks.frameHeight).toFloat()

        var errLpx = -1f
        var errLr = -1f
        var radRatioL = -1f
        var dxL = Float.NaN
        var dyL = Float.NaN
        var mirX = 0
        var mirY = 0
        var swp = 0

        if (legacy.leftDetected && tasks.leftDetected) {
            dxL = tasks.leftIrisX - legacy.leftIrisX
            dyL = tasks.leftIrisY - legacy.leftIrisY
            errLpx = hypot(dxL * w, dyL * h)
            if (legacy.leftRadius > 1e-3f) errLr = errLpx / legacy.leftRadius
            if (legacy.leftRadius > 1e-3f && tasks.leftRadius > 0f) {
                radRatioL = tasks.leftRadius / legacy.leftRadius
                sumRadRatio += radRatioL.toDouble()
                radRatioN++
            }
            val flags = patternFlags(
                legacy.leftIrisX, legacy.leftIrisY, tasks.leftIrisX, tasks.leftIrisY,
            )
            mirX += flags and 1
            mirY += (flags shr 1) and 1
            swp += (flags shr 2) and 1
            accumulateEye(dxL, dyL, errLr)
        }

        var errRpx = -1f
        var errRr = -1f
        var radRatioR = -1f
        var dxR = Float.NaN
        var dyR = Float.NaN

        if (legacy.rightDetected && tasks.rightDetected) {
            dxR = tasks.rightIrisX - legacy.rightIrisX
            dyR = tasks.rightIrisY - legacy.rightIrisY
            errRpx = hypot(dxR * w, dyR * h)
            if (legacy.rightRadius > 1e-3f) errRr = errRpx / legacy.rightRadius
            if (legacy.rightRadius > 1e-3f && tasks.rightRadius > 0f) {
                radRatioR = tasks.rightRadius / legacy.rightRadius
                sumRadRatio += radRatioR.toDouble()
                radRatioN++
            }
            val flags = patternFlags(
                legacy.rightIrisX, legacy.rightIrisY, tasks.rightIrisX, tasks.rightIrisY,
            )
            mirX += flags and 1
            mirY += (flags shr 1) and 1
            swp += (flags shr 2) and 1
            accumulateEye(dxR, dyR, errRr)
        }

        mirrorXFlags += mirX
        mirrorYFlags += mirY
        swapFlags += swp
        sumLegacyMs += legacyMs.toDouble()
        sumTasksMs += tasksMs.toDouble()
        sumConvMs += convMs.toDouble()

        Log.i(
            TAG,
            String.format(
                Locale.US,
                "f,%d,%d,%d,%d,%.1f,%.3f,%.1f,%.3f,%.3f,%.3f,%+.4f,%+.4f,%+.4f,%+.4f,%d,%d,%d,%.1f,%.1f,%.1f,%s",
                frames, rotation,
                if (legacy.detected) 1 else 0, if (tasks.detected) 1 else 0,
                errLpx, errLr, errRpx, errRr, radRatioL, radRatioR,
                dxL, dyL, dxR, dyR, mirX, mirY, swp,
                legacyMs, tasksMs, convMs,
                if (usingGpu) "gpu" else "cpu",
            ),
        )

        if (frames % HUD_EVERY == 0) onHud(hudLine())
        if (frames >= WINDOW) {
            summarize("window")
            resetWindow()
        }
    }

    private fun accumulateEye(dx: Float, dy: Float, errRatio: Float) {
        bothEyeN++
        sumDx += dx.toDouble()
        sumDy += dy.toDouble()
        if (dx > SIGN_EPS) posDx++ else if (dx < -SIGN_EPS) negDx++
        if (dy > SIGN_EPS) posDy++ else if (dy < -SIGN_EPS) negDy++
        if (errRatio >= 0f) {
            sumErrRatio += errRatio.toDouble()
            if (errRatio > maxErrRatio) maxErrRatio = errRatio.toDouble()
        }
    }

    /**
     * 축 스왑/미러/정규화 오류 전형 패턴 플래그 (§4.2 휴리스틱).
     * @return bit0=mirrorX(dx≈1−2x 류), bit1=mirrorY, bit2=axisSwap
     */
    private fun patternFlags(lx: Float, ly: Float, tx: Float, ty: Float): Int {
        var flags = 0
        // 미러 반전: tasks_x ≈ 1−legacy_x 이면서 원좌표끼리는 크게 어긋남
        if (abs(tx - (1f - lx)) < PATTERN_EPS && abs(tx - lx) > PATTERN_GROSS) flags = flags or 1
        if (abs(ty - (1f - ly)) < PATTERN_EPS && abs(ty - ly) > PATTERN_GROSS) flags = flags or 2
        // 축 스왑: (x,y) ↔ (y,x)
        if (abs(tx - ly) < PATTERN_EPS && abs(ty - lx) < PATTERN_EPS &&
            (abs(tx - lx) > PATTERN_GROSS || abs(ty - ly) > PATTERN_GROSS)
        ) {
            flags = flags or 4
        }
        return flags
    }

    private fun summarize(reason: String) {
        if (frames == 0) return
        val meanErrRatio = if (bothEyeN > 0) sumErrRatio / bothEyeN else -1.0
        val meanDx = if (bothEyeN > 0) sumDx / bothEyeN else 0.0
        val meanDy = if (bothEyeN > 0) sumDy / bothEyeN else 0.0
        val signX = signConsistency(posDx, negDx)
        val signY = signConsistency(posDy, negDy)
        val mirrorFrac = if (bothEyeN > 0) (mirrorXFlags + mirrorYFlags).toDouble() / bothEyeN else 0.0
        val swapFrac = if (bothEyeN > 0) swapFlags.toDouble() / bothEyeN else 0.0
        val meanRad = if (radRatioN > 0) sumRadRatio / radRatioN else -1.0

        // T1 보조 라벨 (ADR §10 — 판정 주체는 사용자, 이 라벨은 화면/로그 보조)
        val verdict = when {
            swapFrac > 0.5 -> "T1_SUSPECT_AXIS_SWAP"
            mirrorFrac > 0.5 -> "T1_SUSPECT_MIRROR"
            meanErrRatio >= 1.0 && (signX >= 0.9 || signY >= 0.9) -> "T1_SUSPECT_SYSTEMATIC_OFFSET_GE_100PCT_RADIUS"
            else -> "NO_CONTRACT_VIOLATION_PATTERN"
        }

        Log.i(
            TAG,
            String.format(
                Locale.US,
                "summary,%s,n=%d,legacyDet=%.2f,tasksDet=%.2f,eyePairs=%d," +
                    "meanErrRatio=%.3f,maxErrRatio=%.3f,meanDx=%+.4f,meanDy=%+.4f," +
                    "signX=%.2f,signY=%.2f,mirrorFrac=%.3f,swapFrac=%.3f,meanRadRatio=%.3f," +
                    "legacyMs=%.1f,tasksMs=%.1f,convMs=%.1f,delegate=%s,verdict=%s",
                reason, frames,
                legacyDetN.toDouble() / frames, tasksDetN.toDouble() / frames, bothEyeN,
                meanErrRatio, maxErrRatio, meanDx, meanDy,
                signX, signY, mirrorFrac, swapFrac, meanRad,
                sumLegacyMs / frames, sumTasksMs / frames, sumConvMs / frames,
                if (usingGpu) "gpu" else "cpu", verdict,
            ),
        )
        onHud(hudLine())
    }

    private fun hudLine(): String {
        val meanErrRatio = if (bothEyeN > 0) sumErrRatio / bothEyeN else -1.0
        return String.format(
            Locale.US,
            "AB n=%d err=%.0f%%r mir=%d swp=%d L%.0f/T%.0f+c%.0fms",
            frames,
            meanErrRatio * 100.0,
            mirrorXFlags + mirrorYFlags,
            swapFlags,
            if (frames > 0) sumLegacyMs / frames else 0.0,
            if (frames > 0) sumTasksMs / frames else 0.0,
            if (frames > 0) sumConvMs / frames else 0.0,
        )
    }

    private fun signConsistency(pos: Int, neg: Int): Double {
        val total = pos + neg
        if (total == 0) return 0.0
        return maxOf(pos, neg).toDouble() / total
    }

    private fun resetWindow() {
        frames = 0
        legacyDetN = 0
        tasksDetN = 0
        bothEyeN = 0
        sumErrRatio = 0.0
        maxErrRatio = 0.0
        sumDx = 0.0
        sumDy = 0.0
        posDx = 0
        negDx = 0
        posDy = 0
        negDy = 0
        mirrorXFlags = 0
        mirrorYFlags = 0
        swapFlags = 0
        sumRadRatio = 0.0
        radRatioN = 0
        sumLegacyMs = 0.0
        sumTasksMs = 0.0
        sumConvMs = 0.0
    }

    /** NV21 → RGBA (BT.601 정수 근사, 검출 입력용). 비용은 convMs로 분리 보고. */
    private fun nv21ToRgba(nv21: ByteArray, width: Int, height: Int): ByteBuffer {
        val size = width * height * 4
        var bytes = rgbaBytes
        if (bytes == null || bytes.size != size) {
            bytes = ByteArray(size)
            rgbaBytes = bytes
        }
        var buf = rgbaBuffer
        if (buf == null || buf.capacity() < size) {
            buf = ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder())
            rgbaBuffer = buf
        }
        val frameSize = width * height
        var outIdx = 0
        for (y in 0 until height) {
            val yRow = y * width
            val uvRow = frameSize + (y shr 1) * width
            for (x in 0 until width) {
                val yv = nv21[yRow + x].toInt() and 0xFF
                val uvIdx = uvRow + (x and 0x7FFFFFFE) // NV21: V,U 인터리브
                val v = (nv21[uvIdx].toInt() and 0xFF) - 128
                val u = (nv21[uvIdx + 1].toInt() and 0xFF) - 128
                var r = yv + ((359 * v) shr 8)
                var g = yv - ((88 * u + 183 * v) shr 8)
                var b = yv + ((454 * u) shr 8)
                if (r < 0) r = 0 else if (r > 255) r = 255
                if (g < 0) g = 0 else if (g > 255) g = 255
                if (b < 0) b = 0 else if (b > 255) b = 255
                bytes[outIdx] = r.toByte()
                bytes[outIdx + 1] = g.toByte()
                bytes[outIdx + 2] = b.toByte()
                bytes[outIdx + 3] = -1 // 0xFF
                outIdx += 4
            }
        }
        buf.clear()
        buf.put(bytes, 0, size)
        buf.rewind()
        return buf
    }

    companion object {
        /** 구조화 로그 태그 — `adb logcat -s AB_METRIC` 수집 (plan §7-2) */
        private const val TAG = "AB_METRIC"

        /** FaceTracker.MODEL_ASSET_PATH와 동일 자산 (demo assets/models/) */
        private const val MODEL_ASSET_PATH = "models/face_landmarker.task"

        /** 누적 요약 창 크기 (plan §4.2 기본 300) */
        private const val WINDOW = 300

        private const val HUD_EVERY = 15

        /** 패턴 휴리스틱: 일치 판정 허용 오차 (정규화 좌표) */
        private const val PATTERN_EPS = 0.02f

        /** 패턴 휴리스틱: '크게 어긋남' 임계 (정규화 좌표) */
        private const val PATTERN_GROSS = 0.10f

        /** 부호 일관성 집계에서 0 취급할 미세 오프셋 (정규화 좌표) */
        private const val SIGN_EPS = 0.001f
    }
}
