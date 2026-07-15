package com.irislenssdk.tracking

import android.content.Context
import android.os.SystemClock
import androidx.camera.core.ImageProxy
import com.irislenssdk.tracking.math.CoordMapper
import com.irislenssdk.tracking.math.IrisGeometry
import com.irislenssdk.tracking.math.IrisLumaSampler
import com.irislenssdk.tracking.math.OneEuroFilter
import com.google.mediapipe.framework.image.ByteBufferImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * MediaPipe FaceLandmarker VIDEO(동기) 추적기.
 *
 * 이식 소스: LensSimulator `sdk/android/lenssdk/.../internal/FaceTracker.kt`
 *   리비전 **6bacaec** (값 무변경 이식, plan §2 / ADR §14). 이 리비전과의 차이는
 *   ③-3 적응뿐 — 패키지/임포트 치환, [onRawResult] 탭 추가, [MODEL_ASSET_PATH] 경로.
 *   One-Euro 상수(MIN_CUTOFF/BETA/D_CUTOFF) 및 본체는 6bacaec과 동일하다.
 *
 * LIVE_STREAM(detectAsync)에서 VIDEO(detectForVideo)로 전환 (2026-06-12) —
 * 비동기 모드의 내부 큐잉이 1~2프레임 지연을 추가해 raw 랜드마크부터 눈을 늦게
 * 따라가는 것이 디버그 메시 오버레이로 실측 확인됨. 동기 detect 동안 CameraX
 * KEEP_ONLY_LATEST 백프레셔가 묵은 프레임을 버려 항상 최신 프레임을 처리한다.
 *
 * 스레드 규약:
 * - [analyze]/[close]는 반드시 같은 분석 executor 스레드에서 호출 —
 *   GPU delegate는 초기화한 스레드에서만 detect 가능 (스레드 친화성).
 * - FaceLandmarker는 첫 analyze 시점에 분석 스레드에서 지연 생성한다.
 * - 결과 처리([onResult])도 detectForVideo 직후 같은 분석 스레드에서 동기 수행된다.
 *
 * 입력 경로 (ADR-0002): CameraX RGBA_8888 단일 플레인 → ByteBufferImageBuilder 직접
 * (Bitmap 경유 금지 — 복사 1회 낭비). rowStride 패딩이 있으면 width*4로 압축 복사.
 */
class FaceTracker(
    private val context: Context,
    private val preferGpu: Boolean,
    private val mirror: Boolean,
    private val onSnapshot: (TrackingSnapshot) -> Unit,
    private val onInferenceStats: (inferenceMs: Float, usingGpu: Boolean) -> Unit,
    private val onError: (String) -> Unit,
) {
    @Volatile private var closed = false
    private var landmarker: FaceLandmarker? = null

    /**
     * [③-3 이식 적응 — REFACTOR-3-3 §4.1] 원시 결과 탭.
     *
     * detectForVideo 직후 같은 분석 스레드에서 동기 호출된다. 전달 좌표는
     * **센서(원본·미회전·비미러·무필터) 정규화 공간** 그대로다 — 소비자가
     * CoordMapper.sensorToUpright로 변환한다 (ADR-0001 §7.1, 미러 금지 §7.4).
     *
     * - null(기본)이면 추가 비용 0 — 원본 LensSimulator 동작과 동일.
     * - 이 탭의 소비자는 내장 One-Euro 경로([onSnapshot])를 소비하지 않는 것이
     *   계약이다 — 코어 stabilize와의 이중 필터 금지 (plan §5 주의 3의 '우회' 구현).
     * - [rgba]/[rowStride]는 호출 동안만 유효 (ImageProxy close 전). 절대 인덱스
     *   get만 허용 — position/limit 변경 금지.
     * - onInferenceStats 산출 **후** 호출되므로 inferenceMs를 오염시키지 않는다.
     */
    var onRawResult: ((
        result: FaceLandmarkerResult,
        rotationDegrees: Int,
        srcWidth: Int,
        srcHeight: Int,
        rgba: ByteBuffer,
        rowStride: Int,
        // 분석 프레임의 센서 타임스탬프(ns) — frame-sync 렌더링이 화면측 SurfaceTexture.timestamp와
        // 매칭한다 (같은 센서 클럭). 트래킹 지연 핸드오프 §3-b.
        frameTimestampNs: Long,
    ) -> Unit)? = null

    @Volatile private var usingGpu = false

    // 트래킹 A/B(벤치 임시): 실행 모드 3종 — MODE_VIDEO(동기+ROI 추적, numFaces=1이면 내부
    // 스무딩 활성) / MODE_IMAGE(동기, 스무딩·추적 없음) / MODE_STREAM(LIVE_STREAM 비동기 —
    // 내부 큐잉, LensSim 실측 1~2프레임 지연의 재현 팔).
    // 요청은 아무 스레드나, 실제 재생성은 분석 스레드(ensureLandmarker)에서만.
    @Volatile private var runningModeRequested = MODE_VIDEO
    private var runningModeActive = MODE_VIDEO

    /** IMAGE 모드 A/B 토글 (레거시 표면 — img 버튼) — 어느 스레드에서나 안전. */
    fun setImageMode(enabled: Boolean) {
        runningModeRequested = if (enabled) MODE_IMAGE else MODE_VIDEO
    }

    /** 실행 모드 직접 지정 (벤치 임시) — MODE_VIDEO/IMAGE/STREAM. 다음 analyze에서 재생성. */
    fun setTrackingRunningMode(mode: Int) {
        runningModeRequested = mode.coerceIn(MODE_VIDEO, MODE_STREAM)
    }

    // P8 트래킹 A/B(벤치 임시): numFaces=1 토글 — VIDEO+numFaces=1 조합에서만 활성화되는
    // MediaPipe 내부 LandmarksSmoothingCalculator(One-Euro 0.05/80)를 "켠" 팔을 실기기로
    // 체감하기 위한 토글. 기본 false(=numFaces=2, 스무딩 우회)가 정식 경로.
    @Volatile private var singleFaceRequested = false
    private var singleFaceActive = false

    /** numFaces=1(MP 내부 스무딩 ON) A/B 토글 — 어느 스레드에서나 안전. 다음 analyze에서 재생성. */
    fun setSingleFaceMode(enabled: Boolean) {
        singleFaceRequested = enabled
    }

    /** GPU 추론 중 런타임 오류 발생 시 분석 스레드에서 CPU로 재생성하기 위한 플래그 */
    @Volatile private var cpuFallbackRequested = false
    private var cpuForced = false
    private var creationFailed = false
    private var detectErrorReported = false

    private var lastTimestampMs = 0L

    /** 결과 콜백에서 upright 크기 계산용 — 세션 내 고정값 */
    @Volatile private var lastRotationDegrees = 0
    private var cachedRotation = -1
    private var cachedProcessingOptions: ImageProcessingOptions? = null

    /** rowStride 압축 복사용 재사용 버퍼 (프레임당 힙 할당 방지) */
    private var compactBuffer: ByteBuffer? = null

    // One-Euro 필터 — 눈 2개 × [cx, cy, ux, uy, vx, vy] = 12채널.
    // 분석 upright "픽셀" 좌표 공간에서 필터링 (정규화 공간은 종횡비 왜곡).
    private val filters = Array(12) { OneEuroFilter(MIN_CUTOFF, BETA, D_CUTOFF) }

    // FACE_OVAL 36점 × (x,y) = 72채널 전용 필터 — 뷰티(턱선 워프/피부 마스크)용.
    // 눈과 동일 파라미터, 동일하게 픽셀 공간에서 필터링.
    private val ovalFilters = Array(LandmarkIndices.FACE_OVAL.size * 2) {
        OneEuroFilter(MIN_CUTOFF, BETA, D_CUTOFF)
    }
    private var hadFace = false

    /** 뷰티 게이팅 — false면 onResult에서 뷰티 랜드마크 추출/필터링을 생략한다 (추가 비용 0) */
    @Volatile private var beautyEnabled = false

    // 결과 콜백 내 재사용 배열 (할당 최소화)
    private val basisTmp = FloatArray(6)
    private val ptCenter = FloatArray(2)
    private val ptRight = FloatArray(2)
    private val ptTop = FloatArray(2)
    private val ptTmp = FloatArray(2)

    // ── 홍채 평균 휘도 측정 (ADR-0004 노출 매칭) ──
    // 직전 프레임의 원시(센서/분석 버퍼 좌표계, upright 변환·미러 전) 홍채 중심·반경 캐시.
    // MediaPipe 콜백 스레드가 쓰고 분석 스레드가 읽는다 — volatile 플래그 발행으로 가시성 확보
    // (경합 시 찢어진 읽기는 인접 프레임 값의 혼합이라 무해 — EMA가 흡수).
    private val rawIrisCache = FloatArray(6) // [cxN, cyN, rN(폭 정규화)] × 우안/좌안

    @Volatile private var rawIrisCacheValid = false

    /** EMA(α=0.2) 스무딩된 홍채 평균 휘도 (Rec.601, 0..1). 분석 스레드 갱신 → 콜백 스레드 읽기 */
    @Volatile private var irisLumaEma = DEFAULT_IRIS_LUMA

    /** 샘플 오프셋 재사용 버퍼 (프레임당 힙 할당 0) */
    private val lumaSampleOffsets = IntArray(IrisLumaSampler.MAX_SAMPLES)

    /** 분석 executor 스레드 전용. */
    fun analyze(imageProxy: ImageProxy) {
        if (closed) {
            imageProxy.close()
            return
        }
        ensureLandmarker()
        val lm = landmarker
        if (lm == null) {
            imageProxy.close()
            return
        }

        val rotation = imageProxy.imageInfo.rotationDegrees
        lastRotationDegrees = rotation

        // 분석 프레임의 센서 타임스탬프 (ns) — 프레임 동기 렌더링이 같은 클럭의
        // SurfaceTexture.timestamp와 매칭한다 (MediaPipe용 ts(uptime ms)와 별개 도메인).
        // 클럭 게이트 실기기 검증 완료(06-15): 화면측 SurfaceTexture.timestamp와 동일 클럭.
        val frameTimestampNs = imageProxy.imageInfo.timestamp

        val width = imageProxy.width
        val height = imageProxy.height
        val plane = imageProxy.planes[0]
        val src = plane.buffer
        val rowBytes = width * 4

        // detect 전 동기 샘플 (버퍼 유효 구간) — 직전 프레임 랜드마크의 원시 좌표로
        // 현재 버퍼에서 홍채 평균 휘도를 측정한다.
        sampleIrisLuma(src, width, height, plane.rowStride)

        // LIVE_STREAM(벤치 임시): 비동기 제출 — detectAsync 반환 즉시 imageProxy가 닫히므로
        // 원본 버퍼를 풀 버퍼로 복사해 결과 콜백까지 수명을 보장한다.
        if (runningModeActive == MODE_STREAM) {
            try {
                submitStreamFrame(lm, rotation, frameTimestampNs, width, height,
                    plane.rowStride, src, rowBytes)
            } catch (e: RuntimeException) {
                handleDetectError(e)
            } finally {
                imageProxy.close()
            }
            return
        }

        val mpImage: MPImage = if (plane.rowStride == rowBytes) {
            // 패딩 없음 — 제로카피 경로. detectForVideo가 동기라 호출이 반환될 때까지
            // 버퍼가 유효하며, ImageProxy는 finally에서 닫는다.
            src.rewind()
            ByteBufferImageBuilder(src, width, height, MPImage.IMAGE_FORMAT_RGBA).build()
        } else {
            // rowStride 패딩 존재 → width*4로 압축 복사 (재사용 버퍼)
            val compact = obtainCompactBuffer(rowBytes * height)
            compact.clear()
            val capacity = src.capacity()
            for (row in 0 until height) {
                val pos = row * plane.rowStride
                src.limit(minOf(capacity, pos + rowBytes))
                src.position(pos)
                compact.put(src)
            }
            src.limit(capacity)
            compact.rewind()
            ByteBufferImageBuilder(compact, width, height, MPImage.IMAGE_FORMAT_RGBA).build()
        }

        // 타임스탬프 단조 증가 보장 (VIDEO 모드도 필수 조건)
        val ts = maxOf(SystemClock.uptimeMillis(), lastTimestampMs + 1)
        lastTimestampMs = ts

        try {
            // 동기 추론 — 결과를 즉시 반영 (LIVE_STREAM 내부 큐잉 지연 제거).
            // 추론 동안 이 스레드가 점유되어 CameraX가 자동으로 최신 프레임만 남긴다.
            // IMAGE 모드: result.timestampMs()가 무의미(핸드오프 §3-3) — 아래 onResult가
            // 제출 시각 ts를 별도 운반하므로 두 모드 모두 시간 계약이 동일하게 유지된다.
            val result = if (runningModeActive == MODE_IMAGE) {
                lm.detect(mpImage, processingOptions(rotation))
            } else {
                lm.detectForVideo(mpImage, processingOptions(rotation), ts)
            }
            onResult(result, mpImage, frameTimestampNs, ts)
            // ③-3: 원시 결과 탭 — imageProxy close 전(버퍼 유효 구간)·통계 산출 후 동기 호출.
            // 소비자 예외는 격리한다 — detect 오류 처리(handleDetectError)로 흘러가면
            // GPU 폴백을 오판하기 때문.
            try {
                onRawResult?.invoke(result, rotation, width, height, src, plane.rowStride, frameTimestampNs)
            } catch (e: RuntimeException) {
                onError("onRawResult 소비자 오류: ${e.message}")
            }
        } catch (e: RuntimeException) {
            handleDetectError(e)
        } finally {
            imageProxy.close()
        }
    }

    // ── LIVE_STREAM 벤치 경로 (임시 — 판정 후 제거) ──────────────────────
    // detectAsync는 결과가 MediaPipe 콜백 스레드로 늦게 도착하므로, 제출 시점 메타
    // (센서 ns·회전·크기)와 복사 버퍼를 tsMs 키로 대기열에 보관했다가 결과와 매칭한다.
    // 버퍼는 소형 풀(최대 4) 재사용 — 풀 고갈 시 프레임 드롭(백프레셔).

    private class PendingStreamFrame(
        val tsMs: Long, val frameNs: Long, val rotation: Int,
        val width: Int, val height: Int, val buffer: ByteBuffer,
    )

    private val pendingStreamFrames = ArrayDeque<PendingStreamFrame>()
    private val streamBufferPool = ArrayDeque<ByteBuffer>()
    private var streamBuffersAllocated = 0

    private fun submitStreamFrame(
        lm: FaceLandmarker, rotation: Int, frameTimestampNs: Long,
        width: Int, height: Int, rowStride: Int, src: ByteBuffer, rowBytes: Int,
    ) {
        val buf = obtainStreamBuffer(rowBytes * height) ?: return // 인플라이트 초과 — 드롭
        buf.clear()
        val capacity = src.capacity()
        for (row in 0 until height) {
            val pos = row * rowStride
            src.limit(minOf(capacity, pos + rowBytes))
            src.position(pos)
            buf.put(src)
        }
        src.limit(capacity)
        buf.rewind()
        val mpImage = ByteBufferImageBuilder(buf, width, height, MPImage.IMAGE_FORMAT_RGBA).build()
        val ts = maxOf(SystemClock.uptimeMillis(), lastTimestampMs + 1)
        lastTimestampMs = ts
        synchronized(pendingStreamFrames) {
            pendingStreamFrames.addLast(
                PendingStreamFrame(ts, frameTimestampNs, rotation, width, height, buf))
        }
        try {
            lm.detectAsync(mpImage, processingOptions(rotation), ts)
        } catch (e: RuntimeException) {
            synchronized(pendingStreamFrames) {
                pendingStreamFrames.removeLastOrNull()?.let { recycleStreamBuffer(it.buffer) }
            }
            throw e
        }
    }

    /** MediaPipe 콜백 스레드에서 호출 — 대기열에서 tsMs 매칭 후 동기 경로와 동일 처리. */
    private fun handleStreamResult(result: FaceLandmarkerResult, input: MPImage) {
        if (closed) return
        var matched: PendingStreamFrame? = null
        synchronized(pendingStreamFrames) {
            while (pendingStreamFrames.isNotEmpty()) {
                val p = pendingStreamFrames.removeFirst()
                if (p.tsMs == result.timestampMs()) {
                    matched = p
                    break
                }
                recycleStreamBuffer(p.buffer) // MediaPipe flow limiter가 흘려보낸 프레임
            }
        }
        val p = matched ?: return
        try {
            onResult(result, input, p.frameNs, p.tsMs)
            try {
                onRawResult?.invoke(result, p.rotation, p.width, p.height, p.buffer,
                    p.width * 4, p.frameNs)
            } catch (e: RuntimeException) {
                onError("onRawResult 소비자 오류: ${e.message}")
            }
        } finally {
            recycleStreamBuffer(p.buffer)
        }
    }

    private fun obtainStreamBuffer(size: Int): ByteBuffer? {
        synchronized(pendingStreamFrames) {
            while (streamBufferPool.isNotEmpty()) {
                val b = streamBufferPool.removeFirst()
                if (b.capacity() >= size) return b
                streamBuffersAllocated-- // 해상도 변경 등 크기 불일치 — 폐기
            }
            if (streamBuffersAllocated >= MAX_STREAM_BUFFERS) return null
            streamBuffersAllocated++
        }
        return ByteBuffer.allocateDirect(size)
    }

    private fun recycleStreamBuffer(b: ByteBuffer) {
        synchronized(pendingStreamFrames) { streamBufferPool.addLast(b) }
    }

    /** 모드 전환/종료 시 스트림 상태 초기화 — 인플라이트 결과는 closed/재생성으로 무시된다. */
    private fun resetStreamState() {
        synchronized(pendingStreamFrames) {
            pendingStreamFrames.clear()
            streamBufferPool.clear()
            streamBuffersAllocated = 0
        }
    }

    /**
     * 뷰티 활성 여부 설정 — 어느 스레드에서나 안전.
     * OFF면 onResult가 faceOval/brows/lipsOuter 추출·필터링을 건너뛴다 (추가 비용 0).
     */
    fun setBeautyEnabled(enabled: Boolean) {
        if (enabled && !beautyEnabled) {
            // false→true 전환: 오래된 필터 상태로 인한 턱선 글라이드 방지
            for (f in ovalFilters) f.reset()
        }
        beautyEnabled = enabled
    }

    /** 분석 executor 스레드 전용 — landmarker를 생성한 스레드에서 닫는다. */
    fun close() {
        closed = true
        try {
            landmarker?.close()
        } catch (_: RuntimeException) {
            // 종료 경로 — 무시
        }
        landmarker = null
        resetStreamState()
    }

    // ───────────────────────── 내부 ─────────────────────────

    private fun ensureLandmarker() {
        if (cpuFallbackRequested && usingGpu && landmarker != null) {
            // GPU 추론 런타임 오류 → CPU 재생성
            try {
                landmarker?.close()
            } catch (_: RuntimeException) {
            }
            landmarker = null
            cpuForced = true
            cpuFallbackRequested = false
            resetFilters()
        }
        // 실행 모드/numFaces 전환 — 옵션은 생성 시점 고정이라 재생성 필수 (핸드오프 §3-4).
        // 필터 리셋 동반 (모드 간 시간 특성이 달라 오래된 필터 상태가 글라이드 유발).
        if (landmarker != null && (runningModeRequested != runningModeActive ||
                singleFaceRequested != singleFaceActive)) {
            try {
                landmarker?.close()
            } catch (_: RuntimeException) {
            }
            landmarker = null
            creationFailed = false
            resetFilters()
            resetStreamState()
            hadFace = false
        }
        if (landmarker != null || creationFailed) return

        val wantGpu = preferGpu && !cpuForced && !EmulatorDetector.isEmulator
        runningModeActive = runningModeRequested
        singleFaceActive = singleFaceRequested
        landmarker = try {
            createLandmarker(wantGpu).also { usingGpu = wantGpu }
        } catch (e: RuntimeException) {
            if (wantGpu) {
                onError("GPU delegate 초기화 실패 — CPU로 폴백합니다: ${e.message}")
                cpuForced = true
                try {
                    createLandmarker(false).also { usingGpu = false }
                } catch (e2: RuntimeException) {
                    creationFailed = true
                    onError("FaceLandmarker 생성 실패 (CPU): ${e2.message}")
                    null
                }
            } else {
                creationFailed = true
                onError("FaceLandmarker 생성 실패: ${e.message}")
                null
            }
        }
        if (landmarker != null) {
            lastTimestampMs = 0L
            detectErrorReported = false
        }
    }

    private fun createLandmarker(useGpu: Boolean): FaceLandmarker {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath(MODEL_ASSET_PATH)
            .setDelegate(if (useGpu) Delegate.GPU else Delegate.CPU)
            .build()
        val builder = FaceLandmarker.FaceLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            // 실행 모드 3종 (벤치 토글 — setImageMode / setTrackingRunningMode):
            // IMAGE = 내부 스무딩·ROI 추적 없는 동기, LIVE_STREAM = 비동기(내부 큐잉)
            .setRunningMode(when (runningModeActive) {
                MODE_IMAGE -> RunningMode.IMAGE
                MODE_STREAM -> RunningMode.LIVE_STREAM
                else -> RunningMode.VIDEO
            })
            // ③-3: numFaces=2로 MediaPipe face_landmarker 그래프의 내부
            // LandmarksSmoothingCalculator(One-Euro 0.05/80) 노드를 우회한다 —
            // 이 노드는 num_faces==1 + 스트림 모드(VIDEO)에서만 활성화되며
            // (face_landmarker_graph.cc), 활성 시 코어 stabilize와 직렬 이중 필터가
            // 되어 추적 위상 지연이 누적된다(§5 주의 3 '이중 필터 금지' 위반).
            // 소비자는 전경(최대) 얼굴 1개만 사용하므로 2번째 슬롯은 미사용.
            // singleFaceActive(벤치 임시)면 1로 — 내부 스무딩 ON 팔 체감용 (setSingleFaceMode).
            .setNumFaces(if (singleFaceActive) 1 else 2)
            .setMinFaceDetectionConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .setMinFacePresenceConfidence(0.5f)
            // 렌즈 시착에 불필요 — 지연시간 절약
            .setOutputFaceBlendshapes(false)
            .setOutputFacialTransformationMatrixes(false)
        if (runningModeActive == MODE_STREAM) {
            // LIVE_STREAM은 결과/오류 리스너 필수 — 결과는 MediaPipe 콜백 스레드로 도착
            builder.setResultListener { result, input -> handleStreamResult(result, input) }
            builder.setErrorListener { e -> onError("LIVE_STREAM 추론 오류: ${e.message}") }
        }
        return FaceLandmarker.createFromOptions(context, builder.build())
    }

    private fun processingOptions(rotation: Int): ImageProcessingOptions {
        if (rotation != cachedRotation || cachedProcessingOptions == null) {
            cachedRotation = rotation
            cachedProcessingOptions = ImageProcessingOptions.builder()
                .setRotationDegrees(rotation)
                .build()
        }
        return cachedProcessingOptions!!
    }

    private fun obtainCompactBuffer(size: Int): ByteBuffer {
        val existing = compactBuffer
        if (existing != null && existing.capacity() >= size) return existing
        return ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder()).also { compactBuffer = it }
    }

    private fun handleDetectError(e: RuntimeException) {
        if (usingGpu && !cpuFallbackRequested) {
            cpuFallbackRequested = true
            onError("GPU 추론 오류 — CPU로 폴백합니다: ${e.message}")
        } else if (!detectErrorReported) {
            detectErrorReported = true
            onError("얼굴 추론 오류: ${e.message}")
        }
    }

    /**
     * 분석 스레드에서 detect(ForVideo) 직후 동기 호출되는 결과 처리.
     * [submitTs] = detect 제출 시각(uptime ms) — IMAGE 모드는 result.timestampMs()가
     * 무의미하므로 모든 시간 소비(추론 계측·필터 tSec·스냅샷 ts)를 이 값으로 통일한다.
     * (VIDEO 모드에선 detectForVideo가 제출 ts를 그대로 반환하므로 동작 불변.)
     */
    private fun onResult(result: FaceLandmarkerResult, input: MPImage, frameTimestampNs: Long, submitTs: Long) {
        if (closed) return

        // 추론 시간 = detect 제출 → 결과 처리 시점
        val inferenceMs = (SystemClock.uptimeMillis() - submitTs).toFloat()
        onInferenceStats(inferenceMs, usingGpu)

        // 주의: ImageProcessingOptions.rotationDegrees를 줘도 MediaPipe는 출력 랜드마크를
        // 원본(미회전) 입력 좌표계로 재투영해 반환한다 — upright 변환은 우리가 수행
        // (CoordMapper.sensorToUpright, docs/tech-validation.md 함정 #13).
        val rotation = lastRotationDegrees
        val swap = rotation == 90 || rotation == 270
        val upW = (if (swap) input.height else input.width).toFloat()
        val upH = (if (swap) input.width else input.height).toFloat()

        val faces = result.faceLandmarks()
        if (faces.isEmpty() || faces[0].size < LandmarkIndices.LANDMARK_COUNT) {
            if (hadFace) resetFilters() // 재획득 시 글라이드 방지
            hadFace = false
            rawIrisCacheValid = false // 직전 랜드마크 없음 — 휘도 측정 생략 (EMA 값은 유지)
            onSnapshot(TrackingSnapshot.noFace(submitTs, frameTimestampNs))
            return
        }
        hadFace = true

        val lm = faces[0]
        val tSec = submitTs / 1000.0

        // 다음 analyze의 휘도 샘플용 원시(센서) 홍채 캐시 — upright/미러 변환 전 좌표
        cacheRawIris(lm, input.width.toFloat(), input.height.toFloat())

        // 디버그 오버레이용 무필터 raw 홍채 중심 (스냅샷마다 새 배열 — 렌더 스레드 소비 중 재사용 금지)
        val rawIris = FloatArray(4)
        val rightEye = computeEye(
            lm, tSec, upW, upH, rotation, filterBase = 0,
            centerIdx = LandmarkIndices.RIGHT_IRIS_CENTER,
            rightIdx = LandmarkIndices.RIGHT_IRIS_RIGHT,
            topIdx = LandmarkIndices.RIGHT_IRIS_TOP,
            rawOut = rawIris, rawOffset = 0,
        )
        val leftEye = computeEye(
            lm, tSec, upW, upH, rotation, filterBase = 6,
            centerIdx = LandmarkIndices.LEFT_IRIS_CENTER,
            rightIdx = LandmarkIndices.LEFT_IRIS_RIGHT,
            topIdx = LandmarkIndices.LEFT_IRIS_TOP,
            rawOut = rawIris, rawOffset = 2,
        )
        val rightContour = extractContour(lm, LandmarkIndices.RIGHT_EYE_CONTOUR, rotation)
        val leftContour = extractContour(lm, LandmarkIndices.LEFT_EYE_CONTOUR, rotation)
        // 뷰티 OFF면 뷰티 랜드마크 추출·필터링 생략 — 추가 비용 0 (렌더러는 null 안전)
        val beauty = beautyEnabled
        val faceOval = if (beauty) extractOvalFiltered(lm, rotation, tSec, upW, upH) else null
        val brows = if (beauty) extractBrows(lm, rotation) else null
        val lipsOuter = if (beauty) extractContour(lm, LandmarkIndices.LIPS_OUTER, rotation) else null

        onSnapshot(
            TrackingSnapshot.face(
                timestampMs = submitTs,
                frameTimestampNs = frameTimestampNs,
                srcAspect = upW / upH,
                rightEye = rightEye,
                leftEye = leftEye,
                rightContour = rightContour,
                leftContour = leftContour,
                faceOval = faceOval,
                brows = brows,
                lipsOuter = lipsOuter,
                // 네이티브 내부 전용 스칼라 — 채널/콜백으로 노출 금지 (CLAUDE.md 불변 조건)
                avgIrisLuma = irisLumaEma,
                rawIris = rawIris,
            ),
        )
    }

    /**
     * 분석 버퍼(센서 좌표계)에서 직전 프레임 홍채 디스크(반경×0.6)의 8×8 그리드 샘플로
     * Rec.601 평균 휘도를 측정하고 EMA(α=0.2)로 스무딩한다. 절대 인덱스 get만 사용해
     * 버퍼 position/limit을 건드리지 않는다. 프레임당 힙 할당 0 (배열 재사용).
     */
    private fun sampleIrisLuma(src: ByteBuffer, width: Int, height: Int, rowStride: Int) {
        if (!rawIrisCacheValid) return
        val limit = src.limit()
        var sum = 0f
        var count = 0
        for (eye in 0 until 2) {
            val base = eye * 3
            val rN = rawIrisCache[base + 2]
            if (rN <= 0f) continue
            val n = IrisLumaSampler.computeSampleOffsets(
                cxPx = rawIrisCache[base] * width,
                cyPx = rawIrisCache[base + 1] * height,
                radiusPx = rN * width * LUMA_DISC_SCALE,
                width = width,
                height = height,
                rowStride = rowStride,
                outOffsets = lumaSampleOffsets,
            )
            for (i in 0 until n) {
                val off = lumaSampleOffsets[i]
                if (off < 0 || off + 2 >= limit) continue // 버퍼 경계 방어
                val r = src.get(off).toInt() and 0xFF
                val g = src.get(off + 1).toInt() and 0xFF
                val b = src.get(off + 2).toInt() and 0xFF
                sum += 0.299f * r + 0.587f * g + 0.114f * b
                count++
            }
        }
        if (count > 0) {
            val mean = sum / (count * 255f)
            irisLumaEma += LUMA_EMA_ALPHA * (mean - irisLumaEma)
        }
    }

    /** 원시(센서) 홍채 중심·반경 캐시 — 다음 analyze가 같은 좌표계 버퍼에서 샘플한다. */
    private fun cacheRawIris(lm: List<NormalizedLandmark>, srcW: Float, srcH: Float) {
        cacheRawEye(lm, LandmarkIndices.RIGHT_IRIS_CENTER, LandmarkIndices.RIGHT_IRIS_RIGHT, srcW, srcH, 0)
        cacheRawEye(lm, LandmarkIndices.LEFT_IRIS_CENTER, LandmarkIndices.LEFT_IRIS_RIGHT, srcW, srcH, 3)
        rawIrisCacheValid = true
    }

    private fun cacheRawEye(
        lm: List<NormalizedLandmark>,
        centerIdx: Int,
        rightIdx: Int,
        srcW: Float,
        srcH: Float,
        base: Int,
    ) {
        val cx = lm[centerIdx].x()
        val cy = lm[centerIdx].y()
        // 반경은 픽셀 공간에서 계산 후 폭으로 정규화 (정규화 좌표 직접 계산 금지 — 종횡비 왜곡)
        val ux = (lm[rightIdx].x() - cx) * srcW
        val uy = (lm[rightIdx].y() - cy) * srcH
        rawIrisCache[base] = cx
        rawIrisCache[base + 1] = cy
        rawIrisCache[base + 2] = IrisGeometry.radiusPx(ux, uy) / srcW
    }

    /**
     * 홍채 중심·기저벡터 계산 + One-Euro 필터.
     * 픽셀 공간에서 계산·필터링 후 정규화 좌표로 되돌려 저장한다 (해상도 독립 전달).
     */
    private fun computeEye(
        lm: List<NormalizedLandmark>,
        tSec: Double,
        upW: Float,
        upH: Float,
        rotation: Int,
        filterBase: Int,
        centerIdx: Int,
        rightIdx: Int,
        topIdx: Int,
        rawOut: FloatArray,
        rawOffset: Int,
    ): FloatArray {
        toDisplaySpace(lm[centerIdx].x(), lm[centerIdx].y(), rotation, ptCenter)
        toDisplaySpace(lm[rightIdx].x(), lm[rightIdx].y(), rotation, ptRight)
        toDisplaySpace(lm[topIdx].x(), lm[topIdx].y(), rotation, ptTop)
        // 디버그 오버레이용 — 단일 중심 랜드마크(468/473) 무가공 좌표
        rawOut[rawOffset] = ptCenter[0]
        rawOut[rawOffset + 1] = ptCenter[1]
        // 중심은 단일 랜드마크(468/473) 사용 — 5점 평균 실험은 기각 (2026-06-12 실기기):
        // 경계 4점(특히 상/하 — 눈꺼풀 간섭)의 노이즈가 중심점보다 크고 상관되어
        // 평균이 오히려 출렁임을 키웠다. MediaPipe가 직접 회귀하는 중심점이 가장 안정적.
        IrisGeometry.computeBasisPx(
            cx = ptCenter[0], cy = ptCenter[1],
            rx = ptRight[0], ry = ptRight[1],
            tx = ptTop[0], ty = ptTop[1],
            widthPx = upW, heightPx = upH,
            out = basisTmp,
        )
        val result = FloatArray(6)
        for (i in 0 until 6) {
            val filtered = filters[filterBase + i].filter(basisTmp[i], tSec)
            result[i] = filtered / (if (i % 2 == 0) upW else upH)
        }
        return result
    }

    private fun extractContour(
        lm: List<NormalizedLandmark>,
        indices: IntArray,
        rotation: Int,
    ): FloatArray {
        val out = FloatArray(indices.size * 2)
        fillPoints(lm, indices, rotation, out, outOffset = 0)
        return out
    }

    /**
     * FACE_OVAL 36점 — 디스플레이 변환 후 전용 72채널 One-Euro 필터 적용.
     * 워프 제어점이 흔들리면 턱선이 출렁이므로 눈과 동일하게 픽셀 공간에서 스무딩한다.
     */
    private fun extractOvalFiltered(
        lm: List<NormalizedLandmark>,
        rotation: Int,
        tSec: Double,
        upW: Float,
        upH: Float,
    ): FloatArray {
        val indices = LandmarkIndices.FACE_OVAL
        val out = FloatArray(indices.size * 2)
        for (i in indices.indices) {
            toDisplaySpace(lm[indices[i]].x(), lm[indices[i]].y(), rotation, ptTmp)
            // 픽셀 공간 필터링 후 정규화로 복귀 (정규화 공간 직접 필터링은 종횡비 왜곡)
            out[i * 2] = ovalFilters[i * 2].filter(ptTmp[0] * upW, tSec) / upW
            out[i * 2 + 1] = ovalFilters[i * 2 + 1].filter(ptTmp[1] * upH, tSec) / upH
        }
        return out
    }

    /** 눈썹 2개를 한 배열로 — 우안측 10점 + 좌안측 10점 (40 floats). 필터 불필요 (마스크 페더링). */
    private fun extractBrows(lm: List<NormalizedLandmark>, rotation: Int): FloatArray {
        val out = FloatArray(40)
        fillPoints(lm, LandmarkIndices.RIGHT_BROW, rotation, out, outOffset = 0)
        fillPoints(lm, LandmarkIndices.LEFT_BROW, rotation, out, outOffset = 20)
        return out
    }

    private fun fillPoints(
        lm: List<NormalizedLandmark>,
        indices: IntArray,
        rotation: Int,
        out: FloatArray,
        outOffset: Int,
    ) {
        for (i in indices.indices) {
            toDisplaySpace(lm[indices[i]].x(), lm[indices[i]].y(), rotation, ptTmp)
            out[outOffset + i * 2] = ptTmp[0]
            out[outOffset + i * 2 + 1] = ptTmp[1]
        }
    }

    /**
     * 원본 입력(센서) 좌표 → 렌더 표시 좌표.
     * ① sensorToUpright 회전 보정 ② 전면 카메라면 x를 1-x로 미러 (upright 공간 기준).
     */
    private fun toDisplaySpace(x: Float, y: Float, rotation: Int, out: FloatArray) {
        CoordMapper.sensorToUpright(x, y, rotation, out)
        if (mirror) out[0] = 1f - out[0]
    }

    private fun resetFilters() {
        for (f in filters) f.reset()
        for (f in ovalFilters) f.reset()
    }

    companion object {
        /**
         * demo-app 내장 모델 — assets/models/ (REFACTOR-3-3 plan §2 대상 경로).
         * [이식 적응] 원본(LensSimulator)은 "lenssdk/face_landmarker.task" — 경로만 demo assets에 맞춤.
         */
        private const val MODEL_ASSET_PATH = "models/face_landmarker.task"

        // 실행 모드 (벤치 토글 — setTrackingRunningMode)
        const val MODE_VIDEO = 0
        const val MODE_IMAGE = 1
        const val MODE_STREAM = 2

        /** LIVE_STREAM 인플라이트 복사 버퍼 상한 — 초과 제출은 드롭(백프레셔) */
        private const val MAX_STREAM_BUFFERS = 4

        // One-Euro 시작값 (ADR-0002). beta는 픽셀 공간 운동 지연 재튜닝 값 —
        // 0.007은 빠른 머리 움직임에도 컷오프가 ~2Hz에 머물러 렌즈가 3~4프레임 끌림
        // (치환형 통색 렌더에서 가시화, 2026-06-12 실기기). min_cutoff는 정지 지터 기준 유지.
        // 분업: 정지 지터는 min_cutoff(낮게), 움직임 추종은 beta(높게) + 프레임 동기 렌더링.
        // min_cutoff 1.0 실험은 홍채 세로 지터(눈꺼풀 간섭)가 통과해 출렁임 — 0.5로 환원 (2026-06-12).
        // 프레임 동기 렌더링 하에서는 필터 지연이 곧 화면 어긋남이다 (화면=랜드마크 프레임).
        // 따라서 near-raw 튜닝: 1프레임 스파이크만 걸러내고 raw(흰색 디버그 점)에 수렴시킨다.
        private const val MIN_CUTOFF = 3.0f
        private const val BETA = 0.3f
        private const val D_CUTOFF = 1.0f
        // 주의: 속도 예측(lead) 방식은 2026-06-12 실기기에서 기각 — 1Hz 저역 통과 속도가
        // 방향 전환 시 과거 방향으로 좌표를 밀어 오히려 크게 틀어진다 (기저벡터 변형 포함).

        // 홍채 휘도 측정 (ADR-0004) — 디스크 반경 비율 / EMA 계수 / 측정 전 기본값
        private const val LUMA_DISC_SCALE = 0.6f
        private const val LUMA_EMA_ALPHA = 0.2f
        private const val DEFAULT_IRIS_LUMA = 0.5f
    }
}
