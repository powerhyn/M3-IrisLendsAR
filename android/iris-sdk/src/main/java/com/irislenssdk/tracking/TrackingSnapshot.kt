package com.irislenssdk.tracking

/**
 * 추론 1회 결과의 불변 스냅샷 — 추론 스레드에서 생성, 렌더 스레드에서 소비.
 *
 * 좌표는 모두 **upright 분석 프레임 정규화 좌표** (전면 카메라면 미러 반영 완료).
 * 렌더 스레드가 뷰포트 픽셀로 변환한다 (CoordMapper 참조).
 *
 * 이 타입은 SDK 내부 전용이다 — 랜드마크를 SDK 밖으로 노출하지 않는다 (CLAUDE.md 불변 조건).
 */
class TrackingSnapshot private constructor(
    val detected: Boolean,
    val timestampMs: Long,
    /**
     * 분석 프레임의 카메라 센서 타임스탬프 (ns, `ImageInfo.timestamp`) — 프레임 동기 렌더링용.
     * [timestampMs](분석 스레드 uptime)와 클럭 도메인이 다르다. 0 이하 = 미상 (최신 프레임 폴백).
     */
    val frameTimestampNs: Long,
    /** upright 분석 프레임 종횡비 W/H — 프리뷰와 종횡비가 다를 때 방어 매핑에 사용 */
    val srcAspect: Float,
    /** [cx, cy, ux, uy, vx, vy] — One-Euro 필터 적용 완료. null = 해당 눈 미검출 */
    val rightEye: FloatArray?,
    val leftEye: FloatArray?,
    /** 눈 윤곽 16점 [x0,y0, x1,y1, ...] (32 floats) — 오클루전 마스크용 */
    val rightContour: FloatArray?,
    val leftContour: FloatArray?,
    /** FACE_OVAL 36점 (72 floats) — One-Euro 적용 완료. 뷰티(턱선 워프 + 피부 마스크)용 */
    val faceOval: FloatArray?,
    /** 눈썹 폴리곤 — 우안측 10점 + 좌안측 10점 (40 floats). 필터 없음 (마스크가 페더링됨) */
    val brows: FloatArray?,
    /** 입술 외곽 링 20점 (40 floats). 필터 없음 (마스크가 페더링됨) */
    val lipsOuter: FloatArray?,
    /**
     * 홍채 ROI 평균 휘도 (Rec.601, 0..1, EMA 스무딩 완료) — 치환형 블렌딩의 노출 매칭용
     * (ADR-0004). 측정 전 기본값 0.5 (셰이더 항등 기준). SDK 밖으로 노출 금지.
     */
    val avgIrisLuma: Float,
    /** 무필터 raw 홍채 중심 [rx, ry, lx, ly] — 디버그 메시 오버레이 전용 (진단 도구) */
    val rawIris: FloatArray?,
) {
    val hasEyes: Boolean
        get() = rightEye != null || leftEye != null

    companion object {
        fun face(
            timestampMs: Long,
            frameTimestampNs: Long,
            srcAspect: Float,
            rightEye: FloatArray,
            leftEye: FloatArray,
            rightContour: FloatArray,
            leftContour: FloatArray,
            // 뷰티 게이팅 OFF면 추출이 생략되어 null로 전달된다 (렌더러는 null 안전)
            faceOval: FloatArray?,
            brows: FloatArray?,
            lipsOuter: FloatArray?,
            avgIrisLuma: Float = 0.5f,
            rawIris: FloatArray? = null,
        ): TrackingSnapshot = TrackingSnapshot(
            detected = true,
            timestampMs = timestampMs,
            frameTimestampNs = frameTimestampNs,
            srcAspect = srcAspect,
            rightEye = rightEye,
            leftEye = leftEye,
            rightContour = rightContour,
            leftContour = leftContour,
            faceOval = faceOval,
            brows = brows,
            lipsOuter = lipsOuter,
            avgIrisLuma = avgIrisLuma,
            rawIris = rawIris,
        )

        fun noFace(timestampMs: Long, frameTimestampNs: Long): TrackingSnapshot = TrackingSnapshot(
            detected = false,
            timestampMs = timestampMs,
            frameTimestampNs = frameTimestampNs,
            srcAspect = 0f,
            rightEye = null,
            leftEye = null,
            rightContour = null,
            leftContour = null,
            faceOval = null,
            brows = null,
            lipsOuter = null,
            avgIrisLuma = 0.5f,
            rawIris = null,
        )
    }
}
