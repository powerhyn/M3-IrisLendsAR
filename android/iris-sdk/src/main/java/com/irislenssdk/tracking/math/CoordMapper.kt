package com.irislenssdk.tracking.math

/**
 * 좌표 매핑 순수 함수 모음 — 분석 프레임(정규화) ↔ 렌더 뷰포트 변환.
 *
 * 좌표계 규약:
 * - "upright": 회전 보정 후(머리가 위) 이미지 공간. u,v ∈ [0,1], v는 아래 방향.
 * - "센서": 카메라 버퍼 원본 방향. rotationDegrees(시계방향)만큼 돌리면 upright가 된다.
 * - "디스플레이": 출력 뷰포트에 보이는 영역의 정규화 좌표 (aspect-fill 크롭 반영).
 *
 * 전부 Android 의존성 없는 순수 함수 — 단위 테스트 대상.
 * 할당 회피를 위해 결과는 호출자가 준 FloatArray에 기록한다.
 */
internal object CoordMapper {

    /**
     * aspect-fill(센터 크롭) 시 소스에서 실제로 보이는 부분 사각형.
     *
     * @param srcAspect 소스(upright 카메라) 종횡비 W/H
     * @param dstAspect 출력 뷰포트 종횡비 W/H
     * @param out [u0, v0, width, height] — 소스 정규화 좌표 기준
     */
    fun visibleSubRect(srcAspect: Float, dstAspect: Float, out: FloatArray) {
        if (srcAspect <= 0f || dstAspect <= 0f) {
            out[0] = 0f; out[1] = 0f; out[2] = 1f; out[3] = 1f
            return
        }
        if (srcAspect > dstAspect) {
            // 소스가 더 넓음 → 좌우 크롭
            val w = dstAspect / srcAspect
            out[0] = (1f - w) * 0.5f; out[1] = 0f; out[2] = w; out[3] = 1f
        } else {
            // 소스가 더 좁거나 같음 → 상하 크롭
            val h = srcAspect / dstAspect
            out[0] = 0f; out[1] = (1f - h) * 0.5f; out[2] = 1f; out[3] = h
        }
    }

    /**
     * upright 정규화 좌표 → 센서 정규화 좌표.
     *
     * @param rotationDegrees 센서 버퍼를 시계방향으로 이만큼 돌리면 upright가 되는 각도 (0/90/180/270)
     * @param out [uSensor, vSensor]
     */
    fun uprightToSensor(u: Float, v: Float, rotationDegrees: Int, out: FloatArray) {
        when (((rotationDegrees % 360) + 360) % 360) {
            90 -> { out[0] = v; out[1] = 1f - u }
            180 -> { out[0] = 1f - u; out[1] = 1f - v }
            270 -> { out[0] = 1f - v; out[1] = u }
            else -> { out[0] = u; out[1] = v }
        }
    }

    /**
     * 디스플레이 정규화 좌표 → 센서 GL 텍스처 좌표 (SurfaceTexture transform matrix 적용 전).
     *
     * 체인: 크롭 서브렉트 → (전면이면) 미러 → upright→센서 역회전 → GL t축 플립.
     * 카메라 패스와 렌즈 패스가 같은 체인을 공유해야 픽셀 정합이 보장된다.
     *
     * @param crop [u0, v0, w, h] — [visibleSubRect] 결과
     * @param out [s, t] GL 관례 좌표 (t는 위 방향) — 셰이더에서 ST 행렬을 곱해 사용
     */
    fun displayUvToSensorGl(
        u: Float,
        v: Float,
        crop: FloatArray,
        mirror: Boolean,
        rotationDegrees: Int,
        out: FloatArray,
    ) {
        var cu = crop[0] + u * crop[2]
        val cv = crop[1] + v * crop[3]
        if (mirror) cu = 1f - cu
        uprightToSensor(cu, cv, rotationDegrees, out)
        // 이미지 관례(v 아래 방향) → GL 관례(t 위 방향)
        out[1] = 1f - out[1]
    }

    /**
     * 센서 정규화 좌표 → upright 정규화 좌표 ([uprightToSensor]의 역변환).
     *
     * MediaPipe Tasks는 ImageProcessingOptions.rotationDegrees를 줘도 출력
     * 랜드마크를 **원본(미회전) 입력 좌표계로 재투영**해 반환하므로(LandmarkProjection),
     * 렌더 좌표계(upright)와 일치시키려면 이 변환을 거쳐야 한다.
     *
     * @param rotationDegrees 센서 버퍼를 시계방향으로 이만큼 돌리면 upright가 되는 각도
     * @param out [uUpright, vUpright]
     */
    fun sensorToUpright(u: Float, v: Float, rotationDegrees: Int, out: FloatArray) {
        when (((rotationDegrees % 360) + 360) % 360) {
            90 -> { out[0] = 1f - v; out[1] = u }
            180 -> { out[0] = 1f - u; out[1] = 1f - v }
            270 -> { out[0] = v; out[1] = 1f - u }
            else -> { out[0] = u; out[1] = v }
        }
    }

    /**
     * 디스플레이 정규화 좌표 → 버퍼 GL 텍스처 좌표 (카메라 직결 Surface 경로).
     *
     * 카메라가 Surface에 직접 쓰는 경우(TransformationInfo.hasCameraTransform=true)
     * SurfaceTexture transform matrix가 센서 회전·전면 미러를 이미 포함한다
     * (CameraX SurfaceRequest 문서: "the app only needs to apply the Surface
     * transformation"). 따라서 여기서는 aspect-fill 크롭과 GL t축 플립만 적용한다 —
     * 회전·미러를 다시 적용하면 이중 변환으로 영상이 눕는다.
     *
     * @param crop [u0, v0, w, h] — upright 공간의 [visibleSubRect] 결과
     * @param out [s, t] GL 관례 좌표 — 셰이더에서 ST 행렬을 곱해 사용
     */
    fun displayUvToBufferGl(u: Float, v: Float, crop: FloatArray, out: FloatArray) {
        out[0] = crop[0] + u * crop[2]
        // 이미지 관례(v 아래 방향) → GL 관례(t 위 방향)
        out[1] = 1f - (crop[1] + v * crop[3])
    }

    /**
     * 종횡비가 다른 두 스트림(분석 vs 프리뷰) 간 정규화 좌표 변환 — letterbox/crop 차이 방어 코드.
     *
     * 두 스트림 모두 같은 센서에서 센터 크롭됐다고 가정한다 (CameraX 기본 동작).
     * 종횡비가 같으면 항등 변환 (4:3 강제 시 일반 경로).
     *
     * @param sensorAspect 센서 종횡비 가정값 (기본 4:3)
     * @param out [scaleX, offsetX, scaleY, offsetY] — dst = src * scale + offset
     */
    fun aspectAdapt(
        srcAspect: Float,
        dstAspect: Float,
        sensorAspect: Float = 4f / 3f,
        out: FloatArray,
    ) {
        // src → 센서: 종횡비가 센서보다 넓으면 가로 전체 유지·세로 크롭
        var sx = 1f
        var sy = 1f
        if (srcAspect >= sensorAspect) sy = sensorAspect / srcAspect else sx = srcAspect / sensorAspect
        // dst → 센서 (역변환으로 사용)
        var dx = 1f
        var dy = 1f
        if (dstAspect >= sensorAspect) dy = sensorAspect / dstAspect else dx = dstAspect / sensorAspect

        val scaleX = sx / dx
        val scaleY = sy / dy
        out[0] = scaleX
        out[1] = 0.5f - 0.5f * scaleX
        out[2] = scaleY
        out[3] = 0.5f - 0.5f * scaleY
    }
}
