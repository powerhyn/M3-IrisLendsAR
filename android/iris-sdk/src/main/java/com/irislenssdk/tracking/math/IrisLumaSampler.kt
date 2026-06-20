package com.irislenssdk.tracking.math

/**
 * 홍채 평균 휘도(avgIrisLuma) 측정용 샘플 좌표 계산 순수 함수 (ADR-0004 노출 매칭).
 *
 * 분석 RGBA 버퍼(센서 좌표계)에서 홍채 중심 디스크의 8×8 그리드 샘플 위치를
 * 바이트 오프셋으로 산출한다. 버퍼 읽기 자체는 호출자(FaceTracker)가 수행한다 —
 * 좌표 산출만 분리해 단위 테스트한다 (인덱스 범위 / 경계 클램프 / 디스크 내부성).
 */
internal object IrisLumaSampler {

    /** 그리드 한 변 샘플 수 — 디스크 내부 판정으로 일부 탈락해 실제 샘플은 ≤ 64 */
    const val GRID = 8

    /** [computeSampleOffsets]의 out 배열 최소 크기 */
    const val MAX_SAMPLES = GRID * GRID

    /**
     * 디스크(중심 [cxPx],[cyPx], 반경 [radiusPx]) 내 8×8 셀 중심 그리드 샘플의
     * 버퍼 바이트 오프셋(RGBA 픽셀 시작 위치)을 [outOffsets]에 채운다.
     *
     * - 디스크 밖 그리드 점은 제외한다 (원형 영역만 — 공막/눈꺼풀 오염 방지)
     * - 픽셀 좌표는 [0, width-1]/[0, height-1]로 클램프한다 (버퍼 경계 안전)
     * - 오프셋 = y × rowStride + x × 4 (RGBA_8888)
     *
     * @param radiusPx 샘플 디스크 반경 (호출자가 홍채 반경 × 0.6을 전달)
     * @return 유효 샘플 수 (0이면 측정 생략 — 퇴화 입력 포함)
     */
    fun computeSampleOffsets(
        cxPx: Float,
        cyPx: Float,
        radiusPx: Float,
        width: Int,
        height: Int,
        rowStride: Int,
        outOffsets: IntArray,
    ): Int {
        if (radiusPx <= 0f || width <= 0 || height <= 0 || rowStride < width * 4) return 0
        val r2 = radiusPx * radiusPx
        var count = 0
        for (j in 0 until GRID) {
            // 셀 중심 정규화 좌표 (-1..1): (j + 0.5) / GRID * 2 - 1
            val ny = ((j + 0.5f) / GRID) * 2f - 1f
            val dy = ny * radiusPx
            for (i in 0 until GRID) {
                val nx = ((i + 0.5f) / GRID) * 2f - 1f
                val dx = nx * radiusPx
                if (dx * dx + dy * dy > r2) continue // 디스크 밖 — 제외
                val x = (cxPx + dx).toInt().coerceIn(0, width - 1)
                val y = (cyPx + dy).toInt().coerceIn(0, height - 1)
                outOffsets[count++] = y * rowStride + x * 4
            }
        }
        return count
    }
}
