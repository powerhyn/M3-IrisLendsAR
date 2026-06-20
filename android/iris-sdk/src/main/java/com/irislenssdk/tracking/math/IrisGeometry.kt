package com.irislenssdk.tracking.math

import kotlin.math.sqrt

/**
 * 홍채 기저벡터 계산 순수 함수.
 *
 * 렌즈 쿼드 변형 규약 (ADR-0002):
 * - u = p_right − p_center, v = p_top − p_center
 * - 고개 회전 시 타원화가 기저벡터에 자동 반영된다 (호모그래피 불필요)
 * - 거리·반경 계산은 반드시 픽셀 좌표 변환 후 수행 (정규화 좌표 직접 계산 금지 — 종횡비 왜곡)
 */
internal object IrisGeometry {

    /**
     * 정규화 랜드마크 → 픽셀 변환 후 홍채 중심·기저벡터를 계산한다.
     *
     * @param cx,cy 홍채 중심 정규화 좌표 (468/473)
     * @param rx,ry 홍채 right edge 정규화 좌표 (469/474 — 이미지 좌표 기준)
     * @param tx,ty 홍채 top edge 정규화 좌표 (470/475)
     * @param widthPx,heightPx upright 프레임 픽셀 크기
     * @param out [cxPx, cyPx, ux, uy, vx, vy] (픽셀 단위)
     */
    fun computeBasisPx(
        cx: Float, cy: Float,
        rx: Float, ry: Float,
        tx: Float, ty: Float,
        widthPx: Float, heightPx: Float,
        out: FloatArray,
    ) {
        val cpx = cx * widthPx
        val cpy = cy * heightPx
        out[0] = cpx
        out[1] = cpy
        out[2] = rx * widthPx - cpx   // u.x
        out[3] = ry * heightPx - cpy  // u.y
        out[4] = tx * widthPx - cpx   // v.x
        out[5] = ty * heightPx - cpy  // v.y
    }

    /** 기저벡터 길이 = 홍채 반경 (픽셀). */
    fun radiusPx(ux: Float, uy: Float): Float = sqrt(ux * ux + uy * uy)
}
