package com.irislenssdk.tracking.math

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

/** 홍채 기저벡터 계산 검증 — 픽셀 좌표 변환 후 계산 원칙 확인. */
class IrisGeometryTest {

    private val eps = 1e-4f

    @Test
    fun `픽셀 공간의 원형 홍채는 기저벡터 길이가 같다`() {
        // 640x480 프레임에서 반경 50px 원형 홍채
        val w = 640f
        val h = 480f
        val cxPx = 320f
        val cyPx = 240f
        val r = 50f
        val out = FloatArray(6)
        IrisGeometry.computeBasisPx(
            cx = cxPx / w, cy = cyPx / h,
            rx = (cxPx + r) / w, ry = cyPx / h,    // right edge
            tx = cxPx / w, ty = (cyPx - r) / h,    // top edge (이미지 좌표 — 위는 -y)
            widthPx = w, heightPx = h,
            out = out,
        )
        assertEquals(320f, out[0], eps)
        assertEquals(240f, out[1], eps)
        // u = (50, 0), v = (0, -50)
        assertEquals(50f, out[2], eps)
        assertEquals(0f, out[3], eps)
        assertEquals(0f, out[4], eps)
        assertEquals(-50f, out[5], eps)
        // 픽셀 공간에서 |u| == |v| (정규화 공간에서 직접 계산하면 종횡비 왜곡으로 불일치)
        val uLen = IrisGeometry.radiusPx(out[2], out[3])
        val vLen = IrisGeometry.radiusPx(out[4], out[5])
        assertEquals(uLen, vLen, eps)
    }

    @Test
    fun `정규화 공간 직접 계산은 종횡비 왜곡을 일으킨다 - 픽셀 변환의 근거`() {
        // 같은 원형 홍채를 정규화 공간에서 직접 계산하면 가로/세로 길이가 4:3으로 어긋난다
        val w = 640f
        val h = 480f
        val r = 50f
        val uNorm = r / w  // 0.078125
        val vNorm = r / h  // 0.104167
        val distortion = vNorm / uNorm
        assertTrue("정규화 공간 왜곡 비율은 4/3", abs(distortion - 4f / 3f) < 1e-4f)
    }

    @Test
    fun `기울어진 홍채의 기저벡터는 회전을 보존한다`() {
        // 45도 기울어진 홍채 (고개 롤 회전 시뮬레이션)
        val w = 1000f
        val h = 1000f // 정사각 프레임으로 회전만 검증
        val cx = 0.5f
        val cy = 0.5f
        val r = 0.1f
        val cos45 = 0.70710678f
        val out = FloatArray(6)
        IrisGeometry.computeBasisPx(
            cx = cx, cy = cy,
            rx = cx + r * cos45, ry = cy + r * cos45,
            tx = cx + r * cos45, ty = cy - r * cos45,
            widthPx = w, heightPx = h,
            out = out,
        )
        // u와 v는 직교해야 한다 (내적 = 0)
        val dot = out[2] * out[4] + out[3] * out[5]
        assertEquals(0f, dot, 1e-2f)
        // 길이는 100px
        assertEquals(100f, IrisGeometry.radiusPx(out[2], out[3]), 1e-2f)
    }

    @Test
    fun `radiusPx는 유클리드 길이를 반환한다`() {
        assertEquals(5f, IrisGeometry.radiusPx(3f, 4f), eps)
        assertEquals(0f, IrisGeometry.radiusPx(0f, 0f), eps)
    }
}
