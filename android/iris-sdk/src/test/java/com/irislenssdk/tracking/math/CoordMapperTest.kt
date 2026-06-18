package com.irislenssdk.tracking.math

import org.junit.Assert.assertEquals
import org.junit.Test

/** 좌표 매핑 순수 함수 검증. */
class CoordMapperTest {

    private val eps = 1e-5f

    // ───────────── sensorToUpright (랜드마크 회전 보정) ─────────────

    @Test
    fun `sensorToUpright는 uprightToSensor의 역변환이다`() {
        val toSensor = FloatArray(2)
        val back = FloatArray(2)
        for (rotation in intArrayOf(0, 90, 180, 270)) {
            for ((u, v) in listOf(0.2f to 0.7f, 0f to 0f, 1f to 1f, 0.5f to 0.25f)) {
                CoordMapper.uprightToSensor(u, v, rotation, toSensor)
                CoordMapper.sensorToUpright(toSensor[0], toSensor[1], rotation, back)
                assertEquals("rotation=$rotation u", u, back[0], eps)
                assertEquals("rotation=$rotation v", v, back[1], eps)
            }
        }
    }

    @Test
    fun `회전 270 센서 좌표가 올바른 upright 위치로 간다`() {
        // 센서(가로) 우상단 부근 점 → 270° CW 회전 시 upright 우하단 부근
        val out = FloatArray(2)
        CoordMapper.sensorToUpright(0.9f, 0.1f, 270, out)
        assertEquals(0.1f, out[0], eps)
        assertEquals(1f - 0.9f, out[1], eps)
    }

    // ───────────── displayUvToBufferGl (카메라 직결 ST 경로) ─────────────

    @Test
    fun `직결 경로는 크롭 없이 t축 플립만 적용한다`() {
        val out = FloatArray(2)
        val full = floatArrayOf(0f, 0f, 1f, 1f)
        CoordMapper.displayUvToBufferGl(0f, 0f, full, out) // 화면 좌상단
        assertEquals(0f, out[0], eps)
        assertEquals(1f, out[1], eps) // GL t 위 방향
        CoordMapper.displayUvToBufferGl(1f, 1f, full, out) // 화면 우하단
        assertEquals(1f, out[0], eps)
        assertEquals(0f, out[1], eps)
    }

    @Test
    fun `직결 경로는 회전·미러 없이 크롭 서브렉트를 적용한다`() {
        val out = FloatArray(2)
        val crop = floatArrayOf(0.25f, 0f, 0.5f, 1f) // 좌우 크롭
        CoordMapper.displayUvToBufferGl(0f, 0.5f, crop, out)
        assertEquals(0.25f, out[0], eps)
        assertEquals(0.5f, out[1], eps)
        CoordMapper.displayUvToBufferGl(1f, 0.5f, crop, out)
        assertEquals(0.75f, out[0], eps)
        assertEquals(0.5f, out[1], eps)
    }

    // ───────────── visibleSubRect (aspect-fill 크롭) ─────────────

    @Test
    fun `종횡비가 같으면 전체 영역이 보인다`() {
        val out = FloatArray(4)
        CoordMapper.visibleSubRect(4f / 3f, 4f / 3f, out)
        assertEquals(0f, out[0], eps)
        assertEquals(0f, out[1], eps)
        assertEquals(1f, out[2], eps)
        assertEquals(1f, out[3], eps)
    }

    @Test
    fun `소스가 더 넓으면 좌우 센터 크롭`() {
        val out = FloatArray(4)
        // 4:3 소스 → 1:1 뷰포트: 가로 75%만 보임
        CoordMapper.visibleSubRect(4f / 3f, 1f, out)
        assertEquals(0.125f, out[0], eps)
        assertEquals(0f, out[1], eps)
        assertEquals(0.75f, out[2], eps)
        assertEquals(1f, out[3], eps)
    }

    @Test
    fun `소스가 더 좁으면 상하 센터 크롭`() {
        val out = FloatArray(4)
        // 3:4 소스 → 1:1 뷰포트: 세로 75%만 보임
        CoordMapper.visibleSubRect(3f / 4f, 1f, out)
        assertEquals(0f, out[0], eps)
        assertEquals(0.125f, out[1], eps)
        assertEquals(1f, out[2], eps)
        assertEquals(0.75f, out[3], eps)
    }

    // ───────────── uprightToSensor (회전 역변환) ─────────────

    @Test
    fun `회전 0도는 항등`() {
        val out = FloatArray(2)
        CoordMapper.uprightToSensor(0.3f, 0.7f, 0, out)
        assertEquals(0.3f, out[0], eps)
        assertEquals(0.7f, out[1], eps)
    }

    @Test
    fun `회전 90도 - upright 좌상단은 센서 좌하단`() {
        val out = FloatArray(2)
        // 센서를 시계방향 90도 돌리면 upright: 센서 (0,1)(좌하단)이 upright (0,0)(좌상단)이 된다
        CoordMapper.uprightToSensor(0f, 0f, 90, out)
        assertEquals(0f, out[0], eps)
        assertEquals(1f, out[1], eps)
        CoordMapper.uprightToSensor(1f, 0f, 90, out) // upright 우상단 ← 센서 좌상단
        assertEquals(0f, out[0], eps)
        assertEquals(0f, out[1], eps)
        CoordMapper.uprightToSensor(1f, 1f, 90, out) // upright 우하단 ← 센서 우상단
        assertEquals(1f, out[0], eps)
        assertEquals(0f, out[1], eps)
    }

    @Test
    fun `회전 180도 두 번 적용은 항등`() {
        val out1 = FloatArray(2)
        val out2 = FloatArray(2)
        CoordMapper.uprightToSensor(0.2f, 0.8f, 180, out1)
        CoordMapper.uprightToSensor(out1[0], out1[1], 180, out2)
        assertEquals(0.2f, out2[0], eps)
        assertEquals(0.8f, out2[1], eps)
    }

    @Test
    fun `회전 270도 - upright 좌상단은 센서 우상단`() {
        val out = FloatArray(2)
        CoordMapper.uprightToSensor(0f, 0f, 270, out)
        assertEquals(1f, out[0], eps)
        assertEquals(0f, out[1], eps)
    }

    @Test
    fun `회전 90도와 270도는 서로 역변환`() {
        val out1 = FloatArray(2)
        val out2 = FloatArray(2)
        // 90도 정변환의 역은 270도 적용과 동치 (정규화 사각 공간에서)
        CoordMapper.uprightToSensor(0.3f, 0.6f, 90, out1)
        CoordMapper.uprightToSensor(out1[0], out1[1], 270, out2)
        assertEquals(0.3f, out2[0], eps)
        assertEquals(0.6f, out2[1], eps)
    }

    // ───────────── displayUvToSensorGl (전체 체인) ─────────────

    private val fullCrop = floatArrayOf(0f, 0f, 1f, 1f)

    @Test
    fun `항등 조건에서는 GL t축 플립만 적용된다`() {
        val out = FloatArray(2)
        CoordMapper.displayUvToSensorGl(0.25f, 0.4f, fullCrop, mirror = false, rotationDegrees = 0, out = out)
        assertEquals(0.25f, out[0], eps)
        assertEquals(0.6f, out[1], eps) // 1 - 0.4
    }

    @Test
    fun `미러는 크롭 적용 후 가로 반전`() {
        val out = FloatArray(2)
        CoordMapper.displayUvToSensorGl(0.25f, 0.5f, fullCrop, mirror = true, rotationDegrees = 0, out = out)
        assertEquals(0.75f, out[0], eps)
        assertEquals(0.5f, out[1], eps)
    }

    @Test
    fun `크롭 서브렉트가 반영된다`() {
        val out = FloatArray(2)
        val crop = floatArrayOf(0.125f, 0f, 0.75f, 1f) // 4:3 → 1:1 크롭
        // 디스플레이 좌측 끝(u=0)은 카메라 u=0.125 지점
        CoordMapper.displayUvToSensorGl(0f, 0f, crop, mirror = false, rotationDegrees = 0, out = out)
        assertEquals(0.125f, out[0], eps)
        assertEquals(1f, out[1], eps)
    }

    @Test
    fun `미러와 회전 90도 조합`() {
        val out = FloatArray(2)
        // 디스플레이 (0,0) → 크롭 없음 → 미러 (1,0) → 90도 역회전 (0,0) → GL 플립 (0,1)
        CoordMapper.displayUvToSensorGl(0f, 0f, fullCrop, mirror = true, rotationDegrees = 90, out = out)
        assertEquals(0f, out[0], eps)
        assertEquals(1f, out[1], eps)
    }

    // ───────────── aspectAdapt (letterbox 차이 방어) ─────────────

    @Test
    fun `종횡비가 같으면 항등 변환`() {
        val out = FloatArray(4)
        CoordMapper.aspectAdapt(4f / 3f, 4f / 3f, out = out)
        assertEquals(1f, out[0], eps)
        assertEquals(0f, out[1], eps)
        assertEquals(1f, out[2], eps)
        assertEquals(0f, out[3], eps)
    }

    @Test
    fun `4대3 분석을 16대9 프리뷰로 매핑 - 세로 크롭`() {
        val out = FloatArray(4)
        CoordMapper.aspectAdapt(4f / 3f, 16f / 9f, out = out)
        // 중심은 불변
        assertEquals(0.5f, 0.5f * out[2] + out[3], eps)
        // 분석 v=0.125 (16:9 크롭 상단)는 프리뷰 v=0
        assertEquals(0f, 0.125f * out[2] + out[3], eps)
        // 가로는 항등
        assertEquals(1f, out[0], eps)
        assertEquals(0f, out[1], eps)
    }

    @Test
    fun `중심점은 어떤 종횡비 조합에서도 고정`() {
        val out = FloatArray(4)
        val aspects = floatArrayOf(1f, 4f / 3f, 16f / 9f, 3f / 4f)
        for (src in aspects) {
            for (dst in aspects) {
                CoordMapper.aspectAdapt(src, dst, out = out)
                assertEquals("src=$src dst=$dst", 0.5f, 0.5f * out[0] + out[1], eps)
                assertEquals("src=$src dst=$dst", 0.5f, 0.5f * out[2] + out[3], eps)
            }
        }
    }
}
