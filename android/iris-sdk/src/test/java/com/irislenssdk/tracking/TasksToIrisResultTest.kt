package com.irislenssdk.tracking

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.irislenssdk.IrisResult
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * TasksToIrisResult 변환 계약 검증 (REFACTOR-3-3 §4.1).
 *
 * 핵심 계약:
 *  - 필드 매핑은 코어 레거시 시맨틱: IrisResult.left* ← 인덱스 468그룹 /
 *    right* ← 473그룹 (landmark_injection.h — LandmarkIndices 해부학 라벨과 반대)
 *  - 좌표는 sensorToUpright 적용 후 upright 정규화 (함정 #13), 미러 미적용 (ADR §7.4)
 *  - radius는 중심→경계 4점 평균 '픽셀' 거리 (LEGACY calculateIrisRadius 동일, 함정 #5)
 *  - EAR은 코어 어댑터 computeEyeAspectRatio 동일 수식 (정규화 공간)
 *  - confidence 1.0 고정 (Tasks per-face confidence 미제공 — ADR §6.2)
 */
class TasksToIrisResultTest {

    companion object {
        private const val EPS = 1e-5f
        private const val SENSOR_W = 1000
        private const val SENSOR_H = 500
    }

    /** 478점 합성 — 기본값 (0.5, 0.5), 홍채·EAR 인덱스만 기지값으로 채운다. */
    private fun syntheticLandmarks(): MutableList<NormalizedLandmark> {
        val list = MutableList(478) { NormalizedLandmark.create(0.5f, 0.5f, 0f) }

        // 468그룹 → IrisResult.left* (코어 레거시 시맨틱).
        // 경계 순서 §7.0: right(469) → top(470) → left(471) → bottom(472).
        // 픽셀 거리: x축 0.02×1000=20px, y축 0.04×500=20px — 반경 20px 균일.
        list[468] = NormalizedLandmark.create(0.40f, 0.50f, -0.01f)
        list[469] = NormalizedLandmark.create(0.42f, 0.50f, 0f)
        list[470] = NormalizedLandmark.create(0.40f, 0.46f, 0f)
        list[471] = NormalizedLandmark.create(0.38f, 0.50f, 0f)
        list[472] = NormalizedLandmark.create(0.40f, 0.54f, 0f)

        // 473그룹 → IrisResult.right*
        list[473] = NormalizedLandmark.create(0.60f, 0.50f, -0.02f)
        list[474] = NormalizedLandmark.create(0.62f, 0.50f, 0f)
        list[475] = NormalizedLandmark.create(0.60f, 0.46f, 0f)
        list[476] = NormalizedLandmark.create(0.58f, 0.50f, 0f)
        list[477] = NormalizedLandmark.create(0.60f, 0.54f, 0f)

        // EAR(left 필드 ← 33그룹): p1=33, p2=160, p3=158, p4=133, p5=153, p6=144
        // |p1-p4|=0.04, |p2-p6|=0.02, |p3-p5|=0.02 → EAR=(0.02+0.02)/(2·0.04)=0.5
        list[33] = NormalizedLandmark.create(0.30f, 0.50f, 0f)
        list[160] = NormalizedLandmark.create(0.31f, 0.49f, 0f)
        list[158] = NormalizedLandmark.create(0.33f, 0.49f, 0f)
        list[133] = NormalizedLandmark.create(0.34f, 0.50f, 0f)
        list[153] = NormalizedLandmark.create(0.33f, 0.51f, 0f)
        list[144] = NormalizedLandmark.create(0.31f, 0.51f, 0f)

        return list
    }

    @Test
    fun `rot0 - 코어 레거시 시맨틱 매핑 (left=468그룹, right=473그룹)`() {
        val out = IrisResult()
        val ok = TasksToIrisResult.convert(syntheticLandmarks(), 0, SENSOR_W, SENSOR_H, 1234L, out)

        assertTrue(ok)
        assertTrue(out.detected)
        assertTrue(out.leftDetected)
        assertTrue(out.rightDetected)
        // left* ← 468 (해부학 라벨로는 피험자 우안 — LEGACY detector와 동일 매핑)
        assertEquals(0.40f, out.leftIrisX, EPS)
        assertEquals(0.50f, out.leftIrisY, EPS)
        assertEquals(-0.01f, out.leftIrisZ, EPS)
        // right* ← 473
        assertEquals(0.60f, out.rightIrisX, EPS)
        assertEquals(0.50f, out.rightIrisY, EPS)
        assertEquals(-0.02f, out.rightIrisZ, EPS)
        // 프레임/타임스탬프
        assertEquals(SENSOR_W, out.frameWidth)
        assertEquals(SENSOR_H, out.frameHeight)
        assertEquals(1234L, out.timestampMs)
    }

    @Test
    fun `radius - 중심→경계 4점 평균 픽셀 거리 (LEGACY calculateIrisRadius 동일)`() {
        val out = IrisResult()
        TasksToIrisResult.convert(syntheticLandmarks(), 0, SENSOR_W, SENSOR_H, 0L, out)

        // 경계 4점 모두 20px 거리 (x: 0.02×1000, y: 0.04×500) → 평균 20px.
        // 정규화 좌표 직접 거리였다면 좌/우(0.02)와 상/하(0.04)가 달라진다 — 함정 #5 검증.
        assertEquals(20f, out.leftRadius, 1e-3f)
        assertEquals(20f, out.rightRadius, 1e-3f)
    }

    @Test
    fun `EAR - 코어 computeEyeAspectRatio 동일 수식`() {
        val out = IrisResult()
        TasksToIrisResult.convert(syntheticLandmarks(), 0, SENSOR_W, SENSOR_H, 0L, out)

        assertEquals(0.5f, out.eyelidRatioLeft, 1e-4f)
    }

    @Test
    fun `confidence - Tasks 미제공 → 1_0 고정 (ADR §6_2)`() {
        val out = IrisResult()
        TasksToIrisResult.convert(syntheticLandmarks(), 0, SENSOR_W, SENSOR_H, 0L, out)

        assertEquals(1.0f, out.confidence, 0f)
    }

    @Test
    fun `rot270 - sensorToUpright 적용 + 프레임 치수 스왑 (전면 카메라 자연 경로)`() {
        val out = IrisResult()
        TasksToIrisResult.convert(syntheticLandmarks(), 270, SENSOR_W, SENSOR_H, 0L, out)

        // sensorToUpright(x, y, 270) = (y, 1-x)
        assertEquals(0.50f, out.leftIrisX, EPS)
        assertEquals(1f - 0.40f, out.leftIrisY, EPS)
        assertEquals(0.50f, out.rightIrisX, EPS)
        assertEquals(1f - 0.60f, out.rightIrisY, EPS)
        // 90/270 회전 시 width/height 스왑은 공급자 책임 (ADR §7.1)
        assertEquals(SENSOR_H, out.frameWidth)
        assertEquals(SENSOR_W, out.frameHeight)
        // 반경은 회전 후 픽셀 공간에서도 동일 (등방 20px 구성)
        assertEquals(20f, out.leftRadius, 1e-3f)
    }

    @Test
    fun `faceMesh - 478점 전체 upright 변환 저장`() {
        val out = IrisResult()
        TasksToIrisResult.convert(syntheticLandmarks(), 90, SENSOR_W, SENSOR_H, 0L, out)

        assertTrue(out.faceMeshValid)
        // sensorToUpright(x, y, 90) = (1-y, x): 468=(0.40, 0.50) → (0.50, 0.40)
        assertEquals(0.50f, out.faceMesh[468 * 3], EPS)
        assertEquals(0.40f, out.faceMesh[468 * 3 + 1], EPS)
        // z 패스스루 (비기하 용도 한정 — ADR §7.2)
        assertEquals(-0.01f, out.faceMesh[468 * 3 + 2], EPS)
    }

    @Test
    fun `홍채 범위 이탈 - 해당 눈만 미검출 (LEGACY extractIris 판정 재현)`() {
        val landmarks = syntheticLandmarks()
        landmarks[472] = NormalizedLandmark.create(0.40f, 1.2f, 0f) // bottom이 프레임 밖
        val out = IrisResult()
        TasksToIrisResult.convert(landmarks, 0, SENSOR_W, SENSOR_H, 0L, out)

        assertFalse(out.leftDetected)
        assertTrue(out.rightDetected)
        assertTrue(out.detected) // 한쪽이라도 검출이면 true
    }

    @Test
    fun `478점 미만 - noFace 처리 (프레임 치수는 유지)`() {
        val out = IrisResult()
        val ok = TasksToIrisResult.convert(
            MutableList(100) { NormalizedLandmark.create(0.5f, 0.5f, 0f) },
            270, SENSOR_W, SENSOR_H, 77L, out,
        )

        assertFalse(ok)
        assertFalse(out.detected)
        assertFalse(out.faceMeshValid)
        assertEquals(SENSOR_H, out.frameWidth) // 270 → 스왑
        assertEquals(SENSOR_W, out.frameHeight)
        assertEquals(77L, out.timestampMs)
    }

    @Test
    fun `fillNoFace - stabilize hold 입력 일관성`() {
        val out = IrisResult()
        TasksToIrisResult.fillNoFace(0, SENSOR_W, SENSOR_H, 42L, out)

        assertFalse(out.detected)
        assertEquals(SENSOR_W, out.frameWidth)
        assertEquals(SENSOR_H, out.frameHeight)
        assertEquals(42L, out.timestampMs)
    }

    // ── ④ W4-B4: convertWithLuma (convert + avg_iris_luma SDK 단일 진입) ──

    /** 균일 그레이(v) RGBA 버퍼 — roiLumaLinear off=y*rowStride+x*4 인덱싱 기준 packed. */
    private fun grayRgba(v: Int): java.nio.ByteBuffer {
        val buf = java.nio.ByteBuffer.allocate(SENSOR_W * SENSOR_H * 4)
        java.util.Arrays.fill(buf.array(), v.toByte())
        return buf
    }

    @Test
    fun `convertWithLuma - convert 위임 + 홍채 luma 채움`() {
        val out = IrisResult()
        val rgba = grayRgba(128) // (128/255)^2 ≈ 0.252 linear (Rec.709, 균일채널)
        val ok = TasksToIrisResult.convertWithLuma(
            syntheticLandmarks(), 0, SENSOR_W, SENSOR_H, 1234L, rgba, SENSOR_W * 4, out
        )

        assertTrue(ok)
        assertTrue(out.detected)
        // convert 위임 정상 — 좌표 매핑은 convert()와 동일
        assertEquals(0.40f, out.leftIrisX, EPS)
        assertEquals(0.60f, out.rightIrisX, EPS)
        // P7-W2 luma 채움 (양안 검출 + 균일 그레이 → 양수, ≈ 0.252)
        assertTrue("left luma>0", out.avgIrisLumaLeft > 0f)
        assertTrue("right luma>0", out.avgIrisLumaRight > 0f)
        assertEquals(0.252f, out.avgIrisLumaLeft, 0.03f)
        assertEquals(0.252f, out.avgIrisLumaRight, 0.03f)
    }

    @Test
    fun `convertWithLuma - 미검출(478점 미만)이면 luma 미측정 -1`() {
        val out = IrisResult()
        val ok = TasksToIrisResult.convertWithLuma(
            MutableList(10) { NormalizedLandmark.create(0.5f, 0.5f, 0f) },
            0, SENSOR_W, SENSOR_H, 0L, grayRgba(200), SENSOR_W * 4, out
        )

        assertFalse(ok)
        assertEquals(-1f, out.avgIrisLumaLeft, EPS)
        assertEquals(-1f, out.avgIrisLumaRight, EPS)
    }
}
