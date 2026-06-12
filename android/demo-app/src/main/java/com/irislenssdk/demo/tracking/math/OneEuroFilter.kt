package com.irislenssdk.demo.tracking.math

import kotlin.math.PI
import kotlin.math.abs

/**
 * One-Euro 필터 (Casiez et al.) — 랜드마크 지터 제거용.
 *
 * - 시작값은 ADR-0002 권고: min_cutoff 0.5, beta 0.007, d_cutoff 1.0
 * - MediaPipe 프로덕션 값(min_cutoff 0.05 / beta 80)은 객체 스케일 정규화 전제이므로
 *   여기에 직접 적용하면 안 된다 (docs/research/lens-rendering.md §6).
 * - 순수 로직 (Android 의존성 없음) — 단위 테스트 대상.
 */
internal class OneEuroFilter(
    private val minCutoff: Float = 0.5f,
    private val beta: Float = 0.007f,
    private val dCutoff: Float = 1.0f,
) {
    private var hasPrev = false
    private var prevValue = 0f
    private var prevDeriv = 0f
    private var prevTimeSec = 0.0

    /** 필터 상태 초기화 — 얼굴 재획득 시 잔상 글라이드 방지를 위해 호출한다. */
    fun reset() {
        hasPrev = false
        prevValue = 0f
        prevDeriv = 0f
        prevTimeSec = 0.0
    }

    /**
     * 새 샘플을 필터링한다.
     *
     * @param value 입력 값 (픽셀 좌표 권장 — 정규화 좌표는 종횡비 왜곡)
     * @param timeSec 단조 증가 타임스탬프 (초)
     */
    fun filter(value: Float, timeSec: Double): Float {
        if (!hasPrev) {
            hasPrev = true
            prevValue = value
            prevDeriv = 0f
            prevTimeSec = timeSec
            return value
        }
        val dt = (timeSec - prevTimeSec).toFloat().coerceAtLeast(1e-6f)
        prevTimeSec = timeSec

        // 미분(속도) 추정 — d_cutoff 저역 필터 적용
        val rawDeriv = (value - prevValue) / dt
        val alphaD = smoothingFactor(dCutoff, dt)
        val deriv = alphaD * rawDeriv + (1f - alphaD) * prevDeriv
        prevDeriv = deriv

        // 속도 적응형 컷오프: 느리면 지터 제거(강한 스무딩), 빠르면 랙 제거(약한 스무딩)
        val cutoff = minCutoff + beta * abs(deriv)
        val alpha = smoothingFactor(cutoff, dt)
        val filtered = alpha * value + (1f - alpha) * prevValue
        prevValue = filtered
        return filtered
    }

    private fun smoothingFactor(cutoff: Float, dt: Float): Float {
        val r = 2f * PI.toFloat() * cutoff * dt
        return r / (r + 1f)
    }
}
