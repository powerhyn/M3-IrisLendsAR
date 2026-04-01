/**
 * One Euro Filter - 적응형 노이즈 필터링
 *
 * 느린 움직임에는 강한 스무딩, 빠른 움직임에는 빠른 반응을 제공하는 필터.
 * 깜빡임과 흔들거림을 효과적으로 제거하면서 반응성 유지.
 *
 * @param minCutoff 최소 컷오프 주파수 (낮을수록 부드러움)
 * @param beta 속도 계수 (높을수록 빠른 움직임에 민감)
 * @param dCutoff 미분 컷오프 주파수
 *
 * 참조: https://cristal.univ-lille.fr/~casiez/1euro/
 */
package com.irislenssdk.demo.camera

import kotlin.math.abs

class OneEuroFilter(
    private val minCutoff: Float = 1.0f,
    private val beta: Float = 0.007f,
    private val dCutoff: Float = 1.0f
) {
    private var x: Float = 0f
    private var dx: Float = 0f
    private var lastTime: Long = 0L
    private var initialized: Boolean = false

    /**
     * 새로운 값을 필터링
     * @param value 입력 값
     * @param timestamp 타임스탬프 (밀리초)
     * @return 필터링된 값
     */
    fun filter(value: Float, timestamp: Long): Float {
        if (!initialized) {
            x = value
            dx = 0f
            lastTime = timestamp
            initialized = true
            return value
        }

        // 시간 간격 계산 (초 단위)
        val dt = ((timestamp - lastTime).coerceAtLeast(1L)) / 1000f
        lastTime = timestamp

        // 속도 추정 (미분 필터링)
        val edx = (value - x) / dt
        dx = lowPassFilter(edx, dx, alpha(dCutoff, dt))

        // 적응형 컷오프 주파수 계산
        val cutoff = minCutoff + beta * abs(dx)

        // 위치 필터링
        x = lowPassFilter(value, x, alpha(cutoff, dt))

        return x
    }

    /**
     * 필터 초기화 (검출 실패 후 재검출 시)
     */
    fun reset() {
        initialized = false
    }

    /**
     * 현재 필터링된 값 반환
     */
    fun getValue(): Float = x

    /**
     * 저역 통과 필터
     */
    private fun lowPassFilter(x: Float, prevX: Float, alpha: Float): Float {
        return alpha * x + (1f - alpha) * prevX
    }

    /**
     * 알파 값 계산 (컷오프 주파수 기반)
     */
    private fun alpha(cutoff: Float, dt: Float): Float {
        val tau = 1f / (2f * Math.PI.toFloat() * cutoff)
        return 1f / (1f + tau / dt)
    }
}
