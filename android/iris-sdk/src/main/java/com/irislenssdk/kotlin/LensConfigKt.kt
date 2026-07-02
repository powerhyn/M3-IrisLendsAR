/*
 * Copyright 2024 IrisLensSDK Team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.irislenssdk.kotlin

import com.irislenssdk.LensConfig as JavaLensConfig

/**
 * Kotlin 렌즈 설정 데이터 클래스.
 *
 * 불변 데이터 클래스로, copy()를 통해 새 설정을 생성합니다.
 *
 * 사용 예:
 * ```kotlin
 * val config = LensConfigKt(
 *     opacity = 0.8f,
 *     scale = 1.1f,
 *     blendMode = BlendMode.MULTIPLY
 * )
 *
 * // 일부 값만 변경
 * val newConfig = config.copy(opacity = 0.9f)
 * ```
 *
 * @property opacity 렌즈 투명도 (0.0 ~ 1.0, 기본값 0.7)
 * @property scale 렌즈 크기 배율 (기본값 1.3 — 실기기 튜닝 canonical)
 * @property offsetX X축 오프셋 (-1.0 ~ 1.0, 기본값 0.0)
 * @property offsetY Y축 오프셋 (-1.0 ~ 1.0, 기본값 0.0)
 * @property blendMode 블렌드 모드 (기본값 LUMINANCE_TINT_LINEAR, P6-W2 §5.12 canonical default)
 * @property edgeFeather 가장자리 페더링 (0.0 ~ 1.0, 기본값 0.15 — 실기기 튜닝 canonical)
 * @property applyLeft 왼쪽 눈 적용 여부 (기본값 true)
 * @property applyRight 오른쪽 눈 적용 여부 (기본값 true)
 */
data class LensConfigKt(
    val opacity: Float = DEFAULT_OPACITY,
    val scale: Float = DEFAULT_SCALE,
    val offsetX: Float = DEFAULT_OFFSET,
    val offsetY: Float = DEFAULT_OFFSET,
    val blendMode: BlendMode = BlendMode.LUMINANCE_TINT_LINEAR,
    val edgeFeather: Float = DEFAULT_FEATHER,
    val applyLeft: Boolean = true,
    val applyRight: Boolean = true
) {
    init {
        require(opacity in 0f..1f) { "opacity must be in range [0.0, 1.0]" }
        require(scale > 0f) { "scale must be positive" }
        require(offsetX in -1f..1f) { "offsetX must be in range [-1.0, 1.0]" }
        require(offsetY in -1f..1f) { "offsetY must be in range [-1.0, 1.0]" }
        require(edgeFeather in 0f..1f) { "edgeFeather must be in range [0.0, 1.0]" }
    }

    /**
     * Java LensConfig로 변환합니다.
     *
     * JNI 호출을 위해 Java 클래스가 필요할 때 사용합니다.
     *
     * @return 동일한 설정의 JavaLensConfig
     */
    fun toJava(): JavaLensConfig {
        return JavaLensConfig().apply {
            opacity = this@LensConfigKt.opacity
            scale = this@LensConfigKt.scale
            offsetX = this@LensConfigKt.offsetX
            offsetY = this@LensConfigKt.offsetY
            blendMode = this@LensConfigKt.blendMode.value
            edgeFeather = this@LensConfigKt.edgeFeather
            applyLeft = this@LensConfigKt.applyLeft
            applyRight = this@LensConfigKt.applyRight
        }
    }

    /**
     * 왼쪽 눈만 적용하는 설정을 반환합니다.
     */
    fun leftEyeOnly(): LensConfigKt = copy(applyLeft = true, applyRight = false)

    /**
     * 오른쪽 눈만 적용하는 설정을 반환합니다.
     */
    fun rightEyeOnly(): LensConfigKt = copy(applyLeft = false, applyRight = true)

    /**
     * 투명도를 조정한 설정을 반환합니다.
     *
     * @param newOpacity 새 투명도 값
     * @return 조정된 설정
     */
    fun withOpacity(newOpacity: Float): LensConfigKt = copy(opacity = newOpacity.coerceIn(0f, 1f))

    /**
     * 크기를 조정한 설정을 반환합니다.
     *
     * @param newScale 새 크기 배율
     * @return 조정된 설정
     */
    fun withScale(newScale: Float): LensConfigKt = copy(scale = newScale.coerceAtLeast(0.1f))

    /**
     * 오프셋을 조정한 설정을 반환합니다.
     *
     * @param x X축 오프셋
     * @param y Y축 오프셋
     * @return 조정된 설정
     */
    fun withOffset(x: Float, y: Float): LensConfigKt = copy(
        offsetX = x.coerceIn(-1f, 1f),
        offsetY = y.coerceIn(-1f, 1f)
    )

    companion object {
        private const val DEFAULT_OPACITY = 0.7f
        private const val DEFAULT_SCALE = 1.3f  // 실기기 튜닝(989fdac/6f596a2) canonical 승격 (P6-W2 §5.12 패턴)
        private const val DEFAULT_OFFSET = 0.0f
        private const val DEFAULT_FEATHER = 0.15f  // 실기기 튜닝(989fdac/6f596a2) canonical 승격

        /** 기본 설정 */
        val Default = LensConfigKt()

        /** 투명한 설정 (미리보기용) */
        val Transparent = LensConfigKt(opacity = 0.5f)

        /** 불투명한 설정 */
        val Opaque = LensConfigKt(opacity = 1.0f)

        /**
         * Java LensConfig에서 변환합니다.
         *
         * @param java Java LensConfig
         * @return 동일한 설정의 LensConfigKt
         */
        fun fromJava(java: JavaLensConfig): LensConfigKt {
            return LensConfigKt(
                opacity = java.opacity,
                scale = java.scale,
                offsetX = java.offsetX,
                offsetY = java.offsetY,
                blendMode = BlendMode.fromValue(java.blendMode),
                edgeFeather = java.edgeFeather,
                applyLeft = java.applyLeft,
                applyRight = java.applyRight
            )
        }
    }
}
