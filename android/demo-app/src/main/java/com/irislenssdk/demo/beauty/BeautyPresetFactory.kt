/**
 * 뷰티 필터 프리셋 팩토리
 *
 * 잡티 보정(SkinQuality) + Vivid 포스트프로세싱 조합 프리셋.
 * 기존 스무딩/화이트닝/소프트포커스 등 레거시 뷰티는 0으로 비활성화.
 */
package com.irislenssdk.demo.beauty

import com.irislenssdk.BeautyFilterConfigV2

/**
 * Vivid 프리셋 종류
 */
enum class BeautyPreset(val label: String) {
    NATURAL_GLOW("Natural"),
    SPRING("Spring"),
    STUDIO("Studio"),
    GOLDEN_HOUR("Golden"),
    VIVID_POP("Vivid"),
    CUSTOM("Custom")
}

object BeautyPresetFactory {

    // ========================================================================
    // 베이스 빌더 — 잡티 보정만 켜고 레거시 뷰티는 모두 OFF
    // ========================================================================

    private fun baseBuilder(skinQuality: Float = 0.3f) = BeautyFilterConfigV2.Builder()
        .enabled(true)
        .intensity(1.0f)          // 마스터 강도 100%
        .smoothing(0.0f)
        .brightness(1.0f)
        .whitening(0.0f)
        .colorBalance(0.0f)
        .softFocus(0.0f)
        .skinQuality(skinQuality)
        .smoothIntensity(0.0f)
        .poreReduction(0.0f)
        .wrinkleRemove(0.0f)
        .slimFace(0.0f)
        .enlargeEyes(0.0f)
        .thinChin(0.0f)
        .protectEyes(true)
        .protectLips(true)
        .useGpu(true)
        .roiOnly(false)

    // ========================================================================
    // Vivid 프리셋
    // ========================================================================

    /**
     * Natural Glow — 일상 셀피, "더 좋은 조명" 느낌
     */
    fun createNaturalGlowPreset(): BeautyFilterConfigV2 {
        return baseBuilder(skinQuality = 0.2f)
            .vividIntensity(0.4f)
            .vividSaturation(0.25f)
            .vividBrightness(0.05f)
            .vividWarmth(0.15f)
            .build()
    }

    /**
     * Spring — 화사하고 따뜻한 봄날 톤
     */
    fun createSpringPreset(): BeautyFilterConfigV2 {
        return baseBuilder(skinQuality = 0.3f)
            .vividIntensity(0.6f)
            .vividSaturation(0.4f)
            .vividBrightness(0.1f)
            .vividWarmth(0.3f)
            .build()
    }

    /**
     * Studio — 채도+밝기, 웜톤 없이 깨끗한 스튜디오 느낌
     */
    fun createStudioPreset(): BeautyFilterConfigV2 {
        return baseBuilder(skinQuality = 0.35f)
            .vividIntensity(0.5f)
            .vividSaturation(0.3f)
            .vividBrightness(0.12f)
            .vividWarmth(0.0f)
            .build()
    }

    /**
     * Golden Hour — 웜톤 강조, 일몰 분위기
     */
    fun createGoldenHourPreset(): BeautyFilterConfigV2 {
        return baseBuilder(skinQuality = 0.25f)
            .vividIntensity(0.7f)
            .vividSaturation(0.2f)
            .vividBrightness(0.08f)
            .vividWarmth(0.6f)
            .build()
    }

    /**
     * Vivid Pop — 강한 채도+밝기, SNS/제품 촬영용
     */
    fun createVividPopPreset(): BeautyFilterConfigV2 {
        return baseBuilder(skinQuality = 0.4f)
            .vividIntensity(0.8f)
            .vividSaturation(0.5f)
            .vividBrightness(0.15f)
            .vividWarmth(0.1f)
            .build()
    }

    /**
     * Custom — 잡티 보정 + vivid OFF 기본 상태
     */
    fun createCustomPreset(): BeautyFilterConfigV2 {
        return baseBuilder(skinQuality = 0.3f)
            .vividIntensity(0.0f)
            .vividSaturation(0.0f)
            .vividBrightness(0.0f)
            .vividWarmth(0.0f)
            .build()
    }

    /**
     * 프리셋 종류에 따라 BeautyFilterConfigV2를 생성합니다.
     */
    fun createPreset(preset: BeautyPreset, current: BeautyFilterConfigV2? = null): BeautyFilterConfigV2 {
        return when (preset) {
            BeautyPreset.NATURAL_GLOW -> createNaturalGlowPreset()
            BeautyPreset.SPRING -> createSpringPreset()
            BeautyPreset.STUDIO -> createStudioPreset()
            BeautyPreset.GOLDEN_HOUR -> createGoldenHourPreset()
            BeautyPreset.VIVID_POP -> createVividPopPreset()
            BeautyPreset.CUSTOM -> current?.let { BeautyFilterConfigV2(it) } ?: createCustomPreset()
        }
    }

    // ========================================================================
    // 과보정 방지 (Vivid 전용)
    // ========================================================================

    /**
     * Vivid 과보정 방지 규칙.
     * - vividSaturation 0.6 초과 시 과포화 위험
     * - vividBrightness 0.3 초과 시 하이라이트 뭉개짐
     * - vividWarmth 0.7 초과 시 노란끼 과다
     */
    fun sanitizeConfig(config: BeautyFilterConfigV2): BeautyFilterConfigV2 {
        config.vividIntensity = config.vividIntensity.coerceIn(0.0f, 1.0f)
        config.vividSaturation = config.vividSaturation.coerceIn(0.0f, 0.6f)
        config.vividBrightness = config.vividBrightness.coerceIn(0.0f, 0.3f)
        config.vividWarmth = config.vividWarmth.coerceIn(0.0f, 0.7f)
        config.skinQuality = config.skinQuality.coerceIn(0.0f, 1.0f)
        return config
    }
}
