/**
 * 뷰티 필터 프리셋 팩토리
 *
 * beauty-tuner가 설계한 Natural/Studio/Glamour 프리셋을 생성하고,
 * 과보정 방지를 위한 sanitizeConfig를 제공합니다.
 *
 * 참조: docs/demo_app/beauty_preset_tuning_guide.md
 */
package com.irislenssdk.demo.beauty

import com.irislenssdk.BeautyFilterConfigV2
import kotlin.math.min

/**
 * 뷰티 프리셋 종류
 */
enum class BeautyPreset(val label: String) {
    NATURAL("Natural"),
    STUDIO("Studio"),
    GLAMOUR("Glamour"),
    CUSTOM("Custom")
}

object BeautyPresetFactory {

    // ========================================================================
    // 과보정 방지 하드 리밋
    // ========================================================================

    private const val MAX_SMOOTHING = 0.65f
    private const val MAX_WHITENING = 0.45f
    private const val MAX_BRIGHTNESS = 1.25f
    private const val MAX_SOFT_FOCUS = 0.50f
    private const val MAX_SLIM_FACE = 0.40f
    private const val MAX_ENLARGE_EYES = 0.35f
    private const val MAX_THIN_CHIN = 0.30f

    // ========================================================================
    // 프리셋 생성
    // ========================================================================

    /**
     * Natural 프리셋 생성.
     * "더 좋은 조명에서 찍은 것 같은" 자연스러운 보정.
     */
    fun createNaturalPreset(): BeautyFilterConfigV2 {
        return BeautyFilterConfigV2.Builder()
            .enabled(true)
            .intensity(0.50f)
            .smoothing(0.25f)
            .brightness(1.03f)
            .whitening(0.08f)
            .colorBalance(0.10f)
            .softFocus(0.10f)
            .wrinkleRemove(0.0f)
            .slimFace(0.08f)
            .enlargeEyes(0.05f)
            .thinChin(0.05f)
            .protectEyes(true)
            .protectLips(true)
            .useGpu(true)
            .roiOnly(false)
            .build()
    }

    /**
     * Studio 프리셋 생성.
     * "프로 사진작가가 조명 세팅하고 찍은 듯한" 보정.
     */
    fun createStudioPreset(): BeautyFilterConfigV2 {
        return BeautyFilterConfigV2.Builder()
            .enabled(true)
            .intensity(0.70f)
            .smoothing(0.40f)
            .brightness(1.08f)
            .whitening(0.18f)
            .colorBalance(0.05f)
            .softFocus(0.20f)
            .wrinkleRemove(0.0f)
            .slimFace(0.15f)
            .enlargeEyes(0.12f)
            .thinChin(0.10f)
            .protectEyes(true)
            .protectLips(true)
            .useGpu(true)
            .roiOnly(false)
            .build()
    }

    /**
     * Glamour 프리셋 생성.
     * "매거진 화보 속 모델" 느낌의 확실한 보정.
     */
    fun createGlamourPreset(): BeautyFilterConfigV2 {
        return BeautyFilterConfigV2.Builder()
            .enabled(true)
            .intensity(0.85f)
            .smoothing(0.55f)
            .brightness(1.12f)
            .whitening(0.30f)
            .colorBalance(0.15f)
            .softFocus(0.35f)
            .wrinkleRemove(0.0f)
            .slimFace(0.22f)
            .enlargeEyes(0.20f)
            .thinChin(0.15f)
            .protectEyes(true)
            .protectLips(true)
            .useGpu(true)
            .roiOnly(false)
            .build()
    }

    /**
     * 프리셋 종류에 따라 BeautyFilterConfigV2를 생성합니다.
     * CUSTOM인 경우 현재 설정을 그대로 반환합니다.
     */
    fun createPreset(preset: BeautyPreset, current: BeautyFilterConfigV2? = null): BeautyFilterConfigV2 {
        return when (preset) {
            BeautyPreset.NATURAL -> createNaturalPreset()
            BeautyPreset.STUDIO -> createStudioPreset()
            BeautyPreset.GLAMOUR -> createGlamourPreset()
            BeautyPreset.CUSTOM -> current?.let { BeautyFilterConfigV2(it) } ?: BeautyFilterConfigV2()
        }
    }

    // ========================================================================
    // 과보정 방지 (sanitizeConfig)
    // ========================================================================

    /**
     * 과보정 방지 규칙을 적용합니다.
     *
     * 5가지 조합 제한 규칙:
     * 1. smoothing + whitening <= 0.70 (플라스틱 피부 방지)
     * 2. whitening + (brightness - 1.0) <= 0.35 (과노출 방지)
     * 3. smoothing + softFocus <= 0.75 (디테일 손실 방지)
     * 4. slimFace + thinChin <= 0.45 (뾰족한 얼굴 방지)
     * 5. enlargeEyes <= slimFace * 2.0 (비정상 비율 방지, enlargeEyes > 0.20일 때)
     *
     * @param config 검증할 설정 (in-place로 수정됨)
     * @return 수정된 config (체이닝용)
     */
    fun sanitizeConfig(config: BeautyFilterConfigV2): BeautyFilterConfigV2 {
        // 하드 리밋 적용
        config.smoothing = min(config.smoothing, MAX_SMOOTHING)
        config.whitening = min(config.whitening, MAX_WHITENING)
        config.brightness = min(config.brightness, MAX_BRIGHTNESS)
        config.softFocus = min(config.softFocus, MAX_SOFT_FOCUS)
        config.slimFace = min(config.slimFace, MAX_SLIM_FACE)
        config.enlargeEyes = min(config.enlargeEyes, MAX_ENLARGE_EYES)
        config.thinChin = min(config.thinChin, MAX_THIN_CHIN)

        // 규칙 1: smoothing + whitening 합산 제한 (플라스틱 피부 방지)
        if (config.smoothing + config.whitening > 0.70f) {
            config.whitening = min(config.whitening, 0.70f - config.smoothing)
        }

        // 규칙 2: whitening + (brightness - 1.0) 합산 제한 (과노출 방지)
        if (config.whitening + (config.brightness - 1.0f) > 0.35f) {
            config.brightness = min(config.brightness, 1.0f + 0.35f - config.whitening)
        }

        // 규칙 3: smoothing + softFocus 합산 제한 (디테일 손실 방지)
        if (config.smoothing + config.softFocus > 0.75f) {
            config.softFocus = min(config.softFocus, 0.75f - config.smoothing)
        }

        // 규칙 4: slimFace + thinChin 합산 제한 (뾰족한 얼굴 방지)
        if (config.slimFace + config.thinChin > 0.45f) {
            config.thinChin = min(config.thinChin, 0.45f - config.slimFace)
        }

        // 규칙 5: enlargeEyes / slimFace 비율 제한 (비정상 비율 방지)
        if (config.enlargeEyes > config.slimFace * 2.0f && config.enlargeEyes > 0.20f) {
            config.enlargeEyes = min(config.enlargeEyes, config.slimFace * 2.0f)
        }

        return config
    }

    // ========================================================================
    // LUT 프리셋 연결
    // ========================================================================

    /**
     * 뷰티 프리셋에 연결된 기본 LUT 프리셋
     */
    fun getDefaultLutPreset(preset: BeautyPreset): LutTextureLoader.LutPreset? {
        return when (preset) {
            BeautyPreset.NATURAL -> LutTextureLoader.LutPreset.NATURAL_GLOW
            BeautyPreset.STUDIO -> LutTextureLoader.LutPreset.CLEAN_PORCELAIN
            BeautyPreset.GLAMOUR -> LutTextureLoader.LutPreset.ROSY_GLOW
            BeautyPreset.CUSTOM -> null  // Custom has no default LUT
        }
    }

    /**
     * 기본 LUT 강도
     */
    fun getDefaultLutIntensity(preset: BeautyPreset): Float {
        return when (preset) {
            BeautyPreset.NATURAL -> 0.3f
            BeautyPreset.STUDIO -> 0.5f
            BeautyPreset.GLAMOUR -> 0.6f
            BeautyPreset.CUSTOM -> 0.0f
        }
    }
}
