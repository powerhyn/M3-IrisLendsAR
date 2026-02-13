/**
 * LUT (Look-Up Table) 텍스처 로더
 *
 * 2D LUT PNG 이미지를 OpenGL 3D 텍스처로 변환합니다.
 *
 * LUT 이미지 형식:
 * - 64x64x64 3D LUT를 8x8 그리드의 2D 이미지로 저장
 * - 최종 이미지 크기: 512x512 (64*8 x 64*8)
 * - 각 블록: 64x64 슬라이스 (Blue 채널 고정)
 *
 * 참조: docs/demo_app/beauty_app_benchmark_report.md (LUT 갭 분석)
 */
package com.irislenssdk.demo.beauty

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.opengl.GLES31
import android.util.Log
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder

object LutTextureLoader {

    private const val TAG = "LutTextureLoader"
    private const val LUT_SIZE = 64  // 64x64x64 LUT

    enum class LutPreset(val displayName: String, val assetPath: String) {
        ROSY_GLOW("Rosy Glow", "luts/rosy_glow.png"),
        PEACH_CREAM("Peach Cream", "luts/peach_cream.png"),
        CLEAN_PORCELAIN("Clean Porcelain", "luts/clean_porcelain.png"),
        GOLDEN_HOUR("Golden Hour", "luts/golden_hour.png"),
        FILM_VINTAGE("Film Vintage", "luts/film_vintage.png"),
        COOL_EDITORIAL("Cool Editorial", "luts/cool_editorial.png"),
        WARM_SUNSET("Warm Sunset", "luts/warm_sunset.png"),
        NATURAL_GLOW("Natural Glow", "luts/natural_glow.png")
    }

    /**
     * assets에서 LUT PNG를 로드하여 OpenGL 3D 텍스처를 생성합니다.
     *
     * GL 스레드에서 호출해야 합니다.
     *
     * @param context Android Context
     * @param assetPath assets 내 LUT 파일 경로
     * @return 3D 텍스처 ID (실패 시 0)
     */
    fun loadLutTexture(context: Context, assetPath: String): Int {
        val bitmap = loadBitmapFromAssets(context, assetPath) ?: return 0
        val textureId = createLut3dTexture(bitmap)
        bitmap.recycle()
        return textureId
    }

    /**
     * Bitmap으로부터 OpenGL 3D 텍스처를 생성합니다.
     *
     * GL 스레드에서 호출해야 합니다.
     *
     * @param bitmap 2D LUT 이미지 (512x512 = 8x8 그리드)
     * @return 3D 텍스처 ID (실패 시 0)
     */
    fun createLut3dTexture(bitmap: Bitmap): Int {
        val expectedSize = LUT_SIZE * 8  // 512
        if (bitmap.width != expectedSize || bitmap.height != expectedSize) {
            Log.e(TAG, "Invalid LUT size: ${bitmap.width}x${bitmap.height}, expected ${expectedSize}x${expectedSize}")
            return 0
        }

        // 2D LUT → 3D 데이터 변환
        val lutData = convert2dTo3d(bitmap)

        // 3D 텍스처 생성
        val textures = IntArray(1)
        GLES31.glGenTextures(1, textures, 0)
        val textureId = textures[0]

        GLES31.glBindTexture(GLES31.GL_TEXTURE_3D, textureId)
        GLES31.glTexParameteri(GLES31.GL_TEXTURE_3D, GLES31.GL_TEXTURE_MIN_FILTER, GLES31.GL_LINEAR)
        GLES31.glTexParameteri(GLES31.GL_TEXTURE_3D, GLES31.GL_TEXTURE_MAG_FILTER, GLES31.GL_LINEAR)
        GLES31.glTexParameteri(GLES31.GL_TEXTURE_3D, GLES31.GL_TEXTURE_WRAP_S, GLES31.GL_CLAMP_TO_EDGE)
        GLES31.glTexParameteri(GLES31.GL_TEXTURE_3D, GLES31.GL_TEXTURE_WRAP_T, GLES31.GL_CLAMP_TO_EDGE)
        GLES31.glTexParameteri(GLES31.GL_TEXTURE_3D, GLES31.GL_TEXTURE_WRAP_R, GLES31.GL_CLAMP_TO_EDGE)

        GLES31.glTexImage3D(
            GLES31.GL_TEXTURE_3D, 0, GLES31.GL_RGBA,
            LUT_SIZE, LUT_SIZE, LUT_SIZE, 0,
            GLES31.GL_RGBA, GLES31.GL_UNSIGNED_BYTE, lutData
        )

        GLES31.glBindTexture(GLES31.GL_TEXTURE_3D, 0)

        Log.d(TAG, "3D LUT texture created: id=$textureId, size=${LUT_SIZE}x${LUT_SIZE}x${LUT_SIZE}")
        return textureId
    }

    /**
     * 2D LUT 이미지를 3D 텍스처 데이터로 변환합니다.
     *
     * 2D 이미지 레이아웃 (8x8 그리드):
     * - 행: blueSlice / 8 (하위 3비트)
     * - 열: blueSlice % 8 (상위 3비트)
     * - 각 슬라이스 내: X = Red, Y = Green
     */
    private fun convert2dTo3d(bitmap: Bitmap): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(LUT_SIZE * LUT_SIZE * LUT_SIZE * 4)
            .order(ByteOrder.nativeOrder())

        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        for (blue in 0 until LUT_SIZE) {
            // 8x8 그리드에서 블록 위치 계산
            val blockX = blue % 8
            val blockY = blue / 8

            for (green in 0 until LUT_SIZE) {
                for (red in 0 until LUT_SIZE) {
                    // 2D 이미지에서 픽셀 좌표
                    val pixelX = blockX * LUT_SIZE + red
                    val pixelY = blockY * LUT_SIZE + green

                    val pixel = pixels[pixelY * bitmap.width + pixelX]

                    // ARGB → RGBA
                    buffer.put(((pixel shr 16) and 0xFF).toByte())  // R
                    buffer.put(((pixel shr 8) and 0xFF).toByte())   // G
                    buffer.put((pixel and 0xFF).toByte())            // B
                    buffer.put(((pixel shr 24) and 0xFF).toByte())  // A
                }
            }
        }

        buffer.flip()
        return buffer
    }

    /**
     * assets에서 비트맵 로드
     */
    private fun loadBitmapFromAssets(context: Context, assetPath: String): Bitmap? {
        return try {
            context.assets.open(assetPath).use { inputStream ->
                BitmapFactory.decodeStream(inputStream)
            }
        } catch (e: IOException) {
            Log.e(TAG, "Failed to load LUT from assets: $assetPath", e)
            null
        }
    }

    /**
     * Identity LUT Bitmap을 프로그래밍적으로 생성합니다.
     *
     * 이 LUT를 적용하면 원본 색상이 그대로 유지됩니다.
     * 테스트 및 기본 LUT 용도로 사용합니다.
     *
     * @return 512x512 identity LUT Bitmap
     */
    fun generateIdentityLutBitmap(): Bitmap {
        val size = LUT_SIZE * 8  // 512
        val bitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(size * size)

        for (blue in 0 until LUT_SIZE) {
            val blockX = blue % 8
            val blockY = blue / 8

            for (green in 0 until LUT_SIZE) {
                for (red in 0 until LUT_SIZE) {
                    val pixelX = blockX * LUT_SIZE + red
                    val pixelY = blockY * LUT_SIZE + green

                    // Identity: output = input (정규화된 색상)
                    val r = (red * 255 / (LUT_SIZE - 1))
                    val g = (green * 255 / (LUT_SIZE - 1))
                    val b = (blue * 255 / (LUT_SIZE - 1))

                    pixels[pixelY * size + pixelX] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                }
            }
        }

        bitmap.setPixels(pixels, 0, size, 0, 0, size, size)
        return bitmap
    }

    // ========================================================================
    // 하이브리드 프리셋 로딩 (PNG 우선, 프로그래밍 폴백)
    // ========================================================================

    /**
     * LUT 프리셋을 로드합니다 (하이브리드: PNG 우선, 프로그래밍 폴백).
     *
     * GL 스레드에서 호출해야 합니다.
     *
     * @param context Android Context
     * @param preset LUT 프리셋
     * @return 3D 텍스처 ID (실패 시 0)
     */
    fun loadPresetLut(context: Context, preset: LutPreset): Int {
        // 1. Try loading from assets PNG
        val fromAsset = loadLutTexture(context, preset.assetPath)
        if (fromAsset != 0) {
            Log.d(TAG, "Loaded LUT preset '${preset.displayName}' from assets")
            return fromAsset
        }

        // 2. Fallback: generate programmatically
        Log.d(TAG, "Generating LUT preset '${preset.displayName}' programmatically")
        val bitmap = generatePresetBitmap(preset)
        val textureId = createLut3dTexture(bitmap)
        bitmap.recycle()
        return textureId
    }

    // ========================================================================
    // 프로그래밍 방식 프리셋 생성
    // ========================================================================

    private fun generatePresetBitmap(preset: LutPreset): Bitmap {
        val identity = generateIdentityLutBitmap()
        val size = LUT_SIZE * 8  // 512
        val pixels = IntArray(size * size)
        identity.getPixels(pixels, 0, size, 0, 0, size, size)

        for (i in pixels.indices) {
            val pixel = pixels[i]
            var r = (pixel shr 16) and 0xFF
            var g = (pixel shr 8) and 0xFF
            var b = pixel and 0xFF
            val a = (pixel shr 24) and 0xFF

            // Apply preset-specific color transformation
            val rgb = applyPresetTransform(preset, r, g, b)
            r = rgb[0].coerceIn(0, 255)
            g = rgb[1].coerceIn(0, 255)
            b = rgb[2].coerceIn(0, 255)

            pixels[i] = (a shl 24) or (r shl 16) or (g shl 8) or b
        }

        identity.setPixels(pixels, 0, size, 0, 0, size, size)
        return identity
    }

    @Suppress("LongMethod")
    private fun applyPresetTransform(preset: LutPreset, r: Int, g: Int, b: Int): IntArray {
        val rf = r / 255f
        val gf = g / 255f
        val bf = b / 255f
        var ro: Float
        var go: Float
        var bo: Float

        when (preset) {
            LutPreset.ROSY_GLOW -> {
                // Pink-tinted glow: R+5%, B+3%, midtone pink tint
                ro = rf * 1.05f + 0.01f
                go = gf * 0.98f
                bo = bf * 1.03f + 0.01f
            }
            LutPreset.PEACH_CREAM -> {
                // Warm peach: R+8%, G+4%, warm midtones
                ro = rf * 1.08f
                go = gf * 1.04f
                bo = bf * 0.96f
            }
            LutPreset.CLEAN_PORCELAIN -> {
                // Clean and bright: luminance+5%, saturation-10%, slight cool
                val lum = 0.299f * rf + 0.587f * gf + 0.114f * bf
                val lumBoosted = (lum * 1.05f).coerceAtMost(1f)
                ro = lerp(lumBoosted, rf * 1.05f, 0.90f) - 0.01f
                go = lerp(lumBoosted, gf * 1.05f, 0.90f)
                bo = lerp(lumBoosted, bf * 1.05f, 0.90f) + 0.02f
            }
            LutPreset.GOLDEN_HOUR -> {
                // Golden portrait: R+10%, G+6%, B-5%, warm highlights
                ro = rf * 1.10f
                go = gf * 1.06f
                bo = bf * 0.95f
            }
            LutPreset.FILM_VINTAGE -> {
                // Film look: shadow lift+15%, highlight fade, green tint
                ro = liftShadows(rf, 0.06f) * 0.97f
                go = liftShadows(gf, 0.06f) * 1.02f
                bo = liftShadows(bf, 0.06f) * 0.98f
                // Highlight fade
                ro = fadeHighlights(ro, 0.05f)
                go = fadeHighlights(go, 0.05f)
                bo = fadeHighlights(bo, 0.05f)
            }
            LutPreset.COOL_EDITORIAL -> {
                // Cool modern: B+8%, G+3%, saturation-5%
                val lum = 0.299f * rf + 0.587f * gf + 0.114f * bf
                ro = lerp(lum, rf, 0.95f) - 0.01f
                go = lerp(lum, gf, 0.95f) + gf * 0.03f
                bo = lerp(lum, bf, 0.95f) + bf * 0.08f
            }
            LutPreset.WARM_SUNSET -> {
                // Warm sunset: R+12%, orange tint, contrast+10%
                ro = applySCurve(rf * 1.12f, 0.10f)
                go = applySCurve(gf * 1.04f, 0.10f)
                bo = applySCurve(bf * 0.92f, 0.10f)
            }
            LutPreset.NATURAL_GLOW -> {
                // Natural glow: luminance+3%, saturation+5%, micro warm
                val lum = 0.299f * rf + 0.587f * gf + 0.114f * bf
                ro = lerp(lum * 1.03f, rf * 1.03f, 1.05f) + 0.005f
                go = lerp(lum * 1.03f, gf * 1.03f, 1.05f)
                bo = lerp(lum * 1.03f, bf * 1.03f, 1.05f) - 0.005f
            }
        }

        return intArrayOf(
            (ro * 255f).toInt(),
            (go * 255f).toInt(),
            (bo * 255f).toInt()
        )
    }

    // ========================================================================
    // 색상 변환 유틸리티
    // ========================================================================

    private fun lerp(a: Float, b: Float, t: Float): Float = a + (b - a) * t

    private fun liftShadows(value: Float, amount: Float): Float {
        // Lift dark values, leave highlights unchanged
        return value + amount * (1f - value)
    }

    private fun fadeHighlights(value: Float, amount: Float): Float {
        // Pull bright values toward mid-gray
        return if (value > 0.5f) value - amount * (value - 0.5f) else value
    }

    private fun applySCurve(value: Float, strength: Float): Float {
        // S-curve contrast enhancement
        val v = value.coerceIn(0f, 1f)
        val curved = v * v * (3f - 2f * v)  // smoothstep
        return lerp(v, curved, strength)
    }
}
