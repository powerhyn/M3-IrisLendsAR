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
}
