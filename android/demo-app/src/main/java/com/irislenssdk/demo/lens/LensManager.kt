/**
 * IrisLensSDK Android - LensManager
 *
 * 렌즈 텍스처 로딩 및 관리
 *
 * @version 1.0.0
 */
package com.irislenssdk.demo.lens

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import android.util.LruCache
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.IOException

/**
 * 렌즈 관리자
 *
 * assets/lenses/ 폴더에서 렌즈 이미지를 로딩하고 관리
 */
class LensManager(private val context: Context) {

    companion object {
        private const val TAG = "LensManager"
        private const val LENS_ASSETS_PATH = "lenses"
        private const val THUMBNAIL_SIZE = 128
        private const val TEXTURE_SIZE = 512  // 렌더링용 텍스처 크기

        // 메모리 캐시 크기 (MB)
        private const val CACHE_SIZE_MB = 32
    }

    // 렌즈 목록
    private val lensList = mutableListOf<LensData>()

    // 텍스처 캐시 (LRU)
    private val textureCache: LruCache<String, Bitmap>

    // 현재 선택된 렌즈
    var currentLens: LensData? = null
        private set

    // 선택 변경 리스너
    var onLensChangedListener: ((LensData?) -> Unit)? = null

    init {
        // LRU 캐시 초기화 (최대 CACHE_SIZE_MB)
        val maxMemory = (Runtime.getRuntime().maxMemory() / 1024).toInt()
        val cacheSize = minOf(CACHE_SIZE_MB * 1024, maxMemory / 4)

        textureCache = object : LruCache<String, Bitmap>(cacheSize) {
            override fun sizeOf(key: String, bitmap: Bitmap): Int {
                return bitmap.byteCount / 1024
            }
        }
    }

    /**
     * assets에서 렌즈 목록 로드
     *
     * @return 로드된 렌즈 목록
     */
    suspend fun loadLensesFromAssets(): List<LensData> = withContext(Dispatchers.IO) {
        lensList.clear()

        // "없음" 렌즈 먼저 추가
        lensList.add(NoLens.create())

        try {
            val assetManager = context.assets
            val files = assetManager.list(LENS_ASSETS_PATH)

            files?.filter { it.endsWith(".png", ignoreCase = true) || it.endsWith(".jpg", ignoreCase = true) }
                ?.sorted()
                ?.forEach { fileName ->
                    val id = LensData.createId(fileName)
                    val name = LensData.createDisplayName(fileName)

                    val lensData = LensData(
                        id = id,
                        name = name,
                        fileName = fileName
                    )

                    // 썸네일 로드
                    lensData.thumbnail = loadThumbnail(fileName)

                    lensList.add(lensData)
                    Log.d(TAG, "Loaded lens: $name ($fileName)")
                }

            Log.i(TAG, "Total ${lensList.size} lenses loaded (including 'none')")

        } catch (e: IOException) {
            Log.e(TAG, "Failed to load lenses from assets", e)
        }

        lensList.toList()
    }

    /**
     * 썸네일 로드 (작은 크기로 리사이즈)
     */
    private fun loadThumbnail(fileName: String): Bitmap? {
        return try {
            val options = BitmapFactory.Options().apply {
                inJustDecodeBounds = true
            }

            // 이미지 크기 확인
            context.assets.open("$LENS_ASSETS_PATH/$fileName").use { stream ->
                BitmapFactory.decodeStream(stream, null, options)
            }

            // 샘플 크기 계산
            options.inSampleSize = calculateInSampleSize(options, THUMBNAIL_SIZE, THUMBNAIL_SIZE)
            options.inJustDecodeBounds = false

            // 실제 로드
            context.assets.open("$LENS_ASSETS_PATH/$fileName").use { stream ->
                BitmapFactory.decodeStream(stream, null, options)?.let { bitmap ->
                    // 정확한 크기로 리사이즈
                    Bitmap.createScaledBitmap(bitmap, THUMBNAIL_SIZE, THUMBNAIL_SIZE, true).also {
                        if (it != bitmap) bitmap.recycle()
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load thumbnail: $fileName", e)
            null
        }
    }

    /**
     * 렌더링용 텍스처 로드 (캐시 사용)
     */
    fun getTexture(lens: LensData): Bitmap? {
        if (lens.id == NoLens.ID) return null

        // 캐시 확인
        textureCache.get(lens.id)?.let { return it }

        // 캐시 미스 - 로드
        return loadTexture(lens.fileName)?.also { bitmap ->
            textureCache.put(lens.id, bitmap)
            lens.texture = bitmap
        }
    }

    /**
     * 텍스처 로드 (렌더링용 크기)
     */
    private fun loadTexture(fileName: String): Bitmap? {
        return try {
            val options = BitmapFactory.Options().apply {
                inJustDecodeBounds = true
            }

            context.assets.open("$LENS_ASSETS_PATH/$fileName").use { stream ->
                BitmapFactory.decodeStream(stream, null, options)
            }

            options.inSampleSize = calculateInSampleSize(options, TEXTURE_SIZE, TEXTURE_SIZE)
            options.inJustDecodeBounds = false
            options.inPreferredConfig = Bitmap.Config.ARGB_8888
            // NOTE: inPremultiplied=false는 createScaledBitmap(Canvas)과 호환 불가
            // → ISS-005 EXP-A 보정은 셰이더 unpremultiply로 대체

            context.assets.open("$LENS_ASSETS_PATH/$fileName").use { stream ->
                BitmapFactory.decodeStream(stream, null, options)?.let { bitmap ->
                    Bitmap.createScaledBitmap(bitmap, TEXTURE_SIZE, TEXTURE_SIZE, true).also {
                        if (it != bitmap) bitmap.recycle()
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load texture: $fileName", e)
            null
        }
    }

    /**
     * 샘플 크기 계산
     */
    private fun calculateInSampleSize(
        options: BitmapFactory.Options,
        reqWidth: Int,
        reqHeight: Int
    ): Int {
        val (height: Int, width: Int) = options.outHeight to options.outWidth
        var inSampleSize = 1

        if (height > reqHeight || width > reqWidth) {
            val halfHeight: Int = height / 2
            val halfWidth: Int = width / 2

            while (halfHeight / inSampleSize >= reqHeight && halfWidth / inSampleSize >= reqWidth) {
                inSampleSize *= 2
            }
        }

        return inSampleSize
    }

    /**
     * 렌즈 선택
     */
    fun selectLens(lens: LensData) {
        val previousLens = currentLens
        currentLens = lens

        // 텍스처 미리 로드
        if (lens.id != NoLens.ID) {
            lens.texture = getTexture(lens)
        }

        Log.d(TAG, "Lens selected: ${lens.name} (${lens.id})")

        if (previousLens?.id != lens.id) {
            onLensChangedListener?.invoke(lens)
        }
    }

    /**
     * 렌즈 선택 (ID로)
     */
    fun selectLensById(id: String) {
        val lens = lensList.find { it.id == id }
        if (lens != null) {
            selectLens(lens)
        } else {
            Log.w(TAG, "Lens not found: $id")
        }
    }

    /**
     * "없음" 선택
     */
    fun clearLens() {
        selectLens(NoLens.create())
    }

    /**
     * 렌즈 목록 반환
     */
    fun getLensList(): List<LensData> = lensList.toList()

    /**
     * 리소스 해제
     */
    fun release() {
        textureCache.evictAll()
        lensList.forEach { lens ->
            lens.thumbnail?.recycle()
            lens.texture?.recycle()
        }
        lensList.clear()
        currentLens = null
        Log.d(TAG, "LensManager released")
    }
}
