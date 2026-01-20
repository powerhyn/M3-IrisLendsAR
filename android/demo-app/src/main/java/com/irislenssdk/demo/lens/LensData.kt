/**
 * IrisLensSDK Android - LensData
 *
 * 렌즈 데이터 모델 클래스
 *
 * @version 1.0.0
 */
package com.irislenssdk.demo.lens

import android.graphics.Bitmap

/**
 * 렌즈 데이터 클래스
 *
 * @property id 렌즈 고유 ID
 * @property name 렌즈 이름 (표시용)
 * @property fileName 원본 파일명
 * @property thumbnail 썸네일 비트맵 (미리보기용)
 * @property texture 렌즈 텍스처 비트맵 (렌더링용)
 */
data class LensData(
    val id: String,
    val name: String,
    val fileName: String,
    var thumbnail: Bitmap? = null,
    var texture: Bitmap? = null
) {
    companion object {
        /**
         * 파일명에서 표시용 이름 생성
         */
        fun createDisplayName(fileName: String): String {
            return fileName
                .removeSuffix(".png")
                .removeSuffix(".jpg")
                .replace("_", " ")
                .replace("-", " ")
                .replace(Regex("^\\d+\\.?\\s*"), "")  // 앞의 숫자 제거
                .replace(Regex("\\s+\\d+$"), "")      // 뒤의 숫자 제거
                .trim()
                .ifEmpty { fileName }
        }

        /**
         * 파일명에서 ID 생성
         */
        fun createId(fileName: String): String {
            return fileName
                .lowercase()
                .replace(Regex("[^a-z0-9]"), "_")
                .replace(Regex("_+"), "_")
                .trim('_')
        }
    }
}

/**
 * 특수 렌즈: "없음" 상태
 */
object NoLens {
    const val ID = "none"
    const val NAME = "없음"

    fun create(): LensData = LensData(
        id = ID,
        name = NAME,
        fileName = "",
        thumbnail = null,
        texture = null
    )
}
