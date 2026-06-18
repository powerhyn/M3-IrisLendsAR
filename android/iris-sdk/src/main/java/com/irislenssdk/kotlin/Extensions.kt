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

@file:JvmName("IrisExtensions")

package com.irislenssdk.kotlin

import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.media.Image
import java.nio.ByteBuffer

/**
 * IrisLensSDK Kotlin 확장 함수 모음.
 *
 * Android 카메라 및 이미지 처리를 위한 유틸리티 함수를 제공합니다.
 */

// ========================================================================
// Image 확장
// ========================================================================

/**
 * Camera2/CameraX Image를 ByteArray로 변환합니다.
 *
 * NV21 포맷을 가정합니다.
 *
 * @param reuseBuffer 재사용할 버퍼 (null이면 새로 할당)
 * @return NV21 형식의 ByteArray
 */
fun Image.toNv21ByteArray(reuseBuffer: ByteArray? = null): ByteArray {
    require(format == ImageFormat.YUV_420_888) {
        "Expected YUV_420_888 format, got $format"
    }

    val width = width
    val height = height
    val ySize = width * height
    val uvSize = width * height / 2
    val requiredSize = ySize + uvSize
    val nv21 = reuseBuffer?.takeIf { it.size >= requiredSize } ?: ByteArray(requiredSize)

    val yBuffer = planes[0].buffer
    val uBuffer = planes[1].buffer
    val vBuffer = planes[2].buffer

    val yRowStride = planes[0].rowStride
    val uvRowStride = planes[1].rowStride
    val uvPixelStride = planes[1].pixelStride

    // Y plane
    if (yRowStride == width) {
        yBuffer.get(nv21, 0, ySize)
    } else {
        var pos = 0
        for (row in 0 until height) {
            yBuffer.position(row * yRowStride)
            yBuffer.get(nv21, pos, width)
            pos += width
        }
    }

    // UV planes (interleaved as VU for NV21)
    var uvPos = ySize
    for (row in 0 until height / 2) {
        for (col in 0 until width / 2) {
            val uvIndex = row * uvRowStride + col * uvPixelStride
            nv21[uvPos++] = vBuffer.get(uvIndex) // V
            nv21[uvPos++] = uBuffer.get(uvIndex) // U
        }
    }

    return nv21
}

/**
 * Image에서 FrameFormat을 추론합니다.
 *
 * @return 추론된 FrameFormat, 지원하지 않는 포맷이면 null
 */
fun Image.toFrameFormat(): FrameFormat? {
    return when (format) {
        ImageFormat.YUV_420_888 -> FrameFormat.NV21
        ImageFormat.NV21 -> FrameFormat.NV21
        else -> null
    }
}

// ========================================================================
// Bitmap 확장
// ========================================================================

/**
 * Bitmap을 RGBA ByteArray로 변환합니다.
 *
 * @return RGBA 형식의 ByteArray
 * @throws IllegalArgumentException Bitmap config가 ARGB_8888이 아닌 경우
 */
fun Bitmap.toRgbaByteArray(): ByteArray {
    require(config == Bitmap.Config.ARGB_8888) {
        "Expected ARGB_8888 config, got $config. Use copy(Bitmap.Config.ARGB_8888, false) first."
    }
    val size = width * height * 4
    val buffer = ByteBuffer.allocate(size)
    copyPixelsToBuffer(buffer)
    return buffer.array()
}

// ========================================================================
// ByteArray 확장
// ========================================================================

/**
 * ByteArray가 주어진 프레임 크기에 유효한지 확인합니다.
 *
 * @param width 프레임 너비
 * @param height 프레임 높이
 * @param format 프레임 포맷
 * @return 유효하면 true
 */
fun ByteArray.isValidFrameSize(width: Int, height: Int, format: FrameFormat): Boolean {
    val expectedSize = FrameFormat.calculateBufferSize(format, width, height)
    return size >= expectedSize
}

// ========================================================================
// IrisResultKt 확장
// ========================================================================

/**
 * 두 결과의 홍채 위치 차이를 계산합니다.
 *
 * 추적 안정성 분석에 유용합니다.
 *
 * @param other 비교할 결과
 * @return 왼쪽, 오른쪽 홍채의 거리 차이 Pair
 */
fun IrisResultKt.distanceTo(other: IrisResultKt): Pair<Float, Float> {
    val leftDist = if (leftIris != null && other.leftIris != null) {
        val dx = leftIris.x - other.leftIris.x
        val dy = leftIris.y - other.leftIris.y
        kotlin.math.sqrt(dx * dx + dy * dy)
    } else Float.MAX_VALUE

    val rightDist = if (rightIris != null && other.rightIris != null) {
        val dx = rightIris.x - other.rightIris.x
        val dy = rightIris.y - other.rightIris.y
        kotlin.math.sqrt(dx * dx + dy * dy)
    } else Float.MAX_VALUE

    return leftDist to rightDist
}

/**
 * 이전 결과와의 변화가 임계값 이내인지 확인합니다.
 *
 * 갑작스러운 위치 변화 필터링에 유용합니다.
 *
 * @param previous 이전 결과
 * @param threshold 허용 거리 (정규화 좌표, 기본값 0.1)
 * @return 변화가 임계값 이내면 true
 */
fun IrisResultKt.isStableTo(previous: IrisResultKt, threshold: Float = 0.1f): Boolean {
    val (leftDist, rightDist) = distanceTo(previous)
    return leftDist <= threshold && rightDist <= threshold
}

// ========================================================================
// LensConfigKt 확장
// ========================================================================

/**
 * 애니메이션을 위한 보간 함수.
 *
 * @param target 목표 설정
 * @param fraction 보간 비율 (0.0 ~ 1.0)
 * @return 보간된 설정
 */
fun LensConfigKt.lerp(target: LensConfigKt, fraction: Float): LensConfigKt {
    val f = fraction.coerceIn(0f, 1f)
    return copy(
        opacity = opacity + (target.opacity - opacity) * f,
        scale = scale + (target.scale - scale) * f,
        offsetX = offsetX + (target.offsetX - offsetX) * f,
        offsetY = offsetY + (target.offsetY - offsetY) * f,
        edgeFeather = edgeFeather + (target.edgeFeather - edgeFeather) * f
    )
}
