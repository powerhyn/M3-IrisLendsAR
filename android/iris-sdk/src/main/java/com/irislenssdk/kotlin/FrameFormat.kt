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

import com.irislenssdk.IrisLensSDK as JavaIrisLensSDK

/**
 * 프레임 포맷 열거형.
 *
 * 지원하는 이미지 포맷을 정의합니다.
 * Android 카메라의 기본 포맷은 [NV21]입니다.
 *
 * @property value C API에 전달되는 정수 값
 */
enum class FrameFormat(val value: Int) {
    /** 32비트 RGBA */
    RGBA(JavaIrisLensSDK.FORMAT_RGBA),

    /** 32비트 BGRA */
    BGRA(JavaIrisLensSDK.FORMAT_BGRA),

    /** 24비트 RGB */
    RGB(JavaIrisLensSDK.FORMAT_RGB),

    /** 24비트 BGR */
    BGR(JavaIrisLensSDK.FORMAT_BGR),

    /** Android 카메라 기본 포맷 (YUV420sp) */
    NV21(JavaIrisLensSDK.FORMAT_NV21),

    /** iOS 카메라 포맷 (YUV420sp) */
    NV12(JavaIrisLensSDK.FORMAT_NV12),

    /** 8비트 그레이스케일 */
    GRAYSCALE(JavaIrisLensSDK.FORMAT_GRAY);

    companion object {
        /**
         * 정수 값에서 FrameFormat을 찾습니다.
         *
         * @param value 포맷 값
         * @return 해당 FrameFormat, 없으면 null
         */
        fun fromValue(value: Int): FrameFormat? =
            entries.find { it.value == value }

        /**
         * 포맷별 바이트 크기 계산.
         *
         * @param format 프레임 포맷
         * @param width 프레임 너비
         * @param height 프레임 높이
         * @return 필요한 바이트 크기
         */
        fun calculateBufferSize(format: FrameFormat, width: Int, height: Int): Int {
            return when (format) {
                RGBA, BGRA -> width * height * 4
                RGB, BGR -> width * height * 3
                NV21, NV12 -> width * height * 3 / 2
                GRAYSCALE -> width * height
            }
        }
    }
}
