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
 * 렌즈 블렌드 모드 열거형.
 *
 * 렌즈 텍스처가 원본 이미지와 합성되는 방식을 정의합니다.
 *
 * @property value C API에 전달되는 정수 값
 */
enum class BlendMode(val value: Int) {
    /**
     * 일반 알파 블렌딩.
     *
     * 가장 기본적인 합성 방식으로, 렌즈의 알파 값에 따라 투명하게 합성됩니다.
     */
    NORMAL(JavaLensConfig.BLEND_NORMAL),

    /**
     * 곱하기 블렌딩.
     *
     * 어두운 색상이 강조되어 자연스러운 색조 변화를 제공합니다.
     * 컬러 렌즈에 적합합니다.
     */
    MULTIPLY(JavaLensConfig.BLEND_MULTIPLY),

    /**
     * 스크린 블렌딩.
     *
     * 밝은 색상이 강조되어 밝고 화사한 효과를 제공합니다.
     * 밝은 색상 렌즈에 적합합니다.
     */
    SCREEN(JavaLensConfig.BLEND_SCREEN),

    /**
     * 오버레이 블렌딩.
     *
     * 대비가 강화되어 선명한 효과를 제공합니다.
     * 패턴 렌즈에 적합합니다.
     */
    OVERLAY(JavaLensConfig.BLEND_OVERLAY),

    /**
     * 휘도 보존 틴트 (sRGB 근사).
     *
     * 원본 홍채의 밝기 패턴(줄무늬, 깊이, 명암)을 보존하면서 색상만 교체합니다.
     * 실제 콘택트렌즈의 물리적 원리에 가장 가까운 합성 방식입니다.
     *
     * @experimental 이 모드는 실험적이며 동작이 변경될 수 있습니다.
     */
    LUMINANCE_TINT(JavaLensConfig.BLEND_LUMINANCE_TINT),

    /**
     * 휘도 보존 틴트 (선형 색공간).
     *
     * sRGB 근사 대비 중간톤 정확도가 높고, specular highlight를 보존합니다.
     * GPU 부하가 약간 더 높습니다.
     *
     * @experimental 이 모드는 실험적이며 동작이 변경될 수 있습니다.
     */
    LUMINANCE_TINT_LINEAR(JavaLensConfig.BLEND_LUMINANCE_TINT_LINEAR),

    /**
     * 소프트 라이트 블렌딩.
     *
     * Photoshop Soft Light 공식 기반으로 부드러운 조명 효과를 제공합니다.
     * 은은한 색상 변화에 적합합니다.
     *
     * @experimental 이 모드는 실험적이며 동작이 변경될 수 있습니다.
     */
    SOFT_LIGHT(JavaLensConfig.BLEND_SOFT_LIGHT),

    /**
     * 색상 교체 블렌딩.
     *
     * 상대 밝기 정규화로 홍채 질감을 보존하면서 렌즈 색상을 완전히 발현합니다.
     * 어두운 홍채에서도 렌즈 디자이너 의도대로 색상이 나타납니다.
     *
     * @experimental 이 모드는 실험적이며 동작이 변경될 수 있습니다.
     */
    COLOR_REPLACE(JavaLensConfig.BLEND_COLOR_REPLACE);

    companion object {
        /**
         * 정수 값에서 BlendMode를 찾습니다.
         *
         * @param value 블렌드 모드 값
         * @return 해당 BlendMode, 없으면 [NORMAL]
         */
        fun fromValue(value: Int): BlendMode =
            entries.find { it.value == value } ?: NORMAL
    }
}
