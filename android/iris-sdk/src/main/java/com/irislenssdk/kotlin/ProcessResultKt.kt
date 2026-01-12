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

/**
 * 프레임 처리 결과 (검출 + 렌더링).
 *
 * [IrisLensSDKKt.process] 호출 결과를 담습니다.
 *
 * 사용 예:
 * ```kotlin
 * sdk.process(frameData, width, height, config = myConfig)
 *     .onSuccess { result ->
 *         if (result.rendered) {
 *             // frameData가 렌즈 오버레이로 수정됨
 *             displayFrame(frameData)
 *         }
 *     }
 * ```
 *
 * @property irisResult 홍채 검출 결과
 * @property rendered 렌더링 수행 여부
 * @property renderTimeMs 렌더링 소요 시간 (밀리초)
 */
data class ProcessResultKt(
    val irisResult: IrisResultKt,
    val rendered: Boolean,
    val renderTimeMs: Long
) {
    /**
     * 검출 성공 여부.
     */
    val isDetected: Boolean
        get() = irisResult.isDetected

    /**
     * 검출 신뢰도.
     */
    val confidence: Float
        get() = irisResult.confidence

    /**
     * 전체 처리가 성공했는지 확인합니다.
     * 검출 성공 + 렌더링 성공
     */
    val isFullyProcessed: Boolean
        get() = isDetected && rendered

    companion object {
        /**
         * 실패 결과를 생성합니다.
         *
         * @return 빈 ProcessResultKt
         */
        fun failed(): ProcessResultKt = ProcessResultKt(
            irisResult = IrisResultKt.Empty,
            rendered = false,
            renderTimeMs = 0
        )

        /**
         * 검출만 성공한 결과를 생성합니다.
         *
         * @param irisResult 검출 결과
         * @return 렌더링되지 않은 ProcessResultKt
         */
        fun detectionOnly(irisResult: IrisResultKt): ProcessResultKt = ProcessResultKt(
            irisResult = irisResult,
            rendered = false,
            renderTimeMs = 0
        )
    }
}
