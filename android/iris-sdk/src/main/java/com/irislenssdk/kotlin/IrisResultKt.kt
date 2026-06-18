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

import android.graphics.PointF
import android.graphics.RectF
import com.irislenssdk.IrisResult as JavaIrisResult
import kotlin.math.abs

/**
 * 홍채 랜드마크 좌표.
 *
 * 정규화된 좌표 (0.0 ~ 1.0)를 사용합니다.
 *
 * @property x X 좌표 (정규화)
 * @property y Y 좌표 (정규화)
 * @property z Z 좌표 (깊이, 정규화)
 */
data class IrisLandmark(
    val x: Float,
    val y: Float,
    val z: Float = 0f
) {
    /**
     * 주어진 프레임 크기에서 픽셀 좌표를 계산합니다.
     *
     * @param frameWidth 프레임 너비 (픽셀)
     * @param frameHeight 프레임 높이 (픽셀)
     * @return 픽셀 좌표 PointF
     */
    fun toPixel(frameWidth: Int, frameHeight: Int): PointF {
        return PointF(x * frameWidth, y * frameHeight)
    }
}

/**
 * 얼굴 회전 정보.
 *
 * 오일러 각도를 사용합니다 (도, degree).
 *
 * @property pitch 위아래 기울기 (양수 = 위, 음수 = 아래)
 * @property yaw 좌우 회전 (양수 = 왼쪽, 음수 = 오른쪽)
 * @property roll 기울기 (양수 = 시계방향, 음수 = 반시계방향)
 */
data class FaceRotation(
    val pitch: Float,
    val yaw: Float,
    val roll: Float
) {
    /**
     * 얼굴이 정면을 향하고 있는지 확인합니다.
     *
     * @param thresholdDegrees 허용 각도 (기본값 15도)
     * @return 정면을 향하고 있으면 true
     */
    fun isFacingFront(thresholdDegrees: Float = 15f): Boolean {
        return abs(pitch) < thresholdDegrees &&
                abs(yaw) < thresholdDegrees &&
                abs(roll) < thresholdDegrees
    }
}

/**
 * Kotlin 홍채 검출 결과 데이터 클래스.
 *
 * 불변 데이터 클래스로 스레드 안전합니다.
 *
 * 사용 예:
 * ```kotlin
 * sdk.detect(frameData, width, height)
 *     .onSuccess { result ->
 *         if (result.isDetected) {
 *             val leftCenter = result.leftIris?.toPixel(width, height)
 *             // 렌즈 오버레이 위치 계산
 *         }
 *     }
 * ```
 *
 * @property isDetected 전체 검출 성공 여부
 * @property confidence 검출 신뢰도 (0.0 ~ 1.0)
 * @property leftDetected 왼쪽 눈 검출 여부
 * @property rightDetected 오른쪽 눈 검출 여부
 * @property leftIris 왼쪽 홍채 중심 좌표
 * @property rightIris 오른쪽 홍채 중심 좌표
 * @property leftRadius 왼쪽 홍채 반지름 (픽셀)
 * @property rightRadius 오른쪽 홍채 반지름 (픽셀)
 * @property faceRect 얼굴 바운딩 박스
 * @property faceRotation 얼굴 회전 정보
 * @property frameWidth 프레임 너비 (픽셀)
 * @property frameHeight 프레임 높이 (픽셀)
 * @property timestampMs 타임스탬프 (밀리초)
 * @property eyelidRatioLeft 왼쪽 눈꺼풀 가림 비율 (W3용 별도 트랙)
 * @property eyelidRatioRight 오른쪽 눈꺼풀 가림 비율 (W3용 별도 트랙)
 * @property avgIrisLumaLeft 왼쪽 홍채 ROI 평균 linear luma (P7-W2 렌즈 색 적응, srgb²·Rec.709, -1=미측정)
 * @property avgIrisLumaRight 오른쪽 홍채 ROI 평균 linear luma (-1=미측정)
 *
 * 부재 필드(의도적 — Java IrisResult/C++에는 존재, Kotlin 경량 DTO는 미포함):
 * - faceMesh[478]: 디버그/시각화용 코어·JNI 보유 필드. Kotlin 호출자 요구 발생 시 데모 렌더 로직과
 *   함께 재검토. (avgIrisLumaLeft/Right는 ④ W4-B4에서 SDK AAR 계약으로 승격되어 아래에 노출됨.)
 */
data class IrisResultKt(
    val isDetected: Boolean,
    val confidence: Float,
    val leftDetected: Boolean,
    val rightDetected: Boolean,
    val leftIris: IrisLandmark?,
    val rightIris: IrisLandmark?,
    val leftRadius: Float,
    val rightRadius: Float,
    val faceRect: RectF?,
    val faceRotation: FaceRotation,
    val frameWidth: Int,
    val frameHeight: Int,
    val timestampMs: Long,
    val eyelidRatioLeft: Float = 0f,   // W3용 별도 트랙
    val eyelidRatioRight: Float = 0f,  // W3용 별도 트랙
    // ④ W4-B4: P7-W2 홍채 ROI 평균 linear luma (렌즈 색 적응). -1=미측정. SDK AAR 계약으로 승격.
    val avgIrisLumaLeft: Float = -1f,
    val avgIrisLumaRight: Float = -1f
) {
    /**
     * 양쪽 눈 모두 검출되었는지 확인합니다.
     */
    val bothEyesDetected: Boolean
        get() = leftDetected && rightDetected

    /**
     * 최소 한쪽 눈이 검출되었는지 확인합니다.
     */
    val anyEyeDetected: Boolean
        get() = leftDetected || rightDetected

    /**
     * 얼굴이 정면을 향하고 있는지 확인합니다.
     *
     * @param thresholdDegrees 허용 각도 (기본값 15도)
     * @return 정면을 향하고 있으면 true
     */
    fun isFacingFront(thresholdDegrees: Float = 15f): Boolean {
        return faceRotation.isFacingFront(thresholdDegrees)
    }

    /**
     * 왼쪽 홍채 픽셀 좌표를 반환합니다.
     *
     * @return 픽셀 좌표, 검출되지 않았으면 null
     */
    fun getLeftIrisPixel(): PointF? {
        return leftIris?.toPixel(frameWidth, frameHeight)
    }

    /**
     * 오른쪽 홍채 픽셀 좌표를 반환합니다.
     *
     * @return 픽셀 좌표, 검출되지 않았으면 null
     */
    fun getRightIrisPixel(): PointF? {
        return rightIris?.toPixel(frameWidth, frameHeight)
    }

    companion object {
        /** 빈 결과 (검출 실패) */
        val Empty = IrisResultKt(
            isDetected = false,
            confidence = 0f,
            leftDetected = false,
            rightDetected = false,
            leftIris = null,
            rightIris = null,
            leftRadius = 0f,
            rightRadius = 0f,
            faceRect = null,
            faceRotation = FaceRotation(0f, 0f, 0f),
            frameWidth = 0,
            frameHeight = 0,
            timestampMs = 0,
            eyelidRatioLeft = 0f,
            eyelidRatioRight = 0f,
            avgIrisLumaLeft = -1f,
            avgIrisLumaRight = -1f
        )

        /**
         * Java IrisResult에서 변환합니다.
         *
         * @param java Java IrisResult
         * @return 동일한 데이터의 IrisResultKt
         */
        fun fromJava(java: JavaIrisResult): IrisResultKt {
            val leftIris = if (java.leftDetected) {
                IrisLandmark(java.leftIrisX, java.leftIrisY, java.leftIrisZ)
            } else null

            val rightIris = if (java.rightDetected) {
                IrisLandmark(java.rightIrisX, java.rightIrisY, java.rightIrisZ)
            } else null

            val faceRect = if (java.faceRectWidth > 0 && java.faceRectHeight > 0) {
                RectF(
                    java.faceRectX,
                    java.faceRectY,
                    java.faceRectX + java.faceRectWidth,
                    java.faceRectY + java.faceRectHeight
                )
            } else null

            return IrisResultKt(
                isDetected = java.detected,
                confidence = java.confidence,
                leftDetected = java.leftDetected,
                rightDetected = java.rightDetected,
                leftIris = leftIris,
                rightIris = rightIris,
                leftRadius = java.leftRadius,
                rightRadius = java.rightRadius,
                faceRect = faceRect,
                faceRotation = FaceRotation(
                    pitch = java.facePitch,
                    yaw = java.faceYaw,
                    roll = java.faceRoll
                ),
                frameWidth = java.frameWidth,
                frameHeight = java.frameHeight,
                timestampMs = java.timestampMs,
                eyelidRatioLeft = java.eyelidRatioLeft,
                eyelidRatioRight = java.eyelidRatioRight,
                avgIrisLumaLeft = java.avgIrisLumaLeft,
                avgIrisLumaRight = java.avgIrisLumaRight
            )
        }
    }
}
