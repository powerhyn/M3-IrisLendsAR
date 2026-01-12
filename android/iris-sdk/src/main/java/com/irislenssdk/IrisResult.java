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

package com.irislenssdk;

import androidx.annotation.NonNull;

/**
 * 홍채 검출 결과를 담는 데이터 클래스.
 *
 * <p>이 클래스는 JNI 네이티브 코드에서 직접 필드에 접근하므로,
 * 필드 이름과 타입을 변경하면 안 됩니다.</p>
 *
 * <p>사용 예:</p>
 * <pre>{@code
 * IrisResult result = new IrisResult();
 * int error = IrisLensSDK.detect(frameData, width, height, format, result);
 * if (error == IrisLensSDK.OK && result.detected) {
 *     float leftX = result.leftIrisX;
 *     float leftY = result.leftIrisY;
 *     // 홍채 위치 사용
 * }
 * }</pre>
 *
 * @author IrisLensSDK Team
 * @version 1.0.0
 */
public class IrisResult {

    // ========================================================================
    // 검출 상태
    // ========================================================================

    /**
     * 전체 검출 성공 여부.
     * 최소 한쪽 눈의 홍채가 검출되면 true.
     */
    public boolean detected;

    /**
     * 왼쪽 눈 홍채 검출 여부.
     */
    public boolean leftDetected;

    /**
     * 오른쪽 눈 홍채 검출 여부.
     */
    public boolean rightDetected;

    /**
     * 검출 신뢰도 (0.0 ~ 1.0).
     * 높을수록 검출 결과가 정확합니다.
     */
    public float confidence;

    // ========================================================================
    // 왼쪽 홍채 정보
    // ========================================================================

    /**
     * 왼쪽 홍채 중심 X 좌표 (정규화, 0.0 ~ 1.0).
     */
    public float leftIrisX;

    /**
     * 왼쪽 홍채 중심 Y 좌표 (정규화, 0.0 ~ 1.0).
     */
    public float leftIrisY;

    /**
     * 왼쪽 홍채 중심 Z 좌표 (깊이, 정규화).
     */
    public float leftIrisZ;

    /**
     * 왼쪽 홍채 반지름 (픽셀 단위).
     */
    public float leftRadius;

    // ========================================================================
    // 오른쪽 홍채 정보
    // ========================================================================

    /**
     * 오른쪽 홍채 중심 X 좌표 (정규화, 0.0 ~ 1.0).
     */
    public float rightIrisX;

    /**
     * 오른쪽 홍채 중심 Y 좌표 (정규화, 0.0 ~ 1.0).
     */
    public float rightIrisY;

    /**
     * 오른쪽 홍채 중심 Z 좌표 (깊이, 정규화).
     */
    public float rightIrisZ;

    /**
     * 오른쪽 홍채 반지름 (픽셀 단위).
     */
    public float rightRadius;

    // ========================================================================
    // 얼굴 영역 정보
    // ========================================================================

    /**
     * 얼굴 바운딩 박스 X 좌표 (픽셀).
     */
    public float faceRectX;

    /**
     * 얼굴 바운딩 박스 Y 좌표 (픽셀).
     */
    public float faceRectY;

    /**
     * 얼굴 바운딩 박스 너비 (픽셀).
     */
    public float faceRectWidth;

    /**
     * 얼굴 바운딩 박스 높이 (픽셀).
     */
    public float faceRectHeight;

    // ========================================================================
    // 얼굴 회전 정보
    // ========================================================================

    /**
     * 얼굴 Pitch 각도 (도, degree).
     * 위아래 기울기: 양수 = 위를 봄, 음수 = 아래를 봄
     */
    public float facePitch;

    /**
     * 얼굴 Yaw 각도 (도, degree).
     * 좌우 회전: 양수 = 왼쪽으로 회전, 음수 = 오른쪽으로 회전
     */
    public float faceYaw;

    /**
     * 얼굴 Roll 각도 (도, degree).
     * 기울기: 양수 = 시계 방향, 음수 = 반시계 방향
     */
    public float faceRoll;

    // ========================================================================
    // 프레임 정보
    // ========================================================================

    /**
     * 타임스탬프 (밀리초).
     */
    public long timestampMs;

    /**
     * 원본 프레임 너비 (픽셀).
     */
    public int frameWidth;

    /**
     * 원본 프레임 높이 (픽셀).
     */
    public int frameHeight;

    // ========================================================================
    // 생성자
    // ========================================================================

    /**
     * 기본 생성자.
     * 모든 필드를 기본값으로 초기화합니다.
     */
    public IrisResult() {
        reset();
    }

    // ========================================================================
    // 메서드
    // ========================================================================

    /**
     * 모든 필드를 기본값으로 초기화합니다.
     */
    public void reset() {
        detected = false;
        leftDetected = false;
        rightDetected = false;
        confidence = 0.0f;

        leftIrisX = 0.0f;
        leftIrisY = 0.0f;
        leftIrisZ = 0.0f;
        leftRadius = 0.0f;

        rightIrisX = 0.0f;
        rightIrisY = 0.0f;
        rightIrisZ = 0.0f;
        rightRadius = 0.0f;

        faceRectX = 0.0f;
        faceRectY = 0.0f;
        faceRectWidth = 0.0f;
        faceRectHeight = 0.0f;

        facePitch = 0.0f;
        faceYaw = 0.0f;
        faceRoll = 0.0f;

        timestampMs = 0;
        frameWidth = 0;
        frameHeight = 0;
    }

    /**
     * 왼쪽 홍채 중심의 픽셀 좌표를 반환합니다.
     *
     * @return [x, y] 픽셀 좌표 배열
     */
    public float[] getLeftIrisPixelCoords() {
        return new float[]{
                leftIrisX * frameWidth,
                leftIrisY * frameHeight
        };
    }

    /**
     * 오른쪽 홍채 중심의 픽셀 좌표를 반환합니다.
     *
     * @return [x, y] 픽셀 좌표 배열
     */
    public float[] getRightIrisPixelCoords() {
        return new float[]{
                rightIrisX * frameWidth,
                rightIrisY * frameHeight
        };
    }

    /**
     * 얼굴이 정면을 향하고 있는지 확인합니다.
     *
     * @param thresholdDegrees 허용 각도 (기본값 15도 권장)
     * @return 정면을 향하고 있으면 true
     */
    public boolean isFacingFront(float thresholdDegrees) {
        return Math.abs(facePitch) < thresholdDegrees
                && Math.abs(faceYaw) < thresholdDegrees
                && Math.abs(faceRoll) < thresholdDegrees;
    }

    /**
     * 얼굴이 정면을 향하고 있는지 확인합니다 (기본 15도 허용).
     *
     * @return 정면을 향하고 있으면 true
     */
    public boolean isFacingFront() {
        return isFacingFront(15.0f);
    }

    @NonNull
    @Override
    public String toString() {
        return "IrisResult{" +
                "detected=" + detected +
                ", leftDetected=" + leftDetected +
                ", rightDetected=" + rightDetected +
                ", confidence=" + confidence +
                ", leftIris=(" + leftIrisX + ", " + leftIrisY + ", " + leftIrisZ + ")" +
                ", leftRadius=" + leftRadius +
                ", rightIris=(" + rightIrisX + ", " + rightIrisY + ", " + rightIrisZ + ")" +
                ", rightRadius=" + rightRadius +
                ", faceRect=(" + faceRectX + ", " + faceRectY + ", " +
                faceRectWidth + ", " + faceRectHeight + ")" +
                ", faceRotation=(pitch=" + facePitch + ", yaw=" + faceYaw +
                ", roll=" + faceRoll + ")" +
                ", frame=" + frameWidth + "x" + frameHeight +
                ", timestamp=" + timestampMs +
                '}';
    }
}
