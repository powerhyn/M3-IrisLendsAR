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
    //
    // 모든 faceRect 좌표는 upright 프레임 기준 정규화 [0.0~1.0] 값이다 (ADR-0001 §7.1).
    // 코어 어댑터(deriveIrisResult)가 478점 메시 바운딩 박스를 정규화 좌표로 채우며,
    // JNI·데모 소비측 모두 정규화로 전달·소비한다. 픽셀로 환산하려면 frameWidth/frameHeight를
    // 곱한다. (W4-B3 정정: 이전 "픽셀" 주석은 문서 드리프트였음 — 런타임 값은 무변경.)
    // ========================================================================

    /**
     * 얼굴 바운딩 박스 X 좌표 (정규화, 0.0 ~ 1.0).
     */
    public float faceRectX;

    /**
     * 얼굴 바운딩 박스 Y 좌표 (정규화, 0.0 ~ 1.0).
     */
    public float faceRectY;

    /**
     * 얼굴 바운딩 박스 너비 (정규화, 0.0 ~ 1.0).
     */
    public float faceRectWidth;

    /**
     * 얼굴 바운딩 박스 높이 (정규화, 0.0 ~ 1.0).
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
    // Face Mesh (디버그/시각화용)
    // ========================================================================

    /**
     * Face Mesh 랜드마크 개수 (MediaPipe 표준).
     */
    public static final int FACE_MESH_LANDMARK_COUNT = 478;

    /**
     * Face Mesh 유효 여부.
     * true이면 faceMesh 배열에 유효한 데이터가 있습니다.
     */
    public boolean faceMeshValid;

    /**
     * Face Mesh 랜드마크 좌표 (정규화, 0.0 ~ 1.0).
     * 478개 랜드마크 × 3 좌표 (x, y, z) = 1434개 float.
     * 인덱스: [i*3] = x, [i*3+1] = y, [i*3+2] = z (i = 0~477)
     */
    public float[] faceMesh;

    // ========================================================================
    // Eye Refiner 메타데이터
    // ========================================================================

    /**
     * [W4-D 삭제 예정] 왼쪽 홍채 품질 점수 (0.0 ~ 1.0).
     * Eye Refiner 사용 시에만 유효합니다.
     * ④ W4-D에서 detector 전용 메타 삭제(ADR §6.2) — C++ types.h/sdk_api.h·JNI 매핑·
     * 골든 baseline 18벌과 동시 제거(골든 재캡처 동반). W4-B3는 C++ types.h 마커와 문서 정합까지.
     */
    public float irisQualityLeft;

    /**
     * [W4-D 삭제 예정] 오른쪽 홍채 품질 점수 (0.0 ~ 1.0).
     * Eye Refiner 사용 시에만 유효합니다.
     * ④ W4-D에서 detector 전용 메타 삭제(ADR §6.2). W4-B3는 문서 정합까지.
     */
    public float irisQualityRight;

    /**
     * 왼쪽 눈꺼풀 가림 비율 (0.0 ~ 1.0).
     * 향후 구현 예정 (W3).
     */
    public float eyelidRatioLeft;

    /**
     * 오른쪽 눈꺼풀 가림 비율 (0.0 ~ 1.0).
     * 향후 구현 예정 (W3).
     */
    public float eyelidRatioRight;

    /**
     * [W4-D 삭제 예정] Eye Refiner 사용 여부.
     * true이면 2차 정밀화가 적용된 결과입니다.
     * ④ W4-D에서 detector 전용 메타 삭제(ADR §6.2). W4-B3는 C++ types.h 마커와 문서 정합까지.
     */
    public boolean eyeRefinerUsed;

    /**
     * 왼쪽 홍채 ROI 실측 평균 luma (P7-W2, srgb²+Rec.709 linear, 0~1).
     * 미측정/미검출 시 -1.0f. 디텍트→렌더 round-trip 시 보존되어 SDK가 소비합니다.
     */
    public float avgIrisLumaLeft;

    /**
     * 오른쪽 홍채 ROI 실측 평균 luma (P7-W2, srgb²+Rec.709 linear, 0~1).
     * 미측정/미검출 시 -1.0f.
     */
    public float avgIrisLumaRight;

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

        irisQualityLeft = 0.0f;
        irisQualityRight = 0.0f;
        eyelidRatioLeft = 0.0f;
        eyelidRatioRight = 0.0f;
        eyeRefinerUsed = false;

        // P7-W2: 미측정 sentinel(-1).
        avgIrisLumaLeft = -1.0f;
        avgIrisLumaRight = -1.0f;

        faceMeshValid = false;
        if (faceMesh == null) {
            faceMesh = new float[FACE_MESH_LANDMARK_COUNT * 3];
        }

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
     * 다른 IrisResult의 데이터를 깊은 복사합니다.
     * 스레드 간 불변 스냅샷 전달에 사용합니다.
     *
     * @param src 복사할 원본
     */
    public void copyFrom(@NonNull IrisResult src) {
        this.detected = src.detected;
        this.leftDetected = src.leftDetected;
        this.rightDetected = src.rightDetected;
        this.confidence = src.confidence;

        this.leftIrisX = src.leftIrisX;
        this.leftIrisY = src.leftIrisY;
        this.leftIrisZ = src.leftIrisZ;
        this.leftRadius = src.leftRadius;

        this.rightIrisX = src.rightIrisX;
        this.rightIrisY = src.rightIrisY;
        this.rightIrisZ = src.rightIrisZ;
        this.rightRadius = src.rightRadius;

        this.faceRectX = src.faceRectX;
        this.faceRectY = src.faceRectY;
        this.faceRectWidth = src.faceRectWidth;
        this.faceRectHeight = src.faceRectHeight;

        this.facePitch = src.facePitch;
        this.faceYaw = src.faceYaw;
        this.faceRoll = src.faceRoll;

        this.irisQualityLeft = src.irisQualityLeft;
        this.irisQualityRight = src.irisQualityRight;
        this.eyelidRatioLeft = src.eyelidRatioLeft;
        this.eyelidRatioRight = src.eyelidRatioRight;
        this.eyeRefinerUsed = src.eyeRefinerUsed;

        // P7-W2: iris ROI 실측 luma 보존.
        this.avgIrisLumaLeft = src.avgIrisLumaLeft;
        this.avgIrisLumaRight = src.avgIrisLumaRight;

        this.faceMeshValid = src.faceMeshValid;
        if (src.faceMesh != null) {
            if (this.faceMesh == null || this.faceMesh.length != src.faceMesh.length) {
                this.faceMesh = new float[src.faceMesh.length];
            }
            System.arraycopy(src.faceMesh, 0, this.faceMesh, 0, src.faceMesh.length);
        }

        this.timestampMs = src.timestampMs;
        this.frameWidth = src.frameWidth;
        this.frameHeight = src.frameHeight;
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
                ", eyeRefiner=" + eyeRefinerUsed +
                ", irisQuality=(" + irisQualityLeft + ", " + irisQualityRight + ")" +
                ", frame=" + frameWidth + "x" + frameHeight +
                ", timestamp=" + timestampMs +
                '}';
    }
}
