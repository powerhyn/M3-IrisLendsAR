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
 * 가상 렌즈 렌더링 설정 클래스.
 *
 * <p>이 클래스는 JNI 네이티브 코드에서 직접 필드에 접근하므로,
 * 필드 이름과 타입을 변경하면 안 됩니다.</p>
 *
 * <p>빌더 패턴 사용 예:</p>
 * <pre>{@code
 * LensConfig config = new LensConfig.Builder()
 *     .opacity(0.8f)
 *     .scale(1.1f)
 *     .blendMode(LensConfig.BLEND_MULTIPLY)
 *     .build();
 * }</pre>
 *
 * @author IrisLensSDK Team
 * @version 1.0.0
 */
public class LensConfig {

    // ========================================================================
    // 블렌드 모드 상수
    // ========================================================================

    /**
     * 일반 알파 블렌딩.
     * 가장 기본적인 합성 방식입니다.
     */
    public static final int BLEND_NORMAL = 0;

    /**
     * 곱하기 블렌딩.
     * 어두운 색상이 강조됩니다.
     */
    public static final int BLEND_MULTIPLY = 1;

    /**
     * 스크린 블렌딩.
     * 밝은 색상이 강조됩니다.
     */
    public static final int BLEND_SCREEN = 2;

    /**
     * 오버레이 블렌딩.
     * 대비가 강화됩니다.
     */
    public static final int BLEND_OVERLAY = 3;

    // ========================================================================
    // 필드 (JNI에서 직접 접근)
    // ========================================================================

    /**
     * 렌즈 투명도 (0.0 ~ 1.0).
     * 0.0 = 완전 투명, 1.0 = 완전 불투명
     */
    public float opacity;

    /**
     * 렌즈 크기 배율 (기본값 1.0).
     * 1.0보다 크면 렌즈가 확대됩니다.
     */
    public float scale;

    /**
     * X축 오프셋 (정규화, -1.0 ~ 1.0).
     * 렌즈 위치를 수평으로 조정합니다.
     */
    public float offsetX;

    /**
     * Y축 오프셋 (정규화, -1.0 ~ 1.0).
     * 렌즈 위치를 수직으로 조정합니다.
     */
    public float offsetY;

    /**
     * 블렌드 모드 (BLEND_* 상수 사용).
     */
    public int blendMode;

    /**
     * 가장자리 페더링 (0.0 ~ 1.0).
     * 렌즈 가장자리를 부드럽게 합니다.
     */
    public float edgeFeather;

    /**
     * 왼쪽 눈에 렌즈 적용 여부.
     */
    public boolean applyLeft;

    /**
     * 오른쪽 눈에 렌즈 적용 여부.
     */
    public boolean applyRight;

    // ========================================================================
    // 생성자
    // ========================================================================

    /**
     * 기본 설정으로 LensConfig를 생성합니다.
     */
    public LensConfig() {
        setDefaults();
    }

    /**
     * 복사 생성자.
     *
     * @param other 복사할 LensConfig
     */
    public LensConfig(@NonNull LensConfig other) {
        this.opacity = other.opacity;
        this.scale = other.scale;
        this.offsetX = other.offsetX;
        this.offsetY = other.offsetY;
        this.blendMode = other.blendMode;
        this.edgeFeather = other.edgeFeather;
        this.applyLeft = other.applyLeft;
        this.applyRight = other.applyRight;
    }

    // ========================================================================
    // 메서드
    // ========================================================================

    /**
     * 모든 필드를 기본값으로 설정합니다.
     */
    public void setDefaults() {
        opacity = 0.7f;
        scale = 1.0f;
        offsetX = 0.0f;
        offsetY = 0.0f;
        blendMode = BLEND_NORMAL;
        edgeFeather = 0.1f;
        applyLeft = true;
        applyRight = true;
    }

    /**
     * 설정 값의 유효성을 검증합니다.
     *
     * @return 모든 값이 유효하면 true
     */
    public boolean isValid() {
        return opacity >= 0.0f && opacity <= 1.0f
                && scale > 0.0f
                && offsetX >= -1.0f && offsetX <= 1.0f
                && offsetY >= -1.0f && offsetY <= 1.0f
                && blendMode >= BLEND_NORMAL && blendMode <= BLEND_OVERLAY
                && edgeFeather >= 0.0f && edgeFeather <= 1.0f;
    }

    /**
     * 값을 유효한 범위로 클램프합니다.
     */
    public void clamp() {
        opacity = Math.max(0.0f, Math.min(1.0f, opacity));
        scale = Math.max(0.1f, Math.min(3.0f, scale));
        offsetX = Math.max(-1.0f, Math.min(1.0f, offsetX));
        offsetY = Math.max(-1.0f, Math.min(1.0f, offsetY));
        blendMode = Math.max(BLEND_NORMAL, Math.min(BLEND_OVERLAY, blendMode));
        edgeFeather = Math.max(0.0f, Math.min(1.0f, edgeFeather));
    }

    @NonNull
    @Override
    public String toString() {
        return "LensConfig{" +
                "opacity=" + opacity +
                ", scale=" + scale +
                ", offset=(" + offsetX + ", " + offsetY + ")" +
                ", blendMode=" + getBlendModeName() +
                ", edgeFeather=" + edgeFeather +
                ", applyLeft=" + applyLeft +
                ", applyRight=" + applyRight +
                '}';
    }

    /**
     * 블렌드 모드 이름을 반환합니다.
     *
     * @return 블렌드 모드 이름
     */
    public String getBlendModeName() {
        switch (blendMode) {
            case BLEND_NORMAL:
                return "NORMAL";
            case BLEND_MULTIPLY:
                return "MULTIPLY";
            case BLEND_SCREEN:
                return "SCREEN";
            case BLEND_OVERLAY:
                return "OVERLAY";
            default:
                return "UNKNOWN";
        }
    }

    // ========================================================================
    // 빌더 클래스
    // ========================================================================

    /**
     * LensConfig 빌더.
     *
     * <p>플루언트 API로 설정을 구성할 수 있습니다.</p>
     */
    public static class Builder {
        private final LensConfig config;

        public Builder() {
            config = new LensConfig();
        }

        public Builder(@NonNull LensConfig base) {
            config = new LensConfig(base);
        }

        /**
         * 투명도를 설정합니다.
         *
         * @param opacity 투명도 (0.0 ~ 1.0)
         * @return this
         */
        public Builder opacity(float opacity) {
            config.opacity = opacity;
            return this;
        }

        /**
         * 크기 배율을 설정합니다.
         *
         * @param scale 배율 (기본값 1.0)
         * @return this
         */
        public Builder scale(float scale) {
            config.scale = scale;
            return this;
        }

        /**
         * X축 오프셋을 설정합니다.
         *
         * @param offsetX 오프셋 (-1.0 ~ 1.0)
         * @return this
         */
        public Builder offsetX(float offsetX) {
            config.offsetX = offsetX;
            return this;
        }

        /**
         * Y축 오프셋을 설정합니다.
         *
         * @param offsetY 오프셋 (-1.0 ~ 1.0)
         * @return this
         */
        public Builder offsetY(float offsetY) {
            config.offsetY = offsetY;
            return this;
        }

        /**
         * 오프셋을 한번에 설정합니다.
         *
         * @param x X축 오프셋
         * @param y Y축 오프셋
         * @return this
         */
        public Builder offset(float x, float y) {
            config.offsetX = x;
            config.offsetY = y;
            return this;
        }

        /**
         * 블렌드 모드를 설정합니다.
         *
         * @param blendMode BLEND_* 상수
         * @return this
         */
        public Builder blendMode(int blendMode) {
            config.blendMode = blendMode;
            return this;
        }

        /**
         * 가장자리 페더링을 설정합니다.
         *
         * @param edgeFeather 페더링 (0.0 ~ 1.0)
         * @return this
         */
        public Builder edgeFeather(float edgeFeather) {
            config.edgeFeather = edgeFeather;
            return this;
        }

        /**
         * 왼쪽 눈 적용 여부를 설정합니다.
         *
         * @param apply 적용 여부
         * @return this
         */
        public Builder applyLeft(boolean apply) {
            config.applyLeft = apply;
            return this;
        }

        /**
         * 오른쪽 눈 적용 여부를 설정합니다.
         *
         * @param apply 적용 여부
         * @return this
         */
        public Builder applyRight(boolean apply) {
            config.applyRight = apply;
            return this;
        }

        /**
         * 양쪽 눈 적용 여부를 한번에 설정합니다.
         *
         * @param left 왼쪽 눈 적용 여부
         * @param right 오른쪽 눈 적용 여부
         * @return this
         */
        public Builder apply(boolean left, boolean right) {
            config.applyLeft = left;
            config.applyRight = right;
            return this;
        }

        /**
         * LensConfig를 빌드합니다.
         *
         * @return 구성된 LensConfig
         */
        public LensConfig build() {
            config.clamp();
            return new LensConfig(config);
        }
    }
}
