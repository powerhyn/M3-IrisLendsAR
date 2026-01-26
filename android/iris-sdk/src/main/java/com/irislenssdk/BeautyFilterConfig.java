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
 * 뷰티 필터 설정 클래스.
 *
 * <p>뷰티 카메라 효과(피부 스무딩, 소프트 포커스, 밝기 조절)를 위한 설정입니다.</p>
 *
 * <p>이 클래스는 JNI 네이티브 코드에서 직접 필드에 접근하므로,
 * 필드 이름과 타입을 변경하면 안 됩니다.</p>
 *
 * <p>빌더 패턴 사용 예:</p>
 * <pre>{@code
 * BeautyFilterConfig config = new BeautyFilterConfig.Builder()
 *     .enabled(true)
 *     .intensity(0.6f)
 *     .smoothing(0.5f)
 *     .brightness(1.1f)
 *     .softFocus(0.3f)
 *     .build();
 *
 * IrisLensSDK.setBeautyFilter(config);
 * }</pre>
 *
 * @author IrisLensSDK Team
 * @version 1.0.0
 */
public class BeautyFilterConfig {

    // ========================================================================
    // 필드 (JNI에서 직접 접근)
    // ========================================================================

    /**
     * 필터 활성화 여부.
     * false면 뷰티 필터가 적용되지 않습니다.
     */
    public boolean enabled;

    /**
     * 전체 필터 강도 (0.0 ~ 1.0).
     * 0.0 = 효과 없음, 1.0 = 최대 효과
     * 모든 개별 효과에 곱해지는 마스터 강도입니다.
     */
    public float intensity;

    /**
     * 피부 스무딩 정도 (0.0 ~ 1.0).
     * Bilateral Filter를 사용한 피부 스무딩 효과입니다.
     * 0.0 = 스무딩 없음, 1.0 = 최대 스무딩
     */
    public float smoothing;

    /**
     * 밝기 조절 (0.0 ~ 2.0).
     * 1.0 = 원본 밝기, 1.05 = 5% 밝게, 0.95 = 5% 어둡게
     */
    public float brightness;

    /**
     * 소프트 포커스 정도 (0.0 ~ 1.0).
     * Gaussian Blur를 사용한 소프트 글로우 효과입니다.
     * 0.0 = 효과 없음, 1.0 = 최대 소프트 포커스
     */
    public float softFocus;

    // ========================================================================
    // 기본값 상수
    // ========================================================================

    /** 기본 활성화 여부 */
    public static final boolean DEFAULT_ENABLED = true;
    /** 기본 전체 강도 */
    public static final float DEFAULT_INTENSITY = 0.5f;
    /** 기본 스무딩 정도 */
    public static final float DEFAULT_SMOOTHING = 0.5f;
    /** 기본 밝기 */
    public static final float DEFAULT_BRIGHTNESS = 1.05f;
    /** 기본 소프트 포커스 정도 */
    public static final float DEFAULT_SOFT_FOCUS = 0.3f;

    // ========================================================================
    // 생성자
    // ========================================================================

    /**
     * 기본 설정으로 BeautyFilterConfig를 생성합니다.
     */
    public BeautyFilterConfig() {
        setDefaults();
    }

    /**
     * 복사 생성자.
     *
     * @param other 복사할 BeautyFilterConfig
     */
    public BeautyFilterConfig(@NonNull BeautyFilterConfig other) {
        this.enabled = other.enabled;
        this.intensity = other.intensity;
        this.smoothing = other.smoothing;
        this.brightness = other.brightness;
        this.softFocus = other.softFocus;
    }

    // ========================================================================
    // 메서드
    // ========================================================================

    /**
     * 모든 필드를 기본값으로 설정합니다.
     */
    public void setDefaults() {
        enabled = DEFAULT_ENABLED;
        intensity = DEFAULT_INTENSITY;
        smoothing = DEFAULT_SMOOTHING;
        brightness = DEFAULT_BRIGHTNESS;
        softFocus = DEFAULT_SOFT_FOCUS;
    }

    /**
     * 설정 값의 유효성을 검증합니다.
     *
     * @return 모든 값이 유효하면 true
     */
    public boolean isValid() {
        return intensity >= 0.0f && intensity <= 1.0f
                && smoothing >= 0.0f && smoothing <= 1.0f
                && brightness >= 0.0f && brightness <= 2.0f
                && softFocus >= 0.0f && softFocus <= 1.0f;
    }

    /**
     * 값을 유효한 범위로 클램프합니다.
     */
    public void clamp() {
        intensity = Math.max(0.0f, Math.min(1.0f, intensity));
        smoothing = Math.max(0.0f, Math.min(1.0f, smoothing));
        brightness = Math.max(0.0f, Math.min(2.0f, brightness));
        softFocus = Math.max(0.0f, Math.min(1.0f, softFocus));
    }

    @NonNull
    @Override
    public String toString() {
        return "BeautyFilterConfig{" +
                "enabled=" + enabled +
                ", intensity=" + intensity +
                ", smoothing=" + smoothing +
                ", brightness=" + brightness +
                ", softFocus=" + softFocus +
                '}';
    }

    // ========================================================================
    // 빌더 클래스
    // ========================================================================

    /**
     * BeautyFilterConfig 빌더.
     *
     * <p>플루언트 API로 설정을 구성할 수 있습니다.</p>
     */
    public static class Builder {
        private final BeautyFilterConfig config;

        public Builder() {
            config = new BeautyFilterConfig();
        }

        public Builder(@NonNull BeautyFilterConfig base) {
            config = new BeautyFilterConfig(base);
        }

        /**
         * 필터 활성화 여부를 설정합니다.
         *
         * @param enabled 활성화 여부
         * @return this
         */
        public Builder enabled(boolean enabled) {
            config.enabled = enabled;
            return this;
        }

        /**
         * 전체 필터 강도를 설정합니다.
         *
         * @param intensity 강도 (0.0 ~ 1.0)
         * @return this
         */
        public Builder intensity(float intensity) {
            config.intensity = intensity;
            return this;
        }

        /**
         * 피부 스무딩 정도를 설정합니다.
         *
         * @param smoothing 스무딩 정도 (0.0 ~ 1.0)
         * @return this
         */
        public Builder smoothing(float smoothing) {
            config.smoothing = smoothing;
            return this;
        }

        /**
         * 밝기를 설정합니다.
         *
         * @param brightness 밝기 (0.0 ~ 2.0, 1.0 = 원본)
         * @return this
         */
        public Builder brightness(float brightness) {
            config.brightness = brightness;
            return this;
        }

        /**
         * 소프트 포커스 정도를 설정합니다.
         *
         * @param softFocus 소프트 포커스 (0.0 ~ 1.0)
         * @return this
         */
        public Builder softFocus(float softFocus) {
            config.softFocus = softFocus;
            return this;
        }

        /**
         * BeautyFilterConfig를 빌드합니다.
         *
         * @return 구성된 BeautyFilterConfig
         */
        public BeautyFilterConfig build() {
            config.clamp();
            return new BeautyFilterConfig(config);
        }
    }
}
