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
 * 확장된 뷰티 필터 설정 클래스 (V2).
 *
 * <p>기본 피부 효과 + 고급 효과 + 얼굴 형태 보정을 지원합니다.</p>
 *
 * <p>이 클래스는 JNI 네이티브 코드에서 직접 필드에 접근하므로,
 * 필드 이름과 타입을 변경하면 안 됩니다.</p>
 *
 * <p>빌더 패턴 사용 예:</p>
 * <pre>{@code
 * BeautyFilterConfigV2 config = new BeautyFilterConfigV2.Builder()
 *     .enabled(true)
 *     .intensity(0.6f)
 *     .smoothing(0.5f)
 *     .whitening(0.3f)
 *     .slimFace(0.2f)
 *     .build();
 *
 * IrisLensSDK.setBeautyFilterV2(config);
 * }</pre>
 *
 * @author IrisLensSDK Team
 * @version 2.0.0
 */
public class BeautyFilterConfigV2 {

    // ========================================================================
    // 기본 설정 (V1 호환)
    // ========================================================================

    /**
     * 필터 활성화 여부.
     * false면 뷰티 필터가 적용되지 않습니다.
     */
    public boolean enabled;

    /**
     * 전체 필터 강도 (0.0 ~ 1.0).
     * 모든 개별 효과에 곱해지는 마스터 강도입니다.
     */
    public float intensity;

    // ========================================================================
    // 피부 효과
    // ========================================================================

    /**
     * 피부 스무딩 정도 (0.0 ~ 1.0).
     * Bilateral Filter를 사용한 피부 스무딩 효과입니다.
     */
    public float smoothing;

    /**
     * 밝기 조절 (0.5 ~ 1.5).
     * 1.0 = 원본 밝기
     */
    public float brightness;

    /**
     * 소프트 포커스 정도 (0.0 ~ 1.0).
     * Gaussian Blur를 사용한 소프트 글로우 효과입니다.
     */
    public float softFocus;

    /**
     * 피부톤 화이트닝 (0.0 ~ 1.0).
     * 피부톤을 밝게 만드는 효과입니다.
     */
    public float whitening;

    /**
     * 컬러 밸런스 (-1.0 ~ 1.0).
     * 음수 = 쿨톤, 양수 = 웜톤
     */
    public float colorBalance;

    /**
     * 주름 제거 (0.0 ~ 1.0).
     * 주름 영역을 부드럽게 처리합니다.
     */
    public float wrinkleRemove;

    /**
     * 피부 품질 개선 (0.0 ~ 1.0).
     * Frequency Separation 기반 고급 피부 보정을 활성화합니다.
     * 0.0 = 비활성 (기존 Bilateral 경로), 0.0 초과 = Freq Sep 활성화
     */
    public float skinQuality;

    // ========================================================================
    // 얼굴 형태 보정
    // ========================================================================

    /**
     * 얼굴 슬림화 (0.0 ~ 1.0).
     * V-라인 효과를 적용합니다.
     */
    public float slimFace;

    /**
     * 눈 확대 (0.0 ~ 1.0).
     * 눈을 자연스럽게 확대합니다.
     */
    public float enlargeEyes;

    /**
     * 턱 축소 (0.0 ~ 1.0).
     * 턱 영역을 슬림하게 만듭니다.
     */
    public float thinChin;

    // ========================================================================
    // 처리 옵션
    // ========================================================================

    /**
     * GPU 가속 사용 여부.
     * true면 OpenGL ES를 사용하여 처리 속도를 향상시킵니다.
     */
    public boolean useGpu;

    /**
     * 얼굴 영역만 처리 여부.
     * true면 ROI 기반으로 얼굴 영역만 처리하여 성능을 향상시킵니다.
     */
    public boolean roiOnly;

    /**
     * 눈 영역 보호 여부.
     * true면 눈 영역에는 스무딩을 적용하지 않아 선명함을 유지합니다.
     */
    public boolean protectEyes;

    /**
     * 입술 영역 보호 여부.
     * true면 입술 영역에는 스무딩을 적용하지 않습니다.
     */
    public boolean protectLips;

    /**
     * 다운스케일 팩터 (1, 2, 4).
     * 1 = 원본, 2 = 1/2 해상도, 4 = 1/4 해상도
     * 성능과 품질 트레이드오프를 조절합니다.
     */
    public int downscaleFactor;

    // ========================================================================
    // 기본값 상수
    // ========================================================================

    public static final boolean DEFAULT_ENABLED = false;
    public static final float DEFAULT_INTENSITY = 0.5f;
    public static final float DEFAULT_SMOOTHING = 0.5f;
    public static final float DEFAULT_BRIGHTNESS = 1.0f;
    public static final float DEFAULT_SOFT_FOCUS = 0.3f;
    public static final float DEFAULT_WHITENING = 0.0f;
    public static final float DEFAULT_COLOR_BALANCE = 0.0f;
    public static final float DEFAULT_WRINKLE_REMOVE = 0.0f;
    public static final float DEFAULT_SKIN_QUALITY = 0.0f;
    public static final float DEFAULT_SLIM_FACE = 0.0f;
    public static final float DEFAULT_ENLARGE_EYES = 0.0f;
    public static final float DEFAULT_THIN_CHIN = 0.0f;
    public static final boolean DEFAULT_USE_GPU = true;
    public static final boolean DEFAULT_ROI_ONLY = true;
    public static final boolean DEFAULT_PROTECT_EYES = true;
    public static final boolean DEFAULT_PROTECT_LIPS = true;
    public static final int DEFAULT_DOWNSCALE_FACTOR = 1;

    // ========================================================================
    // 생성자
    // ========================================================================

    /**
     * 기본 설정으로 BeautyFilterConfigV2를 생성합니다.
     */
    public BeautyFilterConfigV2() {
        setDefaults();
    }

    /**
     * 복사 생성자.
     *
     * @param other 복사할 BeautyFilterConfigV2
     */
    public BeautyFilterConfigV2(@NonNull BeautyFilterConfigV2 other) {
        this.enabled = other.enabled;
        this.intensity = other.intensity;
        this.smoothing = other.smoothing;
        this.brightness = other.brightness;
        this.softFocus = other.softFocus;
        this.whitening = other.whitening;
        this.colorBalance = other.colorBalance;
        this.wrinkleRemove = other.wrinkleRemove;
        this.skinQuality = other.skinQuality;
        this.slimFace = other.slimFace;
        this.enlargeEyes = other.enlargeEyes;
        this.thinChin = other.thinChin;
        this.useGpu = other.useGpu;
        this.roiOnly = other.roiOnly;
        this.protectEyes = other.protectEyes;
        this.protectLips = other.protectLips;
        this.downscaleFactor = other.downscaleFactor;
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
        whitening = DEFAULT_WHITENING;
        colorBalance = DEFAULT_COLOR_BALANCE;
        wrinkleRemove = DEFAULT_WRINKLE_REMOVE;
        skinQuality = DEFAULT_SKIN_QUALITY;
        slimFace = DEFAULT_SLIM_FACE;
        enlargeEyes = DEFAULT_ENLARGE_EYES;
        thinChin = DEFAULT_THIN_CHIN;
        useGpu = DEFAULT_USE_GPU;
        roiOnly = DEFAULT_ROI_ONLY;
        protectEyes = DEFAULT_PROTECT_EYES;
        protectLips = DEFAULT_PROTECT_LIPS;
        downscaleFactor = DEFAULT_DOWNSCALE_FACTOR;
    }

    /**
     * V1 설정으로부터 V2 설정을 생성합니다.
     *
     * @param v1 변환할 V1 설정
     * @return V2 설정
     */
    @NonNull
    public static BeautyFilterConfigV2 fromV1(@NonNull BeautyFilterConfig v1) {
        BeautyFilterConfigV2 v2 = new BeautyFilterConfigV2();
        v2.enabled = v1.enabled;
        v2.intensity = v1.intensity;
        v2.smoothing = v1.smoothing;
        v2.brightness = v1.brightness;
        v2.softFocus = v1.softFocus;
        // V2 전용 필드는 기본값 유지
        return v2;
    }

    /**
     * V1 설정으로 변환합니다 (피부 효과만).
     *
     * @return V1 설정
     */
    @NonNull
    public BeautyFilterConfig toV1() {
        BeautyFilterConfig v1 = new BeautyFilterConfig();
        v1.enabled = this.enabled;
        v1.intensity = this.intensity;
        v1.smoothing = this.smoothing;
        v1.brightness = this.brightness;
        v1.softFocus = this.softFocus;
        return v1;
    }

    /**
     * 설정 값의 유효성을 검증합니다.
     *
     * @return 모든 값이 유효하면 true
     */
    public boolean isValid() {
        return intensity >= 0.0f && intensity <= 1.0f
                && smoothing >= 0.0f && smoothing <= 1.0f
                && brightness >= 0.5f && brightness <= 1.5f
                && softFocus >= 0.0f && softFocus <= 1.0f
                && whitening >= 0.0f && whitening <= 1.0f
                && colorBalance >= -1.0f && colorBalance <= 1.0f
                && wrinkleRemove >= 0.0f && wrinkleRemove <= 1.0f
                && skinQuality >= 0.0f && skinQuality <= 1.0f
                && slimFace >= 0.0f && slimFace <= 1.0f
                && enlargeEyes >= 0.0f && enlargeEyes <= 1.0f
                && thinChin >= 0.0f && thinChin <= 1.0f
                && downscaleFactor >= 1 && downscaleFactor <= 4;
    }

    /**
     * 값을 유효한 범위로 클램프합니다.
     */
    public void clamp() {
        intensity = clampFloat(intensity, 0.0f, 1.0f);
        smoothing = clampFloat(smoothing, 0.0f, 1.0f);
        brightness = clampFloat(brightness, 0.5f, 1.5f);
        softFocus = clampFloat(softFocus, 0.0f, 1.0f);
        whitening = clampFloat(whitening, 0.0f, 1.0f);
        colorBalance = clampFloat(colorBalance, -1.0f, 1.0f);
        wrinkleRemove = clampFloat(wrinkleRemove, 0.0f, 1.0f);
        skinQuality = clampFloat(skinQuality, 0.0f, 1.0f);
        slimFace = clampFloat(slimFace, 0.0f, 1.0f);
        enlargeEyes = clampFloat(enlargeEyes, 0.0f, 1.0f);
        thinChin = clampFloat(thinChin, 0.0f, 1.0f);
        downscaleFactor = clampInt(downscaleFactor, 1, 4);
    }

    private static float clampFloat(float value, float min, float max) {
        return Math.max(min, Math.min(max, value));
    }

    private static int clampInt(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }

    @NonNull
    @Override
    public String toString() {
        return "BeautyFilterConfigV2{" +
                "enabled=" + enabled +
                ", intensity=" + intensity +
                ", smoothing=" + smoothing +
                ", brightness=" + brightness +
                ", softFocus=" + softFocus +
                ", whitening=" + whitening +
                ", colorBalance=" + colorBalance +
                ", wrinkleRemove=" + wrinkleRemove +
                ", skinQuality=" + skinQuality +
                ", slimFace=" + slimFace +
                ", enlargeEyes=" + enlargeEyes +
                ", thinChin=" + thinChin +
                ", useGpu=" + useGpu +
                ", roiOnly=" + roiOnly +
                ", protectEyes=" + protectEyes +
                ", protectLips=" + protectLips +
                ", downscaleFactor=" + downscaleFactor +
                '}';
    }

    // ========================================================================
    // 빌더 클래스
    // ========================================================================

    /**
     * BeautyFilterConfigV2 빌더.
     *
     * <p>플루언트 API로 설정을 구성할 수 있습니다.</p>
     */
    public static class Builder {
        private final BeautyFilterConfigV2 config;

        public Builder() {
            config = new BeautyFilterConfigV2();
        }

        public Builder(@NonNull BeautyFilterConfigV2 base) {
            config = new BeautyFilterConfigV2(base);
        }

        // 기본 설정

        public Builder enabled(boolean enabled) {
            config.enabled = enabled;
            return this;
        }

        public Builder intensity(float intensity) {
            config.intensity = intensity;
            return this;
        }

        // 피부 효과

        public Builder smoothing(float smoothing) {
            config.smoothing = smoothing;
            return this;
        }

        public Builder brightness(float brightness) {
            config.brightness = brightness;
            return this;
        }

        public Builder softFocus(float softFocus) {
            config.softFocus = softFocus;
            return this;
        }

        public Builder whitening(float whitening) {
            config.whitening = whitening;
            return this;
        }

        public Builder colorBalance(float colorBalance) {
            config.colorBalance = colorBalance;
            return this;
        }

        public Builder wrinkleRemove(float wrinkleRemove) {
            config.wrinkleRemove = wrinkleRemove;
            return this;
        }

        public Builder skinQuality(float skinQuality) {
            config.skinQuality = skinQuality;
            return this;
        }

        // 얼굴 형태 보정

        public Builder slimFace(float slimFace) {
            config.slimFace = slimFace;
            return this;
        }

        public Builder enlargeEyes(float enlargeEyes) {
            config.enlargeEyes = enlargeEyes;
            return this;
        }

        public Builder thinChin(float thinChin) {
            config.thinChin = thinChin;
            return this;
        }

        // 처리 옵션

        public Builder useGpu(boolean useGpu) {
            config.useGpu = useGpu;
            return this;
        }

        public Builder roiOnly(boolean roiOnly) {
            config.roiOnly = roiOnly;
            return this;
        }

        public Builder protectEyes(boolean protectEyes) {
            config.protectEyes = protectEyes;
            return this;
        }

        public Builder protectLips(boolean protectLips) {
            config.protectLips = protectLips;
            return this;
        }

        public Builder downscaleFactor(int downscaleFactor) {
            config.downscaleFactor = downscaleFactor;
            return this;
        }

        /**
         * BeautyFilterConfigV2를 빌드합니다.
         *
         * @return 구성된 BeautyFilterConfigV2
         */
        public BeautyFilterConfigV2 build() {
            config.clamp();
            return new BeautyFilterConfigV2(config);
        }
    }
}
