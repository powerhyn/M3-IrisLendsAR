# P2-W1-02. BeautyFilterConfigV2 정의 및 C API 확장

## 작업 정보

| 항목 | 내용 |
|------|------|
| **작업 ID** | P2-W1-02 |
| **Phase** | Phase 1: 기반 구조 리팩토링 |
| **상태** | ⏳ 대기 |
| **예상 기간** | 1일 |
| **의존성** | P2-W1-01 (RenderContext) |
| **담당** | systems-programming:cpp-pro |

---

## 1. 목표

기존 `BeautyFilterConfig`를 확장하여 고급 뷰티 필터 기능을 지원하는 V2 구조체 정의

### 핵심 산출물
- `BeautyFilterConfigV2` 구조체
- V2 C API 함수
- 기존 V1 API 하위 호환성 유지

---

## 2. 상세 작업

### 2.1 BeautyFilterConfigV2 구조체

**파일**: `cpp/include/iris_sdk/beauty_filter.h` (확장)

```cpp
/**
 * @brief 확장된 뷰티 필터 설정 (V2)
 *
 * 기존 V1 필드 + 고급 피부 효과 + 얼굴 형태 보정 + 처리 옵션
 */
struct BeautyFilterConfigV2 {
    //===== 기본 설정 (V1 호환) =====
    bool enabled = false;           ///< 필터 활성화 여부
    float intensity = 0.5f;         ///< 전체 강도 (0.0~1.0)

    //===== 피부 효과 =====
    float smoothing = 0.5f;         ///< 피부 스무딩 (0.0~1.0)
    float brightness = 1.0f;        ///< 밝기 (0.5~1.5, 1.0=원본)
    float softFocus = 0.3f;         ///< 소프트 포커스 (0.0~1.0)
    float whitening = 0.0f;         ///< 피부톤 화이트닝 (0.0~1.0)
    float colorBalance = 0.0f;      ///< 컬러 밸런스 (-1.0~1.0)
    float wrinkleRemove = 0.0f;     ///< 주름 제거 (0.0~1.0)

    //===== 얼굴 형태 보정 =====
    float slimFace = 0.0f;          ///< 얼굴 슬림화 (0.0~1.0)
    float enlargeEyes = 0.0f;       ///< 눈 확대 (0.0~1.0)
    float thinChin = 0.0f;          ///< 턱 축소 (0.0~1.0)

    //===== 처리 옵션 =====
    bool useGpu = true;             ///< GPU 가속 사용
    bool roiOnly = true;            ///< 얼굴 영역만 처리
    bool protectEyes = true;        ///< 눈 영역 보호
    bool protectLips = true;        ///< 입술 영역 보호
    int downscaleFactor = 1;        ///< 다운스케일 (1=원본, 2=1/2, 4=1/4)

    //===== 헬퍼 메서드 =====
    /** V1 설정으로부터 변환 */
    static BeautyFilterConfigV2 fromV1(const BeautyFilterConfig& v1);

    /** V1 설정으로 변환 (피부 효과만) */
    BeautyFilterConfig toV1() const;

    /** 유효성 검증 */
    bool isValid() const;

    /** 범위 내로 클램핑 */
    void clamp();
};

//===== C API (V2) =====
#ifdef __cplusplus
extern "C" {
#endif

/** V2 기본 설정 반환 */
IRIS_SDK_EXPORT void iris_sdk_default_beauty_config_v2(BeautyFilterConfigV2* config);

/** V2 뷰티 필터 설정 */
IRIS_SDK_EXPORT IrisSdkError iris_sdk_set_beauty_filter_v2(
    const BeautyFilterConfigV2* config);

/** V2 현재 설정 조회 */
IRIS_SDK_EXPORT IrisSdkError iris_sdk_get_beauty_filter_v2(
    BeautyFilterConfigV2* config);

/** V2 필터 적용 (Face Mesh 연동) */
IRIS_SDK_EXPORT IrisSdkError iris_sdk_apply_beauty_filter_v2(
    uint8_t* frame_data,
    int width, int height,
    IrisFrameFormat format,
    const IrisResult* iris_result  // nullable
);

/** GPU 사용 가능 여부 */
IRIS_SDK_EXPORT bool iris_sdk_beauty_gpu_available(void);

/** 현재 GPU 사용 중 여부 */
IRIS_SDK_EXPORT bool iris_sdk_beauty_using_gpu(void);

#ifdef __cplusplus
}
#endif
```

### 2.2 구현 파일

**파일**: `cpp/src/beauty_filter.cpp` (확장)

```cpp
BeautyFilterConfigV2 BeautyFilterConfigV2::fromV1(const BeautyFilterConfig& v1) {
    BeautyFilterConfigV2 v2;
    v2.enabled = v1.enabled;
    v2.intensity = v1.intensity;
    v2.smoothing = v1.smoothing;
    v2.brightness = v1.brightness;
    v2.softFocus = v1.softFocus;
    // 나머지는 기본값 유지
    return v2;
}

BeautyFilterConfig BeautyFilterConfigV2::toV1() const {
    BeautyFilterConfig v1;
    v1.enabled = enabled;
    v1.intensity = intensity;
    v1.smoothing = smoothing;
    v1.brightness = brightness;
    v1.softFocus = softFocus;
    return v1;
}

bool BeautyFilterConfigV2::isValid() const {
    return (intensity >= 0.0f && intensity <= 1.0f) &&
           (smoothing >= 0.0f && smoothing <= 1.0f) &&
           (brightness >= 0.5f && brightness <= 1.5f) &&
           (softFocus >= 0.0f && softFocus <= 1.0f) &&
           (whitening >= 0.0f && whitening <= 1.0f) &&
           (colorBalance >= -1.0f && colorBalance <= 1.0f) &&
           (wrinkleRemove >= 0.0f && wrinkleRemove <= 1.0f) &&
           (slimFace >= 0.0f && slimFace <= 1.0f) &&
           (enlargeEyes >= 0.0f && enlargeEyes <= 1.0f) &&
           (thinChin >= 0.0f && thinChin <= 1.0f) &&
           (downscaleFactor >= 1 && downscaleFactor <= 4);
}

void BeautyFilterConfigV2::clamp() {
    intensity = std::clamp(intensity, 0.0f, 1.0f);
    smoothing = std::clamp(smoothing, 0.0f, 1.0f);
    brightness = std::clamp(brightness, 0.5f, 1.5f);
    softFocus = std::clamp(softFocus, 0.0f, 1.0f);
    whitening = std::clamp(whitening, 0.0f, 1.0f);
    colorBalance = std::clamp(colorBalance, -1.0f, 1.0f);
    wrinkleRemove = std::clamp(wrinkleRemove, 0.0f, 1.0f);
    slimFace = std::clamp(slimFace, 0.0f, 1.0f);
    enlargeEyes = std::clamp(enlargeEyes, 0.0f, 1.0f);
    thinChin = std::clamp(thinChin, 0.0f, 1.0f);
    downscaleFactor = std::clamp(downscaleFactor, 1, 4);
}

// C API 구현
extern "C" {

void iris_sdk_default_beauty_config_v2(BeautyFilterConfigV2* config) {
    if (config) {
        *config = BeautyFilterConfigV2{};  // 기본 생성자 호출
    }
}

IrisSdkError iris_sdk_set_beauty_filter_v2(const BeautyFilterConfigV2* config) {
    if (!config) return IRIS_SDK_ERROR_INVALID_PARAM;
    if (!config->isValid()) return IRIS_SDK_ERROR_INVALID_PARAM;

    // BeautyProcessor에 설정 전달
    return BeautyProcessor::getInstance().setConfig(*config);
}

// ... 나머지 C API 구현

} // extern "C"
```

### 2.3 JNI 바인딩 (Java 클래스)

**파일**: `android/iris-sdk/src/main/java/com/irislenssdk/BeautyFilterConfigV2.java`

```java
package com.irislenssdk;

/**
 * 확장된 뷰티 필터 설정 (V2)
 *
 * 기본 피부 효과 + 고급 효과 + 얼굴 형태 보정 지원
 */
public class BeautyFilterConfigV2 {
    // 기본 설정
    public boolean enabled = false;
    public float intensity = 0.5f;

    // 피부 효과
    public float smoothing = 0.5f;
    public float brightness = 1.0f;
    public float softFocus = 0.3f;
    public float whitening = 0.0f;
    public float colorBalance = 0.0f;
    public float wrinkleRemove = 0.0f;

    // 얼굴 형태
    public float slimFace = 0.0f;
    public float enlargeEyes = 0.0f;
    public float thinChin = 0.0f;

    // 처리 옵션
    public boolean useGpu = true;
    public boolean roiOnly = true;
    public boolean protectEyes = true;
    public boolean protectLips = true;
    public int downscaleFactor = 1;

    /** 기본 생성자 */
    public BeautyFilterConfigV2() {}

    /** V1에서 변환 */
    public static BeautyFilterConfigV2 fromV1(BeautyFilterConfig v1) {
        BeautyFilterConfigV2 v2 = new BeautyFilterConfigV2();
        v2.enabled = v1.enabled;
        v2.intensity = v1.intensity;
        v2.smoothing = v1.smoothing;
        v2.brightness = v1.brightness;
        v2.softFocus = v1.softFocus;
        return v2;
    }

    /** 유효성 검증 */
    public boolean isValid() {
        return intensity >= 0f && intensity <= 1f &&
               smoothing >= 0f && smoothing <= 1f &&
               brightness >= 0.5f && brightness <= 1.5f &&
               // ... 나머지 검증
               downscaleFactor >= 1 && downscaleFactor <= 4;
    }

    /** 범위 내로 클램핑 */
    public void clamp() {
        intensity = Math.max(0f, Math.min(1f, intensity));
        smoothing = Math.max(0f, Math.min(1f, smoothing));
        brightness = Math.max(0.5f, Math.min(1.5f, brightness));
        // ... 나머지 클램핑
    }

    /** Builder 패턴 */
    public static class Builder {
        private BeautyFilterConfigV2 config = new BeautyFilterConfigV2();

        public Builder enabled(boolean enabled) {
            config.enabled = enabled;
            return this;
        }

        public Builder intensity(float intensity) {
            config.intensity = intensity;
            return this;
        }

        public Builder smoothing(float smoothing) {
            config.smoothing = smoothing;
            return this;
        }

        // ... 나머지 빌더 메서드

        public BeautyFilterConfigV2 build() {
            config.clamp();
            return config;
        }
    }
}
```

---

## 3. 단위 테스트

**파일**: `cpp/tests/test_beauty_config_v2.cpp`

```cpp
TEST(BeautyFilterConfigV2, DefaultValues) {
    BeautyFilterConfigV2 config;
    EXPECT_FALSE(config.enabled);
    EXPECT_FLOAT_EQ(config.intensity, 0.5f);
    EXPECT_FLOAT_EQ(config.smoothing, 0.5f);
    EXPECT_FLOAT_EQ(config.brightness, 1.0f);
    EXPECT_EQ(config.downscaleFactor, 1);
}

TEST(BeautyFilterConfigV2, IsValidRejectsOutOfRange) {
    BeautyFilterConfigV2 config;
    config.intensity = 1.5f;  // Out of range
    EXPECT_FALSE(config.isValid());

    config.intensity = 0.5f;
    config.brightness = 2.0f;  // Out of range
    EXPECT_FALSE(config.isValid());
}

TEST(BeautyFilterConfigV2, ClampCorrectsBoundaries) {
    BeautyFilterConfigV2 config;
    config.intensity = 1.5f;
    config.brightness = 0.0f;
    config.downscaleFactor = 10;

    config.clamp();

    EXPECT_FLOAT_EQ(config.intensity, 1.0f);
    EXPECT_FLOAT_EQ(config.brightness, 0.5f);
    EXPECT_EQ(config.downscaleFactor, 4);
}

TEST(BeautyFilterConfigV2, V1Conversion) {
    BeautyFilterConfig v1;
    v1.enabled = true;
    v1.intensity = 0.7f;
    v1.smoothing = 0.6f;
    v1.brightness = 1.1f;
    v1.softFocus = 0.4f;

    auto v2 = BeautyFilterConfigV2::fromV1(v1);

    EXPECT_EQ(v2.enabled, v1.enabled);
    EXPECT_FLOAT_EQ(v2.intensity, v1.intensity);
    EXPECT_FLOAT_EQ(v2.smoothing, v1.smoothing);
    // V2 전용 필드는 기본값
    EXPECT_FLOAT_EQ(v2.whitening, 0.0f);
    EXPECT_FLOAT_EQ(v2.slimFace, 0.0f);
}

TEST(BeautyFilterConfigV2, CAPI_DefaultConfig) {
    BeautyFilterConfigV2 config;
    config.intensity = 999.0f;  // 임의 값

    iris_sdk_default_beauty_config_v2(&config);

    EXPECT_FLOAT_EQ(config.intensity, 0.5f);  // 기본값으로 리셋
}
```

---

## 4. 완료 기준

- [ ] `BeautyFilterConfigV2` 구조체 정의
- [ ] 필드별 기본값 및 범위 문서화
- [ ] `fromV1()` / `toV1()` 변환 메서드
- [ ] `isValid()` / `clamp()` 헬퍼 메서드
- [ ] V2 C API 함수 선언 및 스텁 구현
- [ ] Java `BeautyFilterConfigV2` 클래스
- [ ] 단위 테스트 100% 통과

---

## 5. 다음 작업

- **P2-W1-03**: BeautyROIManager 구현
