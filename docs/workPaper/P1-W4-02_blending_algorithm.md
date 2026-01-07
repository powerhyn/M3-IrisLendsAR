# P1-W4-02: 블렌딩 알고리즘 구현

**태스크 ID**: P1-W4-02
**상태**: ⏳ 대기
**시작일**: -
**완료일**: -

---

## 1. 계획

### 목표
자연스러운 렌즈 오버레이를 위한 블렌딩 알고리즘을 구현한다. 경계 스무딩, 다양한 블렌딩 모드를 지원한다.

### 산출물
| 파일 | 설명 |
|------|------|
| `cpp/src/lens_renderer.cpp` 내 블렌딩 로직 | 블렌딩 알고리즘 |
| `cpp/include/iris_sdk/blend_modes.h` | 블렌딩 모드 정의 (선택적) |

### 검증 기준
- [ ] Normal 블렌딩 구현
- [ ] Multiply 블렌딩 구현
- [ ] Screen 블렌딩 구현
- [ ] 경계 페더링 (Feathering) 구현
- [ ] 시각적 품질 검증 (경계 자연스러움)

### 선행 조건
- P1-W4-01: LensRenderer 기본 구현 ✅

---

## 2. 분석

### 2.1 블렌딩 모드 종류

| 모드 | 공식 | 효과 |
|------|------|------|
| Normal | dst = src * α + dst * (1-α) | 표준 알파 블렌딩 |
| Multiply | dst = src * dst / 255 | 어두워짐, 색상 강화 |
| Screen | dst = 255 - (255-src) * (255-dst) / 255 | 밝아짐, 광택 효과 |
| Overlay | 조건부 | Multiply + Screen 혼합 |
| Soft Light | 복잡 | 부드러운 조명 효과 |

### 2.2 블렌딩 모드 구현

**2.2.1 BlendMode 열거형**
```cpp
/**
 * @brief 블렌딩 모드
 */
enum class BlendMode {
    Normal = 0,     ///< 표준 알파 블렌딩
    Multiply = 1,   ///< 곱하기 (어두워짐)
    Screen = 2,     ///< 스크린 (밝아짐)
    Overlay = 3,    ///< 오버레이 (대비 강화)
    SoftLight = 4   ///< 소프트 라이트
};
```

**2.2.2 Normal Blend**
```cpp
inline cv::Vec3b blendNormal(const cv::Vec3b& dst, const cv::Vec3b& src, float alpha) {
    return cv::Vec3b(
        static_cast<uint8_t>(dst[0] * (1 - alpha) + src[0] * alpha),
        static_cast<uint8_t>(dst[1] * (1 - alpha) + src[1] * alpha),
        static_cast<uint8_t>(dst[2] * (1 - alpha) + src[2] * alpha)
    );
}
```

**2.2.3 Multiply Blend**
```cpp
inline cv::Vec3b blendMultiply(const cv::Vec3b& dst, const cv::Vec3b& src, float alpha) {
    cv::Vec3b multiplied(
        static_cast<uint8_t>((dst[0] * src[0]) / 255),
        static_cast<uint8_t>((dst[1] * src[1]) / 255),
        static_cast<uint8_t>((dst[2] * src[2]) / 255)
    );
    return blendNormal(dst, multiplied, alpha);
}
```

**2.2.4 Screen Blend**
```cpp
inline cv::Vec3b blendScreen(const cv::Vec3b& dst, const cv::Vec3b& src, float alpha) {
    cv::Vec3b screened(
        static_cast<uint8_t>(255 - ((255 - dst[0]) * (255 - src[0])) / 255),
        static_cast<uint8_t>(255 - ((255 - dst[1]) * (255 - src[1])) / 255),
        static_cast<uint8_t>(255 - ((255 - dst[2]) * (255 - src[2])) / 255)
    );
    return blendNormal(dst, screened, alpha);
}
```

**2.2.5 Overlay Blend**
```cpp
inline uint8_t overlayChannel(uint8_t dst, uint8_t src) {
    if (dst < 128) {
        return static_cast<uint8_t>((2 * dst * src) / 255);
    } else {
        return static_cast<uint8_t>(255 - (2 * (255 - dst) * (255 - src)) / 255);
    }
}

inline cv::Vec3b blendOverlay(const cv::Vec3b& dst, const cv::Vec3b& src, float alpha) {
    cv::Vec3b overlayed(
        overlayChannel(dst[0], src[0]),
        overlayChannel(dst[1], src[1]),
        overlayChannel(dst[2], src[2])
    );
    return blendNormal(dst, overlayed, alpha);
}
```

### 2.3 경계 페더링 (Feathering)

**2.3.1 원형 마스크 생성**
```cpp
cv::Mat createCircularMask(int size, float feather_ratio) {
    cv::Mat mask(size, size, CV_32FC1);
    float center = size / 2.0f;
    float inner_radius = center * (1.0f - feather_ratio);
    float outer_radius = center;

    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            float dist = std::sqrt(std::pow(x - center, 2) + std::pow(y - center, 2));

            if (dist <= inner_radius) {
                mask.at<float>(y, x) = 1.0f;
            } else if (dist <= outer_radius) {
                // 페더링 영역: 부드러운 전환
                float t = (dist - inner_radius) / (outer_radius - inner_radius);
                mask.at<float>(y, x) = 1.0f - smoothstep(t);
            } else {
                mask.at<float>(y, x) = 0.0f;
            }
        }
    }

    return mask;
}
```

**2.3.2 Smoothstep 함수**
```cpp
inline float smoothstep(float t) {
    // Hermite 보간 (3차 곡선)
    t = std::clamp(t, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

inline float smootherstep(float t) {
    // 5차 곡선 (더 부드러움)
    t = std::clamp(t, 0.0f, 1.0f);
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}
```

### 2.4 홍채 마스크 생성

**눈의 해부학적 구조 고려**:
- 동공 (Pupil): 중심, 가장 어두움
- 홍채 (Iris): 동공 주변, 패턴 영역
- 공막 (Sclera): 홍채 바깥, 흰자

```cpp
cv::Mat createIrisMask(int size, float pupil_ratio, float feather) {
    cv::Mat mask(size, size, CV_32FC1);
    float center = size / 2.0f;
    float pupil_radius = center * pupil_ratio;     // 동공 (제외)
    float iris_radius = center;                     // 홍채 경계
    float feather_start = center * (1.0f - feather);

    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            float dist = std::sqrt(std::pow(x - center, 2) + std::pow(y - center, 2));

            if (dist <= pupil_radius) {
                // 동공 영역: 투명하게 (원본 보존)
                mask.at<float>(y, x) = 0.3f;  // 약간의 색상만
            } else if (dist <= feather_start) {
                // 홍채 영역: 완전 불투명
                mask.at<float>(y, x) = 1.0f;
            } else if (dist <= iris_radius) {
                // 경계 페더링
                float t = (dist - feather_start) / (iris_radius - feather_start);
                mask.at<float>(y, x) = 1.0f - smootherstep(t);
            } else {
                mask.at<float>(y, x) = 0.0f;
            }
        }
    }

    return mask;
}
```

### 2.5 블렌딩 함수 통합

```cpp
void blendWithMask(cv::Mat& dst, const cv::Mat& src, const cv::Mat& mask,
                   cv::Point position, float opacity, BlendMode mode) {
    // ROI 계산 및 클리핑
    cv::Rect roi = calculateValidROI(dst, src, position);
    if (roi.empty()) return;

    cv::Mat dst_roi = dst(roi);
    cv::Mat src_roi = src(/* adjusted ROI */);
    cv::Mat mask_roi = mask(/* adjusted ROI */);

    // 블렌딩 함수 선택
    auto blendFunc = [mode](const cv::Vec3b& d, const cv::Vec3b& s, float a) {
        switch (mode) {
            case BlendMode::Multiply: return blendMultiply(d, s, a);
            case BlendMode::Screen:   return blendScreen(d, s, a);
            case BlendMode::Overlay:  return blendOverlay(d, s, a);
            default:                  return blendNormal(d, s, a);
        }
    };

    // 픽셀별 블렌딩
    for (int y = 0; y < dst_roi.rows; ++y) {
        for (int x = 0; x < dst_roi.cols; ++x) {
            float alpha = mask_roi.at<float>(y, x) * opacity;
            if (alpha > 0.01f) {
                dst_roi.at<cv::Vec3b>(y, x) = blendFunc(
                    dst_roi.at<cv::Vec3b>(y, x),
                    src_roi.at<cv::Vec3b>(y, x),
                    alpha
                );
            }
        }
    }
}
```

### 2.6 성능 최적화

**2.6.1 SIMD 최적화 (OpenCV)**
```cpp
void blendNormalOptimized(cv::Mat& dst, const cv::Mat& src,
                          const cv::Mat& alpha, float opacity) {
    // OpenCV의 addWeighted 활용 (SIMD 최적화됨)
    cv::Mat alpha_scaled;
    alpha.convertTo(alpha_scaled, CV_32FC1, opacity);

    // 채널별 처리
    std::vector<cv::Mat> src_channels, dst_channels;
    cv::split(src, src_channels);
    cv::split(dst, dst_channels);

    for (int c = 0; c < 3; ++c) {
        cv::Mat src_f, dst_f;
        src_channels[c].convertTo(src_f, CV_32FC1);
        dst_channels[c].convertTo(dst_f, CV_32FC1);

        dst_f = dst_f.mul(1.0f - alpha_scaled) + src_f.mul(alpha_scaled);
        dst_f.convertTo(dst_channels[c], CV_8UC1);
    }

    cv::merge(dst_channels, dst);
}
```

**2.6.2 마스크 캐싱**
```cpp
class MaskCache {
public:
    cv::Mat getMask(int size, float feather) {
        auto key = std::make_pair(size, static_cast<int>(feather * 100));
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            return it->second;
        }

        cv::Mat mask = createIrisMask(size, 0.3f, feather);
        cache_[key] = mask;
        return mask;
    }

private:
    std::map<std::pair<int, int>, cv::Mat> cache_;
};
```

---

## 3. 실행 내역

### 3.1 블렌딩 함수 구현

```bash
# 예정: cpp/src/lens_renderer.cpp에 블렌딩 로직 추가
```

### 3.2 마스크 생성 함수 구현

```bash
# 예정: 원형 마스크, 홍채 마스크 생성 함수
```

### 3.3 시각적 품질 테스트

```bash
# 예정: 다양한 블렌딩 모드 비교 이미지 생성
```

---

## 4. 검증 결과

### 검증 항목

| 항목 | 결과 | 비고 |
|------|------|------|
| Normal 블렌딩 | ⏳ 대기 | |
| Multiply 블렌딩 | ⏳ 대기 | |
| Screen 블렌딩 | ⏳ 대기 | |
| Overlay 블렌딩 | ⏳ 대기 | |
| 경계 페더링 | ⏳ 대기 | |
| 시각적 품질 | ⏳ 대기 | |

### 블렌딩 모드 비교

| 모드 | 밝은 텍스처 | 어두운 텍스처 | 추천 용도 |
|------|------------|-------------|----------|
| Normal | - | - | 일반 |
| Multiply | - | - | 어두운 색상 렌즈 |
| Screen | - | - | 밝은 색상 렌즈 |
| Overlay | - | - | 대비 강조 |

---

## 5. 이슈 및 학습

### 이슈

| ID | 내용 | 상태 | 해결방안 |
|----|------|------|----------|
| - | - | - | - |

### 결정 사항

| 결정 | 이유 |
|------|------|
| Smootherstep 사용 | Smoothstep보다 더 자연스러운 경계 |
| 동공 영역 30% 투명도 | 동공이 보이면서 렌즈 색상도 적용 |
| 마스크 캐싱 | 동일 크기 마스크 재사용으로 성능 향상 |

### 학습 내용

(실행 후 기록)

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-01-07 | 태스크 문서 생성, 블렌딩 알고리즘 설계 완료 |
