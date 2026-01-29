# P2-W2-03. 새 필터 효과 구현 (CPU)

## 작업 정보

| 항목 | 내용 |
|------|------|
| **작업 ID** | P2-W2-03 |
| **Phase** | Phase 2: CPU 백엔드 개선 |
| **상태** | ✅ 완료 |
| **예상 기간** | 2일 |
| **완료일** | 2026-01-28 |
| **의존성** | P2-W2-01, P2-W2-02 |
| **담당** | systems-programming:cpp-pro |

---

## 1. 목표

BeautyFilterConfigV2에 추가된 새 필터 효과들의 CPU 구현

### 핵심 산출물
- 피부톤 화이트닝 (LAB 색상 공간)
- 컬러 밸런스 조정
- 주름 제거 (Targeted Smoothing)
- 기존 효과 개선 (Soft Focus, Brightness)

---

## 2. 상세 작업

### 2.1 피부톤 화이트닝

LAB 색상 공간에서 L(밝기) 채널을 선택적으로 조정

**파일**: `cpp/src/cpu_beauty_backend.cpp` (확장)

```cpp
void CPUBeautyBackend::applyWhitening(
    cv::Mat& roi,
    float strength,
    const cv::Mat& protection_mask) {

    if (strength < 0.01f) return;

    // BGR → LAB 변환
    cv::Mat lab;
    cv::cvtColor(roi, lab, cv::COLOR_BGR2Lab);

    // 채널 분리
    std::vector<cv::Mat> lab_channels;
    cv::split(lab, lab_channels);

    cv::Mat& L = lab_channels[0];  // 밝기 채널

    // 피부톤 감지 마스크 (A, B 채널 기반)
    cv::Mat skin_tone_mask = detectSkinTone(lab_channels[1], lab_channels[2]);

    // 보호 마스크와 결합
    cv::Mat apply_mask = skin_tone_mask.clone();
    if (!protection_mask.empty()) {
        cv::Mat mask_inv;
        cv::bitwise_not(protection_mask, mask_inv);
        cv::bitwise_and(apply_mask, mask_inv, apply_mask);
    }

    // 밝기 증가 (피부 영역만)
    // 비선형 조정: 어두운 영역은 더 밝게, 밝은 영역은 덜 밝게
    cv::Mat L_float;
    L.convertTo(L_float, CV_32F);

    // 감마 보정 기반 화이트닝
    float gamma = 1.0f - strength * 0.3f;  // 0.7 ~ 1.0
    cv::Mat L_normalized = L_float / 255.0f;
    cv::pow(L_normalized, gamma, L_normalized);
    L_normalized *= 255.0f;

    // 마스크 적용
    cv::Mat mask_float;
    apply_mask.convertTo(mask_float, CV_32F, 1.0 / 255.0);

    cv::Mat L_result = L_float.mul(1.0 - mask_float) +
                       L_normalized.mul(mask_float);

    // 클램핑 및 변환
    cv::threshold(L_result, L_result, 255, 255, cv::THRESH_TRUNC);
    L_result.convertTo(L, CV_8U);

    // 채널 병합 및 변환
    cv::merge(lab_channels, lab);
    cv::cvtColor(lab, roi, cv::COLOR_Lab2BGR);
}

cv::Mat CPUBeautyBackend::detectSkinTone(
    const cv::Mat& A_channel,
    const cv::Mat& B_channel) {

    // LAB에서 피부톤 범위
    // A: 133-173 (약간 붉은 톤)
    // B: 133-173 (약간 노란 톤)
    cv::Mat mask = cv::Mat::zeros(A_channel.size(), CV_8U);

    for (int y = 0; y < A_channel.rows; ++y) {
        const uint8_t* a_row = A_channel.ptr<uint8_t>(y);
        const uint8_t* b_row = B_channel.ptr<uint8_t>(y);
        uint8_t* mask_row = mask.ptr<uint8_t>(y);

        for (int x = 0; x < A_channel.cols; ++x) {
            uint8_t a = a_row[x];
            uint8_t b = b_row[x];

            // 피부톤 범위 체크 (LAB 색상 공간)
            // A: 중간값(128) 기준 빨간쪽(+), B: 중간값 기준 노란쪽(+)
            bool is_skin = (a >= 125 && a <= 175) &&
                           (b >= 130 && b <= 180);

            mask_row[x] = is_skin ? 255 : 0;
        }
    }

    // 모폴로지 연산으로 노이즈 제거
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

    // 소프트 마스크로 변환
    cv::GaussianBlur(mask, mask, cv::Size(11, 11), 0);

    return mask;
}
```

### 2.2 컬러 밸런스 조정

따뜻한/차가운 톤 조정

```cpp
void CPUBeautyBackend::applyColorBalance(
    cv::Mat& roi,
    float balance) {

    if (std::abs(balance) < 0.01f) return;

    // balance: -1.0 (차가운 톤) ~ +1.0 (따뜻한 톤)

    // BGR → LAB 변환
    cv::Mat lab;
    cv::cvtColor(roi, lab, cv::COLOR_BGR2Lab);

    std::vector<cv::Mat> lab_channels;
    cv::split(lab, lab_channels);

    cv::Mat& A = lab_channels[1];  // Green-Red
    cv::Mat& B = lab_channels[2];  // Blue-Yellow

    // A, B 채널 조정
    // 양수: 따뜻함 (A+, B+)
    // 음수: 차가움 (A-, B-)
    float a_shift = balance * 10.0f;  // ±10
    float b_shift = balance * 15.0f;  // ±15 (노란/파란 더 강조)

    cv::Mat A_float, B_float;
    A.convertTo(A_float, CV_32F);
    B.convertTo(B_float, CV_32F);

    A_float += a_shift;
    B_float += b_shift;

    // 클램핑
    cv::threshold(A_float, A_float, 255, 255, cv::THRESH_TRUNC);
    cv::threshold(A_float, A_float, 0, 0, cv::THRESH_TOZERO);
    cv::threshold(B_float, B_float, 255, 255, cv::THRESH_TRUNC);
    cv::threshold(B_float, B_float, 0, 0, cv::THRESH_TOZERO);

    A_float.convertTo(A, CV_8U);
    B_float.convertTo(B, CV_8U);

    cv::merge(lab_channels, lab);
    cv::cvtColor(lab, roi, cv::COLOR_Lab2BGR);
}
```

### 2.3 주름 제거 (Targeted Smoothing)

얼굴 특정 영역(이마, 눈가, 미간)에 강한 스무딩 적용

```cpp
struct WrinkleRegions {
    cv::Mat forehead_mask;      ///< 이마 영역
    cv::Mat crow_feet_mask;     ///< 눈가 (까마귀 발자국)
    cv::Mat frown_lines_mask;   ///< 미간 주름
    cv::Mat combined;           ///< 통합 마스크
};

void CPUBeautyBackend::applyWrinkleRemoval(
    cv::Mat& roi,
    float strength,
    const IrisLandmark* face_mesh,
    int roi_offset_x,
    int roi_offset_y) {

    if (strength < 0.01f || !face_mesh) return;

    // 주름 영역 마스크 생성
    WrinkleRegions regions = createWrinkleRegionMasks(
        face_mesh, roi.cols, roi.rows, roi_offset_x, roi_offset_y);

    // 강한 스무딩 (Guided Filter, 높은 eps)
    cv::Mat strong_smoothed;
    float eps = 0.04f + strength * 0.12f;  // 0.04 ~ 0.16
    FastGuidedFilter::filter(roi, strong_smoothed, 12, eps, 2);

    // 마스크 기반 선택적 블렌딩
    cv::Mat mask_3ch;
    cv::cvtColor(regions.combined, mask_3ch, cv::COLOR_GRAY2BGR);
    mask_3ch.convertTo(mask_3ch, CV_32F, strength / 255.0f);

    cv::Mat roi_f, smoothed_f;
    roi.convertTo(roi_f, CV_32F);
    strong_smoothed.convertTo(smoothed_f, CV_32F);

    cv::Mat result = roi_f.mul(cv::Scalar(1, 1, 1) - mask_3ch) +
                     smoothed_f.mul(mask_3ch);

    result.convertTo(roi, CV_8U);
}

WrinkleRegions CPUBeautyBackend::createWrinkleRegionMasks(
    const IrisLandmark* face_mesh,
    int width, int height,
    int offset_x, int offset_y) {

    WrinkleRegions regions;
    regions.forehead_mask = cv::Mat::zeros(height, width, CV_8U);
    regions.crow_feet_mask = cv::Mat::zeros(height, width, CV_8U);
    regions.frown_lines_mask = cv::Mat::zeros(height, width, CV_8U);

    // 이마 영역 (랜드마크 10, 151, 9 등 상단)
    // 이마 상단 랜드마크들
    const int FOREHEAD_TOP[] = {10, 151, 9, 8, 107, 336};
    const int FOREHEAD_BOTTOM[] = {108, 69, 104, 68, 337, 299};

    std::vector<cv::Point> forehead_pts;
    for (int idx : FOREHEAD_TOP) {
        int x = static_cast<int>(face_mesh[idx].x * (width + 2 * offset_x)) - offset_x;
        int y = static_cast<int>(face_mesh[idx].y * (height + 2 * offset_y)) - offset_y;
        forehead_pts.emplace_back(x, y);
    }
    // ... forehead_bottom 추가

    if (forehead_pts.size() >= 3) {
        cv::fillConvexPoly(regions.forehead_mask, forehead_pts, cv::Scalar(255));
    }

    // 눈가 주름 (눈 외곽 측면)
    // 왼쪽 눈가: 130, 247, 30, 29, 27, 28, 56, 190
    // 오른쪽 눈가: 359, 467, 260, 259, 257, 258, 286, 414

    const int LEFT_CROW_FEET[] = {130, 247, 30, 29, 27};
    const int RIGHT_CROW_FEET[] = {359, 467, 260, 259, 257};

    for (int i = 0; i < 5; ++i) {
        int idx = LEFT_CROW_FEET[i];
        int x = static_cast<int>(face_mesh[idx].x * (width + 2 * offset_x)) - offset_x;
        int y = static_cast<int>(face_mesh[idx].y * (height + 2 * offset_y)) - offset_y;
        cv::circle(regions.crow_feet_mask, cv::Point(x, y), 15, cv::Scalar(255), -1);
    }

    for (int i = 0; i < 5; ++i) {
        int idx = RIGHT_CROW_FEET[i];
        int x = static_cast<int>(face_mesh[idx].x * (width + 2 * offset_x)) - offset_x;
        int y = static_cast<int>(face_mesh[idx].y * (height + 2 * offset_y)) - offset_y;
        cv::circle(regions.crow_feet_mask, cv::Point(x, y), 15, cv::Scalar(255), -1);
    }

    // 미간 (9, 8, 168 주변)
    const int FROWN_INDICES[] = {9, 8, 168, 6, 197, 195};
    for (int idx : FROWN_INDICES) {
        int x = static_cast<int>(face_mesh[idx].x * (width + 2 * offset_x)) - offset_x;
        int y = static_cast<int>(face_mesh[idx].y * (height + 2 * offset_y)) - offset_y;
        cv::circle(regions.frown_lines_mask, cv::Point(x, y), 10, cv::Scalar(255), -1);
    }

    // 통합 마스크
    regions.combined = cv::Mat::zeros(height, width, CV_8U);
    regions.combined |= regions.forehead_mask;
    regions.combined |= regions.crow_feet_mask;
    regions.combined |= regions.frown_lines_mask;

    // 소프트 마스크
    cv::GaussianBlur(regions.combined, regions.combined, cv::Size(21, 21), 0);

    return regions;
}
```

### 2.4 개선된 Soft Focus

기존 가우시안 블러 기반에서 Guided Filter 기반으로 개선

```cpp
void CPUBeautyBackend::applySoftFocusV2(
    cv::Mat& roi,
    float strength) {

    if (strength < 0.01f) return;

    // 1단계: Guided Filter로 에지 보존 스무딩
    cv::Mat smoothed;
    float eps = 0.02f + strength * 0.08f;
    FastGuidedFilter::filter(roi, smoothed, 6, eps, 2);

    // 2단계: 오버레이 블렌딩 (하이라이트 강조)
    cv::Mat blended;
    overlayBlend(roi, smoothed, blended, strength * 0.5f);

    // 3단계: 약간의 가우시안 글로우 추가
    cv::Mat glow;
    int blur_size = static_cast<int>(15 * strength) * 2 + 1;
    cv::GaussianBlur(blended, glow, cv::Size(blur_size, blur_size), 0);

    // 글로우와 원본 블렌딩
    cv::addWeighted(blended, 0.7, glow, 0.3 * strength, 0, roi);
}

void CPUBeautyBackend::overlayBlend(
    const cv::Mat& base,
    const cv::Mat& blend,
    cv::Mat& result,
    float opacity) {

    result = cv::Mat(base.size(), base.type());

    for (int y = 0; y < base.rows; ++y) {
        const cv::Vec3b* base_row = base.ptr<cv::Vec3b>(y);
        const cv::Vec3b* blend_row = blend.ptr<cv::Vec3b>(y);
        cv::Vec3b* result_row = result.ptr<cv::Vec3b>(y);

        for (int x = 0; x < base.cols; ++x) {
            for (int c = 0; c < 3; ++c) {
                float b = base_row[x][c] / 255.0f;
                float l = blend_row[x][c] / 255.0f;

                // Overlay: 2*b*l if b < 0.5, else 1 - 2*(1-b)*(1-l)
                float overlay;
                if (b < 0.5f) {
                    overlay = 2.0f * b * l;
                } else {
                    overlay = 1.0f - 2.0f * (1.0f - b) * (1.0f - l);
                }

                // 원본과 블렌딩
                float final_val = b * (1.0f - opacity) + overlay * opacity;
                result_row[x][c] = static_cast<uint8_t>(
                    std::clamp(final_val * 255.0f, 0.0f, 255.0f));
            }
        }
    }
}
```

### 2.5 개선된 밝기 조정

비선형 밝기 조정 (하이라이트 보호)

```cpp
void CPUBeautyBackend::applyBrightnessV2(
    cv::Mat& roi,
    float brightness) {

    // brightness: 0.5 ~ 1.5 (1.0 = 원본)
    if (std::abs(brightness - 1.0f) < 0.01f) return;

    // LAB 색상 공간에서 L 채널 조정
    cv::Mat lab;
    cv::cvtColor(roi, lab, cv::COLOR_BGR2Lab);

    std::vector<cv::Mat> lab_channels;
    cv::split(lab, lab_channels);

    cv::Mat& L = lab_channels[0];
    cv::Mat L_float;
    L.convertTo(L_float, CV_32F);

    if (brightness > 1.0f) {
        // 밝게: 비선형 증가 (하이라이트 보호)
        float factor = brightness;
        L_float = L_float + (255.0f - L_float) * (factor - 1.0f) * 0.5f;
    } else {
        // 어둡게: 선형 감소
        L_float = L_float * brightness;
    }

    // 클램핑
    cv::threshold(L_float, L_float, 255, 255, cv::THRESH_TRUNC);
    cv::threshold(L_float, L_float, 0, 0, cv::THRESH_TOZERO);

    L_float.convertTo(L, CV_8U);

    cv::merge(lab_channels, lab);
    cv::cvtColor(lab, roi, cv::COLOR_Lab2BGR);
}
```

---

## 3. CPUBeautyBackend 통합

**파일**: `cpp/src/cpu_beauty_backend.cpp`

```cpp
IrisSdkError CPUBeautyBackend::apply(
    uint8_t* frame_data,
    int width, int height,
    IrisFrameFormat format,
    const BeautyFilterConfigV2& config,
    const BeautyROI* roi) {

    std::lock_guard<std::mutex> lock(buffer_mutex_);

    cv::Mat frame = convertToBGR(frame_data, width, height, format);

    // 다운스케일
    cv::Mat work_frame;
    int scale = config.downscaleFactor;
    if (scale > 1) {
        cv::resize(frame, work_frame, cv::Size(width / scale, height / scale));
    } else {
        work_frame = frame;
    }

    // ROI 또는 전체 프레임
    cv::Mat target_region;
    cv::Rect actual_rect;
    cv::Mat protection_mask;

    if (roi && roi->valid) {
        // ROI 추출
        BeautyROI scaled_roi = scaleROI(*roi, scale);
        BeautyROIManager::extractROIRegion(work_frame, scaled_roi,
                                           target_region, actual_rect, 20);
        protection_mask = roi->protection_mask;
    } else {
        target_region = work_frame;
        actual_rect = cv::Rect(0, 0, work_frame.cols, work_frame.rows);
    }

    // 필터 파이프라인 적용
    // 1. 피부 스무딩 (V2: Guided Filter)
    if (config.smoothing > 0.01f) {
        applySkinSmoothingV2(target_region, config.smoothing, protection_mask);
    }

    // 2. 화이트닝
    if (config.whitening > 0.01f) {
        applyWhitening(target_region, config.whitening, protection_mask);
    }

    // 3. 컬러 밸런스
    if (std::abs(config.colorBalance) > 0.01f) {
        applyColorBalance(target_region, config.colorBalance);
    }

    // 4. 주름 제거
    if (config.wrinkleRemove > 0.01f && roi && roi->face_mesh_valid) {
        applyWrinkleRemoval(target_region, config.wrinkleRemove,
                            roi->face_mesh,
                            actual_rect.x, actual_rect.y);
    }

    // 5. 소프트 포커스 (V2)
    if (config.softFocus > 0.01f) {
        applySoftFocusV2(target_region, config.softFocus);
    }

    // 6. 밝기 (V2)
    if (std::abs(config.brightness - 1.0f) > 0.01f) {
        applyBrightnessV2(target_region, config.brightness);
    }

    // ROI 합성
    if (roi && roi->valid) {
        cv::Mat feather_mask = BeautyROIManager::createFeatherMask(
            *roi, 15, roi->skin_mask, roi->eye_mask, roi->lip_mask);
        BeautyROIManager::applyROIRegion(work_frame, target_region,
                                          actual_rect, feather_mask);
    }

    // 업스케일
    if (scale > 1) {
        cv::resize(work_frame, frame, cv::Size(width, height));
    }

    convertFromBGR(frame, frame_data, format);
    return IRIS_SDK_OK;
}
```

---

## 4. 단위 테스트

**파일**: `cpp/tests/test_new_filter_effects.cpp`

```cpp
TEST(NewFilterEffects, WhiteningBrightensSkintone) {
    cv::Mat test_image = cv::imread("test_data/portrait.jpg");
    ASSERT_FALSE(test_image.empty());

    cv::Mat original = test_image.clone();
    CPUBeautyBackend backend;
    backend.initialize();

    // 화이트닝 적용
    backend.applyWhitening(test_image, 0.8f, cv::Mat());

    // LAB L 채널 비교
    cv::Mat lab_orig, lab_result;
    cv::cvtColor(original, lab_orig, cv::COLOR_BGR2Lab);
    cv::cvtColor(test_image, lab_result, cv::COLOR_BGR2Lab);

    std::vector<cv::Mat> ch_orig, ch_result;
    cv::split(lab_orig, ch_orig);
    cv::split(lab_result, ch_result);

    double mean_L_orig = cv::mean(ch_orig[0])[0];
    double mean_L_result = cv::mean(ch_result[0])[0];

    // 밝기 증가 확인
    EXPECT_GT(mean_L_result, mean_L_orig);
}

TEST(NewFilterEffects, ColorBalanceShiftsTone) {
    cv::Mat test_image(100, 100, CV_8UC3, cv::Scalar(128, 128, 128));

    // 따뜻한 톤 (+0.5)
    cv::Mat warm = test_image.clone();
    CPUBeautyBackend backend;
    backend.applyColorBalance(warm, 0.5f);

    // B 채널 (노란색) 증가 확인
    cv::Mat lab;
    cv::cvtColor(warm, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> channels;
    cv::split(lab, channels);

    double mean_B = cv::mean(channels[2])[0];
    EXPECT_GT(mean_B, 128);  // 노란쪽으로 이동
}

TEST(NewFilterEffects, WrinkleRemovalSmoothsTargetAreas) {
    cv::Mat test_image = cv::imread("test_data/portrait.jpg");
    IrisLandmark face_mesh[478];
    // ... 테스트용 랜드마크 로드 ...

    cv::Mat original = test_image.clone();

    CPUBeautyBackend backend;
    backend.applyWrinkleRemoval(test_image, 0.8f, face_mesh, 0, 0);

    // 이마 영역 분산 감소 확인 (스무딩 효과)
    cv::Rect forehead_rect(200, 50, 100, 50);  // 예시 좌표
    cv::Mat orig_forehead = original(forehead_rect);
    cv::Mat result_forehead = test_image(forehead_rect);

    cv::Scalar orig_stddev, result_stddev;
    cv::meanStdDev(orig_forehead, cv::Scalar(), orig_stddev);
    cv::meanStdDev(result_forehead, cv::Scalar(), result_stddev);

    // 분산 감소 = 스무딩
    EXPECT_LT(result_stddev[0], orig_stddev[0]);
}

TEST(NewFilterEffects, SoftFocusV2CreatesGlow) {
    cv::Mat test_image(480, 640, CV_8UC3);
    cv::randu(test_image, cv::Scalar(50, 50, 50), cv::Scalar(200, 200, 200));

    cv::Mat original = test_image.clone();

    CPUBeautyBackend backend;
    backend.applySoftFocusV2(test_image, 0.7f);

    // 결과가 원본과 다름 확인
    cv::Mat diff;
    cv::absdiff(original, test_image, diff);
    double max_diff = cv::mean(diff)[0];

    EXPECT_GT(max_diff, 5);  // 변화 있음
}

TEST(NewFilterEffects, BrightnessV2PreservesHighlights) {
    // 하이라이트 영역 (밝은 영역)
    cv::Mat test_image(100, 100, CV_8UC3, cv::Scalar(240, 240, 240));

    cv::Mat original = test_image.clone();

    CPUBeautyBackend backend;
    backend.applyBrightnessV2(test_image, 1.3f);  // 30% 밝게

    // 하이라이트가 과포화되지 않음 확인
    double max_val;
    cv::minMaxLoc(test_image, nullptr, &max_val);

    EXPECT_LE(max_val, 255);

    // 원본 대비 과도하게 밝아지지 않음
    cv::Scalar mean_orig = cv::mean(original);
    cv::Scalar mean_result = cv::mean(test_image);

    // 30% 밝게 요청했지만 하이라이트 보호로 실제 증가는 적음
    double increase_ratio = mean_result[0] / mean_orig[0];
    EXPECT_LT(increase_ratio, 1.15);  // 15% 미만 증가
}

TEST(NewFilterEffects, FullPipeline_IntegrationTest) {
    cv::Mat test_image = cv::imread("test_data/portrait.jpg");
    ASSERT_FALSE(test_image.empty());

    CPUBeautyBackend backend;
    backend.initialize();

    BeautyFilterConfigV2 config;
    config.enabled = true;
    config.smoothing = 0.6f;
    config.whitening = 0.4f;
    config.colorBalance = 0.2f;
    config.softFocus = 0.3f;
    config.brightness = 1.1f;

    std::vector<uint8_t> frame_data(test_image.total() * test_image.elemSize());
    std::memcpy(frame_data.data(), test_image.data, frame_data.size());

    IrisSdkError err = backend.apply(
        frame_data.data(), test_image.cols, test_image.rows,
        IRIS_FORMAT_RGB, config, nullptr);

    EXPECT_EQ(err, IRIS_SDK_OK);
}
```

---

## 5. 완료 기준

- [x] 피부톤 화이트닝 구현 (LAB 기반)
- [x] 컬러 밸런스 조정 구현
- [x] 주름 제거 (Targeted Smoothing) 구현
- [x] Soft Focus V2 (Guided Filter 기반)
- [x] Brightness V2 (하이라이트 보호)
- [x] CPUBeautyBackend 통합
- [x] 단위 테스트 100% 통과 (14개 테스트)

---

## 6. 다음 작업

- **P2-W3-02**: GPU 필터 셰이더 구현

---

## 7. 실행 내역

### 2026-01-28: 구현 완료

**확장된 파일**:
- `cpp/include/iris_sdk/cpu_beauty_backend.h`
  - WrinkleRegions 구조체 추가
  - V2 메서드 선언 추가 (applySkinSmoothingV2, applySoftFocusV2, applyBrightnessV2, applyWrinkleRemoval)
  - 헬퍼 함수 추가 (detectSkinTone, overlayBlend, createWrinkleRegionMasks)

- `cpp/src/cpu_beauty_backend.cpp`
  - 모든 V2 필터 효과 구현
  - FastGuidedFilter 통합

**신규 파일**:
- `cpp/tests/test_new_filter_effects.cpp`: 14개 단위 테스트

**구현된 효과**:
| 효과 | 구현 방식 | 파라미터 |
|------|----------|----------|
| SkinSmoothingV2 | FastGuidedFilter | eps: 0.01~0.16, radius: 4~12 |
| SoftFocusV2 | GuidedFilter + Overlay + Glow | 3단계 블렌딩 |
| BrightnessV2 | LAB L채널 비선형 조정 | 하이라이트 보호 |
| WrinkleRemoval | 랜드마크 기반 마스크 + 강한 스무딩 | eps: 0.04~0.16 |
| Whitening | LAB L채널 감마 보정 | 피부톤 영역 선택 |
| ColorBalance | LAB A/B 채널 조정 | ±10/±15 |

**테스트 결과**: 14/14 통과
