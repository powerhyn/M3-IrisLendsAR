# P2-W2-01. ROI 기반 처리 및 페더링 통합

## 작업 정보

| 항목 | 내용 |
|------|------|
| **작업 ID** | P2-W2-01 |
| **Phase** | Phase 2: CPU 백엔드 개선 |
| **상태** | ✅ 완료 |
| **예상 기간** | 2일 |
| **의존성** | P2-W1-03 (BeautyROIManager), P2-W1-04 (BeautyProcessor) |
| **담당** | systems-programming:cpp-pro |

---

## 1. 목표

ROI 기반 선택적 필터링으로 성능 향상 및 자연스러운 경계 블렌딩 구현

### 핵심 산출물
- ROI 마스크 기반 선택적 필터 적용
- Soft Feathering으로 자연스러운 경계 처리
- 눈/입술 영역 보호 로직
- 예상 성능 개선: 30-50%

---

## 2. 상세 작업

### 2.1 ROI 기반 필터 적용 아키텍처

```
전체 프레임 (1920x1080)
    │
    ▼
Face Mesh 478 랜드마크
    │
    ▼
BeautyROIManager.computeROI()
    │
    ├── 얼굴 바운딩 박스 (예: 400x500)
    ├── 피부 마스크 (Convex Hull)
    ├── 눈 보호 마스크 (좌/우)
    └── 입술 보호 마스크
    │
    ▼
ROI 영역만 필터 적용 → 성능 30-50% 향상
    │
    ▼
Soft Feathering으로 경계 블렌딩
    │
    ▼
원본 프레임에 합성
```

### 2.2 ROI 추출 및 패딩

**파일**: `cpp/src/beauty_roi_manager.cpp` (확장)

```cpp
namespace iris_sdk {

bool BeautyROIManager::extractROIRegion(
    const cv::Mat& full_frame,
    const BeautyROI& roi,
    cv::Mat& out_roi_region,
    cv::Rect& out_actual_rect,
    int padding) {

    // 패딩 적용된 ROI 계산
    int padded_x = std::max(0, roi.x - padding);
    int padded_y = std::max(0, roi.y - padding);
    int padded_right = std::min(full_frame.cols, roi.x + roi.width + padding);
    int padded_bottom = std::min(full_frame.rows, roi.y + roi.height + padding);

    out_actual_rect = cv::Rect(
        padded_x, padded_y,
        padded_right - padded_x,
        padded_bottom - padded_y
    );

    // ROI 추출
    out_roi_region = full_frame(out_actual_rect).clone();

    return true;
}

void BeautyROIManager::applyROIRegion(
    cv::Mat& full_frame,
    const cv::Mat& roi_region,
    const cv::Rect& actual_rect,
    const cv::Mat& feather_mask) {

    cv::Mat roi_target = full_frame(actual_rect);

    if (feather_mask.empty()) {
        // 직접 복사
        roi_region.copyTo(roi_target);
    } else {
        // 페더링 마스크로 블렌딩
        cv::Mat mask_3ch;
        cv::cvtColor(feather_mask, mask_3ch, cv::COLOR_GRAY2BGR);
        mask_3ch.convertTo(mask_3ch, CV_32F, 1.0 / 255.0);

        cv::Mat roi_f, target_f;
        roi_region.convertTo(roi_f, CV_32F);
        roi_target.convertTo(target_f, CV_32F);

        // 블렌딩: result = roi * mask + original * (1 - mask)
        cv::Mat blended = roi_f.mul(mask_3ch) + target_f.mul(cv::Scalar(1, 1, 1) - mask_3ch);
        blended.convertTo(roi_target, CV_8U);
    }
}

} // namespace iris_sdk
```

### 2.3 Soft Feathering 구현

**파일**: `cpp/src/beauty_roi_manager.cpp` (확장)

```cpp
cv::Mat BeautyROIManager::createFeatherMask(
    const BeautyROI& roi,
    int feather_radius,
    const cv::Mat& skin_mask,
    const cv::Mat& eye_mask,
    const cv::Mat& lip_mask) {

    // 1. 기본 피부 마스크에서 시작
    cv::Mat combined_mask = skin_mask.clone();

    // 2. 눈/입술 영역 제외 (보호)
    if (!eye_mask.empty()) {
        combined_mask.setTo(0, eye_mask);
    }
    if (!lip_mask.empty()) {
        combined_mask.setTo(0, lip_mask);
    }

    // 3. 가우시안 블러로 소프트 페더링
    cv::Mat feathered;
    int blur_size = feather_radius * 2 + 1;
    cv::GaussianBlur(combined_mask, feathered, cv::Size(blur_size, blur_size), 0);

    // 4. ROI 영역으로 크롭
    cv::Rect roi_rect(roi.x, roi.y, roi.width, roi.height);

    // 마스크가 ROI보다 클 수 있으므로 안전한 교차 영역 계산
    cv::Rect mask_rect(0, 0, feathered.cols, feathered.rows);
    cv::Rect intersect = roi_rect & mask_rect;

    if (intersect.empty()) {
        return cv::Mat::zeros(roi.height, roi.width, CV_8U);
    }

    // ROI 기준 오프셋 계산
    int offset_x = intersect.x - roi.x;
    int offset_y = intersect.y - roi.y;

    cv::Mat result = cv::Mat::zeros(roi.height, roi.width, CV_8U);
    cv::Mat src_region = feathered(intersect);
    cv::Mat dst_region = result(cv::Rect(offset_x, offset_y, intersect.width, intersect.height));
    src_region.copyTo(dst_region);

    return result;
}
```

### 2.4 눈/입술 보호 로직

**파일**: `cpp/include/iris_sdk/beauty_roi_manager.h` (확장)

```cpp
struct ProtectionMasks {
    cv::Mat left_eye;       ///< 왼쪽 눈 보호 마스크
    cv::Mat right_eye;      ///< 오른쪽 눈 보호 마스크
    cv::Mat lips;           ///< 입술 보호 마스크
    cv::Mat combined;       ///< 통합 보호 마스크

    bool valid = false;
};

class BeautyROIManager {
public:
    // ... 기존 메서드 ...

    /**
     * @brief 보호 영역 마스크 생성
     *
     * @param face_mesh 478개 랜드마크
     * @param width 프레임 너비
     * @param height 프레임 높이
     * @param config 필터 설정 (protectEyes, protectLips)
     * @param out_masks 출력 마스크들
     * @return 성공 여부
     */
    static bool createProtectionMasks(
        const IrisLandmark* face_mesh,
        int width, int height,
        const BeautyFilterConfigV2& config,
        ProtectionMasks& out_masks
    );

private:
    /**
     * @brief 눈 영역 마스크 생성 (확장)
     *
     * @param face_mesh 478개 랜드마크
     * @param width 프레임 너비
     * @param height 프레임 높이
     * @param expansion_ratio 확장 비율 (1.2 = 20% 확장)
     * @param out_left 왼쪽 눈 마스크
     * @param out_right 오른쪽 눈 마스크
     */
    static void createEyeMasks(
        const IrisLandmark* face_mesh,
        int width, int height,
        float expansion_ratio,
        cv::Mat& out_left,
        cv::Mat& out_right
    );

    /**
     * @brief 입술 영역 마스크 생성
     */
    static void createLipMask(
        const IrisLandmark* face_mesh,
        int width, int height,
        float expansion_ratio,
        cv::Mat& out_lips
    );
};
```

### 2.5 보호 마스크 구현

**파일**: `cpp/src/beauty_roi_manager.cpp` (확장)

```cpp
// 입술 랜드마크 인덱스 (MediaPipe Face Mesh)
const int LIP_OUTER_INDICES[] = {
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    409, 270, 269, 267, 0, 37, 39, 40, 185
};
const int LIP_OUTER_COUNT = 20;

const int LIP_INNER_INDICES[] = {
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    415, 310, 311, 312, 13, 82, 81, 80, 191
};
const int LIP_INNER_COUNT = 20;

bool BeautyROIManager::createProtectionMasks(
    const IrisLandmark* face_mesh,
    int width, int height,
    const BeautyFilterConfigV2& config,
    ProtectionMasks& out_masks) {

    out_masks.combined = cv::Mat::zeros(height, width, CV_8U);
    out_masks.valid = false;

    // 눈 보호
    if (config.protectEyes) {
        createEyeMasks(face_mesh, width, height, 1.3f,
                       out_masks.left_eye, out_masks.right_eye);
        out_masks.combined |= out_masks.left_eye;
        out_masks.combined |= out_masks.right_eye;
    }

    // 입술 보호
    if (config.protectLips) {
        createLipMask(face_mesh, width, height, 1.2f, out_masks.lips);
        out_masks.combined |= out_masks.lips;
    }

    out_masks.valid = true;
    return true;
}

void BeautyROIManager::createEyeMasks(
    const IrisLandmark* face_mesh,
    int width, int height,
    float expansion_ratio,
    cv::Mat& out_left,
    cv::Mat& out_right) {

    out_left = cv::Mat::zeros(height, width, CV_8U);
    out_right = cv::Mat::zeros(height, width, CV_8U);

    // 왼쪽 눈
    std::vector<cv::Point> left_pts;
    for (int i = 0; i < LEFT_EYE_COUNT; ++i) {
        int idx = LEFT_EYE_INDICES[i];
        left_pts.emplace_back(
            static_cast<int>(face_mesh[idx].x * width),
            static_cast<int>(face_mesh[idx].y * height)
        );
    }

    // 바운딩 박스 확장
    cv::Rect left_rect = cv::boundingRect(left_pts);
    int expand_w = static_cast<int>(left_rect.width * (expansion_ratio - 1.0f) / 2);
    int expand_h = static_cast<int>(left_rect.height * (expansion_ratio - 1.0f) / 2);
    left_rect.x = std::max(0, left_rect.x - expand_w);
    left_rect.y = std::max(0, left_rect.y - expand_h);
    left_rect.width = std::min(width - left_rect.x, left_rect.width + expand_w * 2);
    left_rect.height = std::min(height - left_rect.y, left_rect.height + expand_h * 2);

    // 타원으로 마스크 생성 (더 자연스러움)
    cv::ellipse(out_left,
                cv::Point(left_rect.x + left_rect.width / 2,
                          left_rect.y + left_rect.height / 2),
                cv::Size(left_rect.width / 2, left_rect.height / 2),
                0, 0, 360, cv::Scalar(255), -1);

    // 오른쪽 눈 (동일 로직)
    std::vector<cv::Point> right_pts;
    for (int i = 0; i < RIGHT_EYE_COUNT; ++i) {
        int idx = RIGHT_EYE_INDICES[i];
        right_pts.emplace_back(
            static_cast<int>(face_mesh[idx].x * width),
            static_cast<int>(face_mesh[idx].y * height)
        );
    }

    cv::Rect right_rect = cv::boundingRect(right_pts);
    expand_w = static_cast<int>(right_rect.width * (expansion_ratio - 1.0f) / 2);
    expand_h = static_cast<int>(right_rect.height * (expansion_ratio - 1.0f) / 2);
    right_rect.x = std::max(0, right_rect.x - expand_w);
    right_rect.y = std::max(0, right_rect.y - expand_h);
    right_rect.width = std::min(width - right_rect.x, right_rect.width + expand_w * 2);
    right_rect.height = std::min(height - right_rect.y, right_rect.height + expand_h * 2);

    cv::ellipse(out_right,
                cv::Point(right_rect.x + right_rect.width / 2,
                          right_rect.y + right_rect.height / 2),
                cv::Size(right_rect.width / 2, right_rect.height / 2),
                0, 0, 360, cv::Scalar(255), -1);
}

void BeautyROIManager::createLipMask(
    const IrisLandmark* face_mesh,
    int width, int height,
    float expansion_ratio,
    cv::Mat& out_lips) {

    out_lips = cv::Mat::zeros(height, width, CV_8U);

    // 외곽 입술 폴리곤
    std::vector<cv::Point> outer_pts;
    for (int i = 0; i < LIP_OUTER_COUNT; ++i) {
        int idx = LIP_OUTER_INDICES[i];
        outer_pts.emplace_back(
            static_cast<int>(face_mesh[idx].x * width),
            static_cast<int>(face_mesh[idx].y * height)
        );
    }

    // 확장된 바운딩 박스로 타원 마스크
    cv::Rect lip_rect = cv::boundingRect(outer_pts);
    int expand_w = static_cast<int>(lip_rect.width * (expansion_ratio - 1.0f) / 2);
    int expand_h = static_cast<int>(lip_rect.height * (expansion_ratio - 1.0f) / 2);
    lip_rect.x = std::max(0, lip_rect.x - expand_w);
    lip_rect.y = std::max(0, lip_rect.y - expand_h);
    lip_rect.width = std::min(width - lip_rect.x, lip_rect.width + expand_w * 2);
    lip_rect.height = std::min(height - lip_rect.y, lip_rect.height + expand_h * 2);

    cv::ellipse(out_lips,
                cv::Point(lip_rect.x + lip_rect.width / 2,
                          lip_rect.y + lip_rect.height / 2),
                cv::Size(lip_rect.width / 2, lip_rect.height / 2),
                0, 0, 360, cv::Scalar(255), -1);
}
```

### 2.6 CPUBeautyBackend ROI 처리 통합

**파일**: `cpp/src/cpu_beauty_backend.cpp` (확장)

```cpp
IrisSdkError CPUBeautyBackend::applyWithROI(
    cv::Mat& frame,
    const BeautyFilterConfigV2& config,
    const BeautyROI& roi) {

    std::lock_guard<std::mutex> lock(buffer_mutex_);

    // 1. ROI 영역 추출 (패딩 포함)
    const int PADDING = 20;  // 블러 경계 아티팩트 방지
    cv::Mat roi_region;
    cv::Rect actual_rect;

    if (!BeautyROIManager::extractROIRegion(frame, roi, roi_region, actual_rect, PADDING)) {
        return IRIS_SDK_ERROR_INTERNAL;
    }

    // 2. 보호 마스크 생성 (ROI 좌표계로 변환)
    cv::Mat protection_mask;
    if (roi.skin_mask_valid) {
        // ROI 영역에 해당하는 보호 마스크 크롭
        cv::Mat full_protection = roi.protection_mask;
        if (!full_protection.empty()) {
            cv::Rect mask_roi(actual_rect.x, actual_rect.y,
                              std::min(actual_rect.width, full_protection.cols - actual_rect.x),
                              std::min(actual_rect.height, full_protection.rows - actual_rect.y));
            protection_mask = full_protection(mask_roi).clone();
        }
    }

    // 3. ROI 영역에 필터 적용
    if (config.smoothing > 0.01f) {
        applySkinSmoothing(roi_region, config.smoothing, protection_mask);
    }

    if (config.softFocus > 0.01f) {
        applySoftFocus(roi_region, config.softFocus);
    }

    if (std::abs(config.brightness - 1.0f) > 0.01f) {
        applyBrightness(roi_region, config.brightness);
    }

    if (config.whitening > 0.01f) {
        applyWhitening(roi_region, config.whitening, protection_mask);
    }

    // 4. 페더링 마스크 생성
    cv::Mat feather_mask = BeautyROIManager::createFeatherMask(
        roi, config.featherRadius,
        roi.skin_mask, roi.eye_mask, roi.lip_mask
    );

    // 5. 원본에 합성
    BeautyROIManager::applyROIRegion(frame, roi_region, actual_rect, feather_mask);

    return IRIS_SDK_OK;
}

void CPUBeautyBackend::applySkinSmoothing(
    cv::Mat& roi,
    float strength,
    const cv::Mat& protection_mask) {

    // 기존 Bilateral Filter 적용
    cv::Mat smoothed;
    int d = 9;
    double sigmaColor = 75 * strength;
    double sigmaSpace = 75 * strength;
    cv::bilateralFilter(roi, smoothed, d, sigmaColor, sigmaSpace);

    // 보호 마스크 적용 (눈/입술 보호)
    if (!protection_mask.empty()) {
        cv::Mat mask_inv;
        cv::bitwise_not(protection_mask, mask_inv);

        cv::Mat mask_3ch;
        cv::cvtColor(mask_inv, mask_3ch, cv::COLOR_GRAY2BGR);
        mask_3ch.convertTo(mask_3ch, CV_32F, 1.0 / 255.0);

        cv::Mat roi_f, smoothed_f;
        roi.convertTo(roi_f, CV_32F);
        smoothed.convertTo(smoothed_f, CV_32F);

        // 보호 영역은 원본, 나머지는 스무딩
        cv::Mat result = smoothed_f.mul(mask_3ch) +
                         roi_f.mul(cv::Scalar(1, 1, 1) - mask_3ch);
        result.convertTo(roi, CV_8U);
    } else {
        smoothed.copyTo(roi);
    }
}
```

---

## 3. 성능 최적화

### 3.1 다운스케일 처리

```cpp
IrisSdkError CPUBeautyBackend::apply(
    uint8_t* frame_data,
    int width, int height,
    IrisFrameFormat format,
    const BeautyFilterConfigV2& config,
    const BeautyROI* roi) {

    cv::Mat frame = convertToBGR(frame_data, width, height, format);

    // 다운스케일 옵션 적용
    cv::Mat work_frame;
    int scale = config.downscaleFactor;

    if (scale > 1) {
        cv::resize(frame, work_frame,
                   cv::Size(width / scale, height / scale),
                   0, 0, cv::INTER_LINEAR);
    } else {
        work_frame = frame;
    }

    // ROI도 스케일 조정
    BeautyROI scaled_roi;
    if (roi && roi->valid) {
        scaled_roi = *roi;
        scaled_roi.x /= scale;
        scaled_roi.y /= scale;
        scaled_roi.width /= scale;
        scaled_roi.height /= scale;
        // 마스크도 리사이즈 필요...
    }

    // 필터 적용
    IrisSdkError err = applyWithROI(work_frame, config,
                                    roi ? scaled_roi : BeautyROI{});

    // 업스케일
    if (scale > 1) {
        cv::resize(work_frame, frame,
                   cv::Size(width, height),
                   0, 0, cv::INTER_LINEAR);
    }

    convertFromBGR(frame, frame_data, format);
    return err;
}
```

### 3.2 예상 성능 개선

| 시나리오 | 전체 프레임 | ROI Only | 개선율 |
|----------|-------------|----------|--------|
| 1080p 피부 스무딩 | 25-30ms | 8-12ms | ~60% |
| 1080p 전체 필터 | 35-43ms | 15-22ms | ~45% |
| 720p ROI | - | 6-10ms | - |
| 1080p + 다운스케일(2x) | - | 4-6ms | ~80% |

---

## 4. 단위 테스트

**파일**: `cpp/tests/test_roi_processing.cpp`

```cpp
TEST(ROIProcessing, ExtractAndApplyROI) {
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(100, 100, 100));
    BeautyROI roi{100, 100, 200, 200, true};

    cv::Mat roi_region;
    cv::Rect actual_rect;

    ASSERT_TRUE(BeautyROIManager::extractROIRegion(
        frame, roi, roi_region, actual_rect, 10));

    EXPECT_EQ(actual_rect.x, 90);   // 100 - padding(10)
    EXPECT_EQ(actual_rect.y, 90);
    EXPECT_EQ(actual_rect.width, 220);  // 200 + padding*2
    EXPECT_EQ(actual_rect.height, 220);
}

TEST(ROIProcessing, FeatherMaskCreation) {
    cv::Mat skin_mask = cv::Mat::zeros(480, 640, CV_8U);
    cv::circle(skin_mask, cv::Point(200, 200), 100, cv::Scalar(255), -1);

    BeautyROI roi{100, 100, 200, 200, true};

    cv::Mat feather = BeautyROIManager::createFeatherMask(
        roi, 15, skin_mask, cv::Mat(), cv::Mat());

    EXPECT_EQ(feather.rows, 200);
    EXPECT_EQ(feather.cols, 200);

    // 중앙은 255, 가장자리는 그라데이션
    EXPECT_GT(feather.at<uint8_t>(100, 100), 200);  // 중앙
}

TEST(ROIProcessing, ProtectionMasksExcludeEyesAndLips) {
    // Mock face mesh 생성
    IrisLandmark face_mesh[478];
    // ... 테스트용 랜드마크 설정 ...

    BeautyFilterConfigV2 config;
    config.protectEyes = true;
    config.protectLips = true;

    ProtectionMasks masks;
    ASSERT_TRUE(BeautyROIManager::createProtectionMasks(
        face_mesh, 640, 480, config, masks));

    EXPECT_TRUE(masks.valid);
    EXPECT_FALSE(masks.left_eye.empty());
    EXPECT_FALSE(masks.lips.empty());

    // combined 마스크에 눈/입술 포함 확인
    int non_zero = cv::countNonZero(masks.combined);
    EXPECT_GT(non_zero, 0);
}

TEST(ROIProcessing, DownscaleImprovesPerfomance) {
    cv::Mat frame(1080, 1920, CV_8UC3);
    cv::randu(frame, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

    CPUBeautyBackend backend;
    backend.initialize();

    BeautyFilterConfigV2 config;
    config.enabled = true;
    config.smoothing = 0.8f;

    // Scale 1 (원본)
    config.downscaleFactor = 1;
    auto start1 = std::chrono::high_resolution_clock::now();
    backend.apply(frame.data, 1920, 1080, IRIS_FORMAT_RGB, config, nullptr);
    auto dur1 = std::chrono::high_resolution_clock::now() - start1;

    // Scale 2 (1/2)
    config.downscaleFactor = 2;
    auto start2 = std::chrono::high_resolution_clock::now();
    backend.apply(frame.data, 1920, 1080, IRIS_FORMAT_RGB, config, nullptr);
    auto dur2 = std::chrono::high_resolution_clock::now() - start2;

    // 다운스케일이 더 빠름
    EXPECT_LT(dur2.count(), dur1.count());
}
```

---

## 5. 완료 기준

- [x] ROI 추출/합성 로직 구현 (extractROIRegion, applyROIRegion)
- [x] Soft Feathering 마스크 생성 (createFeatherMask)
- [x] 눈/입술 보호 마스크 구현 (createProtectionMasks, createEyeMasks, createLipMask)
- [x] CPUBeautyBackend ROI 처리 통합 (applyWithROI 개선, applySkinSmoothing/applyWhitening 오버로드)
- [ ] 다운스케일 옵션 구현 (P2-W2-02에서 별도 구현)
- [x] 단위 테스트 100% 통과 (44개 테스트 통과)
- [ ] 성능 측정 (30-50% 개선 확인) - 벤치마크 필요

---

## 6. 실행 내역

### 2024-01-28

#### 구현 완료 항목

1. **BeautyROIManager 확장** (`cpp/include/iris_sdk/beauty_roi_manager.h`, `cpp/src/beauty_roi_manager.cpp`)
   - `ProtectionMasks` 구조체 추가
   - `extractROIRegion()` - ROI 영역 추출 (패딩 포함)
   - `applyROIRegion()` - ROI 영역 합성 (페더링 마스크 적용)
   - `createFeatherMask()` - 소프트 페더링 마스크 생성 (cv::Mat 반환)
   - `createProtectionMasks()` - 통합 보호 마스크 생성
   - `createEyeMasks()` - 눈 영역 타원 마스크 (확장 비율 적용)
   - `createLipMask()` - 입술 영역 마스크
   - `LIP_OUTER_INDICES[20]` 랜드마크 인덱스 추가

2. **CPUBeautyBackend 확장** (`cpp/include/iris_sdk/cpu_beauty_backend.h`, `cpp/src/cpu_beauty_backend.cpp`)
   - `applySkinSmoothing(cv::Mat&, float, const cv::Mat&)` 오버로드 - 보호 마스크 지원
   - `applyWhitening(cv::Mat&, float, const cv::Mat&)` 오버로드 - 보호 마스크 지원
   - `applyWithROI()` 개선 - BeautyROIManager 통합, 보호 마스크/페더링 적용

3. **단위 테스트 추가** (`cpp/tests/test_beauty_roi_manager.cpp`)
   - `ProtectionMasksTest` - 기본 생성 테스트
   - `CreateProtectionMasks` 테스트 (6개)
   - `CreateEyeMasks` 테스트 (2개)
   - `CreateLipMask` 테스트 (1개)
   - OpenCV 전용 테스트 (9개): extractROIRegion, applyROIRegion, createFeatherMask

#### 빌드 결과
- `iris_sdk` 라이브러리 빌드 성공
- `test_beauty_roi_manager` 44개 테스트 모두 통과

---

## 7. 다음 작업

- **P2-W2-02**: Fast Guided Filter 직접 구현
