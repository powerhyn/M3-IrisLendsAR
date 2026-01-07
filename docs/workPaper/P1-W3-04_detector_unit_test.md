# P1-W3-04: 검출기 단위 테스트

**태스크 ID**: P1-W3-04
**상태**: ⏳ 대기
**시작일**: -
**완료일**: -

---

## 1. 계획

### 목표
MediaPipeDetector의 단위 테스트를 작성하여 검출 기능의 정확성과 안정성을 검증한다.

### 산출물
| 파일 | 설명 |
|------|------|
| `cpp/tests/test_iris_detector.cpp` | IrisDetector 단위 테스트 |
| `shared/test_data/*.jpg` | 테스트용 이미지 |
| `shared/test_data/expected/*.json` | 예상 결과 (Ground Truth) |

### 검증 기준
- [ ] 모든 테스트 케이스 통과
- [ ] 정상 조건 정확도 95% 이상
- [ ] 경계 조건 처리 확인
- [ ] 에러 케이스 적절한 처리
- [ ] 성능 기준 충족 (33ms 이하)

### 선행 조건
- P1-W3-03: MediaPipeDetector 구현 ✅

---

## 2. 분석

### 2.1 테스트 프레임워크

**Google Test (gtest) 사용**
- C++ 표준 단위 테스트 프레임워크
- Fixture 지원으로 초기화/정리 자동화
- Parameterized 테스트로 다양한 입력 검증

### 2.2 테스트 카테고리

| 카테고리 | 설명 | 테스트 수 |
|----------|------|----------|
| 초기화 | 모델 로드, 설정 | 5 |
| 검출 정확도 | 다양한 조건 이미지 | 10 |
| 경계 조건 | 빈 이미지, 큰 이미지 등 | 6 |
| 에러 처리 | 잘못된 입력 처리 | 4 |
| 성능 | 처리 시간 검증 | 2 |

### 2.3 테스트 케이스 상세

**2.3.1 초기화 테스트**
```cpp
TEST(IrisDetectorTest, InitializeWithValidPath) {
    auto detector = createDetector(DetectorType::MediaPipe);
    EXPECT_TRUE(detector->initialize("./models"));
    EXPECT_TRUE(detector->isInitialized());
}

TEST(IrisDetectorTest, InitializeWithInvalidPath) {
    auto detector = createDetector(DetectorType::MediaPipe);
    EXPECT_FALSE(detector->initialize("/invalid/path"));
    EXPECT_FALSE(detector->isInitialized());
}

TEST(IrisDetectorTest, DoubleInitialize) {
    auto detector = createDetector(DetectorType::MediaPipe);
    EXPECT_TRUE(detector->initialize("./models"));
    // 이미 초기화된 상태에서 재초기화 시도
    EXPECT_TRUE(detector->initialize("./models")); // 또는 EXPECT_FALSE
}

TEST(IrisDetectorTest, ReleaseAndReinitialize) {
    auto detector = createDetector(DetectorType::MediaPipe);
    EXPECT_TRUE(detector->initialize("./models"));
    detector->release();
    EXPECT_FALSE(detector->isInitialized());
    EXPECT_TRUE(detector->initialize("./models"));
}

TEST(IrisDetectorTest, GetDetectorType) {
    auto detector = createDetector(DetectorType::MediaPipe);
    EXPECT_EQ(detector->getDetectorType(), DetectorType::MediaPipe);
}
```

**2.3.2 검출 정확도 테스트**
```cpp
class IrisDetectorAccuracyTest : public ::testing::TestWithParam<std::string> {
protected:
    void SetUp() override {
        detector_ = createDetector(DetectorType::MediaPipe);
        detector_->initialize("./models");
    }

    void TearDown() override {
        detector_->release();
    }

    std::unique_ptr<IrisDetector> detector_;
};

TEST_P(IrisDetectorAccuracyTest, DetectIrisInImage) {
    std::string image_path = GetParam();
    cv::Mat frame = cv::imread(image_path);
    ASSERT_FALSE(frame.empty());

    IrisResult result = detector_->detect(frame);

    // 검출 성공 확인
    EXPECT_TRUE(result.detected);
    EXPECT_GT(result.confidence, 0.8f);

    // Ground truth와 비교
    auto expected = loadExpectedResult(image_path);
    EXPECT_NEAR(result.left_iris[0].x, expected.left_iris[0].x, 0.05f);
    EXPECT_NEAR(result.left_iris[0].y, expected.left_iris[0].y, 0.05f);
}

INSTANTIATE_TEST_SUITE_P(
    ImageTests,
    IrisDetectorAccuracyTest,
    ::testing::Values(
        "./test_data/frontal_face.jpg",
        "./test_data/side_face_15.jpg",
        "./test_data/side_face_30.jpg",
        "./test_data/with_glasses.jpg",
        "./test_data/different_lighting.jpg"
    )
);
```

**2.3.3 경계 조건 테스트**
```cpp
TEST(IrisDetectorEdgeCaseTest, EmptyImage) {
    auto detector = createDetector(DetectorType::MediaPipe);
    detector->initialize("./models");

    cv::Mat empty_frame;
    IrisResult result = detector->detect(empty_frame);

    EXPECT_FALSE(result.detected);
}

TEST(IrisDetectorEdgeCaseTest, VerySmallImage) {
    auto detector = createDetector(DetectorType::MediaPipe);
    detector->initialize("./models");

    cv::Mat small_frame(10, 10, CV_8UC3, cv::Scalar(128, 128, 128));
    IrisResult result = detector->detect(small_frame);

    EXPECT_FALSE(result.detected);
}

TEST(IrisDetectorEdgeCaseTest, VeryLargeImage) {
    auto detector = createDetector(DetectorType::MediaPipe);
    detector->initialize("./models");

    cv::Mat large_frame(4000, 4000, CV_8UC3);
    // 실제 얼굴 이미지 로드 후 리사이즈

    IrisResult result = detector->detect(large_frame);
    EXPECT_TRUE(result.detected);  // 내부 리사이즈 처리
}

TEST(IrisDetectorEdgeCaseTest, NoFaceInImage) {
    auto detector = createDetector(DetectorType::MediaPipe);
    detector->initialize("./models");

    cv::Mat no_face = cv::imread("./test_data/landscape.jpg");
    IrisResult result = detector->detect(no_face);

    EXPECT_FALSE(result.detected);
}

TEST(IrisDetectorEdgeCaseTest, PartialFace) {
    auto detector = createDetector(DetectorType::MediaPipe);
    detector->initialize("./models");

    cv::Mat partial = cv::imread("./test_data/partial_face.jpg");
    IrisResult result = detector->detect(partial);

    // 부분 얼굴은 검출 실패 또는 낮은 신뢰도
    if (result.detected) {
        EXPECT_LT(result.confidence, 0.8f);
    }
}

TEST(IrisDetectorEdgeCaseTest, GrayscaleImage) {
    auto detector = createDetector(DetectorType::MediaPipe);
    detector->initialize("./models");

    cv::Mat color = cv::imread("./test_data/frontal_face.jpg");
    cv::Mat gray;
    cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);

    IrisResult result = detector->detect(gray);
    EXPECT_TRUE(result.detected);  // 그레이스케일 지원
}
```

**2.3.4 에러 처리 테스트**
```cpp
TEST(IrisDetectorErrorTest, DetectWithoutInitialize) {
    auto detector = createDetector(DetectorType::MediaPipe);
    // initialize() 호출 안함

    cv::Mat frame = cv::imread("./test_data/frontal_face.jpg");
    IrisResult result = detector->detect(frame);

    EXPECT_FALSE(result.detected);
}

TEST(IrisDetectorErrorTest, ReleaseMultipleTimes) {
    auto detector = createDetector(DetectorType::MediaPipe);
    detector->initialize("./models");

    // 여러 번 release 호출해도 크래시 없음
    detector->release();
    detector->release();
    detector->release();

    EXPECT_FALSE(detector->isInitialized());
}
```

**2.3.5 성능 테스트**
```cpp
TEST(IrisDetectorPerformanceTest, DetectionTime) {
    auto detector = createDetector(DetectorType::MediaPipe);
    detector->initialize("./models");

    cv::Mat frame = cv::imread("./test_data/frontal_face.jpg");

    // 워밍업
    for (int i = 0; i < 5; ++i) {
        detector->detect(frame);
    }

    // 측정
    const int iterations = 100;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        detector->detect(frame);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    double avg_ms = static_cast<double>(duration.count()) / iterations;

    std::cout << "Average detection time: " << avg_ms << " ms" << std::endl;
    EXPECT_LT(avg_ms, 33.0);  // 30fps 기준
}

TEST(IrisDetectorPerformanceTest, MemoryUsage) {
    // 메모리 사용량 측정 (플랫폼별 구현 필요)
    auto detector = createDetector(DetectorType::MediaPipe);

    size_t mem_before = getCurrentMemoryUsage();
    detector->initialize("./models");
    size_t mem_after = getCurrentMemoryUsage();

    size_t mem_used = mem_after - mem_before;
    std::cout << "Memory used: " << (mem_used / 1024 / 1024) << " MB" << std::endl;

    EXPECT_LT(mem_used, 100 * 1024 * 1024);  // 100MB 이하
}
```

### 2.4 테스트 데이터

| 파일명 | 설명 | 조건 |
|--------|------|------|
| frontal_face.jpg | 정면 얼굴 | 정상 조명, 근거리 |
| side_face_15.jpg | 15도 측면 | 약간 회전 |
| side_face_30.jpg | 30도 측면 | 중간 회전 |
| side_face_45.jpg | 45도 측면 | 많이 회전 (한계) |
| with_glasses.jpg | 안경 착용 | 반사 있음 |
| dark_lighting.jpg | 어두운 조명 | 저조도 |
| bright_lighting.jpg | 밝은 조명 | 과노출 |
| closeup_eyes.jpg | 눈 클로즈업 | 얼굴 없음 (실패 예상) |
| multiple_faces.jpg | 다중 얼굴 | 2명 이상 |
| partial_face.jpg | 부분 얼굴 | 잘린 얼굴 |

---

## 3. 실행 내역

### 3.1 테스트 파일 작성

```bash
# 예정: cpp/tests/test_iris_detector.cpp
```

### 3.2 테스트 데이터 준비

```bash
# 예정: shared/test_data/ 디렉토리에 이미지 배치
```

### 3.3 CMake 설정 업데이트

```cmake
# 예정: cpp/tests/CMakeLists.txt 수정
```

### 3.4 테스트 실행

```bash
# 예정: ctest --output-on-failure
```

---

## 4. 검증 결과

### 검증 항목

| 항목 | 결과 | 비고 |
|------|------|------|
| 초기화 테스트 | ⏳ 대기 | 5개 |
| 정확도 테스트 | ⏳ 대기 | 10개 |
| 경계 조건 테스트 | ⏳ 대기 | 6개 |
| 에러 처리 테스트 | ⏳ 대기 | 4개 |
| 성능 테스트 | ⏳ 대기 | 2개 |

### 정확도 측정

| 조건 | 테스트 수 | 통과 | 정확도 |
|------|----------|------|--------|
| 정면 | - | - | - |
| 15도 | - | - | - |
| 30도 | - | - | - |
| 45도 | - | - | - |
| 저조도 | - | - | - |
| **전체** | - | - | - |

---

## 5. 이슈 및 학습

### 이슈

| ID | 내용 | 상태 | 해결방안 |
|----|------|------|----------|
| - | - | - | - |

### 결정 사항

| 결정 | 이유 |
|------|------|
| Google Test 사용 | C++ 표준, CMake 통합 용이 |
| 정확도 허용 오차 5% | 랜드마크 위치 변동성 고려 |
| 100회 반복 측정 | 통계적 신뢰성 확보 |

### 학습 내용

(실행 후 기록)

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-01-07 | 태스크 문서 생성, 테스트 케이스 설계 완료 |
