# P1-W4-06: 코어 통합 테스트

**태스크 ID**: P1-W4-06
**상태**: ⏳ 대기
**시작일**: -
**완료일**: -

---

## 1. 계획

### 목표
전체 코어 엔진의 통합 테스트를 수행하고, 웹캠 실시간 30fps 처리를 검증한다.

### 산출물
| 파일 | 설명 |
|------|------|
| `cpp/tests/test_integration.cpp` | 통합 테스트 코드 |
| `cpp/examples/camera_demo.cpp` | 웹캠 데모 애플리케이션 |
| `cpp/examples/image_demo.cpp` | 이미지 데모 애플리케이션 |

### 검증 기준
- [ ] 전체 파이프라인 동작 확인
- [ ] 웹캠 실시간 처리 30fps 달성
- [ ] C API를 통한 전체 흐름 검증
- [ ] 메모리 누수 없음 확인
- [ ] 연속 처리 안정성 (1000프레임+)

### 선행 조건
- P1-W4-05: C API 래퍼 구현 ✅

---

## 2. 분석

### 2.1 테스트 범위

| 범위 | 테스트 항목 |
|------|------------|
| 단위 통합 | 컴포넌트 간 인터페이스 |
| E2E 파이프라인 | 입력→검출→렌더링→출력 |
| C API | extern "C" 함수 전체 |
| 실시간 처리 | 웹캠 30fps |
| 안정성 | 장시간 연속 처리 |

### 2.2 통합 테스트 케이스

```cpp
// test_integration.cpp

#include <gtest/gtest.h>
#include "iris_sdk/sdk_api.h"
#include <opencv2/opencv.hpp>

class IntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // SDK 초기화
        IrisSdkError err = iris_sdk_init("./models");
        ASSERT_EQ(err, IRIS_SDK_OK) << "SDK init failed: "
                                    << iris_sdk_error_to_string(err);

        // 텍스처 로드
        err = iris_sdk_load_texture("./textures/sample_lens.png");
        ASSERT_EQ(err, IRIS_SDK_OK) << "Texture load failed";
    }

    void TearDown() override {
        iris_sdk_destroy();
    }
};

/**
 * 전체 파이프라인 테스트 (정적 이미지)
 */
TEST_F(IntegrationTest, FullPipelineWithImage) {
    // 테스트 이미지 로드
    cv::Mat image = cv::imread("./test_data/frontal_face.jpg");
    ASSERT_FALSE(image.empty());

    // BGR → RGBA 변환
    cv::Mat rgba;
    cv::cvtColor(image, rgba, cv::COLOR_BGR2RGBA);

    // 렌즈 설정
    IrisLensConfig config;
    iris_sdk_default_lens_config(&config);
    config.opacity = 0.8f;

    // 처리
    IrisResult result;
    IrisSdkError err = iris_sdk_process(
        rgba.data,
        rgba.cols,
        rgba.rows,
        IRIS_FORMAT_RGBA,
        &config,
        &result
    );

    // 검증
    EXPECT_EQ(err, IRIS_SDK_OK);
    EXPECT_TRUE(result.detected);
    EXPECT_GT(result.confidence, 0.8f);
    EXPECT_TRUE(result.left_detected);
    EXPECT_TRUE(result.right_detected);

    // 결과 저장 (시각적 확인용)
    cv::Mat output;
    cv::cvtColor(rgba, output, cv::COLOR_RGBA2BGR);
    cv::imwrite("./test_output/integration_result.jpg", output);
}

/**
 * C API 전체 흐름 테스트
 */
TEST_F(IntegrationTest, CAPIFullFlow) {
    // 1. 버전 확인
    const char* version = iris_sdk_get_version();
    EXPECT_NE(version, nullptr);
    EXPECT_STREQ(version, "0.1.0");

    // 2. 상태 확인
    EXPECT_TRUE(iris_sdk_is_ready());

    // 3. 검출만 테스트
    cv::Mat image = cv::imread("./test_data/frontal_face.jpg");
    cv::Mat rgba;
    cv::cvtColor(image, rgba, cv::COLOR_BGR2RGBA);

    IrisResult result;
    IrisSdkError err = iris_sdk_detect(
        rgba.data, rgba.cols, rgba.rows,
        IRIS_FORMAT_RGBA, &result
    );
    EXPECT_EQ(err, IRIS_SDK_OK);
    EXPECT_TRUE(result.detected);

    // 4. 렌더링만 테스트
    IrisLensConfig config;
    iris_sdk_default_lens_config(&config);

    err = iris_sdk_render_lens(
        rgba.data, rgba.cols, rgba.rows,
        IRIS_FORMAT_RGBA, &result, &config
    );
    EXPECT_EQ(err, IRIS_SDK_OK);
}

/**
 * 다양한 프레임 포맷 테스트
 */
TEST_F(IntegrationTest, MultipleFrameFormats) {
    cv::Mat image = cv::imread("./test_data/frontal_face.jpg");
    IrisResult result;

    // BGR 포맷
    {
        IrisSdkError err = iris_sdk_detect(
            image.data, image.cols, image.rows,
            IRIS_FORMAT_BGR, &result
        );
        EXPECT_EQ(err, IRIS_SDK_OK);
        EXPECT_TRUE(result.detected);
    }

    // RGBA 포맷
    {
        cv::Mat rgba;
        cv::cvtColor(image, rgba, cv::COLOR_BGR2RGBA);
        IrisSdkError err = iris_sdk_detect(
            rgba.data, rgba.cols, rgba.rows,
            IRIS_FORMAT_RGBA, &result
        );
        EXPECT_EQ(err, IRIS_SDK_OK);
        EXPECT_TRUE(result.detected);
    }

    // Grayscale 포맷
    {
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        IrisSdkError err = iris_sdk_detect(
            gray.data, gray.cols, gray.rows,
            IRIS_FORMAT_GRAY, &result
        );
        EXPECT_EQ(err, IRIS_SDK_OK);
        // Grayscale도 검출 가능해야 함
    }
}

/**
 * 연속 처리 안정성 테스트
 */
TEST_F(IntegrationTest, ContinuousProcessingStability) {
    cv::Mat image = cv::imread("./test_data/frontal_face.jpg");
    cv::Mat rgba;
    cv::cvtColor(image, rgba, cv::COLOR_BGR2RGBA);

    IrisLensConfig config;
    iris_sdk_default_lens_config(&config);

    const int iterations = 1000;
    int success_count = 0;

    for (int i = 0; i < iterations; ++i) {
        IrisResult result;
        cv::Mat frame = rgba.clone();

        IrisSdkError err = iris_sdk_process(
            frame.data, frame.cols, frame.rows,
            IRIS_FORMAT_RGBA, &config, &result
        );

        if (err == IRIS_SDK_OK && result.detected) {
            success_count++;
        }
    }

    float success_rate = static_cast<float>(success_count) / iterations;
    EXPECT_GT(success_rate, 0.99f) << "Success rate: " << success_rate;

    std::cout << "Processed " << iterations << " frames, "
              << "success rate: " << (success_rate * 100) << "%" << std::endl;
}

/**
 * 에러 핸들링 테스트
 */
TEST_F(IntegrationTest, ErrorHandling) {
    IrisResult result;

    // NULL 포인터
    IrisSdkError err = iris_sdk_detect(
        nullptr, 100, 100, IRIS_FORMAT_RGBA, &result
    );
    EXPECT_EQ(err, IRIS_SDK_ERR_NULL_POINTER);

    // 결과 NULL
    cv::Mat image = cv::imread("./test_data/frontal_face.jpg");
    err = iris_sdk_detect(
        image.data, image.cols, image.rows,
        IRIS_FORMAT_BGR, nullptr
    );
    EXPECT_EQ(err, IRIS_SDK_ERR_NULL_POINTER);

    // 에러 메시지 확인
    const char* error_msg = iris_sdk_get_last_error();
    EXPECT_NE(error_msg, nullptr);
}
```

### 2.3 웹캠 데모 애플리케이션

```cpp
// camera_demo.cpp

#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "iris_sdk/sdk_api.h"

// FPS 측정 클래스
class FPSCounter {
public:
    void tick() {
        auto now = std::chrono::high_resolution_clock::now();
        if (frame_count_ > 0) {
            auto duration = std::chrono::duration<double>(now - last_time_).count();
            fps_ = 1.0 / duration;
        }
        last_time_ = now;
        frame_count_++;
    }

    double getFPS() const { return fps_; }
    int getFrameCount() const { return frame_count_; }

private:
    std::chrono::high_resolution_clock::time_point last_time_;
    double fps_ = 0.0;
    int frame_count_ = 0;
};

int main(int argc, char* argv[]) {
    // 모델 경로 (기본값 또는 인자)
    std::string model_path = "./models";
    std::string texture_path = "./textures/sample_lens.png";

    if (argc > 1) model_path = argv[1];
    if (argc > 2) texture_path = argv[2];

    // SDK 초기화
    std::cout << "Initializing IrisLensSDK v" << iris_sdk_get_version() << std::endl;

    IrisSdkError err = iris_sdk_init(model_path.c_str());
    if (err != IRIS_SDK_OK) {
        std::cerr << "Failed to init SDK: " << iris_sdk_error_to_string(err) << std::endl;
        return 1;
    }

    // 텍스처 로드
    err = iris_sdk_load_texture(texture_path.c_str());
    if (err != IRIS_SDK_OK) {
        std::cerr << "Failed to load texture: " << iris_sdk_error_to_string(err) << std::endl;
        // 텍스처 없이 검출만 진행
    }

    // 웹캠 열기
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open camera" << std::endl;
        iris_sdk_destroy();
        return 1;
    }

    // 해상도 설정
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    // 렌즈 설정
    IrisLensConfig config;
    iris_sdk_default_lens_config(&config);
    config.opacity = 0.7f;

    FPSCounter fps_counter;
    cv::Mat frame;

    std::cout << "Press 'q' to quit, '+'/'-' to adjust opacity" << std::endl;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // BGR → RGBA
        cv::Mat rgba;
        cv::cvtColor(frame, rgba, cv::COLOR_BGR2RGBA);

        // 처리
        IrisResult result;
        auto start = std::chrono::high_resolution_clock::now();

        err = iris_sdk_process(
            rgba.data, rgba.cols, rgba.rows,
            IRIS_FORMAT_RGBA, &config, &result
        );

        auto end = std::chrono::high_resolution_clock::now();
        float process_ms = std::chrono::duration<float, std::milli>(end - start).count();

        // RGBA → BGR (표시용)
        cv::cvtColor(rgba, frame, cv::COLOR_RGBA2BGR);

        // FPS 측정
        fps_counter.tick();

        // 정보 표시
        std::string info = cv::format(
            "FPS: %.1f | Process: %.1fms | Detected: %s",
            fps_counter.getFPS(),
            process_ms,
            result.detected ? "Yes" : "No"
        );
        cv::putText(frame, info, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        if (result.detected) {
            std::string conf_info = cv::format(
                "Confidence: %.2f | L: %s R: %s",
                result.confidence,
                result.left_detected ? "O" : "X",
                result.right_detected ? "O" : "X"
            );
            cv::putText(frame, conf_info, cv::Point(10, 60),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

            // 검출된 위치 표시 (디버그용)
            if (result.left_detected) {
                cv::Point center(
                    result.left_iris[0].x * frame.cols,
                    result.left_iris[0].y * frame.rows
                );
                cv::circle(frame, center, 3, cv::Scalar(255, 0, 0), -1);
            }
            if (result.right_detected) {
                cv::Point center(
                    result.right_iris[0].x * frame.cols,
                    result.right_iris[0].y * frame.rows
                );
                cv::circle(frame, center, 3, cv::Scalar(0, 0, 255), -1);
            }
        }

        // 표시
        cv::imshow("IrisLensSDK Demo", frame);

        // 키 입력
        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q') break;
        if (key == '+' || key == '=') {
            config.opacity = std::min(1.0f, config.opacity + 0.1f);
            std::cout << "Opacity: " << config.opacity << std::endl;
        }
        if (key == '-') {
            config.opacity = std::max(0.0f, config.opacity - 0.1f);
            std::cout << "Opacity: " << config.opacity << std::endl;
        }
    }

    // 정리
    cap.release();
    cv::destroyAllWindows();
    iris_sdk_destroy();

    std::cout << "Total frames: " << fps_counter.getFrameCount() << std::endl;
    return 0;
}
```

### 2.4 이미지 데모 애플리케이션

```cpp
// image_demo.cpp

#include <iostream>
#include <opencv2/opencv.hpp>
#include "iris_sdk/sdk_api.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path> [output_path]" << std::endl;
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argc > 2 ? argv[2] : "output.jpg";

    // SDK 초기화
    IrisSdkError err = iris_sdk_init("./models");
    if (err != IRIS_SDK_OK) {
        std::cerr << "Init failed: " << iris_sdk_error_to_string(err) << std::endl;
        return 1;
    }

    // 텍스처 로드
    iris_sdk_load_texture("./textures/sample_lens.png");

    // 이미지 로드
    cv::Mat image = cv::imread(input_path);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << input_path << std::endl;
        iris_sdk_destroy();
        return 1;
    }

    // 처리
    cv::Mat rgba;
    cv::cvtColor(image, rgba, cv::COLOR_BGR2RGBA);

    IrisLensConfig config;
    iris_sdk_default_lens_config(&config);

    IrisResult result;
    err = iris_sdk_process(
        rgba.data, rgba.cols, rgba.rows,
        IRIS_FORMAT_RGBA, &config, &result
    );

    if (err == IRIS_SDK_OK && result.detected) {
        std::cout << "Detection successful!" << std::endl;
        std::cout << "  Confidence: " << result.confidence << std::endl;
        std::cout << "  Left eye: " << (result.left_detected ? "Yes" : "No") << std::endl;
        std::cout << "  Right eye: " << (result.right_detected ? "Yes" : "No") << std::endl;

        // 결과 저장
        cv::Mat output;
        cv::cvtColor(rgba, output, cv::COLOR_RGBA2BGR);
        cv::imwrite(output_path, output);
        std::cout << "Output saved to: " << output_path << std::endl;
    } else {
        std::cout << "Detection failed or no face found" << std::endl;
    }

    iris_sdk_destroy();
    return 0;
}
```

### 2.5 성능 측정 기준

| 항목 | 목표 | 측정 방법 |
|------|------|----------|
| 처리 시간 | 33ms 이하 | chrono 측정 |
| FPS | 30fps 이상 | 프레임 카운터 |
| 메모리 | 100MB 이하 | Valgrind/Instruments |
| 안정성 | 1000+ 프레임 | 연속 처리 테스트 |

---

## 3. 실행 내역

### 3.1 통합 테스트 작성

```bash
# 예정: cpp/tests/test_integration.cpp
```

### 3.2 데모 애플리케이션 작성

```bash
# 예정: cpp/examples/camera_demo.cpp
# 예정: cpp/examples/image_demo.cpp
```

### 3.3 테스트 실행

```bash
# 예정
cd build
ctest --output-on-failure -R Integration
./bin/camera_demo
```

### 3.4 성능 측정

```bash
# 예정: FPS, 메모리, 처리 시간 측정
```

---

## 4. 검증 결과

### 검증 항목

| 항목 | 결과 | 비고 |
|------|------|------|
| 파이프라인 통합 | ⏳ 대기 | |
| C API 흐름 | ⏳ 대기 | |
| 포맷 변환 | ⏳ 대기 | BGR, RGBA, Gray |
| 연속 처리 | ⏳ 대기 | 1000프레임 |
| 웹캠 30fps | ⏳ 대기 | |
| 메모리 안정성 | ⏳ 대기 | |

### 성능 측정 결과

| 항목 | 측정값 | 목표 | 결과 |
|------|--------|------|------|
| 평균 FPS | - | 30+ | ⏳ |
| 처리 시간 | - | 33ms | ⏳ |
| 메모리 사용 | - | 100MB | ⏳ |
| 검출 성공률 | - | 95%+ | ⏳ |

---

## 5. 이슈 및 학습

### 이슈

| ID | 내용 | 상태 | 해결방안 |
|----|------|------|----------|
| - | - | - | - |

### 결정 사항

| 결정 | 이유 |
|------|------|
| 1280x720 기본 해상도 | 성능/품질 균형 |
| RGBA 작업 포맷 | 알파 채널 지원, 모바일 호환 |
| 1000프레임 안정성 테스트 | 실제 사용 시나리오 반영 |

### 학습 내용

(실행 후 기록)

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-01-07 | 태스크 문서 생성, 통합 테스트 설계 완료 |
