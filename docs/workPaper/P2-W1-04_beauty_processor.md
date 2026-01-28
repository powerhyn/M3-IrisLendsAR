# P2-W1-04. IBeautyBackend 인터페이스 및 BeautyProcessor

## 작업 정보

| 항목 | 내용 |
|------|------|
| **작업 ID** | P2-W1-04 |
| **Phase** | Phase 1: 기반 구조 리팩토링 |
| **상태** | ✅ 완료 |
| **예상 기간** | 2일 |
| **의존성** | P2-W1-01, P2-W1-02, P2-W1-03 |
| **담당** | systems-programming:cpp-pro |

---

## 1. 목표

뷰티 필터 처리를 위한 백엔드 전략 인터페이스 및 통합 프로세서 구현

### 핵심 산출물
- `IBeautyBackend` 인터페이스
- `BeautyProcessor` 통합 클래스 (DI 기반)
- 기존 `BeautyFilter`를 `CPUBeautyBackend`로 리팩토링

---

## 2. 상세 작업

### 2.1 IBeautyBackend 인터페이스

**파일**: `cpp/include/iris_sdk/beauty_backend.h`

```cpp
#ifndef IRIS_SDK_BEAUTY_BACKEND_H
#define IRIS_SDK_BEAUTY_BACKEND_H

#include "types.h"
#include "beauty_filter.h"
#include "beauty_roi_manager.h"
#include "gpu/texture_handle.h"
#include <memory>

namespace iris_sdk {

class IRenderContext;

/**
 * @brief 뷰티 필터 백엔드 전략 인터페이스
 *
 * CPU (OpenCV) 또는 GPU (OpenGL ES/Metal) 구현을 추상화
 */
class IBeautyBackend {
public:
    virtual ~IBeautyBackend() = default;

    //=== 라이프사이클 ===
    virtual bool initialize(IRenderContext* render_context = nullptr) = 0;
    virtual void release() = 0;
    virtual bool isInitialized() const = 0;

    //=== 필터 적용 (CPU 버퍼) ===
    /**
     * @brief 필터 적용 (in-place)
     *
     * @param frame_data 프레임 데이터 (입출력)
     * @param width 프레임 너비
     * @param height 프레임 높이
     * @param format 픽셀 포맷
     * @param config 필터 설정
     * @param roi ROI 정보 (nullable, null이면 전체 프레임)
     * @return 에러 코드
     */
    virtual IrisSdkError apply(
        uint8_t* frame_data,
        int width, int height,
        IrisFrameFormat format,
        const BeautyFilterConfigV2& config,
        const BeautyROI* roi = nullptr
    ) = 0;

    //=== 필터 적용 (텍스처) ===
    /**
     * @brief 텍스처 기반 필터 적용 (GPU용)
     *
     * @param input 입력 텍스처
     * @param output 출력 텍스처
     * @param config 필터 설정
     * @param roi ROI 정보
     * @return 에러 코드
     */
    virtual IrisSdkError applyTexture(
        const TextureHandle& input,
        TextureHandle& output,
        const BeautyFilterConfigV2& config,
        const BeautyROI* roi = nullptr
    ) {
        // 기본 구현: 지원하지 않음
        return IRIS_SDK_ERROR_NOT_SUPPORTED;
    }

    //=== 메타데이터 ===
    virtual const char* getName() const = 0;
    virtual bool supportsGpu() const = 0;
    virtual bool supportsTextureProcessing() const { return false; }
};

} // namespace iris_sdk

#endif // IRIS_SDK_BEAUTY_BACKEND_H
```

### 2.2 BeautyProcessor 클래스 (DI 기반)

**파일**: `cpp/include/iris_sdk/beauty_processor.h`

```cpp
#ifndef IRIS_SDK_BEAUTY_PROCESSOR_H
#define IRIS_SDK_BEAUTY_PROCESSOR_H

#include "beauty_backend.h"
#include "beauty_roi_manager.h"
#include "gpu/render_context.h"
#include <memory>
#include <mutex>

namespace iris_sdk {

/**
 * @brief 뷰티 필터 통합 프로세서
 *
 * - DI(Dependency Injection) 기반 설계
 * - CPU/GPU 백엔드 자동 선택
 * - ROI 캐싱 지원
 * - LensRenderer와 RenderContext 공유 가능
 */
class BeautyProcessor {
public:
    /**
     * @brief 생성자 (DI)
     *
     * @param render_context 렌더링 컨텍스트 (공유)
     */
    explicit BeautyProcessor(std::shared_ptr<IRenderContext> render_context = nullptr);

    ~BeautyProcessor();

    // 복사 금지
    BeautyProcessor(const BeautyProcessor&) = delete;
    BeautyProcessor& operator=(const BeautyProcessor&) = delete;

    //=== 초기화 ===
    /**
     * @brief 초기화
     *
     * @param prefer_gpu GPU 선호 여부
     * @return 성공 여부
     */
    bool initialize(bool prefer_gpu = true);

    void release();
    bool isInitialized() const;

    //=== 설정 ===
    IrisSdkError setConfig(const BeautyFilterConfigV2& config);
    IrisSdkError getConfig(BeautyFilterConfigV2& out_config) const;

    //=== 필터 적용 (CPU 버퍼) ===
    /**
     * @brief 프레임에 필터 적용
     *
     * Face Mesh가 제공되면 ROI 기반 처리
     */
    IrisSdkError process(
        uint8_t* frame_data,
        int width, int height,
        IrisFrameFormat format,
        const IrisResult* iris_result = nullptr  // Face Mesh 포함
    );

    //=== 필터 적용 (텍스처) ===
    /**
     * @brief 텍스처 기반 필터 적용 (Zero-Copy)
     *
     * LensRenderer와 RenderContext 공유 시 사용
     */
    IrisSdkError processTexture(
        const TextureHandle& input,
        TextureHandle& output,
        const IrisResult* iris_result = nullptr
    );

    //=== 상태 조회 ===
    bool isUsingGpu() const;
    bool isEnabled() const;

    //=== RenderContext 공유 ===
    std::shared_ptr<IRenderContext> getRenderContext() const { return render_context_; }

private:
    // ROI 계산 (캐싱 포함)
    bool computeROI(const IrisResult* iris_result, int width, int height,
                    BeautyROI& out_roi);

    // 백엔드 선택
    bool selectBackend(bool prefer_gpu);

    std::shared_ptr<IRenderContext> render_context_;
    std::unique_ptr<IBeautyBackend> backend_;
    BeautyFilterConfigV2 config_;

    // ROI 캐시
    BeautyROI cached_roi_;
    int64_t cached_roi_timestamp_ = 0;
    static constexpr int64_t ROI_CACHE_TIMEOUT_MS = 100;

    mutable std::mutex mutex_;
    bool initialized_ = false;
};

} // namespace iris_sdk

#endif // IRIS_SDK_BEAUTY_PROCESSOR_H
```

### 2.3 CPUBeautyBackend 구현 (기존 BeautyFilter 리팩토링)

**파일**: `cpp/include/iris_sdk/cpu_beauty_backend.h`

```cpp
#ifndef IRIS_SDK_CPU_BEAUTY_BACKEND_H
#define IRIS_SDK_CPU_BEAUTY_BACKEND_H

#include "beauty_backend.h"
#include <opencv2/core.hpp>

namespace iris_sdk {

/**
 * @brief CPU 기반 뷰티 필터 백엔드 (OpenCV)
 *
 * 기존 BeautyFilter 기능을 IBeautyBackend 인터페이스로 래핑
 */
class CPUBeautyBackend : public IBeautyBackend {
public:
    CPUBeautyBackend();
    ~CPUBeautyBackend() override;

    //=== IBeautyBackend 구현 ===
    bool initialize(IRenderContext* render_context = nullptr) override;
    void release() override;
    bool isInitialized() const override;

    IrisSdkError apply(
        uint8_t* frame_data,
        int width, int height,
        IrisFrameFormat format,
        const BeautyFilterConfigV2& config,
        const BeautyROI* roi = nullptr
    ) override;

    const char* getName() const override { return "CPUBeautyBackend"; }
    bool supportsGpu() const override { return false; }

private:
    // 기존 BeautyFilter 로직
    void applySkinSmoothing(cv::Mat& frame, float strength);
    void applySoftFocus(cv::Mat& frame, float strength);
    void applyBrightness(cv::Mat& frame, float brightness);
    void applyWhitening(cv::Mat& frame, float strength);

    // ROI 기반 처리
    IrisSdkError applyWithROI(
        cv::Mat& frame,
        const BeautyFilterConfigV2& config,
        const BeautyROI& roi
    );

    // 포맷 변환 헬퍼
    static cv::Mat convertToBGR(const uint8_t* data, int width, int height,
                                IrisFrameFormat format);
    static void convertFromBGR(const cv::Mat& bgr, uint8_t* data,
                               IrisFrameFormat format);

    // 작업 버퍼
    cv::Mat work_buffer_;
    cv::Mat smooth_buffer_;
    std::mutex buffer_mutex_;
    bool initialized_ = false;
};

} // namespace iris_sdk

#endif // IRIS_SDK_CPU_BEAUTY_BACKEND_H
```

### 2.4 BeautyProcessor 구현

**파일**: `cpp/src/beauty_processor.cpp`

```cpp
#include "iris_sdk/beauty_processor.h"
#include "iris_sdk/cpu_beauty_backend.h"

#if defined(__ANDROID__) && defined(IRIS_SDK_HAS_GLES)
#include "iris_sdk/gpu/gpu_beauty_backend.h"
#endif

namespace iris_sdk {

BeautyProcessor::BeautyProcessor(std::shared_ptr<IRenderContext> render_context)
    : render_context_(std::move(render_context)) {
}

BeautyProcessor::~BeautyProcessor() {
    release();
}

bool BeautyProcessor::initialize(bool prefer_gpu) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (initialized_) return true;

    // RenderContext가 없으면 생성
    if (!render_context_) {
        render_context_ = IRenderContext::create(prefer_gpu);
    }

    // 백엔드 선택
    if (!selectBackend(prefer_gpu)) {
        return false;
    }

    initialized_ = true;
    return true;
}

bool BeautyProcessor::selectBackend(bool prefer_gpu) {
#if defined(__ANDROID__) && defined(IRIS_SDK_HAS_GLES)
    if (prefer_gpu && render_context_ && render_context_->supportsGpu()) {
        auto gpu_backend = std::make_unique<GPUBeautyBackend>();
        if (gpu_backend->initialize(render_context_.get())) {
            backend_ = std::move(gpu_backend);
            return true;
        }
        // GPU 실패 → CPU 폴백
    }
#endif

    // CPU 백엔드
    backend_ = std::make_unique<CPUBeautyBackend>();
    return backend_->initialize(nullptr);
}

IrisSdkError BeautyProcessor::process(
    uint8_t* frame_data,
    int width, int height,
    IrisFrameFormat format,
    const IrisResult* iris_result) {

    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_ || !backend_) {
        return IRIS_SDK_ERROR_NOT_INITIALIZED;
    }

    if (!config_.enabled) {
        return IRIS_SDK_OK;  // 비활성화 상태
    }

    // ROI 계산
    BeautyROI* roi_ptr = nullptr;
    BeautyROI computed_roi;

    if (config_.roiOnly && iris_result && iris_result->face_mesh_valid) {
        if (computeROI(iris_result, width, height, computed_roi)) {
            roi_ptr = &computed_roi;
        }
    }

    // 백엔드에 처리 위임
    return backend_->apply(frame_data, width, height, format, config_, roi_ptr);
}

bool BeautyProcessor::computeROI(
    const IrisResult* iris_result,
    int width, int height,
    BeautyROI& out_roi) {

    // 캐시 확인 (100ms 이내)
    int64_t now = getCurrentTimeMs();
    if (cached_roi_.valid && (now - cached_roi_timestamp_) < ROI_CACHE_TIMEOUT_MS) {
        out_roi = cached_roi_;
        return true;
    }

    // 새로 계산
    if (BeautyROIManager::computeROI(
            iris_result->face_mesh, width, height, config_, out_roi)) {
        cached_roi_ = out_roi;
        cached_roi_timestamp_ = now;
        return true;
    }

    return false;
}

} // namespace iris_sdk
```

---

## 3. 단위 테스트

**파일**: `cpp/tests/test_beauty_processor.cpp`

```cpp
TEST(BeautyProcessor, InitializesWithCPUBackend) {
    auto processor = std::make_unique<BeautyProcessor>();
    ASSERT_TRUE(processor->initialize(false));  // CPU만
    EXPECT_FALSE(processor->isUsingGpu());
    EXPECT_TRUE(processor->isInitialized());
}

TEST(BeautyProcessor, DI_AcceptsExternalRenderContext) {
    auto render_ctx = std::make_shared<CPURenderContext>();
    render_ctx->initialize();

    auto processor = std::make_unique<BeautyProcessor>(render_ctx);
    ASSERT_TRUE(processor->initialize(false));

    // 동일한 RenderContext 공유 확인
    EXPECT_EQ(processor->getRenderContext().get(), render_ctx.get());
}

TEST(BeautyProcessor, ProcessWithROI_AppliesFilterToFaceOnly) {
    auto processor = std::make_unique<BeautyProcessor>();
    processor->initialize(false);

    BeautyFilterConfigV2 config;
    config.enabled = true;
    config.smoothing = 0.8f;
    config.roiOnly = true;
    processor->setConfig(config);

    // 테스트 프레임 및 Face Mesh 로드
    std::vector<uint8_t> frame = loadTestFrame("test_portrait.jpg");
    IrisResult iris_result = loadTestIrisResult("test_face_mesh.bin");

    IrisSdkError err = processor->process(
        frame.data(), 640, 480, IRIS_FORMAT_RGB, &iris_result);

    EXPECT_EQ(err, IRIS_SDK_OK);
}

TEST(BeautyProcessor, DisabledConfig_SkipsProcessing) {
    auto processor = std::make_unique<BeautyProcessor>();
    processor->initialize(false);

    BeautyFilterConfigV2 config;
    config.enabled = false;  // 비활성화
    processor->setConfig(config);

    std::vector<uint8_t> original(640 * 480 * 3, 128);
    std::vector<uint8_t> frame = original;

    processor->process(frame.data(), 640, 480, IRIS_FORMAT_RGB, nullptr);

    // 프레임 변경 없음
    EXPECT_EQ(frame, original);
}
```

---

## 4. 완료 기준

- [x] `IBeautyBackend` 인터페이스 정의
- [x] `BeautyProcessor` 클래스 구현 (DI 기반)
- [x] `CPUBeautyBackend` 구현 (기존 로직 래핑)
- [x] ROI 캐싱 로직
- [x] 백엔드 자동 선택 (GPU 선호)
- [x] 단위 테스트 100% 통과 (26개)

---

## 5. 실행 내역

### 2025-01-28: 구현 완료

**생성된 파일**:
- `cpp/include/iris_sdk/beauty_backend.h` - IBeautyBackend 인터페이스
- `cpp/include/iris_sdk/cpu_beauty_backend.h` - CPUBeautyBackend 헤더
- `cpp/include/iris_sdk/beauty_processor.h` - BeautyProcessor 헤더
- `cpp/src/cpu_beauty_backend.cpp` - CPUBeautyBackend 구현
- `cpp/src/beauty_processor.cpp` - BeautyProcessor 구현
- `cpp/tests/test_beauty_processor.cpp` - 단위 테스트 (26개)

**수정된 파일**:
- `cpp/include/iris_sdk/sdk_api.h` - 에러 코드 추가 (IRIS_SDK_ERROR_NOT_SUPPORTED, IRIS_SDK_ERROR_NOT_INITIALIZED)
- `cpp/CMakeLists.txt` - 소스 파일 추가
- `cpp/tests/CMakeLists.txt` - 테스트 타겟 추가

**주요 구현 사항**:
- DI(Dependency Injection) 기반 BeautyProcessor 설계
- RenderContext 공유 지원 (LensRenderer와 연동 가능)
- ROI 캐싱 (100ms 타임아웃)
- GPU 백엔드 폴백 로직 (Android GLES)
- 5가지 필터 효과: SkinSmoothing, SoftFocus, Brightness, Whitening, ColorBalance

**테스트 결과**:
```
[==========] Running 26 tests from 3 test suites.
[  PASSED  ] 26 tests.
```

---

## 6. 다음 작업

- **P2-W2-01**: ROI 기반 처리 및 페더링 통합
