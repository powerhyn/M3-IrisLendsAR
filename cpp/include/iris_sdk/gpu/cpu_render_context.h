/**
 * @file cpu_render_context.h
 * @brief CPU 기반 렌더링 컨텍스트 (Fallback)
 *
 * GPU 미지원 환경이나 Desktop 빌드에서 사용.
 * cv::Mat를 TextureHandle로 래핑하여 동일한 인터페이스 제공.
 */

#ifndef IRIS_SDK_CPU_RENDER_CONTEXT_H
#define IRIS_SDK_CPU_RENDER_CONTEXT_H

#include "render_context.h"

#include <opencv2/core.hpp>

#include <map>
#include <memory>
#include <mutex>

namespace iris_sdk {

/**
 * @brief CPU 기반 렌더링 컨텍스트 (GPU 미지원 환경용)
 *
 * cv::Mat를 TextureHandle로 래핑하여 IRenderContext 인터페이스 제공.
 * Desktop 테스트 및 GPU 미지원 기기의 폴백으로 사용.
 */
class CPURenderContext : public IRenderContext {
public:
    CPURenderContext();
    ~CPURenderContext() override;

    // 복사/이동 금지
    CPURenderContext(const CPURenderContext&) = delete;
    CPURenderContext& operator=(const CPURenderContext&) = delete;
    CPURenderContext(CPURenderContext&&) = delete;
    CPURenderContext& operator=(CPURenderContext&&) = delete;

    //=========================================================================
    // IRenderContext 구현
    //=========================================================================

    bool initialize() override;
    void release() override;
    bool isInitialized() const override;

    TextureHandle createTexture(int width, int height,
                                 TextureFormat format = TextureFormat::RGBA8) override;
    void deleteTexture(TextureHandle& handle) override;

    bool uploadTexture(TextureHandle& handle, const uint8_t* data,
                        int width, int height, TextureFormat format) override;
    bool downloadTexture(const TextureHandle& handle, uint8_t* data,
                          size_t max_size) override;

    // CPU는 컨텍스트 전환 불필요
    bool makeCurrent() override { return true; }
    void doneCurrent() override {}

    const char* getName() const override { return "CPURenderContext"; }
    int getMajorVersion() const override { return 1; }
    int getMinorVersion() const override { return 0; }
    bool supportsGpu() const override { return false; }

    //=========================================================================
    // CPU 전용 메서드
    //=========================================================================

    /**
     * @brief TextureHandle에서 cv::Mat 포인터 추출
     * @param handle 대상 핸들
     * @return cv::Mat 포인터 (무효 시 nullptr)
     */
    cv::Mat* getCvMat(const TextureHandle& handle) const;

    /**
     * @brief TextureHandle에서 const cv::Mat 참조 획득
     * @param handle 대상 핸들
     * @return cv::Mat const 참조 (무효 시 빈 Mat 반환)
     */
    const cv::Mat& getCvMatRef(const TextureHandle& handle) const;

    //=========================================================================
    // 디버깅 도구
    //=========================================================================

    bool dumpTexture(const TextureHandle& handle,
                      const std::string& file_path) override;

    /**
     * @brief 할당된 텍스처 수 조회
     */
    size_t getTextureCount() const;

    /**
     * @brief 할당된 텍스처 총 메모리 (바이트)
     */
    size_t getTextureMemoryUsage() const;

private:
    /**
     * @brief TextureFormat → OpenCV 타입 변환
     */
    int toCvType(TextureFormat format) const;

    /**
     * @brief OpenCV 타입 → TextureFormat 변환
     */
    TextureFormat fromCvType(int cv_type) const;

    //=========================================================================
    // 멤버 변수
    //=========================================================================

    // 텍스처 저장소 (ID → cv::Mat)
    struct TextureData {
        std::unique_ptr<cv::Mat> mat;
        TextureFormat format;
    };
    std::map<uint64_t, TextureData> textures_;

    // 다음 텍스처 ID
    uint64_t next_texture_id_ = 1;

    // 상태
    bool initialized_ = false;

    // 빈 Mat (무효 핸들 참조 시 반환용)
    cv::Mat empty_mat_;

    // 동기화
    mutable std::mutex mutex_;
};

} // namespace iris_sdk

#endif // IRIS_SDK_CPU_RENDER_CONTEXT_H
