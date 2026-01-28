/**
 * @file render_context.h
 * @brief 렌더링 컨텍스트 추상화 인터페이스
 *
 * OpenGL ES, Metal, CPU 백엔드를 통합하는 플랫폼 독립적 인터페이스.
 * DI(Dependency Injection) 패턴으로 테스트 용이성 확보.
 */

#ifndef IRIS_SDK_RENDER_CONTEXT_H
#define IRIS_SDK_RENDER_CONTEXT_H

#include "texture_handle.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace iris_sdk {

/**
 * @brief 렌더링 컨텍스트 추상화 인터페이스
 *
 * 모든 렌더링 백엔드(OpenGL ES, Metal, CPU)가 구현해야 하는 인터페이스.
 * 뷰티 필터, 렌즈 렌더링 등 모든 GPU 작업의 기반이 됨.
 */
class IRenderContext {
public:
    virtual ~IRenderContext() = default;

    //=========================================================================
    // 라이프사이클 관리
    //=========================================================================

    /**
     * @brief 컨텍스트 초기화
     * @return 성공 시 true
     */
    virtual bool initialize() = 0;

    /**
     * @brief 컨텍스트 해제
     *
     * 모든 GPU 리소스를 해제하고 컨텍스트를 무효화합니다.
     */
    virtual void release() = 0;

    /**
     * @brief 초기화 상태 확인
     * @return 초기화되었으면 true
     */
    virtual bool isInitialized() const = 0;

    //=========================================================================
    // 텍스처 관리
    //=========================================================================

    /**
     * @brief 텍스처 생성
     * @param width 너비 (픽셀)
     * @param height 높이 (픽셀)
     * @param format 픽셀 포맷 (기본: RGBA8)
     * @return 생성된 텍스처 핸들 (실패 시 isValid() == false)
     */
    virtual TextureHandle createTexture(int width, int height,
                                         TextureFormat format = TextureFormat::RGBA8) = 0;

    /**
     * @brief 텍스처 삭제
     * @param handle 삭제할 텍스처 (호출 후 무효화됨)
     */
    virtual void deleteTexture(TextureHandle& handle) = 0;

    /**
     * @brief CPU 데이터 → GPU 텍스처 업로드
     * @param handle 대상 텍스처
     * @param data 픽셀 데이터 포인터
     * @param width 데이터 너비
     * @param height 데이터 높이
     * @param format 데이터 포맷
     * @return 성공 시 true
     */
    virtual bool uploadTexture(TextureHandle& handle, const uint8_t* data,
                                int width, int height, TextureFormat format) = 0;

    /**
     * @brief GPU 텍스처 → CPU 데이터 다운로드
     * @param handle 소스 텍스처
     * @param data 출력 버퍼
     * @param max_size 버퍼 최대 크기 (바이트)
     * @return 성공 시 true
     */
    virtual bool downloadTexture(const TextureHandle& handle, uint8_t* data,
                                  size_t max_size) = 0;

    //=========================================================================
    // 컨텍스트 전환
    //=========================================================================

    /**
     * @brief 현재 스레드에 컨텍스트 바인딩
     * @return 성공 시 true
     */
    virtual bool makeCurrent() = 0;

    /**
     * @brief 컨텍스트 언바인딩
     */
    virtual void doneCurrent() = 0;

    //=========================================================================
    // Android 라이프사이클 (Context Loss 대응)
    //=========================================================================

    /**
     * @brief Surface 생성 시 호출 (Activity onResume 등)
     *
     * GL Context가 재생성된 후 리소스를 복구합니다.
     * 기본 구현은 no-op (CPU 백엔드용).
     */
    virtual void onSurfaceCreated() {}

    /**
     * @brief Surface 파괴 시 호출 (Activity onPause/onStop 등)
     *
     * GL Context 손실 전에 리소스를 정리합니다.
     * 이 시점 이후 모든 TextureHandle은 무효화됩니다.
     * 기본 구현은 no-op (CPU 백엔드용).
     */
    virtual void onSurfaceDestroyed() {}

    /**
     * @brief Context Loss 상태 확인
     * @return true면 모든 GPU 리소스가 무효화된 상태
     *
     * Context Loss 상태에서는 새로운 텍스처 생성이 실패합니다.
     * onSurfaceCreated() 호출로 복구될 수 있습니다.
     */
    virtual bool isContextLost() const { return false; }

    //=========================================================================
    // 메타데이터
    //=========================================================================

    /**
     * @brief 컨텍스트 이름 (예: "GLESRenderContext", "CPURenderContext")
     */
    virtual const char* getName() const = 0;

    /**
     * @brief 메이저 버전
     */
    virtual int getMajorVersion() const = 0;

    /**
     * @brief 마이너 버전
     */
    virtual int getMinorVersion() const = 0;

    /**
     * @brief GPU 지원 여부
     * @return true면 하드웨어 가속 지원
     */
    virtual bool supportsGpu() const = 0;

    /**
     * @brief 컨텍스트 정보 문자열 (디버깅용)
     */
    virtual std::string getInfo() const {
        return std::string(getName()) + " v" +
               std::to_string(getMajorVersion()) + "." +
               std::to_string(getMinorVersion());
    }

    //=========================================================================
    // 디버깅 도구
    //=========================================================================

    /**
     * @brief 텍스처를 파일로 덤프 (디버깅용)
     * @param handle 덤프할 텍스처
     * @param file_path 출력 파일 경로 (PNG 권장)
     * @return 성공 시 true
     *
     * 개발 빌드에서만 활성화됩니다.
     */
    virtual bool dumpTexture(const TextureHandle& handle,
                              const std::string& file_path) {
        (void)handle;
        (void)file_path;
        return false;  // 기본 구현은 미지원
    }

    //=========================================================================
    // 팩토리 메서드
    //=========================================================================

    /**
     * @brief 플랫폼에 적합한 RenderContext 생성
     * @param prefer_gpu GPU 선호 여부 (false면 CPU 폴백)
     * @return 생성된 컨텍스트 (소유권 이전)
     *
     * Android: GPU 선호 시 GLESRenderContext, 실패 시 CPURenderContext
     * Desktop: CPURenderContext (OpenGL ES 미지원)
     * iOS: 향후 MetalRenderContext 지원 예정
     */
    static std::unique_ptr<IRenderContext> create(bool prefer_gpu = true);
};

} // namespace iris_sdk

#endif // IRIS_SDK_RENDER_CONTEXT_H
