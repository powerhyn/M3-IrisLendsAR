/**
 * @file shader_manager.h
 * @brief OpenGL ES 셰이더 컴파일 및 캐싱 관리
 *
 * 셰이더 프로그램 컴파일, 링크, 캐싱을 담당합니다.
 * Android OpenGL ES 환경에서 동작하며, Desktop에서는 스텁으로 컴파일됩니다.
 */

#ifndef IRIS_SDK_SHADER_MANAGER_H
#define IRIS_SDK_SHADER_MANAGER_H

#include <mutex>
#include <string>
#include <unordered_map>

// 조건부 컴파일: Android GLES 또는 강제 GLES 모드
#if defined(__ANDROID__) || defined(IRIS_SDK_FORCE_GLES)
#define IRIS_SDK_GPU_AVAILABLE 1
#include <GLES3/gl31.h>
#else
#define IRIS_SDK_GPU_AVAILABLE 0
// Desktop 스텁을 위한 타입 정의
using GLuint = unsigned int;
using GLint = int;
using GLenum = unsigned int;
#define GL_VERTEX_SHADER 0x8B31
#define GL_FRAGMENT_SHADER 0x8B30
#endif

namespace iris_sdk {

/**
 * @brief 셰이더 컴파일 및 캐싱 관리자
 *
 * OpenGL ES 셰이더 프로그램을 컴파일하고 캐싱하여
 * 런타임 성능을 최적화합니다.
 *
 * Thread-safe: 모든 public 메서드는 mutex로 보호됩니다.
 */
class ShaderManager {
public:
    ShaderManager();
    ~ShaderManager();

    // 복사/이동 금지
    ShaderManager(const ShaderManager&) = delete;
    ShaderManager& operator=(const ShaderManager&) = delete;
    ShaderManager(ShaderManager&&) = delete;
    ShaderManager& operator=(ShaderManager&&) = delete;

    //=========================================================================
    // 프로그램 생성/관리
    //=========================================================================

    /**
     * @brief 셰이더 프로그램 생성
     *
     * 버텍스/프래그먼트 셰이더를 컴파일하고 링크하여
     * 사용 가능한 프로그램을 생성합니다.
     *
     * @param vertex_source 버텍스 셰이더 GLSL 소스
     * @param fragment_source 프래그먼트 셰이더 GLSL 소스
     * @param out_program 생성된 프로그램 ID (출력)
     * @return 성공 시 true
     */
    bool createProgram(const char* vertex_source,
                       const char* fragment_source,
                       GLuint& out_program);

    /**
     * @brief 캐시된 프로그램 조회
     *
     * @param name 프로그램 이름
     * @return 프로그램 ID (없으면 0)
     */
    GLuint getProgram(const std::string& name) const;

    /**
     * @brief 프로그램 캐시 등록
     *
     * @param name 프로그램 이름
     * @param program 프로그램 ID
     */
    void cacheProgram(const std::string& name, GLuint program);

    /**
     * @brief 모든 셰이더/프로그램 해제
     *
     * 캐시된 모든 프로그램을 삭제하고 GL 리소스를 해제합니다.
     */
    void releaseAll();

    /**
     * @brief 캐시된 프로그램 수 조회
     */
    size_t getCachedProgramCount() const;

    //=========================================================================
    // 유틸리티 (static)
    //=========================================================================

    /**
     * @brief 개별 셰이더 컴파일
     *
     * @param type 셰이더 타입 (GL_VERTEX_SHADER 또는 GL_FRAGMENT_SHADER)
     * @param source GLSL 소스 코드
     * @param out_shader 컴파일된 셰이더 ID (출력)
     * @return 성공 시 true
     */
    static bool compileShader(GLenum type, const char* source, GLuint& out_shader);

    /**
     * @brief 셰이더 프로그램 링크
     *
     * @param vertex_shader 버텍스 셰이더 ID
     * @param fragment_shader 프래그먼트 셰이더 ID
     * @param out_program 링크된 프로그램 ID (출력)
     * @return 성공 시 true
     */
    static bool linkProgram(GLuint vertex_shader,
                            GLuint fragment_shader,
                            GLuint& out_program);

    /**
     * @brief 셰이더 컴파일 로그 조회
     *
     * @param shader 셰이더 ID
     * @return 컴파일 로그 문자열
     */
    static std::string getShaderLog(GLuint shader);

    /**
     * @brief 프로그램 링크 로그 조회
     *
     * @param program 프로그램 ID
     * @return 링크 로그 문자열
     */
    static std::string getProgramLog(GLuint program);

    /**
     * @brief OpenGL 에러 체크
     *
     * @param operation 현재 작업 이름 (로깅용)
     * @return 에러 없으면 true
     */
    static bool checkGLError(const char* operation);

private:
    std::unordered_map<std::string, GLuint> program_cache_;
    mutable std::mutex mutex_;
};

//=============================================================================
// 내장 셰이더 소스 (shader_sources.cpp에서 정의)
//=============================================================================
namespace shaders {

/// 풀스크린 쿼드 버텍스 셰이더
extern const char* FULLSCREEN_QUAD_VERTEX;

/// 패스스루 프래그먼트 셰이더 (텍스처 그대로 출력)
extern const char* PASSTHROUGH_FRAGMENT;

/// 밝기 조정 프래그먼트 셰이더
extern const char* BRIGHTNESS_FRAGMENT;

/// Bilateral 필터 프래그먼트 셰이더 (피부 스무딩)
extern const char* BILATERAL_FILTER_FRAGMENT;

/// 화이트닝 프래그먼트 셰이더
extern const char* WHITENING_FRAGMENT;

/// 컬러 밸런스 프래그먼트 셰이더
extern const char* COLOR_BALANCE_FRAGMENT;

/// 소프트 포커스 프래그먼트 셰이더
extern const char* SOFT_FOCUS_FRAGMENT;

/// 마스킹(ROI 블렌딩) 프래그먼트 셰이더
extern const char* MASKING_FRAGMENT;

/// Gaussian Blur 프래그먼트 셰이더
extern const char* GAUSSIAN_BLUR_FRAGMENT;

/// 통합 Color Adjustment 셰이더 (Brightness + ColorBalance + Whitening)
extern const char* COMBINED_COLOR_ADJUSTMENT_FRAGMENT;

/// Frequency Separation Gaussian Blur 셰이더
extern const char* FREQ_SEP_GAUSSIAN_FRAGMENT;

/// Frequency Separation Composite 셰이더
extern const char* FREQ_SEP_COMPOSITE_FRAGMENT;

/// Luminance Sharpen 셰이더 (FreqSep 후 선명도 복구)
extern const char* LUMINANCE_SHARPEN_FRAGMENT;

} // namespace shaders

} // namespace iris_sdk

#endif // IRIS_SDK_SHADER_MANAGER_H
