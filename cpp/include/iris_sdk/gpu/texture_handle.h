/**
 * @file texture_handle.h
 * @brief 플랫폼 독립적 텍스처 핸들 추상화
 *
 * GLuint (OpenGL ES), MTLTexture* (Metal), cv::Mat* (CPU) 등을
 * 통합 관리하기 위한 추상화 구조체.
 */

#ifndef IRIS_SDK_TEXTURE_HANDLE_H
#define IRIS_SDK_TEXTURE_HANDLE_H

#include <cstddef>
#include <cstdint>

namespace iris_sdk {

/**
 * @brief 텍스처 픽셀 포맷
 */
enum class TextureFormat : int {
    Unknown = 0,
    RGBA8 = 1,      ///< 32-bit RGBA
    RGB8 = 2,       ///< 24-bit RGB
    R8 = 3,         ///< 8-bit grayscale
    RGBA16F = 4,    ///< 64-bit floating point RGBA (HDR)
};

/**
 * @brief 플랫폼 독립적 텍스처 핸들
 *
 * GLuint (OpenGL ES), MTLTexture* (Metal), cv::Mat* (CPU) 등을
 * 통합 관리하기 위한 추상화 구조체.
 */
struct TextureHandle {
    /**
     * @brief 텍스처 타입 (백엔드 구분)
     */
    enum class Type : int {
        Invalid = 0,
        OpenGLES,   ///< Android OpenGL ES
        Metal,      ///< iOS Metal (향후)
        CPU         ///< CPU Fallback (cv::Mat)
    };

    void* native_handle = nullptr;  ///< 플랫폼별 핸들 (GLuint*, MTLTexture*, cv::Mat*)
    Type type = Type::Invalid;      ///< 텍스처 타입
    int width = 0;                  ///< 텍스처 너비
    int height = 0;                 ///< 텍스처 높이
    TextureFormat format = TextureFormat::Unknown;  ///< 픽셀 포맷
    uint64_t id = 0;                ///< 고유 식별자 (디버깅/추적용)

    /**
     * @brief 유효성 검사
     * @return 유효한 핸들이면 true
     */
    bool isValid() const {
        return native_handle != nullptr && type != Type::Invalid;
    }

    /**
     * @brief 핸들 무효화
     */
    void invalidate() {
        native_handle = nullptr;
        type = Type::Invalid;
        width = 0;
        height = 0;
        format = TextureFormat::Unknown;
        id = 0;
    }

    /**
     * @brief 픽셀당 바이트 수 계산
     */
    int bytesPerPixel() const {
        switch (format) {
            case TextureFormat::RGBA8:   return 4;
            case TextureFormat::RGB8:    return 3;
            case TextureFormat::R8:      return 1;
            case TextureFormat::RGBA16F: return 8;
            default:                     return 0;
        }
    }

    /**
     * @brief 총 메모리 크기 계산 (바이트)
     */
    size_t memorySize() const {
        return static_cast<size_t>(width) * height * bytesPerPixel();
    }
};

} // namespace iris_sdk

#endif // IRIS_SDK_TEXTURE_HANDLE_H
