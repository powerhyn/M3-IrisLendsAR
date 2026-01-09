/**
 * @file lens_renderer.h
 * @brief 가상 렌즈 렌더러 선언
 *
 * 검출된 홍채 위치에 렌즈 텍스처를 오버레이하는 렌더러
 * Pimpl 패턴으로 OpenCV 의존성 분리
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "export.h"
#include "types.h"

// OpenCV 전방 선언 (헤더 의존성 분리)
namespace cv { class Mat; }

namespace iris_sdk {

/**
 * @brief 가상 렌즈 렌더러
 *
 * 검출된 홍채 위치에 렌즈 텍스처를 오버레이하는 클래스.
 * ROI 기반 처리와 알파 블렌딩을 통해 자연스러운 렌즈 효과 제공.
 *
 * @note OpenCV 의존성은 Pimpl로 분리되어 있음
 * @note 스레드 안전하지 않음 - 단일 스레드에서 사용 권장
 *
 * 사용 예시:
 * @code
 * LensRenderer renderer;
 * renderer.initialize();
 * renderer.loadTexture("lens.png");
 *
 * LensConfig config;
 * config.opacity = 0.8f;
 *
 * cv::Mat frame = cv::imread("face.jpg");
 * IrisResult result = detector.detect(...);
 * renderer.render(frame, result, config);
 * @endcode
 */
class IRIS_SDK_EXPORT LensRenderer {
public:
    LensRenderer();
    ~LensRenderer();

    // 복사 금지 (Pimpl 사용)
    LensRenderer(const LensRenderer&) = delete;
    LensRenderer& operator=(const LensRenderer&) = delete;

    /**
     * @brief 이동 생성자
     * @note 이동 후 원본 객체는 초기화되지 않은 상태가 됨.
     *       isInitialized()가 false를 반환하며, 다른 메서드 호출 시
     *       안전하게 false를 반환하거나 아무 작업도 수행하지 않음.
     */
    LensRenderer(LensRenderer&&) noexcept;

    /**
     * @brief 이동 대입 연산자
     * @note 이동 후 원본 객체는 초기화되지 않은 상태가 됨.
     *       isInitialized()가 false를 반환하며, 다른 메서드 호출 시
     *       안전하게 false를 반환하거나 아무 작업도 수행하지 않음.
     */
    LensRenderer& operator=(LensRenderer&&) noexcept;

    // ========================================
    // 초기화 및 해제
    // ========================================

    /**
     * @brief 렌더러 초기화
     *
     * 내부 버퍼 할당 및 초기 설정 수행.
     * 텍스처 로드 전에 반드시 호출해야 함.
     *
     * @return 초기화 성공 여부
     */
    bool initialize();

    /**
     * @brief 리소스 해제
     *
     * 모든 내부 리소스 해제. 텍스처 및 버퍼 포함.
     * 객체 소멸 시 자동 호출됨.
     */
    void release();

    /**
     * @brief 초기화 상태 확인
     * @return 초기화 완료 여부
     * @note 이동된 객체에서 호출해도 안전하게 false 반환
     */
    bool isInitialized() const noexcept;

    // ========================================
    // 텍스처 관리
    // ========================================

    /**
     * @brief 파일에서 렌즈 텍스처 로드
     *
     * PNG, JPEG 등 OpenCV가 지원하는 이미지 포맷 지원.
     * 알파 채널이 있는 PNG 권장.
     *
     * @param texture_path 텍스처 이미지 파일 경로
     * @return 로드 성공 여부
     */
    bool loadTexture(const std::string& texture_path);

    /**
     * @brief 메모리에서 렌즈 텍스처 로드
     *
     * RGBA 형식의 raw 픽셀 데이터에서 텍스처 로드.
     *
     * @param data RGBA 픽셀 데이터 (width * height * 4 바이트)
     * @param width 텍스처 너비
     * @param height 텍스처 높이
     * @return 로드 성공 여부
     */
    bool loadTexture(const uint8_t* data, int width, int height);

    /**
     * @brief 텍스처 언로드
     *
     * 로드된 텍스처 해제. 렌더러는 초기화 상태 유지.
     */
    void unloadTexture();

    /**
     * @brief 텍스처 로드 상태 확인
     * @return 텍스처 로드 완료 여부
     * @note 이동된 객체에서 호출해도 안전하게 false 반환
     */
    bool hasTexture() const noexcept;

    // ========================================
    // 렌더링
    // ========================================

    /**
     * @brief 프레임에 렌즈 렌더링 (양쪽 눈)
     *
     * LensConfig의 apply_left, apply_right 설정에 따라
     * 선택적으로 렌더링. 프레임을 in-place로 수정.
     *
     * @param frame 입출력 이미지 (BGR 또는 BGRA, in-place 수정)
     * @param iris_result 홍채 검출 결과
     * @param config 렌더링 설정
     * @return 렌더링 성공 여부 (검출 결과 없으면 false)
     */
    bool render(cv::Mat& frame,
                const IrisResult& iris_result,
                const LensConfig& config);

    /**
     * @brief 왼쪽 눈에만 렌더링
     *
     * LensConfig의 apply_left 설정과 무관하게 왼쪽 눈에만 렌더링.
     *
     * @param frame 입출력 이미지
     * @param iris_result 홍채 검출 결과
     * @param config 렌더링 설정
     * @return 렌더링 성공 여부
     */
    bool renderLeftEye(cv::Mat& frame,
                       const IrisResult& iris_result,
                       const LensConfig& config);

    /**
     * @brief 오른쪽 눈에만 렌더링
     *
     * LensConfig의 apply_right 설정과 무관하게 오른쪽 눈에만 렌더링.
     *
     * @param frame 입출력 이미지
     * @param iris_result 홍채 검출 결과
     * @param config 렌더링 설정
     * @return 렌더링 성공 여부
     */
    bool renderRightEye(cv::Mat& frame,
                        const IrisResult& iris_result,
                        const LensConfig& config);

    // ========================================
    // 렌더링 통계
    // ========================================

    /**
     * @brief 마지막 렌더링 시간 조회 (밀리초)
     * @return 렌더링 소요 시간 (ms), 이동된 객체의 경우 0.0
     */
    double getLastRenderTimeMs() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace iris_sdk
