/**
 * @file sdk_manager.h
 * @brief SDK 관리자 싱글톤 클래스 선언
 *
 * SDK의 전역 진입점으로 초기화, 설정, 팩토리 메서드를 제공합니다.
 * Meyer's Singleton 패턴으로 스레드 안전한 초기화를 보장합니다.
 */

#pragma once

#include <cstdarg>
#include <functional>
#include <memory>
#include <string>

#include "export.h"
#include "types.h"

namespace iris_sdk {

// 전방 선언
class LensRenderer;
class FrameProcessor;

// ============================================================
// 로깅 관련 타입
// ============================================================

/**
 * @brief 로그 레벨
 */
enum class LogLevel : int {
    Verbose = 0,    ///< 상세 디버그 정보
    Debug = 1,      ///< 디버그 정보
    Info = 2,       ///< 일반 정보
    Warning = 3,    ///< 경고
    Error = 4,      ///< 오류
    None = 5        ///< 로깅 비활성화
};

/**
 * @brief 로그 콜백 함수 타입
 * @param level 로그 레벨
 * @param tag 로그 태그 (컴포넌트 식별용)
 * @param message 로그 메시지
 */
using LogCallback = std::function<void(LogLevel level,
                                       const char* tag,
                                       const char* message)>;

// ============================================================
// SDK 설정
// ============================================================

/**
 * @brief SDK 설정 구조체
 *
 * SDK 초기화 시 전달되는 설정 옵션입니다.
 */
struct IRIS_SDK_EXPORT SDKConfig {
    // 모델 경로 (필수)
    std::string model_path;

    // 검출기 설정 (④ W4-D: detector_type 제거 — 코어 검출 미보유)
    float min_detection_confidence = 0.5f;  ///< 최소 검출 신뢰도 (0.0~1.0)
    float min_tracking_confidence = 0.5f;   ///< 최소 추적 신뢰도 (0.0~1.0)
    int max_faces = 1;                      ///< 최대 얼굴 수

    // 성능 설정
    bool enable_gpu = false;        ///< GPU 가속 사용 여부 (현재 미지원)
    int num_threads = 0;            ///< 스레드 수 (0=자동, CPU 코어 수 기준)

    // 로깅 설정
    LogLevel log_level = LogLevel::Info;    ///< 로그 레벨
    LogCallback log_callback = nullptr;     ///< 커스텀 로그 콜백 (nullptr=stderr 출력)
};

/**
 * @brief SDK 상태
 */
enum class SDKState : int {
    Uninitialized = 0,  ///< 초기화 전
    Initializing = 1,   ///< 초기화 중
    Ready = 2,          ///< 사용 준비 완료
    Error = 3           ///< 오류 상태
};

// ============================================================
// SDKManager 클래스
// ============================================================

/**
 * @brief SDK 관리자 (싱글톤)
 *
 * SDK의 전역 진입점으로 다음 기능을 제공합니다:
 * - 라이프사이클 관리 (initialize → use → shutdown)
 * - 설정 관리
 * - 팩토리 메서드 (FrameProcessor, IrisDetector, LensRenderer)
 * - 로깅 시스템
 *
 * @note Meyer's Singleton으로 구현 (C++11 스레드 안전 보장)
 * @note Pimpl 패턴으로 구현 세부사항 은닉
 *
 * 사용 예시:
 * @code
 * SDKConfig config;
 * config.model_path = "/path/to/models";
 * config.log_level = LogLevel::Debug;
 *
 * if (!SDKManager::getInstance().initialize(config)) {
 *     // 초기화 실패 처리
 * }
 *
 * auto processor = SDKManager::getInstance().createFrameProcessor();
 *
 * // ... 사용 ...
 *
 * SDKManager::getInstance().shutdown();
 * @endcode
 */
class IRIS_SDK_EXPORT SDKManager {
public:
    /**
     * @brief 싱글톤 인스턴스 획득
     *
     * C++11 이상에서 스레드 안전하게 초기화됩니다 (Meyer's Singleton).
     *
     * @return SDKManager 참조
     */
    static SDKManager& getInstance();

    // ========================================
    // 라이프사이클 관리
    // ========================================

    /**
     * @brief SDK 초기화
     *
     * SDK를 지정된 설정으로 초기화합니다.
     * 이미 초기화된 경우 경고 로그를 출력하고 현재 상태를 반환합니다.
     *
     * @param config SDK 설정
     * @return 초기화 성공 여부
     */
    [[nodiscard]] bool initialize(const SDKConfig& config);

    /**
     * @brief SDK 초기화 (간단 버전)
     *
     * 기본 설정으로 SDK를 초기화합니다. 모델 경로만 지정.
     *
     * @param model_path 모델 파일 디렉토리 경로
     * @return 초기화 성공 여부
     */
    [[nodiscard]] bool initialize(const std::string& model_path);

    /**
     * @brief SDK 종료
     *
     * 모든 리소스를 해제하고 초기화 전 상태로 되돌립니다.
     * 여러 번 호출해도 안전합니다.
     */
    void shutdown();

    /**
     * @brief SDK 상태 조회
     * @return 현재 SDK 상태
     */
    [[nodiscard]] SDKState getState() const noexcept;

    /**
     * @brief SDK 준비 상태 확인
     * @return Ready 상태이면 true
     */
    [[nodiscard]] bool isReady() const noexcept { return getState() == SDKState::Ready; }

    // ========================================
    // 팩토리 메서드
    // ========================================

    /**
     * @brief FrameProcessor 생성
     *
     * 새 FrameProcessor 인스턴스를 생성합니다.
     * SDK가 초기화되지 않은 경우 nullptr을 반환합니다.
     *
     * @return 새 FrameProcessor 인스턴스 또는 nullptr
     */
    [[nodiscard]] std::unique_ptr<FrameProcessor> createFrameProcessor();

    /**
     * @brief LensRenderer 생성
     *
     * 새 LensRenderer 인스턴스를 생성합니다.
     * SDK가 초기화되지 않은 경우 nullptr을 반환합니다.
     *
     * @return 새 LensRenderer 인스턴스 또는 nullptr
     */
    [[nodiscard]] std::unique_ptr<LensRenderer> createRenderer();

    // ========================================
    // 설정 접근
    // ========================================

    /**
     * @brief 현재 설정 조회
     *
     * 스레드 안전을 위해 설정의 복사본을 반환합니다.
     *
     * @return 현재 SDK 설정의 복사본
     */
    [[nodiscard]] SDKConfig getConfig() const;

    /**
     * @brief 로그 레벨 변경
     *
     * 런타임에 로그 레벨을 변경합니다.
     *
     * @param level 새 로그 레벨
     */
    void setLogLevel(LogLevel level);

    /**
     * @brief 로그 콜백 설정
     *
     * 런타임에 로그 콜백을 변경합니다.
     *
     * @param callback 새 로그 콜백 (nullptr=stderr 출력으로 복원)
     */
    void setLogCallback(LogCallback callback);

    // ========================================
    // 로깅
    // ========================================

    /**
     * @brief 로그 출력
     *
     * 설정된 로그 레벨에 따라 메시지를 출력합니다.
     * printf 스타일 포맷 문자열을 지원합니다.
     *
     * @param level 로그 레벨
     * @param tag 로그 태그 (컴포넌트 식별용)
     * @param format printf 형식 문자열
     * @param ... 가변 인자
     */
    void log(LogLevel level, const char* tag, const char* format, ...);

    /**
     * @brief 로그 출력 (va_list 버전)
     *
     * @param level 로그 레벨
     * @param tag 로그 태그
     * @param format printf 형식 문자열
     * @param args va_list 인자
     */
    void logV(LogLevel level, const char* tag, const char* format, va_list args);

    // ========================================
    // 정보 조회
    // ========================================

    /**
     * @brief SDK 버전 문자열 반환
     * @return 버전 문자열 (예: "1.0.0")
     */
    [[nodiscard]] static const char* getVersion() noexcept;

    /**
     * @brief 빌드 정보 문자열 반환
     * @return 빌드 정보 (예: "IrisLensSDK v1.0.0 (Debug, 2024-01-01)")
     */
    [[nodiscard]] static const char* getBuildInfo() noexcept;

    // ========================================
    // 싱글톤 규칙
    // ========================================

    // 복사 금지
    SDKManager(const SDKManager&) = delete;
    SDKManager& operator=(const SDKManager&) = delete;

    // 이동 금지
    SDKManager(SDKManager&&) = delete;
    SDKManager& operator=(SDKManager&&) = delete;

private:
    SDKManager();
    ~SDKManager();

    class Impl;
    std::unique_ptr<Impl> impl_;
};

// ============================================================
// 편의 매크로: SDK 로그 출력
// ============================================================

/**
 * @def IRIS_LOG
 * @brief 지정된 레벨로 로그 출력
 */
#define IRIS_LOG(level, tag, ...) \
    iris_sdk::SDKManager::getInstance().log(level, tag, __VA_ARGS__)

/**
 * @def IRIS_LOGV
 * @brief Verbose 레벨 로그
 */
#define IRIS_LOGV(tag, ...) IRIS_LOG(iris_sdk::LogLevel::Verbose, tag, __VA_ARGS__)

/**
 * @def IRIS_LOGD
 * @brief Debug 레벨 로그
 */
#define IRIS_LOGD(tag, ...) IRIS_LOG(iris_sdk::LogLevel::Debug, tag, __VA_ARGS__)

/**
 * @def IRIS_LOGI
 * @brief Info 레벨 로그
 */
#define IRIS_LOGI(tag, ...) IRIS_LOG(iris_sdk::LogLevel::Info, tag, __VA_ARGS__)

/**
 * @def IRIS_LOGW
 * @brief Warning 레벨 로그
 */
#define IRIS_LOGW(tag, ...) IRIS_LOG(iris_sdk::LogLevel::Warning, tag, __VA_ARGS__)

/**
 * @def IRIS_LOGE
 * @brief Error 레벨 로그
 */
#define IRIS_LOGE(tag, ...) IRIS_LOG(iris_sdk::LogLevel::Error, tag, __VA_ARGS__)

} // namespace iris_sdk
