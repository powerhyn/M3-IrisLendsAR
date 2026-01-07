# P1-W4-04: SDKManager 싱글톤

**태스크 ID**: P1-W4-04
**상태**: ⏳ 대기
**시작일**: -
**완료일**: -

---

## 1. 계획

### 목표
SDK의 전역 진입점인 SDKManager 싱글톤을 구현한다. 라이프사이클 관리, 설정, 로깅을 담당한다.

### 산출물
| 파일 | 설명 |
|------|------|
| `cpp/include/iris_sdk/sdk_manager.h` | SDKManager 클래스 선언 |
| `cpp/src/sdk_manager.cpp` | SDKManager 구현 |

### 검증 기준
- [ ] 싱글톤 패턴 동작 확인
- [ ] 스레드 안전한 초기화
- [ ] 라이프사이클 관리 (init → use → destroy)
- [ ] 전역 설정 관리
- [ ] 로깅 인터페이스 제공

### 선행 조건
- P1-W4-03: FrameProcessor 파이프라인 ✅

---

## 2. 분석

### 2.1 책임 범위

| 책임 | 설명 |
|------|------|
| 라이프사이클 | SDK 초기화, 해제 관리 |
| 팩토리 | 컴포넌트 인스턴스 생성 |
| 설정 | 전역 설정 저장/조회 |
| 로깅 | 로그 출력 관리 |
| 버전 | SDK 버전 정보 제공 |

### 2.2 클래스 설계

```cpp
#pragma once

#include "iris_sdk/types.h"
#include "iris_sdk/export.h"
#include <memory>
#include <string>
#include <functional>

namespace iris_sdk {

// Forward declarations
class IrisDetector;
class LensRenderer;
class FrameProcessor;

/**
 * @brief 로그 레벨
 */
enum class LogLevel {
    Verbose = 0,
    Debug = 1,
    Info = 2,
    Warning = 3,
    Error = 4,
    None = 5
};

/**
 * @brief 로그 콜백 함수 타입
 */
using LogCallback = std::function<void(LogLevel level,
                                        const char* tag,
                                        const char* message)>;

/**
 * @brief SDK 설정 구조체
 */
struct SDKConfig {
    // 모델 경로
    std::string model_path;

    // 검출 설정
    DetectorType detector_type = DetectorType::MediaPipe;
    float min_detection_confidence = 0.5f;
    float min_tracking_confidence = 0.5f;
    int max_faces = 1;

    // 성능 설정
    bool enable_gpu = false;
    int num_threads = 0;  // 0 = auto

    // 로깅 설정
    LogLevel log_level = LogLevel::Info;
    LogCallback log_callback = nullptr;
};

/**
 * @brief SDK 상태
 */
enum class SDKState {
    Uninitialized,
    Initializing,
    Ready,
    Error
};

/**
 * @brief SDK 관리자 (싱글톤)
 *
 * SDK의 전역 진입점. 초기화, 설정, 컴포넌트 생성을 담당.
 */
class IRIS_SDK_EXPORT SDKManager {
public:
    /**
     * @brief 싱글톤 인스턴스 획득
     */
    static SDKManager& getInstance();

    /**
     * @brief SDK 초기화
     * @param config 설정
     * @return 성공 여부
     */
    bool initialize(const SDKConfig& config);

    /**
     * @brief SDK 초기화 (기본 설정)
     * @param model_path 모델 파일 경로
     * @return 성공 여부
     */
    bool initialize(const std::string& model_path);

    /**
     * @brief SDK 해제
     */
    void shutdown();

    /**
     * @brief 상태 확인
     */
    SDKState getState() const;
    bool isReady() const { return getState() == SDKState::Ready; }

    /**
     * @brief FrameProcessor 생성
     * @return 새 FrameProcessor 인스턴스
     */
    std::unique_ptr<FrameProcessor> createFrameProcessor();

    /**
     * @brief IrisDetector 생성
     * @param type 검출기 종류
     * @return 새 IrisDetector 인스턴스
     */
    std::unique_ptr<IrisDetector> createDetector(
        DetectorType type = DetectorType::MediaPipe);

    /**
     * @brief LensRenderer 생성
     * @return 새 LensRenderer 인스턴스
     */
    std::unique_ptr<LensRenderer> createRenderer();

    // 설정 접근
    const SDKConfig& getConfig() const;
    void setLogLevel(LogLevel level);
    void setLogCallback(LogCallback callback);

    // 로깅
    /**
     * @brief 로그 출력
     * @param level 로그 레벨
     * @param tag 로그 태그
     * @param format printf 형식 문자열
     * @param ... 가변 인자
     */
    void log(LogLevel level, const char* tag, const char* format, ...);

    // 정보 조회
    static const char* getVersion();
    static const char* getBuildInfo();

    // 싱글톤 규칙
    SDKManager(const SDKManager&) = delete;
    SDKManager& operator=(const SDKManager&) = delete;

private:
    SDKManager();
    ~SDKManager();

    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief 편의 매크로: SDK 로그 출력
 */
#define IRIS_LOG(level, tag, ...) \
    iris_sdk::SDKManager::getInstance().log(level, tag, __VA_ARGS__)

#define IRIS_LOGV(tag, ...) IRIS_LOG(iris_sdk::LogLevel::Verbose, tag, __VA_ARGS__)
#define IRIS_LOGD(tag, ...) IRIS_LOG(iris_sdk::LogLevel::Debug, tag, __VA_ARGS__)
#define IRIS_LOGI(tag, ...) IRIS_LOG(iris_sdk::LogLevel::Info, tag, __VA_ARGS__)
#define IRIS_LOGW(tag, ...) IRIS_LOG(iris_sdk::LogLevel::Warning, tag, __VA_ARGS__)
#define IRIS_LOGE(tag, ...) IRIS_LOG(iris_sdk::LogLevel::Error, tag, __VA_ARGS__)

} // namespace iris_sdk
```

### 2.3 싱글톤 구현 패턴

**Meyer's Singleton (C++11 스레드 안전)**
```cpp
SDKManager& SDKManager::getInstance() {
    static SDKManager instance;  // 스레드 안전 초기화 (C++11 보장)
    return instance;
}
```

### 2.4 라이프사이클 관리

```
┌─────────────────┐
│  Uninitialized  │ ─── initialize() ──▶ ┌─────────────────┐
└─────────────────┘                      │  Initializing   │
        ▲                                └─────────────────┘
        │                                        │
        │                               ┌────────┴────────┐
   shutdown()                           │                 │
        │                          성공  ▼                 ▼ 실패
┌─────────────────┐                ┌─────────┐      ┌─────────┐
│     Ready       │ ◀──────────── │  Ready  │      │  Error  │
└─────────────────┘                └─────────┘      └─────────┘
```

### 2.5 설정 관리

```cpp
class SDKManager::Impl {
public:
    bool initialize(const SDKConfig& config) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (state_ != SDKState::Uninitialized) {
            IRIS_LOGW("SDKManager", "Already initialized");
            return state_ == SDKState::Ready;
        }

        state_ = SDKState::Initializing;
        config_ = config;

        // 로깅 설정
        if (config.log_callback) {
            log_callback_ = config.log_callback;
        }
        log_level_ = config.log_level;

        // 모델 경로 검증
        if (config.model_path.empty()) {
            IRIS_LOGE("SDKManager", "Model path is empty");
            state_ = SDKState::Error;
            return false;
        }

        if (!validateModelPath(config.model_path)) {
            IRIS_LOGE("SDKManager", "Invalid model path: %s",
                      config.model_path.c_str());
            state_ = SDKState::Error;
            return false;
        }

        state_ = SDKState::Ready;
        IRIS_LOGI("SDKManager", "SDK initialized successfully v%s",
                  getVersion());
        return true;
    }

private:
    std::mutex mutex_;
    SDKState state_ = SDKState::Uninitialized;
    SDKConfig config_;
    LogLevel log_level_ = LogLevel::Info;
    LogCallback log_callback_;
};
```

### 2.6 팩토리 메서드

```cpp
std::unique_ptr<FrameProcessor> SDKManager::createFrameProcessor() {
    if (!isReady()) {
        IRIS_LOGE("SDKManager", "SDK not initialized");
        return nullptr;
    }

    auto processor = std::make_unique<FrameProcessor>();
    if (!processor->initialize(impl_->config_.model_path,
                               impl_->config_.detector_type)) {
        IRIS_LOGE("SDKManager", "Failed to create FrameProcessor");
        return nullptr;
    }

    return processor;
}

std::unique_ptr<IrisDetector> SDKManager::createDetector(DetectorType type) {
    if (!isReady()) {
        IRIS_LOGE("SDKManager", "SDK not initialized");
        return nullptr;
    }

    // 내부 팩토리 함수 사용 (detail 네임스페이스)
    auto detector = iris_sdk::detail::createDetector(type);
    if (!detector || !detector->initialize(impl_->config_.model_path)) {
        IRIS_LOGE("SDKManager", "Failed to create IrisDetector");
        return nullptr;
    }

    return detector;
}
```

### 2.7 로깅 시스템

```cpp
// SDKManager의 public log() 메서드 - Impl에 위임
void SDKManager::log(LogLevel level, const char* tag,
                     const char* format, ...) {
    va_list args;
    va_start(args, format);
    impl_->logV(level, tag, format, args);
    va_end(args);
}

// Impl의 실제 로깅 구현
void SDKManager::Impl::logV(LogLevel level, const char* tag,
                            const char* format, va_list args) {
    if (level < log_level_) return;

    char buffer[1024];
    vsnprintf(buffer, sizeof(buffer), format, args);

    if (log_callback_) {
        log_callback_(level, tag, buffer);
    } else {
        // 기본 출력
        const char* level_str = "";
        switch (level) {
            case LogLevel::Verbose: level_str = "V"; break;
            case LogLevel::Debug:   level_str = "D"; break;
            case LogLevel::Info:    level_str = "I"; break;
            case LogLevel::Warning: level_str = "W"; break;
            case LogLevel::Error:   level_str = "E"; break;
            default: level_str = "?"; break;
        }
        fprintf(stderr, "[%s] %s: %s\n", level_str, tag, buffer);
    }
}
```

---

## 3. 실행 내역

### 3.1 헤더 파일 작성

```bash
# 예정: cpp/include/iris_sdk/sdk_manager.h
```

### 3.2 구현 파일 작성

```bash
# 예정: cpp/src/sdk_manager.cpp
```

### 3.3 단위 테스트

```bash
# 예정: cpp/tests/test_sdk_manager.cpp
```

---

## 4. 검증 결과

### 검증 항목

| 항목 | 결과 | 비고 |
|------|------|------|
| 싱글톤 동작 | ⏳ 대기 | |
| 스레드 안전 초기화 | ⏳ 대기 | |
| 라이프사이클 | ⏳ 대기 | |
| 팩토리 메서드 | ⏳ 대기 | |
| 설정 관리 | ⏳ 대기 | |
| 로깅 | ⏳ 대기 | |

---

## 5. 이슈 및 학습

### 이슈

| ID | 내용 | 상태 | 해결방안 |
|----|------|------|----------|
| - | - | - | - |

### 결정 사항

| 결정 | 이유 |
|------|------|
| Meyer's Singleton | C++11 스레드 안전 보장, 간결한 구현 |
| 지연 초기화 | 리소스 효율성, 명시적 초기화 시점 |
| 팩토리 메서드 | 컴포넌트 생성 중앙 집중화 |

### 학습 내용

(실행 후 기록)

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-01-07 | 태스크 문서 생성, 설계 완료 |
| 2026-01-07 | 아키텍처 리뷰: log() public 메서드 선언 추가 (IRIS_LOG 매크로 지원) |
