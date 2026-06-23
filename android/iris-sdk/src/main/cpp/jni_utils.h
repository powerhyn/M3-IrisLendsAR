/**
 * @file jni_utils.h
 * @brief JNI 유틸리티 헬퍼
 *
 * RAII 기반 JNI 리소스 관리 및 데이터 변환 유틸리티
 *
 * @author IrisLensSDK Team
 * @version 1.0.0
 * @copyright Apache 2.0 License
 */

#ifndef IRIS_SDK_JNI_UTILS_H
#define IRIS_SDK_JNI_UTILS_H

#include <jni.h>
#include <android/log.h>
#include <string>
#include <cstdint>

// Forward declaration
struct IrisResult;
struct IrisLensConfig;
struct IrisLandmark;
struct IrisRect;
struct BeautyFilterConfig;
struct IrisBeautyConfigV2;

// ============================================================================
// Android 로깅 매크로
// ============================================================================

#define IRIS_JNI_TAG "IrisSDK-JNI"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, IRIS_JNI_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, IRIS_JNI_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, IRIS_JNI_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, IRIS_JNI_TAG, __VA_ARGS__)

#ifdef NDEBUG
#define LOGV(...)
#else
#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, IRIS_JNI_TAG, __VA_ARGS__)
#endif

namespace iris {
namespace jni {

// ============================================================================
// RAII 래퍼 클래스
// ============================================================================

/**
 * @brief RAII 기반 JNI 문자열 래퍼
 *
 * GetStringUTFChars/ReleaseStringUTFChars 쌍을 자동으로 관리합니다.
 * 예외 안전성과 메모리 누수 방지를 보장합니다.
 *
 * @code
 * ScopedString str(env, jstr);
 * if (str.get()) {
 *     // 안전하게 C 문자열 사용
 *     std::string cpp_str = str.get();
 * }
 * // 스코프 종료 시 자동 해제
 * @endcode
 */
class ScopedString {
public:
    ScopedString(JNIEnv* env, jstring jstr) noexcept
        : env_(env)
        , jstr_(jstr)
        , str_(nullptr) {
        if (env_ && jstr_) {
            str_ = env_->GetStringUTFChars(jstr_, nullptr);
        }
    }

    ~ScopedString() noexcept {
        if (str_ && env_ && jstr_) {
            env_->ReleaseStringUTFChars(jstr_, str_);
        }
    }

    // 복사 금지
    ScopedString(const ScopedString&) = delete;
    ScopedString& operator=(const ScopedString&) = delete;

    // 이동 허용
    ScopedString(ScopedString&& other) noexcept
        : env_(other.env_)
        , jstr_(other.jstr_)
        , str_(other.str_) {
        other.str_ = nullptr;
        other.jstr_ = nullptr;
    }

    ScopedString& operator=(ScopedString&& other) noexcept {
        if (this != &other) {
            if (str_ && env_ && jstr_) {
                env_->ReleaseStringUTFChars(jstr_, str_);
            }
            env_ = other.env_;
            jstr_ = other.jstr_;
            str_ = other.str_;
            other.str_ = nullptr;
            other.jstr_ = nullptr;
        }
        return *this;
    }

    /**
     * @brief C 문자열 포인터 반환
     * @return C 문자열 (nullptr일 수 있음)
     */
    [[nodiscard]] const char* get() const noexcept { return str_; }

    /**
     * @brief 문자열이 유효한지 확인
     */
    [[nodiscard]] bool valid() const noexcept { return str_ != nullptr; }

    /**
     * @brief std::string으로 변환
     * @return C++ 문자열 (유효하지 않으면 빈 문자열)
     */
    [[nodiscard]] std::string toString() const {
        return str_ ? std::string(str_) : std::string();
    }

    /**
     * @brief bool 변환 연산자
     */
    explicit operator bool() const noexcept { return valid(); }

private:
    JNIEnv* env_;
    jstring jstr_;
    const char* str_;
};

/**
 * @brief RAII 기반 JNI 바이트 배열 래퍼
 *
 * GetByteArrayElements/ReleaseByteArrayElements 쌍을 자동으로 관리합니다.
 *
 * @code
 * ScopedByteArray arr(env, jbyteArr);
 * if (arr.data()) {
 *     process_frame(arr.data(), arr.size());
 * }
 * @endcode
 */
class ScopedByteArray {
public:
    /**
     * @brief 생성자
     * @param env JNI 환경
     * @param arr Java 바이트 배열
     * @param mode 해제 모드 (기본값: JNI_ABORT - 복사 안함)
     */
    ScopedByteArray(JNIEnv* env, jbyteArray arr, jint mode = JNI_ABORT) noexcept
        : env_(env)
        , arr_(arr)
        , data_(nullptr)
        , size_(0)
        , mode_(mode) {
        if (env_ && arr_) {
            data_ = env_->GetByteArrayElements(arr_, nullptr);
            size_ = env_->GetArrayLength(arr_);
        }
    }

    ~ScopedByteArray() noexcept {
        if (data_ && env_ && arr_) {
            env_->ReleaseByteArrayElements(arr_, data_, mode_);
        }
    }

    // 복사 금지
    ScopedByteArray(const ScopedByteArray&) = delete;
    ScopedByteArray& operator=(const ScopedByteArray&) = delete;

    // 이동 허용
    ScopedByteArray(ScopedByteArray&& other) noexcept
        : env_(other.env_)
        , arr_(other.arr_)
        , data_(other.data_)
        , size_(other.size_)
        , mode_(other.mode_) {
        other.data_ = nullptr;
        other.arr_ = nullptr;
    }

    ScopedByteArray& operator=(ScopedByteArray&& other) noexcept {
        if (this != &other) {
            if (data_ && env_ && arr_) {
                env_->ReleaseByteArrayElements(arr_, data_, mode_);
            }
            env_ = other.env_;
            arr_ = other.arr_;
            data_ = other.data_;
            size_ = other.size_;
            mode_ = other.mode_;
            other.data_ = nullptr;
            other.arr_ = nullptr;
        }
        return *this;
    }

    /**
     * @brief 데이터 포인터 반환 (uint8_t*)
     */
    [[nodiscard]] uint8_t* data() noexcept {
        return reinterpret_cast<uint8_t*>(data_);
    }

    /**
     * @brief const 데이터 포인터 반환
     */
    [[nodiscard]] const uint8_t* data() const noexcept {
        return reinterpret_cast<const uint8_t*>(data_);
    }

    /**
     * @brief 원시 jbyte 포인터 반환
     */
    [[nodiscard]] jbyte* rawData() noexcept { return data_; }

    /**
     * @brief 배열 크기 반환
     */
    [[nodiscard]] jsize size() const noexcept { return size_; }

    /**
     * @brief 데이터가 유효한지 확인
     */
    [[nodiscard]] bool valid() const noexcept { return data_ != nullptr; }

    explicit operator bool() const noexcept { return valid(); }

private:
    JNIEnv* env_;
    jbyteArray arr_;
    jbyte* data_;
    jsize size_;
    jint mode_;
};

/**
 * @brief RAII 기반 JNI float 배열 래퍼
 *
 * GetFloatArrayElements/ReleaseFloatArrayElements 쌍을 자동으로 관리합니다.
 * 랜드마크 주입(478×3 정규화 좌표)을 C API로 넘길 때 사용합니다.
 * iris_set_landmarks가 호출 내에서 deep-copy하므로 기본 해제 모드는 JNI_ABORT
 * (네이티브가 배열을 수정하지 않음 — 복사 비용 회피).
 *
 * @code
 * ScopedFloatArray arr(env, jfloatArr);
 * if (arr.valid()) {
 *     iris_set_landmarks(arr.data(), 478, ...);
 * }
 * @endcode
 */
class ScopedFloatArray {
public:
    ScopedFloatArray(JNIEnv* env, jfloatArray arr, jint mode = JNI_ABORT) noexcept
        : env_(env)
        , arr_(arr)
        , data_(nullptr)
        , size_(0)
        , mode_(mode) {
        if (env_ && arr_) {
            data_ = env_->GetFloatArrayElements(arr_, nullptr);
            size_ = env_->GetArrayLength(arr_);
        }
    }

    ~ScopedFloatArray() noexcept {
        if (data_ && env_ && arr_) {
            env_->ReleaseFloatArrayElements(arr_, data_, mode_);
        }
    }

    // 복사 금지
    ScopedFloatArray(const ScopedFloatArray&) = delete;
    ScopedFloatArray& operator=(const ScopedFloatArray&) = delete;

    // 이동 허용
    ScopedFloatArray(ScopedFloatArray&& other) noexcept
        : env_(other.env_)
        , arr_(other.arr_)
        , data_(other.data_)
        , size_(other.size_)
        , mode_(other.mode_) {
        other.data_ = nullptr;
        other.arr_ = nullptr;
    }

    ScopedFloatArray& operator=(ScopedFloatArray&& other) noexcept {
        if (this != &other) {
            if (data_ && env_ && arr_) {
                env_->ReleaseFloatArrayElements(arr_, data_, mode_);
            }
            env_ = other.env_;
            arr_ = other.arr_;
            data_ = other.data_;
            size_ = other.size_;
            mode_ = other.mode_;
            other.data_ = nullptr;
            other.arr_ = nullptr;
        }
        return *this;
    }

    /**
     * @brief 데이터 포인터 반환 (const float*)
     */
    [[nodiscard]] const float* data() const noexcept {
        return reinterpret_cast<const float*>(data_);
    }

    /**
     * @brief 배열 길이 반환 (요소 수)
     */
    [[nodiscard]] jsize size() const noexcept { return size_; }

    /**
     * @brief 데이터가 유효한지 확인
     */
    [[nodiscard]] bool valid() const noexcept { return data_ != nullptr; }

    explicit operator bool() const noexcept { return valid(); }

private:
    JNIEnv* env_;
    jfloatArray arr_;
    jfloat* data_;
    jsize size_;
    jint mode_;
};

/**
 * @brief RAII 기반 JNI 로컬 참조 래퍼
 *
 * 로컬 참조를 자동으로 해제합니다.
 *
 * @tparam T JNI 객체 타입 (jobject, jclass, jstring 등)
 */
template<typename T>
class ScopedLocalRef {
public:
    ScopedLocalRef(JNIEnv* env, T ref) noexcept
        : env_(env)
        , ref_(ref) {}

    ~ScopedLocalRef() noexcept {
        if (ref_ && env_) {
            env_->DeleteLocalRef(ref_);
        }
    }

    // 복사 금지
    ScopedLocalRef(const ScopedLocalRef&) = delete;
    ScopedLocalRef& operator=(const ScopedLocalRef&) = delete;

    // 이동 허용
    ScopedLocalRef(ScopedLocalRef&& other) noexcept
        : env_(other.env_)
        , ref_(other.ref_) {
        other.ref_ = nullptr;
    }

    [[nodiscard]] T get() const noexcept { return ref_; }
    [[nodiscard]] bool valid() const noexcept { return ref_ != nullptr; }
    explicit operator bool() const noexcept { return valid(); }

    ScopedLocalRef& operator=(ScopedLocalRef&& other) noexcept {
        if (this != &other) {
            if (ref_ && env_) {
                env_->DeleteLocalRef(ref_);
            }
            env_ = other.env_;
            ref_ = other.ref_;
            other.ref_ = nullptr;
        }
        return *this;
    }

    /**
     * @brief 소유권 해제 (수동 관리로 전환)
     */
    T release() noexcept {
        T temp = ref_;
        ref_ = nullptr;
        return temp;
    }

private:
    JNIEnv* env_;
    T ref_;
};

// ============================================================================
// 캐시된 클래스 및 필드 ID
// ============================================================================

/**
 * @brief 캐시된 JNI 클래스 및 필드 정보
 *
 * JNI_OnLoad에서 초기화되어 성능 최적화를 제공합니다.
 */
struct JniCache {
    // IrisResult 클래스
    jclass irisResultClass = nullptr;
    jfieldID irisResult_detected = nullptr;
    jfieldID irisResult_leftDetected = nullptr;
    jfieldID irisResult_rightDetected = nullptr;
    jfieldID irisResult_confidence = nullptr;
    jfieldID irisResult_leftIrisX = nullptr;
    jfieldID irisResult_leftIrisY = nullptr;
    jfieldID irisResult_leftIrisZ = nullptr;
    jfieldID irisResult_leftRadius = nullptr;
    jfieldID irisResult_rightIrisX = nullptr;
    jfieldID irisResult_rightIrisY = nullptr;
    jfieldID irisResult_rightIrisZ = nullptr;
    jfieldID irisResult_rightRadius = nullptr;
    jfieldID irisResult_faceRectX = nullptr;
    jfieldID irisResult_faceRectY = nullptr;
    jfieldID irisResult_faceRectWidth = nullptr;
    jfieldID irisResult_faceRectHeight = nullptr;
    jfieldID irisResult_facePitch = nullptr;
    jfieldID irisResult_faceYaw = nullptr;
    jfieldID irisResult_faceRoll = nullptr;
    jfieldID irisResult_timestampMs = nullptr;
    jfieldID irisResult_frameWidth = nullptr;
    jfieldID irisResult_frameHeight = nullptr;
    jfieldID irisResult_faceMeshValid = nullptr;
    jfieldID irisResult_faceMesh = nullptr;

    // 눈꺼풀 가림 비율 메타데이터 (W3)
    jfieldID irisResult_eyelidRatioLeft = nullptr;
    jfieldID irisResult_eyelidRatioRight = nullptr;

    // P7-W2: iris ROI 실측 평균 luma (디텍트→렌더 round-trip 시 보존 필요)
    jfieldID irisResult_avgIrisLumaLeft = nullptr;
    jfieldID irisResult_avgIrisLumaRight = nullptr;

    // LensConfig 클래스
    jclass lensConfigClass = nullptr;
    jfieldID lensConfig_opacity = nullptr;
    jfieldID lensConfig_scale = nullptr;
    jfieldID lensConfig_offsetX = nullptr;
    jfieldID lensConfig_offsetY = nullptr;
    jfieldID lensConfig_rotation = nullptr;
    jfieldID lensConfig_blendMode = nullptr;
    jfieldID lensConfig_edgeFeather = nullptr;
    jfieldID lensConfig_applyLeft = nullptr;
    jfieldID lensConfig_applyRight = nullptr;
    jfieldID lensConfig_isMirror = nullptr;

    // BeautyFilterConfig 클래스
    jclass beautyConfigClass = nullptr;
    jfieldID beautyConfig_enabled = nullptr;
    jfieldID beautyConfig_intensity = nullptr;
    jfieldID beautyConfig_smoothing = nullptr;
    jfieldID beautyConfig_brightness = nullptr;
    jfieldID beautyConfig_softFocus = nullptr;

    // BeautyFilterConfigV2 클래스
    jclass beautyConfigV2Class = nullptr;
    jfieldID beautyConfigV2_enabled = nullptr;
    jfieldID beautyConfigV2_intensity = nullptr;
    jfieldID beautyConfigV2_brightness = nullptr;
    jfieldID beautyConfigV2_slimFace = nullptr;
    jfieldID beautyConfigV2_enlargeEyes = nullptr;
    jfieldID beautyConfigV2_thinChin = nullptr;
    jfieldID beautyConfigV2_useGpu = nullptr;
    jfieldID beautyConfigV2_roiOnly = nullptr;
    jfieldID beautyConfigV2_protectEyes = nullptr;
    jfieldID beautyConfigV2_protectLips = nullptr;
    jfieldID beautyConfigV2_protectNose = nullptr;
    jfieldID beautyConfigV2_downscaleFactor = nullptr;

    /**
     * @brief 캐시 초기화
     * @param env JNI 환경
     * @return 성공 여부
     */
    bool init(JNIEnv* env);

    /**
     * @brief 캐시 해제
     * @param env JNI 환경
     */
    void destroy(JNIEnv* env);

    /**
     * @brief 초기화 여부 확인
     */
    [[nodiscard]] bool isInitialized() const noexcept {
        return irisResultClass != nullptr && lensConfigClass != nullptr && beautyConfigClass != nullptr;
    }
};

/**
 * @brief 전역 JNI 캐시 인스턴스
 */
extern JniCache g_jniCache;

// ============================================================================
// 데이터 변환 함수
// ============================================================================

/**
 * @brief C++ IrisResult를 Java IrisResult 객체로 복사
 *
 * @param env JNI 환경
 * @param src C++ IrisResult 구조체
 * @param dest Java IrisResult 객체
 * @return 성공 여부
 */
bool copyResultToJava(JNIEnv* env, const IrisResult& src, jobject dest);

/**
 * @brief Java LensConfig를 C++ IrisLensConfig로 변환
 *
 * @param env JNI 환경
 * @param src Java LensConfig 객체
 * @param dest C++ IrisLensConfig 구조체
 * @return 성공 여부
 */
bool copyConfigFromJava(JNIEnv* env, jobject src, IrisLensConfig& dest);

/**
 * @brief Java BeautyFilterConfig를 C++ BeautyFilterConfig로 변환
 *
 * @param env JNI 환경
 * @param src Java BeautyFilterConfig 객체
 * @param dest C++ BeautyFilterConfig 구조체
 * @return 성공 여부
 */
bool copyBeautyConfigFromJava(JNIEnv* env, jobject src, BeautyFilterConfig& dest);

/**
 * @brief C++ BeautyFilterConfig를 Java BeautyFilterConfig 객체로 복사
 *
 * @param env JNI 환경
 * @param src C++ BeautyFilterConfig 구조체
 * @param dest Java BeautyFilterConfig 객체
 * @return 성공 여부
 */
bool copyBeautyConfigToJava(JNIEnv* env, const BeautyFilterConfig& src, jobject dest);

/**
 * @brief Java BeautyFilterConfigV2를 C IrisBeautyConfigV2로 변환
 *
 * @param env JNI 환경
 * @param src Java BeautyFilterConfigV2 객체
 * @param dest C IrisBeautyConfigV2 구조체
 * @return 성공 여부
 */
bool copyBeautyConfigV2FromJava(JNIEnv* env, jobject src, IrisBeautyConfigV2& dest);

/**
 * @brief C IrisBeautyConfigV2를 Java BeautyFilterConfigV2 객체로 복사
 *
 * @param env JNI 환경
 * @param src C IrisBeautyConfigV2 구조체
 * @param dest Java BeautyFilterConfigV2 객체
 * @return 성공 여부
 */
bool copyBeautyConfigV2ToJava(JNIEnv* env, const IrisBeautyConfigV2& src, jobject dest);

/**
 * @brief Java 예외 확인 및 로깅
 *
 * @param env JNI 환경
 * @return 예외 발생 여부
 */
bool checkAndLogException(JNIEnv* env);

/**
 * @brief Java 예외 발생
 *
 * @param env JNI 환경
 * @param className 예외 클래스 이름
 * @param message 예외 메시지
 */
void throwException(JNIEnv* env, const char* className, const char* message);

/**
 * @brief RuntimeException 발생
 */
inline void throwRuntimeException(JNIEnv* env, const char* message) {
    throwException(env, "java/lang/RuntimeException", message);
}

/**
 * @brief IllegalArgumentException 발생
 */
inline void throwIllegalArgumentException(JNIEnv* env, const char* message) {
    throwException(env, "java/lang/IllegalArgumentException", message);
}

/**
 * @brief IllegalStateException 발생
 */
inline void throwIllegalStateException(JNIEnv* env, const char* message) {
    throwException(env, "java/lang/IllegalStateException", message);
}

// ============================================================================
// 프레임 버퍼 크기 검증
// ============================================================================

/**
 * @brief 프레임 포맷에 따른 예상 버퍼 크기 계산
 *
 * @param width 프레임 너비
 * @param height 프레임 높이
 * @param format 프레임 포맷 (IrisFrameFormat)
 * @return 예상 바이트 수 (0이면 알 수 없는 포맷)
 */
inline jsize calculateExpectedFrameSize(jint width, jint height, jint format) noexcept {
    const jsize pixels = width * height;

    switch (format) {
        case 0:  // IRIS_FORMAT_RGBA
        case 1:  // IRIS_FORMAT_BGRA
            return pixels * 4;
        case 2:  // IRIS_FORMAT_RGB
        case 3:  // IRIS_FORMAT_BGR
            return pixels * 3;
        case 4:  // IRIS_FORMAT_NV21
        case 5:  // IRIS_FORMAT_NV12
            return pixels + (pixels / 2);  // Y + UV interleaved
        case 6:  // IRIS_FORMAT_GRAY
            return pixels;
        default:
            return 0;  // Unknown format
    }
}

/**
 * @brief 프레임 버퍼 크기 검증
 *
 * @param actualSize 실제 바이트 배열 크기
 * @param width 프레임 너비
 * @param height 프레임 높이
 * @param format 프레임 포맷
 * @return 유효하면 true
 */
inline bool validateFrameBufferSize(jsize actualSize, jint width, jint height, jint format) noexcept {
    if (width <= 0 || height <= 0) {
        return false;
    }

    jsize expectedSize = calculateExpectedFrameSize(width, height, format);
    if (expectedSize == 0) {
        LOGW("Unknown frame format: %d, skipping size validation", format);
        return true;  // 알 수 없는 포맷은 검증 스킵
    }

    if (actualSize < expectedSize) {
        LOGE("Frame buffer too small: expected %d bytes, got %d bytes",
             static_cast<int>(expectedSize), static_cast<int>(actualSize));
        return false;
    }

    return true;
}

}  // namespace jni
}  // namespace iris

#endif  // IRIS_SDK_JNI_UTILS_H
