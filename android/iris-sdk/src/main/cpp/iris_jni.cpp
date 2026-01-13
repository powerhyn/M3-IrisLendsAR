/**
 * @file iris_jni.cpp
 * @brief IrisLensSDK JNI 네이티브 메서드 구현
 *
 * Java com.irislenssdk.IrisLensSDK 클래스의 네이티브 메서드를 구현합니다.
 *
 * @author IrisLensSDK Team
 * @version 1.0.0
 * @copyright Apache 2.0 License
 */

#include "jni_utils.h"
#include "iris_sdk/sdk_api.h"

#include <cstring>
#include <mutex>

// ============================================================================
// 전역 상태
// ============================================================================

namespace iris {
namespace jni {

// 전역 JNI 캐시 인스턴스
JniCache g_jniCache;

// JNI 캐시 초기화
bool JniCache::init(JNIEnv* env) {
    if (!env) return false;

    // IrisResult 클래스 찾기
    jclass localIrisResultClass = env->FindClass("com/irislenssdk/IrisResult");
    if (!localIrisResultClass) {
        LOGE("Failed to find IrisResult class");
        return false;
    }
    irisResultClass = static_cast<jclass>(env->NewGlobalRef(localIrisResultClass));
    env->DeleteLocalRef(localIrisResultClass);

    // IrisResult 필드 ID 캐시
    irisResult_detected = env->GetFieldID(irisResultClass, "detected", "Z");
    irisResult_leftDetected = env->GetFieldID(irisResultClass, "leftDetected", "Z");
    irisResult_rightDetected = env->GetFieldID(irisResultClass, "rightDetected", "Z");
    irisResult_confidence = env->GetFieldID(irisResultClass, "confidence", "F");
    irisResult_leftIrisX = env->GetFieldID(irisResultClass, "leftIrisX", "F");
    irisResult_leftIrisY = env->GetFieldID(irisResultClass, "leftIrisY", "F");
    irisResult_leftIrisZ = env->GetFieldID(irisResultClass, "leftIrisZ", "F");
    irisResult_leftRadius = env->GetFieldID(irisResultClass, "leftRadius", "F");
    irisResult_rightIrisX = env->GetFieldID(irisResultClass, "rightIrisX", "F");
    irisResult_rightIrisY = env->GetFieldID(irisResultClass, "rightIrisY", "F");
    irisResult_rightIrisZ = env->GetFieldID(irisResultClass, "rightIrisZ", "F");
    irisResult_rightRadius = env->GetFieldID(irisResultClass, "rightRadius", "F");
    irisResult_faceRectX = env->GetFieldID(irisResultClass, "faceRectX", "F");
    irisResult_faceRectY = env->GetFieldID(irisResultClass, "faceRectY", "F");
    irisResult_faceRectWidth = env->GetFieldID(irisResultClass, "faceRectWidth", "F");
    irisResult_faceRectHeight = env->GetFieldID(irisResultClass, "faceRectHeight", "F");
    irisResult_facePitch = env->GetFieldID(irisResultClass, "facePitch", "F");
    irisResult_faceYaw = env->GetFieldID(irisResultClass, "faceYaw", "F");
    irisResult_faceRoll = env->GetFieldID(irisResultClass, "faceRoll", "F");
    irisResult_timestampMs = env->GetFieldID(irisResultClass, "timestampMs", "J");
    irisResult_frameWidth = env->GetFieldID(irisResultClass, "frameWidth", "I");
    irisResult_frameHeight = env->GetFieldID(irisResultClass, "frameHeight", "I");

    // 필드 ID 검증
    if (!irisResult_detected || !irisResult_leftDetected || !irisResult_rightDetected ||
        !irisResult_confidence || !irisResult_leftIrisX || !irisResult_leftIrisY ||
        !irisResult_leftIrisZ || !irisResult_leftRadius || !irisResult_rightIrisX ||
        !irisResult_rightIrisY || !irisResult_rightIrisZ || !irisResult_rightRadius ||
        !irisResult_faceRectX || !irisResult_faceRectY || !irisResult_faceRectWidth ||
        !irisResult_faceRectHeight || !irisResult_facePitch || !irisResult_faceYaw ||
        !irisResult_faceRoll || !irisResult_timestampMs || !irisResult_frameWidth ||
        !irisResult_frameHeight) {
        LOGE("Failed to get IrisResult field IDs");
        return false;
    }

    // LensConfig 클래스 찾기
    jclass localLensConfigClass = env->FindClass("com/irislenssdk/LensConfig");
    if (!localLensConfigClass) {
        LOGE("Failed to find LensConfig class");
        return false;
    }
    lensConfigClass = static_cast<jclass>(env->NewGlobalRef(localLensConfigClass));
    env->DeleteLocalRef(localLensConfigClass);

    // LensConfig 필드 ID 캐시
    lensConfig_opacity = env->GetFieldID(lensConfigClass, "opacity", "F");
    lensConfig_scale = env->GetFieldID(lensConfigClass, "scale", "F");
    lensConfig_offsetX = env->GetFieldID(lensConfigClass, "offsetX", "F");
    lensConfig_offsetY = env->GetFieldID(lensConfigClass, "offsetY", "F");
    lensConfig_blendMode = env->GetFieldID(lensConfigClass, "blendMode", "I");
    lensConfig_edgeFeather = env->GetFieldID(lensConfigClass, "edgeFeather", "F");
    lensConfig_applyLeft = env->GetFieldID(lensConfigClass, "applyLeft", "Z");
    lensConfig_applyRight = env->GetFieldID(lensConfigClass, "applyRight", "Z");

    // 필드 ID 검증
    if (!lensConfig_opacity || !lensConfig_scale || !lensConfig_offsetX ||
        !lensConfig_offsetY || !lensConfig_blendMode || !lensConfig_edgeFeather ||
        !lensConfig_applyLeft || !lensConfig_applyRight) {
        LOGE("Failed to get LensConfig field IDs");
        return false;
    }

    LOGI("JNI cache initialized successfully");
    return true;
}

void JniCache::destroy(JNIEnv* env) {
    if (!env) return;

    if (irisResultClass) {
        env->DeleteGlobalRef(irisResultClass);
        irisResultClass = nullptr;
    }
    if (lensConfigClass) {
        env->DeleteGlobalRef(lensConfigClass);
        lensConfigClass = nullptr;
    }

    LOGI("JNI cache destroyed");
}

bool copyResultToJava(JNIEnv* env, const IrisResult& src, jobject dest) {
    if (!env || !dest) return false;
    if (!g_jniCache.isInitialized()) {
        LOGE("JNI cache not initialized");
        return false;
    }

    // 검출 상태
    env->SetBooleanField(dest, g_jniCache.irisResult_detected, src.detected);
    env->SetBooleanField(dest, g_jniCache.irisResult_leftDetected, src.left_detected);
    env->SetBooleanField(dest, g_jniCache.irisResult_rightDetected, src.right_detected);
    env->SetFloatField(dest, g_jniCache.irisResult_confidence, src.confidence);

    // 왼쪽 홍채 (center = index 0)
    env->SetFloatField(dest, g_jniCache.irisResult_leftIrisX, src.left_iris[0].x);
    env->SetFloatField(dest, g_jniCache.irisResult_leftIrisY, src.left_iris[0].y);
    env->SetFloatField(dest, g_jniCache.irisResult_leftIrisZ, src.left_iris[0].z);
    env->SetFloatField(dest, g_jniCache.irisResult_leftRadius, src.left_radius);

    // 오른쪽 홍채 (center = index 0)
    env->SetFloatField(dest, g_jniCache.irisResult_rightIrisX, src.right_iris[0].x);
    env->SetFloatField(dest, g_jniCache.irisResult_rightIrisY, src.right_iris[0].y);
    env->SetFloatField(dest, g_jniCache.irisResult_rightIrisZ, src.right_iris[0].z);
    env->SetFloatField(dest, g_jniCache.irisResult_rightRadius, src.right_radius);

    // 얼굴 영역
    env->SetFloatField(dest, g_jniCache.irisResult_faceRectX, src.face_rect.x);
    env->SetFloatField(dest, g_jniCache.irisResult_faceRectY, src.face_rect.y);
    env->SetFloatField(dest, g_jniCache.irisResult_faceRectWidth, src.face_rect.width);
    env->SetFloatField(dest, g_jniCache.irisResult_faceRectHeight, src.face_rect.height);

    // 얼굴 회전
    env->SetFloatField(dest, g_jniCache.irisResult_facePitch, src.face_rotation[0]);
    env->SetFloatField(dest, g_jniCache.irisResult_faceYaw, src.face_rotation[1]);
    env->SetFloatField(dest, g_jniCache.irisResult_faceRoll, src.face_rotation[2]);

    // 프레임 정보
    env->SetLongField(dest, g_jniCache.irisResult_timestampMs, src.timestamp_ms);
    env->SetIntField(dest, g_jniCache.irisResult_frameWidth, src.frame_width);
    env->SetIntField(dest, g_jniCache.irisResult_frameHeight, src.frame_height);

    return !checkAndLogException(env);
}

bool copyConfigFromJava(JNIEnv* env, jobject src, IrisLensConfig& dest) {
    if (!env || !src) return false;
    if (!g_jniCache.isInitialized()) {
        LOGE("JNI cache not initialized");
        return false;
    }

    dest.opacity = env->GetFloatField(src, g_jniCache.lensConfig_opacity);
    dest.scale = env->GetFloatField(src, g_jniCache.lensConfig_scale);
    dest.offset_x = env->GetFloatField(src, g_jniCache.lensConfig_offsetX);
    dest.offset_y = env->GetFloatField(src, g_jniCache.lensConfig_offsetY);
    dest.blend_mode = static_cast<IrisBlendMode>(
        env->GetIntField(src, g_jniCache.lensConfig_blendMode));
    dest.edge_feather = env->GetFloatField(src, g_jniCache.lensConfig_edgeFeather);
    dest.apply_left = env->GetBooleanField(src, g_jniCache.lensConfig_applyLeft);
    dest.apply_right = env->GetBooleanField(src, g_jniCache.lensConfig_applyRight);

    return !checkAndLogException(env);
}

bool checkAndLogException(JNIEnv* env) {
    if (!env) return false;

    if (env->ExceptionCheck()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
        return true;
    }
    return false;
}

void throwException(JNIEnv* env, const char* className, const char* message) {
    if (!env) return;

    jclass exClass = env->FindClass(className);
    if (exClass) {
        env->ThrowNew(exClass, message);
        env->DeleteLocalRef(exClass);
    }
}

}  // namespace jni
}  // namespace iris

using namespace iris::jni;

// ============================================================================
// JNI 라이프사이클
// ============================================================================

extern "C" {

/**
 * @brief JNI 라이브러리 로드 시 호출
 */
JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* /* reserved */) {
    LOGI("JNI_OnLoad called");

    JNIEnv* env = nullptr;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
        LOGE("Failed to get JNI environment");
        return JNI_ERR;
    }

    // JNI 캐시 초기화
    if (!g_jniCache.init(env)) {
        LOGE("Failed to initialize JNI cache");
        return JNI_ERR;
    }

    LOGI("JNI_OnLoad completed successfully");
    return JNI_VERSION_1_6;
}

/**
 * @brief JNI 라이브러리 언로드 시 호출
 */
JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* /* reserved */) {
    LOGI("JNI_OnUnload called");

    JNIEnv* env = nullptr;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) == JNI_OK) {
        g_jniCache.destroy(env);
    }

    LOGI("JNI_OnUnload completed");
}

// ============================================================================
// 네이티브 메서드 구현
// ============================================================================

/**
 * @brief SDK 초기화
 *
 * Java: native int nativeInit(String modelPath);
 */
JNIEXPORT jint JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeInit(
    JNIEnv* env,
    jclass /* clazz */,
    jstring modelPath) {

    LOGD("nativeInit called");

    if (!modelPath) {
        LOGE("nativeInit: modelPath is null");
        return static_cast<jint>(IRIS_SDK_NULL_POINTER);
    }

    ScopedString path(env, modelPath);
    if (!path.valid()) {
        LOGE("nativeInit: failed to get model path string");
        return static_cast<jint>(IRIS_SDK_INVALID_PARAM);
    }

    LOGD("Initializing SDK with model path: %s", path.get());
    IrisSdkError result = iris_sdk_init(path.get());

    if (result == IRIS_SDK_OK) {
        LOGI("SDK initialized successfully");
    } else {
        LOGE("SDK initialization failed: %d (%s)",
             result, iris_sdk_error_to_string(result));
    }

    return static_cast<jint>(result);
}

/**
 * @brief SDK 종료
 *
 * Java: native void nativeDestroy();
 */
JNIEXPORT void JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeDestroy(
    JNIEnv* /* env */,
    jclass /* clazz */) {

    LOGD("nativeDestroy called");
    iris_sdk_destroy();
    LOGI("SDK destroyed");
}

/**
 * @brief SDK 준비 상태 확인
 *
 * Java: native boolean nativeIsReady();
 */
JNIEXPORT jboolean JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeIsReady(
    JNIEnv* /* env */,
    jclass /* clazz */) {

    return iris_sdk_is_ready() ? JNI_TRUE : JNI_FALSE;
}

/**
 * @brief 렌즈 텍스처 로드
 *
 * Java: native int nativeLoadTexture(String path);
 */
JNIEXPORT jint JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeLoadTexture(
    JNIEnv* env,
    jclass /* clazz */,
    jstring texturePath) {

    LOGD("nativeLoadTexture called");

    if (!texturePath) {
        LOGE("nativeLoadTexture: texturePath is null");
        return static_cast<jint>(IRIS_SDK_NULL_POINTER);
    }

    ScopedString path(env, texturePath);
    if (!path.valid()) {
        LOGE("nativeLoadTexture: failed to get texture path string");
        return static_cast<jint>(IRIS_SDK_INVALID_PARAM);
    }

    LOGD("Loading texture from: %s", path.get());
    IrisSdkError result = iris_sdk_load_texture(path.get());

    if (result == IRIS_SDK_OK) {
        LOGI("Texture loaded successfully");
    } else {
        LOGE("Texture load failed: %d (%s)",
             result, iris_sdk_error_to_string(result));
    }

    return static_cast<jint>(result);
}

/**
 * @brief 홍채 검출
 *
 * Java: native int nativeDetect(byte[] frameData, int width, int height,
 *                               int format, IrisResult result);
 */
JNIEXPORT jint JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeDetect(
    JNIEnv* env,
    jclass /* clazz */,
    jbyteArray frameData,
    jint width,
    jint height,
    jint format,
    jobject resultObj) {

    LOGV("nativeDetect called: %dx%d, format=%d", width, height, format);

    // 파라미터 검증
    if (!frameData) {
        LOGE("nativeDetect: frameData is null");
        return static_cast<jint>(IRIS_SDK_NULL_POINTER);
    }
    if (!resultObj) {
        LOGE("nativeDetect: resultObj is null");
        return static_cast<jint>(IRIS_SDK_NULL_POINTER);
    }
    if (width <= 0 || height <= 0) {
        LOGE("nativeDetect: invalid dimensions %dx%d", width, height);
        return static_cast<jint>(IRIS_SDK_INVALID_PARAM);
    }

    // RAII로 바이트 배열 접근
    ScopedByteArray frame(env, frameData, JNI_ABORT);
    if (!frame.valid()) {
        LOGE("nativeDetect: failed to get frame data");
        return static_cast<jint>(IRIS_SDK_INVALID_PARAM);
    }

    // 버퍼 크기 검증
    if (!validateFrameBufferSize(frame.size(), width, height, format)) {
        LOGE("nativeDetect: frame buffer size mismatch");
        return static_cast<jint>(IRIS_SDK_INVALID_PARAM);
    }

    // C API 호출
    IrisResult nativeResult = {};
    IrisSdkError error = iris_sdk_detect(
        frame.data(),
        static_cast<int>(width),
        static_cast<int>(height),
        static_cast<IrisFrameFormat>(format),
        &nativeResult);

    if (error != IRIS_SDK_OK) {
        LOGW("Detection failed: %d (%s)", error, iris_sdk_error_to_string(error));
        return static_cast<jint>(error);
    }

    // 결과를 Java 객체로 복사
    if (!copyResultToJava(env, nativeResult, resultObj)) {
        LOGE("Failed to copy result to Java object");
        return static_cast<jint>(IRIS_SDK_UNKNOWN);
    }

    LOGV("Detection completed: detected=%d, confidence=%.2f",
         nativeResult.detected, nativeResult.confidence);

    return static_cast<jint>(IRIS_SDK_OK);
}

/**
 * @brief 홍채 검출 (회전 지원)
 *
 * Java: native int nativeDetectWithRotation(byte[] frameData, int width, int height,
 *                                           int format, int rotationDegrees, IrisResult result);
 */
JNIEXPORT jint JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeDetectWithRotation(
    JNIEnv* env,
    jclass /* clazz */,
    jbyteArray frameData,
    jint width,
    jint height,
    jint format,
    jint rotationDegrees,
    jobject resultObj) {

    LOGV("nativeDetectWithRotation called: %dx%d, format=%d, rotation=%d",
         width, height, format, rotationDegrees);

    // 파라미터 검증
    if (!frameData) {
        LOGE("nativeDetectWithRotation: frameData is null");
        return static_cast<jint>(IRIS_SDK_NULL_POINTER);
    }
    if (!resultObj) {
        LOGE("nativeDetectWithRotation: resultObj is null");
        return static_cast<jint>(IRIS_SDK_NULL_POINTER);
    }
    if (width <= 0 || height <= 0) {
        LOGE("nativeDetectWithRotation: invalid dimensions %dx%d", width, height);
        return static_cast<jint>(IRIS_SDK_INVALID_PARAM);
    }

    // RAII로 바이트 배열 접근
    ScopedByteArray frame(env, frameData, JNI_ABORT);
    if (!frame.valid()) {
        LOGE("nativeDetectWithRotation: failed to get frame data");
        return static_cast<jint>(IRIS_SDK_INVALID_PARAM);
    }

    // 버퍼 크기 검증
    if (!validateFrameBufferSize(frame.size(), width, height, format)) {
        LOGE("nativeDetectWithRotation: frame buffer size mismatch");
        return static_cast<jint>(IRIS_SDK_INVALID_PARAM);
    }

    // C API 호출 (회전 지원)
    IrisResult nativeResult = {};
    IrisSdkError error = iris_sdk_detect_with_rotation(
        frame.data(),
        static_cast<int>(width),
        static_cast<int>(height),
        static_cast<IrisFrameFormat>(format),
        static_cast<int>(rotationDegrees),
        &nativeResult);

    if (error != IRIS_SDK_OK) {
        LOGW("Detection with rotation failed: %d (%s)", error, iris_sdk_error_to_string(error));
        return static_cast<jint>(error);
    }

    // 결과를 Java 객체로 복사
    if (!copyResultToJava(env, nativeResult, resultObj)) {
        LOGE("Failed to copy result to Java object");
        return static_cast<jint>(IRIS_SDK_UNKNOWN);
    }

    LOGV("Detection with rotation completed: detected=%d, confidence=%.2f",
         nativeResult.detected, nativeResult.confidence);

    return static_cast<jint>(IRIS_SDK_OK);
}

/**
 * @brief 프레임 처리 (검출 + 렌더링)
 *
 * Java: native int nativeProcess(byte[] frameData, int width, int height,
 *                                int format, LensConfig config, IrisResult result);
 */
JNIEXPORT jint JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeProcess(
    JNIEnv* env,
    jclass /* clazz */,
    jbyteArray frameData,
    jint width,
    jint height,
    jint format,
    jobject configObj,
    jobject resultObj) {

    LOGV("nativeProcess called: %dx%d, format=%d", width, height, format);

    // 파라미터 검증
    if (!frameData) {
        LOGE("nativeProcess: frameData is null");
        return static_cast<jint>(IRIS_SDK_NULL_POINTER);
    }
    if (width <= 0 || height <= 0) {
        LOGE("nativeProcess: invalid dimensions %dx%d", width, height);
        return static_cast<jint>(IRIS_SDK_INVALID_PARAM);
    }

    // RAII로 바이트 배열 접근 (쓰기 가능)
    ScopedByteArray frame(env, frameData, 0);  // mode=0: 변경사항 복사
    if (!frame.valid()) {
        LOGE("nativeProcess: failed to get frame data");
        return static_cast<jint>(IRIS_SDK_INVALID_PARAM);
    }

    // 버퍼 크기 검증
    if (!validateFrameBufferSize(frame.size(), width, height, format)) {
        LOGE("nativeProcess: frame buffer size mismatch");
        return static_cast<jint>(IRIS_SDK_INVALID_PARAM);
    }

    // LensConfig 변환 (configObj가 null이면 검출만 수행)
    IrisLensConfig nativeConfig = {};
    IrisLensConfig* configPtr = nullptr;

    if (configObj) {
        if (copyConfigFromJava(env, configObj, nativeConfig)) {
            configPtr = &nativeConfig;
        } else {
            LOGW("Failed to copy config from Java, proceeding with detection only");
        }
    }

    // C API 호출
    IrisResult nativeResult = {};
    IrisResult* resultPtr = resultObj ? &nativeResult : nullptr;

    IrisSdkError error = iris_sdk_process(
        frame.data(),
        static_cast<int>(width),
        static_cast<int>(height),
        static_cast<IrisFrameFormat>(format),
        configPtr,
        resultPtr);

    if (error != IRIS_SDK_OK) {
        LOGW("Process failed: %d (%s)", error, iris_sdk_error_to_string(error));
        return static_cast<jint>(error);
    }

    // 결과를 Java 객체로 복사
    if (resultObj && resultPtr) {
        if (!copyResultToJava(env, nativeResult, resultObj)) {
            LOGE("Failed to copy result to Java object");
            return static_cast<jint>(IRIS_SDK_UNKNOWN);
        }
    }

    LOGV("Process completed: detected=%d", resultPtr ? resultPtr->detected : -1);

    return static_cast<jint>(IRIS_SDK_OK);
}

/**
 * @brief SDK 버전 반환
 *
 * Java: native String nativeGetVersion();
 */
JNIEXPORT jstring JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeGetVersion(
    JNIEnv* env,
    jclass /* clazz */) {

    const char* version = iris_sdk_get_version();
    return env->NewStringUTF(version ? version : "unknown");
}

/**
 * @brief 마지막 에러 메시지 반환
 *
 * Java: native String nativeGetLastError();
 */
JNIEXPORT jstring JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeGetLastError(
    JNIEnv* env,
    jclass /* clazz */) {

    const char* error = iris_sdk_get_last_error();
    return env->NewStringUTF(error ? error : "");
}

/**
 * @brief 에러 코드를 문자열로 변환
 *
 * Java: native String nativeErrorToString(int errorCode);
 */
JNIEXPORT jstring JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeErrorToString(
    JNIEnv* env,
    jclass /* clazz */,
    jint errorCode) {

    const char* str = iris_sdk_error_to_string(static_cast<IrisSdkError>(errorCode));
    return env->NewStringUTF(str ? str : "UNKNOWN");
}

/**
 * @brief 빌드 정보 반환
 *
 * Java: native String nativeGetBuildInfo();
 */
JNIEXPORT jstring JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeGetBuildInfo(
    JNIEnv* env,
    jclass /* clazz */) {

    const char* info = iris_sdk_get_build_info();
    return env->NewStringUTF(info ? info : "");
}

/**
 * @brief 메모리에서 텍스처 로드
 *
 * Java: native int nativeLoadTextureFromMemory(byte[] data, int width, int height);
 */
JNIEXPORT jint JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeLoadTextureFromMemory(
    JNIEnv* env,
    jclass /* clazz */,
    jbyteArray textureData,
    jint width,
    jint height) {

    LOGD("nativeLoadTextureFromMemory called: %dx%d", width, height);

    if (!textureData) {
        LOGE("nativeLoadTextureFromMemory: textureData is null");
        return static_cast<jint>(IRIS_SDK_NULL_POINTER);
    }
    if (width <= 0 || height <= 0) {
        LOGE("nativeLoadTextureFromMemory: invalid dimensions %dx%d", width, height);
        return static_cast<jint>(IRIS_SDK_INVALID_PARAM);
    }

    ScopedByteArray data(env, textureData, JNI_ABORT);
    if (!data.valid()) {
        LOGE("nativeLoadTextureFromMemory: failed to get texture data");
        return static_cast<jint>(IRIS_SDK_INVALID_PARAM);
    }

    IrisSdkError result = iris_sdk_load_texture_from_memory(
        data.data(),
        static_cast<int>(width),
        static_cast<int>(height));

    if (result == IRIS_SDK_OK) {
        LOGI("Texture loaded from memory successfully");
    } else {
        LOGE("Texture load from memory failed: %d (%s)",
             result, iris_sdk_error_to_string(result));
    }

    return static_cast<jint>(result);
}

/**
 * @brief 런타임 설정 변경
 *
 * Java: native int nativeSetConfig(String key, String value);
 */
JNIEXPORT jint JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeSetConfig(
    JNIEnv* env,
    jclass /* clazz */,
    jstring keyStr,
    jstring valueStr) {

    if (!keyStr || !valueStr) {
        LOGE("nativeSetConfig: key or value is null");
        return static_cast<jint>(IRIS_SDK_NULL_POINTER);
    }

    ScopedString key(env, keyStr);
    ScopedString value(env, valueStr);

    if (!key.valid() || !value.valid()) {
        LOGE("nativeSetConfig: failed to get key/value strings");
        return static_cast<jint>(IRIS_SDK_INVALID_PARAM);
    }

    LOGD("nativeSetConfig: %s = %s", key.get(), value.get());

    IrisSdkError result = iris_sdk_set_config(key.get(), value.get());

    if (result != IRIS_SDK_OK) {
        LOGW("Set config failed: %d (%s)", result, iris_sdk_error_to_string(result));
    }

    return static_cast<jint>(result);
}

}  // extern "C"
