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
#include "iris_sdk/beauty_filter.h"

#include <atomic>
#include <cstring>
#include <mutex>

// NV21 → RGBA 변환용
#include <android/bitmap.h>
#include <opencv2/imgproc.hpp>

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
    irisResult_faceMeshValid = env->GetFieldID(irisResultClass, "faceMeshValid", "Z");
    irisResult_faceMesh = env->GetFieldID(irisResultClass, "faceMesh", "[F");

    // Eye Refiner 메타데이터 필드 ID 캐시
    irisResult_irisQualityLeft = env->GetFieldID(irisResultClass, "irisQualityLeft", "F");
    irisResult_irisQualityRight = env->GetFieldID(irisResultClass, "irisQualityRight", "F");
    irisResult_eyelidRatioLeft = env->GetFieldID(irisResultClass, "eyelidRatioLeft", "F");
    irisResult_eyelidRatioRight = env->GetFieldID(irisResultClass, "eyelidRatioRight", "F");
    irisResult_eyeRefinerUsed = env->GetFieldID(irisResultClass, "eyeRefinerUsed", "Z");

    // 필드 ID 검증
    if (!irisResult_detected || !irisResult_leftDetected || !irisResult_rightDetected ||
        !irisResult_confidence || !irisResult_leftIrisX || !irisResult_leftIrisY ||
        !irisResult_leftIrisZ || !irisResult_leftRadius || !irisResult_rightIrisX ||
        !irisResult_rightIrisY || !irisResult_rightIrisZ || !irisResult_rightRadius ||
        !irisResult_faceRectX || !irisResult_faceRectY || !irisResult_faceRectWidth ||
        !irisResult_faceRectHeight || !irisResult_facePitch || !irisResult_faceYaw ||
        !irisResult_faceRoll || !irisResult_timestampMs || !irisResult_frameWidth ||
        !irisResult_frameHeight || !irisResult_faceMeshValid || !irisResult_faceMesh ||
        !irisResult_irisQualityLeft || !irisResult_irisQualityRight ||
        !irisResult_eyelidRatioLeft || !irisResult_eyelidRatioRight ||
        !irisResult_eyeRefinerUsed) {
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
    lensConfig_rotation = env->GetFieldID(lensConfigClass, "rotation", "F");
    lensConfig_blendMode = env->GetFieldID(lensConfigClass, "blendMode", "I");
    lensConfig_edgeFeather = env->GetFieldID(lensConfigClass, "edgeFeather", "F");
    lensConfig_applyLeft = env->GetFieldID(lensConfigClass, "applyLeft", "Z");
    lensConfig_applyRight = env->GetFieldID(lensConfigClass, "applyRight", "Z");
    lensConfig_isMirror = env->GetFieldID(lensConfigClass, "isMirror", "Z");

    // 필드 ID 검증
    if (!lensConfig_opacity || !lensConfig_scale || !lensConfig_offsetX ||
        !lensConfig_offsetY || !lensConfig_rotation || !lensConfig_blendMode || !lensConfig_edgeFeather ||
        !lensConfig_applyLeft || !lensConfig_applyRight || !lensConfig_isMirror) {
        LOGE("Failed to get LensConfig field IDs");
        return false;
    }

    // BeautyFilterConfig 클래스 찾기
    jclass localBeautyConfigClass = env->FindClass("com/irislenssdk/BeautyFilterConfig");
    if (!localBeautyConfigClass) {
        LOGE("Failed to find BeautyFilterConfig class");
        return false;
    }
    beautyConfigClass = static_cast<jclass>(env->NewGlobalRef(localBeautyConfigClass));
    env->DeleteLocalRef(localBeautyConfigClass);

    // BeautyFilterConfig 필드 ID 캐시
    beautyConfig_enabled = env->GetFieldID(beautyConfigClass, "enabled", "Z");
    beautyConfig_intensity = env->GetFieldID(beautyConfigClass, "intensity", "F");
    beautyConfig_smoothing = env->GetFieldID(beautyConfigClass, "smoothing", "F");
    beautyConfig_brightness = env->GetFieldID(beautyConfigClass, "brightness", "F");
    beautyConfig_softFocus = env->GetFieldID(beautyConfigClass, "softFocus", "F");

    // 필드 ID 검증
    if (!beautyConfig_enabled || !beautyConfig_intensity || !beautyConfig_smoothing ||
        !beautyConfig_brightness || !beautyConfig_softFocus) {
        LOGE("Failed to get BeautyFilterConfig field IDs");
        return false;
    }

    // BeautyFilterConfigV2 클래스 찾기
    jclass localBeautyConfigV2Class = env->FindClass("com/irislenssdk/BeautyFilterConfigV2");
    if (!localBeautyConfigV2Class) {
        LOGE("Failed to find BeautyFilterConfigV2 class");
        return false;
    }
    beautyConfigV2Class = static_cast<jclass>(env->NewGlobalRef(localBeautyConfigV2Class));
    env->DeleteLocalRef(localBeautyConfigV2Class);

    // BeautyFilterConfigV2 필드 ID 캐시
    beautyConfigV2_enabled = env->GetFieldID(beautyConfigV2Class, "enabled", "Z");
    beautyConfigV2_intensity = env->GetFieldID(beautyConfigV2Class, "intensity", "F");
    beautyConfigV2_smoothing = env->GetFieldID(beautyConfigV2Class, "smoothing", "F");
    beautyConfigV2_brightness = env->GetFieldID(beautyConfigV2Class, "brightness", "F");
    beautyConfigV2_softFocus = env->GetFieldID(beautyConfigV2Class, "softFocus", "F");
    beautyConfigV2_whitening = env->GetFieldID(beautyConfigV2Class, "whitening", "F");
    beautyConfigV2_colorBalance = env->GetFieldID(beautyConfigV2Class, "colorBalance", "F");
    beautyConfigV2_wrinkleRemove = env->GetFieldID(beautyConfigV2Class, "wrinkleRemove", "F");
    beautyConfigV2_skinQuality = env->GetFieldID(beautyConfigV2Class, "skinQuality", "F");
    beautyConfigV2_smoothIntensity = env->GetFieldID(beautyConfigV2Class, "smoothIntensity", "F");
    beautyConfigV2_poreReduction = env->GetFieldID(beautyConfigV2Class, "poreReduction", "F");
    beautyConfigV2_slimFace = env->GetFieldID(beautyConfigV2Class, "slimFace", "F");
    beautyConfigV2_enlargeEyes = env->GetFieldID(beautyConfigV2Class, "enlargeEyes", "F");
    beautyConfigV2_thinChin = env->GetFieldID(beautyConfigV2Class, "thinChin", "F");
    beautyConfigV2_useGpu = env->GetFieldID(beautyConfigV2Class, "useGpu", "Z");
    beautyConfigV2_roiOnly = env->GetFieldID(beautyConfigV2Class, "roiOnly", "Z");
    beautyConfigV2_protectEyes = env->GetFieldID(beautyConfigV2Class, "protectEyes", "Z");
    beautyConfigV2_protectLips = env->GetFieldID(beautyConfigV2Class, "protectLips", "Z");
    beautyConfigV2_protectNose = env->GetFieldID(beautyConfigV2Class, "protectNose", "Z");
    beautyConfigV2_downscaleFactor = env->GetFieldID(beautyConfigV2Class, "downscaleFactor", "I");
    beautyConfigV2_vividIntensity = env->GetFieldID(beautyConfigV2Class, "vividIntensity", "F");
    beautyConfigV2_vividSaturation = env->GetFieldID(beautyConfigV2Class, "vividSaturation", "F");
    beautyConfigV2_vividBrightness = env->GetFieldID(beautyConfigV2Class, "vividBrightness", "F");
    beautyConfigV2_vividWarmth = env->GetFieldID(beautyConfigV2Class, "vividWarmth", "F");

    // 필드 ID 검증
    if (!beautyConfigV2_enabled || !beautyConfigV2_intensity || !beautyConfigV2_smoothing ||
        !beautyConfigV2_brightness || !beautyConfigV2_softFocus || !beautyConfigV2_whitening ||
        !beautyConfigV2_colorBalance || !beautyConfigV2_wrinkleRemove || !beautyConfigV2_skinQuality ||
        !beautyConfigV2_smoothIntensity || !beautyConfigV2_poreReduction ||
        !beautyConfigV2_slimFace ||
        !beautyConfigV2_enlargeEyes || !beautyConfigV2_thinChin || !beautyConfigV2_useGpu ||
        !beautyConfigV2_roiOnly || !beautyConfigV2_protectEyes || !beautyConfigV2_protectLips ||
        !beautyConfigV2_protectNose || !beautyConfigV2_downscaleFactor ||
        !beautyConfigV2_vividIntensity || !beautyConfigV2_vividSaturation ||
        !beautyConfigV2_vividBrightness || !beautyConfigV2_vividWarmth) {
        LOGE("Failed to get BeautyFilterConfigV2 field IDs");
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
    if (beautyConfigClass) {
        env->DeleteGlobalRef(beautyConfigClass);
        beautyConfigClass = nullptr;
    }
    if (beautyConfigV2Class) {
        env->DeleteGlobalRef(beautyConfigV2Class);
        beautyConfigV2Class = nullptr;
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

    // Eye Refiner 메타데이터
    env->SetFloatField(dest, g_jniCache.irisResult_irisQualityLeft, src.iris_quality_left);
    env->SetFloatField(dest, g_jniCache.irisResult_irisQualityRight, src.iris_quality_right);
    env->SetFloatField(dest, g_jniCache.irisResult_eyelidRatioLeft, src.eyelid_ratio_left);
    env->SetFloatField(dest, g_jniCache.irisResult_eyelidRatioRight, src.eyelid_ratio_right);
    env->SetBooleanField(dest, g_jniCache.irisResult_eyeRefinerUsed, src.eye_refiner_used);

    // Face Mesh 데이터 복사
    env->SetBooleanField(dest, g_jniCache.irisResult_faceMeshValid, src.face_mesh_valid);

    if (src.face_mesh_valid) {
        // Java float[] 배열 가져오기
        jfloatArray faceMeshArray = static_cast<jfloatArray>(
            env->GetObjectField(dest, g_jniCache.irisResult_faceMesh));

        if (faceMeshArray) {
            constexpr int LANDMARK_COUNT = 478;
            constexpr int ARRAY_SIZE = LANDMARK_COUNT * 3;  // x, y, z for each landmark

            // 배열 크기 확인
            jsize arrayLen = env->GetArrayLength(faceMeshArray);
            if (arrayLen >= ARRAY_SIZE) {
                // 임시 버퍼에 데이터 복사
                float tempBuffer[ARRAY_SIZE];
                for (int i = 0; i < LANDMARK_COUNT; ++i) {
                    tempBuffer[i * 3] = src.face_mesh[i].x;
                    tempBuffer[i * 3 + 1] = src.face_mesh[i].y;
                    tempBuffer[i * 3 + 2] = src.face_mesh[i].z;
                }

                // Java 배열로 복사
                env->SetFloatArrayRegion(faceMeshArray, 0, ARRAY_SIZE, tempBuffer);
            }
        }
    }

    return !checkAndLogException(env);
}

bool copyResultFromJava(JNIEnv* env, jobject src, IrisResult& dest) {
    if (!env || !src) return false;
    if (!g_jniCache.isInitialized()) {
        LOGE("JNI cache not initialized");
        return false;
    }

    // 검출 상태
    dest.detected = env->GetBooleanField(src, g_jniCache.irisResult_detected);
    dest.left_detected = env->GetBooleanField(src, g_jniCache.irisResult_leftDetected);
    dest.right_detected = env->GetBooleanField(src, g_jniCache.irisResult_rightDetected);
    dest.confidence = env->GetFloatField(src, g_jniCache.irisResult_confidence);

    // 왼쪽 홍채
    dest.left_iris[0].x = env->GetFloatField(src, g_jniCache.irisResult_leftIrisX);
    dest.left_iris[0].y = env->GetFloatField(src, g_jniCache.irisResult_leftIrisY);
    dest.left_iris[0].z = env->GetFloatField(src, g_jniCache.irisResult_leftIrisZ);
    dest.left_radius = env->GetFloatField(src, g_jniCache.irisResult_leftRadius);

    // 오른쪽 홍채
    dest.right_iris[0].x = env->GetFloatField(src, g_jniCache.irisResult_rightIrisX);
    dest.right_iris[0].y = env->GetFloatField(src, g_jniCache.irisResult_rightIrisY);
    dest.right_iris[0].z = env->GetFloatField(src, g_jniCache.irisResult_rightIrisZ);
    dest.right_radius = env->GetFloatField(src, g_jniCache.irisResult_rightRadius);

    // 얼굴 영역
    dest.face_rect.x = env->GetFloatField(src, g_jniCache.irisResult_faceRectX);
    dest.face_rect.y = env->GetFloatField(src, g_jniCache.irisResult_faceRectY);
    dest.face_rect.width = env->GetFloatField(src, g_jniCache.irisResult_faceRectWidth);
    dest.face_rect.height = env->GetFloatField(src, g_jniCache.irisResult_faceRectHeight);

    // 얼굴 회전
    dest.face_rotation[0] = env->GetFloatField(src, g_jniCache.irisResult_facePitch);
    dest.face_rotation[1] = env->GetFloatField(src, g_jniCache.irisResult_faceYaw);
    dest.face_rotation[2] = env->GetFloatField(src, g_jniCache.irisResult_faceRoll);

    // 프레임 정보
    dest.timestamp_ms = env->GetLongField(src, g_jniCache.irisResult_timestampMs);
    dest.frame_width = env->GetIntField(src, g_jniCache.irisResult_frameWidth);
    dest.frame_height = env->GetIntField(src, g_jniCache.irisResult_frameHeight);

    // Eye Refiner 메타데이터
    dest.iris_quality_left = env->GetFloatField(src, g_jniCache.irisResult_irisQualityLeft);
    dest.iris_quality_right = env->GetFloatField(src, g_jniCache.irisResult_irisQualityRight);
    dest.eyelid_ratio_left = env->GetFloatField(src, g_jniCache.irisResult_eyelidRatioLeft);
    dest.eyelid_ratio_right = env->GetFloatField(src, g_jniCache.irisResult_eyelidRatioRight);
    dest.eye_refiner_used = env->GetBooleanField(src, g_jniCache.irisResult_eyeRefinerUsed);

    // Face Mesh
    dest.face_mesh_valid = env->GetBooleanField(src, g_jniCache.irisResult_faceMeshValid);

    if (dest.face_mesh_valid) {
        jfloatArray faceMeshArray = static_cast<jfloatArray>(
            env->GetObjectField(src, g_jniCache.irisResult_faceMesh));

        if (faceMeshArray) {
            constexpr int LANDMARK_COUNT = 478;
            constexpr int ARRAY_SIZE = LANDMARK_COUNT * 3;

            jsize arrayLen = env->GetArrayLength(faceMeshArray);
            if (arrayLen >= ARRAY_SIZE) {
                float tempBuffer[ARRAY_SIZE];
                env->GetFloatArrayRegion(faceMeshArray, 0, ARRAY_SIZE, tempBuffer);

                for (int i = 0; i < LANDMARK_COUNT; ++i) {
                    dest.face_mesh[i].x = tempBuffer[i * 3];
                    dest.face_mesh[i].y = tempBuffer[i * 3 + 1];
                    dest.face_mesh[i].z = tempBuffer[i * 3 + 2];
                }
            }
            env->DeleteLocalRef(faceMeshArray);
        }
    }

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
    dest.rotation = env->GetFloatField(src, g_jniCache.lensConfig_rotation);
    int rawBlendMode = env->GetIntField(src, g_jniCache.lensConfig_blendMode);
    if (rawBlendMode < IRIS_BLEND_NORMAL || rawBlendMode > IRIS_BLEND_COLOR_REPLACE) {
        // P6-W2 §5.9: invalid blend ID는 TintLinearV2(ID=5)로 fallback. 셰이더 측 §5.9 경고와 정합.
        LOGW("[IrisSDK] Invalid blend mode from Java: %d, falling back to LUMINANCE_TINT_LINEAR(5)",
             rawBlendMode);
        dest.blend_mode = IRIS_BLEND_LUMINANCE_TINT_LINEAR;
    } else {
        dest.blend_mode = static_cast<IrisBlendMode>(rawBlendMode);
    }
    dest.edge_feather = env->GetFloatField(src, g_jniCache.lensConfig_edgeFeather);
    dest.apply_left = env->GetBooleanField(src, g_jniCache.lensConfig_applyLeft);
    dest.apply_right = env->GetBooleanField(src, g_jniCache.lensConfig_applyRight);
    dest.is_mirror = env->GetBooleanField(src, g_jniCache.lensConfig_isMirror);

    return !checkAndLogException(env);
}

bool copyBeautyConfigFromJava(JNIEnv* env, jobject src, BeautyFilterConfig& dest) {
    if (!env || !src) return false;
    if (!g_jniCache.isInitialized()) {
        LOGE("JNI cache not initialized");
        return false;
    }

    dest.enabled = env->GetBooleanField(src, g_jniCache.beautyConfig_enabled);
    dest.intensity = env->GetFloatField(src, g_jniCache.beautyConfig_intensity);
    dest.smoothing = env->GetFloatField(src, g_jniCache.beautyConfig_smoothing);
    dest.brightness = env->GetFloatField(src, g_jniCache.beautyConfig_brightness);
    dest.softFocus = env->GetFloatField(src, g_jniCache.beautyConfig_softFocus);

    return !checkAndLogException(env);
}

bool copyBeautyConfigToJava(JNIEnv* env, const BeautyFilterConfig& src, jobject dest) {
    if (!env || !dest) return false;
    if (!g_jniCache.isInitialized()) {
        LOGE("JNI cache not initialized");
        return false;
    }

    env->SetBooleanField(dest, g_jniCache.beautyConfig_enabled, src.enabled);
    env->SetFloatField(dest, g_jniCache.beautyConfig_intensity, src.intensity);
    env->SetFloatField(dest, g_jniCache.beautyConfig_smoothing, src.smoothing);
    env->SetFloatField(dest, g_jniCache.beautyConfig_brightness, src.brightness);
    env->SetFloatField(dest, g_jniCache.beautyConfig_softFocus, src.softFocus);

    return !checkAndLogException(env);
}

bool copyBeautyConfigV2FromJava(JNIEnv* env, jobject src, IrisBeautyConfigV2& dest) {
    if (!env || !src) return false;
    if (!g_jniCache.beautyConfigV2Class) {
        LOGE("BeautyConfigV2 class not cached");
        return false;
    }

    // 기본값으로 초기화
    iris_sdk_default_beauty_config_v2_c(&dest);

    // Java 객체에서 값 복사
    dest.enabled = env->GetBooleanField(src, g_jniCache.beautyConfigV2_enabled) ? 1 : 0;
    dest.intensity = env->GetFloatField(src, g_jniCache.beautyConfigV2_intensity);
    dest.smoothing = env->GetFloatField(src, g_jniCache.beautyConfigV2_smoothing);
    dest.brightness = env->GetFloatField(src, g_jniCache.beautyConfigV2_brightness);
    dest.soft_focus = env->GetFloatField(src, g_jniCache.beautyConfigV2_softFocus);
    dest.whitening = env->GetFloatField(src, g_jniCache.beautyConfigV2_whitening);
    dest.color_balance = env->GetFloatField(src, g_jniCache.beautyConfigV2_colorBalance);
    dest.wrinkle_remove = env->GetFloatField(src, g_jniCache.beautyConfigV2_wrinkleRemove);
    dest.skin_quality = env->GetFloatField(src, g_jniCache.beautyConfigV2_skinQuality);
    dest.smooth_intensity = env->GetFloatField(src, g_jniCache.beautyConfigV2_smoothIntensity);
    dest.pore_reduction = env->GetFloatField(src, g_jniCache.beautyConfigV2_poreReduction);
    dest.slim_face = env->GetFloatField(src, g_jniCache.beautyConfigV2_slimFace);
    dest.enlarge_eyes = env->GetFloatField(src, g_jniCache.beautyConfigV2_enlargeEyes);
    dest.thin_chin = env->GetFloatField(src, g_jniCache.beautyConfigV2_thinChin);
    dest.use_gpu = env->GetBooleanField(src, g_jniCache.beautyConfigV2_useGpu) ? 1 : 0;
    dest.roi_only = env->GetBooleanField(src, g_jniCache.beautyConfigV2_roiOnly) ? 1 : 0;
    dest.protect_eyes = env->GetBooleanField(src, g_jniCache.beautyConfigV2_protectEyes) ? 1 : 0;
    dest.protect_lips = env->GetBooleanField(src, g_jniCache.beautyConfigV2_protectLips) ? 1 : 0;
    dest.protect_nose = env->GetBooleanField(src, g_jniCache.beautyConfigV2_protectNose) ? 1 : 0;
    dest.downscale_factor = env->GetIntField(src, g_jniCache.beautyConfigV2_downscaleFactor);
    dest.vivid_intensity = env->GetFloatField(src, g_jniCache.beautyConfigV2_vividIntensity);
    dest.vivid_saturation = env->GetFloatField(src, g_jniCache.beautyConfigV2_vividSaturation);
    dest.vivid_brightness = env->GetFloatField(src, g_jniCache.beautyConfigV2_vividBrightness);
    dest.vivid_warmth = env->GetFloatField(src, g_jniCache.beautyConfigV2_vividWarmth);

    return !checkAndLogException(env);
}

bool copyBeautyConfigV2ToJava(JNIEnv* env, const IrisBeautyConfigV2& src, jobject dest) {
    if (!env || !dest) return false;
    if (!g_jniCache.beautyConfigV2Class) {
        LOGE("BeautyConfigV2 class not cached");
        return false;
    }

    env->SetBooleanField(dest, g_jniCache.beautyConfigV2_enabled, src.enabled != 0);
    env->SetFloatField(dest, g_jniCache.beautyConfigV2_intensity, src.intensity);
    env->SetFloatField(dest, g_jniCache.beautyConfigV2_smoothing, src.smoothing);
    env->SetFloatField(dest, g_jniCache.beautyConfigV2_brightness, src.brightness);
    env->SetFloatField(dest, g_jniCache.beautyConfigV2_softFocus, src.soft_focus);
    env->SetFloatField(dest, g_jniCache.beautyConfigV2_whitening, src.whitening);
    env->SetFloatField(dest, g_jniCache.beautyConfigV2_colorBalance, src.color_balance);
    env->SetFloatField(dest, g_jniCache.beautyConfigV2_wrinkleRemove, src.wrinkle_remove);
    env->SetFloatField(dest, g_jniCache.beautyConfigV2_skinQuality, src.skin_quality);
    env->SetFloatField(dest, g_jniCache.beautyConfigV2_smoothIntensity, src.smooth_intensity);
    env->SetFloatField(dest, g_jniCache.beautyConfigV2_poreReduction, src.pore_reduction);
    env->SetFloatField(dest, g_jniCache.beautyConfigV2_slimFace, src.slim_face);
    env->SetFloatField(dest, g_jniCache.beautyConfigV2_enlargeEyes, src.enlarge_eyes);
    env->SetFloatField(dest, g_jniCache.beautyConfigV2_thinChin, src.thin_chin);
    env->SetBooleanField(dest, g_jniCache.beautyConfigV2_useGpu, src.use_gpu != 0);
    env->SetBooleanField(dest, g_jniCache.beautyConfigV2_roiOnly, src.roi_only != 0);
    env->SetBooleanField(dest, g_jniCache.beautyConfigV2_protectEyes, src.protect_eyes != 0);
    env->SetBooleanField(dest, g_jniCache.beautyConfigV2_protectLips, src.protect_lips != 0);
    env->SetBooleanField(dest, g_jniCache.beautyConfigV2_protectNose, src.protect_nose != 0);
    env->SetIntField(dest, g_jniCache.beautyConfigV2_downscaleFactor, src.downscale_factor);
    env->SetFloatField(dest, g_jniCache.beautyConfigV2_vividIntensity, src.vivid_intensity);
    env->SetFloatField(dest, g_jniCache.beautyConfigV2_vividSaturation, src.vivid_saturation);
    env->SetFloatField(dest, g_jniCache.beautyConfigV2_vividBrightness, src.vivid_brightness);
    env->SetFloatField(dest, g_jniCache.beautyConfigV2_vividWarmth, src.vivid_warmth);

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

// ============================================================================
// Detection Slot (더블 버퍼 — lock-free IrisResult 전달)
// ============================================================================
namespace {

struct DetectionSlot {
    IrisResult data{};
    std::atomic<bool> valid{false};
    std::atomic<uint64_t> generation{0};
};

DetectionSlot g_detection_slots[2];
std::atomic<int> g_active_slot_index{-1};  // -1 = 초기 미설정

}  // anonymous namespace

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

// ============================================================================
// GPU 가속 API
// ============================================================================

/**
 * @brief GPU 가속 사용 여부 설정
 *
 * init() 호출 전에 설정해야 합니다.
 *
 * Java: native void nativeSetGpuEnabled(boolean enable);
 */
JNIEXPORT void JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeSetGpuEnabled(
    JNIEnv* /* env */,
    jclass /* clazz */,
    jboolean enable) {

    LOGD("nativeSetGpuEnabled: %s", enable ? "true" : "false");
    iris_sdk_set_gpu_enabled(enable == JNI_TRUE);
}

/**
 * @brief GPU 가속 사용 가능 여부 확인
 *
 * SDK가 GPU delegate와 함께 빌드되었는지 확인합니다.
 *
 * Java: native boolean nativeIsGpuAvailable();
 */
JNIEXPORT jboolean JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeIsGpuAvailable(
    JNIEnv* /* env */,
    jclass /* clazz */) {

    bool available = iris_sdk_is_gpu_available();
    LOGD("nativeIsGpuAvailable: %s", available ? "true" : "false");
    return available ? JNI_TRUE : JNI_FALSE;
}

/**
 * @brief 현재 GPU 사용 상태 확인
 *
 * 런타임에 실제로 GPU delegate가 활성화되어 있는지 확인합니다.
 *
 * Java: native boolean nativeIsUsingGpu();
 */
JNIEXPORT jboolean JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeIsUsingGpu(
    JNIEnv* /* env */,
    jclass /* clazz */) {

    bool using_gpu = iris_sdk_is_using_gpu();
    LOGD("nativeIsUsingGpu: %s", using_gpu ? "true" : "false");
    return using_gpu ? JNI_TRUE : JNI_FALSE;
}

/**
 * @brief 얼굴 검출 최소 신뢰도 설정
 *
 * 얼굴 검출 결과의 최소 신뢰도. 이 값 이하면 검출되지 않은 것으로 처리.
 *
 * Java: native void nativeSetMinDetectionConfidence(float minConfidence);
 */
JNIEXPORT void JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeSetMinDetectionConfidence(
    JNIEnv* /* env */,
    jclass /* clazz */,
    jfloat minConfidence) {

    LOGD("nativeSetMinDetectionConfidence: %.2f", minConfidence);
    iris_sdk_set_min_detection_confidence(minConfidence);
}

/**
 * @brief 랜드마크 추적 최소 신뢰도 설정
 *
 * 랜드마크 추적 결과의 최소 신뢰도.
 * 이 값 이하면 추적 실패로 간주하고 다시 Face Detection 수행.
 *
 * Java: native void nativeSetMinTrackingConfidence(float minConfidence);
 */
JNIEXPORT void JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeSetMinTrackingConfidence(
    JNIEnv* /* env */,
    jclass /* clazz */,
    jfloat minConfidence) {

    LOGD("nativeSetMinTrackingConfidence: %.2f", minConfidence);
    iris_sdk_set_min_tracking_confidence(minConfidence);
}

/**
 * @brief 얼굴 존재 최소 신뢰도 설정
 *
 * 추적 모드에서 이전 프레임 결과를 재사용할지 판단하는 임계값.
 * 이전 프레임의 confidence가 이 값 이상이어야 Face Detection을 스킵.
 *
 * Java: native void nativeSetMinPresenceConfidence(float minConfidence);
 */
JNIEXPORT void JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeSetMinPresenceConfidence(
    JNIEnv* /* env */,
    jclass /* clazz */,
    jfloat minConfidence) {

    LOGD("nativeSetMinPresenceConfidence: %.2f", minConfidence);
    iris_sdk_set_min_presence_confidence(minConfidence);
}

/**
 * @brief InferenceThread 사용 여부 설정 (벤치마크용)
 *
 * init() 호출 전에 설정해야 합니다.
 * false로 설정하면 전용 스레드 없이 직접 호출합니다.
 * GPU 가속은 InferenceThread 사용 시에만 지원됩니다.
 *
 * Java: native void nativeSetUseInferenceThread(boolean enable);
 */
JNIEXPORT void JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeSetUseInferenceThread(
    JNIEnv* /* env */,
    jclass /* clazz */,
    jboolean enable) {

    LOGD("nativeSetUseInferenceThread: %s", enable ? "true" : "false");
    iris_sdk_set_use_inference_thread(enable == JNI_TRUE);
}

/**
 * @brief InferenceThread 사용 상태 확인
 *
 * Java: native boolean nativeIsUsingInferenceThread();
 */
JNIEXPORT jboolean JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeIsUsingInferenceThread(
    JNIEnv* /* env */,
    jclass /* clazz */) {

    bool using_inference_thread = iris_sdk_is_using_inference_thread();
    LOGD("nativeIsUsingInferenceThread: %s", using_inference_thread ? "true" : "false");
    return using_inference_thread ? JNI_TRUE : JNI_FALSE;
}

// ============================================================================
// Beauty Filter API
// ============================================================================

/**
 * @brief 기본 뷰티 필터 설정 가져오기
 *
 * 기본값으로 초기화된 BeautyFilterConfig를 Java 객체에 복사합니다.
 *
 * Java: native void nativeDefaultBeautyConfig(BeautyFilterConfig config);
 */
JNIEXPORT void JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeDefaultBeautyConfig(
    JNIEnv* env,
    jclass /* clazz */,
    jobject configObj) {

    LOGD("nativeDefaultBeautyConfig called");

    if (!configObj) {
        LOGE("nativeDefaultBeautyConfig: configObj is null");
        return;
    }

    // C API로 기본 설정 가져오기
    BeautyFilterConfig nativeConfig = {};
    iris_sdk_default_beauty_config(&nativeConfig);

    // Java 객체로 복사
    if (!copyBeautyConfigToJava(env, nativeConfig, configObj)) {
        LOGE("Failed to copy default beauty config to Java object");
    }
}

/**
 * @brief 뷰티 필터 설정 적용
 *
 * Java 객체의 설정을 네이티브 뷰티 필터에 적용합니다.
 *
 * Java: native int nativeSetBeautyFilter(BeautyFilterConfig config);
 */
JNIEXPORT jint JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeSetBeautyFilter(
    JNIEnv* env,
    jclass /* clazz */,
    jobject configObj) {

    LOGD("nativeSetBeautyFilter called");

    if (!configObj) {
        LOGE("nativeSetBeautyFilter: configObj is null");
        return static_cast<jint>(IRIS_SDK_NULL_POINTER);
    }

    // Java 객체에서 설정 복사
    BeautyFilterConfig nativeConfig = {};
    if (!copyBeautyConfigFromJava(env, configObj, nativeConfig)) {
        LOGE("Failed to copy beauty config from Java object");
        return static_cast<jint>(IRIS_SDK_INVALID_PARAM);
    }

    LOGD("Setting beauty filter: enabled=%d, intensity=%.2f, smoothing=%.2f, brightness=%.2f, softFocus=%.2f",
         nativeConfig.enabled, nativeConfig.intensity, nativeConfig.smoothing,
         nativeConfig.brightness, nativeConfig.softFocus);

    // C API 호출
    IrisSdkError result = iris_sdk_set_beauty_filter(&nativeConfig);

    if (result == IRIS_SDK_OK) {
        LOGI("Beauty filter settings applied successfully");
    } else {
        LOGE("Failed to set beauty filter: %d (%s)",
             result, iris_sdk_error_to_string(result));
    }

    return static_cast<jint>(result);
}

/**
 * @brief 현재 뷰티 필터 설정 가져오기
 *
 * 현재 적용된 뷰티 필터 설정을 Java 객체로 반환합니다.
 *
 * Java: native int nativeGetBeautyFilter(BeautyFilterConfig config);
 */
JNIEXPORT jint JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeGetBeautyFilter(
    JNIEnv* env,
    jclass /* clazz */,
    jobject configObj) {

    LOGD("nativeGetBeautyFilter called");

    if (!configObj) {
        LOGE("nativeGetBeautyFilter: configObj is null");
        return static_cast<jint>(IRIS_SDK_NULL_POINTER);
    }

    // C API로 현재 설정 가져오기
    BeautyFilterConfig nativeConfig = {};
    IrisSdkError result = iris_sdk_get_beauty_filter(&nativeConfig);

    if (result != IRIS_SDK_OK) {
        LOGE("Failed to get beauty filter: %d (%s)",
             result, iris_sdk_error_to_string(result));
        return static_cast<jint>(result);
    }

    // Java 객체로 복사
    if (!copyBeautyConfigToJava(env, nativeConfig, configObj)) {
        LOGE("Failed to copy beauty config to Java object");
        return static_cast<jint>(IRIS_SDK_UNKNOWN);
    }

    LOGD("Got beauty filter: enabled=%d, intensity=%.2f, smoothing=%.2f, brightness=%.2f, softFocus=%.2f",
         nativeConfig.enabled, nativeConfig.intensity, nativeConfig.smoothing,
         nativeConfig.brightness, nativeConfig.softFocus);

    return static_cast<jint>(IRIS_SDK_OK);
}

/**
 * @brief 뷰티 필터 활성화 여부 확인
 *
 * Java: native boolean nativeIsBeautyFilterEnabled();
 */
JNIEXPORT jboolean JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeIsBeautyFilterEnabled(
    JNIEnv* /* env */,
    jclass /* clazz */) {

    bool enabled = iris_sdk_is_beauty_filter_enabled();
    LOGD("nativeIsBeautyFilterEnabled: %s", enabled ? "true" : "false");
    return enabled ? JNI_TRUE : JNI_FALSE;
}

/**
 * @brief 프레임에 뷰티 필터 적용
 *
 * 현재 설정된 뷰티 필터를 프레임에 적용합니다.
 * 프레임 데이터는 in-place로 수정됩니다.
 *
 * Java: native int nativeApplyBeautyFilter(byte[] frameData, int width, int height, int format);
 */
JNIEXPORT jint JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeApplyBeautyFilter(
    JNIEnv* env,
    jclass /* clazz */,
    jbyteArray frameData,
    jint width,
    jint height,
    jint format) {

    LOGV("nativeApplyBeautyFilter called: %dx%d, format=%d", width, height, format);

    // 파라미터 검증
    if (!frameData) {
        LOGE("nativeApplyBeautyFilter: frameData is null");
        return static_cast<jint>(IRIS_SDK_NULL_POINTER);
    }
    if (width <= 0 || height <= 0) {
        LOGE("nativeApplyBeautyFilter: invalid dimensions %dx%d", width, height);
        return static_cast<jint>(IRIS_SDK_INVALID_PARAM);
    }

    // RAII로 바이트 배열 접근 (쓰기 가능 - mode=0: 변경사항 복사)
    ScopedByteArray frame(env, frameData, 0);
    if (!frame.valid()) {
        LOGE("nativeApplyBeautyFilter: failed to get frame data");
        return static_cast<jint>(IRIS_SDK_INVALID_PARAM);
    }

    // 버퍼 크기 검증
    if (!validateFrameBufferSize(frame.size(), width, height, format)) {
        LOGE("nativeApplyBeautyFilter: frame buffer size mismatch");
        return static_cast<jint>(IRIS_SDK_INVALID_PARAM);
    }

    // C API 호출
    IrisSdkError error = iris_sdk_apply_beauty_filter(
        frame.data(),
        static_cast<int>(width),
        static_cast<int>(height),
        static_cast<IrisFrameFormat>(format));

    if (error != IRIS_SDK_OK) {
        LOGW("Apply beauty filter failed: %d (%s)", error, iris_sdk_error_to_string(error));
    }

    return static_cast<jint>(error);
}

// ============================================================================
// NV21 → RGBA 고속 변환 API
// ============================================================================

/**
 * @brief NV21 데이터를 RGBA Bitmap으로 고속 변환
 *
 * OpenCV를 사용한 직접 색공간 변환으로 Java JPEG 방식 대비 10배 이상 빠름.
 * - Java (YuvImage → JPEG → Bitmap): 50-100ms
 * - JNI (OpenCV cvtColor): 5-10ms
 *
 * Java: native int nativeNv21ToRgba(byte[] nv21Data, int width, int height, Bitmap bitmap);
 *
 * @param nv21Data NV21 포맷 바이트 배열
 * @param width 프레임 너비
 * @param height 프레임 높이
 * @param bitmap 출력 Bitmap (ARGB_8888, 크기는 width x height)
 * @return 에러 코드 (0 = 성공)
 */
JNIEXPORT jint JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeNv21ToRgba(
    JNIEnv* env,
    jclass /* clazz */,
    jbyteArray nv21Data,
    jint width,
    jint height,
    jobject bitmap) {

    LOGV("nativeNv21ToRgba called: %dx%d", width, height);

    // 파라미터 검증
    if (!nv21Data) {
        LOGE("nativeNv21ToRgba: nv21Data is null");
        return static_cast<jint>(IRIS_SDK_NULL_POINTER);
    }
    if (!bitmap) {
        LOGE("nativeNv21ToRgba: bitmap is null");
        return static_cast<jint>(IRIS_SDK_NULL_POINTER);
    }
    if (width <= 0 || height <= 0) {
        LOGE("nativeNv21ToRgba: invalid dimensions %dx%d", width, height);
        return static_cast<jint>(IRIS_SDK_INVALID_PARAM);
    }

    // RAII로 바이트 배열 접근 (읽기 전용)
    ScopedByteArray nv21(env, nv21Data, JNI_ABORT);
    if (!nv21.valid()) {
        LOGE("nativeNv21ToRgba: failed to get nv21 data");
        return static_cast<jint>(IRIS_SDK_INVALID_PARAM);
    }

    // NV21 버퍼 크기 검증 (Y + UV = width * height * 1.5)
    jsize expectedSize = width * height * 3 / 2;
    if (nv21.size() < expectedSize) {
        LOGE("nativeNv21ToRgba: buffer size mismatch. Expected %d, got %d",
             expectedSize, nv21.size());
        return static_cast<jint>(IRIS_SDK_INVALID_PARAM);
    }

    // Bitmap 픽셀 버퍼 잠금
    AndroidBitmapInfo bitmapInfo;
    if (AndroidBitmap_getInfo(env, bitmap, &bitmapInfo) != ANDROID_BITMAP_RESULT_SUCCESS) {
        LOGE("nativeNv21ToRgba: failed to get bitmap info");
        return static_cast<jint>(IRIS_SDK_INVALID_PARAM);
    }

    // Bitmap 포맷 및 크기 검증
    if (bitmapInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        LOGE("nativeNv21ToRgba: bitmap format must be ARGB_8888, got %d", bitmapInfo.format);
        return static_cast<jint>(IRIS_SDK_INVALID_FORMAT);
    }
    if (bitmapInfo.width != static_cast<uint32_t>(width) ||
        bitmapInfo.height != static_cast<uint32_t>(height)) {
        LOGE("nativeNv21ToRgba: bitmap size mismatch. Expected %dx%d, got %dx%d",
             width, height, bitmapInfo.width, bitmapInfo.height);
        return static_cast<jint>(IRIS_SDK_INVALID_PARAM);
    }

    void* bitmapPixels = nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &bitmapPixels) != ANDROID_BITMAP_RESULT_SUCCESS) {
        LOGE("nativeNv21ToRgba: failed to lock bitmap pixels");
        return static_cast<jint>(IRIS_SDK_UNKNOWN);
    }

    // OpenCV Mat으로 래핑 (복사 없음)
    cv::Mat nv21Mat(height + height / 2, width, CV_8UC1, const_cast<uint8_t*>(nv21.data()));
    cv::Mat rgbaMat(height, width, CV_8UC4, bitmapPixels, bitmapInfo.stride);

    // NV21 → RGBA 변환 (OpenCV 고속 변환)
    cv::cvtColor(nv21Mat, rgbaMat, cv::COLOR_YUV2RGBA_NV21);

    // Bitmap 픽셀 버퍼 해제
    AndroidBitmap_unlockPixels(env, bitmap);

    LOGV("nativeNv21ToRgba completed successfully");
    return static_cast<jint>(IRIS_SDK_OK);
}

// ============================================================================
// Beauty Filter V2 API
// ============================================================================

/**
 * @brief 기본 V2 뷰티 필터 설정 가져오기
 *
 * 기본값으로 초기화된 BeautyFilterConfigV2를 Java 객체에 복사합니다.
 *
 * Java: native void nativeDefaultBeautyConfigV2(BeautyFilterConfigV2 config);
 */
JNIEXPORT void JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeDefaultBeautyConfigV2(
    JNIEnv* env,
    jclass /* clazz */,
    jobject configObj) {

    LOGD("nativeDefaultBeautyConfigV2 called");

    if (!configObj) {
        LOGE("nativeDefaultBeautyConfigV2: configObj is null");
        return;
    }

    // C API로 기본 설정 가져오기
    IrisBeautyConfigV2 nativeConfig = {};
    iris_sdk_default_beauty_config_v2_c(&nativeConfig);

    // Java 객체로 복사
    if (!copyBeautyConfigV2ToJava(env, nativeConfig, configObj)) {
        LOGE("Failed to copy default beauty config V2 to Java object");
    }
}

/**
 * @brief GPU 뷰티 백엔드 초기화
 *
 * Java: native int nativeInitGpuBeauty();
 */
JNIEXPORT jint JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeInitGpuBeauty(
    JNIEnv* /* env */,
    jclass /* clazz */) {

    LOGD("nativeInitGpuBeauty called");

    IrisSdkError result = iris_sdk_init_gpu_beauty();

    if (result == IRIS_SDK_OK) {
        LOGI("GPU beauty backend initialized successfully");
    } else {
        LOGW("GPU beauty backend initialization failed: %d (%s)",
             result, iris_sdk_error_to_string(result));
    }

    return static_cast<jint>(result);
}

/**
 * @brief GPU 뷰티 백엔드 해제
 *
 * Java: native void nativeReleaseGpuBeauty();
 */
JNIEXPORT void JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeReleaseGpuBeauty(
    JNIEnv* /* env */,
    jclass /* clazz */) {

    LOGD("nativeReleaseGpuBeauty called");
    iris_sdk_release_gpu_beauty();
    LOGI("GPU beauty backend released");
}

/**
 * @brief GPU 뷰티 백엔드 초기화 여부 확인
 *
 * Java: native boolean nativeIsGpuBeautyInitialized();
 */
JNIEXPORT jboolean JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeIsGpuBeautyInitialized(
    JNIEnv* /* env */,
    jclass /* clazz */) {

    int initialized = iris_sdk_is_gpu_beauty_initialized();
    LOGV("nativeIsGpuBeautyInitialized: %s", initialized ? "true" : "false");
    return initialized ? JNI_TRUE : JNI_FALSE;
}

/**
 * @brief V2 뷰티 필터 적용 (CPU 버퍼)
 *
 * Java: native int nativeApplyBeautyV2(byte[] frameData, int width, int height,
 *                                       int format, BeautyFilterConfigV2 config,
 *                                       long detectionPtr);
 */
JNIEXPORT jint JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeApplyBeautyV2(
    JNIEnv* env,
    jclass /* clazz */,
    jbyteArray frameData,
    jint width,
    jint height,
    jint format,
    jobject configObj,
    jlong detectionPtr) {

    LOGV("nativeApplyBeautyV2 called: %dx%d, format=%d", width, height, format);

    // 파라미터 검증
    if (!frameData) {
        LOGE("nativeApplyBeautyV2: frameData is null");
        return static_cast<jint>(IRIS_SDK_NULL_POINTER);
    }
    if (!configObj) {
        LOGE("nativeApplyBeautyV2: configObj is null");
        return static_cast<jint>(IRIS_SDK_NULL_POINTER);
    }
    if (width <= 0 || height <= 0) {
        LOGE("nativeApplyBeautyV2: invalid dimensions %dx%d", width, height);
        return static_cast<jint>(IRIS_SDK_INVALID_PARAM);
    }

    // RAII로 바이트 배열 접근 (쓰기 가능)
    ScopedByteArray frame(env, frameData, 0);  // mode=0: 변경사항 복사
    if (!frame.valid()) {
        LOGE("nativeApplyBeautyV2: failed to get frame data");
        return static_cast<jint>(IRIS_SDK_INVALID_PARAM);
    }

    // 버퍼 크기 검증
    if (!validateFrameBufferSize(frame.size(), width, height, format)) {
        LOGE("nativeApplyBeautyV2: frame buffer size mismatch");
        return static_cast<jint>(IRIS_SDK_INVALID_PARAM);
    }

    // Java 객체에서 설정 복사
    IrisBeautyConfigV2 nativeConfig = {};
    if (!copyBeautyConfigV2FromJava(env, configObj, nativeConfig)) {
        LOGE("Failed to copy beauty config V2 from Java object");
        return static_cast<jint>(IRIS_SDK_INVALID_PARAM);
    }

    // 검출 결과 포인터 변환
    const IrisResult* detection = reinterpret_cast<const IrisResult*>(detectionPtr);

    // C API 호출
    IrisSdkError error = iris_sdk_apply_beauty_v2_c(
        frame.data(),
        static_cast<int>(width),
        static_cast<int>(height),
        static_cast<IrisFrameFormat>(format),
        &nativeConfig,
        detection
    );

    if (error != IRIS_SDK_OK) {
        LOGW("Apply beauty V2 failed: %d (%s)", error, iris_sdk_error_to_string(error));
    }

    return static_cast<jint>(error);
}

/**
 * @brief V2 뷰티 필터 적용 (GPU 텍스처)
 *
 * Java: native int nativeApplyBeautyTextureV2(int inputTexture, int width, int height,
 *                                              BeautyFilterConfigV2 config, long detectionPtr,
 *                                              int lutTextureId, float lutIntensity);
 *
 * @return 출력 텍스처 ID (0이면 실패)
 */
JNIEXPORT jint JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeApplyBeautyTextureV2(
    JNIEnv* env,
    jclass /* clazz */,
    jint inputTexture,
    jint width,
    jint height,
    jobject configObj,
    jlong detectionPtr,
    jint lutTextureId,
    jfloat lutIntensity) {

    LOGV("nativeApplyBeautyTextureV2 called: texture=%d, %dx%d, lut=%d, lutIntensity=%.2f",
         inputTexture, width, height, lutTextureId, lutIntensity);

    if (!configObj) {
        LOGE("nativeApplyBeautyTextureV2: configObj is null");
        return 0;
    }
    if (width <= 0 || height <= 0) {
        LOGE("nativeApplyBeautyTextureV2: invalid dimensions %dx%d", width, height);
        return 0;
    }

    // Java 객체에서 설정 복사
    IrisBeautyConfigV2 nativeConfig = {};
    if (!copyBeautyConfigV2FromJava(env, configObj, nativeConfig)) {
        LOGE("Failed to copy beauty config V2 from Java object");
        return 0;
    }

    // 검출 결과 포인터 변환
    const IrisResult* detection = reinterpret_cast<const IrisResult*>(detectionPtr);

    // C API 호출
    uint32_t outputTexture = 0;
    IrisSdkError error = iris_sdk_apply_beauty_texture_v2(
        static_cast<uint32_t>(inputTexture),
        &outputTexture,
        static_cast<int>(width),
        static_cast<int>(height),
        &nativeConfig,
        detection,
        static_cast<uint32_t>(lutTextureId),
        static_cast<float>(lutIntensity)
    );

    if (error != IRIS_SDK_OK) {
        LOGW("Apply beauty texture V2 failed: %d (%s)", error, iris_sdk_error_to_string(error));
        return static_cast<jint>(inputTexture);  // 실패 시 입력 텍스처 반환
    }

    return static_cast<jint>(outputTexture);
}

JNIEXPORT void JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeSetFreqSepDebugMode(
    JNIEnv* /* env */,
    jclass /* clazz */,
    jint mode) {
    iris_sdk_set_freqsep_debug_mode(static_cast<int>(mode));
}

JNIEXPORT void JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeSetSkinColorFilter(
    JNIEnv* /* env */,
    jclass /* clazz */,
    jint enabled) {
    iris_sdk_set_skin_color_filter(static_cast<int>(enabled));
}

/**
 * @brief Face Warp 적용 (GPU)
 *
 * Java: native int nativeApplyFaceWarp(int inputTexture, int width, int height,
 *                                       float slimFace, float thinChin, float enlargeEyes,
 *                                       long detectionPtr);
 *
 * @return 출력 텍스처 ID
 */
JNIEXPORT jint JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeApplyFaceWarp(
    JNIEnv* /* env */,
    jclass /* clazz */,
    jint inputTexture,
    jint width,
    jint height,
    jfloat slimFace,
    jfloat thinChin,
    jfloat enlargeEyes,
    jlong detectionPtr) {

    LOGV("nativeApplyFaceWarp called: texture=%d, %dx%d, slim=%.2f, chin=%.2f, eyes=%.2f",
         inputTexture, width, height, slimFace, thinChin, enlargeEyes);

    // 검출 결과 포인터 변환
    const IrisResult* detection = reinterpret_cast<const IrisResult*>(detectionPtr);

    // C API 호출
    uint32_t outputTexture = 0;
    IrisSdkError error = iris_sdk_apply_face_warp(
        static_cast<uint32_t>(inputTexture),
        &outputTexture,
        static_cast<int>(width),
        static_cast<int>(height),
        slimFace,
        thinChin,
        enlargeEyes,
        detection
    );

    if (error != IRIS_SDK_OK) {
        LOGW("Apply face warp failed: %d (%s)", error, iris_sdk_error_to_string(error));
        return static_cast<jint>(inputTexture);
    }

    return static_cast<jint>(outputTexture);
}

/**
 * @brief SDK 관리 텍스처 해제
 *
 * Java: native void nativeReleaseTexture(int texture);
 */
JNIEXPORT void JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeReleaseTexture(
    JNIEnv* /* env */,
    jclass /* clazz */,
    jint texture) {

    LOGV("nativeReleaseTexture called: texture=%d", texture);

    IrisSdkError error = iris_sdk_release_texture(static_cast<uint32_t>(texture));

    if (error != IRIS_SDK_OK) {
        LOGV("Release texture result: %d (may not be SDK-managed)", error);
    }
}

/**
 * @brief 텍스처가 SDK 관리인지 확인
 *
 * Java: native boolean nativeIsTextureManaged(int texture);
 */
JNIEXPORT jboolean JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeIsTextureManaged(
    JNIEnv* /* env */,
    jclass /* clazz */,
    jint texture) {

    int managed = iris_sdk_is_texture_managed(static_cast<uint32_t>(texture));
    return managed ? JNI_TRUE : JNI_FALSE;
}

// ============================================================================
// Detection Slot JNI (더블 버퍼 — lock-free IrisResult 전달)
// ============================================================================

/**
 * @brief Detection 슬롯에 IrisResult 업데이트 (Analyzer 스레드에서 호출)
 *
 * 비활성 슬롯에 덮어쓰기 후 atomic swap으로 활성 슬롯 전환.
 * Lock-free, wait-free writer.
 *
 * Java: native void nativeUpdateDetectionSlot(IrisResult result);
 */
JNIEXPORT void JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeUpdateDetectionSlot(
    JNIEnv* env,
    jclass /* clazz */,
    jobject resultObj) {

    if (!resultObj) {
        LOGW("nativeUpdateDetectionSlot: resultObj is null");
        return;
    }

    // 비활성 슬롯 결정
    int current_active = g_active_slot_index.load(std::memory_order_acquire);
    int write_idx = (current_active == 0) ? 1 : 0;

    // 비활성 슬롯에 데이터 복사
    if (!iris::jni::copyResultFromJava(env, resultObj, g_detection_slots[write_idx].data)) {
        LOGE("nativeUpdateDetectionSlot: failed to copy result from Java");
        return;
    }

    // generation 증가 → valid 설정 → active swap (release ordering)
    g_detection_slots[write_idx].generation.fetch_add(1, std::memory_order_release);
    g_detection_slots[write_idx].valid.store(true, std::memory_order_release);
    g_active_slot_index.store(write_idx, std::memory_order_release);
}

/**
 * @brief 현재 활성 Detection 슬롯의 네이티브 포인터 반환 (GL 스레드에서 호출)
 *
 * 반환된 포인터는 applyBeautyFilterTextureV2()의 detectionPtr로 사용.
 * Lock-free, wait-free reader. Generation 검증은 호출측에서 수행.
 *
 * Java: native long nativeGetDetectionSlotPtr();
 * @return 활성 슬롯의 IrisResult 포인터 (jlong), 유효하지 않으면 0L
 */
JNIEXPORT jlong JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeGetDetectionSlotPtr(
    JNIEnv* /* env */,
    jclass /* clazz */) {

    int read_idx = g_active_slot_index.load(std::memory_order_acquire);

    // 초기 미설정 상태
    if (read_idx < 0 || read_idx > 1) {
        return 0L;
    }

    // valid 확인
    if (!g_detection_slots[read_idx].valid.load(std::memory_order_acquire)) {
        return 0L;
    }

    return reinterpret_cast<jlong>(&g_detection_slots[read_idx].data);
}

/**
 * @brief Detection 슬롯 해제 (SDK 종료 시 1회 호출)
 *
 * Java: native void nativeReleaseDetectionSlot();
 */
JNIEXPORT void JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeReleaseDetectionSlot(
    JNIEnv* /* env */,
    jclass /* clazz */) {

    // 슬롯 초기화 (메모리 해제 불필요 — 스택/전역 할당)
    g_detection_slots[0].valid.store(false, std::memory_order_release);
    g_detection_slots[0].generation.store(0, std::memory_order_release);
    g_detection_slots[1].valid.store(false, std::memory_order_release);
    g_detection_slots[1].generation.store(0, std::memory_order_release);
    g_active_slot_index.store(-1, std::memory_order_release);

    LOGI("Detection slots released");
}

// ============================================================================
// Temporal Stabilizer API (P5-W1)
// ============================================================================

/**
 * @brief Temporal Stabilizer 생성
 *
 * Java: native long nativeCreateStabilizer();
 */
JNIEXPORT jlong JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeCreateStabilizer(
    JNIEnv* /* env */,
    jclass /* clazz */) {

    // 기본 설정으로 Stabilizer 생성
    int64_t handle = iris_sdk_create_stabilizer(nullptr);
    if (handle == 0) {
        LOGE("Failed to create stabilizer");
    } else {
        LOGI("Stabilizer created: handle=%lld", static_cast<long long>(handle));
    }
    return static_cast<jlong>(handle);
}

/**
 * @brief 검출 결과 스무딩 (Java IrisResult를 in-place로 수정)
 *
 * Java: native float nativeStabilize(long handle, IrisResult result, double timestampSec);
 *
 * @return visibility (0.0~1.0)
 */
JNIEXPORT jfloat JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeStabilize(
    JNIEnv* env,
    jclass /* clazz */,
    jlong handle,
    jobject resultObj,
    jdouble timestampSec) {

    using namespace iris::jni;

    if (handle == 0 || !resultObj) {
        LOGE("nativeStabilize: invalid args (handle=%lld, resultObj=%p)",
             static_cast<long long>(handle), resultObj);
        return 0.0f;
    }

    // 1. Java IrisResult → C IrisResult
    IrisResult raw = {};
    if (!copyResultFromJava(env, resultObj, raw)) {
        LOGE("nativeStabilize: failed to read IrisResult from Java");
        return 0.0f;
    }

    // 2. C API 스무딩
    IrisStabilizedResult stabilized = {};
    IrisSdkError error = iris_sdk_stabilize(
        static_cast<int64_t>(handle),
        &raw,
        static_cast<double>(timestampSec),
        &stabilized);

    if (error != IRIS_SDK_OK) {
        LOGW("iris_sdk_stabilize failed: %d (%s)", error, iris_sdk_error_to_string(error));
        return 0.0f;
    }

    // 3. 스무딩된 결과를 Java 객체에 다시 씀 (in-place 수정)
    if (!copyResultToJava(env, stabilized.stabilized, resultObj)) {
        LOGE("nativeStabilize: failed to write stabilized result to Java");
        return 0.0f;
    }

    LOGV("Stabilize: visibility=%.2f, held=%d", stabilized.visibility, stabilized.is_held);
    return static_cast<jfloat>(stabilized.visibility);
}

/**
 * @brief Temporal Stabilizer 해제
 *
 * Java: native void nativeDestroyStabilizer(long handle);
 */
JNIEXPORT void JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeDestroyStabilizer(
    JNIEnv* /* env */,
    jclass /* clazz */,
    jlong handle) {

    if (handle != 0) {
        iris_sdk_destroy_stabilizer(static_cast<int64_t>(handle));
        LOGI("Stabilizer destroyed: handle=%lld", static_cast<long long>(handle));
    }
}

// ============================================================================
// GPU 렌즈 렌더링 JNI
// ============================================================================

/**
 * Java: native int nativeInitGpuLens();
 */
JNIEXPORT jint JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeInitGpuLens(
    JNIEnv* /* env */, jclass /* clazz */)
{
    LOGD("nativeInitGpuLens called");
    IrisSdkError result = iris_sdk_init_gpu_lens();
    if (result == IRIS_SDK_OK) {
        LOGI("GPU lens renderer initialized successfully");
    } else {
        LOGW("GPU lens renderer initialization failed: %d", result);
    }
    return static_cast<jint>(result);
}

/**
 * Java: native void nativeReleaseGpuLens();
 */
JNIEXPORT void JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeReleaseGpuLens(
    JNIEnv* /* env */, jclass /* clazz */)
{
    LOGD("nativeReleaseGpuLens called");
    iris_sdk_release_gpu_lens();
    LOGI("GPU lens renderer released");
}

/**
 * Java: native boolean nativeIsGpuLensInitialized();
 */
JNIEXPORT jboolean JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeIsGpuLensInitialized(
    JNIEnv* /* env */, jclass /* clazz */)
{
    return iris_sdk_is_gpu_lens_initialized() ? JNI_TRUE : JNI_FALSE;
}

/**
 * Java: native int nativeLoadLensTexture(byte[] data, int width, int height);
 */
JNIEXPORT jint JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeLoadLensTexture(
    JNIEnv* env, jclass /* clazz */,
    jbyteArray data, jint width, jint height)
{
    if (!data) {
        LOGE("nativeLoadLensTexture: data is null");
        return static_cast<jint>(IRIS_SDK_NULL_POINTER);
    }

    ScopedByteArray arr(env, data, JNI_ABORT);
    if (!arr.valid()) {
        LOGE("nativeLoadLensTexture: failed to get byte array");
        return static_cast<jint>(IRIS_SDK_NULL_POINTER);
    }

    return static_cast<jint>(iris_sdk_load_lens_texture(arr.data(), width, height));
}

/**
 * Java: native void nativeUnloadLensTexture();
 */
JNIEXPORT void JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeUnloadLensTexture(
    JNIEnv* /* env */, jclass /* clazz */)
{
    iris_sdk_unload_lens_texture();
}

/**
 * Java: native int nativeRenderLensTexture(int inputTexture, int width, int height,
 *                                           long detectionPtr, LensConfig config);
 * @return 출력 텍스처 ID (0이면 실패)
 */
JNIEXPORT jint JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeRenderLensTexture(
    JNIEnv* env, jclass /* clazz */,
    jint inputTexture, jint width, jint height,
    jlong detectionPtr, jobject configObj)
{
    LOGV("nativeRenderLensTexture: texture=%d, %dx%d", inputTexture, width, height);

    if (width <= 0 || height <= 0) {
        LOGE("nativeRenderLensTexture: invalid dimensions %dx%d", width, height);
        return 0;
    }

    // 검출 결과 (포인터)
    const IrisResult* detection = nullptr;
    if (detectionPtr != 0) {
        detection = reinterpret_cast<const IrisResult*>(detectionPtr);
    }

    // LensConfig 변환
    IrisLensConfig config = {};
    if (configObj) {
        if (!copyConfigFromJava(env, configObj, config)) {
            LOGE("nativeRenderLensTexture: failed to copy LensConfig from Java");
            return 0;
        }
    } else {
        // 기본값
        config.opacity = 0.7f;
        config.scale = 1.0f;
        config.edge_feather = 0.1f;
        config.apply_left = true;
        config.apply_right = true;
        config.blend_mode = IRIS_BLEND_LUMINANCE_TINT_LINEAR;  // P6-W2 §5.12 canonical default
    }

    uint32_t output_texture = 0;
    IrisSdkError err = iris_sdk_render_lens_texture(
        static_cast<uint32_t>(inputTexture),
        &output_texture,
        width, height,
        detection,
        &config);

    if (err != IRIS_SDK_OK) {
        LOGW("nativeRenderLensTexture failed: %d", err);
        return 0;
    }

    return static_cast<jint>(output_texture);
}

/**
 * Java: native void nativeSetLensScleraProtect(boolean enabled);
 */
JNIEXPORT void JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeSetLensScleraProtect(
    JNIEnv* /* env */, jclass /* clazz */,
    jboolean enabled)
{
    iris_sdk_set_lens_sclera_protect(enabled ? 1 : 0);
}

/**
 * Java: native void nativeSetLensEllipseMask(boolean enabled);
 */
JNIEXPORT void JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeSetLensEllipseMask(
    JNIEnv* /* env */, jclass /* clazz */,
    jboolean enabled)
{
    iris_sdk_set_lens_ellipse_mask(enabled ? 1 : 0);
}

/**
 * Java: native void nativeSetLensHighlight(boolean enabled);
 */
JNIEXPORT void JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeSetLensHighlight(
    JNIEnv* /* env */, jclass /* clazz */,
    jboolean enabled)
{
    iris_sdk_set_lens_highlight(enabled ? 1 : 0);
}

// ============================================================================
// P6-W4 §5.7/§5.11: 환경 반사 internal C API forward declare.
// sdk_api_v2.cpp에 정의됨. 공개 sdk_api.h 미노출 (W4 Phase 벤치 internal 경로).
// ============================================================================
extern IrisSdkError iris_sdk_load_env_map(const uint8_t* data, int width, int height);
extern void iris_sdk_unload_env_map(void);
extern void iris_sdk_set_reflection_mode(int mode);
extern void iris_sdk_set_reflection_intensity(float intensity);

/**
 * Java: native int nativeLoadEnvMap(byte[] data, int width, int height);
 * P6-W4 §5.7: env_map 텍스처 로드 (Android assets `env/`에서 byte 받음).
 */
JNIEXPORT jint JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeLoadEnvMap(
    JNIEnv* env, jclass /* clazz */,
    jbyteArray data, jint width, jint height)
{
    if (!data) {
        LOGE("nativeLoadEnvMap: data is null");
        return static_cast<jint>(IRIS_SDK_NULL_POINTER);
    }

    ScopedByteArray arr(env, data, JNI_ABORT);
    if (!arr.valid()) {
        LOGE("nativeLoadEnvMap: failed to get byte array");
        return static_cast<jint>(IRIS_SDK_NULL_POINTER);
    }

    return static_cast<jint>(iris_sdk_load_env_map(arr.data(), width, height));
}

/**
 * Java: native void nativeUnloadEnvMap();
 */
JNIEXPORT void JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeUnloadEnvMap(
    JNIEnv* /* env */, jclass /* clazz */)
{
    iris_sdk_unload_env_map();
}

/**
 * Java: native void nativeSetReflectionMode(int mode);
 * P6-W4 §5.11: mode 0=OFF, 1=EnvMap, 2=Periphery.
 */
JNIEXPORT void JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeSetReflectionMode(
    JNIEnv* /* env */, jclass /* clazz */,
    jint mode)
{
    iris_sdk_set_reflection_mode(static_cast<int>(mode));
}

/**
 * Java: native void nativeSetReflectionIntensity(float intensity);
 * P6-W4 §5.7: intensity [0.0, 1.0] (W3 기본 0.3).
 */
JNIEXPORT void JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeSetReflectionIntensity(
    JNIEnv* /* env */, jclass /* clazz */,
    jfloat intensity)
{
    iris_sdk_set_reflection_intensity(static_cast<float>(intensity));
}

}  // extern "C"
