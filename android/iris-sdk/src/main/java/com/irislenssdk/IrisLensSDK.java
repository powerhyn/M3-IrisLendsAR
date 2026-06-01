/*
 * Copyright 2024 IrisLensSDK Team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.irislenssdk;

import android.content.Context;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * IrisLensSDK 메인 클래스.
 *
 * <p>실시간 카메라 영상에서 홍채를 추적하고 가상 렌즈를 오버레이하는
 * AR 피팅 SDK의 Android 바인딩입니다.</p>
 *
 * <h3>기본 사용법:</h3>
 * <pre>{@code
 * // 초기화
 * int error = IrisLensSDK.init(context);
 * if (error != IrisLensSDK.OK) {
 *     Log.e("SDK", "Init failed: " + IrisLensSDK.errorToString(error));
 *     return;
 * }
 *
 * // 텍스처 로드
 * IrisLensSDK.loadTexture(texturePath);
 *
 * // 프레임 처리
 * IrisResult result = new IrisResult();
 * LensConfig config = new LensConfig();
 * error = IrisLensSDK.process(frameData, width, height, FORMAT_NV21, config, result);
 *
 * // 종료
 * IrisLensSDK.destroy();
 * }</pre>
 *
 * @author IrisLensSDK Team
 * @version 1.0.0
 */
public final class IrisLensSDK {

    private static final String TAG = "IrisLensSDK";

    // ========================================================================
    // 에러 코드 상수
    // ========================================================================

    /** 성공 */
    public static final int OK = 0;

    /* 초기화 에러 (100-199) */
    /** SDK가 초기화되지 않음 */
    public static final int NOT_INITIALIZED = 100;
    /** SDK가 이미 초기화됨 */
    public static final int ALREADY_INITIALIZED = 101;
    /** 모델 로드 실패 */
    public static final int MODEL_LOAD_FAILED = 102;
    /** 잘못된 경로 */
    public static final int INVALID_PATH = 103;

    /* 파라미터 에러 (200-299) */
    /** 잘못된 파라미터 */
    public static final int INVALID_PARAM = 200;
    /** 널 포인터 */
    public static final int NULL_POINTER = 201;
    /** 지원하지 않는 포맷 */
    public static final int INVALID_FORMAT = 202;

    /* 검출 에러 (300-399) */
    /** 검출 실패 */
    public static final int DETECTION_FAILED = 300;
    /** 얼굴 미검출 */
    public static final int NO_FACE = 301;

    /* 렌더링 에러 (400-499) */
    /** 렌더링 실패 */
    public static final int RENDER_FAILED = 400;
    /** 텍스처 미로드 */
    public static final int NO_TEXTURE = 401;

    /** 알 수 없는 에러 */
    public static final int UNKNOWN = 999;

    // ========================================================================
    // 프레임 포맷 상수
    // ========================================================================

    /** 32비트 RGBA */
    public static final int FORMAT_RGBA = 0;
    /** 32비트 BGRA */
    public static final int FORMAT_BGRA = 1;
    /** 24비트 RGB */
    public static final int FORMAT_RGB = 2;
    /** 24비트 BGR */
    public static final int FORMAT_BGR = 3;
    /** Android 카메라 NV21 (YUV420sp) */
    public static final int FORMAT_NV21 = 4;
    /** iOS 카메라 NV12 (YUV420sp) */
    public static final int FORMAT_NV12 = 5;
    /** 8비트 그레이스케일 */
    public static final int FORMAT_GRAY = 6;

    // ========================================================================
    // 네이티브 라이브러리 로딩
    // ========================================================================

    private static boolean sLibraryLoaded = false;
    private static String sModelPath = null;

    static {
        try {
            System.loadLibrary("iris_jni");
            sLibraryLoaded = true;
            Log.i(TAG, "Native library loaded successfully");
        } catch (UnsatisfiedLinkError e) {
            sLibraryLoaded = false;
            Log.e(TAG, "Failed to load native library: " + e.getMessage());
        }
    }

    // ========================================================================
    // 생성자 (인스턴스화 금지)
    // ========================================================================

    private IrisLensSDK() {
        throw new AssertionError("No instances");
    }

    // ========================================================================
    // 초기화 API
    // ========================================================================

    /**
     * SDK를 초기화합니다.
     *
     * <p>Context를 사용하여 assets에서 모델 파일을 추출하고 초기화합니다.</p>
     *
     * @param context Android Context (Application 또는 Activity)
     * @return 에러 코드 (OK = 성공)
     */
    public static int init(@NonNull Context context) {
        if (!sLibraryLoaded) {
            Log.e(TAG, "Native library not loaded");
            return NOT_INITIALIZED;
        }

        try {
            // assets에서 모델 파일 추출
            String modelDir = extractModelsFromAssets(context);
            if (modelDir == null) {
                Log.e(TAG, "Failed to extract models from assets");
                return MODEL_LOAD_FAILED;
            }

            sModelPath = modelDir;
            return nativeInit(modelDir);

        } catch (Exception e) {
            Log.e(TAG, "Init failed: " + e.getMessage(), e);
            return UNKNOWN;
        }
    }

    /**
     * 지정된 모델 경로로 SDK를 초기화합니다.
     *
     * @param modelPath 모델 파일이 있는 디렉토리 경로
     * @return 에러 코드 (OK = 성공)
     */
    public static int init(@NonNull String modelPath) {
        if (!sLibraryLoaded) {
            Log.e(TAG, "Native library not loaded");
            return NOT_INITIALIZED;
        }

        sModelPath = modelPath;
        return nativeInit(modelPath);
    }

    /**
     * SDK를 종료하고 리소스를 해제합니다.
     *
     * <p>이 메서드는 여러 번 호출해도 안전합니다.</p>
     */
    public static void destroy() {
        if (sLibraryLoaded) {
            nativeDestroy();
            sModelPath = null;
        }
    }

    /**
     * SDK가 사용 가능한 상태인지 확인합니다.
     *
     * @return 사용 가능하면 true
     */
    public static boolean isReady() {
        return sLibraryLoaded && nativeIsReady();
    }

    /**
     * 네이티브 라이브러리가 로드되었는지 확인합니다.
     *
     * @return 로드되었으면 true
     */
    public static boolean isLibraryLoaded() {
        return sLibraryLoaded;
    }

    // ========================================================================
    // 텍스처 API
    // ========================================================================

    /**
     * 파일에서 렌즈 텍스처를 로드합니다.
     *
     * @param path 텍스처 이미지 파일 경로 (PNG, JPEG, BMP)
     * @return 에러 코드 (OK = 성공)
     */
    public static int loadTexture(@NonNull String path) {
        if (!sLibraryLoaded) {
            return NOT_INITIALIZED;
        }
        return nativeLoadTexture(path);
    }

    /**
     * assets에서 렌즈 텍스처를 로드합니다.
     *
     * @param context Android Context
     * @param assetPath assets 내 상대 경로
     * @return 에러 코드 (OK = 성공)
     */
    public static int loadTextureFromAssets(@NonNull Context context, @NonNull String assetPath) {
        if (!sLibraryLoaded) {
            return NOT_INITIALIZED;
        }

        try {
            // assets에서 캐시 디렉토리로 복사
            File cacheDir = new File(context.getCacheDir(), "iris_textures");
            if (!cacheDir.exists() && !cacheDir.mkdirs()) {
                Log.e(TAG, "Failed to create cache directory");
                return INVALID_PATH;
            }

            String fileName = new File(assetPath).getName();
            File destFile = new File(cacheDir, fileName);

            copyAssetToFile(context, assetPath, destFile);

            return nativeLoadTexture(destFile.getAbsolutePath());

        } catch (IOException e) {
            Log.e(TAG, "Failed to load texture from assets: " + e.getMessage(), e);
            return INVALID_PATH;
        }
    }

    /**
     * 메모리에서 RGBA 텍스처를 로드합니다.
     *
     * @param data RGBA 픽셀 데이터 (4바이트/픽셀)
     * @param width 텍스처 너비
     * @param height 텍스처 높이
     * @return 에러 코드 (OK = 성공)
     */
    public static int loadTextureFromMemory(@NonNull byte[] data, int width, int height) {
        if (!sLibraryLoaded) {
            return NOT_INITIALIZED;
        }
        return nativeLoadTextureFromMemory(data, width, height);
    }

    // ========================================================================
    // 검출/처리 API
    // ========================================================================

    /**
     * 프레임에서 홍채를 검출합니다.
     *
     * @param frameData 프레임 데이터 (읽기 전용)
     * @param width 프레임 너비
     * @param height 프레임 높이
     * @param format 프레임 포맷 (FORMAT_* 상수)
     * @param result 검출 결과 출력 객체
     * @return 에러 코드 (OK = 성공)
     */
    public static int detect(@NonNull byte[] frameData, int width, int height,
                             int format, @NonNull IrisResult result) {
        if (!sLibraryLoaded) {
            return NOT_INITIALIZED;
        }
        return nativeDetect(frameData, width, height, format, result);
    }

    /**
     * 회전을 고려하여 프레임에서 홍채를 검출합니다.
     *
     * <p>Android 카메라는 센서 방향에 따라 회전된 이미지를 출력합니다.
     * CameraX ImageProxy.imageInfo.rotationDegrees 값을 전달하여
     * 올바른 방향으로 검출을 수행합니다.</p>
     *
     * @param frameData 프레임 데이터 (읽기 전용)
     * @param width 프레임 너비
     * @param height 프레임 높이
     * @param format 프레임 포맷 (FORMAT_* 상수)
     * @param rotationDegrees 이미지 회전 각도 (0, 90, 180, 270)
     * @param result 검출 결과 출력 객체
     * @return 에러 코드 (OK = 성공)
     */
    public static int detectWithRotation(@NonNull byte[] frameData, int width, int height,
                                          int format, int rotationDegrees,
                                          @NonNull IrisResult result) {
        if (!sLibraryLoaded) {
            return NOT_INITIALIZED;
        }
        return nativeDetectWithRotation(frameData, width, height, format, rotationDegrees, result);
    }

    /**
     * 프레임을 처리합니다 (검출 + 렌더링).
     *
     * <p>프레임 데이터는 in-place로 수정됩니다.</p>
     *
     * @param frameData 프레임 데이터 (수정됨)
     * @param width 프레임 너비
     * @param height 프레임 높이
     * @param format 프레임 포맷 (FORMAT_* 상수)
     * @param config 렌더링 설정 (null이면 검출만 수행)
     * @param result 검출 결과 출력 객체 (null 가능)
     * @return 에러 코드 (OK = 성공)
     */
    public static int process(@NonNull byte[] frameData, int width, int height,
                              int format, @Nullable LensConfig config,
                              @Nullable IrisResult result) {
        if (!sLibraryLoaded) {
            return NOT_INITIALIZED;
        }
        return nativeProcess(frameData, width, height, format, config, result);
    }

    // ========================================================================
    // 설정 API
    // ========================================================================

    /**
     * 런타임 설정을 변경합니다.
     *
     * @param key 설정 키 (예: "min_confidence", "face_tracking")
     * @param value 설정 값
     * @return 에러 코드 (OK = 성공)
     */
    public static int setConfig(@NonNull String key, @NonNull String value) {
        if (!sLibraryLoaded) {
            return NOT_INITIALIZED;
        }
        return nativeSetConfig(key, value);
    }

    // ========================================================================
    // GPU 가속 API
    // ========================================================================

    /**
     * GPU 가속 사용 여부를 설정합니다.
     *
     * <p>반드시 init() 호출 전에 설정해야 합니다.
     * init() 이후에 호출하면 설정이 무시됩니다.</p>
     *
     * <p>GPU 가속이 가능한 조건:
     * <ul>
     *   <li>Android: OpenGL ES 3.1 이상 지원 기기</li>
     *   <li>SDK가 GPU delegate와 함께 빌드됨</li>
     * </ul></p>
     *
     * @param enable true면 GPU 가속 시도, false면 CPU만 사용
     */
    public static void setGpuEnabled(boolean enable) {
        if (!sLibraryLoaded) {
            Log.w(TAG, "setGpuEnabled: library not loaded");
            return;
        }
        nativeSetGpuEnabled(enable);
    }

    /**
     * GPU 가속 사용 가능 여부를 확인합니다.
     *
     * <p>SDK가 GPU delegate와 함께 빌드되었는지 확인합니다.
     * 컴파일 시점에 결정됩니다.</p>
     *
     * @return true면 GPU delegate 라이브러리가 포함됨, false면 CPU만 사용 가능
     */
    public static boolean isGpuAvailable() {
        if (!sLibraryLoaded) {
            return false;
        }
        return nativeIsGpuAvailable();
    }

    /**
     * 현재 GPU 사용 상태를 확인합니다.
     *
     * <p>런타임에 실제로 GPU delegate가 활성화되어 있는지 확인합니다.
     * init() 호출 후에만 정확한 값을 반환합니다.</p>
     *
     * @return true면 GPU 사용 중, false면 CPU 사용 중
     */
    public static boolean isUsingGpu() {
        if (!sLibraryLoaded) {
            return false;
        }
        return nativeIsUsingGpu();
    }

    /**
     * 얼굴 검출 최소 신뢰도를 설정합니다.
     *
     * <p>얼굴 검출 결과의 최소 신뢰도입니다.
     * 이 값 이하면 검출되지 않은 것으로 처리합니다.</p>
     *
     * @param minConfidence 최소 신뢰도 (0.0 ~ 1.0), 기본값 0.3
     */
    public static void setMinDetectionConfidence(float minConfidence) {
        if (!sLibraryLoaded) {
            Log.w(TAG, "setMinDetectionConfidence: library not loaded");
            return;
        }
        nativeSetMinDetectionConfidence(minConfidence);
    }

    /**
     * 랜드마크 추적 최소 신뢰도를 설정합니다.
     *
     * <p>랜드마크 추적 결과의 최소 신뢰도입니다.
     * 이 값 이하면 추적 실패로 간주하고 다시 Face Detection을 수행합니다.</p>
     *
     * @param minConfidence 최소 신뢰도 (0.0 ~ 1.0), 기본값 0.5
     */
    public static void setMinTrackingConfidence(float minConfidence) {
        if (!sLibraryLoaded) {
            Log.w(TAG, "setMinTrackingConfidence: library not loaded");
            return;
        }
        nativeSetMinTrackingConfidence(minConfidence);
    }

    /**
     * 얼굴 존재 최소 신뢰도를 설정합니다.
     *
     * <p>추적 모드에서 이전 프레임 결과를 재사용할지 판단하는 임계값입니다.
     * 이전 프레임의 confidence가 이 값 이상이어야 Face Detection을 스킵합니다.
     * 이 값 미만이면 캐시를 무효화하고 다시 Face Detection을 수행합니다.</p>
     *
     * @param minConfidence 최소 신뢰도 (0.0 ~ 1.0), 기본값 0.5
     */
    public static void setMinPresenceConfidence(float minConfidence) {
        if (!sLibraryLoaded) {
            Log.w(TAG, "setMinPresenceConfidence: library not loaded");
            return;
        }
        nativeSetMinPresenceConfidence(minConfidence);
    }

    /**
     * InferenceThread 사용 여부를 설정합니다. (벤치마크용)
     *
     * <p>반드시 {@link #init(Context, String)} 호출 전에 설정해야 합니다.
     * false로 설정하면 전용 스레드 없이 직접 호출합니다.
     * GPU 가속은 InferenceThread 사용 시에만 지원됩니다.</p>
     *
     * @param enable true면 InferenceThread 사용 (기본값), false면 직접 호출
     */
    public static void setUseInferenceThread(boolean enable) {
        if (!sLibraryLoaded) {
            Log.w(TAG, "setUseInferenceThread: library not loaded");
            return;
        }
        nativeSetUseInferenceThread(enable);
    }

    /**
     * InferenceThread 사용 상태를 확인합니다.
     *
     * @return true면 InferenceThread 사용, false면 직접 호출
     */
    public static boolean isUsingInferenceThread() {
        if (!sLibraryLoaded) {
            return true;  // 기본값
        }
        return nativeIsUsingInferenceThread();
    }

    // ========================================================================
    // 뷰티 필터 API
    // ========================================================================

    /**
     * 기본 뷰티 필터 설정을 가져옵니다.
     *
     * @return 기본 설정이 적용된 BeautyFilterConfig
     */
    @NonNull
    public static BeautyFilterConfig getDefaultBeautyConfig() {
        BeautyFilterConfig config = new BeautyFilterConfig();
        if (sLibraryLoaded) {
            nativeDefaultBeautyConfig(config);
        }
        return config;
    }

    /**
     * 뷰티 필터 설정을 적용합니다.
     *
     * <p>설정된 값은 이후 {@link #applyBeautyFilter} 호출 시 사용됩니다.</p>
     *
     * @param config 뷰티 필터 설정
     * @return 에러 코드 (OK = 성공)
     */
    public static int setBeautyFilter(@NonNull BeautyFilterConfig config) {
        if (!sLibraryLoaded) {
            return NOT_INITIALIZED;
        }
        return nativeSetBeautyFilter(config);
    }

    /**
     * 현재 뷰티 필터 설정을 가져옵니다.
     *
     * @param config 설정을 받을 객체
     * @return 에러 코드 (OK = 성공)
     */
    public static int getBeautyFilter(@NonNull BeautyFilterConfig config) {
        if (!sLibraryLoaded) {
            return NOT_INITIALIZED;
        }
        return nativeGetBeautyFilter(config);
    }

    /**
     * 뷰티 필터 활성화 여부를 확인합니다.
     *
     * @return true면 활성화됨
     */
    public static boolean isBeautyFilterEnabled() {
        if (!sLibraryLoaded) {
            return false;
        }
        return nativeIsBeautyFilterEnabled();
    }

    /**
     * 프레임에 뷰티 필터를 적용합니다.
     *
     * <p>프레임 데이터는 in-place로 수정됩니다.
     * 홍채 검출 후에 호출하는 것을 권장합니다 (검출 정확도 보존).</p>
     *
     * <p>사용 예:</p>
     * <pre>{@code
     * // 1. 홍채 검출 (원본 프레임)
     * IrisResult result = new IrisResult();
     * IrisLensSDK.detect(frameData, width, height, format, result);
     *
     * // 2. 뷰티 필터 적용 (프레임 수정)
     * IrisLensSDK.applyBeautyFilter(frameData, width, height, format);
     *
     * // 3. 렌즈 렌더링 (필터 적용된 프레임)
     * // ...
     * }</pre>
     *
     * @param frameData 프레임 데이터 (수정됨)
     * @param width 프레임 너비
     * @param height 프레임 높이
     * @param format 프레임 포맷 (FORMAT_* 상수)
     * @return 에러 코드 (OK = 성공)
     */
    public static int applyBeautyFilter(@NonNull byte[] frameData, int width, int height, int format) {
        if (!sLibraryLoaded) {
            return NOT_INITIALIZED;
        }
        return nativeApplyBeautyFilter(frameData, width, height, format);
    }

    /**
     * NV21 데이터를 RGBA Bitmap으로 고속 변환합니다.
     *
     * <p>OpenCV를 사용한 직접 색공간 변환으로 Java JPEG 방식 대비 10배 이상 빠릅니다.</p>
     * <ul>
     *   <li>Java (YuvImage → JPEG → Bitmap): 50-100ms</li>
     *   <li>JNI (OpenCV cvtColor): 5-10ms</li>
     * </ul>
     *
     * <p>뷰티 필터 적용된 프레임을 화면에 표시할 때 사용합니다.</p>
     *
     * @param nv21Data NV21 포맷 바이트 배열
     * @param width 프레임 너비
     * @param height 프레임 높이
     * @param bitmap 출력 Bitmap (ARGB_8888, 크기는 width x height)
     * @return 에러 코드 (OK = 성공)
     */
    public static int nv21ToRgba(@NonNull byte[] nv21Data, int width, int height,
                                  @NonNull android.graphics.Bitmap bitmap) {
        if (!sLibraryLoaded) {
            return NOT_INITIALIZED;
        }
        return nativeNv21ToRgba(nv21Data, width, height, bitmap);
    }

    // ========================================================================
    // 뷰티 필터 V2 API
    // ========================================================================

    /**
     * GPU 뷰티 백엔드를 초기화합니다.
     *
     * <p>GPU 가속 뷰티 필터를 사용하려면 먼저 이 메서드를 호출해야 합니다.
     * OpenGL ES 컨텍스트가 현재 스레드에 바인딩되어 있어야 합니다.</p>
     *
     * @return 에러 코드 (OK = 성공, ERROR_NOT_SUPPORTED = GPU 미지원)
     */
    public static int initGpuBeauty() {
        if (!sLibraryLoaded) {
            return NOT_INITIALIZED;
        }
        return nativeInitGpuBeauty();
    }

    /**
     * GPU 뷰티 백엔드를 해제합니다.
     *
     * <p>앱 종료 시 또는 GPU 리소스 정리가 필요할 때 호출합니다.</p>
     */
    public static void releaseGpuBeauty() {
        if (sLibraryLoaded) {
            nativeReleaseGpuBeauty();
        }
    }

    /**
     * GPU 뷰티 백엔드 초기화 여부를 확인합니다.
     *
     * @return true면 초기화됨
     */
    public static boolean isGpuBeautyInitialized() {
        if (!sLibraryLoaded) {
            return false;
        }
        return nativeIsGpuBeautyInitialized();
    }

    /**
     * 기본 V2 뷰티 필터 설정을 가져옵니다.
     *
     * @return 기본 설정이 적용된 BeautyFilterConfigV2
     */
    @NonNull
    public static BeautyFilterConfigV2 getDefaultBeautyConfigV2() {
        BeautyFilterConfigV2 config = new BeautyFilterConfigV2();
        if (sLibraryLoaded) {
            nativeDefaultBeautyConfigV2(config);
        }
        return config;
    }

    /**
     * V2 뷰티 필터를 프레임에 적용합니다 (CPU).
     *
     * <p>프레임 데이터는 in-place로 수정됩니다.
     * 얼굴 검출 결과가 있으면 ROI 기반 처리로 성능이 향상됩니다.</p>
     *
     * @param frameData 프레임 데이터 (수정됨)
     * @param width 프레임 너비
     * @param height 프레임 높이
     * @param format 프레임 포맷 (FORMAT_* 상수)
     * @param config V2 뷰티 필터 설정
     * @param result 얼굴 검출 결과 (null 가능, null이면 전체 프레임 처리)
     * @return 에러 코드 (OK = 성공)
     */
    public static int applyBeautyFilterV2(@NonNull byte[] frameData, int width, int height,
                                           int format, @NonNull BeautyFilterConfigV2 config,
                                           @Nullable IrisResult result) {
        if (!sLibraryLoaded) {
            return NOT_INITIALIZED;
        }
        // IrisResult를 네이티브 포인터로 변환하는 것은 현재 지원하지 않음
        // 추후 네이티브 결과 캐싱 구현 시 사용
        return nativeApplyBeautyV2(frameData, width, height, format, config, 0L);
    }

    /**
     * V2 뷰티 필터를 GPU 텍스처에 적용합니다.
     *
     * <p>GPU 뷰티 백엔드가 초기화되어 있어야 합니다.
     * GLSurfaceView.Renderer의 onDrawFrame 등 OpenGL 컨텍스트 내에서 호출해야 합니다.</p>
     *
     * @param inputTexture 입력 OpenGL ES 텍스처 ID
     * @param width 텍스처 너비
     * @param height 텍스처 높이
     * @param config V2 뷰티 필터 설정
     * @return 출력 텍스처 ID (실패 시 입력 텍스처 반환)
     */
    public static int applyBeautyFilterTextureV2(int inputTexture, int width, int height,
                                                  @NonNull BeautyFilterConfigV2 config) {
        if (!sLibraryLoaded) {
            return inputTexture;
        }
        return nativeApplyBeautyTextureV2(inputTexture, width, height, config, 0L, 0, 0.0f);
    }

    /**
     * V2 뷰티 필터 + LUT를 GPU 텍스처에 적용합니다.
     *
     * <p>GPU 뷰티 백엔드가 초기화되어 있어야 합니다.
     * GLSurfaceView.Renderer의 onDrawFrame 등 OpenGL 컨텍스트 내에서 호출해야 합니다.</p>
     *
     * @param inputTexture 입력 OpenGL ES 텍스처 ID
     * @param width 텍스처 너비
     * @param height 텍스처 높이
     * @param config V2 뷰티 필터 설정
     * @param lutTextureId LUT 3D 텍스처 ID (0이면 LUT 비활성)
     * @param lutIntensity LUT 적용 강도 (0.0~1.0)
     * @return 출력 텍스처 ID (실패 시 입력 텍스처 반환)
     */
    public static int applyBeautyFilterTextureV2(int inputTexture, int width, int height,
                                                  @NonNull BeautyFilterConfigV2 config,
                                                  int lutTextureId, float lutIntensity) {
        if (!sLibraryLoaded) {
            return inputTexture;
        }
        return nativeApplyBeautyTextureV2(inputTexture, width, height, config, 0L,
                lutTextureId, lutIntensity);
    }

    /**
     * V2 뷰티 필터 + LUT를 GPU 텍스처에 적용합니다 (Detection Handle 포함).
     *
     * <p>Detection 슬롯의 검출 결과를 활용하여 ROI 기반 처리를 수행합니다.
     * {@link #getDetectionSlotPtr()}로 얻은 포인터를 전달하세요.</p>
     *
     * @param inputTexture 입력 OpenGL ES 텍스처 ID
     * @param width 텍스처 너비
     * @param height 텍스처 높이
     * @param config V2 뷰티 필터 설정
     * @param detectionHandle 네이티브 검출 결과 포인터 (0L이면 ROI 미사용)
     * @param lutTextureId LUT 3D 텍스처 ID (0이면 LUT 비활성)
     * @param lutIntensity LUT 적용 강도 (0.0~1.0)
     * @return 출력 텍스처 ID (실패 시 입력 텍스처 반환)
     */
    public static int applyBeautyFilterTextureV2(int inputTexture, int width, int height,
                                                  @NonNull BeautyFilterConfigV2 config,
                                                  long detectionHandle,
                                                  int lutTextureId, float lutIntensity) {
        if (!sLibraryLoaded) {
            return inputTexture;
        }
        return nativeApplyBeautyTextureV2(inputTexture, width, height, config, detectionHandle,
                lutTextureId, lutIntensity);
    }

    /**
     * Face Warp 효과를 GPU 텍스처에 적용합니다 (Detection Handle 포함).
     *
     * <p>Detection 슬롯의 검출 결과를 활용하여 얼굴 형태 보정을 수행합니다.</p>
     *
     * @param inputTexture 입력 OpenGL ES 텍스처 ID
     * @param width 텍스처 너비
     * @param height 텍스처 높이
     * @param slimFace 얼굴 슬림화 강도 (0.0~1.0)
     * @param thinChin 턱 축소 강도 (0.0~1.0)
     * @param enlargeEyes 눈 확대 강도 (0.0~1.0)
     * @param detectionHandle 네이티브 검출 결과 포인터 (0L이면 pass-through)
     * @return 출력 텍스처 ID
     */
    public static int applyFaceWarp(int inputTexture, int width, int height,
                                     float slimFace, float thinChin, float enlargeEyes,
                                     long detectionHandle) {
        if (!sLibraryLoaded) {
            return inputTexture;
        }
        return nativeApplyFaceWarp(inputTexture, width, height, slimFace, thinChin, enlargeEyes, detectionHandle);
    }

    /**
     * Face Warp 효과를 GPU 텍스처에 적용합니다.
     *
     * <p>얼굴 형태 보정(슬림 페이스, 턱 축소, 눈 확대)을 수행합니다.
     * 얼굴 검출 결과가 필요합니다.</p>
     *
     * @param inputTexture 입력 OpenGL ES 텍스처 ID
     * @param width 텍스처 너비
     * @param height 텍스처 높이
     * @param slimFace 얼굴 슬림화 강도 (0.0~1.0)
     * @param thinChin 턱 축소 강도 (0.0~1.0)
     * @param enlargeEyes 눈 확대 강도 (0.0~1.0)
     * @return 출력 텍스처 ID
     */
    public static int applyFaceWarp(int inputTexture, int width, int height,
                                     float slimFace, float thinChin, float enlargeEyes) {
        if (!sLibraryLoaded) {
            return inputTexture;
        }
        // 검출 결과 없이 호출 시 pass-through
        return nativeApplyFaceWarp(inputTexture, width, height, slimFace, thinChin, enlargeEyes, 0L);
    }

    /**
     * SDK가 관리하는 텍스처를 해제합니다.
     *
     * <p>applyBeautyFilterTextureV2나 applyFaceWarp에서 반환된 텍스처 중
     * SDK가 내부적으로 생성한 텍스처를 해제합니다.</p>
     *
     * @param texture 해제할 텍스처 ID
     */
    public static void releaseTexture(int texture) {
        if (sLibraryLoaded) {
            nativeReleaseTexture(texture);
        }
    }

    /**
     * FreqSep 디버그 모드 설정 (GL 스레드에서 호출).
     * @param mode 0=off, 1=magnitude heatmap, 2=compression, 3=mask
     */
    public static void setFreqSepDebugMode(int mode) {
        if (sLibraryLoaded) {
            nativeSetFreqSepDebugMode(mode);
        }
    }

    /**
     * 피부색 기반 마스크 필터 설정 (실험용, FreqSep 전용).
     * GL 스레드에서 호출.
     * @param enabled true=on, false=off
     */
    public static void setSkinColorFilter(boolean enabled) {
        if (sLibraryLoaded) {
            nativeSetSkinColorFilter(enabled ? 1 : 0);
        }
    }

    /**
     * 텍스처가 SDK 관리인지 확인합니다.
     *
     * @param texture 확인할 텍스처 ID
     * @return true면 SDK 관리 텍스처
     */
    public static boolean isTextureManaged(int texture) {
        if (!sLibraryLoaded) {
            return false;
        }
        return nativeIsTextureManaged(texture);
    }

    // ========================================================================
    // GPU 렌즈 렌더링
    // ========================================================================

    /**
     * GPU 렌즈 렌더러를 초기화합니다.
     *
     * <p>GL 스레드에서 호출해야 합니다.</p>
     *
     * @return 에러 코드 (OK = 성공)
     */
    public static int initGpuLens() {
        if (!sLibraryLoaded) {
            return NOT_INITIALIZED;
        }
        return nativeInitGpuLens();
    }

    /**
     * GPU 렌즈 렌더러를 해제합니다.
     */
    public static void releaseGpuLens() {
        if (sLibraryLoaded) {
            nativeReleaseGpuLens();
        }
    }

    /**
     * GPU 렌즈 렌더러 초기화 여부를 확인합니다.
     */
    public static boolean isGpuLensInitialized() {
        if (!sLibraryLoaded) {
            return false;
        }
        return nativeIsGpuLensInitialized();
    }

    /**
     * 렌즈 텍스처를 로드합니다 (RGBA 바이트 배열).
     *
     * @param rgbaData RGBA 픽셀 데이터
     * @param width 너비
     * @param height 높이
     * @return 에러 코드
     */
    public static int loadLensTexture(byte[] rgbaData, int width, int height) {
        if (!sLibraryLoaded) {
            return NOT_INITIALIZED;
        }
        return nativeLoadLensTexture(rgbaData, width, height);
    }

    /**
     * 렌즈 텍스처를 SKU와 함께 로드합니다 (RGBA 바이트 배열).
     *
     * <p>P6-W7: sku_id를 함께 전달하여 코어에 등록된 렌즈 메타(림발 등)와
     * 연동합니다. {@link #setLensMetadata(String)}로 등록한 메타에서
     * 해당 sku_id 항목을 찾아 적용합니다.</p>
     *
     * @param rgbaData RGBA 픽셀 데이터
     * @param width 너비
     * @param height 높이
     * @param skuId 렌즈 SKU 식별자 (빈 문자열이면 메타 미적용)
     * @return 에러 코드
     */
    public static int loadLensTexture(byte[] rgbaData, int width, int height, String skuId) {
        if (!sLibraryLoaded) {
            return NOT_INITIALIZED;
        }
        return nativeLoadLensTextureWithSku(rgbaData, width, height, skuId);
    }

    /**
     * 렌즈 메타데이터(JSON)를 코어에 등록합니다.
     *
     * <p>P6-W7: lens_meta.json 등 SKU별 렌즈 메타(림발 강도 등)를
     * 코어에 1회 등록합니다. GPU 렌즈가 초기화되어 있으면 즉시 주입하고,
     * 그렇지 않으면 보관 후 초기화 시점에 주입합니다.</p>
     *
     * @param json 렌즈 메타데이터 JSON 문자열
     * @return 에러 코드 (OK = 성공)
     */
    public static int setLensMetadata(String json) {
        if (!sLibraryLoaded) {
            return NOT_INITIALIZED;
        }
        return nativeSetLensMetadata(json);
    }

    /**
     * 렌즈 텍스처를 해제합니다.
     */
    public static void unloadLensTexture() {
        if (sLibraryLoaded) {
            nativeUnloadLensTexture();
        }
    }

    /**
     * P6-W4 §5.7: 환경 반사 env_map 텍스처 로드 (RGB 8bit, ACES tone-mapped LDR 권장).
     *
     * @param rgbData RGB 픽셀 데이터 (3 bytes/pixel)
     * @param width 너비 (W3 §5.13: 256 권장)
     * @param height 높이 (W3 §5.13: 128 권장)
     * @return 에러 코드 (0=성공)
     */
    public static int loadEnvMap(byte[] rgbData, int width, int height) {
        if (!sLibraryLoaded) {
            return NOT_INITIALIZED;
        }
        return nativeLoadEnvMap(rgbData, width, height);
    }

    /** P6-W4 §5.7: 환경 반사 env_map 해제. */
    public static void unloadEnvMap() {
        if (sLibraryLoaded) {
            nativeUnloadEnvMap();
        }
    }

    /**
     * P6-W4 §5.11: 환경 반사 모드 런타임 토글 (방식 A uniform 스위치, 단일 바이너리).
     *
     * @param mode 0=OFF, 1=EnvMap, 2=Periphery (W4 B2 벤치 3 프로토타입)
     */
    public static void setReflectionMode(int mode) {
        if (sLibraryLoaded) {
            nativeSetReflectionMode(mode);
        }
    }

    /**
     * P6-W4 §5.7 hand-off: 환경 반사 강도 조정 (W3 §5.7 기본 0.3, R1 미토론).
     *
     * @param intensity 0.0~1.0 (자동 clamp)
     */
    public static void setReflectionIntensity(float intensity) {
        if (sLibraryLoaded) {
            nativeSetReflectionIntensity(intensity);
        }
    }

    /**
     * GPU 렌즈 렌더링을 수행합니다.
     *
     * @param inputTexture 입력 카메라 프레임 텍스처 ID
     * @param width 프레임 너비
     * @param height 프레임 높이
     * @param detectionHandle 검출 결과 네이티브 포인터 (0이면 null)
     * @param config 렌즈 설정
     * @return 출력 텍스처 ID (0이면 실패)
     */
    public static int renderLensTexture(int inputTexture, int width, int height,
                                         long detectionHandle, LensConfig config) {
        if (!sLibraryLoaded) {
            return 0;
        }
        return nativeRenderLensTexture(inputTexture, width, height, detectionHandle, config);
    }

    /**
     * GPU 렌즈 Sclera Protection 설정.
     * GL 스레드에서 호출.
     * @param enabled true=on, false=off
     */
    public static void setLensScleraProtect(boolean enabled) {
        if (sLibraryLoaded) {
            nativeSetLensScleraProtect(enabled);
        }
    }

    /**
     * P6-W5 §5.9: sclera veto 수식 토글 (B1/B8 4조합 벤치용).
     * @param mode 0=legacy, 1=color-veto(Codex), 2=luma-only(Gemini). 범위 외 값은 0으로 clamp.
     */
    public static void setScleraVetoMode(int mode) {
        if (sLibraryLoaded) {
            nativeSetScleraVetoMode(mode);
        }
    }

    /**
     * P6-W6 B5: 블링크 up ramp 시간 토글 (벤치용).
     * @param ms 60/80/120ms (자동 clamp [30,200]). 기본 80.
     */
    public static void setBlinkUpMs(float ms) {
        if (sLibraryLoaded) {
            nativeSetBlinkUpMs(ms);
        }
    }

    /**
     * P6-W6 B9: 저조도 디테일 gate 임계값 토글 (벤치용).
     * @param threshold 0.10/0.15/0.25 (자동 clamp [0,1]). 기본 0.15.
     */
    public static void setGateThreshold(float threshold) {
        if (sLibraryLoaded) {
            nativeSetGateThreshold(threshold);
        }
    }

    /**
     * P6-W6 C10: 홍채 디테일 재주입 on/off 토글 (벤치용).
     * @param enabled true=on(기본), false=off.
     */
    public static void setDetailReinject(boolean enabled) {
        if (sLibraryLoaded) {
            nativeSetDetailReinject(enabled);
        }
    }

    /**
     * GPU 렌즈 타원 마스크 설정.
     * GL 스레드에서 호출.
     * @param enabled true=on, false=off
     */
    public static void setLensEllipseMask(boolean enabled) {
        if (sLibraryLoaded) {
            nativeSetLensEllipseMask(enabled);
        }
    }

    /**
     * GPU 렌즈 각막 하이라이트 설정.
     *
     * @deprecated P5-W3-05 S1: 고정 조명 하이라이트 폐기 (환경과 무관해 어색함).
     *             C5 환경 반사 가산 계층(B2 결과 후)이 대체. 현재 no-op.
     *             호출부는 제거 권장. 공개 API 호환성을 위해 심볼만 유지.
     * @param enabled (무시됨)
     */
    @Deprecated
    public static void setLensHighlight(boolean enabled) {
        if (sLibraryLoaded) {
            nativeSetLensHighlight(enabled);
        }
    }

    // ========================================================================
    // Detection Slot API (더블 버퍼, Lock-free)
    // ========================================================================

    /**
     * Detection 슬롯에 최신 검출 결과를 기록합니다.
     *
     * <p>Analyzer 스레드에서 검출 완료 후 호출합니다.
     * 내부적으로 더블 버퍼를 사용하여 GL 스레드와 lock-free로 데이터를 공유합니다.</p>
     *
     * @param result 검출 결과
     */
    public static void updateDetectionSlot(@NonNull IrisResult result) {
        if (sLibraryLoaded) {
            nativeUpdateDetectionSlot(result);
        }
    }

    /**
     * Detection 슬롯에서 최신 검출 결과의 네이티브 포인터를 가져옵니다.
     *
     * <p>GL 스레드에서 뷰티 필터 적용 직전에 호출합니다.
     * 반환값은 C++ IrisResult 구조체 포인터로, GPU 뷰티 함수의
     * detectionHandle 파라미터에 전달합니다.</p>
     *
     * @return 네이티브 IrisResult 포인터 (유효하지 않으면 0L)
     */
    public static long getDetectionSlotPtr() {
        if (!sLibraryLoaded) {
            return 0L;
        }
        return nativeGetDetectionSlotPtr();
    }

    /**
     * Detection 슬롯을 해제합니다.
     *
     * <p>SDK 종료 시 또는 리소스 정리 시 호출합니다.</p>
     */
    public static void releaseDetectionSlot() {
        if (sLibraryLoaded) {
            nativeReleaseDetectionSlot();
        }
    }

    // ========================================================================
    // 정보 API
    // ========================================================================

    /**
     * SDK 버전을 반환합니다.
     *
     * @return 버전 문자열 (예: "1.0.0")
     */
    @NonNull
    public static String getVersion() {
        if (!sLibraryLoaded) {
            return "library not loaded";
        }
        return nativeGetVersion();
    }

    /**
     * 빌드 정보를 반환합니다.
     *
     * @return 빌드 정보 문자열
     */
    @NonNull
    public static String getBuildInfo() {
        if (!sLibraryLoaded) {
            return "library not loaded";
        }
        return nativeGetBuildInfo();
    }

    /**
     * 마지막 에러 메시지를 반환합니다.
     *
     * @return 에러 메시지
     */
    @NonNull
    public static String getLastError() {
        if (!sLibraryLoaded) {
            return "library not loaded";
        }
        return nativeGetLastError();
    }

    /**
     * 에러 코드를 문자열로 변환합니다.
     *
     * @param errorCode 에러 코드
     * @return 에러 설명 문자열
     */
    @NonNull
    public static String errorToString(int errorCode) {
        if (!sLibraryLoaded) {
            return "LIBRARY_NOT_LOADED";
        }
        return nativeErrorToString(errorCode);
    }

    /**
     * 현재 모델 경로를 반환합니다.
     *
     * @return 모델 경로 (초기화 전이면 null)
     */
    @Nullable
    public static String getModelPath() {
        return sModelPath;
    }

    // ========================================================================
    // Temporal Stabilizer API (P5-W1)
    // ========================================================================

    /**
     * Temporal Stabilizer를 생성합니다.
     *
     * <p>SDK 코어의 TemporalStabilizer를 생성하여 홍채 검출 결과의
     * 시간적 안정화를 수행할 수 있게 합니다. OneEuroFilter + 이력현상 +
     * dropout hold 등을 통합 제공합니다.</p>
     *
     * @return Stabilizer 핸들 (0이면 실패)
     */
    public static long createStabilizer() {
        if (!sLibraryLoaded) return 0;
        return nativeCreateStabilizer();
    }

    /**
     * 검출 결과를 스무딩합니다 (in-place).
     *
     * <p>전달된 IrisResult 객체의 값이 스무딩된 결과로 덮어씌워집니다.
     * 반환값은 현재 가시성(0.0~1.0)으로, fade-in/out 상태를 나타냅니다.</p>
     *
     * @param handle createStabilizer()에서 반환된 핸들
     * @param result 검출 결과 (in-place로 스무딩됨)
     * @param timestampSec 타임스탬프 (초 단위, System.nanoTime() / 1e9)
     * @return visibility (0.0~1.0), 실패 시 0.0
     */
    public static float stabilize(long handle, @NonNull IrisResult result, double timestampSec) {
        if (!sLibraryLoaded || handle == 0) return 0f;
        return nativeStabilize(handle, result, timestampSec);
    }

    /**
     * Temporal Stabilizer를 해제합니다.
     *
     * @param handle createStabilizer()에서 반환된 핸들
     */
    public static void destroyStabilizer(long handle) {
        if (!sLibraryLoaded || handle == 0) return;
        nativeDestroyStabilizer(handle);
    }

    // ========================================================================
    // 유틸리티 메서드
    // ========================================================================

    /**
     * assets에서 모델 파일들을 추출합니다.
     *
     * @param context Android Context
     * @return 추출된 모델 디렉토리 경로
     */
    @Nullable
    private static String extractModelsFromAssets(@NonNull Context context) {
        try {
            File modelDir = new File(context.getFilesDir(), "iris_models");
            if (!modelDir.exists() && !modelDir.mkdirs()) {
                Log.e(TAG, "Failed to create model directory");
                return null;
            }

            // assets/models 디렉토리의 파일들 복사
            String[] files = context.getAssets().list("models");
            if (files == null || files.length == 0) {
                Log.w(TAG, "No model files found in assets/models");
                // 모델이 없어도 경로는 반환 (SDK가 에러 처리)
                return modelDir.getAbsolutePath();
            }

            for (String file : files) {
                File destFile = new File(modelDir, file);
                if (!destFile.exists()) {
                    copyAssetToFile(context, "models/" + file, destFile);
                    Log.d(TAG, "Extracted model: " + file);
                }
            }

            return modelDir.getAbsolutePath();

        } catch (IOException e) {
            Log.e(TAG, "Failed to extract models: " + e.getMessage(), e);
            return null;
        }
    }

    /**
     * asset 파일을 로컬 파일로 복사합니다.
     *
     * @param context Android Context
     * @param assetPath assets 내 경로
     * @param destFile 대상 파일
     */
    private static void copyAssetToFile(@NonNull Context context,
                                        @NonNull String assetPath,
                                        @NonNull File destFile) throws IOException {
        try (InputStream in = context.getAssets().open(assetPath);
             OutputStream out = new FileOutputStream(destFile)) {

            byte[] buffer = new byte[8192];
            int length;
            while ((length = in.read(buffer)) > 0) {
                out.write(buffer, 0, length);
            }
        }
    }

    // ========================================================================
    // 네이티브 메서드 선언
    // ========================================================================

    private static native int nativeInit(String modelPath);
    private static native void nativeDestroy();
    private static native boolean nativeIsReady();

    private static native int nativeLoadTexture(String path);
    private static native int nativeLoadTextureFromMemory(byte[] data, int width, int height);

    private static native int nativeDetect(byte[] frameData, int width, int height,
                                           int format, IrisResult result);
    private static native int nativeDetectWithRotation(byte[] frameData, int width, int height,
                                                        int format, int rotationDegrees,
                                                        IrisResult result);
    private static native int nativeProcess(byte[] frameData, int width, int height,
                                            int format, LensConfig config, IrisResult result);

    private static native int nativeSetConfig(String key, String value);

    // GPU 가속 API
    private static native void nativeSetGpuEnabled(boolean enable);
    private static native boolean nativeIsGpuAvailable();
    private static native boolean nativeIsUsingGpu();

    // 신뢰도 설정 API
    private static native void nativeSetMinDetectionConfidence(float minConfidence);
    private static native void nativeSetMinTrackingConfidence(float minConfidence);
    private static native void nativeSetMinPresenceConfidence(float minConfidence);

    // InferenceThread 설정 API (벤치마크용)
    private static native void nativeSetUseInferenceThread(boolean enable);
    private static native boolean nativeIsUsingInferenceThread();

    private static native String nativeGetVersion();
    private static native String nativeGetBuildInfo();
    private static native String nativeGetLastError();
    private static native String nativeErrorToString(int errorCode);

    // Beauty Filter API
    private static native void nativeDefaultBeautyConfig(BeautyFilterConfig config);
    private static native int nativeSetBeautyFilter(BeautyFilterConfig config);
    private static native int nativeGetBeautyFilter(BeautyFilterConfig config);
    private static native boolean nativeIsBeautyFilterEnabled();
    private static native int nativeApplyBeautyFilter(byte[] frameData, int width, int height, int format);

    // NV21 → RGBA 고속 변환 API
    private static native int nativeNv21ToRgba(byte[] nv21Data, int width, int height, android.graphics.Bitmap bitmap);

    // ========================================================================
    // Beauty Filter V2 Native Methods
    // ========================================================================

    /**
     * GPU 뷰티 백엔드 초기화
     * @return 에러 코드 (0 = 성공)
     */
    private static native int nativeInitGpuBeauty();

    /**
     * GPU 뷰티 백엔드 해제
     */
    private static native void nativeReleaseGpuBeauty();

    /**
     * GPU 뷰티 백엔드 초기화 여부 확인
     */
    private static native boolean nativeIsGpuBeautyInitialized();

    /**
     * 기본 V2 설정으로 초기화
     */
    private static native void nativeDefaultBeautyConfigV2(BeautyFilterConfigV2 config);

    /**
     * V2 뷰티 필터 적용 (CPU 버퍼)
     */
    private static native int nativeApplyBeautyV2(
            byte[] frameData, int width, int height, int format,
            BeautyFilterConfigV2 config, long detectionPtr);

    /**
     * V2 뷰티 필터 적용 (GPU 텍스처)
     * @return 출력 텍스처 ID (0이면 실패)
     */
    private static native int nativeApplyBeautyTextureV2(
            int inputTexture, int width, int height,
            BeautyFilterConfigV2 config, long detectionPtr,
            int lutTextureId, float lutIntensity);

    private static native void nativeSetFreqSepDebugMode(int mode);
    private static native void nativeSetSkinColorFilter(int enabled);

    /**
     * Face Warp 적용 (GPU)
     * @return 출력 텍스처 ID
     */
    private static native int nativeApplyFaceWarp(
            int inputTexture, int width, int height,
            float slimFace, float thinChin, float enlargeEyes,
            long detectionPtr);

    /**
     * SDK 관리 텍스처 해제
     */
    private static native void nativeReleaseTexture(int texture);

    /**
     * 텍스처가 SDK 관리인지 확인
     */
    private static native boolean nativeIsTextureManaged(int texture);

    // ========================================================================
    // GPU 렌즈 렌더링 Native Methods
    // ========================================================================

    private static native int nativeInitGpuLens();
    private static native void nativeReleaseGpuLens();
    private static native boolean nativeIsGpuLensInitialized();
    private static native int nativeLoadLensTexture(byte[] data, int width, int height);
    private static native int nativeLoadLensTextureWithSku(byte[] data, int width, int height, String skuId);  // P6-W7
    private static native int nativeSetLensMetadata(String json);  // P6-W7
    private static native void nativeUnloadLensTexture();
    private static native int nativeRenderLensTexture(
            int inputTexture, int width, int height,
            long detectionPtr, LensConfig config);
    private static native void nativeSetLensScleraProtect(boolean enabled);
    private static native void nativeSetScleraVetoMode(int mode);  // P6-W5 §5.9
    private static native void nativeSetLensEllipseMask(boolean enabled);
    private static native void nativeSetLensHighlight(boolean enabled);

    // P6-W4 §5.7/§5.11: 환경 반사 (env_map + 모드 토글 + 강도)
    private static native int nativeLoadEnvMap(byte[] rgbData, int width, int height);
    private static native void nativeUnloadEnvMap();
    private static native void nativeSetReflectionMode(int mode);
    private static native void nativeSetReflectionIntensity(float intensity);

    // P6-W6: 블링크 ramp(B5) / 저조도 gate(B9) / 디테일 재주입(C10) 벤치 토글
    private static native void nativeSetBlinkUpMs(float ms);
    private static native void nativeSetGateThreshold(float threshold);
    private static native void nativeSetDetailReinject(boolean enabled);

    // ========================================================================
    // Temporal Stabilizer Native Methods
    // ========================================================================

    private static native long nativeCreateStabilizer();
    private static native float nativeStabilize(long handle, IrisResult result, double timestampSec);
    private static native void nativeDestroyStabilizer(long handle);

    // ========================================================================
    // Detection Slot Native Methods
    // ========================================================================

    /**
     * Detection 슬롯에 검출 결과 기록 (Analyzer → GL 더블 버퍼)
     */
    private static native void nativeUpdateDetectionSlot(IrisResult result);

    /**
     * Detection 슬롯에서 활성 IrisResult 포인터 반환
     * @return 네이티브 포인터 (jlong), 유효하지 않으면 0L
     */
    private static native long nativeGetDetectionSlotPtr();

    /**
     * Detection 슬롯 해제
     */
    private static native void nativeReleaseDetectionSlot();
}
