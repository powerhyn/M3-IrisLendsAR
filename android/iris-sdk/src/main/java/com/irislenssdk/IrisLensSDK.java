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
}
