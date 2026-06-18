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

package com.irislenssdk.kotlin

import android.content.Context
import com.irislenssdk.IrisLensSDK as JavaIrisLensSDK

/**
 * IrisLensSDK Kotlin API.
 *
 * 홍채 추적 및 가상 렌즈 오버레이 SDK의 Kotlin 인터페이스입니다.
 * Java API를 래핑하여 Kotlin 관용적인 API를 제공합니다.
 *
 * ## 기능
 * - 실시간 홍채 검출
 * - 가상 렌즈 오버레이 렌더링
 * - 다양한 프레임 포맷 지원 (NV21, RGBA, RGB 등)
 *
 * ## 사용 예
 * ```kotlin
 * // 초기화
 * val sdk = IrisLensSDKKt.getInstance()
 * sdk.init(context)
 *     .onSuccess { Log.d(TAG, "SDK initialized") }
 *     .onFailure { error -> Log.e(TAG, "Init failed", error) }
 *
 * // 텍스처 로드
 * sdk.loadTexture(texturePath)
 *
 * // 프레임 처리 (검출 + 렌더링)
 * sdk.process(frameData, width, height)
 *     .onSuccess { result ->
 *         if (result.rendered) {
 *             // frameData에 렌즈 오버레이가 적용됨
 *         }
 *     }
 *
 * // 종료
 * sdk.destroy()
 * ```
 *
 * ## 스레드 안전성
 * 모든 메서드는 스레드 안전합니다.
 * C++ 코어의 mutex가 동시 접근을 보호합니다.
 *
 * @author IrisLensSDK Team
 * @version 1.0.0
 */
class IrisLensSDKKt private constructor() {

    // ========================================================================
    // 내부 상태
    // ========================================================================

    // ========================================================================
    // 초기화 API
    // ========================================================================

    /**
     * Context를 사용하여 SDK를 초기화합니다.
     *
     * assets/models 디렉토리에서 모델 파일을 추출하고 초기화합니다.
     *
     * @param context Android Context (Application 또는 Activity)
     * @return 성공 시 [Result.success], 실패 시 [Result.failure]
     */
    fun init(context: Context): Result<Unit> {
        val errorCode = JavaIrisLensSDK.init(context)
        return errorCodeToResult(errorCode)
    }

    /**
     * 지정된 모델 경로로 SDK를 초기화합니다.
     *
     * @param modelPath 모델 파일이 있는 디렉토리 경로
     * @return 성공 시 [Result.success], 실패 시 [Result.failure]
     */
    fun init(modelPath: String): Result<Unit> {
        val errorCode = JavaIrisLensSDK.init(modelPath)
        return errorCodeToResult(errorCode)
    }

    /**
     * SDK를 종료하고 리소스를 해제합니다.
     *
     * 여러 번 호출해도 안전합니다.
     */
    fun destroy() {
        JavaIrisLensSDK.destroy()
    }

    /**
     * SDK가 사용 가능한 상태인지 확인합니다.
     *
     * @return 사용 가능하면 true
     */
    val isReady: Boolean
        get() = JavaIrisLensSDK.isReady()

    /**
     * 네이티브 라이브러리가 로드되었는지 확인합니다.
     *
     * @return 로드되었으면 true
     */
    val isLibraryLoaded: Boolean
        get() = JavaIrisLensSDK.isLibraryLoaded()

    // ========================================================================
    // 텍스처 API
    // ========================================================================

    /**
     * 파일에서 렌즈 텍스처를 로드합니다.
     *
     * 지원 포맷: PNG, JPEG, BMP
     *
     * @param path 텍스처 이미지 파일 경로
     * @return 성공 시 [Result.success], 실패 시 [Result.failure]
     */
    @Deprecated("cpu-render(CPU 렌즈) 경로는 2.0에서 제거됩니다(ADR-0001 §8.2). GPU 텍스처 렌즈 경로로 이행하세요.")
    @Suppress("DEPRECATION")
    fun loadTexture(path: String): Result<Unit> {
        val errorCode = JavaIrisLensSDK.loadTexture(path)
        return errorCodeToResult(errorCode)
    }

    /**
     * assets에서 렌즈 텍스처를 로드합니다.
     *
     * @param context Android Context
     * @param assetPath assets 내 상대 경로
     * @return 성공 시 [Result.success], 실패 시 [Result.failure]
     */
    @Deprecated("cpu-render(CPU 렌즈) 경로는 2.0에서 제거됩니다(ADR-0001 §8.2). GPU 텍스처 렌즈 경로로 이행하세요.")
    @Suppress("DEPRECATION")
    fun loadTextureFromAssets(context: Context, assetPath: String): Result<Unit> {
        val errorCode = JavaIrisLensSDK.loadTextureFromAssets(context, assetPath)
        return errorCodeToResult(errorCode)
    }

    /**
     * 메모리에서 RGBA 텍스처를 로드합니다.
     *
     * @param data RGBA 픽셀 데이터 (4바이트/픽셀)
     * @param width 텍스처 너비
     * @param height 텍스처 높이
     * @return 성공 시 [Result.success], 실패 시 [Result.failure]
     */
    @Deprecated("cpu-render(CPU 렌즈) 경로는 2.0에서 제거됩니다(ADR-0001 §8.2). GPU 텍스처 렌즈 경로로 이행하세요.")
    @Suppress("DEPRECATION")
    fun loadTextureFromMemory(data: ByteArray, width: Int, height: Int): Result<Unit> {
        val errorCode = JavaIrisLensSDK.loadTextureFromMemory(data, width, height)
        return errorCodeToResult(errorCode)
    }

    // ========================================================================
    // 설정 API
    // ========================================================================

    /**
     * 런타임 설정을 변경합니다.
     *
     * @param key 설정 키
     * @param value 설정 값
     * @return 성공 시 [Result.success], 실패 시 [Result.failure]
     */
    fun setConfig(key: String, value: String): Result<Unit> {
        val errorCode = JavaIrisLensSDK.setConfig(key, value)
        return errorCodeToResult(errorCode)
    }

    // ========================================================================
    // 정보 API
    // ========================================================================

    /**
     * SDK 버전을 반환합니다.
     */
    val version: String
        get() = JavaIrisLensSDK.getVersion()

    /**
     * 빌드 정보를 반환합니다.
     */
    val buildInfo: String
        get() = JavaIrisLensSDK.getBuildInfo()

    /**
     * 마지막 에러 메시지를 반환합니다.
     */
    val lastError: String
        get() = JavaIrisLensSDK.getLastError()

    /**
     * 현재 모델 경로를 반환합니다.
     */
    val modelPath: String?
        get() = JavaIrisLensSDK.getModelPath()

    // ========================================================================
    // 유틸리티
    // ========================================================================

    /**
     * 에러 코드를 Result로 변환합니다.
     */
    private fun errorCodeToResult(errorCode: Int): Result<Unit> {
        return if (IrisException.isSuccess(errorCode)) {
            Result.success(Unit)
        } else {
            Result.failure(errorCode.toIrisException())
        }
    }

    // ========================================================================
    // Companion Object
    // ========================================================================

    companion object {
        @Volatile
        private var instance: IrisLensSDKKt? = null

        /**
         * SDK 싱글톤 인스턴스를 반환합니다.
         *
         * 스레드 안전한 지연 초기화를 사용합니다.
         *
         * @return IrisLensSDKKt 인스턴스
         */
        @JvmStatic
        fun getInstance(): IrisLensSDKKt {
            return instance ?: synchronized(this) {
                instance ?: IrisLensSDKKt().also { instance = it }
            }
        }

        /**
         * SDK 버전을 정적으로 조회합니다.
         *
         * 인스턴스 생성 없이 호출할 수 있습니다.
         */
        @JvmStatic
        fun getVersionStatic(): String = JavaIrisLensSDK.getVersion()

        /**
         * 에러 코드를 문자열로 변환합니다.
         *
         * @param errorCode 에러 코드
         * @return 에러 설명 문자열
         */
        @JvmStatic
        fun errorToString(errorCode: Int): String = JavaIrisLensSDK.errorToString(errorCode)
    }
}

// ========================================================================
// 확장 함수
// ========================================================================

/**
 * Result 성공 시 flatMap 변환.
 *
 * 참고: Result.map은 Kotlin stdlib에서 제공됩니다.
 */
inline fun <T, R> Result<T>.flatMap(transform: (T) -> Result<R>): Result<R> {
    return fold(
        onSuccess = { transform(it) },
        onFailure = { Result.failure(it) }
    )
}
