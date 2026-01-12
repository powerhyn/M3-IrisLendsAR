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

import com.irislenssdk.IrisLensSDK as JavaIrisLensSDK

/**
 * IrisLensSDK 예외 계층.
 *
 * sealed class를 사용하여 타입 안전한 예외 처리를 제공합니다.
 *
 * 사용 예:
 * ```kotlin
 * sdk.detect(frameData, width, height)
 *     .onFailure { error ->
 *         when (error) {
 *             is IrisException.NotInitialized -> reinitialize()
 *             is IrisException.NoFaceDetected -> showHint()
 *             else -> logError(error)
 *         }
 *     }
 * ```
 *
 * @property errorCode 원본 에러 코드
 */
sealed class IrisException(
    message: String,
    val errorCode: Int
) : Exception(message) {

    // ========================================================================
    // 초기화 에러 (100-199)
    // ========================================================================

    /** SDK가 초기화되지 않음 */
    class NotInitialized(
        message: String = "SDK is not initialized"
    ) : IrisException(message, JavaIrisLensSDK.NOT_INITIALIZED)

    /** SDK가 이미 초기화됨 */
    class AlreadyInitialized(
        message: String = "SDK is already initialized"
    ) : IrisException(message, JavaIrisLensSDK.ALREADY_INITIALIZED)

    /** 모델 로드 실패 */
    class ModelLoadFailed(
        message: String = "Failed to load model"
    ) : IrisException(message, JavaIrisLensSDK.MODEL_LOAD_FAILED)

    /** 잘못된 경로 */
    class InvalidPath(
        message: String = "Invalid file path"
    ) : IrisException(message, JavaIrisLensSDK.INVALID_PATH)

    // ========================================================================
    // 파라미터 에러 (200-299)
    // ========================================================================

    /** 잘못된 파라미터 */
    class InvalidParameter(
        message: String = "Invalid parameter"
    ) : IrisException(message, JavaIrisLensSDK.INVALID_PARAM)

    /** 널 포인터 */
    class NullPointer(
        message: String = "Null pointer received"
    ) : IrisException(message, JavaIrisLensSDK.NULL_POINTER)

    /** 지원하지 않는 포맷 */
    class InvalidFormat(
        message: String = "Unsupported frame format"
    ) : IrisException(message, JavaIrisLensSDK.INVALID_FORMAT)

    // ========================================================================
    // 검출 에러 (300-399)
    // ========================================================================

    /** 검출 실패 */
    class DetectionFailed(
        message: String = "Detection failed"
    ) : IrisException(message, JavaIrisLensSDK.DETECTION_FAILED)

    /** 얼굴 미검출 */
    class NoFaceDetected(
        message: String = "No face detected in frame"
    ) : IrisException(message, JavaIrisLensSDK.NO_FACE)

    // ========================================================================
    // 렌더링 에러 (400-499)
    // ========================================================================

    /** 렌더링 실패 */
    class RenderFailed(
        message: String = "Render operation failed"
    ) : IrisException(message, JavaIrisLensSDK.RENDER_FAILED)

    /** 텍스처 미로드 */
    class NoTexture(
        message: String = "Lens texture not loaded"
    ) : IrisException(message, JavaIrisLensSDK.NO_TEXTURE)

    // ========================================================================
    // 알 수 없는 에러
    // ========================================================================

    /** 알 수 없는 에러 */
    class Unknown(
        message: String = "Unknown error occurred",
        code: Int = JavaIrisLensSDK.UNKNOWN
    ) : IrisException(message, code)

    companion object {
        /**
         * 에러 코드에서 적절한 예외를 생성합니다.
         *
         * @param errorCode 에러 코드
         * @param message 추가 메시지 (선택)
         * @return 해당 IrisException
         */
        fun fromErrorCode(errorCode: Int, message: String? = null): IrisException {
            val errorMessage = message ?: JavaIrisLensSDK.errorToString(errorCode)

            return when (errorCode) {
                // 초기화 에러
                JavaIrisLensSDK.NOT_INITIALIZED -> NotInitialized(errorMessage)
                JavaIrisLensSDK.ALREADY_INITIALIZED -> AlreadyInitialized(errorMessage)
                JavaIrisLensSDK.MODEL_LOAD_FAILED -> ModelLoadFailed(errorMessage)
                JavaIrisLensSDK.INVALID_PATH -> InvalidPath(errorMessage)

                // 파라미터 에러
                JavaIrisLensSDK.INVALID_PARAM -> InvalidParameter(errorMessage)
                JavaIrisLensSDK.NULL_POINTER -> NullPointer(errorMessage)
                JavaIrisLensSDK.INVALID_FORMAT -> InvalidFormat(errorMessage)

                // 검출 에러
                JavaIrisLensSDK.DETECTION_FAILED -> DetectionFailed(errorMessage)
                JavaIrisLensSDK.NO_FACE -> NoFaceDetected(errorMessage)

                // 렌더링 에러
                JavaIrisLensSDK.RENDER_FAILED -> RenderFailed(errorMessage)
                JavaIrisLensSDK.NO_TEXTURE -> NoTexture(errorMessage)

                // 기타
                else -> Unknown(errorMessage, errorCode)
            }
        }

        /**
         * 에러 코드가 성공인지 확인합니다.
         *
         * @param errorCode 에러 코드
         * @return 성공이면 true
         */
        fun isSuccess(errorCode: Int): Boolean = errorCode == JavaIrisLensSDK.OK
    }
}

/**
 * Int 에러 코드를 IrisException으로 변환하는 확장 함수.
 */
internal fun Int.toIrisException(message: String? = null): IrisException =
    IrisException.fromErrorCode(this, message)
