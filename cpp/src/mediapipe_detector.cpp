/**
 * @file mediapipe_detector.cpp
 * @brief MediaPipeDetector 구현 - TensorFlow Lite 통합
 *
 * MediaPipe Face Mesh + Iris Landmark 모델을 사용한 홍채 검출
 * TensorFlow Lite 런타임으로 모델 추론 수행
 */

#include "iris_sdk/mediapipe_detector.h"
#include <algorithm>  // std::clamp
#include <filesystem>
#include <cstring>    // std::memcpy
#include <cmath>      // std::sqrt, std::exp
#include <cstdio>     // std::fprintf (디버그용)

// TensorFlow Lite 헤더 (조건부 컴파일)
#ifdef IRIS_SDK_HAS_TFLITE
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#endif

// OpenCV 헤더 (조건부 컴파일)
#ifdef IRIS_SDK_HAS_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#endif

namespace iris_sdk {

// ============================================================
// 모델 입출력 크기 상수 (TFLite/OpenCV 활성화 시만 사용)
// ============================================================
#if defined(IRIS_SDK_HAS_TFLITE) && defined(IRIS_SDK_HAS_OPENCV)
namespace {
    // Face Detection 모델: 128x128 RGB
    constexpr int FACE_DETECTION_INPUT_WIDTH = 128;
    constexpr int FACE_DETECTION_INPUT_HEIGHT = 128;
    constexpr int FACE_DETECTION_INPUT_CHANNELS = 3;

    // Face Landmark 모델: 192x192 RGB
    constexpr int FACE_LANDMARK_INPUT_WIDTH = 192;
    constexpr int FACE_LANDMARK_INPUT_HEIGHT = 192;
    constexpr int FACE_LANDMARK_INPUT_CHANNELS = 3;

    // Iris Landmark 모델: 64x64 RGB
    constexpr int IRIS_LANDMARK_INPUT_WIDTH = 64;
    constexpr int IRIS_LANDMARK_INPUT_HEIGHT = 64;
    constexpr int IRIS_LANDMARK_INPUT_CHANNELS = 3;

    // 랜드마크 개수
    constexpr int FACE_LANDMARK_COUNT = 468;
    constexpr int IRIS_LANDMARK_COUNT = 5;  // center + 4 boundary points

    // 눈 영역 랜드마크 인덱스 (MediaPipe Face Mesh 기준)
    // 왼쪽 눈 (이미지 기준 오른쪽)
    constexpr int LEFT_EYE_INDICES[] = {
        33, 7, 163, 144, 145, 153, 154, 155, 133,  // 눈 윤곽
        173, 157, 158, 159, 160, 161, 246          // 눈꺼풀
    };
    // 오른쪽 눈 (이미지 기준 왼쪽)
    constexpr int RIGHT_EYE_INDICES[] = {
        362, 382, 381, 380, 374, 373, 390, 249, 263,
        466, 388, 387, 386, 385, 384, 398
    };

    // 눈 중심 랜드마크 인덱스 (미래 사용을 위해 예약)
    [[maybe_unused]] constexpr int LEFT_EYE_CENTER_INDEX = 468;   // Face Mesh에서 추가된 iris 중심
    [[maybe_unused]] constexpr int RIGHT_EYE_CENTER_INDEX = 473;
}
#endif  // IRIS_SDK_HAS_TFLITE && IRIS_SDK_HAS_OPENCV

// ============================================================
// Pimpl 구현 클래스
// ============================================================
class MediaPipeDetector::Impl {
public:
    bool initialized = false;
    std::string model_path;

    // MediaPipe 설정
    float min_detection_confidence = 0.5f;
    float min_tracking_confidence = 0.5f;
    int num_faces = 1;

    // ========================================
    // 성능 최적화: 설정
    // ========================================
    int num_threads = 4;           ///< TFLite 추론 스레드 수
    bool use_tracking = true;      ///< 추적 모드 활성화 여부
    float tracking_iou_threshold = 0.5f;  ///< 추적 유지 IoU 임계값

    // ========================================
    // 성능 최적화: 사전 할당 버퍼 (메모리 재사용)
    // ========================================
#if defined(IRIS_SDK_HAS_TFLITE) && defined(IRIS_SDK_HAS_OPENCV)
    // 입력 버퍼 (모델별 전처리 결과 저장)
    std::vector<float> face_detection_input_buffer;
    std::vector<float> face_landmark_input_buffer;
    std::vector<float> left_iris_input_buffer;
    std::vector<float> right_iris_input_buffer;

    // 출력 버퍼 (모델 추론 결과 저장)
    std::vector<float> face_landmarks_buffer;
    std::vector<float> left_iris_landmarks_buffer;
    std::vector<float> right_iris_landmarks_buffer;

    // 이미지 처리용 버퍼 (OpenCV Mat 재사용)
    cv::Mat rgb_buffer;        ///< RGB 변환 결과 저장
    cv::Mat resized_buffer;    ///< 리사이즈 결과 저장
    cv::Mat cropped_buffer;    ///< 크롭 결과 저장
    cv::Mat float_buffer;      ///< float 변환 결과 저장

    // ========================================
    // 성능 최적화: 이전 프레임 캐시 (추적 모드용)
    // ========================================
    IrisResult prev_result;
    bool has_prev_result = false;
    Rect prev_face_rect;       ///< 이전 프레임 얼굴 영역
#endif

#ifdef IRIS_SDK_HAS_TFLITE
    // TFLite 모델 및 인터프리터
    std::unique_ptr<tflite::FlatBufferModel> face_detection_model;
    std::unique_ptr<tflite::FlatBufferModel> face_landmark_model;
    std::unique_ptr<tflite::FlatBufferModel> iris_landmark_model;

    std::unique_ptr<tflite::Interpreter> face_detection_interpreter;
    std::unique_ptr<tflite::Interpreter> face_landmark_interpreter;
    std::unique_ptr<tflite::Interpreter> iris_landmark_interpreter;

    // 입출력 텐서 인덱스
    int face_detection_input_index = -1;
    int face_detection_output_boxes_index = -1;
    int face_detection_output_scores_index = -1;

    int face_landmark_input_index = -1;
    int face_landmark_output_index = -1;

    int iris_landmark_input_index = -1;

    // ========================================
    // BlazeFace 앵커 (SSD-style anchors)
    // ========================================
    struct Anchor {
        float x_center;
        float y_center;
        float width;
        float height;
    };
    std::vector<Anchor> face_detection_anchors;
    bool anchors_generated = false;

    int iris_landmark_output_index = -1;
#endif

    // 필요한 모델 파일 목록
    static constexpr const char* REQUIRED_MODELS[] = {
        "face_detection_short_range.tflite",
        "face_landmark.tflite",
        "iris_landmark.tflite"
    };

    /**
     * @brief 모델 경로 유효성 검사
     * @param path 모델 디렉토리 경로
     * @return 유효 여부
     */
    bool validateModelPath(const std::string& path) {
        // 빈 경로 검사
        if (path.empty()) {
            return false;
        }

        // 경로 존재 여부 검사
        if (!std::filesystem::exists(path)) {
            return false;
        }

        // 디렉토리 여부 검사
        if (!std::filesystem::is_directory(path)) {
            return false;
        }

        // 필수 모델 파일 존재 여부 검사
        for (const auto& model : REQUIRED_MODELS) {
            std::filesystem::path model_file = std::filesystem::path(path) / model;
            if (!std::filesystem::exists(model_file)) {
                return false;
            }
        }

        return true;
    }

#if defined(IRIS_SDK_HAS_TFLITE) && defined(IRIS_SDK_HAS_OPENCV)
    /**
     * @brief 사전 할당 버퍼 초기화
     *
     * 모든 입출력 버퍼를 미리 할당하여 detect() 호출 시
     * 메모리 할당 오버헤드를 제거합니다.
     */
    void initializeBuffers() {
        // Face Detection 입력 버퍼
        face_detection_input_buffer.resize(
            FACE_DETECTION_INPUT_WIDTH * FACE_DETECTION_INPUT_HEIGHT *
            FACE_DETECTION_INPUT_CHANNELS);

        // Face Landmark 입력/출력 버퍼
        face_landmark_input_buffer.resize(
            FACE_LANDMARK_INPUT_WIDTH * FACE_LANDMARK_INPUT_HEIGHT *
            FACE_LANDMARK_INPUT_CHANNELS);
        face_landmarks_buffer.resize(FACE_LANDMARK_COUNT * 3);

        // Iris Landmark 입력/출력 버퍼 (왼쪽, 오른쪽)
        left_iris_input_buffer.resize(
            IRIS_LANDMARK_INPUT_WIDTH * IRIS_LANDMARK_INPUT_HEIGHT *
            IRIS_LANDMARK_INPUT_CHANNELS);
        right_iris_input_buffer.resize(
            IRIS_LANDMARK_INPUT_WIDTH * IRIS_LANDMARK_INPUT_HEIGHT *
            IRIS_LANDMARK_INPUT_CHANNELS);
        left_iris_landmarks_buffer.resize(IRIS_LANDMARK_COUNT * 3);
        right_iris_landmarks_buffer.resize(IRIS_LANDMARK_COUNT * 3);
    }

    /**
     * @brief 추적 캐시 초기화
     */
    void resetTrackingCache() {
        has_prev_result = false;
        prev_result = IrisResult{};
        prev_face_rect = Rect{};
    }

    /**
     * @brief 두 Rect의 IoU(Intersection over Union) 계산
     * @param a 첫 번째 영역
     * @param b 두 번째 영역
     * @return IoU 값 (0.0 ~ 1.0)
     */
    float calculateIoU(const Rect& a, const Rect& b) {
        float x1 = std::max(a.x, b.x);
        float y1 = std::max(a.y, b.y);
        float x2 = std::min(a.x + a.width, b.x + b.width);
        float y2 = std::min(a.y + a.height, b.y + b.height);

        float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
        float area_a = a.width * a.height;
        float area_b = b.width * b.height;
        float union_area = area_a + area_b - intersection;

        if (union_area <= 0.0f) return 0.0f;
        return intersection / union_area;
    }
#endif

#ifdef IRIS_SDK_HAS_TFLITE
    /**
     * @brief Sigmoid 함수 (logit → probability)
     * @param x 입력 logit 값
     * @return 확률 값 (0.0 ~ 1.0)
     */
    static float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }

    /**
     * @brief BlazeFace 앵커 생성
     *
     * MediaPipe BlazeFace short range 모델의 앵커 구성:
     * - 입력: 128x128
     * - Feature maps: 16x16 (stride 8), 8x8 (stride 16)
     * - 각 위치에 2개 앵커
     * - 총 앵커 수: 16*16*2 + 8*8*2 = 512 + 128 = 640
     *
     * 참고: 실제 모델은 896개 앵커를 사용할 수 있음 (추가 feature map)
     */
    void generateAnchors() {
        if (anchors_generated) return;

        face_detection_anchors.clear();

        // BlazeFace short range 앵커 옵션
        // MediaPipe 기본 설정 기반
        struct AnchorOption {
            int feature_map_size;
            int num_anchors;
            float stride;
        };

        // 두 가지 feature map 레벨
        // 실제 MediaPipe BlazeFace는 더 복잡한 앵커 구조를 사용할 수 있음
        std::vector<AnchorOption> options = {
            {16, 2, 8.0f},   // 16x16 feature map, 2 anchors per cell, stride 8
            {8, 6, 16.0f}    // 8x8 feature map, 6 anchors per cell, stride 16
        };

        float input_size = static_cast<float>(FACE_DETECTION_INPUT_WIDTH);

        for (const auto& opt : options) {
            for (int y = 0; y < opt.feature_map_size; ++y) {
                for (int x = 0; x < opt.feature_map_size; ++x) {
                    // 앵커 중심 좌표 (정규화된 0~1)
                    float x_center = (static_cast<float>(x) + 0.5f) * opt.stride / input_size;
                    float y_center = (static_cast<float>(y) + 0.5f) * opt.stride / input_size;

                    for (int n = 0; n < opt.num_anchors; ++n) {
                        Anchor anchor;
                        anchor.x_center = x_center;
                        anchor.y_center = y_center;
                        // BlazeFace는 앵커 크기를 사용하지 않음 (scale만 사용)
                        anchor.width = 1.0f;
                        anchor.height = 1.0f;
                        face_detection_anchors.push_back(anchor);
                    }
                }
            }
        }

        anchors_generated = true;
    }

    /**
     * @brief TFLite 모델 로드 및 인터프리터 생성
     * @param model_file 모델 파일 경로
     * @param model 모델 포인터 (출력)
     * @param interpreter 인터프리터 포인터 (출력)
     * @return 성공 여부
     */
    bool loadModel(const std::string& model_file,
                   std::unique_ptr<tflite::FlatBufferModel>& model,
                   std::unique_ptr<tflite::Interpreter>& interpreter) {
        // 모델 파일 로드
        model = tflite::FlatBufferModel::BuildFromFile(model_file.c_str());
        if (!model) {
            return false;
        }

        // 인터프리터 빌더 생성
        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder builder(*model, resolver);

        // 멀티스레드 설정 (성능 최적화)
        builder.SetNumThreads(num_threads);

        // 인터프리터 생성
        if (builder(&interpreter) != kTfLiteOk) {
            return false;
        }

        if (!interpreter) {
            return false;
        }

        // 추가 스레드 설정 (런타임 변경 가능)
        interpreter->SetNumThreads(num_threads);

        // 텐서 할당
        if (interpreter->AllocateTensors() != kTfLiteOk) {
            return false;
        }

        return true;
    }

    /**
     * @brief 모든 TFLite 모델 로드
     * @param base_path 모델 디렉토리 경로
     * @return 성공 여부
     */
    bool loadAllModels(const std::string& base_path) {
        std::filesystem::path base(base_path);

        // Face Detection 모델 로드
        std::string face_detection_path = (base / REQUIRED_MODELS[0]).string();
        if (!loadModel(face_detection_path, face_detection_model, face_detection_interpreter)) {
            return false;
        }

        // Face Detection 텐서 인덱스 설정
        if (face_detection_interpreter->inputs().empty() ||
            face_detection_interpreter->outputs().empty()) {
            return false;  // 모델 구조 오류
        }
        face_detection_input_index = face_detection_interpreter->inputs()[0];
        // 출력 텐서: [0] = boxes, [1] = scores (모델에 따라 다를 수 있음)
        if (face_detection_interpreter->outputs().size() >= 2) {
            face_detection_output_boxes_index = face_detection_interpreter->outputs()[0];
            face_detection_output_scores_index = face_detection_interpreter->outputs()[1];
        } else {
            // 단일 출력 모델인 경우
            face_detection_output_boxes_index = face_detection_interpreter->outputs()[0];
            face_detection_output_scores_index = -1;
        }

        // Face Landmark 모델 로드
        std::string face_landmark_path = (base / REQUIRED_MODELS[1]).string();
        if (!loadModel(face_landmark_path, face_landmark_model, face_landmark_interpreter)) {
            return false;
        }

        // Face Landmark 텐서 인덱스 설정
        if (face_landmark_interpreter->inputs().empty() ||
            face_landmark_interpreter->outputs().empty()) {
            return false;  // 모델 구조 오류
        }
        face_landmark_input_index = face_landmark_interpreter->inputs()[0];
        face_landmark_output_index = face_landmark_interpreter->outputs()[0];

        // Iris Landmark 모델 로드
        std::string iris_landmark_path = (base / REQUIRED_MODELS[2]).string();
        if (!loadModel(iris_landmark_path, iris_landmark_model, iris_landmark_interpreter)) {
            return false;
        }

        // Iris Landmark 텐서 인덱스 설정
        if (iris_landmark_interpreter->inputs().empty() ||
            iris_landmark_interpreter->outputs().empty()) {
            return false;  // 모델 구조 오류
        }
        iris_landmark_input_index = iris_landmark_interpreter->inputs()[0];
        iris_landmark_output_index = iris_landmark_interpreter->outputs()[0];

        return true;
    }

    /**
     * @brief 모든 TFLite 리소스 해제
     */
    void releaseAllModels() {
        face_detection_interpreter.reset();
        face_landmark_interpreter.reset();
        iris_landmark_interpreter.reset();

        face_detection_model.reset();
        face_landmark_model.reset();
        iris_landmark_model.reset();

        face_detection_input_index = -1;
        face_detection_output_boxes_index = -1;
        face_detection_output_scores_index = -1;
        face_landmark_input_index = -1;
        face_landmark_output_index = -1;
        iris_landmark_input_index = -1;
        iris_landmark_output_index = -1;
    }
#endif  // IRIS_SDK_HAS_TFLITE

#if defined(IRIS_SDK_HAS_TFLITE) && defined(IRIS_SDK_HAS_OPENCV)
    /**
     * @brief FrameFormat을 OpenCV 색상 변환 코드로 변환
     * @param format 입력 프레임 포맷
     * @return OpenCV 변환 코드 (RGB 출력), -1이면 변환 불필요 또는 미지원
     */
    int getColorConversionCode(FrameFormat format) {
        switch (format) {
            case FrameFormat::RGBA:
                return cv::COLOR_RGBA2RGB;
            case FrameFormat::BGRA:
                return cv::COLOR_BGRA2RGB;
            case FrameFormat::RGB:
                return -1;  // 변환 불필요
            case FrameFormat::BGR:
                return cv::COLOR_BGR2RGB;
            case FrameFormat::NV21:
                return cv::COLOR_YUV2RGB_NV21;
            case FrameFormat::NV12:
                return cv::COLOR_YUV2RGB_NV12;
            case FrameFormat::Grayscale:
                return cv::COLOR_GRAY2RGB;
            default:
                return -2;  // 미지원 포맷
        }
    }

    /**
     * @brief 입력 이미지를 RGB로 변환하고 리사이즈 (최적화 버전)
     *
     * 성능 최적화 적용:
     * 1. OpenCV SIMD 연산 활용 (convertTo)
     * 2. 불필요한 clone() 제거
     * 3. 내부 버퍼 재사용
     *
     * @param frame_data 입력 이미지 데이터
     * @param width 입력 너비
     * @param height 입력 높이
     * @param format 입력 포맷
     * @param target_width 목표 너비
     * @param target_height 목표 높이
     * @param output 출력 버퍼 (정규화된 float)
     * @return 성공 여부
     */
    bool preprocessImage(const uint8_t* frame_data, int width, int height,
                         FrameFormat format, int target_width, int target_height,
                         float* output) {
        if (!frame_data || !output) {
            return false;
        }

        // OpenCV Mat 생성 (데이터 복사 없이 래핑)
        cv::Mat input_mat;
        int input_type = CV_8UC3;

        switch (format) {
            case FrameFormat::RGBA:
            case FrameFormat::BGRA:
                input_type = CV_8UC4;
                break;
            case FrameFormat::RGB:
            case FrameFormat::BGR:
                input_type = CV_8UC3;
                break;
            case FrameFormat::NV21:
            case FrameFormat::NV12:
                // YUV 포맷: height * 1.5 크기
                input_mat = cv::Mat(height + height / 2, width, CV_8UC1,
                                   const_cast<uint8_t*>(frame_data));
                break;
            case FrameFormat::Grayscale:
                input_type = CV_8UC1;
                break;
            default:
                return false;
        }

        // NV21/NV12가 아닌 경우 Mat 생성
        if (format != FrameFormat::NV21 && format != FrameFormat::NV12) {
            input_mat = cv::Mat(height, width, input_type,
                               const_cast<uint8_t*>(frame_data));
        }

        // RGB로 변환 (rgb_buffer 재사용)
        int conversion_code = getColorConversionCode(format);
        if (conversion_code == -1) {
            // RGB 포맷인 경우 복사 없이 직접 사용
            // 단, resize에서 입력과 출력이 같으면 문제가 될 수 있으므로 체크
            if (input_mat.cols == target_width && input_mat.rows == target_height) {
                rgb_buffer = input_mat;  // 참조만 저장 (복사 없음)
            } else {
                rgb_buffer = input_mat;  // resize에서 처리
            }
        } else if (conversion_code >= 0) {
            cv::cvtColor(input_mat, rgb_buffer, conversion_code);
        } else {
            return false;  // 미지원 포맷
        }

        // 리사이즈 (resized_buffer 재사용)
        if (rgb_buffer.cols != target_width || rgb_buffer.rows != target_height) {
            cv::resize(rgb_buffer, resized_buffer, cv::Size(target_width, target_height),
                      0, 0, cv::INTER_LINEAR);
        } else {
            resized_buffer = rgb_buffer;
        }

        // ============================================================
        // 성능 최적화: OpenCV SIMD 연산으로 정규화
        // 기존 픽셀별 루프 대신 convertTo 사용 (약 5-10배 빠름)
        // ============================================================
        // CV_32FC3로 변환하면서 1/255 스케일링 적용
        resized_buffer.convertTo(float_buffer, CV_32FC3, 1.0 / 255.0);

        // float_buffer는 연속 메모리이므로 직접 복사
        // OpenCV Mat은 HWC 포맷이고, 출력도 HWC (연속 RGB)
        if (float_buffer.isContinuous()) {
            std::memcpy(output, float_buffer.ptr<float>(),
                       target_width * target_height * 3 * sizeof(float));
        } else {
            // 연속 메모리가 아닌 경우 (드문 경우) 행별 복사
            float* dst = output;
            for (int y = 0; y < target_height; ++y) {
                const float* src = float_buffer.ptr<float>(y);
                std::memcpy(dst, src, target_width * 3 * sizeof(float));
                dst += target_width * 3;
            }
        }

        return true;
    }

    /**
     * @brief 입력 이미지를 RGB cv::Mat으로 변환 (내부 버퍼 사용)
     *
     * detect() 함수에서 한 번만 RGB 변환을 수행하고
     * 이후 크롭/리사이즈에 재사용하기 위한 함수
     *
     * @param frame_data 입력 이미지 데이터
     * @param width 입력 너비
     * @param height 입력 높이
     * @param format 입력 포맷
     * @param output_rgb 출력 RGB Mat (멤버 버퍼 참조)
     * @return 성공 여부
     */
    bool convertToRgb(const uint8_t* frame_data, int width, int height,
                      FrameFormat format, cv::Mat& output_rgb) {
        if (!frame_data) {
            return false;
        }

        // OpenCV Mat 생성 (데이터 복사 없이 래핑)
        cv::Mat input_mat;
        int input_type = CV_8UC3;

        switch (format) {
            case FrameFormat::RGBA:
            case FrameFormat::BGRA:
                input_type = CV_8UC4;
                break;
            case FrameFormat::RGB:
            case FrameFormat::BGR:
                input_type = CV_8UC3;
                break;
            case FrameFormat::NV21:
            case FrameFormat::NV12:
                input_mat = cv::Mat(height + height / 2, width, CV_8UC1,
                                   const_cast<uint8_t*>(frame_data));
                break;
            case FrameFormat::Grayscale:
                input_type = CV_8UC1;
                break;
            default:
                return false;
        }

        if (format != FrameFormat::NV21 && format != FrameFormat::NV12) {
            input_mat = cv::Mat(height, width, input_type,
                               const_cast<uint8_t*>(frame_data));
        }

        // RGB로 변환
        int conversion_code = getColorConversionCode(format);
        if (conversion_code == -1) {
            output_rgb = input_mat;  // 복사 없이 참조
        } else if (conversion_code >= 0) {
            cv::cvtColor(input_mat, output_rgb, conversion_code);
        } else {
            return false;
        }

        return true;
    }

    /**
     * @brief 눈 영역 이미지 추출 (크롭) - 최적화 버전
     *
     * 성능 최적화 적용:
     * 1. 내부 버퍼 재사용 (cropped_buffer, resized_buffer)
     * 2. OpenCV SIMD 연산으로 정규화 (convertTo)
     *
     * @param rgb_image RGB 이미지 (CV_8UC3)
     * @param landmarks 얼굴 랜드마크 (468 * 3)
     * @param eye_indices 눈 영역 랜드마크 인덱스
     * @param num_indices 인덱스 개수
     * @param output_size 출력 크기 (정사각형)
     * @param output 출력 버퍼 (정규화된 float)
     * @param crop_rect 크롭된 영역 (출력, 정규화 좌표)
     * @return 성공 여부
     */
    bool extractEyeRegion(const cv::Mat& rgb_image, const float* landmarks,
                          const int* eye_indices, int num_indices,
                          int output_size, float* output, Rect& crop_rect) {
        if (!landmarks || !output) {
            return false;
        }

        int img_width = rgb_image.cols;
        int img_height = rgb_image.rows;

        // 눈 영역 바운딩 박스 계산
        float min_x = 1.0f, min_y = 1.0f;
        float max_x = 0.0f, max_y = 0.0f;

        for (int i = 0; i < num_indices; ++i) {
            int idx = eye_indices[i];
            float x = landmarks[idx * 3 + 0];
            float y = landmarks[idx * 3 + 1];

            min_x = std::min(min_x, x);
            min_y = std::min(min_y, y);
            max_x = std::max(max_x, x);
            max_y = std::max(max_y, y);
        }

        // 마진 추가 (20%)
        float width = max_x - min_x;
        float height = max_y - min_y;
        float margin = std::max(width, height) * 0.2f;

        min_x = std::max(0.0f, min_x - margin);
        min_y = std::max(0.0f, min_y - margin);
        max_x = std::min(1.0f, max_x + margin);
        max_y = std::min(1.0f, max_y + margin);

        // 정사각형으로 만들기
        float center_x = (min_x + max_x) / 2.0f;
        float center_y = (min_y + max_y) / 2.0f;
        float size = std::max(max_x - min_x, max_y - min_y);

        min_x = std::max(0.0f, center_x - size / 2.0f);
        min_y = std::max(0.0f, center_y - size / 2.0f);
        max_x = std::min(1.0f, center_x + size / 2.0f);
        max_y = std::min(1.0f, center_y + size / 2.0f);

        // 크롭 영역 저장
        crop_rect.x = min_x;
        crop_rect.y = min_y;
        crop_rect.width = max_x - min_x;
        crop_rect.height = max_y - min_y;

        // 픽셀 좌표로 변환
        int px_min_x = static_cast<int>(min_x * img_width);
        int px_min_y = static_cast<int>(min_y * img_height);
        int px_max_x = static_cast<int>(max_x * img_width);
        int px_max_y = static_cast<int>(max_y * img_height);

        // 경계 체크
        px_min_x = std::clamp(px_min_x, 0, img_width - 1);
        px_min_y = std::clamp(px_min_y, 0, img_height - 1);
        px_max_x = std::clamp(px_max_x, 1, img_width);
        px_max_y = std::clamp(px_max_y, 1, img_height);

        // 크롭 (ROI는 복사 없이 참조)
        cv::Rect roi(px_min_x, px_min_y, px_max_x - px_min_x, px_max_y - px_min_y);
        cv::Mat cropped = rgb_image(roi);

        // 리사이즈 (cropped_buffer 재사용)
        cv::resize(cropped, cropped_buffer, cv::Size(output_size, output_size),
                  0, 0, cv::INTER_LINEAR);

        // ============================================================
        // 성능 최적화: OpenCV SIMD 연산으로 정규화
        // ============================================================
        cropped_buffer.convertTo(float_buffer, CV_32FC3, 1.0 / 255.0);

        // 출력 버퍼로 복사
        if (float_buffer.isContinuous()) {
            std::memcpy(output, float_buffer.ptr<float>(),
                       output_size * output_size * 3 * sizeof(float));
        } else {
            float* dst = output;
            for (int y = 0; y < output_size; ++y) {
                const float* src = float_buffer.ptr<float>(y);
                std::memcpy(dst, src, output_size * 3 * sizeof(float));
                dst += output_size * 3;
            }
        }

        return true;
    }

    /**
     * @brief 얼굴 영역을 크롭하여 Face Landmark 모델 입력 준비
     *
     * 성능 최적화: Face Detection 결과 영역만 크롭하여
     * Face Landmark 모델에 입력 (전체 이미지 대신)
     *
     * @param rgb_image RGB 이미지
     * @param face_rect 얼굴 영역 (정규화 좌표)
     * @param target_width 목표 너비
     * @param target_height 목표 높이
     * @param output 출력 버퍼 (정규화된 float)
     * @return 성공 여부
     */
    bool cropFaceRegion(const cv::Mat& rgb_image, const Rect& face_rect,
                        int target_width, int target_height, float* output) {
        if (!output || face_rect.width <= 0 || face_rect.height <= 0) {
            return false;
        }

        int img_width = rgb_image.cols;
        int img_height = rgb_image.rows;

        // 마진 추가 (10%) - 얼굴 영역 약간 확장
        float margin_x = face_rect.width * 0.1f;
        float margin_y = face_rect.height * 0.1f;

        float min_x = std::max(0.0f, face_rect.x - margin_x);
        float min_y = std::max(0.0f, face_rect.y - margin_y);
        float max_x = std::min(1.0f, face_rect.x + face_rect.width + margin_x);
        float max_y = std::min(1.0f, face_rect.y + face_rect.height + margin_y);

        // 픽셀 좌표로 변환
        int px_min_x = static_cast<int>(min_x * img_width);
        int px_min_y = static_cast<int>(min_y * img_height);
        int px_max_x = static_cast<int>(max_x * img_width);
        int px_max_y = static_cast<int>(max_y * img_height);

        // 경계 체크
        px_min_x = std::clamp(px_min_x, 0, img_width - 1);
        px_min_y = std::clamp(px_min_y, 0, img_height - 1);
        px_max_x = std::clamp(px_max_x, 1, img_width);
        px_max_y = std::clamp(px_max_y, 1, img_height);

        // 크롭
        cv::Rect roi(px_min_x, px_min_y, px_max_x - px_min_x, px_max_y - px_min_y);
        cv::Mat cropped = rgb_image(roi);

        // 리사이즈
        cv::resize(cropped, cropped_buffer, cv::Size(target_width, target_height),
                  0, 0, cv::INTER_LINEAR);

        // 정규화
        cropped_buffer.convertTo(float_buffer, CV_32FC3, 1.0 / 255.0);

        // 출력 버퍼로 복사
        if (float_buffer.isContinuous()) {
            std::memcpy(output, float_buffer.ptr<float>(),
                       target_width * target_height * 3 * sizeof(float));
        } else {
            float* dst = output;
            for (int y = 0; y < target_height; ++y) {
                const float* src = float_buffer.ptr<float>(y);
                std::memcpy(dst, src, target_width * 3 * sizeof(float));
                dst += target_width * 3;
            }
        }

        return true;
    }

    /**
     * @brief Face Detection 실행 (BlazeFace 앵커 기반 디코딩)
     * @param input_data 전처리된 입력 데이터 (128x128 RGB float)
     * @param face_rect 검출된 얼굴 영역 (출력, 정규화 좌표)
     * @param confidence 신뢰도 (출력)
     * @return 얼굴 검출 성공 여부
     *
     * BlazeFace short range 모델 출력 구조:
     * - Output 0 (regressors): [1, num_anchors, 16]
     *   - [0-3]: yc, xc, h, w (앵커 상대 오프셋)
     *   - [4-15]: 6개 키포인트 좌표 (각 x, y)
     * - Output 1 (classificators): [1, num_anchors, 1]
     *   - logit 값 (sigmoid 적용 필요)
     */
    bool runFaceDetection(const float* input_data, Rect& face_rect, float& confidence) {
        if (!face_detection_interpreter || !input_data) {
            return false;
        }

        // 앵커 생성 (처음 한 번만)
        generateAnchors();

        // 입력 텐서에 데이터 복사
        float* input_tensor = face_detection_interpreter->typed_input_tensor<float>(0);
        if (!input_tensor) {
            return false;
        }

        int input_size = FACE_DETECTION_INPUT_WIDTH * FACE_DETECTION_INPUT_HEIGHT *
                         FACE_DETECTION_INPUT_CHANNELS;
        std::memcpy(input_tensor, input_data, input_size * sizeof(float));

        // 추론 실행
        if (face_detection_interpreter->Invoke() != kTfLiteOk) {
            return false;
        }

        // 출력 텐서 가져오기
        // Output 0: regressors (box offsets)
        TfLiteTensor* boxes_tensor = face_detection_interpreter->tensor(
            face_detection_interpreter->outputs()[0]);
        if (!boxes_tensor) {
            return false;
        }

        float* boxes_data = face_detection_interpreter->typed_output_tensor<float>(0);
        if (!boxes_data) {
            return false;
        }

        // Output 1: classificators (scores)
        float* scores_data = nullptr;
        int num_anchors = 0;

        // 출력 텐서 차원 분석 및 디버그 출력 (최초 1회만)
        static bool debug_printed = false;
        int num_dims = boxes_tensor->dims->size;
        int num_outputs = static_cast<int>(face_detection_interpreter->outputs().size());

        if (!debug_printed) {
            std::fprintf(stderr, "[DEBUG] Face Detection Model Output Info:\n");
            std::fprintf(stderr, "  - Number of outputs: %d\n", num_outputs);
            std::fprintf(stderr, "  - Output 0 dims: %d [", num_dims);
            for (int d = 0; d < num_dims; ++d) {
                std::fprintf(stderr, "%d%s", boxes_tensor->dims->data[d],
                           (d < num_dims - 1) ? ", " : "");
            }
            std::fprintf(stderr, "]\n");

            if (num_outputs >= 2) {
                TfLiteTensor* scores_tensor = face_detection_interpreter->tensor(
                    face_detection_interpreter->outputs()[1]);
                if (scores_tensor) {
                    std::fprintf(stderr, "  - Output 1 dims: %d [", scores_tensor->dims->size);
                    for (int d = 0; d < scores_tensor->dims->size; ++d) {
                        std::fprintf(stderr, "%d%s", scores_tensor->dims->data[d],
                                   (d < scores_tensor->dims->size - 1) ? ", " : "");
                    }
                    std::fprintf(stderr, "]\n");
                }
            }
            std::fprintf(stderr, "  - Generated anchors: %zu\n", face_detection_anchors.size());
            debug_printed = true;
        }

        if (num_dims >= 2) {
            // [1, num_anchors, 16] 또는 [num_anchors, 16]
            num_anchors = (num_dims == 3) ? boxes_tensor->dims->data[1] : boxes_tensor->dims->data[0];
        }

        if (num_anchors <= 0) {
            return false;
        }

        // 점수 텐서 확인 (두 번째 출력이 있는 경우)
        if (face_detection_interpreter->outputs().size() >= 2) {
            scores_data = face_detection_interpreter->typed_tensor<float>(
                face_detection_interpreter->outputs()[1]);
        }

        // 앵커 수와 모델 출력 확인
        int anchor_count = static_cast<int>(face_detection_anchors.size());
        int effective_anchors = std::min(num_anchors, anchor_count);

        // 디버그: 첫 몇 개 점수 출력
        static bool scores_debug_printed = false;
        if (!scores_debug_printed && scores_data) {
            std::fprintf(stderr, "[DEBUG] First 10 scores (raw logits):\n  ");
            for (int i = 0; i < std::min(10, effective_anchors); ++i) {
                std::fprintf(stderr, "%.4f ", scores_data[i]);
            }
            std::fprintf(stderr, "\n[DEBUG] First 10 scores (after sigmoid):\n  ");
            for (int i = 0; i < std::min(10, effective_anchors); ++i) {
                std::fprintf(stderr, "%.4f ", sigmoid(scores_data[i]));
            }
            std::fprintf(stderr, "\n");
            scores_debug_printed = true;
        }

        // 모든 앵커에서 가장 높은 점수 찾기
        float best_score = 0.0f;
        int best_idx = -1;

        for (int i = 0; i < effective_anchors; ++i) {
            float score_logit;

            if (scores_data) {
                // 별도 점수 텐서가 있는 경우
                score_logit = scores_data[i];
            } else {
                // 점수가 boxes 텐서에 포함된 경우 (일부 모델)
                // 일반적으로 BlazeFace는 별도 텐서 사용
                score_logit = boxes_data[i * 16 + 15];
            }

            // sigmoid로 확률 변환
            float score = sigmoid(score_logit);

            if (score > best_score && score > min_detection_confidence) {
                best_score = score;
                best_idx = i;
            }
        }

        if (best_idx < 0) {
            return false;
        }

        // 바운딩 박스 디코딩 (앵커 기반)
        const Anchor& anchor = face_detection_anchors[best_idx];

        // BlazeFace 출력: [yc_offset, xc_offset, h_scale, w_scale, ...]
        // 실제 좌표 = 앵커 좌표 + 오프셋
        float yc_offset = boxes_data[best_idx * 16 + 0];
        float xc_offset = boxes_data[best_idx * 16 + 1];
        float h_scale = boxes_data[best_idx * 16 + 2];
        float w_scale = boxes_data[best_idx * 16 + 3];

        // 정규화된 좌표로 디코딩
        // 오프셋은 입력 크기에 상대적인 값
        float input_size_f = static_cast<float>(FACE_DETECTION_INPUT_WIDTH);
        float cx = anchor.x_center + xc_offset / input_size_f;
        float cy = anchor.y_center + yc_offset / input_size_f;
        float w = w_scale / input_size_f;
        float h = h_scale / input_size_f;

        // 최종 바운딩 박스 (정규화 좌표 0~1)
        face_rect.x = std::clamp(cx - w / 2.0f, 0.0f, 1.0f);
        face_rect.y = std::clamp(cy - h / 2.0f, 0.0f, 1.0f);
        face_rect.width = std::clamp(w, 0.0f, 1.0f - face_rect.x);
        face_rect.height = std::clamp(h, 0.0f, 1.0f - face_rect.y);

        confidence = best_score;
        return true;
    }

    /**
     * @brief Face Landmark 실행
     * @param input_data 전처리된 입력 데이터 (192x192 RGB float)
     * @param landmarks 출력 랜드마크 (468 * 3 = 1404 floats)
     * @return 성공 여부
     */
    bool runFaceLandmark(const float* input_data, float* landmarks) {
        if (!face_landmark_interpreter || !input_data || !landmarks) {
            return false;
        }

        // 입력 텐서에 데이터 복사
        float* input_tensor = face_landmark_interpreter->typed_input_tensor<float>(0);
        if (!input_tensor) {
            return false;
        }

        int input_size = FACE_LANDMARK_INPUT_WIDTH * FACE_LANDMARK_INPUT_HEIGHT *
                         FACE_LANDMARK_INPUT_CHANNELS;
        std::memcpy(input_tensor, input_data, input_size * sizeof(float));

        // 추론 실행
        if (face_landmark_interpreter->Invoke() != kTfLiteOk) {
            return false;
        }

        // 출력 텐서 읽기
        float* output_data = face_landmark_interpreter->typed_output_tensor<float>(0);
        if (!output_data) {
            return false;
        }

        // 랜드마크 복사 (468 * 3 = 1404 values)
        std::memcpy(landmarks, output_data, FACE_LANDMARK_COUNT * 3 * sizeof(float));

        return true;
    }

    /**
     * @brief Iris Landmark 실행
     * @param input_data 전처리된 입력 데이터 (64x64 RGB float)
     * @param landmarks 출력 랜드마크 (5 * 3 = 15 floats)
     * @return 성공 여부
     */
    bool runIrisLandmark(const float* input_data, float* landmarks) {
        if (!iris_landmark_interpreter || !input_data || !landmarks) {
            return false;
        }

        // 입력 텐서에 데이터 복사
        float* input_tensor = iris_landmark_interpreter->typed_input_tensor<float>(0);
        if (!input_tensor) {
            return false;
        }

        int input_size = IRIS_LANDMARK_INPUT_WIDTH * IRIS_LANDMARK_INPUT_HEIGHT *
                         IRIS_LANDMARK_INPUT_CHANNELS;
        std::memcpy(input_tensor, input_data, input_size * sizeof(float));

        // 추론 실행
        if (iris_landmark_interpreter->Invoke() != kTfLiteOk) {
            return false;
        }

        // 출력 텐서 읽기
        float* output_data = iris_landmark_interpreter->typed_output_tensor<float>(0);
        if (!output_data) {
            return false;
        }

        // 랜드마크 복사 (5 * 3 = 15 values)
        std::memcpy(landmarks, output_data, IRIS_LANDMARK_COUNT * 3 * sizeof(float));

        return true;
    }

    /**
     * @brief 홍채 반지름 계산
     * @param iris_landmarks 홍채 랜드마크 (5개, 정규화 좌표)
     * @param frame_width 프레임 너비
     * @param frame_height 프레임 높이
     * @return 반지름 (픽셀)
     */
    float calculateIrisRadius(const IrisLandmark* iris_landmarks, int frame_width, int frame_height) {
        // 중심점 (인덱스 0)에서 경계점들 (인덱스 1-4)까지의 평균 거리
        float center_x = iris_landmarks[0].x * frame_width;
        float center_y = iris_landmarks[0].y * frame_height;

        float total_dist = 0.0f;
        for (int i = 1; i < 5; ++i) {
            float px = iris_landmarks[i].x * frame_width;
            float py = iris_landmarks[i].y * frame_height;
            float dx = px - center_x;
            float dy = py - center_y;
            total_dist += std::sqrt(dx * dx + dy * dy);
        }

        return total_dist / 4.0f;
    }
#endif  // IRIS_SDK_HAS_TFLITE && IRIS_SDK_HAS_OPENCV
};

// ============================================================
// 생성자/소멸자
// ============================================================

MediaPipeDetector::MediaPipeDetector()
    : impl_(std::make_unique<Impl>()) {
}

MediaPipeDetector::~MediaPipeDetector() = default;

// ============================================================
// IrisDetector 인터페이스 구현
// ============================================================

bool MediaPipeDetector::initialize(const std::string& model_path) {
    // 이미 초기화된 경우 실패
    if (impl_->initialized) {
        return false;
    }

    // 빈 경로 검사
    if (model_path.empty()) {
        return false;
    }

    // 모델 경로 유효성 검사
    if (!impl_->validateModelPath(model_path)) {
        return false;
    }

#ifdef IRIS_SDK_HAS_TFLITE
    // TensorFlow Lite 모델 로딩
    if (!impl_->loadAllModels(model_path)) {
        impl_->releaseAllModels();
        return false;
    }
#endif

#if defined(IRIS_SDK_HAS_TFLITE) && defined(IRIS_SDK_HAS_OPENCV)
    // 성능 최적화: 사전 할당 버퍼 초기화
    impl_->initializeBuffers();
    impl_->resetTrackingCache();
#endif

    impl_->model_path = model_path;
    impl_->initialized = true;

    return true;
}

IrisResult MediaPipeDetector::detect(const uint8_t* frame_data,
                                      int width,
                                      int height,
                                      FrameFormat format) {
    IrisResult result{};
    result.detected = false;
    result.left_detected = false;
    result.right_detected = false;
    result.confidence = 0.0f;
    result.frame_width = width;
    result.frame_height = height;

    // 디버그: 초기화 상태 및 조건부 컴파일 매크로 확인
    static bool init_debug_printed = false;
    if (!init_debug_printed) {
        std::fprintf(stderr, "[DEBUG] detect() entry point\n");
        std::fprintf(stderr, "[DEBUG] impl_->initialized = %s\n",
                    impl_->initialized ? "true" : "false");
#if defined(IRIS_SDK_HAS_TFLITE)
        std::fprintf(stderr, "[DEBUG] IRIS_SDK_HAS_TFLITE = defined\n");
#else
        std::fprintf(stderr, "[DEBUG] IRIS_SDK_HAS_TFLITE = NOT defined\n");
#endif
#if defined(IRIS_SDK_HAS_OPENCV)
        std::fprintf(stderr, "[DEBUG] IRIS_SDK_HAS_OPENCV = defined\n");
#else
        std::fprintf(stderr, "[DEBUG] IRIS_SDK_HAS_OPENCV = NOT defined\n");
#endif
        init_debug_printed = true;
    }

    // 초기화 상태 검사
    if (!impl_->initialized) {
        return result;
    }

    // 프레임 데이터 유효성 검사
    if (frame_data == nullptr) {
        return result;
    }

    // 크기 유효성 검사
    if (width <= 0 || height <= 0) {
        return result;
    }

#if defined(IRIS_SDK_HAS_TFLITE) && defined(IRIS_SDK_HAS_OPENCV)
    // =========================================================
    // 성능 최적화: 추적 모드 확인
    // 이전 프레임 결과가 있고 추적 모드가 활성화된 경우
    // Face Detection 스킵 가능 (얼굴 위치가 크게 변하지 않음)
    // =========================================================
    bool skip_face_detection = false;
    Rect face_rect{};
    float face_confidence = 0.0f;

    if (impl_->use_tracking && impl_->has_prev_result && impl_->prev_result.detected) {
        // 이전 얼굴 영역을 약간 확장하여 사용
        face_rect = impl_->prev_face_rect;
        face_rect.x = std::max(0.0f, face_rect.x - face_rect.width * 0.1f);
        face_rect.y = std::max(0.0f, face_rect.y - face_rect.height * 0.1f);
        face_rect.width = std::min(1.0f - face_rect.x, face_rect.width * 1.2f);
        face_rect.height = std::min(1.0f - face_rect.y, face_rect.height * 1.2f);
        face_confidence = impl_->prev_result.confidence;
        skip_face_detection = true;
    }

    // =========================================================
    // 1. Face Detection: 얼굴 바운딩 박스 검출
    // (추적 모드에서 스킵 가능)
    // =========================================================
    static bool detect_debug_printed = false;
    if (!detect_debug_printed) {
        std::fprintf(stderr, "[DEBUG] detect() called: %dx%d, format=%d\n",
                    width, height, static_cast<int>(format));
        detect_debug_printed = true;
    }

    if (!skip_face_detection) {
        // 사전 할당 버퍼 사용 (메모리 재할당 방지)
        if (!impl_->preprocessImage(frame_data, width, height, format,
                                    FACE_DETECTION_INPUT_WIDTH,
                                    FACE_DETECTION_INPUT_HEIGHT,
                                    impl_->face_detection_input_buffer.data())) {
            static bool preprocess_fail_printed = false;
            if (!preprocess_fail_printed) {
                std::fprintf(stderr, "[DEBUG] preprocessImage failed!\n");
                preprocess_fail_printed = true;
            }
            impl_->resetTrackingCache();
            return result;
        }

        static bool preprocess_ok_printed = false;
        if (!preprocess_ok_printed) {
            std::fprintf(stderr, "[DEBUG] preprocessImage succeeded, calling runFaceDetection\n");
            preprocess_ok_printed = true;
        }

        if (!impl_->runFaceDetection(impl_->face_detection_input_buffer.data(),
                                     face_rect, face_confidence)) {
            // 얼굴 미검출 시 추적 캐시 초기화
            static bool facedet_fail_printed = false;
            if (!facedet_fail_printed) {
                std::fprintf(stderr, "[DEBUG] runFaceDetection failed (no face found)\n");
                facedet_fail_printed = true;
            }
            impl_->resetTrackingCache();
            return result;
        }

        static bool facedet_ok_printed = false;
        if (!facedet_ok_printed) {
            std::fprintf(stderr, "[DEBUG] runFaceDetection succeeded: face_rect=(%.2f,%.2f,%.2f,%.2f), conf=%.2f\n",
                        face_rect.x, face_rect.y, face_rect.width, face_rect.height, face_confidence);
            facedet_ok_printed = true;
        }
    }

    result.face_rect = face_rect;

    // =========================================================
    // 2. RGB 이미지 변환 (한 번만 수행, 이후 재사용)
    // =========================================================
    cv::Mat rgb_mat;
    if (!impl_->convertToRgb(frame_data, width, height, format, rgb_mat)) {
        impl_->resetTrackingCache();
        return result;
    }

    // =========================================================
    // 3. Face Landmark: 468개 얼굴 랜드마크 추출
    // 성능 최적화: 얼굴 영역만 크롭하여 모델에 입력
    // =========================================================
    // 얼굴 영역 크롭 후 전처리 (사전 할당 버퍼 사용)
    if (!impl_->cropFaceRegion(rgb_mat, face_rect,
                               FACE_LANDMARK_INPUT_WIDTH,
                               FACE_LANDMARK_INPUT_HEIGHT,
                               impl_->face_landmark_input_buffer.data())) {
        // 크롭 실패 시 전체 이미지 사용 (폴백)
        if (!impl_->preprocessImage(frame_data, width, height, format,
                                    FACE_LANDMARK_INPUT_WIDTH,
                                    FACE_LANDMARK_INPUT_HEIGHT,
                                    impl_->face_landmark_input_buffer.data())) {
            impl_->resetTrackingCache();
            return result;
        }
    }

    // 사전 할당 버퍼 사용
    if (!impl_->runFaceLandmark(impl_->face_landmark_input_buffer.data(),
                                impl_->face_landmarks_buffer.data())) {
        impl_->resetTrackingCache();
        return result;
    }

    // Face Landmark 좌표 변환: 크롭된 얼굴 영역 → 전체 이미지 좌표
    // Face Landmark 모델은 192x192 픽셀 좌표를 출력
    // 1. 먼저 픽셀 좌표 → 정규화 좌표 (0-1)
    // 2. 그 다음 face_rect 기준 → 전체 이미지 기준으로 변환
    for (int i = 0; i < FACE_LANDMARK_COUNT; ++i) {
        // 픽셀 좌표 → 정규화 좌표 (0-1 범위)
        float local_x = impl_->face_landmarks_buffer[i * 3 + 0] /
                        static_cast<float>(FACE_LANDMARK_INPUT_WIDTH);
        float local_y = impl_->face_landmarks_buffer[i * 3 + 1] /
                        static_cast<float>(FACE_LANDMARK_INPUT_HEIGHT);
        // z 좌표는 변환 없이 유지

        // 크롭 영역 내 좌표를 전체 이미지 좌표로 변환
        impl_->face_landmarks_buffer[i * 3 + 0] = face_rect.x + local_x * face_rect.width;
        impl_->face_landmarks_buffer[i * 3 + 1] = face_rect.y + local_y * face_rect.height;
    }

    // =========================================================
    // 4. Iris Landmark: 왼쪽 눈 (사전 할당 버퍼 사용)
    // =========================================================
    Rect left_eye_crop{};
    if (impl_->extractEyeRegion(rgb_mat, impl_->face_landmarks_buffer.data(),
                                LEFT_EYE_INDICES, sizeof(LEFT_EYE_INDICES) / sizeof(int),
                                IRIS_LANDMARK_INPUT_WIDTH,
                                impl_->left_iris_input_buffer.data(),
                                left_eye_crop)) {
        if (impl_->runIrisLandmark(impl_->left_iris_input_buffer.data(),
                                   impl_->left_iris_landmarks_buffer.data())) {
            // 크롭 좌표를 원본 이미지 좌표로 변환
            // 참고: Iris Landmark 모델은 64x64 픽셀 좌표를 출력하므로 정규화 필요
            for (int i = 0; i < IRIS_LANDMARK_COUNT; ++i) {
                // 픽셀 좌표 → 정규화 좌표 (0-1)
                float local_x = impl_->left_iris_landmarks_buffer[i * 3 + 0] /
                                static_cast<float>(IRIS_LANDMARK_INPUT_WIDTH);
                float local_y = impl_->left_iris_landmarks_buffer[i * 3 + 1] /
                                static_cast<float>(IRIS_LANDMARK_INPUT_HEIGHT);
                float local_z = impl_->left_iris_landmarks_buffer[i * 3 + 2];

                // 크롭 영역 내 좌표를 원본 이미지 좌표로 변환
                result.left_iris[i].x = left_eye_crop.x + local_x * left_eye_crop.width;
                result.left_iris[i].y = left_eye_crop.y + local_y * left_eye_crop.height;
                result.left_iris[i].z = local_z;
                result.left_iris[i].visibility = 1.0f;
            }
            result.left_detected = true;
            result.left_radius = impl_->calculateIrisRadius(result.left_iris, width, height);
        }
    }

    // =========================================================
    // 5. Iris Landmark: 오른쪽 눈 (사전 할당 버퍼 사용)
    // =========================================================
    Rect right_eye_crop{};
    if (impl_->extractEyeRegion(rgb_mat, impl_->face_landmarks_buffer.data(),
                                RIGHT_EYE_INDICES, sizeof(RIGHT_EYE_INDICES) / sizeof(int),
                                IRIS_LANDMARK_INPUT_WIDTH,
                                impl_->right_iris_input_buffer.data(),
                                right_eye_crop)) {
        if (impl_->runIrisLandmark(impl_->right_iris_input_buffer.data(),
                                   impl_->right_iris_landmarks_buffer.data())) {
            // 크롭 좌표를 원본 이미지 좌표로 변환
            // 참고: Iris Landmark 모델은 64x64 픽셀 좌표를 출력하므로 정규화 필요
            for (int i = 0; i < IRIS_LANDMARK_COUNT; ++i) {
                // 픽셀 좌표 → 정규화 좌표 (0-1)
                float local_x = impl_->right_iris_landmarks_buffer[i * 3 + 0] /
                                static_cast<float>(IRIS_LANDMARK_INPUT_WIDTH);
                float local_y = impl_->right_iris_landmarks_buffer[i * 3 + 1] /
                                static_cast<float>(IRIS_LANDMARK_INPUT_HEIGHT);
                float local_z = impl_->right_iris_landmarks_buffer[i * 3 + 2];

                result.right_iris[i].x = right_eye_crop.x + local_x * right_eye_crop.width;
                result.right_iris[i].y = right_eye_crop.y + local_y * right_eye_crop.height;
                result.right_iris[i].z = local_z;
                result.right_iris[i].visibility = 1.0f;
            }
            result.right_detected = true;
            result.right_radius = impl_->calculateIrisRadius(result.right_iris, width, height);
        }
    }

    // =========================================================
    // 6. 최종 결과 구성
    // =========================================================
    result.detected = result.left_detected || result.right_detected;
    if (result.detected) {
        // 신뢰도: 얼굴 신뢰도와 검출된 눈 수 기반
        float eye_factor = 0.5f;
        if (result.left_detected && result.right_detected) {
            eye_factor = 1.0f;
        }
        result.confidence = face_confidence * eye_factor;
    }

    // 얼굴 회전 추정 (간단한 버전: 코, 눈 위치 기반)
    // 실제로는 PnP 알고리즘 등 사용 권장
    result.face_rotation[0] = 0.0f;  // pitch
    result.face_rotation[1] = 0.0f;  // yaw
    result.face_rotation[2] = 0.0f;  // roll

    // =========================================================
    // 7. 추적 캐시 업데이트
    // =========================================================
    if (result.detected) {
        impl_->prev_result = result;
        impl_->prev_face_rect = face_rect;
        impl_->has_prev_result = true;
    } else {
        // 검출 실패 시 추적 캐시 무효화
        impl_->resetTrackingCache();
    }

#endif  // IRIS_SDK_HAS_TFLITE && IRIS_SDK_HAS_OPENCV

    return result;
}

void MediaPipeDetector::release() {
#ifdef IRIS_SDK_HAS_TFLITE
    impl_->releaseAllModels();
#endif
    impl_->initialized = false;
    impl_->model_path.clear();
}

bool MediaPipeDetector::isInitialized() const {
    return impl_->initialized;
}

DetectorType MediaPipeDetector::getDetectorType() const {
    return DetectorType::MediaPipe;
}

// ============================================================
// MediaPipe 전용 설정
// ============================================================

void MediaPipeDetector::setMinDetectionConfidence(float confidence) {
    impl_->min_detection_confidence = std::clamp(confidence, 0.0f, 1.0f);
}

void MediaPipeDetector::setMinTrackingConfidence(float confidence) {
    impl_->min_tracking_confidence = std::clamp(confidence, 0.0f, 1.0f);
}

void MediaPipeDetector::setNumFaces(int num_faces) {
    // 최소 1개 이상
    impl_->num_faces = std::max(1, num_faces);
}

// ============================================================
// 성능 최적화 설정
// ============================================================

void MediaPipeDetector::setNumThreads(int num_threads) {
    // 최소 1개 이상, 최대 16개
    impl_->num_threads = std::clamp(num_threads, 1, 16);
}

void MediaPipeDetector::setTrackingEnabled(bool enable) {
    impl_->use_tracking = enable;
    if (!enable) {
        // 추적 비활성화 시 캐시도 초기화
#if defined(IRIS_SDK_HAS_TFLITE) && defined(IRIS_SDK_HAS_OPENCV)
        impl_->resetTrackingCache();
#endif
    }
}

void MediaPipeDetector::resetTracking() {
#if defined(IRIS_SDK_HAS_TFLITE) && defined(IRIS_SDK_HAS_OPENCV)
    impl_->resetTrackingCache();
#endif
}

} // namespace iris_sdk
