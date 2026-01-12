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

    // MediaPipe iris_landmark 모델 출력 구조:
    // - 총 71개 랜드마크 (71 * 3 = 213 floats)
    // - 인덱스 0-67: 눈 윤곽 (Eye Contour)
    // - 인덱스 68-72: 홍채 (Iris)
    //   - 68: 홍채 중심
    //   - 69-72: 홍채 경계점 (상, 하, 좌, 우)
    constexpr int IRIS_MODEL_OUTPUT_COUNT = 71;  // 전체 출력 랜드마크
    constexpr int IRIS_START_INDEX = 68;         // 홍채 시작 인덱스

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

    // ========================================
    // 눈 모서리 랜드마크 인덱스 (공식 MediaPipe 방식)
    // iris_landmark 모델의 ROI 계산에 사용
    // ========================================
    // 왼쪽 눈: 안쪽 모서리(133), 바깥쪽 모서리(33)
    constexpr int LEFT_EYE_INNER_CORNER = 133;
    constexpr int LEFT_EYE_OUTER_CORNER = 33;
    // 오른쪽 눈: 안쪽 모서리(362), 바깥쪽 모서리(263)
    constexpr int RIGHT_EYE_INNER_CORNER = 362;
    constexpr int RIGHT_EYE_OUTER_CORNER = 263;

    // 공식 MediaPipe iris 모델의 ROI 확대 비율 (2.3x)
    constexpr float IRIS_ROI_SCALE = 2.3f;

    // ========================================
    // Face Landmark V2 모델 상수 (FaceMesh V2)
    // V2는 홍채 랜드마크가 내장되어 있음 (478개 = 468 + 10)
    // ========================================
    constexpr int FACE_LANDMARK_V2_INPUT_WIDTH = 256;
    constexpr int FACE_LANDMARK_V2_INPUT_HEIGHT = 256;
    constexpr int FACE_LANDMARK_V2_COUNT = 478;

    // V2 모델 내장 홍채 인덱스 (Face Landmark 출력에서 직접 추출)
    // 왼쪽 눈 홍채: 인덱스 468-472 (중심 + 4개 경계점)
    constexpr int V2_LEFT_IRIS_CENTER = 468;
    constexpr int V2_LEFT_IRIS_INDICES[] = {468, 469, 470, 471, 472};
    // 오른쪽 눈 홍채: 인덱스 473-477 (중심 + 4개 경계점)
    constexpr int V2_RIGHT_IRIS_CENTER = 473;
    constexpr int V2_RIGHT_IRIS_INDICES[] = {473, 474, 475, 476, 477};
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
    float min_detection_confidence = 0.3f;  // 기본값 낮춤 (더 많은 후보 검출)
    float min_tracking_confidence = 0.5f;
    int num_faces = 1;

    // ========================================
    // 성능 최적화: 설정
    // ========================================
    int num_threads = 4;           ///< TFLite 추론 스레드 수
    bool use_tracking = true;      ///< 추적 모드 활성화 여부
    float tracking_iou_threshold = 0.5f;  ///< 추적 유지 IoU 임계값

    // ========================================
    // 모델 버전 (V1: 192x192, V2: 256x256)
    // ========================================
    int model_version = 1;  ///< 1: V1 (face_landmark.tflite), 2: V2 (face_landmark_v2.tflite)

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
     *
     * @note 모델 버전에 따라 버퍼 크기가 달라짐
     *       - V1: 192x192 입력, 468 랜드마크
     *       - V2: 256x256 입력, 478 랜드마크 (홍채 내장)
     */
    void initializeBuffers() {
        // Face Detection 입력 버퍼 (모든 버전 동일)
        face_detection_input_buffer.resize(
            FACE_DETECTION_INPUT_WIDTH * FACE_DETECTION_INPUT_HEIGHT *
            FACE_DETECTION_INPUT_CHANNELS);

        // Face Landmark 입력/출력 버퍼 (모델 버전에 따라 크기 결정)
        if (model_version == 2) {
            // V2 모델: 256x256 입력, 478 랜드마크
            face_landmark_input_buffer.resize(
                FACE_LANDMARK_V2_INPUT_WIDTH * FACE_LANDMARK_V2_INPUT_HEIGHT *
                FACE_LANDMARK_INPUT_CHANNELS);
            face_landmarks_buffer.resize(FACE_LANDMARK_V2_COUNT * 3);
        } else {
            // V1 모델: 192x192 입력, 468 랜드마크
            face_landmark_input_buffer.resize(
                FACE_LANDMARK_INPUT_WIDTH * FACE_LANDMARK_INPUT_HEIGHT *
                FACE_LANDMARK_INPUT_CHANNELS);
            face_landmarks_buffer.resize(FACE_LANDMARK_COUNT * 3);
        }

        // Iris Landmark 입력/출력 버퍼 (V1에서만 사용)
        if (model_version == 1) {
            left_iris_input_buffer.resize(
                IRIS_LANDMARK_INPUT_WIDTH * IRIS_LANDMARK_INPUT_HEIGHT *
                IRIS_LANDMARK_INPUT_CHANNELS);
            right_iris_input_buffer.resize(
                IRIS_LANDMARK_INPUT_WIDTH * IRIS_LANDMARK_INPUT_HEIGHT *
                IRIS_LANDMARK_INPUT_CHANNELS);
        } else {
            // V2 모델에서는 iris 입력 버퍼 불필요
            left_iris_input_buffer.clear();
            right_iris_input_buffer.clear();
        }

        // 홍채 랜드마크 출력 버퍼 (모든 버전에서 5개 랜드마크)
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

        // ========================================
        // Face Landmark 모델 로드 (V2 우선, V1 폴백)
        // V2 모델: face_landmark_v2.tflite (256x256, 478 랜드마크, 홍채 내장)
        // V1 모델: face_landmark.tflite (192x192, 468 랜드마크)
        // ========================================
        std::string face_landmark_v2_path = (base / "face_landmark_v2.tflite").string();
        std::string face_landmark_v1_path = (base / REQUIRED_MODELS[1]).string();

        if (std::filesystem::exists(face_landmark_v2_path) &&
            loadModel(face_landmark_v2_path, face_landmark_model, face_landmark_interpreter)) {
            // V2 모델 로드 성공
            model_version = 2;
            std::fprintf(stderr, "[INFO] Face Landmark V2 model loaded (256x256, 478 landmarks)\n");
        } else if (loadModel(face_landmark_v1_path, face_landmark_model, face_landmark_interpreter)) {
            // V1 모델로 폴백
            model_version = 1;
            std::fprintf(stderr, "[INFO] Face Landmark V1 model loaded (192x192, 468 landmarks)\n");
        } else {
            return false;
        }

        // Face Landmark 텐서 인덱스 설정
        if (face_landmark_interpreter->inputs().empty() ||
            face_landmark_interpreter->outputs().empty()) {
            return false;  // 모델 구조 오류
        }
        face_landmark_input_index = face_landmark_interpreter->inputs()[0];
        face_landmark_output_index = face_landmark_interpreter->outputs()[0];

        // ========================================
        // Iris Landmark 모델 로드 (V1에서만 필요)
        // V2 모델은 Face Landmark 출력에 홍채가 내장되어 있음
        // ========================================
        if (model_version == 1) {
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
        } else {
            // V2 모델은 별도 iris_landmark 모델 불필요
            iris_landmark_model.reset();
            iris_landmark_interpreter.reset();
            iris_landmark_input_index = -1;
            iris_landmark_output_index = -1;
            std::fprintf(stderr, "[INFO] V2 model: iris_landmark model not required (embedded in face_landmark)\n");
        }

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
     * @brief 눈 영역 이미지 추출 (크롭) - 공식 MediaPipe 방식
     *
     * 공식 MediaPipe iris_landmark 모델의 전처리 방식을 따름:
     * 1. 눈 모서리 두 점(inner, outer)의 중심을 ROI 중심으로 사용
     * 2. 두 점 사이 거리 * 2.3을 ROI 크기로 사용
     * 3. 오른쪽 눈은 수평 반전하여 입력 (flip_horizontal)
     *
     * @param rgb_image RGB 이미지 (CV_8UC3)
     * @param landmarks 얼굴 랜드마크 (468 * 3)
     * @param inner_corner_idx 눈 안쪽 모서리 랜드마크 인덱스
     * @param outer_corner_idx 눈 바깥쪽 모서리 랜드마크 인덱스
     * @param output_size 출력 크기 (정사각형)
     * @param output 출력 버퍼 (정규화된 float)
     * @param crop_rect 크롭된 영역 (출력, 정규화 좌표)
     * @param flip_horizontal 수평 반전 여부 (오른쪽 눈의 경우 true)
     * @return 성공 여부
     */
    bool extractEyeRegionMediaPipe(const cv::Mat& rgb_image, const float* landmarks,
                                    int inner_corner_idx, int outer_corner_idx,
                                    int output_size, float* output, Rect& crop_rect,
                                    bool flip_horizontal = false) {
        if (!landmarks || !output) {
            return false;
        }

        int img_width = rgb_image.cols;
        int img_height = rgb_image.rows;

        // ========================================
        // 공식 MediaPipe 방식: 눈 모서리 두 점 기반 ROI 계산
        // ========================================
        // 눈 안쪽 모서리 좌표
        float inner_x = landmarks[inner_corner_idx * 3 + 0];
        float inner_y = landmarks[inner_corner_idx * 3 + 1];
        // 눈 바깥쪽 모서리 좌표
        float outer_x = landmarks[outer_corner_idx * 3 + 0];
        float outer_y = landmarks[outer_corner_idx * 3 + 1];

        // 두 점의 중심 = ROI 중심
        float center_x = (inner_x + outer_x) / 2.0f;
        float center_y = (inner_y + outer_y) / 2.0f;

        // 두 점 사이 거리 계산
        float dx = outer_x - inner_x;
        float dy = outer_y - inner_y;
        float eye_width = std::sqrt(dx * dx + dy * dy);

        // ROI 크기 = 눈 너비 * 2.3 (공식 MediaPipe 설정)
        float roi_size = eye_width * IRIS_ROI_SCALE;

        // 정사각형 ROI 경계 계산
        float half_size = roi_size / 2.0f;
        float min_x = center_x - half_size;
        float min_y = center_y - half_size;
        float max_x = center_x + half_size;
        float max_y = center_y + half_size;

        // 경계를 벗어나는 경우 클램핑 (원본 이미지 범위 내로)
        min_x = std::max(0.0f, min_x);
        min_y = std::max(0.0f, min_y);
        max_x = std::min(1.0f, max_x);
        max_y = std::min(1.0f, max_y);

        // 크롭 후에도 정사각형 유지
        float actual_width = max_x - min_x;
        float actual_height = max_y - min_y;
        float actual_size = std::min(actual_width, actual_height);

        // 중심 기준으로 재조정
        min_x = center_x - actual_size / 2.0f;
        min_y = center_y - actual_size / 2.0f;
        max_x = center_x + actual_size / 2.0f;
        max_y = center_y + actual_size / 2.0f;

        // 다시 클램핑
        min_x = std::max(0.0f, min_x);
        min_y = std::max(0.0f, min_y);
        max_x = std::min(1.0f, max_x);
        max_y = std::min(1.0f, max_y);

        // 크롭 영역 저장 (좌표 변환에 사용)
        crop_rect.x = min_x;
        crop_rect.y = min_y;
        crop_rect.width = max_x - min_x;
        crop_rect.height = max_y - min_y;

        // 디버그: ROI 계산 결과 확인
        static bool roi_debug_printed = false;
        if (!roi_debug_printed) {
            std::fprintf(stderr, "[DEBUG] MediaPipe Eye ROI Calculation:\n");
            std::fprintf(stderr, "  inner(%d): (%.4f, %.4f)\n", inner_corner_idx, inner_x, inner_y);
            std::fprintf(stderr, "  outer(%d): (%.4f, %.4f)\n", outer_corner_idx, outer_x, outer_y);
            std::fprintf(stderr, "  center: (%.4f, %.4f)\n", center_x, center_y);
            std::fprintf(stderr, "  eye_width: %.4f, roi_size: %.4f (scale=%.1f)\n",
                        eye_width, roi_size, IRIS_ROI_SCALE);
            std::fprintf(stderr, "  crop_rect: x=%.4f, y=%.4f, w=%.4f, h=%.4f\n",
                        crop_rect.x, crop_rect.y, crop_rect.width, crop_rect.height);
            std::fprintf(stderr, "  flip_horizontal: %s\n", flip_horizontal ? "true" : "false");
            roi_debug_printed = true;
        }

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

        // 최소 크기 보장
        if (px_max_x - px_min_x < 4 || px_max_y - px_min_y < 4) {
            return false;
        }

        // 크롭 (ROI는 복사 없이 참조)
        cv::Rect roi(px_min_x, px_min_y, px_max_x - px_min_x, px_max_y - px_min_y);
        cv::Mat cropped = rgb_image(roi);

        // 리사이즈 (cropped_buffer 재사용)
        cv::resize(cropped, cropped_buffer, cv::Size(output_size, output_size),
                  0, 0, cv::INTER_LINEAR);

        // ========================================
        // 공식 MediaPipe 방식: 오른쪽 눈 수평 반전
        // ========================================
        if (flip_horizontal) {
            cv::flip(cropped_buffer, cropped_buffer, 1);  // 1 = 수평 반전
        }

        // OpenCV SIMD 연산으로 정규화
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
     * @brief 눈 영역 이미지 추출 (크롭) - 레거시 버전 (호환성 유지)
     *
     * @deprecated 새 코드에서는 extractEyeRegionMediaPipe() 사용 권장
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
     * @param actual_crop_rect 실제 크롭된 영역 (정규화 좌표, 출력)
     * @return 성공 여부
     */
    bool cropFaceRegion(const cv::Mat& rgb_image, const Rect& face_rect,
                        int target_width, int target_height, float* output,
                        Rect& actual_crop_rect) {
        if (!output || face_rect.width <= 0 || face_rect.height <= 0) {
            return false;
        }

        int img_width = rgb_image.cols;
        int img_height = rgb_image.rows;

        // ========================================
        // 픽셀 기준 정사각형 crop 영역 생성
        // 이미지 종횡비와 관계없이 정사각형이 되도록 함
        // ========================================

        // 얼굴 영역을 픽셀 단위로 변환
        float face_center_x = (face_rect.x + face_rect.width / 2.0f) * img_width;
        float face_center_y = (face_rect.y + face_rect.height / 2.0f) * img_height;
        float face_width_px = face_rect.width * img_width;
        float face_height_px = face_rect.height * img_height;

        // 정사각형 크기 = 더 큰 쪽 기준 + 20% 마진
        float square_size = std::max(face_width_px, face_height_px) * 1.2f;

        // 정사각형 crop 영역 (픽셀 단위)
        int px_min_x = static_cast<int>(face_center_x - square_size / 2.0f);
        int px_min_y = static_cast<int>(face_center_y - square_size / 2.0f);
        int px_max_x = static_cast<int>(face_center_x + square_size / 2.0f);
        int px_max_y = static_cast<int>(face_center_y + square_size / 2.0f);

        // 경계 체크 및 조정 (정사각형 유지하면서)
        if (px_min_x < 0) {
            px_max_x -= px_min_x;
            px_min_x = 0;
        }
        if (px_min_y < 0) {
            px_max_y -= px_min_y;
            px_min_y = 0;
        }
        if (px_max_x > img_width) {
            px_min_x -= (px_max_x - img_width);
            px_max_x = img_width;
        }
        if (px_max_y > img_height) {
            px_min_y -= (px_max_y - img_height);
            px_max_y = img_height;
        }

        // 최종 경계 체크
        px_min_x = std::clamp(px_min_x, 0, img_width - 1);
        px_min_y = std::clamp(px_min_y, 0, img_height - 1);
        px_max_x = std::clamp(px_max_x, 1, img_width);
        px_max_y = std::clamp(px_max_y, 1, img_height);

        // 실제 크롭 영역 저장 (정규화 좌표, 좌표 변환에 사용)
        // 주의: 정사각형 픽셀 영역을 정규화 좌표로 변환하면 직사각형이 됨
        actual_crop_rect.x = static_cast<float>(px_min_x) / img_width;
        actual_crop_rect.y = static_cast<float>(px_min_y) / img_height;
        actual_crop_rect.width = static_cast<float>(px_max_x - px_min_x) / img_width;
        actual_crop_rect.height = static_cast<float>(px_max_y - px_min_y) / img_height;

        // 크롭 (픽셀 기준 정사각형)
        cv::Rect roi(px_min_x, px_min_y, px_max_x - px_min_x, px_max_y - px_min_y);
        cv::Mat cropped = rgb_image(roi);

        // 디버그: 픽셀 기준 정사각형 crop 확인
        static bool crop_debug = false;
        if (!crop_debug) {
            std::fprintf(stderr, "[DEBUG] Square Crop (pixel-based):\n");
            std::fprintf(stderr, "  face_rect: x=%.4f, y=%.4f, w=%.4f, h=%.4f\n",
                        face_rect.x, face_rect.y, face_rect.width, face_rect.height);
            std::fprintf(stderr, "  face_center (px): (%.1f, %.1f)\n", face_center_x, face_center_y);
            std::fprintf(stderr, "  face_size (px): %.1f x %.1f\n", face_width_px, face_height_px);
            std::fprintf(stderr, "  square_size (px): %.1f\n", square_size);
            std::fprintf(stderr, "  crop_roi (px): x=%d, y=%d, w=%d, h=%d\n",
                        roi.x, roi.y, roi.width, roi.height);
            std::fprintf(stderr, "  actual_crop_rect (norm): x=%.4f, y=%.4f, w=%.4f, h=%.4f\n",
                        actual_crop_rect.x, actual_crop_rect.y,
                        actual_crop_rect.width, actual_crop_rect.height);
            crop_debug = true;
        }

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
     * @brief Face Landmark 실행 (V1/V2 모델 지원)
     * @param input_data 전처리된 입력 데이터 (V1: 192x192, V2: 256x256 RGB float)
     * @param landmarks 출력 랜드마크 (V1: 468 * 3 floats, V2: 478 * 3 floats)
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

        // 모델 버전에 따라 입력 크기 결정
        int input_width = (model_version == 2) ? FACE_LANDMARK_V2_INPUT_WIDTH : FACE_LANDMARK_INPUT_WIDTH;
        int input_height = (model_version == 2) ? FACE_LANDMARK_V2_INPUT_HEIGHT : FACE_LANDMARK_INPUT_HEIGHT;
        int input_size = input_width * input_height * FACE_LANDMARK_INPUT_CHANNELS;
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

        // 모델 버전에 따라 출력 랜드마크 수 결정
        int landmark_count = (model_version == 2) ? FACE_LANDMARK_V2_COUNT : FACE_LANDMARK_COUNT;
        std::memcpy(landmarks, output_data, landmark_count * 3 * sizeof(float));

        return true;
    }

    /**
     * @brief Face Landmark V2 출력에서 홍채 좌표 추출
     *
     * V2 모델은 478개 랜드마크 중 인덱스 468-477에 홍채 좌표가 내장되어 있음.
     * 별도의 iris_landmark 모델 호출 없이 직접 추출 가능.
     *
     * @param face_landmarks Face Landmark V2 출력 (478 * 3 floats, 정규화 좌표)
     * @param crop_rect 크롭 영역 (전체 이미지 기준 정규화 좌표)
     * @param left_iris 왼쪽 홍채 출력 버퍼 (5 * 3 floats)
     * @param right_iris 오른쪽 홍채 출력 버퍼 (5 * 3 floats)
     * @param left_detected 왼쪽 홍채 검출 여부 (출력)
     * @param right_detected 오른쪽 홍채 검출 여부 (출력)
     *
     * @note Face Landmark V2 홍채 인덱스:
     *       - 왼쪽 눈: 468 (중심), 469-472 (경계점)
     *       - 오른쪽 눈: 473 (중심), 474-477 (경계점)
     */
    void extractIrisFromFaceLandmarkV2(const float* face_landmarks,
                                        const Rect& crop_rect,
                                        float* left_iris,
                                        float* right_iris,
                                        bool& left_detected,
                                        bool& right_detected) {
        left_detected = false;
        right_detected = false;

        if (!face_landmarks || !left_iris || !right_iris) {
            return;
        }

        // 디버그: V2 홍채 추출 확인 (최초 1회만)
        static bool v2_iris_debug = false;
        if (!v2_iris_debug) {
            std::fprintf(stderr, "[DEBUG] Extracting iris from Face Landmark V2 (embedded):\n");
            std::fprintf(stderr, "  crop_rect: x=%.4f, y=%.4f, w=%.4f, h=%.4f\n",
                        crop_rect.x, crop_rect.y, crop_rect.width, crop_rect.height);
        }

        // ========================================
        // 왼쪽 눈 홍채 추출 (인덱스 468-472)
        // ========================================
        bool left_valid = true;
        for (int i = 0; i < IRIS_LANDMARK_COUNT; ++i) {
            int idx = V2_LEFT_IRIS_INDICES[i];
            float x = face_landmarks[idx * 3 + 0];
            float y = face_landmarks[idx * 3 + 1];
            float z = face_landmarks[idx * 3 + 2];

            // 좌표가 유효한지 확인 (0~1 범위)
            if (x < 0.0f || x > 1.0f || y < 0.0f || y > 1.0f) {
                left_valid = false;
                break;
            }

            // 크롭 영역 → 전체 이미지 좌표로 변환 (이미 변환된 경우 스킵)
            // face_landmarks는 detect()에서 이미 전체 이미지 좌표로 변환됨
            left_iris[i * 3 + 0] = x;
            left_iris[i * 3 + 1] = y;
            left_iris[i * 3 + 2] = z;

            if (!v2_iris_debug) {
                std::fprintf(stderr, "  left_iris[%d]: idx=%d, x=%.4f, y=%.4f, z=%.4f\n",
                            i, idx, x, y, z);
            }
        }
        left_detected = left_valid;

        // ========================================
        // 오른쪽 눈 홍채 추출 (인덱스 473-477)
        // ========================================
        bool right_valid = true;
        for (int i = 0; i < IRIS_LANDMARK_COUNT; ++i) {
            int idx = V2_RIGHT_IRIS_INDICES[i];
            float x = face_landmarks[idx * 3 + 0];
            float y = face_landmarks[idx * 3 + 1];
            float z = face_landmarks[idx * 3 + 2];

            // 좌표가 유효한지 확인 (0~1 범위)
            if (x < 0.0f || x > 1.0f || y < 0.0f || y > 1.0f) {
                right_valid = false;
                break;
            }

            right_iris[i * 3 + 0] = x;
            right_iris[i * 3 + 1] = y;
            right_iris[i * 3 + 2] = z;

            if (!v2_iris_debug) {
                std::fprintf(stderr, "  right_iris[%d]: idx=%d, x=%.4f, y=%.4f, z=%.4f\n",
                            i, idx, x, y, z);
            }
        }
        right_detected = right_valid;

        if (!v2_iris_debug) {
            std::fprintf(stderr, "  Result: left=%s, right=%s\n",
                        left_detected ? "detected" : "not detected",
                        right_detected ? "detected" : "not detected");
            v2_iris_debug = true;
        }
    }

    /**
     * @brief 홍채 좌표 검증 및 수정
     *
     * V2 모델이 잘못된 홍채 좌표를 출력하는 경우(특히 landscape 이미지),
     * 눈 랜드마크를 사용하여 홍채 중심 위치를 보정합니다.
     *
     * @param face_landmarks Face Landmark 출력 버퍼 (변환된 전체 이미지 좌표)
     * @param left_iris 왼쪽 홍채 랜드마크 (수정될 수 있음)
     * @param right_iris 오른쪽 홍채 랜드마크 (수정될 수 있음)
     * @param max_distance_threshold 눈-홍채 거리 임계값 (정규화 좌표 기준)
     */
    void validateAndFixIrisCoordinates(const float* face_landmarks,
                                       float* left_iris, float* right_iris,
                                       float max_distance_threshold = 0.05f) {
        // 눈 모서리 좌표 가져오기
        float left_eye_inner_x = face_landmarks[LEFT_EYE_INNER_CORNER * 3 + 0];  // idx 133
        float left_eye_inner_y = face_landmarks[LEFT_EYE_INNER_CORNER * 3 + 1];
        float left_eye_outer_x = face_landmarks[LEFT_EYE_OUTER_CORNER * 3 + 0];  // idx 33
        float left_eye_outer_y = face_landmarks[LEFT_EYE_OUTER_CORNER * 3 + 1];

        float right_eye_inner_x = face_landmarks[RIGHT_EYE_INNER_CORNER * 3 + 0];  // idx 362
        float right_eye_inner_y = face_landmarks[RIGHT_EYE_INNER_CORNER * 3 + 1];
        float right_eye_outer_x = face_landmarks[RIGHT_EYE_OUTER_CORNER * 3 + 0];  // idx 263
        float right_eye_outer_y = face_landmarks[RIGHT_EYE_OUTER_CORNER * 3 + 1];

        // 눈 중심 계산 (내측/외측 모서리의 중점)
        float left_eye_center_x = (left_eye_inner_x + left_eye_outer_x) / 2.0f;
        float left_eye_center_y = (left_eye_inner_y + left_eye_outer_y) / 2.0f;
        float right_eye_center_x = (right_eye_inner_x + right_eye_outer_x) / 2.0f;
        float right_eye_center_y = (right_eye_inner_y + right_eye_outer_y) / 2.0f;

        // 현재 홍채 중심 위치
        float left_iris_x = left_iris[0];
        float left_iris_y = left_iris[1];
        float right_iris_x = right_iris[0];
        float right_iris_y = right_iris[1];

        // 눈 중심과 홍채 중심의 거리 계산
        float left_distance = std::sqrt(
            std::pow(left_iris_x - left_eye_center_x, 2.0f) +
            std::pow(left_iris_y - left_eye_center_y, 2.0f));
        float right_distance = std::sqrt(
            std::pow(right_iris_x - right_eye_center_x, 2.0f) +
            std::pow(right_iris_y - right_eye_center_y, 2.0f));

        static bool fix_debug_printed = false;

        // 왼쪽 홍채 검증 및 수정
        if (left_distance > max_distance_threshold) {
            if (!fix_debug_printed) {
                std::fprintf(stderr, "[DEBUG] Iris position fix (left):\n");
                std::fprintf(stderr, "  Eye center: (%.4f, %.4f)\n", left_eye_center_x, left_eye_center_y);
                std::fprintf(stderr, "  Iris (before): (%.4f, %.4f), distance=%.4f\n",
                            left_iris_x, left_iris_y, left_distance);
            }
            // 홍채 중심을 눈 중심으로 이동
            left_iris[0] = left_eye_center_x;
            left_iris[1] = left_eye_center_y;
            // 경계점들도 눈 중심 기준으로 조정 (반지름 유지)
            float offset_x = left_eye_center_x - left_iris_x;
            float offset_y = left_eye_center_y - left_iris_y;
            for (int i = 1; i < IRIS_LANDMARK_COUNT; ++i) {
                left_iris[i * 3 + 0] += offset_x;
                left_iris[i * 3 + 1] += offset_y;
            }
            if (!fix_debug_printed) {
                std::fprintf(stderr, "  Iris (after): (%.4f, %.4f)\n", left_iris[0], left_iris[1]);
            }
        }

        // 오른쪽 홍채 검증 및 수정
        if (right_distance > max_distance_threshold) {
            if (!fix_debug_printed) {
                std::fprintf(stderr, "[DEBUG] Iris position fix (right):\n");
                std::fprintf(stderr, "  Eye center: (%.4f, %.4f)\n", right_eye_center_x, right_eye_center_y);
                std::fprintf(stderr, "  Iris (before): (%.4f, %.4f), distance=%.4f\n",
                            right_iris_x, right_iris_y, right_distance);
            }
            // 홍채 중심을 눈 중심으로 이동
            right_iris[0] = right_eye_center_x;
            right_iris[1] = right_eye_center_y;
            // 경계점들도 눈 중심 기준으로 조정
            float offset_x = right_eye_center_x - right_iris_x;
            float offset_y = right_eye_center_y - right_iris_y;
            for (int i = 1; i < IRIS_LANDMARK_COUNT; ++i) {
                right_iris[i * 3 + 0] += offset_x;
                right_iris[i * 3 + 1] += offset_y;
            }
            if (!fix_debug_printed) {
                std::fprintf(stderr, "  Iris (after): (%.4f, %.4f)\n", right_iris[0], right_iris[1]);
            }
        }

        fix_debug_printed = true;
    }

    /**
     * @brief Iris Landmark 실행
     * @param input_data 전처리된 입력 데이터 (64x64 RGB float)
     * @param landmarks 출력 랜드마크 (5 * 3 = 15 floats, 항상 인덱스 0-4에 저장)
     * @return 성공 여부
     *
     * @note 이 함수는 항상 5개의 홍채 랜드마크를 버퍼의 인덱스 0-4에 저장합니다.
     *       - 인덱스 0: 홍채 중심
     *       - 인덱스 1-4: 홍채 경계점 (상, 하, 좌, 우)
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

        // 디버그: 모든 출력 텐서 정보 확인
        static bool output_size_printed = false;
        if (!output_size_printed) {
            int num_outputs = static_cast<int>(iris_landmark_interpreter->outputs().size());
            std::fprintf(stderr, "[DEBUG] Iris Landmark Model: %d output tensors\n", num_outputs);

            for (int out_idx = 0; out_idx < num_outputs; ++out_idx) {
                auto* output_tensor = iris_landmark_interpreter->output_tensor(out_idx);
                if (output_tensor) {
                    int total_elements = 1;
                    std::fprintf(stderr, "  Output[%d]: dims=[", out_idx);
                    for (int d = 0; d < output_tensor->dims->size; ++d) {
                        if (d > 0) std::fprintf(stderr, ", ");
                        std::fprintf(stderr, "%d", output_tensor->dims->data[d]);
                        total_elements *= output_tensor->dims->data[d];
                    }
                    std::fprintf(stderr, "], total=%d elements\n", total_elements);
                }
            }
            output_size_printed = true;
        }

        // iris_landmark 모델 출력 구조 분석:
        // - Output[0]: 눈 윤곽 + 홍채 (71 * 3 = 213 floats)
        //   - 인덱스 0-67: 눈 윤곽
        //   - 인덱스 68-72: 홍채 (center + 4 boundary)
        // - Output[1] (있는 경우): 홍채만 (5 * 3 = 15 floats)
        //
        // 항상 인덱스 0-4에 5개 홍채 랜드마크를 저장
        float* iris_output = iris_landmark_interpreter->typed_output_tensor<float>(1);
        if (iris_output) {
            // Output[1]이 있으면 직접 복사 (이미 5개만 있음)
            std::memcpy(landmarks, iris_output, IRIS_LANDMARK_COUNT * 3 * sizeof(float));
        } else {
            // Output[1]이 없으면 Output[0]의 인덱스 68-72에서 추출
            std::memcpy(landmarks, output_data + IRIS_START_INDEX * 3,
                       IRIS_LANDMARK_COUNT * 3 * sizeof(float));
        }

        // 디버그: 추출된 홍채 랜드마크 확인
        static bool iris_extract_debug = false;
        if (!iris_extract_debug) {
            std::fprintf(stderr, "[DEBUG] Extracted Iris Landmarks (index 0-4):\n");
            for (int i = 0; i < IRIS_LANDMARK_COUNT; ++i) {
                std::fprintf(stderr, "  [%d]: x=%.2f, y=%.2f, z=%.4f\n",
                            i, landmarks[i * 3 + 0], landmarks[i * 3 + 1], landmarks[i * 3 + 2]);
            }
            iris_extract_debug = true;
        }

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
    // 3. Face Landmark: 얼굴 랜드마크 추출
    // V1: 192x192 입력, 468개 랜드마크
    // V2: 256x256 입력, 478개 랜드마크 (홍채 내장)
    // =========================================================

    // 모델 버전에 따른 입력 크기 결정
    int fl_input_width = (impl_->model_version == 2) ? FACE_LANDMARK_V2_INPUT_WIDTH : FACE_LANDMARK_INPUT_WIDTH;
    int fl_input_height = (impl_->model_version == 2) ? FACE_LANDMARK_V2_INPUT_HEIGHT : FACE_LANDMARK_INPUT_HEIGHT;

    // ========================================
    // 🔧 테스트 모드: 전체 이미지 직접 입력 (crop/변환 우회)
    // 모델 입출력을 순수하게 확인하기 위한 모드
    // ========================================
    constexpr bool USE_FULL_IMAGE_TEST_MODE = false;  // 정상 얼굴 crop 모드 사용

    Rect actual_face_crop{};
    bool use_full_image = false;

    if (USE_FULL_IMAGE_TEST_MODE) {
        // ========================================
        // 테스트 모드: 중앙 정사각형 crop (종횡비 유지)
        // 1280x720에서 720x720 중앙 영역만 사용
        // ========================================
        int square_size = std::min(width, height);  // 720
        int crop_x = (width - square_size) / 2;     // (1280-720)/2 = 280
        int crop_y = (height - square_size) / 2;    // 0

        // RGB 변환
        cv::Mat rgb_mat;
        if (!impl_->convertToRgb(frame_data, width, height, format, rgb_mat)) {
            impl_->resetTrackingCache();
            return result;
        }

        // 중앙 정사각형 crop
        cv::Rect center_roi(crop_x, crop_y, square_size, square_size);
        cv::Mat cropped = rgb_mat(center_roi);

        // 256x256으로 리사이즈
        cv::Mat resized;
        cv::resize(cropped, resized, cv::Size(fl_input_width, fl_input_height), 0, 0, cv::INTER_LINEAR);

        // 0-1 정규화
        cv::Mat float_mat;
        resized.convertTo(float_mat, CV_32FC3, 1.0 / 255.0);

        // 출력 버퍼로 복사
        std::memcpy(impl_->face_landmark_input_buffer.data(), float_mat.ptr<float>(),
                   fl_input_width * fl_input_height * 3 * sizeof(float));

        // crop 영역 설정 (정규화 좌표)
        actual_face_crop.x = static_cast<float>(crop_x) / width;
        actual_face_crop.y = static_cast<float>(crop_y) / height;
        actual_face_crop.width = static_cast<float>(square_size) / width;
        actual_face_crop.height = static_cast<float>(square_size) / height;
        use_full_image = true;

        // 매 프레임마다 출력 (디버그용)
        std::fprintf(stderr, "[TEST MODE] Center square crop: %dx%d → crop(%d,%d,%dx%d) → 256x256\n",
                    width, height, crop_x, crop_y, square_size, square_size);
        std::fprintf(stderr, "  actual_face_crop: x=%.4f, y=%.4f, w=%.4f, h=%.4f\n",
                    actual_face_crop.x, actual_face_crop.y, actual_face_crop.width, actual_face_crop.height);
    } else {
        // 원래 로직: 얼굴 영역 크롭 후 전처리
        if (!impl_->cropFaceRegion(rgb_mat, face_rect,
                                   fl_input_width,
                                   fl_input_height,
                                   impl_->face_landmark_input_buffer.data(),
                                   actual_face_crop)) {
            // 크롭 실패 시 전체 이미지 사용 (폴백)
            if (!impl_->preprocessImage(frame_data, width, height, format,
                                        fl_input_width,
                                        fl_input_height,
                                        impl_->face_landmark_input_buffer.data())) {
                impl_->resetTrackingCache();
                return result;
            }
            // 폴백 시 전체 이미지를 크롭 영역으로 설정
            actual_face_crop.x = 0.0f;
            actual_face_crop.y = 0.0f;
            actual_face_crop.width = 1.0f;
            actual_face_crop.height = 1.0f;
            use_full_image = true;
        }
    }

    // 사전 할당 버퍼 사용
    if (!impl_->runFaceLandmark(impl_->face_landmark_input_buffer.data(),
                                impl_->face_landmarks_buffer.data())) {
        impl_->resetTrackingCache();
        return result;
    }

    // 디버그: Face Landmark 모델 RAW 출력 확인
    static bool fl_raw_debug = false;
    if (!fl_raw_debug) {
        std::fprintf(stderr, "[DEBUG] Face Landmark RAW Output (before transform):\n");
        std::fprintf(stderr, "  actual_face_crop: x=%.4f, y=%.4f, w=%.4f, h=%.4f\n",
                    actual_face_crop.x, actual_face_crop.y, actual_face_crop.width, actual_face_crop.height);
        // 눈 관련 랜드마크 원시 값 확인
        std::fprintf(stderr, "  idx 33 (left eye): x=%.4f, y=%.4f, z=%.4f\n",
                    impl_->face_landmarks_buffer[33 * 3 + 0],
                    impl_->face_landmarks_buffer[33 * 3 + 1],
                    impl_->face_landmarks_buffer[33 * 3 + 2]);
        std::fprintf(stderr, "  idx 1 (nose): x=%.4f, y=%.4f, z=%.4f\n",
                    impl_->face_landmarks_buffer[1 * 3 + 0],
                    impl_->face_landmarks_buffer[1 * 3 + 1],
                    impl_->face_landmarks_buffer[1 * 3 + 2]);

        // V2 모델에서 홍채 RAW 좌표 확인 (변환 전)
        if (impl_->model_version == 2) {
            std::fprintf(stderr, "[DEBUG] V2 Iris RAW Output (before transform):\n");
            // 왼쪽 홍채 (468-472)
            for (int i = 0; i < 5; ++i) {
                int idx = 468 + i;
                std::fprintf(stderr, "  left_iris[%d] (idx %d): x=%.4f, y=%.4f, z=%.4f\n",
                            i, idx,
                            impl_->face_landmarks_buffer[idx * 3 + 0],
                            impl_->face_landmarks_buffer[idx * 3 + 1],
                            impl_->face_landmarks_buffer[idx * 3 + 2]);
            }
            // 오른쪽 홍채 (473-477)
            for (int i = 0; i < 5; ++i) {
                int idx = 473 + i;
                std::fprintf(stderr, "  right_iris[%d] (idx %d): x=%.4f, y=%.4f, z=%.4f\n",
                            i, idx,
                            impl_->face_landmarks_buffer[idx * 3 + 0],
                            impl_->face_landmarks_buffer[idx * 3 + 1],
                            impl_->face_landmarks_buffer[idx * 3 + 2]);
            }

            // 홍채 중심과 경계점 거리 계산 (픽셀 단위)
            float left_center_x = impl_->face_landmarks_buffer[468 * 3 + 0];
            float left_center_y = impl_->face_landmarks_buffer[468 * 3 + 1];
            float total_dist = 0.0f;
            for (int i = 1; i < 5; ++i) {
                int idx = 468 + i;
                float px = impl_->face_landmarks_buffer[idx * 3 + 0];
                float py = impl_->face_landmarks_buffer[idx * 3 + 1];
                float dx = px - left_center_x;
                float dy = py - left_center_y;
                float dist = std::sqrt(dx * dx + dy * dy);
                total_dist += dist;
                std::fprintf(stderr, "  left_iris distance[%d]: dx=%.4f, dy=%.4f, dist=%.4f\n",
                            i, dx, dy, dist);
            }
            std::fprintf(stderr, "  left_iris avg_radius (RAW): %.4f\n", total_dist / 4.0f);
        }
        fl_raw_debug = true;
    }

    // Face Landmark 좌표 변환: 크롭된 얼굴 영역 → 전체 이미지 좌표
    //
    // MediaPipe Face Landmark 모델 출력 분석:
    // - 공식 MediaPipe는 0-1 정규화 좌표를 출력 (입력 이미지 기준)
    // - TFLite 모델에 따라 픽셀 좌표(0-192/0-256) 또는 정규화 좌표(0-1) 출력 가능
    //
    // 좌표 범위 확인 후 적절한 변환 적용:
    // - 값이 1.5 초과면 픽셀 좌표 → 입력 크기로 나눔
    // - 값이 1.5 이하면 이미 정규화 좌표 → 나눔 불필요
    //
    // 개선: 첫 번째 랜드마크만 확인하면 불안정함
    //       여러 랜드마크의 최대값으로 판단하여 안정성 확보
    int fl_landmark_count = (impl_->model_version == 2) ? FACE_LANDMARK_V2_COUNT : FACE_LANDMARK_COUNT;

    float max_x = 0.0f, max_y = 0.0f;
    for (int i = 0; i < fl_landmark_count; ++i) {
        max_x = std::max(max_x, impl_->face_landmarks_buffer[i * 3 + 0]);
        max_y = std::max(max_y, impl_->face_landmarks_buffer[i * 3 + 1]);
    }
    // 정규화 좌표는 절대 1.5를 초과할 수 없음 (약간의 여유 포함)
    bool is_pixel_coords = (max_x > 1.5f || max_y > 1.5f);

    static bool coord_type_printed = false;
    if (!coord_type_printed) {
        std::fprintf(stderr, "[DEBUG] Face Landmark V%d coordinate type: %s (max_x=%.2f, max_y=%.2f)\n",
                    impl_->model_version,
                    is_pixel_coords ? (impl_->model_version == 2 ? "PIXEL (0-256)" : "PIXEL (0-192)") : "NORMALIZED (0-1)",
                    max_x, max_y);
        std::fprintf(stderr, "[DEBUG] actual_face_crop BEFORE transform: x=%.4f, y=%.4f, w=%.4f, h=%.4f\n",
                    actual_face_crop.x, actual_face_crop.y, actual_face_crop.width, actual_face_crop.height);
        coord_type_printed = true;
    }

    for (int i = 0; i < fl_landmark_count; ++i) {
        float local_x, local_y;

        if (is_pixel_coords) {
            // 픽셀 좌표 → 정규화 좌표 (0-1 범위)
            local_x = impl_->face_landmarks_buffer[i * 3 + 0] /
                            static_cast<float>(fl_input_width);
            local_y = impl_->face_landmarks_buffer[i * 3 + 1] /
                            static_cast<float>(fl_input_height);
        } else {
            // 이미 정규화된 좌표 (MediaPipe 공식 모델)
            local_x = impl_->face_landmarks_buffer[i * 3 + 0];
            local_y = impl_->face_landmarks_buffer[i * 3 + 1];
        }
        // z 좌표는 변환 없이 유지

        // 실제 크롭 영역 내 좌표를 전체 이미지 좌표로 변환
        impl_->face_landmarks_buffer[i * 3 + 0] = actual_face_crop.x + local_x * actual_face_crop.width;
        impl_->face_landmarks_buffer[i * 3 + 1] = actual_face_crop.y + local_y * actual_face_crop.height;
    }

    // 변환 후 좌표 범위 디버그
    static bool transform_result_printed = false;
    if (!transform_result_printed) {
        float post_min_x = 1.0f, post_max_x = 0.0f;
        float post_min_y = 1.0f, post_max_y = 0.0f;
        int valid_count = 0;
        for (int i = 0; i < fl_landmark_count; ++i) {
            float x = impl_->face_landmarks_buffer[i * 3 + 0];
            float y = impl_->face_landmarks_buffer[i * 3 + 1];
            if (x >= 0 && x <= 1 && y >= 0 && y <= 1) {
                post_min_x = std::min(post_min_x, x);
                post_max_x = std::max(post_max_x, x);
                post_min_y = std::min(post_min_y, y);
                post_max_y = std::max(post_max_y, y);
                valid_count++;
            }
        }
        std::fprintf(stderr, "[DEBUG] AFTER transform coordinate range:\n");
        std::fprintf(stderr, "  Valid: %d / %d landmarks\n", valid_count, fl_landmark_count);
        std::fprintf(stderr, "  X range: %.4f ~ %.4f (expected: %.4f ~ %.4f)\n",
                    post_min_x, post_max_x, actual_face_crop.x, actual_face_crop.x + actual_face_crop.width);
        std::fprintf(stderr, "  Y range: %.4f ~ %.4f (expected: %.4f ~ %.4f)\n",
                    post_min_y, post_max_y, actual_face_crop.y, actual_face_crop.y + actual_face_crop.height);
        transform_result_printed = true;
    }

    // =========================================================
    // 4. Iris Landmark: 홍채 좌표 추출
    // V1: 별도 iris_landmark 모델 사용 (64x64 입력)
    // V2: Face Landmark 출력에서 직접 추출 (인덱스 468-477)
    // =========================================================
    // 디버그: Face Landmark 눈 좌표 확인
    static bool eye_debug_printed = false;
    if (!eye_debug_printed) {
        std::fprintf(stderr, "[DEBUG] Face Landmark V%d Eye Coordinates:\n", impl_->model_version);
        std::fprintf(stderr, "  LEFT_EYE (idx 33): x=%.4f, y=%.4f\n",
                    impl_->face_landmarks_buffer[33 * 3 + 0],
                    impl_->face_landmarks_buffer[33 * 3 + 1]);
        std::fprintf(stderr, "  LEFT_EYE (idx 133): x=%.4f, y=%.4f\n",
                    impl_->face_landmarks_buffer[133 * 3 + 0],
                    impl_->face_landmarks_buffer[133 * 3 + 1]);
        std::fprintf(stderr, "  RIGHT_EYE (idx 362): x=%.4f, y=%.4f\n",
                    impl_->face_landmarks_buffer[362 * 3 + 0],
                    impl_->face_landmarks_buffer[362 * 3 + 1]);
        std::fprintf(stderr, "  RIGHT_EYE (idx 263): x=%.4f, y=%.4f\n",
                    impl_->face_landmarks_buffer[263 * 3 + 0],
                    impl_->face_landmarks_buffer[263 * 3 + 1]);
        // 코 랜드마크 (인덱스 1)
        std::fprintf(stderr, "  NOSE (idx 1): x=%.4f, y=%.4f\n",
                    impl_->face_landmarks_buffer[1 * 3 + 0],
                    impl_->face_landmarks_buffer[1 * 3 + 1]);
        if (impl_->model_version == 2) {
            // V2에서 홍채 인덱스 미리보기
            std::fprintf(stderr, "  V2 LEFT_IRIS (idx 468): x=%.4f, y=%.4f\n",
                        impl_->face_landmarks_buffer[468 * 3 + 0],
                        impl_->face_landmarks_buffer[468 * 3 + 1]);
            std::fprintf(stderr, "  V2 RIGHT_IRIS (idx 473): x=%.4f, y=%.4f\n",
                        impl_->face_landmarks_buffer[473 * 3 + 0],
                        impl_->face_landmarks_buffer[473 * 3 + 1]);
        }
        eye_debug_printed = true;
    }

    // =========================================================
    // 모델 버전에 따른 홍채 추출 분기
    // =========================================================
    if (impl_->model_version == 2) {
        // =========================================================
        // V2 모델: Face Landmark 출력에서 직접 홍채 추출
        // 별도의 iris_landmark 모델 호출 불필요
        // =========================================================
        bool left_detected = false, right_detected = false;
        impl_->extractIrisFromFaceLandmarkV2(
            impl_->face_landmarks_buffer.data(),
            actual_face_crop,  // 참조용 (실제로는 이미 변환됨)
            impl_->left_iris_landmarks_buffer.data(),
            impl_->right_iris_landmarks_buffer.data(),
            left_detected,
            right_detected
        );

        // =========================================================
        // 홍채 좌표 검증 및 수정 (눈 중심에서 너무 멀면 보정)
        // =========================================================
        if (left_detected || right_detected) {
            impl_->validateAndFixIrisCoordinates(
                impl_->face_landmarks_buffer.data(),
                impl_->left_iris_landmarks_buffer.data(),
                impl_->right_iris_landmarks_buffer.data(),
                0.05f  // 정규화 좌표에서 5% 거리 임계값
            );

            // 보정된 홍채 좌표를 face_landmarks_buffer에도 반영
            // (getFaceLandmarks() 호출 시 일관된 좌표 제공)
            if (left_detected) {
                for (int i = 0; i < IRIS_LANDMARK_COUNT; ++i) {
                    int idx = V2_LEFT_IRIS_INDICES[i];
                    impl_->face_landmarks_buffer[idx * 3 + 0] = impl_->left_iris_landmarks_buffer[i * 3 + 0];
                    impl_->face_landmarks_buffer[idx * 3 + 1] = impl_->left_iris_landmarks_buffer[i * 3 + 1];
                    impl_->face_landmarks_buffer[idx * 3 + 2] = impl_->left_iris_landmarks_buffer[i * 3 + 2];
                }
            }
            if (right_detected) {
                for (int i = 0; i < IRIS_LANDMARK_COUNT; ++i) {
                    int idx = V2_RIGHT_IRIS_INDICES[i];
                    impl_->face_landmarks_buffer[idx * 3 + 0] = impl_->right_iris_landmarks_buffer[i * 3 + 0];
                    impl_->face_landmarks_buffer[idx * 3 + 1] = impl_->right_iris_landmarks_buffer[i * 3 + 1];
                    impl_->face_landmarks_buffer[idx * 3 + 2] = impl_->right_iris_landmarks_buffer[i * 3 + 2];
                }
            }
        }

        // 왼쪽 홍채 결과 복사
        if (left_detected) {
            for (int i = 0; i < IRIS_LANDMARK_COUNT; ++i) {
                result.left_iris[i].x = impl_->left_iris_landmarks_buffer[i * 3 + 0];
                result.left_iris[i].y = impl_->left_iris_landmarks_buffer[i * 3 + 1];
                result.left_iris[i].z = impl_->left_iris_landmarks_buffer[i * 3 + 2];
                result.left_iris[i].visibility = 1.0f;
            }
            result.left_detected = true;
            result.left_radius = impl_->calculateIrisRadius(result.left_iris, width, height);

            static bool v2_left_debug = false;
            if (!v2_left_debug) {
                std::fprintf(stderr, "[DEBUG] V2 Left Iris Final (embedded):\n");
                std::fprintf(stderr, "  center: x=%.4f, y=%.4f, radius=%.1f\n",
                            result.left_iris[0].x, result.left_iris[0].y, result.left_radius);
                v2_left_debug = true;
            }
        }

        // 오른쪽 홍채 결과 복사
        if (right_detected) {
            for (int i = 0; i < IRIS_LANDMARK_COUNT; ++i) {
                result.right_iris[i].x = impl_->right_iris_landmarks_buffer[i * 3 + 0];
                result.right_iris[i].y = impl_->right_iris_landmarks_buffer[i * 3 + 1];
                result.right_iris[i].z = impl_->right_iris_landmarks_buffer[i * 3 + 2];
                result.right_iris[i].visibility = 1.0f;
            }
            result.right_detected = true;
            result.right_radius = impl_->calculateIrisRadius(result.right_iris, width, height);

            static bool v2_right_debug = false;
            if (!v2_right_debug) {
                std::fprintf(stderr, "[DEBUG] V2 Right Iris Final (embedded):\n");
                std::fprintf(stderr, "  center: x=%.4f, y=%.4f, radius=%.1f\n",
                            result.right_iris[0].x, result.right_iris[0].y, result.right_radius);
                v2_right_debug = true;
            }
        }
    } else {
        // =========================================================
        // V1 모델: 별도 iris_landmark 모델 사용
        // 공식 MediaPipe 방식: 눈 모서리 점 기반 ROI + 오른쪽 눈 반전
        // =========================================================
        Rect left_eye_crop{};
        if (impl_->extractEyeRegionMediaPipe(rgb_mat, impl_->face_landmarks_buffer.data(),
                                              LEFT_EYE_INNER_CORNER, LEFT_EYE_OUTER_CORNER,
                                              IRIS_LANDMARK_INPUT_WIDTH,
                                              impl_->left_iris_input_buffer.data(),
                                              left_eye_crop,
                                              false)) {  // 왼쪽 눈: 반전 없음
            // 디버그: 눈 크롭 영역 확인
            static bool crop_debug_printed = false;
            if (!crop_debug_printed) {
                std::fprintf(stderr, "[DEBUG] V1 Left Eye Crop (MediaPipe): x=%.4f, y=%.4f, w=%.4f, h=%.4f\n",
                            left_eye_crop.x, left_eye_crop.y, left_eye_crop.width, left_eye_crop.height);
                crop_debug_printed = true;
            }

            if (impl_->runIrisLandmark(impl_->left_iris_input_buffer.data(),
                                       impl_->left_iris_landmarks_buffer.data())) {
                // 크롭 좌표를 원본 이미지 좌표로 변환
                // runIrisLandmark()는 항상 인덱스 0-4에 5개 홍채 랜드마크를 저장
                // Iris Landmark 모델은 64x64 픽셀 좌표를 출력하므로 정규화 필요
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

                // 디버그: 변환된 좌표 확인
                static bool left_result_debug = false;
                if (!left_result_debug) {
                    std::fprintf(stderr, "[DEBUG] V1 Left Iris Final (MediaPipe method):\n");
                    std::fprintf(stderr, "  center: x=%.4f, y=%.4f, radius=%.1f\n",
                                result.left_iris[0].x, result.left_iris[0].y, result.left_radius);
                    left_result_debug = true;
                }
            }
        }

        // =========================================================
        // 5. Iris Landmark: 오른쪽 눈 (공식 MediaPipe 방식: 수평 반전)
        // =========================================================
        Rect right_eye_crop{};
        if (impl_->extractEyeRegionMediaPipe(rgb_mat, impl_->face_landmarks_buffer.data(),
                                              RIGHT_EYE_INNER_CORNER, RIGHT_EYE_OUTER_CORNER,
                                              IRIS_LANDMARK_INPUT_WIDTH,
                                              impl_->right_iris_input_buffer.data(),
                                              right_eye_crop,
                                              true)) {  // 오른쪽 눈: 수평 반전
            if (impl_->runIrisLandmark(impl_->right_iris_input_buffer.data(),
                                       impl_->right_iris_landmarks_buffer.data())) {
                // 크롭 좌표를 원본 이미지 좌표로 변환
                // runIrisLandmark()는 항상 인덱스 0-4에 5개 홍채 랜드마크를 저장
                // Iris Landmark 모델은 64x64 픽셀 좌표를 출력하므로 정규화 필요
                //
                // 오른쪽 눈은 수평 반전되어 입력되었으므로 출력 좌표도 역반전 필요
                for (int i = 0; i < IRIS_LANDMARK_COUNT; ++i) {
                    // 픽셀 좌표 → 정규화 좌표 (0-1)
                    float local_x = impl_->right_iris_landmarks_buffer[i * 3 + 0] /
                                    static_cast<float>(IRIS_LANDMARK_INPUT_WIDTH);
                    float local_y = impl_->right_iris_landmarks_buffer[i * 3 + 1] /
                                    static_cast<float>(IRIS_LANDMARK_INPUT_HEIGHT);
                    float local_z = impl_->right_iris_landmarks_buffer[i * 3 + 2];

                    // ========================================
                    // 공식 MediaPipe 방식: x 좌표 역반전
                    // 입력이 반전되었으므로 출력 x를 다시 반전
                    // ========================================
                    local_x = 1.0f - local_x;

                    result.right_iris[i].x = right_eye_crop.x + local_x * right_eye_crop.width;
                    result.right_iris[i].y = right_eye_crop.y + local_y * right_eye_crop.height;
                    result.right_iris[i].z = local_z;
                    result.right_iris[i].visibility = 1.0f;
                }
                result.right_detected = true;
                result.right_radius = impl_->calculateIrisRadius(result.right_iris, width, height);

                // 디버그: 변환된 좌표 확인
                static bool right_result_debug = false;
                if (!right_result_debug) {
                    std::fprintf(stderr, "[DEBUG] V1 Right Iris Final (MediaPipe method with flip):\n");
                    std::fprintf(stderr, "  center: x=%.4f, y=%.4f, radius=%.1f\n",
                                result.right_iris[0].x, result.right_iris[0].y, result.right_radius);
                    right_result_debug = true;
                }
            }
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

    // Face Mesh 복사 (478개 랜드마크, 시각화/디버그용)
    if (result.detected && !impl_->face_landmarks_buffer.empty()) {
        result.face_mesh_valid = true;
        for (int i = 0; i < IrisResult::FACE_MESH_LANDMARK_COUNT; ++i) {
            result.face_mesh[i].x = impl_->face_landmarks_buffer[i * 3 + 0];
            result.face_mesh[i].y = impl_->face_landmarks_buffer[i * 3 + 1];
            result.face_mesh[i].z = impl_->face_landmarks_buffer[i * 3 + 2];
            result.face_mesh[i].visibility = 1.0f;
        }
    } else {
        result.face_mesh_valid = false;
    }

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

int MediaPipeDetector::getFaceLandmarkCount() const {
#if defined(IRIS_SDK_HAS_TFLITE) && defined(IRIS_SDK_HAS_OPENCV)
    if (!impl_ || !impl_->initialized) {
        return 0;
    }
    return (impl_->model_version == 2) ? FACE_LANDMARK_V2_COUNT : FACE_LANDMARK_COUNT;
#else
    return 0;
#endif
}

bool MediaPipeDetector::getFaceLandmarks(float* out_landmarks) const {
#if defined(IRIS_SDK_HAS_TFLITE) && defined(IRIS_SDK_HAS_OPENCV)
    if (!impl_ || !impl_->initialized || !out_landmarks) {
        return false;
    }

    int count = getFaceLandmarkCount();
    if (count == 0 || impl_->face_landmarks_buffer.empty()) {
        return false;
    }

    std::memcpy(out_landmarks, impl_->face_landmarks_buffer.data(),
                count * 3 * sizeof(float));
    return true;
#else
    (void)out_landmarks;
    return false;
#endif
}

int MediaPipeDetector::getModelVersion() const {
#if defined(IRIS_SDK_HAS_TFLITE) && defined(IRIS_SDK_HAS_OPENCV)
    if (!impl_) {
        return 0;
    }
    return impl_->model_version;
#else
    return 0;
#endif
}

} // namespace iris_sdk
