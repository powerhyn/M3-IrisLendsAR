/**
 * @file image_demo.cpp
 * @brief 단일 이미지 홍채 검출 및 렌즈 오버레이 데모
 *
 * 커맨드라인에서 이미지 파일을 입력받아 홍채 검출 후 렌즈를 오버레이한 결과를 저장합니다.
 *
 * 사용법:
 *   image_demo <input_image> [output_image]
 *
 * 예시:
 *   image_demo face.jpg                    # face_output.jpg로 저장
 *   image_demo face.jpg result.png         # result.png로 저장
 */

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "iris_sdk/sdk_api.h"

namespace fs = std::filesystem;

namespace {

/**
 * @brief 모델 디렉토리 자동 탐색
 * @param exe_path 실행 파일 경로
 * @return 모델 디렉토리 경로 (없으면 빈 문자열)
 */
std::string findModelPath(const fs::path& exe_path) {
    const std::vector<fs::path> candidates = {
        exe_path / "models",
        exe_path / ".." / "models",
        exe_path / ".." / ".." / "models",
        exe_path / ".." / ".." / ".." / "shared" / "models",
        fs::current_path() / "models",
        fs::current_path() / ".." / "shared" / "models",
        fs::current_path() / ".." / ".." / "shared" / "models"
    };

    for (const auto& path : candidates) {
        if (fs::exists(path) && fs::is_directory(path)) {
            return fs::canonical(path).string();
        }
    }
    return {};
}

/**
 * @brief 텍스처 파일 자동 탐색
 * @param exe_path 실행 파일 경로
 * @return 텍스처 파일 경로 (없으면 빈 문자열)
 */
std::string findTexturePath(const fs::path& exe_path) {
    const std::vector<fs::path> candidates = {
        exe_path / ".." / ".." / ".." / "shared" / "test_data",
        exe_path / ".." / ".." / "shared" / "test_data",
        exe_path / ".." / "shared" / "test_data",
        fs::current_path() / "shared" / "test_data",
        fs::current_path() / ".." / "shared" / "test_data",
        fs::current_path() / ".." / ".." / "shared" / "test_data"
    };

    for (const auto& dir : candidates) {
        if (!fs::exists(dir) || !fs::is_directory(dir)) {
            continue;
        }

        // lens_sample_type_alpha_01.png 우선 탐색
        fs::path preferred = dir / "lens_sample_type_alpha_01.png";
        if (fs::exists(preferred)) {
            return fs::canonical(preferred).string();
        }

        // 그 외 lens_sample*.png 탐색
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.path().extension() == ".png" &&
                entry.path().filename().string().find("lens_sample") != std::string::npos) {
                return fs::canonical(entry.path()).string();
            }
        }
    }
    return {};
}

/**
 * @brief 출력 파일 경로 생성
 * @param input_path 입력 파일 경로
 * @param output_arg 사용자 지정 출력 경로 (없으면 빈 문자열)
 * @return 출력 파일 경로
 */
std::string generateOutputPath(const fs::path& input_path, const std::string& output_arg) {
    if (!output_arg.empty()) {
        return output_arg;
    }

    // 입력 파일명 기반으로 출력 경로 생성: image.jpg -> image_output.jpg
    fs::path output_path = input_path.parent_path();
    std::string stem = input_path.stem().string();
    std::string ext = input_path.extension().string();

    return (output_path / (stem + "_output" + ext)).string();
}

/**
 * @brief 검출 결과 출력
 * @param result 검출 결과
 */
void printDetectionResults(const IrisResult& result) {
    std::cout << "\nDetection Results:\n";
    std::cout << "  Detected: " << (result.detected ? "Yes" : "No") << "\n";

    if (result.detected) {
        std::printf("  Confidence: %.2f\n", result.confidence);
        std::cout << "  Left eye: " << (result.left_detected ? "Yes" : "No") << "\n";
        std::cout << "  Right eye: " << (result.right_detected ? "Yes" : "No") << "\n";

        if (result.left_detected) {
            std::printf("  Left iris center: (%.4f, %.4f), radius: %.1f\n",
                        result.left_iris[0].x, result.left_iris[0].y, result.left_radius);
        }
        if (result.right_detected) {
            std::printf("  Right iris center: (%.4f, %.4f), radius: %.1f\n",
                        result.right_iris[0].x, result.right_iris[0].y, result.right_radius);
        }
    }
}

/**
 * @brief 사용법 출력
 * @param program_name 프로그램 이름
 */
void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <input_image> [output_image]\n\n";
    std::cout << "Arguments:\n";
    std::cout << "  input_image   Input image file (required)\n";
    std::cout << "  output_image  Output image file (optional, default: <input>_output.<ext>)\n\n";
    std::cout << "Example:\n";
    std::cout << "  " << program_name << " face.jpg\n";
    std::cout << "  " << program_name << " face.jpg result.png\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    std::cout << "===================================\n";
    std::cout << "   IrisLensSDK Image Demo\n";
    std::cout << "===================================\n";

    // 인자 확인
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    // 경로 설정
    const fs::path exe_path = fs::absolute(argv[0]).parent_path();
    const fs::path input_path = fs::absolute(argv[1]);
    const std::string output_arg = (argc > 2) ? argv[2] : "";

    // 입력 파일 확인
    if (!fs::exists(input_path)) {
        std::cerr << "[Error] Input file not found: " << input_path << "\n";
        return 1;
    }

    // 경로 정보 출력
    const std::string output_path = generateOutputPath(input_path, output_arg);
    const std::string model_path = findModelPath(exe_path);
    const std::string texture_path = findTexturePath(exe_path);

    std::cout << "Input: " << input_path << "\n";
    std::cout << "Output: " << output_path << "\n";
    std::cout << "Model: " << (model_path.empty() ? "(not found)" : model_path) << "\n";
    std::cout << "\n";

    // 모델 경로 확인
    if (model_path.empty()) {
        std::cerr << "[Error] Model directory not found.\n";
        std::cerr << "        Please ensure 'shared/models/' exists.\n";
        return 1;
    }

    // SDK 초기화
    IrisSdkError err = iris_sdk_init(model_path.c_str());
    if (err != IRIS_SDK_OK) {
        std::cerr << "[Error] SDK init failed: " << iris_sdk_error_to_string(err) << "\n";
        std::cerr << "        " << iris_sdk_get_last_error() << "\n";
        return 1;
    }
    std::cout << "[Main] SDK initialized successfully (version: " << iris_sdk_get_version() << ")\n";

    // 텍스처 로드
    bool has_texture = false;
    if (!texture_path.empty()) {
        err = iris_sdk_load_texture(texture_path.c_str());
        if (err == IRIS_SDK_OK) {
            has_texture = true;
            std::cout << "[Main] Texture loaded: " << fs::path(texture_path).filename().string() << "\n";
        } else {
            std::cerr << "[Warning] Texture load failed: " << iris_sdk_error_to_string(err) << "\n";
        }
    } else {
        std::cout << "[Warning] No texture found. Detection only mode.\n";
    }

    // 이미지 로드
    cv::Mat image = cv::imread(input_path.string(), cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "[Error] Failed to load image: " << input_path << "\n";
        iris_sdk_destroy();
        return 1;
    }

    std::cout << "[Main] Image loaded: " << image.cols << "x" << image.rows << "\n";
    std::cout << "[Main] Processing image...\n";

    // 렌즈 설정
    IrisLensConfig lens_config;
    iris_sdk_default_lens_config(&lens_config);

    // 검출 결과
    IrisResult result;

    // 처리 시간 측정
    auto start = std::chrono::high_resolution_clock::now();

    // 프레임 처리 (검출 + 렌더링)
    // OpenCV는 BGR 포맷 사용
    if (has_texture) {
        err = iris_sdk_process(
            image.data,
            image.cols,
            image.rows,
            IRIS_FORMAT_BGR,
            &lens_config,
            &result
        );
    } else {
        // 텍스처 없으면 검출만 수행
        err = iris_sdk_detect(
            image.data,
            image.cols,
            image.rows,
            IRIS_FORMAT_BGR,
            &result
        );
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

    if (err != IRIS_SDK_OK) {
        std::cerr << "[Error] Processing failed: " << iris_sdk_error_to_string(err) << "\n";
        std::cerr << "        " << iris_sdk_get_last_error() << "\n";
        iris_sdk_destroy();
        return 1;
    }

    std::printf("[Main] Processing completed in %.1f ms\n", duration_ms);

    // 결과 출력
    printDetectionResults(result);

    // 출력 이미지 저장
    if (cv::imwrite(output_path, image)) {
        std::cout << "\n[Main] Output saved: " << output_path << "\n";
    } else {
        std::cerr << "\n[Error] Failed to save output: " << output_path << "\n";
        iris_sdk_destroy();
        return 1;
    }

    // 정리
    iris_sdk_free_result(&result);
    iris_sdk_destroy();

    std::cout << "[Main] Done.\n";
    return 0;
}
