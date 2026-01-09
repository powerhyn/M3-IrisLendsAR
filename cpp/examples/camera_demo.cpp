/**
 * @file camera_demo.cpp
 * @brief 웹캠 실시간 홍채 검출 및 렌즈 오버레이 데모
 *
 * macOS 웹캠을 사용하여 실시간으로 홍채를 검출하고
 * 가상 렌즈를 오버레이하는 데모 애플리케이션.
 *
 * 키보드 조작:
 *   ESC/Q - 종료
 *   SPACE - 렌즈 텍스처 변경
 *   L     - 랜드마크 표시 토글
 *   F     - FPS 표시 토글
 *   D     - 디버그 정보 토글
 *   +/-   - 투명도 조절
 *   [/]   - 크기 조절
 *   S     - 스크린샷 저장
 *   R     - 설정 초기화
 */

#include <chrono>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "iris_sdk/frame_processor.h"
#include "iris_sdk/types.h"

namespace fs = std::filesystem;

namespace iris_sdk {

/**
 * @brief 카메라 데모 클래스
 *
 * 웹캠에서 실시간 프레임을 캡처하고
 * FrameProcessor를 사용하여 홍채 검출 및 렌즈 렌더링 수행
 */
class CameraDemo {
public:
    CameraDemo() = default;
    ~CameraDemo() { shutdown(); }

    // 복사/이동 금지
    CameraDemo(const CameraDemo&) = delete;
    CameraDemo& operator=(const CameraDemo&) = delete;
    CameraDemo(CameraDemo&&) = delete;
    CameraDemo& operator=(CameraDemo&&) = delete;

    /**
     * @brief 데모 초기화
     * @param model_path 모델 디렉토리 경로
     * @return 초기화 성공 여부
     */
    bool initialize(const std::string& model_path) {
        std::cout << "[CameraDemo] 초기화 시작...\n";

        // 모델 경로 확인
        if (!fs::exists(model_path)) {
            std::cerr << "[CameraDemo] 모델 경로를 찾을 수 없습니다: " << model_path << "\n";
            return false;
        }

        // FrameProcessor 초기화
        if (!processor_.initialize(model_path)) {
            std::cerr << "[CameraDemo] FrameProcessor 초기화 실패\n";
            return false;
        }

        // 웹캠 열기
        camera_.open(0);
        if (!camera_.isOpened()) {
            std::cerr << "[CameraDemo] 웹캠을 열 수 없습니다\n";
            return false;
        }

        // 카메라 해상도 설정 (720p)
        camera_.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        camera_.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        camera_.set(cv::CAP_PROP_FPS, 30);

        // 실제 해상도 확인
        frame_width_ = static_cast<int>(camera_.get(cv::CAP_PROP_FRAME_WIDTH));
        frame_height_ = static_cast<int>(camera_.get(cv::CAP_PROP_FRAME_HEIGHT));
        std::cout << "[CameraDemo] 카메라 해상도: " << frame_width_ << "x" << frame_height_ << "\n";

        // 기본 렌즈 설정 초기화
        resetConfig();

        initialized_ = true;
        std::cout << "[CameraDemo] 초기화 완료\n";
        return true;
    }

    /**
     * @brief 렌즈 텍스처 로드
     * @param texture_paths 텍스처 파일 경로 목록
     * @return 로드된 텍스처 수
     */
    int loadLensTextures(const std::vector<std::string>& texture_paths) {
        texture_paths_.clear();

        for (const auto& path : texture_paths) {
            if (fs::exists(path)) {
                texture_paths_.push_back(path);
                std::cout << "[CameraDemo] 텍스처 추가: " << path << "\n";
            } else {
                std::cerr << "[CameraDemo] 텍스처 파일 없음: " << path << "\n";
            }
        }

        // 첫 번째 텍스처 로드
        if (!texture_paths_.empty()) {
            current_texture_index_ = 0;
            if (processor_.loadLensTexture(texture_paths_[0])) {
                std::cout << "[CameraDemo] 렌즈 텍스처 로드 완료: " << texture_paths_[0] << "\n";
            }
        }

        return static_cast<int>(texture_paths_.size());
    }

    /**
     * @brief 메인 루프 실행
     */
    void run() {
        if (!initialized_) {
            std::cerr << "[CameraDemo] 초기화되지 않았습니다\n";
            return;
        }

        std::cout << "\n===== IrisLensSDK 카메라 데모 =====\n";
        printHelp();
        std::cout << "\n";

        cv::namedWindow(window_name_, cv::WINDOW_AUTOSIZE);

        running_ = true;
        while (running_) {
            // 프레임 캡처
            cv::Mat frame;
            if (!camera_.read(frame) || frame.empty()) {
                std::cerr << "[CameraDemo] 프레임 캡처 실패\n";
                break;
            }

            // 좌우 반전 (미러 효과)
            cv::flip(frame, frame, 1);

            // 프레임 처리 (검출 + 렌더링)
            ProcessResult result;
            if (processor_.hasLensTexture() && lens_enabled_) {
                result = processor_.process(frame, &lens_config_);
            } else {
                result = processor_.process(frame, nullptr);
            }

            // 오버레이 그리기
            drawOverlays(frame, result);

            // 화면 표시
            cv::imshow(window_name_, frame);

            // 키 입력 처리
            int key = cv::waitKey(1) & 0xFF;
            handleKeyInput(key);
        }

        cv::destroyWindow(window_name_);
    }

    /**
     * @brief 리소스 정리
     */
    void shutdown() {
        if (!initialized_) return;

        running_ = false;

        if (camera_.isOpened()) {
            camera_.release();
        }

        processor_.release();
        initialized_ = false;

        std::cout << "[CameraDemo] 종료 완료\n";
    }

private:
    /**
     * @brief 설정 초기화
     */
    void resetConfig() {
        lens_config_.opacity = 0.7f;
        lens_config_.scale = 1.0f;
        lens_config_.offset_x = 0.0f;
        lens_config_.offset_y = 0.0f;
        lens_config_.blend_mode = BlendMode::Normal;
        lens_config_.edge_feather = 0.1f;
        lens_config_.apply_left = true;
        lens_config_.apply_right = true;

        std::cout << "[CameraDemo] 설정 초기화됨\n";
    }

    /**
     * @brief 키 입력 처리
     */
    void handleKeyInput(int key) {
        switch (key) {
            case 27:  // ESC
            case 'q':
            case 'Q':
                running_ = false;
                break;

            case ' ':  // SPACE - 텍스처 변경
                cycleTexture();
                break;

            case 'l':
            case 'L':
                show_landmarks_ = !show_landmarks_;
                std::cout << "[CameraDemo] 랜드마크 표시: "
                          << (show_landmarks_ ? "ON" : "OFF") << "\n";
                break;

            case 'f':
            case 'F':
                show_fps_ = !show_fps_;
                std::cout << "[CameraDemo] FPS 표시: "
                          << (show_fps_ ? "ON" : "OFF") << "\n";
                break;

            case 'd':
            case 'D':
                show_debug_ = !show_debug_;
                std::cout << "[CameraDemo] 디버그 정보: "
                          << (show_debug_ ? "ON" : "OFF") << "\n";
                break;

            case '+':
            case '=':  // + 키 (shift 없이)
                lens_config_.opacity = std::min(1.0f, lens_config_.opacity + 0.05f);
                std::cout << "[CameraDemo] 투명도: " << lens_config_.opacity << "\n";
                break;

            case '-':
            case '_':
                lens_config_.opacity = std::max(0.0f, lens_config_.opacity - 0.05f);
                std::cout << "[CameraDemo] 투명도: " << lens_config_.opacity << "\n";
                break;

            case ']':
                lens_config_.scale = std::min(2.0f, lens_config_.scale + 0.05f);
                std::cout << "[CameraDemo] 크기: " << lens_config_.scale << "\n";
                break;

            case '[':
                lens_config_.scale = std::max(0.5f, lens_config_.scale - 0.05f);
                std::cout << "[CameraDemo] 크기: " << lens_config_.scale << "\n";
                break;

            case 's':
            case 'S':
                saveScreenshot();
                break;

            case 'r':
            case 'R':
                resetConfig();
                break;

            case 'e':
            case 'E':
                lens_enabled_ = !lens_enabled_;
                std::cout << "[CameraDemo] 렌즈 효과: "
                          << (lens_enabled_ ? "ON" : "OFF") << "\n";
                break;

            case 'h':
            case 'H':
                printHelp();
                break;

            default:
                break;
        }
    }

    /**
     * @brief 텍스처 순환
     */
    void cycleTexture() {
        if (texture_paths_.empty()) {
            std::cout << "[CameraDemo] 로드된 텍스처가 없습니다\n";
            return;
        }

        current_texture_index_ = (current_texture_index_ + 1) % texture_paths_.size();
        const auto& path = texture_paths_[current_texture_index_];

        if (processor_.loadLensTexture(path)) {
            std::cout << "[CameraDemo] 텍스처 변경: " << fs::path(path).filename().string()
                      << " (" << (current_texture_index_ + 1) << "/"
                      << texture_paths_.size() << ")\n";
        }
    }

    /**
     * @brief 스크린샷 저장
     */
    void saveScreenshot() {
        // 현재 시간으로 파일명 생성
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::tm tm = *std::localtime(&time_t);

        std::ostringstream oss;
        oss << "screenshot_"
            << std::put_time(&tm, "%Y%m%d_%H%M%S")
            << ".png";

        std::string filename = oss.str();

        cv::Mat frame;
        if (camera_.read(frame)) {
            cv::flip(frame, frame, 1);
            if (processor_.hasLensTexture() && lens_enabled_) {
                processor_.process(frame, &lens_config_);
            }

            if (cv::imwrite(filename, frame)) {
                std::cout << "[CameraDemo] 스크린샷 저장: " << filename << "\n";
            } else {
                std::cerr << "[CameraDemo] 스크린샷 저장 실패\n";
            }
        }
    }

    /**
     * @brief 오버레이 그리기
     */
    void drawOverlays(cv::Mat& frame, const ProcessResult& result) {
        const int padding = 10;
        int y_offset = padding + 20;

        // FPS 표시
        if (show_fps_) {
            double fps = processor_.getAverageFPS();
            std::ostringstream fps_text;
            fps_text << std::fixed << std::setprecision(1) << "FPS: " << fps;

            cv::putText(frame, fps_text.str(),
                       cv::Point(padding, y_offset),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7,
                       cv::Scalar(0, 255, 0), 2);
            y_offset += 25;
        }

        // 디버그 정보 표시
        if (show_debug_) {
            // 처리 시간
            std::ostringstream time_text;
            time_text << std::fixed << std::setprecision(1)
                      << "Total: " << result.processing_time_ms << "ms"
                      << " (Det: " << result.detection_time_ms << "ms"
                      << ", Rend: " << result.render_time_ms << "ms)";

            cv::putText(frame, time_text.str(),
                       cv::Point(padding, y_offset),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5,
                       cv::Scalar(255, 255, 0), 1);
            y_offset += 20;

            // 검출 상태
            std::string detect_status = result.iris_result.detected
                ? "Detected (L:" + std::string(result.iris_result.left_detected ? "Y" : "N")
                  + " R:" + std::string(result.iris_result.right_detected ? "Y" : "N") + ")"
                : "Not Detected";

            cv::Scalar status_color = result.iris_result.detected
                ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);

            cv::putText(frame, detect_status,
                       cv::Point(padding, y_offset),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5,
                       status_color, 1);
            y_offset += 20;

            // 신뢰도
            if (result.iris_result.detected) {
                std::ostringstream conf_text;
                conf_text << std::fixed << std::setprecision(2)
                          << "Confidence: " << result.iris_result.confidence;

                cv::putText(frame, conf_text.str(),
                           cv::Point(padding, y_offset),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5,
                           cv::Scalar(255, 255, 0), 1);
                y_offset += 20;
            }

            // 현재 설정
            std::ostringstream config_text;
            config_text << std::fixed << std::setprecision(2)
                        << "Opacity: " << lens_config_.opacity
                        << " Scale: " << lens_config_.scale;

            cv::putText(frame, config_text.str(),
                       cv::Point(padding, y_offset),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5,
                       cv::Scalar(200, 200, 200), 1);
            y_offset += 20;

            // 렌즈 상태
            std::string lens_status = processor_.hasLensTexture() && lens_enabled_
                ? "Lens: ON" : "Lens: OFF";

            cv::putText(frame, lens_status,
                       cv::Point(padding, y_offset),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5,
                       lens_enabled_ ? cv::Scalar(0, 255, 0) : cv::Scalar(128, 128, 128), 1);
        }

        // 랜드마크 그리기
        if (show_landmarks_ && result.iris_result.detected) {
            drawLandmarks(frame, result.iris_result);
        }

        // 조작 도움말 (하단)
        cv::putText(frame, "Press 'H' for help",
                   cv::Point(padding, frame.rows - padding),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5,
                   cv::Scalar(128, 128, 128), 1);
    }

    /**
     * @brief 홍채 랜드마크 그리기
     */
    void drawLandmarks(cv::Mat& frame, const IrisResult& iris) {
        const cv::Scalar left_color(255, 0, 0);    // 파란색 (BGR)
        const cv::Scalar right_color(0, 0, 255);   // 빨간색 (BGR)
        const cv::Scalar center_color(0, 255, 0);  // 녹색 (BGR)

        // 왼쪽 눈 랜드마크
        if (iris.left_detected) {
            // 중심점 (더 크게)
            cv::Point center(
                static_cast<int>(iris.left_iris[0].x * frame.cols),
                static_cast<int>(iris.left_iris[0].y * frame.rows)
            );
            cv::circle(frame, center, 5, center_color, -1);

            // 경계점들
            for (int i = 1; i < 5; ++i) {
                cv::Point pt(
                    static_cast<int>(iris.left_iris[i].x * frame.cols),
                    static_cast<int>(iris.left_iris[i].y * frame.rows)
                );
                cv::circle(frame, pt, 3, left_color, -1);
                cv::line(frame, center, pt, left_color, 1);
            }

            // 반지름 원
            cv::circle(frame, center, static_cast<int>(iris.left_radius), left_color, 1);

            // 레이블
            cv::putText(frame, "L",
                       cv::Point(center.x - 30, center.y - 10),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, left_color, 1);
        }

        // 오른쪽 눈 랜드마크
        if (iris.right_detected) {
            // 중심점 (더 크게)
            cv::Point center(
                static_cast<int>(iris.right_iris[0].x * frame.cols),
                static_cast<int>(iris.right_iris[0].y * frame.rows)
            );
            cv::circle(frame, center, 5, center_color, -1);

            // 경계점들
            for (int i = 1; i < 5; ++i) {
                cv::Point pt(
                    static_cast<int>(iris.right_iris[i].x * frame.cols),
                    static_cast<int>(iris.right_iris[i].y * frame.rows)
                );
                cv::circle(frame, pt, 3, right_color, -1);
                cv::line(frame, center, pt, right_color, 1);
            }

            // 반지름 원
            cv::circle(frame, center, static_cast<int>(iris.right_radius), right_color, 1);

            // 레이블
            cv::putText(frame, "R",
                       cv::Point(center.x + 10, center.y - 10),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, right_color, 1);
        }

        // 얼굴 바운딩 박스
        if (iris.detected && iris.face_rect.width > 0) {
            cv::Rect face_rect(
                static_cast<int>(iris.face_rect.x),
                static_cast<int>(iris.face_rect.y),
                static_cast<int>(iris.face_rect.width),
                static_cast<int>(iris.face_rect.height)
            );
            cv::rectangle(frame, face_rect, cv::Scalar(255, 255, 0), 1);
        }
    }

    /**
     * @brief 도움말 출력
     */
    void printHelp() {
        std::cout << "\n=== 키보드 조작 ===\n"
                  << "  ESC/Q    : 종료\n"
                  << "  SPACE    : 렌즈 텍스처 변경\n"
                  << "  E        : 렌즈 효과 ON/OFF\n"
                  << "  L        : 랜드마크 표시 토글\n"
                  << "  F        : FPS 표시 토글\n"
                  << "  D        : 디버그 정보 토글\n"
                  << "  +/-      : 투명도 조절\n"
                  << "  [/]      : 크기 조절\n"
                  << "  S        : 스크린샷 저장\n"
                  << "  R        : 설정 초기화\n"
                  << "  H        : 도움말 표시\n"
                  << "==================\n";
    }

private:
    // 상태
    bool initialized_ = false;
    bool running_ = false;

    // 카메라
    cv::VideoCapture camera_;
    int frame_width_ = 0;
    int frame_height_ = 0;

    // 프로세서
    FrameProcessor processor_;

    // 텍스처
    std::vector<std::string> texture_paths_;
    size_t current_texture_index_ = 0;

    // 렌즈 설정
    LensConfig lens_config_;
    bool lens_enabled_ = true;

    // 표시 옵션
    bool show_landmarks_ = true;
    bool show_fps_ = true;
    bool show_debug_ = true;

    // UI
    const std::string window_name_ = "IrisLensSDK Camera Demo";
};

} // namespace iris_sdk


/**
 * @brief 메인 함수
 */
int main(int argc, char* argv[]) {
    std::cout << "===================================\n";
    std::cout << "   IrisLensSDK Camera Demo\n";
    std::cout << "===================================\n\n";

    // 실행 파일 위치 기준 경로 설정
    fs::path exe_path = fs::absolute(argv[0]).parent_path();

    // 모델 경로 결정 (명령행 인자 또는 기본 경로)
    std::string model_path;
    if (argc > 1) {
        model_path = argv[1];
    } else {
        // 가능한 모델 경로들 시도
        std::vector<fs::path> possible_paths = {
            exe_path / "models",
            exe_path / ".." / "models",
            exe_path / ".." / ".." / "models",
            fs::current_path() / "models",
            fs::current_path() / ".." / "models"
        };

        for (const auto& path : possible_paths) {
            if (fs::exists(path)) {
                model_path = fs::canonical(path).string();
                break;
            }
        }
    }

    if (model_path.empty() || !fs::exists(model_path)) {
        std::cerr << "[Error] 모델 경로를 찾을 수 없습니다.\n";
        std::cerr << "사용법: " << argv[0] << " [model_path]\n";
        return 1;
    }

    std::cout << "[Main] 모델 경로: " << model_path << "\n";

    // 텍스처 경로 결정
    std::vector<std::string> texture_paths;
    std::vector<fs::path> possible_texture_dirs = {
        exe_path / ".." / ".." / "shared" / "test_data",
        exe_path / ".." / "shared" / "test_data",
        fs::current_path() / "shared" / "test_data",
        fs::current_path() / ".." / "shared" / "test_data"
    };

    for (const auto& dir : possible_texture_dirs) {
        if (fs::exists(dir)) {
            fs::path texture_dir = fs::canonical(dir);

            // 렌즈 텍스처 파일들 찾기
            for (const auto& entry : fs::directory_iterator(texture_dir)) {
                if (entry.path().extension() == ".png" &&
                    entry.path().filename().string().find("lens_sample") != std::string::npos) {
                    texture_paths.push_back(entry.path().string());
                }
            }

            if (!texture_paths.empty()) {
                std::cout << "[Main] 텍스처 디렉토리: " << texture_dir << "\n";
                break;
            }
        }
    }

    // CameraDemo 실행
    iris_sdk::CameraDemo demo;

    if (!demo.initialize(model_path)) {
        std::cerr << "[Error] 데모 초기화 실패\n";
        return 1;
    }

    if (!texture_paths.empty()) {
        int loaded = demo.loadLensTextures(texture_paths);
        std::cout << "[Main] 로드된 텍스처: " << loaded << "개\n";
    } else {
        std::cout << "[Warning] 렌즈 텍스처를 찾을 수 없습니다.\n";
        std::cout << "         렌즈 오버레이 없이 검출만 수행합니다.\n";
    }

    demo.run();

    return 0;
}
