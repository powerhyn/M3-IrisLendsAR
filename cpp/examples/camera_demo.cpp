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
     * @param exe_path 실행 파일 경로 (스크린샷 저장용)
     * @return 초기화 성공 여부
     */
    bool initialize(const std::string& model_path, const fs::path& exe_path = fs::path()) {
        std::cout << "[CameraDemo] 초기화 시작...\n";

        // 실행 파일 경로 저장 (스크린샷 저장 위치)
        exe_path_ = exe_path.empty() ? fs::current_path() : exe_path;

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

            // 스무딩 적용 (흔들림 완화)
            if (smoothing_enabled_ && result.iris_result.detected) {
                applySmoothingToResult(result.iris_result);
            }
            previous_result_ = result.iris_result;

            // 오버레이 그리기
            drawOverlays(frame, result);

            // 스크린샷용 프레임 저장 (오버레이 포함)
            last_displayed_frame_ = frame.clone();

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

            case 't':
            case 'T':
                detection_confidence_ = std::min(0.9f, detection_confidence_ + 0.05f);
                processor_.setMinConfidence(detection_confidence_);
                std::cout << "[CameraDemo] 검출 임계값: " << detection_confidence_ << "\n";
                break;

            case 'g':
            case 'G':
                detection_confidence_ = std::max(0.1f, detection_confidence_ - 0.05f);
                processor_.setMinConfidence(detection_confidence_);
                std::cout << "[CameraDemo] 검출 임계값: " << detection_confidence_ << "\n";
                break;

            case 'm':
            case 'M':
                smoothing_enabled_ = !smoothing_enabled_;
                std::cout << "[CameraDemo] 스무딩: "
                          << (smoothing_enabled_ ? "ON" : "OFF") << "\n";
                break;

            case 'v':
            case 'V':
                show_face_mesh_ = !show_face_mesh_;
                std::cout << "[CameraDemo] Face Mesh 전체 표시: "
                          << (show_face_mesh_ ? "ON" : "OFF") << "\n";
                break;

            default:
                break;
        }
    }

    /**
     * @brief 좌표 스무딩 적용 (Exponential Moving Average)
     *
     * 이전 프레임 결과와 현재 결과를 보간하여 흔들림 완화
     */
    void applySmoothingToResult(IrisResult& result) {
        if (!previous_result_.detected) {
            return;  // 이전 결과 없으면 스무딩 불가
        }

        // 왼쪽 홍채 스무딩
        if (result.left_detected && previous_result_.left_detected) {
            for (int i = 0; i < 5; ++i) {
                result.left_iris[i].x = lerp(previous_result_.left_iris[i].x,
                                              result.left_iris[i].x, SMOOTHING_FACTOR);
                result.left_iris[i].y = lerp(previous_result_.left_iris[i].y,
                                              result.left_iris[i].y, SMOOTHING_FACTOR);
            }
            result.left_radius = lerp(previous_result_.left_radius,
                                       result.left_radius, SMOOTHING_FACTOR);
        }

        // 오른쪽 홍채 스무딩
        if (result.right_detected && previous_result_.right_detected) {
            for (int i = 0; i < 5; ++i) {
                result.right_iris[i].x = lerp(previous_result_.right_iris[i].x,
                                               result.right_iris[i].x, SMOOTHING_FACTOR);
                result.right_iris[i].y = lerp(previous_result_.right_iris[i].y,
                                               result.right_iris[i].y, SMOOTHING_FACTOR);
            }
            result.right_radius = lerp(previous_result_.right_radius,
                                        result.right_radius, SMOOTHING_FACTOR);
        }
    }

    /**
     * @brief 선형 보간 (Linear Interpolation)
     */
    static float lerp(float a, float b, float t) {
        return a + t * (b - a);
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
     *
     * 디버그 정보가 켜져있으면 현재 화면에 보이는 그대로 저장.
     * 디버그 정보가 꺼져있으면 렌즈만 적용된 깨끗한 이미지 저장.
     * 저장 위치: 실행 파일 옆 screenshots/ 폴더
     */
    void saveScreenshot() {
        // 스크린샷 디렉토리 생성
        fs::path screenshot_dir = exe_path_ / "screenshots";
        if (!fs::exists(screenshot_dir)) {
            fs::create_directories(screenshot_dir);
            std::cout << "[CameraDemo] 스크린샷 폴더 생성: " << screenshot_dir << "\n";
        }

        // 현재 시간으로 파일명 생성
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::tm tm = *std::localtime(&time_t);

        std::ostringstream oss;
        oss << "screenshot_"
            << std::put_time(&tm, "%Y%m%d_%H%M%S")
            << ".png";

        fs::path filepath = screenshot_dir / oss.str();

        // 디버그/FPS/랜드마크 정보가 하나라도 켜져있으면 화면에 보이는 그대로 저장
        bool save_with_overlays = (show_debug_ || show_fps_ || show_landmarks_);

        if (save_with_overlays && !last_displayed_frame_.empty()) {
            // 현재 화면에 보이는 프레임 저장 (오버레이 포함)
            if (cv::imwrite(filepath.string(), last_displayed_frame_)) {
                std::cout << "[CameraDemo] 스크린샷 저장 (오버레이 포함): " << filepath << "\n";
            } else {
                std::cerr << "[CameraDemo] 스크린샷 저장 실패\n";
            }
        } else {
            // 깨끗한 이미지 저장 (렌즈만 적용)
            cv::Mat frame;
            if (camera_.read(frame)) {
                cv::flip(frame, frame, 1);
                if (processor_.hasLensTexture() && lens_enabled_) {
                    processor_.process(frame, &lens_config_);
                }

                if (cv::imwrite(filepath.string(), frame)) {
                    std::cout << "[CameraDemo] 스크린샷 저장 (클린): " << filepath << "\n";
                } else {
                    std::cerr << "[CameraDemo] 스크린샷 저장 실패\n";
                }
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

        // Face Mesh 전체 그리기 (V 키로 토글)
        if (show_face_mesh_ && result.iris_result.detected) {
            drawFaceMesh(frame);
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

        // 얼굴 바운딩 박스 (정규화 좌표를 픽셀 좌표로 변환)
        if (iris.detected && iris.face_rect.width > 0) {
            // 디버그: face_rect 값 출력 (처음 10프레임)
            static int face_rect_debug_count = 0;
            if (face_rect_debug_count < 10) {
                std::fprintf(stderr, "[DEBUG] drawLandmarks face_rect: x=%.4f, y=%.4f, w=%.4f, h=%.4f\n",
                            iris.face_rect.x, iris.face_rect.y,
                            iris.face_rect.width, iris.face_rect.height);
                face_rect_debug_count++;
            }

            cv::Rect face_rect(
                static_cast<int>(iris.face_rect.x * frame.cols),
                static_cast<int>(iris.face_rect.y * frame.rows),
                static_cast<int>(iris.face_rect.width * frame.cols),
                static_cast<int>(iris.face_rect.height * frame.rows)
            );
            cv::rectangle(frame, face_rect, cv::Scalar(255, 255, 0), 1);
        }
    }

    /**
     * @brief 전체 얼굴 메시 랜드마크 그리기 (478개)
     */
    void drawFaceMesh(cv::Mat& frame) {
        // 랜드마크 수 확인
        int landmark_count = processor_.getFaceLandmarkCount();
        if (landmark_count <= 0) {
            return;
        }

        // 랜드마크 버퍼 할당
        std::vector<float> landmarks(landmark_count * 3);
        if (!processor_.getFaceLandmarks(landmarks.data())) {
            return;
        }

        // 모델 버전 확인
        int model_version = processor_.getModelVersion();

        // 디버그: 좌표 범위 확인 (최초 1회)
        static bool coord_debug = false;
        if (!coord_debug) {
            float min_x = 1.0f, max_x = 0.0f, min_y = 1.0f, max_y = 0.0f;
            int valid_count = 0;
            for (int i = 0; i < landmark_count; ++i) {
                float x = landmarks[i * 3 + 0];
                float y = landmarks[i * 3 + 1];
                if (x >= 0 && x <= 1 && y >= 0 && y <= 1) {
                    min_x = std::min(min_x, x);
                    max_x = std::max(max_x, x);
                    min_y = std::min(min_y, y);
                    max_y = std::max(max_y, y);
                    valid_count++;
                }
            }
            std::fprintf(stderr, "[DEBUG] Face Mesh Coordinate Range:\n");
            std::fprintf(stderr, "  Valid landmarks: %d / %d\n", valid_count, landmark_count);
            std::fprintf(stderr, "  X range: %.4f ~ %.4f\n", min_x, max_x);
            std::fprintf(stderr, "  Y range: %.4f ~ %.4f\n", min_y, max_y);
            // 몇 개의 주요 랜드마크 좌표 출력
            std::fprintf(stderr, "  idx 0 (face top): x=%.4f, y=%.4f\n",
                        landmarks[0], landmarks[1]);
            std::fprintf(stderr, "  idx 1 (nose): x=%.4f, y=%.4f\n",
                        landmarks[3], landmarks[4]);
            std::fprintf(stderr, "  idx 152 (chin): x=%.4f, y=%.4f\n",
                        landmarks[152*3], landmarks[152*3+1]);
            std::fprintf(stderr, "  idx 234 (left cheek): x=%.4f, y=%.4f\n",
                        landmarks[234*3], landmarks[234*3+1]);
            coord_debug = true;
        }

        // 색상 정의
        const cv::Scalar face_color(200, 200, 200);     // 얼굴 (회색)
        const cv::Scalar eye_color(255, 255, 0);        // 눈 (청록)
        const cv::Scalar iris_color(0, 255, 0);         // 홍채 (녹색)
        const cv::Scalar lips_color(128, 0, 255);       // 입술 (분홍)

        // 주요 랜드마크 인덱스 (MediaPipe Face Mesh)
        // 왼쪽 눈: 33, 133, 159, 145
        // 오른쪽 눈: 362, 263, 386, 374
        // 입술: 61, 291, 0, 17
        // 홍채 (V2): 468-472 (왼쪽), 473-477 (오른쪽)

        // 모든 랜드마크 점 그리기
        for (int i = 0; i < landmark_count; ++i) {
            float x = landmarks[i * 3 + 0];
            float y = landmarks[i * 3 + 1];

            // 유효하지 않은 좌표 스킵
            if (x < 0 || x > 1 || y < 0 || y > 1) {
                continue;
            }

            cv::Point pt(
                static_cast<int>(x * frame.cols),
                static_cast<int>(y * frame.rows)
            );

            // 인덱스에 따라 색상 결정
            cv::Scalar color = face_color;
            int radius = 1;

            // 눈 랜드마크 (왼쪽: 33, 133, 159, 145 / 오른쪽: 362, 263, 386, 374)
            if (i == 33 || i == 133 || i == 159 || i == 145 ||
                i == 362 || i == 263 || i == 386 || i == 374) {
                color = eye_color;
                radius = 3;
            }
            // 입술 랜드마크
            else if (i == 61 || i == 291 || i == 0 || i == 17 ||
                     i == 78 || i == 308 || i == 13 || i == 14) {
                color = lips_color;
                radius = 2;
            }
            // 홍채 랜드마크 (V2 모델, 인덱스 468-477)
            else if (model_version == 2 && i >= 468 && i <= 477) {
                color = iris_color;
                radius = 4;
            }

            cv::circle(frame, pt, radius, color, -1);
        }

        // 모델 버전 및 랜드마크 수 표시
        std::string info = "Face Mesh V" + std::to_string(model_version) +
                           " (" + std::to_string(landmark_count) + " pts)";
        cv::putText(frame, info,
                   cv::Point(10, frame.rows - 40),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5,
                   cv::Scalar(0, 255, 255), 1);
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
                  << "  V        : Face Mesh 전체 표시 (478개 포인트)\n"
                  << "  F        : FPS 표시 토글\n"
                  << "  D        : 디버그 정보 토글\n"
                  << "  M        : 스무딩 ON/OFF (흔들림 방지)\n"
                  << "  T/G      : 검출 임계값 증가/감소\n"
                  << "  +/-      : 투명도 조절\n"
                  << "  [/]      : 크기 조절\n"
                  << "  S        : 스크린샷 저장 (screenshots/ 폴더)\n"
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
    bool show_face_mesh_ = false;  // V 키로 전체 Face Mesh 토글

    // 검출 설정
    float detection_confidence_ = 0.3f;  // 기본값 낮춤
    bool smoothing_enabled_ = true;      // 스무딩 활성화

    // 스무딩용 이전 결과 저장
    IrisResult previous_result_;
    static constexpr float SMOOTHING_FACTOR = 0.3f;  // 0.0=이전값 유지, 1.0=새값만 사용

    // UI
    const std::string window_name_ = "IrisLensSDK Camera Demo";

    // 스크린샷용 경로 및 프레임 저장
    fs::path exe_path_;
    cv::Mat last_displayed_frame_;
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

    if (!demo.initialize(model_path, exe_path)) {
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
