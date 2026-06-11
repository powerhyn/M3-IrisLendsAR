/**
 * @file golden_capture.cpp
 * @brief 골든 베이스라인 캡처 도구 (계획 2-2)
 *
 * 표준 입력 세트에 대해 IrisLensSDK C API(CPU 경로)의
 *   ① 랜드마크/검출 결과(JSON 덤프)
 *   ② 렌즈 렌더 결과 + CPU beauty 결과(PNG)
 * 를 결정적으로 캡처한다. 이후 "동작 불변" 단계들이 이 베이스라인과
 * 비교(scripts/golden_compare.py)하여 회귀를 잡는다.
 *
 * 외부 의존성 추가 금지: JSON은 수동 직렬화, 이미지 I/O는 이미 링크된 OpenCV 사용.
 *
 * 전처리 순서(중요):
 *   1) gamma 보정 (저조도 변형 합성)
 *   2) mirror (좌우 반전, 전면 카메라 변형 합성)
 *   3) rotation (이미지를 실제로 회전 → iris_sdk_detect_with_rotation에 전달)
 * rotation은 감사 blocker(회전 경로 무테스트)를 골든이 커버하는 핵심이므로
 * "이미지 픽셀을 실제 회전시킨 뒤" 회전 각도를 SDK에 함께 넘긴다.
 *
 * 사용법:
 *   golden_capture --input <png> --models <dir> --texture <png> --out <dir>
 *                  [--rotation {0|90|180|270}] [--mirror {0|1}] [--gamma <float>]
 *                  [--no-beauty] [--no-render] [--lens-mirror {0|1}]
 *
 *   --no-render   : 렌즈 렌더 PNG 생략 (JSON 전용 변형 — 용량 규율)
 *   --lens-mirror : IrisLensConfig.is_mirror=true 로 렌더 (전면 카메라 렌더 분기 커버)
 *
 * 출력(--out 디렉토리):
 *   <stem>.result.json   결정적 검출 필드 전체
 *   <stem>.render.png    렌즈 렌더 결과 (CPU)
 *   <stem>.beauty.png    CPU beauty 결과 (--no-beauty면 생략)
 * <stem>은 입력 파일명 + 변형 접미사로 호출 스크립트가 지정한다(--out-stem).
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "iris_sdk/sdk_api.h"

namespace fs = std::filesystem;

namespace {

// ---------------------------------------------------------------------------
// 인자 파싱
// ---------------------------------------------------------------------------
struct Args {
    std::string input;
    std::string models;
    std::string texture;
    std::string out;
    std::string out_stem;     // 출력 파일 stem (미지정 시 입력 stem)
    int rotation = 0;         // 0|90|180|270
    int mirror = 0;           // 0|1
    float gamma = 1.0f;       // 1.0 = 무변형
    bool beauty = true;       // CPU beauty PNG 생성 여부
    bool render = true;       // 렌즈 렌더 PNG 생성 여부 (--no-render로 생략)
    int lens_mirror = 0;      // 0|1 — IrisLensConfig.is_mirror (렌더 분기 커버)
};

bool parseArgs(int argc, char* argv[], Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "[Error] " << name << " 인자 값 누락\n";
                return {};
            }
            return argv[++i];
        };
        if (key == "--input") {
            a.input = next("--input");
        } else if (key == "--models") {
            a.models = next("--models");
        } else if (key == "--texture") {
            a.texture = next("--texture");
        } else if (key == "--out") {
            a.out = next("--out");
        } else if (key == "--out-stem") {
            a.out_stem = next("--out-stem");
        } else if (key == "--rotation") {
            a.rotation = std::stoi(next("--rotation"));
        } else if (key == "--mirror") {
            a.mirror = std::stoi(next("--mirror"));
        } else if (key == "--gamma") {
            a.gamma = std::stof(next("--gamma"));
        } else if (key == "--no-beauty") {
            a.beauty = false;
        } else if (key == "--no-render") {
            a.render = false;
        } else if (key == "--lens-mirror") {
            a.lens_mirror = std::stoi(next("--lens-mirror"));
        } else {
            std::cerr << "[Error] 알 수 없는 인자: " << key << "\n";
            return false;
        }
    }
    if (a.input.empty() || a.models.empty() || a.out.empty()) {
        std::cerr << "[Error] --input, --models, --out 은 필수입니다.\n";
        return false;
    }
    if (a.rotation != 0 && a.rotation != 90 && a.rotation != 180 && a.rotation != 270) {
        std::cerr << "[Error] --rotation 은 0|90|180|270 만 허용\n";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// 전처리: gamma → mirror → rotation
// ---------------------------------------------------------------------------

// gamma 보정 (BGR uint8). gamma==1.0이면 무변형. 결정적 LUT 사용.
void applyGamma(cv::Mat& img, float gamma) {
    if (std::fabs(gamma - 1.0f) < 1e-9f) {
        return;
    }
    cv::Mat lut(1, 256, CV_8U);
    uint8_t* p = lut.ptr();
    for (int i = 0; i < 256; ++i) {
        double v = std::pow(i / 255.0, static_cast<double>(gamma)) * 255.0;
        int iv = static_cast<int>(v + 0.5);
        p[i] = static_cast<uint8_t>(iv < 0 ? 0 : (iv > 255 ? 255 : iv));
    }
    cv::LUT(img, lut, img);
}

// 좌우 반전
void applyMirror(cv::Mat& img, int mirror) {
    if (mirror) {
        cv::flip(img, img, 1);  // 1 = 수평 반전
    }
}

// 이미지를 실제로 회전(rotation_degrees 시계방향). 90/270은 폭·높이가 바뀐다.
// 픽셀 손실 없는 90도 배수 회전(cv::rotate)을 사용해 결정성 보장.
void applyRotation(cv::Mat& img, int rotation) {
    switch (rotation) {
        case 90:
            cv::rotate(img, img, cv::ROTATE_90_CLOCKWISE);
            break;
        case 180:
            cv::rotate(img, img, cv::ROTATE_180);
            break;
        case 270:
            cv::rotate(img, img, cv::ROTATE_90_COUNTERCLOCKWISE);
            break;
        default:
            break;  // 0: 변형 없음
    }
}

// ---------------------------------------------------------------------------
// JSON 수동 직렬화 (결정적 필드만, %.6f 고정)
// ---------------------------------------------------------------------------
class JsonWriter {
public:
    explicit JsonWriter(std::ostream& os) : os_(os) {}

    void floatField(const char* key, float v, bool comma = true) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.6f", v);
        os_ << "  \"" << key << "\": " << buf << (comma ? ",\n" : "\n");
    }
    void intField(const char* key, long long v, bool comma = true) {
        os_ << "  \"" << key << "\": " << v << (comma ? ",\n" : "\n");
    }
    void boolField(const char* key, bool v, bool comma = true) {
        os_ << "  \"" << key << "\": " << (v ? "true" : "false") << (comma ? ",\n" : "\n");
    }
    void strField(const char* key, const std::string& v, bool comma = true) {
        os_ << "  \"" << key << "\": \"" << v << "\"" << (comma ? ",\n" : "\n");
    }
private:
    std::ostream& os_;
};

// IrisLandmark 1개를 "{x,y,z,visibility}" 형태로 직렬화
std::string landmarkToStr(const IrisLandmark& lm, bool include_vis) {
    char buf[256];
    if (include_vis) {
        std::snprintf(buf, sizeof(buf),
                      "{\"x\": %.6f, \"y\": %.6f, \"z\": %.6f, \"visibility\": %.6f}",
                      lm.x, lm.y, lm.z, lm.visibility);
    } else {
        std::snprintf(buf, sizeof(buf),
                      "{\"x\": %.6f, \"y\": %.6f, \"z\": %.6f}",
                      lm.x, lm.y, lm.z);
    }
    return buf;
}

// 결과를 JSON으로 덤프. 타임스탬프/포인터 등 비결정 필드는 제외.
void writeResultJson(const fs::path& path,
                     const IrisResult& r,
                     const Args& a,
                     int eff_width,
                     int eff_height) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "[Error] JSON 출력 열기 실패: " << path << "\n";
        return;
    }
    f << "{\n";
    JsonWriter w(f);

    // ---- 캡처 메타(변형 매트릭스 식별. 비결정 아님) ----
    w.strField("schema", "iris_golden_v1");
    w.strField("model_version", iris_sdk_get_version());
    w.intField("rotation", a.rotation);
    w.intField("mirror", a.mirror);
    {
        char gbuf[32];
        std::snprintf(gbuf, sizeof(gbuf), "%.6f", a.gamma);
        f << "  \"gamma\": " << gbuf << ",\n";
    }
    // 회전 적용 후 SDK에 들어간 유효 해상도
    w.intField("input_width", eff_width);
    w.intField("input_height", eff_height);

    // ---- 검출 상태 ----
    w.boolField("detected", r.detected);
    w.boolField("left_detected", r.left_detected);
    w.boolField("right_detected", r.right_detected);
    w.floatField("confidence", r.confidence);

    // ---- 좌/우 홍채 5점 ----
    f << "  \"left_iris\": [";
    for (int i = 0; i < 5; ++i) {
        f << landmarkToStr(r.left_iris[i], true) << (i < 4 ? ", " : "");
    }
    f << "],\n";
    w.floatField("left_radius", r.left_radius);

    f << "  \"right_iris\": [";
    for (int i = 0; i < 5; ++i) {
        f << landmarkToStr(r.right_iris[i], true) << (i < 4 ? ", " : "");
    }
    f << "],\n";
    w.floatField("right_radius", r.right_radius);

    // ---- 얼굴 메타 ----
    {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "{\"x\": %.6f, \"y\": %.6f, \"width\": %.6f, \"height\": %.6f}",
                      r.face_rect.x, r.face_rect.y, r.face_rect.width, r.face_rect.height);
        f << "  \"face_rect\": " << buf << ",\n";
    }
    {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "[%.6f, %.6f, %.6f]",
                      r.face_rotation[0], r.face_rotation[1], r.face_rotation[2]);
        f << "  \"face_rotation\": " << buf << ",\n";
    }

    // ---- Eye Refiner / 품질 메타 ----
    w.floatField("iris_quality_left", r.iris_quality_left);
    w.floatField("iris_quality_right", r.iris_quality_right);
    w.floatField("eyelid_ratio_left", r.eyelid_ratio_left);
    w.floatField("eyelid_ratio_right", r.eyelid_ratio_right);
    w.boolField("eye_refiner_used", r.eye_refiner_used);
    w.floatField("avg_iris_luma_left", r.avg_iris_luma_left);
    w.floatField("avg_iris_luma_right", r.avg_iris_luma_right);

    // ---- Face Mesh (478점 전체) ----
    w.boolField("face_mesh_valid", r.face_mesh_valid);
    w.intField("face_mesh_count", r.face_mesh_valid ? 478 : 0);
    f << "  \"face_mesh\": [";
    if (r.face_mesh_valid) {
        f << "\n";
        for (int i = 0; i < 478; ++i) {
            f << "    " << landmarkToStr(r.face_mesh[i], false)
              << (i < 477 ? ",\n" : "\n");
        }
        f << "  ]\n";
    } else {
        f << "]\n";  // 마지막 필드 (콤마 없음)
    }

    f << "}\n";
}

// ---------------------------------------------------------------------------
// 기본 V2 beauty 설정에서 결정적 효과만 켠다(GPU/vivid 제외).
// CPU 경로(iris_sdk_apply_beauty_v2_c)만 데스크톱 골든 대상.
// ---------------------------------------------------------------------------
void fillBeautyConfig(IrisBeautyConfigV2& c) {
    iris_sdk_default_beauty_config_v2_c(&c);
    c.enabled = 1;
    c.use_gpu = 0;          // CPU 경로 강제
    c.intensity = 0.7f;
    c.smoothing = 0.5f;
    c.brightness = 1.0f;
    c.skin_quality = 0.5f;
    // vivid(GPU 전용)는 0으로 유지
}

}  // namespace

int main(int argc, char* argv[]) {
    Args a;
    if (!parseArgs(argc, argv, a)) {
        return 2;
    }

    const fs::path input_path = fs::absolute(a.input);
    if (!fs::exists(input_path)) {
        std::cerr << "[Error] 입력 파일 없음: " << input_path << "\n";
        return 2;
    }
    fs::create_directories(a.out);

    const std::string stem =
        a.out_stem.empty() ? input_path.stem().string() : a.out_stem;

    // SDK 초기화 (CPU만)
    iris_sdk_set_gpu_enabled(false);
    IrisSdkError err = iris_sdk_init(a.models.c_str());
    if (err != IRIS_SDK_OK) {
        std::cerr << "[Error] SDK init 실패: " << iris_sdk_error_to_string(err)
                  << " — " << iris_sdk_get_last_error() << "\n";
        return 1;
    }

    bool has_texture = false;
    if (!a.texture.empty()) {
        err = iris_sdk_load_texture(a.texture.c_str());
        has_texture = (err == IRIS_SDK_OK);
        if (!has_texture) {
            std::cerr << "[Warning] 텍스처 로드 실패: "
                      << iris_sdk_error_to_string(err) << "\n";
        }
    }

    // 입력 로드 (BGR)
    cv::Mat image = cv::imread(input_path.string(), cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "[Error] 이미지 로드 실패: " << input_path << "\n";
        iris_sdk_destroy();
        return 1;
    }

    // ---- 전처리: gamma → mirror → rotation ----
    applyGamma(image, a.gamma);
    applyMirror(image, a.mirror);
    applyRotation(image, a.rotation);

    // OpenCV는 연속 메모리가 아닐 수 있으므로 보장
    if (!image.isContinuous()) {
        image = image.clone();
    }

    const int eff_w = image.cols;
    const int eff_h = image.rows;

    // ---- 검출: 회전 각도를 SDK에 전달 (blocker 커버 핵심) ----
    IrisResult result;
    std::memset(&result, 0, sizeof(result));
    err = iris_sdk_detect_with_rotation(
        image.data, eff_w, eff_h, IRIS_FORMAT_BGR, a.rotation, &result);
    if (err != IRIS_SDK_OK) {
        std::cerr << "[Error] detect 실패: " << iris_sdk_error_to_string(err)
                  << " — " << iris_sdk_get_last_error() << "\n";
        iris_sdk_free_result(&result);
        iris_sdk_destroy();
        return 1;
    }

    // ---- 덤프 ①: JSON ----
    const fs::path json_path = fs::path(a.out) / (stem + ".result.json");
    writeResultJson(json_path, result, a, eff_w, eff_h);

    // ---- 덤프 ②a: 렌즈 렌더 PNG ----
    if (a.render && has_texture && result.detected) {
        cv::Mat render = image.clone();
        if (!render.isContinuous()) {
            render = render.clone();
        }
        IrisLensConfig lens_cfg;
        iris_sdk_default_lens_config(&lens_cfg);
        lens_cfg.is_mirror = (a.lens_mirror != 0);  // 전면 카메라 렌더 분기 커버
        err = iris_sdk_render_lens(
            render.data, render.cols, render.rows, IRIS_FORMAT_BGR,
            &result, &lens_cfg);
        if (err == IRIS_SDK_OK) {
            const fs::path render_path = fs::path(a.out) / (stem + ".render.png");
            // PNG 무압축 차이 회피를 위해 결정적 압축 레벨 고정
            std::vector<int> png_params = {cv::IMWRITE_PNG_COMPRESSION, 6};
            cv::imwrite(render_path.string(), render, png_params);
        } else {
            std::cerr << "[Warning] render_lens 실패: "
                      << iris_sdk_error_to_string(err) << "\n";
        }
    }

    // ---- 덤프 ②b: CPU beauty PNG ----
    if (a.beauty) {
        cv::Mat beauty = image.clone();
        if (!beauty.isContinuous()) {
            beauty = beauty.clone();
        }
        IrisBeautyConfigV2 bcfg;
        fillBeautyConfig(bcfg);
        err = iris_sdk_apply_beauty_v2_c(
            beauty.data, beauty.cols, beauty.rows, IRIS_FORMAT_BGR,
            &bcfg, result.detected ? &result : nullptr);
        if (err == IRIS_SDK_OK) {
            const fs::path beauty_path = fs::path(a.out) / (stem + ".beauty.png");
            std::vector<int> png_params = {cv::IMWRITE_PNG_COMPRESSION, 6};
            cv::imwrite(beauty_path.string(), beauty, png_params);
        } else {
            std::cerr << "[Warning] beauty 실패: "
                      << iris_sdk_error_to_string(err) << "\n";
        }
    }

    std::cout << "[golden_capture] " << stem
              << " detected=" << (result.detected ? 1 : 0)
              << " conf=" << result.confidence
              << " rot=" << a.rotation
              << " mirror=" << a.mirror
              << " gamma=" << a.gamma
              << " -> " << json_path.filename().string() << "\n";

    iris_sdk_free_result(&result);
    iris_sdk_destroy();
    return 0;
}
