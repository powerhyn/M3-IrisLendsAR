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
 *                  [--inject-from <baseline.result.json>]
 *
 *   --no-render   : 렌즈 렌더 PNG 생략 (JSON 전용 변형 — 용량 규율)
 *   --lens-mirror : IrisLensConfig.is_mirror=true 로 렌더 (전면 카메라 렌더 분기 커버)
 *   --inject-from : ④ W4-D 주입 모드. detector(iris_sdk_detect_with_rotation) 대신
 *                   소스 baseline의 face_mesh 478점을 iris_set_landmarks로 주입하고
 *                   iris_get_injected_result로 결과를 재파생한다. detector 코어 제거 후
 *                   골든 재캡처용 경로. 렌더 배경(gamma→mirror→rotation 전처리)은 유지하되
 *                   검출만 주입으로 대체한다. 주입 치수는 회전 스왑 보정(아래 §주입 치수).
 *
 * §주입 치수(중요 — rot90/270 radius 정합):
 *   deriveIrisResult는 홍채 반경을 픽셀 공간(정규화좌표×frame_dim)에서 계산한다.
 *   detector는 내부적으로 "회전 후(upright)" 치수로 반경을 산출하는데, rot90/270은
 *   회전으로 W↔H가 스왑된다. baseline JSON의 input_width/input_height는 "회전 적용 후"
 *   캡처 치수(rot90이면 W·H가 이미 스왑된 값)이므로, 주입 시에는 이를 다시 upright로
 *   되돌려야 한다: rotation∈{90,270} → (frame_w,frame_h) = (input_height,input_width),
 *   그 외 → (input_width,input_height). 스왑하지 않으면 rot90/270 반경이 0.04~0.34px
 *   어긋난다(스왑 시 ~2e-4px ε). 검증: test_golden_injection_derive.
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
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
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
    // ④ W4-D: injection 모드. 비어 있으면 detector 경로(detect), 지정 시 주입 경로.
    // 소스 baseline result.json에서 face_mesh 478점 + rotation + input dims를 읽어
    // iris_set_landmarks(upright dims) → iris_get_injected_result로 재파생한다.
    std::string inject_from;  // <baseline.result.json> 경로 (빈 문자열 = detect 모드)
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
        } else if (key == "--inject-from") {
            a.inject_from = next("--inject-from");
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

// ---------------------------------------------------------------------------
// ④ W4-D 주입 모드용 경량 JSON 파서 (baseline result.json 전용)
//
// 외부 의존성 추가 금지(헤더 docstring): nlohmann 등 미사용. baseline JSON은
// golden_capture 자신이 쓴 평탄·결정적 포맷이라 토큰 파서로 충분하다. test_landmark_
// injection.cpp의 추출기와 동일 전략(키 검색 후 숫자 토큰 스캔)을 거울 구현한다.
// 추출 대상은 재파생에 필요한 최소 필드: rotation, input_width/height, face_mesh 478×3.
// ---------------------------------------------------------------------------

// face_mesh 478점을 평탄 478×3 (x,y,z) 버퍼로 추출. golden_capture 출력 포맷은
// {"x": <n>, "y": <n>, "z": <n>} (visibility 미포함, landmarkToStr include_vis=false).
struct GoldenSource {
    bool ok = false;
    int rotation = 0;
    int input_width = 0;
    int input_height = 0;
    std::vector<float> mesh_flat;  // 478×3 (x,y,z)
};

// "key": <number> 형태의 스칼라를 추출(정수/실수 공통). 실패 시 false.
bool jsonExtractScalar(const std::string& s, const char* key, double& out) {
    std::string needle = std::string("\"") + key + "\"";
    size_t p = s.find(needle);
    if (p == std::string::npos) {
        return false;
    }
    p = s.find(':', p);
    if (p == std::string::npos) {
        return false;
    }
    ++p;
    while (p < s.size() && (s[p] == ' ' || s[p] == '\t' || s[p] == '\n')) {
        ++p;
    }
    size_t start = p;
    while (p < s.size() &&
           (std::isdigit(static_cast<unsigned char>(s[p])) || s[p] == '-' ||
            s[p] == '+' || s[p] == '.' || s[p] == 'e' || s[p] == 'E')) {
        ++p;
    }
    if (p == start) {
        return false;
    }
    try {
        out = std::stod(s.substr(start, p - start));
    } catch (...) {
        return false;
    }
    return true;
}

// "face_mesh": [ {"x":..,"y":..,"z":..}, ... ] 블록에서 478×3 숫자를 순서대로 추출.
// '['는 face_mesh 하나뿐(다른 배열 left_iris/right_iris는 객체 직전에 등장하지만 키로 식별).
bool jsonExtractMesh(const std::string& s, std::vector<float>& mesh_flat) {
    size_t key = s.find("\"face_mesh\"");
    if (key == std::string::npos) {
        return false;
    }
    size_t lb = s.find('[', key);
    if (lb == std::string::npos) {
        return false;
    }
    size_t rb = s.find(']', lb);
    if (rb == std::string::npos) {
        return false;
    }
    const std::string block = s.substr(lb + 1, rb - lb - 1);
    std::vector<double> nums;
    nums.reserve(478 * 3);
    size_t i = 0;
    while (i < block.size()) {
        const char c = block[i];
        if (c == '-' || c == '+' || c == '.' ||
            std::isdigit(static_cast<unsigned char>(c))) {
            size_t start = i;
            while (i < block.size() &&
                   (std::isdigit(static_cast<unsigned char>(block[i])) ||
                    block[i] == '-' || block[i] == '+' || block[i] == '.' ||
                    block[i] == 'e' || block[i] == 'E')) {
                ++i;
            }
            try {
                nums.push_back(std::stod(block.substr(start, i - start)));
            } catch (...) {
                return false;
            }
        } else {
            ++i;
        }
    }
    if (nums.size() < static_cast<size_t>(478 * 3)) {
        return false;
    }
    mesh_flat.resize(478 * 3);
    for (int k = 0; k < 478 * 3; ++k) {
        mesh_flat[k] = static_cast<float>(nums[k]);
    }
    return true;
}

// baseline result.json을 읽어 주입 재파생에 필요한 필드를 추출한다.
GoldenSource loadGoldenSource(const fs::path& path) {
    GoldenSource g;
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "[Error] --inject-from 소스 열기 실패: " << path << "\n";
        return g;
    }
    std::string s((std::istreambuf_iterator<char>(f)),
                  std::istreambuf_iterator<char>());

    double rot = 0, w = 0, h = 0;
    if (!jsonExtractScalar(s, "rotation", rot)) {
        std::cerr << "[Error] 소스 JSON에 rotation 없음: " << path << "\n";
        return g;
    }
    if (!jsonExtractScalar(s, "input_width", w) ||
        !jsonExtractScalar(s, "input_height", h)) {
        std::cerr << "[Error] 소스 JSON에 input_width/height 없음: " << path << "\n";
        return g;
    }
    if (!jsonExtractMesh(s, g.mesh_flat)) {
        std::cerr << "[Error] 소스 JSON face_mesh 478점 파싱 실패: " << path << "\n";
        return g;
    }
    g.rotation = static_cast<int>(rot);
    g.input_width = static_cast<int>(w);
    g.input_height = static_cast<int>(h);
    g.ok = true;
    return g;
}

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

    // ④ W4-D: deriveIrisResult가 반경을 픽셀 환산한 frame 치수(upright). detect 모드에서는
    // result.frame_width/height(=SDK가 detect에 받은 치수)이며, injection 모드에서는
    // 주입 시 사용한 upright 치수(rot90/270은 input과 W↔H 스왑)다. 향후 재캡처가 input_*만
    // 보고 치수를 잘못 추론(rot90/270 radius 어긋남)하지 않도록 명시 저장(Codex 권고).
    w.intField("frame_width", r.frame_width);
    w.intField("frame_height", r.frame_height);

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

    // ---- 품질 메타 ----
    // ④ W4-D: iris_quality_*/eye_refiner_used 필드 제거(detector 전용 메타).
    w.floatField("eyelid_ratio_left", r.eyelid_ratio_left);
    w.floatField("eyelid_ratio_right", r.eyelid_ratio_right);
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
        // ④ W4-E: CPU 골든 렌더는 deprecated cpu-render API를 정당하게 사용(동작 유지).
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        err = iris_sdk_load_texture(a.texture.c_str());
#pragma GCC diagnostic pop
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

    // ---- 검출 ----
    // ④ W4-D: detector 코어가 제거되어 detect 모드(iris_sdk_detect_with_rotation)는
    //   더 이상 존재하지 않는다. golden 재캡처는 injection 모드(--inject-from)만 사용한다:
    //   소스 baseline의 face_mesh 478점을 iris_set_landmarks(upright 치수)로 주입하고
    //   iris_get_injected_result로 재파생한다.
    IrisResult result;
    std::memset(&result, 0, sizeof(result));

    if (a.inject_from.empty()) {
        std::cerr << "[Error] ④ W4-D: detector 제거 — --inject-from <baseline.json> 필수.\n"
                  << "        golden 재캡처는 주입 경로(iris_set_landmarks)만 지원합니다.\n";
        iris_sdk_destroy();
        return 1;
    }
    {
        // ----- injection 모드(주입 경로 — detector 미사용) -----
        const GoldenSource src = loadGoldenSource(fs::absolute(a.inject_from));
        if (!src.ok) {
            // loadGoldenSource가 구체 사유를 stderr로 보고함.
            iris_sdk_destroy();
            return 1;
        }
        // 주입 치수 = upright. baseline JSON의 input_*은 "회전 적용 후" 치수이므로
        // rot90/270은 W↔H가 이미 스왑돼 있다 → 다시 되돌려 detector가 반경 산출에 쓴
        // upright 공간으로 맞춘다(§주입 치수 — 미스왑 시 rot90/270 반경 0.04~0.34px 오차).
        const bool rot_swaps = (src.rotation == 90 || src.rotation == 270);
        const int upright_w = rot_swaps ? src.input_height : src.input_width;
        const int upright_h = rot_swaps ? src.input_width : src.input_height;

        uint32_t generation = 0;
        err = iris_set_landmarks(
            src.mesh_flat.data(), 478, upright_w, upright_h,
            /*timestamp_us=*/0, &generation);
        if (err != IRIS_SDK_OK) {
            std::cerr << "[Error] iris_set_landmarks 실패: "
                      << iris_sdk_error_to_string(err) << " — "
                      << iris_sdk_get_last_error() << "\n";
            iris_sdk_destroy();
            return 1;
        }
        err = iris_get_injected_result(&result);
        if (err != IRIS_SDK_OK) {
            std::cerr << "[Error] iris_get_injected_result 실패: "
                      << iris_sdk_error_to_string(err) << " — "
                      << iris_sdk_get_last_error() << "\n";
            iris_sdk_destroy();
            return 1;
        }
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
        // ④ W4-E: CPU 골든 렌더는 deprecated cpu-render API를 정당하게 사용(동작 유지).
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        err = iris_sdk_render_lens(
            render.data, render.cols, render.rows, IRIS_FORMAT_BGR,
            &result, &lens_cfg);
#pragma GCC diagnostic pop
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
        // ④ W4-E: CPU 골든 뷰티는 deprecated cpu-render API를 정당하게 사용(동작 유지).
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        err = iris_sdk_apply_beauty_v2_c(
            beauty.data, beauty.cols, beauty.rows, IRIS_FORMAT_BGR,
            &bcfg, result.detected ? &result : nullptr);
#pragma GCC diagnostic pop
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
              << " mode=" << (a.inject_from.empty() ? "detect" : "inject")
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
