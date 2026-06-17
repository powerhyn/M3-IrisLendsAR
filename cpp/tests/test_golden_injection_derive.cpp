/**
 * @file test_golden_injection_derive.cpp
 * @brief ④ W4-D 안전망: 18골든 전체 injection-derive 회전 정합 검증
 *
 * detector 코어(mediapipe_detector) 제거 후, 골든 재캡처는 detector 대신 주입 경로
 * (iris_set_landmarks → iris_get_injected_result)로 수행한다(golden_capture --inject-from).
 * 본 테스트는 그 재캡처가 detector 베이스라인과 ε 일치함을 **18골든 전체**(회전 포함)에서
 * 증명하는 안전망이다. 통과해야 detector 제거가 안전하다.
 *
 * 기존 test_landmark_injection.cpp는 face_closeup__gamma05(비회전) 1케이스만 검증하여
 * 회전 치수 스왑 문제를 못 잡는다. 본 테스트는 18골든을 parametrize하고, 각 케이스에
 * **회전 보정 upright 치수**를 적용한다.
 *
 * §회전 치수 스왑(이 테스트의 정수 — Codex 실측 안전망):
 *   deriveIrisResult는 홍채 반경을 픽셀 공간(정규화좌표×frame_dim)에서 계산한다. detector는
 *   내부적으로 회전 후(upright) 치수로 반경을 산출하는데, rot90/270은 회전으로 W↔H가
 *   스왑된다. baseline JSON의 input_width/input_height는 "회전 적용 후" 캡처 치수이므로
 *   주입 시 다시 upright로 되돌려야 한다:
 *     rotation∈{90,270} → (frame_w,frame_h) = (input_height,input_width),
 *     그 외             → (input_width,input_height).
 *   미스왑 시 rot90/270 반경이 0.04~0.34px 어긋난다(스왑 시 ~2e-4px ε). 본 테스트가
 *   rot90/270에서 PASS하면 golden_capture --inject-from의 스왑 처리가 옳음을 입증한다.
 *
 * 비교 범위(geometry만 — ADR §6.2):
 *   - left/right_iris[0] center(x,y,z), left/right_radius, face_rect를 ε 비교.
 *   - 비교 제외: iris_quality/eye_refiner(injection=0), confidence(injection=detected?1:0),
 *     avg_iris_luma(injection=-1, RGB 미보유). 측정값이 아니라 detector 전용 메타라
 *     주입이 재현할 수 없는 필드.
 *
 * JSON 파싱은 표준 라이브러리만(외부 의존 금지 — golden_compare.py 정책). baseline은
 * 평탄·결정적 포맷이라 경량 토큰 파서로 충분하다(test_landmark_injection.cpp 거울).
 */

#include "iris_sdk/sdk_api.h"

#include <gtest/gtest.h>

#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

// 정규화 좌표 ε (golden_compare.py 기본 — center, face_rect).
constexpr float kEpsNorm = 1e-4f;
// 반경(픽셀) ε. golden_compare.py 기본은 0.5px이지만, 이는 detector 자체의 변형 대조용으로
// 느슨하다. injection-derive는 detector와 **동일 입력(478점)·동일 수식**으로 반경을 재계산하므로
// 올바른 치수(upright)에서는 float 정밀도 수준(~2e-4px)까지 일치한다(측정 최대 2.13e-4px).
// 반면 회전 치수를 미스왑하면 rot90/270 반경이 0.04~0.34px 어긋난다 — 기본 0.5px ε는
// 이를 못 잡으므로(여유 0.16px) 안전망이 무의미해진다. ε를 0.01px로 좁혀 미스왑을 확실히
// 잡는다(2e-4 통과 ↔ 0.04 실패 사이 안전 마진 0.01). 이 ε가 본 안전망을 load-bearing하게 만든다.
constexpr float kEpsPix = 0.01f;

#ifndef GOLDEN_BASELINE_DIR
#define GOLDEN_BASELINE_DIR ""
#endif

// ---------------------------------------------------------------------------
// 경량 JSON 추출기 (baseline result.json 전용 — 평탄·결정적 포맷 가정)
// ---------------------------------------------------------------------------

struct GoldenBaseline {
    bool ok = false;
    int rotation = 0;
    int input_width = 0;
    int input_height = 0;
    std::vector<float> mesh_flat;            // 478×3 (x,y,z) — 주입 입력
    // detector 파생 기준값(geometry만).
    std::array<float, 3> left_center{};      // left_iris[0] (x,y,z)
    std::array<float, 3> right_center{};     // right_iris[0] (x,y,z)
    float left_radius = 0.0f;
    float right_radius = 0.0f;
    float face_rect[4] = {0, 0, 0, 0};       // x,y,width,height
    bool left_detected = false;
    bool right_detected = false;
};

// "key": <number> 스칼라 추출. 실패 시 false.
bool extractScalar(const std::string& s, const std::string& key, double& out) {
    const std::string needle = "\"" + key + "\"";
    size_t p = s.find(needle);
    if (p == std::string::npos) return false;
    p = s.find(':', p);
    if (p == std::string::npos) return false;
    ++p;
    while (p < s.size() && (s[p] == ' ' || s[p] == '\t' || s[p] == '\n')) ++p;
    const size_t start = p;
    while (p < s.size() &&
           (std::isdigit(static_cast<unsigned char>(s[p])) || s[p] == '-' ||
            s[p] == '+' || s[p] == '.' || s[p] == 'e' || s[p] == 'E')) {
        ++p;
    }
    if (p == start) return false;
    try {
        out = std::stod(s.substr(start, p - start));
    } catch (...) {
        return false;
    }
    return true;
}

bool extractBool(const std::string& s, const std::string& key, bool& out) {
    const std::string needle = "\"" + key + "\"";
    size_t p = s.find(needle);
    if (p == std::string::npos) return false;
    p = s.find(':', p);
    if (p == std::string::npos) return false;
    size_t q = p + 1;
    while (q < s.size() && (s[q] == ' ' || s[q] == '\t')) ++q;
    out = (q < s.size() && s[q] == 't');
    return true;
}

// 배열 블록에서 숫자 토큰을 순서대로 수집한다(공통 헬퍼).
std::vector<double> collectNumbers(const std::string& block) {
    std::vector<double> nums;
    size_t i = 0;
    while (i < block.size()) {
        const char c = block[i];
        if (c == '-' || c == '+' || c == '.' ||
            std::isdigit(static_cast<unsigned char>(c))) {
            const size_t start = i;
            while (i < block.size() &&
                   (std::isdigit(static_cast<unsigned char>(block[i])) ||
                    block[i] == '-' || block[i] == '+' || block[i] == '.' ||
                    block[i] == 'e' || block[i] == 'E')) {
                ++i;
            }
            nums.push_back(std::stod(block.substr(start, i - start)));
        } else {
            ++i;
        }
    }
    return nums;
}

// "face_mesh": [ {"x":..,"y":..,"z":..}, ... ] → 평탄 478×3.
// golden_capture는 face_mesh를 visibility 없이 (x,y,z) 3토큰으로 덤프한다.
bool extractMeshFlat(const std::string& s, std::vector<float>& mesh_flat) {
    const size_t key = s.find("\"face_mesh\"");
    if (key == std::string::npos) return false;
    const size_t lb = s.find('[', key);
    if (lb == std::string::npos) return false;
    const size_t rb = s.find(']', lb);
    if (rb == std::string::npos) return false;
    const std::vector<double> nums = collectNumbers(s.substr(lb + 1, rb - lb - 1));
    if (nums.size() < static_cast<size_t>(478 * 3)) return false;
    mesh_flat.resize(478 * 3);
    for (int k = 0; k < 478 * 3; ++k) {
        mesh_flat[k] = static_cast<float>(nums[k]);
    }
    return true;
}

// left_iris / right_iris 배열(5점×4토큰 {x,y,z,visibility})에서 center(인덱스 0)만 추출.
bool extractIrisCenter(const std::string& s, const std::string& key,
                       std::array<float, 3>& center) {
    const std::string needle = "\"" + key + "\"";
    const size_t k = s.find(needle);
    if (k == std::string::npos) return false;
    const size_t lb = s.find('[', k);
    if (lb == std::string::npos) return false;
    const size_t rb = s.find(']', lb);
    if (rb == std::string::npos) return false;
    const std::vector<double> nums = collectNumbers(s.substr(lb + 1, rb - lb - 1));
    if (nums.size() < 3) return false;
    center[0] = static_cast<float>(nums[0]);
    center[1] = static_cast<float>(nums[1]);
    center[2] = static_cast<float>(nums[2]);
    return true;
}

bool extractFaceRect(const std::string& s, float out[4]) {
    const size_t k = s.find("\"face_rect\"");
    if (k == std::string::npos) return false;
    const size_t lb = s.find('{', k);
    if (lb == std::string::npos) return false;
    const size_t rb = s.find('}', lb);
    if (rb == std::string::npos) return false;
    const std::string block = s.substr(lb, rb - lb + 1);
    double x = 0, y = 0, w = 0, h = 0;
    if (!extractScalar(block, "x", x)) return false;
    if (!extractScalar(block, "y", y)) return false;
    if (!extractScalar(block, "width", w)) return false;
    if (!extractScalar(block, "height", h)) return false;
    out[0] = static_cast<float>(x);
    out[1] = static_cast<float>(y);
    out[2] = static_cast<float>(w);
    out[3] = static_cast<float>(h);
    return true;
}

GoldenBaseline loadBaseline(const std::string& path) {
    GoldenBaseline g;
    std::ifstream f(path, std::ios::binary);
    if (!f) return g;
    const std::string s((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());

    double rot = 0, w = 0, h = 0, lr = 0, rr = 0;
    if (!extractScalar(s, "rotation", rot)) return g;
    if (!extractScalar(s, "input_width", w)) return g;
    if (!extractScalar(s, "input_height", h)) return g;
    if (!extractMeshFlat(s, g.mesh_flat)) return g;
    if (!extractIrisCenter(s, "left_iris", g.left_center)) return g;
    if (!extractIrisCenter(s, "right_iris", g.right_center)) return g;
    extractScalar(s, "left_radius", lr);
    extractScalar(s, "right_radius", rr);
    if (!extractFaceRect(s, g.face_rect)) return g;
    extractBool(s, "left_detected", g.left_detected);
    extractBool(s, "right_detected", g.right_detected);

    g.rotation = static_cast<int>(rot);
    g.input_width = static_cast<int>(w);
    g.input_height = static_cast<int>(h);
    g.left_radius = static_cast<float>(lr);
    g.right_radius = static_cast<float>(rr);
    g.ok = true;
    return g;
}

std::string baselinePath(const std::string& stem) {
    std::string dir = GOLDEN_BASELINE_DIR;
    if (dir.empty()) {
        dir = "../tests/golden/baseline";  // CMake 미정의 시 소스 기준 fallback.
    }
    return dir + "/" + stem + ".result.json";
}

// 18골든 stem 전체(입력 3장 × rotation/mirror/gamma 변형).
const std::vector<std::string>& allGoldenStems() {
    static const std::vector<std::string> kStems = {
        "face_closeup__gamma05",
        "face_closeup__lensmirror",
        "face_closeup__mirror",
        "face_closeup__mirror_rot270",
        "face_closeup__rot0",
        "face_closeup__rot180",
        "face_closeup__rot270",
        "face_closeup__rot90",
        "screenshot_640a__mirror",
        "screenshot_640a__rot0",
        "screenshot_640a__rot180",
        "screenshot_640a__rot270",
        "screenshot_640a__rot90",
        "screenshot_720a__mirror",
        "screenshot_720a__rot0",
        "screenshot_720a__rot180",
        "screenshot_720a__rot270",
        "screenshot_720a__rot90",
    };
    return kStems;
}

}  // namespace

// ===========================================================================
// Parametrized: 18골든 전체 injection-derive ε 검증 (C API 표면 경유)
// ===========================================================================
//
// 각 케이스: baseline JSON에서 face_mesh 478점 + rotation + input dims를 읽어,
// upright 치수로 iris_set_landmarks(C API) → iris_get_injected_result로 재파생한 뒤
// baseline의 detector 파생 geometry(center/radius/face_rect)와 ε 비교한다.
// C API는 프로세스 전역 g_landmark_store를 공유하므로 각 케이스가 자기 주입을
// 직전에 수행해 결정적 상태(last-writer-wins)를 보장한다.

class GoldenInjectionDeriveTest : public ::testing::TestWithParam<std::string> {};

TEST_P(GoldenInjectionDeriveTest, MatchesDetectorBaselineWithRotationSwap) {
    const std::string stem = GetParam();
    const std::string path = baselinePath(stem);
    const GoldenBaseline g = loadBaseline(path);
    ASSERT_TRUE(g.ok) << "골든 baseline 로드/파싱 실패: " << path;
    ASSERT_EQ(g.mesh_flat.size(), static_cast<size_t>(478 * 3));
    ASSERT_GT(g.input_width, 0);
    ASSERT_GT(g.input_height, 0);

    // ----- §회전 치수 스왑: input_*(회전 적용 후) → upright -----
    const bool rot_swaps = (g.rotation == 90 || g.rotation == 270);
    const int upright_w = rot_swaps ? g.input_height : g.input_width;
    const int upright_h = rot_swaps ? g.input_width : g.input_height;

    // ----- 주입 → 조회 (공개 C 경계) -----
    uint32_t generation = 0;
    ASSERT_EQ(iris_set_landmarks(g.mesh_flat.data(), 478, upright_w, upright_h,
                                 /*timestamp_us=*/0, &generation),
              IRIS_SDK_OK)
        << "iris_set_landmarks 실패: " << stem;
    EXPECT_EQ(generation & 1u, 0u) << "공개 generation은 완결(짝수)이어야 함";

    ::IrisResult r{};
    ASSERT_EQ(iris_get_injected_result(&r), IRIS_SDK_OK)
        << "iris_get_injected_result 실패: " << stem;

    // ----- detected 일치 (detector도 홍채 5점 [0,1] 판정) -----
    EXPECT_EQ(r.left_detected, g.left_detected) << "left_detected: " << stem;
    EXPECT_EQ(r.right_detected, g.right_detected) << "right_detected: " << stem;

    // ----- 홍채 center(x,y,z) 정규화 좌표: eps_norm -----
    if (g.left_detected) {
        EXPECT_NEAR(r.left_iris[0].x, g.left_center[0], kEpsNorm) << "left_iris[0].x: " << stem;
        EXPECT_NEAR(r.left_iris[0].y, g.left_center[1], kEpsNorm) << "left_iris[0].y: " << stem;
        EXPECT_NEAR(r.left_iris[0].z, g.left_center[2], kEpsNorm) << "left_iris[0].z: " << stem;
    }
    if (g.right_detected) {
        EXPECT_NEAR(r.right_iris[0].x, g.right_center[0], kEpsNorm) << "right_iris[0].x: " << stem;
        EXPECT_NEAR(r.right_iris[0].y, g.right_center[1], kEpsNorm) << "right_iris[0].y: " << stem;
        EXPECT_NEAR(r.right_iris[0].z, g.right_center[2], kEpsNorm) << "right_iris[0].z: " << stem;
    }

    // ----- 반경(픽셀): eps_pix. rot90/270은 스왑이 옳아야만 통과(이 테스트의 정수) -----
    if (g.left_detected) {
        EXPECT_NEAR(r.left_radius, g.left_radius, kEpsPix)
            << "left_radius: " << stem << " (rot=" << g.rotation
            << ", upright=" << upright_w << "x" << upright_h << ")";
    }
    if (g.right_detected) {
        EXPECT_NEAR(r.right_radius, g.right_radius, kEpsPix)
            << "right_radius: " << stem << " (rot=" << g.rotation
            << ", upright=" << upright_w << "x" << upright_h << ")";
    }

    // ----- face_rect(정규화 메시 바운딩): eps_norm -----
    EXPECT_NEAR(r.face_rect.x, g.face_rect[0], kEpsNorm) << "face_rect.x: " << stem;
    EXPECT_NEAR(r.face_rect.y, g.face_rect[1], kEpsNorm) << "face_rect.y: " << stem;
    EXPECT_NEAR(r.face_rect.width, g.face_rect[2], kEpsNorm) << "face_rect.width: " << stem;
    EXPECT_NEAR(r.face_rect.height, g.face_rect[3], kEpsNorm) << "face_rect.height: " << stem;
}

INSTANTIATE_TEST_SUITE_P(
    AllGoldens, GoldenInjectionDeriveTest,
    ::testing::ValuesIn(allGoldenStems()),
    [](const ::testing::TestParamInfo<std::string>& info) {
        // 케이스 이름은 stem(영숫자·언더스코어만 허용). "__"는 그대로 통과한다.
        return info.param;
    });

// ===========================================================================
// 음성 대조: 회전 치수 스왑이 load-bearing임을 직접 증명
// ===========================================================================
//
// 위 parametrized 테스트는 "스왑하면 통과"를 보인다. 본 테스트는 그 대우(對偶)인
// "스왑하지 않으면 rot90/270에서 반경이 어긋난다"를 직접 고정한다. 미래에 누군가 스왑을
// 제거하거나 input dims를 그대로 쓰면(Codex가 경고한 함정) 본 테스트가 즉시 깨진다.
// rot90 케이스에 대해 (a) 올바른 upright(스왑) 주입과 (b) 미스왑(input dims 그대로) 주입을
// 모두 수행하고, (a)는 baseline과 ε 일치, (b)는 baseline에서 명확히 벗어남을 확인한다.
TEST(GoldenInjectionRotationSwapControl, MisswappedDimsDiverge) {
    const std::string stem = "screenshot_720a__rot90";  // 미스왑 오차 최대(0.34px) 케이스.
    const GoldenBaseline g = loadBaseline(baselinePath(stem));
    ASSERT_TRUE(g.ok) << "골든 baseline 로드 실패: " << stem;
    ASSERT_EQ(g.rotation, 90) << "본 음성 대조는 회전 스왑 케이스 전제";
    ASSERT_TRUE(g.left_detected);

    // (a) 올바른 upright 치수(스왑): rot90이므로 (input_height, input_width).
    {
        uint32_t gen = 0;
        ASSERT_EQ(iris_set_landmarks(g.mesh_flat.data(), 478,
                                     g.input_height, g.input_width, 0, &gen),
                  IRIS_SDK_OK);
        ::IrisResult r{};
        ASSERT_EQ(iris_get_injected_result(&r), IRIS_SDK_OK);
        EXPECT_NEAR(r.left_radius, g.left_radius, kEpsPix)
            << "스왑(upright) 주입은 baseline과 ε 일치해야 한다";
    }

    // (b) 미스왑(input dims 그대로): rot90에서 반경이 ε(kEpsPix)를 명확히 넘어야 한다.
    {
        uint32_t gen = 0;
        ASSERT_EQ(iris_set_landmarks(g.mesh_flat.data(), 478,
                                     g.input_width, g.input_height, 0, &gen),
                  IRIS_SDK_OK);
        ::IrisResult r{};
        ASSERT_EQ(iris_get_injected_result(&r), IRIS_SDK_OK);
        const float err = std::abs(r.left_radius - g.left_radius);
        EXPECT_GT(err, kEpsPix)
            << "미스왑(input dims 그대로) 주입이 rot90에서 baseline과 일치해버렸다 — "
               "스왑이 load-bearing이 아니라는 신호(안전망 무력). err=" << err << "px";
    }
}
