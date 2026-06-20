/**
 * @file test_landmark_injection.cpp
 * @brief 랜드마크 주입 경계 단위 테스트 (ADR-0001 §6/§6.1/§6.2/§7)
 *
 * 두 축을 검증한다:
 *   1. 파생 어댑터 동작 불변: 골든 baseline JSON의 face_mesh 478점을 입력으로
 *      deriveIrisResult() 출력과 같은 JSON의 detector 파생값(left_iris/right_iris/
 *      left_radius/right_radius/face_rect)을 ε 비교. 골든 도구 ε(eps_norm=1e-4,
 *      eps_pix=0.5) 참조값 사용.
 *   2. seqlock 더블버퍼 + 입력 유효성: write/readDerived 라운드트립, generation 운용,
 *      478 외/NULL/NaN/Inf/치수≤0 거부 시 스테일 유지.
 *   3. C API 표면 경유 왕복(③-3 §3-3): 골든 baseline 478점을 iris_set_landmarks →
 *      iris_get_injected_result로 왕복시켜 detector 파생값과 ε 비교. ②가 store/어댑터를
 *      직접 호출하는 단위 검증이라면, ③은 공개 C 경계(sdk_api.h)를 통과하는 E2E다.
 *
 * JSON 파싱은 표준 라이브러리만으로 한다(golden_compare.py 정책과 일치 — 외부 의존 금지).
 * baseline JSON은 평탄·결정적 포맷이라 경량 토큰 파서로 충분하다.
 */

#include "iris_sdk/landmark_injection.h"
#include "iris_sdk/sdk_api.h"
#include "iris_sdk/types.h"
#include "iris_sdk/gpu/eye_render_packet_adapter.h"  // W4-B1: 어댑터 통과 visibility 검증

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

// sdk_api.h가 전역 ::IrisResult / ::IrisLandmark(C struct)를 도입하므로,
// C++ 타입은 별칭으로 명시해 전역 스코프(TEST 본문)에서의 이름 모호성을 피한다.
// (anonymous namespace의 using-directive 암묵 노출 + 전역 C typedef = ambiguous.)
using CppIrisResult = iris_sdk::IrisResult;
using CppIrisLandmark = iris_sdk::IrisLandmark;
using iris_sdk::LandmarkInjectionStore;
using iris_sdk::deriveIrisResult;

// 골든 도구 기본값과 동일 (golden_compare.py — ADR §11 참조값).
constexpr float kEpsNorm = 1e-4f;
constexpr float kEpsPix = 0.5f;

// CMake가 정의하는 baseline 디렉토리 경로 (없으면 소스 기준 상대경로 fallback).
#ifndef GOLDEN_BASELINE_DIR
#define GOLDEN_BASELINE_DIR ""
#endif

// ---------------------------------------------------------------------------
// 경량 JSON 추출기 (baseline result.json 전용 — 평탄·결정적 포맷 가정)
// ---------------------------------------------------------------------------

// face_mesh 478점을 추출한다. "face_mesh": [ {"x":..,"y":..,"z":..,"visibility":..}, ... ]
// 결정적 포맷이라 순서대로 숫자 토큰을 읽어 4개씩 묶는다.
struct GoldenResult {
    bool ok = false;
    std::vector<CppIrisLandmark> mesh;  // 478점
    int input_width = 0;
    int input_height = 0;
    // detector 파생 기준값
    std::array<CppIrisLandmark, 5> left_iris{};
    std::array<CppIrisLandmark, 5> right_iris{};
    float left_radius = 0.0f;
    float right_radius = 0.0f;
    float face_rect[4] = {0, 0, 0, 0};
    bool left_detected = false;
    bool right_detected = false;
};

// 문자열에서 "key": <number> 형태의 스칼라 추출. 실패 시 false.
bool extractScalar(const std::string& s, const std::string& key, double& out) {
    std::string needle = "\"" + key + "\"";
    size_t p = s.find(needle);
    if (p == std::string::npos) return false;
    p = s.find(':', p);
    if (p == std::string::npos) return false;
    ++p;
    // 다음 숫자 토큰 파싱
    while (p < s.size() && (s[p] == ' ' || s[p] == '\t' || s[p] == '\n')) ++p;
    size_t start = p;
    while (p < s.size() &&
           (std::isdigit((unsigned char)s[p]) || s[p] == '-' || s[p] == '+' ||
            s[p] == '.' || s[p] == 'e' || s[p] == 'E')) {
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
    std::string needle = "\"" + key + "\"";
    size_t p = s.find(needle);
    if (p == std::string::npos) return false;
    p = s.find(':', p);
    if (p == std::string::npos) return false;
    out = (s.find("true", p) == p + 2 || s.find("true", p + 1) < s.find(',', p));
    // 보다 견고하게: ':' 이후 첫 비공백 토큰이 't'인지 확인
    size_t q = p + 1;
    while (q < s.size() && (s[q] == ' ' || s[q] == '\t')) ++q;
    out = (q < s.size() && s[q] == 't');
    return true;
}

// "face_mesh": [ ... ] 블록에서 478×4 숫자를 순서대로 추출.
bool extractMesh(const std::string& s, std::vector<CppIrisLandmark>& mesh) {
    size_t key = s.find("\"face_mesh\"");
    if (key == std::string::npos) return false;
    size_t lb = s.find('[', key);
    if (lb == std::string::npos) return false;
    // 매칭 ']' 찾기 (중첩 없음 — 객체 배열이지만 '['는 face_mesh 하나뿐)
    size_t rb = s.find(']', lb);
    if (rb == std::string::npos) return false;
    std::string block = s.substr(lb + 1, rb - lb - 1);

    // 숫자 토큰을 순서대로 모은다. 키 문자열("x","y",...)은 무시하고 값만.
    // 포맷: {"x": <n>, "y": <n>, "z": <n>, "visibility": <n>}, ...
    std::vector<double> nums;
    size_t i = 0;
    while (i < block.size()) {
        char c = block[i];
        if (c == '-' || c == '+' || c == '.' || std::isdigit((unsigned char)c)) {
            size_t start = i;
            while (i < block.size() &&
                   (std::isdigit((unsigned char)block[i]) || block[i] == '-' ||
                    block[i] == '+' || block[i] == '.' || block[i] == 'e' ||
                    block[i] == 'E')) {
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
    // face_mesh는 golden_capture가 visibility 없이 (x,y,z) 3토큰만 덤프한다
    // (landmarkToStr include_vis=false). iris 배열만 visibility 포함(4토큰).
    if (nums.size() < 478 * 3) return false;
    mesh.resize(478);
    for (int k = 0; k < 478; ++k) {
        mesh[k].x = static_cast<float>(nums[k * 3 + 0]);
        mesh[k].y = static_cast<float>(nums[k * 3 + 1]);
        mesh[k].z = static_cast<float>(nums[k * 3 + 2]);
        mesh[k].visibility = 1.0f;  // 어댑터도 visibility=1로 채움(JSON엔 없음)
    }
    return true;
}

// left_iris / right_iris 배열(5점) 추출.
bool extractIrisArray(const std::string& s, const std::string& key,
                      std::array<CppIrisLandmark, 5>& out) {
    std::string needle = "\"" + key + "\"";
    size_t k = s.find(needle);
    if (k == std::string::npos) return false;
    size_t lb = s.find('[', k);
    if (lb == std::string::npos) return false;
    size_t rb = s.find(']', lb);
    if (rb == std::string::npos) return false;
    std::string block = s.substr(lb + 1, rb - lb - 1);
    std::vector<double> nums;
    size_t i = 0;
    while (i < block.size()) {
        char c = block[i];
        if (c == '-' || c == '+' || c == '.' || std::isdigit((unsigned char)c)) {
            size_t start = i;
            while (i < block.size() &&
                   (std::isdigit((unsigned char)block[i]) || block[i] == '-' ||
                    block[i] == '+' || block[i] == '.' || block[i] == 'e' ||
                    block[i] == 'E')) {
                ++i;
            }
            nums.push_back(std::stod(block.substr(start, i - start)));
        } else {
            ++i;
        }
    }
    if (nums.size() < 5 * 4) return false;
    for (int j = 0; j < 5; ++j) {
        out[j].x = static_cast<float>(nums[j * 4 + 0]);
        out[j].y = static_cast<float>(nums[j * 4 + 1]);
        out[j].z = static_cast<float>(nums[j * 4 + 2]);
        out[j].visibility = static_cast<float>(nums[j * 4 + 3]);
    }
    return true;
}

bool extractFaceRect(const std::string& s, float out[4]) {
    // face_rect는 객체 {"x":..,"y":..,"width":..,"height":..} 포맷.
    size_t k = s.find("\"face_rect\"");
    if (k == std::string::npos) return false;
    size_t lb = s.find('{', k);
    if (lb == std::string::npos) return false;
    size_t rb = s.find('}', lb);
    if (rb == std::string::npos) return false;
    std::string block = s.substr(lb, rb - lb + 1);
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

GoldenResult loadGolden(const std::string& path) {
    GoldenResult g;
    std::ifstream f(path, std::ios::binary);
    if (!f) return g;
    std::stringstream buf;
    buf << f.rdbuf();
    std::string s = buf.str();

    double w = 0, h = 0, lr = 0, rr = 0;
    if (!extractScalar(s, "input_width", w)) return g;
    if (!extractScalar(s, "input_height", h)) return g;
    g.input_width = static_cast<int>(w);
    g.input_height = static_cast<int>(h);
    if (!extractMesh(s, g.mesh)) return g;
    extractIrisArray(s, "left_iris", g.left_iris);
    extractIrisArray(s, "right_iris", g.right_iris);
    extractScalar(s, "left_radius", lr);
    extractScalar(s, "right_radius", rr);
    g.left_radius = static_cast<float>(lr);
    g.right_radius = static_cast<float>(rr);
    extractFaceRect(s, g.face_rect);
    extractBool(s, "left_detected", g.left_detected);
    extractBool(s, "right_detected", g.right_detected);
    g.ok = true;
    return g;
}

// face_mesh 478점 → 평탄 478×3 (x,y,z) 입력 버퍼.
std::vector<float> meshToFlat(const std::vector<CppIrisLandmark>& mesh) {
    std::vector<float> pts(478 * 3);
    for (int i = 0; i < 478; ++i) {
        pts[i * 3 + 0] = mesh[i].x;
        pts[i * 3 + 1] = mesh[i].y;
        pts[i * 3 + 2] = mesh[i].z;
    }
    return pts;
}

}  // namespace

// ===========================================================================
// 1. 파생 어댑터 동작 불변 (골든 baseline 대조)
// ===========================================================================

class LandmarkAdapterGoldenTest : public ::testing::Test {
protected:
    static std::string baselinePath(const std::string& stem) {
        std::string dir = GOLDEN_BASELINE_DIR;
        if (dir.empty()) {
            // fallback: 소스 기준 상대경로 (CMake 미정의 시).
            dir = "../tests/golden/baseline";
        }
        return dir + "/" + stem + ".result.json";
    }
};

// 골든 baseline의 face_mesh 478점을 어댑터에 넣으면 detector 파생값과 ε 일치해야 한다.
TEST_F(LandmarkAdapterGoldenTest, DerivesIrisGeometryMatchingDetector) {
    // rot0(자연 경로) baseline 사용 — 회전/미러 변형은 좌표 공간 동일성이 검증 목적이 아님.
    const std::string path = baselinePath("face_closeup__gamma05");
    GoldenResult g = loadGolden(path);
    ASSERT_TRUE(g.ok) << "골든 baseline 로드 실패: " << path;
    ASSERT_EQ(g.mesh.size(), 478u);
    ASSERT_GT(g.input_width, 0);
    ASSERT_GT(g.input_height, 0);

    std::vector<float> pts = meshToFlat(g.mesh);
    CppIrisResult r = deriveIrisResult(pts.data(), 478, g.input_width, g.input_height,
                                    /*timestamp_us=*/0);

    // detected: detector도 홍채 5점 [0,1] 범위 판정 — 일치해야 한다.
    EXPECT_EQ(r.left_detected, g.left_detected);
    EXPECT_EQ(r.right_detected, g.right_detected);

    // 홍채 5점(중심+경계) 정규화 좌표: eps_norm.
    for (int i = 0; i < 5; ++i) {
        EXPECT_NEAR(r.left_iris[i].x, g.left_iris[i].x, kEpsNorm) << "left_iris[" << i << "].x";
        EXPECT_NEAR(r.left_iris[i].y, g.left_iris[i].y, kEpsNorm) << "left_iris[" << i << "].y";
        EXPECT_NEAR(r.right_iris[i].x, g.right_iris[i].x, kEpsNorm) << "right_iris[" << i << "].x";
        EXPECT_NEAR(r.right_iris[i].y, g.right_iris[i].y, kEpsNorm) << "right_iris[" << i << "].y";
    }

    // 반경(픽셀): eps_pix.
    EXPECT_NEAR(r.left_radius, g.left_radius, kEpsPix) << "left_radius";
    EXPECT_NEAR(r.right_radius, g.right_radius, kEpsPix) << "right_radius";

    // face_rect(정규화 메시 바운딩): eps_norm.
    EXPECT_NEAR(r.face_rect.x, g.face_rect[0], kEpsNorm) << "face_rect.x";
    EXPECT_NEAR(r.face_rect.y, g.face_rect[1], kEpsNorm) << "face_rect.y";
    EXPECT_NEAR(r.face_rect.width, g.face_rect[2], kEpsNorm) << "face_rect.width";
    EXPECT_NEAR(r.face_rect.height, g.face_rect[3], kEpsNorm) << "face_rect.height";

    // face_mesh 478점 인라인 보존 + valid.
    EXPECT_TRUE(r.face_mesh_valid);
    EXPECT_NEAR(r.face_mesh[468].x, g.mesh[468].x, kEpsNorm);
    EXPECT_NEAR(r.face_mesh[473].y, g.mesh[473].y, kEpsNorm);

    // eyelid_ratio는 현 detector가 0 고정(W3 미구현) — 어댑터 동작 불변.
    EXPECT_FLOAT_EQ(r.eyelid_ratio_left, 0.0f);
    EXPECT_FLOAT_EQ(r.eyelid_ratio_right, 0.0f);
}

// EAR 수식이 temporal_stabilizer computeEAR와 동일 좌표공간(정규화)에서 산출되는지.
TEST_F(LandmarkAdapterGoldenTest, ComputesEarInNormalizedSpace) {
    const std::string path = baselinePath("face_closeup__gamma05");
    GoldenResult g = loadGolden(path);
    ASSERT_TRUE(g.ok);

    float left_ear = iris_sdk::computeEyeAspectRatio(g.mesh.data(), /*left_eye=*/true);
    float right_ear = iris_sdk::computeEyeAspectRatio(g.mesh.data(), /*left_eye=*/false);

    // 정상 검출 프레임에서 EAR은 양수·유한이어야 한다(눈 뜬 상태 ~0.2~0.4 범위).
    EXPECT_TRUE(std::isfinite(left_ear));
    EXPECT_TRUE(std::isfinite(right_ear));
    EXPECT_GT(left_ear, 0.0f);
    EXPECT_GT(right_ear, 0.0f);
}

// W4-B1 핵심 회귀 가드: 주입 결과가 어댑터를 통과하면 검출된 눈의 visibility > 0이어야 한다
// (렌즈 렌더 가능). confidence=0을 두던 시절 어댑터(eye_render_packet_adapter.cpp:89)
// visibility = 0 * (1 - eyelid) = 0 → 렌즈 미렌더였다(주입 경로 확정 결함). 방향 A로
// deriveIrisResult가 detected 시 confidence=1.0(게이트 통과 상수)을 두어 게이트를 연다.
TEST_F(LandmarkAdapterGoldenTest, InjectedDetectedEyeYieldsPositiveVisibility) {
    const std::string path = baselinePath("face_closeup__gamma05");
    GoldenResult g = loadGolden(path);
    ASSERT_TRUE(g.ok) << "골든 baseline 로드 실패: " << path;
    ASSERT_EQ(g.mesh.size(), 478u);

    std::vector<float> pts = meshToFlat(g.mesh);
    CppIrisResult r = deriveIrisResult(pts.data(), 478, g.input_width, g.input_height,
                                       /*timestamp_us=*/0);

    // 방향 A 직접 검증: 검출 시 confidence는 게이트 통과 상수 1.0(측정값 아님).
    ASSERT_TRUE(r.detected) << "baseline 프레임은 최소 한쪽 눈이 검출돼야 한다";
    EXPECT_FLOAT_EQ(r.confidence, 1.0f);

    // 주입 경로 eyelid_ratio=0(W3 미구현) → visibility = 1.0 * (1 - 0) = 1.0.
    // 어댑터는 side별 detected로 게이팅하므로 검출된 눈만 검사한다.
    if (r.left_detected) {
        const auto p = iris_sdk::gpu::adaptIrisResult(
            r, iris_sdk::gpu::EyeSide::Left, r.frame_width, r.frame_height);
        EXPECT_GT(p.visibility, 0.0f) << "주입 좌안 visibility=0 → 렌즈 미렌더 회귀";
        EXPECT_FLOAT_EQ(p.visibility, 1.0f)
            << "주입 eyelid=0 + confidence=1.0 → visibility=1.0 계약";
    }
    if (r.right_detected) {
        const auto p = iris_sdk::gpu::adaptIrisResult(
            r, iris_sdk::gpu::EyeSide::Right, r.frame_width, r.frame_height);
        EXPECT_GT(p.visibility, 0.0f) << "주입 우안 visibility=0 → 렌즈 미렌더 회귀";
        EXPECT_FLOAT_EQ(p.visibility, 1.0f)
            << "주입 eyelid=0 + confidence=1.0 → visibility=1.0 계약";
    }
}

// ===========================================================================
// 2. 입력 유효성 (ADR §6.1) — deriveIrisResult 방어 + 어댑터 직접
// ===========================================================================

TEST(LandmarkAdapterValidityTest, RejectsOutOfRangeIrisAsUndetected) {
    // 홍채 한 점이 [0,1] 밖이면 그쪽 detected=false (extractIris 동작 불변).
    std::vector<float> pts(478 * 3, 0.5f);
    // ④ §7.3 canonical: 인덱스 469는 468그룹(=kRightIris=피험자 우안)의 경계점.
    //   469 y를 1.5로 → right(피험자 우안) 미검출. left(473그룹)은 0.5로 유효.
    pts[469 * 3 + 1] = 1.5f;
    CppIrisResult r = deriveIrisResult(pts.data(), 478, 640, 480, 0);
    EXPECT_FALSE(r.right_detected);
    EXPECT_TRUE(r.left_detected);  // 좌안(473그룹)은 0.5로 모두 유효
}

// W4-B1 대칭 가드: confidence=1.0 변경이 "미검출 눈까지 게이트를 여는" 과잉을 내지 않는다.
// 미검출 눈은 어댑터(eye_render_packet_adapter.cpp:43) side별 early-return으로 visibility=0,
// 검출 눈은 visibility>0. presence 게이트가 confidence와 독립임을 고정한다.
TEST(LandmarkAdapterValidityTest, UndetectedEyeYieldsZeroVisibility) {
    std::vector<float> pts(478 * 3, 0.5f);
    // ④ §7.3 canonical: 469는 468그룹(=kRightIris=피험자 우안) 경계점 → right 미검출.
    pts[469 * 3 + 1] = 1.5f;
    CppIrisResult r = deriveIrisResult(pts.data(), 478, 640, 480, 0);
    ASSERT_FALSE(r.right_detected);
    ASSERT_TRUE(r.left_detected);
    EXPECT_FLOAT_EQ(r.confidence, 1.0f);  // 전체 detected(좌안)=true → 게이트 통과 상수

    const auto pr = iris_sdk::gpu::adaptIrisResult(
        r, iris_sdk::gpu::EyeSide::Right, 640, 480);
    EXPECT_FLOAT_EQ(pr.visibility, 0.0f) << "미검출 우안은 어댑터 early-return으로 차단";

    const auto pl = iris_sdk::gpu::adaptIrisResult(
        r, iris_sdk::gpu::EyeSide::Left, 640, 480);
    EXPECT_GT(pl.visibility, 0.0f) << "검출 좌안은 visibility>0";
}

// W4-B1: 양쪽 모두 미검출이면 confidence=0(게이트 통과 상수 미부여). 방향 A의
// `detected ? 1.0f : 0.0f` 삼항 분기를 양 끝에서 고정한다.
TEST(LandmarkAdapterValidityTest, AllUndetectedYieldsZeroConfidence) {
    std::vector<float> pts(478 * 3, 0.5f);
    // 좌·우 홍채 경계점을 모두 [0,1] 밖으로 → 양안 미검출.
    pts[iris_sdk::landmark_indices::kLeftIris[1] * 3 + 1] = 1.5f;
    pts[iris_sdk::landmark_indices::kRightIris[1] * 3 + 1] = 1.5f;
    CppIrisResult r = deriveIrisResult(pts.data(), 478, 640, 480, 0);
    ASSERT_FALSE(r.left_detected);
    ASSERT_FALSE(r.right_detected);
    EXPECT_FALSE(r.detected);
    EXPECT_FLOAT_EQ(r.confidence, 0.0f);
}

// ===========================================================================
// 3. seqlock 더블버퍼 store — write/readDerived + generation + 거부
// ===========================================================================

TEST(LandmarkInjectionStoreTest, EmptyStoreReadFails) {
    LandmarkInjectionStore store;
    EXPECT_EQ(store.generation(), 0u);
    CppIrisResult r;
    EXPECT_FALSE(store.readDerived(&r));
}

TEST(LandmarkInjectionStoreTest, WriteThenReadRoundTrip) {
    LandmarkInjectionStore store;
    std::vector<float> pts(478 * 3, 0.5f);
    // 좌/우 홍채 5점을 의미있게 채움(중심 + 경계 — 반경 계산 가능하게).
    // ④ §7.3 canonical: 인덱스 468그룹(468~472)은 right_iris(피험자 우안)로 라우팅된다.
    //   center 468 = (0.4,0.5), 경계 4점은 ±0.02 사방. (473그룹은 0.5로 left도 검출됨)
    auto setPt = [&](int idx, float x, float y) {
        pts[idx * 3 + 0] = x; pts[idx * 3 + 1] = y; pts[idx * 3 + 2] = 0.0f;
    };
    setPt(468, 0.40f, 0.50f);
    setPt(469, 0.42f, 0.50f);
    setPt(470, 0.40f, 0.48f);
    setPt(471, 0.38f, 0.50f);
    setPt(472, 0.40f, 0.52f);

    uint32_t gen = 0;
    ASSERT_TRUE(store.write(pts.data(), 478, 640, 480, 1000, &gen));
    EXPECT_EQ(gen, 2u);                 // 첫 write → 세대 2 (짝수)
    EXPECT_EQ(store.generation(), 2u);

    CppIrisResult r;
    ASSERT_TRUE(store.readDerived(&r));
    EXPECT_TRUE(r.right_detected);      // 468그룹 → right_iris(피험자 우안) 검출
    EXPECT_EQ(r.frame_width, 640);
    EXPECT_EQ(r.frame_height, 480);
    EXPECT_EQ(r.timestamp_ms, 1);       // 1000µs → 1ms
    EXPECT_NEAR(r.right_iris[0].x, 0.40f, 1e-6f);  // 인덱스 468 center → right_iris[0]
    // 반경: 경계 4점이 모두 중심에서 0.02 정규화 = 0.02*640=12.8px (x), 0.02*480=9.6px (y).
    EXPECT_GT(r.right_radius, 0.0f);
}

TEST(LandmarkInjectionStoreTest, GenerationIncrementsBy2PerWrite) {
    LandmarkInjectionStore store;
    std::vector<float> pts(478 * 3, 0.5f);
    uint32_t g1 = 0, g2 = 0, g3 = 0;
    ASSERT_TRUE(store.write(pts.data(), 478, 100, 100, 0, &g1));
    ASSERT_TRUE(store.write(pts.data(), 478, 100, 100, 0, &g2));
    ASSERT_TRUE(store.write(pts.data(), 478, 100, 100, 0, &g3));
    EXPECT_EQ(g1, 2u);
    EXPECT_EQ(g2, 4u);
    EXPECT_EQ(g3, 6u);
}

TEST(LandmarkInjectionStoreTest, RejectsWrongPointCountKeepsStale) {
    LandmarkInjectionStore store;
    std::vector<float> pts(478 * 3, 0.5f);
    uint32_t gen = 0;
    ASSERT_TRUE(store.write(pts.data(), 478, 100, 100, 0, &gen));  // 유효 주입
    EXPECT_EQ(store.generation(), 2u);

    // 477점 거부 — generation 불변(스테일 유지).
    uint32_t bad = 999;
    EXPECT_FALSE(store.write(pts.data(), 477, 100, 100, 0, &bad));
    EXPECT_EQ(store.generation(), 2u);   // 증가하지 않음
    EXPECT_EQ(bad, 999u);                // out_generation 미변경

    // 직전 유효 세대를 여전히 읽을 수 있어야 한다.
    CppIrisResult r;
    EXPECT_TRUE(store.readDerived(&r));
}

TEST(LandmarkInjectionStoreTest, RejectsNullKeepsStale) {
    LandmarkInjectionStore store;
    std::vector<float> pts(478 * 3, 0.5f);
    uint32_t gen = 0;
    ASSERT_TRUE(store.write(pts.data(), 478, 100, 100, 0, &gen));

    EXPECT_FALSE(store.write(nullptr, 478, 100, 100, 0, &gen));
    EXPECT_EQ(store.generation(), 2u);
}

TEST(LandmarkInjectionStoreTest, RejectsNonPositiveDimsKeepsStale) {
    LandmarkInjectionStore store;
    std::vector<float> pts(478 * 3, 0.5f);
    uint32_t gen = 0;
    ASSERT_TRUE(store.write(pts.data(), 478, 100, 100, 0, &gen));

    EXPECT_FALSE(store.write(pts.data(), 478, 0, 100, 0, &gen));    // width=0
    EXPECT_FALSE(store.write(pts.data(), 478, 100, -1, 0, &gen));   // height<0
    EXPECT_EQ(store.generation(), 2u);
}

TEST(LandmarkInjectionStoreTest, RejectsNaNInfKeepsStale) {
    LandmarkInjectionStore store;
    std::vector<float> pts(478 * 3, 0.5f);
    uint32_t gen = 0;
    ASSERT_TRUE(store.write(pts.data(), 478, 100, 100, 0, &gen));

    std::vector<float> nan_pts = pts;
    nan_pts[100] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(store.write(nan_pts.data(), 478, 100, 100, 0, &gen));

    std::vector<float> inf_pts = pts;
    inf_pts[200] = std::numeric_limits<float>::infinity();
    EXPECT_FALSE(store.write(inf_pts.data(), 478, 100, 100, 0, &gen));

    EXPECT_EQ(store.generation(), 2u);  // 두 거부 모두 generation 불변
}

// 동시성 스모크: writer 1 + reader 1이 경합해도 reader가 torn 좌표를 보지 않는다.
// (seqlock 일관성 — 읽은 스냅샷의 모든 점이 같은 frame_width로 결속되어야 함)
//
// 플레이크 방지: reader는 (a) 첫 유효 세대(generation != 0)를 본 뒤 카운트를 시작하고,
// (b) 고정 횟수 스핀이 아니라 '유효 read 목표 달성 또는 시간 예산 소진'까지 돈다.
// 따라서 writer 기동이 늦어 reader가 generation==0만 보는 시작 레이스로 reads==0이 되어
// 비결정적으로 실패하던 문제가 제거된다(--gtest_repeat 부하 반복에서도 안정).
TEST(LandmarkInjectionStoreTest, ConcurrentReadNeverTearsSnapshot) {
    LandmarkInjectionStore store;
    std::atomic<bool> stop{false};
    std::atomic<int> torn{0};
    std::atomic<int> reads{0};

    // 시작 동기화: 메인 스레드에서 최초 1회 write를 동기 수행 → reader가 즉시 유효 세대를 본다.
    {
        std::vector<float> seed(478 * 3, 0.5f);
        seed[0] = 0.1f;  // fw=100 ↔ pts[0]=0.1 결속 규약 (짝수 n)
        uint32_t g0 = 0;
        ASSERT_TRUE(store.write(seed.data(), 478, 100, 480, 0, &g0));
        ASSERT_EQ(g0, 2u);
    }

    // writer: frame_width를 세대마다 토글(100/200 교차)해서 결속 검증. n=1부터(seed가 n=0).
    std::thread writer([&] {
        std::vector<float> pts(478 * 3, 0.5f);
        int n = 1;
        while (!stop.load(std::memory_order_relaxed)) {
            int fw = (n & 1) ? 200 : 100;
            // pts[0]에도 같은 신호(0.1 또는 0.2)를 심어 결속 검증.
            pts[0] = (n & 1) ? 0.2f : 0.1f;
            uint32_t g = 0;
            store.write(pts.data(), 478, fw, 480, n, &g);
            ++n;
        }
    });

    // reader: frame_width와 face_mesh[0].x 신호가 결속돼 있는지 확인.
    // 유효 read 목표(kTargetReads) 달성 또는 시간 예산(kBudget) 소진까지 돈다.
    std::thread reader([&] {
        constexpr int kTargetReads = 20000;
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (reads.load(std::memory_order_relaxed) < kTargetReads) {
            if (std::chrono::steady_clock::now() >= deadline) {
                break;  // 시간 예산 소진(과부하 환경 안전장치) — seed로 reads>0은 보장됨.
            }
            CppIrisResult r;
            if (store.readDerived(&r)) {
                reads.fetch_add(1, std::memory_order_relaxed);
                bool expect200 = (r.frame_width == 200);
                bool expect100 = (r.frame_width == 100);
                // fw=200 ↔ pts[0]=0.2, fw=100 ↔ pts[0]=0.1 결속.
                if (expect200 && std::abs(r.face_mesh[0].x - 0.2f) > 1e-4f) {
                    torn.fetch_add(1, std::memory_order_relaxed);
                }
                if (expect100 && std::abs(r.face_mesh[0].x - 0.1f) > 1e-4f) {
                    torn.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    });

    reader.join();
    stop.store(true, std::memory_order_relaxed);
    writer.join();

    EXPECT_EQ(torn.load(), 0) << "seqlock이 torn snapshot을 허용했다";
    EXPECT_GT(reads.load(), 0) << "reader가 한 번도 유효 세대를 못 읽음";
}

// 이중 writer 회귀 테스트 (critical 이슈 봉쇄 검증):
//   두 writer 스레드가 동일 store에 경합 write + reader 1이 검증한다. writer_mutex_로
//   직렬화되므로 (a) reader는 torn snapshot을 보지 않고, (b) 모든 write 종료 후 generation은
//   짝수(완결)로 안착해야 한다(홀수 영구 잔류 = seqlock 붕괴 신호). 직렬화 부재 시 실패한다.
TEST(LandmarkInjectionStoreTest, DualWriterSerializedNoTearEvenGeneration) {
    LandmarkInjectionStore store;
    std::atomic<bool> stop{false};
    std::atomic<int> torn{0};
    std::atomic<int> reads{0};

    // 결속 규약: writer A는 항상 (fw=100, pts[0]=0.1), writer B는 항상 (fw=200, pts[0]=0.2).
    // 어느 writer가 최신을 공개하든 reader가 본 스냅샷은 (fw,pts[0])가 같은 쌍이어야 한다.
    auto make_writer = [&](int fw, float sig) {
        return std::thread([&, fw, sig] {
            std::vector<float> pts(478 * 3, 0.5f);
            pts[0] = sig;
            for (int i = 0; i < 50000 && !stop.load(std::memory_order_relaxed); ++i) {
                uint32_t g = 0;
                store.write(pts.data(), 478, fw, 480, i, &g);
            }
        });
    };

    // 메인에서 1회 seed → reader가 즉시 유효 세대를 본다(시작 레이스 제거).
    {
        std::vector<float> seed(478 * 3, 0.5f);
        seed[0] = 0.1f;
        uint32_t g0 = 0;
        ASSERT_TRUE(store.write(seed.data(), 478, 100, 480, 0, &g0));
    }

    std::thread writer_a = make_writer(100, 0.1f);
    std::thread writer_b = make_writer(200, 0.2f);

    std::thread reader([&] {
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (reads.load(std::memory_order_relaxed) < 50000) {
            if (std::chrono::steady_clock::now() >= deadline) break;
            CppIrisResult r;
            if (store.readDerived(&r)) {
                reads.fetch_add(1, std::memory_order_relaxed);
                // fw=100 ↔ pts[0]=0.1, fw=200 ↔ pts[0]=0.2 결속(둘 중 하나여야 함).
                bool ok100 = (r.frame_width == 100) &&
                             std::abs(r.face_mesh[0].x - 0.1f) <= 1e-4f;
                bool ok200 = (r.frame_width == 200) &&
                             std::abs(r.face_mesh[0].x - 0.2f) <= 1e-4f;
                if (!ok100 && !ok200) {
                    torn.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    });

    reader.join();
    stop.store(true, std::memory_order_relaxed);
    writer_a.join();
    writer_b.join();

    EXPECT_EQ(torn.load(), 0)
        << "이중 writer 경합에서 seqlock이 torn snapshot을 허용했다(직렬화 실패)";
    EXPECT_GT(reads.load(), 0) << "reader가 한 번도 유효 세대를 못 읽음";
    // 모든 write 종료 후 generation은 짝수(완결)여야 한다 — 홀수 잔류 = torn-write 붕괴.
    const uint32_t final_gen = store.generation();
    EXPECT_EQ(final_gen & 1u, 0u)
        << "write 종료 후 generation이 홀수(쓰기 중 마커)로 잔류: " << final_gen;
    EXPECT_NE(final_gen, 0u) << "최소 seed write 한 번은 완결됐어야 한다";
    // 정지 후에는 reader가 항상 유효 세대를 읽을 수 있어야 한다(영구 false 아님).
    CppIrisResult after;
    EXPECT_TRUE(store.readDerived(&after))
        << "정지 후 readDerived 실패 — generation 홀수 잔류로 영구 재시도 의심";
}

// ===========================================================================
// 4. C API 표면 경유 왕복 E2E (③-3 §3-3 — sdk_api.h 공개 경계)
// ===========================================================================
//
// ②가 store/deriveIrisResult를 직접 호출하는 단위 검증이라면, ④는 공개 C 경계를
// 통과한다: iris_set_landmarks(주입) → iris_get_injected_result(파생 조회). C API는
// 프로세스 전역 g_landmark_store를 공유하므로, 각 케이스는 자기 주입을 직전에 수행해
// 결정적 상태(last-writer-wins, generation 단조 증가)를 보장한다.
//
// 전역 C IrisResult(::IrisResult)와 C++ iris_sdk::IrisResult(테스트 namespace의 별칭
// CppIrisResult)는 별개 타입이다 — C API 호출에는 전역 타입(::IrisResult)을 명시한다.

class LandmarkCApiRoundTripTest : public ::testing::Test {
protected:
    static std::string baselinePath(const std::string& stem) {
        std::string dir = GOLDEN_BASELINE_DIR;
        if (dir.empty()) {
            dir = "../tests/golden/baseline";
        }
        return dir + "/" + stem + ".result.json";
    }
};

// 골든 baseline 478점을 C API로 주입→조회하면 detector 파생값과 ε 일치해야 한다.
TEST_F(LandmarkCApiRoundTripTest, InjectThenGetMatchesDetector) {
    const std::string path = baselinePath("face_closeup__gamma05");
    GoldenResult g = loadGolden(path);
    ASSERT_TRUE(g.ok) << "골든 baseline 로드 실패: " << path;
    ASSERT_EQ(g.mesh.size(), 478u);
    ASSERT_GT(g.input_width, 0);
    ASSERT_GT(g.input_height, 0);

    std::vector<float> pts = meshToFlat(g.mesh);

    // 주입: 공개 C 경계. generation은 전역이라 절대값이 아닌 '증가'만 검증한다.
    const uint32_t gen_before = iris_get_landmark_generation();
    uint32_t gen_after = 0;
    ASSERT_EQ(iris_set_landmarks(pts.data(), 478, g.input_width, g.input_height,
                                 /*timestamp_us=*/1000, &gen_after),
              IRIS_SDK_OK);
    EXPECT_GT(gen_after, gen_before) << "주입 후 generation이 증가하지 않음";
    EXPECT_EQ(gen_after & 1u, 0u) << "공개 generation은 완결(짝수)이어야 함";
    EXPECT_EQ(iris_get_landmark_generation(), gen_after);

    // 조회: 공개 C 경계 (C IrisResult).
    ::IrisResult c_out{};
    ASSERT_EQ(iris_get_injected_result(&c_out), IRIS_SDK_OK);

    // detected: detector도 홍채 5점 [0,1] 판정 — baseline과 일치.
    EXPECT_EQ(c_out.left_detected, g.left_detected);
    EXPECT_EQ(c_out.right_detected, g.right_detected);

    // 홍채 5점 정규화 좌표: eps_norm.
    for (int i = 0; i < 5; ++i) {
        EXPECT_NEAR(c_out.left_iris[i].x, g.left_iris[i].x, kEpsNorm) << "left_iris[" << i << "].x";
        EXPECT_NEAR(c_out.left_iris[i].y, g.left_iris[i].y, kEpsNorm) << "left_iris[" << i << "].y";
        EXPECT_NEAR(c_out.right_iris[i].x, g.right_iris[i].x, kEpsNorm) << "right_iris[" << i << "].x";
        EXPECT_NEAR(c_out.right_iris[i].y, g.right_iris[i].y, kEpsNorm) << "right_iris[" << i << "].y";
    }

    // 반경(픽셀): eps_pix.
    EXPECT_NEAR(c_out.left_radius, g.left_radius, kEpsPix) << "left_radius";
    EXPECT_NEAR(c_out.right_radius, g.right_radius, kEpsPix) << "right_radius";

    // face_rect(정규화): eps_norm.
    EXPECT_NEAR(c_out.face_rect.x, g.face_rect[0], kEpsNorm) << "face_rect.x";
    EXPECT_NEAR(c_out.face_rect.y, g.face_rect[1], kEpsNorm) << "face_rect.y";
    EXPECT_NEAR(c_out.face_rect.width, g.face_rect[2], kEpsNorm) << "face_rect.width";
    EXPECT_NEAR(c_out.face_rect.height, g.face_rect[3], kEpsNorm) << "face_rect.height";

    // 주입 부가 필드: frame dims/timestamp가 한 세대에 결속돼 왕복 보존.
    EXPECT_EQ(c_out.frame_width, g.input_width);
    EXPECT_EQ(c_out.frame_height, g.input_height);
    EXPECT_EQ(c_out.timestamp_ms, 1);  // 1000µs → 1ms

    // face_mesh 478점 인라인 보존 + valid (조회 경로가 메시를 운반).
    EXPECT_TRUE(c_out.face_mesh_valid);
    EXPECT_NEAR(c_out.face_mesh[468].x, g.mesh[468].x, kEpsNorm);
    EXPECT_NEAR(c_out.face_mesh[473].y, g.mesh[473].y, kEpsNorm);
}

// C API 표면이 deriveIrisResult 직접 호출과 동일 결과를 낸다(경계가 어댑터를 왜곡하지 않음).
TEST_F(LandmarkCApiRoundTripTest, CApiResultEqualsDirectAdapter) {
    const std::string path = baselinePath("face_closeup__gamma05");
    GoldenResult g = loadGolden(path);
    ASSERT_TRUE(g.ok);
    std::vector<float> pts = meshToFlat(g.mesh);

    // 직접 어댑터 (C++ 타입).
    CppIrisResult direct = deriveIrisResult(pts.data(), 478, g.input_width, g.input_height,
                                         /*timestamp_us=*/2000);

    // C API 왕복 (C 타입).
    uint32_t gen = 0;
    ASSERT_EQ(iris_set_landmarks(pts.data(), 478, g.input_width, g.input_height, 2000, &gen),
              IRIS_SDK_OK);
    ::IrisResult capi{};
    ASSERT_EQ(iris_get_injected_result(&capi), IRIS_SDK_OK);

    // 두 경로의 파생값이 정확히 일치해야 한다(부동소수 동일 입력·동일 수식 — bit-exact 기대).
    EXPECT_FLOAT_EQ(capi.left_iris[0].x, direct.left_iris[0].x);
    EXPECT_FLOAT_EQ(capi.left_iris[0].y, direct.left_iris[0].y);
    EXPECT_FLOAT_EQ(capi.left_radius, direct.left_radius);
    EXPECT_FLOAT_EQ(capi.right_radius, direct.right_radius);
    EXPECT_FLOAT_EQ(capi.face_rect.x, direct.face_rect.x);
    EXPECT_FLOAT_EQ(capi.face_rect.width, direct.face_rect.width);
    EXPECT_EQ(capi.left_detected, direct.left_detected);
    EXPECT_EQ(capi.right_detected, direct.right_detected);
    EXPECT_EQ(capi.frame_width, direct.frame_width);
    EXPECT_EQ(capi.frame_height, direct.frame_height);
}

// C API 가드: NULL out은 IRIS_SDK_NULL_POINTER (전역 상태 무관 — 결정적).
TEST_F(LandmarkCApiRoundTripTest, GetInjectedResultRejectsNull) {
    EXPECT_EQ(iris_get_injected_result(nullptr), IRIS_SDK_NULL_POINTER);
}

// C API 주입 거부도 공개 경계에서 동작 불변 — 478 외/NULL/치수≤0.
TEST_F(LandmarkCApiRoundTripTest, SetLandmarksRejectsInvalidInputs) {
    std::vector<float> pts(478 * 3, 0.5f);
    uint32_t gen = 0;
    // 478 외 점 수.
    EXPECT_EQ(iris_set_landmarks(pts.data(), 477, 640, 480, 0, &gen), IRIS_SDK_INVALID_PARAM);
    // NULL pts.
    EXPECT_EQ(iris_set_landmarks(nullptr, 478, 640, 480, 0, &gen), IRIS_SDK_NULL_POINTER);
    // NULL out_generation.
    EXPECT_EQ(iris_set_landmarks(pts.data(), 478, 640, 480, 0, nullptr), IRIS_SDK_NULL_POINTER);
    // 치수 ≤ 0.
    EXPECT_EQ(iris_set_landmarks(pts.data(), 478, 0, 480, 0, &gen), IRIS_SDK_INVALID_PARAM);
}
