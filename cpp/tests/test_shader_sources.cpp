/**
 * @file test_shader_sources.cpp
 * @brief 셰이더 소스 문자열 정합성 가드 (GL 컨텍스트 불필요 — 순수 문자열 검사)
 *
 * 배경: raw string 여는 구분자(R"glsl() 뒤 개행 때문에 `#version`이 2번째 줄로 밀리면
 * Mali 드라이버가 컴파일을 거부한다(P0005: "#version must be on the first line").
 * Adreno는 통과시켜서 2026-07-22 갤럭시탭 S10 Ultra(Immortalis-G720, r44p1)에서
 * 전 셰이더 컴파일 실패 → 검은 화면으로 처음 드러났다. 벤더 편차라 실기기 1대로는
 * 못 잡으므로 소스 레벨에서 고정한다.
 */

#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

#include "iris_sdk/gpu/shader_manager.h"

namespace {

struct ShaderEntry {
    const char* name;
    const char* source;
};

std::vector<ShaderEntry> allShaders() {
    using namespace iris_sdk::shaders;
    return {
        {"FULLSCREEN_QUAD_VERTEX", FULLSCREEN_QUAD_VERTEX},
        {"PASSTHROUGH_FRAGMENT", PASSTHROUGH_FRAGMENT},
        {"COMBINED_COLOR_ADJUSTMENT_FRAGMENT", COMBINED_COLOR_ADJUSTMENT_FRAGMENT},
        {"SKIN_MASK_FILL_VERTEX", SKIN_MASK_FILL_VERTEX},
        {"SKIN_MASK_FILL_FRAGMENT", SKIN_MASK_FILL_FRAGMENT},
        {"SKIN_SEPARABLE_BLUR_FRAGMENT", SKIN_SEPARABLE_BLUR_FRAGMENT},
        {"SKIN_SMOOTH_COMPOSITE_FRAGMENT", SKIN_SMOOTH_COMPOSITE_FRAGMENT},
        {"WARP_FRAGMENT", WARP_FRAGMENT},
        {"LENS_OVERLAY_VERTEX", LENS_OVERLAY_VERTEX},
        {"LENS_OVERLAY_FRAGMENT", LENS_OVERLAY_FRAGMENT},
    };
}

}  // namespace

/// #version은 반드시 오프셋 0 — 선행 개행·공백·주석 모두 금지 (Mali 드라이버 요구).
TEST(ShaderSources, VersionDirectiveIsFirstByte) {
    for (const auto& s : allShaders()) {
        ASSERT_NE(s.source, nullptr) << s.name << ": null 소스";
        EXPECT_EQ(std::strncmp(s.source, "#version", 8), 0)
            << s.name << ": `#version`이 첫 바이트가 아님 — raw string 여는 구분자 뒤 "
            << "개행을 제거하라(R\"glsl(#version ...). Mali에서 컴파일 실패한다. "
            << "실제 첫 24바이트: \"" << std::string(s.source, 24) << "\"";
    }
}

/// 버전 선언 자체가 기대 프로파일인지 (ES 3.1 — 컴퓨트/이미지 로드스토어 사용 전제).
TEST(ShaderSources, VersionIsEs310) {
    for (const auto& s : allShaders()) {
        ASSERT_NE(s.source, nullptr) << s.name;
        const std::string head(s.source, std::strlen("#version 310 es"));
        EXPECT_EQ(head, "#version 310 es") << s.name << ": 버전 선언 불일치";
    }
}

/// #version 줄은 개행으로 끝나야 한다 (다음 지시문과 붙으면 파싱 실패).
TEST(ShaderSources, VersionLineTerminated) {
    for (const auto& s : allShaders()) {
        ASSERT_NE(s.source, nullptr) << s.name;
        const std::string src(s.source);
        const auto nl = src.find('\n');
        ASSERT_NE(nl, std::string::npos) << s.name << ": 개행 없음";
        EXPECT_EQ(src.substr(0, nl), "#version 310 es")
            << s.name << ": 첫 줄에 버전 외 토큰이 섞임";
    }
}
