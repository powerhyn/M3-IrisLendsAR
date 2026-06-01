/**
 * @file lens_sku_metadata.cpp
 * @brief P6-W7: SKU 메타데이터 경량 JSON 파서 + B4 baked 림발 자동감지.
 *
 * JSON 파서는 의존성 0 — 고정 스키마([{ string/bool 키들 }, ...]) 전용.
 * 범용 파서가 아니다(숫자/중첩/배열 값 미지원, 만나면 파싱 실패).
 */

#include "iris_sdk/lens_sku_metadata.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <utility>
#include <vector>

namespace iris_sdk {

// ============================================================================
// 경량 JSON 파서 (이 스키마 전용)
// ============================================================================
namespace {

// 단일 패스 커서. 모든 헬퍼는 실패 시 false 반환 (예외 없음).
struct JsonCursor {
    const char* p;
    const char* end;

    bool eof() const { return p >= end; }

    void skipWs() {
        while (p < end) {
            const char c = *p;
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++p;
            } else {
                break;
            }
        }
    }

    // 현재 위치가 expected면 소비하고 true.
    bool consume(char expected) {
        skipWs();
        if (!eof() && *p == expected) {
            ++p;
            return true;
        }
        return false;
    }

    char peek() {
        skipWs();
        return eof() ? '\0' : *p;
    }
};

// "..." 문자열 파싱. 여는 따옴표는 이미 소비된 상태로 진입.
// 이스케이프는 \" 와 \\ 만 처리(통제된 git 체크인 파일 전제). UTF-8 바이트 그대로.
bool parseStringBody(JsonCursor& c, std::string& out) {
    out.clear();
    while (!c.eof()) {
        const char ch = *c.p++;
        if (ch == '"') {
            return true;  // 닫는 따옴표
        }
        if (ch == '\\') {
            if (c.eof()) return false;
            const char esc = *c.p++;
            if (esc == '"' || esc == '\\') {
                out.push_back(esc);
            } else {
                // 스키마에 없는 이스케이프 → 파싱 실패(robust하게 거부).
                return false;
            }
        } else {
            out.push_back(ch);
        }
    }
    return false;  // 닫는 따옴표 없이 EOF
}

// 여는 따옴표 포함 문자열 토큰.
bool parseString(JsonCursor& c, std::string& out) {
    if (!c.consume('"')) return false;
    return parseStringBody(c, out);
}

// true / false 리터럴.
bool parseBool(JsonCursor& c, bool& out) {
    c.skipWs();
    if (c.end - c.p >= 4 && std::strncmp(c.p, "true", 4) == 0) {
        c.p += 4;
        out = true;
        return true;
    }
    if (c.end - c.p >= 5 && std::strncmp(c.p, "false", 5) == 0) {
        c.p += 5;
        out = false;
        return true;
    }
    return false;
}

// 단일 SKU 객체 `{ "key": value, ... }` 파싱. 여는 '{'는 호출자가 확인.
bool parseObject(JsonCursor& c, LensSkuMetadata& out) {
    if (!c.consume('{')) return false;

    // 빈 객체 허용.
    if (c.peek() == '}') {
        c.consume('}');
        return true;
    }

    while (true) {
        std::string key;
        if (!parseString(c, key)) return false;
        if (!c.consume(':')) return false;

        // 값 타입은 키로 결정(고정 스키마). 알 수 없는 키는 거부.
        if (key == "sku_id") {
            if (!parseString(c, out.sku_id)) return false;
        } else if (key == "display_name") {
            if (!parseString(c, out.display_name)) return false;
        } else if (key == "has_baked_limbal") {
            if (!parseBool(c, out.has_baked_limbal)) return false;
        } else if (key == "prefers_crl") {
            if (!parseBool(c, out.prefers_crl)) return false;
        } else if (key == "prefers_graphic_outline") {
            if (!parseBool(c, out.prefers_graphic_outline)) return false;
        } else {
            return false;  // 스키마 외 키
        }

        if (c.consume(',')) {
            continue;
        }
        if (c.consume('}')) {
            return true;
        }
        return false;  // ',' 도 '}' 도 아님
    }
}

// 최상위 객체 배열 `[ {...}, {...} ]`.
bool parseArray(JsonCursor& c, std::vector<LensSkuMetadata>& out) {
    if (!c.consume('[')) return false;

    if (c.peek() == ']') {
        c.consume(']');
        return true;  // 빈 배열
    }

    while (true) {
        LensSkuMetadata item;
        if (!parseObject(c, item)) return false;
        out.push_back(std::move(item));

        if (c.consume(',')) {
            continue;
        }
        if (c.consume(']')) {
            return true;
        }
        return false;
    }
}

}  // namespace

// ============================================================================
// LensSkuRegistry
// ============================================================================

bool LensSkuRegistry::loadFromJson(const std::string& json_text) {
    JsonCursor c{json_text.data(), json_text.data() + json_text.size()};

    std::vector<LensSkuMetadata> items;
    if (!parseArray(c, items)) {
        return false;  // 부분 파싱 결과는 버린다.
    }

    // 배열 뒤 트레일링 토큰(공백 제외)은 깨진 JSON으로 간주.
    c.skipWs();
    if (!c.eof()) {
        return false;
    }

    by_sku_id_.clear();
    by_sku_id_.reserve(items.size());
    for (auto& item : items) {
        // sku_id가 비면 키로 쓸 수 없으므로 무시(나머지는 살린다).
        if (item.sku_id.empty()) {
            continue;
        }
        by_sku_id_.emplace(item.sku_id, std::move(item));
    }
    return true;
}

bool LensSkuRegistry::loadFromFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;  // 파일 없음 → throw 금지.
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return loadFromJson(ss.str());
}

std::optional<LensSkuMetadata> LensSkuRegistry::find(const std::string& sku_id) const {
    auto it = by_sku_id_.find(sku_id);
    if (it == by_sku_id_.end()) {
        return std::nullopt;
    }
    return it->second;
}

// ============================================================================
// B4 baked 림발 자동감지 (순수 함수)
// ============================================================================
namespace {

// edge/center band 평균 luma 측정 결과.
struct LimbalBands {
    double edge_lum = 0.0;
    double center_lum = 0.0;
    bool valid = false;  // edge/center 모두 유효 픽셀 1개 이상
};

LimbalBands measureBands(const uint8_t* rgba, int width, int height,
                         float roi_inner, float center_inner, float center_outer) {
    LimbalBands result;
    if (rgba == nullptr || width <= 0 || height <= 0) {
        return result;
    }

    // 짧은 변 절반을 r=1.0로 정규화. (=0 방지: width/height>0 보장됨)
    const double radius_px = std::min(width, height) * 0.5;
    if (radius_px <= 0.0) {
        return result;
    }
    const double inv_radius_sq = 1.0 / (radius_px * radius_px);
    const double cx = width * 0.5;
    const double cy = height * 0.5;

    const double roi_inner_sq = static_cast<double>(roi_inner) * roi_inner;
    const double center_inner_sq = static_cast<double>(center_inner) * center_inner;
    const double center_outer_sq = static_cast<double>(center_outer) * center_outer;

    double edge_sum = 0.0;
    double center_sum = 0.0;
    uint64_t edge_count = 0;
    uint64_t center_count = 0;

    constexpr uint8_t kAlphaMin = 8;  // a < 8/255 픽셀은 제외

    for (int y = 0; y < height; ++y) {
        const double dy = (y + 0.5) - cy;
        for (int x = 0; x < width; ++x) {
            const double dx = (x + 0.5) - cx;
            // r^2 (정규화). sqrt 회피 위해 제곱 비교.
            const double r_sq = (dx * dx + dy * dy) * inv_radius_sq;

            const size_t idx = (static_cast<size_t>(y) * width + x) * 4;
            const uint8_t a = rgba[idx + 3];
            if (a < kAlphaMin) {
                continue;
            }

            const double r = static_cast<double>(rgba[idx + 0]);
            const double g = static_cast<double>(rgba[idx + 1]);
            const double b = static_cast<double>(rgba[idx + 2]);
            // Rec.601 sRGB 근사, 0~1 정규화.
            const double luma = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;

            if (r_sq >= roi_inner_sq && r_sq <= 1.0) {
                edge_sum += luma;
                ++edge_count;
            } else if (r_sq >= center_inner_sq && r_sq < center_outer_sq) {
                // center 환형 밴드 [center_inner, center_outer). center_inner=0이면 원판.
                center_sum += luma;
                ++center_count;
            }
        }
    }

    if (edge_count == 0 || center_count == 0) {
        return result;  // 판정 불가.
    }

    result.edge_lum = edge_sum / static_cast<double>(edge_count);
    result.center_lum = center_sum / static_cast<double>(center_count);
    result.valid = true;
    return result;
}

}  // namespace

float measureLimbalRatio(const uint8_t* rgba, int width, int height,
                         float roi_inner, float center_inner, float center_outer) {
    const LimbalBands bands =
        measureBands(rgba, width, height, roi_inner, center_inner, center_outer);
    if (!bands.valid) {
        return -1.0f;  // 진단용: 측정 불가 신호
    }
    const double ratio = bands.edge_lum / std::max(bands.center_lum, 0.001);
    return static_cast<float>(ratio);
}

bool detectBakedLimbal(const uint8_t* rgba, int width, int height,
                       float roi_inner, float center_inner, float center_outer,
                       float threshold) {
    const LimbalBands bands =
        measureBands(rgba, width, height, roi_inner, center_inner, center_outer);
    if (!bands.valid) {
        return false;  // 보수적으로 림발 없음.
    }
    const double ratio = bands.edge_lum / std::max(bands.center_lum, 0.001);
    return ratio < static_cast<double>(threshold);
}

}  // namespace iris_sdk
