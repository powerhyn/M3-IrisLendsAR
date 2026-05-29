/**
 * @file lens_sku_metadata.h
 * @brief P6-W7: 렌즈 SKU 메타데이터 + baked 림발 자동감지 (B4).
 *
 * 셰이더 림발 on/off 판정 소스 (메타 전용 정책, W7):
 *   1) SKU 메타데이터(lens_meta.json) — 디자이너가 명시, 유일한 런타임 권위.
 *   2) detectBakedLimbal() 자동감지 — 실측 9/10 < 10/10 엄수 기준이라 런타임 권위에서 드롭.
 *      진단/벤치 및 향후 재활성(auto_detect_fallback_=true)용으로만 보존.
 *
 * detectBakedLimbal/measureLimbalRatio는 GL 의존성 없는 순수 함수 →
 * 단위 테스트 / 벤치 하니스에서 직접 호출 가능.
 */

#ifndef IRIS_SDK_LENS_SKU_METADATA_H
#define IRIS_SDK_LENS_SKU_METADATA_H

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

namespace iris_sdk {

struct LensSkuMetadata {
    std::string sku_id;
    std::string display_name;
    bool has_baked_limbal = false;        // 림발 링이 텍스처에 구워져 있음
    bool prefers_crl = false;             // W5 공용 필드(여기선 파싱만, 미사용)
    bool prefers_graphic_outline = false; // 그래픽 자체가 강한 outline 포함
};

/**
 * @brief SKU 메타데이터 레지스트리.
 *
 * thread-safety: 로딩은 init 시점 1회 가정. 로딩 후 read-only이므로
 * find()는 락 없이 const 접근 안전. 로딩과 동시 접근은 호출자 책임.
 */
class LensSkuRegistry {
public:
    LensSkuRegistry() = default;

    /// JSON 문자열 파싱. 성공 시 true. 실패 시 false (기존 항목은 보존 안 됨).
    bool loadFromJson(const std::string& json_text);

    /// 파일을 읽어 loadFromJson에 위임. 파일 없으면 false (throw 금지).
    bool loadFromFile(const std::string& path);

    /// 등록된 SKU면 메타 반환, 없으면 nullopt.
    std::optional<LensSkuMetadata> find(const std::string& sku_id) const;

    size_t size() const { return by_sku_id_.size(); }

private:
    std::unordered_map<std::string, LensSkuMetadata> by_sku_id_;
};

// ============================================================================
// B4 baked 림발 자동감지 (순수 함수, GL 의존성 없음)
// ============================================================================

/**
 * @brief RGBA8 텍스처 버퍼에서 baked 림발 존재 추정 (W7 §5.2).
 *
 * 원형 좌표 r = 중심으로부터 정규화 거리 (이미지 중심=홍채 중심 가정,
 * 짧은 변 절반 = r=1.0). edge band(r in [roi_inner,1.0]) 평균 luma가
 * center 환형 밴드(r in [center_inner, center_outer)) 평균 luma보다
 * 충분히 어두우면 림발로 판정.
 *   ratio = edge_lum / max(center_lum, 0.001)
 *   detected = (ratio < threshold)
 * luma는 Rec.601 sRGB 근사 (0.299R+0.587G+0.114B), 0~1. alpha<8/255 픽셀 제외.
 * edge/center 유효 픽셀이 0이면 false (판정 불가 → 보수적으로 림발 없음).
 * null/0크기 입력은 false.
 *
 * 렌즈 텍스처(투명 동공)는 center_inner≈0.40, center_outer≈0.55 권장. 실측 최적:
 * roi_inner=0.85, center_inner=0.40, center_outer=0.55, threshold=0.48 → 9/10.
 * 단 런타임 권위 아님(진단용). center_inner=0이면 기존 center 원판 동작과 동일.
 */
bool detectBakedLimbal(const uint8_t* rgba, int width, int height,
                       float roi_inner = 0.85f, float center_inner = 0.0f,
                       float center_outer = 0.3f, float threshold = 0.75f);

/// 진단/벤치용: detectBakedLimbal의 ratio(edge_lum/center_lum)를 그대로 노출.
/// 유효 픽셀 부족/잘못된 입력이면 음수(-1.0f) 반환.
float measureLimbalRatio(const uint8_t* rgba, int width, int height,
                         float roi_inner = 0.85f, float center_inner = 0.0f,
                         float center_outer = 0.3f);

} // namespace iris_sdk

#endif // IRIS_SDK_LENS_SKU_METADATA_H
