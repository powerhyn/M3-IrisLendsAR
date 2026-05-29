// P6-W7 B4 측정 + 레지스트리 검증 하니스 (일회성).
// 실제 iris_sdk 함수로 (1) lens_meta.json 파싱 round-trip, (2) 10 테스트 SKU 정확도 측정.
#include "iris_sdk/lens_sku_metadata.h"

#include <opencv2/opencv.hpp>

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace iris_sdk;

static cv::Mat loadRgba(const std::string& p) {
    cv::Mat img = cv::imread(p, cv::IMREAD_UNCHANGED), rgba;
    if (img.empty()) return rgba;
    if (img.channels() == 4)      cv::cvtColor(img, rgba, cv::COLOR_BGRA2RGBA);
    else if (img.channels() == 3) cv::cvtColor(img, rgba, cv::COLOR_BGR2RGBA);
    else return cv::Mat();
    if (rgba.depth() != CV_8U) rgba.convertTo(rgba, CV_8U);
    if (!rgba.isContinuous()) rgba = rgba.clone();
    return rgba;
}

int main(int argc, char** argv) {
    const std::string assets = "android/demo-app/src/main/assets/lenses";
    const std::string meta_path = "android/demo-app/src/main/assets/lens_meta.json";

    // (1) 레지스트리 round-trip
    LensSkuRegistry reg;
    bool ok = reg.loadFromFile(meta_path);
    std::printf("=== registry round-trip ===\n");
    std::printf("loadFromFile: %s, size=%zu\n", ok ? "OK" : "FAIL", reg.size());
    for (const char* k : {"romu_gray_taupe_png", "envie_chameau_brown_png", "claset_doll_choco_png", "nonexistent_png"}) {
        auto m = reg.find(k);
        if (m) std::printf("  find(%-26s) baked=%d graphic=%d crl=%d\n", k, m->has_baked_limbal, m->prefers_graphic_outline, m->prefers_crl);
        else   std::printf("  find(%-26s) -> nullopt\n", k);
    }

    // (2) B4 정확도 (실측 최적 lens-texture 파라미터)
    const float roi_inner = 0.85f, center_inner = 0.40f, center_outer = 0.55f, threshold = 0.48f;
    std::printf("\n=== B4 accuracy (roi_inner=%.2f center=[%.2f,%.2f] thr=%.2f) ===\n",
                roi_inner, center_inner, center_outer, threshold);
    std::vector<std::pair<std::string,bool>> truth = {
        {"romu_gray-taupe",true},{"romu_dear-mellow",true},{"romu_love-gleam",true},
        {"envie_plum-black",true},{"oh_bagel",true},
        {"claset_doll-choco",false},{"claset_runway-gray",false},{"claset_cloud-gray",false},
        {"envie_parfum-glow",false},{"oh_kiwi",false},
    };
    int correct = 0;
    for (auto& [sku, gt] : truth) {
        cv::Mat rgba = loadRgba(assets + "/" + sku + ".png");
        if (rgba.empty()) { std::printf("  %-24s decode FAIL\n", sku.c_str()); continue; }
        float ratio = measureLimbalRatio(rgba.data, rgba.cols, rgba.rows, roi_inner, center_inner, center_outer);
        bool det = detectBakedLimbal(rgba.data, rgba.cols, rgba.rows, roi_inner, center_inner, center_outer, threshold);
        bool hit = (det == gt);
        correct += hit;
        std::printf("  %-24s gt=%-3s ratio=%.4f det=%-3s %s\n",
                    sku.c_str(), gt?"YES":"no", ratio, det?"YES":"no", hit?"OK":"MISS");
    }
    std::printf("\naccuracy: %d/10\n", correct);
    return 0;
}
