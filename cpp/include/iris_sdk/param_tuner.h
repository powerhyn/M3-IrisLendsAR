/**
 * @file param_tuner.h
 * @brief FreqSep attenuation curve parameter tuning infrastructure
 *
 * Grid search 기반 파라미터 튜닝으로 최적의 FreqSep 파라미터 조합을 탐색한다.
 * QualityMetrics의 GateResult를 활용하여 Laplacian, SSIM, Halo 가중 점수를 산출.
 *
 * @author IrisLensSDK Team
 * @version 1.0.0
 * @copyright Apache 2.0 License
 */

#pragma once

#include "quality_metrics.h"

#include <opencv2/core.hpp>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

namespace iris_sdk {

// ============================================================================
// 파라미터 구조체
// ============================================================================

/**
 * @brief FreqSep 파이프라인 튜닝용 파라미터 집합
 *
 * mapSkinQuality()에서 사용하는 핵심 파라미터를 하나의 구조체로 묶어
 * grid search 및 preset 추천에 사용한다.
 */
struct FreqSepTuningParams {
    float attenuation_low   = 0.02f;   ///< 저주파 감쇠 계수
    float attenuation_high  = 0.15f;   ///< 고주파 감쇠 계수
    float sigma_ratio       = 0.4f;    ///< low_freq_smooth_radius_ratio
    float high_freq_preserve = 0.5f;   ///< 고주파 보존율 (1.0=전부 보존, 0.0=전부 제거)
    int   blur_radius       = 14;      ///< 블러 반경 (B-method: face_width*0.05, clamp(6,28))

    /**
     * @brief 파라미터 문자열 표현
     * @return "aLow=0.02,aHigh=0.15,sigma=0.40,preserve=0.50,radius=14"
     */
    [[nodiscard]] std::string toString() const noexcept;
};

/**
 * @brief Grid search 범위 설정
 *
 * 각 파라미터의 최소/최대값과 그리드 단계 수를 지정한다.
 * steps=3이면 min, mid, max 3단계를 탐색한다.
 */
struct TuningRange {
    float attenuation_low_min      = 0.01f;
    float attenuation_low_max      = 0.05f;
    float attenuation_high_min     = 0.08f;
    float attenuation_high_max     = 0.25f;
    float sigma_ratio_min          = 0.3f;
    float sigma_ratio_max          = 0.5f;
    float high_freq_preserve_min   = 0.10f;  ///< 고주파 보존율 최소값
    float high_freq_preserve_max   = 1.0f;   ///< 고주파 보존율 최대값
    int   steps                    = 3;      ///< 파라미터당 그리드 단계 수
};

// ============================================================================
// 결과 구조체
// ============================================================================

/**
 * @brief 단일 파라미터 조합의 평가 결과
 */
struct TuningResult {
    FreqSepTuningParams params;         ///< 사용된 파라미터 조합
    GateResult          gate_result;    ///< QualityMetrics 게이트 결과
    double              overall_score;  ///< 가중 복합 점수 (0~1)
    std::string         condition_label;///< 테스트 조건 설명
};

/**
 * @brief 프리셋 추천 결과
 *
 * NATURAL/MODERATE/STRONG 3단계에 대해
 * 최적 파라미터와 권장 skinQuality 값을 제공한다.
 */
struct PresetRecommendation {
    float natural_value   = 0.3f;  ///< NATURAL 프리셋의 skinQuality 값
    float moderate_value  = 0.5f;  ///< MODERATE 프리셋의 skinQuality 값
    float strong_value    = 0.8f;  ///< STRONG 프리셋의 skinQuality 값

    FreqSepTuningParams natural_params;   ///< NATURAL 최적 파라미터
    FreqSepTuningParams moderate_params;  ///< MODERATE 최적 파라미터
    FreqSepTuningParams strong_params;    ///< STRONG 최적 파라미터
};

// ============================================================================
// 콜백 타입
// ============================================================================

/**
 * @brief 파라미터 적용 콜백
 *
 * ParamTuner는 GPU 셰이더를 직접 실행할 수 없으므로,
 * 호출자가 이 콜백을 통해 파이프라인에 파라미터를 적용하고
 * 처리된 이미지를 반환한다.
 *
 * @param original 원본 이미지 (CV_8UC3 BGR)
 * @param params   적용할 파라미터 조합
 * @return 처리된 이미지 (CV_8UC3 BGR)
 */
using ProcessCallback = std::function<cv::Mat(
    const cv::Mat& original,
    const FreqSepTuningParams& params)>;

// ============================================================================
// ParamTuner 클래스
// ============================================================================

/**
 * @brief FreqSep 파라미터 그리드 서치 튜너
 *
 * TuningRange에 정의된 범위를 순회하며 각 파라미터 조합에 대해
 * ProcessCallback으로 처리한 결과를 QualityMetrics로 평가한다.
 *
 * 사용법:
 * @code
 *   ParamTuner tuner;
 *   auto results = tuner.gridSearch(original, skin_mask, range,
 *       [&](const cv::Mat& img, const FreqSepTuningParams& p) {
 *           return pipeline.processWithParams(img, p);
 *       });
 *   auto best = ParamTuner::findBest(results);
 *   auto presets = ParamTuner::recommendPresets(results);
 *   auto report = ParamTuner::generateReport(results, presets);
 * @endcode
 */
class ParamTuner {
public:
    ParamTuner() noexcept = default;

    /**
     * @brief Grid search 실행
     *
     * TuningRange에 정의된 범위를 steps 단계로 나누어
     * 모든 파라미터 조합을 순회한다.
     * 각 조합마다 process_fn을 호출하여 처리 결과를 얻고,
     * QualityMetrics::evaluateQuantitativeGate()로 평가한다.
     *
     * @param original        원본 이미지 (CV_8UC3 BGR)
     * @param skin_mask       피부 마스크 (CV_8UC1, 255=skin)
     * @param range           탐색 범위
     * @param process_fn      파라미터 적용 콜백
     * @param condition_label 테스트 조건 레이블
     * @return 모든 파라미터 조합의 평가 결과 목록
     */
    std::vector<TuningResult> gridSearch(
        const cv::Mat& original,
        const cv::Mat& skin_mask,
        const TuningRange& range,
        ProcessCallback process_fn,
        const std::string& condition_label = "") noexcept;

    /**
     * @brief 최적 파라미터 선택
     *
     * overall_score가 가장 높은 결과의 파라미터를 반환한다.
     * 결과가 비어있으면 기본 파라미터를 반환한다.
     *
     * @param results gridSearch 결과 목록
     * @return 최적 파라미터 조합
     */
    [[nodiscard]] static FreqSepTuningParams findBest(
        const std::vector<TuningResult>& results) noexcept;

    /**
     * @brief 가중 복합 점수 계산
     *
     * 가중치: Laplacian=0.4, SSIM=0.3, Halo=0.3
     * - Laplacian: 감소율이 45% 중심에 가까울수록 고점 (통과 시)
     * - SSIM: 값이 높을수록 고점 (통과 시)
     * - Halo: gradient 증가가 적을수록 고점 (통과 시)
     *
     * @param gate QualityMetrics 게이트 결과
     * @return 0~1 사이의 가중 점수
     */
    [[nodiscard]] static double computeScore(
        const GateResult& gate) noexcept;

    /**
     * @brief 프리셋 추천 생성
     *
     * 튜닝 결과를 high_freq_preserve 값으로 분류하여
     * NATURAL/MODERATE/STRONG 각각의 최적 파라미터를 선택한다.
     * - NATURAL: high_freq_preserve > 0.6 (약한 스무딩)
     * - MODERATE: 0.3 <= high_freq_preserve <= 0.6 (중간 스무딩)
     * - STRONG: high_freq_preserve < 0.3 (강한 스무딩)
     *
     * @param results gridSearch 결과 목록
     * @return 3단계 프리셋 추천
     */
    [[nodiscard]] static PresetRecommendation recommendPresets(
        const std::vector<TuningResult>& results) noexcept;

    /**
     * @brief B-method blur_radius 독립성 검증
     *
     * face_width * 0.05, clamp(6, 28) 공식이
     * skinQuality 값에 무관하게 동일한 blur_radius를 산출하는지 확인한다.
     */
    struct BlurRadiusValidation {
        bool        independent;   ///< skinQuality 무관하게 동일 여부
        int         radius_at_low; ///< skinQuality=0.3에서의 radius
        int         radius_at_mid; ///< skinQuality=0.5에서의 radius
        int         radius_at_high;///< skinQuality=0.8에서의 radius
        std::string detail;        ///< 검증 상세 설명
    };

    /**
     * @brief blur_radius가 skinQuality에 독립적인지 검증
     *
     * @param face_width 얼굴 너비 (픽셀)
     * @return 검증 결과
     */
    [[nodiscard]] static BlurRadiusValidation validateBlurRadiusIndependence(
        int face_width) noexcept;

    /**
     * @brief 튜닝 리포트 문자열 생성
     *
     * 전체 grid search 결과와 프리셋 추천을 포함하는
     * 사람이 읽을 수 있는 리포트를 생성한다.
     *
     * @param results gridSearch 결과 목록
     * @param presets 프리셋 추천
     * @return 리포트 문자열
     */
    [[nodiscard]] static std::string generateReport(
        const std::vector<TuningResult>& results,
        const PresetRecommendation& presets) noexcept;

private:
    /**
     * @brief 그리드 단계에 따른 파라미터 값 생성
     * @param min_val 최소값
     * @param max_val 최대값
     * @param steps   단계 수
     * @return 각 단계의 값 목록
     */
    [[nodiscard]] static std::vector<float> generateSteps(
        float min_val, float max_val, int steps) noexcept;

    /**
     * @brief B-method blur_radius 계산
     * @param face_width 얼굴 너비
     * @return clamp(face_width * 0.05, 6, 28)
     */
    [[nodiscard]] static int computeBlurRadius(int face_width) noexcept;
};

} // namespace iris_sdk
