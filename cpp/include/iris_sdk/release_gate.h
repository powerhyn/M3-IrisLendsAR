/**
 * @file release_gate.h
 * @brief 릴리즈 게이트 체크리스트 자동화 시스템
 *
 * 3-tier 게이트 구조로 릴리즈 준비 상태를 자동 판정한다.
 * - Tier 1: Hard-Stop (즉시 NO-GO, 면제 불가)
 * - Tier 2: Quantitative (자동 측정, 면제 불가)
 * - Tier 3: Qualitative (수동 입력, CONDITIONAL GO 허용)
 *
 * @author IrisLensSDK Team
 * @version 1.0.0
 * @copyright Apache 2.0 License
 */

#pragma once

#include <string>
#include <vector>

namespace iris_sdk {

// ============================================================================
// 디바이스 티어
// ============================================================================

/**
 * @brief 디바이스 성능 등급
 *
 * gpu_beauty_backend.h에도 정의되어 있으나,
 * release_gate 모듈의 독립적 사용을 위해 별도 선언한다.
 */
enum class DeviceTier { HIGH, MID, LOW };

// ============================================================================
// 게이트 입력 구조체
// ============================================================================

/**
 * @brief Tier 1: Hard-Stop 게이트 입력
 *
 * 모든 항목이 통과해야 릴리즈 가능.
 * 하나라도 실패하면 즉시 NO-GO이며 면제 불가.
 */
struct HardStopInput {
    bool   crash_free     = false;   ///< 크래시 없음 (모든 디바이스/티어)
    bool   no_memory_leak = false;   ///< 메모리 누수 없음 (10분 사용 기준)
    double frame_time_ms  = 999.0;   ///< 프레임 시간 (<=33ms = 30fps 이상)
    double temporal_cv    = 1.0;     ///< 시간적 변동 계수 (< 0.05 = 통과)
};

/**
 * @brief Tier 2: 정량적 게이트 입력
 *
 * 자동 측정되는 품질 지표. 모든 항목 통과 필수.
 */
struct QuantitativeInput {
    double    laplacian_reduction     = 0.0;   ///< Laplacian variance 감소율 (0.3~0.6 = 통과)
    double    non_skin_ssim           = 0.0;   ///< 비-피부 영역 SSIM (> 0.95 = 통과)
    double    freq_sep_time_ms        = 999.0; ///< FreqSep 처리 시간 (티어별 기준)
    int       texture_pool_additional = 99;    ///< TexturePool 추가 텍스처 수 (<= 3 = 통과)
    DeviceTier device_tier            = DeviceTier::HIGH; ///< 대상 디바이스 티어
};

/**
 * @brief Tier 3: 정성적 게이트 입력 (수동)
 *
 * 사용자 테스트 및 시각적 검수 결과.
 * 1개 이하 실패 시 CONDITIONAL GO 허용.
 */
struct QualitativeInput {
    double blind_ab_preference  = 0.0;   ///< Blind A/B 선호도 (> 0.7 = 통과)
    double blurry_feedback_ratio = 1.0;  ///< "뭉개짐" 피드백 비율 (< 0.1 = 통과)
    double fake_feedback_ratio  = 1.0;   ///< "인위적" 피드백 비율 (< 0.1 = 통과)
    bool   skin_tone_uniform    = false; ///< 피부톤 그룹 균일성
    bool   no_halo              = false; ///< 코/입 주변 halo 없음
    bool   no_contour_blur      = false; ///< 눈썹/윤곽 번짐 없음
};

// ============================================================================
// 게이트 결과 구조체
// ============================================================================

/**
 * @brief 개별 게이트 체크 결과
 */
struct GateCheckResult {
    std::string gate_name; ///< 게이트 이름 (예: "crash_free")
    bool        passed;    ///< 통과 여부
    std::string detail;    ///< 상세 정보 (예: "frame_time: 28ms <= 33ms")
};

/**
 * @brief 최종 릴리즈 판정
 */
enum class ReleaseVerdict {
    GO,              ///< 모든 게이트 통과
    CONDITIONAL_GO,  ///< Hard-stop + Quantitative 통과, Qualitative 1개 이하 실패
    NO_GO            ///< Hard-stop 또는 Quantitative 실패
};

/**
 * @brief 종합 릴리즈 게이트 결과
 */
struct ReleaseGateResult {
    ReleaseVerdict                verdict = ReleaseVerdict::NO_GO; ///< 최종 판정
    std::vector<GateCheckResult>  hard_stop_results;     ///< Tier 1 결과
    std::vector<GateCheckResult>  quantitative_results;  ///< Tier 2 결과
    std::vector<GateCheckResult>  qualitative_results;   ///< Tier 3 결과
    int hard_stop_pass_count     = 0;  ///< Hard-stop 통과 수
    int hard_stop_total          = 0;  ///< Hard-stop 전체 수
    int quantitative_pass_count  = 0;  ///< Quantitative 통과 수
    int quantitative_total       = 0;  ///< Quantitative 전체 수
    int qualitative_pass_count   = 0;  ///< Qualitative 통과 수
    int qualitative_total        = 0;  ///< Qualitative 전체 수
    std::string summary;               ///< 한 줄 요약
};

// ============================================================================
// ReleaseGate - 릴리즈 게이트 평가기
// ============================================================================

/**
 * @brief 3-tier 릴리즈 게이트 체크리스트 자동화
 *
 * 판정 로직:
 * - Hard-stop 1개라도 실패 -> NO_GO
 * - Quantitative 1개라도 실패 -> NO_GO
 * - Hard-stop + Quantitative 모두 통과:
 *   - Qualitative 모두 통과 -> GO
 *   - Qualitative 1개 이하 실패 -> CONDITIONAL_GO
 *   - Qualitative 2개 이상 실패 -> NO_GO
 *
 * @code
 *   ReleaseGate gate;
 *   auto result = gate.evaluate(hard_stop, quantitative, qualitative);
 *   std::string report = ReleaseGate::formatReport(result);
 * @endcode
 */
class ReleaseGate {
public:
    ReleaseGate() noexcept = default;

    /**
     * @brief Tier 1: Hard-Stop 게이트 평가
     * @param input Hard-stop 입력 데이터
     * @return 개별 게이트 체크 결과 목록
     */
    [[nodiscard]] std::vector<GateCheckResult> evaluateHardStop(
        const HardStopInput& input) const noexcept;

    /**
     * @brief Tier 2: 정량적 게이트 평가
     * @param input Quantitative 입력 데이터
     * @return 개별 게이트 체크 결과 목록
     */
    [[nodiscard]] std::vector<GateCheckResult> evaluateQuantitative(
        const QuantitativeInput& input) const noexcept;

    /**
     * @brief Tier 3: 정성적 게이트 평가
     * @param input Qualitative 입력 데이터
     * @return 개별 게이트 체크 결과 목록
     */
    [[nodiscard]] std::vector<GateCheckResult> evaluateQualitative(
        const QualitativeInput& input) const noexcept;

    /**
     * @brief 전체 릴리즈 게이트 평가
     *
     * 3개 티어를 모두 평가하고 최종 판정을 산출한다.
     *
     * @param hard_stop     Tier 1 입력
     * @param quantitative  Tier 2 입력
     * @param qualitative   Tier 3 입력
     * @return 종합 릴리즈 게이트 결과
     */
    [[nodiscard]] ReleaseGateResult evaluate(
        const HardStopInput& hard_stop,
        const QuantitativeInput& quantitative,
        const QualitativeInput& qualitative) const noexcept;

    /**
     * @brief 사람이 읽을 수 있는 릴리즈 리포트 생성
     * @param result 릴리즈 게이트 결과
     * @return 포맷된 텍스트 리포트
     */
    [[nodiscard]] static std::string formatReport(
        const ReleaseGateResult& result) noexcept;

private:
    /**
     * @brief 디바이스 티어별 최대 FreqSep 처리 시간
     * @param tier 디바이스 티어
     * @return 허용 최대 시간 (ms)
     */
    [[nodiscard]] static double getMaxFreqSepTimeMs(DeviceTier tier) noexcept;
};

} // namespace iris_sdk
