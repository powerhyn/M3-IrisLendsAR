/**
 * @file landmark_injection.h
 * @brief 랜드마크 주입 경계 — 478점 주입 저장소 + IrisResult 파생 어댑터
 *
 * ADR-0001(추적 레이어 주입형 전환) §6 C API 경계 / §6.1 동시성 계약 /
 * §6.2 detector 파생 데이터의 경계 처리 / §7 좌표·시맨틱 계약의 코어 측 구현.
 *
 * 이 헤더는 두 가지를 제공한다:
 *   1. LandmarkInjectionStore — seqlock 더블버퍼 주입 저장소 (deep-copy, generation 결속)
 *   2. deriveIrisResult()    — 478점 + frame dims → iris_sdk::IrisResult 파생 어댑터
 *
 * ③-1(동작 불변) 단계: 신규 주입 경로를 부가 채널로 가동한다. 기존 detector 직접
 * 경로(iris_sdk_detect*, render_with_result, v2 함수군)는 무변경으로 유지하며,
 * 파생 어댑터는 현 mediapipe_detector.cpp의 파생 수식을 동작 불변으로 재현한다
 * (홍채 중심/반경, face_rect 메시 바운딩, EAR 기반 eyelid). 완전한 경로 단일화는 ④.
 */

#ifndef IRIS_SDK_LANDMARK_INJECTION_H
#define IRIS_SDK_LANDMARK_INJECTION_H

#include "iris_sdk/types.h"

#include <atomic>
#include <array>
#include <cstdint>
#include <mutex>

namespace iris_sdk {

// ============================================================
// 랜드마크 주입 규약 상수 (ADR §7.0 — LensSimulator ADR-0002 승계)
// ============================================================
//
// 좌표 계약(요약, 상세는 types.h IrisResult 주석):
//   - 주입 478점은 회전 보정 완료(upright) 프레임 기준 정규화 좌표 [0,1] (x,y) + z.
//   - 미러는 렌더 단일 책임. 주입 좌표는 항상 비미러(센서 원본 upright).
//   - 홍채 z(인덱스 468~477)는 기하 계산에 사용 금지.
//
// ④ §7.3 canonical relabeling: left/right 라벨을 MediaPipe canonical 해부학 명명에
//    정합한다(LandmarkIndices.kt 정본 — 468=RIGHT_IRIS=피험자 우안, 473=LEFT_IRIS=피험자 좌안).
//    필드/상수 이름은 불변, 인덱스 그룹 값만 left↔right 교환한다.
//    left_iris ← 473그룹(피험자 좌안), right_iris ← 468그룹(피험자 우안).
namespace landmark_indices {

/// 주입 계약: 점 수는 478 고정 (ADR §6.1 입력 유효성 — 478 외 거부)
inline constexpr int kNumPoints = 478;

/// IrisResult.left_iris ← {중심 473, 경계 474~477} (canonical LEFT_IRIS = 피험자 좌안 §7.3)
inline constexpr std::array<int, 5> kLeftIris = {473, 474, 475, 476, 477};

/// IrisResult.right_iris ← {중심 468, 경계 469~472} (canonical RIGHT_IRIS = 피험자 우안 §7.3)
inline constexpr std::array<int, 5> kRightIris = {468, 469, 470, 471, 472};

/// EAR 계산용 6점 (temporal_stabilizer.cpp kLeftEAR/kRightEAR 동일)
/// 순서: p1, p2, p3, p4, p5, p6 → EAR=(|p2-p6|+|p3-p5|)/(2·|p1-p4|)
/// canonical §7.3: kLeftEAR=362그룹(피험자 좌안), kRightEAR=33그룹(피험자 우안).
inline constexpr std::array<int, 6> kLeftEAR = {362, 385, 387, 263, 373, 380};
inline constexpr std::array<int, 6> kRightEAR = {33, 160, 158, 133, 153, 144};

}  // namespace landmark_indices

// ============================================================
// 파생 어댑터 — 478점 + frame dims → IrisResult
// ============================================================

/**
 * @brief 주입된 478점 + upright 프레임 치수로부터 IrisResult 파생값을 계산한다.
 *
 * 현 mediapipe_detector.cpp의 파생 수식을 동작 불변으로 재현한다:
 *   - 홍채 중심·5점: 인덱스 473그룹(left=피험자 좌안)/468그룹(right=피험자 우안) 복사 (§7.3 canonical)
 *   - 반경: 중심에서 경계 4점까지 평균 픽셀 거리 (calculateIrisRadius — x·W, y·H 환산)
 *   - face_rect: 478점 메시 바운딩 박스, 정규화 좌표 (0~1 범위 점만 집계)
 *   - eyelid_ratio_left/right: 0.0f (현 detector가 W3 미구현으로 0 고정 — 동작 불변)
 *   - detected: 홍채 5점 모두 [0,1] 범위면 검출로 판정 (extractIris 로직)
 *
 * @note confidence는 ADR §6.2대로 경계에서 제거된다(주입에 face detection score 부재).
 *       주입 경로에서는 confidence를 0.0f로 두고, 게이팅은 visibility(EAR 파생)로 한다.
 *       동작 불변 검증은 재현 가능한 기하 파생값(중심·반경·face_rect·EAR)으로 한다.
 *
 * @param pts          num_points×3 (x,y,z) 정규화 좌표. 길이 ≥ kNumPoints×3 가정.
 * @param num_points   점 수 (kNumPoints == 478 가정 — 호출 전 검증됨).
 * @param frame_width  upright 프레임 너비 (px) — 픽셀 환산 기준 (렌더 타깃과 별개, §6.1).
 * @param frame_height upright 프레임 높이 (px).
 * @param timestamp_us 단조 증가 타임스탬프 (µs). IrisResult.timestamp_ms로 환산 저장.
 * @return 파생된 IrisResult (face_mesh 478점 인라인 포함, face_mesh_valid=true).
 */
IrisResult deriveIrisResult(const float* pts,
                            int num_points,
                            int frame_width,
                            int frame_height,
                            int64_t timestamp_us);

/**
 * @brief 한쪽 눈 EAR(Eye Aspect Ratio) 계산 (정규화 좌표 기준 — 현 detector 동작 불변).
 *
 * temporal_stabilizer.cpp::computeEAR와 동일 수식·동일 좌표공간(정규화)이다.
 * ADR §7.0은 픽셀 환산을 요구하나 그것은 ③-2 결함 수리 범위이며, ③-1은 동작 불변이다.
 *
 * @param mesh   478점 face_mesh (IrisLandmark 배열).
 * @param left_eye true=피험자 좌안(kLeftEAR=362그룹), false=피험자 우안(kRightEAR=33그룹). §7.3
 * @return EAR 값. mesh 무효(수평거리 ≈0) 시 0.0f.
 */
float computeEyeAspectRatio(const IrisLandmark* mesh, bool left_eye);

// ============================================================
// 주입 저장소 — seqlock 더블버퍼 (ADR §6.1)
// ============================================================

/**
 * @brief 478점 + frame dims + timestamp를 한 세대에 원자 결속하는 seqlock 더블버퍼.
 *
 * 동시성 계약 (ADR §6.1):
 *   - write(): 호출 스레드에서 코어 내부 버퍼로 deep-copy. 호출자 버퍼 수명은 반환 시 종료.
 *   - seqlock: 쓰기 전후 generation 증가(홀수=쓰기 중). 읽기는 전후 generation 일치 +
 *     짝수 확인, 불일치 시 재시도.
 *   - 478점 + frame_width/height + timestamp_us가 한 세대(generation)에 결속된다.
 *
 * 입력 유효성(ADR §6.1) — write() 거부 시 직전 유효 세대 유지(스테일 유지), generation 불변:
 *   - num_points != 478, pts == NULL, NaN/Inf 포함, frame dims ≤ 0 → 거부(false).
 *
 * Writer 다중 / reader 다중을 모두 허용한다. write()는 내부 writer 전용 mutex(writer_mutex_)로
 * 직렬화되므로, 서로 다른 스레드의 write()가 겹쳐도 seqlock generation 규율(짝수=완결)이
 * 보존된다. ③-1 배선상 두 writer 경로(detect 내부 공급 feed_landmark_store / 공개 주입
 * iris_set_landmarks)가 동일 인스턴스를 공유하므로, store 자체가 writer 직렬화를 책임진다
 * (ADR §6.1 동시성 계약 — 단일 writer 전제 폐기, torn-write 봉쇄). reader(readDerived/
 * generation())는 무락 유지라 seqlock 읽기 성능은 불변이다 (검출 스레드 N → 렌더/UI reader 다수).
 */
class LandmarkInjectionStore {
public:
    LandmarkInjectionStore() = default;

    // 비복사 (내부 atomic generation은 복사 불가; 단일 인스턴스 사용 전제)
    LandmarkInjectionStore(const LandmarkInjectionStore&) = delete;
    LandmarkInjectionStore& operator=(const LandmarkInjectionStore&) = delete;

    /**
     * @brief 478점 + frame dims + timestamp를 deep-copy 주입한다.
     * @return 성공 시 true (generation 증가), 입력 거부 시 false (generation 불변).
     * @param out_generation 성공 시 갱신된 세대 번호 출력 (nullptr 허용).
     */
    bool write(const float* pts,
               int num_points,
               int frame_width,
               int frame_height,
               int64_t timestamp_us,
               uint32_t* out_generation);

    /**
     * @brief 가장 최근 완결 세대의 파생 IrisResult를 읽는다 (seqlock 재시도).
     * @param out 파생된 IrisResult 출력.
     * @return 유효 세대가 있으면 true. 주입 이력 없음(generation==0)이면 false.
     */
    bool readDerived(IrisResult* out) const;

    /// 현재 세대 번호 (홀수=쓰기 중일 수 있음 — 진단/노출용).
    uint32_t generation() const noexcept {
        return generation_.load(std::memory_order_acquire);
    }

private:
    // seqlock 더블버퍼 슬롯: 478점 + frame dims + timestamp를 원자 결속.
    struct Slot {
        std::array<float, landmark_indices::kNumPoints * 3> pts{};
        int32_t frame_width = 0;
        int32_t frame_height = 0;
        int64_t timestamp_us = 0;
    };

    // writer 직렬화 전용 mutex. write() 본문 전체를 보호해 다중 writer 경합 시에도
    // seqlock generation 규율(relaxed load → +1 → +2)이 단일 writer처럼 진행되게 한다.
    // reader는 이 mutex를 잡지 않는다(무락 seqlock 읽기 — ADR §6.1).
    std::mutex writer_mutex_;
    // generation: 0=미주입, 짝수=완결, 홀수=쓰기 중. write마다 +2(짝→짝).
    std::atomic<uint32_t> generation_{0};
    // 더블버퍼: 짝수 세대는 buffers_[(gen/2)&1]에 기록.
    Slot buffers_[2];
};

}  // namespace iris_sdk

#endif  // IRIS_SDK_LANDMARK_INJECTION_H
