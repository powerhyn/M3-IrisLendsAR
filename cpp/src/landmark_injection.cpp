/**
 * @file landmark_injection.cpp
 * @brief 랜드마크 주입 경계 구현 (ADR-0001 §6/§6.1/§6.2/§7)
 *
 * LandmarkInjectionStore(seqlock 더블버퍼) + deriveIrisResult(파생 어댑터).
 * 파생 수식은 현 mediapipe_detector.cpp를 동작 불변으로 재현한다 (주석에 출처 명시).
 */

#include "iris_sdk/landmark_injection.h"

#include <cmath>
#include <cstring>

namespace iris_sdk {

namespace {

/// 정규화 좌표 유효성: [0,1] 범위인지 (현 detector extractIrisFromFaceLandmarkV2 로직).
inline bool inUnitRange(float x, float y) {
    return x >= 0.0f && x <= 1.0f && y >= 0.0f && y <= 1.0f;
}

/// NaN/Inf 검출 (ADR §6.1 입력 유효성).
inline bool isFiniteF(float v) {
    return std::isfinite(v);
}

}  // namespace

// ============================================================
// EAR (정규화 좌표) — temporal_stabilizer.cpp::computeEAR 동작 불변 재현
// ============================================================
float computeEyeAspectRatio(const IrisLandmark* mesh, bool left_eye) {
    if (mesh == nullptr) {
        return 0.0f;
    }
    // EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|) — 정규화 좌표 그대로 (현 detector 동작).
    const auto& idx = left_eye ? landmark_indices::kLeftEAR
                               : landmark_indices::kRightEAR;
    auto dist = [&](int a, int b) -> float {
        float dx = mesh[a].x - mesh[b].x;
        float dy = mesh[a].y - mesh[b].y;
        return std::sqrt(dx * dx + dy * dy);
    };
    float vertical1 = dist(idx[1], idx[5]);   // |p2-p6|
    float vertical2 = dist(idx[2], idx[4]);   // |p3-p5|
    float horizontal = dist(idx[0], idx[3]);  // |p1-p4|
    if (horizontal < 1e-6f) {
        return 0.0f;
    }
    return (vertical1 + vertical2) / (2.0f * horizontal);
}

// ============================================================
// 파생 어댑터
// ============================================================
IrisResult deriveIrisResult(const float* pts,
                            int num_points,
                            int frame_width,
                            int frame_height,
                            int64_t timestamp_us) {
    IrisResult result{};  // 모든 필드 0/false; avg_iris_luma_*는 멤버 초기값 -1.0f 유지.

    if (pts == nullptr || num_points < landmark_indices::kNumPoints ||
        frame_width <= 0 || frame_height <= 0) {
        // 호출 전 검증되지만 방어적 — 빈 결과(detected=false) 반환.
        result.detected = false;
        result.frame_width = frame_width > 0 ? frame_width : 0;
        result.frame_height = frame_height > 0 ? frame_height : 0;
        return result;
    }

    result.frame_width = frame_width;
    result.frame_height = frame_height;
    // timestamp: 주입은 µs, IrisResult는 ms (현 detector 계약). 정수 나눗셈으로 환산.
    result.timestamp_ms = timestamp_us / 1000;

    // ----- face_mesh 478점 인라인 복사 (visibility=1, z는 입력 그대로) -----
    for (int i = 0; i < landmark_indices::kNumPoints; ++i) {
        result.face_mesh[i].x = pts[i * 3 + 0];
        result.face_mesh[i].y = pts[i * 3 + 1];
        result.face_mesh[i].z = pts[i * 3 + 2];
        result.face_mesh[i].visibility = 1.0f;
    }
    result.face_mesh_valid = true;

    // ----- 홍채 5점 추출 (extractIrisFromFaceLandmarkV2 동작 불변) -----
    // detector와 동일: 모든 5점이 [0,1] 범위여야 detected=true, 한 점이라도 벗어나면 false.
    auto extractIris = [&](const std::array<int, 5>& indices,
                           IrisLandmark* out_iris) -> bool {
        bool valid = true;
        for (int i = 0; i < 5; ++i) {
            int idx = indices[i];
            float x = pts[idx * 3 + 0];
            float y = pts[idx * 3 + 1];
            float z = pts[idx * 3 + 2];
            if (!inUnitRange(x, y)) {
                valid = false;
                break;
            }
            out_iris[i].x = x;
            out_iris[i].y = y;
            out_iris[i].z = z;
            out_iris[i].visibility = 1.0f;
        }
        return valid;
    };

    result.left_detected = extractIris(landmark_indices::kLeftIris, result.left_iris);
    result.right_detected = extractIris(landmark_indices::kRightIris, result.right_iris);
    result.detected = result.left_detected || result.right_detected;

    // ----- 반경 (calculateIrisRadius 동작 불변: 픽셀 환산 후 평균 거리) -----
    // 중심(인덱스 0)에서 경계 4점(1~4)까지 평균 픽셀 거리. x·W, y·H 환산.
    auto irisRadiusPx = [&](const IrisLandmark* iris) -> float {
        float cx = iris[0].x * static_cast<float>(frame_width);
        float cy = iris[0].y * static_cast<float>(frame_height);
        float total = 0.0f;
        for (int i = 1; i < 5; ++i) {
            float px = iris[i].x * static_cast<float>(frame_width);
            float py = iris[i].y * static_cast<float>(frame_height);
            float dx = px - cx;
            float dy = py - cy;
            total += std::sqrt(dx * dx + dy * dy);
        }
        return total / 4.0f;
    };
    if (result.left_detected) {
        result.left_radius = irisRadiusPx(result.left_iris);
    }
    if (result.right_detected) {
        result.right_radius = irisRadiusPx(result.right_iris);
    }

    // ----- face_rect: 478점 메시 바운딩 박스 (정규화 좌표, 동작 불변) -----
    // mediapipe_detector.cpp:3211-3254 — 0~1 범위 점만 집계.
    {
        float min_x = 1.0f, min_y = 1.0f, max_x = 0.0f, max_y = 0.0f;
        for (int i = 0; i < landmark_indices::kNumPoints; ++i) {
            const auto& lm = result.face_mesh[i];
            if (lm.x >= 0.0f && lm.x <= 1.0f && lm.y >= 0.0f && lm.y <= 1.0f) {
                min_x = std::min(min_x, lm.x);
                min_y = std::min(min_y, lm.y);
                max_x = std::max(max_x, lm.x);
                max_y = std::max(max_y, lm.y);
            }
        }
        result.face_rect.x = min_x;
        result.face_rect.y = min_y;
        result.face_rect.width = max_x - min_x;
        result.face_rect.height = max_y - min_y;
    }

    // ----- eyelid_ratio: 현 detector가 W3 미구현으로 0 고정 (동작 불변) -----
    result.eyelid_ratio_left = 0.0f;
    result.eyelid_ratio_right = 0.0f;

    // confidence는 ADR §6.2대로 경계에서 제거(주입에 face detection score 부재).
    // 게이팅은 visibility(EAR 파생)로 일원화. 여기서는 0.0f 유지(IrisResult{} 초기값).
    // iris_quality_*, eye_refiner_used 등 detector 전용 메타도 0/false 유지(ADR §6.2 삭제 대상).

    return result;
}

// ============================================================
// LandmarkInjectionStore — seqlock 더블버퍼
// ============================================================
//
// generation 운용:
//   0      : 미주입(초기). readDerived는 false.
//   짝수 g : 완결. 데이터는 buffers_[(g/2) & 1]에 있다.
//   홀수   : 쓰기 진행 중(찢김 가능). reader는 재시도.
//
// write 1회마다 generation += 2 (짝→홀→짝). 쓰기 중에는 다음 완결 세대의
// 버퍼 인덱스에 기록하므로, reader가 보유한 직전 완결 버퍼와 인덱스가 항상 다르다
// (더블버퍼 안전성).
//
// 다중 writer 안전성: write() 본문은 writer_mutex_로 직렬화된다(아래 lock_guard).
// 따라서 서로 다른 스레드의 write가 겹쳐도 relaxed load → +1 → +2 시퀀스가 단일
// writer처럼 원자적으로 진행되어, 홀수 generation이 영구 잔류하거나 두 writer가 같은
// 버퍼 인덱스에 동시 기록하는 torn-write가 발생하지 않는다(ADR §6.1).
bool LandmarkInjectionStore::write(const float* pts,
                                   int num_points,
                                   int frame_width,
                                   int frame_height,
                                   int64_t timestamp_us,
                                   uint32_t* out_generation) {
    // writer 직렬화: 다중 writer 경로(detect 내부 공급 / 공개 주입)가 동일 store를 공유해도
    // seqlock generation 규율이 단일 writer처럼 진행되게 한다(ADR §6.1 torn-write 봉쇄).
    // reader는 이 lock을 잡지 않아 무락 seqlock 읽기 성능·골든 출력 모두 불변.
    std::lock_guard<std::mutex> writer_lock(writer_mutex_);

    // ----- 입력 유효성 (ADR §6.1) — 거부 시 generation 불변(스테일 유지) -----
    if (pts == nullptr ||
        num_points != landmark_indices::kNumPoints ||
        frame_width <= 0 || frame_height <= 0) {
        return false;
    }
    const int total = landmark_indices::kNumPoints * 3;
    for (int i = 0; i < total; ++i) {
        if (!isFiniteF(pts[i])) {  // NaN/Inf 거부
            return false;
        }
    }

    // ----- seqlock 쓰기 (writer_mutex_ 보유 하 — writer 간 직렬화됨) -----
    // 현 완결 세대(짝수, writer_mutex_가 다른 writer를 배제하므로 비경합) → g_next 산출.
    const uint32_t g_cur = generation_.load(std::memory_order_relaxed);
    const uint32_t g_next = g_cur + 2;            // 다음 완결 세대 (짝수 유지)
    const int write_idx = (g_next / 2) & 1;       // 직전 완결 버퍼와 다른 인덱스

    // 쓰기 시작 표시: 홀수로 전이. release로 이후 버퍼 기록이 reader에 보이는 순서 보장.
    generation_.store(g_cur + 1, std::memory_order_release);
    std::atomic_thread_fence(std::memory_order_release);

    Slot& slot = buffers_[write_idx];
    std::memcpy(slot.pts.data(), pts, sizeof(float) * static_cast<size_t>(total));
    slot.frame_width = frame_width;
    slot.frame_height = frame_height;
    slot.timestamp_us = timestamp_us;

    // 쓰기 완료: 다음 짝수 세대로 전이(완결 공개).
    std::atomic_thread_fence(std::memory_order_release);
    generation_.store(g_next, std::memory_order_release);

    if (out_generation != nullptr) {
        *out_generation = g_next;
    }
    return true;
}

bool LandmarkInjectionStore::readDerived(IrisResult* out) const {
    if (out == nullptr) {
        return false;
    }

    // seqlock 읽기: 짝수 + 전후 일치까지 재시도. 무한루프 방지 상한.
    constexpr int kMaxRetry = 64;
    for (int attempt = 0; attempt < kMaxRetry; ++attempt) {
        const uint32_t g1 = generation_.load(std::memory_order_acquire);
        if (g1 == 0) {
            return false;  // 미주입
        }
        if (g1 & 1u) {
            continue;  // 쓰기 중 — 재시도
        }
        std::atomic_thread_fence(std::memory_order_acquire);

        const int read_idx = (g1 / 2) & 1;
        const Slot& slot = buffers_[read_idx];

        // 슬롯 로컬 복사 (찢김 검출 전에 값 확보).
        std::array<float, landmark_indices::kNumPoints * 3> pts_copy = slot.pts;
        const int32_t fw = slot.frame_width;
        const int32_t fh = slot.frame_height;
        const int64_t ts_us = slot.timestamp_us;

        std::atomic_thread_fence(std::memory_order_acquire);
        const uint32_t g2 = generation_.load(std::memory_order_acquire);
        if (g1 != g2) {
            continue;  // 읽는 중 writer 개입 — 재시도
        }

        // 일관 스냅샷 확보 — 파생 후 반환.
        *out = deriveIrisResult(pts_copy.data(), landmark_indices::kNumPoints,
                                fw, fh, ts_us);
        return true;
    }
    return false;  // 과도한 경합 — 이번 호출은 실패(reader가 다음 프레임 재시도).
}

}  // namespace iris_sdk
