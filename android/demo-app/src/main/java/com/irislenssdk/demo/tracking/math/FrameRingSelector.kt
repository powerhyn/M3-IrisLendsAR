package com.irislenssdk.demo.tracking.math

import kotlin.math.abs

/**
 * 프레임 동기 렌더링의 링 엔트리 선택 — 순수 함수.
 *
 * 이식 정본: LensSimulator `internal/math/FrameRingSelector.kt` (ADR-0005, 리비전 6bacaec
 * 계열). 값/로직 무변경 — 패키지만 치환. 트래킹 지연 핸드오프 §3-b의 IrisLensSDK 적용.
 *
 * 카메라 프레임 링버퍼에서 추적 결과의 분석 프레임 타임스탬프(센서 ns)와 가장 가까운
 * (|Δ| 최소) 엔트리를 고른다. 랜드마크와 픽셀이 같은 프레임이 되어 움직임 중에도
 * 렌즈-홍채 정합이 유지된다.
 *
 * 폴백 (전부 "최신 엔트리 = 현행 동작으로 자연 강등"):
 * - 스냅샷 타임스탬프 없음 (≤ 0)
 * - 최솟값 |Δ| > [MAX_SKEW_NS] (150ms — 링 깊이를 넘는 지연은 매칭 무의미)
 * 링이 비어 있으면 -1 (호출부가 직접 경로로 처리).
 *
 * 클럭 도메인 검증: `ImageInfo.timestamp`와 `SurfaceTexture.timestamp`는 같은 카메라 센서
 * 클럭이어야 매칭 가능 — 새 프레임마다 [updateClockDomainStreak]로 |Δ|를 점검해, 1초 이내가
 * 한 번이라도 관측되면 같은 도메인으로 확정하고, 연속 [CLOCK_DOMAIN_CONFIRM_STREAK]회 초과면
 * 불일치로 확정해 이후 항상 최신 엔트리를 쓴다 (기능 강등이지 오류 아님). 단발 초과로
 * 확정하지 않는 이유: 콜드 스타트의 첫 프레임은 모델 로드(GPU delegate 셰이더 컴파일 포함
 * 1초 이상 가능) 동안 쥐고 있던 묵은 프레임이라 |Δ|가 일시적으로 1초를 넘는다 — 묵음은 다음
 * 프레임에서 수십 ms로 수렴하는 반면, 진짜 도메인 불일치는 오프셋(수십 초~수일)이 일정하게
 * 유지되므로 연속 카운트가 두 경우를 확실히 구분한다.
 */
object FrameRingSelector {

    /** 매칭 허용 한계 — 이보다 먼 스냅샷은 묵은 데이터로 보고 최신 프레임에 그린다. */
    const val MAX_SKEW_NS: Long = 150_000_000L // 150ms

    /** |Δ|가 이를 넘으면 두 타임스탬프의 클럭 도메인이 다르다고 의심한다 (단발 판정). */
    const val CLOCK_DOMAIN_LIMIT_NS: Long = 1_000_000_000L // 1초

    /** 불일치 확정에 필요한 연속 초과 프레임 수 — 콜드 스타트 묵음과 진짜 불일치 구분. */
    const val CLOCK_DOMAIN_CONFIRM_STREAK = 5

    /**
     * 스냅샷 타임스탬프에 가장 가까운 링 엔트리 인덱스 선택.
     *
     * @param timestampsNs 링 엔트리 타임스탬프 (센서 ns). 인덱스 0..count-1만 유효.
     * @param count 유효 엔트리 수 (워밍업 중엔 링 크기보다 작을 수 있다)
     * @param latestIndex 가장 최근에 기록된 엔트리 인덱스 (폴백 대상)
     * @param snapshotTimestampNs 스냅샷의 분석 프레임 타임스탬프. ≤ 0이면 미상.
     * @return 선택된 인덱스. 폴백 시 [latestIndex], 링이 비었으면 -1.
     */
    fun select(
        timestampsNs: LongArray,
        count: Int,
        latestIndex: Int,
        snapshotTimestampNs: Long,
    ): Int {
        if (count <= 0) return -1
        if (snapshotTimestampNs <= 0L) return latestIndex
        var best = latestIndex
        var bestDelta = Long.MAX_VALUE
        for (i in 0 until count) {
            val delta = abs(timestampsNs[i] - snapshotTimestampNs)
            if (delta < bestDelta) {
                bestDelta = delta
                best = i
            }
        }
        return if (bestDelta > MAX_SKEW_NS) latestIndex else best
    }

    /** 유효 엔트리 중 최솟값 |Δ| (ns). 링이 비었거나 타임스탬프 미상이면 Long.MAX_VALUE. */
    fun minAbsDeltaNs(timestampsNs: LongArray, count: Int, snapshotTimestampNs: Long): Long {
        if (count <= 0 || snapshotTimestampNs <= 0L) return Long.MAX_VALUE
        var min = Long.MAX_VALUE
        for (i in 0 until count) {
            val delta = abs(timestampsNs[i] - snapshotTimestampNs)
            if (delta < min) min = delta
        }
        return min
    }

    /** 단일 스냅샷의 |Δ| > 1초 판정 — 확정은 [updateClockDomainStreak] 연속 카운트로 한다. */
    fun isClockDomainMismatch(minAbsDeltaNs: Long): Boolean = minAbsDeltaNs > CLOCK_DOMAIN_LIMIT_NS

    /**
     * 클럭 도메인 검증 스트릭 갱신 — 새 스냅샷마다 호출한다 (같은 스냅샷 재평가 금지).
     *
     * @return |Δ| > [CLOCK_DOMAIN_LIMIT_NS]면 streak + 1, 한도 이내면 0.
     *   반환값 0은 "같은 클럭 도메인 확정", [CLOCK_DOMAIN_CONFIRM_STREAK] 이상은 "불일치 확정".
     */
    fun updateClockDomainStreak(streak: Int, minAbsDeltaNs: Long): Int =
        if (isClockDomainMismatch(minAbsDeltaNs)) streak + 1 else 0
}
