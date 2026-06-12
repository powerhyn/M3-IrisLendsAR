package com.irislenssdk.demo.tracking

/**
 * MediaPipe FaceLandmarker 478점 인덱스 규약.
 *
 * 이식 소스: LensSimulator `sdk/android/lenssdk/.../internal/LandmarkIndices.kt`
 *   리비전 **6bacaec** (명명 정본 — ADR §7.3, 값 무변경 이식 plan §2). 이 리비전과의
 *   차이는 패키지 치환뿐이다.
 *
 * 홍채 경계 순서는 **이미지 좌표 기준 right→top→left→bottom** (ADR-0002에서 실측 확정).
 * patlevin式 IrisIndex의 LEFT/RIGHT 라벨은 반대이므로 신뢰 금지.
 * "RIGHT"는 피험자 기준 우안 (비미러링 영상에서 화면 왼쪽 눈).
 */
internal object LandmarkIndices {
    const val LANDMARK_COUNT = 478

    // 피험자 우안 홍채 (중심 + 경계 right/top/left/bottom)
    const val RIGHT_IRIS_CENTER = 468
    const val RIGHT_IRIS_RIGHT = 469
    const val RIGHT_IRIS_TOP = 470
    const val RIGHT_IRIS_LEFT = 471
    const val RIGHT_IRIS_BOTTOM = 472

    // 피험자 좌안 홍채
    const val LEFT_IRIS_CENTER = 473
    const val LEFT_IRIS_RIGHT = 474
    const val LEFT_IRIS_TOP = 475
    const val LEFT_IRIS_LEFT = 476
    const val LEFT_IRIS_BOTTOM = 477

    /** 피험자 우안 눈 윤곽 16점 (오클루전 마스크 폴리곤 순서) */
    val RIGHT_EYE_CONTOUR = intArrayOf(
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
    )

    /** 피험자 좌안 눈 윤곽 16점 */
    val LEFT_EYE_CONTOUR = intArrayOf(
        362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
    )

    // ───────── 뷰티 (피부 마스크 + 턱선 워프) ─────────

    /**
     * 얼굴 외곽(FACEMESH_FACE_OVAL) 36점 — 순서 체인:
     * 이마(10) → 피험자 좌측면 → 턱끝(152) → 피험자 우측면 → 닫힘.
     */
    val FACE_OVAL = intArrayOf(
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    )

    // FACE_OVAL 배열 내 주요 포인트 위치 (랜드마크 번호가 아니라 배열 인덱스)
    const val OVAL_FOREHEAD = 0 // 10 — 얼굴 세로축 상단
    const val OVAL_CHEEK_LEFT = 8 // 454 — 피험자 좌측 광대 (얼굴 폭 측정)
    const val OVAL_CHIN = 18 // 152 — 얼굴 세로축 하단
    const val OVAL_CHEEK_RIGHT = 28 // 234 — 피험자 우측 광대

    /**
     * V라인 워프 제어점 (FACE_OVAL 배열 인덱스) — 측면 7점 × 양쪽, 볼 상부→턱끝 옆 순.
     * 좌측: (323,361,288,397,365,379,378) / 우측: (93,132,58,172,136,150,149).
     * 광대 직하(볼 상부)부터 완만하게 시작해야 자연스럽다는 실기기 피드백으로
     * 하악선 5점에서 7점으로 상방 확장 (2026-06-12). 광대 정점(8/28)·턱끝(152)은 제외 —
     * 광대 정점은 눈과 가까워 "눈 높이 변위 ≈ 0" 불변 조건(렌즈 정합)을 위협한다.
     */
    val OVAL_JAW_CONTROL = intArrayOf(9, 10, 11, 12, 13, 14, 15, 27, 26, 25, 24, 23, 22, 21)

    /** 눈썹 폴리곤 (피부 마스크 제외 영역) — 아래 폴리라인 + 위 폴리라인 역순 */
    val RIGHT_BROW = intArrayOf(46, 53, 52, 65, 55, 107, 66, 105, 63, 70)
    val LEFT_BROW = intArrayOf(276, 283, 282, 295, 285, 336, 296, 334, 293, 300)

    /** 입술 외곽 링 20점 (FACEMESH_LIPS 외곽 체인) */
    val LIPS_OUTER = intArrayOf(
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
    )
}
