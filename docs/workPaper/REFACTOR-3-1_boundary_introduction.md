# REFACTOR-3-1: 경계 도입 — 랜드마크 주입 C API (③-1)

| 항목 | 내용 |
|---|---|
| 상태 | ✅ 구현 완료 (2026-06-11) — 실기기 A/B 육안 확인(사용자 판정) 후 머지 |
| 브랜치 | `refactor/p31-boundary` (refactor/p1-audit 기준) |
| 근거 | ADR-0001 (승인, 옵션 B 확정) §6/§6.1/§7/§12, 계획 문서 ③-1 |
| 원칙 | 동작 불변 — 골든 베이스라인 통과가 머지 조건 |

## 수행 내역

1. **주입 저장소** `landmark_injection.{h,cpp}` 신설 — seqlock 더블버퍼 + deep-copy + writer 뮤텍스(reader는 무락), 478점+frame_width/height+timestamp_us가 generation에 원자 결속
2. **C API 신설** (`sdk_api.h` 추가만): `iris_set_landmarks(pts, num_points, frame_width, frame_height, timestamp_us, out_generation)` + `iris_get_landmark_generation()` — 478 외/NULL/NaN/치수≤0 거부 시 기존 세대 유지(스테일 유지, silent false 금지), 스레딩 계약 doxygen 명문화
3. **파생 어댑터** — 478점 → 홍채 중심(468/473)·경계 4점(§7.0 순서)·반경(주입 프레임 치수로 픽셀 환산 — 함정 #5 봉쇄), EAR eyelid_ratio, face_rect. 기존 detector 수식 동일 재현(골든 JSON 대조 테스트 포함)
4. **내부 공급자 연결(safe-minimal)** — `iris_sdk_detect`/`iris_sdk_detect_with_rotation` 경로가 검출 결과를 주입 저장소에 side-channel 공급. 기존 반환 경로·출력 불변. v2 함수군 연결과 완전 경로 단일화는 ④로 이월
5. **types.h 계약 명문화** — ADR §7 좌표 계약(upright 정규화, 홍채 z 금지, 해부학 명명, 미러 렌더 단일 책임) 주석 편입 (동작 무변경)
6. **테스트** `test_landmark_injection.cpp` 12건 (store 동시성 스트레스 + 이중 writer 회귀 + 어댑터 vs 골든 JSON ε 대조)

## 적대 리뷰 (4렌즈) 및 반영

- **critical 1건 (실증)**: 이중 writer(내부 feed ∥ 공개 iris_set_landmarks) 경합 시 seqlock 붕괴 — 적대 테스트로 torn snapshot 16% 수용 + generation 홀수 영구 잔류 실증 → **store 소유 writer 뮤텍스로 봉합** (reader 무락 유지, 동작 불변), 이중 writer 회귀 테스트 추가
- important 5건(대부분 critical과 동근원) 전부 반영: 테스트 플레이크(부하 시 reader 기아) 수정 포함
- minor 이월: C API 표면 에러 코드 직접 테스트(③-2), v2 경로 연결(④), TSan 형식적 race의 atomic_ref 전환 검토(④)

## 검증 (최종 — 오케스트레이터 직접 재실행 포함)

- 골든 게이트: 37/37 PASS exit 0 (베이스라인 무수정 — git diff 클린 확인)
- ctest: 769/776 통과 — 실패 7건은 stash 대조로 전부 pre-existing 확정 (TFLite 링크 실패 2 포함), 신규 회귀 0
- test_landmark_injection: 12/12
- 공개 헤더: 기존 선언 변경 0 (추가만), JNI/Java/데모 무수정

## 잔여 (머지 게이트)

- [x] 실기기 A/B 육안 확인 — **통과** (2026-06-12, Galaxy S23+/Android 16): 렌즈 정합·추적·성능 차이 없음 (사용자 판정)
  - 참고: 뷰티가 "예전 기능 느낌"이라는 관찰 1건 — 원인 분류 결과 ③-1 무관 확정 (③-1 diff는 뷰티 경로 미접촉, 빌드 소스는 develop HEAD=P8-W1 머지본 그대로). P8-W1이 실기기 검증 2항 잔여 상태로 머지된 것이 유력 원인. **사용자 결정: 리팩토링 완료 후 다듬기** (P8-W1 잔여 검증과 함께 처리)
- [ ] 데모 검증 통로 정화는 별도 W (실기기 검증 선행 조건 — 감사 §9-①)

## 다음 단계

- ③-2 코어 품질 리팩토링 (blocker→major→minor, 골든 통과 조건) — 사용자 승인 후
- ④에서: v2 경로 연결/정리, 경로 단일화, MediaPipe Tasks 통합

## 변경 이력

- 2026-06-11: 구현 + 적대 리뷰 critical 봉합 + 골든 PASS
