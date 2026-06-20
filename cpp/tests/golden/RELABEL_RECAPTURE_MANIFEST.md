# 골든 재캡처 manifest — ④ 좌표 canonical relabeling (ADR §7.3)

> 재기준선 사유: 좌표 canonical relabeling으로 IrisResult의 `left_*`/`right_*` 데이터 라우팅을
> 피험자 해부학 기준으로 정정(left=473그룹=피험자 좌안, right=468그룹=피험자 우안). 출력의
> 좌우 의미가 바뀌므로(동작 불변 아님) 명시적 baseline 재캡처. W4-D injection 재캡처 인프라 재사용.

## 재캡처 메타
- **날짜**: 2026-06-19
- **트랙**: REFACTOR-4 ④ 좌표 canonical relabeling (W4-D에서 분리 이월된 마지막 슬라이스)
- **MediaPipe Tasks 버전**: 0.10.35 (ADR §5 고정) — 변경 없음(이번 재캡처는 추적 교체 아닌 라벨 정정)
- **model**: `face_landmarker.task` (SHA 64184e22…) — 변경 없음
- **입력 corpus**: `cpp/tests/golden/inputs/*.png` (기존 동일)
- **캡처 모드**: injection (`golden_capture --inject-from`, `golden_capture_all.sh INJECT_BASELINE`).
  동결 478점은 **재캡처 전 baseline에서 그대로 주입**(478점 mesh는 index-canonical이라 불변),
  스왑된 코어가 left/right 필드를 재파생.
- **산출물**: 18 JSON + 19 PNG (파일 집합 불변).

## 불변식 (invariant) — migration checker로 증명됨
재캡처는 **순수 L/R 라우팅 스왑**이므로 old↔new baseline은 아래 관계를 만족해야 하며,
`scripts/golden_relabel_migration_check.py`로 **PASS 확인**(2026-06-19):

- `new.left_iris      == old.right_iris`  (및 역방향) — 홍채 5점(center+경계4) 전체
- `new.left_radius    == old.right_radius`
- `new.left_detected  == old.right_detected`
- `new.eyelid_ratio_left == old.eyelid_ratio_right`
- `new.avg_iris_luma_left == old.avg_iris_luma_right` (둘 다 injection서 -1)
- 그 외 모든 필드(face_mesh 478, face_rect, face_rotation, frame_w/h, rotation/mirror/gamma, detected, confidence) **byte 동일**
- **PNG byte-identical** (CPU `iris_sdk_render_lens`는 양안 동일 렌더 + is_mirror no-op이라 L/R 스왑에 불변)

→ 재캡처가 "다른 회귀"가 아니라 **정확한 L/R 교환**임을 기계 증명. (golden_compare.py는 key별 엄격
비교라 스왑 자체는 통과 못 하므로 이 1회용 checker가 정합성 게이트.)

## before/after 메트릭
- ctest: 재캡처 전 24 fail(골든 baseline old 의존 21 + pre-existing 3) → 재캡처 후 **3 fail(pre-existing만), 회귀 0**.
- `test_golden_injection_derive` 18골든 + Derive/RoundTrip/MisswappedDims = 재캡처 baseline과 정합 PASS.
- 데스크톱 빌드: 신규 error/warning 0.

## 변경 범위 구분 (재기준선 영향 분리)
- **L/R 라우팅 스왑**(R1): 골든 JSON left/right 값 교환(위 invariant). CPU PNG 불변.
- **안각 inner/outer 정정**(R2, GPU ellipse 전용·기본 OFF): 골든 무관(CPU 골든은 ellipse 미사용).
  device GPU-ellipse 활성 시에만 눈-타원 비대칭 변화(별도 실기기 육안).
- **미러 §7.4 정합**(R3, GPU 전용): 골든 무관(CPU render는 is_mirror no-op). **전면 카메라 실기기 검증 필수**.

## 승인자
- 구현: Claude (cpp-pro 동반, C++ 코어) + 메인(글루/골든/migration checker).
- 외부 교차검증: Codex read-only 적대 리뷰 2회(fitEyeEllipse 비대칭 확인, 골든 PNG 범위·미러 구조 정정).
- **실기기 최종 승인**: 사용자 (전면 카메라 미러 R3 + 안각 R2 육안 — 대기 중).
