# REFACTOR-2: ADR + 골든 베이스라인 (2단계)

| 항목 | 내용 |
|---|---|
| 상태 | ✅ 구축 완료 — ADR 사용자 검토 대기 (2026-06-11) |
| 브랜치 | `refactor/p1-audit` |
| 계획 문서 | `docs/lenssim-handoff/refactoring-audit-and-tracking-migration-plan.md` 2단계 (2-1 ADR, 2-2 골든) |
| 산출물 | `docs/decisions/0001-landmark-injection-tracking-replacement.md`, `cpp/tests/golden/{inputs,baseline,README.md}`, `cpp/examples/golden_capture.cpp`, `scripts/golden_capture_all.sh`, `scripts/golden_compare.py` |

## 목표

1단계 감사(확정 findings 133건) 기반으로 ① 추적 레이어 주입형 전환 ADR을 확정안으로
작성하고 ② 3단계 "동작 불변" 검증의 기준선이 될 골든 베이스라인을 신축한다
(감사 §9-5: 기존 렌더 픽셀 검증 0건 — 구축이 아니라 신축).
**2단계는 동작 불변**: 코어 소스(cpp/src, cpp/include) 무수정, 신규 파일 +
examples/scripts 등록만 허용. 3단계(경계 도입·코드 이동)는 ADR 승인 후에만.

## 수행 내역

### 2-1 ADR (docs/decisions/0001)

- 결정: 478점 랜드마크 주입 경계(`iris_set_landmarks` 4함수안) + MediaPipe Tasks
  0.10.35 고정 교체, 좌표·시맨틱 계약 4종 명문 확정, cpu-render 처분 옵션 비교
  (옵션 B 권고), 16KB 정렬 대응, 후퇴 트리거 4종, 골든 검증 전략, 3단계 머지 게이트.
- 상태: **초안 — 사용자 검토 대기** (결정 항목 ① cpu-render 처분 선택 포함).

### 2-2 골든 베이스라인

- 입력 3종 × 변형 매트릭스(rot0/90/180/270, mirror, gamma05 + 첫 입력 한정
  mirror_rot270/lensmirror) = 캡처 18회, 산출물 37파일(JSON 18, render PNG 15,
  beauty PNG 4), inputs+baseline 14.1MB (≤ 15MB 규율).
- CPU 경로만 동결(데스크톱 EGL 부재) — GPU 골든은 실기기 후속.
- 결정성: 동일 코드/입력 2회 캡처 diff 0 검증 완료.

### 적대 리뷰 반영 (2026-06-11)

critical 5건 / important 8건 접수, ADR 5건 + 골든 7건 반영 (상세는 아래 이슈 절):

- ADR: §7.5 One-Euro 정정(LensSimulator 실코드 검증 — min_cutoff 0.5·픽셀 공간),
  §10 T1 판정 기준을 골든 ε에서 분리, §11 ε 적용 범위 한정 + §12 추적 교체 PR
  대체 절차, §10 후퇴 시 TFLite 16KB 재정렬 수리 트랙 편입, §6.1 입력 유효성
  계약(num_points 478) + §6.4 468 하드코딩 정리 전제.
- 골든: face_highres 입력 교체(→ face_closeup), 캡처 하네스 침묵 실패 차단,
  README에 '알려진 현재 동작'/'미커버 경로' 절 신설, lensmirror·mirror_rot270
  변형 추가, 전체 재캡처 + 결정성 재검증.

## 이슈 및 학습

1. **face_highres 병리 검출 적발 (critical)**: 초기 입력(demo sample_01)이 face
   mesh 오버레이가 얼굴에 박힌 스크린샷이었고, rot0/rot180 검출이 눈이 아닌 뺨
   (~67px 아래)에 고정된 채 conf=0.879로 기준선에 동결되어 있었다. "검출 성공률
   프로빙"만으로는 못 거른다 — **입력 선정 기준에 좌표 시각 확인(마커 합성)을
   추가**했고, 교체 입력(face_closeup, conf=0.955)은 마커 합성으로 홍채 정렬을
   확인했다.
2. **캡처 하네스의 침묵 실패 (critical)**: 720a rot90/270에서 render_lens가
   IRIS_SDK_RENDER_FAILED로 실패해 render.png 2장이 조용히 빠졌는데, 도구는
   [Warning]만 내고 exit 0, 스크립트(set -e)도 통과했다. → 캡처마다 기대 산출물
   존재 검증 + `KNOWN_RENDER_FAILURES` 허용 목록(양방향 앵커) + 위반 시 비0 종료로
   개편.
3. **회전 골든의 실체**: rot0≡rot180, rot90≡rot270 (JSON 메타 제외 완전 동일) —
   4회전의 실효 구분은 2개 값 집합. 회전 변형 render는 렌즈가 홍채 밖에 그려짐
   (rot0만 정렬 정확 — 마커 합성 확인). "회전 경로 커버"는 **JSON 동결에
   한정**됨을 README에 명시. 좌표공간 불일치의 설계 제약/버그 여부는 3단계 전
   확인 항목.
4. **is_mirror CPU no-op 발견**: lensmirror 변형 추가 과정에서
   `convert_to_cpp_lens_config`(sdk_api.cpp)가 `is_mirror`를 복사하지 않아 CPU
   렌더에서 무음 no-op임을 확인(GPU v2만 복사). lensmirror render가 rot0과 byte
   동일 — 이 no-op 자체를 골든이 앵커링한다 (코어 수정은 2단계 범위 밖).
5. **미커버 경로 명시**: 비동기 `submitFrameWithRotation`(실앱 기본 경로),
   `detectOnlyWithRotation`의 rotation==0 분기(detectOnly로 우회), letterbox
   역변환 단위 고정, `IrisLensConfig.rotation` 렌더 분기 — README 한계 절 참조.

## 재캡처 절차

`cpp/tests/golden/README.md`의 "재캡처 방법" 절이 정본이다. 요약:
golden_capture 빌드(기존 cmake-build-debug 재사용, `-DIRIS_SDK_FETCH_TFLITE=OFF`)
→ `scripts/golden_capture_all.sh` (산출물 검증 내장) → 2회 캡처 후
`scripts/golden_compare.py`로 diff 0 확인 → baseline 반영.
기준선을 의도적으로 바꾸는 수정은 의도적 변화로 분리 기록 + 사용자 승인
(ADR-0001 §11).

## 다음 단계

- ADR-0001 사용자 검토/승인 (cpu-render 처분 옵션 선택 포함) → 3단계 착수 게이트.
- 3단계 착수 전 확인 항목: 회전 render 좌표공간 불일치의 성격(설계 제약 vs 버그),
  비동기 경로 골든 보강 옵션.

## 변경 이력

| 일자 | 내용 |
|---|---|
| 2026-06-11 | 2단계 산출물 구축 (ADR 초안 + 골든 34파일) |
| 2026-06-11 | 적대 리뷰 반영 — ADR 5건 수정, face_closeup 입력 교체, 하네스 침묵 실패 차단, 변형 2종 추가, 전체 재캡처(37파일) + 결정성 재검증, 본 workPaper 신설 |
