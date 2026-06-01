# P6-W9 통합 리포트

**일시**: 2026-06-01
**브랜치**: `feature/P6-Works`
**대상 머지**: `feature/P6-Works → develop` (옵션 A `--no-ff` 머지 커밋)
**작성**: Claude Opus 4.7 (사용자 실기기 결과 대기 자리 포함)

---

## 1. Phase 6 전체 종결 상태

### 1.1 살아남은 트랙 (develop 머지 대상)

| W | 산출물 | 머지 SHA |
|---|---|---|
| W1 | EyeRenderPacket 계약 + fallback chain | `9afe095`, `c36d883`, `da15753` (PR #1) |
| W2 | 블렌드 3종 + realSpec 폐기 + LUMA_709 통일 | `94546a2` |
| W5 Phase A | uScleraVetoMode 3-way + API + A/B/C/D 토글 UI | `2d594e0`, `e2f6b6a`, `34f0373`, 머지 `abc9a84` |
| W5 Phase B (1차) | 실제 타겟 42종 lens-ar 도입 + 1차 형광 벤치 결과 기록 | `11fefb0`, `5a12f38`, `eb0049b`, `3173232`, Codex 리뷰 반영 `0fc024f`+머지 `5dc67ec` |
| W6 Phase A | 블링크 ramp (C7) + C10 디테일 재주입 + B9 gate (기본 0.10) + 토글 노출 | `5e48cf6`, `14aac99`, `5fa5fe0`, `97fff27`, 머지 `07d42a0` |
| W7 (종결) | sku_id/LensSkuMetadata + JSON 파서 + B4 자동감지 인프라(메타용 보존) + 셰이더 림발 영구 제거 | `2af328d`, `c193576`, `48cc680`, `083c795`, `de1eeb7`, `8bdf825`, 머지 `9cbf05a` |
| 기타 | 레거시 CPU 경로(MainActivity) 제거 | `1d4b19c` |

### 1.2 Phase 6 이월 트랙 (develop 머지에 포함되지만 OFF 기본)

| W | 보존 상태 | 재개 진입점 |
|---|---|---|
| W3 환경 반사 scaffold | OFF 기본, 토글 가능, `IRIS_SDK_RENDER_MASK_HOOK_ENABLED` 보존 | 메모리 `w4-env-reflection-deferred` |
| W4 B2 환경 반사 벤치 | Phase A 코드 + 벤치 인프라(`docs/bench/P6-W4/`) 보존, 24클립 미실행 | `P6-W4_*.md` §1.16~1.17 |
| W8 Pupil material restore | 자동 폐기 (W4 종속). 본문 Option E 설계 보존 | `P6-W8_*.md` 본문 |

### 1.3 W9 단계 신규 발견 / 후속 분리

| 후속 W (Phase 6 외) | 사유 |
|---|---|
| **W5 Phase C — TintLinearV2 흰자 빛남 수식 개선** | 1차 형광 벤치에서 발견. 별도 W로 분리 (메모리 `w5-b1-tintlinearv2-strength`) |
| **W6 Phase B/C — 실측 source 연결 + ramp/gate 최종 튜닝** | avg_iris_luma fallback만 동작 중. 실측 연결 후 vault (메모리 `w6-avg-iris-luma-measure`) |
| **W9 데모 UI/KT 동기화** | W9 머지 후 별도 cleanup PR (메모리 `w9-demo-ui-sync`) |
| **deprecated no-op 제거** | Phase 7 초반 별도 PR (W9 doc §5.6) |

---

## 2. CI / 빌드 확인

### 2.1 C++ 빌드 (cpp/cmake-build-debug)

- **Target `iris_sdk`**: ✅ 빌드 성공 (8 unused-const-variable warning — `mediapipe_detector.cpp`, Phase 6 도입 코드 아님)
- **ctest**: 99% (745/752 passed, 64.49s). 실패 7건은 모두 **Phase 6 영역 밖 기존 모듈** — 회귀 신호 아님:
  - `test_mediapipe_detector_NOT_BUILT` ×2 (빌드 미생성 placeholder, Not Run)
  - `FrameProcessorTest.InitializeWithInvalid/EmptyPath` ×2 (P1-W4-03 FrameProcessor 영역, Subprocess aborted — assertion abort 동작으로 보임)
  - `GPUBeautyBackendTest.FailsWithNullContext` (P2-W3-01 Beauty 백엔드 영역)
  - `FreqSepMappingTest.RadiusScalesWithFaceWidth` / `MinimumRadiusGuard` (Beauty FreqSep 영역)
  - Phase 6 GPU lens 렌더링/EyeRenderPacket/블렌드/W7 메타 등 본 작업이 건드린 영역 테스트는 통과. 위 7건은 develop 기존 상태에서도 있을 가능성 높음 (별도 추적 권장).

### 2.2 Android demo

- **`./gradlew :iris-sdk:assembleDebug`**: ✅ BUILD SUCCESSFUL (5s, 29 actionable tasks)
- versionCode: 284 (실기기 빌드 카운터)

### 2.3 미실행 CI 항목 (의도적 후속 이월)

- LUMA 계수 shader vs CPU 오차 ≤1% 테스트 케이스 — W6 후속 Phase에서 실측 source 연결과 함께
- valgrind/sanitizer 메모리 누수 — Android 실기기 long-run 관찰로 대체 (§3 참조)

---

## 3. 실기기 통합 회귀 (사용자 진행)

**가이드**: `docs/bench/P6-W9/checklist.md` — 6 SKU × 3 tier = 18 시나리오, 5축(양안/블링크/sclera/블렌드+디테일/림발).

> ⚠️ 환경 반사(W3/W4)·Pupil(W8) 시나리오는 폐기.

### 3.1 결과 자리 (사용자 채움)

```
시나리오 결과 표:
| SKU | Tier | A | B | C | D | E | FPS | 메모 |
|...

FAIL 항목:
- (없음 / 또는 항목 + 후속 처리)

메모리:
- 10분 long-run delta: ___ MB
- 누수 신호: 없음 / 있음

머지 판정:
- ✅ 머지 가능 / ⚠️ 보류 (사유)
```

---

## 4. develop 머지 직전 상태

### 4.1 브랜치 분기 확인 (2026-06-01)

- `git merge-base origin/develop feature/P6-Works` = `fff737f`
- `feature/P6-Works`는 `origin/develop`을 fully ancestor로 포함
- develop ahead of P6-Works: **0 commits**
- P6-Works ahead of develop: **74 commits**
- → 옵션 A `--no-ff` 머지 시 충돌 없음 (W9 doc §1.0.4 확인)

### 4.2 머지 메시지 (확정)

```
Merge branch 'feature/P6-Works' into develop

P6 통합 — W1/W2/W5 Phase A·B/W6 Phase A/W7 완료.
W3·W4·W8 Phase 6 이월(보존 코드 + 재개 진입점 기록).

상세:
- W1: EyeRenderPacket 계약 + fallback (실측 source는 W6 이관)
- W2: 블렌드 3종 + realSpec 폐기 + LUMA_709
- W3: 환경 반사 scaffold (OFF 기본 보존)
- W4: B2 벤치 Phase 6 이월
- W5: B1/B8 Phase A/B 완료, Phase C 별도 W로 분리
- W6: 블링크 ramp + 저조도 gate + 디테일 재주입 (Phase A)
- W7: 셰이더 림발 영구 제거, 메타 인프라(sku_id/registry) 보존
- W8: 자동 폐기 (W4 이월 연쇄)

참조: docs/workPaper/P6-W9_integration_report.md
```

### 4.3 머지 전 사용자 최종 승인 필요 항목

1. 실기기 5축 결과 PASS (§3.1)
2. ctest / `./gradlew :iris-sdk:assembleDebug` 성공 확인 (§2)
3. 머지 메시지 (§4.2) 그대로 vs 수정

---

## 5. Phase 7+ 이월 항목 목록 (99 §4 + W9 신규 추가)

원안 (R3 합의):
- 3D Face Geometry / HDR IBL / Full PBR / Neural rendering / Corneal refraction / 속눈썹 전용 세그멘테이션 / Head-pose 기반 env 회전

W9 단계 추가:
- W4 환경 반사 트랙 (B2 24클립 벤치 미실행, scaffold 보존)
- W8 Pupil material restore (W4 종속)
- W5 Phase C (흰자 빛남 수식 개선)
- W6 Phase B/C (실측 source 연결 + 최종 튜닝)
- W9 데모 UI/KT 동기화 (별도 cleanup PR)
- deprecated no-op 제거 (`setHighlightEnabled` 등)
- LUMA 계수 shader vs CPU 오차 테스트 케이스

---

## 6. 메모리 갱신 권장

W9 종료 후 다음 메모리 갱신/신규 추가 권장:

- 신규: Phase 6 종결 + develop 머지 SHA 기록 (project)
- 갱신: `w9-demo-ui-sync` — W9 머지 후 cleanup PR 진입점
- 갱신: `w6-avg-iris-luma-measure` — 후속 W에서 실측 연결 명시

---

## 7. 참조

- W9 doc: `docs/workPaper/P6-W9_integration.md` §1.0 (현 상태 변화 반영)
- 99 doc: `docs/workPaper/P5-W3-05_brainstorm/99_final_decision.md` §10 (Phase 6 종결 후속 처리)
- 체크리스트: `docs/bench/P6-W9/checklist.md`
- 핸드오프: `docs/workPaper/P6_implementation_handoff.md` §7 (CI 체크리스트 원본)
- 각 W: `docs/workPaper/P6-W{1..8}_*.md`
