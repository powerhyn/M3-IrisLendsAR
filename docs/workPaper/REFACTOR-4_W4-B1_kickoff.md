# W4-B1 착수 킥오프 — confidence/visibility 계약 정정

> **이 문서 하나 + 아래 필독으로 새 세션이 W4-B1을 바로 착수**할 수 있게 작성. 작성: 2026-06-15.
> W4-B1은 ④ 추적 레이어 전환(REFACTOR-4)의 **단일 최우선 선결**이자 가장 작고 안전한 첫 슬라이스.

## 0. 진입 절차 (그대로)
1. **필독**: 이 문서 → `docs/workPaper/REFACTOR-4_plan.md`(§3 W4-B1, §0 핵심사실) → `docs/decisions/0001-landmark-injection-tracking-replacement.md` §6.2(confidence 경계 제거)·§7(좌표·시맨틱 계약).
2. **브랜치**: `refactor/p33-tracking-ab` (이미 체크아웃, develop 미머지). 최근 커밋 `43305a6`(REFACTOR-4 Codex 반영).
3. **C++ 작업이므로 `systems-programming:cpp-pro` 에이전트 동반** (CLAUDE.md 필수 규칙). 빌드는 **기존 `cpp/cmake-build-debug` 재사용**(`-DIRIS_SDK_FETCH_TFLITE=OFF`, 새 빌드 디렉토리 금지).
4. 게이트0(전환 확정 + ④ 승인)은 사용자가 이 착수로 부여한 것으로 간주.

## 1. 목표 (한 줄)
주입 경로(`iris_set_landmarks`→`deriveIrisResult`→`iris_get_injected_result`)로 들어온 결과가 **렌즈를 정상 렌더**하도록 confidence/visibility 게이팅 계약을 정정한다. **detector(LEGACY) 경로 출력은 비트 불변**(골든 ε 유지)이어야 한다.

## 2. 문제 (코드 확인 완료 2026-06-15)
| 위치 | 현재 상태 |
|---|---|
| `cpp/src/gpu/eye_render_packet_adapter.cpp:89` | `packet.visibility = clamp01(result.confidence * (1 - eyelid_ratio))` — **detector·주입 경로 공유**. `render_confidence = visibility`(:92). |
| `cpp/src/landmark_injection.cpp` deriveIrisResult | `eyelid_ratio_left/right = 0.0`(W3 미구현), `confidence`는 **0.0 유지**(ADR §6.2 "경계에서 제거" 주석). |
| → 결과 | 주입 경로 visibility = **0 × (1−0) = 0** → **렌즈 미렌더**. |
| `cpp/src/mediapipe_detector.cpp:3156` | detector는 `confidence = face_confidence * eye_factor`(실값 ≠ 0) → visibility 정상. **이게 골든 불변 제약.** |
| `android/demo-app/.../tracking/TasksToIrisResult.kt:122` | 데모 글루가 `out.confidence = 1.0f`로 **우회 중**(주석 "코어 임계 게이트 통과 상수"). 단 이 값은 옛 DetectionSlot 경로용 — `iris_set_landmarks`는 confidence를 안 받아 주입 경로엔 안 흐름. |

**모순**: ADR §6.2는 "confidence를 경계에서 제거, 게이팅은 visibility(EAR)로 일원화"를 의도하나, 어댑터 수식은 여전히 `confidence`를 곱한다. 주입은 confidence=0이라 게이트가 닫힌다.

> **⚠️ 데모 경로 사실 (fresh-eyes 게이트 검증, 2026-06-15)**: 현재 데모는 **주입 경로를 안 쓴다** — `setLandmarks`/`getInjectedResult` 호출 0건(android/demo-app 전체 grep). LEGACY·TASKS 둘 다 `updateDetectionSlot`(GpuRenderActivity.kt:1095/1272) → `getDetectionSlotPtr`(CameraGLRenderer.kt:682/735) **DetectionSlot 경로**로 렌더하고, `TasksToIrisResult.kt:122`의 confidence=1.0은 그 DetectionSlot 결과에 흐른다. 주입 경로(`iris_get_injected_result`, confidence=0)는 구현됐으나 **`test_landmark_injection`만 호출 = 데모 휴면**. ⇒ **B1의 confidence 수정 효과는 데모로 확인 불가**(데모는 주입 경로 미사용). 데모를 주입 경로로 전환하는 건 **W4-C(글루 승격)** 작업이지 B1 아님. B1 검증은 아래 §4의 C++ 테스트로 닫는다.

## 3. 수정 방향 (cpp-pro와 택1 — 결정 후 착수)
- **A. deriveIrisResult가 confidence=1.0 설정** (데모 우회를 코어로 승격). 최소 변경·주입 경로 한정(detector 무영향=골든 안전). 단 ADR §6.2 "confidence 제거" 문구와 표면 상충 → ADR 주석/문구 정합 필요.
- **B. 어댑터에서 confidence 곱 제거**, visibility=(1−eyelid)만. ADR §6.2 의도 정합. 단 **공유 수식이라 detector 경로 출력이 바뀜 → 골든 깨짐 위험**(detector confidence가 곱에서 빠지므로). 비권장(골든 불변 위반).
- **C. presence/sentinel 분기** — 주입 결과는 "검출 score 부재" 표식을 두고, 어댑터가 그 경우 visibility=(1−eyelid)로, detector 경우 기존 `confidence×(1−eyelid)` 유지. ADR §6.2 정합 + detector 골든 불변 둘 다 충족. **권고 후보**(설계 비용 ↑).
- 권고: **A 또는 C로 좁혀 시작**(B는 detector 골든 깨짐이라 사실상 배제). C 우선 검토, 설계 비용 과하면 A. 어느 쪽이든 **detector 경로 골든 ε 불변**이 절대 제약. **첫 코드 줄 전에 cpp-pro와 A/C 1회 확정**(조사는 그 전에 착수 가능).

## 4. 게이트/검증
1. **골든(필수)**: `bash scripts/golden_capture_all.sh /tmp/p4b1 && python3 scripts/golden_compare.py --baseline cpp/tests/golden/baseline --candidate /tmp/p4b1` → **exit 0**(detector 경로 ε 일치). 베이스라인 덮어쓰기 금지. (주입 경로는 골든 미경유라, 골든이 잡는 건 detector 무회귀.)
2. **빌드/테스트**: `cd cpp/cmake-build-debug && cmake .. -DIRIS_SDK_FETCH_TFLITE=OFF && cmake --build . --parallel`(신규 경고 0) + `ctest` 회귀 0(pre-existing: TFLite NOT_BUILT 2 + GPUBeautyBackendTest.FailsWithNullContext + FreqSep 2 / 병렬 시 MediaPipeDetectorIntegration 3 플레이크).
3. **주입 렌더 검증(핵심, B1 합격 판정)**: `test_landmark_injection`(cpp/tests/)에 **"주입 결과(iris_get_injected_result)의 visibility > 0" 케이스 추가**해 통과. 이게 B1의 정본 검증 — 데모는 주입 경로 미사용이라(§2 ⚠️) 데모로는 확인 불가. **데모 `TasksToIrisResult.kt:122` confidence=1.0 우회는 건드리지 마라** — 그건 DetectionSlot 경로용이고 제거하면 데모가 깨진다(제거는 데모를 주입 경로로 옮기는 W4-C 몫).
4. `./gradlew :demo-app:assembleDebug` 성공 + 실기기 fsync ON 렌즈 정합 유지 — 단 이건 **DetectionSlot 경로(데모) 무회귀** 확인이지 주입 수정 검증 아님(데모 무영향 확인용).

## 5. 워킹트리 주의 (커밋 제외 — 변하지 않음)
다음은 의도적으로 커밋 제외 상태로 워킹트리에 남아 있음(중단 잔여물 아님): `.claude/settings.local.json`, `android/.../env/env_default_256x128.png`(LFS 팬텀 — 스테이징 금지), `cpp/include/iris_sdk/export.h`(CMake 생성물), `docs/lenssim-handoff/{jaw-vline,natural-lens}*.md`(타 트랙). **W4-B1 커밋 시 이들 staging 금지** — 변경한 cpp 파일만.

## 6. 완료 후
`docs/workPaper/REFACTOR-4_plan.md` W4-B1 상태 갱신 + 분리 커밋(`feat(core): ④ W4-B1 — 주입 경로 visibility 게이팅 정정`). 다음은 W4-B2(layout/API 단일화).

## 참고 — 이 트랙 전체 맥락
③-3(추적 A/B 인프라)+frame-sync(트래킹 지연 해결, 커밋 f86b66f) 완료, develop 미머지. ④ 스코핑→Codex 검증→이 계획. confidence=0 함정은 Codex 검증이 발견. 자세한 트랙 상태는 메모리 `refactor-p2-adr-golden` 참조.
