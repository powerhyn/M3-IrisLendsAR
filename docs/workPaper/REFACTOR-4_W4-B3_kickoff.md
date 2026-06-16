# W4-B3 착수 킥오프 — JNI/Java/Kotlin result 정합 + DetectionSlot 원자성

> 이 문서 + 아래 필독으로 **대화 맥락 0인 새 세션이 W4-B3를 바로 착수**할 수 있게 작성. 작성: 2026-06-16.
> W4-B3는 ④ 추적 레이어 전환(REFACTOR-4)의 한 슬라이스. **W4-A/B1/B2 완료 후 다음**.

## 0. 진입 절차 (그대로)
1. **필독**: 이 문서 → `docs/workPaper/REFACTOR-4_plan.md`(§3 W4-B3 = line 41, §0 핵심사실, W4-A/B1/B2 완료 기록) → `docs/decisions/0001-landmark-injection-tracking-replacement.md` §6.3(IrisResult 강등)·§7.3(좌표 left/right 라벨)·§7.1(좌표 공간) → **`docs/workPaper/REFACTOR-3-3_tracking-latency-followups.md` §3**(="ts ↔ DetectionSlot 번들링 — 1프레임 스큐 제거 (④ 범위)" = "후속문서 §3"의 실체; `GpuRenderActivity.kt:1080·1196`·`CameraGLRenderer.kt` file:line 포함). ⚠️ **경로 주의**: `docs/lenssim-handoff/tracking-latency-handoff-from-lenssimulator.md`(이름 유사·다른 문서, §3=해결책)와 혼동 금지 — 후속문서는 **workPaper** 쪽이다.
2. **브랜치**: `refactor/p33-tracking-ab` (이미 체크아웃, develop 미머지). HEAD = `be10d9b`(W4-A 에러코드 완료).
3. **C++ 작업이므로 `systems-programming:cpp-pro` 에이전트 동반**(CLAUDE.md 필수). JNI(C++)+Java+Kotlin 다중 레이어라 surface 일관성 점검 필수. 빌드는 **기존 `cpp/cmake-build-debug` 재사용**(`cmake --build . --parallel -- -k 0` keep-going으로 pre-existing TFLite 2타깃 우회, 새 빌드 디렉토리 금지).
4. **착수 패턴(W4-B2/W4-A에서 검증됨)**: plan 기재가 실제 코드와 다를 수 있음. **먼저 Workflow 다축 조사**(현황·ABI/동작불변 제약·안전 슬라이스)로 규명 → cpp-pro와 설계 확정 → 구현 → 게이트. 무작정 plan대로 지우지 말 것.

## 1. 목표 (한 줄) + 왜 지금
W4-B3 = **JNI/Java/Kotlin IrisResult 표현 정합 + DetectionSlot 정식 재설계(ts-슬롯 원자 번들링)**. 후자가 핵심 — **W4-C(글루 AAR 승격) A/B 검증의 선결**이다(plan §3: 1프레임 스큐가 남으면 A/B가 '추적기 차이'와 '동기화 버그'를 분리 판정 못 함, Codex critical).

## 2. 현재 상태 — 검증된 사실 (2026-06-16 코드/git로 확인)
- **HEAD `be10d9b`**. ④ 진척: W4-B1(af07e04, confidence)·폴백결정(29d080f)·W4-B2(022ab4d, 축소)·W4-A(776302f internal헤더 + be10d9b 에러코드) 완료. W4-A 기본값 단일소스화는 **P8 뷰티 트랙 이월**(메모리 [[refactor-p2-adr-golden]] 참조).
- **인접 사실 (W4-A/B2 조사에서 파악 — 착수 시 재확인 필요, file:line 스테일 가능)**:
  - IrisResult **C/C++ offsetof 21필드 + sizeof static_assert 완비** = `cpp/src/sdk_api_v2.cpp:44-83`(W4-B2에서 sizeof를 GLES `#ifdef` 밖으로 이동). 즉 C/C++ 2벌은 이미 레이아웃 가드됨.
  - detector 메타 `iris_quality_*`/`eye_refiner_used`는 **W4-D 이월 @deprecated 주석 부착됨**(types.h/sdk_api.h, W4-B2). **W4-B3에서 이 필드 물리 제거 금지**(W4-D에서 골든 재캡처와 원자 삭제). IrisResultKt 정리도 "메타 잔존 인지·문서화"까지이지 물리 삭제는 W4-D.
  - JNI IrisResult→Java 필드매핑은 `android/iris-sdk/src/main/cpp/iris_jni.cpp`(W4-B1 조사에서 `copyResultToJava`/`irisResult_*` 필드ID 캐시 :272 부근 확인). 필드별 Get/SetField(memcpy 아님)라 레이아웃 무의존.
  - DetectionSlot: ③-3 frame-sync(커밋 f86b66f)가 RGBA 링버퍼로 픽셀-랜드마크 센서ts 매칭. **ts/슬롯이 별채널이라 1프레임 스큐 잔존**(메모리 기록 + 후속문서 §3). W4-B3가 이걸 원자 번들로 해소.
- **plan §3 W4-B3 원문**(line 41) = 4개 하위: ① Java/Kotlin 2벌 → JNI 필드매핑 생성/검증, ② **IrisResultKt 정리**(detector 메타 잔존 + avg_luma/faceMesh 부재, ADR §47-53 removalScope), ③ faceRect 좌표 단위 문서 드리프트 정정(Java=pixel vs 데모=normalized), ④ **DetectionSlot 정식 재설계 + ts-슬롯 원자 번들링**(핵심, W4-C 선결).

## 3. 할 일 / 결정 / ⚠️ 절대 제약
**착수 1단계 = 조사**(W4-B2/W4-A 패턴). Workflow 다축으로: (a) IrisResult Java/Kotlin 2벌 + JNI 매핑 현황·드리프트, (b) IrisResultKt 정리 범위(메타 잔존/avg_luma·faceMesh 부재) + ABI 영향, (c) faceRect 단위 드리프트(Java pixel vs 데모 normalized) 실측, (d) **DetectionSlot 재설계 + ts 원자 번들** 현 구조(frame-sync 경로)·스큐 원인·W4-C 선결 요건. 각 축에 "동작 불변으로 안전한 것 vs ABI/W4-D 이월" 분리(W4-B2 교훈).

**⚠️ 절대 제약(불변식)**:
1. **detector 경로 골든 ε 불변** — 어떤 변경도 `result.json`(detector 출력) 비트 불변(§4 게이트). 추적/JNI는 surface지만 골든 안 깨야.
2. **ABI 보존** — IrisResult는 **C++(types.h)/C(sdk_api.h)/Java/JNI 4면 미러**. 필드 추가/삭제/순서변경 = ABI break → W4-B2 교훈대로 1.x 금지(deprecated/no-op), 물리 변경은 2.0/W4-D. faceRect 단위 정정은 "문서/주석" 한정(런타임 값 변경 = 동작 변경이라 분리).
3. **detector 메타(iris_quality_*/eye_refiner_used) 물리 제거 금지** — W4-D 이월 확정(이미 @deprecated). W4-B3는 "정리·문서화"까지.
4. **DetectionSlot 재설계는 데모 frame-sync(f86b66f) 무회귀** — 실기기 fsync ON 렌즈 정합 유지(③-3 성과). ts 원자 번들이 거울 지연 trade-off를 악화시키지 않아야.
5. **surface 변경(에러코드 류)은 '의도된 정정'으로 분리 기록** — W4-B1 confidence/W4-A 에러코드 선례.

## 4. 게이트/검증 (W4-A에서 실행 검증된 명령)
1. **빌드**: `cd cpp/cmake-build-debug && cmake --build . --parallel -- -k 0` → 신규 error/warning 0. **FAILED는 pre-existing 2개만**(`test_mediapipe_detector`, `test_mediapipe_detector_performance` — TFLite NOT_BUILT 링크 실패). 그 외 FAILED = 회귀.
2. **골든(필수)**: `bash scripts/golden_capture_all.sh /tmp/p4b3 && python3 scripts/golden_compare.py --baseline cpp/tests/golden/baseline --candidate /tmp/p4b3` → **"결과: PASS — 불일치 0건"**. 베이스라인 덮어쓰기 금지.
3. **ctest**: `ctest -j1` → **회귀 0**. pre-existing 5건만 허용: NOT_BUILT 2(#59/60) + `GPUBeautyBackendTest.FailsWithNullContext`(#531, GPU컨텍스트 부재 376/377줄·에러코드 무관) + `FreqSepMappingTest`(#698/699, skinQuality stale·뷰티 트랙). `ctest --rerun-failed -N`로 실행실패 3건(531/698/699) 확인.
4. **assembleDebug(JNI/Java/Kotlin 변경 시 필수)**: `cd android && ./gradlew :demo-app:assembleDebug` → BUILD SUCCESSFUL(exit 0). W4-B3는 JNI/Java/Kotlin을 직접 건드리므로 이 게이트가 핵심.
5. **DetectionSlot 재설계 시**: 실기기 fsync ON 렌즈 정합 무회귀(사용자 육안) — 데모 frame-sync 경로 보존 확인.

## 5. 워킹트리 주의 (커밋 제외 — 변하지 않음, 정상 잔존)
다음은 의도적 커밋 제외(중단 잔여물 아님): `.claude/settings.local.json`, `android/.../env/env_default_256x128.png`(LFS 팬텀 — 스테이징 금지), `cpp/include/iris_sdk/export.h`(CMake 생성물), `docs/lenssim-handoff/{jaw-vline-warp,natural-lens-fit}*.md`(타 트랙). **W4-B3 커밋 시 이들 staging 금지** — 변경한 파일만 명시적 `git add`. (이 kickoff 문서 `docs/workPaper/REFACTOR-4_W4-B3_kickoff.md`는 위 제외분과 달리 **이번 인계로 커밋됨** — 워킹트리에 미커밋으로 남아있지 않으니 중단 잔여물로 오해 말 것.)

## 6. 완료 후
`docs/workPaper/REFACTOR-4_plan.md` W4-B3 상태 갱신 + 분리 커밋(`refactor(core): ④ W4-B3 — ...`). 메모리 [[refactor-p2-adr-golden]] 진입점 갱신. 다음 = **W4-C**(글루 iris-sdk AAR 승격 — B3 ts 원자성이 선결). 그 후 W4-D(추적 코어 제거 + enum/메타 물리삭제 + 좌표 canonical) → W4-E(cpu-render deprecation + 16KB). **④ 완료 후 P8 뷰티**(핵심=피부 skin smoothing + 턱깎기 형태워프, 곁가지 제거 — 메모리 [[p8-facemesh478-substrate]]).

## 참고 — 트랙 전체 맥락
③-3(A/B 인프라)+frame-sync 완료, develop 미머지. ④ W4-A/B1/B2 완료. 검출 폴백=글루 결정(ADR §3). 자세한 트랙 상태·P8 뷰티 방향은 메모리 [[refactor-p2-adr-golden]] 참조. 작업 패턴: 조사 Workflow → cpp-pro → 게이트(골든/ctest/assembleDebug).
