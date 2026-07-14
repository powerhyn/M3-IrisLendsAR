# ⚠️ 폐기 (2026-07-08) — 현행 진입점은 `NLR-W2_resume_kickoff.md`

> W1은 종결됨(cap sweep 무효 판정 + 문제 재정의 — `docs/bench/P7-W4/cap_sweep_result.md`).
> 아래 내용은 이력 보존용이며 file:line·상태 서술이 스테일함. **새 세션은 이 문서로 착수 금지.**

# NLR-W1 재개 킥오프 — cap sweep 벤치 → 판정 → W2 진입 (폐기본)

> 이 문서 + 아래 필독으로 새 세션이 대화 맥락 0에서 착수. 작성: 2026-07-02 (구현 완료·일시 중단 시점).
> 재개 세션은 **ultracode 모드**로 진행 예정 (사용자 지시).

## 0. 진입 절차

- **필독 순서**: 이 문서 → `docs/workPaper/NLR-W0_index.md`(트랙 전체 지도) → `docs/workPaper/P7-W4_sclera_luma_attenuation.md` §4.1(DoD)·§5.8(수식 합의). W2 진입 시 → `docs/plans/iridescent-dreaming-neumann.md`(계획서 정본, NLR-W2 절 + 자료조사 6주제).
- **브랜치**: `feature/P7-W4-sclera-luma-atten` (develop `075a5de` 위 6커밋, **develop 미머지**). HEAD `2a64c32`. 구현 커밋 4개: `e354bd8`(cpp) → `edd61ea`(JNI/Java) → `f5b1a97`(demo+복원보수) → `2a64c32`(docs).
- **빌드**: ⚠️ 새 빌드 디렉토리 생성 금지 — `cpp/cmake-build-debug` 재사용(`-DIRIS_SDK_FETCH_TFLITE=OFF`). APK는 `./scripts/build_and_install.sh`(versionCode 자동 증가).
- **실기기**: S23+ `ANDROID_SERIAL="adb-R3CW20BBFCM-HZeo2S._adb-tls-connect._tcp"`. **b282(versionCode=282, cap 배선 포함)가 이미 설치됨** — 재빌드 불필요, 바로 벤치 가능.
- **동반 에이전트**: `cpp/` 수정 시 `systems-programming:cpp-pro` 필수(CLAUDE.md). W 구현 워크플로는 `ar-lens-implement` 스킬.
- **Codex 교차 검증**: 이 트랙은 **Codex 단독**(Gemini 제외, 사용자 확정). 직전 세션 확인 타깃 = tmux **`20_CGG-Backend:3.1`**(codex CLI, IrisLensSDK 루트에서 기동). ⚠️ 타 프로젝트 codex 팬(`30_Aims-Frontend:7.1`)과 혼동 금지 — **재개 시 `tmux list-panes -a -F '#{session_name}:#{window_index}.#{pane_index} #{pane_current_command}' | grep codex`로 재확인 필수**(팬 배치는 세션 간 가변). 송신 규칙: 텍스트와 Enter 분리(`tmux send-keys -t <팬> -l '...'` → sleep 0.5 → `tmux send-keys -t <팬> Enter`).

## 1. 목표 (한 줄) + 왜 지금 이것

NLR-W1(흰자 빛남 비율 cap)의 **실기기 cap sweep 벤치 → 판정 → 종결(develop 머지)**, 이후 NLR-W2(tone_class) 진입. 구현·빌드·설치는 끝났고 **육안 벤치만 남음** — cap 확정값이 이후 모든 W의 벤치 베이스라인이라 W2보다 선행 필수.

## 2. 현재 상태 — 검증된 사실 (2026-07-02 코드/기기 직접 확인)

- `cpp/src/gpu/shader_sources.cpp:351` — `float tintMul = min(lum * scale, uScleraTintMax);` (blendTintLinearV2 내부, uniform 선언 :276)
- `cpp/src/gpu/gpu_lens_renderer.cpp:1257` — uniform 업로드. setter clamp [1.0, 1e6]. 멤버 기본 `sclera_tint_max_ = 1.275f`
- 데모 `GpuRenderActivity.kt:507` — `btnW4Cap` 사이클 리스너, sweep `{1.275, 1.5, 2.0, 1e6(OFF)}`(:541), 초기 인덱스 0
- `GpuRenderActivity.kt:669-687` — `restoreLensRenderState()`에 cap + 기존 갭 5종(gate/blinkUp/vetoMode/scleraProtect/contactShadow) 복원 등재 완료
- S23+ 설치 상태: `versionCode=282` (adb dumpsys로 확인) — cap 기본 1.275 즉시 활성 상태
- **재베이스라인 판정(§4.1 1항 완료)**: SKU 6(claset 누드 애쉬 로제) 빛남 **lum:meas/lum:fb 모두 ②(거슬림)** → 구현 진행 확정. 기록: `docs/bench/P7-W4/rebaseline_checklist.md` 결과 기입란
- ⭐ **신규 관측(이월)**: 사용자가 **lum:fb를 선호**(meas보다 렌즈 패턴이 더 잘 느껴짐) — W2 실측 교정이 패턴 가시성을 약화시킨 것으로 추정. **W1 범위 아님**(cap은 홍채 평균 대비 ~1.5배 이상 밝은 픽셀만 자름 — 패턴 강도와 독립). 틴트 강도/정규화 상수 재검토는 **NLR-W2/W6 벤치 축**으로 명시 이월됨
- 재베이스라인의 조명 조건 라벨(일반 실내/밝은 조명)은 미확정 — 사용자 미응답. cap sweep 벤치에서 두 조건 모두 재확인하면 됨

## 3. 할 일 / 결정 — 순서 + ⚠️절대 제약

1. **cap sweep 실기기 벤치** (사용자 육안, 유일한 잔여 DoD 2건):
   - 앱 실행 → 렌즈 선택(`Lens applied` 로그 확인) → ⚙ 개발 패널 → `cap1.28` 버튼으로 sweep {1.275→1.5→2.0→OFF}
   - SKU 순서: 6 누드 애쉬 로제 → 2 런웨이 그레이 → 5 샤모 그래픽 → 1 돌 초코(대조군). 조명 2조건(일반 실내 + 밝은 조명)
   - 앵커 질문 ①: 빛남이 재베이스라인(OFF와 동일) 대비 ~50%+ 감소 체감? ②: **sclera 경계 평탄화/하이라이트 칙칙함**이 거슬리는가? (§5.8 Gemini 소수 의견 검증 항목 — cap 도달 영역 틴트 평탄화 리스크)
2. **판정 분기** (§4.1/§5.8):
   - 만족 → cap 확정값 선택(1.275/1.5/2.0 중) → 종결: DoD 체크 + W 문서 §8 이력 + `NLR-W0_index.md` 상태 갱신 + **develop 머지(`--no-ff`, PR 생략 — 내부 통합 관례)** + origin push
   - 경계 평탄화 거슬림 → **soft-knee fallback** 진입(§5.8: cap 위에 smoothstep 절충 — 구 P7-W0 안 강등분)
   - 빛남 감소 부족 → **P7-Spike-A(Oklab+Laplacian PoC) 진입점 명시** 후 사용자 결정(§5.2)
3. **W1 종결 후 → NLR-W2 진입** (계획서 NLR-W2 절): deep-research 갭 6주제 → Codex R1 → 파서 확장(+단위 테스트) → JSON 확장 순서 강제. 계획서의 "사전 검증 결과" 절(auto 센티널 3중 차단, effective alpha 이중 감쇄 금지, dead 슬라이더)이 W2 구현의 필수 전제.

⚠️ **절대 제약(불변식)**:
- 셰이더에 분기/조건부 texture fetch 추가 금지 (§8.9 Adreno implicit-LOD 사고 이력 2회)
- cap은 ID=5(TintLinearV2) 단독 적용 (§5.12) — 타 블렌드 모드 확장 금지
- `uScleraVetoMode` 기본값(legacy 0) 불변 — B8 판정은 W4/W1과 분리(§5.10)
- 셰이더 절차 림발 금지(에셋 책임) · 저조도 범위 외 · 최종 사용자 수동 블렌드 UI 금지 (트랙 하드 제약, `NLR-W0_index.md` §4)
- 벤치는 정량 점수화 지양 — 1인 실시간 토글 육안 체감(⓪~③ 스케일)

## 4. 게이트/검증 — 명령 → 기대값

- 설치 확인: `ANDROID_SERIAL="adb-R3CW20BBFCM-HZeo2S._adb-tls-connect._tcp" adb shell dumpsys package com.irislenssdk.demo | grep versionCode` → `versionCode=282` 이상
- 앱 실행: `adb shell am start -n com.irislenssdk.demo/.GpuRenderActivity` (같은 SERIAL)
- cap 토글 동작 로그: `adb logcat -s GpuRenderActivity | grep "P7-W4 sclera tint cap"` → 버튼 탭마다 `→ 1.5` 등 출력
- cpp 재빌드 필요 시: `cd cpp/cmake-build-debug && cmake --build . --parallel --target iris_sdk` → `Linking CXX static library lib/libiris_sdkd.a` (경고는 temporal_stabilizer 기존분만 정상)

## 5. 워킹트리 주의 — 의도적 제외분 (정상 잔존, 중단 잔여물 아님)

- `.claude/settings.local.json` (M): 로컬 권한 설정 누적 — 커밋 금지
- `docs/plans/bubbly-prancing-boot.md` (untracked): 이전 데모 UI 재구성 플랜 — 본 트랙 무관, 방치 정상
- `android/demo-app/src/main/assets/env/env_default_256x128.png`: **skip-worktree 플래그 설정됨**(`git ls-files -v | grep ^S`로 확인 가능). HEAD에 raw PNG가 커밋됐는데 `.gitattributes:14`의 `*.png` LFS 규칙이 clean 필터로 포인터 변환을 시도해 **영구 modified로 보이는 아티팩트**(파일 내용은 5222B 무손상). rebase/checkout 차단 회피용 로컬 우회 — LFS 정규화 커밋은 별도 판단(환경 반사 재개 시점 권장)

## 6. 완료 후 — 문서/커밋 규칙, 다음 단계

- W1 종결 시 갱신: `P7-W4_sclera_luma_attenuation.md`(§4.1 잔여 2항 + §8 이력 + 상태), `NLR-W0_index.md`(W1 행), `docs/bench/P7-W4/`에 cap sweep 결과 기록(체크리스트 서식 재사용)
- 커밋: Conventional Commits 접두사 영문 + 설명 한글 + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. W 단위 분할 커밋 → develop 직접 머지(`--no-ff`, PR 생략)
- 다음: NLR-W2 — 계획서 NLR-W2 절 + 자료조사 6주제부터. Codex R1은 팬 재확인 후 진행
