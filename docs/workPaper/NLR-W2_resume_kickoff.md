# NLR-W2 재개 킥오프 — 잔여 판정 6건 → canonical 조립 → 정식 구현

> 이 문서 + 아래 필독으로 새 세션이 대화 맥락 0에서 착수. 작성: 2026-07-08 (실기기 반복 벤치 중단 시점).
> 구 킥오프 `NLR-W1_resume_kickoff.md`는 **폐기됨** (W1 종결 — cap 무효 + 문제 재정의로 완료).

## 0. 진입 절차

- **필독 순서**: 이 문서 → `docs/bench/NLR-W2/formula_bench_notes.md`(**라운드별 판정 전체 기록 — 가장 중요**) → `docs/workPaper/NLR-W2_blend_formula_research.md`(검증 리서치 + 정정 주석) → `docs/workPaper/NLR-W2_brainstorm/codex_r1.md`(Codex 수식 검토). 트래킹 축 배경: `docs/lenssim-handoff/mediapipe-internal-smoothing-bypass-handoff-from-lenssimulator.md`.
- **브랜치**: `feature/P7-W4-sclera-luma-atten` (develop `075a5de` 위, **미머지**). 최근 커밋: `fcf01f2`(벤치 인프라 R5~R7+트래킹 A/B), `60a5df5`(수식 후보 1차), `d506051`(리서치), `839c9bc`(W1 종결).
- **기기**: S23+ `ANDROID_SERIAL="adb-R3CW20BBFCM-HZeo2S._adb-tls-connect._tcp"`, **b301 설치됨**(모든 토글 + 트래킹 스윕 R3 프리셋 사이클: norm/2/150a/2/150c/1/150c). 무선 adb는 폰 화면 꺼지면 세션 만료 — 재연결 후 `adb devices`에 S916N 보이는지 확인. 빌드는 `./scripts/build_and_install.sh`(⚠️ cpp 빌드는 `cpp/cmake-build-debug` 재사용, 새 디렉토리 금지).
- **동반 에이전트**: `cpp/` 수정은 `systems-programming:cpp-pro`. Codex 교차는 **Codex 단독**(tmux `20_CGG-Backend:3.1` — 재개 시 `tmux list-panes -a ... | grep codex`로 재확인, 세션 죽어있으면 `codex --dangerously-bypass-approvals-and-sandbox`로 기동 후 텍스트/Enter 분리 송신).

## 1. 목표 + 왜 지금

"보편 최적 렌더링" canonical 확정 직전 단계. 벤치 라운드로 **원인과 처방이 전부 분해·검증됨** — 남은 건 사용자 판정 6건 수집 → canonical 조립 → Codex R2 → 정식 구현.

## 2. 현재 상태 — 검증된 사실 (2026-07-08 코드 확인, 커밋 fcf01f2)

### 벤치 슬롯 (데모 ⚙ 패널 A/B/C/D + 임시 ID 3/4/6 재배선 — `shader_sources.cpp` 디스패치 :705-745)
- **A (ID5)** = 현행 canonical TintLinearV2 (`scale=clamp(0.85/avgLum, 0.8, 7.0)` — 상한 7.0이 빛남 병인)
- **B (ID3)** = TintLinear **고정 K=4.2** + 슬라이더=채도 부스트(:713-728, satBoost 1.0~2.2 휘도 보존)
- **C (ID4)** = 기하 디버그(:729-739): 초록<f / 빨강 f~f+0.2 / 파랑>f+0.2 + **cap≤1.2면 보라 혼색**(:736 — cap 배선 진단)
- **D (ID6)** = V2 + 흰자 조기 페이드(`blendTintLinearRadial`, 창=uFadeStart~+0.2)
- 슬라이더("밝기", 렌즈 탭, f0.8~1.4) = **uFadeStart 겸용**: B에선 채도, C/D에선 페이드 창. cap 버튼 sweep {0.95, 1.05, 1.15, off} (setter clamp [0.85, 1e6])
- 트래킹 토글: **img**(W6 행 끝) = FaceTracker RunningMode.IMAGE (`FaceTracker.kt:91 setImageMode`, 분석 스레드 재생성) / **stab**(fsync 옆) = 코어 stabilizer near-raw 프리셋 (`createStabilizerFast` — `iris_jni.cpp:1789`, 3.0/200)

### 확정된 판정 (라운드 기록은 bench_notes 참조)
1. **W1 종결**: cap(1.275+) 무효 — visibility budget 누락. 문제 재정의 = "휘도 비례 틴트의 경계 톤 불연속" (사용자).
2. **리서치**(21클레임 검증): 휘도 곱셈 주입은 상용 선례 없음. 3계열 대안 → **전부 스티커 판정** (질감 전달 부족).
3. **곱셈 골격은 유지가 정답** (A가 홍채 자연도 최고) — 병인은 **적응 증폭의 상한 7.0**. **빛남 임계 K≈4.4 실측** (B 스윕 f1.1) → canonical = scale 상한 4.0~4.2로 인하.
4. 기하 디버그로 **검출 홍채 반경 > 실제** 확정 (빛나는 링이 초록 존까지 침범) → 반경 보정 계수 필요(트래킹 레벨). 서클렌즈 의도 겹침(착색부/홍채 ≈ 1.1~1.2)은 완벽 트래킹에도 존재 — 실물은 외곽 최암 잉크(US6,827,440 검증)라 얌전함.
5. **트래킹**: img 효과 미미 = numFaces=2 우회 실증. **stab:fast(3.0/200) 효과 큼** = LensSim 델타 주범은 코어 필터 보수 튜닝(4.0/15). 환산 근거: LensSim 픽셀 공간 beta 0.3 × 분석폭 640 ≈ 200. → **후속 스윕 R1~R3(b299~b301)로 ③④ 종결: canonical = iris 중심 1.0/150(중심만)** — bench_notes §트래킹 인사이트 참조.

### ⚠️ 남은 판정 — 렌더 4건 잔여 (①②⑤⑥; 트래킹 ③④는 2026-07-09 종결)
| # | 판정 | 방법 |
|---|---|---|
| ① | R7 채도 부스트 최적값 + 투명도 조합 | B + 슬라이더 스윕 (+ 투명도 75~85%) |
| ② | cap 배선 여부 | C에서 cap 버튼 → 밴드 보라 변화 유무 |
| ③ | ✅ **종결** — 상수 스윕 R1~R3(b299~b301)로 지터 원인 분해: 추종 노브=beta(150 합격선), 지터 주범=radius/eyelid 일괄 near-raw화. **canonical = iris 중심 1.0/150(중심만), radius/eyelid 코어 기본 유지** | bench_notes §트래킹 스윕 R1~R3 + 인사이트 |
| ④ | ✅ **종결** — 1/150c 사카드 실측 잘림 미발생 ("잘리는 거 없이 잘 따라가"). 렌더러 contour 정렬 보류(후속 백로그) | 〃 |
| ⑤ | D 페이드 최적 f값 | C로 링을 파랑에 넣고 → D 확인 |
| ⑥ | 반경 보정 계수 | C에서 초록 경계가 실제 홍채 경계와 일치하는 f값 판독 |

## 3. 할 일 (판정 수집 후) + ⚠️절대 제약

1. **canonical 조립안 §5 작성**: V2 골격 + scale 상한 4.0~4.2 + (①에서 유효하면) 채도 α + (⑤) 페이드 기본값 + (⑥) 반경 보정. 트래킹(✅확정): 코어 기본(`iris_sdk_default_stabilizer_config`, sdk_api.cpp:766)의 **iris 상수만 1.0/150으로 승격** (radius/eyelid 기본 유지 — 축 분리 인사이트). 렌더러 contour 정렬은 ④ 미발생으로 보류(백로그). 벤치 임시 API `createStabilizerTuned`(JNI/Java/데모 프리셋 사이클)는 정식 구현 시 정리.
2. **Codex R2** (조립안 + visibility budget 수치 명시 — cap 실패 재발 방지 조항).
3. **정식 구현**: 임시 슬롯(3/4/6) 원복("deprecated→ID5 fallback"), 진단 코드(C 디버그·보라) 제거, canonical은 ID5 수식 자체 수정 + 새 상수. cpp-pro 위임.
4. W2 문서화(작업 규칙) + 분할 커밋 → develop 머지(`--no-ff`, PR 생략).

⚠️ **불변식**: ID 3/4/6 임시 재배선 상태로 **develop 머지 금지** · 셰이더 분기 내 조건부 texture fetch 금지(§8.9) · 절차 림발 금지 · 저조도 후순위 · 수동 블렌드 최종 UI 금지 · 벤치는 육안 체감 정본 · Codex 단독 교차(Gemini 제외) · 커밋 접두사 영문+설명 한글.

## 4. 게이트/검증

- 기기·앱: `adb shell dumpsys package com.irislenssdk.demo | grep versionCode` → 298+ / 재빌드 시 `./scripts/build_and_install.sh` → "빌드 완료: demo-app-debug-bNNN.apk"
- cpp: `cd cpp/cmake-build-debug && cmake --build . --parallel --target iris_sdk` → `Linking CXX static library lib/libiris_sdkd.a`
- 토글 로그: `adb logcat -s GpuRenderActivity:I` → 버튼별 "bench → X" / "sclera tint cap → V" / "NLR tracking mode → ..." / "NLR stabilizer → ..."

## 5. 워킹트리 주의 — 의도적 잔존 (정상)

- `.claude/settings.local.json` (M): 로컬 권한 — 커밋 금지
- `docs/plans/bubbly-prancing-boot.md` (untracked): 본 트랙 무관 구 플랜
- env PNG skip-worktree (LFS 아티팩트 우회, `git ls-files -v | grep ^S`) — W1 킥오프 §5와 동일

## 6. 완료 후

- `NLR-W0_index.md` W2 행 갱신, `formula_bench_notes.md` 판정 기입, 메모리 `nlr-track` 갱신
- 다음: NLR-W3(캐치라이트— quotient의 부산물로 부분 해결 가능성 기록됨)~W6은 계획서(`docs/plans/iridescent-dreaming-neumann.md`) 참조. 단 W2 결과로 W 구조 재평가 여지 있음(페이드=W4 radial ramp 선행 구현에 해당, 반경 보정은 신규 트랙)
