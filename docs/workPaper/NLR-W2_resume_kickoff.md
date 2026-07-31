# NLR-W2 재개 킥오프 — 에셋 축(리밸런싱 v4.2)까지 완료, 미결 과제 6건

> 🔴 **먼저 `docs/workPaper/LENS-RENDER_resume_kickoff.md` (2026-07-31 델타)를 읽을 것.**
> 이 문서의 **§0 브랜치·기기·빌드 서술과 §4·§5 는 스테일**하다 — 트랙은 이미 develop 에 머지됐고
> (`c8262ef`), 여기 적힌 브랜치 `feature/P7-W4-sclera-luma-atten` 은 **고유 커밋 0 · develop 보다
> 32커밋 뒤처진 껍데기**다. 체크아웃하면 이후 작업분이 사라진다.
> 빌드 명령 `./scripts/build_and_install.sh` 도 **쓰지 말 것**(versionCode 를 추적 파일째 증가시키고
> `-s` 없이 install 해 기기 2대 환경에서 실패). 델타 §1 이 정정 목록을 준다.
> **§2 확정 사실 · §3 미결 6건은 유효하다.**

> 이 문서 + 아래 필독으로 새 세션이 대화 맥락 0에서 착수. 갱신: 2026-07-13 (에셋 축 v4.2 체크포인트 시점 — 구판은 git 히스토리).
> 트랙 구조가 재편됨: "수식 canonical 단독" → **3층 구조 (트래킹 ✅종결 / 렌더 수식 / ⭐에셋 파이프라인)**. 에셋 축이 W2 후반의 주 무대였다.

## 0. 진입 절차

- **필독 순서**: 이 문서 → `docs/bench/NLR-W2/formula_bench_notes.md` **"트래킹 스윕 R1~R3" 섹션부터 끝까지** (판정·인사이트·에셋 축 전체 기록, 가장 중요. 단 §3-5의 ② cap 보라 진단·⑥ 반경 f값 판독의 조작 정의는 같은 파일 **상단 Round 5~6b**에 있음) → `docs/workPaper/NLR-W2_brainstorm/codex_asset_review.md`(에셋 파이프라인 검증) + `codex_tracking_review.md`(트래킹 검증). tuck(§3-4) 배경은 `docs/lenssim-handoff/clipping-accuracy-handoff-from-lenssimulator.md` §3.
- **시각 자료**: `docs/bench/NLR-W2/rebalance_gallery_v42_b314.html` 더블클릭 (버전 진화 v1→현행 + 42종 실물 PNG 비교). 에셋 변경 후 재생성: `python3 scripts/generate_rebalance_gallery.py <라벨>`.
- **브랜치**: `feature/P7-W4-sclera-luma-atten` (develop 미머지). 체크포인트 커밋: `974a9ff`(v4.2 에셋) / `f06742a`(문서) 이후 — 이 문서 자체의 최신판 커밋 여부는 `git log --oneline -3 -- docs/workPaper/NLR-W2_resume_kickoff.md`로 확인.
- **기기**: S23+ `ANDROID_SERIAL="adb-R3CW20BBFCM-HZeo2S._adb-tls-connect._tcp"`, **b314 설치됨**(v4.2 rebal 에셋 42종 + 전 토글). 무선 adb는 화면 꺼지면 만료 — 재연결 후 versionCode 314+ 확인. 빌드 `./scripts/build_and_install.sh` (⚠️ cpp는 `cpp/cmake-build-debug` 재사용, 새 디렉토리 금지).
- **타겟 레퍼런스 원본**: `/Volumes/M3-P31/Projects/MerooMong/EtcWorkStage/lens-items/` — 타겟 솔루션의 facemorph 시트 62장(973×216 RGBA, **브랜드 하위폴더**(envie/OH/OOHA/Qrsessed/rom'u/URIA/main) 아래 제품명 파일). 카탈로그 38/42종과 자동 매칭됨. URIA 19장+main 5장은 카탈로그 미보유(확장 후보).
- **동반 에이전트**: cpp/ 수정은 `systems-programming:cpp-pro`. 외부 교차는 **Codex 단독**(tmux `20_CGG-Backend:3.1` — 세션마다 `tmux list-panes -a ... | grep codex`로 재확인, 죽어있으면 `codex --dangerously-bypass-approvals-and-sandbox`로 기동. 송신은 `tmux load-buffer`+`paste-buffer -p` 후 Enter 분리).

## 1. 목표 + 현 위치

"보편 최적 렌더링" canonical 확정. **트래킹은 종결**, 렌더 수식은 병인 제거 완료(고정 K), **에셋 파이프라인은 v4.2까지 구현 후 실기기 판정 직전에 중단**. 남은 것 = §3 미결 과제 처리 → canonical 조립 → Codex R2 → 정식 구현.

## 2. 확정된 사실 (2026-07-13 검증)

### 확정 판정 (재론 불요)
1. **트래킹 ✅종결**: canonical = **iris 중심 1.0/150 (중심만), radius/eyelid 코어 기본 유지**. 정식 구현 시 `iris_sdk_default_stabilizer_config`(sdk_api.cpp) 승격. Codex 교차 완료. A23 게이트는 사용자 결정으로 제외(출시 게이트 이월).
2. **조명 적응 ✅제거 확정**: ID5/ID6 `scale = 4.2` 고정 (셰이더 내 `mix(4.2, 구적응식, uAdaptK)` — uAdaptK=0 기본). K:adapt 복귀 실측에서 밝은 렌즈 빛남 재발 → 적응 복귀 선택지 종결.
3. **에셋 축 = 솔루션의 천장**: 타겟은 에셋을 4노브로 캘리브레이션 — ① 착색부 농도 밴드 재설정(80±17, 원본 무관 양방향) ② 잉크색 다크닝(전역 luma ×~0.74 + 림 ×~0.66) ③ 패턴 대비 상대 보존 ④ 형태(림 두께·클리어존·구조) 원본 계승. **하프톤 발견**: 타겟 시트는 공통 ~3px 도트 격자(파이프라인이 하프톤 재래스터) — "타겟 패턴이 더 촘촘"의 정체.
4. **블렌드 A/B 중간 판정**: TintLinear(고정K)+우리 원본 에셋 = "너무 어색" / **Color Replace(ID7)+타겟 에셋 = "볼만하다"(현 최선)**. 기본 블렌드 교체 후보로 ID7 부상 (§3-3).
5. tuck 리매핑(LensSim §3 이식): 1.0 과클립, **0.75 정당·0.85 후보** — 정밀 판정 미완.

### 도구·코드 상태
- **`scripts/rebalance_lens_assets.py` v4.2**: lens-items 자동 매칭(퍼지 ≥0.82 + 마진 경고) → 쌍 실측 스케일 + 패턴 k 로그 스캔(양방향 수렴 플래그) + 잉크색 mid/rim 반경 보간 + `OVERRIDES` 딕셔너리. 실행하면 42종 `-rebal` 재생성(카탈로그 자동 등록, LensManager 폴더 스캔).
- **수렴 실패 플래그 2종**: doll-choco(0.45) / dear-mellow(0.50, 포화 24%) — α-only 모델 밖(타겟이 재드로잉/별도 래스터 추정). §3-2 개별 결정 대상.
- 데모 벤치 토글(⚙, 버튼 행 가로 스크롤): 슬롯 A/B/C/D(ID5/3/4/6 임시 재배선) · cap sweep · **tuck {off,0.75,0.85,1.0}** · **K:fix↔K:adapt** · stab 프리셋(norm/2/150a/2/150c/**1/150c**=확정값) · img · Mask 3-way(tuck은 Contour에서) · "밝기" 슬라이더(B=채도, C/D=페이드 창).

## 3. 미결 과제 (우선순위순)

1. **v4.2 실기기 판정** (b314 설치됨 — 첫 작업): ① 잉크색 전역 보정 톤 — 타겟답게 가라앉았나, 과해서 칙칙한가 ② 패턴 스캔 수렴 체감(nude-ash-rose 과보정 해소 등) ③ 빛남 취약 렌즈(라구나 등)에서 rebal+고정K 조합의 빛남 잔존. 과하면 렌즈별 `OVERRIDES` 보정 → 재생성.
2. **플래그 2종 개별 결정**: doll-choco·dear-mellow — (a) 현상 수용 (b) 타겟 시트 크롭을 에셋으로(저해상 소프트) (c) 재드로잉.
3. **기본 블렌드 채용 결정**: Lum Tint(고정 K=4.2) vs **Color Replace(ID7)** — rebal 에셋 위에서 재비교 후 결정. ID7 채택 시 [[w5-b1-color-replace-decision]] 종결.
4. **tuck 정밀**: 0.75 vs 0.85 (Mask: Contour).
5. **구 렌더 판정 잔여 재스코프**: ② cap 배선 진단(C 슬롯 보라 — 여전히 미결) · ⑥ 반경 보정 계수(C 슬롯 f값 판독). ⚠️ ⑤(흰자 페이드)·①(채도 부스트)은 **에셋 축이 사실상 대체** — 에셋에서 외곽 알파가 죽으므로 페이드 불요 가능성 높음, 채도는 저채도 렌즈 무효 판정. canonical 조립 때 폐기 여부 명시.
6. **canonical 조립안 → Codex R2 → 정식 구현**: 트래킹 승격(iris 1.0/150) + 고정 K 정리(uAdaptK 토글 제거) + tuck 기본값 채택 + 기본 블렌드 확정 + 벤치 임시물 정리(ID 3/4/6 "deprecated→ID5 fallback" 원복, createStabilizerTuned·K:adapt·C 디버그 제거) + W2 문서화 + 분할 커밋 → develop 머지(--no-ff, PR 생략).
- **백로그**(조립 불포함): v5 하프톤 재래스터(blue-noise 필수 — 정격자 모아레, Codex), RGB hue 보정(~11.5° 차이 미구현), 트래킹 후속(d_cutoff·beta 150~200·contour 정렬·A23·2-state), miyu 4종 시트 미보유(정규화 추정 상태), Codex 지적 잔여(n=38 비독립성 캐비앗, 프리멀티/감마/edge dilation 미측정).

## 4. 게이트/검증

- 기기·앱: `adb shell dumpsys package com.irislenssdk.demo | grep versionCode` → **314+**
- 에셋 재생성: `python3 scripts/rebalance_lens_assets.py` → "대상 42개 — 쌍 실측 38 / 정규화 4" + 플래그 2종(doll-choco/dear-mellow)만 ⚠️
- cpp 빌드: `cd cpp/cmake-build-debug && cmake --build . --parallel --target iris_sdk` → `libiris_sdkd.a` 링크
- 토글 로그: `adb logcat -s GpuRenderActivity:I` → "NLR clip tuck → X" / "NLR tint K → FIX/ADAPT" / "NLR stabilizer → ..."

## 5. 워킹트리 주의 — 의도적 잔존 (정상)

- `.claude/settings.local.json` (M): 로컬 권한 — 커밋 금지
- `docs/plans/bubbly-prancing-boot.md` (untracked): 본 트랙 무관 구 플랜
- (참고) `cpp/include/iris_sdk/export.h`는 안드로이드 빌드마다 CMake가 NOLINT 주석 1줄을 제거해 더럽힘 — 커밋 전 `git restore`가 관례

## 6. 완료 후

- `formula_bench_notes.md` 판정 기입 + 이 문서 갱신 + 메모리 `nlr-track` 갱신
- ⚠️ 불변식: ID 3/4/6 재배선 + 벤치 API(createStabilizerTuned/uAdaptK/uClipTuck 벤치 배선) 상태로 **develop 머지 금지** · 셰이더 분기 내 조건부 texture fetch 금지(GLSL ES 3.0 spec §8.9 — 배경은 `docs/workPaper/P7-W1_0x501_spec_fix.md`) · 절차 림발 금지(에셋 책임 — 에셋 축 발견으로 재입증) · 벤치는 육안 체감 정본 · Codex 단독 교차 · 커밋 접두사 영문+설명 한글
