# 렌즈 렌더링 고도화 트랙 (NLR-W2) 착수 킥오프

> **이 문서 하나로 대화 맥락 0인 새 세션이 착수한다.** 작성 2026-07-31.
> fresh-eyes 적대 게이트 2라운드(3관점 × 2회)를 거쳐 지적을 반영한 판이다.
>
> 트랙 정본 `NLR-W2_resume_kickoff.md`(2026-07-13)는 **§0 진입절차·§4 게이트·§5 워킹트리가 스테일**하다.
> 그 문서의 유효분(확정 사실·미결 과제·불변식)은 **여기 인라인**했으므로 **읽지 않아도 착수 가능**하다.
> 배경을 더 파고 싶을 때만 §9 의 참고 문서로 갈 것.

---

## 0. 진입 절차

### 0-0. ⛔ 사용자 조치가 선행되어야 한다

**이 트랙의 판정은 전부 실기기 육안이다.** 작성 시점 기준 **두 기기 모두 adb 연결이 끊겨 있고**,
무선 adb 포트는 재활성화할 때마다 바뀌므로 **문서에 적어둘 수 없다.**

> **에이전트 단독 완주 범위 = §3 순서 1(재빌드)뿐이다. 2~7 은 전부 기기가 필요하다.**

착수하면 **재빌드를 백그라운드로 먼저 걸고**, 동시에 사용자에게 요청할 것:

> 1. 두 기기 화면을 깨우고 `설정 → 개발자 옵션 → 무선 디버깅`을 켜 주세요
> 2. 표시된 **IP 주소 및 포트**를 알려주세요 (포트는 매번 바뀝니다)
> 3. 무선 디버깅 화면을 **연 채로** 두시면 포트가 유지됩니다

```bash
ADB=~/Library/Android/sdk/platform-tools/adb
$ADB devices -l                     # 비어 있으면 위 요청
$ADB connect <IP:PORT>              # 사용자가 알려준 값
```
- ⚠️ **mDNS 광고(`adb mdns services`)가 보여도 `Connection refused` 가 날 수 있다** — 광고 레코드만
  캐시로 남고 서비스가 안 뜬 상태다. 기기에서 무선 디버깅을 껐다 켜 **새 포트**를 받아야 한다.
  2~3회 시도해도 안 되면 붙잡지 말고 사용자에게 되물을 것.
- 기기 식별은 serial 이 아니라 `adb devices -l` 의 `model:` 로 (serial 은 재연결마다 바뀐다).
  - 폰 **SM-S916N** (S23+, **세로 잠금**) ← **NLR 판정의 주 기기**
  - 태블릿 **SM-X920** (Tab S10 Ultra, 가로)
- 화면 절전 방지: `svc power stayon true` 는 **충전 중일 때만** 유효(무선 adb + 미충전이면 무효).
  절전되면 `screencap` 이 전부 검게 나온다 — 안 되면 화면 꺼짐 시간을 늘려 달라고 요청할 것.

**사용자 응답을 기다리는 동안 할 수 있는 무해한 준비**: §3-0 비교 대상 렌즈 3~5종 선정,
`rebalance_gallery_v42_b314.html` 로 타겟 톤 대조표 확인.

### 0-1. 빌드·설치·기동

```bash
cd /Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/android && ./gradlew :demo-app:assembleDebug

ADB=~/Library/Android/sdk/platform-tools/adb
APK=$(ls -t /Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/android/build/modules/demo-app/outputs/apk/debug/*.apk | head -1)
ls -l "$APK"                                    # ① 빌드 직후 타임스탬프인지 확인
$ADB -s <serial> install -r "$APK"              # 무선이라 1~3분. run_in_background 로 돌릴 것

# ② 설치 신선도 검증 — lastUpdateTime 이 위 ① 과 맞는가 (versionCode 는 314 고정이라 판별력 0)
$ADB -s <serial> shell dumpsys package com.irislenssdk.demo | grep -E 'lastUpdateTime|versionName'

# ③ 콜드스타트 (런처 = .GpuRenderActivity)
$ADB -s <serial> logcat -c
$ADB -s <serial> shell am force-stop com.irislenssdk.demo
$ADB -s <serial> shell am start -n com.irislenssdk.demo/.GpuRenderActivity
```

> ⛔ **`./scripts/build_and_install.sh` 를 쓰지 말 것** (정본 §0 이 지시하는 스크립트). 검증한 이유:
> 1. **`versionCode` 를 추적 파일째 증가**시킨다 — `:57-68` 이 `build.gradle.kts` 를 `sed` 로 314→315.
>    워킹트리가 더러워지고 APK 이름이 `b315` 로 바뀌어 아래 경로 전제가 깨진다. (`--no-increment` 옵트아웃은 `:34` 에 있다.)
> 2. **`.cxx` 삭제 + `gradlew clean`** — 증분 네이티브 캐시를 지워 불필요한 풀 리빌드 강제.
> 3. **`-s` 없는 adb 호출** — `:112` uninstall / `:115` install 둘 다 serial 미지정이라 기기 2대에서 실패.

- ⚠️ **설치 완료를 확인한 뒤** 앱을 띄울 것. 설치 중에 실행하면 구 빌드가 뜬다(실제로 겪은 함정).
- ⚠️ `cpp/` 는 `cpp/cmake-build-debug` **재사용**. 새 디렉토리 금지(TFLite 400MB 재다운로드).
- ⚠️ `cpp/include/iris_sdk/export.h` 는 안드로이드 빌드마다 CMake 가 NOLINT 주석 1줄을 지워 더럽힌다 —
  커밋 전 `git restore cpp/include/iris_sdk/export.h` 가 관례.

**브랜치**: `develop` 에서 직접 착수. 착수 시 `git log --oneline -1` 로 현재 팁을 확인할 것.

**동반 에이전트**: `cpp/` 수정은 `systems-programming:cpp-pro` (CLAUDE.md 규칙).
외부 교차는 **Codex 단독** — ⚠️ 정본이 적은 tmux 좌표 `20_CGG-Backend:3.1` 은 **없다**(`3.0` 만 존재,
codex 실행 pane 0건). 빈 pane 을 잡아 `codex --dangerously-bypass-approvals-and-sandbox` 로 기동하고,
송신은 `tmux load-buffer` + `paste-buffer -p` 후 Enter 분리.

---

## 1. 목표 + 현 위치

**"보편 최적 렌더링" canonical 확정.** 트랙은 3층이다 — 트래킹 ✅종결 / 렌더 수식(병인 제거 완료) /
⭐**에셋 파이프라인**(v4.2까지 구현 후 실기기 판정 직전 중단). 남은 것 = §3 미결 처리 → canonical 조립.

여기에 **오늘 이월된 판정 2건**(SHARP G4 홍채 정합 · 분석 스트림 1280×720)이 얹힌다. 둘 다
"렌즈를 켜고 봐야 판별력이 생기는" 성격이라 같은 세션에서 처리한다.

---

## 2. 확정된 사실 — 재론 불요

### 2-1. 판정 (정본 §2 에서 인라인)

1. **트래킹 ✅종결**: canonical = **iris 중심 1.0/150 (중심만)**, radius/eyelid 코어 기본 유지.
   정식 구현 시 `iris_sdk_default_stabilizer_config`(`sdk_api.cpp`) 승격. Codex 교차 완료.
   A23 게이트는 사용자 결정으로 출시 게이트 이월.
2. **조명 적응 ✅제거 확정**: ID5/ID6 `scale = 4.2` 고정(셰이더 `mix(4.2, 구적응식, uAdaptK)`, `uAdaptK=0` 기본).
   K:adapt 복귀 실측에서 밝은 렌즈 빛남 재발 → 적응 복귀 선택지 종결.
3. **에셋 축 = 솔루션의 천장**: 타겟은 에셋을 4노브로 캘리브레이션 —
   ① 착색부 농도 밴드 재설정(80±17, 원본 무관 양방향) ② 잉크색 다크닝(전역 luma ×~0.74 + 림 ×~0.66)
   ③ 패턴 대비 상대 보존 ④ 형태(림 두께·클리어존·구조) 원본 계승.
   **하프톤 발견**: 타겟 시트는 공통 ~3px 도트 격자 — "타겟 패턴이 더 촘촘"의 정체.
4. **블렌드 A/B 중간 판정**: TintLinear(고정K)+우리 원본 에셋 = "너무 어색" /
   **Color Replace(ID7)+타겟 에셋 = "볼만하다"(현 최선)**. 기본 블렌드 교체 후보로 ID7 부상(→ 미결 3).
5. **tuck 리매핑**(LensSim §3 이식): 1.0 과클립, **0.75 정당 · 0.85 후보** — 정밀 판정 미완(→ 미결 4).

### 2-2. 도구·코드 상태 (2026-07-31 실측)

| 항목 | 상태 |
|---|---|
| **에셋 생성** `scripts/rebalance_lens_assets.py` | v4.2. lens-items 자동 매칭(퍼지 ≥0.82) → 쌍 실측 스케일 + 패턴 k 로그 스캔 + 잉크색 mid/rim 반경 보간. **`OVERRIDES` 는 `:45`**(현재 `{}` — 비어 있음), 소비부 `:218`. 출력 = `-rebal` PNG |
| **갤러리 생성** `scripts/generate_rebalance_gallery.py` | ⚠️ **에셋을 만들지 않는다.** HTML 대조표만 출력(`docs/bench/NLR-W2/rebalance_gallery_<label>.html`) |
| **에셋 구성** | 총 88 PNG = **제품 원본 42 ↔ `-rebal` 42 (완전 쌍, 고아 없음)** + 타겟 참조 4장(`claset_never-olive-target{,-a135,-a170}`, `zz_target-ref`). `LensManager` 가 폴더를 전수 스캔해 **타겟 4장도 렌즈 목록에 뜬다** — 대조 대상에서 제외할 것 |
| **수렴 실패 2종** | doll-choco(0.45) / dear-mellow(0.50, 포화 24%) — α-only 모델 밖(→ 미결 2) |
| **데모 벤치 토글** (⚙ 우상단 → 버튼 행) | 슬롯 **A/B/C/D**(ID5/3/4/6 임시 재배선) · cap sweep · **tuck {off,0.75,0.85,1.0}** · **K:fix↔K:adapt** · stab 프리셋(norm/2/150a/2/150c/**1/150c**=확정) · Mask 3-way · "밝기" 슬라이더. **진단 제거(`f7d0fbd`) 후에도 전부 생존 확인** |
| **블렌드 ID** (`cpp/include/iris_sdk/types.h:25-36`) | 활성 `0`Normal `1`Multiply `2`ScreenLinear **`5`LuminanceTintLinear(기본)** **`7`ColorReplace** / deprecated `3``4``6` → 셰이더 TintLinearV2 fallback |
| **타겟 레퍼런스 원본** | `/Volumes/M3-P31/Projects/MerooMong/EtcWorkStage/lens-items/` — 브랜드 7폴더(envie/OH/OOHA/Qrsessed/rom'u/URIA/main), facemorph 시트 62장(973×216 RGBA). 카탈로그 38/42 자동 매칭. **마운트 확인됨** |

> ⚠️ **벤치 슬롯 라벨이 코드 안에서 3중으로 어긋난다.** 정본은 **`benchCombos`(`GpuRenderActivity.kt:680-686`)와
> 셰이더**다 — C = ID4 = 기하 디버그(초록/빨강/파랑 밴드 + cap≤1.2 보라)이며 `cpp/src/gpu/shader_sources.cpp:740`
> 에서 실증된다. **스테일**: 주석 `GpuRenderActivity.kt:773-775`(KM/OkShift/Pivot)와 스피너 라벨 `:780`
> (`Quot†`/`Quot+F†`/`V2+Fade†`). 미결 6 조립 때 세 곳을 함께 정리할 것.

### 2-3. ⚠️ 정본에서 스테일해진 것 (그대로 따르면 안 되는 것)

| 정본 서술 | 실제 (2026-07-31 검증) |
|---|---|
| §0 "브랜치 `feature/P7-W4-sclera-luma-atten` (develop **미머지**)" | ❌ **이미 머지 완료**(`c8262ef`, 2026-07-14). 그 브랜치는 **고유 커밋 0 · develop 보다 32커밋 뒤처진 껍데기** — 체크아웃하면 이후 작업분이 사라진다 |
| §0 빌드 `./scripts/build_and_install.sh` | ⛔ **쓰지 말 것** — §0-1 참조 |
| §0 "b314 설치됨" / §4 게이트 "versionCode → 314+" | ⚠️ **판별력 0**(314 고정). 설치 확인은 **`lastUpdateTime`**(§0-1 ②) |
| §0 Codex tmux `20_CGG-Backend:3.1` | ❌ 그 pane 없음(`3.0` 만). §0-1 참조 |
| §3-6 / §6 / 코드주석 `:773-775` "벤치 임시물 정리 **전 머지 금지**" | ⚠️ **이미 포함한 채 머지됐다**(`c8262ef`). `bench_toggles.h` 현존. 금지 조항 **무효** — 정리는 미결 6 의 잔여 과제 |
| §5 워킹트리 목록 | ❌ 현재와 다르다. **§7 이 대체** |
| §4 나머지 3게이트(에셋 재생성·cpp 빌드·토글 로그) | ✅ **유효** — §6 에 인라인했다 |

---

## 3. 할 일 — 실행 순서

| 순서 | 항목 | 기기 | 판정축 (무엇을 보면 통과인가) |
|---|---|---|---|
| 1 | 재빌드(백그라운드) + 기기 연결 요청 | — | §0-0. **사용자 응답 없으면 이 세션은 여기서 정지** |
| 2 | **미결 1** v4.2 에셋 실기기 판정 | **폰** | §3-0 |
| 3 | **미결 2** 수렴 실패 2종 결정 | **폰** | §3-1 |
| 4 | **3-A** SHARP G4 홍채 정합 재확인 | **태블릿** | §3-2 |
| 5 | **미결 3** 기본 블렌드 (ID5 vs ID7) | **폰** | §3-3 |
| 6 | **미결 4** tuck 0.75 vs 0.85 | **폰** | §3-4 |
| 7 | **3-B** 1280×720 채택 여부 | **태블릿** | §3-5. **선택** — 시간 남을 때만 |
| 8 | 미결 5·6 (재스코프·canonical 조립) | — | **다음 세션** — 분량이 크다 |

**세션 종료 조건**: **2~6 완료 = 최소 구획.** 7은 시간이 남을 때만 착수하되 **착수했으면 원복·재설치까지
같은 세션에서 끝낸다**(중간 종료 금지 — 워킹트리를 건드리는 유일한 항목).

> **블렌드 고정**: 2~4·6 판정은 **블렌드 스피너 = `Lum Tint Linear`(ID5, 앱 기본)로 고정**하고 수행한다.
> 미결 3(순서 5)에서만 **같은 렌즈·같은 `-rebal` 에셋으로 ID5↔ID7만 바꿔** 재실행한다.
> 이렇게 해야 미결 1과 미결 3 이 서로 오염되지 않는다.

### 3-0. 미결 1 — v4.2 에셋 톤 판정

1. **비교 대상 3~5종을 먼저 고정**한다. 권장: 과보정 해소 확인용(nude-ash-rose 계열) ·
   빛남 취약(라구나 계열) · 수렴 실패(doll-choco).
2. 앱 렌즈 목록에서 **같은 제품의 원본 ↔ `-rebal` 을 번갈아 선택**해 대조한다
   (제품 42종 전부 완전 쌍이라 어느 것이든 가능. `... target ...` 4장은 제외).
3. **판정축**: ① 잉크색 전역 보정 톤이 **타겟답게 가라앉았나 / 과해서 칙칙한가**
   ② 패턴 스캔 수렴이 체감되나 ③ 빛남 취약 렌즈에서 rebal+고정K 조합의 **빛남 잔존** 여부.
4. 톤 기준이 필요하면 `docs/bench/NLR-W2/rebalance_gallery_v42_b314.html` 을 데스크톱에서 열어
   같은 제품의 **타겟 시트**를 옆에 띄운다.
5. **과하면 시정** — ⚠️ 순서 주의:
   ```bash
   # ① scripts/rebalance_lens_assets.py:45 의 OVERRIDES 에 렌즈별 항목 추가 (현재 {} — 소비부 :218 에서 키 스키마 확인)
   python3 scripts/rebalance_lens_assets.py            # ② 에셋 재생성 (게이트: §6)
   # ③ APK 재빌드·재설치 (§0-1)
   python3 scripts/generate_rebalance_gallery.py <라벨>  # ④ 데스크톱 대조표가 필요할 때만
   ```
   **에셋을 만드는 건 `rebalance_lens_assets.py` 다.** `generate_rebalance_gallery.py` 는 HTML만 만든다.

### 3-1. 미결 2 — 수렴 실패 2종 (doll-choco 0.45 / dear-mellow 0.50)

- **이번 세션 범위는 "(a) 현상 수용 가능한가" 판정 하나로 좁힌다.**
- 판정축: 해당 2종을 `-rebal` 로 봤을 때 **다른 렌즈들과 톤이 튀는가 / 그냥 써도 되는가.**
- (b) 타겟 시트 크롭을 에셋으로 / (c) 재드로잉 은 **절차가 어느 문서에도 없다**
  (973×216 facemorph 시트 → 렌즈 PNG 규격 변환 방법 미정의). **백로그로 내릴 것.**

### 3-2. 3-A — SHARP G4 홍채 정합 재확인 (이월 항목)

- **무엇**: 진단 표면 제거(`f7d0fbd`) 후 렌즈 ON 상태의 홍채 정합을 **육안으로 못 닫았다**.
  캡처 시점에 시선이 크게 옆으로 가 홍채가 가려졌다 — 사람 눈이 필요하다.
- **조건 고정**(원 G4 와 같아야 '재확인'이 성립): **기기 = 태블릿 SM-X920**,
  **렌즈 = `claset doll choco`**(원본 쪽). 폰에서도 보고 싶으면 별건으로 기록할 것.
- **판정**: **정면**을 본 상태에서 양쪽 홍채에 **원형·중심 정합·좌우 대칭**인가.
  타원 찌그러짐·반전·오프셋이 없으면 통과.
- **이미 확인된 것**: 렌즈 정상 활성 · `nativeRenderLensTexture 1080x1920`(불변식 §5-1) ·
  `TexturePool` 실패 로그 0건. G4 자체는 **진단 제거 이전**에 통과했었다(§7-3 표의 ✅ 는 그것).
- **위험도 낮음** — 제거는 순수 삭제였고 렌즈 좌표 경로를 안 건드렸다. 형식 확인에 가깝다.

### 3-3. 미결 3 — 기본 블렌드 채용 (ID5 vs ID7)

- **조작**: 동일 렌즈 · 동일 `-rebal` 에셋에서 **블렌드 스피너만 ID5(Lum Tint Linear) ↔ ID7(Color Replace)**.
- **판정축**: ① 톤 자연스러움(타겟 대비) ② **흰자 빛남 잔존** 유무.
- 중간 판정은 ID7+타겟 에셋 우세였다(§2-1-4). rebal 에셋 위에서 뒤집히는지가 이번 관건.
- ID7 채택 시 `w5-b1-color-replace-decision` 트랙도 함께 종결된다.

### 3-4. 미결 4 — tuck 0.75 vs 0.85

- **조작**: `Mask: Contour` 고정 → tuck 버튼으로 `0.75` ↔ `0.85` 토글.
- **판정축**: **눈꺼풀 경계에서 렌즈가 깎여 나가는가(과클립)** ↔ **눈꺼풀 위로 삐져나오는가.**
  1.0 은 과클립으로 이미 기각됐다.

### 3-5. 3-B — 분석 스트림 1280×720 채택 여부 (이월 항목, **선택**)

- **현재 상태**: 요청 기본값 **960×540**(`GpuRenderActivity.kt:1258` 의 `else` 가지).
  이 값은 어느 기기에도 없어 폴백이 갈린다 — **태블릿 512×288 · 폰 640×360**
  (폰 값 근거: `SHARP_ring_fbo_transpose.md` §7-3b `analysis buffer=640x360`).
- **1차 판정은 이미 부정**: Detect FPS **30 → 27**, 육안 "뭔가 불안정하다".
  이번에 볼 것은 **"렌즈 정합에서의 정밀도 이득이 그 부정 판정을 뒤집는가"** 하나다.
- **편집**: `GpuRenderActivity.kt:1258` 의 `else Size(960, 540)` → `else Size(1280, 720)`
- **⚠️⚠️ 이 한 줄은 폰에도 적용된다.** `totalRamMb <= 3072` 의 `else` 는 두 기기 공용이고 둘 다 3GB 초과다.
  → **미결 1~6 판정과 같은 APK 로 섞지 말 것.** 3-B 전용으로 빌드·판정한 뒤 반드시 원복:
  ```bash
  git checkout -- android/demo-app/src/main/java/com/irislenssdk/demo/GpuRenderActivity.kt
  ```
  ⚠️ **`git checkout HEAD -- <path>` / `git restore --source=HEAD` 는 쓰지 말 것** — 인덱스에 다른 변경이
  있을 때 함께 날린다. 원복 후 **재빌드·재설치**까지 해야 NLR 판정을 이어갈 수 있다.
- **확인 명령** — ⚠️ `analysis buffer=` 는 **회전이 바뀔 때만** 찍힌다(`GpuRenderActivity.kt:1398`
  의 `if (rotation != lastRotation)`). 세로 잠금 폰은 **콜드스타트 1회뿐**이라 실행 중에 `logcat -c`
  하면 영영 안 나온다. 반드시 이 순서로:
  ```bash
  $ADB -s <serial> logcat -c
  $ADB -s <serial> shell am force-stop com.irislenssdk.demo
  $ADB -s <serial> shell am start -n com.irislenssdk.demo/.GpuRenderActivity
  sleep 12
  $ADB -s <serial> logcat -d -s GpuRenderActivity:D | grep -E "Analysis resolution|analysis buffer"
  # "Analysis resolution:" = 요청값 / "analysis buffer=" = 실개통값 ← 후자를 봐야 한다
  ```
- **판정 → 조치**
  - **뚜렷한 이득** → 채택. Detect 27fps 는 프로젝트 30fps 기준의 **명시적 예외**로 기록.
    ⚠️ 채택 시 **폰에서도 재확인 필수**(1280×720 개통 여부 + Detect fps) — 기기 공통 경로다.
  - **애매** → 960×540 유지, 이 항목 종결
  - **품질은 좋은데 끊김이 걸린다** → **단일 스트림 경로**로 이월 ← 유력

---

## 4. 🆕 오늘 develop 에 들어온 변화가 렌즈 판정에 미치는 영향

**정본 §2 의 시각 판정은 SHARP 수정 이전에 내려졌다.** 오늘 머지된 SHARP 수정(`e4d85e4`)은 링 FBO
전치 리샘플을 제거해 **선명도를 크게 올렸다** — 폰도 예외가 아니다:

| 기기 | 개선 |
|---|---|
| 태블릿 SM-X920 | 가로 2차차분 **2.35배** |
| 폰 SM-S916N (NLR 주 기기) | 링 공간 **1.92배** / 화면 **2.09배** |

→ ⚠️ 에셋 톤·질감·패턴 수렴처럼 **선명도에 민감한 판정은 영향을 받는다.** 다만 미결 **1·2** 는 아직
판정 전이므로 **지금 보는 것이 오히려 맞다** — 개선된 렌더로 처음 보는 셈이다.
이미 확정된 판정(트래킹 종결, 고정 K=4.2)은 기하·수식 축이라 선명도와 독립적이다.

TRACK-ROT 수정으로 **가로 화면 추적도 안정화**됐다 — 태블릿 가로 판정 시 메시가 종전처럼 일그러지지 않는다.

> ⛔ **선명도를 재측정할 수단은 현재 없다.** `SHARP_ring_fbo_transpose.md` §8 이 안내하는
> `DUMP_RING` / `SET_RING_SWAP` / `SET_UPSCALE` 브로드캐스트는 `f7d0fbd` 에서 **리시버가 전부 삭제**됐고
> **명령을 쳐도 조용히 no-op 한다**(검증: `grep -rn 'DUMP_RING\|SET_RING_SWAP\|SET_UPSCALE' android/
> --exclude-dir=build` = 0건). 재측정하려면 `f7d0fbd` 이전을 체크아웃하거나 진단 경로를 재삽입해야 한다.
> 측정 스크립트는 `docs/ar-report/sharp-measure/{axis_metrics,measure_mag,ring_vs_fmlens}.py`
> (⚠️ `docs/ar-report` 는 gitignore 라 **이 디스크에만** 존재).

---

## 5. ⚠️ 절대 제약 (불변식 — 깨면 앞선 트랙이 무효화된다)

1. **렌즈·뷰티에 넘기는 텍스처 치수는 반드시 `ringW`/`ringH`** (`CameraGLRenderer`).
   어긋나면 native 패스가 링을 **재전치 리샘플**해 SHARP 수정이 통째로 무효가 되는데,
   **렌즈 기하는 UV 불변이라 육안으로는 멀쩡하다.** 선명도 측정으로만 잡힌다.
   **확인 명령** — 로그는 `LOGV`(VERBOSE) + 태그 `IrisSDK-JNI` 다(`iris_jni.cpp:2036`,
   `jni_utils.h:32` 태그 / `:39-42` NDEBUG 가드). **태그·레벨이 틀리면 아무것도 안 나와
   '불변식 깨짐'으로 오판한다**:
   ```bash
   $ADB -s <serial> logcat -c
   $ADB -s <serial> logcat -d -s IrisSDK-JNI:V | grep nativeRenderLensTexture
   # 기대: nativeRenderLensTexture: texture=N, 1080x1920
   ```
   ⚠️ `LOGV` 는 `NDEBUG`(Release)에서 **컴파일 제거**된다 — debug 빌드에서만 확인 가능.
2. **좌표 변환 경로(`TasksToIrisResult` / `CoordMapper` / native 렌즈 좌표)는 건드리지 않는다.**
   MediaPipe 가 힌트와 무관하게 원본 센서 공간으로 재투영한다는 계약 위에 TRACK-ROT 수정이 서 있다.
3. **`ensureAnalysisTarget`(`CameraGLRenderer`)은 손대지 않는다.** 같은 전치 버그가 잠복해 있으나
   현재 미배선이다. 배선 시 **치수와 회전을 반드시 함께** 바꿔야 하며 반쪽만 바꾸면 렌즈가 3.16:1 타원이 된다.
   (§3-5 의 "단일 스트림 경로"가 이 배선 — 착수하면 **별도 트랙**.)
4. **`recreateIntermediateBuffers` 에 '치수 동일 시 조기 return' 가드를 넣지 말 것.**
   `markGlHandlesStale` 후 같은 치수로 재진입하므로 조기 return 하면 링이 영영 안 만들어져 블랙스크린.
5. **`kTaper`·`kMaxDispRatio`·`kSigmaRatio` 등 jaw/upper 값은 S23+ 검증 고정값 — 수정 금지**
   (`cpp/include/iris_sdk/warp/jaw_warp_geometry.h`). 2026-07-31 뷰티 재판정에서 재확인됨.
6. **셰이더 분기 내 조건부 texture fetch 금지** (GLSL ES 3.0 spec §8.9 — 배경 `docs/workPaper/P7-W1_0x501_spec_fix.md`).
7. **절차적 림발 금지** — 림발은 **에셋의 책임**이다(에셋 축 발견으로 재입증). 셰이더에서 darkening 하지 말 것.
8. **시각 판정에 정량 지표를 쓰지 말 것.** 전례 2건: 정지 상태 마커 흔들림 지표는 판별력이 없었고
   (TRACK-ROT §3-3 — 1라운드 결론이 3라운드에서 뒤집힘), 보간 커널 비교에 라플라시안 분산을 쓰면
   bilinear 계단 아티팩트가 수치를 부풀린다(SHARP §4-3). **육안 판정 축이다.**
9. **외부 교차는 Codex 단독.** 커밋 접두사 영문 + 설명 한글.

---

## 6. 게이트 / 검증 (정본 §4 유효분 인라인)

```bash
# 에셋 재생성 검증
python3 scripts/rebalance_lens_assets.py
#   기대: "대상 42개 — 쌍 실측 38 / 정규화 4" + 플래그 2종(doll-choco/dear-mellow)만 ⚠️

# cpp 빌드
cd cpp/cmake-build-debug && cmake --build . --parallel --target iris_sdk
#   기대: libiris_sdkd.a 링크

# 토글 반영 확인 — 내가 누른 토글이 실제로 먹었는가 (미결 3·4 에서 필수)
$ADB -s <serial> logcat -s GpuRenderActivity:I | grep -E "NLR clip tuck|NLR tint K|NLR stabilizer"
#   기대: "NLR clip tuck → 0.75" / "NLR tint K → FIX(4.2, 확정)" / "NLR stabilizer → ..."
#   (코드 실존 확인: GpuRenderActivity.kt:640, :658, :666)
```

---

## 7. 워킹트리 주의 — 커밋 금지 의도적 제외분 (정상 잔존)

```
 M .gitignore                                  ← 사용자가 docs/ar-report, docs/fm-lens-images 추가. 손대지 말 것
?? .claude/settings.local.json.doctor-backup   ← 도구 백업
?? docs/plans/bubbly-prancing-boot.md          ← 이 트랙 무관 구 플랜
?? scripts/__pycache__/                        ← 파이썬 캐시
```
중단 잔여물이 아니라 **정상 상태**다. 커밋하지 말 것.
**착수 시 `git status --short` 로 위 4건 외에 뭐가 더 있는지 먼저 확인할 것.**

> ⚠️ **`scripts/__pycache__` 는 gitignore 되어 있지 않다**(검증: `git check-ignore -v scripts/__pycache__`
> 미매치, `.gitignore` 에 `__pycache__` 항목 없음). 종전 문서의 "gitignore 등록됨" 서술은 **거짓**이다.
> → **`git add -A` / `git commit -a` 금지. 경로를 지정해서 add 할 것.**
> `git clean -fd` / `git stash -u` 도 금지.

**gitignore 되어 git 에 없는 산출물** (디스크에만 존재):
- `docs/ar-report/` — SHARP 측정 스크립트(`sharp-measure/*.py`)·게이트 캡처
- `docs/fm-lens-images/` — FM 제공 리소스 **63폴더**

---

## 8. 완료 후

**이번 구획(§3 순서 2~6) 종료 시 — 반드시**
- `docs/workPaper/NLR-W2_pending_tasks.md` 체크 + 한 줄 결과 기입 (미결 상태 추적 단일 창구)
- 전체 판정 기록은 `docs/bench/NLR-W2/formula_bench_notes.md`
- 3-A / 3-B 결과는 각각 `SHARP_ring_fbo_transpose.md` §7-4-7 / `ANALYSIS-RES_tablet_stream_fallback.md` §6 에 반영
- 메모리 `MEMORY.md` 진입점 갱신

**트랙 종결 시에만**
- 미결 6 canonical 조립: 트래킹 승격(iris 1.0/150) + 고정 K 정리(`uAdaptK` 토글 제거) + tuck 기본값 +
  기본 블렌드 확정 + **벤치 임시물 정리**(ID 3/4/6 "deprecated→ID5 fallback" 원복, `createStabilizerTuned`·
  K:adapt·C 디버그 제거, 주석 `:773-775`·스피너 라벨 `:780` 동기화) → develop 머지(`--no-ff`, PR 생략)
- 이 킥오프 문서를 정본에 흡수하고 제거 (⚠️ 미결이 남은 상태에서 지우면 진입점이 사라진다)

**백로그** (조립 불포함): v5 하프톤 재래스터(blue-noise 필수 — 정격자 모아레) · RGB hue 보정(~11.5° 미구현) ·
트래킹 후속(d_cutoff·beta 150~200·contour 정렬·A23 출시 게이트·2-state) · miyu 4종 시트 미보유 ·
Codex 잔여 지적(n=38 비독립성·프리멀티/감마/edge dilation 미측정) · 미결 2 의 (b)(c) 경로

**다음 트랙 후보**
- **단일 스트림 경로 배선**(`ensureAnalysisTarget`) — §3-5 맞바꿈을 없애는 근본 해법. 불변식 3 참조
- **FM-CAL v2** — `feature/fm-lens-calibration`(고유 16커밋, develop 보다 19커밋 뒤처짐, **미머지**).
  킥오프가 그 브랜치에만 있다: `git show feature/fm-lens-calibration:docs/workPaper/FM-CAL_resume_kickoff.md`.
  ⚠️ 무관 커밋(Mali·카메라·가로뷰)이 인터리브돼 단순 merge 로는 분리 불가
- `screenRotation 180` 외삽 검증 · 자동회전 잠금 시 회전 힌트 미동작

---

## 9. 참고 문서 (배경을 더 팔 때만 — 착수에는 불필요)

| 문서 | 내용 |
|---|---|
| `docs/bench/NLR-W2/formula_bench_notes.md` | **판정 기록 정본.** "트래킹 스윕 R1~R3" 이후 전체 + 상단 Round 5~6b(② cap·⑥ 반경 조작 정의) |
| `docs/workPaper/NLR-W2_resume_kickoff.md` | 트랙 정본(2026-07-13). §0·§4·§5 스테일 — §2-3 참조 |
| `docs/workPaper/NLR-W2_pending_tasks.md` | 미결 체크리스트 (상태 추적 창구) |
| `docs/workPaper/NLR-W2_brainstorm/codex_asset_review.md` · `codex_tracking_review.md` | Codex 교차 검증 |
| `docs/lenssim-handoff/clipping-accuracy-handoff-from-lenssimulator.md` §3 | tuck 배경 |
| `docs/workPaper/SHARP_ring_fbo_transpose.md` | 선명도 트랙(종결). §7-3b 폰 실측 · §7-4-7 G4 이월 |
| `docs/workPaper/ANALYSIS-RES_tablet_stream_fallback.md` | 분석 스트림 트랙. §5-b 실측 · §6 이월 근거 |

### 부록 — 미결 5 의 ①~⑥ 판정 정의 (현행 문서에서 소실된 표)

정본 §3-5 와 `pending_tasks` 5번은 번호로만 참조하는데 **정의표가 현행 문서 어디에도 없다.**
구 리비전 `3346772` 에만 남아 있어 전재한다.

| # | 판정 | 방법 |
|---|---|---|
| ① | R7 채도 부스트 최적값 + 투명도 조합 | B + 슬라이더 스윕 (+ 투명도 75~85%) |
| ② | cap 배선 여부 | C 에서 cap 버튼 → 밴드 보라 변화 유무 |
| ③ | stab:fast 정지 지터 허용 여부 | stab:fast + 정면 응시 |
| ④ | 사카드 시 눈꺼풀 마스크 위상 지연 | stab:fast + 빠른 시선 이동 — 렌즈가 눈꺼풀에 잘리는 순간 유무 |
| ⑤ | D 페이드 최적 f값 | C 로 링을 파랑에 넣고 → D 확인 |
| ⑥ | 반경 보정 계수 | C 에서 초록 경계가 실제 홍채 경계와 일치하는 f값 판독 |

**현재 상태**: 정본은 **②·⑥만 미결**로 남기고, ⑤·①은 "에셋 축이 사실상 대체"(외곽 알파가 에셋에
내장돼 페이드 불요 / 채도는 저채도 렌즈 무효)로 본다. **③·④는 트래킹 종결에 흡수**돼 목록에서 빠졌다 —
canonical 조립 때 **폐기 여부를 명시**할 것.
**검증**: C(ID4) 기하 디버그는 `cpp/src/gpu/shader_sources.cpp:740` 에 실존하므로 위 조작은 현행 빌드에서 유효하다.
