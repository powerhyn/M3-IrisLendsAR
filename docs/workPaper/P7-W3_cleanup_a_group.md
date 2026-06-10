# P7-W3: A 그룹 cleanup — demo UI/KT 동기화 + deprecated 제거 + SKU 정정

> **상태**: ✅ **구현 완료** (2026-06-10). deprecated setLensHighlight + 3D Light UI 제거(C8 C API 보존) + SKU 톤 정정 + Phase 9 cross-link. Android BUILD SUCCESSFUL. blend dropdown은 P7-W4 후로 보류(사용자 확정).
> **작성**: 2026-06-04
> **선행 의존**: 없음 (P7-W1과 병렬 가능, risk 0)
> **병렬 가능**: P7-W1 (0x501 spec fix)
> **소요 추정**: 0.5~1.0 작업일
> **분리 PR**: P7-W1과 별도 PR (P7-W0 §4 사용자 확정)
> **참조**: `docs/workPaper/P7-W0_index.md` §2.P7-W3, `P7-W0_brainstorm/synthesis.md` §2.Q4

---

## 1. 인사이트 (세션 간 맥락 보존) ⭐

### 1.1 이 W의 정체성

**Phase 6 종료 후 잔재 cleanup 일괄**. 시각/기능 변경 없는 정리 작업이라 **risk 0** — P7-W1(0x501 spec fix)과 병렬 진행 가능. R1에서 Claude/Gemini는 W1과 병렬 우선, Codex는 W3 결정 뒤에 닫기 권장 — 다수결로 병렬 가능 (단 blend dropdown 축소는 P7-W4(흰자 빛남) 결정 뒤로 미룰 수도).

본 W는 다음 4가지 일괄:
1. **demo UI / KT 동기화** — 블렌드 drop-down 3종 축소, 3D Light 버튼 제거, 기본 blendMode TintLinearV2 ID=5
2. **deprecated no-op 제거** — `setLensHighlight()` 등 호환성 transition 종료
3. **SKU 톤 분류 정정** — "누드 애쉬 로제" 웜톤 X → 애쉬+누드
4. **Phase 6 보존 산출물 cross-link** — W3/W4/W8 이월 트랙 재개 진입점 명시 (Phase 9 분리, 메모리 `phase9-deferred-tracks`)

### 1.2 변경 대상 (권위 소스 확인 2026-06-04)

#### A. deprecated `setLensHighlight()`
- 정의: `android/iris-sdk/src/main/java/com/irislenssdk/IrisLensSDK.java:1107` — no-op
- 호출처: `android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLRenderer.kt:1612`
- 처리: 정의 + 호출 모두 완전 제거 (호환성 transition 종료)

#### B. 3D Light UI 토글
- 로직: `android/demo-app/src/main/java/com/irislenssdk/demo/GpuRenderActivity.kt:444` (`btnToggleHighlight.text = if (highlightOn) "3D Light: ON" else "3D Light: OFF"`)
- 레이아웃: `android/demo-app/src/main/res/layout/activity_gpu_render.xml:474` (P5-W3 토글 그룹), `:498` (`android:text="3D Light: OFF"` 버튼)
- 처리: 로직 + 레이아웃 둘 다 제거. P5-W3-05 S1에서 셰이더 하이라이트 제거된 이후 무의미한 UI.

#### C. 블렌드 drop-down 축소
- spinner index 매핑 주석: `GpuRenderActivity.kt:514` ("blendMode 0/7은 spinner index와 1:1 매핑 — Normal=0, ColorReplace=7")
- spinner adapter items: Kotlin 또는 res/values/arrays.xml에 정의 (구현 시 정확 위치 grep)
- 현 상태 (P6 시점): 6~7개 항목 (Normal/Multiply/Screen/LTL/CRL 등)
- 목표 (P7): **3종 축소** — TintLinearV2(ID=5) / Multiply(ID=2) / ScreenLinear(ID=4)
- 기본값: TintLinearV2 ID=5 (canonical default, W2 §5.10 + P6 commit `94546a2`)
- ⚠️ **주의 (R1 Codex 보강)**: W5 A/B/C/D sclera veto 토글이 blend × veto 4조합으로 구성됨. 본 W에서 blend dropdown 3종 축소 시 A/B/C/D 토글과 정합성 확인 필요. CRL(ID=7) 슬롯이 dropdown에서 사라지면 A/B/C/D의 C/D(CRL 조합)도 동시에 정리. → **W5 Phase A/B 결정 (luma-only 유력)이 Phase C에서 최종 확정 후 처리 권장**. 단순 dropdown 축소 vs A/B/C/D 4조합 보존 trade-off.

#### D. SKU 톤 분류 정정 "누드 애쉬 로제"
- 잘못된 분류 위치 (권위 소스):
  - `docs/workPaper/P6-W9_integration.md:144` — "클라셋_누드 애쉬 로제 (웜톤)"
  - `docs/workPaper/P6-W9_integration.md:589` — 같은 SKU 재등장
  - `docs/workPaper/P6-W0_index.md` §1.5 — 동일 분류 (확인 필요)
- 정정: "웜톤" → "애쉬+누드 (회색조 중성 톤)"
- 메타: `android/demo-app/src/main/assets/lens_meta.json` — 메타에 톤 카테고리가 있다면 정정 (현재 SKU 카탈로그 메타는 `has_baked_limbal`, `prefers_crl`, `prefers_graphic_outline` 3 플래그만 — 톤 분류 없음 — 단 향후 추가 가능)
- 사용자 확정 (W9): "누드 애쉬 로제는 ash(회색조) + 누드 톤으로 웜톤 로제핑크 아님"

#### E. Phase 9 보존 산출물 cross-link
- 메모리 `phase9-deferred-tracks`에 트랙 별 재개 진입점 명시됨
- 본 W에서: P6 doc들 (W3/W4/W8 본문) §재개 진입점 섹션 cross-link 추가 또는 명시. 이미 P6-W4 §1.17, P6-W8 본문에 보존 사항 명시됨.
- 추가 cross-link: P6-W0_index.md §1.10 "Phase 6 범위 밖 이월" 표에 Phase 9 분리 명시 추가 권장

### 1.3 위험 + 롤백 경로

**위험 거의 0**:
- 모든 변경이 **시각/기능 변경 없음** — deprecated 제거, UI 제거(이미 무의미), 문서 정정
- blend dropdown 축소만 **시각 영향 가능성** — sclera veto A/B/C/D 토글과 정합성 (위 §1.2 C 참조)

**롤백 경로**:
- 각 변경을 작은 sub-commit으로 분리 → 문제 있는 항목만 단독 revert 가능
- blend dropdown 축소는 **별도 commit** 권장 (이상이 가장 가능성 높음)

### 1.4 W1과의 관계 + 분리 PR

- **P7-W0 §4 사용자 확정**: P7-W1(0x501) ↔ P7-W3 분리 PR
- W1: 셰이더 수정 + 회귀 검증
- W3: UI/KT 정리 + 문서 정정
- **각각 단독 develop 머지** (메모리 `skip-pr-for-internal-w-merges` — feature/P7-Works → develop 같은 phase-level 통합은 PR 가능, 단순 W는 직접 머지 default)
- 사용자 결정에 따름 — 본 doc에서는 옵션만 명시

### 1.5 다음 W로 들어가는 후속

- W3 완료 → cleaner demo 상태 + Phase 9 진입점 명확
- blend dropdown 3종 축소가 **W5 Phase C 후속(P7-W4 흰자 빛남)** 결과와 묶일 수 있음 (예: Phase C 결과로 CRL ID=7 슬롯 최종 폐기 확정 시) → 본 W에서 결정 보류하고 **P7-W4 후 처리**로 미루는 옵션도 가능 (Codex R1 의견)

---

## 2. 배경/맥락

### 2.1 Phase 6 종료 후 잔재

P6-W9 통합 + develop 머지(`64ba08b`) 완료 후, 다음 잔재가 남음:
- deprecated no-op 함수 (`setLensHighlight` 등) — P6-W9 §5.6 "Phase 7 초반 별도 PR로 제거" 결정
- 3D Light 버튼 — W3-04 시절 도입, P5-W3-05 S1 D5에서 셰이더 하이라이트 블록 제거된 이후 무의미
- blend dropdown — W2 블렌드 3종 확정(`94546a2`) 후 spinner items가 3종으로 정리되지 않음
- SKU 6 "누드 애쉬 로제" 톤 분류 오기 — W9 사용자 실측에서 정정 보고됨

### 2.2 P7 적정 시점

- 모든 항목 risk 0 + 변경 적음 + 사용자 관심 항목 (특히 SKU 톤 분류 정정)
- P7-W1과 병렬 가능 (다른 영역 — W1은 셰이더, W3은 UI/KT/문서)
- Phase 8(뷰티) 진입 전 demo 상태 깔끔하게 정리해두면 P8 작업 효율 ↑

### 2.3 메모리 `w9-demo-ui-sync` 정합

> "W9 통합 단계 demo UI/KT 셰이더 SDK 코어 동기화" — 이 메모리의 P7 후속 처리가 본 W.

---

## 3. 전제 조건

1. ✅ Phase 6 develop 머지 완료 (`64ba08b`)
2. ✅ P7-W0_index.md §2.P7-W3 정의 (commit `1ff3a6d`)
3. ⚠️ **P7-W4 결과 후 blend dropdown 결정 보류 가능성** — 본 W에서 dropdown 축소 미루기 옵션 검토 (§1.5 참조)
4. ✅ HIGH tier (Galaxy S23+) 보유 — W3 후 회귀 확인용

---

## 4. 목표

1. **deprecated no-op 제거** (`setLensHighlight` 등 정의 + 호출처 일괄)
2. **3D Light UI 완전 제거** (로직 + 레이아웃)
3. **blend dropdown 3종 축소 + 기본값 TintLinearV2 ID=5** (조건부 — §1.5 참조)
4. **SKU "누드 애쉬 로제" 톤 분류 정정** (P6-W9 doc + P6-W0 + 카탈로그)
5. **Phase 9 분리 cross-link** (이월 트랙 재개 진입점 명시)

### 4.1 Definition of Done

- [x] `IrisLensSDK.java` `setLensHighlight()` Java 공개 메서드 삭제 (공개 C API `iris_sdk_set_lens_highlight`는 C8 ABI 보존)
- [x] `CameraGLRenderer.kt`/`CameraGLView.kt` `setHighlight` 래퍼 체인 제거 (setLensHighlight 호출처)
- [x] `GpuRenderActivity.kt` 3D Light 토글 로직 (decl/bind/listener + `highlightOn`) 제거
- [x] `activity_gpu_render.xml` 3D Light 버튼 제거 (Ellipse 토글 유지)
- [~] blend dropdown 3종 축소 — **P7-W4 후로 보류** (사용자 확정, sclera veto A/B/C/D 정합성)
- [x] `P6-W9_integration.md:144` SKU "(웜톤)" → "(애쉬+누드 회색조 중성 톤)" 정정 (589는 단순 나열, 정정 불요)
- [x] `P6-W0_index.md` SKU 오기 부재 확인 (정정 불필요)
- [x] `P6-W0_index.md` §1.10 Phase 9 분리 cross-link 추가
- [x] Android BUILD SUCCESSFUL (시각 무변경 = 회귀 위험 0; 실기기 런타임 확인 선택)
- [x] deprecated 호출 0건 검증 (`grep setLensHighlight` = Java 공개+데모 0, C API만 유지)

### 4.2 Out of scope

- 셰이더 변경 (P7-W1)
- avg_iris_luma 측정 패스 (P7-W2)
- 흰자 빛남 알고리즘 (P7-W4)
- MID/LOW tier 검증 (P7-W5)
- 이월 트랙 W3/W4/W8 본문 수정 (Phase 9 — 분리 확정)

---

## 5. 확정 사항 (R1 합의 + 권위 소스)

### 5.1 deprecated `setLensHighlight()` 제거

**파일 + 라인**:
- 정의: `android/iris-sdk/src/main/java/com/irislenssdk/IrisLensSDK.java:1107` — 함수 본문 + `@Deprecated` 어노테이션 + javadoc 전체 삭제
- 호출처: `android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLRenderer.kt:1612` — 라인 삭제

**검증**:
```bash
grep -rn "setLensHighlight" android/ docs/ 2>&1 | grep -v "P7-W3\|MEMORY\|기타 변경 이력"
# 기대: 매치 0건 (현재 docs/와 변경 이력 외)
```

### 5.2 3D Light UI 제거

**파일 + 라인**:
- 로직 (`GpuRenderActivity.kt`):
  - line 444 (`btnToggleHighlight.text = ...`) + 주변 토글 함수 + `highlightOn` 변수
  - 관련 setOnClickListener (line 추적 필요 — `btnToggleHighlight` grep)
- 레이아웃 (`activity_gpu_render.xml`):
  - line 474~498 "Ellipse / 3D Light 토글" 블록 전체

**주의**: Ellipse 토글이 3D Light와 같은 블록에 있으므로 (line 474 주석 "Ellipse / 3D Light 토글 (P5-W3)") Ellipse 토글이 살아남는지 확인. 살아남는 경우 3D Light 부분만 제거, 죽은 경우 블록 전체 제거.

**검증**:
```bash
grep -rn "highlightOn\|btnToggleHighlight\|3D Light\|3D ?Light" android/demo-app/ 2>&1
# 기대: 매치 0건
```

### 5.3 blend dropdown 3종 축소 (조건부)

**선택 1 — P7-W3에 포함**: blend spinner items를 TintLinearV2/Multiply/ScreenLinear 3종으로 즉시 축소.

**선택 2 — P7-W4 후로 보류**: W5 Phase C(흰자 빛남) 결과로 CRL(ID=7) 슬롯 최종 폐기 확정 후 처리. A/B/C/D sclera veto 토글과 정합성 확보.

**추천**: **선택 2 (보류)** — Codex R1 보강 의견 + sclera veto A/B/C/D 토글 정합성 위험 회피. 본 W에서는 "spinner items 정리 P7-W4 후"만 명시하고 실제 코드 변경은 P7-W4 후속 작업으로 분리.

**선택 1 채택 시 절차**:
1. blend spinner adapter 위치 grep (`grep -rn "spinner.*adapter\|ArrayAdapter" GpuRenderActivity.kt`)
2. items를 ["TintLinearV2", "Multiply", "ScreenLinear"] 또는 ["TintLinearV2 (기본)", "Multiply", "Screen"] 으로 축소
3. blendMode 매핑: 0=TintLinearV2(ID=5), 1=Multiply(ID=2), 2=ScreenLinear(ID=4)
4. 기본 선택 인덱스 0 (TintLinearV2)
5. CRL ID=7 슬롯은 셰이더에서는 보존(W2 구현), spinner에서만 제거
6. A/B/C/D sclera veto 토글은 유지 (별개 축)

### 5.4 SKU "누드 애쉬 로제" 톤 정정

**대상**:
- `docs/workPaper/P6-W9_integration.md:144` — "클라셋_누드 애쉬 로제 (웜톤)"
- `docs/workPaper/P6-W9_integration.md:589` — 동일
- `docs/workPaper/P6-W0_index.md` §1.5 — 확인 후 동일 정정

**정정**:
- Before: `클라셋_누드 애쉬 로제 (웜톤)`
- After: `클라셋_누드 애쉬 로제 (애쉬+누드, 회색조 중성 톤)`

**근거**: P6-W9 사용자 실측 — "ash(회색조) + 누드 톤으로 웜톤 로제핑크 아님". `docs/workPaper/P6-W9_integration_report.md:95`에 이미 정정 사유 기록.

**메타** (`android/demo-app/src/main/assets/lens_meta.json`): 현재 톤 카테고리 필드 없음 — 변경 불필요. 향후 추가 시점에 반영.

### 5.5 Phase 9 분리 cross-link

**대상**: `docs/workPaper/P6-W0_index.md`

**추가 섹션** (§1.10 또는 §1.11 신규):

```markdown
### 1.X Phase 9 이월 트랙 (2026-06-04 분리 확정)

P6-W3/W4/W8 이월 트랙은 Phase 9(또는 7.5)로 분리. P8 뷰티 우선.

- **W3** 환경 반사 scaffold: OFF 기본 보존 (메모리 [[w4-env-reflection-deferred]])
- **W4** B2 24클립 벤치: 인프라 보존 (`docs/bench/P6-W4/`)
- **W8** Pupil material: Option E 설계 보존 (`P6-W8_*.md`)

재개 시점: Phase 8 완료 후 비즈니스 임팩트 재평가. 진입점: 메모리 [[phase9-deferred-tracks]].
```

### 5.6 회귀 검증 (HIGH tier S23+)

W3 완료 후:
- 빌드 + 설치 (`./scripts/build_and_install.sh`)
- 데모 UI 시각 확인: 3D Light 버튼 없음, blend dropdown 정합성 (선택 1 채택 시), Phase 9 cross-link 문서 확인
- 6 SKU × 5축 회귀 (Phase 6 패턴 동일 — 시각 결과 동일해야 함)
- `grep -rn "setLensHighlight" android/`로 deprecated 호출 0건 확인

### 5.7 커밋 분할 (제안)

**3 sub-commit (또는 1 통합)**:

1. `refactor(sdk): P7-W3 deprecated setLensHighlight 완전 제거`
   - IrisLensSDK.java 정의 + CameraGLRenderer.kt 호출처
2. `refactor(demo): P7-W3 3D Light UI + Ellipse 토글 정리`
   - GpuRenderActivity.kt 로직 + activity_gpu_render.xml 레이아웃
3. `docs(P7-W3): SKU 누드 애쉬 로제 톤 정정 + Phase 9 분리 cross-link`
   - P6-W9_integration.md + P6-W0_index.md
4. (조건부) `refactor(demo): P7-W3 blend dropdown 3종 축소` — 선택 1 채택 시
5. (선택) 1 통합 commit `chore: P7-W3 A 그룹 cleanup 일괄`

**추천**: 3 sub-commit 분할 (각 변경 단독 revert 가능). blend dropdown은 별도 (조건부).

---

## 6. 미결 사항

### 6.1 R1 합의 표

| 번호 | 쟁점 | 상태 | 출처 |
|---|---|---|---|
| 6.1.1 | A 그룹 cleanup 범위 (4가지) | ✅ 닫힘 (3/3 합의) | P7-W0 §2.P7-W3 |
| 6.1.2 | P7-W1과 분리 PR | ✅ 닫힘 (사용자 확정) | P7-W0 §4 |
| 6.1.3 | blend dropdown 축소 시점 | ⚠️ **사용자 판단** — 선택 1(W3에 포함) vs 선택 2(W4 후 보류) | §5.3 |

### 6.2 사용자 최종 판단 필요

- **blend dropdown 축소를 본 W에 포함할지** — 추천: 선택 2 (보류, P7-W4 후) — sclera veto A/B/C/D 토글과 정합성 위험 회피

---

## 7. 체크리스트 (구현자용)

### 7.1 읽을 파일

- `android/iris-sdk/src/main/java/com/irislenssdk/IrisLensSDK.java` (`setLensHighlight` 정의 근처)
- `android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLRenderer.kt` (line 1612 근처)
- `android/demo-app/src/main/java/com/irislenssdk/demo/GpuRenderActivity.kt` (line 444 + spinner 매핑 line 514)
- `android/demo-app/src/main/res/layout/activity_gpu_render.xml` (line 474~498)
- `docs/workPaper/P6-W9_integration.md` (line 144, 589)
- `docs/workPaper/P6-W0_index.md` (§1.5 SKU + §1.10 이월)

### 7.2 수정할 파일

- `IrisLensSDK.java` (deprecated 함수 정의 삭제)
- `CameraGLRenderer.kt` (호출처 삭제)
- `GpuRenderActivity.kt` (3D Light 로직 삭제)
- `activity_gpu_render.xml` (3D Light UI 삭제)
- `P6-W9_integration.md` (SKU 톤 정정)
- `P6-W0_index.md` (SKU 톤 정정 + Phase 9 cross-link)
- (조건부) blend spinner adapter (선택 1 채택 시)

### 7.3 검증 명령

```bash
# deprecated 호출 0건 확인
grep -rn "setLensHighlight" android/ docs/workPaper/P7-W3* | grep -v "변경 이력"

# 3D Light 잔재 확인
grep -rn "3D Light\|highlightOn\|btnToggleHighlight" android/demo-app/

# 빌드
cd android && ./gradlew :iris-sdk:assembleDebug :demo-app:assembleDebug 2>&1 | tail -5

# 설치
./scripts/build_and_install.sh
```

### 7.4 회귀 시 즉시 행동

- 시각 회귀 발견 → 해당 sub-commit 단독 revert
- blend dropdown 정합성 문제 → 즉시 선택 2 (P7-W4 후 처리)로 분리

---

## 8. 완료 정의 + 다음 W

### 8.1 완료 정의

§4.1 DoD 9~10개 항목 + 빌드 + 회귀 통과 + `P7-W0_index.md` §2.P7-W3 상태 ✅.

### 8.2 커밋 전략

- 3 sub-commit 분할 (deprecated 제거 / 3D Light UI / docs 정정) — 각 단독 revert 가능
- 또는 1 통합 commit `chore: P7-W3 A 그룹 cleanup 일괄`
- 분리 PR — P7-W1과 별도 (P7-W0 §4 사용자 확정)
- develop으로 직접 머지 (메모리 `skip-pr-for-internal-w-merges` default)

### 8.3 다음 W

- 본 W 종결 시점이 W1 결과와 무관 (병렬 가능)
- **P7-W2** (avg_iris_luma): W1 안정 확인 후 (병렬 X)
- **P7-W4** (흰자 빛남): W2 후, blend dropdown 축소 미뤘다면 W4 결정 후 본 W 재방문 (또는 별도 cleanup commit)

### 8.4 P7 인덱스 갱신

W3 완료 시 `P7-W0_index.md` §2.P7-W3에 상태 ✅ + 결과 1줄 추가 + commit SHA 기록.

---

## 변경 이력

| 날짜 | 변경 |
|---|---|
| 2026-06-04 | 초안 작성. P7-W0 R1 합의 + 권위 소스 직접 확인 반영. `ar-lens-implement` 호출 대기. |
| 2026-06-10 | **구현 완료**. deprecated `setLensHighlight` Java 공개+데모 체인 제거(C8 C API 보존), 3D Light UI 제거(Ellipse 유지), SKU 누드 애쉬 로제 톤 정정, Phase 9 cross-link(P6-W0 §1.10). blend dropdown은 P7-W4 후 보류. Android BUILD SUCCESSFUL. 라인 앵커 W2로 이동분 재확인(setLensHighlight 1107→1118). |
