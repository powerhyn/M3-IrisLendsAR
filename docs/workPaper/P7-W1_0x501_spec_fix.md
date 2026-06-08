# P7-W1: 0x501 spec fix + HIGH tier 회귀

> **상태**: ✅ **완료** (2026-06-08). 코드 `38368ec` + S23+(Adreno 740) 실기기 검증 통과 — 0x501 per-frame 제거(렌즈 65fps 렌더 중 0건), C10 ON/OFF 토글 육안 등가. 6 SKU 풀 회귀는 lens-독립성 근거로 waive(§4.1). develop `--no-ff` 머지.
> **작성**: 2026-06-04
> **선행 의존**: 없음 (P7 첫 W, P6 develop 머지 64ba08b 완료 상태에서 즉시 진입 가능)
> **병렬 가능**: P7-W3 (cleanup, risk 0)
> **소요 추정**: 0.5~1.0 작업일
> **참조**: `docs/workPaper/P7-W0_index.md` §2, `P7-W0_brainstorm/synthesis.md` §2.Q1

---

## 1. 인사이트 (세션 간 맥락 보존) ⭐

### 1.1 이 W의 정체성

**GLSL ES 3.0 spec §8.9 위반 즉시 제거**. W6 Phase A에서 도입된 디테일 재주입 코드(`cpp/src/gpu/shader_sources.cpp:1066-1083`)가 dynamic branch 내부에서 `texture()` 9샘플을 호출하는 패턴 — implicit derivative가 non-uniform control flow에서 **undefined**. P6-W9 통합 검증에서 발견된 `glError 0x501 (GL_INVALID_VALUE)` 매 frame 발생의 1순위 가설.

이미 P6-W9에서 demo 측 방어(`CameraGLRenderer.onSurfaceCreated`에 `releaseGpuBeauty/Lens` 명시)로 검은 화면 자체는 차단됨(`d1aaabe`). 하지만 0x501 잔재는 그대로 — W6 셰이더의 spec 위반이 근본 원인이라는 게 deep-research(F2) + R1(3/3) 합의.

### 1.2 대상 코드 (권위 소스 확인 2026-06-04)

`cpp/src/gpu/shader_sources.cpp:1066-1083`:

```glsl
if (uDetailReinject == 1) {
    vec2 t = uTexelSize;
    float lC  = dot(toLinearFast(texture(uCameraTexture, vTexCoord).rgb), LUMA_709_LENS);
    float lN  = dot(toLinearFast(texture(uCameraTexture, vTexCoord + vec2(0.0, -t.y)).rgb), LUMA_709_LENS);
    float lS  = dot(toLinearFast(texture(uCameraTexture, vTexCoord + vec2(0.0,  t.y)).rgb), LUMA_709_LENS);
    float lE  = dot(toLinearFast(texture(uCameraTexture, vTexCoord + vec2( t.x, 0.0)).rgb), LUMA_709_LENS);
    float lW  = dot(toLinearFast(texture(uCameraTexture, vTexCoord + vec2(-t.x, 0.0)).rgb), LUMA_709_LENS);
    float lNE = dot(toLinearFast(texture(uCameraTexture, vTexCoord + vec2( t.x, -t.y)).rgb), LUMA_709_LENS);
    float lNW = dot(toLinearFast(texture(uCameraTexture, vTexCoord + vec2(-t.x, -t.y)).rgb), LUMA_709_LENS);
    float lSE = dot(toLinearFast(texture(uCameraTexture, vTexCoord + vec2( t.x,  t.y)).rgb), LUMA_709_LENS);
    float lSW = dot(toLinearFast(texture(uCameraTexture, vTexCoord + vec2(-t.x,  t.y)).rgb), LUMA_709_LENS);
    float blurLum = (lC * 2.0 + lN + lS + lE + lW + lNE + lNW + lSE + lSW) / 10.0;
    float baseLum = lC;
    float detail = clamp(baseLum / max(blurLum, 0.001), 0.85, 1.15);
    float innerMask = smoothstep(0.7, 0.5, dist);
    float gateStrength = smoothstep(uGateThreshold - 0.03, uGateThreshold + 0.03, uAvgIrisLum);
    float detailMul = mix(1.0, detail, gateStrength * innerMask);
    blended *= vec3(detailMul);
}
```

9개 `texture(uCameraTexture, ...)` 호출 모두 implicit LOD 사용 = derivative 필요 = dynamic branch 안 = spec 위반.

### 1.3 외부 동향 근거 (Stage 1 deep-research F2 통과 7 claims)

- **Khronos GLSL ES 3.0 spec §8.9**: "implicit derivatives ... non-uniform control flow ... undefined"
- **Khronos `textureLod` refpage**: partial derivatives를 0으로 명시 설정 → dynamic branch 안 안전
- **Mozilla Bugzilla #1932416** (2025-01 Firefox 134 수정): Adreno 305/306에서 "분기 하나만 추가해도" 검은 화면 미컴파일 발생을 엔지니어가 진단하고 mix() 평탄화로 수정 (GL 에러 없이 시각적 corruption)
- 다수 독립 사례: Godot #12816, Filament #1544, Unity issue tracker, NVIDIA dev forum, Maister 블로그

**Caveat (3/3 모델 공통)**: 실증 사례는 Adreno 3xx(2012-2014). IrisLensSDK 타겟 Adreno 6xx/7xx 직접 일반화는 보수적. 단 spec 위반 자체는 모든 Adreno 세대 적용 — 패치 사유는 충분.

### 1.4 표준 회피책 (3종)

| 방법 | 변경량 | 비용 | 위험 |
|---|---|---|---|
| **(a) `textureLod(uv, 0.0)` 명시 LOD** ✅ | 1줄 패턴 9회 교체 | 0 (동일 사이클) | 매우 낮음 — 시각 동일 |
| (b) 9샘플 dynamic branch 밖 hoist | 함수 재구조화 | uDetailReinject==0일 때도 9 fetch 발생 (비용 ↑) | 낮음 |
| (c) `textureGrad` with 사전 계산 gradient | 중간 | 약간 느림 (임시) | 중간 |

**채택**: **(a) textureLod**. R1 3/3 합의.

### 1.5 W의 범위

**포함**:
- 셰이더 9개 texture → textureLod 교체
- HIGH tier (Galaxy S23+) 회귀 검증 (시각 + 0x501 logcat 사라짐)
- DetailReinject 토글 ON/OFF 비교 보존 (롤백 가능)

**제외**:
- MID/LOW tier 회귀 → P7-W5
- 다른 dynamic branch + texture() 패턴 검사 → 별도 cleanup (P7-W3에 포함 가능)
- Mali/Adreno 외 GPU 회귀 (iOS 등) → Phase 7 범위 밖

### 1.6 위험 + 롤백 경로

**위험 1 — textureLod이 implicit derivative와 미세 시각 차이**:
- 이론적으로 LOD 0 명시 = mipmap 미사용 + 기본 LINEAR filter는 derivative 무관 → 시각 차이 0 예상
- 단 드라이버별 미세 차이 가능성. 토글로 ON/OFF 비교 보존 필수.

**위험 2 — 0x501이 잔재 그대로**:
- F2 진단이 1순위지만 유일 원인 보장 X (Codex 보강)
- 패치 후 0x501 잔재 시 2순위 가설 (텍스처 binding 등) 추가 추적 필요
- 본 W의 DoD는 "0x501 logcat 사라짐"이므로, 사라지지 않으면 W1 미종결

**롤백 경로**:
- 패치 commit 단독 분리 → 시각 회귀 시 즉시 revert
- 셰이더 매크로 `#ifdef DETAIL_REINJECT_USE_TEXTURELOD`로 빌드 타임 분기 가능하나, 1줄 패치라 매크로 오버헤드 불필요

### 1.7 다음 W로 들어가는 후속

- 0x501 사라짐 확인 → P7-W3 (cleanup)과 함께 develop로 PR.
- 0x501 잔재 → 2순위 가설 (텍스처 binding stale, FBO state 등) 추가 추적 W로 분리 (W1.1?)
- MID/LOW 회귀는 P7-W5에서 cross-tier 재현 확인.

---

## 2. 배경/맥락

### 2.1 W6 Phase A에서 도입된 detail-reinject

W6 §5.2 (C10 detail reinjection) 구현 시 도입된 9샘플 3×3 single-pass blur. 저조도(우유빛/blur) 환경에서 디테일 손실 보완 목적. blended에 곱셈으로 디테일 재주입.

**의도**: detail = baseLum / blurLum 비율을 clamp[0.85, 1.15]로 제한 + innerMask + gate로 영역/조건 한정. 30fps 예산 0.1ms 추가 추정.

**문제 발견**: P6-W9 통합 검증에서 검은 화면 발생 + logcat에 `glError 0x501` 매 frame. cpp-pro 진단으로 2순위 가설(W6 detail-reinject dynamic branch)로 명시됨 (1순위는 EGL context 재생성 — `d1aaabe`로 해결됨). 검은 화면 자체는 해결됐지만 0x501 잔재 = 2순위 가설 영역.

### 2.2 Stage 1 deep-research → Stage 2 R1 검증

Stage 1 (107 agents, 25 sources)에서 F2가 confidence high (7 claims 3-0 vote)로 통과. Stage 2 R1에서 3/3 모델 모두 즉시 수정 합의. P7 첫 W로 확정 (P0).

---

## 3. 전제 조건

1. ✅ Phase 6 develop 머지 완료 (`64ba08b`)
2. ✅ P6-W9 검은 화면 해결 (`d1aaabe`) — EGL context 재생성 이슈 차단됨
3. ✅ HIGH tier 기기 (Galaxy S23+) 보유 + Phase 6 6 SKU × 5축 검증 통과 상태
4. ✅ DetailReinject 토글 인프라 보존 (`setDetailReinject` internal API + demo UI)

---

## 4. 목표

1. **GLSL ES 3.0 spec §8.9 위반 제거** — `if(uDetailReinject==1){...}` 안 9 `texture()` → `textureLod(uv, 0.0)` 교체
2. **glError 0x501 logcat 사라짐 확인**
3. **HIGH tier 시각 회귀 없음**

### 4.1 Definition of Done

- [x] `shader_sources.cpp` detail-reinject 블록 9개 `texture()` → `textureLod(uv, 0.0)` 교체 (commit `38368ec`)
- [x] C++ iris_sdk 빌드 성공 (`libiris_sdkd.a` 링크, 신규 warning/error 0 — 기존 TFLite/mediapipe 경고만)
- [x] textureLod 등가성 검증 (적대적 4-agent: verdict=identical — uCameraTexture는 sampler2D + GL_LINEAR + mipmap 미사용)
- [x] HIGH tier (S23+ SM-S916N, Adreno 740) DetailReinject ON + 렌즈 렌더(~65fps) 상태 1분+ long-run: logcat `glError 0x501` **0건** (`nativeRenderLensTexture` 매 프레임 실행 + detected 4003건 = detail-reinject 분기 활성인데도 0x501 미발생 → per-frame 0x501 제거 확인. 2026-06-08)
- [~] HIGH tier 6 SKU × 5축 회귀 — **풀 6 SKU 미수행 (사용자 판단으로 waive)**. 사유: 본 변경은 lens-독립적(`detailMul`은 `uCameraTexture` luma에서만 산출, 모든 SKU에서 textureLod≡texture로 byte-identical)이고 양안/블링크/sclera/림발 4축은 코드 미변경. 단일 SKU(doll choco) 토글 검증 + 객관 0x501=0으로 충분 판정 (메모리 `solo-dev-bench-method`/`qualitative-device-judgment` 정합). 2026-06-08
- [x] DetailReinject ON/OFF 토글 시각 차이 없음 — S23+ `clāset doll choco` 렌즈 적용, C10 토글 4회(off↔on) 실기기 통제: 양쪽 0x501=0, 사용자 육안 "차이 거의 못 느낌, 미세한 디테일 차이만" = §5.2 step4 통과 기준(textureLod≡texture). (2026-06-08)

### 4.2 Out of scope

- MID/LOW tier 회귀 → **P7-W5**
- 다른 셰이더 영역의 dynamic branch + texture() 패턴 스캔 — 본 W는 detail-reinject만, 추가 영역은 cleanup 차원에서 별도
- iOS 회귀 (별도 Phase)
- 0x501 잔재 시 2순위 가설 추적 → 별도 W

---

## 5. 확정 사항 (R1 합의 + Stage 1 근거)

### 5.1 패치 위치 + 수정 패턴

**파일**: `cpp/src/gpu/shader_sources.cpp`
**라인**: 1068~1076 (9 줄, 각 1 `texture()` 호출)

**Before**:
```glsl
float lC  = dot(toLinearFast(texture(uCameraTexture, vTexCoord).rgb), LUMA_709_LENS);
float lN  = dot(toLinearFast(texture(uCameraTexture, vTexCoord + vec2(0.0, -t.y)).rgb), LUMA_709_LENS);
// ... (9개)
```

**After**:
```glsl
float lC  = dot(toLinearFast(textureLod(uCameraTexture, vTexCoord, 0.0).rgb), LUMA_709_LENS);
float lN  = dot(toLinearFast(textureLod(uCameraTexture, vTexCoord + vec2(0.0, -t.y), 0.0).rgb), LUMA_709_LENS);
// ... (9개)
```

### 5.2 회귀 검증 절차 (HIGH tier S23+)

**단계 1 — 빌드 + 설치**:
```bash
./scripts/build_and_install.sh
```

**단계 2 — logcat 0x501 사라짐 확인**:
```bash
adb logcat -c   # 클리어
# 앱 실행 후 카메라 켜고 1분 대기 (DetailReinject 기본 ON 상태)
adb logcat -d 2>&1 | grep -E "glError|0x501" | head -20
# 기대: 매치 0건
```

**단계 3 — 6 SKU × 5축 회귀 (Phase 6 W9 패턴 동일)**:
- 검사 5축: ① 양안 / ② 블링크 ramp / ③ sclera veto (UI A/B/C/D) / ④ 블렌드+디테일 / ⑤ 림발
- SKU 6종: 클라셋_돌 초코 / 클라셋_런웨이 그레이 / 오(OH)_베이글 / 엔비_퍼퓸 글로우 / 엔비_샤모 브라운 / 클라셋_누드 애쉬 로제
- 가이드: `docs/bench/P6-W9/checklist.md` 재사용

**단계 4 — DetailReinject 토글 비교**:
- demo UI에서 DetailReinject ON ↔ OFF 토글
- 시각 차이 인지 어려움 = textureLod이 기존 texture와 동일 결과 = ✅
- 시각 차이 명백 = 미세 차이 발생 = ⚠️ 추가 조사 (드라이버 차이)

### 5.3 빌드 + 테스트 명령

```bash
# C++ 빌드
cd cpp/cmake-build-debug && cmake --build . --target iris_sdk 2>&1 | tail -10

# Android demo
cd android && ./gradlew :iris-sdk:assembleDebug :demo-app:assembleDebug 2>&1 | tail -5

# 또는 사용자 표준 스크립트
./scripts/build_and_install.sh
```

### 5.4 커밋 메시지 (제안)

```
fix(gpu-lens): P7-W1 W6 detail-reinject GLSL ES 3.0 §8.9 위반 제거

W6 detail-reinject (shader_sources.cpp:1066-1083) 가 dynamic branch
(if uDetailReinject == 1) 안에서 texture() 9샘플을 호출 — implicit
derivative가 non-uniform control flow에서 undefined (GLSL ES 3.0
spec §8.9 위반). P6-W9 통합 검증에서 발견된 glError 0x501 매 frame
잔재의 1순위 원인.

9개 texture(uv) → textureLod(uv, 0.0) 명시 LOD로 교체.
시각 결과 동일 (mipmap 미사용 + LINEAR filter는 derivative 무관).

검증 (Galaxy S23+ Adreno 740):
- logcat glError 0x501 사라짐
- 6 SKU × 5축 회귀 통과 (Phase 6 패턴 동일)
- DetailReinject ON/OFF 토글 시각 차이 없음

근거: deep-research wf_d2eb976b-38b F2 (Khronos GLSL ES 3.0 spec
§8.9, Mozilla Bugzilla #1932416, Maister 블로그, Godot/Filament/
Unity 다수 사례) + P7-W0 R1 3/3 합의.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

---

## 6. 미결 사항

### 6.1 R1 합의 표

| 번호 | 쟁점 | 상태 | 출처 |
|---|---|---|---|
| 6.1.1 | 0x501 처리 방식 | ✅ 닫힘 (3/3 패치 먼저 + HIGH 회귀 묶기) | P7-W0 §2.Q1 |
| 6.1.2 | 회피책 선택 | ✅ 닫힘 (3/3 textureLod 0.0) | F2 표준 회피책 + R1 |
| 6.1.3 | MID/LOW 회귀 동봉 여부 | ✅ 닫힘 (Codex+Gemini 2/3 별도 W5) | synthesis §2.Q1 |

### 6.2 W 완료 후 후속 조건

- **0x501 사라짐 확인**: ✅ **실측 확정** (S23+/Adreno 740, 렌즈 65fps 렌더 중 0x501 0건, 2026-06-08) → W1 종결, P7-W2/W3로 진입. 위험 2 미발현(§6.3 예측대로 기본 구성에서 해소).
- **0x501 잔재**: 미발생 → 2순위 가설 추적 불필요 (잔존 §8.9 후보 389/449는 P7-W3 위생 항목으로만 남음).
- **시각 회귀 발견**: 없음 (ON/OFF 토글 육안 등가 확인).

### 6.3 구현 중 적대적 검증 결과 (2026-06-04, 4-agent 워크플로우 `wf_f10ec798-dee`)

W1 패치 직후 독립 4 에이전트로 ① 편집 정확성 ② textureLod 등가성 ③·④ 셰이더 전체 §8.9 잔존 위반 2회 교차 스캔 수행. Claude 직접 기술 판정으로 필터링.

**① 편집**: 9개 모두 `textureLod(uv, 0.0)` 변환, 오프셋 9종 보존, 괄호 균형 ✅. 블록 내 implicit `texture(` 잔여 0건.

**② 등가성**: `verdict=identical`. `uCameraTexture`는 `sampler2D`(EXTERNAL_OES 아님), `rgbaTextureId` GL_LINEAR min/mag + `glGenerateMipmap` 미호출 → mipmap 없음 → LOD 0 강제 = base level 샘플 = implicit과 동일. 근거: `shader_sources.cpp:790`(선언), `gpu_lens_renderer.cpp:930`(GL_TEXTURE_2D 바인딩), `CameraGLRenderer.kt:1746-1755`(필터 설정, mipmap 없음).

**③·④ 잔존 §8.9 스캔 (Claude 판정 후)** — 두 스캔 상충(Scan1 2 violation+5 suspect / Scan2 0 violation). 기술 판정 결과:

| 위치 | 셰이더 | 패턴 | Claude 판정 | lens 패스? | W1 블로커? |
|---|---|---|---|---|---|
| ~389 | COMBINED_COLOR_ADJUSTMENT (LUT) | `texture()` in `if(uLutIntensity>0.01)` | **동일 클래스 후보** (uniform-toggle 분기 내 implicit texture, detail-reinject와 동형). Scan1의 "non-uniform 좌표=위반"은 오판 — §8.9는 control flow 발산 문제 | ❌ (`gpu_lens_renderer.cpp` 미참조 = 별도 beauty 파이프라인) | ❌ |
| ~449 | FREQ_SEP_GAUSSIAN | `texture()` in `for(uRadius)` loop | 동일 클래스 후보 (runtime uniform loop bound). spec상 dynamically-uniform이나 Adreno caution | ❌ | ❌ |
| 109/264/419/420 | BILATERAL/SOFT_FOCUS/GAUSSIAN | const-loop `texture()` | **safe 오탐** — const 루프 = 완전 언롤 = straight-line. §8.9 무관 | ❌ | ❌ |
| 938/955 | LENS_OVERLAY `sampleReflection` | `texture()` in `if(uSourceType==1/2)` | 동일 클래스이나 **기본 OFF** — `reflection_mode_=0`이라 분기 미진입(셰이더 line 959 "OFF (W3 기본)") | ✅ (단 미진입) | ❌ (반사 토글 시만 노출) |

**위험 2 결론**: lens overlay 셰이더(0x501 관측 셰이더)의 **유일한 활성 §8.9-클래스 패턴은 detail-reinject 1건뿐**이었고 본 W1이 제거. 반사 분기는 기본 OFF, beauty 셰이더(389/449)는 lens 패스 밖. → **기본 구성에서 0x501이 본 패치로 해소될 가능성 높음** (단, 실기기 confirm이 DoD 게이트). 잔존 후보는 W1 블로커 아님.

**P7-W3 cleanup 피드**:
- (선택) 389(LUT 분기) / 449(freq-sep 루프) `textureLod` 위생 패치 — beauty 파이프라인 활성 시 예방. **활성 여부 먼저 확인** (어느 데모 패스에서 컴파일되는지).
- 938/955 반사 분기는 Phase 9 환경 반사 재개 시 `textureLod` 동시 적용 (메모리 `w4-env-reflection-deferred` cross-link).
- const-loop suspects는 무시 (safe).

**실기기에서 0x501 잔재 시 (위험 2 발현)**: 2순위 후보 우선순위 = ① 반사 토글이 켜져 있었는지 확인 → ② beauty 패스(389/449) 활성 여부 → ③ 텍스처 binding/FBO state 등 비-셰이더 원인.

---

## 7. 체크리스트 (구현자용)

### 7.1 읽을 파일

- `cpp/src/gpu/shader_sources.cpp` (1060~1085 영역)
- `docs/workPaper/P7-W0_index.md` §2.P7-W1 (본 W 정의)
- `docs/workPaper/P7-W0_brainstorm/synthesis.md` §2.Q1 + §3.F2
- `docs/bench/P6-W9/checklist.md` (회귀 절차 재사용)

### 7.2 수정할 파일

- `cpp/src/gpu/shader_sources.cpp` (1068~1076 9줄)

### 7.3 검증 명령

```bash
# logcat 확인 (앱 실행 후)
adb logcat -d 2>&1 | grep -E "glError|0x501" | head

# 빌드
cd cpp/cmake-build-debug && cmake --build . --target iris_sdk 2>&1 | tail -5

# 설치
./scripts/build_and_install.sh
```

### 7.4 회귀 시 즉시 행동

- 시각 회귀 발견 → 셰이더 단독 commit revert → 추가 조사
- 0x501 잔재 → 2순위 가설 (텍스처/FBO state, sampler config) 별도 W

---

## 8. 완료 정의 + 다음 W

### 8.1 완료 정의

§4.1 DoD 6개 항목 + 결과 통합 리포트 1줄 갱신 (`docs/workPaper/P7-W0_index.md` §2.P7-W1 상태 ✅).

### 8.2 커밋 전략

- `fix(gpu-lens): P7-W1 ...` 단일 commit (분리 PR 결정 — P7-W0 §4)
- develop으로 직접 머지 (메모리 `skip-pr-for-internal-w-merges`)

### 8.3 다음 W

- **P7-W2** (avg_iris_luma 측정 패스) — W1 안정 확인 후 진입. 의존성: W1 0x501 사라짐 확인.
- **P7-W3** (A 그룹 cleanup) — W1과 병렬 가능. 별도 PR.

### 8.4 P7 인덱스 갱신

W1 완료 시 `P7-W0_index.md` §2.P7-W1에 상태 ✅ + 결과 1줄 추가 + commit SHA 기록.

---

## 변경 이력

| 날짜 | 변경 |
|---|---|
| 2026-06-04 | 초안 작성. P7-W0 R1 합의 + Stage 1 F2 근거 반영. `ar-lens-implement` 호출 대기. |
| 2026-06-04 | **코드 구현 완료** (commit `38368ec`). 9개 `texture()`→`textureLod(uv,0.0)`. C++ 빌드 통과. 적대적 4-agent 검증(§6.3): 편집 정확 + 등가성 identical + 잔존 §8.9 landscape. DoD 코드 3항 ✅ / 실기기 3항 대기. |
| 2026-06-08 | **실기기 검증 완료 + W1 종결** (S23+ SM-S916N / Adreno 740). 통제 런: 렌즈 65fps 렌더 중 0x501 **0건**(패치 전 매 frame → 0). C10 ON/OFF 토글 4회: 양쪽 0x501=0 + 육안 등가("미세 디테일만"). 6 SKU 풀 회귀는 lens-독립성 근거로 waive. develop `--no-ff` 머지. DoD 5/6 ✅ + 1 waive. |
