# P7-W4: 흰자 빛남 1단계 — sclera luma attenuation (W5 Phase C 트랙)

> **상태**: ✅ 브레인스토밍 R1 종료 + §5 확정 반영 (2026-06-10, 사용자 승인) — 3/3 합의 5건 + 2/3 다수결 1건(6.2→(c)), R2 불필요. 구현 대기 (`ar-lens-implement`). 첫 작업: 코드 전 S23+ 재베이스라인 (§5.7, 두 조명 조건).
> **작성**: 2026-06-10
> **선행 의존**: ✅ P7-W2 완료 (avg_iris_luma 실측 default ON, develop 머지 `5dc1a04`)
> **소요 추정**: 2.0~4.0 작업일 (P7-W0 §1.5)
> **참조**: `P7-W0_index.md` §2.P7-W4, `P6-W9_integration_report.md` §3.1, 메모리 `w5-b1-tintlinearv2-strength`

---

## 1. 인사이트 (세션 간 맥락 보존) ⭐

### 1.1 이 W의 정체성

TintLinearV2(canonical default, ID=5)가 **밝은 렌즈 SKU에서 sclera(흰자) 영역을 과증폭**하는 "흰자 빛남"의 1차 완화. P6-W5 1차 형광 벤치에서 발견되어 Phase C로 분리된 트랙 (메모리 `w5-b1-tintlinearv2-strength`). Oklab+Laplacian 본질 해결책은 본 W 결과 부족 시에만 P7-Spike-A로 진입 (P7-W0 R1 F1 정정: "본질 해결책" 표현 과함 → 단계적 접근).

### 1.2 흰자 빛남 메커니즘 (코드 실측 2026-06-10)

**TintLinearV2 현재 수식** (`shader_sources.cpp:875-882`):
```glsl
vec3 blendTintLinearV2(vec3 base, vec3 blend, float opacity) {
    vec3 baseL = toLinearFast(base);
    float lum = dot(baseL, LUMA_709_LENS);                      // 카메라 픽셀 linear Rec.709
    float scale = clamp(0.85 / max(0.01, uAvgIrisLum), 0.8, 7.0);
    vec3 tinted = toLinearFast(blend) * lum * scale;
    vec3 result = mix(baseL, tinted, opacity);
    return toSRGBFast(result);
}
```

- `tinted = lensL · lum · (0.85/avgLum)` = `lensL · 0.85 · (lum/avgLum)` — **brightness-invariant 틴트** 설계 (W2 §5.6).
- **iris 내부**: `lum ≈ avgLum` → 비율 ≈ 1 → 틴트 강도 0.85 근방. 의도대로.
- **sclera 픽셀**: 흰자 `lum ≈ 0.6~0.9` ≫ 홍채 `avgLum ≈ 0.07~0.6` → **비율이 상한 없이 수 배로 증폭** → 밝은 SKU(그레이/누드 애쉬)에서 렌즈색이 흰자 위에서 빛남.
- 렌즈 에셋은 의도적으로 iris보다 큼(서클렌즈 직경 > 홍채) + eyelid feather → **sclera 위 렌즈 픽셀은 구조적으로 항상 존재**. `scale` 자체는 clamp되지만 `lum` 항은 clamp 없음 — 빛남의 직접 원인.

### 1.3 ⚠️ P7-W0 인덱스 수식 인용 정정 (권위 소스 확인)

인덱스 §2.P7-W4가 인용한 `out = lens * detail`, `detail = clamp(pow(lum/avgLum, 0.7), 0.75, 1.25)`는 **ColorReplaceLinear(ID=7) 수식**이다 (`shader_sources.cpp:885-892`). W9에서 흰자 빛남 주범으로 실측 지목된 것은 **TintLinearV2(ID=5)** — 실제 수식은 §1.2가 맞다. luma 기반 attenuation 개념 자체는 동일하게 적용 가능하므로 설계 방향은 유효. (메모리 `verify-loadbearing-facts` 적용)

### 1.4 기존 감쇄 계층 인벤토리 — 신규 계층이 무엇과 직교해야 하나

| 계층 | 감쇄 대상 | 게이트 | 현재 상태 |
|---|---|---|---|
| sclera veto (`uScleraProtect`, `shader_sources.cpp:1014-1036`) | **finalAlpha** | geometry(`smoothstep(0.75,1.0,irisEdgeDist)`) × 색/luma veto | 기본 ON, **`uScleraVetoMode=0`(legacy)** — P6-W5 B8 판정 미실행, 3-way 토글 보존 |
| CRL maxDetail edge 감쇄 (`:1042`) | detail 상한 → 1.0 | geometry(동일 smoothstep) | ID=7 한정 |
| eyelidMask / edgeAlpha (`:1011`) | finalAlpha | geometry | — |
| **(신규 제안) luma attenuation** | **틴트 색** | **luma**(`smoothstep(0.65,0.85,lum)`) | 본 W |

핵심 차이: 기존 계층은 전부 **geometry-gated alpha 감쇄**, 신규 제안은 **luma-gated color 감쇄**. 따라서:
- iris **내부**의 밝은 픽셀(각막 반사 하이라이트, 밝은 홍채)에도 걸릴 수 있음 → 부작용 후보 (§6.5).
- sclera 영역에선 기존 veto(알파)와 **이중 감쇄** 가능 (§6.4).

### 1.5 ⚠️ 적용 위치 제약 — post-blend 일괄 감쇄 불가 (코드 실측)

블렌드 함수들은 내부에서 `mix(baseL, tinted, opacity)`로 **카메라와 이미 합성된 색**을 반환한다 (`blended`가 곧 최종 픽셀, `:1134` `return vec4(blended, camera.a)`). 따라서 인덱스 의사코드처럼 블렌드 **결과**에 `mix(out, out*uScleraAttenuation, sclera_factor)`를 적용하면 **카메라 원본(흰자 자체)까지 어두워진다**. 감쇄는 **블렌드 함수 내부의 `tinted` 항** (또는 opacity 항)에 적용해야 함. 구현 형태는 §6.2.

### 1.6 베이스라인 문제 — W9 관측은 W2 교정 **이전** 데이터

- W9 흰자 빛남 실측(강도 패턴 `1 < 4 ≈ 3 < 5 < 2 ≤ 6`, `P6-W9_integration_report.md` §3.1)은 **fallback 0.1225 시절** = 밝은 환경에서 scale 7.0 포화 over-tint 상태에서 관측됨.
- P7-W2에서 실측 연결로 밝은 환경 scale **7.0 → 1.27** 교정 + "육안 더 자연스러움" 확인 (W2 §4.1).
- → **흰자 빛남이 이미 부분 완화됐을 가능성**. sclera 픽셀 증폭비도 fallback 시절 `lum·7.0`에서 실측 시 `0.85·(lum/avgLum)`로 바뀌어 **조명 의존적**이 됨 (어두운 환경 avgLum~0.07이면 여전히 clamp 7.0 영역).
- → W4 착수 전 **재베이스라인 실기기 확인 필수** (메모리 `real-data-first`). DoD "W9 대비 50% 감소"의 기준점도 재정의 필요 (§6.1).

### 1.7 검증 중점 SKU (W9 실측 기준)

SKU 6 누드 애쉬 로제(**가장 강함**) > SKU 2 런웨이 그레이 ≥ SKU 5 샤모 그래픽(그래픽 LTL에서 두드러짐). 짙은 SKU 1 돌 초코는 대조군(무회귀 확인용).

---

## 2. 배경/맥락

- P6-W5 Phase A/B에서 sclera veto 3-way(`uScleraVetoMode`) + A/B/C/D 토글 인프라 구축 완료. **Phase C(B1/B8 실기기 벤치 판정 + 흰자 빛남 수식 개선)는 미실행** — 흰자 빛남 부분만 본 W로 분리.
- P6-W9 통합 검증에서 흰자 빛남이 6 SKU 중 4 SKU에서 ⚠️로 관측, 렌즈 명도와 양의 상관 확정 → 시급성 데이터 확보.
- P7-W2에서 avg_iris_luma 실측 연결 + 밝은 환경 over-tint 교정 — 본 W의 직접 선행 의존 충족.
- **W3 보류분 연계**: blend dropdown 3종 축소(TintLinearV2/Multiply/ScreenLinear)는 "P7-W4 후"로 사용자 확정 (P7-W3 §status). ID=7 CRL 채택/제거 결정(메모리 `w5-b1-color-replace-decision`, W2 §6.7 보류)도 미결 — W4와의 경계는 §6.6.

---

## 3. 전제 조건

1. ✅ P7-W2 완료 (실측 avg_iris_luma default ON, fallback↔실측 A/B 토글 보존)
2. ✅ HIGH tier 기기 (S23+/Adreno 740) 보유
3. ✅ 토글 인프라 패턴 (`setDetailReinject`류 internal API + JNI/Java/demo 7단 체인 — W2 §4.1 재사용)
4. ✅ 6 SKU 에셋 + `lens_meta.json` 42 SKU 메타 인프라 (SKU별 오버라이드 §6.7 대비)

---

## 4. 목표

1. 밝은 렌즈 SKU의 흰자 빛남 강도 1차 완화 (luma attenuation)
2. 짙은 SKU 자연도 무회귀
3. ON/OFF 토글 인프라 보존 (Spike-A 진입 판단용 정량/정성 비교 가능)

### 4.1 Definition of Done (R1 확정 반영)

- [ ] **(실기기)** 재베이스라인: W2 실측 ON 상태, **두 조명 조건(일반 실내 + 밝은 조명)** × SKU 2/5/6+대조군 1 빛남 잔존 확인 (§5.7). "잔존 미미"(SKU 6 본인 체감 기준)면 **코드 0줄 조기 종결**
- [ ] **(코드)** 비율 cap 구현 (§5.8: `min(lum*scale, uScleraTintMax)`, 초기 cap≈1.275) + uniform 토글
- [ ] **(코드)** internal API + demo 토글 체인 (W2 패턴)
- [ ] **(실기기)** SKU 2/5/6 빛남 체감 ~50% 이상 감소 (재베이스라인 대비, 본인 실시간 토글 체감)
- [ ] **(실기기)** SKU 1(대조군) 포함 6 SKU 자연도 무회귀 + **sclera 경계 평탄화/하이라이트 칙칙함 육안 확인** (§5.8 Gemini 소수 의견 검증 항목)
- [ ] **(판정)** 만족 → 종결 / 부족 → soft-knee fallback(§5.8) 또는 P7-Spike-A 진입점 명시

---

## 5. 확정 사항 (P7-W0 상속 §5.1~5.6 + W4 R1 §5.7~5.13)

### 5.1 단계적 접근 — **1단계 = lightweight luma attenuation** (P7-W0 R1 3/3)
- Oklab+Laplacian은 학술 검증되었지만 GPU 예산 미검증 → 본 W에서 도입 금지, Spike-A 전용 (deep-research F1 정정).

### 5.2 Spike-A 진입 판단 — **W4 결과 후 사용자 결정** (P7-W0 §4.2 사용자 확정)
- luma attenuation으로 충분히 완화되면 미진행 + P8 즉시 진입. 부족하면 즉시 Spike-A.

### 5.3 검증 중점 SKU — **2/5/6 + 대조군 1** (P7-W0 §2.P7-W4 + W9 실측)

### 5.4 평가 방법 — **본인 실시간 토글 체감** (메모리 `solo-dev-bench-method`, `feedback_qualitative_device_judgment`)
- 정량 수치 보조 가능하나 판정은 육안 체감. 밝은 환경 우선 (메모리 `low-light-usage-rare`).

### 5.5 토글 인프라 보존 (P7-W0 §2.P7-W4 DoD)
- uniform 토글로 즉시 ON/OFF 비교 + 롤백 가능 상태 유지.

### 5.6 인덱스 1차 설계안 (출발점 — 세부 형태는 §6에서 확정)
```glsl
float sclera_factor = smoothstep(0.65, 0.85, lum);   // 흰자 영역 luma 추정
// 틴트 항에 attenuation (적용 위치 주의 — §1.5)
uScleraAttenuation 기본 0.85, 토글 가능
```
- W5 Phase A/B geometry veto와 **직교** — 둘 다 적용 가능 (인덱스 명시).
- → **R1에서 (c) 비율 cap으로 대체 확정 (§5.8)**. smoothstep 감쇄(a)는 soft-knee fallback으로만 보존.

---

> 이하 §5.7~5.13: **W4 R1 브레인스토밍 확정** (2026-06-10, 사용자 승인). 원문 `P7-W4_brainstorm/{codex,gemini,claude}_w4.md`, 종합 `synthesis.md`.

### 5.7 재베이스라인 선행 — **코드 작성 전 S23+ 두 조명 조건** (3/3)
- W2 실측 ON 기본 상태에서 SKU 2/5/6 + 대조군 1, **일반 실내 + 밝은 조명** 두 조건 확인.
- 근거(종합 단계 발견): scale 포화 경계 avgLum = 0.85/7.0 ≈ **0.121**. W2 실측상 정적 실내 ema≈0.067 → **짙은 홍채 + 일반 실내는 W2 교정 후에도 clamp 7.0 영역** = W9 빛남 조건과 동일 가능성 높음. "이미 완화" 기대는 밝은 조명(ema≈0.6, scale 1.27) 한정.
- **"잔존 미미" 판단 기준**: 가장 강한 SKU 6에서 본인 토글 체감 "거슬리지 않음" (수치 기준 도입 안 함 — `qualitative-device-judgment`). 미미 시 **코드 0줄 조기 종결 + Spike-A 미진행 + P8 진입**.
- 어두운 환경(저조도)은 W4 범위 외 유지 (`low-light-usage-rare`).

### 5.8 수식 형태 — **(c) 유효 틴트 배율 상한(비율 cap)** (2/3 다수결, 사용자 승인)
```glsl
// blendTintLinearV2 내부 (shader_sources.cpp:875-882 수정)
float tintMul = min(lum * scale, uScleraTintMax);   // 초기 cap = 0.85 * 1.5 ≈ 1.275
vec3 tinted = toLinearFast(blend) * tintMul;
```
- 다수(Codex+Claude, 구현형 독립 수렴): 빛남 원인(`lum` 무상한) 직접 차단 + brightness-invariant 의도 보존 + CRL detail clamp(`:889`) 동형 + 파라미터 1개.
- **소수(Gemini) 보존**: smoothstep 감쇄(a) — "고대비 경계 평탄화 우려". 기각 아닌 검증 항목화: 벤치에서 sclera 경계 평탄화/하이라이트 칙칙함 육안 확인, 거슬리면 **cap 위 soft-knee로 (a) 절충 진입** (fallback 경로).
- cap 초기값 1.275(=0.85×1.5), 실측 분포 로그(W2 디버그 인프라 재사용) 후 1.5/2.0/무제한 토글 비교 — 구현 중 조정.

### 5.9 luma 공간 — **linear Rec.709** (3/3) + 임계 쟁점 소멸
- `lum`·`uAvgIrisLum` 모두 기존 linear Rec.709 그대로 (W2 §5.7 측정=정규화 계약 정합). (c) 채택으로 별도 sclera_factor·smoothstep 임계(0.65/0.85)·luma 공간 선택 쟁점 자체가 소멸. 기존 veto의 sRGB 0.299 luma는 별도 경로로 불변.

### 5.10 B8 판정 분리 — **기존 veto 불변, W4 벤치는 veto 기본값(legacy 0) 고정** (3/3)
- `uScleraVetoMode` 0→1/2 단일화(P6-W5 B8)는 W4 결과 후 별도 처리 (W2 §5.6 변수 분리 전례).
- (c)는 "증폭 상한"이라 alpha veto와 작용 축 분리 — 이중 감쇄 우려 없음. 신규 geometry 보조 게이트 불요.

### 5.11 iris 내부 밝은 픽셀 — **허용** (3/3)
- 하이라이트 위 틴트 상한은 실제 물리(강한 반사가 렌즈색을 씻음)와 부합. 밝은 홍채는 비율≈1이라 cap(1.5)에 구조적으로 안 걸림. 벤치 육안 확인 항목으로만 관리 (§4.1).

### 5.12 적용 범위 — **ID=5(TintLinearV2) 단독** (3/3)
- Multiply는 어두워지는 방향, ScreenLinear는 밝아짐이 수식 의도, CRL은 detail 상한(≤maxDetail) 기보유 — 공통 적용 불요.
- **순서**: W4로 ID=5 개선 → 이후 ID=7 채택/제거(`w5-b1-color-replace-decision`) + blend dropdown 3종 축소(W3 보류분) 결정. 빛남 있는 ID=5와 CRL 비교는 CRL 과대평가라 W4 선행이 공정.

### 5.13 SKU별 오버라이드 — **글로벌 단일값 + ON/OFF 토글** (3/3)
- 1차 메타 도입 안 함 (W5 §5.10 "1차 튜닝 금지" 전례). W9 패턴이 SKU 예외가 아닌 "렌즈 명도 양의 상관"이라 글로벌 상한이 먼저 맞는 문제 (Codex). 특정 SKU만 과/부족 확인 시 `has_baked_limbal`/`prefers_crl` 패턴으로 후속 — 진입점만 명시.

---

## 6. 미결 사항 (R1로 전체 닫힘 → §5. 잔여는 구현 중 판정)

### 6.0 R1 결과 요약 (2026-06-10, 사용자 승인)

| 번호 | 원 쟁점 | 상태 | 반영 위치 |
|------|---------|------|-----------|
| 6.1 | 재베이스라인 gating | ✅ **닫힘** (3/3 선행 필수 + 미미 시 0줄 종결) | §5.7 |
| 6.2 | 수식 형태 | ✅ **닫힘** (2/3 → (c) 비율 cap, Gemini 소수 (a)는 soft-knee fallback 보존) | §5.8 |
| 6.3 | luma 공간/임계 | ✅ **닫힘** (3/3 linear Rec.709, 임계 쟁점은 (c) 채택으로 소멸) | §5.9 |
| 6.4 | geometry veto 결합 | ✅ **닫힘** (3/3 B8 분리, veto 기본값 고정) | §5.10 |
| 6.5 | 밝은 픽셀 오발 | ✅ **닫힘** (3/3 허용, 보조 게이트 불요) | §5.11 |
| 6.6 | 적용 범위/ID=7 | ✅ **닫힘** (3/3 ID=5 단독, W4 후 ID=7·dropdown) | §5.12 |
| 6.7 | SKU 오버라이드 | ✅ **닫힘** (3/3 글로벌 단일값) | §5.13 |

원문: `P7-W4_brainstorm/{codex,gemini,claude}_w4.md`. 종합: `P7-W4_brainstorm/synthesis.md`.

### 6.8 구현 중 판정 (저위험)

- **cap 초기값**: 1.275(=0.85×1.5)로 시작, 실측 분포 로그 후 1.5/2.0/무제한 토글 비교 (§5.8).
- **soft-knee fallback 진입 조건**: 벤치에서 sclera 경계 평탄화/하이라이트 칙칙함이 본인 체감으로 거슬릴 때만 (§5.8).
- **uniform/API 네이밍**: `uScleraTintMax` + internal API (W2 토글 체인 패턴 — 구현 시 확정).

---

## 7. 체크리스트 (브레인스토밍용)

### 7.1 읽을 파일

**필수**:
- 본 문서 (특히 §1.2~1.6 코드 실측, §6)
- `cpp/src/gpu/shader_sources.cpp:871-892` (블렌드 수식), `:1011-1063` (veto·블렌드 분기), `:1042` (maxDetail edge 감쇄)
- `docs/workPaper/P6-W9_integration_report.md` §3.1 (흰자 빛남 실측 패턴)
- `docs/workPaper/P7-W2_avg_iris_luma_measure.md` §5.6 (블렌드 정규화 설계 + over-tint 교정)

**선택**:
- `docs/workPaper/P6-W5_blend_sclera_bench.md` §5.4 (B8 veto 수식 후보), §5.10 (CRL clamp)
- `docs/workPaper/P7-W0_index.md` §2.P7-W4 (원 설계안)

### 7.2 송신 프롬프트 (Codex/Gemini 동일)

```
@docs/workPaper/P7-W4_sclera_luma_attenuation.md 읽고 (§7.1 필수 파일 포함),
섹션 6 미결 쟁점 7개(6.1~6.7)에 대해 각자 입장 정리 후
docs/workPaper/P7-W4_brainstorm/{codex|gemini}_w4.md로 저장.

규칙:
- 새 쟁점 제기 금지 (§6 범위 내)
- 각 쟁점 "추천 + 근거 1~2줄". 코드 주장 시 파일:라인 인용
- §1.5 적용 위치 제약(post-blend 불가)과 §1.6 베이스라인 문제를 전제로 답변
- Oklab/Laplacian 도입 제안 금지 (Spike-A 전용 — §5.1 확정)
- 한국어
```

### 7.3 예상 대립

- 6.2 수식 형태: Codex가 (b) lum cap(원인 직접 차단) vs Gemini가 (a) smoothstep 감쇄(GPU 비용·단순성) 갈릴 가능성
- 6.4 B8 동시 종결: Codex "변수 분리"(W2 §5.6 전례) vs "이중 감쇄 회피 위해 함께" 갈림 가능
- 6.6 ID=7 경계: 모델별 순서(W4 먼저 vs ID=7 결정 먼저) 입장 갈릴 수 있음

### 7.4 예상 합의

- 6.1 재베이스라인 필요성 자체 (real-data-first는 W2에서도 3/3 패턴)
- 6.7 글로벌 단일값 시작 (변수 최소화, W5 §5.10 "1차 튜닝 금지" 전례)
- 6.3 블렌드와 동일 linear Rec.709 (측정=정규화 정합 계약 W2 §5.7 전례)

---

## 8. 변경 이력

| 날짜 | 변경 |
|---|---|
| 2026-06-10 | 초안 — P7-W0 §2.P7-W4 + 코드 실측(§1.2~1.5) + W2 교정 후 베이스라인 문제(§1.6) 반영. 브레인스토밍 R1 준비. |
| 2026-06-10 | R1 완료 — Codex/Gemini/Claude 3모델 응답 + 종합(`P7-W4_brainstorm/synthesis.md`). 3/3 합의 5건(6.1/6.4/6.5/6.6/6.7) + 2/3 다수결 1건(6.2→(c) 비율 cap, Gemini 소수 (a)) + 파생 소멸 1건(6.3 임계). §5 이동은 사용자 승인 대기. |
| 2026-06-10 | **사용자 §5 이동 승인** — §5.7~5.13 확정 추가, §6 닫힘(6.0 요약표 + 6.8 구현 중 판정), §4.1 DoD 갱신(두 조명 재베이스라인 + (c) cap 구현 + 평탄화 검증 항목). 구현 대기 상태. |
