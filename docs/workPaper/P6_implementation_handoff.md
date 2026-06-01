# P6 Implementation Handoff

**작성일:** 2026-04-24
**대상:** `ar-lens-implement` 스킬 + Phase 6 구현 담당자
**목적:** 9개 W 브레인스토밍 R1 완료 후, 실제 구현 진입을 위한 준비도/주의사항/전역 규약/호출 순서 통합 레퍼런스.

> **이 문서를 `implement P6-W{N}` 호출 시마다 참조문서로 포함 권장.**
> 각 W 문서의 §5가 "무엇을 구현할지", 본 문서가 "어떻게/어떤 순서로/무엇을 주의하며" 담당.

---

## 1. 브레인스토밍 R1 완료 상태

- **9개 W × 63개 쟁점** 전부 R1 응답 + synthesis 완료.
- **합의 71% (45개)** / **다수결 25% (16개)** / **실기기 이관 2% (1개, W1 §6.2 ROI 반경)**.
- Claude 편향 정정 5회 (W3/W4/W5/W6 · 메모리 `feedback_multi_ai_orchestration_bias` 작동).
- 참여 모델: Codex(gpt-5.4 xhigh), Gemini(gemini-3-flash), Claude(opus-4-7).

### W별 산출물 위치

```
docs/workPaper/
├── P6-W{N}_*.md                          # 각 W 본문 — §5에 R1 확정 사항
└── P6-W{N}_brainstorm/
    ├── codex_w{N}.md                     # 독립 응답
    ├── gemini_w{N}.md                    # 독립 응답
    ├── claude_w{N}.md                    # 독립 응답 (히스토리, 편향 정정 증거 포함)
    └── synthesis.md                      # 세 응답 종합 + 최종 결정 근거
```

구현자는 **각 W 본문 §5**를 1차 참조. 의사결정 히스토리가 필요하면 `synthesis.md`.

---

## 2. 구현 착수 준비도

### 하드 블로커: 0건

사전 점검에서 발견된 5건 버그 모두 해소됨 (§3 참조). 컴파일 에러나 수식 버그 유발 요소 없음.

### 소프트 갭 3건 — 구현자 현장 판단 허용 (스킬 Preflight "구현 중 판정" 수준)

| # | 위치 | 내용 | 처리 방침 |
|---|------|------|-----------|
| A | W5 §5.8 | SKU 수 "5 → 6 가능, 구현 단계에서 조정" | W5 착수 시 B1 매트릭스와 함께 확정. 5 SKU로 매트릭스가 이미 유의미하면 5로, 디자인 차이 관찰 필요하면 6으로. |
| B | W3 §5.1 | `apply_contact_shadow(blended)` placeholder | 실제 구현은 `gpu_lens_renderer.cpp` `contact_shadow_` + `uContactShadow` uniform 기반. shader 내부 `if (uContactShadow == 1) { ... }` 블록 찾아 해당 영역 바로 다음에 반사 합성 삽입. |
| C | W4 | env_map 에셋 제작·평가자 섭외·촬영 세팅 | `ar-lens-implement` 범위 밖. 구현 후 사용자가 수행하는 운영 작업. |

---

## 3. 구현 전 수정된 5건 버그 (재발 방지용 기록)

### 버그 1: W1 §5.2.1 색공간 규약 — Rec.601 sRGB → Rec.709 linear 통일
- **증상:** W1 §5.2.1(초안)은 Rec.601 sRGB, W2 §5.10(R1 확정)은 Rec.709 linear → 문서 간 상반.
- **원인:** W1 §5.2.1이 브레인스토밍 전 초안이었고 W2 R1 합의("전 경로 linear") 반영 누락.
- **수정:** W1 §5.2.1을 Rec.709 linear 통일 + CPU 측정 절차 linear로 기술.

### 버그 2 & 3: W2 §5.1/§5.3 — `uAvgIrisLum * uAvgIrisLum` 이중 변환
- **증상:** `avgLumLinear = uAvgIrisLum * uAvgIrisLum` 코드가 "uAvgIrisLum이 sRGB" 가정. W1 §5.2.1 linear 통일 후 squaring은 이중 변환 버그.
- **수정:** squaring 제거, 주석 "이미 linear 공간 값, squaring 금지" 추가.

### 버그 4: W3 §5.1 C5 합성 — 3개 시그니처/단위 버그
- `calcFresnel(normal, viewDir)` → `calcFresnel(dist)` (§5.8 가짜 Fresnel, 노멀 제외 확정)
- `sampleReflection(reflectUV, normal, viewDir)` → `sampleReflection(reflectUV)` (§5.11 방식 A 확정)
- `smoothstep(iris_radius * 1.2, 0.0, dist)` → `smoothstep(1.2, 0.0, dist)` (§5.5 "dist 이미 정규화, iris_radius 곱하기 금지" 위반)

### 버그 5: W3 §5.8 Fresnel 수식 부재 + 방향 반대
- **증상:** "중심=1, 가장자리 감쇄" 기술. 물리 Fresnel은 grazing angle(외곽)에서 강해짐.
- **수정:** "중심=0, 외곽=1" 방향으로 정정 + 실제 GLSL `smoothstep(0.7, 1.0, dist)` 삽입 + 튜닝 범위 `[0.6, 0.7, 0.8]` 명시.
- **편향 정정 히스토리:** `claude_w3.md` 원문에 "중심=1" 표현 그대로 보존(편향 증거). 실제 구현 참조점만 정정.

---

## 4. 전역 규약 (여러 W가 공유)

### 색공간 / LUMA 규약

- **전 렌더 경로: Linear space.** sRGB 평균 금지.
- **LUMA_COEFFS: Rec.709 linear — `vec3(0.2126, 0.7152, 0.0722)`**. 공통 매크로/상수 `LUMA_709`.
- **CPU 측정 (W1 avg_iris_luma):** 픽셀 sRGB→linear 변환(`pixel * pixel` 감마 2.0 근사) → Rec.709 계수로 luma → ROI 평균 → uniform 주입.
- **Shader에서 uAvgIrisLum 사용:** linear 값 그대로. **squaring 금지.**
- **검증:** W9 CI 체크리스트에 "shader vs CPU luma 오차 ≤1% 테스트" 포함.

### `dist` 정규화 규약

- shader의 `dist`는 **이미 `/scaledRadius`로 정규화**된 값 (0=중심, 1=외곽). `float dist = distance(adjustedCoord, adjustedCenter) / scaledRadius;`
- **`iris_radius` 또는 `scaledRadius` 추가 곱하기 금지** — 이중 정규화 오류.
- 이 규약 위반이 5건 수정 중 1건 (W3 §5.1). 구현 시 유의.

### 영역 경계 매핑 (여러 W가 같은 정규화 공간 공유)

| 영역 | dist 범위 | 담당 W |
|------|-----------|--------|
| Pupil (material 적용) | 0.0 ~ 0.4 | W8 §5.6 centerProximity |
| Iris 내부 (디테일 재주입 innerMask) | 0.0 ~ 0.5~0.7 | W6 §5.9 |
| avg_iris_luma 측정 ROI | 0.0 ~ 0.60 (실기기 튜닝) | W1 §5.7 |
| Iris 외곽 (Fresnel) | 0.7 ~ 1.0 | W3 §5.8 |
| 림발 (자동감지 ROI) | 0.85 ~ 1.0 | W7 §5.6 |
| Periphery (얼굴 반사) | 1.8 ~ 2.5 | W4 §5.8 |

### GLSL 패스 규약

- **Single-pass 원칙** (99 §1.2 C-F). Separable Gaussian 같은 multi-pass 금지.
- **Compute shader 금지** (GLES 3.1 드라이버 품질 리스크).
- **블렌드/디테일/반사 순서** (Codex R3 §1.11, W6 §5.11 재확인):
  1. blendMode ID 분기 (W2)
  2. C10 디테일 재주입 (W6): `blended *= vec3(detailMul)`
  3. W8 renderMask 확장 (조건부, `#ifdef ENABLE_PUPIL_MATERIAL_RESTORE`)
  4. C5 반사 합성 (W3): `blended += reflection * fresnel * uReflectionIntensity * renderMask`
  5. W8 Pupil material 적용 (조건부, §5.9 옵션 A: material → 반사 순서)
  6. contact shadow 기존 분기

### SKU 메타 규약 (W5 + W7 공용)

- **파일:** `android/demo-app/src/main/assets/lens_meta.json`
- **스키마:**
  ```json
  [
    {
      "sku_id": "...",
      "display_name": "...",
      "has_baked_limbal": false,
      "prefers_crl": false,
      "prefers_graphic_outline": false
    }
  ]
  ```
- **Default fallback:** 누락 SKU는 모든 플래그 `false`. WARN 로그 출력.
- **로그 형식:** `[IrisSDK] SKU meta missing for "<sku_id>", using default`

---

## 5. W 간 의존성 + 구현 순서

```
W1 (EyeRenderPacket + avg_iris_luma)
 │
 ├──► W2 (블렌드 3종) ──► W5 (B1/B8 벤치)
 │                      └─► W7 (림발 정책) ─── 메타 JSON 공용
 │
 └──► W3 (환경 반사 scaffold) ──► W4 (B2 벤치)
                                  │
                                  ├─ 2/3 Y ──► W8 (Pupil material)
                                  └─ 0/3 Y ──► W8 폐기

         W6 (temporal/저조도) — W1·W3 완료 후 독립 병렬

                       ↓
                       W9 (통합 + 머지) — 모든 W 완료 후
```

### 권장 호출 순서

```
implement P6-W1          # 독립, 즉시 시작 가능
implement P6-W2          # W1 완료 후 (uAvgIrisLum uniform 의존)
implement P6-W3          # W2 완료 후
implement P6-W4          # W3 완료 후, B2 촬영 포함
# ↑ W4 B2 결과 "2/3 Y/N" 확인 후 W8 활성 여부 결정
implement P6-W5          # W2/W3 완료 후 (병렬 가능)
implement P6-W6          # W1/W3 완료 후 (병렬 가능)
implement P6-W7          # W3 완료 후 (병렬 가능)
implement P6-W8          # W4 B2 = 2/3 Y일 때만
implement P6-W9          # 모든 W 완료 후
```

### 병렬 실행 가능 포인트
- W5/W6/W7: W3 완료 후 3개 동시 진행 가능.
- W4 벤치 촬영 중 W5/W6/W7 코드 구현 진행 가능.

---

## 6. W별 핵심 구현 포인트 (구현 시 놓치지 말 것)

### W1 — EyeRenderPacket + avg_iris_luma
- **어댑터 위치:** `cpp/src/gpu/eye_render_packet_adapter.{h,cpp}`
- **CPU 측정:** linear 공간, Rec.709, `pow(0.3, dt/target)` 형 EMA (§4 전역 규약 참조).
- **EMA:** `L_t = 0.3 · measured_t + 0.7 · L_{t-1}`, `L_0 = 0.35` (fallback).
- **ROI 초기값:** `r < 0.60 * iris_radius * eyelidMask`, final clamp `[0.1, 0.9]`. 실기기 A/B `{0.55, 0.60, 0.65}` 스위프 후 최종 확정.
- **optional 필드:** `head_pose_yaw_roll`, `reflection_dir`, `render_confidence` 3개 `std::optional<T>`로 예약. W1 렌더러 미활용.
- **render_confidence:** `TemporalStabilizer::visibility` 값 재사용. 신규 수식 금지.

### W2 — 블렌드 3종
- **함수:** `blendTintLinearV2`, `blendScreenLinear`, `blendColorReplaceLinear` 3종.
- **ID 매핑:** TintLinearV2 = 5 (canonical default), ID 7 CRL = W2에서 분기 활성 (W5 B1 결과 후 롤백 가능).
- **ID 3/4/6:** 빈 슬롯 유지, TintLinearV2 fallback + debug 빌드에서 WARN 로그.
- **realSpec:** 완전 삭제. Git history로 복원. (선택) `docs/workPaper/P6-W2_brainstorm/realSpec_archive.md`.
- **sdk_api.h:** 기본값 TintLinearV2 ID=5 주석 추가.

### W3 — 환경 반사 scaffold
- **sampleReflection:** `vec3 sampleReflection(vec2 uv)` — uniform `uSourceType` 스위치 (0=OFF, 1=env-map, 2=periphery). W3 scaffold는 `uSourceType=0` 기본.
- **Fresnel:** 옵션 C 가짜 `smoothstep(0.7, 1.0, dist)`. 중심=0, 외곽=1. 노멀 없음.
- **reflectUV:** `(uv - iris_center_uv) / iris_radius_uv * 0.5 + 0.5`. iris local.
- **renderMask hook:** `#ifdef ENABLE_PUPIL_MATERIAL_RESTORE`, W3 기본 off.
- **env_map 위치:** `android/demo-app/src/main/assets/env/env_default_256x128.png`.

### W4 — B2 환경 반사 벤치
- **매트릭스:** 4환경 × 2동작 × 3프로토타입 = 24 클립. 동일 take에서 uniform 토글로 3프로토타입 연속 캡처.
- **평가자 3명:** 아시아 짙은 홍채(필수) + 밝은/중간 + 개발자.
- **촬영:** 스크린 레코딩, 30fps, 10초 take → 5초 trim.
- **블라인드:** 파일명 무작위 ID, 정답표 암호화.
- **Pupil 체감 Y/N:** 각 클립 별도 기록 → W8 발동 판정.

### W5 — B1 + B8 조합 벤치
- **SKU:** 로뮤_디어 멜로우 + 클라셋_돌 초코 별도 확보 (5 or 6 SKU — 구현 시 조정).
- **라벨링:** A=Normal+color, B=Normal+luma, C=CRL+color, D=CRL+luma. 같은 take 연속 토글.
- **런타임 UI:** 4버튼. 정답표 암호화.
- **CRL clamp:** `[0.75, 1.25]` 유지. W5 1차 튜닝 금지.
- **조건부 채택:** `prefers_crl: bool` 메타. 기본 false. UI/API 오버라이드 없음.
- **color-veto 강도:** W5 1차 0.6 고정. B8 color-veto 채택 시 phase-2에서 `{0.4, 0.6, 0.8}` 스위프.
- **calcScleraFactor:** 함수명 유지, 내부 수식 교체.

### W6 — 블링크 ramp + 저조도 + C10
- **EMA 공식:** `computeEmaAlpha(dt_ms, target_ms) = 1 - pow(0.05, dt_ms / target_ms)`. target = **95% 도달 시간**. 실측 dt 사용.
- **B5 user 수:** 3명 × 5 블링크 × 3 시간 = 45 이벤트.
- **B9 gate:** `smoothstep(threshold - 0.03, threshold + 0.03, avg_iris_luma)`. threshold 후보 `{0.10, 0.15, 0.25}`.
- **blur:** 3×3, single-pass.
- **innerMask:** `smoothstep(0.7, 0.5, dist)`. 중심=1, 외곽=0.
- **B5×B9 동시 측정:** 저조도 블링크 세션. 세션 구조 "고정 5초 → 블링크 10초 → 고정 5초".
- **C10 튜닝:** B5/B9 결과 후 후행.

### W7 — 림발 정책
- **자동감지:** ROI `[0.85, 1.0]`, 임계 0.75, ratio = edge_lum / center_lum.
- **10/10 엄수** + 실기기 실패 SKU는 `has_baked_limbal: true` 메타로 개별 명시.
- **메타 JSON:** `lens_meta.json` (§4 SKU 메타 규약).
- **로그 레벨:** WARN.
- **엔비_샤모:** `prefers_graphic_outline: true` → `uApplyLimbal = 0`.

### W8 — Pupil material (조건부)
- **착수 조건:** W4 B2 = 2/3 이상 Y 확정 후만.
- **centerProximity:** `1.0 - smoothstep(0.0, 0.4, dist)`. 중심=1, 외곽=0.
- **materialAlpha:** 0.12 기본. `[0.10, 0.15]` 실기기 스위프 선택적.
- **평균 색 ROI:** `[0.5, 0.85]` (CPU 계산, uniform 주입).
- **순서:** material → C5 반사 (옵션 A, material이 반사 하이라이트 덮지 않음).
- **활성 방식:** `#ifdef ENABLE_PUPIL_MATERIAL_RESTORE`. W8 빌드 default 1, 실패 시 0 롤백.
- **실패 시:** `#ifdef` 0 + Phase 7+ 이월. 코드 유지(재시도 경로).

### W9 — 통합 + 머지
- **머지 방식:** 머지 커밋 (옵션 A). W별 히스토리 보존.
- **Android demo UI / deprecated 제거:** Phase 7 초반 별도 cleanup PR.
- **릴리즈 노트:** Phase 6 단일 1장. W별 결정 + 벤치 결과 + W8 조건부 여부 + 이월 항목.
- **CI 체크:** §7 참조.

---

## 7. CI / 머지 전 체크리스트 (W9 §5.10)

### 기본 빌드·테스트
- [ ] C++ 빌드 (`cpp/cmake-build-debug` 재사용, `scripts/build_android.sh`, `scripts/build_ios.sh`)
- [ ] 단위 테스트 (`cd cpp/cmake-build-debug && ctest`)
- [ ] Android demo APK 빌드
- [ ] iOS Framework 빌드 (해당 시)
- [ ] 기존 정적 분석/린트 통과
- [ ] shader compile 에러/warning 0

### Phase 6 특이 체크 (W9 R1 3모델 분업으로 보완)
- [ ] **LUMA 계수 shader vs CPU 오차 ≤1% 테스트** (Claude) — W1 §5.2.1 검증
- [ ] **메모리 누수 테스트** (Gemini) — `EyeRenderPacket`/`LensSkuMetadata` 교체 시나리오, valgrind or sanitizer
- [ ] **3 tier 기기 프레임 타임 프로파일링** (Gemini) — jitter 확인
- [ ] **W1 §6.2 ROI 반경 closure 상태 명기** (Codex) — 실기기 튜닝 최종값
- [ ] **99 P6-W1 → P6-W8 rename cross-reference 추가** (Codex) — 99 문서의 Pupil 트랙 이름 정리
- [ ] **W4 B2 실측 결과 확정** — W8 포함 / skip / 사용자 판단

### 문서 반영
- [ ] 각 W 문서 §5 최종 반영 확인
- [ ] `99_final_decision.md` §1.2 C5/D3 + §2 B1~B9 상태 "확정/구현됨" 반영
- [ ] `CHANGELOG` 작성 (Phase 6 단일 1장)

### 성능 기준 (CLAUDE.md)
- [ ] 30fps+
- [ ] 검출 지연 33ms 이하
- [ ] 메모리 100MB 이하
- [ ] SDK 20MB 이하

---

## 8. ar-lens-implement 호출 가이드

### 스킬 Preflight 통과 여부 (본 문서가 그 근거)

스킬이 요구하는 3조건 모두 충족:
1. ✅ §5 확정 사항에 실제 구현 가능한 결정들 — **본 문서 §6이 W별 핵심 체크**
2. ✅ §6 미결 사항이 비어있거나 "구현 중 판정" 수준 — 모든 W §6이 "R1 결과 요약 표"로 교체됨
3. ✅ 선행 W 완료 — **본 문서 §5 의존성 그래프 참조**

### 호출 시 참조 명령

```
implement P6-W{N}  # 기본 모드
implement P6-W{N} quick   # 최소 구현 + 빌드만
implement P6-W{N} full    # 전체 + 단위 테스트 + 3 tier 실기기
implement P6-W{N} debug   # 문제 있는 W 재작업
```

### 구현 중 본 문서 재참조 시점

- **색공간/LUMA 코드 작성 시** → §4 전역 규약 재확인 (특히 `uAvgIrisLum * uAvgIrisLum` 금지).
- **dist 사용 시** → §4 "dist 정규화 규약" 확인 (iris_radius 곱하기 금지).
- **영역 경계 수식 작성 시** → §4 "영역 경계 매핑" 표 확인 (다른 W와 겹침 방지).
- **SKU 메타 접근 시** → §4 "SKU 메타 규약" 스키마 확인.
- **블렌드/디테일/반사 순서 의심 시** → §4 "GLSL 패스 규약" 확인.

---

## 9. 변경 이력

| 날짜 | 변경 |
|------|------|
| 2026-04-24 | 초안 작성. 9개 W 브레인스토밍 R1 완료 + 5건 버그 수정 후 정리. |

---

## 10. 관련 문서

- `docs/workPaper/P6-W0_index.md` — Phase 6 W 인덱스
- `docs/workPaper/P6-W{1..9}_*.md` — 각 W 본문
- `docs/workPaper/P6-W{1..9}_brainstorm/synthesis.md` — 각 W 브레인스토밍 종합
- `docs/workPaper/P5-W3-05_brainstorm/99_final_decision.md` — Phase 5 최종 합의 (Phase 6 전제)
- `.claude/skills/ar-lens-implement/SKILL.md` — 구현 스킬 정의
- `.claude/skills/ar-lens-brainstorm/SKILL.md` — 브레인스토밍 스킬 정의
- `CLAUDE.md` — 프로젝트 전역 규칙
