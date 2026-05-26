# P6-W6: B5 블링크 up ramp + B9 저조도 디테일 gate + C10 튜닝

> **상태**: 인사이트 작성 완료. 세부 계획 본문 작성 대기.
> **작성**: 2026-04-23
> **선행 의존**: P6-W1 (avg_iris_luma — B9 gate 입력), P6-W3 (환경 반사 계층 스캐폴드 — detail 합성 순서)
> **병렬 가능**: P6-W4, W5, W7와 병렬

---

## 1. 인사이트 (세션 간 맥락 보존) ⭐

### 1.1 두 벤치를 한 W로 묶은 이유

**B5** (블링크 up ramp) + **B9** (저조도 디테일 gate) + **C10 튜닝**:
- 모두 **시간적/조도 조건 관련** 파라미터. 벤치 시 같은 "저조도 환경 + 블링크 반복 동작"으로 관찰 가능.
- C10 디테일 재주입은 이미 수식 확정되어 있고, gate 임계값(B9)과 함께 튜닝.

### 1.2 B5: 블링크 up ramp 시간 — Gemini 반박의 핵심

**R3 Claude 자기비판 §1.4 I5**:
> "Gemini가 R2에서 '60ms Fade-out'을 명시하고 up 시간은 별도 언급 안 함. Claude가 Codex의 up 100~120ms를 수용한 건 Gemini 동의 없이."

**Gemini R3 §2**:
> "I5 (블링크 ramp): 부동의. `up 100~120ms`는 너무 김. `down 60ms / up 60ms` 대칭 또는 `up`을 더 짧게 가져가야 함."

→ **B5 벤치에서 판정 필요**.

**Codex R3 §8**:
> "down은 Claude 범위인 `50~80ms`, up은 내 R1 범위인 `100~120ms`로 수렴하자. 블링크 클립 10개만 보면 충분하다."

**Gemini R3 §7 추가 우려**: "120ms의 Fade-in은 UX 관점에서 명백한 퇴보". "AR 앱은 '반응성'이 우선".

### 1.3 B5 프로토타입 3안

**down 60ms 고정**. up만 변동:

| 프로토타입 | up 시간 | 주장자 |
|----------|---------|--------|
| up-fast | 60ms | Gemini |
| up-mid | 80ms | (중간 타협) |
| up-slow | 120ms | Codex |

실제 블링크: 100~150ms 지속. up ramp가 블링크 시간보다 짧아야 "튀는" 느낌 없음.

### 1.4 Codex R3 §4 — EMA 계수 검증 안 됨

**Codex R3 지적**:
> "C7 수정 필요: 시간 범위는 맞다. EMA 계수는 검증되지 않았고 목표 시간과 불일치한다."

즉 `α_close=0.15`, `α_open=0.08` 같은 계수가 30fps 기준 60~120ms 시간 목표와 안 맞음. W6 구현 시 **시간 목표 → EMA 계수 계산식** 필요:

※ 위 근사 `α ≈ 3 / (fps * target_sec)`는 참고용. **실제 계수 산정은 §5.1의 정밀 공식 `computeEmaAlpha(target_ms, fps) = 1 - pow(0.05, dt/target_ms)` 채택**. Codex/Gemini R4 리뷰 공통 지적 반영.

→ 정밀 공식 구현 + 실기기 타이밍 측정으로 목표 ms 달성 검증.

### 1.5 B5 판정 메트릭

```
정성 Y/N:
- `팝 느낌` (즉시 off 수준) — 너무 빠르면 Y
- `늦게 나타나는 지연` (Gemini 우려) — 너무 느리면 Y

판정:
- 60ms 우세 → Gemini 안 채택 (대칭 60/60)
- 80ms 우세 → 중간 타협
- 120ms 우세 → Codex 안 채택
```

### 1.6 B5 매트릭스

- 5 사용자 × 10 블링크/사용자 × 3 up 시간 = **150 이벤트**
- 실제로는 사용자 3명 × 5 블링크 × 3 = 45 이벤트로 축소 가능 (W6 브레인스토밍에서 조정)

### 1.7 B9: 저조도 디테일 gate 임계값

**C10 수식 (99 §1.2)**:
```glsl
detail = clamp(baseLum / blur3x3(baseLum), 0.85, 1.15);
// iris inner (r<0.65) 영역만 적용
// spec/reflection 영역 제외
// 저조도 gate: avg_iris_luma < THRESHOLD일 때 감쇄
```

**미결**: THRESHOLD 값.

**Gemini R3 §4 C10 지적**:
> "저조도 gate 수치(0.15)가 너무 낮을 수 있음. 실측 결과에 따라 유동적이어야 함."

**Codex R3 §1**:
> "`avg_iris_luma < 0.15` 임계값은 합의된 값이 아니다."

→ B9 벤치로 확정.

### 1.8 B9 프로토타입 3안

| 프로토타입 | 임계값 | 특성 |
|----------|-------|------|
| gate-tight | 0.10 | 저조도에서도 디테일 재주입 유지 (노이즈 증폭 위험) |
| gate-mid | 0.15 | 초안 값 (Claude R2) |
| gate-wide | 0.25 | 저조도 조기 감쇄 (안전) |

### 1.9 B9 매트릭스

- 3 조도(밝음/보통/저조도) × 3 gate 임계값 = **9 클립**
- 같은 SKU (오(OH)_베이글 등 중간 톤, 디테일 관찰 용이)

### 1.10 B9 판정 메트릭

```
- 저조도에서 홍채 노이즈 증폭 Y/N
- 보통 조도에서 디테일 살아있음 Y/N
- 밝은 조도에서 과도한 디테일 강조 Y/N

판정:
- gate-tight 노이즈 심함 + gate-wide 디테일 부족 → gate-mid 채택
- gate-tight 허용 가능 → 더 낮은 임계값 선호 (디테일 유지)
- gate-wide도 디테일 충분 → 더 높은 임계값 (안전성)
```

### 1.11 C10 구현 체크리스트 (W6에서 완료)

**99 §1.2 C10**:
> "`detail = clamp(baseLum / blur3x3(baseLum), 0.85, 1.15)`를 **iris inner(r<0.65)에만**, **spec/reflection 계층 계산 전에 합성**, **spec/reflection 영역 제외**. 저조도 gate 임계값은 **B9 벤치로 확정**"

**Codex R3 §1 추가 지적**:
> "§1 C10은 내 R2의 `spec/reflection 제외` 조건을 빠뜨렸다. detail reinjection은 반사 위에 다시 곱해지면 안 된다."

→ 반영: C10은 **반사 계층(C5) 합성 전**에 블렌드 결과에 적용.

### 1.12 C10 수식 구현 상세

```glsl
// iris inner 마스크 (r = iris 중심 기준 정규화 거리)
float innerMask = (dist < 0.65) ? 1.0 : 0.0;

// 3x3 주변 픽셀의 평균 휘도 (blur 근사)
float blurLum = (lum(texel + offsets[0]) + lum(texel + offsets[1]) + ...) / 9.0;

// detail 계산
float baseLum = lum(baseL);
float detail = clamp(baseLum / max(blurLum, 0.01), 0.85, 1.15);

// 저조도 gate (B9에서 확정 예정)
float gateStrength = smoothstep(GATE_THRESHOLD_LOW, GATE_THRESHOLD_HIGH, avg_iris_luma);
detail = mix(1.0, detail, gateStrength * innerMask);

// 블렌드 결과에 곱셈
blended *= vec3(detail);

// 이후 C5 환경 반사 계층 합성 (detail 이후 / 반사 이전에)
blended += reflection * fresnel * renderMask;
```

### 1.13 B5 + B9 동시 벤치 전략

**동시 관찰 가능**: "저조도 환경"에서 블링크 반복 촬영하면 B5 (up ramp 시간)와 B9 (저조도 디테일 gate) 둘 다 관찰. 단 판정 지표는 별도.

### 1.14 W6 브레인스토밍 시 Codex/Gemini에게 던질 질문

**B5 관련**:
1. EMA α → ms 변환 공식 정확화 (30fps 가정 맞나?)
2. 실기기 블링크 녹화 방법 (60fps 카메라 필요?)
3. 판정자 3명 중 블링크 특성 다른 사람 섞기 (빠른 눈/느린 눈)?
4. down과 up 비대칭이 자연스러운지 (심리적 인지 연구 참고?)

**B9 관련**:
5. `blur3x3` 시 텍스처 fetch 9개 추가 — 성능 예산 영향?
6. `spec/reflection 제외` 구체 구현: reflection 계층 합성 전에 detail → 반사 영역 (iris 외곽 near limbal 포함) 어떻게 특정?
7. 0.10 / 0.15 / 0.25 외 다른 임계값 필요?

**공통**:
8. B5와 B9 캡처를 같은 시나리오로 묶어도 판정 독립성 유지 가능?
9. avg_iris_luma 측정 주기 (매 프레임 vs N프레임 간격)?

### 1.15 구현 범위 (W6)

- 수정:
  - `shader_sources.cpp`:
    - C10 수식 구현 (detail reinjection with innerMask + gate)
    - C7 블링크 ramp 수식 구현 (up 시간 variant 교체 가능하게)
  - `gpu_lens_renderer.cpp`:
    - EMA 계수 → 시간 목표 계산 로직
    - renderMask와 C10 영역 분리 로직
- 벤치 자료:
  - `docs/bench/P6-W6/` 폴더

### 1.16 C7 EMA 수식 재작성

```cpp
// 시간 목표 → EMA α
float computeEmaAlpha(float target_ms, float fps) {
    // 95% 도달 시점 = -ln(0.05) / α / dt
    float dt = 1000.0f / fps;  // 프레임당 ms
    return 1.0f - std::pow(0.05f, dt / target_ms);
}

// down/up 별도 계수
float alpha_close = computeEmaAlpha(60.0f, fps);  // 60ms 목표
float alpha_open = computeEmaAlpha(up_time, fps);  // B5 결과에 따라 60/80/120
```

### 1.17 성능 예산

- C10 추가 fetch 9개 (3x3 blur) → iris 영역만 분기. MID tier +0.1~0.2ms 예상.
- C7 EMA: ALU 연산 2~4개. 무시.
- B5/B9 벤치용 실험 flag: 런타임 성능 영향 없음 (컴파일 타임 분기 권장).

### 1.18 B5/B9/C10 결과 종합

W6 완료 시점:
- B5 → C7 up ramp 시간 확정 (60/80/120 중)
- B9 → C10 저조도 gate 임계값 확정
- C10 구현 완료 (수식 + spec/reflection 제외 + gate)

이후 W9 통합 테스트에서 전체 동작 재확인.

---

## 2. 배경/맥락

### 2.1 세 가지 과제 묶음

- **B5**: 블링크 up ramp 시간 (60/80/120ms) — Gemini가 R3에서 명시 부동의한 마지막 쟁점
- **B9**: 디테일 재주입 저조도 gate 임계값 (0.10/0.15/0.25) — R3에서 Claude도 Gemini도 미확정 인정
- **C10**: 홍채 디테일 재주입 실제 구현 — 수식은 확정, 하지만 **spec/reflection 제외 조건** (Codex R3 지적) + gate 임계값이 미완

세 작업 모두 "저조도 조건 + 시간적 변동"이라는 공통 관찰 상황에서 확인 가능.

### 2.2 W1 TemporalStabilizer와의 경계 재확인

**Claude R2 확정** (99 §1.2 C7): "렌더러는 material-only temporal envelope만". 좌표 스무딩은 W1 TemporalStabilizer 소유.

W6에서 다룰 것:
- **블링크 ramp (material)**: render_alpha EMA, ok
- **디테일 재주입 (material)**: detail multiplier, ok

다루지 않을 것:
- iris_center/iris_radius 스무딩 — W1 소유
- visibility hysteresis — W1 소유 또는 render_confidence 영역

### 2.3 W6가 해결하는 것

- **C7 블링크 ramp 구현 완료**: down 50~80ms + up 60/80/120ms (B5 결과 반영)
- **C10 디테일 재주입 구현 완료**: 수식 + inner mask + spec/reflection 제외 + gate (B9 결과 반영)
- **C9 avg_iris_luma** 활용 — 저조도 gate가 이 값을 입력으로 사용 (W1에서 준비됨)

### 2.4 W6 해결하지 않는 것

- render_alpha 기본 스무딩 이외의 시간적 처리 (visibility fade, confidence hysteresis 등) — W1 영역
- 환경 반사 intensity 튜닝 — W4

---

## 3. 전제 조건

1. ✅ **W3 완료** — 반사 계층 구조 (디테일 재주입이 반사 전에 와야 하므로 순서 확립)
2. ✅ **W1 완료** — avg_iris_luma 측정 경로. B9 gate 입력.
3. ✅ **실기기 + 블링크 촬영 가능** (60fps 이상 권장)
4. ✅ **저조도 환경 확보** — 어두운 실내 (야간 or 커튼 닫은 방)

---

## 4. 목표

1. **B5 결과 확정** — up ramp 시간 (60/80/120ms 중)
2. **B9 결과 확정** — 저조도 gate 임계값
3. **C7 블링크 ramp 구현 완료**
4. **C10 디테일 재주입 구현 완료** (spec/reflection 제외 포함)
5. **EMA 계수 공식화** — 시간 목표 → α 계산 로직 (Codex R3 "계수 검증 안 됨" 해결)

### 4.1 Definition of Done

- [ ] `computeEmaAlpha(target_ms, fps)` 유틸 함수 구현
- [ ] B5 3 프로토타입 (60/80/120ms up) 토글 구현
- [ ] B9 3 프로토타입 (0.10/0.15/0.25 gate) 토글 구현
- [ ] C10 수식 (iris inner 마스크 + spec/reflection 제외) 셰이더 반영
- [ ] 벤치 촬영 + 평가 완료
- [ ] 결과 반영 커밋
- [ ] 99 §1.2 C7/C10 + §2 B5/B9 업데이트

### 4.2 Out of scope

- 블링크 감지 자체 로직 — W1 detector 영역 (visibility, eye_top/bottom)
- Contact shadow 강도 — 현 W에선 기존 유지

---

## 5. 99에서 확정된 사항

> 🔧 **구현 시 참조**: [`P6_implementation_handoff.md`](P6_implementation_handoff.md) §4.1 색공간/LUMA 규약 (`uAvgIrisLum` linear 값 직접 사용), §4.2 dist 정규화 규약, §4.3 영역 경계 매핑 (innerMask 0.5~0.7, W7 림발 0.85~1.0과 중간 0.7~0.85 공백 영역 존재), §4.4 GLSL 패스 규약 (블렌드→디테일→반사 순서).

### 5.1 C7 블링크 ramp 구조

```cpp
// gpu_lens_renderer.cpp
// R2 C-1: 실측 dt_ms를 매 프레임 직접 입력 (fps 인자 금지). §5.5와 시그니처 통일.
float computeEmaAlpha(float dt_ms, float target_ms) {
    return 1.0f - std::pow(0.05f, dt_ms / target_ms);
}

// 프레임 단위 업데이트 (dt_ms = 직전 프레임 delta time, runtime 실측)
if (eye_closing) {
    float alpha_close = computeEmaAlpha(dt_ms, 60.0f);  // down 60ms
    render_alpha_ = alpha_close * 0.0f + (1.0f - alpha_close) * render_alpha_;
} else {
    float alpha_open = computeEmaAlpha(dt_ms, target_up_ms);  // B5 결과
    render_alpha_ = alpha_open * target_alpha + (1.0f - alpha_open) * render_alpha_;
}
```

**`target_up_ms`**: B5 결과 (60/80/120).
**`dt_ms`**: 직전 프레임 delta time 실측값 (30fps 고정 가정 금지 — §5.5 / R2 C-1).

### 5.2 C10 디테일 재주입 수식

```glsl
// iris inner mask (동공 제외 iris 영역)
// dist는 shader에서 이미 /scaledRadius로 정규화됨 (0~1 범위). 추가 나누기 금지.
float innerMask = smoothstep(0.7, 0.5, dist);  // r<0.65 근방

// 3x3 blur (이웃 평균)
float blurLum = (lum_center * 2.0 + lum_n + lum_s + lum_e + lum_w + 
                 lum_ne + lum_nw + lum_se + lum_sw) / 10.0;

// detail 계수
float baseLum = dot(baseL, LUMA_709);
float detail = clamp(baseLum / max(blurLum, 0.001), 0.85, 1.15);

// 저조도 gate (B9 결과)
float gateStrength = smoothstep(GATE_LOW, GATE_HIGH, uAvgIrisLum);

// 최종 multiplier
float detailMul = mix(1.0, detail, gateStrength * innerMask);

// 순서: 블렌드 → 디테일 → 반사 (Codex R3 §1.11 순서)
blended *= vec3(detailMul);
// 이후 블렌드 += reflection * fresnel * renderMask (C5, W3)
```

**spec/reflection 제외**: 순서상 디테일이 반사보다 앞서 적용 → 반사 결과는 재주입 영향 없음 (Codex R3 우려 해결).

### 5.3 B5 매트릭스 (99 §2 B5)

- 사용자 5명 × 블링크 10회 × 3 up 시간 = **150 이벤트**
- 축소 가능: 사용자 3명 × 5 블링크 × 3 = 45 이벤트

### 5.4 B9 매트릭스

- 3 조도 × 3 gate = **9 클립**
- SKU: 오(OH)_베이글 (중간 톤, 디테일 관찰 용이)

### 5.5 EMA 공식 — **`1 - pow(0.05, dt_ms / target_ms)` 확정** (W6 R1 다수 2/3)

- **공식:** `computeEmaAlpha(dt_ms, target_ms) = 1.0 - pow(0.05, dt_ms / target_ms)`.
- **target_ms 정의:** **95% 도달 시간** (5% 잔존 시점).
- **실측 dt 사용:** 30fps 고정 가정 금지. 실제 frame delta time을 runtime에서 주입.
- **검증:** 구현 후 30fps/60fps 실기기 로그로 "80ms 만에 0.95 도달" 확인. Codex R3 계수 불일치 지적 해소.
- Claude R1 원안 `α = 1 - exp(-dt/τ)` (τ=63% 도달) 공식은 소수 의견으로 정정.
- **R2 C-1 정합성:** 시그니처를 `computeEmaAlpha(dt_ms, target_ms)` 단일형으로 확정. §5.1의 `(target_ms, fps)` 예시는 이 형태로 통일됨 (구버전 폐기).
- 출처: `P6-W6_brainstorm/synthesis.md` §2 + `codex_w6_r2.md` §4.

### 5.6 B5 user 수 — **3명 × 5 × 3 = 45 이벤트 확정** (W6 R1 합의 3/3)

- 정성 판정 규모. 3명 중 2명 이상 "팝"/"지연" 불만 Y면 ramp 계수 추가 조정 판정.
- 출처: `P6-W6_brainstorm/synthesis.md` §1.

### 5.7 B9 gate smoothstep — **±0.03 대칭 폭 확정** (W6 R1 합의 3/3)

- `smoothstep(threshold - 0.03, threshold + 0.03, avg_iris_luma)`.
- **0.06 폭 linear-luma 연속 전이** → 루마 축 계단(밴딩) 방지 + threshold 의미 유지.
- ⚠️ **R2 C-2 정정:** smoothstep은 **루마 값 매핑**이라 시간/프레임 평활화가 아님. "30fps 2~3프레임 transition"이라는 시간 해석은 부정확하므로 폐기. 입력 루마의 프레임 간 flicker 억제는 이 식이 보장하지 않음 (필요 시 별도 temporal filter).
- 출처: `P6-W6_brainstorm/synthesis.md` §1 + `codex_w6_r2.md` §4.

### 5.8 blur 커널 — **3×3 확정** (W6 R1 합의 3/3)

- 9 fetch × iris ROI 한정. Single-pass 원칙(99 §1.2 C-F) 준수.
- 5×5 / separable Gaussian 불채택.
- 출처: `P6-W6_brainstorm/synthesis.md` §1.

### 5.9 innerMask 경계 — **smoothstep `[0.7, 0.5]` 확정** (W6 R1 합의 3/3)

- `smoothstep(0.7, 0.5, dist)` 형태. inner=1, outer=0.
- ⚠️ **R2 C-3 정정:** `dist`는 셰이더에서 이미 `/scaledRadius`로 정규화됨(§5.2). `dist/iris_radius`는 **중복 정규화이므로 금지** — `dist` 그대로 사용.
- hard cutoff의 "경계 픽셀 번쩍임" 방지.
- 이 마스크는 중심에서 1이므로 **동공 제외를 단독 수행하지 않음** — 기존 iris/pupil 적용 마스크와 교차된 영역 안에서 사용 (C10 "동공 제외 iris 영역" 조건 유지).
- W7 림발 영역(0.85~1.0)과 분리 — 중간 0.7~0.85는 iris 본체 (아무 처리 없음).
- 출처: `P6-W6_brainstorm/synthesis.md` §1 + `codex_w6_r2.md` §4.

### 5.10 저조도 정의 — **B9 threshold로 자동 정의** (W6 R1 합의 3/3)

- 장면명(실내/실외)이 아닌 **`avg_iris_luma` 기준** 정의.
- B9 벤치 threshold 후보 {0.10, 0.15, 0.25} 중 채택값이 "저조도 경계".
- Claude 제안 시작값 0.15는 후보 중 하나로 포함.
- 출처: `P6-W6_brainstorm/synthesis.md` §1.

### 5.11 B5 × B9 동시 측정 — **동시 캡처, 지표 독립 확정** (W6 R1 합의 3/3)

- 저조도 환경에서 블링크 반복 = 단일 시나리오에서 B5+B9 동시 데이터 수집.
- 세션 구조: **고정 응시 5초 → 블링크 반복 10초 → 고정 응시 5초** (교차 영향 점검).
- 판정표 분리: B5 지표(블링크 직후 디테일 복귀) vs B9 지표(저조도 디테일 거북함).
- **R2 C-4 — C10 프로토타입 선행, 최종 튜닝만 후행:** B9 자체가 gate 3안(0.10/0.15/0.25) 비교라 **C10 gate 토글 프로토타입은 캡처 *전* 필수**. "C10 후행"은 *최종 튜닝*만을 의미. 순서: **C10 프로토타입 구현(gate 토글 포함) → B5/B9 동시 캡처 → 판정 → C10 최종 튜닝(gate threshold 확정 + 재주입 강도) → 최종 실기기**.
- 출처: `P6-W6_brainstorm/synthesis.md` §1 + `codex_w6_r2.md` §1/§5.

---

## 6. 미결 사항 (W6 브레인스토밍 R1 결과)

### 6.0 R1 결과 요약 (2026-04-24)

| 번호 | 원 쟁점 | 상태 | 반영 위치 |
|------|---------|------|-----------|
| 6.1 | EMA 공식 정확도 | ✅ **닫힘** (2/3 다수 + Claude 편향 정정) | §5.5 |
| 6.2 | B5 user 수 | ✅ **닫힘** (3/3) | §5.6 |
| 6.3 | gate smoothstep | ✅ **닫힘** (3/3) | §5.7 |
| 6.4 | blur 커널 | ✅ **닫힘** (3/3) | §5.8 |
| 6.5 | innerMask 경계 | ✅ **닫힘** (3/3) | §5.9 |
| 6.6 | 저조도 환경 정의 | ✅ **닫힘** (3/3) | §5.10 |
| 6.7 | B5×B9 동시 측정 | ✅ **닫힘** (3/3) | §5.11 |

**B5/B9/C10 동시 수행 타당성 (3/3 재확인).** B5×B9 동시 캡처 → 판정 → C10 후행 튜닝.

**미결 없음.** 가장 수렴도 높은 W. 후속: C10 튜닝 (B5/B9 결과 후), B9 threshold 확정 (0.10/0.15/0.25 중 실기기 선택).

원문: `docs/workPaper/P6-W6_brainstorm/{codex,gemini,claude}_w6.md`.
종합: `docs/workPaper/P6-W6_brainstorm/synthesis.md`.

### 6.0.1 R2 교차검증 결과 (2026-05-27)

R1 결정값(EMA 95%·gate ±0.03·3x3·innerMask smoothstep·luma 저조도 정의·동시측정) **전부 유지 — 재논의 불필요**. 단 Codex R2가 **문서 내부 정합성 4건**을 지적, 구현 전 정정 완료:

| 코드 | 정정 | 반영 |
|------|------|------|
| C-1 | EMA 시그니처 `(target_ms, fps)` → `(dt_ms, target_ms)` 단일화 (실측 dt) | §5.1 / §5.5 |
| C-2 | gate "30fps 2~3프레임 transition" 시간 주장 폐기 → 0.06 루마폭 연속전이 | §5.7 |
| C-3 | innerMask `dist/iris_radius` 중복 정규화 제거 → `dist` 그대로 | §5.9 |
| C-4 | "C10 후행" → 프로토타입 선행 / 최종 튜닝만 후행 명료화 | §5.11 |

- Gemini R2: 전면 동의(R1 재확인 가치 O). 단 위 4건 미발견 + ±0.03 "인간 눈 대비" 근거는 검증 안 됨 → 미채택.
- **R3 불필요.** 검토: `docs/workPaper/P6-W5_brainstorm/codex_phase_ab_review.md` 패턴의 critical-review 적용.
- 원문: `docs/workPaper/P6-W6_brainstorm/{codex,gemini}_w6_r2.md`.

---

## (원 미결 사항 세부 — 참고용)

### 6.1 EMA α → ms 공식 정확도

Codex R3 "α=0.15 계수가 60~120ms 시간 목표와 수학적으로 안 맞음" 지적.
- `computeEmaAlpha` 공식 검증 필요 (`pow(0.05, dt/target)`).
- 30fps 가정 맞나? 실기기 측정?

**Claude 제안**: 공식 구현 후 실제 로그 기반 타이밍 측정.

### 6.2 B5 user 수 축소

99는 5명 × 10 블링크. 축소(3×5 = 15 이벤트) 허용?
- Gemini가 "정성 판정이면 충분" 주장 가능
- Codex가 "통계적 유의성 필요 5명"

**Claude 추천**: 3명 × 5 블링크 × 3 시간 = 45 이벤트. 정성 판정엔 충분.

### 6.3 B9 gate smoothstep 경계값

`smoothstep(GATE_LOW, GATE_HIGH, avg_iris_luma)` — 두 값 어떻게?
- 예: GATE_LOW=0.10, GATE_HIGH=0.20 → 어두우면 완전 off
- B9는 "임계값 하나"인데 smoothstep은 두 값 필요 → 실제로 `smoothstep(threshold-0.05, threshold+0.05, ...)` 식?

**Claude 제안**: `GATE_LOW = threshold - 0.03`, `GATE_HIGH = threshold + 0.03`. B9 결과가 threshold 값 결정.

### 6.4 디테일 재주입 blur 커널 크기

3x3 vs 5x5 vs 2-tap 방향성 분리?
- 3x3: 9 fetch, iris 영역만 → 성능 미미
- 5x5: 25 fetch, 과도
- 분리 가능 가우시안 (h+v 2패스): single pass 금지라 부적합

**Claude 추천**: 3x3 유지. Codex R1 원안.

### 6.5 innerMask 경계

`smoothstep(0.7, 0.5, dist/iris_radius)` vs `(dist < 0.65) ? 1.0 : 0.0` (hard)?
- 부드러운 전환 vs 명확한 경계
- 림발 영역과의 겹침 주의

**Claude 제안**: 부드러운 smoothstep. 림발 영역(0.7~1.0)과 자연 분리.

### 6.6 저조도 환경 정의

"저조도 = avg_iris_luma 몇 이하?"
- 0.10: 매우 어두움 (야간 실내)
- 0.15: 보통 저조도 (창문 없는 실내)
- 0.20: 실내 형광 기준값

**B9는 임계값 자체가 이 질문에 답**.

### 6.7 B5/B9 동시 측정 가능성

- B5는 "블링크 반복" 필요
- B9는 "저조도 환경" 필요
- **동시 측정**: 저조도 환경에서 블링크 반복 → 한 번에 두 벤치 데이터

**Claude 추천**: 동시 측정. 단 판정 지표는 독립.

---

## 7. W6 브레인스토밍 시작 체크리스트

### 7.1 읽을 파일

**필수**: P6-W0, P6-W6, 99 §2 B5/B9 + §1.2 C7/C10, 13_codex_r3.md §1 (EMA 계수 불일치)
**선택**: 14_gemini_r3.md §2 (up 120ms 부동의), §4 C10 (gate 0.15 우려)

### 7.2 송신 프롬프트

```
@docs/workPaper/P6-W6_temporal_detail_bench.md 읽고, 섹션 6 미결 7개에 대해
각자 입장 정리. docs/workPaper/P6-W6_brainstorm/{codex|gemini}_w6.md.

특히:
- 6.1 EMA 공식 정확도 (Codex 계수 검증 지적)
- 6.2 B5 user 수 (통계 유의 vs 정성)
- 6.3 gate smoothstep 경계 방식
- 6.5 innerMask 경계 smoothstep vs hard

규칙:
- 새 쟁점 금지
- B5/B9/C10 동시 수행 타당성 평가
```

### 7.3 W6 소요

- 프로토타입 구현 (B5 3안 + B9 3안 토글): 1.5h
- 벤치 (블링크 녹화 + 저조도 3조도 × 3 gate): 2h
- 평가: 1h
- C10 실제 구현 (spec/reflection 제외 순서): 1.5h
- 반영 + 문서: 1h

**총 7h**.

---

## 8. 완료 정의 + 다음 W 트리거

### 8.1 완료 정의

§4.1 체크리스트.

### 8.2 커밋 전략

- `docs(P6-W6): 섹션 2~8 본문 작성`
- `feat(gpu-lens): P6-W6 computeEmaAlpha 유틸 + 블링크 ramp 토글`
- `feat(gpu-lens): P6-W6 C10 디테일 재주입 구현 (spec/reflection 제외 순서)`
- `chore(bench): P6-W6 B5/B9 결과 report`
- `feat(gpu-lens): P6-W6 B5/B9 결과 반영`

### 8.3 다음 W 트리거

W6는 독립. 완료 후 W7/W8/W9로.

### 8.4 W6 실패 시

- B5 결과 모호 → 80ms 중간값 채택 (타협)
- B9 결과 모호 → 0.15 중간값 유지
- EMA 공식 오류 → 실기기 측정 기반 경험적 조정

### 8.5 기대 효과

- 블링크 "팝 느낌" 제거 → 사용자 체감 개선
- 저조도 노이즈 증폭 방지 → 야간 사용 품질 확보
- EMA 계수 공식화 → 다른 temporal 값에도 재사용 가능

---

## 참조

- 99_final_decision.md §2 B5, B9, §1.2 C7, C10
- 13_codex_r3.md §1 (EMA 계수 불일치, C10 spec 제외 빠짐)
- 14_gemini_r3.md §2 (up 100~120ms 부동의), §4 (gate 0.15 우려)
- 08_codex_r2.md I5 (Claude 시간 범위 수용)
- 07_claude_r2.md I5 (Claude 원 타이밍)
