# P6-W6: B5 블링크 up ramp + B9 저조도 디테일 gate + C10 튜닝

> **상태**: 인사이트 작성 완료. 세부 계획 본문 작성 대기.
> **작성**: 2026-04-23
> **선행 의존**: P6-W3 (환경 반사 계층 스캐폴드)
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

```
EMA: new = α * target + (1-α) * current
거의 0에서 95% 도달 시간: T ≈ -ln(0.05) / (fps * α) ≈ 3 / (fps * α)

30fps 기준:
  60ms (2 프레임) → α = 3 / (30 * 0.060) ≈ 0.22... 실제론 더 큰 α 필요
  실측해보면 α=0.5가 약 100ms 소요
```

→ **수학 공식 검증 + 실기기 타이밍 측정** 필요. W6 브레인스토밍에서 정확화.

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
_TODO_

## 3. 전제 조건
_TODO_

## 4. 목표
_TODO_

## 5. 99에서 확정된 사항
_TODO_

## 6. 미결 사항
_TODO_

## 7. W 브레인스토밍 시작 체크리스트
_TODO_

## 8. 완료 정의 + 다음 W 트리거
_TODO_

---

## 참조

- 99_final_decision.md §2 B5, B9, §1.2 C7, C10
- 13_codex_r3.md §1 (EMA 계수 불일치, C10 spec 제외 빠짐)
- 14_gemini_r3.md §2 (up 100~120ms 부동의), §4 (gate 0.15 우려)
- 08_codex_r2.md I5 (Claude 시간 범위 수용)
- 07_claude_r2.md I5 (Claude 원 타이밍)
