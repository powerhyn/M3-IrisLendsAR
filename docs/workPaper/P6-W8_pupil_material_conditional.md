# P6-W8: Pupil Material Restore (조건부 트랙)

> **상태**: 인사이트 작성 완료. 세부 계획 본문 작성 대기.
> **조건부**: P6-W4 B2 벤치에서 "중앙 공동 체감" 지표 2/3 이상일 때만 착수. 미달 시 트랙 폐기.
> **작성**: 2026-04-23
> **선행 의존**: P6-W4 (B2 결과)
> **후속 의존**: P6-W9 (통합 테스트)

---

## 1. 인사이트 (세션 간 맥락 보존) ⭐

### 1.1 이 W의 독특한 성격

**조건부 트랙**. R3 합의본에서 Phase 6 이월 → R4에서 "조건부 P6-W1 트랙"으로 분류 → P6 구조화하면서 **P6-W8**로 이동.

**발동 조건 (99 §4)**:
> "B2 벤치에서 평가자 3명 중 2명 이상 '중앙 공동 체감'이면 P6-W1 착수. 모두 '체감 없음'이면 P6-W1 폐기. 1명만 체감 시 사용자 판단."

즉 W8은 **자동 발동 아님**. W4 벤치 결과 보고 판정.

### 1.2 사용자 핵심 관점 (재확인)

**사용자 의견 (이번 세션 대화)**:
> "재질에서 오는 감도가 있는거니까 적용할꺼면 전반적으로 적용돼야 하는데 이게 사실 눈에 있는 수분들과 만나면 투명에 가까워지는 효과가 날거같아서 단순히 이렇게 이미지나 재질만 보고 판단하기는 어렵고 자연슬럽게 빛 반사 효과나 이런거로 인해 커버되지 않을까 하는 부분에 더 가까워, 정 어색하면 그때 가서 방향을 다시 결정하자에 가깝지"

**핵심**:
- 사용자는 **W8 발동을 선호하지 않음** — 자연 커버 기대
- "어색하면 그때" — 실기기 체감 기반 결정
- 따라서 W8 시작 전 "진짜 필요한가" 재확인 필수

### 1.3 Option E: "렌즈 재질 반투명 복원" (확정된 방향)

**핵심 수식 (R4 Patch 3 초안)**:
```glsl
// 중앙 완전 투명 영역에 렌즈 재질감을 약하게 씌움
// 원본 홍채 건드리지 않음 — 렌즈만 옅게 덧씌움

float centerProximity = 1.0 - smoothstep(0.0, iris_radius * 0.4, 
                                          distance(uv, iris_center));
float materialAlpha = 0.12 * centerProximity;  // 0.10~0.15 튜닝 대상

// 렌즈 색을 평균으로 요약 → 옅은 오버레이
vec3 lensMaterial = dot(lens.rgb, vec3(0.33)) * mix(vec3(1.0), lens.rgb, 0.5);
blended = mix(blended, lensMaterial, materialAlpha);
```

⚠️ **수식은 예시**. W8 시작 시 브레인스토밍에서 재확정 (Codex R4 단서).

### 1.4 Option A 기각 근거 (baseLum 하한, Gemini 제안)

**기각 사유**: 
- 원본 홍채의 어두운 디테일(속눈썹 그림자 등)도 같이 밝아짐 → detail 훼손
- iris 외곽(진한 림발 영역)도 영향 → 렌즈 경계 자연스러움 훼손

→ Option E가 우월: **원본은 건드리지 않고, 렌즈만 재질 오버레이**.

### 1.5 W3 renderMask hook과의 연결

**99 §1.2 C5 (Patch 4)**:
```glsl
// P6-W8 활성화 시: 동공 영역까지 반사 확장 가능
// #ifdef ENABLE_PUPIL_MATERIAL_RESTORE
//   renderMask = max(finalAlpha, smoothstep(iris_radius * 1.2, 0.0, dist));
// #endif
```

즉 W8은 두 단계:

1. **환경 반사 계층 확장** (renderMask 변경) — 동공 영역에도 환경 반사가 옅게 나타남
2. **렌즈 재질 오버레이** (Option E 수식) — 렌즈 평균 색의 재질감 중앙 영역에 적용

이 두 효과가 **합쳐서** "재질 반투명 + 눈물막 반사" 느낌을 구현.

### 1.6 W8 작업 순서

**단계 1**: W4 결과 해석 → 착수 판정

- 3명 모두 "공동 체감 없음" → W8 폐기, 트랙 종료
- 2명 이상 Y → W8 착수
- 1명 Y → 사용자 최종 판단

**단계 2** (착수 시): 구체 구현 브레인스토밍
- Option E 수식 최종화
- renderMask hook 실제 활성화 전략 (#ifdef vs runtime flag)
- centerProximity 범위 (0.4 vs 0.5 vs 0.3)
- materialAlpha 강도 (0.10 / 0.12 / 0.15)

**단계 3**: 구현 + 실기기 재검증
- 셰이더 수정
- B2 시나리오 재촬영 (W8 적용 전 vs 후)
- 개선 여부 확인

### 1.7 재질 복원 vs "추가 하이라이트" 혼동 금지

**주의**: 사용자 원래 의도는 "재질 감도 회복"이지 "각막 specular 하이라이트 추가"가 아님.

- **재질 감도**: 렌즈 전체에 은은한 반투명감. 동공 영역까지 포함. 상시 유지.
- **각막 specular**: 점 모양 하이라이트. 조명 있을 때만. 이미 C5 환경 반사 계층이 담당.

W8 구현 시 두 가지 혼동 금지. **Option E는 재질 감도 전용**.

### 1.8 W4 B2 결과에 따른 C5 반사 소스와의 상호작용

**시나리오 1 (B2: env-map 채택)**:
- C5 반사: env map 샘플링
- W8 renderMask 확장: 동공 영역에도 env map이 옅게 나타남 → 재질감 자연 복원
- Option E 재질 오버레이 추가 필요성: 낮을 수 있음 (env map이 이미 재질 역할)

**시나리오 2 (B2: periphery 채택)**:
- C5 반사: 주변 카메라 평균 색
- W8 renderMask 확장: 동공에 주변 평균 색 반영 → 자연스러움 중
- Option E 보강: 중간

**시나리오 3 (B2: OFF — 환경 반사 Phase 6 이월)**:
- C5 반사: 없음
- W8 renderMask 확장 무효 (반사 계층 자체가 0)
- Option E 수식만 적용: 렌즈 평균 색 재질 오버레이 단독

→ W8 구현은 **시나리오 3 대비** 기본 적용. 시나리오 1~2는 W8 필요성 자체가 감소.

### 1.9 렌즈 평균 색 계산

Option E 수식 `dot(lens.rgb, vec3(0.33))` 은 **순수 휘도**. 색 정보 손실. 재질감 표현에 부적합할 수 있음.

**대안**:
```glsl
// 렌즈 텍스처의 "주색" 샘플링 (중심부 외곽의 대표 색)
vec3 dominantLensColor = texture(uLensTexture, vec2(0.75, 0.5)).rgb;
// 또는 사전 계산된 평균 색 uniform 주입
vec3 lensMaterial = uLensAverageColor;  // CPU에서 계산해서 uniform 주입
```

W8 브레인스토밍에서 결정.

### 1.10 Gemini R3 §6 — Pupil cutout 강조 재확인

**Gemini R3 §6**:
> "부적절한 이월: 'Pupil cutout 동적 처리'는 W3-05에 포함해야 함. 렌즈 착용 시 동공이 텅 비어 보이는 현상은 경쟁사 대비 가장 큰 약점임."

Gemini는 W3-05 포함을 강하게 주장했으나 사용자가 "자연 커버 기대" 입장 → Phase 6로 이월 합의 (R4 Patch 3).

**Gemini 우려 요점**: "경쟁사 대비 가장 큰 약점". 즉 W4 체감 지표가 애매하게 나오면 **Gemini 우려를 반영해서 W8 착수 권장** 판단이 필요할 수 있음.

### 1.11 P6-W8 브레인스토밍 시 질문 (착수 확정 시)

1. **Option E 수식 확정**: centerProximity 범위, materialAlpha 강도.
2. **렌즈 평균 색 계산 방법**: dot 휘도 vs 주색 샘플링 vs 사전 계산 uniform.
3. **renderMask hook 활성 방식**: #ifdef vs runtime flag vs SKU 메타 연동.
4. **C5 반사 소스와의 시너지**: 시나리오별 W8 강도 차등?
5. **vertex 내부 텍스처 uniform 주입**: 성능 비용.
6. **실기기 재검증 매트릭스**: W4 시나리오 전부 재촬영? 선별?

### 1.12 구현 범위 (W8 착수 시)

**수정**:
- `shader_sources.cpp`:
  - renderMask hook 활성화 (C5 내부)
  - Option E 재질 오버레이 수식 추가 (applyLens 말미)
  - 새 uniform: `uLensAverageColor`, `uPupilMaterialStrength`
- `gpu_lens_renderer.cpp`:
  - SKU 로딩 시 렌즈 텍스처 평균 색 CPU 계산
  - uniform 주입
- `gpu_lens_renderer.h`:
  - uniform location
  - 새 API `setPupilMaterialEnabled(bool)` (내부용)

### 1.13 성능 예산

- renderMask 확장: smoothstep 1회. 무시.
- Option E 오버레이: mix 1회. 무시.
- 렌즈 평균 색 uniform: 1회만 주입 (SKU 바뀔 때).
- **총 런타임 비용: ~0**.

### 1.14 W8 폐기 판정 시 처리

만약 W4에서 3명 모두 "공동 체감 없음" → W8 폐기:
- 99_final_decision.md §4 표에 "P6-W8 트랙 폐기 (W4 2026-MM-DD 벤치 결과)" 기록
- 99 §6 미결 표 해당 행 제거
- renderMask hook은 여전히 C5에 유지 (추후 필요 시 활용)
- `uLensAverageColor` 같은 준비 uniform은 제거 or 유지 판단

### 1.15 W4 결과 대기 중에 W8 준비

**Claude 권장**: W4 벤치 실행 전에 W8 **수식 프로토타입만 준비**:
- Shader에 `#ifdef ENABLE_PUPIL_MATERIAL_RESTORE` 블록 추가
- 활성화 시 Option E 수식 즉시 동작
- W4 벤치 중 비공식으로 토글해서 "이게 있으면 공동 체감 사라지나?" 즉석 확인

이렇게 하면 W4 결과 + 비공식 W8 테스트까지 한 번에 수집. W8 착수 판정 근거 강화.

**Codex R3의 완벽주의** vs **Gemini의 실용성** 사이 타협안. W4 브레인스토밍 시 제안.

### 1.16 제품 임팩트 정리

**Gemini 경고**: "경쟁사 대비 가장 큰 약점"
**사용자 기대**: "빛 반사로 자연 커버"
**실측**: 20/20 중앙 투명

**결론**: 제품 타겟(아시아인 + 밝은 컬러 포인트 렌즈) 시나리오에서 **Pupil 공동 체감이 발생할 확률 높음**. W8 착수 가능성 30~60%. 폐기 시 사용자가 "내가 말한 대로 자연 커버됐네" 검증, 착수 시 "Gemini 우려 맞았네" 인정.

---

## 2. 배경/맥락
_TODO_

## 3. 전제 조건
_TODO: P6-W4 B2 벤치 결과, 2/3 이상 공동 체감_

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

- 99_final_decision.md §4 (P6-W1 조건부 트랙 — 이제 P6-W8로 재명명)
- 14_gemini_r3.md §6 (W3-05 포함 주장 원문)
- 사용자 대화 기록 (이번 세션) — "자연 커버 기대" 입장
- 15_asset_analysis.md §2.3 (20/20 중심 투명 실측)
- R4 Patch 3 (조건부 트랙 이관 합의)
- 이 세션 내 Option A 기각 / Option E 채택 논의
