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

// dist는 이미 /scaledRadius로 정규화됨 (0~1). 중심=0, 외곽=1.
// iris_radius * 0.4 대신 정규화 거리 0.4로 직접 비교.
float centerProximity = 1.0 - smoothstep(0.0, 0.4, dist);
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

### 2.1 조건부 트랙의 자기 정체

- **자동 발동 아님**. P6-W4 B2 "중앙 공동 체감" 지표 기반 **2/3 룰**로 판정.
- **폐기 가능성 존재**. 3명 모두 체감 없으면 이 W 전체 미수행.

### 2.2 사용자 입장 재확인

> "재질에서 오는 감도가 있는거니까... 이게 사실 눈에 있는 수분들과 만나면 투명에 가까워지는 효과가 날거같아서... 자연스럽게 빛 반사 효과나 이런거로 인해 커버되지 않을까... 정 어색하면 그때 가서 방향을 다시 결정하자"

**사용자는 W8 발동을 원하지 않음** — 자연 커버 기대. 발동 시 "어색하게 나왔다"는 반증.

### 2.3 Option E (재질 반투명 복원) 채택 배경

- Option A (baseLum 하한): 원본 홍채 detail 훼손 → 기각
- Option E (렌즈 재질 오버레이): 원본 건드리지 않음, 렌즈 평균 색을 옅게 덧씌움 → 채택

### 2.4 W3 renderMask hook과의 연결

이미 W3에서 `#ifdef ENABLE_PUPIL_MATERIAL_RESTORE` 분기 준비됨. W8 착수 시:
1. 해당 `#ifdef` 플래그 활성화
2. Option E 수식 구현

### 2.5 W8이 해결하지 않는 것

- Pupil 자체 검출 정확도 — W2 refiner 영역 (Phase 7+)
- 동공 동적 크기 변화 — 생리학적 반응, 범위 밖

---

## 3. 전제 조건

1. ✅ **P6-W4 B2 벤치 완료**
2. ✅ **Pupil 체감 지표 2/3 이상 Y** — 그렇지 않으면 W8 폐기
3. ✅ W3 완료 (renderMask hook 존재)
4. ✅ 사용자 최종 승인 (1/3만 Y인 경우)

---

## 4. 목표

1. **Option E 수식 최종 확정** — centerProximity 범위, materialAlpha 강도
2. **렌즈 평균 색 uniform 주입** — CPU 계산 (SKU 로딩 시 1회)
3. **renderMask hook 활성** — `#ifdef ENABLE_PUPIL_MATERIAL_RESTORE`
4. **실기기 재검증** — B2 시나리오 재촬영, W8 적용 전/후 비교

### 4.1 Definition of Done

- [ ] Option E 수식 셰이더 구현
- [ ] 렌즈 평균 색 계산 + uniform 주입
- [ ] renderMask hook 활성 빌드 variant
- [ ] B2 시나리오 4개 재촬영 (Pupil 체감 SKU 중심)
- [ ] 3명 재평가: "공동 체감 개선" Y/N
- [ ] 개선 확인 시 정식 빌드에 포함. 없으면 롤백.

### 4.2 Out of scope

- Pupil 동공 추적 정확도 향상 — W2 refiner 영역
- 각막 specular 하이라이트 추가 — W4 환경 반사 담당

---

## 5. 99에서 확정된 사항

> 🔧 **구현 시 참조**: [`P6_implementation_handoff.md`](P6_implementation_handoff.md) §4.2 dist 정규화 규약, §4.3 영역 경계 매핑 (centerProximity 0.0~0.4 / 평균 색 ROI 0.5~0.85), §4.4 GLSL 패스 규약 (material → C5 반사 순서, `#ifdef ENABLE_PUPIL_MATERIAL_RESTORE`). **착수 조건: W4 B2 = 2/3 이상 Y.**

### 5.1 Option E 수식 초안 (W8 시작 시 확정)

```glsl
// applyLens 말미, C5 반사 합성 이후 or 이전 (W8 브레인스토밍에서 순서 결정)
// dist는 이미 /scaledRadius로 정규화됨 (0~1). 중심=0, 외곽=1.
// iris_radius * 0.4 대신 정규화 거리 0.4로 직접 비교.
float centerProximity = 1.0 - smoothstep(0.0, 0.4, dist);
float materialAlpha = uPupilMaterialStrength * centerProximity;

vec3 lensMaterial = uLensAverageColor;  // CPU 계산 후 주입
blended = mix(blended, lensMaterial, materialAlpha);
```

⚠️ **Codex R4 단서**: "smoothstep 수식은 예시. 실제 구현 시 재확정".

### 5.2 uPupilMaterialStrength 기본값: 0.12

범위 0.10~0.15 튜닝. W8 시작 시 확정.

### 5.3 렌즈 평균 색 계산 (CPU)

```cpp
// SKU 로딩 시 1회
vec3 compute_average_color(const Bitmap& lens_texture) {
    // 외곽 링(r ∈ [0.5, 0.85]) 평균 색 추출
    // 내부 투명 영역과 바깥 페이드 제외
    glm::vec3 sum(0.0f);
    int count = 0;
    for (pixels) {
        float dist_norm = ...;
        if (dist_norm > 0.5 && dist_norm < 0.85) {
            sum += pixel.rgb;
            count++;
        }
    }
    return sum / float(count);
}
```

### 5.4 renderMask hook 활성

```glsl
#ifdef ENABLE_PUPIL_MATERIAL_RESTORE
    // dist는 이미 정규화됨. 1.2는 iris_radius 단위 정규화 값 (바깥 20% 포함).
    renderMask = max(finalAlpha, smoothstep(1.2, 0.0, dist));
#endif
```

**W8에서 실제 수식 재확정** (Codex R4 단서).

### 5.5 B2 재검증 매트릭스

- W4 B2 벤치 시 "공동 체감 Y"로 판정된 시나리오만 재촬영
- 이전 : Option E ON/OFF 비교
- 3명 재평가: "공동 체감 개선" Y/N

### 5.6 centerProximity — **0.4 기본 확정** (W8 R1 합의 3/3)

- 0.4 = 공동 커버(pupil 영역 대응) + 홍채 본체 보존 균형.
- 0.3은 공동 커버 부족, 0.5는 material 과다 침범 위험.
- 실기기 스위프 (선택적): [0.3, 0.5] 범위 3안 — 기본 0.4가 첫 구현 적정하면 sweep 생략.
- 출처: `P6-W8_brainstorm/synthesis.md` §1.

### 5.7 materialAlpha — **0.12 기본 확정** (W8 R1 합의 3/3)

- 문서 원 기본값 유지. 은은한 재질감 + 환경 반사와 결합 시 눈물막 반투명도 근사.
- 실기기 스위프 (선택적): [0.10, 0.15].
- 출처: `P6-W8_brainstorm/synthesis.md` §1.

### 5.8 평균 색 ROI — **외곽 링 `[0.5, 0.85]` 확정** (W8 R1 합의 3/3)

- 중심부(pupil cutout 영향) 제외 + 림발 영역(0.85~1.0) 제외.
- 렌즈 본체 material의 대표 색 추출 — 재질 덧씌움 색과 자연 동기화.
- 출처: `P6-W8_brainstorm/synthesis.md` §1.

### 5.9 레이어 순서 — **옵션 A (material → C5 반사) 확정** (W8 R1 합의 3/3)

- 깊이 순서: material(iris 깊이) → C5 반사(각막 표면 최외층).
- 반사 하이라이트를 material이 덮지 않음 → 광택 cue 보존.
- 옵션 B(반사 후 material)는 반사 효과를 cover 영역에서 덮어 기각.
- 출처: `P6-W8_brainstorm/synthesis.md` §1.

### 5.10 활성 방식 — **`#ifdef ENABLE_PUPIL_MATERIAL_RESTORE` 확정** (W8 R1 합의 3/3)

- W3 §5.10 `#ifdef` 원칙 연속. 조건부 트랙이므로 기본 바이너리 배제 원칙.
- **W8 빌드 default 1**, B2 재검증 통과 후 프로덕션 빌드 1 전환.
- 실패 시 0 롤백 (§5.12와 정합).
- 출처: `P6-W8_brainstorm/synthesis.md` §1.

### 5.11 Option A(baseLum 하한) 재고려 — **기각 유지 확정** (W8 R1 합의 3/3)

- 전제 "2/3 Y"가 성립하므로 Option A 부활 조건(1/3 Y) 아님.
- baseLum 하한은 원본 홍채 디테일 훼손 리스크 ("뿌연 안개" 효과).
- Option E 유지.
- 출처: `P6-W8_brainstorm/synthesis.md` §1.

### 5.12 W8 실패 시 롤백 — **`#ifdef` 비활성 + Phase 7+ 이월 확정** (W8 R1 합의 3/3)

체크리스트:
- [ ] `#define ENABLE_PUPIL_MATERIAL_RESTORE 0` 전환.
- [ ] W8 문서에 "수식 실패" 기록 + 실패 원인 (예: materialAlpha 0.15까지 올려도 체감 개선 없음).
- [ ] 코드는 주석/삭제 불필요. `#ifdef` 블록 그대로 유지 (향후 재시도 경로).
- [ ] B2 결과 "문제 있음"은 유지. "Option E가 해결 못함"만 기록.
- [ ] Phase 7+ 대안 후보 (참고):
  - Pupil 영역 별도 texture 블렌드 (디자이너 제작 아트 에셋).
  - Gaze estimation 기반 시선 방향 material 회전.
- 출처: `P6-W8_brainstorm/synthesis.md` §1, §5.

### 5.13 메타 노트 — Gemini R2/R3 Pupil cutout 우려 실증

W4 B2 벤치 2/3 Y 결과는 Gemini가 P5 R2/R3에서 지속 강조한 "동공 공동 현상"이 실기기 체감에서 실증된 것. **멀티-AI 교차 검토 프로세스의 가치 증명 사례**. 메모리 `feedback_multi_ai_orchestration_bias` 적용.

향후 Gemini의 "재질 연속성" 축 의견에 가중치 상향 근거.

출처: `P6-W8_brainstorm/synthesis.md` §3.

---

## 6. 미결 사항 (W8 브레인스토밍 R1 결과)

### 6.0 R1 결과 요약 (2026-04-24)

| 번호 | 원 쟁점 | 상태 | 반영 위치 |
|------|---------|------|-----------|
| 6.1 | centerProximity | ✅ **닫힘** (3/3) | §5.6 |
| 6.2 | materialAlpha | ✅ **닫힘** (3/3) | §5.7 |
| 6.3 | 평균 색 ROI | ✅ **닫힘** (3/3) | §5.8 |
| 6.4 | C5 순서 | ✅ **닫힘** (3/3 옵션 A) | §5.9 |
| 6.5 | 활성 방식 | ✅ **닫힘** (3/3 #ifdef) | §5.10 |
| 6.6 | Option A 재고려 | ✅ **닫힘** (3/3 기각 유지) | §5.11 |
| 6.7 | 실패 시 롤백 | ✅ **닫힘** (3/3) | §5.12 |

**가장 깔끔한 W.** 전 쟁점 3/3 합의, 미결 없음.

**실제 착수 조건:** W4 B2 벤치에서 2/3 이상 "중앙 공동 체감" Y 확정 후.

원문: `docs/workPaper/P6-W8_brainstorm/{codex,gemini,claude}_w8.md`.
종합: `docs/workPaper/P6-W8_brainstorm/synthesis.md`.

---

## (원 미결 사항 세부 — 참고용)

### 6.1 centerProximity 범위 (0.3/0.4/0.5)

Pupil 영역 넓이 조절. 브레인스토밍 시 3안 비교.

### 6.2 materialAlpha 강도 (0.10/0.12/0.15)

W8 브레인스토밍에서 결정. 기본 0.12.

### 6.3 렌즈 평균 색 계산 ROI

외곽 링 `[0.5, 0.85]` vs 중심부 제외 전체? 패턴 영향 고려.

### 6.4 순서: C5 반사 전/후

- **옵션 A**: 반사 전에 재질 오버레이 (반사가 재질 위에 얹힘)
- **옵션 B**: 반사 후에 재질 오버레이 (재질이 반사도 덮음)

**Claude 추천**: 옵션 A. 반사는 재질 위에 얹히는 것이 물리적.

### 6.5 `#ifdef` vs runtime flag

W3에서 `#ifdef` 선택 확정됨. W8도 그 방침 따름.

### 6.6 Option A 재고려?

B2 벤치 중 체감 지표가 "애매"하면 (1/3 Y) → Option A (baseLum 하한) 다시 고려? 
- **Claude 입장**: 기각 유지. 원본 훼손 리스크 여전.

### 6.7 W8 실패 시 (재검증 개선 없음)

- 롤백: renderMask hook `#ifdef` 비활성
- 사용자 보고: "체감 지표로는 Y였지만 실제 수식이 문제 해결 못함"
- Phase 7+로 이월 (다른 접근 필요)

---

## 7. W8 브레인스토밍 시작 체크리스트

### 7.1 읽을 파일

**필수**: P6-W0, P6-W8, P6-W4 B2 결과 report, 99 §4 (Pupil cutout 조건부 트랙)
**필수**: `14_gemini_r3.md` §6 (Gemini Pupil cutout 강조 원문)

### 7.2 송신 프롬프트

```
@docs/workPaper/P6-W8_pupil_material_conditional.md 읽고, 섹션 6 미결에
대해 입장 정리. docs/workPaper/P6-W8_brainstorm/{codex|gemini}_w8.md.

전제: W4 B2 벤치에서 2/3 체감 Y로 W8 착수 확정된 상태.

특히:
- 6.1 centerProximity 범위
- 6.2 materialAlpha 강도
- 6.4 반사 전/후 순서

Gemini는 R2/R3 Pupil cutout 우려의 실증(B2 체감 Y)에 대한 소회.

규칙: 새 쟁점 금지.
```

### 7.3 소요

- 수식 확정 + 셰이더 구현: 2h
- 렌즈 평균 색 CPU 계산: 1h
- 재검증 촬영 + 평가: 1.5h
- 튜닝 반복: 1~2h

**총 6h** (발동 시).

---

## 8. 완료 정의 + 다음 W 트리거

### 8.1 완료 정의 (발동 시)

§4.1 체크리스트 + 재검증 결과 "공동 체감 개선" 다수 동의.

### 8.2 완료 정의 (폐기 시)

- B2 체감 3/3 N → 문서에 "P6-W8 폐기 (W4 결과 근거)" 기록
- renderMask hook은 `#ifdef` 상태로 유지 (향후 재검토 여지)
- 99 §4 표에 "P6-W8 폐기" 갱신

### 8.3 커밋 전략 (발동 시)

- `docs(P6-W8): 섹션 2~8`
- `feat(gpu-lens): P6-W8 Option E 재질 반투명 복원 구현 (#ifdef 활성)`
- `feat(sdk): P6-W8 렌즈 평균 색 CPU 계산`
- `chore(bench): P6-W8 재검증 결과 + 튜닝 반영`

### 8.4 커밋 전략 (폐기 시)

- `docs(P6-W8): 폐기 결정 — W4 B2 결과 공동 체감 없음`
- (코드 변경 없음)

### 8.5 W9 머지 직전 확인

- W8 활성 빌드 variant가 기본 빌드에 포함될지?
- 개발 시 `-DENABLE_PUPIL_MATERIAL_RESTORE=ON` 등 CMake 옵션으로 제어

### 8.6 기대 효과 (발동 시)

- 경쟁사 대비 "Pupil 구멍" 약점 해소 (Gemini R2/R3 우려 해소)
- 렌즈 재질감 전체적 통일성 (동공 영역만 매끈함 다름 해소)
- Phase 7+에서 Pupil center 정교화 시 Option B 업그레이드 경로 확보

---

## 참조

- 99_final_decision.md §4 (Pupil cutout 조건부 트랙 — 99 원문에선 "P6-W1"이라 명명됐으나 P6 구조화 시 P6-W8로 재배치됨)
- 14_gemini_r3.md §6 (W3-05 포함 주장 원문)
- 사용자 대화 기록 (이번 세션) — "자연 커버 기대" 입장
- 15_asset_analysis.md §2.3 (20/20 중심 투명 실측)
- R4 Patch 3 (조건부 트랙 이관 합의)
- 이 세션 내 Option A 기각 / Option E 채택 논의
