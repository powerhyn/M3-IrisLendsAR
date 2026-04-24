# P6-W3: 환경 반사 가산 계층 분리 + renderMask hook

> **상태**: 인사이트 작성 완료. 세부 계획 본문 작성 대기.
> **작성**: 2026-04-23
> **선행 의존**: P6-W2 (블렌드 3종 확정)
> **후속 의존**: P6-W4 (B2 벤치), P6-W8 (Pupil material)

---

## 1. 인사이트 (세션 간 맥락 보존) ⭐

### 1.1 이 W의 핵심 — "구조만 먼저, 소스는 나중"

**W3은 소스를 확정하지 않는다.** B2 벤치(P6-W4)가 env map vs periphery camera vs OFF를 판정할 예정. W3에서는:
- 셰이더 내부 **가산 합성 구조** 확립
- `renderMask` hook 박기 (P6-W8 Pupil material 조건부 활성화용)
- 반사 소스 인터페이스만 정의 (구체 소스는 W4 벤치 후)

### 1.2 Codex R3 원문 — realSpec/반사 계층 분리 근거

**Codex R2 원문**:
> "Claude의 임계값 하향과 Gemini의 sigmoid 전이는 둘 다 현재보다 덜 나쁠 뿐, 여전히 틀린 문제를 푼다. realSpec은 실반사 보호가 아니라 밝은 픽셀 보호다... 반사 계층 분리의 순서 문제는 오히려 장점이다. pigment blend가 먼저 iris/lens 층을 만들고, corneal reflection이 그 위에 얹히는 게 맞다. 'blend 이후 톤이 바뀐 픽셀 위에 반사를 더한다'는 게 잘못이 아니라 실제로 원하는 레이어링이다."

즉 **반사는 블렌드 결과 위에 가산(+)으로 얹히는 것**이 맞고, 블렌드 내부에서 반사를 다루는 건 개념 오류. W3은 이 레이어링을 코드로 실현.

### 1.3 99 §1.2 C5 공식 수식

```glsl
// 기본: 기존 finalAlpha 마스킹 (렌즈 영역에만 반사)
float renderMask = finalAlpha;

// P6-W8 활성화 시: 동공 영역까지 반사 확장 가능
// #ifdef ENABLE_PUPIL_MATERIAL_RESTORE
//   renderMask = max(finalAlpha, smoothstep(iris_radius * 1.2, 0.0, dist));
// #endif

blended += reflection * fresnel * renderMask;
```

⚠️ **Codex R4 중립 단서 (Patch 4)**:
> "주석 안의 `smoothstep` 수식은 최종 구현 수식으로 확정하면 안 된다."

이 smoothstep은 **예시**일 뿐. P6-W8 구현 시 실제 수식은 재확정.

### 1.4 Fresnel 항 미결

현재 99에 `fresnel` 변수가 등장하지만 구체 수식은 명시 안 됨. W3 브레인스토밍에서 확정:

**옵션 A**: Schlick's approximation
```glsl
float fresnel = F0 + (1.0 - F0) * pow(1.0 - max(dot(N, V), 0.0), 5.0);
// F0 = 0.04 (각막)
```

**옵션 B**: 간단한 각도 기반
```glsl
float fresnel = 1.0 - abs(dot(N, V));  // gaze 방향 비교
```

**옵션 C**: 분석적 구면 노멀 기반 (S1에서 삭제된 D1 노멀 재활용?)
- 이건 S1 롤백에서 제거된 분석 노멀을 일부 복원하는 셈. "재조율 vs 해체" 원칙 위반 소지.
- 대신 W3에서 **"가짜 Fresnel"** — iris 중심에서 거리 기반으로 대체:
```glsl
float fresnelApprox = pow(dist, 2.0);  // 외곽일수록 반사 강함
```

사용자 "refactor-vs-retune" 메모리 반영하면 **옵션 C의 가짜 Fresnel** 가장 깔끔. S1 롤백 복구 없음.

### 1.5 반사 소스 인터페이스 (W3에서 추상화)

```glsl
// W3에서 반사 소스 추상화 — 어느 방식이든 아래 함수만 구현하면 됨
vec3 sampleReflection(vec2 reflectUV, vec3 normal, vec3 viewDir) {
    // W4 벤치 결과에 따라 구체 구현 교체
    // 옵션 1 (env-map): return texture(uEnvMap, reflectUV).rgb;
    // 옵션 2 (periphery camera): return periphery_avg_color();
    // 옵션 3 (OFF): return vec3(0.0);
    return vec3(0.0);  // W3 기본은 OFF
}
```

W3 단계에서는 이 함수를 **no-op으로 두고 구조만**. W4에서 각 프로토타입이 이 함수를 오버라이드.

### 1.6 Gemini R3 재강조 — Periphery 샘플링 실시간성

**Gemini R3 §7**:
> "Codex는 전면 카메라 프레임에 광원이 없다고 비판하지만, 모바일 사용 환경(카페, 사무실)에서는 화면 가장자리에 천장 등기구나 창문이 걸리는 경우가 다수임. 에셋 env map은 '반사광이 눈을 따라 움직이지 않는' 치명적 결함이 있음을 재강조함."

즉 **periphery camera가 성공하면 env map보다 우월** (실시간 환경 반응). 실패 조건은 "얼굴이 화면 대부분을 차지하는 경우". W3 추상화는 양쪽 모두 구현 가능한 인터페이스 필수.

### 1.7 반사 강도(intensity) 튜닝 파라미터

S1에서 삭제된 D1의 고정 조명 specular 강도가 `0.7`였음 (Claude 편향 자인: "0.25~0.35가 자연스러움"). W3 반사 계층의 기본 강도는?

**Claude R1 자기비판 §12**:
> "specular * 0.7 강도: ON/OFF 체감 목적으로 과다. 자연스러운 값은 **0.25~0.35**."

**W3 초기 값**: `reflection_intensity = 0.3` (uniform). 실기기 테스트 시 튜닝 가능하게 노출.

### 1.8 realSpec 완전 폐기 조건부 — W3이 발동 조건

99 §1.1 D3:
> "조건부 폐기: I1 환경 반사 계층 도입 성공(B2) 확정 후 제거. I1 실패 시 재평가"

W3 자체는 "구조만 도입"이라 **D3 완전 폐기는 아직 아님**. B2 벤치(W4)가 성공해야 realSpec이 돌아올 일이 없다고 확정.

W3 완료 시점: "C5 스캐폴드 + renderMask hook + 반사 소스 추상화 완료, 모드 OFF (sampleReflection returns 0)".

### 1.9 세션 간 맥락: W3과 W4의 미묘한 경계

**W3**: 구조 (셰이더 인프라, 함수 시그니처, uniform 설계).
**W4**: 소스 (env-map / periphery / OFF 실기기 비교).

W3 브레인스토밍에서 **W4 벤치 매트릭스를 미리 설계**해두면 W4 시작 시 프로토타입 바로 돌릴 수 있음. 따라서 W3 §7 체크리스트에 "W4 벤치 매트릭스 drafts" 포함 권장.

### 1.10 W3에서 수정할 파일

- 수정:
  - `cpp/src/gpu/shader_sources.cpp` — `LENS_OVERLAY_FRAGMENT` 내부. `applyLens` 마지막 단계에 반사 합성 추가.
  - `cpp/include/iris_sdk/gpu/shader_manager.h` — 새 uniform 선언 (uEnvMap, uReflectionIntensity 등)
  - `cpp/src/gpu/gpu_lens_renderer.cpp` — 새 uniform 처리
- 신규 없음 (W4에서 프로토타입 코드 신규 추가)

### 1.11 주의 사항 (Codex R3 경고)

**Codex R3 §6 부분 동의**:
> "C5 조건부 동의: 반사 계층 분리는 맞다. 반사 소스는 B2 결과 후다."

즉 W3 시점에 소스 확정하지 말 것. W3 완료 시 Codex에게 "W3 범위가 올바른가, 소스 미정 상태가 괜찮은가" 재확인 가능.

### 1.12 W3 브레인스토밍 시 Codex/Gemini에게 던질 질문

1. **Fresnel 구체 수식**: Schlick vs 각도 기반 vs iris-dist 가짜 Fresnel? 각 렌즈 타입별 차이?
2. **반사 강도 기본값**: 0.25 / 0.3 / 0.35 중 W3 기본 uniform 값?
3. **`sampleReflection` 추상화 방식**: 함수 포인터 vs #ifdef 분기 vs 셰이더 컴파일 variant?
4. **uniform 명명**: `uReflectionIntensity`, `uEnvMapTexture`, `uReflectionSourceType` 등. 네이밍 관례 확인.
5. **블렌드 결과 위 가산 순서**: 현재 블렌드 후 바로 `blended += ...`. contact shadow와 디테일 재주입(C10)과의 순서 명확화.
6. **P6-W8 renderMask hook 활성 전환 절차**: 런타임 flag vs 컴파일 타임 매크로 중 뭐가 안전?
7. **Codex R3 C5 "소스 미정" 입장 재확인**: 벤치 전 구현 범위가 적절한지.

### 1.13 성능 예산

- 추가 FBO pass 0개 엄수
- 추가 fetch: env map 1 or periphery 4 (W4 벤치에서 확정)
- Fresnel 계산: pow 1~2회. 무시 가능
- 최종 합성 (mul + add): 몇 ALU, 무시

### 1.14 Composition 순서 제안

```glsl
// applyLens 내부 순서 (W3 적용 후)
vec3 blended = [blend result];              // W2 완료된 블렌드
blended += detail_reinjection();            // W6 C10 (제외 영역 있음)
blended += reflection * fresnel * renderMask;  // W3 C5 (신규)
blended = mix(blended, contact_shadow);     // 기존
```

⚠️ **순서 중요**: 디테일 재주입이 반사 전에 와야 한다 (Codex R3 C10 "spec/reflection 제외" 조건). W3에서 순서 확정.

---

## 2. 배경/맥락

### 2.1 W3의 핵심 — "구조만 먼저"

환경 반사 **가산 합성 구조**를 셰이더에 박는 W. 소스(env-map vs periphery vs OFF)는 W4 벤치에서 확정. W3는:
- `blended += reflection * fresnel * renderMask` 합성 지점 확립
- `sampleReflection()` 추상화 함수 정의 (기본은 no-op)
- renderMask hook 구조 (W8 조건부 활성화용)
- Fresnel 수식 확정

W3 완료 시점에 실기기 렌더링은 **변하지 않아야 함** — 반사 소스가 no-op이므로 기존과 동일 결과.

### 2.2 99 C5가 이 W에서 실현되는 것

99 §1.2 C5 원문:
```
환경 반사 가산 계층 분리. 기본: float renderMask = finalAlpha;
blended += reflection * fresnel * renderMask;
**반사 소스는 B2 벤치로 결정**. **renderMask hook**: P6-W8 활성화 시
#ifdef ENABLE_PUPIL_MATERIAL_RESTORE 분기로 ... 확장 가능.
```

W3는 이 C5의 **구조 부분만** 실현. 반사 소스 확정은 W4, hook 활성화는 W8.

### 2.3 이 W가 해결하는 다른 결정

- **D3 realSpec 폐기** (99 §1.1): realSpec은 S1에서 **코드 삭제됨**. 99 상태는 "조건부 폐기" — B2 성공 시 확정 폐기. W3가 **반사 계층 구조를 도입**하면, B2 결과에 따라 W4에서 최종 폐기 결정. W3 자체는 realSpec을 건드리지 않음 (이미 삭제 상태).

### 2.4 이 W가 해결하지 **않는** 것

- 반사 소스 확정 — W4
- 반사 강도 실제 튜닝 — W4 (벤치 중 관찰), W9 (통합)
- renderMask hook 활성 — W8 조건부
- Pupil material 오버레이 — W8

### 2.5 Claude 편향 경계

**Claude R1 "분석 노멀 + 고정 조명" 폐기 이후**: S1에서 D1 코드 삭제됨. W3에서 **다시 Claude가 "가짜 Fresnel" 수식을 제안**하는데, 이게 D1의 변주일 수 있음.

**경계선**:
- ❌ 고정 light vector (`vec3(0.3, 0.4, 1.0)`) 재도입 — D1 복귀 금지
- ❌ 분석 노멀 라이팅 계산 — D1 복귀 금지
- ✅ 분석 노멀을 **Fresnel 계산 입력**으로만 사용 — 라이팅 아닌 반사 방향 가중치

"가짜 Fresnel"도 S1 D1 재도입으로 비판받을 수 있음. W3 브레인스토밍에서 재검증.

---

## 3. 전제 조건

1. ✅ **W2 완료** — 블렌드 3종 안정. TintLinearV2/Multiply/ScreenLinear 등록 + realSpec 삭제 확인.
2. ✅ **W1 완료** — EyeRenderPacket. `reflection_dir` optional 필드 준비됨.
3. ✅ **S1 상태 확인** — shader_sources.cpp에 `uHighlightEnabled` 및 각막 하이라이트 블록 없는지 재확인 (있으면 W3 구현 충돌).
4. ✅ **99 §1.2 C5 + 12_claude_r3.md §Watch out** 읽기 (Claude 편향 회피)

---

## 4. 목표

**W3 완료 시 달성 상태**:

1. **셰이더에 반사 계층 통합점 존재**: `applyLens` 말미에 `blended += reflection * fresnel * renderMask;` 패턴 등록
2. **`sampleReflection()` 추상화 함수 정의** — 기본 `return vec3(0.0)`, 즉 no-op
3. **Fresnel 수식 확정 및 구현** (W3 브레인스토밍에서 결정)
4. **renderMask hook 준비** — `#ifdef ENABLE_PUPIL_MATERIAL_RESTORE` 분기 구조
5. **새 uniform 선언**: `uReflectionIntensity` (0.0~1.0, 기본 0.3), `uSourceType` (0=OFF, 1=EnvMap, 2=Periphery) 등
6. **실기기 회귀 없음** — 반사 소스 OFF 상태라 기존 렌더링과 동일 결과

### 4.1 Definition of Done

- [ ] shader_sources.cpp `applyLens` 함수에 반사 합성 지점 존재
- [ ] `sampleReflection()` 함수 정의, no-op 기본 동작
- [ ] Fresnel 항 구체 수식 문서화 + 구현
- [ ] renderMask hook `#ifdef` 블록 존재
- [ ] 새 uniform들 gpu_lens_renderer.cpp에서 location 캐시 + 주입
- [ ] 빌드 통과
- [ ] 실기기 회귀 없음 — W2 시각 결과와 동일

### 4.2 Out of scope

- 반사 소스 실제 소스 구현 — W4
- 반사 강도 값 튜닝 — W9
- Pupil material 오버레이 — W8
- compute shader 도입 — 99에서 기각 상태

---

## 5. 99에서 확정된 사항

> 🔧 **구현 시 참조**: [`P6_implementation_handoff.md`](P6_implementation_handoff.md) §4.2 dist 정규화 규약 (**`iris_radius` 곱하기 금지, 이중 정규화**), §4.3 영역 경계 매핑 (Fresnel 0.7~1.0), §4.4 GLSL 패스 규약 (블렌드→디테일→W8 hook→C5 반사→Pupil material→contact shadow).

### 5.1 C5 합성 수식 구조

```glsl
// applyLens 내부, blendMode 분기 후
// 사전 조건: dist는 이미 /scaledRadius로 정규화된 값 (0=중심, 1=외곽).
vec3 blended = [블렌드 결과];  // ID별 분기 결과

// [선택적] C10 디테일 재주입 (W6 §5.2 구현, spec/reflection 제외 영역)
// blended *= vec3(detailMul);  ← W6 §5.2 순서: 블렌드 → 디테일 → 반사

// C5 반사 합성 (W3 §5.8/§5.11/§5.12 R1 확정 반영)
float renderMask = finalAlpha;

// W8 활성화 시 renderMask 확장 (§5.5 참조, dist는 정규화 — iris_radius 곱하기 금지)
#ifdef ENABLE_PUPIL_MATERIAL_RESTORE
    renderMask = max(finalAlpha, smoothstep(1.2, 0.0, dist));
    // ⚠️ Codex R4 단서: smoothstep 인자 예시. W8 구현 시 실기기 튜닝.
#endif

// reflectUV: iris local 좌표 (§5.12 R1 확정 — 노멀 없음)
vec2 reflectUV = (uv - iris_center_uv) / iris_radius_uv * 0.5 + 0.5;

// sampleReflection: uniform 스위치 방식 A (§5.11 R1 확정)
vec3 reflection = sampleReflection(reflectUV);

// calcFresnel: 옵션 C 가짜 Fresnel (§5.8 R1 확정 — dist 기반, 노멀 없음)
float fresnel = calcFresnel(dist);

blended += reflection * fresnel * uReflectionIntensity * renderMask;

// contact shadow 기존 적용
blended = apply_contact_shadow(blended);
```

### 5.2 sampleReflection 추상화 (W3 no-op 구현)

```glsl
// W3 기본 (no-op)
vec3 sampleReflection(vec2 reflectUV, vec3 normal, vec3 viewDir) {
    return vec3(0.0);
}

// W4에서 각 프로토타입이 이 함수를 치환:
// 프로토타입 2 (env-map):
//   return texture(uEnvMap, reflectUV).rgb;
// 프로토타입 3 (periphery):
//   return sample_periphery_avg();
```

### 5.3 Fresnel 옵션 (W3에서 선택)

**옵션 A (Schlick)**: 물리적 정확도 ↑, 비용 중간
```glsl
float F0 = 0.04;  // 각막
float fresnel = F0 + (1.0 - F0) * pow(1.0 - max(dot(N, V), 0.0), 5.0);
```

**옵션 B (각도 기반 단순)**: 비용 저, 직관적
```glsl
float fresnel = 1.0 - abs(dot(N, V));
```

**옵션 C (가짜 Fresnel — iris 거리 기반)**: 노멀 계산 없음, 비용 최저. 외곽일수록 반사 강함
```glsl
// dist는 이미 /scaledRadius로 정규화됨 (0~1). 추가 나누기 금지.
float fresnel = pow(dist, 2.0);
```

**Claude 추천**: 옵션 C — 노멀 계산(D1 변주)을 회피. 외곽 반사 강화는 실제 각막 특성과도 일치 (limbal 근처 하이라이트). 구현 비용도 최저.

**W3 브레인스토밍에서 확정**.

### 5.4 composition 순서 (고정)

1. 블렌드 (W2) → `blended`
2. 디테일 재주입 (W6, 예정) → `blended *= detail`
3. **반사 합성 (W3)** → `blended += reflection * fresnel * renderMask`
4. contact shadow → `blended = apply_shadow(blended)`

**Codex R3 §1 C10 경고**: "detail reinjection은 반사 위에 다시 곱해지면 안 된다" → 이 순서(디테일 먼저 → 반사 나중)로 고정.

### 5.5 renderMask hook 구조 (Codex R4 Patch 4 단서)

```glsl
// 참고: 기존 shader의 dist는 이미 iris_radius 기준 정규화 (0~1 범위)
//       → float dist = distance(adjustedCoord, adjustedCenter) / scaledRadius;
float renderMask = finalAlpha;

#ifdef ENABLE_PUPIL_MATERIAL_RESTORE
    // W8에서 활성 — 동공 영역까지 반사 확장
    // 아래 smoothstep 수식은 예시. W8 구현 시 실제 수식 재확정.
    // dist는 이미 정규화됐으므로 iris_radius 곱하기 금지 (이중 정규화 오류)
    renderMask = max(finalAlpha, smoothstep(1.2, 0.0, dist));
#endif
```

**`#ifdef` vs runtime flag**:
- **`#ifdef`**: 컴파일 타임 분기. 바이너리 2종. 깔끔하지만 변경 시 재컴파일.
- **`uniform int uPupilRestoreEnabled`**: 런타임 분기. 단일 바이너리. 동적 토글 가능.

**Claude 추천**: **`#ifdef`**. 이유: W8이 조건부 발동이라 "빌드에 포함하느냐"가 결정 포인트. 발동 안 하면 코드 자체 미포함이 깔끔.

**W3 브레인스토밍에서 확정**.

### 5.6 uniform 신규

```cpp
// gpu_lens_renderer.h LensUniforms struct 추가
GLint uReflectionIntensity = -1;   // float 0~1, 기본 0.3
GLint uSourceType = -1;            // int 0=OFF, 1=EnvMap, 2=Periphery
GLint uEnvMap = -1;                // sampler2D (W4에서 env-map 프로토타입 사용 시)
// ... W4에서 더 추가 가능
```

### 5.7 반사 강도 기본값: 0.3 (Claude R1 자기비판 §12)

S1에서 삭제된 D1의 specular 강도는 0.7. Claude가 자기비판에서 "자연스러운 값 0.25~0.35" 인정. W3 초기값 **0.3** (uniform으로 튜닝 가능).

### 5.8 Fresnel 수식 — **옵션 C (가짜 Fresnel) 확정** (W3 R1 다수 2/3)

iris 거리 기반 근사. 실제 Fresnel은 grazing angle(시선과 표면이 이루는 각도가 큼)에서 반사 강해짐. 각막 기하에서 **iris 외곽으로 갈수록 grazing**이므로 **중심=0, 외곽=1** 방향.

```glsl
// calcFresnel — 옵션 C 가짜 Fresnel (노멀 벡터 없음, dist 기반)
// dist는 이미 /scaledRadius로 정규화 (0=중심, 1=외곽)
float calcFresnel(float dist) {
    return smoothstep(0.7, 1.0, dist);
}
```

- **초기 boundary (0.7, 1.0):** iris 내부 70%는 반사 0, 외곽 30%에서 점증 → 1. 물리 Fresnel(grazing에서 강함) 근사.
- **W4 B2 실기기 튜닝:** inner boundary [0.6, 0.7, 0.8] 스위프 후보 (외곽 boundary 1.0 고정).
- **노멀 벡터 불필요** → D1 재도입 없음 (3/3 모델 공통 확인).
- **W4 B2 재검토 조항:** 실기기 벤치에서 "시점 의존성 부족으로 효과 약함" 피드백 2/3 이상 → 후속 W(W8 또는 별도)에서 옵션 A(Schlick) 전환 검토. W3 scaffold는 옵션 C로 완료.
- 출처: `P6-W3_brainstorm/synthesis.md` §2.

### 5.9 분석 노멀 — **W3 제외 확정** (W3 R1 다수 2/3)

- S1 D1 해체 취지 유지. Codex "최소 노멀" 제안은 6.1이 후속 W에서 A/B로 전환될 때 재고.
- W3 shader 코드에서 normal vector 계산 경로를 새로 만들지 않음.
- 출처: `P6-W3_brainstorm/synthesis.md` §2.

### 5.10 renderMask hook 활성 — **`#ifdef` (컴파일 타임) 확정** (W3 R1 다수 2/3)

- `#ifdef RENDER_MASK_HOOK_ENABLED` 블록으로 감싸고 **기본 빌드에서 0으로 비활성**.
- 프로덕션 바이너리에서 dead code + 런타임 오버헤드 제거.
- **W4 벤치 전용 debug 빌드:** CMake 옵션으로 `RENDER_MASK_HOOK_ENABLED=1` 켠 APK를 별도 산출. 벤치 중 on/off 비교.
- Claude R1 원안 uniform flag는 소수 의견으로 Codex+Gemini 다수에 의해 뒤집힘 (편향 경계 작동).
- 출처: `P6-W3_brainstorm/synthesis.md` §2.

### 5.11 sampleReflection — **방식 A (uniform 스위치) 확정** (W3 R1 합의 3/3)

```glsl
uniform int uSourceType; // 0=OFF, 1=env-map, 2=periphery
vec3 sampleReflection(vec2 uv) {
  if (uSourceType == 1) return texture(uEnvMap, uv).rgb;
  if (uSourceType == 2) return periphery_sample(uv);
  return vec3(0.0);  // OFF
}
```

- 단일 바이너리, 런타임 토글. W3 scaffold는 `uSourceType=0` (OFF) 기본.
- W4 B2 벤치가 3프로토타입 A/B/C 비교할 때 uniform 스위치로 실시간 전환.
- 출처: `P6-W3_brainstorm/synthesis.md` §1.

### 5.12 reflectUV — **옵션 C (iris local 좌표) 확정** (W3 R1 다수 2/3)

```glsl
vec2 reflectUV = (uv - iris_center_uv) / iris_radius_uv * 0.5 + 0.5;
```

- 6.1/6.2에서 노멀 제외 확정 → 옵션 B(reflect 기반) 자동 배제.
- env-map/periphery 공통 UV 공간 → 방식 A uniform 스위치와 정합.
- 출처: `P6-W3_brainstorm/synthesis.md` §2.

### 5.13 env_map 에셋 — **demo assets 위치 + LDR PNG 확정** (W3 R1 다수 2/3 위치 + 3/3 포맷)

- **위치:** `android/demo-app/src/main/assets/env/` (demo 주도).
- **포맷:** 256×128 LDR PNG RGB (3/3 합의).
- **초기 에셋 3종:**
  - `env_default_256x128.png` (기본/스튜디오 라이팅)
  - `env_office.png` (실내 형광)
  - `env_outdoor.png` (실외 낮)
- **SDK 내장 보류:** W8~W9 단계에서 필요성 판단 후 결정. P6 범위 아님.
- 출처: `P6-W3_brainstorm/synthesis.md` §2.

### 5.14 B2 벤치 매트릭스 초안 — **4×2×3 = 24 클립 확정** (W3 R1 합의 3/3 골격)

| 차원 | 값 |
|------|-----|
| **환경 (4)** | 실내 형광, 창가 측광, 야간 실내, 실외 낮 |
| **동작 (2)** | 정면 미세(블링크 포함), head turn (±15°) |
| **프로토타입 (3)** | OFF / env-map / periphery |
| **SKU** | Tint(Linear)V2 + 고발광 iris_mat_B |

- **우선 비교:** env-map vs OFF (Claude 제안). periphery는 2차.
- **Pupil 체감 Y/N:** 각 클립에서 "중앙 공동 체감" 별도 체크. W8 조건부 트랙 발동 근거 (2/3 이상 Y면 W8 착수).
- **실시간 체감 (Gemini 강조):** 촬영 직후 앱에서 즉시 A/B 토글 (방식 A uniform 스위치 기반).
- W4 시작 시 본 매트릭스를 `P6-W4_env_reflection_bench.md`에 복사하고 세부 확정.
- 출처: `P6-W3_brainstorm/synthesis.md` §3.

---

## 6. 미결 사항 (W3 브레인스토밍 R1 결과)

### 6.0 R1 결과 요약 (2026-04-24)

| 번호 | 원 쟁점 | 상태 | 반영 위치 |
|------|---------|------|-----------|
| 6.1 | Fresnel 수식 (A/B/C) | ✅ **닫힘** (2/3 다수 C) + W4 재검토 조항 | §5.8 |
| 6.2 | 분석 노멀 포함 여부 | ✅ **닫힘** (2/3 다수 제외) | §5.9 |
| 6.3 | renderMask hook (#ifdef vs uniform) | ✅ **닫힘** (2/3 다수 #ifdef) | §5.10 |
| 6.4 | sampleReflection 추상화 | ✅ **닫힘** (3/3 합의 방식 A) | §5.11 |
| 6.5 | reflectUV | ✅ **닫힘** (2/3 다수 옵션 C) | §5.12 |
| 6.6 | env_map 에셋 위치/포맷 | ✅ **닫힘** (위치 2/3, 포맷 3/3) | §5.13 |
| 6.7 | B2 매트릭스 | ✅ **닫힘** (3/3 합의 4×2×3=24) | §5.14 |

참여 모델: Codex, Gemini, Claude.
원문: `docs/workPaper/P6-W3_brainstorm/{codex,gemini,claude}_w3.md`.
종합: `docs/workPaper/P6-W3_brainstorm/synthesis.md`.

**미결 없음.** 7개 쟁점 모두 R1 결론. 6.1은 옵션 C 확정이지만 W4 B2 결과에 따라 후속 W에서 옵션 A 재검토 조항 유지(§5.8).

**편향 경계 작동:** 6.3에서 Claude 원안(uniform flag)이 Codex+Gemini 다수(#ifdef)로 뒤집힘. Claude 자기비판 메모리(`multi-ai-orchestration-bias`) 적용 결과.

---

## 7. W3 브레인스토밍 시작 체크리스트

### 7.1 읽을 파일

**필수**:
1. P6-W0 §1
2. P6-W3 이 문서 전체
3. 99 §1.2 C5 + §1.1 D3 + R4 Patch 4
4. 02_claude_response.md §Claude 자기비판 — D1 폐기 근거
5. 13_codex_r3.md §6 C5 "조건부 동의" 원문

**선택**:
- 04_codex_response.md §A — Codex env-map 주장 원문
- 14_gemini_r3.md §7 — Gemini Periphery 재강조

### 7.2 송신 프롬프트

```
@docs/workPaper/P6-W3_env_reflection_scaffold.md 읽고, 섹션 6 미결 7개에
대해 각자 입장 정리 후 docs/workPaper/P6-W3_brainstorm/{codex|gemini}_w3.md
로 작성해줘.

특히:
- 6.1 Fresnel 수식 (A Schlick / B 각도 / C 가짜)
- 6.2 분석 노멀 계산 포함 여부 (D1 재도입 리스크)
- 6.3 hook 활성 방식 (#ifdef vs uniform)
- 6.4 sampleReflection 추상화 방식

규칙:
- 새 쟁점 제기 금지
- 각 항목 "추천 + 근거 1~2줄"
- Fresnel C 옵션이 D1 재도입으로 보이는지 명시 판단
```

### 7.3 예상 대립 지점

- 6.1 Fresnel: Codex가 옵션 A (Schlick) 주장할 가능성. Claude C 옵션 "가짜 Fresnel"이 정교함 부족으로 비판받을 수도.
- 6.4 추상화: Codex가 "셰이더 variant 2~3개" 주장 가능. 성능 최적 주장으로.

### 7.4 1시간 브레인스토밍 예상

```
0~5분    송신
5~20분   응답 대기
20~35분  7개 쟁점 정리
35~50분  Fresnel 심층 논의 (주된 대립)
50~60분  합의 + W4 벤치 매트릭스 draft
```

### 7.5 구현 예상 소요

- Fresnel + reflectUV + sampleReflection 추상화: 1.5h
- renderMask hook + #ifdef 구조: 30분
- uniform 추가 + 주입: 45분
- 실기기 회귀 없음 확인: 30분

**총 2.5~3h**.

---

## 8. 완료 정의 + 다음 W 트리거

### 8.1 완료 정의

§4.1 체크리스트 전체 ✅.

### 8.2 커밋 전략

**커밋 1**: `docs(P6-W3): 섹션 2~8 본문 작성`
**커밋 2**: `feat(gpu-lens): P6-W3 환경 반사 가산 계층 구조 + sampleReflection 추상화 (no-op 기본)`
**커밋 3**: `feat(gpu-lens): P6-W3 renderMask hook #ifdef 구조 (W8 활성 대기)`
**커밋 4** (선택): `feat(gpu-lens): P6-W3 Fresnel 수식 구현 (옵션 C)`

### 8.3 다음 W 트리거

**P6-W4 (B2 벤치) 시작 조건**:
- W3 완료 (구조 + 추상화)
- env-map 에셋 준비 방안 확정
- Pupil 체감 지표 수집 방법 확정

**P6-W5 / W6 / W7 시작 조건**:
- W3 완료 (이들은 W3 구조 위에서 돌아감)
- W5/W6/W7는 **W4와 병렬 가능** (반사 소스 확정 안 돼도 진행 가능)

### 8.4 W3 실패 시 롤백 전략

- W3 커밋 revert → W2 상태 복귀
- 반사 계층 구조만 되돌리면 렌더링 동작 영향 없음 (no-op이었으므로)

### 8.5 W3 성공 시 기대 효과

- **구조 완성**: 반사 합성 지점 확립. W4/W8에서 즉시 활용.
- **realSpec 대체 경로 확보**: W4 B2 성공 시 "반사 계층이 있으므로 realSpec 완전 폐기" 바로 확정.
- **W8 조건부 트랙 준비**: renderMask hook이 이미 있음. W8 착수 시 `#ifdef` 활성만으로 Pupil material restore 시작 가능.

### 8.6 W3이 Phase 6에서 갖는 "기반 역할"

W1이 "입력 계약", W2가 "블렌드 수식"이라면 W3는 **"확장 포인트"** — 반사, Pupil material, 향후 Phase 7+ 기능 추가 시 여기서 통합. Phase 6 중반부에 구조 설계 완료된다는 의미.

---

## 참조

- 99_final_decision.md §1.2 C5
- 07_claude_r2.md I3 (realSpec 폐기 방향)
- 08_codex_r2.md I3 (반사 계층 분리 원문)
- 13_codex_r3.md §4 C5 (조건부 동의)
- 14_gemini_r3.md §7 (Periphery 재강조)
- 02_claude_response.md §Claude 자기비판 (specular 강도 0.7 과다)
