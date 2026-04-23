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
_TODO_

## 3. 전제 조건
_TODO: P6-W2 블렌드 3종 안정 상태_

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

- 99_final_decision.md §1.2 C5
- 07_claude_r2.md I3 (realSpec 폐기 방향)
- 08_codex_r2.md I3 (반사 계층 분리 원문)
- 13_codex_r3.md §4 C5 (조건부 동의)
- 14_gemini_r3.md §7 (Periphery 재강조)
- 02_claude_response.md §Claude 자기비판 (specular 강도 0.7 과다)
