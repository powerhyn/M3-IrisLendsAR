# P6-W4 Phase A 실기기 검증 — 시각 효과 인지 불가 이슈

**작성일:** 2026-05-18
**상황:** Phase A 코드 구현 완료 + feature/P6-Works 머지 후 실기기 검증 단계.
**요청:** 원인 분석 + 우리 진단(아래 §3) 검증 + 해결 방향 추천.

---

## 1. 컨텍스트

- **W3** (`P6-W3_env_reflection_scaffold.md` §5) — 환경 반사 가산 계층 scaffold + Fresnel 옵션 C + renderMask hook #ifdef 머지 완료 (PR #3, `91c81b9`).
- **W4 Phase A** — 3 프로토타입(OFF/EnvMap/Periphery) 활성화. 머지: `feature/P6-Works` 121a262.
  - sampleReflection 시그니처 확장: `(vec2 reflectUV, vec2 irisCenterAdjusted, float scaledRadius)`
  - Periphery annular ring `r∈[1.8, 2.5]` 8포인트 샘플링 (RING_R=2.15, N=8)
  - env_map texture upload + binding (unit 2)
  - JNI + Java wrapper + CameraGLView queueEvent setter
- **Demo UI** — VOLUME_UP 키로 mode 토글 (OFF→EnvMap→Periphery 순환) + Toast 표시.

---

## 2. 현재 셰이더 합성 식 (`shader_sources.cpp` applyLens)

```glsl
float finalAlpha = lens.a * uOpacity * edgeAlpha * eyelidMask;
// blendMode 분기 — Normal(0) / Multiply(1) / ScreenLinear(2) / TintLinearV2(5) / ColorReplaceLinear(7)
vec3 blended = [블렌드 결과];

// P6-W3 §5.1 반사 합성 (W3 §5.4 순서: 블렌드 → (디테일 W6) → 반사 → contact shadow)
float renderMask = finalAlpha;
#ifdef RENDER_MASK_HOOK_ENABLED
    renderMask = max(finalAlpha, smoothstep(1.2, 0.0, dist));  // W8 hook
#endif

vec2 reflectUV = (adjustedCoord - adjustedCenter) / scaledRadius * 0.5 + 0.5;
vec3 reflection = sampleReflection(reflectUV, adjustedCenter, scaledRadius);
float fresnel = calcFresnel(dist);  // W3 §5.8: smoothstep(0.7, 1.0, dist)
blended += reflection * fresnel * uReflectionIntensity * renderMask;

// contact shadow 적용
```

추가 컨텍스트:
- `dist`는 이미 `/scaledRadius`로 정규화 (line 922~923). 0=중심, 1=외곽.
- `edgeAlpha = smoothstep(1.0, featherStart, dist)`, `featherStart = 1.0 - uEdgeFeather`. uEdgeFeather 기본 ≈ 0.2 → featherStart ≈ 0.8.
- `uReflectionIntensity` 기본 0.3 (W3 §5.7, R1 미토론).
- `uOpacity` 기본 ≈ 0.5~0.8 (사용자 설정).
- `lens.a`: SKU 텍스처 알파 (보통 0.7~1.0).

sampleReflection:
```glsl
vec3 sampleReflection(vec2 reflectUV, vec2 irisCenterAdjusted, float scaledRadius) {
    if (uSourceType == 1) {
        return texture(uEnvMap, reflectUV).rgb;  // EnvMap
    }
    if (uSourceType == 2) {
        // Periphery — annular ring r∈[1.8, 2.5] 8 포인트
        const int N = 8;
        const float RING_R = 2.15;
        vec3 sum = vec3(0.0);
        for (int i = 0; i < N; ++i) {
            float angle = float(i) * (6.28318530718 / float(N));
            vec2 adjustedRingPoint = irisCenterAdjusted + RING_R * scaledRadius * vec2(cos(angle), sin(angle));
            vec2 ringUV = vec2(adjustedRingPoint.x / uFrameAspect, adjustedRingPoint.y);
            ringUV = clamp(ringUV, vec2(0.0), vec2(1.0));
            sum += texture(uCameraTexture, ringUV).rgb;
        }
        return sum / float(N);
    }
    return vec3(0.0);  // OFF
}
```

calcFresnel:
```glsl
float calcFresnel(float dist) {
    return smoothstep(0.7, 1.0, dist);  // 외곽 30%에서만 점증
}
```

---

## 3. 발견된 문제 + 우리 진단

### 증상
- **디버그용 자극적 env_map** (256×128 LDR PNG, 4 사분면 빨/노/녹/파, 채도 최대) 사용
- intensity 0.3 기본 (W3 §5.7 그대로) + LUMINANCE_TINT_LINEAR(ID=5) 블렌드
- VOLUME_UP 토글 시 OFF/EnvMap/Periphery **시각 차이 인지 불가**
- logcat에 mode 토글 메시지 정상 출력 (실제 uniform 변경됨 확인)
- shader compile error 없음

### 우리 진단
**Fresnel C와 edgeAlpha가 같은 외곽 영역에서 상쇄.** Fresnel 외곽 강조(`smoothstep(0.7, 1.0)`)와 edgeAlpha 외곽 페이드(`smoothstep(1.0, 0.8)`)가 동일 dist 구간에서 작동.

| dist | Fresnel | edgeAlpha | (Fresnel × edgeAlpha) |
|------|---------|-----------|----------------------|
| 0.7  | 0.0     | 1.0       | 0 |
| 0.85 | 0.5     | 0.5       | 0.25 |
| 0.95 | 0.83    | 0.25      | 0.21 |
| 1.0  | 1.0     | **0.0**   | **0** |

최대 가산 (dist≈0.9, reflection=빨강 1.0):
```
1.0 × 0.5 × 0.3 × (lens.a=0.8 × uOpacity=0.7 × edgeAlpha=0.5 × eyelidMask=1.0)
= 1.0 × 0.5 × 0.3 × 0.28
= 0.042  → RGB 11  → 거의 인지 불가
```

### 부수 관찰
- blendTintLinearV2는 `toSRGBFast` (sqrt)로 sRGB 공간 반환 → 그 위에 sRGB 공간 reflection 가산.
- Periphery 8포인트 ring은 얼굴 피부 평균색에 가까워서 (회갈색) EnvMap보다 더 약함.

---

## 4. 질문 (4개)

**Q1.** Fresnel + edgeAlpha 외곽 충돌 진단이 정확한가? 다른 후보 원인 있나?
- (예: blended sRGB와 reflection sRGB 가산 단위 문제, reflectUV 옵션 C 계산 오류, Periphery 좌표 변환 `adjustedRingPoint.x / uFrameAspect` 오류, mipmap LOD 문제 등)

**Q2.** 만약 위 충돌이 원인이라면 W3 §5.4/§5.5/§5.8 R1 결정의 **설계 누락**인가, 아니면 의도된 "자연스러움" 설계(외곽 페이드 + 미세 반사) 인가?
- W3 R1 합의는 외곽 강조 Fresnel + finalAlpha 기반 renderMask 둘 다 인정한 상태. 시각 효과 검증은 W4 B2 벤치 단계로 미뤘었음.

**Q3.** 해결 방향 우선순위 (다음 중 추천):
- (a) **renderMask 재정의** — finalAlpha에서 edgeAlpha 제거: `renderMask = lens.a * uOpacity * eyelidMask`. 렌즈 외곽 페이드 영역에 반사 sharp하게 끊김.
- (b) **Fresnel 영역 안쪽 이동** — `smoothstep(0.4, 0.8, dist)`. 외곽 60%에 반사 분포 + edgeAlpha와 일부만 겹침.
- (c) **intensity 대폭 상승** — 0.3 → 1.5~3.0. 구조 그대로 두고 곱셈으로 보완. "자연스러움" 속성 잃음.
- (d) **Composition 순서 변경** — 반사를 edgeAlpha 곱하기 전에 더하기? (구조 큰 변경)
- (e) 기타

**Q4.** 이게 Phase B (24클립 실기기 벤치) 진입 전 **해결 필요 수준**인가, 아니면 **Phase B에서 이 한계 그대로 평가**해도 되나?
- Phase B 목적은 OFF/env-map/periphery 비교. 셋 다 거의 안 보이면 비교 의미 자체 사라짐.

---

## 5. 참조 파일

- `cpp/src/gpu/shader_sources.cpp` — applyLens(line 914~), sampleReflection, calcFresnel
- `cpp/src/gpu/gpu_lens_renderer.cpp` — render() uniform 주입 (line 829~)
- `docs/workPaper/P6-W3_env_reflection_scaffold.md` §5.1/§5.4/§5.5/§5.7/§5.8
- `docs/workPaper/P6-W4_env_reflection_bench.md` §1.15 (W3 hand-off), §5.7 env_map

---

## 6. 응답 형식

`docs/workPaper/P6-W4_brainstorm/phase_a_issue_codex.md`로 저장. 각 질문 명시적 답변 + 권장 액션 1~3순위.
