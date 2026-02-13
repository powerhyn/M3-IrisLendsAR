# P4-W2-01: Sclera Protection + Contact Shadow 구현

## 작업 개요
- **Phase**: P4 (시각적 리얼리즘 — Visual Fidelity)
- **기간**: TBD
- **상태**: ⏳ 대기
- **선행 조건**: P4-W1-03 완료 + 주관 평가 ≥ 3.0/5.0
- **근거**: 브레인스토밍 Section 7, 8, 10, 13, 14, 16 합의

## 목표

1. **Sclera-Aware Alpha**: 흰자위 영역에서 렌즈 alpha를 자동 감쇠하여 랜드마크 오차 보정
2. **Contact Shadow**: 상안검 경계 아래 부드러운 그림자로 "가려짐" 공간감 부여
3. **이중 감쇠 방지**: mask 전이 구간과 shadow 적용 영역 분리

## Phase Gate 정책

```
Gate 2 (Phase 2 → Phase 3 진입):
  필수 조건:
  ├─ Luminance Tint 모드에서 주관 평가 ≥ 3.0/5.0 (최소 2명)
  ├─ sRGB vs Linear A/B 비교 결과 문서화
  └─ Sclera Protection 오탐률 < 10% (밝은 홍채 포함 테스트)
```

## 수정 대상 파일

| # | 파일 | 수정 내용 |
|---|------|-----------|
| 1 | `CameraGLRenderer.kt` | GLSL sclera/shadow 함수 + feature flag 분기 |

## 상세 구현 사항

### 1. Sclera-Aware Alpha (수정판)

랜드마크 오차로 렌즈가 흰자위를 침범할 때 자연스럽게 페이드:

```glsl
// Sclera-Aware Alpha (브레인스토밍 Section 13 수정판)
float calcScleraFactor(vec3 cameraColor) {
    float brightness = dot(cameraColor, vec3(0.299, 0.587, 0.114));
    float maxC = max(cameraColor.r, max(cameraColor.g, cameraColor.b));
    float minC = min(cameraColor.r, min(cameraColor.g, cameraColor.b));
    float saturation = (maxC - minC) / max(maxC, 1e-4); // epsilon 보호

    // 흰자위: 밝고(>0.6) 채도 낮음(<0.15) → 렌즈 alpha 감쇠
    float brightFactor = smoothstep(0.5, 0.7, brightness);
    float lowSatFactor = 1.0 - smoothstep(0.05, 0.2, saturation);
    return brightFactor * lowSatFactor;
}

// 적용: finalAlpha *= (1.0 - scleraFactor * 0.5);
```

**수정 이력**:
- `smoothstep(0.2, 0.05, sat)` → `1.0 - smoothstep(0.05, 0.2, sat)` — GLSL undefined behavior 제거 (Section 12 Codex 지적)
- 분모에 `max(maxC, 1e-4)` 보호 — NaN 방지 (Section 12 Codex 지적)
- 감쇠 강도 `0.8` → `0.5` — 보수적 설정 (Section 10, 13 합의)

**주의사항**:
- Sclera 보호는 "안전망"이지 "주방어선"이 아님 — 1차 방어는 기하학적 마스킹
- 저조도/노란 조명에서 오탐 가능 → 보수적 감쇠 강도 0.5로 부작용 최소화
- 밝은 홍채(파란/녹색)에서 오탐 가능 → threshold 튜닝 필요

### 2. Contact Shadow (Safe Shadow Logic)

상안검 경계 아래 부드러운 어둡기 그라데이션:

```glsl
// Contact Shadow — 클리핑의 "잘림"을 "가려짐"으로 전환
float calcContactShadow(float vTexCoordY, float minY, float eyelidFeather, float eyeOpening) {
    float shadowDepthPx = 4.0;
    float shadowDepth = shadowDepthPx / uDetH;
    float shadowIntensity = clamp(uShadowIntensity, 0.0, 0.25);

    // shadow는 mask 전이 끝점 이후에서 시작 (이중 감쇠 방지)
    float shadowZone = smoothstep(
        minY + eyelidFeather,
        minY + eyelidFeather + shadowDepth,
        vTexCoordY
    );
    float shadowFactor = (1.0 - shadowZone) * shadowIntensity;

    // 눈이 닫히면 shadow 자동 비활성화
    float shadowEnable = smoothstep(0.015, 0.025, eyeOpening);
    shadowFactor *= shadowEnable;

    // mask alpha와 곱하여 전이 구간에서 중복 방지
    float maskAlpha = smoothstep(minY, minY + eyelidFeather, vTexCoordY);
    return shadowFactor * maskAlpha;
}

// 적용: result.rgb *= (1.0 - contactShadow);
```

**핵심 설계 결정** (브레인스토밍 Section 10, 14 합의):
- **Shadow 시작점 = mask 전이 끝점 이후** — 두 효과의 영역 분리로 이중 감쇠(black crush) 방지
- `shadowIntensity` 범위: `clamp(0.0, 0.25)` — 과도한 어두움 방지
- `shadowEnable`: 눈 열림 높이 < 0.015 시 shadow OFF — "눈 감을 때 그림자가 눈을 덮는" 방지

**안전 가드** (브레인스토밍 Section 10 합의):
```
featherPx + shadowDepthPx < eyeOpeningPx × 0.3
(눈 열림 높이의 30% 이내로 feather + shadow 합산 제한)
```

### 3. Feature Flag 매트릭스

| 기능 | Flag Key | HIGH | MID | LOW | 기본값 |
|------|----------|:---:|:---:|:---:|:---:|
| Sclera Protection | `sclera_protect` | ✓ | ✓ | ✓ | ON |
| Contact Shadow | `contact_shadow` | ✓ | ✓ | △ | OFF |
| Specular Exclusion | `spec_exclusion` | ✓ | △ | ✗ | OFF |
| Fake Specular | `fake_spec` | ✓ | ✗ | ✗ | OFF |

△ = 성능 측정 후 결정, ✗ = 비활성화

### 4. 처리 순서 (파이프라인)

```
1. camera.rgb에서 lum 추출
2. Blend mode에 따라 렌즈 합성 (tint/softlight/etc.)
3. Sclera Protection: 흰자위 영역 alpha 감쇠
4. Contact Shadow: 상안검 아래 rgb 감쇠
5. Eyelid mask: 기존 smoothstep alpha 클리핑
6. 최종 출력
```

Sclera → Shadow → Mask 순서를 고정하여 감쇠 순서 의존성 제거.

## 검증 체크리스트

- [ ] Sclera Protection: 눈을 좌우로 돌렸을 때 흰자위에 렌즈 색이 번지지 않음
- [ ] Sclera Protection: 밝은 홍채(파란/녹색)에서 오탐률 < 10%
- [ ] Contact Shadow: 상안검 아래 자연스러운 그림자 확인
- [ ] Contact Shadow: 눈 감을 때 shadow 자동 비활성화 확인
- [ ] 이중 감쇠 방지: mask 전이 구간에서 과도한 어두움 없음
- [ ] `adb logcat`에서 NaN/Inf 관련 GL 에러 0건

## 다음 단계

1. 주관 평가: "자연스러움" 점수 Gate 2 판정
2. P4-W2-02: 비대칭 타원 Eye Mask (선택적)

## 변경 이력

| 날짜 | 내용 |
|------|------|
| 2026-02-13 | 작업 계획 문서 작성 |
