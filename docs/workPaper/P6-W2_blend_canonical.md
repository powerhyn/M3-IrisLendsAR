# P6-W2: 블렌드 3종 확정 + realSpec 폐기 구조

> **상태**: 인사이트 작성 완료. 세부 계획 본문 작성 대기.
> **작성**: 2026-04-23
> **선행 의존**: P6-W1 (EyeRenderPacket, avg_iris_luma 실측)
> **후속 의존**: P6-W3 (환경 반사 계층)

---

## 1. 인사이트 (세션 간 맥락 보존) ⭐

### 1.1 이미 S1에서 부분 완료된 것

`9aee86d` 커밋에서:
- `blendOverlay` (D6), `blendLuminanceTint(non-linear)` (D6), `blendSoftLight` (D6) **함수 및 분기 삭제 완료**
- `blendLuminanceTintLinear` 내부의 `realSpec` 2줄 (D3) **삭제 완료**
- 분기 재번호화 없이 ID 유지 (Normal=0, Multiply=1, Screen=2, LuminanceTintLinear=5, ColorReplace=7)

**현재 상태**:
```glsl
if (uBlendMode == 0) blended = blendNormal(...);
else if (uBlendMode == 1) blended = blendMultiply(...);
else if (uBlendMode == 2) blended = blendScreen(...);
else if (uBlendMode == 5) blended = blendLuminanceTintLinear(...);
else if (uBlendMode == 7) blended = blendColorReplace(...);
else blended = blendNormal(...);  // fallback
```

### 1.2 W2에서 할 남은 작업

1. **`blendLuminanceTintLinear` → `blendTintLinearV2` 리네이밍**: 의미 명확화. realSpec 제거된 상태가 "V2".
2. **`blendScreen` (sRGB 공간) → `blendScreenLinear` (선형 공간) 교체**: 99 §1.2 C3. 수식: `out = sqrt(mix(baseL, 1-(1-baseL)*(1-lensL), a))`. 기존 sRGB Screen 제거.
3. **`blendColorReplace` → `blendColorReplaceLinear` 교체** 후보: 99 §1.2 C4. **단 B1 벤치 전까지 실제 교체는 보류** (Normal vs CRL 대결이 먼저).
4. **realSpec 완전 폐기 확정**: 현재 "조건부 폐기" 상태. P6-W4 B2 벤치 성공 시 확정. 실패 시 재평가.
   - S1에서 이미 코드는 삭제됨 → W2 범위 밖일 수도. W2에서는 **문서상 확정**만.
5. **Normal 유지 확정**: B1 벤치 전까지 유지. 99 §1.1 D6 "Normal 제거 철회" 반영.

### 1.3 Codex R3 원문 — W2에서 주의할 점

**Codex R3 §1**:
> "§1 C1은 '블렌드 모드 4종 세트'를 즉시 확정처럼 썼다. 내 R2는 `ColorReplaceLinear`를 `Normal`과 1:1 대결 후 결정하자는 입장이었다. 따라서 C1은 '3종 확정 + 4번째 슬롯 벤치'로 써야 한다."

**Codex R3 §4 C1 수정 필요**:
> "즉시 확정은 `TintLinearV2 / Multiply / ScreenLinear` 3종이다. `ColorReplaceLinear`는 후보 슬롯이다."

즉 W2에서 **ColorReplaceLinear 수식 구현은 해도 되지만, 기본 활성 등록은 금지**. B1(P6-W5) 벤치 결과를 기다려야 함.

### 1.4 Gemini R2→R3 입장 변화 — 설득된 근거

Gemini R1: 하이브리드 블렌드 제안 (Circle, Vivid — Normal+Overlay 혼합, SoftLight+Multiply 혼합).
Gemini R2: Codex 4종 세트 수용 (하이브리드 철회).
Gemini R3: "TintLinearV2 + Multiply + ScreenLinear 3종 압축에 동의" + CRL은 B1 결과 기다림.

**Gemini의 하이브리드 철회 사유 (자기 R2)**: "튜닝 파라미터를 블랙박스화할 위험". 즉 하이브리드는 "특정 상황 튜닝 프리셋"일 뿐이지 별도 블렌드로 유지할 이유 없음. W2 브레인스토밍에서 Gemini에게 이 철회 확정 재확인.

### 1.5 수식별 정확한 GLSL (W2 구현 시 그대로 사용)

```glsl
// Claude 종합 99 §1.2에서 합의된 수식

// C2. TintLinearV2 (기본값, realSpec 제거된 LuminanceTintLinear)
vec3 blendTintLinearV2(vec3 base, vec3 blend, float opacity) {
    vec3 baseL = toLinearFast(base);    // base * base
    float lum = dot(baseL, vec3(0.2126, 0.7152, 0.0722));
    float avgLumLinear = uAvgIrisLum * uAvgIrisLum;
    float scale = clamp(0.5 / max(0.01, avgLumLinear), 0.8, 5.0);
    vec3 tinted = toLinearFast(blend) * lum * scale;
    vec3 result = mix(baseL, tinted, opacity);
    return toSRGBFast(result);          // sqrt(max(result, 0))
}

// C3. ScreenLinear (신규 — Screen의 선형 공간 버전)
vec3 blendScreenLinear(vec3 base, vec3 blend, float opacity) {
    vec3 baseL = toLinearFast(base);
    vec3 lensL = toLinearFast(blend);
    vec3 screened = vec3(1.0) - (vec3(1.0) - baseL) * (vec3(1.0) - lensL);
    return toSRGBFast(mix(baseL, screened, opacity));
}

// C4. ColorReplaceLinear (벤치 대기 — B1 결과 후 채택/기각)
vec3 blendColorReplaceLinear(vec3 base, vec3 blend, float opacity, float maxDetail) {
    vec3 baseL = toLinearFast(base);
    vec3 lensL = toLinearFast(blend);
    float lum = dot(baseL, vec3(0.2126, 0.7152, 0.0722));
    float avgLumLinear = uAvgIrisLum * uAvgIrisLum;
    float detail = clamp(pow(lum / max(0.01, avgLumLinear), 0.7), 0.75, maxDetail);
    vec3 colored = lensL * detail;
    return toSRGBFast(mix(baseL, colored, opacity));
}
```

### 1.6 기존 Screen/ColorReplace 처리 (sRGB 공간 버전)

현재 `blendScreen` (sRGB), `blendColorReplace` (sRGB)는 남아있는 상태.

**W2에서 결정할 것**:
- **옵션 A**: 즉시 Linear 버전으로 교체 (sRGB 버전 함수 삭제). 간단.
- **옵션 B**: 두 버전 병존. ID 2는 sRGB Screen 유지, ID 2 재활용으로 ScreenLinear 할당 시 충돌. ID 재매핑 필요.

**Claude 의견**: 옵션 A 추천. 기존 sRGB 버전 제거해도 외부 호환성 문제 없음 (Android demo에서 ID 2 = Screen이라는 계약은 그대로, 구현만 선형화).

**ColorReplace도 동일**: ID 7 유지, 구현만 ColorReplaceLinear로 교체. 단 B1 결과가 "Normal 유지" 쪽으로 나오면 ID 7 CRL 자체를 폐기하고 Normal만 유지.

### 1.7 블렌드 모드 전체 플로우 (W2 완료 후)

```
ID 0: Normal (B1 벤치 대기)
ID 1: Multiply (유지)
ID 2: ScreenLinear (W2에서 선형화 교체, C3)
ID 3: [빈] — 이전 Overlay 자리, fallback
ID 4: [빈] — 이전 LumTint(non-linear) 자리, fallback
ID 5: TintLinearV2 (W2에서 리네이밍 + realSpec 제거 확정, C1/C2)
ID 6: [빈] — 이전 SoftLight 자리, fallback
ID 7: ColorReplaceLinear (B1 결과 대기. 유지 or 폐기)
```

**외부(Android demo, Java API)에서 ID 3/4/6 전송 시**: shader의 default 분기(`else`)가 blendNormal로 fallback. 깨지지 않음.

### 1.8 maxDetail uniform 처리

현재 `uMaxDetail` uniform이 `blendColorReplace` 함수에서만 쓰임. `blendColorReplaceLinear`도 동일하게 사용 예정.

```glsl
float maxDetail = mix(uMaxDetail, 1.0, smoothstep(0.75, 1.0, irisEdgeDist));
```

이 계산은 이미 `applyLens` 안에서 CRL 호출 직전에 있음. 유지.

### 1.9 성능 예산

- 기존 대비 변동 거의 없음. `toLinearFast` (mul) + `toSRGBFast` (sqrt)는 이미 LTL에서 쓰이고 있어 ScreenLinear도 같은 수준.
- `blendColorReplaceLinear`의 `pow(x, 0.7)`은 추가 비용 있지만, iris 영역만 분기라 무시 가능.

### 1.10 W2 브레인스토밍 시 Codex/Gemini에게 던질 질문

1. **sRGB Screen/ColorReplace 즉시 제거 vs 병존**: 옵션 A(즉시 교체) vs B(병존)? 외부 호환성 리스크는?
2. **블렌드 ID 재매핑 vs 유지**: ID 3/4/6 빈 슬롯으로 두고 fallback? 아니면 재매핑해서 0~4로 압축?
3. **ColorReplaceLinear 구현 시점**: W2에서 수식만 넣어두고 비활성 분기 vs B1 결과 후 구현?
4. **fallback 중립 블렌드**: else 분기가 `blendNormal` 맞나? 아니면 `blendTintLinearV2`가 더 나은가? (Normal이 사실 B1 대기지만 현재는 유지)
5. **LUMA 계수 정합성**: TintLinearV2의 `vec3(0.2126, 0.7152, 0.0722)` (Rec.709 linear)와 다른 셰이더(beauty 등)의 계수 통일돼 있나?
6. **realSpec 복원 경로**: B2가 실패하면 realSpec를 되살려야 하는데, 어떤 형태로? Codex R3 "반사 계층 분리" 입장 재확인.

### 1.11 W2에서 수정할 파일

- 수정:
  - `cpp/src/gpu/shader_sources.cpp` — 블렌드 함수 재작성, 분기 갱신, 주석 업데이트
  - `cpp/include/iris_sdk/gpu/shader_manager.h` — (수정 필요 없을 가능성. 확인만)
  - `99_final_decision.md` §1.1 D3 상태 업데이트 (조건부 → W4 B2 결과 종속)

### 1.12 B1 벤치와의 경계 명확화

**P6-W2 범위**: 블렌드 **수식 구조** 정리. Normal 유지 or 제거 결정은 W2 범위 밖.

**P6-W5 범위 (B1 벤치)**: Normal vs ColorReplaceLinear 실기기 대결. 결과 반영은 W5에서.

즉 W2 완료 후 shader_sources.cpp는:
- TintLinearV2 / Multiply / ScreenLinear / ColorReplaceLinear 4종 함수 **모두 존재**
- 분기는 Normal(0) / Multiply(1) / ScreenLinear(2) / TintLinearV2(5) / ColorReplaceLinear(7) **5종 등록**
- B1 결과 후 W5에서 Normal 또는 CRL 제거

### 1.13 Android demo UI 정리는 별도

Android demo의 블렌드 선택 UI는 현재 8종 라벨일 가능성. W2 완료 후 UI가 4~5종만 보이게 정리하는 작업은 **별도 cleanup W로 분리** or W9 통합 테스트 단계에서 처리. W2 범위 밖.

---

## 2. 배경/맥락
_TODO_

## 3. 전제 조건
_TODO: P6-W1 완료 (EyeRenderPacket + avg_iris_luma 측정 경로)_

## 4. 목표
_TODO_

## 5. 99에서 확정된 사항
_TODO_

## 6. 미결 사항
_TODO: §1.10 질문 정리_

## 7. W 브레인스토밍 시작 체크리스트
_TODO_

## 8. 완료 정의 + 다음 W 트리거
_TODO_

---

## 참조

- 99_final_decision.md §1.1 D3/D6, §1.2 C1/C2/C3/C4
- 13_codex_r3.md §1, §4 (C1 수정 지적)
- 07_claude_r2.md I2 (블렌드 결정 히스토리)
- 08_codex_r2.md I2 (4종 vs 3종 신중화)
- 09_gemini_r2.md (하이브리드 철회)
- 14_gemini_r3.md (3종 확정 동의)
- S1 커밋 `9aee86d` — Overlay/LumTint/SoftLight 제거 완료
