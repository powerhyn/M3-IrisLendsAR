# P6-W2: 블렌드 3종 확정 + realSpec 폐기 구조

> **상태**: 구현 완료 (2026-04-29). 실기기 시각 회귀 검증은 PR 단계에서 수행.
> **작성**: 2026-04-23
> **선행 의존**: P6-W1 (EyeRenderPacket, avg_iris_luma 실측)
> **후속 의존**: P6-W3 (환경 반사 계층)
> **구현 브랜치**: `feature/P6-W2`

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

### 2.1 W2가 처리하는 것

**블렌드 수식 정리**. 99의 확정 사항 중 블렌드 관련 모든 항목:
- **C1** 블렌드 3종 확정 (TintLinearV2/Multiply/ScreenLinear)
- **C2** TintLinearV2 수식 (realSpec 제거됨)
- **C3** ScreenLinear 수식 신규
- **C4** ColorReplaceLinear 수식 신규 **후보** (B1 벤치 결과 대기 — W5에서 확정)
- **D3** realSpec 폐기 **조건부** (B2 벤치 결과 대기 — W4에서 확정)

### 2.2 S1에서 이미 완료된 부분 (9aee86d)

S1 롤백 커밋에서 다음이 이미 적용됨:
- blendOverlay, blendLuminanceTint(non-linear), blendSoftLight **함수 삭제**
- uBlendMode 3/4/6 **분기 삭제** (default blendNormal fallback)
- blendLuminanceTintLinear 내부 **realSpec 2줄 삭제**

**즉 W2 시작 시점에 이미 "블렌드 5종 체제" 상태**: Normal(0), Multiply(1), Screen(2, sRGB), LuminanceTintLinear(5, realSpec 없는 형태), ColorReplace(7, sRGB).

### 2.3 W2가 해야 할 남은 작업

| 작업 | 구체 |
|------|------|
| 리네이밍 | `blendLuminanceTintLinear` → `blendTintLinearV2` (의미 명확화) |
| 선형화 교체 | `blendScreen` (sRGB) → `blendScreenLinear` |
| 선형화 교체 (조건부) | `blendColorReplace` (sRGB) → `blendColorReplaceLinear` (B1 전까진 병존 or 후보 등록) |
| 주석 정리 | S1에서 박아둔 인라인 주석을 W2 최종본으로 업데이트 |
| 문서 동기화 | 99 §1.2 C2/C3/C4 수식 정확도 확인 |

### 2.4 이 W가 해결하지 **않는** 것

- **Normal vs ColorReplaceLinear 최종 선택** — W5 B1 벤치
- **D3 realSpec 완전 폐기 확정** — W4 B2 결과 후 (W4에서 처리)
- **블렌드 UI 정리** (Android demo) — W9 또는 별도
- **블렌드 모드별 기본 파라미터** (opacity, maxDetail 기본값) — 각 W 튜닝 시

### 2.5 99 편향 교정 히스토리

이 W가 특히 주의할 Codex R3 지적:
> "§1 C1은 `블렌드 모드 4종 세트`를 즉시 확정처럼 썼다. 내 R2는 `ColorReplaceLinear`를 `Normal`과 1:1 대결 후 결정하자는 입장이었다. 따라서 C1은 `3종 확정 + 4번째 슬롯 벤치`로 써야 한다."

**W2 반영**: TintLinearV2/Multiply/ScreenLinear **3종만 확정**. ColorReplaceLinear는 수식은 준비하되 ID 7 등록 여부는 W5 B1 결과까지 보류.

---

## 3. 전제 조건

**반드시 확인할 것**:

1. ✅ **W1 완료** — EyeRenderPacket + avg_iris_luma 측정 경로. LTL이 정상 tint 동작.
2. ✅ **S1 롤백 커밋 확인** — `9aee86d`. Overlay/LumTint/SoftLight 삭제 및 realSpec 삭제 완료.
3. ✅ **99 §1.2 C1/C2/C3/C4** 숙지
4. ✅ **현재 shader_sources.cpp의 블렌드 함수 5개** (Normal, Multiply, Screen, LuminanceTintLinear, ColorReplace) 실제 수식 읽기 — 각 함수의 행 번호 파악

**환경**:
- C++ 빌드 가능
- 실기기 최소 1대 (LTL → TintLinearV2 전환 후 시각 회귀 확인)

---

## 4. 목표

**W2 완료 시 달성 상태**:

1. **`blendLuminanceTintLinear` → `blendTintLinearV2` 리네이밍 완료**
2. **`blendScreenLinear` 신규 추가** (sRGB `blendScreen` 대체)
3. **`blendColorReplaceLinear` 수식 구현 완료** (활성 등록은 W5 B1 후)
4. **주석 및 문서 업데이트** (99 §1.2, 이 W 문서)
5. **빌드 + 실기기 시각 회귀 확인** — LTL이 TintLinearV2로 바뀌어도 렌더링 동일

### 4.1 Definition of Done

- [x] `shader_sources.cpp`의 블렌드 분기 정리 (ID 0/1/2/5/7) — fallback default = TintLinearV2
- [x] 함수명 `blendLuminanceTintLinear` → `blendTintLinearV2` 전면 반영 + uAvgIrisLum squaring 제거
- [x] `blendScreen`(sRGB) 제거, 내부 구현 `blendScreenLinear`로 교체 (ID 2는 유지) — 옵션 A 채택
- [x] `blendColorReplaceLinear` 수식 함수 존재 + ID 7 정식 분기 활성 (§5.7 옵션 B, B1 벤치 대상 — 채택 확정 아님)
- [x] LUMA_709_LENS 상수로 블렌드 LUMA 계수 Rec.709 linear 통일 (§5.10)
- [x] invalid blend ID(3/4/6/etc.) debug 빌드 1회 경고 + 셰이더 TintLinearV2 fallback (§5.9)
- [x] BlendMode enum 주석 갱신 + canonical default 명시 (sdk_api.h, types.h, §5.4/5.12)
- [x] realSpec_archive.md 보존 (§5.11 Claude 부가 제안)
- [x] C++ 빌드 통과 (`cmake --build . --target iris_sdk` no work to do)
- [ ] 실기기 1회 확인 — 기존 LTL SKU 렌더링 결과가 TintLinearV2로 시각적으로 동일 (PR 단계)
- [ ] 단위 테스트: 새 수식들의 edge case (선택, default 모드 생략)

### 4.2 Out of scope

- Normal 제거 — W5 B1 결과 전 금지 (Codex R3 경고: "결론 선반영")
- realSpec 완전 폐기 확정 — W4 B2 결과 후
- 블렌드 UI 정리 — W9

---

## 5. 99에서 확정된 사항

> 🔧 **구현 시 참조**: [`P6_implementation_handoff.md`](P6_implementation_handoff.md) §4.1 색공간/LUMA 규약 (**`uAvgIrisLum * uAvgIrisLum` squaring 금지**, linear 공간 값 직접 사용), §4.4 GLSL 패스 규약 (블렌드→디테일→반사 순서).

### 5.1 C2 TintLinearV2 수식 (원 LTL에서 realSpec 제거)

```glsl
vec3 blendTintLinearV2(vec3 base, vec3 blend, float opacity) {
    vec3 baseL = toLinearFast(base);                            // base * base (감마 2.0 근사)
    float lum = dot(baseL, vec3(0.2126, 0.7152, 0.0722));       // Rec.709 linear
    // uAvgIrisLum은 CPU에서 이미 linear 공간으로 계산된 값 (W1 §5.2.1).
    // squaring 금지 — 이중 변환 버그.
    float scale = clamp(0.5 / max(0.01, uAvgIrisLum), 0.8, 5.0);
    vec3 tinted = toLinearFast(blend) * lum * scale;
    vec3 result = mix(baseL, tinted, opacity);
    return toSRGBFast(result);                                  // sqrt(max(result, 0))
}
```

**주의**:
- S1에서 realSpec 2줄 이미 제거됨
- 함수명만 `blendLuminanceTintLinear` → `blendTintLinearV2`로 변경
- uniform `uAvgIrisLum`는 W1에서 복구됨 (0.35 하드코드 제거, self-measure 경로)
- **색공간 계약 (W1 §5.2.1 + W2 §5.10):** `uAvgIrisLum`는 linear 공간 값. Rec.709 계수 고정.

### 5.2 C3 ScreenLinear 수식 (신규)

```glsl
vec3 blendScreenLinear(vec3 base, vec3 blend, float opacity) {
    vec3 baseL = toLinearFast(base);
    vec3 lensL = toLinearFast(blend);
    vec3 screened = vec3(1.0) - (vec3(1.0) - baseL) * (vec3(1.0) - lensL);
    return toSRGBFast(mix(baseL, screened, opacity));
}
```

### 5.3 C4 ColorReplaceLinear 수식 (B1 전 대기 상태로 등록)

```glsl
vec3 blendColorReplaceLinear(vec3 base, vec3 blend, float opacity, float maxDetail) {
    vec3 baseL = toLinearFast(base);
    vec3 lensL = toLinearFast(blend);
    float lum = dot(baseL, vec3(0.2126, 0.7152, 0.0722));  // Rec.709 linear
    // uAvgIrisLum은 이미 linear 공간 값 (W1 §5.2.1). squaring 금지.
    float detail = clamp(pow(lum / max(0.01, uAvgIrisLum), 0.7), 0.75, maxDetail);
    vec3 colored = lensL * detail;
    return toSRGBFast(mix(baseL, colored, opacity));
}
```

### 5.4 블렌드 모드 ID 매핑 (W2 완료 후)

| ID | 함수명 | 용도 | 상태 |
|----|--------|------|------|
| 0 | blendNormal | 불투명 단순 오버레이 | 유지 (W5 B1 대기) |
| 1 | blendMultiply | 짙은 렌즈 | 유지 |
| 2 | blendScreenLinear (신규) | 밝은 톤 렌즈 | W2 신규 등록 |
| 3 | [빈] → fallback blendNormal | 이전 Overlay | S1에서 분기 제거 |
| 4 | [빈] → fallback blendNormal | 이전 LumTint(non-linear) | S1에서 분기 제거 |
| 5 | blendTintLinearV2 | **기본값** | W2 리네이밍 |
| 6 | [빈] → fallback blendNormal | 이전 SoftLight | S1에서 분기 제거 |
| 7 | blendColorReplaceLinear | W5 B1 결과 대기 | **함수만 정의, ID 7 분기 등록은 W5 결과 후** |

**⚠️ CRL 등록 상태 명확화** (Codex R4 리뷰 반영):
- W2 시점: `blendColorReplaceLinear` **함수 정의 존재**. uBlendMode==7 분기는 **미등록 또는 fallback 분기**.
- 벤치 토글용 별도 플래그로 임시 활성 가능 (W5 B1 비교 시).
- 정식 활성(분기 등록)은 W5 B1 결과가 "CRL 채택" 또는 "조건부" 일 때만.

### 5.5 내부 이전 함수 제거 판정

**Codex R3 §1 경고**: "Normal 제거는 B1 벤치 전에는 틀렸다". 즉 ID 0 유지 필수.

**W2 제거 대상**:
- `blendScreen` (sRGB) → `blendScreenLinear`로 대체 (ID 2 내부 교체)
- `blendColorReplace` (sRGB) → `blendColorReplaceLinear`로 대체 (ID 7 교체. 단 W5 결과에 따라 다시 제거 가능)

### 5.6 sRGB 함수 완전 제거 정책

**옵션 A (즉시 제거)**: `blendScreen` (sRGB) 함수 삭제. ID 2는 blendScreenLinear로 전환.
**옵션 B (병존)**: 두 버전 모두 유지. ID 2는 Linear, ID 8 같은 빈 ID에 sRGB 버전.

**Claude 추천**: 옵션 A. B1 결과가 sRGB로 돌아갈 가능성 0이고, 병존은 코드 냄새.

**W2 브레인스토밍에서 확정**.

### 5.7 CRL 구현 시점 — **옵션 B 확정** (W2 R1 다수 2/3)

- **W2에서 함수 정의 + ID 7 정식 분기 활성**.
- 주석 명시: `// W2 활성 — W5 B1 벤치 대상, 채택 확정 아님`.
- W5 B1 결과가 "CRL 제거" 판정 시 함수 + 분기 모두 제거하는 롤백 커밋을 W5 단계에서 수행.
- 근거: W5 B1 Normal vs CRL 1:1 비교 준비 비용 절감. Codex 절차적 우려("결론 선반영")는 주석 + 롤백 계획으로 완화.
- 출처: `P6-W2_brainstorm/synthesis.md` §2.

### 5.8 ID 3/4/6 빈 슬롯 — **유지 확정** (W2 R1 합의 3/3)

- 재번호화 금지. ID 3/4/6은 **fallback 분기로 흡수** (§5.9 참조).
- 근거: 외부 호출자(Android demo/클라이언트) 호환성 유지, 수정 범위 최소화.
- 출처: `P6-W2_brainstorm/synthesis.md` §1.

### 5.9 Fallback default — **TintLinearV2 확정** (W2 R1 다수 2/3)

- 무효/빈 ID 전송 시 **TintLinearV2(ID=5)로 fallback**.
- **Debug 빌드 한정** 로그 1회: `[IrisSDK] Unknown blend ID=<n>, falling back to TintLinearV2`.
- 근거: 기본 모드 = TintLinearV2라는 canonical 일관성. Codex "디버그 명료성" 지적은 로그로 해소.
- 출처: `P6-W2_brainstorm/synthesis.md` §2.

### 5.10 LUMA_COEFFS — **Rec.709 linear 통일 확정** (W2 R1 합의 3/3)

- 계수: `vec3(0.2126, 0.7152, 0.0722)` (Rec.709 linear).
- **Shader 전 경로 + CPU 측정(W1 avg_iris_luma) 동일 상수**. 공통 매크로/상수(`LUMA_709`)로 통일.
- Linear-space 연산 강제. sRGB 평균 금지.
- **테스트:** shader 계산 결과와 CPU 계산 결과 오차 ≤1% 검증 케이스 추가.
- 근거: 계수 불일치 시 W5 B1 비교에서 수식 차이와 계수 차이가 섞여 판정 불가.
- 출처: `P6-W2_brainstorm/synthesis.md` §1.

### 5.11 realSpec 복원 경로 — **완전 삭제 확정** (W2 R1 합의 3/3)

- realSpec 관련 주석 + 코드 **완전 제거**. 주석 코드 냄새 제거.
- 복원 경로: Git history.
- **선택적 보존:** `docs/workPaper/P6-W2_brainstorm/realSpec_archive.md`에 수식 블록 + 커밋 SHA 기록(Claude 부가 제안). 강제 조건 아님, 구현 단계 판단.
- 근거: W4 B2 실패 확률 낮고, Git history로 충분한 복원.
- 출처: `P6-W2_brainstorm/synthesis.md` §1.

### 5.12 기본값 블렌드 모드 — **TintLinearV2 (ID=5) 확정** (W2 R1 합의 3/3)

- SDK canonical default = TintLinearV2 (99 §1.1 D6 유지).
- **Android demo 초기 blendMode = ID 5** 설정.
- **C API 문서화:** `sdk_api.h`에 기본값 명시 주석 추가 (ex: `// Default blend mode: TintLinearV2 (ID=5)`).
- 출처: `P6-W2_brainstorm/synthesis.md` §1.

### 5.13 Android Demo UI — **W9 이관 확정** (W2 R1 합의 3/3)

- W2 범위에서 demo UI drop-down 정리 **금지**.
- W9 integration 단계에서 일괄 정리.
- 근거: W2 = shader/C++ 범위. UI는 별도 리뷰 범위로 분리하여 병합 위험 감소.
- 출처: `P6-W2_brainstorm/synthesis.md` §1.

---

## 6. 미결 사항 (W2 브레인스토밍 R1 결과)

### 6.0 R1 결과 요약 (2026-04-24)

| 번호 | 원 쟁점 | 상태 | 반영 위치 |
|------|---------|------|-----------|
| 6.1 | CRL 구현 시점 (W2 vs W5) | ✅ **닫힘** (2/3 다수 옵션 B) | §5.7 |
| 6.2 | ID 3/4/6 빈 슬롯 | ✅ **닫힘** (3/3 합의 유지) | §5.8 |
| 6.3 | fallback default | ✅ **닫힘** (2/3 다수 TintLinearV2) | §5.9 |
| 6.4 | LUMA_COEFFS 정합성 | ✅ **닫힘** (3/3 합의 Rec.709 linear 통일) | §5.10 |
| 6.5 | realSpec 주석 vs 삭제 | ✅ **닫힘** (3/3 합의 완전 삭제) | §5.11 |
| 6.6 | 기본값 블렌드 모드 | ✅ **닫힘** (3/3 합의 TintLinearV2 ID=5) | §5.12 |
| 6.7 | Android Demo UI | ✅ **닫힘** (3/3 합의 W9 이관) | §5.13 |

참여 모델: Codex (gpt-5.4 xhigh), Gemini (gemini-3-flash), Claude (opus-4-7).
원문: `docs/workPaper/P6-W2_brainstorm/{codex,gemini,claude}_w2.md`.
종합: `docs/workPaper/P6-W2_brainstorm/synthesis.md`.

**미결 없음.** 7개 쟁점 모두 R1에서 결론. 다수결 2개(6.1, 6.3)는 Codex 지적을 완화 조건(주석/로그)으로 수용.

---

## 7. W2 브레인스토밍 시작 체크리스트

### 7.1 새 세션 시작 시 읽을 파일

**필수**:
1. `docs/workPaper/P6-W0_index.md` §1 (5분)
2. `docs/workPaper/P6-W2_blend_canonical.md` 전체 (10분)
3. `docs/workPaper/P5-W3-05_brainstorm/99_final_decision.md` §1.1 D3/D6 + §1.2 C1/C2/C3/C4 (5분)
4. `cpp/src/gpu/shader_sources.cpp` 블렌드 함수 부분 직접 확인 (5분)

**선택**:
- `08_codex_r2.md` §I2 — "3종 확정 + 4번째 슬롯 벤치" 원문
- `13_codex_r3.md` §1, §4 — Normal 제거 경고

### 7.2 Codex/Gemini 송신 프롬프트 초안

```
@docs/workPaper/P6-W2_blend_canonical.md 읽고, 섹션 6 미결 7개에 대해
각자 입장 정리 후 docs/workPaper/P6-W2_brainstorm/{codex|gemini}_w2.md
로 작성해줘.

중점:
- 6.1 CRL 구현 시점 (W2 vs W5 대기)
- 6.3 fallback default (Normal vs TintLinearV2)
- 6.5 realSpec 주석 유지 vs 완전 삭제

규칙:
- 새 쟁점 제기 금지
- 각 항목 "추천 + 근거 1~2줄"
- W5 B1 벤치 설계와의 연관성 고려
```

### 7.3 예상 합의 지점

- 6.2 ID 유지 vs 재번호화 → **유지**로 빠르게 합의 예상
- 6.4 LUMA 계수 정합성 → 확인만, 이슈 없을 것
- 6.6 기본값 TintLinearV2 → 쉽게 합의
- 6.7 Android demo UI → "W9에서" 합의

### 7.4 예상 대립 지점

- 6.1 CRL 구현 시점: Claude(B) vs Codex(A)? Codex가 "조기 구현 = 확정처럼 보임" 우려할 수도.
- 6.3 fallback default: Claude(TintLinearV2) vs Codex(Normal 유지)? Normal도 아직 살아있으니 Codex 입장 불명확.
- 6.5 realSpec 주석 vs 삭제: Claude(주석 유지) vs Codex(삭제)? 전형적 냄새 vs 안전 트레이드오프.

### 7.5 1시간 브레인스토밍 진행

```
0~5분    송신
5~15분   응답 대기
15~30분  7개 쟁점 표 정리
30~50분  대립 지점 3개 심층 논의
50~60분  합의 + 구현 순서 확정
```

### 7.6 구현 예상 소요

- 함수 리네이밍: 30분
- ScreenLinear 구현 + 분기 교체: 30분
- ColorReplaceLinear 구현 + 분기 등록 (조건부): 45분
- realSpec 주석 정리: 15분
- 실기기 시각 회귀 확인: 30분
- 단위 테스트 (선택): 1h

**총 2.5~4h**.

---

## 8. 완료 정의 + 다음 W 트리거

### 8.1 완료 정의

§4.1 체크리스트 전체 ✅.

### 8.2 커밋 전략 (실제 적용 — feature/P6-W2)

원안 4분할에서 5분할로 조정 (cpp 변경이 같은 파일/같은 영역에 응집되어 본체는 단일 커밋, 보조 변경을 논리 단위로 분리).

| # | 해시 | 메시지 | 영역 |
|---|------|-------|------|
| 1 | `b70490b` | feat(gpu-lens): P6-W2 블렌드 3종 선형 공간 정리 + LUMA_709 Rec.709 통일 | shader_sources.cpp (블렌드 함수 + 분기 + LUMA 상수) |
| 2 | `25ebf99` | feat(gpu-lens): P6-W2 invalid blend ID 1회 debug 경고 (§5.9) | gpu_lens_renderer.h/cpp |
| 3 | `a7fcdd7` | docs(api): P6-W2 BlendMode enum canonical default 주석 갱신 (§5.4/5.12) | types.h, sdk_api.h |
| 4 | `6af6a4f` | docs(P6-W2): realSpec 폐기 아카이브 보존 (§5.11) | realSpec_archive.md |
| 5 | (이 커밋) | docs(P6-W2): 구현 완료 마킹 + DoD 체크 + 99 §1.1/§1.2 갱신 | P6-W2 문서 + 99_final_decision.md |

### 8.3 다음 W 트리거

**P6-W3 (환경 반사 계층) 시작 조건**:
- W2 완료 (블렌드 3종 안정, realSpec 삭제 완료)
- LTL → TintLinearV2 시각 회귀 없음 확인

**P6-W5 (B1 + B8 벤치) 시작 조건**:
- W3 완료 (환경 반사 계층 스캐폴드, renderMask hook 존재)
- TintLinearV2 + ScreenLinear + ColorReplaceLinear 수식 모두 셰이더에 등록됨

### 8.4 W2 실패 시 롤백 전략

- W2 커밋 revert → S1 상태(`9aee86d`)로 복귀
- 재검토 후 재브레인스토밍

### 8.5 W2 성공 시 기대 효과

- **SDK 가독성 향상**: 블렌드 분기 5개 → 3 확정 + 1 후보로 명확
- **선형 공간 일관성**: Screen/ColorReplace가 TintLinearV2와 동일 공간에서 작동 → 수식 일관성 ↑
- **realSpec 냄새 완전 제거**: 셰이더 파일에서 "밝은 픽셀 보호" 해킹 흔적 소멸
- **B1 벤치 준비 완료**: W5에서 Normal vs CRL 바로 비교 가능

---

## 참조

- 99_final_decision.md §1.1 D3/D6, §1.2 C1/C2/C3/C4
- 13_codex_r3.md §1, §4 (C1 수정 지적)
- 07_claude_r2.md I2 (블렌드 결정 히스토리)
- 08_codex_r2.md I2 (4종 vs 3종 신중화)
- 09_gemini_r2.md (하이브리드 철회)
- 14_gemini_r3.md (3종 확정 동의)
- S1 커밋 `9aee86d` — Overlay/LumTint/SoftLight 제거 완료
