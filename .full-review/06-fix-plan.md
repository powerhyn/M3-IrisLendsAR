# P4-W4-01c 리뷰 수정 계획 — ✅ 전건 완료

## 수정 대상

최종 리포트(`05-final-report.md`) 기준 병합 전 필수 4건. **모두 적용 완료, 17/17 테스트 통과.**
모든 수정은 `shader_sources.cpp`의 FREQ_SEP_COMPOSITE_FRAGMENT 셰이더 내부에서 완결됨.

---

## Fix 1: P1-1 — sRGB/Linear 색공간 통일 (Option B: gamma 2.0 근사)

**파일**: `cpp/src/gpu/shader_sources.cpp` line 503-506
**분류**: 출력 품질 회귀 리스크

**현재**: 4-neighbor 텍스처 샘플을 sRGB 공간에서 직접 사용
```glsl
float lumR = dot(texture(uOriginal, vTexCoord + vec2( texelSize.x, 0.0)).rgb, LUMA_601);
float lumL = dot(texture(uOriginal, vTexCoord + vec2(-texelSize.x, 0.0)).rgb, LUMA_601);
float lumU = dot(texture(uOriginal, vTexCoord + vec2(0.0,  texelSize.y)).rgb, LUMA_601);
float lumD = dot(texture(uOriginal, vTexCoord + vec2(0.0, -texelSize.y)).rgb, LUMA_601);
```

**변경**: gamma 2.0 근사로 linearize 후 Rec.709 계수 적용
```glsl
vec3 sR = texture(uOriginal, vTexCoord + vec2( texelSize.x, 0.0)).rgb;
vec3 sL = texture(uOriginal, vTexCoord + vec2(-texelSize.x, 0.0)).rgb;
vec3 sU = texture(uOriginal, vTexCoord + vec2(0.0,  texelSize.y)).rgb;
vec3 sD = texture(uOriginal, vTexCoord + vec2(0.0, -texelSize.y)).rgb;
// gamma 2.0 근사 linearize (pow(x,2.2) 대비 오차 ~5%, 4×pow 절약)
float lumR = dot(sR * sR, LUMA_709);
float lumL = dot(sL * sL, LUMA_709);
float lumU = dot(sU * sU, LUMA_709);
float lumD = dot(sD * sD, LUMA_709);
```

**부수 변경**:
- `LUMA_601` 상수 삭제 (더 이상 사용처 없음)
- 주석 업데이트: "Option B (sRGB)" → "gamma 2.0 근사 linearize"

---

## Fix 2: P2-1 — `diff` 중복 변수 제거

**파일**: `cpp/src/gpu/shader_sources.cpp` line 512-515
**분류**: 코드 정리

**현재**:
```glsl
vec3 diff = orig - low;
float lumDiff = dot(diff, LUMA_709);
vec3 chromaDiff = diff - vec3(lumDiff);
```

**변경**: `high`를 직접 재사용
```glsl
float lumHigh = dot(high, LUMA_709);
vec3 chromaDiff = high - vec3(lumHigh);
```

---

## Fix 3: P2-2 — LUMA_709 상수 통일

**파일**: `cpp/src/gpu/shader_sources.cpp` line 534
**분류**: 코드 정리 (DRY)

**현재**:
```glsl
float baseLum = dot(smoothLow, vec3(0.2126, 0.7152, 0.0722));
```

**변경**:
```glsl
float baseLum = dot(smoothLow, LUMA_709);
```

---

## Fix 4: P2-3 — 매직 넘버 상수 명명

**파일**: `cpp/src/gpu/shader_sources.cpp` line 521-522
**분류**: 코드 정리

**현재**:
```glsl
    * (1.0 - uEdgeWeight * clamp(edgeStrength * 5.0, 0.0, 1.0))
    * (1.0 + uChromaWeight * clamp(chromaDev * 10.0, 0.0, 1.0));
```

**변경**: 셰이더 상단에 상수 선언 + 결합식에서 참조
```glsl
// 상단 상수 추가
const float EDGE_SCALE = 5.0;    // maps typical edge range [0, ~0.2] to [0, 1]
const float CHROMA_SCALE = 10.0; // maps typical chroma dev range [0, ~0.1] to [0, 1]

// 결합식
    * (1.0 - uEdgeWeight * clamp(edgeStrength * EDGE_SCALE, 0.0, 1.0))
    * (1.0 + uChromaWeight * clamp(chromaDev * CHROMA_SCALE, 0.0, 1.0));
```

---

## 수정 순서

1. Fix 4 (상수 선언) → Fix 1 (색공간 통일 + LUMA_601 삭제) → Fix 2 (diff 제거) → Fix 3 (인라인 리터럴)
   - Fix 1이 LUMA_601을 삭제하므로 상수 선언(Fix 4)을 먼저 처리
   - Fix 2, 3은 독립적

## 설계 문서 업데이트

- `P4-W4-01c_edge_aware_attenuation.md`: KI-1 해결 기록, Option B → gamma 2.0 근사로 변경 반영
