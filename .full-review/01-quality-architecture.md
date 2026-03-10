# Phase 1: Code Quality & Architecture Review

## 리뷰 대상: P4-W4-01d 톤커브 미드톤 리프트 (2 commits, +54/-8)

---

## Code Quality Findings

### [Medium] CQ-1: 테스트 주석이 수정 전(s 기반) 로직을 설명

- **파일**: `test_beauty_config_v2.cpp:419, 430`
- **설명**: `ToneLiftFixedAboveThreshold` 테스트의 주석이 `s ≈ 0.5 (smoothstep) > 0.1`라고 설명하지만, 실제 코드는 085732f 커밋에서 `t > 0.1`(raw skinQuality)로 수정됨. `ToneLiftGradualAtLowQuality`도 `s ≈ 0.0073 (smoothstep) ≤ 0.1 → tone_lift = s * 1.5`라고 설명하나 실제로는 `t * 1.5`를 사용.
- **영향**: 테스트는 통과하지만 주석이 구현과 불일치하여 향후 유지보수 시 혼란
- **수정**:
```cpp
// skinQuality 0.5 → t = 0.5 > 0.1 → tone_lift = 0.15 고정
// skinQuality 0.05 → t = 0.05 ≤ 0.1 → tone_lift = t * 1.5 = 0.075
```

### [Medium] CQ-2: `mapSkinQuality()` 내 tone_lift 매핑의 임계값 불연속 검증 부재

- **파일**: `gpu_beauty_backend.cpp:1026`
- **설명**: `(t > 0.1f) ? 0.15f : t * 1.5f` — `t = 0.1`일 때 `0.1 * 1.5 = 0.15`, `t = 0.100001`일 때 `0.15`. 수학적으로 연속이지만, **정확히 `t == 0.1f`일 때는 `else` 분기**를 타서 `0.15f`가 됨 → 우연히 연속. 그러나 이 연속성이 의도적인지 우발적인지 코드에서 불명확.
- **영향**: 현재는 문제 없으나, 향후 0.15f나 1.5f 계수를 변경하면 불연속 점프 발생 가능
- **수정**: 연속성 의도를 주석으로 명시하거나, `std::min(0.15f, t * 1.5f)` 패턴으로 자연스러운 연속성 보장
```cpp
// At t=0.1: t*1.5 = 0.15 = fixed value (continuous by design)
p.tone_lift = (t > 0.1f) ? 0.15f : t * 1.5f;
```

### [Low] CQ-3: `ToneLiftRange` 테스트의 상한 검증이 느슨

- **파일**: `test_beauty_config_v2.cpp:446`
- **설명**: `EXPECT_LE(params.tone_lift, 0.30f)` — 실제 `tone_lift` 최대값은 0.15f (고정). 상한 0.30f는 2배 넓은 범위로, `tone_lift`가 0.25f로 잘못 계산되어도 테스트 통과.
- **수정**: `EXPECT_LE(params.tone_lift, 0.16f)` 또는 `EXPECT_NEAR(params.tone_lift, 0.15f, 0.01f)`로 정밀화

### [Low] CQ-4: FreqSepParams 기본값 `tone_lift = 0.15f`와 `enabled = false`의 관계

- **파일**: `gpu_beauty_backend.h:241-242`
- **설명**: `tone_lift` 기본값이 0.15f이지만 `enabled = false`가 기본. `mapSkinQuality(0.0f, ...)`는 `enabled = false`를 반환하므로 `tone_lift`는 무시됨. 그러나 누군가 `FreqSepParams` 구조체를 직접 구성하면(`mapSkinQuality` 경유 않고) `enabled = false`인데 `tone_lift = 0.15f`인 혼란스러운 상태가 됨.
- **영향**: `enabled` 체크가 반드시 선행하므로 실질적 문제 없음. 기존 필드들도 동일 패턴(기본값 + enabled=false).

---

## Architecture Findings

### [Low] AR-1: 기존 아키텍처 패턴 완벽 준수

- **설명**: 변경사항이 기존 `edge_weight`/`chroma_weight` 추가 패턴과 1:1 동일한 구조를 따름:
  1. 셰이더 uniform 선언 → 2. Uniform 캐시 구조체 멤버 → 3. `glGetUniformLocation` → 4. `glUniform1f` → 5. `FreqSepParams` 필드 → 6. `mapSkinQuality` 매핑
- **판정**: 아키텍처 일관성 우수. 새 패스 추가 없이 기존 composite 셰이더에 ALU 3 ops만 추가.

### [Low] AR-2: 셰이더 수식 적용 위치 적절

- **설명**: `beauty` 변수에만 적용 (mask 블렌딩 전), `orig`에는 미적용. mask `mix(orig, beauty, mask)` 이후가 아닌 이전에 적용하여 보정 영역만 톤 리프트.
- **판정**: 설계 의도에 부합하며, 비보정 영역(마스크 외부)에 영향 없음 확인.

---

## Critical Issues for Phase 2 Context

- **성능**: ALU 3 ops (곱셈 2 + 덧셈 1) 추가, 텍스처 페치 없음 → 성능 검증 필요하나 매우 경미
- **수치 안정성**: `beauty` 값이 [0,1] 범위 보장인지 → `uToneLift ∈ [0, 0.3]`이고 `beauty ∈ [0,1]`이면 출력도 [0,1] (수학적 증명 문서에 포함)
- **uniform 클램핑**: `uToneLift`에 대한 입력 범위 검증이 셰이더/C++ 양쪽에서 없음
