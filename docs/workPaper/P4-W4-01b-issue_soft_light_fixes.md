# P4-W4-01b-issue: Soft Light 리뷰 이슈 수정

> **상위 문서**: `P4-W4-01b_soft_light.md`
> **상태**: ✅ 완료
> **난이도**: 중간 | **리뷰 출처**: `.full-review/05-final-report.md`
> **선행 조건**: P4-W4-01b (Soft Light 합성 전환) 완료

---

## 수정 대상 이슈 목록

| ID | 심각도 | 요약 | 대상 파일 |
|----|--------|------|-----------|
| P2 | Critical | Soft Light gain 손실 50-90% — 톤 의존 보상 필요 | `shader_sources.cpp` |
| GLSL-4 | Medium | Soft Light 수식 MAD 최적화 (곱셈 4→3회) | `shader_sources.cpp` |
| CPP-1 | Medium | 매직 넘버 0.70f 상수화 | `gpu_beauty_backend.cpp` |
| D1 | Medium | 작업 문서 gain 손실 한계 미문서화 + 완료 상태 불일치 | `P4-W4-01b_soft_light.md` |
| F2 | Low | gain 특성 주석 누락 (셰이더 내) | `shader_sources.cpp` |
| T1 | Low | CPU 참조 테스트 부재 (Pegtop 수식 검증) | `test_beauty_config_v2.cpp` |

---

## 수정 계획

### Fix 1: P2 — Gain 보상 스케일러 도입 (Critical)

**문제**: `SoftLight(a, 0.5+h) = a + 2h·a·(1-a)`. 유효 gain `2a(1-a)`가 중간톤에서 0.5, 어두운/밝은 톤에서 0.18까지 떨어져 고주파 디테일 50-90% 손실.

**접근 방향**: 셰이더 내에서 `adjusted_high`를 Soft Light에 전달하기 전, 톤 의존 gain 보상을 적용.

```glsl
// 보상 전략: gain factor 2a(1-a)의 역수로 pre-scale
// 단, 극단값(a≈0, a≈1)에서 발산 방지를 위해 하한 클램프
float gain = 2.0 * smoothLow.r * (1.0 - smoothLow.r);  // luminance 대표값
float compensation = 1.0 / max(gain, 0.25);  // 최대 4x 보상, 하한 0.25
vec3 compensated_high = adjusted_high * compensation;
```

**주의사항**:
- luminance 단일 채널(`smoothLow.r` 또는 dot product)로 gain 계산 — 채널별 보상은 색상 왜곡 위험
- 하한값 0.25 → 최대 4x 보상. 너무 낮으면 노이즈 증폭, 너무 높으면 보상 부족
- 보상 후 `clamp(0.5 + compensated_high)` 범위가 넓어지므로 기존 clamp가 더 자주 작동 — 이는 안전장치로 정상 동작
- **튜닝 필요**: 하한값과 `high_freq_preserve` 계수(현재 0.70f) 재조정 필수

**검증**:
- gain 보상 전/후 고주파 보존율 비교표 작성
- skinQuality 0.2/0.5/1.0 × 피부톤 밝음/중간/어두움 9개 조합 검증

---

### Fix 2: GLSL-4 — MAD 최적화 (Medium)

**현재**:
```glsl
vec3 beauty = (vec3(1.0) - 2.0 * blend) * smoothLow * smoothLow
            + 2.0 * blend * smoothLow;
```

**변경**: 곱셈 4회 → 3회, MAD(Multiply-Add) 패턴 적합
```glsl
vec3 beauty = smoothLow * (smoothLow + 2.0 * blend * (vec3(1.0) - smoothLow));
```

수학적 동치 증명: `a*(a + 2b*(1-a)) = a² + 2ab - 2a²b = a² + 2ab(1-a) = (1-2b)*a² + 2b*a` ✓

---

### Fix 3: CPP-1 — 매직 넘버 상수화 (Medium)

**현재** (`gpu_beauty_backend.cpp:1006`):
```cpp
p.high_freq_preserve = 1.0f - s * 0.70f;
```

**변경**:
```cpp
constexpr float kMaxHighFreqAttenuation = 0.70f;
p.high_freq_preserve = 1.0f - s * kMaxHighFreqAttenuation;
```

---

### Fix 4: F2 — 셰이더 gain 특성 주석 추가 (Low)

`shader_sources.cpp` Soft Light 합성 블록에 gain 특성 주석 보강:

```glsl
// Soft Light 합성 (Pegtop variant)
// SoftLight(a, 0.5+h) = a + 2h·a·(1-a)
// → 유효 gain = 2a(1-a): 중간톤(a=0.5) 50%, 어두운/밝은(a=0.1/0.9) 18%
// → gain 보상 스케일러로 톤 의존 손실 보정
```

---

### Fix 5: T1 — CPU 참조 테스트 작성 (Low)

`test_beauty_config_v2.cpp`에 Pegtop Soft Light 수식 검증 테스트 추가:

| 테스트 케이스 | 검증 항목 |
|--------------|-----------|
| `SoftLightIdentity` | h=0 → blend=0.5 → beauty=base |
| `SoftLightBrighten` | h>0 → beauty>base |
| `SoftLightDarken` | h<0 → beauty<base |
| `SoftLightOutputRange` | base∈[0,1], blend∈[0,1] → result∈[0,1] |
| `SoftLightGainCharacteristic` | gain = 2a(1-a) 수치 검증 |
| `GainCompensationEffectiveness` | 보상 후 고주파 보존율 ≥ 80% (전 톤 범위) |

---

### Fix 6: D1 — 작업 문서 보완 (Medium)

`P4-W4-01b_soft_light.md` 수정:
1. Section 1에 gain 손실 한계 추가 (이점과 함께 한계도 기술)
2. Section 2.3 "C++ 측 변경: 없음" → 실제 `gpu_beauty_backend.cpp` 변경 반영
3. 완료 기준 미체크 2건 상태와 전체 상태 정합성 확인

---

## 실행 순서

```
Fix 1 (P2 gain 보상) ← 핵심, 셰이더 변경
  ↓
Fix 2 (GLSL-4 MAD 최적화) ← Fix 1과 같은 셰이더 블록, 동시 적용
  ↓
Fix 4 (F2 주석) ← Fix 1/2 완료 후 최종 주석
  ↓ (병렬)
Fix 3 (CPP-1 상수화) ← 독립적, cpp 파일
Fix 5 (T1 테스트) ← Fix 1 결과 반영한 테스트
  ↓
Fix 6 (D1 문서) ← 모든 수정 완료 후 문서화
```

---

## 변경 이력

| 날짜 | 내용 |
|------|------|
| 2026-03-09 | 이슈 수정 계획서 작성 |
| 2026-03-09 | Fix 1-6 전체 적용 완료, 52개 테스트 통과 (기존 46 + 신규 6) |
