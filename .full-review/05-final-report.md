# Comprehensive Code Review Report

## Review Target

**P4-W4-01b**: Soft Light 합성 전환 — FreqSep Composite 셰이더 Additive→Soft Light(Pegtop) + 감쇠 계수 재튜닝

**브랜치**: `feature/P4-W4-01b` (develop 기준 1커밋, 3파일, +11/-8)

**리뷰 범위**: 이번 브랜치에서 변경된 코드만 대상. 기존 코드(P4-W4-01a 등)의 이슈는 "참고 사항"으로 분리.

---

## Executive Summary

Soft Light (Pegtop variant) 수식은 **수학적으로 정확하게** 구현되었으며, 아키텍처 변경 없이 셰이더 ALU만 교체하는 깔끔한 변경입니다. 그러나 **1건의 Critical 이슈**가 식별되었습니다:

- **P2 (gain 손실)**: Soft Light의 본질적 특성으로 고주파 디테일의 50~90%가 손실됨. `2a(1-a)` gain 계수가 최대 0.5이므로 `uHighFreqPreserve=1.0`에서도 절반 이상 손실. 0.65→0.70 조정(5%p)으로는 보상 불가. skinQuality 전 범위에서 사용자 체감 블러 회귀 위험.

---

## 변경 사항 요약

| 파일 | 변경 내용 |
|------|-----------|
| `shader_sources.cpp:499-503` | Additive `smoothLow + adjusted_high` → Soft Light `(1-2b)*a²+2b*a` |
| `gpu_beauty_backend.cpp:1005-1006` | `high_freq_preserve` 감쇠 계수 0.65→0.70, 주석 업데이트 |
| `P4-W4-01b_soft_light.md` | 작업 상태 ⏳→✅, 완료 기준 체크 |

---

## Findings — 이번 변경 한정

### Critical — 즉시 수정 필요 — 1건

| ID | 위치 | 요약 | 개선 방향 |
|----|------|------|-----------|
| **P2** | shader_sources.cpp:499-503 | **Soft Light gain 손실 50-90%**: `SoftLight(a,0.5+h) = a+2h·a·(1-a)`. gain `2a(1-a)` — 중간톤(a=0.5) 50% 손실, 어두운/밝은 톤(a=0.1/0.9) 82~90% 손실. 감쇠 계수 5%p 조정으로 보상 불가. | gain 보상 스케일러 도입 (예: 톤 의존 보정) 또는 Additive-SoftLight 블렌딩 방식 검토. 구체적 수식은 프로파일링 후 결정 권장. |

**gain 손실 상세:**

| smoothLow (a) | gain = 2a(1-a) | 고주파 보존율 |
|----------------|----------------|-------------|
| 0.10 (매우 어두움) | 0.18 | **18%** |
| 0.20 (어두운 피부) | 0.32 | **32%** |
| 0.50 (중간톤) | 0.50 | **50%** |
| 0.80 (밝은 피부) | 0.32 | **32%** |
| 0.90 (매우 밝음) | 0.18 | **18%** |

### Medium — 3건

| ID | 카테고리 | 위치 | 요약 |
|----|----------|------|------|
| **GLSL-4** | Best Practices | shader_sources.cpp:502-503 | **Soft Light 수식 MAD 최적화**: 현재 `(1-2b)*a²+2b*a` (곱셈 4회) → `a*(a+2b*(1-a))` (곱셈 3회). MAD 패턴 적합. |
| **CPP-1** | Best Practices | gpu_beauty_backend.cpp:1006 | **매직 넘버 상수화**: `0.70f` → `constexpr float kMaxHighFreqAttenuation = 0.70f;` |
| **D1** | Documentation | P4-W4-01b_soft_light.md | **gain 손실 한계 미문서화**: gain `2a(1-a)` 특성과 50-90% 고주파 손실이 작업 문서에 미기록. 이점만 기술하고 한계 누락. 완료 기준 2건 미체크인데 상태 "✅ 완료". |

### Low — 2건

| ID | 카테고리 | 위치 | 요약 |
|----|----------|------|------|
| **F2** | Code Quality | shader_sources.cpp:499-503 | **gain 특성 주석 누락**: `2a(1-a)` 최대 0.5라는 특성과 `high_freq_preserve`의 의미론 변경(1.0이 더이상 "100% 보존"이 아님)에 대한 주석 부재. |
| **T1** | Testing | — | **CPU 참조 테스트 부재**: Pegtop 수식의 identity, 출력 범위, gain 특성을 검증하는 단위 테스트 없음. 셰이더 롤백 시 어떤 테스트도 실패하지 않음. |

---

## 긍정적 평가

| 항목 | 판정 |
|------|------|
| Pegtop 수식 구현 정확성 | ✅ Identity(h=0→base), 출력 [0,1] 보장, edge case 안전 |
| 아키텍처 영향 | ✅ Pass 추가 없음, Uniform 변경 없음, 구조체 변경 없음 |
| ALU 성능 영향 | ✅ +3-4 ops, texture fetch latency에 숨겨짐. 30fps 영향 없음 |
| Warp divergence | ✅ 조건 분기 없음 (Overlay 대비 이점) |
| clamp 안전장치 | ✅ adjusted_high ∈ [-1,1] 가능하므로 clamp 필수, 올바르게 적용됨 |
| 기존 테스트 호환성 | ✅ 46개 테스트 전부 통과 — `cmake --build build --target test_beauty_config_v2 && ./build/bin/test_beauty_config_v2` 실행 확인. 경계값 0.30이 기존 테스트 허용 범위 [0.30, 0.40] 내 정확히 포함. |

---

## Recommended Action Plan

| 우선순위 | 작업 | 노력 | 효과 |
|----------|------|------|------|
| **즉시** | P2 gain 보상 방안 설계 및 검증 | Medium | skinQuality 전 범위 블러 회귀 해소 |
| 이번 스프린트 | T1 CPU 참조 테스트 작성 | Small | 수식 검증 + 회귀 방지 |
| 이번 스프린트 | GLSL-4 수식 리팩터링 `a*(a+2b*(1-a))` | Small | 곱셈 1회 절감 |
| 다음 스프린트 | D1 문서 보완 + F2 주석 추가 + CPP-1 상수화 | Small | 유지보수성 개선 |

---

## 참고 사항 — 기존 코드 이슈 (이번 변경 범위 밖)

아래는 이번 브랜치 변경과 직접 관련 없으나, 동일 셰이더/파이프라인에서 발견된 기존 이슈입니다. 별도 작업으로 추적 권장.

| ID | 출처 | 요약 | 권장 작업 |
|----|------|------|-----------|
| P1 | P4-W4-01a | 8-bit 렌더 타겟에 linear-light 저장 → 어두운 영역 밴딩 | `GL_R11F_G11F_B10F` 전환 |
| SRGB-1 | P4-W4-01a | 수동 pow(2.2) 비용+부정확 → 하드웨어 sRGB 활용 | `GL_SRGB8_ALPHA8` + `GL_FRAMEBUFFER_SRGB` |
| SEC-2 | 기존 | mapSkinQuality NaN/Inf 가드 누락 | `std::isnan` 가드 추가 |

---

## Review Metadata

- **Review date**: 2026-03-09
- **Scope**: 이번 브랜치(feature/P4-W4-01b) 변경분 한정
- **Phases completed**: 1~5 (전체)
- **Flags applied**: Performance Critical
- **Total findings**: 6건 (Critical: 1 / Medium: 3 / Low: 2) + 참고 3건
- **Reviewers**: code-reviewer, security-auditor (specialized agents)
