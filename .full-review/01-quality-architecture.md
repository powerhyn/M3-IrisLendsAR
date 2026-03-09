# Phase 1: Code Quality & Architecture Review

## 대상: P4-W4-01b Soft Light 합성 전환

---

## Code Quality Findings (6건)

### Critical (2건)

| ID | 위치 | 요약 |
|----|------|------|
| **P1** | texture_pool.cpp:337 | **Linear-light 값의 8-bit 양자화**: 중간 버퍼(lowFreq, smoothedLow)가 GL_RGBA/GL_UNSIGNED_BYTE로 생성됨. linear 색공간에서 8-bit는 어두운 영역에서 ~8% 밝기 점프 → 심각한 밴딩/포스터라이제이션. P4-W4-01a에서 도입된 문제이나 Soft Light 합성이 `smoothLow`를 base로 사용하므로 아티팩트 증폭. |
| **P2** | shader_sources.cpp:499-503 | **Soft Light 고주파 gain 손실**: `SoftLight(a, 0.5+h) = a + 2h·a·(1-a)`. gain 계수 `2a(1-a)`의 최대값이 0.5(a=0.5)이고 극단 톤에서 0.18까지 하락. uHighFreqPreserve=1.0에서도 50~90% 고주파 손실. 0.65→0.70 튜닝 조정(5%p)으로는 보상 불가. skinQuality 전 범위에서 사용자 체감 회귀 발생 가능. |

### Medium (1건)

| ID | 위치 | 요약 |
|----|------|------|
| F1 | shader_sources.cpp:481-484 | **smoothLow/low의 양자화 노이즈 혼입**: P1으로 인해 `low`가 양자화된 값 → `high = orig - low`에서 가짜 고주파 생성. P1 수정 시 자동 해결. |

### Low (3건)

| ID | 위치 | 요약 |
|----|------|------|
| F2 | shader_sources.cpp:499-503 | gain 특성 `2a(1-a)` 및 설계 의도에 대한 주석 누락 |
| F3 | gpu_beauty_backend.cpp:1005 | `high_freq_preserve` 파라미터 의미론 변경 미반영 — Additive에서 1.0="100% 보존"이었으나 Soft Light에서는 최대 50% 보존 |
| F4 | shader_sources.cpp:499-503 | ALU 3-4 ops 추가 — 성능 영향 무시 가능 (info) |

---

## Architecture Findings (4건)

### 긍정적 평가
| 항목 | 판정 |
|------|------|
| 파이프라인 구조 | 유지됨 — Pass 추가 없음, 기존 3-pass 동일 |
| Uniform 인터페이스 | 유지됨 — 새 uniform 없음 |
| FreqSepParams 구조체 | 유지됨 — 필드 추가/삭제 없음 |
| DeviceTier 분기 | 영향 없음 — MID half-res, LOW fallback 동일 |
| Bilateral fallback 경로 | 영향 없음 |

### 개선 필요
| ID | 항목 | 설명 |
|----|------|------|
| A1 | Soft Light 수식 정확성 | Pegtop 공식 구현 자체는 수학적으로 정확. Identity 조건, 출력 범위 [0,1] 보장 확인됨. |
| A2 | gain 비보상 설계 | 구조적 문제: Additive→Soft Light 전환 시 gain 보상 없이 수식만 교체하여 파라미터 semantics 불일치 |
| A3 | 테스트 호환성 | `test_beauty_config_v2.cpp:331-332`의 high_freq_preserve 범위 [0.30, 0.40] 검증이 경계값(0.30)에 걸림 |
| A4 | 중간 버퍼 포맷 하드코딩 | TexturePool::createTexture()가 GL_RGBA8로 고정 → 색공간 전환 시 유연성 부재 |

---

## Critical Issues for Phase 2 Context

1. **P1 (8-bit 양자화)**: 성능 리뷰에서 GL_RGBA16F/GL_R11F_G11F_B10F 전환 비용 분석 필요
2. **P2 (gain 손실)**: 보상 스케일러 도입 시 추가 ALU 비용 및 register pressure 분석 필요
3. **pow() 비용**: sRGB↔Linear 변환에 pow() 2회 사용 — 모바일 GPU에서의 SFU 비용 분석 필요
