# Comprehensive Code Review Report

## Review Target

**P4-W4-01c: Edge-aware Attenuation (3-신호 결합)**
브랜치: `feature/feature/P4-W4-01c` vs `develop`
변경 규모: 5 files, +80/-11 lines

## Executive Summary

단일 magnitude 신호를 3-신호 결합(Magnitude + Edge Gradient + Chroma Deviation)으로 확장하는 변경. 기존 FreqSep 파이프라인의 구조적 패턴을 정확히 따르며 최소한의 표면적 변경으로 구현되었습니다.

**주요 리스크**: Edge Gradient가 gamma-encoded sRGB 공간에서 계산되면서 linear 공간의 magnitude/chromaDev와 곱셈으로 결합됩니다. 이로 인해 에지 보존 강도가 노출/피부톤에 의존하게 되며, 설계 문서(`P4-W4-01c_edge_aware_attenuation.md` line 175)에서 명시한 "에지 계산도 linear 공간에서 수행" 방향과 최종 구현 사이에 괴리가 있습니다. 이 이슈는 병합 전 해결 또는 명시적 수용 결정이 필요합니다.

**성능 판단 유보**: Performance Critical 플래그가 설정되었으나, 실측 프레임 타임/GPU 프로파일링/셰이더 컴파일 결과 없이 추정치만으로 평가되었습니다. 30fps 달성 가능성은 이론적 분석상 양호하나, 정량적 근거가 부족하므로 실 기기 측정 전까지 확정할 수 없습니다.

---

## Findings by Priority

### Critical Issues (P0) — 없음

### High Priority (P1) — 1건

**P1-1: Edge Gradient의 sRGB/Linear 색공간 불일치 — 출력 품질 회귀 리스크**
- `shader_sources.cpp` line 491, 500-522
- **문제**: `orig`는 line 491에서 `pow(orig, vec3(2.2))`로 linearize됨. 그러나 에지 검출용 4-neighbor 샘플(line 503-506)은 `texture(uOriginal, ...)`를 직접 사용하여 gamma-encoded sRGB 공간에 남아 있음. 이 sRGB `edgeStrength`가 linear 공간의 `magnitude`와 곱셈으로 결합됨(line 520-522)
- **영향 범위**: 밝은 하이라이트 **및** 어두운 그림자 피부 양쪽에서 발생하는 노출 의존성 문제
  - 어두운 영역: 감마 확장으로 에지 과대평가 → 에지 보존 과잉 활성 (잡티도 보존)
  - 밝은 영역: 감마 압축으로 에지 과소평가 → 에지 보호 부족 (주름 삭제 위험)
  - 즉, 동일한 물리적 에지가 피부톤/조명에 따라 다르게 처리됨
- **설계 문서와의 괴리**: 설계 문서 line 175에 "에지 계산도 linear 공간에서 수행 (orig을 이미 linearize했으므로)"라고 명시되어 있으나, 실제 구현은 Option B(sRGB 에지 검출, pow 4회 절약)로 변경됨. 이 변경 결정이 설계 문서에 반영되지 않았으며, 트레이드오프에 대한 명시적 승인 기록이 없음
- **KI-1 기록 상태**: 작업 문서에 이미 기록됨. 그러나 "P2, QA 후 판단"이 아닌 "병합 전 해결 또는 명시적 수용 결정" 수준의 이슈
- **수정 옵션**:
  - **(A)** `pow(sample, vec3(2.2))` 4회 추가 — 정확하지만 +4 pow() 비용
  - **(B 권장)** `sample * sample` (gamma 2.0 근사) — 오차 ~5%, 비용 최소, linear 공간 근사 달성
  - **(C)** sRGB 유지 + 설계 문서 업데이트 + 스케일 팩터 별도 튜닝 — 의식적 수용 경로

### Medium Priority (P2) — 3건

**P2-1: `diff`와 `high` 변수 중복 계산**
- `shader_sources.cpp` line 495 vs 512
- `vec3 high = orig - low`과 `vec3 diff = orig - low`가 동일한 연산
- GPU CSE 최적화 가능하나, 모바일 드라이버 CSE가 불안정할 수 있으며 가독성 혼란
- **수정**: `diff` 제거 → `high` 직접 사용

**P2-2: LUMA_709 상수와 인라인 리터럴 혼재**
- `shader_sources.cpp` line 483 vs 534
- `LUMA_709` 상수를 도입했으나 Soft Light 코드에서 인라인 `vec3(0.2126, 0.7152, 0.0722)` 사용
- DRY 위반, 계수 변경 시 동기화 누락 위험
- **수정**: `float baseLum = dot(smoothLow, LUMA_709);`

**P2-3: 매직 넘버 스케일 팩터**
- `shader_sources.cpp` line 521-522
- `edgeStrength * 5.0`, `chromaDev * 10.0` 하드코딩
- 정규화 범위의 의미가 코드에서 드러나지 않음
- **수정**: `const float EDGE_SCALE = 5.0;` / `const float CHROMA_SCALE = 10.0;`으로 명명

### Low Priority (P3) — 5건

**P3-1: FreqSepParams 기본값과 mapSkinQuality 범위 소폭 불일치**
- `chroma_weight` 기본값 0.3f는 매핑 범위 [0.2, 0.5]의 하한 근처
- `enabled = false` 기본이므로 실질적 영향 없음

**P3-2: Uniform 값 클램핑 미적용**
- `edge_weight > 1.0` 시 blemishScore 음수 반전 가능 (기능 무력화, 보안 위험 아님)
- 현재 `mapSkinQuality()`만이 생성 경로이므로 즉각적 위험 없음
- 방어적 프로그래밍 관점에서 `std::clamp` 적용 권장

**P3-3: 테스트 범위 검증 정밀도**
- 실제 범위 [0.3, 0.7]에 대해 [0.0, 1.0]으로 검증 → 범위 2배 이상 넓음

**P3-4: sqrt() 2회 최적화 후보**
- 제곱 도메인 비교로 대체 가능하나 비선형 응답 변경 → 시각 검증 필요
- Mali-G52 이하 저가 디바이스에서 측정 후 판단

**P3-5: 경계값 테스트 부재**
- `skinQuality` 극단값(0, >1, 음수)에서 edge/chroma weight 명시적 검증 없음

---

## Findings by Category

| 카테고리 | 건수 | Critical | High | Medium | Low |
|----------|------|----------|------|--------|-----|
| 출력 품질/동작 회귀 | 1 | 0 | 1 | 0 | 0 |
| Code Quality | 3 | 0 | 0 | 3 | 0 |
| Architecture | 2 | 0 | 0 | 0 | 2 |
| Security/방어 코드 | 1 | 0 | 0 | 0 | 1 |
| Performance | 1 | 0 | 0 | 0 | 1 |
| 테스트 커버리지 | 1 | 0 | 0 | 0 | 1 |
| **합계** | **9** | **0** | **1** | **3** | **5** |

---

## 검증 상태 및 한계

### 검증된 것
- C++ 측 `mapSkinQuality()` 범위/단조성 테스트 17건 통과 (`test_beauty_config_v2.cpp`)
- 수치 안전성: clamp/smoothstep으로 NaN/Inf/GLSL UB 위험 없음
- 메모리 안전성: 동적 할당/포인터 연산 없음, GLint -1 초기값 방어

### 검증되지 않은 것 (리뷰 한계)
- **셰이더 출력 품질**: 피부톤별/노출별 시각적 에지 보존 회귀 미검증
- **GPU 성능**: 실 기기 프레임 타임, GPU 프로파일링 데이터 없음 (이론적 추정만 수행)
- **셰이더 컴파일**: 타겟 GPU 드라이버에서의 실제 컴파일 결과/레지스터 사용량 미확인
- **디바이스 매트릭스**: Mali-G52 이하 저가 기기에서의 실측 없음

이 리뷰는 코드 수준 정적 분석이며, 위 항목들은 실 기기 통합 테스트에서 검증되어야 합니다.

---

## Positive Observations

- **컴포넌트 경계**: 기존 레이어 구조(shader/params/backend) 완벽 준수
- **Uniform 패턴 일관성**: 3단계 흐름(헤더→cache→execute) 기존 패턴과 1:1 일치
- **하위 호환성**: in-class initializer로 기존 코드 경로 안전
- **분기 분산 없음**: 셰이더에 if/else 없이 ALU만으로 구현 — 이상적인 GPU 실행 패턴
- **패스 추가 없음**: 기존 composite 패스 내 ALU 확장만으로 구현

---

## Recommended Action Plan

| # | 작업 | 우선순위 | 분류 | 노력 |
|---|------|---------|------|------|
| 1 | **P1-1 해결**: sRGB/Linear 혼용 수정 또는 명시적 수용 결정 | **병합 전** | 동작 회귀 | Medium |
| 2 | `diff` → `high` 재사용 (P2-1) | 병합 전 | 코드 정리 | Small |
| 3 | `LUMA_709` 상수 통일 (P2-2) | 병합 전 | 코드 정리 | Small |
| 4 | 스케일 팩터 상수 명명 (P2-3) | 병합 전 | 코드 정리 | Small |
| 5 | 실 기기 GPU 프로파일링 (성능 검증) | 병합 후 즉시 | 성능 검증 | Medium |
| 6 | Uniform 클램핑 방어 코드 (P3-2) | 다음 스프린트 | 방어 코드 | Small |
| 7 | 테스트 범위 정밀화 (P3-3, P3-5) | 백로그 | 테스트 | Small |
| 8 | sqrt 최적화 프로파일링 (P3-4) | 저가 기기 테스트 시 | 성능 | Medium |

**병합 전 필수**: #1 (출력 품질 리스크 해결) + #2~#4 (코드 정리)
**병합 후 즉시**: #5 (성능 실측)

### P1-1 해결 경로 옵션

| 옵션 | 내용 | 비용 | 권장 |
|------|------|------|------|
| A | `pow(sample, vec3(2.2))` 4회 추가 | +4 pow(), ~0.3ms 추가 | 정확성 최우선 시 |
| **B** | **`sample * sample` (gamma 2.0 근사)** | **+4 mul, 오차 ~5%** | **권장 — 비용/정확성 균형** |
| C | sRGB 유지 + 설계 문서 업데이트 + 스케일 팩터 재튜닝 | 문서 작업 | 현 동작 수용 시 |

---

## Review Metadata

- Review date: 2026-03-10
- Phases completed: Phase 1 (Code Quality & Architecture), Phase 2 (Security & Performance)
- Flags: Performance Critical (GPU shader, 30fps target)
- Framework: C++17 / GLSL ES 3.1
- **Review limitation**: 정적 코드 분석만 수행. 셰이더 출력 품질, GPU 프로파일링, 디바이스 매트릭스 테스트는 미포함.
