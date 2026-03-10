# Comprehensive Code Review Report — P4-W4-01e

## Review Target

P4-W4-01e Luminance Sharpen 패스 추가
- Branch: `feature/P4-W4-01e` vs `develop`
- Commit: `060b011`
- 변경: 5 files, +170 lines (code only)

## Executive Summary

FreqSep 파이프라인에 Luminance-only Unsharp Mask 패스를 추가하는 변경으로, 기존 아키텍처 패턴을 충실히 따르며 graceful degradation 설계가 우수합니다. 그러나 **sRGB/Linear 색공간 불일치**(Critical)가 파이프라인 일관성을 위반하고, **full-res RT 추가 할당**(High)이 저가 디바이스 메모리 대역폭에 영향을 줄 수 있습니다. sharpen_amount가 0.12~0.18로 작아 시각적 임팩트는 제한적이지만, 기술적 부채로 관리가 필요합니다.

---

## Findings by Priority

### Critical (P0) — 1건

| ID | 카테고리 | 설명 | 파일 |
|----|----------|------|------|
| F-1 | Code Quality / 셰이더 | sRGB 공간에서 Linear LUMA_709 계수 사용 — Composite가 gamma 인코딩 후 출력하므로 Sharpen이 sRGB 데이터에 linear 계산 적용. 톤 영역별 샤프닝 불균일 | `shader_sources.cpp:580-604` |

**영향**: sharpen_amount 0.12~0.18 범위에서 실질적 시각 차이는 미미하나, 파이프라인 색공간 원칙 위반. 어두운 피부톤에서 과도한 샤프닝 가능.

**권장 조치**: 주석으로 의도적 trade-off 명시 + 후속 태스크로 추적 (현재 범위에서 blocking은 아님)

---

### High (P1) — 2건

| ID | 카테고리 | 설명 | 파일 |
|----|----------|------|------|
| F-2 | Code Quality | Sharpen 셰이더 컴파일 실패 시 `luminance_sharpen_program_` 미초기화 + 오해 유발 "successfully" 로그 | `gpu_beauty_backend.cpp:288-296` |
| PERF-1 | Performance | Full-res compositeRT 추가 할당 — MID 디바이스 대역폭 13.7% 추가 소비 추정 | `gpu_beauty_backend.cpp:1290` |

---

### Medium (P2) — 5건

| ID | 카테고리 | 설명 | 파일 |
|----|----------|------|------|
| F-3 | Architecture | temp RT 조기 릴리스로 메모리 피크 절감 가능 | `gpu_beauty_backend.cpp:1289` |
| F-4 / SEC-2 | Security | `sharpen_amount` uniform 범위 클램핑 없음 (public struct) | `gpu_beauty_backend.cpp:1364` |
| F-5 | Testing | 테스트 tolerance 0.02f 과도 — 로직 변경 감지 불가 | `test_beauty_config_v2.cpp:478` |
| SEC-1 | Security | compositeRT GPU 리소스 유효성에 대한 암묵적 mutex 의존 | `gpu_beauty_backend.cpp:1290` |
| PERF-3 | Performance | Luminance ratio RGB 곱셈 — 채도 높은 영역 색상 왜곡 가능 | `shader_sources.cpp:601-603` |

---

### Low (P3) — 6건

| ID | 카테고리 | 설명 |
|----|----------|------|
| F-6 | Code Quality | LUMA_709 상수 셰이더 간 중복 |
| F-7 | Testing | SharpenAmountRange 경계값 느슨 (0.11~0.19 vs 실제 0.12~0.18) |
| F-8 | Testing | SharpenAmountDisabledWhenZero에서 기본값 미검증 |
| SEC-3 | Security | uTexelSize division width=0 간접 방어 |
| SEC-4 | Security | LOGE vs LOGW 불일치 (graceful degradation인데 error 로그) |
| PERF-4 | Performance | `> 0.01f` 임계값과 매핑 최솟값 0.12 불일치 |

---

## Findings by Category

| 카테고리 | 건수 | Critical | High | Medium | Low |
|----------|------|----------|------|--------|-----|
| Code Quality | 5 | 1 | 1 | 1 | 2 |
| Security | 4 | 0 | 0 | 2 | 2 |
| Performance | 4 | 0 | 1 | 1 | 2 |
| Testing | 3 | 0 | 0 | 1 | 2 |
| **Total** | **14** | **1** | **2** | **5** | **6** |

---

## Recommended Action Plan

### 병합 전 (Small effort)

| # | 작업 | 노력 | ID |
|---|------|------|-----|
| 1 | `luminance_sharpen_program_ = 0` 명시적 초기화 + `LOGE`→`LOGW` + 로그 수정 | 2분 | F-2, SEC-4 |
| 2 | `std::clamp(params.sharpen_amount, 0.0f, 0.5f)` 방어 코드 | 1분 | F-4/SEC-2 |
| 3 | F-1에 대한 주석 추가 (sRGB 공간에서 의도적 동작임을 명시) | 2분 | F-1 |

### 병합 후 (QA 단계)

| # | 작업 | 시점 |
|---|------|------|
| 1 | Mali-G52 이하 실 기기 6패스 프레임 타임 측정 | 즉시 |
| 2 | 다양한 피부톤에서 색상 왜곡 시각적 QA | QA |
| 3 | 텍스처 풀 메모리 사용량 모니터링 | QA |

### 다음 스프린트 (Medium effort)

| # | 작업 | ID |
|---|------|-----|
| 1 | temp RT 조기 릴리스로 메모리 피크 절감 | F-3 |
| 2 | 테스트 tolerance 축소 + 경계값 정밀화 | F-5, F-7, F-8 |
| 3 | sRGB/Linear 색공간 정리 (파이프라인 전체) | F-1 |

---

## 긍정적 관찰

1. **기존 아키텍처 패턴 완벽 준수** — 셰이더 선언~mapSkinQuality 6단계 패턴 1:1 일치
2. **Graceful degradation 3단계 방어** — 셰이더 실패, RT 할당 실패, 런타임 스킵
3. **프로파일러 통합** — `FreqSep_Sharpen` 태그로 성능 회귀 감지 가능
4. **리소스 수명 관리** — acquire/release 쌍이 블록 내 완결, 누수 없음
5. **테스트 커버리지** — 5개 테스트로 중간값/최대값/비활성/범위/단조성 검증

---

## Review Metadata

- Review date: 2026-03-10
- Phases completed: Phase 1 (Quality & Architecture), Phase 2 (Security & Performance)
- Phases skipped: Documentation, CI/CD (사용자 요청)
- Flags: performance_critical, critical_perspective, skip_docs, skip_cicd
- Framework: C++17 / GLSL ES 3.1
