# Comprehensive Code Review Report — Vivid Post-Processing Filter

**Date**: 2026-03-12
**Branch**: develop 대비 현재 브랜치
**Scope**: +307/-12 lines, 10 files

---

## Review Target

화면 전체에 화사한 느낌을 주는 GPU 전용 포스트프로세싱 필터 구현.
Vibrance + 밝기 리프트 + 웜톤 시프트를 단일 패스 GLSL 셰이더로 구현하여 기존 beauty 파이프라인 끝에 삽입. `enabled=false` 상태에서도 vivid만 독립 활성화 가능.

## Executive Summary

**Mixed — C API(JNI) 경로는 양호하나, 공개 C++ API 경로에 미수정 결함 존재. 머지 전 수정 필수.**

기존 SDK 레이어 아키텍처(C++ → C API → JNI → Java)를 충실히 따르면서 최소 침투로 vivid 기능을 통합했다. C API 경로(`iris_sdk_apply_beauty_texture_v2`)의 셰이더 설계, 파이프라인 통합, 바인딩 일관성은 프로덕션 수준이다. 그러나 **smoothstep GLSL ES spec 위반(디바이스별 렌더링 불일치)**, **공개 C++ API 경로(`applyTexture`) vivid 완전 누락 + 미초기화 텍스처 반환** 2건은 기능적 결함이므로 머지 전 반드시 수정해야 한다.

---

## Findings by Priority

### P0 — 머지 전 수정 필수 (2건)

| ID | 카테고리 | 이슈 | 파일 | 수정 규모 |
|----|----------|------|------|-----------|
| **SHADER-01** | **Correctness** | **`smoothstep(0.4, 0.0, sat)` — GLSL ES spec에서 edge0 ≥ edge1일 때 undefined behavior. 디바이스/드라이버별 vibrance 효과 차이 또는 소실 발생 가능.** `1.0 - smoothstep(0.0, 0.4, sat)`로 교체 필요. | `shader_sources.cpp:660` | **1줄 수정** |
| **API-01** | **Correctness** | **`applyTexture(TextureHandle)` 경로에 vivid 패스 완전 누락** — (1) `enabled=false + vivid` 시 즉시 패스스루로 vivid 미실행, (2) `enabled=true + vivid only` 시 beauty 체인이 모두 스킵되어 `current_input`이 원본 그대로인데, line 701의 ping/pong 판별이 미초기화 pong 텍스처를 반환하는 심각한 버그. 공개 C++ API가 깨진 상태. | `gpu_beauty_backend.cpp:605-713` | ~30줄 |

### P1 — 다음 릴리즈 전 수정 권장 (5건)

| ID | 카테고리 | 이슈 | 수정 규모 |
|----|----------|------|-----------|
| ROBUST-01 | Robustness | `toCppConfigV2()`에서 NaN/Inf 방어 부재 — `clamp()`의 `<`/`>` 비교는 NaN을 통과시키므로, `std::isfinite()` 기반 검증 또는 NaN-safe 클램핑(`std::isnan(v) ? lo : ...`) 필요. Builder/clamp를 우회하는 모든 입력 경로가 영향 (C API 직접 호출, Java에서 public 필드 직접 설정 후 JNI 전달 등). | 15줄 |
| A-M-02 | Architecture | vivid 셰이더 컴파일 실패 = 전체 초기화 실패 → FreqSep처럼 non-fatal 처리 | 10줄 |
| H-01 | Quality | ROI 계산 로직 3중 중복 → 공통 헬퍼 함수 추출 | 1시간 |
| M-04/A-H-02 | Quality+Arch | SoftFocus 후 ping-pong 스왑이 `needsVivid`에 하드코딩 → 일관된 스왑 규칙 | 10분 |
| ABI-01 | Robustness | POD 구조체 ABI 버전링 부재 → `uint32_t struct_size` 도입 검토 (iOS/Flutter 바인딩 확장 전 대응) | 2시간 |

### P2 — 백로그 (6건)

| ID | 카테고리 | 이슈 |
|----|----------|------|
| M-01/A-M-03 | Quality+Arch | buildEffectiveConfig에서 vivid 독립성 주석 추가 |
| M-02 | Quality | warmth 셰이더 매직 넘버(0.04, 0.02, 0.03) 주석 보강 |
| A-M-01 | Architecture | ConfigV2 구조체 비대화 경향 — 포스트프로세싱 효과 추가 시 서브 구조체 분리 |
| Low-threshold | Arch | needsVivid 임계값 0.01f 매직 넘버 → 상수 추출 |
| Low-precision | Performance | `precision highp` → `mediump` 전환 검토 (~15-25% ALU 개선) |
| S-07/S-08 | Robustness | int overflow guard, static_assert 추가 |

---

## Findings by Category

| 카테고리 | 건수 | 설명 |
|----------|------|------|
| Correctness | 2 | smoothstep spec 위반, applyTexture 경로 누락 |
| Robustness | 2 | NaN/Inf 입력 방어, ABI 버전링 |
| Architecture | 3 | 셰이더 non-fatal 처리, ROI 중복, ping-pong 스왑 규칙 |
| Quality | 3 | 주석 보강, 매직 넘버, 구조체 비대화 |
| Performance | 2 | mediump 전환, 상수 추출 |

**중복 제거 후 실제 고유 이슈: 13건** (P0: 2, P1: 5, P2: 6)
*카테고리 표는 주제별 분류이며, 일부 이슈가 복수 카테고리에 걸쳐 있어 합산과 고유 건수가 다름*

---

## 잘 구현된 부분

1. **Config/Binding 레이어 일관성** — C++ struct, C API struct, JNI field ID cache, Java class에서 vivid 4필드가 빈틈없이 반영. Builder, isValid, clamp, toString, 복사 생성자 모두 완비.
2. **buildEffectiveConfig 분리** — enabled=false 시 beauty 수치를 중립값으로 덮어써서 파이프라인 하류의 조건 분기 최소화. 설계 의도가 코드에 잘 반영됨.
3. **셰이더 설계** — 단일 패스, 텍스처 샘플 1회 + ALU 31ops. MID-tier에서 ~0.25ms로 0.3ms 목표 달성. 프레임 버짓의 0.75%만 추가.
4. **ROI/Scissor 통합** — Scissor 해제 후 vivid 전체 프레임 적용, ROI 교집합 empty 시 vivid-only 처리 등 엣지 케이스 정확히 처리.
5. **C API/JNI 경로에서 기존 패턴을 잘 따름** — Profiler 통합, uniform cache, release 정리, ping-pong 버퍼, JNI field ID 캐시 등. 단, 공개 C++ API 경로(`applyTexture`)에는 vivid 반영이 누락되어 해당 경로의 패턴 준수는 불완전.
6. **하위 호환** — 기본값 0.0f로 기존 사용자 동작 변화 없음.

---

## Recommended Action Plan

### 즉시 (P0, ~1시간)

1. `smoothstep(0.4, 0.0, sat)` → `1.0 - smoothstep(0.0, 0.4, sat)` 교체 (GLSL ES spec 준수)
2. `applyTexture(TextureHandle)` 진입부에 `needsVivid` 가드 + 필터 체인 끝에 vivid 패스 추가 + 0-pass 시 output 판별 로직 수정

### 다음 릴리즈 전 (P1, ~4시간)

3. `toCppConfigV2()`에 `std::isfinite()` 기반 NaN/Inf 검증 추가 (단순 clamp로는 NaN 통과)
4. vivid 셰이더 컴파일 실패를 non-fatal (LOGW) 처리 + executeVividPass에 program==0 가드
5. ROI 계산 공통 헬퍼 함수 추출
6. SoftFocus 후 스왑 조건을 무조건 스왑으로 통일
7. POD 구조체 ABI 버전링 전략 검토

---

## Review Metadata

- Phases completed: Phase 1 (Code Quality + Architecture), Phase 2 (Security + Performance)
- Phases skipped: Phase 3 (Testing & Documentation), Phase 4 (Best Practices) — 사용자 판단으로 생략
- Total unique findings: 13건 (P0: 2, P1: 5, P2: 6)
- Post-review feedback incorporated:
  - SHADER-01 추가 (smoothstep spec violation)
  - API-01 설명 보강 (uninitialized texture return 시나리오)
  - pong "누수" 주장 철회 → 지연 반환으로 정상 동작 확인
  - NaN clamp 수정안 교체 → `std::isfinite()` 기반으로, P1 robustness로 격하
  - Executive Summary 톤 조정 → "Mixed, not ready to merge without fixes"
