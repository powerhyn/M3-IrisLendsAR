# Comprehensive Code Review Report

## Review Target

**P4-W3-04: Temporal Stability + Device Tier 분기 구현**
- 브랜치: `feature/P4-W3-04`
- 프레임워크: C++17 / OpenGL ES 3.1
- 성능 중요도: High (실시간 30fps GPU 렌더링 파이프라인)
- 변경 규모: 3 files, +318 insertions, -34 deletions

### 변경 요약
- One Euro Filter로 blur_radius 및 mask center(cx, cy) temporal stability 확보
- GPU 렌더러 문자열 기반 DeviceTier(HIGH/MID/LOW) 판정
- MID tier 하이브리드 해상도 FreqSep 파이프라인 (블러 half-res, Composite full-res)
- LOW tier Bilateral fallback 분기

---

## Executive Summary

P4-W3-04는 GPU 뷰티 파이프라인에 디바이스 적응형 성능 최적화를 도입했다. MID tier에서 대역폭 -54.6%, GPU 메모리 -75% 절감을 달성했으며, temporal filtering으로 시각적 안정성을 개선했다. 코드 설계는 전반적으로 견고하나, **파이프라인 코드 중복(~140줄)**, **P4-W3-04 신규 기능 전용 테스트 부재**, **CI/CD 파이프라인 미구축**이 주요 우려사항이다.

Codex 리뷰 피드백 반영으로 3건의 실질적 결함이 수정되었다 (커밋 b80377f): OneEuro 리셋 누락(Q3), std::stoi 예외 위험(Q5), static public 메서드 오용 가능성(Q6). P2(mask center smoothing 타이밍)는 scissor 타이밍만 수정되었으며, 마스크 안정화 효과는 제한적이다(아래 상세 참조).

### P2 정정 사항 (Codex 2차 피드백 반영)

초기 리뷰에서 P2를 "Critical — 완전 수정됨"으로 평가했으나, Codex 2차 검증 결과 **과대 평가**로 확인됨:

- **수정된 부분**: smoothing이 scissor 계산 전에 적용되도록 이동 → scissor 영역 안정화 ✅
- **효과가 제한적인 부분**: face_rect.x/y 오프셋은 FreqSep 마스크 내용에 영향 없음. `combined_mask`는 `computeROI()`에서 face mesh 랜드마크 기반으로 이미 생성 완료되며(`gpu_beauty_backend.cpp:1481`), Composite 셰이더는 `uSkinMask`를 `vTexCoord` UV로 직접 샘플링(`shader_sources.cpp:480`). 마스크 지터링은 face mesh 랜드마크 안정성에 의존하며, face_rect 이동으로는 해결되지 않음.
- **정정된 심각도**: Critical → **Medium** (scissor 안정화는 유효하나, 피부 마스크 플리커 해결이라는 원래 목표에는 미치지 못함)

---

## Findings by Priority

### Critical Issues (P0 — 즉시 수정 필요)

| # | 출처 | 이슈 | 상태 |
|---|------|------|------|
| D1 | Phase 4 | CI/CD 파이프라인 완전 부재 | ❌ 미해결 |
| D2 | Phase 4 | P4-W3-04 신규 기능 전용 테스트 없음 | ❌ 미해결 |
| T1 | Phase 3 | OneEuroFilter/DeviceTier/HalfRes 전용 테스트 0건 | ❌ 미해결 |

### High Priority (P1 — 다음 릴리즈 전 수정)

| # | 출처 | 이슈 | 상태 |
|---|------|------|------|
| Q2/A2 | Phase 1 | 파이프라인 코드 ~140줄 중복 | 📋 P4-W3-04-R1 문서로 추적 |
| Q3 | Phase 1 | OneEuro Filter 리셋 누락 | ✅ **수정됨** (b80377f) |
| Q5/S1 | Phase 1,2 | std::stoi 예외 미처리 | ✅ **수정됨** (b80377f — strtol로 교체) |
| Q6/S2 | Phase 1,2 | detectDeviceTier() static public + GL 의존 | ✅ **수정됨** (b80377f — private 인스턴스 메서드로 변경) |
| F3 | Phase 4 | OneEuroFilter 타임스탬프 불일치 (3필터 각각 별도 now() 호출) | ❌ 미해결 |
| P2 | Phase 1 | Mask center smoothing scissor 타이밍 | ⚠️ **부분 수정** (scissor 안정화만 유효, 마스크 내용 무영향 — 상세: Executive Summary) |
| T2 | Phase 3 | OneEuroFilter 단위 테스트 없음 | ❌ 미해결 |
| T3 | Phase 3 | detectDeviceTier() 테스트 불가 구조 (GL 의존) | ❌ 미해결 |
| T4 | Phase 3 | DeviceTier 분기 통합 테스트 부재 | ❌ 미해결 |
| D3 | Phase 4 | TFLite 의존성 캐시 전략 없음 | ❌ 미해결 |
| D4 | Phase 4 | SDK 릴리즈 프로세스 미정의 | ❌ 미해결 |
| D5 | Phase 4 | ABI 버전 관리 메커니즘 없음 | ❌ 미해결 |
| D6 | Phase 4 | 재현 가능한 빌드 환경 없음 | ❌ 미해결 |
| D7 | Phase 4 | GPU 티어별 헤드리스 시뮬레이션 없음 | ❌ 미해결 |
| D1-Doc | Phase 3 | 작업 문서 §3.2 mask center 구현 불일치 | ❌ 미해결 |
| P1-Perf | Phase 2 | 정적 DeviceTier — 열 스로틀링 미대응 | 📋 P4-W3-05에서 처리 예정 |

### Medium Priority (P2 — 다음 스프린트 계획)

| # | 출처 | 이슈 | 상태 |
|---|------|------|------|
| Q4/A5/C1 | Phase 1,4 | GL_LINEAR 텍스처 필터 미복원 | TexturePool 기본값 GL_LINEAR 확인(`texture_pool.cpp:331`) → **버그 아님**, 정책/비용 이슈 |
| Q7 | Phase 1 | applyTextureId 중첩 깊이 4단계 | R1 리팩터링에서 자연 해소 예정 |
| Q8 | Phase 1 | Mali 분류 기준값 비대칭 | ❌ 주석 추가 필요 |
| S3 | Phase 2 | OneEuro Filter release() 미리셋 | ✅ **수정됨** (b80377f — release() 및 추적 끊김 분기에서 reset 추가) |
| S6 | Phase 2 | Viewport 복원 RAII 미적용 | ❌ 미해결 |
| S7 | Phase 2 | OneEuroFilter thread safety 주석 부재 | ❌ 미해결 |
| F5 | Phase 4 | computeGaussianWeights C 스타일 배열 | ❌ 미해결 |
| F6 | Phase 4 | DeviceTier enum public 노출 불필요 | ❌ 미해결 |
| F7 | Phase 4 | Temporal filter 리셋 로직 분산 | ❌ 미해결 |
| F9 | Phase 4 | LOG 매크로 dangling-else 취약 | ❌ 미해결 |
| D2-Doc | Phase 3 | DeviceTier enum 파이프라인 동작 미문서화 | ❌ 미해결 |
| D3-Doc | Phase 3 | GPUBeautyBackend 클래스 Doxygen 미갱신 | ❌ 미해결 |
| D4-Doc | Phase 3 | OneEuro Filter 파라미터 선택 근거 부재 | ❌ 미해결 |
| D8 | Phase 4 | 프로덕션 GPU 성능 가시성 없음 | ❌ 미해결 |
| T7 | Phase 3 | Half-Res 경계 테스트 부재 | ❌ 미해결 |

### Low Priority (P3 — 백로그 추적)

| # | 출처 | 이슈 |
|---|------|------|
| Q9 | Phase 1 | 매직 넘버 28 → constexpr kMaxGaussianRadius |
| Q10 | Phase 1 | DeviceTier public 노출 (Codex: 근거 약함) |
| S8 | Phase 2 | GPU 렌더러 파싱 휴리스틱 한계 (Mali-G78 MID 분류) |
| S9 | Phase 2 | half_w/half_h 홀수 해상도 오프셋 |
| S10 | Phase 2 | LOGD 매크로 do-while 미적용 |
| F10 | Phase 4 | static_cast<unsigned char> 반복 |
| F11 | Phase 4 | M_PI 비표준 사용 |
| D5-Doc | Phase 3 | Mali 분류 비대칭 근거 미기록 |
| D9-Doc | Phase 3 | executeFreqSepPipelineHalfRes() 상세 주석 부족 |
| D10-Doc | Phase 3 | CHANGELOG 미갱신 |
| D11 | Phase 4 | SDK 릴리즈 롤백 절차 문서 없음 |
| D12 | Phase 4 | git-workflow 문서 브랜치 미갱신 |

---

## Findings by Category

| 카테고리 | 총 건수 | Critical | High | Medium | Low |
|---------|--------|----------|------|--------|-----|
| Code Quality | 10 | 0 | 3 (2 수정됨) | 5+1(P2 하향) | 2 |
| Architecture | 5 | 0 | 2 | 3 | 0 |
| Security | 10 | 0 | 2 (수정됨) | 5 (1 수정됨) | 3 |
| Performance | 8 | 0 | 1 | 3 | 4 |
| Testing | 9 | 2 | 4 | 3 | 0 |
| Documentation | 12 | 1 | 3 | 4 | 4 |
| Best Practices | 14 | 0* | 4 | 6* | 4 |
| CI/CD & DevOps | 12 | 2 | 5 | 3 | 2 |
| **합계** | **80** | **5** | **24** | **33** | **19** |

*C1(GL_LINEAR 미복원)은 TexturePool 기본값 확인으로 **버그 아님** → Medium으로 하향
*P2(mask center smoothing)는 scissor 안정화만 유효, 마스크 내용 무영향 → Critical에서 **Medium**으로 하향

### 수정 현황

- **수정 완료**: 4건 (Q3, Q5/S1, Q6/S2, S3) — 커밋 b80377f
- **부분 수정**: 1건 (P2 — scissor 타이밍만 수정, 마스크 안정화 효과 제한적)
- **추적 중**: 2건 (Q2/A2 → P4-W3-04-R1, P1-Perf → P4-W3-05)
- **미해결**: 73건
- **과대 평가 → 하향**: 2건 (P2 Critical→Medium, C1 Critical→Medium)

### 중복 제거 후 실질 미해결 이슈

여러 Phase에서 동일 이슈가 반복 발견됨. 중복 제거 시:

| 실질 이슈 | 관련 Finding |
|----------|-------------|
| 테스트 부재 | T1, T2, T3, T4, T5, T6, D2 |
| CI/CD 없음 | D1 |
| 파이프라인 중복 | Q2, A2, F1 |
| 타임스탬프 불일치 | F3 |
| 마스크 안정화 효과 제한 | P2 (scissor만 유효, 마스크 UV 무영향) |
| 문서 불일치/부재 | D1-Doc, D2-Doc, D3-Doc, D4-Doc |
| 릴리즈 프로세스 | D4, D5 |

---

## Recommended Action Plan

> Codex Validity Check (`06-codex-validity-check.md`) 우선순위 제안을 반영하여 재정렬됨.

### 즉시 (이번 스프린트)

1. **테스트 보강 — OneEuroFilter + detectDeviceTier 헤드리스 테스트 작성** [Small]
   - `classifyGpuRenderer()` 정적 함수 추출 → GL 컨텍스트 없이 단위 테스트 가능
   - OneEuroFilter 수렴성, 리셋, 파라미터 민감도 테스트
   - 관련 Finding: T1~T6, D2

2. **문서 정합성 수정 — 작업 문서 §3.2 mask center 구현 내용** [Small]
   - "UV 좌표 오프셋" → "face_rect.x/y 직접 수정 (scissor 안정화 용도)" 반영
   - 마스크 내용 자체는 face mesh 랜드마크에 의존, face_rect 이동으로는 마스크 플리커 미해결임을 명시
   - 관련 Finding: D1-Doc, P2

3. **OneEuroFilter 타임스탬프 통일** [Small]
   - `frame_ts` 변수로 동일 프레임 내 3개 필터(cx, cy, radius) 동기화
   - 관련 Finding: F3

### 다음 스프린트

4. **P4-W3-04-R1 파이프라인 리팩터링 실행** [Medium]
   - `executeFreqSepPipelineImpl()` 공통 함수 추출
   - ~140줄 중복 제거
   - Q9 매직 넘버, Q7 중첩 깊이 동시 해소
   - 관련 Finding: Q2, A2, F1, Q7, Q9

5. **GitHub Actions CI 워크플로우 구축** [Medium]
   - 최소: build + ctest
   - TFLite 캐시 전략 포함
   - 관련 Finding: D1, D3

6. **DeviceTier 문서화 보강** [Small]
   - enum 값별 파이프라인 분기 동작 Doxygen
   - OEF 파라미터 선택 근거 주석
   - 관련 Finding: D2-Doc, D3-Doc, D4-Doc

### 이후 계획

7. **SDK 버전 관리 체계 도입** [Medium]
   - `IRIS_SDK_API_VERSION`, `IRIS_SDK_ABI_VERSION` 매크로
   - 관련 Finding: D4, D5

8. **P4-W3-05 런타임 적응형 tier 전환** [Large]
   - 열 스로틀링 대응, 동적 tier 강등
   - 관련 Finding: P1-Perf

9. **빌드 환경 Dockerfile + 실기기 테스트 자동화** [Large]
   - 재현 가능한 빌드 환경
   - Firebase Test Lab / AWS Device Farm 연동
   - 관련 Finding: D6, D7, D10

---

## Positive Observations

P4-W3-04에서 잘 구현된 부분:

- **TexturePool acquire/release** 패턴 일관성 양호, RAII 정리 올바름
- **Profiler 통합** 정확 (`_Half` 접미사로 tier 구분)
- **Gaussian weights CPU 사전 계산** + 셰이더 전환 최소화
- **Bilinear 하드웨어 보간** 활용으로 별도 업샘플링 패스 불필요
- **Bilateral fallback 안전망** (LOW tier graceful degradation)
- **detectDeviceTier() 1회 호출 + 캐싱** (초기화 시에만 실행)
- **Mutex 일관성** (모든 public 메서드)
- **Copy/Move 삭제** (Rule of Five)
- **GL 상태 관리**: Scissor, Viewport, PixelStorei 복원 올바름
- **MID tier 성능**: 대역폭 -54.6%, GPU 메모리 -75% 절감

---

## Review Metadata

- Review date: 2026-03-04
- Phases completed: 1A, 1B, 2A, 2B, 3A, 3B, 4A, 4B, 5
- Flags applied: performance_critical
- Codex 1차 feedback: 반영 완료 (3건 수정, 1건 R1 문서 추적)
- Codex 2차 feedback: P2 심각도 하향 (Critical→Medium), C1 하향 (Critical→Medium), T1 표현 정정
- Codex Validity Check: `.full-review/06-codex-validity-check.md` — 우선순위 재정렬 반영
- Framework: C++17 / OpenGL ES 3.1
