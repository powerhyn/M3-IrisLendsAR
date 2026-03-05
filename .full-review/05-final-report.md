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

P4-W3-04는 GPU 뷰티 파이프라인에 디바이스 적응형 성능 최적화를 도입했다. MID tier에서 대역폭 -54.6%, GPU 메모리 -75% 절감을 달성했으며, temporal filtering으로 시각적 안정성을 개선했다.

리뷰 후 4차에 걸친 피드백 반영으로 코드 품질 이슈가 대부분 해결되었다:
- **b80377f**: 실질적 결함 4건 수정 (OneEuro 리셋, strtol 교체, detectDeviceTier private 전환, release 리셋)
- **f8b96fb**: 13개 항목 일괄 반영 (타임스탬프 통일, enum 접근성, 매직 넘버, 주석/Doxygen 보강 등)
- **5c14979**: 테스트 35건 추가 (OneEuroFilter T2/T6, DeviceTier T3/T4/T5)
- **997b5c5**: FreqSep 파이프라인 중복 ~130줄 통합 (R1 리팩터링)

**현재 남은 코드 품질 이슈는 없으며**, 미해결 항목은 CI/CD 인프라 및 릴리즈 프로세스 영역에 집중되어 있다.

### P2 정정 사항 (Codex 2차 피드백 반영)

초기 리뷰에서 P2를 "Critical — 완전 수정됨"으로 평가했으나, Codex 2차 검증 결과 **과대 평가**로 확인됨:

- **수정된 부분**: smoothing이 scissor 계산 전에 적용되도록 이동 → scissor 영역 안정화 ✅
- **효과가 제한적인 부분**: face_rect.x/y 오프셋은 FreqSep 마스크 내용에 영향 없음. `combined_mask`는 `computeROI()`에서 face mesh 랜드마크 기반으로 이미 생성 완료되며(`gpu_beauty_backend.cpp:1481`), Composite 셰이더는 `uSkinMask`를 `vTexCoord` UV로 직접 샘플링(`shader_sources.cpp:480`). 마스크 지터링은 face mesh 랜드마크 안정성에 의존하며, face_rect 이동으로는 해결되지 않음.
- **정정된 심각도**: Critical → **Medium** (scissor 안정화는 유효하나, 피부 마스크 플리커 해결이라는 원래 목표에는 미치지 못함)
- **후속 추적**: TODO(P4-W3-04-R2)로 코드에 기록. 실기기 테스트(P4-W3-05)에서 체감 여부 확인 후 판단.

---

## Findings by Priority

### Critical Issues (P0 — 즉시 수정 필요)

| # | 출처 | 이슈 | 상태 |
|---|------|------|------|
| D1 | Phase 4 | CI/CD 파이프라인 완전 부재 | 🔧 CI/CD 인프라 별도 작업 |
| D2 | Phase 4 | P4-W3-04 신규 기능 전용 테스트 없음 | ✅ **해결** (5c14979 — 35건 추가) |
| T1 | Phase 3 | OneEuroFilter/DeviceTier/HalfRes 전용 테스트 0건 | ✅ **해결** (5c14979 — T2~T6 테스트 추가) |

### High Priority (P1 — 다음 릴리즈 전 수정)

| # | 출처 | 이슈 | 상태 |
|---|------|------|------|
| Q2/A2 | Phase 1 | 파이프라인 코드 ~140줄 중복 | ✅ **해결** (997b5c5 — executeFreqSepPipelineImpl 통합) |
| Q3 | Phase 1 | OneEuro Filter 리셋 누락 | ✅ **해결** (b80377f) |
| Q5/S1 | Phase 1,2 | std::stoi 예외 미처리 | ✅ **해결** (b80377f — strtol로 교체) |
| Q6/S2 | Phase 1,2 | detectDeviceTier() static public + GL 의존 | ✅ **해결** (b80377f — private 인스턴스 메서드로 변경) |
| F3 | Phase 4 | OneEuroFilter 타임스탬프 불일치 | ✅ **해결** (f8b96fb — frame_ts 캡처 후 3필터 동기화) |
| P2 | Phase 1 | Mask center smoothing scissor 타이밍 | ⚠️ **부분 수정** (scissor 안정화만 유효, 마스크 내용 무영향 — 상세: Executive Summary) |
| T2 | Phase 3 | OneEuroFilter 단위 테스트 없음 | ✅ **해결** (5c14979 — 수렴/리셋/파라미터 테스트) |
| T3 | Phase 3 | detectDeviceTier() 테스트 불가 구조 (GL 의존) | ✅ **해결** (5c14979 — classifyGpuRenderer() static 분리, GL 없이 테스트) |
| T4 | Phase 3 | DeviceTier 분기 통합 테스트 부재 | ✅ **해결** (5c14979 — tier별 파이프라인 분기 테스트) |
| D3 | Phase 4 | TFLite 의존성 캐시 전략 없음 | 🔧 CI/CD 인프라 별도 작업 |
| D4 | Phase 4 | SDK 릴리즈 프로세스 미정의 | 🔧 CI/CD 인프라 별도 작업 |
| D5 | Phase 4 | ABI 버전 관리 메커니즘 없음 | 🔧 CI/CD 인프라 별도 작업 |
| D6 | Phase 4 | 재현 가능한 빌드 환경 없음 | 🔧 CI/CD 인프라 별도 작업 |
| D7 | Phase 4 | GPU 티어별 헤드리스 시뮬레이션 없음 | 🔧 CI/CD 인프라 별도 작업 |
| D1-Doc | Phase 3 | 작업 문서 §3.2 mask center 구현 불일치 | ✅ **해결** (f8b96fb — 문서 정합성 수정) |
| P1-Perf | Phase 2 | 정적 DeviceTier — 열 스로틀링 미대응 | 📋 P4-W3-05에서 처리 예정 |

### Medium Priority (P2 — 다음 스프린트 계획)

| # | 출처 | 이슈 | 상태 |
|---|------|------|------|
| Q4/A5/C1 | Phase 1,4 | GL_LINEAR 텍스처 필터 중복 설정 | 🟢 **이슈 아님** — TexturePool 기본값 GL_LINEAR(`texture_pool.cpp:331`), 방어적 코드로 유지 타당. 성능 영향 0. |
| Q7 | Phase 1 | applyTextureId 중첩 깊이 4단계 | ✅ **해결** (997b5c5 — R1 리팩터링에서 파이프라인 추출로 해소) |
| Q8 | Phase 1 | Mali 분류 기준값 비대칭 | ✅ **해결** (f8b96fb — Valhall 아키텍처 전환 근거 주석 추가) |
| S3 | Phase 2 | OneEuro Filter release() 미리셋 | ✅ **해결** (b80377f — release() 및 추적 끊김 분기에서 reset 추가) |
| S6 | Phase 2 | Viewport 복원 RAII 미적용 | 🟢 **이슈 아님** — 에러 경로(텍스처 획득 실패)가 viewport 변경 전에 반환되므로 복원 누락 불가. RAII guard는 과잉 엔지니어링. |
| S7 | Phase 2 | OneEuroFilter thread safety 주석 부재 | ✅ **해결** (f8b96fb — thread safety 주석 추가) |
| F5 | Phase 4 | computeGaussianWeights C 스타일 배열 | 🟢 **수정 불필요** — 스택 할당(116 bytes), constexpr 크기, std::clamp 경계 보호 완비. 스타일 선호 수준. |
| F6 | Phase 4 | DeviceTier enum public 노출 불필요 | ✅ **해결** (f8b96fb — private 이동) |
| F7 | Phase 4 | Temporal filter 리셋 로직 분산 | ✅ **해결** (f8b96fb — resetTemporalFilters() 헬퍼 추출) |
| F9 | Phase 4 | LOG 매크로 dangling-else 취약 | ✅ **해결** (f8b96fb — do{...}while(0) 래핑) |
| D2-Doc | Phase 3 | DeviceTier enum 파이프라인 동작 미문서화 | ✅ **해결** (f8b96fb — Doxygen 갱신) |
| D3-Doc | Phase 3 | GPUBeautyBackend 클래스 Doxygen 미갱신 | ✅ **해결** (f8b96fb — Doxygen 갱신) |
| D4-Doc | Phase 3 | OneEuro Filter 파라미터 선택 근거 부재 | ✅ **해결** (f8b96fb — 파라미터 근거 주석 추가) |
| D8 | Phase 4 | 프로덕션 GPU 성능 가시성 없음 | 🔧 CI/CD 인프라 별도 작업 |
| T7 | Phase 3 | Half-Res 경계 테스트 부재 | 🟢 **수정 불필요** — 실제 카메라 해상도는 항상 짝수(720p/1080p/4K), blur_w<1 가드 존재. 실제 발생 시나리오 없음. |

### Low Priority (P3 — 백로그 추적)

| # | 출처 | 이슈 | 상태 |
|---|------|------|------|
| Q9 | Phase 1 | 매직 넘버 28 → constexpr kMaxGaussianRadius | ✅ **해결** (f8b96fb) |
| Q10 | Phase 1 | DeviceTier public 노출 | ✅ **해결** (f8b96fb — private 이동) |
| S8 | Phase 2 | GPU 렌더러 파싱 휴리스틱 한계 | 📋 런타임 적응형 tier(P4-W3-05)에서 근본 해결 예정 |
| S9 | Phase 2 | half_w/half_h 홀수 해상도 오프셋 | 🟢 **이슈 아님** — 카메라 해상도는 항상 짝수, 저주파 bilinear 보간에서 1px 무의미 |
| S10 | Phase 2 | LOGD 매크로 do-while 미적용 | ✅ **해결** (f8b96fb) |
| F10 | Phase 4 | static_cast\<unsigned char\> 반복 | 🟢 **수정 불필요** — 2회 사용(Adreno/Mali 파서), 헬퍼 추출은 과잉 |
| F11 | Phase 4 | M_PI 비표준 사용 | ✅ **해결** (f8b96fb — constexpr kPi) |
| D5-Doc | Phase 3 | Mali 분류 비대칭 근거 미기록 | ✅ **해결** (f8b96fb — 주석 추가) |
| D9-Doc | Phase 3 | executeFreqSepPipelineHalfRes() 상세 주석 부족 | ✅ **해결** (997b5c5 — Impl 통합 + FreqSepExecConfig 구조체로 자기 문서화) |
| D10-Doc | Phase 3 | CHANGELOG 미갱신 | 🟢 **해당 없음** — 프로젝트에 CHANGELOG 파일 없음 (향후 도입 시 작성) |
| D11 | Phase 4 | SDK 릴리즈 롤백 절차 문서 없음 | 🔧 CI/CD 인프라 별도 작업 |
| D12 | Phase 4 | git-workflow 문서 브랜치 미갱신 | 🔧 CI/CD 인프라 별도 작업 |

---

## Findings by Category (최종)

| 카테고리 | 총 건수 | 해결 | 이슈 아님 | CI/CD 별도 | 미해결 |
|---------|--------|------|----------|-----------|--------|
| Code Quality | 10 | 8 | 2 | 0 | 0 |
| Architecture | 5 | 3 | 0 | 0 | 2 (P2 부분수정, P1-Perf→P4-W3-05) |
| Security | 10 | 7 | 3 | 0 | 0 |
| Performance | 8 | 3 | 3 | 1 | 1 (P1-Perf→P4-W3-05) |
| Testing | 9 | 6 | 1 | 1 | 1 (P2 마스크 안정화 — 실기기 확인 후 판단) |
| Documentation | 12 | 8 | 1 | 1 | 2 (D11, D12 → CI/CD) |
| Best Practices | 14 | 10 | 2 | 0 | 2 (S8→P4-W3-05, P2) |
| CI/CD & DevOps | 12 | 0 | 0 | 12 | 0 |
| **합계** | **80** | **45** | **12** | **15** | **8** |

### 수정 현황 (최종)

- **해결 완료**: 45건 — 커밋 b80377f, f8b96fb, 5c14979, 997b5c5
- **이슈 아님 (재검증)**: 12건 — 코드 확인 결과 실제 문제 아닌 것으로 판정
- **CI/CD 인프라**: 15건 — 별도 인프라 구축 작업으로 분리
- **부분 수정/추적**: 3건 — P2(마스크 안정화, 실기기 확인 후), P1-Perf(P4-W3-05), S8(P4-W3-05)
- **미해결 (CI/CD 제외)**: 0건

### 중복 제거 후 실질 미해결 이슈

| 실질 이슈 | 관련 Finding | 상태 |
|----------|-------------|------|
| ~~테스트 부재~~ | T1, T2, T3, T4, D2 | ✅ **해결** (35건 추가) |
| CI/CD 없음 | D1, D3, D4, D5, D6, D7, D8, D11, D12 | 🔧 별도 인프라 작업 |
| ~~파이프라인 중복~~ | Q2, A2 | ✅ **해결** (R1 리팩터링) |
| ~~타임스탬프 불일치~~ | F3 | ✅ **해결** (frame_ts 동기화) |
| 마스크 안정화 효과 제한 | P2 | ⚠️ 부분 수정 — P4-W3-05 실기기 테스트에서 체감 확인 후 판단 |
| 열 스로틀링 미대응 | P1-Perf, S8 | 📋 P4-W3-05 런타임 적응형 tier에서 해결 예정 |

---

## Recommended Action Plan (갱신)

### ✅ 완료된 작업

1. ~~**테스트 보강 — OneEuroFilter + detectDeviceTier 테스트**~~ → 5c14979 (35건)
2. ~~**문서 정합성 수정 — 작업 문서 §3.2 mask center**~~ → f8b96fb
3. ~~**OneEuroFilter 타임스탬프 통일**~~ → f8b96fb (frame_ts)
4. ~~**P4-W3-04-R1 파이프라인 리팩터링**~~ → 997b5c5 (~130줄 감소)
5. ~~**DeviceTier 문서화 보강**~~ → f8b96fb (Doxygen + 주석)

### 남은 작업 (CI/CD 인프라)

6. **GitHub Actions CI 워크플로우 구축** [Medium]
   - 최소: build + ctest
   - TFLite 캐시 전략 포함
   - 관련 Finding: D1, D3

7. **SDK 버전 관리 체계 도입** [Medium]
   - `IRIS_SDK_API_VERSION`, `IRIS_SDK_ABI_VERSION` 매크로
   - 관련 Finding: D4, D5

8. **빌드 환경 Dockerfile + 실기기 테스트 자동화** [Large]
   - 재현 가능한 빌드 환경
   - Firebase Test Lab / AWS Device Farm 연동
   - 관련 Finding: D6, D7

### 다음 기능 작업

9. **P4-W3-05 튜닝 + 테스트 + 릴리즈 게이트** [Large]
   - 런타임 적응형 tier 전환 (열 스로틀링 대응)
   - 실기기 테스트에서 마스크 플리커 체감 여부 확인
   - 관련 Finding: P1-Perf, S8, P2

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
- **FreqSepExecConfig**: R1 리팩터링으로 full/half-res 분기를 매개변수화하여 ~130줄 중복 제거

---

## Review Metadata

- Review date: 2026-03-04
- **최종 갱신**: 2026-03-05 (해결 현황 반영)
- Phases completed: 1A, 1B, 2A, 2B, 3A, 3B, 4A, 4B, 5
- Flags applied: performance_critical
- 후속 커밋 반영:
  - b80377f: Codex 1차 피드백 (4건 수정)
  - f8b96fb: Comprehensive review 13개 항목 일괄 반영
  - 5c14979: 테스트 35건 추가 (T1~T6)
  - 997b5c5: FreqSep 파이프라인 R1 리팩터링
- Codex 2차 feedback: P2 심각도 하향 (Critical→Medium), C1 하향 (Critical→Medium), T1 표현 정정
- Codex Validity Check: `.full-review/06-codex-validity-check.md` — 우선순위 재정렬 반영
- **재검증 결과**: S6(Viewport RAII), F5(C배열), T7(경계테스트), Q4(GL_LINEAR), S9(홀수해상도) — 코드 확인 결과 실질적 이슈 아님으로 판정
- Framework: C++17 / OpenGL ES 3.1
