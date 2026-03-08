# Phase 4: Best Practices & Standards

## Framework & Language Findings

### Critical (1건)

| # | 이슈 | 위치 | 설명 |
|---|------|------|------|
| C1 | Half-Res 텍스처 glTexParameteri 사이드 이펙트 | `gpu_beauty_backend.cpp:1322-1330` | TexturePool 텍스처에 GL_LINEAR 설정 후 원복 없이 풀 반환. full-res 경로는 동일 작업 미수행으로 비대칭. Phase 2에서 TexturePool 기본값이 GL_LINEAR로 확인되어 **실제 버그 아님**이나, 명시적 계약 부재로 잠재적 리스크 |

> **참고**: Phase 2 성능 리뷰에서 TexturePool 기본값이 GL_LINEAR임을 확인(P2 결론). 따라서 C1은 실제 심각도 **Medium**으로 하향 조정 가능.

### High (4건)

| # | 이슈 | 설명 |
|---|------|------|
| F1 | 파이프라인 코드 ~140줄 중복 (Q2 재확인) | `executeFreqSepPipeline` vs `executeFreqSepPipelineHalfRes` 구조 동일. P4-W3-04-R1 리팩터링 문서로 추적 중 |
| F2 | detectDeviceTier() 문자열 파싱 한계 | `std::string_view` 미사용, Adreno/Mali 파싱 패턴 불일치, "Qualcomm Adreno(TM) 740" 변형 미대응 |
| F3 | OneEuroFilter 타임스탬프 불일치 | 동일 프레임 내 3개 필터(cx, cy, radius)가 각각 별도 `steady_clock::now()` 호출. 동일 `frame_ts`를 공유해야 함 |
| F4 | LowPassFilter 멤버 초기화 순서 | `computeAlpha(d_cutoff)` 인자가 파라미터를 직접 사용해 현재 안전하나, 멤버 참조로 변경 시 UB 위험. in-class member initializer 권장 |

### Medium (5건)

| # | 이슈 | 설명 |
|---|------|------|
| F5 | `computeGaussianWeights` C 스타일 배열 파라미터 | `float weights[29]` → `std::array<float, 29>&`로 타입 안전성 확보 |
| F6 | DeviceTier enum public 노출 불필요 | detectDeviceTier() private 전환 완료(Q6 fix)되었으나 enum은 여전히 public. getter 없으면 private으로 이동 |
| F7 | Temporal filter 리셋 타이밍 로직 분산 | `skin_radius_filter_` 리셋이 roi_ptr 무효 분기와 freq_sep_params 분기 두 곳에 산재 |
| F8 | `executeCombinedColorPass` width/height 미사용 파라미터 | 시그니처 일관성 위해 유지되나 `[[maybe_unused]]` 표기 권장 |
| F9 | LOG 매크로 dangling-else 취약 | `do { ... } while(0)` 래핑 필요 (Phase 2 S10 재확인) |

### Low (4건)

| # | 이슈 | 설명 |
|---|------|------|
| F10 | `static_cast<unsigned char>` 반복 | isDigit 람다 추출로 중복 제거 가능 |
| F11 | `M_PI` 비표준 사용 | `constexpr float kPi` 또는 C++20 `std::numbers::pi_v<float>` |
| F12 | `executeFreqSepPipelineHalfRes` #else 스타일 불일치 | `(void)` 억제 패턴이 full-res 버전과 미세하게 다름 |
| F13 | FBO completeness 체크 Debug 전용 | Production에서도 LOGW 레벨로 활성화 권장 |

### OpenGL ES 3.1 패턴 준수 현황

| 항목 | 상태 |
|------|------|
| VAO 사용 (ES 3.0+) | ✅ 준수 |
| FBO completeness 체크 | ⚠️ Debug 전용 |
| glFenceSync/glClientWaitSync | ✅ 준수 |
| GL_TEXTURE_3D neutral LUT | ✅ 준수 |
| glPixelStorei 원복 | ✅ 준수 |
| GL_SCISSOR_TEST 정리 | ✅ 준수 |
| glViewport HalfRes→full 복원 | ✅ 준수 |

---

## CI/CD & DevOps Findings

### Critical (2건)

| # | 이슈 | 운영 리스크 | 설명 |
|---|------|-----------|------|
| D1 | CI/CD 파이프라인 완전 부재 | PR 병합 전 빌드/테스트 자동 차단 게이트 없음 | `.github/workflows/` 없음. 모든 빌드/테스트는 수동 로컬 실행 |
| D2 | P4-W3-04 테스트 게이트 없음 | 318줄 추가, GPU 테스트 0건 | OneEuroFilter, detectDeviceTier 파싱은 헤드리스 테스트 가능하나 미작성 |

### High (4건)

| # | 이슈 | 운영 리스크 | 설명 |
|---|------|-----------|------|
| D3 | TFLite 의존성 캐시 전략 없음 | CI 도입 시 빌드 30분/PR | FetchContent ~400MB 매번 재다운로드 |
| D4 | SDK 릴리즈 프로세스 미정의 | 버전 추적, 핫픽스 복구 불가 | AAR/xcframework 패키징→배포 자동화 없음 |
| D5 | ABI 버전 관리 메커니즘 없음 | 외부 배포 시 데이터 손상 위험 | `sdk_api.h`에 `IRIS_SDK_API_VERSION` 매크로 없음 |
| D6 | 재현 가능한 빌드 환경 없음 | 로컬/CI 결과 불일치 | Dockerfile/devcontainer 없음, 의존성 버전 미고정 |
| D7 | GPU 티어별 헤드리스 시뮬레이션 없음 | CI에서 HIGH/MID/LOW 분기 테스트 불가 | `setDeviceTierForTesting()` API 없음 |

### Medium (3건)

| # | 이슈 | 설명 |
|---|------|------|
| D8 | 프로덕션 GPU 성능 가시성 없음 | GPUProfiler 데이터 수집/집계/경보 인프라 없음 |
| D9 | 빌드 디렉토리 상태 의존 | `cmake-build-debug` 재사용 패턴이 CI 환경과 충돌 |
| D10 | 실기기 테스트 매트릭스 수동 프로세스 | Firebase Test Lab/AWS Device Farm 연동 없음 |

### Low (2건)

| # | 이슈 | 설명 |
|---|------|------|
| D11 | SDK 릴리즈 롤백 절차 문서 없음 | INCIDENT_RESPONSE.md 필요 |
| D12 | git-workflow 문서 P4-W3-02 기준 고정 | 브랜치 메타데이터 미갱신 |

### 즉시 조치 우선순위

| 우선순위 | 항목 | 노력 |
|---------|------|------|
| P1 | GitHub Actions CI 워크플로우 생성 | 4시간 |
| P1 | detectDeviceTier() 파싱 + OneEuroFilter 헤드리스 테스트 | 4시간 |
| P2 | setDeviceTierForTesting() API + IRIS_SDK_API_VERSION 매크로 | 3.5시간 |
| P3 | 빌드 환경 Dockerfile + 성능 기준선 갱신 | 4시간 |
| P4 | Firebase Test Lab 연동 | 1~2주 |
