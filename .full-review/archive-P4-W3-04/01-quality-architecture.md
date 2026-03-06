# Phase 1: Code Quality & Architecture Review

## Code Quality Findings

### Critical (1건)

| # | 이슈 | 위치 | 설명 |
|---|------|------|------|
| Q1 | ROI 포인터 직접 수정 | `gpu_beauty_backend.cpp:1622-1624` | `roi_ptr->face_rect.x/y`를 직접 변경하여 같은 메서드 내 후속 코드가 원본 face_rect를 기대하면 예상치 못한 동작 발생. 로컬 복사본을 사용해야 함 |

### High (2건)

| # | 이슈 | 위치 | 설명 |
|---|------|------|------|
| Q2 | 파이프라인 코드 ~140줄 중복 | `gpu_beauty_backend.cpp:1216-1362` vs `1024-1149` | `executeFreqSepPipelineHalfRes`와 `executeFreqSepPipeline`이 구조적으로 거의 동일. 해상도 매개변수만 다름. 공통 `executeFreqSepPipelineImpl()`로 추출 권장 |
| Q3 | One Euro Filter 리셋 누락 | `gpu_beauty_backend.h:446-448` | 얼굴 추적 끊김→재획득 시 필터에 이전 상태가 남아 비정상적 전환 발생. `roi_ptr` 무효 시 `filter.reset()` 호출 필요 |

### Medium (5건)

| # | 이슈 | 위치 | 설명 |
|---|------|------|------|
| Q4 | GL_LINEAR 텍스처 필터 미복원 | `gpu_beauty_backend.cpp:1296-1305` | half-res 텍스처에 GL_LINEAR 설정 후 풀 반환 시 원래 필터 모드 미복원 |
| Q5 | std::stoi 예외 미처리 | `gpu_beauty_backend.cpp:1172, 1186` | 비정상 GPU 문자열에서 std::out_of_range 크래시 가능 |
| Q6 | static 메서드의 GL 컨텍스트 의존 | `gpu_beauty_backend.h:249` | detectDeviceTier()가 static이지만 glGetString() 호출. 테스트 불가, 오용 가능 |
| Q7 | applyTextureId 중첩 깊이 증가 | `gpu_beauty_backend.cpp:1645-1690` | tier 분기 추가로 4단계 중첩. 전략 선택 함수 추출 권장 |
| Q8 | Mali 분류 기준값 비대칭 | `gpu_beauty_backend.cpp:1183-1188` | 2자리/3자리 혼재, 향후 G800 시리즈 대응 주석 필요 |

### Low (2건)

| # | 이슈 | 위치 | 설명 |
|---|------|------|------|
| Q9 | 매직 넘버 28 | `gpu_beauty_backend.cpp:36` | `constexpr kMaxGaussianRadius = 28`로 명명 권장 |
| Q10 | DeviceTier public 노출 | `gpu_beauty_backend.h:242-248` | 내부 구현 세부사항이 불필요하게 public API에 노출 |

---

## Architecture Findings

### High (2건)

| # | 이슈 | 아키텍처 영향 | 설명 |
|---|------|-------------|------|
| A1 | API 표면 불완전 | High | DeviceTier enum은 public이지만 getter 없음. static 메서드의 GL 컨텍스트 의존이 계약에 미반영. 테스트용 오버라이드 불가 |
| A2 | Temporal filtering 관심사 분리 | High | `applyTextureId()` (200줄+ 오케스트레이션) 내에 temporal filtering이 인라인. `stabilizeFreqSepParams()`, `stabilizeFaceRect()` 헬퍼로 추출 권장 |

### Medium (3건)

| # | 이슈 | 아키텍처 영향 | 설명 |
|---|------|-------------|------|
| A3 | DeviceTier 배치 | Medium | 현재 GPUBeautyBackend에 적절하나, 다른 GPU 컴포넌트에서 사용 시 별도 헤더 분리 필요 |
| A4 | 티어 기반 분기 패턴 | Medium | 현재 3경로에서 적절. 5개+ 경로 시 Strategy 패턴 리팩터링 검토 |
| A5 | Half-res GL 상태 오염 | Medium | 텍스처 풀 반환 시 필터 상태 리셋 정책 확인 필요 |

### Positive (잘 된 부분)

- TexturePool acquire/release 패턴 일관성 양호
- Profiler begin/end 통합 정확 (`_Half` 접미사로 구분)
- RAII 패턴: 텍스처 획득 실패 시 이미 획득한 텍스처 정리
- `#if IRIS_SDK_GPU_AVAILABLE` 가드 일관 적용
- `computeGaussianWeights`의 radius 경계 보호 추가

---

## Critical Issues for Phase 2 Context

Phase 2 (Security & Performance) 리뷰에서 주목할 사항:

1. **성능**: `executeFreqSepPipelineHalfRes`에서 GL_LINEAR 텍스처 파라미터 변경이 프레임당 반복되면서 GPU 상태 전환 오버헤드 유발 가능
2. **성능**: `detectDeviceTier()`의 문자열 파싱이 `initialize()`에서만 호출되는지 확인 (반복 호출 시 성능 저하)
3. **안전성**: `std::stoi` 예외가 초기화 경로에서 발생하면 전체 SDK 초기화 실패
4. **메모리**: One Euro Filter 3개 인스턴스의 추가 메모리 영향 (무시 가능하나 확인)
