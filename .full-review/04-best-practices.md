# Phase 4: Best Practices & Standards

## Framework & Language Findings (C++17)

### Critical (1건)
| ID | 위치 | 요약 |
|----|------|------|
| BP-1 | 전체 .cpp | catch(...) 예외 삼킴 — 구체적 예외 타입 분리 + 로깅 필요 |

### High (4건)
| ID | 위치 | 요약 |
|----|------|------|
| BP-4 | quality_metrics.cpp:411 | vector::erase(begin()) → std::deque 또는 circular buffer |
| BP-7 | release_gate.h, gpu_beauty_backend.h | DeviceTier enum 중복 → 공통 헤더 추출 |
| BP-8 | ab_compare.cpp | JSON 수동 조립 이스케이핑 없음 |
| BP-11 | 전체 공개 헤더 | Pimpl 미적용 (프로젝트 컨벤션 위반) |

### Medium (6건)
| ID | 위치 | 요약 |
|----|------|------|
| BP-2 | param_tuner.cpp | std::max/min 체인 → std::clamp 통일 |
| BP-3 | param_tuner.cpp, release_gate.cpp | snprintf + 하드코딩 버퍼 혼용 |
| BP-5 | quality_metrics.h | All-static class → namespace 함수 고려 |
| BP-6 | 전체 | 매직 넘버 중복 → inline constexpr 공통 상수 헤더 |
| BP-10 | param_tuner.h | std::function ProcessCallback 오버헤드 |
| BP-14 | ab_compare.cpp 등 | noexcept + vector::push_back → bad_alloc 시 terminate 위험 |

### Low (7건)
- std::string_view 미사용, structured bindings 미사용, toString 중복, constexpr 활용 부족, using namespace in tests, if-with-initializer 미사용, range-based for 양호

## CI/CD & DevOps Findings

**제외** — 사용자 요청에 따라 CI/CD 검증을 건너뜁니다 (SDK 프로젝트, 미배포 상태).
