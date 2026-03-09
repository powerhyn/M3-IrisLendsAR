# Phase 2: Security & Performance Review

## Security Findings (15건)

### Critical (1건)
| ID | CWE | 위치 | 요약 |
|----|-----|------|------|
| SEC-01 | CWE-400 | ab_compare.cpp:155-157 | ABCompare::addResult() 무제한 메모리 성장 |

### High (4건)
| ID | CWE | 위치 | 요약 |
|----|-----|------|------|
| SEC-02 | CWE-755 | 12+ 위치 | catch(...) 보안 관련 실패 무시 (bad_alloc, cv::Exception 등) |
| SEC-03 | CWE-400 | param_tuner.cpp:32-102 | gridSearch() steps 상한 미검증 → CPU/메모리 고갈 |
| SEC-04 | CWE-407 | quality_metrics.cpp:411-413 | vector::erase(begin()) O(n) → 프레임 드롭 가능 |
| SEC-05 | CWE-116 | ab_compare.cpp:255-256 | JSON 문자열 이스케이핑 미처리 |

### Medium (6건)
| ID | CWE | 위치 | 요약 |
|----|-----|------|------|
| SEC-06 | CWE-190 | param_tuner.cpp:58-59 | 32비트 플랫폼 size_t 곱셈 오버플로우 |
| SEC-07 | CWE-1078 | quality_metrics.cpp, release_gate.cpp | 매직 넘버 중복 → 게이트 우회 위험 |
| SEC-08 | CWE-120 | param_tuner.cpp:268-275 | snprintf 버퍼 잘림 가능성 |
| SEC-09 | CWE-20 | quality_metrics.cpp:66-89 | cv::Mat 이미지 크기 상한 미검증 → OOM |
| SEC-10 | CWE-20 | release_gate.cpp:55-140 | NaN/Inf 입력 미검증 |
| SEC-11 | CWE-20 | param_tuner.cpp:377-381 | 음수 face_width 미처리 |

### Low (4건)
- SEC-12: DeviceTier enum 중복 (ODR 위험)
- SEC-13: generateReport() 전체 벡터 복사
- SEC-14: toGray() 얕은 복사 (참조 공유)
- SEC-15: 테스트 코드 고정 시드 RNG (영향 없음)

### 긍정적 보안 소견
- noexcept 정책 일관 적용, snprintf 사용, clamp 적용
- 온디바이스 처리 원칙 준수 (네트워크 전송 코드 없음)
- TemporalAnalyzer kMaxFrames=300 제한 적용

---

## Performance Findings (13건)

### Critical (2건)
| ID | 컴포넌트 | Frame-path | 예상 영향 |
|----|----------|------------|-----------|
| PF-1 | TemporalAnalyzer | Yes | vector::erase(begin()) O(n), 매 프레임 |
| PF-2 | TemporalAnalyzer | Yes | addFrame에서 매 프레임 Laplacian 연산 (~2-5ms/frame, 예산 6-15%) |

### High (3건)
| ID | 컴포넌트 | Frame-path | 예상 영향 |
|----|----------|------------|-----------|
| PF-3 | QualityMetrics | Indirect | SSIM 21개 전체 이미지 Mat 할당 (~336MB peak, 100MB 제한 초과) |
| PF-4 | QualityMetrics | Indirect | cv::split 채널 분리 (~12MB 추가/call) |
| PF-5 | ParamTuner | Offline | Grid search steps^4 조합 폭발 (steps=5에서 ~10초) |

### Medium (5건)
| ID | 컴포넌트 | 예상 영향 |
|----|----------|-----------|
| PF-6 | ParamTuner | generateReport 전체 벡터 복사+정렬 |
| PF-7 | QualityMetrics | 중복 toGray 4회 호출 (~4ms/call 낭비) |
| PF-8 | QualityMetrics | Sobel CV_64F 전체 이미지 (~48MB/call) |
| PF-9 | ABCompare | compare()에서 gate 2회 호출 (개선폭 2배 증폭) |
| PF-10 | QualityMetrics | Laplacian CV_64F (~16MB/call) |

### Low (2건)
- PF-11: TemporalAnalyzer thread safety (잠재적 data race)
- PF-12: ABCompare results_ 무제한 성장

### 예상 최적화 효과
| 메트릭 | Before | After (Phase 1+2) |
|--------|--------|-------------------|
| addFrame 비용 | ~2-5ms | ~0ms |
| SSIM 메모리 peak | ~336MB | ~0 추가 할당 |
| evaluateQuantitativeGate | ~8-15ms | ~4-7ms |
| Grid search (steps=5) | ~9.4s | ~2.4s (coarse-to-fine) |

---

## Critical Issues for Phase 3 Context

1. **테스트 커버리지**: PF-2(addReductionRatio 오버로드)와 ring buffer 교체 후 기존 테스트 수정 필요
2. **SSIM 구현 교체**: zero-allocation 버전의 정밀도 검증 테스트 필요
3. **gridSearch 상한 검증**: steps 제한에 대한 경계값 테스트 추가 필요
4. **문서화**: TemporalAnalyzer thread safety 정책 명시, 각 메서드의 성능 특성 문서화 필요
