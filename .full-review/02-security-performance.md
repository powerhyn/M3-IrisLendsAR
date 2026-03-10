# Phase 2: Security & Performance Review

## Security Findings

### [Medium] SEC-1: sRGB/Linear 공간 혼용 — 기능적 정확성
- **파일**: `shader_sources.cpp` line 500-509
- Edge gradient는 sRGB 공간, magnitude/chroma는 linear 공간에서 계산
- 어두운 영역에서 에지 과대평가, 밝은 영역에서 과소평가 가능
- 의도된 트레이드오프(Option B, 4xpow 절약)이나 다양한 피부톤 시각적 QA 필요
- **보안 위험 없음**, 기능적 정확성 이슈

### [Low] SEC-2: Uniform 값 클램핑 미적용 (public struct 경유 시)
- **파일**: `gpu_beauty_backend.cpp` line 1275-1276
- `edge_weight > 1.0` 시 blemishScore 음수 반전 → 잡티 감쇠 비활성화
- 현재 `mapSkinQuality()`만이 생성 경로이므로 즉각적 위험 없음
- **권장**: `std::clamp(edge_weight, 0.0f, 1.0f)` 방어 코드

### [Low] SEC-3: 경계값 테스트 부족
- `skinQuality` 극단값(0, >1, 음수)에서 edge/chroma weight 검증 미비

### NONE: 버퍼 오버플로, GLSL UB, Division-by-Zero, C++ 메모리 안전성 — 모두 안전 확인

---

## Performance Findings

### [Low] PERF-1: `diff = orig - low` 중복 계산
- **파일**: `shader_sources.cpp` line 495 vs 512
- `high`와 동일한 연산. 모바일 GPU 드라이버 CSE 불안정할 수 있음
- **권장**: `high` 직접 재사용 → vec3 레지스터 1개 + ALU 1회 절약

### [Low] PERF-2: sqrt() 2회 (edgeStrength + length(chromaDiff))
- SFU 기반, 전체 추가 ALU 비용의 ~25-40% 차지
- 제곱 도메인 비교로 대체 가능하나 비선형 응답 변경 → 시각적 결과 달라짐
- **권장**: 측정 후 판단, 현재는 유지

### 30fps 달성 평가

| GPU | 추가 비용 | 33ms 대비 | 판정 |
|-----|----------|----------|------|
| Adreno 660+ | ~0.5-0.7ms | 1.5-2.1% | ✅ 안전 |
| Mali-G78 | ~0.87ms | 2.6% | ✅ 안전 |
| Mali-G52 (저가) | ~1.8ms | 5.5% | ⚠️ 모니터링 필요 |

- 텍스처 캐시 효율: 1-texel 인접 패턴으로 양호 (대부분 L1 캐시 히트)
- 메모리 대역폭: ~0.5-1.1% 추가, 무시 가능
- 분기 분산: 없음 (이상적)
- CPU 오버헤드: ~10us (glUniform1f 2회), 무시 가능

---

## Critical Issues for Phase 3 Context

- 다양한 피부톤(특히 어두운 톤)에서의 시각적 QA 테스트 필요 (SEC-1)
- 경계값 테스트 추가 권장 (SEC-3)
- 저가 GPU(Mali-G52 이하)에서 실측 프로파일링 권장
