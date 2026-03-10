# Phase 2: Performance Review

## 리뷰 대상: P4-W4-01d 톤커브 미드톤 리프트 (2 commits)

---

## Performance Findings

### [Low] PERF-1: 셰이더 ALU 추가 비용 무시 가능

- **파일**: `shader_sources.cpp:552`
- **수식**: `beauty = beauty + uToneLift * beauty * (vec3(1.0) - beauty)`
- **ALU 분석**: `vec3(1.0) - beauty` (SUB×3) + `beauty * result` (MUL×3) + `uToneLift * result` (MUL×3) + `beauty + result` (ADD×3) = 12 scalar ALU ops
- 실제로는 MAD 최적화로 6-9 ops로 축소 가능
- **텍스처 페치**: 추가 없음 (기존 `beauty` 레지스터 재사용)
- **예상 비용**: Adreno 660: ~0.1ms, Mali-G78: ~0.15ms, Mali-G52: ~0.3ms
- **판정**: 33ms 프레임 버짓 대비 0.3-0.9% 추가. **안전**

### [Low] PERF-2: Uniform 업로드 추가 1회

- **파일**: `gpu_beauty_backend.cpp:1282`
- `glUniform1f(freq_sep_composite_uniforms_.uToneLift, params.tone_lift)` 1회 추가
- **비용**: ~1μs. 무시 가능

### [Low] PERF-3: `mapSkinQuality()` 분기 1개 추가

- **파일**: `gpu_beauty_backend.cpp:1026`
- 삼항 연산자 1회 + 곱셈 1회 추가. CPU 측 ~1ns. 무시 가능

### NONE: 메모리, 동시성, 스케일링 이슈 — 없음

- 새 텍스처 할당 없음
- 새 렌더 패스 없음
- 기존 mutex/fence 패턴에 영향 없음
- 해상도 의존 추가 비용 없음 (per-fragment ALU만)

---

## 30fps 달성 평가

| GPU | 추가 비용 | 33ms 대비 | 판정 |
|-----|----------|----------|------|
| Adreno 660+ | ~0.1ms | 0.3% | ✅ 안전 |
| Mali-G78 | ~0.15ms | 0.5% | ✅ 안전 |
| Mali-G52 (저가) | ~0.3ms | 0.9% | ✅ 안전 |

---

## 수치 안전성 검증

| 조건 | `beauty` 입력 | `uToneLift` | 결과 | 범위 내? |
|------|--------------|-------------|------|---------|
| Identity | 0.5 | 0.0 | 0.5 | ✅ |
| 미드톤 최대 | 0.5 | 0.15 | 0.5375 | ✅ |
| 하이라이트 | 0.9 | 0.15 | 0.9135 | ✅ |
| 섀도우 | 0.1 | 0.15 | 0.1135 | ✅ |
| 극한 intensity | 0.5 | 0.30 | 0.575 | ✅ |
| 경계 x=1 | 1.0 | 0.30 | 1.0 | ✅ |
| 경계 x=0 | 0.0 | 0.30 | 0.0 | ✅ |

**`intensity ≤ 1.0`이면 출력 항상 [0, 1]** — 클램핑 불필요 확인

---

## Critical Issues for Phase 2 Context

- 성능 위험 없음
- 수치 안정성 확인됨
- 유일한 주의점: `uToneLift` 범위 방어가 없으나, `mapSkinQuality()`만이 유일한 생성 경로이므로 실질적 위험 없음
