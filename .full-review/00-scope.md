# Review Scope

## Target

P4-W4-01c: Edge-aware Attenuation (3-신호 결합) — develop 브랜치 대비 변경사항
브랜치: `feature/feature/P4-W4-01c` vs `develop`

## Changes Summary

5 files changed, +80/-11 lines

## Files

| 파일 | 변경 내용 |
|------|----------|
| `cpp/include/iris_sdk/gpu/gpu_beauty_backend.h` | FreqSepParams에 edge_weight/chroma_weight 필드, FreqSepCompositeUniforms에 GLint 멤버 추가 (+4) |
| `cpp/src/gpu/gpu_beauty_backend.cpp` | Uniform 초기화, Composite pass 설정, mapSkinQuality 매핑 (+8) |
| `cpp/src/gpu/shader_sources.cpp` | 3-신호 GLSL: Edge Gradient + Chroma Deviation + 결합식 (+37/-2) |
| `cpp/tests/test_beauty_config_v2.cpp` | 신규 테스트 4개 (EdgeWeightRange, ChromaWeightRange, Baseline, Increase) (+26) |
| `docs/workPaper/P4-W4-01c_edge_aware_attenuation.md` | 상태 업데이트 ⏳→✅, 완료 기준 체크 (+5/-5) |

## Context

GPU Beauty 파이프라인의 Frequency Separation Composite 셰이더에서 단일 magnitude 신호 → 3-신호 결합(Magnitude + Edge Gradient + Chroma Deviation)으로 확장하여 주름/경계 보존 및 색소침착 감지를 개선하는 작업.

## Flags

- Security Focus: no
- Performance Critical: yes (GPU 셰이더 — 실시간 30fps 필수)
- Strict Mode: no
- Framework: C++17 / GLSL ES 3.1

## Review Phases

1. Code Quality & Architecture
2. Security & Performance
3. Testing & Documentation
4. Best Practices & Standards
5. Consolidated Report
