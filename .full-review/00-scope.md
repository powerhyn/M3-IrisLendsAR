# Review Scope

## Target

P4-W4-01d 톤커브 미드톤 리프트 — 2 commits
- `d6c925a` feat(beauty): P4-W4-01d 톤커브 미드톤 리프트 구현
- `085732f` fix(beauty): P4-W4-01d tone_lift 임계값을 raw skinQuality 기준으로 수정

5 files changed, +54 insertions, -8 deletions

## Files

| 파일 | 변경 내용 |
|------|----------|
| `cpp/include/iris_sdk/gpu/gpu_beauty_backend.h` | `FreqSepParams.tone_lift` 필드 + `FreqSepCompositeUniforms.uToneLift` 멤버 (+2) |
| `cpp/src/gpu/gpu_beauty_backend.cpp` | Uniform 초기화 + Composite 패스 설정 + `mapSkinQuality()` 매핑 (+7) |
| `cpp/src/gpu/shader_sources.cpp` | `uToneLift` uniform 선언 + 미드톤 리프트 GLSL 수식 (+5) |
| `cpp/tests/test_beauty_config_v2.cpp` | ToneLift 관련 5개 테스트 (+32) |
| `docs/workPaper/P4-W4-01d_tone_curve.md` | 상태 ⏳→✅, 완료 기준 체크 (+8/-8) |

## Core Change

이차 곡선 기반 미드톤 리프트: `f(x) = x + intensity * x * (1 - x)`
- Soft Light 합성 후, mask 블렌딩 전에 적용
- `uToneLift` uniform으로 강도 제어 (기본 0.15)
- `mapSkinQuality()`에서 `t > 0.1` → 0.15 고정, `t ≤ 0.1` → `t * 1.5` 점진적 진입

## Flags

- Security Focus: no
- Performance Critical: yes (GPU 셰이더 — 실시간 30fps)
- Strict Mode: no
- Framework: C++17 / GLSL ES 3.1
