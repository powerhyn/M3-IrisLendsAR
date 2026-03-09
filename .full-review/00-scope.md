# Review Scope

## Target

P4-W4-01b: Soft Light 합성 전환 — FreqSep Composite 셰이더에서 Additive 합성을 Soft Light (Pegtop variant)로 전환 + high_freq_preserve 감쇠 계수 재튜닝.

커밋: 3d3d0e7 (feature/P4-W4-01b 브랜치, develop 기준 1커밋)

## Files

### 변경된 파일 (3개)
1. `cpp/src/gpu/shader_sources.cpp` — FREQ_SEP_COMPOSITE_FRAGMENT 셰이더: Additive → Soft Light (Pegtop) 합성 전환
2. `cpp/src/gpu/gpu_beauty_backend.cpp` — mapSkinQuality: high_freq_preserve 감쇠 0.65→0.70 + 주석 업데이트
3. `docs/workPaper/P4-W4-01b_soft_light.md` — 작업 문서 상태 업데이트

### 컨텍스트 참조 파일
- `cpp/include/iris_sdk/gpu/gpu_beauty_backend.h` — FreqSepParams 구조체
- `cpp/src/gpu/texture_pool.cpp` — 중간 버퍼 텍스처 포맷 (P1 이슈 관련)
- `cpp/tests/test_beauty_config_v2.cpp` — 기존 46개 테스트 (전부 통과)

## Change Summary

### 셰이더 변경 (핵심)
```glsl
// Before (Additive):
vec3 beauty = smoothLow + adjusted_high;

// After (Soft Light Pegtop):
vec3 blend = clamp(vec3(0.5) + adjusted_high, 0.0, 1.0);
vec3 beauty = (vec3(1.0) - 2.0 * blend) * smoothLow * smoothLow
            + 2.0 * blend * smoothLow;
```

### 튜닝 변경
```cpp
// Before: p.high_freq_preserve = 1.0f - s * 0.65f;
// After:  p.high_freq_preserve = 1.0f - s * 0.70f;
```

## Pre-identified Issues (외부 리뷰 피드백)

- **P1**: Linear-light 블러 경로의 8-bit 양자화 문제 (GL_RGBA8 → GL_RGBA16F 필요 가능)
- **P2**: Soft Light 합성에서 고주파 이득(gain) 50~82% 손실 — 정규화 미적용

## Flags

- Security Focus: no
- Performance Critical: yes (실시간 GPU 셰이더, 30fps 목표)
- Strict Mode: no
- Framework: C++17 / GLSL ES 3.1

## Exclusions

- CI/CD 검증 제외 (SDK 프로젝트, 미배포 상태)

## Review Phases

1. Code Quality & Architecture
2. Security & Performance
3. Testing & Documentation
4. Best Practices & Standards (CI/CD 제외)
5. Consolidated Report
