# Review Scope

## Target

P4-W4-01e: Luminance Sharpen 패스 추가
- Branch: `feature/P4-W4-01e` vs `develop`
- Commits: `060b011` feat(beauty): P4-W4-01e Luminance Sharpen 패스 추가
- PR 스타일 리뷰 (비판적 시각)

## Summary of Changes

FreqSep beauty pipeline 마지막에 Luminance-only Unsharp Mask 패스를 추가하여 blur로 인한 선명도 손실을 복구한다.

### 변경 사항
1. **LUMINANCE_SHARPEN_FRAGMENT** GLSL 셰이더 신규 작성 (4-neighbor cross blur, luminance ratio 방식)
2. **gpu_beauty_backend.h**: `luminance_sharpen_program_`, `LuminanceSharpenUniforms` 구조체, `FreqSepParams::sharpen_amount` 추가
3. **gpu_beauty_backend.cpp**: 셰이더 초기화, uniform 캐싱, Pass 4 (Sharpen) 파이프라인 추가, mapSkinQuality() 매핑
4. **shader_manager.h**: extern 선언 3개 추가 (FREQ_SEP_GAUSSIAN, FREQ_SEP_COMPOSITE, LUMINANCE_SHARPEN)
5. **test_beauty_config_v2.cpp**: 5개 신규 테스트

## Files

| 파일 | 변경 내용 |
|------|----------|
| `cpp/include/iris_sdk/gpu/gpu_beauty_backend.h` | `FreqSepParams.sharpen_amount` + `luminance_sharpen_program_` + `LuminanceSharpenUniforms` (+10) |
| `cpp/include/iris_sdk/gpu/shader_manager.h` | FreqSep/Sharpen extern 선언 3개 (+9) |
| `cpp/src/gpu/gpu_beauty_backend.cpp` | 셰이더 초기화, uniform 캐싱, Pass 4 파이프라인, mapSkinQuality 매핑 (+70) |
| `cpp/src/gpu/shader_sources.cpp` | LUMINANCE_SHARPEN_FRAGMENT 셰이더 (+43) |
| `cpp/tests/test_beauty_config_v2.cpp` | Sharpen 관련 5개 테스트 (+38) |

Total: +170 lines (code only)

## Flags

- Security Focus: no
- Performance Critical: yes (실시간 GPU 파이프라인, 30fps)
- Strict Mode: no
- Framework: C++17 / GLSL ES 3.1
- Skip: docs, CI/CD analysis
- Perspective: 비판적
