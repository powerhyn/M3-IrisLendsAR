# P4-W4-01: FreqSep 피부 잡티 보정 품질 개선

## 작업 개요

현재 FreqSep 파이프라인의 품질을 업계 레퍼런스 수준으로 개선한다.
기존 5-pass GPU 구조를 유지하면서 composite 셰이더 개선 + 후처리 패스 추가로
상용 SDK 80%+ 품질을 확보하는 것이 목표이다.

## 배경

### 현재 문제점

1. **흐림감**: Gaussian blur가 에지를 무시하여 눈/머리카락/입술 경계에서 halo 발생
2. **플라스틱 피부**: additive 합성이 톤/콘트라스트를 평탄화
3. **주름 오삭제**: magnitude 기반 smoothstep이 주름(중간 진폭)을 잡티로 오판
4. **색소침착 미검출**: 넓지만 진폭 낮은 색소침착이 감쇠 임계값 이하로 통과
5. **선명도 손실**: 스무딩 후 복구 패스가 없어 전반적 흐림

### 분석 경위

- `FreqSep_Quality_Analysis_Report.md`: 레퍼런스 3건과 비교 분석
- `FreqSep_Council_Review.md`: Agent Council (Codex/Gemini/Claude) Q1~Q5 리뷰 결과
- Council 합의 적용 순서: **Linear RGB → Soft Light → Edge-aware → 톤커브 → Sharpen**

## 작업 상태

| Step | 작업 | 상태 |
|:----:|------|:----:|
| 1 | Linear RGB 색공간 전환 | ✅ 완료 |
| 2 | Soft Light 합성 전환 | ⏳ 대기 |
| 3 | Edge-aware Attenuation | ⏳ 대기 |
| 4 | 톤커브 미드톤 리프트 | ⏳ 대기 |
| 5 | Luminance Sharpen | ⏳ 대기 |

## 세부 작업 문서

각 Step의 상세 구현 계획은 별도 문서로 분리:

| 문서 | 내용 |
|------|------|
| `P4-W4-01a_linear_rgb.md` | Step 1: Linear RGB 색공간 전환 |
| `P4-W4-01b_soft_light.md` | Step 2: Soft Light 합성 전환 |
| `P4-W4-01c_edge_aware_attenuation.md` | Step 3: Edge-aware Attenuation (3신호 결합) |
| `P4-W4-01d_tone_curve.md` | Step 4: 톤커브 미드톤 리프트 |
| `P4-W4-01e_luminance_sharpen.md` | Step 5: Luminance Sharpen 패스 추가 |

## 현재 파이프라인 (변경 전 기준선)

```
입력 프레임 (sRGB, GL_TEXTURE_2D)
    │
    ├─ input_tex (원본)
    ├─ mask_tex (skinMask)
    │
[Pass 1a] freq_sep_gaussian_program_ — Horizontal Gaussian → temp
[Pass 1b] freq_sep_gaussian_program_ — Vertical Gaussian   → lowFreq
    │
[Pass 2a] freq_sep_gaussian_program_ — Horizontal Gaussian on lowFreq → temp
[Pass 2b] freq_sep_gaussian_program_ — Vertical Gaussian on temp      → smoothedLow
    │
[Pass 3]  freq_sep_composite_program_ — Composite
    ├─ texture unit 0: uSmoothedLow (smoothedLow)
    ├─ texture unit 1: uLowFreq (lowFreq)
    ├─ texture unit 2: uOriginal (input_tex)
    ├─ texture unit 3: uSkinMask (mask_tex)
    ├─ uniform: uHighFreqPreserve, uAttenuationLow, uAttenuationHigh
    └─ output → output_fbo
```

**패스 수**: 5 (Gaussian H/V × 2 + Composite)
**개선 후 예상**: 6~7 (+ Sharpen 1~2 패스)

## 변경 대상 파일 목록

| 파일 | 변경 내용 | 관련 Step |
|------|----------|:---------:|
| `cpp/src/gpu/shader_sources.cpp` | FREQ_SEP_GAUSSIAN_FRAGMENT, FREQ_SEP_COMPOSITE_FRAGMENT 수정, SHARPEN_FRAGMENT 신규 | 1,2,3,4,5 |
| `cpp/src/gpu/gpu_beauty_backend.cpp` | mapSkinQuality 확장, initializeShaders에 sharpen 추가, executeFreqSepPipelineImpl에 sharpen 패스 연결 | 1,3,4,5 |
| `cpp/include/iris_sdk/gpu/gpu_beauty_backend.h` | FreqSepParams 필드 추가, FreqSepCompositeUniforms 필드 추가, sharpen 프로그램/uniform 멤버 | 3,4,5 |
| `cpp/tests/test_beauty_config_v2.cpp` | FreqSep 테스트 기대값 업데이트 | 1,3 |

## 테스트 전략 (Phase 1 공통)

현재 자동 테스트는 `mapSkinQuality()` 파라미터 범위 검사에 머무르고 있다.
핵심 리스크(셰이더 수학, MID half-res 경로, mask 경계)를 커버하려면 아래가 필요:

| 테스트 유형 | 커버리지 | 우선순위 |
|------------|---------|---------|
| **골든 이미지 테스트** | full-res + MID half-res에서 기준 출력 비교 (PSNR/SSIM) | 높음 |
| **Shader fixture** | 소규모 텍스처로 셰이더 수학 검증 (Linear 변환, attenuation, edge) | 높음 |
| **Half-res 경로 테스트** | res_divisor=2에서 temp/compositeRT 해상도 정합성 | 높음 |
| **Mask 경계 테스트** | mask 0→1 전이 영역에서 아티팩트 없음 확인 | 중간 |

> Phase 1 완료 후 Android 디바이스 실측과 병행하여 골든 이미지 기준선 확정

## Phase 2 (조건부)

Phase 1 완료 후에도 품질이 부족한 경우:
- **3-Scale Wavelet Decompose** (~10 pass) 우선 검토
- **O(1) Bilateral Filter** (~12 pass) 대안

진입 기준: 품질 게이트 미달 + 성능 게이트 (30fps 기준 GPU 여유 3~4ms) 충족

## 변경 이력

| 일자 | 내용 |
|------|------|
| 2026-03-08 | 초안 작성, Phase 1 Step 1~5 정의 |
| 2026-03-09 | 세부 구현 문서 5건으로 분리, 코드 레벨 상세화 |
| 2026-03-09 | Step 1 구현 완료, 피드백 6건 반영 (temp/half-res, chromaDev, luma 계수, 문서 정합성, 상태 동기화, 테스트 전략) |
