# Commit Message

## Single Commit (Recommended)

```
feat(beauty): Frequency Separation GPU 파이프라인 — 셰이더 기반 고급 피부 스무딩 [P4-W3-02]

Separable Gaussian + Composite 셰이더를 사용한 5서브패스 Frequency Separation
파이프라인을 GPUBeautyBackend에 추가합니다.

주요 구현:
- GLSL ES 3.1 셰이더: 1D Gaussian blur (adaptive radius 6~28) + Composite
  (high-freq inline extraction, non-linear blemish attenuation, mask blending)
- executeFreqSepPipeline: GaussH → GaussV → LowSmoothH → LowSmoothV → Composite
- uploadSkinMask: CPU combined_mask → GL_R8 텍스처 (크기 변경 시만 재할당)
- mapSkinQuality: smoothstep S-curve 기반 skinQuality → FreqSepParams 매핑
- skinQuality API 필드 (0~1): C++ / C API / JNI / Android Java 전 레이어 반영
- Graceful degradation: 셰이더 실패, 텍스처 부족, 마스크 업로드 실패 시
  Bilateral fallback (최소 강도 skinQuality * 0.5)

안전성 강화:
- executeFreqSepPipeline bool 반환으로 실패 전파 + 호출자 fallback
- freq_sep_composite_program_ 체크 추가 (glUseProgram(0) 방지)
- FreqSep 멀티패스 시 ROI scissor 비활성화/복원
- uploadSkinMask 버퍼 크기 검증 (OOB read 방지)
- executeSmoothingWithFallbackStrength #if IRIS_SDK_GPU_AVAILABLE guard

테스트: mapSkinQuality 13건 + skinQuality validation 2건 추가 (총 35개 통과)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

## Rationale
- **Single commit**: 모든 변경이 하나의 작업 단위(P4-W3-02)에 속함
- **feat type**: 신규 GPU 파이프라인 기능 추가
- **beauty scope**: GPUBeautyBackend 뷰티 파이프라인 대상
- **Korean body**: 프로젝트 기존 커밋 메시지 스타일과 일치
