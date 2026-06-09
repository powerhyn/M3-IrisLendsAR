# P7-W2 R2 — 블렌드 파급(blast radius) 집중 라운드

> R1(6.1~6.5 측정 메커니즘)은 3모델 합의로 확정. R2는 R1이 놓친 **소비 측 파급** 단일 이슈만 다룬다.
> 배경 발견(Claude, 코드 검증): `uAvgIrisLum`이 detail/gate뿐 아니라 **메인 블렌드**를 구동.

## 발견된 사실 (코드 검증 완료)

`cpp/src/gpu/shader_sources.cpp`:
```glsl
// blendTintLinearV2 (canonical default, ID=5) — 877~878행
float scale  = clamp(0.85 / max(0.01, uAvgIrisLum), 0.8, 7.0);
vec3  tinted = toLinearFast(blend) * lum * scale;
// → tinted = blendL * 0.85 * (lum / uAvgIrisLum)   ← uAvgIrisLum은 per-pixel lum의 정규화 분모

// blendColorReplaceLinear (ID=7) — 888행
float detail = clamp(pow(lum / max(0.01, uAvgIrisLum), 0.7), 0.75, maxDetail);
```

**정밀 해석**: `lum * scale = 0.85 * (lum/avgLuma)`. avgLuma가 iris ROI의 실제 lum 평균과 같으면, iris 영역에서 `lum/avgLuma ≈ 1` → tint ≈ blendL*0.85, **밝기-불변**. 즉 brightness-invariant 틴트가 설계 의도이고 실측 연결이 그 완성.

**현 상태**: avgLuma=0.1225 고정 → scale = 6.94 (clamp 7.0 포화 근처). 실제 iris luma > 0.1225인 **밝은 조건에서 scale이 안 떨어져 over-tint**. W5/W6 시각 튜닝(K=0.85, clamp[0.8,7.0])은 이 고정값 기준으로 됨.

**위험**: 실측 avgLuma가 셰이더 per-pixel lum의 진짜 평균과 어긋나면(색공간/ROI/sclera 혼입) 정규화가 깨져 틴트가 들쭉날쭉. 정확도 요구가 R1 가정("gate는 근사면 충분")보다 높음.

## R2 질의 (각 항목 추천+근거 1~2줄)

- **R2.1 정규화 해석 확인**: 위 "avgIrisLum = per-pixel lum 정규화 분모 → brightness-invariant 틴트" 해석이 맞나? 코드로 확인.
- **R2.2 현 0.1225 고정 = 버그 vs 의도**: scale 7.0 포화로 밝은 조건 over-tint가 발생하는 현 상태는 (a) 실측으로 고쳐야 할 결함인가, (b) 보존할 튜닝 타겟인가?
- **R2.3 clamp[0.8,7.0] + K=0.85 재튜닝**: 실측 linear iris luma 분포(`srgb*srgb` 공간, 갈색~연한 홍채 × 조도 범위)를 고려할 때 clamp/K 조정 필요? 권장 방향.
- **R2.4 측정=정규화 정합 보장**: 셰이더는 `uCameraTexture`(데모: OES→2D `rgbaTextureId`)를, detector(C안)는 `rgb_buffer`를 본다. **둘이 같은 픽셀/색공간인가?** C안에서 정규화 정확도를 어떻게 보장(같은 프레임 출처·동일 srgb² 선형화·동일 ROI)?
- **R2.5 W5/W6 튜닝 보존 전략**: (a) 실측+재튜닝으로 기존 룩 보존, (b) 적응형 신 룩 수용+재벤치, (c) gate와 blend 분리(별도 uniform로 blend는 안정값 유지)? 추천.

## 응답 저장
- Codex → `docs/workPaper/P7-W2_brainstorm/codex_w2_r2.md`
- Gemini → `docs/workPaper/P7-W2_brainstorm/gemini_w2_r2.md`
