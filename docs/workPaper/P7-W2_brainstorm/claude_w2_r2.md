# P7-W2 R2 — Claude 응답 (블렌드 파급)

> 한 모델 독립 의견. Codex/Gemini R2 보기 전 작성.

## R2.1 정규화 해석 — **맞음**
`tinted = blendL * lum * scale = blendL * 0.85 * (lum/avgLuma)`. `avgIrisLum`은 per-pixel `lum`의 정규화 분모 → iris 영역 `lum/avgLuma ≈ 1` → tint ≈ blendL*0.85, **brightness-invariant**가 설계 의도. 실측 연결 = 이 설계의 올바른 완성 (현재는 분모가 가짜 상수라 미완성).

## R2.2 현 0.1225 = **불완전(사실상 결함)**, 단 다운스트림 튜닝이 그 위에 얹힘
scale = 0.85/0.1225 ≈ 6.94로 clamp 7.0 포화 근처 → 밝은 조건(실제 iris luma>0.1225)에서 **scale이 안 떨어져 over-tint**. 즉 현재는 정규화가 한 밝기에서만 맞음. 실측은 밝은 환경 over-tint를 **교정**. 단 W5/W6 시각 벤치가 이 고정값 위에서 됐으므로 "교정"이 곧 룩 변화 → **재벤치 필수**. 결론: 보존할 타겟이 아니라 고칠 불완전.

## R2.3 clamp/K 재튜닝 — **K=0.85 유지, clamp는 실기기 재벤치 (선제 변경 금지)**
실측 linear iris luma(srgb² 공간) 대략 분포: 갈색 sRGB~0.25→linear~0.06, 연한/블루 sRGB~0.55→~0.30. avgLuma 0.06~0.30 → scale 2.8~7.0+. 어두운 홍채는 여전히 7.0 clamp(약한 under-normalize). **K=0.85는 평균 픽셀 틴트(blendL*0.85)를 정하므로 벤치가 만족한 그 타겟 = 유지**. clamp 상한(7.0)만 어두운 홍채 범위 보고 조정 여지 → 실기기 데이터 후 결정(메모리 `real-data-first`).

## R2.4 측정=정규화 정합 — **핵심 리스크. 단 systematic 오차는 K로 흡수**
셰이더는 `uCameraTexture`(데모 OES→2D `rgbaTextureId`), detector(C)는 `rgb_buffer`. 둘 다 같은 카메라 프레임 출처지만 해상도/색변환(OES→2D shader vs cvtColor)/range가 다를 수 있음. 정합 요건: **rgb_buffer가 sRGB-encoded RGB이고, 측정도 셰이더와 동일 `srgb*srgb`+Rec.709 적용**.
- **핵심 통찰**: avgLuma의 **일정한(systematic) 오차**(예: detector가 10% 밝음)는 tint를 전역 상수배 → K 재튜닝으로 흡수 가능(perceptually 약한 전역 강도 shift). **해로운 건 noisy/비정상 오차**(프레임마다 다른 편차)인데 EMA가 완화.
- → C는 여전히 viable. 단 rgb_buffer 색공간이 셰이더와 **계통적으로** 정합하는지 1차 확인 필요. 계통 불일치가 크면 A(PBO로 rgbaTextureId ROI 비동기 readback = 픽셀 정확)로 fallback. R1의 "A는 fallback" 위치 그대로 유효, 단 사유에 "정규화 정확도"가 추가됨.

## R2.5 보존 전략 — **(a) 실측 + 재벤치/최소 재튜닝 + 토글 A/B**
- (c) gate/blend 분리(별도 uniform)는 **비추천** — 블렌드가 정규화를 위해 실측을 *필요로* 함(분리하면 블렌드는 영원히 가짜 상수). 분리는 설계 역행.
- (b) 적응형 신 룩 수용은 (a)에 포함(재벤치로 확인).
- → 실측 연결 → `setDetailReinject` 토글처럼 **fallback↔실측 토글 보존** → 6 SKU 메인 블렌드(특히 밝은 환경) 재벤치 → K/clamp 미세 조정 필요 시만. DoD에 "메인 블렌드 회귀" 명시.

## 종합
R2는 6.1(C) 결정을 뒤집지 않음 — 단 **정규화 정확도** 차원을 추가(R2.4)하고, W2 DoD를 "측정 연결"에서 "**측정 연결 + 메인 블렌드 재벤치/재튜닝 + A/B 토글**"로 확장. 측정 오차의 systematic 성분은 K 흡수, noisy 성분은 EMA 완화라 C 경로는 유지.
