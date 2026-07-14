# P7-W4 R1 — Codex 입장

> 작성: 2026-06-10
> 전제: §1.5 적용 위치 제약(post-blend 불가)과 §1.6 베이스라인 문제 수용. Oklab/Laplacian 도입 제안 없음.

---

## 6.1 재베이스라인 gating

**추천**: 코드 작성 전 S23+에서 `lum:meas` 기본 상태로 SKU 2/5/6(+대조군 1)을 재베이스라인한다. 밝은 환경에서 SKU 6/2의 흰자 빛남이 즉시 거슬리면 원안 진행, SKU 6도 "밝은 렌즈 특성" 정도로만 보이면 조기 종결 또는 기본 강도 완화로 축소한다.

**근거**: W9의 빛남 패턴은 ID=5 고정 벤치에서 확인됐지만(`docs/workPaper/P6-W9_integration_report.md:75-90`), W2 이후 밝은 환경 scale이 7.0에서 1.27로 교정되어 현재 기준점이 달라졌다(`docs/workPaper/P7-W2_avg_iris_luma_measure.md:63-69`). 어두운 환경(avgLum<0.12)만 남는 빛남은 저조도 우선순위가 낮으므로 W4 진행 여부의 주 기준으로 삼지 않는다.

## 6.2 수식 형태

**추천**: (c) 비율 clamp 계열을 택하되, 구현은 현재 `lum * scale` 유효 틴트 배율의 상한만 거는 형태가 좋다. 예: `float tintMul = min(lum * scale, uScleraTintMax); vec3 tinted = lensL * tintMul;`이며 초기 `uScleraTintMax`는 `0.85 * 1.5` 수준에서 토글 비교한다.

**근거**: 현 빛남 원인은 `tinted = lensL * lum * scale`에서 sclera의 높은 `lum`이 그대로 틴트 배율이 되는 구조다(`cpp/src/gpu/shader_sources.cpp:877-880`). post-blend 감쇄는 이미 카메라와 섞인 `blended`를 어둡게 만들 수 있으므로 배제하고(`cpp/src/gpu/shader_sources.cpp:880`, `cpp/src/gpu/shader_sources.cpp:1134`), ID=7도 ratio clamp 패턴을 이미 쓴다(`cpp/src/gpu/shader_sources.cpp:888-890`).

## 6.3 sclera_factor의 luma 공간 + 임계값

**추천**: 6.2의 비율 clamp를 채택하면 별도 `sclera_factor`와 `smoothstep(0.65, 0.85, ...)` 임계는 만들지 않는다. 만약 (a)를 강행한다면 블렌드 내부와 같은 linear Rec.709를 쓰고, 임계값은 linear 기준으로 재튜닝해야 한다.

**근거**: 블렌드 luma는 `toLinearFast(base)` 뒤 `LUMA_709_LENS`로 계산되며, 해당 상수도 CPU 측 avg_iris_luma와 동일 계약이다(`cpp/src/gpu/shader_sources.cpp:848-850`, `cpp/src/gpu/shader_sources.cpp:876-879`). 기존 veto의 sRGB 0.299/0.587/0.114 luma는 finalAlpha 보호용 별도 경로라 신규 틴트 배율 판단 기준으로 섞지 않는 편이 안전하다(`cpp/src/gpu/shader_sources.cpp:1021-1023`).

## 6.4 기존 geometry veto와의 결합 방식

**추천**: W4에서는 기존 sclera veto를 현재 기본값 그대로 둔 채 ID=5 틴트 배율만 조정한다. 신규 계층에 geometry gate를 추가하지 않고, B8의 color-veto vs luma-only 단일화도 W4 결과 뒤로 분리한다.

**근거**: 현재 sclera protect는 이미 기본 ON/legacy mode이며(`cpp/include/iris_sdk/gpu/gpu_lens_renderer.h:241-242`), 셰이더에서 geometry 기반 `finalAlpha` 감쇄로 적용된다(`cpp/src/gpu/shader_sources.cpp:1011-1035`). W4가 `tinted` 배율 상한만 다루면 alpha veto와 역할이 분리되어 이중 감쇄 위험을 키우지 않는다.

## 6.5 iris 내부 밝은 픽셀 오발

**추천**: 1차에서는 허용하고 geometry 보조 게이트를 넣지 않는다. 단, 재베이스라인/토글 벤치에서 밝은 홍채나 각막 하이라이트가 칙칙해지는지 육안 확인한다.

**근거**: 비율 clamp는 평균 홍채 근처의 정상 배율은 유지하고 과도한 `lum * scale`만 제한하므로, 밝은 홍채 전체를 sclera로 오인하는 smoothstep 감쇄보다 보수적이다(`cpp/src/gpu/shader_sources.cpp:877-880`). 하이라이트의 렌즈색 약화는 실제 반사가 렌즈색을 덮는 방향이라 1단계 부작용으로 보기 어렵다.

## 6.6 적용 블렌드 범위 + ID=7 경계

**추천**: 적용 범위는 `blendTintLinearV2` 단독으로 제한한다. 순서는 W4로 ID=5를 먼저 안정화한 뒤, 그 결과를 기준으로 ID=7 채택/제거와 dropdown 축소를 결정한다.

**근거**: W9의 문제 벤치는 TintLinearV2 ID=5 고정이었다(`docs/workPaper/P6-W9_integration_report.md:75-76`). 셰이더도 ID=5와 invalid fallback만 TintLinearV2로 가고, ID=7은 별도 `maxDetail` clamp 경로를 탄다(`cpp/src/gpu/shader_sources.cpp:1057-1063`, `cpp/src/gpu/shader_sources.cpp:1042`, `cpp/src/gpu/shader_sources.cpp:1059-1060`).

## 6.7 SKU별 오버라이드

**추천**: 1차는 글로벌 단일값 + ON/OFF 토글로 시작한다. SKU별 메타 오버라이드는 글로벌 값으로 SKU 2/5/6 중 특정 SKU만 명확히 과/부족할 때 후속으로 연다.

**근거**: W9 패턴은 특정 SKU 예외라기보다 렌즈 명도와 양의 상관으로 정리되어 글로벌 상한이 먼저 맞는 문제다(`docs/workPaper/P6-W9_integration_report.md:87-90`). 재베이스라인 전부터 SKU별 임계/강도를 열면 §1.6의 기준점 재정의와 수식 검증이 동시에 흔들린다.

---

## 요약

| 쟁점 | Codex 추천 |
|---|---|
| 6.1 | 재베이스라인 선행, 잔존 미미 시 축소/조기 종결 |
| 6.2 | `lum * scale` 유효 배율 상한 clamp |
| 6.3 | 별도 `sclera_factor` 없음; 필요 시 linear Rec.709 |
| 6.4 | 기존 veto 고정, B8 결정 분리 |
| 6.5 | iris 밝은 픽셀 1차 허용, 토글 벤치로 확인 |
| 6.6 | ID=5만 적용, W4 후 ID=7 결정 |
| 6.7 | 글로벌 단일값 우선, SKU 메타는 후속 |
