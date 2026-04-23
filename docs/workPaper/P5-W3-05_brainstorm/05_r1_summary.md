# R1 결론 요약 (Claude 관점의 수렴 상태)

> **작성**: Claude (모더레이터)
> **시점**: R1 3응답 완료 직후, R2 쟁점 추출 직전
> **성격**: **Claude 1인 종합**. 세 모델 합의본 아님. R2/R3 과정에서 수정 가능.

---

## 세 모델 R1 입장 한 줄 요약

| 모델 | 한 줄 결론 |
|------|----------|
| **Claude** | 블렌드 3종 압축 + 카메라 프레임 기반 환경 반사로 고정 조명 대체. full PBR·3D geometry·neural 모두 기각. |
| **Gemini** | 정적 분석 조명 버리고 카메라 프레임 SSR + LTL+Circle+Vivid 3종 하이브리드. 이중 노멀 굴절 제안. |
| **Codex** | LDR/RGBM eye env map 에셋 + Fresnel 반사 + 4종 블렌드(TintLinearV2/Multiply/ScreenLinear/ColorReplaceLinear). 카메라 기반 환경맵은 기각. EyeRenderPacket 구조화. |

**응답 분량**: Codex 19KB (가장 상세, 파일:라인 레퍼런스 포함) > Claude 14KB > Gemini 7.5KB.

---

## 합의 지점 (세 모델 공통)

| # | 항목 | 근거 |
|---|------|------|
| 1 | HDR IBL 에셋(~1.8MB) 도입 기각 | 정적 조명이라 실시간 환경 반응 불가 |
| 2 | Full PBR 기각 | 렌즈 도메인에 과잉, 튜닝 비용만 증가 |
| 3 | 3D FaceGeometry 재도입 기각 | 입력 결합도 파괴, 검출기 교체 시 파급 |
| 4 | Neural rendering / harmonization 기각 | SDK 크기·지연 예산 파괴 |
| 5 | Corneal refraction 기각 | 육안 체감 미미 |
| 6 | LTL 계열을 기본 블렌드로 | 세 모델 합의 |
| 7 | Multiply 블렌드 유지 | 짙은 렌즈 필수 |
| 8 | 추가 FBO pass 0개 (single-pass) | 성능 예산 |
| 9 | W1 TemporalStabilizer가 좌표 스무딩 소유, 렌더러는 material-only | Claude/Codex 명시, Gemini 미반대 |
| 10 | W3-04 고정 조명 `vec3(0.3, 0.4, 1.0)` 폐기 | 세 모델 비판 |
| 11 | 블렌드 8종은 과다, 정리 필요 | 세 모델 동의 |

---

## 대립 지점 (R2에서 다룰 쟁점)

### 축별 요약 표

| 축 | Claude R1 | Gemini R1 | Codex R1 | 1차 대립 구도 |
|----|-----------|-----------|----------|--------------|
| A 광학 리얼리즘 | 카메라 frame mip | 카메라 SSR (고휘도 샘플) | env map 에셋 + Fresnel | **Claude+Gemini vs Codex** |
| B 블렌드 세트 | Normal/LTL/Multiply | LTL/Circle/Vivid 하이브리드 | TintLinearV2/Multiply/ScreenLinear/ColorReplaceLinear | 3개 다른 안 |
| C 마스킹·림발 | opt-in 기본 OFF | 동적 생성, 밝기 반비례 | 기본 ON + SKU 메타 플래그 | **Claude vs Gemini/Codex** |
| D 시간적 안정성 | alpha ramp down | 즉시 Off | ease-out 60~80ms + ease-in 100~120ms | 3개 다른 안 |
| E 입력 계약 | pupil + render_confidence | gaze vector | EyeRenderPacket 구조화 | 3개 다른 안 |
| F 성능 | 추가 pass 0 | Single-pass + LUT | Single-pass + env 1 + detail 4 | **동일 방향** (세부 차이) |
| G 와일드카드 | 모두 기각 | 절차적 홍채 섬유 | eye-only refiner | 3개 다른 안 |

### Claude가 R1을 놓친 축

- **축 B-보조 (홍채 디테일 보존)**: Claude R1에서 언급 없음. Codex가 `detail reinjection`으로 원본 휘도 재주입 제안. Gemini는 절차적 섬유 제안. R2에서 Claude 입장 정리 필요.
- **`uAvgIrisLum = 0.35` 하드코드 문제**: Claude R1에서 짚지 않음. Codex가 파일:라인으로 "priors 덮어씌움" 지적. R2에서 반드시 다룰 것.

---

## Claude가 본 수렴 가능성 평가 (R1 종결 시점)

| 대립 축 | R2에서 수렴 가능성 | 이유 |
|---------|------------------|------|
| A 환경 반사 소스 | **낮음** (실기기 이관 가능성 높음) | 말로는 "재귀 vs 정합성" 트레이드오프. 실기기 벤치 불가피 |
| B 블렌드 세트 | **중** | 수식 비교 + 시나리오 매트릭스로 좁힐 여지. 단 4번째 슬롯(Normal vs CRL)은 벤치 이관 |
| C 림발 | **중** | 메타 vs 자동 감지는 말로 좁히기 가능, 기본 ON은 합의될 것 |
| D 블링크 | **높음** | 시간 범위만 맞추면 수렴 쉬움 |
| E 입력 계약 | **높음** | EyeRenderPacket이 가장 포괄적, 세 모델의 필드를 모두 수용 가능 |
| F 성능 | **이미 수렴** | 세부 차이만 조율 |
| G 와일드카드 | **중** | 각 모델이 제시한 와일드카드 중 W2 트랙(Codex) vs 렌더링 보존(Claude) 경계 정리 필요 |

**R2 전략**: 합의된 11개는 닫고, 대립 7개 축을 쟁점별로 분해해서 각 모델에 "유지/수정/철회"를 강제한다. 수렴 낮은 축은 벤치 이관 명시.

---

## Claude R1 자기비판 (R1 자기 응답에 포함된 내용 중 R2 전 재평가 필요)

- 분석적 노멀 + 고정 조명 방향 자체가 틀렸음 (Claude 본인이 자기 구현 인정)
- `specular * 0.7` 강도 과다 (적정 0.25~0.35)
- `LIMBAL_ENABLED = const false` 하드코드 확장성 없음
- LTL 실반사 보호 임계값(0.7/0.95) 너무 높음 → 대부분 눈 밝기 보호 안 걸림

→ Codex가 **파일:라인으로 더 예리하게** 같은 지점들 + 추가 지점(avgIrisLum hardcode, lightMask 중심 강제, blend*lighting 물리 오류)을 비판. R2에서 Codex 비판을 정면으로 받아야 함.

---

## 문서 한계 명시

이 문서는 **Claude 1인이 R1 3응답을 읽고 본 수렴 상태**를 정리한 것이다:
- "합의 지점" 판단은 Claude가 한 것. 세 모델이 명시 합의한 게 아님.
- "대립 구도" 분류도 Claude 관점.
- "수렴 가능성 평가"는 Claude 예측.

따라서 이 요약은 **R2 쟁점 추출의 입력**으로만 쓰인다. 최종 결정의 근거로는 세 모델 R2 + R3 검토를 거친 `99_final_decision.md`를 참조할 것.
