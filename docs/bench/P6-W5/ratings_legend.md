# P6-W5 평가 응답 시트 — 컬럼 의미

`ratings_template.csv` 작성 가이드. B1(블렌드) + B8(sclera) 메트릭을 한 시트에서 수집하되 **집계에서 독립 분리** (§5.9 / §1.9).

| 컬럼 | 의미 | 대상 벤치 | 입력 |
|------|------|-----------|------|
| `clip_id` | 무작위 클립 ID | — | (사전 채워짐, 변경 금지) |
| `rater_id` | 평가자 ID (R1/R2/R3) | — | (사전 채워짐) |
| `blend_natural_1to5` | **블렌드 색 자연스러움** (양성) | B1 | 1=매우 부자연 / 5=매우 자연 |
| `iris_detail_1to5` | 홍채 무늬/입체감 보존 (양성, CRL 강점 영역) | B1 | 1=뭉개짐 / 5=또렷·입체 |
| `sclera_bleed_y_n` | **흰자(공막)에 렌즈 번짐?** (음성, B8 핵심) | B8 | Y=번짐 있음 / N=깔끔 |
| `iris_edge_cut_y_n` | 홍채 외곽이 부자연스럽게 깎임? (음성, luma-only 저조도 위험) | B8 | Y=깎임 / N=정상 |
| `overall_1to5` | 종합 선호 (전체 인상) | 공통 | 1=나쁨 / 5=좋음 |
| `free_note` | 자유 기술 (1줄, 선택) | — | 텍스트 |

## 작성 원칙 (평가자에게 사전 안내)

1. **블라인드 평가** — 어떤 조합(A/B/C/D)인지 모르는 상태에서 평가.
2. **재생 환경 통일** — 동일 디바이스/화면 밝기. 가능하면 같은 방.
3. **한 클립 5초 trim** — 같은 클립 여러 번 봐도 됨.
4. **자유 기술은 1줄** — 평가 일관성 우선.
5. **두 관점 분리해서 보기**:
   - 블렌드(색/디테일): 홍채 *중앙부* 색이 자연스럽고 무늬가 살아있나?
   - sclera(흰자): 홍채 *바깥 흰자*로 렌즈 색이 번지거나, 반대로 홍채 가장자리가 깎이지 않았나?

## 판정 룰 (§5.2 / §5.5 / synthesis §1)

> R4 Patch 1: **"17/30"류 수치는 정량 가이드이지 엄격 임계값 아님**. 다수 의견 기반 정성 판정.

### B1 (블렌드: Normal vs ColorReplaceLinear)

- 각 take의 **A·B(Normal) 평균** vs **C·D(CRL) 평균** 비교: `blend_natural` + `iris_detail` + `overall`.
- 5 SKU(T1~T5)에 걸쳐 집계:
  - **CRL이 다수 SKU에서 우세** → 4종 확정 (TintLinearV2 + Multiply + ScreenLinear + CRL).
  - **CRL이 화이트/그래픽(S5)에서만 우세** → 조건부 채택 (`prefers_crl` 메타, §5.11).
  - **Normal 우세 또는 차이 미미** → 3종 + Normal (단순성 우선, §5.2).

### B8 (sclera: color-veto vs luma-only)

- 각 take의 **A·C(color-veto)** vs **B·D(luma-only)** 비교: `sclera_bleed` + `iris_edge_cut` 다수결.
- 조명별 집계 (T3/T6/T7 밝은그레이 형광/저조도/측광 + T1/T8 다크브라운):
  - **color-veto 명확 우세** (번짐 적음 + 깎임 없음) → Codex 수식 채택 (§5.4).
  - **luma-only 명확 우세** → Gemini 수식 (더 단순, §5.5).
  - **차이 미미** → luma-only (Occam's razor, §5.5).
  - **luma-only가 저조도(E3)에서 `iris_edge_cut` Y 다수** → 후속 W에서 임계 `0.3~0.5` 하향 검토 (§5.12).

### 음성 메트릭 다수결

- 평가자 3명 중 **2명 이상 Y**면 해당 조합에 마이너스.

## 집계 양식 (Phase C에서 작성)

`report.md`에 다음 두 표로 분리 정리:

**B1 표**
| 블렌드 | 평균 blend_natural | 평균 iris_detail | 평균 overall | SKU별 우세 |
|--------|--------------------|------------------|--------------|------------|
| Normal (A·B) | x.x | x.x | x.x | — |
| CRL (C·D) | x.x | x.x | x.x | (S1~S5 중 우세 셀) |

**B8 표**
| sclera veto | sclera_bleed 다수결 | iris_edge_cut 다수결 | 조명별 비고 |
|-------------|---------------------|----------------------|-------------|
| color-veto (A·C) | Y/N | Y/N | E1/E2/E3 |
| luma-only (B·D) | Y/N | Y/N | E1/E2/E3 |
