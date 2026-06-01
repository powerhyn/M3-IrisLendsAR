# P6-W5 평가 시트 — 컬럼 의미 (1인 실시간 토글 체감)

> **평가 방식**: 1인 개발 체제 → 평가자 3명 블라인드 다수결 대신 **본인 실시간 토글 체감**.
> 실기기에서 take(SKU×조명)별로 A/B/C/D 버튼을 토글하며 즉석 비교, take당 1행 판정.
> 근거: 메모리 `feedback_qualitative_device_judgment` (정량 수치보다 실기기 육안 체감 우선).

`ratings_template.csv` 작성 가이드.

| 컬럼 | 의미 | 입력 |
|------|------|------|
| `take` | Take ID (T1~T9) | (사전 채워짐) |
| `sku` | 렌즈 SKU | (사전 채워짐) |
| `lighting` | 조명 환경 | (사전 채워짐) |
| `B1_blend_winner` | **블렌드 비교**: A·B(Normal) vs C·D(CRL) 중 더 자연스럽고 홍채 디테일 살아있는 쪽 | `Normal` / `CRL` / `tie` |
| `B1_note` | B1 체감 메모 (색 자연스러움, 무늬 보존 등) | 텍스트 (1줄) |
| `B8_sclera_winner` | **sclera 비교**: A·C(color-veto) vs B·D(luma-only) 중 흰자 번짐 적고 홍채 외곽 깎임 없는 쪽 | `color` / `luma` / `tie` |
| `B8_note` | B8 체감 메모 (흰자 번짐, 외곽 깎임 등) | 텍스트 (1줄) |
| `overall_note` | take 전체 인상 (선택) | 텍스트 |

## 토글 비교 방법 (실기기)

각 take에서 같은 자세/조명으로 4조합을 번갈아 토글하며:

- **B1 (블렌드)**: sclera 조건 고정하고 블렌드만 비교
  - A↔C 토글 (둘 다 color-veto, Normal vs CRL)
  - B↔D 토글 (둘 다 luma-only, Normal vs CRL)
  - → 홍채 *중앙부* 색·무늬가 어느 쪽이 자연스러운가 → `B1_blend_winner`
- **B8 (sclera)**: 블렌드 조건 고정하고 sclera만 비교
  - A↔B 토글 (둘 다 Normal, color vs luma)
  - C↔D 토글 (둘 다 CRL, color vs luma)
  - → 홍채 *바깥 흰자* 번짐·외곽 깎임이 어느 쪽이 깔끔한가 → `B8_sclera_winner`

> 토글하며 직접 비교하므로 미묘한 차이도 즉석에서 감지 가능 (1인 체감의 강점).
> 단, 확증 편향 주의 — 어느 게 CRL인지 알고 보므로 "CRL이 좋아야 한다"는 선입견 경계.
> 애매하면 `tie`로 두고, 정 판단 어려운 take만 `recording_guide.md`의 self-blind 보조(randomize) 활용.

## 판정 룰 (§5.2 / §5.5)

> 통계적 다수결이 아닌 **정성 판단**. take별 winner를 모아 경향 파악.

### B1 (블렌드: Normal vs CRL)

- T1~T5(5 SKU 형광)의 `B1_blend_winner` 경향:
  - **CRL이 다수 SKU에서 우세** → 4종 확정 (TintLinearV2 + Multiply + ScreenLinear + CRL).
  - **CRL이 화이트/그래픽(S5/T5)에서만 우세** → 조건부 채택 (`prefers_crl` 메타, §5.11).
  - **Normal 우세 또는 대부분 tie** → 3종 + Normal (단순성 우선, §5.2).

### B8 (sclera: color-veto vs luma-only)

- 조명별 `B8_sclera_winner` 경향 — 밝은그레이 {T3/T7/T6} + 다크브라운 {T1/T9/T8}:
  - **color-veto 명확 우세** (번짐 적음 + 깎임 없음) → Codex 수식 채택 (§5.4).
  - **luma-only 명확 우세 또는 대부분 tie** → Gemini 수식 (더 단순, §5.5 Occam).
  - **luma-only가 저조도(T6/T8)에서 외곽 깎임 발생** → 후속 W 임계 `0.3~0.5` 하향 검토 (§5.12).

## 집계 양식 (Phase C — `report.md`)

take별 winner 표를 그대로 옮기고 B1/B8 경향 요약. 통계 N/M 대신 "X take 중 Y에서 CRL 우세, 특히 S5" 식 정성 서술.
