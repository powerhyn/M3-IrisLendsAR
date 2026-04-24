# P6-W7 브레인스토밍 R1 — 종합

**작성일:** 2026-04-24
**참여:** Codex, Gemini, Claude
**편향 경계:** Claude 림발 정책 R1~R3 3번 뒤집힘 히스토리, R4 실기기 10/10 확인 이후 결과 반영.

---

## 1. 합의 (3/3 동의 → 닫힘)

| 쟁점 | 확정 |
|------|------|
| **6.1** 자동감지 ROI | **`[0.85, 1.0]` 유지** (false positive 리스크 최소화) |
| **6.2** 임계값 | **`0.75` 유지** (99 합의본 기준, 변수 분리) |
| **6.4** 메타 누락 로그 | **WARN** (앱 무중단 + 개발자 즉시 인지) |
| **6.6** 엔비_샤모 브라운 | **`prefers_graphic_outline = true` 별도 플래그 + 셰이더 림발 강제 OFF** (uApplyLimbal = 0) |
| **6.7** B10 신규 벤치 | **추가 금지, W7 out of scope 유지** |

## 2. 다수결 결정 (2/3 동의)

### 6.3 메타데이터 저장 방식 — **JSON 파일 채택 (Gemini+Claude 다수)**
- Gemini, Claude: JSON (`android/demo-app/src/main/assets/lens_meta.json` 또는 `lens_sku_metadata.json`).
- Codex: 하드코드 레지스트리 `LensSkuRegistry` (버전 고정, 내부 필드).
- **판정:** JSON 다수 채택. Codex 절차 우려는 다음 조치로 완화:
  - **버전 고정:** JSON 파일도 git 체크인 → 버전 고정 효과 동일.
  - **default fallback:** 메타 로더에 누락 SKU는 모든 플래그 `false` 기본값 (§6.4 WARN 로그 병행).
  - **판별 가능성:** SKU 제작자(디자이너) JSON 직접 편집 가능 → 재빌드 사이클 제거.
- **최종 스키마:**
  ```json
  [
    {
      "sku_id": "클라셋_돌_초코",
      "display_name": "다크브라운",
      "has_baked_limbal": true,
      "prefers_crl": false,
      "prefers_graphic_outline": false
    }
  ]
  ```

### 6.5 9/10 vs 10/10 — **10/10 엄수 채택 (Codex+Claude 다수)**
- Codex, Claude: **10/10 엄수** (R4 실기기 증거).
- Gemini: 9/10 완화 (실기기 노이즈 우려).
- **판정:** **실기기 R4 결과 "10/10 달성"이 팩트로 확인**된 상태. Gemini의 "실기기 노이즈 우려"는 이론적 가능성인데 R4 실측이 이를 기각.
- **완화 조항 (Gemini 우려 흡수):**
  - 배포 후 특정 디바이스에서 "자동감지 실패 1/N" 발견 시, **해당 SKU만 `has_baked_limbal: true` 메타 플래그로 명시 fallback**.
  - 자동감지 rule은 10/10 엄수, 실패 SKU는 **메타 플래그로 처리** — 자동감지 완화가 아닌 **개별 명시**.
- **Claude 편향 경계 재확인:** Claude 이번 R1 10/10 입장은 **R4 실측 팩트 일치**. Codex 편향 아님. P5 R1~R3에서 뒤집혔던 주관과 다름.

## 3. 미결 / 후속 확인

**없음.** 7개 쟁점 모두 R1 결론.

- **6.5 완화 조항 운영:** 배포 후 1개월 모니터링, 실기기 실패 발견 시 해당 SKU만 메타 fallback 처리.

## 4. 편향 체크

- **Claude 편향 경계 적중 (6.3, 6.5):**
  - 6.3 Claude JSON 제안이 Gemini JSON과 합쳐져 다수. Codex 하드코드는 "P5 R3에서 Claude가 흔들렸던 경로"는 아니라서 편향이라기보다 취향 차이 — 완화 조건으로 수용.
  - 6.5 Claude 10/10 입장이 **R4 실기기 결과 재확인** 기반. P5 R1~R3의 주관적 뒤집힘과 달리 이번은 실측 팩트. **편향 아닌 사실 수렴**.
- **Gemini 편향 (6.5):** "9/10 완화"는 Gemini의 지속적 스탠스였으나 R4 실측 후에도 유지된 입장. 이번 R1에서 다수(10/10 + 메타 fallback)에 의해 흡수됨.
- **Codex 편향 없음:** 6.5 10/10 엄수는 R4 실측과 일치. 6.3 하드코드는 소수 취향이지만 기술적 위험 없음.

## 5. W7 문서 §5에 반영할 항목

- **§5.6** 자동감지 ROI: `[0.85, 1.0]` 유지. 튜닝 금지.
- **§5.7** 임계값: `0.75` 유지. W7 1차 튜닝 금지.
- **§5.8** 메타 저장: **`android/demo-app/src/main/assets/lens_meta.json`** JSON 파일. SDK 공용 파서 또는 경량 파서 추가. default fallback = 모든 플래그 `false`.
- **§5.9** 메타 누락 로그: **WARN**. 형식: `[IrisSDK] SKU meta missing for "<sku_id>", using default`.
- **§5.10** 9/10 vs 10/10: **10/10 엄수 + 실기기 실패 SKU는 `has_baked_limbal: true` 메타로 개별 명시**.
- **§5.11** 엔비_샤모 브라운: `prefers_graphic_outline: true` 플래그. `uApplyLimbal = 0` (셰이더 림발 강제 OFF).
- **§5.12** B10 신규 벤치: **불허 확정**. 별도 제안 문서 필요 (P6 범위 밖).

## 6. 다음 액션

1. W7 문서 §5/§6 업데이트.
2. Task #7 완료 표시.
3. W8 브레인스토밍 착수 (**W8은 W4 B2 결과에 조건부이지만, R1 사전 준비 목적으로 진행**).

## 7. R2 필요 여부

**R2 불필요.** 합의 5개 + 다수결 2개. Hard veto 없음. Gemini 9/10 우려는 "개별 메타 fallback"으로 흡수. Claude R4 실측 팩트 수렴 확인.
