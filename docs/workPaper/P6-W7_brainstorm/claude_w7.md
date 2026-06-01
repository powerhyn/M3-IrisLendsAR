# P6-W7 브레인스토밍 R1 — Claude 응답

**모델:** claude-opus-4-7 (1M)
**라운드:** R1
**대상:** `docs/workPaper/P6-W7_limbal_policy.md` §6.1 ~ §6.7
**숨김.**

**편향 히스토리 인식:** P5 R1~R3에서 Claude가 림발 정책을 3번 뒤집은 이력 존재. 이번 R1은 실기기 결과(R4 patch) 기반 사실에 충실, 주관 최소화.

---

## 6.1 자동감지 ROI 경계 튜닝 — **`[0.85, 1.0]` 유지, 실기기 결과 후 ±0.03 선회 여지**

근거:
- (1) R4 patch 수식이 `[0.85, 1.0]`로 고정된 상태에서 실기기 테스트가 이미 완료됐음 (근거: W7 문서 §5 "R4 patch 반영"). 이 범위를 바꾸면 R4에서 검증된 경로 재검증 필요 — 작은 W 범위 유지 원칙에 반함.
- (2) 현재 범위가 "바깥 15%만 샘플링"이라 iris 본체 영향 0. W7 1차 구현 완료 후 특정 SKU에서 "림발 탐지 누락"이 실기기에서 반복되면 `[0.82, 1.0]`로 하향 검토 (±0.03 이내).

## 6.2 임계값 `0.75` 튜닝 — **0.75 기본 유지**

근거:
- (1) R4 patch의 기본값이 0.75. 0.70/0.80은 R4 시점에 Codex/Gemini가 제안했다가 실기기 "5 SKU 전체 통과" 확인으로 0.75에 수렴한 상태(R4 결정).
- (2) 엔비_샤모 브라운 같이 그래픽 림발 안내 SKU는 §6.6 `prefers_graphic_outline` 별도 플래그로 처리되므로 임계값 단일 0.75로 충분.

## 6.3 메타데이터 저장 방식 — **JSON 파일 (`android/demo-app/src/main/assets/lens_meta.json`)**

근거:
- (1) 외부 편집 용이 + 새 SKU 추가 시 코드 재컴파일 불필요. SKU 제작자(디자이너)가 직접 `prefers_crl` / `has_baked_limbal` / `prefers_graphic_outline` 플래그 설정 가능.
- (2) 하드코드 레지스트리(`LensSkuRegistry`)는 SKU 수 증가 시 빌드 시간 증가 + 메타 변경마다 릴리스 필요. 파일명 규약(`_bl` 접미사)은 플래그 조합 3개 이상 되면 파일명 가독성 저하.
- (3) JSON 로더는 SDK 공용 JSON parser 있으면 재사용. 없으면 경량 파서 200줄 내외 추가.
- (4) **스키마 제안:**
  ```json
  {
    "sku_id": "클라셋_돌_초코",
    "display_name": "다크브라운",
    "has_baked_limbal": true,
    "prefers_crl": false,
    "prefers_graphic_outline": false
  }
  ```

## 6.4 메타 누락 경고 로그 레벨 — **WARN**

근거:
- (1) 앱 자체는 깨지지 않음 (기본 플래그 `false`로 자동 적용). ERROR은 운영 모니터링 알람 남발.
- (2) DEBUG은 개발자도 놓칠 위험. 새 SKU 추가 시 메타 등록 누락을 초기에 포착하려면 WARN이 적절.
- (3) 로그 형식: `[IrisSDK] SKU meta missing for "<sku_id>", using default (has_baked_limbal=false, prefers_crl=false)`.

## 6.5 Gemini 완화안 채택 — **10/10 엄수 유지 (Claude 편향 경계)**

근거:
- (1) Claude는 P5 R1~R3에서 림발 정책을 3번 뒤집었고, R3에서 Gemini 완화안(9/10) 쪽으로 기울었다가 R4 실기기에서 "10/10 달성 가능, 단 SKU 1개만 메타 필요"로 돌아온 히스토리 — **R4 결과가 실측 사실**.
- (2) 9/10 완화안은 "1개 SKU는 자동감지 실패 허용"인데, 실기기에서 이미 10/10 달성했으니 완화는 불필요. Gemini가 "9/10 여지"라 R1에서 또 주장하면 **실기기 10/10 증거**로 반박.
- (3) Codex R3 엄격안(10/10)이 실기기 R4로 재확인됨. Codex 편향이 아니라 **팩트 일치**.

## 6.6 엔비_샤모 브라운 특별 처리 — **`prefers_graphic_outline` 플래그 + 셰이더 무분기**

근거:
- (1) 에셋 자체가 강한 outline 포함 → 셰이더 림발 수식 활성 시 이중 림발 발생. `prefers_graphic_outline=true` SKU는 **셰이더 림발 스킵** (uniform `uApplyLimbal = 0`).
- (2) 셰이더 내부 분기 없음 (uniform 레벨에서 on/off). `prefers_graphic_outline`은 `has_baked_limbal`과 별도 플래그 (전자는 "그래픽 자체가 outline 포함"이고, 후자는 "림발 링 텍스처가 이미 구워져 있음"). 용도 다름.
- (3) 효과: has_baked_limbal OR prefers_graphic_outline → uApplyLimbal = 0.

## 6.7 B10 신규 벤치 — **추가 금지 (W7 범위 밖 확정)**

근거:
- (1) 문서 §6.7에 이미 "Out of scope, Codex R4 리뷰 반영" 명시. 99 합의본이 채택 안 한 쟁점. 재공론화하지 않음.
- (2) W7 범위는 99 §1.2 C6 "림발 기본 ON + 메타 플래그" 내에서만. B10 같은 신규 벤치는 별도 제안 문서(P6-W10이나 Phase 7)에서.
- (3) Claude R1 편향 경계: W7이 작은 W이라 체감 상 "새 쟁점 추가 유혹"이 생기지만, 히스토리에서 3번 뒤집힘이 모두 이런 충동에서 비롯됐음. 확정.

---

## 요약 표

| 쟁점 | Claude 추천 |
|------|-------------|
| 6.1 자동감지 ROI | `[0.85, 1.0]` 유지 |
| 6.2 임계값 | 0.75 유지 |
| 6.3 메타 저장 | JSON `lens_meta.json` |
| 6.4 메타 누락 로그 | WARN |
| 6.5 완화안 9/10 | **거부, 10/10 엄수 (R4 실기기 증거)** |
| 6.6 엔비_샤모 | `prefers_graphic_outline` + uApplyLimbal=0 |
| 6.7 B10 신규 | 추가 금지 (W7 범위 밖) |

---

## 자기비판 (편향 경계)

- **림발 정책 3번 뒤집힘 맥락 재확인:** Claude는 R1(기본 ON)→R2(기본 OFF+메타 ON)→R3(9/10 완화)→R4(실기기 10/10 확인) 경로. 이번 W7 R1은 **R4 결과를 팩트로 유지**하는 입장. 새 뒤집기 유혹(예: "9/10이 유연함" 같은 말투) 발견 즉시 중단.
- **Codex 편향 경계:** 위 추천은 R4 Codex 엄격안과 일치하지만, 그건 Codex 편향이 아니라 실기기 실측 결과 반영이라는 점을 synthesis에서 명시.
- **Gemini 반응 대비:** Gemini가 "완화 여지" 주장하면 R4 문서(실기기 5 SKU 전수 통과)를 근거로 반박. 새 벤치(B10) 제안 나오면 §6.7 근거로 차단.
