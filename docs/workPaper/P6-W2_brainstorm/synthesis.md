# P6-W2 브레인스토밍 R1 — 종합

**작성일:** 2026-04-24
**참여:** Codex (gpt-5.4 xhigh), Gemini (gemini-3-flash), Claude (opus-4-7)
**대상:** `docs/workPaper/P6-W2_blend_canonical.md` §6.1 ~ §6.7

---

## 1. 합의 현황 (3/3 동의 → 닫힘)

| 쟁점 | 확정 |
|------|------|
| **6.2** ID 3/4/6 빈 슬롯 | **유지 + fallback 처리** (재번호화 금지) |
| **6.4** LUMA_COEFFS | **Rec.709 linear 계수 (0.2126, 0.7152, 0.0722) 전 경로 통일**. Shader + CPU(W1) 동일 상수 사용, linear-space 강제 |
| **6.5** realSpec 복원 경로 | **완전 삭제** (Git history로 복원 가능) |
| **6.6** 기본값 블렌드 모드 | **TintLinearV2 (ID=5)를 canonical default로 확정** (99 §1.1 D6 유지) |
| **6.7** Android Demo UI | **W9 integration에서 정리. W2 범위 밖.** |

## 2. 다수결 결정 (2/3 동의)

### 6.1 ColorReplaceLinear 구현 시점 — **옵션 B 채택 (Gemini+Claude 다수)**
- Gemini, Claude: **옵션 B** — W2에서 함수 정의 + ID 7 분기 활성 (W5 B1 벤치 준비).
- Codex: 옵션 A — W2에서 함수만. ID 7 분기 활성은 B1 결론 선반영처럼 보임.
- **판정 근거:** W5 B1 벤치에서 "Normal vs CRL" 1:1 비교하려면 ID 7이 활성되어 있어야 벤치용 토글 오버헤드 없음. 다수 선택.
- **Codex 지적 수용 (완화 조건):** ID 7 분기 구현 시 주석 명시 "`// W2 활성 — W5 B1 벤치 대상, 채택 확정 아님`". W5 결과가 "CRL 제거"면 함수 + 분기 모두 제거하는 롤백 커밋을 W5 단계에서 수행.

### 6.3 Fallback default — **옵션 B (TintLinearV2) 채택 (Gemini+Claude 다수)**
- Gemini, Claude: **옵션 B** — TintLinearV2 fallback.
- Codex: 옵션 A — Normal fallback (단순 baseline, 디버그 명료성).
- **판정 근거:** "빈 ID = 기본 모드"가 의미적으로 일관. Normal은 B1 벤치 재평가 대상이라 fallback에 묶으면 B1 결과에 따라 fallback 깨짐.
- **Codex 지적 수용 (완화 조건):** invalid ID 전송 시 렌더러에서 **경고 로그 1회 출력** (debug 빌드 한정) → 디버그 명료성 보강. 로그 형식: `[IrisSDK] Unknown blend ID=<n>, falling back to TintLinearV2`.

## 3. 미결 / 후속 확인

**없음.** 7개 쟁점 모두 R1에서 결론.

- **6.5 realSpec 문서 보존(Claude 부가 제안):** 3/3 합의는 "완전 삭제"이고 문서 보존은 강제 조건 아님. 다만 W4 B2 실패 대비로 realSpec 수식을 `docs/workPaper/P6-W2_brainstorm/realSpec_archive.md`에 **선택적으로** 보존하는 것을 권장. Git history + archive 문서 이중으로 복원 경로 확보. 구현 단계에서 판단.

## 4. 편향 체크

- **Claude 편향:** 6.1, 6.3에서 Claude 의견이 Gemini와 동일(B/B). Codex가 반대편에 혼자 남았지만 두 사안 모두 Codex의 "절차적/디버그 관점"이 유효한 지적이라 **완화 조건**으로 수용 → "다수결에 기계적으로 기대지 않음" 확인.
- **Claude-Gemini 동반 가능성 주의:** 다음 W부터 Claude 응답을 숨김 파일로 저장하여 Codex/Gemini가 Claude 응답을 "형식 참고"로 읽는 편향 경로 차단됨. W1 경험을 W2에 적용한 결과 Codex/Gemini가 더 독립적으로 작성됨.

## 5. W2 문서 §5에 반영할 항목

현재 §5는 5.1~5.6(6항목)까지. 새로 5.7 ~ 5.12 추가.

- **§5.7** CRL 구현 시점: 옵션 B. W2에서 `blendColorReplaceLinear` 함수 정의 + ID 7 정식 분기 활성. 주석 `// W2 활성 — W5 B1 벤치 대상, 채택 확정 아님`. W5 결과에 따라 함수+분기 제거 롤백 커밋 가능.
- **§5.8** ID 3/4/6 빈 슬롯: 재번호화 금지. Fallback 분기로 흡수.
- **§5.9** Fallback default: **TintLinearV2** (ID=5). Debug 빌드에서 invalid ID 경고 로그 1회 출력.
- **§5.10** LUMA_COEFFS: Rec.709 linear 계수 (0.2126, 0.7152, 0.0722)로 shader + CPU(W1) 통일. Linear-space 연산 강제. 테스트 케이스 shader vs CPU 오차 1% 이내 검증.
- **§5.11** realSpec: **완전 삭제**. Git history로 복원 경로 확보. 선택적 archive: `docs/workPaper/P6-W2_brainstorm/realSpec_archive.md`.
- **§5.12** 기본값 블렌드 모드: TintLinearV2 (ID=5) canonical default. Android demo 초기값 TintLinearV2 ID로 설정. C API 문서(sdk_api.h 주석)에 기본값 명시.

## 6. 다음 액션

1. W2 문서 §5 / §6 업데이트 (R1 결과 반영).
2. Task #2 완료 표시.
3. W3 브레인스토밍 착수.
4. W2 구현은 별도 스킬 `ar-lens-implement`로 이관.

## 7. R2 필요 여부

**R2 불필요.** Hard veto 없음. 2개 다수결(6.1, 6.3) 모두 Codex 지적을 완화 조건으로 수용하여 절차적 리스크 흡수. 나머지는 합의.
