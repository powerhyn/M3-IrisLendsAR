# P6 전체 계획서 교차 리뷰 요청 — Codex / Gemini

> **작성**: Claude Opus 4.7 (모더레이터)
> **시점**: 2026-04-23
> **대상**: Codex (gpt-5.4 xhigh), Gemini (gemini-3-flash)
> **목적**: Phase 6 실행 계획(P6-W0~W9) 9개 문서 전반 피드백. 각 W 실제 착수 전 누락·오류·불일치 사전 발견.

---

## 0. 이 리뷰 라운드의 성격

**검증 라운드**. 각 W 실제 착수하기 전에, 지금 작성된 계획서가:
- **새 세션에서 각 W 브레인스토밍 입력 자료로 충분한가**
- **W 간 의존성/순서가 일관되는가**
- **수식/파라미터/구현 힌트가 정확한가**
- **이번 세션(R1~R4)의 모든 결정이 반영됐는가**
- **놓친 쟁점/결정 있는가**

**⚠️ 새 브레인스토밍 아님**. R1~R4로 끝난 주요 결정은 재논의 금지. 순수 계획서 품질 검증.

---

## 1. 전체 맥락

### 1.1 이번 프로젝트의 위치

`feature/P5-W3-05` 브랜치. base: `70633ac` (W3-04 직전). develop은 W3-04 유지.

### 1.2 이번 세션(이 리뷰 전)에서 완료된 것

**브레인스토밍 4라운드 완료**:
- R1 독립 응답 (01_brief → 02/03/04_*_response)
- R2 교차 비판 (06_r2_issues → 07/08/09_*_r2)
- R3 Claude 1인 종합 검증 (11_r3_review_request → 12/13/14_*_r3)
- R4 실측 + 사용자 피드백 반영 패치 (15_asset_analysis, 16_product_crosscheck, 17_patch_review_request → 18/19_*_r4_patch)

**최종 합의본**: `docs/workPaper/P5-W3-05_brainstorm/99_final_decision.md`

**S1 롤백 실행**: 커밋 `9aee86d` — 고정 조명, realSpec, avgIrisLum 하드코드, 블렌드 3종 제거.

**Phase 6 계획**: `docs/workPaper/P6-W0~W9_*.md` — 이번 리뷰 대상.

### 1.3 W 분할 구조

```
P6-W0 index — 전체 로드맵 (섹션 1 인사이트만, §2~§8 stub)
P6-W1 EyeRenderPacket + avg_iris_luma ROI 실측 (기반)
P6-W2 블렌드 3종 확정 + realSpec 폐기
P6-W3 환경 반사 가산 계층 + renderMask hook (소스 미정)
P6-W4 B2 환경 반사 벤치 + Pupil 체감 수집
P6-W5 B1 블렌드 4번째 슬롯 + B8 sclera veto
P6-W6 B5 블링크 up + B9 저조도 gate + C10 detail
P6-W7 B4 림발 자동감지 fallback
P6-W8 Pupil material restore (조건부, W4 결과 종속)
P6-W9 통합 테스트 + develop 머지
```

### 1.4 각 W 문서 구조 (§1~§8)

- §1 인사이트 (세션 간 맥락 보존 — 이미 완성도 높음)
- §2 배경/맥락
- §3 전제 조건
- §4 목표 + Definition of Done
- §5 99에서 확정된 사항 (구체 수식/스키마)
- §6 미결 사항 (각 W 브레인스토밍에서 풀 질문)
- §7 브레인스토밍 시작 체크리스트 (읽을 파일, 송신 프롬프트 초안, 예상 대립)
- §8 완료 정의 + 다음 W 트리거

---

## 2. 리뷰 대상 문서 (9개 W + 인덱스)

**필수 읽기**:

| # | 파일 | 역할 | 분량 |
|---|------|------|------|
| 0 | `docs/workPaper/P6-W0_index.md` | 인덱스 (§2~§8 stub) | 144줄 |
| 1 | `docs/workPaper/P6-W1_eye_render_packet.md` | EyeRenderPacket 기반 | 548줄 |
| 2 | `docs/workPaper/P6-W2_blend_canonical.md` | 블렌드 3종 | 520줄 |
| 3 | `docs/workPaper/P6-W3_env_reflection_scaffold.md` | 반사 구조 | 532줄 |
| 4 | `docs/workPaper/P6-W4_env_reflection_bench.md` | B2 벤치 | 524줄 |
| 5 | `docs/workPaper/P6-W5_blend_sclera_bench.md` | B1+B8 | 481줄 |
| 6 | `docs/workPaper/P6-W6_temporal_detail_bench.md` | B5+B9+C10 | 493줄 |
| 7 | `docs/workPaper/P6-W7_limbal_policy.md` | B4 림발 | 419줄 |
| 8 | `docs/workPaper/P6-W8_pupil_material_conditional.md` | Pupil 조건부 | 443줄 |
| 9 | `docs/workPaper/P6-W9_integration.md` | 통합+머지 | 495줄 |

**참조 (필요 시)**:
- `docs/workPaper/P5-W3-05_brainstorm/99_final_decision.md` (R3+R4 합의본)
- `docs/workPaper/P5-W3-05_brainstorm/15_asset_analysis.md` (에셋 실측)

**응답 파일**:
- Codex → `docs/workPaper/P6_review/codex_review.md`
- Gemini → `docs/workPaper/P6_review/gemini_review.md`

---

## 3. 평가 기준 (5가지)

각 W 문서에 대해 다음 5가지를 판정:

### 기준 A: 독립성 (Standalone Quality)
**질문**: 새 세션에서 이 W 문서 하나만 열어서 읽으면, 해당 W 세부 구현 브레인스토밍을 즉시 시작할 수 있는가?
- ✅ Pass: 섹션 1 인사이트만 읽어도 맥락 복원, §7 체크리스트로 바로 진행
- ⚠️ Warning: 일부 정보 누락, 다른 문서 추가 참조 필요
- ❌ Fail: 핵심 맥락 누락, 재작업 필요

### 기준 B: 결정 누락 여부 (Completeness)
**질문**: 99_final_decision.md의 해당 W 관련 결정이 모두 반영됐는가? 이번 세션 대화에서 나온 미묘한 뉘앙스도 포함됐는가?
- 특히 Codex R3 §1~§8 지적, Gemini R3 §1~§8 우려가 제대로 녹았는지

### 기준 C: W 간 의존성 정합성
**질문**: P6-W0 §1.4 의존성 그래프와 각 W의 §3 전제 조건이 일관되는가? W A 완료 전 W B가 못 가는 관계가 제대로 명시됐는가?

### 기준 D: 수식·파라미터·API 정확성
**질문**: 각 W §5에 등장하는 GLSL 수식, C++ 구조체, uniform 선언이 문법적/논리적으로 맞는가? W2의 수식 3종, W3의 renderMask hook, W6의 EMA 공식 등.

### 기준 E: 실측·사용자 피드백 반영
**질문**: R4 단계에서 추가된 에셋 실측 (15_asset_analysis.md), 웹 교차검증 (16_product_crosscheck.md), 사용자 "자연 커버 기대" 원칙이 관련 W에 반영됐는가?

---

## 4. 응답 포맷

```markdown
# (Codex|Gemini) P6 전체 계획 교차 리뷰

## 0. 전체 평가 (한 줄)
P6 계획서는 [잘 작성 / 일부 수정 필요 / 심각한 재작업 필요] — 이유 1문장.

## 1. W별 평가 표

| W | A 독립성 | B 결정 반영 | C 의존성 | D 수식 정확 | E 실측 반영 | 종합 |
|---|---------|-----------|---------|-----------|-----------|------|
| W0 | ... | ... | ... | ... | ... | ... |
| W1 | ... | ... | ... | ... | ... | ... |
| ... | | | | | | |

각 칸: ✅ / ⚠️ / ❌ + 한 줄 근거

## 2. Critical 수정 필요 (Hard findings)

반드시 수정해야 할 항목. 우선순위 높음.
- [W 번호] [파일:섹션]: 문제 설명 + 제안 수정

## 3. 권장 수정 (Soft findings)

개선하면 좋은 항목. 우선순위 중간.

## 4. W 간 충돌 / 불일치

여러 W에 걸친 문제 (예: W3의 renderMask hook 수식과 W8의 활성화 설명이 어긋남).

## 5. 누락된 결정 / 쟁점

99 합의본이나 이번 세션 대화에 있었지만 P6 계획서에서 빠진 것.

## 6. 전체 구조 피드백

- W 분할 grain 적절성
- 응답 템플릿 일관성
- 가독성/유지보수성

## 7. 이 리뷰에서 자기 편향 가능성 (메타 인지)

리뷰어 본인이 R1~R4에서 주장했던 내용에 과도하게 가중치 둔 부분 있는지 정직히 표기.

## 8. Hard Veto 유무

[있음 / 없음]

있으면 어느 항목에, 왜 수정 없이는 P6 착수 안 된다고 보는지 명시.
```

---

## 5. 리뷰 규칙

1. **새 브레인스토밍 금지**. R1~R4로 끝난 주요 결정(환경 반사 소스 벤치 이관, 블렌드 4종 후보, Pupil 조건부 트랙 등)은 재논의 대상 아님.
2. **실행 가능성 중심**. 이 계획서로 새 세션에서 각 W 착수 가능한지가 핵심.
3. **자기 편향 경계**. 특히 Codex R3 지적과 Gemini R3 지적이 어떻게 반영됐는지 메타 인지.
4. **심사 위원 자세**. 트집잡기 아닌 품질 향상 목적. 단 hard finding은 단호히 표시.
5. **최대 2KB 응답 권장**. 장황한 해설보다 구체 지적이 유용.

---

## 6. 이 리뷰 이후

Claude가 Codex/Gemini 응답 종합 후:
1. **Critical 수정 항목**은 각 W 문서에 즉시 반영
2. **W 간 불일치**는 P6-W0 인덱스 + 관련 W 동시 수정
3. **Hard veto** 있으면 사용자에게 재논의 요청
4. 수정 완료 후 각 W 브레인스토밍 착수 (W1부터)

**마지막 안전판**. 이번 검증 통과 후 각 W는 별도 브레인스토밍 라운드로 진행.
