---
name: ar-lens-brainstorm
description: Phase 6 이후 각 W 시작 시 Codex/Gemini tmux 팬과 멀티라운드 교차 브레인스토밍 오케스트레이션. 송신/수집/종합 자동화.
---

# AR Lens Brainstorm Workflow Skill

IrisLensSDK의 Phase 6+ W별 독립 브레인스토밍 라운드를 자동화. Claude 1인 결정을 피하기 위해 Codex(gpt-5.4 xhigh)와 Gemini(gemini-3-flash)를 tmux 팬으로 교차 질의.

## Trigger Patterns

- "brainstorm P6-W1" / "P6-W1 brainstorm"
- "P6-W1 브레인스토밍 돌려줘" / "P6-W1 브레인스토밍 시작"
- "start P6-W{N} brainstorm"
- "R2 돌려줘 P6-W1" (추가 라운드 요청)

## Phase 6 범위

- **대상**: `docs/workPaper/P6-W{N}_*.md` 각 W 문서
- **전제**: 해당 W 문서에 §1 인사이트 + §5 확정 사항 + §6 미결 사항 + §7 브레인스토밍 체크리스트가 **이미 존재**.
- **관련 메모리** (자동 로드됨, 행동에 반영):
  - `feedback_multi_ai_orchestration_bias` — Claude 1인 결정 금지
  - `feedback_real_data_first` — 실측 데이터 우선
  - `feedback_qualitative_device_judgment` — 정성 체감 > 정량 수치

## Workflow

### Step 1: Preflight Check

1. 현재 git 브랜치 확인. 예상: `feature/P6-Works` 또는 `feature/P6-W{N}`
2. 대상 W 문서 존재 확인: `docs/workPaper/P6-W{N}_*.md`
3. tmux 팬 확인: Codex 팬 (`20_CGG-Backend:3.1`), Gemini 팬 (`20_CGG-Backend:3.2`)
   - 실행 디렉토리가 프로젝트 루트인지 확인 (scripts 같은 하위면 재시작 필요)
   - 프롬프트 대기 상태인지 확인 (다른 작업 중이면 중단 확인)

### Step 2: Read W Document

- `docs/workPaper/P6-W{N}_*.md` 전체 읽기
- 특히 §6 미결 사항, §7.1 읽을 파일 목록, §7.2 송신 프롬프트 초안 확인

### Step 3: Draft Brainstorm Request

`P6-W{N}_brainstorm/` 폴더 생성. 송신 프롬프트 기반:

```
@docs/workPaper/P6-W{N}_*.md 읽고, 섹션 6 미결 사항에 대해 각자 입장
정리 후 docs/workPaper/P6-W{N}_brainstorm/{codex|gemini}_w{N}.md로 저장.

규칙:
- 새 쟁점 제기 금지
- 각 미결 항목 "추천 + 근거 1~2줄"
- 상대 모델 R1~R4 입장 재확인 필요 시 원문 인용
- 한국어
```

W 문서 §7.2에 이미 초안이 있으면 그걸 기본으로 사용하고 조정.

### Step 4: Parallel Send to Both Panes

⚠️ **중요 규칙** (이전 세션 경험):

1. **텍스트와 Enter를 분리**. 한 번의 send-keys로 보내면 TUI paste 버퍼가 Enter를 텍스트 일부로 해석해 제출 안 됨.
   ```bash
   tmux send-keys -t <pane> "프롬프트 텍스트"
   sleep 0.5
   tmux send-keys -t <pane> Enter
   ```
2. Codex와 Gemini에 **동일 프롬프트** 병렬 송신. 응답 파일명만 다르게:
   - Codex → `codex_w{N}.md`
   - Gemini → `gemini_w{N}.md`

### Step 5: Background Wait

```bash
until [ -f docs/workPaper/P6-W{N}_brainstorm/codex_w{N}.md ] \
   && [ -f docs/workPaper/P6-W{N}_brainstorm/gemini_w{N}.md ]; do
    sleep 5
done
```

`run_in_background: true`로 실행. 완료 시 알림.

### Step 6: Claude Self-Response (병렬 작성)

응답 대기 동안 Claude도 **자체 응답** 작성: `claude_w{N}.md`

원칙:
- **모더레이터 자임 금지**. 자기 의견을 평범한 한 모델의 의견으로 제시.
- R4에서 반영된 Claude 편향 자기비판 참조 — 타 모델 논리에 과도 쏠림 경계
- 메모리 `feedback_multi_ai_orchestration_bias` 적용

### Step 7: Handle Common Issues

**Gemini 파일 저장 안 됨**:
- 승인 대기 상태일 수 있음. 팬 캡처 확인: "Allow once" 프롬프트 떠있으면 Enter 보냄.
  ```bash
  tmux send-keys -t <gemini-pane> Enter
  ```

**Codex "stream disconnected before completion"**:
- OpenAI capacity 에러. 재시도:
  ```bash
  tmux send-keys -t <codex-pane> "이전에 서버 에러로 중단됐어. 다시 시도: {원래 프롬프트}"
  sleep 0.5
  tmux send-keys -t <codex-pane> Enter
  ```
- 반복 실패 시 사용자에게 보고 후 R2 잠시 연기 제안

**Gemini workspace 제한** (이전에 scripts 디렉토리 문제):
- workspace가 상위 디렉토리 접근 거부 시 Gemini CLI 재시작 (`/quit` → `cd 루트` → `gemini`)

### Step 8: Synthesize Responses

세 응답 파일 읽고 종합:

1. **합의된 것** (3/3 동의) → 닫힘. §6 미결 → §5 확정으로 이동 제안.
2. **부분 합의** (2/3 동의) → Claude가 다수결 판정 + 소수 의견 별도 기록. 사용자 최종 판단 제안.
3. **3분립** → 실기기 벤치 이관 검토 or R2 추가 라운드 제안.

종합 문서: `P6-W{N}_brainstorm/synthesis.md` 또는 W 문서 §6 직접 업데이트.

### Step 9: Present to User + Next Action Proposal

```
🎯 P6-W{N} 브레인스토밍 R1 완료

📊 결과:
- 닫힌 쟁점: {N}개 (합의)
- 다수결 결정: {N}개
- 미결/실기기 이관: {N}개

⚠️ 주의:
- Hard veto 있음/없음
- R2 필요 여부

💡 다음 단계:
1. 종합 내용 검토 후 §5 확정 이동 승인
2. (선택) R2 추가 라운드
3. 구현 착수 → `ar-lens-implement` 스킬 호출
```

## Round Management

### R1 (기본)
- 각 모델 독립 응답. Claude도 동등 참여.
- 모든 브레인스토밍의 첫 라운드.

### R2 (교차 비판, 필요 시)
- R1 응답 3개를 모든 모델이 읽음.
- 자기 R1 입장 유지/수정/철회 + 상대 모델 논리 정면 평가.
- 트리거: "R2 돌려줘 P6-W{N}" 또는 Claude 판정으로 "3분립 심각" 시 제안.

### R3+ (rare)
- R2에서도 미수렴 시. Claude 종합 초안을 다른 모델이 비판.
- P5-W3-05 브레인스토밍처럼 full orchestration. **지양** (시간 비용 크고 W별로 이런 수준 필요 거의 없음).

## Output Structure

```
docs/workPaper/P6-W{N}_brainstorm/
├── codex_w{N}.md         R1 Codex 응답
├── gemini_w{N}.md        R1 Gemini 응답
├── claude_w{N}.md        R1 Claude 응답
├── synthesis.md          Claude 종합 (편향 경계)
├── (R2 시) r2_issues.md
├── (R2 시) *_w{N}_r2.md
└── (R3 시) ...
```

## Integration with `ar-lens-implement`

브레인스토밍 완료 후 구현 착수:

1. 종합 결과를 W 문서 §5로 반영 ("99에서 확정된 사항"에 새 결정 추가)
2. §6 미결 사항 업데이트 (닫힌 것 제거)
3. 사용자에게 `implement P6-W{N}` 제안

**브레인스토밍과 구현은 완전히 별도 스킬**. 이 스킬은 브레인스토밍 라운드만 담당.

## Do NOT

- **Claude 단독 결정**: 세 모델 응답 모두 받기 전에 "내가 이미 알고 있으니까"라며 응답 대기 생략 금지.
- **새 쟁점 제기**: 각 W 문서 §6 범위 외 질문을 Codex/Gemini에 던지지 않음. 새 쟁점은 별도 W로 처리.
- **구현 혼입**: 이 스킬은 브레인스토밍만. 코드 변경 금지.
- **W 간 병합 금지**: 한 호출에 두 W 동시 브레인스토밍 안 함. 하나씩.
- **모델 응답 조작**: "이렇게 답해줘" 같은 가이드 주입 금지. 독립 응답 보장.

## Notes

- 이 스킬은 `P6-W1 브레인스토밍 돌려줘` 같은 자연어 트리거에 반응.
- 실제 브레인스토밍 소요: R1당 10~30분 (응답 대기 포함). Claude 자체 작성 5~10분.
- Codex/Gemini 응답 품질은 각 W 문서 §7 체크리스트 품질에 직결. 미리 잘 써야 함.
- 메모리 자동 로드되므로 명시적 참조 불필요. 단 새 피드백 발견 시 저장 권장.

---

**Version:** 1.0.0
**Project:** IrisLensSDK Phase 6+
**Created:** 2026-04-23
**Related Skill:** `ar-lens-implement` (브레인스토밍 완료 후 구현 담당)
