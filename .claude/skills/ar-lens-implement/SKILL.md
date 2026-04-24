---
name: ar-lens-implement
description: IrisLensSDK W 단위 구현 워크플로우. 브레인스토밍 완료된 W의 §5 확정 사항을 실제 코드로 옮기고 빌드·테스트·실기기 회귀·커밋까지. 브레인스토밍은 별도 스킬(`ar-lens-brainstorm`).
---

# AR Lens Implementation Workflow Skill

IrisLensSDK W 단위 구현 전담. **브레인스토밍 이후 단계**만 담당.

## Trigger Patterns

- "implement P6-W1" / "execute P6-W1" / "run P6-W1"
- "P6-W1 구현해줘" / "P6-W1 실행해줘"
- Legacy: "implement P1-W3-01" 등 (Phase 1 형식도 지원)

Mode options:
- `implement P6-W1 quick` — 최소 구현 + 빌드만
- `implement P6-W1 full` — 빌드 + 테스트 + 실기기 + 분할 커밋

## Prerequisite

**반드시** 다음 상태에서 호출:

1. 대상 W 문서(`docs/workPaper/P6-W{N}_*.md`)의 **§5 확정 사항에 실제 구현 가능한 결정들이 들어 있음**
2. §6 미결 사항이 **비어있거나, 남은 항목은 "구현 중 판정" 수준**
3. 만약 §6 미결이 많다면 먼저 `ar-lens-brainstorm` 스킬을 돌릴 것

브레인스토밍 미완료 상태에서 이 스킬 호출 시 → 사용자에게 경고 + `ar-lens-brainstorm` 권장.

## Workflow

### Step 1: Parse Task & Branch Setup

1. W 번호 파싱 (예: `P6-W1` → `{N}=1`)
2. 대상 문서 찾기: `docs/workPaper/P6-W{N}_*.md` (glob)
3. git 브랜치 확인/생성:
   ```bash
   git branch --show-current
   # 현재 통합 브랜치(feature/P6-Works)이면 W별 브랜치로 분기
   git checkout feature/P6-Works
   git pull
   git checkout -b feature/P6-W{N}  # 이미 있으면 체크아웃
   ```

### Step 2: Extract Implementation Plan

W 문서에서 추출:
- §4.1 Definition of Done 체크리스트
- §5 확정 사항 (수식/스키마/API/파일 위치)
- §7.6 구현 예상 소요 (시간 가이드)
- §8.2 커밋 전략 (분할 커밋 목록)

브레인스토밍 결과 반영 확인:
- 브레인스토밍 결과가 W 문서에 이미 반영됐으면 OK
- 반영 안 됐으면 `docs/workPaper/P6-W{N}_brainstorm/synthesis.md` 참조 후 **먼저 W 문서 업데이트**

### Step 3: Implementation

**Phase 6 프로젝트 규칙**:
- C++17 표준
- RAII, thread-safety, const-correctness, 이동 시맨틱
- `cpp/` 폴더 아래 코드 작성 시 `systems-programming:cpp-pro` 에이전트 활용 가능 (CLAUDE.md 규칙)
- 기존 Serena MCP 도구 우선 (`find_symbol`, `replace_symbol_body` 등)

파일별 단위로 구현 진행:
- 신규 파일: Write
- 기존 파일 수정: Edit (또는 serena symbol tool)
- 헤더 + 구현 분리
- 주석은 "왜" 필요 없으면 생략 (메모리 `user_work_style` 간결 선호)

### Step 4: Build Verification

**CLAUDE.md 지침 준수** — CLion 빌드 디렉토리 재사용:

```bash
cd cpp/cmake-build-debug
cmake --build . --parallel --target iris_sdk 2>&1 | tail -30
```

에러 발생 시:
- 컴파일 에러: 파일 수정 후 재빌드
- 링커 에러: symbol 누락 확인
- 경고만: 무시 가능 (단 새로 도입된 것만)

**TFLite 재다운로드 방지**: `-DIRIS_SDK_FETCH_TFLITE=OFF` 추가 (이미 cmake-build-debug 기존 설정이면 불필요).

### Step 5: Unit Test (Optional, full 모드만)

해당 W가 단위 테스트 추가 대상이면:
```bash
cmake --build . --target iris_sdk_tests
./bin/iris_sdk_tests --gtest_filter="*W{N}*"
```

`unit-testing:test-automator` 에이전트 활용 가능.

### Step 6: Real Device Regression (시각 영향 있는 W만)

- W1 (EyeRenderPacket): 회귀 없음 가정. 실기기 건너뛰기 OK.
- W2~W8 (셰이더 변경): **실기기 1대 필수**.
- W9 (통합): 3 tier 모두.

```bash
# Android APK 빌드 (필요 시)
./scripts/build_android.sh
# 실기기 설치 + 렌즈 토글 확인
```

실기기 결과 확인:
- 해당 W의 시각 변화 (예: 블렌드 결과, 환경 반사, 림발 표시 등)
- 이전 상태(feature/P6-Works) 대비 회귀 없음
- 메모리 `feedback_qualitative_device_judgment` 원칙: **정성 체감 기반 판단**

### Step 7: Commit Strategy

W 문서 §8.2 커밋 전략 따름. 예시 (W1):

```bash
git add cpp/include/iris_sdk/gpu/eye_render_packet.h
git commit -m "feat(gpu-lens): P6-W1 EyeRenderPacket 구조체 정의"

git add cpp/src/gpu/eye_render_packet_adapter.*
git commit -m "feat(gpu-lens): P6-W1 IrisResult → EyeRenderPacket 어댑터"

git add cpp/src/gpu/gpu_lens_renderer.cpp cpp/include/iris_sdk/gpu/gpu_lens_renderer.h
git commit -m "refactor(gpu-lens): P6-W1 GPULensRenderer render API → EyeRenderPacket"

git add cpp/src/gpu/gpu_lens_renderer.cpp
git commit -m "feat(gpu-lens): P6-W1 avg_iris_luma masked ROI 평균 self-measure"
```

**커밋 메시지 규칙** (프로젝트 CLAUDE.md):
- Conventional Commits 접두사 영어 (feat:, refactor:, fix:, docs:, test:)
- 설명 한글
- `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` 추가

### Step 8: Push + PR

```bash
git push -u origin feature/P6-W{N}
gh pr create --base feature/P6-Works --title "P6-W{N}: {W 제목}" --body "..."
```

PR 본문 템플릿:
```markdown
## Summary
- P6-W{N} 구현 완료
- 주요 변경: ...

## 체크리스트 (W 문서 §4.1)
- [x] 항목 1
- [x] 항목 2
...

## 벤치 결과 (해당 W에 벤치 있으면)
- ...

## 다음 W
- P6-W{N+1}
```

### Step 9: Document Update + Completion Summary

W 문서 업데이트:
- 상태: "인사이트 작성 완료" → "구현 완료"
- §8.1 Definition of Done 체크리스트 채우기 ([x])
- §8.2 실제 커밋 해시 기록

완료 보고:
```
✅ P6-W{N} 구현 완료

📊 요약:
- 신규 파일: {목록}
- 수정 파일: {목록}
- 커밋 수: {N}개
- 빌드: ✅ 통과
- 실기기 회귀: ✅ 없음 (해당 시)

📝 다음 단계:
1. PR 리뷰 대기
2. feature/P6-Works 머지 후 다음 W:
   - `ar-lens-brainstorm P6-W{N+1}` (브레인스토밍)
   - 또는 이미 완료됐다면 `implement P6-W{N+1}`
```

## Mode Details

### Default (standard)
Step 1~9 모두 수행.

### Quick
Step 1~4만 (+ Step 7 간단 커밋). 실기기 회귀/테스트 건너뜀. **임시 개발 빌드 확인용**.

### Full
Standard + Step 5 필수 + Step 6 상세 (3 tier 시도 권장).

## Common Issues

### 빌드 실패

```
❌ Build failed: {에러 요약}

가능한 원인:
- 헤더 누락 → #include 확인
- 네임스페이스 충돌 → using 제거
- C++17 기능 미지원 → 컴파일러 버전 확인

💡 대응:
1. 에러 메시지 풀텍스트 확인
2. Serena MCP로 관련 symbol 추적
3. debugging-toolkit:debugger 에이전트 호출
```

### 의존성 미충족

```
⚠️ P6-W{N} 선행 W 미완료

W 문서 §3 전제 조건 확인:
- {미완료 W 번호}

💡 먼저 완료 필요:
implement P6-W{M}
```

### 실기기 회귀 발견

```
⚠️ 회귀 감지

시각 변화:
- {관찰 내용}

💡 대응:
1. git diff 확인 (어느 커밋에서 변화?)
2. 의도된 변화인가? (W 문서 §4 목표 비교)
3. 의도 밖이면 revert + 재구현
```

## Project Context (Phase 6+)

**Branch Structure**:
```
develop (W3-04 포함)
└── feature/P6-Works (Phase 6 통합)
    ├── feature/P6-W1
    ├── feature/P6-W2
    └── ...
```

**Performance Targets**:
- FPS: ≥ 30 (MID tier), ≥ 60 (HIGH)
- Detection Latency: < 33ms
- Memory: < 100MB
- SDK Size: < 20MB

**Tech Stack**:
- C++17
- OpenGL ES 3.1 (`#version 310 es` 셰이더)
- MediaPipe + TFLite
- OpenCV 4.x
- CMake 3.18+ (Ninja)

**Related Skills**:
- **`ar-lens-brainstorm`** — W 브레인스토밍 라운드 담당. 이 스킬 전에 돌릴 것.

## Do NOT

- **브레인스토밍 혼입**: §6 미결 있으면 먼저 brainstorm 스킬 호출. 이 스킬은 구현만.
- **W 문서 §5 우회**: 확정 사항 없이 "알아서 구현" 금지. 계획 따를 것.
- **테스트 생략 (full 모드)**: 시각 영향 있는 W는 반드시 실기기 확인.
- **develop 직접 커밋**: feature/P6-W{N} 브랜치에서만 작업.
- **여러 W 동시 구현**: 한 번의 호출 = 한 W. 여러 W 섞지 말 것.

## Legacy Phase 1 Support

`implement P1-Wx-xx` 형식도 계속 지원. 단 Phase 1 전용 플로우(standard/performance/quick/debug 모드)는 별도 처리:

- P1 태스크는 `docs/workPaper/P1-Wx-xx_*.md`에서 찾음
- W 브레인스토밍 불필요 (이미 완료된 Phase)
- Phase 1 문서는 v1.0.0 시대 구조라 §1~§8 템플릿과 다름 — "목표/산출물/검증 기준/선행 조건" 섹션 파싱

## Notes

- Phase 6+부터 브레인스토밍과 구현이 분리. 이 스킬은 구현만.
- 각 W 완료 시 PR 단위로 머지 → feature/P6-Works 누적
- 전체 Phase 6 완료 후 feature/P6-Works → develop 한 번에 머지
- 구체 수식/파일 경로는 각 W 문서 §5에 있음. 이 스킬은 "어떻게 구현하나"의 메타 프로세스.

---

**Version:** 2.0.0
**Project:** IrisLensSDK Phase 6+
**Updated:** 2026-04-23
**Related Skill:** `ar-lens-brainstorm` (이 스킬 호출 전 W 브레인스토밍 담당)
