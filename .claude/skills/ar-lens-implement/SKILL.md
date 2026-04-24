---
name: ar-lens-implement
description: IrisLensSDK W 단위 구현 워크플로우. 브레인스토밍 완료된 W의 §5 확정 사항을 실제 코드로 옮기고 빌드·테스트·실기기 회귀·분할 커밋·PR까지. 브레인스토밍은 별도 스킬(`ar-lens-brainstorm`).
---

# AR Lens Implementation Workflow Skill

IrisLensSDK W 단위 구현 전담. **브레인스토밍 이후 단계**만 담당. Claude + cpp-pro 에이전트 + Serena MCP 도구를 활용한 자동화된 구현 파이프라인.

## Trigger Patterns

**Phase 6+ (현행 주력)**:
- "implement P6-W1" / "execute P6-W1" / "run P6-W1"
- "P6-W1 구현해줘" / "P6-W1 실행해줘"

**Mode options**:
- `implement P6-W1` — default (Step 1~9 모두)
- `implement P6-W1 quick` — 최소 구현 + 빌드만 (Step 1~4, 7)
- `implement P6-W1 full` — 전체 + 단위 테스트 + 3 tier 실기기 검증
- `implement P6-W1 debug` — 문제 있는 W 재작업 시 (root cause 추적)

**Legacy Phase 1**:
- `implement P1-Wx-xx` 계속 지원. 단 Phase 1 문서 구조(§목표/§산출물/§검증 기준)가 P6 §1~§8 템플릿과 달라 파싱 분기.

## Prerequisite Check

**반드시** 다음 상태에서 호출:

1. ✅ 대상 W 문서(`docs/workPaper/P6-W{N}_*.md`)의 **§5 확정 사항에 실제 구현 가능한 결정들**이 들어 있음
2. ✅ §6 미결 사항이 **비어있거나, 남은 항목은 "구현 중 판정" 수준**
3. ✅ 선행 W 완료 (§3 전제 조건 참조)

브레인스토밍 미완료 시 → **중단 + 사용자에게 `ar-lens-brainstorm P6-W{N}` 권장**.

```
⚠️ Preflight Check Failed

P6-W{N} §6 "미결 사항"이 X개 남음:
- §6.1: {내용}
- §6.3: {내용}

💡 먼저 브레인스토밍 필요:
   brainstorm P6-W{N}

결정 확정 후 다시 implement 호출.
```

## Workflow

### Step 1: Parse Task & Branch Setup

```bash
# 브랜치 확인
CURRENT=$(git branch --show-current)
echo "Current: $CURRENT"

# 통합 브랜치에 있으면 W 브랜치로 분기
if [[ "$CURRENT" == "feature/P6-Works" ]]; then
    git pull
    git checkout -b feature/P6-W{N}   # 이미 존재 시 checkout만
fi

# 문서 로케이트
find docs/workPaper -name "P6-W{N}_*.md"
```

**브랜치 네이밍 규칙**:
- 짧은 형식: `feature/P6-W1`
- 명확한 형식: `feature/P6-W1-eye-render-packet` (W 문서 파일명 슬러그 활용)
- 사용자 선호에 맞춰 선택

### Step 2: Extract Implementation Plan

W 문서에서 구현 계획 파싱:

- **§4 목표 + §4.1 Definition of Done** — 완료 체크리스트
- **§5 99에서 확정된 사항** — 수식/스키마/API/파일 위치
- **§7.6 구현 예상 소요** — 시간 가이드
- **§8.2 커밋 전략** — 분할 커밋 목록

**브레인스토밍 결과 반영 확인**:
```bash
# 브레인스토밍 결과물 존재 확인
ls docs/workPaper/P6-W{N}_brainstorm/ 2>/dev/null
```
- 존재하고 결과가 W 문서에 반영됐으면 진행
- 결과가 반영 안 됐으면 **먼저 W 문서 §5/§6 업데이트** 후 진행

### Step 3: Implementation

**Phase 6 프로젝트 규칙** (CLAUDE.md 준수):

1. **C++ 작업 (`cpp/` 폴더 하위)**: `systems-programming:cpp-pro` 에이전트와 병용
   - C++17 표준
   - RAII, thread-safety, const-correctness, 이동 시맨틱
   - 헤더 + 구현 분리
2. **Serena MCP 도구 우선**:
   - 파일 구조 파악: `get_symbols_overview`
   - 클래스/메서드 찾기: `find_symbol`
   - 메서드 수정: `replace_symbol_body`
   - 심볼 이름 변경: `rename_symbol`
   - 참조 추적: `find_referencing_symbols`
3. **기본 도구 예외**: 설정 파일, YAML, 마크다운은 Read/Edit 사용
4. **주석 최소화**: "왜"가 명확하지 않으면 생략 (메모리 `user_work_style`)

**구현 순서** (권장):
1. 헤더 파일 정의 (structs, classes, API)
2. 구현 파일 (.cpp) 작성
3. 필요 시 새 uniform location / CMakeLists 업데이트
4. 어댑터/연동 레이어 (W1처럼 계약 변경 시)

**코드 예시 — P6-W1 EyeRenderPacket**:

```cpp
// cpp/include/iris_sdk/gpu/eye_render_packet.h
#pragma once

#include <cstdint>
#include <optional>
#include <glm/glm.hpp>

namespace iris_sdk::gpu {

struct EyeRenderPacket {
    // 필수
    glm::vec2 iris_center_norm;
    float     iris_radius_norm;
    glm::vec2 ellipse_center;
    glm::vec3 ellipse_radii;      // (rxInner, rxOuter, ry)
    float     ellipse_rotation;
    float     eye_top;
    float     eye_bottom;
    float     visibility;         // 0.0~1.0
    uint64_t  timestamp_ms;

    // 선택 (없으면 기능 자동 off)
    std::optional<glm::vec2> pupil_center_norm;
    std::optional<glm::vec2> head_pose_yaw_roll;
    std::optional<glm::vec3> reflection_dir;
    std::optional<float>     avg_iris_luma;
    std::optional<float>     eye_depth_mm;
    std::optional<float>     render_confidence;
};

} // namespace iris_sdk::gpu
```

**코드 예시 — avg_iris_luma ROI 측정 (W1)**:

```cpp
// cpp/src/gpu/gpu_lens_renderer.cpp
float GPULensRenderer::measureAvgIrisLuma(
    const uint8_t* frame_rgb,
    int width, int height,
    const EyeRenderPacket& packet
) {
    // iris 중심 ± 0.65 * iris_radius 범위 sRGB 공간 평균 luma
    // LUMA_COEFFS = Rec.601 sRGB (0.299, 0.587, 0.114)
    float cx_px = packet.iris_center_norm.x * width;
    float cy_px = packet.iris_center_norm.y * height;
    float r_px = packet.iris_radius_norm * width * 0.65f;

    float luma_sum = 0.0f;
    int count = 0;
    for (int y = std::max(0, (int)(cy_px - r_px)); 
         y < std::min(height, (int)(cy_px + r_px)); ++y) {
        for (int x = std::max(0, (int)(cx_px - r_px));
             x < std::min(width, (int)(cx_px + r_px)); ++x) {
            float dx = x - cx_px, dy = y - cy_px;
            if (dx*dx + dy*dy <= r_px*r_px) {
                int idx = (y * width + x) * 3;
                float luma = frame_rgb[idx] * 0.299f / 255.0f
                           + frame_rgb[idx+1] * 0.587f / 255.0f
                           + frame_rgb[idx+2] * 0.114f / 255.0f;
                luma_sum += luma;
                ++count;
            }
        }
    }

    if (count == 0) return last_valid_luma_;  // hold fallback
    last_valid_luma_ = luma_sum / count;
    return last_valid_luma_;
}
```

### Step 4: Build Verification

**CLAUDE.md 지침** — CLion 빌드 디렉토리 재사용:

```bash
cd cpp/cmake-build-debug
cmake --build . --parallel --target iris_sdk 2>&1 | tail -30
```

**TFLite 재다운로드 방지**:
```bash
cmake .. -DIRIS_SDK_FETCH_TFLITE=OFF   # 최초 1회만 필요. 이후엔 cached.
cmake --build . --parallel
```

**빌드 에러 대응**:
- **컴파일 에러**: 파일 수정 후 재빌드. 에러 메시지 전체 확인.
- **링커 에러**: symbol 누락 → Serena `find_symbol`로 정의 추적
- **undefined reference**: CMakeLists의 타깃 소스에 새 파일 추가됐는지 확인
- **경고만**: 무시 OK (기존 경고 다수). 단 새로 도입된 것만 점검.

### Step 5: Unit Test (Full 모드만)

해당 W의 테스트 대상이면:

```bash
# 테스트 빌드
cd cpp/cmake-build-debug
cmake --build . --target iris_sdk_tests

# W별 필터링 실행
./bin/iris_sdk_tests --gtest_filter="*W{N}*"
# 예: ./bin/iris_sdk_tests --gtest_filter="*EyeRenderPacket*"
```

테스트 생성 시 `unit-testing:test-automator` 에이전트 활용 가능.

**테스트 항목** (예: W1):
- EyeRenderPacket 구조체 기본/복사/move 생성
- Optional 필드 has_value/value_or 동작
- 어댑터 IrisResult → EyeRenderPacket 변환 정확성
- avg_iris_luma 측정 edge case (0 픽셀, 경계, 범위 초과)
- Fallback 체인 (packet → self-measure → hold → default)

### Step 6: Real Device Regression

**W별 실기기 필요성**:

| W | 실기기 필수? | 이유 |
|---|-------------|------|
| W1 EyeRenderPacket | ❌ (default), ✅ (full) | 계약 변경, 시각 차이 없음 |
| W2 블렌드 | ✅ | 시각 직접 영향 |
| W3 반사 스캐폴드 | ❌ (default) | no-op 상태, 회귀 없음 가정 |
| W4 반사 벤치 | ✅ (3 환경+) | 벤치 자체가 실기기 필수 |
| W5 B1+B8 | ✅ | 블렌드/Sclera 시각 |
| W6 B5+B9 | ✅ (블링크+저조도) | 시각 + 시간적 |
| W7 림발 | ✅ | 림발 시각 확인 |
| W8 Pupil | ✅ | Pupil 재질 확인 |
| W9 통합 | ✅ 3 tier (HIGH/MID/LOW) | 최종 통합 |

**실기기 테스트 절차**:
```bash
# Android APK 빌드
./scripts/build_android.sh

# 설치 (연결된 디바이스 1개 가정)
adb install -r android/demo-app/build/outputs/apk/debug/demo-app-debug.apk

# 앱 실행 후 확인:
# - 해당 W의 시각 변화가 의도대로 나타나는지
# - 이전 상태(feature/P6-Works HEAD) 대비 회귀 없음
# - 메모리 feedback_qualitative_device_judgment: 정성 체감 기반
```

**회귀 발견 시**:
```
⚠️ Visual Regression Detected

관찰: {구체 증상}

💡 진단 순서:
1. git diff HEAD~1 -- cpp/ 로 최근 변경 확인
2. 의도된 변화인지 W 문서 §4 목표와 비교
3. 의도 밖이면:
   - 로컬 revert: git checkout HEAD~1 -- <file>
   - 재구현 or 부분 revert

4. debugging-toolkit:debugger 에이전트 호출 고려
```

### Step 7: Commit Strategy

**W 문서 §8.2 커밋 전략** 엄격 따름. 커밋 메시지 규칙 (프로젝트 CLAUDE.md):
- Conventional Commits 접두사 **영어** (`feat:`, `refactor:`, `fix:`, `docs:`, `test:`)
- 설명 **한글**
- Co-Authored-By 추가

**예시 — P6-W1 분할 커밋**:

```bash
# 커밋 1: 구조체 정의
git add cpp/include/iris_sdk/gpu/eye_render_packet.h
git commit -m "$(cat <<'EOF'
feat(gpu-lens): P6-W1 EyeRenderPacket 구조체 정의

99 §1.3 스키마 기반 내부 계약 구조체.
필수 7필드 + optional 6필드 (pupil_center, head_pose_yaw_roll 등).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"

# 커밋 2: 어댑터
git add cpp/src/gpu/eye_render_packet_adapter.* \
        cpp/include/iris_sdk/gpu/eye_render_packet_adapter.h
git commit -m "feat(gpu-lens): P6-W1 IrisResult → EyeRenderPacket 어댑터 레이어"

# 커밋 3: GPULensRenderer API 전환
git add cpp/src/gpu/gpu_lens_renderer.cpp cpp/include/iris_sdk/gpu/gpu_lens_renderer.h
git commit -m "refactor(gpu-lens): P6-W1 GPULensRenderer render() → EyeRenderPacket 입력"

# 커밋 4: avg_iris_luma 측정
git add cpp/src/gpu/gpu_lens_renderer.cpp
git commit -m "feat(gpu-lens): P6-W1 avg_iris_luma masked ROI 평균 self-measure + fallback 체인"
```

**커밋 가이드**:
- 한 커밋 = 한 논리 단위 (구조체 / 어댑터 / API 변경 / 구현 로직)
- 파일 크로스 경계 피함
- 각 커밋 빌드 통과 상태 유지
- Push 전 `git log --oneline`로 커밋 순서 검토

### Step 8: Push + PR

```bash
git push -u origin feature/P6-W{N}

gh pr create \
  --base feature/P6-Works \
  --title "P6-W{N}: {W 제목}" \
  --body "$(cat <<'EOF'
## Summary
- P6-W{N} {한 줄 요약}

## 주요 변경
- {변경 1}
- {변경 2}

## 체크리스트 (W 문서 §4.1 Definition of Done)
- [x] 항목 1
- [x] 항목 2
- [ ] (벤치 결과 있으면 미완료 가능)

## 벤치 결과 (해당 W에 벤치 있으면)
- {결과 요약}

## 실기기 회귀 확인
- ✅ 의도된 시각 변화만 발생
- ✅ 이전 상태 대비 회귀 없음

## 브레인스토밍 결과 반영
- docs/workPaper/P6-W{N}_brainstorm/synthesis.md
- 주요 결정: {요약}

## 다음 W
- P6-W{N+1}: {제목}

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

### Step 9: Document Update + Completion Summary

**W 문서 업데이트**:
- 상태: "인사이트 작성 완료" → "구현 완료"
- §4.1 Definition of Done 체크리스트 채우기 (`[x]`)
- §8.2 실제 커밋 해시 기록
- §8.1 완료 정의 달성 여부 요약

**`99_final_decision.md` 업데이트** (해당 W가 영향 주는 경우):
- 예: W1 완료 시 §1.2 C8/C9/C11 "완료" 마킹

**Completion Summary 템플릿**:

```
🎉 P6-W{N} 구현 완료

📊 Summary:
- W: {제목}
- 브랜치: feature/P6-W{N}
- 커밋 수: {N}개
- 신규 파일: {목록 요약}
- 수정 파일: {목록 요약}
- 총 변경: +{insertions} / -{deletions}

✅ 검증:
- 빌드: ✅ cmake --build iris_sdk 통과
- 단위 테스트: ✅ {테스트 수} 통과 (full 모드)
- 실기기 회귀: ✅ 없음 (해당 W 적용 시)

📝 문서 업데이트:
- docs/workPaper/P6-W{N}_*.md: 상태 "구현 완료"
- 99_final_decision.md: §{섹션} 완료 마킹

💡 다음 단계:
1. PR 리뷰 대기 → feature/P6-Works 머지
2. 다음 W:
   - `ar-lens-brainstorm P6-W{N+1}` — 브레인스토밍 시작
   - 또는 브레인스토밍 완료 시 `implement P6-W{N+1}`

🔗 PR: {URL}
```

## Mode Details

| Mode | Step 1 (브랜치) | Step 2 (plan) | Step 3 (구현) | Step 4 (빌드) | Step 5 (단위 테스트) | Step 6 (실기기) | Step 7 (커밋) | Step 8 (Push/PR) | Step 9 (문서) |
|------|---------------|---------------|--------------|--------------|---------------------|----------------|--------------|------------------|---------------|
| **default** | ✅ | ✅ | ✅ | ✅ | 선택 | 시각 W만 | ✅ | ✅ | ✅ |
| **quick** | ✅ | ✅ 간략 | ✅ | ✅ | ❌ | ❌ | 단일 커밋 | ❌ | ❌ |
| **full** | ✅ | ✅ | ✅ | ✅ | ✅ 필수 | ✅ 3 tier | ✅ 분할 | ✅ | ✅ 상세 |
| **debug** | ✅ | + root cause | 재작업 | + sanitizer | ✅ regression test | ✅ | fix 커밋 | (선택) | 학습 기록 |

### Quick Mode — 임시 개발 빌드 확인용

```
[1/4] 브랜치 확인 (브랜치 이미 있다고 가정)
[2/4] 핵심 변경만 (헤더 + 주요 구현)
[3/4] 빌드 통과 확인
[4/4] 단일 커밋: "wip(gpu-lens): P6-W{N} quick iteration"
```

### Full Mode — 프로덕션 후보 검증

```
[+ Step 5 필수] 단위 테스트 전 범위
[+ Step 6 확장]
  - HIGH tier 디바이스: 60fps 유지, 시각 품질 최상
  - MID tier: 30fps 유지, 기본 기능 동작
  - LOW tier: 20fps 최소, 크래시 없음
[+ Step 9 상세]
  - W 문서에 성능 프로파일링 결과 첨부
  - PR 본문에 tier별 결과 표
```

### Debug Mode — 문제 있는 W 재작업

```
[1/5] 🐛 문제 재현
  - 버그 증상 기록
  - git bisect로 introducing 커밋 추적

[2/5] 🔍 Root Cause 추적
  - sanitizer 빌드: -fsanitize=address, thread
  - debugging-toolkit:debugger 에이전트 호출
  - 스택 트레이스 분석
  - 메모리 누수 / race condition 확인

[3/5] 💻 Fix 구현 + 재빌드

[4/5] 🧪 Regression Test 추가
  - unit-testing:test-automator 에이전트
  - 같은 증상 재발 방지 테스트 작성

[5/5] ✅ 검증
  - 원래 기능 동작 확인
  - 새 regression test 통과
  - fix 커밋: "fix(gpu-lens): P6-W{N} {증상 요약}"
```

## Error Handling

### Document Not Found

```
❌ Error: W 문서를 찾을 수 없음

Task ID: P6-W{N}
검색 경로: docs/workPaper/P6-W{N}_*.md

💡 확인:
- W 번호 정확성 (P6-W1~W9 범위)
- 문서 파일 존재 여부: ls docs/workPaper/P6-W*
- 올바른 네이밍 (언더스코어 + 주제)
```

### Prerequisite Not Met

```
⚠️ Warning: 선행 W 미완료

현재 W: P6-W{N}
전제 조건 (§3):
- {선행 W 1} — 상태: 미완료
- {선행 W 2} — 상태: 완료 ✅

💡 먼저 완료 필요:
implement P6-W{M}

순서대로 진행하세요.
```

### Build Failure

```
❌ Build failed

Target: iris_sdk
에러 요약:
{에러 첫 3줄}

💡 진단 순서:
1. 전체 에러 로그 확인: tail -100 < build.log
2. 관련 symbol 추적:
   - find_symbol("{symbol_name}")
   - 정의 위치 확인
3. CMakeLists.txt 갱신 필요 여부 확인
4. 재시도: cmake --build . --parallel --target iris_sdk

반복 실패 시 debugging-toolkit:debugger 에이전트 호출.
```

### Test Failure (Full 모드)

```
❌ Unit Test Failed

실패 테스트: {N} / 총 {M}
대표 실패: {test_name}

💡 대응:
1. 상세 출력: --gtest_filter="{test_name}" --gtest_also_run_disabled_tests
2. 구현 vs 테스트 기대값 비교
3. 테스트가 잘못됐는지 vs 구현이 잘못됐는지 판정
```

### Git Push Rejected

```
❌ Push rejected

원인: 원격 branch가 앞서감 (rare) 또는 인증 문제

💡 대응:
- git pull --rebase origin feature/P6-W{N}
- 충돌 해결 후 push 재시도
- 인증: gh auth status
```

### Device Not Connected (Step 6)

```
⚠️ Android device not found

💡 대응:
1. adb devices 확인
2. USB 디버깅 활성 확인
3. 실기기 필수 W가 아니면 default 모드로 진행 (실기기 건너뛰기)
4. 필수면 사용자에게 기기 연결 요청
```

## Post-Implementation Checklist

각 W 완료 전 최종 확인:

- [ ] `§4.1 Definition of Done`의 모든 항목 ✅
- [ ] C++ 빌드 통과 (`cmake --build . --target iris_sdk`)
- [ ] 신규 파일은 CMakeLists.txt에 등록됨
- [ ] 공개 API 변경 없음 (§5.5 원칙 준수)
- [ ] 커밋 분할 적절 (한 논리 = 한 커밋)
- [ ] 커밋 메시지 프로젝트 규칙 준수 (영어 접두사, 한글 설명)
- [ ] PR 본문 템플릿 채움
- [ ] 해당 W 실기기 회귀 없음 (시각 영향 W)
- [ ] W 문서 §4.1/§8.2 업데이트
- [ ] `99_final_decision.md` 해당 섹션 완료 마킹

## Integration Examples

### Example 1: P6-W1 (EyeRenderPacket) — Default 모드

```
User: "implement P6-W1"

Skill actions:
1. 브랜치: feature/P6-Works → feature/P6-W1 분기
2. 문서: docs/workPaper/P6-W1_eye_render_packet.md 파싱
3. §5 확정 사항 → 스키마 + avg_iris_luma 수식
4. 구현:
   - eye_render_packet.h (구조체)
   - eye_render_packet_adapter.{h,cpp}
   - gpu_lens_renderer.{h,cpp} 수정
5. 빌드 통과 확인
6. 실기기: W1은 default에서 skip (계약 변경만)
7. 분할 커밋 4개 (위 §Step 7 예시)
8. Push + PR → feature/P6-Works
9. W 문서 업데이트 + 요약 보고
```

### Example 2: P6-W4 (환경 반사 벤치) — Full 모드

```
User: "implement P6-W4 full"

Skill actions:
1. 브랜치 설정
2. 문서 파싱 + 브레인스토밍 결과 확인
3. 구현:
   - 3 프로토타입 (OFF / env-map / periphery)
   - 벤치 토글 UI 추가
   - env_map.png 에셋 임포트
4. 빌드 통과
5. 단위 테스트 (프로토타입 스위칭 로직)
6. **실기기 벤치 (4 환경 × 2 동작 × 3 프로토타입 = 24 클립)**
   - 평가자 3명 블라인드 평가
   - Pupil 체감 지표 수집
7. 결과 기반 채택 구현 (env-map or periphery) + realSpec 폐기 확정
8. 분할 커밋 + 벤치 report 커밋
9. W 문서에 결과 첨부 + PR
```

### Example 3: P6-W2 (블렌드 3종) — Quick 모드

```
User: "implement P6-W2 quick"

Skill actions:
1. 브랜치
2. W 문서 §5 빠른 확인
3. 핵심 변경만:
   - blendLuminanceTintLinear → blendTintLinearV2 리네이밍
   - blendScreenLinear 신규 추가
4. 빌드 통과
5~6. skip
7. 단일 커밋: "wip(gpu-lens): P6-W2 블렌드 3종 리네이밍 + ScreenLinear"
8~9. skip (사용자가 나중에 full로 이어서)
```

## Project Context (Phase 6+)

**Branch Structure**:
```
develop  (W3-04 포함, 계속 유지)
└── feature/P6-Works  (Phase 6 통합, 현재 주력)
    ├── feature/P6-W1-eye-render-packet
    ├── feature/P6-W2-blend-canonical
    └── ... (각 W별)
```

**Performance Targets** (W9 기준):
- FPS: HIGH ≥ 60, MID ≥ 30, LOW ≥ 20
- Detection Latency: < 33ms
- Memory: < 100MB
- SDK Size: < 20MB

**Tech Stack**:
- C++17 (표준)
- OpenGL ES 3.1 (`#version 310 es` 셰이더)
- MediaPipe (Face Mesh + Iris Tracking)
- TensorFlow Lite 2.x
- OpenCV 4.x
- CMake 3.18+ (Ninja)
- GoogleTest (단위 테스트)

**Key Agents**:
- `systems-programming:cpp-pro` — C++ 구현 (모던 C++, RAII)
- `unit-testing:test-automator` — 테스트 생성
- `debugging-toolkit:debugger` — 문제 재현 + root cause
- `code-documentation:code-reviewer` — AI 코드 리뷰

**Related Skills**:
- **`ar-lens-brainstorm`** — W 브레인스토밍 라운드 담당. **이 스킬 호출 전에 돌릴 것**.

## Do NOT

- **브레인스토밍 혼입**: §6 미결 있으면 먼저 brainstorm 스킬. 이 스킬은 구현만.
- **W 문서 §5 우회**: 확정 사항 없이 "알아서 구현" 금지.
- **여러 W 동시 구현**: 한 호출 = 한 W.
- **develop 직접 커밋**: feature/P6-W{N} 브랜치에서만.
- **테스트 생략 (full 모드)**: 시각 영향 있는 W는 실기기 필수.
- **큰 단일 커밋**: 논리 단위로 분할.
- **새 아이디어 구현**: §5에 없는 기능 추가 금지. 필요하면 brainstorm 재호출.
- **공개 API 변경**: `sdk_api.h` 시그니처는 불변 (C8 원칙).

## Legacy Phase 1 Support

`implement P1-Wx-xx` 형식도 지원. Phase 1 문서 구조 파싱:

- 위치: `docs/workPaper/P1-Wx-xx_*.md`
- 섹션: "목표 / 산출물 / 검증 기준 / 선행 조건" (Phase 6 §1~§8과 다름)
- 브레인스토밍 없음 (이미 완료된 Phase)
- v1.0.0 시절 standard/performance/quick/debug 모드 호환:
  - standard → default
  - performance → full (+ application-performance 에이전트)
  - quick → quick
  - debug → debug

## Notes

- **Phase 6+는 브레인스토밍/구현 분리**가 기본. 이 스킬은 구현만.
- 각 W 완료 시 PR 단위로 → feature/P6-Works 누적
- 전체 Phase 6 완료 후 feature/P6-Works → develop 한 번에 머지
- 구체 수식/파일 경로는 각 W 문서 §5에 있음. 이 스킬은 **"어떻게 구현하나"의 메타 프로세스**.
- Serena MCP 도구 우선 (CLAUDE.md 글로벌 규칙)
- 커밋 메시지 한글 + Conventional Commits 접두사 영어

---

**Version:** 2.0.0
**Project:** IrisLensSDK Phase 6+ (Phase 1 legacy 호환)
**Updated:** 2026-04-23
**Related Skill:** `ar-lens-brainstorm` (이 스킬 호출 전 W 브레인스토밍 담당)
