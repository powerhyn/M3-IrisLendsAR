# P6-W9 브레인스토밍 R1 — Claude 응답

**모델:** claude-opus-4-7 (1M)
**라운드:** R1 (경량 리뷰)
**대상:** `docs/workPaper/P6-W9_integration.md` §6.1 ~ §6.7 + Phase 6 전체 점검
**숨김.**

---

## 6.1 머지 방식 — **A (머지 커밋) 채택**

근거:
- (1) W별 커밋이 의미 있음 (벤치 결과 역추적, "W7 림발 정책 구현"을 한 커밋으로 식별). Squash는 이 정보 손실.
- (2) Rebase는 기록 재작성 + push 필요 시 force-push 위험. 팀 워크플로에서 머지 커밋이 가장 안전.
- (3) 머지 커밋 메시지에 "P6 통합 — W1~W8 완료, 99 §1/§2 전 항목 반영" 명기.

## 6.2 Android demo UI 정리 — **W9 범위 밖, 별도 PR**

근거:
- (1) W2 §5.13에서 확정 ("W2 범위 밖, W9 integration 단계"). 단 W9가 integration 범위 자체를 최소화해야 한다면 demo UI는 그 다음 PR로.
- (2) 블렌드 drop-down 축소, 3D Light 버튼 제거 등 UI 작업은 리뷰 범위가 Kotlin 영역으로 확장됨. C++ 통합 PR과 분리.
- (3) PR 분리 이유: 기능 PR(P6 통합) 리뷰와 UI 정리 PR 리뷰의 집중도를 각각 높임.

## 6.3 deprecated no-op 함수 — **머지 유지, Phase 7 초기 제거**

근거:
- (1) `setHighlightEnabled` 같은 deprecated no-op은 외부 사용자(Android demo 이외의 클라이언트)가 쓰고 있을 수 있음. 일괄 제거는 breaking change.
- (2) Phase 6 머지에서는 deprecated 상태 유지 (경고 로그 또는 주석). Phase 7 초기 "호환성 transition 종료" 릴리즈에서 제거 + 릴리즈 노트에 명기.
- (3) Phase 7 착수 시 최우선 TODO로 기록.

## 6.4 W9 브레인스토밍 필수 여부 — **경량 리뷰 1라운드 (현재 이 작업)로 충분**

근거:
- (1) W9 자체가 구현 변경이 거의 없고 통합/검증이 주 역할. 본 브레인스토밍이 그 경량 리뷰 역할.
- (2) 추가 R2 불필요 — 본 R1에서 Phase 6 전체 점검 수행 (아래 "Phase 6 전체 점검" 섹션).
- (3) 결과 파일 이름: W9 폴더 내 `codex_w9.md`, `gemini_w9.md`, `claude_w9.md`, `synthesis.md`.

## 6.5 성능 벤치 자동화 — **기존 Android demo FPS 로직 재사용 우선**

근거:
- (1) Android demo에 이미 frame counter 있을 가능성 높음 (Phase 5까지 30fps 목표 검증하는 동안 구현됐을 것). 재사용 가능.
- (2) 없으면 간단한 `Choreographer.FrameCallback` 기반 60프레임 평균 FPS logcat 출력. 10줄 이내 코드.
- (3) **성능 기준 (CLAUDE.md):** 30fps+, 검출 33ms 이하, 메모리 100MB 이하, SDK 20MB 이하. W9에서 각 확인.

## 6.6 릴리즈 노트 작성 — **CHANGELOG + W별 결정 요약**

근거:
- (1) `CHANGELOG.md` 또는 별도 릴리즈 노트에 다음 구조:
  - "Phase 6: AR Lens Material & Sclera Bench" 제목.
  - W1~W8 주요 결정 1줄씩.
  - 99 §1/§2 최종 상태 요약 (Fresnel C, TintLinearV2 canonical, 림발 10/10 등).
  - Breaking change: 없음 (공개 API 불변).
  - New features: env reflection scaffold, blend 3종 확정, conditional Pupil material restore.
- (2) 벤치 결과 (B2, B5, B9, B1, B8, B4, Pupil 체감 2/3)는 별도 벤치 리포트 문서로 링크.
- (3) 사용자/고객사 전달본: 기술 세부 제거, "렌즈 품질 개선 + 성능 유지" 톤.

## 6.7 CI 통과 항목 — **체크리스트**

- [ ] C++ 빌드 (CLion `cmake-build-debug` 기준 + `scripts/build_android.sh` + `scripts/build_ios.sh`).
- [ ] 단위 테스트 (`cd cpp/cmake-build-debug && ctest`).
- [ ] Android demo APK 빌드 성공 (`./gradlew assembleDebug` 또는 equivalent).
- [ ] iOS Framework 빌드 성공 (해당 시).
- [ ] 정적 분석 (clang-tidy 설정돼 있으면, warning 0).
- [ ] **shader compile 에러/warning 0** (Phase 6에서 shader 대규모 수정).
- [ ] `docs/workPaper/` 각 W 문서 §5 최종 반영 확인.
- [ ] `99_final_decision.md` §1.2 C5/D3 + §2 B1/B2/B4/B5/B8/B9 상태 "확정" 또는 "구현됨" 반영.

---

## Phase 6 전체 점검 (W1~W8 vs 99_final_decision.md)

### 99 §1.1 기본 렌더 (D 축) 반영 상태

| 축 | 99 확정 | 반영 W |
|----|---------|--------|
| D1 분석 노멀 | 폐기 (S1) | W3 §5.9 제외 유지 ✅ |
| D3 highlight | 확정(W4 B2 결과 따라 재해석 가능) | W3 §5.8 Fresnel C / W4 B2 ✅ |
| D6 기본 블렌드 | TintLinearV2 | W2 §5.12 ID=5 canonical ✅ |

### 99 §1.2 core (C 축) 반영 상태

| 축 | 99 확정 | 반영 W |
|----|---------|--------|
| C4 CRL 수식 | 등록 (B1 벤치 대상) | W2 §5.7 ID 7 활성 ✅ |
| C5 환경 반사 | 가산 계층 (B2 기반 확정 대기) | W3 §5.11 + W4 B2 결과 대기 |
| C6 림발 정책 | 기본 ON + 메타 | W7 §5.10 10/10 엄수 + §5.8 JSON ✅ |
| C7 블링크 ramp | EMA 기반 | W6 §5.5 공식 확정 ✅ |
| C8 EyeRenderPacket | 스키마 확정 | W1 §5.1 + §5.11 optional ✅ |
| C9 avg_iris_luma | Self-measure | W1 §5.6/5.7/5.8/5.9 ✅ |
| C10 디테일 재주입 | 구현 | W6 §5.8 3×3 blur + §5.9 innerMask ✅ |

### 99 §2 벤치 (B 축) 준비 상태

| 벤치 | 상태 |
|------|------|
| B1 Normal vs CRL | W5 매트릭스 확정 ✅ |
| B2 env reflection | W4 매트릭스 확정 ✅ |
| B4 자동감지 fallback | W7 결과 통합 ✅ |
| B5 블링크 ramp | W6 매트릭스 확정 ✅ |
| B8 sclera veto | W5 매트릭스 확정 ✅ |
| B9 저조도 gate | W6 매트릭스 확정 ✅ |

### 놓친 점 점검

- ✅ **공개 C API 불변** (99 §1.2 C8): W1 §5.5에서 재확인.
- ✅ **sRGB 함수 완전 제거** (99 §1.1): W2 §5.6 sRGB 제거 정책.
- ✅ **realSpec 완전 삭제** (W4 B2 결과 연동): W2 §5.11 완전 삭제 확정.
- ⚠️ **LUMA 계수 통일 검증**: W2 §5.10 Rec.709 linear 통일 "테스트 케이스 추가" 기재됨. 실제 테스트 코드 W9 구현 전에 작성 필요.
- ⚠️ **conditional W8 활성 조건**: W4 B2 결과 2/3 Y 확정되지 않으면 W8 스킵. W9 머지 시점에 **W4 B2 결과 대기**라면 P6 머지를 B2 벤치 후로 보류 필요.
- ⚠️ **Android demo 기본 blendMode**: W2 §5.12에서 TintLinearV2 ID=5 설정 명시. Kotlin 쪽 실제 수정 W9 범위로 이관 여부 명확화 필요 (Claude 추천: demo UI 정리 PR에 포함).

### 메모리 반영 확인

- ✅ `feedback_multi_ai_orchestration_bias`: 8개 W 모두 3모델 독립 응답 + synthesis 작성. Claude 편향 정정 사례(W3 6.3, W5 6.5, W6 6.1) 기록.
- ✅ `feedback_real_data_first`: W4/W5/W6 매트릭스가 실기기 정성 체감 기반 설계.
- ✅ `feedback_qualitative_device_judgment`: Pupil 2/3 룰, B1/B5/B9 판정 모두 정성 체감 기준.
- ✅ `feedback_refactor_vs_retune`: W3 D1 재도입 금지 + realSpec 삭제/복원 Git history.
- ✅ `feedback_eye_refiner_v2_sufficient`: V2 좌표 기반 iris_center/radius 유지, iris_landmark 재추론 미도입.

---

## 요약 표

| 쟁점 | Claude 추천 |
|------|-------------|
| 6.1 머지 방식 | A (머지 커밋) |
| 6.2 Android demo UI | W9 범위 밖, 별도 PR |
| 6.3 deprecated 제거 | Phase 7 초기 |
| 6.4 W9 브레인스토밍 | 경량 R1로 충분 |
| 6.5 FPS 자동화 | 기존 재사용, 없으면 Choreographer 간단 추가 |
| 6.6 릴리즈 노트 | CHANGELOG + W별 1줄 + 벤치 링크 |
| 6.7 CI 체크 | 8항 체크리스트 |

**Phase 6 전체 점검:** 99 전 조항 반영 ✅, 놓친 점 3개 (LUMA 테스트 / W8 조건부 대기 / demo blendMode)는 W9 머지 전 보완.

---

## 자기비판

- "놓친 점 3개"는 Claude 독단 점검. Codex/Gemini도 각자 Phase 6 전체 훑어보고 다른 놓친 점 발견하면 synthesis에서 추가.
- 6.4 "경량 R1로 충분" 판정은 안이할 수 있음. Codex/Gemini가 "P6는 범위가 커서 R2 필요"라 주장하면 재검토.
