# P6-W0: Phase 6 인덱스 — 렌즈 렌더링 자연스러움 실행 단계

> **상태**: 인사이트 작성 완료. 세부 계획 본문 작성 대기.
> **작성**: 2026-04-23
> **브랜치**: `feature/P5-W3-05` (base: `70633ac`)
> **최신 커밋**: `9aee86d` (S1 롤백 완료)

> 🔧 **구현 착수 시 반드시 참조**: [`P6_implementation_handoff.md`](P6_implementation_handoff.md)
> — 9개 W 브레인스토밍 R1 완료 후 작성한 구현 핸드오프. 전역 규약(색공간/LUMA, dist 정규화, 영역 경계, SKU 메타), 5건 버그 수정 내역, 의존성 그래프, W별 핵심 포인트, CI 체크리스트.

---

## 1. 인사이트 (세션 간 맥락 보존) ⭐ — 먼저 읽기

### 1.1 이 Phase 6의 정체

**Phase 6은 "새 기능 브레인스토밍"이 아니다.** P5-W3-05 브레인스토밍 4라운드(R1 독립 → R2 교차비판 → R3 Claude 종합 검증 → R4 패치 검증)로 도출된 합의본 `99_final_decision.md`의 **실행 단계**다. 합의된 17개 항목 즉시 적용 + 6개 벤치로 5개 잔여 쟁점 확정 + 1개 조건부 트랙 (Pupil cutout).

Phase 6 안의 각 W는 **독립된 브레인스토밍/벤치/구현 단위**. 각 W 시작 전:
1. 해당 W 문서를 새 세션에서 읽음
2. Codex/Gemini에 W 세부 구현 확정 라운드를 돌림 (이 문서의 §7 체크리스트 기반)
3. 구현 착수

### 1.2 이번 세션에서 확정된 것 (모든 W의 기반)

- **렌더러 입력 계약**: `EyeRenderPacket` 구조체 (내부 어댑터용, 공개 C API 불변)
- **블렌드 3종 확정** (ID 유지): TintLinearV2(5) / Multiply(1) / ScreenLinear(2, 신규). Normal(0)은 B1 벤치 대기, ColorReplace(7)는 `ColorReplaceLinear`로 교체 후보.
- **환경 반사 가산 계층 분리**: `blended += reflection * fresnel * renderMask`. 소스는 B2 벤치로 확정.
- **realSpec 폐기 조건부**: B2 성공 시 확정 폐기. 실패 시 재평가.
- **avg_iris_luma masked ROI 평균 실측** (Codex R3 지적 반영).
- **블링크 down 50~80ms ramp, up 시간은 B5 벤치로 확정**.
- **홍채 디테일 재주입** (원본 휘도 기반) + **저조도 gate 임계값은 B9 벤치로 확정**.
- **림발 정책**: SKU 메타데이터 기반 (자동 감지는 B4 fallback).
- **sclera veto**: geometry-first (B8로 color vs luma-only 최종 확정).
- **Pupil cutout**: P6-W8 조건부 트랙. B2 "중앙 공동 체감" 지표 2/3 이상 시만 발동.
- **텍스처 크기 256×256** 표준화. 사용자 측 리사이징 공급.

### 1.3 Phase 6 전체 철학

- **"추측으로 결정하지 말고 실측으로"**: R1~R3에서 에셋을 블랙박스로 가정한 게 맹점이었음. R4 실측으로 C6 림발 기본값 재검토 + Pupil cutout 근거 강화. 각 W 역시 **실측 → 브레인스토밍 → 구현** 순서 엄수.
- **"정량 점수보다 실기기 체감"**: 사용자 원칙. "수치로 정량 비교보단 실제 여러 환경에서 렌더링되는 것을 눈으로 확인하고 체감해야 알 수 있는 부분".
- **"의도적 처리보다 자연 커버 우선"**: P6-W8 Pupil cutout 정책의 핵심. 환경 반사 + 눈물막 반사 같은 기존 인프라로 해결될 가능성 먼저 확인. 어색하면 그때 별도 레이어.
- **"Claude 1인 종합 금지"**: 각 W 시작 전 Codex/Gemini와 세부 구현 확정 라운드 필수. `multi-ai-orchestration-bias` 메모리 참조.

### 1.4 W 간 의존성 그래프

```
P6-W0 (index, 이 문서)
  │
  ├─▶ P6-W1 (EyeRenderPacket + ROI 평균)
  │     │
  │     ├─▶ P6-W2 (블렌드 3종 + realSpec 폐기)
  │     │     │
  │     │     └─▶ P6-W3 (환경 반사 계층 스캐폴드)
  │     │           │
  │     │           ├─▶ P6-W4 (B2 환경 반사 벤치) ──▶ [Pupil 체감 수집]
  │     │           │                                    │
  │     │           │                                    ▼
  │     │           │                              P6-W8 조건부
  │     │           │
  │     │           ├─▶ P6-W5 (B1 + B8 벤치)
  │     │           ├─▶ P6-W6 (B5 + B9 벤치)
  │     │           └─▶ P6-W7 (B4 림발 벤치)
  │     │
  │     └─────(공통 기반)───────────────────────▶ 모든 후속 W
  │
  └─▶ P6-W9 (통합 테스트 + develop 머지) — 모든 W 완료 후
```

- **P6-W1/W2/W3**은 순차적 (기반 → 블렌드 → 반사)
- **P6-W4/W5/W6/W7**은 W3 완료 후 병렬 가능 (각 벤치 독립)
- **P6-W8**은 W4 결과 종속 (조건부)
- **P6-W9**는 최종

### 1.5 각 W별 핵심 한 줄 요약

| W | 이름 | 핵심 한 줄 | 의존 |
|---|------|-----------|------|
| W0 | Index (이 문서) | Phase 6 로드맵 + 세션 간 맥락 보존 | — |
| W1 | EyeRenderPacket + ROI 평균 | 렌더러 계약 재설계, 측정 경로 확보 | W0 |
| W2 | 블렌드 3종 + realSpec 폐기 | 수식 정리 (S1에서 이미 부분 구현) | W1 |
| W3 | 환경 반사 계층 스캐폴드 | C5 구조 + renderMask hook. 소스 미정. | W2 |
| W4 | B2 환경 반사 벤치 | 3 프로토타입 실기기 비교 + Pupil 체감 수집 | W3 |
| W5 | B1 + B8 벤치 | Normal vs CRL + sclera color/luma veto | W3 |
| W6 | B5 + B9 벤치 | 블링크 up ramp + 저조도 디테일 gate | W3 |
| W7 | B4 림발 벤치 | 자동 감지 fallback 채택 여부 | W3 |
| W8 | Pupil material (조건부) | W4 체감 2/3 이상 시만. Option E 재질 반투명 복원 | W4 |
| W9 | 통합 테스트 + 머지 | HIGH/MID/LOW 실기기 + develop 머지 | 전체 |

### 1.6 각 W 문서의 공통 구조 (8 섹션)

각 P6-W* 문서는 다음 구조를 따른다:

1. **인사이트** (이 섹션에 해당 — 세션 간 맥락 보존 ⭐)
2. **배경/맥락** — 99의 어느 결정에서 파생
3. **전제 조건** — 의존 W 완료 여부
4. **목표** — 이 W 완료 시 달성할 상태
5. **99에서 확정된 사항** — 수식·파라미터·API
6. **미결 사항** — W 시작 전 브레인스토밍에서 풀 질문
7. **W 브레인스토밍 시작 체크리스트** — 읽을 파일, 송신 프롬프트 초안, 예상 쟁점
8. **완료 정의 + 다음 W 트리거**

**현재 상태**: 섹션 1(인사이트)만 작성 완료. 섹션 2~8은 stub. Step B 단계에서 채움.

### 1.7 git 브랜치 전략

**3-tier 구조** (P5-W3-05 → P6 통합 → W별):

```
develop  (W3-04 포함, Phase 6 완료까지 유지)
  └── feature/P6-Works  (Phase 6 통합 브랜치, 현재 작업 중)
        ├── feature/P6-W1-eye-render-packet (또는 feature/P6-W1)
        ├── feature/P6-W2-blend-canonical
        ├── feature/P6-W3-env-reflection-scaffold
        ├── feature/P6-W4-env-reflection-bench
        ├── feature/P6-W5-blend-sclera-bench
        ├── feature/P6-W6-temporal-detail-bench
        ├── feature/P6-W7-limbal-policy
        ├── feature/P6-W8-pupil-material-conditional (조건부)
        └── feature/P6-W9-integration
```

**플로우**:
1. 각 W는 `feature/P6-Works`에서 분기 → W별 브레인스토밍/구현
2. W 완료 시 `feature/P6-Works`로 PR 머지
3. 전체 Phase 6 완료 후 `feature/P6-Works` → `develop` 한 번에 머지

**참고**: 각 W 구현 시 브랜치명은 짧게(`feature/P6-W1`) 또는 설명 포함(`feature/P6-W1-eye-render-packet`) 중 선택. 명령어 편의 vs 의미 명확성 trade-off.

### 1.8 새 세션 시작 체크리스트 (재개 시)

새 세션에서 작업 재개할 때:

1. **브랜치 확인/생성**:
   ```bash
   git branch --show-current          # 현재 위치 확인
   git checkout feature/P6-Works      # 통합 브랜치로 이동
   git pull                            # 최신 상태
   git checkout -b feature/P6-W{N}-*   # W별 분기 (이미 있으면 checkout만)
   ```
2. `docs/workPaper/P6-W0_index.md` (이 파일) 먼저 읽기 (5분)
3. 작업할 W의 `P6-W{N}_*.md` 문서 읽기 (10분)
4. `docs/workPaper/P5-W3-05_brainstorm/99_final_decision.md` 해당 섹션 확인 (5분)
5. 필요 시 `15_asset_analysis.md`, `16_product_crosscheck.md` 재확인 (5분)
6. 필요 시 `docs/workPaper/P6_review/` Codex/Gemini 리뷰 의견 확인
7. Codex/Gemini 팬 재기동 + W 세부 구현 확정 브레인스토밍 송신 (섹션 7 체크리스트 기반)

**30분 내 맥락 완전 재개 가능**을 목표로 각 W 문서가 작성됨.

### 1.9 이번 세션에서 확인된 주요 이슈 (R4 Critical 수정 반영 완료)

각 W 브레인스토밍 시 이 수정 사항이 반영된 상태에서 시작한다는 점 인지:

- **W1 §5.2.1**: `uAvgIrisLum`는 **sRGB 공간 평균 luma**로 통일. W2 수식이 이를 제곱해 linear 근사.
- **W1 §5.5**: 공개 C API 실제 이름 `iris_sdk_render_lens_texture` / `iris_sdk_render_with_result` (이전에 잘못 쓴 `iris_sdk_render_lens_gpu`는 존재 안 함).
- **W3/W6/W8**: 셰이더의 `dist`는 이미 `/scaledRadius`로 정규화된 0~1 값. `dist / iris_radius`, `smoothstep(iris_radius * 1.2, ...)` 같은 이중 정규화 수식 금지.
- **W7**: "B10 신규 벤치", "기본 ON vs OFF 재판정"은 out of scope. 99 §1.2 C6 범위(B4 fallback + SKU 메타)로 제한.
- **W6**: EMA 계수는 §5.1 정밀 공식 `1 - pow(0.05, dt/target_ms)` 사용. 근사 `α ≈ 3/(fps·target)`은 참고용.

### 1.10 Phase 9 이월 트랙 (2026-06-04 분리 확정 / P7-W3 cross-link)

P6-W3/W4/W8 이월 트랙은 **Phase 9(또는 7.5)로 분리**. P8 뷰티 우선.

- **W3** 환경 반사 scaffold: OFF 기본 보존 (메모리 `w4-env-reflection-deferred`)
- **W4** B2 24클립 벤치: 인프라 보존 (`docs/bench/P6-W4/`)
- **W8** Pupil material: Option E 설계 보존 (`P6-W8_*.md`)

재개 시점: Phase 8 완료 후 비즈니스 임팩트 재평가. 진입점: 메모리 `phase9-deferred-tracks`, 로드맵 `P7-W0_index.md` §8.

---

## 2. 배경/맥락
_TODO: Step B에서 작성_

## 3. 전제 조건
_TODO_

## 4. 목표
_TODO_

## 5. 99에서 확정된 사항
_TODO (§1.2 요약을 상세화)_

## 6. 미결 사항
_TODO_

## 7. W 브레인스토밍 시작 체크리스트
_TODO (이 인덱스 문서 자체는 브레인스토밍 대상 아니지만, "Phase 6 전체를 시작할 때" 사용자 승인 체크리스트)_

## 8. 완료 정의 + 다음 W 트리거
_TODO_

---

## 참조

- `docs/workPaper/P5-W3-05_brainstorm/99_final_decision.md` — 전체 결정 근거
- `docs/workPaper/P5-W3-05_brainstorm/15_asset_analysis.md` — 에셋 실측
- `docs/workPaper/P5-W3-05_brainstorm/16_product_crosscheck.md` — 웹 교차검증 한계
- `~/.claude/projects/.../memory/MEMORY.md` — 자동 로드 메모리
