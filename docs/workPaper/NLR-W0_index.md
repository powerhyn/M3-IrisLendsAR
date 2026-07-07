# NLR-W0: 렌즈 렌더링 자연스러움 개선 트랙 인덱스 (Natural Lens Rendering)

> **상태**: 🔄 진행 중 (2026-07-02 트랙 개시, 계획 승인 완료)
> **계획서(정본)**: `docs/plans/iridescent-dreaming-neumann.md` — 로컬 조사(에이전트 3) → Ultraplan 원격 정제 → 로컬 재검증 → **Codex 교차 검증 R0**(수정 7건 반영) 완료본
> **목표**: 렌즈 렌더링을 "보편적으로 쓰기 가장 최적인" 단일 기본값으로 확정
> **브랜치**: `feature/P7-W4-sclera-luma-atten` (develop 075a5de 위로 rebase, 2026-07-02)

---

## 1. 트랙 구성 (상세 게이트는 계획서 참조)

| W | 내용 | 상태 |
|---|---|---|
| **NLR-W1** | sclera tint cap (`uScleraTintMax`) — P7-W4 승계 | ✅ 종결(2026-07-03) — **cap 무효 판정 + 문제 재정의**: 빛남의 실체는 휘도 비례 틴트의 **경계 톤 불연속** (`docs/bench/P7-W4/cap_sweep_result.md`). cap 코드는 OFF 보존. 처방은 합성 수식 원점 재조사(deep-research)로 이관 |
| **NLR-W2** | (재스코프) 합성 수식 원점 재조사 → canonical 확정: deep-research 21클레임 + Codex R1 + 실기기 7라운드로 병인 분해(빛남 임계 K≈4.4) + 트래킹 A/B(stab:fast). tone_class/effective alpha는 후속 W로 이월 | 🔄 판정 6건 대기 — **진입점 `NLR-W2_resume_kickoff.md`** |
| **NLR-W3** | 캐치라이트 보존 레이어 (iris/lens 마스크 한정 + 재합성 상한) | ⏳ |
| **NLR-W4** | radial ramp (alpha/pupil/edge 전이 완화 한정, 기각이 기본 자세) | ⏳ |
| **NLR-W5** | (조건부) 노출/WB 색 매칭 — 진입 조건 2중 | ⏳ |
| **NLR-W6** | 통합 확정 — 최종 블라인드 벤치 → canonical 승격 + 기각분 제거 | ⏳ |

순서 근거: cap(W1) = 이후 모든 벤치의 베이스라인 → tone_class(W2)가 CRL(ID=7) 지위 결정 → opaque 클래스는 캐치라이트 보존(W3)이 게이트.

## 2. 프로세스 확정 사항

- **교차 검증은 Codex 단독** (Gemini 제외) — tmux 팬 3.1. [사용자 확정 2026-07-02]
- W 단위 순차 완결 (구현 → 실기기 벤치 → 판정 → 다음).
- 벤치: 1인 실시간 토글 육안 체감 (S23+ SM-S916N), 일반 실내 + 밝은 조명 2조건, 저조도 범위 외.
- W1 문서는 `P7-W4_sclera_luma_attenuation.md` 승계 (§5 확정 사항 그대로 유효).

## 3. 사전 검증에서 확인된 코드 사실 (구현 시 필수 참조)

계획서 "사전 검증 결과" 절 참조. 요약:
- auto 센티널(-1) 차단 경로 3중: Java `LensConfig.clamp()`/`isValid()`/`Builder.build()` + JNI `copyConfigFromJava` + Kotlin `BlendMode.fromValue`
- `lens_sku_metadata.cpp` 파서가 미지 키 하드 거부 (all-or-nothing) → 커밋 순서: 파서 확장 → JSON 확장
- 데모 밝기 슬라이더(`seekMaxDetail`)는 dead — native `uMaxDetail` 1.25 하드코드
- `restoreLensRenderState()` 미복원 갭 4종: gateThreshold / blinkUpMs / scleraVetoMode / scleraProtect(지역 변수)
- 렌즈 에셋 알파 실측: 커버리지가 이미 알파맵에 내장 → effective alpha 이중 감쇄 금지 (렌더 α = lens.a × element_opacity)

## 4. 하드 제약 (기각 이력 — 재제안 금지)

셰이더 절차 림발(에셋 책임) · 외곽 광택 Fresnel(물리 오류) · 저조도 정교화 · 최종 사용자 수동 블렌드 선택 UI · eye_opening alpha-fade · 환경 반사 재개(Phase 9 이월 유지).

## 5. 변경 이력

| 날짜 | 변경 |
|---|---|
| 2026-07-02 | 트랙 개시 — 계획 수립(로컬+Ultraplan 병합) + Codex R0 교차 검증 반영 + 사용자 승인. 브랜치 rebase. W1 재베이스라인용 APK 빌드 착수. |
