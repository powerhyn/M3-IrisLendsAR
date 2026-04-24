# P6-W3 브레인스토밍 R1 — 종합

**작성일:** 2026-04-24
**참여:** Codex (gpt-5.4 xhigh), Gemini (gemini-3-flash), Claude (opus-4-7)
**대상:** `docs/workPaper/P6-W3_env_reflection_scaffold.md` §6.1 ~ §6.7

---

## 1. 합의 (3/3 동의 → 닫힘)

| 쟁점 | 확정 |
|------|------|
| **6.4** sampleReflection 추상화 | **방식 A (uniform `uSourceType` 스위치, 단일 바이너리)** |
| **6.7** B2 벤치 매트릭스 골격 | **4환경 × 2동작 × 3프로토타입 = 24 클립** 초안 확정 |

## 2. 다수결 결정 (2/3 동의)

### 6.1 Fresnel 수식 — **옵션 C (가짜 Fresnel) 채택 (Gemini+Claude 다수)**
- Gemini, Claude: 옵션 C (iris 거리 기반 근사).
- Codex: 옵션 A (Schlick `dot(N,V)` 기반).
- **판정:** 노멀 회피 + D1 재도입 방지가 우선. 3모델 모두 "옵션 C ≠ D1 재도입" 명시.
- **Codex 지적 수용 (W4 재검토 조항):** W4 B2 결과에서 "env-map이 가짜 Fresnel로도 시각 체감 충분" → C 확정. "시점 의존성 부족으로 효과 약함" 피드백 2/3 이상 → **W8 또는 별도 후속 W**에서 옵션 A/B(노멀 포함) 재검토. W3 scaffold 단계에서는 C로 완료.

### 6.2 분석 노멀 W3 포함 여부 — **제외 (Gemini+Claude 다수)**
- Gemini, Claude: 제외. S1 D1 해체 취지 유지.
- Codex: 최소 포함 (`reflect(-V,N)` + `dot(N,V)`만, 고정 광원 제외).
- **판정:** 6.1에서 옵션 C 채택 → 노멀 자체가 필요 없어짐. Codex의 "D1 재도입 아님" 방어 논리는 6.1이 A/B로 전환될 때만 유효.
- **W4 재검토 조항:** 6.1이 후속 W에서 A/B로 재검토되면 6.2도 Codex 제안(최소 노멀, 고정 광원 제외) 형태로 함께 고려.

### 6.3 renderMask hook 활성 방식 — **`#ifdef` 채택 (Codex+Gemini 다수)**
- Codex, Gemini: `#ifdef` (컴파일 타임 분기, 기본 바이너리에서 완전 비활성).
- Claude: uniform flag (런타임 토글).
- **판정:** W8 조건부 트랙의 dead code를 프로덕션 바이너리에서 제거하는 편이 맞음. **Claude 편향 경계 적중** (self-doubt에서 이미 "2:1 재검토" 언급).
- **Claude 지적 완화:** W4 벤치 동안 hook on/off 비교가 필요하면 **벤치 전용 debug 빌드 플래그** 추가 (`#define RENDER_MASK_HOOK_ENABLED 1`). 프로덕션 빌드는 0 유지.

### 6.5 reflectUV 계산 — **옵션 C (iris local 좌표) 채택 (Gemini+Claude 다수)**
- Gemini, Claude: 옵션 C (`reflectUV = (uv - iris_center) / iris_radius * 0.5 + 0.5`).
- Codex: 옵션 B (`reflect(-viewDir, normal)` 기반).
- **판정:** 6.1/6.2에서 노멀 제외 확정 → 옵션 B 자동 배제. Codex의 "방향성 상실" 우려는 6.1 Fresnel이 C일 때 이미 시점 의존성이 약화된 상태이므로 현재 scaffold 설계에서는 수용 가능.

### 6.6 env_map 에셋 위치 — **demo assets 우선, SDK 내장 보류 (Codex+Claude 다수)**
- Codex, Claude: `android/demo-app/src/main/assets/env/` (demo 주도).
- Gemini: SDK 내장.
- **판정:** W4 벤치가 demo 기반이고 P6 범위에서 SDK 내장은 불필요. **SDK 내장은 W8~W9 단계에서 필요성 판단 후 결정**.
- **포맷 합의 (3/3):** 256×128 LDR. PNG.
- **초기 에셋:** 3종 — `env_default_256x128.png`, `env_office.png`, `env_outdoor.png` (Claude 제안).

## 3. 6.7 매트릭스 상세 (3/3 합의 골격 위에 Claude+Gemini 확장 사항 병합)

- **환경 4종:** 실내 형광, 창가 측광, 야간 실내, 실외 낮.
- **동작 2종:** 정면 미세(블링크 포함), head turn (±15°).
- **프로토타입 3종:** OFF / env-map / periphery.
- **조합:** 24 클립.
- **우선순위 (Claude):** env-map vs OFF를 1차 비교, periphery는 2차.
- **Pupil 체감 기록:** 각 클립에서 "중앙 공동 체감 Y/N" 별도 체크 (W8 조건부 트랙 발동 기준).
- **실시간 체감 (Gemini 강조):** 촬영 직후 앱에서 바로 A/B 토글 — sampleReflection uniform 스위치 덕분에 가능.
- **SKU:** Tint(Linear)V2 + 고발광 iris_mat_B.

## 4. 미결 / 실기기 이관

- **6.1 Fresnel 옵션 C 최종 판정은 W4 B2 결과에 종속.** "효과 약함" 피드백 시 후속 W에서 옵션 A 전환 가능성 열어둠. 단 W3 scaffold는 옵션 C로 완료.

## 5. 편향 체크

- **Claude 편향 경계 적중:** 6.3에서 Claude 원안(uniform flag)이 Codex+Gemini 다수(#ifdef)에 의해 뒤집힘. Claude가 자기비판에서 "2:1이면 재검토" 명시한 대로 수용 → Claude-only 편향 확인.
- **Codex 편향:** 6.1, 6.2, 6.5 모두 Codex가 소수 (A/포함/B). Codex가 "시점 의존성 중요" 축을 일관되게 밀고 있는데, W3 scaffold는 "D1 재도입 금지" 제약이 더 강한 상황이라 판정이 다수 쪽. Codex 주장은 W4 결과에 따라 후속 W에서 재부활 가능성으로 기록.

## 6. W3 문서 §5에 반영할 항목

- **§5.8** Fresnel: **옵션 C (가짜 Fresnel)**. W4 B2 결과에 따라 후속 W 재검토 조항 명시.
- **§5.9** 분석 노멀: **W3에서 제외**. 6.1이 후속 W에서 A/B로 전환 시 Codex 제안(최소 노멀, 고정 광원 제외) 패키지로 함께 검토.
- **§5.10** renderMask hook: **`#ifdef` (컴파일 타임)**. 프로덕션 기본 off. W4 벤치용 debug 플래그 별도.
- **§5.11** sampleReflection: **방식 A (uniform `uSourceType`)**, 단일 바이너리, W3에선 no-op 기본 구현.
- **§5.12** reflectUV: **옵션 C (iris local)**.
- **§5.13** env_map 에셋: **`android/demo-app/src/main/assets/env/` 위치**, 256×128 LDR PNG. 초기 3종 (default/office/outdoor). SDK 내장은 W8~W9 판단.
- **§5.14** B2 매트릭스: **4×2×3=24 클립**. env-map vs OFF 우선 비교. Pupil 체감 Y/N 별도 기록.

## 7. 다음 액션

1. W3 문서 §5/§6 업데이트.
2. Task #3 완료 표시.
3. W4 브레인스토밍 착수.

## 8. R2 필요 여부

**R2 불필요.** 다수결 결과들이 "W3 scaffold 범위" 제약 하에서는 자연스럽고, Codex 소수 의견은 W4 벤치 결과에 따라 후속 W로 재검토 경로가 열려 있음.
