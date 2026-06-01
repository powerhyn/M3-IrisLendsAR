# P6-W3 브레인스토밍 R1 — Claude 응답

**모델:** claude-opus-4-7 (1M)
**라운드:** R1
**대상:** `docs/workPaper/P6-W3_env_reflection_scaffold.md` §6.1 ~ §6.7
**특수:** 숨김 파일 저장 → 두 모델 응답 완료 후 rename (편향 방지).

---

## 6.1 Fresnel 수식 — **옵션 C (가짜 Fresnel) 추천**

근거:
- (1) 옵션 A(Schlick) / B(view·normal 각도)는 모두 **법선 벡터가 필요**. W3 scaffold가 D1 재도입 금지 제약을 우선한다면 노멀 없이 근사 가능한 옵션 C가 유일한 정합 선택지.
- (2) 가짜 Fresnel(iris 반경 기반, 중심=1, 가장자리=0에 가까운 smoothstep 반전)은 "각도에 따른 반사 강도" 효과를 실기기 체감 수준에서 거의 동일하게 재현. W4 B2에서 정확도 문제 발견 시 옵션 A/B로 전환 여지(2차 벤치)는 남겨둠.

## 6.2 분석 노멀 계산 W3 포함 여부 — **미포함 확정 권고**

근거:
- (1) S1에서 D1 분석 노멀 수식을 명시적으로 삭제한 결정의 취지를 W3에서 뒤집으면 "해체한 걸 다시 조립" 비판이 정당. 메모리 `feedback_refactor_vs_retune`은 "해체보다 재조율"이지 "한 번 해체한 걸 바로 복원"은 아님.
- (2) W4 B2 결과가 "노멀 기반 Fresnel이 실기기 체감 유의미"로 나오면 그때 별도 W(또는 W8 등)에서 노멀 복원. 지금은 옵션 C로 scaffold만.

## 6.3 renderMask hook 활성 방식 — **`uniform flag` (런타임 on/off) 추천** — Claude 원래 §6.3 추천(#ifdef)에서 선회

근거:
- (1) `#ifdef`는 컴파일 시 결정이라 W4 벤치 동안 "hook on vs off" A/B 토글이 불가능. 벤치 중 시각 체감 비교에는 uniform flag가 필수.
- (2) 프로덕션 릴리스에서 hook off로 최적화하고 싶으면 빌드 플래그 추가로 preprocessor 제거 가능(2단계). W3 scaffold 단계에서는 유연성 우선.
- (3) 6.4 sampleReflection 추상화 방식 A(uniform 스위치)와 자연스럽게 묶임.

## 6.4 sampleReflection 추상화 — **방식 A (uniform 스위치, 단일 바이너리) 추천**

근거:
- (1) W4 B2 벤치가 OFF/env-map/periphery 3프로토타입 런타임 스위치를 요구. 방식 B(shader variant)는 APK 1개에서 3변종 토글 불가.
- (2) 방식 A의 런타임 분기 오버헤드는 `uniform int uSourceType` 스위치 1회뿐 → 체감 성능 차이 없음.
- (3) 방식 C compute shader는 이미 기각 상태 유지.

## 6.5 reflectUV 계산 — **옵션 C (iris local 좌표) 추천**

근거:
- (1) 6.2에서 노멀 미포함 확정 → 옵션 A/B는 자동 배제.
- (2) env map이든 periphery든 같은 UV 공간 쓸 수 있어 방식 A의 uniform 스위치와 결합 용이. `reflectUV = (uv - iris_center) / iris_radius * 0.5 + 0.5`.

## 6.6 env_map 에셋 위치 — **`android/demo-app/src/main/assets/env/` + SDK assets 옵션 병행**

근거:
- (1) W4 벤치는 Android demo 실행 기반이므로 demo assets이 주 위치. 파일명 제안: `env_default_256x128.png`(기본), `env_office.png`(실내), `env_outdoor.png`(실외) — 3종 LDR로 시작.
- (2) SDK 내장은 "demo 없이 SDK만 썼을 때 기본 env map 보장" 용도인데 P6 범위에서는 demo 테스트가 목적이므로 **SDK 내장은 W4 결과 후 결정**(W8~W9 시점). W3에선 demo assets만 확보.

## 6.7 B2 벤치 매트릭스 초안

권고:
- **환경 4종:** 실내 형광, 창가 측광, 야간 실내, 실외 낮.
- **동작 2종:** 정면 미세(블링크 포함), head turn(±15°).
- **프로토타입 3종:** OFF / env-map / periphery.
- **조합:** 4×2×3 = 24 클립. 단, 블라인드 A/B 비교 기준으로 **env-map vs OFF** 우선 집계, periphery는 2차 비교.
- **SKU 우선순위:** Tint(Linear)V2 + 고발광 SKU 1종(iris_mat_B).
- **Pupil 체감 관찰:** W8 조건부 트랙 발동 여부 판정에 매트릭스의 "중앙 공동 체감 Y/N" 별도 기록.
- W3에서 초안만, W4 시작 시 확정.

---

## 요약 표

| 쟁점 | Claude 추천 | D1 재도입 리스크 판정 |
|------|-------------|-------------------------|
| 6.1 Fresnel | **옵션 C (가짜)** | C는 D1 재도입 아님 |
| 6.2 분석 노멀 | **미포함** | - |
| 6.3 hook 활성 | **uniform flag** (원 Claude §6.3 추천 #ifdef에서 선회) | - |
| 6.4 sampleReflection | **방식 A (uniform 스위치)** | - |
| 6.5 reflectUV | **옵션 C (iris local)** | - |
| 6.6 env map 에셋 | **demo assets 3종, SDK 내장은 보류** | - |
| 6.7 B2 매트릭스 | **4×2×3=24 클립, env-map vs OFF 우선** | - |

---

## 자기비판 (편향 경계)

- 6.3에서 원 문서 Claude 추천(#ifdef)과 현 추천(uniform flag)이 다름. 이유는 "W4 벤치 런타임 토글 필요성"을 재평가한 결과. Codex/Gemini가 #ifdef로 가면 다수결로 재검토.
- 6.1/6.2는 "D1 재도입 금지" 전제가 강력하여 옵션 C 외 대안이 사실상 봉쇄된 상태. Codex가 "노멀 기반 필수"라 주장하면 W3 scaffold 범위 재협상 필요 (그 경우 W4 B2로 이관 검토).
- 6.7 매트릭스 수치(24 클립)는 W4 실측 직전에 다시 다듬어야 함. Gemini가 피실험자 수/촬영 표준 절차로 구체화해주면 수용.
