# P8 뷰티 — 진입 킥오프

> 이 문서 + 아래 필독으로 **대화 맥락 0인 새 세션이 P8를 스코핑·착수**. 작성: 2026-06-19.
> ④(추적 외부화 리팩토링) 완전 종결 + develop 머지(df5721d)·origin push 완료 후 진입점.

## 0. 진입 절차
1. **필독**: 이 문서 → 메모리 [[p8-facemesh478-substrate]](P8 substrate 확정) + [[refactor-p2-adr-golden]]의 "④ 완료 후 P8 뷰티" 블록(상세 plan) → `docs/workPaper/P8-W1_skin_smoothing_port.md`(① skin smoothing 완료 기록) → **P8 입력 핸드오프**: `docs/lenssim-handoff/jaw-vline-warp-handoff-from-lenssimulator.md`(②형태워프 확정 알고리즘·파라미터), `docs/lenssim-handoff/radiance-skin-tone-handoff-from-lenssimulator.md`(①피부 화사함), `docs/lenssim-handoff/natural-lens-fit-blending-research.md`(렌즈 톤별 블렌딩, 보조).
2. **브랜치**: `develop` (df5721d — ③-3+④ 통합 머지 + origin/develop push 완료, local==origin). P8는 develop 직접 또는 새 P8 브랜치(사용자 결정).
3. **빌드**: 데스크톱은 기존 `cpp/cmake-build-debug` 재사용(`cmake .. -DIRIS_SDK_FETCH_TFLITE=OFF && cmake --build . --parallel -- -k 0`, **새 빌드 디렉토리 금지**). Android `cd android && ./gradlew :demo-app:assembleDebug`(또는 `:iris-sdk:testDebugUnitTest`).
4. **C++ 작업이므로 `systems-programming:cpp-pro` 동반**(CLAUDE.md). 각 W 시작 시 Codex/Gemini 교차 브레인스토밍(`ar-lens-brainstorm` 스킬)이 프로젝트 패턴.
5. **착수 1단계 = 현황 재실측**: 아래 §2 line은 2026-06-19 develop 기준이나 P8 진행 중 시프트 가능 → symbol grep으로 재확인.
6. ⚠️ **C API 헤더 실제 경로 = `cpp/include/iris_sdk/sdk_api.h`** (아래 §2의 `sdk_api.h:NNN`은 이 경로). **CLAUDE.md 컴포넌트 표의 `core/include/` 표기는 스테일**(core/→cpp/ 이관 완료, core/include 디렉토리 없음). 코어 헤더는 모두 `cpp/include/iris_sdk/`, 구현은 `cpp/src/`.

## 1. 목표 (한 줄) + 왜 지금
P8 = **뷰티 simulation 본격화**(사용자 비즈니스 최우선). ④로 추적 아키텍처가 정리됐으니 이제 뷰티 가치 기능. **핵심 2개만**(2026-06-16 사용자 확정): ① 피부 skin smoothing + ② 턱깎기/얼굴축소 형태워프. **곁가지(LUT/레거시FreqSep/색보정/vivid)는 제거.**

## 2. 현재 상태 — 검증된 사실 (2026-06-19 develop 코드 확인)
### ① 피부 skin smoothing — ✅ 구현됨 (P8-W1)
- 진입: 데모 `btnP8Skin`(GpuRenderActivity.kt:118/232/502) → `setSkinMaskSmoothing`(:505) → C API `iris_sdk_set_skin_mask_smoothing(int enabled, float strength)`(sdk_api.h:739).
- substrate: `cpp/include/iris_sdk/gpu/skin_mask_geometry.h`, 전용 셰이더 3종(shader_sources.cpp SKIN_MASK_FILL/BLUR/COMPOSITE), gpu_beauty_backend, `cpp/tests/test_skin_mask_geometry.cpp`.
- **BeautyFilterConfigV2 무관 독립 채널**(use_skin_mask 게이트). 잔여: 실기기 검증 2항([[p8-w1-skin-smoothing-port]]) + 화사함(radiance) 추가 검토(radiance-skin-tone 핸드오프).
### ② 턱깎기/얼굴축소 형태워프 — 🔶 substrate 존재, 구현 완성도 착수 시 실측
- 진입: C API `iris_sdk_apply_face_warp(... slim_face, thin_chin, enlarge_eyes ...)`(sdk_api.h:757) + `IrisBeautyConfigV2`.slim_face(:609)/enlarge_eyes(:610)/thin_chin(:611) + Java `IrisLensSDK.applyFaceWarp`.
- substrate: `cpp/src/warp/face_warp_controller.cpp`+`cpp/include/iris_sdk/warp/face_warp_controller.h`, `cpp/src/warp/grid_mesh.cpp`+`.h`(grid_mesh = 형태워프 토대). 알고리즘 입력 = `jaw-vline-warp` 핸드오프(LensSim 실기기 S23+ 검증 확정, 6bacaec). Snap Liquify식 per-point Radius/Intensity.
- ⚠️ **형태워프는 사실상 백지(완전 스텁) — 게이트 실측 확정(2026-06-19)**: `iris_sdk_apply_face_warp`(sdk_api_v2.cpp:475) → `GPUBeautyBackend::applyFaceWarp`(gpu_beauty_backend.cpp:2445)이 패스스루 후 `LOGW("not supported... P4")` + `NOT_SUPPORTED` 반환, **grid_mesh 미호출**. 즉 ② 구현은 처음부터 작성. grid_mesh resize(468→478) 확장 포함 여부도 착수 시 확인.
### 곁가지 (제거 대상)
- LUT(3D 색), 레거시 FreqSep(`skin_quality`:602 / `smooth_intensity` / `pore_reduction`), 색보정(`whitening` / `color_balance` / `iris_sdk_set_skin_color_filter`), vivid(`vivid_*`). 위치: sdk_api_v2.cpp, cpu_beauty_backend.cpp, gpu_beauty_backend.cpp, shader_sources.cpp.

## 3. 할 일 / 첫 슬라이스 결정 + ⚠️불변식
**P8는 아직 W로 스코핑 안 됨 — 첫 행동 = 첫 슬라이스 결정**(사용자 확인 또는 ar-lens-brainstorm 스코핑). 후보:
- **(A) 곁가지 제거** (plan 순서: A 데모UI → B GPU패스[vivid→FreqSep→LUT]+stale테스트[FreqSepMappingTest 등] → C CPU색보정[beauty.png 골든 재캡처] → D 구조체 ABI필드[2.0 or 1.x no-op]). de-risk + substrate 명료화.
- **(B) ② 형태워프 구현** (비즈니스 가치, jaw-vline-warp 핸드오프 알고리즘 이식, grid_mesh+face_warp_controller).
- **(C) ① skin smoothing 마무리** (실기기 검증 2항 + radiance 화사함).
- **권고**: 첫 슬라이스는 사용자 결정(plan은 곁가지 제거 선행을 언급하나 비즈니스 가치는 ② 형태워프). 새 세션은 이 결정부터 확인하고 착수.

**⚠️ 절대 제약(불변식)**:
1. **핵심 2개(피부 skin smoothing + 형태워프)만 + 곁가지 제거** — 사용자 2026-06-16 확정. 새 곁가지 추가 금지.
2. **grid_mesh substrate 보존** — 형태워프 토대(삭제 금지; 과거 '삭제' 제안은 철회됨, ⑤ 478 확장은 형태워프가 메인이라 유효).
3. **skin smoothing 독립 채널 보존** — BeautyFilterConfigV2 무관, use_skin_mask 게이트(건드리지 말 것).
4. **ABI 보존** — 곁가지 config 필드(IrisBeautyConfigV2) 제거는 2.0 또는 1.x no-op 유지(공개 POD ABI 불변).
5. **곁가지 CPU 색보정 제거 시 beauty.png 골든 재캡처** 동반(manifest 기록).
6. **이식 정본 = LensSimulator**(핸드오프 출처 6bacaec), 값 무변경 이식 — 잘못된 공간/스케일 적용 금지([[real-data-first]]).

## 4. 게이트/검증
- 데스크톱 빌드 신규 warn0 / 골든(beauty.png 영향 시 재캡처+manifest, 그 외 ε 일치) / ctest 회귀0(**pre-existing 3 허용**: GPUBeautyBackendTest#401 + FreqSepMappingTest#587/588) / assembleDebug BUILD SUCCESSFUL / **실기기 육안**(밝은 환경 우선 — [[low-light-usage-rare]]; 본인 실시간 토글 체감 — [[solo-dev-bench-method]]·[[qualitative-device-judgment]]).

## 5. 워킹트리 주의 (커밋 제외 — 정상 잔존)
- `.claude/settings.local.json`(로컬 설정), `android/demo-app/src/main/assets/env/env_default_256x128.png`(Git LFS 팬텀 — 영구 ` M` 표시, **스테이징 금지**). 중단 잔여물 아님.

## 6. 완료 후
- P8 각 W는 `docs/workPaper/P8-WN_*.md`로 문서화(P8-W1 선례). 메모리 [[p8-facemesh478-substrate]] + [[refactor-p2-adr-golden]] 진입점 갱신. 내부 통합 머지는 PR 생략 직접 머지([[skip-pr-for-internal-w-merges]]).
