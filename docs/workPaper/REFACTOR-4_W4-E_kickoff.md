# W4-E 착수 킥오프 — cpu-render deprecation + 16KB 정렬 (+ 남은 ④ 로드맵)

> 이 문서 + 아래 필독으로 **대화 맥락 0인 새 세션이 W4-E를 바로 착수**하고, **남은 ④ 전체(W4-E → 좌표 relabeling → W4-B4 → 그 후 P8 뷰티)를 끊김 없이 이어가게** 작성. 작성: 2026-06-18.
> W4-D(자체 추적 코어 제거)까지 완전 종결. W4-E는 ④의 마지막 surface 정리 슬라이스.

## 0. 진입 절차 (그대로)
1. **필독**: 이 문서 → `docs/workPaper/REFACTOR-4_plan.md`(§3 W4-E = line 96 부근, §2 범위, §0 핵심사실, §3 W4-D 완료 기록) → `docs/decisions/0001-landmark-injection-tracking-replacement.md` §8(cpu-render 옵션 B 처분)·§9(16KB 페이지 정렬).
2. **브랜치**: `refactor/p33-tracking-ab` (이미 체크아웃, develop 미머지). 최근 커밋 `1b03c17`(W4-D 육안 통과 기록) ← `c7e2665`(W4-D 코어 제거) ← `c2104e2`(W4-D 안전망).
3. **C++ 작업이므로 `systems-programming:cpp-pro` 동반**(CLAUDE.md). 빌드는 **기존 `cpp/cmake-build-debug` 재사용**(`cmake .. -DIRIS_SDK_FETCH_TFLITE=OFF && cmake --build . --parallel -- -k 0`, 새 빌드 디렉토리 금지). Android는 `cd android && ./gradlew :demo-app:assembleDebug`.
4. **착수 패턴(W4-A~D에서 검증)**: plan 기재가 실제 코드와 다를 수 있음(특히 W4-D가 surface를 바꿔 line 스테일). **착수 1단계 = 현황 실측**(아래 §2는 2026-06-18 실측이나 재확인 권장) → cpp-pro와 설계 → 구현 → 게이트. **외부 모델(Codex tmux) 교차검증은 사용자 요청 시**(W4-D 선례 — tmux pane의 codex에 `/tmp/*.md` 브리프 전달).

## 1. 목표 (한 줄) + 왜 지금
W4-E = **cpu-render 공개 API에 `IRIS_SDK_DEPRECATED` 마킹(파일 삭제 금지 — 2.0 제거 고지) + 자체 산출물 .so 16KB 페이지 정렬 검증·확보**. ④의 마지막 표면 정리. cpu-render는 옵션 B(ADR §8.2 — 삭제 아닌 deprecation), 16KB는 §9 실측 확정 결함 대응.

## 2. 현재 상태 — 검증된 사실 (2026-06-18 코드로 확인, 착수 시 재확인)
### cpu-render API (W4-D 후 잔존 — sdk_api.h)
- **`iris_sdk_render_lens`** (`cpp/include/iris_sdk/sdk_api.h:362`) — CPU 렌즈 렌더. **deprecation 핵심 대상**. golden_capture가 CPU 골든 렌더에 사용(보존 필요 — deprecated이되 동작 유지).
- `iris_sdk_load_texture`(:331)·`iris_sdk_load_texture_from_memory`(:343) — CPU 렌더용 텍스처 로딩(cpu-render 계열).
- `iris_sdk_render_with_result`(:979) — IrisResult 주입 후 CPU 렌더(W4-D FrameProcessor 절제에서 보존한 render 경로).
- ⚠️ **`iris_sdk_process`는 W4-D에서 이미 물리 제거됨**(plan 원문 §3 W4-E의 ":376" 참조는 **스테일** — 더 이상 없음). plan의 ":428" 류 line도 W4-D가 sdk_api.h를 바꿔 스테일 가능 → **직접 grep 재확인**.
- **`IRIS_SDK_DEPRECATED` 매크로는 이미 존재**(`cpp/include/iris_sdk/export.h:25` = `__attribute__((deprecated))`, `IRIS_SDK_DEPRECATED_EXPORT`도 :29). **신설 불요 — 적용만.**
- GPU 계열(`iris_sdk_render_lens_texture`:851, `iris_sdk_apply_beauty_texture_v2`:694, `iris_sdk_init_gpu_lens`:784 등)은 **미래 경로 — deprecation 대상 아님**. cpu vs gpu 경계를 ADR §8.2로 확정.
- 소비처 확인 필요: deprecated 마킹 시 내부 호출자(golden_capture·테스트·데모)에서 `-Werror=deprecated` 경고가 빌드를 깨지 않게 — golden_capture는 cpu render를 정당하게 쓰므로 `[[gnu::deprecated]]` 경고 억제(`#pragma`/`-Wno-deprecated-declarations` 국소 적용) 설계 필요(W4-A internal·W4-B 선례 참고).

### 16KB 페이지 정렬 (ADR §9)
- AGP **8.5.0** (`android/build.gradle.kts:10-11`) → §9는 **8.5.1+** 요구.
- **ndkVersion 미고정**(`android/*/build.gradle.kts` 무명시) → r27+ 핀 필요.
- 16KB 플래그(`-Wl,-z,max-page-size=16384` 또는 `ANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON`) **미설정** → 신설.
- **APK arm64-v8a .so 실측**(demo-app-debug APK): `libiris_jni.so`(자체 SDK, **OpenCV 정적 내장 추정 — 별도 libopencv*.so 없음**), `libmediapipe_tasks_jni.so`·`libimage_processing_util_jni.so`(MP Tasks transitive). **TFLite prebuilt .so는 사라짐**(W4-D 코어 TFLite 제거로 자동 해소 — §9 가정 확인됨). → **검증 대상 .so = 이 3종**(자체 iris_jni + MP Tasks 2). 플랜의 "OpenCV(.so) 별도 검증"은 정정: OpenCV는 별도 .so 아닌 iris_jni 내장이므로 iris_jni.so 정렬에 포함.
- 게이트(§9 신설): 최종 AAR/APK 전 .so `objdump --private-headers`에서 `align 2**14` + `zipalign -c -P 16`. iris_jni.so(자체 제어 가능)는 max-page-size 플래그로, MP Tasks 2종(transitive)은 0.10.35가 16KB 정렬 보장하는지 실측(불충족 시 §10 후퇴 트리거).

## 3. 할 일 / 결정 / ⚠️ 절대 제약
**착수 1단계 = 현황 재실측**(§2가 2026-06-18 기준이나 line 스테일 가능). cpu-render 정확한 대상 집합(render_lens 계열만 vs render_with_result 포함)을 ADR §8.2로 확정 + 16KB .so 3종 현재 정렬 상태 objdump 실측.

**⚠️ 절대 제약(불변식)**:
1. **cpu-render는 deprecation만 — 파일/구현 삭제 금지**(2.0 몫, ADR §8.2 옵션 B). OpenCV 잔존은 1.x 정상. `iris_sdk_render_lens`는 deprecated이되 **동작 유지**(golden_capture CPU 골든이 의존).
2. **ABI 보존** — deprecated 마킹은 시그니처 불변(attribute만 추가). 공개 IrisResult 4면 미러 불변.
3. **detector 경로 골든 = injection 모드 baseline**(W4-D 재캡처본). 골든 게이트는 injection vs injection byte-identical(W4-D `golden_capture_all.sh INJECT_BASELINE` + `golden_compare.py`). cpu-render deprecation은 동작 불변이라 골든 PASS 유지해야.
4. **16KB는 자체 .so(iris_jni) 우선** — MP Tasks transitive .so 미정렬 시 §10 후퇴(전환 자체가 0.10.35로 항구 해소 가정이나 실측 필수, Codex 검증대로 '자동 해소' 가정 금지).

## 4. 게이트/검증 (W4-A~D에서 실행 검증된 명령)
1. **데스크톱 빌드**: `cd cpp/cmake-build-debug && cmake .. -DIRIS_SDK_FETCH_TFLITE=OFF && cmake --build . --parallel -- -k 0` → 신규 error/warning 0. (deprecated 마킹이 내부 소비처 경고를 안 깨게 처리.)
2. **골든(동작 불변)**: `INJECT_BASELINE="$PWD/cpp/tests/golden/baseline" bash scripts/golden_capture_all.sh /tmp/p4e && python3 scripts/golden_compare.py --baseline cpp/tests/golden/baseline --candidate /tmp/p4e` → **PASS 불일치 0**(injection vs injection). + `./bin/test_golden_injection_derive` 19/19.
3. **ctest**: `ctest` → 회귀 0. **pre-existing 3건만 허용**: `GPUBeautyBackendTest.FailsWithNullContext`(#401 부근, GPU컨텍스트 부재) + `FreqSepMappingTest` 2(skinQuality stale, 뷰티 트랙). (W4-D로 TFLite NOT_BUILT 2건은 소멸.)
4. **assembleDebug**: `cd android && ./gradlew :demo-app:assembleDebug` → BUILD SUCCESSFUL.
5. **16KB(핵심 신설 게이트)**: 최종 APK 추출 후 전 .so에 `objdump --private-headers <so> | grep LOAD` align 2**14(=16384) 확인 + `zipalign -c -P 16 <apk>`. 3종(iris_jni + MP Tasks 2) 모두 통과.
6. **실기기 무회귀**: cpu-render deprecation은 동작 불변이라 데모 TASKS 경로 무영향 — 육안 1회(선택).

## 5. 워킹트리 주의 (커밋 제외 — 정상 잔존)
의도적 커밋 제외(중단 잔여물 아님): `.claude/settings.local.json`, `android/.../env/env_default_256x128.png`(LFS 팬텀 — 스테이징 금지), `docs/lenssim-handoff/{jaw-vline-warp,natural-lens-fit}*.md`(타 트랙), `cpp/third_party/tflite/`(32MB, **git 미추적** — W4-D로 CMake 참조 0, 물리 삭제는 선택적·커밋 무관). **W4-E 커밋 시 변경 파일만 명시적 `git add`.**
- ⚠️ **`cpp/include/iris_sdk/export.h`는 git-tracked·현재 clean**(generate_export_header 산출물이나 커밋됨 — 워킹트리 잔존물 아님). 빌드가 동일 내용으로 재생성하면 clean 유지. **deprecation 마킹은 export.h가 아니라 `sdk_api.h`의 함수 선언에 `IRIS_SDK_DEPRECATED` attribute를 붙이는 것**(export.h는 매크로 정의만 제공, :25). export.h를 의도 변경할 일은 없을 것.

## 6. 완료 후 + 남은 ④ 전체 로드맵
`docs/workPaper/REFACTOR-4_plan.md` W4-E 상태 갱신 + 분리 커밋(`refactor(core): ④ W4-E — ...`). 메모리 [[refactor-p2-adr-golden]] 진입점 갱신.

**남은 ④ 순서 (이 kickoff가 W4-E만이 아니라 전체를 인계)**:
1. **W4-E** (이 문서) — cpu-render deprecation + 16KB. ④ surface 정리 마무리.
2. **좌표 canonical relabeling (ADR §7.3)** — **별도 슬라이스, 자체 다축 조사부터**. W4-D에서 분리 이월(출력 의미 변경·조사 미흡으로 de-risk). 코어 left/right 468↔473 반전 + 내/외안각 + beauty_roi_manager 명명을 canonical(피험자 해부학) 기준 정정, LensSimulator `com.irislenssdk.tracking.LandmarkIndices`(W4-C로 iris-sdk 이관됨)를 명명 정본. **출력(좌표 의미) 변경 → 골든 재캡처 동반**(W4-D injection 재캡처 인프라 재사용, 단 라벨이 바뀌므로 별도 manifest). 착수 시 plan §3 W4-D 본문 §84 부근 + ADR §7.3 정독 + relabeling 지점 전수 조사(beauty_roi_manager.cpp/h·인덱스 상수·데모 screen_left/right 분리) 필요.
3. **W4-B4** (boundary/visibility 운반 + avg_iris_luma SDK 승격) — plan §3 W4-B4. **비블로커 확인됨**(W4-D 검증: 런타임 TASKS는 `TasksToIrisResult.fillIrisLuma`로 avg_luma 실측 유지, 골든만 -1). copyResultFromJava boundary[1..4]/visibility 정식 운반 + avg_iris_luma를 SDK AAR 계약으로 승격(GPU self-measure 또는 글루 측정-주입). 작은 슬라이스.
4. **④ 종료 → P8 뷰티** (사용자 비즈니스 최우선, 메모리 [[p8-facemesh478-substrate]]): 핵심 2개 = ① 피부 skin smoothing(P8-W1 구현됨) + ② 턱깎기 형태워프(미구현, grid_mesh substrate 보존). 곁가지(LUT/레거시FreqSep/색보정/vivid) 제거.

## 참고 — 트랙 전체 맥락
③-3(A/B 인프라)+frame-sync 완료, develop 미머지. ④ W4-A/B1/B2/B3/C/D 완료(코어 추적 제거+TFLite-free 달성). 검출 폴백=글루(ADR §3). 자세한 트랙 상태는 메모리 [[refactor-p2-adr-golden]]. 작업 패턴: 조사(Workflow/Explore 또는 cpp-pro) → cpp-pro 설계 → (사용자 요청 시 Codex tmux 교차검증) → 구현 → 게이트(골든/ctest/assembleDebug/objdump).
