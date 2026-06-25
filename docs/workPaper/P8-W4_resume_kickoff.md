# P8-W4 ② 형태워프 — 재개 킥오프 (실기기 검증 지점)

> 이 문서 + 아래 필독으로 **대화 맥락 0인 새 세션이 P8-W4 실기기 검증 지점부터 착수**. 작성: 2026-06-25.
> ② 턱 V라인 워프 = **구현·빌드 완료, S23+ 설치됨. 남은 것 = 실기기 육안 검증 1건** + 결과별 후속.

## 0. 진입 절차
1. **필독(커밋된 문서 — 자급자족 입력)**: 이 문서 → `docs/workPaper/P8-W4_face_warp.md`(전체 W 문서 — §2 알고리즘/§5 브레인스토밍 확정/§7 구현 이력) → 정본 `docs/lenssim-handoff/jaw-vline-warp-handoff-from-lenssimulator.md`(알고리즘). 본 kickoff는 이 3개 + 저장소 코드만으로 착수 가능(메모리 불요).
   - 참고(보조, **블로킹 필독 아님**): `[[p8-w2-legacy-removal]]`/`[[p8-facemesh478-substrate]]`/`[[skip-pr-for-internal-w-merges]]`/`[[real-data-first]]`는 Claude 자동 로드 메모리(`~/.claude/projects/.../memory/`, MEMORY.md 인덱스)라 새 Claude 세션엔 컨텍스트로 떠 있음 — serena `.serena/memories/`엔 없음(혼동 주의). 본 kickoff는 이들 없이도 자급자족(머지 규칙 등 load-bearing은 아래 인라인).
2. **브랜치**: `feature/p8-warp` (tip `85657c8`, **미머지**). develop=`8fb62fa`(W2+W3 머지본)에서 분기. 커밋: d51ac87(브레인스토밍)·269d6ae(W4-A)·48480cb(W4-B/C)·85657c8(slim→뷰티탭 fix).
3. **빌드**: 데스크톱 `cpp/cmake-build-debug` 재사용(`cmake .. -DIRIS_SDK_FETCH_TFLITE=OFF && cmake --build . --parallel -- -k 0`, 새 빌드 디렉토리 금지). Android `cd android && ANDROID_SERIAL="adb-R3CW20BBFCM-HZeo2S._adb-tls-connect._tcp" ./gradlew :demo-app:installDebug`(S23+=SM-S916N에만; A23=SM-A235N 제외).
4. **C++ 작업이므로 `systems-programming:cpp-pro` 동반**(CLAUDE.md). 배경: 백그라운드 sub-agent는 빌드/ctest/gradle 권한 프롬프트 못 받음 → **편집은 에이전트, 빌드/검증은 메인 세션** 분업.

## 1. 목표 (한 줄) + 왜 지금
② 턱 V라인 슬림 워프(slim_face)가 **S23+ 실기기에서 자연스럽게·올바른 위치에·렌즈 무영향으로** 동작하는지 육안 확인. 통과 시 develop 머지로 ② 종결(P8 핵심 2개 완성). 이게 W4의 유일 잔여 게이트 — 좌표 미러/회전 정합은 데스크톱서 검증 불가, 실기기가 유일 수단.

## 2. 현재 상태 — 검증된 사실 (2026-06-25 코드/빌드 확인)
- **구현 완료**(전부 커밋·빌드 검증):
  - W4-A: `cpp/src/warp/jaw_warp_geometry.cpp` + 헤더 `cpp/include/iris_sdk/warp/jaw_warp_geometry.h` `computeJawWarp`(픽셀공간 14 제어점, 핸드오프 §1) + `evaluateWarpDisplacement`. 7 GoogleTest(`cpp/tests/test_jaw_warp_geometry.cpp`) 통과. **눈높이<5% = 실제 골든 랜드마크로 검증(3.7%)**.
  - W4-B: `cpp/src/gpu/shader_sources.cpp` `WARP_FRAGMENT`(인버스 RBF) + `gpu_beauty_backend.cpp:1167 executeWarpPass` + 파이프라인 통합(`:1324 needs_warp` gate=`enabled&&(slimFace‖thinChin)>0&&face_mesh_valid`, skin→brightness→warp 순서, scissor off, `:1329 jawStrength=clamp(max(slimFace,thinChin),0,1)`).
  - W4-C: 데모 `GpuRenderActivity.kt:495 btnP8Slim`(off→0.25→0.50, beautyConfig.slimFace) — **뷰티 탭**에 전체폭 버튼 "턱슬림(slim): off"(렌즈탭 행 폭초과로 이동, 85657c8).
- **게이트 통과**: 데스크톱 빌드 신규경고0 / ctest 571개 570통과(유일 실패=`GPUBeautyBackendTest.FailsWithNullContext`=pre-existing, 곁가지·워프 무관) / assembleDebug(NDK arm64+Kotlin) SUCCESS / **S23+ 설치됨**(현재 APK=85657c8).
- **🔴 좌표 정합 fix 지점**: `cpp/src/gpu/gpu_beauty_backend.cpp:65 alignWarpToRenderSpace` — 현재 `cx'=W-cx, cy'=H-cy, dx'=-dx, dy'=-dy`(미러X + Y-flip, prepareSkinFans 동형 추정). **이게 실기기서 워프 위치/방향이 틀릴 때 조정할 곳**(아래 §3).

## 3. 할 일 = 실기기 결과별 후속 (사용자 육안 보고 받고 분기)
**테스트 방법**: S23+ 데모 → **뷰티 탭** → `Beauty: ON` → `턱슬림(slim)` 토글 off→0.25→0.50 → 얼굴 비추며 관찰.

| 실기기 결과 | 진단 | 후속 행동 |
|---|---|---|
| **턱선 갸름 + 올바른 위치 + 렌즈 무영향 + 30fps** | ✅ 성공 | **W4 종결 → develop 머지.** 프로젝트 컨벤션=**내부 W 머지는 PR 생략, `git switch develop && git merge --no-ff feature/p8-warp`**(W2+W3도 동일했음). 머지 후 origin push는 사용자 확인. W4 문서 status ✅. |
| **턱은 줄지만 위치/방향 엉뚱** | 좌표 미러/Y-flip 부호 오류 | `alignWarpToRenderSpace`(`gpu_beauty_backend.cpp:65`) 부호 조정. **증상→부호 매핑**: ⓐ 좌우만 반대(왼턱 깎이는데 오른쪽 줄어듦) → **미러X 토글**(cx'=cx, dx'=dx로). ⓑ 상하만 반대(턱 대신 이마쪽 일그러짐) → **Y-flip 토글**(cy'=cy, dy'=dy로). ⓒ 위치는 맞는데 바깥으로 부풂(슬림 반대) → **dx/dy만 부호 반전**(변위 방향). ⓓ 완전 엉뚱 → 둘 다 검토. bbox(min↔max 교차)도 cx/cy 변경에 맞춰 대응. 재빌드+S23+ 재설치+재확인. |
| **렌즈/홍채가 어긋남**(slim 켜면 렌즈 밀림) | 눈높이 변위 누출(W4-A는 3.7% 통과했으나 실기기 좌표공간서 누출 가능) | 1차(quick): σ↓/strength↓ 또는 제어점 최상단(323/93) 테이퍼↓로 눈높이 기여 축소. **2차(큰 재설계, quick fix 아님)**: 브레인스토밍 쟁점2 후퇴 = (a) 렌즈 합성 **전** 워프 — 현 skin→brightness→warp(렌즈 후)를 렌즈 전 패스로 재배치. `synthesis.md`엔 *결정만* 있고 구현 지침 없음 → **별도 스코핑/재브레인스토밍 필요**(파이프라인 순서 변경 + 렌즈가 워프된 프레임 위에 그려지도록). 저확률(W4-A 검증)이라 1차 우선. |
| **워프 안 보임/변화 없음** | gate 미충족 | Beauty ON·얼굴검출(face_mesh_valid)·slimFace>0 확인. `adb logcat | grep "P8-W4"` (onClick 로그 "P8-W4 slim face → 0.25"). needs_warp gate(:1324) 디버그. |

⚠️ **절대 제약(불변식)**:
1. **grid_mesh/face_warp_controller 미사용 dead substrate — 삭제·수정 금지**(불변식, ②는 fragment-direct로 별도 구현). jaw_warp_geometry가 ② 정본.
2. **공개 ABI 불변**: slim_face/thin_chin은 기존 IrisBeautyConfigV2 필드(W2-D 보존). 새 config 필드 추가 금지.
3. **skin smoothing/radiance/brightness/렌즈 경로 무손상**(워프는 추가 패스, 기존 패스 순서·동작 불변).
4. **enlarge_eyes 본 W 제외**(사용자 결정 — 검증 핸드오프 없음 + audit #4). 추가 금지.
5. **이식 정본 = jaw-vline 핸드오프 값 무변경**(테이퍼/3.2%/13%/제어점14).

## 4. 게이트/검증
- 데스크톱: `cd cpp/cmake-build-debug && cmake --build . --parallel -- -k 0`(신규 warn0) → `ctest`(570/571 통과, 유일 실패 GPUBeautyBackendTest.FailsWithNullContext 허용).
- Android: `cd android && ANDROID_SERIAL="adb-R3CW20BBFCM-HZeo2S._adb-tls-connect._tcp" ./gradlew :demo-app:installDebug` → BUILD SUCCESSFUL + Installed on 1 device.
- 골든: 워프=GPU 경로라 CPU beauty.png 골든 무관(재캡처 불필요).
- 🔴 실기기 육안(위 §3 표) — 유일 미완 게이트.

## 5. 워킹트리 주의 (커밋 제외 — 정상 잔존)
- `.claude/settings.local.json`(로컬 설정), `android/demo-app/src/main/assets/env/env_default_256x128.png`(Git LFS 팬텀 — 영구 ` M`, **스테이징 금지**). 중단 잔여물 아님.
- 커밋 시 `cpp/include/iris_sdk/export.h`가 빌드 재생성 diff로 뜨면 `git checkout`으로 되돌림(W2~W4 내내 동일).

## 6. 완료 후
- 실기기 통과 → `git merge --no-ff feature/p8-warp`(develop) → (자동 로드 메모리 사용 환경이면) 메모리 `project_p8_w2_legacy_removal.md`/`project_p8_facemesh478_substrate.md`를 ② 완료로 갱신 + MEMORY.md 인덱스. P8 핵심 2개(skin smoothing+radiance / 턱 V라인 워프) 완성.
- 후속 슬라이스 후보(보류): enlarge_eyes(P8-W5, 사용자 "불필요" 의견), thin_chin 별도 가중 정밀화, jawStrength 매핑 재튜닝.
</content>
