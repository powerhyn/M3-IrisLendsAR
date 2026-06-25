# P8-W4 — ② 형태워프 (턱 V라인 / 얼굴 슬림 / 눈 확대)

> 상태: ✅ 완료 (2026-06-25 실기기 S23+ 육안 검증 통과 — 턱선 V라인 자연스러움 + 컬러렌즈/홍채 무영향 + 30fps). 범위=**턱 V라인 슬림만**(slim_face/thin_chin), enlarge_eyes 제외.
> 브랜치: `feature/p8-warp` (develop 8fb62fa에서 분기).
> 구현 분할: **W4-A** computeJawWarp CPU 제어점 산출(픽셀공간, roll-robust face_width) + 7 GoogleTest → **W4-B** GPU warp 패스(셰이더+백엔드+파이프라인 gate+applyFaceWarp 구현) → **W4-C** SDK 표면(C API/JNI/Java) + 데모 토글.
> 성격: P8 뷰티 핵심 2개 중 **② = 유일 잔여** (① skin smoothing/radiance 완료). 비즈니스 가치 최우선.
> 정본 입력: `docs/lenssim-handoff/jaw-vline-warp-handoff-from-lenssimulator.md`(알고리즘) + `docs/lenssim-handoff/audit-report.md`(substrate 결함, 2026-06-22 검증).

## 1. 인사이트 (세션 간 맥락 보존) ⭐
- ② = 얼굴 **형태/윤곽 변형**(픽셀 위치 이동) — ①(색/질감)과 차원 다름. 한국 뷰티앱 핵심(턱깎기/V라인/얼굴축소/눈확대).
- 현재 **완전 스텁**: `iris_sdk_apply_face_warp`→`GPUBeautyBackend::applyFaceWarp`(gpu_beauty_backend.cpp)가 NOT_SUPPORTED 반환, grid_mesh 미호출.
- substrate `grid_mesh`+`face_warp_controller` 존재하나 **4결함 확정**(audit, 코드+Codex 검증): #1 RBF 정규화(Shepard)+zero-anchor 제외→자연 falloff 부재(평탄+절벽)+비등방σ, #2 468/478(OOB가드 추가됨, iris center ⑤이월), #3 dy 얼굴크기 미반영·face_width=|Δx| roll비강건, #4 눈확대 정규화거리 왜곡. **전부 프로덕션 미연결(스텁)이라 사용자 영향 0이나 구현 시 이전됨.**
- 핸드오프 §5: "grid_mesh 수리 대신 검증된 비정규 RBF 모델로 **교체**가 단순 — 프래그먼트서 직접 평가(메시 보간 불필요)."

## 2. 확정 알고리즘 (핸드오프 §3·§4 — 재논의 금지, LensSim S23+ 검증)
- **인버스 워프**: 출력픽셀 p의 소스 = `p − Σᵢ dᵢ·exp(−|p−cᵢ|²/2σ²)`, **디스플레이 픽셀 공간**(정규화 금지).
- **제어점 14** = FACE_OVAL 측면 7점×2 (좌:323,361,288,397,365,379,378 / 우:93,132,58,172,136,150,149). **광대정점(454/234)·턱끝(152) 제외**(눈높이 변위0 보장).
- **테이퍼** [0.2,0.45,0.65,0.85,1.0,0.85,0.6] (볼상부→턱끝옆).
- **최대변위** = 얼굴폭(광대간) × 3.2% × strength. **σ** = 얼굴폭 × 13%.
- **변위방향** = 얼굴 세로축(10→152) 수직성분의 반대(안쪽). 좌우대칭·roll불변 자동.
- **셰이더 early-out**: 제어점 bbox + 3σ 마진 밖 루프 생략.
- **입력** = One-Euro 필터링된 FACE_OVAL. **데모 기본 25%**(풀100%=상한).
- **불변식**: 눈높이 변위≈0(렌즈 정합), 적용순서 피부→워프→렌즈, σ 작게(눈높이 테스트 게이트), 퇴화방어(세로축<1px or 광대폭<8px→σ=0).

## 3. 게이트/검증 (예정)
- 데스크톱 빌드 warn0 / ctest 회귀0(pre-existing GPUBeautyBackendTest.FailsWithNullContext만) / assembleDebug / 단위테스트 5종(§6-5) / 골든(워프는 GPU 경로면 CPU 골든 무관, 확인 필요) / 🔴 실기기 S23+ 육안(자연스러운 V라인·렌즈 무영향·30fps).

## 4. 읽을 파일 (브레인스토밍/구현 입력)
- 핸드오프: jaw-vline-warp-handoff (알고리즘 정본), audit-report §grid_mesh/face_warp findings.
- 현재 코드: `cpp/src/warp/grid_mesh.cpp`(+`.h`), `cpp/src/warp/face_warp_controller.cpp`(+`.h`), `cpp/src/gpu/gpu_beauty_backend.cpp`(applyFaceWarp 스텁 + skin composite 경로 + 파이프라인), `cpp/src/gpu/shader_sources.cpp`(SKIN_SMOOTH_COMPOSITE — fragment-direct 선례), `android/.../CameraGLRenderer.kt`(렌즈→뷰티 순서).
- 표면: `cpp/include/iris_sdk/sdk_api.h`(IrisBeautyConfigV2 slim_face/thin_chin/enlarge_eyes + iris_sdk_apply_face_warp).

## 5. 확정 사항 (브레인스토밍 R1 종합 — Codex+Claude 합의, 2026-06-25)
> 참여: Codex+Claude (Gemini auth 미설정 불참). 4/5 완전 합의 + 쟁점3 사용자 결정. synthesis=`P8-W4_brainstorm/synthesis.md`.

1. **아키텍처 = (A) fragment-direct RBF** (합의). 14 제어점 uniform → 컴포지트 셰이더 인버스 워프 직접 평가. grid_mesh는 **미사용 dead substrate로 남김**(불변식 #2 삭제금지 준수; ②=grid_mesh 암묵가정 뒤집힘). 4결함 grid_mesh 수리 안 함.
2. **파이프라인 = (b) 렌즈 합성 후 뷰티 단계 워프, 눈높이0** (합의). 재배열 없음. 워프 패스는 (lens+skin) 합성 프레임을 인버스워프 UV로 리샘플. 순서 카메라→렌즈→skin→**warp**(마지막). 눈높이≈0이 렌즈 보호. **별도 warp 패스**(skin과 독립 — blur/mask 불필요)로 구현, gate=slim_face>0‖thin_chin>0.
3. **SDK 표면 = slim_face/thin_chin → V라인 (공개 config + iris_sdk_apply_face_warp 구동)**. **enlarge_eyes = 본 W 제외**(사용자 결정 2026-06-25 "턱슬림만 먼저, 눈확대 불필요"; 검증 핸드오프 없음 + audit #4). slim_face=메인 강도, thin_chin=하단(턱끝) 제어점 가중(또는 Codex식 jawStrength=clamp(max(thin_chin,0.75·slim_face))). internal API는 보조 A/B만.
4. **좌표공간 = CPU 픽셀 산출 + 셰이더 평가** (합의). 제어점 [cx,cy,dx,dy]·σ·bounds를 CPU에서 디스플레이 픽셀공간(viewportPx) 산출 → uWarp[14] uniform. 셰이더 vUv×viewportPx로 RBF만(정규화 비등방 회피). **미러/회전 단일 변환 헬퍼**(P8-W1 x→1-x 선례). 제어점=워프전 원본 oval(이마확장 미적용)+One-Euro FACE_OVAL.
5. **테스트 = 핸드오프 5종 + 대칭 + roll 2종** (합의). 변위방향/strength≤0/눈높이<5%/3σbbox/퇴화방어 + 좌우대칭 + roll등변(=audit #3 face_width 유클리드 수정 게이트). 합성 타원 얼굴 C++ GoogleTest.

**공통 최대 리스크**: 좌표 미러/회전 픽셀공간 정합 + 눈높이0 렌즈정합 → 단일 변환 헬퍼 + 눈높이<5% 테스트 + 실기기 렌즈정합이 게이트. 깨지면 쟁점2 (a) 렌즈 전 워프로 후퇴.

## 6. 미결 (해소됨)
- ~~쟁점 1~5~~ → §5로 이동(브레인스토밍 R1 종결). enlarge_eyes는 별도 슬라이스 후보(P8-W5, 사용자 "불필요" 의견이라 보류).

## 7. 변경 이력
- 2026-06-25: 착수 + 브레인스토밍 R1 프레이밍. ② 정본 2문서 + audit 4결함 검증 반영. §6 5쟁점 확정.
- 2026-06-25: **W4-A 구현(편집 완료, 빌드/ctest는 메인 세션)**. 신규 순수 기하 모듈
  `cpp/include/iris_sdk/warp/jaw_warp_geometry.h` + `cpp/src/warp/jaw_warp_geometry.cpp`
  (`iris_sdk::jaw_warp` 네임스페이스, GL/OpenCV 무의존). `computeJawWarp`(픽셀공간 제어점
  14 산출, face_width=유클리드 roll-robust, 퇴화방어 축<1px·폭<8px) + `evaluateWarpDisplacement`
  (검증용 비정규 RBF, bbox+3σ early-out). 핸드오프 §1 파라미터 정확 반영(테이퍼
  [0.2,0.45,0.65,0.85,1.0,0.85,0.6]/3.2%/13%/14점/3σ). GoogleTest 7종
  `cpp/tests/test_jaw_warp_geometry.cpp`(방향/strength0/눈높이<5%/3σbbox/퇴화/대칭/roll).
  CMake 등록(라이브러리 소스·헤더 + test_skin_mask_geometry 블록 미러). grid_mesh/
  face_warp_controller 미접촉(dead substrate 불변식 준수).
- 2026-06-25: **W4-A 메인세션 검증 완료**. 빌드 exit0 신규경고0, ctest 571개(564+7) 570통과(유일실패=pre-existing GPUBeautyBackendTest.FailsWithNullContext) 신규회귀0.
  ⚠️ **EyeHeightDisplacementSmall 최초 실패→근본규명**: 합성 타원이 최상단 볼 제어점(323/93)을 눈에서 ~0.5σ에 배치(비현실적)→13% 누출. **실제 골든 랜드마크(Python 복제 검증)에선 눈꼬리 3.7%/홍채 1.6% 전부 <5% = 알고리즘 정확**. 따라서 눈높이 테스트만 **실제 face_mesh fixture**(face_closeup__rot0.result.json 24점 스냅샷)로 교체([[real-data-first]]) → 통과. 나머지 6테스트는 합성 유지(수학 속성). **눈높이≈0 렌즈정합 = 실제 데이터로 입증**(브레인스토밍 #1 리스크 해소). **다음=W4-B GPU 워프 패스.**
- 2026-06-25: **W4-B 구현(편집 완료, 빌드/ctest/assembleDebug는 메인 세션)**. GPU 워프 패스 + 파이프라인 통합.
  - **셰이더**: `WARP_FRAGMENT`(shader_sources.cpp) — 풀스크린 fragment-direct 비정규 RBF 인버스 워프(핸드오프 §4 이식). 유니폼 `uTexture`/`uWarp[14]`(cx,cy,dx,dy px)/`uWarpCount`/`uWarpSigma`/`uWarpBounds`/`uViewportPx`. 핸드오프의 `vUv`→SDK 규약 `vTexCoord`로 정합(FULLSCREEN_QUAD_VERTEX out). bbox+3σ early-out, sigma=0이면 패스스루. shader_manager.h extern 추가.
  - **백엔드**: `warp_program_` + `WarpUniforms` 캐시 구조체 + `executeWarpPass(input,output_fbo,w,h,JawWarpParams)`(SoA→vec4 AoS 패킹 후 glUniform4fv). initializeShaders 등록(non-fatal, skin smoothing 패턴) + cacheUniformLocations 캐시 + release 리셋. jaw_warp_geometry.h include.
  - **파이프라인**(applyTextureId): `needs_warp` gate=`config.enabled && (slimFace>0‖thinChin>0) && detected && face_mesh_valid && warp_program_!=0`, active_filter_count 포함. **skin→brightness→warp 순서 마지막 패스**. scissor OFF(전체 프레임, skin 1345 패턴). `jawStrength=clamp(max(slimFace,thinChin),0,1)`.
  - 🔴 **좌표 정합**(최대 리스크): `alignWarpToRenderSpace` 헬퍼 — computeJawWarp 출력(원본 비미러·top-down 이미지 픽셀)을 렌더 텍스처 공간(미러·Y-flip)으로 변환. **prepareSkinFans 동형**: prepareSkinFans가 `mx=(1-x)·W`(미러)+NDC `ny=1-2y`(Y-flip)이므로 ⇒ `cx'=W-cx, cy'=H-cy, dx'=-dx, dy'=-dy`(선형변환 부호반전), bbox는 X·Y 뒤집혀 min↔max 교차, σ 유지. warp 셰이더가 skin composite와 동일 `vTexCoord` 공간 샘플이라 정합.
  - **face_mesh 타입**: `detection->face_mesh`는 `iris_sdk::IrisLandmark[478]` = computeJawWarp 인자 타입 → **캐스팅 불필요**(C `::IrisResult`→`iris_sdk::IrisResult` reinterpret_cast는 C API 경계 sdk_api_v2.cpp:440에서 이미 수행, 레이아웃 동일 POD).
  - **applyFaceWarp**: standalone 진입점은 **NOT_SUPPORTED 스텁 유지**(config 경로 applyTextureId가 정본 — 텍스처 풀 deferred-release/패스 체인 우회 방지). 주석만 P4→P8-W4-B로 정정.
  - 보존: grid_mesh/face_warp_controller 미접촉, jaw_warp_geometry(W4-A) 미수정(사용만), skin/brightness/렌즈 무손상(추가 패스), 공개 ABI 미변경(slimFace/thinChin 기존 필드). 비-GPU 빌드 #if IRIS_SDK_GPU_AVAILABLE 가드.
  - **다음=메인세션 빌드/ctest/assembleDebug 검증 + W4-C(SDK 표면/데모 토글) + 실기기 S23+ 육안.**
- 2026-06-25: **W4-B 메인세션 검증 완료**: 데스크톱 빌드 exit0 신규경고0(warp 파일 0, skin_target_* pre-existing) + ctest 571개 570통과(유일실패=pre-existing) 신규회귀0.
- 2026-06-25: **W4-C 데모 토글 + 빌드 검증**. GpuRenderActivity `btnP8Slim`(off→0.25→0.50, beautyConfig.slimFace 설정 + setBeautyConfig) + 레이아웃 btnP8Slim 위젯(btnP8Radiance 인접). applyFaceWarp 별도 C API/JNI는 스텁 유지(config 경로가 정본). **assembleDebug(iris-sdk+demo): 네이티브 W4-B(NDK arm64) + Kotlin W4-C 컴파일·패키징 성공**(compileDebugKotlin+packageDebug 에러0). ⚠️ installDebug는 S23+ 무선 adb 세션 만료로 "No connected devices" — **빌드는 통과, 설치만 기기 재연결 대기**.
  🔴 **잔여=실기기 검증**: 기기 재연결 후 S23+ 설치 → btnP8Slim(slim:off→0.25→0.50) 토글로 턱 V라인 워프 육안. **좌표 정합(미러/회전, alignWarpToRenderSpace)·렌즈 무영향·30fps**가 device 게이트(W4-B #1 리스크). 어긋나면 alignWarpToRenderSpace 부호 조정.
- 2026-06-25: **✅ 실기기 S23+ 육안 검증 통과 → ② 형태워프 종결.** 데모 UI 개편(풀스크린+오버레이 바텀시트+뷰티 단계형 슬라이더, 커밋 6ce760e)으로 턱슬림을 기본 0.2(20%)·연속 단계로 노출 → 사용자 육안 확인: **턱선 V라인 자연스러움 + 컬러렌즈 정합(슬림 0→0.5 올려도 홍채 위 렌즈 무밀림 = 눈높이 변위≈0 불변식 실증) + 30fps**. `alignWarpToRenderSpace` 미러·Y-flip 좌표 정합 정상 — **부호 조정 불필요**(W4-B #1 리스크 해소). **P8 뷰티 핵심 2개(① skin smoothing/radiance + ② 턱 V라인 워프) 완성.** 다음=`feature/p8-warp` → develop 머지(내부 W 머지 컨벤션=PR 생략, `git merge --no-ff`).
</content>
