# REFACTOR-4: 추적 레이어 주입형 전환 (④) — 실행 계획

> 새 세션이 이 문서 + ADR-0001만 읽고 ④를 실행할 수 있도록 작성. 작성: 2026-06-15.
> 상태: 🔄 **④ 진행 중** — **W4-B1**(방향 A)+**W4-B2**(축소, enum/메타 W4-D 이월)+**W4-A**(internal 헤더 776302f + 에러코드 501→100; 기본값은 P8 이월)+**W4-B3**(DetectionSlot ts 원자 번들)+**W4-C**(글루 9파일 → iris-sdk AAR 승격, Option A)+**W4-D**(자체 추적 코어 제거: mediapipe_detector/inference_thread/iris_detector + detect surface + LEGACY 데모 + enum/메타 물리삭제 + TFLite 제거 + injection 골든 재캡처; 코어 TFLite-free) 완료. 검출 폴백(Eye-Only)=글루 결정(ADR §3). **W4-E**(cpu-render deprecation 5종 + Java/Kotlin 미러 + 16KB: 자체 .so 2**14 달성, CameraX 1.3.1→1.4.2로 transitive .so까지 정렬; 전 게이트 + **실기기 무회귀 2대(SM-S916N/A235N) 통과**, 2026-06-18) 완료. **W4-B4**(avg_iris_luma SDK 계약 승격 — convertWithLuma 단일진입 + IrisResultKt luma 노출; 조사+Codex 교차검증으로 visibility=drop·boundary/outlier게이트=stabilizer 재튜닝 트랙 이월로 재정의; W4-C 테스트 부채 3종 iris-sdk 이관 부수해소; output-invariant, 2026-06-18) 완료. **좌표 canonical relabeling(§7.3, ④ 마지막 슬라이스)**: left/right 라벨 canonical 정정(필드명 유지+그룹 값 스왑) + R2 안각 + R3 미러 §7.4 — **데스크톱 완료(2026-06-19: 골든 재캡처+migration checker new.left==old.right PASS, ctest 회귀0, assembleDebug)**, 🔴 **전면 카메라 실기기 검증 대기**(R3 미러 데스크톱 미검증). 통과 시 ④ 완전 종결. **④ 완료 후 P8 뷰티**(사용자 확정 2026-06-16: 핵심=① 피부 skin smoothing[P8-W1 구현됨] + ② 턱깎기 형태워프[미구현, grid_mesh substrate 보존]; 곁가지 LUT/레거시FreqSep/색보정/vivid 제거).
> 근거: ADR-0001(승인 2026-06-11) §3/§6/§8/§9/§12, REFACTOR-3-3 §9 이월, 스코핑 워크플로(wf_207a105f-dcc 수확 + wf_4f1249e6-81a 연속), **Codex 외부 검증 + critical-review(2026-06-15)**.

## 0. 핵심 사실 (스코핑 실측 2026-06-15)

- ④ **추적 코어 제거는 100% 미착수** — mediapipe_detector.cpp(150KB)·inference_thread·iris_detector(죽은 nullptr 팩토리)·frame_processor 한 줄도 안 빠짐.
- ③-1~③-3가 만든 **주입 경계(C API `iris_set_landmarks`/`iris_get_landmark_generation`/`iris_get_injected_result` + JNI 3종 + 데모 글루 12파일 + 골든 베이스라인)** 는 ④가 재활용할 토대 — 완성됨.
- §6.4 선행 5종 중 **[4]468 하드코딩·[5]offsetof 가드는 ③-2가 처리**(`cpp/src/warp/grid_mesh.cpp:183/186/206` landmark_count 파라미터+`<468` 가드, beauty_roi_manager.cpp:99/105, sdk_api_v2.cpp:44-79). resize(468)→478 **확장**만 ⑤ 보류. **[1]에러코드·[2]기본값·[3]internal 헤더는 미처리** → W4-A.
- ADR §12.3 게이트3(데모 검증 통로 정화: torn 스냅샷·KT 무음 폴백·OverlayView 정책)은 별도 트랙에서 완료(7a02fa0)로 추정 — 착수 전 1회 재확인 권장.
- **⚠️ 주입 경로 confidence=0 함정 (Codex 검증, 코드 확인)**: `landmark_injection.cpp:158` deriveIrisResult가 confidence를 0으로 둠(ADR §6.2 "경계에서 제거, EAR 게이팅"). 그러나 `eye_render_packet_adapter.cpp:89`는 `visibility = confidence·(1-eyelid_ratio)` — confidence=0이면 **visibility=0 → 렌즈 미렌더**. 데모 TASKS 경로는 `TasksToIrisResult.kt`가 confidence=1.0을 별도 주입해 회피 중. 주입을 정식 공급자로 승격(W4-B)하기 **전에** 이 모순(ADR 의도 vs 어댑터 수식)을 반드시 해소. **이번 ④의 단일 최우선 선결.**
- **플랫폼 상태(Codex 우려 검증)**: `ios/`,`flutter/`,`web/`는 **소스 0개 빈 스텁**, 코어 detect 참조 0, 빌드 스크립트 android 전용. 따라서 W4-D 코어 추적 제거는 **활성 소비자가 없어 안전** — 미래 iOS/Web은 ADR §3대로 각자 MediaPipe Tasks 글루로 추적(제거된 코어 detector 미사용). ADR §57의 "iOS/Web 후 제거" 순서는 미구현 상태에서 공허하게 충족.

## 1. 게이트0 — 착수 전제 (사용자 승인 필요)

ADR §10/§12 + REFACTOR-3-3 §7.4: ④ 착수는 **A/B 판정 통과 + 사용자 '전환 확정' 선언 + ④ 별도 승인**이 전제다.
- T1 통과(계통 오프셋 반경 6.9%, 미러/스왑 없음 — 2026-06-15). T2(품질 육안)는 frame-sync로 실기기 정합 성공(거울 지연 +2~3프레임 trade-off 수용).
- 실기기 A/B: B3(SM-S916N)·W4-C(SM-A235N) LEGACY/TASKS 양 모드 정합 무회귀 통과.
- ✅ **게이트0 통과 (2026-06-17)**: 사용자 '전환 확정(TASKS로 전환한다)' 선언 + W4-D 착수 승인 받음. 전환의 본질=유지보수·아키텍처 이득(코어 ~5,200줄+TFLite 제거, 단일 추적 경로, Google 검증 강건성)이며 **검출 지연은 LEGACY가 더 빨랐으나(TASKS ~27ms vs LEGACY ~11ms) frame-sync로 체감 허용 수준 + 그 지연이 강건성/QA 비용을 일부 포함**임을 사용자가 인지하고 확정. **TASKS 단계별 프로파일링은 ④ 전환 완료 후로 이월**(결정 번복 가능성 낮음, 근거 보강용).

## 2. 목표·범위

**목표**(ADR §3): 추적(TFLite/MediaPipe)을 C++ 코어 밖으로 분리. 코어 = 지오메트리+이펙트(입력: 카메라 텍스처 + 478 랜드마크 + 파라미터). 플랫폼 글루가 MediaPipe Tasks로 추적. **Android 먼저.**

**범위 내**: §6.4 선행([1][2][3]), types.h 정리 + 주입 승격, 데모 글루 → iris-sdk AAR 승격, 자체 추적 removalScope 제거, cpu-render deprecation 마킹, 자체 .so 16KB 재정렬.
**범위 외(⑤/2.0/후속)**: grid_mesh 478 확장, 좌표 canonical 라벨 정정, iOS/Web 글루, cpu-render 물리 삭제(2.0), 코어 TFLite 물리 잔재 최종 삭제.

## 3. 단계 (W4-A → E, 의존 순서)

### W4-A — §6.4 표면 정리 — 🔶 부분 완료 (2026-06-16; internal+에러코드 완료, 기본값 뷰티 이월)
신규 주입 함수 승격 전 기존 C API 표면 결함 정리(ADR §6.4). 3축 조사(wf_3ab20fa3)로 각 항목이 ABI/언어/제품 제약으로 일부 축소·이월됨.
- **internal 9종 헤더 선언화 — ✅ 완료 (776302f)**: `iris_jni.cpp` 수동 extern 9종(sclera_veto, env_map/reflection 4종, blink/gate/detail/luma 4종)을 `cpp/include/iris_sdk/internal/bench_toggles.h`로 정식화. 정의(sdk_api_v2.cpp)+소비(JNI) 양쪽 include로 시그니처 강제, install에서 internal/ 제외. 순수 선언 정리(동작·ABI 불변). 검증: 빌드/골든/ctest/assembleDebug 통과.
- **에러코드 정본 단일화 — ✅ 완료 (커밋 예정)**: NotInitialized 정본을 `IRIS_SDK_NOT_INITIALIZED=100`으로 일원화 — v2/beauty/gpu 16곳 `=501`→100, v2 switch/errorToString/NotInitialized 테스트 2개 정합. `IRIS_SDK_ERROR_NOT_INITIALIZED=501`은 deprecated alias로 **ABI 보존**(2.0 삭제). 효과: v2 GPU 경로의 Kotlin Unknown-오분류 잠복결함 해소. 소비처 0이라 '의도된 surface 정정'(W4-B1 류, 동작 불변 아님). 검증: 빌드/골든(에러코드 미검증→무영향)/ctest 회귀0(pre-existing 5)/assembleDebug exit0.
- **BeautyFilterConfigV2 기본값 단일소스화 — ⏭️ P8 뷰티 트랙 이월**: 실제 4원 분기(C++ Helper/Java DEFAULT_/C 미러/문서) + 드리프트 실재(문서 smoothing 0.5/softFocus 0.3 vs 정본 0.0; **enabled 충돌** C미러=1 vs Java/C++=false). 핵심인 enabled 정본 통일이 'P8 뷰티 기본값 재정의'(skin smoothing이 enabled 게이트 의존)에 묶이므로 P8 뷰티 트랙에서 곁가지 정리와 함께 처리. 진짜 단일정의는 언어 경계로 불가 → 현실 해법=정본 명문화+드리프트 가드+문서 정정.

### W4-B — types.h 정리 + 주입 승격 (Codex 권고로 B1~B4 분할, 회귀 원인 분리)
한 PR로 묶지 않고 독립 PR 4개로 — 각 단계 동작 불변(게이트=골든 일치), 단 B1의 confidence 수정은 의도된 동작 변경(주입 경로 한정).
- **B1 — confidence/visibility 계약 정정 (선결 critical) — ✅ 완료 (2026-06-16, 방향 A)**: 주입 경로 visibility=0(렌즈 미렌더) 결함 정정. **방향 A 채택**(cpp-pro 기술검토 + 적대검증 5종 반증 실패로 일치): 어댑터(`eye_render_packet_adapter.cpp:89`)·detector를 건드리지 않고, `deriveIrisResult`가 검출 시 `confidence = detected ? 1.0f : 0.0f`(게이트 통과 상수)를 설정 → 어댑터 `visibility = confidence·(1-eyelid_ratio)`가 (1-eyelid_ratio)로 환원되어 ADR §6.2 'visibility 일원화'를 곱셈 항등원으로 달성. detector 경로 무수정이라 **골든 비트 불변 자명**. 방향 B/C(어댑터 분기·sentinel)는 detector 공유 수식이라 골든 위험·표면 확대로 배제. **데모 `TasksToIrisResult.kt:122` confidence=1.0은 건드리지 않음**(킥오프 정정: DetectionSlot 경로용이며 제거는 W4-C 몫 — 데모는 주입 경로 미사용이라 B1은 C-API surface latent 결함 정정 + Kotlin 형제와의 일관화). **검증**: 골든 PASS(JSON 18/PNG 19, 불일치 0, detector ε 불변) + `test_landmark_injection` 신규 3케이스(InjectedDetectedEyeYieldsPositiveVisibility 등, visibility>0 회귀가드) 통과 + ctest 회귀 0(pre-existing 5건만: TFLite NOT_BUILT 2 + GPUBeauty 1 + FreqSep 2) + assembleDebug exit0(데모 무영향). ADR §6.2 구현 노트 정합 + types.h confidence 주석 정정.
- **B2 — layout/API 단일화 — ✅ 완료 (2026-06-16, 축소 수행)**: 4축 조사(wf_48f426f7) 결과 plan 원안(enum/메타 제거)은 소비처가 전부 W4-D 제거 대상(iris_detector·mediapipe_detector·frame_processor·sdk_manager + 테스트 5종)이라 **동작 불변으로 불가** → **W4-D 이월**(위 W4-D 항목 참조). IrisResult 단일화 안전조건(필드별 offsetof static_assert 21개)은 `sdk_api_v2.cpp:44-74`에 이미 완비(ADR §6.3 충족), `iris_set_landmarks` §6.1 가드도 ③-1 완비. **B2 실수행(동작 불변, cpp-pro 검토 동반)**: (1) `sizeof(::IrisResult)` static_assert를 GLES `#ifdef` **밖**으로 이동 — 기존엔 안에 있어 non-GLES(데스크톱) 빌드에서 죽어, offsetof가 못 잡는 trailing-padding 드리프트가 미검출되던 구멍 수정(실버그). (2) types.h/sdk_api.h의 W4-D 이월 대상(DetectorType/EyeRefinerPolicy enum·IrisEyeRefinerPolicy C 미러·iris_quality_*/eye_refiner_used 메타)에 `@deprecated W4-D` 주석(C/C++ 미러 동기). (3) `iris_set_landmarks`를 '정식 랜드마크 공급 경계 승격'으로 문서화(코드는 ③-1 완비라 무변경) + `iris_sdk_set_eye_refiner_policy` @todo를 'W4-D 삭제/W4-E deprecation(ADR §8.2)'으로 갱신. **검증**: 빌드 통과(sizeof 가드가 데스크톱서 C/C++ 레이아웃 일치 확인) + 골든 PASS(불일치 0, detector ε 불변) + ctest 회귀 0(pre-existing 5: TFLite NOT_BUILT 2+GPUBeauty 1+FreqSep 2). 빌드가 C 블록주석 `*/` 조기종료 1건 잡음(수정). enum/메타 물리 제거는 W4-D(detector 인프라 + 골든 재캡처와 원자).
- **B3 — JNI/Java/Kotlin result + 슬롯 원자성 — ✅ 완료 (2026-06-16)**: 4축 조사(wf_81d464e2) + cpp-pro 적대검증으로 설계 확정. **핵심 결정**: DetectionSlot은 `iris_jni.cpp` 익명 namespace 내부 구조체(공개 4면 미러 IrisResult ABI 아님)라, ns ts·detected를 슬롯에 번들해 **공개 ABI/골든/ctest 전부 무영향**으로 스큐 제거 가능. 실수행:
  - **(④ 핵심) DetectionSlot ts-슬롯 원자 번들** — 스큐 원인은 데모 두 채널(volatile `latestLandmarkFrameTsNs`(ns) vs 네이티브 슬롯 좌표) happens-before 미보장. 해소: `DetectionSlot`에 `int64_t frame_ts_ns` 추가(plain, active_slot_index release store 이전 기록 → data와 원자 publish), `nativeUpdateDetectionSlot(result, frameTsNs)`로 ts 동반 주입, **신규 단일 스냅샷 reader `nativeGetActiveDetectionSlot(long[])`**(active index 1회 acquire load로 ptr·ts·detected 일관 반환 — getDetectionSlotPtr 다중 호출 race 대체). 데모 onDrawFrame이 1회 스냅샷으로 배경 ts·렌즈 좌표·렌즈 게이트(detected)를 **모두 같은 슬롯**에서 취득 → 1프레임 스큐 + cpp-pro 지적 제3 채널(`irisResult?.detected` 게이트=406, 비-volatile data race) 동시 해소. `setLandmarkFrameTimestamp`/`latestLandmarkFrameTsNs` 폐기. `getDetectionSlotPtr`는 @deprecated 보존(ABI). updateDetectionSlot 게이트는 기존 유지(detectWithRotation always-OK 계약 검증, ts 번들이 격차 해소).
  - **(③) faceRect 단위 주석 정정** — C++/JNI/데모 모두 normalized[0,1]인데 `IrisResult.java` 주석만 "픽셀" 오기(드리프트, TasksToIrisResult.kt:128이 이미 감사 finding 기록). 4필드+클래스 주석 정규화로 정정(런타임 무변경, ADR §7.1).
  - **(②) IrisResultKt 정리** — detector 메타(iris_quality_*/eye_refiner_used)에 `[W4-D 삭제 예정]` 문서 주석을 C++ types.h 마커와 정합(Java IrisResult에도 누락분 추가). faceMesh·avg_iris_luma 부재는 의도적 경량 DTO 선택(faceMesh→W4-C, avg_luma→W4-B4 승격)임을 KDoc 명문화. **물리 삭제 없음**(W4-D). 어노테이션 미사용(C++ Doxygen 패턴 미러, 내부 소비처 경고 회피).
  - **(①) JNI 매핑 검증** — C/C++/Java/JNI 4면 매핑 정확 일치 + offsetof/sizeof static_assert 완비 확인(sdk_api_v2.cpp:40-84), 신규 getActiveDetectionSlot이 detected 매핑 행사. 자동 검증기는 2.0 이월.
  - **검증**: assembleDebug BUILD SUCCESSFUL(exit0, JNI 시그니처 정합) + 골든 PASS(불일치 0, JSON18/PNG19, detector ε 불변) + ctest 회귀 0(pre-existing 5: NOT_BUILT 2+GPUBeauty 1+FreqSep 2) + cpp 코어 빌드 신규 err/warn 0. **실기기 fsync ON 렌즈 정합 무회귀 통과(2026-06-17, SM-S916N — LEGACY/TASKS 양 모드 잘 따라옴, 회귀 없음).**
  - **이월 명문화**: pre-existing torn-read 윈도우(LEGACY ~11ms<GL ~16ms 구간, dead generation 가드 미사용) — ts 번들이 악화 안 시킴(찢겨도 동일 슬롯=정합), generation 필드는 미래 seqlock 자리로 보존. W4-D/별도 트랙 후보.
- **B4 — avg_iris_luma SDK 계약 승격 — ✅ 완료 (2026-06-18)**. **조사+Codex 교차검증으로 plan 원안 3항이 재정의됨**(wf_748491fd 3축 조사 + codex read-only 적대검증):
  - **① visibility 운반 → DROP (할 일 아님)**: `IrisResult`(types.h)에 result-level visibility 필드가 **없다**. visibility는 adapter(eye_render_packet_adapter.cpp:89)가 `confidence·(1-eyelid_ratio)`로 매 프레임 재산출(B1 단일 진실원). 운반할 필드 없음 + 추가 시 B1 계약 충돌. Codex 동의.
  - **② boundary[1..4] 운반 → stabilizer 재튜닝 트랙 이월**: 활성 GPU 렌더는 center+radius만 소비(boundary 미사용). 유일 소비처=temporal_stabilizer `smoothEye`가 iris[1]로 outlier `norm_radius` 산출. **단 게이트는 이중으로 죽어있음**: (1) Java가 iris[1]=0 운반 → norm_radius ~20× 팽창 **+** (2) `outlier_confirm_frames` 기본=1이라 reject 분기(`1<1`) 미도달(Codex 정정 — 내 초기 "boundary=동작변경" 주장 과장). 즉 boundary 운반은 활성 경로 **output-invariant**이고 게이트 복원은 norm_radius fix + confirm_frames≥2 **튜닝**이 함께라야 → stabilizer near-raw 재튜닝 트랙으로 이월. (부수 발견: stabilizer config doc drift — sdk_api.h 주석 "기본 2.0/2" vs 실제 4.0/1 → 본 커밋서 정정.)
  - **③ avg_iris_luma SDK 계약 승격 → 글루 측정-주입 정식화 (GPU self-measure 아님)**: GPU self-measure는 과거 EXTERNAL_OES+FBO+glReadPixels 검은화면 회귀로 revert(고위험·인프라 부재) → **글루 정식화 채택**. 코어 계약(types.h avg_iris_luma_*, EyeRenderPacket, adapter >0 가드, updateAvgIrisLuma fallback, sdk_api 라운드트립)은 이미 완비, `use_measured_luma_` 기본 true(실측 ON — doc drift "false" 정정). 실수행: (a) `TasksToIrisResult.convertWithLuma`(convert+fillIrisLuma 단일 진입) 신설 — SDK 단독 소비자도 1콜로 avg_iris_luma 채운 완전 IrisResult 취득(데모는 같은 전경 lm 전달로 얼굴 정합 유지=Codex 리스크 회피, output-invariant), (b) `IrisResultKt`(Kotlin DTO)에 avgIrisLumaLeft/Right 노출(계약 완결), (c) FaceTracker irisLumaEma(Rec.601 단일스칼라)→TrackingSnapshot.avgIrisLuma 발산 경로는 진단/레거시로 doc 명시(정본=per-eye Rec.709 fillIrisLuma).
  - **부수: W4-C 테스트 부채 해소** — W4-C가 글루 소스만 iris-sdk로 git mv하고 데모 테스트 3종(TasksToIrisResultTest/CoordMapperTest/IrisGeometryTest) 참조 미갱신 → assembleDebug가 test 미컴파일이라 잠복. iris-sdk/src/test로 이동(internal 접근 복원, 패키지 정정). 발견 경위=W4-B4 testDebugUnitTest 첫 실행.
  - **검증**: assembleDebug BUILD SUCCESSFUL(iris-sdk+demo) + `:iris-sdk:testDebugUnitTest` TasksToIrisResultTest 11(신규 convertWithLuma 2 포함)/CoordMapperTest 19/IrisGeometryTest 4 전부 0 실패 + output-invariant(데모 동작 동일=같은 측정·같은 얼굴). 실기기 검증은 output-invariant라 선택(권장 스모크).

### W4-C — 글루 iris-sdk AAR 승격 — ✅ 완료 (2026-06-17, Option A)
4축 조사(wf_ae07efae) + 사용자 결정 = **Option A(글루 이관 + 데모 오케스트레이션 유지)**. 완전 캡슐화(analyze→IrisResult 자동 슬롯)는 LEGACY 제거하는 **W4-D**로(LEGACY/A/B 공존 중엔 대칭 오케스트레이션 필요). 실수행:
- **글루 9파일 → iris-sdk `com.irislenssdk.tracking[.math]`**(git mv): FaceTracker·TasksToIrisResult·TrackingSnapshot·LandmarkIndices·**EmulatorDetector**(FaceTracker가 의존 → SDK가 데모 역의존 불가라 동반 이동, 조사 분류 정정)·math/{CoordMapper·IrisGeometry·IrisLumaSampler·OneEuroFilter}. **데모 잔존**: AbMeasure(A/B 하니스, SDK import 추가)·math/FrameRingSelector(렌더 글루, CameraGLRenderer 전용). 가시성: 데모 소비 4종(FaceTracker·TasksToIrisResult·TrackingSnapshot·EmulatorDetector) `internal`→`public`, LandmarkIndices+math 4종 `internal` 유지(SDK 표면 최소화, 컴파일러가 공개-노출 누수 0 확인).
- **의존성**: iris-sdk += `api("com.google.mediapipe:tasks-vision:0.10.35")` + `api("androidx.camera:camera-core:1.3.1")`(FaceTracker가 ImageProxy·FaceLandmarkerResult를 public 시그니처 노출 → api 필수). demo -= tasks-vision(전이 제공). 버전 핀 SDK 단일 관리.
- **모델 에셋**: `face_landmarker.task` → iris-sdk `assets/models/`(git mv, SHA `64184e22…`), 데모 중복 2벌(assets/·assets/models/) 제거, MediaPipeBenchmarkActivity 경로 "models/"로 정합. AAR 병합으로 APK에 단일 제공(이전 2벌 → 1벌, **APK ~3.6MB 감소**).
- **consumer-rules.pro**: `com.irislenssdk.tracking.**` + `com.google.mediapipe.**` keep + dontwarn 추가.
- **데모 전환**: GpuRenderActivity/AbMeasure가 SDK tracking 패키지 import. **LEGACY/TASKS 토글·오케스트레이션·frame-sync(FrameRingSelector) 보존**(Option A, 무회귀). LEGACY 제거는 W4-D.
- **검증**: assembleDebug BUILD SUCCESSFUL(exit0) + APK 모델 단일 병합·.so 중복 0(ABI당 1) + 골든 PASS(불일치 0, detector ε 불변 — cpp/ 무변경) + ctest 회귀 0(pre-existing 5) + AAR 패키징 게이트(T4: model SHA·의존성 충돌 0·크기 회귀=감소). **실기기 A/B 무회귀 통과(2026-06-17, SM-A235N — LEGACY/TASKS 양 모드 잘 따라옴, 추적 글루 모듈 재배치 무회귀).**
- **이월**: LEGACY 제거·완전 캡슐화·canonical 라벨·메타 물리삭제 → W4-D / 16KB 전수검증(OpenCV+MP Tasks .so) → W4-E / publishing(maven-publish)·재캡처 manifest 도구화 → 2.0.

**원안(참고)**: demo-app `tracking/` → iris-sdk AAR 승격(복사 아님 — public API·lifecycle·model asset·의존·ProGuard·ABI 동반 API 전환). tasks-vision·모델 SDK 이동. TASKS 단일 경로는 A/B 종료 후(LEGACY 제거 W4-D). AAR 패키징 게이트(ADR T4).

### W4-D — removalScope 자체 추적 제거 + 좌표 canonical 정정 (게이트=A/B + 명시적 재기준선)

> **실행 상태 (2026-06-18)**: ✅ **W4-D 코어 제거 완료** — 1~3단계(골든 injection 재배선/회전치수/18골든 검증, c2104e2 선커밋) + 5~9단계(leaf-first 제거) + 11단계(injection 재캡처 + manifest) 완료. **10단계=좌표 canonical relabeling은 별도 후속 슬라이스로 분리 이월**(출력 의미 변경·조사 미흡으로 de-risk; 이번 재캡처는 현 라벨 유지). **실기기 TASKS 단일경로 무회귀 통과(2026-06-18, SM-A235N+SM-S916N 2대 — 클린 재설치·실행·크래시0 스모크 + 육안 추적 무회귀) → W4-D 완전 종결.** 다음=W4-E + 별도 relabeling 슬라이스.
> **11단계 재캡처 (2026-06-18)**: detector 제거로 detect-mode 캡처 불가 → injection 모드로 baseline 18 JSON+19 PNG 재생성(동결 478점 재주입, 현 라벨). diff=메타3 제거+confidence→1.0+avg_luma→-1+frame_w/h 추가, geometry·render 시각 ε-동일. manifest: `cpp/tests/golden/W4-D_RECAPTURE_MANIFEST.md`. **golden 게이트 idempotency PASS**(injection vs injection byte-identical). `golden_capture_all.sh` `INJECT_BASELINE` 모드가 재생성 도구.
>
> **이번 제거 내역(5~9단계)**:
> - **5단계 (detect 표면 4면)**: JNI `nativeDetect`/`nativeDetectWithRotation`/`nativeProcess` 제거. `IrisLensSDK.java` public `detect`/`detectWithRotation`/`process` + native 선언 제거. Kotlin `IrisLensSDKKt` `detect`/`process`/`detectOnly` + `Extensions.kt` `detectIris` 확장 제거(미사용 import/`reusableResult` 정리).
> - **6단계 (C API + sdk_manager)**: `sdk_api.cpp`에서 `iris_sdk_detect`/`_detect_with_rotation`/`_process`/`submit_frame`×2/`get_latest_result` 하드 제거, `init`/`init_with_config`를 render-only(`FrameProcessor::initialize()` 인자 없음)로 정정. `set_min_*_confidence`/`set_use_inference_thread`/`is_using_inference_thread`는 **no-op stub 보존(ABI)**. `convert_error_code`는 `[[maybe_unused]]`로 매핑 정본 보존. `sdk_api.h` 선언 정리(`IrisEyeRefinerPolicy`/`set_eye_refiner_policy` 보존). `sdk_manager` `createDetector` 제거, `createFrameProcessor` 보존.
> - **7단계 (FrameProcessor 절제, render-only)**: detector 멤버(`inference_thread_`/`direct_detector_`/`detector_type_`/`use_inference_thread_`/min*Confidence/`face_tracking_`/cache) + 메서드(`process`×2/`detectOnly`*/`submitFrame`*/`getLatestResult`/setMin*/`setFaceTracking`/`setUseInferenceThread`/`isUsingInferenceThread`/getFaceLandmark*/getModelVersion) 제거 + `ProcessResult` 구조체 제거. **보존**: `renderer_`/`work_buffer_`/`renderOnly`/`renderWithResult`/`loadLensTexture`/`hasLensTexture`/convert*/`setGpuEnabled`/`isUsingGpu`.
> - **8단계 (코어 삭제 + 메타 4면)**: `mediapipe_detector`/`inference_thread`/`iris_detector` `.{cpp,h}` 6파일 삭제. `types.h` `DetectorType`/`EyeRefinerPolicy` enum + `iris_quality_*`/`eye_refiner_used` 제거. `sdk_api.h` C 미러 동일 3필드 제거. `sdk_api_v2.cpp` offsetof assert 3줄 제거. JNI(jni_utils.h/iris_jni.cpp field-id+Set/GetField)·`IrisResult.java`(필드/reset/copyFrom/toString)·`IrisResultKt.kt`(생성자/디폴트/fromJava) 메타 3필드 제거. **`eyelid_ratio_*`(W3)·`avg_iris_luma_*`(P7-W2) 전면 보존**. 최종 필드 순서 C++/C 동일(offsetof+sizeof assert 통과).
> - **9단계 (테스트/예제/CMake/TFLite)**: detector 의존 테스트 8종 삭제(`test_lens_renderer_integration` 포함 — SetUp부터 MediaPipeDetector 의존이라 보존 불가, 순수 단위 `test_lens_renderer`가 커버). `test_sdk_api`/`test_sdk_manager`/`test_types` 제거 API/enum 케이스 정리. examples `camera_demo`/`image_demo` 삭제(`golden_capture`는 **injection 전용**으로 정정, `hello_iris` 보존). `cpp/CMakeLists.txt` TFLite 블록 + detector 소스 제거, `cpp/tests/CMakeLists.txt` dead TFLite 블록 5개 제거.
>
> **게이트 결과 (전부 통과)**: ① 데스크톱 빌드 exit 0, 신규 error/warning 0, **코어 .a TFLite/tensorflow 심볼 0개(TFLite-free 확정)**, injection 심볼(`iris_set_landmarks`/`iris_get_injected_result`/`deriveIrisResult`) 생존. ② **`test_golden_injection_derive` 19/19 PASS**(geometry 보존 핵심 증거 — injection 경로 무손상). ③ ctest 683개 중 FAIL 3건=pre-existing만(GPUBeauty#531 + FreqSep#698/699), **회귀 0**(TFLite NOT_BUILT 2건은 테스트 삭제로 소멸). ④ **Android `assembleDebug` BUILD SUCCESSFUL**(`--rerun-tasks` 강제 재빌드 포함 — 메타 4면 정합). ⑤ offsetof/sizeof static_assert 데스크톱 통과.
>
> **미처리(사용자 검증 단계)**: `cpp/third_party/tflite/`(32MB) = **git 미추적**이라 `git rm` 불가(커밋 무관). 안드로이드 JNI CMake는 주석 처리된 참조뿐(활성 링크 0, assembleDebug 통과)이라 물리 삭제 가능하나 안드로이드 클린 빌드 영향 우려로 11단계와 함께 처리 권장. **baseline 재캡처는 미수행**(11단계, 사용자 검증 후).

**W4-C로 SDK가 추적을 책임진 후에만 안전**(현 LEGACY 기본 — 먼저 지우면 데모·골든 깨짐). iOS/Web은 소스 0개 빈 스텁이라 코어 제거에 영향받는 활성 소비자 없음(§0):
- 코어: mediapipe_detector(150KB)+.h, inference_thread, iris_detector, frame_processor(검출/NV21·NV12/비동기 submitFrame).
- sdk_api detect 계열, sdk_manager createDetector, JNI nativeDetect*/confidence 계열.
- **types.h enum/메타 (W4-B2에서 이월 — W4-B2 조사 2026-06-16)**: `DetectorType`(types.h:88)/`EyeRefinerPolicy`(:99) enum + detector 전용 메타(`iris_quality_*`/`eye_refiner_used`) 필드. 소비처(iris_detector 죽은 팩토리·mediapipe_detector·frame_processor·sdk_manager createDetector + 테스트 5종)가 전부 이 단계 제거 대상이라 **동작 불변(W4-B2)으로 못 지움** → detector 인프라와 원자 삭제. 메타 필드 제거는 IrisResult 레이아웃 변경(offsetof 가드/JNI 매핑/Java IrisResult/골든 18벌 동시 수정)이라 **골든 재캡처 동반**. **코어 폴백 미보유 확정**(ADR §3 검출 폴백 결정 2026-06-16 — Eye-Only는 글루 책임)이라 EyeOnly/Hybrid 자리표시(구현 0)까지 완전 삭제. C enum `IrisEyeRefinerPolicy`+no-op `iris_sdk_set_eye_refiner_policy`(호출자 0)는 SDK surface라 W4-E deprecation(ADR §8.2 패턴).
- TFLite CMake 블록(cpp/CMakeLists.txt:224-537) + cpp/third_party/tflite/ + .tflite/.task 에셋.
- 추적 의존 테스트 12파일(test_mediapipe_detector*/test_iris_detector/test_frame_processor/test_integration 등) + examples 2종.
- **좌표 canonical left/right 라벨 정정 (ADR §7.3 — Codex 지적, ⑤에서 이동)**: 코어 'left'=468-472=MediaPipe FACEMESH_RIGHT 반전 + 내/외안각 인덱스 반전 + beauty_roi_manager 명명 반전을 canonical(피험자 해부학) 기준으로 일괄 정정, LensSimulator LandmarkIndices.kt를 명명 정본으로. **출력이 바뀌므로 아래 재기준선과 한 묶음**(동작 불변 아님이라 W4-B 아닌 여기).
- **골든**: 추적 교체+라벨 정정은 이종/출력 변경이라 ε 일치 불가 → A/B(§10 T1) + **명시적 베이스라인 재캡처**.
- **재캡처 manifest 강제 (Codex HIGH, 무언의 재기준선 차단)**: Tasks 버전·model SHA·기기·입력 corpus·before/after 메트릭·luma/visibility/timestamp invariant·승인자를 파일로 기록해야 재기준선 인정.

### W4-E — cpu-render deprecation + 16KB 재정렬 — ✅ 완료 (2026-06-18, 실기기 무회귀 통과)

> **실수행 (2026-06-18)**:
> - **cpu-render deprecation (5종, ADR §8.2 옵션 B — 5종 스코프 사용자 확정)**: 공개 CPU 픽셀 C API 5종에 `IRIS_SDK_DEPRECATED`(export.h:25 기존 매크로) 마킹 + `@deprecated` Doxygen 고지(2.0 제거, GPU 경로 이행). 대상: `iris_sdk_render_lens`·`iris_sdk_render_with_result`·`iris_sdk_load_texture`·`iris_sdk_load_texture_from_memory`(전부 g_processor CPU 경로 — GPU `iris_sdk_load_lens_texture`(sdk_api_v2)와 별개임을 코드 검증) + `iris_sdk_apply_beauty_v2_c`(CPU 뷰티, sdk_api_v2.cpp — 킥오프 §2 미열거였으나 ADR §8.1 "CPU 뷰티 계열"로 포함 확정). 시그니처/구현/ABI 불변. GPU 계열(render_lens_texture/apply_beauty_texture_v2/load_lens_texture/init_gpu_*)은 미래 경로라 비대상. export.h 미수정(빌드 재생성 NOLINT diff는 되돌림).
> - **소비처 경고 억제**: golden_capture.cpp(3건 국소)·test_sdk_api.cpp(렌더 섹션 1 push/pop)·iris_jni.cpp(JNI 래퍼 3건 국소)에 clang 호환 `#pragma GCC diagnostic ignored "-Wdeprecated-declarations"`. 정의부는 deprecated 함수끼리 cross-call 없어 억제 불요(실측).
> - **표면 일관성 미러 (사용자 확정)**: Java `@Deprecated(forRemoval=true)` 4종(loadTexture/loadTextureFromAssets/loadTextureFromMemory/applyBeautyFilterV2) + Kotlin `@Deprecated`+`@Suppress("DEPRECATION")` 3종(텍스처 래퍼). C API만이 아닌 외부 Java/Kotlin 소비자에게도 고지 도달([[sdk-surface-consistency]]).
> - **16KB (ADR §9 대응1)**: AGP 8.5.0→8.5.1, ndkVersion `27.0.12077973` 핀(신설), `-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON`(iris-sdk externalNativeBuild). 자체 `libiris_jni.so` 2**14 달성.
> - **⚠️ 킥오프 §2 .so 출처 정정 (실측)**: 미정렬(2**12) 발견된 `libimage_processing_util_jni.so`는 **MP Tasks transitive가 아니라 CameraX `camera-core`** 산출물(킥오프 §2/plan 원문의 "MP Tasks 2종"·"OpenCV(.so) 별도검증" 분류 부정확). OpenCV는 libiris_jni 정적 내장(별도 .so 0). 실제 MP Tasks `libmediapipe_tasks_jni.so`는 0.10.35에서 이미 16KB(ADR §9 가정 확인). → §10 후퇴(MediaPipe 되돌리기)는 무관. **해법=CameraX 1.3.1→1.4.2**(iris-sdk api + demo 단일관리, 1.4.2 .so 16KB objdump 사전확인). camera-core를 iris-sdk가 api()로 전파하므로 SDK 패키징 결함 성격(ADR §9 동류, §9 실측 범위 밖 신규 발견). 1.4.2가 `libsurface_util_jni.so` 추가(이것도 16KB).
>
> **게이트 결과 (전부 통과)**: ① 데스크톱 빌드 44/44, 신규 error/warning 0(-Wdeprecated 0건, cpp-pro 동반). ② **골든 idempotency PASS**(injection vs injection, JSON18/PNG19 불일치 0 — deprecation 동작 불변 입증) + `test_golden_injection_derive` 19/19. ③ ctest 676/679, FAIL 3=pre-existing만(GPUBeautyBackendTest#401 + FreqSepMappingTest#587/588), 회귀 0. ④ `assembleDebug` BUILD SUCCESSFUL(NDK r27 native 재빌드 + JNI/Java/Kotlin deprecation 경고 누출 0). ⑤ **16KB PASS** — APK arm64-v8a 전 .so 4종(libiris_jni/libmediapipe_tasks_jni/libimage_processing_util_jni/libsurface_util_jni) objdump align 2**14 + zipalign -P 16 Verification successful.
>
> **실기기 무회귀 통과 (2026-06-18, SM-S916N + SM-A235N 2대)**: CameraX 1.3.1→1.4.2 버전 업에도 양 기기에서 데모 카메라·홍채 추적 무회귀(사용자 육안 "둘 다 잘 따라와"). cpu-render deprecation은 동작 불변이라 무영향. **→ W4-E 완전 종결.**

**원안 스펙(참고)**:
- **cpu-render(옵션 B, §8.2)**: `iris_sdk_render_lens`(sdk_api.h:362, W4-D 후 실측)·`iris_sdk_render_with_result`(:979)·`iris_sdk_load_texture`(:331/343) 등 공개 CPU 픽셀 API에 `IRIS_SDK_DEPRECATED`(export.h:25 기존 매크로) 마킹 + 구현 동결 + '2.0 제거' 고지. ⚠️ **`iris_sdk_process`는 W4-D에서 이미 물리 제거됨**(plan 원문 :376 스테일). **파일 삭제 금지(2.0)**, OpenCV 잔존은 1.x 정상. **착수 진입점=`docs/workPaper/REFACTOR-4_W4-E_kickoff.md`**(검증된 현황·게이트·남은 ④ 로드맵).
- **16KB(§9 대응1)**: AGP 8.5.0→8.5.1+(android/build.gradle.kts:10-11), ndkVersion r27+ 핀(현재 미고정), `-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON` 또는 `-Wl,-z,max-page-size=16384`. 검증 게이트 신설: **최종 AAR/APK의 전 .so 전수** objdump --private-headers align 2**14 + zipalign -c -P 16. TFLite prebuilt 2종은 W4-D 제거로 자동 해소되나, **OpenCV(.so) — iris-sdk CMakeLists:136 `find_package(OpenCV REQUIRED)` — 와 MediaPipe Tasks transitive native lib는 잔존하므로 별도 검증 대상**(Codex 검증, '자동 해소' 가정 금지).

### 좌표 canonical relabeling (ADR §7.3) — ✅ 데스크톱 완료 (2026-06-19), 🔴 실기기 검증 대기
> ④ 마지막 슬라이스(W4-D에서 분리 이월). 출력 좌우 의미 변경 → 골든 재캡처 동반. 정밀 스펙: `docs/workPaper/REFACTOR-4_relabel_spec.md`. 조사 wf_2b44bca7(4축) + **Codex read-only 적대검증 2회** 반영.

- **핵심**: system B의 left/right 라벨 전면 반전(자기일관)을 canonical로 정정 — **공개 필드명 left/right 유지 + 인덱스 그룹 값 스왑**(API rename 회피). 정정 후 `left_*`=피험자 좌안(473/362그룹), `right_*`=피험자 우안(468/33그룹). (canonical 체계 A=LandmarkIndices/FaceTracker/TrackingSnapshot은 이미 정합, 무변경.)
- **R1 코어 라우팅 스왑**(cpp-pro): landmark_injection.h(kLeftIris/kRightIris, kLeftEAR/kRightEAR), eye_render_packet_adapter·temporal_stabilizer 눈꺼풀 상수, gpu_lens_renderer 윤곽/눈꺼풀, face_warp_controller(iris+contour+eyebrow triple), types.h 주석. + 글루 TasksToIrisResult(RESULT_LEFT/RIGHT_IRIS·EAR).
- **R2 안각 inner/outer 정정**: gpu_lens_renderer.cpp:62-65 side+corner 이중반전(LEFT_INNER 33→362, OUTER 133→263, RIGHT_INNER 263→133, OUTER 362→33). **GPU ellipse 전용(기본 OFF)→골든 무관**, device ellipse 시 눈-타원 비대칭 정상화(진짜 버그수정, fitEyeEllipse rx_inner=0.85/rx_outer=1.0 비대칭이 셰이더 `d.x<0?radii.x:radii.y`로 실소비 — Codex 확인).
- **R3 미러 §7.4 정합**(cpp-pro): is_mirror의 X-flip은 유지, **eye-swap(std::swap left/right) 전면 제거** — 홍채(986-993)+눈꺼풀(1043-1046)+**타원(1100-1112)+alpha(1145-1150)**(스펙 밖이나 cpp-pro가 §7.4 일관성 위해 추가발견·제거). GPU 전용·CPU 골든 is_mirror no-op이라 데스크톱 미검증 → **전면 카메라 실기기 검증 필수**.
- **골든 재캡처**: 18 JSON injection 재캡처(left↔right 값 교환), **CPU PNG byte-invariant**(양안 동일 렌더). `scripts/golden_relabel_migration_check.py` 신규 — **invariant `new.left==old.right` + 나머지 불변 + PNG byte-eq PASS로 스왑 정확성 기계 증명**(golden_compare는 key별 엄격이라 별도 checker 필요 — Codex 지적). manifest=`cpp/tests/golden/RELABEL_RECAPTURE_MANIFEST.md`.
- **OverlayView 정리**: 자체 inverted 상수(LEFT_IRIS_CENTER=468 등)+해부학/화면 혼용 주석 → canonical(검증도구 신뢰성, device-verify 前).
- **테스트**: cpp test_landmark_injection/test_eye_enlargement(cpp-pro), Kotlin TasksToIrisResultTest L/R 단언 스왑(메인).
- **게이트 결과(데스크톱 전부 통과)**: 데스크톱 빌드 신규 warn0 / migration checker PASS / ctest 676/679(pre-existing 3만, 회귀0) / iris-sdk testDebugUnitTest 0실패 / assembleDebug SUCCESS(native 재컴파일).
- **잔여(🔴 사용자 게이트)**: **전면 카메라 실기기 육안**(R3 미러 상쇄/이중반전 없는지 + R2 안각 변화) → 통과 시 ④ 완전 종결 + 분리 커밋. **미러 R3가 데스크톱 미검증 핵심 위험.**
- **이월(canonical 완전성 후속, 동작중립)**: `beauty_roi_manager.cpp` LEFT/RIGHT_EYE_INDICES 라벨(union mask라 출력/테스트 무영향) — cpp-pro 보고, 별도 정정 후보.

## 4. 머지 게이트 (PR별, ADR §12)
- **동작 불변 단계(W4-A/B, 단 B1 confidence는 주입 경로 한정 의도 변경)**: 골든 ε 일치. **추적 교체+라벨 정정 단계(W4-C/D)**: A/B(T1) + 명시적 베이스라인 재캡처 + **재캡처 manifest**(Tasks 버전/model SHA/기기/corpus/before-after/invariant/승인자).
- ctest 회귀 0(pre-existing 외). 16KB(Android 산출물 변경 시): **최종 AAR/APK 전 .so** objdump align 2**14 + zipalign -c -P 16(OpenCV·MP Tasks .so 포함).
- AAR 패키징(T4): model SHA·ProGuard·의존성 충돌·크기 회귀.
- 실기기 A/B 토글 육안(사용자 판정).

## 5. 리스크
- R1 좌표 계약(전환 시 upright/미러/L-R 정합) / R2 detector 메타 필드 소실 / R3 avg_iris_luma 소실(P7-W2 회귀) / R4 torn-read / R5 ABI 변경.
- 게이트0 미기록(§1) — 착수 전 해소. 코어 삭제 시점(W4-D)이 글루 승격 완료에 종속.

## 6. 이월 (④ 범위 밖)
⑤: grid_mesh resize(468)→478 확장 + geometry 수식 수리. (좌표 canonical 라벨 정정은 ADR §7.3 따라 ④ W4-D로 이동 — Codex 지적 반영.) 2.0: cpu-render 물리 삭제, 코어 TFLite 잔재 최종 삭제. 후속: iOS(Swift 글루)/Web(TS 글루) 전환 — 현재 빈 스텁이라 ④ 코어 제거에 무영향, 구현 시 각자 Tasks 글루로 추적(ADR §3). **Eye-Only 검출 폴백(글루 트랙, Phase 9+)**: MediaPipe Tasks 얼굴 검출 실패(눈만 클로즈업/극단 각도) 시 Eye-Only `.tflite` 폴백 — ADR §3 결정(2026-06-16)으로 **플랫폼 글루 책임**(코어 밖, TFLite 글루 위임해 코어 .so TFLite-free 유지). 코어는 추적 미보유 유지. Eye-Only 모델 출력(눈 주변 점)→478 주입 계약 변환 설계 동반.

## 7. 견적
**20~28 사람·일 (1인 4~6주)**. ADR §13 원견적(18~25인일+cpu-render 2~4일)에 W4-B 분할·confidence 정정·AAR 패키징·재캡처 manifest·16KB 전수검증 부담 반영(Codex 권고로 상향 — 초안 15~22는 낙관적).

## 변경 이력
- 2026-06-15: 스코핑 워크플로 결과로 작성 (게이트0 승인 대기).
- 2026-06-15: Codex 외부 검증 + critical-review 반영 — confidence=0→visibility 0 선결 critical 신설(B1), W4-B를 B1~B4 분할, ts 원자성을 W4-C A/B 선결로, canonical 라벨 정정 ⑤→W4-D 이동(ADR §7.3), recapture manifest·AAR 패키징·OpenCV 16KB 게이트 추가, IrisResultKt 정리 추가, 견적 20~28로 상향. iOS/Web 위험론은 빈 스텁 확인으로 기각(제거 안전 명시).
