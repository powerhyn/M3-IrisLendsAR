# REFACTOR-4: 추적 레이어 주입형 전환 (④) — 실행 계획

> 새 세션이 이 문서 + ADR-0001만 읽고 ④를 실행할 수 있도록 작성. 작성: 2026-06-15.
> 상태: 🔄 **④ 진행 중** — **W4-B1**(방향 A)+**W4-B2**(축소, enum/메타 W4-D 이월)+**W4-A**(internal 헤더 776302f + 에러코드 501→100; 기본값은 P8 이월) 완료. 검출 폴백(Eye-Only)=글루 결정(ADR §3). 다음=**W4-B3**(DetectionSlot 재설계+ts원자성) → W4-C/D/E. **④ 완료 후 P8 뷰티**(사용자 확정 2026-06-16: 핵심=① 피부 skin smoothing[P8-W1 구현됨] + ② 턱깎기 형태워프[미구현, grid_mesh substrate 보존]; 곁가지 LUT/레거시FreqSep/색보정/vivid 제거).
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
- **미기록**: 사용자의 공식 '전환 확정(TASKS로 전환한다)' 선언 + ④ 착수 승인. → **이 문서 승인이 곧 게이트0 통과.**

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
- **B3 — JNI/Java/Kotlin result + 슬롯 원자성**: Java/Kotlin 2벌 → JNI 필드매핑 생성/검증, **IrisResultKt 정리**(detector 메타 잔존+avg_luma/faceMesh 부재 — ADR §47-53 removalScope, Codex 놓침 지적), faceRect 좌표 단위 문서 드리프트 정정(Java=pixel vs 데모=normalized). **DetectionSlot 정식 재설계 + ts-슬롯 원자 번들링**(frame-sync 1프레임 스큐 해결, 후속문서 §3) — **W4-C A/B의 선결**(스큐 남으면 A/B가 추적기/동기버그 분리 불가, Codex critical).
- **B4 — boundary/visibility 운반 + avg_iris_luma 승격**: `copyResultFromJava` boundary[1..4]/visibility 정식 운반. **avg_iris_luma를 SDK AAR 계약으로 정식 승격**(GPU self-measure 이전 또는 글루 측정-주입 — '또는'이 아니라 **필수**, 미승격 시 W4-D서 P7-W2 default ON 조용히 퇴화, Codex HIGH).

### W4-C — 글루 iris-sdk AAR 승격 (게이트=A/B + 재캡처 + AAR 패키징)
- demo-app `tracking/` 12파일 → iris-sdk AAR로 승격(현재 iris-sdk에 FaceTracker/CoordMapper/TasksToIrisResult 없음). **복사 아님 — public API·lifecycle·model asset·의존 publishing·ProGuard·ABI까지 동반하는 API 전환**(Codex).
- tasks-vision 0.10.35 → iris-sdk `build.gradle.kts`(현재 demo만), face_landmarker.task → SDK 번들(현 데모 assets 중복 해소).
- 데모는 SDK 추적 API 소비로 전환, **TASKS 단일 경로**(LEGACY 토글은 A/B 검증 종료 후 제거 가능). **선결: B3 ts 원자성**(A/B 채널 오염 방지).
- **AAR 패키징 게이트(Codex 놓침, ADR T4)**: model asset SHA/버전 고정, consumer ProGuard 규칙, 의존성 충돌 점검, AAR/APK 크기 회귀(T4 후퇴 트리거).

### W4-D — removalScope 자체 추적 제거 + 좌표 canonical 정정 (게이트=A/B + 명시적 재기준선)
**W4-C로 SDK가 추적을 책임진 후에만 안전**(현 LEGACY 기본 — 먼저 지우면 데모·골든 깨짐). iOS/Web은 소스 0개 빈 스텁이라 코어 제거에 영향받는 활성 소비자 없음(§0):
- 코어: mediapipe_detector(150KB)+.h, inference_thread, iris_detector, frame_processor(검출/NV21·NV12/비동기 submitFrame).
- sdk_api detect 계열, sdk_manager createDetector, JNI nativeDetect*/confidence 계열.
- **types.h enum/메타 (W4-B2에서 이월 — W4-B2 조사 2026-06-16)**: `DetectorType`(types.h:88)/`EyeRefinerPolicy`(:99) enum + detector 전용 메타(`iris_quality_*`/`eye_refiner_used`) 필드. 소비처(iris_detector 죽은 팩토리·mediapipe_detector·frame_processor·sdk_manager createDetector + 테스트 5종)가 전부 이 단계 제거 대상이라 **동작 불변(W4-B2)으로 못 지움** → detector 인프라와 원자 삭제. 메타 필드 제거는 IrisResult 레이아웃 변경(offsetof 가드/JNI 매핑/Java IrisResult/골든 18벌 동시 수정)이라 **골든 재캡처 동반**. **코어 폴백 미보유 확정**(ADR §3 검출 폴백 결정 2026-06-16 — Eye-Only는 글루 책임)이라 EyeOnly/Hybrid 자리표시(구현 0)까지 완전 삭제. C enum `IrisEyeRefinerPolicy`+no-op `iris_sdk_set_eye_refiner_policy`(호출자 0)는 SDK surface라 W4-E deprecation(ADR §8.2 패턴).
- TFLite CMake 블록(cpp/CMakeLists.txt:224-537) + cpp/third_party/tflite/ + .tflite/.task 에셋.
- 추적 의존 테스트 12파일(test_mediapipe_detector*/test_iris_detector/test_frame_processor/test_integration 등) + examples 2종.
- **좌표 canonical left/right 라벨 정정 (ADR §7.3 — Codex 지적, ⑤에서 이동)**: 코어 'left'=468-472=MediaPipe FACEMESH_RIGHT 반전 + 내/외안각 인덱스 반전 + beauty_roi_manager 명명 반전을 canonical(피험자 해부학) 기준으로 일괄 정정, LensSimulator LandmarkIndices.kt를 명명 정본으로. **출력이 바뀌므로 아래 재기준선과 한 묶음**(동작 불변 아님이라 W4-B 아닌 여기).
- **골든**: 추적 교체+라벨 정정은 이종/출력 변경이라 ε 일치 불가 → A/B(§10 T1) + **명시적 베이스라인 재캡처**.
- **재캡처 manifest 강제 (Codex HIGH, 무언의 재기준선 차단)**: Tasks 버전·model SHA·기기·입력 corpus·before/after 메트릭·luma/visibility/timestamp invariant·승인자를 파일로 기록해야 재기준선 인정.

### W4-E — cpu-render deprecation + 16KB 재정렬 (게이트=objdump/zipalign)
- **cpu-render(옵션 B, §8.2)**: iris_sdk_render_lens(sdk_api.h:428)·iris_sdk_process(:376) 등 공개 CPU 픽셀 API에 `IRIS_SDK_DEPRECATED` 마킹 + 구현 동결 + '2.0 제거' 고지. **파일 삭제 금지(2.0)**, OpenCV 잔존은 1.x 정상.
- **16KB(§9 대응1)**: AGP 8.5.0→8.5.1+(android/build.gradle.kts:10-11), ndkVersion r27+ 핀(현재 미고정), `-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON` 또는 `-Wl,-z,max-page-size=16384`. 검증 게이트 신설: **최종 AAR/APK의 전 .so 전수** objdump --private-headers align 2**14 + zipalign -c -P 16. TFLite prebuilt 2종은 W4-D 제거로 자동 해소되나, **OpenCV(.so) — iris-sdk CMakeLists:136 `find_package(OpenCV REQUIRED)` — 와 MediaPipe Tasks transitive native lib는 잔존하므로 별도 검증 대상**(Codex 검증, '자동 해소' 가정 금지).

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
