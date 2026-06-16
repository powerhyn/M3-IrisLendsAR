# REFACTOR-4: 추적 레이어 주입형 전환 (④) — 실행 계획

> 새 세션이 이 문서 + ADR-0001만 읽고 ④를 실행할 수 있도록 작성. 작성: 2026-06-15.
> 상태: ⏳ **게이트0(전환 확정 선언 + ④ 승인) 대기** — 승인 후 W4-A(또는 B1 confidence 선결) 착수.
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

### W4-A — §6.4 표면 정리 (동작 불변, 게이트=골든 일치)
신규 주입 함수를 정식 승격하기 **전에** 기존 C API 표면 결함 정리(ADR §6.4):
- **에러코드 정본 단일화**: `IRIS_SDK_NOT_INITIALIZED=100`(sdk_api.h:41) ↔ `IRIS_SDK_ERROR_NOT_INITIALIZED=501`(:73) 이중 변환을 단일 정본으로, sdk_api.cpp/sdk_api_v2.cpp 변환 일원화.
- **BeautyFilterConfigV2 기본값 단일소스화**: C++(beauty_filter.h) / Java(`BeautyFilterConfigV2.java:217-240` DEFAULT_) / 문서 3원 분기 → 단일 파생(현재 값 우연 일치하나 드리프트 무검출 구조).
- **internal 9종 헤더 선언화**: `iris_jni.cpp:2238-2348` 수동 extern 9종(sclera_veto, env_map 3종, reflection 2종, blink_up, gate_threshold, detail_reinject, use_measured_luma)을 내부 헤더로 정식화.

### W4-B — types.h 정리 + 주입 승격 (Codex 권고로 B1~B4 분할, 회귀 원인 분리)
한 PR로 묶지 않고 독립 PR 4개로 — 각 단계 동작 불변(게이트=골든 일치), 단 B1의 confidence 수정은 의도된 동작 변경(주입 경로 한정).
- **B1 — confidence/visibility 계약 정정 (선결 critical)**: 주입 경로 visibility 게이팅을 confidence 곱(`eye_render_packet_adapter.cpp:89`)에서 ADR §6.2 의도대로 **EAR/visibility 직접 산출**로 정정(또는 deriveIrisResult가 EAR 파생 confidence를 채움). 데모 TASKS의 confidence=1.0 우회 제거. **이게 안 되면 주입 승격 시 렌즈 미렌더** → B2 이후의 전제.
- **B2 — layout/API 단일화**: DetectorType(types.h:88)/EyeRefinerPolicy(:99) enum 제거, IrisResult C/C++ 2벌 → 단일정의 공유(types.h:42, ADR §6.3), detector 전용 메타 강등(eye_refiner_used:196/iris_quality_*:192-193). `iris_set_landmarks`(③-1) 정식 공급자 승격(generation 검증+§6.1 가드).
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
⑤: grid_mesh resize(468)→478 확장 + geometry 수식 수리. (좌표 canonical 라벨 정정은 ADR §7.3 따라 ④ W4-D로 이동 — Codex 지적 반영.) 2.0: cpu-render 물리 삭제, 코어 TFLite 잔재 최종 삭제. 후속: iOS(Swift 글루)/Web(TS 글루) 전환 — 현재 빈 스텁이라 ④ 코어 제거에 무영향, 구현 시 각자 Tasks 글루로 추적(ADR §3).

## 7. 견적
**20~28 사람·일 (1인 4~6주)**. ADR §13 원견적(18~25인일+cpu-render 2~4일)에 W4-B 분할·confidence 정정·AAR 패키징·재캡처 manifest·16KB 전수검증 부담 반영(Codex 권고로 상향 — 초안 15~22는 낙관적).

## 변경 이력
- 2026-06-15: 스코핑 워크플로 결과로 작성 (게이트0 승인 대기).
- 2026-06-15: Codex 외부 검증 + critical-review 반영 — confidence=0→visibility 0 선결 critical 신설(B1), W4-B를 B1~B4 분할, ts 원자성을 W4-C A/B 선결로, canonical 라벨 정정 ⑤→W4-D 이동(ADR §7.3), recapture manifest·AAR 패키징·OpenCV 16KB 게이트 추가, IrisResultKt 정리 추가, 견적 20~28로 상향. iOS/Web 위험론은 빈 스텁 확인으로 기각(제거 안전 명시).
