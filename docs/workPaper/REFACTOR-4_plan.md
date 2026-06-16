# REFACTOR-4: 추적 레이어 주입형 전환 (④) — 실행 계획

> 새 세션이 이 문서 + ADR-0001만 읽고 ④를 실행할 수 있도록 작성. 작성: 2026-06-15.
> 상태: ⏳ **게이트0(전환 확정 선언 + ④ 승인) 대기** — 승인 후 W4-A 착수.
> 근거: ADR-0001(승인 2026-06-11) §3/§6/§8/§9/§12, REFACTOR-3-3 §9 이월, 스코핑 워크플로(wf_207a105f-dcc 수확 + wf_4f1249e6-81a 연속).

## 0. 핵심 사실 (스코핑 실측 2026-06-15)

- ④ **추적 코어 제거는 100% 미착수** — mediapipe_detector.cpp(150KB)·inference_thread·iris_detector(죽은 nullptr 팩토리)·frame_processor 한 줄도 안 빠짐.
- ③-1~③-3가 만든 **주입 경계(C API `iris_set_landmarks`/`iris_get_landmark_generation`/`iris_get_injected_result` + JNI 3종 + 데모 글루 12파일 + 골든 베이스라인)** 는 ④가 재활용할 토대 — 완성됨.
- §6.4 선행 5종 중 **[4]468 하드코딩·[5]offsetof 가드는 ③-2가 처리**(grid_mesh.cpp:183/186, beauty_roi_manager.cpp:99/105, sdk_api_v2.cpp:44-79). **[1]에러코드·[2]기본값·[3]internal 헤더는 미처리** → W4-A.
- ADR §12.3 게이트3(데모 검증 통로 정화: torn 스냅샷·KT 무음 폴백·OverlayView 정책)은 별도 트랙에서 완료(7a02fa0)로 추정 — 착수 전 1회 재확인 권장.

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

### W4-B — types.h 정리 + 주입 승격 (동작 불변, 게이트=골든 일치)
- DetectorType(types.h:88)/EyeRefinerPolicy(:99) enum 제거.
- IrisResult 단일화: C/C++ 2벌(offsetof 가드됨) → 단일정의 공유, Java/Kotlin 2벌 → JNI 필드매핑 생성/검증(types.h:42, ADR §6.3).
- detector 전용 메타 필드 강등/정리(eye_refiner_used:196 / iris_quality_*:192-193 / eyelid_ratio_*:194-195). confidence는 visibility/EAR 게이팅으로 일원화.
- `iris_set_landmarks`(③-1)를 정식 추적 공급자로 승격(generation 검증 + §6.1 가드).
- **DetectionSlot 정식 재설계**: 데모를 옛 JNI 슬롯(updateDetectionSlot/getDetectionSlotPtr)에서 ③-1 주입 경로로 단일채널화 + `copyResultFromJava` boundary[1..4]/visibility 정식 운반 + **ts-슬롯 원자 번들링**(frame-sync 1프레임 스큐 해결, 후속문서 §3).
- **avg_iris_luma 처리(R3)**: mediapipe_detector calculateIrisLuma 소실 대비 GPU self-measure 이전(gpu_lens_renderer.cpp:804/1128 'W6 이관' 주석) 또는 글루 측정-주입필드 정식 승격 — P7-W2 default ON 회귀 방지, 경계 도입과 동시 처리.

### W4-C — 글루 iris-sdk AAR 승격 (게이트=A/B + 재캡처)
- demo-app `tracking/` 12파일 → iris-sdk AAR로 승격(현재 iris-sdk에 FaceTracker/CoordMapper/TasksToIrisResult 없음).
- tasks-vision 0.10.35 → iris-sdk `build.gradle.kts`(현재 demo만), face_landmarker.task → SDK 번들(현 데모 assets 중복 해소).
- 데모는 SDK 추적 API 소비로 전환, **TASKS 단일 경로**(LEGACY 토글은 A/B 검증 종료 후 제거 가능).

### W4-D — removalScope 자체 추적 제거 (게이트=A/B + 명시적 재기준선)
**W4-C로 SDK가 추적을 책임진 후에만 안전**(현 LEGACY 기본 — 먼저 지우면 데모·골든 깨짐):
- 코어: mediapipe_detector(150KB)+.h, inference_thread, iris_detector, frame_processor(검출/NV21·NV12/비동기 submitFrame).
- sdk_api detect 계열, sdk_manager createDetector, JNI nativeDetect*/confidence 계열.
- TFLite CMake 블록(cpp/CMakeLists.txt:224-537) + cpp/third_party/tflite/ + .tflite/.task 에셋.
- 추적 의존 테스트 12파일(test_mediapipe_detector*/test_iris_detector/test_frame_processor/test_integration 등) + examples 2종.
- **골든**: 추적 교체는 이종 모델이라 ε 일치 원리상 불가 → A/B(§10 T1) + **명시적 베이스라인 재캡처**(무언의 재기준선 금지).

### W4-E — cpu-render deprecation + 16KB 재정렬 (게이트=objdump/zipalign)
- **cpu-render(옵션 B, §8.2)**: iris_sdk_render_lens(sdk_api.h:428)·iris_sdk_process(:376) 등 공개 CPU 픽셀 API에 `IRIS_SDK_DEPRECATED` 마킹 + 구현 동결 + '2.0 제거' 고지. **파일 삭제 금지(2.0)**, OpenCV 잔존은 1.x 정상.
- **16KB(§9 대응1)**: AGP 8.5.0→8.5.1+(android/build.gradle.kts:10-11), ndkVersion r27+ 핀, `-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON` 또는 `-Wl,-z,max-page-size=16384`. 검증 게이트 신설: objdump --private-headers align 2**14 + zipalign -c -P 16. TFLite prebuilt 2종은 W4-D 제거로 자동 해소.

## 4. 머지 게이트 (PR별, ADR §12)
- **동작 불변 단계(W4-A/B)**: 골든 ε 일치. **추적 교체 단계(W4-C/D)**: A/B(T1) + 명시적 베이스라인 재캡처.
- ctest 회귀 0(pre-existing 외). 16KB(Android 산출물 변경 시): objdump align 2**14 + zipalign -c -P 16.
- 실기기 A/B 토글 육안(사용자 판정).

## 5. 리스크
- R1 좌표 계약(전환 시 upright/미러/L-R 정합) / R2 detector 메타 필드 소실 / R3 avg_iris_luma 소실(P7-W2 회귀) / R4 torn-read / R5 ABI 변경.
- 게이트0 미기록(§1) — 착수 전 해소. 코어 삭제 시점(W4-D)이 글루 승격 완료에 종속.

## 6. 이월 (④ 범위 밖)
⑤: grid_mesh 478 확장 + geometry 수식 수리, 좌표 canonical 라벨 정정(types.h:150). 2.0: cpu-render 물리 삭제, 코어 TFLite 잔재 최종 삭제. 후속: iOS(Swift 글루)/Web(TS 글루) 전환.

## 7. 견적
15~22 사람·일 (1인 3~4주). ADR §13 원견적(17~24인일+α)에서 ③-1~③-3 선행 완료분 반영해 소폭 감소.

## 변경 이력
- 2026-06-15: 스코핑 워크플로 결과로 작성 (게이트0 승인 대기).
