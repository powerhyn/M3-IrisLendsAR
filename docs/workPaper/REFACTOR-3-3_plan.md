# REFACTOR-3-3: 추적 교체 A/B 설계 (③-3) — 실행 대기

> 이 문서는 **새 세션이 이 문서만 읽고 ③-3를 실행할 수 있도록** 작성된 설계서다.
> 작성: 2026-06-12 (설계 세션), 상태: ⏳ 실행 대기

## 0. 새 세션 진입 절차 (그대로 수행)

1. 이 문서 + `docs/decisions/0001-landmark-injection-tracking-replacement.md`(ADR, 특히 §5~§7, §10~§12) 정독
2. `git checkout develop && git checkout -b refactor/p33-tracking-ab` (develop HEAD는 b88e070 이후)
3. 워크플로 실행: `Workflow({name: "p33-tracking-ab"})` (스크립트: `.claude/workflows/p33-tracking-ab.js`)
4. 완료 후: 빌드 → 실기기 설치 → §7 A/B 실기기 절차를 사용자에게 안내
5. 게이트/판정(§8) 통과 시 사용자 확정을 받아 ④(자체 추적 제거)는 **별도 승인 후** — 이번 범위 아님

## 1. 목표와 범위

**목표**: 같은 코어에 기존 자체 TFLite 추적 vs MediaPipe Tasks(공식 바인딩)를 꽂아 실기기 A/B 비교가 가능한 상태를 만든다 (계획 문서 ③-3 전반부). 전환 확정·자체 추적 삭제는 A/B 데이터로 사용자가 판정한 후(④).

**범위 내**: demo-app에 Tasks 글루 이식, JNI 주입 표면 추가, A/B 토글·듀얼 비교·메트릭, 단위 테스트 이식
**범위 외**: 자체 TFLite 제거, iris-sdk AAR로의 글루 승격(④), iOS/Web, 정확성 수리(⑤), cpu-render deprecation(④)

## 2. 이식 자산 매핑 (LensSimulator → IrisLensSDK)

원본 루트: `/Volumes/M3-P31/Projects/TedMong/CGG/LensSimulator/sdk/android/lenssdk/`
대상 루트: `android/demo-app/src/main/java/com/irislenssdk/demo/tracking/` (패키지 `com.irislenssdk.demo.tracking`)

| 원본 (src/main/kotlin/com/cgg/lenssdk/internal/) | 대상 | 비고 |
|---|---|---|
| `FaceTracker.kt` | `tracking/FaceTracker.kt` | LIVE_STREAM+GPU delegate+CPU 폴백+rowStride 압축. **값 무변경 이식** (ADR §14 원칙 — 이식 정본은 코드) |
| `internal/math/CoordMapper.kt` | `tracking/math/CoordMapper.kt` | sensorToUpright/uprightToSensor — 함정 #13 해법, 단위 테스트 동반 |
| `internal/LandmarkIndices.kt` | `tracking/LandmarkIndices.kt` | 478 규약 정본 (ADR §7.3 — 명명 충돌 해소 정본) |
| `internal/TrackingSnapshot.kt` | `tracking/TrackingSnapshot.kt` | 불변 스냅샷 — torn read 구조 해법 |
| `internal/math/OneEuroFilter.kt` | (이식 보류) | 코어 C++ OneEuro가 이미 적용 중 — 이중 필터 금지 (§5 주의 3) |
| `internal/math/IrisGeometry.kt` | `tracking/math/IrisGeometry.kt` | 478→홍채 중심/반경 파생 — 듀얼 비교 메트릭 산출에 사용 |
| `internal/CameraController.kt` | (이식 보류) | 데모는 기존 CameraX 경로 유지 — RGBA_8888 스트림 추가는 §5 주의 2 참조 |
| `src/test/.../CoordMapperTest.kt`, `IrisGeometryTest.kt`, `OneEuroFilterTest.kt`(보류분 제외) | `demo-app/src/test/java/com/irislenssdk/demo/tracking/` | 패키지명 치환 외 무수정 |

추가 자산:
- gradle: `demo-app/build.gradle.kts`에 `implementation("com.google.mediapipe:tasks-vision:0.10.35")` — **버전 고정, latest.release 금지** (ADR §5 4사유)
- 모델: `shared/models/face_landmarker.task`(3.6MB) → `demo-app/src/main/assets/models/face_landmarker.task` (이미 assets/models에 .tflite들 있음 — 같은 폴더)
- 패키지 치환: `com.cgg.lenssdk.internal` → `com.irislenssdk.demo.tracking`, 로그 태그 유지

## 3. JNI 주입 표면 추가 (③-1 경계의 첫 외부 소비자)

③-1이 만든 C API(`cpp/include/iris_sdk/sdk_api.h:1133` `iris_set_landmarks`, `:1149` `iris_get_landmark_generation`)는 JNI 미노출. 파생 결과 추출 C API도 없음. 추가할 것:

1. **C API 1개 신설** (`sdk_api.h`/`landmark_injection.cpp`): `IRIS_SDK_EXPORT IrisSdkError iris_get_injected_result(IrisResult* out);` — 내부 `LandmarkInjectionStore::readDerived()` 위임 (reader 무락, seqlock 재시도). 미주입 시 명시 에러. **기존 함수 시그니처 불변 — 추가만**
2. **JNI 3개** (`android/iris-sdk/src/main/cpp/iris_jni.cpp` + `IrisLensSDK.java`):
   - `nativeSetLandmarks(float[] pts478x3, int frameWidth, int frameHeight, long timestampUs) → int(에러코드)` — JNI에서 길이 == 478*3 이중 가드(ADR §6.1)
   - `nativeGetLandmarkGeneration() → long`
   - `nativeGetInjectedResult(IrisResult out) → int` — 기존 `copyResultToJava` 재사용
   - Java 래퍼: `setLandmarks(...)/getLandmarkGeneration()/getInjectedResult(IrisResult)` — 기존 패턴(IrisLensSDK.java) 따름
3. 단위 검증: 골든 baseline JSON의 face_mesh 478점을 `iris_set_landmarks`→`iris_get_injected_result`로 왕복시켜 detector 파생값과 ε 비교 — **데스크톱 C++ 테스트로** (`cpp/tests/test_landmark_injection.cpp`에 케이스 추가; ③-1에서 어댑터 단위는 검증됐고 이번엔 C API 표면 경유 E2E)

## 4. 데모 A/B 인프라

### 4.1 공급자 토글 (실사용 경로 비교 — T2 육안/지연/안정성)

- `GpuRenderActivity`에 추적 공급자 상태 `TrackerMode { LEGACY, TASKS }` + UI 토글(기존 스위치 패턴 따름)
- **LEGACY**(현행 그대로): `IrisLensSDK.detectWithRotation(nv21...)` → stabilize → `updateDetectionSlot(irisResult)` → 렌더
- **TASKS**: FaceTracker(LIVE_STREAM, RGBA_8888 — §5 주의 2) → 478점 TrackingSnapshot → Kotlin에서 `IrisResult` 채움(아래 변환 계약) → 동일하게 `updateDetectionSlot` → 렌더. **동일 DetectionSlot 채널 공유로 렌더 경로 완전 동일** — 비교 변인은 추적기뿐
- TASKS 결과 변환 계약 (Kotlin, `tracking/TasksToIrisResult.kt` 신설):
  - 좌표: Tasks 출력은 원본(미회전) 정규화 좌표 (함정 #13) → `CoordMapper.sensorToUpright(rotation)` 적용 → upright 정규화 (ADR §7.1, 미러 적용 금지 §7.4)
  - 홍채: center 468/473, boundary 469-472/474-477 (§7.0 순서), radius = IrisGeometry 픽셀 환산 (frame dims 사용 — 함정 #5)
  - eyelid/visibility: EAR 계산(IrisGeometry) — 코어 어댑터와 동일 수식
  - confidence: Tasks 미제공 → 1.0 고정 + 주석 (ADR §6.3 — 검출 실패=주입 부재로 일원화)
  - timestamp: SystemClock 단조 → timestampMs
  - 참고: DetectionSlot 마샬링은 boundary[1..4]/visibility를 운반하지 않음(감사 finding) — 렌더는 [0]+radius만 소비하므로 A/B에 무해. 정식 운반은 ④ 주입 채널 공식화에서
- One-Euro/stabilize: **TASKS 경로도 LEGACY와 동일하게 코어 stabilize 적용** (`IrisLensSDK.stabilize`) — 비교 변인 최소화. FaceTracker 내장 필터는 비활성(파라미터 0 또는 우회 — 이중 필터 금지)

### 4.2 듀얼 비교 모드 (T1 정밀 판정 — 동일 프레임)

- 토글과 별개의 "A/B 측정 모드": 같은 NV21 프레임을 ① 자체 `detectWithRotation`(stabilize 전 raw) ② Tasks **IMAGE 모드**(`FaceLandmarker.detect` 동기) 로 양쪽 실행
- 메트릭 (N프레임 누적, 기본 300):
  - 홍채 중심 픽셀 오차: |c_legacy − c_tasks| (upright 픽셀, 좌/우 각각) — **검출 홍채 반경 대비 비율**로 정규화 (T1 기준: 계통적 오프셋 ≥ 반경 100% = 계약 위반)
  - 축 스왑/미러 반전/정규화 오류 전형 패턴 검출: dx≈frame_w−2x 류 휴리스틱 + 오프셋의 부호 일관성
  - radius 비율, 지연 ms (양쪽), 검출 성공률
- 출력: logcat 구조화 라인(`AB_METRIC` 태그, CSV형) + 종료 시 요약 + HUD 1줄
- **rot0(실기기 자연 경로) 중심 판정** — 회전 입력은 기존 추적기 기준 오염(rot0≡rot180 퇴화, ADR §10 주의)이므로 T1에 사용 금지. 회전 변형은 Tasks 쪽 §7.1 계약 검증 용도로만(좌표가 upright로 정합하는지 단독 확인)

## 5. 주의 (실측·감사 근거 — 위반 시 A/B 무효)

1. **함정 #3 (스레드 친화성)**: FaceTracker는 GPU delegate 생성 스레드에서만 detect — 원본 구현이 이미 처리, 이식 시 스레드 구조 보존. 에뮬레이터 CPU 강제 분기도 보존
2. **입력 경로**: Tasks는 YUV 직접 입력 불가(함정 #2) — LIVE_STREAM용 RGBA_8888 스트림: CameraX ImageAnalysis를 RGBA_8888로 **별도 use case 추가** 또는 기존 NV21→RGBA 변환(비교 공정성: LEGACY가 NV21이므로 변환 비용을 지연 메트릭에서 분리 표기). 듀얼 비교(IMAGE 모드)는 같은 NV21→Bitmap 변환 1회로 양쪽 공정
3. **이중 필터 금지**: 코어 stabilize가 켜진 상태에서 FaceTracker 내장 OneEuro까지 돌리면 비교 오염
4. **버전**: tasks-vision 0.10.35 외 금지 (latest.release=레거시 해석, <0.10.26=16KB 미정렬 — ADR §5/§9)
5. **자체 추적 경로 무변경**: LEGACY 모드 코드는 한 줄도 바꾸지 않는다 — A/B 기준선 보존. 데모 정화 W 산출물(스냅샷 복사·HUD)도 보존
6. **코어(cpp) 변경은 §3의 C API 1개 추가뿐** — 골든 영향 없음(주입 경로는 골든 미경유), 단 골든 게이트는 동일 실행

## 6. 게이트 (머지 조건 — 워크플로가 검증)

1. `cd cpp/cmake-build-debug && cmake .. -DIRIS_SDK_FETCH_TFLITE=OFF && cmake --build . --parallel` — 신규 경고 0 (⚠️ 새 빌드 디렉토리 금지)
2. ctest 직렬 — pre-existing 외 회귀 0 (pre-existing: TFLite 링크 NOT_BUILT 2, GPUBeautyBackendTest.FailsWithNullContext, FreqSep 2 / 병렬 시 MediaPipeDetectorIntegration 3은 기존 플레이크)
3. 골든: `bash scripts/golden_capture_all.sh /tmp/p33 && python3 scripts/golden_compare.py --baseline cpp/tests/golden/baseline --candidate /tmp/p33` → exit 0 (베이스라인 덮어쓰기 금지)
4. `cd android && ./gradlew :demo-app:assembleDebug` 성공 + `:demo-app:testDebugUnitTest`(이식 테스트 포함) 통과
5. 16KB: 이번엔 tasks-vision AAR 추가뿐(자체 .so 무변경) — `find android -path "*merged_native_libs*arm64*" -name "*.so" | xargs objdump --private-headers | grep align` 기록만 (자체 .so 재정렬은 ④)
6. LEGACY 경로 diff 무변경 검증: `git diff` 에서 기존 검출·렌더 코드 변경이 §3 JNI/Java 추가와 데모 A/B 신설 외에 없는지

## 7. 실기기 A/B 절차 (사용자 안내용)

1. 토글 LEGACY ↔ TASKS 전환하며 육안: 렌즈 정합·떨림·눈꺼풀 마스킹·지연 체감 (T2 — 밝은 환경 우선, 저조도는 우선순위 낮음)
2. A/B 측정 모드 ON → 정면 응시 10초 + 좌우 회전 + 근접/원거리 → `adb logcat -s AB_METRIC` 수집 → 요약의 중심 오차(반경 비율)·패턴 플래그 확인
3. T1 판정: 계통적 오프셋 ≥ 반경 100% 또는 축 스왑/미러 패턴 → 어댑터(변환 계약) 수정 2회 내 해소 안 되면 후퇴 트리거 (ADR §10)
4. 판정 주체: 사용자. 통과 시 "전환 확정" 선언 → ④ 착수 가능 (별도 승인)

## 8. 워크플로 구성 (.claude/workflows/p33-tracking-ab.js)

Build 병렬 2 (`tracker-port`: §2 이식+gradle+모델 / `jni-bridge`: §3) → Integrate 1 (`ab-demo`: §4) → Verify 병렬 3 (게이트 §6 독립 재실행 / 좌표 규약 적대 검토: §4.1 변환 계약 vs ADR §7·함정 #13/#12/#10/#5, LensSimulator 원본 코드 대조 / 범위·불변 검토: §5-5·§6-6) → Revise 1 (critical/important)
— 에이전트는 모두 이 문서와 ADR을 필독. 마커: '당신은 ③-3 …작업자다/적대적 검증자다' (수확 호환). 커밋은 메인 루프가.

## 9. 이월·후속 (이번에 하지 않음)

- ④: 글루 iris-sdk 승격, 자체 TFLite 제거(removalScope는 감사 §6.4), DetectionSlot 정식 재설계(boundary/visibility 운반), IrisResult 단일 정의, 16KB 자체 .so 재정렬+검증 게이트, cpu-render deprecation(§8.2)
- ⑤: geometry 수식 수리(grid_mesh 478 포함)
- 뷰티 파라미터 0 pass-through 단서(데모 기본값) — P8-W1 잔여와 함께

## 변경 이력

- 2026-06-12: 설계 작성 (실행은 새 세션 — §0 절차)
