# ③-3 트래킹 지연 — 후속 개선 항목 (이월)

> 작성: 2026-06-15. 트래킹 지연 해결(커밋 f86b66f, frame-sync) 후 **의도적으로 이월**한 개선들.
> 현 상태로도 실기기에서 "렌즈가 눈을 따라옴" 검증 완료 — 아래는 **효과를 더 끌어올리거나
> 엄밀성을 높이는 선택적 후속**이다. 우선순위 낮음(구조 전환 ④ 이후 또는 병행).

근거 문서: `docs/lenssim-handoff/tracking-latency-handoff-from-lenssimulator.md`,
LensSim ADR-0002/0005, 적대 리뷰(워크플로 wf_fa34ea2c-9e3).

---

## 1. 코어 stabilize near-raw 재튜닝 (핸드오프 §5) — 효과 ↑

- **무엇**: 코어 `TemporalStabilizer`의 One-Euro 파라미터 `iris_min_cutoff=4.0 / iris_beta=15.0`
  (`cpp/include/iris_sdk/temporal_stabilizer.h:18-19`)을 near-raw로 약화.
- **왜**: frame-sync 체제에서는 화면 = 랜드마크 프레임이라 **필터 위상 지연이 곧 화면 어긋남**이 된다
  (핸드오프 §5). 현재 4.0/15.0은 frame-sync **이전** 체제의 강한 필터다. LensSim 동기-후 검증값은
  `min_cutoff=3.0 / beta=0.3` (단, **값 정본은 코드 — 데모 좌표계/단위에 맞춰 재검증** 필요. ADR §7.5
  단위 계약: 정규화 공간 vs 픽셀 공간 혼동 금지).
- **영향**: 양 모드(LEGACY/TASKS) 공통 stabilize라 A/B 비대칭 아님. 정지 시 지터 ↔ 추종성 트레이드오프.
- **주의**: 코어(`cpp/`) 변경 → 골든 게이트 동일 실행 필요(주입/stabilize는 골든 미경유지만 회귀 확인).
  `systems-programming:cpp-pro` 동반. 실기기 정지 지터 재검증.
- **위치**: `cpp/include/iris_sdk/temporal_stabilizer.h:18-19`, 호출부 `GpuRenderActivity.kt:1074(LEGACY)/1247(TASKS)`.

## 2. TASKS 분석 해상도 인하 — 미러 지연 격차 ↓

- **무엇**: TASKS 경로 ImageAnalysis 해상도를 낮춰 detect 사이클(~27ms) 단축.
- **왜**: frame-sync는 정합을 맞추지만 TASKS는 사이클이 길어(LEGACY ~11ms 대비) 더 오래된 프레임을
  그리므로 **거울 지연이 LEGACY보다 크다**. 사이클을 줄이면 격차가 좁혀진다(원본 LensSim은 분석
  640x480 / 프리뷰 고해상도 분리).
- **영향**: 저해상도 검출의 정확도·휘도 측정 영향 점검 필요. ResolutionSelector 분리
  (`GpuRenderActivity.kt:939-942` 부근, 현재 Preview/Analysis 공통).
- **주의**: 양 모드 공통 적용해야 A/B 변인 격리 유지. 검출 정확도 회귀 실기기 확인.

## 3. ts ↔ DetectionSlot 번들링 — 1프레임 스큐 제거 (④ 범위) — ✅ 해소 (W4-B3, 2026-06-16)

> **해소됨**: `DetectionSlot`(iris_jni.cpp 내부 구조체)에 `frame_ts_ns` 추가 + `nativeUpdateDetectionSlot(result, frameTsNs)`로 ts 동반 주입 + 단일 스냅샷 reader `nativeGetActiveDetectionSlot`로 GL이 배경 ts·렌즈 좌표·게이트(detected)를 한 슬롯에서 원자 취득. volatile `latestLandmarkFrameTsNs` 사이드채널 폐기로 1프레임 스큐 원천 제거. 공개 ABI/골든 무영향. (REFACTOR-4_plan.md W4-B3 참조. 실기기 fsync 무회귀 통과 — 2026-06-17 SM-S916N, LEGACY/TASKS 양 모드.) 이하 원 분석 보존:


- **무엇**: frame-sync 배경 슬롯 키(`latestLandmarkFrameTsNs`, volatile 사이드 채널)와 렌즈 좌표
  (네이티브 `DetectionSlot`)를 **단일 채널로 원자 전달**.
- **왜**: 현재 두 채널 happens-before 미보장 → GL `onDrawFrame`이 두 쓰기 사이에 읽으면 배경=프레임N·
  렌즈=N-1의 **간헐 1프레임 스큐**(비크래시·자가수정·양 모드 대칭, 빠른 움직임 시 미세 떨림 가능).
  코드 주석: `CameraGLRenderer.kt` `setLandmarkFrameTimestamp` 한계 노트.
- **해결**: 센서 ns를 `IrisResult` 필드 또는 `DetectionSlot` 동반 메타로 편입 → ④ 주입 채널 공식화에서
  결정(SDK surface 변경 — C++/JNI/Java/문서 일관성 점검 필요).
- **위치**: `GpuRenderActivity.kt:1080·1196` (ts 전파), `CameraGLRenderer.kt`(ts read vs getDetectionSlotPtr).

## 4. 메모리 +~25MB 실측 (측정 기록)

- **무엇**: frame-sync RGBA 링버퍼 4장 상시 할당. 고사양 1080x1920 RGBA8 ≈ 8.3MB/장 → **+~25MB**
  (중간 720x1280 +~11MB, 저사양 480x640 +~3.5MB). 추정(해상도×바이트), **dumpsys meminfo 미실측**.
- **할 일**: 실기기 `adb shell dumpsys meminfo com.irislenssdk.demo`로 frame-sync ON/OFF 증분 1회 실측.
  빠듯하면 저사양 `ringSize` 2~3 인하 또는 토글 ON 시 lazy 확장 검토.
- **위치**: `CameraGLRenderer.kt` `ringSize=4`(하드코딩), `recreateIntermediateBuffers`.
- 참고: 누수 없음(`deleteRingBuffers` 모든 경로 선행), OFF 회귀 없음(OFF=최신 슬롯, 비트 동일).

## 5. 클럭 도메인 다기기 검증 (엄밀성)

- **무엇**: frame-sync 매칭은 `imageInfo.timestamp`(분석) ↔ `SurfaceTexture.timestamp`(화면)가 같은
  센서 클럭이어야 성립. S23+ 단일 기기 게이트는 통과(둘 다 ~1.382e15 ns 동일).
- **할 일**: 다른 벤더/기기에서 `AB_METRIC`/`frame-sync: minΔ` 로그(`CameraGLRenderer.kt`)가 정상 시
  0~수십 ms로 수렴하는지 확인. **수십~수백 ms로 일정하게** 어긋나면(1s 미만이라 강등 가드에 안 걸림)
  부분 불일치 → 별도 기록. 킬스위치로 안전.

## 6. (minor) frame-sync ON 시 진단 오버레이 시점 차이

- frame-sync ON이면 GL 렌즈는 과거(매칭) 프레임에, OverlayView 진단 점(마젠타/녹색)은 최신 UI
  타임라인에 그려져 두 레이어가 +2~3프레임 어긋나 보일 수 있다. **정합 판정은 렌즈-눈 기준**으로 하고
  진단 오버레이는 "필터 이전부터 늦는가" 확인 용도로만. A/B 절차 안내에 1줄 명시 권장.

---

## 변경 이력
- 2026-06-15: 트래킹 지연 해결(f86b66f) 후 이월 항목 정리.
