# ADR-0001: 추적 레이어 주입형 전환 (랜드마크 주입 + MediaPipe Tasks 교체)

- 상태: **승인** (2026-06-11 — LensSimulator 세션 교차 검토 수용 8/부분 수용 2/거부 0, 사용자 확인 완료. 결정 항목 ① 옵션 B 확정)
- 근거 문서:
  - `docs/lenssim-handoff/audit-report.md` — 1단계 감사 보고서 (확정 findings 133건: blocker 2 / major 63 / minor 68, Codex gpt-5.5 xhigh 교차 검토 반영)
  - `docs/lenssim-handoff/refactoring-audit-and-tracking-migration-plan.md` — 단계 계획 (2-1 ADR 정의, 목표 아키텍처, C API 경계안)
  - `/Volumes/M3-P31/Projects/TedMong/CGG/LensSimulator/docs/decisions/0002-tracking-rendering-stack.md` — MediaPipe Tasks 버전 고정 사유 + 478점 랜드마크 규약 (본 ADR이 승계)
  - `/Volumes/M3-P31/Projects/TedMong/CGG/LensSimulator/docs/tech-validation.md` — MediaPipe 결정적 함정 13건
- 결정자: 1단계 감사(멀티에이전트 적대적 리뷰 + Codex 교차 검토) 결과 기반, 사용자 최종 승인 대상

## 1. 맥락

IrisLensSDK 코어는 MediaPipe 그래프를 수제 TFLite 파이프라인으로 재구현한 자체 추적기(mediapipe_detector.cpp 3,425줄+)를 내장한다. 1단계 감사가 이 구조의 현황을 확정했다:

- **추적 모듈은 수리 대상이 아니라 교체 대상이다** (감사 §7 양 패널 일치 판정 "재작성", §9-1). BlazeFace 입력 정규화 `[0,1]` vs canonical `[-1,1]`, 박스 디코딩 좌표 순서 반전, Eye Refiner 활성화 시 좌표 붕괴, InferenceThread 초기화 실패 시 `std::terminate`(blocker) 등 결함이 집중되어 있고, Codex 교차 검토가 blocker 2건과 표본 major 6건을 코드 직접 대조로 전부 사실 확인했다 (§11).
- **결합은 '코드는 얕고 데이터 계약은 깊다'** (§6.1). TFLite include는 mediapipe_detector.cpp 단일 파일 격리(가드 26곳), CMake 링크는 PRIVATE — 빌드 분리는 기계적이다. 반면 IrisResult(face_mesh[478] + detector 메타)가 C++/C/JNI/Java 4중 표현으로 전 레이어를 왕복한다.
- **목표 아키텍처의 seam이 이미 코드에 존재한다** (§6.3, §9-3). sdk_api_v2의 GPU 함수군(`iris_sdk_render_lens_texture` 등)은 이미 '텍스처 ID + IrisResult 포인터' 주입형이고, JNI DetectionSlot은 외부 검출 결과 주입의 실증 프로토타입으로 동작 중이다. seam 적대 검증 결과 구조 유효(refuted=false).
- LensSimulator 기술검증에서 "추적은 플랫폼 공식 바인딩(MediaPipe Tasks) 주입 + 코어는 이펙트/렌더 전담" 구조의 이점이 실기기로 확인되었고, 검증된 글루 코드(FaceTracker.kt, CoordMapper.kt, IrisGeometry.kt)가 재활용 가능하다.

## 2. 결정 요약

1. **코어(iriscore)는 478점 랜드마크 + 제어 파라미터 주입을 받아 지오메트리·이펙트·렌더만 전담한다.** 추적(TFLite/MediaPipe 추론)은 코어 밖으로 나간다. 프레임 픽셀은 C API 경계를 넘지 않는다 (텍스처 ID 기반).
2. **추적은 플랫폼 공식 바인딩 MediaPipe Tasks FaceLandmarker — 버전 0.10.35 고정.** 버전 고정 사유는 LensSimulator ADR-0002를 그대로 승계한다 (§5).
3. **경계 도입은 신규 설계가 아니라 기존 경계의 공식 승격이다** (감사 §6.6 요약, §9-3). sdk_api_v2 주입형 함수군 + JNI DetectionSlot 패턴을 C API `iris_set_landmarks`로 승격하고, detector가 부수 생산하던 파생 데이터는 코어 측 어댑터로 이식한다 (§6).
4. **좌표·시맨틱 계약 4종(회전 후 좌표 공간 / z 스케일 / left-right 명명 / 미러 규약)을 본 ADR로 명문 확정한다** (§7). 감사가 "전부 모호 판정"(§8 패널 A)한 항목이며, LensSimulator 478점 규약과 정합시킨다.
5. **gpu-render(~5.8k줄)는 재작성하지 않고 리팩토링 보존한다** (감사 §7 양 패널 일치, §9-2). P5~P8 실기기 육안 검증이 누적된 유일한 자산이다.
6. 결정 항목 ① cpu-render 처분과 ② 16KB 정렬 대응은 §8, §9에서 각각 다룬다. ②는 실측으로 확정 결함 판정이 났고 대응을 본 ADR에 포함한다. ①은 권고안을 제시하되 최종 선택은 사용자 몫이다.

## 3. 목표 아키텍처

```
┌────────────────────────────────────────────────────┐
│ iriscore (C++ 단일 코어)                            │
│  입력: 카메라 텍스처 + 랜드마크 478점 + 제어 파라미터 │
│  처리: 지오메트리(One-Euro·기저벡터·워프) + 이펙트    │
│  ※ 추적(TFLite/MediaPipe)은 코어 밖                 │
└──────┬──────────────┬───────────────┬──────────────┘
       │ NDK .so      │ static lib    │ wasm
  Android AAR     iOS XCFramework   Web npm
  Kotlin 글루       Swift 글루        TS 글루
  + MediaPipe       + MediaPipe      + @mediapipe/
    Tasks (POM)       Tasks(prelink)   tasks-vision
```

(계획 문서 2-1의 다이어그램 승계. Web 포함 전 플랫폼에 공식 배포 채널 존재 — 0.10.35 동시 배포 확인됨)

seam 기준 분류 (감사 §6.3, §6.4 removalScope 승계):

| 구분 | 대상 |
|---|---|
| **코어 밖으로 (제거/교체)** | mediapipe_detector(3,425줄+), inference_thread, iris_detector(전 분기 nullptr 반환 죽은 Strategy 팩토리), frame_processor(검출/NV21·NV12 변환/비동기 submitFrame — GPU-only 전환 시 파일 전체), sdk_api detect 계열 함수군, sdk_manager createDetector, JNI nativeDetect*/confidence 설정 계열, TFLite CMake 블록(옵션·prebuilt imported target·GPU delegate EGL 링크) + `cpp/third_party/tflite/`, .tflite/.task 모델 에셋, 추적 의존 테스트 12파일, examples 중 FrameProcessor 기반 2종 |
| **코어에 남는 것** | temporal_stabilizer, eye_render_packet_adapter, gpu_lens_renderer, gpu_beauty_backend, warp, beauty ROI, lens_sku_metadata, shader, one_euro_filter, skin_mask_geometry |
| **누락 sweep 추가분 (§6.2 보충 9건)** | sdk_api.cpp 라이프사이클 진입점(g_processor), CMake TFLite 블록 실범위(:395-537 포함), scripts/(download_models.sh 등), Kotlin SDK 표면(IrisLensSDKKt/ProcessResultKt/FrameFormat), IrisResultKt 4중 정의, sdk_manager::createFrameProcessor, 하드 결합 테스트 2개 추가, .task 번들 3개, 478 인덱스 소프트 결합 파일 추가분 |

전환 근거(계획 문서 승계): 모델/delegate/16KB 정렬 관리를 구글에 위임, 플랫폼 공식 바인딩의 유지보수 승계, 전 플랫폼 공식 배포 채널 존재.

**플랫폼 순차 전환** (계획 ③-3 승계): **Android 먼저** — 실기기 검증 가능 + LensSimulator 글루 재활용 (CameraX RGBA_8888 → ByteBufferImageBuilder → FaceLandmarker LIVE_STREAM, GPU delegate + CPU 폴백). 같은 코어에 기존 추적 vs 공식 바인딩을 꽂아 A/B 비교(정확도·지연·안정성) 후 데이터로 전환 확정. iOS → Web 순차 진행, 완료 후 자체 TFLite 파이프라인 제거.

### 검출 폴백 (Eye-Only) — 플랫폼 글루 책임 (2026-06-16 결정)

MediaPipe Tasks가 얼굴 검출에 실패하는 경우(눈만 클로즈업, 극단 각도 — CLAUDE.md "MediaPipe 한계점 Phase 1", HybridDetector 원계획)의 **Eye-Only 폴백은 플랫폼 글루(코어 밖)가 책임진다.** 글루가 "Tasks 검출 실패 → Eye-Only `.tflite` 모델 추론 → 478 랜드마크(부분 mesh 포함) 주입"을 수행하고, 코어는 변함없이 "주입받아 렌더"만 한다.

- **근거**: ④의 "추적은 글루, 코어는 렌더" 전제(§2-1, §3)와 일관 — Eye-Only도 '눈 찾기 = 추적'이므로 글루 책임이 맞다. **TFLite 런타임은 글루 레이어(MediaPipe Tasks가 이미 사용)에 위임**되어 코어 `.so`는 TFLite-free(§9 16KB 위임 + §8 OpenCV 제거 이득)를 유지한다. 코어가 폴백을 보유하면 ④가 제거하려는 TFLite 인프라(mediapipe_detector·TFLite CMake·prebuilt .so)가 코어에 잔존해 ④ 목적이 훼손되므로 **기각**(검토 2026-06-16: 글루 vs 코어 폴백 — 글루 채택).
- **현 코어 DetectorType/EyeOnlyDetector/HybridDetector는 코어 폴백 미보유 확정에 따라 W4-D에서 완전 삭제**한다(iris_detector.cpp `createDetector`가 이미 전 분기 nullptr 반환 = 구현 0인 빈 껍데기 — 폴백 능력 0이므로 보존 가치 없음). 미래에도 코어는 추적/검출을 보유하지 않는다.
- **Eye-Only 폴백 구현은 ④ 완료 후 별도 글루 트랙**(Phase 9+ 후보). Eye-Only 모델 출력(눈 주변 점)을 478 주입 계약(§6.1 num_points 478 고정)에 맞추는 변환(부분 mesh 주입 허용 여부 등)은 그 트랙에서 설계한다 — 글루/코어 위치와 무관한 별도 과제.

## 4. 결정의 전제 — 감사가 확정한 사실

| 사실 | 출처 |
|---|---|
| TFLite 격리: include 2파일(mediapipe_detector.{cpp,h}), CMake PRIVATE — 제거는 기계적 | §6.2 pt9, §6.3 검증 1 |
| sdk_api_v2 렌더 경로는 `iris_sdk_init` 없이 `iris_sdk_init_gpu_lens`만으로 동작 — 검출과 이미 독립 | §6.3 검증 1 |
| seam 아래(렌더/뷰티/워프/안정화)에서 detector 헤더 include 0건 | §6.3 검증 1 |
| 478점 인덱스 규약은 소프트 결합 — MediaPipe Tasks도 동일 규약이라 주입 후 그대로 유효 | §6.2 pt7 |
| CPU lens_renderer 데모 소비처 0 실측 ("항상 ~45° 잘못 회전" 결함이 미발견 출하된 것이 그 증거) | §7 cpu-render 패널 A |
| 골든 베이스라인은 "구축"이 아니라 "신축" — 렌더 픽셀 검증 0건, 좌표 회귀 0건 | §9-5 |
| 데모 검증 통로가 torn read·독립 게이트·무음 폴백으로 오염 — 정화가 모든 단계의 선행 조건 | §9-5 |

## 5. 추적: MediaPipe Tasks FaceLandmarker — 버전 0.10.35 고정

| | Android | iOS | Web |
|---|---|---|---|
| 의존성 | `com.google.mediapipe:tasks-vision:0.10.35` | pod `MediaPipeTasksVision '0.10.35'` | `@mediapipe/tasks-vision` |
| 모드 | `RunningMode.LIVE_STREAM` + ResultListener | `.liveStream` + delegate(weak!) | VIDEO |
| 입력 | CameraX RGBA_8888 → `ByteBufferImageBuilder` 직접 (Bitmap 경유 금지) | `kCVPixelFormatType_32BGRA` → `MPImage(sampleBuffer:orientation:)` | — |
| 추론 | GPU delegate(생성 스레드에서만 detect — 스레드 친화성), 실패 시 CPU 폴백 | CPU(XNNPACK) — 공식 지원 경로 | — |
| 회전 | `ImageProcessingOptions.rotationDegrees` | `orientation` 파라미터. **미러링 미지원** — 미러는 렌더 단계 | — |

**버전 고정 사유 (LensSimulator ADR-0002 승계, 절대 변경 금지 항목)**
- `latest.release`는 레거시 `0.20230731`로 해석됨 (maven-metadata `<latest>` 오염, mediapipe#5588)
- 16KB page size 정렬은 0.10.26+ — Play 정책상 2026-06 현재 그 미만은 업데이트 제출 불가
- iOS 0.10.33은 패키징 결함(framework 내부 Info.plist 누락)으로 빌드 깨짐 — 사용 금지
- 양 플랫폼 0.10.35는 Apple Silicon 시뮬레이터 슬라이스 포함 (실측 검증됨)

Android 글루는 LensSimulator 검증본(`sdk/android/.../internal/FaceTracker.kt` — rowStride 압축·GPU delegate 스레드 친화성·CPU 폴백 처리 완료, `CoordMapper.kt` — 단위 테스트 동반)을 이식한다. skin_mask_geometry.cpp(P8-W1)가 '값 무변경 이식' 패턴의 성공 선례다 (감사 §8 양 패널 공통 인정).

## 6. C API 경계 (확정안)

계획 문서의 4함수 안을 기본으로 채택하고, 감사 결과로 보강한다.

```c
/* 주입 — 추적(코어 밖) → 코어. 프레임 픽셀은 경계를 넘지 않는다. */
IrisSdkStatus iris_set_landmarks(const float* pts,        /* num_points×3 (x,y,z) 정규화, §7 계약 */
                                 int32_t num_points,      /* 점 수 — 478 외 거부 (§6.1 입력 유효성) */
                                 int32_t frame_width,     /* upright 프레임 치수 (px) — 파생 어댑터의 */
                                 int32_t frame_height,    /*  픽셀 환산(§7.0) 기준. 렌더 타깃 치수와 별개 */
                                 int64_t timestamp_us,    /* 단조 증가, One-Euro dt 산출용 */
                                 uint32_t* out_generation);/* 주입 세대 번호 반환 */
IrisSdkStatus iris_set_lens(const IrisLensParams* params);    /* SKU, 강도, blend mode 등 */
IrisSdkStatus iris_set_beauty(const IrisBeautyParams* params);/* skin, jaw 등 */
IrisSdkStatus iris_render(uint64_t target_texture, int width, int height);
```

### 6.1 동시성 계약 — generation 검증 노출은 필수다

감사 blocker급 계약 결함(§3 GL/GPU "Detection Slot generation 검증이 어디에도 구현되지 않음 — 주석상 '호출측 수행'이지만 generation이 Java에 미노출이라 이행 불가능한 계약", §6.6 리스크 4 "DetectionSlot torn-read race가 정식 주입 경계로 승격되며 표면화 — 승격 전 수정 필수")을 경계 설계로 해소한다:

- `iris_set_landmarks`는 호출 스레드에서 코어 내부 버퍼로 **deep-copy**한다. 호출자 버퍼의 수명은 호출 반환 시점에 끝난다 (raw 포인터 보유 금지 — detectSync UAF finding의 재발 방지).
- 코어 내부는 seqlock 패턴 더블버퍼: 쓰기 전후 generation 증가(홀수=쓰기 중), 렌더 측은 읽기 전후 generation 일치 확인 후 불일치 시 재시도.
- generation은 C API로 노출한다 (`out_generation` 반환 + `iris_get_landmark_generation()`). "raw 포인터를 Java로 반환하는 API 자체를 폐기하고 '슬롯 핸들 + 내부 복사' 형태로 재설계"(감사 수정안)를 그대로 채택한다.
- **upright 프레임 치수(frame_width/height)는 랜드마크와 같은 호출로 주입되어 generation에 원자적으로 묶인다** (교차 검토 보완 — 2026-06-11). 코어 파생 어댑터(홍채 반경·EAR·face_rect)의 픽셀 환산(§7.0)은 반드시 이 치수를 사용한다. 렌더 타깃 치수(`iris_render`의 w/h)는 호스트가 화면 비율 타깃을 쓰면 프레임 종횡비와 달라질 수 있으므로 픽셀 환산에 사용을 금지한다 — 함정 #5(종횡비 왜곡)가 경계 설계 안에서 재발하는 유일한 구멍을 봉쇄.

**입력 유효성 계약 — 478 미만 입력 가드는 경계 설계의 일부다** (감사가 3곳에서 명시한 "주입 계약 도입 즉시 터지는 지뢰": beauty_roi_manager의 478 미만 silent false·인덱스 466 무검증 OOB, grid_mesh 468 하드코딩 + applyWarp 잠재 OOB, 패널 A "+1일 계약 가드 작업이 ①에 묶임"):

- **점 수는 478 고정 계약**으로 명문화하고, 시그니처에 `num_points`를 포함한다(상기 4함수안 반영). `num_points != 478`은 즉시 거부(에러 반환, silent false 금지) — 향후 점 수 확장은 메이저 버전 + 계약 개정으로만 한다.
- `pts == NULL`, NaN/Inf 포함 입력은 거부한다. 거부 시 직전 유효 주입(스테일)을 유지하며 generation은 증가하지 않는다.
- 추가 방어선으로 **바인딩 레벨 길이 가드**(JNI: 배열 길이 ≥ 478×3 검증 후 호출)를 의무화한다 — C 경계와 바인딩의 이중 가드.
- 이 계약 가드 작업(+1일)은 실행 순서 ①(경계 도입)에 귀속된다 (감사 패널 A 판정 — §13 견적 반영).

### 6.2 detector 파생 데이터의 경계 처리 (항목별 결정)

감사가 식별한 "실질 난이도는 detector가 부수 생산하던 파생 데이터의 코어 측 이식"(§6.1)에 대한 항목별 결정:

| 데이터 | 결정 | 근거·방법 |
|---|---|---|
| 478점 랜드마크 (x,y,z) | **주입 필드** (유일한 기하 정본) | `iris_set_landmarks` |
| timestamp | **주입 필드** | One-Euro dt. 카메라 타임스탬프 기준 단조 증가 |
| generation | **경계 신설 필드** (코어 발급, 노출 필수) | §6.1. 감사 torn-read finding |
| 홍채 중심·반경 | **코어 내 파생 어댑터** | 인덱스 468~477에서 유도, 픽셀 변환 후 거리 계산(§7.0). LensSimulator `IrisGeometry.kt`가 참조 구현 (감사 §6.3) |
| eyelid_ratio → visibility | **코어 내 파생 어댑터** | EAR 재계산 — 수식은 `temporal_stabilizer.cpp:315 computeEAR`에 이미 존재(감사 §6.2 pt6). 단 픽셀 공간 환산 후 계산으로 수정 (정규화 EAR 결함 finding 반영) |
| confidence | **경계에서 제거**(게이팅 무력화) | MediaPipe Tasks는 per-face confidence를 노출하지 않음. '검출 실패 = 주입 부재'로 표현하고, 게이팅은 visibility(EAR 파생)로 일원화. **구현 노트(W4-B1)**: 어댑터 수식 `visibility = confidence·(1-eyelid_ratio)`이 detector 경로와 공유되어 confidence=0이면 게이트가 닫힌다. 주입 경로는 detector 골든 불변을 위해 어댑터를 건드리지 않고, `deriveIrisResult`가 검출 시 confidence를 **게이트 통과 상수 1.0(곱셈 항등원)** 으로 고정한다 → visibility가 (1-eyelid_ratio)로 환원되어 'EAR 일원화' 효과를 달성(measurement로서의 confidence는 제거, 게이팅에서 중립화). presence 게이트는 detected(어댑터 side별 early-return)가 담당. 데모 Kotlin 형제 `TasksToIrisResult.kt`도 동일하게 confidence=1.0 고정 |
| avg_iris_luma | **GPU self-measure로 이전** (1순위) + 주입 옵션 필드(보조) | 감사 §6.2 pt5: gpu_lens_renderer.cpp:898, 1121에 'W6 이관' 주석으로 계획 기존재. P7-W2 default ON 실측이 fallback 상수 체인으로 후퇴하지 않도록 **경계 도입과 동시 처리** (§6.6 리스크 3) |
| face_rect | **코어 내 파생 어댑터** | 478점 메시 바운딩 박스 (감사 §6.3) |
| iris_quality_*, eye_refiner_used 등 detector 전용 메타 | **삭제** | Eye Refiner는 활성화 자체가 불가능한 결함 상태(좌표 폭주 finding) — 추적 외부화로 존재 이유 소멸 |

### 6.3 IrisResult의 강등

IrisResult는 코어 내부 파생 타입으로 강등한다 (감사 §6.2 pt3 분리 방안 채택). C/C++ 이중 정의 + `reinterpret_cast` + sizeof-only 가드(§6.6 리스크 5) 위에 새 경계 타입을 얹지 않는다 — 경계 도입 시 단일 정의 공유 또는 필드별 `offsetof static_assert`로 정리한다. `DetectorType`, `EyeRefinerPolicy`는 types.h에서 제거한다 (§6.4).

### 6.4 경계 도입의 표면 정리 전제 (감사 orchestration 판정 반영)

신규 4함수를 얹기 전에 기존 C API 표면의 구조 결함을 먼저 정리한다 (감사 §7 orchestration "부분 재작성" 양 패널 일치 — "리팩토링으로 못 고치는 구조 문제가 밀집"):

- **에러 코드 정본 단일화**: 동일 NotInitialized가 v1=100/v2=501로 이중 변환되는 현 상태를 단일 정본으로 통일하고, 신규 4함수는 그 정본만 사용한다.
- **기본값 단일 소스화**: BeautyFilterConfigV2 기본값 3원 분기(C++/Java/문서)를 단일 소스로 고정한다 — `iris_set_lens`/`iris_set_beauty` 파라미터 구조체의 기본값도 같은 소스에서 파생한다.
- **internal 함수의 헤더 선언화**: 헤더 없이 JNI에 수동 extern 중복 선언된 internal 9종(시그니처 드리프트 무검출)을 내부 헤더로 정식화한다.
- **코어 내부 468/478 하드코딩 정리**: grid_mesh(:82 468 하드코딩 + applyWarp 잠재 OOB), beauty_roi_manager(count 파라미터 없는 마스크 API의 인덱스 466 무검증) 등 랜드마크 개수 가정 하드코딩을 `landmark_count` 파라미터 전달 + 최소 개수 검증으로 정리한다 — §6.1 입력 유효성 계약이 코어 내부에서 무력화되지 않기 위한 **경계 승격의 전제**다 (감사 수정안 채택).
- sdk_api_v2의 함수 골격과 전역 상태 관리는 재작성하지 않고 정본으로 승격한다 (패널 A: "재작성하면 P7-W2 검증이 무효" — 보존 결정).

## 7. 좌표·시맨틱 계약 (명문 확정)

감사 §8 패널 A가 "findings에서 전부 모호 판정"한 4종을 확정한다. 기반 규약으로 LensSimulator ADR-0002의 478점 랜드마크 규약을 그대로 승계한다.

### 7.0 승계 규약 — 478점 랜드마크 (LensSimulator ADR-0002)

- 홍채: 중심 **468**(피험자 우안) / **473**(피험자 좌안), 경계 4점 순서는 **right(469/474) → top(470/475) → left(471/476) → bottom(472/477)** (이미지 좌표 기준 — 검증 에이전트가 실측으로 확정. patlevin IrisIndex의 LEFT/RIGHT 라벨은 반대이므로 신뢰 금지)
- 눈 윤곽(오클루전용) 16점: 우안 `33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246`, 좌안 `362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398`
- **반경·거리 계산은 반드시 픽셀 좌표 변환 후** (정규화 좌표 그대로 쓰면 종횡비 왜곡으로 가짜 타원 발생) — 감사가 확인한 EAR/radius/타원 fitting 결함 계열(temporal_stabilizer, fitEyeEllipse)의 재발 방지 원칙
- **홍채 z값(468~477) 사용 금지** (산출 방식 비공개/신뢰 불가)

CPU LensRenderer의 "iris[3]/iris[4]를 좌/우 쌍으로 가정해 회전각 ~45° 왜곡" finding이 보여주듯, 경계점 순서는 추측이 아니라 위 규약을 단일 출처로 강제한다. 코어 내 인덱스 상수는 단일 헤더로 통합한다 (감사 §6.2 pt7 권장).

### 7.1 회전 후 좌표 공간 — "upright 정규화 공간"이 주입 계약이다

- 주입 좌표는 **회전 보정이 완료된(upright) 프레임 기준의 정규화 좌표 `[0,1]`**이다. 회전 책임은 공급자(추적 글루)에 있다: MediaPipe Tasks에 `ImageProcessingOptions.rotationDegrees`(Android) / `orientation`(iOS)을 전달하면 출력이 upright 공간으로 나온다.
- 이는 현 detector의 규약("cv::rotate로 픽셀을 회전 후 검출하고 좌표를 upright 공간 + result.frameWidth/Height 계약으로 반환" — 감사 함정 #13 대조 결과 '설계 회피로 비해당')과 동일 공간이므로, 코어 소비측(resolveCoordinateSpace 등)은 무변경으로 유효하다.
- 코어는 upright 프레임의 width/height를 픽셀 환산 기준으로 받는다 (`iris_render`의 w/h 또는 별도 set). 90/270 회전 시 width/height 스왑은 공급자 책임이다.

### 7.2 z 스케일 — 홍채 z 금지, 비홍채 z는 x 동일 스케일·비기하 용도 한정

- 홍채 10점(468~477)의 z는 **사용 금지** (§7.0 승계).
- 비홍채 468점의 z는 MediaPipe 표준 규약(x와 동일 스케일로 정규화, 머리 중심 기준 상대 깊이)을 따르는 값으로 주입된다. 현 detector의 z는 "crop 종속 raw 값, MediaPipe 표준과 불일치"(감사 §7 infra 패널 B)였으므로, **전환 후 z 의미가 바뀐다** — 코어에서 z를 소비하는 곳은 전환 시 전수 재검토하며, z는 깊이 순서 판정 등 비기하 용도로만 허용하고 거리·반경 계산에 사용하지 않는다.

### 7.3 left/right 명명 — 인덱스가 정본, 라벨은 피험자(해부학) 기준

- **인덱스(468 그룹/473 그룹)가 정본이고 라벨은 보조 표기다.** 라벨은 MediaPipe canonical과 동일한 **피험자 해부학 기준**으로 통일한다: `right_eye` = 피험자 우안 = 468~472 = 비미러 프레임의 화면 왼쪽.
- 현 코어는 "left/right 라벨이 MediaPipe 해부학적 명명과 반전"(감사 정확성 finding: 코드 'left'=468-472=MediaPipe FACEMESH_RIGHT) + "내안각/외안각 인덱스 반전"(caruncle 보호 반대편 적용) + beauty_roi_manager 명명 반전 등 3곳 이상에서 충돌한다 — 3단계 계약 명문화 시 canonical 기준으로 일괄 정정하고, LensSimulator `LandmarkIndices.kt`를 명명 정본으로 삼는다 (감사 §8 패널 A 권고 채택).
- 화면 기준 게이트(예: 데모의 applyLeft)는 `screen_left/screen_right`로 명시 분리해 해부학 라벨과 혼용을 금지한다.

### 7.4 미러 규약 — 주입 좌표는 항상 비미러, 미러는 렌더 단계 단일 책임

- **주입되는 478점은 항상 비미러(센서 원본 upright) 공간이다.** 전면 카메라 미러는 렌더/표시 단계에서만 적용한다 — MediaPipe Tasks가 미러링을 지원하지 않으므로(ADR-0002) 이 규약이 공급자 측 자연 규약이기도 하다.
- 미러링 책임은 한 층으로 고정한다: 코어 렌더 측 단일 적용, 호출자별 분기 금지. 감사가 확인한 "미러링 규약이 호출자마다 다르고 모듈 내 주석은 허위"(beauty_roi_manager), "같은 입력에 두 곳에서 다른 규약(미러링 유/무)으로 computeROI 호출"(sdk_api_v2) 상태의 재발 방지 결정이다.
- L/R 의미는 미러 여부와 무관하게 §7.3(피험자 기준)을 유지한다 — "GL은 스왑·Overlay는 무스왑" 불일치(감사 finding)는 '무스왑(detector/피험자 기준) 단일 계약'으로 해소한다.

### 7.5 부칙 — 스무딩 파라미터의 단위 계약

One-Euro 필터 등 스무딩 파라미터는 **입력의 단위(정규화 좌표 vs 픽셀)를 계약의 일부로 헤더에 명시한다.** 스케일 전제가 다른 공간에 파라미터를 직접 적용하는 것을 금지한다 — LensSimulator 함정 #10이며, 감사가 확인한 "radius OneEuroFilter beta 7.5(정규화 공간 튜닝)를 픽셀 단위 radius에 적용해 스무딩이 사실상 무효" finding의 재발 방지 결정이다.

시작값과 적용 공간은 **LensSimulator 검증 구현(`FaceTracker.kt`)을 정본**으로 한다 (실코드 직접 확인, 2026-06-11):
- 파라미터: `min_cutoff=0.5, beta=0.007, d_cutoff=1.0` (`FaceTracker.kt:424-426`, `OneEuroFilter.kt` 기본값 동일)
- 적용 공간: **upright '픽셀' 좌표 공간에서 필터링 후 정규화 좌표로 환원해 전달** (`FaceTracker.kt:63` "정규화 공간은 종횡비 왜곡", `computeEye` 문서 "픽셀 공간에서 계산·필터링 후 정규화 좌표로 되돌려 저장") — 이식 시 동일하게 **픽셀 공간 적용을 계약**으로 한다. §7.0의 픽셀 거리 원칙과 일치한다.
- MediaPipe 프로덕션 값(min_cutoff 0.05/beta 80 — 객체 스케일 정규화 전제)의 직접 적용은 금지한다 (ADR-0002 승계).

> **각주 — 문서 드리프트 주의**: LensSimulator ADR-0002 텍스트에 구버전 표기("1.0→0.3 튜닝")가 남아 있었으나, 본 ADR의 지적을 받아 **LensSimulator 쪽이 실코드 대조(FaceTracker.kt:424-426 = 0.5/0.007/1.0, 픽셀 공간) 후 ADR-0002를 정정 완료했다** (2026-06-11 교차 검토). 원칙은 유지한다: §14 '값 무변경 이식' 원칙에 따라 **이식 정본은 문서가 아니라 코드(FaceTracker.kt)**다. 잘못된 공간 라벨(정규화)로 픽셀 튜닝값을 정규화 공간에 적용하면 radius beta 7.5 finding과 동일 구조의 결함이 재생산된다.

## 8. 결정 항목 ① — cpu-render 처분 (**확정: 옵션 B** — 2026-06-11 사용자 승인)

감사 §9-4: 유일한 패널 판정 이견(A: 부분 재작성 / B: 폐기·교체). Codex 교차 검토 보강: "내부 소비처가 0이어도 `iris_sdk_render_lens` 등 공개 C API가 살아 있으므로 삭제는 기술 판단이 아니라 API 호환성·버전 정책 판단"(§9-4, §11).

### 8.1 비교표

| 기준 | 옵션 A: 부분 재작성 유지 | 옵션 B: 코어에서 폐기 + 필요 시 별도 모듈화 |
|---|---|---|
| "픽셀은 경계를 넘지 않음" 목표 | **충돌 잔존** — cv::Mat 공개 헤더 노출(cpu_beauty_backend.h:13, lens_renderer.h:156)이 유일한 충돌 경로로 남음 (감사 §6.5) | **정합** — 충돌 경로 소멸 |
| OpenCV 의존 | 잔존 (헤더 11개·구현 9개 + Android JNI CMake `find_package(OpenCV REQUIRED)`) | **완전 제거 가능** (frame_processor의 NV21/NV12 변환도 추적 측과 함께 소멸 — 감사 §6.5) |
| 수리 비용 | lens_renderer 회전 산출 + cpu_beauty ROI 좌표 경로 재작성 + GPU 경로와 결과 동등성 신규 확보 (1,724줄 — 감사 §7 패널 B) | 0 (deprecate 마킹 + 문서) |
| 수리로 얻는 것 | GPU 불가 기기 폴백 | — (2026 tier 재정의상 레거시 제외 — 사용자 컨텍스트) |
| 검증 부담 | CPU/GPU 이중 경로의 실기기 육안 검증 이중 지불 (1인 개발) | 단일 GPU 경로만 검증 |
| 현 품질 실측 | CPU 렌즈: 항상 ~45° 회전 왜곡 출하, 소비처 0. CPU 뷰티 ROI: "사실상 한 번도 올바르게 동작한 적 없음" (감사 §7 패널 A) | 동일 사실이 폐기 근거 |
| 공개 API 영향 | 없음 | `iris_sdk_render_lens`(sdk_api.h:428), `iris_sdk_process`, CPU 뷰티 계열 deprecation 필요 |

### 8.2 deprecation / 메이저 버전 정책 (옵션 B 채택 시)

- **현행 1.x 라인**: `iris_sdk_render_lens` 등 공개 CPU 픽셀 API는 시그니처를 유지한 채 `IRIS_SDK_DEPRECATED` 마킹 + 헤더·문서에 "2.0에서 제거, GPU 텍스처 경로(sdk_api_v2)로 이행" 고지. 구현은 동결(결함 수리하지 않음 — 소비처 0 실측).
- **2.0 (랜드마크 주입 경계가 공개되는 메이저 버전)**: CPU 픽셀 API 제거. CPU 폴백이 외부 계약상 필요해지는 시점에만 별도 모듈(코어 밖, OpenCV 의존 포함)로 재작성한다.
- 견적 반영: 공개 API 호환 유지(점진 deprecation 경로) 포함 시 **+2~4일 버퍼** (감사 §9 Codex 보정).

### 8.3 권고

**옵션 B를 권고한다.** 근거: (1) 1인 개발 — 이중 경로 실기기 검증의 비용이 가장 비싼 프로젝트 (감사 §7 gpu-render 패널 A), (2) 내부 소비처 0 실측 (lens_renderer 데모 호출 0건 확인), (3) "픽셀은 경계를 넘지 않음" 목표와 충돌하는 유일한 경로이며 폐기 시 OpenCV 완전 제거 달성, (4) Codex도 패널 B(폐기·별도 모듈화) 지지 (§11).

> ✅ **확정 (2026-06-11)**: 사용자가 ADR 승인과 함께 **옵션 B를 선택**했다 — 폐기 + 1.x `IRIS_SDK_DEPRECATED` 마킹·구현 동결 → 2.0 제거, §8.2 정책 그대로.

## 9. 결정 항목 ② — 16KB 페이지 정렬 (실측으로 확정 결함, 대응 결정)

감사 §10의 "16KB 정렬은 빌드 산출물 실측 검증으로 확정" 항목은 **2026-06-11 실측으로 해소되었다 — '가능성'이 아니라 확정 결함이다**:

> `android/build/.../merged_native_libs/debug/.../arm64-v8a/`의 `libiris_jni.so`, `libtensorflowlite.so`, `libtensorflowlite_gpu_delegate.so` 전부 LOAD align 2**12 (4KB) — 16KB 페이지 정렬 미달 실측 확정 (objdump --private-headers). AGP 8.5.0, max-page-size 링커 플래그 부재.

**판정**: 이 AAR을 쓰는 호스트 앱은 Android 15+ 타깃 Play 제출 시 16KB 미정렬 네이티브 라이브러리로 거부/경고 대상이고, 16KB 페이지 기기에서 로드 실패 가능 — 외부 배포 SDK로서 패키징 결함 확정 (감사 §3 정확성 finding의 잠정 판정을 확정으로 승격).

**대응 결정**:
1. **자체 .so (libiris_jni.so 및 코어 .so)**: ndkVersion r27+ 핀 + cmake `-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON`(또는 링커 플래그 `-Wl,-z,max-page-size=16384`) + **AGP 8.5.1+ 업그레이드**로 zip(uncompressed .so) 정렬까지 확보.
2. **TFLite prebuilt .so 2종**: 본 ADR의 추적 외부화(3단계)로 삭제 대상 — 별도 재정렬 작업을 하지 않는다. MediaPipe Tasks 0.10.35는 16KB 정렬이 보장된 버전(0.10.26+)이므로 전환 자체가 이 결함의 항구 해소다 ("16KB 정렬 관리를 구글에 위임" — 전환 근거). **단, §10 후퇴 트리거 발동 시 이 전제(삭제)가 소멸하므로, TFLite prebuilt 2종의 16KB 재정렬(또는 16KB 정렬 빌드본 교체)이 후퇴 수리 트랙에 의무 편입된다** — 확정 패키징 결함의 해소 경로가 분기 어디에서도 끊기지 않게 한다.
3. **검증 절차를 빌드 파이프라인에 추가**: 산출물에 대해 `objdump --private-headers`로 전 LOAD 세그먼트 `align 2**14` 확인 + `zipalign -c -P 16` 검사 (감사 수정안의 check_elf_alignment 절차 채택). 3단계 머지 게이트에 포함한다.

## 10. 기각(축소) 옵션 — "현 추적 유지 + 인터페이스만 도입"

계획 문서 2-1의 축소 옵션이다. 감사 결과 결합도가 낮음(코드 결합 얕음)으로 확정되어 전면 전환을 채택하지만, 다음 **후퇴 트리거** 중 하나라도 발동하면 이 옵션으로 축소한다:

| # | 트리거 | 판정 기준 |
|---|---|---|
| T1 | **MediaPipe Tasks 좌표 규약 정합 실패** | ③-3 A/B에서 동일 프레임의 자체 추적기 대비 홍채 중심 오차가 **좌표 계약 위반 규모** — 계통적 오프셋 ≥ 검출 홍채 반경의 100%(픽셀 환산), 또는 축 스왑·미러 반전·정규화 오류의 전형 패턴(프레임 폭/높이 단위 오프셋) — 로 나타나고, 어댑터(회전/미러/정규화) 수정 2회 이내로 해소되지 않을 때 — 감사 §6.6 리스크 1("정합하지 않으면 렌즈 위치가 전부 어긋남"). **골든 ε(§11)는 이 판정에 차용하지 않는다**: ε는 동일 detector 출력의 동작 불변 회귀 감지 전용이며, 이종 모델 간에는 정상 정합 상태에서도 ε 초과의 미세 차이가 항상 존재한다(감사 확정 — 자체 추적기의 canonical 이탈만으로도 초과). 수렴 목표는 'ε 이내 일치'가 아니라 '계약 위반 패턴의 소멸'이고, 잔여 모델 수준 차이는 T2 육안 판정으로 평가한다 |
| T2 | **실기기 A/B 품질 미달** | 같은 코어에 기존 추적 vs Tasks를 꽂은 A/B(계획 ③-3)에서 사용자 육안 판정(본 프로젝트의 합격 판정 방식)이 정확도·안정성 체감 열세일 때. 밝은 환경 우선 비교 (저조도는 우선순위 낮음 — 사용자 컨텍스트) |
| T3 | **성능 회귀** | Tasks GPU delegate 초기화 실패 기기군(2026 tier 재정의 기준)에서 CPU 폴백 지연이 33ms 프레임 버짓을 상시 초과해 30fps 목표 미달일 때 |
| T4 | **패키징/배포 회귀** | Tasks AAR 의존 추가로 SDK 크기 20MB 목표 초과 또는 호스트 의존성 충돌이 해소 불가일 때 |

후퇴 시에도 **주입 경계(`iris_set_landmarks`)·좌표 계약(§7)·골든 인프라(§11)는 그대로 자산으로 유지된다** — 후퇴는 '경계 뒤의 기본 공급자를 자체 추적기로 유지'하는 것만 의미하며, ③-1(경계 도입)의 산출물은 무효화되지 않는다. 단 이 경우 다음이 별도 수리 트랙으로 전환된다: ① 자체 추적기의 blocker 2건(InferenceThread terminate 계열), ② canonical 이탈 major들, ③ **TFLite prebuilt .so 2종의 16KB 재정렬(또는 16KB 정렬 빌드본 교체)** — §9 대응 2의 '전환으로 항구 해소' 전제가 후퇴로 소멸하므로 확정 패키징 결함이 수리 트랙에 의무 편입된다. (수리 비용이 교체 비용을 상회한다는 감사 판정이 후퇴 결정 시 재평가 대상이 된다.)

**후퇴 판정 절차**: 판정 시점은 ③-3 Android A/B 비교 종료 시점 1회로 고정한다 (트리거 T1~T3은 이 시점에 일괄 평가, T4는 의존성 추가 시점에 즉시 평가). 판정 주체는 사용자(실기기 육안 + A/B 데이터)다. 부분 후퇴는 허용하지 않는다 — 플랫폼별로 공급자가 갈리는 상태(Android=Tasks, iOS=자체)는 패리티 부채를 재생산하므로, 후퇴 결정 시 전 플랫폼이 자체 추적기 공급자로 통일된다.

> **③-3 A/B 주의 (교차 검토 보완 — 2026-06-11)**: 회전 입력에서는 "기존 추적기 대비 오차"(T1) 기준이 성립하지 않는다 — 골든이 입증한 rot0≡rot180 퇴화로 기존 추적기 쪽 비교 기준 자체가 회전 입력에서 오염돼 있다. A/B 판정은 rot0(실기기 자연 경로) 중심으로 수행하고, 회전 변형은 신규 공급자의 §7.1 계약 검증 용도로만 사용한다.

## 11. 검증 전략 — 골든 베이스라인 (2-2, 병렬 구축 중)

모든 동작 불변 단계의 머지 조건은 골든 베이스라인 통과다 (계획 문서 원칙). 감사 §9-5 판정("구축이 아니라 신축 — 렌더 픽셀 검증 0건, 좌표 회귀 0건, 기존 골든 단정은 2배 범위 허용으로 무의미")에 따라 신축하며, 본 ADR과 병렬로 2단계 작업자가 구축 중이다 (`cpp/tests/golden/{inputs,baseline}`).

**구조**:
- **입력**: 정적 이미지 세트 + 합성 변형(회전 0/90/180/270 × 미러 유/무 × 감마 변형) — 좌표 계약 4종(§7)의 각 축을 독립 검증하는 조합. blocker finding("좌표 변환·회전 경로 회귀 테스트 전무")의 직접 해소 수단.
- **랜드마크 골든**: 현재 코드(현 detector 기준)의 478점 + 파생 데이터(홍채 중심·반경, EAR, face_rect) **JSON 덤프**. 전환 전/후 좌표 동등성 비교의 기준선 — tracking 삭제는 반드시 이 베이스라인 확보 후순위다 (감사 §8 패널 A).
- **렌더 골든**: CPU 경로 렌더 출력 이미지 (데스크톱에서 결정적 재현 가능). **GPU 렌더 골든은 데스크톱 EGL 부재로 실기기 후속** — 실기기 캡처 경로가 마련되는 시점에 추가하며, 그전까지 GPU 경로의 합격 판정은 실기기 육안(사용자 방법론) + A/B 토글로 한다.
- **통과 기준**: **골든 도구(`scripts/golden_compare.py`)의 기본값을 참조값으로 한다** — 현재 기본값: 정규화 좌표 ε `1e-4`(`--eps-norm`), 픽셀 필드 ε `0.5`(`--eps-pix` — 반지름 등 픽셀 단위 필드 전용. 홍채 중심·face_mesh 좌표는 정규화 필드라 `eps-norm` 적용), 렌더 PNG는 바이트 동일성(불일치 시 diff 통계 리포트). 기준 변경은 골든 도구 쪽에서만 한다 (이중 정의 금지 — 본 ADR의 수치는 참조 인용).
- **ε의 적용 범위 (한정)**: 이 ε는 **같은 detector 출력을 전제로 한 동작 불변 검증 전용**이다. 적용 대상은 동작 불변을 표방하는 PR(③-1 경계 도입, 코드 이동, 표면 정리)이며, **이종 추적기 간 비교(③-3 A/B, §10 T1)에는 차용하지 않는다** — 모델이 다르면 ε 초과는 결함이 아니라 정상이다. 추적 교체 PR(④)은 랜드마크 골든 ε 일치가 원리상 불가능하므로 §12의 대체 절차(A/B 비교 + 재기준선)를 따른다.

골든은 현 detector 출력 기준이므로 "현재 코드의 결함이 포함된 스냅샷"임을 명시한다 — 골든의 목적은 정답성 증명이 아니라 **동작 불변 단계의 조용한 변화 감지**다. 결함 수리(③-2, geometry 수식)는 골든 기준값 갱신을 동반하는 의도적 변화로 분리 기록한다.

**옵션 B(cpu-render 폐기)와의 정합**: 골든이 CPU 렌더 경로를 사용하는 것은 '데스크톱에서 결정적으로 재현 가능한 검증 수단'의 선택이지 CPU 경로 보존 결정이 아니다. §8에서 옵션 B를 채택하더라도 CPU 렌더 골든은 (a) GPU 렌더 골든의 실기기 캡처 경로가 가동될 때까지 테스트 전용(공개 표면 제외)으로 동결 유지하거나, (b) 랜드마크 골든 + 실기기 육안으로 대체한다 — 처분은 GPU 골든 가동 시점에 결정하며, 그전까지 CPU 렌더 골든의 기준 구현은 동결한다(결함 수리 금지 — §8.2 구현 동결과 일치).

## 12. 3단계 진입 조건 (머지 게이트) — 개요만

3단계(경계 도입·코드 이동)는 본 ADR의 사용자 승인 후에만 착수한다. 진입·머지 게이트:

1. **본 ADR 승인** (결정 항목 ① 선택 확정 포함)
2. **골든 베이스라인 가동** — §11의 랜드마크/CPU 렌더 골든이 현 코드 기준으로 캡처·재현 확인 완료
3. **데모 검증 통로 정화 선행** — 감사 §9 실행 순서 ①: torn 스냅샷 제거, KT 무음 폴백 제거, OverlayView 정책 통일. 모든 단계의 실기기 검증이 이것에 의존한다 (§9-5 "모든 단계의 선행 조건")
4. 이후 감사 §9 권장 실행 순서를 따른다: ② 골든 캡처 → ③ types.h 계약 명문화 + `iris_set_landmarks` 경계 승격(generation 검증 + §6.1 입력 유효성 가드 + §6.4 468 하드코딩 정리 포함) → ④ MediaPipe Tasks 통합 + 자체 추적기·죽은 코드 제거 → ⑤ geometry 수식 수리 병행 → ⑥ 실기기 회귀. 각 단계는 독립 PR(W 단위 PR 정책), 종료 시 항상 동작 상태(스트랭글러 패턴), 골든 통과가 머지 조건.

각 PR의 머지 게이트 체크리스트:

- [ ] 골든 통과 (랜드마크 좌표 ε + CPU 렌더 픽셀 diff — §11 도구 기본값) — **동작 불변 PR에 한함**. 추적 교체 PR(④)은 현 detector 기준 랜드마크 골든과의 ε 일치가 원리상 불가능하므로(이종 모델), 이 항목을 **A/B 비교(§10 T1 판정 기준) + 베이스라인 재캡처(재기준선)**로 대체하고 §11 원칙(의도적 변화 분리 기록)의 확장으로 기록한다 — 무언의 재기준선 우회 금지, 재기준선은 명시 절차로만
- [ ] 기존 테스트 전부 통과 (`cpp/cmake-build-debug` + ctest)
- [ ] 16KB 정렬 검증 통과 (§9-3 절차 — Android 산출물 변경이 있는 PR에 한함)
- [ ] 동작 불변 단계는 실기기 A/B 토글에서 육안 차이 없음 (사용자 판정)

상세 설계(함수 시그니처 최종형, 파일 이동 순서, 플랫폼별 일정)는 3단계 각 W의 작업 문서에서 다룬다 — 본 ADR은 경계와 계약만 확정한다.

## 13. 리스크 및 견적 (감사 §6.6 승계)

| # | 리스크 | 본 ADR의 대응 |
|---|---|---|
| R1 | 좌표 규약 차이 — 현 detector는 회전 보정 후 좌표 반환, Tasks 출력 좌표계와 정합 실패 시 렌즈 위치 전부 어긋남. 실기기 검증 필수 | §7.1 upright 계약 명문화 + §11 회전/미러 합성 변형 골든 + §10 T1 후퇴 트리거 |
| R2 | detector 메타 소실 — confidence/eyelid_ratio/iris_quality 소실로 visibility 게이팅·blink 품질 회귀 | §6.2 EAR 파생 어댑터(픽셀 공간) + 튜닝 재검증을 ⑥ 실기기 회귀에 배정 |
| R3 | avg_iris_luma 실측 소실(P7-W2 default ON) — fallback 상수 체인 후퇴 시 TintLinearV2 품질 회귀 | §6.2: W6 이관분 GPU self-measure를 **경계 도입과 동시 처리** |
| R4 | DetectionSlot torn-read race가 정식 주입 경계로 승격되며 표면화 | §6.1 seqlock + generation 노출을 경계 설계에 내장 (승격 전 수정) |
| R5 | C/C++ IrisResult reinterpret_cast 수동 동기화 위에 새 경계 타입을 얹으면 ABI 취약 누적 | §6.3 단일 정의 공유/offsetof static_assert 후 경계 타입 도입 |

**견적** (감사 §9, 양 패널 보정 + Codex 보정): 원견적 12~16 사람·일에 골든 인프라 신축 +3~5일, 데모 정화 +1~2일, geometry 병행 +2~3일, 478 입력 계약 가드 +1일(감사 패널 A — 실행 순서 ①에 귀속, §6.1), LensSimulator 이식 -1~2일 상쇄를 반영해 **약 18~25 사람·일 (1인 4~5주)**. 공개 CPU 픽셀 API 호환 유지(점진 deprecation) 포함 시 **+2~4일**. CPU 경로를 공식 deprecated 처리(§8 옵션 B)하고 LensSimulator 자산을 그대로 이식하면 하한(17일)에 수렴 가능하다.

## 14. 사용자 컨텍스트 반영 (검증 방법론 전제)

- **1인 개발 — 실기기 육안 검증이 합격 판정이다**: 시각 품질 벤치는 평가자 다수결이 아니라 본인 실시간 토글 체감으로 한다. 따라서 검증 통로(데모)의 신뢰성 회복(§12-3)이 다른 모든 검증보다 선행하며, 재작성 회피(§2-5)의 핵심 근거도 이 검증 비용이다.
- **저조도 우선순위 낮음**: 뷰티 시뮬레이션 특성상 저조도 사용이 드물다. A/B 품질 비교(§10 T2)와 골든 입력 세트는 밝은 환경을 우선한다.
- **W 단위 PR 정책**: 3단계 각 W는 독립 PR로 진행한다 (내부 W 통합 머지는 PR 생략 가능 — 기존 정책 유지).
- **레퍼런스 이식 우선**: "값 무변경 이식"(skin_mask_geometry 선례)을 신규 작성보다 우선한다 — LensSimulator의 FaceTracker.kt/CoordMapper.kt/IrisGeometry.kt/LandmarkIndices.kt가 이식 대상 정본이다.

## 15. 결과 (이 결정으로 확정되는 것 / 열려 있는 것)

**확정**: 목표 아키텍처(§3), MediaPipe Tasks 0.10.35(§5), C API 경계 4함수 + 파생 데이터 처리(§6), 좌표·시맨틱 계약 4종(§7), 16KB 대응(§9), 후퇴 트리거(§10), 골든 전략(§11), 3단계 게이트(§12).

**확정 완료**: 결정 항목 ① cpu-render 처분 = **옵션 B** (폐기 + deprecation 정책 §8.2) — 2026-06-11 ADR 승인과 함께 사용자 확정.

**3단계로 이월**: 함수 시그니처 최종형·types.h 필드 정리 상세·플랫폼별(iOS/Web) 전환 일정 — 본 ADR의 계약 범위 밖.

---

### 변경 이력

| 일자 | 내용 |
|---|---|
| 2026-06-11 | 초안 작성 — 1단계 감사(133건, Codex 교차 검토 반영) 기반. 16KB 실측 확정 반영. 사용자 검토 대기 |
| 2026-06-11 | 적대 리뷰 반영 — ① §7.5 One-Euro 정정(실코드 검증: min_cutoff 0.5·픽셀 공간, ADR-0002 텍스트 드리프트 각주) ② §10 T1 판정 기준을 골든 ε에서 분리(좌표 계약 위반 규모) + §11 ε 적용 범위 한정 + §12 게이트에 추적 교체 PR 대체 절차 ③ §10 후퇴 시 TFLite 16KB 재정렬 수리 트랙 편입(§9 대응 2 단서) ④ §6.1 입력 유효성 계약(num_points 478 고정 + 바인딩 이중 가드) + §6.4 468 하드코딩 정리 전제 + §13 견적 +1일 |
| 2026-06-11 | **승인** — LensSimulator 세션 교차 검토(수용 8/부분 2/거부 0) + 사용자 확인. 결정 항목 ① **옵션 B 확정**. 보완 반영: §6 `iris_set_landmarks`에 frame_width/height 편입(generation 원자 결속, 함정 #5 봉쇄), §10 ③-3 A/B 회전 입력 주의(rot0 중심 판정). §7.5 각주 갱신(LensSimulator ADR-0002 정정 완료) |
| 2026-06-16 | **§3 보강 — 검출 폴백(Eye-Only) 위치 결정**: ADR 공백(MediaPipe 얼굴 검출 실패 시 폴백 미정의)을 사용자 결정으로 메움. Eye-Only 폴백 = **플랫폼 글루 책임**(코어 밖, TFLite를 글루에 위임해 코어 .so TFLite-free 유지). 코어 폴백(TFLite 코어 잔존) 기각. → 코어 DetectorType/EyeOnly/Hybrid는 코어 폴백 미보유 확정으로 W4-D 완전 삭제, Eye-Only 폴백 구현은 ④ 이후 별도 글루 트랙(Phase 9+). W4-B1 완료(af07e04) 후 W4-B2 착수 중 발견·결정 |
