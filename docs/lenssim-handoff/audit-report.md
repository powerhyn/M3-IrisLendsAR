# IrisLensSDK 리팩토링 감사 보고서 (1단계)

| 항목 | 내용 |
|---|---|
| 일자 | 2026-06-11 |
| 브랜치 | `refactor/p1-audit` (develop d224854 기준, 코드 수정 없음) |
| 근거 계획 | `docs/lenssim-handoff/refactoring-audit-and-tracking-migration-plan.md` 1단계 |
| 방법 | 멀티에이전트 적대적 리뷰 — finder 9개(6관점) → finding별 반박 검증(blocker/major 2표+캐스팅보트, minor 1표, 불확실 시 반박 우선) → 결합도 누락 sweep + 주입지점(seam) 적대 검증 → 완전성 비평 → 2라운드 보완 감사 4건 → 모듈 판정 패널 2인(보수파/부채파) |
| 규모 | 서브에이전트 약 380개, 도구 호출 3,000+ (사용량 한도 2회·수동 중단 2회 → 트랜스크립트 수확 체크포인트로 전량 회수) |

> ⚠️ **본 보고서는 Claude 멀티에이전트 감사 + Codex(gpt-5.5 xhigh) 외부 교차 검토 1회를 반영한 판본이다** (§11). Codex는 blocker 2건과 지정 표본 major를 코드 직접 대조로 전부 사실 확인했고, 반박 3건·추가 발견 2건은 본문에 반영했다. Gemini 등 추가 교차 리뷰는 미실시.

## 1. 결과 요약

- 원시 findings **141건** 발굴 → 적대적 검증 통과(확정) **133건**, 반박 폐기 8건 (부록 A)
- 확정 severity 분포 (검증자 다수 의견으로 9건 조정 반영, 부록 B): **blocker 2 / major 63 / minor 68**

| 관점 | blocker | major | minor | 계 |
|---|---|---|---|---|
| 테스트 안전망 | 1 | 7 | 3 | 11 |
| GL/GPU 상태 | 0 | 12 | 14 | 26 |
| 스레드/수명 | 1 | 4 | 5 | 10 |
| 정확성 | 0 | 17 | 18 | 35 |
| 결합도 | 0 | 3 | 4 | 7 |
| 플랫폼 패리티 | 0 | 9 | 4 | 13 |
| 2라운드 보완 감사 | 0 | 11 | 20 | 31 |
| **합계** | **2** | **63** | **68** | **133** |

### Blocker 목록 (2건)

- **InferenceThread 초기화 실패 시 joinable 스레드 미회수 → std::terminate (프로세스 크래시)** — `cpp/src/inference_thread.cpp:23-25, 49, 55-59, 313-317` (스레드/수명)
- **좌표 변환·회전 경로(detectOnlyWithRotation/submitFrameWithRotation/letterbox 역변환) 회귀 테스트 전무** — `cpp/include/iris_sdk/frame_processor.h:213-276` (테스트 안전망)

## 2. Blocker 상세

### InferenceThread 초기화 실패 시 joinable 스레드 미회수 → std::terminate (프로세스 크래시)
- 위치: `cpp/src/inference_thread.cpp:23-25, 49, 55-59, 313-317` | 관점: 스레드/수명 | confidence: high | (검증자 의견으로 blocker 유지)
- **근거**: threadLoop()는 detector 초기화 실패 시 `state_ = ThreadState::Stopped; return;`(314-316행)으로 종료하지만 스레드 객체는 join될 때까지 joinable로 남는다. 그런데 stop()은 `if (current == ThreadState::Stopped) return;`(56-59행)으로 조기 반환하여 join을 건너뛴다. 따라서 (a) FrameProcessor::Impl::initialize의 실패 경로 `inference_thread_.reset()`(frame_processor.cpp:190-192) → ~InferenceThread → stop() 조기반환 → joinable 상태의 `std::thread thread_` 멤버 소멸 → C++ 표준에 따라 std::terminate. (b) 동일 객체로 start() 재호출 시에도 `thread_ = std::thread(...)`(49행)가 joinable 스레드에 move-대입되어 std::terminate. 모델 파일 누락/손상 등으로 MediaPipeDetector::initialize가 실패하면 iris_sdk_init이 에러 코드를 반환하는 대신 결정적으로 앱이 크래시한다 (InferenceThread 모드가 기본값 — sdk_api.cpp:56 `g_use_inference_thread_request = true`).
- **영향**: detector 초기화 실패라는 평범한 에러 경로가 graceful 에러 반환 대신 프로세스 abort로 이어짐. 아키텍처 전환 전 회귀 테스트(모델 경로 오류 주입)에서 반드시 걸릴 결함.
- **수정안**: stop()에서 state와 무관하게 `if (thread_.joinable()) thread_.join();`을 항상 수행하도록 변경. 조기 반환 조건을 'thread_가 joinable이 아닐 때'로 교체. start() 진입 시에도 joinable이면 먼저 join.

### 좌표 변환·회전 경로(detectOnlyWithRotation/submitFrameWithRotation/letterbox 역변환) 회귀 테스트 전무
- 위치: `cpp/include/iris_sdk/frame_processor.h:213-276` | 관점: 테스트 안전망 | confidence: high | (검증자 의견으로 blocker 유지)
- **근거**: frame_processor.h:219 `IrisResult detectOnlyWithRotation(const uint8_t* frame_data, ..., int rotation_degrees)`, :275 `void submitFrameWithRotation(...)` 선언 확인. cpp/tests/ 전체 grep에서 `submitFrame|WithRotation|detectOnlyWithRotation|async` 매치 0건. letterbox 역변환 상태(mediapipe_detector.cpp:225-229 `letterbox_scale/letterbox_pad_x/letterbox_pad_y`)와 anchor 생성(:418 `generateAnchors()`)도 tests에서 `decode|letterbox|anchor` 매치 0건. 테스트 내 'rotation' 매치는 전부 LensConfig.rotation 기본값(test_sdk_api.cpp:341)과 face_rotation 구조체 저장(test_types.cpp:206-214) 같은 무관 항목.
- **영향**: 이번 전환의 핵심 계약이 '478점 랜드마크 좌표를 코어에 주입'인데, 좌표가 생성·역변환되는 경로(회전 0/90/180/270 + letterbox 패딩 역변환)에 회귀 감지 수단이 0. 추적 분리 과정에서 좌표 오프셋·축 스왑·회전 책임 누락이 발생해도 어떤 테스트도 실패하지 않는다 — 전환 검증을 차단하는 구조적 공백.
- **수정안**: 전환 착수 전: 동일 프레임을 4개 회전으로 입력해 detectOnlyWithRotation 결과 좌표를 상호 변환 비교하는 회귀 테스트 추가. letterbox 역변환을 순수 함수로 헤더 추출(test_skin_mask_geometry 선례)해 단위 고정.

## 3. 관점별 Major findings

### 정확성 (17건)

- **자체 libiris_sdk.so 16KB 페이지 정렬 미조치 — AGP 8.5.0 + ndkVersion 미지정 + max-page-size 링커 플래그 부재** — `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/android/iris-sdk/build.gradle.kts:20-116 (+ android/build.gradle.kts:10-11)`
  - 영향: 이 AAR을 쓰는 호스트 앱이 Android 15+ 타깃으로 Play 제출 시 16KB 미정렬 네이티브 라이브러리로 거부/경고 대상. 16KB 페이지 기기(Pixel 9 계열 등)에서 로드 실패 가능. 외부 배포 SDK로서 치명적 패키징 결함.
  - 수정안: ndkVersion을 r27+로 핀하고 cmake arguments에 "-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON"(또는 링커 플래그 -Wl,-z,max-page-size=16384) 추가, AGP를 8.5.1+로 올려 zip 정렬까지 확보. 빌드 산출물에 대해 check_elf_alignment 스크립트로 검증 절차 추가.
- **consumer-rules.pro에 BeautyFilterConfig/BeautyFilterConfigV2 keep 규칙 누락 — JNI 리플렉션 접근 클래스가 호스트 R8에서 제거/난독화되어 release 빌드 SDK 초기화 실패** — `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/android/iris-sdk/consumer-rules.pro:전체 (iris_jni.cpp:131-186 대조)`
  - 영향: 함정 #9의 자체-SDK 버전: 외부 고객사 release 빌드에서 뷰티 기능뿐 아니라 JNI 캐시 초기화 단계 전체가 실패해 SDK가 동작하지 않음(배포 관점에선 blocker성). 데모 앱은 minify 미사용이라 개발 중 발견 불가.
  - 수정안: consumer-rules.pro에 추가: -keep public class com.irislenssdk.BeautyFilterConfig { public *; } 및 BeautyFilterConfigV2 동일. 장기적으로는 JNI가 필드를 읽는 모든 클래스를 @Keep 애노테이션 또는 keep 규칙 생성 스크립트로 단일 관리.
- **TemporalStabilizer의 radius OneEuroFilter — 정규화 공간에서 튜닝한 beta(7.5)를 픽셀 단위 radius에 적용해 스무딩이 사실상 무효** — `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/cpp/include/iris_sdk/temporal_stabilizer.h:17-21 (+ temporal_stabilizer.cpp:273-275, mediapipe_detector.cpp:209-210/3108-3143)`
  - 영향: 홍채 반지름 지터(렌즈 크기 펄럭임)가 설계상 스무딩되어야 하나 수학적으로 거의 무보정 통과 — 함정 #10(스케일 정규화 전제 파라미터를 다른 스케일에 직접 적용 금지)의 정확한 사례. 함정 문서의 'beta 80' 대신 'beta 7.5/0.3'이라는 점만 다를 뿐 단위 전제 불일치 구조가 동일.
  - 수정안: radius를 정규화 단위(radius_px/frame_width)로 변환해 필터링 후 복원하거나, radius_beta를 픽셀 스케일에 맞게 재튜닝(예: 7.5/frame_width 수준). 필터 입력의 단위 계약을 헤더 주석에 명시.
- **내안각/외안각 인덱스 반전 — GPU 경로(C++/Kotlin 동일)에서 caruncle 보호 0.85 반경이 반대편(귀 쪽)에 적용** — `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/cpp/src/gpu/gpu_lens_renderer.cpp:62-65, 678-690 (Kotlin 동일: CameraGLRenderer.kt:71-74, 1489-1503, 셰이더 274-285)`
  - 영향: 비대칭 타원 Eye Mask의 설계 의도(눈물언덕/caruncle 침범 방지) 반전: 렌즈 마스크가 코 쪽으로 과확장되어 caruncle을 덮고, 귀 쪽은 15% 일찍 잘림. 양안 모두, GPU 렌즈 경로 전체(C++ 코어 + Kotlin 데모 셰이더) 공통. 함정 #4의 본질(유명 구현의 LEFT/RIGHT 라벨 반전 맹신)과 동일 계열 결함이 '코어 내부 두 파일 간 상충'으로 존재.
  - 수정안: gpu_lens_renderer.cpp와 CameraGLRenderer.kt의 INNER/OUTER 상수를 mediapipe_detector.cpp 정의(INNER=133/362, OUTER=33/263)와 일치시키고, 단일 헤더 상수로 통합해 재발 방지. 수정 후 실기기에서 양안 마스크 비대칭 방향 육안 확인.
- **홍채 5점 경계 순서 오인 — CPU LensRenderer가 iris[3]/iris[4]를 좌/우 쌍으로 가정해 렌즈 회전각이 항상 대각선(~45°)으로 왜곡** — `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/cpp/src/lens_renderer.cpp:105-142, 551`
  - 영향: CPU 렌더 경로(iris_sdk_render_lens / processFrame)에서 패턴 있는 렌즈 텍스처가 항상 약 45° 잘못 회전되어 합성됨. 머리 기울임(roll) 보정 의도가 완전히 빗나가고, 랜드마크 노이즈에 따라 회전이 흔들리는 아티팩트 발생. 랜드마크 주입 아키텍처 전환 시 이 코드가 '경계점 시맨틱'을 가정하는 유일한 코어 소비처이므로 전환 전 반드시 정리 필요.
  - 수정안: 회전각은 (iris[1]=right, iris[3]=left) 쌍으로 계산: angle = atan2(right.y - left.y, right.x - left.x). 주석의 라벨도 center/right/top/left/bottom으로 정정. 더 견고하게는 눈꼬리 랜드마크(33↔133) 기반 회전 사용 권장.
- **EAR(깜빡임 판정)을 정규화 좌표 거리로 계산하면서 픽셀 공간 기준 고정 임계값 0.2와 비교 — 프레임 종횡비에 따라 blink 오판** — `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/cpp/src/temporal_stabilizer.cpp:315-338 (+ temporal_stabilizer.h:42)`
  - 영향: blink hold/fade 로직이 입력 해상도·회전 방향에 따라 비결정적으로 동작: 렌즈가 멋대로 사라지거나(상시 blink 오판) 감은 눈 위에 렌즈가 남는(blink 미검출) 두 방향 모두 가능. 함정 #5의 전형적 사례.
  - 수정안: computeEAR에서 거리 계산 전 x에 frame_width, y에 frame_height를 곱해 픽셀로 변환하거나, 최소한 dx *= aspect(=w/h) 보정. IrisResult에 이미 frameWidth/frameHeight가 있으므로 추가 정보 불필요.
- **CPU 뷰티 ROI 경로의 마스크 좌표계 불일치 — 풀프레임 래스터 마스크를 ROI 사각형에 리사이즈 적용** — `cpp/src/cpu_beauty_backend.cpp:756-826 (vs cpp/src/beauty_roi_manager.cpp:233-234)`
  - 영향: CPU 백엔드(GPU 불가 시 fallback)의 ROI 모드에서 눈/눈썹/입술 보호 마스크와 페더 마스크가 실제 얼굴 부위와 크게 어긋난 위치에 적용 — 눈/입술이 블러되고 엉뚱한 영역이 보호되는 잘못된 결과.
  - 수정안: 마스크 좌표계를 한 가지로 확정: (a) 마스크를 ROI-local로 래스터(landmarkToMaskCoord 활용)하고 GPU 셰이더 샘플 UV를 ROI 기준으로 변환하거나, (b) CPU 경로에서 풀프레임 마스크의 actual_rect 부분영역을 crop 후 리사이즈. dead code인 landmarkToMaskCoord는 제거 또는 채택.
- **EyeRenderPacket 홍채 반경 정규화 규약 모순 — adapter는 /frame_width, 실제 렌더러는 /frame_height** — `cpp/src/gpu/eye_render_packet_adapter.cpp:35, 50-56, 82-84 (vs cpp/src/gpu/gpu_lens_renderer.cpp:962-967)`
  - 영향: 현재 렌더러는 packet에서 avg_iris_luma만 소비(gpu_lens_renderer.cpp:898-904)하므로 잠복 결함이지만, EyeRenderPacket은 헤더 주석대로 '후속 W가 의존하는 최소 고정 계약'이며 랜드마크 주입 아키텍처 전환의 핵심 경계면. 이 규약대로 구현될 미래 소비자는 반경이 W/H배(세로 9:16에서 1.78배) 틀어진다. fallback Y-slab도 세로 프레임에서 과대 높이.
  - 수정안: packet 규약을 렌더러 실규약(height 정규화)으로 통일하고 adapter에서 `radius_px / frame_height` 사용, eye_render_packet.h:27 주석 수정. fallback slab도 동일 축 단위로 정리. 전환 1단계에서 계약 문서부터 바로잡을 것(코드 수정은 지금 하지 말 것).
- **fitEyeEllipse 회전각·반경을 정규화 좌표에서 산출, 셰이더 타원 마스크는 aspect 보정 없이 회전 적용 — head roll 시 전단 왜곡** — `cpp/src/gpu/gpu_lens_renderer.cpp:672-705 (+ cpp/src/gpu/shader_sources.cpp:964-973)`
  - 영향: 머리를 기울인(roll) 얼굴에서 눈꺼풀 aperture 타원이 실제 눈 형상 대비 전단·과소/과대 회전되어 렌즈 가림 마스크가 부정확. roll 각이 클수록, 프레임이 정사각형에서 멀수록 오차 증가. 데모 Kotlin CameraGLRenderer의 중복 구현도 동일 계열.
  - 수정안: fitEyeEllipse에서 좌표를 aspect 보정 공간((x·W/H, y) 등)으로 변환 후 중심/회전/반경을 피팅하고, 셰이더도 동일 공간에서 타원 평가(applyLens의 adjustedCoord 방식과 통일).
- **Eye Refiner가 iris_landmark의 픽셀 좌표(0~64) 출력을 정규화 없이 ROI 비율로 사용 — 활성화 시 홍채 좌표 붕괴** — `cpp/src/mediapipe_detector.cpp:2066-2073, 2110-2117 (대조: 3019-3029)`
  - 영향: EyeRefinerPolicy를 Never 외로 설정하면 refined 홍채 좌표가 화면 밖(crop.x + 수십×crop.width)으로 폭주하여 V2 coarse 좌표를 덮어씀. 사용자 메모리의 'iris_landmark 2차 추론은 오히려 품질 저하' 관측과 부합. 현재 기본 정책 Never라 휴면이지만 Refiner 기능 자체가 사용 불가 상태.
  - 수정안: runEyeRefiner의 좌/우 모두 V1 경로와 동일하게 `local_x = refined[i*3+0] / IRIS_LANDMARK_INPUT_WIDTH`, `local_y = refined[i*3+1] / IRIS_LANDMARK_INPUT_HEIGHT` 정규화 후 (오른눈은 그 다음 1-local_x) ROI 역변환 적용.
- **shouldRunEyeRefiner가 아직 설정되지 않은 result.confidence(항상 0.0f)를 판단 기준으로 사용 — Conditional 정책이 Always로 오동작** — `cpp/src/mediapipe_detector.cpp:2910 (confidence 설정: 3148-3156)`
  - 영향: Conditional 정책이 사실상 Always와 동일하게 동작 — 매 프레임 2회의 iris_landmark 추론 추가(성능)이며, finding #1과 결합 시 Conditional 활성화만으로 매 프레임 홍채 좌표가 파괴됨.
  - 수정안: refiner 판단 전에 `float current_conf = face_confidence * (left_detected && right_detected ? 1.0f : 0.5f);`를 계산해 전달하거나, confidence 산출(§6)을 refiner 분기 앞으로 이동.
- **Face Detection 입력 정규화가 [0,1] — canonical BlazeFace short range는 zero-center [-1,1] 요구** — `cpp/src/mediapipe_detector.cpp:1049 (소비: 2489-2492)`
  - 영향: 검출 score 저하와 박스 회귀 정확도 저하(입력 분포 이동). min_detection_confidence=0.5 경계 근처에서 검출 플리커·원거리 얼굴 미검출 증가 가능. finding #4(디코딩 순서)와 함께 face detection 단계의 누적 부정확 요인.
  - 수정안: face detection 입력 경로만 `(v/255)*2-1` 스케일 적용(convertTo(alpha=2.0/255.0, beta=-1.0)) 후 score/박스 안정성 A/B 실측.
- **BlazeFace 박스 디코딩 좌표 순서가 canonical과 반대로 보임 ([y,x,h,w]로 해석, canonical은 reverse_output_order=true → [x,y,w,h])** — `cpp/src/mediapipe_detector.cpp:1665-1678`
  - 영향: 첫 프레임/재검출 프레임에서 얼굴 박스의 중심 오프셋과 종횡비가 뒤바뀜 — 화면 중앙·정면 얼굴에서는 오차가 작아 동작하지만, 비중앙·기울어진 얼굴에서 초기 crop이 어긋나 랜드마크 실패→재검출 루프 또는 첫 프레임 좌표 부정확의 원인이 됨.
  - 수정안: 디코딩을 [0]=x_center, [1]=y_center, [2]=w, [3]=h로 교정하고, 고정 입력 이미지(비중앙 얼굴)로 face_rect 골든 값 회귀 테스트 추가. 모델 파일의 실제 출력 순서를 1회 실측(키포인트 4-15와 눈 위치 비교)으로 확정할 것.
- **TemporalStabilizer outlier rejection이 기본 설정에서 완전 무동작(no-op)** — `cpp/src/temporal_stabilizer.cpp:255-265 (+ cpp/include/iris_sdk/temporal_stabilizer.h:39)`
  - 영향: 단일 프레임 스파이크 거부 기능이 프로덕션 기본값에서 전혀 동작하지 않음. 검출 노이즈 스파이크가 OneEuroFilter에 그대로 유입되어 렌즈 위치 순간 점프 발생 가능. 헤더 주석 "1프레임만 확인 (지연 최소화)"이 의도를 잘못 기술해 후속 작업자가 기능이 살아있다고 오인.
  - 수정안: 비교를 `<=`로 바꾸거나(1프레임 거부 의도 유지) 기본값을 2로 변경. 어느 쪽이든 기본 config에서 SingleFrameSpikeRejected가 통과하도록 테스트를 기본값 기준으로 재작성.
- **computeEAR가 정규화 좌표로 계산되어 블링크 임계값 0.2의 의미가 프레임 종횡비에 종속** — `cpp/src/temporal_stabilizer.cpp:323-338 (+ temporal_stabilizer.h:42)`
  - 영향: 4:3 가로 버퍼(×1.33)에서는 우연히 동작하지만, 세로 입력(3:4, ×0.75)에서는 눈을 뜬 상태의 EAR_norm(≈0.19~0.23)이 임계 0.2 부근에 걸려 상시/간헐 블링크 오판. 블링크 판정 시 stabilize()가 홍채 스무딩을 건너뛰고 last_valid 위치를 hold하므로(:175-186), 오판이 지속되면 홍채 좌표가 과거 값에 고정되는 동작 결함으로 직결. 랜드마크 주입 아키텍처에서 입력 프레임 종횡비가 다양해지면 위험 증가.
  - 수정안: EAR 계산 시 frame_width/frame_height(IrisResult에 이미 존재)로 픽셀 환산 후 거리 계산: dx_px=Δx·W, dy_px=Δy·H. 또는 종횡비 보정 계수(W/H)를 수직거리에 곱해 픽셀 공간과 등가화.
- **GridMesh RBF 보간 수학 결함 — zero-변위 컨트롤 포인트 제외로 거리 감쇠 소실 + 임계 절벽 + 정규화 공간 비등방 sigma** — `cpp/src/warp/grid_mesh.cpp:374-433 (+ grid_mesh.h:118)`
  - 영향: 워프(슬림페이스/V라인/눈 확대) 변위장이 컨트롤 포인트에서 멀어져도 감쇠하지 않고 임계 경계에서 불연속 — 의도한 '자연스러운 falloff'가 수학적으로 성립하지 않음. 현재 프로덕션 파이프라인에는 미연결(테스트 전용)이라 즉시 사용자 영향은 없으나, 이펙트 코어로 이식 시 그대로 결함 이전.
  - 수정안: zero-변위 컨트롤 포인트를 가중평균에 포함(자연스러운 0 anchor)하거나, 정규화 대신 비정규 RBF 합(또는 가중치에 거리 감쇠 envelope 곱)을 사용. 거리 계산은 픽셀 환산(setControlPoints가 이미 받는 image_width/height 활용 — 현재 검증 외 미사용) 후 수행.
- **468/478 랜드마크 수 불일치 — iris center 컨트롤 포인트 등록 조용한 실패 + applyWarp의 잠재 OOB 읽기** — `cpp/src/warp/grid_mesh.cpp:82 (+ face_warp_controller.cpp:284, 471-475, addControlPoints:314)`
  - 영향: (1) 홍채 중심을 보간 anchor로 쓰려던 의도가 침묵 실패 — 눈 확대 시 동공 중심 고정이 보장되지 않음. (2) 468 배열 전달 시 UB(OOB read). (3) '코어가 478점을 주입받는' 전환 목표에서 코어 내부 모듈이 468을 하드코딩하고 있어 계약 정리 필수 지점.
  - 수정안: landmark_to_vertex_를 478로 확장하고 applyWarp에 landmark_count 파라미터 추가 + 최소 474 검증. grid_mesh.h:176 주석의 '468개' 표기를 478 표준으로 통일.

### 스레드/수명 (4건)

- **데모: glIrisResult/uiIrisResult 단일 인스턴스를 Analyzer 스레드가 매 프레임 변경하면서 GL/UI 스레드가 동시 읽음** — `android/demo-app/src/main/java/com/irislenssdk/demo/GpuRenderActivity.kt:153-155, 998-999, 1026-1034`
  - 영향: 오버레이/렌즈 게이팅(`irisResult?.detected`)과 좌표가 서로 다른 프레임 값으로 섞여 시각적 지터·간헐적 오동작. '스냅샷'이라는 주석과 실제 동작 불일치로 유지보수 함정.
  - 수정안: queueEvent/runOnUiThread 람다 내부에서 복사하거나, 전달 시마다 새 IrisResult 복사본(또는 불변 DTO)을 생성해 넘긴다. 또는 SDK의 DetectionSlot처럼 GL 스레드 측에서 copy-in.
- **DetectionSlot 더블버퍼: GL 스레드가 보유한 슬롯 포인터를 writer가 wrap-around로 덮어씀 — generation 검증 수단 부재로 torn read** — `android/iris-sdk/src/main/cpp/iris_jni.cpp:559-568, 1870-1924`
  - 영향: 렌더 1프레임이 추론 1프레임보다 오래 걸리는 순간(뷰티+렌즈 풀파이프) landmark 좌표가 프레임 간 혼합 → 렌즈/마스크 지터·찢김. 메모리 안전성은 유지(정적 슬롯)되나 결과 정합성 결함. 랜드마크 주입이 코어의 유일한 입력이 되는 전환 후 아키텍처에서는 이 채널이 핵심 경로이므로 전환 설계에서 최우선 보강 대상.
  - 수정안: (1) seqlock 패턴: 읽기 전후 generation 비교 가능하도록 `nativeGetDetectionSlotGeneration()` 추가 + 사용 후 재검증, 또는 (2) GL 스레드 스택/전용 버퍼로 슬롯을 통째로 복사해 반환하는 JNI(nativeCopyDetectionSlot) 제공, 또는 (3) 슬롯 3개 이상의 triple buffering으로 reader 보유 슬롯 재사용 금지.
- **detectSync 타임아웃 후 상태기계 영구 고착(ResultReady) + 호출자 버퍼 무락(raw pointer) 사용으로 경합/UAF 가능** — `cpp/src/inference_thread.cpp:153-191, 343-355, 367-372 (+ cpp/src/frame_processor.cpp:655-659)`
  - 영향: 단 한 번의 >5s 추론(저사양 기기 콜드스타트, GPU delegate 셰이더 컴파일 지연 등) 이후 동기 검출이 재초기화 전까지 영구 실패. 해상도 전환과 겹치면 메모리 안전성 위반. 데모 앱의 기본 경로(detectWithRotation→detectSync)가 직접 영향권.
  - 수정안: 요청에 세대 번호(sequence)를 부여해 워커가 완료 시 '현재 세대와 일치할 때만' ResultReady로 전이하고 불일치 시 Idle로 환원. 동기 경로도 비동기처럼 입력 딥카피. 타임아웃 시 워커 완료를 기다리는 drain 플래그 추가.
- **stop()이 Starting 중에 호출되면 threadLoop가 Stopping을 Idle로 덮어써 join 영구 대기(행)** — `cpp/src/inference_thread.cpp:62, 312, 322 (+ cpp/src/frame_processor.cpp:190-192)`
  - 영향: 초기화가 10초를 초과하는 저사양/GPU 컴파일 지연 기기에서 iris_sdk_init 호출 스레드(보통 메인 스레드)가 영구 블로킹 → ANR. 발생 조건은 드물지만 코드상 결정적.
  - 수정안: threadLoop의 초기화 완료 전이를 `ThreadState expected = Starting; state_.compare_exchange_strong(expected, Idle)`로 바꾸고 실패(=Stopping) 시 즉시 정리 후 종료.

### GL/GPU 상태 (12건)

- **GLESRenderContext 자체 EGL 모드: 호출 스레드 바인딩 강탈 + eglTerminate(EGL_DEFAULT_DISPLAY) + 실패 시 임의 컨텍스트에서 삭제** — `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/cpp/src/gpu/gles_render_context.cpp:56-72, 153, 165-182, 412-432`
  - 영향: 현재는 휴면. 그러나 아키텍처 전환에서 '코어 자체 컨텍스트' 옵션을 켜는 순간 호스트 렌더링 중단·교차 컨텍스트 리소스 오염 가능.
  - 수정안: makeCurrent 계열에서 기존 바인딩(eglGetCurrentContext/Surface) 저장 후 복원하는 scoped-current 패턴 도입, eglMakeCurrent 반환값 검사 후 실패 시 GL 호출 중단, eglTerminate는 자체 생성 리소스(context/surface)만 파괴하고 디스플레이 terminate는 옵션화.
- **applyTexture(TextureHandle) 경로 ping/pong 풀 반환 전무 — 4회 호출 후 풀 영구 고갈 + 댕글링 핸들 노출** — `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/cpp/src/gpu/gpu_beauty_backend.cpp:734-832`
  - 영향: 이 경로가 활성화되는 즉시(아키텍처 전환에서 TextureHandle 기반 zero-copy 표면을 쓰면) 4프레임 만에 beauty 파이프라인 전체가 실패하고, trim과 결합 시 해제된 메모리 역참조 가능.
  - 수정안: applyTextureId와 동일한 fence+이월 반환 패턴 적용(또는 applyTexture를 applyTextureId 위임으로 재구현). output 핸들은 포인터 노출 대신 GLuint 값 복사 기반으로 변경. 전환 설계 시 IBeautyBackend::applyTexture 계약(출력 텍스처 소유권/수명)을 명문화.
- **호스트 컨텍스트 차용 모드의 'current' 가정이 완전히 암묵적 + context loss 시 stale GL id 삭제를 새 컨텍스트에서 실행** — `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/cpp/src/gpu/gpu_beauty_backend.cpp:151-155, 562-654`
  - 영향: 컨텍스트 재생성 시나리오(백그라운드 복귀, 권한 모달)에서 동작이 호출 순서 운에 의존. 호스트 GL 객체 오삭제 잠재 위험 + 잘못된 스레드 호출 시 GL 리소스 정리 누락. '랜드마크 주입형 렌더 코어' 전환 시 호스트 다양성이 커지므로 우선 정리 대상.
  - 수정안: (1) initialize/applyTextureId/release 진입부에 eglGetCurrentContext() != EGL_NO_CONTEXT 검증 + 컨텍스트 식별자(EGLContext 값) 기록. (2) 기록된 컨텍스트와 현재 컨텍스트가 다르면 GL 삭제를 건너뛰고 id만 폐기하는 'context generation' 가드 도입. (3) 코어에 onContextLost()/onContextRecreated() 공식 API를 추가하고 데모의 release+init 우회를 대체.
- **호스트 GL 상태 오염 — viewport/blend/depth/scissor/texture unit/VAO0 attrib/UNPACK_ALIGNMENT/program 미복원** — `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/cpp/src/gpu/gpu_beauty_backend.cpp:2093-2095, 1860-1898, 1104-1131, 1168-1185`
  - 영향: 현재 데모는 매 드로우마다 자체 상태를 재설정해 증상이 가려져 있으나, 임의 호스트 앱에 임베드되면 블렌딩/뎁스/유닛 바인딩/정점 attrib가 오염되어 호스트 렌더링이 깨질 수 있다. SDK 전환 목표(호스트 컨텍스트 차용 렌더 코어)와 직접 충돌.
  - 수정안: applyTextureId/renderToTexture 경계에 GL 상태 save/restore 스코프 도입(최소: viewport, BLEND/DEPTH/SCISSOR enable 비트, active texture unit, unit0-3 바인딩, array buffer/VAO, current program, UNPACK_ALIGNMENT). skin mask fan은 클라이언트 배열 대신 전용 VBO+VAO 사용. executeCombinedColorPass cleanup은 unit0로 복귀 후 unbind하도록 수정.
- **TexturePool 1920x1080 하드코딩 상한 — portrait/고해상도 입력 시 파이프라인 전체가 매 프레임 silent 실패** — `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/cpp/src/gpu/gpu_beauty_backend.cpp:171 (및 gpu_lens_renderer.cpp:137, texture_pool.cpp:88-93)`
  - 영향: 호스트가 pre-rotate된 portrait 텍스처나 1080p 초과 프레임을 넘기는 즉시 효과 전체가 조용히 꺼진다(에러 로그만). 코어를 범용 렌더 엔진으로 전환할 때 입력 계약 결함.
  - 수정안: 초기화 시 상한을 입력 프레임 크기 기반으로 동적 설정(또는 max(width,height)x max(width,height) 허용), 최소한 sdk_api.h에 상한 계약을 문서화하고 거부 시 전용 에러 코드 반환.
- **TexturePool '더 큰 텍스처 재사용'과 호출부의 정확한 크기 가정 충돌 — 해상도 전환 시 출력 왜곡** — `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/cpp/src/gpu/texture_pool.cpp:388-404`
  - 영향: 런타임 해상도 변경(전/후면 카메라 전환, 프리뷰 크기 변경) 직후 beauty/lens 출력이 축소·왜곡되거나 쓰레기 픽셀 표시. 정상 크기 텍스처가 새로 생성될 때까지가 아니라, 큰 텍스처가 trim될 때까지 지속.
  - 수정안: (택1) findAvailable에서 정확한 크기 매칭만 허용하고 불일치 유휴 텍스처는 재할당(glTexImage2D로 리사이즈), 또는 TextureInfo에 논리 크기(viewport)와 물리 크기를 분리하고 소비자가 UV 스케일을 적용. 전환 설계에서는 전자가 단순.
- **공유 IrisResult 인스턴스를 스레드 간 가변 상태로 재사용 — '불변 스냅샷' 주석과 달리 GL/UI 스레드 읽기 중 analyzer 스레드가 동시 덮어씀** — `android/demo-app/src/main/java/com/irislenssdk/demo/GpuRenderActivity.kt:153-155, 997-999, 1026-1033`
  - 영향: 프레임 단위로 좌/우 눈·faceMesh 절반이 서로 다른 검출 프레임에서 섞인 찢어진(torn) 랜드마크가 셰이더 uniform으로 들어감 → 비결정적 렌즈/눈꺼풀 글리치. 실기기 육안 벤치(이 프로젝트의 핵심 검증 방법)의 신뢰도를 직접 훼손. 랜드마크 주입형 전환 시 주입 데이터 무결성 계약의 반례.
  - 수정안: queueEvent 클로저에서 새 IrisResult를 생성해 복사본을 넘기거나(할당 회피가 필요하면 GL 스레드 소유 인스턴스에 queueEvent 내부에서 copyFrom 수행: `queueEvent { glOwned.copyFrom(snapshot) }` 형태로 복사 시점을 GL 스레드로 이동), 더블 버퍼 + seq-lock 패턴 적용. uiIrisResult도 동일 처리.
- **GL 컨텍스트 수명 계약 부재: 컨텍스트 손실 시 SDK 통지 API 없음, releaseGpuBeauty가 main 스레드(컨텍스트 없음)에서 호출, detach 경로 정리 미보장** — `android/demo-app/src/main/java/com/irislenssdk/demo/GpuRenderActivity.kt:1287-1307 (+ CameraGLRenderer.kt:688-698, CameraGLView.kt:430-447, cpp/src/gpu/texture_pool.cpp:54-77)`
  - 영향: GPU 객체 정리가 '우연히 동작'하는 상태. 컨텍스트 재생성 시 stale 핸들 → glError 0x501 회귀(실제 발생 이력), main 스레드 glDelete는 무음 무시 → 네이티브 풀 상태와 GL 실체 불일치. 코어를 렌더 전담 라이브러리로 전환하면 호스트 앱의 GL 수명과의 계약이 1급 API가 되어야 하는데 현재 그 계약이 주석+우회로만 존재.
  - 수정안: SDK surface에 iris_sdk_notify_gl_context_lost() (GL 삭제 호출 없이 핸들만 무효화) 추가, release 계열에 '현재 GL 컨텍스트 필수' 계약 명문화 + eglGetCurrentContext() 가드. 데모는 releaseGpuBeauty/releaseGpuLens를 모두 GL 스레드 정리 경로로 이동하고 onDetachedFromWindow에서 super 호출 전 정리 수행 또는 setPreserveEGLContextOnPause(true) 채택.
- **GpuRenderActivity.yuvToNv21이 rowStride/패딩 미처리 — FrameAnalyzer의 동일 변환과 달리 stride 가정 위반 기기에서 검출 입력 오염** — `android/demo-app/src/main/java/com/irislenssdk/demo/GpuRenderActivity.kt:1047-1088`
  - 영향: rowStride≠width인 기기/해상도(특히 1080폭)에서 NV21이 행마다 어긋나 detectWithRotation 입력이 오염 → 랜드마크 품질 저하·검출 실패가 '추적기 문제'로 오진될 수 있음. 현재 테스트 기기에서 stride가 우연히 일치하면 잠복.
  - 수정안: FrameAnalyzer.imageProxyToNV21과 동일한 stride-aware 구현으로 교체(또는 해당 함수를 공용 유틸로 추출해 양쪽에서 사용).
- **KT fallback 렌즈 셰이더와 SDK 셰이더의 blendMode 의미 불일치(ID 2/3/4/5/6/7) + 실패 시 프레임 단위 무음 폴백** — `android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLRenderer.kt:154-440, 1234-1267 (+ cpp/src/gpu/shader_sources.cpp:943-955, 1114-1133)`
  - 영향: 동일 blendMode 값이 경로에 따라 전혀 다른 시각 결과를 냄. SDK 간헐 실패 시 프레임 단위로 두 알고리즘이 섞여 깜빡임 + 실기기 육안 벤치에서 '지금 보는 화면이 SDK 결과인지' 보증 불가(성공 로그는 최초 1회뿐, 1257-1260). 추적 분리 전환 후에도 fallback이 남으면 코어 검증 통로 오염 지속. (w9-demo-ui-sync 메모리의 잔여 이슈와 일치 — 여전히 미해결임을 코드로 확인)
  - 수정안: KT fallback 셰이더를 삭제하고 SDK 실패 시 '렌즈 미적용 + 화면 경고 오버레이'로 명시적 실패 처리. spinner는 SDK가 실제 지원하는 {0,1,2,5,7}만 SDK 명칭으로 노출. 폴백을 유지해야 한다면 매 프레임 활성 렌더러를 HUD에 표시.
- **EGL 컨텍스트 재생성 시 이전 SurfaceTexture/Surface 미해제 + isGLInitialized 미리셋 — resume 시 stale Surface가 CameraX에 제공되는 레이스** — `android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLView.kt:44-48, 125-136, 144-156 (+ CameraGLRenderer.kt:683-686)`
  - 영향: 타이밍에 따라 resume 후 프리뷰 정지/블랙(새 컨텍스트의 updateTexImage가 프레임을 영원히 못 받음). 매 재생성마다 SurfaceTexture BufferQueue 네이티브 리소스 누수(GC 파이널라이저 의존).
  - 수정안: onSurfaceTextureAvailable 시 이전 cameraSurface.release()/cameraSurfaceTexture.release() 수행, GLSurfaceView surfaceDestroyed 시점에 isGLInitialized=false 리셋, 새 SurfaceTexture 생성 시 진행 중 SurfaceRequest에 invalidate/재제공 경로 추가 (CameraX request.invalidate() 활용).
- **Detection Slot 더블버퍼의 generation 검증이 어디에도 구현되지 않음 — 주석상 '호출측 수행'이지만 generation이 Java에 미노출이라 이행 불가능한 계약** — `android/iris-sdk/src/main/cpp/iris_jni.cpp:559-566, 1881-1894, 1898-1923, 2186-2190`
  - 영향: GL 렌더 도중 IrisResult(faceMesh 478×3 float 포함)가 부분 덮어쓰기되어 찢어진 좌표가 SDK 렌즈/뷰티 패스에 들어감 — 비결정적 글리치. 정적 슬롯이라 메모리 안전성 문제는 아니나 C++ 기준 형식적 UB. '랜드마크 주입' 아키텍처의 직계 전신이 될 채널의 계약 결함.
  - 수정안: JNI 진입 직후 슬롯 내용을 스택 로컬 IrisResult로 복사하고 복사 전후 generation 일치(짝수/동일) 확인 후 재시도하는 seq-lock 검증을 nativeRenderLensTexture/nativeApplyBeautyFilterTextureV2 내부에서 수행. raw 포인터를 Java로 반환하는 API 자체를 폐기하고 '슬롯 핸들 + 내부 복사' 형태로 재설계 (지금 수정하지 말 것).

### 결합도 (3건)

- **DetectionSlot 더블버퍼 torn-read: GL 스레드가 렌더 중인 슬롯을 Analyzer 스레드가 덮어쓸 수 있음 (generation 검증 수단 자체가 미노출)** — `android/iris-sdk/src/main/cpp/iris_jni.cpp:1881-1894, 1906-1924`
  - 영향: detect가 GL 렌더보다 빨라지는 순간(GL 부하 스파이크 등) 프레임 내 좌표 일관성이 깨져 렌즈/마스크 글리치 발생 가능. 이 슬롯을 정식 랜드마크 주입 경계로 승격하면(2단계 계획) 결함도 함께 승격됨
  - 수정안: 지금 수정하지 말 것. 2단계에서 승격 시: (a) 슬롯 3개 순환 + reader가 acquire한 슬롯 인덱스를 점유 표시, 또는 (b) generation을 렌더 진입/종료 시 비교해 불일치 시 직전 스냅샷 사용, 또는 (c) GL 스레드 로컬 복사(7.7KB memcpy는 프레임당 무시 가능) 중 택일
- **Strategy 패턴 붕괴: 팩토리 전체가 nullptr 반환하는 죽은 코드, 파이프라인은 MediaPipeDetector 구체 타입에 하드 의존** — `cpp/src/iris_detector.cpp:13-31 (+frame_processor.cpp:129,189-199, inference_thread.h:233, sdk_manager.cpp:150-176)`
  - 영향: IrisDetector 추상이 장식으로 전락 — 검출기 교체가 코드상 불가능하고, 추적 분리 시 '인터페이스만 갈아끼우면 된다'는 가정이 성립하지 않는다. 죽은 팩토리는 외부 사용자에게 항상 실패하는 공개 API로 노출 중
  - 수정안: 아키텍처 전환 시 IrisDetector/createDetector/SDKManager::createDetector를 제거 대상에 포함시키고(외부 주입이면 추상 자체가 불필요), 전환 전이라면 팩토리에서 MediaPipeDetector를 실제 반환하거나 deprecated 처리
- **C/C++ IrisResult 이중 정의를 reinterpret_cast로 교환하면서 가드는 sizeof 하나뿐 — 필드 순서/타입 어긋남은 통과** — `cpp/src/sdk_api_v2.cpp:39-42 (+243, 361, 377, 472, 666)`
  - 영향: 향후 경계 타입 변경(랜드마크 주입 API 도입이 정확히 이 지점을 건드림) 때 UB성 데이터 오염이 조용히 발생할 수 있는 구조적 함정. 실제 P7-W2에서 이미 한 차례 필드 추가가 있었음
  - 수정안: 지금 수정하지 말 것. 2단계에서: 단일 C 정의를 양쪽이 공유(C++은 using 또는 상속 없는 포함)하거나, 최소한 주요 필드별 offsetof static_assert(detected/face_mesh/timestamp_ms/avg_iris_luma_* 등)를 추가

### 플랫폼 패리티 (9건)

- **internal C API 9종이 어떤 헤더에도 선언 없이 JNI에 수동 extern 중복 선언 — 시그니처 드리프트 시 컴파일 검출 불가** — `android/iris-sdk/src/main/cpp/iris_jni.cpp:2238, 2278-2281, 2344-2348 (정의: cpp/src/sdk_api_v2.cpp:713-845)`
  - 영향: 의도된 internal 경로라도 단일 진실 소스가 없어, 파라미터 타입/순서 변경이 침묵 손상으로 이어질 수 있는 구조적 유지보수 결함. 함수 수가 9개로 이미 임계 초과.
  - 수정안: sdk_api_internal.h(비공개 배포) 헤더를 만들어 양쪽이 동일 선언을 include하게 변경. 전환 시 이 internal 함수들의 공개 승격/폐기 여부를 함께 결정.
- **JNI 경계에서 홍채 boundary 랜드마크 [1..4]와 visibility가 구조적으로 소실 — 랜드마크 주입 전환의 직접 장애 요소** — `android/iris-sdk/src/main/cpp/iris_jni.cpp:244-246, 327-336 (소비처: cpp/src/lens_renderer.cpp:132-133)`
  - 영향: '외부 추적기 → 478점 주입' 전환에서 이 JNI 왕복이 바로 주입 표면인데, 5점 홍채 표현과 visibility를 운반할 수 없는 손실 채널이다. 전환 시 주입 데이터 스키마를 그대로 쓰면 CPU 경로 시각 결함이 활성화된다.
  - 수정안: Java IrisResult에 boundary 4점(또는 float[] 배열)과 visibility를 추가해 왕복 무손실화하거나, 전환 시 주입 스키마를 face_mesh 478점 + 홍채 5점 + visibility 포함 POD로 재정의.
- **v2 표면 에러 코드 500/501이 Java 상수·Kotlin 예외 매핑·error_to_string 전부에서 누락** — `android/iris-sdk/src/main/java/com/irislenssdk/kotlin/IrisException.kt:137-158 (또한 cpp/include/iris_sdk/sdk_api.h:70-76, cpp/src/sdk_api.cpp:810-840, IrisLensSDK.java:641)`
  - 영향: Kotlin/Java 앱이 GPU beauty·lens 경로의 '미초기화' 에러를 식별·복구할 수 없고, 로그 문자열도 UNKNOWN으로 찍혀 진단이 오도된다.
  - 수정안: 500/501을 IrisSdkError 정식 멤버로 문서화하고 iris_sdk_error_to_string에 case 추가, Java에 NOT_SUPPORTED=500/상수 추가(또는 501을 100으로 통합), IrisException.fromErrorCode에 분기 추가. 랜드마크 주입 아키텍처 전환 시 에러 코드 표를 단일 정본으로 재정의.
- **Kotlin 표면 커버리지 결손: 래퍼는 detect/process/config만, IrisResultKt에 faceMesh(478점)·avgIrisLuma 부재** — `android/iris-sdk/src/main/java/com/irislenssdk/kotlin/IrisLensSDKKt.kt:89-399 (IrisResultKt.kt:107-125)`
  - 영향: Kotlin-first 앱은 사실상 Java 클래스로 우회해야 하며, 478 랜드마크 중심 아키텍처로 가면 Kotlin 결과 타입이 핵심 데이터를 운반하지 못한다. 표면 4개(C/JNI/Java/Kotlin) 중 Kotlin만 2세대 뒤처진 상태로 이중 유지보수 부채.
  - 수정안: 전환 시 Kotlin 래퍼를 '결과 주입+이펙트 제어' 신표면 기준으로 재생성(자동 생성 검토)하거나, 유지 불가면 Kotlin 래퍼를 deprecated로 명시하고 Java 단일 표면 선언.
- **C API가 3개 표면으로 분산: sdk_api.h + beauty_filter.h(V2 config 구조체 이중화) + 헤더 미선언 internal 9종** — `cpp/include/iris_sdk/beauty_filter.h:24-320 (대조: sdk_api.h:637-677, sdk_api_v2.cpp:62-137)`
  - 영향: 필드 추가 시 최소 5곳(두 C struct, 변환 2함수, 기본값 2함수, Java, JNI 캐시) 수동 동기화 필요. feather_radius dead field(별도 finding)가 이미 이 비용의 실증 사례. 이펙트 전담 코어 전환 시 어느 표면을 이식할지 모호.
  - 수정안: 전환 1단계에서 IrisBeautyConfigV2(sdk_api.h)를 정본으로 선언, beauty_filter.h V2 표면 8종 + BeautyFilterConfigV2 C struct를 deprecated 처리 후 제거. JNI가 쓰는 경로(_c 계열)만 유지.
- **C API ↔ JNI 전수 대조: sdk_api.h 50종 중 13종 미노출(비동기 API 전체 포함), JNI는 자체 detection slot 메커니즘으로 대체** — `cpp/include/iris_sdk/sdk_api.h:1030-1092 (slot 대체 구현: iris_jni.cpp:1871-1929)`
  - 영향: 공식 C API의 비동기 표면이 Android에서 죽은 코드이고, 실제 '랜드마크 주입' 경로는 헤더에 없는 JNI 사설 메커니즘이다. 추적 분리 아키텍처 전환의 주입 표면 설계가 바로 이 지점인데, 현재 정본이 없다. eye refiner 정책·stabilizer 튜닝도 Java/Kotlin에서 제어 불가.
  - 수정안: 전환 설계에서 '랜드마크 주입 C API'를 정본으로 신설(슬롯 메커니즘을 C API로 승격하거나 submit/get_latest 계열로 통합)하고, 미노출 13종은 노출 또는 제거를 명시 결정. 죽는 함수는 헤더에서 삭제.
- **V1 BeautyFilterConfig 기본값 불일치: C++ smoothing=0.5/softFocus=0.3 vs Java 0.0/0.0** — `cpp/src/beauty_filter.cpp:38-42 (대조: BeautyFilterConfig.java:89-97)`
  - 영향: 어느 생성 경로를 쓰느냐에 따라 V1 뷰티 효과 강도가 달라지는 잘못된 결과. 플랫폼 추가 시 기본값 복제 드리프트의 전형.
  - 수정안: Java 상수를 C++ 값과 일치시키거나(또는 역방향), 기본값을 C 측 default 함수 단일 소스로 강제하고 Java 생성자가 native 호출로 채우게 변경.
- **동일 C++ ErrorCode::NotInitialized가 v1 표면에서는 100, v2 표면에서는 501로 이중 변환 — 에러 코드 정본 부재** — `cpp/src/sdk_api_v2.cpp:690 (대조: cpp/src/sdk_api.cpp:81-82)`
  - 영향: 바인딩/앱 레벨에서 '미초기화'를 단일 코드로 검사할 수 없음. finding 1과 결합해 v2 경로의 미초기화 에러가 사실상 분류 불가가 된다.
  - 수정안: NotInitialized → 100 단일 매핑으로 통일하고 501은 제거(또는 deprecated 별칭). 전환 1단계에서 에러 변환 함수를 한 곳(공유 유틸)으로 모은다.
- **BeautyFilterConfigV2 '기본값'이 3원 분기: C POD 기본 enabled=1 vs C++ Helper/Java 기본 enabled=false** — `cpp/src/sdk_api_v2.cpp:166 (대조: cpp/include/iris_sdk/beauty_filter.h:441, BeautyFilterConfigV2.java:217)`
  - 영향: 기본 설정 획득 경로에 따라 필터가 켜지거나 꺼지는 비결정적 사용자 경험. 문서·테스트가 어느 쪽을 정본으로 보느냐에 따라 회귀 판정도 갈린다.
  - 수정안: 세 곳의 enabled 기본값을 false(opt-in)로 통일하고, _c 함수가 Helper::defaults()를 fromCppConfigV2로 변환해 반환하도록 단일 소스화.

### 테스트 안전망 (7건)

- **Android 계층(JNI/Java/Kotlin) 테스트 소스 전무 — faceMesh 478*3 마샬링 무보호** — `android/iris-sdk/src/main/cpp/iris_jni.cpp:283-300`
  - 영향: 랜드마크 주입 전환 시 JNI 시그니처가 가장 크게 바뀌는 경계인데(좌표 배열 방향이 SDK→앱에서 앱→SDK로 역전), 배열 길이·스트라이드·게이팅 회귀를 잡을 테스트가 0. 위험 모듈 1순위.
  - 수정안: Robolectric 기반 IrisResult 마샬링 왕복 단위 테스트 + 라이브러리 로드/버전 확인 androidTest 최소 1본 신설.
- **CI 전무 + 테스트 스위트 정기 실행 안 됨 정황 (바이너리 2개 부재, 대부분 2개월 전 빌드)** — `cpp/cmake-build-debug/bin`
  - 영향: '테스트는 존재하나 실행되지 않는' 상태. 전환처럼 광범위한 변경에서 회귀가 누적돼도 감지가 사후적·선택적이 된다. 이미 2개 타겟은 빌드조차 안 되는 상태일 가능성.
  - 수정안: 최소 1본의 GitHub Actions(macOS, BUILD_TESTS=ON, ctest --output-on-failure, 모델 의존 테스트 라벨 분리) 도입 + 머지 전 전체 ctest 의무화. test_mediapipe_detector 2종 빌드 가능 여부 우선 확인.
- **InferenceThread(스레드 수명 경로) 테스트 0건** — `cpp/src/inference_thread.cpp:24-70`
  - 영향: 추적을 코어 밖으로 빼면 InferenceThread는 해체/이동 1순위 대상. start/stop 경합, 처리 중 destroy, 이중 stop 같은 시나리오가 무보호라 전환 중 크래시 회귀를 사전에 잡을 수 없다.
  - 수정안: MockDetector를 주입한 수명 테스트(시작→submit→stop 정상 종료, busy 중 destroy, double stop 무해성) 추가 후 해체 착수.
- **V2 GPU C API 표면(sdk_api_v2.cpp 847줄, 30+ 함수) 테스트 0건** — `cpp/src/sdk_api_v2.cpp:265-836`
  - 영향: 안드로이드 실경로(JNI→V2 GPU 파이프라인)의 C 계약 전체가 무테스트. 전환 시 이 표면이 '랜드마크 주입 API'로 개편될 핵심 지점인데 회귀 기준선이 없다.
  - 수정안: GL 컨텍스트 없이 검증 가능한 부분(null/범위 인자 검증, not-initialized 에러 코드, init/release 상태 머신)만이라도 V2 계약 테스트를 신설해 개편 전 기준선 확보.
- **자기참조(mirror copy) 테스트: test_iris_luma_measure가 프로덕션 코드를 검증하지 못함 — private Impl 구조의 증거** — `cpp/tests/test_iris_luma_measure.cpp:1-45`
  - 영향: 회귀 보호 효과 0인 테스트가 골든처럼 존재. 더 중요한 건 구조 신호: mediapipe_detector.cpp 3425줄 private Impl이 순수 로직의 단위 테스트를 원천 차단 — 전환 시 이 Impl에서 빼낼 모든 로직(letterbox 역변환, luma 측정 등)의 characterization이 같은 벽에 부딪힌다.
  - 수정안: 전환 1단계에서 calculateIrisLuma·letterbox 역변환 등 순수 함수를 헤더로 추출(P8-W1 skin_mask_geometry 선례)하고 mirror 테스트를 실코드 직접 테스트로 교체.
- **렌더 출력 픽셀 검증 0건 — 모든 렌더 테스트가 return code/시간만 단정, GPU 렌더러는 테스트 자체 부재** *(blocker→major 조정)* — `cpp/tests/test_lens_renderer.cpp:297-352`
  - 영향: 전환 후 코어의 유일한 책임이 '이펙트/렌더'가 되는데, 렌더 결과의 정합을 기계적으로 검증하는 테스트가 단 한 건도 없다. 전환 전/후 출력 동등성(2단계 골든 비교)의 출발점이 현재 부재.
  - 수정안: CPU 경로(LensRenderer+CPURenderContext)에 결정적 입력→출력 픽셀 diff/해시 골든 테스트 신설. GPU 경로는 GLES dumpTexture 구현(별도 finding)과 묶어 실기기 캡처 비교 체계 구축.
- **검출 텐서 디코딩의 골든 단정이 무의미하게 느슨함 (landmark가 이미지 2배 범위면 통과)** — `cpp/tests/test_mediapipe_detector_integration.cpp:342-363, 615-617`
  - 영향: 추적 분리 후 '주입된 랜드마크 = 기존 검출 랜드마크' 동등성을 증명해야 하는데, 현 단정으로는 좌표가 수십~수백 픽셀 어긋나도 전부 통과 — 골든 기능을 전혀 못 함.
  - 수정안: 2단계에서 커밋된 shared/test_data 이미지별 478점 좌표를 JSON 골든으로 덤프(getFaceLandmarks 활용)하고 허용오차(±1px 수준) 비교 테스트로 승격.

### 2라운드 보완 감사 (11건)

- **OverlayView가 자체 confidence 게이트(0.5)·좌표 동결·2초 홀드를 적용 — GL 렌더 출력과 다른 좌표를 그리는 독립 정책** — `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/android/demo-app/src/main/java/com/irislenssdk/demo/camera/OverlayView.kt:97, 366-386, 477-487`
  - 영향: 디버그 마커(녹색 원/중심점)가 GL이 실제로 렌즈를 그리는 위치의 ground truth 역할을 하지 못한다. 측면 각도·저조도 등 confidence가 0.5 부근에서 진동하는 조건에서 마커는 동결된 과거 좌표, GL 렌즈는 현재 좌표를 그려 '렌즈가 마커에서 벗어난다'는 거짓 양성 신호를 만든다. 랜드마크 주입 아키텍처 전환 후 좌표 계약 검증 시 이 도구로는 회귀 판정이 불가능하다.
  - 수정안: 디버그 모드에서는 게이트·홀드를 모두 우회하고 GL에 전달된 것과 동일한 raw stabilized 좌표를 그리는 'GL-mirror 모드'를 추가하거나, MIN_RENDER_CONFIDENCE 게이트와 persistence 홀드를 debugMode에서 비활성화. 최소한 동결 상태(stale)임을 마커 색상으로 구분 표시.
- **consumer-rules.pro/proguard-rules.pro에 BeautyFilterConfig·BeautyFilterConfigV2 keep 규칙 전무 — minify 소비자 앱에서 JNI_OnLoad 확정 실패** — `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/android/iris-sdk/consumer-rules.pro:14-55 (keep 대상: IrisLensSDK, IrisResult, IrisResult$Iris, LensConfig, LensConfig$Builder, kotlin.** 뿐)`
  - 영향: minifyEnabled=true인 소비자 앱에서 System.loadLibrary 시 JNI_ERR → UnsatisfiedLinkError로 SDK 전체 사용 불가(배포 관점 blocker급). 동시에 Finding 1(GetFieldID 연쇄 UB)의 가장 현실적인 트리거. 현재 demo-app(minify off 추정)에서는 미발현되어 잠복 중.
  - 수정안: consumer-rules.pro에 -keep public class com.irislenssdk.BeautyFilterConfig { public *; } 및 BeautyFilterConfigV2 동일 규칙 추가. JNI에서 필드 접근하는 모든 클래스(IrisResult 포함)는 '@Keep + 필드 명시 keep'으로 격상 권장.
- **nativeNv21ToRgba — cv::cvtColor 예외 시 AndroidBitmap_unlockPixels 미호출 + C++ 예외가 JNI 경계 탈출(abort)** — `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/android/iris-sdk/src/main/cpp/iris_jni.cpp:1499-1513`
  - 영향: 매 프레임 변환 경로(주석 1428-1430: Java JPEG 경로 대체용 핫패스)에서 메모리 압박·드라이버 이상 등으로 cv::Exception 발생 시 프로세스 abort. 발생 확률은 낮으나 발생 시 복구 불능 크래시이며, 부분 실패 시에도 Bitmap lock leak.
  - 수정안: cvtColor 호출을 try { ... } catch (const cv::Exception& e) { AndroidBitmap_unlockPixels(env, bitmap); LOGE(...); return IRIS_SDK_UNKNOWN; } catch (...) { 동일 } 로 감싸기. 같은 패턴으로 unlock을 RAII 가드화 권장.
- **비원자적 모델 복사 + exists() 캐시로 잘린 .tflite 영구 고착** *(blocker→major 조정)* — `android/iris-sdk/src/main/java/com/irislenssdk/IrisLensSDK.java:1312-1318, 1335-1347`
  - 영향: 한 번 잘린 모델이 남으면 앱 데이터 삭제 전까지 모든 init이 손상 모델을 네이티브에 전달한다. 네이티브가 무검증 로드(별도 finding)하므로 영구 크래시 루프 또는 영구 기능 저하로 이어진다. 사용자 복구 수단이 사실상 없다.
  - 수정안: temp 파일에 쓴 뒤 `File.renameTo`(동일 디렉토리 내 원자적)로 마무리하고, 실패 시 temp 삭제. exists() 검사를 '존재+크기 일치(AssetFileDescriptor.getLength 또는 사전 기록한 매니페스트)'로 강화.
- **destFile.exists() 캐시로 앱 업데이트 후에도 stale 모델 영구 사용** — `android/iris-sdk/src/main/java/com/irislenssdk/IrisLensSDK.java:1312-1318`
  - 영향: SDK/앱 업데이트로 assets/models의 .tflite를 교체(정확도 개선, 버그 픽스)해도 기기에는 구버전이 영구 로드된다. 검출 품질 회귀를 배포로 고칠 수 없게 되며, 추적 외부화 전환 시 '구 모델 + 신 코어' 조합이 무한정 잔존한다.
  - 수정안: APK versionCode(또는 BuildConfig 버전)를 모델 디렉토리에 마커 파일로 기록하고 불일치 시 전체 재추출. 또는 파일 크기 비교(assets fd length vs destFile.length()) 최소 적용.
- **init/모델 추출 경로에 동기화 부재 — 동시 init·멀티프로세스 레이스** — `android/iris-sdk/src/main/java/com/irislenssdk/IrisLensSDK.java:162-183, 1296-1326`
  - 영향: 최초 실행에서 두 진입점이 동시에 init하면 한쪽이 잘린 모델로 네이티브 초기화 — MODEL_LOAD_FAILED 또는 (무검증 로드 시) 크래시. 비결정적이라 재현·디버깅이 어렵다.
  - 수정안: extractModelsFromAssets를 synchronized 블록(클래스 락)으로 감싸고, 멀티프로세스 대비가 필요하면 FileLock 사용. 원자적 rename(별도 finding)과 병행하면 프로세스 간 레이스도 '완성된 파일만 보임'으로 완화됨.
- **BeautyROI 마스크 좌표계 계약 파괴 — 래스터화는 전 프레임 공간, 구조/소비측 설계는 ROI-로컬 공간 (landmarkToMaskCoord는 정의만 있고 호출 0건)** — `cpp/src/beauty_roi_manager.cpp:77-91, 141-149, 232-234, 287-289, 363-365`
  - 영향: CPU 뷰티 경로(iris_sdk_apply_beauty_v2 → applyWithROI)에서 보호/페더 마스크가 전부 오정렬되어 눈·눈썹·입술에 스무딩/화이트닝이 침범하고 피부 경계 페더가 엉뚱한 위치에 생긴다(기존 cpu_beauty_backend major finding의 생산자측 근원). 부수 효과로 전 프레임을 ≤256×256에 담으므로 얼굴 영역 유효 마스크 해상도가 설계 의도보다 크게 낮아지고(원거리 얼굴에서는 눈/입술 폴리곤이 수 px로 붕괴해 보호가 사실상 소멸), GaussianBlur(31×31) 페더가 프레임 공간에서 축별 비등방(예: 720×1280 프레임 → 256×256 마스크에서 가로 ~42px vs 세로 ~75px)이 된다. 랜드마크 주입 아키텍처 전환 시 이 모호한 마스크-공…
  - 수정안: 마스크 공간을 하나로 확정하라. 권장: 마스크를 ROI-로컬로 통일 — 래스터화에서 (사장된) landmarkToMaskCoord를 실제로 사용해 face_rect 상대 좌표로 변환하고, GPU 소비측 셰이더는 uSkinMask 샘플링 UV를 ROI rect 기준으로 remap(uniform으로 face_rect 전달). 또는 반대로 전-프레임 공간을 공식 계약으로 선언하고 BeautyROI 주석/필드명을 갱신한 뒤 cpu_beauty_backend의 resize 대상을 actual_rect가 아닌 전체 프레임으로 수정하고, 마스크 해상도 결정(:141-149)을 프레임 종횡비 보존 다운샘플로 변경. 어느 쪽이든 lan…
- **LIPS_INDICES가 아랫입술 윤곽만 구성 — protectLips가 윗입술을 보호하지 못함** — `cpp/src/beauty_roi_manager.cpp:40-43, 350-374`
  - 영향: protectLips=true여도 윗입술이 보호되지 않아 GPU FreqSep(combined_mask 사용, gpu_beauty_backend.cpp:2253-2262)와 CPU 경로 모두에서 스무딩·화이트닝이 윗입술 질감/립 컬러를 뭉갠다. 뷰티 시뮬레이션 제품 특성상 입술 디테일 손상은 직접적인 시각 품질 결함.
  - 수정안: LIPS_INDICES를 외곽 전체 링(LIP_OUTER_INDICES 20점)으로 교체하거나, 외곽 윗입술 호를 추가해 양 입술을 닫는 폴리곤으로 수정. (입 안쪽 공동까지 보호할지는 정책 결정 필요 — 현재 LIP_OUTER 타원 경로는 입 전체를 덮음)
- **네이티브가 잘린/손상 모델을 무검증 로드 — 에러 코드가 아닌 크래시 가능** *(blocker→major 조정)* — `cpp/src/mediapipe_detector.cpp:279-304, 477`
  - 영향: 잘린 flatbuffer가 mmap된 채 InterpreterBuilder가 임의 오프셋을 역참조하면 OOB read/SIGSEGV — 깨끗한 IRIS_SDK_MODEL_LOAD_FAILED 대신 네이티브 크래시로 나타날 수 있다(파일 내용에 따라 확률적). exists() 캐시 고착과 결합 시 매 실행 크래시가 영구 반복된다. 랜드마크 주입 아키텍처 전환 후에도 외부 추적기가 동일 파일을 로드하면 같은 문제가 이전된다.
  - 수정안: BuildFromFile → VerifyAndBuildFromFile로 교체(검증 실패 시 false 반환 → 기존 IRIS_SDK_MODEL_LOAD_FAILED 경로로 자연 수렴). validateModelPath에 최소 파일 크기(>0) 검사 추가.
- **손상/stale face_landmark_v2 → V1 무음 폴백으로 478→468 랜드마크 다운그레이드** — `cpp/src/mediapipe_detector.cpp:664-694`
  - 영향: exists() 캐시에 잘린 v2가 고착되면(BuildFromFile이 클린 실패하는 경우) SDK는 영구적으로 468 랜드마크 V1로 동작한다. '코어가 478점을 주입받는' 아키텍처 전환의 핵심 계약(478점)이 호출자 모르게 깨지는 경로이며, P8 뷰티(478 기반 substrate)도 무음 회귀한다.
  - 수정안: V2 파일이 존재하는데 로드 실패한 경우를 Warning 이상으로 승격하고, model_version(또는 landmark count)을 C API로 조회 가능하게 노출. 전환 후에는 주입 시점에 랜드마크 개수를 명시적으로 검증.
- **sdk_api_v2 GPU 경로의 computeROI 결과가 미사용 — 프레임마다 마스크 5종 래스터화+erode+GaussianBlur를 수행하고 폐기 (게다가 미러링 누락 버전)** — `cpp/src/sdk_api_v2.cpp:347-380`
  - 영향: GPU 뷰티가 활성인 모든 프레임에서 OpenCV fillPoly 5회 + convexHull + erode + 31×31 GaussianBlur(≤256×256) + 마스크 합성 루프가 순수 낭비로 실행된다(33ms 프레임 버짓 잠식, roi_only 활성 시 매 프레임). 동시에 '같은 입력에 대해 두 곳에서 다른 규약(미러링 유/무)으로 computeROI를 호출'하는 형태라, 랜드마크 주입 전환 시 어느 쪽이 정본인지 혼동을 일으키는 유지보수 함정이다.
  - 수정안: sdk_api_v2.cpp:347-368의 ROI 생성 블록을 삭제(GPU 경로는 applyTextureId 내부 재계산이 정본). 전환 설계 시 미러링 책임(랜드마크 공급자 vs 코어)을 주입 API 계약에 명시.

## 4. Minor findings (요약표)

| 관점 | 제목 | 위치 | 수정안 요지 |
|---|---|---|---|
| 2라운드 보완 감사 | '별도 스냅샷' 주석과 달리 uiIrisResult는 동기화 없는 공유 가변 인스턴스 — 디버그 시각화 torn read | `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/android/demo-…` | 프레임마다 새 IrisResult를 할당해 전달하거나(풀링 시 in-flight 추적), OverlayView.setIrisResult 진입 시점에 모든 필드를 OverlayView 내부 캐시로 복사 완료하고 참조 … |
| 2라운드 보완 감사 | 동일 파일 내 좌/우 눈 컨투어 명명 모순: companion은 33그룹=왼쪽, drawEyeContours는 362그룹=왼쪽 | `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/android/demo-…` | 프로젝트 전체에서 'left=detector 좌표계(468그룹)'인지 'left=MediaPipe 해부학(473그룹)'인지 한 규약으로 통일하고, drawEyeContours의 로컬 배열을 companion 상수 재… |
| 2라운드 보완 감사 | onDraw 매 프레임 무조건 Log.d 다량 호출 — 검증 세션 중 UI 스레드 부하로 체감 fps 왜곡 | `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/android/demo-…` | 디버그 로그를 `if (debugMode)` 또는 BuildConfig.DEBUG + 주기 샘플링(예: 1초 1회)으로 게이트. |
| 2라운드 보완 감사 | [검증 완료 — 결함 아님] COVER 스케일·offset·미러 수식은 GL Cover 출력과 동치, width/height 전달도 회전 후 기… | `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/android/demo-…` | 조치 불요. 단 3중 구현 유지비 절감을 위해 아키텍처 전환 시 매핑을 단일 유틸(공유 변환 함수)로 통합 권장. |
| 2라운드 보완 감사 | 미러 시 L/R 의미 체계 불일치: GL은 left/right 스왑(screen 기준), OverlayView는 무스왑(detector 기준) … | `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/android/demo-…` | L/R 의미를 'detector 좌표계 기준(무스왑)'으로 단일 계약화하고 GL의 미러 스왑을 제거하거나(셰이더는 어차피 두 눈을 대칭 처리하므로 스왑 불필요, 단 applyLeft 게이트 위치만 주의), 스왑을 유… |
| 2라운드 보완 감사 | JniCache::init — GetFieldID 31연쇄가 pending exception 상태에서 후속 JNI 호출 지속 (JNI 명세 UB… | `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/android/iris-…` | 각 FindClass/GetFieldID 블록 직후 env->ExceptionCheck() 검사 후 즉시 ExceptionClear()하고 return false. 실패 경로에서 destroy(env) 호출로 부분 … |
| 2라운드 보완 감사 | copyResultToJava — faceMesh 478*3 복사 실패(배열 null/부족)를 silent skip하며 faceMeshValid… | `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/android/iris-…` | faceMeshArray null 또는 arrayLen 부족 시 faceMeshValid=false로 되돌리고 false 반환(또는 IRIS_SDK_INVALID_PARAM 전파). copyResultFromJava… |
| 2라운드 보완 감사 | copyResultToJava — faceMeshArray local ref 미해제 (copyResultFromJava와 비대칭) | `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/android/iris-…` | 양쪽 모두 ScopedLocalRef<jfloatArray>(env, ...) 사용으로 통일. |
| 2라운드 보완 감사 | nativeSetConfig — 첫 GetStringUTFChars OOM 실패 시 pending exception 중 두 번째 GetStrin… | `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/android/iris-…` | key 생성 후 if (!key.valid()) return ... 검사 뒤 value 생성. 또는 ScopedString 생성자에서 실패 시 ExceptionClear 정책 일원화. |
| 2라운드 보완 감사 | checkAndLogException — OutOfMemoryError 포함 모든 Java 예외를 무조건 ExceptionClear로 삼킴 | `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/android/iris-…` | Error 계열은 Clear 후 throwRuntimeException으로 재던지기 또는 최소한 예외 클래스명을 에러 메시지(iris_sdk_get_last_error 채널)로 보존. |
| 2라운드 보완 감사 | detectionPtr — Java가 전달한 jlong을 무검증 reinterpret_cast하여 역참조 | `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/android/iris-…` | detectionPtr이 &g_detection_slots[0].data 또는 [1].data와 일치하는지 검증하거나, slot index+generation을 인코딩한 불투명 핸들로 교체. |
| 2라운드 보완 감사 | throwException 계열 — 정의만 있고 호출처 0건(dead code) + FindClass 실패 시 자체 예외 처리 부재 | `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/android/iris-…` | throw* 헬퍼 4종 제거 또는 정책 주석 추가. 유지한다면 FindClass 실패 시 ExceptionDescribe 후 폴백(RuntimeException) 처리 추가. |
| 2라운드 보완 감사 | 텍스처 에셋 캐시: 파일명 충돌 + 비원자 쓰기 공유 | `android/iris-sdk/src/main/java/com/irislenssdk/IrisLensSDK.j…` | 캐시 파일명에 assetPath 전체의 해시를 포함하거나, 모델 경로처럼 메모리 디코드 후 loadTextureFromMemory 사용으로 통일. |
| 2라운드 보완 감사 | CLAUDE.md 요구 '모델 파일 암호화 옵션' 완전 부재 | `android/iris-sdk/src/main/java/com/irislenssdk/IrisLensSDK.j…` | 아키텍처 전환에서 추적(모델 로딩)이 코어 밖으로 나가면 이 요구사항의 책임 주체도 외부 추적기로 이동한다 — 전환 설계 문서에 암호화 옵션의 소유권을 명시하고, 커스텀(자체 학습) 모델 도입 시점 전까지 백로그로 … |
| 2라운드 보완 감사 | QA 모듈 4종(quality_metrics/ab_compare/release_gate/param_tuner) + profiler.cpp + b… | `cpp/CMakeLists.txt` | 4종 QA 모듈 + profiler.cpp + beauty_processor.cpp를 코어 소스 목록에서 제외하고 tests 전용 보조 타깃(예: iris_sdk_qa STATIC, BUILD_TESTS=ON 시에만… |
| 2라운드 보완 감사 | 랜드마크 주입 전환 대비 계약 결함 — 478 미만은 무조건 silent false, count 파라미터 없는 public 마스크 API는 인덱… | `cpp/src/beauty_roi_manager.cpp` | (1) computeROI 가드를 '사용 최대 인덱스+1'(467) 기준으로 완화하거나 468/478을 명시 지원하고 실패 시 로그 1회 출력. (2) 모든 public 마스크 함수에 landmark_count를 추… |
| 2라운드 보완 감사 | 눈/눈썹 인덱스: left/right 명명이 MediaPipe canonical과 반전 + 눈썹은 canonical 10점 중 8점만 사용(46… | `cpp/src/beauty_roi_manager.cpp` | 명명을 canonical 기준으로 교정(또는 subject-left/right 규약 주석 명시), 눈썹 인덱스에 46,53/276,283 추가. |
| 2라운드 보완 감사 | 미러링 규약이 호출자마다 다르고 모듈 내 주석은 허위 — CPU 경로는 미러링 없음, GPU 경로는 X 반전, createSkinMask 주석은… | `cpp/src/beauty_roi_manager.cpp` | :233 허위 주석 삭제. computeROI 계약(입력 좌표 기준: 비미러 원본 정규화 / 출력 face_rect: 정규화)을 헤더에 명시하고, 미러링 책임을 한 층(권장: 호출자가 아닌 주입 어댑터)으로 고정. |
| 2라운드 보완 감사 | IRIS_SDK_HAS_OPENCV ifdef가 허구 — 헤더만 조건부, 구현은 OpenCV 무조건 사용이라 OpenCV 미발견 구성에서 컴파일… | `cpp/src/beauty_roi_manager.cpp` | ifdef를 제거하고 OpenCV를 이 모듈의 필수 의존으로 명시(CMake에서 REQUIRED)하거나, 반대로 fillPoly/blur를 자체 스캔라인 래스터화로 대체해 OpenCV-free 경로를 실제로 제공. … |
| 2라운드 보완 감사 | test_beauty_roi_manager가 마스크 '크기·유효성'만 검증하고 위치 정합은 무검증 — 좌표계 결함이 테스트를 전부 통과 | `cpp/tests/test_beauty_roi_manager.cpp` | 합성 얼굴 랜드마크(예: 정규화 좌표를 명시 배치한 478점 fixture)로 '눈 중심 좌표의 combined_mask 값 ≈ 0, 이마/볼 좌표 ≈ 255, 윗입술 좌표 ≈ 0' 같은 위치 기반 어서션을 추가. … |
| GL/GPU 상태 | ES 3.0 폴백 컨텍스트와 #version 310 es 전 셰이더 불일치 — 폴백 경로 사장 | `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/cpp/src/gpu/g…` | 셰이더가 3.1 전용 기능을 쓰지 않으면 #version 300 es로 통일(현재 셰이더에 compute/SSBO 사용 없음 — 확인 범위 내), 아니면 3.0 폴백 분기를 제거하고 최소 요구 사항을 3.1로 문서화… |
| GL/GPU 상태 | 프레임 시작마다 glClientWaitSync 최대 16ms CPU 블로킹 — 단일 컨텍스트에선 불필요한 스톨 + timeout 시 보호 무력 | `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/cpp/src/gpu/g…` | 단일 컨텍스트 전제라면 fence wait 제거(생성/삭제만 유지하거나 전부 제거). 멀티 컨텍스트 대비가 목적이면 timeout 시 풀 반환을 지연시키는 방향으로 일관성 확보 + 대기 시간 프로파일링. |
| GL/GPU 상태 | ROI 활성 시 매 프레임 ping/pong 전면 passthrough 사전채움 2패스 — 대역폭 낭비 | `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/cpp/src/gpu/g…` | glCopyTexSubImage2D/glBlitFramebuffer로 입력→출력 복사로 대체하거나, scissor 외부 영역만 4개 사각형으로 채우기, 또는 최종 컴포지트에서 ROI 외부를 입력에서 직접 샘플하는 구… |
| GL/GPU 상태 | renderSkinBasePasses에서 매 프레임 glGetUniformLocation 호출 — uniform 캐시 정책 위반 | `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/cpp/src/gpu/g…` | cacheUniformLocations()에 passthrough uTexture location을 추가하고 캐시 사용. |
| GL/GPU 상태 | 런타임 에러 처리 정책 비일관 — FBO completeness/glGetError가 사실상 디버그 빌드 전용, checkGLError 단일 소… | `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/cpp/src/gpu/g…` | release에서도 프레임당 1회의 경량 glGetError 드레인 루프(while)를 frameEnd 지점에 두고 카운터만 누적, checkGLError는 루프로 전체 소비. 디버그 LOGI는 LOGD로 강등. |
| GL/GPU 상태 | 데드 코드/잔류 상태: applyTextureId의 미사용 input_handle, write-only previous_output_textur… | `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/cpp/src/gpu/g…` | (1)(2) 삭제. (3) 조기 반환 전에 이월 반환/fence 정리 블록을 먼저 수행하도록 호이스팅. |
| GL/GPU 상태 | GPUProfiler 라운드로빈 쿼리 재사용 — in-flight 쿼리 덮어쓰기로 측정 유실 가능 | `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/cpp/src/gpu/g…` | 쿼리별 'pending' 상태를 추적해 결과 수집 전 재사용을 건너뛰고, 같은 이름 충돌 시 이름+세대 키로 보관하거나 수집 후 덮어쓰기. |
| GL/GPU 상태 | TexturePool::onMemoryPressure 설계가 GL 스레드 친화성을 무시 — 현재 미연결(죽은 경로)이나 연결 즉시 결함 | `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/cpp/src/gpu/t…` | onMemoryPressure는 '정리 요청 플래그'만 세우고 실제 glDelete는 다음 applyTextureId(GL 스레드) 진입 시 수행하는 지연 정리 패턴으로 변경. 헤더의 스레드 안전성 주석에 'GL 호… |
| GL/GPU 상태 | 카메라 프레임-랜드마크 동기화 부재: Preview/ImageAnalysis 별도 스트림 + timestamp 매칭 없음 → 구조적 렌즈 밀림 | `android/demo-app/src/main/java/com/irislenssdk/demo/GpuRende…` | 단기: updateTexImage 후 surfaceTexture.timestamp와 IrisResult.timestampMs 차이를 로깅해 실측 시차 정량화. 전환 설계: 주입 API를 (frame_id¦timest… |
| GL/GPU 상태 | renderLensOverlay의 '동적 eyelidFeather 픽셀 기반 계산'이 상수 4.0px로 고정된 죽은 코드 | `android/demo-app/src/main/java/com/irislenssdk/demo/camera/g…` | 상수 4.0f로 단순화하고 주석 수정, 또는 의도했던 동적 식(검출 크기 비례 등)을 실제로 구현. |
| GL/GPU 상태 | createProgram 실패 경로의 셰이더 객체 누수 + 컴파일 실패 시 0 핸들 attach | `android/demo-app/src/main/java/com/irislenssdk/demo/camera/g…` | compileShader 반환 0 검사 후 조기 return 0, 링크 실패 분기에서도 glDeleteShader 2회 호출. |
| GL/GPU 상태 | 매 프레임 glGetUniformLocation 호출 (renderLensOverlay/renderToScreen) | `android/demo-app/src/main/java/com/irislenssdk/demo/camera/g…` | onSurfaceCreated의 location 캐시 블록에 두 프로그램의 해당 uniform 4종을 추가. |
| GL/GPU 상태 | Preview 스트림 변환을 ST 행렬의 기기 의존 동작에 위임 — SurfaceRequest.TransformationInfo 미사용, 회전값… | `android/demo-app/src/main/java/com/irislenssdk/demo/camera/g…` | request.setTransformationInfoListener로 rotationDegrees/crop을 수신해 Preview 스트림의 단일 권위 소스로 사용하고, ST 행렬에는 buffer layout 변환만 … |
| GL/GPU 상태 | gpu_lens_renderer.cpp의 낡은 주석: 'avg_iris_luma 실측 미연결, fallback만 동작' — 실제로는 P7-W2에… | `cpp/src/gpu/gpu_lens_renderer.cpp` | 803-805, 898-899 주석을 현재 상태(P7-W2 실측 연결, EXTERNAL_OES self-measure는 제거됨)로 갱신. |
| 결합도 | copyResultFromJava가 홍채 boundary 랜드마크 [1..4]와 visibility를 복사하지 않음 — Java→네이티브 주입 … | `android/iris-sdk/src/main/cpp/iris_jni.cpp` | 2단계 경계 설계 시 주입 계약을 '478점 + 파생 요약'으로 명시 재정의하고, boundary 5점 표현을 계약에서 제거하거나 완전 복사로 정정 |
| 결합도 | 공개 헤더에 cv::Mat 시그니처 노출 — OpenCV가 SDK 공개 표면의 일부 | `cpp/include/iris_sdk/frame_processor.h` | 2단계에서 GPU-only 코어로 좁히며 해당 헤더들을 제거 대상에 포함하거나, CPU 경로 유지 시 cv::Mat을 Pimpl 내부로 강등하고 공개 표면은 uint8_t*+stride로 통일 |
| 결합도 | eye_render_packet_adapter가 렌더러에 역의존 + 눈꺼풀 인덱스 상수를 '동일값' 주석에 기대어 중복 재선언 | `cpp/src/gpu/eye_render_packet_adapter.cpp` | 2단계에서 fitEyeEllipse/medianLandmarkY와 눈꺼풀 인덱스를 skin_mask_geometry.h처럼 독립 geometry 유닛(예: eye_geometry.h)으로 추출해 렌더러·어댑터 양쪽이… |
| 결합도 | selfie_multiclass_256x256.tflite가 코드 참조 0건인 dead asset으로 2곳에 중복 탑재 | `shared/models/selfie_multiclass_256x256.tflite` | 2단계 에셋 정리 시 양쪽 모두 삭제 (헤어 세그멘테이션 등 향후 계획이 있다면 docs로 이동) |
| 스레드/수명 | GPU 리소스 해제가 GL 컨텍스트 비현재 스레드/소멸된 컨텍스트에서 실행되는 경로 | `android/demo-app/src/main/java/com/irislenssdk/demo/GpuRende…` | releaseGpuBeauty/releaseGpuLens 호출을 GL 스레드 queueEvent로 일원화하고, onPause 이전(컨텍스트 생존 중)에 실행되도록 라이프사이클 훅 위치 조정. 또는 setPreserv… |
| 스레드/수명 | BufferPool: 파이프라인 미사용 죽은 코드 + move-assign ABBA 잠재 데드락 | `cpp/src/buffer_pool.cpp` | 전환 시 삭제 후보로 표기. 유지한다면 move 대입에 std::scoped_lock(mutex_, other.mutex_) 적용. |
| 스레드/수명 | async_cv_ 죽은 대기자: 비동기 프레임이 notify가 아닌 5ms 폴링으로만 처리됨 | `cpp/src/inference_thread.cpp` | threadLoop가 단일 cv로 동기/비동기 요청을 함께 대기하도록 통합(predicate에 has_new_frame_ 포함)하거나, async_cv_ 및 notify 호출을 제거해 폴링 설계임을 명시. |
| 스레드/수명 | mediapipe_detector.cpp 함수-로컬 static bool 디버그 플래그의 다중 인스턴스 경합 | `cpp/src/mediapipe_detector.cpp` | std::atomic<bool> + exchange로 교체하거나 인스턴스 멤버로 이동. |
| 스레드/수명 | SDKManager log_callback_ 락 규율 불일치 (mutex_로 쓰고 log_mutex_로 읽음) | `cpp/src/sdk_manager.cpp` | log_callback_의 모든 읽기/쓰기를 log_mutex_로 통일 (initialize/shutdown 내 대입을 log_mutex_ 스코프로 이동). |
| 정확성 | CameraX TransformationInfo(hasCameraTransform) 완전 미처리 — GL 전처리 삽입 기기에서 프리뷰 방향/미러… | `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/android/demo-…` | SurfaceRequest.setTransformationInfoListener로 hasCameraTransform()을 구독하고, false일 때 frameRotation 기반 수동 회전/미러 행렬을 stMatri… |
| 정확성 | CPU 검출 경로 프레임당 풀프레임 변환·복사 5회 이상, 뷰티 데모 경로는 JNI 경계 3회 왕복 | `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/cpp/src/frame…` | 단기: convertNV21toBGR+cvtColor를 NV21→RGB 단일 변환으로 통합, 회전을 detector 입력 letterbox에 융합. 장기(아키텍처 전환): 검출 입력은 추적기 소유로 옮기고 코어는 랜… |
| 정확성 | ISS-004 홍채 위치 보정의 거리·임계값이 정규화 좌표 공간에서 계산 — 종횡비 왜곡으로 보정 트리거가 방향 의존적 | `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/cpp/src/media…` | 거리 계산 전 픽셀 변환(x×frame_width, y×frame_height) 또는 dx에 aspect 곱 보정. isOutlier/norm_radius도 동일 보정 권장. |
| 정확성 | FrameAnalyzer NV21 fast path가 마지막 chroma 바이트를 기록하지 않음 (uvSize-1 복사) | `android/demo-app/src/main/java/com/irislenssdk/demo/camera/F…` | 복사 후 `buffer[totalSize-1] = uBuffer.get(uBuffer.limit()-1)` 식으로 마지막 U 바이트를 보충하거나, 마지막 바이트를 직전 U 값으로 복제. |
| 정확성 | C++/Kotlin OneEuro 구현 분기 — dt 폴백 상이, eyelid beta 3원 분기(10/12/12) | `android/demo-app/src/main/java/com/irislenssdk/demo/camera/O…` | dt<=0 처리 정책을 한쪽(직전 dt 재사용 또는 1/60 폴백)으로 통일하고, eyelid beta는 실기기 검증값 하나로 결정해 3곳 동기화(코어로 단일화가 전환 목표에 부합). |
| 정확성 | Java applyBeautyFilterV2가 IrisResult 인자를 무시하고 detectionPtr=0L 전달 — 문서의 'ROI 기반 처… | `android/iris-sdk/src/main/java/com/irislenssdk/IrisLensSDK.j…` | copyResultFromJava 경유로 detection을 네이티브에 전달하는 오버로드 구현. 이때 FrameAnalyzer의 NV21(센서 방향)과 detectWithRotation 결과(회전 후 공간)의 좌표계… |
| 정확성 | 회전 검출의 좌표 공간 계약(회전 후 공간 반환, frame_width/height 갱신)이 공개 API 문서에 부재 | `cpp/include/iris_sdk/sdk_api.h` | sdk_api.h와 Java/Kotlin API doc에 '반환 좌표는 회전 적용 후 프레임 기준 정규화이며 result->frame_width/height가 회전 후 크기'임을 명시. 전환 설계 문서에 좌표 공간 … |
| 정확성 | 타원 rotation의 OneEuro 필터링에 각도 wrap 처리 부재 — 우안 rotation이 ±π 경계에 상주 | `cpp/src/gpu/gpu_lens_renderer.cpp` | rotation을 필터링 전 (-π/2, π/2]로 정규화(타원은 π 주기)하거나 sin/cos 쌍을 각각 필터링 후 atan2 재합성. |
| 정확성 | V1/Refiner 눈 ROI가 정규화 공간 정사각 — 비정사각 프레임에서 64x64 입력이 비등방 왜곡 | `cpp/src/mediapipe_detector.cpp` | cropFaceRegion처럼 픽셀 단위로 ROI를 계산(센터·eye_width를 픽셀로 환산 후 정사각 crop)하고 역변환도 픽셀 기준으로 일치시킴. |
| 정확성 | face_mesh z 좌표 스케일 불일치 — x/y는 전체 이미지 정규화로 변환되나 z는 모델 raw 값 유지 | `cpp/src/mediapipe_detector.cpp` | z도 입력 크기로 나누고 crop_scale_x를 곱해 전체 이미지 정규화 스케일로 통일하거나, 헤더 문서에 z 단위를 명시하고 전환 시 주입 z 계약을 확정. |
| 정확성 | iris_landmark 출력 구조 주석 자기모순 + 폴백 경로 텐서 범위 밖 읽기(6 floats OOB) | `cpp/src/mediapipe_detector.cpp` | 폴백 제거(Output[1] 없으면 실패 반환)하거나 71점 텐서에는 홍채가 없음을 반영해 에러 처리. 주석의 68-72 → 'Output[1] 전용'으로 정정. |
| 정확성 | left/right 라벨이 MediaPipe 해부학적 명명과 반전 (코드 'left'=468-472=MediaPipe FACEMESH_RIGHT… | `cpp/src/mediapipe_detector.cpp` | types.h/IrisResult 문서에 'left=비미러 프레임의 화면 왼쪽 눈(피사체의 오른눈)'임을 명시하고, 전환 시 주입 인터페이스에서 명명 규약(화면 기준 vs 해부학 기준)을 단일화. |
| 정확성 | 추적 모드에서 한쪽 눈만 검출 시 confidence가 프레임마다 0.5배 기하 감쇠 — 의도치 않은 재검출 강제 | `cpp/src/mediapipe_detector.cpp` | 추적 모드에서는 face_confidence를 이전 '얼굴' 신뢰도(별도 저장)로 유지하거나, 최종 confidence 산출에서 eye_factor 누적 곱을 막는 구조(검출 시점의 원본 face detection … |
| 정확성 | setMinTrackingConfidence가 데드 설정 — 저장만 되고 파이프라인 어디서도 미사용 | `cpp/src/mediapipe_detector.cpp` | min_tracking_confidence를 실제 추적 유지 판단에 연결하거나, API/문서에서 deprecated 처리 후 제거. |
| 정확성 | 홍채 반지름 = 4 경계점 평균거리 — 원근(yaw)에서 단축 평균으로 과소, 임계 8px는 해상도 비스케일 | `cpp/src/mediapipe_detector.cpp` | yaw 보정이 필요하면 4점 중 최대 거리 또는 타원 피팅 장축 채택 검토. Refiner 임계는 frame_width 비례(정규화)로 변경. |
| 정확성 | Eye Refiner 기본 정책 문서/구현 불일치 + iris_sdk_set_eye_refiner_policy가 no-op인데 OK 반환 | `cpp/src/sdk_api.cpp` | 스텁은 IRIS_SDK_NOT_SUPPORTED 류 에러 반환으로 정직화하거나 실제 연결 구현. 헤더 기본값 문구를 Never로 정정. 구현 전 finding #1/#2 선행 수정 필수. |
| 정확성 | TemporalStabilizer outlier 거리·기준 반경이 정규화 좌표 유클리드 — 이동 방향별 임계 비등방 | `cpp/src/temporal_stabilizer.cpp` | isOutlier/norm_radius 계산을 frame_width/height로 픽셀 환산 후 수행(IrisResult에 이미 보유). #1 수정과 함께 처리. |
| 정확성 | 워프 변위 스케일링 비일관 — V라인 dy는 얼굴 크기 미반영, face_width는 roll 비강건 | `cpp/src/warp/face_warp_controller.cpp` | dy도 얼굴 세로 크기(예: 이마-턱 거리) 비례로 스케일, face_width는 두 점 유클리드(픽셀 환산) 거리로 계산. |
| 테스트 안전망 | 모델 의존 통합 테스트의 silent skip 40+곳 — 환경 결손 시 스위트가 초록인 채 커버리지 소실 | `cpp/tests/test_integration.cpp` | 필수 자산 존재를 단정하는 가드 테스트 1개 추가(자산 있는데 skip되면 실패), 또는 CI에서 skip 수 임계 검사. |
| 테스트 안전망 | 벽시계 타이밍 단정 다수 — CI 도입 시 flaky 예약 | `cpp/tests/test_mediapipe_detector_performance.cpp` | 성능 단정을 PERF 라벨로 분리해 기능 스위트에서 제외하거나 환경변수로 게이트. 기능 검증과 성능 벤치를 같은 테스트에 섞지 않기. |
| 테스트 안전망 | 사문화 잔존물: 미등록 test_placeholder.cpp + tests/에 방치된 Mach-O 바이너리·스크린샷 | `cpp/tests/test_placeholder.cpp` | test_placeholder.cpp·바이너리·스크린샷 삭제(별도 정리 커밋), tests/ 산출물 .gitignore 보강. |
| 플랫폼 패리티 | 루트 CLAUDE.md가 존재하지 않는 core/·bindings/ 구조와 빌드 스크립트를 기술, 루트 README 부재, 엄브렐러 헤더는 0.… | `CLAUDE.md` | CLAUDE.md 경로 표를 cpp/include/iris_sdk/·android/iris-sdk/로 갱신, 없는 스크립트 제거, iris_sdk.h 스텁은 삭제하거나 실제 엄브렐러로 갱신. (이번 감사는 읽기 전용… |
| 플랫폼 패리티 | JNI 에러 삼킴 패턴 비일관: nativeStabilize는 에러→0.0f, nativeApplyFaceWarp는 에러→inputTexture… | `android/iris-sdk/src/main/cpp/iris_jni.cpp` | 텍스처/스칼라 반환 natives에 out-param 또는 에러 콜백을 추가하거나, 최소한 getLastError 연동 문서화. 전환 시 신표면에서는 (에러코드, 값) 분리 반환으로 통일. |
| 플랫폼 패리티 | deprecated 블렌드 모드(3/4/6)의 의미론이 Java/Kotlin 문서와 데모 KT fallback 셰이더에 미전파 | `android/iris-sdk/src/main/java/com/irislenssdk/LensConfig.ja…` | Java/Kotlin 상수에 @Deprecated + 'TintLinearV2 fallback' 문서 동기화, 데모 KT fallback 셰이더의 3/4/6 분기를 5로 통일(메모리상 w9 잔여 항목과 동일). |
| 플랫폼 패리티 | C API 문서화 필드 feather_radius가 어디에도 소비되지 않는 dead field | `cpp/include/iris_sdk/sdk_api.h` | toCppConfigV2/fromCppConfigV2에 연결하고 C++/Java 구조체에 필드 추가하거나, C struct에서 필드를 제거(ABI 주의)하고 문서에서 deprecated 명시. |

## 5. LensSimulator 함정 목록 13건 대조

알려진 함정 13건 대조 감사(읽기 전용) 결과. 본 SDK는 MediaPipe Tasks가 아닌 자체 TFLite 파이프라인이므로 7건은 직접 비해당이나, 유사 함정 점검에서 4건의 실위반을 추가 확인했다. 직접 대조 대상 4건(#4/#5/#10/#12) 중 3건에서 실제 위반을 코드로 확정했다.

【13건 판정표】
| # | 판정 | 한줄 근거 |
|---|------|-----------|
| 1 | 비해당+유사위반 | Tasks 미사용. 단 자체 .so 16KB 정렬 미조치(AGP 8.5.0, ndkVersion 미지정, max-page-size 0건) → Play 제출 리스크 |
| 2 | 비해당+유사위반 | FaceLandmarker 미사용. 단 CPU 경로 프레임당 풀프레임 변환·복사 5회+(YUV→NV21→JNI→BGR→RGB→rotate), NV21→BGR→RGB 중복 |
| 3 | 준수 | InferenceThread 기본 ON으로 init+detect 동일 워커 스레드(inference_thread.cpp:294-308,351), direct 모드는 GPU 명시 차단(frame_processor.cpp:196-199 "스레드 제약으로 지원하지 않음"), GPU 실패 시 CPU/XNNPACK 폴백 존재(mediapipe_detector.cpp:518-534) |
| 4 | 위반 2건 | (a) lens_renderer.cpp:105-141 iris[3]/iris[4]를 좌/우로 가정 → 렌즈 회전각 상시 ~45° 왜곡 (b) gpu_lens_renderer.cpp:62-65·CameraGLRenderer.kt:71-74 내안각/외안각 반전(33을 코 쪽으로 오라벨, detector 정의 102-106과 상충) → caruncle 보호 0.85가 귀 쪽에 적용 |
| 5 | 위반 다수 | computeEAR 정규화 거리+픽셀 기준 임계 0.2(temporal_stabilizer.cpp:315-338) → 종횡비별 blink 오판, ISS-004 거리/임계(mediapipe_detector.cpp:1897-1923), 눈 확대 워프 radial(face_warp_controller.cpp:298-331). 반면 calculateIrisRadius(2239-2254)·lens_renderer 반경(110-117)은 픽셀 변환 후 계산으로 준수 |
| 6 | 비해당 | iOS 바인딩 코드 없음(bindings/ios .gitkeep만) |
| 7 | 비해당 | Flutter 바인딩 코드 없음 |
| 8 | 비해당 | iOS 코드 없음 |
| 9 | 유사위반 | consumer-rules.pro 존재·배선(build.gradle.kts:66)되나 JNI가 FindClass/GetFieldID로 접근하는 BeautyFilterConfig/V2(iris_jni.cpp:131-186) keep 누락 → 호스트 R8 release에서 initJniCache 실패, SDK 전체 초기화 불가 |
| 10 | 위반 | TemporalStabilizer radius_beta=7.5("tuned from Android demo"=정규화 공간 튜닝)를 픽셀 단위 radius에 적용(temporal_stabilizer.cpp:275) → 컷오프 폭주로 radius 스무딩 사실상 무효. detector IrisOneEuroFilter도 정규화 x/y+픽셀 r 혼용(mediapipe_detector.cpp:3108-3143) |
| 11 | 비해당+유사주의 | Flutter 플랫폼 채널 없음. 단 CPU 뷰티 데모 경로에서 프레임이 JNI 경계 3회 왕복(detect/beauty/nv21ToRgba) + 매 적용 Bitmap 재할당 |
| 12 | 부분준수+유사위반 | rotationDegrees를 GL에 중복 적용하지 않아 이중 회전은 없음(stMatrix 단독 의존, frameRotation은 종횡비 스왑만). 단 TransformationInfo/hasCameraTransform 조회 0건(CameraGLView.kt:161-181) → 폴백 기기 분기 부재, uMirror 수동 적용은 ST 미러 포함 기기에서 이중 미러 가능 |
| 13 | 비해당(설계 회피)+준수 | LandmarkProjection 함정 자체가 없음: SDK가 cv::rotate로 픽셀을 회전 후 검출하고 좌표를 upright 공간+result.frameWidth/Height 계약으로 반환(frame_processor.cpp:702-726 주석 명시), 데모 resolveCoordinateSpace(CameraGLRenderer.kt:1434-1448)가 동일 계약 소비, 미러는 렌더 단계(회전 후) 적용으로 순서 준수. 대가는 매 프레임 풀프레임 회전 복사(#2 finding에 포함) |

【종합】 blocker는 없으나, 아키텍처 전환과 직결되는 major 6건: (1) 홍채 경계점 시맨틱 오인 2건(#4)은 '랜드마크 주입 코어'가 물려받을 좌표 계약 결함이라 전환 전 정리 필수. (2) 정규화 거리(#5)·One-Euro 단위 혼용(#10)은 코어가 "정규화 478점 주입"을 표준 입력으로 삼는 순간 더 광범위해질 패턴 — 전환 시 좌표 단위 계약(정규화 vs 픽셀, aspect 보정 책임자)을 단일 문서로 강제할 것을 권고. (3) consumer ProGuard(#9)와 16KB(#1 유사)는 외부 배포 차단성 패키징 결함. 한편 #3(스레드 친화성)과 #13(회전 좌표 계약)은 LensSimulator가 실기기에서 밟은 함정을 본 SDK가 이미 구조적으로 회피하고 있음을 확인 — 전환 시 이 두 설계(추론 전용 스레드, upright 좌표+frameW/H 계약)는 보존 가치가 있다.

함정 연계 확정 findings: #JNI spec ch.11 'Exception Handling' — pending exception 중 호출 가능 함수 화이트리스트, #cache-key-collision, #coupling finder dead-code 목록 누락분 보완, #cpu_beauty_backend ROI 마스크 좌표계 불일치(기존 major finding)의 생산자측 근원; grid_mesh 468/478 불일치와 같은 '계약 모호' 계열, #grid_mesh 468/478 불일치 전례, #left-right-semantics, #model-encryption, #observer-effect, #sdk-surface-consistency — 동일 surface 내 이중 경로 불일치, #shared-mutable-snapshot, #triple-mapping-equivalence, #verification-tool-divergence, #verify-loadbearing-facts — 주석만 믿지 말 것의 실례, #점검 1, #점검 1 (정규화 좌표 거리 계산 종횡비 왜곡), #점검 1 + 점검 4, #점검 2, #점검 2 + 점검 5, #점검 3, #점검 3 + 점검 1, #점검 3 + 점검 5, #점검 4 (히스테리시스/스냅 로직), #점검 5, #점검항목 1 (cv/notify 정합), 2 (드랍 정책), #점검항목 1 (락 없이 읽고 쓰는 상태), 데모 스레드 경계, #점검항목 1 (락 없이 읽고 쓰는 플래그), #점검항목 1 (뮤텍스가 보호 대상 전체를 덮는지), #점검항목 2 (buffer_pool 반환 경합), #점검항목 2 (버퍼 use-after-free), 1 (atomic 상태기계), #점검항목 2 (카메라→추론→렌더 핸드오프), 1 (락 없는 포인터), #점검항목 3 (GL 객체가 사라진 컨텍스트에서 해제), #점검항목 3 (thread join 누락/타임아웃), 1 (atomic 오용), #점검항목 3 (종료/재시작), 4 (수명)

## 6. 결합도 — 추적 레이어 교체 가능성 평가 (핵심 관점)

### 6.1 종합

전수 감사 결과, 추적(TFLite/MediaPipe)과 이펙트/렌더 코어의 결합은 '코드 결합은 얕고 데이터 계약 결합은 깊은' 구조다. TFLite 헤더 include는 mediapipe_detector.cpp 단일 파일에 격리(조건부 가드 26곳)되고 CMake 링크도 PRIVATE라 빌드 분리는 기계적이다. 반면 검출 산출물 IrisResult(face_mesh[478] 인라인 + detector 전용 메타)가 C++/C/JNI/Java 4중 표현으로 전 레이어를 왕복하며, 렌더 코어는 478점 전체(ellipse fitting, skin mask fan, warp control point)와 detector 부수 산출물(confidence·eyelid_ratio→visibility, avg_iris_luma 실측)을 소비한다. 결정적으로 GPU 렌더/뷰티/워프 C API는 이미 '텍스처 ID + IrisResult 포인터' 주입형이고, JNI DetectionSlot(Java→네이티브 lock-free 슬롯)이 외부 검출 결과 주입의 실증 프로토타입으로 동작 중이라 목표 아키텍처(iris_set_landmarks + iris_render)와 구조적으로 동형이다. 따라서 경계 도입은 신규 설계가 아닌 기존 경계의 공식 승격이며, 실질 난이도는 detector가 부수 생산하던 파생 데이터(홍채 중심/반경 유도, EAR, luma 실측, 회전 보정 좌표 규약)의 코어 측 이식·실기기 검증에 있다(견적 12~16 사람·일). 결함으로는 Strategy 팩토리 전체가 nullptr을 반환하는 죽은 코드(파이프라인은 MediaPipeDetector 구체 의존), DetectionSlot 더블버퍼의 torn-read race(generation 검증 API 미노출), C/C++ IrisResult reinterpret_cast의 sizeof-only 가드 등 major 3건을 포함해 7건을 확인했다. 픽셀 경계 목표 관점에서 GPU 경로는 이미 충족하며, CPU 뷰티/렌즈(OpenCV cv::Mat 공개 노출)가 유일한 충돌 경로다.

### 6.2 결합 지점 인벤토리 (10건 + 누락 sweep 검증)

| 위치 | 강도 | 내용 | 분리 방안 |
|---|---|---|---|
| `cpp/src/frame_processor.cpp:128-129, 186-203` | hard | FrameProcessor::Impl이 IrisDetector 추상이 아닌 MediaPipeDetector 구체 타입(direct_detector_)과 InferenceThread를 직접 소유·생성(make_unique<MediaPipeDetector>(), make_unique<InferenceThread>()). 검출… | FrameProcessor 자체를 추적 측으로 분류해 통째로 제거. 코어에는 renderOnly 계열만 남기지 말고 GPU 텍스처 경로(sdk_api_v2)로 일원화 |
| `cpp/include/iris_sdk/inference_thread.h:233 (std::unique_ptr<MediaPipe…` | hard | InferenceThread가 IrisDetector 추상이 아닌 MediaPipeDetector를 직접 소유. 존재 이유 자체가 TFLite GPU delegate의 스레드 친화성 제약(헤더 주석 5-6행) — 코어 디렉토리에 있지만 본질적으로 추적 인프라 | 추적과 함께 통째로 삭제. 외부 MediaPipe Tasks가 자체 스레딩을 처리하므로 코어에 대체물 불필요 |
| `cpp/include/iris_sdk/types.h:138-181 + cpp/include/iris_sdk/sdk_api.h:…` | medium | IrisResult(C++/C 이중 정의)가 face_mesh[478] 인라인(~7.7KB POD) + detector 전용 메타(eye_refiner_used, iris_quality_*, eyelid_ratio_*, avg_iris_luma_*)를 함께 운반. 검출기 산출물 타입이 그대로 렌더 입력 계약 | 목표 C API의 iris_set_landmarks(float*478x3, ts)를 입력 계약으로 삼고, IrisResult를 코어 내부 파생 타입으로 강등. detector 메타 필드는 분리하거나 코어 측 재계산으로 대체 |
| `cpp/src/sdk_api_v2.cpp:39-42, 243, 361, 377, 472, 666` | medium | C IrisResult ↔ C++ iris_sdk::IrisResult를 reinterpret_cast로 교환. 두 struct는 수동 동기화되는 별도 정의이며 가드는 sizeof static_assert 하나뿐 | 경계 도입 시 단일 정의 공유(헤더 1곳) 또는 필드별 offsetof static_assert 추가. 신규 경계 타입을 또 얹기 전에 정리 필요 |
| `cpp/src/mediapipe_detector.cpp:3164-3169 (calculateIrisLuma)` | medium | avg_iris_luma 실측(P7-W2 default ON)이 detector 내부에서 입력 프레임 픽셀로 수행됨. 추적 제거 시 실측 소스가 함께 사라져 TintLinearV2 블렌드 정규화가 fallback 상수 체인(gpu_lens_renderer.cpp:804-818)으로 후퇴 | GPU self-measure로 이전(gpu_lens_renderer.cpp:898, 1121에 'W6 이관' 주석으로 이미 계획 존재) 또는 주입 API에 luma 파라미터 추가 |
| `cpp/src/gpu/eye_render_packet_adapter.cpp:96-100 + types.h:170-171` | medium | EyeRenderPacket.visibility = confidence × (1−eyelid_ratio). confidence/eyelid_ratio_*가 detector(Eye Refiner) 산출 메타 — 외부 478점 주입 시 이 값들의 공급자가 사라짐 | 478점에서 EAR 재계산으로 대체 — 수식은 temporal_stabilizer.cpp:315 computeEAR에 이미 존재, 코어 측 어댑터로 이동 |
| `cpp/src/gpu/gpu_lens_renderer.cpp:652-789 / gpu/skin_mask_geometry.h:2…` | soft | 렌더/뷰티/워프 코어 전체가 MediaPipe 478점 인덱스 규약(눈꺼풀 159/145/386/374 contour, 얼굴 윤곽 등)에 의미적으로 결합. 코드 의존(헤더 include)은 없음 | 조치 불필요 — MediaPipe Tasks FaceLandmarker도 동일 478 규약이라 외부 주입 후에도 그대로 유효. 인덱스 상수의 단일 출처화만 권장 |
| `android/iris-sdk/src/main/cpp/iris_jni.cpp:559-566, 1870-1924 (Detecti…` | soft | Java IrisResult → 네이티브 더블버퍼 슬롯 → GL 스레드가 jlong 포인터로 렌더 함수에 전달. 이미 '외부가 검출 결과를 주입'하는 사실상의 주입 경계가 JNI 레벨에 존재 | 이 패턴을 C API 레벨 iris_set_landmarks로 공식 승격. 승격 전 torn-read race(findings 참조) 수정 필수 |
| `cpp/CMakeLists.txt:224-394 + cpp/third_party/tflite/ + android/iris-sd…` | hard | TFLite 탐색(시스템/brew/FetchContent/prebuilt 4경로)·링크·IRIS_SDK_HAS_TFLITE/HAS_GPU_DELEGATE/HAS_XNNPACK 매크로. 모두 iris_sdk 타겟 PRIVATE — 공개 헤더로 새지 않음. TFLite #include는 mediapipe_detector.cp… | CMake 블록 235-394 삭제 + third_party/tflite(arm64-v8a/armeabi-v7a prebuilt .so) 삭제. 격리가 양호해 기계적 제거 가능 |
| `cpp/include/iris_sdk/frame_processor.h:20,183 / cpu_beauty_backend.h:1…` | medium | OpenCV(cv::Mat)가 CPU 이펙트 경로의 공개 헤더 시그니처·멤버에 노출 (헤더 11개, 구현 9개). GPU 이펙트 경로(gpu_beauty_backend/gpu_lens_renderer/shader/warp)는 OpenCV-free | GPU-only 코어 선택 시 CPU 경로(cpu_beauty_backend, beauty_filter, beauty_processor, lens_renderer, fast_guided_filter, cpu_render_context)와 함께 Open… |

**누락 sweep 판정**: 전수 재조사 방법: (1) 지정 키워드(tflite|interpreter|xnnpack|delegate|mediapipe|face_mesh|face_detection|iris_landmark)를 cpp/include, cpp/src, cpp/CMakeLists.txt, 루트 CMakeLists.txt, android/iris-sdk/src, scripts/에 case-insensitive grep(빌드 디렉토리 제외) 후 파일별 매치를 인벤토리와 대조, (2) find로 .tflite/.task/.binarypb 에셋 전수 확인, (3) types.h include 파일 23개 전수 대조 + cpp/tests 추적 키워드 grep 교차 확인. 추가로 sdk_api.cpp/sdk_manager.cpp/sdk_api_v2.cpp의 FrameProcessor 의존, Kotlin 바인딩 디렉토리, demo-app gradle 의존성을 직접 열어 검증했다.

총평: 인벤토리는 코어 C++의 주요 하드 결합(MediaPipeDetector/InferenceThread/FrameProcessor 소유 구조, IrisResult 이중 정의 reinterpret_cast, … (누락 보고 9건)
- 누락: `cpp/src/sdk_api.cpp:43-44, 324-374, 376-430, 446 (iris_sdk_init / iris_sdk_init_with_config / g_processor / iris_sdk_destroy)` — C API 라이프사이클 진입점 자체가 추적에 하드 결합. iris_sdk_init(324행)과 iris_sdk_init_with_config(376행)가 전역 g_processor(43-44행 `std::unique_ptr<iris_sdk::FrameProcessor> g_processor`)를 직접 생성하고 `g_processor->initialize(m…
- 누락: `cpp/CMakeLists.txt:395-537 (인벤토리는 224-394만 명시)` — TFLite CMake 블록의 실제 범위가 인벤토리 명시(224-394)보다 약 145행 더 길다. 395-441행: TFLite include 경로 대체 탐색(`foreach(ALT_BUILD_DIR "build_tflite" "build")` 399행, `IRIS_SDK_HAS_TFLITE` 정의 438행), 444-523행: FetchContent 다…
- 누락: `scripts/download_models.sh:24-37,63-65 / scripts/check_dependencies.sh:52-66 / scripts/setup_env.sh:69` — scripts/ 디렉토리가 removalScope에 통째로 누락. download_models.sh는 파일 전체가 MediaPipe tflite/task 모델 다운로드 전용(`FACE_DETECTION_URL=...blaze_face_short_range.tflite` 24행, `IRIS_LANDMARK_URL=...iris_landmark.tflite` …
- 누락: `android/iris-sdk/src/main/java/com/irislenssdk/kotlin/IrisLensSDKKt.kt:187,233,292 + ProcessResultKt.kt:81 + FrameFormat.kt + IrisLensSDK.java:355,400,416,431,496,509` — Kotlin SDK 표면 전체가 removalScope에 누락. IrisLensSDKKt.kt의 detect(:187)/process(:233)/detectOnly(:292)는 Java detect/process를 감싸는 공개 Kotlin API이고, ProcessResultKt.detectionOnly(:81)와 FrameFormat.kt(NV21/NV1…
- 누락: `android/iris-sdk/src/main/java/com/irislenssdk/kotlin/IrisResultKt.kt:125,187,236` — 결과 타입이 C struct(sdk_api.h) / C++ struct(types.h) / Java class(IrisResult.java)에 더해 Kotlin data class(IrisResultKt)까지 4중 수동 동기화 정의임이 인벤토리에 빠짐(인벤토리 pt4는 C↔C++ 이중 정의만, pt3 데이터 흐름은 Java까지만 언급). IrisResult…
- 누락: `cpp/src/sdk_manager.cpp:128-148, 372-374 + cpp/include/iris_sdk/sdk_manager.h:63, 191` — SDKManager::createFrameProcessor 팩토리가 removalScope의 sdk_manager 항목(createDetector :150-176, detector_type만 명시)에서 누락. 128-148행에서 `std::make_unique<FrameProcessor>()` 생성 후 `processor->initialize(config_…
- 누락: `cpp/tests/test_frame_processor.cpp:18 / cpp/tests/test_lens_renderer_integration.cpp:14,49,158 / cpp/tests/test_types.cpp:392` — 인벤토리 테스트 목록(명시 9파일 + '외 grep 매치 12건')에서 하드 결합 테스트 2개가 미명명: test_frame_processor.cpp(:18 `#include "iris_sdk/frame_processor.h"`, FrameProcessor 전체 삭제 시 파일 전체 제거 대상)와 test_lens_renderer_integration.cpp…
- 누락: `shared/models/face_landmarker.task, shared/models/face_landmarker_latest.task / android/demo-app/src/main/assets/face_landmarker.task + MediaPipeBenchmarkActivity.kt:49 + demo-app/build.gradle.kts:85` — find 전수 결과 .task 모델 번들 3개가 에셋 인벤토리에 누락. shared/models/face_landmarker.task와 face_landmarker_latest.task는 코드 참조 0건(scripts/download_models.sh:28의 다운로드 URL만 존재)인 정리 대상. 반면 android/demo-app/src/main/asse…
- 누락: `cpp/src/warp/grid_mesh.cpp:82,182 / cpp/src/temporal_stabilizer.cpp:8 / cpp/src/cpu_beauty_backend.cpp:436,581 / cpp/src/beauty_processor.cpp:146-154 / cpp/src/gpu/gpu_beauty_backend.cpp:1769-1790` — 인벤토리 pt7(478점 인덱스 규약 소프트 결합)의 파일 목록이 4개 파일로 과소 집계. 동일한 의미적 결합이 grid_mesh.cpp(:82 `landmark_to_vertex_.resize(468, -1)` — 478이 아닌 468 기준, :182 `landmark_count < 468` 가드), temporal_stabilizer.cpp(:8 'Me…

### 6.3 랜드마크 주입 지점(injection seam) 제안 + 적대 검증

최적 주입 지점은 이미 코드에 존재하는 2단 경계의 공식화다. (1) C API 레벨: sdk_api_v2.cpp의 GPU 렌더 계열 함수가 받는 `const IrisResult* detection` 인자 — iris_sdk_render_lens_texture(sdk_api.h:907), iris_sdk_apply_beauty_texture_v2(:750), iris_sdk_apply_face_warp(:801), iris_sdk_stabilize(:994). 이들은 이미 픽셀이 아닌 텍스처 ID + 검출 결과 포인터만 받으므로 목표 아키텍처(iris_render(target,w,h))와 동형이다. (2) JNI 레벨: DetectionSlot(iris_jni.cpp:559-566) + nativeUpdateDetectionSlot(:1870-1895) — Java가 만든 IrisResult를 네이티브 슬롯에 복사하고 GL 스레드가 포인터로 소비하는 구조로, 외부 MediaPipe Tasks 결과를 Java IrisResult(faceMesh float[478*3], IrisResult.java:183)에 채워 넣으면 코어 렌더 경로는 무변경으로 동작한다. 신설할 iris_set_landmarks(float* pts478x3, int64 ts)는 이 슬롯을 C API로 승격하고, 478점→IrisResult 파생 필드(홍채 중심/반경: MediaPipe 규약상 인덱스 468-477, eyelid EAR: temporal_stabilizer.cpp:315 computeEAR 수식 재사용, face_rect: 메시 바운딩)를 유도하는 어댑터를 코어에 두면 된다. 이 seam 기준 위(추적 측)로 mediapipe_detector, inference_thread, frame_processor, sdk_api detect 계열, JNI nativeDetect*가 나가고, 아래(코어)에 temporal_stabilizer, eye_render_packet_adapter, gpu_lens_renderer, gpu_beauty_backend, warp, beauty ROI/CPU, lens_sku_metadata, shader가 남는다.

**seam 적대 검증**: 구조 유효 판정 — 결론: 제안된 2단 seam(C API IrisResult* 인자 + JNI DetectionSlot)은 구조적으로 유효하다. 단, 어댑터 사양과 removalScope에 보완 필수 갭 4건이 있다(아래 G1~G4). seam 위치 자체를 무효화하는 결함은 발견하지 못했으므로 refuted=false.

[검증 1 — seam이 정말 추적을 분리하는가: 성립]
- /Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/cpp/src/sdk_api_v2.cpp 전체(847줄)를 읽음: iris_sdk_render_lens_texture(:647), iris_sdk_apply_beauty_texture_v2(:313), iris_sdk_apply_face_warp(:435)는 자체 g_gpu_lens/g_gpu_beauty 전역만 사용하고 SDKManager/FrameProcessor/detector를 전혀 참조하지 않음. iris_sdk_init 없이 iris_sdk_init_gpu_lens(:530)만으로 동작 — 렌더 경로는 이미 검출과 독립.
- TFLite include 격리 확인: `grep -rl "tensorflow/lite|tflite"` 결과 cpp/src/mediapipe_detector.cpp + cpp/include/iris_sdk/mediapipe_detector.h 단 2개 파일(빌드 산출물 제외) — 제안의 주장과 일치.
- MediaPipeDetector 참조는 frame_processor.cpp/inference_thread.{cpp,h}/iris_detector.{cpp,h}/mediapipe_detector.{cpp,h}에만 존재. seam 아래(gpu_lens_renderer, gpu_beauty_backend, warp, temporal_stabilizer, eye_render_packet_adapter, beauty ROI/CPU)에서 detector 헤더 include 0건.
- iris_detector.cpp(:13-49) 전 분기 nullptr 반환 죽은 팩토리 — 제안 주장 그대로 확인.
- 데모는 이미 seam 형태로 동작: GpuRenderActivity.kt:977 detectWithRotation → :993 stabilize(Java 오케스트레이션) → :1009 updateDetectionSlot → CameraGLRendere…

### 6.4 추적 제거 시 삭제 범위 (removalScope)

- cpp/src/mediapipe_detector.cpp(3425줄) + cpp/include/iris_sdk/mediapipe_detector.h(228줄) — TFLite #include가 격리된 유일한 파일
- cpp/src/inference_thread.cpp(431줄) + cpp/include/iris_sdk/inference_thread.h(260줄)
- cpp/src/iris_detector.cpp(52줄, 전부 nullptr 반환 죽은 팩토리) + cpp/include/iris_sdk/iris_detector.h(116줄)
- cpp/src/frame_processor.cpp(1171줄) + frame_processor.h(447줄) — 검출/NV21·NV12 포맷변환/비동기 submitFrame/결과 캐싱 전부. GPU-only 전환 시 파일 전체
- cpp/src/sdk_manager.cpp 부분 — createDetector(:150-176), SDKConfig.detector_type(sdk_manager.h:66)
- cpp/src/sdk_api.cpp 부분 함수: iris_sdk_detect, iris_sdk_detect_with_rotation, iris_sdk_process, iris_sdk_submit_frame(+with_rotation), iris_sdk_get_latest_result, iris_sdk_render_with_result, iris_sdk_set_min_{detection,tracking,presence}_confidence, iris_sdk_set_use_inference_thread, iris_sdk_is_using_inference_thread, iris_sdk_set_gpu_enabled/is_gpu_available/is_using_gpu(TFLite delegate 의미), iris_sdk_set_eye_refiner_policy + sdk_api.h 해당 선언
- cpp/include/iris_sdk/types.h 부분: DetectorType(:88-93), EyeRefinerPolicy(:99-103), IrisResult detector 메타 필드 정리(선택)
- JNI(iris_jni.cpp): nativeDetect, nativeDetectWithRotation, nativeProcess, nativeSetMinDetectionConfidence, nativeSetMinTrackingConfidence, nativeSetMinPresenceConfidence, nativeSetUseInferenceThread, nativeIsUsingInferenceThread, nativeSetGpuEnabled, nativeIsGpuAvailable, nativeIsUsingGpu (+nativeInit의 model_path 의미 축소)
- Java/Kotlin: IrisLensSDK.java detect(:310)/detectWithRotation(:333)/setMin*Confidence 계열, demo FrameAnalyzer.kt 검출 경로, GpuRenderActivity.kt 검출 호출부(:977)
- 모델 에셋: shared/models/{face_detection_short_range,face_landmark,face_landmark_v2,iris_landmark}.tflite + shared/models/backup_v1/ + shared/models/face_landmarker_extracted/ + android/demo-app/src/main/assets/models/ 동일 4종 (selfie_multiclass_256x256.tflite는 현재도 코드 참조 0건인 dead asset)
- 빌드: cpp/CMakeLists.txt:224-394 TFLite 블록(IRIS_SDK_USE_SYSTEM_TFLITE/IRIS_SDK_TFLITE_ENABLE_XNNPACK/IRIS_SDK_FETCH_TFLITE 옵션, prebuilt imported target, GPU delegate EGL 링크) + cpp/third_party/tflite/(android arm64-v8a, armeabi-v7a prebuilt .so + include) + android/iris-sdk/src/main/cpp/CMakeLists.txt:79 tflite 참조 + CMake 소스 목록의 mediapipe_detector/inference_thread/iris_detector/frame_processor 항목
- 테스트(12파일): test_mediapipe_detector.cpp, test_mediapipe_detector_integration.cpp(1466줄), test_mediapipe_detector_performance.cpp, test_iris_detector.cpp, test_iris_luma_measure.cpp, test_integration.cpp(1161줄), test_sdk_api.cpp, test_sdk_manager.cpp, test_camera_demo.cpp 외 grep 매치 12건
- cpp/examples/camera_demo.cpp(914줄)·image_demo.cpp(287줄) — FrameProcessor 기반

### 6.5 잔존 서드파티 의존성

OpenCV 침투도: 이원화 구조다. GPU 이펙트 코어(gpu_beauty_backend.cpp 2460줄, gpu_lens_renderer.cpp 1169줄, shader_sources/shader_manager/texture_pool/gles_render_context, warp 2종, eye_render_packet_adapter, temporal_stabilizer, lens_sku_metadata)는 OpenCV를 전혀 포함하지 않는다(opencv2 grep 0건 — GLES/EGL만 의존). 반면 CPU 이펙트 경로는 cv::Mat이 공용 데이터 타입으로 공개 헤더에 노출된다: cpu_beauty_backend.h:13 `#include <opencv2/core.hpp>` + cv::Mat 멤버 4개(:24-27), beauty_roi_manager.h:259-276 cv::Mat/cv::Rect 시그니처, lens_renderer.h:156 render(cv::Mat&), frame_processor.h:20 전방선언+:183 process(cv::Mat&) 공개 오버로드, fast_guided_filter.h, buffer_pool.h, ab_compare.h, quality_metrics.h, param_tuner.h, gpu/cpu_render_context.h — 헤더 11개·구현 9개. 추적 제거 시 frame_processor의 NV21/NV12 변환(OpenCV 최대 사용처)이 함께 사라지므로, 코어를 GPU 전용으로 좁히면 OpenCV 완전 제거가 가능하다. CPU 뷰티/CPU 렌즈(iris_sdk_apply_beauty_v2_c, lens_renderer)를 유지하면 OpenCV 잔존 + '픽셀이 경계를 넘지 않는다' 목표와 충돌하는 유일한 경로로 남는다. TFLite는 mediapipe_detector.cpp 1파일 격리(가드 26곳)·CMake PRIVATE 링크라 깔끔히 빠지고, GLES/EGL은 GPU 코어 본질 의존으로 유지된다.

**교차 검토 보충(Codex, §11)**: OpenCV 완전 제거 목표 시 Android JNI 빌드 스크립트도 수정 범위에 명시 포함해야 한다 — `android/iris-sdk/src/main/cpp/CMakeLists.txt`가 OpenCV를 REQUIRED로 탐색·링크한다 (:83 OpenCV_DIR 설정, :136 `find_package(OpenCV REQUIRED COMPONENTS core imgproc)`, :156 링크. 직접 재확인 완료).

### 6.6 교체 난이도 견적

- **작업/공수**: ① C 경계 4함수(iris_set_landmarks/set_lens/set_beauty/iris_render) + 478점→IrisResult 파생 어댑터(홍채 중심·반경, EAR 기반 eyelid_ratio, face_rect): 3~4일(최대 난점, 좌표 규약 실기기 검증 포함) ② avg_iris_luma GPU self-measure 이전(기존 W6 이관 계획 활용) 또는 주입 필드화: 1~2일 ③ 추적 코드 제거(mediapipe_detector/inference_thread/frame_processor) + sdk_api/CMake/JNI/Java 표면 정리: 2~3일(격리 양호, 기계적) ④ MediaPipe Tasks FaceLandmarker 데모 통합 + mirror/rotation/정규화 규약 정합: 2~3일 ⑤ 테스트 재편(추적 의존 12파일 제거, 어댑터 단위 테스트 신설): 2일 ⑥ 실기기 품질 회귀(luma 블렌드, visibility/blink, stabilizer): 2일 — 합계 12~16 사람·일(1인 개발 기준 약 3주)
- **리스크**: (1) 좌표 규약 차이: 현 detector는 detectOnlyWithRotation에서 회전 보정 후 좌표를 반환하는 규약 — MediaPipe Tasks의 출력 좌표계(회전/미러 처리 위치)와 정합하지 않으면 렌즈 위치가 전부 어긋남. 실기기 검증 필수. (2) detector 메타 소실: confidence/eyelid_ratio/iris_quality가 사라지면 visibility 게이팅·blink 처리 품질 회귀 — EAR 재계산으로 대체하되 튜닝 재검증 필요. (3) avg_iris_luma 실측 소실(P7-W2 default ON 상태): fallback 상수 체인으로 후퇴하면 TintLinearV2 품질 회귀 — W6 이관분 self-measure를 경계 도입과 동시 처리해야 함. (4) DetectionSlot torn-read race가 정식 주입 경계로 승격되며 표면화 — 승격 전 수정 필수. (5) C/C++ IrisResult reinterpret_cast 수동 동기화 위에 새 경계 타입을 얹으면 ABI 취약 누적.
- **요약**: GPU 렌더/뷰티/워프 경로가 이미 '텍스처 ID + IrisResult 포인터' 주입형이고 JNI DetectionSlot이 주입 패턴의 실증 프로토타입이므로, 경계 도입(③-1)은 신규 설계가 아니라 기존 경계의 공식 승격이다. 난이도 핵심은 추적 코드 삭제가 아니라 detector가 부수 생산하던 파생 데이터(홍채 중심/반경, eyelid_ratio→visibility, avg_iris_luma 실측, 회전 보정 좌표 규약)를 코어 측 어댑터로 이식·검증하는 작업

**판정 패널의 견적 비판:**

- **패널 A(보수파)**: 결합도 조사의 12~16 사람·일(1인 3주) 견적에 대한 비판적 평가: 방향은 타당하나 2개 항목이 과소, 1개 전제가 미계상이다. 타당한 부분 — TFLite 격리(mediapipe_detector.cpp 단일 파일, CMake PRIVATE 링크)와 seam 동형성(sdk_api_v2 주입형 + DetectionSlot 프로토타입)은 적대 검증까지 통과한 사실이므로 '③ 추적 제거 2~3일(기계적)'과 '① C 경계 4함수 신설'의 골격 산정은 신뢰할 수 있다. 과소 평가 1 — ①의 좌표 규약 실기기 검증 3~4일: findings 기준 정합시켜야 할 규약이 최소 4종(회전 후 좌표 공간 계약 미문서화, z 스케일 모델 raw vs 이미지 비례, left/right 명명 canonical 반전 3곳, 미러링 규약 호출자별 상이+허위 주석)이고, 검증 수단이 1인 실기기 육안 루프인데 데모 자체가 torn 스냅샷·OverlayView 독립 정책·KT 무음 폴백으로 오염돼 있어 '어긋남이 실제 결함인지 시각화 race인지 구분 불가'(round2 finding 명시) — 데모 정화 없이는 검증 루프 1회당 비용이 비결정적이다. 과소 평가 2 — ⑤ 테스트 재편 2일: 현재 골든 단정이 '이미지 2배 범위면 통과' 수준으로 무의미하고 렌더 픽셀 검증이 0건이라, '어댑터 단위 테스트 신설'이 아니라 골든 비교 인프라 신축(현 detector 출력 덤프 + 픽셀/좌표 동등성 하니스)이 실제 작업량이다. 미계상 전제 — 데모 검증 통로 정화(스냅샷 불변화, fallback 제거, OverlayView 정책 통일)는 견적 어디에도 없는데 ①⑥의 실기기 검증이 전부 이것에 의존한다: +2~3일. 또한 beauty_roi_manager 478 미만 silent false/OOB와 grid_mesh 468 하드코딩은 '주입 계약 도입 즉시' 터지는 지뢰라 ①에 +1일의 계약 가드 작업이 묶여야 한다. 보수파 견적: 데모 정화 2~3일 + 골든 베이스라인 구축 2~3일을 선행 단계로 명시 분리하고, 본 전환 13~17일 — 합계 17~23 사람·일(1인 기준 4~5주). 단축 조건 2가지를 채택하면 하한(17일)에 수렴 가능: (a) CPU 렌더/뷰티 경로를 이식하지 않고 deprecate(소비처 0 실측 — lens_renderer 데모 호출 0건 확인), (b) LensSimulator FaceTracker.kt/CoordMapper.kt를 신규 작성 대신 이식(④의 2~3일 중 1일 절감 + 단위 테스트 동반 확보). 원 견적 12~16일은 '골든 베이스라인이 이미 있고 데모가 신뢰 가능하다'는 성립하지 않는 전제 위에서만 달성 가능하다.

- **패널 B(부채파)**: 결합도 조사의 12~16 사람·일 견적은 'seam 동형성' 진단(GPU C API가 이미 주입형, DetectionSlot이 실증 프로토타입)은 정확하나, 부채파 관점에서 세 가지를 과소 계상했다. (a) ⑤ 테스트 재편 2일은 비현실 — 현재는 '재편'할 테스트가 없다. 렌더 픽셀 검증 0건, 좌표 회귀 0건, 골든 단정은 2배 범위 허용(test_mediapipe_detector_integration.cpp:342-363)이라 전환 전/후 동등성 증명의 출발점부터 신설해야 하고, 골든 비교 인프라(렌더 출력 캡처+비교) 구축은 3~5일이다. (b) 검증 통로 수리 미포함 — 이 프로젝트의 합격 판정이 실기기 육안인데 demo의 torn read·OverlayView 독립 정책·무음 fallback을 먼저 고치지 않으면 ⑥ 실기기 품질 회귀 2일의 판정 자체가 신뢰 불가. +1~2일. (c) geometry 수식 결함(EAR/마스크 좌표계/478 계약/반경 규약)을 전환 범위 밖으로 둔 견적인데, 주입 계약 명문화가 이 좌표 공간들을 어차피 건드리므로 분리하면 실기기 검증 이중 지불 — 같이 하면 +2~3일, 미루면 부채 복리. 반면 과대 계상 상쇄 요인도 있다: ④ MediaPipe Tasks 데모 통합 2~3일은 LensSimulator FaceTracker.kt/CoordMapper.kt 직접 이식(단위 테스트 포함, skin_mask_geometry 이식 전례 확인)으로 하한에 수렴 가능하다. 보정 견적: 최소 경로(geometry 수리를 후속 분리, 골든은 최소 스모크) 14~17 사람·일, 권고 경로(골든 인프라 + 검증 도구 수리 + geometry 수식 동시 수리) 18~24 사람·일 — 1인 개발 기준 약 4~5주. 원 견적 12~16일은 '성공 경로만 밟았을 때의 하한'이지 기대값이 아니다.

## 7. 모듈별 판정표

| 모듈 | 패널 A(보수파) | 패널 B(부채파) | 일치 |
|---|---|---|---|
| tracking (추적) | 재작성 | 재작성 | ✅ |
| orchestration (오케스트레이션/C API) | 부분 재작성 | 부분 재작성 | ✅ |
| geometry (지오메트리/안정화) | 부분 재작성 | 부분 재작성 | ✅ |
| gpu-render (GPU 렌더/이펙트) | 리팩토링 | 리팩토링 | ✅ |
| cpu-render (CPU 렌더/이펙트) | 부분 재작성 | 재작성 | ❌ |
| infra (인프라/유틸) | 리팩토링 | 리팩토링 | ✅ |
| android-binding (바인딩) | 부분 재작성 | 부분 재작성 | ✅ |
| demo-gl (데모 GL — 유일한 검증 통로) | 부분 재작성 | 부분 재작성 | ✅ |

### 모듈별 근거 (패널별)

**tracking (추적)**
- A(재작성): 보수파 입장에서도 인정할 수밖에 없는 케이스다. 단 '재작성'의 실체는 코어 내 재구현이 아니라 외부 교체(MediaPipe Tasks FaceLandmarker)다. 근거: (1) 이 모듈은 MediaPipe 그래프의 수제 TFLite 재구현인데 canonical 대비 이탈(입력 정규화, 디코딩 순서, z 스케일)이 누적돼 있고 Eye Refiner는 활성화 자체가 불가능한 상태(좌표 폭주) — '동작 중인 코드'가 아니라 '우연히 중앙 정면 얼굴에서 견디는 코드'다. (2) blocker 2건이 모두 이 모듈의 수명 경로에 있다. (3) LensSimulator FaceTracker.kt(/Volumes/M3-P31/Projects/TedMong/CGG/LensSimulator/sdk/android/lenssdk/src/main/kotlin/com/cgg/lenssdk/internal/FaceTracker.kt)를 직접 확인한 결과 GPU delegate 스레드 친화성·CPU 폴백·rowStride 압축까지 처리된 프로덕션 수준 외부 추적기가 이미 존재한다. 단 보수파 조건: 2단계 골든 베이스라인(전환 전/후 좌표 동등성 비교 기준)이 확보될 때까지 이…
- B(재작성): 외부 교체(MediaPipe Tasks)가 곧 재작성이며, 수리는 손해다. 부채파 관점의 결정적 논거: 이 자체 추적기는 canonical MediaPipe 대비 '구현 결함이 누적된 비자산'이다 — BlazeFace 박스 디코딩 좌표 순서 반전(mediapipe_detector.cpp, canonical reverse_output_order 미준수), 입력 정규화 [0,1] vs canonical [-1,1](:1049), Eye Refiner 픽셀 좌표를 ROI 비율로 오용해 활성화 시 좌표 붕괴(:2066-2073), shouldRunEyeRefiner가 미설정 confidence 참조(:2910), 무검증 flatbuffer 로드 크래시(blocker), 손상 v2→468 무음 다운그레이드. 운영 골격도 blocker급 — inference_thread.cpp:23-25 joinable 미회수 std::terminate, :62 Starting 중 stop 시 영구 행, detectSync 타임아웃 후 영구 고착(:153-191). iris_detector.cpp는 전 팩토리가 nullptr 반환하는 죽은 코드로 Strategy 추상이 이미 붕괴. 3…

**orchestration (오케스트레이션/C API)**
- A(부분 재작성): 절반은 삭제, 절반은 승격·정리다. frame_processor.cpp는 검출/NV21·NV12 변환/비동기 submitFrame이 본체라 추적 외부화 시 사실상 전체가 제거 범위(공개 헤더의 cv::Mat 노출 frame_processor.h:20,183도 함께 해소). sdk_api.cpp의 detect 계열 함수군과 sdk_manager의 createDetector도 삭제 대상. 반면 sdk_api_v2.cpp는 직접 확인된 대로 이미 '텍스처 ID + IrisResult*' 주입형이고 SDKManager/detector 무참조로 동작해(seam 검증 결과 iris_sdk_init 없이 init_gpu_lens만으로 동작) 목표 아키텍처와 동형 — 이 표면은 버리는 게 아니라 정본으로 승격해야 한다. 다만 승격 전 reinterpret_cast 가드 강화(static_assert 필드 단위), 에러 코드 정본 단일화, 기본값 단일 소스화, iris_set_landmarks 신설이 필요해 '리팩토링'을 넘는 표면 재설계가 불가피하다. 전면 재작성 주장에는 반대: V2의 함수 골격과 전역 상태 관리는 그대로 쓸 수 있고, 재작성하면 P7-W2(avg_iri…
- B(부분 재작성): 절반은 삭제, 절반은 정본화 재설계다. frame_processor는 검출/NV21·NV12 변환/비동기 submitFrame 전부가 추적 측이라 GPU-only 전환 시 파일 전체 소멸(제거 범위 확정 사항) — cv::Mat 공개 노출(frame_processor.h:20,183)도 함께 사라진다. 잔존 표면은 리팩토링으로 못 고치는 구조 문제가 밀집: C API가 3개 표면으로 분산(sdk_api.h + beauty_filter.h V2 config 이중화 + 헤더 미선언 internal 9종), 동일 NotInitialized가 v1=100/v2=501 이중 변환(정본 부재), BeautyFilterConfigV2 기본값 3원 분기, C/C++ IrisResult reinterpret_cast의 sizeof-only 가드(sdk_api_v2.cpp — 랜드마크 주입 API가 정확히 이 지점을 건드림), iris_sdk_set_eye_refiner_policy no-op이 OK 반환(sdk_api.cpp:1198-1203), feather_radius dead field, 비동기 API 13종이 Android에서 죽은 코드. 847줄 V2 표면 테스트 …

**geometry (지오메트리/안정화)**
- A(부분 재작성): 모듈 내부가 극명하게 갈린다. 살릴 것: one_euro_filter.h는 4개 소비처(stabilizer/GPU렌더러 2종/뷰티)를 가진 검증된 핵심이고, skin_mask_geometry.cpp는 직접 읽어본 결과 LensSimulator BeautyGeometry.kt에서 값 변경 없이 이식된 깨끗한 순수 함수다(P8-W1) — 이 둘은 유지. temporal_stabilizer는 결함 3건이 모두 '단위/공간 전제' 한 계열이라 픽셀 공간 통일 + 설정 기본값 수정으로 국소 치유 가능 — 리팩토링 수준. 재작성이 필요한 것: beauty_roi_manager는 좌표 계약 자체가 파괴된 상태(래스터화는 전 프레임 공간, 소비측 설계는 ROI-로컬, 보정 함수는 정의만 있고 미호출)이고 미러링 규약이 호출자마다 다르며 주석이 허위 — 패치 누적으로 못 고치고 좌표 계약을 처음부터 다시 써야 한다. warp/grid_mesh는 핵심 목적(자연스러운 falloff)의 수학이 성립하지 않고 478 하드코딩 OOB까지 있는데 프로덕션 미연결이라 보존 가치가 낮다 — 이펙트 코어 이식 시점에 RBF부터 재작성. 종합하면 모듈의 ~60%(beauty_roi_mana…
- B(부분 재작성): 전환 후 코어의 두뇌인데 결함이 '구조'가 아니라 '수식 자체'에 있어 구조 보존형 리팩토링으로는 못 고친다. (1) temporal_stabilizer: EAR을 정규화 좌표로 계산해 임계 0.2가 종횡비 종속(:323-338, 세로 입력에서 blink 오판→좌표 동결), outlier rejection이 기본값에서 완전 no-op(:255-265), radius OneEuro beta 7.5를 픽셀 단위에 적용해 스무딩 무효 — 함정 #5/#10의 정확한 사례 3연타. (2) beauty_roi_manager: 마스크 좌표계 계약 파괴(landmarkToMaskCoord 정의만 있고 호출 0건 — 래스터화는 전 프레임, 소비는 ROI-로컬), LIPS_INDICES 아랫입술만(뷰티 제품에서 윗입술 뭉갬), 478 미만 silent false + count 없는 public API의 인덱스 466 무검증 OOB. (3) grid_mesh: RBF 보간이 zero-변위 컨트롤 포인트 제외로 falloff 수학 불성립 + 468/478 하드코딩 충돌(:82 vs face_warp_controller.cpp:284). (4) eye_render_packet_ad…

**gpu-render (GPU 렌더/이펙트)**
- A(리팩토링): 이 프로젝트의 축적 가치 그 자체이며, 재작성 주장이 가장 위험한 모듈이다. 근거: (1) OpenCV 의존 0(grep 0건, GLES/EGL만), 이미 '텍스처+IrisResult*' 주입형으로 목표 아키텍처에 부합 — 구조 문제가 없다. (2) major findings를 분류하면 전부 '임베딩 계약'(상태 복원, 컨텍스트 수명, 풀 크기 정책, 해상도 상한) 아니면 '국소 인덱스/수식 버그'(caruncle 인덱스 반전, 타원 aspect 보정)다. 어느 것도 셰이더 파이프라인 구조나 패스 설계를 무효화하지 않으며, 각각 save/restore 블록·풀 반환 경로·인덱스 스왑·보정항 추가로 제자리 수정 가능하다. (3) 반대로 재작성하면 P5~P8에 걸친 실기기 육안 검증(TintLinearV2 K=0.85 튜닝, eyelid 마스크, P8-W1 skin smoothing)이 전부 무효가 되는데, 사용자의 검증 방법론이 실기기 체감 1인 판정이라 이 재검증 비용이 가장 비싼 프로젝트다. 회귀 위험 대비 이득이 없다. 단 리팩토링의 1순위는 '호스트 컨텍스트 계약 명문화'(상태 복원 + 컨텍스트 손실 통지 API + 크기 상한 제거)로, 코어가 범용 렌…
- B(리팩토링): 부채파로서 회의적으로 검증했으나 증거가 리팩토링 충분을 지지한다. findings 17건의 분포가 결정적: major 6건 전부가 수명/상태 관리 계층(호스트 GL 상태 미복원, TexturePool 1920x1080 하드코딩 상한, applyTexture 풀 고갈, '큰 텍스처 재사용' 크기 가정 충돌, 컨텍스트 차용 'current' 암묵 가정, 자체 EGL 모드의 바인딩 강탈)이고 렌더 수학·셰이더 본체 결함은 국소 2건(타원 aspect 미보정 전단, 내외안각 인덱스 반전 — 둘 다 수 라인 수정)뿐이다. OpenCV include 0건(grep 확인), 이미 '텍스처 ID + IrisResult*' 주입형 표면, Phase 6~8 실기기 육안 벤치가 누적된 코드베이스 유일의 검증 자산 — 셰이더 1232줄과 블렌드 모드 튜닝(TintLinearV2 K=0.85 등)은 재작성하면 검증 이력이 통째로 증발한다. 단 리팩토링 범위는 작지 않다: '호스트 컨텍스트 차용 렌더 코어'가 1급 계약이 되므로 GL 상태 save/restore 래퍼, 컨텍스트 수명 통지 API, 풀 크기 정책 재설계는 신규 계층 추가 수준이다. ES3.0 폴백(전 셰이더가 #ver…

**cpu-render (CPU 렌더/이펙트)**
- A(부분 재작성): 보수파지만 이 모듈의 '동작 중인 코드의 가치'는 실측으로 부정된다. 직접 확인 결과 데모의 렌즈 경로는 GPU 텍스처 경로(CameraGLRenderer.kt:1248 renderLensTexture)뿐이고 CPU lens_renderer 소비처는 0 — '항상 45° 잘못 회전'이라는 결함이 출하된 채 아무도 발견 못 한 것이 그 증거다(검증된 적 없는 코드). CPU 뷰티 ROI도 마스크 오정렬(major)인데 Java가 detectionPtr=0을 넘겨 우연히 회피 중 — 즉 ROI 기능은 사실상 한 번도 올바르게 동작한 적이 없다. 1순위 권고는 수리가 아니라 범위 결정이다: 새 코어를 GPU 전용으로 좁히면(잔존 의존성 조사대로 OpenCV 완전 제거 가능) 이 모듈은 deprecate가 합리적이고, 그 경우 비용은 0이다. 외부 SDK로서 CPU 폴백이 계약상 필요하다고 판단될 때만 '부분 재작성'을 집행한다: lens_renderer 회전 산출과 cpu_beauty ROI 좌표 경로는 재작성, fast_guided_filter·beauty_filter 본체 알고리즘은 유지, beauty_processor는 삭제. 판정을 '부분 재작성'으로 두되…
- B(재작성): 실질 권고는 '코어에서 폐기, 필요 시 추후 별도 모듈로 재작성'이다. 근거: (1) lens_renderer.cpp는 iris[3]/iris[4]를 좌/우 쌍으로 오인해 렌즈 회전이 항상 ~45° 왜곡되는 상시 발현 major — '잠복'이 아니라 이 경로의 출력 자체가 늘 틀렸고, GPU 경로와 blendMode 의미도 불일치해 결과 동등성이 애초에 없다. (2) cpu_beauty_backend ROI 모드는 풀프레임 마스크를 ROI 사각형에 리사이즈하는 좌표계 오정렬(:756-826)로 눈/입술이 블러되는 잘못된 결과. (3) beauty_processor.cpp는 소비자 0 죽은 코드. (4) V1 기본값 C++ 0.5/0.3 vs Java 0.0/0.0 분기. (5) 구조적으로 cv::Mat이 공개 헤더에 박혀(cpu_beauty_backend.h:13, lens_renderer.h:156) '코어는 픽셀 경계 비통과' 목표와 정면 충돌하는 유일한 경로이며, 이 경로를 들어내야 OpenCV 완전 제거가 가능하다(결합도 조사 확인 사항). 1,724줄(lens_renderer+cpu_beauty)을 고쳐서 얻는 것은 'GPU 불가 기기 폴백'인데 2…

**infra (인프라/유틸)**
- A(리팩토링): findings가 적은 이유를 직접 확인했다 — '아무도 안 봐서'가 아니라 '죽은 코드라서'다. quality_metrics/ab_compare/release_gate/param_tuner/profiler/buffer_pool은 소비자 0으로 전수 확인됐고(round2 finding), 이들의 처분은 재작성이 아니라 삭제다 — CMakeLists에서 빼면 끝이라 리팩토링 분류가 맞다. 살아있는 부분: lens_sku_metadata는 sdk_api_v2가 소비 중(직접 grep 확인)이고 결함 보고 0 — 유지. types.h는 이 전환의 진짜 작업 지점이다: IrisResult가 C++/C/JNI/Java 4중 표현으로 전 레이어를 왕복하는 계약의 정본인데, 추적 외부화 시 DetectorType·EyeRefinerPolicy 삭제, detector 메타 필드(confidence 감쇠 누적, z 스케일 비례) 의미 재정의, 478점 주입 계약(홍채 5점+visibility 포함 여부) 명문화가 필요하다. 이는 구조 변경이 아닌 계약 문서화+필드 정리이므로 리팩토링 범위.
- B(리팩토링): 구조 재설계가 아니라 '대량 삭제 + 계약 명문화'라 리팩토링으로 충분하다 — 단, 삭제가 작업의 본체다. QA 4종(quality_metrics/ab_compare/release_gate/param_tuner)+profiler+beauty_processor 합계 1,982줄이 소비자 0인 채 visibility hidden 미적용으로 릴리즈 .so에 export 심볼로 잔류(CMakeLists.txt 확인 finding) — 전환 인벤토리를 오염시키므로 1순위 삭제. buffer_pool은 미사용 + move-assign ABBA 잠재 데드락으로 삭제. 살릴 것은 둘: lens_sku_metadata는 sdk_api_v2.cpp/gpu_lens_renderer.h가 실소비하는 산 코드임을 직접 확인했고(findings가 적은 이유가 '단순해서'임을 검증), types.h는 IrisResult 4중 표현(C++/C/JNI/Java)의 정본으로서 전환의 심장이다. types.h에는 DetectorType/EyeRefinerPolicy 제거 + z 스케일 규약(현재 crop 종속 raw 값, MediaPipe 표준과 불일치), 홍채 boundary[1..4] 시…

**android-binding (바인딩)**
- A(부분 재작성): 전환에서 시그니처가 가장 크게 바뀌는 경계(좌표 흐름이 SDK→앱에서 앱→SDK로 역전)인 데다, 현 주입 채널의 결함이 패치로 안 되는 계약 수준이다: DetectionSlot은 generation이 Java에 미노출이라 torn-read 방지가 구조적으로 불가능하고, 마샬링은 홍채 boundary 5점과 visibility를 운반할 수 없는 손실 채널이며, faceMesh 복사 실패를 silent skip하면서 valid=true를 유지한다 — 이 채널이 1급 주입 API로 승격되면 minor들이 major로 일괄 승격된다(findings 명시). 따라서 마샬링/슬롯 계층과 Kotlin 표면은 주입 스키마(478점+홍채5점+visibility+timestamp+generation) 기준으로 재설계가 맞다. 반면 보수파로서 지킬 것: JniCache 인프라, 텍스처 패스스루 JNI(렌더/뷰티 호출부), 에러 코드 변환 골격은 동작 검증된 부분이라 유지하고, blocker인 모델 캐시 고착은 추적 외부화로 모델 추출 자체가 사라져 자연 해소된다(고치지 말고 삭제). consumer-rules/16KB는 코드가 아닌 패키징 수정 1일거리지만 배포 전 필수. 종…
- B(부분 재작성): 전환에서 가장 크게 바뀌는 경계(좌표 배열 방향이 SDK→앱에서 앱→SDK로 역전)인데 현 상태가 그 역전을 감당 못 한다. 주입 채널 자체의 결함: DetectionSlot generation 검증이 '호출측 수행' 주석과 달리 Java에 미노출되어 이행 불가능한 계약(iris_jni.cpp:559-566, torn read), copyResultFromJava가 홍채 boundary[1..4]+visibility를 복사하지 않는 손실 채널, copyResultToJava의 faceMesh 복사 실패 silent skip(faceMeshValid=true 유지 — 주입이 1급 입력이 되면 major 승격 예고됨). 구조 결함: internal 9종이 헤더 없이 JNI에 수동 extern 중복 선언(시그니처 드리프트 무검출), GetFieldID 31연쇄 pending exception UB, consumer-rules.pro keep 규칙 전무(minify 소비자 앱에서 JNI_OnLoad 확정 실패 — 배포 blocker급), 16KB 페이지 정렬 미조치, 에러코드 500/501 전 표면 미매핑. Kotlin 표면은 faceMesh/avgIrisLuma …

**demo-gl (데모 GL — 유일한 검증 통로)**
- A(부분 재작성): 이 프로젝트의 검증 방법론(실기기 육안 1인 판정)을 데모가 스스로 오염시키고 있다는 점이 결정적이다: torn 스냅샷, 마커와 렌더가 다른 좌표를 그리는 OverlayView 정책, 'SDK 결과인지 KT fallback인지' 보증 불가한 무음 폴백 — 이 상태로는 2단계 골든 비교든 3단계 동등성 검증이든 데모를 신뢰 기준으로 쓸 수 없다. 재작성 범위: (1) 검출 결과 데이터 흐름을 불변 스냅샷+timestamp 페어링으로 재설계, (2) KT fallback 렌즈 셰이더 제거(단일 렌더 경로 — w9-demo-ui-sync 잔여와 일치), (3) OverlayView 정책 제거(GL과 동일 데이터·동일 게이트). 유지 범위: COVER 변환 수식은 검증 완료로 동치 판정이 이미 났고, EGL/SurfaceTexture 골격은 수명 처리 수정으로 충분하다. 결정적 재활용 기회: 3단계에서 검출이 MediaPipe Tasks로 바뀌면 FrameAnalyzer/yuvToNv21/detectWithRotation 호출부는 통째로 사라지고 LensSimulator FaceTracker.kt(RGBA_8888 직접 입력, stride 처리 완료)+Trackin…
- B(부분 재작성): 이 프로젝트의 검증 방법론이 '실기기 육안 체감'인데 그 유일한 검증 도구가 스스로 오염돼 있다 — 공유 IrisResult 인스턴스의 torn read(GL/UI 스레드 동시 읽기 중 Analyzer 덮어쓰기), OverlayView의 독립 confidence 게이트(0.5)+2초 홀드로 GL 출력의 ground truth 역할 불능, 미러 시 GL은 L/R 스왑·Overlay는 무스왑으로 육안 판정 교차 오염, KT fallback 셰이더의 blendMode 의미 불일치+프레임 단위 무음 폴백으로 '지금 보는 게 SDK 결과인지' 보증 불가, yuvToNv21 rowStride 미처리, Preview/Analysis 별도 스트림에 timestamp 페어링 부재. 재작성 대상은 추적/입력/검증 계층이고, 여기에 LensSimulator 자산이 직접 꽂힌다: FaceTracker.kt는 rowStride 압축 복사·GPU delegate 스레드 친화성·CPU 폴백을 이미 해결한 MediaPipe Tasks LIVE_STREAM 구현(파일 직접 확인)이고, CoordMapper.kt의 uprightToSensor/sensorToUpright는 단위 테스트 딸린…

## 8. 리팩토링 vs 재작성 — 패널 권고 전문

### 패널 A (보수파)

전체 권고: 전면 재작성 반대, '경계 승격형 리팩토링'(모듈 2건 리팩토링 + 5건 부분 재작성 + tracking만 외부 교체) — 단, 순서가 생명이다. 보수파 핵심 논거 3가지. (1) 재작성이 버리게 될 자산이 이 프로젝트에서 가장 비싼 종류다: gpu-render 5,800줄은 P5~P8 실기기 육안 검증(TintLinearV2 튜닝, eyelid 마스크, P8-W1 skin smoothing)의 결정체이고, 검증 방법이 1인 실기기 체감이라 재작성 시 이 검증을 전부 반복해야 한다 — 자동화된 픽셀 테스트가 0건(testing blocker)이라 재작성의 회귀 감지 수단도 없다. 재작성은 '안전망 없는 절벽에서 점프'다. (2) 목표 아키텍처가 이미 코드에 존재한다: sdk_api_v2의 IrisResult* 주입형 함수군과 JNI DetectionSlot은 seam 적대 검증에서 구조 유효 판정을 받았고, 전환은 신규 설계가 아니라 기존 경계의 공식 승격이다. 재작성 주장은 이 사실 앞에서 근거를 잃는다. (3) 결함 분포가 재작성을 지지하지 않는다: blocker 4건 중 3건(InferenceThread terminate, 모델 무검증 로드, 모델 캐시 고착)이 전부 제거될 tracking 측에 있어 외부화로 자연 소멸하고, 코어 측 major는 계약 명문화·인덱스 수정·좌표 공간 통일로 제자리 치유 가능한 계열이다. 예외적으로 재작성이 정당한 곳은 증거가 압도적인 두 곳뿐: beauty_roi_manager 좌표 계약(보정 함수 정의만 있고 호출 0건 — 설계 자체가 미완성)과 warp RBF 수학(목적 함수가 수학적으로 불성립 + 프로덕션 미연결이라 보존 가치 0). 2단계와의 상호작용: 골든 베이스라인은 현 detector 출력을 기준으로 만들어야 하므로 tracking 삭제는 반드시 베이스라인 확보 후순위다 — 권장 순서는 [데모 검증 통로 정화(demo-gl 스냅샷/폴백 제거) → 골든 캡처(현 detector 기준 478점+파생 데이터 덤프) → seam 승격(iris_set_landmarks + types.h 계약 명문화: 회전 후 좌표·z 스케일·left/right 명명·미러 규약 4종이 findings에서 전부 모호 판정) → 3단계 추적 교체 → tracking 삭제]. 3단계와의 상호작용 및 LensSimulator 재활용: FaceTracker.kt는 외부 추적기 요건(LIVE_STREAM, GPU delegate 스레드 친화성, CPU 폴백, stride 처리)을 이미 충족한 프로덕션 코드이고, CoordMapper.kt/IrisGeometry.kt/OneEuroFilter.kt는 단위 테스트가 동반된(IrisLensSDK에 없는 것) 검증 자산이다. skin_mask_geometry.cpp가 BeautyGeometry.kt의 '값 변경 없는 이식' 선례로 이미 성공했으므로 같은 방식의 이식 신뢰도가 실증돼 있다. 특히 left/right 명명이 MediaPipe canonical과 반전된 현 코어(3개 findings)에 외부 표준 명명 좌표를 주입할 때, LandmarkIndices.kt가 명명 충돌 해소의 정본 역할을 할 수 있다.

### 패널 B (부채파)

전면 재작성도, 전면 리팩토링도 아닌 '비대칭 외과수술'을 권고한다: GPU 렌더 코어(유일한 검증 자산)는 리팩토링으로 보존하고, 추적·CPU 픽셀 경로는 폐기·교체하며, 지오메트리·바인딩 경계는 부분 재작성한다. 부채파의 핵심 논거 4가지. (1) 자체 추적기는 보존 가치가 없는 부채다 — BlazeFace 디코딩 순서/정규화가 canonical과 어긋나고 Eye Refiner는 활성화 즉시 좌표가 붕괴하며 inference_thread는 평범한 에러 경로에서 std::terminate한다. 이것을 '리팩토링'하는 것은 canonical MediaPipe를 결함 있게 재구현한 3,425줄을 계속 소유하는 선택이고, MediaPipe Tasks 교체(3단계)가 곧 수리다. 따라서 3단계는 2단계와 분리할 이유가 없다 — 주입 경계가 생기는 순간 MediaPipe Tasks가 첫 외부 공급자가 되어야 하며, 자체 추적기를 '주입 형태로 감싸 한동안 병행'하는 중간 단계는 결함 추적기의 수명만 연장한다. (2) LensSimulator 검증 자산의 재활용 가치가 재작성 비용을 구조적으로 낮춘다 — FaceTracker.kt(MediaPipe Tasks LIVE_STREAM + rowStride 압축 + GPU 폴백: demo-gl의 stride 결함과 추적 통합을 동시에 해결), CoordMapper.kt(uprightToSensor/sensorToUpright: 이번 전환 최대 리스크인 좌표 규약의 단위 테스트 딸린 해법), TrackingSnapshot(불변 스냅샷: torn read 계열 3건의 구조적 해법), IrisGeometry.kt(478점→홍채 중심/반경 파생: 코어 어댑터의 참조 구현). 결정적으로 skin_mask_geometry.cpp:5가 'LensSimulator 값 무변경 이식' 패턴의 성공 전례를 이미 증명했다. (3) 2단계 골든 베이스라인은 현 상태로는 구축 불가능하다 — 렌더 픽셀 검증 0건(blocker), 좌표 변환 회귀 테스트 0건(blocker), 기존 골든 단정은 2배 범위 허용으로 무의미, 게다가 육안 검증 통로(demo)는 torn read·독립 게이트·무음 폴백으로 오염돼 있다. 골든 인프라 신설 + demo 검증 도구 수리가 전환의 선행 조건이며, 이를 건너뛰면 전환 전/후 동등성 주장 자체가 성립하지 않는다. (4) geometry의 수식 결함(EAR 종횡비, 마스크 좌표계, RBF, 반경 규약)은 주입 계약 정의와 같은 작업이다 — '코어가 받는 좌표의 기준 공간'을 명문화하는 순간 이 결함들을 어차피 건드리므로, 전환과 분리하면 실기기 검증을 두 번 치른다. 권고 실행 순서: ① 골든/검증 인프라 + demo 검증 도구 수리 → ② types.h 좌표·시맨틱 계약 명문화 + iris_set_landmarks 경계 승격(DetectionSlot 공식화, generation 포함) → ③ MediaPipe Tasks 통합(LensSimulator FaceTracker 이식)과 동시에 자체 추적기·CPU 픽셀 경로·죽은 코드 1,982줄 제거 → ④ geometry 수식 재작성(LensSimulator 수학 이식 패턴) + 실기기 회귀.

## 9. 종합 권고 (Codex 교차 검토 반영판 — §11)

양 패널이 **전면 재작성 기각, 전면 리팩토링도 기각**으로 수렴했다. 공통 결론:

1. **tracking(자체 TFLite 추적 3,425줄+)은 수리 대상이 아니라 교체 대상** — BlazeFace 디코딩/정규화의 canonical 불일치, Eye Refiner 좌표 붕괴, InferenceThread std::terminate 등 결함이 집중돼 있고, 3단계(MediaPipe Tasks 교체)가 곧 수리다. 패널 B는 "주입 경계가 생기는 순간 MediaPipe Tasks가 첫 외부 공급자가 되어야 하며, 자체 추적기 병행 기간을 최소화하라"고 권고.
2. **gpu-render(~5.8k줄)는 유일하게 실기기 검증이 누적된 자산 — 리팩토링 보존** (양 패널 일치). P5~P8 육안 튜닝의 결정체이며 재작성 시 회귀 감지 수단(픽셀 테스트 0건)이 없다.
3. **목표 아키텍처의 seam은 이미 코드에 존재** — sdk_api_v2의 IrisResult* 주입형 함수군 + JNI DetectionSlot이 적대 검증에서 구조 유효 판정. ③-1(경계 도입)은 신규 설계가 아니라 기존 경계의 공식 승격이다.
4. **유일한 판정 이견: cpu-render** (A: 부분 재작성 / B: 폐기·교체). "프레임 픽셀은 경계를 넘지 않는다"는 목표 아키텍처를 ADR에서 확정하면 B안(폐기)이 정합적이다. 단 Codex 교차 검토 지적대로, 내부 소비처가 0이어도 `iris_sdk_render_lens` 등 **공개 C API가 살아 있으므로 삭제는 기술 판단이 아니라 API 호환성·버전 정책 판단**이다 — 2단계 ADR에서 "공개 CPU 픽셀 API의 deprecation/메이저 버전 정책"을 별도 결정 항목으로 다룰 것.
5. **2단계 골든 베이스라인은 "구축"이 아니라 "신축"** — 렌더 픽셀 검증 0건, 좌표 회귀 0건, 기존 골든 단정은 2배 범위 허용(blocker 2건). 또한 실기기 육안 검증 통로인 데모가 torn read·독립 게이트·무음 폴백으로 오염돼 있어, **데모 검증 통로 정화가 모든 단계의 선행 조건**이다.

**권장 실행 순서 (양 패널 종합):**
① 데모 검증 통로 정화(torn 스냅샷/무음 폴백 제거) + 골든 비교 인프라 신축 → ② 골든 캡처(현 detector 기준 478점+파생 데이터+렌더 출력) → ③ types.h 좌표·시맨틱 계약 명문화(회전 후 좌표/z 스케일/left-right 명명/미러 규약 4종) + iris_set_landmarks 경계 승격(DetectionSlot 공식화, generation 검증 포함) → ④ MediaPipe Tasks 통합(LensSimulator FaceTracker.kt/CoordMapper.kt 이식) + 자체 추적기·죽은 코드 제거 → ⑤ geometry 수식 수리(EAR/마스크 좌표계/반경 규약)를 주입 계약 작업과 병행 → ⑥ 실기기 회귀.

**견적**: 결합도 조사 원견적 12~16 사람·일(1인 ~3주)에 양 패널 보정(골든 인프라 신축 +3~5일, 데모 정화 +1~2일, geometry 병행 +2~3일, LensSimulator 이식으로 -1~2일 상쇄)을 반영하면 **약 17~24 사람·일(1인 4~5주)**. Codex 보정: 공개 CPU 픽셀 API의 호환 유지(점진 deprecation 경로)까지 포함하면 **+2~4일 버퍼** 권장. CPU 경로를 공식 deprecated 처리하고 LensSimulator 자산을 그대로 이식하면 하한(17일)에 수렴 가능.

## 10. 다음 단계

- 본 보고서 사용자 검토 → 승인 시 계획 문서 2단계(ADR 작성 + 골든 베이스라인) 착수
- 외부 교차 리뷰: Codex(gpt-5.5 xhigh) 1회 완료·반영 (§11). 필요 시 Gemini 추가 교차 리뷰 후 ADR 확정
- 2단계 ADR 결정 항목에 추가 (교차 검토 반영): ① cpu-render 폐기/별도 모듈화 + 공개 CPU 픽셀 API deprecation·메이저 버전 정책 ② 16KB 정렬은 빌드 산출물 실측 검증(`zipalign -c -P 16` / `objdump p_align`)으로 확정

## 11. 외부 교차 검토 — Codex (gpt-5.5 xhigh, 2026-06-11)

읽기 전용 정적 코드 대조 + MediaPipe 원본 설정(face_detection.pbtxt) 확인 기준의 독립 검토 (빌드/테스트 미실행). Claude의 비판적 검토(critical-review)로 분류·반영한 결과:

### 동의 — 감사 결과 독립 확인
- **blocker 2건 + 지정 표본 4건 전부 사실 확인**: InferenceThread joinable 미회수 terminate(inference_thread.cpp:23,55,315 + frame_processor.cpp:190), BlazeFace 입력 정규화 [0,1]·[yc,xc,h,w] 해석 vs canonical [-1,1]·reverse_output_order(mediapipe_detector.cpp:1049,1665 — MediaPipe 원본 pbtxt 대조), DetectionSlot torn-read(iris_jni.cpp:559,1881,1923 + IrisLensSDK.java:1515), IrisResult 이중 정의 sizeof 단일 가드(types.h:138, sdk_api.h:175, sdk_api_v2.cpp:41,666)
- **표본 major 6건 사실 확인**: Eye Refiner 픽셀 좌표의 ROI 비율 오용(:2066/:3020), shouldRunEyeRefiner 미설정 confidence 참조(:2910/:3148), TemporalStabilizer radius 픽셀·EAR 정규화 혼용(:275/:323), CPU ROI 마스크 좌표계 불일치(cpu_beauty_backend.cpp:756/beauty_roi_manager.cpp:233), CPU 렌더러 iris boundary 순서 가정(lens_renderer.cpp:130), consumer ProGuard keep 누락(consumer-rules.pro:14)
- §7 모듈 판정·§9 권고 방향·실행 순서·견적(17~24인일 "성공 경로 기대값")에 동의. cpu-render는 패널 B(폐기·별도 모듈화) 지지

### 반박 → 검증·반영 내역
1. **패널 A 근거 정오 (수용)**: §8 패널 A의 tracking 근거 (2) "blocker 2건이 모두 이 모듈의 수명 경로에 있다"는 부정확 — blocker 중 1건(InferenceThread)만 해당하고, 나머지 1건은 테스트 안전망 관점(좌표 회전 경로 회귀 테스트 공백)이다. 패널 인용문은 원문 보존하고 본 정오표로 정정한다. tracking 재작성 판정 자체는 나머지 근거(canonical 이탈, Eye Refiner 붕괴, 외부 대체재 존재)로 유지된다.
2. **cpu-render 공개 API 호환성 (수용)**: "내부 소비처 0 = 삭제 비용 0"으로 읽히지 않도록 §9-4 보강 — `iris_sdk_render_lens`(sdk_api.h:428, sdk_api.cpp:677, frame_processor.cpp:771)는 공개 표면이므로 삭제는 API 호환성·버전 정책 판단.
3. **16KB 단정 수위 (부분 수용)**: 본문(§1 표 52행 인근)은 이미 "로드 실패 **가능**"으로 가능성 표현이나, "Play 제출 거부/경고 대상"은 빌드 산출물 정렬 실측 전까지 잠정 판정으로 둔다 — §10에 실측 검증 항목 추가. (설정 누락 사실 자체는 양측 합의: AGP 8.5.0 + max-page-size 플래그 부재)

### 추가 발견 → 수용·반영 내역
- **JNI CMake의 OpenCV REQUIRED**: §6.5 보충에 반영 (android/iris-sdk/src/main/cpp/CMakeLists.txt:83,136,156 — Claude 재확인 완료)
- **공개 CPU 픽셀 API deprecation 정책의 별도 작업화**: §9-4·§10 ADR 결정 항목 + §6.6 견적 버퍼(+2~4일)에 반영

### 거부
- 없음 — Codex 지적 전 항목을 코드로 재확인한 결과 사실과 부합. (반박 1의 인용 라인 31-32는 blocker 목록 자체를 가리켜 다소 부정확했으나, 실제 문제 문구가 453행 패널 A 본문에 실재하므로 지적의 실체는 유효)

## 부록 A. 반박 폐기된 findings (8건)

- **GLESRenderContext::dumpTexture 미구현 — 실기기 GPU 렌더 골든 캡처 경로 부재** (major/테스트 안전망) — `cpp/src/gpu/gles_render_context.cpp`
  - 반박(downgrade): 인용 사실 자체는 정확하다: cpp/src/gpu/gles_render_context.cpp:471-479의 dumpTexture는 `LOGW("dumpTexture not implemented on Android"); return false;` 스텁이고, cpp/src/gpu/cpu_render_context.cpp:204-230은 cv::imwrite로 구현돼 있으며, android/ 전체(demo-app java + ir…
  - 반박(downgrade): 표면 사실(스텁)은 맞으나 주장 영향("GPU 경로 골든 캡처 수단 부재")은 과장으로 반박됨. 직접 확인 근거: (1) cpp/include/iris_sdk/gpu/render_context.h:180-193 — dumpTexture는 베이스 인터페이스부터 "디버깅용" 옵션 훅이며 기본 구현이 `return false; // 기본 구현은 미지원`. GLES 스텁(cpp/src/gpu/gles_render_context.cpp…
- **sdk_api_v2 전역 GPU 객체의 정적 소멸 시 GL 컨텍스트 부재 해제** (minor/스레드/수명) — `cpp/src/sdk_api_v2.cpp`
  - 반박(downgrade): 인용 자체는 정확하다: cpp/src/sdk_api_v2.cpp:44-49에 g_gpu_beauty/g_gpu_lens/g_sku_registry가 파일 스코프 static unique_ptr로 존재하고, GPUBeautyBackend::~GPUBeautyBackend()(cpp/src/gpu/gpu_beauty_backend.cpp:125-127)와 GPULensRenderer::~GPULensRenderer()(cpp/sr…
- **JNI 왕복에서 iris boundary 랜드마크(인덱스 1~4) 소실 → Android 경로 TemporalStabilizer outlier rejection 무력화** (major/정확성) — `android/iris-sdk/src/main/cpp/iris_jni.cpp`
  - 반박(downgrade): finding의 코드 인용 자체는 대부분 정확하나, 주장된 핵심 영향이 현재 코드에서 성립하지 않아 반박한다.

[사실로 재확인된 부분]
- iris_jni.cpp:243-253 copyResultToJava는 src.left_iris[0]/right_iris[0]+radius만 Java로 전달, 327-337 copyResultFromJava도 [0]만 복원 — 정확.
- IrisResult.java:76-115 leftIr…
- **QualityMetrics가 feathered(연속값) 마스크를 이진 의미로 사용 — 페더 밴드가 skin/non-skin 양쪽에 중복 포함** (minor/정확성) — `cpp/src/quality_metrics.cpp`
  - 반박(downgrade): 기계적 주장은 사실: quality_metrics.cpp:313-314의 bitwise_not + cv::mean(151-158)/meanStdDev(113)의 nonzero 마스크 의미론상, 연속값 마스크가 입력되면 페더 픽셀(1~254)이 skin/non-skin 양쪽에 중복 포함됨. beauty_roi_manager.cpp:203/588의 applyFeathering(GaussianBlur)로 연속값 combined_ma…
- **GridMesh::setControlPoints 컨트롤 포인트 충돌 가드 부재 — stale landmark→vertex 매핑** (minor/정확성) — `cpp/src/warp/grid_mesh.cpp`
  - 반박(downgrade): 코드 인용 자체는 정확하다: cpp/src/warp/grid_mesh.cpp:200-233의 setControlPoints에는 충돌 가드가 없고(나중 랜드마크가 vertex.landmark_idx/x/y 덮어쓰기, 먼저 등록된 landmark_to_vertex_ 항목 잔존), addControlPoints에는 가드가 있으며(grid_mesh.cpp:339), EYES에 33/133/159/145 포함(cpp/include/ir…
- **눈 확대 워프의 radial 변위를 정규화 공간에서 등방 계산 — 픽셀 공간에서 비등방 확대** (minor/정확성) — `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/cpp/src/warp/face_warp_controller.cpp`
  - 반박: 반박 성립. face_warp_controller.cpp:303-331에서 dx = (vec_x/dist) * (dist * scale_factor * expansion_weight) = vec_x * s, dy = vec_y * s로 환원된다. 즉 각 점의 변환은 정규화 공간에서 p' = c + (1+s)(p−c)인 중심 기준 균일 배율(호모테티)이다. 정규화→픽셀 변환 A = diag(w,h)에 대해 A·p' = A·c +…
- **COVER 기준 크기 소스 불일치: OverlayView는 검출(ImageAnalysis) 프레임, GL은 Preview 스트림 크기 — ViewPort 미사용으로 종횡비 상이 시 계통 오차** (major/2라운드 보완 감사) — `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/android/demo-app/src/main/java/com/irislenssdk/demo/camera/OverlayView.kt`
  - 반박(downgrade): 코드 사실관계는 맞다: OverlayView COVER는 검출 프레임 크기(GpuRenderActivity.kt:1016-1031 → OverlayView.kt:330-331, 507-517, computeScreenTransform 302-313), GL Cover는 Preview SurfaceRequest.resolution(gpu/CameraGLView.kt:168-173 → gpu/CameraGLRenderer.kt:1…
- **ScopedByteArray 생성자 — GetByteArrayElements 실패(OutOfMemoryError pending) 직후 무검사 GetArrayLength 호출 (pending exception 중 JNI 호출 UB)** (major/2라운드 보완 감사) — `/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/android/iris-sdk/src/main/cpp/jni_utils.h`
  - 반박(downgrade): 코드 인용 자체는 정확하다(jni_utils.h:167-170의 무검사 GetArrayLength, jni_utils.h:237 valid()=data_!=nullptr, 호출측 iris_jni.cpp:753-757 사후 검사 — 모두 직접 확인). 그러나 finding의 핵심 전제가 코드/런타임 사실과 다르다. (1) JNI 스펙상 Get<PrimitiveType>ArrayElements에는 THROWS 절이 없으며 "NUL…
  - 반박(downgrade): 코드 표면 사실(jni_utils.h:167-170에서 data_ null 미검사 후 GetArrayLength 호출)은 정확하나, 핵심 전제인 "GetByteArrayElements 실패 시 OutOfMemoryError pending 상태에서 GetArrayLength 호출 = UB" 시나리오가 Android/ART에서 도달 불가능하다. (1) JNI 스펙상 Get<PrimitiveType>ArrayElements에는 TH…

## 부록 B. severity 조정 내역 (검증자 2표 이상 일치 시 1단계 조정)

- blocker → major: **렌더 출력 픽셀 검증 0건 — 모든 렌더 테스트가 return code/시간만 단정, GPU 렌더러는 테스트 자체 부재** — 검증자 다수가 조정 제안
- major → minor: **카메라 프레임-랜드마크 동기화 부재: Preview/ImageAnalysis 별도 스트림 + timestamp 매칭 없음 → 구조적 렌즈 밀림** — 검증자 다수가 조정 제안
- blocker → major: **비원자적 모델 복사 + exists() 캐시로 잘린 .tflite 영구 고착** — 검증자 다수가 조정 제안
- blocker → major: **네이티브가 잘린/손상 모델을 무검증 로드 — 에러 코드가 아닌 크래시 가능** — 검증자 다수가 조정 제안
- major → minor: **'별도 스냅샷' 주석과 달리 uiIrisResult는 동기화 없는 공유 가변 인스턴스 — 디버그 시각화 torn read** — 검증자 다수가 조정 제안
- major → minor: **미러 시 L/R 의미 체계 불일치: GL은 left/right 스왑(screen 기준), OverlayView는 무스왑(detector 기준) — applyLeft가 반대 눈을 게이트** — 검증자 다수가 조정 제안
- major → minor: **랜드마크 주입 전환 대비 계약 결함 — 478 미만은 무조건 silent false, count 파라미터 없는 public 마스크 API는 인덱스 466까지 무검증 접근(OOB 위험)** — 검증자 다수가 조정 제안
- major → minor: **QA 모듈 4종(quality_metrics/ab_compare/release_gate/param_tuner) + profiler.cpp + beauty_processor.cpp = 프로덕션 .so에 컴파일되는 소비자-0 죽은 코드 1,982줄 (visibility hidden 미적용으로 링커 제거도 불가)** — 검증자 다수가 조정 제안
- major → minor: **JniCache::init — GetFieldID 31연쇄가 pending exception 상태에서 후속 JNI 호출 지속 (JNI 명세 UB)** — 검증자 다수가 조정 제안

## 부록 C. 완전성 비평(critic) 및 감사 실행 이력

**critic 평가**: 9개 finder의 커버리지는 추적 파이프라인(mediapipe_detector), GPU 백엔드, 스레딩, 표면 패리티, 지오메트리 수학에 집중되어 있고 상호 교차 검증도 견고하다. 그러나 직접 확인 결과 4개의 실질 사각지대가 남아 있다. (1) 모델/에셋 추출 수명주기: IrisLensSDK.java extractModelsFromAssets가 destFile.exists()만으로 캐시를 판정해(IrisLensSDK.java:1314) SDK 업데이트로 assets의 .tflite가 갱신돼도 filesDir의 구버전 모델을 영구 사용하고, 복사 중단 시 부분 파일도 영구 고착된다 — 어떤 finder도 에셋 로딩 경로를 다루지 않았다. (2) OverlayView.kt 1239줄: 인벤토리 최대급 파일 중 유일하게 어느 요약에도 등장하지 않으며, FIT/COVER+mirror 자체 좌표 매핑(OverlayView.kt:504-529)을 가진 제3의 독립 좌표계 구현이다. 사용자 검증 방법론이 '실기기 육안 체감'(디버그 오버레이 의존)이므로 검증 도구 자체의 미감사는 모든 좌표계 finding의 실기기 확인을 오염시킬 수 있다. (3) beauty_roi_manager.cpp 736줄: sdk_api_v2.cpp:244,362에서 실사용되며 face_mesh 478점을 직접 소비하는 프로덕션 랜드마크 소비자인데 어떤 finder도 다루지 않았다. 더불어 quality_metrics/param_tuner/release_gate/ab_compare(~1,600줄)는 cpp/CMakeLists.txt:81-87로 프로덕션 라이브러리에 컴파일되지만 소비자는 tests/test_quality_tuning.cpp뿐이다(SDK 20MB 크기 목표 대비 죽은 무게). (4) JNI 예외 규율: jni_utils.h 598줄에 ExceptionCheck 0건, iris_jni.cpp 2398줄에 예외 검사 2건뿐 — pending exception 상태에서 후속 JNI 호출은 JNI 명세 위반(UB)이며, parity finder는 표면 대조만, threading finder는 슬롯만 다뤄 이 영역이 정확히 두 관점 사이에 빠졌다. 그 외 검토 후 제외한 후보: examples/ 3종은 현행 sdk_api 사용으로 큰 문제 없음, C API 메모리 소유권은 free_result JNI 미노출 finding이 이미 커버, 데스크톱 빌드의 심볼 가시성 전수출(export.h가 빈 매크로)은 Android 산출물이 -fvisibility=hidden+strip(android cpp/CMakeLists.txt:183-201,220)으로 별도 처리되어 제품 영향 제한적, shader_sources.cpp는 geometry/gpu-core finding이 핵심 결함(타원 aspect, blend 의미)을 이미 포착.

**식별된 사각지대(전부 2라운드로 처리됨)**: 모델/에셋 추출·캐시 수명주기 (stale model 영구 고착), OverlayView 디버그 시각화 좌표계 — 제3의 독립 매핑 구현 미감사, beauty_roi_manager.cpp(프로덕션 랜드마크 소비자) 미감사 + 테스트 전용 QA 모듈 4종의 프로덕션 .so 포함, JNI 예외 규율 부재 — pending exception 상태의 후속 JNI 호출 (JNI 명세 위반 UB)

**실행 이력**: 1차(한도로 전멸·결과 0) → 2~4차(수동 중단 2회 + 캐시 prefix 한계 발견) → 트랜스크립트 수확 체크포인트 전환 → 연속1(보충 검증 44건, 막판 한도) → 연속2(라운드2+패널, 완주). resume 캐시의 prefix-순서 매칭 한계와 그 대응(수확 패턴)은 글로벌 스킬 `safety-workflow-checkpoint`로 문서화됨.
