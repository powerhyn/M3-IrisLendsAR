# P8-W4 ② 형태워프 브레인스토밍 R1 — Codex/Gemini 송신 프롬프트

당신은 IrisLensSDK(실시간 AR 렌즈+뷰티 SDK, C++ 코어 + OpenGL ES 3.1 GPU + Android JNI) 설계 검토자입니다.
아래 5개 설계 쟁점에 대해 **각각 [추천안] + [근거 1~2줄]**을 한국어로 답하세요. 새 쟁점 제기 금지, 주어진 범위만.
가능하면 저장소 파일을 직접 읽으세요(읽기전용): `docs/lenssim-handoff/jaw-vline-warp-handoff-from-lenssimulator.md`, `docs/lenssim-handoff/audit-report.md`, `cpp/src/warp/grid_mesh.cpp`, `cpp/src/warp/face_warp_controller.cpp`, `cpp/src/gpu/shader_sources.cpp`(SKIN_SMOOTH_COMPOSITE_FRAGMENT), `cpp/src/gpu/gpu_beauty_backend.cpp`(applyFaceWarp 스텁 + skin composite). 파일 접근 불가 시 아래 요약으로 답하세요.

## 작업 = ② 형태워프 (턱 V라인 / 얼굴 슬림 / 눈 확대) 구현
한국 뷰티앱 핵심. 현재 완전 스텁(applyFaceWarp가 NOT_SUPPORTED 반환, grid_mesh 미호출).

## 확정 알고리즘 (LensSim S23+ 실기기 검증 — 변경 금지, 이식만)
- 인버스 워프: 출력픽셀 p의 소스좌표 = p − Σᵢ dᵢ·exp(−|p−cᵢ|²/2σ²), **디스플레이 픽셀 공간**.
- 제어점 14 = 얼굴 측면 윤곽(FACE_OVAL) 7점×2, 광대정점·턱끝 제외(→눈높이 변위0 보장).
- 테이퍼 [0.2,0.45,0.65,0.85,1.0,0.85,0.6], 최대변위=얼굴폭×3.2%×strength, σ=얼굴폭×13%.
- 변위방향=얼굴 세로축 수직성분의 안쪽 반대. 셰이더 early-out(bbox+3σ). 적용순서 피부→워프→렌즈. 데모 기본 25%.

## 현재 아키텍처 사실
- GPU 뷰티 파이프라인: 카메라 OES→RGBA → **렌즈 합성** → **뷰티(skin smoothing/radiance composite)** → 화면. 즉 **렌즈 먼저, 뷰티 나중**.
- skin smoothing/radiance는 이미 **fragment composite 셰이더(SKIN_SMOOTH_COMPOSITE)에서 base/blur/mask 직접 평가** 방식으로 구현됨(P8-W1/W3). 풀스크린 쿼드 프래그먼트.
- 기존 grid_mesh(메시 기반 워프) substrate는 4결함 확정(코드+Codex 검증, 전부 프로덕션 미연결 스텁):
  1. RBF가 정규화 가중평균(Shepard) + zero-변위 컨트롤 제외 → 자연스러운 거리 falloff 없음(가까이 평탄→임계서 절벽으로 0). 비정규 RBF 합(disp+=w·d, 정규화 안 함)이 처방.
  2. 468/478 불일치(OOB 가드는 추가됨, iris center anchor는 별도 이월).
  3. 변위 스케일링: V라인 dy가 얼굴 크기 미반영, face_width=|Δx|(x성분만)이라 head roll에 비강건.
  4. 눈 확대가 정규화 좌표 거리 → 비정방 이미지 종횡비 왜곡.
- 공개 ABI(IrisBeautyConfigV2, POD): slim_face/thin_chin/enlarge_eyes 3개 float 필드 보존됨. iris_sdk_apply_face_warp C API 스텁 존재. (참고: 곁가지 제거된 SDK라 internal API 추가 선례=iris_sdk_set_skin_radiance.)

## 5개 쟁점 (각각 추천+근거)

**쟁점1 — 아키텍처**: (A) fragment-direct RBF 평가(14 제어점을 uniform으로, 컴포지트 셰이더서 인버스 워프 직접 계산 — skin/radiance와 동형, 메시 보간 불필요, 핸드오프 권고) vs (B) 기존 grid_mesh 메시워프 4결함 수리 후 재사용. 어느 쪽? grid_mesh substrate를 버리는 비용 vs fragment-direct의 단순성.

**쟁점2 — 파이프라인 배치**: 핸드오프 순서는 피부→워프→렌즈(워프가 렌즈보다 먼저). 우리는 렌즈먼저→뷰티나중. 눈높이 변위≈0이라 렌즈(눈높이)는 무영향 기대. (a) 워프를 카메라 프레임에 **렌즈 합성 전** 별도 패스로 vs (b) 뷰티 composite 단계(렌즈 합성 후)에서 눈높이0으로 워프. 렌즈 무영향·좌표정합·GPU 비용 측면 어느 것?

**쟁점3 — SDK 표면 매핑**: 핸드오프는 단일 jaw워프+strength. 우리는 slim_face/thin_chin/enlarge_eyes 3필드. 3필드를 워프에 어떻게 매핑? (예: slim_face+thin_chin을 V라인 워프의 강도/영역으로 합성, enlarge_eyes는 별도 radial 패스?) 그리고 공개 config 필드로 구동 vs internal API(set_skin_radiance류)로?

**쟁점4 — 좌표공간/회전/미러**: 핸드오프=디스플레이 픽셀공간(정규화 비등방 회피). 우리 랜드마크는 정규화 + 전면카메라 미러 + 회전. 픽셀공간 워프 계산을 어디서(셰이더서 vUv×viewportPx? CPU서 제어점 픽셀 산출 후 uniform?) + 미러/회전 정합 방법.

**쟁점5 — 단위테스트**: 핸드오프 권고 5종(변위방향 세로축 향함 / strength≤0 비활성 / 눈높이 변위<최대5% / 3σ bbox 정합 / 퇴화입력 방어)을 C++ GoogleTest로 이식. 합성 타원 얼굴로 실랜드마크 없이. 추가로 점검할 불변식 있는지.

## 출력 형식
각 쟁점마다: `쟁점N: [추천 A/B/...] — 근거`. 마지막에 `종합 한 줄: 이 워프 구현의 가장 큰 리스크는?`
</content>
