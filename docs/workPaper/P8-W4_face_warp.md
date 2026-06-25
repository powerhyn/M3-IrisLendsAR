# P8-W4 — ② 형태워프 (턱 V라인 / 얼굴 슬림 / 눈 확대)

> 상태: 🔄 구현 중 (2026-06-25 브레인스토밍 R1 종결 → 착수). 범위=**턱 V라인 슬림만**(slim_face/thin_chin), enlarge_eyes 제외.
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
</content>
