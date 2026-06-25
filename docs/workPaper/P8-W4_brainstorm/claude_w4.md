# P8-W4 ② 형태워프 브레인스토밍 R1 — Claude 응답

(모더레이터 아닌 동등 참여자 의견. 타 모델 응답 수신 전 독립 작성.)

## 쟁점1 — 아키텍처: **[A] fragment-direct RBF**
- 근거: 핸드오프 검증본(LensSim COMPOSITE_FS)이 이미 fragment-direct이고, 우리 skin/radiance도 동일 패턴(SKIN_SMOOTH_COMPOSITE에서 base/blur/mask 직접 평가) → 인프라·정신모델 재사용. 14 제어점×exp는 bbox+3σ 내에서만 = 저비용. grid_mesh 4결함 수리는 결국 RBF 재구현이라 노력 대비 이득 없음 + 메시 보간·468/478 매핑 부채.
- ⚠️ **단서**: kickoff 불변식 #2 "grid_mesh substrate 보존(삭제 금지)"와 충돌. 해석: fragment-direct 채택 시 grid_mesh는 **미사용 dead substrate가 됨**. 본 W에서 grid_mesh를 **삭제하지 않고 미사용으로 둔다**(불변식 준수). 단 "②=grid_mesh 위에 구현"이라는 암묵 가정은 뒤집힘 — 사용자 확인 필요.

## 쟁점2 — 파이프라인 배치: **[b] 뷰티 composite 단계(렌즈 후)에서 눈높이0 워프**
- 근거: (1) 파이프라인 재배열 불필요(렌즈먼저→뷰티 유지) = 저위험. (2) **눈높이 변위≈0 제약이 정확히 이걸 안전하게 만드는 장치** — 렌즈는 눈높이에 있어 변위≈0 → 워프해도 렌즈 픽셀 안 움직임. (3) 핸드오프의 "피부→워프→렌즈" 순서의 본질 불변식은 "눈높이 변위0으로 홍채/렌즈 정합"인데, (b)에서도 렌즈가 눈높이라 성립. (4) 워프=뷰티 composite 마지막 단계(skin→warp 순, 핸드오프 피부→워프 일치). full: 카메라→렌즈→[skin→warp]→화면.
- 반론 인지: (a) 렌즈 전 워프가 핸드오프 순서엔 더 충실하나 재배열 비용. eye-height0이 (b)를 安全케 하므로 (b) 선호. **실기기서 렌즈 정합이 본 결정의 핵심 검증점.**

## 쟁점3 — SDK 표면 매핑: **slim_face=메인 V라인 강도, thin_chin=하단 제어점 가중, enlarge_eyes=별도(이번 deferred 권고)**
- 근거: 핸드오프 검증 알고리즘 = jaw V라인(턱선 슬림) = `slim_face`에 직결. `thin_chin`은 하악/턱끝 제어점(테이퍼 하단)에 추가 가중으로 매핑. **`enlarge_eyes`는 별도 radial 메커니즘이고 검증된 핸드오프 없음 + audit #4 결함** → 본 W는 V라인(slim/chin)만, eyes는 후속 슬라이스 분리 권고.
- 표면: 공개 ABI `slim_face/thin_chin`(이미 보존) + `iris_sdk_apply_face_warp`(스텁, 3param 이미 시그니처 보유)를 정식 구현. 데모는 토글(btnP8Slim 등). config 필드가 이미 공개라 internal API 신설보다 기존 표면 활성이 정합.

## 쟁점4 — 좌표공간/회전/미러: **CPU서 제어점·변위를 디스플레이 픽셀공간 산출 → uWarp[14] uniform, 셰이더는 vUv×viewportPx로 픽셀공간 평가**
- 근거: 정규화 비등방(audit #1c/#4) 회피의 정본 = 픽셀공간. LensSim COMPOSITE_FS와 동일(uWarp[14]=[cx,cy,dx,dy] px, uWarpSigma px). 미러/회전 정합은 P8-W1 선례 재사용 — 제어점을 composite가 샘플하는 동일 vUv 디스플레이 공간으로 변환(전면 mx=1-x 등). σ·변위도 픽셀(얼굴폭 px 기준). 셰이더 early-out도 픽셀 bbox.
- 핵심: 제어점 산출은 **워프 전 원본 oval**(이마확장 미적용 — 핸드오프 §3 불변식), One-Euro 필터링된 FACE_OVAL 입력.

## 쟁점5 — 단위테스트: **핸드오프 5종 + 대칭성 + roll-강건성 2종 추가**
- 5종(변위방향 세로축 향함 / strength≤0 비활성 / 눈높이 변위<최대5% / 3σ bbox 정합 / 퇴화입력 σ=0) C++ GoogleTest, 합성 타원 얼굴.
- 추가 권고: **(6) 좌우 대칭**(미러 제어점 변위 대칭) + **(7) roll-강건성**(face_width를 유클리드 픽셀거리로 → audit #3 수정 검증; 합성 얼굴 roll 회전 시 워프 안정). 이 둘이 audit #3 결함 수정을 직접 게이트.

## 종합 한 줄 — 가장 큰 리스크
**파이프라인 (b) 배치에서 "눈높이 변위≈0"이 실제로 충분히 0인가** — σ나 제어점 선택이 눈 영역에 변위를 누출하면 렌즈/홍채가 육안으로 어긋난다. → 눈높이<5% 단위테스트 + 실기기 렌즈정합이 필수 게이트. (이게 깨지면 (a) 렌즈 전 워프로 후퇴.)
</content>
