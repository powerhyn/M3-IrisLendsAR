쟁점1: [추천 A] — `fragment-direct RBF`로 가는 것이 맞습니다. `grid_mesh`는 미연결 스텁일 뿐 아니라 정규화 가중평균/468-478/스케일/종횡비 결함을 모두 고쳐야 하므로, 이미 검증된 14점 픽셀공간 RBF를 skin composite 계열 셰이더에 직접 넣는 쪽이 단순하고 리스크가 낮습니다.

쟁점2: [추천 b] — V라인/슬림은 뷰티 composite 단계에서 렌즈 합성 후 워프하는 쪽을 추천합니다. 눈높이 변위 5% 미만 게이트를 지키면 렌즈 영향은 사실상 없고, 기존 풀스크린 composite에 합쳐 추가 FBO/패스를 피할 수 있습니다.

쟁점3: [추천: 공개 config 필드 구동] — `slim_face`와 `thin_chin`은 단일 검증 V라인 strength로 합성하되 additive 폭주를 막기 위해 `jawStrength = clamp(max(thin_chin, 0.75 * slim_face), 0, 1)`처럼 보수적으로 매핑합니다. `enlarge_eyes`는 jaw RBF에 섞지 말고 별도 픽셀공간 radial warp 항으로 같은 fragment 평가부에 추가하며, 이미 공개 ABI 필드와 `iris_sdk_apply_face_warp`가 있으므로 internal API는 보조 A/B용으로만 두는 편이 맞습니다.

쟁점4: [추천: CPU 픽셀 uniform 산출 + shader 평가] — 제어점 `cx,cy,dx,dy`, `sigma`, `bounds`는 CPU에서 출력 텍스처 픽셀공간으로 산출해 uniform으로 넘기고, 셰이더는 `vTexCoord * viewportPx`로 RBF만 평가하게 합니다. 미러/회전은 skin mask의 `x -> 1-x`처럼 현재 렌더 텍스처 좌표계로 들어가기 전 단일 변환 헬퍼에서 처리해야 좌표 계약이 흔들리지 않습니다.

쟁점5: [추천: 5종 그대로 + mirror/roll 불변식 추가] — 핸드오프 5종은 합성 타원 얼굴 GoogleTest로 그대로 이식하고, 추가로 좌우 미러 대칭성과 roll 회전 등변성을 넣는 것을 추천합니다. 이번 결함들이 대부분 좌표계/종횡비/roll에서 났기 때문에 이 두 테스트가 회귀 탐지력이 큽니다.

종합 한 줄: 이 워프 구현의 가장 큰 리스크는 알고리즘 자체보다 “랜드마크 정규화 좌표를 현재 렌더 텍스처의 미러/회전된 픽셀공간으로 일관되게 맞추는 것”입니다.
