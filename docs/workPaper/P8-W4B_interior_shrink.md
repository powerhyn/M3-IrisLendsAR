# P8-W4B — 얼굴 내부 축소 (face-small lite)

**상태**: ✅ 완료 (2026-07-15 착수 → 동일 실기기 판정 후 develop 머지)
**브랜치**: `feature/P8-W4B-interior-shrink` (develop 분기)
**선행**: P8-W4 턱 V라인 워프 (fragment-direct RBF, 머지 18158da)

## 1. 배경 — 사용자 실기기 피드백 (갤럭시탭 S10 Ultra)

> "턱만 줄이면 턱은 줄어드는데 코·입이 그대로면 어색하다. 눈꼬리 높이쯤부터 축소가 시작되고,
> 안쪽 메시(코·입)도 중심 쪽으로 당겨져야 '턱을 깎은' 느낌이 아니라 '얼굴이 작아진' 느낌이 든다."

대화면(태블릿)에서 턱-only 워프의 비율 부조화가 폰보다 두드러짐.

## 2. 설계 결정 (3축 감사 워크플로 → 사용자 확정)

### 검토한 대안과 판정
| 대안 | 판정 | 이유 |
|------|------|------|
| 전역 radial shrink 항 (피벗+O(1)) | ❌ 기각 | canonical 좌표 계산 결과 눈꼬리가 턱 변위의 ~70-75%를 받음 → 렌즈 follow-warp 필수. **사용자 결정: 눈은 안 건드린다** |
| 렌즈 follow-warp (풀 face-small) | ⏸ 보류 | 눈 고정 결정으로 불필요. 필요 시 후속 트랙 (CPU 미러 `evaluateWarpDisplacement` 활용 경로 확인됨) |
| **국소 내부 제어점 확장 (채택)** | ✅ | σ=0.13×얼굴폭 국소 가우시안이라 눈 영향을 게이트 안으로 관리 가능 |

### 눈 고정 근거
- 렌즈 SDK 특성: 눈이 상품 초점 — 눈 축소는 렌즈 어필과 상충. 상용 face-small도 눈은 유지/확대 페어링.
- P8-W4 불변식 "눈높이 변위 ≈ 0" (렌즈 쿼드는 무워프 좌표에 그려짐) 유지 → follow-warp 작업 불필요.

## 3. 확정 스펙 — 2노브 24제어점 모델

| 그룹 | 인덱스 (좌/우) | taper | 강도 노브 |
|------|---------------|-------|-----------|
| 턱 라인 7×2 (S23+ 고정, 무변경) | 323,361,288,397,365,379,378 / 93,132,58,172,136,150,149 | 0.2,0.45,0.65,0.85,1.0,0.85,0.6 | `slimFace` |
| 상방 실루엣 1×2 (신규) | 454 / 234 | 0.05 | `slimFace` |
| 내부 4×2 (신규): 볼 중앙, 볼 하부, 입꼬리, 콧볼 | 280,425,291,358 / 50,205,61,129 | 0.12, 0.35, 0.30, 0.20 | `thinChin` (재정의) |

- 변위 모델 무변경: perp-to-axis(10→152) inward, max_disp = 얼굴폭 × 0.032 × strength(그룹별)
- 조건부 패킹: strength≤0 그룹은 업로드 생략 (thinChin=0이면 기존과 수치 동일 = 회귀 게이트)
- `thinChin` 의미 재정의: "턱 축소" → "얼굴 내부 축소". JNI/Java 배선 기존 그대로 재활용 (바인딩 무변경)
- 셰이더 `uWarp[14]` → `[24]` (GLSL 리터럴 수동 동기화 지점)
- 미드라인 점(코끝 1, 입술 0/17)은 세로축 위라 perp=0 → 구조적으로 안 움직임. 코·입 축소는 콧볼·입꼬리가 담당.

### 게이트 재정의
- **홍채 중심 변위 < 최대 변위의 5%** (하드, 기존 유지)
- 눈꼬리 < 12% (완화 — 절대 px 미미: strength 0.3·폭 800px 기준 ≈ 0.6px. 기존 5%는 턱-only 기준)
- 최종 판정 = 실기기(S10 Ultra) 렌즈 정합 육안 + 슬라이더 조합 A/B

## 4. 변경 파일

- `cpp/include/iris_sdk/warp/jaw_warp_geometry.h` — 신규 그룹 상수, kMaxControlPoints 24, 2노브 문서
- `cpp/src/warp/jaw_warp_geometry.cpp` — computeJawWarp(…, jaw_strength, interior_strength, …) + 조건부 패킹
- `cpp/src/gpu/shader_sources.cpp` — uWarp[24]
- `cpp/src/gpu/gpu_beauty_backend.cpp/.h` — max() 합성 제거, 2노브 전달
- `cpp/include/iris_sdk/sdk_api.h`, `beauty_filter.h` — thin_chin 문서 재정의
- `cpp/tests/test_jaw_warp_geometry.cpp` — 회귀(기대값 무변경) + 패킹 + 게이트 테스트
- `android/iris-sdk/.../BeautyFilterConfigV2.java` — thinChin javadoc 재정의
- `android/demo-app/.../GpuRenderActivity.kt` + `activity_gpu_render.xml` — "얼굴축소" 슬라이더 (기본 0)

## 5. 진행 기록

- 2026-07-15: 3축 감사 워크플로(구현 제약/canonical 토폴로지 실측/설계 히스토리) → 설계안 2종 제시 → 사용자 "눈 고정" 확정 → 국소 내부 제어점 방식 채택, 구현 착수 (C++ = cpp-pro 위임, 데모/Java = 직접)
- 2026-07-15: C++ 구현 완료. `test_jaw_warp_geometry` 10/10 PASS (+`test_beauty_config_v2` 25/25, `test_face_warp_controller` 23/23 무변경 통과)

### 게이트 실측치 (golden `face_closeup__rot0`, 960×720, 분모 = max_disp 10.08px)

| 케이스 | 홍채 468 | 홍채 473 | 눈꼬리 33 | 눈꼬리 263 |
|--------|---------|---------|----------|-----------|
| jaw-only (interior=0) | 2.73% | 0.31% | 6.23% | 2.21% |
| jaw+interior 풀강도 | **4.73%** ✅<5% | 1.74% | **8.36%** ✅<12% | 3.68% |

### 구현 편차 (계획 대비)

1. **kInteriorTaper[0] (볼 중앙) 0.12 → 0.08 하향**: 초기값에서 풀강도 홍채 468이 5.02%로 하드 게이트 근소 초과 → 0.08에서 4.73%.
2. **jaw-only count = 16 (구 14)**: 상방 실루엣(454/234)이 `slimFace` 스케일이라 기존 턱슬림 슬라이더만 올려도 활성. 의도된 동작("V라인이 더 위에서 시작") — 단 기존 턱슬림의 눈꼬리 변위가 3.7%→6.23%(절대 0.63px)로 증가. 실기기에서 렌즈/마스크 어긋남 감지 시 레버 = `kUpperTaper`(0.05).
3. **눈꼬리 게이트 5%→12% 완화 공식화** (§3 게이트 재정의대로). 홍채 5% 하드는 유지.

### 알려진 이슈

- `test_gpu_beauty_backend.FailsWithNullContext` 선재 실패 (baseline에서도 재현, 본 작업과 무관 — 데스크톱 스텁 initialize(nullptr) 동작)
- macOS 데스크톱은 `IRIS_SDK_GPU_AVAILABLE=0`이라 GPU 백엔드 호출부는 Android NDK 빌드로 컴파일 검증

## 6. 미결 → 종결 처리

- [x] C++ 구현 + 게이트 테스트 실측치 (taper 조정: 볼 중앙 0.12→0.08)
- [x] 데모 빌드 (NDK 컴파일 포함, APK b314)
- [x] 실기기 판정: S23+(SM-S916N) 설치·사용자 체감 "잘 됨" → develop 머지 확정 (2026-07-15)
- [ ] (후속 후보) 태블릿 S10 Ultra 대화면 재확인 — 원 피드백 발원 기기
- [ ] (후속 후보) 내부 taper 미세 튜닝 — 체감 이슈 발생 시
- [ ] (후속 후보) "눈꼬리 높이(356/127)부터 시작"은 눈 고정과 양립 불가로 454/234(귀 중간)까지만 반영 — 시작 높이 부족 체감 시 렌즈 follow-warp 트랙으로 재논의
