# ④ 좌표 canonical relabeling — 구현 스펙 (ADR §7.3)

> 조사(wf_2b44bca7 4축) + Codex 2차 적대검증 반영. left/right 라벨이 system B에서 전면 반전(자기일관).
> **전략(확정)**: 공개 필드명 left/right **유지** + 인덱스 그룹 **값 스왑**으로 canonical 달성(API rename 금지).

## Canonical 정본 (LensSim LandmarkIndices.kt = 명명 정본)
- **홍채**: 468–472 = 피험자 **RIGHT** eye / 473–477 = 피험자 **LEFT** eye.
- **눈 윤곽 16점**: 33-그룹(33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246) = **RIGHT** / 362-그룹(362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398) = **LEFT**.
- **EAR 6점**: 33,160,158,133,153,144 = **RIGHT** / 362,385,387,263,373,380 = **LEFT**.
- **눈꺼풀**: upper 159/160/161 + lower 145/144/153 = **RIGHT** / upper 386/385/384 + lower 374/373/380 = **LEFT**.
- **안각(canthus)**: 33 = RIGHT **outer**(외/lateral, 귀쪽), 133 = RIGHT **inner**(내/medial, 코쪽) / 263 = LEFT **outer**, 362 = LEFT **inner**.

## 변경 원칙
현재 코드는 위 RIGHT 그룹(468/33...)을 `left_*`/`kLeft*`/`LEFT_*`에 넣고, LEFT 그룹(473/362...)을 `right_*`에 넣음(반전). **정정 = 각 left/right 상수 쌍의 인덱스 값을 서로 교환**(이름·필드명 불변). → 정정 후 `left_*`=피험자 LEFT(473/362그룹), `right_*`=피험자 RIGHT(468/33그룹).

## R1 — 코어 C++ 라우팅 스왑 (값 교환, 자기일관 필수)
대상(각 파일에서 현재 값 확인 후 left↔right 그룹 값 교환):
1. `cpp/include/iris_sdk/landmark_injection.h`: `kLeftIris`(현 468그룹)↔`kRightIris`(473그룹), `kLeftEAR`(33그룹)↔`kRightEAR`(362그룹), 눈 윤곽 상수 있으면 동일. `cpp/src/landmark_injection.cpp` deriveIrisResult(109-110/125-132)는 상수 경유라 자동 정합.
2. `cpp/include/iris_sdk/types.h`: IrisResult `left_iris`(158)/`right_iris`(162) 주석을 "left=473그룹=피험자 좌안"으로 정정(값 아닌 문서).
3. `cpp/src/gpu/eye_render_packet_adapter.cpp`(19-22)+`.h`: 눈꺼풀 kLeft*/kRight* 상수 + EyeSide 매핑 교환.
4. `cpp/src/temporal_stabilizer.cpp`(11-20)+`.h`(126-127): kLeftEAR/kLeftEyelid* ↔ kRight* 교환. `left_eye_`/`right_eye_` 상태는 이름 유지(상수만 교환되면 자동 정합).
5. `cpp/src/gpu/gpu_lens_renderer.cpp`(47-71): `LEFT_EYE_CONTOUR`(33그룹)↔`RIGHT_EYE_CONTOUR`(362그룹), `LEFT_UPPER/LOWER_EYELID`(159/145그룹)↔`RIGHT_*`(386/374그룹).
6. `cpp/include/iris_sdk/warp/face_warp_controller.h`(106-132): `LEFT_IRIS_CENTER`(468)↔`RIGHT_IRIS_CENTER`(473), `LEFT_EYE_CONTOUR`↔`RIGHT_EYE_CONTOUR`, `LEFT_EYEBROW`↔`RIGHT_EYEBROW`(눈썹도 canonical 맞춰). 각 triple(iris+contour+eyebrow)이 한 눈을 가리키게 자기일관 유지.

## R2 — 안각 inner/outer 정정 (gpu_lens_renderer.cpp:62-65) — side+corner 이중반전
**GPU ellipse 전용(기본 OFF) → 골든 무관, device GPU-ellipse 시 육안 변화(눈-타원 비대칭 정상화, 진짜 버그수정).** 정확한 목표값:
- `LEFT_INNER_CORNER`: 33 → **362** (LEFT inner)
- `LEFT_OUTER_CORNER`: 133 → **263** (LEFT outer)
- `RIGHT_INNER_CORNER`: 263 → **133** (RIGHT inner)
- `RIGHT_OUTER_CORNER`: 362 → **33** (RIGHT outer)
fitEyeEllipse(651-703) `rx_inner=dist*0.85`/`rx_outer=dist*1.0` 비대칭이 올바른 코너에 적용됨. is_left_eye 분기는 그대로(LEFT_EYE_CONTOUR도 R1서 362그룹 됨).

## R3 — 미러 §7.4 정합 (gpu_lens_renderer.cpp:986-993, 1043-1046) — ⚠️ device-verify 필수
현재 미러: X-flip **+ std::swap(left,right)**. ADR §7.4 = 미러는 렌더 X-flip 단일책임, L/R 라벨은 피험자 기준 유지(eye-swap 금지). **정정**: `std::swap(left_x,right_x)`+swap 4종(989-992) + 눈꺼풀 swap(1044-1045) **제거**, X-flip(987-988, l/r 각각 1-x)만 유지. 근거: X-flip만으로 각 눈 좌표가 미러 화면 위치로 정확히 이동(피험자-우안 sensor low-x → 1-x high → 미러 화면 우측, 실제 미러서 피험자 우안이 화면 우측). swap은 불필요한 이중처리. **R1 라벨 정정과 결합해 전면 카메라 실기기 육안 검증 필수**(상쇄/이중반전 확인).

## 글루/데모/테스트 (Kotlin — 메인이 담당)
- `TasksToIrisResult.kt`(47-57): `RESULT_LEFT_IRIS`(468그룹)↔`RESULT_RIGHT_IRIS`(473그룹), `RESULT_LEFT_EAR`(33)↔`RESULT_RIGHT_EAR`(362) 값 교환. 주석(27-32) 정정.
- `OverlayView.kt`(65-71): 자체 `LEFT_IRIS_CENTER=468`/contour 상수를 canonical로(LandmarkIndices 참조 또는 값 교환) — **검증도구라 device-verify 前 정리 필수**. 해부학/화면 혼용 주석 정정.
- 테스트(값 교환 동반 — 회귀 가드): `cpp/tests/test_landmark_injection.cpp`(386-389/417-421) L/R expected 스왑, `android/iris-sdk/src/test/.../TasksToIrisResultTest.kt`(71-77/148-149/200-206) L/R assert 스왑 + 테스트명, `cpp/tests/test_beauty_roi_manager.cpp`(68-91) 주석 라벨. test_golden_injection_derive/test_types는 자기일관(주석만).

## 골든 재캡처 + migration checker
- 재캡처: `golden_capture_all.sh INJECT_BASELINE`로 18 JSON 재생성. R1 swap으로 left_*↔right_* 값 교환됨. CPU PNG **byte-invariant**(양안 렌더, is_mirror no-op).
- **migration checker (신규, 스왑이 옳음을 증명)**: old baseline vs new baseline에서 `new.left_iris==old.right_iris`, `new.left_radius==old.right_radius`, left_detected/eyelid_ratio_left/avg_iris_luma_left 동일 교환, PNG byte-equal 확인. (golden_compare는 key별 엄격이라 스왑 자체는 PASS 안 됨 — 별도 1회 checker.)
- manifest: `cpp/tests/golden/RELABEL_RECAPTURE_MANIFEST.md`(Tasks ver/model SHA/기기/invariant=new.left==old.right/승인자).

## 게이트
데스크톱 빌드(신규 warn0) + 골든 재캡처 + migration checker PASS + ctest 회귀0(pre-existing 3) + assembleDebug + **전면 카메라 실기기**(미러 R3 + 안각 R2 육안).

## 이월(범위 외)
- fitEyeEllipse가 정규화 거리로 rx/ry 계산(§7.0 픽셀원칙 위반) = 별도 버그, relabeling 무관.
- 데모 per-eye apply 토글 UI 부재(둘 다 true) → screenLeft/Right 분리 불요(passthrough + 문서화).

## ✅ 코어 C++ (R1+R2+R3) 완료 기록 (2026-06-19)
**상태**: R1/R2/R3 + C++ 테스트 동반 갱신 완료. 데스크톱 빌드 신규 warn0/error0. 신규 ctest 회귀0.

### 스왑한 상수 쌍 (현재값→목표값, 자기일관 검증 완료)
- `landmark_injection.h`: kLeftIris {468그룹}→{473그룹}, kRightIris {473그룹}→{468그룹}; kLeftEAR {33그룹}→{362그룹}, kRightEAR {362그룹}→{33그룹}. deriveIrisResult는 상수 경유라 자동 정합.
- `types.h`: IrisResult left/right_iris 주석만 canonical 정정(값 불변, offsetof static_assert 불변).
- `eye_render_packet_adapter.cpp`: kLeft/RightUpper·LowerEyelid (159/145그룹↔386/374그룹) 스왑. EyeSide 매핑은 result.left_*/상수 경유로 자동 정합.
- `temporal_stabilizer.cpp`: kLeft/RightEAR + kLeft/RightEyelidTop·Bottom (33/159/145그룹↔362/386/374그룹) 스왑. left_eye_/right_eye_ 상태명 불변.
- `gpu_lens_renderer.cpp`: LEFT/RIGHT_EYE_CONTOUR (33그룹↔362그룹), LEFT/RIGHT_UPPER·LOWER_EYELID (159/145↔386/374그룹) 스왑.
- `face_warp_controller.h`: LEFT/RIGHT_IRIS_CENTER (468↔473), LEFT/RIGHT_EYE_CONTOUR (33그룹↔362그룹), LEFT/RIGHT_EYEBROW (70계열↔300계열) 스왑 — iris+contour+eyebrow triple 자기일관.
- (부수) `face_warp_controller.cpp:35` + `.h:185` kMinWarpLandmarkCount 주석 정정(max idx=LEFT_IRIS_CENTER=473, 값 474 불변).

### R2 안각 (gpu_lens_renderer.cpp:62-65) — side+corner 이중반전
- LEFT_INNER_CORNER 33→**362**, LEFT_OUTER_CORNER 133→**263**, RIGHT_INNER_CORNER 263→**133**, RIGHT_OUTER_CORNER 362→**33**. fitEyeEllipse(is_left)에서 contour(R1)와 corner가 같은 눈 가리킴(자기일관).

### R3 미러 (gpu_lens_renderer.cpp) — X-flip 단일책임, eye-swap 전면 제거
- 홍채(986-993): std::swap(x/y/r/det) 4종 제거, X-flip(1-x)만 유지.
- 눈꺼풀(1043-1046): swap(top/bot) 2종 제거(블록 no-op화). Y값이라 X-flip 무관.
- **타원(1100-1112, 스펙 명시 밖이나 §7.4 단일책임 정합 위해 동반)**: eye-swap 5종 제거, per-eye X-flip 기하(1-cx, π-rot, swap rxi/rxo) 유지.
- **alpha(1145-1150, 동반)**: eye-swap 제거(스칼라라 X-flip 무관). render_alpha_[0]→uLeft, [1]→uRight 그대로(라벨 일관).
- 근거: ADR §7.4 "미러는 렌더 X-flip 단일책임, L/R 라벨 무스왑 단일계약". ⚠️ **전면 카메라 실기기 육안 검증 필수**(상쇄/이중반전 확인).

### 테스트 동반 갱신
- `test_landmark_injection.cpp`: RejectsOutOfRange/UndetectedEye (469깸→right 미검출로 expected 스왑), WriteThenReadRoundTrip (468그룹→right_iris 검사로 스왑). 골든 대조 테스트(529/544 계열)는 baseline old 의존이라 미수정.
- `test_eye_enlargement.cpp`: SetUp 좌표 정합 (33그룹 contour cx 0.35→0.65, 362그룹 cx 0.65→0.35; iris center 불변) — R1-6 상수 스왑의 동반 회귀(470 MaxStrengthBounded, 475 ExpansionDirection) 해소.
- `test_beauty_roi_manager.cpp`: setEyeLandmarks 주석만 canonical 라벨 보강(BeautyROIManager는 union mask라 동작 무관).

### ctest 결과 (679 중 24 failed, 신규 회귀 0)
- **골든 baseline 의존 21개(메인 재캡처 대상, swap 정확함 증명)**: 529 DerivesIris, 544 LandmarkCApiRoundTrip, 548 MisswappedDims, 549-566 GoldenInjectionDerive(18). 실패 패턴=정확히 `new.left==old.right`(left/right 값 교환).
- **pre-existing 3개(stash baseline에서도 실패 확인, relabeling 무관)**: 401 GPUBeautyBackend.FailsWithNullContext, 587/588 FreqSepMapping.

### 메인 잔여(범위 밖, 미처리)
- 골든 재캡처 + migration checker + manifest, Kotlin(TasksToIrisResult/OverlayView) swap, assembleDebug, 실기기 R2/R3 육안.
- **BeautyROIManager.cpp LEFT/RIGHT_EYE_INDICES(33↔362그룹) 라벨 미정정**(스펙 R1 대상 밖, union mask라 동작 무관). canonical 완전성 위해 후속 정정 후보로 보고.
