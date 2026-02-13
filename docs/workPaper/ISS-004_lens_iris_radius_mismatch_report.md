# ISS-004: 홍채 반경 디버그/렌더링 불일치로 인한 눈 영역 과대 렌더링

- 작성일: 2026-02-11
- 상태: Fixed (디바이스 검증 대기)
- 우선순위: High
- 영향 범위: Android Demo GPU 렌즈 렌더링 경로
- 수정일: 2026-02-11

## 1) 문제 요약

디버그 화면에서 홍채 랜드마크(보라색 점)는 홍채 내부에 위치하지만, 렌즈 반경 디버그 원(녹색)이 홍채보다 크게 표시되어 눈 양 끝(안구 폭)에 가까운 영역을 덮는 현상이 발생한다.

사용자 관찰 기준:
- "홍채 중심 기반 원"이 아니라 "눈 영역 기반 원"처럼 보임

## 2) 기대 동작 vs 실제 동작

- 기대 동작:
  - 디버그 반경 원은 홍채 경계(iris 4 boundary points)와 유사한 크기로 표시되어야 함
  - 렌즈 합성도 홍채 중심/반경 기반으로 자연스럽게 정렬되어야 함

- 실제 동작:
  - 디버그 원이 홍채 포인트보다 크게 표시됨
  - 특정 케이스에서 렌즈가 홍채가 아닌 눈 영역 전체에 가깝게 커져 보임

## 3) 코드 근거 및 원인 분석

### RC-1 (주원인): 디버그 원이 "raw 홍채 반경"이 아니라 "렌즈 적용 반경"을 표시

- 홍채 포인트는 실제 iris landmark(468~477) 기준으로 그림
  - `android/demo-app/src/main/java/com/irislenssdk/demo/camera/OverlayView.kt:1102`
- 디버그 원은 아래 식으로 그림:
  - `r = radius * scaleFactor * lensConfig.scale`
  - `android/demo-app/src/main/java/com/irislenssdk/demo/camera/OverlayView.kt:826`
- 반면 디버그 텍스트 `r=...`는 raw radius(픽셀) 값 표시
  - `android/demo-app/src/main/java/com/irislenssdk/demo/camera/OverlayView.kt:932`

결론:
- 점(원본)과 원(스케일 적용)의 기준이 다르므로 "반경이 눈폭 기준으로 잡힌다"는 인식이 강하게 발생함.

### RC-2 (고확률 보조원인): GPU 렌즈 반경 정규화 축/거리 계산 불일치 가능성

- Kotlin에서 반경 정규화:
  - `normalizedRadius = radius / detW`
  - `android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLRenderer.kt:620`
- 셰이더에서 거리 계산은 `x * aspectRatio` 보정 사용:
  - `android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLRenderer.kt:186`
  - `android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLRenderer.kt:190`

결론:
- 반경 입력 축과 셰이더 거리 공간 축이 완전히 동일하지 않을 경우, portrait/비정사각 프레임에서 체감 반경이 과대 렌더링될 수 있음.

### RC-3 (조건부 보조원인): V2 홍채 보정 로직이 중심을 눈 중심으로 강제 이동

- V2 경로에서 보정 호출:
  - `cpp/src/mediapipe_detector.cpp:2570`
- 임계 초과 시 iris 중심을 eye center로 이동:
  - `cpp/src/mediapipe_detector.cpp:1899`
  - `cpp/src/mediapipe_detector.cpp:1922`

결론:
- 오검출/좌표 불안정 프레임에서는 렌즈 중심이 홍채 추적보다 눈 중심 고정처럼 보일 수 있음.

## 4) 증상 증폭 요인

- 기본 렌즈 스케일이 큼 (`2.5`)
  - `android/iris-sdk/src/main/java/com/irislenssdk/LensConfig.java:161`
- 디버그 원은 스케일을 포함해 표시되므로 체감 과대 현상이 더 커짐

## 5) 재현 절차

1. Demo 앱 실행 후 전면 카메라 활성화
2. `Debug` + `Mesh` ON
3. 렌즈 활성화 상태에서 `Scale` 기본값(또는 1.5 이상) 유지
4. 보라색 홍채 포인트와 녹색 원의 경계 차이를 관찰

## 6) 수정 제안

### Fix-A (P0): 디버그 표기 분리 (가시성/오해 제거)

- 디버그 원 2개를 분리 표시:
  - Raw Iris Radius 원 (홍채 반경, 스케일 미적용)
  - Effective Lens Radius 원 (렌즈 스케일 적용)
- 디버그 텍스트에 `rawR`, `effectiveR`, `scale` 동시 출력

### Fix-B (P1): GPU 반경 계산 좌표계 통일

- 반경 정규화 축과 셰이더 거리 공간을 동일 기준으로 통일
- 권장:
  - 셰이더에서 픽셀 공간(또는 동등한 isotropic 공간)으로 거리 계산
  - Kotlin 측 normalized radius와 셰이더 측 distance metric의 축 정의를 문서화

### Fix-C (P2): V2 보정 로직 가드 강화

- `validateAndFixIrisCoordinates()` 적용 조건을 강화:
  - 임계값을 동적(얼굴 크기/눈폭 비례)으로 조정
  - eye-center 강제 이동 대신 보간(lerp) 보정

## 7) 검증 기준 (Acceptance Criteria)

- AC-1: Raw 반경 원이 iris boundary 포인트와 오차 ±10% 내
- AC-2: Effective 반경 원이 `raw * scale`와 ±5% 내 일치
- AC-3: 회전(0/90/270) + 미러 조합에서 반경 왜곡/과대 현상 재발 없음
- AC-4: 시선 이동 시 렌즈 중심 추적이 눈 중심 고정으로 보이지 않음

## 8) 테스트 매트릭스

- 모델 버전:
  - V1(468), V2(478)
- 화면 조건:
  - Portrait/ Landscape
  - 전면 미러 on/off
  - 렌즈 스케일 1.0 / 1.5 / 2.5
- 거리 조건:
  - 근거리(눈 크게 보임), 원거리(눈 작게 보임)

## 9) 구현 내역

### Fix-A 구현 (P0) — `OverlayView.kt`

- `rawIrisPaint` 추가 (파란색 점선, `#4488FF`, `DashPathEffect`)
- `drawIrisMarker()` 수정:
  - `rawR = radius * scaleFactor` → 파란색 점선 원 (홍채 실제 크기)
  - `effectiveR = radius * scaleFactor * lensConfig.scale` → 녹색 실선 원 (렌즈 적용 크기)
- 디버그 텍스트에 `rawR`, `effR` 분리 출력

### Fix-B 구현 (P1) — `CameraGLRenderer.kt`

- 반경 정규화 축 수정: `radius / detW` → `radius / detH`
- 근거: 셰이더가 `adjustedCoord = vec2(texCoord.x * aspectRatio, texCoord.y)` 로 isotropic height 단위 공간 사용
- Portrait(1080x1920) 기준 기존 대비 ~1.78x 과대 렌더링 해소

### Fix-C 구현 (P2) — `mediapipe_detector.cpp`

- `validateAndFixIrisCoordinates()` 보정 방식 변경:
  - 기존: 임계 초과 시 iris 중심을 eye center로 즉시 스냅
  - 변경: lerp 보간 (`t = min(excess / threshold / 2, 1.0)`) 으로 점진적 보정
- 임계값 동적화: 고정 `0.05` → 눈폭(`left_eye_width`)의 50%로 비례 조정
- 시선 추적 정보 보존 효과

## 10) 결론

현재 현상은 "디버그 원 계산 기준"과 "실제 홍채 포인트 기준"이 혼재되어 발생하는 인지 불일치가 1차 원인이다.
여기에 GPU 반경 정규화/거리공간 불일치와 V2 eye-center 보정 로직이 결합되면, 실제 렌즈도 홍채보다 크게 보이는 문제가 강화된다.

Fix-A/B/C 적용으로 3개 원인 모두 코드 수정 완료. 디바이스 검증 후 AC-1~4 달성 여부를 확인해야 한다.
