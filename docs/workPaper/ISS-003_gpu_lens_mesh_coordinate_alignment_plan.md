# ISS-003: GPU 렌더링 경로에서 Mesh 불일치 및 렌즈 세로 늘어짐 정렬 계획

## 작업 정보

| 항목 | 내용 |
|------|------|
| 이슈 ID | ISS-003 |
| 상태 | 🟡 계획 수립 |
| 우선순위 | P0 (Critical) |
| 시작일 | 2026-02-11 |
| 완료 목표 | 2026-02-13 |
| 관련 이슈 | `ISS-001`, `ISS-002` |
| 대상 모듈 | `android/demo-app` (GPU renderer, overlay), `cpp` (검출 결과 좌표 계약) |

---

## 1. 문제 요약

GPU 데모 경로에서 아래 증상이 동시에 관찰됨.

1. Face Mesh가 얼굴 실루엣과 미세하게 불일치
2. 렌즈가 원형이 아닌 세로로 늘어난 타원 형태로 렌더링

관찰 환경:
- `GpuRenderActivity` 실시간 프리뷰 (전면 카메라)
- Portrait UI
- 렌즈 + Mesh 디버그 동시 확인 시 재현

---

## 2. 원인 가설 (우선순위 순)

### 가설 A (P0): 회전된 검출 좌표와 렌더 좌표의 기준 프레임이 다름

근거:
- 검출은 회전 적용 후 이미지 기준으로 수행됨
  - `cpp/src/frame_processor.cpp:695`
  - `cpp/src/frame_processor.cpp:718`
- 검출 결과 프레임 메타는 해당 회전 후 해상도 기준으로 기록됨
  - `cpp/src/mediapipe_detector.cpp:2121`
- 렌즈 렌더는 `fboWidth/fboHeight`를 기준으로 반경 정규화 및 aspect 계산
  - `android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLRenderer.kt:618`
  - `android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLRenderer.kt:661`

영향:
- `detect space`와 `render space`가 섞이면 `uFrameAspect` 보정 축이 틀어져 렌즈가 타원으로 왜곡됨.

### 가설 B (P0): 렌즈 반경 정규화 축이 불명확함

근거:
- 셰이더는 x축을 `aspectRatio`로 변환해 y축 기준 공간에서 거리 계산
  - `android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLRenderer.kt:186`
- 반경은 현재 width 기준 정규화
  - `android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLRenderer.kt:618`

영향:
- 현재 수식 계약이 명확히 문서화되어 있지 않아 회전/해상도 조합에서 반경 스케일 오차가 누적될 수 있음.

### 가설 C (P1): Overlay와 GL 본 렌더의 화면 매핑 정책이 다름

근거:
- Overlay는 `fill-center(max)` 기준 매핑
  - `android/demo-app/src/main/java/com/irislenssdk/demo/camera/OverlayView.kt:476`
- GL 최종 출력은 `fit` 성격 스케일(축소) 사용
  - `android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLRenderer.kt:861`

영향:
- Mesh 디버그 점과 실제 GL 영상의 대응이 어긋나 보임.

### 가설 D (P1): 분석/렌더 간 `IrisResult` 공유 객체 재사용으로 프레임 경합

근거:
- Activity에서 단일 `irisResult` 객체 재사용
  - `android/demo-app/src/main/java/com/irislenssdk/demo/GpuRenderActivity.kt:130`
  - `android/demo-app/src/main/java/com/irislenssdk/demo/GpuRenderActivity.kt:720`

영향:
- GL 스레드/UI 스레드가 다른 프레임 시점 데이터를 읽으며 불일치가 악화될 수 있음.

---

## 3. 수정 전략

핵심 원칙:
- 좌표 계약을 단일화한다.
- 검출 결과를 소비하는 모든 렌더 경로가 동일한 프레임 기준(`width/height/rotation`)을 사용한다.

### 전략 1 (필수): 렌즈 셰이더 입력 기준 통일

수정 대상:
- `android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLRenderer.kt`

작업:
1. `result.frameWidth/frameHeight`를 렌즈 계산의 1순위 기준으로 사용
2. `uFrameAspect`를 동일 기준으로 산출
3. 반경 정규화 축(width/height)을 셰이더 수식 계약과 일치하도록 고정
4. 회전(90/270) 상황에서 width/height swap 여부를 단일 함수로 캡슐화

산출물:
- `resolveCoordinateSpace(result, frameRotation, fboW, fboH)` 유틸 함수
- 로그: `detW/detH`, `renderW/renderH`, `aspect`, `leftRadiusNorm/rightRadiusNorm`

### 전략 2 (필수): Overlay 좌표 매핑을 GL 정책과 맞춤

수정 대상:
- `android/demo-app/src/main/java/com/irislenssdk/demo/camera/OverlayView.kt`

작업:
1. GPU 모드에서는 `fit` 기반 매핑 사용 (GL 화면 출력과 동일)
2. 기존 CPU/PreviewView 경로와 호환되도록 매핑 모드를 플래그화
3. Mesh/iris marker/lens debug draw 모두 같은 변환 함수 재사용

산출물:
- `computeScreenTransform(imageW, imageH, viewW, viewH, mode)` 공통 함수

### 전략 3 (권장): 결과 객체 불변 스냅샷 전달

수정 대상:
- `android/demo-app/src/main/java/com/irislenssdk/demo/GpuRenderActivity.kt`
- `android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLView.kt`

작업:
1. Analyzer에서 `IrisResult`를 깊은 복사한 스냅샷을 GL/UI에 전달
2. 스레드 간 공유 mutable 객체 사용 제거

산출물:
- `copyIrisResult(src)` 유틸
- GL/UI 전달 타입 안정화

---

## 4. 상세 작업 항목 (체크리스트)

### Phase 0: 진단 로깅 (0.5일)
- [ ] `CameraGLRenderer`에 렌즈 입력 파라미터 로그 추가
- [ ] Overlay의 `scaleFactor/offset` 로그와 GL의 `scaleX/scaleY` 로그 동시 수집
- [ ] 회전값(0/90/270)별 재현 스크린샷 저장

### Phase 1: 렌즈 왜곡 수정 (0.5일)
- [ ] `uFrameAspect` 기준 해상도 통일
- [ ] 반경 정규화 수식 고정 및 코드 주석 명문화
- [ ] 좌/우 미러 교환 시 반경/중심 동시 일관성 확인

### Phase 2: Mesh 정렬 수정 (0.5일)
- [ ] Overlay 매핑을 GPU 모드에서 GL과 동일 정책으로 전환
- [ ] Mesh/iris marker 기준점 오차 확인

### Phase 3: 동시성 안정화 (0.5일)
- [ ] `IrisResult` 스냅샷 전달 구조 적용
- [ ] GL/UI 경합 이슈 재현 불가 확인

---

## 5. 검증 계획

### 기능 검증
1. 전면 카메라, Portrait, 렌즈 ON
2. Mesh ON/OFF, Debug ON/OFF
3. 회전 상태 0/90/270 모두 확인

### 시각 품질 기준
- 렌즈 원형도: 정면 기준 가로/세로 직경 비율 `0.95 ~ 1.05`
- Mesh 중심 오차: 양안 중심 기준 렌즈 중심과의 거리 `<= 5 px` (1080p 기준)
- 좌우 눈 뒤바뀜/미러 반전 오류 없음

### 성능/안정성 기준
- Render FPS 저하 5% 이내
- 2분 연속 구동 시 텍스처 누수/깜빡임/튀는 프레임 없음

---

## 6. 리스크 및 대응

리스크:
1. 기존 PreviewView 경로와 좌표 정책 충돌 가능
2. 회전/미러링 조합에서 조건 분기 복잡도 증가
3. 스냅샷 복사로 미세한 메모리/CPU 오버헤드 발생

대응:
1. 매핑 모드를 GPU/Legacy 분리
2. 회전/미러 계산을 단일 유틸 함수로 통합
3. `IrisResult` 복사 비용 프로파일링 후 필요 시 풀링 적용

---

## 7. 완료 정의 (Definition of Done)

아래를 모두 만족하면 완료로 판단:
1. 렌즈가 타원 변형 없이 원형으로 유지됨
2. Mesh 디버그 점이 얼굴 윤곽과 안정적으로 일치함
3. 0/90/270 회전 및 전면 미러에서 동일 품질 유지
4. 코드에 좌표 계약(기준 프레임, 정규화 축)이 주석으로 명시됨

---

## 8. 관련 코드 레퍼런스

- `android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLRenderer.kt:618`
- `android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLRenderer.kt:661`
- `android/demo-app/src/main/java/com/irislenssdk/demo/camera/OverlayView.kt:476`
- `cpp/src/frame_processor.cpp:695`
- `cpp/src/mediapipe_detector.cpp:2121`
