# P1-W6-06: Face Mesh 허공 감지 (False Positive) 문제 수정

## 작업 정보

| 항목 | 내용 |
|------|------|
| 작업 ID | P1-W6-06 |
| 상태 | ✅ 완료 |
| 시작일 | 2026-01-20 |
| 완료일 | 2026-01-20 |

---

## 문제 증상

1. **False Positive 감지**: 얼굴이 없는 천장/벽에서 Face Mesh가 그려짐
2. **이중 감지**: 실제 얼굴 + 머리 위 허공에 추가 Face Mesh 표시
3. **공통 원인**: 낮은 신뢰도 검출 결과가 필터링 없이 렌더링됨

---

## 근본 원인 분석

### 원인 1: 낮은 얼굴 검출 신뢰도 임계값
- **위치**: `cpp/src/mediapipe_detector.cpp:130`
- **현재값**: `min_detection_confidence = 0.3f`
- **문제**: 30%는 너무 낮아서 천장 조명/환기구가 얼굴로 오인식됨

### 원인 2: 추적 캐시가 False Positive 지속
- **위치**: `cpp/src/mediapipe_detector.cpp:1918-1939`
- **문제**: False Positive가 캐시되어 여러 프레임에 걸쳐 지속됨

### 원인 3: 렌더링 레이어에서 신뢰도 검증 부재
- **위치**: `OverlayView.kt`
- **문제**: `result.confidence` 검증 없이 `detected=true`만 확인 후 렌더링

---

## 수정 내용

### 수정 파일
`android/demo-app/src/main/java/com/irislenssdk/demo/camera/OverlayView.kt`

### 변경 1: MIN_RENDER_CONFIDENCE 상수 추가 (72-75행)
```kotlin
// 최소 렌더링 신뢰도 임계값 (False Positive 방지)
// 이 값 미만의 신뢰도를 가진 검출 결과는 렌더링하지 않음
// 허공/천장 감지 문제 해결을 위해 추가
private const val MIN_RENDER_CONFIDENCE = 0.5f
```

### 변경 2: onDraw() 시작에 신뢰도 검증 추가 (262-264행)
```kotlin
// 신뢰도 검증 (False Positive 방지)
// 낮은 신뢰도의 검출 결과는 허공/천장 오인식일 가능성이 높음
if (result.confidence < MIN_RENDER_CONFIDENCE) return
```

### 변경 3: Face Mesh 표시 조건 강화 (334-338행)
```kotlin
// Face Mesh 표시 (충분한 신뢰도로 얼굴 감지 시에만)
// 이중 검증: onDraw() 시작의 신뢰도 검증 + 여기서의 추가 검증 (방어적 프로그래밍)
if (showFaceMesh && result.faceMeshValid && result.faceMesh != null
    && result.confidence >= MIN_RENDER_CONFIDENCE) {
    drawFaceMesh(canvas, result, scaleFactor, offsetX, offsetY)
}
```

---

## 검증 방법

1. 앱 빌드 후 설치
2. 천장/벽만 보이게 카메라 향함 → Face Mesh 미표시 확인
3. 얼굴 감지 시 정상 추적 확인
4. 디버그 모드에서 confidence 값 모니터링

---

## 향후 고려사항 (C++ 레이어)

SDK 레벨에서 근본 수정이 필요하면:
- `min_detection_confidence`: 0.3 → 0.5
- `min_presence_confidence`: 0.5 → 0.6

현재 수정은 렌더링 레이어에서의 필터링으로, 즉각적인 효과를 제공합니다.
SDK 레벨 수정은 별도 작업으로 진행 가능합니다.

---

## 변경 이력

| 날짜 | 작업자 | 내용 |
|------|--------|------|
| 2026-01-20 | Claude | OverlayView.kt에 신뢰도 기반 렌더링 필터 추가 |
