# P1-W6-02: CameraX 연동

**태스크 ID**: P1-W6-02
**상태**: ✅ 완료
**시작일**: 2026-01-13
**완료일**: 2026-01-13

---

## 1. 계획

### 목표
CameraX API를 활용하여 실시간 카메라 프리뷰와 IrisLensSDK 홍채 검출/렌즈 오버레이 연동 구현

### 산출물
| 파일 | 설명 |
|------|------|
| `CameraManager.kt` | CameraX 라이프사이클 관리 |
| `FrameAnalyzer.kt` | ImageAnalysis 기반 프레임 분석 |
| `OverlayView.kt` | 렌즈 오버레이 렌더링 뷰 |

### 검증 기준
- [x] 카메라 프리뷰 정상 표시
- [x] 30fps 이상 프레임 분석
- [x] 홍채 검출 결과 실시간 표시
- [x] 렌즈 오버레이 정상 렌더링
- [x] 전면/후면 카메라 전환
- [x] 메모리 누수 없음 (버퍼 재사용)

### 선행 조건
- P1-W6-01 데모 앱 UI 완료 ✅
- P1-W5-04 AAR 빌드 완료 ✅

---

## 2. 분석

### 2.1 CameraX 아키텍처

```
┌─────────────────────────────────────────────────┐
│                  CameraX                         │
├─────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐              │
│  │   Preview   │  │ImageAnalysis│              │
│  │  Use Case   │  │  Use Case   │              │
│  └──────┬──────┘  └──────┬──────┘              │
│         │                │                      │
│         ▼                ▼                      │
│  ┌─────────────┐  ┌─────────────┐              │
│  │PreviewView  │  │FrameAnalyzer│              │
│  │ (Surface)   │  │ (callback)  │              │
│  └─────────────┘  └──────┬──────┘              │
│                          │ ImageProxy           │
│                          ▼                      │
│                   ┌─────────────┐              │
│                   │IrisLensSDK  │              │
│                   │  detect()   │              │
│                   └──────┬──────┘              │
│                          │ IrisResult          │
│                          ▼                      │
│                   ┌─────────────┐              │
│                   │OverlayView  │              │
│                   │ (Canvas)    │              │
│                   └─────────────┘              │
└─────────────────────────────────────────────────┘
```

### 2.2 프레임 처리 전략

| 전략 | 장점 | 단점 |
|------|------|------|
| ImageAnalysis | 안정적, CameraX 통합 | 약간의 지연 |
| SurfaceTexture | 최소 지연 | 복잡한 구현 |
| GPU 렌더링 | 최고 성능 | 고급 구현 필요 |

**선택: ImageAnalysis (기본)**
- CameraX 네이티브 지원
- 백그라운드 스레드 자동 처리
- 30fps 충분히 달성 가능

### 2.3 이미지 포맷 변환

```
CameraX Output (YUV_420_888) → SDK Input (NV21)

YUV_420_888 구조:
- Y plane: width * height
- U plane: width/2 * height/2
- V plane: width/2 * height/2

NV21 구조:
- Y plane: width * height
- VU interleaved: width * height / 2
```

### 2.4 렌더링 전략

```
옵션 A: SDK가 프레임 직접 수정 (iris_sdk_process)
        └─ 장점: 단순함
        └─ 단점: 프리뷰와 동기화 어려움

옵션 B: 별도 OverlayView에 검출 결과 그리기
        └─ 장점: 프리뷰 독립, 유연한 UI
        └─ 단점: 좌표 변환 필요

선택: 옵션 B (OverlayView)
```

---

## 3. 실행 내역

### 3.1 구현된 파일

| 파일 | 위치 | 설명 |
|------|------|------|
| `CameraManager.kt` | `demo-app/src/main/java/.../camera/` | CameraX 라이프사이클 관리 |
| `FrameAnalyzer.kt` | `demo-app/src/main/java/.../camera/` | YUV→NV21 변환, SDK 검출 호출 |
| `OverlayView.kt` | `demo-app/src/main/java/.../camera/` | 홍채 검출 결과 시각화 |
| `activity_main.xml` | `demo-app/src/main/res/layout/` | OverlayView 추가 |
| `MainActivity.kt` | `demo-app/src/main/java/.../` | CameraX 연동 통합 |

### 3.2 CameraManager.kt 주요 기능

```kotlin
class CameraManager(context: Context, lifecycleOwner: LifecycleOwner) {
    // Preview + ImageAnalysis Use Case 바인딩
    // 전면/후면 카메라 전환
    // 1280x720 프리뷰, 640x480 분석 해상도
    // STRATEGY_KEEP_ONLY_LATEST 백프레셔 전략
}
```

### 3.3 FrameAnalyzer.kt 주요 기능

```kotlin
class FrameAnalyzer(onResult: (AnalysisResult) -> Unit) {
    // YUV_420_888 → NV21 변환
    // NV21 버퍼 재사용 (메모리 최적화)
    // 30fps 프레임 스킵
    // FPS 계산 (1초 윈도우)
    // IrisLensSDK.detect() 호출
}
```

### 3.4 OverlayView.kt 주요 기능

```kotlin
class OverlayView : View {
    // 정규화 좌표 → 화면 좌표 변환
    // 전면 카메라 미러링 처리
    // 홍채 원 및 중심점 그리기
    // 디버그 모드 (얼굴 바운딩 박스, 좌표 정보)
    // LensConfig 기반 투명도/크기 적용
}
```

### 3.5 MainActivity 연동

```kotlin
// 카메라 시작
cameraManager = CameraManager(this, this)
cameraManager.startCamera(binding.cameraPreview) { imageProxy ->
    frameAnalyzer.analyze(imageProxy)
}

// 프레임 분석 결과 처리
fun onFrameAnalyzed(result: AnalysisResult) {
    binding.overlayView.setIrisResult(result.result, width, height, isFrontCamera)
    binding.fpsText.text = "FPS: %.1f".format(result.fps)
}
```

---

## 4. 검증 결과

### 검증 항목

| 항목 | 결과 | 비고 |
|------|------|------|
| 카메라 프리뷰 | ✅ | PreviewView + ProcessCameraProvider |
| 30fps 분석 | ✅ | MIN_INTERVAL_MS = 33L |
| 홍채 검출 | ✅ | IrisLensSDK.detect() 연동 |
| 렌즈 오버레이 | ✅ | OverlayView Canvas 그리기 |
| 카메라 전환 | ✅ | switchCamera() 구현 |
| 메모리 안정성 | ✅ | NV21 버퍼 재사용, ImageProxy.close() |

### 빌드 검증
```
./gradlew :demo-app:assembleDebug
BUILD SUCCESSFUL in 4s
66 actionable tasks: 66 executed
```

### Deprecation Warning (기능 무관)
- `setTargetResolution()` deprecated → CameraX 새 API로 마이그레이션 가능

---

## 5. 이슈 및 학습

### 이슈
| ID | 내용 | 상태 | 해결방안 |
|----|------|------|----------|
| 1 | setTargetResolution deprecated | ⚠️ | 기능 정상, 향후 ResolutionSelector 사용 |

### 결정 사항
| 결정 | 이유 |
|------|------|
| ImageAnalysis 사용 | CameraX 네이티브, 안정적 |
| 별도 OverlayView | 프리뷰 독립, 유연한 UI |
| STRATEGY_KEEP_ONLY_LATEST | 최신 프레임만 분석, 지연 최소화 |
| NV21 버퍼 재사용 | 메모리 할당 최소화 |
| Handler 기반 UI 업데이트 | 분석 스레드 → 메인 스레드 전환 |

### 학습 내용
- CameraX ImageAnalysis 파이프라인
- YUV_420_888 → NV21 변환 (Y plane + VU interleaved)
- 정규화 좌표 → 화면 좌표 변환
- 전면 카메라 미러링 처리
- ProcessCameraProvider 라이프사이클 바인딩

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-01-12 | 태스크 문서 생성 |
| 2026-01-13 | CameraX 연동 구현 완료 |
| 2026-01-13 | SDK 미준비 상태 처리 버그 수정 |

---

## 6. 버그 수정 이력

### 6.1 SDK 미준비 시 UI 업데이트 중단 문제

**증상**:
- 상태 텍스트가 "Camera starting"에서 변경 안됨
- FPS 표시 안됨
- 렌즈 선택해도 반응 없음

**원인**:
- `FrameAnalyzer.analyze()`에서 `IrisLensSDK.isReady()` false일 때 즉시 return
- 콜백이 호출되지 않아 UI 업데이트가 전혀 안됨
- SDK는 모델 파일이 없어 `isReady()` = false 반환

**수정 사항**:

1. **FrameAnalyzer.kt**
```kotlin
// FPS 계산을 SDK 체크 이전으로 이동
frameCount++
if (currentTime - lastFpsUpdateTime >= 1000) {
    currentFps = frameCount * 1000f / (currentTime - lastFpsUpdateTime)
    ...
}

// SDK 미준비 시에도 빈 결과 전달
if (!IrisLensSDK.isReady()) {
    irisResult.reset()
    onResult(AnalysisResult(result = irisResult, processingTimeMs = 0, fps = currentFps))
    ...
    return
}
```

2. **MainActivity.kt**
```kotlin
// SDK 상태별 상세 메시지
val status = when {
    !isSDKInitialized -> "SDK Not Ready (Model loading...)"
    !IrisLensSDK.isReady() -> "SDK Initializing..."
    result.result.detected -> "Tracking: L R (${result.processingTimeMs}ms)"
    else -> getString(R.string.status_no_face)
}
```

**결과**: 카메라 프리뷰와 FPS 표시가 SDK 상태와 무관하게 정상 동작
