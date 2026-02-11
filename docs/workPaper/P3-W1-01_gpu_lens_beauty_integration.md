# P3-W1-01: GPU 렌즈 + 뷰티 통합

## 작업 개요
- **Phase**: P3 (렌즈 + 뷰티 GPU 통합 및 기능 고도화)
- **기간**: 2026-02-06
- **상태**: ✅ 완료

## 목표
GpuRenderActivity에 렌즈 기능을 추가하여 렌즈와 뷰티 필터를 GPU 파이프라인에서 통합 렌더링

## 수행 내역

### 1. GPU 렌즈 셰이더 구현 (`CameraGLRenderer.kt`)
- **렌즈 오버레이 프래그먼트 셰이더** 추가
  - 블렌드 모드 지원 (Normal, Multiply, Screen, Overlay)
  - 가장자리 페더링 (smoothstep)
  - 양쪽 눈 독립 적용 가능
- **렌즈 FBO/텍스처** 관리
- **렌즈 텍스처 업로드** (GL 스레드 안전)
- **파이프라인 수정**: OES → RGBA → **렌즈** → 뷰티 → 화면

### 2. Tab UI 레이아웃 (`activity_gpu_render.xml`)
- **TabLayout** 추가 (렌즈/뷰티 탭)
- **렌즈 탭 콘텐츠**:
  - RecyclerView (렌즈 선택 그리드)
  - 슬라이더: 투명도, 크기, 경계
- **뷰티 탭 콘텐츠**:
  - 토글 버튼
  - 슬라이더: 스무딩, 밝기, 화이트닝, 컬러밸런스, 소프트포커스

### 3. Activity 수정 (`GpuRenderActivity.kt`)
- **LensManager** 연동
- **LensAdapter** RecyclerView 설정
- **렌즈 설정 슬라이더** 연결
- **Tab 전환 리스너** 구현
- **상태 표시** (Lens Status)

### 4. CameraGLView API 확장
- `setLensConfig(config: LensConfig)` 추가
- `setLensEnabled(enabled: Boolean)` 추가
- `setLensTexture(bitmap: Bitmap?)` 추가

## 파일 변경 목록

| 파일 | 변경 내용 |
|------|----------|
| `CameraGLRenderer.kt` | 렌즈 셰이더, FBO, 렌더링 파이프라인 |
| `CameraGLView.kt` | 렌즈 API (setLensConfig, setLensTexture 등) |
| `GpuRenderActivity.kt` | LensManager 연동, Tab UI 로직 |
| `activity_gpu_render.xml` | Tab 기반 UI 레이아웃 |

## 렌더링 파이프라인

```
OES Texture (카메라)
       │
       ▼ OES → RGBA 변환
RGBA FBO
       │
       ▼ 렌즈 오버레이 (IrisResult + LensConfig)
Lens FBO
       │
       ▼ GPU Beauty Filter
Beauty FBO
       │
       ▼ Passthrough
Screen
```

## 검증 결과

- ✅ 빌드 성공: `./gradlew :demo-app:assembleDebug`
- ⏳ 기기 테스트 필요: Tab 전환, 렌즈 적용, 렌즈+뷰티 동시 적용

## 다음 단계

1. **기기 테스트**: 실제 디바이스에서 렌즈 렌더링 검증
2. **성능 최적화**: 렌즈 + 뷰티 동시 적용 시 FPS 측정
3. **셰이더 튜닝**: 블렌드 모드, 페더링 파라미터 조정
4. **추가 기능**: 렌즈 회전/오프셋 지원

## 변경 이력

| 날짜 | 내용 |
|------|------|
| 2026-02-06 | 초기 구현 완료 |
