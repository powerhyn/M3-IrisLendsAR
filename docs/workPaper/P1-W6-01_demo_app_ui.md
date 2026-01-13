# P1-W6-01: 데모 앱 UI

**태스크 ID**: P1-W6-01
**상태**: ✅ 완료
**시작일**: 2026-01-13
**완료일**: 2026-01-13

---

## 1. 계획

### 목표
IrisLensSDK 기능을 시연하는 Android 데모 앱 기본 UI 구현. 카메라 프리뷰, 렌즈 선택, 설정 조절 기능 제공

### 산출물
| 파일 | 설명 |
|------|------|
| `android/demo-app/src/main/java/com/irislenssdk/demo/MainActivity.kt` | 메인 액티비티 |
| `android/demo-app/src/main/res/layout/activity_main.xml` | 메인 레이아웃 |
| `android/demo-app/src/main/res/drawable/` | UI 드로어블 리소스 |
| `android/demo-app/src/main/res/values/` | 문자열/색상/테마 리소스 |

### 검증 기준
- [x] 앱 빌드 성공
- [x] 카메라 권한 요청 UI
- [x] 렌즈 텍스처 선택 UI 동작
- [x] 투명도/크기 조절 슬라이더 동작
- [x] 전면/후면 카메라 전환 버튼
- [ ] CameraX 실제 프리뷰 (P1-W6-02에서 구현)

### 선행 조건
- P1-W5-04 AAR 빌드 설정 완료 ✅

---

## 2. 분석

### 2.1 UI 구성

```
┌─────────────────────────────────────┐
│        Camera Preview Area          │  ← CameraX PreviewView
│                                     │
│        (FPS 표시)                   │
│                                     │
│        (상태 메시지)                │
│                                     │
├─────────────────────────────────────┤
│ [Blue] [Green] [Brown] [Gray] [Off] │  ← 렌즈 선택 HorizontalScrollView
├─────────────────────────────────────┤
│ Opacity: 80%    ──────●──────       │  ← Material Slider
│ Scale:   100%   ──────●──────       │  ← Material Slider
├─────────────────────────────────────┤
│   📷     [Capture]     🔄     ⚙️   │  ← 갤러리/캡처/카메라전환/설정
└─────────────────────────────────────┘
```

### 2.2 기술 스택 (실제 구현)

| 컴포넌트 | 기술 |
|----------|------|
| UI 프레임워크 | Android View 시스템 + ViewBinding |
| 카메라 | CameraX (P1-W6-02에서 연동) |
| 슬라이더 | Material Design 3 Slider |
| 렌즈 선택 | HorizontalScrollView + CardView |

### 2.3 SDK 연동 방식

IrisLensSDK는 정적 메서드 패턴 사용:
```kotlin
// 초기화
IrisLensSDK.init(context)

// 버전 확인
IrisLensSDK.getVersion()

// 종료
IrisLensSDK.destroy()
```

---

## 3. 실행 내역

### 3.1 구현 파일 목록

```
android/demo-app/src/main/
├── java/com/irislenssdk/demo/
│   └── MainActivity.kt              # UI 이벤트 핸들러, SDK 연동
├── res/
│   ├── layout/
│   │   └── activity_main.xml        # 전체 UI 레이아웃
│   ├── drawable/
│   │   ├── circle_button_bg.xml     # 버튼 Ripple 배경
│   │   ├── lens_item_selector.xml   # 렌즈 선택 상태
│   │   ├── ic_switch_camera.xml     # 카메라 전환 아이콘
│   │   ├── lens_blue_thumb.xml      # 파란 렌즈 썸네일
│   │   ├── lens_green_thumb.xml     # 초록 렌즈 썸네일
│   │   ├── lens_brown_thumb.xml     # 갈색 렌즈 썸네일
│   │   ├── lens_gray_thumb.xml      # 회색 렌즈 썸네일
│   │   └── lens_off_thumb.xml       # 렌즈 끄기 썸네일
│   └── values/
│       ├── strings.xml              # 문자열 리소스
│       ├── colors.xml               # 색상 리소스
│       └── themes.xml               # 테마 설정
```

### 3.2 MainActivity 주요 기능

```kotlin
class MainActivity : AppCompatActivity() {
    // 상태 변수
    private var isSDKInitialized: Boolean = false
    private var currentLensId: String = "off"
    private var isFrontCamera: Boolean = true
    private var lensOpacity: Float = 0.8f
    private var lensScale: Float = 1.0f

    // 핵심 메서드
    private fun initializeSDK()      // SDK 초기화 (정적 메서드)
    private fun releaseSDK()         // SDK 해제
    private fun selectLens(lensId)   // 렌즈 선택
    private fun applyLensSettings()  // 설정 적용
    private fun switchCamera()       // 카메라 전환
    private fun captureImage()       // 이미지 캡처
}
```

### 3.3 레이아웃 구조

- **PreviewView**: CameraX 카메라 프리뷰 (전체 화면)
- **상태 표시**: FPS, 상태 메시지 오버레이
- **렌즈 선택기**: 5개 렌즈 옵션 (HorizontalScrollView)
- **슬라이더**: 투명도(0-100%), 크기(50-150%)
- **하단 버튼**: 갤러리, 캡처, 카메라 전환, 설정

### 3.4 권한 처리

ActivityResultContracts 기반 권한 요청:
```kotlin
private val requestPermissionLauncher = registerForActivityResult(
    ActivityResultContracts.RequestMultiplePermissions()
) { permissions ->
    val allGranted = permissions.all { it.value }
    if (allGranted) {
        initializeSDK()
        startCamera()
    } else {
        // 권한 거부 처리
    }
}
```

---

## 4. 검증 결과

### 검증 항목

| 항목 | 결과 | 비고 |
|------|------|------|
| demo-app 빌드 | ✅ 성공 | assembleDebug 성공 |
| SDK 초기화 연동 | ✅ 성공 | 정적 메서드 패턴 사용 |
| 렌즈 선택 UI | ✅ 구현 완료 | 5개 렌즈 + 선택 상태 표시 |
| 슬라이더 UI | ✅ 구현 완료 | 투명도/크기 조절 |
| 버튼 UI | ✅ 구현 완료 | 캡처/갤러리/카메라전환/설정 |
| CameraX 프리뷰 | ⏳ P1-W6-02 | 레이아웃만 준비, 실제 연동은 다음 태스크 |

### 빌드 결과

```
$ ./gradlew :demo-app:assembleDebug
BUILD SUCCESSFUL in 4s
66 actionable tasks: 13 executed, 53 up-to-date
```

---

## 5. 이슈 및 학습

### 이슈
| ID | 내용 | 상태 | 해결방안 |
|----|------|------|----------|
| 1 | SDK API 불일치 | ✅ 해결 | getInstance() → init() 정적 메서드로 변경 |

### 결정 사항
| 결정 | 이유 |
|------|------|
| View 시스템 사용 | 기존 demo-app 구조 유지, CameraX와 안정적 연동 |
| Material 3 Slider | 현대적 UI, 터치 피드백 우수 |
| HorizontalScrollView | 렌즈 개수 확장 용이 |
| ViewBinding | 타입 안전한 뷰 접근 |

### 학습 내용
- IrisLensSDK 정적 API 패턴 (싱글톤이 아닌 정적 메서드)
- Material 3 Slider 구현 (stepSize, value range)
- 드로어블 선택자 (selector)를 통한 UI 상태 관리

---

## 6. 다음 단계

### P1-W6-02: CameraX 통합
- [ ] CameraX 프리뷰 연동
- [ ] ImageAnalysis 콜백 설정
- [ ] SDK 홍채 검출 파이프라인 연결
- [ ] 실시간 FPS 측정

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-01-12 | 태스크 문서 생성 |
| 2026-01-13 | UI 구현 완료, SDK API 연동 수정, 빌드 검증 완료 |
