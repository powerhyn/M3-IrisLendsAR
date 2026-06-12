# REFACTOR-DEMO: 데모 검증 통로 정화 W

| 항목 | 내용 |
|---|---|
| 상태 | ✅ 구현 완료 (2026-06-12) — 실기기 육안 확인(사용자) 후 머지 |
| 브랜치 | `refactor/demo-verification-cleanup` (develop 016ebce 기준) |
| 근거 | 감사 §9 실행 순서 ① "데모 검증 통로 정화" + ADR-0001 §12 게이트 3 — 모든 실기기 검증의 선행 조건 |
| 목적 | 검증 도구(데모)가 거짓말하지 않게 — torn 좌표/무음 폴백/독립 정책 제거 |

## 수행 내역 (감사 확정 findings 기준)

| # | finding (severity) | 조치 |
|---|---|---|
| A | 공유 IrisResult 가변 인스턴스 torn read (major×2) | `glIrisResult`/`uiIrisResult` 공유 필드 삭제 → 프레임별 새 복사본으로 소유권 이전. CameraGLView에 호출 계약 주석 |
| B | KT fallback 셰이더 blendMode 불일치 + 무음 폴백 (major) | **fallback 일체 731줄 삭제** (LENS_OVERLAY 셰이더 286줄, renderLensOverlay 293줄, fitEyeEllipse, 유니폼 location 30개, ellipse/eyelid 캐시, 윤곽 상수). SDK 실패 시 렌즈 미적용 + `sdkLensFailure` 신호 → FPS HUD "⚠ SDK 렌즈 실패: 사유" + 60프레임 스로틀 Log.e |
| C | OverlayView 독립 정책 — confidence 게이트(0.5)·2초 홀드가 GL과 다른 좌표 표시 (major) | debugMode에서 게이트·홀드 우회: 이번 프레임 검출 사실만, GL에 전달되는 raw 좌표 그대로. 일반 모드 동작 불변 |
| D | GpuRenderActivity.yuvToNv21 rowStride 미처리 — 검출 입력 오염 (major) | stride-aware 구현 교체 (+ 마지막 chroma 바이트 보충). 공용 유틸 추출은 ③-2 이월 |
| E | 프레임-랜드마크 동기화 부재 (major — 단기 조치만) | 렌더 시점 landmark age 120프레임 주기 로그 (정량화). 구조 개선(frame_id 튜플 주입)은 ④ |

부수: 메모리 `w9-demo-ui-sync` 잔여 "KT fallback 셰이더 동기화/제거" 항목이 본 W로 종결.

## 검증

- 빌드 성공 (CameraGLRenderer 1949→1218+α줄), S23+ 설치·기동 정상, 크래시 0
- 신규 로그 동작 확인: SDK 렌즈 실패 신호 미발생(정상), 뷰티 파라미터 로그 가동
- 실기기 육안 (사용자): 렌즈 정합 동일 + debugMode 오버레이가 검출 사실을 즉답하는지

## 부가 발견 (이번 W 범위 외 — 기록)

- **뷰티 pass-through 단서**: 데모 초기 상태에서 `smoothing/skinQuality/softFocus/whitening=0.0` 으로 SDK 뷰티가 pass-through 됨 (로그 실측). "뷰티가 예전 기능 느낌" 관찰의 유력 원인 후보 — P8-W1 피부 보정이 파라미터 0으로 사실상 비활성. 데모 기본값/프리셋 점검을 P8-W1 잔여 검증과 함께 처리할 것

## 이월 (데모 품질 — 검증 통로와 무관, ③-2 또는 별도)

- GL 컨텍스트 수명/EGL 재생성 stale Surface (major 2건), createProgram 누수, 매 프레임 glGetUniformLocation, TransformationInfo(hasCameraTransform) 미처리, FrameAnalyzer 마지막 chroma 바이트, OneEuro Kotlin/C++ 분기, eyelidFeather 죽은 코드(삭제로 해소됨)

## 변경 이력

- 2026-06-12: A~E 구현, 빌드·설치·기동 검증
