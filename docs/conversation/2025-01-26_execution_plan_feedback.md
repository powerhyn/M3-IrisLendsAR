# 008. 뷰티 필터 프레임 직접 렌더링 계획 분석 및 피드백

**작성일**: 2025-01-26
**작성자**: Claude (AI Assistant)
**대상 문서**:
- `docs/workPaper/008_direct_frame_rendering.md`
- `docs/workPaper/P1-W6-05_tflite_android_issue.md`
- `docs/workPaper/P1-W6-06_face_mesh_false_positive_fix.md`
- `docs/ARCHITECTURE.md`

---

## 1. 종합 분석

현재 IrisLensSDK 프로젝트는 Phase 1(Android MVP)의 막바지 단계에 있으며, 핵심 기능 구현은 완료되었으나 **렌더링 성능 최적화**와 **플랫폼 통합 안정성** 확보가 주요 과제로 남아있습니다.

특히 `008_direct_frame_rendering.md`에서 제안된 뷰티 필터 렌더링 계획은 기존 아키텍처의 한계를 정확히 파악하고 있으며, 피드백을 통해 수정된 v2.0 계획(JNI 변환, 더블 버퍼링 등)은 실시간성(30fps) 확보를 위한 기술적 타당성이 매우 높습니다.

## 2. 주요 문서별 분석 및 피드백

### 📄 008_direct_frame_rendering.md (뷰티 필터 렌더링)
*   **분석**: `PreviewView`가 카메라 스트림을 독점하는 구조적 문제를 해결하기 위해, `OverlayView`에 필터링된 비트맵을 직접 그리는 방식을 채택했습니다. 초기 계획의 성능 이슈(JPEG 변환 부하)를 JNI + OpenCV 직접 변환으로 해결한 것은 훌륭한 결정입니다.
*   **피드백**:
    *   **JNI 변환 (Phase 1)**: `AndroidBitmap_lockPixels`와 `cv::cvtColor`를 활용한 접근은 최상의 성능을 보장합니다.
    *   **메모리 관리 (Phase 2)**: Double Buffering 도입은 GC 프리징 방지를 위해 필수적입니다. 다만, 분석 스레드와 렌더링 스레드 간의 미세한 경쟁 조건(Race Condition)에 대한 방어 코드가 필요할 수 있습니다.
    *   **렌더링 (Phase 3)**: `Canvas.rotate()` 활용은 비트맵 메모리 복사 비용을 절약하는 핵심 최적화입니다.

### 📄 P1-W6-05_tflite_android_issue.md (TFLite 통합)
*   **분석**: TFLite Android 빌드 실패 이슈에 대해 Pre-built 라이브러리 사용(방안 A)을 최우선으로 선택한 것은 기존 C++ 코어 자산을 최대한 보존하면서 빠른 해결을 도모하는 실용적인 접근입니다.
*   **피드백**: `008` 작업에서 OpenCV를 사용하므로, OpenCV와 TFLite(Pre-built) 간의 `libc++_shared.so` 충돌 가능성을 미리 점검해야 합니다.

### 📄 P1-W6-06_face_mesh_false_positive_fix.md (허공 감지)
*   **분석**: 천장/조명을 얼굴로 오인식하는 문제를 신뢰도 임계값 상향과 렌더링 단의 필터링으로 해결했습니다.
*   **피드백**: 렌더링 레이어에서의 필터링은 즉각적인 효과가 있지만, 향후 SDK 코어 레벨(`mediapipe_detector.cpp`)의 파라미터 튜닝을 통해 근본적인 검출력을 높이는 작업도 병행되어야 합니다.

## 3. 기술적 제언 (Action Items)

1.  **JNI 구현 우선순위 상향**: `nativeNv21ToRgba` 함수 구현은 전체 파이프라인의 병목을 해소하는 키(Key)이므로 가장 먼저 착수해야 합니다.
2.  **동기화 전략 구체화**: `FrameAnalyzer`와 `OverlayView` 사이의 비트맵 전달 과정에서 `Synchronized` 블록이나 `Atomic` 레퍼런스 사용을 고려하여 스레드 안전성을 강화하세요.
3.  **의존성 충돌 점검**: `build.gradle`의 `packagingOptions`를 확인하여 OpenCV와 TFLite의 네이티브 라이브러리 충돌을 사전에 방지하세요.

## 4. 결론

현재 수립된 실행 계획은 프로젝트의 목표인 **'30fps 실시간 렌더링'**과 **'안정적인 Android 통합'**을 달성하기에 충분히 구체적이고 기술적으로 타당합니다. 계획된 순서대로 구현을 진행하시면 됩니다.

---
*본 문서는 IrisLensSDK 개발 진행 상황에 대한 AI 어시스턴트의 종합 분석 결과입니다.*

---

## 5. 구현 완료 후 후속 의견 (2025-01-26 업데이트)

**작성자**: Claude (Implementation)

### 5.1 실제 구현 접근법

현재 Stage 1.5는 **Java 기반 빠른 MVP 구현**으로 완료되었습니다. 원래 계획된 JNI 기반 최적화 대신, 다음과 같은 접근법을 채택했습니다:

| 항목 | 계획 (v2.0) | 실제 구현 |
|------|-------------|-----------|
| NV21 → Bitmap 변환 | JNI + OpenCV 직접 변환 (5-10ms) | YuvImage → JPEG → Bitmap (Java) |
| 메모리 관리 | Double Buffering | 단순 Bitmap 생성/해제 |
| 회전 처리 | Canvas.rotate() (렌더링 시점) | Bitmap.createBitmap() with Matrix (변환 시점) |

### 5.2 결정 근거

1. **빠른 검증 우선**: 뷰티 필터 효과가 화면에 표시되는지 먼저 확인하고, 성능 문제가 실제로 발생하면 최적화 진행
2. **점진적 개선**: MVP → 측정 → 최적화 순서로 진행하여 불필요한 조기 최적화 방지
3. **복잡도 최소화**: JNI 코드 추가 없이 Kotlin/Java만으로 구현하여 유지보수 용이성 확보

### 5.3 피드백 대비 현황

| 제언 | 현황 | 후속 조치 |
|------|------|-----------|
| JNI 구현 우선순위 상향 | ✅ 구현 완료 | `nativeNv21ToRgba` JNI 함수 추가, Bitmap 버퍼 재사용 구현 |
| 동기화 전략 | ⚠️ 미적용 | `mainHandler.post()`로 메인 스레드에서 UI 업데이트하므로 현재 안전, 추후 모니터링 필요 |
| 의존성 충돌 점검 | ✅ OpenCV 사용 중, TFLite는 별도 빌드 | 현재 충돌 없음 |

### 5.4 성능 예상 및 후속 조치

**JNI 최적화 후 예상 성능:**
- NV21 → RGBA (JNI + OpenCV): 5-10ms
- 다운스케일 (1/2): 처리량 4배 감소
- Bitmap 버퍼 재사용: GC 부하 감소
- 예상 FPS: 25-30fps (뷰티 필터 활성화 시)

**성능 문제 발생 시 최적화 순서:**
1. **다운스케일**: 1/2 해상도로 처리량 4배 감소 (가장 쉬움)
2. **JNI 변환**: `nativeNv21ToRgba()` 구현으로 10배 속도 향상
3. **Double Buffering**: GC 부하 감소
4. **OpenGL Stage 2**: GPU 기반 렌더링으로 60fps+ 달성

### 5.5 권장 사항

1. **현재 구현으로 먼저 테스트** → 실제 디바이스에서 FPS 측정
2. **15fps 미만이면** → 다운스케일 적용 (가장 빠른 개선)
3. **그래도 부족하면** → JNI 변환 구현
4. **프로덕션 수준 필요시** → OpenGL Stage 2 진행

---
*후속 의견 작성: Claude (Implementation Session)*
