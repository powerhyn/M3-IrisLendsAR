# ISS-005 블렌드 모드 색 희석 수정 — 실행 계획

## Context

ISS-005 브레인스토밍(Sec 21 합의 + E/F절)에서 도출된 합의 사항을 구현한다.
상세 작업 문서: `docs/workPaper/P4-W1-04_blend_color_fading_fix.md`

### 사전 작업
- `ISS-005_color_fading_fix.md` → `P4-W1-04_blend_color_fading_fix.md` 리네이밍
- 문서 내 제목/Phase 표기 갱신

## 3-Phase 실행

### Phase 1: EXP-A — Alpha Fix (1파일, 1줄)
- `LensManager.kt` line 168 뒤: `options.inPremultiplied = false`
- 검증: Before/After 스크린샷 + 경계 영역 비교

### Phase 2: EXP-B — Color Replace Mode 7 (10파일)
- Core→API→JNI→Java→Kotlin→GLSL→UI 전 레이어 수정
- GLSL: `detail = lum / max(0.01, uAvgIrisLum)`, `clamp(0.2, 2.5)`
- 검증: Mode 7 vs Mode 0/4 Color Lift 비교

### Phase 3: EXP-C — Mode 5 색공간 정합 (1파일)
- `blendLuminanceTintLinear()`: `uAvgIrisLum` → `uAvgIrisLum * uAvgIrisLum` (linear 변환)
- 검증: Mode 4 vs Mode 5 비교 → 존치 판정

## 수정 파일 총괄

| Phase | 파일 | 변경 |
|:-----:|------|------|
| EXP-A | `demo-app/.../lens/LensManager.kt` | +1줄 |
| EXP-B | `cpp/include/iris_sdk/types.h` | enum 추가 |
| EXP-B | `cpp/include/iris_sdk/sdk_api.h` | C enum 추가 |
| EXP-B | `cpp/src/sdk_api.cpp` | switch case 추가 |
| EXP-B | `cpp/src/lens_renderer.cpp` | fallthrough 추가 |
| EXP-B | `cpp/tests/test_sdk_api.cpp` | assertion 추가 |
| EXP-B | `android/iris-sdk/.../iris_jni.cpp` | 범위 검증 갱신 |
| EXP-B | `android/iris-sdk/.../LensConfig.java` | 상수 + 3메서드 |
| EXP-B | `android/iris-sdk/.../BlendMode.kt` | enum entry |
| EXP-B | `demo-app/.../CameraGLRenderer.kt` | GLSL + dispatch |
| EXP-B | `demo-app/.../GpuRenderActivity.kt` | Spinner 배열 |
| EXP-C | `demo-app/.../CameraGLRenderer.kt` | ~3줄 수정 |

## 검증
1. `cd android && ./gradlew :demo-app:assembleDebug`
2. `cd cpp/cmake-build-debug && cmake --build . && ctest`
3. 실기기 스크린샷 비교 (Color Lift + 경계 영역)
