# P5-W3 디버그: SDK GPULensRenderer 빈 텍스처 이슈

## 상태: ✅ 해결 완료 (2026-04-16)

## 문제 요약

SDK C++ `GPULensRenderer`가 초기화/셰이더 컴파일까지 성공하지만, 실제 렌즈 렌더링 시 **빈 텍스처(렌즈 안 보임)**를 출력한다. 기존 Kotlin 셰이더(`renderLensOverlay`)로 폴백하면 정상 동작.

## 현재 동작 흐름

```
CameraGLRenderer.kt (onDrawFrame)
  → applyGpuLensRenderer(inputTexture)
    → IrisLensSDK.renderLensTexture(inputTex, w, h, detectionPtr, lensConfig)  [Java]
      → nativeRenderLensTexture(...)  [JNI: iris_jni.cpp]
        → iris_sdk_render_lens_texture(...)  [C API: sdk_api_v2.cpp]
          → GPULensRenderer::renderToTexture(...)  [C++: gpu_lens_renderer.cpp]
            → FBO 바인딩 → 셰이더 실행 → 출력 텍스처 반환
```

## 성공하는 것 (로그 확인됨)

```
nativeInitGpuLens called
GPULensRenderer: Using current thread's EGL context
ShaderManager: Program cached: 'lens_overlay' = 48
GPULensRenderer: Lens overlay shader compiled: program=48
GPULensRenderer: Fullscreen quad VAO=3, VBO=3
GPULensRenderer: Lens uniforms cached (3 locations)  ← ⚠️ 3개만?
GPULensRenderer: GPULensRenderer initialized successfully
GPU Lens Renderer init: 0  (IRIS_SDK_OK)
```

- `nativeRenderLensTexture: texture=3, 960x720` 매 프레임 호출됨
- 폴백 로그(`SDK lens render failed`) 없음 → SDK가 유효한 텍스처 ID를 반환하고 있음
- 하지만 해당 텍스처의 내용이 비어있음 (렌즈가 안 보임)

## 의심 원인 (우선순위순)

### 1. Uniform Location 캐싱 문제 (가장 유력)
로그에 `Lens uniforms cached (3 locations)`만 나옴. 셰이더에 30+개 uniform이 있는데 3개만 찾았다면 대부분의 uniform이 -1로 설정되어 셰이더가 의미없는 값으로 동작 중.

**확인 방법**: `gpu_lens_renderer.cpp`의 `cacheLensUniforms()` 함수에서 각 uniform 이름이 셰이더 소스의 uniform 이름과 정확히 일치하는지 확인.

**파일**: `cpp/src/gpu/gpu_lens_renderer.cpp` — `cacheLensUniforms()` 메서드
**셰이더**: `cpp/src/gpu/shader_sources.cpp` — `LENS_OVERLAY_FRAGMENT` 문자열

### 2. 렌즈 텍스처 미로드
`uploadPendingLensTexture()`에서 SDK로 RGBA 전달:
```kotlin
val rgbaBytes = ByteArray(bitmap.width * bitmap.height * 4)
val buffer = java.nio.ByteBuffer.wrap(rgbaBytes)
bitmap.copyPixelsToBuffer(buffer)
IrisLensSDK.loadLensTexture(rgbaBytes, bitmap.width, bitmap.height)
```
- Bitmap의 config가 ARGB_8888인지 확인 필요 (ARGB vs RGBA 순서 문제 가능)
- `loadLensTexture` JNI가 `ScopedByteArray`로 데이터를 받아 C API로 전달
- C++ `GPULensRenderer::loadLensTexture()`에서 `glTexImage2D` 호출 확인 필요

**파일**: `cpp/src/gpu/gpu_lens_renderer.cpp` — `loadLensTexture()` 메서드

### 3. IrisResult 변환 문제
`sdk_api_v2.cpp`에서 `reinterpret_cast`로 C→C++ 변환:
```cpp
const iris_sdk::IrisResult* cpp_result = 
    reinterpret_cast<const iris_sdk::IrisResult*>(detection);
```
C `IrisResult`와 C++ `iris_sdk::IrisResult`는 동일 POD 레이아웃이지만, Eye Refiner 필드 추가로 구조체 크기가 달라졌을 수 있음. `face_mesh_valid`가 false이면 타원 피팅/눈꺼풀 클리핑이 모두 비활성화됨.

**확인**: detection pointer의 `detected`, `left_detected`, `confidence`, `face_mesh_valid` 값 로깅

### 4. FBO 바인딩/텍스처 출력 문제
`TexturePool::acquireRenderTarget()`이 반환하는 텍스처가 유효한 FBO에 attach되어 있는지 확인.

**파일**: `cpp/src/gpu/gpu_lens_renderer.cpp` — `renderToTexture()` 메서드 (FBO 바인딩 부분)

### 5. 좌표계 변환 누락
데모 Kotlin에서는 Y-flip + 미러(전면 카메라) 보정을 하지만, SDK C++ 셰이더에서는 이 보정이 없을 수 있음. 홍채 좌표가 화면 밖에 있으면 렌즈가 그려지지 않음.

## 핵심 파일 목록

### C++ 코어
| 파일 | 역할 | 주요 라인 |
|------|------|-----------|
| `cpp/include/iris_sdk/gpu/gpu_lens_renderer.h` | 헤더 | LensUniforms 구조체, 클래스 전체 |
| `cpp/src/gpu/gpu_lens_renderer.cpp` | 구현 | `initialize()`, `loadLensTexture()`, `renderToTexture()`, `cacheLensUniforms()`, `fitEyeEllipse()` |
| `cpp/src/gpu/shader_sources.cpp` | 셰이더 | `LENS_OVERLAY_VERTEX`, `LENS_OVERLAY_FRAGMENT` (파일 끝) |

### C API
| 파일 | 역할 |
|------|------|
| `cpp/include/iris_sdk/sdk_api.h` | `iris_sdk_render_lens_texture` 등 선언 |
| `cpp/src/sdk_api_v2.cpp` | C API 구현 (파일 끝에 GPU 렌즈 함수들) |

### JNI / Java
| 파일 | 역할 |
|------|------|
| `android/iris-sdk/src/main/cpp/iris_jni.cpp` | `nativeRenderLensTexture` 등 (파일 끝) |
| `android/iris-sdk/src/main/java/com/irislenssdk/IrisLensSDK.java` | Java API + native 선언 |

### 데모 앱
| 파일 | 역할 |
|------|------|
| `android/demo-app/.../CameraGLRenderer.kt` | `applyGpuLensRenderer()` (SDK 호출), `renderLensOverlay()` (기존 Kotlin 셰이더) |

## 기존 Kotlin 셰이더 (동작하는 참조 코드)

`CameraGLRenderer.kt`의 `renderLensOverlay()` (862~1154줄):
- FBO 생성/바인딩
- Camera texture → unit 0, Lens texture → unit 1
- 홍채 좌표 변환 (Y-flip, mirror, 정규화)
- Uniform 설정 (47개)
- 풀스크린 쿼드 드로우

## 비교 참조: GPUBeautyBackend (동작하는 SDK GPU 경로)

`applyGpuBeautyFilter()` (1214~1261줄) → `IrisLensSDK.applyBeautyFilterTextureV2()`:
- `getDetectionSlotPtr()`로 검출 결과 포인터 취득
- JNI → C API → `GPUBeautyBackend::applyTextureId()` 호출
- TexturePool에서 출력 텍스처 획득 → FBO 바인딩 → 셰이더 실행
- 출력 텍스처 ID 반환

**이 경로는 정상 동작**하므로, GPULensRenderer도 동일 패턴을 따라야 함.

## 디버깅 순서 제안

1. **cacheLensUniforms()의 uniform 이름 vs 셰이더 소스 대조** — 이름 불일치 시 glGetUniformLocation이 -1 반환
2. **renderToTexture()에 디버그 로깅 추가** — detection.detected, left_detected, confidence, face_mesh_valid, iris 좌표, 렌즈 텍스처 ID
3. **loadLensTexture()에서 실제 GL 텍스처 ID와 크기 로깅** — 0이면 업로드 실패
4. **FBO completeness 체크** — `glCheckFramebufferStatus(GL_FRAMEBUFFER)` 결과 로깅
5. **최소 테스트**: 셰이더에서 `fragColor = vec4(1.0, 0.0, 0.0, 1.0);`로 하드코딩하여 빨간 화면이 나오는지 확인 → FBO/쿼드 문제인지 셰이더 로직 문제인지 구분

## 코드 리뷰에서 발견한 구체적 문제점

### A. `iris_result.left_radius` 단위 불일치 (유력)
`renderToTexture()` 659줄:
```cpp
glUniform1f(lens_uniforms_.uLeftIrisRadius, iris_result.left_radius);
```
`left_radius`는 **픽셀 단위**인데, 셰이더에서는 **정규화 좌표(0~1)**를 기대한다.
Kotlin 데모에서는 `result.leftRadius / detHf`로 정규화해서 전달한다 (CameraGLRenderer.kt 913줄).
→ 픽셀값(예: 15.0)이 그대로 들어가면 렌즈가 화면 전체보다 커서 보이지 않을 수 있음.

**수정**: `iris_result.left_radius / height` 또는 `/ frame_height`로 정규화 필요.

### B. `uDetH` 계산 오류
`renderToTexture()` 736~738줄:
```cpp
float det_h = static_cast<float>(iris_result.frame_height) /
              static_cast<float>(std::max(iris_result.frame_height, 1));
```
이건 항상 `1.0f`가 된다 (자기 자신으로 나누기). 
Kotlin에서는 `uDetH = detHf` (검출 프레임 높이 픽셀값)을 그대로 전달한다.
→ Contact Shadow depth 계산(`shadowDepthPx / uDetH`)이 잘못됨.

**수정**: `glUniform1f(lens_uniforms_.uDetH, static_cast<float>(iris_result.frame_height));`

### C. 좌표계 변환 누락
Kotlin 데모 (CameraGLRenderer.kt 917~933줄)에서는 홍채 좌표에 Y-flip과 mirror 변환을 적용한다:
```kotlin
// Y-flip: glY = 1.0f - meshY
// Mirror (전면 카메라): glX = 1.0f - meshX, 좌우 눈 스왑
```
C++ `renderToTexture()`에서는 `iris_result.left_iris[0].x/y`를 **변환 없이 그대로** 전달한다.
→ 카메라 프레임 텍스처의 좌표계와 홍채 좌표의 좌표계가 불일치하면 렌즈가 엉뚱한 위치에 그려짐.

**확인 필요**: 입력 텍스처가 이미 OES→RGBA 변환 후인지, 그때 좌표계가 어떻게 되는지.

### D. `uMaxDetail` 타입 불일치
`renderToTexture()` 704줄:
```cpp
glUniform1i(lens_uniforms_.uMaxDetail, 1);
```
셰이더에서 `uMaxDetail`은 `int`로 선언되어 있는데, ColorReplace 블렌드 모드에서 `float`처럼 사용한다:
```glsl
detail = clamp(detail, 0.2, float(uMaxDetail));
```
`int 1`이면 `float(1)` = 1.0으로 동작하긴 하지만, 의도는 0.9~1.2 범위의 float 값.

### E. 렌즈 텍스처 로드 확인
`loadLensTexture()`가 실제로 호출되었는지 로그가 없다.
`uploadPendingLensTexture()`에서 SDK 로드가 실행되려면:
1. `IrisLensSDK.isGpuLensInitialized()` == true
2. 렌즈 선택 후 bitmap이 `pendingLensBitmap`에 설정됨
3. GL 스레드에서 `uploadPendingLensTexture()` 호출

앱 시작 시 기본 렌즈가 이미 로드된 상태면, `onSurfaceCreated`에서 SDK 초기화 전에 bitmap이 세팅될 수 있어 타이밍 이슈 가능.

## 수정 완료 항목 (2026-04-13, 초기 패치)

| 버그 | 수정 내용 |
|------|----------|
| **A** | `left_radius / frame_height`로 정규화 (픽셀→0~1) |
| **B** | `uDetH`에 `det_hf` (픽셀 높이) 직접 전달 |
| **C** | 홍채 Y-flip, 눈꺼풀 Y-flip, 타원 cy Y-flip + rotation 부호 반전 |
| **D** | `glUniform1i` → `glUniform1f` (셰이더 float uniform) |
| **E** | eyelidFeather `feather_px / det_hf`로 동적 정규화 |

## 추가 수정 (2026-04-16, 최종 해결)

### 렌즈 안 보이는 문제 (주요 원인 3가지)

| 원인 | 수정 |
|------|------|
| **Y-flip 후 eyelid top/bottom 부등호 역전** → `eyelidMask=0` (렌즈 완전 투명) | 셰이더 전체를 Kotlin 1:1 포팅. 셰이더 내부에서 `min(top,bot), max(top,bot)` 처리 |
| **전면 카메라 mirror 미적용** → iris 좌표가 텍스처와 반대쪽에 찍힘 | `IrisLensConfig.is_mirror` 필드 추가. JNI→C API→C++ 전달 경로 구축. `renderToTexture`에서 X-flip + 좌우 swap |
| **`uFrameAspect`를 texture(960/720) 기준으로 계산** → detection(720/960)과 불일치 | `iris_result.frame_width / frame_height` 사용 (detection 기준) |

### 블렌딩 차이 (셰이더 전체 교체)

| 원인 | 수정 |
|------|------|
| C++ 셰이더가 Kotlin과 **완전히 다른 구현** (blend 시그니처, sclera/shadow 식, linearize 함수 등) | **LENS_OVERLAY_FRAGMENT를 Kotlin 셰이더와 1:1 포팅** (`shader_sources.cpp`) |
| `contact_shadow_` 기본 true (Kotlin은 false) | 기본값을 Kotlin과 일치 (false) |
| `use_ellipse_mask_` 기본 true (Kotlin은 false) | 기본값을 Kotlin과 일치 (false) |
| `uMaxDetail` int 1 (Kotlin은 float 1.2) | `glUniform1f(uMaxDetail, 1.2f)` |
| `uAvgIrisLum` 0.3 (Kotlin은 0.35) | 0.35로 수정 |

### Shimmer 방지

| 원인 | 수정 |
|------|------|
| 렌즈 텍스처 mipmap 없음 → 매 프레임 다른 배율로 축소 시 자글거림 | `glGenerateMipmap` + `GL_LINEAR_MIPMAP_LINEAR` (에러 시 GL_LINEAR 폴백) |

### is_mirror 전달 경로

```
CameraGLRenderer.kt (lensConfig.isMirror = isMirror)
  → LensConfig.java (public boolean isMirror)
    → JNI copyConfigFromJava (dest.is_mirror = GetBooleanField)
      → sdk_api_v2.cpp (cpp_config.is_mirror = config->is_mirror)
        → GPULensRenderer::renderToTexture (X-flip + swap)
```

## 검증 결과 (2026-04-16)

```
Lens uniforms cached (31/31 locations)
Lens texture mipmap enabled (anti-shimmer)
renderToTexture: det=720x960 aspect=0.750 | mirror=1 tex=960x720 | lens_tex=5
```

- 31/31 uniform 캐싱 (이전 29/31 → 셰이더 교체 후 모두 활성)
- Mipmap 정상 생성
- Mirror 정상 전달
- 렌즈 렌더링 Kotlin과 비슷한 결과 확인
