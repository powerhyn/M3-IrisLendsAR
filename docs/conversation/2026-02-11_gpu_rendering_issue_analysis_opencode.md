# GPU 렌더링 이슈 분석 보고서 (ROI & LUT)

- **작성일**: 2026-02-11
- **작성자**: OpenCode (Gemini 3 Pro)
- **대상**: `CameraGLRenderer.kt`, `gpu_beauty_backend.cpp`, `shader_sources.cpp`
- **관련 이슈**: 
    1. 뷰티 쉐이더가 화면 일부만 렌더링함 (ROI Scissor 문제)
    2. LUT 프리셋 변경 시 렌더링 중단 (검은 화면)

---

## 1. 뷰티 쉐이더 화면 일부 렌더링 현상

**증상**: 뷰티 필터가 적용될 때 화면 전체가 아닌 일부분(주로 좌하단이나 잘린 영역)만 렌더링되거나, 나머지 영역이 검게 나오거나 이전 프레임 잔상이 남음.

### 원인 분석

**1.1 `glScissor` 좌표계 변환 및 클램핑 오류 (Critical)**
`GPUBeautyBackend::applyTextureId` 내부의 ROI `glScissor` 설정 로직에서 좌표 변환과 클램핑 처리의 잠재적 오류가 확인됨.

*   **Y축 뒤집기 (Flip-Y)**:
    ```cpp
    // 현재 코드
    int sy = static_cast<int>(std::floor(static_cast<float>(height) - (roi_ptr->face_rect.y + roi_ptr->face_rect.height)));
    ```
    OpenGL `glScissor`는 좌하단(Bottom-Left)이 원점이고, `roi_ptr->face_rect`는 좌상단(Top-Left) 기준임.
    만약 `face_rect.y + height`가 텍스처 높이보다 클 경우(화면 아래로 벗어남), `sy`는 음수가 됨.

*   **잘못된 클램핑 (Clamping)**:
    ```cpp
    // 현재 코드
    sy = std::max(0, std::min(sy, height - 1));
    sh = std::max(1, std::min(sh, height - sy));
    ```
    `sy`가 음수였다가 `0`으로 클램핑되면, `sh`는 `height` 전체를 덮으려 시도할 수 있음. 하지만 `face_rect`가 실제로 화면을 벗어난 정도를 반영하지 못해 엉뚱한 영역을 자르게 됨.
    올바른 방식은 **Intersection(교집합)**을 구하는 것임.

**1.2 Passthrough Blit 실패 가능성**
ROI 외부 영역을 원본으로 채우기 위해 `executeCombinedColorPass`를 호출하여 프리-필(Pre-fill)을 수행하고 있음.
```cpp
// Pre-fill logic
executeCombinedColorPass(input_tex_id, ping->fbo_id, width, height, 1.0f, ...);
```
만약 이 패스가 실패하거나, `glScissor`가 활성화된 상태에서 이 패스가 호출된다면(순서 오류), 외부 영역은 그려지지 않아 검은색이나 쓰레기 값이 남게 됨. 현재 코드 순서는 올바르나(Scissor 활성화 전), FBO 바인딩 상태나 뷰포트 설정이 정확한지 재확인이 필요함.

---

## 2. LUT 프리셋 변경 시 렌더링 중단

**증상**: LUT 프리셋을 하나 적용한 후, 다른 프리셋으로 변경하면 화면이 멈추거나 검게 변함.

### 원인 분석

**2.1 텍스처 삭제와 사용 간의 동기화 문제 (Race Condition)**
`CameraGLRenderer.kt`에서 LUT 텍스처를 교체하는 로직:
```kotlin
if (lut3dTextureId != 0) {
    GLES31.glDeleteTextures(1, intArrayOf(lut3dTextureId), 0)
}
lut3dTextureId = newTextureId
```
*   **문제**: `glDeleteTextures`는 즉시 실행되지만, GPU 파이프라인에는 이전 프레임의 렌더링 명령(해당 텍스처를 사용하는)이 남아있을 수 있음.
*   **결과**: 사용 중인 텍스처가 삭제되면서 GL 에러가 발생하거나 컨텍스트가 오염됨.

**2.2 유효하지 않은 텍스처 ID 전달**
C++ 측 `executeCombinedColorPass`에서:
```cpp
if (lut_texture != 0 && lut_intensity > 0.01f) {
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, lut_texture);
}
```
`CameraGLRenderer`가 `lut3dTextureId`를 0으로 설정했다가 새 ID로 바꾸는 짧은 순간에 렌더링이 발생하면, 0번 텍스처(Default Texture)가 바인딩되거나, 삭제된 ID가 전달될 수 있음.
특히 `glBindTexture`에 삭제된 ID를 넣으면 `GL_INVALID_OPERATION`이 발생하여 이후 렌더링이 모두 중단될 수 있음.

---

## OpenCodeComment

> 작성: OpenCode (2026-02-11)
> 분석 기반: `feature/P3-beauty-enhancement` 코드베이스

### 제언 및 해결 방향

**1. ROI Scissor 수정**
단순 클램핑(`max/min`) 대신, **사각형 교차(Rectangle Intersection)** 로직을 사용하여 유효한 Scissor 영역을 계산해야 합니다.
```cpp
// Intersection Logic
int intersect_x = std::max(0, rect_x);
int intersect_y = std::max(0, rect_y);
int intersect_r = std::min(width, rect_x + rect_w);
int intersect_b = std::min(height, rect_y + rect_h);

int scissor_x = intersect_x;
int scissor_y = height - intersect_b; // GL Bottom-Left 변환
int scissor_w = intersect_r - intersect_x;
int scissor_h = intersect_b - intersect_y;

if (scissor_w <= 0 || scissor_h <= 0) { /* Skip Scissor or Disable ROI */ }
```

**2. LUT 텍스처 생명주기 관리**
*   **Safe Release**: 텍스처를 즉시 `glDeleteTextures`하지 말고, **"사용 종료 대기 큐(Garbage Queue)"**에 넣은 뒤, 몇 프레임(예: 3프레임) 후에 삭제하거나 `glFenceSync`를 통해 GPU 작업 완료를 확인한 후 삭제해야 합니다.
*   **Atomic Update**: `lut3dTextureId` 변수를 `Volatile`이나 `Atomic`으로 관리하거나, 렌더링 스레드(GLThread) 내부 큐(`queueEvent`)를 통해서만 변경하도록 강제해야 합니다. 현재 `uploadPendingLut3dTexture`가 GL 스레드에서 돌지만, `setLut3dTexture` 호출 시점과의 경합을 주의해야 합니다.

이 분석을 바탕으로 `fix` 커밋을 준비하는 것을 권장합니다.

---

## CodexComment

> 작성: Codex (2026-02-12)
> 검토 범위: `GpuRenderActivity.kt`, `CameraGLRenderer.kt`, `gpu_beauty_backend.cpp`, `iris_jni.cpp`

### 종합 코멘트

- OpenCode 리포트의 큰 방향은 타당함.
- 다만 LUT 이슈는 "동기화" 이전에 **삭제 주체 중복(ownership 중복)** 이 더 직접적인 원인으로 보임.

### 이슈 1: 뷰티가 화면 일부만 적용되는 현상

1. `roiOnly=true`로 동작하는 경로가 기본값임.
   - `BeautyFilterConfigV2.DEFAULT_ROI_ONLY = true`
   - `BeautyPresetFactory`의 NATURAL/STUDIO/GLAMOUR 모두 `.roiOnly(true)`
   - 이 경우 `GPUBeautyBackend::applyTextureId()`에서 `glScissor`가 켜지고 얼굴 ROI 내부만 필터 패스가 실행됨.

2. 현재 구현은 ROI 외부를 원본으로 pre-fill 하므로 "화면 일부만 필터 적용" 자체는 의도 동작일 수 있음.
   - 하지만 scissor 계산은 OpenCode가 지적한 것처럼 **교집합 기반으로 바꾸는 것이 안전**함.
   - 특히 ROI가 프레임 경계를 넘는 케이스에서 단순 clamp보다 교집합 계산이 오동작 가능성이 낮음.

3. 좌표 계약 리스크:
   - Java `IrisResult.faceRectX/Y/Width/Height` 주석은 "픽셀"로 되어 있으나,
   - 실제 SDK detect 경로 값은 정규화(0~1)로 전달됨.
   - 이 불일치가 다른 호출자에 의해 픽셀 값 주입으로 이어지면 scissor 오작동 가능.

### 이슈 2: LUT 프리셋 변경 후 렌더링 중단

1. 가장 유력한 직접 원인: **LUT 텍스처를 Activity와 Renderer가 모두 삭제**함.
   - Activity: `GpuRenderActivity.applyLutPreset()`에서 `currentLutTextureId` 즉시 `glDeleteTextures`
   - Renderer: `uploadPendingLut3dTexture()`에서 기존 `lut3dTextureId`를 다시 `glDeleteTextures`

2. 이중 삭제는 특히 **ID 재사용 시 치명적**임.
   - 이전 텍스처 ID를 삭제한 직후 새 LUT 생성 시 동일 ID가 재할당될 수 있음.
   - 이후 Renderer가 "기존 ID 삭제"를 수행하면서 새 LUT를 삭제해버릴 수 있음.
   - 결과적으로 LUT 바인딩 실패/무효 텍스처 사용으로 렌더링 중단(또는 LUT 미적용) 발생 가능.

3. 권장 방향:
   - LUT 3D 텍스처 삭제 책임을 Renderer 한 곳으로 단일화.
   - Activity에서는 `glDeleteTextures`를 제거하고 `setLut3dTexture(newId)`만 전달.
   - OFF 시에도 `setLut3dTexture(0)` 경로를 명시해 소유권/상태 전이를 일관화.

---

## ClaudeCodeComment

> 작성: Claude Code (Opus 4.6, 2026-02-12)
> 검토 범위: 전체 렌더링 파이프라인 (`GpuRenderActivity.kt`, `CameraGLRenderer.kt`, `gpu_beauty_backend.cpp`, `BeautyPresetFactory.kt`)

### OpenCode 리포트에 대한 평가

OpenCode의 분석 방향은 대체로 정확하나, 몇 가지 보완이 필요함.

**이슈 1에 대해:**
- OpenCode가 지적한 `glScissor` 클램핑 문제는 유효함. 교집합(Intersection) 로직이 단순 clamp보다 안전한 것은 맞음.
- 다만, **더 직접적인 원인이 추가로 존재**: `renderToScreen()`의 aspect ratio 스케일 계산이 주석과 불일치. "Cover 모드"라고 주석되어 있으나 실제 구현은 **Contain 모드**(블랙바 생성). 카메라 해상도(e.g., 720x1280)와 디스플레이(e.g., 1080x2400) 비율 차이로 최대 **40%까지 블랙바**가 발생할 수 있음.
- Codex가 지적한 `roiOnly=true` 기본값도 복합 원인. 데모에서는 전체 화면 뷰티가 더 적합.

**이슈 2에 대해:**
- OpenCode의 "동기화 문제(Race Condition)" 분석은 이론적으로는 맞지만, 실제 코드에서는 `queueEvent`가 GL 스레드에서 실행되므로 race condition은 아님.
- Codex가 지적한 **"삭제 주체 중복(ownership 중복)"이 정확한 근본 원인**. `applyLutPreset()`에서 `glDeleteTextures`한 후, `uploadPendingLut3dTexture()`에서 동일 ID를 다시 삭제 시도. GL이 ID를 재할당하면 새 텍스처가 삭제됨.
- OpenCode의 "Garbage Queue" 제안은 과도한 엔지니어링. 소유권 단일화만으로 충분.

### 적용된 수정사항 (4건)

| Fix | 파일 | 내용 |
|-----|------|------|
| Fix-A | `GpuRenderActivity.kt` | `applyLutPreset()`에서 `glDeleteTextures` 제거, Renderer로 ownership 단일화. OFF 시 `setLut3dTexture(0)` 경로 추가 |
| Fix-B | `CameraGLRenderer.kt` | `renderToScreen()` Contain→Cover 모드로 수정. 블랙바 제거, 화면 꽉 채우기 |
| Fix-C | `gpu_beauty_backend.cpp` | `glScissor` 계산을 교집합(Intersection) 기반으로 변경. 프레임 경계를 넘는 ROI에 안전 |
| Fix-D | `BeautyPresetFactory.kt` | 데모 프리셋 3종 `.roiOnly(true)`→`.roiOnly(false)`. 전체 화면 뷰티 적용 |

### Codex 코멘트에 대한 동의

- "삭제 주체 중복이 더 직접적인 원인" → **동의**. Fix-A에서 Activity의 `glDeleteTextures` 제거로 해결.
- "교집합 기반으로 바꾸는 것이 안전" → **동의 + 적용**. Fix-C에서 구현.
- "좌표 계약 리스크" → 현재 코드에서는 `detection->face_rect`가 정규화(0~1)로 전달되고 `applyTextureId()` 내부에서 픽셀 변환하므로 정합. 다만 향후 직접 픽셀 값 주입 경로가 추가되면 위험할 수 있음.
