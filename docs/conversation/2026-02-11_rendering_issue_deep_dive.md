# GPU 렌더링 이슈 심층 분석 및 해결 가이드

- **작성일**: 2026-02-11
- **작성자**: OpenCode (Gemini 3 Pro)
- **관련 이슈**: 
    1. 뷰티 필터 적용 시 화면 중앙 일부만 렌더링됨 (배경 블랙)
    2. 뷰티 필터 변경 시 화면 멈춤 (Freeze)
    3. LUT 필터 적용 시 정상화되거나, 교체 시 블랙 스크린 발생

---

## 1. 현상 분석 (Symptom Flow)

사용자가 제보한 플로우를 기술적으로 해석하면 다음과 같습니다.

1.  **초기 진입**: 화면 중앙 일부만 렌더링 (ROI 영역만 그려짐, 배경 Pre-fill 실패)
2.  **뷰티 필터 조절**: 렌더링 멈춤 (GPU 파이프라인 에러/크래시)
3.  **LUT 필터 적용**: 전체 화면 렌더링 복구 (배경 Pre-fill 성공)
4.  **LUT 교체**: 블랙 스크린 (텍스처 삭제 타이밍 이슈)

## 2. 심층 원인 분석 (Root Cause)

### 2.1 셰이더 샘플러 충돌 (The "Hidden" Sampler Conflict) - Critical
**현상 1(일부 렌더링)과 현상 2(멈춤)의 주범입니다.**

*   **메커니즘**: `CombinedColorPass` 셰이더는 2개의 샘플러를 사용합니다.
    *   `sampler2D uTexture` (Unit 0, 원본 영상)
    *   `sampler3D uLutTexture` (Unit 1, LUT 데이터)
*   **버그 코드 (`gpu_beauty_backend.cpp`)**:
    ```cpp
    // LUT가 있을 때만 유니폼을 1로 설정함
    if (lut_texture != 0 && lut_intensity > 0.01f) {
        glUniform1i(combined_color_uniforms_.uCombinedLutTexture, 1);
    }
    // else: 유니폼 설정 안 함 -> 기본값 0 (Unit 0) 유지!
    ```
*   **문제 발생**:
    *   LUT가 없는 경우(초기 진입, 뷰티 단독 조절, Pre-fill 단계), `uCombinedLutTexture`가 `Unit 0`을 가리킵니다.
    *   `Unit 0`에는 이미 `uTexture`(2D)가 바인딩되어 있습니다.
    *   **OpenGL 규약 위반**: 하나의 텍스처 유닛(Unit 0)에 `sampler2D`와 `sampler3D`가 동시에 접근하면 **Undefined Behavior (렌더링 실패)**가 발생합니다.
*   **결과**:
    *   ROI 외부를 채우는 **Pre-fill 그리기 명령이 무시됨** → 배경이 검게 나옴.
    *   뷰티 필터 조절 시 전체 화면 렌더링이 실패하여 화면이 멈춤.
    *   LUT를 켜면 `glUniform1i(..., 1)`이 호출되어 충돌이 해소되므로 정상 렌더링됨.

### 2.2 Scissor 클램핑 로직 오류
**현상 1(일부 렌더링)의 시각적 형태를 결정짓는 부차적 원인입니다.**

*   **문제**: `glScissor`의 `y`, `height` 계산 시 단순 `max/min` 클램핑을 사용.
    ```cpp
    int sy = ...; // 화면 아래로 벗어나면 음수 발생
    sy = max(0, sy); // 0으로 강제 변환 -> 위치 왜곡
    ```
*   **결과**: ROI 영역이 정확한 얼굴 위치가 아니라, 화면 하단이나 엉뚱한 곳에 잘려서 표시될 수 있습니다.

### 2.3 텍스처 삭제 Race Condition
**현상 4(LUT 교체 시 블랙)의 원인입니다.**

*   **문제**: `CameraGLRenderer`가 `glDeleteTextures`를 호출하는 시점에, GPU 파이프라인은 아직 그 텍스처를 사용하는 렌더링 명령을 수행 중일 수 있습니다.
*   **결과**: 사용 중인 리소스가 강제 해제되어 GL 에러 발생 및 렌더링 컨텍스트 오염.

---

## 3. 해결 가이드 (Action Plan)

### 3.1 [P0] 샘플러 유니폼 강제 설정 (Fix Freeze & Partial Render)
`gpu_beauty_backend.cpp`의 `executeCombinedColorPass` 함수 수정:

```cpp
// 1. 항상 명시적으로 유니폼 설정
glUniform1i(combined_color_uniforms_.uCombinedLutTexture, 1); // 항상 Unit 1 사용

// 2. 텍스처 바인딩 로직 개선
glActiveTexture(GL_TEXTURE1);
if (lut_texture != 0 && lut_intensity > 0.01f) {
    glBindTexture(GL_TEXTURE_3D, lut_texture);
} else {
    // LUT가 없어도 빈 텍스처나 0번을 바인딩하여 안전하게 처리
    glBindTexture(GL_TEXTURE_3D, 0); 
}
// 3. 셰이더 내에서 uLutIntensity가 0이면 texture() 호출을 skip하도록 보장 (이미 되어 있음)
```

### 3.2 [P1] Scissor 교차(Intersection) 로직 적용
단순 클램핑 대신 사각형 교차 로직 사용:

```cpp
int intersect_x = max(0, rect_x);
int intersect_y = max(0, rect_y);
int intersect_right = min(width, rect_x + rect_w);
int intersect_bottom = min(height, rect_y + rect_h);

int final_x = intersect_x;
int final_y = height - intersect_bottom; // GL Bottom-Left
int final_w = intersect_right - intersect_x;
int final_h = intersect_bottom - intersect_y;
```

### 3.3 [P2] Safe Texture Release
`CameraGLRenderer.kt`에서 삭제 로직 변경:
*   삭제할 텍스처 ID를 리스트(`garbageTextures`)에 담아두고, 3프레임 뒤에 삭제하거나 `glFenceSync` 활용.
*   가장 쉬운 방법: `lut3dTextureId` 교체 시 즉시 삭제하지 않고, **다음번 `uploadPending` 호출 시(즉, 새 텍스처가 준비되었을 때) 이전 것을 삭제**하는 방식으로 변경 (이미 일부 적용되어 있으나 타이밍 점검 필요).

---

이 문서를 기반으로 즉시 수정 작업(`fix` 커밋)을 진행할 것을 권장합니다.

---

## CodexComment (2026-02-12)

Gemini 리포트의 방향은 전반적으로 타당합니다. 다만 현재 코드 상태 기준으로 아래 4가지는 정밀 보정이 필요합니다.

### 1) 2.1 Sampler 충돌 가설은 핵심 원인으로 유효

- `executeCombinedColorPass()`에서 LUT 활성 시에만 `uLutTexture=1`을 설정합니다.
- LUT 비활성 경로에서 `uLutTexture`가 기본값(0)에 머무르면, `uTexture(2D)`와 동일 texture unit을 공유하게 되어 undefined behavior 가능성이 있습니다.
- 이 이슈는 "초기 부분 렌더링 + 뷰티 변경 시 freeze + LUT 켜면 정상화" 플로우와 정합성이 높습니다.

### 2) 2.2 Scissor 오류는 "현재 이슈"보다 "기존 이슈(반영 완료)"로 분류 필요

- 현재 브랜치의 `gpu_beauty_backend.cpp`는 이미 교집합(intersection) 기반 계산과 GL Y축 변환(`sy = height - intersect_b`)을 사용합니다.
- 따라서 문서의 2.2는 "과거 원인 후보"로 표현하고, 현재는 재발 여부 검증 항목으로 두는 것이 정확합니다.

### 3) 2.3 LUT 삭제 문제는 "race 단일 원인"보다 "삭제 소유권/ID 재사용 리스크"로 명시 권장

- Activity 측 즉시 삭제는 제거되어 ownership이 개선되었습니다.
- 다만 LUT 텍스처 lifecycle은 Renderer 단일 소유로 엄격히 유지하고, 교체 시점에 ID 재사용/중복 삭제가 절대 일어나지 않도록 정책을 문서화해야 합니다.

### 4) 본 리포트에 추가되어야 할 핵심 연결고리: `roiOnly` 상태 전이

- 기본 `BeautyFilterConfigV2`는 `roiOnly=true`입니다.
- 프리셋(Natural/Studio/Glamour)은 `roiOnly=false`로 생성됩니다.
- 즉, 사용자 플로우에서 "초기 진입(ROI 기반 부분 처리)"과 "프리셋 선택 후(full-frame 기대)"의 동작 차이는 단순 UI 상태 변화가 아니라, 렌더링 정책 전환입니다.
- 이 상태 전이를 문서에 포함해야 재현/수정 경로가 명확해집니다.

## 보강 액션 플랜 (권장)

### P0. CombinedColor sampler 바인딩을 항상 deterministic 하게 고정

- `uLutTexture`는 매 프레임 항상 unit 1로 설정.
- LUT OFF여도 unit 1에 안전한 fallback(권장: neutral 3D LUT 1x1x1)을 바인딩.
- 셰이더는 기존처럼 `uLutIntensity <= 0.01`에서 LUT sample을 건너뜀.

### P1. ROI 정책 명시화

- 제품 정책이 전체화면 뷰티가 기본이면 `initSDK()`에서 `beautyConfig.roiOnly = false`를 명시.
- ROI 기반 최적화가 목표면, UI/문서/디버그 오버레이에서 ROI 모드임을 명확히 노출.

### P1. Detection slot stale 데이터 방지

- `detectResult != OK` 또는 `detected=false` 프레임에서도 slot을 갱신해 stale ROI가 장시간 유지되지 않도록 처리.
- reader 측에서도 `valid + detected + timestamp` 조건을 확인하는 방어 로직 권장.

### P2. 관측성(Observability) 강화

- `executeCombinedColorPass` 진입 시 `lut_texture`, `lut_intensity`, sampler uniform 값, `glGetError`를 프레임 단위 샘플링 로그로 남겨서 freeze 직전 상태를 추적 가능하게 구성.
