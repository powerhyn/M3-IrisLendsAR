# Phase C: Color-Space Skin Likelihood 실험

## Context

뷰티 마스크 정밀도 개선 5번 항목. Phase A 적용 후 잔여 haze를 픽셀 색상 기반으로 추가 감쇠하는 실험.

**설계 원칙**: backend-local toggle (`skin_color_filter_` 멤버), public config struct 변경 없음.
C API 함수(`iris_sdk_set_skin_color_filter`) + Java SDK 메서드(`setSkinColorFilter`)는 추가.
`freqsep_debug_mode_` 패턴을 그대로 따름.

## Codex 피드백 반영사항

| # | 이슈 | 해결 |
|---|------|------|
| 1 | uniform 세팅 scope에 config 없음 | `skin_color_filter_` 멤버 사용 (L1500 `freqsep_debug_mode_` 옆) |
| 2 | debug 7 접근 불가 + rawMask 필요 | 데모 순환 범위 7→8, 셰이더에서 `rawMask` 보존 후 heatmap에 사용 |
| 3 | dark×chroma 곱이 어두운 피부 과감쇠 | `max(dark, chroma)` 로 변경 — 둘 중 하나만 통과해도 피부로 판정 |
| 4 | FreqSep-only 제한 미문서화 | bilateral fallback 시 no-op임을 문서화 |
| 5 | Java/demo wiring 복잡도 | public API 건드리지 않음, `setSkinColorFilter()` 단독 함수만 추가 |

## 수정 파일 (9개)

### 1. 셰이더 (`cpp/src/gpu/shader_sources.cpp`)

**uniform 추가** (L483 `uDebugMode` 뒤):
```glsl
uniform int uSkinColorFilter;   // 0=off, 1=on (backend-local)
```

**skin likelihood 로직** (L496 `float mask = ...` 직후):
```glsl
// --- Phase C: Color-space skin likelihood ---
// rawMask 보존 (debug mode 7에서 사용)
float rawMask = mask;

// Linear → Gamma sRGB (피부 범위는 gamma 공간 기준)
vec3 gammaRGB = pow(max(orig, vec3(0.0)), vec3(1.0 / 2.2));
float lum = dot(gammaRGB, vec3(0.299, 0.587, 0.114));
float maxDev = max(max(abs(gammaRGB.r - lum), abs(gammaRGB.g - lum)),
                   abs(gammaRGB.b - lum));

// Signal 1: 어두운 픽셀 제외 (안경테, 콧구멍, 머리카락)
float darkLikelihood = smoothstep(0.08, 0.22, lum);
// Signal 2: 무채색 픽셀 제외 (금속, 깊은 그림자)
float chromaLikelihood = smoothstep(0.02, 0.08, maxDev);
// max: 어두운 무채색 구조물(안경테, 콧구멍, 머리카락)만 억제
// 밝거나 채도 있으면 통과 — 범용 피부 분류기가 아닌 exclusion 필터
// floor 0.05: 완전 제로 방지
float skinLikelihood = max(max(darkLikelihood, chromaLikelihood), 0.05);

if (uSkinColorFilter != 0) {
    mask *= skinLikelihood;
}
```

**핵심 변경**: `darkLikelihood * chromaLikelihood` → `max(darkLikelihood, chromaLikelihood)`
- 어두운 피부(lum 낮음)도 채도가 있으면 chromaLikelihood로 통과
- 밝은 금속(lum 높음)도 무채색이면... 사실 밝은 금속은 mask 밖이므로 무관
- 실제 타깃(안경테, 콧구멍, 머리카락)은 어둡고 AND 무채색 → 둘 다 낮음 → max도 낮음

**역할 범위**: 범용 피부 분류기가 아닌 **"어두운 무채색 누수 구조물 억제"** 용도. `max(dark, chroma)`이므로 밝은 무채색(금속 하이라이트 등)은 통과할 수 있으나, 그런 구조물은 보통 마스크 외부에 있으므로 실질적 문제 없음.

**기존 protection과의 관계**: composite에 `chromaProtection`(L533), `shadowProtection`(L544) 이미 있음. skinLikelihood는 **mask 단계**(적용 영역 축소), 기존 protection은 **compression 단계**(강도 감소). 역할이 다르지만 그림자 경계에서 양쪽 감쇠가 겹칠 수 있으므로 실기기 테스트에서 확인 필요.

**디버그 모드 7** (기존 모드 6 뒤):
```glsl
} else if (uDebugMode == 7) {
    // rawMask 사용: skinLikelihood 적용 전 마스크로 overlay
    // 빨강=제외(low likelihood), 초록=피부(high likelihood)
    vec3 heatmap = vec3(1.0 - skinLikelihood, skinLikelihood, 0.0);
    result = mix(orig, heatmap, rawMask * 0.8);
    result = pow(max(result, vec3(0.0)), vec3(1.0 / 2.2));
    fragColor = vec4(result, 1.0);
    return;
}
```

### 2. GPU 백엔드 헤더 (`cpp/include/iris_sdk/gpu/gpu_beauty_backend.h`)

`FreqSepCompositeUniforms` (L502 뒤):
```cpp
GLint uSkinColorFilter = -1;
```

멤버 + getter/setter (L538 `freqsep_debug_mode_` 옆):
```cpp
bool skin_color_filter_ = false;
public:
void setSkinColorFilter(bool enabled) { skin_color_filter_ = enabled; }
bool getSkinColorFilter() const { return skin_color_filter_; }
```

### 3. GPU 백엔드 구현 (`cpp/src/gpu/gpu_beauty_backend.cpp`)

uniform 캐시 (L424 뒤):
```cpp
freq_sep_composite_uniforms_.uSkinColorFilter =
    glGetUniformLocation(freq_sep_composite_program_, "uSkinColorFilter");
```

uniform 설정 (L1500 뒤):
```cpp
glUniform1i(freq_sep_composite_uniforms_.uSkinColorFilter,
            skin_color_filter_ ? 1 : 0);
```

### 4. C API (`cpp/include/iris_sdk/sdk_api.h` + `cpp/src/sdk_api_v2.cpp`)

L717 `iris_sdk_set_freqsep_debug_mode` 옆에 복사:
```cpp
// sdk_api.h
IRIS_SDK_EXPORT void iris_sdk_set_skin_color_filter(int enabled);

// sdk_api_v2.cpp
void iris_sdk_set_skin_color_filter(int enabled) {
#ifdef IRIS_SDK_HAS_GLES
    std::lock_guard<std::mutex> lock(g_gpu_mutex);
    if (g_gpu_beauty && g_gpu_beauty->isInitialized()) {
        g_gpu_beauty->setSkinColorFilter(enabled != 0);
    }
#else
    (void)enabled;
#endif
}
```

### 5. JNI + Java SDK (`iris_jni.cpp` + `IrisLensSDK.java`)

`nativeSetFreqSepDebugMode` 패턴 복사:
```java
// IrisLensSDK.java (L846 부근, setFreqSepDebugMode 옆)
public static void setSkinColorFilter(boolean enabled) {
    if (sLibraryLoaded) nativeSetSkinColorFilter(enabled ? 1 : 0);
}
private static native void nativeSetSkinColorFilter(int enabled);
```
```cpp
// iris_jni.cpp (L1710 부근)
JNIEXPORT void JNICALL
Java_com_irislenssdk_IrisLensSDK_nativeSetSkinColorFilter(
    JNIEnv*, jclass, jint enabled) {
    iris_sdk_set_skin_color_filter(static_cast<int>(enabled));
}
```

### 6. 데모앱 (`GpuRenderActivity.kt` + `activity_gpu_render.xml`)

**디버그 모드 순환 확장** (L602-604):
```kotlin
val debugModeNames = arrayOf("OFF", "Magnitude", "MicroBand", "EdgeProt",
    "EffectStr", "Compression×3", "Mask", "SkinColor")  // 7 추가
freqSepDebugMode = (freqSepDebugMode + 1) % 8  // 7→8
```

**토글 버튼** (layout + activity):
```kotlin
btnSkinColorFilter.setOnClickListener {
    skinColorFilterEnabled = !skinColorFilterEnabled
    cameraGLView.queueEvent {
        IrisLensSDK.setSkinColorFilter(skinColorFilterEnabled)
    }
    btnSkinColorFilter.text = if (skinColorFilterEnabled) "피부색필터: ON" else "피부색필터: OFF"
}
```

## 변경하지 않는 것 (config struct 미변경)

- ❌ `BeautyFilterConfigV2` (C++ config struct)
- ❌ `IrisBeautyConfigV2` (C API config struct)
- ❌ `BeautyFilterConfigV2.java` (Java config struct)
- ❌ `jni_utils.h` (field ID 캐시)
- ❌ `beauty_roi_manager.cpp` (CPU 마스크 파이프라인)

## 제한사항

- **FreqSep 전용**: bilateral fallback (LOW tier) 시 no-op. FreqSep composite 셰이더에만 존재.
- **ROI 필수**: face mesh 없거나 ROI 없으면 마스크 자체가 없으므로 무관.
- **데모 preset 주의**: `BeautyPresetFactory.baseBuilder`가 `roiOnly(false)`를 설정하므로, preset 적용 시 FreqSep 자체가 비활성화됨 → skinColorFilter도 no-op. 실험 시 preset 대신 수동 슬라이더 사용 (초기 `roiOnly=true` 유지).

## 검증

1. **디버그 모드 7 히트맵**: `rawMask * 0.8`로 overlay → 안경테=빨강, 피부=초록, 어두운 피부톤도 초록 확인
2. **ON/OFF 비교**: 매끈하게 0.7~0.8 + 안경 착용 시 haze 차이
3. **무안경 테스트**: OFF와 시각적으로 동일한지 (skinLikelihood ≈ 1.0)
4. **어두운 피부 테스트**: 그림자 경계에서 이중 감쇠(skinLikelihood × effectStrength) 패치 발생 여부
5. **성능 실측**: composite 패스 GPU time profiler로 delta 측정 (추정 아닌 실측)
