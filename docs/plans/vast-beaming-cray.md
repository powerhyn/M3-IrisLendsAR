# 화사한 필터 (Vivid Post-Processing) 구현 계획

## Context

현재 뷰티 필터(스무딩, 밝기, 화이트닝, FreqSep 등)는 모두 **ROI(얼굴 영역) 기반**으로 적용됩니다. 화면 전체적으로 화사한 느낌을 주는 포스트프로세싱 필터가 필요합니다. 기존 피부 보정 필터(skinQuality/FreqSep)는 건드리지 않습니다.

## 설계 결정

### 1. enabled 분리 전략 (GPU 경로 2곳만 수정)

vivid는 GPU 전용 기능이므로, CPU C API(`iris_sdk_apply_beauty_v2_c`)는 수정하지 않습니다.
수정 대상은 **GPU 경로 2곳**:

- `sdk_api_v2.cpp:317` — `iris_sdk_apply_beauty_texture_v2()`의 enabled 가드
- `gpu_beauty_backend.cpp:1483` — `applyTextureId()`의 enabled 가드

```cpp
// 변경 패턴 (2곳 동일):
bool needsVivid = config.vividIntensity > 0.01f;
if (!config.enabled && !needsVivid) {
    *output_texture = input_texture;
    return IRIS_SDK_OK;
}
```

### 2. enabled=false에서 beauty 패스 명시적 차단

`applyTextureId()`의 beauty 필터 체인(스무딩~소프트포커스)은 `config.enabled`가 아니라 각 수치값의 `> 0.01f` 가드로 실행 여부를 결정합니다 (`gpu_beauty_backend.cpp:1579-1827`). 따라서 `enabled=false`여도 UI에 smoothing, brightness 등의 값이 남아있으면 beauty 패스가 실행됩니다.

**해결**: `buildEffectiveConfig()`에서 `config.enabled == false`일 때 beauty 관련 수치를 무효화:

```cpp
static BeautyFilterConfigV2 buildEffectiveConfig(const BeautyFilterConfigV2& config) {
    BeautyFilterConfigV2 effective = config;

    if (!config.enabled) {
        // beauty 비활성 → vivid-only 경로. 모든 beauty 수치를 중립값으로 설정.
        effective.smoothing = 0.0f;
        effective.softFocus = 0.0f;
        effective.whitening = 0.0f;
        effective.colorBalance = 0.0f;
        effective.brightness = 1.0f;
        effective.skinQuality = 0.0f;
        return effective;
    }

    // 기존 master intensity 적용 로직 (변경 없음)
    const float master = std::clamp(config.intensity, 0.0f, 1.0f);
    effective.smoothing *= master;
    // ... 이하 기존 코드 ...
}
```

이렇게 하면 beauty 패스들이 각자의 `> 0.01f` 가드에 의해 자연스럽게 스킵됩니다. `active_filter_count`도 beauty 패스 0개 + vivid 1개로 정확히 계산됩니다.

### 3. ROI/scissor 억제 (applyTextureId 내부)

ROI/scissor 설정은 `sdk_api_v2.cpp`가 아니라 `applyTextureId()` 내부(line 1507)에서 수행됩니다. `sdk_api_v2.cpp`의 ROI 변수는 실제로 backend에 전달되지 않으므로 (detection만 전달), ROI 억제는 **backend 내부**에서 해야 합니다.

**해결**: `applyTextureId()`의 ROI 생성 조건(line 1507)에 `config.enabled` 추가:

```cpp
// 기존: if (detection && detection->detected && config.roiOnly) {
// 변경:
if (config.enabled && detection && detection->detected && config.roiOnly) {
```

`enabled=false`면 ROI가 생성되지 않으므로 `roi_ptr`은 `nullptr`, scissor는 비활성 상태를 유지합니다.

### 4. ROI empty 상황 처리

ROI 교집합이 비어있으면 `gpu_beauty_backend.cpp:1688`에서 early return하여 vivid에 도달 불가.

**해결**: early return 직전에 vivid 체크 삽입. 기존 cleanup 패턴(ping/pong 저장, fence) 유지.

```cpp
} else {
    // ROI 교집합 비어있음 → beauty 필터 스킵
    if (needsVivid) {
        executeVividPass(current_input, ping->fbo_id,
                         width, height, config.vividIntensity,
                         config.vividSaturation, config.vividBrightness,
                         config.vividWarmth);
        *output_texture = ping->texture_id;
    } else {
        *output_texture = current_input;
    }
    if (ping) { previous_output_ping_ = ping; }
    if (pong) { previous_output_pong_ = pong; }
    previous_fence_ = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return IRIS_SDK_OK;
}
```

### 5. Ping-pong 버퍼 안전성

softFocus는 마지막 패스라는 전제로 `current_output` 스왑을 생략함 (`gpu_beauty_backend.cpp:1821-1827`).
vivid가 뒤에 추가되면 피드백 루프 발생.

**해결**: vivid가 활성일 때 `active_filter_count`에 포함 + softFocus 이후 스왑 수행.

```cpp
// active_filter_count에 추가
bool needsVivid = config.vividIntensity > 0.01f;
if (needsVivid) active_filter_count++;

// softFocus 패스 이후 (기존 line 1827)
current_input = current_output->texture_id;
if (pong && needsVivid) current_output = (current_output == ping) ? pong : ping;
```

### 6. GPU 전용, CPU 미지원

vivid 포스트프로세싱은 **GPU 전용**입니다. CPU 경로(`iris_sdk_apply_beauty_v2_c`)는 수정하지 않으며, vivid 파라미터는 무시됩니다. 구조체에는 존재하지만 CPU 백엔드에서 처리하지 않습니다.

### 7. "화사한" 효과 구성

- **Vibrance**: 저채도 영역 우선 부스트 (과포화 방지)
- **밝기 리프트**: 미드톤 위주 소프트 리프트 (`x*(1-x)` 커브, 하이라이트 클리핑 방지)
- **웜톤 시프트**: R/G 약간 증가 + B 약간 감소

---

## 구현 단계

### Step 1: C++ 내부 구조체 확장 — `BeautyFilterConfigV2`

**파일**: `cpp/include/iris_sdk/beauty_filter.h`

`BeautyFilterConfigV2` 구조체 `downscaleFactor` 필드 뒤에 추가:

```c
//===== 화면 전체 포스트프로세싱 (Vivid, GPU 전용) =====
/** @brief 화사한 필터 마스터 강도 (0.0~1.0, 기본값 0.0, 0이면 비활성) */
float vividIntensity;
/** @brief 채도 부스트 - Vibrance 방식 (0.0~1.0, 기본값 0.0) */
float vividSaturation;
/** @brief 밝기 리프트 (0.0~0.5, 기본값 0.0) */
float vividBrightness;
/** @brief 웜톤 시프트 (0.0~1.0, 기본값 0.0) */
float vividWarmth;
```

같은 파일 하단 `BeautyFilterConfigV2Helper`의 `defaults()`, `isValid()`, `clamp()` 업데이트.

### Step 2: C API 공개 구조체 확장 — `IrisBeautyConfigV2`

**파일**: `cpp/include/iris_sdk/sdk_api.h`

`IrisBeautyConfigV2` 구조체 `feather_radius` 필드 뒤에 추가:

```c
/* 화면 전체 포스트프로세싱 (Vivid, GPU 전용) */
float vivid_intensity;     /**< 화사한 필터 강도 (0.0~1.0, 0=비활성) */
float vivid_saturation;    /**< 채도 부스트 (0.0~1.0) */
float vivid_brightness;    /**< 밝기 리프트 (0.0~0.5) */
float vivid_warmth;        /**< 웜톤 시프트 (0.0~1.0) */
```

### Step 3: C API 변환 함수 + enabled 가드 업데이트

**파일**: `cpp/src/sdk_api_v2.cpp`

- `toCppConfigV2()` (line 52): vivid 4개 필드 매핑 추가
- `fromCppConfigV2()` (line 85): 역방향 매핑 추가
- `iris_sdk_default_beauty_config_v2_c()` (line 132): vivid 기본값 0.0f 설정
- `iris_sdk_apply_beauty_texture_v2()` (line 317): enabled 가드에 vivid 체크 추가

### Step 4: Vivid 셰이더 작성

**파일**: `cpp/src/gpu/shader_sources.cpp`

단일 패스 프래그먼트 셰이더 `VIVID_POSTPROCESS_FRAGMENT` 추가:
- 텍스처 샘플 1회, ALU 연산 위주 → ~0.3ms (MID tier)
- Vibrance: `smoothstep` 기반 저채도 선택적 부스트
- Brightness: `result += brightness * result * (1.0 - result)` 미드톤 커브
- Warmth: R/G 미세 증가, B 미세 감소
- 마스터 intensity로 원본과 `mix`

extern 선언 추가 (`gpu_beauty_backend.cpp:89`):
```cpp
extern const char* VIVID_POSTPROCESS_FRAGMENT;
```

### Step 5: GPUBeautyBackend에 vivid 패스 통합

**파일**: `cpp/include/iris_sdk/gpu/gpu_beauty_backend.h`
- `vivid_program_` 셰이더 프로그램 ID 추가 (line 407 근처)
- `VividUniforms` 구조체 추가 + 인스턴스 (UniformLocations 섹션)
- `executeVividPass()` private 메서드 선언 (필터 패스 섹션)

**파일**: `cpp/src/gpu/gpu_beauty_backend.cpp`

**(a) 초기화**
- `initializeShaders()`: vivid 셰이더 컴파일 추가
- `cacheUniformLocations()`: vivid uniform 캐싱 추가
- `release()` (line 516 근처): `vivid_program_ = 0` 리셋 추가 (프로그램 삭제는 `shader_manager_->releaseAll()`이 담당)

**(b) buildEffectiveConfig() 수정** (line 56)
`config.enabled == false`일 때 beauty 수치를 중립값으로 설정 (설계 섹션 2 참조). vivid 파라미터는 master intensity와 독립적으로 원본 값 유지.

**LUT 정책**: LUT는 `applyTextureId()`의 별도 인자(`lut_texture_id`, `lut_intensity`)로 전달되므로 `buildEffectiveConfig()`로 막을 수 없습니다. `enabled=false`일 때 LUT도 비활성화하기 위해, `needsLut` 계산(line 1586)에 `config.enabled` 조건을 추가합니다:
```cpp
bool needsLut = config.enabled && (lut_texture_id != 0 && lut_intensity > 0.01f);
```

**(c) executeVividPass() 구현** — 기존 `executeBrightnessPass()`와 동일 패턴

**(d) applyTextureId() 수정** — 5개 지점:

**지점 1: enabled 가드** (line 1483)
```cpp
bool needsVivid = config.vividIntensity > 0.01f;
if (!config.enabled && !needsVivid) {
    *output_texture = input_texture;
    return IRIS_SDK_OK;
}
```

**지점 2: ROI 생성 조건** (line 1507)
```cpp
// 기존: if (detection && detection->detected && config.roiOnly) {
// 변경:
if (config.enabled && detection && detection->detected && config.roiOnly) {
```
`enabled=false`면 ROI/scissor가 비활성화되어 vivid-only 경로에서 전체 프레임 처리 보장.

**지점 3: active_filter_count** (line 1579)
```cpp
if (needsVivid) active_filter_count++;
```
`buildEffectiveConfig()`가 beauty 수치를 중립화하므로, beauty 필터 count는 자동으로 0. vivid만 1개로 텍스처 정상 할당.

**지점 4: ROI empty early return** (line 1688-1699)
vivid 활성이면 vivid-only 처리 후 기존 cleanup 패턴으로 반환 (설계 섹션 4 참조).

**지점 5: 메인 필터 체인 끝** (scissor 해제 후, line 1832)

softFocus 이후 ping-pong 스왑 수정 + vivid 패스 삽입:
```cpp
// softFocus 패스 (기존 line 1821-1827)
if (effective_config.softFocus > 0.01f) {
    // ... 기존 코드 ...
    current_input = current_output->texture_id;
    if (pong && needsVivid) current_output = (current_output == ping) ? pong : ping;
}

// Scissor 해제
if (scissor_active) {
    glDisable(GL_SCISSOR_TEST);
}

// Vivid 포스트프로세싱 (전체 프레임, ROI 무관)
if (needsVivid) {
    if (profiling) profiler_->begin("Vivid");
    executeVividPass(current_input, current_output->fbo_id,
                     width, height,
                     config.vividIntensity,
                     config.vividSaturation,
                     config.vividBrightness,
                     config.vividWarmth);
    if (profiling) profiler_->end("Vivid");
    current_input = current_output->texture_id;
}

*output_texture = current_input;
```

### Step 6: Android JNI 바인딩 업데이트

**파일**: `android/iris-sdk/src/main/java/com/irislenssdk/BeautyFilterConfigV2.java`
- `vividIntensity`, `vividSaturation`, `vividBrightness`, `vividWarmth` 필드 추가
- Builder 메서드, defaults, isValid, clamp, toString, copy constructor 업데이트

**파일**: `android/iris-sdk/src/main/cpp/jni_utils.h`
- `JniCache`에 vivid 관련 field ID 4개 선언 추가 (line 369 근처, 기존 `beautyConfigV2_*` 필드 선언 영역)

**파일**: `android/iris-sdk/src/main/cpp/iris_jni.cpp`
- `JniCache` 초기화에서 vivid field ID 조회 추가
- `copyBeautyConfigV2FromJava()`: Java → `IrisBeautyConfigV2` 매핑
- `copyBeautyConfigV2ToJava()`: 역방향 매핑

### Step 7: 데모 앱 UI 연동

데모 앱에서 vivid 파라미터 슬라이더를 추가하여 실기기 테스트 가능하도록 함.

---

## 수정 파일 목록

| 파일 | 변경 내용 |
|------|----------|
| `cpp/include/iris_sdk/beauty_filter.h` | `BeautyFilterConfigV2` + Helper 업데이트 |
| `cpp/include/iris_sdk/sdk_api.h` | `IrisBeautyConfigV2` 공개 구조체 확장 |
| `cpp/src/sdk_api_v2.cpp` | 변환 함수 + enabled 가드 + 기본값 |
| `cpp/src/gpu/shader_sources.cpp` | `VIVID_POSTPROCESS_FRAGMENT` 셰이더 |
| `cpp/include/iris_sdk/gpu/gpu_beauty_backend.h` | vivid 멤버/메서드 선언 |
| `cpp/src/gpu/gpu_beauty_backend.cpp` | buildEffectiveConfig + vivid 패스 + applyTextureId 5개 지점 |
| `android/.../BeautyFilterConfigV2.java` | Java 필드 + Builder |
| `android/.../jni_utils.h` | JniCache vivid field ID 선언 |
| `android/.../iris_jni.cpp` | JNI field ID 초기화 + 매핑 함수 |

## 검증 방법

1. **빌드 확인**: `cd cpp/cmake-build-debug && cmake --build . --parallel`
2. **단위 테스트**: vivid 파라미터 기본값/클램핑/유효성 + C API round-trip 테스트
3. **실기기 테스트**:
   - vivid ON + beauty OFF → 화면 전체 화사 (피부 보정 없음)
   - vivid ON + beauty ON → 피부 보정 + 화면 전체 화사
   - vivid ON + softFocus ON → 두 효과 공존 (피드백 루프 없음)
   - ROI empty (얼굴 화면 밖) + vivid ON → vivid만 정상 적용
   - vivid OFF → 기존 동작과 100% 동일
   - enabled=false + smoothing/brightness 잔여값 + vivid ON → beauty 미실행, vivid만 실행
4. **성능**: GPU 프로파일러로 vivid 패스 비용 확인 (목표: < 0.5ms)
5. **JNI round-trip**: Java에서 설정한 vivid 값이 GPU 셰이더까지 도달하는지 로그 확인

## 비고

- **ABI 변경**: `BeautyFilterConfigV2`와 `IrisBeautyConfigV2` 모두 필드 추가됨. 프리릴리즈 SDK이므로 재컴파일 전제.
- **CPU 미지원**: vivid는 GPU 전용. CPU 경로(`iris_sdk_apply_beauty_v2_c`)는 수정하지 않으며, vivid 파라미터는 구조체에 존재하지만 무시됨.
- **release()**: `vivid_program_`은 `shader_manager_->releaseAll()`이 프로그램 삭제를 담당하므로, `release()`에서는 `vivid_program_ = 0` 리셋만 추가.
- **범위 밖**: `BeautyProcessor::processTexture()` → `GPUBeautyBackend::applyTexture()` 경로는 이번 작업 범위에 포함하지 않습니다. 이 경로는 C++ 공개 API로, 현재 Android 앱에서 사용하지 않습니다 (Android는 C API 경로 `iris_sdk_apply_beauty_texture_v2()` → `applyTextureId()`를 사용). `BeautyProcessor`와 `applyTexture()`의 enabled 가드(`beauty_processor.cpp:178`, `gpu_beauty_backend.cpp:589`)및 내부 파이프라인(`gpu_beauty_backend.cpp:624-655`)에는 vivid 패스를 추가하지 않습니다. 향후 이 경로에서도 vivid가 필요해지면 별도 작업으로 진행합니다.
