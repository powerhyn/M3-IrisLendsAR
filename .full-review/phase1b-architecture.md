# Phase 1b: Architectural Review -- Vivid Post-Processing Filter

**Review scope**: develop 브랜치 대비 변경 사항 (10 files, +307/-12 lines)
**Feature**: 화면 전체 화사한 포스트프로세싱 필터 (GPU-only, single-pass shader)

---

## 1. Component Boundaries

### 1.1 Vivid 파라미터의 BeautyFilterConfigV2 내장

**Severity**: Medium
**Architectural Impact**: High (장기적 struct 비대화)

`BeautyFilterConfigV2`에 4개의 vivid 필드(`vividIntensity`, `vividSaturation`, `vividBrightness`, `vividWarmth`)가 직접 추가되었다. 현재 이 구조체는 이미 21개 필드를 보유하고 있으며, vivid 추가로 25개가 되었다.

**문제점**:
- Vivid는 설계 문서상 beauty 필터와 "독립적으로 동작"하는 기능이다. 그런데 물리적으로 동일 구조체에 소속되어 있어, 개념적 독립성과 물리적 결합도 사이에 불일치가 존재한다.
- 향후 Vignette, Film Grain, Color Grading 등 전체 프레임 포스트프로세싱이 추가될 때마다 이 구조체가 계속 비대해진다.
- 모든 바인딩 레이어(JNI, Obj-C++, dart:ffi, WASM)가 이 단일 POD 구조체에 의존하므로, 필드 추가마다 전 플랫폼 바인딩을 수정해야 한다.

**현재 상태에서의 장점**:
- 단일 구조체이므로 API 호출이 한 번으로 끝난다 (앱 측에서 config 하나만 전달).
- POD FFI 호환성이 자연스럽게 유지된다.
- 기존 파이프라인에 최소 침투로 통합할 수 있다.

**권장 사항**:
현 단계에서는 수용 가능하지만, 포스트프로세싱 효과가 2개 이상 추가될 시점에서 `PostProcessConfig` 서브 구조체를 분리하고 `BeautyFilterConfigV2`에 compose하는 방향을 검토해야 한다. POD 제약은 nested struct로도 충족 가능하다.

```c
typedef struct PostProcessConfig {
    float vividIntensity;
    float vividSaturation;
    float vividBrightness;
    float vividWarmth;
    // 향후: float vignetteStrength; float filmGrainAmount; ...
} PostProcessConfig;

typedef struct BeautyFilterConfigV2 {
    // ... 기존 beauty 필드 ...
    PostProcessConfig postProcess;
} BeautyFilterConfigV2;
```

---

## 2. Dependency Direction & Coupling

### 2.1 C API 레이어의 enabled/vivid 분기 로직 중복

**Severity**: Low
**Architectural Impact**: Low

`sdk_api_v2.cpp`의 `iris_sdk_apply_beauty_texture_v2` 함수와 `GPUBeautyBackend::applyTextureId` 양쪽 모두에서 `needsVivid` 판정 및 enabled 가드가 존재한다:

```cpp
// sdk_api_v2.cpp (line 330-333)
bool needsVivid = config->vivid_intensity > 0.01f;
if (!config->enabled && !needsVivid) { ... }

// gpu_beauty_backend.cpp (applyTextureId, line 1548-1551)
bool needsVivid = config.vividIntensity > 0.01f;
if (!config.enabled && !needsVivid) { ... }
```

이 이중 가드는 방어적 프로그래밍으로 볼 수 있으나, 0.01f 임계값이 매직 넘버로 두 곳에 하드코딩되어 있다. 한쪽만 수정하면 동작 불일치가 발생한다.

**권장 사항**:
`static constexpr float kVividMinThreshold = 0.01f` 상수를 헤더에 정의하고, 가능하면 `BeautyFilterConfigV2Helper`에 `hasActiveVivid()` 같은 헬퍼 메서드를 추가하여 판정 로직을 단일화할 것.

### 2.2 Vivid 셰이더 유니폼에 effective_config 대신 raw config 사용

**Severity**: Medium
**Architectural Impact**: Medium (일관성 문제)

`applyTextureId` 내에서 beauty 관련 패스는 모두 `effective_config`(master intensity 적용 후)를 사용하지만, vivid 패스만 raw `config`를 직접 사용한다:

```cpp
// Beauty passes use effective_config
executeCombinedColorPass(... effective_config.brightness, effective_config.colorBalance ...);

// Vivid pass uses raw config
executeVividPass(... config.vividIntensity, config.vividSaturation, config.vividBrightness, config.vividWarmth);
```

이는 의도적 설계일 수 있다 (vivid가 beauty master intensity에 영향받지 않아야 하므로). 그러나 `buildEffectiveConfig()`에서 vivid 값이 아무 변환 없이 pass-through되는 것과 합쳐 보면, vivid에도 자체 master intensity (`vividIntensity`)가 존재하기 때문에 논리적으로는 문제가 없다.

**권장 사항**:
현재 설계가 의도적이라면, `buildEffectiveConfig`에서 vivid 관련 필드를 명시적으로 "pass-through" 주석을 추가하여 의도를 문서화할 것. 향후 코드 리뷰어가 실수로 vivid 값에 master를 곱하는 것을 방지한다.

---

## 3. API Design & ABI Implications

### 3.1 POD 구조체 확장 전략의 ABI 안정성

**Severity**: High
**Architectural Impact**: High

`BeautyFilterConfigV2`와 `IrisBeautyConfigV2` 구조체 모두 끝에 vivid 필드를 추가하는 방식으로 확장되었다. 이는 단방향 호환성(새 SDK + 구 앱)에서 문제가 발생할 수 있다.

**구체적 시나리오**:
- 앱이 이전 버전 SDK 헤더로 빌드된 `sizeof(IrisBeautyConfigV2)`로 구조체를 할당한 후, 새 SDK 라이브러리가 vivid 필드를 읽으면 할당되지 않은 메모리를 접근한다.
- `memset(config, 0, sizeof(IrisBeautyConfigV2))`로 초기화하는 기존 코드는 문제없지만, stack 변수로 일부만 초기화하는 경우 vivid 필드가 garbage 값을 가질 수 있다.

**현재 완화 요소**:
- `iris_sdk_default_beauty_config_v2_c()`가 vivid 포함 전체를 0으로 초기화한다.
- vivid_intensity의 0.0f 기본값이 비활성화 상태이므로 garbage가 아닌 이상 안전하다.

**권장 사항**:
중장기적으로 구조체에 `uint32_t version` 필드를 첫 멤버로 추가하는 버전링 전략을 도입할 것. 현재 변경에서는 "끝에 추가" 패턴이 유효하지만, 향후 iOS/Flutter 바인딩 추가 시 ABI 안정성이 더 중요해진다.

### 3.2 C API에 vivid 전용 함수 부재

**Severity**: Low
**Architectural Impact**: Low

Vivid는 beauty 필터와 독립적으로 사용 가능한 기능이지만, 활성화하려면 반드시 `IrisBeautyConfigV2` 전체 구조체를 구성해야 한다. 단순히 화면에 화사한 효과만 원하는 앱도 beauty 관련 17개 필드를 모두 설정해야 한다.

**권장 사항**:
현 단계에서 별도 API는 과도하다. `iris_sdk_default_beauty_config_v2_c()` + vivid 필드만 설정하는 패턴이 이미 유효하므로, 문서화 수준에서 대응 가능하다.

---

## 4. Data Model Assessment

### 4.1 buildEffectiveConfig의 enabled=false 경로

**Severity**: Low (정확히 구현됨)
**Architectural Impact**: Medium (코드 흐름 이해도)

`buildEffectiveConfig()`이 `enabled=false`일 때 beauty 값을 중립값으로 설정하고 early return 한다. 이 함수가 vivid 값은 변환하지 않는다는 사실이 코드 구조상 함축적이다.

```cpp
if (!config.enabled) {
    effective.smoothing = 0.0f;
    effective.softFocus = 0.0f;
    effective.whitening = 0.0f;
    effective.colorBalance = 0.0f;
    effective.brightness = 1.0f;
    effective.skinQuality = 0.0f;
    return effective;  // vivid 필드는 원본 그대로
}
```

설계 의도는 명확하나, 명시적이지 않다.

**권장 사항**:
early return 직전에 `// vivid 필드는 enabled 플래그와 독립이므로 원본 유지` 주석을 추가할 것.

### 4.2 Java/JNI 필드 미러링의 일관성

**Severity**: Low
**Architectural Impact**: Low (양호하게 구현됨)

Java `BeautyFilterConfigV2`, JNI `jni_utils.h`, `iris_jni.cpp`의 vivid 필드 추가가 완전히 대칭적이다:
- Java 4필드, JNI cache 4 field ID, Get/Set 각 4개 -- 누락 없음
- 필드 이름 매핑: Java `vividIntensity` -> C `vivid_intensity` (카멜케이스/스네이크케이스 변환 일관)
- Builder 패턴에 vivid 4개 setter 추가 완료
- `isValid()`, `clamp()`, 기본값 상수, 복사 생성자, `toString()` 모두 vivid 반영

이 부분은 빈틈 없이 구현되었다.

---

## 5. Pipeline Integration Pattern

### 5.1 Ping-Pong 버퍼 스왑의 조건부 처리

**Severity**: High
**Architectural Impact**: High (렌더링 결함 가능성)

SoftFocus 패스의 ping-pong 스왑이 vivid 존재 여부에 따라 조건부로 변경되었다:

```cpp
// 기존 (develop):
// SoftFocus 후 ping-pong 스왑 없음 (SoftFocus가 마지막 패스였으므로)

// 변경 후:
if (pong && needsVivid) current_output = (current_output == ping) ? pong : ping;
```

이 변경의 의도는 올바르다: vivid가 뒤따라올 때 SoftFocus 출력과 다른 FBO에 vivid를 렌더링해야 하므로 스왑이 필요하다. 그러나 문제가 있는 경우가 있다.

**잠재적 문제 시나리오**: `enabled=false, needsVivid=true`일 때:
1. `buildEffectiveConfig`이 beauty 값을 모두 0으로 설정
2. `active_filter_count`는 vivid만 1이므로 ping 하나만 할당 (pong은 null)
3. SoftFocus 조건: `effective_config.softFocus > 0.01f` -- 0.0f이므로 진입 안 함
4. Vivid 패스: `current_input`은 `input_texture`, `current_output`은 `ping` -- OK

이 경우는 문제없다. 그러나 **CombinedColor 패스에서는 vivid 후를 위한 스왑이 없다**:

```cpp
// CombinedColor 후:
if (pong) current_output = (current_output == ping) ? pong : ping;
// ^-- 이 스왑은 vivid 존재와 무관하게 pong이 있으면 항상 실행
```

이 패턴은 CombinedColor + Vivid 두 패스 모두 활성인 경우:
1. CombinedColor -> ping에 출력, current_output = pong으로 스왑
2. SoftFocus 비활성이면 스킵
3. Vivid -> pong에 출력 -- OK

CombinedColor + SoftFocus + Vivid 세 패스:
1. CombinedColor -> ping 출력, current_output = pong
2. SoftFocus -> pong 출력, needsVivid이므로 current_output = ping
3. Vivid -> ping 출력 -- OK (CombinedColor 결과를 덮어쓰지만, SoftFocus 입력이 CombinedColor 결과였으므로 이미 사용된 버퍼)

분석 결과 로직은 올바르게 동작한다. 그러나 조건부 스왑 패턴(`if (pong && needsVivid)`)이 기존 무조건 스왑(`if (pong)`)과 혼재되면서 코드 가독성이 저하되었다.

**권장 사항**:
패스 체이닝 로직을 "마지막 패스인지 판별" 방식이 아닌, "이후에 추가 패스가 있으면 항상 스왑"하는 일관된 규칙으로 통일할 것. 현재는 각 패스가 "다음에 누가 오는지"를 알아야 하므로 결합도가 높다.

### 5.2 Scissor Disable 후 Vivid 실행 순서

**Severity**: Low (올바르게 구현됨)
**Architectural Impact**: N/A

Vivid가 scissor test 해제 후 실행되므로 전체 프레임에 정확히 적용된다. ROI scissor로 beauty가 제한된 후, 전체 프레임 vivid가 적용되는 순서가 올바르다.

```cpp
// ROI Scissor 해제
if (scissor_active) {
    glDisable(GL_SCISSOR_TEST);
}
// 4. Vivid 포스트프로세싱 (전체 프레임, ROI 무관)
if (needsVivid) { ... }
```

### 5.3 ROI 교집합이 빈 경우의 Vivid 처리

**Severity**: Low (올바르게 구현됨)
**Architectural Impact**: N/A

ROI scissor 교집합이 비어 beauty를 스킵해야 할 때도 vivid는 실행되도록 분기가 추가되었다:

```cpp
if (needsVivid) {
    executeVividPass(current_input, ping->fbo_id, ...);
    *output_texture = ping->texture_id;
} else {
    *output_texture = current_input;
}
```

이 early-exit 경로에서 vivid를 누락하지 않은 점이 잘 처리되었다.

---

## 6. Shader Management

### 6.1 Vivid 셰이더 실패 시 초기화 전체 실패

**Severity**: Medium
**Architectural Impact**: Medium

`initializeShaders()`에서 vivid 셰이더 컴파일 실패가 전체 GPU 백엔드 초기화 실패로 이어진다:

```cpp
if (!shader_manager_->createProgram(
        shaders::FULLSCREEN_QUAD_VERTEX,
        shaders::VIVID_POSTPROCESS_FRAGMENT,
        vivid_program_)) {
    LOGE("Failed to create vivid program");
    return false;  // 전체 초기화 실패
}
```

비교: FreqSep 셰이더 실패는 non-fatal로 처리된다:
```cpp
if (!initializeFreqSepShaders()) {
    LOGW("Failed to create Freq Sep shaders (non-fatal)");
}
```

Vivid는 선택적 기능이므로 FreqSep처럼 non-fatal 처리가 더 적절하다. 특정 저사양 GPU에서 셰이더 컴파일이 실패할 경우, vivid 하나 때문에 beauty 필터 전체가 사용 불가능해지는 것은 과도하다.

**권장 사항**:
vivid 셰이더 실패를 LOGW로 처리하고, `vivid_program_ = 0`을 유지한 채 초기화를 계속 진행하도록 변경할 것. `executeVividPass`에서 `vivid_program_ == 0`이면 early return하는 가드를 추가해야 한다.

### 6.2 셰이더 코드 품질

**Severity**: N/A (잘 구현됨)

`VIVID_POSTPROCESS_FRAGMENT` 셰이더는 단일 패스 설계 원칙을 잘 준수한다:
- 텍스처 샘플 1회 + ALU 위주 연산
- Vibrance의 smoothstep 기반 저채도 우선 부스트로 과포화 방지
- 밝기 리프트에 `result * (1 - result)` 미드톤 커브 사용으로 클리핑 방지
- 웜톤 시프트 계수 (R +0.04, G +0.02, B -0.03)가 자연스러운 범위
- `uIntensity` master mix로 원본과 블렌딩하는 최종 단계가 적절

`precision highp float` 사용이 MID-tier 디바이스에서 0.3ms 타겟을 충족하는지는 실기기 프로파일링으로 확인해야 한다. mediump으로 전환 가능한 경우 성능 이점이 있을 수 있다.

---

## 7. Architectural Consistency

### 7.1 기존 패턴과의 일관성 -- 양호

다음 항목들이 프로젝트의 기존 아키텍처 패턴을 충실히 따르고 있다:

| 패턴 | 준수 여부 | 설명 |
|------|-----------|------|
| POD 구조체 + extern "C" | O | C/C++ 구조체 모두 POD 유지 |
| C API -> C++ 변환 함수 | O | `toCppConfigV2` / `fromCppConfigV2` 업데이트 |
| Uniform location 캐싱 | O | `VividUniforms` struct + `cacheUniformLocations` |
| Ping-pong 버퍼 패턴 | O | 기존 패턴 확장 |
| Profiler 통합 | O | `profiler_->begin/end("Vivid")` |
| release()에서 자원 정리 | O | `vivid_program_ = 0` |
| JNI field ID 캐시 패턴 | O | `JniCache`에 추가 + 검증 |
| Builder 패턴 (Java) | O | vivid 4개 setter 추가 |
| Helper struct (C++) | O | `isValid`, `clamp`, `defaults` 모두 업데이트 |

### 7.2 CPU 경로 미지원에 대한 명시적 문서화 부재

**Severity**: Low
**Architectural Impact**: Low

Vivid는 GPU-only로 설계되었으나, `iris_sdk_apply_beauty_v2_c` (CPU 경로)에서 vivid 관련 처리가 전혀 없다. CPU 경로에서 vivid 값이 설정되어 있어도 조용히 무시된다.

이는 의도된 설계이지만, CPU 경로(`useGpu=false`)에서 vivid가 무시되는 것을 API 문서에 명시해야 한다.

---

## 8. Summary of Findings

| # | Finding | Severity | Impact | 상태 |
|---|---------|----------|--------|------|
| 1.1 | ConfigV2 구조체 비대화 경향 | Medium | High | 현 단계 수용, 향후 분리 검토 |
| 2.1 | needsVivid 판정 로직 중복 (매직 넘버) | Low | Low | 상수화 권장 |
| 2.2 | Vivid가 raw config 사용 (effective 아님) | Medium | Medium | 의도 문서화 필요 |
| 3.1 | POD 구조체 ABI 버전링 부재 | High | High | 중장기 version 필드 도입 권장 |
| 3.2 | Vivid 전용 C API 부재 | Low | Low | 문서화로 충분 |
| 4.1 | buildEffectiveConfig vivid pass-through 암묵적 | Low | Medium | 주석 추가 권장 |
| 4.2 | Java/JNI 필드 미러링 일관성 | -- | -- | 완벽하게 구현됨 |
| 5.1 | Ping-pong 조건부 스왑 가독성 | High | High | 스왑 규칙 통일 권장 |
| 5.2 | Scissor 해제 후 vivid 실행 순서 | -- | -- | 올바르게 구현됨 |
| 5.3 | ROI 교집합 빈 경우 vivid 처리 | -- | -- | 올바르게 구현됨 |
| 6.1 | Vivid 셰이더 실패 = 전체 초기화 실패 | Medium | Medium | non-fatal 처리 권장 |
| 6.2 | 셰이더 코드 품질 | -- | -- | 잘 구현됨 |
| 7.1 | 기존 패턴 일관성 | -- | -- | 전반적으로 우수 |
| 7.2 | CPU 경로 vivid 미지원 문서화 부재 | Low | Low | API doc 보강 필요 |

---

## 9. Overall Architectural Assessment

**등급**: Good (양호) -- 소규모 개선 필요

이 구현은 기존 SDK 아키텍처 패턴을 충실히 따르면서도, 새로운 독립 기능을 최소한의 침투로 통합한 점에서 높이 평가된다. 특히:

- **레이어 관통 일관성**: C++ core -> C API -> JNI 바인딩 -> Java 전 레이어에서 빈틈 없이 vivid 필드가 반영됨
- **GPU 파이프라인 통합**: 기존 ping-pong 버퍼, profiler, scissor 관리 패턴을 존중하면서 vivid 패스를 자연스럽게 삽입
- **독립 동작 설계**: `enabled=false`에서도 vivid만 독립 활성화 가능한 `buildEffectiveConfig` + ROI 억제 로직이 적절
- **셰이더 설계**: 단일 패스, ALU 위주, 과포화/클리핑 방지 등 성능과 품질 사이의 균형이 좋음

**즉시 대응 권장 사항** (이번 PR 내):
1. vivid 셰이더 컴파일 실패를 non-fatal로 변경 (Finding 6.1)
2. `buildEffectiveConfig`에 vivid pass-through 의도 주석 추가 (Finding 4.1)
3. `kVividMinThreshold` 상수 추출 (Finding 2.1)

**차기 이터레이션 대응 권장 사항**:
1. ping-pong 스왑 규칙 리팩토링 -- "마지막 패스 여부" 기반이 아닌 "후속 패스 존재 여부" 기반으로 통일 (Finding 5.1)
2. POD 구조체 버전링 전략 도입 (Finding 3.1)
3. PostProcessConfig 서브 구조체 분리 검토 (Finding 1.1)
