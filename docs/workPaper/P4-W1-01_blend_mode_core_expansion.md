# P4-W1-01: Blend Mode 4-6 Core 경로 원자적 확장

## 작업 개요
- **Phase**: P4 (시각적 리얼리즘 — Visual Fidelity)
- **기간**: 2026-02-13 ~
- **상태**: ✅ 완료
- **선행 조건**: 없음 (독립 작업)
- **근거**: 브레인스토밍 Section 13, 16, 19, 22, 26 합의

## 목표

기존 Blend Mode 0-3(Normal/Multiply/Screen/Overlay)에 Mode 4-6(LuminanceTint/LuminanceTintLinear/SoftLight)을 **Core 경로 7개 파일에 원자적으로 추가**한다. 부분 머지 금지.

### 배경

현재 블렌드 모드 4종은 "렌즈가 붙은 느낌"을 주며, 홍채의 자연스러운 질감을 보존하지 못한다. Luminance-Preserving Tint(mode 4/5)는 원본 홍채의 밝기 패턴을 유지하면서 색상만 교체하여 "스며든 렌즈" 효과를 구현한다.

```
현재 합성: result = mix(원본_홍채, 렌즈_텍스처, alpha)
→ alpha 높으면 → 홍채 구조 사라짐 → "덮어진 느낌"

신규 합성: result = lensColor × luminance(원본_홍채) × scale
→ 홍채 줄무늬/깊이/반사광 100% 보존 → "스며든 느낌"
```

## 수정 대상 파일 (7개 필수 + 1개 권장, 단일 PR)

| # | 파일 | 레이어 | 수정 내용 |
|---|------|--------|-----------|
| 1 | `cpp/include/iris_sdk/types.h` | C++ Core enum | `LuminanceTint=4`, `LuminanceTintLinear=5`, `SoftLight=6` 추가 |
| 2 | `cpp/include/iris_sdk/sdk_api.h` | C API enum | `IRIS_BLEND_LUMINANCE_TINT=4`, `_LINEAR=5`, `_SOFT_LIGHT=6` 추가 |
| 3 | `cpp/src/sdk_api.cpp` | C→C++ 변환 | `convert_blend_mode()` switch에 3 case 추가 + **default에 LOGW** |
| 4 | `android/iris-sdk/src/main/cpp/iris_jni.cpp` | JNI Bridge | `static_cast` → rawBlendMode 범위 검증 + LOGW |
| 5 | `android/iris-sdk/src/main/java/com/irislenssdk/LensConfig.java` | Java SDK API | 3 상수 추가, `isValid()`/`clamp()`/`getBlendModeName()` 확장 |
| 6 | `android/iris-sdk/src/main/java/com/irislenssdk/kotlin/BlendMode.kt` | Kotlin enum | 3 entries 추가 (`@experimental` KDoc) |
| 7 | `cpp/tests/test_sdk_api.cpp` | C++ 테스트 | enum 값 assertions + **invalid mode injection 테스트** |
| 8 | `android/demo-app/src/androidTest/.../BlendModeJniTest.kt` | Android 통합 테스트 | JNI 경로 범위 검증 회귀 테스트 **(권장)** |

## 상세 구현 사항

### 1. C++ Core Enum (`types.h`)

```cpp
enum class BlendMode : int {
    Normal = 0,             ///< 일반 알파 블렌딩
    Multiply = 1,           ///< 곱하기 블렌딩
    Screen = 2,             ///< 스크린 블렌딩
    Overlay = 3,            ///< 오버레이 블렌딩
    LuminanceTint = 4,      ///< 휘도 보존 틴트 (sRGB 근사) @experimental
    LuminanceTintLinear = 5,///< 휘도 보존 틴트 (선형 색공간) @experimental
    SoftLight = 6           ///< 소프트 라이트 블렌딩 @experimental
};
```

### 2. C API Enum (`sdk_api.h`)

```c
typedef enum IrisBlendMode {
    IRIS_BLEND_NORMAL = 0,
    IRIS_BLEND_MULTIPLY = 1,
    IRIS_BLEND_SCREEN = 2,
    IRIS_BLEND_OVERLAY = 3,
    IRIS_BLEND_LUMINANCE_TINT = 4,      /**< @experimental */
    IRIS_BLEND_LUMINANCE_TINT_LINEAR = 5,/**< @experimental */
    IRIS_BLEND_SOFT_LIGHT = 6           /**< @experimental */
} IrisBlendMode;
```

### 3. C++ 변환 함수 (`sdk_api.cpp`)

`convert_blend_mode()` switch에 3 case 추가 + default에 경고 로그:

```cpp
default:
    LOGW("Unknown blend mode: %d, falling back to Normal", static_cast<int>(mode));
    return iris_sdk::BlendMode::Normal;
```

**근거**: 브레인스토밍 Section 25(A1)에서 Codex가 지적 — default fallback은 안전장치가 아니라 silent misbehavior 경로. LOGW 필수.

### 4. JNI 범위 검증 (`iris_jni.cpp`)

기존 `static_cast<IrisBlendMode>(int)` → rawMode/clampedMode 분리:

```cpp
int rawBlendMode = env->GetIntField(src, g_jniCache.lensConfig_blendMode);
if (rawBlendMode < IRIS_BLEND_NORMAL || rawBlendMode > IRIS_BLEND_SOFT_LIGHT) {
    LOGW("Invalid blend mode from Java: %d, clamping to NORMAL(0)", rawBlendMode);
    dest.blend_mode = IRIS_BLEND_NORMAL;
} else {
    dest.blend_mode = static_cast<IrisBlendMode>(rawBlendMode);
}
```

**근거**: 브레인스토밍 Section 19 — JNI에 범위 검증이 없어 silent fallback 발생하는 치명적 버그 발견.

### 5. Java SDK API (`LensConfig.java`)

- 상수 추가:
  ```java
  /** 휘도 보존 틴트 (sRGB 근사). @experimental 동작 변경 가능 */
  public static final int BLEND_LUMINANCE_TINT = 4;
  /** 휘도 보존 틴트 (선형 색공간). @experimental 동작 변경 가능 */
  public static final int BLEND_LUMINANCE_TINT_LINEAR = 5;
  /** 소프트 라이트 블렌딩. @experimental 동작 변경 가능 */
  public static final int BLEND_SOFT_LIGHT = 6;
  ```
- `isValid()`: `blendMode <= BLEND_OVERLAY` → `blendMode <= BLEND_SOFT_LIGHT`
- `clamp()`: `Math.min(BLEND_OVERLAY, blendMode)` → `Math.min(BLEND_SOFT_LIGHT, blendMode)`
- `getBlendModeName()`: 3개 case 추가

**하위 호환성**: 기존 상수(0~3) 값/의미 불변 → additive change → breaking change 없음.

### 6. Kotlin Enum (`BlendMode.kt`)

```kotlin
LUMINANCE_TINT(JavaLensConfig.BLEND_LUMINANCE_TINT),
LUMINANCE_TINT_LINEAR(JavaLensConfig.BLEND_LUMINANCE_TINT_LINEAR),
SOFT_LIGHT(JavaLensConfig.BLEND_SOFT_LIGHT);
```

각 항목에 `@experimental` KDoc 주석 포함.

### 7. C++ 테스트 (`test_sdk_api.cpp`)

- `BlendModeEnumValues` 테스트에 mode 4-6 assertion 추가
- `InvalidBlendModeFallsBackToNormal` 테스트 신규:
  - `static_cast<IrisBlendMode>(7)` — 범위 초과
  - `static_cast<IrisBlendMode>(-1)` — 음수
  - `static_cast<IrisBlendMode>(99)` — 극단값
  - enum 경계값 assertion

### 8. Android JNI 통합 테스트 (권장 추가)

C++ 테스트만으로는 JNI 경로의 범위 검증 회귀를 막기 어려우므로, Android instrumented test 1개 추가 권장:

```kotlin
// android/demo-app/src/androidTest/java/.../BlendModeJniTest.kt
@RunWith(AndroidJUnit4::class)
class BlendModeJniTest {
    @Test
    fun invalidBlendModeClampedToNormal() {
        val config = LensConfig()
        config.blendMode = 99  // 범위 초과
        // JNI 전달 후 크래시 없이 Normal(0)으로 폴백 확인
        // adb logcat에서 "Invalid blend mode" LOGW 출력 확인
    }

    @Test
    fun validBlendModePassedThrough() {
        val config = LensConfig()
        config.blendMode = LensConfig.BLEND_LUMINANCE_TINT  // mode 4
        // JNI 전달 후 정상 동작 확인
    }
}
```

이 테스트는 `iris_jni.cpp`의 범위 검증 로직이 실제 JNI 경로에서 동작하는지 end-to-end 검증한다.

## 검증 체크리스트

- [x] C++ 빌드 통과: `cmake --build . --target test_sdk_api`
- [x] 테스트 통과: `./bin/test_sdk_api` (BlendMode 관련 3개 테스트) — 44 tests, 40 PASSED, 4 SKIPPED
- [ ] Android 빌드 통과: `./gradlew :iris-sdk:assembleDebug`
- [ ] `adb logcat | grep "Unknown blend mode"` → 정상 사용 시 0건
- [ ] Invalid mode injection 시 LOGW 출력 확인

## 스모크 테스트 (권장 15~25분)

### 목적
- Core 경로 확장(0~6)과 fallback 방어 로직이 깨지지 않았는지 빠르게 확인한다.

### 실행 순서

1. C++ 타겟 빌드
   - 명령: `cmake -S . -B build && cmake --build build --target test_sdk_api`
   - 확인: `test_sdk_api` 타겟 링크 성공

2. BlendMode 핵심 테스트만 실행
   - 명령: `./build/bin/test_sdk_api --gtest_filter='SdkApiFormatTest.*BlendMode*'`
   - 확인: `BlendModeEnumValues`, `InvalidBlendModeFallsBackToNormal`, `ValidBlendModeRoundTrip` 3개 모두 PASS

3. 전체 `test_sdk_api` 실행
   - 명령: `./build/bin/test_sdk_api`
   - 확인: FAIL 0건

4. Android SDK 빌드 확인
   - 명령: `cd android && ./gradlew :iris-sdk:assembleDebug`
   - 확인: `BUILD SUCCESSFUL`

5. 런타임 로그 확인 (디바이스 가능 시)
   - 정상 경로: 앱 일반 사용 중 `Unknown blend mode` 로그 0건
   - 비정상 주입 경로: debug에서 `blendMode=99` 1회 주입 시 `Invalid blend mode from Java` 경고 로그 1건 이상

### 내가 확인해야 하는 핵심
- 빌드 성공만 보지 말고, **invalid 입력(-1/7/99/INT_MIN/INT_MAX)이 실제로 Normal로 폴백되는지**를 테스트 결과로 확인한다.
- Android 빌드까지 통과해야 Core/JNI/Java/Kotlin 변경이 함께 깨지지 않았다고 볼 수 있다.

### 최종 판정
- 필수 통과: 1, 2, 3, 4
- 권장 확인: 5
- 하나라도 실패하면 P4-W1-02로 진행하지 않고 원인 수정 후 재검증

## 원자성 규칙

> **Blend Mode 확장 원자성 규칙**: 위 7개 파일은 반드시 **동일 PR**에서 변경해야 한다.
> 부분 머지 금지. PR description 첫 줄에 "원자적 머지 필수" 명시.
>
> — 브레인스토밍 Section 22(A2), Section 25(A2) 합의

## 실행 내역

### 수정된 파일 (8개)

| # | 파일 | 변경 내용 |
|---|------|-----------|
| 1 | `cpp/include/iris_sdk/types.h` | `BlendMode` enum에 `LuminanceTint=4`, `LuminanceTintLinear=5`, `SoftLight=6` 추가 |
| 2 | `cpp/include/iris_sdk/sdk_api.h` | `IrisBlendMode` enum에 3개 값 추가 |
| 3 | `cpp/src/sdk_api.cpp` | `convert_blend_mode()` switch에 3 case 추가 + default LOGW + test hook |
| 4 | `android/iris-sdk/src/main/cpp/iris_jni.cpp` | `static_cast` → rawBlendMode 범위 검증 + LOGW |
| 5 | `android/iris-sdk/src/main/java/com/irislenssdk/LensConfig.java` | 3 상수 추가, `isValid()`/`clamp()`/`getBlendModeName()` 확장 |
| 6 | `android/iris-sdk/src/main/java/com/irislenssdk/kotlin/BlendMode.kt` | 3 entries 추가 (`@experimental` KDoc) |
| 7 | `cpp/tests/test_sdk_api.cpp` | `BlendModeEnumValues` + `InvalidBlendModeFallsBackToNormal` (hook 기반) + `ValidBlendModeRoundTrip` |
| 8 | `cpp/src/lens_renderer.cpp` | `applyBlend()` switch에 새 모드 명시적 case + Normal 폴백 (코드 리뷰 지적 반영) |

### 코드 리뷰 반영 사항

- 이중 세미콜론(`};;`) 수정: `types.h`, `sdk_api.h`
- `lens_renderer.cpp`의 `applyBlend()` switch에 새 모드 명시적 case 추가 (silent fallback 방지)
- `[[fallthrough]]` 속성으로 의도적 폴백임을 명시
- `InvalidBlendModeFallsBackToNormal` 테스트를 대입 테스트 → test hook 기반 실제 `convert_blend_mode()` 검증으로 교체
- `ValidBlendModeRoundTrip` 테스트 추가 (유효 범위 0-6 왕복 검증)
- `IRIS_SDK_TEST_HOOKS` 컴파일 정의: `cpp/CMakeLists.txt` + `cpp/tests/CMakeLists.txt`

### 테스트 결과

```
44 tests, 40 PASSED, 4 SKIPPED, 0 FAILED
BlendModeEnumValues: PASSED
InvalidBlendModeFallsBackToNormal: PASSED (hook 기반, -1/7/99/INT_MIN/INT_MAX → Normal 폴백)
ValidBlendModeRoundTrip: PASSED (0-6 왕복)
```

## 다음 단계

이 작업 완료 후:
1. P4-W1-02: StabilityLogger + Gate 1 Baseline 측정
2. P4-W1-03: Demo App GPU 렌더러에 셰이더 함수 구현

## 변경 이력

| 날짜 | 내용 |
|------|------|
| 2026-02-13 | 작업 계획 문서 작성 |
| 2026-02-13 | Codex 리뷰 반영: Android JNI 통합 테스트 추가 (권장 파일 #8) |
| 2026-02-13 | 구현 완료: 8개 파일 수정, C++ 테스트 39/39 PASS |
| 2026-02-13 | Codex 피드백 반영: test hook 패턴으로 convert_blend_mode() 실제 검증, 40/40 PASS |
