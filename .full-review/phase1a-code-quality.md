# Phase 1A: Code Quality Review -- Vivid Post-Processing Filter

**Scope**: develop 브랜치 대비 +344/-51 lines across 10 files (vivid 관련 변경만 분석)
**Date**: 2026-03-12

---

## Summary

Vivid Post-Processing 필터는 전반적으로 기존 beauty 파이프라인의 아키텍처와 패턴을 일관되게 따르고 있으며, 셰이더 구현과 파이프라인 통합 모두 높은 수준으로 작성되었다. 단일 패스 셰이더, buildEffectiveConfig을 통한 beauty/vivid 독립성, scissor 해제 후 vivid 적용 등의 설계 판단이 적절하다. 아래는 개선이 필요하거나 주의해야 할 항목이다.

**심각도 분포**: Critical 0 / High 2 / Medium 5 / Low 3

---

## Findings

### [H-01] ROI 계산 로직 3중 중복 (Code Duplication)

- **Severity**: High
- **Files**:
  - `cpp/src/sdk_api_v2.cpp` lines 205-243 (CPU path)
  - `cpp/src/sdk_api_v2.cpp` lines 341-373 (GPU texture path)
  - `cpp/src/gpu/gpu_beauty_backend.cpp` lines 1576-1621 (applyTextureId)
- **Description**: face_rect를 픽셀 좌표로 변환하고 20% 마진을 추가하는 ROI 계산 로직이 세 곳에서 거의 동일하게 복사-붙여넣기되어 있다. 마진 비율이나 경계 클램핑 로직을 변경할 때 세 곳 모두 동기화해야 하므로 버그가 발생하기 쉽다. 특히 applyTextureId에서는 face mesh 미러링 보정이 추가되어 있어 세 번째 복사본은 다른 두 개와 미묘하게 다르다.
- **Recommendation**: 공통 헬퍼 함수를 추출한다.

```cpp
// iris_sdk/beauty_roi_utils.h (신규 또는 기존 beauty_roi_manager.h에 추가)
namespace iris_sdk {

struct PixelROI {
    int x, y, w, h;
};

inline PixelROI computePixelROI(const IrisRect& face_rect,
                                 int frame_w, int frame_h,
                                 float margin_ratio = 0.2f) {
    int face_x = static_cast<int>(face_rect.x * frame_w);
    int face_y = static_cast<int>(face_rect.y * frame_h);
    int face_w = static_cast<int>(face_rect.width * frame_w);
    int face_h = static_cast<int>(face_rect.height * frame_h);

    int mx = static_cast<int>(face_w * margin_ratio);
    int my = static_cast<int>(face_h * margin_ratio);
    face_x = std::max(0, face_x - mx);
    face_y = std::max(0, face_y - my);
    face_w = std::min(frame_w - face_x, face_w + 2 * mx);
    face_h = std::min(frame_h - face_y, face_h + 2 * my);

    return {face_x, face_y, face_w, face_h};
}

} // namespace iris_sdk
```

---

### [H-02] applyTexture(TextureHandle) 경로에 vivid 패스 누락

- **Severity**: High
- **File**: `cpp/src/gpu/gpu_beauty_backend.cpp` lines 620-710 (applyTexture 메서드)
- **Description**: `applyTexture(const TextureHandle& input, TextureHandle& output, ...)` 메서드에서는 `config.enabled == false`이면 즉시 패스스루를 반환한다 (line 622-625). `applyTextureId` 경로와 달리 `needsVivid` 체크가 없어, vivid만 활성화(enabled=false, vividIntensity > 0)한 경우 vivid가 적용되지 않는다. 또한, 필터 체인 이후에도 vivid 패스가 실행되지 않는다. `applyTextureId`와의 동작 불일치가 발생한다.
- **Recommendation**: `applyTexture` 메서드에도 동일한 `needsVivid` 로직을 추가한다.

```cpp
// applyTexture 시작 부분 (line 622)
bool needsVivid = config.vividIntensity > 0.01f;
if (!config.enabled && !needsVivid) {
    output = input;
    return IRIS_SDK_OK;
}
// ... 필터 체인 마지막에 vivid 패스 추가
```

---

### [M-01] buildEffectiveConfig에서 vivid 파라미터가 master intensity의 영향을 받지 않음 (의도적 설계이나 문서화 부재)

- **Severity**: Medium
- **File**: `cpp/src/gpu/gpu_beauty_backend.cpp` lines 56-82
- **Description**: `buildEffectiveConfig`에서 smoothing, softFocus, whitening 등은 `master (= config.intensity)`로 스케일링되지만, vivid 계열 파라미터(vividIntensity, vividSaturation 등)는 master의 영향을 받지 않는다. 이것이 의도적 설계라면 (vivid는 자체 마스터 강도가 있으므로) 코드에 명시적 주석이 필요하다. vivid에는 이미 `uIntensity` 유니폼이 별도로 존재하므로 현재 동작이 맞을 가능성이 높지만, 개발자가 나중에 실수로 master 적용을 빠뜨린 것으로 오해할 수 있다.
- **Recommendation**: buildEffectiveConfig에 vivid 독립성에 대한 주석을 추가한다.

```cpp
// vivid 파라미터는 자체 마스터(vividIntensity)를 가지므로
// beauty master intensity와 독립적으로 동작한다.
// effective.vividIntensity = config.vividIntensity; (변환 없이 그대로)
```

---

### [M-02] Vivid 셰이더에서 warmth 채널 조정의 하드코딩된 매직 넘버

- **Severity**: Medium
- **File**: `cpp/src/gpu/shader_sources.cpp` lines 670-674
- **Description**: 웜톤 시프트에서 `0.04`, `0.02`, `0.03`이라는 매직 넘버가 하드코딩되어 있다. 이 값들은 R/G/B 채널 시프트 양이며, 튜닝 시 셰이더 소스를 직접 수정해야 한다. 다른 셰이더(예: COLOR_BALANCE_FRAGMENT)에서도 동일한 패턴(`0.08`, `0.04`)이 사용되고 있어 프로젝트 전반의 관행이긴 하지만, vivid는 런타임 조정 가능한 유니폼으로 노출하는 것이 향후 유연성에 유리하다.

```glsl
// 현재 코드 (line 670-674)
result.r += uWarmth * 0.04;
result.g += uWarmth * 0.02;
result.b -= uWarmth * 0.03;
```

- **Recommendation**: 현재 단계에서는 기존 관행과 일관되므로 수용 가능하다. 향후 셰이더별 튜닝 파라미터를 유니폼으로 분리하는 리팩터링 시 일괄 처리한다. 지금은 매직 넘버에 주석을 추가하는 것만으로 충분하다.

---

### [M-03] applyTextureId의 scissor-empty + vivid-only 경로에서 pong 버퍼 누수 가능성

- **Severity**: Medium
- **File**: `cpp/src/gpu/gpu_beauty_backend.cpp` lines 1762-1776
- **Description**: ROI scissor 교집합이 비어있을 때 vivid-only 경로에서 `ping->fbo_id`를 출력으로 사용한 후, ping/pong을 `previous_output_ping_`/`previous_output_pong_`에 저장하여 다음 프레임에 반환한다. 그런데 vivid를 `ping->fbo_id`에만 렌더링하고, `pong`은 사용하지 않았음에도 `previous_output_pong_`에 저장한다. pong 텍스처는 아무도 참조하지 않는 상태에서 다음 프레임까지 풀에 반환되지 않아 한 프레임 동안 불필요하게 점유된다.
- **Recommendation**: vivid-only 경로에서 pong이 사용되지 않았다면 즉시 반환한다.

```cpp
if (needsVivid) {
    executeVividPass(current_input, ping->fbo_id, ...);
    *output_texture = ping->texture_id;
} else {
    *output_texture = current_input;
}
if (ping) { previous_output_ping_ = ping; }
// pong은 이 경로에서 사용되지 않았으므로 즉시 반환
if (pong) {
    texture_pool_->releaseTexture(pong);
    pong = nullptr;
}
```

---

### [M-04] SoftFocus 후 ping-pong 스왑 조건이 `needsVivid`에만 의존

- **Severity**: Medium
- **File**: `cpp/src/gpu/gpu_beauty_backend.cpp` line 1904
- **Description**: SoftFocus 패스 이후의 ping-pong 스왑이 `if (pong && needsVivid)`로 조건 분기된다. 이것은 SoftFocus가 마지막 beauty 패스이고, 다음에 vivid만 올 수 있다는 현재 파이프라인 구조에 의존한다. 향후 SoftFocus 이후에 새로운 패스가 추가되면 이 조건을 업데이트해야 하는데, 이를 놓치기 쉽다.

```cpp
// line 1904 (현재)
if (pong && needsVivid) current_output = (current_output == ping) ? pong : ping;
```

- **Recommendation**: 패턴을 다른 패스와 일관되게 만든다. 다른 패스들은 무조건 `if (pong) current_output = ...`으로 스왑한다. SoftFocus도 동일하게 하고, vivid 이후 추가 패스 여부와 무관하게 동작하도록 한다.

```cpp
// 일관된 패턴
if (pong) current_output = (current_output == ping) ? pong : ping;
```

---

### [M-05] C API 기본값 불일치: `iris_sdk_default_beauty_config_v2_c`의 enabled vs `defaults()`

- **Severity**: Medium
- **File**:
  - `cpp/src/sdk_api_v2.cpp` line 145: `config->enabled = 1;`
  - `cpp/include/iris_sdk/beauty_filter.h` line 420: `cfg.enabled = false;`
  - `android/.../BeautyFilterConfigV2.java` line 199: `DEFAULT_ENABLED = false;`
- **Description**: C API의 `iris_sdk_default_beauty_config_v2_c()`는 `enabled = 1 (true)`을 기본값으로 설정하지만, C++ Helper의 `defaults()`와 Java의 `DEFAULT_ENABLED`은 모두 `false`를 기본값으로 사용한다. 이는 vivid 변경사항 이전부터 존재하는 불일치이지만, vivid 필드 추가와 함께 이 불일치가 더 중요해졌다 -- vivid만 사용하려는 호출자가 C API를 통해 기본값을 가져오면 beauty까지 활성화되기 때문이다.
- **Recommendation**: 이 불일치는 기존 코드의 이슈이므로 이번 PR 범위에서 수정할 필요는 없지만, 별도 이슈로 추적하는 것을 권장한다.

---

### [L-01] VividUniforms 구조체가 기존 UniformLocations 패턴과 별도 구조체로 분리

- **Severity**: Low
- **File**: `cpp/include/iris_sdk/gpu/gpu_beauty_backend.h` lines 508-514
- **Description**: 다른 셰이더 유니폼들은 `UniformLocations` 구조체를 재사용하지만, vivid 유니폼은 전용 `VividUniforms` 구조체를 사용한다. 마찬가지로 `FreqSepGaussianUniforms`, `FreqSepCompositeUniforms`, `LuminanceSharpenUniforms`도 별도 구조체이다. 이는 기존 패턴(FreqSep 계열)을 따른 것이므로 일관성 측면에서는 문제 없으나, 장기적으로 `UniformLocations`의 범용 필드가 점점 늘어나는 것보다 이 접근이 더 낫다.
- **Recommendation**: 현재 접근 유지. 참고 사항으로만 기록.

---

### [L-02] JNI 필드 ID null 체크에서 vivid 필드가 beauty 필드와 동일 수준으로 검증됨

- **Severity**: Low
- **File**: `android/iris-sdk/src/main/cpp/iris_jni.cpp` lines 177-178
- **Description**: `beautyConfigV2_vividIntensity || !beautyConfigV2_vividSaturation || ...` null 체크가 기존 패턴과 동일하게 구현되어 있다. 전체 초기화 성공/실패가 원자적으로 판단되므로, 하나라도 실패하면 전체가 실패한다. 이 부분은 적절하다.
- **Recommendation**: 변경 불필요. 적절한 구현.

---

### [L-03] Vibrance 알고리즘에서 sat 계산의 edge case

- **Severity**: Low
- **File**: `cpp/src/gpu/shader_sources.cpp` lines 657-661
- **Description**: Vibrance 구현에서 `sat = max(R,G,B) - min(R,G,B)`로 현재 채도를 계산한다. `smoothstep(0.4, 0.0, sat)`은 sat이 0에 가까울수록 vibrance 적용이 강해지고, 0.4 이상이면 0이 된다. 이 구현은 정확하고 GPU-friendly하나, 입력이 이미 HDR 범위(1.0 초과)인 경우 sat이 0.4를 초과하여 vivid가 의도보다 약하게 적용될 수 있다. 다만 이전 뷰티 패스에서 이미 `clamp(result, 0.0, 1.0)`을 적용하므로 실제 문제는 발생하지 않는다.

```glsl
float sat = max(max(result.r, result.g), result.b) -
            min(min(result.r, result.g), result.b);
float vibranceAmount = uSaturation * smoothstep(0.4, 0.0, sat);
result = mix(vec3(lum), result, 1.0 + vibranceAmount);
```

- **Recommendation**: 변경 불필요. 파이프라인 순서상 안전.

---

## Positive Observations

### 잘 구현된 부분

1. **buildEffectiveConfig 분리**: beauty disabled + vivid-only 경로를 깔끔하게 처리. beauty 수치를 중립값으로 덮어쓰는 방식이 파이프라인 하류의 조건 분기를 최소화한다.

2. **Scissor 해제 후 vivid 적용** (line 1907-1910): ROI 기반 beauty 처리가 끝난 후 glScissor를 해제하고 전체 프레임에 vivid를 적용하는 흐름이 정확하다. Vivid는 전체 프레임 효과이므로 ROI와 독립적이어야 한다는 설계 의도가 코드에 명확히 반영되었다.

3. **셰이더 효율성**: Vivid 셰이더가 텍스처 샘플 1회 + ALU 연산만으로 구현되어, MID tier 기기에서도 약 0.3ms 이내에 실행 가능한 구조이다. branch 기반 early-out (`if (uSaturation > 0.01)` 등)도 적절하다.

4. **레이어 간 일관성**: C struct(`IrisBeautyConfigV2`) -> C++ struct(`BeautyFilterConfigV2`) -> Java class(`BeautyFilterConfigV2`)까지 vivid 4개 필드가 모든 레이어에 빠짐없이 추가되었고, 변환 함수(`toCppConfigV2`/`fromCppConfigV2`), JNI 매핑, Builder 패턴, `isValid()`, `clamp()` 모두 업데이트되었다.

5. **기본값 0.0f 전략**: 모든 vivid 파라미터의 기본값이 0.0f이므로, 기존 사용자가 업데이트해도 동작이 변하지 않는다 (하위 호환).

---

## Action Items Summary

| ID | Severity | Action | Effort |
|----|----------|--------|--------|
| H-01 | High | ROI 계산 로직 3중 중복 → 공통 헬퍼 추출 | 1h |
| H-02 | High | applyTexture 경로에 vivid 패스 추가 | 30m |
| M-01 | Medium | buildEffectiveConfig에 vivid 독립성 주석 추가 | 5m |
| M-02 | Medium | warmth 매직 넘버 주석 보강 | 5m |
| M-03 | Medium | scissor-empty 경로에서 pong 즉시 반환 | 15m |
| M-04 | Medium | SoftFocus 후 스왑 조건을 무조건 스왑으로 변경 | 10m |
| M-05 | Medium | C API 기본값 불일치 이슈 추적 등록 | 5m |
| L-01 | Low | 현재 유지 (참고) | - |
| L-02 | Low | 변경 불필요 | - |
| L-03 | Low | 변경 불필요 | - |

**총 예상 수정 시간**: 약 2시간 10분 (H 항목 90분 + M 항목 40분)

---

## Conclusion

Vivid Post-Processing 필터 구현은 기존 아키텍처를 잘 따르고 있으며, 셰이더 품질과 파이프라인 통합 모두 프로덕션 수준이다. Critical 이슈는 없다. High 2건(ROI 중복, applyTexture 경로 누락)은 머지 전 수정을 권장하며, Medium 항목 중 M-03(pong 누수)과 M-04(스왑 조건)도 가능하면 함께 처리하는 것이 좋다. 나머지 Medium/Low는 후속 작업으로 처리 가능하다.
