# P6-W1: EyeRenderPacket 도입 + avg_iris_luma ROI 평균 실측

> **상태**: 인사이트 작성 완료. 세부 계획 본문 작성 대기.
> **작성**: 2026-04-23
> **선행 의존**: P6-W0
> **후속 의존**: P6-W2, W3, W4~W7 (모두 기반)

---

## 1. 인사이트 (세션 간 맥락 보존) ⭐

### 1.1 왜 이 W가 중요한가

**모든 후속 W의 전제.** EyeRenderPacket은 렌더러와 검출기 사이 **인터페이스 고정**이다. 지금 이 계약을 엉성하게 두면 후속 W에서 "새 필드 추가할 때마다 공개 C API/JNI/Java 모두 흔들리는" 지옥을 만든다.

avg_iris_luma ROI 측정은 **블렌드(W2) 정규화의 기반**. 현재(S1 직후) `uAvgIrisLum` uniform 미설정 → GL 기본값 0.0 → 셰이더 clamp 하한(0.01)에 걸려 LTL `scale`이 최대치로 고정됨 = **LTL이 과도 tint** 상태. 임시 동작으로는 돌아가지만 W2 전에 반드시 복구.

### 1.2 R3에서 발견된 Claude 편향 — 이 W에서 반드시 반영

**Codex R3 원문**: 
> "§1 C9의 `textureLod(camera, iris_center, 3.0)` 1회 샘플은 내 R2 수식이 아니다. 내 R2는 `avg = sum(dot(rgb,w)*mask)/sum(mask)`, `mask = inner iris ∩ eyelidMask`였다. 중심 1샘플은 pupil/dark center를 iris 평균으로 오인할 수 있다."

즉 Claude가 R2 종합에서 편의상 "1샘플"로 적었는데, 이는 **동공(검은 점) 픽셀을 iris 평균으로 착각**시키는 오류. 짙은 홍채 사용자(아시아인 주류)에서 특히 파탄. 반드시 masked ROI 평균으로 구현.

**구체 수식**:
```glsl
// inner iris 영역 (홍채 중심 반경 0.65 이내) ∩ eyelidMask
// 동공 검은 픽셀 자동 배제 가능 (pupil_center 없어도 iris 중앙 어두운 영역 비중이 낮음)
float mask = (dist_from_iris_center < 0.65 * iris_radius) ? 1.0 : 0.0;
mask *= eyelid_mask(uv);  // 눈꺼풀 가린 영역 제외
float weight_sum = sum(mask over ROI);
vec3 weighted_rgb_sum = sum(rgb * mask);
float avg_iris_luma = dot(weighted_rgb_sum / weight_sum, LUMA_COEFFS);
```

### 1.3 측정 위치: shader vs CPU

**선택지 A (shader 내)**: 매 프레임 프래그먼트에서 계산. 비용: iris ROI 내 여러 fetch 필요. 정확하지만 무거움.

**선택지 B (CPU / compute shader 1회)**: 매 프레임 CPU에서 iris ROI 크롭 후 평균. `gpu_lens_renderer.cpp`에서 frame 단위 단발 측정 후 uniform 주입. 1프레임 지연 있음.

**Codex R3 권고 (묵시적)**: "1프레임 지연은 허용 가능하다" → **선택지 B 유력**. 단 구체 구현은 W1 브레인스토밍에서 확정.

**Claude 의견 (W1 브레인스토밍 때 제시 예정)**: CPU 경로는 프레임 캡처가 이미 RGB로 돌고 있으니 iris 주변 픽셀 평균 계산 비용 미미. 셰이더에서 하면 fetch 늘어나고 GPU bandwidth 낭비.

### 1.4 EyeRenderPacket 최종 스키마 (99 §1.3에서 확정)

```cpp
struct EyeRenderPacket {
    // 필수
    glm::vec2 iris_center_norm;
    float     iris_radius_norm;
    glm::vec2 ellipse_center;
    glm::vec3 ellipse_radii;          // (rxInner, rxOuter, ry)
    float     ellipse_rotation;
    float     eye_top, eye_bottom;    // Y-slab fallback
    float     visibility;             // 0.0~1.0
    uint64_t  timestamp_ms;

    // 선택 (없으면 기능 자동 off)
    std::optional<glm::vec2> pupil_center_norm;       // parallax + 동공 정렬용
    std::optional<glm::vec2> head_pose_yaw_roll;      // env rotation용
    std::optional<glm::vec3> reflection_dir;          // head_pose 대체 가능
    std::optional<float>     avg_iris_luma;           // blend 정규화용 (없으면 self-measure)
    std::optional<float>     eye_depth_mm;            // 스케일 보정용
    std::optional<float>     render_confidence;       // alpha hysteresis용
};
```

**금지**: raw `face_mesh` 의존, detector-specific landmark 인덱스.
**Optional 처리**: 없으면 기능 off. priors로 채우지 않는다 (avg_iris_luma는 self-measure 경로만 예외).

### 1.5 occlusion 필드 설계 (Codex R3 경고)

**Codex R3 원문**:
> "C11 `occlusion` 언급과 실제 스키마가 불일치한다. `occlusion = visibility + aperture mask로 표현`인지, 별도 optional인지 정리해야 한다."

**결정**: occlusion은 별도 필드가 아니라 **`visibility`(0~1) + aperture mask(`ellipse_*`)의 조합**으로 표현. 이 결정 근거를 W1 브레인스토밍에서 Codex에게 재확인 필요.

### 1.6 기존 코드와의 접점

**현재 `gpu_lens_renderer.cpp`의 입력**: `IrisResult` 구조체를 직접 받음 (detector 산출물 그대로). 이걸 EyeRenderPacket으로 **래핑**하는 어댑터 레이어 필요.

**어댑터 위치 제안**: `cpp/src/gpu/eye_render_packet_adapter.{h,cpp}` 신규. `IrisResult` → `EyeRenderPacket` 변환 담당.

**공개 C API (`sdk_api_v2.cpp`) 변경 없음**: 외부에서 보는 API는 그대로. 내부에서만 어댑터 경유.

### 1.7 기존 하드코드 흔적 — S1에서 주석 처리됨

```cpp
// P5-W3-05 S1 D4: uAvgIrisLum = 0.35f 하드코드 제거
// 근거: "priors 덮어씌움" (Codex R2/R3)...
// glUniform1f(lens_uniforms_.uAvgIrisLum, 0.35f);  // 제거됨
```

W1 구현 시 이 주석을 실제 측정 로직으로 **치환**. "S2에서 EyeRenderPacket.avg_iris_luma 또는 self-measure로 교체" 주석도 업데이트.

### 1.8 Fallback 체인 (측정 실패 대응)

Codex R3 제안:
```
1. EyeRenderPacket.avg_iris_luma 가 있으면 사용
2. 없으면 renderer에서 self-measure
3. 측정 실패 (검출 실패 등) 시 이전 유효값 hold
4. 장기 hold (3프레임 이상) 시 중립 상수 fallback
```

**중립 상수 값**: S1에서 0.35는 제거됨. 새 값은 "중립"으로 재명명해야 하고, 0.35는 한국인 평균 근사라 **중립 값으로 부적절**. W1 브레인스토밍에서 합의 필요:
- 옵션 A: 0.5 (중간 회색)
- 옵션 B: 0.3 (보수적, 과도 tint 방지)
- 옵션 C: 이전 N프레임 평균의 지수 이동 평균 (EMA 시동값)

### 1.9 W1 브레인스토밍 시 Codex/Gemini에게 던질 핵심 질문

1. **측정 위치**: CPU 단발(1프레임 지연) vs shader 내? 각자 추천 이유는?
2. **ROI 수식**: `inner iris r<0.65 AND eyelidMask`가 충분한가, 더 세밀한 마스크 필요?
3. **Fallback 중립값**: 0.3 / 0.5 / EMA 중?
4. **어댑터 위치**: `eye_render_packet_adapter.{h,cpp}` 제안 vs 더 좋은 구조?
5. **시간적 스무딩**: avg_iris_luma 자체에 EMA 적용? (프레임 간 급변 방지). W1 TemporalStabilizer와 중복 가능성 체크.
6. **occlusion**: visibility + aperture_mask로 흡수 결정 재확인.
7. **head_pose_yaw_roll / reflection_dir**: W2 refiner 없이 지금 활용 가능한 경로 있는가? 없으면 optional만 준비해두고 이 W에서는 구현 스킵?

### 1.10 구현 범위 (이 W에서 건드리는 파일)

- 신규:
  - `cpp/include/iris_sdk/gpu/eye_render_packet.h`
  - `cpp/src/gpu/eye_render_packet_adapter.{h,cpp}` (위치 미정, 브레인스토밍 후 확정)
- 수정:
  - `cpp/include/iris_sdk/gpu/gpu_lens_renderer.h` — 입력 파라미터 `IrisResult&` → `EyeRenderPacket&`
  - `cpp/src/gpu/gpu_lens_renderer.cpp` — avg_iris_luma measurement 로직, uniform 주입 재구현
  - `cpp/src/sdk_api_v2.cpp` — 내부 어댑터 호출로 전환 (C API 시그니처 불변)

### 1.11 주의: W1 TemporalStabilizer와의 영역 분리

**Claude R2에서 확정 (99 §1.2 C7)**:
> 렌더러는 material-only temporal envelope만 가진다. geometry hold는 상위 계층(W1 TemporalStabilizer)에서만.

즉 avg_iris_luma가 프레임 간 심하게 튀는 경우 **렌더러에서 EMA를 걸 수도 있지만**, 이는 "material 속성"이라 렌더러 책임. 반면 iris_center/iris_radius 스무딩은 TemporalStabilizer가 이미 처리하므로 렌더러가 중복 스무딩 금지.

`avg_iris_luma`는 material에 해당 → 렌더러가 EMA 적용 가능. 단 W1 브레인스토밍에서 "프레임 간 변동 심각도" 확인 후 적용 여부 결정.

### 1.12 성능 예산

- 렌더러 추가 pass 0개 (Codex R3 엄수)
- avg_iris_luma CPU 측정: iris ROI ≈ 50×50px 수준 → 2500 픽셀 평균. 메인 스레드에서 < 0.1ms 예상.
- uniform 주입은 기존대로 매 프레임 1회.

---

## 2. 배경/맥락

### 2.1 왜 이 W가 Phase 6의 첫 번째인가

**모든 후속 W의 기반**이기 때문. 구체적으로:

- **W2 (블렌드)**: `blendTintLinearV2`의 `uAvgIrisLum` uniform이 제대로 주입돼야 LTL 정규화가 동작. W1이 avg_iris_luma 측정 경로를 확보해야 W2 의미 있음.
- **W3 (환경 반사 계층)**: `EyeRenderPacket.reflection_dir` 또는 `head_pose_yaw_roll` optional 필드를 참조. 구조체가 없으면 W3 인터페이스 정의 불가.
- **W4~W7 (벤치)**: 벤치 프로토타입이 측정 데이터(avg_iris_luma, pupil_center 등)를 필요로 함. W1 없으면 프로토타입 설계 부정확.
- **W8 (Pupil material 조건부)**: renderMask hook은 W3에 있지만, hook 활성화 판정이 W4 결과에 의존. W4 벤치가 W1 산출물을 쓰므로 연쇄 의존.

### 2.2 99_final_decision.md에서 이 W가 해결하는 결정

- **C8 EyeRenderPacket 도입** (§1.2) — 렌더러 ↔ 검출기 계약 구조화
- **C9 avg_iris_luma masked ROI 평균 실측** (§1.2) — 0.35 하드코드 폐기 후속 측정 경로
- **C11 W2-W3 경계 명확화** (§1.2) — occlusion은 visibility + aperture_mask로 흡수, 별도 필드 아님

### 2.3 이 W가 해결하지 **않는** 것

- 블렌드 수식 변경 — W2 범위
- 환경 반사 합성 — W3/W4 범위
- pupil_center 실제 활용 (parallax, Pupil material restore) — W8 조건부
- head_pose 실제 활용 (env 회전) — W4 벤치 후

즉 W1은 **"계약 + 측정 인프라"**이지 **"기능 활용"**이 아니다.

---

## 3. 전제 조건

**반드시 확인할 것** (W1 브레인스토밍 시작 전):

1. ✅ **P6-W0 인덱스 읽기** — 전체 로드맵 맥락 파악
2. ✅ **S1 롤백 커밋 확인** — `git log --oneline | grep 9aee86d`. S1에서 `uAvgIrisLum=0.35f` 하드코드 제거된 상태 확인.
3. ✅ **99_final_decision.md §1.2 C8/C9/C11 + §1.3 EyeRenderPacket 스키마** 읽기
4. ✅ **08_codex_r2.md I6, I7, I9** 읽기 — C9 측정 수식 원문, EyeRenderPacket 초안 원출처
5. ✅ **13_codex_r3.md §1, §2** 읽기 — "1샘플 오류" 지적 + 스키마 불일치 경고
6. ✅ **현재 gpu_lens_renderer.cpp** 의 avg_iris_luma 주입 지점 확인 (S1 주석 남아있음)

**GPU 빌드 환경 준비**:
- `cpp/cmake-build-debug`가 최신 상태여야 함 (재빌드 없이 incremental)
- 실기기 1대 연결 가능 (avg_iris_luma 실측 검증용 — MID tier 1대 충분)

---

## 4. 목표

**W1 완료 시 달성 상태**:

1. **`EyeRenderPacket` 구조체 정의 완료** (`cpp/include/iris_sdk/gpu/eye_render_packet.h`)
   - 99 §1.3 스키마 그대로 + 브레인스토밍으로 확정된 세부 (occlusion 흡수 확정 등)
2. **내부 adapter 레이어 구현** — `IrisResult` → `EyeRenderPacket` 변환
3. **`GPULensRenderer` 입력 변경** — 기존 `IrisResult&` → `EyeRenderPacket&`
4. **avg_iris_luma masked ROI 평균 측정 동작** — S1에서 비운 자리 복구
5. **Fallback 체인** — packet optional → self-measure → hold → 중립 상수
6. **공개 C API 불변** — `sdk_api_v2.cpp` 시그니처 유지, 내부만 어댑터 경유
7. **실기기 1회 확인** — LTL 렌더링이 "과도 tint" 상태에서 정상 복귀

### 4.1 Definition of Done (구체 검증)

- [ ] `eye_render_packet.h` 파일 존재, 필수/optional 필드 모두 포함
- [ ] `gpu_lens_renderer.h` 의 render API가 `EyeRenderPacket&`를 받음
- [ ] S1 주석 `// glUniform1f(lens_uniforms_.uAvgIrisLum, 0.35f); // 제거됨`이 실제 측정 코드로 치환됨
- [ ] C++ 빌드 통과 (`cmake --build . --target iris_sdk`)
- [ ] 단위 테스트: EyeRenderPacket 구조체 생성/복사/optional 처리 (최소 1개)
- [ ] 실기기 1회 렌더링: LTL 모드에서 과도 tint 없음 (육안 확인)
- [ ] 성능 회귀 없음 (MID tier 기준 FPS 유지)

### 4.2 Out of scope (W1에서 하지 않는 것)

- pupil_center 실제 활용 로직 (parallax 등) — 필드 준비만, 활용은 W8
- head_pose_yaw_roll 실제 활용 — 필드 준비만, 활용은 W4
- 블렌드 수식 변경 — W2
- 시간적 스무딩 대대적 개편 — W6
- Android demo UI 정리 — W9 머지 후

---

## 5. 99에서 확정된 사항

### 5.1 EyeRenderPacket 스키마 (99 §1.3 확정)

```cpp
// cpp/include/iris_sdk/gpu/eye_render_packet.h (신규)
// 내부 구조체. 공개 C API는 변경 없음.

#pragma once
#include <cstdint>
#include <optional>
#include <glm/glm.hpp>

namespace iris_sdk::gpu {

struct EyeRenderPacket {
    // 필수 필드
    glm::vec2 iris_center_norm;
    float     iris_radius_norm;

    // 비대칭 타원 (aperture mask)
    glm::vec2 ellipse_center;
    glm::vec3 ellipse_radii;      // (rxInner, rxOuter, ry)
    float     ellipse_rotation;

    // 눈꺼풀 Y-slab fallback
    float     eye_top;
    float     eye_bottom;

    // 가시성 (0.0~1.0). occlusion은 이 값 + aperture mask로 표현
    float     visibility;
    uint64_t  timestamp_ms;

    // 선택 필드 (없으면 기능 자동 off)
    std::optional<glm::vec2> pupil_center_norm;       // W8 parallax + 동공 정렬용
    std::optional<glm::vec2> head_pose_yaw_roll;      // W4 env rotation용
    std::optional<glm::vec3> reflection_dir;          // head_pose 대체 가능
    std::optional<float>     avg_iris_luma;           // W2 blend 정규화용 (없으면 self-measure)
    std::optional<float>     eye_depth_mm;            // 스케일 보정용
    std::optional<float>     render_confidence;       // alpha hysteresis용
};

} // namespace iris_sdk::gpu
```

### 5.2 avg_iris_luma self-measure 수식 (99 §1.2 C9, Codex R2 원문)

```
avg = sum(dot(rgb, LUMA_COEFFS) * mask) / sum(mask)

mask = (distance(uv, iris_center) < 0.65 * iris_radius) AND eyelid_mask(uv)
LUMA_COEFFS = vec3(0.2126, 0.7152, 0.0722)  // Rec.709 linear
```

**Codex R3 명시**: "1프레임 지연은 허용 가능하다" → CPU 경로 기본. 프레임 캡처 완료 직후 CPU에서 iris ROI 추출 → 평균 휘도 계산 → 다음 프레임 렌더링 시 uniform 주입.

### 5.3 Fallback 체인 (W1에서 확정됨)

```
1. EyeRenderPacket.avg_iris_luma.has_value()
   → 해당 값 사용
2. else if self-measure 성공
   → ROI 평균 사용, 이번 프레임 결과를 다음 프레임으로 전달
3. else if 이전 유효값 hold_count < 3
   → 이전 값 유지, hold_count++
4. else
   → 중립 상수 사용 (기본값: 0.35 from Codex R2 / W1 브레인스토밍에서 재확인)
```

**중립 상수 값**: 99에서는 미확정. W1 브레인스토밍에서 확정. Claude 추천: **0.35 유지** (실측 어려울 때 보수적, Codex 원래 값).

### 5.4 occlusion 필드 처리 (Codex R3 §2)

99 §1.3 스키마에는 occlusion이 **별도 필드 없음**. `visibility` (0~1) + `ellipse_radii/rotation` (aperture mask)로 표현:

- 눈 감김: visibility 감소 + ellipse_radii.y(세로 반경) 감소
- 렌즈 가림: aperture mask에서 감쇄 영역이 표현됨
- 속눈썹 그림자: renderer 내부에서 `eye_top/bottom` fallback으로 처리

**재확인 포인트**: W1 브레인스토밍에서 "occlusion 별도 필드 vs visibility+aperture 흡수" Codex/Gemini 재확인.

### 5.5 공개 C API 불변 (99 §1.2 C8)

기존:
```c
// cpp/include/iris_sdk/sdk_api.h — 변경 없음
iris_sdk_result_t iris_sdk_render_lens_gpu(
    const uint8_t* frame_data,
    int width, int height,
    int format,
    IrisResult* result,    // ← raw 유지
    LensConfig* lens_config
);
```

내부에서만 어댑터 경유:
```c++
// cpp/src/sdk_api_v2.cpp 수정
iris_sdk_result_t iris_sdk_render_lens_gpu(...) {
    // IrisResult → EyeRenderPacket 어댑터 호출
    auto packet = adapt_iris_result_to_packet(*result);
    return g_gpu_lens->render(packet, lens_config);
}
```

---

## 6. 미결 사항 (W1 브레인스토밍에서 풀 질문)

### 6.1 측정 경로: CPU 1회 vs shader 내 (CRITICAL)

**옵션 A (CPU 1회)**: 프레임 캡처 후 CPU에서 iris ROI 픽셀 직접 접근 → 평균 계산 → uniform 주입. 1프레임 지연.
- 장점: 구현 단순, GPU 부하 0
- 단점: 1프레임 지연 (눈 감김 전환 시 약간 부정확할 수 있음)

**옵션 B (shader 내)**: 프래그먼트 셰이더에서 ROI 내 여러 fetch 평균. 매 프레임 실시간.
- 장점: 지연 0
- 단점: 매 프레임 fetch 9~16개 추가, GPU bandwidth 낭비

**옵션 C (compute shader 1회/프레임)**: 별도 compute pass에서 ROI 평균 → uniform 주입.
- 장점: GPU에서 처리, 지연 0~1프레임
- 단점: GLES 3.1 compute 드라이버 이슈 (Adreno 일부). 99 §1.2 C-F에서 compute shader 도입은 이미 기각 상태.

**Codex R2 묵시적**: 옵션 A (1프레임 지연 허용). 확정 전 Codex/Gemini 재확인.

**Claude 의견**: 옵션 A 강력 추천. 블렌드 정규화가 1프레임 지연으로 시각적으로 알아챌 수 없음.

### 6.2 ROI 마스크 구체: `inner iris r<0.65 AND eyelidMask` 충분?

Codex R2의 원문: "inner iris ∩ eyelidMask". 
**질문**:
- r<0.65는 iris 반경의 65%. pupil 영역(보통 30~40%)을 제외하고도 안전한 margin?
- eyelidMask는 어떻게 얻나? `ellipse_radii`로 파생?
- 동공이 밝은 경우(홀수 조명 반사)를 평균에 포함해야 하나?

**Claude 제안**: 
```
mask = (dist < 0.55 * iris_radius)  // pupil 제외 강화
     * aperture_mask(ellipse)        // 눈꺼풀 가림 제외
     * (luminance_not_outlier)       // 이상치 제외 (optional)
```

W1 브레인스토밍에서 Codex/Gemini 피드백.

### 6.3 Fallback 중립 상수 값

- 옵션 α: 0.35 (Codex 원래 값, 한국인 평균)
- 옵션 β: 0.5 (중간 회색)
- 옵션 γ: 0.3 (보수적, 과도 tint 방지)
- 옵션 δ: 이전 N프레임 EMA 시동값

**Claude 추천**: 0.35 그대로. 이번 세션 메모리 `feedback_real_data_first` 고려 — "priors 덮어씌움"은 사용 중 적용이 문제지, fallback 상수 자체는 괜찮음.

### 6.4 어댑터 레이어 위치

**옵션 1**: `cpp/src/gpu/eye_render_packet_adapter.{h,cpp}` — GPU 네임스페이스 하위
**옵션 2**: `cpp/src/eye_render_packet_adapter.{h,cpp}` — SDK 공통
**옵션 3**: `gpu_lens_renderer.cpp` 내부 static 함수 — 가장 단순

**Claude 추천**: 옵션 1. GPU 렌더링 전용 계약이라 GPU 네임스페이스 적절.

### 6.5 시간적 스무딩 중복 체크

**질문**: avg_iris_luma가 프레임 간 급변하면 EMA 적용?

- 적용 필요: 블링크 시 iris 영역 축소 → 평균 급변 → LTL 시각적 급변
- 미적용 OK: W1 TemporalStabilizer가 iris_center/radius 스무딩하니까 연쇄 효과로 안정적

**Claude 의견**: 측정값 EMA **렌더러 레벨에서 적용**. 이유: avg_iris_luma는 "material 속성"이라 W1 경계 원칙(렌더러 = material temporal envelope)과 일치. 계수 `α = 0.3` 정도로 완만하게.

### 6.6 head_pose_yaw_roll / reflection_dir optional

**질문**: W1 시점에 이 필드 준비만 하고 실제 활용은 W4?

**Claude 의견**: 맞음. 스키마에 optional로 추가하되 current renderer에서는 무시 (향후 W4 벤치에서 env rotation 구현 시 활용). Adaptation 구조만 확보.

### 6.7 render_confidence optional 설계 — W1 TemporalStabilizer와의 연결

**질문**: render_confidence는 W1 TemporalStabilizer(P5-W1)에서 이미 제공 중인 값? 신규 계산?

**Claude 추정**: TemporalStabilizer에 `getConfidence()` 같은 API 이미 존재. 어댑터에서 그걸 읽어 packet에 포함.

**확인 필요**: `cpp/src/temporal_stabilizer.cpp`에서 confidence 제공 여부.

---

## 7. W1 브레인스토밍 시작 체크리스트

### 7.1 새 세션 시작 시 읽을 파일 (15분)

**필수**:
1. `docs/workPaper/P6-W0_index.md` §1 (5분)
2. `docs/workPaper/P6-W1_eye_render_packet.md` 전체 (10분)
3. `docs/workPaper/P5-W3-05_brainstorm/99_final_decision.md` §1.2 C8/C9 + §1.3 (5분)

**선택** (쟁점 깊이 이해 시):
- `docs/workPaper/P5-W3-05_brainstorm/08_codex_r2.md` §I6/I7 — Codex R2 원문
- `docs/workPaper/P5-W3-05_brainstorm/13_codex_r3.md` §1 — "1샘플 오류" 지적

### 7.2 Codex/Gemini 송신 프롬프트 초안

```
@docs/workPaper/P6-W1_eye_render_packet.md 읽고, 섹션 6 "미결 사항" 7개
에 대해 각자 입장/권고 정리해서 docs/workPaper/P6-W1_brainstorm/
{codex|gemini}_w1.md로 작성해줘.

규칙:
- 6.1~6.7 각 항목에 대해 "추천 + 근거 1~2줄"
- 새 쟁점 제기 금지. 이미 정의된 7개에만 답.
- 특히 6.1 (CPU vs shader) 측정 경로와 6.5 (EMA 적용 여부)
  는 Codex R2/R3 입장을 명시 재확인.
- Gemini는 Pupil 관련 W8 조건부 트랙에서 이 W1 결정이
  어떤 영향 주는지 특히 지적 환영.
```

### 7.3 예상 쟁점 (W1 브레인스토밍에서 의견 갈릴 지점)

- **6.1 측정 경로**: Codex 옵션 A 지지 예상. Gemini도 A. 일치 가능성 높음.
- **6.2 ROI 마스크**: Claude 제안 `r<0.55 + luma outlier 제외`가 수용될지. Codex가 "R2 원문대로 0.65"로 돌아올 수도.
- **6.3 중립 상수**: 모델별 취향 차이 있을 수 있음. 최소 5분 논의 필요.
- **6.5 EMA 적용**: Gemini가 "매 프레임 튀면 안 됨"으로 찬성 예상. Codex는 "W1 스무딩과 중복" 우려 가능.

### 7.4 Claude의 W1 브레인스토밍 참여 자세

- **Codex 편향 경계** (메모리 `multi-ai-orchestration-bias`): Codex R2 원문이 "1샘플"로 잘못 쓰인 과거 있음. 이번엔 정확한 수식 제시할지 재확인.
- **Gemini "정성 체감" 원칙** (메모리 `qualitative-device-judgment`): 6.3 중립 상수 "0.35 vs 0.5" 같은 숫자 결정은 실기기 확인 가능한 범위로 한정.
- **사용자 원칙 "refactor-vs-retune"**: 측정 경로를 완전 새로 만들기보다 기존 텍스처 fetch 인프라 재활용.

### 7.5 1시간 브레인스토밍 예상 진행

```
0~5분   Codex/Gemini에 프롬프트 송신
5~25분  응답 대기 (백그라운드 while)
25~40분 응답 파싱, 7개 쟁점 표 작성
40~55분 Claude 의견 vs 상대 모델 충돌 지점 재질의
55~60분 합의 + 미결 정리
```

미결 있으면 R2 한 번 더. 없으면 구현 착수.

### 7.6 구현 예상 소요 (브레인스토밍 이후)

- EyeRenderPacket 구조체 + 어댑터: 1h
- avg_iris_luma self-measure 구현 + uniform 주입: 1.5h
- Fallback 체인 + hold 로직: 1h
- 단위 테스트: 30min
- 실기기 1회 확인 + 디버깅: 1h

**총 5h** (99 §3 S2 "4~5h"와 일치).

---

## 8. 완료 정의 + 다음 W 트리거

### 8.1 완료 정의 (Definition of Done)

§4.1 체크리스트 전체 ✅:
- [ ] eye_render_packet.h 생성
- [ ] adapter 구현
- [ ] gpu_lens_renderer API 변경
- [ ] avg_iris_luma 측정 복구 (0.35 하드코드 제거 실질 반영)
- [ ] Fallback 4단계
- [ ] 공개 C API 불변
- [ ] C++ 빌드 통과
- [ ] 단위 테스트 1개 이상
- [ ] 실기기 1회 확인

### 8.2 커밋 전략

**커밋 1**: `docs(P6-W1): 섹션 2~8 본문 작성 — 세부 계획 상세화` (지금 이 수정)
**커밋 2**: `refactor(gpu-lens): P6-W1 EyeRenderPacket 도입 + 어댑터 레이어`
**커밋 3**: `feat(gpu-lens): P6-W1 avg_iris_luma masked ROI 평균 self-measure`
**커밋 4** (선택): `test(gpu-lens): P6-W1 EyeRenderPacket 단위 테스트`

W별 멀티 커밋 권장 — 벤치 결과 역추적 용이.

### 8.3 다음 W 트리거

**P6-W2 (블렌드 3종 확정) 시작 조건**:
- W1 완료 (특히 avg_iris_luma 측정 동작)
- 실기기 LTL 모드 렌더링이 S1 "과도 tint" 상태에서 정상 복귀 확인

**P6-W3 (환경 반사 계층) 시작 조건**:
- W2 완료 (블렌드 3종 안정)
- EyeRenderPacket.reflection_dir / head_pose_yaw_roll 스키마 확정 (W1에서 끝나 있음)

### 8.4 W1 실패 시 롤백 전략

- W1 커밋 revert
- S1 커밋(`9aee86d`)까지 돌아감
- 재시도 or Codex/Gemini에 W1 브레인스토밍 재라운드

### 8.5 Phase 6 전체에서의 W1 위치

```
[과거]
  70633ac W3-03 (base)
    │
  d09bf72, d86598f, 9aee86d  S1 롤백
    │
[현재]
  5d977cd  P6 W0~W9 인사이트 작성
    │
[W1 완료 시점]
  + P6-W1 본문 세부
  + P6-W1 구현 커밋들
    │
[이후]
  P6-W2 → W3 → W4~W7 → W8(조건부) → W9
```

---

## 참조

- 99_final_decision.md §1.2 (C8, C9, C11)
- 99_final_decision.md §1.3 EyeRenderPacket 스키마
- 08_codex_r2.md — C9 "masked ROI 평균" 수식 원문
- 13_codex_r3.md §1 "1샘플 오류" 지적
- 07_claude_r2.md — Claude I7 "측정" 입장 변경 히스토리
- S1 커밋 `9aee86d` — uAvgIrisLum 0.35f 제거 주석
