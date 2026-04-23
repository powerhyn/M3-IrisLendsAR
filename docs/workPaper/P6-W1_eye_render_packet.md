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
_TODO: Step B에서 작성. 99 §1.3, §1.2 C8/C9 인용 + 이번 W의 자리매김_

## 3. 전제 조건
_TODO: P6-W0 읽음, S1 롤백 완료(9aee86d 확인)_

## 4. 목표
_TODO_

## 5. 99에서 확정된 사항
_TODO: §1.4 스키마를 §5로 이동하며 상세화_

## 6. 미결 사항
_TODO: §1.9 질문 7개를 각 항목별 배경과 함께 정리_

## 7. W 브레인스토밍 시작 체크리스트
_TODO: 다음 세션 송신용 프롬프트 초안_

## 8. 완료 정의 + 다음 W 트리거
_TODO_

---

## 참조

- 99_final_decision.md §1.2 (C8, C9, C11)
- 99_final_decision.md §1.3 EyeRenderPacket 스키마
- 08_codex_r2.md — C9 "masked ROI 평균" 수식 원문
- 13_codex_r3.md §1 "1샘플 오류" 지적
- 07_claude_r2.md — Claude I7 "측정" 입장 변경 히스토리
- S1 커밋 `9aee86d` — uAvgIrisLum 0.35f 제거 주석
