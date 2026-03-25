# 뷰티 마스크 정밀도 개선 구현 계획

## Context

프로토타이핑(`feature/beauty-smoothing-prototype`)에서 매끈하게/모공 2축 분리를 구현했으나, B612 대비 **"뿌연 느낌(haze)"**이 발생. 원인은 비피부 영역(코 윤곽, 안경테, 헤어라인)에 스무딩이 누수되는 것으로, 3자 토론(Claude/Codex/Gemini)에서 **마스크 정밀도가 1순위 병목**으로 합의됨.

토론 문서: `docs/demo_app/beauty_mask_refinement_discussion.md`

### 핵심 설계 원칙 (Codex 피드백 반영)

1. **`combined_mask`는 단일 채널(CV_8UC1) legacy contract 유지** — CPU 경로, 분석 유틸이 기대하는 의미를 깨뜨리지 않음
2. **dual mask는 GPU 전용 별도 버퍼** (`packed_dual_mask`) — `combined_mask`와 별도로 BeautyROI에 저장, GPU 업로드 직전에 생성
3. **`protectNose` 기본값 `false`** — 레거시 경로에 영향 없음, 실기기 검증 후 켜기

---

## 실행 계획

### Phase A: 코 보호 + 정규화 erode

#### A-1. 콧구멍/콧볼 보호 마스크 추가

**`cpp/include/iris_sdk/beauty_roi_manager.h`**
- 상수 추가:
  ```
  NOSE_CAVITY_COUNT = 5
  NOSE_CAVITY_INDICES[5]  // {1, 2, 4, 98, 327} — 콧구멍 내부
  NOSE_WING_COUNT = 6
  NOSE_WING_INDICES[6]    // {218, 417, 419, 438, 446, 248} — 콧볼 외곽
  ```
- `BeautyROI` 구조체에 `nose_protect_mask` 벡터 추가
- `createNoseProtectionMask()` static 메서드 선언
- `combineMasks()` 시그니처에 `nose_protect_mask` 파라미터 추가

**`cpp/include/iris_sdk/beauty_filter.h`**
- `BeautyFilterConfigV2`에 `bool protectNose` 추가 — **구조체 맨 끝** (vivid 뒤), 기본값 **`false`**
- Helper의 defaults/clamp/isValid/fromV1에 반영

**`cpp/src/beauty_roi_manager.cpp`**
- `NOSE_CAVITY_INDICES`, `NOSE_WING_INDICES` 정의
- `createNoseProtectionMask()` 구현:
  - NOSE_CAVITY로 fillPoly → NOSE_WING 좌표로 convexHull 추가 → dilate(5×5 타원)
  - **코 전체가 아닌 콧구멍/콧볼 경계만** 보호 (콧등 미포함)
- `computeROI()`에서 `config.protectNose`이면 호출
- `combineMasks()` 수정: `combined = skin × (1-eye) × (1-eyebrow) × (1-lip) × (1-nose)`

#### A-2. 정규화 erode

**`cpp/src/beauty_roi_manager.cpp`** — `computeROI()` 내부

combineMasks() 이후, applyFeathering() 이전에 삽입 — **`protectNose==true` 또는 2축 모드일 때만** 적용 (레거시 경로 보호):
```
if (config.protectNose || config.smoothIntensity > 0 || config.poreReduction > 0) {
    min_dim = min(mask_width, mask_height)
    erode_size = clamp(round(min_dim * 0.02), 1, 5) | 1  // 홀수 보장, 상한 5px
    cv::erode(combined, combined, MORPH_ELLIPSE(erode_size))
}
```

- 기준: `min(mask_width, mask_height)` (종횡비 무관 안전)
- 비율: 0.02 (기존 축소 + 누적 고려하여 보수적)
- 상한: 5px (256px 마스크에서도 과도한 축소 방지)
- **레거시 skinQuality 경로에서는 erode 미적용** → combined_mask 완전 동일

#### A-3. Android 바인딩

**`cpp/include/iris_sdk/sdk_api.h`** — `IrisBeautyConfigV2`에 `int protect_nose` 추가 (구조체 **맨 끝**, vivid_warmth 뒤에 append)
**`cpp/src/sdk_api_v2.cpp`** — toCppConfigV2/fromCppConfigV2에 매핑, default에 0 설정
**`android/.../BeautyFilterConfigV2.java`** — `protectNose` 필드(기본 false) + Builder
**`android/.../jni_utils.h`** — fieldID 추가
**`android/.../iris_jni.cpp`** — fieldID 초기화 + 복사
**`android/.../BeautyPresetFactory.kt`** — baseBuilder에 `.protectNose(false)` (데모앱에서 수동으로 켜서 테스트)

#### A 검증

**자동 테스트:**
- config default round-trip 테스트: `protectNose` 기본값 false 확인
- `protectNose=true`/`false`에 따른 ROI mask 생성 단위 테스트 (combined_mask 크기, 값 범위)
- legacy `skinQuality=0.3` + `protectNose=false`에서 기존과 동일한 combined_mask 출력

**수동 테스트:**
- 디버그 Mask 모드로 코 보호 영역 시각적 확인
- `protectNose=true` 상태에서 매끈하게 100: 코 윤곽 haze 개선 비교 (수정 전/후 스크린샷)
- `protectNose=false` 상태에서 잡티보정 0.3: 기존 품질 동일 확인

---

### Phase B: smooth_mask / pore_mask 분리 (유력 실험안)

> Phase A 테스트 후, 추가 개선이 필요할 때 진행

#### B-1. CPU dual mask 생성

**`cpp/include/iris_sdk/beauty_roi_manager.h`**
- `BeautyROI`에 추가:
  - `std::vector<uint8_t> packed_dual_mask` — GPU 전용 RG 패킹 버퍼 (2 bytes/pixel)
  - `bool has_dual_mask = false` — dual mask 유효 여부 플래그
- **`combined_mask`는 변경하지 않음** — 단일 채널 legacy contract 유지

**`cpp/src/beauty_roi_manager.cpp`**
- `computeROI()` 확장 — dual mask 생성 (smoothIntensity/poreReduction 2축 모드일 때만):
  ```
  smooth_mask = erode(combined, clamp(min_dim * 0.03, 1, 7))  // 보수적
  pore_mask   = combined  // 또는 light erode(min_dim * 0.01)
  packed_dual_mask = interleave(smooth_mask, pore_mask)  // R=smooth, G=pore
  has_dual_mask = true
  ```
- smooth_mask/pore_mask는 **임시 지역 변수**, packed_dual_mask만 BeautyROI에 저장

#### B-2. GPU GL_RG8 전환

**`cpp/include/iris_sdk/gpu/gpu_beauty_backend.h`**
- `uploadDualSkinMask()` 메서드 추가 (기존 `uploadSkinMask()` 유지)

**`cpp/src/gpu/gpu_beauty_backend.cpp`**
- `uploadDualSkinMask()`: GL_RG8 포맷으로 업로드
- `applyTextureId()` 분기:
  - `has_dual_mask == true` → `uploadDualSkinMask(packed_dual_mask)` (GL_RG8)
  - `has_dual_mask == false` → **R=G=combined로 패킹하여 GL_RG8 업로드** (셰이더 코드 단일 경로 유지)
- **레거시 경로도 GL_RG8**: combined_mask를 R=G 동일값으로 패킹 → 셰이더에서 smoothMask==poreMask → 기존 `mask * max(poreBlend, floor)` 동작과 수학적으로 동일

#### B-3. 셰이더 분리

**`cpp/src/gpu/shader_sources.cpp`** — Composite 셰이더:
```glsl
vec2 maskRG = texture(uSkinMask, ...).rg;
float smoothMask = maskRG.r;
float poreMask = maskRG.g;
// 레거시(R=G=combined): smoothMask==poreMask → 기존 동작과 동일
// 2축 모드: smoothMask < poreMask (smooth가 더 보수적)

float poreEffect = poreMask * poreBlend;
float smoothEffect = smoothMask * smoothFloorProtected;
float textureBlend = max(poreEffect, smoothEffect);

// toneFinish: smoothMask 사용
vec3 result = mix(textureFinished, toneFinish, smoothMask * (0.65 * effectStrength));
```

**하위 호환 증명:** 레거시에서 R=G=m이면:
- `poreEffect = m * poreBlend`, `smoothEffect = m * smoothFloorProtected`
- `textureBlend = max(m*poreBlend, m*floor) = m * max(poreBlend, floor)` → 기존 식과 동일

**Sharpen 셰이더:**
- `max(maskRG.r, maskRG.g)` 사용 — pore-only 영역에서도 샤프닝 복구 보장
- 레거시(R=G=m): `max(m, m) = m` → 기존 동작 유지

#### B-4. ROI 캐시 무효화 + 새 필드 정리

**`cpp/include/iris_sdk/beauty_roi_manager.h`** — BeautyROI 초기화/무효화:
- 기존 `combined_mask.clear()` 경로에 `packed_dual_mask.clear()`, `has_dual_mask = false`도 같이 비우기
- 생성자/reset 메서드에서 새 필드 초기화 보장

**`cpp/src/beauty_processor.cpp`** (또는 `gpu_beauty_backend.cpp`의 ROI 캐시 로직):
- `protectNose` 토글, 2축 모드 전환 시 캐시된 ROI를 무효화
- `has_dual_mask` 상태 변경 감지 → `packed_dual_mask` 재생성
- 구현: config 변경 감지 해시 또는 이전 프레임 설정값과 비교

#### B-5. 디버그 모드 확장

Mask 히트맵(`uDebugMode==6`): `result = vec3(smoothMask, poreMask, 0.0)` — 빨강=smooth, 초록=pore

#### B 검증

**자동 테스트:**
- legacy `skinQuality=0.3` → `has_dual_mask=false` → R=G=combined 패킹, GL_RG8 업로드 확인
- 2축 모드 → `has_dual_mask=true` → GL_RG8 업로드 확인, 데이터 크기 = W*H*2
- `packed_dual_mask`의 R채널(smooth) 값이 G채널(pore) 값 이하인지 (smooth가 더 보수적)
- 레거시 R=G=combined에서 `max(m*poreBlend, m*floor) == m * max(poreBlend, floor)` 수학적 동일 확인
- `protectNose` 토글 시 ROI 캐시 무효화 → 다음 프레임에서 마스크 재생성 확인
- 2축↔레거시 모드 전환 시 `has_dual_mask` 상태 정확히 변경 확인

**수동 테스트:**
- 매끈하게만 100: smooth_mask(보수적)으로 haze 감소 확인
- 모공만 100: pore_mask(넓은)으로 효과 유지 확인
- 둘 다 100: 간섭 없이 독립 동작 확인
- 잡티보정(레거시): R=G=combined GL_RG8 경로, 기존과 동일 동작

---

## 수정 파일 요약

| 파일 | Phase | 변경 내용 |
|------|-------|----------|
| `cpp/include/iris_sdk/beauty_roi_manager.h` | A,B | 코 인덱스 상수, nose_protect_mask, packed_dual_mask, 메서드 |
| `cpp/src/beauty_roi_manager.cpp` | A,B | createNoseProtectionMask, erode, dual mask 패킹 |
| `cpp/include/iris_sdk/beauty_filter.h` | A | `protectNose` 필드 (**맨 끝**, 기본 false), fromV1/defaults/clamp/isValid |
| `cpp/include/iris_sdk/sdk_api.h` | A | `protect_nose` (구조체 **맨 끝** append) |
| `cpp/src/sdk_api_v2.cpp` | A | 매핑 함수 + default |
| `cpp/include/iris_sdk/gpu/gpu_beauty_backend.h` | B | uploadDualSkinMask() 추가 |
| `cpp/src/gpu/gpu_beauty_backend.cpp` | B | GL_RG8 업로드, has_dual_mask 분기 |
| `cpp/src/gpu/shader_sources.cpp` | B | .rg 분리, Sharpen max(r,g) |
| `cpp/src/beauty_processor.cpp` | B | ROI 캐시 무효화 (config 변경 감지) |
| `android/.../BeautyFilterConfigV2.java` | A | protectNose 필드(기본 false) |
| `android/.../jni_utils.h` | A | fieldID |
| `android/.../iris_jni.cpp` | A | 초기화+복사 |
| `android/.../BeautyPresetFactory.kt` | A | protectNose=false |
