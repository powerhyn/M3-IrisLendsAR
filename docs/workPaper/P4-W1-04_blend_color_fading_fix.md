# P4-W1-04: 블렌드 모드 색 희석 수정 (ISS-005)

## 작업 개요
- **Work Paper**: P4-W1-04
- **Phase**: P4 (시각적 리얼리즘 — Visual Fidelity)
- **기간**: 2026-02-24 ~
- **상태**: 🔄 진행 중
- **선행 조건**: P4-W1-03 완료
- **근거**: ISS-005 브레인스토밍 Sec 21 합의 (A~D절) + E/F절 최종 합의
- **브랜치**: `feature/P3-W1-03` (현재)

## 문제 요약

렌즈 색상이 원본 대비 크게 바래 보이는 현상. 2개의 **독립적 원인**이 동시 작용:

| 원인 | 기여도 | 근거 등급 |
|------|:------:|:---------:|
| Premultiplied Alpha 이중 곱셈 | ~50% | `[Measured + Calculated]` |
| 절대 밝기 곱셈 (어두운 홍채) | ~33% | `[Calculated]` |
| sRGB/Linear 축 불일치 (Mode 5) | ~10% | `[Theoretical]` |

> 기여도는 ~2x 복원 예상 `[Calculated, EXP-A로 확정 예정]` (E/F절 합의)

## 실행 순서 (ISS-005 D절)

| 순위 | 작업 | Phase | 합의 | 커밋 분리 |
|:----:|------|:-----:|:----:|:---------:|
| 1 | `inPremultiplied = false` | EXP-A | A-1 완전 합의 | 단독 커밋 |
| 2 | Color Replace 블렌드 모드 | EXP-B | A-2 완전 합의 | 단독 커밋 |
| 3 | Mode 5 색공간 정합 | EXP-C | B-1 조건부 | 단독 커밋 |
| 4 | 8-bit Banding 대응 | EXP-C | B-3 조건부 | 필요 시 |
| - | Specular Layer 고도화 | - | C-5 보류 | 별도 이슈 |
| - | Shader Fallback | - | B-4 조건부 | 별도 이슈 |

**중단 규칙(A-7)**: 동일 축 2회 연속 Color Lift(C*) 개선폭 <5% → 해당 축 중단

---

## Phase 1: EXP-A — Alpha Fix (우선순위 1)

### 목적
`BitmapFactory` 기본값 `inPremultiplied = true`로 인해 렌즈 텍스처 RGB가 alpha와 사전 곱셈된 상태에서 셰이더가 다시 alpha를 곱하는 이중 감쇄를 수정한다.

### 원인 상세

```
현재 경로:
  BitmapFactory (inPremultiplied=true)
    → lens.rgb = originalRGB × alpha        ← 1차 곱셈 (디코더)
    → GLUtils.texImage2D(bitmap)
    → 셰이더: mix(base, lens.rgb, finalAlpha)
             = base×(1-α) + (originalRGB×α)×α  ← 2차 곱셈 (셰이더)
    → partial alpha(avg 0.50) 영역에서 색상 ~50% 손실

수정 후:
  BitmapFactory (inPremultiplied=false)
    → lens.rgb = originalRGB                 ← 원본 유지
    → GLUtils.texImage2D(bitmap)
    → 셰이더: mix(base, lens.rgb, finalAlpha)
             = base×(1-α) + originalRGB×α    ← 1회만 곱셈
    → 렌즈 디자이너 의도대로 색상 발현
```

### 수정 파일 (1개, 1줄)

**`android/demo-app/src/main/java/com/irislenssdk/demo/lens/LensManager.kt`**

| 라인 | 현재 | 수정 |
|:----:|------|------|
| 168 뒤 | (없음) | `options.inPremultiplied = false` 추가 |

```kotlin
// LensManager.kt:loadTexture() (line 166~168)
options.inSampleSize = calculateInSampleSize(options, TEXTURE_SIZE, TEXTURE_SIZE)
options.inJustDecodeBounds = false
options.inPreferredConfig = Bitmap.Config.ARGB_8888
options.inPremultiplied = false  // ← ISS-005 A-1: Straight Alpha 디코딩
```

### 영향 경로

```
LensManager.loadTexture() (line 156)
  → BitmapFactory.decodeStream() with inPremultiplied=false (line 171)
  → Bitmap (Straight Alpha)
  → textureCache (LRU, line 140)
  → GpuRenderActivity.onLensSelected() (line 303)
  → CameraGLView.setLensTexture() (line 253, queueEvent)
  → CameraGLRenderer.setLensTexture() (line 1225, pendingLensBitmap)
  → CameraGLRenderer.uploadPendingLensTexture() (line 665)
  → GLUtils.texImage2D(GL_TEXTURE_2D, 0, bitmap, 0) (line 685)
  → GLSL: vec4 lens = texture(uLensTexture, lensCoord) (line 262)
```

`GLUtils.texImage2D`는 Straight Alpha Bitmap을 정상 처리한다.

### Mode 0~6 영향도

| 모드 | 변화 방향 | 이유 |
|:----:|:---------:|------|
| Mode 0 (Normal) | 색상 향상 | `lens.rgb` 원본값 복원 |
| Mode 1 (Multiply) | 효과 강화 | `base * lens.rgb` → lens.rgb 밝아짐 |
| Mode 2 (Screen) | 효과 강화 | screen 계산에 lens.rgb 반영 |
| Mode 3 (Overlay) | 효과 강화 | overlay 계산에 lens.rgb 반영 |
| Mode 4-5 (LumTint) | 색상 향상 | `blend * lum * scale` → blend 원본 복원 |
| Mode 6 (SoftLight) | 효과 강화 | softlight 계산에 blend 반영 |

**전 모드에서 regression이 아닌 개선 방향.**

### EXP-A 검증 체크리스트

- [ ] Before 스크린샷 촬영 (Mode 0, 4 최소, 동일 렌즈/조명/opacity)
- [ ] `inPremultiplied = false` 1줄 추가
- [ ] After 스크린샷 촬영 (완전 동일 조건)
- [ ] Color Lift: 홍채 ROI HSV Saturation 비교
- [ ] **경계 영역 before/after 비교** (dark halo 제거 확인, E/F절 합의)
- [ ] Mode 0~3 regression 없음 확인
- [ ] 동공 영역 무변화 확인 (alpha=0)
- [ ] Pass 기준: 가시적 색상 개선 + 경계 아티팩트 없음

### 커밋
```
fix(blend): inPremultiplied=false로 Alpha 이중 곱셈 수정 [ISS-005 EXP-A]
```

---

## Phase 2: EXP-B — Color Replace 블렌드 모드 (우선순위 2)

### 목적
현재 Luminance Tint(Mode 4/5)의 `blend * lum * scale` 공식은 **절대 밝기 곱셈**이다. 어두운 홍채(한국인 평균 uAvgIrisLum ≈ 0.15)에서 렌즈 색이 33%만 발현된다.

새 블렌드 모드 **Color Replace(Mode 7)**는 `lum / uAvgIrisLum` **상대 밝기 정규화**로 이 문제를 해결한다.

### 공식

```glsl
// Mode 7: Color Replace
vec3 blendColorReplace(vec3 base, vec3 blend, float opacity) {
    float lum = dot(base, vec3(0.299, 0.587, 0.114));  // sRGB luminance
    float detail = lum / max(0.01, uAvgIrisLum);        // 상대 밝기
    detail = clamp(detail, 0.2, 2.5);                    // 범위 제한
    vec3 colored = blend * detail;                        // 색상 × 상대밝기
    return mix(base, colored, opacity);
}
```

**동작 원리**:
- `detail = 1.0` (평균 밝기 픽셀) → 렌즈 색 100% 발현
- `detail < 1.0` (어두운 결/groove) → 자연스럽게 어두움 → 질감 보존
- `detail > 1.0` (밝은 결/ridge) → 밝은 포인트 유지 → 입체감 보존
- `clamp(0.2, 2.5)` → 하한: 과도한 어둠 방지, 상한: 하이라이트 클리핑 방지 (Muse 권고)

### 수정 파일 (7개, 13개소)

#### 1. C++ Core Enum — `cpp/include/iris_sdk/types.h` (line 31)

```cpp
SoftLight = 6,          ///< 소프트 라이트 블렌딩 @experimental
ColorReplace = 7        ///< 색상 교체 블렌딩 (상대 밝기 정규화) @experimental
```

#### 2. C API Enum — `cpp/include/iris_sdk/sdk_api.h` (line 114)

```c
IRIS_BLEND_SOFT_LIGHT = 6,          /**< 소프트 라이트 블렌딩 @experimental */
IRIS_BLEND_COLOR_REPLACE = 7        /**< 색상 교체 블렌딩 @experimental */
```

#### 3. C++ 변환 — `cpp/src/sdk_api.cpp` (line 143 뒤)

```cpp
case IRIS_BLEND_COLOR_REPLACE:
    return iris_sdk::BlendMode::ColorReplace;
```

#### 4. CPU 렌더러 폴백 — `cpp/src/lens_renderer.cpp` (line 520)

```cpp
case BlendMode::LuminanceTint:
case BlendMode::LuminanceTintLinear:
case BlendMode::SoftLight:
case BlendMode::ColorReplace:   // ← 추가: GPU 전용, CPU에서는 Normal 폴백
    [[fallthrough]];
```

#### 5. JNI 범위 검증 — `android/iris-sdk/src/main/cpp/iris_jni.cpp` (line 358)

```cpp
// 변경: IRIS_BLEND_SOFT_LIGHT → IRIS_BLEND_COLOR_REPLACE
if (rawBlendMode < IRIS_BLEND_NORMAL || rawBlendMode > IRIS_BLEND_COLOR_REPLACE) {
```

#### 6. C++ 테스트 — `cpp/tests/test_sdk_api.cpp` (line 437 뒤)

```cpp
EXPECT_EQ(7, IRIS_BLEND_COLOR_REPLACE);
```

#### 7. Java SDK — `android/iris-sdk/src/main/java/com/irislenssdk/LensConfig.java`

| 위치 | 변경 |
|:----:|------|
| line 88 뒤 | `public static final int BLEND_COLOR_REPLACE = 7;` |
| line 203 | `isValid()`: `<= BLEND_SOFT_LIGHT` → `<= BLEND_COLOR_REPLACE` |
| line 216 | `clamp()`: `Math.min(BLEND_SOFT_LIGHT, ...)` → `Math.min(BLEND_COLOR_REPLACE, ...)` |
| line 255 뒤 | `getBlendModeName()`: `case BLEND_COLOR_REPLACE: return "COLOR_REPLACE";` |

#### 8. Kotlin Enum — `android/iris-sdk/src/main/java/com/irislenssdk/kotlin/BlendMode.kt`

| 위치 | 변경 |
|:----:|------|
| line 88 | `SOFT_LIGHT` 세미콜론 → 콤마, `COLOR_REPLACE(JavaLensConfig.BLEND_COLOR_REPLACE);` 추가 |

#### 9. GLSL 셰이더 + Dispatch — `android/demo-app/.../camera/gpu/CameraGLRenderer.kt`

| 위치 | 변경 |
|:----:|------|
| line 171 | 주석: `0-6` → `0-7` |
| line ~242 뒤 | `blendColorReplace()` GLSL 함수 추가 |
| line 296-298 | `else { softLight }` → `else if (==6) { softLight } else { colorReplace }` |

#### 10. Demo App UI — `android/demo-app/.../GpuRenderActivity.kt`

| 위치 | 변경 |
|:----:|------|
| line 353-356 | `blendModes` 배열 끝에 `"Color Replace"` 추가 |

### 수정 총괄표 (Phase 2)

| # | 파일 | 레이어 | 수정 내용 |
|---|------|:------:|-----------|
| 1 | `cpp/include/iris_sdk/types.h` | C++ Core | `ColorReplace = 7` 추가 |
| 2 | `cpp/include/iris_sdk/sdk_api.h` | C API | `IRIS_BLEND_COLOR_REPLACE = 7` 추가 |
| 3 | `cpp/src/sdk_api.cpp` | C++ 변환 | switch case 추가 |
| 4 | `cpp/src/lens_renderer.cpp` | CPU 렌더러 | fallthrough에 ColorReplace 추가 |
| 5 | `cpp/tests/test_sdk_api.cpp` | C++ 테스트 | enum 값 assertion 추가 |
| 6 | `android/iris-sdk/src/main/cpp/iris_jni.cpp` | JNI | 범위 검증 상한 갱신 |
| 7 | `android/iris-sdk/.../LensConfig.java` | Java SDK | 상수 + 3 메서드 범위 갱신 |
| 8 | `android/iris-sdk/.../kotlin/BlendMode.kt` | Kotlin | enum entry 추가 |
| 9 | `android/demo-app/.../CameraGLRenderer.kt` | GLSL | 셰이더 함수 + dispatch |
| 10 | `android/demo-app/.../GpuRenderActivity.kt` | Demo UI | Spinner 배열 |

### EXP-B 검증 체크리스트

전제: EXP-A(Alpha Fix) 적용 상태

- [ ] Mode 7 vs Mode 0 vs Mode 4 스크린샷 비교
- [ ] 어두운 렌즈(LAGUNA BEIGE) + 밝은 렌즈 각각 테스트
- [ ] Color Lift: Mode 7이 Mode 0/4 대비 홍채 ROI Saturation 향상 확인
- [ ] 홍채 질감(결/패턴) 보존 확인
- [ ] clamp(0.2, 2.5) 범위: 하이라이트 영역 클리핑 없는지
- [ ] 중단 규칙 판정: 이전 Phase 대비 <5% 개선이 2회 연속인지
- [ ] C++ 테스트 통과: `ctest` (test_sdk_api BlendModeEnumValues)

### 커밋
```
feat(blend): Color Replace 블렌드 모드 추가 [ISS-005 EXP-B]
```

---

## Phase 3: EXP-C — Mode 5 색공간 정합 (우선순위 3)

### 목적
Mode 5(`blendLuminanceTintLinear`)에서 `uAvgIrisLum`은 sRGB Y채널(CPU NV21)이지만, 셰이더 내 `lum`은 linear 공간에서 계산된다. 이 축 불일치를 수정한다.

### 수정 파일 (1개)

**`android/demo-app/.../camera/gpu/CameraGLRenderer.kt`** — `blendLuminanceTintLinear()` (line 225-234)

```glsl
// 변경 전 (line 228):
float scale = clamp(0.5 / max(0.1, uAvgIrisLum), 0.8, 2.5);

// 변경 후:
float avgLumLinear = uAvgIrisLum * uAvgIrisLum;  // ISS-005 B-1: sRGB→linear 근사
float scale = clamp(0.5 / max(0.1, avgLumLinear), 0.8, 2.5);
```

### 주의사항
- `avgLumLinear` = 0.15² = 0.0225 → `0.5 / 0.0225` = 22.2 → clamp(2.5) → 결과 동일할 수 있음
- clamp 상한이 sRGB 기준으로 설계되었으므로, linear에서는 **상한 확대**(예: 5.0~8.0) 필요 가능
- 정확한 값은 EXP-C 시각 비교로 결정

### 8-bit Banding 대응 (B-3)

색공간 정합 후 banding 시각 확인:
- 심하지 않으면 → 추가 조치 불필요
- 심하면 → 옵션 평가:
  - (a) `GL_RGBA16F` FBO (메모리 2배, GPU 부하 증가)
  - (b) Mode 5를 감마 공간 유지 + 보정 (의미 변경)
  - (c) 디더링 (최소 비용, 노이즈)

### Mode 5 존치 판정 (B-2)

Mode 4 vs Mode 5(수정 후) 비교:
- 품질 우위 → 존치
- 우위 없음 → Spinner에서 비노출 (코드 유지, UI에서 숨김)

### EXP-C 검증 체크리스트

전제: EXP-A + EXP-B 적용 상태

- [ ] Mode 5 색공간 정합 적용
- [ ] Mode 4 vs Mode 5 스크린샷 비교
- [ ] 어두운 홍채 영역 banding 아티팩트 확인
- [ ] Mode 5 존치/비노출 판정
- [ ] 중단 규칙 판정

### 커밋
```
fix(blend): Mode 5 uAvgIrisLum sRGB→Linear 색공간 정합 [ISS-005 EXP-C]
```

---

## Phase 4: 후순위 (ISS-005 범위 밖)

| 작업 | 상태 | 판정 시점 |
|------|:----:|:---------:|
| Specular Layer 고도화 (C-5) | 보류 → 별도 이슈 | ISS-005 완료 후 |
| Shader Fallback (B-4) | 조건부 → 범위 미확정 | ISS-005 완료 후 |

---

## Git 전략

```
feature/P3-W1-03
  ├── commit: fix(blend): inPremultiplied=false [EXP-A]
  │   └── LensManager.kt 1줄
  │   → EXP-A 검증
  │
  ├── commit: feat(blend): Color Replace [EXP-B]
  │   └── 10개 파일 (Core→API→JNI→Java→Kotlin→GLSL→UI)
  │   → EXP-B 검증
  │
  └── commit: fix(blend): Mode 5 색공간 정합 [EXP-C]
      └── CameraGLRenderer.kt ~3줄
      → EXP-C 검증 + Mode 5 존치 판정
```

각 커밋 분리로 개별 revert 가능.

---

## 위험 요소 및 대응

| 위험 | 가능성 | 대응 |
|------|:------:|------|
| GLUtils.texImage2D non-premultiplied Bitmap 예상 외 동작 | 낮음 | Android 문서상 정상. 실패 시 셰이더 unpremultiply 폴백 |
| Color Replace clamp(0.2, 2.5) 범위 부적절 | 중간 | EXP-B에서 튜닝. 하한 0.1~0.3, 상한 2.0~3.5 범위로 조정 |
| Mode 5 색공간 정합 후 scale clamp 도달로 효과 없음 | 높음 | clamp 상한 확대 또는 scale 공식 재설계 |
| Mode 5 banding 심각 | 중간 | 디더링 추가 → 감마 공간 유지 → GL_RGBA16F 순서로 시도 |
| 기존 Mode 0~3 시각적 변화 | 확실 | 방향이 "개선"이므로 regression 아님. 스크린샷 확인 |

---

## 검증 방법

1. **빌드**: `cd android && ./gradlew :demo-app:assembleDebug`
2. **C++ 테스트**: `cd cpp/cmake-build-debug && cmake --build . && ctest`
3. **실기기 테스트**: 데모 앱 → 렌즈 선택 → 블렌드 모드 전환 → 스크린샷
4. **Color Lift**: 홍채 ROI HSV Saturation 추출 (외부 이미지 분석 도구)
5. **Regression**: Mode 0~3 before/after 비교

---

## 실행 기록

### EXP-A 결과
- **상태**: ✅ 완료 (v2)
- **v1**: `inPremultiplied = false` → `createScaledBitmap(Canvas)` 호환 불가로 렌즈 미표시
- **v2**: 셰이더에서 `lens.rgb /= lens.a` unpremultiply로 대체. 동일 효과 + Canvas 호환
- **실기기 검증**: ✅ 렌즈 정상 표시, 색상 개선 확인

### EXP-B 결과
- **상태**: ✅ 완료
- **수정 파일**: 11개 (Core→C API→JNI→Java→Kotlin→GLSL→UI 전 레이어)
- **피드백 반영**:
  - GLSL invalid fallback → Normal로 변경 (안전성 강화)
  - test_types.cpp BlendMode 테스트 커버리지 0~7 확장
- **실기기 검증**: ✅ Color Replace 시각적 품질 확인 — "느낌이 괜찮다"

### EXP-C 결과
- **상태**: ✅ 완료 (코드 적용 + 존치 판정)
- **수정**: `avgLumLinear = uAvgIrisLum²`, `max(0.01, ...)`, `clamp(_, 0.8, 5.0)`
- **Mode 4 vs Mode 5 비교**: 시각적 차이 없음
- **존치 판정 (B-2)**: Mode 5 Spinner 비노출, 코드 유지. Mode 4 존치.
- **최종 Spinner**: Normal, Multiply, Screen, Overlay, Luminance Tint, Soft Light, Color Replace

---

## 변경 이력

| 날짜 | 내용 |
|------|------|
| 2026-02-24 | 작업 문서 생성 (ISS-005 Sec 21 합의 + E/F절 합의 기반) |
| 2026-02-24 | EXP-A/B/C 코드 적용 완료. Android 빌드 + C++ 테스트 통과. |
| 2026-02-24 | EXP-A v2: 셰이더 unpremultiply로 대체 (Canvas 호환 이슈). |
| 2026-02-24 | 피드백 반영: GLSL fallback 안전성, test_types 커버리지, Mode 5 clamp 조정. |
| 2026-02-24 | 실기기 검증 완료. Mode 5 비노출 판정. **ISS-005 마감**. |
