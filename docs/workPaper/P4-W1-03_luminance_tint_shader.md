# P4-W1-03: Luminance Tint 셰이더 + uAvgIrisLum + Demo App UI

## 작업 개요
- **Phase**: P4 (시각적 리얼리즘 — Visual Fidelity)
- **기간**: 2026-02-13 ~
- **상태**: ⏳ 대기
- **선행 조건**: P4-W1-01 완료 + P4-W1-02 Gate 1 PASS
- **근거**: 브레인스토밍 Section 8, 10, 11, 13, 16, 17, 19, 22, 26 합의

## 목표

1. Fragment Shader에 **3개 블렌드 함수** (LuminanceTint, LuminanceTintLinear, SoftLight) 구현
2. **uAvgIrisLum** uniform으로 Adaptive Scale 적용 (CPU EMA + Hold + Rate Limit)
3. Demo App UI를 **RadioGroup → Spinner**로 전환하여 7개 블렌드 모드 지원
4. **onResume()** 시 timestamp 리셋으로 resume jump 방지

## 핵심 개념: Luminance-Preserving Color Tint

```
실제 콘택트렌즈의 물리적 원리:
- 반투명 착색 필름이 각막 위에 위치
- 원본 홍채의 luminance(밝기 패턴 = 줄무늬, 깊이, 명암)가 렌즈를 통해 투과
- 각막 표면의 specular highlight(반사광)은 렌즈 위에 그대로 존재

→ 실제 렌즈는 "색상(hue/saturation)만 교체하고 밝기(luminance)는 보존"
```

## 수정 대상 파일 (4개, 단일 PR)

| # | 파일 | 수정 내용 |
|---|------|-----------|
| 1 | `CameraGLRenderer.kt` | GLSL blend 함수 3개 + uBlendMode 분기 + uAvgIrisLum uniform + EMA 로직 |
| 2 | `GpuRenderActivity.kt` | RadioGroup → Spinner + when 분기 7개 + onResume() timestamp 리셋 |
| 3 | `activity_gpu_render.xml` | RadioButtons → Spinner 위젯 |
| 4 | `CameraGLRenderer.kt` (동일 파일) | GPU tier 기반 feature flag 분기 |

## 상세 구현 사항

### 1. GLSL 블렌드 함수

#### Mode 4: blendLuminanceTint (sRGB 근사)

```glsl
vec3 blendLuminanceTint(vec3 base, vec3 blend, float opacity) {
    float lum = dot(base, vec3(0.299, 0.587, 0.114));
    float scale = clamp(0.5 / max(0.1, uAvgIrisLum), 0.8, 2.5);
    vec3 tinted = blend * lum * scale;
    return mix(base, tinted, opacity);
}
```

- Specular 처리 없음 (가장 단순, 가장 먼저 실험)
- sRGB 감마 공간에서 직접 계산 → 저사양 GPU 친화

#### Mode 5: blendLuminanceTintLinear (Fast Linear)

```glsl
vec3 toLinearFast(vec3 srgb) { return srgb * srgb; }
vec3 toSRGBFast(vec3 linear) { return sqrt(max(linear, vec3(0.0))); }

vec3 blendLuminanceTintLinear(vec3 base, vec3 blend, float opacity) {
    vec3 baseL = toLinearFast(base);
    float lum = dot(baseL, vec3(0.2126, 0.7152, 0.0722)); // BT.709
    float scale = clamp(0.5 / max(0.1, uAvgIrisLum), 0.8, 2.5);
    vec3 tinted = toLinearFast(blend) * lum * scale;
    vec3 result = mix(baseL, tinted, opacity);

    // Real Specular 복원 (원본 밝은 영역 보존)
    float realSpec = smoothstep(0.7, 0.95, lum);
    result = mix(result, baseL, realSpec);

    return toSRGBFast(result);
}
```

- Fast Linearization (`x*x` / `sqrt(x)`) — `pow(2.2)` 대비 ~3-5배 빠름
- `sqrt(max(linear, vec3(0.0)))` — 음수 보호 (브레인스토밍 Section 16 합의)
- Real Specular 복원만 (Fake Specular 없음)

#### Mode 6: blendSoftLight (Photoshop 공식)

```glsl
vec3 blendSoftLight(vec3 base, vec3 blend, float opacity) {
    vec3 result;
    result = (blend <= vec3(0.5))
        ? base - (1.0 - 2.0 * blend) * base * (1.0 - base)
        : base + (2.0 * blend - 1.0) * (sqrt(base) - base);
    return mix(base, result, opacity);
}
```

- Specular 처리 없음
- 부드러운 조명 효과, 은은한 색상 변화

### 2. Shader if/else 분기 확장

```glsl
// 기존 (mode 0-3) — base=카메라RGB, blend=렌즈RGB, opacity=렌즈alpha
if (uBlendMode == 0) { blended = blendNormal(cameraColor, lensColor.rgb, lensAlpha); }
else if (uBlendMode == 1) { blended = blendMultiply(cameraColor, lensColor.rgb, lensAlpha); }
else if (uBlendMode == 2) { blended = blendScreen(cameraColor, lensColor.rgb, lensAlpha); }
else if (uBlendMode == 3) { blended = blendOverlay(cameraColor, lensColor.rgb, lensAlpha); }

// 추가 (mode 4-6) — 동일 시그니처: (vec3 base, vec3 blend, float opacity)
else if (uBlendMode == 4) { blended = blendLuminanceTint(cameraColor, lensColor.rgb, lensAlpha); }
else if (uBlendMode == 5) { blended = blendLuminanceTintLinear(cameraColor, lensColor.rgb, lensAlpha); }
else if (uBlendMode == 6) { blended = blendSoftLight(cameraColor, lensColor.rgb, lensAlpha); }
```

> **주의**: 기존 mode 0-3 함수의 실제 시그니처가 `(vec3, vec4)` 형태라면, mode 4-6 추가 전에 `(vec3 base, vec3 blend, float opacity)` 통일 시그니처로 리팩터링 필요. 구현 시 기존 셰이더 코드 확인 필수.

### 3. uAvgIrisLum Uniform

#### CPU 측 (Kotlin — CameraGLRenderer.kt)

```kotlin
// 프로퍼티
private var avgIrisLum = 0.35f // 어두운 홍채 기본값 (한국인 평균 근사)
private var lastScale = 1.5f
private var lastScaleTimestamp = 0L
private var lastValidFaceTimeMs = 0L
private const val FACE_INVALID_TIMEOUT_MS = 2000L

// setIrisResult() 내부에서 호출
private fun updateAvgIrisLum(faceMeshValid: Boolean, currentTimeMs: Long) {
    if (faceMeshValid) {
        lastValidFaceTimeMs = currentTimeMs
        val newLum = sampleIrisLuminance(yuvBuffer, irisCenter, irisRadius)
        avgIrisLum = avgIrisLum * 0.9f + newLum * 0.1f // EMA (α=0.1)
        avgIrisLum = avgIrisLum.coerceIn(0.05f, 0.95f)
    } else {
        // Hold: 미검출 시 마지막 유효값 유지
        if (currentTimeMs - lastValidFaceTimeMs > FACE_INVALID_TIMEOUT_MS) {
            avgIrisLum = 0.35f // 기본값 리셋
        }
    }
}
```

#### Adaptive Scale Rate Limit (dt 기반)

```kotlin
fun computeAdaptiveScale(avgIrisLum: Float, currentTimeMs: Long): Float {
    val dt = (currentTimeMs - lastScaleTimestamp).coerceIn(1L, 100L) / 1000f
    lastScaleTimestamp = currentTimeMs

    val targetLum = 0.5f
    val rawScale = (targetLum / maxOf(0.1f, avgIrisLum)).coerceIn(0.8f, 2.5f)

    // Rate limit: 초당 최대 1.5 변화 (FPS 무관)
    val maxDelta = 1.5f * dt
    val clampedScale = lastScale + (rawScale - lastScale).coerceIn(-maxDelta, maxDelta)
    lastScale = clampedScale
    return clampedScale
}
```

**근거**:
- dt 기반으로 FPS 변동에 무관하게 동작 (Section 22 A4 합의)
- `coerceIn(1L, 100L)` dt 클램프로 cold-start/resume 폭주 방지 (Section 24 B3 합의)

#### YUV 밝기 샘플링

```kotlin
private fun sampleIrisLuminance(yuvBuffer: ByteBuffer, center: PointF, radius: Float): Float {
    // Y 채널만 읽으면 됨 (RGB 변환 불필요)
    // 홍채 중심 ± radius*0.5 범위에서 5~9개 포인트 샘플링
    // 평균 Y값을 0.0~1.0 범위로 정규화하여 반환
}
```

### 4. Demo App UI — Spinner 전환

#### GpuRenderActivity.kt

```kotlin
// RadioGroup 제거 → Spinner 추가
val blendModes = arrayOf(
    "Normal", "Multiply", "Screen", "Overlay",
    "Luminance Tint", "Luminance Tint (Linear)", "Soft Light"
)
val spinner = findViewById<Spinner>(R.id.spinnerBlendMode)
spinner.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, blendModes)
spinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
    override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
        lensConfig.blendMode = position
        cameraGLView.setLensConfig(lensConfig)
    }
    override fun onNothingSelected(parent: AdapterView<*>?) {}
}
```

**근거**: 브레인스토밍 Section 16 — RadioButton 7개는 가로 공간 수용 불가 → Spinner 전환.

#### onResume() Timestamp 리셋

`lastScaleTimestamp`는 `CameraGLRenderer`(렌더러) 소유 변수이므로, Activity에서 직접 접근하지 않고 렌더러 메서드를 통해 리셋:

```kotlin
// CameraGLRenderer.kt — 렌더러 측 리셋 메서드
fun resetTemporalState() {
    lastScaleTimestamp = 0L // dt 클램프(coerceIn 1~100ms)에 의해 안전하게 시작
}

// GpuRenderActivity.kt — Activity에서 호출
override fun onResume() {
    super.onResume()
    // ...existing code...
    cameraGLRenderer.resetTemporalState()
}
```

**근거**: 브레인스토밍 Section 24(B3) — resume 시 timestamp 리셋 1줄로 충분. 0L 리셋 후 첫 프레임 dt는 `coerceIn(1L, 100L)`에 의해 최대 100ms로 클램프되어 폭주 방지.

### 5. GPU Tier 기반 Feature Flag

| 기능 | Flag Key | HIGH | MID | LOW | 기본값 |
|------|----------|:---:|:---:|:---:|:---:|
| Luminance Tint (sRGB) | `tint_srgb` | ✓ | ✓ | ✓ | ON |
| Luminance Tint (Linear) | `tint_linear` | ✓ | ✓ | △ | OFF (A/B용) |
| Adaptive Scale | `adaptive_scale` | ✓ | ✓ | ✓ | ON |

△ = 성능 측정 후 결정

## Blend Mode 스펙 정리 (최종 합의)

| Mode | 이름 | 특성 | Specular | 선형화 |
|------|------|------|:---:|:---:|
| 0 | Normal | 기존 알파 블렌딩 | - | - |
| 1 | Multiply | 어두운 색 강조 | - | - |
| 2 | Screen | 밝은 색 강조 | - | - |
| 3 | Overlay | 대비 강화 | - | - |
| 4 | LuminanceTint | sRGB 근사 휘도 틴트 | 없음 | 없음 |
| 5 | LuminanceTintLinear | 선형 공간 휘도 틴트 | Real 복원 | x*x/sqrt |
| 6 | SoftLight | Photoshop Soft Light | 없음 | 없음 |

**Fake Specular**는 blend mode와 분리 — 별도 feature flag로 제어 (Section 19 합의).

## 검증 체크리스트

- [ ] Android 빌드 통과: `./gradlew :demo-app:assembleDebug`
- [ ] Spinner UI에서 7개 모드 선택 가능
- [ ] Mode 4 (LuminanceTint): 홍채 줄무늬가 보존되면서 색상 변경 확인
- [ ] Mode 5 (LuminanceTintLinear): Mode 4 대비 중간톤 탁함 개선 확인
- [ ] Mode 6 (SoftLight): 부드러운 조명 효과 확인
- [ ] 어두운 환경에서 scale 자동 상승, 밝은 환경에서 scale 자동 하강 확인
- [ ] onResume() 후 렌즈 깜빡임/점프 없음 확인

## 스모크 테스트 (권장 20~35분)

### 목적
- Mode 4/5/6이 실제 화면에서 동작하는지, UI/Resume/Adaptive scale 회귀가 없는지 빠르게 확인한다.

### 실행 순서

1. 빌드
   - 명령: `cd android && ./gradlew :demo-app:assembleDebug`
   - 확인: `BUILD SUCCESSFUL`

2. UI 확인
   - 동작: Demo App 실행 → Blend Spinner 열기
   - 확인: 모드 7개(0~6) 모두 표시/선택 가능

3. 모드별 시각 확인
   - Mode 4: 색상은 변하지만 홍채 결(줄무늬/명암) 유지
   - Mode 5: Mode 4 대비 중간톤 탁함 감소
   - Mode 6: 과한 대비 없이 부드러운 조명 효과
   - 확인: 모드 전환 시 렌더 정지/검은 화면/프레임 멈춤 없음

4. Adaptive scale 확인
   - 동작: 밝은 환경 ↔ 어두운 환경 전환(화면 조도 변화 유도)
   - 확인: scale이 급격히 튀지 않고 점진적으로 변함

5. Resume 안정성 확인
   - 동작: 홈으로 나갔다가 복귀 5회 반복
   - 확인: 복귀 직후 렌즈 위치/크기 점프, 1프레임 깜빡임 없음

6. 로그 확인
   - 명령 예시: `adb logcat -d | rg -i "shader|compile|link|gl_error|nan|inf"`
   - 확인: 셰이더 컴파일/링크 실패 및 NaN/Inf 관련 오류 0건

### 내가 확인해야 하는 핵심
- "모드가 선택된다"가 아니라, **모드별 시각 차이가 실제로 분리되어 보이는지**를 확인한다.
- Resume와 조도 변화에서 렌더가 안정적인지(점프/정지 없음)를 본다.

### 최종 판정
- 필수 통과: 1, 2, 3, 5
- 권장 통과: 4, 6
- 필수 항목 실패 시 P4-W2-01 진행 금지

## 다음 단계

1. A/B 비교: sRGB(mode 4) vs Linear(mode 5) 화질 차이 평가
2. 다양한 홍채 색상(어두운 갈색, 밝은 갈색, 파란색)에서 범용성 확인
3. P4-W2-01: Sclera Protection + Contact Shadow 착수

## 변경 이력

| 날짜 | 내용 |
|------|------|
| 2026-02-13 | 작업 계획 문서 작성 |
| 2026-02-13 | Codex 리뷰 반영: onResume() 리셋 렌더러 위임 명확화, blend 함수 시그니처 통일 |
