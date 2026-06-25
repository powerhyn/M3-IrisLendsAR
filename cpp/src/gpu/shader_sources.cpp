/**
 * @file shader_sources.cpp
 * @brief 내장 GLSL 셰이더 소스 정의
 *
 * OpenGL ES 3.1 셰이더 소스 코드를 C++ 문자열로 정의합니다.
 * 모든 셰이더는 #version 310 es를 사용합니다.
 */

#include "iris_sdk/gpu/shader_manager.h"

namespace iris_sdk {
namespace shaders {

//=============================================================================
// 풀스크린 쿼드 버텍스 셰이더
//=============================================================================
const char* FULLSCREEN_QUAD_VERTEX = R"glsl(
#version 310 es

layout(location = 0) in vec2 aPosition;
layout(location = 1) in vec2 aTexCoord;

out vec2 vTexCoord;

void main() {
    gl_Position = vec4(aPosition, 0.0, 1.0);
    vTexCoord = aTexCoord;
}
)glsl";

//=============================================================================
// 패스스루 프래그먼트 셰이더 (텍스처 그대로 출력)
//=============================================================================
const char* PASSTHROUGH_FRAGMENT = R"glsl(
#version 310 es
precision highp float;

uniform sampler2D uTexture;

in vec2 vTexCoord;
out vec4 fragColor;

void main() {
    fragColor = texture(uTexture, vTexCoord);
}
)glsl";

//=============================================================================
// (dead 정리) BRIGHTNESS_FRAGMENT / MASKING_FRAGMENT 셰이더 제거 — 호출 0.
//             brightness는 COMBINED_COLOR_ADJUSTMENT_FRAGMENT가 담당, masking은 미사용.
// (P8-W2 제거) Bilateral Filter / 화이트닝 / 컬러 밸런스 / 소프트 포커스 곁가지 셰이더 제거.
//=============================================================================

//=============================================================================
// 통합 Color Adjustment 프래그먼트 셰이더 (Brightness 잔존)
// (P8-W2) ColorBalance/Whitening/LUT 발췌 제거 — brightness GLSL만 유지.
//=============================================================================
const char* COMBINED_COLOR_ADJUSTMENT_FRAGMENT = R"glsl(
#version 310 es
precision highp float;

uniform sampler2D uTexture;
uniform float uBrightness;   // 0.5 ~ 1.5, 1.0 = 원본

in vec2 vTexCoord;
out vec4 fragColor;

void main() {
    vec4 color = texture(uTexture, vTexCoord);
    vec3 result = color.rgb;

    // Brightness
    if (abs(uBrightness - 1.0) > 0.01) {
        result *= uBrightness;
    }

    fragColor = vec4(clamp(result, 0.0, 1.0), color.a);
}
)glsl";

//=============================================================================
// (dead 정리) GAUSSIAN_BLUR_FRAGMENT 제거 — createProgram 호출조차 없는 완전 dead.
//             skin smoothing은 SKIN_SEPARABLE_BLUR_FRAGMENT가 담당.
// (P8-W2 제거) FreqSep Gaussian / FreqSep Composite / Luminance Sharpen /
//             Vivid Postprocess 곁가지 셰이더 제거.
//=============================================================================

//=============================================================================
// P8-W1: landmark-masked skin smoothing (LensSimulator 이식)
// 출처: Renderer.kt MASK_VS/MASK_FS/BLUR_FS/COMPOSITE_FS (1196-1210, 1180-1194, 1108-1167)
//=============================================================================

// 피부 마스크 단색 채움 — UV 불필요, NDC position만 (원본 MASK_VS)
const char* SKIN_MASK_FILL_VERTEX = R"glsl(
#version 310 es
layout(location = 0) in vec2 aPosition;
void main() {
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
)glsl";

// 마스크 값을 R 채널에 기록 (외곽=1.0, 제외 폴리곤=0.0 덮어쓰기). 원본 MASK_FS.
const char* SKIN_MASK_FILL_FRAGMENT = R"glsl(
#version 310 es
precision mediump float;
uniform float uValue;
out vec4 fragColor;
void main() {
    fragColor = vec4(uValue);
}
)glsl";

// 분리형 가우시안 5-fetch (linear sampling 트릭). 원본 BLUR_FS 원문 그대로.
// uDirection = 텍셀 단위 방향 (1/w,0) 또는 (0,1/h); uOffsetScale 컬러 1.6 / 마스크 1.0.
const char* SKIN_SEPARABLE_BLUR_FRAGMENT = R"glsl(
#version 310 es
precision highp float;
uniform sampler2D uTexture;
uniform vec2 uDirection;
uniform float uOffsetScale;
in vec2 vTexCoord;
out vec4 fragColor;
void main() {
    vec2 o1 = uDirection * (1.3846154 * uOffsetScale);
    vec2 o2 = uDirection * (3.2307692 * uOffsetScale);
    fragColor = texture(uTexture, vTexCoord) * 0.2270270
        + (texture(uTexture, vTexCoord + o1) + texture(uTexture, vTexCoord - o1)) * 0.3162162
        + (texture(uTexture, vTexCoord + o2) + texture(uTexture, vTexCoord - o2)) * 0.0702703;
}
)glsl";

// 에지 가드 컴포지트 — base/blur/mask 동일 UV (P8-W1 §1.4: OES/ST/크롭/미러/워프 체인 제거).
// 피부 스무딩 코어는 원본 COMPOSITE_FS(1159-1163) 수치 그대로.
const char* SKIN_SMOOTH_COMPOSITE_FRAGMENT = R"glsl(
#version 310 es
precision highp float;
uniform sampler2D uTexture;     // 원본 (풀해상도, base)
uniform sampler2D uBlurTex;     // 컬러 블러 결과 (1/4)
uniform sampler2D uSkinMaskTex; // 마스크 블러 결과 (1/4, R 채널)
uniform float uSkin;            // 0..1 강도. 0이면 패스 호출 안 됨
uniform float uRadiance;        // P8-W3: 0..1 화사함(soft-glow) 강도. 0이면 블록 생략
in vec2 vTexCoord;
out vec4 fragColor;
void main() {
    vec3 base = texture(uTexture, vTexCoord).rgb;
    vec3 blur = texture(uBlurTex, vTexCoord).rgb;
    float mask = texture(uSkinMaskTex, vTexCoord).r;
    // 에지 가드 (surface blur 근사): 강한 에지(코선·윤곽·머리카락·안경)는 휘도차가 커서
    // 스무딩에서 자동 제외 — 저대비 질감(모공·잡티)만 평탄화.
    float lumaBase = dot(base, vec3(0.299, 0.587, 0.114));
    float lumaBlur = dot(blur, vec3(0.299, 0.587, 0.114));
    float edge = smoothstep(0.06, 0.18, abs(lumaBase - lumaBlur));
    vec3 smoothed = blur + (base - blur) * 0.5; // 고주파 디테일 50% 보존
    base = mix(base, smoothed, mask * uSkin * (1.0 - edge));
    // P8-W3 radiance (soft-glow) — skin 블러+마스크 재활용, mask>0 게이팅.
    // 입력은 위 스무딩이 적용된 base + 공유 blur/mask (새 패스/fetch 0).
    // 상수·알고리즘은 LensSim S23+(기본 0.40) + 적대리뷰 확정값 그대로 이식.
    if (uRadiance > 0.0 && mask > 0.0) {
        const vec3 LW = vec3(0.299, 0.587, 0.114);
        const float RAD_GLOW  = 0.20;
        const float RAD_LIFT  = 0.08;
        const float RAD_DESAT = 0.10;
        const float RAD_KNEE  = 0.78;
        const float RAD_WARM  = 0.012;
        float rad = uRadiance;
        float lumaB = dot(base, LW);
        float hiRoll = 1.0 - smoothstep(RAD_KNEE - 0.06, RAD_KNEE + 0.17, lumaB);
        float radEdge = smoothstep(0.06, 0.18, abs(lumaB - dot(blur, LW)));
        vec3 screenC = 1.0 - (1.0 - base) * (1.0 - blur);
        vec3 bloomed = mix(base, screenC, RAD_GLOW * rad * hiRoll * (1.0 - radEdge));
        float midW = smoothstep(0.10, 0.35, lumaB) * (1.0 - smoothstep(0.70, 0.92, lumaB));
        float yIn  = max(dot(bloomed, LW), 1e-4);
        float yOut = yIn + RAD_LIFT * rad * midW * (1.0 - yIn);
        vec3  lifted = bloomed * (yOut / yIn);
        float y2   = dot(lifted, LW);
        vec3  even = mix(lifted, vec3(y2), RAD_DESAT * rad);
        even.r += RAD_WARM * rad * midW;
        even.b -= RAD_WARM * 0.5 * rad * midW;
        base = clamp(mix(base, even, mask * rad), 0.0, 1.0);
    }
    fragColor = vec4(base, 1.0);
}
)glsl";

//=============================================================================
// P8-W4: 턱 V라인 워프 (fragment-direct 비정규 RBF 인버스 워프)
// 출처: docs/lenssim-handoff/jaw-vline-warp-handoff-from-lenssimulator.md §4 (LensSim S23+ 검증).
// 풀스크린 패스. 제어점 14개(cx,cy,dx,dy 픽셀)를 uniform으로 받아 출력 픽셀 p의 소스를
//   src(p) = p − Σᵢ dᵢ·exp(−|p−cᵢ|²/2σ²)  (인버스 워프)
// 로 구해 입력 텍스처를 리샘플한다. bbox+3σ 밖이면 루프 생략(early-out), sigma=0이면 패스스루.
// 좌표계: vTexCoord/uViewportPx 는 렌더 텍스처(미러·Y-flip 적용) 공간 — 제어점도 CPU에서
//   prepareSkinFans 와 동형으로 같은 공간에 정렬해 넘긴다(gpu_beauty_backend.cpp executeWarpPass).
//=============================================================================
const char* WARP_FRAGMENT = R"glsl(
#version 310 es
precision highp float;

uniform sampler2D uTexture;   // 입력 (skin/brightness 거친 프레임)
uniform vec4 uWarp[14];       // [cx, cy, dx, dy] — 렌더 텍스처 픽셀 공간
uniform int uWarpCount;       // 활성 제어점 수 (보통 14)
uniform float uWarpSigma;     // 가우시안 σ (px). 0이면 워프 off(패스스루)
uniform vec4 uWarpBounds;     // [minX, minY, maxX, maxY] (제어점 bbox + 3σ), px
uniform vec2 uViewportPx;     // 뷰포트 px

in vec2 vTexCoord;
out vec4 fragColor;

void main() {
    vec2 px = vTexCoord * uViewportPx;
    vec2 dispUv = vTexCoord;
    // uniform 기반 분기라 타일 단위로 코히런트. bbox+3σ 밖이면 가중치 < e⁻⁴·⁵ ≈ 0.011.
    if (uWarpSigma > 0.0 &&
        all(greaterThanEqual(px, uWarpBounds.xy)) &&
        all(lessThanEqual(px, uWarpBounds.zw))) {
        vec2 disp = vec2(0.0);
        float inv2s2 = 0.5 / (uWarpSigma * uWarpSigma);
        for (int i = 0; i < uWarpCount; i++) {
            vec2 d = px - uWarp[i].xy;
            disp += uWarp[i].zw * exp(-dot(d, d) * inv2s2);
        }
        dispUv = (px - disp) / uViewportPx;  // 인버스 워프
    }
    fragColor = texture(uTexture, dispUv);
}
)glsl";

//=============================================================================
// 렌즈 오버레이 버텍스 셰이더 (풀스크린 쿼드, 텍스처 좌표 패스스루)
//=============================================================================
const char* LENS_OVERLAY_VERTEX = R"glsl(
#version 310 es

layout(location = 0) in vec2 aPosition;
layout(location = 1) in vec2 aTexCoord;

out vec2 vTexCoord;

void main() {
    gl_Position = vec4(aPosition, 0.0, 1.0);
    vTexCoord = aTexCoord;
}
)glsl";

//=============================================================================
// 렌즈 오버레이 프래그먼트 셰이더
// Kotlin 참조 구현(CameraGLRenderer.kt LENS_OVERLAY_FRAGMENT_SHADER)을
// 1:1로 포팅. 함수/식/uniform 시그니처가 모두 동일해야 시각적 결과가 일치한다.
//=============================================================================
const char* LENS_OVERLAY_FRAGMENT = R"glsl(
#version 310 es
precision highp float;

uniform sampler2D uCameraTexture;
uniform sampler2D uLensTexture;

uniform vec2 uLeftIrisCenter;
uniform float uLeftIrisRadius;
uniform vec2 uRightIrisCenter;
uniform float uRightIrisRadius;

uniform float uOpacity;
uniform float uLensScale;
uniform float uEdgeFeather;
uniform int uBlendMode;
uniform int uApplyLeft;
uniform int uApplyRight;
uniform float uFrameAspect;

uniform float uLeftEyeTop;
uniform float uLeftEyeBottom;
uniform float uRightEyeTop;
uniform float uRightEyeBottom;
uniform float uEyelidFeather;
uniform float uAvgIrisLum;
uniform float uDetH;

uniform int uScleraProtect;
uniform int uScleraVetoMode;  // P6-W5: 0=legacy, 1=color-veto(Codex), 2=luma-only(Gemini)
uniform int uContactShadow;
uniform float uShadowIntensity;
uniform float uMaxDetail;
// P5-W3-05 S1 D5: uHighlightEnabled uniform 제거

// P6-W3 §5.6/§5.11: C5 환경 반사 가산 계층 uniform.
// W3 scaffold는 uSourceType=0 (OFF) 기본 — 실기기 시각 변화 없음.
// W4 B2 벤치에서 uSourceType 토글로 OFF/EnvMap/Periphery 비교.
uniform int uSourceType;            // 0=OFF, 1=EnvMap, 2=Periphery
uniform float uReflectionIntensity; // 0.0~1.0, W3 기본 0.3
uniform sampler2D uEnvMap;          // W4 env-map 프로토타입용 (W3 미사용)

uniform int uUseEllipseMask;
uniform vec2 uLeftEyeEllipseCenter;
uniform vec3 uLeftEyeEllipseRadii;
uniform float uLeftEyeEllipseRot;
uniform vec2 uRightEyeEllipseCenter;
uniform vec3 uRightEyeEllipseRadii;
uniform float uRightEyeEllipseRot;

// P6-W6 §5.2/§5.7: C10 홍채 디테일 재주입 + B9 저조도 gate.
uniform vec2  uTexelSize;        // C10 3x3 blur 샘플 간격 (1/width, 1/height)
uniform float uGateThreshold;    // B9 gate 임계값 (토글 0.10/0.15/0.25, 기본 0.15)
uniform int   uDetailReinject;   // C10 on/off (기본 1)
uniform float uLowLightActive;   // P7-W2 §5.4: gate 전용 저조도 래치(0..1, hysteresis)
// P6-W6 §1.3 C7: 블링크 시간적 envelope (좌/우 EMA ramp). main()에서 좌→Left, 우→Right.
uniform float uLeftRenderAlpha;
uniform float uRightRenderAlpha;

in vec2 vTexCoord;
out vec4 fragColor;

// P6-W2 §5.10: 블렌드 LUMA 계수 Rec.709 linear로 통일.
// CPU 측 W1 avg_iris_luma 측정과 동일 상수. sRGB 평균 금지.
const vec3 LUMA_709_LENS = vec3(0.2126, 0.7152, 0.0722);

vec3 toLinearFast(vec3 srgb) { return srgb * srgb; }
vec3 toSRGBFast(vec3 linear_color) { return sqrt(max(linear_color, vec3(0.0))); }

vec3 blendNormal(vec3 base, vec3 blend, float opacity) {
    return mix(base, blend, opacity);
}

vec3 blendMultiply(vec3 base, vec3 blend, float opacity) {
    return mix(base, base * blend, opacity);
}

// P6-W2 §5.2 C3: ScreenLinear (선형 공간 Screen). sRGB blendScreen 대체.
vec3 blendScreenLinear(vec3 base, vec3 blend, float opacity) {
    vec3 baseL = toLinearFast(base);
    vec3 lensL = toLinearFast(blend);
    vec3 screened = vec3(1.0) - (vec3(1.0) - baseL) * (vec3(1.0) - lensL);
    return toSRGBFast(mix(baseL, screened, opacity));
}

// P6-W2 §5.1 C2: TintLinearV2 (canonical default). LTL 리네이밍 + squaring 제거.
// uAvgIrisLum은 W1 fallback chain에서 이미 linear 공간 값으로 공급됨 (W1 §5.2.1).
// K(=0.7) + clamp upper(7.0)는 실기기 시각 튜닝값. ColorReplaceLinear(ID=7)는 별도
// 수식이라 K 영향 없음 — 5번 단독 강도 조정 가능.
vec3 blendTintLinearV2(vec3 base, vec3 blend, float opacity) {
    vec3 baseL = toLinearFast(base);
    float lum = dot(baseL, LUMA_709_LENS);
    float scale = clamp(0.85 / max(0.01, uAvgIrisLum), 0.8, 7.0);
    vec3 tinted = toLinearFast(blend) * lum * scale;
    vec3 result = mix(baseL, tinted, opacity);
    return toSRGBFast(result);
}

// P6-W2 §5.3 C4: ColorReplaceLinear. W2 활성 — W5 B1 벤치 대상, 채택 확정 아님.
vec3 blendColorReplaceLinear(vec3 base, vec3 blend, float opacity, float maxDetail) {
    vec3 baseL = toLinearFast(base);
    vec3 lensL = toLinearFast(blend);
    float lum = dot(baseL, LUMA_709_LENS);
    float detail = clamp(pow(lum / max(0.01, uAvgIrisLum), 0.7), 0.75, maxDetail);
    vec3 colored = lensL * detail;
    return toSRGBFast(mix(baseL, colored, opacity));
}

float asymmetricEllipseMask(vec2 uv, vec2 center, vec3 radii, float rotation, float feather) {
    vec2 d = uv - center;
    float cosR = cos(rotation);
    float sinR = sin(rotation);
    d = vec2(d.x * cosR + d.y * sinR, -d.x * sinR + d.y * cosR);
    float rx = (d.x < 0.0) ? radii.x : radii.y;
    float ry = radii.z;
    float ellipseDist = length(vec2(d.x / max(rx, 1e-5), d.y / max(ry, 1e-5)));
    return smoothstep(1.0, 1.0 - feather, ellipseDist);
}

float calcScleraFactor(vec3 cameraColor) {
    float brightness = dot(cameraColor, vec3(0.299, 0.587, 0.114));
    float maxC = max(cameraColor.r, max(cameraColor.g, cameraColor.b));
    float minC = min(cameraColor.r, min(cameraColor.g, cameraColor.b));
    float saturation = (maxC - minC) / max(maxC, 1e-4);
    float brightFactor = smoothstep(0.3, 0.5, brightness);
    float lowSatFactor = 1.0 - smoothstep(0.1, 0.3, saturation);
    return brightFactor * lowSatFactor;
}

float calcContactShadow(float fragY, float minY, float eyelidFeather, float eyeOpening) {
    float shadowDepthPx = 4.0;
    float shadowDepth = shadowDepthPx / max(uDetH, 1.0);
    float shadowIntensity = clamp(uShadowIntensity, 0.0, 0.25);
    float shadowZone = smoothstep(
        minY + eyelidFeather,
        minY + eyelidFeather + shadowDepth,
        fragY
    );
    float shadowFactor = (1.0 - shadowZone) * shadowIntensity;
    float shadowEnable = smoothstep(0.015, 0.025, eyeOpening);
    shadowFactor *= shadowEnable;
    float maskAlpha = smoothstep(minY, minY + eyelidFeather, fragY);
    return shadowFactor * maskAlpha;
}

// P6-W3 §5.11 / P6-W4 §5.7/§5.8: sampleReflection — 방식 A (uniform 스위치, 단일 바이너리).
// W4 Phase A: OFF/EnvMap/Periphery 3 프로토타입 활성.
// reflectUV: env-map sampling UV (옵션 C iris local 좌표).
// irisCenterAdjusted: Periphery annular ring 중심 (aspectRatio 보정된 좌표).
// scaledRadius: Periphery ring 반경 단위 (irisRadius * uLensScale).
vec3 sampleReflection(vec2 reflectUV, vec2 irisCenterAdjusted, float scaledRadius) {
    if (uSourceType == 1) {
        // EnvMap (Codex 안)
        return texture(uEnvMap, reflectUV).rgb;
    }
    if (uSourceType == 2) {
        // Periphery (Gemini 안) — annular ring r∈[1.8, 2.5] 8포인트, uCameraTexture에서 샘플.
        // 얼굴 중앙 영역(70%) 자연 제외 효과: ring이 iris 외곽 1.8r 이상이라 동공/홍채 자체 제외.
        const int N = 8;
        const float RING_R = 2.15;  // [1.8, 2.5] 중앙값. W4 벤치 시 미세 조정 가능.
        vec3 sum = vec3(0.0);
        for (int i = 0; i < N; ++i) {
            float angle = float(i) * (6.28318530718 / float(N));
            // ring 위 한 점을 adjusted-coord 공간에서 계산 후 vTexCoord 공간으로 환원.
            // adjustedCoord = vTexCoord * vec2(aspectRatio, 1) 이므로 역변환 시 aspect 나눠야.
            vec2 adjustedRingPoint = irisCenterAdjusted + RING_R * scaledRadius * vec2(cos(angle), sin(angle));
            // adjusted → vTexCoord 환원 (aspectRatio는 셰이더 전체 const 수준 — uFrameAspect 사용)
            vec2 ringUV = vec2(adjustedRingPoint.x / uFrameAspect, adjustedRingPoint.y);
            // 화면 밖 fallback: clamp로 가장자리 색 사용 (검은색 강제 회피)
            ringUV = clamp(ringUV, vec2(0.0), vec2(1.0));
            sum += texture(uCameraTexture, ringUV).rgb;
        }
        return sum / float(N);
    }
    return vec3(0.0);  // OFF (W3 기본)
}

// P6-W3 §5.8 + W4 Phase A 보완: calcFresnel — 옵션 C 가짜 Fresnel.
// dist는 이미 /scaledRadius로 정규화 (0=중심, 1=외곽). 노멀 벡터 없음 (D1 재도입 회피).
// boundary (0.6, 1.0): R1 inner 후보 [0.6, 0.7, 0.8] 중 가장 안쪽 채택.
// 사유: Phase A 시각 검증에서 inner 0.7 + outer 1.0이 edgeAlpha 페이드와 정확히 중첩 →
//       가시성 거의 0. inner 0.6으로 안쪽 이동해 외곽 40%에 반사 분포.
//       outer 1.0은 R1 합의 그대로 유지 (W3 §5.8). W4 §1.15 참조.
float calcFresnel(float dist) {
    return smoothstep(0.6, 1.0, dist);
}

vec4 applyLens(vec4 camera, vec2 irisCenter, float irisRadius, float aspectRatio,
               float eyeTop, float eyeBottom,
               vec2 ellipseCenter, vec3 ellipseRadii, float ellipseRot,
               float renderAlpha) {
    if (irisRadius <= 0.0) return camera;

    vec2 adjustedCoord = vec2(vTexCoord.x * aspectRatio, vTexCoord.y);
    vec2 adjustedCenter = vec2(irisCenter.x * aspectRatio, irisCenter.y);

    float scaledRadius = irisRadius * uLensScale;
    float dist = distance(adjustedCoord, adjustedCenter) / scaledRadius;

    // early return 제거 — mipmap + dynamic branch에서 texture() gradient가
    // undefined되어 검은 화면 발생하는 Adreno 드라이버 이슈 방지.
    // dist >= 1.0이면 edgeAlpha=0 → finalAlpha=0 → mix 결과=camera로 동일 결과.
    vec2 lensCoord = clamp(
        (adjustedCoord - adjustedCenter) / scaledRadius * 0.5 + 0.5,
        vec2(0.0), vec2(1.0));

    vec4 lens = texture(uLensTexture, lensCoord);
    if (lens.a > 0.001) lens.rgb /= lens.a;

    float featherStart = 1.0 - uEdgeFeather;
    float edgeAlpha = smoothstep(1.0, featherStart, dist);

    float eyelidFeather = uEyelidFeather;
    float minY = min(eyeTop, eyeBottom);
    float maxY = max(eyeTop, eyeBottom);
    float eyelidMask;

    if (uUseEllipseMask == 1 && ellipseRadii.z > 0.0) {
        eyelidMask = asymmetricEllipseMask(vTexCoord, ellipseCenter, ellipseRadii, ellipseRot, eyelidFeather * 3.0);
    } else {
        float topClip = smoothstep(minY - eyelidFeather, minY + eyelidFeather, vTexCoord.y);
        float bottomClip = 1.0 - smoothstep(maxY - eyelidFeather, maxY + eyelidFeather, vTexCoord.y);
        eyelidMask = topClip * bottomClip;
    }

    float finalAlpha = lens.a * uOpacity * edgeAlpha * eyelidMask;

    float irisEdgeDist = dist * uLensScale;
    if (uScleraProtect == 1) {
        float geom = smoothstep(0.75, 1.0, irisEdgeDist);
        float scleraFade;
        if (uScleraVetoMode == 1) {
            // P6-W5 §5.4 / §5.14: Codex color-veto (sat + luma). 0.6 강도 W5 1차 고정.
            float maxC = max(camera.r, max(camera.g, camera.b));
            float minC = min(camera.r, min(camera.g, camera.b));
            float sat = (maxC - minC) / max(maxC, 1e-4);
            float lum = dot(camera.rgb, vec3(0.299, 0.587, 0.114));
            float veto = smoothstep(0.18, 0.32, sat) * (1.0 - smoothstep(0.45, 0.65, lum));
            scleraFade = 1.0 - geom * (1.0 - 0.6 * veto);
        } else if (uScleraVetoMode == 2) {
            // P6-W5 §5.4 / §5.12: Gemini luma-only (sat 항 제거). 임계값 (0.45, 0.65) 유지.
            float lum = dot(camera.rgb, vec3(0.299, 0.587, 0.114));
            float veto = 1.0 - smoothstep(0.45, 0.65, lum);
            scleraFade = 1.0 - geom * (1.0 - 0.6 * veto);
        } else {
            // Legacy (W5 이전 수식). W5 결과 반영 시점에 §5.13 따라 단일 수식으로 정리.
            float colorFactor = calcScleraFactor(camera.rgb);
            scleraFade = 1.0 - geom * (0.5 + 0.5 * colorFactor);
        }
        finalAlpha *= scleraFade;
    }

    // P6-W6 §1.3 C7: 블링크 시간적 envelope. 눈 감김/뜸 EMA ramp(CPU 측 계산)를
    // 최종 가시성에 곱해 깜빡임 hard pop 제거. blended 합성 전에 적용.
    finalAlpha *= renderAlpha;

    float maxDetail = mix(uMaxDetail, 1.0, smoothstep(0.75, 1.0, irisEdgeDist));

    // P6-W2 §5.4/§5.8/§5.9: 블렌드 분기 5종 등록 (0/1/2/5/7) + 빈 ID(3/4/6) fallback.
    //   - ID 0 Normal: 유지 (W5 B1 Normal vs CRL 벤치 대기)
    //   - ID 2 ScreenLinear: 선형 공간 (W2 sRGB Screen 대체)
    //   - ID 5 TintLinearV2: canonical default
    //   - ID 7 ColorReplaceLinear: W2 활성 — W5 B1 벤치 대상, 채택 확정 아님
    //   - ID 3/4/6/기타: TintLinearV2 fallback (디버그 로그는 CPU 측에서 1회)
    vec3 blended;
    if (uBlendMode == 0) {
        blended = blendNormal(camera.rgb, lens.rgb, finalAlpha);
    } else if (uBlendMode == 1) {
        blended = blendMultiply(camera.rgb, lens.rgb, finalAlpha);
    } else if (uBlendMode == 2) {
        blended = blendScreenLinear(camera.rgb, lens.rgb, finalAlpha);
    } else if (uBlendMode == 5) {
        blended = blendTintLinearV2(camera.rgb, lens.rgb, finalAlpha);
    } else if (uBlendMode == 7) {
        blended = blendColorReplaceLinear(camera.rgb, lens.rgb, finalAlpha, maxDetail);
    } else {
        blended = blendTintLinearV2(camera.rgb, lens.rgb, finalAlpha);
    }

    // P6-W6 §5.2 C10: iris inner 디테일 재주입. 블렌드 직후 / 반사 이전 (handoff §4 패스 순서).
    // 입력은 원본 카메라(uCameraTexture, 렌즈 적용 전)의 고주파 디테일 — linear Rec.709 luma.
    if (uDetailReinject == 1) {
        vec2 t = uTexelSize;
        // P7-W1: dynamic branch(uDetailReinject==1) 안 implicit-LOD texture() 는 GLSL ES 3.0
        // spec §8.9 위반(non-uniform control flow 에서 derivative undefined). textureLod(uv,0.0)
        // 으로 명시 LOD 지정 → derivative 불필요. mipmap 미사용 + LINEAR filter라 시각 결과 동일.
        float lC  = dot(toLinearFast(textureLod(uCameraTexture, vTexCoord, 0.0).rgb), LUMA_709_LENS);
        float lN  = dot(toLinearFast(textureLod(uCameraTexture, vTexCoord + vec2(0.0, -t.y), 0.0).rgb), LUMA_709_LENS);
        float lS  = dot(toLinearFast(textureLod(uCameraTexture, vTexCoord + vec2(0.0,  t.y), 0.0).rgb), LUMA_709_LENS);
        float lE  = dot(toLinearFast(textureLod(uCameraTexture, vTexCoord + vec2( t.x, 0.0), 0.0).rgb), LUMA_709_LENS);
        float lW  = dot(toLinearFast(textureLod(uCameraTexture, vTexCoord + vec2(-t.x, 0.0), 0.0).rgb), LUMA_709_LENS);
        float lNE = dot(toLinearFast(textureLod(uCameraTexture, vTexCoord + vec2( t.x, -t.y), 0.0).rgb), LUMA_709_LENS);
        float lNW = dot(toLinearFast(textureLod(uCameraTexture, vTexCoord + vec2(-t.x, -t.y), 0.0).rgb), LUMA_709_LENS);
        float lSE = dot(toLinearFast(textureLod(uCameraTexture, vTexCoord + vec2( t.x,  t.y), 0.0).rgb), LUMA_709_LENS);
        float lSW = dot(toLinearFast(textureLod(uCameraTexture, vTexCoord + vec2(-t.x,  t.y), 0.0).rgb), LUMA_709_LENS);
        float blurLum = (lC * 2.0 + lN + lS + lE + lW + lNE + lNW + lSE + lSW) / 10.0;  // W6 §5.8 3x3 single-pass
        float baseLum = lC;
        float detail = clamp(baseLum / max(blurLum, 0.001), 0.85, 1.15);                 // W6 §5.2
        float innerMask = smoothstep(0.7, 0.5, dist);                                    // W6 §5.9 중심=1, 외곽=0
        float gateStrength = smoothstep(uGateThreshold - 0.03, uGateThreshold + 0.03, uAvgIrisLum); // W6 §5.7 ±0.03
        // P7-W2 §5.4: 저조도 래치가 켜지면 디테일 재주입을 안정적으로 끈다(경계 진동 차단).
        //   uLowLightActive는 CPU dual-threshold(enter0.08/exit0.12) 래치 상태.
        //   uAvgIrisLum(블렌드 정규화 분모)은 그대로 — gate에만 영향(§5.4).
        gateStrength *= (1.0 - uLowLightActive);
        float detailMul = mix(1.0, detail, gateStrength * innerMask);
        blended *= vec3(detailMul);
    }

    // P6-W7 (제거): 셰이더 림발 수식. 실기기 적용 결과 렌즈마다 림발 색·스타일이 달라
    // 고정 darkening이 디자인을 훼손함(예: 브라운 림발 렌즈에 회색 darkening 덮음).
    // 림발은 렌즈 에셋이 책임. P5-W3-05 S1 D2 제거 사유와 동일.

    // P5-W3-05 S1 D5: uHighlightEnabled 각막 하이라이트 블록 제거
    // 고정 위치 하이라이트는 환경과 무관해 어색함. C5 환경 반사 가산 계층(B2 결과 후)이 대체
    // 삭제된 수식:
    //   if (uHighlightEnabled == 1) {
    //     vec2 localDir = (adjustedCoord - adjustedCenter) / max(scaledRadius, 1e-5);
    //     vec2 highlightCenter = vec2(-0.3, 0.4);
    //     float highlightDist = distance(localDir, highlightCenter);
    //     float highlight = smoothstep(0.25, 0.0, highlightDist);
    //     blended = mix(blended, vec3(1.0), highlight * 0.5 * finalAlpha);
    //   }

    // P6-W3 §5.1/§5.4 + W4 Phase A 보완: C5 환경 반사 가산 합성.
    // W3 §5.5 원안: renderMask = finalAlpha (= lens.a * uOpacity * edgeAlpha * eyelidMask).
    // W4 Phase A 1차 보완: edgeAlpha 제거 → 외곽 가산 살아남 + 가시성 확보.
    // W4 Phase A 2차 보완: lens silhouette 가드 누락 발견 (lensCoord clamp가 dist>1.0에서도
    //   외곽 lens.a 반환 → 얼굴/안경 영역까지 반사 누수 → 노란 가로 띠 발생).
    //   `step(dist, 1.0)`로 hard cutoff. edgeAlpha의 soft fade 역할은 포기하고
    //   가드 역할만 복원. W3 §5.5/§5.8 + W4 §1.16 참조.
    float renderMask = lens.a * uOpacity * eyelidMask * step(dist, 1.0);
#ifdef RENDER_MASK_HOOK_ENABLED
    // P6-W3 §5.10: W8 Pupil material 조건부 트랙. CMake 옵션으로만 활성 (프로덕션 비활성).
    // smoothstep 인자는 Codex R4 Patch 4 단서대로 예시 — W8 구현 시 실기기 튜닝.
    renderMask = max(finalAlpha, smoothstep(1.2, 0.0, dist));
#endif
    // P6-W3 §5.12: reflectUV 옵션 C (iris local 좌표). 노멀 없음 → 옵션 B(reflect) 배제.
    vec2 reflectUV = (adjustedCoord - adjustedCenter) / scaledRadius * 0.5 + 0.5;
    vec3 reflection = sampleReflection(reflectUV, adjustedCenter, scaledRadius);
    float fresnel = calcFresnel(dist);
    blended += reflection * fresnel * uReflectionIntensity * renderMask;

    if (uContactShadow == 1) {
        float eyeOpening = abs(maxY - minY);
        float shadow = calcContactShadow(vTexCoord.y, minY, eyelidFeather, eyeOpening);
        blended *= (1.0 - shadow);
    }

    return vec4(blended, camera.a);
}

void main() {
    vec4 camera = texture(uCameraTexture, vTexCoord);
    vec4 result = camera;

    float aspectRatio = uFrameAspect;

    if (uApplyLeft == 1 && uLeftIrisRadius > 0.0) {
        result = applyLens(result, uLeftIrisCenter, uLeftIrisRadius, aspectRatio,
                           uLeftEyeTop, uLeftEyeBottom,
                           uLeftEyeEllipseCenter, uLeftEyeEllipseRadii, uLeftEyeEllipseRot,
                           uLeftRenderAlpha);
    }

    if (uApplyRight == 1 && uRightIrisRadius > 0.0) {
        result = applyLens(result, uRightIrisCenter, uRightIrisRadius, aspectRatio,
                           uRightEyeTop, uRightEyeBottom,
                           uRightEyeEllipseCenter, uRightEyeEllipseRadii, uRightEyeEllipseRot,
                           uRightRenderAlpha);
    }

    fragColor = result;
}
)glsl";

} // namespace shaders
} // namespace iris_sdk
