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
// 밝기 조정 프래그먼트 셰이더
//=============================================================================
const char* BRIGHTNESS_FRAGMENT = R"glsl(
#version 310 es
precision highp float;

uniform sampler2D uTexture;
uniform float uBrightness;  // 0.5 ~ 1.5, 1.0 = 원본

in vec2 vTexCoord;
out vec4 fragColor;

void main() {
    vec4 color = texture(uTexture, vTexCoord);

    // 밝기 조정 (RGB만, 알파 유지)
    color.rgb *= uBrightness;

    // 클램핑
    fragColor = vec4(clamp(color.rgb, 0.0, 1.0), color.a);
}
)glsl";

//=============================================================================
// Bilateral Filter 프래그먼트 셰이더 (피부 스무딩)
//=============================================================================
const char* BILATERAL_FILTER_FRAGMENT = R"glsl(
#version 310 es
precision highp float;

uniform sampler2D uTexture;
uniform vec2 uTexelSize;    // 1.0 / (width, height)
uniform float uStrength;    // 0.0 ~ 1.0

in vec2 vTexCoord;
out vec4 fragColor;

// 간단한 Bilateral Filter 구현
// 실제 프로덕션에서는 더 최적화된 버전 사용
void main() {
    vec4 centerColor = texture(uTexture, vTexCoord);

    if (uStrength < 0.01) {
        fragColor = centerColor;
        return;
    }

    // Bilateral 파라미터
    float sigmaSpace = 3.0 + uStrength * 5.0;  // 공간 시그마 (3~8)
    float sigmaColor = 0.1 + uStrength * 0.4;  // 색상 시그마 (0.1~0.5)

    // 커널 반경 (성능을 위해 제한)
    const int RADIUS = 4;

    vec3 sum = vec3(0.0);
    float weightSum = 0.0;

    for (int y = -RADIUS; y <= RADIUS; y++) {
        for (int x = -RADIUS; x <= RADIUS; x++) {
            vec2 offset = vec2(float(x), float(y)) * uTexelSize;
            vec4 sampleColor = texture(uTexture, vTexCoord + offset);

            // 공간 가중치 (가우시안)
            float spatialDist = float(x * x + y * y);
            float spatialWeight = exp(-spatialDist / (2.0 * sigmaSpace * sigmaSpace));

            // 색상 가중치 (색상 유사도)
            vec3 colorDiff = sampleColor.rgb - centerColor.rgb;
            float colorDist = dot(colorDiff, colorDiff);
            float colorWeight = exp(-colorDist / (2.0 * sigmaColor * sigmaColor));

            float weight = spatialWeight * colorWeight;
            sum += sampleColor.rgb * weight;
            weightSum += weight;
        }
    }

    vec3 filtered = sum / max(weightSum, 0.001);

    // 원본과 블렌딩 (강도 조절)
    vec3 result = mix(centerColor.rgb, filtered, uStrength);

    fragColor = vec4(result, centerColor.a);
}
)glsl";

//=============================================================================
// 화이트닝 프래그먼트 셰이더
//=============================================================================
const char* WHITENING_FRAGMENT = R"glsl(
#version 310 es
precision highp float;

uniform sampler2D uTexture;
uniform float uStrength;  // 0.0 ~ 1.0

in vec2 vTexCoord;
out vec4 fragColor;

// RGB to YCbCr
vec3 rgb2ycbcr(vec3 rgb) {
    float y = 0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b;
    float cb = -0.169 * rgb.r - 0.331 * rgb.g + 0.500 * rgb.b + 0.5;
    float cr = 0.500 * rgb.r - 0.419 * rgb.g - 0.081 * rgb.b + 0.5;
    return vec3(y, cb, cr);
}

// YCbCr to RGB
vec3 ycbcr2rgb(vec3 ycbcr) {
    float y = ycbcr.x;
    float cb = ycbcr.y - 0.5;
    float cr = ycbcr.z - 0.5;
    float r = y + 1.402 * cr;
    float g = y - 0.344 * cb - 0.714 * cr;
    float b = y + 1.772 * cb;
    return vec3(r, g, b);
}

void main() {
    vec4 color = texture(uTexture, vTexCoord);

    if (uStrength < 0.01) {
        fragColor = color;
        return;
    }

    // YCbCr 변환
    vec3 ycbcr = rgb2ycbcr(color.rgb);

    // 하이라이트 보호: Y가 높은 영역에서 boost를 점진적으로 줄임
    float highlightProtection = 1.0 - smoothstep(0.7, 0.95, ycbcr.x) * 0.7;

    // 적응형 밝기 증가 (하이라이트 영역은 boost 감소)
    float baseLuminanceBoost = 1.0 + uStrength * 0.2;
    float luminanceBoost = 1.0 + (baseLuminanceBoost - 1.0) * highlightProtection;
    float saturationReduce = 1.0 - uStrength * 0.15;  // 최대 15% 채도 감소

    ycbcr.x = min(ycbcr.x * luminanceBoost, 1.0);
    ycbcr.y = mix(0.5, ycbcr.y, saturationReduce);
    ycbcr.z = mix(0.5, ycbcr.z, saturationReduce);

    // RGB 변환
    vec3 result = ycbcr2rgb(ycbcr);

    fragColor = vec4(clamp(result, 0.0, 1.0), color.a);
}
)glsl";

//=============================================================================
// 컬러 밸런스 프래그먼트 셰이더
//=============================================================================
const char* COLOR_BALANCE_FRAGMENT = R"glsl(
#version 310 es
precision highp float;

uniform sampler2D uTexture;
uniform float uBalance;  // -1.0 (쿨톤) ~ 1.0 (웜톤)

in vec2 vTexCoord;
out vec4 fragColor;

void main() {
    vec4 color = texture(uTexture, vTexCoord);

    if (abs(uBalance) < 0.01) {
        fragColor = color;
        return;
    }

    vec3 result = color.rgb;

    if (uBalance > 0.0) {
        // 웜톤: R/Yellow 증가
        result.r += uBalance * 0.08;
        result.g += uBalance * 0.04;
    } else {
        // 쿨톤: B/Cyan 증가
        result.b += abs(uBalance) * 0.08;
        result.g += abs(uBalance) * 0.02;
    }

    fragColor = vec4(clamp(result, 0.0, 1.0), color.a);
}
)glsl";

//=============================================================================
// 소프트 포커스 프래그먼트 셰이더
//=============================================================================
const char* SOFT_FOCUS_FRAGMENT = R"glsl(
#version 310 es
precision highp float;

uniform sampler2D uTexture;
uniform vec2 uTexelSize;
uniform float uStrength;  // 0.0 ~ 1.0

in vec2 vTexCoord;
out vec4 fragColor;

void main() {
    vec4 centerColor = texture(uTexture, vTexCoord);

    if (uStrength < 0.01) {
        fragColor = centerColor;
        return;
    }

    // 간단한 Box Blur로 소프트 포커스 효과
    const int RADIUS = 3;
    vec3 sum = vec3(0.0);
    float count = 0.0;

    for (int y = -RADIUS; y <= RADIUS; y++) {
        for (int x = -RADIUS; x <= RADIUS; x++) {
            vec2 offset = vec2(float(x), float(y)) * uTexelSize * 2.0;
            sum += texture(uTexture, vTexCoord + offset).rgb;
            count += 1.0;
        }
    }

    vec3 blurred = sum / count;

    // 원본 + 블러를 스크린 블렌딩 (소프트 글로우)
    vec3 softFocus = 1.0 - (1.0 - centerColor.rgb) * (1.0 - blurred * 0.3);

    // 강도에 따라 블렌딩
    vec3 result = mix(centerColor.rgb, softFocus, uStrength * 0.5);

    fragColor = vec4(result, centerColor.a);
}
)glsl";

//=============================================================================
// 마스킹 프래그먼트 셰이더 (ROI 블렌딩)
//=============================================================================
const char* MASKING_FRAGMENT = R"glsl(
#version 310 es
precision highp float;

uniform sampler2D uFiltered;   // 필터 적용된 텍스처
uniform sampler2D uOriginal;   // 원본 텍스처
uniform sampler2D uMask;       // 마스크 (R 채널 사용, 1=필터, 0=원본)

in vec2 vTexCoord;
out vec4 fragColor;

void main() {
    vec4 filtered = texture(uFiltered, vTexCoord);
    vec4 original = texture(uOriginal, vTexCoord);
    float mask = texture(uMask, vTexCoord).r;

    // 마스크에 따라 블렌딩
    // mask = 1.0: 필터 적용된 영역 (피부)
    // mask = 0.0: 원본 유지 영역 (눈, 입술)
    fragColor = mix(original, filtered, mask);
}
)glsl";

//=============================================================================
// 통합 Color Adjustment 프래그먼트 셰이더 (Brightness + ColorBalance + Whitening)
// 3개 패스를 1개로 병합하여 FBO 전환 오버헤드 감소
//=============================================================================
const char* COMBINED_COLOR_ADJUSTMENT_FRAGMENT = R"glsl(
#version 310 es
precision highp float;

uniform sampler2D uTexture;
uniform float uBrightness;   // 0.5 ~ 1.5, 1.0 = 원본
uniform float uBalance;      // -1.0 (쿨톤) ~ 1.0 (웜톤)
uniform float uWhitening;    // 0.0 ~ 1.0
uniform highp sampler3D uLutTexture;
uniform float uLutIntensity;  // 0.0 = LUT disabled

in vec2 vTexCoord;
out vec4 fragColor;

// RGB to YCbCr
vec3 rgb2ycbcr(vec3 rgb) {
    float y = 0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b;
    float cb = -0.169 * rgb.r - 0.331 * rgb.g + 0.500 * rgb.b + 0.5;
    float cr = 0.500 * rgb.r - 0.419 * rgb.g - 0.081 * rgb.b + 0.5;
    return vec3(y, cb, cr);
}

// YCbCr to RGB
vec3 ycbcr2rgb(vec3 ycbcr) {
    float y = ycbcr.x;
    float cb = ycbcr.y - 0.5;
    float cr = ycbcr.z - 0.5;
    float r = y + 1.402 * cr;
    float g = y - 0.344 * cb - 0.714 * cr;
    float b = y + 1.772 * cb;
    return vec3(r, g, b);
}

void main() {
    vec4 color = texture(uTexture, vTexCoord);
    vec3 result = color.rgb;

    // 1. Brightness (가장 먼저 적용)
    if (abs(uBrightness - 1.0) > 0.01) {
        result *= uBrightness;
    }

    // 2. Color Balance
    if (abs(uBalance) > 0.01) {
        if (uBalance > 0.0) {
            // 웜톤: R/Yellow 증가
            result.r += uBalance * 0.08;
            result.g += uBalance * 0.04;
        } else {
            // 쿨톤: B/Cyan 증가
            result.b += abs(uBalance) * 0.08;
            result.g += abs(uBalance) * 0.02;
        }
    }

    // 3. Whitening
    if (uWhitening > 0.01) {
        // YCbCr 변환
        vec3 ycbcr = rgb2ycbcr(result);

        // 하이라이트 보호: Y가 높은 영역에서 boost를 점진적으로 줄임
        float highlightProtection = 1.0 - smoothstep(0.7, 0.95, ycbcr.x) * 0.7;

        // 적응형 밝기 증가 (하이라이트 영역은 boost 감소)
        float baseLuminanceBoost = 1.0 + uWhitening * 0.2;
        float luminanceBoost = 1.0 + (baseLuminanceBoost - 1.0) * highlightProtection;
        float saturationReduce = 1.0 - uWhitening * 0.15;  // 최대 15% 채도 감소

        ycbcr.x = min(ycbcr.x * luminanceBoost, 1.0);
        ycbcr.y = mix(0.5, ycbcr.y, saturationReduce);
        ycbcr.z = mix(0.5, ycbcr.z, saturationReduce);

        // RGB 변환
        result = ycbcr2rgb(ycbcr);
    }

    // 4. LUT Application (after color correction, for stylization)
    if (uLutIntensity > 0.01) {
        vec3 lutColor = texture(uLutTexture, result).rgb;
        result = mix(result, lutColor, uLutIntensity);
    }

    fragColor = vec4(clamp(result, 0.0, 1.0), color.a);
}
)glsl";

//=============================================================================
// Gaussian Blur 프래그먼트 셰이더 (추후 사용)
//=============================================================================
const char* GAUSSIAN_BLUR_FRAGMENT = R"glsl(
#version 310 es
precision highp float;

uniform sampler2D uTexture;
uniform vec2 uTexelSize;
uniform vec2 uDirection;  // (1,0) = horizontal, (0,1) = vertical

in vec2 vTexCoord;
out vec4 fragColor;

// 9-tap Gaussian weights (sigma ~= 2.0)
const float weights[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

void main() {
    vec3 result = texture(uTexture, vTexCoord).rgb * weights[0];

    for (int i = 1; i < 5; i++) {
        vec2 offset = uDirection * uTexelSize * float(i);
        result += texture(uTexture, vTexCoord + offset).rgb * weights[i];
        result += texture(uTexture, vTexCoord - offset).rgb * weights[i];
    }

    fragColor = vec4(result, 1.0);
}
)glsl";

//=============================================================================
// Frequency Separation Gaussian Blur 프래그먼트 셰이더
// Separable 1D Gaussian — adaptive radius for low-frequency extraction
//=============================================================================
const char* FREQ_SEP_GAUSSIAN_FRAGMENT = R"glsl(
#version 310 es
precision highp float;

in vec2 vTexCoord;
out vec4 fragColor;

uniform sampler2D uTexture;
uniform vec2 uDirection;        // (1/w, 0) or (0, 1/h)
uniform int uRadius;            // adaptive radius (6~28)
uniform float uWeights[29];     // CPU-precomputed normalized half-kernel weights
uniform bool uLinearize;        // sRGB→Linear 변환 여부 (Pass 1a에서만 true)

void main() {
    vec3 sum = vec3(0.0);

    for (int i = -uRadius; i <= uRadius; i++) {
        vec2 offset = uDirection * float(i);
        vec3 s = texture(uTexture, vTexCoord + offset).rgb;
        if (uLinearize) {
            s = pow(s, vec3(2.2));
        }
        sum += s * uWeights[abs(i)];
    }

    fragColor = vec4(sum, 1.0);
}
)glsl";

//=============================================================================
// Frequency Separation Composite 프래그먼트 셰이더
// High Frequency inline extraction + non-linear attenuation + mask blending
//=============================================================================
const char* FREQ_SEP_COMPOSITE_FRAGMENT = R"glsl(
#version 310 es
precision highp float;

in vec2 vTexCoord;
out vec4 fragColor;

uniform sampler2D uSmoothedLow;   // Pass 2b result (additionally blurred low freq)
uniform sampler2D uLowFreq;       // Pass 1b result (original low freq — high freq extraction basis)
uniform sampler2D uOriginal;      // Original frame
uniform sampler2D uSkinMask;      // ROI mask texture

uniform float uHighFreqPreserve;   // Internal mapped value (0.1~1.0)
uniform float uAttenuationLow;     // smoothstep lower bound (default 0.02)
uniform float uAttenuationHigh;    // smoothstep upper bound (default 0.15)
uniform float uEdgeWeight;         // Edge preservation strength (0.3~0.7)
uniform float uChromaWeight;       // Chroma deviation sensitivity (0.2~0.5)
uniform float uToneLift;          // Mid-tone lift intensity (0.0~0.3, default 0.15)
uniform float uTextureBlendFloor; // textureBlend mix lower bound (default 0.38)
uniform int uDebugMode;           // 0=off, 1=magnitude heatmap, 2=compression heatmap, 3=mask
uniform int uSkinColorFilter;    // 0=off, 1=on (backend-local)

// Rec.709 luminance coefficients (linear-space)
const vec3 LUMA_709 = vec3(0.2126, 0.7152, 0.0722);
// Normalization scale factors — maps typical signal range to [0, 1]
const float EDGE_SCALE = 5.0;    // edge gradient range [0, ~0.2] → [0, 1]
const float CHROMA_SCALE = 10.0; // chroma deviation range [0, ~0.1] → [0, 1]

void main() {
    vec3 smoothLow = texture(uSmoothedLow, vTexCoord).rgb;
    vec3 low       = texture(uLowFreq, vTexCoord).rgb;
    vec3 orig      = texture(uOriginal, vTexCoord).rgb;
    orig = pow(orig, vec3(2.2));  // sRGB → Linear
    float mask     = texture(uSkinMask, vec2(vTexCoord.x, 1.0 - vTexCoord.y)).r;

    // --- Phase C: Color-space skin likelihood ---
    float rawMask = mask;

    vec3 gammaRGB = pow(max(orig, vec3(0.0)), vec3(1.0 / 2.2));
    float lum = dot(gammaRGB, vec3(0.299, 0.587, 0.114));
    float maxDev = max(max(abs(gammaRGB.r - lum), abs(gammaRGB.g - lum)),
                       abs(gammaRGB.b - lum));

    float darkLikelihood = smoothstep(0.08, 0.22, lum);
    float chromaLikelihood = smoothstep(0.02, 0.08, maxDev);
    float skinLikelihood = max(max(darkLikelihood, chromaLikelihood), 0.05);

    if (uSkinColorFilter != 0) {
        mask *= skinLikelihood;
    }

    // High Frequency inline extraction (ALU operation, no separate pass/texture)
    vec3 high = orig - low;

    // === Signal 1: Y(luminance) based high-freq magnitude — Rec.709 (linear-light 기준) ===
    float magnitude = dot(abs(high), LUMA_709);

    // === Signal 2: Edge Gradient (center difference, gamma 2.0 근사 linearize) ===
    // pow(x, 2.2) 대신 x*x (gamma 2.0)로 근사 linearize — 오차 ~5%, 4×pow 절약
    // linear 공간에서 에지 검출하여 magnitude/chromaDev와 색공간 통일
    vec2 texelSize = vec2(1.0) / vec2(textureSize(uOriginal, 0));
    vec3 sR = texture(uOriginal, vTexCoord + vec2( texelSize.x, 0.0)).rgb;
    vec3 sL = texture(uOriginal, vTexCoord + vec2(-texelSize.x, 0.0)).rgb;
    vec3 sU = texture(uOriginal, vTexCoord + vec2(0.0,  texelSize.y)).rgb;
    vec3 sD = texture(uOriginal, vTexCoord + vec2(0.0, -texelSize.y)).rgb;
    float lumR = dot(sR * sR, LUMA_709);
    float lumL = dot(sL * sL, LUMA_709);
    float lumU = dot(sU * sU, LUMA_709);
    float lumD = dot(sD * sD, LUMA_709);
    float gx = lumR - lumL;
    float gy = lumU - lumD;
    float edgeStrength = sqrt(gx * gx + gy * gy);

    // === Signal 3: Chroma Deviation (YCbCr UV separation) ===
    // high(=orig-low)의 luminance 성분을 제거하여 순수 색차만 추출
    float lumHigh = dot(high, LUMA_709);
    vec3 chromaDiff = high - vec3(lumHigh);
    float chromaDev = length(chromaDiff);

    // === 3-Signal Combination ===
    // 목표는 "큰 잡티 제거"가 아니라 "미세 피부결 압축"이다.
    // 따라서 작은/중간 고주파만 선택적으로 눌러주고,
    // 큰 점, 털, 진한 그림자 경계는 magnitude 상단 구간에서 다시 보호한다.
    float microTextureBand = smoothstep(uAttenuationLow * 0.55, uAttenuationHigh * 1.35, magnitude);
    float largeDetailProtection = smoothstep(uAttenuationHigh * 4.8, uAttenuationHigh * 9.5, magnitude);
    float edgeProtection = 1.0 - uEdgeWeight * clamp(edgeStrength * EDGE_SCALE * 0.68, 0.0, 1.0);
    float chromaProtection = 1.0 - uChromaWeight * clamp(chromaDev * CHROMA_SCALE * 0.66, 0.0, 1.0);
    float largeDetailCompression = mix(1.0, 0.38, largeDetailProtection);
    float poreCompression = microTextureBand
                          * largeDetailCompression
                          * clamp(edgeProtection, 0.0, 1.0)
                          * clamp(chromaProtection, 0.0, 1.0);

    // Deep shadow / highlight 보호:
    // 수염, 콧망울 그림자, 턱 그림자처럼 어두운 영역과 강한 반사광 영역은
    // beauty 결과를 덜 섞어 판화처럼 뭉개지는 현상을 줄인다.
    float baseLum = dot(smoothLow, LUMA_709);
    float shadowProtection = smoothstep(0.04, 0.14, baseLum);
    float highlightProtection = 1.0 - smoothstep(0.62, 0.88, baseLum) * 0.22;
    float effectStrength = clamp(shadowProtection * highlightProtection, 0.0, 1.0);

    // Compression only affects the selected micro-texture band.
    float compression = poreCompression * effectStrength;
    float preserve = mix(1.0, uHighFreqPreserve, compression);

    vec3 adjusted_high = high * preserve;

    // Direct foundation-style recomposition:
    // use the smoother low-frequency base, then re-inject protected detail.
    vec3 foundationBase = clamp(smoothLow + adjusted_high, 0.0, 1.0);

    // Texture blend: max of (pore-driven compression blend) and (smooth floor).
    // 매끈하게 floor에도 에지/그림자/색차 보호를 적용하여 경계가 무너지지 않게 함.
    float poreBlend = mix(0.38, 0.88, compression);
    float smoothFloorProtected = uTextureBlendFloor
                               * clamp(edgeProtection, 0.0, 1.0)
                               * clamp(chromaProtection, 0.0, 1.0)
                               * effectStrength;
    float textureBlend = mask * max(poreBlend, smoothFloorProtected);
    vec3 textureFinished = mix(orig, foundationBase, textureBlend);

    // Tone finish rides on top of the texture-compressed base so the result
    // feels like slight makeup, not fog.
    vec3 toneFinish = textureFinished
                    + (uToneLift * effectStrength) * textureFinished * (vec3(1.0) - textureFinished);

    // Tone finish is applied broadly within the face ROI.
    vec3 result = mix(textureFinished, toneFinish, mask * (0.65 * effectStrength));

    // === Debug visualization ===
    if (uDebugMode == 1) {
        // Magnitude heatmap (확대: 0~0.04 범위 → 모공 대역에 집중)
        float v = clamp(magnitude / 0.04, 0.0, 1.0);
        vec3 heatmap = vec3(smoothstep(0.3, 0.7, v), smoothstep(0.0, 0.5, v) - smoothstep(0.7, 1.0, v), 0.0);
        result = mix(orig, heatmap, mask * 0.8);
        result = pow(max(result, vec3(0.0)), vec3(1.0 / 2.2));
        fragColor = vec4(result, 1.0);
        return;
    } else if (uDebugMode == 2) {
        // microTextureBand heatmap (compression의 첫 단계)
        float v = clamp(microTextureBand, 0.0, 1.0);
        vec3 heatmap = vec3(v, 1.0 - v, 0.0);
        result = mix(orig, heatmap, mask * 0.8);
        result = pow(max(result, vec3(0.0)), vec3(1.0 / 2.2));
        fragColor = vec4(result, 1.0);
        return;
    } else if (uDebugMode == 3) {
        // edgeProtection heatmap (1=보호안함 → 초록, 0=완전보호 → 빨강)
        float v = clamp(edgeProtection, 0.0, 1.0);
        vec3 heatmap = vec3(1.0 - v, v, 0.0);
        result = mix(orig, heatmap, mask * 0.8);
        result = pow(max(result, vec3(0.0)), vec3(1.0 / 2.2));
        fragColor = vec4(result, 1.0);
        return;
    } else if (uDebugMode == 4) {
        // effectStrength heatmap (그림자/하이라이트 보호)
        float v = clamp(effectStrength, 0.0, 1.0);
        vec3 heatmap = vec3(v, v, 0.0);
        result = mix(orig, heatmap, mask * 0.8);
        result = pow(max(result, vec3(0.0)), vec3(1.0 / 2.2));
        fragColor = vec4(result, 1.0);
        return;
    } else if (uDebugMode == 5) {
        // 최종 compression heatmap (모든 요소 곱한 결과)
        float v = clamp(compression * 3.0, 0.0, 1.0); // 3x 증폭해서 미세한 차이도 보이게
        vec3 heatmap = vec3(v, 1.0 - v, 0.0);
        result = mix(orig, heatmap, mask * 0.8);
        result = pow(max(result, vec3(0.0)), vec3(1.0 / 2.2));
        fragColor = vec4(result, 1.0);
        return;
    } else if (uDebugMode == 6) {
        // Mask visualization
        result = mix(orig, vec3(0.0, mask, 0.0), 0.5);
        result = pow(max(result, vec3(0.0)), vec3(1.0 / 2.2));
        fragColor = vec4(result, 1.0);
        return;
    } else if (uDebugMode == 7) {
        // Skin color likelihood heatmap (rawMask for overlay, not filtered mask)
        float v = clamp(skinLikelihood, 0.0, 1.0);
        vec3 heatmap = vec3(1.0 - v, v, 0.0);
        result = mix(orig, heatmap, rawMask * 0.8);
        result = pow(max(result, vec3(0.0)), vec3(1.0 / 2.2));
        fragColor = vec4(result, 1.0);
        return;
    }

    // Linear → sRGB (음수 방어: pow(음수, 비정수)는 GLSL undefined behavior)
    result = pow(max(result, vec3(0.0)), vec3(1.0 / 2.2));

    fragColor = vec4(result, 1.0);
}
)glsl";

//=============================================================================
// Luminance Sharpen 프래그먼트 셰이더
// Luminance-only Unsharp Mask — FreqSep 파이프라인 마지막에 적용
//=============================================================================
const char* LUMINANCE_SHARPEN_FRAGMENT = R"glsl(
#version 310 es
precision highp float;

in vec2 vTexCoord;
out vec4 fragColor;

uniform sampler2D uTexture;      // Composite 결과 (beauty)
uniform sampler2D uSkinMask;     // ROI mask
uniform float uSharpenAmount;    // 샤프닝 강도 (0.0~0.5, 기본 0.15)
uniform vec2 uTexelSize;         // (1/width, 1/height)

void main() {
    vec3 center = texture(uTexture, vTexCoord).rgb;
    float mask = texture(uSkinMask, vec2(vTexCoord.x, 1.0 - vTexCoord.y)).r;

    const vec3 LUMA_709 = vec3(0.2126, 0.7152, 0.0722);
    float lumCenter = dot(center, LUMA_709);

    // 인접 픽셀의 mask 값 샘플링 (Y-flip 적용)
    float maskL = texture(uSkinMask, vec2(vTexCoord.x - uTexelSize.x, 1.0 - vTexCoord.y)).r;
    float maskR = texture(uSkinMask, vec2(vTexCoord.x + uTexelSize.x, 1.0 - vTexCoord.y)).r;
    float maskU = texture(uSkinMask, vec2(vTexCoord.x, 1.0 - (vTexCoord.y - uTexelSize.y))).r;
    float maskD = texture(uSkinMask, vec2(vTexCoord.x, 1.0 - (vTexCoord.y + uTexelSize.y))).r;

    // 인접 luminance 샘플링
    float rawLumL = dot(texture(uTexture, vTexCoord - vec2(uTexelSize.x, 0.0)).rgb, LUMA_709);
    float rawLumR = dot(texture(uTexture, vTexCoord + vec2(uTexelSize.x, 0.0)).rgb, LUMA_709);
    float rawLumU = dot(texture(uTexture, vTexCoord - vec2(0.0, uTexelSize.y)).rgb, LUMA_709);
    float rawLumD = dot(texture(uTexture, vTexCoord + vec2(0.0, uTexelSize.y)).rgb, LUMA_709);

    // mask가 0인(비피부) 인접 픽셀은 center luminance로 대체하여
    // composite 경계의 합성 에지가 unsharp mask에 반응하지 않도록 함
    float lumL = mix(lumCenter, rawLumL, maskL);
    float lumR = mix(lumCenter, rawLumR, maskR);
    float lumU = mix(lumCenter, rawLumU, maskU);
    float lumD = mix(lumCenter, rawLumD, maskD);

    float lumBlur = (lumCenter * 2.0 + lumL + lumR + lumU + lumD) / 6.0;

    float lumSharp = lumCenter + uSharpenAmount * (lumCenter - lumBlur);
    lumSharp = clamp(lumSharp, 0.0, 1.0);

    float ratio = (lumCenter > 0.001) ? min(lumSharp / lumCenter, 2.0) : 1.0;
    vec3 sharpened = center * ratio;
    sharpened = clamp(sharpened, 0.0, 1.0);

    vec3 result = mix(center, sharpened, mask);

    fragColor = vec4(result, 1.0);
}
)glsl";

//=============================================================================
// Vivid 포스트프로세싱 프래그먼트 셰이더
// 화면 전체 화사한 효과: Vibrance + 밝기 리프트 + 웜톤 시프트
// 단일 패스, 텍스처 샘플 1회 + ALU 연산 위주 → ~0.3ms (MID tier)
//=============================================================================
const char* VIVID_POSTPROCESS_FRAGMENT = R"glsl(
#version 310 es
precision highp float;

uniform sampler2D uTexture;
uniform float uIntensity;    // 마스터 강도 (0.0~1.0)
uniform float uSaturation;   // Vibrance 채도 부스트 (0.0~1.0)
uniform float uBrightness;   // 밝기 리프트 (0.0~0.5)
uniform float uWarmth;       // 웜톤 시프트 (0.0~1.0)

in vec2 vTexCoord;
out vec4 fragColor;

void main() {
    vec4 color = texture(uTexture, vTexCoord);
    vec3 result = color.rgb;

    // 1. Vibrance: 저채도 영역 우선 부스트 (과포화 방지)
    if (uSaturation > 0.01) {
        float lum = dot(result, vec3(0.2126, 0.7152, 0.0722));
        float sat = max(max(result.r, result.g), result.b) - min(min(result.r, result.g), result.b);
        // 저채도일수록 부스트 강함
        float vibranceAmount = uSaturation * (1.0 - smoothstep(0.0, 0.4, sat));
        result = mix(vec3(lum), result, 1.0 + vibranceAmount);
    }

    // 2. 밝기 리프트: 미드톤 위주 소프트 커브 (하이라이트 클리핑 방지)
    if (uBrightness > 0.001) {
        result += uBrightness * result * (1.0 - result);
    }

    // 3. 웜톤 시프트: R/G 미세 증가 + B 미세 감소
    if (uWarmth > 0.01) {
        result.r += uWarmth * 0.04;
        result.g += uWarmth * 0.02;
        result.b -= uWarmth * 0.03;
    }

    // 마스터 intensity로 원본과 mix
    result = mix(color.rgb, result, uIntensity);

    fragColor = vec4(clamp(result, 0.0, 1.0), color.a);
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
