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
uniform int uContactShadow;
uniform float uShadowIntensity;
uniform float uMaxDetail;
uniform int uHighlightEnabled;

uniform int uUseEllipseMask;
uniform vec2 uLeftEyeEllipseCenter;
uniform vec3 uLeftEyeEllipseRadii;
uniform float uLeftEyeEllipseRot;
uniform vec2 uRightEyeEllipseCenter;
uniform vec3 uRightEyeEllipseRadii;
uniform float uRightEyeEllipseRot;

in vec2 vTexCoord;
out vec4 fragColor;

vec3 blendNormal(vec3 base, vec3 blend, float opacity) {
    return mix(base, blend, opacity);
}

vec3 blendMultiply(vec3 base, vec3 blend, float opacity) {
    return mix(base, base * blend, opacity);
}

vec3 blendScreen(vec3 base, vec3 blend, float opacity) {
    return mix(base, 1.0 - (1.0 - base) * (1.0 - blend), opacity);
}

vec3 blendOverlay(vec3 base, vec3 blend, float opacity) {
    vec3 result;
    for (int i = 0; i < 3; i++) {
        if (base[i] < 0.5) {
            result[i] = 2.0 * base[i] * blend[i];
        } else {
            result[i] = 1.0 - 2.0 * (1.0 - base[i]) * (1.0 - blend[i]);
        }
    }
    return mix(base, result, opacity);
}

vec3 toLinearFast(vec3 srgb) { return srgb * srgb; }
vec3 toSRGBFast(vec3 linear_color) { return sqrt(max(linear_color, vec3(0.0))); }

vec3 blendLuminanceTint(vec3 base, vec3 blend, float opacity) {
    float lum = dot(base, vec3(0.299, 0.587, 0.114));
    float scale = clamp(0.5 / max(0.1, uAvgIrisLum), 0.8, 2.5);
    vec3 tinted = blend * lum * scale;
    return mix(base, tinted, opacity);
}

vec3 blendLuminanceTintLinear(vec3 base, vec3 blend, float opacity) {
    vec3 baseL = toLinearFast(base);
    float lum = dot(baseL, vec3(0.2126, 0.7152, 0.0722));
    float avgLumLinear = uAvgIrisLum * uAvgIrisLum;
    float scale = clamp(0.5 / max(0.01, avgLumLinear), 0.8, 5.0);
    vec3 tinted = toLinearFast(blend) * lum * scale;
    vec3 result = mix(baseL, tinted, opacity);
    float realSpec = smoothstep(0.7, 0.95, lum);
    result = mix(result, baseL, realSpec);
    return toSRGBFast(result);
}

vec3 blendSoftLight(vec3 base, vec3 blend, float opacity) {
    vec3 lo = base - (1.0 - 2.0 * blend) * base * (1.0 - base);
    vec3 hi = base + (2.0 * blend - 1.0) * (sqrt(base) - base);
    vec3 result = mix(lo, hi, step(vec3(0.5), blend));
    return mix(base, result, opacity);
}

vec3 blendColorReplace(vec3 base, vec3 blend, float opacity, float maxDetail) {
    float lum = dot(base, vec3(0.299, 0.587, 0.114));
    float detail = lum / max(0.01, uAvgIrisLum);
    detail = clamp(detail, 0.2, maxDetail);
    vec3 colored = blend * detail;
    return mix(base, colored, opacity);
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

vec4 applyLens(vec4 camera, vec2 irisCenter, float irisRadius, float aspectRatio,
               float eyeTop, float eyeBottom,
               vec2 ellipseCenter, vec3 ellipseRadii, float ellipseRot) {
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
        float geomFactor = smoothstep(0.75, 1.0, irisEdgeDist);
        float colorFactor = calcScleraFactor(camera.rgb);
        float scleraFade = 1.0 - geomFactor * (0.5 + 0.5 * colorFactor);
        finalAlpha *= scleraFade;
    }

    float maxDetail = mix(uMaxDetail, 1.0, smoothstep(0.75, 1.0, irisEdgeDist));

    vec3 blended;
    if (uBlendMode == 0) {
        blended = blendNormal(camera.rgb, lens.rgb, finalAlpha);
    } else if (uBlendMode == 1) {
        blended = blendMultiply(camera.rgb, lens.rgb, finalAlpha);
    } else if (uBlendMode == 2) {
        blended = blendScreen(camera.rgb, lens.rgb, finalAlpha);
    } else if (uBlendMode == 3) {
        blended = blendOverlay(camera.rgb, lens.rgb, finalAlpha);
    } else if (uBlendMode == 4) {
        blended = blendLuminanceTint(camera.rgb, lens.rgb, finalAlpha);
    } else if (uBlendMode == 5) {
        blended = blendLuminanceTintLinear(camera.rgb, lens.rgb, finalAlpha);
    } else if (uBlendMode == 6) {
        blended = blendSoftLight(camera.rgb, lens.rgb, finalAlpha);
    } else if (uBlendMode == 7) {
        blended = blendColorReplace(camera.rgb, lens.rgb, finalAlpha, maxDetail);
    } else {
        blended = blendNormal(camera.rgb, lens.rgb, finalAlpha);
    }

    // 림발 다크닝: 홍채 외곽(r≈0.7~1.0)에 어두운 고리
    // 현재 비활성 — 대부분의 렌즈 텍스처에 이미 림발이 포함되어 이중 적용 방지
    const bool LIMBAL_ENABLED = false;
    if (LIMBAL_ENABLED) {
        float limbalDist = dist;
        float limbal = smoothstep(0.7, 1.0, limbalDist);
        blended = mix(blended, blended * 0.4, limbal * 0.8);
    }

    // 각막 하이라이트: 홍채 로컬 좌표 기반 (lensCoord 미사용 — mipmap gradient 회피)
    if (uHighlightEnabled == 1) {
        vec2 localDir = (adjustedCoord - adjustedCenter) / max(scaledRadius, 1e-5);
        vec2 highlightCenter = vec2(-0.3, 0.4);
        float highlightDist = distance(localDir, highlightCenter);
        float highlight = smoothstep(0.25, 0.0, highlightDist);
        blended = mix(blended, vec3(1.0), highlight * 0.5 * finalAlpha);
    }

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
                           uLeftEyeEllipseCenter, uLeftEyeEllipseRadii, uLeftEyeEllipseRot);
    }

    if (uApplyRight == 1 && uRightIrisRadius > 0.0) {
        result = applyLens(result, uRightIrisCenter, uRightIrisRadius, aspectRatio,
                           uRightEyeTop, uRightEyeBottom,
                           uRightEyeEllipseCenter, uRightEyeEllipseRadii, uRightEyeEllipseRot);
    }

    fragColor = result;
}
)glsl";

} // namespace shaders
} // namespace iris_sdk
