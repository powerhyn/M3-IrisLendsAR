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
    float sigmaColor = 0.1 + uStrength * 0.2;  // 색상 시그마 (0.1~0.3)

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

    // 밝기(Y) 증가 + 채도(Cb, Cr) 감소로 화이트닝 효과
    float luminanceBoost = 1.0 + uStrength * 0.2;  // 최대 20% 밝기 증가
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

        // 밝기(Y) 증가 + 채도(Cb, Cr) 감소로 화이트닝 효과
        float luminanceBoost = 1.0 + uWhitening * 0.2;  // 최대 20% 밝기 증가
        float saturationReduce = 1.0 - uWhitening * 0.15;  // 최대 15% 채도 감소

        ycbcr.x = min(ycbcr.x * luminanceBoost, 1.0);
        ycbcr.y = mix(0.5, ycbcr.y, saturationReduce);
        ycbcr.z = mix(0.5, ycbcr.z, saturationReduce);

        // RGB 변환
        result = ycbcr2rgb(ycbcr);
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

} // namespace shaders
} // namespace iris_sdk
