# P2-W3-02. GPU 필터 셰이더 구현

## 작업 정보

| 항목 | 내용 |
|------|------|
| **작업 ID** | P2-W3-02 |
| **Phase** | Phase 3: GPU 백엔드 구현 |
| **상태** | ⏳ 대기 |
| **예상 기간** | 3일 |
| **의존성** | P2-W3-01 (GPU 인프라) |
| **담당** | systems-programming:cpp-pro |

---

## 1. 목표

OpenGL ES 3.1 셰이더로 뷰티 필터 효과 구현

### 핵심 산출물
- 공통 버텍스 셰이더
- Bilateral/Guided Filter 셰이더
- 화이트닝 셰이더
- 컬러 밸런스 셰이더
- 소프트 포커스 셰이더
- 밝기 조정 셰이더
- 마스킹 블렌딩 셰이더

---

## 2. 셰이더 구현

### 2.1 공통 버텍스 셰이더

**파일**: `cpp/src/gpu/shaders/fullscreen_quad.vert`

```glsl
#version 310 es
precision highp float;

layout(location = 0) in vec2 a_Position;
layout(location = 1) in vec2 a_TexCoord;

out vec2 v_TexCoord;

void main() {
    gl_Position = vec4(a_Position, 0.0, 1.0);
    v_TexCoord = a_TexCoord;
}
```

### 2.2 Bilateral Filter 셰이더 (피부 스무딩)

**파일**: `cpp/src/gpu/shaders/bilateral_filter.frag`

```glsl
#version 310 es
precision highp float;

in vec2 v_TexCoord;
out vec4 fragColor;

uniform sampler2D u_Texture;
uniform vec2 u_TexelSize;      // 1.0 / texture_size
uniform float u_SigmaSpatial;  // 공간 시그마 (픽셀 단위)
uniform float u_SigmaRange;    // 범위 시그마 (색상 차이)
uniform float u_Strength;      // 효과 강도 (0.0 ~ 1.0)

const int KERNEL_SIZE = 9;  // -4 ~ +4

// 가우시안 가중치 계산
float gaussian(float x, float sigma) {
    return exp(-(x * x) / (2.0 * sigma * sigma));
}

// 색상 차이 기반 가중치
float colorWeight(vec3 c1, vec3 c2, float sigma) {
    vec3 diff = c1 - c2;
    float dist = dot(diff, diff);
    return exp(-dist / (2.0 * sigma * sigma));
}

void main() {
    vec3 centerColor = texture(u_Texture, v_TexCoord).rgb;

    vec3 sumColor = vec3(0.0);
    float sumWeight = 0.0;

    int halfKernel = KERNEL_SIZE / 2;

    for (int i = -halfKernel; i <= halfKernel; i++) {
        for (int j = -halfKernel; j <= halfKernel; j++) {
            vec2 offset = vec2(float(i), float(j)) * u_TexelSize;
            vec3 sampleColor = texture(u_Texture, v_TexCoord + offset).rgb;

            // 공간 가중치
            float spatialDist = length(vec2(float(i), float(j)));
            float spatialWeight = gaussian(spatialDist, u_SigmaSpatial);

            // 색상 가중치
            float rangeWeight = colorWeight(centerColor, sampleColor, u_SigmaRange);

            // 결합 가중치
            float weight = spatialWeight * rangeWeight;

            sumColor += sampleColor * weight;
            sumWeight += weight;
        }
    }

    vec3 filteredColor = sumColor / sumWeight;

    // 원본과 블렌딩 (강도 조절)
    vec3 result = mix(centerColor, filteredColor, u_Strength);

    fragColor = vec4(result, 1.0);
}
```

### 2.3 최적화된 분리형 Bilateral Filter

9x9 필터를 두 번의 1D 패스로 분리하여 성능 향상

**파일**: `cpp/src/gpu/shaders/bilateral_horizontal.frag`

```glsl
#version 310 es
precision highp float;

in vec2 v_TexCoord;
out vec4 fragColor;

uniform sampler2D u_Texture;
uniform vec2 u_TexelSize;
uniform float u_SigmaSpatial;
uniform float u_SigmaRange;

const int KERNEL_RADIUS = 4;

void main() {
    vec3 centerColor = texture(u_Texture, v_TexCoord).rgb;

    vec3 sumColor = vec3(0.0);
    float sumWeight = 0.0;

    for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++) {
        vec2 offset = vec2(float(i) * u_TexelSize.x, 0.0);
        vec3 sampleColor = texture(u_Texture, v_TexCoord + offset).rgb;

        float spatialWeight = exp(-float(i * i) / (2.0 * u_SigmaSpatial * u_SigmaSpatial));

        vec3 colorDiff = centerColor - sampleColor;
        float colorDist = dot(colorDiff, colorDiff);
        float rangeWeight = exp(-colorDist / (2.0 * u_SigmaRange * u_SigmaRange));

        float weight = spatialWeight * rangeWeight;
        sumColor += sampleColor * weight;
        sumWeight += weight;
    }

    fragColor = vec4(sumColor / sumWeight, 1.0);
}
```

**파일**: `cpp/src/gpu/shaders/bilateral_vertical.frag`

```glsl
#version 310 es
precision highp float;

in vec2 v_TexCoord;
out vec4 fragColor;

uniform sampler2D u_Texture;
uniform vec2 u_TexelSize;
uniform float u_SigmaSpatial;
uniform float u_SigmaRange;
uniform float u_Strength;

const int KERNEL_RADIUS = 4;

uniform sampler2D u_OriginalTexture;  // 원본 (블렌딩용)

void main() {
    vec3 centerColor = texture(u_Texture, v_TexCoord).rgb;

    vec3 sumColor = vec3(0.0);
    float sumWeight = 0.0;

    for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++) {
        vec2 offset = vec2(0.0, float(i) * u_TexelSize.y);
        vec3 sampleColor = texture(u_Texture, v_TexCoord + offset).rgb;

        float spatialWeight = exp(-float(i * i) / (2.0 * u_SigmaSpatial * u_SigmaSpatial));

        vec3 colorDiff = centerColor - sampleColor;
        float colorDist = dot(colorDiff, colorDiff);
        float rangeWeight = exp(-colorDist / (2.0 * u_SigmaRange * u_SigmaRange));

        float weight = spatialWeight * rangeWeight;
        sumColor += sampleColor * weight;
        sumWeight += weight;
    }

    vec3 filteredColor = sumColor / sumWeight;
    vec3 originalColor = texture(u_OriginalTexture, v_TexCoord).rgb;

    // 강도 블렌딩
    vec3 result = mix(originalColor, filteredColor, u_Strength);

    fragColor = vec4(result, 1.0);
}
```

### 2.4 화이트닝 셰이더

LAB 색상 공간에서 L 채널 조정 (GPU에서는 근사 계산)

**파일**: `cpp/src/gpu/shaders/whitening.frag`

```glsl
#version 310 es
precision highp float;

in vec2 v_TexCoord;
out vec4 fragColor;

uniform sampler2D u_Texture;
uniform float u_Strength;  // 0.0 ~ 1.0

// sRGB → Linear RGB
vec3 srgbToLinear(vec3 srgb) {
    return pow(srgb, vec3(2.2));
}

// Linear RGB → sRGB
vec3 linearToSrgb(vec3 linear) {
    return pow(linear, vec3(1.0 / 2.2));
}

// RGB → LAB (근사 변환)
vec3 rgbToLab(vec3 rgb) {
    // XYZ 변환
    mat3 rgb2xyz = mat3(
        0.4124564, 0.3575761, 0.1804375,
        0.2126729, 0.7151522, 0.0721750,
        0.0193339, 0.1191920, 0.9503041
    );

    vec3 xyz = rgb2xyz * srgbToLinear(rgb);

    // D65 기준
    xyz /= vec3(0.95047, 1.0, 1.08883);

    // f(t) 변환
    vec3 f = mix(
        7.787 * xyz + 16.0 / 116.0,
        pow(xyz, vec3(1.0 / 3.0)),
        step(vec3(0.008856), xyz)
    );

    float L = 116.0 * f.y - 16.0;
    float a = 500.0 * (f.x - f.y);
    float b = 200.0 * (f.y - f.z);

    return vec3(L, a, b);
}

// LAB → RGB (근사 변환)
vec3 labToRgb(vec3 lab) {
    float L = lab.x;
    float a = lab.y;
    float b = lab.z;

    float fy = (L + 16.0) / 116.0;
    float fx = a / 500.0 + fy;
    float fz = fy - b / 200.0;

    vec3 f = vec3(fx, fy, fz);

    vec3 xyz = mix(
        (f - 16.0 / 116.0) / 7.787,
        f * f * f,
        step(vec3(0.206893), f)
    );

    // D65 기준
    xyz *= vec3(0.95047, 1.0, 1.08883);

    // XYZ → RGB
    mat3 xyz2rgb = mat3(
         3.2404542, -1.5371385, -0.4985314,
        -0.9692660,  1.8760108,  0.0415560,
         0.0556434, -0.2040259,  1.0572252
    );

    vec3 rgb = xyz2rgb * xyz;
    return linearToSrgb(clamp(rgb, 0.0, 1.0));
}

// 피부톤 감지 (LAB 공간)
float detectSkinTone(vec3 lab) {
    // A: 125-175 범위 (붉은 톤)
    // B: 130-180 범위 (노란 톤)
    float a_norm = (lab.y + 128.0) / 255.0;  // -128~128 → 0~1
    float b_norm = (lab.z + 128.0) / 255.0;

    float a_in_range = step(0.49, a_norm) * step(a_norm, 0.69);
    float b_in_range = step(0.51, b_norm) * step(b_norm, 0.71);

    return a_in_range * b_in_range;
}

void main() {
    vec3 color = texture(u_Texture, v_TexCoord).rgb;

    // LAB 변환
    vec3 lab = rgbToLab(color);

    // 피부톤 영역만 처리
    float skinMask = detectSkinTone(lab);

    // L 채널 감마 보정 (밝기 증가)
    float gamma = 1.0 - u_Strength * 0.3;  // 0.7 ~ 1.0
    float L_normalized = lab.x / 100.0;
    float L_corrected = pow(L_normalized, gamma) * 100.0;

    // 피부톤 영역에만 적용
    lab.x = mix(lab.x, L_corrected, skinMask);

    // RGB 변환
    vec3 result = labToRgb(lab);

    fragColor = vec4(result, 1.0);
}
```

### 2.5 컬러 밸런스 셰이더

**파일**: `cpp/src/gpu/shaders/color_balance.frag`

```glsl
#version 310 es
precision highp float;

in vec2 v_TexCoord;
out vec4 fragColor;

uniform sampler2D u_Texture;
uniform float u_Balance;  // -1.0 (차가움) ~ +1.0 (따뜻함)

void main() {
    vec3 color = texture(u_Texture, v_TexCoord).rgb;

    // 따뜻한 톤: Red↑, Blue↓
    // 차가운 톤: Red↓, Blue↑
    float warmthShift = u_Balance * 0.1;

    vec3 result = color;
    result.r = clamp(color.r + warmthShift, 0.0, 1.0);
    result.b = clamp(color.b - warmthShift, 0.0, 1.0);

    // Green은 약간만 조정 (피부톤 보존)
    result.g = clamp(color.g + warmthShift * 0.3, 0.0, 1.0);

    fragColor = vec4(result, 1.0);
}
```

### 2.6 소프트 포커스 셰이더

**파일**: `cpp/src/gpu/shaders/soft_focus.frag`

```glsl
#version 310 es
precision highp float;

in vec2 v_TexCoord;
out vec4 fragColor;

uniform sampler2D u_Texture;
uniform sampler2D u_BlurTexture;  // 사전 블러 처리된 텍스처
uniform float u_Strength;

// Overlay 블렌드 모드
vec3 overlayBlend(vec3 base, vec3 blend) {
    vec3 result;
    for (int i = 0; i < 3; i++) {
        float b = base[i];
        float l = blend[i];
        result[i] = b < 0.5 ? 2.0 * b * l : 1.0 - 2.0 * (1.0 - b) * (1.0 - l);
    }
    return result;
}

void main() {
    vec3 original = texture(u_Texture, v_TexCoord).rgb;
    vec3 blurred = texture(u_BlurTexture, v_TexCoord).rgb;

    // Overlay 블렌딩
    vec3 overlay = overlayBlend(original, blurred);

    // 강도에 따라 원본과 블렌딩
    vec3 result = mix(original, overlay, u_Strength * 0.5);

    // 약간의 글로우 추가
    result = mix(result, blurred, u_Strength * 0.2);

    fragColor = vec4(result, 1.0);
}
```

### 2.7 가우시안 블러 셰이더 (소프트 포커스용)

**파일**: `cpp/src/gpu/shaders/gaussian_blur.frag`

```glsl
#version 310 es
precision highp float;

in vec2 v_TexCoord;
out vec4 fragColor;

uniform sampler2D u_Texture;
uniform vec2 u_Direction;  // (1,0) for horizontal, (0,1) for vertical
uniform vec2 u_TexelSize;
uniform float u_BlurSize;

// 9탭 가우시안 가중치
const float weights[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

void main() {
    vec3 result = texture(u_Texture, v_TexCoord).rgb * weights[0];

    for (int i = 1; i < 5; i++) {
        vec2 offset = u_Direction * u_TexelSize * float(i) * u_BlurSize;
        result += texture(u_Texture, v_TexCoord + offset).rgb * weights[i];
        result += texture(u_Texture, v_TexCoord - offset).rgb * weights[i];
    }

    fragColor = vec4(result, 1.0);
}
```

### 2.8 밝기 조정 셰이더

**파일**: `cpp/src/gpu/shaders/brightness.frag`

```glsl
#version 310 es
precision highp float;

in vec2 v_TexCoord;
out vec4 fragColor;

uniform sampler2D u_Texture;
uniform float u_Brightness;  // 0.5 ~ 1.5 (1.0 = 원본)

void main() {
    vec3 color = texture(u_Texture, v_TexCoord).rgb;

    vec3 result;

    if (u_Brightness > 1.0) {
        // 밝게: 비선형 증가 (하이라이트 보호)
        float factor = u_Brightness;
        result = color + (vec3(1.0) - color) * (factor - 1.0) * 0.5;
    } else {
        // 어둡게: 선형 감소
        result = color * u_Brightness;
    }

    fragColor = vec4(clamp(result, 0.0, 1.0), 1.0);
}
```

### 2.9 마스킹 블렌딩 셰이더

**파일**: `cpp/src/gpu/shaders/masking.frag`

```glsl
#version 310 es
precision highp float;

in vec2 v_TexCoord;
out vec4 fragColor;

uniform sampler2D u_FilteredTexture;  // 필터 적용된 텍스처
uniform sampler2D u_OriginalTexture;  // 원본 텍스처
uniform sampler2D u_MaskTexture;      // ROI/보호 마스크

void main() {
    vec3 filtered = texture(u_FilteredTexture, v_TexCoord).rgb;
    vec3 original = texture(u_OriginalTexture, v_TexCoord).rgb;
    float mask = texture(u_MaskTexture, v_TexCoord).r;

    // 마스크 영역: 필터 적용, 비마스크 영역: 원본
    vec3 result = mix(original, filtered, mask);

    fragColor = vec4(result, 1.0);
}
```

---

## 3. 셰이더 소스 관리

### 3.1 빌드 타임 변환

**파일**: `cpp/cmake/ConvertShaders.cmake`

```cmake
# 셰이더 파일을 C++ 문자열로 변환

set(SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/shaders")
set(OUTPUT_FILE "${CMAKE_CURRENT_BINARY_DIR}/shader_sources.cpp")

file(GLOB SHADER_FILES "${SHADER_DIR}/*.vert" "${SHADER_DIR}/*.frag")

file(WRITE ${OUTPUT_FILE} "// Auto-generated shader sources\n")
file(APPEND ${OUTPUT_FILE} "#include \"iris_sdk/gpu/shader_manager.h\"\n\n")
file(APPEND ${OUTPUT_FILE} "namespace iris_sdk {\n")
file(APPEND ${OUTPUT_FILE} "namespace shaders {\n\n")

foreach(SHADER_FILE ${SHADER_FILES})
    get_filename_component(SHADER_NAME ${SHADER_FILE} NAME_WE)
    string(TOUPPER ${SHADER_NAME} SHADER_NAME_UPPER)

    # .vert → _VERTEX, .frag → _FRAGMENT
    get_filename_component(SHADER_EXT ${SHADER_FILE} EXT)
    if(SHADER_EXT STREQUAL ".vert")
        set(SUFFIX "_VERTEX")
    else()
        set(SUFFIX "_FRAGMENT")
    endif()

    file(READ ${SHADER_FILE} SHADER_CONTENT)

    # 이스케이프 처리
    string(REPLACE "\\" "\\\\" SHADER_CONTENT "${SHADER_CONTENT}")
    string(REPLACE "\"" "\\\"" SHADER_CONTENT "${SHADER_CONTENT}")
    string(REPLACE "\n" "\\n\"\n\"" SHADER_CONTENT "${SHADER_CONTENT}")

    file(APPEND ${OUTPUT_FILE} "const char* ${SHADER_NAME_UPPER}${SUFFIX} = \n\"${SHADER_CONTENT}\";\n\n")
endforeach()

file(APPEND ${OUTPUT_FILE} "} // namespace shaders\n")
file(APPEND ${OUTPUT_FILE} "} // namespace iris_sdk\n")
```

### 3.2 생성된 shader_sources.cpp 예시

```cpp
// Auto-generated shader sources
#include "iris_sdk/gpu/shader_manager.h"

namespace iris_sdk {
namespace shaders {

const char* FULLSCREEN_QUAD_VERTEX =
"#version 310 es\n"
"precision highp float;\n"
"\n"
"layout(location = 0) in vec2 a_Position;\n"
"layout(location = 1) in vec2 a_TexCoord;\n"
"\n"
"out vec2 v_TexCoord;\n"
"\n"
"void main() {\n"
"    gl_Position = vec4(a_Position, 0.0, 1.0);\n"
"    v_TexCoord = a_TexCoord;\n"
"}\n";

const char* BILATERAL_FILTER_FRAGMENT =
// ... 전체 소스 ...
;

// ... 나머지 셰이더들 ...

} // namespace shaders
} // namespace iris_sdk
```

---

## 4. GPUBeautyBackend 필터 패스 구현

**파일**: `cpp/src/gpu/gpu_beauty_backend.cpp` (확장)

```cpp
void GPUBeautyBackend::executeSmoothingPass(
    GLuint input_tex, GLuint output_fbo,
    int width, int height,
    const BeautyFilterConfigV2& config) {

    // 분리형 Bilateral Filter (2-Pass)
    // Pass 1: Horizontal
    TexturePool::TextureInfo* temp = texture_pool_->acquireRenderTarget(width, height);

    glBindFramebuffer(GL_FRAMEBUFFER, temp->fbo_id);
    glViewport(0, 0, width, height);

    glUseProgram(bilateral_h_program_);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, input_tex);
    glUniform1i(glGetUniformLocation(bilateral_h_program_, "u_Texture"), 0);

    glUniform2f(glGetUniformLocation(bilateral_h_program_, "u_TexelSize"),
                1.0f / width, 1.0f / height);
    glUniform1f(glGetUniformLocation(bilateral_h_program_, "u_SigmaSpatial"),
                3.0f + config.smoothing * 3.0f);  // 3 ~ 6
    glUniform1f(glGetUniformLocation(bilateral_h_program_, "u_SigmaRange"),
                0.1f + config.smoothing * 0.2f);  // 0.1 ~ 0.3

    renderFullscreenQuad();

    // Pass 2: Vertical
    glBindFramebuffer(GL_FRAMEBUFFER, output_fbo);

    glUseProgram(bilateral_v_program_);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, temp->texture_id);
    glUniform1i(glGetUniformLocation(bilateral_v_program_, "u_Texture"), 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, input_tex);  // 원본 (블렌딩용)
    glUniform1i(glGetUniformLocation(bilateral_v_program_, "u_OriginalTexture"), 1);

    glUniform2f(glGetUniformLocation(bilateral_v_program_, "u_TexelSize"),
                1.0f / width, 1.0f / height);
    glUniform1f(glGetUniformLocation(bilateral_v_program_, "u_SigmaSpatial"),
                3.0f + config.smoothing * 3.0f);
    glUniform1f(glGetUniformLocation(bilateral_v_program_, "u_SigmaRange"),
                0.1f + config.smoothing * 0.2f);
    glUniform1f(glGetUniformLocation(bilateral_v_program_, "u_Strength"),
                config.smoothing);

    renderFullscreenQuad();

    texture_pool_->releaseTexture(temp);
}

void GPUBeautyBackend::executeWhiteningPass(
    GLuint input_tex, GLuint output_fbo,
    int width, int height,
    float whitening) {

    glBindFramebuffer(GL_FRAMEBUFFER, output_fbo);
    glViewport(0, 0, width, height);

    glUseProgram(whitening_program_);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, input_tex);
    glUniform1i(glGetUniformLocation(whitening_program_, "u_Texture"), 0);

    glUniform1f(glGetUniformLocation(whitening_program_, "u_Strength"), whitening);

    renderFullscreenQuad();
}

void GPUBeautyBackend::executeSoftFocusPass(
    GLuint input_tex, GLuint output_fbo,
    int width, int height,
    float strength) {

    // 먼저 가우시안 블러 생성 (2-pass)
    TexturePool::TextureInfo* blur_h = texture_pool_->acquireRenderTarget(width, height);
    TexturePool::TextureInfo* blur_v = texture_pool_->acquireRenderTarget(width, height);

    float blur_size = 5.0f + strength * 10.0f;

    // Horizontal blur
    glBindFramebuffer(GL_FRAMEBUFFER, blur_h->fbo_id);
    glViewport(0, 0, width, height);
    glUseProgram(gaussian_blur_program_);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, input_tex);
    glUniform1i(glGetUniformLocation(gaussian_blur_program_, "u_Texture"), 0);
    glUniform2f(glGetUniformLocation(gaussian_blur_program_, "u_Direction"), 1.0f, 0.0f);
    glUniform2f(glGetUniformLocation(gaussian_blur_program_, "u_TexelSize"),
                1.0f / width, 1.0f / height);
    glUniform1f(glGetUniformLocation(gaussian_blur_program_, "u_BlurSize"), blur_size);

    renderFullscreenQuad();

    // Vertical blur
    glBindFramebuffer(GL_FRAMEBUFFER, blur_v->fbo_id);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, blur_h->texture_id);
    glUniform2f(glGetUniformLocation(gaussian_blur_program_, "u_Direction"), 0.0f, 1.0f);

    renderFullscreenQuad();

    // Soft focus 블렌딩
    glBindFramebuffer(GL_FRAMEBUFFER, output_fbo);
    glUseProgram(soft_focus_program_);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, input_tex);
    glUniform1i(glGetUniformLocation(soft_focus_program_, "u_Texture"), 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, blur_v->texture_id);
    glUniform1i(glGetUniformLocation(soft_focus_program_, "u_BlurTexture"), 1);

    glUniform1f(glGetUniformLocation(soft_focus_program_, "u_Strength"), strength);

    renderFullscreenQuad();

    texture_pool_->releaseTexture(blur_h);
    texture_pool_->releaseTexture(blur_v);
}
```

---

## 5. 단위 테스트

**파일**: `cpp/tests/test_gpu_shaders.cpp`

```cpp
#if defined(__ANDROID__) && defined(IRIS_SDK_HAS_GLES)

TEST_F(GPUBeautyBackendTest, SmoothingShaderWorks) {
    // ... 테스트 이미지 준비 ...

    BeautyFilterConfigV2 config;
    config.enabled = true;
    config.smoothing = 0.8f;

    TextureHandle output;
    EXPECT_EQ(backend_->applyTexture(input, output, config, nullptr), IRIS_SDK_OK);

    // GPU → CPU 읽기
    std::vector<uint8_t> result_data(width * height * 4);
    render_context_->downloadTexture(output, result_data.data());

    // 분산 감소 확인 (스무딩 효과)
    // ...
}

TEST_F(GPUBeautyBackendTest, WhiteningIncreasesLuminance) {
    // 어두운 이미지로 테스트
    // 화이트닝 후 평균 밝기 증가 확인
}

#endif
```

---

## 6. 완료 기준

- [ ] 공통 버텍스 셰이더 구현
- [ ] Bilateral Filter 셰이더 (분리형) 구현
- [ ] 화이트닝 셰이더 (LAB 근사) 구현
- [ ] 컬러 밸런스 셰이더 구현
- [ ] 소프트 포커스 + 가우시안 블러 셰이더 구현
- [ ] 밝기 조정 셰이더 구현
- [ ] 마스킹 블렌딩 셰이더 구현
- [ ] 셰이더 빌드 타임 변환 스크립트
- [ ] 단위 테스트 통과

---

## 7. 다음 작업

- **P2-W4-01**: Face Warp 구현 (Grid Mesh 기반)
