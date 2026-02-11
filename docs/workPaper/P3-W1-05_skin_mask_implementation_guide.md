# P3-W1-05: SkinMaskRenderer C++ 구현 상세 가이드

> **작성자**: ml-vision
> **상태**: ✅ 완료
> **관련 태스크**: Task #21, #22
> **참조 문서**: P3-W1-04 섹션 4 (기본 설계), P3-W1-02 (Face Mesh 모델 스펙)
> **구현 담당**: cpp-engine

---

## 목차

1. [개요](#1-개요)
2. [피부 영역 랜드마크 인덱스](#2-피부-영역-랜드마크-인덱스)
3. [buildTriangleFan() - Face Oval EBO 인덱스 배열](#3-buildtrianglefan---face-oval-ebo-인덱스-배열)
4. [buildExclusionPolygons() - 눈/입술 삼각형 분할](#4-buildexclusionpolygons---눈입술-삼각형-분할)
5. [Gaussian Blur Pass - 경계 스무딩 셰이더](#5-gaussian-blur-pass---경계-스무딩-셰이더)
6. [FBO 해상도 전략](#6-fbo-해상도-전략)
7. [마스크 업데이트 빈도 전략](#7-마스크-업데이트-빈도-전략)
8. [renderSkinMask() 전체 파이프라인](#8-renderskinmask-전체-파이프라인)
9. [C++ 헤더 (skin_mask_renderer.h)](#9-c-헤더)
10. [GPUBeautyBackend 통합](#10-gpubeautybackend-통합)
11. [메모리 및 성능 예산](#11-메모리-및-성능-예산)
12. [테스트 방안](#12-테스트-방안)

---

## 1. 개요

### 1.1 목적

Face Mesh 478점 랜드마크를 활용하여 **ML 모델 없이** GPU에서 직접 피부 마스크를 생성한다.
뷰티 필터(Smoothing, Whitening 등)를 피부 영역에만 선택적으로 적용하여 눈/입술의 자연스러움을 보존한다.

### 1.2 접근 방식

```
GPU 직접 렌더링 방식 (CPU 마스크 생성 대비 선택)

장점:
- GPU→CPU→GPU 왕복 없음 (zero-copy)
- 마스크 해상도 유연 (1/2, 1/4 등)
- Beauty 파이프라인 FBO 체인에 자연스럽게 통합

구현:
1. Face Oval 36점 → Triangle Fan (흰색, mask=1.0)
2. 눈/입술 → Triangle Fan 덮어쓰기 (검은색, mask=0.0)
3. Separable Gaussian Blur 2-pass (경계 스무딩)
4. Beauty 셰이더에서 마스크 텍스처 샘플링
```

### 1.3 코드베이스 참조

| 참조 대상 | 파일 | 라인 |
|-----------|------|------|
| Face Oval 인덱스 (36점) | `cpp/src/beauty_roi_manager.cpp` | 19-24 |
| Left/Right Eye 인덱스 | `cpp/src/beauty_roi_manager.cpp` | 27-36 |
| Lips 인덱스 | `cpp/src/beauty_roi_manager.cpp` | 39-42 |
| Fullscreen Quad VAO/VBO 패턴 | `cpp/src/gpu/gpu_beauty_backend.cpp` | 241-281 |
| FBO 생성 패턴 | `cpp/src/gpu/texture_pool.cpp` | 315-346 |
| 렌더 패스 실행 패턴 | `cpp/src/gpu/gpu_beauty_backend.cpp` | 524-541 |
| Gaussian Blur 셰이더 | `cpp/src/gpu/shader_sources.cpp` | 384-409 |
| 이전 프레임 얼굴 추적 | `cpp/src/mediapipe_detector.cpp` | 188-190 |

---

## 2. 피부 영역 랜드마크 인덱스

`beauty_roi_manager.cpp`에 이미 정의된 인덱스를 사용한다.

| 영역 | 개수 | 인덱스 |
|------|------|--------|
| **Face Oval** | 36 | 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109 |
| **Left Eye** | 16 | 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246 |
| **Right Eye** | 16 | 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398 |
| **Lips (inner)** | 22 | 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78 |
| **Left Eyebrow** | 8 | 70, 63, 105, 66, 107, 55, 65, 52 |
| **Right Eyebrow** | 8 | 300, 293, 334, 296, 336, 285, 295, 282 |

---

## 3. buildTriangleFan() - Face Oval EBO 인덱스 배열

### 3.1 정점 레이아웃

Face Oval 36점에 중심점(centroid)을 추가하여 **37개 정점, 36개 삼각형**을 생성한다.

```
정점 배열 구조 (VBO, stride = 3 floats):
  index 0:  centroid (36점 무게중심)   mask_value = 1.0
  index 1:  oval[0]  = landmark[10]    mask_value = 1.0
  index 2:  oval[1]  = landmark[338]   mask_value = 1.0
  index 3:  oval[2]  = landmark[297]   mask_value = 1.0
  ...
  index 36: oval[35] = landmark[109]   mask_value = 1.0
```

### 3.2 구현 코드

```cpp
void SkinMaskRenderer::buildTriangleFan(
    const IrisLandmark* face_mesh,
    std::vector<float>& vertices,
    std::vector<uint16_t>& indices) {

    // Face Oval 36점 인덱스 (beauty_roi_manager.cpp line 19-24)
    static constexpr int FACE_OVAL_INDICES[36] = {
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
        361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
        176, 149, 150, 136, 172,  58, 132,  93, 234, 127,
        162,  21,  54, 103,  67, 109
    };

    // 1. Centroid 계산
    float cx = 0.0f, cy = 0.0f;
    for (int i = 0; i < 36; ++i) {
        int idx = FACE_OVAL_INDICES[i];
        cx += face_mesh[idx].x;
        cy += face_mesh[idx].y;
    }
    cx /= 36.0f;
    cy /= 36.0f;

    // 2. 정점 배열 생성 (stride: x, y, mask_value = 3 floats)
    vertices.clear();
    vertices.reserve(37 * 3);  // 37 vertices * 3 components

    // index 0: centroid
    vertices.push_back(cx);
    vertices.push_back(cy);
    vertices.push_back(1.0f);  // mask_value = 1.0 (피부)

    // index 1~36: oval points
    for (int i = 0; i < 36; ++i) {
        int idx = FACE_OVAL_INDICES[i];
        vertices.push_back(face_mesh[idx].x);
        vertices.push_back(face_mesh[idx].y);
        vertices.push_back(1.0f);
    }

    // 3. EBO 인덱스 배열 (36 triangles * 3 indices = 108)
    indices.clear();
    indices.reserve(36 * 3);

    for (int i = 0; i < 36; ++i) {
        indices.push_back(0);                   // centroid
        indices.push_back(i + 1);               // oval[i]
        indices.push_back((i + 1) % 36 + 1);   // oval[(i+1) % 36]
    }
}
```

### 3.3 구체적 EBO 인덱스 테이블

```
삼각형  | EBO 인덱스                | 실제 랜드마크
--------|--------------------------|----------------------------
 T0     | 0,  1,  2               | centroid, lm[10],  lm[338]
 T1     | 0,  2,  3               | centroid, lm[338], lm[297]
 T2     | 0,  3,  4               | centroid, lm[297], lm[332]
 T3     | 0,  4,  5               | centroid, lm[332], lm[284]
 T4     | 0,  5,  6               | centroid, lm[284], lm[251]
 T5     | 0,  6,  7               | centroid, lm[251], lm[389]
 T6     | 0,  7,  8               | centroid, lm[389], lm[356]
 T7     | 0,  8,  9               | centroid, lm[356], lm[454]
 T8     | 0,  9,  10              | centroid, lm[454], lm[323]
 T9     | 0, 10, 11               | centroid, lm[323], lm[361]
 T10    | 0, 11, 12               | centroid, lm[361], lm[288]
 T11    | 0, 12, 13               | centroid, lm[288], lm[397]
 T12    | 0, 13, 14               | centroid, lm[397], lm[365]
 T13    | 0, 14, 15               | centroid, lm[365], lm[379]
 T14    | 0, 15, 16               | centroid, lm[379], lm[378]
 T15    | 0, 16, 17               | centroid, lm[378], lm[400]
 T16    | 0, 17, 18               | centroid, lm[400], lm[377]
 T17    | 0, 18, 19               | centroid, lm[377], lm[152]
 T18    | 0, 19, 20               | centroid, lm[152], lm[148]
 T19    | 0, 20, 21               | centroid, lm[148], lm[176]
 T20    | 0, 21, 22               | centroid, lm[176], lm[149]
 T21    | 0, 22, 23               | centroid, lm[149], lm[150]
 T22    | 0, 23, 24               | centroid, lm[150], lm[136]
 T23    | 0, 24, 25               | centroid, lm[136], lm[172]
 T24    | 0, 25, 26               | centroid, lm[172], lm[58]
 T25    | 0, 26, 27               | centroid, lm[58],  lm[132]
 T26    | 0, 27, 28               | centroid, lm[132], lm[93]
 T27    | 0, 28, 29               | centroid, lm[93],  lm[234]
 T28    | 0, 29, 30               | centroid, lm[234], lm[127]
 T29    | 0, 30, 31               | centroid, lm[127], lm[162]
 T30    | 0, 31, 32               | centroid, lm[162], lm[21]
 T31    | 0, 32, 33               | centroid, lm[21],  lm[54]
 T32    | 0, 33, 34               | centroid, lm[54],  lm[103]
 T33    | 0, 34, 35               | centroid, lm[103], lm[67]
 T34    | 0, 35, 36               | centroid, lm[67],  lm[109]
 T35    | 0, 36,  1               | centroid, lm[109], lm[10]  ← 닫힘
```

총 108개 인덱스, `GL_TRIANGLES` 모드로 `glDrawElements()` 호출.

---

## 4. buildExclusionPolygons() - 눈/입술 삼각형 분할

제외 영역은 `mask_value = 0.0`으로 **검은색 삼각형**을 Face Oval 위에 덮어쓰기한다. 각 영역도 triangle fan 방식으로 분할한다.

### 4.1 구현 코드

```cpp
void SkinMaskRenderer::buildExclusionPolygons(
    const IrisLandmark* face_mesh,
    std::vector<float>& vertices,
    std::vector<uint16_t>& indices) {

    // Left Eye 16점 (beauty_roi_manager.cpp line 27-30)
    static constexpr int LEFT_EYE_INDICES[16] = {
        33, 7, 163, 144, 145, 153, 154, 155,
        133, 173, 157, 158, 159, 160, 161, 246
    };

    // Right Eye 16점 (beauty_roi_manager.cpp line 33-36)
    static constexpr int RIGHT_EYE_INDICES[16] = {
        362, 382, 381, 380, 374, 373, 390, 249,
        263, 466, 388, 387, 386, 385, 384, 398
    };

    // Lips 22점 (beauty_roi_manager.cpp line 39-42)
    static constexpr int LIPS_INDICES[22] = {
        61, 146, 91, 181, 84, 17, 314, 405,
        321, 375, 291, 308, 324, 318, 402, 317,
        14, 87, 178, 88, 95, 78
    };

    vertices.clear();
    indices.clear();

    // === Eyes (mask_value = 0.0) ===
    if (config_.exclude_eyes) {
        appendTriangleFan(face_mesh, LEFT_EYE_INDICES, 16,
                          0.0f, vertices, indices);
        appendTriangleFan(face_mesh, RIGHT_EYE_INDICES, 16,
                          0.0f, vertices, indices);
    }

    // === Lips (mask_value = 0.0) ===
    if (config_.exclude_lips) {
        appendTriangleFan(face_mesh, LIPS_INDICES, 22,
                          0.0f, vertices, indices);
    }

    // === Eyebrows (선택적, mask_value = 0.0) ===
    if (config_.exclude_eyebrows) {
        static constexpr int LEFT_BROW[8] = {
            70, 63, 105, 66, 107, 55, 65, 52
        };
        static constexpr int RIGHT_BROW[8] = {
            300, 293, 334, 296, 336, 285, 295, 282
        };
        appendTriangleFan(face_mesh, LEFT_BROW, 8,
                          0.0f, vertices, indices);
        appendTriangleFan(face_mesh, RIGHT_BROW, 8,
                          0.0f, vertices, indices);
    }
}
```

### 4.2 공통 Triangle Fan 헬퍼

```cpp
void SkinMaskRenderer::appendTriangleFan(
    const IrisLandmark* face_mesh,
    const int* polygon_indices,
    int count,
    float mask_value,
    std::vector<float>& vertices,
    std::vector<uint16_t>& indices) {

    // 1. centroid 계산
    float cx = 0.0f, cy = 0.0f;
    for (int i = 0; i < count; ++i) {
        cx += face_mesh[polygon_indices[i]].x;
        cy += face_mesh[polygon_indices[i]].y;
    }
    cx /= static_cast<float>(count);
    cy /= static_cast<float>(count);

    // 2. 현재 정점 배열 오프셋
    uint16_t base = static_cast<uint16_t>(vertices.size() / 3);

    // 3. centroid 정점
    vertices.push_back(cx);
    vertices.push_back(cy);
    vertices.push_back(mask_value);

    // 4. 다각형 정점
    for (int i = 0; i < count; ++i) {
        int idx = polygon_indices[i];
        vertices.push_back(face_mesh[idx].x);
        vertices.push_back(face_mesh[idx].y);
        vertices.push_back(mask_value);
    }

    // 5. 삼각형 인덱스
    for (int i = 0; i < count; ++i) {
        indices.push_back(base);                           // centroid
        indices.push_back(base + i + 1);                   // polygon[i]
        indices.push_back(base + (i + 1) % count + 1);    // polygon[(i+1) % N]
    }
}
```

### 4.3 제외 영역 삼각형 테이블

```
영역        | 정점 수    | 삼각형 수 | EBO 인덱스 수
------------|-----------|-----------|-------------
Face Oval   | 36+1=37   | 36        | 108
Left Eye    | 16+1=17   | 16        | 48
Right Eye   | 16+1=17   | 16        | 48
Lips        | 22+1=23   | 22        | 66
------------|-----------|-----------|-------------
합계 (기본)  | 94        | 90        | 270

Eyebrows 추가 시:
Left Brow   | 8+1=9     | 8         | 24
Right Brow  | 8+1=9     | 8         | 24
------------|-----------|-----------|-------------
합계 (전체)  | 112       | 106       | 318
```

### 4.4 렌더링 순서와 단일 Draw Call 통합

VBO에 모든 정점(94개)을, EBO에 모든 인덱스(270개)를 넣어 **단일 draw call**로 처리한다.

```
EBO 구조 (배열 순서):
  [0..107]    Face Oval (mask=1.0, 흰색) ← 먼저 그려짐
  [108..155]  Left Eye  (mask=0.0, 검은색) ← 위에 덮어씌워짐
  [156..203]  Right Eye (mask=0.0, 검은색)
  [204..269]  Lips      (mask=0.0, 검은색)
```

Face Oval이 먼저 그려진 뒤 제외 영역이 덮어쓰기되므로, **깊이 테스트 비활성화** (`glDisable(GL_DEPTH_TEST)`)가 필요하다. OpenGL ES에서 `glDrawElements` 단일 호출 시 인덱스 순서대로 삼각형이 래스터화되므로, EBO 배열에서 Face Oval 인덱스를 먼저, 제외 영역 인덱스를 뒤에 배치하면 올바르게 동작한다.

---

## 5. Gaussian Blur Pass - 경계 스무딩 셰이더

마스크 경계의 하드 에지를 부드럽게 만들기 위해 **Separable Gaussian Blur** 2-pass를 적용한다.

### 5.1 마스크 전용 Gaussian Blur 셰이더

기존 `GAUSSIAN_BLUR_FRAGMENT` (`shader_sources.cpp` line 384-409)는 RGB 3채널을 처리하지만, 마스크는 R 채널만 필요하다. 성능 최적화를 위해 단일 채널 전용 셰이더를 추가한다.

```glsl
// SKIN_MASK_BLUR_FRAGMENT (shader_sources.cpp에 추가)
#version 310 es
precision mediump float;

uniform sampler2D uTexture;
uniform vec2 uTexelSize;
uniform vec2 uDirection;     // (1,0) = horizontal, (0,1) = vertical
uniform float uBlurRadius;   // config.edge_blur_radius에 비례 (3.0~8.0)

in vec2 vTexCoord;
out vec4 fragColor;

// 13-tap Gaussian weights (sigma ~= 3.0)
const int KERNEL_SIZE = 7;
const float weights[7] = float[](
    0.1964826, 0.1748685, 0.1209854, 0.0651082,
    0.0272290, 0.0088503, 0.0022345
);

void main() {
    float result = texture(uTexture, vTexCoord).r * weights[0];

    for (int i = 1; i < KERNEL_SIZE; i++) {
        vec2 offset = uDirection * uTexelSize * float(i) * (uBlurRadius / 5.0);
        result += texture(uTexture, vTexCoord + offset).r * weights[i];
        result += texture(uTexture, vTexCoord - offset).r * weights[i];
    }

    fragColor = vec4(result, result, result, 1.0);
}
```

### 5.2 마스크 렌더링 셰이더 (Triangle Fan용)

기존 코드베이스의 셰이더는 모두 `#version 310 es`를 사용하므로 동일하게 맞춘다.

```glsl
// SKIN_MASK_VERTEX (shader_sources.cpp에 추가)
#version 310 es
precision mediump float;

layout(location = 0) in vec2 a_position;    // 정규화 좌표 (0~1)
layout(location = 1) in float a_mask_value;  // 1.0=피부, 0.0=비피부

out float v_mask_value;

void main() {
    // 정규화 좌표 → NDC (-1 ~ +1)
    vec2 ndc = a_position * 2.0 - 1.0;
    ndc.y = -ndc.y;  // Y축 반전 (OpenGL ↔ 이미지 좌표)
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_mask_value = a_mask_value;
}

// SKIN_MASK_FRAGMENT (shader_sources.cpp에 추가)
#version 310 es
precision mediump float;

in float v_mask_value;
out vec4 fragColor;

void main() {
    fragColor = vec4(v_mask_value, 0.0, 0.0, 1.0);  // R 채널만 사용
}
```

### 5.3 Separable Blur 실행 패턴

`executeSmoothingPass()` (gpu_beauty_backend.cpp line 524-541) 패턴을 따른다.

```cpp
void SkinMaskRenderer::executeMaskBlur(int width, int height) {
#if IRIS_SDK_GPU_AVAILABLE
    // === Pass 1: Horizontal Blur ===
    // mask_texture_ → blur_temp_fbo_
    glBindFramebuffer(GL_FRAMEBUFFER, blur_temp_fbo_);
    glViewport(0, 0, width, height);
    glUseProgram(mask_blur_program_);

    glUniform1i(blur_uniforms_.uTexture, 0);
    glUniform2f(blur_uniforms_.uTexelSize, 1.0f / width, 1.0f / height);
    glUniform2f(blur_uniforms_.uDirection, 1.0f, 0.0f);  // Horizontal
    glUniform1f(blur_uniforms_.uBlurRadius, config_.edge_blur_radius);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, mask_texture_);

    renderMaskQuad();  // fullscreen quad

    // === Pass 2: Vertical Blur ===
    // blur_temp_texture_ → mask_fbo_
    glBindFramebuffer(GL_FRAMEBUFFER, mask_fbo_);
    glViewport(0, 0, width, height);

    glUniform2f(blur_uniforms_.uDirection, 0.0f, 1.0f);  // Vertical

    glBindTexture(GL_TEXTURE_2D, blur_temp_texture_);

    renderMaskQuad();

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
#endif
}
```

### 5.4 Blur용 Fullscreen Quad

GPUBeautyBackend의 `setupFullscreenQuad()` (line 241-281) 패턴을 그대로 따른다.

```cpp
void SkinMaskRenderer::setupMaskQuad() {
#if IRIS_SDK_GPU_AVAILABLE
    float quad_vertices[] = {
        // Position    // TexCoord
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };

    glGenVertexArrays(1, &blur_quad_vao_);
    glGenBuffers(1, &blur_quad_vbo_);

    glBindVertexArray(blur_quad_vao_);
    glBindBuffer(GL_ARRAY_BUFFER, blur_quad_vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad_vertices),
                 quad_vertices, GL_STATIC_DRAW);

    // position (location=0): 2 floats
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE,
                          4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // texcoord (location=1): 2 floats
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE,
                          4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
#endif
}

void SkinMaskRenderer::renderMaskQuad() {
#if IRIS_SDK_GPU_AVAILABLE
    glBindVertexArray(blur_quad_vao_);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
#endif
}
```

---

## 6. FBO 해상도 전략

### 6.1 해상도별 비교 분석

| 해상도 | 카메라 1080x1920 기준 | 텍스처 크기 (GL_R8) | 장점 | 단점 |
|--------|----------------------|---------------------|------|------|
| Full (1x) | 1080x1920 | 2.07MB | 정밀한 경계 | 메모리 낭비, blur 비용 높음 |
| **1/2 (권장)** | **540x960** | **0.52MB** | **경계 품질 충분, 성능 양호** | **약간의 정밀도 손실** |
| 1/4 | 270x480 | 0.13MB | 최소 메모리/성능 | 경계 뭉개짐 가능 |

**권장: 1/2 해상도**

- 마스크는 블러 처리되므로 픽셀 단위 정밀도가 불필요
- GL_R8 포맷 사용 시 메모리 효율 극대화 (RGBA 대비 75% 절감)
- Gaussian blur의 텍셀 수가 1/4로 감소하여 blur 비용도 절감

### 6.2 해상도 계산 및 정렬

```cpp
static constexpr float MASK_RESOLUTION_SCALE = 0.5f;

int SkinMaskRenderer::calculateMaskWidth(int camera_width) const {
    int w = static_cast<int>(camera_width * config_.resolution_scale);
    return (w + 3) & ~3;  // 4의 배수 정렬 (GPU 텍스처 효율)
}

int SkinMaskRenderer::calculateMaskHeight(int camera_height) const {
    int h = static_cast<int>(camera_height * config_.resolution_scale);
    return (h + 3) & ~3;
}
```

### 6.3 FBO/텍스처 생성 및 재사용

카메라 해상도가 변경될 때만 FBO/텍스처를 재생성한다. `texture_pool.cpp` line 315-346 패턴을 따른다.

```cpp
bool SkinMaskRenderer::ensureMaskFBO(int camera_width, int camera_height) {
#if IRIS_SDK_GPU_AVAILABLE
    int mask_w = calculateMaskWidth(camera_width);
    int mask_h = calculateMaskHeight(camera_height);

    // 해상도 변경 없으면 기존 FBO 재사용
    if (mask_w == current_width_ && mask_h == current_height_
        && mask_fbo_ != 0) {
        return true;
    }

    // 기존 리소스 해제
    releaseMaskFBO();

    // === 마스크 텍스처 생성 ===
    glGenTextures(1, &mask_texture_);
    glBindTexture(GL_TEXTURE_2D, mask_texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    // R8 포맷: 단일 채널 (RGBA의 1/4 메모리)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, mask_w, mask_h, 0,
                 GL_RED, GL_UNSIGNED_BYTE, nullptr);

    // === 마스크 FBO 생성 ===
    glGenFramebuffers(1, &mask_fbo_);
    glBindFramebuffer(GL_FRAMEBUFFER, mask_fbo_);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, mask_texture_, 0);

    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        LOGE("SkinMask FBO incomplete: 0x%x", status);
        releaseMaskFBO();
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        return false;
    }

    // === Blur 임시 텍스처/FBO 생성 ===
    glGenTextures(1, &blur_temp_texture_);
    glBindTexture(GL_TEXTURE_2D, blur_temp_texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, mask_w, mask_h, 0,
                 GL_RED, GL_UNSIGNED_BYTE, nullptr);

    glGenFramebuffers(1, &blur_temp_fbo_);
    glBindFramebuffer(GL_FRAMEBUFFER, blur_temp_fbo_);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, blur_temp_texture_, 0);

    status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        LOGE("SkinMask blur temp FBO incomplete: 0x%x", status);
        releaseMaskFBO();
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        return false;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    current_width_ = mask_w;
    current_height_ = mask_h;

    LOGI("SkinMask FBO created: mask=%u/%u, blur_temp=%u/%u (%dx%d)",
         mask_texture_, mask_fbo_, blur_temp_texture_, blur_temp_fbo_,
         mask_w, mask_h);
    return true;
#else
    return false;
#endif
}
```

### 6.4 GL_R8 포맷 근거

| 포맷 | 바이트/픽셀 | 540x960 크기 | 비고 |
|------|------------|-------------|------|
| GL_RGBA | 4 | 2.07MB | 과다 (마스크는 단일 채널) |
| **GL_R8** | **1** | **0.52MB** | **권장 - 마스크에 최적** |
| GL_R16F | 2 | 1.04MB | 부동소수점 불필요 |

마스크 FBO 2개(마스크 + blur 임시) 합계: 약 **1.04MB** (GL_R8, 540x960 기준).

---

## 7. 마스크 업데이트 빈도 전략

### 7.1 문제 정의

마스크를 매 프레임(30fps) 재생성하면 GPU 부하가 추가된다. 하지만 얼굴이 거의 움직이지 않을 때는 이전 프레임의 마스크를 재사용할 수 있다.

### 7.2 전략 비교

| 전략 | GPU 부하 | 시각적 품질 | 구현 복잡도 | 권장 여부 |
|------|---------|-----------|-----------|---------|
| **A. 매 프레임 업데이트** | ~1.0ms/frame | 완벽 추적 | 최소 | Phase 1 권장 |
| B. N 프레임마다 업데이트 | ~0.3ms/frame(N=3) | 약간의 지연 | 낮음 | Phase 2 |
| C. 얼굴 이동 감지 시만 | ~0.1ms/frame(정지 시) | 이동 시 지연 | 중간 | Phase 2 |
| D. A+C 혼합 (적응형) | 가변 | 최상 | 높음 | Phase 3 |

### 7.3 Phase 1 권장: 매 프레임 업데이트

**근거:**
- 마스크 렌더링 비용이 **<1.0ms**로 프레임 예산(33ms)의 3%에 불과
- 구현이 가장 단순하고 디버깅이 용이
- Face Mesh 478점은 매 프레임 변동하므로, 캐싱하면 미세한 경계 불일치 발생 가능
- 최적화는 실측 후 필요 시 Phase 2에서 진행

### 7.4 Phase 2 옵션: 얼굴 이동 감지 기반 업데이트

기존 `mediapipe_detector.cpp`에 `prev_face_rect` (line 189) 추적이 구현되어 있으므로, 이를 활용한 이동 감지가 가능하다.

```cpp
// SkinMaskRenderer 멤버 추가
struct UpdatePolicy {
    enum class Mode {
        EVERY_FRAME,        // 매 프레임 (Phase 1 기본)
        MOTION_DETECT,      // 이동 감지 시만 (Phase 2)
        ADAPTIVE            // 적응형 (Phase 3)
    };

    Mode mode = Mode::EVERY_FRAME;
    float motion_threshold = 0.005f;  // 정규화 좌표 기준 이동 임계값
    int max_skip_frames = 5;          // 최대 건너뛸 프레임 수
};

// 이동 감지 로직
bool SkinMaskRenderer::shouldUpdateMask(const IrisLandmark* face_mesh) {
    if (update_policy_.mode == UpdatePolicy::Mode::EVERY_FRAME) {
        return true;
    }

    if (!has_prev_landmarks_) {
        has_prev_landmarks_ = true;
        storePrevLandmarks(face_mesh);
        return true;  // 첫 프레임
    }

    skip_frame_count_++;

    // 최대 건너뛰기 초과 → 강제 업데이트
    if (skip_frame_count_ >= update_policy_.max_skip_frames) {
        skip_frame_count_ = 0;
        storePrevLandmarks(face_mesh);
        return true;
    }

    // Face Oval 대표점 4개(이마/턱/좌/우)의 이동량 확인
    // 인덱스: 10(이마 중앙), 152(턱 중앙), 234(좌측), 454(우측)
    static constexpr int SAMPLE_INDICES[4] = {10, 152, 234, 454};
    float max_delta = 0.0f;

    for (int i = 0; i < 4; ++i) {
        int idx = SAMPLE_INDICES[i];
        float dx = face_mesh[idx].x - prev_sample_landmarks_[i].x;
        float dy = face_mesh[idx].y - prev_sample_landmarks_[i].y;
        float delta = dx * dx + dy * dy;  // 제곱 거리
        max_delta = std::max(max_delta, delta);
    }

    float threshold_sq = update_policy_.motion_threshold
                       * update_policy_.motion_threshold;

    if (max_delta > threshold_sq) {
        skip_frame_count_ = 0;
        storePrevLandmarks(face_mesh);
        return true;
    }

    return false;  // 이동 없음 → 이전 마스크 재사용
}

void SkinMaskRenderer::storePrevLandmarks(const IrisLandmark* face_mesh) {
    static constexpr int SAMPLE_INDICES[4] = {10, 152, 234, 454};
    for (int i = 0; i < 4; ++i) {
        prev_sample_landmarks_[i] = face_mesh[SAMPLE_INDICES[i]];
    }
}
```

### 7.5 Motion Detection 대표점 선택 근거

Face Oval 36점 전체를 비교하는 대신, **4개 대표점**만 비교하여 CPU 오버헤드를 최소화한다.

```
대표점 배치:

        lm[10] (이마 중앙, Face Oval 최상단)
           |
  lm[234]--+--lm[454]  (좌/우 관자놀이, 수평 이동 감지)
           |
        lm[152] (턱 끝, Face Oval 최하단)
```

| 대표점 | 랜드마크 | 감지 대상 |
|--------|---------|----------|
| lm[10] | 이마 중앙 | 상하 이동, 고개 숙임/젖힘 |
| lm[152] | 턱 끝 | 상하 이동, 고개 숙임/젖힘 |
| lm[234] | 좌측 관자놀이 | 좌우 이동, 고개 회전 |
| lm[454] | 우측 관자놀이 | 좌우 이동, 고개 회전 |

### 7.6 성능 예상치

```
                          매 프레임     이동 감지 (정지 시)    이동 감지 (이동 시)
마스크 렌더링 (90 tri)    ~0.2ms        0ms (skip)            ~0.2ms
Blur H-pass              ~0.3ms        0ms (skip)            ~0.3ms
Blur V-pass              ~0.3ms        0ms (skip)            ~0.3ms
shouldUpdateMask()        -             ~0.001ms              ~0.001ms
VBO/EBO upload           ~0.1ms        0ms (skip)            ~0.1ms
합계                     ~0.9ms        ~0.001ms              ~0.9ms
프레임 평균 (30% 이동)    0.9ms         ~0.27ms               -
```

정지 상태가 70%라고 가정하면, 이동 감지 방식은 평균 **~0.27ms/frame** (매 프레임 대비 70% 절감).

### 7.7 구현 권장 순서

```
Phase 1 (즉시): EVERY_FRAME 모드
  → 가장 단순, 디버깅 용이
  → 예상 비용 ~1.0ms/frame, 허용 범위 내

Phase 2 (성능 측정 후): MOTION_DETECT 모드
  → 실측 결과 마스크 비용 > 1.5ms이면 전환
  → motion_threshold 튜닝 필요 (0.003~0.01 범위)

Phase 3 (고급): ADAPTIVE 모드
  → GPU 프레임 시간 모니터링 → 동적 전환
  → 배터리 세이버 모드 연동
```

---

## 8. renderSkinMask() 전체 파이프라인

```cpp
uint32_t SkinMaskRenderer::renderSkinMask(
    const IrisLandmark* face_mesh,
    int camera_width, int camera_height) {
#if IRIS_SDK_GPU_AVAILABLE
    // 1. FBO 확보 (해상도 변경 시 재생성)
    if (!ensureMaskFBO(camera_width, camera_height)) {
        return 0;
    }

    // 2. 업데이트 필요 여부 확인 (Phase 2+ 에서 활성화)
    if (!shouldUpdateMask(face_mesh)) {
        return mask_texture_;  // 이전 마스크 재사용
    }

    // 3. 정점/인덱스 데이터 빌드
    std::vector<float> vertices;
    std::vector<uint16_t> indices;

    // Face Oval (흰색 = 1.0)
    buildTriangleFan(face_mesh, vertices, indices);

    // 제외 영역 (검은색 = 0.0)
    std::vector<float> excl_vertices;
    std::vector<uint16_t> excl_indices;
    buildExclusionPolygons(face_mesh, excl_vertices, excl_indices);

    // 인덱스 오프셋 조정 후 병합
    uint16_t offset = static_cast<uint16_t>(vertices.size() / 3);
    for (auto& idx : excl_indices) {
        idx += offset;
    }
    vertices.insert(vertices.end(), excl_vertices.begin(), excl_vertices.end());
    indices.insert(indices.end(), excl_indices.begin(), excl_indices.end());

    // 4. VBO/EBO 업데이트
    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER,
                 vertices.size() * sizeof(float),
                 vertices.data(), GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 indices.size() * sizeof(uint16_t),
                 indices.data(), GL_DYNAMIC_DRAW);

    // 5. 마스크 FBO에 렌더링
    glBindFramebuffer(GL_FRAMEBUFFER, mask_fbo_);
    glViewport(0, 0, current_width_, current_height_);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glUseProgram(mask_program_);

    // position (location=0): stride 3 floats, offset 0
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE,
                          3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // mask_value (location=1): stride 3 floats, offset 2 floats
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE,
                          3 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glDrawElements(GL_TRIANGLES,
                   static_cast<GLsizei>(indices.size()),
                   GL_UNSIGNED_SHORT, 0);

    glBindVertexArray(0);

    // 6. Gaussian blur로 경계 스무딩
    if (config_.edge_blur_radius > 0.1f) {
        executeMaskBlur(current_width_, current_height_);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return mask_texture_;
#else
    return 0;
#endif
}
```

---

## 9. C++ 헤더

```cpp
// cpp/include/iris_sdk/gpu/skin_mask_renderer.h

#pragma once

#include "iris_sdk/types.h"
#include <vector>
#include <cstdint>

namespace iris_sdk {

class SkinMaskRenderer {
public:
    SkinMaskRenderer();
    ~SkinMaskRenderer();

    // GPU 리소스 소유 → 복사/이동 금지
    SkinMaskRenderer(const SkinMaskRenderer&) = delete;
    SkinMaskRenderer& operator=(const SkinMaskRenderer&) = delete;

    /// 초기화 (셰이더 컴파일, VAO/VBO/EBO 생성, blur quad 설정)
    bool initialize();

    /// 피부 마스크 텍스처 생성/업데이트
    /// @param face_mesh 478개 랜드마크 (정규화 좌표 0-1)
    /// @param camera_width 카메라 원본 너비
    /// @param camera_height 카메라 원본 높이
    /// @return 마스크 텍스처 ID (GLuint), 실패 시 0
    uint32_t renderSkinMask(
        const IrisLandmark* face_mesh,
        int camera_width, int camera_height);

    /// 리소스 해제
    void release();

    /// 마스크 영역 설정
    struct MaskConfig {
        bool exclude_eyes = true;
        bool exclude_eyebrows = false;
        bool exclude_lips = true;
        float edge_blur_radius = 5.0f;   ///< Gaussian blur 반경 (px)
        float resolution_scale = 0.5f;   ///< 마스크 해상도 비율 (0.25~1.0)
    };

    void setMaskConfig(const MaskConfig& config);
    const MaskConfig& getMaskConfig() const { return config_; }

    /// 마스크 업데이트 정책
    struct UpdatePolicy {
        enum class Mode {
            EVERY_FRAME,        ///< 매 프레임 (기본, Phase 1)
            MOTION_DETECT,      ///< 얼굴 이동 감지 시만 (Phase 2)
            ADAPTIVE            ///< 적응형 (Phase 3)
        };

        Mode mode = Mode::EVERY_FRAME;
        float motion_threshold = 0.005f;  ///< 이동 임계값 (정규화 좌표)
        int max_skip_frames = 5;          ///< 최대 연속 스킵 프레임
    };

    void setUpdatePolicy(const UpdatePolicy& policy);

private:
    // Face Oval 삼각형 팬 생성 (mask_value = 1.0)
    void buildTriangleFan(const IrisLandmark* face_mesh,
                          std::vector<float>& vertices,
                          std::vector<uint16_t>& indices);

    // 제외 영역 다각형 생성 (mask_value = 0.0)
    void buildExclusionPolygons(const IrisLandmark* face_mesh,
                                std::vector<float>& vertices,
                                std::vector<uint16_t>& indices);

    // 공통 triangle fan 빌더
    void appendTriangleFan(const IrisLandmark* face_mesh,
                           const int* polygon_indices, int count,
                           float mask_value,
                           std::vector<float>& vertices,
                           std::vector<uint16_t>& indices);

    // Gaussian blur 실행 (2-pass separable)
    void executeMaskBlur(int width, int height);

    // 마스크 업데이트 판정
    bool shouldUpdateMask(const IrisLandmark* face_mesh);
    void storePrevLandmarks(const IrisLandmark* face_mesh);

    // FBO/텍스처 관리
    bool ensureMaskFBO(int camera_width, int camera_height);
    void releaseMaskFBO();

    // 해상도 계산
    int calculateMaskWidth(int camera_width) const;
    int calculateMaskHeight(int camera_height) const;

    // Blur quad
    void setupMaskQuad();
    void renderMaskQuad();

    // --- GPU 리소스 ---
    // 마스크 렌더링
    uint32_t mask_fbo_ = 0;
    uint32_t mask_texture_ = 0;
    uint32_t mask_program_ = 0;
    uint32_t vao_ = 0;
    uint32_t vbo_ = 0;
    uint32_t ebo_ = 0;

    // Blur pass
    uint32_t blur_temp_fbo_ = 0;
    uint32_t blur_temp_texture_ = 0;
    uint32_t mask_blur_program_ = 0;
    uint32_t blur_quad_vao_ = 0;
    uint32_t blur_quad_vbo_ = 0;

    // Uniform locations
    struct BlurUniforms {
        int32_t uTexture = -1;
        int32_t uTexelSize = -1;
        int32_t uDirection = -1;
        int32_t uBlurRadius = -1;
    } blur_uniforms_;

    // 설정
    MaskConfig config_;
    UpdatePolicy update_policy_;
    int current_width_ = 0;
    int current_height_ = 0;
    bool initialized_ = false;

    // 이동 감지용
    bool has_prev_landmarks_ = false;
    int skip_frame_count_ = 0;
    IrisLandmark prev_sample_landmarks_[4];  // 4개 대표점
};

} // namespace iris_sdk
```

---

## 10. GPUBeautyBackend 통합

### 10.1 수정 대상

`cpp/src/gpu/gpu_beauty_backend.cpp` - `applyTextureId()` (line 797-989)

### 10.2 통합 코드

```cpp
// applyTextureId() 내부, 필터 체인 실행 전에 추가

// 피부 마스크 생성 (Face Mesh 기반)
GLuint skin_mask_tex = 0;
if (skin_mask_renderer_ && detection && detection->face_mesh_valid) {
    skin_mask_tex = skin_mask_renderer_->renderSkinMask(
        detection->face_mesh, width, height);
}

// 각 필터 패스에 마스크 전달
if (config.smoothing > 0.01f) {
    executeSmoothingPass(current_input, current_output->fbo_id,
                         width, height, config,
                         skin_mask_tex);  // 마스크 추가
    // ping-pong swap...
}
```

### 10.3 Beauty 셰이더에서 마스크 사용

```glsl
// 기존 Beauty 셰이더에 추가
uniform sampler2D u_skin_mask;
uniform float u_mask_enabled;  // 0.0 = 전체, 1.0 = 마스크 적용

// 마스크 기반 필터 강도 조절
float mask = u_mask_enabled > 0.5
    ? texture(u_skin_mask, v_texCoord).r
    : 1.0;

// 원본과 필터 결과를 마스크로 블렌딩
vec3 final_color = mix(original_color, filtered_color, mask * u_intensity);
```

### 10.4 영역별 뷰티 강도 차별화 (beauty-tuner 연계)

마스크를 단순 이진(0/1)이 아닌 영역별 가중치로 확장 가능:

| 영역 | 마스크 값 | 용도 |
|------|-----------|------|
| 이마 | 0.8 | 스무딩 약하게 (모공 텍스처 유지) |
| 볼 | 1.0 | 스무딩 최대 (피부결 보정 핵심) |
| 코 | 0.6 | 하이라이트 보존 |
| 턱 | 0.9 | 피부톤 균일화 |
| 눈 | 0.0 | 제외 |
| 입술 | 0.0 | 제외 (별도 립 필터) |

이 기능은 beauty-tuner Task #20의 설계와 연계된다.

---

## 11. 메모리 및 성능 예산

### 11.1 메모리 예산 (카메라 1080x1920, scale=0.5)

| 리소스 | 크기 | 비고 |
|--------|------|------|
| mask_texture_ (GL_R8, 540x960) | 0.52MB | 최종 마스크 |
| blur_temp_texture_ (GL_R8, 540x960) | 0.52MB | Blur 중간 결과 |
| VBO (94 verts * 3 floats * 4B) | 1.1KB | 매 프레임 갱신 |
| EBO (270 indices * 2B) | 0.5KB | 매 프레임 갱신 |
| Blur quad VBO (6 verts * 4 floats * 4B) | 96B | 고정 |
| **합계** | **~1.05MB** | |

### 11.2 성능 예산 (Snapdragon 855 기준)

| 단계 | EVERY_FRAME | MOTION_DETECT (정지) |
|------|-------------|---------------------|
| buildTriangleFan + buildExclusion (CPU) | <0.1ms | skip |
| VBO/EBO glBufferData | <0.1ms | skip |
| mask FBO 렌더링 (90 triangles) | <0.2ms | skip |
| Gaussian blur H-pass | <0.3ms | skip |
| Gaussian blur V-pass | <0.3ms | skip |
| shouldUpdateMask() | - | <0.001ms |
| **합계** | **<1.0ms** | **<0.001ms** |
| **프레임 예산 대비** | **3%** | **~0%** |

---

## 12. 테스트 방안

### 12.1 시각적 검증

```
1. 마스크 텍스처를 화면에 직접 출력하여 확인
   - Face Oval 내부 = 밝은 영역 (흰색)
   - 눈/입술 = 어두운 영역 (검은색)
   - 경계 = 부드러운 그라데이션 (blur 효과)

2. 마스크 적용 전/후 비교
   - 마스크 OFF: Smoothing이 눈/입술에도 적용 (부자연스러움)
   - 마스크 ON: Smoothing이 피부에만 적용 (자연스러움)
```

### 12.2 성능 측정

```
Android 디바이스에서 GPU 타이머 쿼리:
1. renderSkinMask() 전체 시간
2. executeMaskBlur() 시간
3. shouldUpdateMask() 스킵 비율
4. 전체 beauty 파이프라인 시간 (마스크 추가 전/후)
```

### 12.3 경계 조건

```
1. 얼굴이 화면 밖으로 일부 벗어난 경우
   - Face Oval 좌표가 0~1 범위를 벗어날 수 있음
   - NDC 변환 후 클리핑에 의해 자동 처리됨

2. 다중 얼굴
   - 현재 단일 얼굴만 지원
   - 복수 얼굴 시 가장 큰 얼굴의 마스크만 생성

3. 해상도 변경 (카메라 전환)
   - ensureMaskFBO()에서 자동 재생성
   - 기존 FBO 해제 후 새로운 크기로 재할당
```

---

## 변경 이력

| 날짜 | 내용 |
|------|------|
| 2026-02-09 | 초안 작성 - P3-W1-04 섹션 4/7 내용 통합 + 마스크 업데이트 빈도 전략 추가 |
