# P2-W5-01. JNI 바인딩 및 Android 통합

## 작업 정보

| 항목 | 내용 |
|------|------|
| **작업 ID** | P2-W5-01 |
| **Phase** | Phase 5: 플랫폼 통합 |
| **상태** | ⏳ 대기 |
| **예상 기간** | 2일 |
| **의존성** | P2-W3-01 ~ P2-W4-03 (전체 구현) |
| **담당** | systems-programming:cpp-pro |

---

## 1. 목표

뷰티 필터 V2 기능을 Android JNI를 통해 노출

### 핵심 산출물
- BeautyFilterConfigV2 JNI 래퍼
- GPU 백엔드 JNI 인터페이스
- Face Warp JNI 인터페이스
- Zero-Copy 텍스처 처리

---

## 2. C API 확장

### 2.1 뷰티 필터 V2 API

**파일**: `cpp/include/iris_sdk/sdk_api.h` (확장)

```c
#ifndef IRIS_SDK_API_H
#define IRIS_SDK_API_H

#include "iris_sdk/types.h"

#ifdef __cplusplus
extern "C" {
#endif

//=== 기존 API ===
// ... (iris_sdk_init, iris_sdk_detect, etc.)

//=== 뷰티 필터 V2 API ===

/**
 * @brief 뷰티 필터 V2 설정 구조체
 */
typedef struct {
    // 기본 (V1 호환)
    int enabled;
    float intensity;
    float smoothing;
    float brightness;
    float soft_focus;

    // V2 확장
    float whitening;
    float color_balance;
    float wrinkle_remove;
    float slim_face;
    float enlarge_eyes;
    float thin_chin;

    // 옵션
    int use_gpu;
    int roi_only;
    int protect_eyes;
    int protect_lips;
    int downscale_factor;
    int feather_radius;
} IrisBeautyConfigV2;

/**
 * @brief 기본 설정 생성
 */
IRIS_SDK_API IrisBeautyConfigV2 iris_sdk_beauty_config_default_v2(void);

/**
 * @brief 뷰티 필터 V2 적용 (CPU 버퍼)
 *
 * @param frame_data 프레임 데이터
 * @param width 너비
 * @param height 높이
 * @param format 프레임 포맷
 * @param config V2 설정
 * @param detection 검출 결과 (Face Mesh 포함)
 * @return 에러 코드
 */
IRIS_SDK_API IrisSdkError iris_sdk_apply_beauty_v2(
    uint8_t* frame_data,
    int width, int height,
    IrisFrameFormat format,
    const IrisBeautyConfigV2* config,
    const IrisDetectionResult* detection
);

/**
 * @brief 뷰티 필터 V2 적용 (GPU 텍스처)
 *
 * @param input_texture OpenGL ES 텍스처 ID
 * @param output_texture 출력 텍스처 ID (0이면 새로 생성)
 * @param width 너비
 * @param height 높이
 * @param config V2 설정
 * @param detection 검출 결과
 * @return 에러 코드
 */
IRIS_SDK_API IrisSdkError iris_sdk_apply_beauty_texture_v2(
    uint32_t input_texture,
    uint32_t* output_texture,
    int width, int height,
    const IrisBeautyConfigV2* config,
    const IrisDetectionResult* detection
);

/**
 * @brief GPU 뷰티 백엔드 초기화
 *
 * @param egl_context 현재 EGL 컨텍스트 (nullptr = 현재 컨텍스트 사용)
 * @return 에러 코드
 */
IRIS_SDK_API IrisSdkError iris_sdk_init_gpu_beauty(void* egl_context);

/**
 * @brief GPU 뷰티 백엔드 해제
 */
IRIS_SDK_API void iris_sdk_release_gpu_beauty(void);

/**
 * @brief Face Warp 적용 (GPU)
 *
 * @param input_texture 입력 텍스처
 * @param output_texture 출력 텍스처
 * @param width 너비
 * @param height 높이
 * @param slim_face 슬림 강도
 * @param thin_chin V-라인 강도
 * @param enlarge_eyes 눈 확대 강도
 * @param detection 검출 결과 (Face Mesh 필요)
 * @return 에러 코드
 */
IRIS_SDK_API IrisSdkError iris_sdk_apply_face_warp(
    uint32_t input_texture,
    uint32_t* output_texture,
    int width, int height,
    float slim_face,
    float thin_chin,
    float enlarge_eyes,
    const IrisDetectionResult* detection
);

//=== 텍스처 소유권 관리 ===

/**
 * @brief SDK가 생성한 텍스처 해제
 *
 * iris_sdk_apply_beauty_texture_v2 또는 iris_sdk_apply_face_warp에서
 * 생성된 출력 텍스처를 해제합니다.
 *
 * **소유권 규칙**:
 * - SDK가 TexturePool에서 할당한 텍스처는 반드시 이 함수로 해제
 * - 호출자가 직접 glDeleteTextures 호출 금지 (풀 관리 충돌 방지)
 * - 입력 텍스처(input_texture)는 호출자 소유, SDK가 해제하지 않음
 *
 * @param texture SDK가 생성한 텍스처 ID
 * @return 에러 코드
 */
IRIS_SDK_API IrisSdkError iris_sdk_release_texture(uint32_t texture);

/**
 * @brief 텍스처 소유권 조회
 *
 * @param texture 텍스처 ID
 * @return 1 = SDK 관리, 0 = 외부 소유 또는 알 수 없음
 */
IRIS_SDK_API int iris_sdk_is_texture_managed(uint32_t texture);

#ifdef __cplusplus
}
#endif

#endif // IRIS_SDK_API_H
```

### 2.2 C API 구현

**파일**: `cpp/src/sdk_api.cpp` (확장)

```cpp
#include "iris_sdk/sdk_api.h"
#include "iris_sdk/sdk_manager.h"
#include "iris_sdk/beauty_processor.h"
#include "iris_sdk/gpu/gpu_beauty_backend.h"
#include "iris_sdk/warp/face_warp_controller.h"
#include "iris_sdk/warp/gpu_mesh_renderer.h"

namespace {
    std::unique_ptr<iris_sdk::GPUBeautyBackend> g_gpu_beauty;
    std::unique_ptr<iris_sdk::GPUMeshRenderer> g_mesh_renderer;
    std::unique_ptr<iris_sdk::GridMesh> g_grid_mesh;
}

IrisBeautyConfigV2 iris_sdk_beauty_config_default_v2(void) {
    IrisBeautyConfigV2 config = {};
    config.enabled = 1;
    config.intensity = 0.5f;
    config.smoothing = 0.5f;
    config.brightness = 1.0f;
    config.soft_focus = 0.0f;
    config.whitening = 0.0f;
    config.color_balance = 0.0f;
    config.wrinkle_remove = 0.0f;
    config.slim_face = 0.0f;
    config.enlarge_eyes = 0.0f;
    config.thin_chin = 0.0f;
    config.use_gpu = 1;
    config.roi_only = 1;
    config.protect_eyes = 1;
    config.protect_lips = 1;
    config.downscale_factor = 1;
    config.feather_radius = 15;
    return config;
}

IrisSdkError iris_sdk_init_gpu_beauty(void* egl_context) {
    try {
        auto* context = iris_sdk::SDKManager::getInstance().getRenderContext();
        if (!context) {
            return IRIS_SDK_ERROR_NOT_INITIALIZED;
        }

        g_gpu_beauty = std::make_unique<iris_sdk::GPUBeautyBackend>();
        if (!g_gpu_beauty->initialize(context)) {
            g_gpu_beauty.reset();
            return IRIS_SDK_ERROR_INTERNAL;
        }

        g_mesh_renderer = std::make_unique<iris_sdk::GPUMeshRenderer>();
        g_mesh_renderer->initialize(
            dynamic_cast<iris_sdk::GLESRenderContext*>(context));

        g_grid_mesh = std::make_unique<iris_sdk::GridMesh>();

        return IRIS_SDK_OK;
    } catch (...) {
        return IRIS_SDK_ERROR_INTERNAL;
    }
}

void iris_sdk_release_gpu_beauty(void) {
    g_mesh_renderer.reset();
    g_gpu_beauty.reset();
    g_grid_mesh.reset();
}

IrisSdkError iris_sdk_apply_beauty_texture_v2(
    uint32_t input_texture,
    uint32_t* output_texture,
    int width, int height,
    const IrisBeautyConfigV2* config,
    const IrisDetectionResult* detection) {

    if (!g_gpu_beauty || !config) {
        return IRIS_SDK_ERROR_INVALID_PARAM;
    }

    // C 구조체 → C++ 구조체 변환
    iris_sdk::BeautyFilterConfigV2 cpp_config;
    cpp_config.enabled = config->enabled != 0;
    cpp_config.smoothing = config->smoothing * config->intensity;
    cpp_config.brightness = config->brightness;
    cpp_config.softFocus = config->soft_focus;
    cpp_config.whitening = config->whitening;
    cpp_config.colorBalance = config->color_balance;
    cpp_config.wrinkleRemove = config->wrinkle_remove;
    cpp_config.slimFace = config->slim_face;
    cpp_config.enlargeEyes = config->enlarge_eyes;
    cpp_config.thinChin = config->thin_chin;
    cpp_config.useGpu = config->use_gpu != 0;
    cpp_config.roiOnly = config->roi_only != 0;
    cpp_config.protectEyes = config->protect_eyes != 0;
    cpp_config.protectLips = config->protect_lips != 0;
    cpp_config.downscaleFactor = config->downscale_factor;
    cpp_config.featherRadius = config->feather_radius;

    // TextureHandle 생성
    iris_sdk::TextureHandle input_handle;
    input_handle.native_handle = new GLuint(input_texture);
    input_handle.type = iris_sdk::TextureHandle::Type::OpenGLES;
    input_handle.width = width;
    input_handle.height = height;

    iris_sdk::TextureHandle output_handle;

    // ROI 생성 (선택적)
    std::unique_ptr<iris_sdk::BeautyROI> roi;
    if (detection && detection->face_detected && cpp_config.roiOnly) {
        roi = std::make_unique<iris_sdk::BeautyROI>();
        iris_sdk::BeautyROIManager::computeROI(
            detection->face_mesh, width, height, *roi);
    }

    // 뷰티 필터 적용
    IrisSdkError err = g_gpu_beauty->applyTexture(
        input_handle, output_handle, cpp_config, roi.get());

    if (err == IRIS_SDK_OK && output_handle.isValid()) {
        *output_texture = *static_cast<GLuint*>(output_handle.native_handle);
    }

    delete static_cast<GLuint*>(input_handle.native_handle);
    return err;
}

IrisSdkError iris_sdk_apply_face_warp(
    uint32_t input_texture,
    uint32_t* output_texture,
    int width, int height,
    float slim_face,
    float thin_chin,
    float enlarge_eyes,
    const IrisDetectionResult* detection) {

    if (!g_mesh_renderer || !g_grid_mesh || !detection) {
        return IRIS_SDK_ERROR_INVALID_PARAM;
    }

    if (!detection->face_detected) {
        *output_texture = input_texture;  // 패스스루
        return IRIS_SDK_OK;
    }

    // 얼굴 바운딩 박스 계산
    iris_sdk::Rect face_rect = iris_sdk::BeautyROIManager::computeFaceBoundingBox(
        detection->face_mesh, width, height);

    // Grid Mesh 초기화/업데이트
    g_grid_mesh->initialize(20, face_rect);
    g_grid_mesh->setControlPoints(detection->face_mesh, width, height);

    // Face Warp 적용
    iris_sdk::FaceWarpController warp;
    iris_sdk::FaceWarpController::WarpConfig warp_config;
    warp_config.slimFace = slim_face;
    warp_config.thinChin = thin_chin;
    warp_config.enlargeEyes = enlarge_eyes;

    warp.applyWarp(*g_grid_mesh, detection->face_mesh, warp_config);

    // GPU 렌더링
    // TODO: 출력 텍스처/FBO 관리
    // g_mesh_renderer->render(input_texture, output_fbo, *g_grid_mesh, width, height);

    return IRIS_SDK_OK;
}
```

### 2.3 텍스처 소유권 규칙

> **중요**: 메모리 누수와 이중 해제(double-free) 방지를 위한 명확한 규칙

#### 소유권 모델

```
┌─────────────────────────────────────────────────────────┐
│                    Texture Ownership                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  input_texture                  output_texture           │
│  ┌─────────┐                   ┌─────────┐              │
│  │ 호출자  │ ───(전달)───→    │   SDK   │              │
│  │  소유   │                   │ TexturePool │           │
│  └─────────┘                   └─────────┘              │
│       │                              │                   │
│       │                              │                   │
│       ▼                              ▼                   │
│  호출자가                      iris_sdk_release_texture()│
│  glDeleteTextures()            로 반환                   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

#### 규칙 요약

| 텍스처 종류 | 생성 주체 | 해제 책임 | 해제 방법 |
|------------|----------|----------|----------|
| input_texture | 호출자 (앱) | 호출자 | `glDeleteTextures()` |
| output_texture | SDK TexturePool | SDK | `iris_sdk_release_texture()` |
| 카메라 텍스처 | Android CameraX | Android | 자동 관리 |

#### 사용 예시 (Android/Kotlin)

```kotlin
class BeautyRenderer {
    private var lastOutputTexture: Int = 0

    fun processFrame(inputTexture: Int, detection: DetectionResult) {
        // 이전 출력 텍스처 반환 (SDK 관리 텍스처)
        if (lastOutputTexture != 0) {
            IrisSDKNative.releaseTexture(lastOutputTexture)
        }

        // 새 출력 텍스처 획득
        lastOutputTexture = IrisSDKNative.applyBeautyTextureV2(
            inputTexture,  // 입력: 호출자 소유 (반환 불필요)
            width, height,
            beautyConfig,
            detection
        )

        // lastOutputTexture 사용...
    }

    fun release() {
        // 정리 시 마지막 텍스처 반환
        if (lastOutputTexture != 0) {
            IrisSDKNative.releaseTexture(lastOutputTexture)
            lastOutputTexture = 0
        }
    }
}
```

#### 주의사항

1. **이중 해제 금지**: `iris_sdk_release_texture()`로 반환한 텍스처를 다시 반환하지 않음
2. **입력 텍스처 보존**: SDK는 입력 텍스처를 수정하거나 해제하지 않음
3. **프레임 간 재사용**: 동일 텍스처 ID가 다음 프레임에서 재할당될 수 있음
4. **GL 컨텍스트**: 텍스처 해제는 생성한 GL 컨텍스트에서 수행해야 함

---

## 3. JNI 바인딩

### 3.1 Java 클래스

**파일**: `bindings/android/src/main/java/com/meroomon/irissdk/BeautyFilterV2.java`

```java
package com.meroomon.irissdk;

/**
 * 뷰티 필터 V2 설정
 */
public class BeautyFilterV2 {

    // 기본 설정
    public boolean enabled = true;
    public float intensity = 0.5f;
    public float smoothing = 0.5f;
    public float brightness = 1.0f;
    public float softFocus = 0.0f;

    // V2 확장
    public float whitening = 0.0f;
    public float colorBalance = 0.0f;
    public float wrinkleRemove = 0.0f;
    public float slimFace = 0.0f;
    public float enlargeEyes = 0.0f;
    public float thinChin = 0.0f;

    // 옵션
    public boolean useGpu = true;
    public boolean roiOnly = true;
    public boolean protectEyes = true;
    public boolean protectLips = true;
    public int downscaleFactor = 1;
    public int featherRadius = 15;

    public BeautyFilterV2() {}

    public static BeautyFilterV2 createDefault() {
        return new BeautyFilterV2();
    }

    public static BeautyFilterV2 createNatural() {
        BeautyFilterV2 config = new BeautyFilterV2();
        config.intensity = 0.3f;
        config.smoothing = 0.4f;
        config.whitening = 0.2f;
        config.softFocus = 0.2f;
        return config;
    }

    public static BeautyFilterV2 createGlamour() {
        BeautyFilterV2 config = new BeautyFilterV2();
        config.intensity = 0.7f;
        config.smoothing = 0.6f;
        config.whitening = 0.4f;
        config.softFocus = 0.4f;
        config.slimFace = 0.3f;
        config.enlargeEyes = 0.2f;
        return config;
    }
}
```

### 3.2 JNI 네이티브 인터페이스

**파일**: `bindings/android/src/main/java/com/meroomon/irissdk/IrisSDKNative.java` (확장)

```java
package com.meroomon.irissdk;

public class IrisSDKNative {

    // ... 기존 메서드 ...

    //=== 뷰티 필터 V2 ===

    /**
     * GPU 뷰티 백엔드 초기화
     */
    public static native int initGpuBeauty();

    /**
     * GPU 뷰티 백엔드 해제
     */
    public static native void releaseGpuBeauty();

    /**
     * 뷰티 필터 V2 적용 (CPU 버퍼)
     */
    public static native int applyBeautyV2(
        byte[] frameData,
        int width, int height,
        int format,
        BeautyFilterV2 config,
        DetectionResult detection
    );

    /**
     * 뷰티 필터 V2 적용 (GPU 텍스처)
     *
     * @return 출력 텍스처 ID (실패 시 0)
     */
    public static native int applyBeautyTextureV2(
        int inputTexture,
        int width, int height,
        BeautyFilterV2 config,
        DetectionResult detection
    );

    /**
     * Face Warp 적용 (GPU 텍스처)
     *
     * @return 출력 텍스처 ID
     */
    public static native int applyFaceWarp(
        int inputTexture,
        int width, int height,
        float slimFace,
        float thinChin,
        float enlargeEyes,
        DetectionResult detection
    );

    //=== 텍스처 소유권 관리 ===

    /**
     * SDK가 생성한 텍스처 반환
     *
     * applyBeautyTextureV2, applyFaceWarp 등에서 반환된
     * 출력 텍스처는 반드시 이 메서드로 반환해야 함.
     *
     * @param texture SDK가 생성한 텍스처 ID
     */
    public static native void releaseTexture(int texture);

    /**
     * 텍스처가 SDK 관리인지 확인
     *
     * @param texture 텍스처 ID
     * @return true = SDK 관리, false = 외부 소유
     */
    public static native boolean isTextureManagedBySDK(int texture);
}
```

### 3.3 JNI 구현

**파일**: `bindings/android/jni/iris_sdk_jni_beauty_v2.cpp`

```cpp
#include <jni.h>
#include "iris_sdk/sdk_api.h"
#include "jni_utils.h"

extern "C" {

JNIEXPORT jint JNICALL
Java_com_meroomon_irissdk_IrisSDKNative_initGpuBeauty(
    JNIEnv* env, jclass clazz) {

    return static_cast<jint>(iris_sdk_init_gpu_beauty(nullptr));
}

JNIEXPORT void JNICALL
Java_com_meroomon_irissdk_IrisSDKNative_releaseGpuBeauty(
    JNIEnv* env, jclass clazz) {

    iris_sdk_release_gpu_beauty();
}

JNIEXPORT jint JNICALL
Java_com_meroomon_irissdk_IrisSDKNative_applyBeautyTextureV2(
    JNIEnv* env, jclass clazz,
    jint inputTexture,
    jint width, jint height,
    jobject configObj,
    jobject detectionObj) {

    // Java 객체에서 설정 추출
    IrisBeautyConfigV2 config = extractBeautyConfigV2(env, configObj);

    // 검출 결과 추출
    IrisDetectionResult detection = {};
    if (detectionObj != nullptr) {
        detection = extractDetectionResult(env, detectionObj);
    }

    uint32_t outputTexture = 0;
    IrisSdkError err = iris_sdk_apply_beauty_texture_v2(
        static_cast<uint32_t>(inputTexture),
        &outputTexture,
        width, height,
        &config,
        detectionObj ? &detection : nullptr
    );

    if (err != IRIS_SDK_OK) {
        return 0;
    }

    return static_cast<jint>(outputTexture);
}

JNIEXPORT jint JNICALL
Java_com_meroomon_irissdk_IrisSDKNative_applyFaceWarp(
    JNIEnv* env, jclass clazz,
    jint inputTexture,
    jint width, jint height,
    jfloat slimFace,
    jfloat thinChin,
    jfloat enlargeEyes,
    jobject detectionObj) {

    IrisDetectionResult detection = {};
    if (detectionObj != nullptr) {
        detection = extractDetectionResult(env, detectionObj);
    }

    uint32_t outputTexture = 0;
    IrisSdkError err = iris_sdk_apply_face_warp(
        static_cast<uint32_t>(inputTexture),
        &outputTexture,
        width, height,
        slimFace, thinChin, enlargeEyes,
        detectionObj ? &detection : nullptr
    );

    if (err != IRIS_SDK_OK) {
        return static_cast<jint>(inputTexture);  // 실패 시 원본 반환
    }

    return static_cast<jint>(outputTexture);
}

// 헬퍼 함수: Java BeautyFilterV2 → C IrisBeautyConfigV2
IrisBeautyConfigV2 extractBeautyConfigV2(JNIEnv* env, jobject configObj) {
    IrisBeautyConfigV2 config = iris_sdk_beauty_config_default_v2();

    jclass clazz = env->GetObjectClass(configObj);

    config.enabled = env->GetBooleanField(configObj,
        env->GetFieldID(clazz, "enabled", "Z")) ? 1 : 0;
    config.intensity = env->GetFloatField(configObj,
        env->GetFieldID(clazz, "intensity", "F"));
    config.smoothing = env->GetFloatField(configObj,
        env->GetFieldID(clazz, "smoothing", "F"));
    config.brightness = env->GetFloatField(configObj,
        env->GetFieldID(clazz, "brightness", "F"));
    config.soft_focus = env->GetFloatField(configObj,
        env->GetFieldID(clazz, "softFocus", "F"));
    config.whitening = env->GetFloatField(configObj,
        env->GetFieldID(clazz, "whitening", "F"));
    config.color_balance = env->GetFloatField(configObj,
        env->GetFieldID(clazz, "colorBalance", "F"));
    config.wrinkle_remove = env->GetFloatField(configObj,
        env->GetFieldID(clazz, "wrinkleRemove", "F"));
    config.slim_face = env->GetFloatField(configObj,
        env->GetFieldID(clazz, "slimFace", "F"));
    config.enlarge_eyes = env->GetFloatField(configObj,
        env->GetFieldID(clazz, "enlargeEyes", "F"));
    config.thin_chin = env->GetFloatField(configObj,
        env->GetFieldID(clazz, "thinChin", "F"));
    config.use_gpu = env->GetBooleanField(configObj,
        env->GetFieldID(clazz, "useGpu", "Z")) ? 1 : 0;
    config.roi_only = env->GetBooleanField(configObj,
        env->GetFieldID(clazz, "roiOnly", "Z")) ? 1 : 0;
    config.protect_eyes = env->GetBooleanField(configObj,
        env->GetFieldID(clazz, "protectEyes", "Z")) ? 1 : 0;
    config.protect_lips = env->GetBooleanField(configObj,
        env->GetFieldID(clazz, "protectLips", "Z")) ? 1 : 0;
    config.downscale_factor = env->GetIntField(configObj,
        env->GetFieldID(clazz, "downscaleFactor", "I"));

    return config;
}

} // extern "C"
```

---

## 4. 단위 테스트

**파일**: `bindings/android/src/androidTest/java/com/meroomon/irissdk/BeautyV2Test.java`

```java
@RunWith(AndroidJUnit4.class)
public class BeautyV2Test {

    @Before
    public void setUp() {
        IrisSDK.initialize(getContext());
        IrisSDKNative.initGpuBeauty();
    }

    @After
    public void tearDown() {
        IrisSDKNative.releaseGpuBeauty();
    }

    @Test
    public void testDefaultConfig() {
        BeautyFilterV2 config = BeautyFilterV2.createDefault();
        assertEquals(0.5f, config.intensity, 0.01f);
        assertTrue(config.useGpu);
    }

    @Test
    public void testGpuTextureProcessing() {
        // OpenGL 컨텍스트 필요
        // GPU 텍스처 생성 및 처리 테스트
    }

    @Test
    public void testFaceWarpWithDetection() {
        // Face Mesh 검출 후 Face Warp 테스트
    }
}
```

---

## 5. 완료 기준

- [ ] C API 확장 (BeautyConfigV2, GPU 함수)
- [ ] 텍스처 소유권 관리 API (`iris_sdk_release_texture`)
- [ ] 소유권 규칙 문서화
- [ ] JNI 바인딩 구현
- [ ] Java 래퍼 클래스
- [ ] 텍스처 처리 인터페이스
- [ ] Face Warp JNI
- [ ] Android 테스트

---

## 6. 다음 작업

- **P2-W5-02**: 성능 프로파일링 및 최적화
