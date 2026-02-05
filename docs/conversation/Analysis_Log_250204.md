# Analysis of Log `250204_logcat.txt` (GPU Beauty Issue)

**Date:** 2026-02-05
**Log File:** `docs/demo_app/250204_logcat.txt`
**Reference:** `docs/workPaper/GPU_Beauty_Rendering_Issue.md`

## 1. Executive Summary
The "Black Screen" issue is caused by a **Double-Free / Ownership Conflict** between the Java/Kotlin layer and the C++ native layer.

*   **Symptom:** Screen goes black when Beauty Filter is enabled/adjusted.
*   **Root Cause:** `CameraGLRenderer.kt` explicitly releases the texture returned by the native backend. However, `GPUBeautyBackend` (C++) internally manages these textures (Ping-Pong) and expects to reuse them. The `releaseTexture` call from Kotlin causes `glDeleteTextures` to be executed on a texture that is still living in the C++ `TexturePool`.
*   **Mechanism:**
    1. Frame N: C++ returns Texture 3. C++ keeps a reference to it for reuse.
    2. Frame N+1: C++ returns Texture 4.
    3. **Kotlin detects output changed (3 -> 4) and calls `IrisLensSDK.releaseTexture(3)`.**
    4. **C++ `releaseTexture` implementation unconditionally calls `glDeleteTextures(3)`.**
    5. Frame N+2: C++ `TexturePool` reuses Texture ID 3 (it doesn't know it was deleted).
    6. Shader tries to sample from Texture 3 -> `glIsTexture(3)` returns false -> **Black Screen**.

## 2. Evidence from Logs

### A. The "Healthy" State (Smoothing=0.5)
Initially, the system works because the ping-pong buffer hasn't wrapped around or been corrupted yet, or the timing aligns such that the delete happens after use.

### B. The Failure Sequence (Around Timestamp 19:33:14.230)

1.  **Texture 3 is Active:**
    ```log
    Line 1381: PingPong acquired: ping(tex=3, fbo=2)...
    Line 1384: Beauty filter result: input=2, output=3...
    ```
    Texture 3 is the output for this frame.

2.  **Kotlin Releases Texture 3:**
    In the *next* frame (or shortly after), when output switches to 4:
    ```log
    Line 1392: Beauty filter result: input=2, output=4...
    Line 1393: IrisSDK-JNI ... nativeReleaseTexture called: texture=3
    Line 1394: GPUBeautyBackend ... Released texture 3
    ```
    **`Released texture 3` means `glDeleteTextures` was called.**

3.  **C++ Tries to Reuse Texture 3 (and Fails):**
    A few frames later, C++ re-acquires the "available" Texture 3 from its pool.
    ```log
    Line 1403: PingPong acquired: ping(tex=3, fbo=2)...
    Line 1404: CombinedColor: program=30, fbo=3, input=3, brightness=0.98
    Line 1405: CombinedColor: input_tex=3 valid=0
    ```
    **`valid=0` confirms that Texture 3 is no longer a valid GL texture object.**

## 3. Code Analysis

### A. Kotlin Side (`CameraGLRenderer.kt`)
Lines 325-330:
```kotlin
if (outputTexture != 0 && outputTexture != inputTexture) {
    // 이전 출력 텍스처가 있으면 해제
    if (beautyOutputTextureId != 0 && beautyOutputTextureId != outputTexture) {
        IrisLensSDK.releaseTexture(beautyOutputTextureId) // <--- CULPRIT
    }
    beautyOutputTextureId = outputTexture
    // ...
}
```
This logic assumes the caller owns the texture and must release it. This is incorrect for the managed `TexturePool` model.

### B. C++ Side (`gpu_beauty_backend.cpp`)
Lines 1045-1053 (`releaseTexture`):
```cpp
    // 텍스처 풀에서 관리하는 텍스처인지 확인 후 반환
    if (texture_pool_) {
        // ...
        // 텍스처 풀에 없는 경우에만 직접 삭제 (TODO: Check implementation missing)
        glDeleteTextures(1, &tex_id); // <--- UNCONDITIONAL DELETE
        LOGI("Released texture %u", tex_id);
    }
```
The code deletes the texture without checking if it belongs to the pool. Even if it checked, it should probably *ignore* the release request from Java if the pool owns it, because the pool handles lifecycle internally via `applyTextureId`'s `previous_output_ping_` logic.

## 4. Recommended Fix

**Option 1 (Quick Fix - Kotlin Side):**
Remove the `IrisLensSDK.releaseTexture(beautyOutputTextureId)` call in `CameraGLRenderer.kt`. The C++ backend's `TexturePool` recycles textures automatically.

**Option 2 (Robust Fix - C++ Side):**
Update `GPUBeautyBackend::releaseTexture` to check if `texture` is managed by `texture_pool_`.
- If yes: Do nothing (or warn "Ignored external release of pool texture").
- If no: Call `glDeleteTextures`.

**Recommendation:** Apply **Option 1** immediately to verify the fix, as it aligns with the "Internal Texture Pool" architecture.
