/# GPU Beauty Rendering Issue Analysis Report

**Date:** 2026-02-05
**Analyzed File:** `cpp/src/gpu/gpu_beauty_backend.cpp`, `cpp/src/gpu/texture_pool.cpp`
**Reference:** `docs/workPaper/GPU_Beauty_Rendering_Issue.md`

## 1. Codebase verification
I have verified the current state of the C++ core code against the issue document.

*   **`glFinish()` Presence:** Confirmed. The `glFinish()` call exists in `GPUBeautyBackend::applyTextureId` (Line 880) before releasing `previous_output_ping_`. This matches the "Test Fix" described in the issue paper.
*   **Texture Release Logic:** The logic correctly delegates to `texture_pool_->releaseTexture` for internal textures.
*   **Ping-Pong Logic:** The logic correctly swaps `current_input` and `current_output` pointers.

## 2. Deep Dive Analysis

### A. The "Black Screen" Trigger (Smoothing 0 → Up)
The issue happens specifically when transitioning from a "Pass Skipped" state to a "Pass Executed" state.

**Scenario Trace:**
1.  **State A (Smoothing=0, Brightness=1):**
    *   `applyTextureId` is called.
    *   `acquirePingPongPair` allocates/reserves 2 textures (Ping, Pong).
    *   **Crucial:** These textures are **NOT used** because the filter blocks are skipped (`if (config.smoothing > 0.01f)`).
    *   The function returns `input_texture` (Source).
    *   `previous_output_ping_` is assigned the **unused** Ping texture.
2.  **State B (Smoothing=0.5):**
    *   `applyTextureId` is called.
    *   `glFinish()` executes.
    *   `previous_output_ping_` (the unused one from State A) is released.
    *   New Ping/Pong acquired.
    *   Smoothing Pass executes using `current_input` (Source) -> writes to `Ping`.
    *   `current_input` becomes `Ping`.
    *   Result returns `Ping`.

**Potential Root Causes:**

1.  **Texture Type Mismatch (High Probability)**
    *   The document mentions: `Camera OES Texture → RGBA Conversion → GPU Beauty`.
    *   **Hypothesis:** If the RGBA Conversion is somehow optimized out or bypassed when "Beauty is effectively off" (Smoothing=0), the `input_texture` passed to `GPUBeautyBackend` might be the raw OES texture.
    *   `GPUBeautyBackend` binds it as `GL_TEXTURE_2D`.
    *   **Result:** Sampling an OES texture as `sampler2D` results in undefined behavior (often black) without generating a GL Error in some drivers.
    *   **Action:** Verify `CameraGLRenderer.kt` ensures RGBA conversion happens **always** when `applyTextureId` is called, regardless of filter values.

2.  **Wasteful Resource Locking**
    *   Currently, `acquirePingPongPair` is called **unconditionally** at the start of `applyTextureId`.
    *   If `smoothing=0` and `brightness=1`, we lock 2 textures but output the Source.
    *   These 2 textures remain "in use" until the *next* frame starts.
    *   While logs say the pool isn't full, this puts unnecessary pressure on the pool.
    *   **Optimization Recommendation:** Move `acquirePingPongPair` inside the first `if` block that actually needs it, or check if any filter is active before acquiring.

3.  **Framebuffer Feedback Loop (Low Probability)**
    *   If `input_texture` happens to be the same ID as `ping->texture_id` (reused ID), and we bind `ping->fbo_id` for drawing, we get a feedback loop (Reading and Writing to same texture).
    *   This usually causes undefined results (black).
    *   However, `input_texture` usually comes from the Camera/Converter (External), while `ping` is internal, so IDs should be distinct.

## 3. Recommended Debugging Steps

**Do not modify the C++ core yet. Perform these checks first:**

1.  **Verify Input Texture Type:**
    *   In `CameraGLRenderer.kt` (or where `iris_sdk_detect` is called), log the texture ID being passed.
    *   Check if this ID changes when toggling Smoothing 0 -> 0.5.
    *   If the ID changes to the OES texture ID, the "RGBA Converter" is the culprit.

2.  **Shader Hardcode Test (If Step 1 fails):**
    *   Temporarily modify `BILATERAL_FILTER_FRAGMENT` to output solid RED (`gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);`).
    *   If the screen turns Red when Smoothing is enabled, the pipeline is working, and the issue is sampling the input texture (supports Type Mismatch theory).
    *   If the screen remains Black, the issue is FBO/Draw failure.

3.  **Optimize Acquisition (Refactoring):**
    *   After the bug is fixed, refactor `gpu_beauty_backend.cpp` to only call `acquirePingPongPair` if `needsSmoothing || needsCombined || needsSoftFocus` is true.

## 4. Conclusion
The `glFinish` fix addresses race conditions, but the "Smoothing 0 -> Up" black screen strongly suggests a **Logic/State issue**, likely related to the input texture being invalid (OES vs 2D) or the upstream converter state, rather than a GPU race condition inside the Beauty Backend itself.
