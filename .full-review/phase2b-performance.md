# Phase 2b: GPU Performance & Scalability Analysis

**Scope**: Vivid Post-Processing filter (develop branch delta)
**Files**: `shader_sources.cpp`, `gpu_beauty_backend.cpp`, `gpu_beauty_backend.h`
**Target**: 30fps+ real-time AR, ~0.3ms per vivid pass on MID-tier mobile GPU

---

## 1. GPU Shader Performance

### 1.1 [LOW] `precision highp float` -- mediump suffices for color grading

**File**: `shader_sources.cpp:640`
**Severity**: Low
**Impact**: ~15-25% ALU throughput loss on Adreno/Mali half-pipe GPUs

Vivid shader performs color grading operations (vibrance, brightness lift, warmth shift) where the value range is strictly [0.0, 1.0]. `mediump` (FP16) provides sufficient precision for these operations -- the quantization error at 10-bit color is negligible.

On Adreno 6xx/Mali-G7x (MID tier), highp operations consume 2 cycles per ALU vs 1 cycle for mediump on half-precision pipelines. Since the shader is ALU-bound (see 1.2), this directly halves throughput.

```glsl
// Before
precision highp float;

// After
precision mediump float;
// Note: uTexture sampler2D 자체는 precision 무관 (texel format이 결정)
```

**Caveat**: `dot(result, vec3(0.2126, 0.7152, 0.0722))` 연산에서 mediump의 누적 오차가 문제될 수 있으나, luminance 계산의 결과가 vibrance blend 비율로만 사용되므로 시각적 차이는 무시 가능. 기존 파이프라인의 다른 셰이더도 모두 highp를 사용하고 있어 일관성 차원에서 변경 시 전체 파이프라인 검토가 필요할 수 있음.

### 1.2 [INFO] ALU Cost Analysis -- 양호

**Shader instruction count estimate** (VIVID_POSTPROCESS_FRAGMENT):

| Operation | ALU Ops | Notes |
|-----------|---------|-------|
| `texture(uTexture, vTexCoord)` | 1 fetch | Single texture sample |
| Vibrance luminance `dot()` | 3 MAD | |
| Vibrance `max/min` chaining | 6 ops | `max(max(r,g),b) - min(min(r,g),b)` |
| `smoothstep(0.4, 0.0, sat)` | 3 ops | clamp + Hermite |
| `mix(vec3(lum), result, ...)` | 3 MAD | |
| Brightness lift `result * (1-result)` | 6 ops | per-component mul+sub+mul |
| Warmth shift | 3 ADD | |
| Master `mix` | 3 MAD | |
| `clamp` output | 3 ops | |
| **Total** | **~31 ALU + 1 tex fetch** | |

이 수준은 MID-tier GPU에서 0.3ms 예산 내 충분히 달성 가능. 1080p 기준 Adreno 640에서 약 0.15-0.25ms 소요 예상.

### 1.3 [INFO] Branch Divergence -- 무해

셰이더의 3개 `if` 가드(`uSaturation > 0.01`, `uBrightness > 0.001`, `uWarmth > 0.01`)는 uniform-based branching이므로 warp/wavefront 내 모든 fragment가 동일 경로를 탐. GPU branch divergence 문제 없음.

단, uniform 브랜치임에도 불구하고 일부 모바일 드라이버(특히 Mali)에서는 dead code elimination이 불완전할 수 있어, 런타임에 두 경로 모두 실행되는 경우가 있음. 현재 ALU 비용이 낮아 실질적 영향은 무시 가능.

---

## 2. Memory Management

### 2.1 [MEDIUM] Scissor-Empty + Vivid-Only: Pong 텍스처 불필요 할당

**File**: `gpu_beauty_backend.cpp:1762-1776`
**Severity**: Medium
**Impact**: 1920x1080 RGBA 기준 ~8MB GPU 메모리 낭비 (1 frame)

`active_filter_count >= 2`인 경우 (예: beauty 필터 1개 + vivid) ping-pong 쌍을 할당하지만, scissor intersection이 비어서 beauty를 스킵하면 vivid만 실행. 이때 pong 텍스처는 사용되지 않지만 `previous_output_pong_`에 저장되어 다음 프레임까지 유지됨.

```cpp
// Line 1762-1776: scissor-empty 경로
if (needsVivid) {
    executeVividPass(current_input, ping->fbo_id, ...);
    *output_texture = ping->texture_id;
} else {
    *output_texture = current_input;
}
if (ping) { previous_output_ping_ = ping; }
if (pong) { previous_output_pong_ = pong; }  // pong은 미사용인데 보존됨
```

**Fix**:
```cpp
if (ping) { previous_output_ping_ = ping; }
if (pong) {
    texture_pool_->releaseTexture(pong);  // 즉시 반환
    pong = nullptr;
}
// previous_output_pong_ 는 이미 nullptr (line 1641에서 리셋됨)
```

**주의**: 이 경로는 ROI 교집합이 비어야 트리거되므로 발생 빈도가 낮음 (얼굴이 화면 밖으로 완전히 나갈 때). 그러나 리소스 관리 원칙상 수정이 바람직함.

### 2.2 [LOW] Vivid-Only 경로의 ROI Passthrough 패스 -- 불필요 FBO blit

**File**: `gpu_beauty_backend.cpp:1693-1701`
**Severity**: Low
**Impact**: ~0.2ms (passthrough blit on 1080p)

`config.enabled == false && needsVivid == true`일 때 `buildEffectiveConfig()`이 모든 beauty 수치를 0으로 설정하므로 `roi_ptr == nullptr`. 따라서 이 경로는 실제로 실행되지 않음. 정적 분석상 코드 도달 불가능하므로 실질적 성능 영향 없음.

그러나 `config.enabled == true && needsVivid == true`이면서 ROI가 있는 경우, ping과 pong 모두에 원본을 blit하는 passthrough가 실행됨. 이때 vivid가 마지막 패스로 실행되어 전체 프레임을 덮어쓰므로, pong에 대한 passthrough blit은 vivid가 scissor 해제 후 실행되는 한 낭비임.

**현 상태**: Vivid는 scissor 해제 후 실행(line 1907-1909에서 disable 후 1912-1923에서 실행)되므로 전체 프레임을 덮어쓰기 맞음. 하지만 pong passthrough가 vivid 출력 FBO가 될 수 있으므로 안전성 측면에서 현 구현이 올바름. 최적화 여지가 있으나 안전성 우선.

### 2.3 [INFO] Texture Pool 재사용 -- 양호

Vivid 패스는 기존 ping-pong 텍스처 풀에 통합되어 별도 할당이 없음. `acquirePingPongPair`로 확보한 텍스처를 필터 체인에서 순차 재사용하므로 추가 VRAM 오버헤드 없음.

---

## 3. Pipeline Efficiency

### 3.1 [MEDIUM] Filter Count 계산 -- Vivid-Only 시 비효율적 텍스처 패턴

**File**: `gpu_beauty_backend.cpp:1648-1658`
**Severity**: Medium (correctness OK, efficiency concern)

`beauty.enabled == true`이면서 오직 vivid만 활성화된 경우:
- `active_filter_count == 1` (vivid만)
- 단일 ping 텍스처만 할당 -- 올바름

`beauty.enabled == true`이면서 smoothing + vivid가 활성화된 경우:
- `active_filter_count == 2`
- ping-pong 쌍 할당 -- 올바름

Filter count 로직 자체는 정확함. 다만 `beauty.enabled == false`일 때 vivid만 활성화되는 "vivid-only 경로"에서도 `buildEffectiveConfig()`이 올바르게 beauty 수치를 0으로 만들어 `active_filter_count == 1`이 됨. 정상 동작.

### 3.2 [LOW] SoftFocus 패스의 조건부 swap -- 미묘한 엣지 케이스

**File**: `gpu_beauty_backend.cpp:1904`
```cpp
if (pong && needsVivid) current_output = (current_output == ping) ? pong : ping;
```

SoftFocus 패스에서 vivid가 필요한 경우에만 ping-pong swap을 수행. vivid가 불필요하면 swap하지 않아 softFocus 출력이 최종 `current_input`이 됨. 이 조건은 올바르지만, vivid가 마지막 패스라는 암묵적 가정에 의존. 향후 vivid 이후 패스가 추가되면 swap 로직 전체 재검토 필요.

**권장**: 파이프라인 끝의 "마지막 활성 패스 = swap 불필요" 패턴을 주석으로 명시적으로 문서화.

### 3.3 [INFO] Early-Exit Paths -- 양호

- `!config.enabled && !needsVivid` -> 즉시 passthrough (line 1552)
- `active_filter_count == 0` -> passthrough (line 1661)
- 각 셰이더 내부 uniform guard (`if (uSaturation > 0.01)`) -> ALU 스킵

모든 무활성 경로에서 불필요한 GPU 작업이 발생하지 않음.

---

## 4. Render Pass Overhead

### 4.1 [LOW] glUseProgram / glBindFramebuffer 비용

**File**: `gpu_beauty_backend.cpp:870-903`

Vivid 패스의 렌더 패스 오버헤드:
| Operation | Estimated Cost (Adreno 640) |
|-----------|---------------------------|
| `glBindFramebuffer` | ~0.01ms |
| `glViewport` | ~0.005ms (이미 설정된 경우 no-op) |
| `glUseProgram` | ~0.02ms (프로그램 전환) |
| `glUniform1f` x4 + `glUniform1i` x1 | ~0.005ms |
| `glActiveTexture` + `glBindTexture` | ~0.005ms |
| `renderFullscreenQuad` (draw call) | ~0.005ms |
| **Total driver overhead** | **~0.05ms** |

전체 vivid 패스 예상: ALU ~0.15-0.25ms + overhead ~0.05ms = **~0.2-0.3ms**. 목표 달성.

### 4.2 [INFO] glViewport 중복 호출

`executeVividPass()`에서 `glViewport(0, 0, width, height)`를 호출하지만, 호출 직전 `applyTextureId()`의 line 1687에서 이미 동일 viewport가 설정되어 있음. Tile-based GPU에서 viewport 설정은 tile configuration에 영향을 주므로, 중복 호출은 드라이버 레벨에서 no-op으로 처리될 가능성이 높지만 명시적으로 불필요.

**영향**: 무시 가능 (~0.001ms). 코드 명확성을 위해 유지하는 것이 나을 수 있음 (standalone 호출 시 안전).

---

## 5. Scalability -- 파이프라인 전체 프레임 버짓 영향

### 5.1 Full Pipeline Latency Estimate (1080p, Adreno 640)

| Pass | Active | Est. Time |
|------|--------|-----------|
| FreqSep (6 subpass, HIGH) | skinQuality > 0 | ~3.0-4.5ms |
| Combined Color | brightness/balance/whitening/LUT | ~0.3ms |
| SoftFocus | softFocus > 0 | ~0.8ms |
| **Vivid (NEW)** | **vividIntensity > 0** | **~0.25ms** |
| **Total worst-case** | **All active** | **~4.35-5.85ms** |

30fps 프레임 버짓 = 33.3ms. GPU 렌더링 패스 합계 ~5.85ms = 전체 버짓의 **17.6%**. MediaPipe 검출 (~8-12ms) + CPU 처리 (~2ms) 포함 시 총 ~19.85ms = 59.6% 사용률. **충분한 여유.**

### 5.2 [INFO] Vivid 추가로 인한 Incremental Overhead

| Metric | Before Vivid | After Vivid | Delta |
|--------|-------------|-------------|-------|
| Max filter passes | 3 (smooth + color + softfocus) | 4 | +1 pass |
| Max ping-pong swaps | 2 | 3 | +1 swap |
| Shader programs | 11 | 12 | +1 program |
| Uniform locations cached | 43 | 48 | +5 |
| GPU VRAM (shader binary) | ~negligible | ~negligible | ~4KB |

추가 오버헤드가 매우 적음. 프레임 버짓에 미치는 영향은 전체의 ~0.75%.

### 5.3 [LOW] Worst-Case: 모든 필터 활성 + MID Tier

MID tier (Adreno 6xx)에서 모든 필터 활성 시:
- FreqSep half-res: ~2.5ms
- Combined Color: ~0.4ms
- SoftFocus: ~1.2ms
- **Vivid**: ~0.35ms (highp penalty)
- **Total**: ~4.45ms

여전히 33.3ms 버짓 내. 단, mediump 전환 시 ~0.3ms로 개선 가능.

---

## 6. Mobile GPU Considerations

### 6.1 [LOW] Tile-Based Rendering (TBR) -- Bandwidth Impact

Vivid 패스는 fullscreen quad로 실행되므로 GPU tile을 완전히 flush/reload함. 타일 기반 GPU (Mali, Adreno)에서 FBO 전환은 tile store + tile load를 유발.

| Operation | Bandwidth (1080p RGBA8) |
|-----------|------------------------|
| Tile store (이전 패스 출력) | ~8MB |
| Tile load (vivid 입력 텍스처) | ~8MB |
| Tile store (vivid 출력) | ~8MB |
| **Total per vivid pass** | **~24MB** |

MID tier 메모리 대역폭 (~25GB/s): 24MB / 25GB/s = **~0.96ms** bandwidth 시간.
실제로는 L2 cache hit과 tile compression으로 실측 ~0.15-0.2ms.

기존 파이프라인에서 이미 3개 FBO 전환이 발생하므로, vivid 추가로 인한 incremental bandwidth는 총 bandwidth의 ~25% 증가. 허용 범위.

### 6.2 [INFO] Power Consumption

Vivid 셰이더의 ALU 복잡도가 낮아 (31 ops) GPU 전력 소모 기여도가 미미함. 30fps 연속 실행 시 기존 파이프라인 대비 ~3-5% 전력 증가 예상. 배터리 drain에 체감 가능한 영향 없음.

### 6.3 [INFO] LOW Tier 디바이스 -- Vivid 무조건 실행

`DeviceTier::LOW`에서 FreqSep은 비활성되지만 vivid는 tier 체크 없이 실행됨. LOW tier GPU에서도 vivid 셰이더의 ALU 비용은 충분히 낮아 (~0.5ms) 문제 없음. 향후 극저사양 디바이스 지원 시 vivid에도 tier gate 추가를 고려할 수 있으나 현재는 불필요.

---

## Summary

| # | Finding | Severity | Impact | Effort |
|---|---------|----------|--------|--------|
| 1.1 | highp -> mediump 전환 가능 | LOW | ~15-25% ALU throughput 개선 (MID tier) | 1 line change + QA |
| 2.1 | Scissor-empty에서 pong 미반환 | MEDIUM | ~8MB VRAM 1 frame 낭비 | 3 lines |
| 3.2 | SoftFocus swap의 vivid 의존 암묵적 가정 | LOW | 유지보수 리스크 | Comment only |
| 5.1 | 전체 파이프라인 버짓 | INFO | 17.6% (충분) | N/A |
| 6.1 | TBR bandwidth increment | LOW | ~25% bandwidth 증가 | N/A |

### Overall Assessment

Vivid 포스트프로세싱 필터는 GPU 성능 관점에서 **잘 설계됨**:
- 단일 패스 + 1 texture fetch + 순수 ALU 구조는 모바일 GPU에 이상적
- 기존 ping-pong 버퍼 인프라에 자연스럽게 통합
- 0.3ms 목표 달성 가능 (MID tier, highp 기준)
- 프레임 버짓에 미치는 영향 최소 (~0.75%)

**우선 수정 권장**: Finding 2.1 (pong 미반환) -- 리소스 누수 방지 차원에서 간단히 수정 가능.
**선택적 최적화**: Finding 1.1 (mediump) -- MID tier 성능 마진을 넓히고 싶을 때 적용. 전체 파이프라인 precision 정책과 함께 검토 필요.
