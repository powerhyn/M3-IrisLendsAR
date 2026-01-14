# IrisLensSDK Performance Report

## Test Environment

- **Date**: 2025-01-13
- **SDK Version**: 1.0.0-alpha
- **Test Duration**: 60 seconds
- **Mode**: CPU Only (GPU delegate not enabled)

### Device Information
- **Model**: (테스트 기기)
- **Android Version**: (버전)
- **CPU ABI**: arm64-v8a

---

## Executive Summary

| Metric | Measured | Target | Status |
|--------|----------|--------|--------|
| FPS | 21-26 fps | ≥30 fps | ❌ FAIL |
| Avg Latency | 10-17 ms | ≤33 ms | ✅ PASS |
| P95 Latency | 26-31 ms | ≤50 ms | ✅ PASS |
| Memory (PSS) | 141-202 MB | ≤100 MB | ❌ FAIL |
| Drop Rate | 2.8-4.1% | <5% | ✅ PASS |
| **Overall** | **3/5 Pass** | | ⚠️ |

---

## Detailed Results

### 1. FPS Performance

```
Target: ≥30 fps
Measured: 21-26 fps (stable state)
Status: ❌ FAIL (-4 to -9 fps from target)
```

**Observations**:
- Stable FPS around 21-26 during active detection
- FPS drops to ~2 when app loses focus (background)
- No significant FPS degradation over time (1 minute test)

**Time Series** (first 20 samples):
| Time (s) | FPS | Status |
|----------|-----|--------|
| 1 | 22 | ⚠️ |
| 2 | 23 | ⚠️ |
| 3 | 25 | ⚠️ |
| 4 | 26 | ⚠️ |
| 5 | 24 | ⚠️ |
| 6 | 24 | ⚠️ |
| 7 | 22 | ⚠️ |
| 8 | 24 | ⚠️ |
| 9 | 21 | ⚠️ |
| 10 | 21 | ⚠️ |

### 2. Latency Analysis

```
Target: ≤33 ms (average), ≤50 ms (P95)
Measured:
  - Average: 10-17 ms ✅
  - Min: 4 ms
  - Max: 1052 ms (outlier during background)
  - P95: 26-31 ms ✅
  - P99: 38-57 ms
Status: ✅ PASS
```

**Latency Distribution**:
- Normal operation: 4-44 ms
- Spike during background: 1052 ms (excluded from normal analysis)
- P95 within acceptable range

### 3. Memory Usage

```
Target: ≤100 MB (Total PSS)
Measured: 141-202 MB
Status: ❌ FAIL (+41 to +102 MB over target)
```

**Memory Breakdown**:
| Component | Range | Notes |
|-----------|-------|-------|
| Total PSS | 141-202 MB | Exceeds 100MB target |
| Native Heap | 59-102 MB | TFLite models loaded |
| JVM Heap | 6-20 MB | Kotlin objects |

**Memory Trends**:
- Initial: ~163 MB
- Peak: ~202 MB
- Post-GC: ~141 MB
- Trend: Slight increase over time, GC reclaims memory

### 4. Frame Drops

```
Target: <5% drop rate
Measured: 2.8-4.1%
Status: ✅ PASS
```

| Metric | Value |
|--------|-------|
| Total Frames | 1047 |
| Dropped Frames | 29 |
| Drop Rate | 2.77% |

### 5. Stability

```
Test Duration: 60 seconds
Crashes: 0
ANRs: 0
Memory Leaks: Not detected (minor)
FPS Degradation: <5%
Status: ✅ STABLE
```

---

## Performance Bottleneck Analysis

### 1. FPS Bottleneck (Primary Issue)

**Root Cause**: CPU-only TFLite inference

**Evidence**:
- 3 models run sequentially: face_detection → face_landmark → iris_landmark
- Each model takes ~5-15ms on CPU
- Combined inference: ~20-45ms per frame
- Maximum theoretical FPS: ~22-50 fps

**Solution**: Enable GPU delegate
- Expected improvement: 2-3x faster inference
- Target: 45-60 fps achievable

### 2. Memory Bottleneck (Secondary Issue)

**Root Cause**: Three TFLite models loaded simultaneously

**Breakdown**:
| Model | Estimated Size |
|-------|----------------|
| face_detection_short.tflite | ~1.5 MB |
| face_landmark.tflite | ~2.8 MB |
| iris_landmark.tflite | ~2.7 MB |
| TFLite Runtime | ~15 MB |
| OpenCV Runtime | ~30 MB |
| Interpreter Buffers | ~20-40 MB |
| **Total Native** | ~70-100 MB |

**Solutions**:
1. Model quantization (FP16 → INT8): ~50% size reduction
2. Lazy model loading
3. Shared interpreter buffers

---

## Optimization Roadmap

### Phase 1: GPU Acceleration (High Priority)
- [ ] Build TFLite GPU delegate with Bazel
- [ ] Integrate GPU delegate into MediaPipeDetector
- [ ] Expected FPS improvement: 30-60 fps

### Phase 2: Memory Optimization (Medium Priority)
- [ ] Quantize models to INT8
- [ ] Optimize buffer management
- [ ] Target: <100 MB PSS

### Phase 3: Advanced Optimizations (Low Priority)
- [ ] NNAPI delegate for compatible devices
- [ ] Model pruning and optimization
- [ ] Hexagon DSP delegate (Qualcomm devices)

---

## Test Data Reference

**Raw Data**: `docs/demo_app/benchmark_20260113_182747.csv`

**Data Schema**:
```csv
timestamp,fps,avgLatencyMs,minLatencyMs,maxLatencyMs,p95LatencyMs,
p99LatencyMs,totalFrames,droppedFrames,dropRate,
totalPssMB,nativeHeapMB,jvmUsedMB,gpuEnabled
```

**Sample Count**: 59 samples (1 second intervals)

---

## Conclusion

### Current State
- **FPS**: Below target, requires GPU acceleration
- **Latency**: Acceptable, within target range
- **Memory**: Above target, requires optimization
- **Stability**: Good, no crashes or significant issues

### Next Steps
1. **Immediate**: Build and integrate GPU delegate
2. **Short-term**: Re-run benchmark with GPU enabled
3. **Medium-term**: Memory optimization if still needed

### Expected After GPU Optimization
| Metric | Current | Expected | Target |
|--------|---------|----------|--------|
| FPS | 21-26 | 45-60 | ≥30 |
| Latency | 10-17 ms | 5-10 ms | ≤33 ms |
| Memory | 141-202 MB | ~150 MB | ≤100 MB |

---

*Report generated: 2025-01-13*
*IrisLensSDK Performance Benchmark v1.0*
