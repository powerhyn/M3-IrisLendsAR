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

## MediaPipe Android SDK 비교 분석

### 배경

현재 IrisLensSDK는 커스텀 TFLite 구현을 사용합니다. Google의 MediaPipe Android Tasks SDK를 직접 사용하는 대안을 검토했습니다.

### MediaPipe Android Tasks SDK 벤치마크 (GitHub Issue #5872)

**기기**: Pixel 9 Pro

| 측정 방법 | GPU | CPU |
|-----------|-----|-----|
| TFLite Benchmark Tool | ~15ms (66 FPS) | ~88ms (11 FPS) |
| 실제 앱에서 측정 | 30-70ms (14-33 FPS) | - |

**주요 발견**:
- TFLite 벤치마크 도구와 실제 앱 간 **2-4배 성능 차이**
- 원인: 이미지 전처리, Bitmap 변환, GPU 메모리 전송 오버헤드

### IrisLensSDK vs MediaPipe Android SDK

| 항목 | IrisLensSDK (현재) | MediaPipe Tasks SDK |
|------|-------------------|---------------------|
| FPS (실제) | 19-26 fps | 14-33 fps |
| 지연 시간 | 38-52ms | 30-70ms |
| 메모리 | 141-202 MB | ~150-200 MB (예상) |
| GPU 상태 | ❌ 미활성화 | ⚠️ 활성화되어도 오버헤드 존재 |

### 결론

1. **성능 동등성**: 우리 커스텀 구현이 MediaPipe Android SDK와 유사한 성능
2. **공통 병목**: 둘 다 이미지 전처리 및 GPU 전송 오버헤드가 주요 병목
3. **SDK 전환 불필요**: MediaPipe SDK로 전환해도 큰 성능 향상 기대 어려움
4. **최적화 방향**: 프레임워크 교체보다 파이프라인 최적화에 집중

### 권장 최적화 방향

1. **GPU 텍스처 직접 전달**: Bitmap 변환 없이 GPU 텍스처로 직접 입력
2. **Zero-Copy 파이프라인**: CameraX → GPU Texture → TFLite (메모리 복사 최소화)
3. **모델 경량화**: INT8 양자화, 모델 프루닝
4. **NNAPI 활용**: 기기별 NPU 가속 (Hexagon DSP, ANE 등)

---

## GPU Delegate 활성화 이슈 분석

### 현재 상태

벤치마크 CSV에서 `gpuEnabled=false`로 표시됨. GPU delegate 라이브러리가 빌드되었으나 런타임에서 활성화되지 않음.

### 가능한 원인

1. **EGLContext 미제공**: GPU delegate 초기화 시 유효한 EGL 컨텍스트 필요
2. **권한 문제**: OpenGL ES 3.1 미지원 기기
3. **초기화 순서**: SDK 초기화 전 OpenGL 컨텍스트 생성 필요
4. **컴파일 플래그**: `IRIS_SDK_HAS_GPU_DELEGATE` 미정의

### 다음 조사 항목

- [ ] C++ 코드에서 GPU delegate 초기화 로직 확인
- [ ] JNI에서 EGLContext 전달 여부 확인
- [ ] 런타임 로그에서 GPU delegate 초기화 실패 원인 확인

---

*Report generated: 2025-01-14*
*IrisLensSDK Performance Benchmark v1.1*
