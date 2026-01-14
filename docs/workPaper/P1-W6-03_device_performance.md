# P1-W6-03: 실기기 성능 테스트

**태스크 ID**: P1-W6-03
**상태**: ⏳ 대기
**시작일**: -
**완료일**: -

---

## 1. 계획

### 목표
다양한 Android 실기기에서 SDK 성능 측정 및 최적화. 목표 성능(30fps+, 100MB 이하) 달성 검증

### 산출물
| 파일 | 설명 |
|------|------|
| `docs/PERFORMANCE_REPORT.md` | 성능 테스트 리포트 |
| `android/demo-app/benchmark/` | 벤치마크 테스트 코드 |
| `scripts/benchmark_android.sh` | 벤치마크 실행 스크립트 |

### 검증 기준
- [ ] 고급 기기: 60fps 달성
- [ ] 중급 기기: 30fps 달성
- [ ] 메모리 사용량 100MB 이하
- [ ] 검출 지연 33ms 이하
- [ ] 1시간 연속 실행 안정성

### 선행 조건
- P1-W6-02 CameraX 연동 완료

---

## 2. 분석

### 2.1 테스트 기기 선정

| 등급 | 기기 | SoC | RAM | 목표 |
|------|------|-----|-----|------|
| 고급 | Galaxy S21+ / Pixel 6 Pro | Exynos 2100 / Tensor | 8GB | 60fps |
| 중급 | Galaxy A52 / Pixel 4a | Snapdragon 750G / 730G | 6GB | 30fps |
| 저급 | Galaxy A12 / Redmi 9 | Helio G35 / Helio G80 | 4GB | 20fps |

### 2.2 성능 지표

| 지표 | 측정 방법 | 목표 |
|------|----------|------|
| FPS | 프레임 카운터 | ≥30fps |
| 검출 지연 | 타임스탬프 차이 | ≤33ms |
| 메모리 사용 | Android Profiler | ≤100MB |
| CPU 사용률 | Android Profiler | ≤50% |
| 배터리 소모 | Battery Historian | ≤10%/hour |
| 발열 | 표면 온도 | ≤45°C |

### 2.3 테스트 시나리오

```
시나리오 1: 기본 성능 (5분)
- 전면 카메라, 기본 설정
- 안정 상태 FPS 측정

시나리오 2: 스트레스 테스트 (30분)
- 연속 실행, 메모리 모니터링
- 프레임 드랍 카운트

시나리오 3: 장시간 테스트 (1시간)
- 배터리 소모, 발열, 안정성
- 메모리 누수 확인

시나리오 4: 다양한 조건
- 조명 변화 (밝음/어두움)
- 얼굴 각도 변화
- 거리 변화 (가까움/멀음)
```

---

## 3. 실행 내역

### 3.1 벤치마크 코드

```kotlin
// benchmark/PerformanceTracker.kt
class PerformanceTracker {
    private val frameTimestamps = LinkedList<Long>()
    private val windowSizeMs = 1000L  // 1초 윈도우
    private var totalFrames = 0L
    private var droppedFrames = 0L

    private var minLatency = Long.MAX_VALUE
    private var maxLatency = 0L
    private var sumLatency = 0L

    fun onFrameProcessed(latencyMs: Long) {
        val now = System.currentTimeMillis()

        // FPS 계산을 위한 타임스탬프 추가
        synchronized(frameTimestamps) {
            frameTimestamps.add(now)
            // 1초 이전 타임스탬프 제거
            while (frameTimestamps.isNotEmpty() &&
                   now - frameTimestamps.first > windowSizeMs) {
                frameTimestamps.removeFirst()
            }
        }

        // 지연 시간 통계
        totalFrames++
        sumLatency += latencyMs
        if (latencyMs < minLatency) minLatency = latencyMs
        if (latencyMs > maxLatency) maxLatency = latencyMs

        // 프레임 드랍 감지 (33ms 초과)
        if (latencyMs > 33) {
            droppedFrames++
        }
    }

    fun getCurrentFps(): Float {
        synchronized(frameTimestamps) {
            return frameTimestamps.size.toFloat()
        }
    }

    fun getAverageLatency(): Float {
        return if (totalFrames > 0) sumLatency.toFloat() / totalFrames else 0f
    }

    fun getStats(): PerformanceStats {
        return PerformanceStats(
            fps = getCurrentFps(),
            avgLatencyMs = getAverageLatency(),
            minLatencyMs = if (minLatency == Long.MAX_VALUE) 0L else minLatency,
            maxLatencyMs = maxLatency,
            totalFrames = totalFrames,
            droppedFrames = droppedFrames,
            dropRate = if (totalFrames > 0) droppedFrames.toFloat() / totalFrames else 0f
        )
    }

    fun reset() {
        frameTimestamps.clear()
        totalFrames = 0
        droppedFrames = 0
        minLatency = Long.MAX_VALUE
        maxLatency = 0
        sumLatency = 0
    }
}

data class PerformanceStats(
    val fps: Float,
    val avgLatencyMs: Float,
    val minLatencyMs: Long,
    val maxLatencyMs: Long,
    val totalFrames: Long,
    val droppedFrames: Long,
    val dropRate: Float
)
```

### 3.2 메모리 모니터링

```kotlin
// benchmark/MemoryMonitor.kt
class MemoryMonitor(private val context: Context) {

    fun getMemoryInfo(): MemoryInfo {
        val runtime = Runtime.getRuntime()
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE)
                as ActivityManager

        // JVM 메모리
        val jvmUsedMB = (runtime.totalMemory() - runtime.freeMemory()) / 1024 / 1024
        val jvmMaxMB = runtime.maxMemory() / 1024 / 1024

        // 네이티브 메모리
        val nativeHeapMB = Debug.getNativeHeapAllocatedSize() / 1024 / 1024

        // 전체 앱 메모리
        val memInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)

        val processInfo = Debug.MemoryInfo()
        Debug.getMemoryInfo(processInfo)
        val totalPssMB = processInfo.totalPss / 1024

        return MemoryInfo(
            jvmUsedMB = jvmUsedMB,
            jvmMaxMB = jvmMaxMB,
            nativeHeapMB = nativeHeapMB,
            totalPssMB = totalPssMB
        )
    }
}

data class MemoryInfo(
    val jvmUsedMB: Long,
    val jvmMaxMB: Long,
    val nativeHeapMB: Long,
    val totalPssMB: Int
)
```

### 3.3 벤치마크 화면

```kotlin
// ui/BenchmarkScreen.kt
@Composable
fun BenchmarkScreen(
    performanceTracker: PerformanceTracker,
    memoryMonitor: MemoryMonitor
) {
    val stats by remember {
        derivedStateOf { performanceTracker.getStats() }
    }
    val memoryInfo by remember {
        derivedStateOf { memoryMonitor.getMemoryInfo() }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        Text("Performance Benchmark", style = MaterialTheme.typography.headlineMedium)

        Spacer(modifier = Modifier.height(16.dp))

        // FPS
        StatRow("FPS", "%.1f".format(stats.fps), if (stats.fps >= 30) Color.Green else Color.Red)

        // Latency
        StatRow("Avg Latency", "%.1f ms".format(stats.avgLatencyMs),
                if (stats.avgLatencyMs <= 33) Color.Green else Color.Red)
        StatRow("Min/Max Latency", "${stats.minLatencyMs}/${stats.maxLatencyMs} ms")

        // Frames
        StatRow("Total Frames", stats.totalFrames.toString())
        StatRow("Dropped Frames", "${stats.droppedFrames} (${(stats.dropRate * 100).toInt()}%)",
                if (stats.dropRate < 0.05) Color.Green else Color.Red)

        Divider(modifier = Modifier.padding(vertical = 8.dp))

        // Memory
        Text("Memory Usage", style = MaterialTheme.typography.titleMedium)
        StatRow("JVM Heap", "${memoryInfo.jvmUsedMB}/${memoryInfo.jvmMaxMB} MB")
        StatRow("Native Heap", "${memoryInfo.nativeHeapMB} MB")
        StatRow("Total PSS", "${memoryInfo.totalPssMB} MB",
                if (memoryInfo.totalPssMB < 100) Color.Green else Color.Red)
    }
}

@Composable
private fun StatRow(label: String, value: String, color: Color = Color.Unspecified) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(label)
        Text(value, color = color)
    }
}
```

### 3.4 ADB 명령어

```bash
# 메모리 덤프
adb shell dumpsys meminfo com.irislenssdk.demo

# CPU 사용률
adb shell top -n 1 | grep irislenssdk

# GPU 렌더링 프로파일
adb shell dumpsys gfxinfo com.irislenssdk.demo

# 배터리 통계
adb shell dumpsys batterystats --charged com.irislenssdk.demo

# 시스템 트레이스 (Perfetto)
adb shell perfetto --txt -c - <<EOF
buffers: { size_kb: 32768 }
data_sources: { config { name: "linux.process_stats" } }
duration_ms: 10000
EOF
```

### 3.5 성능 리포트 템플릿

```markdown
# IrisLensSDK Performance Report

## Test Environment
- **Date**: YYYY-MM-DD
- **SDK Version**: 1.0.0-alpha01
- **Test Device**: [Device Name]
- **Android Version**: [Version]
- **App Version**: 1.0.0

## Results Summary

### FPS Performance
| Condition | FPS | Status |
|-----------|-----|--------|
| Normal lighting | XX | ✅/❌ |
| Low lighting | XX | ✅/❌ |
| Multiple faces | XX | ✅/❌ |

### Latency
| Metric | Value | Target |
|--------|-------|--------|
| Average | XX ms | ≤33ms |
| P95 | XX ms | ≤50ms |
| P99 | XX ms | ≤100ms |

### Memory Usage
| Metric | Value | Target |
|--------|-------|--------|
| JVM Heap | XX MB | - |
| Native Heap | XX MB | - |
| Total PSS | XX MB | ≤100MB |

### Stability (1-hour test)
| Metric | Value |
|--------|-------|
| Crashes | 0 |
| ANRs | 0 |
| Memory leaks | None |
| FPS degradation | <5% |

## Recommendations
1. [Optimization suggestions]
2. [Known issues]
```

---

## 4. 검증 결과

### 검증 항목

| 항목 | 결과 | 비고 |
|------|------|------|
| 고급 기기 60fps | ⏳ | - |
| 중급 기기 30fps | ⏳ | - |
| 메모리 100MB 이하 | ⏳ | - |
| 지연 33ms 이하 | ⏳ | - |
| 1시간 안정성 | ⏳ | - |

---

## 5. 이슈 및 학습

### 이슈
| ID | 내용 | 상태 | 해결방안 |
|----|------|------|----------|
| - | - | - | - |

### 결정 사항
| 결정 | 이유 |
|------|------|
| 1초 윈도우 FPS | 안정적인 FPS 측정 |
| PSS 기준 메모리 | 실제 물리 메모리 반영 |

### 학습 내용
- Android 메모리 프로파일링 기법
- 성능 벤치마킹 방법론
- ADB 성능 분석 명령어

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-01-12 | 태스크 문서 생성 |
