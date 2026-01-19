# SDK 성능 비교 분석 리포트

**작성일**: 2026-01-19
**비교 대상**: IrisLens SDK vs MediaPipe (CPU/GPU)
**테스트 환경**: Android Demo App (Build 13-14)

---

## 테스트 데이터

| 파일 | SDK 타입 | GPU | Build |
|------|----------|-----|-------|
| `benchmark_thread_b13_20260116_182633.csv` | IrisLens SDK | ❌ | 13 |
| `benchmark_mediapipe_cpu_b14_20260116_185332.csv` | MediaPipe | ❌ | 14 |
| `benchmark_mediapipe_gpu_b14_20260116_185442.csv` | MediaPipe | ✅ | 14 |

---

## 핵심 지표 요약

| 지표 | **IrisLens SDK** | **MediaPipe CPU** | **MediaPipe GPU** |
|------|-----------------|-------------------|-------------------|
| **평균 Latency** | **~12ms** ✅ | ~33ms | ~40ms |
| **최소 Latency** | 5-6ms | 23-25ms | 29-34ms |
| **최대 Latency** | 56ms | 95ms | 94ms |
| **P95 Latency** | **24-26ms** ✅ | 48-49ms | 55-58ms |
| **P99 Latency** | **35-38ms** ✅ | 55-57ms | 59-65ms |
| **Drop Rate** | **1-2%** ✅ | ~35% ⚠️ | ~76% ❌ |
| **메모리 (PSS)** | **~190MB** ✅ | ~265MB | ~315MB |
| **Native Heap** | **~75MB** ✅ | ~120MB | ~105MB |
| **FPS (raw)** | 21-24 | 25-30 | 21-27 |

---

## 상세 분석

### 1. 레이턴시 성능

```
IrisLens SDK:  ████░░░░░░░░░░░░░░░░  12ms (기준)
MediaPipe CPU: ████████████░░░░░░░░  33ms (2.75x 느림)
MediaPipe GPU: ████████████████░░░░  40ms (3.3x 느림)
```

**분석**:
- IrisLens SDK는 MediaPipe 대비 **2.7~3.3배 빠른 추론 속도**
- 33ms 목표 기준, IrisLens SDK만 여유있게 달성
- MediaPipe는 프레임 버짓(33ms) 근처 또는 초과

### 2. 프레임 드롭률 (사용자 경험 핵심)

```
IrisLens SDK:  ██░░░░░░░░░░░░░░░░░░  ~2%  ✅ 매끄러운 AR 경험
MediaPipe CPU: ███████░░░░░░░░░░░░░  ~35% ⚠️ 눈에 띄는 끊김
MediaPipe GPU: ███████████████░░░░░  ~76% ❌ 심각한 프레임 손실
```

**분석**:
- **IrisLens SDK**: 98% 프레임 처리 → 실시간 AR에 적합
- **MediaPipe CPU**: 65% 프레임 처리 → 3프레임 중 1프레임 드롭
- **MediaPipe GPU**: 24% 프레임 처리 → 4프레임 중 3프레임 드롭 (사용 불가)

### 3. 메모리 효율성

```
IrisLens SDK:  ████████████████████  190MB (기준)
MediaPipe CPU: ██████████████████████████░  265MB (+40%)
MediaPipe GPU: █████████████████████████████████  315MB (+66%)
```

**분석**:
- IrisLens SDK가 **40~66% 적은 메모리** 사용
- 저사양 기기에서 더 안정적인 동작 기대

---

## 성능 우위 비교

| 항목 | 우위 | 배수/비율 |
|------|-----|----------|
| **추론 속도** | 🏆 IrisLens SDK | 2.7~3.3배 빠름 |
| **안정성 (Drop Rate)** | 🏆 IrisLens SDK | 17~38배 낮은 드롭률 |
| **메모리 효율** | 🏆 IrisLens SDK | 40~66% 적은 메모리 |
| **P95 레이턴시** | 🏆 IrisLens SDK | 2배 낮음 |
| **원시 FPS** | MediaPipe CPU | 5-10% 높음 (드롭 제외 시) |

---

## MediaPipe GPU 성능 저하 원인 분석

MediaPipe GPU 모드가 CPU보다 오히려 성능이 나쁜 이유:

1. **GPU 컨텍스트 전환 오버헤드**: 카메라 → CPU → GPU → CPU → 렌더링 파이프라인에서 병목
2. **메모리 복사 비용**: GPU 텍스처와 CPU 버퍼 간 데이터 전송 지연
3. **동기화 대기**: GPU 연산 완료 대기 시간이 프레임 버짓 초과
4. **Face Mesh 전체 계산**: 468개 랜드마크 전체 계산으로 불필요한 연산

---

## 결론 및 권장사항

### 최종 선택: **IrisLens SDK (CPU + InferenceThread)**

**선택 이유**:
1. ✅ 가장 낮은 레이턴시 (12ms) - 33ms 목표 대비 충분한 여유
2. ✅ 가장 낮은 드롭률 (2%) - 매끄러운 AR 경험 보장
3. ✅ 가장 효율적인 메모리 사용 - 저사양 기기 지원
4. ✅ 안정적인 성능 - 최대 레이턴시도 56ms로 관리 가능

### 권장 설정

```kotlin
// Android 권장 설정
val config = IrisSDKConfig(
    gpuEnabled = false,        // CPU 모드 사용
    inferenceThread = true,    // 별도 스레드에서 추론
    // ... 기타 설정
)
```

### 향후 개선 방향

1. **GPU 파이프라인 최적화**: Zero-copy 텍스처 공유로 GPU 모드 개선 검토
2. **Eye-Only 모델**: Face Mesh 대신 눈 영역만 처리하는 경량 모델로 추가 성능 향상
3. **적응형 품질**: 기기 성능에 따른 동적 해상도/정확도 조절

---

## 부록: 원본 데이터 위치

- `docs/demo_app/benchmark_thread_b13_20260116_182633.csv`
- `docs/demo_app/benchmark_mediapipe_cpu_b14_20260116_185332.csv`
- `docs/demo_app/benchmark_mediapipe_gpu_b14_20260116_185442.csv`
