# P4-W1-02: 안정성 Baseline 측정 — StabilityLogger + Gate 1

## 작업 개요
- **Phase**: P4 (시각적 리얼리즘 — Visual Fidelity)
- **기간**: 2026-02-13 ~
- **상태**: ⏳ 대기
- **선행 조건**: P4-W1-01 (Blend Mode Core 확장) 완료
- **근거**: 브레인스토밍 Section 13, 18, 19, 22 합의

## 목표

`5d845d8` 커밋(One Euro Filter, mipmap, 다중 랜드마크, 동적 feather)의 안정화 효과를 **정량적으로 검증**하고, Phase 2(시각적 리얼리즘) 진입을 위한 **Gate 1 통과 여부를 판정**한다.

### Phase Gate 정책 (3자 합의)

```
Gate 1 (Phase 1 → Phase 2 진입):
  PASS (즉시 진입):     반경 σ < 0.3px, 중심 σ < 0.5px
  CONDITIONAL (재튜닝): 반경 σ 0.3~0.8px → 파라미터 조정 후 재측정 1회 허용
  FAIL (진입 불가):     반경 σ > 0.8px → 근본 원인 분석 필요
```

**근거**: 브레인스토밍 Section 22(A1) — Pass/Marginal/Fail 기준 통일.

## 구현 사항

### 1. StabilityLogger 유틸리티 클래스

**파일**: `android/demo-app/src/main/java/com/irislenssdk/demo/util/StabilityLogger.kt`

```
기능:
- CSV 파일로 프레임별 raw/filtered 값 기록
- 5초 세그먼트 단위 시작/종료
- adb pull로 기기에서 추출 가능
- 디버그 빌드에서만 동작
```

### 2. CSV 로그 스키마 (22 컬럼)

| # | 컬럼명 | 타입 | 단위 | 설명 |
|---|--------|------|------|------|
| 1 | `frame_id` | int | - | 프레임 순번 |
| 2 | `timestamp_ms` | long | ms | System.nanoTime() / 1e6 |
| 3 | `face_detected` | bool | - | FaceMesh 유효 여부 |
| 4 | `raw_left_cx` | float | norm | 왼쪽 홍채 중심 X (필터 전) |
| 5 | `raw_left_cy` | float | norm | 왼쪽 홍채 중심 Y (필터 전) |
| 6 | `raw_left_r` | float | norm | 왼쪽 홍채 반경 (필터 전) |
| 7 | `flt_left_cx` | float | norm | 왼쪽 홍채 중심 X (필터 후) |
| 8 | `flt_left_cy` | float | norm | 왼쪽 홍채 중심 Y (필터 후) |
| 9 | `flt_left_r` | float | norm | 왼쪽 홍채 반경 (필터 후) |
| 10 | `raw_right_cx` | float | norm | 오른쪽 홍채 중심 X (필터 전) |
| 11 | `raw_right_cy` | float | norm | 오른쪽 홍채 중심 Y (필터 전) |
| 12 | `raw_right_r` | float | norm | 오른쪽 홍채 반경 (필터 전) |
| 13 | `flt_right_cx` | float | norm | 오른쪽 홍채 중심 X (필터 후) |
| 14 | `flt_right_cy` | float | norm | 오른쪽 홍채 중심 Y (필터 후) |
| 15 | `flt_right_r` | float | norm | 오른쪽 홍채 반경 (필터 후) |
| 16 | `eyelid_lt` | float | norm | 왼쪽 상안검 경계 (필터 후) |
| 17 | `eyelid_lb` | float | norm | 왼쪽 하안검 경계 (필터 후) |
| 18 | `eyelid_rt` | float | norm | 오른쪽 상안검 |
| 19 | `eyelid_rb` | float | norm | 오른쪽 하안검 |
| 20 | `hold_active` | bool | - | temporal hold 활성 여부 |
| 21 | `hold_remaining` | int | frames | 잔여 hold 프레임 수 |
| 22 | `render_time_us` | long | μs | renderLensOverlay 소요 시간 |

**파일명 규칙**: `iris_stability_log_{timestamp}.csv`
**기록 트리거**: `renderLensOverlay()` 호출 시
**최대 기록 시간**: 30초 (세그먼트 단위, 수동 시작/종료)

### 3. 로그 수집 인터페이스

`CameraGLRenderer.kt`에서 raw/filtered 값 쌍을 StabilityLogger로 전달하는 콜백 인터페이스:

```kotlin
interface StabilityLogCallback {
    fun onFrame(
        frameId: Int,
        faceDetected: Boolean,
        rawLeft: FloatArray,    // [cx, cy, r]
        filteredLeft: FloatArray,
        rawRight: FloatArray,
        filteredRight: FloatArray,
        eyelids: FloatArray,    // [lt, lb, rt, rb]
        holdActive: Boolean,
        holdRemaining: Int,
        renderTimeUs: Long
    )
}
```

### 4. GPU Tier 판별

```kotlin
val gpuRenderer = GLES31.glGetString(GLES31.GL_RENDERER) ?: ""
val gpuTier = when {
    gpuRenderer.contains("Adreno 7", ignoreCase = true) -> GPU_TIER_HIGH
    gpuRenderer.contains("Adreno 6", ignoreCase = true) -> GPU_TIER_MID
    gpuRenderer.contains("Mali-G7", ignoreCase = true)  -> GPU_TIER_MID
    gpuRenderer.contains("Mali-G5", ignoreCase = true)  -> GPU_TIER_LOW
    else -> GPU_TIER_MID // 안전한 기본값
}
```

GPU tier 정보도 CSV 헤더에 포함하여 기기별 분석 가능하게 한다.

### 5. 디버그 HUD (선택적)

실시간 `filteredRadius - rawRadius` 차이를 화면에 표시:

```
[HUD] L: σ=0.2px  R: σ=0.3px  FPS: 30  Hold: OFF
```

## 테스트 시나리오

| 시나리오 | 동작 | 시간 | 주요 확인 항목 |
|----------|------|------|---------------|
| S1-정지 | 정면 주시 | 5초 | 반경 σ, 중심 σ |
| S2-좌우 | 좌→우→좌 반복 | 5초 | 필터 지연, 오버슈트 |
| S3-깜빡임 | 자연 깜빡임 10회 | 10초 | hold 활성/복귀, eyelid 전이 |
| S4-근접 | 30cm→60cm 이동 | 5초 | 반경 변화 추적, deadband 동작 |
| S5-측면 | 30° 좌우 회전 | 5초 | 중심 추적, 마스킹 안정성 |

## Pass/Fail 기준 (5초 정지 기준)

| 지표 | 산출 방법 | Pass | Conditional | Fail |
|------|-----------|:---:|:---:|:---:|
| 반경 안정성 (σ) | `std(flt_left_r) × detH` | < 0.3px | 0.3~0.8px | > 0.8px |
| 중심 안정성 (σ) | `std(flt_left_cx) × detW` | < 0.5px | 0.5~1.0px | > 1.0px |
| 필터 지연 | `max(flt - raw 위상차)` | < 2fr | 2~4fr | > 4fr |
| Hold 빈도 | `count(hold_active) / total` | < 5% | 5~15% | > 15% |
| 렌더 시간 | `avg(render_time_us)` | < 2ms | 2~4ms | > 4ms |

## 디바이스 테스트 대상

| GPU 계열 | 대표 기기 | 검증 항목 |
|----------|-----------|-----------|
| Adreno 7xx | Pixel 8 / Galaxy S24 | 기본 동작, mipmap, 프레임 레이트 |
| Adreno 6xx | Pixel 6 / Galaxy S21 | One Euro 부하, 셰이더 호환성 |
| Mali-G7xx | Galaxy S23 (Exynos) | `pow` 연산 비용, 텍스처 필터링 |
| Mali-G5x | 중저가 기기 | fragment shader 병목, 메모리 |

**주의사항**:
- `glGenerateMipmap`은 일부 구형 Mali에서 비동기 동작 → 첫 프레임 검은 텍스처 가능
- One Euro Filter의 `kotlin.math.exp` 호출이 JIT 최적화 전 초기 프레임에서 느릴 수 있음

## One Euro Filter 현재 파라미터

| 파라미터 | 값 | 설정 근거 |
|----------|-----|-----------|
| `minCutoff` | 15.0 | CPU 경로에서 검증된 값 이식 |
| `beta` (중심) | 0.5 | 시선 이동 시 빠른 추적 필요 |
| `beta` (반경) | 0.25 | 반경은 급변할 이유 없음 → 안정성 우선 |
| `beta` (eyelid) | 0.3 | 깜빡임 응답과 안정성의 중간점 |
| `deadband` | 0.0003 | ~0.5px@1920. 서브픽셀 노이즈 억제 |

Gate 1 Conditional 시 이 파라미터를 조정 후 재측정.

## 검증 체크리스트

- [ ] StabilityLogger 빌드 성공
- [ ] CSV 파일 `adb pull` 정상 추출
- [ ] S1 시나리오 5초 로그 수집 성공
- [ ] 반경 σ 계산 결과 Pass 기준 충족
- [ ] GPU tier 판별 정상 동작

## 다음 단계

- Gate 1 **PASS** → P4-W1-03 (Luminance Tint 셰이더) 착수
- Gate 1 **CONDITIONAL** → One Euro 파라미터 재튜닝 → 재측정 1회
- Gate 1 **FAIL** → 근본 원인 분석 필요 → P4-W1-03 착수 불가

## 변경 이력

| 날짜 | 내용 |
|------|------|
| 2026-02-13 | 작업 계획 문서 작성 |
