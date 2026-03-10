# Phase 2: Security & Performance Review

## 리뷰 대상: P4-W4-01e Luminance Sharpen 패스 (feature/P4-W4-01e vs develop)

---

## Security Findings

### [Medium] SEC-1: compositeRT 할당 후 GPU 리소스 유효성에 대한 암묵적 mutex 의존

- **파일**: `gpu_beauty_backend.cpp:1290-1296`
- **설명**: `acquireRenderTarget` 반환 후 `fbo_id`/`texture_id` 유효성은 `mutex_` 락에 의해 보호되나, 이 보호가 암묵적. 리팩토링 시 락 구조 변경 시 노출 가능.
- **권장**: `compositeRT->fbo_id != 0` 검증 추가 또는 스레드 안전성 문서화

### [Medium] SEC-2: `sharpen_amount` uniform 범위 검증 부재

- **파일**: `gpu_beauty_backend.cpp:1361`
- **설명**: `FreqSepParams`가 public struct이므로 외부에서 임의 값 설정 가능. `> 1.0`이면 ringing, `< 0`이면 blur.
- **권장**: `std::clamp(params.sharpen_amount, 0.0f, 0.5f)`

### [Low] SEC-3: `uTexelSize` division — width=0 간접 방어

- **파일**: `gpu_beauty_backend.cpp:1362`
- **설명**: `1.0f / width`에서 width=0이면 inf. 상위에서 `blur_w < 1` 검사로 간접 방어되나 명시적이지 않음.

### [Low] SEC-4: Sharpen 셰이더 실패 시 LOGE vs LOGW 불일치

- **파일**: `gpu_beauty_backend.cpp:289`
- **설명**: Graceful degradation인데 `LOGE`(error) 사용. `LOGW`가 의미적으로 정확.

### NONE: 버퍼 오버플로, Use-After-Free, GLSL UB — 없음

---

## Performance Findings

### [High] PERF-1: Full-res compositeRT 추가 할당 — MID 디바이스 메모리 대역폭 13.7% 추가

- **파일**: `gpu_beauty_backend.cpp:1290`
- **분석**:
  - 1080p RGBA8 = ~8.3MB 추가 할당
  - Composite→compositeRT write + Sharpen 6회 read + output write = ~58.3MB/frame 추가 대역폭
  - Mali-G52 기준 30fps에서 ~1.75GB/s = 대역폭 13.7% 추가 소비
- **권장**: MID 디바이스 실측 필수, LOW에서 sharpen 비활성화 정책 확인

### [Medium] PERF-2: 6회 texture fetch (요약에서 5회로 기재 — 실제 6회)

- **파일**: `shader_sources.cpp:589-597`
- **분석**: uTexture 5회(center+4neighbor) + uSkinMask 1회 = 6회. 1-texel 간격이므로 캐시 히트율 양호.
- **권장**: 현재 합리적. 추가 최적화 시 `textureGather` 검토 가능

### [Medium] PERF-3: Luminance ratio RGB 곱셈의 색상 왜곡 가능성

- **파일**: `shader_sources.cpp:601-603`
- **분석**: `center * ratio`에서 채도 높은 픽셀의 RGB 불균형 가능. sharpen_amount가 작아 ratio ~1.02 수준이므로 실질적 영향 미미하나 다양한 피부톤 QA 필요.

### [Low] PERF-4: `> 0.01f` 임계값이 매핑 최솟값 0.12와 불일치

- **파일**: `gpu_beauty_backend.cpp:1287`
- **설명**: `mapSkinQuality` 최솟값 0.12이므로 `> 0.01f`는 항상 true. 임계값 의도 불명확.

### [Low] PERF-5: 6패스 상태에서 LOW 디바이스 프레임 버짓 91% 소비 추정

- **분석**: Mali-G52 기존 5패스 ~28ms + Sharpen ~2ms = ~30ms (33ms의 91%). 추가 패스 여지 거의 없음.

---

## 30fps 달성 평가

| GPU 등급 | 추가 비용 | 33ms 대비 | 판정 |
|----------|----------|----------|------|
| HIGH (Adreno 7xx) | ~0.5ms | 1.5% | ✅ 안전 |
| MID (Mali-G78) | ~0.8ms | 2.4% | ✅ 안전 |
| MID-LOW (Mali-G52) | ~1.5-2ms | 4.5-6% | ⚠️ 실측 필요 |
| LOW | bilateral only | — | N/A (FreqSep 미사용) |

---

## Critical Issues for Phase 3 Context

- **PERF-1**: full-res RT 추가 대역폭 — 실 기기 프로파일링 테스트 필요
- **SEC-2/F-4**: sharpen_amount 방어적 클램핑 — 테스트에서도 경계값 검증 강화 필요
