# Phase 2: Security & Performance Review

**Date**: 2026-03-12
**Scope**: Vivid Post-Processing 필터 — develop 대비 +307/-12 lines, 10 files

---

## Security Findings (Phase 2A)

### HIGH: 2건

**S-01: NaN/Inf 파라미터 미검증으로 GPU 파이프라인 오염**
- CWE-20 (Improper Input Validation)
- `toCppConfigV2()`가 float 값을 검증 없이 GPU uniform으로 전달
- NaN uniform → 일부 Mali GPU에서 stall 유발 가능
- **Fix**: `toCppConfigV2()` 끝에 `BeautyFilterConfigV2Helper::clamp()` 호출 추가

**S-02: POD 구조체 ABI 비버전링**
- CWE-131 (Incorrect Calculation of Buffer Size)
- 이전 헤더로 컴파일한 바이너리가 새 .so 로드 시 vivid 필드에 garbage 값
- 중장기: `uint32_t struct_size` 첫 멤버로 추가

### MEDIUM: 3건

- **S-03**: `applyTexture(TextureHandle)` 경로에서 vivid 미적용 (CWE-684)
- **S-04**: `toCppConfigV2()`에서 `clamp()` 미수행 — 범위 초과 값이 셰이더 전달 (CWE-20)
- **S-05**: vivid-only + ROI 스킵 경로에서 텍스처 풀 반환 누락 → 장시간 GPU 메모리 누수 (CWE-401)

### LOW: 3건

- S-06: `feather_radius` 필드가 `toCppConfigV2`에서 누락 (기존 이슈)
- S-07: `calculateExpectedFrameSize`에서 int32 overflow 가능성 (극단적 해상도)
- S-08: `reinterpret_cast<IrisLandmark*>` — `static_assert` 부재

### INFO: 2건

- 신규 의존성 없음 확인
- 셰이더 인젝션 해당 없음 (컴파일 타임 상수)

---

## Performance Findings (Phase 2B)

### Overall Assessment: **잘 설계됨**

Vivid 셰이더: 단일 패스, 1 texture fetch, ~31 ALU ops → MID-tier에서 ~0.25-0.3ms. 프레임 버짓의 ~0.75% 추가.

### MEDIUM: 1건

**P-01: Scissor-empty + vivid-only 경로에서 pong 텍스처 미반환**
- `gpu_beauty_backend.cpp:1772` — ~8MB VRAM 1 frame 낭비
- ROI 교집합이 비어 beauty 스킵 시 미사용 pong 텍스처가 다음 프레임까지 보존
- Fix: 3줄 수정으로 즉시 `releaseTexture()` 호출

### LOW: 3건

- `precision highp float` → `mediump` 전환 가능 (~15-25% ALU 개선, 일관성 검토 필요)
- SoftFocus swap의 vivid 의존 암묵적 가정 (유지보수 리스크)
- TBR bandwidth ~25% 증가 (허용 범위)

### 파이프라인 버짓 (1080p, Adreno 640, worst-case)

| Pass | Est. Time |
|------|-----------|
| FreqSep (HIGH) | ~3.0-4.5ms |
| Combined Color | ~0.3ms |
| SoftFocus | ~0.8ms |
| **Vivid (NEW)** | **~0.25ms** |
| **Total** | **~4.35-5.85ms (17.6%)** |

30fps 프레임 버짓 33.3ms 중 GPU 렌더링 ~5.85ms + MediaPipe ~10ms + CPU ~2ms = ~59.6% 사용률. **충분한 여유**.

---

## Critical Issues for Phase 3 Context

1. **S-01 + S-04**: `toCppConfigV2()`에 clamp 추가 — NaN 방어 + 범위 검증을 한 번에 해결
2. **S-03 + H-02**: `applyTexture(TextureHandle)` 경로 vivid 누락 — 두 리뷰에서 공통 지적
3. **S-05 + P-01**: pong 텍스처 누수 — 보안/성능 양쪽에서 동일 문제 발견
4. **테스트 커버리지**: 위 3가지 이슈에 대한 테스트 존재 여부 확인 필요
