# Phase 1: Code Quality & Architecture Review

**Date**: 2026-03-12
**Scope**: Vivid Post-Processing 필터 — develop 대비 +307/-12 lines, 10 files

---

## Code Quality Findings (Phase 1A)

### Critical: 0건

### High: 2건

**H-01: ROI 계산 로직 3중 중복**
- Files: `sdk_api_v2.cpp` (2곳) + `gpu_beauty_backend.cpp` (1곳)
- face_rect → 픽셀 좌표 변환 + 20% 마진 추가 로직이 복사-붙여넣기. 공통 헬퍼 함수 추출 필요.

**H-02: applyTexture(TextureHandle) 경로에 vivid 패스 누락**
- File: `gpu_beauty_backend.cpp` lines 620-710
- `applyTextureId`와 달리 `applyTexture(TextureHandle&)`에서 `needsVivid` 체크 없음. enabled=false + vivid 활성 시 동작 불일치.

### Medium: 5건

- **M-01**: buildEffectiveConfig에서 vivid가 master intensity 영향받지 않는 것이 의도적이나 주석 부재
- **M-02**: warmth 셰이더 매직 넘버 (0.04, 0.02, 0.03) — 기존 관행과 일관되나 주석 보강 권장
- **M-03**: scissor-empty + vivid-only 경로에서 pong 텍스처 불필요 점유 (한 프레임)
- **M-04**: SoftFocus 후 ping-pong 스왑 조건이 `needsVivid`에 하드코딩 — 향후 패스 추가 시 깨질 수 있음
- **M-05**: C API `iris_sdk_default_beauty_config_v2_c()`의 enabled 기본값 불일치 (기존 이슈)

### Low: 3건

- L-01: VividUniforms 별도 구조체 — 기존 패턴(FreqSep) 따름, 적절
- L-02: JNI 필드 ID null 체크 — 빈틈없이 구현
- L-03: Vibrance sat 계산 edge case — 파이프라인 순서상 안전

---

## Architecture Findings (Phase 1B)

### Critical: 0건

### High: 2건

**A-H-01: POD 구조체 ABI 버전링 부재**
- `BeautyFilterConfigV2`, `IrisBeautyConfigV2` 구조체에 version 필드 없음
- 현재는 "끝에 추가" 패턴으로 유효하지만, iOS/Flutter 바인딩 확장 시 ABI 안정성 리스크
- 중장기적 `uint32_t version` 필드 도입 권장

**A-H-02: Ping-pong 버퍼 조건부 스왑 가독성**
- `if (pong && needsVivid)` 같은 "다음 패스를 아는" 조건 추가로 결합도 상승
- 스왑 규칙의 일관된 패턴화 필요 (차기 이터레이션)

### Medium: 3건

- **A-M-01**: ConfigV2 구조체 비대화 경향 (21→25 필드) — 포스트프로세싱 2개+ 시 서브 구조체 분리 검토
- **A-M-02**: Vivid 셰이더 실패가 전체 초기화를 중단 — FreqSep처럼 non-fatal 처리 권장
- **A-M-03**: Vivid가 raw config 사용 (effective 아님) — 의도적 설계이나 문서화 필요

### Low: 4건

- needsVivid 임계값 0.01f 매직 넘버 중복 → 상수화 권장
- buildEffectiveConfig vivid pass-through 의도 암묵적 → 주석 추가
- Vivid 전용 C API 부재 → 문서화로 충분
- CPU 경로 vivid 미지원 문서화 부재 → API doc 보강

---

## Positive Observations

1. **레이어 관통 일관성**: C++ → C API → JNI → Java 전 레이어에서 vivid 4필드 빈틈없이 반영
2. **buildEffectiveConfig 분리**: beauty disabled + vivid-only 경로를 깔끔하게 처리
3. **셰이더 효율성**: 텍스처 샘플 1회 + ALU 위주, ~0.3ms 타겟에 적합
4. **Scissor 해제 후 vivid 적용**: ROI 기반 beauty 후 전체 프레임 vivid — 정확한 순서
5. **ROI 교집합 empty 처리**: vivid-only 경로에서 누락 없이 처리
6. **기본값 0.0f 전략**: 하위 호환성 보장
7. **Profiler, uniform cache, release 정리 등 기존 패턴 100% 준수**

---

## Critical Issues for Phase 2 Context

Phase 2 (Security & Performance) 리뷰에서 주의해야 할 사항:

1. **성능**: Vivid 셰이더 `precision highp float` — mediump 전환 시 성능 이점 가능, 실기기 프로파일링 필요
2. **메모리**: scissor-empty 경로에서 pong 텍스처 불필요 점유 (M-03)
3. **보안/안정성**: POD 구조체 ABI 미버전링 상태에서 garbage 값 읽기 가능성 (방어적 처리 검증 필요)
4. **API 경로 불일치**: `applyTexture(TextureHandle)` 경로에 vivid 누락 (H-02) — 이 경로 사용 시 기능 미동작
