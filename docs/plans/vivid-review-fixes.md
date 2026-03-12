# Vivid 리뷰 이슈 수정 계획

**Date**: 2026-03-12
**Base**: 리뷰 리포트 `.full-review/05-final-report.md`
**범위**: P0 2건 + P1 5건 = 총 7건

---

## P0 — 즉시 수정 (2건)

### Fix-1: SHADER-01 — smoothstep reversed edges
- **파일**: `cpp/src/gpu/shader_sources.cpp:660`
- **변경**: `smoothstep(0.4, 0.0, sat)` → `1.0 - smoothstep(0.0, 0.4, sat)`
- **이유**: GLSL ES 3.10 spec에서 edge0 ≥ edge1일 때 undefined behavior

### Fix-2: API-01 — applyTexture(TextureHandle) vivid 통합
- **파일**: `cpp/src/gpu/gpu_beauty_backend.cpp:605-718`
- **변경 요약**:
  1. enabled 가드에 `needsVivid` 추가 (line 622)
  2. `buildEffectiveConfig` 이후 vivid 패스 추가 (softFocus 다음)
  3. 0-pass 시 output이 미초기화 pong을 가리키는 버그 수정 — `current_input == input_tex`이면 input을 그대로 반환
- **주의**: 이 경로는 원래 구현 계획에서 "범위 밖"으로 명시했으나, 리뷰에서 공개 API가 깨진 상태임이 확인되어 수정 포함

---

## P1 — 다음 릴리즈 전 (5건)

### Fix-3: ROBUST-01 — NaN/Inf 방어
- **파일**: `cpp/include/iris_sdk/beauty_filter.h` (clampf 람다)
- **변경**: `clampf` 내부에 NaN guard 추가 → `std::isnan(v) || std::isinf(v) ? lo : (v < lo ? lo : (v > hi ? hi : v))`
- **이유**: `<`/`>` 비교는 NaN을 통과시킴. 모든 입력 경로(C API, Java public 필드 직접 설정 등) 영향
- **부가**: `toCppConfigV2()` 끝에 `BeautyFilterConfigV2Helper::clamp(config)` 호출 추가

### Fix-4: A-M-02 — vivid 셰이더 non-fatal
- **파일**: `cpp/src/gpu/gpu_beauty_backend.cpp` (initializeShaders, executeVividPass)
- **변경**:
  1. vivid 셰이더 컴파일 실패 시 `LOGW` + 계속 진행 (return false 제거)
  2. `executeVividPass()` 시작에 `if (vivid_program_ == 0) return;` 가드 추가

### Fix-5: H-01 — ROI 계산 3중 중복 제거
- **파일**: `cpp/src/sdk_api_v2.cpp` (2곳), `cpp/src/gpu/gpu_beauty_backend.cpp` (1곳)
- **변경**: 공통 인라인 함수 추출 → 3곳에서 호출
- **위치**: `beauty_filter.h` (또는 별도 유틸 헤더)에 `computeExpandedROI()` 추가

### Fix-6: M-04/A-H-02 — SoftFocus 후 스왑 무조건화
- **파일**: `cpp/src/gpu/gpu_beauty_backend.cpp:1904`
- **변경**: `if (pong && needsVivid)` → `if (pong)` (다른 패스와 동일한 무조건 스왑)

### Fix-7: ABI-01 — POD 구조체 버전링 검토
- **이번에는 검토 + 설계만 수행**, 구현은 iOS/Flutter 바인딩 작업 시 일괄 적용
- **문서화**: 결론을 이 문서 하단에 기록

---

## 수정 순서

1. Fix-1 (SHADER-01) — 1줄, 독립적
2. Fix-6 (SoftFocus 스왑) — 1줄, 독립적
3. Fix-4 (vivid non-fatal) — 10줄, 독립적
4. Fix-3 (NaN 방어) — clampf 수정 + toCppConfigV2 clamp 호출
5. Fix-2 (applyTexture vivid) — 가장 큰 변경, Fix-1/3/4/6 반영 후 진행
6. Fix-5 (ROI 중복 제거) — 리팩터링, 기능 변경 없음
7. Fix-7 (ABI 검토) — 문서만

## 검증

- `cd cpp/cmake-build-debug && cmake --build . --parallel` 빌드 확인
- `ctest` 기존 테스트 통과 확인

---

## 수정 결과

| Fix | 상태 | 비고 |
|-----|------|------|
| Fix-1 (SHADER-01) | ✅ 완료 | `shader_sources.cpp:660` 1줄 수정 |
| Fix-2 (API-01) | ✅ 완료 | `gpu_beauty_backend.cpp:605-724` applyTexture 재작성 |
| Fix-3 (ROBUST-01) | ✅ 완료 | `beauty_filter.h` clampf NaN-safe + `sdk_api_v2.cpp` clamp 호출 |
| Fix-4 (A-M-02) | ✅ 완료 | initializeShaders LOGW + executeVividPass 가드 |
| Fix-5 (H-01) | ✅ 완료 | `beauty_roi_manager.h`에 `computeExpandedFaceRect()` 추출, 3곳 교체 |
| Fix-6 (M-04) | ✅ 완료 | SoftFocus 스왑 무조건화 |
| Fix-7 (ABI-01) | 📝 검토 완료 | 아래 참조 |

### Fix-7: ABI 버전링 검토 결론

**현재 상황**: `BeautyFilterConfigV2`(C++)와 `IrisBeautyConfigV2`(C API) 모두 POD 구조체. 필드 추가 시 ABI 호환성이 깨짐.

**결론**: 현재 단계에서는 `struct_size` 필드 도입을 **보류**합니다.
- SDK가 프리릴리즈 상태이며, 바인딩(Android JNI)이 항상 SDK와 동일 버전으로 빌드됨
- iOS/Flutter 바인딩이 아직 미구현 — 이 바인딩 구현 시 ABI 버전링을 일괄 설계
- `struct_size` 도입 시 기존 모든 초기화 코드(C, Java, JNI)에 `sizeof()` 설정이 필요하여 변경 범위가 큼
- **향후 계획**: iOS/Flutter 바인딩 작업 시 `uint32_t struct_size`를 첫 번째 필드로 추가하고, 런타임에 `struct_size < expected`면 새 필드를 기본값으로 처리하는 패턴 적용

### 빌드 검증 결과

- **컴파일**: 수정된 모든 소스 파일 정상 컴파일 ✅
- **테스트**: 715개 중 710개 통과 (99%). 실패 5건은 기존 이슈 (TFLite 링커, GPU context null, FreqSep mapping)
- **변경과 무관한 기존 실패**: `FrameProcessorTest` abort(2건), `GPUBeautyBackendTest` null context(1건), `FreqSepMappingTest`(2건)
