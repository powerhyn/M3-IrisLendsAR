# P4-W3-03: skinQuality API + 프리셋

| 항목 | 내용 |
|------|------|
| **작업 ID** | P4-W3-03 |
| **유형** | 구현 |
| **상태** | ✅ 완료 |
| **근거 문서** | P4-W3-01 (브레인스토밍), P4-W3-02 (셰이더/파이프라인/매핑) |
| **선행 조건** | P4-W3-02 완료 (Freq Sep 파이프라인 + mapSkinQuality 동작) |
| **작성일** | 2026-03-03 |
| **일정** | Day 5~6 (2일) |
| **후속 문서** | P4-W3-04 (Temporal/Tier), P4-W3-05 (튜닝/릴리즈) |

---

## 1. 목표

외부 API에 `skinQuality` 단일 파라미터를 추가하고, 프리셋 API를 제공한다.

> **참고**: `mapSkinQuality()` 매핑 함수는 모듈 경계 보존을 위해
> `GPUBeautyBackend` 내부 static 메서드로 P4-W3-02에서 구현 완료.
> 본 문서는 외부 API 레이어만 담당한다.

### 1.1 완료 조건

- [x] `BeautyFilterConfigV2`에 `skinQuality` 필드 추가 (기본값 0.0) — P4-W3-02에서 완료
- [x] C API 함수 추가: `iris_sdk_set_skin_quality()`, `iris_sdk_get_skin_quality()`
- [x] 프리셋 enum + API 추가 (`IrisBeautyPreset`, `iris_sdk_set_beauty_preset()`)
- [x] `applyTextureId()`에서 `config.skinQuality` → `GPUBeautyBackend::mapSkinQuality()` 호출 연결 — P4-W3-02에서 완료
- [x] `skinQuality=0` → 기존 Bilateral 경로 (하위 호환 확인) — 테스트 검증 완료
- [x] `skinQuality>0` → Freq Sep 경로 활성화 확인 — 테스트 검증 완료

### 1.2 실패 기준 (No-Go)

- 기존 API 사용자에게 동작 변경 발생 (하위 호환 깨짐)
- `skinQuality=0`에서 Freq Sep 코드 실행됨

---

## 2. 변경 대상 파일

| 파일 | 변경 내용 | 영향도 |
|------|----------|--------|
| `cpp/include/iris_sdk/beauty_filter.h` | `skinQuality` 필드 + C API + 프리셋 enum | 중간 |
| `cpp/src/beauty_filter.cpp` | skinQuality 기본값/검증 로직 | 중간 |
| `cpp/src/gpu/gpu_beauty_backend.cpp` | `applyTextureId()`에서 skinQuality → mapSkinQuality 호출 연결 | 낮음 |

> **변경되지 않는 파일**: `beauty_processor.h/cpp` — `mapSkinQuality()`는
> `GPUBeautyBackend` 내부에 있으므로 BeautyProcessor는 수정하지 않음.

---

## 3. API 설계

### 3.1 BeautyFilterConfigV2 확장

```cpp
// beauty_filter.h — 구조체에 필드 추가
typedef struct BeautyFilterConfigV2 {
    // ... 기존 필드 유지 ...

    // === NEW: Freq Sep 피부 보정 ===
    float skinQuality;    // 0.0~1.0, 기본값: 0.0 (비활성)
                          // 0.0 = 처리 없음 (바이패스)
                          // 0.6 = 기본 추천값
                          // 1.0 = 최대 스무딩
} BeautyFilterConfigV2;
```

> **ABI 안정성 참고**: `BeautyFilterConfigV2`는 현재 pre-release 단계이므로
> 필드 추가가 자유롭다. **외부 SDK 릴리즈 전에 반드시 version/size 핸드셰이크를
> 추가해야 한다** (예: `uint32_t struct_size` 필드를 첫 번째 멤버로 배치하여
> 바이너리 호환성 보장). 이 작업은 릴리즈 준비 단계(P4-W3-05 이후)에서 처리한다.

### 3.2 C API 추가

```cpp
// beauty_filter.h — C API (기존 IRIS_SDK_EXPORT + IrisSdkError 패턴 준수)
IRIS_SDK_EXPORT IrisSdkError iris_sdk_set_skin_quality(float quality);  // 0.0~1.0
IRIS_SDK_EXPORT IrisSdkError iris_sdk_get_skin_quality(float* out_quality);
```

> **스타일 참고**: 기존 beauty_filter.h의 C API 규약을 따른다.
> `iris_sdk_set_beauty_filter_v2()` 등과 동일하게 `IRIS_SDK_EXPORT` 매크로와
> `IrisSdkError` 반환 타입을 사용한다. getter는 out 파라미터 패턴을 따른다.

### 3.3 skinQuality와 기존 smoothing의 관계

```
skinQuality > 0 이면:
  → Freq Sep 파이프라인 활성화
  → 기존 smoothing 파라미터는 무시됨

skinQuality == 0 이면:
  → Freq Sep 바이패스
  → 기존 smoothing이 적용됨 (Bilateral, 하위 호환)
```

### 3.4 프리셋

```cpp
// beauty_filter.h에 추가
typedef enum IrisBeautyPreset {
    IRIS_BEAUTY_PRESET_NATURAL  = 0,   // skinQuality=0.3
    IRIS_BEAUTY_PRESET_MODERATE = 1,   // skinQuality=0.5
    IRIS_BEAUTY_PRESET_STRONG   = 2,   // skinQuality=0.8
    IRIS_BEAUTY_PRESET_CUSTOM   = 3    // 사용자 직접 설정
} IrisBeautyPreset;

IRIS_SDK_EXPORT IrisSdkError iris_sdk_set_beauty_preset(IrisBeautyPreset preset);
```

### 3.5 applyTextureId() 내부 연결

`GPUBeautyBackend::applyTextureId()`에서 skinQuality를 읽어 내부 매핑을 호출:

```cpp
// applyTextureId() 내부 — skinQuality 분기 (P4-W3-02 §4.4와 연동)
if (config.skinQuality > 0.0f && roi_ptr && roi_ptr->isValid()) {
    int face_width = static_cast<int>(roi_ptr->face_rect.width);
    auto freq_sep_params = GPUBeautyBackend::mapSkinQuality(
        config.skinQuality, face_width
    );
    // ... P4-W3-04의 temporal filter → P4-W3-02 §4.3의 mask 업로드 + Freq Sep 분기로 진행
} else if (config.skinQuality > 0.0f) {
    // roi_ptr null 또는 invalid → Freq Sep 불가, Bilateral fallback
    // (얼굴 미검출 상태에서 skinQuality > 0이어도 안전하게 처리)
}
```

> **안전성 참고**: `roi_ptr`는 얼굴 미검출 시 nullptr일 수 있다.
> `roi_ptr->face_rect.width` 접근 전 반드시 null + validity 체크가 필요하다.

---

## 4. 테스트

### 4.1 단위 테스트

| 테스트 | 검증 내용 |
|--------|----------|
| `test_freq_sep_bypass` | `skinQuality=0` → Freq Sep 미실행, 기존 경로 동작 |
| `test_freq_sep_clamp` | `skinQuality=-0.5` → bypass, `skinQuality=1.5` → clamp 1.0 |
| `test_freq_sep_backward_compat` | 기존 smoothing API 단독 사용 시 동작 변경 없음 |
| `test_preset_mapping` | NATURAL→0.3, MODERATE→0.5, STRONG→0.8 |

### 4.2 통합 테스트

| 테스트 | 검증 내용 |
|--------|----------|
| `test_skinquality_api_roundtrip` | set → get 값 일치 |
| `test_skinquality_activates_freqsep` | skinQuality=0.6 → Freq Sep 파이프라인 실행 |
| `test_preset_activates_freqsep` | MODERATE 프리셋 → skinQuality=0.5 → Freq Sep 실행 |

---

## 5. 실행 일정

| Day | 작업 | 산출물 | 완료 기준 |
|-----|------|--------|----------|
| **5** | `BeautyFilterConfigV2`에 skinQuality 추가 + C API + 프리셋 enum | beauty_filter.h/cpp 변경 | 기존 API 하위 호환 유지, 새 API 동작 |
| **6** | applyTextureId() skinQuality 분기 연결 + 단위/통합 테스트 | 테스트 통과 | 바이패스/활성화 동작 확인, 프리셋 매핑 검증 |

---

## 6. 리스크

| 리스크 | 확률 | 대응 |
|--------|------|------|
| 기존 API 사용자 호환성 이슈 | 낮 | skinQuality=0 기본값으로 완전 바이패스 |
| 프리셋 값 조정 필요 | 중 | P4-W3-05 튜닝 단계에서 최종 확정 |

---

## 변경 이력

| 날짜 | 변경 내용 | 작성자 |
|------|----------|--------|
| 2026-03-03 | 통합 문서에서 API/매핑 분리 | Claude |
| 2026-03-03 | 리뷰 반영: mapSkinQuality를 GPUBeautyBackend로 이동 (P4-W3-02), beauty_processor.h/cpp 변경 대상에서 제거, applyTextureId() 연결 로직 추가 (§3.5) | Claude |
| 2026-03-03 | 리뷰 2차 반영: §3.5 roi_ptr null/validity 체크 추가, §3.1 ABI pre-release 안정성 노트 추가 | Claude |
| 2026-03-03 | 리뷰 3차 반영: §3.2/§3.4 C API 시그니처를 IRIS_SDK_EXPORT + IrisSdkError 패턴으로 수정, getter를 out 파라미터 패턴으로 변경 | Claude |
| 2026-03-04 | 구현 완료: C API 3개 함수 + IrisBeautyPreset enum + 단위/통합 테스트 11개 추가 (전체 46개 PASSED) | Claude |

---

## 7. 실행 내역

### 7.1 구현 산출물

| 파일 | 변경 내용 | 라인 |
|------|----------|------|
| `beauty_filter.h` | `IrisBeautyPreset` enum + C API 선언 3개 | 255~297 |
| `beauty_filter.cpp` | C API 구현 3개 함수 | 522~561 |
| `test_beauty_config_v2.cpp` | 단위/통합 테스트 11개 추가 | 신규 테스트 수트 2개 |

### 7.2 테스트 결과

| 테스트 수트 | 테스트 수 | 결과 |
|------------|----------|------|
| BeautyFilterConfigV2Test | 14 | ✅ PASSED |
| BeautyFilterConfigV2CAPI | 8 | ✅ PASSED |
| FreqSepParamsTest | 13 | ✅ PASSED |
| SkinQualityCAPITest (신규) | 5 | ✅ PASSED |
| BeautyPresetTest (신규) | 6 | ✅ PASSED |
| **합계** | **46** | **✅ 전체 PASSED** |

### 7.3 검증 결과

| 검증 항목 | 결과 |
|----------|------|
| skinQuality=0 → Bilateral 하위호환 | ✅ `BackwardCompatDefaultZero`, `ZeroBypassesFreqSep` |
| skinQuality>0 → Freq Sep 활성화 | ✅ `EnabledWhenSkinQualityPositive` |
| set/get round-trip 일치 | ✅ `SetAndGetRoundTrip` |
| 범위 초과 거부 | ✅ `SetRejectsOutOfRange` |
| 경계값 수용 (0.0, 1.0) | ✅ `SetAcceptsBoundaryValues` |
| nullptr 안전성 | ✅ `GetRejectsNullptr` |
| 프리셋 매핑 (NATURAL→0.3, MODERATE→0.5, STRONG→0.8) | ✅ 각각 테스트 통과 |
| CUSTOM 프리셋 → 값 유지 | ✅ `CustomPresetKeepsCurrentValue` |
| 유효하지 않은 프리셋 거부 | ✅ `InvalidPresetReturnsError` |
