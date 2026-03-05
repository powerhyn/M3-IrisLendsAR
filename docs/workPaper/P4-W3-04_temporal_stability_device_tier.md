# P4-W3-04: Temporal Stability + Device Tier 분기

| 항목 | 내용 |
|------|------|
| **작업 ID** | P4-W3-04 |
| **유형** | 구현 |
| **상태** | ✅ 완료 |
| **근거 문서** | P4-W3-01 (브레인스토밍), P4-W3-02 (셰이더/파이프라인/매핑), P4-W3-03 (API) |
| **선행 조건** | P4-W3-03 완료 (skinQuality API 동작) |
| **작성일** | 2026-03-03 |
| **일정** | Day 7~8 (2일) |
| **후속 문서** | P4-W3-05 (튜닝/릴리즈) |

---

## 1. 목표

One Euro Filter로 Freq Sep 파라미터의 temporal stability를 확보하고, GPU 성능 기반 디바이스 tier 분기를 구현한다.

### 1.1 완료 조건

- [x] One Euro Filter로 `blur_radius` 프레임 간 안정화
- [x] One Euro Filter로 mask 중심 좌표(face_rect center) 안정화 → 경계 flicker 방지
- [x] DeviceTier 판정 로직 (HIGH/MID/LOW) 구현
- [x] MID tier: 하이브리드 해상도 Freq Sep 경로 구현 (블러 half-res, Composite full-res)
- [x] LOW tier: Bilateral fallback 분기 확인
- [x] tier별 성능 프로파일링 (GPUProfiler)

### 1.2 실패 기준 (No-Go)

- One Euro Filter 적용 후 오히려 응답 지연 발생
- MID 하이브리드 해상도에서 비피부 영역 선명도 저하
- tier 판정 오류 (HIGH 기기가 LOW로 판정)

---

## 2. 변경 대상 파일

| 파일 | 변경 내용 | 영향도 |
|------|----------|--------|
| `cpp/include/iris_sdk/gpu/gpu_beauty_backend.h` | DeviceTier enum + One Euro Filter 멤버 + tier 관련 멤버 | 중간 |
| `cpp/src/gpu/gpu_beauty_backend.cpp` | applyTextureId() 내 필터링 + tier 분기 + 다운스케일 경로 | 높음 |

> **변경되지 않는 파일**: `beauty_processor.h/cpp` — One Euro Filter는
> `GPUBeautyBackend::applyTextureId()` 내부에서 `mapSkinQuality()` 직후에 적용되므로,
> BeautyProcessor는 FreqSepParams를 알 필요가 없다. `BeautyProcessor::process()`는
> `backend_->apply(frame_data, ..., config_, roi_ptr)` 호출만 담당한다.

---

## 3. Temporal Stability

### 3.1 One Euro Filter 적용 대상

| 대상 | 필터 파라미터 | 이유 |
|------|-------------|------|
| `blur_radius` | min_cutoff=0.5, beta=0.01 | 얼굴 크기 변화에 따른 radius 흔들림 방지 |
| `face_rect 중심 좌표` (cx, cy) | min_cutoff=1.0, beta=0.02 | face_rect 중심 jitter → **scissor 영역 안정화**. ⚠️ FreqSep 마스크 내용에는 영향 없음 (마스크는 computeROI()에서 face mesh 기반 생성, composite에서 UV 직접 샘플링) |

### 3.2 구현

One Euro Filter는 `GPUBeautyBackend` 내부에 배치한다.
`BeautyProcessor::process()`는 `backend_->apply(frame_data, ..., config_, roi_ptr)` 호출만 하므로,
`FreqSepParams`를 전달할 채널이 없다. 따라서 필터링은 `applyTextureId()` 내부에서
`mapSkinQuality()` 직후, `executeFreqSepPipeline()` 직전에 수행한다.

```cpp
// gpu_beauty_backend.h — private 멤버 추가
OneEuroFilter skin_radius_filter_{0.5f, 0.01f, 1.0f};
OneEuroFilter mask_center_x_filter_{1.0f, 0.02f, 1.0f};
OneEuroFilter mask_center_y_filter_{1.0f, 0.02f, 1.0f};

// gpu_beauty_backend.cpp — applyTextureId() 내부
// (P4-W3-03 §3.5의 skinQuality 분기 내부에서)
if (config.skinQuality > 0.0f && roi_ptr && roi_ptr->isValid()) {
    int face_width = static_cast<int>(roi_ptr->face_rect.width);
    auto freq_sep_params = GPUBeautyBackend::mapSkinQuality(
        config.skinQuality, face_width
    );

    // One Euro Filter로 blur_radius temporal stabilization
    float raw_radius = static_cast<float>(freq_sep_params.blur_radius);
    float filtered_radius = skin_radius_filter_.filter(raw_radius);
    freq_sep_params.blur_radius = static_cast<int>(std::round(filtered_radius));

    // One Euro Filter로 face_rect center 안정화 (scissor 영역 안정화)
    // 효과 범위: face_rect.x/y를 직접 이동하여 scissor 경계 흔들림 방지
    // 제한사항: FreqSep 마스크 내용에는 영향 없음
    //   - 마스크는 computeROI()에서 face mesh 랜드마크 기반으로 이미 생성됨
    //   - composite 셰이더에서 uSkinMask를 vTexCoord UV로 직접 샘플링 (오프셋 없음)
    //   - 또한 ROI에 기본 padding/margin이 있어 scissor 안정화의 체감 효과도 제한적
    // TODO(P4-W3-04-R2): 진짜 마스크 안정화가 필요하면
    //   uSkinMask 샘플링에 UV offset 도입 또는 computeROI() 이전 안정화 검토
    float cx = roi_ptr->face_rect.x + roi_ptr->face_rect.width * 0.5f;
    float cy = roi_ptr->face_rect.y + roi_ptr->face_rect.height * 0.5f;
    float stable_cx = mask_center_x_filter_.filter(cx);
    float stable_cy = mask_center_y_filter_.filter(cy);
    float dx = stable_cx - cx;
    float dy = stable_cy - cy;
    roi_ptr->face_rect.x += dx;
    roi_ptr->face_rect.y += dy;

    // mask 업로드 + executeFreqSepPipeline() 진행
    // ...
}
```

> **설계 근거**: `mapSkinQuality()`와 `executeFreqSepPipeline()` 사이에 필터링을
> 끼워넣는 구조이므로, BeautyProcessor↔GPUBeautyBackend 간 인터페이스 변경 없이
> temporal stability를 달성할 수 있다. One Euro Filter 상태(이전 프레임 값)는
> `GPUBeautyBackend` 인스턴스의 수명과 동일하게 유지된다.

> **⚠️ 랜드마크 레벨 안정화 (별도 작업 검토 대상)**: blur_radius jitter의 근본
> 원인은 MediaPipe 랜드마크 → `face_rect` 계산 시 프레임 간 좌표 흔들림이다.
> 현재 코드베이스에서 One Euro Filter는 홍채 좌표에만 적용되어 있으며
> (`mediapipe_detector.cpp:209`), `beauty_roi_manager`의 `face_rect` 계산에는
> 프레임 간 안정화가 없다 (ROI 캐싱 100ms만 존재).
> 이상적으로는 `face_rect`에도 One Euro Filter를 적용하는 것이 더 근본적인
> 해결이지만, 이는 렌즈 렌더링/워핑 등 모든 하류 기능에 영향을 주므로
> Freq Sep 작업 범위를 넘는 **별도 작업**으로 검토한다.
> 현재 `blur_radius` 스칼라 필터링은 유효한 국소 완화로서 유지한다.

### 3.3 One Euro Filter 파라미터 튜닝 기준

| 시나리오 | 기대 동작 |
|----------|----------|
| 얼굴 거리 천천히 변화 | radius 부드럽게 추종 |
| 얼굴 거리 급격히 변화 | 0.3초 내 수렴 |
| 동일 거리 유지 | radius 변동 ±1 이내 |
| 얼굴 미검출 → 재검출 | 이전 radius에서 부드럽게 전이 |

---

## 4. Device Tier 분기

### 4.1 tier 판정 로직

```cpp
enum class DeviceTier { HIGH, MID, LOW };

DeviceTier detectDeviceTier() {
    auto* renderer = (const char*)glGetString(GL_RENDERER);
    // Adreno 7xx, Mali-Gxx → HIGH
    // Adreno 6xx, Mali-G7x → MID
    // 그 외 → LOW
}
```

### 4.2 tier별 처리

| tier | Freq Sep | 해상도 | 성능 목표 |
|------|----------|--------|----------|
| HIGH | 활성화 | 원본 | ≤8ms |
| MID | 활성화 | 하이브리드 (블러 1/2, Composite 원본) | ≤12ms |
| LOW | 비활성화 (Bilateral fallback) | 원본 | ≤6ms |

> **LOW tier fallback 강도**: `skinQuality > 0`이지만 LOW tier인 경우,
> P4-W3-02 §4.4의 공통 fallback 강도 정책(`executeSmoothingWithFallbackStrength`)을
> 적용하여 최소 smoothing 강도를 보장한다.

### 4.3 MID tier 하이브리드 해상도 처리

**설계 원칙**: 블러 연산(Pass 1, 2)만 half-res에서 수행하여 성능을 확보하고,
Composite(Pass 3)는 **반드시 full-res**에서 수행하여
비피부 영역(눈, 머리카락, 배경)의 선명도를 100% 보존한다.
High Frequency 추출은 Composite 셰이더 내 ALU 연산으로 인라인화 (P4-W3-02 §3.2.2).

> **⚠️ 이전 설계(전체 half-res→upscale)가 폐기된 이유**: 입력 전체를 half-res로
> 다운스케일하면, Composite의 `mix(orig, beauty, mask)`에서 `orig` 자체가
> half-res이므로 mask=0인 비피부 영역까지 bilinear 업스케일 열화가 발생한다.
> 이것은 뷰티 필터가 아니라 화면 훼손이다.

```
MID tier 하이브리드 해상도 파이프라인 (5서브패스):

[Pass 1a] Gaussian Blur H   (half-res FBO)  ← input_tex(full-res)를 half-res에 렌더
[Pass 1b] Gaussian Blur V   (half-res FBO)  → lowFreq (half-res, 보존)
[Pass 2a] Low Smooth H      (half-res FBO)  ← lowFreq(half-res) → temp
[Pass 2b] Low Smooth V      (half-res FBO)  → smoothedLow (half-res, 별도 텍스처)
[Pass 3]  Composite          (full-res FBO)  ← smoothedLow(half-res, GL_LINEAR)
                                              + lowFreq(half-res, GL_LINEAR) [high = orig - lowFreq 인라인]
                                              + orig(full-res) + mask
```

**핵심**: half-res 텍스처를 full-res 패스에서 샘플링할 때 GPU 하드웨어의
GL_LINEAR(bilinear interpolation)이 자동으로 부드러운 업샘플링을 수행한다.
lowFreq/smoothedLow는 정의상 저주파 성분이므로 bilinear 업샘플 품질 손실이 거의 없다.
Extract 패스 삭제로 full-res 텍스처 1장(highFreq_full)이 불필요해져 **메모리도 절약**.

```cpp
if (tier == DeviceTier::MID) {
    int half_w = width / 2;
    int half_h = height / 2;

    // 블러 연산용 half-res 텍스처 (lowFreq는 Composite까지 보존)
    auto* lowFreq_half = texture_pool_->acquireRenderTarget(half_w, half_h);
    auto* smoothedLow_half = texture_pool_->acquireRenderTarget(half_w, half_h);
    auto* temp_half = texture_pool_->acquireRenderTarget(half_w, half_h);

    if (!lowFreq_half || !smoothedLow_half || !temp_half) {
        if (lowFreq_half) texture_pool_->releaseTexture(lowFreq_half);
        if (smoothedLow_half) texture_pool_->releaseTexture(smoothedLow_half);
        if (temp_half) texture_pool_->releaseTexture(temp_half);
        // fallback: 공통 강도 정책 적용 (P4-W3-02 §4.4)
        executeSmoothingWithFallbackStrength(input_tex, output_fbo,
                                             width, height, config);
        return;
    }

    // half-res 블러 반경 (물리적 블러 범위를 유지하기 위해 절반)
    int half_radius = std::max(3, params.blur_radius / 2);

    // Pass 1a: Horizontal Gaussian (half-res)
    glUseProgram(freq_sep_gaussian_program_);
    glUniform2f(freq_sep_gaussian_uniforms_.uDirection, 1.0f / half_w, 0.0f);
    glUniform1i(freq_sep_gaussian_uniforms_.uRadius, half_radius);
    bindAndDraw(input_tex, temp_half->fbo_id, half_w, half_h);

    // Pass 1b: Vertical Gaussian (half-res) → lowFreq_half (보존 — Composite에서 필요)
    glUniform2f(freq_sep_gaussian_uniforms_.uDirection, 0.0f, 1.0f / half_h);
    bindAndDraw(temp_half->texture_id, lowFreq_half->fbo_id, half_w, half_h);

    // Pass 2a: Low Freq 추가 Gaussian H (half-res)
    // lowFreq_half를 READ만 하고 보존
    int low_radius = std::max(3,
        static_cast<int>(half_radius * params.low_freq_smooth_radius_ratio));
    glUseProgram(freq_sep_gaussian_program_);
    glUniform2f(freq_sep_gaussian_uniforms_.uDirection, 1.0f / half_w, 0.0f);
    glUniform1i(freq_sep_gaussian_uniforms_.uRadius, low_radius);
    bindAndDraw(lowFreq_half->texture_id, temp_half->fbo_id, half_w, half_h);

    // Pass 2b: Low Freq 추가 Gaussian V (half-res) → smoothedLow_half (별도 텍스처)
    glUniform2f(freq_sep_gaussian_uniforms_.uDirection, 0.0f, 1.0f / half_h);
    bindAndDraw(temp_half->texture_id, smoothedLow_half->fbo_id, half_w, half_h);

    // Pass 3: Composite (full-res)
    // Composite 셰이더 내에서 high = orig - lowFreq 인라인 계산
    // lowFreq_half, smoothedLow_half 모두 GL_LINEAR으로 full-res 업샘플
    glUseProgram(freq_sep_composite_program_);
    glUniform1f(freq_sep_composite_uniforms_.uHighFreqPreserve, params.high_freq_preserve);
    glUniform1f(freq_sep_composite_uniforms_.uAttenuationLow, params.attenuation_low);
    glUniform1f(freq_sep_composite_uniforms_.uAttenuationHigh, params.attenuation_high);
    // 유니폼 순서: uSmoothedLow, uLowFreq, uOriginal, uSkinMask
    bindTextures(smoothedLow_half->texture_id, lowFreq_half->texture_id,
                 input_tex, mask_tex);
    drawToFBO(output_fbo, width, height);

    texture_pool_->releaseTexture(lowFreq_half);
    texture_pool_->releaseTexture(smoothedLow_half);
    texture_pool_->releaseTexture(temp_half);
}
```

**텍스처 요약**:

| 텍스처 | 해상도 | 용도 | 비고 |
|--------|--------|------|------|
| `lowFreq_half` | half-res | Pass 1 결과 (Composite까지 보존) | GL_LINEAR 필수 |
| `smoothedLow_half` | half-res | Pass 2b 결과 (추가 블러) | GL_LINEAR 필수 |
| `temp_half` | half-res | 블러 핑퐁 임시 | |

> **이전 설계 대비 개선**: Extract 패스 삭제로 full-res 텍스처(`highFreq_full`) 불필요.
> MID tier는 half-res 텍스처 3장만으로 전체 파이프라인 완료.
> HIGH tier도 동일하게 `executeFreqSepPipeline()`이 5서브패스로 동작한다.

### 4.4 성능 프로파일링 포인트

```cpp
bool profiling = profiler_ && profiler_->isEnabled();
if (profiling) profiler_->begin("FreqSep_Total");
// ... 전체 Freq Sep 파이프라인 ...
if (profiling) profiler_->end("FreqSep_Total");

// 개별 패스는 P4-W3-02의 executeFreqSepPipeline()에서 이미 측정
```

---

## 5. 테스트

### 5.1 Temporal Stability 테스트

| 테스트 | 검증 내용 |
|--------|----------|
| `test_radius_filter_smooth` | 동일 입력 연속 10프레임 → radius 변동 ±1 이내 |
| `test_radius_filter_tracking` | 선형 변화 입력 → 지연 없이 추종 |
| `test_radius_filter_sudden_change` | 급격한 변화 → 0.3초(9프레임@30fps) 내 수렴 |

### 5.2 Device Tier 테스트

| 테스트 | 검증 내용 |
|--------|----------|
| `test_tier_high_fullres` | HIGH tier에서 원본 해상도 Freq Sep 실행 |
| `test_tier_mid_hybrid_resolution` | MID tier 하이브리드: 블러 패스 half-res, Composite full-res 정상 동작 (5서브패스) |
| `test_tier_low_bilateral` | LOW tier에서 기존 Bilateral 경로 실행 |
| `test_tier_mid_nonskin_sharpness` | MID tier에서 비피부 영역(눈, 머리카락) 선명도가 HIGH tier 대비 SSIM ≥ 0.95 |
| `test_tier_mid_texture_sizes` | MID tier 텍스처 3장 모두 half-res 확인 (lowFreq_half, smoothedLow_half, temp_half) |

### 5.3 성능 테스트

| 측정 항목 | 도구 | 기준 |
|-----------|------|------|
| 패스별 ms | GPUProfiler | 각 패스 개별 측정 |
| 전체 Freq Sep ms | GPUProfiler | tier별: HIGH ≤8ms, MID ≤12ms |
| 텍스처 메모리 | TexturePool::getStats() | 추가 3장 이내 (MID: +2장) |
| FPS 영향 | 전체 파이프라인 | 30fps 유지 |

---

## 6. 실행 일정

| Day | 작업 | 산출물 | 완료 기준 |
|-----|------|--------|----------|
| **7** | One Euro Filter temporal stability 적용 | gpu_beauty_backend.cpp 수정 (applyTextureId 내부) | 연속 프레임에서 radius/마스크 flicker 없음 |
| **8** | tier별 프로파일링 + 다운스케일 분기 (중간 버퍼 분리) | GPUProfiler 리포트 | HIGH ≤8ms, MID ≤12ms 확인, 버퍼 분리 동작 |

---

## 7. 리스크

| 리스크 | 확률 | 대응 |
|--------|------|------|
| One Euro Filter 파라미터 튜닝 수렴 안 됨 | 낮 | 기존 Iris tracking에서 검증된 파라미터 기반 조정 |
| MID 하이브리드 해상도에서 블러 경계 아티팩트 | 중 | half-res lowFreq의 GL_LINEAR 업샘플이 피부-비피부 경계에서 halo 유발 시, Composite 셰이더 마스크 블렌딩으로 1차 완화, 심각하면 3/4 스케일로 상향 |
| GPU 렌더러 문자열 기반 판정 부정확 | 중 | 벤치마크 기반 동적 판정으로 전환 (P4-W3-05에서) |
| MID 하이브리드 추가 텍스처(half-res 3장)로 메모리 압박 | 낮 | Extract 패스 삭제로 full-res 텍스처 불필요; half-res 3장은 full-res 1장 미만의 메모리 |

---

## 변경 이력

| 날짜 | 변경 내용 | 작성자 |
|------|----------|--------|
| 2026-03-03 | 통합 문서에서 Temporal/Tier 분리 | Claude |
| 2026-03-03 | 리뷰 반영: MID 다운스케일 중간 버퍼 분리 (§4.3), API 호출 패턴 수정 (texture_pool_->, profiler_->begin/end), 버퍼 분리 테스트 항목 추가 (§5.2) | Claude |
| 2026-03-03 | 리뷰 2차 반영: One Euro Filter를 BeautyProcessor → GPUBeautyBackend::applyTextureId() 내부로 이동 (§2, §3.2), beauty_processor.h/cpp 변경 대상에서 제거, Day 7 산출물 수정 (§6) | Claude |
| 2026-03-04 | 리뷰 4차 반영: LOW tier fallback에 공통 강도 정책 참조 추가 (§4.2), MID tier 텍스처 획득 실패 fallback에 executeSmoothingWithFallbackStrength 적용 (§4.3) | Claude |
| 2026-03-04 | Gemini 2차 리뷰 반영: MID tier 전체 half-res→upscale 설계를 하이브리드 해상도(블러 half-res, Extract/Composite full-res)로 전면 재설계 (§4.3), 완료 조건/실패 기준 업데이트 (§1.1, §1.2), tier 테이블 수정 (§4.2), 테스트 케이스 하이브리드 반영 (§5.2), 리스크 갱신 (§7), 랜드마크 레벨 안정화 별도 작업 노트 추가 (§3.2) | Claude |
| 2026-03-04 | Codex 리뷰 반영: "mask 경계값" One Euro 필터링 구현 구체화 — face_rect center(cx, cy)를 필터링 대상으로 확정 (§1.1, §3.1, §3.2), feather_radius는 15px 고정이므로 필터 불필요 확인 | Claude |
| 2026-03-04 | Gemini 3차 리뷰 반영: §4.3 MID tier를 5서브패스로 업데이트 — Extract 패스 삭제, lowFreq 보존, smoothedLow 별도 할당, highFreq_full 텍스처 불필요. §5.2 테스트/§7 리스크 반영 | Claude |
| 2026-03-04 | **구현 완료**: One Euro Filter temporal stability (blur_radius + mask center cx/cy), DeviceTier 판정 (Adreno/Mali/Apple/Desktop), MID tier 하이브리드 해상도 파이프라인, LOW tier Bilateral fallback, 코드 리뷰 반영 (Adreno 파싱 안전성, computeGaussianWeights 경계 보호, half-res 0 나누기 방지) | Claude |
| 2026-03-04 | **Codex 리뷰 피드백 반영**: (1) 마스크 중심 스무딩을 scissor 이전으로 이동 (효과 무효화 버그 수정), (2) OneEuroFilter release()/얼굴 추적 끊김 시 reset 추가, (3) std::stoi → std::strtol 교체 (예외 안전성), (4) detectDeviceTier() static public → private 인스턴스 메서드 변경. 미수정 항목(Q2/A2 파이프라인 중복 ~140줄)은 P4-W3-04-R1으로 분리 | Claude |
| 2026-03-04 | **Codex 2차 검증 반영**: mask center smoothing 효과 범위 정정 — scissor 안정화에만 유효, FreqSep 마스크 내용에는 무영향 (마스크는 computeROI()에서 face mesh 기반 생성, composite에서 UV 직접 샘플링). §3.1 필터 대상 표 수정, §3.2 코드 주석 정정. 코드 주석도 동일하게 정정 (gpu_beauty_backend.cpp:1571-1575). 진짜 마스크 안정화 필요 시 UV offset 도입 검토 (P4-W3-04-R2) | Claude |
