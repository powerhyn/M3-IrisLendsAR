# P6-W7 B4 자동감지 정확도 측정 리포트

**측정일:** 2026-05-29
**대상:** `android/demo-app/src/main/assets/lenses/*.png` (실제 42종 라인업)
**함수:** `iris_sdk::detectBakedLimbal` / `measureLimbalRatio` (실제 C++ 코어, OpenCV PNG 디코딩)
**ground-truth:** `docs/workPaper/P5-W3-05_brainstorm/15_asset_analysis.md §2.2` (육안 분류)

---

## 1. 결론 요약

| 항목 | 결과 |
|------|------|
| §5.2 원래 공식 (center = r<0.3) | **작동 불능** — 모든 렌즈 텍스처 42/42에서 측정 실패 |
| center 밴드 보정 후 최고 정확도 | **9/10** (10/10 불가) |
| 최적 파라미터 | `roi_inner=0.85, center_inner=0.40, center_outer=0.55, threshold=0.48` |
| 유일 미스 | `oh_bagel` (약-베이크인데 edge fade 약해 미검출) |
| **채택 결정** | **메타데이터 전용 (자동감지 런타임 권위 드롭)** — §1.16/§5.10 사전 합의 폴백 |

## 2. 결정적 발견: 원래 공식이 렌즈 텍스처에서 작동 불능

§5.2 자동감지 공식은 `center_lum = mean(pixels in r < 0.3)`로 정의됐다. 이는 **카메라 프레임의 홍채**(중심이 동공)를 가정한 것이다.

그러나 실제 검출 대상은 **렌즈 텍스처 에셋**이고, 모든 렌즈 텍스처는 **중심부(r<0.3)가 완전 투명한 동공 구멍**이다(알파 0%). 컬러 홍채 링은 r≈0.40~1.0에 분포한다.

| 샘플 | center(r<0.3) 불투명률 | edge[0.85,1.0] 불투명률 | 불투명 픽셀 r 범위 |
|------|----------------------|------------------------|-------------------|
| oh_bagel | 0.0% | 72.9% | 0.45~1.01 |
| claset_doll-choco | 0.0% | 71.5% | 0.40~1.01 |
| romu_dear-mellow | 0.0% | 29.3% | 0.41~0.99 |
| envie_chameau-brown | 0.0% | 57.9% | 0.53~1.00 |
| envie_parfum-glow | 0.0% | 28.7% | 0.40~0.97 |

→ center 밴드가 비어 `measureBands`가 항상 `valid=false` 반환 → 42/42 측정 불가(`ratio=-1.0`).

**대응:** 자동감지 함수에 `center_inner` 파라미터 추가(환형 center 밴드 `[center_inner, center_outer)`). 렌즈 텍스처는 `center_inner≈0.40`으로 동공 구멍을 피하고 홍채 본체를 기준으로 삼는다. 기본값 0.0f는 기존 원판 동작과 호환.

## 3. center 보정 후에도 10/10 불가 (본질적 중첩)

가능한 (center 밴드 × threshold) 전 조합 스윕 → 최고 9/10. baked와 non-baked의 edge/body ratio가 겹친다:

```
center=[0.40,0.75] 기준 정렬:
  baked ratios: 0.194  0.295  0.349  0.467  0.558   ← 최대 0.558 (oh_bagel)
  none  ratios: 0.503  0.645  0.657  0.762  0.917   ← 최소 0.503 (claset_cloud-gray)
  → baked max(0.558) > none min(0.503) → OVERLAP, 단일 임계값 분리 불가
```

"림발 없음" 렌즈도 가장자리 자연 페이드가 있어, 단순 edge/body ratio로는 약-베이크(oh_bagel)와 강한-페이드 무림발(cloud-gray)을 못 가린다. ground-truth 라벨 자체도 100×100 다운샘플 육안 분류라 경계가 모호하다.

## 4. 최적 파라미터 10 SKU 측정 (실제 C++ 함수)

`roi_inner=0.85, center_inner=0.40, center_outer=0.55, threshold=0.48`:

| SKU | ground-truth | ratio | 검출 | 판정 |
|-----|-------------|-------|------|------|
| romu_gray-taupe | YES (baked) | 0.3157 | YES | ✓ |
| romu_dear-mellow | YES | 0.4704 | YES | ✓ |
| romu_love-gleam | YES | 0.3914 | YES | ✓ |
| envie_plum-black | YES | 0.1907 | YES | ✓ |
| oh_bagel | YES | 0.6544 | **no** | **✗ MISS** |
| claset_doll-choco | no | 0.7317 | no | ✓ |
| claset_runway-gray | no | 0.6060 | no | ✓ |
| claset_cloud-gray | no | 0.5010 | no | ✓ |
| envie_parfum-glow | no | 0.5677 | no | ✓ |
| oh_kiwi | no | 1.0363 | no | ✓ |

**정확도: 9/10.**

## 5. 채택 결정 (메타 전용)

브레인스토밍 R2에서 세 모델이 "10/10 엄수"로 합의했으나, 그 근거였던 "R4 실기기 10/10"은 **실제 에셋 측정으로 재현되지 않았다**(W7 DoD의 "B4 정확도 측정"이 미체크였던 점이 방증).

§1.16/§5.10 사전 합의 분기 "10/10 미달 → 자동감지 드롭, 메타데이터 only, 누락 SKU 기본 ON 폴백"에 따라:

- **런타임:** `lens_meta.json` 메타데이터가 유일 권위. `GPULensRenderer::auto_detect_fallback_ = false` (기본).
- **메타 누락 SKU:** `has_baked_limbal=false`(셰이더 림발 ON) + WARN(§5.8/§5.9).
- **자동감지 코드:** 진단/향후 재활성용 보존(`setAutoDetectFallback(true)` + center_inner 파라미터). 9/10 수준 hint로 사용 가능.

이는 실측이 보수적 경로(인간 판단 메타)를 정당화한 사례다. "10/10 엄수"의 취지(불안정한 자동감지 출시 방지)가 실측으로 확인됐다.

## 6. 재현 방법

```bash
cd <repo-root>
g++ -std=c++17 -O2 -I cpp/include \
    docs/bench/P6-W7/bench_b4_limbal.cpp cpp/src/lens_sku_metadata.cpp \
    $(pkg-config --cflags --libs opencv4) -o /tmp/bench_b4
/tmp/bench_b4   # repo 루트에서 실행 (상대 경로 assets 참조)
```

소스: `docs/bench/P6-W7/bench_b4_limbal.cpp` (일회성 측정 하니스, 메인 빌드 미포함).
