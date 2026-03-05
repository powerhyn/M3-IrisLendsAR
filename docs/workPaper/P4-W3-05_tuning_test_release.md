# P4-W3-05: 튜닝 + 테스트 + 릴리즈 게이트 판정

| 항목 | 내용 |
|------|------|
| **작업 ID** | P4-W3-05 |
| **유형** | 검증/릴리즈 |
| **상태** | 🔄 진행 중 |
| **근거 문서** | P4-W3-01 (브레인스토밍), P4-W3-02~04 (구현) |
| **선행 조건** | P4-W3-04 완료 (Temporal + Tier 동작) |
| **작성일** | 2026-03-03 |
| **일정** | Day 9~15 (5일 + 2일 리스크 버퍼) |

---

## 1. 목표

Freq Sep 파이프라인의 파라미터를 다양한 조건에서 튜닝하고, 정량/정성 품질 게이트를 통과시켜 릴리즈 판정을 완료한다.

### 1.1 완료 조건

- [ ] attenuation 커브 최적화 (다양한 피부톤/조명)
- [ ] 실기기 테스트 (Android 3 + iOS 2)
- [ ] 피부톤 3그룹별 균등 품질 확인
- [ ] A/B 비교 (Bilateral vs Freq Sep) 정량 측정
- [ ] 블라인드 주관평가 통과
- [ ] 릴리즈 게이트 전 항목 통과
- [ ] 프리셋 최종값 확정
- [ ] 문서화 + 코드 리뷰

### 1.2 실패 기준 (No-Go → 리스크 버퍼 Day 14~15 활용)

- 정량 게이트 미달 (Laplacian, SSIM, temporal)
- 블라인드 평가 선호도 < 70%
- "흐리다/가짜" 피드백 > 10%
- tier별 성능 게이트 미달

---

## 2. 튜닝 작업

### 2.1 attenuation 커브 튜닝

**튜닝 변수**:

| 변수 | 현재 기본값 | 튜닝 범위 | 영향 |
|------|-----------|----------|------|
| `uAttenuationLow` | 0.02 | 0.01~0.05 | 작을수록 미세 텍스처까지 감쇠 |
| `uAttenuationHigh` | 0.15 | 0.08~0.25 | 클수록 잡티 판정 느슨 |
| `sigma 비율` | 0.4 | 0.3~0.5 | Gaussian 폭 조절 |

**테스트 조건 매트릭스**:

| 조건 | 변수 |
|------|------|
| 피부톤 | 밝음 / 중간 / 어두움 |
| 조명 | 자연광 / 실내 형광등 / 역광 |
| 거리 | 근접(20cm) / 표준(40cm) / 원거리(80cm) |
| 피부 상태 | 매끄러운 / 모공 큰 / 잡티 있는 |

### 2.2 매핑 테이블 최종 조정

`mapSkinQuality()` 결과값을 실기기 테스트 후 보정:

```
조정 예시:
- skinQuality=0.6에서 "약간 부족" → highFreqPreserve 0.50 → 0.45로 하향
- skinQuality=1.0에서 "인형 느낌" → highFreqPreserve 0.10 → 0.15로 상향
```

### 2.2.1 blur_radius 독립성 검증

P4-W3-02 §4.6의 `mapSkinQuality()`는 **B방식(고정 ratio)을 기본 구현**으로 채택했다.
cutoff frequency를 skinQuality와 독립시켜 슬라이더 조작 시 예측 가능한 단일 축 변화를 보장한다.

A방식(연동형)은 품질 문제 발견 시 대안으로 비교 검증한다.

| 방식 | blur_radius | skinQuality 역할 | 상태 |
|------|-------------|-----------------|------|
| **B (기본, 현재 구현)** | face_width × 0.05, clamp(6,28) | attenuation + preserve만 제어 | ✅ P4-W3-02 반영 |
| **A (대안)** | face_width × (0.03 + s×0.04), clamp(6,28) | radius + attenuation + preserve 연동 | 품질 gap 시 전환 |

**B→A 전환 판정 기준**: B방식에서 아래 품질 문제가 발견될 경우에만 A로 전환 검토.
- skinQuality 0.3에서 큰 잡티(반경 > radius)가 제거되지 않음
- skinQuality 1.0에서 과도한 피부결 손실 (fixed radius가 너무 큼)
- 블라인드 주관평가에서 B가 A 대비 유의미하게 낮은 선호도

"사용자가 슬라이더를 올릴 때 더 예측 가능한 변화"를 보이는 쪽을 채택한다.

### 2.3 프리셋 값 최종 확정

| 프리셋 | 초기값 | 최종값 (Day 9 이후 확정) |
|--------|--------|------------------------|
| NATURAL | 0.3 | TBD |
| MODERATE | 0.5 | TBD |
| STRONG | 0.8 | TBD |

---

## 3. 실기기 테스트 매트릭스

### 3.1 테스트 기기

| 기기 | OS | GPU | 예상 Tier |
|------|-----|-----|----------|
| Galaxy S24 | Android 14 | Adreno 750 | HIGH |
| Galaxy A54 | Android 13 | Adreno 642L | MID |
| Redmi Note 12 | Android 13 | Adreno 619 | LOW |
| iPhone 15 Pro | iOS 17 | A17 Pro | HIGH |
| iPhone 13 | iOS 17 | A15 | MID |

### 3.2 측정 항목

| 항목 | Android 도구 | iOS 도구 | 기준 |
|------|-------------|----------|------|
| Freq Sep 전체 ms | GPUProfiler (SDK 내장) | Xcode Instruments / Metal System Trace | HIGH ≤8ms, MID ≤12ms, LOW ≤6ms |
| 전체 프레임 ms | FPS counter | FPS counter | ≤33ms (30fps) |
| 텍스처 메모리 | TexturePool stats | TexturePool stats | 추가 ≤3장 |
| 발열 | 5분 연속 사용 | 5분 연속 사용 | 체감 온도 정상 범위 |
| 배터리 | 10분 사용 | 10분 사용 | 기존 대비 +10% 이내 |

> **플랫폼별 계측 책임 분리**: iOS 성능 게이트는 SDK 내 GPUProfiler 결과를
> 사용하지 않으며, 외부 프로파일러(Xcode Instruments / Metal System Trace)
> 계측값을 릴리즈 근거로 사용한다. GPUProfiler는 Android(EGL) 전용이며,
> iOS 빌드에서는 스텁으로 동작한다.

---

## 4. 품질 측정

### 4.1 정량 측정

| 지표 | 측정 방법 | 기준 |
|------|----------|------|
| **Laplacian variance 감소율** | 피부 영역 ROI 크롭 → Laplacian → variance 계산 | 30~60% 감소 |
| **비피부 에지 SSIM** | 비피부 영역 원본 vs 처리 결과 | > 0.95 |
| **Temporal variance** | 연속 30프레임 동일 포즈 → 스무딩 강도 표준편차 | < 5% |
| **Halo 검출** | 피부/비피부 경계에서 gradient 분석 | 경계 gradient 증가 < 15% |

### 4.2 정성 평가 (블라인드 A/B)

**프로토콜**:
1. 평가자 5명 (내부 팀)
2. 동일 인물/조건에서 Bilateral vs Freq Sep 영상 쌍 제시
3. 라벨 없이 "어느 쪽이 더 자연스러운가?" 선택
4. 각 쌍에 대해 추가 질문: "흐리다고 느껴지는가?", "가짜 같은가?"

**기준**:
- Freq Sep 선호도 > 70%
- "흐리다" 응답 < 10%
- "가짜 같다" 응답 < 10%

### 4.3 피부톤별 균등 품질

| 그룹 | Laplacian 감소율 | 주관 선호도 |
|------|-----------------|------------|
| 밝은 피부톤 | 30~60% | > 70% |
| 중간 피부톤 | 30~60% | > 70% |
| 어두운 피부톤 | 30~60% | > 70% |

어두운 피부톤에서 attenuation 보정이 필요할 수 있음 (§2.1 튜닝에서 처리).

---

## 5. 릴리즈 게이트 체크리스트

### 5.1 하드스톱 게이트 (즉시 NO-GO, 완화 불가)

아래 항목은 1개라도 미달 시 **즉시 NO-GO**이며, CONDITIONAL GO로 완화할 수 없다.

- [ ] **Crash 없음**: 전 기기/전 tier에서 Freq Sep 경로 crash-free
- [ ] **메모리 누수 없음**: 10분 연속 사용 시 텍스처/버퍼 누수 0건
- [ ] **FPS: 30fps 유지**: 전체 파이프라인 ≤33ms
- [ ] **Temporal variance**: 프레임 간 스무딩 강도 변동 < 5%

### 5.2 정량 게이트 (자동, 완화 불가)

- [ ] Laplacian variance 감소율: 피부 영역 30~60%
- [ ] 비피부 에지 보존율: SSIM > 0.95
- [ ] 성능: tier별 기준표 준수 (HIGH ≤8ms, MID ≤12ms, LOW ≤6ms)
- [ ] 메모리: TexturePool 추가 ≤3장

### 5.3 정성 게이트 (수동, CONDITIONAL GO 허용)

- [ ] 블라인드 A/B 선호도 > 70%
- [ ] "흐리다/가짜 같다" 피드백 < 10%
- [ ] 피부톤 3그룹 균등 품질 확인
- [ ] 코 옆/입 주변 halo 없음 (육안)
- [ ] 눈썹/윤곽 흐려짐 없음 (육안)

### 5.4 판정 기준

| 결과 | 조건 | 조치 |
|------|------|------|
| **GO** | 하드스톱 + 정량 + 정성 전항목 통과 | 릴리즈 진행 |
| **CONDITIONAL GO** | 하드스톱 + 정량 전항목 통과, **정성** 1개 미달 | Day 14~15 버퍼로 보완 후 재평가 |
| **NO-GO** | 하드스톱 1개+ 미달 **또는** 정량 1개+ 미달 | 아키텍처 재검토 (Bilateral + alpha 대안) |

> **정량 게이트 완화 불가 원칙**: §5.2의 정량 게이트는 자동 측정 항목이므로
> pass/fail 판정이 명확하다. 정량 1개라도 미달이면 즉시 NO-GO이며,
> CONDITIONAL GO는 정성 게이트(§5.3)에만 적용된다.

> **하드스톱 원칙**: Crash, 메모리 누수, 30fps 미달, Temporal variance 초과는
> 사용자 경험에 직접적 영향을 주므로 절대 완화할 수 없다.
> CONDITIONAL GO는 블라인드 선호도 등 정성 지표에만 적용된다.

---

## 6. 실행 일정

| Day | 작업 | 산출물 | 완료 기준 |
|-----|------|--------|----------|
| **9** | attenuation 커브 + 매핑 테이블 품질 튜닝 | 최적 파라미터 확정 | 다양한 피부톤/조명에서 자연스러운 결과 |
| **10** | A/B 비교 (Bilateral vs Freq Sep) + 정량 측정 | Laplacian, SSIM 결과 | 정량 게이트 통과 |
| **11-12** | 실기기 테스트 (Android 3 + iOS 2) | 기기별 성능/품질 리포트 | tier별 성능 게이트 통과 |
| **13** | 피부톤 그룹별 튜닝 (밝음/중간/어두움) + 프리셋 확정 | 그룹별 attenuation 보정값 | 균등 품질, 프리셋 최종값 |

### Week 3 리스크 버퍼

| Day | 작업 | 조건 |
|-----|------|------|
| **14** | 블라인드 주관평가 + 릴리즈 게이트 판정 | 평가 결과: 선호도 > 70%, "흐리다" < 10% |
| **15** | 문서화 + 코드 리뷰 + 최종 확인 | 리뷰 통과, 문서 갱신 완료 |

---

## 7. 리스크

| 리스크 | 확률 | 대응 |
|--------|------|------|
| 비선형 attenuation 튜닝 수렴 안 됨 | 중 | 선형 감쇠로 일단 출시 → 이후 고도화 |
| 어두운 피부톤에서 과도한 스무딩 | 중 | Y-based 임계값 피부톤별 보정 테이블 |
| 블라인드 평가 선호도 미달 | 중 | 리스크 버퍼 활용, attenuation 재튜닝 |
| 특정 기기에서 성능 초과 | 중 | 해당 기기 tier 하향 조정 |
| Week 2까지 품질 미달 | 중 | Week 3 리스크 버퍼 전체 활용 |
| Gaussian Blur 경계 색 번짐 (halo) | 중 | Low Freq 경계에서 비피부 색이 섞임 → composite 마스크 블렌딩이 1차 완화. 심각하면 mask-aware blur(bilateral on lowFreq)로 업그레이드 |
| 고ISO/저조도에서 attenuation 오판 | 중 | 노이즈가 고주파에 포함 → 피부결까지 과도 감쇠. §2.1 테스트 매트릭스(역광/실내)에서 확인 후, 필요시 adaptive threshold 도입 (post-MVP) |
| ~~highFreq 텍스처 최적화 (6→5 패스)~~ | — | ✅ **해결됨**: P4-W3-02에서 Extract 패스를 Composite에 인라인화 완료. 5서브패스가 MVP 기본 설계. |
| MID tier half-res 블러 경계 아티팩트 | 중 | half-res lowFreq의 GL_LINEAR 업샘플이 피부-비피부 경계에서 halo 유발 시, 3/4 스케일 상향 또는 mask-aware blur 검토 |

---

## 변경 이력

| 날짜 | 변경 내용 | 작성자 |
|------|----------|--------|
| 2026-03-03 | 통합 문서에서 튜닝/릴리즈 분리 | Claude |
| 2026-03-03 | 리뷰 3차 반영: §3.2 플랫폼별 계측 도구 분리 (Android: GPUProfiler, iOS: Xcode Instruments), §5.1 하드스톱 게이트 신설 (Crash/누수/FPS/Temporal → 즉시 NO-GO), §5.3 정성 게이트만 CONDITIONAL GO 허용, §5.4 판정 기준 재정리 | Claude |
| 2026-03-04 | 리뷰 4차 반영: §5.4 정량 1개+ 미달 → NO-GO로 수정 (판정 gap 해소), 정량 완화 불가 원칙 명시 | Claude |
| 2026-03-04 | Gemini 리뷰 반영: §2.2.1 blur_radius 독립성 A/B 비교 항목 추가, §7 리스크에 Gaussian halo/고ISO attenuation/6→5패스 최적화/MID half-res 품질 추가 | Claude |
| 2026-03-04 | Gemini 3차 리뷰 반영: §2.2.1 A/B 기본값 전환 — B(고정 ratio)를 기본 구현으로 채택, A(연동형)는 품질 gap 시 대안. §7 Extract 패스 리스크 해결 처리, MID tier 리스크 설명 보정 | Claude |
| 2026-03-05 | **구현 착수**: 튜닝/테스트/릴리즈 인프라 코드 구현 — QualityMetrics (Laplacian/SSIM/Halo/Temporal), ABCompare (Bilateral vs FreqSep A/B 비교), ReleaseGate (3-tier 판정 자동화), ParamTuner (attenuation 그리드 서치 + 프리셋 추천 + B방식 blur_radius 독립성 검증). 단위 테스트 32건 전 PASS | Claude |

---

## 실행 내역

### Day 9 (2026-03-05) — 튜닝/테스트 인프라 구현

#### 산출물

| 파일 | 유형 | 설명 |
|------|------|------|
| `cpp/include/iris_sdk/quality_metrics.h` | 헤더 | Laplacian variance, SSIM, Halo 검출, TemporalAnalyzer |
| `cpp/src/quality_metrics.cpp` | 구현 | Wang et al. SSIM, Sobel 기반 Halo, 종합 Gate 판정 |
| `cpp/include/iris_sdk/ab_compare.h` | 헤더 | A/B 비교 프레임워크 (피부톤별 요약) |
| `cpp/src/ab_compare.cpp` | 구현 | FreqSep vs Bilateral 비교 + JSON 리포트 |
| `cpp/include/iris_sdk/release_gate.h` | 헤더 | 3-tier 릴리즈 게이트 (HardStop/Quantitative/Qualitative) |
| `cpp/src/release_gate.cpp` | 구현 | GO/CONDITIONAL_GO/NO_GO 판정 로직 |
| `cpp/include/iris_sdk/param_tuner.h` | 헤더 | 그리드 서치 + 프리셋 추천 + B방식 검증 |
| `cpp/src/param_tuner.cpp` | 구현 | 가중 점수 계산, blur_radius 독립성 검증 |
| `cpp/tests/test_quality_tuning.cpp` | 테스트 | 5그룹 32건 단위 테스트 |

#### 검증 결과

| 항목 | 결과 |
|------|------|
| 컴파일 | ✅ 성공 (libiris_sdk에 정상 포함) |
| 단위 테스트 | ✅ 32/32 PASS (144ms) |
| 기존 테스트 호환성 | ✅ 기존 테스트에 영향 없음 |
