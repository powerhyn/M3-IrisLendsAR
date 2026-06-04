# Phase 7 R1 브레인스토밍 Brief

**작성일**: 2026-06-04
**대상**: Phase 7 전체 그림 — `docs/workPaper/P7-W0_index.md` 완성
**Phase 6 종료 직후 첫 브레인스토밍**.
**참여 모델**: Claude(Opus 4.7), Codex(gpt-5.5 xhigh), Gemini(gemini-3-flash-preview)
**라운드**: R1 1회 (P6-W9 §1.8 패턴 — 새 설계 아닌 종합 검증)

---

## 0. 응답 규칙

- Q1~Q4에 모두 답변. **각 질문에 "추천안 + 근거 2~3줄"** 형식.
- Stage 1 deep-research 4개 finding을 **비판적으로 검토**. 반증 가능하면 명시.
- P7 W 분할 시 **우선순위 + 의존성** 명시.
- P8(뷰티, 턱 깎기 / 얼굴 사이즈 축소)은 인지만, **P7 범위 미포함**.
- 한국어.
- 응답 파일: `docs/workPaper/P7-W0_brainstorm/{codex|gemini|claude}_p7w0.md` 로 저장.

---

## 1. 컨텍스트 — Phase 6 완료 상태 (요약)

**살아남은 트랙 (develop 머지 완료, 머지 SHA `64ba08b`)**:
- **W1**: EyeRenderPacket 계약 + fallback (실측 source는 W6 이관)
- **W2**: 블렌드 3종 (TintLinearV2/Multiply/ScreenLinear) + realSpec 폐기 + LUMA_709 통일
- **W5 Phase A/B**: sclera veto 3-way 토글 (A/B/C/D=Normal+color/Normal+luma/CRL+color/CRL+luma) + 1차 형광 벤치 완료, luma-only 유력
- **W6 Phase A**: 블링크 ramp(C7) + 디테일 재주입(C10) + 저조도 gate(B9, 기본 0.10) + 토글 인프라
- **W7**: 셰이더 림발 영구 제거 (림발은 에셋 책임, 메모리 `limbal-in-asset-not-shader`), 메타 인프라(sku_id/registry/lens_meta.json 42 SKU) 보존
- **W9 통합**: HIGH tier 6 SKU × 5축 검증 통과 (Galaxy S23+, Render 60fps / Detection 30fps)
- **검은 화면 회귀**: CameraGLRenderer onSurfaceCreated에 `releaseGpuBeauty/Lens` 명시 추가로 차단 (commit `d1aaabe`). 하지만 **glError 0x501은 잔재** — 본 R1 Q1 대상.

**Phase 6 이월 트랙**:
- W3 환경 반사 scaffold (OFF 기본 보존, 메모리 `w4-env-reflection-deferred`)
- W4 B2 환경 반사 24클립 벤치 (Phase 7+ 재개 가능 형태로 인프라 보존)
- W8 Pupil material restore (W4 종속 자동 폐기)

**W9에서 발견된 후속 데이터**:
- **흰자 빛남이 렌즈 명도와 양의 상관** (SKU 1 짙은 초코 < SKU 4 형광 ≈ SKU 3 자연 < SKU 5 그래픽 < SKU 2 그레이 ≤ SKU 6 누드 애쉬 로제). 가장 강한 SKU 6에서 본질 한계 노출.
- 메모리 `w5-b1-tintlinearv2-strength` 시급성 확정.

---

## 2. Stage 1 — Deep-Research 4개 핵심 Finding

107 agents · 25 sources · 83 claims → 16 verified → 4 synthesized.

### F1. 흰자 빛남 본질 해결책 — Oklab + Multi-scale Laplacian
- **Confidence: high** (vote 2-1, 3-0).
- 핵심: TintLinearV2 같은 luminance-normalized 블렌드의 본질 한계는 **perceptually-uniform 색공간(Oklab) 블렌딩 + multi-scale Laplacian edge-aware 융합**으로 학술 검증된 우회 가능.
- 출처: Bottosson 2020 (Oklab), Wronski-Mertens 2007 (Exposure fusion), W3C CSS oklab() 표준화.
- **Open question**: 모바일 GPU에서 sRGB↔Oklab cube root 비용 30fps 예산 수용 여부 미검증.
- Refute됨: Oklab 정량 수치(RMS error 0.20 vs CIELAB 1.70) — 정성적 hue shift 회피만 통과.

### F2. 0x501 = GLSL ES 3.0 spec §8.9 실제 위반 (즉시 수정 대상)
- **Confidence: high** (vote 3-0, 3-0, 3-0, 3-0, 3-0, 2-1, 3-0 — 7 claims 통과).
- 핵심: GLSL ES 3.0 spec §8.9 명문화 — **dynamic branch 내부 `texture()` implicit derivative는 undefined**.
- W6 코드 `shader_sources.cpp:1066-1083` 내 `if(uDetailReinject==1){ 9 sample texture() }` 패턴은 **명백한 spec 위반**.
- 출처: Khronos refpage, Mozilla Bugzilla #1932416 (2025-01 Firefox 134 수정, Adreno 305/306 검은 화면), Maister 블로그, Godot #12816, Filament #1544, Unity, NVIDIA dev forum.
- 표준 회피: **(a) `textureLod(uv, 0.0)` 명시 LOD**, (b) fetch를 분기 밖으로 hoist, (c) textureGrad with 사전 계산 gradient (느림, 임시).
- **Caveat**: 실증 사례는 Adreno 3xx(2012-2014). IrisLensSDK 타겟 Adreno 6xx/7xx 직접 일반화는 보수적 — 단 spec 위반 자체는 모든 Adreno 세대 적용.

### F3. EXTERNAL_OES → 2D FBO 변환 강제
- **Confidence: high** (vote 3-0, 3-0).
- 핵심: GL_TEXTURE_EXTERNAL_OES는 **mipmap 생성/LINEAR_MIPMAP 필터/CLAMP_TO_EDGE 외 wrap 모드 모두 미지원** → 카메라 텍스처에서 ROI luminance 통계 직접 측정 불가.
- 표준 경로: **OES → 2D FBO 변환 → 다운샘플/mipmap 통계**.
- 출처: Android 공식 (source.android.com), Khronos OES_EGL_image_external extension spec.
- **W6 avg_iris_luma 실측 source 연결(현재 fallback 0.1225)은 단순 wiring이 아니라 변환 패스 신규 설계 필요**.
- Refute됨: PBO 비동기 readback 3개 claim (출처 단일 블로그 — 검증 실패, PBO 자체가 무효라는 뜻 아님. P7 사용 시 Khronos 별도 검증 권장).

### F4. Phase 8 face slimming substrate 확정 (P7 범위 밖, 인지만)
- **Confidence: high** (vote 3-0, 2-1×2, 3-0, 3-0).
- 핵심: **MediaPipe FaceMesh 468 vertex = 표준 anchor** (Snap/Perfect Corp/Banuba 공통).
- **Snap Lens Studio Face Liquify** = per-point **Radius + Intensity** (<1 계수 = inward warp "black hole effect") — 모바일 실시간 face slimming 표준 메커니즘.
- **Banuba는 iris/sclera/pupil 분리 API 노출** ("Full eye, sclera, and iris recolor" 별도 모드) → IrisLensSDK W5 sclera veto 방향성이 **업계 표준 아키텍처** 임을 확인.
- P8 W 분할 토대: (a) FaceMesh 468 vertex anchor, (b) per-point Radius/Intensity 모델, (c) inward warp shader.
- **Caveat**: Banuba는 vendor 마케팅 — "분리 API 노출"은 사실이지만 "흰자 over-brightening 해결" 입증 X (API surface precedent로만 활용).

### Refuted (참고)
- Oklab 정량 수치 RMS — 정성적 hue 안정 수준으로만 주장 가능
- if→branchless mix() 평탄화가 Adreno 3xx 검은 화면 해결 — 출처 부족
- Face Liquify가 spherical warping을 주 기법으로 사용 — 출처 부족
- PBO 비동기 DMA readback (3 claims) — 단일 블로그 출처 검증 실패
- Qualcomm Adreno "Garbage LOD" — 출처 부족

---

## 3. A/B 그룹 (P7 후보 항목 원안)

### A. 즉시 cleanup PR
- W9 데모 UI/KT 동기화 (블렌드 drop-down 3종 축소, 3D Light 버튼 제거, 기본 blendMode TintLinearV2 ID=5)
- deprecated no-op 제거 (`setLensHighlight`, `setHighlightEnabled` 등)
- SKU "누드 애쉬 로제" 톤 분류 정정 (웜톤 X → 애쉬+누드)
- MID/LOW tier 회귀 검증

### B. Phase 6 미완 후속
- **W5 Phase C** — TintLinearV2 흰자 빛남 수식 개선 (실측 데이터 시급성 확정 + F1 알고리즘 후보 확보)
- **W6 Phase B/C** — avg_iris_luma 실측 source 연결 + 블링크 ramp/저조도 gate 최종 튜닝 (F3로 변환 패스 강제 확정)
- LUMA 계수 shader vs CPU 오차 ≤1% 테스트
- **0x501 잔재 추적** (F2로 즉시 수정 대상 확정)

---

## 4. R1 4개 핵심 질문 (Q1~Q4)

### Q1. 0x501 즉시 수정 vs 디바이스 매트릭스 회귀 우선

F2 confidence는 high이지만 Adreno 3xx 사례 일반화 caveat. P7 첫 W 구조 후보:
- **(a)** textureLod 즉시 패치 + 회귀 검증 묶기 (한 W 안에)
- **(b)** Adreno 6xx/7xx 회귀 검증 우선 → 재현되면 패치
- **(c)** 둘 다 같은 W, 패치 먼저 + 회귀로 cross-tier 확인
- **(d)** 패치는 cleanup PR로, 회귀 검증은 별도 W

**근거 + 추천 + 위험 명시**.

### Q2. W5 Phase C 알고리즘 선택

F1의 Oklab + Laplacian edge-aware는 학술 검증되었지만 GPU 예산 미확보.
- **(a)** 단계적 접근: 기존 TintLinearV2 + sclera mask 강화 (가벼운 패치)부터 → 효과 부족 시 Oklab/Laplacian
- **(b)** 한 번에 Oklab 블렌드 PoC + 30fps 벤치 확인 후 채택/롤백
- **(c)** Multi-scale Laplacian만 (sclera 영역 attenuation) + Oklab 별도 trade-off 평가

**근거 + 추천 + Phase 6 W5 Phase A/B 결정(luma-only)과의 조합 영향**.

### Q3. W6 avg_iris_luma 측정 패스 설계

F3로 OES → 2D 변환 강제 확정. 후보:
- **(a)** 별도 small ROI FBO 다운샘플 (예: 64×64 → mipmap mean)
- **(b)** 이미 존재하는 2D 변환 패스 재사용 + ROI 영역 통계
- **(c)** compute shader 기반 atomic accumulation
- **(d)** 매 N frame만 측정 (30fps에서 N=5 등) + EMA 평활

30fps 예산 + Adreno mipmap 안정성 + GLES 3.1 compute 호환성 고려.

### Q4. P7 W 분할 우선순위 + 의존성

A/B 항목 + Stage 1 finding을 W로 묶는 제안. 예시 분할:
- P7-W1: 0x501 수정 + Adreno 6xx/7xx 회귀 검증
- P7-W2: avg_iris_luma 측정 패스 설계 + W6 Phase B 토글 검증
- P7-W3: W5 Phase C 흰자 빛남 (sclera mask vs Oklab)
- P7-W4: A 그룹 cleanup 일괄 (UI 동기화 + deprecated 제거 + SKU 정정)
- P7-W5: MID/LOW tier 회귀 (전체 통합)

**또는 다른 분할 제안**. 의존성 그래프 + 병렬 가능 포인트 + 소요 추정.

---

## 5. 제약 조건

- **P8은 뷰티 기능 (턱 깎기, 얼굴 사이즈 축소) 예약** — P7 범위에 face slimming 미포함. F4는 인지용.
- **공개 sdk_api.h v1.0.0 동결** — 내부 API 추가는 가능, 공개 surface 변경 금지.
- 메모리 **`solo-dev-bench-method` / `qualitative-device-judgment`** — 평가는 사용자 1인 실기기 토글 체감 우선.
- 사용자 보유 기기: **HIGH tier(Galaxy S23+) 확정**. MID/LOW는 별도 확보 필요 (P7-W5 제약).
- 메모리 `feedback_multi_ai_orchestration_bias` — Claude 단독 결정 금지, 다수 모델 응답 필수.

---

## 6. 산출물

- 각 모델 응답 파일: `{codex|gemini|claude}_p7w0.md`
- 종합: `synthesis.md`
- 최종 인덱스: `docs/workPaper/P7-W0_index.md` (Claude가 종합 후 작성)

종합 시 다음 명시:
- 3/3 합의 / 2/1 다수결 / 3분립 분류
- Hard veto 있음/없음
- R2 필요 여부 (대부분 1라운드면 충분 예상)
- 사용자 최종 판단 필요 항목
