# P5-W3-05 렌즈 렌더링 자연스러움 — 최종 결정

> **작성**: Claude Opus 4.7 (모더레이터)
> **시점**: 2026-04-22 R2 종료 직후
> **Base**: Codex(gpt-5.4 xhigh), Gemini(3-flash-preview), Claude 3라운드(R1 독립 + R2 교차 비판) 결과 종합

---

## 0. 한 줄 결론

**블렌드 모드 4종으로 압축 + LTL realSpec 해킹 제거 + 환경 반사를 별도 가산 계층으로 분리 + EyeRenderPacket 도입**. W3-04 고정 조명은 폐기. 3개 남은 쟁점(I1·I4·I8)은 실기기 벤치 6시간으로 확정.

---

## 1. 즉시 확정 (실기기 검증 없이 채택)

R2에서 세 모델이 수렴했거나, 기술적 근거가 충분해 벤치 불필요한 항목.

### 1.1 제거할 것 (W3-04 롤백)

| ID | 제거 대상 | 파일:라인 | 근거 |
|----|----------|----------|------|
| D1 | 분석적 노멀 + 고정 조명 `vec3(0.3, 0.4, 1.0)` 블록 | `shader_sources.cpp:1008-1032` | 세 모델 공통 비판. eye-local decal 조명이라 각막 반사처럼 안 움직임 |
| D2 | `LIMBAL_ENABLED = false` 하드코드 전역 비활성 | `shader_sources.cpp:999-1006` | 제품 책임을 asset authoring에 떠넘김 |
| D3 | `realSpec = smoothstep(0.7, 0.95, lum)` + `result = mix(result, baseL, realSpec)` | `shader_sources.cpp:865-874` (`blendLuminanceTintLinear` 내부) | Codex: "spec 보호가 아니라 밝은 픽셀 보호, 개념 오류". 반사 분리 후 불필요 |
| D4 | `uAvgIrisLum = 0.35` 하드코드 기본값 | `gpu_lens_renderer.cpp:811-813` 근처 | Codex: "priors 덮어씌움". 개인화 파괴 |
| D5 | `uHighlightEnabled` 토글 UI ("3D Light" 버튼) | demo app 및 shader uniform | 환경 반사 계층이 대체 |
| D6 | 기존 블렌드 모드 중 `Normal`(0), `Overlay`(3), `LuminanceTint`(4, non-linear), `SoftLight`(6) 4종 | `shader_sources.cpp:979~997` 분기 | 4종 세트에서 제거 |

### 1.2 추가/변경할 것

| ID | 변경 | 구현 위치 | 근거 (합의 출처) |
|----|------|----------|-----------------|
| C1 | **블렌드 모드 4종 세트로 재편**: `TintLinearV2`(0, 기본값), `Multiply`(1), `ScreenLinear`(2), `ColorReplaceLinear`(3) | `shader_sources.cpp` + ID 재번호화 | Codex R1/R2, Claude R2 수용, Gemini R2 수용 |
| C2 | `TintLinearV2` 수식: `baseL=base*base; lensL=lens*lens; lum=dot(baseL,w); out=sqrt(mix(baseL, lensL*lum*scale, a));` — realSpec 제거 | `blendLuminanceTintLinear` 재작성 | 세 모델 합의 (Codex 수식 채택) |
| C3 | `ScreenLinear` 수식: `out = sqrt(mix(baseL, 1-(1-baseL)*(1-lensL), a))` 신규 추가 | shader 신규 블렌드 함수 | Codex R1 제안, Claude/Gemini R2 수용 |
| C4 | `ColorReplaceLinear` 수식: `detail = clamp(pow(lum/avgLum, 0.7), 0.75, 1.25); out = sqrt(mix(baseL, lensL*detail, a))` 신규 추가 | shader 신규 | Codex R1 제안, Claude/Gemini R2 수용 (* 단 Normal과 실기기 대결 후 채택 확정 — §2.B1 벤치) |
| C5 | **환경 반사 가산 계층 분리**: pigment blend 이후 `blended += reflection * fresnel * finalAlpha`로 합성. LTL 내부 realSpec 완전 제거 | `applyLens` 구조 재편 | Codex R1/R2 주장, Claude R2 전면 수용, Gemini R2 수용 |
| C6 | **림발 링 기본 ON**: `uLimbalEnabled` uniform 기본 true. SKU 메타데이터 `has_baked_limbal: true` 시 강도 0 | shader + `LensConfig`에 `has_baked_limbal` bool 추가 | Codex R1/R2, Gemini R1/R2, Claude R2 수용 (자동 감지는 §2.B4에서 판정) |
| C7 | **블링크 alpha ramp**: 눈 감김 감지 시 down 60~80ms (α_close=0.15), 재열림 up 100~120ms (α_open=0.08). `eyelidMask`로 필터 통과시킨 후 적용 | `render_alpha` EMA 2계수 추가 | Claude 타이밍 + Codex 비대칭 ease 수렴 |
| C8 | **EyeRenderPacket 구조체 도입** — 렌더러 ↔ 검출기 사이 계약 | 내부 어댑터 레이어 (C API 공개 변경 없음) | 세 모델 합의 (Codex 구조 + Claude/Gemini 필드 조합) |
| C9 | **avg_iris_luma 실측** — iris ROI 중심 반경 0.4 이내의 픽셀 평균 휘도. `textureLod(camera, iris_center, 3.0)` 1회 샘플. 측정 실패 시 이전값 hold, 3프레임 이상 실패 시 중립 상수 | `gpu_lens_renderer.cpp` 전처리 | Codex R1/R2, Claude/Gemini R2 수용 |
| C10 | **홍채 디테일 재주입** (Codex 수식): blend 이후 `detail = clamp(baseLum / blur3x3(baseLum), 0.85, 1.15)`를 iris inner(r<0.65)에만 곱함. 저조도 gate: `avg_iris_luma < 0.15`면 강도 감쇄 | `applyLens` 마지막 단계 | Codex 원본 재주입 + gate (Claude/Gemini R2 수용) |
| C11 | **W2와 경계 명확화**: pupil_center·occlusion·(optional) gaze는 W2(eye refiner)가 출력, W3는 EyeRenderPacket으로 수신만 | 워크페이퍼 정리, 실제 구현은 W2 | Codex R2 철회 수용 |

### 1.3 EyeRenderPacket 최종 스키마

```cpp
// 내부 구조체. 공개 C API는 변경 없음.
struct EyeRenderPacket {
    // 필수
    glm::vec2 iris_center_norm;       // 정규화 좌표
    float     iris_radius_norm;
    // 비대칭 타원 (없으면 top/bottom fallback)
    glm::vec2 ellipse_center;
    glm::vec3 ellipse_radii;          // (rxInner, rxOuter, ry)
    float     ellipse_rotation;
    float     eye_top, eye_bottom;    // Y-slab fallback
    float     visibility;             // 0.0~1.0
    uint64_t  timestamp_ms;

    // 선택 (없으면 기능 자동 off)
    std::optional<glm::vec2> pupil_center_norm;       // parallax용
    std::optional<glm::vec2> head_pose_yaw_roll;      // env rotation용
    std::optional<glm::vec3> reflection_dir;          // head_pose 대체 가능
    std::optional<float>     avg_iris_luma;           // blend 정규화용
    std::optional<float>     eye_depth_mm;            // 스케일 보정용
    std::optional<float>     render_confidence;       // alpha hysteresis용
};
```

- **금지**: raw `face_mesh` 의존, detector-specific landmark 인덱스.
- **Optional 처리 원칙**: 없으면 **기능을 끈다**. priors로 채우지 않는다 (C9의 avg_iris_luma는 예외 — self-measure 경로 있음).

---

## 2. 실기기 벤치로 확정 (남은 3개 쟁점)

말로 끝내지 못하는 쟁점. 각각 6~8시간 내 확정 가능.

### B1 — 블렌드 4번째 슬롯: Normal vs ColorReplaceLinear (I2 잔여)

**쟁점**: Codex는 "Normal은 flat, ColorReplaceLinear가 불투명 렌즈에서 더 자연"이라 주장. Claude R1은 Normal 유지 주장했으나 R2에서 철회. 단 Codex 본인도 R2에서 "4종 고정 수정, ColorReplaceLinear는 Normal과 1:1 대결 후 채택"으로 신중하게 변경.

**벤치 계획**:
- **매트릭스**: 4 SKU(다크브라운 자연 / 헤이즐 자연 / 밝은 그레이·블루 / 불투명 서클) × 3 홍채 톤(짙음·중간·밝음) × 2 모드(Normal / ColorReplaceLinear) = **24 클립**
- **판정**: 3명 평가자 블라인드 A/B. "자연스러움" 선호 투표. 과반수 기준.
- **결론 시나리오**:
  - ColorReplaceLinear가 14/24 이상 승리 → **4종 확정** (TintLinearV2, Multiply, ScreenLinear, ColorReplaceLinear)
  - 동률 (12~13 승리) → **ColorReplaceLinear 채택** (flat 리스크 고려)
  - Normal이 14/24 이상 승리 → **3종** (TintLinearV2, Multiply, ScreenLinear) + Normal
- **소요**: 3시간 (캡처 1h + 투표 1h + 반영 1h)

### B2 — 환경 반사 소스: env map vs Periphery 카메라 샘플링 vs OFF (I1)

**쟁점**: Codex = 정적 env map 에셋, Gemini R2 = 카메라 가장자리 8포인트 Periphery 샘플링, Claude R2 = env map 디폴트 + 상단 crop hybrid.

**벤치 계획**:
- **프로토타입 3종**: 
  1. **OFF** (환경 반사 없음, 베이스라인)
  2. **env-map-only**: 256×128 LDR/RGBM 에셋 (mid-tier 환경 가정). `reflection_dir` 없으면 정적.
  3. **periphery-camera**: 화면 상단 1/3 + 좌우 가장자리 8포인트 평균 휘도/색상을 `textureLod(camera, edge_uv, 3.0)`로 추출. 얼굴 영역 재귀 방지는 `finalAlpha × (1 - face_region_mask)` 마스킹.
- **매트릭스**: 4 환경(실내형광 / 창가측광 / 야간실내 / 실외낮) × 2 동작(정면 미세움직임 / 좌우 head turn) × 3 프로토타입 = **24 클립**
- **판정 메트릭** (실패 카운트, 낮을수록 좋음):
  - `가운데 붙은 반짝이` (eye-local decal 같은 부자연)
  - `얼굴 재귀 반사처럼 보임` (자기 얼굴이 눈에 비치는 느낌)
  - `좌우 눈 불일치` (양쪽 반사 방향이 어긋남)
  - `환경과 무관해 보임` (실제 조명과 반사 위치 어긋남)
  - `자연스러움 블라인드 선호도` (양성 메트릭, 높을수록 좋음)
- **결론 시나리오**:
  - env-map-only가 명확 우세 → **Codex 안 채택**
  - periphery-camera가 명확 우세 → **Gemini 안 채택**
  - 둘이 환경별로 갈림 → **hybrid 채택** (`periphery_camera_confidence`가 낮으면 env-map fallback)
  - 셋 다 비슷하거나 OFF 대비 유의미 차이 없음 → **Phase 6로 이관** (W3-05는 기본 OFF)
- **소요**: 4~5시간 (프로토타입 3개 구현 2h + 캡처 1.5h + 블라인드 평가 1h + 반영 0.5h)

### B4 — 림발 자동 감지 vs 메타데이터 only (I4 잔여)

**쟁점**: 메타데이터 없는 구형 SKU 대응을 위해 자동 감지(텍스처 edge region 밝기 분석)를 켤지. Codex는 false positive 위험 경고, Gemini/Claude는 "edge vs center 대비 제한하면 안전" 주장.

**벤치 계획**:
- **프로토타입**: 텍스처 로딩 시 CPU에서 1회 측정 — `edge_lum = avg(pixels in r∈[0.85, 1.0])`, `center_lum = avg(pixels in r<0.3)`, `edge/center < 0.75` 시 `baked_limbal_detected = true`
- **테스트 샘플**: 림발 내장 SKU 5개 + 림발 없는 SKU 5개 = **10개 고정 텍스처**
- **판정**:
  - 자동 감지의 **정확도 (accuracy)**: 올바른 분류 9/10 이상 → 자동 감지 채택
  - 8~9/10 → 자동 감지는 fallback only (메타 있으면 메타 우선, 없을 때만 자동)
  - 7/10 이하 → Codex 안 채택 (메타 only, 자동 감지는 드롭)
- **소요**: 1시간 (프로토타입 30min + 10개 SKU 평가 30min)

### B8 — sclera color veto: 채도 포함 vs luma-veto only (I8 잔여)

**쟁점**: Codex는 `(saturation + luma) veto`, Gemini는 `luma-veto only`(채도 드롭), Claude R2는 Codex 수용.

**벤치 계획**:
- **매트릭스**: 3 SKU 중 그레이/블루 계열 1개 + 다크브라운 1개 × 3 조명(형광·측광·저조도) × 2 방식(color-veto / luma-veto-only) = **12 클립**
- **판정**: 
  - "흰자에 렌즈 번짐"이 더 적은 방식 선택 (음성 메트릭)
  - "iris 외곽 어두운 무늬 잘림"이 발생하면 해당 방식 감점 (양성 보호 실패)
- **결론 시나리오**:
  - 차이 유의미 없음 → 더 단순한 Gemini `luma-veto only` 채택
  - color-veto 유의미 우위 → Codex 안 채택
- **소요**: 1시간

---

## 3. 구현 순서 (W3-05 작업 분할)

**S1: 롤백 (1~2시간)** — §1.1 제거 작업 D1~D6 일괄 커밋.
- Commit: "feat(gpu-lens): W3-04 고정조명·realSpec·Normal/Overlay/SoftLight 블렌드 제거"
- 이 단계에서 **렌즈 렌더링 품질은 일시적으로 하락**할 수 있음. 벤치 비교 기준점(OFF baseline)으로 활용.

**S2: EyeRenderPacket 도입 + 단순 수용분 구현 (3~4시간)** — §1.2 C1~C11 중 벤치 독립 항목.
- 블렌드 4종 재편 (C1, C2, C3, C4) — 단 C4는 B1 결과 반영 대기
- 환경 반사 계층 분리(C5)는 **스캐폴드만 준비**, 실제 반사 소스(env map or camera)는 B2 결과 후
- 림발 기본 ON + 메타 플래그(C6)는 C6 스캐폴드, 자동 감지는 B4 결과 후
- 블링크 ramp(C7), avg_iris_luma 측정(C9), 디테일 재주입(C10) 구현
- EyeRenderPacket 도입(C8)
- sclera color-veto 수식은 B8 결과 후

**S3: 벤치 실행 (6~8시간)** — §2 B1·B2·B4·B8 병렬/순차 실행.

**S4: 벤치 결과 반영 (2~3시간)** — 벤치 결론대로 C4, C5, C6(자동감지), C10(sclera) 최종 구현.

**S5: 통합 테스트 (2~3시간)** — 실기기 3대(HIGH/MID/LOW tier) 30분씩 + 양안 동작 검증.

**총 소요**: 14~20시간 = 2~3일 작업량.

---

## 4. Phase 6 (W3-05 범위 밖)으로 밀 것

| 항목 | 출처 | Phase 6로 미루는 이유 |
|------|------|---------------------|
| 3D Face Geometry 기반 렌즈 렌더링 | 피팅몬스터 관찰 | 결합도 증가 + 현 구조에서 파급 큼 |
| HDR IBL (Perfect `eye_ibl.hdr`급) | Perfect 관찰 | LDR/RGBM env map으로 충분 검증 전까지 불필요 |
| Full PBR (Cook-Torrance) | 업계 표준 | 렌즈 도메인엔 과잉 |
| Neural rendering / harmonization | 업계 트렌드 | SDK 크기·지연 예산 파괴 |
| Corneal refraction 실시간 시뮬 | 광학 이론 | 육안 체감 미미 |
| 속눈썹 전용 세그멘테이션 모델 | Perfect 관찰 | 눈꺼풀 타원 feather로 충분 |
| Pupil cutout 동적 처리 (Gemini R2 추가 지적) | Gemini R2 | 동공 영역 baseLum 하한 강제 — 측정 후 추가 가능 |
| Head-pose 기반 env 회전 | Codex R1 | W3-05에서 optional 필드로만 준비, 실제 활용은 W2 refiner 출력 이후 |

---

## 5. 합의 축약 (R2에서 수렴된 것)

| 합의 | 수렴 방법 |
|------|----------|
| HDR IBL 에셋(1.8MB) 기각 | R1 세 모델 합의 |
| Full PBR / 3D FaceGeometry / Neural rendering 기각 | R1 세 모델 합의 |
| LTL 계열을 기본 블렌드로 | R1 세 모델 합의 |
| Multiply 블렌드 유지 | R1 세 모델 합의 |
| 추가 FBO pass 0개 (single-pass 절대 조건) | R1 세 모델 합의 |
| W1 TemporalStabilizer가 좌표 스무딩 소유, 렌더러는 material-only | Claude/Codex R1, Gemini 미반대 |
| 고정 조명 폐기 | R1 세 모델 비판 → R2 철회 확정 |
| 림발 비활성 기본값 폐기 | R1 비판 → R2 전원 수용 |
| realSpec 임계값 로직 폐기 | R1/R2 세 모델 합의 |
| 블렌드 4종(TintLinearV2/Multiply/ScreenLinear/ColorReplaceLinear) | R2 Gemini 수용, Claude 수용, Codex 제안 |
| EyeRenderPacket 구조체 | R2 세 모델 수용 |
| avg_iris_luma 하드코드 폐기, ROI 측정 | R2 세 모델 수용 |
| 홍채 디테일 재주입 (원본 휘도) | R2 Gemini 철회, Claude 수용, Codex 제안 |
| 블링크 alpha ramp (down 60~80ms, up 100~120ms) | R2 수렴 |
| 림발 기본 ON + 메타데이터 플래그 | R2 Claude 철회 |
| W2-W3 경계: refiner는 W2, 렌더러는 W3 | R2 Codex 철회 |
| sclera color는 veto로만 (주판정은 geometry) | R2 Claude 철회 (단 Gemini는 채도까지 드롭 주장 — §2.B8 벤치) |

---

## 6. R3 불필요 판단

3개 남은 쟁점(I1·I4·I8) 모두 실기기 벤치로 확정 가능. 말로 더 돌려도 진전 없음. **R3 생략, 바로 S1~S5 구현 착수 권장**.

---

## 7. 부록 — R2 응답 파일 참조

| 파일 | 크기 | 설명 |
|------|------|------|
| `brief.md` | 20KB | R1 공통 브리프 |
| `claude_response.md` | 14KB | Claude R1 |
| `gemini_response.md` | 7.5KB | Gemini R1 |
| `codex_response.md` | 19KB | Codex R1 (가장 상세, 파일:라인 레퍼런스 포함) |
| `r2_issues.md` | 13KB | Claude 작성 R2 쟁점 목록 (닫힘 12 + 열림 10) |
| `claude_r2.md` | 13KB | Claude R2 (9/10 쟁점 Codex 수용) |
| `codex_r2.md` | 12KB | Codex R2 (I10만 철회, 나머지 근거 보강) |
| `gemini_r2.md` | 5.8KB | Gemini R2 (I7 전격 수용, I1/I4/I8/I10 유지) |
| `final_decision.md` | 이 파일 | 종합 결론 |

---

## 8. 종합 통계

- **닫힌 결정**: 17개 (§1.1 제거 6 + §1.2 변경 11)
- **벤치 대상**: 4개 (B1, B2, B4, B8)
- **Phase 6 이월**: 8개
- **R1→R2 Claude 입장 변경**: 9/10 쟁점 (Codex 논리 우위)
- **R1→R2 Gemini 입장 변경**: 5/10 쟁점 (수정 or 철회)
- **R1→R2 Codex 입장 변경**: 3/10 쟁점 (I2 신중 수정, I5 Claude 시간 수용, I10 철회)

**결론**: Codex가 R1에서 가장 탄탄한 논리를 제시했고, Claude와 Gemini가 대부분 수렴하는 형태로 정리됨. 남은 쟁점 3개는 실기기로 판정 가능한 성격이므로 R3 없이 구현 착수가 최적.
