# P6-W4: B2 환경 반사 소스 벤치 + Pupil 체감 지표 수집

> **상태**: 인사이트 작성 완료. 세부 계획 본문 작성 대기.
> **작성**: 2026-04-23
> **선행 의존**: P6-W3 (환경 반사 계층 스캐폴드 완료)
> **후속 의존**: P6-W8 (조건부 — B2 체감 지표 기반 발동)

---

## 1. 인사이트 (세션 간 맥락 보존) ⭐

### 1.1 이 W의 두 가지 목적

1. **B2 벤치**: env map / periphery camera / OFF 3 프로토타입 비교 → C5 반사 소스 확정 → realSpec 완전 폐기(D3) 여부 확정
2. **Pupil cutout 체감 수집**: B2 벤치 중 "중앙 공동 체감" 지표 Y/N 체크 → P6-W8 조건부 트랙 착수 판정

두 작업은 **같은 촬영 클립에서 동시 수집** 가능 → 효율적.

### 1.2 세 모델 입장 재확인 (벤치 설계 근거)

- **Codex R1/R3**: env-map-only 지지. "전면 카메라는 얼굴밖에 안 찍어 오프스크린 광원 부재. 재귀 반사 위험."
- **Gemini R1/R3**: Periphery 샘플링 (카메라 가장자리 8포인트) 지지. "에셋 env map은 정적이라 실시간 환경 변화에 반응 못함."
- **Claude R1 (부분 철회)**: 카메라 mip-down 지지했으나 R2에서 "env map 디폴트 + 상단 crop hybrid"로 부분 철회. Gemini R3에서 "hybrid는 복잡도만 늘림" 비판.

**Gemini R3 §3 요구**:
> "Claude의 'hybrid' 옵션은 프로토타입 복잡도만 높이므로 `env-map-only` vs `periphery-only` 대결이 선행되어야 함."

즉 1차 벤치는 **3 프로토타입 (OFF / env-map / periphery)** 만. Hybrid는 2차.

### 1.3 B2 프로토타입 3종 구체

**프로토타입 1: OFF (baseline)**
- `sampleReflection` returns `vec3(0.0)`
- D3 realSpec은 여전히 제거된 상태 (S1에서 삭제 완료). 기본 baseline.
- 이건 "현재 S1 상태"와 동일 — 추가 작업 필요 없음

**프로토타입 2: env-map-only (Codex 안)**
```glsl
vec3 sampleReflection(vec2 reflectUV, vec3 normal, vec3 viewDir) {
    return texture(uEnvMap, reflectUV).rgb * uReflectionIntensity;
}
```
- 에셋 크기: 256×128 LDR/RGBM (Codex R1 권고)
- `reflectUV`: normal과 viewDir로 sphere map 계산
- `pose`가 있으면 env map 회전 (`uEnvRotation`), 없으면 정적
- 제작: 일반 실내 장면 HDR을 LDR로 압축한 가상 환경맵 1장

**프로토타입 3: periphery-camera (Gemini 안)**
```glsl
vec3 sampleReflection(vec2 reflectUV, vec3 normal, vec3 viewDir) {
    // 화면 상단 1/3 + 좌우 가장자리 8 포인트 샘플링
    vec3 avg = vec3(0.0);
    for (int i = 0; i < 8; i++) {
        vec2 edgeUV = PERIPHERY_POINTS[i];  // 사전 정의된 가장자리 포인트
        avg += textureLod(uCameraTexture, edgeUV, 3.0).rgb;
    }
    return (avg / 8.0) * uReflectionIntensity;
}
```
- 얼굴 영역 배제: `face_region_mask` 미확정 (Codex R3 지적). 초기 구현은 "대략적 중앙 얼굴 영역 제외 — 화면 중앙 70%는 샘플링 안 함"으로 근사.
- `face_region_mask` 정밀화는 1차 결과 보고 2차에서.

### 1.4 Codex R3 §3 B2 벤치 매트릭스 검증

**Codex 동의**: 환경 4종 × 동작 2종 = 8 시나리오 × 3 프로토타입 = 24 클립.
**Codex 지적**:
> "`face_region_mask`는 R2에서 합의된 구현 요소가 아니고 정의도 없다. 벤치 프로토타입 설명에서 빼거나 별도 미확정 구현으로 표기해야 한다."

→ **반영**: "face_region_mask 구현은 미확정 — 프로토타입 3 설명에 '대략적 얼굴 영역 제외' 명시 + W4 브레인스토밍에서 구체화".

### 1.5 판정 메트릭 (R4 Patch 1 + Patch 2 반영)

```
정성 체감 Y/N 체크 + 다수 의견 (정량 카운트 아님)
- 자연스러움 블라인드 선호도 (양성, 1순위)
- 가운데 붙은 반짝이 체감 (음성)
- 얼굴 재귀 반사처럼 보임 (음성, Codex 우려)
- 좌우 눈 불일치 체감 (음성)
- 환경과 무관해 보임 (음성, Gemini 우려)
- 중앙 공동(cavity) 체감 (음성, P6-W8 착수 판정 근거)
  → 특히 밝은 렌즈 × 짙은 홍채 조합 중점 관찰
     (엔비_퍼퓸글로우, 클라셋_런웨이그레이 × 아시아 짙은 홍채)
```

### 1.6 벤치 환경 4종 × 동작 2종 구체

**환경 4종**:
1. 실내 형광등 (사무실, 오피스 카페)
2. 창가 측광 (한쪽에서 들어오는 자연광)
3. 야간 실내 (어두운 조명, 노란 불빛)
4. 실외 낮 (자연광 직접)

**동작 2종**:
1. 정면 미세 움직임 (얼굴 고정, 눈 깜빡임만)
2. 좌우 head turn (얼굴 회전, 환경 반사 변화 관찰용)

### 1.7 테스트 SKU 선정

B2는 **Pupil 체감 수집이 핵심 목적 중 하나**라 특히 "중앙 공동 체감"이 잘 발생하는 조합 필요:

**주 SKU (반사 성격 관찰)**:
- 클라셋_런웨이 그레이 (밝은 톤, 반사광 잘 드러남)
- 엔비_퍼퓸 글로우 (쿨톤, 반사 색 변화 민감)

**부 SKU (Pupil 체감 강조)**:
- 클라셋_런웨이 그레이 × **짙은 홍채** 사용자 (Pupil 문제 최대)
- 엔비_퍼퓸 글로우 × 짙은 홍채
- 대조: 클라셋_돌 초코 (짙은 렌즈, Pupil 문제 미미) × 짙은 홍채

총 3 SKU × 4 환경 × 2 동작 × 3 프로토타입 = **72 클립** (실제로는 벤치 범위 조정 가능 — 24 정도로 줄이되 대표 조합만)

**Claude 제안**: 24 클립 타겟 (2 SKU × 4 환경 × 2 동작 × 3 프로토타입 / but 실제론 매트릭스 관리 상 조정).

→ W4 브레인스토밍에서 최적 조합 확정.

### 1.8 Pupil 체감 수집의 제품 임팩트

사용자 원칙 (1번 답변): "재질에서 오는 감도... 눈 수분과 만나면 투명에 가까워지는 효과가 날거같고 자연스럽게 빛 반사 효과나 이런거로 인해 커버되지 않을까 하는 부분에 더 가까워, 정 어색하면 그때 가서 방향을 다시 결정하자에 가깝지"

→ **B2에서 env 반사/periphery 반사가 동공 영역을 자연 커버**하면 P6-W8 불필요. 그러니까 B2의 렌즈 중앙 영역에서 반사가 얼마나 스며드는지가 핵심 관찰.

**판정**:
- 3명 중 2명 이상 "중앙 공동 체감" Y → **P6-W8 착수** (재질 반투명 복원)
- 모두 "체감 없음" → P6-W8 폐기
- 1명만 Y → 사용자 최종 판단

### 1.9 B2 결과 시나리오 → 후속 W 영향

| B2 결과 | C5 소스 | D3 realSpec | P6-W8 Pupil 착수? |
|---------|---------|-------------|------------------|
| env-map 명확 우세 | env-map 채택 | 완전 폐기 | 체감 지표 따로 |
| periphery 명확 우세 | periphery 채택 | 완전 폐기 | 체감 지표 따로 |
| 둘 다 OFF 대비 유의미 개선 | 2차 hybrid 검토 | 완전 폐기 | 체감 지표 따로 |
| 둘 다 차이 미미 | **환경 반사 Phase 6 이월** | **조건부 유지 (재도입 검토)** | 보통 체감 높음 → 착수 |

### 1.10 Codex R2 경고 — env-map의 약점 인정

Codex도 R1에서 약점 인정:
> "env가 너무 generic하면 실내/역광에서 반사 성격이 과장될 수 있다. pose 없이 쓰면 움직임 설득력이 약해진다."

즉 env-map-only도 "완벽한 해결책이 아니라 덜 나쁜 옵션" 인식. W4 벤치 결과 해석 시 이 관점 유지.

### 1.11 벤치 실행 소요

- 프로토타입 구현 (env-map + periphery): 2~3h (env map 에셋 제작 포함)
- 벤치 캡처 (24 클립): 1.5~2h (환경 4개 × 동작 2개 × 3 프로토타입, 실기기 이동 포함)
- 블라인드 평가 (3명): 1h
- 결과 정리 + 반영: 0.5~1h

**총: 5~7h** (99 §2 B2 추정 5~6h와 일치).

### 1.12 W4 브레인스토밍 시 Codex/Gemini에게 던질 질문

1. **env map 에셋 제작**: 어떤 HDR 장면을 LDR로? 일반 실내/사무실/카페 중? 톤매핑 방식?
2. **Periphery 포인트 좌표**: 화면의 어느 8개 지점? 상단 4 + 좌우 4? 얼굴 영역 제외 로직 근사 형태?
3. **`face_region_mask`**: 초기 근사 — 화면 중앙 몇 % 제외? MediaPipe Face Detector 사용 가능?
4. **환경 4종 선정 재확인**: 야간 실내와 실외 낮이 너무 극단 아닌지? 실용 범위 중심 선정?
5. **SKU 조합**: 72 클립은 너무 많고 24는 부족할 수 있음. 실제 조합 선정 기준?
6. **평가자 3명 표준화**: 아시아 짙은 홍채 사용자 1명 필수? 밝은 홍채 대조 1명 필요?
7. **Pupil 체감 기록 방법**: 각 클립에 "공동 체감 Y/N" + 한 줄 자유 기술? 통일 양식 필요.
8. **반사 강도 uniform 실시간 조정**: 벤치 중에 intensity 바꿔가며 관찰? 고정값으로 판정?

### 1.13 구현 범위 (이 W에서 건드리는 파일)

- 신규:
  - 프로토타입 2 (env-map): `cpp/src/gpu/experimental/env_map_reflection.{h,cpp}` (위치 미정)
  - 프로토타입 3 (periphery): `cpp/src/gpu/experimental/periphery_reflection.{h,cpp}`
  - env map 에셋: `android/demo-app/src/main/assets/env/office_ldr.png` 같은
- 수정:
  - `shader_sources.cpp` — sampleReflection 구현을 프로토타입별로 컴파일 variant 또는 uniform 스위치
  - gpu_lens_renderer.cpp — 프로토타입 교체 API
  - Android demo UI — 프로토타입 선택 토글 (벤치 편의)

### 1.14 벤치 캡처 자료 저장 위치

- 캡처 비디오/이미지: `docs/bench/P6-W4/{env_type}/{prototype}/{sku}_{action}.mov`
- 평가자 응답: `docs/bench/P6-W4/ratings_YYYY-MM-DD.csv`
- 최종 리포트: `docs/workPaper/P6-W4_bench_report.md` (이 W 완료 시 생성)

### 1.15 W3 R1 재검토 hand-off (2026-05-13 추가)

W3 R1 재검토(2026-05-13)에서 W4로 넘어온 hand-off 3건. 원본 재검토 노트: `P6-W3_env_reflection_scaffold.md` §1.15.

1. **`RENDER_MASK_HOOK_ENABLED=1` debug APK 구성 절차** ⚠️ **W4 액션 필요**
   - W3 §5.10 `#ifdef` 채택 결과 W4 벤치 동안 실시간 on/off 토글 불가
   - W3 §5.10: "W4 벤치 전용 debug 빌드: CMake 옵션으로 `RENDER_MASK_HOOK_ENABLED=1` 켠 APK를 별도 산출"
   - **W4 액션**: 프로토타입 구현 단계에서 production APK + debug APK 2종 빌드 스크립트 준비
   - 단, W4 1차 벤치 핵심(OFF/env-map/periphery 비교)은 §5.11 방식 A uniform 스위치로 토글 → renderMask hook은 W8 트랙 발동 시 별도 검증 영역

2. **`uReflectionIntensity = 0.3` W3 R1 외부 모델 미토론** (note)
   - W3 R1 응답에 강도 토론 없음. W3 §1.7에서 §5.7로 직승된 값
   - W4 벤치 실기기 튜닝 자유. 24클립 1차 촬영은 0.3 고정. 강도 sweep은 후속(W9 통합 시)

3. **env-map SDK 내장 보류 — W8~W9 hand-off** (note)
   - W4 단계는 demo assets 위치 그대로 사용 (W4 §5.7)
   - SDK 내장 결정은 W9 통합 단계 SDK surface 정합성 점검 시. W4에서는 변경 없음

### 1.16 Phase A 검증 결과 — Visibility Budget 누락 사후 보완 (2026-05-18)

**증상**: W4 Phase A 코드 구현 완료 후 실기기 검증에서 OFF/EnvMap/Periphery 3 프로토타입 모두 시각 차이 인지 불가. 디버그용 자극적 env_map(4 사분면 빨/노/녹/파, 채도 최대) + intensity 0.3 + LUMINANCE_TINT_LINEAR 블렌드에서도 동일.

**원인 분석 (Codex gpt-5.4 medium 검증 + Claude critical-review)**:
- 출처: `P6-W4_brainstorm/phase_a_issue.md`, `phase_a_issue_codex.md`
- 1차 원인: **`renderMask = finalAlpha` (W3 §5.5)와 Fresnel C `smoothstep(0.7, 1.0)` (W3 §5.8)의 외곽 중첩 충돌**. Fresnel이 최강일 때(dist≈1.0) edgeAlpha=0 → 가산 0. 최대 가산 ≈ 0.04 (RGB ~10) → 인지 불가.
- 2차 원인: **intensity CPU clamp 0.0~1.0** (`gpu_lens_renderer.cpp:408`). 1.5~3.0 sweep 시도 자체 봉쇄.
- 3차 원인: sRGB 가산 단위 (`blendTintLinearV2` toSRGBFast 반환 위에 가산). 강도 약화.
- W3 R1 합의 시 visibility budget 검증 누락 → "자연스러움" 철학과 별개로 가시성 0인 설계.

**적용 패치** (W3 §5 R1 합의 부분 보완):

| 항목 | R1 원안 | Phase A 보완 | 변경 사유 |
|------|---------|------------|-----------|
| `renderMask` | `finalAlpha` (= lens.a × uOpacity × edgeAlpha × eyelidMask) | **`lens.a × uOpacity × eyelidMask × step(dist, 1.0)`** (edgeAlpha 제거 + silhouette hard cutoff) | Fresnel + edgeAlpha 중첩 해소. edgeAlpha의 silhouette 가드 역할은 `step(dist, 1.0)`로 분리 복원 (lensCoord clamp가 dist>1.0에서도 lens.a를 반환하는 사이드 이펙트 차단) |
| Fresnel inner | `smoothstep(0.7, 1.0)` | **`smoothstep(0.6, 1.0)`** (outer 1.0 유지) | R1 후보 [0.6, 0.7, 0.8] 중 가장 안쪽 채택. 외곽 40%에 반사 분포 |
| intensity clamp | `0.0~1.0` (영구) | **`0.0~5.0`** (디버그/벤치 sweep용) | sweep 봉쇄 해제. 자연스러움 권장값 0.3은 유지 |
| Demo UI | VOLUME_UP 모드 토글 | **+ VOLUME_DOWN intensity sweep** (0.3 → 1.0 → 2.0 → 3.0) | 벤치 중 강도 변경 + Toast 표시 |

**거부된 Codex 권장 사항**:
- ❌ `reflectionMask` 신규 명명 — W3 §5.5 `renderMask` 명명 보존 정합. 같은 변수 재정의.
- ❌ `reflectionMask *= smoothstep(1.05, 0.85, dist)` 추가 곱셈 — GLSL ES 3.10 §8.3 명세상 `edge0 >= edge1`은 undefined behavior. magic number 추가 + 검증 없는 가정.
- ❌ Fresnel `smoothstep(0.4, 0.8)` — R1 후보 [0.6, 0.7, 0.8] 밖. R1 합의 번복 시 brainstorm 재호출 필요.

**Phase B 진입 조건** (Codex 권장 + Claude 수용):
- ✅ OFF/EnvMap 정지 화면 비교만으로도 구분 가능
- ✅ 관찰자가 3초 내 차이 인지
- (Periphery는 약해도 OK — 본질적 소스 한계, 8포인트 평균)

**관련 파일**:
- `cpp/src/gpu/shader_sources.cpp` (renderMask + calcFresnel 변경)
- `cpp/src/gpu/gpu_lens_renderer.cpp` (clamp 0~5)
- `android/demo-app/src/main/java/com/irislenssdk/demo/GpuRenderActivity.kt` (VOLUME_DOWN sweep)
- `P6-W3_env_reflection_scaffold.md` §5.5/§5.8 (R1 보완 사유 명시)

---

## 2. 배경/맥락

### 2.1 W4가 Phase 6 중간 관문인 이유

**"말로 결론낼 수 없는 쟁점"**의 첫 벤치. R3까지의 논쟁(Codex env-map vs Gemini Periphery vs Claude hybrid)을 **실기기 블라인드 비교**로 종결. 결과에 따라:
- W3에서 스캐폴드만 둔 **반사 소스 확정**
- **realSpec 폐기 확정 여부** (W2에서 "조건부" 상태)
- **W8 Pupil material 조건부 트랙 발동 여부**

즉 W4 하나가 **3개 후속 결정**을 연쇄 해결.

### 2.2 두 가지 목적 동시 수행

1. **B2 벤치**: 반사 소스 비교 (OFF/env-map/periphery)
2. **Pupil 체감 수집**: B2 촬영 중 "중앙 공동(cavity) 체감" 지표 Y/N 수집

두 목적이 **같은 촬영 세트**에서 동시 수행 가능 → 비용 절약 + 판정 일관성.

### 2.3 Claude 편향 결과 ("hybrid" 안 축소)

**R4 Patch 기반** (99 §2 B2):
> "Claude의 'hybrid(env + 상단 crop)'는 프로토타입 복잡도만 늘리므로 **1차 대결에서 제외**. env-only vs periphery-only 선행, 두 프로토타입 모두 OFF 대비 유의미 개선 시 2차로 hybrid 검토."

→ W4 **1차 벤치는 3 프로토타입 (OFF / env-map / periphery)**. Hybrid 제외.

### 2.4 이 W가 해결하지 않는 것

- 실제 반사 소스 구현 **외** 다른 결정 — W5/W6/W7 범위
- Pupil material 실제 구현 — W8 (조건부 발동 시)
- 반사 intensity 세밀 튜닝 — W9 통합 단계

### 2.5 W4의 독특한 성격 — "구현 비중 < 벤치 비중"

W1~W3는 브레인스토밍 + 구현. W4는 브레인스토밍 + **실기기 촬영** + 평가가 주. 프로토타입 구현 2~3h, 촬영 1.5h, 평가 1h 구조.

---

## 3. 전제 조건

1. ✅ **W3 완료** — sampleReflection 추상화 존재, renderMask hook 구조. 단일 셰이더에서 uniform으로 OFF/env-map/periphery 전환 가능해야.
2. ✅ **W1 완료** — EyeRenderPacket.reflection_dir 필드 활용 가능 (env-map 회전 여부 판단)
3. ✅ **실기기 4 환경 이동 가능** — 실내 형광/창가 측광/야간 실내/실외 낮
4. ✅ **평가자 3명 확보** — 아시아 짙은 홍채 1 + 밝은 홍채 1 + 개발자/디자이너 1 권장
5. ✅ **env_map 에셋 공급** — W3 브레인스토밍에서 확정된 포맷 (256×128 LDR/RGBM)
6. ✅ **6개 대표 SKU 준비** — 15_asset_analysis.md에서 선정된 것
7. ✅ **녹화 장비** — 실기기 + 스크린 레코딩 or 외부 카메라

---

## 4. 목표

**W4 완료 시 달성 상태**:

1. **B2 결과 확정**: 3 프로토타입 중 하나 채택 or "Phase 6 이월"
2. **D3 realSpec 상태 확정**: 완전 폐기 or 조건부 유지
3. **Pupil 체감 지표 수집 완료** → W8 착수 판정 근거
4. **env_map 에셋 완성** (채택 시) — 실제 사용 가능 상태
5. **채택 소스의 sampleReflection 실제 구현 완료** — W3 no-op 교체

### 4.1 Definition of Done

- [ ] 3 프로토타입 구현 (env-map, periphery, OFF baseline)
- [ ] 4 환경 × 2 동작 × 3 프로토타입 = **24 클립** 촬영 완료
- [ ] 평가자 3명 블라인드 평가 완료
- [ ] 판정 결과 문서화 (`docs/bench/P6-W4/report.md` or 유사)
- [ ] 채택 소스의 sampleReflection 실제 구현 머지 완료
- [ ] "중앙 공동 체감" Y/N 3명×24클립 = 72 응답 수집
- [ ] W8 착수/폐기 판정 (2/3 룰)
- [ ] 99_final_decision.md §1.2 C5 "반사 소스 확정" 업데이트
- [ ] 99 §1.1 D3 "realSpec 상태 확정" 업데이트
- [ ] 실기기 회귀 확인 (채택 소스 적용 후 성능 유지)

### 4.2 Out of scope

- Hybrid 프로토타입 — 1차 결과가 "둘 다 유의미 개선"일 때만 2차
- Head-pose 기반 env 회전 — W4 범위에선 static env map 우선, 회전은 선택
- W8 실제 구현 — 조건 발동 시 별도 W

---

## 5. 99에서 확정된 사항

> 🔧 **구현 시 참조**: [`P6_implementation_handoff.md`](P6_implementation_handoff.md) §4.3 영역 경계 매핑 (Periphery annular ring 1.8~2.5r), §6 W4 핵심 포인트 (매트릭스·촬영·평가자·블라인드 절차). **W4 B2 결과에 따라 W8 활성 여부 결정 — Pupil 체감 Y/N 별도 기록 필수.**

### 5.1 B2 프로토타입 3종 (R4 Patch 반영)

**프로토타입 1: OFF (baseline)**
```glsl
vec3 sampleReflection(vec2 uv, vec3 n, vec3 v) { return vec3(0.0); }
```
- W3 기본 동작. 추가 작업 0.

**프로토타입 2: env-map-only (Codex)**
```glsl
vec3 sampleReflection(vec2 reflectUV, vec3 normal, vec3 viewDir) {
    vec3 env = texture(uEnvMap, reflectUV).rgb;
    return env;
}
```
- 에셋: 256×128 LDR/RGBM PNG. W3 브레인스토밍에서 파일 위치/이름 확정.
- reflection_dir가 있으면 env 회전 적용 가능 (W4에서 추가 결정).

**프로토타입 3: periphery-camera (Gemini)**
```glsl
vec3 sampleReflection(vec2 reflectUV, vec3 normal, vec3 viewDir) {
    // 화면 가장자리 8포인트 샘플링
    const vec2 PERIPHERY[8] = vec2[](
        vec2(0.1, 0.1), vec2(0.5, 0.05), vec2(0.9, 0.1),
        vec2(0.05, 0.5),                 vec2(0.95, 0.5),
        vec2(0.1, 0.9), vec2(0.5, 0.95), vec2(0.9, 0.9)
    );
    vec3 sum = vec3(0.0);
    for (int i = 0; i < 8; i++) {
        sum += textureLod(uCameraTexture, PERIPHERY[i], 3.0).rgb;
    }
    return sum / 8.0;
}
```
- **`face_region_mask`**: 99 §2 B2 "미확정" 명시. W4에서 "대략적 얼굴 영역 제외" 근사.
- 초기 구현: 화면 **중앙 70% 제외** (포인트 좌표 자체를 가장자리로 제한). 더 정교한 마스크는 1차 결과 후.

### 5.2 판정 메트릭 (R4 Patch 1 + 2 반영)

**정성 체감 Y/N 체크 + 다수 의견 판정**:

| 지표 | 성격 | 낮을수록 좋음 |
|------|------|--------------|
| 자연스러움 블라인드 선호도 | 양성 (1순위) | ❌ 높을수록 좋음 |
| 가운데 붙은 반짝이 체감 | 음성 | ✅ |
| 얼굴 재귀 반사처럼 보임 (Codex 우려) | 음성 | ✅ |
| 좌우 눈 불일치 체감 | 음성 | ✅ |
| 환경과 무관해 보임 (Gemini 우려) | 음성 | ✅ |
| **중앙 공동(cavity) 체감** (P6-W8 근거) | 음성 + Y/N | ✅ |

### 5.3 매트릭스: 4 환경 × 2 동작 × 3 프로토타입 = 24 클립

**환경**:
1. 실내 형광 (사무실/카페)
2. 창가 측광 (한쪽 자연광)
3. 야간 실내 (어두운 조명, 노란 불빛)
4. 실외 낮 (자연광)

**동작**:
1. 정면 미세 움직임
2. 좌우 head turn

**프로토타입**: OFF / env-map / periphery

### 5.4 Pupil 체감 집중 관찰 SKU

**밝은 렌즈 × 짙은 홍채** 조합에서 공동 체감 최대:
- **핵심 SKU**: 엔비_퍼퓸 글로우, 클라셋_런웨이 그레이
- **대조 SKU**: 클라셋_돌 초코 (짙은 렌즈, 체감 거의 없음)

**테스터 조건**: 아시아 짙은 홍채 주류.

### 5.5 결론 시나리오 (99 §2 B2)

| 결과 | C5 소스 확정 | D3 realSpec | W8 착수 |
|------|------------|-------------|---------|
| env-map 명확 우세 | env-map | 완전 폐기 | 체감 지표 별도 판정 |
| periphery 명확 우세 | periphery | 완전 폐기 | 체감 지표 별도 판정 |
| 둘 다 OFF 대비 개선 | 2차 hybrid 검토 | 완전 폐기 | 체감 지표 별도 판정 |
| 둘 다 차이 미미 | **Phase 6 이월** | **조건부 유지** (재도입 검토) | 보통 체감 높음 → 착수 |

### 5.6 Pupil 체감 W8 착수 판정 (Hard)

- 3명 중 2명 이상 "중앙 공동 체감" Y → **P6-W8 착수**
- 모두 "체감 없음" → P6-W8 폐기
- 1명만 Y → **사용자 최종 판단**

### 5.7 env_map 에셋 — **1장, ACES, PNG RGB 8bit 확정** (W4 R1 다수/합의)

- **장면:** 1장 generic 실내 사무실(중립 톤). 1차 벤치에서 변수 통제.
- **톤매핑:** **ACES Filmic** (Gemini+Claude 다수). highlight roll-off 자연스러움.
- **포맷:** PNG RGB 8bit 256×128. RGBM 불채택 (GPU decode 복잡도).
- **위치:** `android/demo-app/src/main/assets/env/env_default_256x128.png`.
- **확장 조항:** 1차 벤치 결과가 "env-map 채택 + 환경별 편차 큼"이면 2~3장(office/outdoor)으로 확장 판단 (W4-phase-2).
- 출처: `P6-W4_brainstorm/synthesis.md` §2.

### 5.8 Periphery face_region_mask — **옵션 A 확정** (W4 R1 합의 3/3)

- 중앙 70% 영역 제외 + 가장자리 고정 샘플 포인트.
- **구현 제안:** iris_center_uv 기준 **annular ring r ∈ [1.8, 2.5]** 에서 연속 샘플링 (Claude R1 제안). 동공 중심 공백을 채우면서 얼굴 피부 재귀 반사 방지.
- 동적 Face ROI(옵션 B)는 1차 벤치 결과 Periphery가 유효하면 후속 W에서 검토.
- 출처: `P6-W4_brainstorm/synthesis.md` §1.

### 5.9 Env map 회전 — **옵션 A (정적) 확정** (W4 R1 합의 3/3)

- 1차 벤치 정적 env only. `reflection_dir` / `head_pose_yaw_roll` optional 필드(W1 §5.11)는 예약 상태 그대로.
- 회전 검토는 "env-map 채택 후 자연스러움 부족" 피드백 시 2차 변수로.
- 출처: `P6-W4_brainstorm/synthesis.md` §1.

### 5.10 평가자 3명 — **구성 확정** (W4 R1 합의 3/3)

| # | 프로파일 | 필수 여부 |
|---|----------|-----------|
| 1 | 아시아 짙은 홍채 | **필수** (Pupil 체감 타겟) |
| 2 | 밝은/중간 홍채 | 대조군 |
| 3 | 개발자 또는 디자이너 | 기술/심미 관점 |

- 내부 3명 섭외로 1차 R1. 외부 평가자는 W4-phase-2 또는 W8 시점에 추가 수집.
- 출처: `P6-W4_brainstorm/synthesis.md` §1.

### 5.11 촬영 절차 — **표준화 확정** (W4 R1 합의/절충)

- **매체:** Android demo 앱 내 스크린 레코딩 (실기기).
- **디바이스:** 기준 1대 통일 (Galaxy S23 또는 사용자 지정 타겟).
- **프레임율:** 30fps.
- **클립 길이:** **동일 take 10초 녹화 → 평가용 5초 trim**. 블링크 1~2회 포착 + 평가자 부담 제한.
- **프로토타입 전환:** 같은 take에서 uniform 토글로 3프로토타입 연속 캡처 → 후편집으로 분리. 동작 차이 배제 (Codex 제안).
- **블라인드:** 파일명 무작위 ID(`clip_01.mp4`~`clip_24.mp4`) + 정답표 별도 암호화.
- 출처: `P6-W4_brainstorm/synthesis.md` §2.

### 5.12 SKU 및 클립 수 — **1차 단일 SKU 24 + 2차 부가 6 확정** (W4 R1 합의+Gemini 제안)

- **원칙:** 한 클립 = 1 환경 × 1 동작 × 1 프로토타입 × 1 SKU (3/3 합의).
- **1차 (필수):** **TintLinearV2 + iris_mat_B (고발광)** → 4×2×3=24 클립.
- **2차 (시간 여유 시):** 자연색 대조 SKU (iris_mat_A 등) → 환경 4 × 프로토타입 3 중 핵심 케이스 6 클립.
- 총 최대 30 클립 예상 시간: 녹화 30분 + 편집 60분 + 평가 3명 × 40분.
- 출처: `P6-W4_brainstorm/synthesis.md` §2.

### 5.13 결과 반영 체크리스트 — **일괄 PR 1개로 통합** (W4 R1 합의 3/3)

W4 종료 직후 체크:
- [ ] `99_final_decision.md` §1.2 C5 상태 업데이트
- [ ] 99 §1.1 D3 상태 업데이트
- [ ] 99 §2 B2 결과 섹션 신규 추가 (정량 + 정성 + Pupil Y/N 집계 + 2/3 판정)
- [ ] `P6-W3_env_reflection_scaffold.md` §5.11 sampleReflection no-op → 실제 구현 커밋 SHA 참조
- [ ] `P6-W4_env_reflection_bench.md` §5 최종 반영 (채택 프로토타입 명시)
- [ ] **Pupil 2/3 판정 결과 → W8 착수/폐기/사용자 판단 트리거**
- [ ] 반사 소스 판정 + realSpec 상태 + W8 착수는 연결 — 분리 업데이트 금지 (Codex 강조)
- 출처: `P6-W4_brainstorm/synthesis.md` §3.

---

## 6. 미결 사항 (W4 브레인스토밍 R1 결과)

### 6.0 R1 결과 요약 (2026-04-24)

| 번호 | 원 쟁점 | 상태 | 반영 위치 |
|------|---------|------|-----------|
| 6.1 | env_map 에셋 | ✅ **닫힘** (장면 Codex / 톤매핑 Gemini+Claude / 포맷 Codex+Claude) | §5.7 |
| 6.2 | face_region_mask | ✅ **닫힘** (3/3 합의 옵션 A) | §5.8 |
| 6.3 | env 회전 | ✅ **닫힘** (3/3 합의 정적) | §5.9 |
| 6.4 | 평가자 3명 | ✅ **닫힘** (3/3 합의) | §5.10 |
| 6.5 | 촬영 절차 | ✅ **닫힘** (합의 + 클립 길이 5초 절충) | §5.11 |
| 6.6 | SKU | ✅ **닫힘** (1차 단일 + 2차 부가) | §5.12 |
| 6.7 | 결과 반영 | ✅ **닫힘** (3/3 합의 일괄 PR) | §5.13 |

**Pupil 2/3 룰 재확인 (3/3 합의):** 2~3Y → W8 착수 / 0Y → W8 폐기 / 1Y → 사용자 판단.

원문: `docs/workPaper/P6-W4_brainstorm/{codex,gemini,claude}_w4.md`.
종합: `docs/workPaper/P6-W4_brainstorm/synthesis.md`.

**실기기 이관 항목 (W4 실행 시 수집):** env_map 장면 확장 여부, 2차 SKU 촬영 여부, 디바이스 편차 검증.

---

## (원 미결 사항 세부 — 참고용)

### 6.1 env_map 에셋 제작

- **장면 선정**: 실내 사무실? 카페? 일반적 형광 환경?
- **HDR → LDR 톤매핑**: 어느 방식? (Reinhard, ACES 등)
- **알파 채널**: 없음 (RGB only)
- **포맷**: PNG vs KTX? GPU 메모리 효율 고려 시 KTX RGBM이 나을 수도. 단 로딩 복잡성 증가.

**Claude 제안**: PNG RGBM 8bit — 호환성 최고. 200~400KB.

### 6.2 Periphery face_region_mask 근사

99 §2 "미확정" 명시. W4 구현 시:
- **옵션 A (간단)**: 8포인트 좌표 자체를 가장자리 고정. 얼굴 중앙 70% 자동 제외.
- **옵션 B (정교)**: MediaPipe Face Detector 결과로 face ROI 동적 계산 후 그 외 영역만 샘플링.

**Claude 추천**: 옵션 A 우선. 1차 벤치 결과 보고 옵션 B 필요성 판단.

### 6.3 Env map 회전 (reflection_dir 활용)

EyeRenderPacket.reflection_dir 또는 head_pose_yaw_roll optional 활용 여부:
- **옵션 A**: W4 1차 벤치는 **정적 env**로만. pose 회전은 별도 2차 벤치.
- **옵션 B**: W4부터 회전 적용. Pose 없으면 정적 fallback.

**Claude 추천**: 옵션 A. 벤치 변수 증가 방지.

### 6.4 평가자 3명 구성

- 아시아 짙은 홍채 1명 (필수 — Pupil 체감)
- 밝은 홍채 or 중간 1명 (대조)
- 개발자/디자이너 1명 (기술 관점)

**확인 필요**: 실제 평가자 섭외 경로.

### 6.5 촬영 절차 표준화

- 스크린 레코딩 vs 외부 카메라?
- 프레임율 (30 vs 60)?
- 녹화 길이 (클립당 3초 vs 10초)?
- 프로토타입 전환은 셰이더 uniform 토글 (런타임) vs 별도 APK 3종?

**Claude 추천**: 셰이더 uniform 토글 + 스크린 레코딩. 평가자 모름 (블라인드).

### 6.6 클립당 녹화 내용

- 한 클립 = 한 환경 + 한 동작 + 한 프로토타입 (단일 SKU)
- 여러 SKU 섞으면 평가 복잡 → 일단 1~2 SKU로 24 클립 돌리고, 필요 시 SKU 늘림.

### 6.7 결과 반영 절차

W4 완료 후:
- 99_final_decision.md §1.2 C5 업데이트
- §1.1 D3 상태 업데이트 ("조건부" → "완전 폐기" or "유지")
- §2 B2 결과 섹션 추가
- P6-W3의 sampleReflection no-op → 실제 구현 교체 (W4 산출물)

---

## 7. W4 브레인스토밍 시작 체크리스트

### 7.1 읽을 파일

**필수**:
1. P6-W0 §1
2. P6-W4 이 문서
3. P6-W3 완료 상태 확인 (hook/추상화 존재)
4. 99 §2 B2
5. R4 Patch (Patch 1 정성 메트릭, Patch 2 Pupil 체감 지표)

**선택**:
- 04_codex_response.md §A (env-map 원문)
- 14_gemini_r3.md §7 (Periphery 재강조)
- 15_asset_analysis.md §3 (SKU 선정)

### 7.2 송신 프롬프트 초안

```
@docs/workPaper/P6-W4_env_reflection_bench.md 읽고, 섹션 6 미결 7개에 
대해 각자 입장 정리 후 docs/workPaper/P6-W4_brainstorm/{codex|gemini}_w4.md
로 작성해줘.

특히:
- 6.1 env_map 에셋 제작 (Codex 원하는 장면/톤매핑)
- 6.2 Periphery face_region_mask 근사 (Gemini가 구체화 어디까지)
- 6.3 Env map 회전 (pose 있을 때 어떻게)
- 6.5 촬영 절차 (블라인드 보장 방법)

규칙:
- 새 쟁점 제기 금지
- 각 항목 "추천 + 근거 1~2줄"
- Pupil 체감 판정 2/3 룰 재확인
```

### 7.3 예상 대립

- 6.1 env 장면: Codex가 "사무실 실내" vs Gemini가 "다양한 환경 복수 에셋" 주장 가능. 벤치 범위 관리 필요.
- 6.3 env 회전: Codex가 "pose 있으면 회전 필수" 주장, Claude가 "1차에선 정적" 주장.

### 7.4 W4 실행 소요

- 프로토타입 구현: 2~3h (env-map 에셋 제작 포함)
- 벤치 촬영: 1.5~2h (환경 4 × 동작 2 × 프로토타입 3, 이동 시간 포함)
- 평가 1h
- 결과 반영 0.5~1h

**총 5~7h**.

---

## 8. 완료 정의 + 다음 W 트리거

### 8.1 완료 정의

§4.1 체크리스트 + `docs/bench/P6-W4/report.md` 존재.

### 8.2 커밋 전략

**커밋 1**: `docs(P6-W4): 섹션 2~8 본문 작성`
**커밋 2**: `feat(gpu-lens): P6-W4 env-map 프로토타입 추가 (B2 벤치용)`
**커밋 3**: `feat(gpu-lens): P6-W4 periphery 프로토타입 추가`
**커밋 4**: `feat(android): P6-W4 벤치용 프로토타입 토글 UI`
**커밋 5** (벤치 후): `chore(bench): P6-W4 B2 결과 report 및 산출물`
**커밋 6** (반영): `feat(gpu-lens): P6-W4 채택 소스 정식 적용 + realSpec 폐기 확정`

### 8.3 다음 W 트리거

**P6-W8 (조건부) 시작**:
- Pupil 체감 지표 2/3 이상 Y → W8 브레인스토밍 + 구현
- 1/3 Y → 사용자 판단 대기
- 0/3 Y → W8 폐기 + 관련 hook 정리

**W5/W6/W7**: W4와 **독립적** (반사 결과와 무관). W3 완료 후부터 병렬 진행 가능.

**W9**: 모든 W 완료 후.

### 8.4 W4 실패 시 전략

- 3 프로토타입 모두 OFF 대비 차이 미미 → **Phase 6 이월** 판정. D3 realSpec 조건부 유지.
- 이 경우 99 §4에 "환경 반사 Phase 6 이월" 명시. Phase 7+ 검토.
- W8 Pupil material은 여전히 조건부 발동 가능 (체감 지표 별도).

### 8.5 B2 결과가 Phase 6 전체에 주는 영향

**env-map 채택 시**:
- env_map.png 에셋 추가 (~300KB)
- 반사 계층 정식 동작 → C5 완전 실행
- realSpec 완전 폐기

**periphery 채택 시**:
- 추가 에셋 없음
- 반사 계층 정식 동작 → C5 완전 실행
- realSpec 완전 폐기
- face_region_mask 정교화 후속 작업 가능 (Phase 7+)

**OFF 확정 시** (모두 미미):
- 99 §4 Phase 6 이월 명시
- realSpec 재도입 or 대체 안 검토
- W8은 체감 지표 따라 여전히 발동 가능 → **가장 중요한 pivot**

---

## 참조

- 99_final_decision.md §2 B2
- 04_codex_response.md §축A (env map 주장 원문)
- 14_gemini_r3.md §7 Periphery 재강조
- 13_codex_r3.md §3 B2 매트릭스 검증
- 07_claude_r2.md I1 (Claude 부분 철회)
- 15_asset_analysis.md (SKU 컬러 스펙트럼)
- `feedback_real_data_first` 메모리
- `feedback_qualitative_device_judgment` 메모리
