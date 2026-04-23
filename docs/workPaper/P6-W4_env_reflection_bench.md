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

---

## 2. 배경/맥락
_TODO_

## 3. 전제 조건
_TODO: P6-W3 완료 (sampleReflection 추상화 + renderMask hook 존재)_

## 4. 목표
_TODO_

## 5. 99에서 확정된 사항
_TODO_

## 6. 미결 사항
_TODO: §1.12 질문 정리_

## 7. W 브레인스토밍 시작 체크리스트
_TODO_

## 8. 완료 정의 + 다음 W 트리거
_TODO_

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
