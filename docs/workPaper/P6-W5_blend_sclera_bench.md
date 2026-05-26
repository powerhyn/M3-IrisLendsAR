# P6-W5: B1 블렌드 4번째 슬롯 + B8 sclera veto 벤치

> **상태**: Phase A/B 구현 완료 (2026-05-25) — 실기기 벤치 촬영/평가 대기 → Phase C에서 결과 반영.
> **작성**: 2026-04-23
> **선행 의존**: P6-W3 (환경 반사 계층 스캐폴드). P6-W2 (블렌드 3종)도 당연
> **병렬 가능**: P6-W4, W6, W7와 병렬 진행 가능 (W3 완료 후)

---

## 1. 인사이트 (세션 간 맥락 보존) ⭐

### 1.1 두 벤치를 한 W로 묶은 이유

**B1** (Normal vs ColorReplaceLinear) + **B8** (sclera color-veto vs luma-only)
- 둘 다 **iris 경계 처리 관련**: B1은 블렌드가 iris 전체를 어떻게 덮는지, B8은 iris/sclera 경계에서 흰자 번짐을 얼마나 막는지.
- 같은 SKU로 두 벤치 **동시 캡처** 가능 → 촬영 1회, 판정 2회.
- 프로토타입도 셰이더의 분기 플래그만 바꾸면 됨 (별도 프로토타입 필요 없음).

### 1.2 B1: Normal vs ColorReplaceLinear — Codex R3 §3 경고

**Codex R3 §3**:
> "내 R2 SKU에는 `화이트/그래픽 렌즈`가 포함됐는데 §2 B1은 `불투명 서클`로 축소했다. ColorReplaceLinear의 존재 이유가 색 정확도 높은 불투명/화이트/그래픽 셀이라 이 셀은 빠지면 안 된다."

→ **반영 (R4 Patch에서 이미 업데이트됨)**: SKU 5종 추가:
- 다크브라운 자연 → 클라셋_돌 초코
- 헤이즐 자연 → 오(OH)_베이글
- 밝은 그레이·블루 → 클라셋_런웨이 그레이 또는 엔비_퍼퓸 글로우
- 불투명 서클 → 클라셋_돌 초코 (중복 가능) or 로뮤_디어 멜로우
- **화이트·그래픽 → 엔비_샤모 브라운** (실측에서 "강한 블랙 아웃라인 그래픽 렌즈"로 분류됨)

### 1.3 B1 판정 시나리오

```
CRL 명확 우세 (다수 셀에서) → 4종 확정: TintLinearV2 / Multiply / ScreenLinear / ColorReplaceLinear
CRL 특정 SKU(화이트·그래픽)에서만 우세 → 조건부 채택:
  SKU metadata `prefers_crl: true` 플래그로 자동 선택
  다른 SKU는 Normal fallback
Normal 명확 우세 → 3종 확정 + Normal (TintLinearV2/Multiply/ScreenLinear/Normal)
차이 미미 → 3종 확정 + Normal (단순성 우선)
```

### 1.4 Codex R3 §3 — SKU 조합에서 Normal과 CRL의 차이 예측

**수식 비교**:
```glsl
// Normal:
out = mix(base, lens, a);

// ColorReplaceLinear:
detail = clamp(pow(lum/avgLum, 0.7), 0.75, 1.25);
out = sqrt(mix(baseL, lensL * detail, a));
```

**예측**:
- **불투명 서클 (짙은 초코 × 짙은 홍채)**: `detail ≈ 1.0` 근처 유지 → CRL ≈ linear Normal → sqrt로 감마 적용. 거의 동일.
- **화이트/그래픽 (엔비_샤모 × 짙은 홍채)**: baseL 어두움, detail 상한 1.25 도달 → CRL이 더 밝게. **CRL 우세 예측**.
- **밝은 그레이 × 밝은 홍채**: 둘 다 비슷할 듯. 차이 미미.

### 1.5 B8 sclera veto — Codex vs Gemini 대립

**R2 입장**:
- **Claude (부분 철회)**: `brightFactor 하한 0.3 추가` → R3에서 미봉책이라 철회, Codex veto 방식 수용.
- **Gemini**: "완전 폐기, luma-veto만" (조명 변화에 채도 불안정).
- **Codex**: `geometry-first + color-veto` (채도 + 밝기).

**구체 수식 (Codex R2)**:
```glsl
float geom = smoothstep(0.75, 1.0, irisEdgeDist);
float veto = smoothstep(0.18, 0.32, sat) * (1.0 - smoothstep(0.45, 0.65, lum));
finalAlpha *= 1.0 - geom * (1.0 - 0.6 * veto);
```

**Gemini R3 §8**:
> "Geometric-First + Luma-Veto. 색상(채도)은 버리고, 기하학적 영역 밖인데 너무 어두운 경우(그림자)만 렌즈를 깎는 luma 기반 가중치만 남김."

즉 Gemini는 **Codex 수식에서 `sat` 항 제거 + lum만 유지** 제안:
```glsl
float geom = smoothstep(0.75, 1.0, irisEdgeDist);
float veto = (1.0 - smoothstep(0.45, 0.65, lum));  // sat 제거
finalAlpha *= 1.0 - geom * (1.0 - 0.6 * veto);
```

### 1.6 B8 판정 시나리오

```
color-veto (Codex) 명확 우세 → 수식 전체 채택
luma-only (Gemini) 명확 우세 → sat 항 제거
차이 미미 → 더 단순한 luma-only (Gemini) 채택
그레이/블루 SKU에서만 luma-only 안전 → 최소공통 수식
```

### 1.7 B8 매트릭스 (99 §2 기준)

**SKU 2종 (쟁점 SKU)**:
- 밝은 그레이/블루 (채도 낮음 — Codex veto의 위험 영역)
- 다크브라운 (대조군)

**조명 3종 (채도 불안정 테스트)**:
- 형광 (정상 대비)
- 측광 (한쪽 그림자)
- 저조도 (채도 부정확)

**방식 2종**: color-veto / luma-only

**총 12 클립**.

### 1.8 B1 + B8 동시 캡처 전략

**핵심 아이디어**: 같은 실기기 촬영에서 블렌드 모드와 sclera 방식만 바꿔가며 4가지 조합 녹화:

```
조합 1: Normal + color-veto
조합 2: Normal + luma-only
조합 3: CRL + color-veto
조합 4: CRL + luma-only
```

각 조합을 화면 분할 또는 연속 토글로 촬영. 1 SKU × 1 환경당 4 조합 동시 비교 가능.

**효율**: B1 24 + B8 12 = 36 클립이 아니라, 조합형 20 클립 수준으로 감축 가능 (W5 브레인스토밍에서 확정).

### 1.9 주의: 블렌드와 sclera는 독립 변수

B1 결과와 B8 결과는 **개별적으로 해석**. 즉:
- B1에서 CRL 우세 → B1에서만 반영
- B8에서 color-veto 패배 → B8에서만 반영

서로 교차 영향 없음. 판정 독립성 유지.

### 1.10 Codex R3 §8 — B8 "color veto만"의 단서

**Codex R3 §8**:
> "원칙은 'geometry-first'로 이미 닫혔다. 남은 것은 `채도 포함 veto vs luma-only veto`라 벤치 이관이 맞다."

즉 geometry-first 자체는 이미 합의. B8은 **색상 veto 항의 세부**만 판정. R3에서 확정된 frame 내.

### 1.11 W5 브레인스토밍 시 Codex/Gemini에게 던질 질문

**B1 관련**:
1. CRL 수식의 `pow(lum/avgLum, 0.7)`에서 0.7 exponent 최적인가? 다른 값(0.5/0.9) 대안?
2. `detail` clamp 범위 `[0.75, 1.25]` 적절한가? `maxDetail` uniform 동적 조정 필요?
3. Normal이 특정 SKU(화이트/그래픽)에서 우세하면 "SKU metadata 플래그" 구조는?
4. 테스트 평가자가 판단 어려워하면 추가 기준?

**B8 관련**:
5. Codex 수식의 `smoothstep(0.18, 0.32, sat)` 임계값 재확인 (현재 코드의 `calcScleraFactor`와 차이)
6. Gemini luma-only가 저조도에서 "iris 외곽 어두운 무늬 잘림" 위험 어떻게 방어?
7. color-veto와 luma-only 중간안(sat 가중치를 낮춰서 혼합) 고려?

**공통**:
8. 동시 캡처 조합 4개 세팅 — 셰이더 uniform 스위치로 가능한가, 별도 빌드 variants 필요?
9. 블라인드 평가 시 조합 라벨 숨기기 (예: A/B/C/D로만 표시)
10. 평가자 3명 구성: 아시아 짙은 홍채 1 + 밝은 홍채 1 + 전문 (개발자/디자이너) 1?

### 1.12 구현 범위

- 수정:
  - `shader_sources.cpp`:
    - `blendColorReplaceLinear` 추가 (아직 없으면. W2에서 수식은 준비됨)
    - `calcScleraFactor` 수식 변경 (Codex 수식 또는 Gemini 단순화)
    - uniform 스위치 추가: `uBlendExperiment`, `uScleraVetoMode`
  - `gpu_lens_renderer.cpp` — 스위치 주입
  - Android demo UI — 벤치 토글 (B1/B8 조합 선택)
- 벤치 자료:
  - `docs/bench/P6-W5/` 폴더 생성
  - 평가지 표준 양식

### 1.13 B1 결과 시나리오별 W 완료 후 작업

- CRL 채택 → `shader_sources.cpp` 분기 `uBlendMode == 7`에 CRL 등록. `blendColorReplace` (sRGB) 함수 제거.
- Normal 유지 → `uBlendMode == 7` 제거, Normal 그대로.
- 조건부 CRL → `EyeRenderPacket`에 SKU 메타 전달 필드 추가? 또는 `uBlendMode` 호출자가 판단?

### 1.14 B8 결과 시나리오별 W 완료 후 작업

- color-veto 채택 → Codex 수식 그대로. 기존 `calcScleraFactor` 완전 교체.
- luma-only 채택 → sat 항 제거. 수식 단순화.

### 1.15 성능 예산

B1: 블렌드 분기 하나 추가 (또는 유지). 비용 동일.
B8: 기존 `calcScleraFactor`와 동일 호출 수. 수식 내부만 변경 → 비용 동일.

**총 추가 비용: 0**. W5 구현은 순수 치환 작업.

---

## 2. 배경/맥락

### 2.1 두 벤치 묶음의 논리

**B1**(블렌드 4번째 슬롯 Normal vs ColorReplaceLinear) + **B8**(sclera color-veto vs luma-only) 모두 **iris 경계 처리 관련**. 같은 SKU로 동시 촬영해 조합형 평가 가능.

### 2.2 W2에서 준비된 것

W2 완료 시점에:
- blendTintLinearV2/Multiply/ScreenLinear 3종 정식 등록
- blendColorReplaceLinear **수식 구현**만 (ID 7 등록 보류 or 조건부)
- Normal(0) 유지

W5는 여기서 **B1 A/B 테스트**로 Normal(0) 대 ColorReplaceLinear(7) 비교.

### 2.3 B8의 상태

- 현재 `calcScleraFactor` 함수 (기하학적 + 색상 기반) 존재 (S1 이전부터)
- 99에서 "geometry-first + color-veto"로 방향 전환
- 하지만 **수식 변경은 아직 안 됨** → W5에서 Codex 수식(color+luma) vs Gemini 수식(luma-only)으로 변경 + 벤치

### 2.4 독립성

**B1 결과와 B8 결과는 개별 해석**. 블렌드 수식 결정과 sclera veto는 서로 영향 없음. 판정도 따로.

### 2.5 W5가 해결하는 것

- B1 결과 → 99 §1.1 D6 "Normal 제거/유지" 확정, §1.2 C4 "CRL 채택/기각" 확정
- B8 결과 → 99 §1.2 "sclera color-veto 수식" 확정

### 2.6 W5가 해결하지 않는 것

- 다른 블렌드 수식 변경 (TintLinearV2 등) — W2 범위
- 환경 반사 — W4 범위
- 림발 처리 — W7 범위

---

## 3. 전제 조건

1. ✅ **W2 완료** — 블렌드 3종 확정, blendColorReplaceLinear 수식 존재
2. ✅ **W3 완료** — 환경 반사 계층 (이 벤치 동안 OFF로 두거나 W4 결과 반영 상태)
3. ✅ **실기기 + SKU 6종 준비** — 99 §2 B1 요구 SKU 5종 + B8 SKU 2종. 중복 SKU 활용 가능.
4. ✅ **평가자 3명** (W4와 동일 구성 or 교체)

---

## 4. 목표

**W5 완료 시 달성 상태**:

1. **B1 판정 완료** — Normal(0), ColorReplaceLinear(7) 중 채택/기각/조건부
2. **B8 판정 완료** — color-veto vs luma-only 중 채택
3. **셰이더 수식 최종 반영** — calcScleraFactor 교체, Normal/CRL ID 정리
4. **벤치 리포트 작성** — `docs/bench/P6-W5/report.md`

### 4.1 Definition of Done

- [x] **B1 조합 4종 동시 프로토타입** (Normal+color, Normal+luma, CRL+color, CRL+luma) 구현 — Phase A
- [x] **4조합 토글 인프라** — `uScleraVetoMode` 셰이더 3-way + JNI/Java API + demo A/B/C/D 버튼 — Phase A
- [x] **벤치 산출물 준비** — `docs/bench/P6-W5/` (체크리스트/응답시트/촬영가이드/스크립트) — Phase B
- [ ] 9 take × 4조합 = 36 클립 촬영 — **실기기 벤치 대기** (Codex 리뷰 F-03 반영: B8 2×3 조명 대칭)
- [ ] 3명 블라인드 평가 완료 — 실기기 벤치 대기
- [ ] B1/B8 결과 문서화 (`report.md`) — Phase C
- [ ] 셰이더 반영 (ID 0/7 처리, calcScleraFactor 수정, uScleraVetoMode 제거) — Phase C
- [ ] 99_final_decision.md §1.1 D6 / §1.2 C4 업데이트 — Phase C
- [ ] 회귀 확인 — Phase C

### 4.2 Out of scope

- 블렌드 완전 재설계 — 현재 결정 범위만
- ColorReplaceLinear 외 새 블렌드 추가 제안 — R4 이후 종결

---

## 5. 99에서 확정된 사항

> 🔧 **구현 시 참조**: [`P6_implementation_handoff.md`](P6_implementation_handoff.md) §4.5 SKU 메타 규약 (`lens_meta.json` 스키마, **W7과 공유 파일**), §2 소프트 갭 A (SKU 5 or 6 구현 단계 조정).

### 5.1 B1 매트릭스 (R4 Patch 반영, 5 SKU)

**SKU**:
1. 다크브라운 자연 → 클라셋_돌 초코
2. 헤이즐 자연 → 오(OH)_베이글
3. 밝은 그레이/블루 → 엔비_퍼퓸 글로우 또는 클라셋_런웨이 그레이
4. 불투명 서클 → 로뮤_디어 멜로우 또는 클라셋_돌 초코 중복
5. **화이트/그래픽** → 엔비_샤모 브라운 (CRL 존재 이유의 핵심 셀)

**홍채 톤**: 짙음(아시아인 주류, 50% 비중), 중간, 밝음

**모드**: Normal(ID 0), ColorReplaceLinear(ID 7)

총 5 × 3 × 2 = **30 클립** (또는 조합형으로 축소)

### 5.2 B1 판정 시나리오 (99 §2 B1)

```
CRL 17/30+ 승리 → 4종 확정 (TintLinearV2 + Multiply + ScreenLinear + CRL)
CRL 특정 SKU(화이트/그래픽)에서만 승리 → 조건부 채택 (SKU meta `prefers_crl`)
Normal 17/30+ 승리 → 3종 유지 + Normal
차이 미미 → 3종 유지 + Normal (단순성 우선)
```

R4 Patch 1 반영: **"17/30"은 정량 가이드이지 엄격 임계값 아님**. 다수 의견 기반 정성 판정.

### 5.3 B8 매트릭스 (99 §2 B8)

- SKU 2: 밝은 그레이/블루 (채도 낮음, 위험 영역), 다크브라운 (대조)
- 조명 3: 형광, 측광, 저조도
- 방식 2: color-veto, luma-only

12 클립.

### 5.4 B8 수식 후보

**Codex color-veto (R3 §8)**:
```glsl
float geom = smoothstep(0.75, 1.0, irisEdgeDist);
float veto = smoothstep(0.18, 0.32, sat) 
           * (1.0 - smoothstep(0.45, 0.65, lum));
finalAlpha *= 1.0 - geom * (1.0 - 0.6 * veto);
```

**Gemini luma-only (R3 §8)**:
```glsl
float geom = smoothstep(0.75, 1.0, irisEdgeDist);
float veto = 1.0 - smoothstep(0.45, 0.65, lum);  // sat 항 제거
finalAlpha *= 1.0 - geom * (1.0 - 0.6 * veto);
```

### 5.5 B8 판정 시나리오

- color-veto 명확 우세 → Codex 수식 채택
- luma-only 명확 우세 → Gemini 수식 (더 단순)
- 차이 미미 → Gemini 수식 (Occam's razor)

### 5.6 동시 캡처 전략 (효율)

4 조합 (Normal+color / Normal+luma / CRL+color / CRL+luma)을 **같은 촬영에서 셰이더 토글**로 녹화:

- 평가자는 A/B/C/D 라벨만 보고 블라인드 평가
- SKU 5종 × 환경 1~2종 × 4 조합 = 20~40 클립

### 5.7 기존 calcScleraFactor 제거 정책

S1 이전부터 존재하는 `calcScleraFactor` 함수는 **현재 기하학적 + 색상** 혼합식. W5 시작 시:
- W2에서 건드리지 않았음 (W5 범위)
- Codex 또는 Gemini 수식으로 교체
- 함수명 유지 or `calcScleraVeto`로 명확화?

**W5 브레인스토밍에서 결정**.

### 5.8 SKU 구성 — **별도 확보 확정** (W5 R1 합의 3/3)

- **로뮤_디어 멜로우** (불투명 서클) + **클라셋_돌 초코** (다크브라운) 별도 SKU.
- 디자인 차이(랜덤 도트 vs 균일 채움)가 CRL 수식 감수성에 차이를 만들 수 있어 겸용 불가.
- 5 → 6 SKU 가능. 매트릭스 변경은 W5 구현 단계에서 조정.
- 출처: `P6-W5_brainstorm/synthesis.md` §1.

### 5.9 조합 라벨링 — **A/B/C/D 블라인드 + 런타임 UI 확정** (W5 R1 합의 3/3)

- 같은 take에서 **4조합 연속 토글 캡처**:
  - A = Normal + color-veto
  - B = Normal + luma-only
  - C = CRL + color-veto
  - D = CRL + luma-only
- 평가자에게는 A/B/C/D만 노출. 정답표 암호화 별도 관리.
- **런타임 UI 버튼 4개** (디버그 커맨드 대신 — 평가 흐름 끊김 방지).
- **판정표에서 B1 점수(Normal vs CRL)와 B8 점수(color-veto vs luma-only) 분리 기록** → 독립 판정 보장.
- 출처: `P6-W5_brainstorm/synthesis.md` §1.

### 5.10 CRL clamp 범위 — **`[0.75, 1.25]` 유지 확정** (W5 R1 합의 3/3)

- `detail = clamp(pow(lum/avgLum, 0.7), 0.75, 1.25)`.
- `uMaxDetail` 기본값 = 1.25 (상한 그대로).
- **W5 1차에서 clamp 튜닝 금지** — 모드 비교(Normal vs CRL)에 집중, 튜닝 변수 배제.
- 1차 결과 "CRL 채택 + 과함" 피드백 시 후속 phase에서 1.15로 하향 검토.
- 출처: `P6-W5_brainstorm/synthesis.md` §1.

### 5.11 조건부 채택 — **`prefers_crl: bool` 메타 플래그 확정** (W5 R1 합의 3/3)

- SKU 메타데이터에 `prefers_crl: bool` (기본 `false`) 추가.
- 런타임 자동 선택: 플래그 있으면 CRL, 없으면 TintLinearV2 (W2 §5.12 default).
- **공개 UI/C API 오버라이드 없음**. `has_baked_limbal` 기존 패턴 동일.
- 근거: "사용자가 블렌드 모드 선택"은 W3-04 실패 패턴 (피팅 체험 자연스러움 저해).
- 출처: `P6-W5_brainstorm/synthesis.md` §1.

### 5.12 Gemini luma-only 임계값 — **`smoothstep(0.45, 0.65, lum)` 유지 확정** (W5 R1 다수 2/3)

- W5 1차 B8 벤치에서 원 임계 유지. 변수 분리 (color vs luma 구조만 비교, 임계 조정 배제).
- **Claude R1 원안 0.35/0.60 조정 제안은 소수 의견**으로 Codex+Gemini 다수에 의해 정정.
- 1차 결과 "luma-only 채택 + 저조도 홍채 외곽 깎임" 피드백 시 후속 W에서 0.3~0.5 하향 검토.
- 출처: `P6-W5_brainstorm/synthesis.md` §2.

### 5.13 calcScleraFactor — **함수명 유지 + 수식 교체 확정** (W5 R1 합의 3/3)

- 함수명 `calcScleraFactor` 유지. 호출부 영향 0.
- 내부 수식만 B8 채택 수식(color-veto 또는 luma-only)으로 전면 교체.
- 기존 수식 주석 유지 금지 (W2 §5.11 원칙 적용).
- 출처: `P6-W5_brainstorm/synthesis.md` §1.

### 5.14 color-veto `0.6` 강도 — **W5 1차 고정, phase-2 스위프 여지** (W5 R1 절충)

- **W5 1차 B8 벤치: `0.6` 고정** (Codex 원칙 — 구조 비교와 강도 분리).
- **phase-2 (B8 color-veto 채택 시):** 대표 SKU 2종에 `0.4 / 0.6 / 0.8` 스위프.
- **B8 luma-only 채택 시:** color-veto 튜닝 skip.
- 출처: `P6-W5_brainstorm/synthesis.md` §2.

---

## 6. 미결 사항 (W5 브레인스토밍 R1 결과)

### 6.0 R1 결과 요약 (2026-04-24)

| 번호 | 원 쟁점 | 상태 | 반영 위치 |
|------|---------|------|-----------|
| 6.1 | SKU 중복 | ✅ **닫힘** (3/3 별도) | §5.8 |
| 6.2 | 조합 라벨링 | ✅ **닫힘** (3/3 A/B/C/D + UI) | §5.9 |
| 6.3 | CRL clamp | ✅ **닫힘** (3/3 유지) | §5.10 |
| 6.4 | 조건부 채택 | ✅ **닫힘** (3/3 prefers_crl) | §5.11 |
| 6.5 | luma 임계 | ✅ **닫힘** (2/3 유지, Claude 편향 정정) | §5.12 |
| 6.6 | calcScleraFactor | ✅ **닫힘** (3/3 함수명 유지+수식 교체) | §5.13 |
| 6.7 | color-veto 강도 | ✅ **닫힘** (1차 고정, phase-2 스위프) | §5.14 |

**B1/B8 독립 판정 + 조합형 캡처 타당성 (3/3 재확인).**

**미결 없음.** 조건부 후속 작업: phase-2 color-veto 스위프 (B8 color-veto 채택 시), luma 임계 조정 (luma-only 채택 시).

원문: `docs/workPaper/P6-W5_brainstorm/{codex,gemini,claude}_w5.md`.
종합: `docs/workPaper/P6-W5_brainstorm/synthesis.md`.

---

## (원 미결 사항 세부 — 참고용)

### 6.1 SKU 중복 사용 여부

5 SKU 중 "불투명 서클"과 "다크브라운"이 겹칠 수 있음. 별도 SKU vs 중복?
- 별도 확보: 로뮤_디어 멜로우 (불투명 서클), 클라셋_돌 초코 (다크브라운)
- 중복: 클라셋_돌 초코 하나로 겸용

**Claude 제안**: 별도. 디자인 차이(랜덤 도트 vs 균일 채움) 관찰 가치.

### 6.2 B1과 B8 동시 캡처 조합 라벨링

4 조합을 A/B/C/D로 라벨:
- 평가자에게 라벨만 보여주고 어느 게 어느 조합인지 숨김 (블라인드)
- 판정 후 라벨 매핑 공개

**확인 필요**: 실기기에서 4 조합 토글 방식 (UI 버튼 vs 디버그 커맨드).

### 6.3 CRL `detail` clamp 범위 재확인

**현재 수식**: `detail = clamp(pow(lum/avgLum, 0.7), 0.75, 1.25)`

**maxDetail uniform**: `mix(uMaxDetail, 1.0, smoothstep(0.75, 1.0, irisEdgeDist))`

이게 `clamp(..., 0.75, maxDetail)`로 상한 대체. W5 벤치 결과 따라 maxDetail 기본값 조정 가능.

### 6.4 B1에서 "조건부 채택" 결과 시 구현

CRL이 화이트/그래픽에서만 승리 → SKU 메타데이터 `prefers_crl` 플래그:
- 메타 있는 SKU: CRL 자동 선택
- 메타 없는 SKU: Normal or TintLinearV2 자동
- UI에 노출? (사용자가 수동 덮어쓰기 가능?)

**W5 브레인스토밍에서 결정**.

### 6.5 Gemini luma-only 임계값 재확인

`smoothstep(0.45, 0.65, lum)` 값 그대로 쓸지, 튜닝?
- Codex 원 수식도 같은 값 사용
- 저조도 시나리오에선 0.45 하한이 너무 높을 수도 (실제 iris 외곽도 0.2~0.4 영역이 있음)

**W5 브레인스토밍에서 확인**.

### 6.6 calcScleraFactor 기존 구현과 새 수식의 관계

- 이름만 바꿔서 재작성?
- 새 함수 추가 + 기존 deprecated?
- 단순 수식 교체?

**Claude 추천**: 단순 수식 교체. 함수명 유지로 호출부 영향 없음.

### 6.7 color-veto 수식의 Codex 의도 재확인

Codex R3 §8 원문:
> "즉 색상은 감쇄를 켜는 스위치가 아니라 감쇄를 일부 되돌리는 veto다."

`veto` 가 **감쇄 해제**(1.0 - 0.6 * veto)로 쓰임. W5 브레인스토밍 시 Codex에게 재확인: "0.6"은 veto 강도 조절. 튜닝 가능?

---

## 7. W5 브레인스토밍 시작 체크리스트

### 7.1 읽을 파일

**필수**: P6-W0, P6-W5, 99 §2 B1/B8, P6-W2 (블렌드 수식 맥락)
**선택**: 13_codex_r3.md §8 (color-veto 원문), 14_gemini_r3.md §8 (luma-only)

### 7.2 송신 프롬프트 초안

```
@docs/workPaper/P6-W5_blend_sclera_bench.md 읽고, 섹션 6 미결 7개에 대해
각자 입장 정리. docs/workPaper/P6-W5_brainstorm/{codex|gemini}_w5.md.

특히:
- 6.3 CRL detail clamp 범위 (maxDetail 기본값)
- 6.4 "조건부 채택" 시 구현 (SKU 메타 플래그)
- 6.7 color-veto `0.6` 강도 튜닝 가능성

규칙:
- 새 쟁점 금지
- B1/B8 독립 판정 인정
- 조합형 캡처 전략 타당성 평가
```

### 7.3 예상 대립

- 6.4 조건부 채택: Codex가 "UI 숨김 자동 선택" vs "수동 덮어쓰기 허용" 중 입장 갈림 가능
- 6.7 Codex가 "0.6 엄수" 주장 가능 — Gemini는 "수식 자체 단순화 주장 (sat 항 제거)"와 별도 논점

### 7.4 예상 합의

- B1/B8 독립 판정 — 쉽게 합의
- 조합형 캡처 효율성 — Gemini 동의 예상

### 7.5 W5 소요

- 셰이더 조합 4종 토글 구현: 1h
- 벤치 촬영: 1.5h
- 평가: 1h
- 반영 + 문서: 1h

**총 ~4.5h**.

---

## 8. 완료 정의 + 다음 W 트리거

### 8.1 완료 정의

§4.1 체크리스트 전체 ✅.

### 8.2 커밋 전략

**Phase A/B (2026-05-25 완료, 브랜치 `feature/P6-W5`)**:
- `2d594e0` feat(gpu-lens): P6-W5 sclera veto 3-way 분기 + JNI/Java API (B1/B8 벤치)
- `e2f6b6a` feat(demo): P6-W5 4조합 A/B/C/D 블라인드 벤치 토글 UI
- `12f15a3` docs(bench): P6-W5 B1/B8 벤치 산출물 — 체크리스트/응답시트/촬영가이드/자동화 스크립트

**Phase C (벤치 후 — 미실행)**:
- `chore(bench): P6-W5 B1/B8 결과 report` (촬영/평가 집계)
- `feat(gpu-lens): P6-W5 B1/B8 결과 반영 — {Normal 유지|CRL 채택} / {color-veto|luma-only}`
  (calcScleraFactor 단일 수식 교체 + uScleraVetoMode 제거 + demo 토글 정리)

### 8.3 다음 W 트리거

W5는 독립적. 완료 후 바로 W6, W7로 넘어감 (이미 병렬 가능).
W9 통합 테스트 단계에서 B1/B8 결과 재검증.

### 8.4 W5 실패 시 전략

- 4 조합 모두 "차이 없음"이면 → 99 단순성 원칙 적용 (Normal + Gemini luma-only)
- 일부 조합만 문제 → 해당 항목 수식 미세 조정 후 재측정 (별도 W는 아님)

### 8.5 W5 성공 시 기대

- 블렌드 모드 최종 3종 or 4종 확정 → SDK 사용성 단순화
- sclera veto 단순화 가능성 → 저조도 안정성 향상
- 99 §1.1/§1.2 최종 확정 → Phase 6 마무리 단계 가시화

---

## 참조

- 99_final_decision.md §2 B1, B8
- 13_codex_r3.md §3 B1 SKU 지적
- 13_codex_r3.md §8 color-veto 수식
- 14_gemini_r3.md §8 luma-only 고수
- 07_claude_r2.md I8 (Claude 철회)
- 15_asset_analysis.md (SKU 선정 근거)
