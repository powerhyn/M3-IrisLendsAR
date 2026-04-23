# P6-W5: B1 블렌드 4번째 슬롯 + B8 sclera veto 벤치

> **상태**: 인사이트 작성 완료. 세부 계획 본문 작성 대기.
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
_TODO_

## 3. 전제 조건
_TODO_

## 4. 목표
_TODO_

## 5. 99에서 확정된 사항
_TODO_

## 6. 미결 사항
_TODO_

## 7. W 브레인스토밍 시작 체크리스트
_TODO_

## 8. 완료 정의 + 다음 W 트리거
_TODO_

---

## 참조

- 99_final_decision.md §2 B1, B8
- 13_codex_r3.md §3 B1 SKU 지적
- 13_codex_r3.md §8 color-veto 수식
- 14_gemini_r3.md §8 luma-only 고수
- 07_claude_r2.md I8 (Claude 철회)
- 15_asset_analysis.md (SKU 선정 근거)
