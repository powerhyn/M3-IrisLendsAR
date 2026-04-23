# P6-W7: B4 림발 자동감지 fallback + 림발 정책 확정

> **상태**: 인사이트 작성 완료. 세부 계획 본문 작성 대기.
> **작성**: 2026-04-23
> **선행 의존**: P6-W3 (환경 반사 계층 스캐폴드)
> **병렬 가능**: P6-W4, W5, W6과 병렬. 가장 작은 W.

---

## 1. 인사이트 (세션 간 맥락 보존) ⭐

### 1.1 가장 많이 뒤집힌 쟁점

R1 → R2 → R3 → R4로 오면서 Claude 입장이 **3번 바뀐** 축:

| 라운드 | Claude 입장 |
|-------|-----------|
| R1 | 기본 OFF, opt-in. 텍스처에 대부분 림발 내장 가정 |
| R2 | 기본 OFF 철회, 기본 ON + SKU 메타 + 자동감지 fallback |
| R3 자기비판 | Codex "책임 떠넘김" 비판에 수용, R2 유지 |
| R4 에셋 실측 후 | **재검토 여지** — 20개 중 14개 림발 없음, Claude R1이 맞았을 수도 |

### 1.2 에셋 실측 — R4에서 뒤집힌 전제

**15_asset_analysis.md §2.2**:
- 림발 없음/미미: ~14개 (대다수)
- 약하게 베이크: ~5개 (로뮤_그레이토프, 디어멜로우, 러브글림, 엔비_플럼블랙, 오(OH)_베이글)
- 강한 블랙 아웃라인: 1개 (엔비_샤모 브라운 — 그래픽 디자인)

**결론**: R1의 "대부분 림발 내장" 전제가 **실측과 정반대**. 그렇다고 "Claude R1이 맞았다"고 단정도 못함 (16_product_crosscheck.md §0-1 보정 리스크).

### 1.3 16_product_crosscheck.md §3.4 — "재검토 여지"

**보수적 결론**:
> "C6를 '기본 OFF'로 바로 뒤집기보다, 실기기 벤치(B10 신규)에서 '셰이더 림발 ON / OFF' 양쪽 렌더링을 같은 SKU로 비교해 실측 데이터로 최종 결정."

그러나 99_final_decision.md에서 **B10 신규 벤치 제안은 따로 추가 안 됨**. B4 재설계로 "자동 감지 fallback 채택 여부"로 범위 축소됨 (99 §2 B4).

즉 W7 = B4 재설계 = **"자동 감지가 fallback으로 쓸만한 정확도인가"** 판정만.

### 1.4 Codex R3 §3 B4 재설계 주장

**Codex R3 §3 I4**:
> "벤치 이관 판단: 부분 동의. `림발 기본 ON + 메타데이터 off`는 벤치 없이 확정 가능하다. 자동감지는 fallback 후보로만 벤치하면 된다."

> "판정 강화: 정확도 10/10 → fallback 으로 채택 (드문 메타 누락 시 사용). 9/10 이하 → 자동 감지 드롭, 메타 only. 플래그 누락 텍스처는 경고 로그"

즉 **엄격 기준**: 10/10 아니면 드롭. Codex가 자동 감지의 false positive 리스크 우려가 크기 때문.

### 1.5 Gemini R3 §3 B4 불필요 주장

**Gemini R3 §3 I4**:
> "벤치 없이 확정 가능. 메타데이터 우선 원칙에 동의하되, 자동 감지는 '안전한 임계값' 내에서만 동작하도록 셰이더에 구현하면 됨. 굳이 벤치로 시간을 끌 필요 없음."

Gemini는 **벤치 자체가 불필요**하다 주장. Codex의 "10/10 기준"은 과도하다는 입장.

**Claude 의견**: 벤치 비용 1시간 정도니 하되, 기준은 Codex보다 완화하자.

### 1.6 B4 프로토타입 상세

**자동 감지 수식**:
```
텍스처 로딩 시 CPU 1회 측정:
  edge_lum = mean(pixels in r ∈ [0.85, 1.0])
  center_lum = mean(pixels in r < 0.3)
  ratio = edge_lum / max(center_lum, 0.001)
  baked_limbal_detected = (ratio < 0.75)  // edge가 center보다 어두우면 림발 있음
```

**변수**:
- ROI 경계 (r 구간) — 현재 [0.85, 1.0] vs [0.80, 1.0] 대안
- 임계값 — 0.75 vs 0.70 vs 0.80

### 1.7 B4 테스트 샘플

- 림발 내장 5개:
  - 로뮤_그레이 토프
  - 로뮤_디어 멜로우
  - 로뮤_러브 글림
  - 엔비_플럼 블랙
  - 오(OH)_베이글
- 림발 없음 5개:
  - 클라셋_돌 초코
  - 클라셋_런웨이 그레이
  - 클라셋_클라우드 그레이
  - 엔비_퍼퓸 글로우
  - 오(OH)_키위

정답 레이블은 15_asset_analysis.md §2.2 기준.

### 1.8 B4 판정

```
정확도 / 총 10개
10/10 (100%) → fallback 채택. 드문 메타 누락 시 사용
9/10 or 이하 → fallback 드롭. 메타 only. 플래그 누락은 "경고 로그"로 대응
```

**Gemini 관점의 완화안 (W7 브레인스토밍에서 제기 가능)**:
- 9/10도 채택 (false positive 1회는 허용). 임계값 튜닝으로 10/10 도달 가능.
- 7~8/10이면 드롭.

### 1.9 SKU 메타데이터 구조 (핵심)

**99 §1.2 C6 (R4 수정)**:
> "림발 기본 ON + SKU 메타데이터 플래그. 플래그는 내부 material 모델 필드(공개 `LensConfig`가 아닌 내부 표현)"

**⚠️ 공개 API 변경 없음**. 내부 메타데이터 구조 필요:

```cpp
struct LensSkuMetadata {
    std::string sku_id;
    bool has_baked_limbal = false;       // W7에서 20개 SKU 기본값 설정
    bool prefers_graphic_outline = false; // 엔비_샤모 브라운 같은 그래픽 SKU
    // ... 기타 메타
};
```

**제공 경로**: 에셋과 함께 메타데이터 파일 공급. 예: `{sku}.png` + `{sku}.meta.json`. 또는 공통 `lens_sku_metadata.json`.

### 1.10 `LensConfig` vs 내부 material 구분

**Codex R3 §4 C6 부분 동의**:
> "림발 기본 ON + `has_baked_limbal` 메타는 맞다. 다만 `LensConfig`에 bool을 추가하면 공개 API 변경 가능성이 있으므로 '내부 SKU 메타데이터'인지 '공개 config'인지 분리해야 한다."

**결정 (99 §1.2 C6)**: 내부 material 모델 필드. 공개 API 불변.

**구현**: 
- `LensConfig`는 그대로
- 내부에 `LensMaterial` 또는 `LensSkuRegistry` 같은 구조 신규
- SKU 로딩 시 메타데이터 조회 → material 셰이더 uniform 주입

### 1.11 셰이더 림발 수식 (S1에서 삭제된 것 부활? 아니면 새로?)

**S1에서 삭제**:
```glsl
const bool LIMBAL_ENABLED = false;
if (LIMBAL_ENABLED) {
    float limbal = smoothstep(0.7, 1.0, dist);
    blended = mix(blended, blended * 0.4, limbal * 0.8);
}
```

**W7에서 부활**:
```glsl
// uHasBakedLimbal = 0 → 셰이더 림발 활성
// uHasBakedLimbal = 1 → 셰이더 림발 비활성 (에셋이 이미 포함)
if (uHasBakedLimbal == 0) {
    float limbal = smoothstep(0.7, 1.0, dist);
    blended = mix(blended, blended * 0.4, limbal * 0.8);
}
```

수식 자체는 S1 전과 동일. 다만:
- `LIMBAL_ENABLED = const false` → `uHasBakedLimbal` uniform
- 플래그 의미: baked면 셰이더 림발 OFF (이중 적용 방지)

### 1.12 자동 감지 fallback이 채택된 경우

B4 결과: 정확도 10/10 (또는 완화 시 9/10)

```cpp
// CPU 측정 (텍스처 로딩 시 1회)
bool auto_detected_limbal = measure_limbal(texture);

// SKU 메타 우선, 없으면 자동감지 결과
bool has_baked_limbal = sku_meta.has_baked_limbal.value_or(auto_detected_limbal);
```

### 1.13 W7 브레인스토밍 시 질문

1. **임계값 조정 자유도**: B4에서 [0.85, 1.0] + 0.75 외 대안 값 시도 가능 (하이퍼파라미터 튜닝 포함)?
2. **Gemini 완화안 vs Codex 엄격안**: 9/10 허용 vs 10/10 엄격 — 실기기 한 번 돌려보고 평가자 판단?
3. **SKU 메타데이터 저장 방식**: JSON 파일 공급 vs 코드 내 하드코드 레지스트리 vs SKU 파일명 규약?
4. **엔비_샤모 브라운 같은 그래픽 SKU**: `prefers_graphic_outline` 플래그가 림발 셰이더 그대로 OFF 한다고 해서 충분한가? 별도 처리 필요?
5. **메타 누락 경고 로그**: 어느 레벨(DEBUG/WARN/ERROR)? 사용자 앱에 에러 팝업 뜨지 않게?
6. **B4 결과 드롭 시 폴백**: 메타 누락 SKU는 림발 기본 ON으로 렌더링? OFF로 렌더링? 경고만?

### 1.14 구현 범위 (W7)

**신규**:
- `cpp/include/iris_sdk/lens_sku_metadata.h` — 메타 구조체
- `cpp/src/lens_sku_metadata.cpp` — 레지스트리 + 로딩 로직
- (선택) `android/demo-app/src/main/assets/lens_meta/*.json` — SKU별 메타 파일

**수정**:
- `shader_sources.cpp` — 림발 수식 부활 + `uHasBakedLimbal` uniform
- `gpu_lens_renderer.cpp` — uniform 주입
- `gpu_lens_renderer.h` — uniform location

### 1.15 성능 예산

- 자동 감지: **CPU 1회만** (텍스처 로딩 시). 런타임 비용 0.
- 셰이더 림발: ALU 연산 2~3개. 이미 shader에 있던 것 부활.

### 1.16 B4 벤치 결과 시나리오별 W 완료

| 결과 | W7 완료 시 상태 |
|------|---------------|
| 10/10 | 자동감지 fallback 채택. 메타 있는 SKU는 메타 우선, 없으면 자동감지 |
| 9/10 (Gemini 완화안 채택) | 동일. 단 false positive 1회 경고 로그 |
| 7~8/10 | 자동감지 드롭. 메타 only. 누락 SKU는 "기본 ON" 폴백 |
| 6/10 이하 | 자동감지 완전 드롭 |

### 1.17 이 W의 상대적 중요도

**가장 작은 W**. 벤치 시간 1h + 구현 2~3h. 후속 W에 영향 미미.

하지만 **Claude 편향 가장 심했던 지점**이라 세션 간 맥락 보존이 중요. 새 세션에서 "왜 R1→R2→R3에서 Claude가 계속 바뀌었나"를 §1.1 히스토리 표로 바로 이해하게 설계.

---

## 2. 배경/맥락

### 2.1 가장 작은 W

벤치 시간 1h + 구현 2~3h. 하지만 **Claude 편향 가장 심했던 지점** (R1→R2→R3 세 번 뒤집힘). 세션 간 맥락 보존 중요.

### 2.2 실측이 R3 결정을 흔든 사례

- R3 합의: "림발 기본 ON + 메타데이터 + 자동감지 fallback"
- R4 실측: 20개 중 14개 림발 없음/미미 → 기본 ON 재검토 여지
- 16 웹 교차검증 (보정 리스크 명시): 업계 트렌드도 "얇은 써클라인" → Claude R1 원안(기본 OFF) 재부상

**W7 결론 도출 전 **실측 + 교차검증 한계** 모두 고려**.

### 2.3 W7이 해결하는 것

- **B4 자동감지 fallback 채택 여부** (정확도 기반)
- **SKU 메타데이터 구조** 확정
- **셰이더 림발 수식 부활** (S1에서 삭제된 것 재도입)
- **기본 ON vs OFF 최종 판정** (B10 같은 신규 벤치 논의 여지)

---

## 3. 전제 조건

1. ✅ W3 완료 (환경 반사 구조 — 림발이 반사 전/후 순서 명확)
2. ✅ 20 SKU 에셋 접근 + 분류 결과 (15_asset_analysis.md §2.2)
3. ✅ 5 + 5 테스트 SKU 선정됨

---

## 4. 목표

1. **B4 결과 확정** — 자동감지 fallback 채택/드롭/완화
2. **SKU 메타데이터 구조 정의** (`LensSkuMetadata`)
3. **셰이더 림발 수식 부활** (기본 OFF 또는 기본 ON 최종)
4. **20 SKU 메타 플래그 설정** (W7 산출물로 `lens_sku_metadata.json`)

### 4.1 Definition of Done

- [ ] B4 프로토타입 + 10 SKU 정확도 측정 완료
- [ ] SKU 메타 구조 구현 (header + 로딩 로직)
- [ ] 셰이더 림발 수식 부활 (uniform 스위치)
- [ ] 20 SKU 메타 플래그 초기 설정
- [ ] B4 결과에 따른 자동감지 채택/드롭 반영
- [ ] 99 §1.2 C6 + §2 B4 업데이트

---

## 5. 99에서 확정된 사항

### 5.1 메타데이터 구조 (내부, 공개 API 불변)

```cpp
struct LensSkuMetadata {
    std::string sku_id;
    bool has_baked_limbal = false;         // 기본 false (실측 반영)
    bool prefers_graphic_outline = false;  // 엔비_샤모 브라운 같은 그래픽
};
```

### 5.2 자동감지 수식 (B4 프로토타입)

```
edge_lum = mean(pixels in r ∈ [0.85, 1.0])
center_lum = mean(pixels in r < 0.3)
ratio = edge_lum / max(center_lum, 0.001)
baked_limbal_detected = (ratio < 0.75)
```

### 5.3 셰이더 림발 수식 (부활)

```glsl
// uHasBakedLimbal = 0 → 셰이더 림발 활성
// uHasBakedLimbal = 1 → 비활성 (에셋이 이미 포함)
if (uHasBakedLimbal == 0) {
    float limbal = smoothstep(0.7, 1.0, dist);
    blended = mix(blended, blended * 0.4, limbal * 0.8);
}
```

### 5.4 B4 판정 시나리오

- 10/10 정확도 → 자동감지 fallback 채택
- 9/10 (Gemini 완화안 수용 시) → 동일. false positive 1회 허용 경고 로그
- 7~8/10 → fallback only 한계. 메타 누락 시 "기본 ON" 폴백
- 6/10 이하 → 자동감지 완전 드롭

### 5.5 테스트 SKU (15 §2.2)

**림발 내장 5**: 로뮤_그레이 토프, 로뮤_디어 멜로우, 로뮤_러브 글림, 엔비_플럼 블랙, 오(OH)_베이글
**림발 없음 5**: 클라셋_돌 초코, 클라셋_런웨이 그레이, 클라셋_클라우드 그레이, 엔비_퍼퓸 글로우, 오(OH)_키위

---

## 6. 미결 사항

### 6.1 자동감지 ROI 경계 튜닝

`[0.85, 1.0]` vs `[0.80, 1.0]` 등. W7 브레인스토밍에서 여러 값 시도 권장.

### 6.2 임계값 `0.75` 튜닝

`0.70` (엄격) / `0.75` (기본) / `0.80` (느슨) 비교 가능.

### 6.3 메타데이터 저장 방식

- JSON 파일 (`android/demo-app/src/main/assets/lens_meta.json`)
- 코드 내 하드코드 레지스트리 (`LensSkuRegistry`)
- SKU 파일명 규약 (파일명에 `_bl` 접미사 = has_baked_limbal)

**Claude 추천**: JSON 파일. 외부 편집 용이.

### 6.4 메타 누락 경고 로그 레벨

DEBUG (조용) / WARN (알림) / ERROR (에러)?

**Claude 추천**: WARN. 앱은 안 깨지고 개발자 모니터링만.

### 6.5 Gemini 완화안 채택 여부

"9/10도 OK" vs "10/10 엄수". W7 브레인스토밍에서 결정.

### 6.6 엔비_샤모 브라운 특별 처리

`prefers_graphic_outline = true` 플래그 별도. 셰이더 림발 수식도 건드리지 않음 (에셋 자체가 강한 아웃라인).

**확인**: 이 플래그의 셰이더 효과 명확화.

### 6.7 B10 신규 벤치 필요성

16_product_crosscheck.md에서 제안한 "같은 SKU로 셰이더 림발 ON/OFF 비교" — W7에서 추가 수행?

**Claude 제안**: B4 결과 후 판정. 자동감지가 채택되면 B10 불필요. 드롭되면 B10으로 "기본 ON/OFF" 재확인 가능.

---

## 7. W7 브레인스토밍 시작 체크리스트

### 7.1 읽을 파일

**필수**: P6-W0, P6-W7, 99 §1.2 C6 + §2 B4, 15_asset_analysis.md §2.2, 16_product_crosscheck.md
**선택**: 13_codex_r3.md §3 I4 (엄격 기준), 14_gemini_r3.md §3 I4 (벤치 불필요 주장)

### 7.2 송신 프롬프트

```
@docs/workPaper/P6-W7_limbal_policy.md 읽고, 섹션 6 미결 7개에 대해
입장 정리. docs/workPaper/P6-W7_brainstorm/{codex|gemini}_w7.md.

특히:
- 6.5 9/10 vs 10/10 엄수
- 6.3 JSON vs 하드코드 레지스트리
- 6.7 B10 신규 벤치 추가 여부

규칙: 새 쟁점 금지. R1~R3 Claude 편향 3번 뒤집힘 맥락 인식.
```

### 7.3 예상 대립

- 6.5 Gemini 9/10 완화안 vs Codex 10/10 엄수
- 6.7 B10 벤치 — Codex는 "필요 없음", Gemini는 "이미 B4로 충분"

### 7.4 소요

- 자동감지 CPU 구현 + 10 SKU 측정: 1h
- SKU 메타 구조 + JSON 로딩: 1h
- 셰이더 림발 수식 부활 + uniform: 30min
- B4 평가 + 결과 반영: 30min

**총 3h**.

---

## 8. 완료 정의 + 다음 W 트리거

### 8.1 완료 정의

§4.1 체크리스트.

### 8.2 커밋

- `docs(P6-W7): 섹션 2~8`
- `feat(sdk): P6-W7 LensSkuMetadata 구조 + JSON 로딩`
- `feat(gpu-lens): P6-W7 림발 셰이더 수식 부활 + uHasBakedLimbal uniform`
- `chore(bench): P6-W7 B4 결과 report + 20 SKU 메타 플래그 초기 설정`

### 8.3 다음 W

W9 통합 테스트. W7은 독립.

### 8.4 W7 기대

- 림발 처리 정책 최종 확정 → Claude 3번 뒤집기 종결
- SKU 메타 구조 확립 → Phase 7+에서 활용 가능 (향후 SKU별 특수 설정 가능)

---

## 참조

- 99_final_decision.md §1.2 C6, §2 B4
- 15_asset_analysis.md §2.2 (에셋 림발 분포 실측)
- 16_product_crosscheck.md §3 (C6 교정 논의)
- 13_codex_r3.md §3 I4 (엄격 기준 주장)
- 14_gemini_r3.md §3 I4 (벤치 불필요 주장)
- 07_claude_r2.md I4 (Claude R1 철회 히스토리)
- S1 커밋 `9aee86d` — LIMBAL_ENABLED 제거 주석
