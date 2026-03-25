# 뷰티 스무딩 방향 토론: 매끈하게 & 모공

> **문서 구성** — 본 문서의 초안과 1차 결론은 Claude 의견으로 작성되었고, 하단에 Codex 보완 의견이 추가되었습니다. B612 내부 구현은 비공개이며, 아래 내용은 공개 결과물의 시각적 특성과 업계 알고리즘 특성을 기반으로 한 추론입니다.

---

## 1. 배경

B612 앱의 뷰티 보정 중 **매끈하게**와 **모공** 기능을 IrisLensSDK에 구현하기 위한 방향 논의.

### 참고 이미지

| 이미지 | 설명 |
|--------|------|
| `b612_filter_none.jpg` | B612 필터 없음 (원본) |
| `b612_filter_매끈하게.jpg` | B612 매끈하게 적용 |
| `b612_filter_모공.jpg` | B612 모공 적용 |

---

## 2. B612 결과물 시각적 분석 (CLAUDE 의견)

### 매끈하게 (vs 원본)

- 피부 톤이 균일해짐 — 볼, 코 주변 붉은기/톤 차이가 상당히 감소
- 피부 결 자체가 밀림 — 모공뿐 아니라 피부 텍스처 전반이 부드러워짐
- 에지 보존 우수 — 안경테, 눈썹, 머리카락 경계가 선명하게 유지
- 수염/잔털은 약간 남아있음 — 완전히 제거하지 않음
- halo 아티팩트 없음 — 안경테 주변에 번짐 없음

### 모공 (vs 원본)

- 매끈하게보다 훨씬 미묘한 변화
- 모공/잔결만 선택적으로 줄어듦 — 피부의 큰 구조(그림자, 윤곽)는 유지
- 톤 균일화는 거의 없음 — 붉은기/색차는 원본과 비슷하게 유지
- 더 자연스러움 — "보정했다"는 느낌이 적음

---

## 3. 알고리즘 방향 추론 (CLAUDE 의견)

### 매끈하게 → Guided Filter 계열로 추정

**근거:**
- 강한 스무딩에도 에지 보존이 우수하고 halo가 없음
- Bilateral Filter였다면 안경테 근처에 gradient reversal/halo가 생겼을 가능성이 높음
- 톤 균일화까지 동시에 이루어지는 것은 큰 커널 Guided Filter의 특성과 일치
- Guided Filter는 커널 크기에 무관한 O(1) 복잡도로, 강한 스무딩에서도 성능 저하가 없음

**Guided Filter 특성:**
- `r` (window radius): 커질수록 더 넓은 영역을 스무딩
- `ε` (regularization): 작을수록 더 강한 스무딩, 클수록 디테일 보존
- 매끈하게는 r 크고 ε 작은 설정에 해당

### 모공 → Frequency Separation 계열로 추정

**근거:**
- 톤은 건드리지 않고 텍스처만 줄이는 것은 FreqSep의 전형적 동작
- 피부의 큰 구조는 유지하면서 미세 반복 텍스처(모공)만 선택적으로 압축
- low = blur(original), high = original - low 분리 후, high 중 미세 밴드만 약화하고 재합성

---

## 4. 현재 IrisLensSDK 상태

| 기능 | 현재 구현 | 방식 | 상태 |
|------|-----------|------|------|
| `smoothing` | Bilateral Filter 단일 패스 | 에지 보존 블러 | 프리셋에서 0.0으로 비활성 |
| `skinQuality` | Frequency Separation 6패스 | 미세 텍스처 압축 | 활성 사용 중 (0.2~0.4) |

### B612과의 매핑

| B612 기능 | IrisLensSDK 대응 | 평가 |
|-----------|-------------------|------|
| **매끈하게** | `smoothing` (Bilateral) | 방향은 맞지만 **품질 부족** — Bilateral은 강한 스무딩 시 halo/성능 문제 |
| **모공** | `skinQuality` (FreqSep) | **방향 일치** — 파라미터 튜닝으로 근접 가능 |

---

## 5. 제안 방향 (CLAUDE 의견)

### 매끈하게: Guided Filter 도입

현재 Bilateral 기반 `smoothing`을 **Guided Filter 셰이더로 교체 또는 별도 패스 추가**.

**이유:**
- Bilateral 대비 강한 스무딩에서도 halo 아티팩트 없음
- O(1) 복잡도 — 커널 크기와 무관한 성능 (모바일 실시간에 유리)
- 톤 균일화 효과가 자연스럽게 포함됨
- CPU 경로에 이미 `FastGuidedFilter` 구현이 존재 (`cpp/src/fast_guided_filter.cpp`) — GPU 셰이더 포팅 가능

**구현 선택지:**

| 옵션 | 설명 | 장점 | 단점 |
|------|------|------|------|
| A. Bilateral → Guided Filter 교체 | 기존 `smoothing` 패스를 Guided Filter로 변경 | 코드 변경 최소, 기존 파라미터 호환 | 기존 동작 변경됨 |
| B. Guided Filter 패스 별도 추가 | `smoothing`은 유지하고 새 파라미터 추가 | 기존 영향 없음, 독립 제어 | 파이프라인 패스 증가 |
| C. skinQuality 내부에 통합 | FreqSep 파이프라인의 blur 단계를 Guided Filter로 교체 | 패스 수 동일, 매끈+모공 동시 개선 | FreqSep 튜닝 전체 재조정 필요 |

### 모공: skinQuality 파라미터 튜닝

현재 구현의 방향 자체는 맞으므로, **파라미터 조정으로 B612 수준에 근접 가능**.

**조정 포인트:**
- `highFreqPreserve` 값을 좀 더 낮춰서 모공 압축 강화
- `microTextureBand` 범위를 넓혀서 더 넓은 모공 크기 대응
- 실기기에서 B612 모공 결과와 비교하며 반복 튜닝

---

## 6. 확인이 필요한 사항

- [ ] B612 매끈하게 + 모공 동시 적용 시 결과물 확인 (상호작용 파악)
- [ ] 매끈하게 강도별 스크린샷 (50%, 100% 등) 비교
- [ ] 우리 SDK의 Bilateral `smoothing`을 강하게 올렸을 때 실기기 결과 확인 (현재 기준선 파악)
- [ ] Guided Filter GPU 셰이더 프로토타입 구현 후 실기기 성능 측정
- [ ] skinQuality 파라미터 튜닝 후 B612 모공 결과와 나란히 비교

---

## 7. 확정 사실 vs 추론 구분

| 항목 | 구분 | 근거 |
|------|------|------|
| B612에 매끈하게/모공 기능이 분리되어 있음 | **확정** | 앱 UI에서 직접 확인 |
| 매끈하게는 에지 보존 + 톤 균일화 + halo 없음 | **확정** | 스크린샷에서 관찰 |
| 모공은 톤 안 건드리고 텍스처만 줄임 | **확정** | 스크린샷에서 관찰 |
| 매끈하게가 Guided Filter 계열 | **CLAUDE 추론** | 시각적 특성 + 알고리즘 특성 매칭 |
| 모공이 Frequency Separation 계열 | **CLAUDE 추론** | 시각적 특성 + 알고리즘 특성 매칭 |
| B612의 실제 내부 구현 | **비공개/불명** | 소스 비공개, 리버스 엔지니어링 미수행 |
| 우리 `skinQuality`가 모공에 가까움 | **확정** | 코드 분석 (FreqSep composite 셰이더) |
| 우리 `smoothing`이 매끈하게에 부족함 | **확정** | 코드 분석 (Bilateral, 프리셋에서 비활성) |

---

## 8. Codex 의견

### 핵심 판단

- 현재 IrisLensSDK의 `skinQuality`는 B612의 **매끈하게**보다 **모공**에 더 가깝다.
- 현재 `smoothing`은 개념상 B612의 **매끈하게**에 대응시키기 쉽지만, 실제 구현 품질은 그 역할을 맡기기 어렵다.
- 제품 관점에서는 `smoothing`과 `skinQuality`를 각각 노출하기보다, **매끈하게**와 **모공**을 별도 UX 파라미터로 재정의하는 편이 맞다.

### Claude 의견에 대한 동의

- **모공 → Frequency Separation 계열**이라는 해석에는 동의한다.
- 현재 GPU `skinQuality`는 미세 텍스처 압축, 큰 디테일 보호, 후단 샤픈 복구까지 포함하므로 "모공/잔결 축"으로 보는 것이 가장 자연스럽다.
- 따라서 B612의 모공 대응 기능을 만들 때는 레거시 `smoothing`이 아니라 현재 `skinQuality` 계열을 기준으로 잡아야 한다.

### Claude 의견에 대한 보완

**매끈하게를 Guided Filter 단일 계열로 단정하기는 아직 이르다.**

스크린샷 기준으로 B612의 매끈하게는 단순한 "강한 에지 보존 스무딩"보다는 아래가 합쳐진 결과에 더 가깝다.

- 피부 저주파 정리: 볼, 코 옆, 인중, 턱의 톤 차이를 완만하게 정리
- 미세 텍스처 압축: 모공과 잔결도 함께 약화
- 강한 보호 마스킹: 안경테, 눈썹, 헤어라인, 입술 경계 유지
- 후단 finish: 얼굴 전체가 한 번 더 정돈된 듯한 피부 표현

즉 B612의 매끈하게는 **large-radius edge-aware smoothing 1개**라기보다,  
**low-frequency skin finish + mild pore compression + skin mask + edge protection**의 복합 동작일 가능성이 더 높다.

Guided Filter는 이 중 한 축으로는 유력하지만, **그 자체만으로 B612의 매끈하게를 재현한다고 보긴 어렵다.**

### IrisLensSDK 매핑에 대한 Codex 결론

| 제품 기능 관점 | 현재 SDK에서 가장 가까운 축 | 판단 |
|----------------|-----------------------------|------|
| **매끈하게** | `smoothing`의 의도 + 일부 `skinQuality`의 finish 성분 | 현재는 대응 기능이 분산되어 있음 |
| **모공** | `skinQuality` | 가장 직접적으로 대응됨 |

정리하면:

- 현재 `skinQuality`는 **모공 중심**
- 현재 `smoothing`은 **매끈하게의 낡은/단순한 버전**
- B612의 매끈하게는 지금 SDK에서 **단일 파라미터 하나로 매핑되지 않는다**

### 구현 방향 제안

#### 1. 레거시 `smoothing`을 제품 기준선으로 삼지 않기

현재 `smoothing`은 Bilateral 기반이라 강도를 올릴수록 blur 인상이 강해지고, B612류의 "정돈된 피부 finish"와는 결과 질감이 다르다.  
따라서 제품 슬라이더 이름을 `smoothing`에 맞추기보다, 제품 의미를 먼저 정의하고 내부 파이프라인을 재배치하는 편이 맞다.

#### 2. `skinQuality` 기반을 중심으로 2축 분리

가장 현실적인 방향은 FreqSep 파이프라인을 중심으로 아래 두 축을 분리하는 것이다.

- **매끈하게**
  - low-frequency smoothing 강도
  - tone/foundation finish 강도
  - texture blend floor
  - sharpen 감소 또는 중립 유지
- **모공**
  - micro-texture band 압축 강도
  - `high_freq_preserve` 감소 폭
  - `attenuation` 밴드 폭

즉 매끈하게는 "피부 전체 finish", 모공은 "미세 반복 텍스처 억제"로 역할을 나누는 것이 자연스럽다.

#### 3. Guided Filter 도입은 가능하지만 위치를 명확히 해야 함

Guided Filter를 넣는다면 목적은 **모공**이 아니라 **매끈하게의 저주파 정리**여야 한다.

권장 위치:

- FreqSep 이전/내부의 low-frequency base 생성 보조
- 또는 별도의 skin finish pass

비권장 위치:

- 현재 `skinQuality`를 통째로 Guided Filter로 대체

이유는 `skinQuality`는 이미 모공/잔결 억제에 맞는 구조를 갖고 있기 때문이다. 이 축을 바꾸면 오히려 B612의 모공 대응력을 잃을 수 있다.

#### 4. 비교 기준은 GPU texture 경로로 고정

현재 SDK는 호출 경로에 따라 뷰티 품질 수준이 다르다.  
특히 Android GPU texture 경로는 `skinQuality` 기반 FreqSep를 사용하지만, 일부 일반 V2 API/CPU 경로는 아직 레거시 뷰티 로직 비중이 남아 있다.

따라서 B612 비교, 파라미터 튜닝, UX 설계는 우선 **GPU texture 경로를 기준 구현**으로 삼는 것이 맞다.

### 최종 의견

Codex 관점에서 가장 중요한 포인트는 다음이다.

- B612 **모공**과 우리 `skinQuality`는 같은 축이다.
- B612 **매끈하게**는 현재 우리 SDK에 대응 축이 분산되어 있으며, 단순히 `smoothing` 하나로 보면 안 된다.
- 따라서 다음 단계의 올바른 제품 설계는  
  **`skinQuality`를 모공 축으로 인정하고**,  
  **매끈하게용 저주파 skin finish 축을 별도로 설계하는 것**이다.

---

## 9. Gemini 의견

### 핵심 분석: "면(Area)의 정리" vs "점(Point)의 채움"

B612의 결과물을 시각적/기술적으로 분석했을 때, 두 기능의 핵심 차이를 다음과 같이 정의한다.

- **매끈하게 (Area Smoothing):** 피부의 매크로(Macro) 레벨을 다룬다. 단순히 블러를 먹이는 것이 아니라, **얼굴의 입체감(Low Frequency)은 유지하되 피부톤의 불균형과 큰 굴곡을 평탄화**하 는 작업이다. (Guided Filter의 Wide Radius 영역)
- **모공 (Point Filling):** 피부의 마이크로(Micro) 레벨을 다룬다. 피부 질감(Texture)은 최대한 보존하면서, **모공과 같은 '어두운 입자(Dark High Frequency)'만 골라내어 주변 색상으로 메우는(Filling)** 정밀한 픽셀 연산이다. (Frequency Separation의 Selective High-Pass 영역)

### R&D 전략: 독립적 최적화 후 지능적 통합 (Decoupling then Integration)

현재 SDK의 `skinQuality`는 이 두 기능이 하이브리드로 섞여 있어 최적의 값을 찾기 어려운 구조이다. 사용자(개발자)의 제안대로 **두 축을 완전히 분리하여 독립적으로 튜닝**하는 과정이 선행되어야 한다.

1.  **파라미터 분리 제어 (Decoupling):**
    - 테스트용 API를 통해 `smooth_intensity` (저주파 반경 및 톤 정리)와 `pore_reduction` (고주파 모공 억제)을 각각 독립적으로 조절한다.
    - **Smoothing 최적화:** 모공을 전혀 건드리지 않은 상태에서, 안경이나 눈썹 경계가 무너지지 않는 최대 스무딩 반경(Radius)과 강도를 찾는다.
    - **Pore 최적화:** 스무딩을 전혀 하지 않은 상태에서, 피부가 플라스틱처럼 변하지 않으면서 모공만 사라지는 주파수 대역(Attenuation Band)과 억제 강도를 찾는다.

2.  **인지적 매핑 기반의 재통합 (Smart Hybrid Mapping):**
    - 각 축의 최적값을 찾은 후, 최종 제품용 `skin_quality` 파라미터 하나로 합칠 때 **비선형 매핑(Non-linear Mapping)**을 적용한다.
    - 예: `skin_quality` 저구간(0.1~0.4)에서는 모공(Pore) 보정 위주로 자연스러움을 확보하고, 고구간(0.5~1.0)으로 갈수록 매끈함(Smooth)의 비중을 높여 드라마틱한 효과를 연출한다.

### 기술적 구현 제언

- **Frequency Separation 구조 유지:** 현재 SDK의 FreqSep 구조는 이미 매우 우수하므로, 이를 버리고 Guided Filter로 교체하기보다는 **FreqSep 내부의 Low-Pass Pass를 Guided Filter로 강화**하는 방향이 성능과 품질 면에서 가장 유리하다.
- **Dynamic Radius:** 현재 구현된 `face_width` 기반의 반경 조절 로직은 B612 수준의 하이엔드 최적화이므로 이를 반드시 유지 및 고도화해야 한다.

---

## 10. 수렴 결론 (Claude 2차 — Codex/Gemini 의견 반영)

### Claude 초안 수정 사항

1. **매끈하게 = Guided Filter 단일 계열이라는 단정을 철회합니다.** Codex 지적대로, B612 스크린샷에서 관찰되는 결과는 톤 균일화 + 텍스처 전반 약화 + skin finish가 복합적으로 일어나고 있어 단일 필터보다 복합 파이프라인일 가능성이 높습니다.
2. **Guided Filter의 위치를 재조정합니다.** 별도 패스 추가보다 Gemini가 제안한 FreqSep 내부 Low-Pass 단계 강화가 패스 수 동일 + 품질 향상으로 더 실용적입니다.
3. **제품 UX 재정의에 동의합니다.** Codex 제안대로 내부 파라미터명(`smoothing`, `skinQuality`)을 그대로 노출하지 않고, B612처럼 제품 축(매끈하게/모공)으로 매핑하는 것이 맞습니다.

### 3자 합의 방향

#### 아키텍처: FreqSep 파이프라인 유지 + 내부 2축 분리

```
FreqSep Pipeline (기존 6패스 구조 유지)
│
├─ [매끈하게 축] Low-Pass 단계
│   ├─ Gaussian → Guided Filter로 교체/강화
│   ├─ tone finish 강도 제어
│   └─ sharpen 감소 또는 중립
│
├─ [모공 축] High-Freq Composite 단계
│   ├─ microTextureBand 압축 강도
│   ├─ high_freq_preserve 감소 폭
│   └─ attenuation 밴드 폭
│
└─ 공통: skin mask + edge/chroma protection
```

#### 제품 파라미터 설계

| 제품 슬라이더 | 내부 제어 대상 | 범위 |
|---------------|---------------|------|
| **매끈하게** | Low-Pass radius/ε, tone finish, texture blend floor | 0.0 ~ 1.0 |
| **모공** | High-Freq preserve, attenuation band, micro-texture compression | 0.0 ~ 1.0 |

- 두 슬라이더는 **독립 제어** (B612과 동일한 UX 구조)
- 내부적으로는 하나의 FreqSep 파이프라인을 공유하되, 각 축이 다른 단계를 제어

#### 구현 우선순위

| 순서 | 작업 | 이유 |
|------|------|------|
| 1 | FreqSep Low-Pass를 Guided Filter로 교체 (GPU 셰이더) | 매끈하게의 핵심 품질 개선. CPU에 이미 `FastGuidedFilter` 존재 |
| 2 | 매끈하게/모공 2축 파라미터 분리 | `mapSkinQuality()` 내부에서 두 축 독립 매핑 |
| 3 | 실기기 튜닝 — B612 결과물과 나란히 비교 | 파라미터 최적값 탐색 |
| 4 | 데모앱 UI 반영 — 매끈하게/모공 슬라이더 분리 | 제품 UX 확정 |

### 남은 확인 사항

- [ ] FreqSep의 Gaussian → Guided Filter 교체 시 기존 모공 품질에 영향 없는지 검증
- [ ] Guided Filter GPU 셰이더의 모바일 성능 벤치마크 (패스 수 동일해도 연산량 차이 가능)
- [ ] 매끈하게 + 모공 동시 최대 시 과보정 방지 로직 필요 여부
- [ ] B612 매끈하게 강도별(50%, 100%) 추가 스크린샷 비교

---

## 11. 2차 피드백

### Codex 2차 피드백

수렴 결론에 대해 **부분 동의**.

**타당한 정리로 인정하는 부분:**
- `skinQuality`를 B612 모공 축에 가깝게 보는 판단
- 현재 `smoothing`만으로는 B612 매끈하게를 설명하기 어렵다는 판단
- 제품 UX를 매끈하게/모공 2축으로 재정의해야 한다는 정리
- FreqSep를 버리기보다 그 위에서 분리 설계하는 게 현실적이라는 정리

**수렴 결론으로 확정하기엔 이른 부분:**
- FreqSep Low-Pass를 Guided Filter로 교체를 사실상 채택안처럼 쓴 부분
- "패스 수 동일 + 품질 향상으로 더 실용적"이라는 표현
- 구현 우선순위 1번에 Guided Filter 교체를 바로 둔 부분

**이유:** 성능, 패스 구조, 기존 FreqSep와의 상호작용이 아직 검증되지 않았다. GPU에서 Guided Filter를 넣는 건 "유력 후보"이지 "1차 수렴결론"이라고 부를 수준의 검증은 없다.

**Codex 제안 — 다음 토론에서 다룰 3가지:**
1. 매끈하게가 제어해야 할 내부 신호가 무엇인지
2. 모공이 제어해야 할 내부 신호가 무엇인지
3. 둘을 FreqSep 내부에서 해결할지, 별도 finish pass로 뺄지

### Gemini 2차 피드백

수렴 결론에 대해 **강한 동의**.

- 매끈하게 = 면(Area)/Guided Filter, 모공 = 점(Point)/FreqSep 매핑은 하이엔드 앱의 표준 방식과 "완벽하게 일치"
- 현재 SDK의 결점(Bilateral 기반 smoothing의 한계)을 정확히 타격
- 이대로 진행하면 B612의 질감 차이를 잡아낼 "가장 빠른 길"

---

## 12. Claude 최종 재수정 (3차)

### Codex vs Gemini 피드백에 대한 판단

**Codex 지적이 더 정확합니다.**

Guided Filter를 확정안처럼 올린 건 과잉이었습니다. Gemini가 그 부분까지 포함해서 "완벽하게 일치", "확신합니다"라고 한 것은 비공개 구현에 대한 추론에 대해 과한 긍정입니다. 이는 제가 초 안에서 B612 구현을 단정적으로 말했다가 지적받은 것과 같은 패턴입니다.

### 확정 사항 (토론 기준선으로 사용 가능)

| 항목 | 상태 |
|------|------|
| `skinQuality` ≈ B612 모공 축 | **확정** |
| 현재 `smoothing`(Bilateral)은 매끈하게에 부족 | **확정** |
| 제품 UX를 매끈하게/모공 2축으로 재정의 | **확정** |
| FreqSep 구조 유지 + 그 위에서 설계 | **확정** |

### 가설로 격하 (검증 필요)

| 항목 | 상태 | 이유 |
|------|------|------|
| Low-Pass를 Guided Filter로 교체 | **유력 가설** | GPU 성능, FreqSep 상호작용 미검증 |
| 패스 수 동일 + 품질 향상 | **미검증 주장** | Guided Filter의 GPU 연산량은 Gaussian과 다를 수 있음 |
| 구현 우선순위 1~4 | **삭제** | 해법 확정 전에 순서를 정하는 건 성급 |

### 다음 토론 주제 (Codex 제안 채택)

수렴 결론의 "구현 우선순위"를 삭제하고, 대신 아래 3개를 다음 토론 주제로 설정:

1. **매끈하게가 제어해야 할 내부 신호 정의** — 저주파 톤 차이? 중주파 텍스처? 피부 finish? 구체적으로 어떤 신호를 눌러야 B612 매끈하게 결과가 나오는지
2. **모공이 제어해야 할 내부 신호 정의** — 현재 FreqSep의 microTextureBand가 모공을 정확히 잡고 있는지, 밴드 범위가 맞는지
3. **구조 결정: FreqSep 내부 vs 별도 finish pass** — 매끈하게를 FreqSep 안에서 해결할지, 별도 패스로 뺄지. 각각의 성능/품질 트레이드오프

---

## 13. 2차 토론: 내부 신호 정의와 구조 결정 (CLAUDE 의견)

> 코드 분석 근거: `shader_sources.cpp` (Composite 셰이더), `gpu_beauty_backend.cpp` (`mapSkinQuality()`, 6패스 구조)

### 주제 1: 매끈하게가 제어해야 할 내부 신호

B612 스크린샷에서 관찰된 매끈하게의 효과는 3가지이며, **모두 현재 FreqSep 파이프라인 안에 이미 존재하지만 독립 제어가 안 되고 있습니다.**

#### A. 톤 균일화 (저주파 정리)

- **관찰**: 볼/코/인중/턱의 붉은기, 명도 차이가 줄어듦
- **현재 위치**: Pass 2의 `smoothedLow` — 저주파 베이스를 추가 Gaussian으로 한 번 더 블러
- **현재 제어**: `low_freq_smooth_radius_ratio` (0.22~0.32)로 간접 제어, **독립 파라미터 없음**
- **매끈하게가 할 일**: 이 ratio를 독립적으로 올려서 더 넓은 톤 균일화

#### B. 텍스처 전반 약화 (블렌딩 floor)

- **관찰**: 모공뿐 아니라 피부 결 전체가 부드러워짐
- **현재 위치**: Composite 셰이더의 `textureBlend = mask * mix(0.38, 0.88, compression)`
- **0.38의 의미**: compression이 0이어도 최소 38%는 foundation 쪽으로 블렌딩됨
- **매끈하게가 할 일**: 이 **floor 값(0.38)을 높여서** 원본 대비 foundation 비중 증가

#### C. Foundation Finish (화장감)

- **관찰**: 얼굴 전체가 한 번 더 정돈된 듯한 피부 표현
- **현재 위치**: Composite 셰이더의 `toneFinish = textureFinished + (toneLift × effectStrength) × textureFinished × (1 - textureFinished)`
- **toneLift**: 0.020~0.050, 미드톤 리프트로 명도 범위 압축 → 화장감 연출
- **매끈하게가 할 일**: toneLift를 독립적으로 제어

#### 매끈하게 제어 대상 요약

| 신호 | 현재 파이프라인 위치 | 매끈하게 슬라이더가 제어할 것 |
|------|---------------------|---------------------------|
| 저주파 톤 차이 | Pass 2 radius (`low_freq_smooth_radius_ratio`) | ratio 확대 → 더 넓은 톤 균일화 |
| 텍스처 블렌딩 floor | Composite `mix(0.38, 0.88, ...)` | 0.38 → 더 높은 값 → 원본 비중 감소 |
| 미드톤 리프트 | Composite `toneLift` | 독립적 증가 → foundation 느낌 강화 |

---

### 주제 2: 모공이 제어해야 할 내부 신호

현재 FreqSep의 모공 제어는 **이미 잘 설계되어 있으며, 파라미터 범위 확장과 튜닝으로 B612 수준에 근접 가능**하다고 판단합니다.

#### 핵심 신호 3개

**A. microTextureBand (모공 대역 선택)**
```
smoothstep(attLow × 0.55, attHigh × 1.35, magnitude)
```
- 현재 범위: magnitude 0.003 ~ 0.035
- 실제 모공 신호: magnitude ≈ 0.015 ~ 0.05
- **불확실점**: 상한(0.035)이 큰 모공을 놓치고 있을 수 있음

**B. high_freq_preserve (압축 강도)**
- 현재: 0.72 ~ 0.30 (skinQuality에 따라)
- 낮을수록 모공이 더 강하게 눌림

**C. largeDetailProtection (큰 디테일 보호)**
- magnitude > 0.08부터 보호 시작
- 점, 흉터, 털은 건드리지 않도록

#### 모공 제어 대상 요약

| 신호 | 현재 상태 | 모공 슬라이더가 제어할 것 |
|------|----------|-------------------------|
| `attenuation_high` | 0.016 ~ 0.026 | 상한 확대 → 더 큰 모공까지 포함 |
| `high_freq_preserve` | 0.72 ~ 0.30 | 더 낮은 값 허용 → 강한 압축 |
| `largeDetailProtection` 임계값 | attHigh × 4.8 | **유지** (보호 건드리면 안 됨) |

#### 판단

모공 축은 **새 알고리즘 없이 파라미터 범위 확장 + 실기기 튜닝**으로 충분합니다. 현재 FreqSep의 3-Signal 분석(magnitude, edge, chroma) 구조 자체가 이미 모공 선택적 압축에 적합하게 설계 되어 있습니다.

---

### 주제 3: FreqSep 내부 vs 별도 finish pass

**Claude 의견: FreqSep 내부에서 해결하는 것을 우선 시도해야 합니다.**

#### FreqSep 내부 처리 근거

**1. 매끈하게의 3가지 신호가 이미 FreqSep 안에 있음**
- 톤 균일화 → Pass 2 (smoothedLow)
- 텍스처 블렌딩 → Pass 3 Composite의 `textureBlend`
- Foundation finish → Pass 3 Composite의 `toneFinish`
- 별도 pass를 만들면 이 신호들을 중복 계산해야 함

**2. 패스 추가 = GPU 비용 증가**
- 현재 6패스 (blur 4 + composite 1 + sharpen 1)
- 별도 finish pass 추가 시 7~8패스
- MID tier 디바이스(Adreno 6xx, Mali-G7x)에서 30fps 유지 위험

**3. 분리 설계는 파라미터 레벨에서 가능**

`mapSkinQuality()`를 2개 함수로 분리:

```
mapSmoothing(smooth_intensity) → {
    low_freq_smooth_radius_ratio   // 톤 균일화 강도
    texture_blend_floor            // 새 파라미터: 최소 블렌딩 비율
    tone_lift                      // foundation finish 강도
}

mapPoreReduction(pore_intensity) → {
    attenuation_high               // 모공 대역 상한
    high_freq_preserve             // 고주파 압축 강도
    sharpen_amount                 // 후단 샤프닝
}

공통 (두 축의 max 또는 가중평균):
    blur_radius                    // 저주파 추출 반경
    edge_weight, chroma_weight     // 보호 마스킹
```

#### 불확실점 (검증 필요)

이것은 **가설**입니다. 실기기 검증이 필요한 부분:

- [ ] `texture_blend_floor`를 올렸을 때 모공 축과 간섭이 생기는지
- [ ] Pass 2의 radius만으로 B612 수준의 톤 균일화가 되는지 (안 되면 Guided Filter 또는 별도 pass 재검토)
- [ ] 매끈하게 최대 + 모공 최대 시 과보정(플라스틱 느낌) 발생 여부
- [ ] `blur_radius`를 두 축이 공유할 때 적절한 결합 공식 (max? 가중평균?)

#### 판단 흐름

```
1차: FreqSep 내부 파라미터 분리로 시도
  ↓
실기기 테스트: B612 매끈하게와 비교
  ↓
톤 균일화가 충분한가?
  ├─ YES → FreqSep 내부로 확정
  └─ NO → 2차 검토: Pass 2의 Gaussian을 Guided Filter로 교체
            ↓
          그래도 부족한가?
            ├─ YES → 별도 finish pass 추가
            └─ NO → Guided Filter로 확정
```

---

## 14. Codex 의견 (3차)

이번 2차 토론 정리는 지금까지 나온 내용 중 가장 실무적으로 유효한 방향입니다.  
특히 `매끈하게`와 `모공`을 제품 슬라이더 이름이 아니라 **내부 신호 단위로 분해해서 본 것**은 맞습니다.

### 동의하는 부분

- `매끈하게`는 단일 필터명이 아니라
  - 저주파 톤 정리
  - 최소 texture blend 비율
  - tone finish
  의 조합으로 보는 것이 타당합니다.
- `모공`은 현재 FreqSep 구조 안의
  - attenuation band
  - high-frequency preserve
  - detail protection
  축으로 설명하는 것이 타당합니다.
- 구조적으로는 별도 pass를 바로 추가하기보다, **FreqSep 내부 파라미터 분리**를 먼저 시도하는 것이 맞습니다.

### 보완이 필요한 부분

#### 1. `매끈하게`의 내부 신호에 `effectStrength`를 포함해야 함

현재 결과는 `toneLift`와 `textureBlend`만이 아니라, shadow/highlight 보호 로직에 의해 실제 체감 강도가 달라집니다.

- 현재 Composite 셰이더에서 `effectStrength = shadowProtection * highlightProtection`
- `toneFinish`와 최종 결과 mix 모두 이 값의 영향을 받음
- 따라서 `매끈하게`는 단순히 `toneLift`만 올리는 축이 아니라, **어떤 조명 구간에서 finish가 얼마나 먹는지**까지 포함해서 봐야 합니다.

#### 2. `largeDetailProtection`은 아직 고정 전제로 두기에는 이르다

현재 문서에서는 모공 축에서 `largeDetailProtection` 임계값을 유지 대상으로 두고 있는데, 지금 단계에서는 약간 이릅니다.

- 현재 구현에서 이것은 "완전 보호"가 아니라 **큰 디테일에 대한 압축 완화**에 가깝습니다.
- 값이 너무 이르면 큰 피부결/넓은 모공이 보호 구간으로 빠질 수 있고,
- 너무 늦으면 털, 점, 깊은 경계까지 눌릴 수 있습니다.

즉 `largeDetailProtection`은 현재 상태에서 "건드리면 안 되는 상수"라기보다, **모공 축 검증 대상 파라미터**로 남겨두는 편이 맞습니다.

#### 3. `blur_radius`는 공통 파라미터로 단순 공유하기 어렵다

문서의 `mapSmoothing()` / `mapPoreReduction()` 예시에서 `blur_radius`를 공통 결합 항목으로 두었는데, 이 값은 단순 강도 파라미터가 아니라 **주파수 분해 기준 자체**입니다.

- `매끈하게`는 더 넓은 low-frequency skin base를 원할 수 있고
- `모공`은 너무 큰 radius가 되면 오히려 분리 기준이 달라져 미세결 선택성이 바뀔 수 있습니다.

따라서 `blur_radius`는
- 완전 공통 항목으로 묶기보다
- low-pass 분해 기준과 pore band 선택에 미치는 영향을 따로 검토해야 합니다.

즉 다음 토론에서는 `blur_radius`를 "공유 파라미터"로 바로 두지 말고, **분리 기준 파라미터**로 별도 취급하는 것이 좋습니다.

#### 4. `모공 축은 새 알고리즘 없이 충분`은 표현을 약하게 가져가는 편이 안전함

방향성에는 동의하지만, 현재 단계에서 `충분`이라고 쓰면 확정처럼 읽힙니다.

더 적절한 표현은 다음 정도입니다.

- `현재 구조 + 파라미터 분리/튜닝으로 1차 검증 가능`
- `새 알고리즘 도입 전에 현 구조의 한계를 먼저 측정`

즉 모공 축은 **당장 구조 변경 없이 실험 가능한 상태**라는 의미로 쓰는 것이 맞고, 품질 충분성 자체는 아직 검증 전입니다.

### 다음 토론에서 추가로 정리할 항목

#### 매끈하게 축

- `effectStrength`를 매끈하게 파라미터에 포함할지
- shadow/highlight 보호를 완화할지 유지할지
- `textureBlend floor`의 상한을 어디까지 허용할지

#### 모공 축

- `largeDetailProtection` 임계값을 고정할지 슬라이더 연동할지
- `sharpen_amount`를 모공 축에 종속시킬지 별도 보정할지
- 실제 모공 크기에 대응하는 magnitude band 재측정 필요 여부

#### 공통 구조

- `blur_radius`를 두 축이 공유할지, 한 축만 제어할지
- 두 슬라이더가 동시에 높을 때의 결합 규칙
- `FreqSep 내부 해결`이 실패라고 판단하는 기준을 무엇으로 둘지

### Codex 결론

이번 2차 토론의 큰 방향은 맞습니다.  
다만

- `매끈하게 = low_freq_smooth + blend floor + toneLift`
- `모공 = attenuation + preserve`

로 단순화해 확정하기보다는,

- `effectStrength`
- `largeDetailProtection`
- `blur_radius`

까지 포함한 **전체 신호 체계**로 한 단계 더 정리하는 것이 다음 토론의 핵심입니다.

즉 지금 문서는 좋은 2차 정리이고, 다음 단계에서는 "어떤 파라미터를 추가할까"보다 먼저  
**각 슬라이더가 실제로 어떤 내부 신호 집합을 소유하는지**를 확정해야 합니다.

---

## 15. Gemini 의견 (3차)

Codex의 신중한 접근과 Claude의 세밀한 신호 분해에 **전적으로 동의하며, 이를 실험적으로 증명하기 위한 '인지적 간섭 제어' 관점**을 추가합니다.

### 1. 신호의 독립성(Orthogonality) 검증의 필요성

Claude가 제안한 `mapSmoothing`과 `mapPoreReduction` 분리는 매우 훌륭하지만, 실제 GPU 파이프라인에서는 **두 축이 '비선형적'으로 간섭**할 가능성이 큽니다.

- **간섭 시나리오**: `매끈하게`를 위해 `low_freq_smooth_radius_ratio`를 올리면, 저주파 베이스가 더 뭉개지면서 `모공` 축에서 잡아야 할 '미세 디테일(magnitude)'의 기준선이 함께 흔들릴 수 있습니다.
- **해결안**: 두 축을 결합할 때, 단순히 파라미터를 따로 던지는 것이 아니라 **"인지적 주파수 마스킹"** 개념을 도입해야 합니다. 즉, 매끈함이 강해질수록 모공 보정의 임계값(attenuation)을 지능적으로 시프팅(Shifting)하는 연산이 필요할 수 있습니다.

### 2. `blur_radius`: 공유 변수가 아닌 '동적 기준선(Dynamic Baseline)'

Codex가 지적한 `blur_radius` 문제는 이 토론의 핵심입니다.

- **Gemini 제안**: `blur_radius`는 두 축의 '입력값'이 아니라, **두 축의 요구사항을 반영한 '결과값'**이어야 합니다.
  - `매끈하게`가 요구하는 $R_{smooth}$와 `모공`이 요구하는 $R_{pore}$를 각각 계산한 뒤,
  - `final_blur_radius = max(R_smooth, R_pore)`와 같은 방식으로 결정하되,
  - 이 반경 변화가 FreqSep의 **'주파수 분해 해상도'**에 미치는 영향을 보정하는 가중치를 Composite 셰이더에 전달해야 합니다.

### 3. 구체적인 수치 실험 프로토콜 (Empirical Protocol)

이론적 논의를 확정하기 위해 다음 3가지 실험을 우선 제안합니다.

1.  **Texture Blend Floor 실험 (0.38 → 0.60):** `textureBlend`의 최소값을 단계적으로 올렸을 때, B612 특유의 '파운데이션을 바른 듯한 질감'이 실제 구현되는지 확인.
2.  **Magnitude Band 재측정:** `attenuation_high`를 현재의 0.026에서 0.050까지 확장하여, 소위 '왕모공'이 압축되는 임계값을 실측.
3.  **Tone Lift 인지 테스트:** `toneLift` 증가가 다크서클이나 팔자주름 부위의 명도를 실제로 자연스럽게 끌어올리는지(Skin Finish 효과), 아니면 단순히 이미지를 하얗게 띄우는지 검증.

### Gemini 결론

현재 논의된 **"FreqSep 내부 파라미터 분리 및 신호 재정의"**는 B612 수준의 품질을 확보하기 위한 가장 정교한 설계도입니다. 

다음 단계에서는 **`blur_radius`의 결정 공식을 확립**하고, 위 3가지 실험 수치를 바탕으로 **`mapSkinQualityDetailed` 함수의 내부 상수값들을 확정**하는 방향으로 진행할 것을 권장합니다.

---

## 16. 최종 수렴 (Claude 5차 — 토론 마무리)

### 토론 종료 판단

3자 토론을 4라운드 진행한 결과, **더 토론해도 코드 없이 나올 수 있는 인사이트는 한계에 도달**했습니다. 큰 방향은 합의되었고, 남은 질문은 모두 "실험해봐야 안다"로 수렴합니다.

### 확정된 합의 사항

| 항목 | 상태 |
|------|------|
| `skinQuality` ≈ B612 모공 축 | 확정 (3자 합의) |
| 현재 `smoothing`(Bilateral)은 매끈하게에 부족 | 확정 (3자 합의) |
| 제품 UX를 매끈하게/모공 2축으로 재정의 | 확정 (3자 합의) |
| FreqSep 구조 유지 + 내부 파라미터 분리 우선 시도 | 확정 (3자 합의) |
| 매끈하게 신호: 저주파 톤 정리 + blend floor + tone finish + effectStrength | 확정 (Claude 제안 + Codex 보완) |
| 모공 신호: attenuation band + high_freq_preserve + detail protection | 확정 (Claude 제안 + Codex 보완) |
| `blur_radius`는 단순 공유 불가, 분해 기준으로 별도 취급 | 확정 (Codex 지적) |
| `largeDetailProtection`은 고정 상수가 아닌 검증 대상 | 확정 (Codex 지적) |

### 미해결 — 실험으로만 답할 수 있는 질문

| 질문 | 실험 방법 |
|------|----------|
| blend floor를 올리면 B612 매끈하게 느낌이 나는가? | floor 0.38 → 0.50 → 0.60 단계별 실기기 비교 |
| 현재 attenuation band가 실제 모공을 다 잡는가? | attHigh 0.026 → 0.035 → 0.050 확장 후 비교 |
| toneLift 증가가 자연스러운 skin finish인가, 단순 밝아짐인가? | toneLift 0.05 → 0.08 → 0.12 실기기 비교 |
| 매끈하게 + 모공 동시 최대 시 간섭이 심한가? | 두 축 독립 변경 후 조합 테스트 |
| Pass 2 radius만으로 톤 균일화가 충분한가? | ratio 0.32 → 0.50 확장 후 B612과 비교 |
| FreqSep 내부로 충분한가, Guided Filter/별도 pass가 필요한가? | 위 실험 결과에 따라 판단 |

### 다음 단계: 실험 진행 순서

```
1. 데모앱에 매끈하게/모공 독립 슬라이더 추가 (UI만)
   → mapSkinQuality()를 2축으로 분리하는 프로토타입

2. 실기기에서 파라미터 흔들기
   → blend floor, attenuation band, toneLift 각각 단독 변경
   → B612 스크린샷과 나란히 비교

3. 간섭 테스트
   → 매끈하게 max + 모공 0, 매끈하게 0 + 모공 max, 둘 다 max

4. 결과에 따라 구조 결정
   → 충분하면 FreqSep 내부 확정
   → 부족하면 Guided Filter 또는 별도 pass 검토
```

---

## 17. 프로토타이핑 실험 결과 (2026-03-24)

> 브랜치: `feature/beauty-smoothing-prototype`
> 비교 기준: B612 앱의 매끈하게/모공 기능 (실기기 스크린샷 비교)

### 구현 내용

- `BeautyFilterConfigV2`에 `smoothIntensity`, `poreReduction` 2개 필드 추가
- `mapSmoothingAndPore()` 함수로 2축 독립 FreqSepParams 매핑
- Composite 셰이더에 `uTextureBlendFloor` uniform 추가
- 데모앱에 매끈하게/모공 독립 슬라이더 추가
- 기존 `skinQuality`와 하위 호환 유지

### 실험 1: 매끈하게 축

#### 발견한 문제들과 수정 과정

| 이터레이션 | 문제 | 원인 | 수정 |
|-----------|------|------|------|
| 1차 | 슬라이더를 올려도 변화 없음 | `active_filter_count`에 2축 파라미터 미포함 → fast-path 패스스루 (외부 리뷰에서 발견) | `active_filter_count` 조건에 `smoothIntensity`/`poreReduction` 추가 |
| 2차 | 효과가 너무 미미 | `texture_blend_floor`가 `compression`에 종속 → compression 낮으면 floor 무의미 | `max(poreBlend, smoothFloorProtected)` 로직으로 변경 |
| 3차 | 여전히 미미 | `high_freq_preserve=0.72` — 매끈하게 축에서 고주파 72% 보존 중 | `hfp`를 매끈하게 축에서도 제어 (`min(hfp_smooth, hfp_pore)`) |
| 4차 | 효과 나지만 **뿌연 느낌** (haze) | `tone_lift`가 밝기를 올림 + 에지(안경테, 코)까지 같이 밀림 | `tone_lift` 최소화 + 에지/색차 보호를 `textureBlend`에도 적용 |

#### 최종 매끈하게 파라미터 (극단 테스트 기준)

```
blur_radius:              face_w * 0.138 (clamp 5~28)
low_freq_smooth_ratio:    0.22 ~ 0.70
texture_blend_floor:      0.38 ~ 0.95
tone_lift:                0.002 ~ 0.007
high_freq_preserve:       0.72 ~ 0.17
sharpen_amount:           0.11 ~ 0.25
edge_weight:              0.42 ~ 0.62
```

#### 매끈하게 결론

- **방향은 맞음** — FreqSep 내부 파라미터 분리로 매끈하게 효과가 동작함
- **B612 대비 아직 부족** — 극단값에서도 B612보다 약하고 뿌연 느낌 있음
- **핵심 병목: 마스크 정밀도** — 안경테, 콧볼, 헤어라인 등 비피부 영역에 스무딩이 먹혀서 뿌옇게 보임. B612은 이 영역을 정밀하게 제외함
- **부차 병목: Gaussian blur의 한계** — 넓은 blur에서 에지 근처에 halo 발생. Guided Filter 도입이 여전히 유효한 후보

### 실험 2: 모공 축

#### 테스트 조건

- 잡티보정 0, 매끈하게 0, 모공 100
- B612 모공 100과 나란히 비교

#### 결과

| | B612 모공 | 우리 모공 |
|---|---|---|
| 모공 감소 | 볼/코 옆 확실히 줄어듦 | **변화 거의 없음** |
| 톤 변화 | 없음 | 없음 |
| 에지 보존 | 완벽 | 유지됨 |

#### 로그 확인

```
[2AXIS] smooth=0.00 pore=1.00 face_w=621 enabled=1 blur_r=19 blendFloor=0.38 toneLift=0.002 hfp=0.20
```

파라미터는 정상 전달됨. `hfp=0.20`이면 고주파 80% 제거인데도 시각적 변화가 미미.

#### 모공 축 문제 분석

**디버그 히트맵 결과 (2026-03-25)**

6개 세부 디버그 모드(Magnitude, MicroBand, EdgeProt, EffectStr, Compression×3, Mask)로 시각적 분석한 결과:

- **Magnitude**: 피부 전체에 고주파 신호 충분히 분포 — 신호 자체는 문제 없음
- **MicroBand**: 볼/코 옆에 빨간색(대역 내) 분포 — 모공 대역을 잡고 있음
- **EdgeProt**: 피부 영역은 초록(보호 안 됨) — 에지 보호가 과하지 않음
- **EffectStr**: 대부분 노란색(활성) — 그림자 보호도 문제 아님
- **Compression×3**: 빨간 점 분포하지만 전체적으로 초록 우세 — **compression은 동작하나 값이 중간(0.2~0.4)**

**결론: 개별 단계는 모두 동작하지만, 5개 보호 신호의 곱셈 구조로 인한 감쇠가 근본 원인.**

```
compression = microTextureBand(~0.5) × largeDetailComp(~0.8) × edgeProt(~0.85) × chromaProt(~0.9) × effectStr(~0.8)
            ≈ 0.24
```

**compression 증폭 실험 (실패)**

포물선 부스트 `raw * (2.0 - raw)` 적용 → compression 0.24→0.42로 증가했지만 **뿌연 느낌이 심해짐**. 모공뿐 아니라 모든 영역의 블렌딩이 올라가서, 기존 잡티보정(skinQuality)보다 결과가 나빴음. **compression 전체를 올리는 접근은 잘못됨.**

**남은 가설**

- compression 자체를 올리면 뿌옇게 됨 → 전체 부스트가 아닌 **모공 특정 대역만 선택적 증폭** 필요
- 또는 **`preserve` 공식 자체를 변경** — 현재 `mix(1.0, hfp, compression)` 대신 모공 대역에서만 더 강한 압축을 적용하는 비선형 매핑
- 또는 **현재 FreqSep 구조의 한계** — "보호하면서 강하게 눌러야" 하는 상충적 요구를 곱셈 구조로는 해결 불가. 별도 모공 전용 pass 검토 필요

### 참고 스크린샷

| 파일 | 설명 |
|------|------|
| `b612_filter_none.jpg` | B612 필터 없음 (원본) |
| `b612_filter_매끈하게.jpg` | B612 매끈하게 적용 |
| `b612_filter_모공.jpg` | B612 모공 적용 |
| `Screenshot_20260324_irislens.png` | SDK 매끈하게 100 (1차) |
| `Screenshot_20260324_b618.png` | B612 매끈하게 (비교용) |
| `Screenshot_20260324_irislens_after.png` | SDK 매끈하게 100 (blur 강화 후) |
| `Screenshot_20260324_160519.png` | SDK 매끈하게 100 (극단값) |
| `Screenshot_20260324_180653.png` | SDK 매끈하게 100 (hfp 수정 후) |
| `Screenshot_20260324_181424.png` | SDK 매끈하게 100 (에지 보호 추가 후) |
| `Screenshot_20260324_b162_mogong_off.png` | B612 모공 OFF |
| `Screenshot_20260324_b162_mogong_full.png` | B612 모공 100 |
| `Screenshot_20260324_iris_mogong_off.png` | SDK 모공 OFF |
| `Screenshot_20260324_iris_mogong_full.png` | SDK 모공 100 |

### 프로토타이핑 종합 판단

| 항목 | 결과 |
|------|------|
| 2축 독립 제어 동작 | **확인** |
| FreqSep 내부 파라미터 분리 가능성 | **확인** — 매끈하게/모공 각각 다른 파라미터 세트를 제어 |
| 매끈하게 품질 | **부족** — 방향은 맞지만 마스크 정밀도 + blur 방식의 한계로 뿌옇게 됨 |
| 모공 품질 | **부분 동작** — 효과는 있지만 B612 대비 약함. 곱셈 감쇠 구조가 근본 한계 |
| compression 전체 부스트 | **실패** — 뿌옇게 됨, 기존 잡티보정보다 나쁨 |
| B612 수준 도달 가능 여부 | **현재 FreqSep 곱셈 구조만으로는 어려움** |

### 프로토타이핑에서 확인된 핵심 인사이트

1. **2축 분리 자체는 유효** — 매끈하게/모공이 서로 다른 신호를 제어하는 구조가 동작함
2. **매끈하게의 병목은 마스크 정밀도** — 안경테, 코 윤곽 등 비피부 영역에 스무딩이 먹혀서 뿌옇게 보임
3. **모공의 병목은 곱셈 감쇠** — 5개 보호 신호를 곱하면 compression이 0.2~0.3 수준으로 떨어져 효과 미미
4. **compression 전체 부스트는 해법이 아님** — 모공뿐 아니라 모든 영역이 뿌옇게 됨
5. **기존 잡티보정(skinQuality)이 현재 구조에서는 더 나은 체감** — 단일 축이지만 튜닝이 잘 되어 있음

### 다음 단계 (프로토타이핑 이후)

1. **마스크 정밀도 개선** — 피부 세그멘테이션 모델 도입 or ROI 마스크 고도화 (매끈하게/모공 모두에 영향)
2. **모공 전용 처리 방식 재설계** — 곱셈 구조 대신 모공 대역만 선택적으로 강하게 눌리는 별도 로직 검토
3. **Guided Filter 재검토** — 매끈하게의 Gaussian blur 한계가 확인됨, 에지 보존 스무딩이 여전히 유효
4. **현재 프로토타입은 실험 인프라로 유지** — 디버그 히트맵, 2축 슬라이더, 파라미터 분리 구조는 후속 작업의 기반

---

## 수정이력

| 날짜 | 작성자 | 내용 |
|------|--------|------|
| 2026-03-23 | Claude | 초안 작성 — B612 스크린샷 분석, 알고리즘 방향 추론, SDK 매핑, 제안 방향 정리 |
| 2026-03-23 | Codex | `Codex 의견` 섹션 추가 — B612 매끈하게/모공과 SDK `smoothing`/`skinQuality`의 매핑 보완, Guided Filter 단정에 대한 유보, 제품/구현 방향 제안 정리 |
| 2026-03-23 | Gemini | `Gemini 의견` 섹션 추가 — '면(Area) vs 점(Point)' 관점의 기술 분석, 독립적 최적화 후 지능적 통합(Decoupling then Integration) 전략 제안, FreqSep 기반 Guided Filter 강화안 제시 |
| 2026-03-23 | Claude (2차) | `수렴 결론` 섹션 추가 — 초안의 Guided Filter 단정 철회, 3자 합의 방향 정리 (FreqSep 유지 + 2축 분리), 구현 우선순위 확정 |
| 2026-03-23 | Codex (2차) | 수렴 결론에 부분 동의 — 문제 정의/제품 축 분리는 타당, Guided Filter 채택과 구현 우선순위는 가설 수준으로 유보. 다음 토론 주제 3가지 제안 |
| 2026-03-23 | Gemini (2차) | 수렴 결론에 강한 동의 — 매핑이 업계 표준과 일치, 이대로 진행 권장 |
| 2026-03-23 | Claude (3차) | 최종 재수정 — Codex 2차 지적 수용, Guided Filter를 유력 가설로 격하, 구현 우선순위 삭제, 다음 토론 주제 설정 |
| 2026-03-23 | Claude (4차) | 2차 토론 의견 추가 — 매끈하게/모공 내부 신호 정의, FreqSep 내부 우선 시도 제안, 단계적 판단 흐름 제시 |
| 2026-03-23 | Codex (3차) | `14. Codex 의견 (3차)` 추가 — 2차 토론에 대한 보완 의견 정리, `effectStrength`/`largeDetailProtection`/`blur_radius`를 추가 검토 항목으로 제안 |
| 2026-03-23 | Gemini (3차) | `15. Gemini 의견 (3차)` 추가 — 신호의 독립성(Orthogonality) 및 인지적 간섭 제어 관점 보완, `blur_radius`의 동적 결정 공식 및 구체적 수치 실험 프로토콜 제안 |
| 2026-03-23 | Claude (5차) | 토론 마무리 — 확정 합의 8개 항목 정리, 미해결 질문을 실험 계획으로 전환, 다음 단계 실행 순서 확정 |
| 2026-03-24 | Claude (6차) | 프로토타이핑 실험 결과 추가 — 매끈하게 4회 이터레이션, 모공 테스트, 종합 판단 및 다음 단계 기록 |
| 2026-03-25 | Claude (7차) | 디버그 히트맵 분석 결과 추가, compression 포물선 부스트 실험(실패) 기록, 프로토타이핑 종합 판단 및 다음 단계 업데이트 |
| 2026-03-25 | Codex (4차) | 프로토타이핑 결과 피드백 — "실험 프레임으로 성공, 해법으로 미완" 프레이밍, 마스크→결합식→Guided Filter 우선순위 제안, C API stale 값 지적 |
| 2026-03-25 | Gemini (4차) | 프로토타이핑 결과 피드백 — 곱셈 감쇠 분석 공감, Mask Refinement Pass(Erode/Dilate) 제안, Pore-specific Nonlinear Operator 제안 |
| 2026-03-25 | Claude (8차) | 마스크 정밀도 개선 토론을 별도 문서(`beauty_mask_refinement_discussion.md`)로 분리 |
