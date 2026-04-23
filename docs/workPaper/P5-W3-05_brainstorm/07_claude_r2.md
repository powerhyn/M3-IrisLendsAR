# Claude R2 — 쟁점별 입장 재평가

> **작성**: Claude Opus 4.7 (1M), R1/R2 이슈 문서 작성자이자 모더레이터
> **시점**: 2026-04-22 R2 시작 시점 (Codex/Gemini의 R2 응답 보기 전)
> **원칙**: Codex의 R1이 가장 예리하고 구체적(파일:라인까지). 내 R1 중 Codex 논리에 밀리는 지점은 정직하게 철회한다.

---

## I1 — 환경 반사 소스: **부분 철회**

**내 R1**: 카메라 프레임 mip-down을 환경맵 대체로 쓰자.

**재평가**: Codex의 반론 "전면 카메라는 얼굴밖에 안 찍어 재귀 + 조명원 부재"가 구조적으로 맞다. 셀피 프레임의 60~80%가 얼굴·배경 근처고, 실제 **오프스크린 광원(창문/형광등)이 프레임에 거의 안 잡힌다**. 사용자는 alignment보다 "하이라이트 존재 자체"를 더 인지한다는 감각은 맞지만, 잘못 된 위치의 반사는 정렬 오류로 오히려 어색을 증폭시킬 수 있다.

**입장**: 내 R1은 부분 철회. **Codex의 LDR/RGBM eye env map 에셋(256×128~512×256)을 디폴트로 채택**. 카메라 프레임 mip-down은 "상단 1/4 영역만 env 증강 소스로 쓰는 optional hybrid"로 한정 — Codex의 `optional reflection_dir`이 들어오면 env map 회전, 없으면 정적 env + 상단 crop 증강.

**Gemini 평가**: Gemini의 "고휘도 영역 샘플링"도 같은 재귀 문제. 기각.

**수렴 제안**: 실기기 2시간 A/B (정적 env map only vs env+상단crop hybrid vs 현 W3-04 고정조명). 매트릭스: 3 환경(실내형광/창가/야간) × 2 조명방향(정면/측면) = 6 클립. 평가자 3명 블라인드 비교.

---

## I2 — 블렌드 세트: **4종으로 확장 수용**

**내 R1**: Normal / LTL / Multiply (3종).

**재평가 (서브쟁점별)**:

- **I2-a (Normal 드롭)**: Codex "Normal은 flat"이 맞다. 불투명 서클렌즈에서 `mix(base, blend, a)`는 휘도 구조 0 — 플라스틱 공 효과. ColorReplaceLinear의 `detail = clamp(pow(lum/avgLum, 0.7), 0.75, 1.25)`이 휘도 비율 기반으로 미세 구조를 보존하면서 색 정확도 유지. **내 "단순성"은 제품 가치가 아님. Normal 유지 철회.**

- **I2-b (ScreenLinear 별도)**: LTL은 `baseL × lensL × lum × scale`로 **승산적**이라 dark iris(baseL 낮음) 위 밝은 렌즈(그레이/블루/바이올렛)를 올리면 `baseL * scale` 상한에 막힘. Screen `1-(1-base)(1-blend)`은 **가산적**이라 이 셀을 넘길 수 있음. 수학적으로 LTL로 흡수 불가. Codex 논리 맞음. **ScreenLinear 추가 수용.**

- **I2-c (Gemini 하이브리드 Circle/Vivid)**: Normal+Overlay 하이브리드는 옵션 2개(rxInner/opacity 또는 contrast_strength)로 튜닝한 단일 블렌드로 치환 가능. 별도 모드로 유지할 가치 없음. **Gemini 하이브리드 기각.**

- **I2-d (ColorReplaceLinear 신규)**: sRGB `ColorReplace`는 감마 왜곡으로 밝은 홍채에서 색 변질. Codex의 선형 공간 버전은 별도 가치 있음. **채택.**

**입장**: **Codex의 4종 세트 수용**: `TintLinearV2 / Multiply / ScreenLinear / ColorReplaceLinear`. 기본값 `TintLinearV2`. 내 R1의 3종(Normal 유지)은 철회.

**수렴 제안**: 실기기 매트릭스 3 SKU(다크브라운·헤이즐·밝은블루) × 2 홍채 톤(짙음·밝음) × 4 블렌드 = 24 클립. 각 셀에서 "가장 자연스러운 블렌드" 블라인드 투표. 4종 중 미사용(0득표) 블렌드 발생 시 3종으로 재축소 검토.

---

## I3 — LTL 실반사 보호: **Codex 수용, 모드에서 완전 제거**

**내 R1**: 임계값 0.7/0.95 → 0.45/0.75 튜닝.

**재평가**: Codex "realSpec은 spec 보호가 아니라 밝은 픽셀 보호 (개념 자체 틀림)"이 정확. 실제 specular는 normal·light·view 기하로 계산되는데, 우리는 lum 임계값을 specular proxy로 썼을 뿐. I1에서 환경 반사가 **분리된 계층**으로 들어오면 LTL 내부 realSpec은 필요 없어짐. 밝은 픽셀 보호가 정말 필요하면 별도 `highlight preservation` 후처리로.

**입장**: 내 R1 "임계값 튜닝" 철회. **realSpec 블록 완전 제거**. 환경 반사 계층이 specular 담당.

**수렴 제안**: I1(환경 반사 도입)과 묶어서 동시에 구현. LTL 내부 `smoothstep(0.7, 0.95, lum)` + baseL 복귀 2줄 제거.

---

## I4 — 림발 링: **기본 OFF 철회, ON + 메타데이터 + 밝기 fallback**

**내 R1**: 기본 OFF, opt-in. 텍스처 가이드라인으로 해결.

**재평가**: Codex "품질 책임을 asset authoring에 떠넘기는 것은 SDK 설계 실패"가 맞다. B2B 파트너사 텍스처 품질 편차 큰데 SDK가 경계 cue를 내려주지 않으면 자연스러움이 PNG 품질에 종속.

**입장**: 내 R1 철회. **기본 ON**. 이중 적용 방지 전략은:
1. SKU 메타데이터 `has_baked_limbal: true` 있으면 셰이더 림발 강도 0 (Codex 안)
2. 메타데이터 없으면 텍스처 `edge region(r∈0.7~1.0)`의 평균 밝기가 `center region(r<0.3)` 대비 낮으면 이미 림발 포함으로 판정, 자동 감쇄 (Gemini 안의 파생)

즉 **Codex(명시) + Gemini(자동) 하이브리드**. 메타 우선, 메타 없을 때 자동 fallback.

**수렴 제안**: 텍스처 밝기 분석은 텍스처 로딩 시점 1회만 수행 (CPU측, 런타임 비용 0). 메타데이터 포맷은 `LensConfig`에 `bool has_baked_limbal` 추가.

---

## I5 — 블링크 처리: **Codex 수용, Gemini 반박**

**내 R1**: alpha ramp down (3~5프레임 ~50~80ms).

**재평가**: Gemini "페이드 = 눈꺼풀 위 투영 부자연" 우려는 **이미 `eyelidMask`가 눈꺼풀 영역에서 자동 감쇄**하므로 페이드가 눈꺼풀 위로 새지 않음. Gemini 논리 틀렸다.

Codex의 ease-out 60~80ms + ease-in 100~120ms는 실제 블링크(100~150ms)와 조화. 내 R1의 ~50~80ms ease-out보다 Codex의 비대칭 시간(close는 짧게, open은 길게)이 **자연스러운 인지 심리에 더 부합** (열릴 때 서서히 나타나는 게 덜 튀어 보임).

**입장**: Codex 시간 수용. Gemini "즉시 Off" 반박 — eyelidMask로 이미 방어된 우려.

**수렴 제안**: `render_alpha` EMA 2계수: α_close=0.15(빠르게), α_open=0.08(천천히). 실기기 5사용자 블링크 10회씩 샘플링, 시각적 "팝" 체감 투표.

---

## I6 — 입력 계약: **Codex EyeRenderPacket 수용, gaze_vector 기각**

**내 R1**: pupil_center + render_confidence (optional).
**Gemini**: gaze_vector 추가.

**재평가**: Codex의 `EyeRenderPacket` 구조화는 내가 제안한 pupil_center/render_confidence를 포괄하면서 확장성까지. gaze_vector는 `pupil_center - iris_center`로 파생 가능해서 **redundant**. 입력 필드 증가는 결합도 증가.

**입장**: **Codex EyeRenderPacket 수용**. 내 R1 pupil_center 제안은 Codex 패킷의 optional 필드로 흡수. Gemini gaze_vector 기각 (pupil_center로 파생).

**Codex에게 묻는 것**: `avg_iris_luma` optional fallback 수식 확정 필요 (I7과 묶음).

**수렴 제안**: 공개 C API는 변경하지 않고, 내부 렌더러 ↔ 검출기 어댑터 계약만 `EyeRenderPacket` 구조체로 정의. 기존 C API는 어댑터에서 패킷으로 변환.

---

## I7 — uAvgIrisLum 하드코드: **Codex 비판 수용, ROI 측정 도입**

**Codex R1 지적**: `uAvgIrisLum = 0.35`는 priors 덮어씌움.

**재평가**: 맞다. 개인별 홍채 밝기 + 조명 노출 큰 편차인데 고정값으로 블렌드 정규화는 개인화 파괴. 내 R1에서 이 문제를 놓친 것.

**입장**: Codex 비판 수용. 대안:
- **측정**: iris ROI 중심 반경 0.4 이내의 픽셀 평균 휘도. 매 프레임 `textureLod(camera, iris_center, 3.0)`으로 mip level 3 샘플 1개만 가져옴 (추가 fetch 1개, 비용 무시).
- **Fallback**: 측정 실패(검출 실패, iris 안 보임) 시 이전 값 hold. 장기 hold(3프레임 이상) 시 0.35 fallback.

**수렴 제안**: `avg_iris_luma`를 EyeRenderPacket의 optional 필드로 — 상위 레이어가 측정해서 내려주면 받고, 없으면 렌더러가 self-measure.

---

## I8 — calcScleraFactor: **하한 추가 철회, Codex veto 방식 수용**

**내 R1**: `brightFactor` 하한 0.3 추가.

**재평가**: Gemini "조명 변화 취약" + Codex "그레이/블루 렌즈 오인"이 공통 비판. 내 하한 추가는 미봉책. 그러나 Gemini의 "완전 폐기" 안은 지나침 — 색상 기반이 유효한 영역(어두운 홍채 + 밝은 흰자)도 있음.

Codex의 "색상은 veto 정도만"이 균형. 즉 **기하학 기반이 주, 색상은 거부권만**:
- 기하학이 "감쇄 필요"라고 판정하고
- 색상이 "분명 흰자 아님(낮은 밝기 또는 높은 채도)"이라고 강하게 반박하면
- 감쇄 해제.

**입장**: 내 하한 추가 철회. Codex veto 수용. Gemini 완전 폐기 반박.

**구체 수식**: `scleraFade = 1.0 - geomFactor × (1.0 - colorVeto)`, 여기서 `colorVeto = smoothstep(0.15, 0.05, brightness) + smoothstep(0.2, 0.35, saturation)` — 밝기 낮거나 채도 높으면 veto 활성.

---

## I9 — 홍채 디테일 재주입: **Codex 원본 재주입 수용**

**내 R1**: 미언급.
**Gemini**: 절차적 노이즈 섬유.
**Codex**: 원본 휘도 재주입.

**재평가**: 내 R1이 이 축을 빼먹은 건 실수. "지금 문제는 환경 반응성"이라는 입장 때문에 디테일 재주입 가치를 과소평가.

Gemini 절차적 생성은 사용자 본인 홍채와 **불일치하는 가짜 디테일**. 자연스러움 목표에 반함. Codex 원본 재주입은 사용자 본인 홍채의 미세 구조를 살려 자연스러움.

**입장**: Codex 원본 재주입 수용. Gemini 절차적 생성 기각.

**저조도 노이즈 대응**: magnitude gate 추가 — `detail`을 `if abs(detail-1) < 0.1 then 1.0`로 gate, 큰 변화만 통과.

**수렴 제안**: 실기기 3조도(밝음/보통/저조도)에서 재주입 ON/OFF 비교. 저조도에서 노이즈 증폭 시 gate 임계값 조정.

---

## I10 — W2-W3 경계: **Codex 기각, W2 트랙 유지**

**Codex R1**: W3 와일드카드에 eye-only refiner 포함 (pupil_center/occlusion/gaze 출력).

**재평가**: Codex 논리 "렌더링 품질 직접 기여"는 맞지만, **아키텍처 구분이 원래 목적**. W2가 refiner 담당, W3가 receiver. 이를 합치면 Codex 자신이 I6에서 주장한 "렌더러는 raw detector 산출물이 아닌 EyeRenderPacket만 받는다"와 모순.

**입장**: Codex 기각. W2 트랙 유지. W3 브레인스토밍의 스코프 밖.

**수렴 제안**: W2 워크페이퍼에 "pupil_center 출력 포함" 요구사항 추가. W3는 W2 산출물을 EyeRenderPacket으로 수신만.

---

## 정리 — 내 R1 철회/수용 현황

| 쟁점 | 내 R1 | 내 R2 (변경 후) | 변경 이유 |
|------|-------|----------------|----------|
| I1 | 카메라 프레임 mip | **부분 철회** — env map 에셋 디폴트 + 상단 crop optional | Codex 재귀/광원 부재 논리 수용 |
| I2 | Normal/LTL/Multiply (3종) | **철회 — Codex 4종 수용** | Normal flat, ScreenLinear 필요성, ColorReplaceLinear 가치 |
| I3 | 임계값 튜닝 | **철회 — realSpec 완전 제거** | 반사 분리 시 불필요 |
| I4 | 기본 OFF | **철회 — 기본 ON + 메타+자동 hybrid** | 책임을 asset에 떠넘김 |
| I5 | 50~80ms ramp | **Codex 수용 — 비대칭 60/100ms** | Gemini 우려 eyelidMask로 방어 |
| I6 | pupil + confidence | **Codex 수용 — EyeRenderPacket** | 구조화 우월 |
| I7 | 미언급 | **Codex 수용 — ROI 측정** | 하드코드 개인화 파괴 지적 |
| I8 | 하한 추가 | **철회 — veto 방식** | 하한은 미봉책 |
| I9 | 미언급 | **Codex 수용 — 원본 재주입** | 실수 인정 |
| I10 | W2 밖 | **입장 유지** — Codex 기각 | W2-W3 분리 원칙 |

**내 R1 대비 최종 유지 항목**: I10뿐. 나머지 9개 쟁점에서 모두 Codex(일부 Gemini) 논리에 수용.

---

## 모더레이터로서 R3 필요성 판단 (초안)

R2에서 Codex/Gemini 응답이 오면 재확인할 쟁점:

1. **I1 Codex vs Gemini**: Gemini가 SSR 입장 유지할지, Codex에 수용할지. 유지 시 **하이브리드 안이 실기기 벤치의 지배 승자인지만 R3로 검증.**
2. **I2 Gemini의 하이브리드 기각 수용할지**: 수용 시 Codex 4종으로 수렴.
3. **I4 메타+자동 hybrid 실제 구현 복잡도**: 텍스처 분석 로직의 false positive 리스크를 Gemini/Codex가 어떻게 평가하는지.
4. **I9 노이즈 gate 임계값**: 저조도 실측 데이터 없이 말로만 결정 가능한지.

R3 필요 여부는 위 4개가 R2에서 **모두 수렴**하면 불필요. 하나라도 남으면 실기기 벤치 계획(`final_decision.md`의 Phase 6 계획)으로 이관.

---

## R3로 넘기지 말고 바로 실기기 벤치로 이관해야 하는 쟁점

- **I1 환경 반사 구현**: 말로 결정 불가. 프로토타입 2종(env map only, env+상단crop) + 3종 조도 = 6 클립 A/B.
- **I2 블렌드 매트릭스**: 24 클립 블라인드 투표.
- **I5 블링크 시간**: 5 사용자 블링크 10회 샘플링.
- **I9 디테일 재주입 노이즈**: 3 조도 실기기 ON/OFF.

총 4개 벤치, 총 소요 ~6시간. W3-05 구현 착수 전 수행.
