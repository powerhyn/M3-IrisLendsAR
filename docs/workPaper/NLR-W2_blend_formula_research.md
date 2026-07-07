# NLR-W2 리서치: 렌즈 착색 합성 수식 원점 조사 (검증 완료)

> 방법: deep-research 워크플로 2단 실행 (원본 wf_b3b0c768 87/102 + 한도 중단 → checkpoint 수확 → 연속 wf_d3e6b386 15/15 완주).
> 검증: 클레임당 3표 적대 검증, **21클레임 생존(주요 3-0) → 12 findings 병합**. 검증 기각/미완 0.
> 입력 맥락: NLR-W1 cap 무효 판정 + 문제 재정의("경계 톤 불연속") — `docs/bench/P7-W4/cap_sweep_result.md`.
> 관계: 2026-06-11 리서치(`docs/lenssim-handoff/natural-lens-fit-blending-research.md`)는 톤 클래스/커버리지 축, 본 리서치는 **합성 수식 축** — 상보적.

## 종합 요약 (synthesis 원문)

## AR 콘택트렌즈 틴트 셰이더 — 검증 완료 연구 종합 (16 claims, 3-0 생존 → 12 findings로 병합)

### 핵심 결론: "틴트 강도를 바탕 휘도에 곱하는" 수식은 조사된 어떤 상용 특허·논문에도 없다

현재 문제(tint = lensColor × baseLum × scale → 홍채/흰자 경계 두 톤 분리)의 근본 원인은 **틴트의 '주입량' 자체가 바탕 휘도에 비례**하는 구조다. 검증된 모든 선례(Meta, Shiseido, Chanel, Guo&Sim, Li et al., Jin et al.)는 이 결합을 끊는다: **색(chroma)은 위치와 무관하게 일정 강도로 주입하고, 명암(lightness)은 바탕에서 그대로(또는 국소 정규화 형태로) 통과시킨다.** 그러면 경계 양쪽이 같은 색을 받되 밝기만 바탕을 따라가므로 톤 분리가 구조적으로 사라진다.

### 검증된 3대 수식 계열 (모두 single-pass 프래그먼트 셰이더 구현 가능)

**계열 1 — CIELAB/Oklab 휘도 보존 recolor (Guo&Sim 2009 → Jin 2019 → Shiseido 특허 계보)**
바탕을 Lab로 분해, a*b*만 렌즈색으로 alpha blend, L*은 통과. Shiseido 공식이 가장 셰이더 친화적:
`L' = α·(L_base − μ_base) + μ_lens` (α=1이면 바탕 음영 패턴 100% 유지)
→ **프로젝트가 이미 보유한 avg_iris_luma가 정확히 μ_base 역할.** 홍채 내부는 μ_base=avg_iris_luma, chroma는 `mix(ab_base, ab_lens, w·lensAlpha)`. 검증된 한계: chroma-only로는 어두운 홍채 위에 밝은 렌즈를 못 만든다(Guo&Sim·Jin 모두 L 보정 수단을 별도 추가; 이를 과대 포장한 claim 2건은 refute됨) — 그래서 mean-shift 항(μ_lens − avg_iris_luma)이 필요하다.

**계열 2 — 국소 평균 피벗 양방향 블렌드 (Meta US11069094B1 + Chanel 특허)**
Meta: 제품색에 mid-대역을 screen(~50%, 밝은 곳을 밝힘) → low-대역을 multiply(~50%, 어두운 곳 유지). "밝은 부분은 하이라이트, 어두운 부분은 원래 밝기 근처 유지" — 단방향 곱셈/가산이 아닌 **양방향** 블렌드가 톤 분리 회피의 핵심. Chanel: `R' = ((R − R̄_local)/σ_local)·σ_target + R̄_target` — 국소 통계 정규화. Meta는 여기에 4×8 그리드 국소 luma 통계 기반 s-curve 톤매핑까지 추가(오버레이의 국소 휘도 정규화). 셰이더 1-tap 근사: 바탕 luma가 국소 평균(홍채=avg_iris_luma)보다 위면 screen, 아래면 multiply로 피벗하는 soft-light 변형.

**계열 3 — Kubelka-Munk 반투명 코팅 (Li et al. CVPR 2015)**
`out = R_c + T_c²·base/(1 − R_c·base)` (per-channel, R_c=렌즈 반사색, T_c=투과율)
바탕색이 투과항으로 물리적으로 기여하는 반투명 코팅 모델 — **실물 콘택트렌즈(반투명 안료층)와 구조적으로 동일**하고, closed-form 몇 연산이라 single-pass 비용 최소. 홍채 위든 흰자 위든 같은 물리 법칙이 적용되므로 경계 불연속이 원리적으로 없다.

### 흰자 겹침 처리
검증된 선례는 ModiFace US8725560B2의 **radial fade-out**(중심 불투명 → 방사형 투명 증가 마스크 가중평균, 2009 priority)뿐. 프로젝트 원칙(림발·페이드는 에셋 책임, RGBA PNG 부분 알파 내장)과 정합 — 셰이더는 흰자에서도 동일 수식을 쓰되 에셋 알파+현행 uScleraTintMax cap으로 감쇠하는 현 방향이 선례와 충돌하지 않는다. 실물 서클렌즈 잉크 배치(외곽 어두움) 근거 claim은 검증 미완이라 채택 불가.

### 실기기 벤치 권고 순서
① 계열 3(KM coating) — 물리 정합 최상·비용 최소·구현 최단, ② 계열 1(Oklab mean-shift, avg_iris_luma 직결) — 어두운 홍채→밝은 렌즈 표현력 우수, ③ 계열 2 피벗 블렌드 — ①②가 육안 미달 시. 세 안 모두 "틴트×바탕휘도 곱셈 제거"라는 동일 원리의 다른 구현이므로 A/B 토글로 판정.

## Findings (12건, 전건 3표 검증 통과)

### F1. Meta US11069094B1(AR 메이크업, 2021 등록)은 휘도 비례 곱셈 틴트 대신 양방향 비대칭 블렌드를 쓴다: 탈채도 프레임에서 추출한 mid-range 휘도 대역을 제품 albedo 색 위에 screen(~50%)으로 블렌드해 밝은 부위를 밝히고, 이어 low-range 대역을 multiply(~50%)로 블렌드해 '밝은 부분은 하이라이트하되 어두운 부분은 원래 밝기 근처로 유지'한다 — 밝음/어둠 양방향 처리로 공간 톤 분리를 회피하는 직접 선례.
- 신뢰도: high · 투표: 3-0 (동일 특허 3개 claim group 각각 3-0)
- 출처: https://patents.google.com/patent/US11069094B1/en , https://patents.justia.com/patent/11069094
- 검증 근거: 특허 원문 verbatim 확인(3개 claim group 각 3-0 생존): "screening the albedo color with the first set of frequencies... then blends the first blending output... by multiplying"; "which further highlights the brighter portions while maintaining the darker portions of the image frame at approximately similar brightness levels"; 50% 블렌드 값·mid/low 대역 순서·탈채도(grayscale) 추출 모두 원문 확인. 특허 스스로 기존 직접 색 적용 방식과 대비함.

### F2. 같은 Meta 특허는 RGB 대역 블렌드 후 CIELAB로 변환해 재질별(gloss/glitter/matte/metallic) 셰이딩 모델로 L(lightness)만 수정하고 chroma는 고정한 채 RGB로 복귀한다 — 'chroma는 제품색에서, 공간 음영은 바탕 얼굴 휘도에서'라는 휘도 보존 recolor 구조.
- 신뢰도: high · 투표: 3-0 (2개 claim group 각각 3-0)
- 출처: https://patents.google.com/patent/US11069094B1/en , https://patents.justia.com/patent/11069094
- 검증 근거: 특허 청구항 언어로 확인: "modifying, in the LAB color space, a lightness of the blended color by applying at least one shading model based on the material" + "convert the updated makeup color back to the RGB color space". a/b 채널 수정 기술은 특허 전체에서 미발견(검증자가 표적 검색으로 확인). 주의: 특허 상태는 Expired-Fee Related(기술 공개 정확성에는 무관).

### F3. 같은 Meta 특허는 국소(공간 가변) 톤매핑을 추가 적용한다: 프레임을 1/16 다운샘플 후 4×8 그리드로 분할, 섹션별 평균/최소/최대 luma(log 공간)를 계산해 AR 레이어 픽셀 색을 그 국소 luma 통계 기반 s-curve로 리맵 — 오버레이를 바탕 이미지에 대해 국소 휘도 정규화하는 상용 선례.
- 신뢰도: high · 투표: 3-0
- 출처: https://patents.google.com/patent/US11069094B1/en
- 검증 근거: 특허 원문 verbatim: "divide the downsampled digital video stream into a plurality of separate sections (e.g., a 4 × 8 grid). For each section... determine the average luma and luma ranges... remap the colors of each pixel in the augmented reality layer to an s-curve based on the calculated luma values." 특허 자체가 'localized tone mapping'이라는 용어 사용.

### F4. Shiseido 립스틱 recolor 특허(US20200015575A1, 등록 US11344102B2/US10939742B2 Active)는 립 영역과 목표색을 먼저 CIELAB로 변환해 L을 a*b*에서 분리한 뒤, L 채널에 가산형 mean-shift를 적용한다: L' = α·(L_lip − μ_lip) + μ_color (0≤α≤1). α가 커버리지를 제어하며 α=1이면 바탕 립 음영 패턴이 전부 유지된다 — 바탕 휘도에 틴트를 곱하는 항이 전혀 없는, avg_iris_luma를 μ_base로 직결할 수 있는 공식.
- 신뢰도: high · 투표: 3-0 (2개 claim group 각각 3-0)
- 출처: https://patents.google.com/patent/US20200015575A1/en , https://patents.google.com/patent/US10939742B2/en
- 검증 근거: 등록본 B2 문단 0173에서 α 포함 정확한 식 확인(A1 페이지는 OCR로 α 글리프 누락): "I_Lip^L' = α(I_Lip^L − μ_Lip^L) + μ_Color^L" + verbatim "zero α fully covers all lip patterns underneath, while unit α retains all lip patterns". a/b는 공간 마스크 lerp로 별도 대체. 주의: L 평균은 목표색 쪽으로 이동시키므로 '바탕 휘도 완전 무변경'은 아니고 '음영 편차 보존' 구조.

### F5. Guo & Sim (CVPR 2009, 'Digital Face Makeup by Example')은 CIELAB에서 L*을 명암 레이어, a*b*를 색 레이어로 분리(분리 성능·지각 균일성이 채택 이유)하고, 색 전이는 chroma 채널만의 alpha blend Rc(p)=(1−γ)Ic(p)+γEc(p)로 수행한다 — 색 전이 단계에서 바탕 L* 음영이 그대로 통과하므로 recolor 영역이 바탕 음영을 연속적으로 상속, 휘도 비례 틴트의 톤 불연속을 원천 회피하는 계보의 시조.
- 신뢰도: high · 투표: 3-0
- 출처: http://vigir.missouri.edu/~gdesouza/Research/Conference_CDs/IEEE_CVPR_2009/data/papers/0145.pdf
- 검증 근거: 원문 PDF 전체 열람 검증: §3.2 verbatim "The L* channel is considered as lightness layer and a*, b* channel the color layer. We choose the CIELAB colorspace because it performs better than other color spaces in separating lightness from color, and it is approximately perceptually uniform" + §3.4 Eq.6. 중요 caveat: 전체 파이프라인은 skin detail 대체(§3.3)와 L* gradient 전이(§3.5)로 L*도 수정 — chroma-only가 전부라는 과대 주장 claim 2건은 별도로 refute됨(3-0 kill).

### F6. Guo & Sim은 L* 구조(음영) 레이어의 직접 복사·블렌드를 명시적으로 거부하고, gradient domain에서 선택 규칙 ∇Rs(p)=∇Es(p) if β(p)‖∇Es(p)‖>‖∇Is(p)‖ else ∇Is(p)로 하이라이트/음영만 전이한 뒤 Poisson 방정식(Dirichlet 경계, Gauss-Seidel SOR)으로 재구성한다 — 바탕 조명을 보존하며 이음매 없는 결과를 얻는 gradient-domain shading transfer의 근거 선례(단, 실시간 single-pass에는 비용상 부적합).
- 신뢰도: high · 투표: 3-0
- 출처: https://www.cse.iitd.ac.in/~pkalra/col783/OLD/assignment2/face_makeup_cvpr09_lowres.pdf
- 검증 근거: 원문 PDF 검증: "we can neither directly copy Es over Is nor blend them. Instead, we adapt a gradient-based editing method... Gradient-based editing can preserve the illumination of I, transfer the highlight and shading effects, and meanwhile yield smooth result" + Eq.7·솔버 상세 verbatim 일치. 전제 조건: 예시(E) 이미지 조명이 근사 균일해야 함.

### F7. Jin et al. (arXiv 1907.03398, IEEE Access 2019)은 Guo&Sim 계보의 canonical 3-레이어 구조(structure=edge-preserving 스무딩된 휘도, detail=휘도−structure, color=a,b chroma)를 재확인하며, 색 전이는 휘도 항이 전혀 없는 chroma alpha blend O_ab(p)=(1−α)I_ab(p)+α·R_ab(p), α=0.95로만 수행한다. 단 어두운(검정)·밝은(흰) 메이크업은 chroma-only로 재현 불가라 별도 illumination transfer(Eq.4, β=30)를 추가 — chroma-only 전이의 표현력 한계를 명시한 문헌.
- 신뢰도: high · 투표: 3-0 (2개 claim group 각각 3-0)
- 출처: https://ar5iv.labs.arxiv.org/html/1907.03398
- 검증 근거: ar5iv 원문 verbatim 검증(2개 claim group 각 3-0): 3-레이어 정의와 Eq.2(α=0.95, 피부 영역 C₁) 정확 일치, 해당 식에 휘도 가중 없음을 표적 검색으로 확인. 검증자 명시 한계: "chroma alpha-blend is the color-transfer equation but not the paper's entire recolor pipeline, and lightness-shifting tints still require an L-channel mechanism" — 어두운 홍채 위 밝은 렌즈에는 L 수단 필요.

### F8. Li et al. (CVPR 2015, 'Simulating Makeup through Physics-based Manipulation of Intrinsic Image Layers')는 이미지를 I = A·D + S(알베도·확산음영·스페큘러)로 고유 분해한 뒤 화장 색 변경을 알베도에만 적용하고 대상의 음영·조명을 보존한다(아이섀도·블러셔는 Table 1에서 알베도만 변경). 바탕의 연속 음영 필드 D가 무변경 통과하므로 조명 기인 톤 불연속이 원리적으로 생기지 않는 구조.
- 신뢰도: high · 투표: 3-0
- 출처: https://openaccess.thecvf.com/content_cvpr_2015/papers/Li_Simulating_Makeup_Through_2015_CVPR_paper.pdf
- 검증 근거: 원문 PDF 전체 검증: Eq.1 verbatim + Table 1(eye shadow/blush는 Albedo만 체크) + 각주 4 "blush and eye shadow only change albedo" + abstract "preserving the appearance characteristics and lighting conditions of the target face". caveat: 오프라인 방법(600×400에 ~1.5s, Retinex 분해) — 원리 선례이지 실시간 구현이 아님.

### F9. 같은 Li et al. 논문의 알베도 recolor 수식은 Kubelka-Munk 기반 반투명 코팅 합성 A_M = R_c + T_c²·R_B/(1−R_c·R_B) (R_B=바탕 알베도)이다: 새 색 = 화장품 반사색 + 바탕색이 투과색 T_c를 두 번 통과해 비쳐 보이는 항 + 상호반사 급수 — 순수 색 치환이 아니라 바탕이 물리적으로 기여하는 모델로, 반투명 콘택트렌즈 착색과 기하 구조가 동일하며 closed-form이라 단일 패스 셰이더에 그대로 옮길 수 있다.
- 신뢰도: high · 투표: 3-0
- 출처: https://openaccess.thecvf.com/content_cvpr_2015/papers/Li_Simulating_Makeup_Through_2015_CVPR_paper.pdf
- 검증 근거: 원문 §3.2.1 Eqs.6-7 verbatim 확인, R_c/T_c는 KM 모델(Kubelka&Munk 1931)에서 유도, 기하급수 합 형태가 고전 KM 층 합성 공식과 일치. 2025년 arXiv 2507.07333이 상용 파운데이션 VTO에 KM을 사용해 현재성도 방증. caveat: 논문은 고유 분해 후 알베도 공간에서 적용(스페큘러 별도) — raw RGB에 직접 적용은 번안이며, 렌즈 유추('구조 동일')는 검증자가 타당하다고 본 claimant의 유추.

### F10. Chanel의 메이크업 렌더링 특허(US20200170383A1 → US11594071B2 등록·Active)는 픽셀을 공간 영역(그리드 셀)별 통계로 Reinhard식 전이한다: Ri2 = ((Ri1 − R̄1)/σ1)·σ2 + R̄2 — 바탕의 국소 평균·표준편차로 정규화 후 목표 통계로 재스케일해 국소 음영/대비 패턴을 보존하면서 색 통계만 이동. two-tone recolor 불연속에 대한 '국소 휘도(통계) 정규화' 해법의 구체적 상용 특허 선례.
- 신뢰도: high · 투표: 3-0
- 출처: https://patents.google.com/patent/US20200170383A1/en
- 검증 근거: 특허 원문 verbatim: Ri1/Ri2, 공간 영역 평균 R̄1/R̄2, 표준편차 σ1/σ2 정의와 식 정확 일치. 주파수 분해된 secondary image 위 그리드 셀 단위 통계 적용도 확인. caveat: 레퍼런스 DB 기반 사진 시뮬레이션(목표 통계를 유사 인물 DB에서 예측)이지 실시간 AR 셰이더가 아님 — 정규화 원리의 선례이며, per-RGB 채널 적용이라 '휘도 정규화'는 해석적 라벨.

### F11. ModiFace US8725560B2(priority 2009-02-02, 2014 등록, 현 L'Oréal 소유)는 립 색을 '중심은 진하고 불투명, 바깥 방사 방향으로 투명도가 증가하는 radial-gradient 반투명 색 마스크'를 원본 이미지와 가중평균해 적용한다 — recolor 오버레이의 radial fade-out에 대한 최초기 상용 VTO 선례로, 렌즈-흰자 겹침부 감쇠(에셋 알파 + 경계 fade) 관행의 근거.
- 신뢰도: high · 투표: 3-0
- 출처: https://patents.google.com/patent/US8725560B2/en
- 검증 근거: 원문 HTML curl+grep으로 verbatim 확인: "a radial-gradient translucent colored mask consisting of intense less translucent colors in the center and having increasing translucency in the outward radial directions may be weighted-averaged... within the boundary of the lip box" + 다음 문단의 "weighted-averaging of the original image with the colored mask". 원 양수인 Modiface Inc·발명자 Parham Aarabi·priority 2009 메타데이터 일치. caveat: 실시간 AR이 아닌 사진 기반이며 'may be' 선택적 실시예 언어.

### F12. Banuba의 상용 렌즈 VTO는 바탕 적응형 recolor를 표방한다: '사용자 원래 눈 색과 무관하게 자연스럽고, AI가 원 홍채 톤에 정확히 적응한다'고 주장 — 렌즈 도메인에서 바탕 홍채 외관에 조건화된 틴트를 상용 제품이 지향한다는 방증. 단 수식·메커니즘은 일절 미공개(마케팅 수준 근거).
- 신뢰도: medium · 투표: 3-0
- 출처: https://www.banuba.com/virtual-contact-lens-try-on
- 검증 근거: 라이브 제품 페이지에서 verbatim 확인: "They look natural no matter the user's original eye color... the AI accurately adapts to the original iris tone." 2023-06-28 보도자료의 "Realistic interaction with lighting and natural eye color"가 '평면 오버레이 아님' 해석을 보강. 검증자 명시: 벤더 마케팅이며 실제 바탕 조건화 수식의 증거로 승격 금지 — '주장의 존재'만 검증됨.

## Caveats (synthesis 원문)

① 도메인 전이 유추: 검증된 수식 선례는 전부 립스틱/피부 메이크업 도메인이다(Meta·Shiseido·Chanel·ModiFace·Guo&Sim·Li·Jin). 콘택트렌즈 도메인 직접 증거는 Banuba뿐이며 그것은 수식 없는 마케팅 주장 — '입술도 명암 큰 곡면'이라는 유사성에 기반한 합리적 유추이지 렌즈 전용 검증이 아니다. ② chroma-only의 표현력 한계(중요): 바탕 L 통과형(계열 1)은 어두운 홍채 위에 더 밝은 렌즈색을 만들 수 없다 — Guo&Sim·Jin 모두 이 이유로 별도 L 수정 수단을 추가했고, 'chroma-only로 충분'이라는 취지의 claim 2건은 적대 검증에서 3-0으로 refute됐다. 밝은 렌즈 표현에는 Shiseido식 mean-shift 항이나 screen 계열 항이 필요하며, 여기서 avg_iris_luma가 μ_base로 쓰인다. ③ IP: Meta US11069094B1은 Expired-Fee Related로 확인됐으나 Shiseido US11344102B2·Chanel US11594071B2는 Active 등록 특허다. 특정 특허의 청구 방식을 그대로 제품화할 경우 검토 여지가 있다(본 조사는 기술 선례 조사이며 법률 자문 아님). ④ 실시간성: Chanel(레퍼런스 DB 기반)·Li(오프라인 ~1.5s)·Guo&Sim(Poisson 솔버)은 원리 선례이지 그대로 single-pass 이식 대상이 아니다. 셰이더 번안(KM을 raw RGB에, mean-shift를 Oklab 근사에)은 본 종합의 제안이지 검증된 사실이 아니며, 프로젝트 원칙대로 실기기 육안 A/B로 판정해야 한다. ⑤ 미완 검증 5건 사용 금지: Google US11250632(shadows/mid-tones/highlights 3-밴드 분해 + 블렌드 시퀀스 + 정규화 불변성) 관련 3건과 실물 서클렌즈 잉크 설계(간헐 도트 패턴, 외곽 어두운 잉크 집중) 2건은 검증 votes가 미완(INCOMPLETE)으로 checkpoint의 pending_claims에 남아 있다 — 본 보고서에서 제외했고 근거로 인용하면 안 된다. ⑥ 합성 프롬프트가 12,000자에서 절단되어 있었으나, harvest JSON(102 vote records)에서 16개 생존 claim 전체(각 3-0)를 재구성해 종합했다 — 절단으로 인한 누락은 없다.

> ⚠️ **정정 (2026-07-03, caveat ⑤·아래 미완 2줄은 스테일)**: 종합 프롬프트의 구 지시문을 따라 남은 문구다. 실제로는 연속 워크플로(wf_d3e6b386)에서 해당 5건 — **Google US11250632B2** 3건(3-밴드 분해 + 블렌드 시퀀스 + 사전 정규화) + **실물 서클렌즈 잉크 설계 US6,827,440** 2건(간헐 도트 패턴 투과, 최암 잉크 외곽 최대 밀도) — **전부 재검증 통과(반박 0)**. 인용 가능하며, US6,827,440은 "흰자 쪽 밝은 틴트는 실물과 반대"의 물리 근거로 확정.

## Open Questions

- ~~Google US11250632B2 3개 claim 검증 미완~~ → 상단 정정 참조 (검증 통과). 모바일 실시간(Google 저자) 특허라 본 프로젝트와 가장 가까운 선례 — 3-밴드 분해는 후보 G의 확장 방향으로 유효
- ~~실물 서클렌즈 잉크 설계 특허 2건 검증 미완~~ → 상단 정정 참조 (검증 통과, US6,827,440)
- Oklab vs CIELAB의 모바일 GPU 변환 비용·품질 비교는 검증된 claim이 없음 — 계열 1 채택 시 셰이더에서 어느 근사를 쓸지는 실측 필요
- 눈물막 모사(흰자 겹침부 채도/명도 가라앉힘, specular 처리) 관행은 검증된 선례를 확보하지 못함
- avg_iris_luma 단일 스칼라만으로 충분한가 vs Meta 4×8 그리드/Chanel 셀 통계처럼 흰자 쪽 국소 평균이 별도로 필요한가 — 현 P7-W4 uScleraTintMax cap과의 조합을 실기기 A/B로 판정 필요
- 합의된 세 계열(KM coating / Oklab mean-shift / 피벗 screen-multiply)의 실기기 육안 비교 벤치가 다음 단계 — 세 안 모두 '틴트×바탕휘도 곱 제거'라는 동일 원리의 구현 변형

## 다음 단계 (프로세스)

1. 본 리포트 + 셰이더 후보 3계열을 **Codex 교차 R1**으로 검토 (물리 가정 + visibility budget 명시 계산 포함 — cap 실패 재발 방지)
2. §5 확정 후 3계열 A/B/C 토글 구현 → 실기기 육안 벤치 (blendModeEntries 확장 or benchCombos 재정의)
3. 채택 수식 → canonical 승격 검토는 NLR-W6