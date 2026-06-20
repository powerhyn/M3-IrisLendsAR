# 핸드오프: 자연스러운 렌즈 피팅 — 톤별 블렌딩 전략 리서치

> 출처: LensSimulator 세션 딥 리서치 (2026-06-11). 22개 소스 → 109개 클레임 추출 → 상위 25개 3표 적대적 검증 **전원 통과(0 킬)**.
> 원문: `LensSimulator/docs/research/natural-lens-fit-blending.md` (커밋 a38e1ca)
> ⚠️ **적용 시점 주의**: 현재 3단계 리팩토링(ADR-0001 경계 도입) 진행 중이며 gpu-render는 "리팩토링 보존" 판정 모듈이다. 본 문서는 **즉시 구현 지시가 아니라 렌즈 품질 트랙의 설계 입력**이다 — 전환 완료 후 W 단위로 반영할 것.

## 0. 한 줄 요약

자연스러움의 핵심은 단일 알파 합성이 아니라 **실물 컬러렌즈의 광학 구조 모사**다: ① 톤별 블렌딩 분기(틴트=투과형 / 불투명=치환형 — 물리적 근거 있음), ② 치환도 100%가 아닌 **부분 커버리지(effective alpha = 요소 불투명도 × 커버리지)**, ③ 본연 홍채 텍스처 투과(show-through)가 "스티커 룩" 실패 모드의 해독제.

## 1. 검증된 물리적 근거 (특허 원문 verbatim 확인, 3표 검증)

| 근거 | 출처 | 함의 |
|---|---|---|
| 컬러렌즈 2분법: 반투명 인핸스먼트 틴트("augment the natural color") vs 불투명("masks the natural iris color"). **틴트는 갈색 홍채를 파랗게 못 만든다** | US6761451 (J&J, 렌즈 프리뷰 장치 — VTO 동일 도메인) | 어두운 홍채 위 틴트 발색 실패는 버그가 아니라 물리 현상. 치환형 블렌딩은 불투명 클래스 전용 |
| 불투명도는 연속 스펙트럼: opaque=광투과 0~50%, translucent=50~85% | US6761451 | 치환은 boolean이 아닌 파라미터형 |
| '내추럴' 렌즈 실물 = 불투명 간헐 패턴(도트/방사). 커버리지: 내추럴 톤 10~30%(선호 ~20%), 어두운 눈 전용 30~80% | US7210778, US6132043 (Wesley-Jessen) | 본연 홍채가 면적 70~90% 비침 |
| **지각 색 변화 = 착색제 불투명도 × 커버리지로 등가 교환 가능** | US6132043 | 셰이더에서 `effective_alpha = element_opacity × coverage` 단일 파라미터 근사 가능 |
| 홍채 전면 단색 불투명 = "very unnatural appearance" (특허 명시). 자연스러움 = 비불투명 간극의 텍스처 투과 | US7210778, US6132043 | flat sticker look의 원인 규명 — 카메라 홍채 텍스처(특히 고주파)를 항상 일정 비율 투과시킬 것 |

주의: 면적 커버리지와 지각 색 변화는 비선형(도트 masking effect) — effective alpha 근사의 지각 충실도는 실기기 육안으로 자체 검증 필요.

## 2. IrisLensSDK 현 구조에의 매핑

### 2.1 BlendMode — 이미 보유한 모드에 "SKU 톤별 선택 규칙"을 부여

| 렌즈 톤 분류 | 권장 모드 (현 enum 기준) | effective alpha | 비고 |
|---|---|---|---|
| 클리어/비저빌리티 틴트 | 합성 생략 (또는 α≤0.05) | — | |
| 반투명 인핸서 틴트 | **LuminanceTintLinear (5)** — canonical default 그대로 | SKU α | 어두운 홍채에서 발색 약화가 물리적으로 올바름. 보정 여부는 제품 정책 (미해결 질문 #1) |
| 내추럴/서클렌즈 | **ColorReplace (7) 계열 + 낮은 α** | **0.2~0.5** (= element_opacity × coverage) | 부분 커버리지 불투명 합성의 균일 근사. 카메라 휘도로 셰이딩 변조 + 홍채 텍스처 투과 유지. **B1 벤치 대기 중인 ColorReplaceLinear의 구체적 역할 부여** |
| 불투명 코스프레 | ColorReplace (7) + 높은 α | 0.7~0.9 | 캐치라이트는 렌즈 색 '위' 레이어로 보존 |

핵심 제안: 호스트가 blend_mode를 직접 고르게 하는 대신(또는 그에 더해), **SKU 메타데이터가 톤 클래스를 선언하고 SDK가 모드+α를 유도**하는 레이어를 두면 "톤에 따라 다른 블렌딩"이 데이터 주도로 된다.

### 2.2 LensSkuMetadata 확장 제안

```cpp
struct LensSkuMetadata {
    // ... 기존 필드 (has_baked_limbal, prefers_crl, prefers_graphic_outline) ...
    LensToneClass tone_class = LensToneClass::Tint;  // tint | natural | opaque | clear
    float coverage = 0.25f;          // 불투명 요소 면적 비율 (natural/opaque)
    float element_opacity = 0.9f;    // 착색 요소 자체 불투명도
    // → effective_alpha = coverage × element_opacity (× edge_fade × pupil_mask)
    // limbal ring 합성 파라미터 (baked가 아닐 때): 위치/페더/강도
    float limbal_pos = 0.92f, limbal_feather = 0.1f, limbal_intensity = 0.4f;
};
```

`detectBakedLimbal`(이미 보유)이 텍스처 분석으로 coverage/limbal 추정값을 오프라인 산출하는 도구로 확장 가능 — LensSimulator의 `analyze_assets.py`가 같은 접근(meanAlpha/outerRadiusPx 자동 추출)으로 42종을 처리한 선례.

### 2.3 이미 갖춘 것 — 리서치가 1차 출처로 정당화

| IrisLensSDK 기존 자산 | 리서치 근거 |
|---|---|
| LuminanceTintLinear (휘도 보존 틴트) | show-through 원리의 저비용 구현 — 특허군이 명시한 자연스러움 메커니즘과 정합 |
| baked 림발 감지 + 림발 정책 (P6-W7) | limbal ring은 1급 요소: Unity HDRP 전용 파라미터 4종(Size Iris/Size Sclera/Fade/Intensity), Chiang & Fyffe(USC)는 물리 굴절의 실시간 대체로 "예술적 가장자리 다크닝 함수" 사용. (a) baked 보존 + (b) 합성 경계 radial darkening 병행이 권장 패턴 |
| avg_iris_luma 실측 default ON (P7-W2) | 노출 적응의 특허 선례 US9224248("언더노출이면 이펙트 휘도를 낮추고 오버노출이면 높임"). 확장 방향: 홍채→장면/피부 ROI 휘도로 렌즈 휘도·WB 매칭 |
| pupil material conditional (P6-W8) | US6508553(최초 렌즈 VTO 특허): 동공 영역도 완전 제외가 아닌 **상수 10% 미세 틴트** — 동공 처리 계수 설계의 검증된 베이스라인 |

### 2.4 검증된 계수 설계 (차용 가치)

- **US6508553 radial ramp** (3표×3 검증): 동공(r<0.4R) 렌즈 기여 상수 10% → 내측 경계 최대 20% → 외곽(1.4R) 0% 선형 페이드. 특허가 이 점진 페이드 자체를 자연스러움 근거로 명시("no color jump or any artifacts"). 1998년 flat-color 시대 값이므로 그대로가 아니라 "낮은 최대 기여 + 동공 미세 틴트 + 외곽 선형 페이드" 패턴으로 차용.
- **ModiFace 계보 (US20070258656→US8660319)**: 색 전이 = RGB 히스토그램 평균 매칭(단순 lerp 아님) + **공간 가중(중심 약화·주변 강화)** + 그라디언트 페더링. 모바일 구현: ROI 평균 휘도/색 1회 → 셰이더 유니폼 (avg_iris_luma 인프라가 이미 이 형태).

## 3. 실패 모드 → 해법 표

| 실패 모드 | 원인 | 해법 |
|---|---|---|
| flat sticker look | 홍채 텍스처 완전 차폐 | effective alpha < 1 + 휘도 변조 (show-through) |
| 어두운 홍채에서 발색 안 됨 | 틴트(투과형)로 어두운 베이스를 못 이김 | 물리적으로 올바른 동작 — SKU를 natural/opaque 클래스로 분류해 치환형 적용 (또는 제품 정책상 보정) |
| 죽은 눈 (하이라이트 소실) | 렌즈 색이 캐치라이트를 덮음 | 캐치라이트는 렌즈 위 분리 레이어 (Unity HDRP 각막/홍채 레이어 분리와 동일 원리) |
| 렌즈 가장자리 하드 에지 / 공막 침범 | radial fade 부재 | 외곽 선형/smoothstep 페이드 → 0 (§2.4) |
| 장면과 따로 노는 인상 | 노출/WB 불일치 | ROI 휘도 기반 노출 매칭 (avg_iris_luma 확장) |

## 4. 실시간 vs 오프라인 구분선 (수치 확정)

- 경로 추적(SynthesEyes/Cycles): 120×80px 1장 5.26초(GTX660) — 모바일 프레임 버짓 대비 **~128배**
- NeRF(EyeNeRF, SIGGRAPH 2022): 800×800 1장 ~30초(V100급) — **~730배**
- → 실시간은 image-based 프래그먼트 셰이더 근사가 유일한 현실적 선택. GAN/diffusion 실시간성은 검증된 클레임 없음(미답변). 현 gpu_lens_renderer 접근이 맞는 방향.

## 5. 주의사항 / 미해결 질문

1. **상용 셰이더 내부는 어디도 비공개** — Banuba는 마케팅 페이지(귀속 주장), ModiFace는 2007 특허 계보 추정. 위 내용은 특허·프로덕션 문서·학술로 검증 가능한 범위.
2. 특허의 '자연스러움'은 출원인 주장(지각 실험 아님). 커버리지 수치는 톤별 상이 → 단일 상수 금지, SKU 파라미터로.
3. **미해결 질문 4건**: ① 틴트의 어두운 홍채 발색 — 물리 충실 vs 판매 전환용 보정(제품 정책), ② 도트 패턴 절차적 재현 vs effective-alpha 균일 근사의 지각 차이(눈 영역 ~100-200px에서), ③ 경쟁 VTO 블랙박스 테스트(demo.tintvto.com 등에 다양한 홍채 톤 입력), ④ 동공 반경 실시간 추정(산동/축동) — MediaPipe는 동공 경계 미제공.

## 6. 출처 (검증 통과분)

- **특허(primary)**: US6761451(J&J 프리뷰), US6132043(Wesley-Jessen 간헐 패턴), US7210778(내추럴 커버리지 10-30%), US6508553(최초 렌즈 VTO, radial ramp), US20070258656/US8660319(ModiFace 색 전이), US9224248(노출 적응), US9443343(삼성+USC 부위별 렌더링)
- **학술(primary)**: Wood et al. ICCV 2015(SynthesEyes), Chiang & Fyffe USC ICT-TR-01-2010, EyeNeRF SIGGRAPH 2022
- **프로덕션 문서(primary)**: Unity HDRP Eye Shader
- **상용(귀속 주장)**: Banuba virtual-contact-lens-try-on
