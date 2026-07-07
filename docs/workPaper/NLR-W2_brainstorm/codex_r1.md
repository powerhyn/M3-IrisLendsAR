# NLR-W2 셰이더 수식 후보 교차 검토 R1 (Codex)

## 0. 읽은 맥락과 전제

- 리서치: `docs/workPaper/NLR-W2_blend_formula_research.md`
  - 결론: 현행 `TintLinearV2`의 핵심 결함은 `tint = lensColor * baseLum * scale` 구조다. 틴트 주입량이 바탕 휘도에 비례하므로 홍채/흰자 경계에서 같은 렌즈가 두 톤으로 갈라진다.
  - 후보 3계열: KM 반투명 코팅, Lab/Oklab mean-shift, 국소 피벗 양방향 blend.
- 직전 실패: `docs/bench/P7-W4/cap_sweep_result.md`
  - `uScleraTintMax` cap은 무효. cap은 이미 밝은 tint target을 몇 % 줄이는 축이라 실제 `finalAlpha`가 작을 때 1~3% 변화로 묻혔다.
  - 문제는 "흰자 밝음" 자체가 아니라 "경계 톤 불연속"이다.
- 현행 셰이더: `cpp/src/gpu/shader_sources.cpp`
  - `blendTintLinearV2()`는 `baseL`, `lum`, `scale = clamp(0.85 / uAvgIrisLum, 0.8, 7.0)`, `tintMul = min(lum * scale, uScleraTintMax)`, `tinted = lensL * tintMul`, `mix(baseL, tinted, opacity)` 구조다.
  - `applyLens()`는 `texture(uLensTexture, lensCoord)`를 분기 전에 무조건 fetch하고, `finalAlpha = lens.a * uOpacity * edgeAlpha * eyelidMask * scleraFade * renderAlpha`를 만든 뒤 blend 함수에 넘긴다.
  - 따라서 후보 함수는 `opacity`를 이 `finalAlpha`로 그대로 받고, 함수 내부는 ALU만 추가하면 GLES 3.1/Adreno의 "분기 내 조건부 texture fetch 금지" 제약과 충돌하지 않는다.

이 문서의 수치 계산은 단순화를 위해 neutral scalar linear 값으로 한다. 즉 `baseL = vec3(lum)`, `lensL = vec3(0.75)`, `finalAlpha = 0.35`, detail reinject/반사/contact shadow는 제외한다. `uAvgIrisLum`은 홍채 대표 lum과 같다고 둔다. 현행 baseline은 체크인 코드의 기본 cap `uScleraTintMax = 1.275` 기준이다. 실제 흰자 픽셀에서 edge feather/veto가 더 곱해져 `finalAlpha`가 0.35보다 작으면 아래 변화량은 거의 선형으로 같이 줄어든다.

## 1. 후보 A: KM 반투명 코팅

### GLSL 적응식

핵심은 full coating target을 만든 뒤 기존 `finalAlpha`로 합성하는 것이다. `R_c`와 `T_c`를 별도 uniform으로 받지 않기 때문에 lens texture 색으로부터 보수적으로 추정한다. 아래 추정은 full film이 흰 배경 위에서 lens texture 색으로 보이도록 맞춘다.

```glsl
vec3 blendKMCoating(vec3 base, vec3 blend, float opacity) {
    vec3 baseL = toLinearFast(base);
    vec3 lensL = clamp(toLinearFast(blend), vec3(0.0), vec3(0.98));

    // Rc: 안료층 자체 반사. 0.80은 R1 bench 상수.
    // Tc2는 base=1.0일 때 coated ~= lensL가 되도록 역산:
    // lensL = Rc + Tc2 / (1 - Rc)  =>  Tc2 = (lensL - Rc) * (1 - Rc)
    vec3 Rc = clamp(lensL * 0.80, vec3(0.0), vec3(0.95));
    vec3 Tc2 = max(lensL - Rc, vec3(0.0)) * (vec3(1.0) - Rc);

    vec3 denom = max(vec3(1.0) - Rc * baseL, vec3(1e-4));
    vec3 coated = Rc + (Tc2 * baseL) / denom;

    vec3 outL = mix(baseL, clamp(coated, vec3(0.0), vec3(1.0)), clamp(opacity, 0.0, 1.0));
    return toSRGBFast(outL);
}
```

특징:
- `uAvgIrisLum`은 직접 쓰지 않는다. 그래도 `lens.a`, edge feather, sclera veto, blink alpha는 `opacity`에 이미 포함되므로 기존 체인을 재사용한다.
- `uScleraTintMax`는 후보 함수에서 쓰지 않는다. cap은 baseline/롤백용 uniform으로만 남긴다.
- ALU 비용은 낮다. Oklab 변환보다 훨씬 싸고 texture fetch 추가가 없다.

### Visibility budget

현행 baseline:

| 픽셀 | `uAvgIrisLum` | 현행 target | 현행 final linear |
|---|---:|---:|---:|
| 어두운 홍채 `lum=0.07` | 0.07 | 0.3675 | 0.1741 |
| 중간 홍채 `lum=0.25` | 0.25 | 0.6375 | 0.3856 |
| 흰자 `lum=0.70` | 0.07/0.25 | 0.9563 | 0.7897 |

KM 후보:

| 픽셀 | KM target | KM final linear | 현행 대비 변화 |
|---|---:|---:|---:|
| 어두운 홍채 | 0.6044 | 0.2570 | +8.29%p / +47.6% |
| 중간 홍채 | 0.6176 | 0.3787 | -0.69%p / -1.8% |
| 흰자 | 0.6724 | 0.6903 | -9.93%p / -12.6% |

판정:
- 어두운 홍채와 흰자에서 변화가 충분히 크다. cap 실패처럼 1~3%에 묻히는 축이 아니다.
- 중간 홍채는 거의 유지된다. 현재 중간 홍채가 이미 자연스럽고 흰자 경계만 거슬리는 SKU에는 장점이다.
- 실제 흰자 `finalAlpha`가 0.10까지 낮아지면 -9.93%p는 약 -2.84%p로 줄어든다. 그래도 세 후보 중 흰자 변화 보존력이 가장 낫다.

## 2. 후보 B: Oklab 가산 mean-shift + 흰자 guard

### GLSL 적응식

리서치의 Shiseido 계열 원식 `L' = (L_base - mu_base) + mu_lens`를 그대로 쓰면, `mu_base`가 홍채 평균인 상태에서 흰자 `L_base`가 들어올 때 흰자가 과상승한다. 이 프로젝트의 문제는 렌즈 마스크가 흰자에 겹치는 경우이므로, single-pass 적응식에는 `min(baseLightness, irisPivot)` guard가 필요하다.

```glsl
vec3 linearSrgbToOklab(vec3 c) {
    float l = 0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b;
    float m = 0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b;
    float s = 0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b;

    vec3 lms = pow(max(vec3(l, m, s), vec3(1e-6)), vec3(1.0 / 3.0));
    return vec3(
        0.2104542553 * lms.x + 0.7936177850 * lms.y - 0.0040720468 * lms.z,
        1.9779984951 * lms.x - 2.4285922050 * lms.y + 0.4505937099 * lms.z,
        0.0259040371 * lms.x + 0.7827717662 * lms.y - 0.8086757660 * lms.z
    );
}

vec3 oklabToLinearSrgb(vec3 c) {
    float l_ = c.x + 0.3963377774 * c.y + 0.2158037573 * c.z;
    float m_ = c.x - 0.1055613458 * c.y - 0.0638541728 * c.z;
    float s_ = c.x - 0.0894841775 * c.y - 1.2914855480 * c.z;

    float l = l_ * l_ * l_;
    float m = m_ * m_ * m_;
    float s = s_ * s_ * s_;

    return vec3(
         4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    );
}

vec3 blendOklabMeanShiftGuard(vec3 base, vec3 blend, float opacity) {
    vec3 baseL = toLinearFast(base);
    vec3 lensL = clamp(toLinearFast(blend), vec3(0.0), vec3(1.0));

    vec3 baseOk = linearSrgbToOklab(baseL);
    vec3 lensOk = linearSrgbToOklab(lensL);

    // uAvgIrisLum은 linear Rec.709 luma이므로 neutral gray의 Oklab L 근사로 변환한다.
    float irisPivotOk = pow(clamp(uAvgIrisLum, 1e-4, 1.0), 1.0 / 3.0);

    // 원식 그대로: lensOk.x + (baseOk.x - irisPivotOk)
    // 프로젝트 적응: 흰자처럼 pivot보다 밝은 픽셀은 mean-shift 상승분을 차단한다.
    float guardedBaseL = min(baseOk.x, irisPivotOk);
    float targetOkL = clamp(lensOk.x + (guardedBaseL - irisPivotOk), 0.0, 1.0);

    vec3 targetL = oklabToLinearSrgb(vec3(targetOkL, lensOk.yz));
    vec3 outL = mix(baseL, clamp(targetL, vec3(0.0), vec3(1.0)), clamp(opacity, 0.0, 1.0));
    return toSRGBFast(outL);
}
```

중요:
- guard 없는 raw mean-shift는 흰자에서 실패한다. 대표값에서 `target = clamp(0.70 + 0.75 - avg, 0, 1) = 1.0`이 되고, final은 `0.805`로 현행 `0.7897` 대비 +1.5%p뿐이다. 방향도 흰자를 더 밝게 하는 쪽이라 문제 정의와 맞지 않는다.
- 따라서 이 계열을 벤치하려면 "Oklab mean-shift"가 아니라 "Oklab mean-shift + 흰자 guard"를 후보로 올려야 한다.

### Visibility budget

neutral scalar 근사에서는 guard가 걸린 target이 홍채/흰자 모두 0.75가 된다.

| 픽셀 | 후보 target | 후보 final linear | 현행 대비 변화 |
|---|---:|---:|---:|
| 어두운 홍채 | 0.7500 | 0.3080 | +13.39%p / +76.9% |
| 중간 홍채 | 0.7500 | 0.4250 | +3.94%p / +10.2% |
| 흰자 | 0.7500 | 0.7175 | -7.22%p / -9.1% |

판정:
- 세 대표 지점 모두 side-by-side에서 보일 가능성이 높다.
- 어두운 홍채 발색을 가장 크게 끌어올린다. 밝은 렌즈 SKU에서 장점이지만, 자연 SKU에서는 "렌즈가 덜 녹고 올라오는" 부작용을 봐야 한다.
- 실제 흰자 `finalAlpha=0.10`이면 흰자 변화는 약 -2.06%p로 줄어든다. alpha가 낮은 외곽에서는 KM보다 묻힐 가능성이 크다.

## 3. 후보 C: 국소 피벗 양방향 blend

### GLSL 적응식

Meta 계열을 single-tap으로 줄인 형태다. 피벗은 `uAvgIrisLum`이고, 피벗 이하에서는 multiply 쪽으로 음영을 보존하고, 피벗 이상에서는 screen 쪽으로 하이라이트를 허용한다. 다만 full screen은 흰자 변화가 너무 작으므로 50% screen으로 제한한다.

```glsl
vec3 blendPivotBiDir(vec3 base, vec3 blend, float opacity) {
    vec3 baseL = toLinearFast(base);
    vec3 lensL = clamp(toLinearFast(blend), vec3(0.0), vec3(1.0));

    float lum = dot(baseL, LUMA_709_LENS);
    float pivot = clamp(uAvgIrisLum, 0.03, 0.80);

    // Low side: 피벗보다 어두운 곳은 multiply 성분으로 가라앉힌다.
    // lum==pivot이면 lensL, lum<pivot이면 lensL보다 어두워진다.
    float lowRatio = clamp(lum / pivot, 0.0, 1.0);
    vec3 multiplyHalf = mix(lensL, lensL * lowRatio, 0.50);

    // High side: 밝은 곳은 screen을 허용하되 50%만 섞어 과한 흰자 부양을 막는다.
    vec3 screened = vec3(1.0) - (vec3(1.0) - baseL) * (vec3(1.0) - lensL);
    vec3 screenHalf = mix(lensL, screened, 0.50);

    float highEdge = max(pivot + 0.08, 0.65);
    float highW = smoothstep(pivot, highEdge, lum);
    vec3 targetL = mix(multiplyHalf, screenHalf, highW);

    vec3 outL = mix(baseL, clamp(targetL, vec3(0.0), vec3(1.0)), clamp(opacity, 0.0, 1.0));
    return toSRGBFast(outL);
}
```

특징:
- Oklab 변환이 없어 비용은 KM 다음으로 낮다.
- screen을 남기므로 흰자 위 target은 KM/Oklab보다 높다. 즉 물리적으로는 "밝은 배경의 투과광"을 보존하지만, 이번 문제인 흰자 링 제거에는 약할 수 있다.

### Visibility budget

대표값에서 홍채 픽셀은 `lum == pivot`이라 target이 0.75가 된다. 흰자는 high side로 가며 `screen(0.70, 0.75) = 0.925`, 50% screen target은 `0.8375`다.

| 픽셀 | 후보 target | 후보 final linear | 현행 대비 변화 |
|---|---:|---:|---:|
| 어두운 홍채 | 0.7500 | 0.3080 | +13.39%p / +76.9% |
| 중간 홍채 | 0.7500 | 0.4250 | +3.94%p / +10.2% |
| 흰자 | 0.8375 | 0.7481 | -4.16%p / -5.3% |

판정:
- 홍채 쪽 변화는 충분하다.
- 흰자 변화는 세 후보 중 가장 작다. `finalAlpha=0.35`에서는 보일 수 있지만, 실제 overlap alpha가 0.10이면 약 -1.19%p라 cap 실패와 같은 비가시 영역으로 내려간다.
- 따라서 1차 후보라기보다 KM/Oklab이 너무 렌즈를 칠한 듯 보일 때의 보수적 대안이다.

## 4. 물리 정합성 비교

실물 반투명 컬러렌즈의 흰자 겹침은 "바탕 휘도에 비례해 안료가 더 발광"하는 현상이 아니다. 투명 폴리머와 인쇄 안료층이 흰자 위에 올라간 상태라서, 흰 배경이 투과되어 더 밝고 덜 포화되어 보일 수는 있다. 하지만 같은 인쇄 dot/edge는 같은 반사·투과 특성을 유지하며, 겹침 강도는 주로 인쇄 밀도, 렌즈 에셋 alpha, radial fade, 실제 위치 오차, 눈물막/조명에 의해 정해진다.

현행 `TintLinearV2`는 이 물리와 어긋난다. 대표값에서 같은 lensL=0.75인데 full target이 어두운 홍채 0.3675, 중간 홍채 0.6375, 흰자 0.9563으로 갈라진다. 즉 안료층이 흰자 위에서 별도 고휘도 색으로 재해석된다.

후보별 비교:

- KM 반투명 코팅: 가장 물리 정합성이 높다. 같은 coating law가 홍채/흰자에 적용되고, 흰 배경 위 full film이 lens texture 색에 수렴하도록 `Tc2`를 잡았기 때문에 흰자 target도 bounded된다. 실제 반투명 인쇄층의 "흰 배경에서는 조금 밝지만 lens 색 이상으로 과발광하지 않음"과 가장 가깝다.
- Oklab mean-shift + guard: 물리 모델은 아니고 지각적 recolor다. guard를 넣으면 같은 lens chroma/lightness를 안정적으로 주입하므로 톤 불연속은 잘 줄인다. 다만 흰자 위에서도 target을 lens 색 근처로 붙들기 때문에, 실제 흰 배경 투과로 약간 더 밝아지는 느낌은 덜하다. 자연성은 SKU/alpha에 민감하다.
- 국소 피벗 양방향 blend: 실물 광학보다는 AR 메이크업 톤매핑에 가깝다. high side screen이 흰 배경 투과감을 일부 보존해 물리 직관과 완전히 반대는 아니지만, 이번 실패 지점인 흰자 high-luma target을 충분히 낮추지 못할 위험이 있다.

## 5. 구현 및 벤치 순서 권고

리서치의 "KM 우선" 권고에 동의한다. 단, raw `R_c=lensL`, `T_c=1-lensL` 같은 직접 대입이 아니라 위의 white-back constraint 버전으로 시작해야 한다.

권고 순서:

1. KM 반투명 코팅
   - 이유: 비용 최저, texture fetch 추가 없음, `finalAlpha` 체인 재사용, 물리 정합성 최상.
   - visibility도 어두운 홍채 +8.29%p, 흰자 -9.93%p라 cap 실패보다 훨씬 크다.
   - 중간 홍채 변화가 작아서 기존 자연도 회귀 위험도 상대적으로 낮다.
2. Oklab mean-shift + guard
   - 이유: 밝은 렌즈의 어두운 홍채 발색을 가장 확실히 올린다.
   - 단점: Oklab ALU 비용과 "칠해진 느낌" 리스크. guard 없는 원식은 흰자에서 부적합하므로 벤치 후보에서 제외한다.
3. 국소 피벗 양방향 blend
   - 이유: 비용은 낮고 보수적인 look을 줄 수 있다.
   - 단점: 흰자 변화량이 -4.16%p뿐이라 실제 alpha가 낮으면 다시 비가시가 될 수 있다. KM/Oklab이 과하면 대안으로 본다.

벤치 구성:

- 1차 A/B/C/D:
  - A: 현행 `TintLinearV2` ID 5, cap 기본 1.275
  - B: KM coating
  - C: Oklab mean-shift + guard
  - D: Pivot bidirectional
- 고정 조건:
  - `uScleraProtect=ON`
  - `uScleraVetoMode`는 현재 production/default 한 값으로 고정. 수식 판정 전에 veto까지 같이 흔들지 않는다.
  - `lum:meas` 고정. `lum:fb`는 2차 확인 축이다.
  - cap sweep은 하지 않는다. 후보 수식은 cap을 쓰지 않으며, cap은 current control의 일부로만 기록한다.
- 구현 슬롯:
  - throwaway 벤치 브랜치라면 deprecated ID 3/4/6을 임시로 B/C/D에 매핑하는 것이 가장 빠르다. 단, 이 상태를 제품 브랜치에 남기면 기존 "3/4/6은 TintLinearV2 fallback" 문서/테스트와 충돌한다.
  - 채택 후보를 오래 유지할 계획이면 새 ID 8/9/10을 C/C++/Java/Kotlin enum과 demo `blendModeEntries`까지 동기화하는 쪽이 맞다.
- 합격 기준:
  - 같은 SKU/조명에서 흰자 경계가 current 대비 명확히 한 톤으로 붙어야 한다.
  - 어두운 홍채에서 렌즈색이 갑자기 opaque하게 떠 보이면 감점한다.
  - 후보 간 차이가 애매하면 흰자 overlap alpha가 낮아 visibility budget이 무너진 것이므로, 더 높은 alpha/edge 구간 캡처를 추가로 확보해야 한다.

## 6. R1 결론

1차 구현은 KM coating을 권한다. 이 후보가 현 문제 정의에 가장 직접적으로 맞는다. 같은 반투명층 법칙을 홍채와 흰자에 적용하고, 흰자 target을 현행 0.9563에서 0.6724로 내리므로 대표값에서 최종 출력이 -9.93%p 변한다. cap 실패의 1~3% 비가시 축과 다르다.

Oklab mean-shift는 반드시 흰자 guard를 포함한 형태로만 벤치해야 한다. guard 없는 원식은 흰자에서 target이 1.0으로 clamp되어 문제를 개선하지 못한다. Pivot bidirectional은 유효한 3순위지만, 흰자 변화량이 작아 실제 edge/veto alpha가 낮으면 다시 묻힐 가능성이 높다.
