# P6-W4 Phase A 이슈 검토 답변

작성일: 2026-05-18

검토 대상:
- [phase_a_issue.md](/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/docs/workPaper/P6-W4_brainstorm/phase_a_issue.md)
- [shader_sources.cpp](/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/cpp/src/gpu/shader_sources.cpp:926)
- [gpu_lens_renderer.cpp](/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/cpp/src/gpu/gpu_lens_renderer.cpp:396)
- [P6-W3_env_reflection_scaffold.md](/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/docs/workPaper/P6-W3_env_reflection_scaffold.md:1)

## Q1. Fresnel + edgeAlpha 외곽 충돌 진단이 정확한가? 다른 후보 원인 있나?

명시적 답변:

**대체로 정확하다. 다만 "유일 원인"이라기보다 "가장 강한 1차 원인"으로 보는 게 맞다.**

근거:

1. 현재 반사 가산식은 `reflection * fresnel * uReflectionIntensity * renderMask`이고, `renderMask = finalAlpha`이며 `finalAlpha = lens.a * uOpacity * edgeAlpha * eyelidMask`다. 즉 Fresnel이 커지는 외곽에서 `edgeAlpha`가 동시에 0으로 가기 때문에, 반사를 가장 보여줘야 할 구간의 가중치가 구조적으로 깎인다. 이건 [shader_sources.cpp](/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/cpp/src/gpu/shader_sources.cpp:997)와 [shader_sources.cpp](/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/cpp/src/gpu/shader_sources.cpp:1046)에서 직접 확인된다.

2. 문제는 단순히 `Fresnel x edgeAlpha`만이 아니라, 그 앞에 이미 `lens.a * uOpacity * eyelidMask`가 있고, `uScleraProtect == 1`이면 외곽에서 `finalAlpha`를 한 번 더 줄인다. 따라서 실제 유효 반사량은 문서의 표보다 더 낮아질 수 있다. 현재 진단은 방향은 맞지만 감쇠 체인의 전체 길이를 약간 과소평가했다.

3. `reflectUV` 옵션 C 자체가 완전히 틀린 것으로 보이지는 않는다. `reflectUV = (adjustedCoord - adjustedCenter) / scaledRadius * 0.5 + 0.5`는 최소한 홍채 로컬 좌표계를 안정적으로 만들고 있고, 사분면 원색 env map이면 강도가 충분할 때는 무조건 차이가 보여야 한다. 따라서 "좌표가 완전히 잘못돼서 아무것도 안 보인다"는 주원인 가설은 약하다.

다른 후보 원인에 대한 판단:

- **후보 A: intensity 자체가 너무 낮다**
  - 유력한 2차 원인이다.
  - 기본값이 `0.3`이고, CPU에서 `reflection_intensity_`를 `0.0~1.0`으로 clamp하고 있다. 따라서 문서의 `(c) 1.5~3.0`은 현재 구현 그대로는 실험조차 불가능하다. [gpu_lens_renderer.cpp](/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/cpp/src/gpu/gpu_lens_renderer.cpp:406)

- **후보 B: sRGB/linear 혼합 위치 문제**
  - 존재하는 문제다. 특히 `blendTintLinearV2()`는 내부 선형 계산 후 `toSRGBFast()`로 반환하고, 그 뒤에 reflection을 그대로 더한다. 즉 "선형 조명 합성"이 아니라 "sRGB 결과 위에 가산"이다. [shader_sources.cpp](/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/cpp/src/gpu/shader_sources.cpp:865)
  - 다만 이 문제는 "물리적으로 부정확하고 강도 해석이 꼬이는" 쪽이지, OFF/EnvMap 차이가 아예 안 보일 정도의 1차 원인으로 보긴 어렵다. 지금 증상은 그보다 앞단의 마스킹 감쇠가 더 직접적이다.

- **후보 C: Periphery 좌표 변환 `adjustedRingPoint.x / uFrameAspect` 오류**
  - 현재 식은 셰이더 내부 좌표계 정의와 일관된다. `adjustedCoord = vTexCoord * vec2(aspect, 1)`로 갔으니, 역변환에서 `x / aspect`를 하는 건 맞다. 따라서 "명백한 버그"로 보긴 어렵다. [shader_sources.cpp](/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/cpp/src/gpu/shader_sources.cpp:985)

- **후보 D: mipmap LOD 문제**
  - 가능성은 낮다. 256x128 4사분면 원색 env map이면 mip가 좀 섞여도 완전히 무색화되진 않는다. 지금처럼 OFF/EnvMap 차이가 거의 없다는 현상은 LOD보다 앞단 곱셈 체인의 영향이 더 크다.

- **후보 E: Periphery 소스 자체가 본질적으로 약하다**
  - 매우 유력하다. 이건 버그라기보다 소스 품질 한계다.
  - 현재 Periphery는 8포인트 평균이고, 반경 `2.15r`에서 피부/머리카락/배경 저주파 평균색을 가져올 가능성이 높다. 거기에 평균을 내므로 대비가 더 죽는다. 따라서 EnvMap보다 약하게 보이는 건 설계상 자연스럽다.

정리:

**현재 진단은 맞다.** 다만 정확한 표현은 "`Fresnel + edgeAlpha` 충돌이 핵심이고, 낮은 intensity clamp, outer-region 추가 감쇠, Periphery 저대비 특성이 함께 겹쳐 체감이 거의 0이 되었다"가 더 정확하다.

## Q2. 이 충돌은 설계 누락인가, 의도된 자연스러움 설계인가?

명시적 답변:

**설계 의도가 자연스러움 쪽에 있었던 것은 맞지만, 결과적으로는 W3 설계 누락으로 분류하는 것이 맞다.**

이유:

1. W3 문서는 `renderMask = finalAlpha`와 외곽형 Fresnel을 동시에 채택했지만, 둘의 중첩으로 실제 가시성이 얼마 남는지에 대한 "visibility budget" 검증이 없었다. 즉 철학은 자연스러움이었지만, 검증 빠진 결합이다.

2. W3 문서에도 반사 강도 `0.3`은 "W4 튜닝 후보"로 남겨져 있다. 즉 이 값과 외곽 마스킹 조합이 실제로 보이는지는 아직 닫힌 결정이 아니었다. [P6-W3_env_reflection_scaffold.md](/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/docs/workPaper/P6-W3_env_reflection_scaffold.md:120)

3. 따라서 이건 "의도된 미세 반사라서 안 보이는 것이 정상"으로 해석하면 안 된다. W4 Phase A의 목적은 최소한 OFF/EnvMap/Periphery 차이를 관찰 가능하게 만드는 것인데, 지금은 그 관찰 자체가 무효화됐다.

정리:

**설계 철학은 자연스러움이었지만, W4 검증 목적을 만족할 정도의 최소 가시성 조건을 명세하지 못한 설계 누락**이다.

## Q3. 해결 방향 우선순위는?

명시적 답변:

**우선순위는 (e) 전용 reflectionMask 도입 > (b) Fresnel 안쪽 이동 > (c) 강도 상향 순이 맞다.**

세부 판단:

### 1순위: (e) `finalAlpha`와 분리된 `reflectionMask` 도입

가장 추천한다.

이유:

- 문제의 핵심은 "렌즈 색 블렌드 마스크"와 "반사 가시성 마스크"를 같은 `finalAlpha`에 묶은 것이다.
- 반사는 렌즈 색보다 외곽에서 더 살아야 하는데, 현재는 렌즈 외곽 페이드 정책에 종속돼 같이 죽는다.
- 따라서 `renderMask = finalAlpha`를 유지한 채 숫자만 만지는 것보다, 반사용 마스크를 분리하는 게 구조적으로 맞다.

권장 형태:

```glsl
float lensAlpha = lens.a * uOpacity * eyelidMask;
float edgeAlpha = smoothstep(1.0, featherStart, dist);
float finalAlpha = lensAlpha * edgeAlpha;

float reflectionMask = lensAlpha;
reflectionMask *= smoothstep(1.05, 0.85, dist); // sharp cut 대신 완만한 외곽 유지

blended += reflection * fresnel * uReflectionIntensity * reflectionMask;
```

핵심은 `(a) edgeAlpha 제거`를 그대로 직선 적용하는 게 아니라, **반사용 soft mask를 따로 설계**하는 것이다. 그래야 외곽 hard cutoff 부작용을 줄일 수 있다.

### 2순위: (b) Fresnel 영역 안쪽 이동

최소 수정으로 효과를 빨리 확인하려면 이게 가장 쉬운 패치다.

추천 범위:

- `smoothstep(0.4, 0.8, dist)` 또는
- `smoothstep(0.45, 0.85, dist)`

이유:

- 현재 `0.7~1.0`은 edge fade 구간과 거의 정면 충돌한다.
- Fresnel을 안쪽으로 당기면 기존 `finalAlpha` 구조를 유지해도 반사가 살아날 가능성이 크다.
- 다만 이것만으로는 "왜 반사가 렌즈 블렌드 마스크에 종속돼야 하는가"라는 구조 문제는 남는다.

### 3순위: (c) intensity 상향

디버그/스모크 테스트용으로는 필요하지만, 구조 수정보다 뒤다.

주의:

- 현재 CPU clamp 때문에 `>1.0` 실험은 불가능하다.
- 즉 `(c)`를 하려면 먼저 [gpu_lens_renderer.cpp](/Volumes/M3-P31/Projects/MerooMong/IrisLensSDK/cpp/src/gpu/gpu_lens_renderer.cpp:406)의 clamp 범위를 넓히거나, 디버그 전용 override가 필요하다.
- 구조를 그대로 두고 강도만 올리면 외곽 페이드와의 충돌을 숫자로 억지 보상하는 셈이라, 자연스러움과 재현성이 둘 다 나빠질 가능성이 높다.

비추천:

- **(a) renderMask에서 edgeAlpha를 완전히 제거**: 단독 적용은 비추천. 외곽에서 반사가 렌즈 silhouette 밖으로 딱 끊기는 인상이 날 수 있다.
- **(d) composition 순서 변경**: 지금 문제를 풀기 위해 필요한 첫 수단은 아니다. 순서보다 마스크 설계가 먼저다.

정리:

1. **1순위**: (e) `reflectionMask` 분리 도입
2. **2순위**: (b) Fresnel을 `0.4~0.8` 근방으로 안쪽 이동
3. **3순위**: (c) intensity 상향. 단, clamp 수정 없이는 `>1.0` 실험 불가

## Q4. Phase B 전 해결 필요 수준인가, 아니면 그대로 평가해도 되나?

명시적 답변:

**Phase B 진입 전에 반드시 해결해야 한다. 그대로 가면 벤치가 무의미해질 가능성이 높다.**

이유:

1. Phase B 목적은 OFF / EnvMap / Periphery의 상대 비교인데, 현재처럼 셋 다 거의 안 보이면 "어느 소스가 더 낫다"가 아니라 "현재 파이프라인에선 아무것도 안 보인다"만 확인하게 된다.

2. 그 상태에서 24클립을 돌리면, 결과 해석이 소스 비교가 아니라 "가시성 부족한 설정의 실패"를 대량 수집하는 쪽으로 흐른다. 그건 B2 질문에 답하지 못한다.

3. 다만 "완성형 광학 설계"까지 끝낼 필요는 없다. **Phase B 전에 필요한 건 최소한 1개 visible configuration을 확보하는 것**이다.

권장 기준:

- OFF와 EnvMap이 정지 화면 비교만으로도 눈에 띄게 구분될 것
- Periphery는 약해도 되지만 OFF와 완전히 동일해 보이진 않을 것
- 적어도 1개 디버그 세팅에서 관찰자가 3초 내 차이를 말할 수 있을 것

정리:

**Phase B 전 선행 수정 필요**다. 단, 범위는 "구조 재설계"가 아니라 "가시성 확보용 최소 패치 + 5분 스모크 검증"이면 충분하다.

## 권장 액션 1~3순위

1. **반사용 마스크 분리 패치 적용**
   - `renderMask = finalAlpha`를 끝내고, `lens.a * uOpacity * eyelidMask` 기반의 `reflectionMask`를 별도로 둔다.
   - edge softness는 reflection 전용 smoothstep으로 약하게만 유지한다.

2. **Fresnel 구간을 안쪽으로 이동한 디버그 세팅 추가**
   - `smoothstep(0.45, 0.85, dist)` 정도로 옮긴 variant를 만들어 OFF/EnvMap 차이가 즉시 보이는지 확인한다.
   - 이 단계에서 EnvMap과 Periphery를 다시 비교해야 원인 분리가 된다.

3. **강도 디버그 상한 해제 후 스모크 테스트**
   - `reflection_intensity_` clamp를 일시적으로 `2.0` 또는 `3.0`까지 열어, 구조 수정 후에도 여전히 효과가 미약한지 확인한다.
   - 이건 최종 디자인용이 아니라 "마스크/좌표 문제인지 단순 강도 부족인지" 분리하기 위한 진단용 액션이다.
