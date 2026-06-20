# 핸드오프: 피부 화사함(soft-glow radiance) — LensSim 검증 → IrisLensSDK 이식

작성: 2026-06-18 | 출처: LensSimulator(실험 상류) | 대상: IrisLensSDK(이식 하류)
상태: **LensSim Android 구현·빌드·적대 리뷰 통과 + 실기기(S23+) 기본 강도 0.40 확정.**
관련 커밋: `d526944`(feat: 화사함 추가). 패턴: 투트랙(LensSim 실험 → IrisLens 반영).

> ⚠️ 즉시 구현 지시가 아니라 **이식 설계 입력**이다. IrisLens 구조 변환(W4) 완료 후
> 렌즈 품질 트랙에서 반영할 것. 셰이더 알고리즘 변경이므로 양 플랫폼 동시 수정 원칙 적용.

---

## 1. 한 줄 인사이트

**"피부 스무딩용으로 이미 계산하는 가우시안 블러를 공짜 bloom 소스로 재활용한다."**
화사함 = `screen(blur)` 윤기 + 휘도비율 lift 화사 + 약한 채도감소(맑은 톤) + 미세 웜.
새 패스 0, 새 텍스처 fetch 0 (스무딩 패스의 blur/mask와 공유).

## 2. 왜 이 접근인가 (5개 후보 판정 결과)

판정단 평가에서 soft-glow가 1위(8.7/10). 핵심 이유:
- **유일하게 진짜 "윤기/광채"** — screen-blend는 밝은 피부(광대·콧대)에만 빛을 inflation
  하므로 "전체가 밝아짐"이 아니라 물리적 dewy 반사로 읽힘.
- **hue 보존** — lift를 채널 가산이 아니라 **휘도 비율(ratio multiply)**로 적용 → 떠보임/표백 구조적 불가.
- **최저 비용** — 블러가 공짜라 ~24 ALU 단일 블록. (Oklab 라운드트립·6-term LGG보다 쌈)
- 차점: YCbCr luma-lift(8.0, IrisLens 기존 WHITENING과 동류 — brighten+even이지 glow 아님).

## 3. 셰이더 알고리즘 (실제 GLSL, LensSim COMPOSITE_FS step⑥)

피부 마스크 게이팅(`mask>0`) + 강도(`uRadiance`) 안에서만 실행:

```glsl
// blur, mask 는 스무딩 패스와 공유 (디스플레이 공간, 워프된 UV)
if (uRadiance > 0.0 && mask > 0.0) {
    const vec3 LW = vec3(0.299, 0.587, 0.114);
    const float RAD_GLOW  = 0.20; // screen-bloom 강도 (윤기)
    const float RAD_LIFT  = 0.08; // 미드톤 휘도 리프트 (화사)
    const float RAD_DESAT = 0.10; // 채도 감소 (맑은 톤)
    const float RAD_KNEE  = 0.78; // 하이라이트 보호 knee
    const float RAD_WARM  = 0.012; // 웜 바이어스 (혈색)
    float rad = uRadiance;
    float lumaB = dot(base, LW);
    float hiRoll = 1.0 - smoothstep(RAD_KNEE - 0.06, RAD_KNEE + 0.17, lumaB);
    // 에지 가드 — 고대비 경계에서 quarter-res 블룸 halo 억제 (스무딩과 동일 기준)
    float radEdge = smoothstep(0.06, 0.18, abs(lumaB - dot(blur, LW)));
    // (a) screen-blend bloom — 밝은 피부만 빛 inflation = 윤기
    vec3 screenC = 1.0 - (1.0 - base) * (1.0 - blur);
    vec3 bloomed = mix(base, screenC, RAD_GLOW * rad * hiRoll * (1.0 - radEdge));
    // (b) 미드톤 휘도 리프트 — hue 보존 위해 휘도 비율로
    float midW = smoothstep(0.10, 0.35, lumaB) * (1.0 - smoothstep(0.70, 0.92, lumaB));
    float yIn  = max(dot(bloomed, LW), 1e-4);
    float yOut = yIn + RAD_LIFT * rad * midW * (1.0 - yIn); // 점근적, 클립 없음
    vec3  lifted = bloomed * (yOut / yIn);                  // 비율 스케일 = hue/sat 보존
    // (c) 약한 채도 감소 — 칙칙함 제거
    float y2   = dot(lifted, LW);
    vec3  even = mix(lifted, vec3(y2), RAD_DESAT * rad);
    // (d) 미세 웜 바이어스 — 채도 감소가 차갑게 가지 않도록
    even.r += RAD_WARM * rad * midW;
    even.b -= RAD_WARM * 0.5 * rad * midW;
    base = clamp(mix(base, even, mask * rad), 0.0, 1.0);
}
```

## 4. 튜닝 상수 (실기기 S23+ 기준)

| 상수 | 값 | 의미 |
|---|---|---|
| 기본 강도 `uRadiance` | **0.40 (확정)** | UI 기본값. S23+에서 자연스러운 톤업 지점 |
| `RAD_GLOW` | 0.20 | screen-bloom 윤기 |
| `RAD_LIFT` | 0.08 | 미드톤 화사 |
| `RAD_DESAT` | 0.10 | 채도감소(맑은 톤). ≤0.16 유지(초과 시 인형같이 평탄) |
| `RAD_KNEE` | 0.78 | 하이라이트 보호 (T존 클립 방지) |
| `RAD_WARM` | 0.012 | 웜 바이어스(혈색) |

## 5. 렌즈 무영향 보장 (적대 리뷰 5차원 통과 — 코드로 증명)

- **마스크 게이팅**: skin mask가 face-oval(1) 후 눈/눈썹/입술 fan(0)으로 덮어 **눈을 제외**.
  홍채와 톤적응 환형(r∈[0.45,0.85])은 mask=0 → radiance 미적용.
- **avgIrisLuma upstream**: 렌즈 톤적응 입력은 **raw 센서 버퍼에서 GL 이전에 측정**
  (FaceTracker.sampleIrisLuma), GL read-back 없음 → radiance가 측정값을 못 건드림.
- **렌즈 셰이더 독립**: 렌즈 패스는 raw 카메라를 독립 샘플 → composite의 radiance가 렌즈 입력에 안 샘.
- 결론: **렌즈 출력은 radiance ON/OFF에 불변.** (단 마스크 페더링 fringe가 공막·눈꺼풀 가장자리
  비-렌즈 피부에 보일 수 있음 — 렌즈/홍채 아님.)

## 6. IrisLensSDK 이식 가이드

- IrisLens는 이미 `WHITENING_FRAGMENT`(YCbCr luma-lift+채도↓+하이라이트보호)를 보유.
  soft-glow는 그것과 **다른 차원의 효과(윤기/광채)** 이므로 **추가**로 넣을 가치가 있음.
- **bloom 소스**: IrisLens의 freq-sep 스무딩이 만드는 블러를 bloom 소스로 재활용
  (LensSim과 동일 원리). 별도 블러 신설 불필요.
- **마스크**: IrisLens skin mask가 눈 contour를 제외하는지 확인(LensSim과 동형 구조 확인됨).
  제외하면 렌즈 무영향이 그대로 성립.
- **통합 위치**: 렌즈 합성 전 피부/뷰티 단계. (이펙트 순서 피부→워핑→렌즈 유지)
- IrisLens는 C++ 단일 코어라 한 번 넣으면 Android/iOS/Web 바인딩에 자동 반영(LensSim의 2벌 중복 없음).

## 7. 미해결 / 검증 필요 (이식 시 같이 챙길 것)

- **dark-skin 캘리브레이션**: `midW`/`hiRoll` knee가 절대 luma 기준이라 어두운 피부에서
  lift 일관성↓. 필요 시 face-relative 평균 luma 기준으로 윈도우 이동.
- **radiance=1.0 극단 halo/과밝음**: 내부 cap 없음(최종 clamp만). 정식 승격 시 유효 max를
  낮추거나(예 0.6) 가중치 cap. 기본 0.40에서는 문제 없음.
- **iOS 이식**: LensSim은 Android 선행(뷰티 체인 예외). IrisLens는 단일 코어라 동시 반영 가능.

## 8. LensSim 구현 참조 (이식 대조용)

- 셰이더: `sdk/android/lenssdk/.../render/Renderer.kt` COMPOSITE_FS step⑥
- 배선: `setRadiance`(LensEngine/Plugin/controller) + 광채 슬라이더(lens_shelf/demo_page)
- 격리: 기존 `setBeauty` 계약 미변경, `[임시 디버그]` 마커 (확정 후 setBeauty 통합)
- 검증: 멀티에이전트 설계(5후보 판정)→구현→적대 리뷰(5차원, confirmed 20: info16/low3/med1)
