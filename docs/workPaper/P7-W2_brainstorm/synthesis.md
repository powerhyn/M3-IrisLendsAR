# P7-W2 R1 종합 (Claude, 편향 경계)

> Codex(gpt-5.5 xhigh) + Gemini(gemini-3-flash) + Claude 3모델 R1. 모든 코드 주장 직접 검증 완료.
> 합의 분포: **3/3 만장일치 4건, 2/3 다수결 1건, Hard veto 0, R2 불필요**.

## 합의 표

| 쟁점 | Codex | Gemini | Claude | 판정 | 확정 |
|---|---|---|---|---|---|
| **6.1 측정 위치** | C (A fallback, B 제외) | C | C (A 차선) | **3/3 → C** | ✅ detector CPU 버퍼 |
| **6.2 ROI 방법** | CPU 원형마스크 ±0.65r ≤64² | CPU iris ROI ≤64² | CPU masked-ROI ±0.65r | **3/3 → CPU 루프** | ✅ |
| **6.3 N=5 + EMA** | N=5 producer + consumer EMA만 | N=5 producer + consumer EMA | **N=5 제거**, 매프레임 | **2/3 → N=5 유지** | 🟡 다수결 |
| **6.4 hysteresis** | enter0.08/exit0.12 CPU | enter0.08/exit0.12 | enter0.08/exit0.12 Schmitt | **3/3 → dual 0.08/0.12** | ✅ |
| **6.5 LUMA+테스트** | Rec.709 srgb*srgb match, ≤1% | Rec.709 ≤1% | Rec.709 linear ≤1% | **3/3 → Rec.709 + ≤1% 테스트** | ✅ |

## 검증된 코드 사실 (3모델 인용 → Claude 직접 확인)

1. **detector CPU RGB 버퍼 존재** — `mediapipe_detector.cpp:191` `cv::Mat rgb_buffer`(cvtColor RGB 변환, `:984`). → **C 실현 가능 확정**.
2. **`toLinearFast(srgb)=srgb*srgb`** — `shader_sources.cpp:851`. 주석(`:849-850`) "CPU 측 W1 avg_iris_luma 측정과 동일 상수, sRGB 평균 금지". → CPU 측정은 **정확 sRGB 곡선 아니라 `srgb*srgb` fast 근사로 매칭**해야 ≤1% 통과 (Codex 정밀 캐치).
3. **`kAvgLumaMaxHoldFrames=3`** — `gpu_lens_renderer.h:377` + hold 체인(`:201`). → **N=5 채택 시 hold를 ≥5로 상향** 필수 (안 하면 매 측정 사이 frame 4~5가 fallback 상수로 떨어져 진동). Codex catch.
4. **`LUMA_709_LENS=(0.2126,0.7152,0.0722)`** — `shader_sources.cpp:850`. Rec.709 확정.

## 6.3 다수결 상세 (유일한 2/3)

- **다수(Codex+Gemini)**: N=5 주기 측정(producer 카운터) + consumer 기존 EMA(α=0.3)만, 이중 평활 금지.
- **소수(Claude)**: CPU 측정이 싸므로 N 제거하고 매 detection 측정 → hold/N 결합 복잡도 제거.
- **판정**: 다수결 채택(N=5 + **hold≥5 상향**). 단 매-프레임 측정도 동등 저위험이라 **구현 중 판정 가능**(hold/N 결합이 번거로우면 N 제거가 더 단순). 사용자 최종 선택 여지.
- 공통: **producer는 raw만, EMA는 consumer 단일** (3/3 일치).

## 미해결/구현 중 판정 (구현 단계 결정)

- **detector rgb_buffer 좌표공간/해상도**: rgb_buffer는 resize 전 원본 RGB. iris center/radius가 이 좌표계인지 확인 + ROI 매핑 (구현 시 1차 확인). C 실현성엔 영향 없음, 정확도 디테일.
- 6.3 N=5 vs 매-프레임 (위).

## 결론

R2 불필요 (만장일치 4 + 저위험 다수결 1). §5 확정 이동 가능. W 추정 2~3일이나 consumer EMA 재사용 + C(CPU)로 단순화 → 단축 가능.
