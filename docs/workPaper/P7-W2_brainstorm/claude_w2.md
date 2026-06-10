# P7-W2 R1 — Claude 응답

> 한 모델의 독립 의견 (모더레이터 아님). Codex/Gemini 응답 보기 전 작성. 코드 실측 기반.

## 6.1 측정 패스 위치 — **추천: C (detector CPU 버퍼), 차선 A**

- **근거**: C는 GL을 완전히 우회 → EXTERNAL_OES 충돌(메모리 `w1-measure-external-oes-collision`)·glReadPixels GL state 오염(`gpu_lens_renderer.cpp:1098` 주석)을 **원천 제거**. detector(MediaPipe)는 이미 CPU에 프레임 + iris center/radius를 보유하므로, iris ROI 평균 luma는 detection 스레드에서 **수천 px CPU 루프 1회**(거의 0 비용, render 스레드 무영향). 데이터 흐름도 깔끔: IrisResult → adapter → `packet.avg_iris_luma` → 기존 consumer EMA (주석이 예고한 "detector CPU 버퍼" 경로와 정확히 일치).
- **C의 유일 리스크**: detector 입력 버퍼의 해상도/색공간이 shader가 보는 `rgbaTextureId`(OES→2D 결과)와 다를 수 있음. 단 gate 목적은 "저조도 근사 판정"이라 픽셀 정확도 불필요(smoothstep ±0.03 + EMA 흡수). 색공간만 6.5에서 맞추면 됨.
- **B 기각**: core glReadPixels = 주석 1098의 state 오염 = W1 회귀 재유발. **A 차선**: rgbaTextureId가 shader와 동일 텍스처라 luma 정확하나 glReadPixels 동기 stall(30fps 위험) → 쓰려면 PBO 비동기 readback 필수. C가 안 되면 A(PBO).

## 6.2 ROI 추출/다운샘플 — **추천: CPU masked-ROI 평균 (6.1=C 종속)**

- **근거**: 6.1=C면 방법은 자동으로 **CPU 루프**. iris center ± 0.65·radius 원형 마스크 평균 (W1 skill `measureAvgIrisLuma` 공식 재사용). iris ROI는 작아 비용 무시. mipmap level read / reduction shader는 GL 경로(A/B) 전용이라 C에선 불필요. 별도 64×64 다운샘플 패스도 불필요(마스킹 루프가 곧 다운샘플).

## 6.3 N=5 주기 + EMA 위치 — **추천: producer raw 매 detection + consumer 단일 EMA, N=5 제거**

- **근거**: N=5는 측정이 비쌌을 때(GPU readback) 전제. C(CPU)는 싸므로 **매 detection 측정**이 더 안정적. **이중 EMA 금지** — producer는 raw 측정값만 packet에 넣고, consumer의 기존 EMA(α=0.3, `updateAvgIrisLuma`)가 유일 평활. N=5 서브샘플링은 비용 절감 목적이 사라져 제거 권장(detection 자체가 ~30fps 이하라 주기 충분).

## 6.4 hysteresis — **추천: dual-threshold(Schmitt) enter 0.08 / exit 0.12, CPU 적용**

- **근거**: gate 기본 0.10 주변에서 저조도 상태 플래그를 enter<0.08·exit>0.12로 래치 → 경계 진동 차단(Gemini가 P7-W0에서 보강한 1프레임 지연 oscillation). EMA가 이미 저역통과라 진동 위험은 낮지만, 명시적 Schmitt가 안전. 적용은 CPU(상태 래치) — 셰이더 smoothstep gate는 연속값이라 hysteresis를 binary 상태로 CPU에서 관리하고 그 결과를 uniform/threshold로 전달. 수치는 실기기 튜닝(메모리 `qualitative-device-judgment`).

## 6.5 LUMA 계수 + 테스트 — **추천: Rec.709 통일 + sRGB→linear 선적용, CPU 참조 비교 테스트**

- **근거**: 셰이더가 `LUMA_709_LENS`(Rec.709 = 0.2126/0.7152/0.0722) + `toLinearFast` 사용. producer CPU 측정도 **각 픽셀 sRGB→linear 변환 후 Rec.709 dot**으로 동일 정의. ⚠️ W1 skill 예시의 Rec.601(0.299...)은 **불일치라 폐기** — Rec.709로 락. 테스트: 셰이더 luma 수식을 C++로 복제한 참조 구현 vs producer 측정 함수를 동일 RGB 입력셋에 대해 비교, ≤1% 오차 assert (GoogleTest).

## 종합 코멘트

W2의 80%는 "6.1 위치" 한 결정에 종속(6.2 방법 + 6.5 색공간이 따라옴). C(CPU)면 GL 리스크 0 + 구현 최소 + consumer EMA 재사용으로 가장 깔끔. 핵심 검증 포인트는 **detector 버퍼 색공간/해상도** 확인 1건. C 채택 시 W2는 추정 2~3일보다 짧아질 수 있음(producer CPU 측정 + 색공간 변환 + 테스트 + hysteresis).
