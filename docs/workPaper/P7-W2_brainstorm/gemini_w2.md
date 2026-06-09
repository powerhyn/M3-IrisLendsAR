# P7-W2 Brainstorming: avg_iris_luma 측정 패스 (Gemini 입장 정리)

§6 미결 사항에 대해 EXTERNAL_OES 충돌, GL state 오염 방지, 그리고 30fps 유지 관점에서 다음과 같이 추천합니다.

## 6.1 측정 패스 위치 (핵심 결정)
- **추천**: **(C) detector CPU 버퍼 (MediaPipe)**
- **근거**: `gpu_lens_renderer.cpp:1092` 주석에서 경고한 "EXTERNAL_OES + 임시 FBO + glReadPixels 경로의 GL state 오염" 이슈를 원천 차단할 수 있는 유일한 경로입니다. MediaPipe detector가 이미 CPU 상에 RGB 버퍼(`rgb_mat`)와 정밀한 홍채 랜드마크를 보유하고 있어 별도의 GL-to-CPU 복사 비용 없이 가장 안정적인 측정이 가능합니다.

## 6.2 ROI 추출 / 다운샘플 방법
- **추천**: **CPU 루프 기반 Iris ROI 가중 평균**
- **근거**: 위치 (C) 선정에 따라, detector의 원본 RGB 버퍼에서 홍채 중심 좌표 기준 ±반지름(r) 영역을 64x64 이하의 ROI로 추출하여 계산합니다. Mipmap이나 셰이더 리덕션 대신 CPU 루프를 사용해도 ROI 크기가 작아 30fps 예산 내에서 무시할만한 오버헤드로 정밀한 휘도 산출이 가능합니다.

## 6.3 N=5 주기 + EMA 위치
- **추천**: **Producer(Detector)에서 N=5 주기 측정 + Consumer(Renderer)의 기존 EMA 활용**
- **근거**: 매 프레임 측정은 불필요한 연산이므로 N=5 주기로 충분합니다. `GPULensRenderer::updateAvgIrisLuma()`에 이미 α=0.3인 EMA 골격이 구축되어 있으므로, Producer는 측정된 raw 값만 `EyeRenderPacket`에 실어 보내어 이중 평활화(over-smoothing)에 의한 반응 지연을 방지해야 합니다.

## 6.4 hysteresis 파라미터
- **추천**: **Dual-threshold Hysteresis (Enter < 0.08 / Exit > 0.12) 도입**
- **근거**: 저조도 gate 기본 임계값(0.10) 부근에서 미세한 조도 변화로 인한 렌즈 디테일/gate의 시각적 진동(flicker)을 방지하기 위해, ±0.02의 여유폭을 둔 이중 임계값 로직을 Renderer의 luma 업데이트 로직에 추가할 것을 추천합니다.

## 6.5 LUMA 계수 일관성 + 테스트
- **추천**: **Rec.709 계수 (`0.2126R + 0.7152G + 0.0722B`) 통일 및 오차 검증**
- **근거**: 셰이더의 휘도 보존 블렌딩(`LUMA_709_LENS`)과 동일한 Rec.709 표준을 사용하여 일관성을 확보합니다. 동일한 텍스처 입력에 대해 CPU 측정치와 셰이더 계산값의 오차가 1% 이내임을 보장하는 단위 테스트를 설계하여 측정 신뢰도를 확보해야 합니다.
