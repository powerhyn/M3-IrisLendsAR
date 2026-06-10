# P7-W2 avg_iris_luma 측정 패스 - Codex R1

## 6.1 측정 패스 위치

추천: **C(detector CPU 버퍼)**를 1순위로 채택한다. A(demo Kotlin GL)는 데모 한정 fallback으로만 두고, B(SDK core GL)는 비추천한다.

근거: core 주석이 `EXTERNAL_OES + 임시 FBO + glReadPixels`의 GL state 오염을 실기기 확인 사항으로 명시한다(`cpp/src/gpu/gpu_lens_renderer.cpp:1098-1100`, `cpp/include/iris_sdk/gpu/gpu_lens_renderer.h:369-371`). C는 GL을 완전히 우회하므로 OES 충돌/상태 오염 리스크가 0이고, N=5 + 64x64 ROI CPU 루프는 30fps 예산상 가장 안전하다.

근거: A는 이미 `rgbaFboId`에 OES->RGBA 결과를 만들지만(`CameraGLRenderer.kt:805-828`), render thread `glReadPixels`와 demo->SDK JNI 전달 결합이 남는다. B는 경고된 실패 형태를 SDK core에서 되살리는 선택이라 제외한다.

## 6.2 ROI 추출 / 다운샘플 방법

추천: 6.1의 C에 맞춰 detector CPU RGB 버퍼에서 `iris center +- 0.65r` 원형 마스크를 최대 64x64로 샘플링해 평균 linear luma를 계산한다.

근거: mipmap/reduction shader는 GL 경로라 6.1의 회피 목표와 맞지 않는다. detector는 이미 프레임 버퍼와 iris center/radius를 같은 타이밍에 갖고 있고, 현재 adapter는 `avg_iris_luma`를 nullopt로 비워 둔다(`cpp/src/gpu/eye_render_packet_adapter.cpp:94-95`)는 슬롯만 채우면 된다.

## 6.3 N=5 주기 + EMA 위치

추천: N=5 서브샘플링 카운터는 producer 측정부에 둔다. producer는 EMA를 걸지 말고, consumer `updateAvgIrisLuma()`의 EMA(alpha=0.3)만 사용한다.

근거: consumer가 이미 양안 평균과 EMA를 수행한다(`cpp/src/gpu/gpu_lens_renderer.cpp:800-821`)서 producer EMA를 추가하면 이중 평활로 gate 반응이 늦어진다. 측정하지 않는 프레임은 optional을 비우고 hold로 유지하되, N=5와 맞게 hold는 최소 4프레임 이상으로 맞춘다(현재 `kAvgLumaMaxHoldFrames=3`, `cpp/include/iris_sdk/gpu/gpu_lens_renderer.h:376-377`).

## 6.4 hysteresis 파라미터

추천: CPU/renderer 쪽 상태값으로 dual-threshold를 적용한다: **enter low-light `< 0.08`**, **exit low-light `> 0.12`**. 시간 기반 N-frame 연속 조건은 쓰지 않는다.

근거: 0.08/0.12는 현재 기본 gate 0.10을 사이에 두고, shader의 기존 soft band `uGateThreshold +- 0.03`과 정합된다(`cpp/src/gpu/shader_sources.cpp:1084`). N=5 측정 주기에서 시간 기반 조건을 추가하면 최소 수 프레임 지연이 누적되어 blink/detail 반응이 둔해진다.

## 6.5 LUMA 계수 일관성 + 테스트

추천: producer 측정도 shader와 동일하게 **Rec.709 linear**로 통일한다. RGB 샘플을 먼저 `srgb * srgb` 근사로 linear 변환한 뒤 `dot(linear, vec3(0.2126, 0.7152, 0.0722))`를 평균하고, sRGB 평균은 금지한다.

근거: shader가 `LUMA_709_LENS`와 `toLinearFast(srgb) { return srgb * srgb; }`를 쓰며 CPU 측 측정과 같은 상수라고 명시한다(`cpp/src/gpu/shader_sources.cpp:847-851`), renderer header도 `uAvgIrisLum`을 linear 값으로 정의한다(`cpp/include/iris_sdk/gpu/gpu_lens_renderer.h:373-379`). 테스트는 black/white/gray/R/G/B와 64x64 synthetic ROI에 대해 CPU producer helper와 shader-equivalent reference를 비교해 오차 <= 1%로 둔다.
