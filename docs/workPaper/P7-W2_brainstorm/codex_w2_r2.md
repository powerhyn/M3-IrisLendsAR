# P7-W2 R2 - Codex

## R2.1 정규화 해석 확인

추천: 해석은 맞다. `blendTintLinearV2`는 clamp가 없다고 보면 `tinted = blendL * 0.85 * (lum / uAvgIrisLum)`이고, ID=7도 `lum / uAvgIrisLum`을 detail 분모로 쓴다.

근거: 셰이더가 `baseL = toLinearFast(base)`, `lum = dot(baseL, LUMA_709_LENS)`, `scale = clamp(0.85 / max(0.01, uAvgIrisLum), 0.8, 7.0)` 뒤 `blendL * lum * scale`을 만든다(`cpp/src/gpu/shader_sources.cpp:874-878`). `uAvgIrisLum`이 같은 ROI 평균이면 ROI 평균 밝기 기준 tint가 거의 일정해진다.

## R2.2 현 0.1225 고정 = 버그 vs 의도

추천: 보존할 튜닝 타겟이 아니라 **실측 연결로 고쳐야 할 fallback 상태**로 본다.

근거: `uAvgIrisLum`은 블렌드 ID=5/7의 정규화 분모인데(`cpp/src/gpu/shader_sources.cpp:1055-1061`), 고정 0.1225는 밝은 홍채에서 scale을 과하게 유지한다. 따라서 fallback은 미검출/측정 실패용으로만 남기고, 정상 경로는 실측값으로 전환해야 한다.

## R2.3 clamp[0.8,7.0] + K=0.85 재튜닝

추천: 실측 연결 1차 적용에서는 `K=0.85`, clamp `[0.8, 7.0]`을 유지해 blast radius를 분리하고, 실측 분포 로그 후 K 중심으로 재튜닝한다.

근거: clamp는 극단값 안전장치이고 K가 실제 룩 강도 노브다. 측정 경로와 상수를 동시에 바꾸면 over-tint 해소가 실측 때문인지 재튜닝 때문인지 분리하기 어렵다.

## R2.4 측정=정규화 정합 보장

추천: 현재 코드만으로 `uCameraTexture`와 detector `rgb_buffer`가 **완전히 같은 픽셀/색공간이라고 볼 수는 없다**. C안에서는 detector가 쓰는 동일 RGB 프레임에서 ROI luma를 계산하고, 셰이더와 동일한 `srgb * srgb` + Rec.709 + 동일 iris ROI 계약으로 정합을 보장해야 한다.

근거: 렌더 경로는 OES를 SurfaceTexture matrix/mirror/Y-flip 적용 후 `rgbaTextureId`로 만들고(`CameraGLRenderer.kt:735-741`, `805-825`), core는 그 2D 텍스처를 `uCameraTexture`로 샘플한다(`cpp/src/gpu/gpu_lens_renderer.cpp:929-931`). detector 경로는 ImageProxy NV21을 받아 회전 포함 검출을 수행하고(`GpuRenderActivity.kt:953-970`), `convertToRgb()`에서 OpenCV RGB로 변환한다(`cpp/src/mediapipe_detector.cpp:896-908`, `1084-1124`, `2451-2455`).

근거: 셰이더의 색공간 기준은 명확하다: `LUMA_709_LENS`와 `toLinearFast(srgb) { return srgb * srgb; }`이며 sRGB 평균 금지를 주석으로 못박았다(`cpp/src/gpu/shader_sources.cpp:847-851`). 기존 Kotlin NV21 Y채널 5점 샘플(`GpuRenderActivity.kt:1088-1128`)은 블렌드 분모로 쓰면 색공간/ROI가 맞지 않는다.

## R2.5 W5/W6 튜닝 보존 전략

추천: **(a) 실측 + 재튜닝으로 기존 룩 보존**을 채택한다. (b)는 불필요하게 룩 변화를 수용하고, (c)는 blend 정규화 의도를 깨므로 비추천한다.

근거: `uAvgIrisLum`은 detail/gate뿐 아니라 ID=5/7 메인 블렌드의 분모다(`cpp/src/gpu/shader_sources.cpp:877-888`, `1071-1084`). gate와 blend를 분리해 blend만 안정값에 묶으면 밝기-불변 틴트 완성이라는 R2 해석을 포기하게 된다.
