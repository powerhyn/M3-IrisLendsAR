# P7-W2: avg_iris_luma 측정 패스 + W6 Phase B/C

> **상태**: 🔄 코드 구현 완료 (2026-06-09, 커밋 `5c22075`/`c228703`/`7702ae6`/`087e1b8`). C++ SDK + 단위테스트 7/7 + test_types + **Android 빌드 BUILD SUCCESSFUL**. **실기기 A/B 재벤치(§5.6 DoD) 대기** — `lum:fb`↔`lum:meas` 토글로 6 SKU 블렌드 재벤치. 브레인스토밍 R1+R2 완료 (2026-06-08~09, 3모델 합의 + 코드 검증).
> **작성**: 2026-06-08
> **선행 의존**: P7-W1 완료 (0x501 제거, develop `046e86f`)
> **소요 추정**: 2.0~3.0 작업일 (단, consumer EMA 이미 구축 → producer 측정 패스가 실제 작업 대부분)
> **참조**: `P7-W0_index.md` §2.P7-W2, `P7-W0_brainstorm/synthesis.md` §2 (deep-research F2 OES→2D 강제 / F3 ROI 다운샘플)

---

## 1. 인사이트 (세션 간 맥락 보존) ⭐

### 1.1 이 W의 정체성

W6에서 도입된 디테일/저조도 gate가 쓰는 `uAvgIrisLum`이 **항상 fallback 0.1225**로만 동작 중. 실측 source를 연결해 블링크 ramp / 저조도 gate / 디테일 재주입이 **실제 홍채 밝기에 반응**하도록 만드는 W. (메모리 `w6-avg-iris-luma-measure`)

### 1.2 코드 실측 — 무엇이 이미 있고 무엇이 없나 (2026-06-08 권위 확인)

| 영역 | 상태 | 근거 |
|---|---|---|
| **Consumer (소비)** | ✅ **구축됨** | `GPULensRenderer::updateAvgIrisLuma()` (`gpu_lens_renderer.cpp:800`)에 **EMA(α=0.3)** + fallback 진입 로직 + `current_avg_luma_`/`avg_luma_has_valid_` 상태. `uAvgIrisLum` uniform 배선 완료(`:495`,`:1101`). |
| **Producer (측정)** | ❌ **부재** | `avg_iris_luma`를 WRITE하는 코드 0건. 어댑터(`eye_render_packet_adapter.cpp:94`)가 명시적으로 미설정. → 항상 `kAvgLumaFallback=0.1225`(`gpu_lens_renderer.h:376`) |
| **EyeRenderPacket 필드** | ✅ 존재 | `std::optional<float> avg_iris_luma` (`eye_render_packet.h:43`) — producer가 채울 슬롯 준비됨 |
| **OES→2D 패스** | demo Kotlin만 | `OES_TO_2D_FRAGMENT_SHADER`/`rgbaTextureId`/`rgbaFboId`/`convertOesToRgb` (`CameraGLRenderer.kt:123,501,803`). SDK core엔 없음 |
| **EXTERNAL_OES 충돌 제약** | 명시됨 | `gpu_lens_renderer.cpp:1098` 주석: "EXTERNAL_OES + 임시 FBO + glReadPixels 경로가 GL state 오염". 메모리 `w1-measure-external-oes-collision`(직접 측정 회귀) |
| **저조도 gate** | ✅ 부분 | `setGateThreshold`/`uGateThreshold`(기본 0.10). hysteresis 없음 |
| **셰이더 LUMA 계수** | Rec.709 | 셰이더는 `LUMA_709_LENS` 사용 |

**핵심 통찰**: consumer EMA가 이미 있으므로 W2 실제 코드 작업은 **producer 측정 경로 1개**로 좁혀짐. 단 "어디서 어떻게 측정하나"가 EXTERNAL_OES 충돌·GL state 오염·30fps 예산이 얽힌 **아키텍처 결정**이라 브레인스토밍 필요.

### 1.3 P7-W0 R1에서 이미 확정된 framework (§5)

deep-research F2/F3 + P7-W0 R1 3/3 통합으로 **방향**은 정해짐 (§5 참조): 기존 OES→2D RGBA FBO 결과 재사용 + small ROI 다운샘플 + N=5 frame 주기 + EMA + hysteresis. 단 **구체 구현 위치/방법이 OR로 미해결** → §6.

---

## 2. 배경/맥락

- W6 §5.2(C10 detail) / §5.7(gate) 구현 시 `uAvgIrisLum` 소비처는 만들었으나 측정 source는 "W6에서 비동기 readback or detector CPU 버퍼로 packet에 채울 예정"(주석 `gpu_lens_renderer.cpp:803`)으로 **미룸**.
- P6-W1에서 EyeRenderPacket optional 필드 + consumer updateAvgIrisLuma(EMA) 골격은 만듦. producer만 비어있음.
- detector(MediaPipe)는 이미 CPU에서 카메라 프레임을 처리하며 iris landmark(center/radius)를 보유 → CPU 측정 경로가 GL 우회 옵션으로 존재.

---

## 3. 전제 조건

1. ✅ P7-W1 완료 (0x501 제거, develop `046e86f`)
2. ✅ EyeRenderPacket consumer 골격 (updateAvgIrisLuma EMA α=0.3)
3. ✅ HIGH tier 기기 (S23+/Adreno 740) 보유

---

## 4. 목표

1. `uAvgIrisLum`에 **실측 avg_iris_luma 공급** (fallback 0.1225 대체)
2. N=5 frame 주기 측정 + EMA로 30fps 유지
3. 저조도 gate hysteresis (1프레임 지연 진동 차단)
4. shader vs CPU LUMA 계수 ≤1% 오차 단위 테스트

### 4.1 Definition of Done (브레인스토밍 후 확정)

- [x] **(코드)** detector 측정 producer + 어댑터 wiring + consumer 토글 연결 (커밋 `5c22075`). 매-detection 측정(§5.3 택일).
- [x] **(코드)** LUMA 계수 단위 테스트 통과 (≤1%, `srgb²`+Rec.709 — `test_iris_luma_measure` 7/7)
- [x] **(코드)** 저조도 gate hysteresis(enter0.08/exit0.12) — gate 전용 `uLowLightActive`, 블렌드 격리
- [x] **(코드)** fallback↔실측 A/B 토글(`lum:fb`/`lum:meas`) 7단 체인 + demo UI (커밋 `5c22075`/`7702ae6`/`087e1b8`)
- [ ] **(실기기)** avg_iris_luma 실측값 공급 확인 (`lum:meas` 토글 시 logcat 값 변동) + 30fps+ 유지
- [ ] **(실기기)** hysteresis 진동 없음 (밝음↔어두움 경계 안정 — 단 binary 래치 exit pop 관찰 권장)
- [ ] **(실기기·R2)** 메인 블렌드(ID=5/7) 6 SKU 재벤치 — `lum:fb`↔`lum:meas` A/B, **밝은 환경 over-tint 교정** 확인 (메모리 `qualitative-device-judgment`)
- [ ] **(실기기·R2)** 필요 시 K/clamp 재튜닝 (Kotlin fallback `0.5/[0.8,2.5]@0.35` prior-art) + 측정-정규화 systematic 정합

---

## 5. 확정 사항 (R1 브레인스토밍 2026-06-08, 3모델 + 코드 검증)

> 합의: 3/3 만장일치 4건 + 2/3 다수결 1건 + Hard veto 0. 종합: `P7-W2_brainstorm/synthesis.md`. 모든 코드 주장 직접 검증.

### 5.1 측정 위치 — **C: detector CPU 버퍼** (3/3)
- MediaPipe detector의 CPU RGB 버퍼(`mediapipe_detector.cpp:191` `cv::Mat rgb_buffer`)에서 iris ROI 평균 linear luma 계산 → `IrisResult` → 어댑터 → `EyeRenderPacket.avg_iris_luma`(현재 nullopt 슬롯) → 기존 consumer.
- **GL 완전 우회** → EXTERNAL_OES 충돌(`w1` 메모리)·glReadPixels GL state 오염(`gpu_lens_renderer.cpp:1098`) 원천 제거.
- A(demo Kotlin rgbaTexture + PBO readback)는 C 불가 시 fallback. B(SDK core GL)는 기각(주석 1098 회귀 재유발).

### 5.2 ROI 추출 — **CPU masked-ROI 평균** (3/3)
- iris center ± 0.65·radius 원형 마스크, 최대 64×64 샘플, 평균 linear luma. (W1 skill `measureAvgIrisLuma` 공식 재사용.) mipmap/reduction shader 불필요(GL 경로라 C와 불일치).

### 5.3 측정 주기 + EMA — **N=5 producer + consumer 단일 EMA** (2/3 다수결)
- producer: N=5 frame 주기 raw 측정만 (EMA 금지). consumer: 기존 EMA(α=0.3, `updateAvgIrisLuma`)만 → **이중 평활 금지** (3/3 공통).
- ⚠️ **`kAvgLumaMaxHoldFrames`를 3→≥5 상향** 필수 (현 hold=3 < N=5면 측정 사이 frame이 fallback으로 떨어져 진동). Codex catch, `gpu_lens_renderer.h:377` 검증.
- (소수안 보존) CPU 측정이 싸므로 **매-프레임 측정(N 제거)** 도 동등 저위험 — hold/N 결합 회피. 구현 중 택일 가능.

### 5.4 hysteresis — **dual-threshold enter 0.08 / exit 0.12, CPU 적용** (3/3)
- 저조도 상태 래치를 enter<0.08·exit>0.12로 (gate 기본 0.10 + 셰이더 soft band ±0.03 정합). 시간기반 N-frame 조건 미사용(지연 누적 회피). CPU/renderer luma 업데이트 로직에 상태 추가.

### 5.5 LUMA 계수 + 테스트 — **Rec.709 linear, `srgb*srgb` 매칭, ≤1% 테스트** (3/3)
- producer 측정: 각 픽셀 **`srgb*srgb`**(셰이더 `toLinearFast`, `shader_sources.cpp:851`와 동일 fast 근사 — 정확 sRGB 곡선 아님) 후 `dot(linear, (0.2126,0.7152,0.0722))` 평균. sRGB 평균 금지.
- 단위 테스트: black/white/gray/R/G/B + 64×64 synthetic ROI에 대해 CPU producer helper vs shader-equivalent reference 비교, 오차 ≤1% assert (GoogleTest).

### 5.6 블렌드 파급 — **메인 블렌드 재벤치 + K/clamp 재튜닝 + A/B 토글** (R2, 3/3)
- **발견**: `uAvgIrisLum`은 detail/gate뿐 아니라 **메인 블렌드 ID=5(`blendTintLinearV2`) + ID=7(`blendColorReplaceLinear`)의 정규화 분모**. `tinted = blendL·0.85·(lum/avgLuma)` = brightness-invariant 틴트 설계. 현 0.1225 고정은 scale 7.0 포화 → 밝은 조건 over-tint (= 실측으로 교정할 결함, 3/3).
- **DoD 확장**: W2는 "측정 연결"이 아니라 **측정 연결 + 메인 블렌드 재벤치(6 SKU, 특히 밝은 환경) + 필요 시 K/clamp 재튜닝**. fallback↔실측 A/B 토글 보존(`setDetailReinject`류 internal API).
- **재튜닝**: K=0.85 1차 유지(3/3, 평균 픽셀 틴트 타겟). clamp는 실측 분포 로그 후 재튜닝 — **하단 clamp 0.8→1.0~1.2 상향**(Gemini, 밝은 환경 최소 시인성)을 1순위 후보로 검증. 측정과 상수 동시 변경 금지(over-tint 해소 원인 분리, Codex).
- 🎁 **재튜닝 prior-art**: Kotlin fallback 셰이더(`CameraGLRenderer.kt:237`)가 실측 luma(~0.35)로 이미 `scale=0.5/luma, clamp[0.8,2.5]` 튜닝됨 → SDK 실측 재튜닝 출발점.

### 5.7 측정=정규화 정합 계약 (R2.4, 3/3)
- detector `rgb_buffer`에 **셰이더와 동일** 변환 강제: 각 픽셀 `srgb*srgb`(`toLinearFast`) → `dot(0.2126,0.7152,0.0722)` → iris ROI(±0.65r) 평균.
- geometric 차이(SDK uCameraTexture는 SurfaceTexture matrix/mirror/flip 적용, detector는 NV21+rotation)는 **평균 luma에 무영향**(orientation-invariant). 색공간/ROI 정합 + systematic offset(K 흡수)·noisy(EMA 완화)만 관리.
- ⚠️ 기존 P4-W1-03 NV21 **Y채널 5점 샘플 재사용 금지**(`GpuRenderActivity.kt:1078~` — Y≠srgb²+Rec.709, 5점≠ROI평균). SDK producer는 신규.

### 5.8 구조 메모 (브레인스토밍 R2 발견)
- 렌즈 렌더는 **이중 경로**: PRIMARY=C++ SDK `renderLensTexture`(실제 렌더), FALLBACK=Kotlin `renderLensOverlay`(SDK 실패 시만, dormant). 기존 P4 producer는 **dormant Kotlin fallback에만** 연결 → 실제 렌더 SDK 경로는 여전히 0.1225. **W2는 SDK 경로 producer 신규**가 맞음.
- SDK↔Kotlin 셰이더 상수 불일치는 메모리 `w9-demo-ui-sync` 부채 = **P7-W3 영역**(W2 아님).

---

## 6. 미결 사항 (R1로 6.1~6.5 닫힘 → §5. 잔여는 구현 중 판정)

### 6.1~6.5 — ✅ 닫힘 (R1 3모델 합의 + 코드 검증, §5 이동)

### 6.6 구현 중 판정 (저위험)

- **detector rgb_buffer 좌표공간/해상도**: iris center/radius가 rgb_buffer 좌표계인지 확인 후 ROI 매핑. C 실현성엔 무영향.
- **N=5 vs 매-프레임** (§5.3 소수안): hold/N 결합 번거로우면 N 제거 택일. 둘 다 저위험.
- **측정 호출 지점**: detection 완료 직후 eye별 측정 → IrisResult/packet (consumer가 양안 평균).

### 6.7 R2 발견 → 다른 W로 분리 (W2 범위 밖)

- **SDK↔Kotlin 셰이더 상수 불일치** (SDK `0.85/[0.8,7.0]@0.1225` vs Kotlin fallback `0.5/[0.8,2.5]@~0.35`): 메모리 `w9-demo-ui-sync` 부채 → **P7-W3**. W2는 SDK 경로만 다룸.
- **dormant Kotlin fallback producer 정리** (`sampleIrisLuminanceNv21` 등): SDK 경로 안정 후 dead-code 여부 판단 → P7-W3 cleanup 후보.
- **ColorReplaceLinear(ID=7) 채택/제거** (메모리 `w5-b1-color-replace-decision`) + **블링크 ramp(B5) 튜닝**: W2 §6 결정 보류 (사용자 2026-06-09). W2는 avg_iris_luma 측정+블렌드 재벤치까지, ID=7 활성결정·blink는 후속.

---

## 7. 체크리스트 (브레인스토밍용)

### 7.1 읽을 파일

- `docs/workPaper/P7-W2_avg_iris_luma_measure.md` (본 문서)
- `docs/workPaper/P7-W0_index.md` §2.P7-W2
- `cpp/src/gpu/gpu_lens_renderer.cpp` (`updateAvgIrisLuma` :800~, uniform push :1098~)
- `cpp/include/iris_sdk/gpu/eye_render_packet.h` (avg_iris_luma 필드)
- `cpp/src/gpu/eye_render_packet_adapter.cpp` (producer 미설정 지점)
- `android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLRenderer.kt` (OES→2D, rgbaTexture/Fbo, convertOesToRgb)

### 7.2 송신 프롬프트 (Codex/Gemini 동일)

> `@docs/workPaper/P7-W2_avg_iris_luma_measure.md` 읽고, §6 미결 5건(6.1~6.5) 각각 "추천 + 근거 1~2줄". 특히 6.1(측정 위치 A/B/C)이 핵심 — EXTERNAL_OES 충돌·GL state 오염·30fps 예산 기준으로 판정. 필요 시 `@cpp/src/gpu/gpu_lens_renderer.cpp` 등 실제 코드 인용. 새 쟁점 제기 금지. 한국어. 저장: `docs/workPaper/P7-W2_brainstorm/{codex|gemini}_w2.md`.

---

## 8. 변경 이력

| 날짜 | 변경 |
|---|---|
| 2026-06-08 | 초안 작성 (코드 실측 기반 §6 미결 5건). `ar-lens-brainstorm P7-W2` R1 진행. |
| 2026-06-08 | **R1 완료**. Codex+Gemini+Claude 3모델: 만장일치 4(6.1 C/6.2 CPU/6.4 hysteresis/6.5 Rec.709) + 다수결 1(6.3 N=5). 코드 주장 4건 직접 검증(rgb_buffer/toLinearFast=srgb²/hold=3/LUMA709). §5 확정 이동, §6 닫힘. 구현 대기. |
| 2026-06-09 | **R2 완료** (블렌드 파급 집중). 3/3: uAvgIrisLum이 메인 블렌드(ID=5/7) 정규화 분모 = brightness-invariant, 0.1225은 교정할 결함, (a)실측+재튜닝+토글, srgb²+Rec.709 정합. §5.6~5.8 추가, DoD에 메인 블렌드 재벤치/A-B 토글 추가. **구조 발견**: 이중 렌즈 경로(SDK PRIMARY/Kotlin fallback), 기존 P4 producer는 dormant fallback 전용 → SDK 경로 producer 신규 확정. SDK↔KT 상수 불일치는 P7-W3로 분리. |
| 2026-06-09 | **코드 구현 완료** (cpp-pro 위임 + Claude 검증). 5컴포넌트: detector 측정(srgb²+Rec.709 ±0.65r)/어댑터 wiring(>0 가드)/A-B 토글 7단 체인/gate 전용 hysteresis(uLowLightActive, 블렌드 격리)/demo UI(`lum:fb`↔`lum:meas`). 완전성 수정: IrisResult default-init(-1) + 어댑터 >0(memset 0 거부). FFI: static_assert(sizeof) + C/C++ 동기 + JNI round-trip. 검증: iris_sdk 빌드, LUMA 7/7, test_types(triviality 불변), **Android BUILD SUCCESSFUL**. 매-detection 측정 택일 → hold 변경 불요. 실기기 재벤치 대기. |
| 2026-06-09 | **코드 구현 완료**. 5개 컴포넌트: (1) Producer `Impl::calculateIrisLuma` (srgb²+Rec.709, ±0.65r 원형마스크, [0.01,0.81] clamp, -1 sentinel) + `IrisResult.avg_iris_luma_{left,right}` 추가 (C++/C 양쪽 struct 끝, reinterpret_cast 정합 `static_assert`로 가드, JNI 마샬·Java IrisResult 양방향 round-trip). (2) 어댑터 wiring (eye별 -1이면 nullopt 유지). (3) A/B 토글 `setUseMeasuredLuma(bool)`=false 기본 (C API `iris_sdk_set_use_measured_luma` internal + JNI `nativeSetUseMeasuredLuma` + Java wrapper). (4) hysteresis: gate 전용 `uLowLightActive` 신규 uniform (enter0.08/exit0.12 CPU 래치 → `gateStrength *= (1-uLowLightActive)`), uAvgIrisLum 블렌드 분모 불변. (5) 단위 테스트 `test_iris_luma_measure` 7건 통과 (≤1% 색공간 정합). `iris_sdk` 빌드 통과. hold=3 유지(매-detection 측정이라 §5.3 N=5 상향 불요). |
