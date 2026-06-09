# P7-W2 R2 종합 — 블렌드 파급 + 구조 발견 (Claude)

> R2 단일 이슈(uAvgIrisLum 블렌드 파급) 3모델 + 코드 검증. R1(6.1~6.5)은 불변.

## R2 합의 표

| 쟁점 | Codex | Gemini | Claude | 판정 |
|---|---|---|---|---|
| R2.1 정규화 해석 | 맞음 | 맞음 (brightness-invariant) | 맞음 | **3/3 ✅** |
| R2.2 0.1225 = 버그? | 고칠 fallback | 고칠 결함 | 불완전(결함) | **3/3 → 실측으로 교정** |
| R2.3 clamp/K | K유지, clamp 1차 유지 후 데이터 재튜닝 | K유지, **하단 clamp 0.8→1.0~1.2 상향 검토** | K유지, clamp 재벤치 | **K=0.85 3/3 유지** · clamp 데이터 기반(Gemini안=후보) |
| R2.4 측정=정규화 정합 | 같다고 볼 수 없음, srgb²+Rec.709+ROI 계약 강제 | rgb_buffer=sRGB, (srgb)²+Rec.709 강제 | systematic 오차는 K 흡수, noisy는 EMA | **3/3 → srgb²+Rec.709 강제** · 정합 검증 필요 |
| R2.5 보존 전략 | (a) 실측+재튜닝 | (a) 실측+재튜닝 | (a) 실측+재벤치+토글 | **3/3 → (a), (c)분리 비추천** |

## 검증된 코드 사실 (R2 신규)

1. **이중 렌즈 파이프라인 (구조)**: 
   - **PRIMARY = C++ SDK** `renderLensTexture`(`CameraGLRenderer.kt:1248`) → `shader_sources.cpp` uAvgIrisLum=0.1225(producer 부재). 실기기 렌더 경로(`nativeRenderLensTexture` 확인).
   - **FALLBACK = Kotlin** `renderLensOverlay`(`CameraGLRenderer.kt:1232,1237,1265`) — SDK 실패 시만. dormant.
2. **기존 P4-W1-03 producer는 fallback 전용**: `sampleIrisLuminanceNv21`(`GpuRenderActivity.kt:1078~`) → `setRawIrisLuminance`(`CameraGLView.kt:91`) → Kotlin `updateAvgIrisLum`(`CameraGLRenderer.kt:1541`, EMA α=0.1, 기본 0.35) → **Kotlin fallback 셰이더 uAvgIrisLum**(`:976`). SDK 경로엔 미연결. → R1 "SDK producer 부재" 정확.
3. **재튜닝 prior-art**: Kotlin fallback은 실측 luma(~0.35)로 `scale=clamp(0.5/uAvgIrisLum,0.8,2.5)`(`CameraGLRenderer.kt:237`). SDK는 가짜 0.1225용 `0.85/.../[0.8,7.0]`. → 실측 연결 시 재튜닝 출발점 = Kotlin fallback 상수 영역(numerator↓ ~0.5, clamp 상한↓).
4. **NV21 Y채널 샘플 재사용 금지**(Codex): 색공간(Y≠srgb²+Rec.709)·5점(≠ROI평균) 부적합. SDK용 producer는 신규 작성.
5. **셰이더↔Kotlin 상수 불일치 = `w9-demo-ui-sync` 부채 → P7-W3** (W2 범위 밖).

## 결론

- R2는 6.1=C를 불변. **W2 DoD 확장**: 측정 연결 → 측정 연결 + **메인 블렌드(ID=5/7) 재벤치 + K/clamp 재튜닝 + fallback↔실측 A/B 토글**.
- 측정 정합: detector rgb_buffer에 **셰이더와 동일 `srgb*srgb`+Rec.709+iris ROI** 강제. geometric(mirror/flip/rotation)은 평균 luma에 무영향(orientation-invariant), 색공간/ROI/systematic offset만 관리(K 흡수).
- R3 불필요. §5 확정 가능.
