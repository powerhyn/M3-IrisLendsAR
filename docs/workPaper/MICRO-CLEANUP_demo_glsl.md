# MICRO-CLEANUP — demo blend UI + GLSL §8.9 종결 + LensConfig canonical 승격

> 2026-07-02. EYECLIP 트랙 종결 직후, 이월 후보 2건([[w9-demo-ui-sync]] 잔여 + [[p7-residual-glsl-spec-cleanup-unassigned]])을 묶은 소형 트랙.
> 브랜치 `chore/micro-cleanup-demo-glsl` (develop 기반). 조사 = 4-agent 병렬(§8.9는 2관점 교차 스캔) + 종합.

## 0. 착수 전 현행 확인으로 밝혀진 스테일 (메모리 대비)
- ~~KT fallback 렌즈 셰이더 제거~~ → **이미 해소**: `renderLensOverlay`/`sampleIrisLuminanceNv21` 현행 0건(리팩토링 ④ 추정). 남은 KT 셰이더 3종(VERTEX/OES→2D/PASSTHROUGH)은 카메라 변환용 기반 — 제거 대상 아님.
- ~~P7 §8.9 후보 2건(uLutTexture/FREQ_SEP)~~ → **코드 소멸**(P8 뷰티 개편): `uLutTexture` 0건, `FREQ_SEP` 0건.

## 1. §8.9 GLSL 스캔 — 코드 수정 0건으로 종결 ✅
2관점 교차 스캔(셰이더별 순회 A ↔ 분기/루프 역추적 B, 상호 누락 검증) 결과:

- **신규 위반 0건.** P8에서 live화된 beauty 셰이더(SKIN_SEPARABLE_BLUR/SMOOTH_COMPOSITE/WARP)는 모든 fetch가 분기 밖 convergent + 비-mipmap, varying 분기 내부는 전부 ALU-only. EYECLIP A-2 `contourEyelidMask`도 순수 ALU 재확인. P7-W1의 textureLod 9샘플 회귀 없음.
- **uniform-분기 내 implicit-LOD 3건 처분**:
  | 위치 | 상태 | 처분 |
  |------|------|------|
  | L479 `texture(uLensTexture)` | mipmap + live | **현행 유지** — textureLod(0.0) 치환은 anti-shimmer mipmap 목적 무력화(유일 안전 치환=textureGrad), P6 이래 장기 실기기 무사고 |
  | L426 `texture(uEnvMap)` | mipmap + dormant(reflection_mode_=0 기본) | **Phase 9 env 반사 재개 시 textureGrad 전환 필수** ([[w4-env-reflection-deferred]] 체크리스트 부기) |
  | L443 `texture(uCameraTexture, ringUV)` (Periphery 8점 링) | 비-mipmap + dormant | **Phase 9 재개 시 textureLod(…,0.0) 1줄 치환**(비-mipmap이라 시각 무료 — P7-W1 L567 주석 논리) |
- 스캔 불일치 1건(L443 safe vs suspect)은 **suspect 채택**: P7-W1 0x501 사건의 수정 대상이 바로 비-mipmap GL_LINEAR uCameraTexture의 uniform-분기 내 fetch였음 → "mipmap 아니면 무해" 논거는 역사적 사실과 모순. 드라이버 컴파일러는 분기 내 implicit-derivative만으로 발화.
- ⚠️ Phase 9 env 반사 재개 트랙의 실기기 항목으로 이월: L426/L443 치환 적용 후 Adreno(S23+)에서 0x501 무발화 + EnvMap/Periphery 시각 확인.

## 2. blend 드롭다운 8종→5종 축소 ✅ (GpuRenderActivity.kt)
- `blendModeEntries` 클래스 필드 승격 + 활성 ID {0,1,2,5,7}만: Normal/Multiply/Screen Linear/Lum Tint Linear/Color Replace. deprecated 3/4/6(Overlay/LumTint/SoftLight)은 셰이더가 ID5 fallback시키는 거짓 UI라 제외.
- **지뢰 처리(조사 발견)**: `applyBenchCombo`가 `setSelection(combo.blendMode)`로 blend ID를 스피너 position으로 직접 사용 — 축소 시 벤치 C/D(ID=7)가 어댑터 범위(0..4) 초과 → IndexOutOfBounds **크래시**. 값 기반 역조회(indexOfFirst)로 치환(원자 적용).
- 초기 위치 동기화(:433 indexOfFirst)는 이미 값 기반이라 무변경 — 기본 ID5가 새 idx3에 자동 매핑.
- stale P6-W1 주석(:411-414 "W2 재검토 예정") 제거.
- SDK surface(LensConfig.java 8종 상수/getBlendModeName, BlendMode.kt)는 sdk_api.h "enum 값 보존" 의도대로 불변.

## 3. LensConfig scale/edgeFeather canonical 승격 ✅ (cpp-pro)
- **실체**: 실기기 튜닝 1.3/0.15(커밋 989fdac→6f596a2 "실제 컬러렌즈에 가까운 크기")가 Java LensConfig에만 반영되고 native(types.h 1.0/0.1)/Kotlin(LensConfigKt)/JNI null-fallback으로 전파 누락 — 의도된 이원화가 아니라 **표면 드리프트**(opacity/blendMode는 P6-W2 §5.12에서 이미 canonical 승격, scale/feather만 누락된 반쪽 규약).
- **처방**: Java 값(1.3f/0.15f)을 canonical로 승격 — types.h + sdk_api.cpp + sdk_api.h 주석 + LensConfigKt + iris_jni null-fallback + 테스트 단정(test_sdk_api/test_types). **데모는 매 프레임 Java 객체 마샬링이라 비트 동일 = 시각 변화 0.** 역방향(Java→1.0)은 실기기 체감 기준 파괴라 금지.
- 향후 골든 재캡처 시 .render.png 픽셀 변화 가능성 메모(golden 게이트 자체는 landmark JSON만 비교라 불파손).

## 4. 후속 후보 (이번 트랙 범위 밖 — 기록만)
- SDK Java/Kt blend 상수에 @Deprecated 마커 부재(sdk_api.h:138-141과 불일치).
- LensConfigKt에 rotation/isMirror 필드 결손(KT 공개 API 시그니처 변경 사안).
- CHANGELOG에 C API 기본값 변경(1.0/0.1→1.3/0.15) 명기.

## 5. 검증 게이트
- [ ] cpp 빌드 + test_sdk_api/test_types 통과 (cpp-pro)
- [ ] demo assembleDebug
- [ ] 실기기(blend): 초기 스피너 = "Lum Tint Linear"(idx3) / 벤치 A~D 토글 무크래시 + 스피너 동기 / 5종 전환 로그 ID(0/1/2/5/7) 정합
- [ ] 실기기(lensconfig): 렌즈 크기·feather 육안 "변화 없음" = 통과 (변하면 즉시 롤백)
