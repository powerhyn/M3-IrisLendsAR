# P8-W1: 피부 보정(skin smoothing) 이식 — LensSimulator 검증 스펙

> **상태**: 🔄 구현 중 (2026-06-10)
> **작성**: 2026-06-10
> **성격**: P7 진행 중 사용자 요청으로 삽입된 **선행 P8(뷰티) 트랙**. P7-W4(재베이스라인 대기)와 독립 병렬.
> **출처 스펙**: `docs/skin-smoothing-handoff-from-lenssimulator.md` (LensSimulator/CGG에서 S23+ 실기기 검증 완료 — "아주 마음에 듦", 30fps 유지, 적대적 리뷰 통과)
> **브레인스토밍 생략 사유**: 알고리즘·파라미터·함정이 원 프로젝트에서 이미 실기기 검증+적대적 리뷰로 확정됨. 본 W의 결정 사항은 "IrisLensSDK 통합 지점"만이며 §1.3에 코드 실측 기반으로 기록.

---

## 1. 인사이트 (세션 간 맥락 보존) ⭐

### 1.1 알고리즘 (핸드오프 §개요 — 변경 없이 이식)

```
① 스킨 마스크 (1/4 해상도 R8): FACE_OVAL 36점(값1) − 눈썹×2/입술/눈×2(값0 덮어쓰기)
   → TRIANGLE_FAN(무게중심) → 가우시안 블러 1패스(페더)
② 컬러 블러 (1/4 해상도 RGBA): 분리형 가우시안 5-fetch H/V, offsetScale 1.6 (마스크는 1.0)
③ 컴포지트 (풀해상도): 에지 가드 smoothstep(0.06, 0.18, |lumaBase−lumaBlur|)
   × 디테일 50% 보존 (smoothed = blur + (base−blur)·0.5)
   base = mix(base, smoothed, mask · strength · (1−edge))
```

**에지 가드가 품질의 핵심** (없으면 전체 뿌옇게 뭉개짐 — 원 프로젝트 실기기 확인 실패 모드).
5-fetch 계수: weights `0.2270270 / 0.3162162 / 0.0702703`, offsets `1.3846154 / 3.2307692 × offsetScale`.

### 1.2 SDK 측 코드 실측 (2026-06-10, Explore 검증)

| 사실 | 근거 |
|---|---|
| **face_mesh 478점 전체가 이미 노출** — `IrisLandmark face_mesh[478]` + `face_mesh_valid` | `sdk_api.h:195-196` (직접 확인) |
| 기존 뷰티 = C++ `GPUBeautyBackend` (FreqSep 6-pass / Bilateral fallback / Combined color / Vivid) | `gpu_beauty_backend.h`(570줄) / `.cpp`(2105줄) |
| GPU 뷰티 진입점이 detection을 이미 받음 | `iris_sdk_apply_beauty_texture_v2(..., const IrisResult* detection, ...)` `sdk_api.h:750` |
| demo 파이프라인 순서: OES→RGBA → **렌즈** → **뷰티** → 화면 | `CameraGLRenderer.kt:722-771` |
| One-Euro 필터 C++ 구현 기존재 (`OneEuroFilter`/`2D`) | `cpp/include/iris_sdk/one_euro_filter.h` |
| `IrisBeautyConfigV2`는 공개 POD — **ABI 고정, 필드 추가 금지** | `sdk_api.h:637-676` + C8 원칙 |

### 1.3 통합 아키텍처 결정 (본 W의 유일한 신규 결정)

**C++ `GPUBeautyBackend` 내부 신규 스무딩 경로**로 구현 (demo Kotlin 주입 기각 — `w9-demo-ui-sync` 부채 패턴 재발 방지, C++ 코어 원칙):

- **신규 모드**: landmark-masked smoothing. 활성 시 FreqSep/Bilateral 스무딩을 대체(블렌드 정규화 등 다른 패스는 불변). 기존 FreqSep과 **internal 토글로 A/B 비교 가능** (프로젝트 토글 문화, `qualitative-device-judgment`).
- **랜드마크 소스**: `detection->face_mesh` (478, `face_mesh_valid` 가드). 추출 자체는 detection 부산물이라 추가 비용 0 — One-Euro 필터링만 모드 활성 시 수행.
- **파라미터 노출**: 공개 config 불변. **internal C API**(W6 `setDetailReinject`류 전례) + JNI + demo 토글 체인.

### 1.4 LensSimulator 함정 → IrisLensSDK 매핑

| 핸드오프 함정 | IrisLensSDK 해당 여부 |
|---|---|
| ① 렌즈 패스가 보정 결과 덮어씀 (알파 블렌딩 전환이 blocker였음) | **구조적 비해당** — 본 SDK는 렌즈 **먼저** → 뷰티 나중 (`CameraGLRenderer.kt:741→759`). 뷰티가 렌즈 합성 결과를 입력으로 받고, 눈 제외 폴리곤이 렌즈 영역을 마스크에서 빼므로 렌즈 시각 불변. 알파 전환 불필요. |
| ② 마스크/블러를 컴포지트와 동일 공간 UV로 샘플 | **해당** — 본 SDK 입력은 이미 2D RGBA(카메라 프레임 공간)이고 랜드마크 정규 좌표도 동일 공간 → OES/ST행렬/크롭/미러 체인 전부 불필요(원본 COMPOSITE_FS 대비 대폭 단순화). 마스크 팬 정점을 입력 텍스처 공간 NDC로 생성하면 정합. |
| ③ 제외 폴리곤은 마스크 블러 **전에** 드로우 | **해당** — 패스 순서 그대로 유지. |

추가 비해당: 원본의 턱선 워프(uWarp)·크롭·ST행렬·페이드 로직은 **이식 범위 밖** (워프는 P8 본 작업에서 별도, 페이드는 SDK 쪽 visibility 체계 기존재).

### 1.5 원본 구현 참조 좌표 (Explore 분석, 이식 시 대조용)

- 패스 구성/저해상도 5타깃(RGBA8 BT_LOW/TMP/BLUR + R8 BT_MASK/MASK_BLUR): `Renderer.kt:103-107, 840-868`
- 팬 정점 카운트 `[38,12,12,22,18,18]`(무게중심+N+첫점반복): `Renderer.kt:1070, 927-944`
- BLUR_FS/COMPOSITE_FS/MASK_FS 원문: `Renderer.kt:1108-1210`
- 이마 확장 `extendForehead`(+단위 테스트 5케이스): `BeautyGeometry.kt:40-61`, `BeautyGeometryTest.kt:155-179`
- 비용 게이팅(강도0→패스 생략, lazy 타깃, 추출 게이팅): `Renderer.kt:574-600, 452-475`, `FaceTracker.kt:143-149, 294-297`
- One-Euro: `MIN_CUTOFF=0.5, BETA=0.007, D_CUTOFF=1.0`, **픽셀 공간** 적용: `FaceTracker.kt:419-427, 373-374`

---

## 2. 배경/맥락

- P8(뷰티)은 P7 종료 후 예정이었으나(P7-W0 §8), LensSimulator에서 검증 완료된 피부 보정의 이식을 사용자가 직접 요청 (2026-06-10) → 선행 트랙으로 삽입.
- 기존 `GPUBeautyBackend`의 FreqSep은 face_rect 기반 — 랜드마크 폴리곤 마스크 방식이 아니어서 헤어라인/눈썹/입술 경계 정밀도가 다름. 본 이식은 검증된 룩을 그대로 가져오는 것이 목표.

## 3. 전제 조건

1. ✅ `face_mesh[478]` 노출 + `apply_beauty_texture_v2`가 detection 수신
2. ✅ One-Euro C++ 구현
3. ✅ S23+ 실기기 (최종 육안 검증용)

## 4. 목표 + DoD

1. LensSimulator와 **동일 파라미터·동일 룩**의 피부 보정이 GPU 뷰티 경로에서 동작
2. 비용 규율: 강도 0 → 추출/필터/패스/메모리 전부 0
3. 기존 FreqSep 경로 무회귀 (모드 OFF 시 바이트 동일 동작)

### 4.1 Definition of Done

- [ ] **(코드)** 스킨 마스크 패스 (1/4 R8, 6팬, 이마 확장 0.35, One-Euro 0.5/0.007 픽셀 공간)
- [ ] **(코드)** 분리형 가우시안 5-fetch (컬러 1.6 / 마스크 1.0) + 1/4 타깃 lazy 생성
- [ ] **(코드)** 에지 가드 컴포지트 (0.06/0.18, 디테일 0.5) — 확정 파라미터 그대로
- [ ] **(코드)** 비용 게이팅 (강도 0 → 전부 생략) + internal C API + JNI + demo 토글
- [ ] **(테스트)** 이마 확장 GoogleTest (원본 5케이스 포팅)
- [ ] **(빌드)** `cmake --build --target iris_sdk` 통과 + Android 빌드 통과
- [ ] **(실기기)** S23+ 육안: LensSimulator 룩 재현 + 에지(코선/안경/머리카락) 선명 유지 + FreqSep 대비 A/B + 30fps
- [ ] **(실기기)** 모드 OFF 무회귀

## 5. 확정 사항 (핸드오프 상속 — 재논의 금지)

| 파라미터 | 값 |
|---|---|
| 에지 가드 | `smoothstep(0.06, 0.18, |lumaBase−lumaBlur|)` (luma = sRGB 0.299/0.587/0.114) |
| 디테일 보존 | 0.5 |
| 컬러 블러 offsetScale | **1.6** (2.0은 뭉개짐 — 실기기 튜닝값) |
| 마스크 블러 offsetScale | 1.0 |
| 이마 확장 | 0.35 (광대 454·234 중점 피벗, 턱152→이마10 축 방향, t>0만) — **마스크 전용** |
| 저해상도 | 뷰포트 1/4, RGBA8×3 + R8×2, lazy |
| 랜드마크 | FACE_OVAL 36 / BROW 10×2 / LIPS 20 / EYE 16×2 (핸드오프 §랜드마크 표 그대로) |
| One-Euro | min_cutoff 0.5 / beta 0.007 / d_cutoff 1.0, 픽셀 공간, 모드 활성 시만 |

## 6. 미결 (구현 중 판정, 저위험)

- internal API 네이밍 (`iris_sdk_set_skin_mask_smoothing(enabled, strength)` 류)
- 기존 FreqSep 디버그 모드(`set_freqsep_debug_mode`)와의 토글 UI 공존 방식
- 좌표 정합 세부: 랜드마크 정규 좌표 ↔ 입력 텍스처 공간 (기존 beauty face_rect 변환 패턴 따름 — 구현 시 검증)

## 7. 커밋 전략 (분할)

1. `feat(gpu-beauty): P8-W1 스킨 마스크 패스 (랜드마크 폴리곤 + 이마 확장 + One-Euro)`
2. `feat(gpu-beauty): P8-W1 1/4 해상도 분리형 가우시안 + lazy 타깃`
3. `feat(gpu-beauty): P8-W1 에지 가드 컴포지트 + 비용 게이팅`
4. `feat(sdk): P8-W1 internal API + JNI + demo 토글`
5. `test: P8-W1 이마 확장 GoogleTest`

## 변경 이력

| 날짜 | 변경 |
|---|---|
| 2026-06-10 | 초안 — 핸드오프 스펙 + SDK/원본 양측 Explore 실측 + 통합 아키텍처 결정(§1.3). 구현 착수. |
