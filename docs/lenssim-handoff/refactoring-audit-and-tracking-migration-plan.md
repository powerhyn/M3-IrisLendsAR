# 리팩토링 감사 + 추적 레이어 전환 계획

> 작성: 2026-06-10, LensSimulator 세션에서 전달 (사용자 검토 후 진행)
> 배경: IrisLensSDK 코어는 이전 모델로 작성되어 품질 감사가 필요하고,
> LensSimulator 기술검증에서 "추적은 MediaPipe Tasks 공식 바인딩 주입 + 코어는 이펙트/렌더 전담"
> 구조의 이점이 확인됨. 두 작업을 안전하게 묶어 진행하기 위한 단계 계획.

## 원칙

- **품질 리팩토링(동작 불변)과 아키텍처 변경(추적 레이어 교체)을 같은 단계에서 섞지 않는다** —
  문제 발생 시 원인 분리가 불가능해진다.
- 각 단계 종료 시 SDK는 항상 동작하는 상태를 유지한다 (스트랭글러 패턴).
- 모든 동작 불변 단계는 골든 베이스라인으로 검증한다.

---

## 1단계 — 감사 리뷰 (현재 코드 품질 측정)

멀티에이전트 적대적 리뷰 (관점별 독립 에이전트 + findings 스키마). 대상: `cpp/` 코어 전체 + 플랫폼 바인딩.

### 리뷰 관점 (각각 별도 에이전트)

| 관점 | 검토 내용 |
|---|---|
| 정확성 | 좌표계 변환(회전/미러/정규화↔픽셀), 랜드마크 인덱스 사용, 수치 안정성(퇴화 입력), 알려진 함정 대조 |
| 스레드/수명 | 카메라·추론·렌더 스레드 경계, 공유 상태 보호, 종료/재시작 경로 누수 |
| GL/GPU 상태 | 컨텍스트 관리, FBO/텍스처 수명, 프레임당 할당, 상태 누수 |
| **결합도 (교체 가능성 평가 — 핵심)** | TFLite 추적 파이프라인과 이펙트/렌더 코어의 결합 지점 전수 조사: 랜드마크가 어디서 어떤 타입으로 흘러가는지, "랜드마크 주입 인터페이스"를 끼울 자리, 추적 제거 시 딸려 나가는 코드 범위 |
| 플랫폼 패리티 | Android/iOS/Flutter/Web 바인딩 간 동작·파라미터 불일치 |
| 테스트 안전망 | 커버리지 실태, 회귀 테스트 부재 영역, 골든 테스트 구축 가능 지점 |

### 산출물

1. `docs/lenssim-handoff/audit-report.md` — 관점별 findings (severity: blocker/major/minor, 파일·근거·수정안 필수)
2. **모듈별 판정표**: 리팩토링 / 부분 재작성 / 유지 + 근거
3. **추적 레이어 교체 난이도 견적**: 결합 지점 목록 + 인터페이스 도입 작업 범위

### 리뷰 시 참조할 외부 자료 (LensSimulator 검증 산출물)

- `/Volumes/M3-P31/Projects/TedMong/CGG/LensSimulator/docs/tech-validation.md` — **"결정적 함정 목록" 13건**
  (홍채 경계점 순서, 정규화 좌표 거리 계산 금지, MediaPipe 회전 재투영, ST 행렬 이중 회전,
  One-Euro 파라미터 스케일 전제 등 — 기존 코드가 이 함정들을 밟고 있는지 대조 검사)
- `/Volumes/M3-P31/Projects/TedMong/CGG/LensSimulator/docs/research/` — MediaPipe/렌더링 기법 리서치 11건

---

## 2단계 — 결정 + 안전망 구축

### 2-1. ADR 작성: 추적 레이어 주입형 전환

감사 결과(결합도 견적)를 근거로 확정. 목표 아키텍처:

```
┌────────────────────────────────────────────────────┐
│ iriscore (C++ 단일 코어)                            │
│  입력: 카메라 텍스처 + 랜드마크 478점 + 제어 파라미터 │
│  처리: 지오메트리(One-Euro·기저벡터·워프) + 이펙트    │
│  ※ 추적(TFLite/MediaPipe)은 코어 밖                 │
└──────┬──────────────┬───────────────┬──────────────┘
       │ NDK .so      │ static lib    │ wasm
  Android AAR     iOS XCFramework   Web npm
  Kotlin 글루       Swift 글루        TS 글루
  + MediaPipe       + MediaPipe      + @mediapipe/
    Tasks (POM)       Tasks(prelink)   tasks-vision
```

- C API 경계(안): `iris_set_landmarks(float* pts478x3, int64 ts)` / `iris_set_lens(...)` /
  `iris_set_beauty(skin, jaw)` / `iris_render(target, w, h)` — 프레임 픽셀은 경계를 넘지 않음
- 전환 근거: 모델/delegate/16KB 정렬 관리를 구글에 위임, 플랫폼 공식 바인딩의 유지보수 승계,
  Web 포함 전 플랫폼에 공식 배포 채널 존재 (버전 0.10.35 동시 배포 확인됨)
- 기각 가능: 감사 결과 결합도가 지나치게 높으면 "현 추적 유지 + 인터페이스만 도입"으로 축소

### 2-2. 골든 베이스라인 캡처 (리팩토링 착수 전 필수)

- 표준 테스트 영상 세트(정면/측면/저조도/안경)에 대해 **현재 코드**의:
  ① 랜드마크 출력(프레임별 좌표 덤프) ② 렌더 출력(프레임 이미지) 저장
- 이후 모든 동작 불변 단계는 이 베이스라인과 비교 검증 (좌표 ε 허용, 이미지 픽셀 diff 허용치)
- AI 에이전트 대규모 리팩토링에서 "조용한 동작 변화"를 잡는 유일한 안전망

---

## 3단계 — 단계적 실행 (각 단계 독립 PR, 종료 시 항상 동작 상태)

### ③-1. 경계 도입 (동작 불변)

- 랜드마크 주입 C API 인터페이스 신설
- **기존 TFLite 추적을 그 인터페이스 "뒤로" 이동** (코어 외부의 기본 추적 공급자로 재배치)
- 골든 베이스라인 통과 = 완료 기준

### ③-2. 코어 품질 리팩토링 (동작 불변)

- 1단계 findings 반영 (blocker → major → minor 순)
- 모듈별 판정표 기준: 재작성 판정 모듈은 LensSimulator 검증 코드를 레퍼런스로 활용
- 골든 베이스라인 통과 = 완료 기준

### ③-3. 플랫폼별 추적 교체 (점진)

- **Android 먼저** (실기기 검증 가능 + LensSimulator 글루 재활용):
  CameraX RGBA_8888 → ByteBufferImageBuilder → FaceLandmarker LIVE_STREAM(GPU delegate+CPU 폴백)
  — 검증 완료 코드: `LensSimulator/sdk/android/.../internal/FaceTracker.kt`, `CameraController.kt`
- 같은 코어에 기존 추적 vs 공식 바인딩을 꽂아 **A/B 비교** (정확도·지연·안정성) → 데이터로 전환 확정
- iOS → Web 순차 진행. 완료 후 자체 TFLite 파이프라인 제거

### 재활용 가능한 LensSimulator 자산

| 자산 | 경로 (루트: /Volumes/M3-P31/Projects/TedMong/CGG/LensSimulator) |
|---|---|
| Android 추적 글루 (검증본) | `sdk/android/.../internal/FaceTracker.kt`, `CameraController.kt` |
| iOS 추적 글루 (검증본) | `sdk/ios/LensSDK/Sources/CameraSource.swift`, `FaceTracker.swift` |
| 순수 함수 + 테스트 (C++ 이식 용이) | `.../math/OneEuroFilter.kt`, `CoordMapper.kt`, `IrisGeometry.kt`, `BeautyGeometry.kt` + `src/test/` |
| 렌즈/뷰티 셰이더 (GLSL) | `.../render/Renderer.kt` 하단 상수 |
| Flutter 텍스처 브리지 | `sdk/flutter/android·ios` 글루 |
| MediaPipe 함정 목록 | `docs/tech-validation.md` |

---

## 일정 가드레일

- 1단계는 읽기 전용(코드 수정 없음) — 언제든 안전하게 실행 가능
- 2단계 ADR은 사용자 승인 후 3단계 착수
- ③-1/③-2/③-3은 각각 골든 테스트 통과를 머지 조건으로

---

## 부록 — 세션 요청 프롬프트 (단계별)

### 1단계 요청 (감사 리뷰만 — 코드 수정 없음)

```
docs/lenssim-handoff/refactoring-audit-and-tracking-migration-plan.md 를 정독하고 1단계(감사 리뷰)만 실행해줘.

- 문서의 6개 관점대로 멀티에이전트 적대적 리뷰 워크플로를 구성하고 (ultracode),
  특히 "결합도(교체 가능성)" 관점은 TFLite 추적과 코어의 결합 지점을 전수 조사해줘.
- LensSimulator 함정 목록 13건(/Volumes/M3-P31/Projects/TedMong/CGG/LensSimulator/docs/tech-validation.md)과
  기존 코드를 대조 검사하는 것 포함.
- 코드는 수정하지 마. 산출물은 docs/lenssim-handoff/audit-report.md (findings + 모듈별 판정표 + 교체 난이도 견적).
- 끝나면 blocker/major 요약과 "리팩토링 vs 재작성" 권고를 보고해줘. 2단계는 내 승인 후 진행.
```

### 2단계 요청 (감사 결과 검토 후)

```
audit-report.md 기반으로 계획 문서의 2단계를 실행해줘:
① 추적 레이어 주입형 전환 ADR 초안 작성 (감사의 결합도 견적을 근거로, 기각 옵션 포함)
② 골든 베이스라인 구축 — 표준 테스트 영상에 대한 현재 코드의 랜드마크/렌더 출력 캡처 + 비교 스크립트
ADR은 내가 검토할 테니 3단계는 착수하지 마.
```

### 3단계 요청 (ADR 승인 후, 단계별로 나눠서)

```
계획 문서의 ③-1(경계 도입)만 실행해줘. 동작 불변 — 골든 베이스라인 통과가 완료 기준이야.
완료 보고 후 ③-2 진행 여부를 내가 정할게.
```
