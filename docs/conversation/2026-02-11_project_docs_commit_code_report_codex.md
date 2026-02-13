# IrisLensSDK 프로젝트 분석 보고서 (Codex)

- 작성일: 2026-02-11
- 분석 브랜치: `feature/P3-beauty-enhancement`
- 분석 대상: `docs/**`, Git 커밋 이력, `cpp/**`, `android/**`

## 1. `docs` 문서 분석

### 1.1 전체 규모와 구성
- `docs` 전체 파일 수: 963
- `docs/mediapipe_sample`: 845 (레퍼런스/샘플 비중이 큼)
- `docs/workPaper`: 60
- `docs/conversation`: 11
- `docs/reports`: 8

### 1.2 핵심 문서와 흐름
실제 프로젝트 진행 맥락은 아래 문서군에 집중되어 있음.
- `docs/PROJECT_SPEC.md`
- `docs/DEVELOPMENT_ROADMAP.md`
- `docs/ARCHITECTURE.md`
- `docs/ARCHITECTURE_COMPARISON.md`
- `docs/PIPELINE_ANALYSIS.md`
- `docs/DECISION_RECORD.md`
- `docs/PERFORMANCE_REPORT.md`
- `docs/workPaper/*`
- `docs/conversation/*`

문서 상 개발 흐름은 Phase1 -> P2 -> P3로 정리되며, 반복되는 핵심 이슈는 다음과 같음.
- 좌표계/종횡비 정합 이슈 (ISS-001, ISS-002)
- GPU 렌더링 안정화
- 성능 최적화 (XNNPACK, GPU delegate)
- Beauty 필터 파이프라인 확장

### 1.3 문서 기준 상태 요약
- 문서화 수준은 높고 의사결정 기록이 잘 축적되어 있음.
- 다만 `docs/PERFORMANCE_REPORT.md` 기준 CPU-only 성능은 목표 FPS/메모리 타깃 대비 미달 흔적이 존재함.

## 2. 커밋 이력 분석

### 2.1 기본 통계
- 총 커밋 수: 100
- 기간: 2026-01-07 ~ 2026-02-11
- 주요 작성자: `MerooMong` (단일 중심)

### 2.2 커밋 성격 분류 (메시지 prefix 기준 휴리스틱)
- `feat`: 49
- `fix`: 11
- `docs`: 11
- `perf`: 3
- `refactor`: 3
- `test`: 3
- `chore`: 3
- 기타: 17

### 2.3 변경 집중 구간과 핫스팟
- 개발 피크: 1월 중순~말, 2월 11일 GPU/LUT 관련 마무리 커밋 집중
- 변경량 상위 디렉터리: `docs`, `cpp`, `android`
- 변경량 상위 파일:
  - `cpp/src/mediapipe_detector.cpp`
  - `android/iris-sdk/src/main/cpp/iris_jni.cpp`
  - `android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLRenderer.kt`
  - `cpp/src/gpu/gpu_beauty_backend.cpp`

## 3. 코드 구현 분석

### 3.1 구현된 핵심 축
- C API 계층:
  - `cpp/include/iris_sdk/sdk_api.h`
  - `cpp/src/sdk_api.cpp`
  - `cpp/src/sdk_api_v2.cpp`
- 처리 파이프라인:
  - `cpp/src/frame_processor.cpp`
  - `cpp/src/mediapipe_detector.cpp`
- Android 연동:
  - `android/iris-sdk/src/main/cpp/iris_jni.cpp`
  - `android/iris-sdk/src/main/java/com/irislenssdk/IrisLensSDK.java`
- Demo GPU 렌더러:
  - `android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLRenderer.kt`

### 3.2 부분 구현/미완 경로
- `cpp/src/gpu/gpu_beauty_backend.cpp`의 CPU 버퍼 경로 `apply()`는 실질적으로 stub 상태
- 동일 파일의 `applyFaceWarp()`도 stub 성격
- ROI 마스킹 적용 TODO 잔존
- 텍스처 풀 ownership 경계가 완전히 정리되지 않아 이중 해제 리스크 존재
- Java API에서 detection handle 연계가 제한적(0 전달)이라 ROI/face-aware 품질 저하 가능성 존재

### 3.3 유지보수성 관점
- 대형 파일 집중이 큼:
  - `cpp/src/mediapipe_detector.cpp` (약 3037 lines)
  - `android/iris-sdk/src/main/cpp/iris_jni.cpp` (약 1667 lines)
  - `android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLRenderer.kt` (약 1255 lines)
- 기능 확장 속도는 높지만, 파일 복잡도 증가로 리그레션 리스크가 커짐.

## 4. 종합 진단
- 프로젝트는 문서화/기능 확장 모두 빠르게 진행됨.
- 현재 핵심 리스크는 신규 기능 부족보다 GPU 리소스 수명주기, 일부 미완성(stub) 경로, API 계층 간 데이터 연동 일관성에 있음.
- 즉, "기능 추가"보다 "안정화/완성도 제고"의 우선순위가 높음.

## 5. 우선순위 권고안
1. `gpu_beauty_backend`의 stub 경로(`apply`, `applyFaceWarp`)를 구현 완료하거나 명시적 비활성화
2. 텍스처 풀 ownership 정책을 C API/JNI/Renderer 전 계층에서 단일 규약으로 고정
3. Java <-> Native detection handle 전달 체계 보강으로 ROI/face-aware 처리 일관성 확보
4. 대형 파일 분리(탐지/좌표 변환/렌더/디버그 로깅)로 결합도 완화
5. 모델/리소스 의존 테스트를 CI 친화 형태로 정리해 회귀 검증 자동화

## 6. 분석 한계
- 이번 분석은 정적 코드/문서/커밋 기반이며, 테스트 실행 결과는 포함하지 않음.

---

## ClaudeComment

> 작성: Claude Opus 4.6 (2026-02-11)
> 기반: `feature/P3-beauty-enhancement` 브랜치에서의 LUT 통합 작업 경험 (`54b6f50`)

### 전반적 평가

보고서의 전체적인 진단은 정확하다. "기능 추가보다 안정화/완성도 제고"라는 핵심 메시지에 동의하며, 각 항목에 대해 구체적으로 코멘트한다.

### Section 1 — 문서 분석

`docs/mediapipe_sample`이 845개로 전체의 88%를 차지한다는 지적이 유의미하다. 레퍼런스 코드는 별도 서브모듈이나 `.gitignore` 처리를 고려할 만하다. 실질적인 프로젝트 문서(`workPaper`, `conversation`, `reports`)는 약 80개로 적절한 규모이며, 의사결정 추적이 잘 되어 있다는 평가에 동의한다.

### Section 2 — 커밋 이력

`feat: 49`로 기능 커밋이 전체의 약 50%인 점은 빠른 기능 확장 속도를 반영한다. 다만 `test: 3`은 기능 대비 테스트 비율이 낮다. P3 안정화 단계에서 테스트 커밋 비중을 의도적으로 올리는 것을 권장한다.

### Section 3.2 — 부분 구현/미완 경로 (업데이트 필요)

LUT 통합 작업(`54b6f50`)으로 일부 상태가 변경되었다:

| 보고서 지적 | 현재 상태 | 비고 |
|-------------|-----------|------|
| `apply()` CPU 버퍼 경로가 stub | **여전히 유효** | GPU 텍스처 경로만 활성화, CPU fallback 미구현 |
| `applyFaceWarp()` stub | **여전히 유효** | GridMesh LOD는 구현됐으나 GPU 워프 패스 자체는 프레임워크 수준 |
| 텍스처 풀 ownership 이중 해제 리스크 | **부분 개선** | `CameraGLRenderer`에서 `releaseTexture()` 호출 제거로 이중 해제 방지. 단, C++ TexturePool과 Android GL 텍스처 간 소유권 규약이 명문화되지 않음 |
| Java API detection handle 연계 제한(0 전달) | **여전히 유효** | `applyBeautyFilterTextureV2()`에서 detection handle을 0으로 전달하여 ROI 미적용 |

### Section 3.3 — 대형 파일

동의하되 우선순위 조정을 제안한다.

- `mediapipe_detector.cpp` (3037줄): 검출기 로직이 단일 파일에 집중. 좌표 변환 / 모델 로딩 / 검출 로직 분리가 유의미
- `iris_jni.cpp` (1667줄): V1/V2 API가 혼재. V2 전용 JNI를 별도 파일로 분리하면 유지보수성 향상
- `CameraGLRenderer.kt` (현재 ~1255줄): LUT 패스 제거로 약 150줄 감소. 여전히 크지만 렌더링 단계별 분리(OES변환/뷰티/렌즈/화면출력)는 구조적으로 명확

### Section 4-5 — 종합 진단 및 권고안 (우선순위 재배열 제안)

보고서의 5개 권고안에 대해, 현재 시점의 실질적 임팩트 기준으로 재배열한다:

1. **(높음) 텍스처 소유권 규약 명문화** — 현재 가장 큰 런타임 리스크. C++ TexturePool이 생성/소유하는 텍스처를 Android가 삭제하면 검은 화면 발생 (실제 발생 이력 있음)
2. **(높음) Detection handle 전달 체계 보강** — ROI 미적용으로 뷰티 필터가 전체 프레임에 적용되어 불필요한 GPU 부하 + 비얼굴 영역 변색
3. **(중간) stub 경로 정리** — `apply()` CPU 경로는 GPU 미지원 기기 대비 필요. `applyFaceWarp()`은 P3 스코프 내 완성 필요 여부 판단 필요
4. **(중간) CI 테스트 자동화** — 모델 파일 의존 테스트가 CI에서 skip되어 회귀 감지 불가
5. **(낮음) 대형 파일 분리** — 구조적 개선이나 기능/안정성 임팩트는 낮음. 다음 리팩토링 사이클에서 진행

### Section 6 — 분석 한계에 추가할 사항

- Android 빌드(Gradle) 결과를 포함하지 않아, JNI 바인딩 정합성은 실기기 테스트에서만 확인 가능
- GPU 셰이더 컴파일은 런타임에 수행되므로 정적 분석으로 검증 불가 (Mali/Adreno/PowerVR 간 호환성 차이)

## GeminiComment

> 작성: Gemini 3 Pro (2026-02-11)
> 기반: `feature/P3-beauty-enhancement` 브랜치 코드 및 `docs` 분석

### 전반적 평가 및 동의

**Codex의 분석 보고서와 ClaudeComment의 진단에 전적으로 동의한다.**
특히 **"기능 구현(Make it work) 단계에서 제품화(Make it right) 단계로의 전환"**이 시급하다는 판단은 현재 프로젝트 상태를 정확히 관통하고 있다. 화려한 기능 추가보다 **안정성 확보**가 선행되어야만 상용 수준의 SDK 품질을 보장할 수 있다.

### 1. 우선순위 재배열에 대한 추가 의견

ClaudeComment가 제안한 우선순위 재배열(텍스처 소유권 > ROI 연동 > Stub 정리)은 엔지니어링 관점에서 매우 타당하며, 이에 대한 구체적인 근거를 보강한다.

#### 1.1 텍스처 소유권 규약 명문화 (Critical Priority)
- **현황**: `CameraGLRenderer.kt` 주석(`// NOTE: 텍스처 해제는 C++ TexturePool에서 관리함`)에 의존하는 '신사협정' 상태다.
- **리스크**: Java 레벨에서의 실수(ex: `finalize()`나 `release()` 호출)나 GC 동작 시, Native 힙 메모리 침범(Double-free)으로 인한 **즉시 크래시** 위험이 상존한다.
- **제언**: 단순 문서화를 넘어, `TextureHandle` 객체를 래핑하여 소유권을 명시적으로 관리하거나, `SharedPtr` 패턴을 JNI 레벨까지 확장하는 구조적 안전장치 도입을 검토해야 한다.

#### 1.2 Detection Handle / ROI 연동 (Performance & Quality)
- **현황**: `applyBeautyFilterTextureV2` 호출 시 Detection Handle이 `0`(null)으로 전달되고 있어, 얼굴 영역(ROI) 마스킹 없이 전체 화면에 필터가 적용된다.
- **문제점**:
    1. **품질 저하**: 배경까지 하얗게 되거나 뭉개지는 부자연스러운 결과 초래.
    2. **성능 낭비**: 얼굴 영역만 처리하면 GPU 부하를 획기적으로 줄일 수 있음에도, 전체 해상도를 처리하고 있어 배터리/발열 이슈 우려.
- **제언**: Super Color Shader 통합 후 최우선 과제로 진행해야 한다.

### 2. 구조적 문제 진단에 대한 의견

- **대형 파일 문제**: `mediapipe_detector.cpp`(3000줄+)와 `iris_jni.cpp`(1600줄+)는 유지보수의 병목이 될 가능성이 높다. 특히 JNI 레이어에 V1(Legacy)과 V2(New) 로직이 혼재된 상태는 향후 리팩토링 비용을 급격히 증가시킬 것이다. V2 전용 JNI 파일 분리 전략에 동의한다.
- **테스트 부족**: 그래픽스/JNI 코드 특성상 단위 테스트 작성이 어렵지만, 로직 분리(순수 C++ 알고리즘 vs 안드로이드 의존성)를 통해 테스트 커버리지를 높여야 한다.

### 3. 향후 로드맵 제언

현재 수립된 `crystalline-dreaming-bonbon.md` 계획(Super Color Shader 구현)은 **셰이더 효율을 높이고 코드 복잡도를 낮추는 작업**이므로, 위에서 지적한 '안정화' 방향성과 일치한다.

따라서 다음과 같은 순서로 진행하는 것을 강력히 권장한다:
1.  **Phase 1 (Super Color Shader)**: 셰이더 통합으로 성능 최적화 및 코드 간소화
2.  **Stability Fix**: 텍스처 소유권 명확화 및 ROI 연동 구현
3.  **Feature Expansion**: LUT 에셋 추가 및 Stub 경로 구현

**결론**: 현재 방향성은 매우 훌륭하며, 제안된 안정화 작업을 통해 SDK의 완성도를 높일 수 있다.

## CodexComment

> 작성: Codex GPT-5 (2026-02-11)
> 기반: `docs` 분석 + 커밋 이력 + `cpp/android` 코드 정적 검토

### 코드 경로 기반 검증

뷰티 텍스처 V2 호출 체인은 다음과 같이 확인된다.
- `android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLRenderer.kt:782`
- `android/iris-sdk/src/main/java/com/irislenssdk/IrisLensSDK.java:746`
- `android/iris-sdk/src/main/cpp/iris_jni.cpp:1532`
- `cpp/src/sdk_api_v2.cpp:286`
- `cpp/src/gpu/gpu_beauty_backend.cpp:834`

### 1. ROI 미적용 경로 (핵심)

Java wrapper가 detection 포인터를 고정값 `0L`로 전달한다.
- `android/iris-sdk/src/main/java/com/irislenssdk/IrisLensSDK.java:729`
- `android/iris-sdk/src/main/java/com/irislenssdk/IrisLensSDK.java:752`

Native에서는 `detectionPtr`를 `IrisResult*`로 변환해 C API로 넘긴다.
- `android/iris-sdk/src/main/cpp/iris_jni.cpp:1561`
- `android/iris-sdk/src/main/cpp/iris_jni.cpp:1565`

ROI 생성은 아래 조건에서만 활성화된다.
- `cpp/src/sdk_api_v2.cpp:321` (`detection && detection->detected && config->roi_only`)

현재 경로에서는 detection이 null이므로 ROI 계산이 사실상 비활성화되고, GPU backend 내부도 ROI 마스킹 TODO가 남아 있다.
- `cpp/src/gpu/gpu_beauty_backend.cpp:1032`

### 2. 텍스처 소유권/해제 경로

SDK는 뷰티 출력 텍스처를 관리 집합에 등록한다.
- `cpp/src/sdk_api_v2.cpp:371` (`g_managed_textures.insert`)

해제 API는 관리 집합에 있는 텍스처만 허용한다.
- `cpp/src/sdk_api_v2.cpp:449`
- `cpp/src/sdk_api_v2.cpp:457`

문제는 backend release 구현이 풀 관리 여부를 엄밀히 구분하지 않고 `glDeleteTextures`를 직접 호출하는 지점이 남아 있다는 점이다.
- `cpp/src/gpu/gpu_beauty_backend.cpp:1119`
- `cpp/src/gpu/gpu_beauty_backend.cpp:1120`

Android 쪽은 프레임 루프에서 즉시 해제를 제거했지만, 종료 시점 해제 호출은 유지되어 있다.
- 제거 근거 주석: `android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLRenderer.kt:795`
- 종료 해제 호출: `android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLRenderer.kt:1203`

### 3. Stub/부분 구현 상태

CPU 버퍼 경로는 경고 로그와 함께 실질 stub 상태다.
- `cpp/src/gpu/gpu_beauty_backend.cpp:364`
- `cpp/src/gpu/gpu_beauty_backend.cpp:398`

Face Warp GPU 경로도 TODO 상태로 pass-through 반환이다.
- `cpp/src/gpu/gpu_beauty_backend.cpp:1043`
- `cpp/src/gpu/gpu_beauty_backend.cpp:1079`

### 4. Super Color Shader 용어 정리

`Super Color Shader`는 코드 내 정식 클래스/파일명이 아니라, 리뷰 문서에서 제안한 명칭이다.
- 출처: `docs/conversation/feedback_P3_gpu_rendering_review.md:80`

코드상 대응 구현은 사실상 `Combined Color + LUT` 통합 패스로 존재한다.
- 프로그램 생성: `cpp/src/gpu/gpu_beauty_backend.cpp:188`
- 실행 경로: `cpp/src/gpu/gpu_beauty_backend.cpp:710`
- LUT uniform 처리: `cpp/src/gpu/gpu_beauty_backend.cpp:753`
- 셰이더 본문(밝기/밸런스/화이트닝 + LUT): `cpp/src/gpu/shader_sources.cpp:311`

즉, 문서의 "Super Color Shader"는 현재 코드에서는 `COMBINED_COLOR_ADJUSTMENT_FRAGMENT`와 `executeCombinedColorPass`로 구현되어 있는 개념에 해당한다.

### 실행 우선순위 (기술 기준)

1. Java API에 detection-handle 전달 overload 추가 및 호출부 연결
2. ROI 적용 전후 출력 비교(품질) + GPU 시간(성능) 계측 테스트 추가
3. `releaseTexture`에서 texture-pool 소유 텍스처와 외부 텍스처 분기 로직 명문화
4. `apply`/`applyFaceWarp`를 명시적 미지원 에러 또는 완전 구현으로 상태 확정
