# P6-W5 Phase A/B 코드 리뷰 - Codex

- 검토 범위: `git diff e51a4e6..abc9a84` (`feature/P6-Works` HEAD)
- 기준 문서: `docs/workPaper/P6-W5_blend_sclera_bench.md` §5.1~§5.14, `docs/workPaper/P6-W5_brainstorm/synthesis.md`
- 결과: High 2건, Medium 1건

## Findings

### [High] F-01. Native lens fallback 시 A/B/C/D가 B1/B8 후보 구현을 비교하지 않는다

**근거**

- `android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLView.kt:292-295`는 `setScleraVetoMode()`를 native SDK에만 전달한다.
- `android/demo-app/src/main/java/com/irislenssdk/demo/camera/gpu/CameraGLRenderer.kt:1228-1259`는 native GPU lens 초기화 실패 또는 렌더 실패 시 `renderLensOverlay()` Kotlin 셰이더로 fallback한다.
- fallback 셰이더는 `CameraGLRenderer.kt:263-270`에서 기존 sRGB `blendColorReplace()`를 사용하고, `CameraGLRenderer.kt:287-298,371-403`에서 legacy `calcScleraFactor()`만 사용한다. `uScleraVetoMode` 분기와 `blendColorReplaceLinear()`가 없다.

**영향**

- fallback 상태에서 B8은 A/B 및 C/D 간 veto 차이가 없어져 color-veto 대 luma-only 판정 자료가 성립하지 않는다.
- fallback 상태에서 B1의 C/D도 명세의 `ColorReplaceLinear`가 아닌 구형 Color Replace를 렌더하므로 Normal 대 CRL 판정 자료가 성립하지 않는다.
- 현재 촬영 가이드는 native lens 활성 여부를 유효성 조건으로 기록하지 않으므로, 실패 프레임을 정상 32클립으로 수집할 수 있다.

**판정**

- §5.4 및 §5.9의 4조합 런타임 비교를 실제 출력에서 보장하지 못한다. 촬영에 사용되는 렌더 경로가 후보 수식을 렌더하는 상태임을 보장하거나, 그렇지 않은 take는 유효 데이터로 취급하지 않아야 한다.

### [High] F-02. B1 CRL 상한이 확정값 `1.25`가 아니라 `1.2`로 고정되어 있다

**근거**

- §5.10 및 `synthesis.md` §1은 W5 1차에서 `detail` clamp `[0.75, 1.25]`, `uMaxDetail = 1.25` 고정을 확정했다.
- 활성 native 경로는 `cpp/src/gpu/gpu_lens_renderer.cpp:955-956`에서 `uMaxDetail`을 매 프레임 `1.2f`로 주입한다.
- 새 벤치 문서도 `docs/bench/P6-W5/checklist.md:17` 및 `docs/bench/P6-W5/recording_guide.md:17-20`에서 `1.2`를 W5 고정값으로 지시한다.

**영향**

- C/D의 CRL 결과가 확정된 비교 조건보다 낮은 detail 상한으로 촬영된다.
- §5.10이 금지한 clamp 변수 변경이 벤치 프로토콜에 포함되어, B1 판정 결과를 확정 사양의 결과로 해석할 수 없다.

**판정**

- 촬영 전에 native uniform 값과 벤치 설정 기준을 모두 `1.25` 조건과 일치시켜야 한다.

### [Medium] F-03. 8-take/32클립 표는 확정된 B1/B8 매트릭스를 완성하지 못한다

**근거**

- §5.1은 B1에 대해 5 SKU와 홍채 톤 3종(짙음, 중간, 밝음)을 명시한다.
- §5.3은 B8에 대해 2 SKU(밝은 그레이/블루, 다크브라운)와 조명 3종(형광, 측광, 저조도)의 교차를 명시한다.
- `docs/bench/P6-W5/checklist.md:52-70`의 B1 표본은 T1~T5의 SKU별 형광 take만 있고 홍채 톤 축이 없다.
- 같은 표의 B8 표본은 밝은 그레이의 E1/E2/E3와 다크브라운의 E1/E3만 포함하며, `다크브라운 x 측광(E2)` 셀이 없다.
- `scripts/p6w5_bench_helper.sh:28-44,115-125`도 이 8개 take만 처리하도록 고정되어 있다.

**영향**

- 저조도 채도 위험 영역 자체는 밝은 그레이 및 다크브라운 E3로 포함되어 있다.
- 그러나 B1의 홍채 톤별 판정과 B8의 전체 2 x 3 조명 비교는 현재 32클립 자료로 수행할 수 없다. 특히 측광에서 위험 SKU와 대조군을 함께 비교할 수 없다.

**판정**

- 현재 매트릭스는 저조도 확인용 부분 표본으로는 유효하지만, §5.1 및 §5.3 완료 판정 자료로는 불충분하다.

## 확인 결과

| 리뷰 포인트 | 결과 |
|---|---|
| 3-way 셰이더 수식 | `cpp/src/gpu/shader_sources.cpp:1007-1019`의 mode 1은 §5.4 color-veto와 동일하고, mode 2는 sat 항만 제거하며 §5.12의 `0.45/0.65` 및 §5.14의 `0.6`을 유지한다. |
| Legacy 동작 | mode 0의 `shader_sources.cpp:1020-1024`는 기존 `geomFactor`를 `geom`으로 이름만 바꾼 등가식이다. legacy 출력 변경은 발견하지 못했다. |
| 색공간 일관성 | 새 `sat`/`lum`은 texture sample인 `camera.rgb`에서 계산되며, legacy `calcScleraFactor()`와 동일한 sRGB camera 값 및 `vec3(0.299, 0.587, 0.114)` 밝기 계수를 사용한다. |
| `calcScleraFactor` 정책 | §4.1과 §8.2가 단일 수식 교체를 Phase C 작업으로 명시하므로, Phase A/B에서 legacy 함수가 mode 0용으로 남은 것은 별도 finding이 아니다. |
| setter/thread-safety/default | `sclera_veto_mode_` 기본값은 0이고, `GPULensRenderer::setScleraVetoMode()` 및 `renderToTexture()`는 동일 mutex를 사용한다. C API도 `g_gpu_mutex` 아래 전달되며, demo 호출은 GL queue를 경유한다. |
| 공개 C API | `cpp/include/iris_sdk/sdk_api.h`에는 veto setter 추가 diff가 없다. internal C 함수 추가가 공개 C header surface를 변경하지는 않는다. |
| A/B/C/D 및 spinner | `GpuRenderActivity.kt:379-398,460-478`의 A/B/C/D 매핑은 §5.9와 일치하며, 현 entries 배열에서는 blend ID `0`과 `7`이 각각 spinner index `0`과 `7`이다. |
| 독립 판정 | `docs/bench/P6-W5/ratings_legend.md:30-45`는 B1과 B8 지표를 분리 집계한다. 단, F-01이 해소되고 촬영 전제인 `Sclera Protect ON`이 유지되는 경우에만 실제 출력도 이 독립 설계를 따른다. |

## 검증

- `bash -n scripts/p6w5_bench_helper.sh`: 통과.
- `./gradlew clean build` (`android/`): 변경된 Kotlin/JNI/C++ 컴파일 및 `:demo-app:assembleDebug`, `:iris-sdk:build`까지 통과했다. 전체 명령은 `:demo-app:minifyReleaseWithR8`에서 `javax.annotation.processing.*` 등 누락 클래스 오류로 실패했으며, 이 실패가 본 diff에서 유발되었다는 근거는 확인하지 못했다.

