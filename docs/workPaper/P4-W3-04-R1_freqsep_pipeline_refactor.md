# P4-W3-04-R1: FreqSep 파이프라인 중복 코드 리팩터링

| 항목 | 내용 |
|------|------|
| **작업 ID** | P4-W3-04-R1 |
| **유형** | 리팩터링 |
| **상태** | ✅ 완료 |
| **근거** | P4-W3-04 Codex 리뷰 (Q2/A2), Comprehensive Review Phase 1 |
| **선행 조건** | P4-W3-04 완료 |
| **작성일** | 2026-03-04 |
| **우선순위** | Medium (기능 결함 아님, 유지보수 개선) |

---

## 1. 배경

P4-W3-04에서 MID tier 하이브리드 해상도 파이프라인(`executeFreqSepPipelineHalfRes()`)을 구현하면서, 기존 `executeFreqSepPipeline()`과 **~140줄의 구조적 중복**이 발생했다.

두 함수는 동일한 5-subpass 구조(Gaussian H/V → LowSmooth H/V → Composite)를 따르며, 차이점은 블러 패스의 해상도와 viewport 전환 로직뿐이다.

### 리뷰 근거

| 리뷰어 | 이슈 ID | 심각도 | 내용 |
|--------|---------|--------|------|
| Code Reviewer | Q2 | High | `executeFreqSepPipelineHalfRes`와 `executeFreqSepPipeline`이 구조적으로 거의 동일 |
| Architect Reviewer | A2 | High | Temporal filtering + tier 분기로 `applyTextureId()` 200줄+ 오케스트레이션 비대화 |
| Codex | - | 타당 | 중복/오케스트레이션 비대화는 유지보수 관점에서 타당한 지적 |

---

## 2. 목표

### 2.1 완료 조건

- [x] `executeFreqSepPipeline()`과 `executeFreqSepPipelineHalfRes()`를 공통 `executeFreqSepPipelineImpl()`로 통합
- [x] 해상도 파라미터(blur 해상도, viewport 전환 여부)를 매개변수로 추출
- [x] 기존 동작(HIGH full-res, MID half-res, LOW bilateral) 유지 확인
- [ ] `applyTextureId()`의 temporal filtering 로직을 헬퍼로 추출 검토 (별도 작업으로 분리)

### 2.2 비목표 (Not in Scope)

- DeviceTier 판정 로직 변경 (런타임 적응형은 P4-W3-05)
- 셰이더 코드 변경
- 새로운 기능 추가

---

## 3. 설계

### 3.1 공통 파이프라인 함수 (구현 완료)

```cpp
/// FreqSep 파이프라인 실행 설정 (full-res / half-res 분기 매개변수화)
struct FreqSepExecConfig {
    int res_divisor;                       // 1 = full-res, 2 = half-res
    bool linear_upsample;                  // true: composite 입력에 GL_LINEAR 설정
    const char* blur_profiler_suffix;      // "" 또는 "_Half"
    const char* composite_profiler_suffix; // "" 또는 "_Full"
};

bool executeFreqSepPipelineImpl(
    GLuint input_tex, GLuint mask_tex, GLuint output_fbo,
    int width, int height,
    const FreqSepParams& params,
    const FreqSepExecConfig& exec_cfg);
```

### 3.2 호출 패턴 (구현 완료)

```cpp
// HIGH tier (full-res) — 3줄 래퍼
bool executeFreqSepPipeline(...) {
    return executeFreqSepPipelineImpl(..., {1, false, "", ""});
}

// MID tier (hybrid half-res) — 3줄 래퍼
bool executeFreqSepPipelineHalfRes(...) {
    return executeFreqSepPipelineImpl(..., {2, true, "_Half", "_Full"});
}
```

### 3.3 applyTextureId() 헬퍼 추출 (별도 작업으로 분리)

temporal filtering 추출은 별도 리팩터링 작업으로 분리.

---

## 4. 변경 대상 파일

| 파일 | 변경 내용 | 영향도 |
|------|----------|--------|
| `gpu_beauty_backend.h` | `FreqSepPipelineConfig` 구조체, `executeFreqSepPipelineImpl()` 선언 | 낮음 |
| `gpu_beauty_backend.cpp` | 공통 함수 구현, 기존 두 함수를 래퍼로 변환 또는 제거 | 높음 |

---

## 5. 검증

| 테스트 | 검증 내용 |
|--------|----------|
| 기존 FreqSep 테스트 통과 | HIGH/MID/LOW tier 전체 동작 유지 |
| 비피부 선명도 | MID tier SSIM >= 0.95 유지 |
| 성능 | 리팩터링 후 프레임 타이밍 변동 없음 |
| 코드 라인 수 | ~140줄 이상 감소 확인 |

---

## 6. 리스크

| 리스크 | 확률 | 대응 |
|--------|------|------|
| 리팩터링 과정에서 기존 동작 변경 | 낮 | 리팩터링 전/후 GPU 출력 비교 (pixel diff) |
| 공통 함수 매개변수 과다 | 낮 | FreqSepPipelineConfig 구조체로 캡슐화 |

---

## 7. 부수 개선 사항 (리뷰에서 낮은 심각도로 분류된 항목)

리팩터링 시 함께 처리할 수 있는 항목:

| 이슈 | 심각도 | 내용 | 처리 방안 |
|------|--------|------|----------|
| Q4/A5 | Low | 중복 glTexParameteri(GL_LINEAR) 호출 | TexturePool 기본값 의존 주석 추가, 불필요한 호출 제거 |
| Q7 | Low | applyTextureId 중첩 깊이 | 전략 선택 함수 추출로 자연스럽게 해소 |
| Q9 | Low | 매직 넘버 28 | `constexpr kMaxGaussianRadius = 28` 명명 |
| P5 | Low | 중복 glUseProgram | 공통 함수에서 1회 호출로 통합 |

---

## 변경 이력

| 날짜 | 변경 내용 | 작성자 |
|------|----------|--------|
| 2026-03-04 | 초기 작성 (P4-W3-04 Codex 리뷰 피드백 기반) | Claude |
| 2026-03-05 | 구현 완료: FreqSepExecConfig + executeFreqSepPipelineImpl 통합, ~130줄 순감소 | Claude |
