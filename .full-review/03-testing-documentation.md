# Phase 3: Testing & Documentation Review

## Test Coverage Findings

### Critical (2건)

| # | 이슈 | 설명 |
|---|------|------|
| T1 | P4-W3-04 변경사항 테스트 완전 부재 | One Euro Filter, DeviceTier 판정, Half-Res 파이프라인, Temporal filtering 등 신규 코드에 대한 단위 테스트가 **0건**. 318줄 추가에 대해 테스트 커버리지 0% |
| T2 | OneEuroFilter 단위 테스트 없음 | 필터 수렴성, 리셋 동작, 파라미터 민감도, 첫 프레임 패스스루 동작 검증 부재. 향후 파라미터 튜닝 시 회귀 감지 불가 |

### High (4건)

| # | 이슈 | 설명 |
|---|------|------|
| T3 | detectDeviceTier() 테스트 불가 구조 | GL 컨텍스트 의존으로 순수 단위 테스트 불가. 파싱 로직을 `static classifyGpuRenderer(const std::string&)` 정적 함수로 분리하여 테스트 가능성 확보 필요 |
| T4 | DeviceTier 분기 통합 테스트 부재 | HIGH/MID/LOW 각 tier에서 올바른 파이프라인(FreqSep full/half/Bilateral)이 선택되는지 검증 없음 |
| T5 | std::strtol 에지 케이스 테스트 부재 | 빈 문자열, 범위 초과, 비숫자 입력, "Adreno (TM) 999" 등 비정상 GPU 문자열 처리 검증 필요 |
| T6 | release() 필터 리셋 테스트 부재 | release() 호출 후 재초기화 시 필터 상태가 올바르게 리셋되는지 검증 없음 |

### Medium (3건)

| # | 이슈 | 설명 |
|---|------|------|
| T7 | Half-Res 파이프라인 경계 테스트 부재 | 홀수 해상도(1920x1081), 극소 해상도(32x32), half 계산 시 minimum guard 동작 확인 없음 |
| T8 | Mali 분류 경계값 테스트 부재 | G77→MID, G78→HIGH 등 경계 GPU 분류 정확성 미검증 |
| T9 | Temporal filtering 안정성 테스트 부재 | blur_radius 급변 시 필터 응답, 연속 프레임 입력 시 수렴 속도 검증 없음 |

### 권장 테스트 코드 예시

```cpp
// OneEuroFilter 단위 테스트
TEST(OneEuroFilterTest, FirstSamplePassthrough) {
    OneEuroFilter filter(1.0f, 0.007f);
    EXPECT_FLOAT_EQ(filter.filter(5.0f), 5.0f);
}

TEST(OneEuroFilterTest, ConvergesToStableInput) {
    OneEuroFilter filter(1.0f, 0.007f);
    float target = 10.0f;
    float val = 0.0f;
    for (int i = 0; i < 100; ++i) val = filter.filter(target);
    EXPECT_NEAR(val, target, 0.01f);
}

TEST(OneEuroFilterTest, ResetClearsState) {
    OneEuroFilter filter(1.0f, 0.007f);
    filter.filter(5.0f);
    filter.filter(10.0f);
    filter.reset();
    EXPECT_FLOAT_EQ(filter.filter(20.0f), 20.0f);
}

// DeviceTier 분류 테스트 (classifyGpuRenderer 추출 시)
TEST(DeviceTierTest, Adreno7xxIsHigh) {
    EXPECT_EQ(classifyGpuRenderer("Adreno (TM) 730"), DeviceTier::HIGH);
}
TEST(DeviceTierTest, Adreno6xxIsMid) {
    EXPECT_EQ(classifyGpuRenderer("Adreno (TM) 640"), DeviceTier::MID);
}
TEST(DeviceTierTest, UnknownGpuIsLow) {
    EXPECT_EQ(classifyGpuRenderer("Unknown GPU"), DeviceTier::LOW);
}
TEST(DeviceTierTest, EmptyStringIsLow) {
    EXPECT_EQ(classifyGpuRenderer(""), DeviceTier::LOW);
}
```

### 리팩터링 권장: 테스트 가능 구조

```cpp
// gpu_beauty_backend.h에 추가
static DeviceTier classifyGpuRenderer(const std::string& renderer_str);

// detectDeviceTier()는 이를 호출
DeviceTier detectDeviceTier() {
    const char* renderer = (const char*)glGetString(GL_RENDERER);
    return classifyGpuRenderer(renderer ? renderer : "");
}
```

---

## Documentation Findings

### Critical (1건)

| # | 이슈 | 설명 |
|---|------|------|
| D1 | 작업 문서 §3.2 mask center 구현 불일치 | `P4-W3-04_temporal_stability_device_tier.md` §3.2에서 mask center smoothing을 "UV 좌표 오프셋"으로 기술하지만, 실제 코드는 `face_rect.x/y`를 직접 수정. 문서와 구현이 불일치 |

### High (3건)

| # | 이슈 | 설명 |
|---|------|------|
| D2 | DeviceTier enum 파이프라인 동작 미문서화 | enum 값별(HIGH/MID/LOW) 실제 파이프라인 분기 동작(FreqSep full-res / half-res / Bilateral)이 코드 내 문서화 안됨 |
| D3 | GPUBeautyBackend 클래스 Doxygen 미갱신 | FreqSep 파이프라인, DeviceTier 분기, temporal filtering 등 주요 기능이 클래스 레벨 문서에 반영 안됨 |
| D4 | One Euro Filter 파라미터 선택 근거 부재 | `min_cutoff=0.5, beta=0.01` (blur_radius), `min_cutoff=1.0, beta=0.02` (mask center) 값의 선택 이유/실험 결과 미기록 |

### Medium (4건)

| # | 이슈 | 설명 |
|---|------|------|
| D5 | Mali 분류 비대칭 근거 미기록 | Adreno는 x00+→HIGH이지만 Mali는 G-series 넘버링 기준이 다름. 비대칭 분류 기준의 하드웨어적 근거 주석 필요 |
| D6 | detectDeviceTier() Doxygen 불완전 | GL 컨텍스트 활성 전제조건, 분류 기준표, 반환값 의미 등 문서화 부족 |
| D7 | Half-Res 리소스 정보 미기록 | 추가 텍스처 2장(half-res pair), 추가 GPU 메모리 ~6.22MB 등 리소스 요구사항 문서 부재 |
| D8 | DeviceTier 임계값 근거 미기록 | Adreno 700+→HIGH, 600+→MID 등의 기준이 벤치마크 기반인지 경험적 추정인지 명시 안됨 |

### Low (4건)

| # | 이슈 | 설명 |
|---|------|------|
| D9 | executeFreqSepPipelineHalfRes() 함수 주석 | 5-subpass 구조, 해상도 전환 시점 등 상세 주석 부족 |
| D10 | CHANGELOG 미갱신 | P4-W3-04 변경사항이 CHANGELOG에 미반영 |
| D11 | applyTextureId() tier 분기 인라인 주석 | 각 tier 분기 진입 조건과 의도 설명 주석 부족 |
| D12 | BUILD_GUIDE.md DeviceTier 디버깅 | LOGD로 출력되는 DeviceTier 값 확인 방법 미기록 |
