# Phase 1: Code Quality & Architecture Review

## 리뷰 대상: P4-W4-01e Luminance Sharpen 패스 (feature/P4-W4-01e vs develop, +170 lines)

---

## Code Quality Findings

### [Critical] F-1: sRGB/Linear 색공간 불일치 — Sharpen 셰이더가 sRGB 공간에서 동작

- **파일**: `cpp/src/gpu/shader_sources.cpp:580-604`
- **설명**: Composite 셰이더(Pass 3)는 `pow(result, vec3(1.0/2.2))`로 Linear→sRGB 변환 후 출력. Sharpen 셰이더는 이 sRGB 데이터를 읽어 Rec.709 LUMA 계수(linear-light 전제)를 적용. 동일 파이프라인 내 색공간 처리 불일치.
- **영향**: sharpen_amount 0.12~0.18로 작아 시각적 차이는 미미하나, 어두운 피부톤에서 과도한 샤프닝, 밝은 피부톤에서 부족한 샤프닝 가능. 파이프라인 색공간 일관성 원칙 위반.
- **권장**: Option A(셰이더 내 gamma 보정), Option B(sRGB 근사 가중치 + 주석 명시), Option C(Composite에서 gamma 제거 후 Sharpen 후 한 번만 변환) 중 택일

### [High] F-2: `initializeFreqSepShaders()`에서 Sharpen 실패 시 불완전한 상태 처리

- **파일**: `cpp/src/gpu/gpu_beauty_backend.cpp:288-296`
- **설명**: Sharpen 컴파일 실패 시 `LOGE`만 출력하고 `return true`. `luminance_sharpen_program_`이 `createProgram` 실패 후 어떤 값을 갖는지 불명확. "created successfully" 로그가 실제 상태와 불일치.
- **권장**: 실패 시 `luminance_sharpen_program_ = 0` 명시적 초기화 + `LOGE`→`LOGW` + 로그 메시지에 sharpen 상태 반영

### [Medium] F-3: compositeRT full-res 할당 시 temp 조기 릴리스로 메모리 피크 절감 가능

- **파일**: `cpp/src/gpu/gpu_beauty_backend.cpp:1289-1294`
- **설명**: `temp`은 Pass 2b 이후 미사용이나 함수 끝에서 릴리스. compositeRT 할당 전에 temp를 먼저 릴리스하면 HIGH tier에서 풀 재활용 가능.
- **권장**: Pass 2b 완료 후 `texture_pool_->releaseTexture(temp); temp = nullptr;`

### [Medium] F-4: `uSharpenAmount` 입력 범위 검증 없음

- **파일**: `cpp/src/gpu/gpu_beauty_backend.cpp:1364`
- **설명**: `FreqSepParams`가 public struct이므로 외부에서 임의 값 설정 가능. `sharpen_amount > 1.0`이면 과도한 ringing, `< 0`이면 blur 효과.
- **권장**: `std::clamp(params.sharpen_amount, 0.0f, 0.5f)` 방어 코드

### [Medium] F-5: 테스트 tolerance 과도 — 로직 변경 감지 불가

- **파일**: `cpp/tests/test_beauty_config_v2.cpp:478`
- **설명**: `EXPECT_NEAR(0.15f, 0.02f)` — 0.13~0.17 모두 통과. 매핑 수식 변경을 감지할 수 없음.
- **권장**: tolerance를 `0.005f`로 축소

### [Low] F-6: `LUMA_709` 상수의 셰이더 간 중복 선언

- **파일**: `shader_sources.cpp:484` (Composite), `shader_sources.cpp:584` (Sharpen)
- **설명**: 향후 계수 변경 시 동기화 누락 위험. GLSL 특성상 불가피하나 C++ 측 문자열 결합으로 해결 가능.

### [Low] F-7: `SharpenAmountRange` 테스트 경계값 느슨

- **파일**: `cpp/tests/test_beauty_config_v2.cpp:500-501`
- **설명**: 실제 범위 [0.12, 0.18]인데 [0.11, 0.19]로 검증. 매핑 로직 오류를 놓칠 수 있음.
- **권장**: `EXPECT_GE(0.119f)`, `EXPECT_LE(0.181f)`

### [Low] F-8: `SharpenAmountDisabledWhenZero` 테스트가 `sharpen_amount` 값 미검증

- **파일**: `cpp/tests/test_beauty_config_v2.cpp:491-494`
- **설명**: `enabled=false`만 검증, 기본값 0.15f 유지 여부 미확인.

---

## Architecture Findings

### 긍정적 관찰

1. **기존 6단계 패턴 완벽 준수**: 셰이더 선언 → uniform 캐시 → glGetUniformLocation → glUniform → FreqSepParams → mapSkinQuality
2. **Graceful Degradation 2단계 설계**: 셰이더 컴파일 실패 → compositeRT 할당 실패 → 각각 독립 fallback
3. **별도 패스 분리 결정 적절**: 이미 복잡한 Composite에 인라인하지 않아 인지적 복잡도 관리
4. **프로파일러 통합**: `FreqSep_Sharpen` 태그로 GPU 프로파일링 가능
5. **리소스 수명 관리**: acquire/release 쌍이 `if (sharpen_enabled)` 블록 내 완결

### 구조적 우려

1. **파이프라인 패스 수 6개로 증가**: 향후 추가 패스 시 패스 병합 전략 필요
2. **`FreqSepParams` 10개 필드로 팽창**: 파라미터 그룹화(SharpenConfig 등 서브구조체) 검토 시점

---

## Critical Issues for Phase 2 Context

- **F-1 (sRGB 불일치)**: 성능에는 무관하나 시각적 품질에 영향
- **F-3 (메모리 피크)**: MID 디바이스에서 full-res RT 추가 할당 영향 검토 필요
- **F-4 (범위 검증)**: 방어적 클램핑 추가 권장
