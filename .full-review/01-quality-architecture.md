# Phase 1: Code Quality & Architecture Review

## Code Quality Findings

### [Medium] CQ-1: 셰이더 내 `diff`와 `high` 변수 중복 계산
- **파일**: `shader_sources.cpp` line 495 vs 512
- `vec3 high = orig - low;`과 `vec3 diff = orig - low;`가 동일한 연산
- GPU 컴파일러 CSE로 최적화 가능하나, 가독성 측면에서 혼란
- **권장**: `diff` 제거 → `high` 직접 사용

### [Medium] CQ-2: LUMA_709 상수와 인라인 리터럴 혼재
- **파일**: `shader_sources.cpp` line 483 vs 534
- `LUMA_709` 상수 도입했으나 Soft Light gain 보상 코드에서 인라인 `vec3(0.2126, 0.7152, 0.0722)` 사용
- DRY 원칙 위반, 향후 계수 변경 시 동기화 누락 위험
- **권장**: `float baseLum = dot(smoothLow, LUMA_709);`

### [Medium] CQ-3: 매직 넘버 스케일링 계수
- **파일**: `shader_sources.cpp` line 521-522
- `edgeStrength * 5.0`, `chromaDev * 10.0` 하드코딩
- 정규화 범위의 의미가 코드에서 드러나지 않음
- **권장**: `const float EDGE_SCALE = 5.0; const float CHROMA_SCALE = 10.0;`으로 명명

### [Low] CQ-4: 테스트 범위 검증 정밀도
- **파일**: `test_beauty_config_v2.cpp` line 376-400
- 실제 범위 [0.3, 0.7]에 대해 [0.0, 1.0]으로 검증 → 범위 2배 이상 넓음
- **권장**: 기대 범위를 실제 매핑에 가깝게 조정

---

## Architecture Findings

### [Medium] AR-1: FreqSepParams 기본값과 mapSkinQuality 범위 불일치
- `chroma_weight` 기본값 0.3f는 매핑 범위 [0.2, 0.5]의 하한 근처
- `enabled = false` 기본이므로 실질적 영향 없으나, 직접 구성 경로에서 의미 불명확
- **권장**: 기본값을 범위 중간값(0.35f)으로 조정 또는 주석 명시

### [Low] AR-2: 매직 넘버 정규화 스케일 팩터
- `5.0`, `10.0`이 해상도/색공간 특성에 민감할 수 있음
- 현 단계에서는 하드코딩 적절, 셰이더 상수 분리가 첫 단계
- **권장**: 범위 의미 주석 추가 (`// maps typical range [0, ~0.2] to [0, 1]`)

### [Low] AR-3: sqrt 최적화 후보
- `edgeStrength = sqrt(gx*gx + gy*gy)` → 제곱 비교로 대체 가능
- 비선형 응답 곡선 변경 → 시각적 결과 달라질 수 있음
- **권장**: 프로파일링 시 최적화 후보로만 기록

### [Low] AR-4: 경계값 테스트 부재
- `skinQuality = 1.0f`에서 `edge_weight == 0.7f` 정밀 검증 없음
- 기존 테스트도 이 수준의 정밀도를 요구하지 않으므로 일관성 있음

---

## Positive Observations

- **컴포넌트 경계**: 기존 레이어 구조(shader/params/backend) 완벽 준수
- **Uniform 초기화 패턴**: 3단계 흐름(헤더→cache→execute) 기존 패턴과 1:1 일치
- **하위 호환성**: in-class initializer로 기존 코드 경로 안전
- **clamp 방어**: edge/chroma 신호 0-1 범위 보장
- **패스 추가 없음**: ALU + 4 fetch만으로 30fps 목표 안전

---

## Critical Issues for Phase 2 Context

- **Performance**: 추가 텍스처 샘플링 4회 + pow() 절약(Option B) → 성능 리뷰에서 정밀 평가 필요
- **sRGB/Linear 혼용**: 에지 검출은 sRGB, magnitude/chroma는 linear 공간 → 수치 정확성 검증
- **스케일 팩터**: `5.0`/`10.0` 하드코딩 → 디바이스/해상도별 민감도 확인
