# P4-W3-05 최종 수정 리포트 (수정 여부 + 수정 방향)

**Date**: 2026-03-06  
**기준 문서**: `claudedocs/validated-review-report-2026-03-06.md`  
**결정 원칙**: 우선순위보다 `수정 필요 여부`와 `수정 방법의 타당성`을 기준으로 확정

---

## 1. 의사결정 원칙

1. `NO`로 확정된 항목만 이번 수정 범위에서 제외한다.
2. `YES`, `DEFER`, `P2`, `P3` 항목은 모두 수정 대상으로 포함한다.
3. 단, 구현은 묶음(패키지) 단위로 진행하여 회귀 리스크를 줄인다.

---

## 2. 최종 수정 대상/제외 대상

### 2.1 수정 제외 (NO 확정)

- N1: `addFrame()` Laplacian 비용 이슈를 P0/P1 결함으로 간주한 주장
- N2: `vector::erase(begin())` 자체를 즉시 결함으로 간주한 주장
- N3: Pimpl 미적용을 즉시 결함으로 간주한 주장

위 3건은 "영구 기각"이 아니라 "현 시점 필수 수정 제외"로 기록한다.

### 2.2 수정 대상 (NO 제외 전부)

- 수정 대상 총계: **48건**
- 근거: `P0/P1(YES+DEFER) + P2 + P3`
- 실행 시 핵심 보정 포인트:
  - Halo 게이트 정확도 2건은 반드시 포함
  - `catch(...)`는 단순 stderr 출력이 아닌 실패 상태 분리까지 포함
  - SSIM은 `CV_64F -> CV_32F` 전환 + 중간 `Mat` 생애주기 축소를 함께 수행

---

## 3. 최종 수정 방향 (패키지별)

우선순위 표기는 생략하고, 구현 패키지 기준으로 명세한다.

### Package A: 게이트 정확도/신뢰성 보정

**대상 파일**
- `cpp/src/quality_metrics.cpp`
- `cpp/src/ab_compare.cpp`
- `cpp/src/param_tuner.cpp`

**필수 수정 항목**
1. `detectHalo` zero-baseline 처리 추가
   - 위치: `quality_metrics.cpp:328-335`
   - 기존: `original_boundary_gradient <= 1e-12`면 ratio=0.0 유지
   - 수정: zero-baseline 분기에서 `processed_boundary_gradient` 절대 임계치 기반 fail 가능하도록 처리

2. `halo_improvement` 비양수 baseline 왜곡 제거
   - 위치: `ab_compare.cpp:141-143`
   - 기존: `b_halo <= 1e-9`이면 무조건 0.0
   - 수정: 비양수 baseline에서도 signed delta가 보존되도록 식 변경

3. `catch(...)` 정책 보강
   - 위치: `quality_metrics.cpp`, `ab_compare.cpp`, `param_tuner.cpp`
   - 기존: 무음 기본값 반환
   - 수정: 최소 `cv::Exception`, `std::exception` 분기 + 실패 원인 추적 가능한 상태값/로그 경로 제공

4. `computeScore()`의 "실패=0점 절벽" 완화
   - 위치: `param_tuner.cpp:108-133`
   - 수정: gate fail이어도 연속적인 페널티 스코어를 부여해 탐색 신호 손실 방지

**완료 기준**
- Halo 관련 2개 재현 케이스에서 기존 오판정이 사라질 것
- 계산 실패와 품질 불량이 결과 구조에서 구분될 것

### Package B: 안전 가드레일 (OOM/Hang 방지)

**대상 파일**
- `cpp/src/ab_compare.cpp`
- `cpp/src/param_tuner.cpp`
- 필요 시 입력 경로(호출부)

**필수 수정 항목**
1. `ABCompare::addResult()` 저장 개수 상한 도입
   - 위치: `ab_compare.cpp:155-157`

2. `gridSearch()` 상한 검증
   - 위치: `param_tuner.cpp:32-60`
   - `steps` clamp + `total combinations` 상한 + 안전 곱셈(overflow 방지)

3. 이미지 크기 상한 및 비정상 입력 가드
   - NaN/Inf, 음수/이상치(face_width 포함) 사전 차단

**완료 기준**
- 비정상 입력/과대 조합에서도 OOM/무한 대기 없이 즉시 실패 반환

### Package C: 계산 비용/메모리 경량화

**대상 파일**
- `cpp/src/quality_metrics.cpp`

**필수 수정 항목**
1. SSIM 경량화
   - 위치: `quality_metrics.cpp:112+`
   - `CV_64F -> CV_32F` 전환
   - 중간 `Mat` 재사용 또는 생성 개수 축소

2. `toGray` 중복 제거
   - 위치: `quality_metrics.cpp:230-231, 320-321`
   - 게이트 평가 경로에서 gray 변환 1회 공유 구조로 리팩토링

3. Sobel/Laplacian 데이터 타입 하향
   - 위치: `quality_metrics.cpp:100, 201-202`
   - 모바일 대상 메모리/대역폭 효율 확보

**완료 기준**
- 기능 동일성 유지 + 기존 대비 실행 시간/메모리 사용 감소 확인

### Package D: 리포트 안정성

**대상 파일**
- `cpp/src/ab_compare.cpp`

**필수 수정 항목**
1. JSON escape 유틸리티 도입
   - 위치: `ab_compare.cpp:255+`
   - `condition_label` 등 문자열 필드 escape 처리

**완료 기준**
- 따옴표/역슬래시/개행 포함 라벨에서도 JSON 파싱 성공

### Package E: 테스트 보강

**대상 파일**
- `cpp/tests/test_quality_tuning.cpp`

**필수 수정 항목**
1. 현재 누락 테스트 추가
   - `ParamTuner::recommendPresets()`
   - `ParamTuner::generateReport()`
   - `ABCompare::reset()`

2. ReleaseGate 경계값 테스트 추가
   - `33.0ms`, `0.30/0.60`, `0.95`, 티어별 시간 경계 등

3. Halo 엣지케이스 테스트 추가
   - zero-baseline 경계 생성 케이스
   - `b_halo <= 0` 개선도 계산 케이스

4. 예외/실패 상태 분리 테스트 추가
   - 계산 실패가 "품질 실패"로 오인되지 않는지 검증

**완료 기준**
- 신규 로직별 회귀 테스트가 모두 추가되고 재현 케이스가 고정될 것

### Package F: 문서/구조 정리 (이번 사이클 내 포함)

**대상 파일**
- `cpp/include/iris_sdk/quality_metrics.h`
- `cpp/include/iris_sdk/release_gate.h`
- `cpp/include/iris_sdk/gpu/gpu_beauty_backend.h`
- 워크페이퍼 문서

**필수 수정 항목**
1. TemporalAnalyzer 스레드 안전성 계약 명시
2. DeviceTier 중복 개념 정리(ODR 이슈가 아니라 의미 일관성 이슈로 정리)
3. 임계값/가중치 근거 문서화
4. 워크페이퍼의 변수/체크리스트 누락 반영

**완료 기준**
- "왜 이 값인가"와 "어떻게 써야 안전한가"를 문서만으로 이해 가능

---

## 4. 검증 체크리스트 (수정 완료 판정)

1. 기능 정확성
- Halo 2건(zero-baseline, non-positive baseline) 재현 테스트 통과
- 계산 실패/품질 실패 구분 확인

2. 안전성
- 결과 벡터/그리드 탐색 상한 동작 확인
- 비정상 입력(NaN/Inf/음수/초대형 입력) 방어 확인

3. 성능/메모리
- SSIM/gradient 경량화 전후 비교 기록 확보
- 불필요 중복 변환 제거 확인

4. 리포트
- JSON escape 케이스 파싱 검증

5. 테스트
- `./build/bin/test_quality_tuning` 통과
- 필요 시 관련 단위 테스트 추가 실행

---

## 5. 비고

- 본 문서는 "수정 우선순위" 문서가 아니라 "수정 여부/방향 확정" 문서다.
- NO 3건은 이번 사이클 필수 수정에서 제외하되, 사용 맥락 변화 시 재평가한다.
- 특히 `addFrame()` 비용 항목은 실시간 호출 경로에 편입될 경우 즉시 재승격한다.
