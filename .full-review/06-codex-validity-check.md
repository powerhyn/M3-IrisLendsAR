# Review Validity Check (Codex)

## 목적

`.full-review/01~05` 문서의 핵심 주장들이 **현재 코드(HEAD: `c5b773e`, 2026-03-04)** 기준으로 타당한지 재검증한다.

## 판정 기준

- `타당`: 현재 코드에서 동일 문제가 재현되거나 근거가 명확함
- `부분 타당`: 방향은 맞지만 심각도/표현이 과장되었거나 전제가 부족함
- `비타당(구식)`: 과거에는 맞았지만 현재 코드는 이미 수정됨

## 요약 결론

- 전체적으로 리뷰 방향은 유의미하다.
- 다만 일부 항목은 **이미 수정된 이슈**를 계속 “미해결”로 표기하거나, 표현이 과하다.
- 현재 기준으로는 `타당` + `부분 타당` 중심으로 우선순위를 재정렬하는 것이 맞다.

---

## 1) 비타당(구식) 항목

아래 항목은 과거 이슈였으나 현재 코드에서는 해결되어, “현재 결함”으로 보긴 어렵다.

1. `Q5 / S1` (`std::stoi` 예외 위험)
   - 현재 `std::strtol` 사용으로 예외 전파 리스크 완화됨
   - 근거: `cpp/src/gpu/gpu_beauty_backend.cpp:1179, 1197`

2. `Q6 / S2` (`detectDeviceTier()` static public + GL 의존)
   - 현재 `private` 인스턴스 메서드
   - 근거: `cpp/include/iris_sdk/gpu/gpu_beauty_backend.h:248-254`

3. `Q3 / S3` (One Euro Filter reset 누락)
   - `release()` 및 얼굴 추적 끊김 분기에서 reset 수행
   - 근거: `cpp/src/gpu/gpu_beauty_backend.cpp:468-471, 1584-1587`

---

## 2) 부분 타당 항목

1. `P2` (mask center smoothing 적용 시점)
   - scissor 이전 이동으로 **일부 개선**은 맞다.
   - 그러나 FreqSep 마스크는 `computeROI()`에서 이미 생성되고, composite에서 UV 기준 샘플링되므로 “완전 해결”로 보긴 어렵다.
   - 근거:
     - 스무딩 위치: `cpp/src/gpu/gpu_beauty_backend.cpp:1571-1583`
     - 마스크 생성: `cpp/src/gpu/gpu_beauty_backend.cpp:1469`, `cpp/src/beauty_roi_manager.cpp:138-170`
     - 마스크 샘플링: `cpp/src/gpu/shader_sources.cpp:480`

2. `Q4 / C1` (GL_LINEAR 미복원)
   - 상태 복원 정책 이슈로는 타당하나, 즉시 기능 버그로 단정하긴 어려움
   - TexturePool 기본 생성값도 GL_LINEAR
   - 근거:
     - half-res 경로 재설정: `cpp/src/gpu/gpu_beauty_backend.cpp:1322-1330`
     - 풀 기본값: `cpp/src/gpu/texture_pool.cpp:331-334`

3. `T1`의 “커버리지 0%” 표현
   - **P4-W3-04 신규 기능 전용 테스트 부족**은 맞다.
   - 다만 프로젝트 전체 테스트가 0건이라는 의미로 읽히면 부정확하다.
   - 근거: `cpp/tests/` 다수 테스트 존재, 단 P4-W3-04 핵심 키워드 기반 테스트는 부재

---

## 3) 현재도 타당한 항목

1. `T2/T3/T4` (P4-W3-04 핵심 기능 테스트 공백)
   - OneEuro/DeviceTier/HalfRes 분기 전용 테스트 미확인
   - 근거: `cpp/tests` 내 키워드 검색 결과 부재, `mapSkinQuality` 중심 테스트만 존재
   - 예시: `cpp/tests/test_beauty_config_v2.cpp:300-386`

2. `D1` (CI/CD 파이프라인 부재)
   - `.github/workflows` 디렉터리 없음

3. `D1-Doc` (작업 문서와 실제 구현 불일치)
   - 문서는 “`mapSkinQuality()` 직후/UV 오프셋 적용” 뉘앙스
   - 실제 구현은 scissor 전 `face_rect.x/y` 직접 보정
   - 근거:
     - 문서: `docs/workPaper/P4-W3-04_temporal_stability_device_tier.md:64-65, 97-98`
     - 코드: `cpp/src/gpu/gpu_beauty_backend.cpp:1571-1583`

4. `Q2/A2/F1` (full-res/half-res 파이프라인 중복)
   - 구조적 중복이 크고 유지보수 비용 증가 리스크 존재
   - 근거: `cpp/src/gpu/gpu_beauty_backend.cpp:1030-1149` vs `1229-1368`

5. `F3` (동일 프레임 타임스탬프 불일치)
   - `OneEuroFilter::filter(value)`가 호출마다 now()를 읽어 축별/반경 필터가 동일 timestamp를 공유하지 않음
   - 근거:
     - 필터 구현: `cpp/include/iris_sdk/one_euro_filter.h:88-92`
     - 호출 지점: `cpp/src/gpu/gpu_beauty_backend.cpp:1577, 1578, 1648`

---

## 4) 우선순위 재정렬 제안

현재 코드 기준 추천 우선순위:

1. `테스트 보강` (T2/T3/T4)  
2. `문서 정합성 수정` (D1-Doc)  
3. `파이프라인 중복 리팩터링` (Q2/A2/F1)  
4. `frame_ts 공유`로 OneEuro 타임스탬프 일관화 (F3)  

구식 항목(Q3/Q5/Q6 등)은 최종 리포트에서 “해결됨”으로 명확히 분리하는 것이 적절하다.

