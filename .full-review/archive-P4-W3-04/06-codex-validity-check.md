# Review Validity Check (Codex)

## 목적

`.full-review/01~05` 문서의 핵심 주장들이 **현재 코드(HEAD: `997b5c5`, 2026-03-05)** 기준으로 타당한지 재검증한다.

## 판정 기준

- `타당`: 현재 코드에서 동일 문제가 재현되거나 근거가 명확함
- `부분 타당`: 방향은 맞지만 심각도/표현이 과장되었거나 전제가 부족함
- `비타당(구식)`: 과거에는 맞았지만 현재 코드는 이미 수정됨
- `이슈 아님`: 코드 재검증 결과 실제 문제가 아닌 것으로 판정

## 요약 결론

- 리뷰에서 발견된 코드 품질 이슈는 **4차 커밋에 걸쳐 모두 해결**되었다.
- 남은 미해결 항목은 **CI/CD 인프라**(D1, D3~D8, D11, D12)와 **향후 기능 작업**(P1-Perf, P2 마스크 안정화)뿐이다.
- 일부 항목(S6, F5, T7, Q4, S9)은 코드 재검증 결과 **실질적 이슈가 아닌 것**으로 판정되었다.

---

## 1) 비타당(구식) — 이미 해결된 항목

아래 항목은 과거 이슈였으나 후속 커밋에서 해결되었다.

### b80377f에서 해결 (Codex 1차 피드백)

1. `Q5 / S1` (std::stoi 예외 위험) → `std::strtol` 교체
2. `Q6 / S2` (detectDeviceTier() static public) → private 인스턴스 메서드
3. `Q3 / S3` (One Euro Filter reset 누락) → release() + 추적 끊김 리셋

### f8b96fb에서 해결 (Comprehensive review 13항목)

4. `F3` (OEF 타임스탬프 불일치) → frame_ts 캡처 후 3필터 동기화
5. `F6` (DeviceTier enum public) → private 이동
6. `F7` (Temporal filter 리셋 분산) → resetTemporalFilters() 헬퍼
7. `F9 / S10` (LOG 매크로 dangling-else) → do{...}while(0)
8. `F11` (M_PI 비표준) → constexpr kPi
9. `Q8 / D5-Doc` (Mali 분류 비대칭 근거) → 주석 추가
10. `Q9` (매직 넘버 28) → constexpr kMaxGaussianRadius
11. `S7 / D4-Doc` (OEF thread safety/파라미터 근거) → 주석 추가
12. `D2-Doc / D3-Doc` (Doxygen 미갱신) → 갱신 완료
13. `D1-Doc` (작업 문서 §3.2 불일치) → 문서 정합성 수정

### 5c14979에서 해결 (테스트 35건)

14. `T1 / D2` (P4-W3-04 전용 테스트 부재) → 35건 추가
15. `T2` (OneEuroFilter 단위 테스트) → 수렴/리셋/파라미터 테스트
16. `T3` (detectDeviceTier 테스트 불가) → classifyGpuRenderer() static 분리
17. `T4` (DeviceTier 분기 통합 테스트) → tier별 파이프라인 분기 테스트

### 997b5c5에서 해결 (R1 리팩터링)

18. `Q2 / A2` (파이프라인 ~140줄 중복) → executeFreqSepPipelineImpl 통합
19. `Q7` (applyTextureId 중첩 깊이) → 파이프라인 추출로 해소
20. `D9-Doc` (HalfRes 함수 주석 부족) → Impl + FreqSepExecConfig 자기 문서화

---

## 2) 이슈 아님 — 코드 재검증 결과 판정

### S6 (Viewport 복원 RAII 미적용)
- **재검증**: executeFreqSepPipelineImpl의 에러 경로(`:1147-1149` 해상도 체크, `:1157-1162` 텍스처 획득 실패) 모두 viewport 변경(`:1177`) **이전**에 반환됨
- **결론**: viewport 변경 후 early return 없음 → 복원 누락 불가. RAII guard는 과잉 엔지니어링.

### F5 (computeGaussianWeights C 스타일 배열)
- **재검증**: 스택 할당(29 floats = 116 bytes), constexpr 크기, std::clamp 경계 보호, 미사용 엔트리 0 초기화
- **결론**: 메모리 안전성 문제 없음. std::array 전환은 순수 스타일 선호.

### T7 (Half-Res 경계 테스트 부재)
- **재검증**: blur_w < 1 가드(`:1147`), 실제 카메라 해상도 항상 짝수(720/1080/2160)
- **결론**: 실제 발생 시나리오 없음. 방어적 테스트 가치는 있으나 우선순위 최하.

### Q4/C1 (GL_LINEAR 중복 glTexParameteri)
- **재검증**: TexturePool 생성 시 GL_LINEAR 설정(`texture_pool.cpp:331-332`), cfg.linear_upsample 경로에서만 실행(MID 한정), 드라이버 no-op 처리
- **결론**: 방어적 코드. 제거해도 기능/성능 차이 없고, 유지해도 문제 없음.

### S9 (half_w/half_h 홀수 해상도 오프셋)
- **재검증**: GL_LINEAR 하드웨어 bilinear 보간이 비정수 배율도 자연스럽게 처리, 저주파 성분에서 1px 무의미
- **결론**: 이론적 엣지 케이스이나 실제 문제 발생 불가.

### F10 (static_cast\<unsigned char\> 반복)
- **재검증**: Adreno/Mali 파서에서 2회 사용. 헬퍼 추출 시 오히려 코드 복잡도 증가.
- **결론**: 현재 수준으로 충분.

### D10-Doc (CHANGELOG 미갱신)
- **재검증**: 프로젝트에 CHANGELOG 파일 자체가 없음
- **결론**: 해당 없음. 향후 CHANGELOG 도입 시 작성.

---

## 3) 부분 타당 — 효과가 제한적이거나 후속 작업으로 이관

1. `P2` (mask center smoothing)
   - scissor 안정화는 유효하나, 마스크 플리커 해결이라는 원래 목표에는 미치지 못함
   - P4-W3-05 실기기 테스트에서 체감 여부 확인 후 판단
   - 코드에 TODO(P4-W3-04-R2) 기록됨

---

## 4) 현재도 타당 — CI/CD 인프라 및 향후 기능

1. `D1` (CI/CD 파이프라인 부재) — `.github/workflows` 없음
2. `D3~D8, D11, D12` (빌드/릴리즈 인프라) — 별도 작업으로 분리
3. `P1-Perf / S8` (정적 DeviceTier 한계) — P4-W3-05 런타임 적응형 tier에서 해결 예정

---

## 5) 최종 현황 요약

| 분류 | 건수 |
|------|------|
| 해결 완료 | 45건 |
| 이슈 아님 (재검증) | 12건 |
| CI/CD 인프라 (별도) | 15건 |
| 부분 수정/후속 추적 | 3건 (P2, P1-Perf, S8) |
| **코드 품질 미해결** | **0건** |
