# P6-W4 B2 벤치 결과 리포트

> **Phase C 진입 시 작성.** 현재는 placeholder.
> **명세 출처**: `docs/workPaper/P6-W4_env_reflection_bench.md` §4.1 / §5.5 / §5.6

---

## 메타

| 항목 | 값 |
|------|-----|
| 촬영일 | ____ |
| 평가일 | ____ |
| 디바이스 | (예: Galaxy S23) |
| 빌드 SHA | (촬영 시점 origin/feature/P6-Works) |
| SKU | TintLinearV2 + iris_mat_B |
| env_map | (placeholder 회색 사무실 / 정식 ACES 사무실 — 선택) |
| intensity | 0.3 기본 (sweep 시 별도 명기) |
| 평가자 3명 | R1 아시아 짙은 홍채 / R2 밝은 홍채 / R3 개발자 |

---

## 1. 정성 메트릭 집계 (24 클립 × 3 평가자 = 72 응답)

### 1.1 자연스러움 평균 (1~5)

| 프로토타입 | 평균 | 표준편차 | 최저 | 최고 |
|----------|------|---------|------|------|
| P0 (OFF) | x.xx | x.xx | x | x |
| P1 (EnvMap) | x.xx | x.xx | x | x |
| P2 (Periphery) | x.xx | x.xx | x | x |

### 1.2 음성 메트릭 다수결 (2/3 룰)

| 프로토타입 | center_glare | face_reflect | asymmetry | env_disconnect |
|----------|--------------|--------------|-----------|----------------|
| P0 (OFF) | Y/N | Y/N | Y/N | Y/N |
| P1 (EnvMap) | Y/N | Y/N | Y/N | Y/N |
| P2 (Periphery) | Y/N | Y/N | Y/N | Y/N |

### 1.3 Pupil cavity 2/3 룰 (W8 발동 판정)

- 24 클립 중 2/3 이상 Y 비율: P0 ___%, P1 ___%, P2 ___%
- 환경별/동작별 패턴: (관찰 메모)

**W8 판정**:
- [ ] 착수 (2/3 이상 클립에서 Pupil cavity Y)
- [ ] 폐기 (모든 클립에서 N)
- [ ] 사용자 최종 판단 (경계)

---

## 2. 채택 프로토타입 결정 (§5.5 시나리오)

| 시나리오 | C5 소스 | D3 realSpec | W8 |
|---------|---------|-------------|----|
| env-map 명확 우세 | env-map | 완전 폐기 | (별도) |
| periphery 명확 우세 | periphery | 완전 폐기 | (별도) |
| 둘 다 OFF 대비 개선 | 2차 hybrid 검토 | 완전 폐기 | (별도) |
| 둘 다 차이 미미 | **Phase 6 이월** | **조건부 유지** (재도입 검토) | 보통 체감 → 착수 |

**판정 결과**: ____

**근거 요약**: (자연스러움 + 음성 메트릭 + Pupil 체감 종합)

---

## 3. Phase C 액션 체크리스트 (§5.13)

- [ ] `99_final_decision.md` §1.2 C5 상태 업데이트
- [ ] 99 §1.1 D3 상태 업데이트
- [ ] 99 §2 B2 결과 섹션 신규 추가
- [ ] `P6-W3_env_reflection_scaffold.md` §5.11 sampleReflection 채택 소스 커밋 SHA 참조
- [ ] `P6-W4_env_reflection_bench.md` §5 최종 반영 + 본 리포트 링크
- [ ] Pupil 2/3 판정 → W8 착수/폐기/사용자 판단 트리거
- [ ] 채택 소스 정식 구현 + 머지 (Phase A 코드 위에서 OFF 분기 제거 또는 정리)
- [ ] 실기기 회귀 확인 (채택 소스 적용 후 30fps 유지)

---

## 4. 학습 내용 / 다음 W 영향

(Phase C 후 작성 — 본 벤치에서 발견된 후속 W 의제, 메모리 저장 후보 등)
