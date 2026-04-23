# P6-W9: 통합 테스트 + develop 머지

> **상태**: 인사이트 작성 완료. 세부 계획 본문 작성 대기.
> **작성**: 2026-04-23
> **선행 의존**: 전체 W 완료 (W1~W7 + W8 여부 결정)

---

## 1. 인사이트 (세션 간 맥락 보존) ⭐

### 1.1 이 W의 역할

**Phase 6 최종 관문**. 각 W에서 단위 검증은 끝났지만 **통합 상태에서 재검증**. 특히:
- 여러 W의 결정이 상호작용하는 부분 (예: C9 avg_iris_luma 측정이 C10 저조도 gate와 같이 쓰일 때)
- HIGH/MID/LOW tier별 실기기 성능
- feature/P5-W3-05 → develop 머지 충돌 대응

### 1.2 통합 테스트 시나리오 (99 §3 S5)

**HIGH tier 실기기** (최신 갤럭시/픽셀, Adreno 740+ / Mali-G78+):
- 전체 기능 ON (환경 반사, LTL, 모든 블렌드)
- FPS 유지 여부 (목표: 60fps)
- 메모리 사용량

**MID tier** (보급형, Adreno 630~650 / Mali-G76):
- 기본 기능 동작 확인
- FPS 목표: 30fps
- W3-04에서 겪은 Adreno mipmap + dynamic branch 이슈 재발 여부

**LOW tier** (저사양, Adreno 530 이하):
- 최소 기능 (렌즈 오버레이만)
- FPS 목표: 20fps
- 크래시 방지 우선

### 1.3 실기기 테스트 체크리스트

**각 tier에서 확인할 것**:
1. **양안 정상 동작** — 좌우 대칭, 좌우 독립 렌더링
2. **블링크 ramp** (W6 결과) — down/up 자연스러움
3. **sclera veto** (W5 B8) — 흰자 번짐 없음
4. **환경 반사** (W4 B2) — 다양한 환경에서 자연 반응
5. **블렌드 모드 전환** — Normal/Multiply/Screen/LTL/CRL 매끄럽게
6. **림발 표시** (W7 B4) — 메타 있는 SKU, 없는 SKU 모두
7. **Pupil 체감** (W8 적용 여부) — 중앙 공동 감각
8. **detail 재주입** (W6 C10) — 저조도 노이즈 없음
9. **성능**: FPS 측정, 프로파일링

### 1.4 테스트 SKU 세트 (통합 테스트용)

**6개 대표 SKU**:
- 클라셋_돌 초코 (짙은 불투명)
- 클라셋_런웨이 그레이 (밝은 톤)
- 오(OH)_베이글 (중간 자연)
- 엔비_퍼퓸 글로우 (쿨톤, Pupil 체감)
- 엔비_샤모 브라운 (그래픽)
- 클라셋_누드 애쉬 로제 (웜톤)

각 SKU × 3 tier = 18 시나리오. 기기당 1 SKU 테스트에 10분 → 총 3시간.

### 1.5 develop 머지 준비 체크리스트

**1. 브랜치 정리**:
```
feature/P5-W3-05 (현재)
  ├── d09bf72 docs R1+R2
  ├── d86598f docs R3+R4
  ├── 9aee86d refactor S1 롤백
  ├── W1 commits
  ├── W2 commits
  ├── W3 commits
  ├── W4 commits + bench report
  ├── W5 commits + bench report
  ├── W6 commits + bench report
  ├── W7 commits + bench report
  ├── [W8 commits if applicable]
  └── W9 통합 + 머지 준비
```

**2. develop과 리베이스 or 머지**:
- **옵션 A (머지)**: `git merge feature/P5-W3-05` → develop에 P5-W3-05 작업 전체 추가. 커밋 히스토리 보존.
- **옵션 B (리베이스)**: develop 위에 feature 커밋 재배치. 히스토리 깔끔.
- **옵션 C (squash 머지)**: feature 전체를 1 커밋으로 압축. 히스토리 간결.

**Claude 제안**: 옵션 A. W별 커밋이 의미 있음 (벤치 결과 역추적). 리베이스 충돌 리스크 회피.

**3. develop의 W3-04 코드 처리**:
- 현재 develop은 W3-04 (고정 조명, realSpec 등) 포함. feature/P5-W3-05는 W3-04 **이전** 시점(70633ac) 기반.
- 머지 시 충돌 발생 가능성 — W3-04 코드가 삭제되어야 함.
- **해결**: 머지 전에 develop에서 W3-04 리버트 또는 feature를 develop 위에 리베이스하면서 W3-04 코드 삭제.

**4. 머지 PR 준비**:
- PR 제목: `feat(gpu-lens): P5-W3-05 렌즈 렌더링 자연스러움 완성 (Phase 6 전체)`
- PR 본문: Phase 6 각 W 요약 + 벤치 결과 링크 + 99_final_decision.md 인용
- CI 통과 확인 (빌드, 테스트)

### 1.6 성능 프로파일링

**프레임당 GPU 시간 측정**:
- 기존 (W3-04 상태): 측정치 확인 (develop에서 베이스라인)
- Phase 6 완료 상태: 비교 측정

**예상 변화**:
- 블렌드 정리: -0~0.1ms (코드 simpler)
- 환경 반사 계층: +0.2~0.4ms (fetch 추가)
- detail 재주입: +0.1~0.2ms (3x3 blur)
- avg_iris_luma measurement: +0.1ms (CPU 경로)
- **총 +0.4~0.7ms 예상**. 30fps(33ms)에서 여유 있음.

### 1.7 각 W 결과 최종 집계

**W1 완료 시점**:
- EyeRenderPacket 도입 ✓
- avg_iris_luma ROI 측정 ✓

**W2 완료 시점**:
- 블렌드 3종 확정 (TintLinearV2/Multiply/ScreenLinear) ✓
- realSpec 폐기 구조 완료 (W4 결과 종속 조건부 유지는 폐기 확정까지)

**W3 완료 시점**:
- 환경 반사 계층 스캐폴드 ✓
- renderMask hook 구조 ✓
- Fresnel 수식 확정

**W4 완료 시점**:
- B2 반사 소스 확정 (env-map / periphery / OFF)
- D3 realSpec 최종 처리
- P6-W8 착수 여부 확정

**W5 완료 시점**:
- B1 블렌드 4번째 슬롯 (Normal or CRL)
- B8 sclera veto 방식 (color-veto or luma-only)

**W6 완료 시점**:
- B5 블링크 up 시간 (60/80/120ms)
- B9 저조도 gate 임계값
- C10 detail 재주입 구현 완료

**W7 완료 시점**:
- B4 림발 자동 감지 fallback (채택/드롭)
- SKU 메타데이터 구조

**W8 (조건부) 완료 시점**:
- Pupil 재질 반투명 복원 (Option E 적용)
- OR 트랙 폐기 결정 문서화

### 1.8 W9 브레인스토밍은 필요 없을 수도

**특성**: W9는 **통합 테스트 실행 + 머지 준비**. 브레인스토밍보다는 실행 위주.

단 머지 전에 Codex/Gemini에 최종 리뷰 요청 가능:
- "전체 Phase 6 작업이 99_final_decision.md와 일치하는가?"
- "놓친 부분 없는가?"
- "머지 전 추가 테스트 제안?"

1 라운드 경량 리뷰 충분.

### 1.9 W9에서 수정할 파일 (거의 없음)

- 신규:
  - `docs/workPaper/P6-W9_integration_report.md` — 통합 테스트 결과
  - PR 본문 (GitHub)
- 수정:
  - 99_final_decision.md — 각 벤치 결과 최종 반영, "완료" 표시
  - P6-W*_*.md 각 문서 — 상태 "완료" 갱신
- 제거:
  - Deprecated no-op 함수들 (setHighlightEnabled 등) — 호환성 끝났으니 완전 제거 고려. 또는 유지. 사용자 판단.

### 1.10 Known Issues / 향후 작업 (W9에서 정리)

**Phase 6 범위 밖 이월**:
- 3D Face Geometry
- HDR IBL
- Full PBR
- Neural rendering
- Corneal refraction
- 속눈썹 전용 세그멘테이션
- Head-pose 기반 env 회전 실제 활용 (W2 refiner 출력 후)

이 목록은 **Phase 7 or 이후** 트랙으로 넘김. 99_final_decision.md §4 참조.

### 1.11 머지 후 Android Demo 정리

**S1에서 deprecated로 둔 것들**:
- `IrisLensSDK.setLensHighlight()` @Deprecated
- 관련 UI 토글 (3D Light 버튼 등 W3-04에서 추가됐으나 현 branch엔 없음)

**머지 후 별도 PR**: Android demo에서 setLensHighlight 호출 제거. deprecated 함수 완전 제거.

### 1.12 회귀 테스트 (develop 머지 직전)

1. **기존 기능 불변 확인**:
   - 얼굴 검출 정상
   - iris 좌표 안정화 (W1 TemporalStabilizer)
   - 기본 렌즈 오버레이 (렌즈 없을 때 카메라 그대로)
2. **새 기능 통합 동작**:
   - EyeRenderPacket 경유 정상
   - 블렌드 4종 전환 매끄러움
   - 환경 반사 기본 동작
3. **성능 회귀 없음**:
   - FPS 유지
   - 메모리 누수 없음 (extended run 10분 이상)

### 1.13 W9 최종 체크리스트 템플릿

```markdown
## W9 완료 체크리스트

### 통합 테스트
- [ ] HIGH tier 디바이스 테스트 (SKU 6개)
- [ ] MID tier 디바이스 테스트
- [ ] LOW tier 디바이스 테스트
- [ ] 성능 프로파일링 (FPS, GPU ms, 메모리)
- [ ] 통합 테스트 리포트 작성

### 머지 준비
- [ ] develop과의 충돌 해결 전략 확정
- [ ] W3-04 코드 처리 확인 (리베이스 or 리버트)
- [ ] PR 작성
- [ ] CI 통과 확인
- [ ] Codex/Gemini 최종 리뷰 (선택)

### 문서 갱신
- [ ] 99_final_decision.md 각 벤치 결과 최종 반영
- [ ] P6-W*_*.md 모든 문서 "완료" 상태 전환
- [ ] README/CHANGELOG 업데이트 (해당 시)

### Known issues 정리
- [ ] Phase 7+ 이월 목록 최종 확정
- [ ] Android demo 정리 별도 PR 준비
```

### 1.14 W9 소요 시간

- 통합 테스트: 3~4시간
- 성능 프로파일링: 1시간
- 머지 준비 + PR: 1~2시간
- 리뷰 라운드: 1시간 (선택)

**총 5~8시간**.

---

## 2. 배경/맥락
_TODO_

## 3. 전제 조건
_TODO: W1~W7 완료 + W8 결정_

## 4. 목표
_TODO_

## 5. 99에서 확정된 사항
_TODO_

## 6. 미결 사항
_TODO_

## 7. W 통합 테스트 체크리스트
_TODO (§1.13 템플릿 확장)_

## 8. 완료 정의 + Phase 6 종료
_TODO_

---

## 참조

- 99_final_decision.md §3 S5
- 각 P6-W*_*.md 문서
- `docs/workPaper/P5-W3-05_brainstorm/` 폴더 전체 (브레인스토밍 히스토리)
