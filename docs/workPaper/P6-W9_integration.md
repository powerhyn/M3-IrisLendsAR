# P6-W9: 통합 테스트 + develop 머지

> **상태**: 🔄 진행 중 (2026-06-01). §1.0 현 상태 변화 반영 블록 우선 참조.
> **작성**: 2026-04-23 / **현 상태 갱신**: 2026-06-01
> **선행 의존**: 살아남는 W 완료 (W1/W2/W5 Phase A/B/W6 Phase A/W7 종결)
> **이월 트랙**: W3·W4·W8 — Phase 6 이월 (§1.0 참조)

---

## 1.0 현 상태 변화 반영 (2026-06-01) ⭐

W9 본문(§1.1~§1.14)은 2026-04-23 시점 가정으로 작성됨. 이후 다음 결정으로 시나리오/체크리스트가 바뀜. 본 블록이 **최신 단일 출처**.

### 1.0.1 트랙 상태 (현 시점)

| W | doc 본문 가정 | 실제 현 상태 | 출처 |
|---|---|---|---|
| W1 | 인터페이스 + ROI 측정 | ✅ 계약/fallback 완료, ROI 실측 source는 W6 이관 | `P6-W1_eye_render_packet.md` §1.0, 메모리 [[w1-measure-external-oes-collision]] |
| W2 | 블렌드 3종 + realSpec 폐기 | ✅ 완료 (LUMA_709 통일) | 커밋 `94546a2` |
| W3 | 환경 반사 scaffold (활성) | ⏸️ **Phase 6 이월** (OFF 기본 보존) | `P6-W3_*.md` §1.16, 메모리 [[w4-env-reflection-deferred]] |
| W4 | B2 24클립 벤치 | ⏸️ **Phase 6 이월** — 벤치 미진행 | `P6-W4_*.md` §1.17 |
| W5 | B1 + B8 벤치 | ✅ Phase A/B 완료, Phase C(흰자 빛남 수식)는 별도 W로 분리 | 머지 `abc9a84`, 메모리 [[w5-phase-a-b-done]] |
| W6 | B5/B9/C10 | ✅ Phase A 완료 (gate 기본 0.10) | 머지 `07d42a0` |
| W7 | B4 자동감지 fallback 채택 | ✅ 종결 — **셰이더 림발 영구 제거** (메타 인프라 보존) | §6.0.4, 커밋 `de1eeb7`/`8bdf825`, 메모리 [[limbal-in-asset-not-shader]] |
| W8 | Pupil material 조건부 | ⏸️ **자동 폐기** (W4 종속 + W4 이월) | `P6-W4_*.md` §1.17 연쇄 |

### 1.0.2 시나리오 재구성

§1.3 원안 9개 중 살아남는 항목 + 이월 항목:

| # | 원안 항목 | 현 시점 처리 |
|---|---|---|
| 1 | 양안 정상 동작 | ✅ 유지 |
| 2 | 블링크 ramp (W6) | ✅ 유지 |
| 3 | Sclera veto (W5 B8) | ✅ 유지 |
| 4 | **환경 반사 (W4 B2)** | ❌ **폐기 시나리오** (W3/W4 OFF 기본) |
| 5 | 블렌드 모드 전환 (W5 B1 4조합) | ✅ 유지 |
| 6 | 림발 표시 (W7 B4) | ⚠️ **단순화** — "에셋 baked 림발만 자연스러운지" 확인 (셰이더 림발은 영구 제거) |
| 7 | **Pupil 체감 (W8)** | ❌ **폐기 시나리오** (W8 이월) |
| 8 | 디테일 재주입 (W6 C10) | ✅ 유지 |
| 9 | 성능 (FPS, 메모리) | ✅ 유지 (단 GPU ms 정밀 측정은 생략, FPS + 평균 프레임시간만) |

### 1.0.3 SKU 세트는 원안 유지 (6개)

§1.4의 6 SKU × 3 tier = 18 시나리오 그대로. 단 환경 반사/Pupil 관련 항목은 평가에서 제외.

### 1.0.4 develop 머지 충돌 — **충돌 없음 확인**

`feature/P6-Works`는 `origin/develop`을 fully ancestor로 포함 (`git merge-base = fff737f`, 0 commits behind, 74 ahead). 즉:
- W3-04 (realSpec/고정 조명) 코드는 이미 P6-Works 안 `9aee86d` (S1 롤백)에서 제거됨
- §1.5 "W3-04 코드 처리" 항목 폐기
- §6.0 R1 결과 표의 6.1 "옵션 A 머지 커밋" 그대로 유효 (`--no-ff` 머지 커밋으로 W별 히스토리 + Phase 6 통합 시점 기록)

### 1.0.5 성능 프로파일링 범위 축소

§1.6의 GPU ms 정밀 측정(GLES timer query)은 W3-04 베이스라인 정밀치가 없어 비교 의미 없음 + 환경 반사 폐기로 +0.2~0.4ms 변동 항목 자체 사라짐. 그래서 **FPS + 평균 프레임시간**만 측정 (Android demo 기존 frame counter 재사용 or 최소 추가).

### 1.0.6 CI 체크리스트 슬림화 — `P6_implementation_handoff.md` §7 기준에서 제외/유지

| 항목 | 현 처리 |
|---|---|
| C++ 빌드 + ctest + APK 빌드 | ✅ 유지 |
| Shader compile 에러/warning 0 | ✅ 유지 |
| LUMA 계수 shader vs CPU 오차 ≤1% | ⚠️ 별도 테스트 케이스 미작성. 통합 리포트에 "후속 이월" 명기 (W1 측정 source가 W6 이관됨에 따라 검증 위치도 W6 후속 Phase로 이동) |
| 메모리 누수 (`EyeRenderPacket`/`LensSkuMetadata` 교체) | ⚠️ valgrind/sanitizer 미실행. 통합 리포트에 "Android 실기기 long-run 10분+ 메모리 변동 관찰"로 대체 |
| 3 tier jitter 프로파일링 | ✅ 유지 (FPS + 평균 프레임시간 측정으로 충족) |
| **W1 §6.2 ROI 반경 closure** | ✅ "W6 이관 — 실측 source 연결 후 결정" 명기 (메모리 [[w6-avg-iris-luma-measure]]) |
| **99 P6-W1→P6-W8 rename cross-ref** | ✅ 99 doc §4 이월 표 정정으로 처리 (별도 cross-ref 줄 추가) |
| **W4 B2 결과 확정** | ❌ 폐기 — W4 이월로 확정 자체 폐기 |

### 1.0.7 머지 메시지 (확정안)

```
Merge branch 'feature/P6-Works' into develop

P6 통합 — W1/W2/W5 Phase A·B/W6 Phase A/W7 완료.
W3·W4·W8 Phase 6 이월(보존 코드 + 재개 진입점 기록).

상세:
- W1: EyeRenderPacket 계약 + fallback (실측 source는 W6 이관)
- W2: 블렌드 3종 + realSpec 폐기 + LUMA_709
- W3: 환경 반사 scaffold (OFF 기본 보존)
- W4: B2 벤치 Phase 6 이월
- W5: B1/B8 Phase A/B 완료, Phase C 별도 W로 분리
- W6: 블링크 ramp + 저조도 gate + 디테일 재주입 (Phase A)
- W7: 셰이더 림발 영구 제거, 메타 인프라(sku_id/registry) 보존
- W8: 자동 폐기 (W4 이월 연쇄)

참조: docs/workPaper/P6-W9_integration_report.md
```

---

## 1. 인사이트 (세션 간 맥락 보존) ⭐ — 원본 (2026-04-23, 본문)

> ⚠️ §1.1~§1.14는 작성 시점 가정. **현 시점 변화는 §1.0이 권위 소스**. 본문은 의사결정 히스토리로 보존.


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
- 클라셋_누드 애쉬 로제 (애쉬+누드, 회색조 중성 톤 — P7-W3 정정: 웜톤 로제핑크 아님)

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

### 2.1 Phase 6 최종 관문

모든 단위 검증(W1~W8)이 끝난 상태에서 **통합 동작 재검증**. 특히:
- W 간 상호작용 (예: C9 avg_iris_luma × C10 저조도 gate × C7 블링크 ramp)
- 3 tier 디바이스 실기기 성능
- develop 머지 시 W3-04 코드 처리

### 2.2 W9 = 구현보다 검증/머지 위주

브레인스토밍 경량. 구현은 거의 없음 (fix-up 수준). 시간 대부분은:
- 실기기 테스트
- 성능 프로파일링
- 머지 준비 (PR, 충돌 해결)

### 2.3 W3-04 코드 처리 (feature 머지 전)

- 현재 develop: W3-04(고정 조명, realSpec 등) 포함
- feature/P5-W3-05: 70633ac(W3-04 이전) 기반 + W1~W8
- 머지 시 **W3-04 코드 삭제** 필요 → 충돌 해결 전략 확정.

---

## 3. 전제 조건

1. ✅ **W1~W7 완료**
2. ✅ **W8 판정 완료** (발동 시 구현, 폐기 시 문서화)
3. ✅ 3 tier 디바이스 각 1대 이상 확보 (HIGH/MID/LOW)
4. ✅ 평가자 1~2명 (단위 W와 다른 사람 권장, fresh eye)
5. ✅ develop 브랜치 최신 상태 확인

---

## 4. 목표

1. **통합 테스트 완료** — 전체 기능 3 tier 회귀 없음
2. **성능 프로파일링 완료** — FPS 목표 유지
3. **feature/P5-W3-05 → develop 머지**
4. **W3-04 코드 완전 제거** (develop에서 사라진 상태 확정)
5. **99_final_decision.md 최종 상태 갱신** — 모든 벤치 결과 반영
6. **CHANGELOG / Release notes** 작성 (해당 시)

### 4.1 Definition of Done

- [ ] 6 SKU × 3 tier = 18 시나리오 실기기 테스트
- [ ] FPS: HIGH 60+, MID 30+, LOW 20+ 유지
- [ ] 메모리 누수 없음 (10분+ 연속 사용)
- [ ] develop 머지 PR 작성 + CI 통과
- [ ] W3-04 코드 전부 삭제된 상태
- [ ] 99 및 모든 P6-W 문서 상태 "완료" 갱신
- [ ] 통합 테스트 리포트 (`docs/workPaper/P6-W9_integration_report.md`)

### 4.2 Out of scope

- Phase 7+ 이월 작업 (Head-pose 기반 env 회전, 3D FaceGeometry, HDR IBL 등)
- Android demo UI 대대적 재디자인 — 별도 PR 권장

---

## 5. 99에서 확정된 사항 (W9 최종 반영 대상)

> 🔧 **구현 시 참조**: [`P6_implementation_handoff.md`](P6_implementation_handoff.md) §7 CI / 머지 전 체크리스트 (**LUMA 오차 테스트 / 메모리 누수 / 3 tier jitter / W1 §6.2 ROI closure / 99 naming rename / W4 B2 결과 확정** 6개 Phase 6 특이 체크 포함), §2 소프트 갭 3건 처리 방침.

### 5.1 전체 기능 체크리스트 (§1.14 템플릿)

```
□ W1 EyeRenderPacket + avg_iris_luma ROI
□ W2 블렌드 3종 + realSpec 폐기
□ W3 환경 반사 가산 계층 + renderMask hook
□ W4 B2 결과 반영 (env / periphery / OFF)
□ W5 B1 결과 (Normal or CRL)
□ W5 B8 결과 (color-veto or luma-only)
□ W6 B5 결과 (블링크 up)
□ W6 B9 결과 (저조도 gate)
□ W6 C10 디테일 재주입
□ W7 B4 결과 (자동감지 fallback or 드롭)
□ W7 SKU 메타데이터
□ W8 발동/폐기 상태
```

### 5.2 성능 목표

| tier | FPS 목표 | 추가 GPU ms (Phase 6) |
|------|---------|---------------------|
| HIGH | 60+ | +0.4~0.7ms |
| MID | 30+ | +0.3~0.5ms |
| LOW | 20+ | +0.2~0.4ms |

### 5.3 머지 전략 (Claude 추천: 옵션 A — 머지 커밋)

```bash
git checkout develop
git merge feature/P5-W3-05  # fast-forward 아닌 merge commit 생성
# 충돌 발생 시 W3-04 코드(D1/D5 관련) 삭제 쪽 선택
git push origin develop
```

### 5.4 머지 방식 — **옵션 A (머지 커밋) 확정** (W9 R1 합의 3/3)

- W별 커밋 히스토리 보존 (벤치/판정 역추적).
- 충돌 발생 시 W3-04 삭제 쪽 선택.
- 머지 커밋 메시지: "P6 통합 — W1~W8 완료, 99 §1/§2 전 항목 반영".
- 출처: `P6-W9_brainstorm/synthesis.md` §1.

### 5.5 Android demo UI 정리 — **W9 범위 밖, Phase 7 초반 cleanup PR 확정** (W9 R1 합의 3/3)

- 블렌드 drop-down 축소, 3D Light 버튼 제거, 기본 blendMode TintLinearV2 ID=5 설정 포함.
- deprecated no-op 제거와 함께 Phase 7 초반 cleanup PR로 묶기 권장.
- 출처: `P6-W9_brainstorm/synthesis.md` §1, §2.

### 5.6 deprecated no-op 제거 — **W9 머지 유지 + Phase 7 초반 PR 확정** (W9 R1 합의 3/3)

- `setHighlightEnabled` 등 호환성 유예 기간 제공.
- Phase 7 초반 "호환성 transition 종료" 릴리즈에서 제거 + 릴리즈 노트 명기.
- 출처: `P6-W9_brainstorm/synthesis.md` §1.

### 5.7 W9 브레인스토밍 — **1라운드 경량 리뷰 완결 확정** (W9 R1 합의 3/3)

- 본 synthesis가 그 경량 리뷰. R2 불필요.
- W1~W8과 `99_final_decision.md` 정합성 확인 완료. 일치 매우 높음.
- 출처: `P6-W9_brainstorm/synthesis.md` §2.

### 5.8 성능 벤치 자동화 — **기존 FPS 재사용 + 최소 추가 확정** (W9 R1 합의 3/3)

- Android demo 기존 frame counter/timer 있으면 재사용.
- 없으면 `Choreographer.FrameCallback` 기반 60프레임 평균 FPS logcat 출력 (10줄 이내).
- CI 게이트 아닌 통합 리포트 항목으로 분리.
- **성능 기준 (CLAUDE.md):** 30fps+, 검출 33ms 이하, 메모리 100MB 이하, SDK 20MB 이하.
- 출처: `P6-W9_brainstorm/synthesis.md` §1.

### 5.9 릴리즈 노트 — **Phase 6 단일 1장 + 벤치 결과 요약 확정** (W9 R1 합의 3/3)

구조:
- 제목: "Phase 6: AR Lens Material & Sclera Bench"
- W1~W8 주요 결정 1줄씩
- 99 §1/§2 최종 상태 요약 (Fresnel C, TintLinearV2 canonical, 림발 10/10 등)
- **벤치 지표 수치화** (Gemini 강조): B1~B9 결과 정량 + 정성 체감 Y/N 집계
- **W8 조건부 여부 명기** (Codex 강조)
- **이월 항목 명기** (W1 §6.2 ROI closure 상태 등)
- Breaking change: 없음 (공개 API 불변)
- New features: env reflection scaffold, blend 3종 확정, conditional Pupil material restore
- 출처: `P6-W9_brainstorm/synthesis.md` §1.

### 5.10 CI 통과 확인 — **체크리스트 확정** (W9 R1 합의 3/3 기본 + 놓친 점 통합)

필수:
- [ ] C++ 빌드 (CLion `cmake-build-debug` + `scripts/build_android.sh` + `scripts/build_ios.sh`)
- [ ] 단위 테스트 (`cd cpp/cmake-build-debug && ctest`)
- [ ] Android demo APK 빌드 성공
- [ ] iOS Framework 빌드 성공 (해당 시)
- [ ] 기존 연결된 정적 분석/린트 통과
- [ ] **shader compile 에러/warning 0** (Phase 6 대규모 shader 수정)
- [ ] **LUMA 계수 shader vs CPU 오차 ≤1% 테스트 케이스** (Claude)
- [ ] **메모리 누수 테스트** — EyeRenderPacket/LensSkuMetadata 교체 시나리오 (Gemini)
- [ ] **3 tier 기기 프레임 타임 프로파일링** — jitter 확인 (Gemini)
- [ ] `docs/workPaper/` 각 W 문서 §5 최종 반영 확인
- [ ] `99_final_decision.md` §1.2 C5/D3 + §2 B1~B9 상태 "확정/구현됨" 반영
- [ ] **W1 §6.2 ROI 마스크 반경 closure 상태 명기** (Codex)
- [ ] **99 P6-W1 → P6-W8 rename cross-reference 추가** (Codex)
- [ ] **W4 B2 실측 결과 확정** — W8 포함 여부 결정
- 출처: `P6-W9_brainstorm/synthesis.md` §1, §2.

### 5.11 Phase 6 전체 점검 — **완료** (W9 R1 3모델 분업 병합)

3모델이 각자 다른 관점으로 Phase 6 전체 훑기 완료:
- **Codex:** 문서 정합성 (W1 ROI 이관 상태, 99 naming)
- **Gemini:** 런타임 건전성 (메모리 누수, jitter)
- **Claude:** 테스트 및 조건부 경로 (LUMA 테스트, W8 대기, demo blendMode)

놓친 점 7개 모두 §5.10 CI 체크리스트에 통합됨.

**99 정합성:** "매우 높음" (Gemini) / "전반적 일치" (Codex) / 전 축 매핑 (Claude §Phase 6 전체 점검 표).

출처: `P6-W9_brainstorm/synthesis.md` §2.

---

## 6. 미결 사항 (W9 브레인스토밍 R1 결과)

### 6.0 R1 결과 요약 (2026-04-24)

| 번호 | 원 쟁점 | 상태 | 반영 위치 |
|------|---------|------|-----------|
| 6.1 | 머지 방식 | ✅ **닫힘** (3/3 옵션 A) | §5.4 |
| 6.2 | Android demo UI | ✅ **닫힘** (3/3 별도 PR) | §5.5 |
| 6.3 | deprecated 제거 | ✅ **닫힘** (3/3 Phase 7) | §5.6 |
| 6.4 | W9 브레인스토밍 필수 | ✅ **닫힘** (3/3 경량 R1 완결) | §5.7 |
| 6.5 | 성능 벤치 | ✅ **닫힘** (3/3 기존 재사용) | §5.8 |
| 6.6 | 릴리즈 노트 | ✅ **닫힘** (3/3 단일 1장) | §5.9 |
| 6.7 | CI 통과 항목 | ✅ **닫힘** (3/3 기본 + 보완) | §5.10 |

**미결 없음.** Phase 6 전체 점검 3모델 분업 완료 (§5.11).

원문: `docs/workPaper/P6-W9_brainstorm/{codex,gemini,claude}_w9.md`.
종합: `docs/workPaper/P6-W9_brainstorm/synthesis.md`.

---

## (원 미결 사항 세부 — 참고용)

### 6.1 머지 방식 (A/B/C)

- A: 머지 커밋 (권장)
- B: 리베이스
- C: squash

**Claude 추천**: A. 이유: W별 커밋이 의미 있음 (벤치 결과 역추적).

### 6.2 Android demo UI 정리 범위

- W9 포함: 블렌드 drop-down 3종으로 축소, 3D Light 버튼 제거
- W9 범위 밖: 머지 후 별도 PR

**Claude 추천**: 후자. W9 범위 최소화.

### 6.3 deprecated no-op 함수 완전 제거 시점

- S1에서 `setHighlightEnabled` deprecated no-op 상태
- W9 머지 전에 완전 제거? 유지?

**Claude 추천**: 머지 유지, Phase 7 초기에 제거. 호환성 transition 기간 확보.

### 6.4 W9 브레인스토밍 필수 여부

W9는 검증 위주 → 경량. 단 머지 전 Codex/Gemini에게 최종 리뷰 요청 가능:
- "전체 Phase 6 작업이 99와 일치하는가?"
- "놓친 부분 없는가?"

**Claude 추천**: 경량 리뷰 1라운드 권장. 결과 파일 `20_w9_final_review.md`.

### 6.5 성능 벤치 자동화

Android demo에 FPS 측정 로직 이미 존재 여부?
- 있으면 그대로 활용
- 없으면 간단한 frame counter 추가

### 6.6 릴리즈 노트 작성

- Phase 6 전체 변경사항 요약 (CHANGELOG)
- 각 W 주요 결정 + 벤치 결과
- 사용자/고객사에 전달할 내용

### 6.7 CI 통과 확인 항목

- C++ 빌드 (3 platform 가능 시)
- 단위 테스트
- Android demo APK 빌드
- 정적 분석 (clang-tidy 등, 해당 시)

---

## 7. W9 통합 테스트 체크리스트

### 7.1 각 tier 테스트 시나리오 (6 SKU × 3 tier = 18 시나리오)

**SKU 6종** (P6-W0 §1.5):
- 클라셋_돌 초코, 클라셋_런웨이 그레이, 오(OH)_베이글
- 엔비_퍼퓸 글로우, 엔비_샤모 브라운, 클라셋_누드 애쉬 로제

**각 시나리오 확인 항목** (P6-W9 §1.3):
1. 양안 정상 동작
2. 블링크 ramp 자연스러움
3. Sclera veto 흰자 번짐 없음
4. 환경 반사 (W4 채택 소스) 자연 반응
5. 블렌드 모드 전환 매끄러움
6. 림발 표시 (메타 있음/없음)
7. Pupil 체감 (W8 발동 시)
8. 디테일 재주입 저조도 노이즈 없음
9. FPS 목표 유지
10. 메모리 누수 없음

### 7.2 머지 전 체크리스트

```
□ 전체 기능 테스트 완료
□ 성능 목표 달성
□ 모든 W 문서 상태 "완료"
□ 99_final_decision.md 최종 반영
□ W3-04 코드 완전 삭제 확인
□ CHANGELOG 작성
□ PR 제목 + 본문 작성
□ CI 통과
□ Codex/Gemini 경량 리뷰 (선택)
□ 리뷰어 승인
```

### 7.3 머지 후 Post-mortem (선택)

- Phase 6 전체 소요 시간 vs 예상 (99 §3: 20~28h)
- 브레인스토밍 오버헤드 평가
- R4 Patch 방식 유효성
- Claude 편향 교정 횟수 집계
- 메모리/로그 갱신 필요 사항

---

## 8. 완료 정의 + Phase 6 종료

### 8.1 완료 정의

§4.1 체크리스트 + develop 머지 완료.

### 8.2 커밋 전략

- `docs(P6-W9): 섹션 2~8`
- `chore(integration): P6-W9 통합 테스트 리포트`
- `docs(99): 모든 벤치 결과 최종 반영`
- `chore(demo): W3-04 잔여 코드 정리` (머지 충돌 해결 커밋)

### 8.3 Phase 6 종료 선언

W9 머지 완료 시점에:
- 99_final_decision.md 전체 "완료" 상태
- MEMORY.md 갱신 (feedback 관련 업데이트)
- Phase 7 계획서 초안 (별도 워크페이퍼 `P7-W0_index.md` 등)

### 8.4 Phase 7+ 이월 목록 (§1.10 반복)

- 3D Face Geometry 기반 렌더링
- HDR IBL
- Full PBR
- Neural rendering
- Corneal refraction 실시간
- 속눈썹 전용 세그멘테이션
- Head-pose 기반 env 회전 실제 활용 (W2 refiner 출력 후)
- Pupil 검출 정확도 향상 (Pupil_center 기반 Option B 업그레이드)

### 8.5 W9 완료 시 파일 상태

```
docs/workPaper/
├── P5-W3-05_brainstorm/  (전체, 브레인스토밍 히스토리)
│   └── 99_final_decision.md  (최종 완료 상태)
├── P6-W0_index.md        ("완료" 표기)
├── P6-W1~W9              (모든 상태 완료)
├── (W8 발동 시) P6-W8 결과
└── P7-W0_index.md        (Phase 7 시작)
```

### 8.6 기대 효과

- **렌즈 렌더링 자연스러움 경쟁사 Parity 달성** (Perfect Corp, 피팅몬스터 수준)
- **SDK 코드 품질 향상** — 블렌드 단순화, realSpec 제거, 구조화된 계약
- **확장성 확보** — renderMask hook, EyeRenderPacket, SKU 메타 등 향후 기능 추가 기반
- **프로세스 모범 사례** — 4라운드 브레인스토밍 + 실측 반영 + 다중 AI 교차 검증 (메모리 기록 완료)

---

## 참조

- 99_final_decision.md §3 S5
- 각 P6-W*_*.md 문서
- `docs/workPaper/P5-W3-05_brainstorm/` 폴더 전체 (브레인스토밍 히스토리)
