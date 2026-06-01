# P6-W5 B1+B8 벤치 — 36클립 촬영 체크리스트

> **목적**: B1(Normal vs CRL) + B8(color-veto vs luma-only) 4조합 동시 비교.
> **명세**: `docs/workPaper/P6-W5_blend_sclera_bench.md` §5.1 / §5.3 / §5.6
> **시작일**: ____
> **완료일**: ____

---

## 메타 정보

| 항목 | 값 |
|------|-----|
| 디바이스 | (예: Galaxy S23, OS 14) |
| 디바이스 시리얼 | (예: R3CW20BBFCM) |
| 투명도/크기/경계 | 기본값 유지 (slider 조정 금지 — 변수 통제) |
| 투명도 (opacity) | **80% 고정** (블렌드 차이 가시성 최적 — 앱 기본 40%는 너무 옅어 차이 안 보임, 100%는 차이 압축) |
| maxDetail (CRL clamp) | **1.25 고정** (§5.10 — W5 1차 튜닝 금지). native 하드코딩값, demo 밝기 슬라이더는 Kotlin 폴백 전용이라 무관 |
| color-veto 강도 | 0.6 고정 (§5.14 — W5 1차 고정) |
| Sclera Protect | **ON** (4조합 모두 veto 분기 활성 전제) |
| 빌드 SHA | (촬영 시점 origin/feature/P6-Works HEAD) |
| 촬영자 | (이름) |

> ⚠️ **촬영 전 필수 (F-01 가드)**: 앱 실행 후 logcat에서 `SDK C++ GPULensRenderer active` 로그를 확인할 것.
> `fallback to Kotlin shader` 경고가 뜨면 native lens 비활성 상태 → 폴백 Kotlin 셰이더는 `uScleraVetoMode`/`blendColorReplaceLinear`가 없어 **4조합이 동일하게 렌더되어 B1/B8 판정이 무효**. 해당 take는 폐기하고 native 활성 후 재촬영.

---

## SKU 6종 (§5.1 + §5.8 별도 확보)

> demo 렌즈는 실제 타겟 라인업 42종(`docs/lens-ar/`, 1000px)으로 교체됨. 아래는 demo 표시명 ↔ 파일명.

| 코드 | 분류 (실측 재분류 2026-05-27) | demo 표시명 | 파일명 |
|------|------|-------------|--------|
| S1 | 짙은 갈색 + 패턴 (대조군) | claset doll choco | `claset_doll-choco.png` |
| S2 | 헤이즐/베이지 (자연 톤) | oh bagel | `oh_bagel.png` |
| S3 | 밝은 그레이/블루 (채도↓, B8 위험) | envie parfum glow | `envie_parfum-glow.png` |
| S4 | 강한 방사 그래픽 | claset never olive | `claset_never-olive.png` |
| S5 | 도트 그래픽 (CRL 검증 핵심) | envie chameau brown | `envie_chameau-brown.png` |

> ⚠️ **실측 정정 (2026-05-27, 실기기+contact sheet)**: lens-ar 42종 전부 *도트/방사 패턴 + 중심 투명* 구조 — **완전 "불투명 서클" 없음**. 기존 분류 정정:
> - S4 "불투명 서클"(romu dear-mellow)은 실물상 옅은 회색 도트라 부적합 → **강한 방사 그래픽(claset never-olive)** 으로 교체.
> - S5 "화이트/그래픽"은 화이트 아님(chameau-brown=연갈+주황 도트) → **그래픽(도트 외곽)** 으로 정정.
> - 1차 형광 체감 결과: `report.md` §2 (B1: TintLinearV2 충분, CRL 4번째 슬롯 약함).

---

## 조명 3종 (§5.3 — B8 채도 불안정 테스트)

- **E1 형광**: 실내 형광등 (5000K~6500K, 정상 대비)
- **E2 측광**: 창가 한쪽 자연광 (그림자 영역 발생)
- **E3 저조도**: 야간 실내 (3000K 이하, 채도 부정확 — luma-only 위험 영역)

---

## Take 매트릭스 (9 take × 4조합 = 36클립)

> 📌 §5.6 핵심 — **같은 take 12초 녹화 중 A→B→C→D 버튼 연속 토글**. 후편집으로 4 클립 분리.
> 즉 한 take당 **녹화 1번** → 후편집에서 A/B/C/D 클립 4개로 분할.

| Take | SKU | 조명 | 주 목적 | 원본 파일 | A | B | C | D | 상태 |
|------|-----|------|---------|-----------|---|---|---|---|------|
| T1 | S1 다크브라운 | E1 형광 | B1 (대조군) | `raw_T1.mp4` | (랜덤) | (랜덤) | (랜덤) | (랜덤) | ⏳ |
| T2 | S2 헤이즐 | E1 형광 | B1 | `raw_T2.mp4` | (랜덤) | (랜덤) | (랜덤) | (랜덤) | ⏳ |
| T3 | S3 밝은그레이 | E1 형광 | B1 + B8 | `raw_T3.mp4` | (랜덤) | (랜덤) | (랜덤) | (랜덤) | ⏳ |
| T4 | S4 불투명서클 | E1 형광 | B1 | `raw_T4.mp4` | (랜덤) | (랜덤) | (랜덤) | (랜덤) | ⏳ |
| T5 | S5 화이트그래픽 | E1 형광 | B1 (CRL 핵심) | `raw_T5.mp4` | (랜덤) | (랜덤) | (랜덤) | (랜덤) | ⏳ |
| T6 | S3 밝은그레이 | E3 저조도 | B8 (채도 위험) | `raw_T6.mp4` | (랜덤) | (랜덤) | (랜덤) | (랜덤) | ⏳ |
| T7 | S3 밝은그레이 | E2 측광 | B8 (그림자) | `raw_T7.mp4` | (랜덤) | (랜덤) | (랜덤) | (랜덤) | ⏳ |
| T8 | S1 다크브라운 | E3 저조도 | B8 (대조군) | `raw_T8.mp4` | (랜덤) | (랜덤) | (랜덤) | (랜덤) | ⏳ |
| T9 | S1 다크브라운 | E2 측광 | B8 (대조군 측광) | `raw_T9.mp4` | (랜덤) | (랜덤) | (랜덤) | (랜덤) | ⏳ |

**평가 단위: 9 take** — 각 take에서 A/B/C/D 토글하며 즉석 비교 (클립 분할/익명화 불필요).
나중 비교용 영상을 남기려면 녹화 후 helper `trim` 사용 (선택).

상태 범례: ⏳ 대기 / 👁️ 토글 평가 완료

### 매트릭스 의도

- **B1 (블렌드)**: T1~T5 (5 SKU × 형광)에서 Normal(A·B) vs CRL(C·D) 비교. S5 화이트그래픽이 CRL 존재 이유 핵심 셀.
  - 홍채 톤(§5.1 짙음/중간/밝음)은 SKU가 아닌 촬영 대상자 눈 색 속성 → 매트릭스 축이 아닌 **촬영 대상 다양화**로 커버 (`recording_guide.md` 참조).
- **B8 (sclera)**: §5.3 2 SKU × 3 조명 대칭 완성 — 밝은그레이 {E1(T3)/E2(T7)/E3(T6)} + 다크브라운 {E1(T1)/E2(T9)/E3(T8)}. color-veto(A·C) vs luma-only(B·D) 비교. 저조도(E3)가 luma-only 위험 영역, 측광(E2)이 그림자 영역.

---

## 평가 방식 (1인 실시간 토글 체감)

> 1인 개발 체제 → 평가자 3명 블라인드 다수결 비현실적. **본인 실기기 토글 체감**으로 대체.
> 근거: 메모리 `feedback_qualitative_device_judgment`. 상세: `ratings_legend.md`.

- **평가자**: 본인 1인.
- **방법**: take별로 A↔C / B↔D 토글(B1 블렌드), A↔B / C↔D 토글(B8 sclera) → `ratings_template.csv` take당 1행(winner + 메모).
- **편향 주의**: 어느 게 CRL/color인지 알고 보므로 선입견 경계. 애매하면 `tie`.

### (선택) self-blind 보조

판단이 정 애매한 take만 녹화 → `trim` → `randomize`로 익명 클립을 만들어 정답 모르고 재평가.
`recording_guide.md` §블라인드 무작위화 참조. `_truth.csv`는 봉인 후 평가 끝나고 매칭.

---

## 진행 단계

- [ ] **준비** — APK 설치, SKU 6종 확보, **native active 로그 확인 (F-01 가드)**
- [ ] **토글 평가** — 9 take 각각 실기기에서 A/B/C/D 토글 비교 → `ratings_template.csv` 9행 채움
- [ ] **(선택) self-blind 검증** — 애매한 take만 익명 클립 재평가
- [ ] **집계** — B1/B8 경향 정성 판정 (`ratings_legend.md`)
- [ ] **Phase C 진입** — `report.md` 작성 + 셰이더 최종 수식 정리 + 99 §1.1 D6 / §1.2 C4 갱신
