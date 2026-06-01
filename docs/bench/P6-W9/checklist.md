# P6-W9 통합 회귀 체크리스트 — 실기기 18 시나리오

**작성**: 2026-06-01
**범위**: 살아남은 W (W1/W2/W5/W6/W7)만. W3/W4/W8 환경 반사·Pupil 시나리오 폐기.
**평가 방식**: 1인 실시간 토글 체감 (메모리 `solo-dev-bench-method` / `qualitative-device-judgment`).
**소요**: SKU 6 × tier 3 = 18 시나리오. 시나리오당 평균 8분 → 약 2.4시간.

---

## 1. 기기 매핑

| Tier | 권장 기기 | 권장 GPU |
|---|---|---|
| HIGH | Galaxy S23+ / Pixel 8+ | Adreno 740+ / Mali-G715 |
| MID | Galaxy A 시리즈 / 보급형 | Adreno 630~650 / Mali-G76 |
| LOW | 구형 / 저사양 | Adreno 530 이하 |

> 한 tier에서만 검증 시: HIGH 우선 (메모리 `solo-dev-bench-method`). MID/LOW는 회귀 확인용.

## 2. SKU 6종 (P6-W0 §1.5)

1. **클라셋_돌 초코** (짙은 불투명, color veto 테스트)
2. **클라셋_런웨이 그레이** (밝은 톤, 흰자 빛남 후보)
3. **오(OH)_베이글** (중간 자연)
4. **엔비_퍼퓸 글로우** (쿨톤, 형광 luma-only 검증)
5. **엔비_샤모 브라운** (그래픽 outline, `prefers_graphic_outline=true` 적용 확인)
6. **클라셋_누드 애쉬 로제** (웜톤)

## 3. 시나리오당 확인 항목 (5축)

각 SKU × tier 조합에서 **5축**만 평가. PASS/FAIL/NOTE.

| 축 | 평가 내용 | 토글 위치 | 통과 기준 |
|---|---|---|---|
| **A. 양안 정상** | 좌우 대칭, 좌우 독립 렌더링, 한쪽 깜빡임 시 한쪽만 사라짐 | 카메라 ON | 좌우 동시 시 동일 렌즈, 한쪽 블링크 시 한쪽만 ramp |
| **B. 블링크 ramp (W6)** | down 60ms / up 100~120ms 자연스러움 | 블링크 자연 발생 | 갑작스러운 ON/OFF 없음, ramp 인지 자연 |
| **C. Sclera veto (W5 B8)** | 흰자 번짐 없음 (luma-only 모드, 토글 B) | 데모 UI A/B/C/D 토글 (B=Normal+luma 권장) | 흰자에 렌즈 색 번짐 없음, 외곽 깎임 없음 |
| **D. 블렌드 + 디테일 (W2/W6)** | TintLinearV2(ID=5) 자연도 + 디테일 재주입 노이즈 없음 | 데모 UI blendMode ID 5 기본 | 짙은 SKU에서 자연 tint, 저조도에서 노이즈 증폭 없음 |
| **E. 림발 (에셋 baked만)** | 에셋에 baked된 림발 디자인이 SDK 처리 없이 자연 렌더 | 자동 (셰이더 림발 영구 제거됨) | 림발 색·강도가 에셋 디자인대로 보임, SDK가 덧칠하지 않음 |

> ✅ W4 환경 반사 / W8 Pupil 항목은 평가 제외 (Phase 6 이월).

## 4. 시나리오 시트 양식

`ratings_sheet.md` 또는 `ratings_template.csv` 채워 진행:

```
SKU=클라셋_돌 초코, Tier=HIGH (S23+)
A. 양안 정상      : PASS / FAIL / NOTE: ___
B. 블링크 ramp    : PASS / FAIL / NOTE: ___
C. Sclera veto    : PASS / FAIL / NOTE: ___
D. 블렌드+디테일  : PASS / FAIL / NOTE: ___
E. 림발 (에셋)    : PASS / FAIL / NOTE: ___
FPS 평균          : ___ fps
프레임시간 평균   : ___ ms
메모리 (10분 후)  : 시작 ___ MB → 종료 ___ MB (delta ___ MB)
```

## 5. FPS / 프레임시간 측정

Android demo의 frame counter (있으면 재사용, 없으면 logcat에 다음 패턴 출력):
- 60프레임 평균 FPS
- 평균 프레임시간 (ms) — `Choreographer.FrameCallback` 또는 기존 counter

### logcat 필터 명령

```bash
# 기기 시리얼 확인
adb devices

# FPS 로그 추출 (실측 source 연결 후 수치 변동 확인용)
adb logcat -v time -s IrisSDK GpuRenderActivity CameraGLView | grep -E "fps|frame|avg"
```

### 메모리 측정 명령

```bash
# 초기
adb shell dumpsys meminfo com.irislenssdk.demo | head -20

# 10분 후 (앱은 계속 ON 상태)
adb shell dumpsys meminfo com.irislenssdk.demo | head -20
```

## 6. 회귀 신호 — FAIL 시 즉시 보고

다음 중 하나라도 발견되면 머지 전 디버깅 필수:

- 좌우 비대칭 렌더링 (예: 한쪽만 렌즈 적용)
- 블링크 시 렌즈가 갑작스럽게 사라지거나 잔상 남김
- 흰자에 렌즈 색이 번지거나 외곽이 누에 살이 안쪽으로 깎임
- 짙은 SKU에서 화이트 톤 과증폭 (TintLinearV2 알고리즘 본질 한계, 후속 W로 분리되어 있지만 정도가 심하면 보고)
- 림발이 에셋 디자인과 다르게 보임 (셰이더가 림발 darkening을 다시 입히는 회귀)
- FPS 30 미만 (HIGH tier)
- 메모리 10분 후 +20MB 이상 (CLAUDE.md 100MB 한도 대비)

## 7. 결과 정리

`docs/bench/P6-W9/report.md`에 다음 형식으로:

```markdown
# P6-W9 통합 회귀 결과

**일시**: 2026-06-XX
**기기**: ___
**SDK 빌드**: feature/P6-Works @ <SHA>

## 시나리오 결과 (18)
| SKU | Tier | A | B | C | D | E | FPS | 메모 |
|---|---|---|---|---|---|---|---|---|
| 클라셋_돌 초코 | HIGH | ✅ | ✅ | ✅ | ✅ | ✅ | 60 | - |
...

## FAIL 항목
- (없음 / 또는 항목 + 후속 처리)

## 머지 판정
- ✅ 머지 가능 / ⚠️ 보류 (사유)
```

## 8. 보고 후 다음 단계

결과 → `docs/workPaper/P6-W9_integration_report.md`로 합쳐서 머지 PR 본문에 포함.
