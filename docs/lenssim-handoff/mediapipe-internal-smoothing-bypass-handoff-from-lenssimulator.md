# MediaPipe VIDEO 모드 내부 스무딩 우회(IMAGE 모드) 인사이트 — LensSimulator → IrisLensSDK 핸드오프

작성: 2026-07-06 (LensSimulator 세션)
근거: LensSimulator 실기기(Galaxy S23+) A/B — `FaceTracker.kt` IMAGE 모드 토글 (커밋 b8349eb 계열)
선행 문서: `tracking-latency-handoff-from-lenssimulator.md` (프레임 동기 — **이 문서는 그 후속**)

> 이 문서는 **구현 코드 이식이 아니라 진단·교훈 인사이트**다. IrisLensSDK는 C++ 코어에서
> MediaPipe 그래프를 직접 소유하므로, LensSim(Java Tasks API)보다 **더 좋은 해법**(스무딩
> 계산기만 제거)을 쓸 수 있다 — §4 참조.

---

## TL;DR

프레임 동기(선행 문서)로 "픽셀-랜드마크 프레임 불일치"를 제거한 **뒤에도** 렌즈가 눈 중심을
늦게 따라오는 잔여 온셋 지연이 남았다. 원인은 파이프라인이 아니라 **MediaPipe FaceLandmarker
VIDEO 모드가 그래프 내부에서 돌리는 자체 랜드마크 스무딩**이었다. `RunningMode.IMAGE`로
전환해 내부 스무딩을 우회하자 실기기에서 "렌즈가 눈에 더 잘 붙는다"로 체감 개선 확인
(2026-07-06, 1차 주관 평가). **자체 One-Euro 스태빌라이저를 가진 파이프라인에서 MP 내부
스무딩은 이중 스무딩 = 이중 지연이다.**

---

## 1. 증상 (프레임 동기 이후의 잔여 지연)

- 프레임 동기 렌더링으로 화면 프레임 == 랜드마크 프레임인데도, 빠른 시선 이동(saccade)에서
  렌즈가 홍채 중심을 1~2프레임 늦게 따라온다.
- 디버그 오버레이 기준: **무필터 raw 랜드마크(흰 점)조차** 표시 프레임의 홍채보다 늦다 —
  즉 자체 필터(One-Euro) 이전, MediaPipe 출력 자체가 이미 시간적으로 스무딩돼 있다.

## 2. 근본 원인 — VIDEO/LIVE_STREAM 모드의 내장 랜드마크 스무딩

- MediaPipe face landmarker 그래프는 VIDEO·LIVE_STREAM 모드에서 **landmarks smoothing
  계산기**(One-Euro 계열, 객체 스케일 정규화)를 자동으로 태운다. IMAGE 모드에는 없다.
- 이 스무딩은 Tasks Java/Swift API에 **끄는 옵션이 노출돼 있지 않다** — LensSim이 IMAGE
  모드로 통째 전환한 이유.
- 자체 후처리 필터(One-Euro)를 가진 파이프라인에서는 **스무딩이 2중으로 걸려** 온셋 지연이
  누적된다. 지터 억제는 한 곳에서만 하면 된다 — 튜닝 가능한 자기 필터가 남는 쪽이 맞다.

## 3. LensSim의 해법과 실측 트레이드오프 (IMAGE 모드 A/B)

- `RunningMode.IMAGE` + `detect()` (동기) — 내부 스무딩·ROI 추적 없음. A/B 토글로 실기기 비교.
- **개선**: 시선 이동 시 렌즈가 눈에 더 잘 붙음 (체감, 1차 평가).
- **비용/함정 (이식 시 그대로 적용됨)**:
  1. **프레임당 풀 검출** — VIDEO 모드의 ROI 추적 재사용이 사라져 추론 비용 증가 가능
     (정량 미측정 — S23+ GPU에서 체감 프레임 저하는 없었음).
  2. **지터 증가** — 내부 스무딩이 걸러주던 랜드마크 노이즈가 raw로 나온다. 자체 필터가
     감당해야 한다 (LensSim은 near-raw One-Euro 3.0/0.3 유지로 1차 통과).
     지터는 **클리핑 마스크 경계** 같은 하류 소비자에도 전파된다 — LensSim은 윤곽 16점에도
     기저와 동일 상수 One-Euro를 걸어 위상을 정렬해 뒀다.
  3. **타임스탬프 계약** — IMAGE 모드 result의 timestamp는 무의미(0)다. 추론 시간 계측·필터
     dt·스냅샷 타임스탬프는 **제출 시각을 별도로 운반**해야 한다 (LensSim은 onResult 시그니처에
     명시 인자로 추가).
  4. running mode는 생성 시점 고정 — 전환 시 landmarker **재생성** 필요 (추론 스레드에서,
     필터 리셋 동반).

## 4. IrisLensSDK에의 적용 — 권고 (LensSim보다 좋은 선택지가 있다)

IrisLens는 C++에서 그래프를 직접 소유하므로 IMAGE 모드 점프가 아니라 **외과적 제거**가 가능:

1. **1순위 권고: 그래프에서 landmarks smoothing 계산기만 제거/우회하고 VIDEO식 ROI 추적은
   유지** — 온셋 지연 제거 + 프레임당 풀 검출 비용 회피, 양쪽 다 갖는다. Java Tasks API로는
   불가능한 옵션이라 LensSim은 검증 못 한 경로 — IrisLens에서 검증 가치가 가장 크다.
   (확인 지점: face landmarker 그래프의 smoothing 관련 calculator/옵션 — IrisLens의 TFLite
   기반 파이프라인이 해당 계산기를 포함하는지 먼저 grep.)
2. 이미 포함돼 있지 않다면(자체 그래프가 스무딩 없이 raw를 뽑는 구조라면) 이 인사이트는
   "이중 스무딩 여부 점검" 체크리스트로 소화하면 된다 — **temporal_stabilizer(자체 One-Euro)
   앞단에 또 다른 시간 필터가 없는지**가 점검 항목.
3. p33 추적 A/B 트랙과의 접점: 추적 백엔드 비교 시 **후보 간 "내장 스무딩 유무"를 통제 변수로
   맞춰야** 지연 비교가 공정하다 — 스무딩 포함 백엔드는 온셋 지연에서 구조적으로 불리하다.
4. 검증 방법 재사용: raw 랜드마크(자체 필터 이전)를 표시 프레임에 오버레이 → 표시 프레임의
   홍채와 비교. **raw가 늦으면 상류(내장 스무딩/파이프라인), raw는 붙는데 필터 출력만 늦으면
   자기 필터** — 선행 문서의 진단법과 동일한 이분법이 여기서도 유효했다.

## 5. 검증 상태 (정직한 라벨)

| 항목 | 상태 |
|---|---|
| 체감 개선 (S23+, 렌즈 밀착) | ✅ 1차 확인 (사용자 주관 평가, 2026-07-06) |
| 추론 비용 증가 정량 | ⏳ 미측정 (fps/추론 ms 오버레이로 측정 예정) |
| 지터 증가 정량 | ⏳ 미측정 (자체 One-Euro가 1차 흡수 중) |
| 기본 모드 승격 여부 | ⏳ 미결 — LensSim은 A/B 토글 유지 상태 |

LensSim 쪽 후속 결과(승격/기각, 계측치)는 이 문서를 갱신하지 않고 LensSim ADR에 기록된다 —
필요 시 LensSim `docs/decisions/0008-lens-fit-stability-pass.md` 이후 커밋 로그를 참조.
