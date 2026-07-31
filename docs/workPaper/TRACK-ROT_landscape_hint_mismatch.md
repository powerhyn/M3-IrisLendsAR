# TRACK-ROT — 가로 화면 얼굴 추적 불안정: MediaPipe 회전 힌트 불일치

> 상태: **✅ 원인 확정 · ✅ 수정 완료 · ✅ 판별 실험으로 공식 확정(screenRot 90·270 두 지점 블라인드) · ✅ 적대 검토 반영** / ⏳ screenRot 180만 외삽
> 브랜치: `fix/ring-fbo-transpose` (커밋 `3bcc2a8` → 검토 반영 `7896095`, 리소스 `9b0b44d`) — **미머지·미푸시**
> ⚠️ 이 브랜치에는 선명도 트랙(SHARP, 4커밋)이 함께 들어 있다. 성격이 다르므로 분리 여부 미결.
> 실측: 2026-07-29~30 / SM-X920 (Tab S10 Ultra), 대조 SM-S916N (Galaxy S23+)
> 관련: `SHARP_ring_fbo_transpose.md` (별개 트랙 — 링 FBO 치수 문제. 서로 독립)

---

## 0. 세 줄 요약

1. **증상**: 태블릿 **가로**에서 고개를 조금만 돌려도 얼굴 메시가 일그러진다. 같은 기기 **세로**, 폰 세로는 안정적.
2. **원인**: `targetRotation = ROTATION_0` 핀 때문에 `imageInfo.rotationDegrees`가 **기기 자세와 무관한 상수**(270)다. 이 값 하나가 ①MediaPipe 검출 힌트와 ②랜드마크→렌더 매핑 두 역할을 겸하는데, 기기를 90° 돌려 들면 ①만 틀려서 **MediaPipe가 옆으로 누운 얼굴을 본다**.
3. **수정**: 힌트에만 화면 회전을 더한다 — `hint = (bufferRotation + screenRotation) % 360`. **좌표 변환 경로는 한 줄도 안 건드린다.** `screenRotation == 0`이면 현행과 동일이라 폰·태블릿 세로는 구조적 무회귀.

---

## 1. 왜 화면은 멀쩡한데 추적만 틀어지나

렌더 경로는 최종 blit에서 `rotK`(= `screenRotationQuadrant()`)로 화면 방향을 보정한다. 그래서 **영상은 정상으로 보인다.** 반면 분석 경로는 `imageInfo.rotationDegrees`를 그대로 MediaPipe 힌트로 쓰는데, 이 값에는 화면 회전이 반영되지 않는다.

```
FaceTracker.kt:243   val rotation = imageProxy.imageInfo.rotationDegrees   // 항상 270
FaceTracker.kt:~544  ImageProcessingOptions.setRotationDegrees(rotation)
```

| | 물리 자세 | 힌트 | MediaPipe가 보는 얼굴 |
|---|---|---|---|
| 폰 | 세로(= natural) | 270 | 똑바로 ✅ |
| 태블릿 | 세로 | 270 | 똑바로 ✅ |
| 태블릿 | **가로**(natural에서 90°) | 270 | **90° 누움** ❌ |

FaceLandmarker의 검출 단계(BlazeFace)는 정립 얼굴 위주로 학습돼 있어, 누운 얼굴은 검출은 되지만 랜드마크가 불안정해진다. 고개를 돌릴 때 특히 심하다.

**핵심 구조**: 값 하나가 두 역할을 겸한다.

| 역할 | 필요한 기준 | 현행 |
|---|---|---|
| ① MediaPipe 검출 힌트 | **세상 기준 정립** | ❌ natural 기준 |
| ② 랜드마크 → 링/렌더 공간 매핑 | natural 기준 | ✅ 맞음 |

---

## 2. 전제 검증 — 힌트와 좌표는 분리 가능한가

수정이 성립하려면 "힌트를 바꿔도 출력 랜드마크 좌표계는 안 바뀐다"가 참이어야 한다. `FaceTracker.kt:58-60`의 계약 주석이 그렇게 주장한다("전달 좌표는 센서(원본·미회전·비미러·무필터) 정규화 공간 그대로").

**실측 확인**: 힌트를 0/90/180/270 네 값으로 바꿔가며 캡처한 결과 **마커가 모두 얼굴 위에 남았다.** 좌표계가 힌트를 따라 회전했다면 마커가 얼굴 밖으로 튀었을 것이다.

→ **좌표 변환 경로(`TasksToIrisResult` / `CoordMapper` / 렌즈 좌표)를 손대지 않아도 된다.** 만약 튀었다면 연쇄 수정이 필요한 훨씬 큰 작업이 됐다.

---

## 3. 판정 — 블라인드 A/B

### 3-1. 왜 블라인드가 필요했나

후보 간 차이가 미세하고, "90이 예상 정답"이라는 사전 정보가 이미 공유된 상태였다. 기대 효과를 배제하려면 매핑을 가려야 했다.

**다만 전환 자체는 숨기지 않았다.** 실사용자 관찰: *"보고 있으면 익숙해지는 거랑 바뀌는 순간이 체감이 제일 크다."* 그래서 매핑(A~D ↔ 오프셋)만 무작위로 가리고, 전환은 **토스트 + HUD 라벨(`◀ 팔 A ▶`)** 로 확실히 인지시켰다.

### 3-2. 결과

4팔 순환 → B↔C 재확인 → **D 판정**. 공개 결과:

| 팔 | 오프셋 | 힌트 | 판정 |
|---|---|---|---|
| A | 270 | 180 | |
| B | 0 | 270 | ← **현행 기본값**, 선택되지 않음 |
| C | 180 | 90 | |
| **D** | **90** | **0** | ✅ **확실히 안정적** |

D = 오프셋 90 = `screenRotation`(90)과 일치. 사전 예측 및 물리 모델과 부합한다 — 이 태블릿은 전면 카메라가 **긴 변**에 있어, 가로로 들면 센서 원본 래스터가 이미 세상 기준 정립이다. 그래서 회전이 필요 없는 것(힌트 0)이 정답이다.

### 3-2b. 판별 실험 — 가설 A vs B (적대 검토가 지적한 설계 결함 해소)

**지적**: 3-2의 승리 팔 D는 오프셋 90 = **힌트 0**, 즉 '회전을 아예 안 줌'이다. `screenRot=90`에서는
두 가설의 예측이 **완전히 같다**:

| | 가설 A (`offset = screenRotation`) | 가설 B ("힌트 0이 늘 최선") |
|---|---|---|
| screenRot 90 | 오프셋 90 | 오프셋 90 | ← 3-2가 잰 지점. **구분 불가** |
| screenRot 270 | 오프셋 **270** | 오프셋 **90** | ← 갈림 |

가설 B가 실재할 이유도 있었다 — 태블릿 분석 버퍼가 512×288로 작아 MediaPipe 회전 전처리의
리샘플이 품질을 깎는 경로일 수 있다. B가 참이면 이 수정은 180/270에서 **오히려 나쁘게** 만든다.

**판별 실험** (2026-07-30, SM-X920 **역가로**, `Screen rotation set: 270`, 매핑 재셔플):

| 팔 | 오프셋 | 힌트 | 판정 |
|---|---|---|---|
| **D** | **270** | **180** | ✅ **제일 안정** |
| A | 180 | 90 | 2위 |
| B | **90** | **0** | ❌ 완전히 망가짐 |
| C | 0 | 270 | ❌ 완전히 망가짐 |

→ **가설 A 확정, 가설 B 기각.** B가 참이었다면 B팔(힌트 0)이 이겼어야 하는데 최악군이었다.
3-2보다 대비가 훨씬 크다("확실히 안정적" → "둘은 완전히 망가짐") — 회전 힌트가 추적 품질을
지배한다는 것이 확증된다. 셔플도 3-2와 달랐다(A=270/B=0/C=180/D=90 → A=180/B=90/C=0/D=270)
— D 연속 승리는 우연이고 블라인드 절차 자체도 유효했다.

**공식 확정 근거 지점**: screenRot 90(3-2) + 270(여기) 두 곳 블라인드 A/B, screenRot 0은
현상유지 관찰(원래 안정). 회전은 군(group)이라 가법성으로 180이 따라온다.

### 3-3. ⚠️ 폐기된 정량 지표 (재시도 금지)

정지 상태에서 마커 무리의 프레임 간 흔들림을 재는 지표를 만들었으나 **판별력이 없었다.**

| 오프셋 | 퍼짐 흔들림 (3라운드) | 중앙값 |
|---|---|---|
| 0 | 1.96 / 0.72 / 0.70 | 0.72 |
| 90 | 2.18 / 0.71 / 4.62 | 2.18 |
| 180 | 0.93 / 1.17 / 2.06 | 1.17 |
| 270 | 5.48 / 2.05 / 0.45 | 2.05 |

**같은 오프셋 안의 편차(0.45~5.48)가 오프셋 간 차이보다 크다.** 1라운드만 보고 "오프셋 180 최적"이라 판단했다가 순서를 무작위화한 3라운드에서 뒤집혔다(그때는 오프셋 0이 최고로 나왔다).

원인: **증상은 고개를 돌릴 때 나타나는데 이 지표는 정지 상태를 잰다.** 애초에 다른 것을 측정하고 있었다. 이 축은 육안 판정이 정답이다.

---

## 4. 수정 내용

```kotlin
// GpuRenderActivity.pushScreenRotation() — 값만 갱신(자동 유도)
if (detRotAuto && detRotOffset != deg) { detRotOffset = deg }

// GpuRenderActivity.processFrameTasks() — 분석 스레드가 매 프레임 pull (멱등)
ensureFaceTracker().also { it.setDetectionRotationOffset(detRotOffset) }.analyze(imageProxy)

// FaceTracker.processingOptions() — 값 생성 후 (값, 키) 순으로 커밋
val offset = detectionRotationOffset          // volatile 1회 읽기
val hint = ((rotation + offset) % 360 + 360) % 360
val opts = ImageProcessingOptions.builder().setRotationDegrees(hint).build()
cachedProcessingOptions = opts; cachedRotation = hint
```

> push 가 아니라 **pull** 인 이유: `faceTracker` 는 non-volatile 이고 '분석 스레드 전용' 규약이라
> 메인 스레드에서 밀어 넣으면 트래커 생성과 인터리브될 때 유실될 수 있고, 갱신이 1회성이라
> 재시도 경로가 없었다(적대 검토 지적). pull 은 volatile int 비교뿐이라 비용이 사실상 0이다.

배선 확인 (SM-X920, `user_rotation` 스윕):

| user_rotation | screenRot | 오프셋(자동) | 힌트 |
|---|---|---|---|
| 0 (세로) | 0 | 0 | **270** ← 현행과 동일 |
| 2 (역세로) | 180 | 180 | 90 |
| 3 (역가로) | 270 | 270 | 180 |
| 1 (가로) | 90 | 90 | **0** ← 판정값 |

---

## 5. 미검증 / 남은 일

1. **~~screenRotation 270~~ 완료** (§3-2b). 남은 것은 **180(거꾸로 세로)뿐**이며 실사용 빈도가 낮고, 90·270 두 지점이 확정돼 가법성으로 따라오므로 위험이 낮다.
1-b. **자동회전 잠금 시 미동작** — `currentScreenRotationDeg()`는 기기 물리 자세가 아니라 **윈도우 회전**을 읽는다. 사용자가 자동회전을 잠그면 기기를 눕혀도 `display.rotation`이 0에 머물러 이 수정이 engage 하지 않는다(폰은 `allow_landscape=false`라 항상 이 상태). 근본 해결은 `OrientationEventListener` 기반 기기 자세지만 이번 범위 밖이다.
2. **`rotSignInverted` 커플링** — GL blit에는 적용되는데 이 힌트에는 미적용이다. 현재 정방향 확정이라 무영향이나, 부호가 뒤집히는 기기가 나오면 두 경로가 갈린다.
3. **태블릿 분석 스트림 512×288** — 폰은 640×360(둘 다 960×540 요청, 태블릿이 더 낮게 폴백). 면적 1.56배 차이로 랜드마크 정밀도에 직접 불리하다. 회전과 무관한 **별개 요인**이며 회전 건이 해결된 지금 다음 후보다.
4. **적대 검토(2026-07-30, 4관점 17에이전트) 반영 완료** — 커밋 `7896095`:
   - `detRotAuto` 래치(진단 1회 사용 시 자동 유도 영구 사망, REVEAL도 복구 안 함 → **남은 검증을 오염시킬 뻔함**)
   - `setDetectionRotationOffset` 90배수 미검증 → MediaPipe 예외가 추론 오류로 오진되어 GPU→CPU 폴백 유발
   - `processingOptions` 캐시 키를 값 생성 전 커밋 → `build()` 실패 시 조용히 이전 힌트로 계속 검출
   - `faceTracker` non-volatile 경합 → push를 pull로 전환(매 프레임 멱등)
   - 커밋 누락 리소스 3종(`bools.xml` 등) 추가 — 커밋 `9b0b44d`, **깨끗한 체크아웃이 빌드 안 되던 blocker**
5. **~~진단 표면 제거~~ 완료** (2026-07-31) — `SET_DET_ROT` / `DET_BLIND_*` 브로드캐스트, 블라인드 상태(`blindOn`/`blindArms`/`blindIdx`/`blindLabel`), `applyDetRot`, `detRotAuto` 래치 제거. HUD 는 `det:$detRotOffset` 로 단순화(팔 라벨·`(auto)` 접미사 삭제). `pushScreenRotation` 조건은 `if (detRotOffset != deg)` 로 — **자동 유도가 이제 무조건 동작**한다(래치 제거의 의도된 결과).
   - 실기기 재확인: 태블릿 가로 `검출 힌트 회전 → 0 (버퍼 270 + 오프셋 90)`, 폰 세로 `→ 270 (버퍼 270 + 오프셋 0)` — 제거 전과 동일. 폰의 오프셋 0 은 §4 무회귀 불변식(`screenRotation==0` ⇒ 종전과 동일)이 지켜짐을 뜻한다.
6. **브랜치 분리 여부** — 이 커밋이 SHARP(선명도) 트랙과 한 브랜치에 섞여 있다. 미결.

---

## 6. 진단 사용법 (제거 전까지)

```bash
# 수동 오프셋 (자동 유도 해제됨)
adb shell am broadcast -a com.irislenssdk.demo.SET_DET_ROT --ei offset 90

# 블라인드 A/B — 매핑은 가리고 전환은 토스트+HUD로 인지
adb shell am broadcast -a com.irislenssdk.demo.DET_BLIND_START
adb shell am broadcast -a com.irislenssdk.demo.DET_BLIND_NEXT
adb shell am broadcast -a com.irislenssdk.demo.DET_BLIND_NEXT --es arm B   # 재확인
adb shell am broadcast -a com.irislenssdk.demo.DET_BLIND_REVEAL
```

> 반영 타이밍: 브로드캐스트 → `@Volatile` 대입(즉시) → 다음 분석 프레임(~33ms) → 랜드마크가 코어 stabilize를 타고 몇 프레임에 걸쳐 수렴. **"딱 바뀌는 순간"은 안 보이는 게 정상**이므로, 순간 비교가 아니라 각 상태의 정상 상태끼리 비교할 것.

---

## 변경 이력

| 일시 | 내용 |
|---|---|
| 2026-07-29 | 원인 확정(회전 힌트 불일치) + 블라인드 A/B로 오프셋=screenRotation 판정 + 자동 유도 반영(3bcc2a8). 정량 지표는 판별력 없음으로 폐기 |
| 2026-07-30 | 판별 실험(역가로 screenRot=270 블라인드)으로 **가설 A 확정·B 기각** → 공식 `offset=screenRotation` 확정. 적대 검토 4관점 반영(7896095) + 커밋 누락 리소스(9b0b44d) |
| 2026-07-31 | 진단 표면 제거(§5-5) + 실기기 재확인(태블릿 가로·폰 세로 모두 제거 전과 동일). SHARP 트랙과 함께 develop 머지 |
