# P6-W5 B1+B8 벤치 — 촬영 가이드

> **출처**: W5 §5.6 (4조합 동시 캡처 절차)

---

## 사전 준비

1. **APK 빌드/설치 확인** — `feature/P6-Works` HEAD에서 빌드된 데모 앱:
   ```bash
   cd android && ./gradlew :demo-app:assembleDebug
   adb install -r -d build/modules/demo-app/outputs/apk/debug/demo-app-debug-*.apk
   ```

2. **SKU 설정** — 앱 실행 후 해당 take의 SKU 렌즈 선택 (`checklist.md` Take 표 참조).

3. **공통 설정 (변수 통제)**:
   - **Sclera Protect: ON** (화면 하단 렌즈 탭 'Sclera' 버튼이 초록 ON)
   - 투명도/크기/경계: 기본값 유지 (slider 조정 금지)
   - 밝기(maxDetail): native 경로는 §5.10 확정값 **1.25 하드코딩**. 화면 밝기 slider는 Kotlin 폴백 셰이더 전용이라 native 벤치에 영향 없음 — 조정 불필요.

4. **native lens 활성 확인 (F-01 가드, 필수)**:
   ```bash
   adb logcat -c && adb logcat | grep -E "GPULensRenderer active|fallback to Kotlin"
   ```
   - `SDK C++ GPULensRenderer active` → 정상 (촬영 진행).
   - `fallback to Kotlin shader` → native 비활성. 폴백 셰이더는 `uScleraVetoMode`/`blendColorReplaceLinear`가 없어 **4조합이 동일 렌더되어 B1/B8 무효**. 앱 재시작/재초기화로 native 활성 후 촬영. 폴백 상태로 찍은 take는 폐기.

5. **디바이스 잠금 해제 + 화면 자동 잠금 비활성**.

---

## 촬영 절차 (take 1개당 약 4분)

### 핵심 규칙 (§5.6)

- **같은 take 12초 녹화 → 후편집으로 5초씩 4클립 trim**
- **녹화 중 화면 우상단 A→B→C→D 버튼 연속 토글** (4조합)
  - 같은 take 내 4조합 캡처로 동작/표정 변수 배제
- **블링크 1~2회 포함**
- **30fps 고정**

### 버튼 ↔ 조합 (평가자에게 비공개)

| 버튼 | 블렌드 | sclera |
|------|--------|--------|
| A | Normal | color-veto |
| B | Normal | luma-only |
| C | CRL | color-veto |
| D | CRL | luma-only |

> 버튼 누르면 Toast로 "Bench A/B/C/D"만 표시 (정답 비노출). logcat에는 정답 desc 기록.

### Take 1회 절차 (예: T1)

1. SKU/조명 세팅 (checklist Take 표).
2. 디바이스 정면 거치 (얼굴 정면, 화면이 잘 보이게).
3. `adb shell screenrecord /sdcard/raw_T1.mp4 --time-limit 14` (14초 = 2초 여유 + 12초 본)
4. 녹화 시작 후 **즉시 자세 잡기** (~1초).
5. **버튼 A 누름** → "Bench A" Toast 확인 → **0~3초**: 정자세 + 블링크 1회
6. **3초 시점**: 버튼 B → **3~6초**: 정자세 + 블링크 1회
7. **6초 시점**: 버튼 C → **6~9초**: 정자세 + 블링크 1회
8. **9초 시점**: 버튼 D → **9~12초**: 정자세 + 블링크 1회
9. 녹화 자동 종료.
10. `adb pull /sdcard/raw_T1.mp4 ./docs/bench/P6-W5/raw/`

### 9 take 반복

`checklist.md` Take 표의 T1~T9 (SKU × 조명) 각각 위 절차 반복.
- T1~T5: E1 형광 (SKU만 교체)
- T6: S3 밝은그레이 + E3 저조도 (조명 끄고 야간등)
- T7: S3 밝은그레이 + E2 측광 (창가 이동)
- T8: S1 다크브라운 + E3 저조도
- T9: S1 다크브라운 + E2 측광 (B8 2×3 대칭 — 측광 대조군)

> 홍채 톤(§5.1 짙음/중간/밝음)은 가능하면 **촬영 대상자를 톤별로 다양화** (한 명만 찍지 말 것). SKU 매트릭스 축이 아니라 모델 다양성으로 커버.

**총 9회 녹화 → `docs/bench/P6-W5/raw/raw_T*.mp4` 9개 파일**

---

## 후편집 (FFmpeg 기반)

각 raw 파일을 4구간으로 자르고 5초로 trim:

```bash
mkdir -p docs/bench/P6-W5/clips
RAW=docs/bench/P6-W5/raw/raw_T1.mp4

# A 구간 (0~3초) — 0.5~5.5초 (시작 0.5초 정리 마진은 3초 구간이라 trim 짧게)
ffmpeg -ss 0.5 -i "$RAW" -t 2.5 -c copy docs/bench/P6-W5/clips/_T1_A.mp4
# B 구간 (3~6초) — 3.5~6.0초
ffmpeg -ss 3.5 -i "$RAW" -t 2.5 -c copy docs/bench/P6-W5/clips/_T1_B.mp4
# C 구간 (6~9초) — 6.5~9.0초
ffmpeg -ss 6.5 -i "$RAW" -t 2.5 -c copy docs/bench/P6-W5/clips/_T1_C.mp4
# D 구간 (9~12초) — 9.5~12.0초
ffmpeg -ss 9.5 -i "$RAW" -t 2.5 -c copy docs/bench/P6-W5/clips/_T1_D.mp4
```

→ 36개 클립 (`_T*_[ABCD].mp4`).

자동화: `scripts/p6w5_bench_helper.sh trim` 사용.

---

## 블라인드 파일명 무작위화

평가자가 어느 조합인지 모르게:

```bash
bash scripts/p6w5_bench_helper.sh randomize
```

→ `clips/clip_01.mp4` ~ `clips/clip_36.mp4` + `docs/bench/P6-W5/_truth.csv` (gitignore/봉인).

`checklist.md`의 "클립 ID ↔ 진실 정답표"를 `_truth.csv`로 채움 (평가 끝나기 전 비공개).

---

## 주의 사항

- **디바이스 1대 통일** — 변수 통제.
- **버튼 토글 타이밍 정확** — 3/6/9초. 전환 직후 frame이 평가에 섞이지 않게 trim 마진 확보.
- **Sclera Protect ON 유지** — 4조합 전부 veto 분기가 살아야 B8 비교 의미 있음. OFF면 A=B, C=D가 됨.
- **저조도(E3) take에서 노출 자동조정 주의** — 카메라가 밝기 자동 올리면 채도 위험 영역 재현 안 됨. 가능하면 고정.

---

## adb 자동화 (`scripts/p6w5_bench_helper.sh`)

```bash
bash scripts/p6w5_bench_helper.sh launch        # 디바이스 wake + 앱 시작
bash scripts/p6w5_bench_helper.sh record T1     # 14초 녹화
bash scripts/p6w5_bench_helper.sh pull T1       # 녹화 종료 후 pull
bash scripts/p6w5_bench_helper.sh trim          # 36 클립 후편집 (raw → A/B/C/D)
bash scripts/p6w5_bench_helper.sh randomize     # 블라인드 ID + 정답표 생성
```
