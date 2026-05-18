# P6-W4 B2 벤치 — 촬영 가이드

> **출처**: W4 §5.11 (촬영 절차 표준화)

---

## 사전 준비

1. **APK 빌드/설치 확인** — `feature/P6-Works` HEAD에서 빌드된 데모 앱 설치되어 있어야:
   ```bash
   cd android && ./gradlew :demo-app:assembleDebug
   adb install -r -d build/modules/demo-app/outputs/apk/debug/demo-app-debug-*.apk
   ```

2. **SKU 설정** — 앱 실행 후:
   - 렌즈 SKU: **TintLinearV2 + iris_mat_B (고발광)** 선택 (§5.12 1차 단일 SKU)
   - 블렌드 모드: **LUMINANCE_TINT_LINEAR (ID=5, 화면 하단 '블렌드' 영역 확인)**
   - 투명도/크기/경계: 기본값 유지 (slider 조정 금지 — 변수 통제)

3. **env_map 에셋 확인** — `assets/env/env_default_256x128.png` 있어야 (placeholder 또는 정식 ACES tone-mapped):
   ```bash
   ls android/demo-app/src/main/assets/env/
   ```
   정식 ACES env_map으로 교체할 거면 촬영 전에 빌드/재설치.

4. **녹화 방식** — Android 스크린 레코딩 (디바이스 기본 또는 `adb shell screenrecord`).
5. **디바이스 잠금 해제 + 화면 자동 잠금 비활성** (촬영 중 꺼지지 않게).

---

## 촬영 절차 (환경 1개당 약 4분)

### 핵심 규칙 (§5.11)

- **같은 take 10초 녹화 → 후편집으로 5초 trim**
- **녹화 중 VOLUME_UP 키로 3 프로토타입 (OFF → EnvMap → Periphery) 연속 토글**
  - 동작 차이 배제 위해 같은 take 내 3 모드 연속 캡처 (Codex 제안)
- **블링크 1~2회 포함** (정면 동작 시)
- **30fps 고정**

### 환경 4종 × 동작 2종 = 8회 녹화

**E1 실내 형광 (사무실/카페)**

#### Take E1-M1 (정면 미세, 블링크 포함)
1. 디바이스 정면 거치 (얼굴 정면, 화면이 잘 보이게)
2. `adb shell screenrecord /sdcard/raw_E1_M1.mp4 --time-limit 12` (12초 = 2초 여유 + 10초 본)
3. 녹화 시작 후 **즉시 카메라 앞에 자세 잡기** (~1초)
4. **0~3초**: OFF 모드 — 정면 정자세 + 블링크 1회
5. **3초 시점**: VOLUME_UP 1회 → 화면에 "Reflection: EnvMap" Toast 확인
6. **3~6.5초**: EnvMap 모드 — 정면 정자세 + 블링크 1회
7. **6.5초 시점**: VOLUME_UP 1회 → "Reflection: Periphery"
8. **6.5~10초**: Periphery 모드 — 정면 정자세 + 블링크 1회
9. 녹화 자동 종료
10. **다시 OFF로 복귀**: VOLUME_UP 1회 (다음 take 전)
11. `adb pull /sdcard/raw_E1_M1.mp4 ./docs/bench/P6-W4/raw/`

#### Take E1-M2 (좌우 head turn ±15°)
1. `adb shell screenrecord /sdcard/raw_E1_M2.mp4 --time-limit 12`
2. **0~3초** OFF: 정면 → 좌 15° → 정면
3. **3초** VOLUME_UP → EnvMap
4. **3~6.5초** EnvMap: 정면 → 우 15° → 정면
5. **6.5초** VOLUME_UP → Periphery
6. **6.5~10초** Periphery: 정면 → 좌 15° → 정면
7. `adb pull /sdcard/raw_E1_M2.mp4 ./docs/bench/P6-W4/raw/`

**E2 창가 측광** (한쪽에서 자연광 들어오는 위치로 이동)
- Take E2-M1, Take E2-M2 (E1과 동일 절차)

**E3 야간 실내** (어두운 노란 조명, 형광등 끄고 야간등만)
- Take E3-M1, Take E3-M2

**E4 실외 낮** (직접 자연광)
- Take E4-M1, Take E4-M2

**총 8회 녹화 → `docs/bench/P6-W4/raw/raw_E*_M*.mp4` 8개 파일**

---

## 후편집 (FFmpeg 기반)

각 raw 파일을 3구간으로 자르고 5초로 trim:

```bash
# E1-M1 예시
mkdir -p docs/bench/P6-W4/clips
RAW=docs/bench/P6-W4/raw/raw_E1_M1.mp4

# OFF 구간 (0~3초) — 0.5~5.5초 발췌 (시작 0.5초 정리 마진)
ffmpeg -ss 0.5 -i "$RAW" -t 5 -c copy docs/bench/P6-W4/clips/_E1_M1_P0.mp4

# EnvMap 구간 (3~6.5초) — 3.5~8.5초 (전환 후 0.5초 마진)
ffmpeg -ss 3.5 -i "$RAW" -t 5 -c copy docs/bench/P6-W4/clips/_E1_M1_P1.mp4

# Periphery 구간 (6.5~10초) — 7.0~10.0초 (3초만)
ffmpeg -ss 7.0 -i "$RAW" -t 3 -c copy docs/bench/P6-W4/clips/_E1_M1_P2.mp4
```

→ 24개 클립 (`_E*_M*_P*.mp4`).

자동화: `scripts/p6w4_bench_helper.sh trim` 사용.

---

## 블라인드 파일명 무작위화

평가자가 어느 프로토타입인지 모르게:

```bash
cd docs/bench/P6-W4/clips
# 24개 파일을 random_ids.sh 스크립트로 clip_01.mp4 ~ clip_24.mp4로 rename
# + 정답표 _truth.csv 생성 (별도 위치 보관)
bash ../../../scripts/p6w4_bench_helper.sh randomize
```

→ `clips/clip_01.mp4` ~ `clips/clip_24.mp4` + `docs/bench/P6-W4/_truth.csv` (gitignore 또는 봉인).

`checklist.md`의 "클립 ID ↔ 진실 정답표" 표를 `_truth.csv` 내용으로 채움 (평가 끝나기 전엔 비공개).

---

## 주의 사항

- **디바이스 1대 통일** — 다른 디바이스 섞이면 변수 증가. §5.11 권장.
- **밝기/화이트밸런스 자동 고정** — 카메라 앱이 자동 조정하지 못하게 (가능하면 카메라 설정에서).
- **VOLUME_UP 타이밍 정확** — 3초/6.5초 시점 정확히. 너무 빠르면 전환 직후 frame이 평가에 섞임.
- **녹화 중 손 떨림 최소화** — 거치대 권장. 머리 움직임만.

---

## adb 자동화 (`scripts/p6w4_bench_helper.sh`)

```bash
# 디바이스 wake + 앱 시작
bash scripts/p6w4_bench_helper.sh launch

# 환경 라벨 + 동작 라벨로 녹화 (12초)
bash scripts/p6w4_bench_helper.sh record E1 M1

# 녹화 종료 후 자동 pull
bash scripts/p6w4_bench_helper.sh pull E1 M1

# 모드 토글 (수동 백업용)
bash scripts/p6w4_bench_helper.sh mode_next

# 강도 sweep (수동 백업용)
bash scripts/p6w4_bench_helper.sh intensity_next

# 24 클립 후편집 (raw → P0/P1/P2 트림)
bash scripts/p6w4_bench_helper.sh trim

# 블라인드 ID 부여 + 정답표 생성
bash scripts/p6w4_bench_helper.sh randomize
```
