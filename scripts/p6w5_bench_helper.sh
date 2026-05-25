#!/usr/bin/env bash
# P6-W5 B1+B8 벤치 자동화 헬퍼
# 사용법: bash scripts/p6w5_bench_helper.sh <command> [args]
#
# 명령:
#   launch          디바이스 wake + 앱 강제 재시작
#   record <take>   14초 스크린 녹화 (예: record T1) — 녹화 중 A/B/C/D 버튼 수동 토글
#   pull <take>     녹화 파일 디바이스에서 호스트로 가져옴
#   trim            raw 디렉토리 전체 → A/B/C/D 클립 32개 생성 (ffmpeg 필요)
#   randomize       32 클립 → 무작위 ID 부여 + 정답표 _truth.csv 생성
#
# 주의: 4조합 토글은 화면 버튼(A/B/C/D)이라 좌표가 디바이스마다 달라 자동화하지 않음.
#       녹화 중 3/6/9초 시점에 버튼을 직접 누른다 (recording_guide.md §촬영 절차).
#
# 출처: docs/bench/P6-W5/recording_guide.md

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="$ROOT/docs/bench/P6-W5"
RAW_DIR="$BENCH_DIR/raw"
CLIPS_DIR="$BENCH_DIR/clips"
APP_PKG="com.irislenssdk.demo"
APP_ACT=".GpuRenderActivity"

# Take → SKU/조명 매핑 (checklist.md Take 표).
# macOS 기본 bash 3.2는 associative array 미지원이라 case 함수로 구현.
take_sku() {
    case "$1" in
        T1|T8) echo "S1_darkbrown" ;;
        T2)    echo "S2_hazel" ;;
        T3|T6|T7) echo "S3_graygray" ;;
        T4)    echo "S4_opaque" ;;
        T5)    echo "S5_whitegfx" ;;
        *)     echo "" ;;
    esac
}
take_light() {
    case "$1" in
        T1|T2|T3|T4|T5) echo "E1_fluor" ;;
        T6|T8) echo "E3_lowlight" ;;
        T7)    echo "E2_sidelight" ;;
        *)     echo "" ;;
    esac
}
# 조합 라벨 → 정답 (블라인드 — _truth.csv에만 기록)
combo_desc() {
    case "$1" in
        A) echo "Normal+color-veto" ;;
        B) echo "Normal+luma-only" ;;
        C) echo "CRL+color-veto" ;;
        D) echo "CRL+luma-only" ;;
        *) echo "" ;;
    esac
}

require_adb() {
    if ! command -v adb >/dev/null 2>&1; then
        echo "❌ adb not in PATH"
        exit 1
    fi
    if ! adb devices | grep -q "device$"; then
        echo "❌ No Android device connected"
        adb devices
        exit 1
    fi
}

cmd_launch() {
    require_adb
    adb shell input keyevent KEYCODE_WAKEUP
    sleep 1
    adb shell am force-stop "$APP_PKG"
    sleep 1
    adb logcat -c
    adb shell am start -n "$APP_PKG/$APP_ACT"
    sleep 5
    echo "✅ App launched. Sclera Protect를 ON으로 두고 SKU 선택 후 record 시작."
}

cmd_record() {
    local take="$1"
    if [[ -z "$take" || -z "$(take_sku "$take")" ]]; then
        echo "Usage: record <T1..T8>"
        exit 1
    fi
    require_adb
    local fname="raw_${take}.mp4"
    echo "🎥 Recording $fname (14s) — SKU=$(take_sku "$take") 조명=$(take_light "$take")"
    echo "   ⏱  3s→버튼B, 6s→버튼C, 9s→버튼D (시작 시 A)"
    adb shell screenrecord "/sdcard/$fname" --time-limit 14
    echo "✅ Recording done. Pull next: bash $0 pull $take"
}

cmd_pull() {
    local take="$1"
    require_adb
    mkdir -p "$RAW_DIR"
    local fname="raw_${take}.mp4"
    adb pull "/sdcard/$fname" "$RAW_DIR/$fname"
    echo "✅ $RAW_DIR/$fname"
}

cmd_trim() {
    if ! command -v ffmpeg >/dev/null 2>&1; then
        echo "❌ ffmpeg not installed (brew install ffmpeg)"
        exit 1
    fi
    if [[ ! -d "$RAW_DIR" ]]; then
        echo "❌ Raw dir not found: $RAW_DIR (record first)"
        exit 1
    fi
    mkdir -p "$CLIPS_DIR"
    local count=0
    for take in T1 T2 T3 T4 T5 T6 T7 T8; do
        local raw="$RAW_DIR/raw_${take}.mp4"
        if [[ ! -f "$raw" ]]; then
            echo "⚠️  Skip (not found): $raw"
            continue
        fi
        # A(0~3s)→0.5, B(3~6s)→3.5, C(6~9s)→6.5, D(9~12s)→9.5. 각 2.5s.
        ffmpeg -y -ss 0.5 -i "$raw" -t 2.5 -c copy "$CLIPS_DIR/_${take}_A.mp4" 2>/dev/null
        ffmpeg -y -ss 3.5 -i "$raw" -t 2.5 -c copy "$CLIPS_DIR/_${take}_B.mp4" 2>/dev/null
        ffmpeg -y -ss 6.5 -i "$raw" -t 2.5 -c copy "$CLIPS_DIR/_${take}_C.mp4" 2>/dev/null
        ffmpeg -y -ss 9.5 -i "$raw" -t 2.5 -c copy "$CLIPS_DIR/_${take}_D.mp4" 2>/dev/null
        count=$((count + 4))
        echo "✅ Trimmed $take → A/B/C/D"
    done
    echo "Total clips: $count (target 32)"
}

cmd_randomize() {
    if [[ ! -d "$CLIPS_DIR" ]]; then
        echo "❌ Clips dir not found: $CLIPS_DIR (trim first)"
        exit 1
    fi
    local truth="$BENCH_DIR/_truth.csv"
    echo "clip_id,take,combo,combo_desc,sku,lighting" > "$truth"
    local sources=()
    for f in "$CLIPS_DIR"/_T?_[ABCD].mp4; do
        [[ -f "$f" ]] && sources+=("$f")
    done
    if [[ ${#sources[@]} -ne 32 ]]; then
        echo "⚠️  Expected 32 clips, found ${#sources[@]}"
    fi
    local shuffled
    shuffled=$(printf "%s\n" "${sources[@]}" | sort -R)
    local idx=1
    while IFS= read -r src; do
        local base
        base=$(basename "$src")
        # _T1_A.mp4 → T1, A
        local take=$(echo "$base" | cut -d_ -f2)
        local combo=$(echo "$base" | cut -d_ -f3 | cut -d. -f1)
        local newname=$(printf "clip_%02d.mp4" "$idx")
        mv "$src" "$CLIPS_DIR/$newname"
        echo "$newname,$take,$combo,$(combo_desc "$combo"),$(take_sku "$take"),$(take_light "$take")" >> "$truth"
        idx=$((idx + 1))
    done <<< "$shuffled"
    echo "✅ Randomized → $CLIPS_DIR/clip_01.mp4 ~ clip_$(printf '%02d' $((idx - 1))).mp4"
    echo "✅ Truth table → $truth"
    echo "⚠️  $truth는 평가 끝까지 비공개. .gitignore 또는 봉인 권장."
}

# Dispatch
case "$1" in
    launch) cmd_launch ;;
    record) cmd_record "$2" ;;
    pull) cmd_pull "$2" ;;
    trim) cmd_trim ;;
    randomize) cmd_randomize ;;
    *)
        echo "Usage: $0 {launch|record <take>|pull <take>|trim|randomize}"
        echo "See docs/bench/P6-W5/recording_guide.md"
        exit 1
        ;;
esac
