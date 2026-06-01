#!/usr/bin/env bash
# P6-W4 B2 벤치 자동화 헬퍼
# 사용법: bash scripts/p6w4_bench_helper.sh <command> [args]
#
# 명령:
#   launch                  디바이스 wake + 앱 강제 재시작
#   record <env> <motion>   12초 스크린 녹화 (예: record E1 M1)
#   pull <env> <motion>     녹화 파일 디바이스에서 호스트로 가져옴
#   mode_next               reflection mode 1단계 토글 (VOLUME_UP 시뮬레이션)
#   intensity_next          intensity sweep 1단계 (VOLUME_DOWN 시뮬레이션)
#   trim                    raw 디렉토리 전체 → P0/P1/P2 클립 24개 생성 (ffmpeg 필요)
#   randomize               24 클립 → 무작위 ID 부여 + 정답표 _truth.csv 생성
#
# 출처: docs/bench/P6-W4/recording_guide.md

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="$ROOT/docs/bench/P6-W4"
RAW_DIR="$BENCH_DIR/raw"
CLIPS_DIR="$BENCH_DIR/clips"
APP_PKG="com.irislenssdk.demo"
APP_ACT=".GpuRenderActivity"

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
    adb shell input keyevent KEYCODE_BACK
    sleep 1
    adb shell am force-stop "$APP_PKG"
    sleep 1
    adb logcat -c
    adb shell am start -n "$APP_PKG/$APP_ACT"
    sleep 5
    echo "✅ App launched. EnvMap load check:"
    adb logcat -d | grep -E "EnvMap load|GPU Lens Renderer init" | tail -5
}

cmd_record() {
    local env="$1"
    local motion="$2"
    if [[ -z "$env" || -z "$motion" ]]; then
        echo "Usage: record <E1|E2|E3|E4> <M1|M2>"
        exit 1
    fi
    require_adb
    local fname="raw_${env}_${motion}.mp4"
    echo "🎥 Recording $fname (12s — VOLUME_UP at 3s + 6.5s)"
    adb shell screenrecord "/sdcard/$fname" --time-limit 12
    echo "✅ Recording done. Pull next: bash $0 pull $env $motion"
}

cmd_pull() {
    local env="$1"
    local motion="$2"
    require_adb
    mkdir -p "$RAW_DIR"
    local fname="raw_${env}_${motion}.mp4"
    adb pull "/sdcard/$fname" "$RAW_DIR/$fname"
    echo "✅ $RAW_DIR/$fname"
}

cmd_mode_next() {
    require_adb
    adb shell input keyevent KEYCODE_VOLUME_UP
    sleep 0.5
    adb logcat -d | grep "reflection mode →" | tail -1
}

cmd_intensity_next() {
    require_adb
    adb shell input keyevent KEYCODE_VOLUME_DOWN
    sleep 0.5
    adb logcat -d | grep "reflection intensity →" | tail -1
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
    for env in E1 E2 E3 E4; do
        for motion in M1 M2; do
            local raw="$RAW_DIR/raw_${env}_${motion}.mp4"
            if [[ ! -f "$raw" ]]; then
                echo "⚠️  Skip (not found): $raw"
                continue
            fi
            # OFF 구간 (0.5~5.5s) — 5s
            ffmpeg -y -ss 0.5 -i "$raw" -t 5 -c copy "$CLIPS_DIR/_${env}_${motion}_P0.mp4" 2>/dev/null
            # EnvMap (3.5~8.5s) — 5s
            ffmpeg -y -ss 3.5 -i "$raw" -t 5 -c copy "$CLIPS_DIR/_${env}_${motion}_P1.mp4" 2>/dev/null
            # Periphery (7.0~10.0s) — 3s
            ffmpeg -y -ss 7.0 -i "$raw" -t 3 -c copy "$CLIPS_DIR/_${env}_${motion}_P2.mp4" 2>/dev/null
            count=$((count + 3))
            echo "✅ Trimmed $env/$motion → P0/P1/P2"
        done
    done
    echo "Total clips: $count (target 24)"
}

cmd_randomize() {
    if [[ ! -d "$CLIPS_DIR" ]]; then
        echo "❌ Clips dir not found: $CLIPS_DIR (trim first)"
        exit 1
    fi
    local truth="$BENCH_DIR/_truth.csv"
    echo "clip_id,source_env,source_motion,prototype" > "$truth"
    # 24 클립 무작위 순서로 ID 부여
    local sources=()
    for f in "$CLIPS_DIR"/_E?_M?_P?.mp4; do
        [[ -f "$f" ]] && sources+=("$f")
    done
    if [[ ${#sources[@]} -ne 24 ]]; then
        echo "⚠️  Expected 24 clips, found ${#sources[@]}"
    fi
    # 셔플
    local shuffled
    shuffled=$(printf "%s\n" "${sources[@]}" | sort -R)
    local idx=1
    while IFS= read -r src; do
        local base
        base=$(basename "$src")
        # _E1_M1_P0.mp4 → E1, M1, P0
        local env=$(echo "$base" | cut -d_ -f2)
        local motion=$(echo "$base" | cut -d_ -f3)
        local proto=$(echo "$base" | cut -d_ -f4 | cut -d. -f1)
        local newname=$(printf "clip_%02d.mp4" "$idx")
        mv "$src" "$CLIPS_DIR/$newname"
        echo "$newname,$env,$motion,$proto" >> "$truth"
        idx=$((idx + 1))
    done <<< "$shuffled"
    echo "✅ Randomized → $CLIPS_DIR/clip_01.mp4 ~ clip_$(printf '%02d' $((idx - 1))).mp4"
    echo "✅ Truth table → $truth"
    echo "⚠️  $truth는 평가 끝까지 비공개. .gitignore 또는 봉인 권장."
}

# Dispatch
case "$1" in
    launch) cmd_launch ;;
    record) cmd_record "$2" "$3" ;;
    pull) cmd_pull "$2" "$3" ;;
    mode_next) cmd_mode_next ;;
    intensity_next) cmd_intensity_next ;;
    trim) cmd_trim ;;
    randomize) cmd_randomize ;;
    *)
        echo "Usage: $0 {launch|record <env> <motion>|pull <env> <motion>|mode_next|intensity_next|trim|randomize}"
        echo "See docs/bench/P6-W4/recording_guide.md"
        exit 1
        ;;
esac
