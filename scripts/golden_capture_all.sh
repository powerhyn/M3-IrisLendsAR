#!/usr/bin/env bash
#
# golden_capture_all.sh — 골든 베이스라인 일괄 캡처 (계획 2-2)
#
# 선정 입력(cpp/tests/golden/inputs/*.png) × 변형 매트릭스를
# golden_capture 도구로 일괄 실행해 출력 디렉토리에 베이스라인을 만든다.
#
# 변형 매트릭스(입력별):
#   - rot0   (기본)            : render + beauty + json
#   - rot90  (회전 경로 커버)   : render + json
#   - rot180 (회전 경로 커버)   : render + json
#   - rot270 (회전 경로 커버)   : render + json
#   - mirror (전면 카메라 변형)  : render + json
# 추가로 첫 입력 1장에만:
#   - gamma05      (저조도 변형)              : render + beauty + json
#   - mirror_rot270(미러×회전 조합, 전면 카메라 실경로) : json 전용
#   - lensmirror   (IrisLensConfig.is_mirror=1 렌더 분기) : render + json
#
# beauty PNG는 용량 규율(15MB)을 위해 기본/gamma 변형에만 생성한다.
#
# ⚠️ 침묵 실패 차단:
#   golden_capture가 render/beauty 실패를 [Warning]으로만 알리고 exit 0 하므로,
#   본 스크립트가 캡처마다 기대 산출물의 존재를 검증한다.
#   - 기대 파일 누락(미허용)  → 최종 비0 종료
#   - KNOWN_RENDER_FAILURES   → 현재 SDK의 알려진 실패(기대 동작)로 허용.
#     단, 허용 목록 stem에서 render가 "생성되면" 그것도 동작 변화로 보고 비0 종료
#     (양방향 앵커). 목록 변경은 README '알려진 현재 동작' 절과 함께 갱신할 것.
#
# 사용법:
#   scripts/golden_capture_all.sh [OUT_DIR] [BUILD_DIR]
#
# 인자(둘 다 선택, 기본값 제공):
#   OUT_DIR    출력 디렉토리 (기본: cpp/tests/golden/baseline)
#   BUILD_DIR  golden_capture 바이너리가 있는 빌드 디렉토리
#              (기본: cpp/cmake-build-debug)
#
# 환경변수:
#   TEXTURE    렌즈 텍스처 경로 (기본: shared/textures/2. KYOTO BROWN 1.png)
#   MODELS     모델 디렉토리 (기본: <BUILD_DIR>/bin/models)

set -euo pipefail

# ---- 경로 결정 (스크립트 위치 기준 저장소 루트 산출) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUT_DIR="${1:-${REPO_ROOT}/cpp/tests/golden/baseline}"
BUILD_DIR="${2:-${REPO_ROOT}/cpp/cmake-build-debug}"

INPUT_DIR="${REPO_ROOT}/cpp/tests/golden/inputs"
BIN="${BUILD_DIR}/bin/golden_capture"
MODELS="${MODELS:-${BUILD_DIR}/bin/models}"
TEXTURE="${TEXTURE:-${REPO_ROOT}/shared/textures/2. KYOTO BROWN 1.png}"

# 알려진 render 실패 stem (IRIS_SDK_RENDER_FAILED — 회전 변형에서 detect 좌표공간과
# render 버퍼 공간 불일치로 좌표가 프레임 폭을 벗어남. README '알려진 현재 동작' 참조).
# 이 stem들은 render.png 부재가 "기대 동작"이며, 생성되면 오히려 동작 변화다.
KNOWN_RENDER_FAILURES=(
  "screenshot_720a__rot90"
  "screenshot_720a__rot270"
)

# ---- 사전 점검 ----
if [[ ! -x "${BIN}" ]]; then
  echo "[Error] golden_capture 바이너리 없음: ${BIN}" >&2
  echo "        먼저 빌드: cd ${BUILD_DIR} && cmake .. -DIRIS_SDK_FETCH_TFLITE=OFF && cmake --build . --target golden_capture" >&2
  exit 1
fi
if [[ ! -d "${MODELS}" ]]; then
  echo "[Error] 모델 디렉토리 없음: ${MODELS}" >&2
  exit 1
fi
if [[ ! -f "${TEXTURE}" ]]; then
  echo "[Error] 텍스처 파일 없음: ${TEXTURE}" >&2
  exit 1
fi
if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "[Error] 입력 디렉토리 없음: ${INPUT_DIR}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

echo "=== golden_capture_all ==="
echo "  BIN     : ${BIN}"
echo "  MODELS  : ${MODELS}"
echo "  TEXTURE : ${TEXTURE}"
echo "  INPUTS  : ${INPUT_DIR}"
echo "  OUT     : ${OUT_DIR}"
echo

# 입력 목록 (정렬해 결정적 순서). bash 3.2 호환을 위해 mapfile 대신 while-read 사용.
INPUTS=()
while IFS= read -r line; do
  INPUTS+=("${line}")
done < <(find "${INPUT_DIR}" -maxdepth 1 -name '*.png' | sort)
if [[ "${#INPUTS[@]}" -eq 0 ]]; then
  echo "[Error] 입력 PNG 없음: ${INPUT_DIR}" >&2
  exit 1
fi

# ---- 실패 집계 (bash 3.2 호환: 일반 배열) ----
PROBLEMS=()      # 기대 산출물 누락 / 허용 목록 위반 / [Warning] 요약
WARNED=()        # [Warning] 발생 stem 목록 (정보 표시용)
EXPECTED_FILES=0
PRODUCED_FILES=0

is_known_render_failure() {
  local stem="$1" k
  for k in "${KNOWN_RENDER_FAILURES[@]}"; do
    [[ "${k}" == "${stem}" ]] && return 0
  done
  return 1
}

# capture <input> <stem> <rotation> <mirror> <gamma> <beauty:0|1> <render:0|1> [extra args...]
capture() {
  local input="$1" stem="$2" rot="$3" mirror="$4" gamma="$5" beauty="$6" render="$7"
  shift 7
  local args=(--input "${input}" --models "${MODELS}" --texture "${TEXTURE}"
              --out "${OUT_DIR}" --out-stem "${stem}"
              --rotation "${rot}" --mirror "${mirror}" --gamma "${gamma}")
  if [[ "${beauty}" -eq 0 ]]; then
    args+=(--no-beauty)
  fi
  if [[ "${render}" -eq 0 ]]; then
    args+=(--no-render)
  fi
  args+=("$@")

  # 출력을 보존해 [Warning] (render/beauty 실패의 유일한 신호)을 집계한다.
  local out
  out="$("${BIN}" "${args[@]}" 2>&1)"
  printf '%s\n' "${out}"
  if grep -q '\[Warning\]' <<< "${out}"; then
    WARNED+=("${stem}: $(grep '\[Warning\]' <<< "${out}" | tr '\n' ' ')")
  fi

  # ---- 기대 산출물 검증 (침묵 실패 차단) ----
  # JSON은 무조건 기대
  EXPECTED_FILES=$((EXPECTED_FILES + 1))
  if [[ -f "${OUT_DIR}/${stem}.result.json" ]]; then
    PRODUCED_FILES=$((PRODUCED_FILES + 1))
  else
    PROBLEMS+=("[누락] ${stem}.result.json")
  fi
  # render 기대 여부: render=1 이고 허용 목록 밖이면 기대
  if [[ "${render}" -eq 1 ]]; then
    if is_known_render_failure "${stem}"; then
      if [[ -f "${OUT_DIR}/${stem}.render.png" ]]; then
        PROBLEMS+=("[동작 변화] ${stem}.render.png 가 생성됨 — KNOWN_RENDER_FAILURES 기대(실패)와 불일치. README/허용 목록 갱신 필요")
      fi
    else
      EXPECTED_FILES=$((EXPECTED_FILES + 1))
      if [[ -f "${OUT_DIR}/${stem}.render.png" ]]; then
        PRODUCED_FILES=$((PRODUCED_FILES + 1))
      else
        PROBLEMS+=("[누락] ${stem}.render.png (render_lens 실패 또는 미검출 — 위 [Warning] 로그 확인)")
      fi
    fi
  fi
  # beauty 기대 여부
  if [[ "${beauty}" -eq 1 ]]; then
    EXPECTED_FILES=$((EXPECTED_FILES + 1))
    if [[ -f "${OUT_DIR}/${stem}.beauty.png" ]]; then
      PRODUCED_FILES=$((PRODUCED_FILES + 1))
    else
      PROBLEMS+=("[누락] ${stem}.beauty.png")
    fi
  fi
}

first=1
for input in "${INPUTS[@]}"; do
  base="$(basename "${input}" .png)"

  # 기본 (beauty 포함)                      rot mir gamma beauty render
  capture "${input}" "${base}__rot0"   0   0 1.0 1 1
  # 회전 경로 (render+json만, beauty 생략)
  capture "${input}" "${base}__rot90"  90  0 1.0 0 1
  capture "${input}" "${base}__rot180" 180 0 1.0 0 1
  capture "${input}" "${base}__rot270" 270 0 1.0 0 1
  # mirror (render+json만)
  capture "${input}" "${base}__mirror" 0   1 1.0 0 1

  # 첫 입력에만 추가 변형
  if [[ "${first}" -eq 1 ]]; then
    # gamma 저조도 변형 (beauty 포함)
    capture "${input}" "${base}__gamma05" 0 0 0.5 1 1
    # 미러×회전 조합 (전면 카메라 실경로) — JSON 전용 (용량 규율)
    capture "${input}" "${base}__mirror_rot270" 270 1 1.0 0 0
    # IrisLensConfig.is_mirror=1 렌더 분기 — 검출 동일 입력이라 JSON은 rot0과 동일(의도)
    capture "${input}" "${base}__lensmirror" 0 0 1.0 0 1 --lens-mirror 1
    first=0
  fi
done

echo
echo "=== 산출물 검증 ==="
echo "기대 파일: ${EXPECTED_FILES}, 생성 확인: ${PRODUCED_FILES}"
if [[ "${#WARNED[@]}" -gt 0 ]]; then
  echo "[Warning] 발생 캡처 (${#WARNED[@]}건):"
  for w in "${WARNED[@]}"; do
    echo "  - ${w}"
  done
fi

echo
echo "=== 완료. 출력 용량 ==="
# du는 환경에 따라 dust 등으로 alias될 수 있어 절대 경로/POSIX 옵션으로 호출
/usr/bin/du -sh "${OUT_DIR}" 2>/dev/null || true
echo "파일 수: $(find "${OUT_DIR}" -type f | wc -l | tr -d ' ')"

if [[ "${#PROBLEMS[@]}" -gt 0 ]]; then
  echo
  echo "=== 실패: 기대 산출물 불일치 ${#PROBLEMS[@]}건 ===" >&2
  for p in "${PROBLEMS[@]}"; do
    echo "  - ${p}" >&2
  done
  exit 1
fi
echo "산출물 검증: PASS (기대 ${EXPECTED_FILES} == 생성 ${PRODUCED_FILES}, 허용 목록 위반 0)"
