#!/usr/bin/env python3
"""golden_compare.py — 골든 베이스라인 비교 (계획 2-2)

baseline 디렉토리와 candidate 디렉토리의 캡처 산출물을 비교한다.

비교 규칙:
  - *.result.json : 결정적 필드를 의미 단위로 비교
      * 좌표류(x, y, z, face_rect, face_rotation, confidence, visibility,
        eyelid_ratio, iris_quality, avg_iris_luma, gamma) : |Δ| <= EPS_NORM (기본 1e-4)
      * 반지름/픽셀류(left_radius, right_radius) : |Δ| <= EPS_PIX (기본 0.5)
      * 정수/문자열/bool(schema, model_version, rotation, mirror,
        input_width/height, *_detected, *_valid, *_count, eye_refiner_used)
        : 정확 일치
  - *.png : 바이트 동일성. 불일치 시 크기 + 바이트 diff 통계 리포트.

표준 라이브러리만 사용한다(PIL/opencv-python 미사용). PNG는 IHDR 청크에서
직접 폭/높이를 파싱한다.

종료코드: 0 = 모든 비교 통과, 1 = 불일치 또는 누락.

사용법:
  golden_compare.py --baseline <dir> --candidate <dir>
                    [--eps-norm 1e-4] [--eps-pix 0.5] [--quiet]
"""

import argparse
import json
import os
import struct
import sys


# 픽셀 단위 절대 임계값을 쓰는 float 필드(키 이름 기준)
PIXEL_FIELDS = {"left_radius", "right_radius"}

# 정확 일치를 요구하는 비-float 필드
EXACT_FIELDS = {
    "schema", "model_version", "rotation", "mirror",
    "input_width", "input_height",
    "detected", "left_detected", "right_detected",
    "face_mesh_valid", "face_mesh_count", "eye_refiner_used",
}


def list_files(d, suffix):
    """디렉토리 d에서 suffix로 끝나는 파일명(베이스명) 집합."""
    out = set()
    if not os.path.isdir(d):
        return out
    for name in os.listdir(d):
        if name.endswith(suffix):
            out.add(name)
    return out


def png_dimensions(path):
    """PNG IHDR에서 (width, height) 파싱. 실패 시 (None, None)."""
    try:
        with open(path, "rb") as f:
            sig = f.read(8)
            if sig != b"\x89PNG\r\n\x1a\n":
                return (None, None)
            f.read(4)            # IHDR length
            f.read(4)            # "IHDR"
            w, h = struct.unpack(">II", f.read(8))
            return (w, h)
    except (OSError, struct.error):
        return (None, None)


def byte_diff_stats(a_path, b_path):
    """두 파일의 바이트 diff 통계 딕셔너리 반환."""
    with open(a_path, "rb") as f:
        a = f.read()
    with open(b_path, "rb") as f:
        b = f.read()
    stats = {
        "size_baseline": len(a),
        "size_candidate": len(b),
    }
    n = min(len(a), len(b))
    diff_count = 0
    first_diff = -1
    for i in range(n):
        if a[i] != b[i]:
            diff_count += 1
            if first_diff < 0:
                first_diff = i
    diff_count += abs(len(a) - len(b))
    stats["differing_bytes"] = diff_count
    stats["first_diff_offset"] = first_diff
    aw, ah = png_dimensions(a_path)
    bw, bh = png_dimensions(b_path)
    stats["dim_baseline"] = (aw, ah)
    stats["dim_candidate"] = (bw, bh)
    return stats


def flatten_json(obj, prefix=""):
    """중첩 JSON을 (경로키, 값) 쌍 리스트로 평탄화.
    리스트는 [i] 인덱스로, dict는 .key 로 펼친다.
    """
    items = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            items.extend(flatten_json(v, prefix + ("." if prefix else "") + k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            items.extend(flatten_json(v, "{}[{}]".format(prefix, i)))
    else:
        items.append((prefix, obj))
    return items


def leaf_field_name(path):
    """평탄화 경로에서 마지막 식별자(필드명)를 추출.
    예: 'left_iris[0].x' -> 'x', 'left_radius' -> 'left_radius'
    """
    last = path.split(".")[-1]
    # 인덱스 제거: x[0] 형태는 없음(인덱스는 식별자 뒤). 안전하게 '[' 앞만.
    return last.split("[")[0]


def compare_json(base_path, cand_path, eps_norm, eps_pix, problems):
    """두 JSON을 비교해 불일치를 problems 리스트에 추가."""
    try:
        with open(base_path) as f:
            b = json.load(f)
        with open(cand_path) as f:
            c = json.load(f)
    except (OSError, ValueError) as e:
        problems.append("[JSON 파싱 실패] {} : {}".format(
            os.path.basename(base_path), e))
        return

    bflat = dict(flatten_json(b))
    cflat = dict(flatten_json(c))

    bkeys = set(bflat.keys())
    ckeys = set(cflat.keys())

    name = os.path.basename(base_path)

    for k in sorted(bkeys - ckeys):
        problems.append("[{}] candidate에 없는 키: {}".format(name, k))
    for k in sorted(ckeys - bkeys):
        problems.append("[{}] baseline에 없는 키: {}".format(name, k))

    for k in sorted(bkeys & ckeys):
        bv = bflat[k]
        cv = cflat[k]
        field = leaf_field_name(k)

        if isinstance(bv, bool) or isinstance(cv, bool):
            if bv != cv:
                problems.append("[{}] {} bool 불일치: {} != {}".format(
                    name, k, bv, cv))
            continue

        if isinstance(bv, (int, float)) and isinstance(cv, (int, float)) \
                and not (isinstance(bv, bool) or isinstance(cv, bool)):
            # 정확 일치 필드(정수 메타)는 == 비교
            if field in EXACT_FIELDS:
                if bv != cv:
                    problems.append("[{}] {} 정확 불일치: {} != {}".format(
                        name, k, bv, cv))
                continue
            eps = eps_pix if field in PIXEL_FIELDS else eps_norm
            if abs(float(bv) - float(cv)) > eps:
                problems.append(
                    "[{}] {} 수치 불일치: |{:.6f} - {:.6f}| = {:.6g} > eps {:g}".format(
                        name, k, float(bv), float(cv),
                        abs(float(bv) - float(cv)), eps))
            continue

        # 문자열/기타
        if bv != cv:
            problems.append("[{}] {} 값 불일치: {!r} != {!r}".format(
                name, k, bv, cv))


def compare_png(base_path, cand_path, problems):
    """두 PNG를 바이트 동일성으로 비교."""
    name = os.path.basename(base_path)
    with open(base_path, "rb") as f:
        a = f.read()
    with open(cand_path, "rb") as f:
        b = f.read()
    if a == b:
        return
    stats = byte_diff_stats(base_path, cand_path)
    problems.append(
        "[{}] PNG 바이트 불일치: size {}->{} dim {}->{} 다른바이트 {} 첫오프셋 {}".format(
            name,
            stats["size_baseline"], stats["size_candidate"],
            stats["dim_baseline"], stats["dim_candidate"],
            stats["differing_bytes"], stats["first_diff_offset"]))


def main():
    ap = argparse.ArgumentParser(description="골든 베이스라인 비교")
    ap.add_argument("--baseline", required=True, help="기준 디렉토리")
    ap.add_argument("--candidate", required=True, help="후보 디렉토리")
    ap.add_argument("--eps-norm", type=float, default=1e-4,
                    help="정규화 좌표 절대 임계값 (기본 1e-4)")
    ap.add_argument("--eps-pix", type=float, default=0.5,
                    help="픽셀 단위(반지름) 절대 임계값 (기본 0.5)")
    ap.add_argument("--quiet", action="store_true", help="통과 항목 출력 생략")
    args = ap.parse_args()

    problems = []

    # ---- JSON 비교 ----
    base_json = list_files(args.baseline, ".result.json")
    cand_json = list_files(args.candidate, ".result.json")
    for missing in sorted(base_json - cand_json):
        problems.append("[누락] candidate에 없는 JSON: {}".format(missing))
    for extra in sorted(cand_json - base_json):
        problems.append("[추가] baseline에 없는 JSON: {}".format(extra))
    json_compared = 0
    for name in sorted(base_json & cand_json):
        compare_json(os.path.join(args.baseline, name),
                     os.path.join(args.candidate, name),
                     args.eps_norm, args.eps_pix, problems)
        json_compared += 1

    # ---- PNG 비교 ----
    base_png = list_files(args.baseline, ".png")
    cand_png = list_files(args.candidate, ".png")
    for missing in sorted(base_png - cand_png):
        problems.append("[누락] candidate에 없는 PNG: {}".format(missing))
    for extra in sorted(cand_png - base_png):
        problems.append("[추가] baseline에 없는 PNG: {}".format(extra))
    png_compared = 0
    for name in sorted(base_png & cand_png):
        compare_png(os.path.join(args.baseline, name),
                    os.path.join(args.candidate, name),
                    problems)
        png_compared += 1

    # ---- 리포트 ----
    print("=" * 60)
    print("골든 비교 리포트")
    print("  baseline : {}".format(args.baseline))
    print("  candidate: {}".format(args.candidate))
    print("  eps_norm = {:g}, eps_pix = {:g}".format(args.eps_norm, args.eps_pix))
    print("  JSON 비교: {}건, PNG 비교: {}건".format(json_compared, png_compared))
    print("=" * 60)

    if not problems:
        print("결과: PASS — 모든 항목 일치 (불일치 0건)")
        return 0

    print("결과: FAIL — 불일치 {}건".format(len(problems)))
    for p in problems:
        print("  - " + p)
    return 1


if __name__ == "__main__":
    sys.exit(main())
