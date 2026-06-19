#!/usr/bin/env python3
"""④ 좌표 canonical relabeling — 골든 재캡처 migration checker.

relabeling은 left↔right 라우팅 스왑이므로, 재캡처된 baseline은 old baseline과
"L/R 필드가 정확히 교환 + 나머지 동일 + PNG byte-equal" 관계여야 한다.
golden_compare.py는 key별 엄격 비교라 스왑 자체를 통과시키지 못하므로, 이 1회용
checker가 **스왑이 옳음(다른 회귀가 아님)**을 증명한다.

invariant:
  new.left_iris      == old.right_iris      (및 역방향)
  new.left_radius    == old.right_radius
  new.left_detected  == old.right_detected
  new.eyelid_ratio_left == old.eyelid_ratio_right
  new.avg_iris_luma_left == old.avg_iris_luma_right
  그 외 모든 필드(face_mesh/face_rect/frame_*/rotation/...) 동일
  PNG byte-identical

usage:
  python3 scripts/golden_relabel_migration_check.py --old <old_dir> --new <new_dir>
"""
import argparse
import json
import sys
from pathlib import Path

# (left_key, right_key) — 재캡처 후 교환되어야 하는 쌍
SWAP_PAIRS = [
    ("left_iris", "right_iris"),
    ("left_radius", "right_radius"),
    ("left_detected", "right_detected"),
    ("eyelid_ratio_left", "eyelid_ratio_right"),
    ("avg_iris_luma_left", "avg_iris_luma_right"),
]
SWAP_KEYS = {k for pair in SWAP_PAIRS for k in pair}


def check_json(old: dict, new: dict, stem: str, errors: list):
    # 1) L/R 쌍이 정확히 교환됐는지
    for lk, rk in SWAP_PAIRS:
        if new.get(lk) != old.get(rk):
            errors.append(f"[{stem}] new.{lk} != old.{rk} (스왑 위반)")
        if new.get(rk) != old.get(lk):
            errors.append(f"[{stem}] new.{rk} != old.{lk} (스왑 위반)")
    # 2) 나머지 필드는 동일해야
    all_keys = set(old) | set(new)
    for k in sorted(all_keys - SWAP_KEYS):
        if old.get(k) != new.get(k):
            errors.append(f"[{stem}] 비-L/R 필드 '{k}' 변경됨 (불변이어야)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="재캡처 전 baseline 디렉토리(보존본)")
    ap.add_argument("--new", required=True, help="재캡처 후 baseline 디렉토리")
    args = ap.parse_args()

    old_dir, new_dir = Path(args.old), Path(args.new)
    errors: list = []

    old_jsons = sorted(old_dir.glob("*.result.json"))
    if not old_jsons:
        print(f"[FAIL] old baseline JSON 없음: {old_dir}", file=sys.stderr)
        return 2

    json_n = png_n = 0
    for of in old_jsons:
        nf = new_dir / of.name
        stem = of.name
        if not nf.exists():
            errors.append(f"[{stem}] new baseline에 없음")
            continue
        check_json(json.load(open(of)), json.load(open(nf)), stem, errors)
        json_n += 1

    # PNG byte-identical (L/R 스왑은 양안 동일 렌더라 PNG 불변이어야)
    for op in sorted(old_dir.glob("*.png")):
        npf = new_dir / op.name
        if not npf.exists():
            errors.append(f"[{op.name}] new baseline PNG 없음")
            continue
        if op.read_bytes() != npf.read_bytes():
            errors.append(f"[{op.name}] PNG byte 불일치 (스왑은 PNG 불변이어야)")
        png_n += 1

    print(f"migration check: JSON {json_n}건, PNG {png_n}건 대조 (old={old_dir} new={new_dir})")
    if errors:
        print(f"\n결과: FAIL — {len(errors)}건 위반")
        for e in errors[:40]:
            print("  " + e)
        return 1
    print("\n결과: PASS — L/R 정확 교환 + 나머지 불변 + PNG byte-equal (스왑이 옳음 증명)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
