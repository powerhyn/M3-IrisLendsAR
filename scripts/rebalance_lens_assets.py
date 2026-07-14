#!/usr/bin/env python3
"""렌즈 에셋 리밸런싱 v3 — 타겟 솔루션 농도 이식 (NLR-W2 에셋 축).

모델 변천 (근거: docs/bench/NLR-W2/formula_bench_notes.md §타겟 에셋 실측):
- v1: 네버 올리브 단일 쌍의 반경별 커브를 전 카탈로그 적용 → 7쌍 분석으로 기각
  (쌍별 커브가 스타일마다 상이 — 내측 bin 표준편차 0.33~0.43)
- v2: 타겟 정책 = "착색부 절대 농도 ~80±17 밴드로 재설정" (원본 농도와 상관 0.18)
  → 쌍 실측 7종 정확 비율 + 나머지 clamp(80/농도, 0.45, 1.0)
- v3: 전 제품 타겟 시트 입수(lens-items, 제품명 포함)로 **카탈로그 38/42종 쌍 실측**.
  이름 퍼지 매칭(≥0.82)으로 자동 쌍 구성. 쌍 없는 렌즈(miyu 4종)만 v2 정규화 유지.

- 시트 형식: facemorph 973×216 RGBA (양안 배치, 좌측 렌즈 사용)
- 적용: 균일 알파 스케일(반경 모양 보존). RGB(패턴·색)는 무변경.
- 입력: 우리 원본(1000×1000) → 출력: 512×512 (다운샘플만, 업스케일 없음)
- 사용: python3 scripts/rebalance_lens_assets.py [--suffix -rebal] [--dry-run]
"""
import argparse
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
LENS_DIR = REPO / "android/demo-app/src/main/assets/lenses"
SHEETS_DIR = Path("/Volumes/M3-P31/Projects/MerooMong/EtcWorkStage/lens-items")
OUT_SIZE = 512
T_INK = 80.0                    # 타겟 착색부 농도 밴드 중심 (쌍 실측 mean)
SCALE_MIN, SCALE_MAX = 0.45, 1.0
MATCH_THRESHOLD = 0.82

# 육안 벤치 후 개별 보정용 (stem → 스케일 강제값)
OVERRIDES: dict[str, float] = {}

EXCLUDE_SUBSTRINGS = ("-target", "zz_target", "-rebal")


def norm_name(s: str) -> str:
    n = re.sub(r"[^a-z0-9]", "", s.lower())
    return n.rstrip("0123456789") or n


def ink_density(alpha: np.ndarray, cx: float, cy: float, r: float) -> float:
    """착색부(0.4 ≤ r/R ≤ 1.0) 평균 알파."""
    yy, xx = np.ogrid[: alpha.shape[0], : alpha.shape[1]]
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
    return float(alpha[(rr >= 0.4) & (rr <= 1.0)].mean())


def sheet_ink_density(path: Path) -> float:
    """facemorph 시트 좌측 렌즈의 착색부 농도."""
    a = np.asarray(Image.open(path).convert("RGBA"))[..., 3].astype(np.float64)
    half = a[:, : a.shape[1] // 2]
    ys, xs = np.where(half > 8)
    cx, cy = xs.mean(), ys.mean()
    r = np.sqrt(((xs - cx) ** 2 + (ys - cy) ** 2)).max()
    return ink_density(half, cx, cy, r)


def asset_ink_density(path: Path) -> float:
    a = np.asarray(Image.open(path).convert("RGBA"))[..., 3].astype(np.float64)
    h, w = a.shape
    return ink_density(a, w / 2.0, h / 2.0, min(h, w) / 2.0)


def build_pairs(catalog: list[Path]) -> dict[str, Path]:
    """제품명 퍼지 매칭으로 카탈로그 stem → 타겟 시트 경로 자동 구성."""
    stems = {p.stem: norm_name(p.stem.split("_", 1)[1] if "_" in p.stem else p.stem)
             for p in catalog}
    pairs: dict[str, Path] = {}
    for sheet in sorted(SHEETS_DIR.rglob("*.png")):
        n = norm_name(sheet.stem)
        stem, score = max(
            ((s, SequenceMatcher(None, n, v).ratio()) for s, v in stems.items()),
            key=lambda kv: kv[1],
        )
        if score >= MATCH_THRESHOLD and (
            stem not in pairs or score > pairs[stem][1]  # type: ignore[index]
        ):
            pairs[stem] = (sheet, score)  # type: ignore[assignment]
    return {s: v[0] for s, v in pairs.items()}  # type: ignore[index]


def decide_scale(path: Path, pairs: dict[str, Path]) -> tuple[float, str]:
    stem = path.stem
    if stem in OVERRIDES:
        return OVERRIDES[stem], "override"
    ours = asset_ink_density(path)
    if stem in pairs:
        target = sheet_ink_density(pairs[stem])
        return target / ours, f"쌍({pairs[stem].parent.name}/{pairs[stem].stem})"
    return float(np.clip(T_INK / ours, SCALE_MIN, SCALE_MAX)), f"정규화(80/{ours:.0f})"


def rebalance(path: Path, scale: float, suffix: str) -> Path:
    im = Image.open(path).convert("RGBA")
    arr = np.asarray(im).astype(np.float64)
    arr[..., 3] = np.clip(arr[..., 3] * scale, 0, 255)
    out_im = Image.fromarray(arr.astype(np.uint8))
    if out_im.size != (OUT_SIZE, OUT_SIZE):
        out_im = out_im.resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS)  # 다운샘플만
    out_path = path.with_name(path.stem + suffix + ".png")
    out_im.save(out_path)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="-rebal")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    catalog = sorted(
        p for p in LENS_DIR.glob("*.png")
        if not any(s in p.name for s in EXCLUDE_SUBSTRINGS)
    )
    pairs = build_pairs(catalog)
    n_paired = sum(1 for p in catalog if p.stem in pairs)
    print(f"대상 {len(catalog)}개 — 쌍 실측 {n_paired} / 정규화 {len(catalog) - n_paired}")
    for p in catalog:
        scale, origin = decide_scale(p, pairs)
        if args.dry_run:
            print(f"  [dry] {p.name}: ×{scale:.2f} ({origin})")
            continue
        rebalance(p, scale, args.suffix)
        print(f"  {p.name}: ×{scale:.2f} ({origin})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
