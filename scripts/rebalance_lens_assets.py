#!/usr/bin/env python3
"""렌즈 에셋 리밸런싱 v4 — 타겟 솔루션 캘리브레이션 4-노브 이식 (NLR-W2 에셋 축).

모델 변천 (근거: docs/bench/NLR-W2/formula_bench_notes.md §타겟 에셋 실측):
- v1: 네버 올리브 단일 쌍의 반경별 커브 → 7쌍 분석으로 기각 (스타일별 커브 상이)
- v2: 착색부 농도 80±17 밴드 정규화 (타겟↔원본 농도 상관 0.18 = 자체 재설정 정책)
- v3: 전 제품 타겟 시트(lens-items) 입수 — 38/42쌍 실측 균일 알파 스케일
- v4: 38쌍 5차원 분석으로 발견한 타겟 4-노브 정책 구현:
  ① 농도(α) = 밴드 재설정 (쌍 실측 / 정규화 80)          [v3 계승]
  ② 림 잉크 색 = 다크닝 (휘도 평균 73→48, 실효어둠 유지)   [신규]
  ③ 패턴 대비 = 절대값 보존 (농도 깎아도 잔차 유지, 인구 +8%) [신규]
  ④ 형태(림 두께·클리어존·구조) = 원본 계승 (상관 0.77~0.83) [무변경 정책]
  구현: α' = renorm(방사평균×s + 잔차×k) → 착색부 농도를 목표에 정확히 재정규화.
  v4.1: k는 쌍별 타겟 시트의 실측 패턴 에너지(공통 256px 공간, 도트 억제 블러 후
  ink 존 잔차 std)에 수렴하도록 반복 보정 — 재정규화가 패턴을 재감쇠하는 문제 해소
  (돌초코 실기기 판정 "타겟보다 패턴 약함" 반영). 쌍 없는 렌즈는 k=max(1,s) 유지.
  림 RGB × ratio(r 0.72→0.88 램프). 예외(과두꺼운 림 슬리밍 2종 등)는 OVERRIDES로.

- 시트: facemorph 973×216 RGBA (좌측 렌즈 사용). 입력 1000² → 출력 512² (다운샘플만).
- 사용: python3 scripts/rebalance_lens_assets.py [--suffix -rebal] [--dry-run]
"""
import argparse
import re
from typing import Optional
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
LENS_DIR = REPO / "android/demo-app/src/main/assets/lenses"
SHEETS_DIR = Path("/Volumes/M3-P31/Projects/MerooMong/EtcWorkStage/lens-items")
OUT_SIZE = 512
T_INK = 80.0                       # 타겟 착색부 농도 밴드 중심
SCALE_MIN, SCALE_MAX = 0.45, 1.2
RIM_RATIO_DEFAULT = 0.66           # 쌍 없는 렌즈용 림 다크닝 (인구 평균 48/73)
MID_RATIO_DEFAULT = 0.74           # 쌍 없는 렌즈용 mid 잉크색 보정 (인구 평균 luma ratio, Codex 재계산)
RIM_RATIO_CLAMP = (0.15, 1.0)
MATCH_THRESHOLD = 0.82
N_BINS = 32

# 육안 벤치 후 개별 보정 (stem → dict(scale=..., rim_ratio=...) 부분 지정 가능)
OVERRIDES: dict[str, dict] = {}

EXCLUDE_SUBSTRINGS = ("-target", "zz_target", "-rebal")
LUMA = np.array([0.299, 0.587, 0.114])


def norm_name(s: str) -> str:
    n = re.sub(r"[^a-z0-9]", "", s.lower())
    return n.rstrip("0123456789") or n


def polar(shape, cx, cy, r):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r


def ink_density(alpha, rr) -> float:
    return float(alpha[(rr >= 0.4) & (rr <= 1.0)].mean())


def ink_lum(arr, rr, lo: float, hi: float) -> float:
    """반경 구간 [lo, hi] 알파가중 잉크 휘도."""
    a = arr[..., 3].astype(np.float64)
    L = arr[..., :3].astype(np.float64) @ LUMA
    m = (rr >= lo) & (rr <= hi) & (a > 16)
    if m.sum() < 20:
        return float("nan")
    w = a[m]
    return float((L[m] * w).sum() / w.sum())


def rim_ink_lum(arr, rr) -> float:
    """림(r 0.8~1.0) 알파가중 잉크 휘도."""
    return ink_lum(arr, rr, 0.8, 1.0)


def sheet_left(path: Path):
    im = Image.open(path).convert("RGBA")
    a = np.asarray(im)[..., 3]
    half = a.shape[1] // 2
    ys, xs = np.where(a[:, :half] > 8)
    cx, cy = xs.mean(), ys.mean()
    r = np.sqrt(((xs - cx) ** 2 + (ys - cy) ** 2)).max()
    arr = np.asarray(im.crop((int(cx - r), int(cy - r), int(cx + r), int(cy + r))))
    rr = polar(arr.shape[:2], arr.shape[1] / 2, arr.shape[0] / 2, min(arr.shape[:2]) / 2)
    return arr, rr


def build_pairs(catalog, verbose: bool = False):
    stems = {p.stem: norm_name(p.stem.split("_", 1)[1] if "_" in p.stem else p.stem)
             for p in catalog}
    best: dict[str, tuple] = {}
    for sheet in sorted(SHEETS_DIR.rglob("*.png")):
        n = norm_name(sheet.stem)
        scored = sorted(((SequenceMatcher(None, n, v).ratio(), s) for s, v in stems.items()),
                        reverse=True)
        score, stem = scored[0]
        # Codex 리뷰 (b): 1·2등 마진 검사 — 좁으면 오매칭 위험 경고 (신제품 온보딩 안전장치)
        if verbose and score >= MATCH_THRESHOLD and score - scored[1][0] < 0.03:
            print(f"  ⚠️ 매칭 마진 좁음: {sheet.stem} → {stem} ({score:.3f}) vs {scored[1][1]} ({scored[1][0]:.3f})")
        if score >= MATCH_THRESHOLD and (stem not in best or score > best[stem][1]):
            best[stem] = (sheet, score)
    return {s: v[0] for s, v in best.items()}


def pattern_energy(alpha: np.ndarray, rr: np.ndarray) -> float:
    """패턴 에너지 = 방사 평균 제거 잔차의 std (ink 존 0.45~0.8).
    호출측에서 공통 스케일(256px)·도트 억제 블러를 맞춰서 넘겨야 쌍 비교 가능."""
    prof = np.zeros(N_BINS)
    for i in range(N_BINS):
        m = (rr >= i / N_BINS) & (rr < (i + 1) / N_BINS)
        prof[i] = alpha[m].mean() if m.any() else 0.0
    centers = (np.arange(N_BINS) + 0.5) / N_BINS
    mean_r = np.interp(rr, centers, prof)
    ink = (rr >= 0.45) & (rr < 0.8)
    return float((alpha - mean_r)[ink].std())


def to_common_alpha(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RGBA 배열 → 공통 256px 알파(도트 억제 블러) + 극좌표."""
    from PIL import ImageFilter
    im = Image.fromarray(arr.astype(np.uint8)).resize((256, 256), Image.LANCZOS)
    im = im.filter(ImageFilter.GaussianBlur(1.2))
    a = np.asarray(im)[..., 3].astype(np.float64)
    return a, polar((256, 256), 128.0, 128.0, 128.0)


def transform_v4(arr: np.ndarray, target_ink: float, mid_ratio: float, rim_ratio: float,
                 target_pattern: Optional[float] = None):
    """4-노브 변환 v4.2 (Codex 교차 리뷰 반영):
    - 패턴 k = 고정점 반복 → 로그 간격 스캔 + 최근접 선택 + 양방향 수렴 플래그
      (과보정 케이스 nude-ash-rose 1.85×, 과소 doll-choco 0.42× 감지용)
    - RGB 잉크 보정 전역화: 타겟은 림만이 아니라 착색부 전체 잉크가 어두움
      (알파가중 luma 107→78) → mid/rim 비율을 반경 보간해 적용
    - 농도 목표는 clamp된 s 기준으로 정합 (SCALE_MIN/MAX 무력화 버그 수정)
    반환: (uint8 RGBA, 진단 dict)"""
    h, w = arr.shape[:2]
    rr = polar((h, w), w / 2.0, h / 2.0, min(h, w) / 2.0)
    a = arr[..., 3].astype(np.float64)

    # 방사 평균/잔차 분해
    prof = np.zeros(N_BINS)
    for i in range(N_BINS):
        m = (rr >= i / N_BINS) & (rr < (i + 1) / N_BINS)
        prof[i] = a[m].mean() if m.any() else 0.0
    centers = (np.arange(N_BINS) + 0.5) / N_BINS
    mean_r = np.interp(rr, centers, prof)
    res = a - mean_r

    ours = ink_density(a, rr)
    s = float(np.clip(target_ink / max(ours, 1e-6), SCALE_MIN, SCALE_MAX))
    ink_goal = ours * s                          # clamp 정합: 목표 농도도 clamp를 존중

    def compose(k: float) -> np.ndarray:
        a2 = np.clip(mean_r * s + res * k, 0, 255)
        achieved = ink_density(a2, rr)           # 0/255 클리핑 잔차 재정규화
        if achieved > 1e-6:
            a2 = np.clip(a2 * (ink_goal / achieved), 0, 255)
        return a2

    diag = {"pattern_flag": None, "k": max(1.0, s)}
    if target_pattern is not None and target_pattern > 1e-3:
        best_k, best_err, best_a2, best_pat = None, 1e18, None, 0.0
        for k in np.geomspace(0.15, 6.0, 14):
            a2c = compose(float(k))
            tmp = arr.astype(np.float64).copy()
            tmp[..., 3] = a2c
            ca, crr = to_common_alpha(tmp)
            pat = pattern_energy(ca, crr)
            err = abs(pat - target_pattern)
            if err < best_err:
                best_k, best_err, best_a2, best_pat = float(k), err, a2c, pat
        a2, diag["k"] = best_a2, best_k
        ratio_pat = best_pat / target_pattern
        if ratio_pat < 0.7:
            diag["pattern_flag"] = f"수렴실패↓ {ratio_pat:.2f} (α-only 모델 밖 — 재드로잉/하프톤 의심)"
        elif ratio_pat > 1.3:
            diag["pattern_flag"] = f"수렴실패↑ {ratio_pat:.2f}"
    else:
        a2 = compose(diag["k"])

    diag["sat255"] = float((a2 >= 254).mean() * 100)

    # RGB 잉크 보정: mid(≤0.72) → rim(≥0.88) 반경 보간
    mid = float(np.clip(mid_ratio, *RIM_RATIO_CLAMP))
    rim = float(np.clip(rim_ratio, *RIM_RATIO_CLAMP))
    t = np.clip((rr - 0.72) / 0.16, 0.0, 1.0)
    factor = mid + (rim - mid) * t

    out = arr.astype(np.float64).copy()
    out[..., :3] *= factor[..., None]
    out[..., 3] = a2
    return np.clip(out, 0, 255).astype(np.uint8), diag


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="-rebal")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    catalog = sorted(
        p for p in LENS_DIR.glob("*.png")
        if not any(s in p.name for s in EXCLUDE_SUBSTRINGS)
    )
    pairs = build_pairs(catalog, verbose=True)
    n_paired = sum(1 for p in catalog if p.stem in pairs)
    print(f"대상 {len(catalog)}개 — 쌍 실측 {n_paired} / 정규화 {len(catalog) - n_paired} (v4.2: 농도+잉크색보정+패턴스캔)")

    for p in catalog:
        arr = np.asarray(Image.open(p).convert("RGBA"))
        rr = polar(arr.shape[:2], arr.shape[1] / 2, arr.shape[0] / 2, min(arr.shape[:2]) / 2)
        ours_ink = ink_density(arr[..., 3].astype(np.float64), rr)
        ov = OVERRIDES.get(p.stem, {})

        if p.stem in pairs:
            s_arr, s_rr = sheet_left(pairs[p.stem])
            target_ink = ink_density(s_arr[..., 3].astype(np.float64), s_rr)
            # Codex 리뷰 (e): RGB 잉크색 보정은 림만이 아니라 전역 — mid/rim 각각 실측
            tm, om = ink_lum(s_arr, s_rr, 0.4, 0.72), ink_lum(arr, rr, 0.4, 0.72)
            tr, orim = rim_ink_lum(s_arr, s_rr), rim_ink_lum(arr, rr)
            mid_ratio = tm / om if np.isfinite(tm) and np.isfinite(om) and om > 1 else MID_RATIO_DEFAULT
            rim_ratio = tr / orim if np.isfinite(tr) and np.isfinite(orim) and orim > 1 else RIM_RATIO_DEFAULT
            ca, crr = to_common_alpha(s_arr)
            target_pattern = pattern_energy(ca, crr)     # 쌍별 패턴 에너지 목표
            origin = f"쌍({pairs[p.stem].parent.name}/{pairs[p.stem].stem})"
        else:
            target_ink = min(T_INK, ours_ink * SCALE_MAX)
            mid_ratio, rim_ratio = MID_RATIO_DEFAULT, RIM_RATIO_DEFAULT
            target_pattern = None
            origin = f"정규화(80/{ours_ink:.0f})"

        target_ink = ov.get("target_ink", target_ink)
        mid_ratio = ov.get("mid_ratio", mid_ratio)
        rim_ratio = ov.get("rim_ratio", rim_ratio)
        target_pattern = ov.get("target_pattern", target_pattern)
        if ov:
            origin += "+override"

        pat_tag = f", 패턴목표 {target_pattern:.1f}" if target_pattern else ""
        tag = (f"α {ours_ink:.0f}→{target_ink:.0f}, 잉크색 mid×{np.clip(mid_ratio, *RIM_RATIO_CLAMP):.2f}"
               f"/rim×{np.clip(rim_ratio, *RIM_RATIO_CLAMP):.2f}{pat_tag} ({origin})")
        if args.dry_run:
            print(f"  [dry] {p.name}: {tag}")
            continue
        out, diag = transform_v4(arr, target_ink, mid_ratio, rim_ratio, target_pattern)
        flag = f" ⚠️{diag['pattern_flag']}" if diag["pattern_flag"] else ""
        sat = f" 포화{diag['sat255']:.0f}%" if diag["sat255"] > 3 else ""
        tag += f" k={diag['k']:.2f}{sat}{flag}"
        out_im = Image.fromarray(out)
        if out_im.size != (OUT_SIZE, OUT_SIZE):
            out_im = out_im.resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS)  # 다운샘플만
        out_im.save(p.with_name(p.stem + args.suffix + ".png"))
        print(f"  {p.name}: {tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
