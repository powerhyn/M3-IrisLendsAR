#!/usr/bin/env python3
"""리밸런싱 비교 갤러리 생성 — 실물 PNG(흰 배경) 기준 (NLR-W2 에셋 축).

출력: docs/bench/NLR-W2/rebalance_gallery_<label>.html (자급자족 단일 파일, 브라우저로 열람)
- 섹션 1: 버전 진화 스트립 (대표 5종 × 원본→v1→v2→v3→v4(현행 스크립트)→타겟) — v1~v3는 본
  스크립트가 과거 모델을 재현해 인메모리 생성 (에셋 파일은 항상 현행 버전만 존재)
- 섹션 2: 전체 42종 그리드 (원본 · 현행 -rebal · 타겟 시트)

사용: python3 scripts/generate_rebalance_gallery.py [label]   (기본 label = "latest")
의존: scripts/rebalance_lens_assets.py (변환 로직·쌍 매칭 재사용)
"""
import base64
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("rb", REPO / "scripts/rebalance_lens_assets.py")
rb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rb)

LENS_DIR = rb.LENS_DIR
TH = 176
SHOW = ["claset_never-olive", "claset_doll-choco", "claset_runway-gray",
        "ooha_dew-e-dew-choco", "ooha_travel-laguna-beige"]
V2_EXACT = {"claset_never-olive": 0.53, "claset_lovey-beige": 0.85, "ooha_travel-laguna-beige": 0.69,
            "claset_dusty-grape": 0.86, "envie_plum-black": 0.93, "envie_parfum-bleu": 0.65,
            "claset_doll-choco": 0.46}


def thumb_raw(lens_img) -> str:
    bg = Image.new("RGBA", (TH, TH), (255, 255, 255, 255))
    bg.alpha_composite(lens_img.convert("RGBA").resize((TH, TH), Image.LANCZOS))
    buf = io.BytesIO()
    bg.convert("RGB").save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def v1_curve(pairs):
    def profile(a, rr, nb=20):
        return np.array([a[(rr >= i / nb) & (rr < (i + 1) / nb)].mean()
                         if ((rr >= i / nb) & (rr < (i + 1) / nb)).any() else 0 for i in range(nb)])
    sa, srr = rb.sheet_left(pairs["claset_never-olive"])
    oa = np.asarray(Image.open(LENS_DIR / "claset_never-olive.png").convert("RGBA"))
    orr = rb.polar(oa.shape[:2], oa.shape[1] / 2, oa.shape[0] / 2, min(oa.shape[:2]) / 2)
    tp, op = profile(sa[..., 3].astype(float), srr), profile(oa[..., 3].astype(float), orr)
    c = np.ones(20)
    for i in range(20):
        if op[i] >= 2:
            c[i] = min(tp[i] / op[i], 1.2)
    sm = c.copy()
    for i in range(1, 19):
        sm[i] = (c[i - 1] + 2 * c[i] + c[i + 1]) / 4
    return sm


def apply_version(p: Path, ver: str, pairs, curve):
    arr = np.asarray(Image.open(p).convert("RGBA")).astype(float)
    rr = rb.polar(arr.shape[:2], arr.shape[1] / 2, arr.shape[0] / 2, min(arr.shape[:2]) / 2)
    a = arr[..., 3]
    ours = rb.ink_density(a, rr)
    if ver == "v1":
        arr[..., 3] = np.clip(a * np.interp(rr, (np.arange(20) + .5) / 20, curve), 0, 255)
    elif ver == "v2":
        arr[..., 3] = np.clip(a * V2_EXACT.get(p.stem, float(np.clip(80 / ours, 0.45, 1.0))), 0, 255)
    elif ver == "v3":
        if p.stem in pairs:
            sa, srr = rb.sheet_left(pairs[p.stem])
            s = rb.ink_density(sa[..., 3].astype(float), srr) / ours
        else:
            s = float(np.clip(80 / ours, 0.45, 1.0))
        arr[..., 3] = np.clip(a * s, 0, 255)
    elif ver == "v4":
        raw = np.asarray(Image.open(p).convert("RGBA"))
        if p.stem in pairs:
            sa, srr = rb.sheet_left(pairs[p.stem])
            ti = rb.ink_density(sa[..., 3].astype(float), srr)
            tm, om = rb.ink_lum(sa, srr, 0.4, 0.72), rb.ink_lum(raw, rr, 0.4, 0.72)
            tr, orim = rb.rim_ink_lum(sa, srr), rb.rim_ink_lum(raw, rr)
            mid = tm / om if np.isfinite(tm) and np.isfinite(om) and om > 1 else rb.MID_RATIO_DEFAULT
            rim = tr / orim if np.isfinite(tr) and np.isfinite(orim) and orim > 1 else rb.RIM_RATIO_DEFAULT
            tpat = rb.pattern_energy(*rb.to_common_alpha(sa))
        else:
            ti, mid, rim, tpat = min(80.0, ours * 1.2), rb.MID_RATIO_DEFAULT, rb.RIM_RATIO_DEFAULT, None
        out, _ = rb.transform_v4(raw, ti, mid, rim, tpat)
        return Image.fromarray(out)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def main() -> int:
    label = sys.argv[1] if len(sys.argv) > 1 else "latest"
    catalog = sorted(p for p in LENS_DIR.glob("*.png")
                     if not any(s in p.name for s in rb.EXCLUDE_SUBSTRINGS))
    pairs = rb.build_pairs(catalog)
    curve = v1_curve(pairs)

    versions = [("orig", "원본"), ("v1", "v1 단일커브"), ("v2", "v2 농도정규화"),
                ("v3", "v3 38쌍실측"), ("v4", "현행 스크립트"), ("target", "타겟 시트")]
    evo_html = ""
    for stem in SHOW:
        p = LENS_DIR / f"{stem}.png"
        cells = ""
        for key, lab in versions:
            if key == "orig":
                im = Image.open(p)
            elif key == "target":
                im = Image.fromarray(rb.sheet_left(pairs[stem])[0])
            else:
                im = apply_version(p, key, pairs, curve)
            cells += (f'<figure><img src="data:image/jpeg;base64,{thumb_raw(im)}">'
                      f'<figcaption>{lab}</figcaption></figure>')
        evo_html += f'<div class="strip"><h3>{stem}</h3><div class="six">{cells}</div></div>'
        print("스트립:", stem)

    cards = ""
    for p in catalog:
        name = p.stem
        reb = Image.open(p.with_name(name + "-rebal.png"))
        tcol = (f'<figure><img src="data:image/jpeg;base64,'
                f'{thumb_raw(Image.fromarray(rb.sheet_left(pairs[name])[0]))}">'
                f'<figcaption>타겟</figcaption></figure>'
                if name in pairs else '<figure><div class="ph">시트 없음</div><figcaption>타겟</figcaption></figure>')
        cards += (f'<article class="card"><h3>{name}</h3><div class="tri">'
                  f'<figure><img src="data:image/jpeg;base64,{thumb_raw(Image.open(p))}"><figcaption>원본</figcaption></figure>'
                  f'<figure><img src="data:image/jpeg;base64,{thumb_raw(reb)}"><figcaption>리밸(현행)</figcaption></figure>'
                  f'{tcol}</div></article>')
    print("그리드:", len(catalog))

    html = f'''<title>렌즈 리밸런싱 비교 — {label}</title>
<style>
:root {{ --bg:#171512; --surface:#201d19; --line:#37322a; --text:#e8e2d8; --muted:#9a917f; --accent:#c8a24a; }}
@media (prefers-color-scheme: light) {{ :root {{ --bg:#f5f2ec; --surface:#fff; --line:#e2dccf; --text:#26221b; --muted:#6f6758; --accent:#8a6d1f; }} }}
* {{ box-sizing:border-box; }} body {{ background:var(--bg); color:var(--text); font-family:system-ui,"Apple SD Gothic Neo",sans-serif; margin:0; padding:28px 20px 80px; }}
.wrap {{ max-width:1200px; margin:0 auto; }} h1 {{ font-size:1.35rem; }}
h2 {{ font-size:1rem; border-bottom:1px solid var(--line); padding-bottom:8px; margin:36px 0 16px; }}
.strip {{ margin-bottom:24px; }} .strip h3 {{ font-size:0.88rem; color:var(--accent); margin:0 0 8px; }}
.six {{ display:grid; grid-template-columns:repeat(6,1fr); gap:8px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(330px,1fr)); gap:14px; }}
.card {{ background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }}
.card h3 {{ font-size:0.88rem; margin:0 0 10px; }} .tri {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; }}
figure {{ margin:0; }} figure img {{ width:100%; border-radius:7px; display:block; border:1px solid var(--line); }}
figcaption {{ font-size:0.73rem; color:var(--muted); text-align:center; margin-top:4px; }}
.ph {{ aspect-ratio:1; border:1px dashed var(--line); border-radius:7px; display:flex; align-items:center; justify-content:center; color:var(--muted); font-size:0.75rem; }}
@media (max-width:900px) {{ .six {{ grid-template-columns:repeat(3,1fr); }} }}
</style>
<div class="wrap"><h1>렌즈 리밸런싱 비교 — {label}</h1>
<p style="color:var(--muted);font-size:0.9rem">실물 PNG를 흰 배경에 그대로 표시. 실기기 육안 판정이 정본.</p>
<h2>버전 진화 — 대표 5종</h2>{evo_html}
<h2>전체 — 원본 · 리밸(현행) · 타겟</h2><div class="grid">{cards}</div></div>'''

    out = REPO / f"docs/bench/NLR-W2/rebalance_gallery_{label}.html"
    out.write_text(html)
    print("저장:", out, len(html) // 1024, "KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
