"""Graphs and HTML pages from the pipeline outputs.

Writes to out_dir/report/:
  index.html   overview: QC, ancestry (ADMIXTURE, PCA), deep ancestry, ROH, traits, PRS, ClinVar, PharmCAT, CRAM depth
  qpadm.html   every qpAdm model, CRAM vs CSV, with a filterable table
  figures/     the main graphs as PNG (for sharing / printing)

Each section is skipped when its inputs don't exist yet, so this can run at any stage of the pipeline.
The HTML pages are self-contained (no internet needed), follow the system light/dark setting, and show
details on hover.
"""

import html
import importlib
import json
import math
import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import AUTOSOMES, config_arg, load_config

sys.path.insert(0, str(Path(__file__).parent))
pca_module = importlib.import_module("08_plot_ancestry")

GRCH37_LEN = {  # chromosome lengths (bp)
    "1": 249250621, "2": 243199373, "3": 198022430, "4": 191154276, "5": 180915260, "6": 171115067,
    "7": 159138663, "8": 146364022, "9": 141213431, "10": 135534747, "11": 135006516, "12": 133851895,
    "13": 115169878, "14": 107349540, "15": 102531392, "16": 90354753, "17": 81195210, "18": 78077248,
    "19": 59128983, "20": 63025520, "21": 48129895, "22": 51304566, "X": 155270560,
}
# Validated reference palette (light mode), fixed slot order. Same values as the CSS below.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, MUTED, GRID, TRACK = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#e9e8e3"
SUPERPOP_NAMES = {"AFR": "African", "AMR": "Admixed American", "EAS": "East Asian", "EUR": "European",
                  "SAS": "South Asian"}


# ---------------------------------------------------------------- data

def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def qpadm_results_path(out: Path, name: str | None) -> Path:
    """--qpadm-results, else the fast engine's results (13b) if present, else ADMIXTOOLS' (13)."""
    if name:
        return out / "qpadm" / name
    fast = out / "qpadm" / "qpadm_fast_results.csv"
    return fast if fast.exists() else out / "qpadm" / "qpadm_results.csv"


def load_qpadm(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    split = lambda s: [float(x) for x in str(s).split()] if isinstance(s, str) else []
    df["weights"] = df["weights"].map(split)
    df["se"] = df["se"].map(split)
    one = df["n_sources"] == 1  # older 13_qpadm.py runs stored ADMIXTOOLS' 0.000 for single sources
    df.loc[one, "weights"] = pd.Series([[1.0]] * one.sum(), index=df.index[one], dtype=object)
    df.loc[one, "se"] = pd.Series([[0.0]] * one.sum(), index=df.index[one], dtype=object)
    df["src"] = df["sources"].str.split(" + ", regex=False)
    df["valid"] = df["weights"].map(lambda w: bool(w) and all(0 <= x <= 1 for x in w))
    return df


def load_roh(out: Path) -> pd.DataFrame | None:
    path = out / "roh.hom"
    if not path.exists():
        return None
    return pd.read_csv(path, sep=r"\s+").astype({"CHR": str})


def load_coverage(out: Path) -> dict | None:
    cov_path, hist_path = out / "coverage.tsv", out / "chr1.depth_hist.tsv"
    res = {}
    if cov_path.exists():
        cov = pd.read_csv(cov_path, sep="\t").rename(columns=lambda c: c.lstrip("#"))
        cov["rname"] = cov["rname"].astype(str).str.removeprefix("chr")
        res["per_chrom"] = cov[cov["rname"].isin(GRCH37_LEN)].set_index("rname").reindex(
            [c for c in GRCH37_LEN if c in set(cov["rname"])])
    if hist_path.exists():
        res["hist"] = pd.read_csv(hist_path, sep="\t", header=None, names=["depth", "n"])
    return res or None


def load_pca(cfg: dict):
    out = cfg["out_dir"]
    if not (out / "ancestry.eigenvec").exists():
        return None
    try:
        pcs, _ = pca_module.load_pcs(cfg)
    except FileNotFoundError:
        return None
    return pcs


# ---------------------------------------------------------------- HTML helpers

def esc(x) -> str:
    return html.escape(str(x), quote=True).replace("\n", "&#10;")


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "–"
    return f"{p:.2f}" if p >= 0.1 else f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"


def pct(x: float, digits: int = 0) -> str:
    return f"{x * 100:.{digits}f}%"


def tile(label: str, value: str, note: str = "") -> str:
    note = f'<div class="tile-note">{esc(note)}</div>' if note else ""
    return f'<div class="tile"><div class="tile-label">{esc(label)}</div><div class="tile-value">{esc(value)}</div>{note}</div>'


def legend(items: list[tuple[str, int]]) -> str:
    keys = "".join(f'<span class="key"><i class="sw s{slot}"></i>{esc(name)}</span>' for name, slot in items)
    return f'<div class="legend">{keys}</div>'


def seg_bar(parts: list[tuple[str, float, int, str]], cls: str = "", named: bool = False) -> str:
    """Tall segmented bar: one whole split into parts (name, share, slot, tooltip). Zero parts draw nothing.
    named=True prints the part's name in its block (when colours can't carry identity)."""
    blocks = "".join(
        f'<div class="seg s{slot}" style="flex:{share:.5f} 1 0" tabindex="0" data-tip="{esc(tip)}">'
        f'<span>{esc(name) + " " if named else ""}{pct(share)}</span></div>'
        for name, share, slot, tip in parts if share > 0.0005)
    label = "; ".join(f"{name} {pct(share)}" for name, share, _, _ in parts)
    return f'<div class="segbar {cls}" role="img" aria-label="{esc(label)}">{blocks}</div>'


def hbars(rows: list[tuple[str, float, str, str]], xmax: float | None = None, note: str = "") -> str:
    """Horizontal bars, one series: (label, value, value_text, tooltip)."""
    xmax = xmax or max(v for _, v, _, _ in rows) or 1
    body = "".join(
        f'<div class="hb-label">{esc(label)}</div>'
        f'<div class="hb-track"><div class="hb-bar s1" style="width:{max(v / xmax, 0) * 85:.2f}%;flex:none" '
        f'tabindex="0" data-tip="{esc(tip)}"></div><span class="hb-val">{esc(text)}</span></div>'
        for label, v, text, tip in rows)
    note = f'<div class="chart-note">{esc(note)}</div>' if note else ""
    return f'<div class="hbars">{body}</div>{note}'


def columns(rows: list[tuple[str, float, str]], height: int = 160, y_label: str = "") -> str:
    """Vertical columns, one series: (x label, value, tooltip)."""
    ymax = max(v for _, v, _ in rows) or 1
    ticks = nice_ticks(0, ymax, 4)
    top = ticks[-1]
    grid = "".join(f'<div class="col-grid" style="bottom:{t / top * 100:.2f}%"><span>{fmt_num(t)}</span></div>'
                   for t in ticks)
    cols = "".join(
        f'<div class="col"><div class="col-bar s1" style="height:{v / top * 100:.2f}%" tabindex="0" '
        f'data-tip="{esc(tip)}"></div></div>' for _, v, tip in rows)
    labels = "".join(f'<div class="col-x">{esc(x)}</div>' for x, _, _ in rows)
    y_label = f'<div class="chart-note">{esc(y_label)}</div>' if y_label else ""
    return (f'<div class="cols-wrap"><div class="cols" style="height:{height}px">{grid}{cols}</div>'
            f'<div class="cols-x">{labels}</div></div>{y_label}')


def nice_ticks(lo: float, hi: float, n: int) -> list[float]:
    step = 10 ** math.floor(math.log10((hi - lo) / n or 1))
    for m in (1, 2, 2.5, 5, 10):
        if (hi - lo) / (m * step) <= n:
            step *= m
            break
    return list(np.arange(lo, math.ceil(hi / step) * step + step / 2, step))


def fmt_num(v: float) -> str:
    if abs(v) >= 1e6:
        return f"{v / 1e6:g}M"
    if abs(v) >= 1e4:
        return f"{v / 1e3:g}K"
    return f"{v:,.0f}" if v == int(v) else f"{v:g}"


def table(df: pd.DataFrame, cls: str = "", row_attrs=None) -> str:
    head = "".join(f"<th>{esc(c)}</th>" for c in df.columns)
    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        attrs = row_attrs(i) if row_attrs else ""
        rows.append(f"<tr {attrs}>" + "".join(f"<td>{esc(v)}</td>" for v in r) + "</tr>")
    return f'<div class="table-wrap"><table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def details(summary: str, content: str) -> str:
    return f"<details><summary>{esc(summary)}</summary>{content}</details>"


def section(sec_id: str, title: str, lede: str, body: str) -> str:
    lede = f'<p class="lede">{lede}</p>' if lede else ""
    return f'<section id="{sec_id}"><h2>{esc(title)}</h2>{lede}{body}</section>'


def card(title: str, body: str, sub: str = "") -> str:
    sub = f'<div class="card-sub">{sub}</div>' if sub else ""
    return f'<div class="card"><h3>{esc(title)}</h3>{sub}{body}</div>'


# ---------------------------------------------------------------- charts that need SVG

def pca_svg(pcs: pd.DataFrame, me: pd.Series, x: str, y: str, subset=None, label_by: str = "super_pop",
            highlight: str | None = None, title: str = "") -> str:
    """Scatter of reference samples (gray, or one group in slot 1) with group labels at centroids and you."""
    ref = pcs.dropna(subset=["pop"])
    if subset is not None:
        ref = ref[subset(ref)]
    W, H, pad = 460, 360, 28
    xs = pd.concat([ref[x], pd.Series([me[x]])])
    ys = pd.concat([ref[y], pd.Series([me[y]])])
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    mx, my = (x1 - x0) * 0.05 or 1e-3, (y1 - y0) * 0.05 or 1e-3
    sx = lambda v: pad + (v - x0 + mx) / (x1 - x0 + 2 * mx) * (W - 2 * pad)
    sy = lambda v: H - pad - (v - y0 + my) / (y1 - y0 + 2 * my) * (H - 2 * pad)
    pts = []
    for _, r in ref.iterrows():
        cls = "pt hi" if highlight and r["super_pop"] == highlight else "pt"
        pts.append(f'<circle class="{cls}" cx="{sx(r[x]):.1f}" cy="{sy(r[y]):.1f}" r="2.6" '
                   f'data-tip="{esc(r["IID"])}&#10;{esc(r["pop"])} · {esc(r["super_pop"])}"/>')
    mex, mey = sx(me[x]), sy(me[y])
    # Group labels at centroids, nudged down when they would collide with an earlier label or with "you"
    placed = [(mex + 22, mey, 40)]  # (centre x, baseline y, half width) of the "you" label
    labels = []
    for g, d in sorted(ref.groupby(label_by), key=lambda kv: sy(kv[1][y].mean())):
        name = SUPERPOP_NAMES.get(g, pca_module.POP_NAMES.get(g, g))
        lx, ly, half = sx(d[x].mean()), sy(d[y].mean()) - 6, 4 + 3.5 * len(str(g))
        for _ in range(12):
            if not any(abs(lx - px) < half + ph and abs(ly - py) < 13 for px, py, ph in placed):
                break
            ly += 13
        placed.append((lx, ly, half))
        labels.append(f'<text class="grp" x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle">{esc(g)}'
                      f'<title>{esc(name)} (n={len(d)})</title></text>')
    star = (f'<circle class="me" cx="{mex:.1f}" cy="{mey:.1f}" r="6" tabindex="0" data-tip="You&#10;{x} {me[x]:.4f}, '
            f'{y} {me[y]:.4f}"/><text class="me-label" x="{mex + 10:.1f}" y="{mey + 4:.1f}">you</text>')
    axes = (f'<line class="axis" x1="{pad}" y1="{H - pad}" x2="{W - pad}" y2="{H - pad}"/>'
            f'<line class="axis" x1="{pad}" y1="{pad}" x2="{pad}" y2="{H - pad}"/>'
            f'<text class="ax" x="{W - pad}" y="{H - 8}" text-anchor="end">{x}</text>'
            f'<text class="ax" x="10" y="{pad}" transform="rotate(-90 10 {pad})" text-anchor="end">{y}</text>')
    return (f'<figure class="svg-fig"><svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}">'
            f'{axes}{"".join(pts)}{"".join(labels)}{star}</svg><figcaption>{esc(title)}</figcaption></figure>')


# ---------------------------------------------------------------- page sections: overview

def sec_summary(cfg, qc, anc, roh, cov) -> str:
    tiles = []
    if qc:
        tiles += [tile("SNPs in the raw file", f"{qc['n_variants']:,}", qc["header"].get("method", "")),
                  tile("Inferred sex", qc["inferred_sex"], f"X heterozygosity {qc['het_x_nonpar']:.3f}"),
                  tile("Autosomal heterozygosity", f"{qc['het_autosomal']:.3f}", "share of SNPs that are heterozygous")]
    if anc:
        top = max(anc["admixture"], key=anc["admixture"].get)
        tiles.append(tile("Main ADMIXTURE component", f"{top} {pct(anc['admixture'][top], 1)}", "1000 Genomes superpopulations"))
    if roh is not None:
        tiles.append(tile("Runs of homozygosity", f"{roh['KB'].sum() / 1000:.1f} Mb",
                          f"{len(roh)} segments, F_ROH {roh['KB'].sum() * 1000 / sum(GRCH37_LEN[c] for c in AUTOSOMES):.3f}"))
    if cov and "per_chrom" in cov:
        pc = cov["per_chrom"].loc[[c for c in AUTOSOMES if c in cov["per_chrom"].index]]
        depth = (pc["meandepth"] * pc["endpos"]).sum() / pc["endpos"].sum()
        tiles.append(tile("CRAM mean depth", f"{depth:.2f}×", "autosomes"))
    return f'<div class="tiles">{"".join(tiles)}</div>' if tiles else ""


def sec_ancestry(cfg, anc, pcs) -> str:
    if not anc and pcs is None:
        return ""
    cards = []
    if anc:
        order = ["EUR", "SAS", "EAS", "AFR", "AMR"]
        comps = sorted(anc["admixture"].items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else 9)
        slot = {k: i + 1 for i, k in enumerate(order)}
        parts = [(k, v, slot.get(k, 8), f"{SUPERPOP_NAMES.get(k, k)}\n{pct(v, 1)}") for k, v in comps]
        cards.append(card("Supervised ADMIXTURE", seg_bar(parts) + legend([(SUPERPOP_NAMES.get(k, k), slot.get(k, 8))
                                                                            for k, _ in comps]),
                          "Your genome as a mix of the 1000 Genomes superpopulations. A broad view only: 1000 Genomes "
                          "has no Eastern European groups, see the qpAdm page for finer sources."))
        near = anc["nearest_populations"][:8]
        rows = [(f"{n['pop']} · {n['name']}", n["distance"], f"{n['distance']:.3f}",
                 f"{n['name']} ({n['super_pop']})\ndistance {n['distance']:.4f}") for n in near]
        cards.append(card("Nearest 1000 Genomes populations", hbars(rows, note="Distance in PCA space, weighted by "
                                                                    "eigenvalue. Shorter bar = closer to you."),
                          "Distance from you to each population's centre"))
    body = '<div class="grid2">' + "".join(cards) + "</div>"
    if pcs is not None:
        me = pcs[pcs["IID"] == cfg["sample_id"]].iloc[0]
        a = pca_svg(pcs, me, "PC1", "PC2", label_by="super_pop", highlight="EUR",
                    title="All 2,504 samples, PC1 vs PC2 (Europeans in blue)")
        b = pca_svg(pcs, me, "PC1", "PC2", subset=lambda d: d["super_pop"] == "EUR", label_by="pop",
                    title="Europeans only, PC1 vs PC2 (hover a label for the population)")
        body += card("Principal components", f'<div class="grid2 svg-row">{a}{b}</div>',
                     "Each dot is one 1000 Genomes person; hover for details. Labels sit at each group's centre.")
    return section("ancestry", "Ancestry", "Where your genome sits among modern reference populations.", body)


def deep_best(df: pd.DataFrame, target: str = "me_cram"):
    d = df[(df["analysis"] == "deep") & (df["target"] == target) & df["fits"]]
    if d.empty:
        return None
    n = d["n_sources"].min()
    return d[d["n_sources"] == n].sort_values("p", ascending=False).iloc[0]


def source_slots(cfg, analysis: str, df: pd.DataFrame | None = None) -> dict[str, int] | None:
    """Fixed colour slot per source, in config order. Only sources that appear in a model that fits get one;
    with more than 8 of those, colours can't stay distinct, so return None (bars then name their blocks)."""
    sources = cfg["qpadm"]["analyses"][analysis]["sources"]
    if df is not None:
        a = df[df["analysis"] == analysis]
        fit_any = a.groupby("sources")["fits"].any()
        used = {s for x in a[a["sources"].isin(fit_any[fit_any].index)]["src"] for s in x}
        sources = [s for s in sources if s in used]
    return {s: i + 1 for i, s in enumerate(sources)} if len(sources) <= len(SLOTS) else None


def model_bar(r, slots: dict[str, int] | None, cls: str = "") -> str:
    parts = [(s, w, slots[s] if slots else i + 1, f"{s}\n{pct(w, 1)} ± {pct(e, 1)}")
             for i, (s, w, e) in enumerate(zip(r["src"], r["weights"], r["se"]))]
    return seg_bar(parts, cls, named=slots is None)


def slot_legend(slots: dict[str, int] | None, sources) -> str:
    if slots is None:
        return '<div class="chart-note">Too many sources for distinct colours: each block is named instead.</div>'
    return legend([(s, slots[s]) for s in sorted(set(sources), key=slots.get)])


def sec_deep_teaser(cfg, df) -> str:
    if df is None or "deep" not in set(df["analysis"]):
        return ""
    body = glance_card(cfg, df, "deep", "Simplest deep-ancestry model that fits",
                       'qpAdm with ancient sources. All models: <a href="qpadm.html">qpAdm page</a>.')
    body += rep_brief(cfg, df)
    return section("deep", "Deep ancestry (qpAdm)", "Your genome as a mix of ancient populations.", body)


def rep_brief(cfg, qp: pd.DataFrame) -> str:
    """Overview card: robustness of qpAdm to the random read (20_qpadm_replicates.py)."""
    path = cfg["out_dir"] / "qpadm" / "replicates" / "summary.csv"
    if not path.exists():
        return ""
    s = pd.read_csv(path)
    R = int(s["replicates"].max())
    multi = s[(s["fit_share"] > 0) & (s["n_sources"] > 1)]
    sd_cols = [c for c in s if c.endswith("_sd")]
    sd = np.nanmedian(multi[sd_cols].to_numpy(float))
    jk = np.nanmedian(multi[[c.replace("_sd", "_jk_se") for c in sd_cols]].to_numpy(float))
    ng = [r[f"w{j + 1}_sd"] for _, r in multi.iterrows() for j, src in enumerate(r["sources"].split(" + "))
          if src == "Nganasan"]
    always, never = int((s["fit_share"] == 1).sum()), int((s["fit_share"] == 0).sum())
    flip = len(s) - always - never
    tiles = [tile("Replicate draws", f"{R}", f"{len(s)} models rerun each time"),
             tile("Weight spread between draws", f"{sd * 100:.1f} pts", f"qpAdm's own SE: {jk * 100:.1f} pts"),
             tile("Nganasan share spread", f"± {np.median(ng) * 100:.1f} pts" if ng else "–", "between draws"),
             tile("Verdicts that flip", f"{flip} of {len(s)}", f"{always} always fit, {never} never")]

    def pick(d, n=8):
        d = d.assign(k=d["analysis"].isin(["deep", "proximal"])).sort_values(
            ["k", "n_sources", "p_median"], ascending=[False, True, False])
        return [f"<li><b>{esc(r['sources'])}</b> <span class='muted'>({esc(r['analysis'])})</span>: fits "
                f"{round(r['fit_share'] * R)}/{R}, p {fmt_p(r['p_median'])}</li>" for _, r in d.head(n).iterrows()]
    robust = pick(s[s["fit_share"] == 1], 6)
    m = qp[(qp["target"] == "me_cram") & qp["valid"] & (qp["p"] >= cfg["qpadm"]["p_min"])]
    passed = set(zip(m["analysis"], m["sources"]))
    s_main = s[[k in passed for k in zip(s["analysis"], s["sources"])]]
    fragile = pick(s_main[s_main["fit_share"] < 0.5], 8) or ["<li>none</li>"]
    body = (f'<div class="tiles">{"".join(tiles)}</div><div class="grid2">'
            f'<div><p class="sub-h">Fit in every draw</p><ul>{"".join(robust)}</ul></div>'
            f'<div><p class="sub-h">Fit in the main results, but in fewer than half the draws</p><ul>{"".join(fragile)}</ul></div></div>')
    note = (f"Your data has one random read per SNP. Redrawing that read {R} times moves the ancestry weights far "
            "less than qpAdm's own uncertainty, so the proportions shown are representative. Pass/fail verdicts near "
            'p = 0.01 can flip; trust models in the left list. Details: <a href="qpadm.html#replicates">qpAdm page</a>.')
    return card("Robustness to the random read", body, note)



# ---------------------------------------------------------------- ancient "matches" (15_ibs_ancient.py)

IBS_GROUPS = [("you", "You", 0), ("EUR", "Europeans", 1), ("EAS", "East Asians", 2), ("AFR", "Africans", 3)]
IBS_SUPER = {"FIN": "EUR", "GBR": "EUR", "TSI": "EUR", "IBS": "EUR", "CEU": "EUR", "Russian": "EUR",
             "CHB": "EAS", "JPT": "EAS", "Han": "EAS", "YRI": "AFR", "LWK": "AFR", "Mbuti": "AFR", "you": "you",
             "PJL": "SAS", "PEL": "AMR"}


def ibs_line_svg(df: pd.DataFrame) -> str:
    """Mean number of ≥ min_cm runs per pair against the ancient genome's SNP count (log x), one line per group."""
    edges = [0, 2000, 5000, 10000, 20000, 30000, 40000, 50000, 60000]
    df = df.assign(bin=pd.cut(df["ancient_snps"], edges))
    W, H, L, R, T, B = 960, 340, 52, 24, 14, 42
    xs = np.log10([500, 60000])
    X = lambda v: L + (np.log10(max(v, 500)) - xs[0]) / (xs[1] - xs[0]) * (W - L - R)
    ymax = max(5.0, float(df.groupby(["super", "bin"], observed=True)["n_segments_cm_only"].mean().max()))
    ticks = nice_ticks(0, ymax, 5)
    ymax = max(ymax, ticks[-1])
    Y = lambda v: T + (1 - v / ymax) * (H - T - B)
    g = "".join(f'<line class="gridl" x1="{L}" x2="{W - R}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>'
                f'<text class="ax" x="{L - 6}" y="{Y(t) + 4:.1f}" text-anchor="end">{fmt_num(t)}</text>' for t in ticks)
    g += "".join(f'<text class="ax" x="{X(v):.1f}" y="{H - B + 16}" text-anchor="middle">{lbl}</text>'
                 for v, lbl in [(1000, "1k"), (3000, "3k"), (10000, "10k"), (30000, "30k"), (57000, "57k")])
    g += (f'<line class="axis" x1="{L}" x2="{W - R}" y1="{H - B}" y2="{H - B}"/>'
          f'<text class="ax" x="{(L + W - R) / 2:.0f}" y="{H - 6}" text-anchor="middle">SNPs the ancient genome has '
          f'on the shared panel (log scale)</text>')
    lines = ""
    for key, name, slot in IBS_GROUPS:
        d = df[df["super"] == key]
        if d.empty:
            continue
        m = d.groupby("bin", observed=True).agg(n=("n_segments_cm_only", "mean"), x=("ancient_snps", "median"),
                                                k=("ancient", "nunique"))
        pts = [(X(r.x), Y(r.n), r) for r in m.itertuples()]
        col = "var(--ink)" if key == "you" else f"var(--s{slot})"
        w = 2.5 if key == "you" else 2
        lines += (f'<polyline fill="none" stroke="{col}" stroke-width="{w}" stroke-linejoin="round" '
                  f'points="{" ".join(f"{x:.1f},{y:.1f}" for x, y, _ in pts)}"/>')
        lines += "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{col}" stroke="var(--surface)" stroke-width="1.5" '
                         f'tabindex="0" data-tip="{esc(name)}&#10;ancient genomes with {esc(str(r.Index))} SNPs '
                         f'({r.k})&#10;{r.n:.1f} runs ≥ 7 cM per genome"/>' for x, y, r in pts)
    return (f'<figure class="svg-fig"><svg viewBox="0 0 {W} {H}" role="img" aria-label="Runs ≥ 7 cM vs ancient SNP count">'
            f'{g}{lines}</svg></figure>')


def sec_ibs(out: Path) -> str:
    path = out / "ibs" / "summary.csv"
    if not path.exists():
        return ""
    df = pd.read_csv(path)
    if "n_segments_cm_only" not in df:
        return ""
    st = read_json(out / "ibs" / "settings.json") or {"min_cm": 7, "min_snps": 500}
    df["super"] = df["modern_pop"].map(IBS_SUPER).fillna("other")
    you = df[df["super"] == "you"]
    ctrl = df[df["super"] != "you"]
    r = np.corrcoef(df["n_segments_cm_only"], np.log10(df["ancient_snps"].clip(lower=1)))[0, 1]
    tiles = [tile(f"Segments ≥ {st['min_cm']:g} cM and ≥ {st['min_snps']} SNPs", f"{int(you['n_segments'].sum())}",
                  f"you, with {you['ancient'].nunique()} ancient genomes; "
                  f"{int(ctrl['n_segments'].sum())} for all {ctrl['modern'].nunique()} controls together"),
             tile("Correlation with the ancient genome's SNP count", f"r = {r:.2f}",
                  f"runs ≥ {st['min_cm']:g} cM (no SNP minimum), all pairs")]
    per = df.groupby(["ancient", "super"])["n_segments_cm_only"].mean().unstack()
    info = df.drop_duplicates("ancient").set_index("ancient")
    pair = [(a, b) for a, b in [("Loschbour.DG", "Loschbour.AG"), ("LaBrana1.AG", "LaBrana1.SG")]
            if a in per.index and b in per.index]
    if pair:
        a, b = pair[0]
        tiles.append(tile("Same person, two data files", f"{per.at[a, 'you']:.0f} vs {per.at[b, 'you']:.0f}",
                          f"your runs ≥ {st['min_cm']:g} cM with {a.split('.')[0]}: "
                          f"{info.at[a, 'ancient_snps']:,} vs {info.at[b, 'ancient_snps']:,} SNPs"))
    named = info[info["set"] == "named"].sort_values("ancient_snps")
    rows = []
    for a, i in named.iterrows():
        kind = a.rsplit(".", 1)[-1] if "." in a else ""
        date = f"{i['date_bp'] / 1000:.1f}k BP" if pd.notna(i["date_bp"]) and i["date_bp"] > 0 else ""
        rows.append({"Ancient genome": a.split(".")[0], "Data": kind, "Age": date,
                     "SNPs on panel": f"{i['ancient_snps']:,}",
                     **{n: f"{per.at[a, k]:.1f}" if k in per and pd.notna(per.at[a, k]) else "–" for k, n, _ in IBS_GROUPS}})
    tbl = pd.DataFrame(rows)
    same = {x.split(".")[0] for p_ in pair for x in p_}
    attrs = lambda k: 'class="hl"' if tbl.iloc[k]["Ancient genome"] in same else ""
    body = f'<div class="tiles">{"".join(tiles)}</div>'
    body += card(f"Runs ≥ {st['min_cm']:g} cM: set by the ancient genome's data, not by who you are",
                 ('<div class="legend"><span class="key"><i class="sw" style="background:var(--ink)"></i>You</span>'
                  + "".join(f'<span class="key"><i class="sw s{s}"></i>{esc(n)}</span>' for k, n, s in IBS_GROUPS if s)
                  + "</div>") + ibs_line_svg(df),
                 "Mean number of half-identical runs ≥ 7 cM per ancient genome (SNP minimum dropped), binned by how "
                 "many SNPs that genome has. Fewer SNPs → fewer chances to break a run → more “matches”, for everyone.")
    body += card("The famous genomes from the GEDmatch chart", table(tbl, "dense sortable", attrs),
                 f"Runs ≥ {st['min_cm']:g} cM per ancient genome: you, and the mean of each control group. "
                 "Highlighted: the same person in two AADR files (.DG/.SG = shotgun, .AG = capture).")
    lede = ("GEDmatch-style half-identical segments between you and ancient genomes, with 1000 Genomes and AADR "
            f"people of other ancestries as controls, on the {int(df['shared_snps'].max()):,} SNPs your raw file shares "
            "with the AADR panel. All ancient genomes here are pseudo-haploid (one read per SNP, never heterozygous). "
            "Nobody, you included, has a single segment that passes both thresholds. Without the SNP minimum, the count "
            "of “matches” follows the ancient genome's missing data almost perfectly, and Neanderthals “match” as well "
            "as Stone Age Europeans. The small gap between Africans and Europeans is shared common alleles at the "
            "population level, not genealogy. A long-dead individual's chart says little about relatedness.")
    return section("ibs", "Ancient “matches”: a control experiment", esc(lede), body)


# ---------------------------------------------------------------- Ancestrify-style scan of every ancient (27_ibs_all.py)

def sec_ibs_all(out: Path) -> str:
    work = out / "ibs_all"
    per_path, st = work / "per_ancient.csv", read_json(work / "settings.json")
    if not per_path.exists() or st is None or st.get("limit"):
        return ""
    per = pd.read_csv(per_path)
    people = pd.read_csv(work / "per_person.csv")
    seg = pd.read_csv(work / "segments.csv")
    n_anc, n_ctrl, chroms = len(per), int(st["controls"]), st["chromosomes"]
    partial = len(chroms) < 22
    you = people[people["super"] == "you"].iloc[0]
    ctrl = people[people["super"] != "you"]
    eur = ctrl[ctrl["super"] == "EUR"]
    rank = int((ctrl["segments"] > you["segments"]).sum()) + 1
    thr = f"≥ {st['min_cm']:g} cM and ≥ {st['min_snps']} SNPs"
    matched = per[per["you_n"] > 0]
    only_you = int((matched["controls_with_segment"] == 0).sum())
    tiles = [
        tile(f"Your segments ({thr})", f"{int(you['segments'])}",
             f"with {int(you['ancients_matched'])} of {n_anc:,} ancient individuals"),
        tile("Europeans (controls)", f"{eur['segments'].median():.0f}",
             f"median segments per person; range {eur['segments'].min()}–{eur['segments'].max()} ({len(eur)} people)"),
        tile("East Asians / Africans", " / ".join(f"{ctrl.loc[ctrl['super'] == g, 'segments'].median():.0f}"
                                                  for g in ["EAS", "AFR"]),
             "median segments per person: people with no recent European ancestry"),
        tile("Your place among the controls", f"{rank} of {n_ctrl + 1}",
             "ranked by number of segments, 1 = most"),
    ]
    if len(matched):
        tiles.append(tile("Your matches no control shares", f"{only_you} of {len(matched)}",
                          f"ancients where none of the {n_ctrl} controls has a passing segment"))
    lab = {"you": "▶ You"}
    rows = [(lab.get(r.super, f"{r.pop} ({r.person.split('.')[0]})"), float(r.segments), f"{int(r.segments)}",
             f"{r.pop}: {int(r.segments)} segments with {int(r.ancients_matched)} ancients, {r.total_cm:.0f} cM in all;"
             f"\n{int(r.segments_cm_only):,} runs ≥ {st['min_cm']:g} cM without the SNP minimum")
            for r in people.sort_values("segments", ascending=False).itertuples()]
    body = f'<div class="tiles">{"".join(tiles)}</div>'
    body += card(f"Segments per person over all {n_anc:,} ancient individuals", hbars(rows),
                 f"Half-identical segments {thr} on {st['snps']:,} Human Origins SNPs"
                 + (f" (chromosome{'s' if len(chroms) > 1 else ''} {', '.join(chroms)} only)" if partial else "")
                 + ". Controls are present-day AADR genomes scanned exactly like you.")
    # by period: do the matches follow ancestry or data?
    per["period"] = [classify_period(g, d) for g, d in zip(per["group"], per["date_bp"])]
    cols = [c for c in ["EUR_n_mean", "EAS_n_mean", "AFR_n_mean"] if c in per]
    by = per.groupby("period", observed=True).agg(n=("id", "size"), snps=("ancient_snps", "median"), you=("you_n", "sum"),
                                                  **{c: (c, "sum") for c in cols})
    by = by.reindex([p for p in PERIOD_NAMES if p in by.index])
    tb = pd.DataFrame({"Period": by.index, "Ancients": [f"{v:,}" for v in by["n"]],
                       "Median SNPs": [f"{v:,.0f}" for v in by["snps"]], "You": by["you"].astype(int),
                       **{f"{c[:3]} mean": [f"{v:.1f}" for v in by[c]] for c in cols}})
    body += card("By period", table(tb, "dense"),
                 "Segments summed over the ancient individuals of each period: you, and the mean of the European, "
                 "East Asian and African controls. If the matches reflected your ancestry, “You” would stand out "
                 "against the East Asian and African columns.")
    if len(matched):
        ys = seg[seg["pop"] == "you"].sort_values("length_cm", ascending=False).drop_duplicates("ancient")
        top = matched.merge(ys[["ancient", "chrom", "start_cm", "end_cm", "snps"]], left_on="id", right_on="ancient",
                            how="left").sort_values(["you_cm", "you_longest_cm"], ascending=False).head(40)
        tt = pd.DataFrame({
            "Ancient": top["iid"], "Group": top["group"], "Age": [age_txt(v) for v in top["date_bp"]],
            "Country": top["country"], "Segments": top["you_n"], "cM": [f"{v:.1f}" for v in top["you_cm"]],
            "Longest": [f"chr{int(c)} {a:.0f}–{b:.0f} cM, {int(n)} SNPs" for c, a, b, n in
                        zip(top["chrom"], top["start_cm"], top["end_cm"], top["snps"])],
            "f3 rank": [f"{int(v):,}" for v in top["rank"]],
            "Controls matching": [f"{int(v)} of {n_ctrl}" for v in top["controls_with_segment"]]})
        body += card("Your matches", table(tt, "dense sortable"),
                     f"Ancient individuals with at least one segment {thr}, longest first. “Controls matching”: how "
                     "many of the present-day controls also have a passing segment with the same ancient genome; "
                     "“f3 rank”: the individual's place in your outgroup-f3 ranking (1 = most shared drift).")
    body += card("How to read this", (
        "<p>An IBS segment means only that no SNP in the stretch contradicts sharing (you AA, them BB). Ancient "
        "genomes are mostly one read per SNP and miss many SNPs, so a stretch can look shared simply because few "
        "SNPs were tested; similar common haplotypes do the rest. The marker minimum removes most of these, and "
        "the controls show what is left over by chance. A match that controls of every ancestry share is not "
        "evidence of ancestry; a match only you have is a candidate worth checking with real IBD detection "
        "(ancIBD, which models genotype uncertainty and needs imputed ancient genomes), not proof of descent.</p>"))
    lede = (f"The scan ancient-DNA “match” services run: your genome against all {n_anc:,} ancient individuals of "
            f"your f3 ranking, counting half-identical segments {thr}, with {n_ctrl} present-day people of "
            "European, East Asian and African ancestry scanned the same way. Your genotypes are the GLIMPSE "
            f"imputation of your sequencing reads (calls with probability ≥ {st['min_gp']})"
            + (f"; so far only chromosome{'s' if len(chroms) > 1 else ''} {', '.join(chroms)} "
               f"{'are' if len(chroms) > 1 else 'is'} imputed; rerun 27_ibs_all.py when all 22 are done" if partial else "") + ".")
    return section("ibs_all", "Ancestrify-style scan of every ancient individual", esc(lede), body)


# ---------------------------------------------------------------- outgroup f3 with ancient individuals (18_f3_ancient.py)

# Archaeological period from the AADR group label (its period tokens: _EBA, _Mesolithic, _IronGates, _Viking, ...), the
# individual's date where the label has none. Labels beat fixed date cut-offs, which would put Anatolian farmers
# (8.3k years) in the Mesolithic. With several tokens (Sweden_LN_Viking) the one closest to the date wins.
PERIODS = [("Palaeolithic", "before ~11,700 years ago"), ("Mesolithic", "hunter-gatherers after the Ice Age"),
           ("Neolithic", "first farmers"), ("Copper Age", "Eneolithic, Chalcolithic, Bell Beaker"),
           ("Bronze Age", "incl. Yamnaya, Corded Ware"), ("Iron Age", "incl. Roman and Late Antiquity"),
           ("Medieval", "Migration period to ~1500 CE"), ("Post-medieval", "after ~1500 CE")]
PERIOD_NAMES = [p for p, _ in PERIODS]
PERIOD_TOKENS = [
    ("Palaeolithic", r"UP|EUP|LUP|Paleolithic|Palaeolithic|E[pP]i[pP]al?eolithic|Magdalenian|Gravettian|Aurignacian|"
                     r"Natufian|Azilian|Epigravettian|LGM"),
    ("Mesolithic", r"Mesolithic|Meso|IronGates|EMesolithic|LMesolithic"),
    ("Neolithic", r"N|EN|MN|LN|MLN|ENMN|MNLN|EMN|Neolithic|PPN|PPNA|PPNB|PPNC|PN|LBK|TRB|GlobularAmphora|PittedWare"),
    ("Copper Age", r"C|EC|LC|MC|EarlyC|LateC|MiddleC|Chalcolithic|Eneolithic|CA|ChL|Copper|BellBeaker|Afanasievo"),
    ("Bronze Age", r"BA|EBA|MBA|LBA|MLBA|EMBA|Bronze"),
    ("Iron Age", r"IA|EIA|MIA|LIA|PreRoman|Roman|EarlyRoman|LateRoman|ImperialRoman|Hellenistic|Antiquity|"
                 r"LateAntiquity|Classical|Imperial|Republican|Punic|Etruscan|LaTene|Hallstatt|Scythian|Sarmatian|Xiongnu"),
    ("Medieval", r"Medieval|EarlyMedieval|EMedieval|HighMedieval|LateMedieval|Viking|VikingAge|Migration|"
                 r"MigrationPeriod|Avar|EarlyAvar|LateAvar|MiddleAvar|MiddleLateAvar|Conqueror|Saxon|AngloSaxon|Slav|"
                 r"EarlySlav|Langobard|Lombards|Merovingian|Byzantine|Islamic|GoldenHorde|Mongol|Khazar|Norse|EarlyNorse|"
                 r"LateNorse|Crusader|EarlyArpadian|Arpadian"),
    ("Post-medieval", r"EarlyModern|PostMedieval|Modern|Colonial|Industrial|Napoleonic")]
PERIOD_RX = [(name, re.compile(rf"^(?:{rx})$")) for name, rx in PERIOD_TOKENS]
PERIOD_DATE = [(11700, "Palaeolithic"), (8000, "Mesolithic"), (5000, "Neolithic"), (2800, "Bronze Age"),
               (1500, "Iron Age"), (450, "Medieval")]
# Kept for the ancient page's "by period" text: name -> span shown in its headings
F3_PERIODS = [(None, None, name, span) for name, span in PERIODS]


def period_by_date(bp: float) -> str:
    for lim, name in PERIOD_DATE:
        if bp > lim:
            return name
    return "Post-medieval"


def classify_period(group: str, bp: float) -> str:
    by_date = period_by_date(bp) if np.isfinite(bp) else None
    hits = {name for t in re.split(r"[_\-.]", str(group)) for name, rx in PERIOD_RX if rx.match(t)}
    if not hits and re.search(r"(^|_)HG($|_)", str(group)):
        hits = {"Palaeolithic" if bp > 11700 else "Mesolithic"}
    if not hits:
        return by_date or "Post-medieval"
    if by_date is None or len(hits) == 1:
        return min(hits, key=PERIOD_NAMES.index)
    k = PERIOD_NAMES.index(by_date)
    return min(hits, key=lambda h: (abs(PERIOD_NAMES.index(h) - k), PERIOD_NAMES.index(h)))


REGIONS = {
    "Eastern Europe": ["Ukraine", "Belarus", "Moldova", "Crimea"],
    "Baltic": ["Estonia", "Latvia", "Lithuania"],
    "Scandinavia & North Atlantic": ["Denmark", "Sweden", "Norway", "Finland", "Iceland", "Faroe Islands", "Greenland"],
    "Britain & Ireland": ["United Kingdom", "Ireland", "Channel Islands"],
    "Western Europe": ["France", "Belgium", "Netherlands", "Luxembourg"],
    "Central Europe": ["Germany", "Poland", "Czechia", "Slovakia", "Austria", "Switzerland", "Hungary"],
    "Iberia": ["Spain", "Portugal", "Gibraltar"],
    "Italy & Malta": ["Italy", "Malta"],
    "Balkans": ["Serbia", "Croatia", "Romania", "Bulgaria", "Greece", "Albania", "Montenegro", "North Macedonia",
                "Slovenia", "Bosnia-Herzegovina"],
    "Caucasus": ["Armenia", "Georgia", "Azerbaijan"],
    "Anatolia & Near East": ["Turkey", "Israel", "Lebanon", "Jordan", "Syria", "Iraq", "Iran", "Yemen", "Bahrain"],
    "Central & South Asia": ["Kazakhstan", "Kyrgyzstan", "Turkmenistan", "Uzbekistan", "Tajikistan", "Afghanistan",
                             "Pakistan", "India", "Nepal"],
    "East & Southeast Asia": ["China", "Mongolia", "Republic of Korea", "Japan", "Taiwan", "Vietnam", "Thailand", "Laos",
                              "Malaysia", "Indonesia"],
    "Africa": ["Kenya", "Tunisia", "Sudan", "Morocco", "South Africa", "Tanzania", "Malawi", "Algeria", "Cameroon",
               "Democratic Republic of the Congo", "Botswana", "Saint Helena", "Egypt", "Ethiopia", "Uganda", "Zambia",
               "Canary Islands"],
    "Oceania": ["Guam", "Papua New Guinea", "Vanuatu", "Palau", "Federated States of Micronesia", "Australia", "Tonga",
                "Solomon Islands", "French Polynesia"],
}
REGION_OF = {c: r for r, cs in REGIONS.items() for c in cs}
REGION_NAMES = ["Eastern Europe", "Baltic", "Scandinavia & North Atlantic", "Britain & Ireland", "Western Europe",
                "Central Europe", "Iberia", "Italy & Malta", "Balkans", "Caucasus", "Anatolia & Near East",
                "Siberia", "Central & South Asia", "East & Southeast Asia", "Africa", "Americas", "Oceania", "Unknown"]


def region_of(country: str, lat: float, lon: float) -> str:
    if country == "Russia":
        if np.isfinite(lon) and lon >= 60:
            return "Siberia"
        if np.isfinite(lat) and lat < 45 and 37 <= lon <= 50:
            return "Caucasus"
        return "Eastern Europe"
    if country in REGION_OF:
        return REGION_OF[country]
    return "Unknown" if country in ("..", "") or pd.isna(country) else "Americas"


# Archaeological cultures for the map's population filter: name, regex on the AADR group label
CULTURES = [("Hunter-gatherers (all)", r"(^|_)HG($|_)|Mesolithic|IronGates|Paleolithic|Palaeolithic|Magdalenian"),
            ("Iron Gates", r"IronGates"), ("Magdalenian", r"Magdalenian"), ("Gravettian", r"Gravettian"),
            ("Natufian", r"Natufian"), ("Pitted Ware", r"PittedWare|_PWC"), ("Comb Ceramic", r"CombCeramic|_CCC"),
            ("LBK", r"LBK"), ("Funnel Beaker", r"TRB|FunnelBeaker"), ("Globular Amphora", r"GlobularAmphora|_GAC"),
            ("Trypillia", r"Trypillia|Tripolye|Cucuteni"), ("Khvalynsk", r"Khvalynsk"), ("Yamnaya", r"Yamnaya"),
            ("Afanasievo", r"Afanasievo"), ("Corded Ware", r"CordedWare|Corded_Ware|_CWC"),
            ("Bell Beaker", r"BellBeaker|Bell_Beaker|_BB($|_)"), ("Fatyanovo", r"Fatyanovo"), ("Catacomb", r"Catacomb"),
            ("Únětice", r"Unetice"), ("Sintashta", r"Sintashta"), ("Srubnaya", r"Srubnaya"),
            ("Andronovo", r"Andronovo|Alakul|Fedorovo"), ("Okunevo", r"Okunevo"), ("Hallstatt", r"Hallstatt"),
            ("La Tène", r"LaTene"), ("Scythian", r"Scyth"), ("Sarmatian", r"Sarmat"), ("Xiongnu", r"Xiongnu"),
            ("Roman", r"Roman"), ("Huns", r"Hun($|_|nic)"), ("Avars", r"Avar"), ("Anglo-Saxon", r"Saxon"),
            ("Langobards", r"Langobard|Lombard"), ("Slavs", r"Slav"), ("Viking", r"Viking"),
            ("Hungarian Conquerors", r"Conqueror"), ("Golden Horde", r"GoldenHorde"), ("Mongol era", r"Mongol")]
CULTURE_RX = [(n, re.compile(rx)) for n, rx in CULTURES]


ALTAI_BOX = {"lat": (48, 56), "lon": (78, 100)}   # Altai, Altai Krai, Tuva, Minusinsk basin


def load_f3(out: Path) -> pd.DataFrame | None:
    path = out / "f3" / "outgroup_f3_ancient.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["period"] = pd.Categorical([classify_period(g, b) for g, b in zip(df["group"], df["date_bp"])],
                                  categories=PERIOD_NAMES, ordered=True)
    df["region"] = [region_of(c, la, lo) for c, la, lo in zip(df["country"], df["lat"], df["lon"])]
    df["altai"] = df["lat"].between(*ALTAI_BOX["lat"]) & df["lon"].between(*ALTAI_BOX["lon"])
    top = df.iloc[0]
    # tied with the closest: difference to it within 2 combined standard errors
    df["tied_top"] = (top["f3"] - df["f3"]) < 2 * np.sqrt(top["se"] ** 2 + df["se"] ** 2)
    return df


def f3_groups(df: pd.DataFrame, min_n: int = 3) -> pd.DataFrame:
    """Inverse-variance mean f3 per AADR group label (>= min_n individuals): far less noisy than single genomes."""
    w = 1 / df["se"] ** 2
    g = df.assign(w=w, wf=w * df["f3"]).groupby("group").agg(
        n=("f3", "size"), w=("w", "sum"), wf=("wf", "sum"), date_bp=("date_bp", "median"),
        country=("country", lambda x: x.mode().iat[0]), altai=("altai", "mean"),
        period=("period", lambda x: x.mode().iat[0]))
    g = g[g["n"] >= min_n].assign(f3=lambda x: x["wf"] / x["w"], se=lambda x: 1 / np.sqrt(x["w"]))
    g = g.sort_values("f3", ascending=False).reset_index()
    g["rank"] = np.arange(1, len(g) + 1)
    g["period"] = pd.Categorical(g["period"].astype(str), categories=PERIOD_NAMES, ordered=True)
    return g.assign(iid=g["group"], group=g["n"].map(lambda n: f"{n} people"), locality="", mt_hg="",
                    snps_shared=0, altai=g["altai"] > 0.5)


def age_txt(bp: float) -> str:
    if not np.isfinite(bp):
        return ""
    return f"{bp / 1000:.1f}k yrs" if bp >= 1000 else f"{bp:.0f} yrs"


def f3_dots(rows: pd.DataFrame, lo: float, hi: float, n_label: int = 60) -> str:
    """One row per individual: f3 dot ± 1.96 SE on a zoomed axis."""
    pos = lambda v: (min(max(v, lo), hi) - lo) / (hi - lo) * 100
    ticks = nice_ticks(lo, hi, 5)
    grid = "".join(f'<div class="pp-grid" style="left:{pos(t):.2f}%"></div>' for t in ticks if lo <= t <= hi)
    out = []
    for _, r in rows.iterrows():
        lab = f"{r['iid']} · {r['group']}"
        place = f"{esc(r['locality'])}, {esc(r['country'])}" if r["locality"] else esc(r["country"])
        tip = (f"{esc(r['iid'])} ({esc(r['group'])})&#10;{place}&#10;{age_txt(r['date_bp'])} ago"
               f"&#10;f3 = {r['f3']:.4f} ± {r['se']:.4f}&#10;rank {int(r['rank'])}"
               + (f" · {int(r['snps_shared']):,} SNPs" if r["snps_shared"] else "")
               + (f"&#10;mtDNA {esc(r['mt_hg'])}" if str(r["mt_hg"]) not in ("..", "nan", "") else ""))
        cls = "s2" if r.get("altai") else "s1"
        out.append(f'<div class="pp-label" title="{esc(lab)}">{esc(lab[:n_label])} '
                   f'<span class="muted">{age_txt(r["date_bp"])}, {esc(r["country"])}</span></div>'
                   f'<div class="fp-track">{grid}<div class="fp-ci {cls}" style="left:{pos(r.f3 - 1.96 * r.se):.2f}%;'
                   f'width:{pos(r.f3 + 1.96 * r.se) - pos(r.f3 - 1.96 * r.se):.2f}%;top:50%"></div>'
                   f'<div class="fp-dot {cls}" style="left:{pos(r.f3):.2f}%;top:50%" tabindex="0" data-tip="{tip}"></div></div>')
    axis = "".join(f'<span style="left:{pos(t):.2f}%">{t:.3f}</span>' for t in ticks if lo <= t <= hi)
    return f'<div class="pp fp">{"".join(out)}<div></div><div class="pp-axis">{axis}</div></div>'


def f3_age_svg(df: pd.DataFrame, n_top: int = 20) -> str:
    """f3 against age (log scale) for every individual; closest ones and the Altai region highlighted."""
    W, H, L, R, T, B = 960, 360, 56, 20, 14, 44
    x0, x1 = np.log10(150), np.log10(120000)
    lo, hi = np.nanpercentile(df["f3"], 0.5), df["f3"].max()
    pad = (hi - lo) * 0.04
    lo, hi = lo - pad, hi + pad
    X = lambda v: L + (np.log10(np.clip(v, 150, 120000)) - x0) / (x1 - x0) * (W - L - R)
    Y = lambda v: T + (1 - (np.clip(v, lo, hi) - lo) / (hi - lo)) * (H - T - B)
    g = ""
    for t in nice_ticks(lo, hi, 5):
        if lo <= t <= hi:
            g += (f'<line class="gridl" x1="{L}" x2="{W - R}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>'
                  f'<text class="ax" x="{L - 6}" y="{Y(t) + 4:.1f}" text-anchor="end">{t:.3f}</text>')
    for v, lbl in [(300, "300"), (1000, "1k"), (3000, "3k"), (10000, "10k"), (30000, "30k"), (100000, "100k")]:
        g += f'<text class="ax" x="{X(v):.1f}" y="{H - B + 16}" text-anchor="middle">{lbl}</text>'
    g += (f'<line class="axis" x1="{L}" x2="{W - R}" y1="{H - B}" y2="{H - B}"/>'
          f'<text class="ax" x="{(L + W - R) / 2:.0f}" y="{H - 6}" text-anchor="middle">years before present (log scale)</text>')
    bg = df.iloc[n_top:]
    bg = bg[~bg["altai"]]
    dots = "".join(f'<circle cx="{X(a):.0f}" cy="{Y(f):.0f}" r="2"/>' for a, f in zip(bg["date_bp"], bg["f3"]))
    hl = ""
    for cls, d in (("s2", df[df["altai"]]), ("s1", df.head(n_top))):
        for _, r in d.iterrows():
            hl += (f'<circle cx="{X(r.date_bp):.1f}" cy="{Y(r.f3):.1f}" r="4" style="fill:var(--{cls[0]}{cls[1:]})" '
                   f'stroke="var(--surface)" stroke-width="1" tabindex="0" data-tip="{esc(r.iid)} ({esc(r.group)})&#10;'
                   f'{esc(r.country)}, {age_txt(r.date_bp)} ago&#10;f3 = {r.f3:.4f} ± {r.se:.4f} (rank {int(r["rank"])})"/>')
    return (f'<figure class="svg-fig"><svg viewBox="0 0 {W} {H}" role="img" aria-label="f3 against age">{g}'
            f'<g style="fill:var(--muted);opacity:.28">{dots}</g>{hl}</svg></figure>')


def f3_table(d: pd.DataFrame, cls: str = "dense sortable") -> str:
    t = pd.DataFrame({"Rank": d["rank"].astype(int), "Individual": d["iid"], "Group": d["group"],
                      "Place": d["locality"].astype(str).str.split("(").str[0].str.strip() + ", " + d["country"].astype(str),
                      "Age": d["date_bp"].map(age_txt), "f3": d["f3"].map(lambda v: f"{v:.4f}"),
                      "± SE": d["se"].map(lambda v: f"{v:.4f}"), "SNPs": d["snps_shared"].map(lambda v: f"{int(v):,}"),
                      "mtDNA": d["mt_hg"].replace({"..": "", "nan": ""}).fillna("")})
    return table(t, cls)



def f3_specific(df: pd.DataFrame, n_show: int = 15) -> str:
    """Your f3 relative to the control Eastern Europeans', per ancient group, centred on the median group.
    Genotype errors in your calls shrink all your f3 by about the same factor, which the ratio and centring remove."""
    if "f3_ctrl" not in df:
        return ""
    d = df[df["f3_ctrl"] > 0].assign(ratio=lambda x: x["f3"] / x["f3_ctrl"], w=lambda x: (x["f3_ctrl"] / x["se"]) ** 2)
    g = d.assign(wr=d["w"] * d["ratio"]).groupby("group").agg(n=("ratio", "size"), w=("w", "sum"), wr=("wr", "sum"),
                                                                date_bp=("date_bp", "median"),
                                                                country=("country", lambda x: x.mode().iat[0]))
    g = g[g["n"] >= 3].assign(ratio=lambda x: x["wr"] / x["w"], se=lambda x: 1 / np.sqrt(x["w"]))
    med = g["ratio"].median()
    g["rel"] = g["ratio"] / med - 1
    g["rel_se"] = g["se"] / med
    g = g.sort_values("rel", ascending=False).reset_index()
    pick = pd.concat([g.head(n_show), g.tail(n_show)])
    lo = (pick["rel"] - 2 * pick["rel_se"]).min()
    hi = (pick["rel"] + 2 * pick["rel_se"]).max()
    pos = lambda v: (min(max(v, lo), hi) - lo) / (hi - lo) * 100
    ticks = nice_ticks(lo, hi, 6)
    grid = "".join(f'<div class="pp-grid" style="left:{pos(t):.2f}%"></div>' for t in ticks if lo <= t <= hi)
    grid += f'<div class="fp-mean" style="left:{pos(0):.2f}%"></div>'
    rows = []
    for i, (_, r) in enumerate(pick.iterrows()):
        if i == n_show:
            rows.append('<div class="pp-label f4-cat">…</div><div></div>')
        cls = "s1" if r.rel > 0 else "s2"
        rows.append(f'<div class="pp-label">{esc(r.group)} <span class="muted">{age_txt(r.date_bp)}, {esc(r.country)}, '
                    f'n={int(r.n)}</span></div><div class="fp-track">{grid}'
                    f'<div class="fp-ci {cls}" style="left:{pos(r.rel - 1.96 * r.rel_se):.2f}%;width:'
                    f'{pos(r.rel + 1.96 * r.rel_se) - pos(r.rel - 1.96 * r.rel_se):.2f}%;top:50%"></div>'
                    f'<div class="fp-dot {cls}" style="left:{pos(r.rel):.2f}%;top:50%" tabindex="0" data-tip="'
                    f'{esc(r.group)}&#10;your f3 / controls\' f3 = {r.ratio:.4f} ({r.rel * 100:+.2f}% vs median group)'
                    f'&#10;{int(r.n)} individuals"></div></div>')
    axis = "".join(f'<span style="left:{pos(t):.2f}%">{t * 100:+.1f}%</span>' for t in ticks if lo <= t <= hi)
    chart = (f'<div class="pp fp">{"".join(rows)}<div></div><div class="pp-axis">{axis}</div></div>'
             f'<div class="chart-note">{len(g):,} ancient groups with ≥ 3 people. 0 = the median group. Top: groups you '
             f'share relatively more with than typical Russians, Belarusians and Ukrainians do; bottom: less.</div>')
    # With many groups, the largest |Z| expected by chance alone (two-sided, 5% family-wise)
    zmax = float(np.abs((g["ratio"] - med) / g["se"]).max())
    z_chance = float(np.sqrt(2 * np.log(len(g) / 0.05)))
    shift = (med - 1) * 100
    if zmax < z_chance:
        verdict = (f'<p class="status good"><b>No ancient group stands out.</b> The largest deviation is |Z| = {zmax:.1f}; '
                   f'with {len(g):,} groups, chance alone gives up to about {z_chance:.1f}. The groups at the top of the list '
                   f'(scattered across Africa, Tibet and the Americas) confirm it is noise. Outgroup f3 cannot tell you apart '
                   f'from a typical Eastern European; the f4 test on the overview page is the sensitive one.</p>'
                   f'<p class="chart-note">Your f3 is {abs(shift):.1f}% {"lower" if shift < 0 else "higher"} than the controls\' '
                   f'for every ancient group alike: genotype errors in low-coverage calls, removed by the ratio.</p>')
        return verdict + details("Show the groups at both ends anyway", chart)
    return (f'<p class="status"><b>Largest deviation |Z| = {zmax:.1f}</b> (chance level ≈ {z_chance:.1f}).</p>' + chart)


def f3_page(df: pd.DataFrame) -> str:
    n = len(df)
    top = df.head(30)
    lo = (top["f3"] - 2.2 * top["se"]).min()
    hi = (top["f3"] + 2.2 * top["se"]).max()
    tied = int(df["tied_top"].sum())
    intro = ("<p>Outgroup f3(Mbuti; you, X) measures the genetic drift you share with ancient individual X since both "
             "lineages split from an African outgroup: the higher, the more of your history you share with that person's "
             "population. It is the valid version of GEDmatch's “Ancient Calculator”. It says nothing about direct descent; "
             "and with one genome on each side, neighbouring values overlap. You enter as <code>me_cram</code> "
             f"(one read per SNP, like the ancient genomes). {n:,} ancient individuals with ≥ 30,000 shared SNPs.</p>")
    secs = [section("about", "How to read this", "", intro + (
        "<p><b>Why hunter-gatherers come out on top.</b> f3 counts shared drift, not recent kinship. About half of your "
        "ancestry goes back to Western and Eastern hunter-gatherers (qpAdm), and those small, isolated populations "
        "carried a lot of drift that you inherited. Northeast Europeans therefore share the most f3 with Mesolithic "
        "Baltic, Scandinavian and Iron Gates foragers (Lazaridis et al. 2014 saw the same), followed by Baltic, "
        "Viking-age and Early Medieval groups that are closer to you in time. Read the ranking as “whose population "
        "history overlaps most with yours”, not “your ancestor”.</p>"))]
    grp = f3_groups(df)
    gt = grp.head(30)
    secs.append(section("groups", "Closest ancient populations",
                        f"Mean of the individuals in each AADR group with at least 3 people ({len(grp):,} groups).",
                        card("Top 30 populations", f3_dots(gt, (gt["f3"] - 2.2 * gt["se"]).min(),
                                                           (gt["f3"] + 2.2 * gt["se"]).max()),
                             "Dot = precision-weighted mean f3, line = 95% interval. Hover for details.")))
    spec = f3_specific(df)
    if spec:
        secs.append(section("specific", "What is specific to you", (
            "The same f3 was computed for 20 AADR Russians, Belarusians and Ukrainians (one random allele per SNP, like "
            "you). Dividing yours by theirs removes what all Eastern Europeans share and leaves where you differ. Single "
            "genomes differ only slightly, so read this as a direction, not a ranking."), card(
            "Your f3 relative to typical Eastern Europeans", spec)))
    secs.append(section("closest", "Closest single individuals",
                        f"Single genomes are noisy: {tied} individuals are statistically tied with the first one "
                        "(within 2 standard errors), so the exact order inside the top 30 means little. A single individual far above "
                        "its own group (e.g. TOR.8 vs TOR.6 from the same site) more likely carries some modern-European "
                        "contamination than special closeness: contamination makes an ancient genome look like you.",
                        card("Top 30 individuals", f3_dots(top, lo, hi),
                             "Dot = f3, line = 95% interval. Orange: Altai/southern Siberia. Hover for place, date, haplogroup.")))
    rows = []
    for lo_, hi_, name, span in F3_PERIODS:
        d = df[df["period"] == name]
        if d.empty:
            continue
        gp = grp[grp["period"] == name].head(3)
        gl = "; ".join(f"{r.iid} ({r.f3:.4f})" for r in gp.itertuples())
        rows.append(f'<h3 class="sub-h">{esc(name)} <span class="muted">({esc(span)}, {len(d):,} individuals)</span></h3>'
                    + (f'<p class="chart-note">Closest populations: {esc(gl)}</p>' if gl else "")
                    + f3_table(d.head(5), "dense"))
    secs.append(section("periods", "Closest in each period", "The five closest individuals of each age.", card(
        "By period", "".join(rows))))
    secs.append(section("time", "f3 through time", "", card(
        "Every ancient individual", f3_age_svg(df),
        "Grey: all individuals. Blue: the 20 closest. Orange: Altai and southern Siberia. Shared drift rises toward the "
        "present for people whose populations contributed to yours.")))
    alt = df[df["altai"]]
    if len(alt):
        ag = f3_groups(alt, 2).head(3)
        recent = alt[alt["date_bp"] < 3000]
        secs.append(section("altai", "Altai and southern Siberia", esc(
            "Ancient people of the region where your family settled about a century ago. The closest of them are "
            + ", ".join(ag["iid"]) + f" (best individual rank {int(alt['rank'].min()):,} of {n:,}): Steppe-derived "
            "Bronze Age groups that came from the west. Iron Age and later locals"
            + (f" (best rank {int(recent['rank'].min()):,})" if len(recent) else "")
            + " carry East Asian ancestry you don't, so they rank far lower."),
            card(f"{len(alt):,} individuals (ranks among all {n:,})", f3_table(alt.head(40)))))
    secs.append(section("all", "Top 500", "", card("Sortable", f3_table(df.head(500)))))
    nav = [("about", "How to read"), ("groups", "Populations")] + ([("specific", "Specific to you")] if "f3_ctrl" in df else []) + [
           ("closest", "Individuals"), ("periods", "By period"),
           ("time", "Through time")]
    nav += ([("altai", "Altai")] if len(alt) else []) + [("all", "Top 500")]
    return page("Ancient individuals closest to you", "Outgroup f3(Mbuti; you, X), AADR v66", nav, "".join(secs), back=True)


def sec_f3_teaser(df: pd.DataFrame | None) -> str:
    if df is None:
        return ""
    grp = f3_groups(df)
    best = grp.groupby("period", observed=True).head(1).set_index("period")
    rows = [(f"{p}: {r.iid}", r.f3, f"{r.f3:.4f}", f"{r.iid}, {r.country}\n{age_txt(r.date_bp)} ago, {r.n} people\n"
             f"f3 = {r.f3:.4f} ± {r.se:.4f}, rank {int(r['rank'])} of {len(grp):,} populations") for p, r in best.iterrows()]
    base = min(v for _, v, _, _ in rows) - 0.002
    body = card("Closest ancient population in each period", hbars([(a, v - base, t, tip) for a, v, t, tip in rows]),
                f'Outgroup f3(Mbuti; you, X), mean per AADR group, {len(df):,} individuals (bars start at {base:.3f}; '
                'hunter-gatherer groups score high because about half your ancestry comes from them). '
                'Details: <a href="ancient.html">ancient individuals page</a> and the '
                '<a href="map.html">interactive map</a>.')
    return section("f3", "Closest ancient populations (outgroup f3)", "", body)



# ---------------------------------------------------------------- interactive map of the ancient individuals (map.html)

LEAFLET = "https://unpkg.com/leaflet@1.9.4/dist/"
MAP_CSS = """
.mc{display:flex;flex-wrap:wrap;gap:10px 16px;align-items:flex-end;margin:6px 0 10px}
.mc label{display:flex;flex-direction:column;gap:3px;font-size:12.5px;color:var(--ink2)}
.mc input[type=search]{font:inherit;padding:3px 8px;border-radius:6px;border:1px solid var(--axis);background:var(--surface);
color:var(--ink);min-width:190px}
.mc button,.chip{font:inherit;font-size:13px;color:var(--ink2);background:var(--surface);border:1px solid var(--axis);
border-radius:999px;padding:2px 10px;cursor:pointer}
.mc button:hover,.chip:hover{color:var(--ink)}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 8px}
.chip{display:inline-flex;align-items:center;gap:6px}
.chip[aria-pressed="false"]{opacity:.45;text-decoration:line-through}
.chip .sw{width:9px;height:9px;border-radius:50%}
.chip .n{color:var(--muted);font-size:12px}
#map{height:min(72vh,680px);min-height:420px;border-radius:10px;border:1px solid var(--border);background:var(--track)}
#map-off{padding:24px;color:var(--ink2)}
.map-status{font-size:13.5px;color:var(--ink2);margin:8px 0 6px;font-variant-numeric:tabular-nums}
.map-status b{color:var(--ink)}
.leaflet-container{font:inherit}
.leaflet-popup-content-wrapper,.leaflet-popup-tip{background:var(--surface);color:var(--ink);
box-shadow:0 4px 16px rgba(0,0,0,.18)}
.leaflet-popup-content{margin:12px 14px;font-size:13px;line-height:1.45;font-variant-numeric:tabular-nums;min-width:240px}
.leaflet-popup-content h4{margin:0 0 2px;font-size:14.5px}
.leaflet-popup-content .pk{color:var(--muted)}
.leaflet-popup-content table{font-size:12.5px;margin-top:6px}
.leaflet-popup-content td{padding:1px 8px 1px 0;border:0}
.site-list{margin-top:8px;border-top:1px solid var(--grid);padding-top:6px;max-height:170px;overflow:auto}
.site-list div{cursor:pointer;color:var(--ink2)}.site-list div:hover{color:var(--ink)}.site-list .cur{color:var(--ink);font-weight:600}
.leaflet-control-attribution{background:var(--surface)!important;color:var(--muted)!important}
.leaflet-control-attribution a{color:var(--link)!important}
.leaflet-bar a{background:var(--surface);color:var(--ink);border-color:var(--border)}
#map-table tbody tr{cursor:pointer}#map-table tbody tr:hover td{background:var(--track)}
"""

MAP_JS = r"""
(function(){
const M=JSON.parse(document.getElementById('map-data').textContent);
const R=M.rows.map(r=>({id:r[0],iid:r[1],g:M.groups[r[2]],loc:M.locs[r[3]],c:M.countries[r[4]],lat:r[5],lon:r[6],bp:r[7],
 f3:r[8],se:r[9],rank:r[10],snps:r[11],mt:r[12],y:r[13],pub:M.pubs[r[14]],per:r[15],reg:r[16],cul:r[17],diff:r[18]}));
R.forEach(r=>{r.hay=[r.id,r.iid,r.g,r.loc,r.c,r.mt,r.y,r.pub].join(' ').toLowerCase();r.site=r.lat+','+r.lon});
R.sort((a,b)=>b.rank-a.rank);                          // best last: drawn on top
const N=M.n_total,$=id=>document.getElementById(id);
const fmtAge=bp=>{const y=1950-bp,cal=y<0?Math.round(-y).toLocaleString()+' BCE':Math.round(y)+' CE';
 return (bp>=1000?(bp/1000).toFixed(1)+'k':Math.round(bp))+' years ago (c. '+cal+')'};
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let pal={};
function readPal(){pal={pt:css('--pt'),surface:css('--surface'),q:[1,2,3,4].map(k=>css('--q'+k)),s:[1,2,3,4,5,6,7,8].map(k=>css('--s'+k))}}
readPal();
const mode=$('m-color');
const tier=r=>r.rank<=20?3:r.rank<=100?2:r.rank<=500?1:r.rank<=2000?0:-1;
const colorOf=r=>mode.value==='period'?pal.s[r.per]:(tier(r)<0?pal.pt:pal.q[tier(r)]);
const radius=r=>r.rank<=20?7:r.rank<=100?5.5:r.rank<=500?4.5:3.5;
function legend(){const L=$('map-legend');
 const k=(c,t)=>'<span class="key"><i class="sw" style="background:'+c+';border-radius:50%"></i>'+t+'</span>';
 const sz='<span class="key"><i class="sw" style="width:14px;height:14px;border-radius:50%;background:var(--ink2)"></i>'+
  '<i class="sw" style="width:10px;height:10px;border-radius:50%;background:var(--ink2)"></i>'+
  '<i class="sw" style="width:7px;height:7px;border-radius:50%;background:var(--ink2)"></i>Larger = closer (top 20, 100, 500)</span>';
 L.innerHTML=mode.value==='period'?'<span class="key">Colour = period, as on the buttons above</span>'+sz
  :k(pal.q[3],'Top 20')+k(pal.q[2],'21–100')+k(pal.q[1],'101–500')+k(pal.q[0],'501–2,000')+k(pal.pt,'Rank over 2,000')}

// --- map
let map=null,layer=null,tiles=null;
const dark=()=>{const t=document.documentElement.dataset.theme;return t?t==='dark':matchMedia('(prefers-color-scheme: dark)').matches};
// Basemaps without an API key: Esri gray canvas (light/dark); OpenStreetMap if Esri tiles fail to load
const ESRI='https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/';
const PROVIDERS=[{url:()=>ESRI+(dark()?'World_Dark_Gray_Base':'World_Light_Gray_Base')+'/MapServer/tile/{z}/{y}/{x}',
  attr:'Tiles &copy; <a href="https://www.esri.com">Esri</a> &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors',max:16},
 {url:()=>'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
  attr:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',max:18}];
let prov=0,tileErr=0;
const tileUrl=()=>PROVIDERS[prov].url();
const EUROPE=[[33,-12],[68,80]];
if(window.L){
 map=L.map('map',{preferCanvas:true,worldCopyJump:true,minZoom:2}).fitBounds(EUROPE);
 const addTiles=()=>{tileErr=0;const p=PROVIDERS[prov];
  tiles=L.tileLayer(p.url(),{maxZoom:p.max,maxNativeZoom:p.max,attribution:p.attr}).addTo(map);
  tiles.on('tileerror',()=>{if(++tileErr===6&&prov<PROVIDERS.length-1){map.removeLayer(tiles);prov++;addTiles()}})};
 addTiles();
 const renderer=L.canvas({padding:.5,tolerance:3});
 layer=L.layerGroup().addTo(map);
 R.forEach(r=>{r.m=L.circleMarker([r.lat,r.lon],{renderer,radius:radius(r),weight:1.2,color:pal.surface,
  fillColor:colorOf(r),fillOpacity:r.rank<=2000?.9:.55});
  r.m.on('mouseover',e=>{const o=e.originalEvent;show({dataset:{tip:r.iid+'\n'+r.g+'\n'+fmtAge(r.bp)+'\nRank '+r.rank.toLocaleString()+
   ' of '+N.toLocaleString()+' · f3 '+r.f3.toFixed(4)}},o.clientX,o.clientY)});
  r.m.on('mouseout',()=>{tip.hidden=true});
  r.m.on('click',()=>openPop(r))})
}else{$('map').innerHTML='<div id="map-off">The map needs an internet connection (Leaflet and the map tiles load '+
 'from the web). The filters and the table below still work.</div>'}
function recolor(){readPal();legend();R.forEach(r=>r.m&&r.m.setStyle({fillColor:colorOf(r),color:pal.surface}))}
mode.addEventListener('change',recolor);
function themeChanged(){if(tiles)tiles.setUrl(tileUrl());recolor()}
new MutationObserver(themeChanged).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',themeChanged);

// --- filters
const st={per:new Set(M.periods.map((_,i)=>i)),reg:-1,top:0,cul:-1,q:''};
const chips=$('m-periods');
chips.innerHTML=M.periods.map((p,i)=>'<button type="button" class="chip" aria-pressed="true" data-i="'+i+'" title="'+
 M.period_notes[i]+'"><i class="sw" style="background:var(--s'+(i+1)+')"></i>'+p+' <span class="n">'+
 M.period_n[i].toLocaleString()+'</span></button>').join('');
chips.addEventListener('click',e=>{const b=e.target.closest('.chip');if(!b)return;const i=+b.dataset.i;
 if(e.altKey||e.shiftKey){st.per=new Set([i])}else if(st.per.has(i))st.per.delete(i);else st.per.add(i);
 chips.querySelectorAll('.chip').forEach(c=>c.setAttribute('aria-pressed',st.per.has(+c.dataset.i)));apply()});
$('m-only').addEventListener('click',()=>{const all=st.per.size===M.periods.length;
 st.per=new Set(all?[]:M.periods.map((_,i)=>i));chips.querySelectorAll('.chip').forEach(c=>c.setAttribute('aria-pressed',!all));apply()});
const sel=(id,key)=>$(id).addEventListener('change',e=>{st[key]=+e.target.value;apply()});
sel('m-region','reg');sel('m-top','top');sel('m-culture','cul');
let qt;$('m-q').addEventListener('input',e=>{clearTimeout(qt);qt=setTimeout(()=>{st.q=e.target.value.trim().toLowerCase();apply()},150)});
$('m-reset').addEventListener('click',()=>{st.per=new Set(M.periods.map((_,i)=>i));st.reg=-1;st.top=0;st.cul=-1;st.q='';
 ['m-region','m-culture'].forEach(i=>$(i).value='-1');$('m-top').value='0';$('m-q').value='';
 chips.querySelectorAll('.chip').forEach(c=>c.setAttribute('aria-pressed','true'));apply();if(map)map.fitBounds(EUROPE)});
$('m-europe').addEventListener('click',()=>map&&map.fitBounds(EUROPE));
$('m-world').addEventListener('click',()=>map&&map.setView([30,30],2));
$('m-fit').addEventListener('click',()=>{if(!map)return;const v=R.filter(pass);if(v.length)map.fitBounds(L.latLngBounds(v.map(r=>[r.lat,r.lon])).pad(.15))});
const pass=r=>st.per.has(r.per)&&(st.reg<0||r.reg===st.reg)&&(!st.top||r.rank<=st.top)&&(st.cul<0||r.cul.includes(st.cul))&&
 (!st.q||r.hay.includes(st.q));
let shown=[];
function apply(){shown=R.filter(pass);
 if(layer){layer.clearLayers();shown.forEach(r=>layer.addLayer(r.m))}
 const best=shown.length?shown[shown.length-1]:null;
 $('map-status').innerHTML=shown.length?'<b>'+shown.length.toLocaleString()+'</b> of '+R.length.toLocaleString()+
  ' individuals shown · closest: <b>'+best.iid+'</b> ('+best.g+', rank '+best.rank.toLocaleString()+')':'No individual matches these filters.';
 table()}

// --- table of the closest in view
function table(){const rows=shown.slice(-100).reverse();
 $('map-tbody').innerHTML=rows.map(r=>'<tr data-k="'+R.indexOf(r)+'"><td>'+r.rank.toLocaleString()+'</td><td>'+r.iid+'</td><td>'+
  r.g+'</td><td>'+(r.loc.split('(')[0].trim()||'–')+', '+r.c+'</td><td>'+(r.bp>=1000?(r.bp/1000).toFixed(1)+'k':r.bp)+
  '</td><td>'+M.periods[r.per]+'</td><td>'+r.f3.toFixed(4)+'</td><td>'+r.snps.toLocaleString()+'</td><td>'+(r.mt||'')+
  '</td><td>'+(r.y||'')+'</td></tr>').join('');
 $('map-tcount').textContent=shown.length>100?'the closest 100 of '+shown.length.toLocaleString():'all '+shown.length}
$('map-tbody').addEventListener('click',e=>{const tr=e.target.closest('tr');if(!tr)return;const r=R[+tr.dataset.k];
 if(map){map.setView([r.lat,r.lon],Math.max(map.getZoom(),6));openPop(r);$('map').scrollIntoView({behavior:'smooth',block:'center'})}});

// --- popup: the individual, then everyone shown at the same site
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function popHtml(r){const site=shown.filter(x=>x.site===r.site).sort((a,b)=>a.rank-b.rank);
 const row=(k,v)=>v?'<tr><td class="pk">'+k+'</td><td>'+v+'</td></tr>':'';
 let h='<h4>'+esc(r.iid)+'</h4><div>'+esc(r.g)+'</div><table>'+
  row('Site',esc(r.loc)+', '+esc(r.c))+row('Age',fmtAge(r.bp))+row('Period',M.periods[r.per]+' · '+M.regions[r.reg])+
  row('f3',r.f3.toFixed(4)+' ± '+r.se.toFixed(4))+row('Rank','<b>'+r.rank.toLocaleString()+'</b> of '+N.toLocaleString())+
  (r.diff===null?'':row('vs E. Europeans',(r.diff>=0?'+':'')+r.diff.toFixed(4)))+
  row('Shared SNPs',r.snps.toLocaleString())+row('mtDNA',esc(r.mt))+row('Y-DNA',esc(r.y))+row('Study',esc(r.pub))+
  row('AADR ID',esc(r.id))+'</table>';
 if(site.length>1)h+='<div class="site-list"><div class="pk" style="cursor:default">'+site.length+' individuals shown at this site:</div>'+
  site.slice(0,40).map(x=>'<div data-k="'+R.indexOf(x)+'" class="'+(x===r?'cur':'')+'">#'+x.rank.toLocaleString()+' '+esc(x.iid)+
  ' · '+(x.bp>=1000?(x.bp/1000).toFixed(1)+'k':x.bp)+' · '+esc(x.g)+'</div>').join('')+'</div>';
 return h}
let pop=null;
function openPop(r){if(!map)return;tip.hidden=true;pop=L.popup({maxWidth:340,autoPanPadding:[30,30]}).setLatLng([r.lat,r.lon])
 .setContent(popHtml(r)).openOn(map)}
document.addEventListener('click',e=>{const d=e.target.closest('.site-list [data-k]');if(d&&pop)pop.setContent(popHtml(R[+d.dataset.k]))});
legend();apply();
})();
"""


def map_page(df: pd.DataFrame) -> str:
    d = df[df["lat"].notna() & df["lon"].notna()].copy()
    n_total, n_off = len(df), int(len(df) - len(d))

    def codes(values) -> tuple[list[str], list[int]]:
        cats = list(dict.fromkeys(values))
        idx = {v: i for i, v in enumerate(cats)}
        return cats, [idx[v] for v in values]

    clean = lambda s: "" if pd.isna(s) or str(s) in ("..", "nan", "n/a") else str(s)
    groups, gi = codes(d["group"].astype(str).tolist())
    locs, li = codes(d["locality"].map(clean).tolist())
    countries, ci = codes(d["country"].map(clean).tolist())
    pubs, pi = codes(d["pub"].map(clean).tolist())
    cul_of = {g: [k for k, (_, rx) in enumerate(CULTURE_RX) if rx.search(g)] for g in groups}
    diff = d["f3_diff"] if "f3_diff" in d else pd.Series(np.nan, index=d.index)
    rows = []
    for r, g, l, c, p_, dv in zip(d.itertuples(), gi, li, ci, pi, diff):
        y = clean(r.y_hg)
        rows.append([r.id, r.iid, g, l, c, round(float(r.lat), 4), round(float(r.lon), 4), int(r.date_bp),
                     round(float(r.f3), 5), round(float(r.se), 5), int(r.rank), int(r.snps_shared), clean(r.mt_hg),
                     "female" if y.startswith("n/a") else y, p_, PERIOD_NAMES.index(str(r.period)),
                     REGION_NAMES.index(r.region), cul_of[groups[g]], None if pd.isna(dv) else round(float(dv), 5)])
    cul_n = [sum(1 for r in rows if k in r[17]) for k in range(len(CULTURES))]
    reg_n = d["region"].value_counts()
    per_n = d["period"].value_counts()
    data = {"rows": rows, "groups": groups, "locs": locs, "countries": countries, "pubs": pubs, "n_total": n_total,
            "periods": PERIOD_NAMES, "period_notes": [n for _, n in PERIODS],
            "period_n": [int(per_n.get(p, 0)) for p in PERIOD_NAMES], "regions": REGION_NAMES}
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")

    reg_opts = "".join(f'<option value="{i}">{esc(r)} ({int(reg_n[r]):,})</option>'
                       for i, r in enumerate(REGION_NAMES) if reg_n.get(r, 0))
    cul_opts = "".join(f'<option value="{k}">{esc(n)} ({cul_n[k]:,})</option>'
                       for k, (n, _) in enumerate(CULTURES) if cul_n[k] >= 3)
    top_opts = "".join(f'<option value="{v}">{t}</option>' for v, t in
                       [(0, f"All {n_total:,}"), (20, "Top 20"), (100, "Top 100"), (500, "Top 500"), (1000, "Top 1,000"),
                        (2000, "Top 2,000")])
    controls = (
        '<div class="chips" id="m-periods" role="group" aria-label="Periods"></div>'
        '<div class="mc">'
        f'<label>Region<select id="m-region"><option value="-1">All regions</option>{reg_opts}</select></label>'
        f'<label>Genetic affinity<select id="m-top">{top_opts}</select></label>'
        f'<label>Culture<select id="m-culture"><option value="-1">All cultures</option>{cul_opts}</select></label>'
        '<label>Search<input type="search" id="m-q" placeholder="ID, group, site, haplogroup…"></label>'
        '<label>Colour by<select id="m-color"><option value="period">Period</option>'
        '<option value="rank">Affinity (rank)</option></select></label>'
        '<button type="button" id="m-only">All / no periods</button>'
        '<button type="button" id="m-fit">Zoom to shown</button><button type="button" id="m-europe">Europe</button>'
        '<button type="button" id="m-world">World</button><button type="button" id="m-reset">Reset</button></div>')
    top = df.iloc[0]
    tied = int(df["tied_top"].sum()) if "tied_top" in df else 0
    lede = (f"Every ancient individual in the outgroup-f3 comparison ({n_total:,} genomes from the AADR), placed at its "
            "burial site. f3(Mbuti; you, X) measures the genetic drift you share with that person's population; rank 1 "
            f"is the highest ({esc(top['iid'])}, {esc(str(top['group']))}). Click a point for details and for everyone "
            "buried at the same site; shift-click a period to show only that period.")
    body = (controls + '<div class="map-status" id="map-status"></div><div class="legend" id="map-legend"></div>'
            '<div id="map"></div>')
    notes = (f"<p><b>Read ranks loosely.</b> Single ancient genomes are noisy: {tied} individuals are statistically tied "
             "with rank 1, and neighbouring ranks overlap. Groups of individuals (the populations on the "
             "<a href=\"ancient.html\">ancient individuals page</a>) are more reliable than any single point.</p>"
             "<p><b>Why hunter-gatherers rank high.</b> f3 measures shared drift, and small, isolated foraging "
             "populations carried a lot of it. Read the map as “whose population history overlaps with yours”, "
             "not as a map of your ancestors.</p>"
             f"<p><b>Data.</b> Coordinates, dates, groups and haplogroups from the AADR v66 annotation. Periods come from "
             "the group label (_EBA, _Mesolithic, _Viking, …), or from the date where the label has none. "
             f"{n_off} individuals without coordinates are left out. Basemap: Esri gray canvas (© Esri, HERE, Garmin, © OpenStreetMap contributors), no API key needed.</p>")
    tbl = ('<div class="table-wrap"><table class="dense" id="map-table"><thead><tr><th>Rank</th><th>Individual</th>'
           '<th>Group</th><th>Site</th><th>Age</th><th>Period</th><th>f3</th><th>SNPs</th><th>mtDNA</th><th>Y-DNA</th>'
           '</tr></thead><tbody id="map-tbody"></tbody></table></div>')
    secs = (section("explore", "Map of your ancient matches", lede, card("Ancient individuals by outgroup f3", body))
            + section("list", "Closest individuals in view", "",
                      card("Individuals matching the filters", tbl,
                           'Showing <span id="map-tcount"></span>, by rank. Click a row to find it on the map.'))
            + section("notes", "How to read the map", "", card("Notes", notes)))
    secs += (f'<script type="application/json" id="map-data">{blob}</script>'
             f'<script>{MAP_JS}</script>')
    head = (f'<link rel="stylesheet" href="{LEAFLET}leaflet.css" crossorigin="">'
            f'<script src="{LEAFLET}leaflet.js" crossorigin=""></script><style>{MAP_CSS}</style>')
    html_ = page("Map of ancient matches", f"Outgroup f3(Mbuti; you, X), {n_total:,} ancient individuals, AADR v66",
                 [("explore", "Map"), ("list", "Closest in view"), ("notes", "How to read")], secs, back=True, head=head)
    return html_.replace('<button class="theme-btn"', '<a href="ancient.html">Ancient individuals →</a>'
                                                       '<button class="theme-btn"', 1)


# ---------------------------------------------------------------- G25-style coordinates (26_g25_style.py)

def sec_g25(out: Path, rep: Path) -> str:
    g = out / "g25"
    if not (g / "distances.csv").exists() or not (g / "fits.csv").exists():
        return ""
    dist = pd.read_csv(g / "distances.csv")
    fits = pd.read_csv(g / "fits.csv")
    ctrl = pd.read_csv(g / "controls.csv") if (g / "controls.csv").exists() else pd.DataFrame()
    summ = json.loads((g / "summary.json").read_text())
    (rep / "g25").mkdir(exist_ok=True)
    sheets = ["you.txt", "modern_populations.txt", "ancient_populations.txt", "ancient_individuals.txt"]
    for f in sheets:
        if (g / f).exists():
            shutil.copy(g / f, rep / "g25" / f)
    t = "you_CRAM"
    d = dist[dist["target"] == t]
    mp = d[d["kind"] == "modern population"].head(15)
    ap = d[d["kind"] == "ancient population"].head(15)
    ai = d[d["kind"] == "ancient individual"].head(25)
    cd = ctrl["dist_to_pop"] if len(ctrl) else pd.Series(dtype=float)
    tiles = [tile("Closest present-day population", f"{mp.iloc[0]['name']}", f"distance {mp.iloc[0]['distance']:.4f}"),
             tile("Closest ancient population", str(ap.iloc[0]['name']).replace("_", "_\u200b"), f"distance {ap.iloc[0]['distance']:.4f}, "
                  f"{age_txt(ap.iloc[0]['date_bp'])}"),
             tile("Russians with your data quality", f"{cd.min():.3f}–{cd.max():.3f}" if len(cd) else "–",
                  "distance to the Russian average: what 'close' means here"),
             tile("Your CRAM vs MyHeritage CSV", f"{summ['cram_vs_csv_distance']:.4f}"
                  if summ.get("cram_vs_csv_distance") is not None else "–", "same person, two data sources")]
    body = f'<div class="tiles">{"".join(tiles)}</div>'
    coords = (g / "you.txt").read_text()
    links = " · ".join(f'<a href="g25/{f}" download>{f}</a>' for f in sheets if (rep / "g25" / f).exists())
    body += card("Your coordinates (25 scaled principal components)",
                 f'<pre class="g25">{esc(coords)}</pre><p class="chart-note">Download the sheets: {links}</p>',
                 "To use them in Vahaduo, paste <code>you.txt</code> into its Target box and one of the population sheets "
                 "into its Source box. Don't mix them with real Global25 sheets: the axes differ.")

    def bars(dd, ancient=False):
        return hbars([(r.name if not ancient else f"{r.name} ({age_txt(r.date_bp)})", r.distance, f"{r.distance:.4f}",
                       f"{r.name}\ndistance {r.distance:.4f}\n{int(r.n)} {'people' if r.n != 1 else 'person'}"
                       + (f"\n{age_txt(r.date_bp)} ago" if ancient else "")) for r in dd.itertuples()],
                     note="Euclidean distance over 25 scaled PCs; shorter = closer.")
    body += card("Closest present-day populations", bars(mp)) + card("Closest ancient populations", bars(ap, True))

    # Mixture fits
    f = fits[fits["target"] == t]
    rows = []
    for r in f.itertuples():
        parts = [(w.rsplit(" ", 1)[0], float(w.rsplit(" ", 1)[1])) for w in str(r.weights).split("; ") if w]
        major = [(n, v) for n, v in parts if v >= 0.03][:7]
        rest = 1 - sum(v for _, v in major)
        segs = [(n, v, k + 1, f"{n}: {v:.1%}") for k, (n, v) in enumerate(major)]
        if rest > 0.0005:
            segs.append(("Other", rest, 8, "Other sources (< 3% each): "
                         + ", ".join(f"{n} {v:.1%}" for n, v in parts if v < 0.03)))
        c = fits[fits["target"].str.startswith("control") & (fits["model"] == r.model)]["distance"]
        ref = f" · Russian controls {c.min():.4f}–{c.max():.4f}" if len(c) else ""
        txt = " · ".join(f"{n} {v:.1%}" for n, v in parts[:10]) + (" · …" if len(parts) > 10 else "")
        rows.append(f'<div class="fit-h"><b>{esc(r.model)}</b> <span class="muted">distance {r.distance:.4f}{esc(ref)}'
                    f'</span></div>{seg_bar(segs, named=True)}<div class="chart-note">{esc(txt)}</div>')
    body += card("Mixture models (the Vahaduo method)", "".join(rows),
                 "Weights ≥ 0 summing to 100% that bring the mix closest to you. The 'Open' models let the fit pick from "
                 "every population in the set. A distance near the controls' is a good fit. Unlike qpAdm there is no "
                 "p-value: a mixture can fit well and still be historically wrong.")
    t_ai = pd.DataFrame({"Rank": ai["rank"].astype(int), "Individual": ai["name"], "Group": ai["group"],
                         "Country": ai["country"], "Age": ai["date_bp"].map(age_txt),
                         "Distance": ai["distance"].map(lambda v: f"{v:.4f}")})
    body += card("Closest ancient individuals", details(f"Show the closest {len(t_ai)} of "
                 f"{summ['ancient_individuals']:,}", table(t_ai, "dense sortable")),
                 "Single ancient genomes are noisy (one read per SNP, often far fewer SNPs than you); populations are "
                 "more reliable.")
    body += card("How this was made and how it differs from Global25", (
        f"<p><b>Axes.</b> A PCA of {summ['axes_genomes']:,} present-day genomes from {summ['axes_populations']} "
        f"populations worldwide (AADR, at most 4 per population), {summ['snps']:,} SNPs. Each coordinate is a principal "
        f"component scaled by the square root of its eigenvalue, the same recipe as Global25; the 25 axes hold "
        f"{summ['variance_share_25']:.0%} of the variance.</p>"
        "<p><b>Projection.</b> You, every other present-day genome and "
        f"{summ['ancient_individuals']:,} ancient genomes are projected by least squares on the SNPs each one has. "
        "Population rows average genomes that were not used to build the axes, like you.</p>"
        "<p><b>Not Global25.</b> The real G25 SNP weights are private, so these numbers are a different coordinate "
        "system: close in spirit, not interchangeable. Distances and mixtures are comparable within these sheets only."
        "</p><p><b>Compared with qpAdm.</b> qpAdm tests whether a model is statistically adequate (p-value) using "
        "outgroups; a distance fit always returns a mixture. Read these fits as a cross-check of the qpAdm results on "
        "the Deep ancestry page.</p>"))
    lede = ("Global25-style coordinates: your position on a 25-dimensional genetic map, with distances to present-day "
            "and ancient populations and mixture fits, made from your own data and the AADR.")
    return section("g25", "G25-style coordinates", esc(lede), body)


# ---------------------------------------------------------------- f4 Uralic / Siberian test (19_f4_uralic.py)

F4_CTRL = {"Russian": 1, "Belarusian": 3, "Ukrainian": 2}   # slot per control population


def sec_f4(out: Path) -> str:
    path = out / "f4" / "f4_uralic_all_tests.csv"
    if not path.exists():
        return ""
    d = pd.read_csv(path)
    lefts = [l for l in dict.fromkeys(d["left"]) if l != "Mbuti"]
    if not lefts:
        return ""
    left = lefts[0]
    d = d[(d["left"] == left) & d["z"].notna()]
    lo, hi = min(-3.0, d["z"].min() - 0.5), max(9.0, d["z"].max() + 0.5)
    pos = lambda v: (min(max(v, lo), hi) - lo) / (hi - lo) * 100
    ticks = [t for t in range(int(np.ceil(lo)), int(hi) + 1) if t % 3 == 0]
    grid = "".join(f'<div class="pp-grid" style="left:{pos(t):.2f}%"></div>' for t in ticks)
    grid += f'<div class="fp-mean" style="left:{pos(0):.2f}%"></div>'
    rows = []
    for cat, dc in d.groupby("category", sort=False):
        rows.append(f'<div class="pp-label f4-cat">{esc(cat)}</div><div></div>')
        for x, dx in dc.groupby("X", sort=False):
            you = dx[dx["test"] == "you"]
            marks = ""
            for _, r in dx[dx["test"] != "you"].iterrows():
                marks += (f'<div class="f4-c s{F4_CTRL.get(r.test_pop, 4)}" style="left:{pos(r.z):.2f}%" tabindex="0" '
                          f'data-tip="{esc(r.test)} ({esc(r.test_pop)})&#10;Z = {r.z:.2f}"></div>')
            if len(you):
                r = you.iloc[0]
                marks += (f'<div class="f4-you" style="left:{pos(r.z):.2f}%" tabindex="0" data-tip="You&#10;'
                          f'f4({esc(left)}, {esc(x)}; Ukrainian, you) = {r.f4:.5f} ± {r.se:.5f}&#10;Z = {r.z:.2f}"></div>')
            n = int(dx["n"].iloc[0])
            rows.append(f'<div class="pp-label">{esc(x)} <span class="muted">n={n}</span></div>'
                        f'<div class="fp-track">{grid}{marks}</div>')
    axis = "".join(f'<span style="left:{pos(t):.2f}%">{t}</span>' for t in ticks)
    leg = ('<div class="legend"><span class="key"><i class="sw" style="background:var(--ink)"></i>You</span>'
           + "".join(f'<span class="key"><i class="sw s{s_}"></i>{esc(p_)} (one dot per person)</span>'
                     for p_, s_ in F4_CTRL.items()) + "</div>")
    chart = leg + f'<div class="pp fp">{"".join(rows)}<div></div><div class="pp-axis">{axis}</div></div>'
    yz = d[d["test"] == "you"].set_index("X")["z"]
    sib = [x for x in ("Nganasan", "Selkup", "Khanty", "Mansi", "Bolshoy Oleni Ostrov (MBA)") if x in yz]
    cz = d[(d["test"] != "you") & d["X"].isin(sib)].groupby("test_pop")["z"].mean()
    tiles = [tile("Siberian groups (mean Z)", f"{yz[sib].mean():.1f}", "Nganasan, Selkup, Khanty, Mansi, Bolshoy Oleni Ostrov")]
    tiles += [tile(f"{p_} controls", f"{cz[p_]:.1f}", "mean Z for the same groups") for p_ in F4_CTRL if p_ in cz]
    lede = (f"f4({left}, X; Ukrainian, test) is positive when the test genome shares more with X than Ukrainians do. "
            "With a Slavic population on the left instead of Mbuti, drift that all Europeans share cancels, so errors in "
            "your low-coverage calls can only weaken a signal, not create one. AADR Russians, Belarusians and Ukrainians "
            "(one random allele per SNP, like you) are tested in your place. |Z| > 3 is significant. Your Siberian/Uralic "
            "signal is about as strong as a typical Russian's and absent in Belarusians and Ukrainians: it is the "
            "north-eastern (Finnic/Volga) component that Russians absorbed, matching qpAdm's ~7% Nganasan-related share.")
    body = f'<div class="tiles">{"".join(tiles)}</div>' + card(
        "Who shares more with X than Ukrainians do?", chart,
        "Vertical dashed line: Z = 0 (no difference from Ukrainians). Hover for values.")
    return section("f4", "Uralic and Siberian signal (f4)", esc(lede), body)


# ---------------------------------------------------------------- farmer ancestry (21_farmer_f4.py, 22_pca_ancient.py)

PCA_ERA = {"WHG": 1, "EHG": 1, "Anatolia_N": 3, "Europe_EN (LBK)": 3, "Yamnaya": 2, "Sintashta": 2,
           "Scythian (Ukraine)": 2, "Corded Ware": 7, "Bell Beaker": 7, "Early Slavs": 7, "Viking (Sweden)": 7,
           "Iran_N": 4, "Natufian": 4}
PCA_ERA_NAMES = {1: "Hunter-gatherers", 3: "Early farmers", 2: "Steppe herders", 7: "Later Europeans (mixed)",
                 4: "Near Eastern early farmers / foragers"}
PCA_MODERN_LABELS = ["Sardinian", "Basque", "Spanish", "English", "Norwegian", "Finnish", "Russian", "Ukrainian",
                     "Mordovian", "Chuvash", "Bashkir", "Greek", "Georgian", "Turkish", "Iranian", "Druze",
                     "Palestinian", "Lithuanian", "Hungarian"]


def pca_ancient_svg(mods: pd.DataFrame, proj: pd.DataFrame, eig: pd.DataFrame | None, zoom: bool = False) -> str:
    W, H, L, R, T, B = 960, 600 if not zoom else 520, 46, 20, 16, 44
    allx = pd.concat([mods["PC1"], proj["PC1"]]); ally = pd.concat([mods["PC2"], proj["PC2"]])
    x0, x1 = allx.quantile(0.002), allx.quantile(0.998)
    y0, y1 = ally.quantile(0.002), ally.quantile(0.998)
    px, py = (x1 - x0) * 0.05, (y1 - y0) * 0.05
    x0, x1, y0, y1 = x0 - px, x1 + px, y0 - py, y1 + py
    me = proj[proj["id"] == "you (CRAM)"].iloc[0]
    hx, hy = (x1 - x0) * 0.13, (y1 - y0) * 0.17
    box = (me.PC1 - hx, me.PC1 + hx, me.PC2 - hy, me.PC2 + hy)
    if zoom:
        x0, x1, y0, y1 = box
    inside = lambda d: d[(d["PC1"] >= x0) & (d["PC1"] <= x1) & (d["PC2"] >= y0) & (d["PC2"] <= y1)]
    X = lambda v: L + (v - x0) / (x1 - x0) * (W - L - R)
    Y = lambda v: T + (1 - (v - y0) / (y1 - y0)) * (H - T - B)
    var = lambda k: f" ({eig['variance_share'].iloc[k] * 100:.1f}% of variance)" if eig is not None else ""
    g = (f'<line class="axis" x1="{L}" x2="{W - R}" y1="{H - B}" y2="{H - B}"/>'
         f'<line class="axis" x1="{L}" x2="{L}" y1="{T}" y2="{H - B}"/>'
         f'<text class="ax" x="{(L + W - R) / 2:.0f}" y="{H - 12}" text-anchor="middle">PC1{var(0)}</text>'
         f'<text class="ax" transform="translate(14 {(T + H - B) / 2:.0f}) rotate(-90)" text-anchor="middle">PC2{var(1)}</text>')
    if not zoom:
        g += (f'<rect x="{X(box[0]):.1f}" y="{Y(box[3]):.1f}" width="{X(box[1]) - X(box[0]):.1f}" '
              f'height="{Y(box[2]) - Y(box[3]):.1f}" fill="none" stroke="var(--muted)" stroke-dasharray="4 3" rx="4"/>'
              f'<text class="ax" x="{X(box[1]) - 4:.1f}" y="{Y(box[2]) + 14:.1f}" text-anchor="end">zoomed below</text>')
    m = inside(mods)
    r_pt = 3.2 if zoom else 2.6
    pts = "".join(f'<circle class="pt" cx="{X(r.PC1):.1f}" cy="{Y(r.PC2):.1f}" r="{r_pt}" tabindex="0" '
                  f'data-tip="{esc(r.pop)} (modern)&#10;{esc(r.id)}"/>' for r in m.itertuples())
    anc = inside(proj[proj["group"] != "you"])
    for r in anc.itertuples():
        slot = PCA_ERA.get(r.group, 8)
        age = f"&#10;~{int(r.date_bp):,} years ago" if pd.notna(getattr(r, "date_bp", np.nan)) else ""
        pts += (f'<circle cx="{X(r.PC1):.1f}" cy="{Y(r.PC2):.1f}" r="{5 if zoom else 4}" fill="var(--s{slot})" '
                f'fill-opacity=".8" stroke="var(--surface)" stroke-width="1" tabindex="0" data-tip="{esc(r.group)}&#10;'
                f'{esc(r.id)} ({esc(r.label)}){age}&#10;{r.snps:,} SNPs"/>')
    # Labels: You first, then ancient groups, then moderns; each tries a few spots and is dropped if all collide
    placed: list[tuple[float, float, float, float]] = []
    dots = []

    def put(text: str, x: float, y: float, cls: str, spots) -> str:
        w, h = len(text) * (7.4 if cls != "ax" else 6.0), 12
        for dx, dy, anchor in spots:
            left = x + dx - (w if anchor == "end" else w / 2 if anchor == "middle" else 0)
            bx = (left - 2, y + dy - h, left + w + 2, y + dy + 3)
            if all(bx[2] < a or bx[0] > c_ or bx[3] < b or bx[1] > d_ for a, b, c_, d_ in placed) and \
                    all(not (bx[0] - 4 < px_ < bx[2] + 4 and bx[1] - 4 < py_ < bx[3] + 4) for px_, py_ in dots) and \
                    bx[0] > L and bx[2] < W - R and bx[1] > T:
                placed.append(bx)
                return (f'<text class="{cls}" x="{x + dx:.1f}" y="{y + dy:.1f}" text-anchor="{anchor}">'
                        f'{esc(text)}</text>')
        return ""
    labels = ""
    for r in proj[proj["group"] == "you"].itertuples():
        main_ = "CRAM" in r.id
        x, y = X(r.PC1), Y(r.PC2)
        labels += (f'<circle class="me" cx="{x:.1f}" cy="{y:.1f}" r="{(8 if zoom else 7) if main_ else 5}" '
                   f'{"" if main_ else "fill-opacity=.5 "}tabindex="0" data-tip="{esc(r.id)}&#10;{r.snps:,} SNPs&#10;'
                   f'PC1 {r.PC1:.1f}, PC2 {r.PC2:.1f}"/>')
        if main_:
            labels += put("You", x, y, "me-label", [(-12, 5, "end"), (0, -13, "middle"), (12, 5, "start")])
            dots.append((x, y))
    ca = anc.groupby("group")[["PC1", "PC2"]].median()
    for grp_, r in ca.iterrows():
        if not zoom and box[0] <= r.PC1 <= box[1] and box[2] <= r.PC2 <= box[3]:
            continue
        x, y = X(r.PC1), Y(r.PC2)
        labels += put(grp_, x, y, "grp", [(9, 4, "start"), (0, -10, "middle"), (0, 18, "middle"), (-9, 4, "end"),
                                          (24, 4, "start"), (0, 30, "middle")])
    c = m.groupby("pop")[["PC1", "PC2"]].median()
    show = c.index if zoom else [p_ for p_ in PCA_MODERN_LABELS if p_ in c.index]
    for pop in show:
        if not zoom and box[0] <= c.at[pop, "PC1"] <= box[1] and box[2] <= c.at[pop, "PC2"] <= box[3]:
            continue                                     # named in the zoom instead
        x, y = X(c.at[pop, "PC1"]), Y(c.at[pop, "PC2"])
        labels += put(pop.replace("_", " "), x, y, "ax", [(0, -6, "middle"), (0, 14, "middle"), (6, 4, "start"),
                                                         (-6, 4, "end")])
    svg = (f'<figure class="svg-fig"><svg viewBox="0 0 {W} {H}" role="img" aria-label="West Eurasian PCA with '
           f'ancient genomes projected{" (zoom)" if zoom else ""}">{g}{pts}{labels}</svg></figure>')
    if zoom:
        return svg
    leg = ('<div class="legend"><span class="key"><i class="sw" style="background:var(--ink)"></i>You (large: CRAM, '
           'faint: MyHeritage CSV)</span><span class="key"><i class="sw" style="background:var(--pt)"></i>Modern '
           'West Eurasians (define the axes)</span>'
           + "".join(f'<span class="key"><i class="sw s{k}"></i>{esc(v)}</span>' for k, v in PCA_ERA_NAMES.items())
           + "</div>")
    return leg + svg


FARMER_CTRL = {"Sardinian": 3, "Russian": 1, "Ukrainian": 4, "Nganasan": 7}


def farmer_f4_chart(d: pd.DataFrame) -> str:
    v = d["f4"] * 1000
    lo, hi = float(np.floor(v.min() - 0.5)), float(np.ceil(v.max() + 0.5))
    pos = lambda x: (min(max(x, lo), hi) - lo) / (hi - lo) * 100
    ticks = [t for t in range(int(lo), int(hi) + 1) if t % 2 == 0]
    grid = "".join(f'<div class="pp-grid" style="left:{pos(t):.2f}%"></div>' for t in ticks)
    grid += f'<div class="fp-mean" style="left:{pos(0):.2f}%"></div>'
    rows = []
    for base, db in d.groupby("baseline", sort=False):
        you = db[db["kind"].str.startswith("you")]
        marks = ""
        for r in you.itertuples():
            main_ = r.pop == "you"
            a, b = pos((r.f4 - 2 * r.se) * 1000), pos((r.f4 + 2 * r.se) * 1000)
            if main_:
                marks += (f'<div style="position:absolute;top:50%;height:2px;margin-top:-1px;left:{a:.2f}%;'
                          f'width:{b - a:.2f}%;background:var(--ink);opacity:.35"></div>')
            marks += (f'<div class="f4-you" style="left:{pos(r.f4 * 1000):.2f}%{"" if main_ else ";opacity:.45;transform:scale(.7)"}" '
                      f'tabindex="0" data-tip="{esc(r.test)}&#10;f4 = {r.f4 * 1000:.2f} ± {r.se * 1000:.2f} (×10⁻³)&#10;Z = {r.z:.1f}"></div>')
        rows.append(f'<div class="pp-label"><b>You</b></div><div class="fp-track">{grid}{marks}</div>')
        for pop, dp in db[~db["kind"].str.startswith("you")].groupby("pop", sort=False):
            whole = dp[dp["kind"] == "pop"]
            marks = ""
            for r in dp[dp["kind"] == "ind"].itertuples():
                marks += (f'<div class="f4-c s{FARMER_CTRL.get(pop, 8)}" style="left:{pos(r.f4 * 1000):.2f}%" tabindex="0" '
                          f'data-tip="{esc(r.test)}, one read per SNP like you&#10;f4 = {r.f4 * 1000:.2f}, Z = {r.z:.1f}"></div>')
            for r in whole.itertuples():
                marks += (f'<div style="position:absolute;top:50%;left:{pos(r.f4 * 1000):.2f}%;width:3px;height:16px;'
                          f'margin:-8px 0 0 -1.5px;border-radius:2px;background:var(--ink2)" tabindex="0" '
                          f'data-tip="{esc(pop)} (whole population)&#10;f4 = {r.f4 * 1000:.2f} ± {r.se * 1000:.2f}&#10;Z = {r.z:.1f}"></div>')
            rows.append(f'<div class="pp-label">{esc(pop)}</div><div class="fp-track">{grid}{marks}</div>')
    axis = "".join(f'<span style="left:{pos(t):.2f}%">{t}</span>' for t in ticks)
    leg = ('<div class="legend"><span class="key"><i class="sw" style="background:var(--ink)"></i>You (line: ±2 SE; '
           'faint: MyHeritage CSV)</span><span class="key"><i class="sw" style="background:var(--ink2);width:3px"></i>'
           'Whole population</span>'
           + "".join(f'<span class="key"><i class="sw s{s_}"></i>{esc(p_)} individuals, one read per SNP</span>'
                     for p_, s_ in FARMER_CTRL.items()) + "</div>")
    return leg + f'<div class="pp fp">{"".join(rows)}<div></div><div class="pp-axis">{axis}</div></div>'


def sec_farmer(cfg, out: Path) -> str:
    f4p, f3p = out / "farmer" / "f4_farmer.csv", out / "farmer" / "admix_f3.csv"
    pdir = out / "pca_ancient"
    if not f4p.exists() and not (pdir / "projected.csv").exists():
        return ""
    body, tiles = "", []
    if f4p.exists():
        d = pd.read_csv(f4p)
        for base in dict.fromkeys(d["baseline"]):
            db = d[d["baseline"] == base]
            me = db[db["test"] == "you (CRAM)"].iloc[0]
            ru = db[(db["kind"] == "ind") & (db["pop"] == "Russian")]["f4"]
            tiles.append(tile(f"Farmer excess over {base}", f"Z = {me.z:.1f}",
                              f"Russians {ru.min() * 1000:.1f}–{ru.max() * 1000:.1f}, you {me.f4 * 1000:.1f} (×10⁻³)"))
        out_ = cfg["farmer"]["outgroup"]; fa = cfg["farmer"]["farmer"]
        body += f'<div class="tiles">{"".join(tiles)}</div>'
        note = ("Right of the dashed zero line = more farmer-related than the baseline. Sardinians (mostly farmer "
                "ancestry) and Nganasans (none) bracket the scale. Hover for values.")
        bases = list(dict.fromkeys(d["baseline"]))
        first = "Yamnaya" if "Yamnaya" in bases else bases[0]
        body += card(f"f4({out_}, {fa}; {first}, test) ×10⁻³: farmer ancestry on top of steppe ancestry",
                     farmer_f4_chart(d[d["baseline"] == first]), note)
        for b_ in [b_ for b_ in bases if b_ != first]:
            body += (f"<details><summary>Same test against {esc(b_)}</summary>"
                     + card(f"f4({out_}, {fa}; {b_}, test) ×10⁻³", farmer_f4_chart(d[d["baseline"] == b_]), note)
                     + "</details>")
    if (pdir / "projected.csv").exists():
        mods, proj = pd.read_csv(pdir / "moderns.csv"), pd.read_csv(pdir / "projected.csv")
        eig = pd.read_csv(pdir / "eigen.csv") if (pdir / "eigen.csv").exists() else None
        body += card("West Eurasian PCA with ancient genomes projected",
                     pca_ancient_svg(mods, proj, eig) + '<p class="sub-h">Zoom: Eastern and Northern Europe</p>'
                     + pca_ancient_svg(mods, proj, eig, zoom=True),
                     "Axes from modern Human Origins samples; ancient genomes and you placed by least squares on the SNPs "
                     "each has (smartpca lsqproject). Hunter-gatherers, early farmers and steppe herders form the corners; "
                     "Europeans lie between them according to their mix. Hover a dot for details.")
    if f3p.exists():
        f3 = pd.read_csv(f3p)
        t = f3.assign(**{"Statistic": "f3(" + f3["target"] + "; " + f3["source1"] + ", " + f3["source2"] + ")",
                         "f3 ×10⁻³": (f3["f3"] * 1000).round(2), "Z": f3["z"].round(1),
                         "Reading": np.where(f3["z"] < -3, "admixed between the two",
                                             np.where(f3["z"] > 3, "not a mix of these two", "inconclusive"))})
        body += ("<details><summary>Admixture f3 for populations (not for you)</summary>"
                 + card("f3(target; A, B) < 0 with Z < −3: the target is a mix of A and B",
                        table(t[["Statistic", "f3 ×10⁻³", "Z", "Reading"]], "dense sortable"),
                        "This test needs a heterozygosity correction for the target, which a one-read-per-SNP genome "
                        "can't provide, so only diploid AADR populations are targets here.") + "</details>")
    lede = ("qpAdm estimates your Anatolian-farmer share at ~39%. These two views show the same ancestry directly. "
            "f4 asks whether you share more with Anatolian farmers than populations without farmer ancestry do "
            "(hunter-gatherers, Yamnaya steppe herders). The PCA places you among ancient genomes.")
    return section("farmer", "Farmer ancestry, seen directly", esc(lede), body)


NEA_CTRL = {"Russian": 1, "French": 3, "Han": 4, "Papuan": 7, "Mende": 2, "Yoruba": 2}
NEA_NAMES = {"Russian": "Russians", "French": "French", "Han": "Han Chinese", "Papuan": "Papuans",
             "Mende": "Mende (Sierra Leone)", "Yoruba": "Yoruba (Nigeria)"}


def sec_neanderthal(out: Path) -> str:
    path = out / "neanderthal" / "neanderthal.csv"
    if not path.exists():
        return ""
    d = pd.read_csv(path)
    me = d[d["test"] == "you"].iloc[0]
    c = d[d["test"] != "you"]
    v = d["neanderthal"] * 100
    lo, hi = min(-1.0, float(np.floor(v.min()))), max(4.0, float(np.ceil(v.max())))
    pos = lambda x: (min(max(x, lo), hi) - lo) / (hi - lo) * 100
    ticks = list(range(int(lo), int(hi) + 1))
    grid = "".join(f'<div class="pp-grid" style="left:{pos(t):.2f}%"></div>' for t in ticks)
    grid += f'<div class="fp-mean" style="left:{pos(0):.2f}%"></div>'
    a, b = pos((me.neanderthal - 2 * me.neanderthal_se) * 100), pos((me.neanderthal + 2 * me.neanderthal_se) * 100)
    marks = (f'<div style="position:absolute;top:50%;height:2px;margin-top:-1px;left:{a:.2f}%;width:{b - a:.2f}%;'
             f'background:var(--ink);opacity:.35"></div><div class="f4-you" style="left:{pos(me.neanderthal * 100):.2f}%" '
             f'tabindex="0" data-tip="You&#10;{me.neanderthal * 100:.2f}% ± {me.neanderthal_se * 100:.2f}&#10;'
             f'{int(me.snps):,} SNPs"></div>')
    rows = [f'<div class="pp-label"><b>You</b></div><div class="fp-track">{grid}{marks}</div>']
    for pop, dp in c.groupby("pop", sort=False):
        mk = "".join(f'<div class="f4-c s1" style="left:{pos(r.neanderthal * 100):.2f}%" tabindex="0" '
                     f'data-tip="{esc(r.test)} ({esc(NEA_NAMES.get(pop, pop))})&#10;{r.neanderthal * 100:.2f}% ± '
                     f'{r.neanderthal_se * 100:.2f}"></div>' for r in dp.itertuples())
        m_ = dp["neanderthal"].mean() * 100
        mk += (f'<div style="position:absolute;top:50%;left:{pos(m_):.2f}%;width:3px;height:16px;margin:-8px 0 0 -1.5px;'
               f'border-radius:2px;background:var(--ink2)" tabindex="0" data-tip="{esc(NEA_NAMES.get(pop, pop))}: mean '
               f'{m_:.2f}%"></div>')
        rows.append(f'<div class="pp-label">{esc(NEA_NAMES.get(pop, pop))} <span class="muted">n={len(dp)}</span></div>'
                    f'<div class="fp-track">{grid}{mk}</div>')
    axis = "".join(f'<span style="left:{pos(t):.2f}%">{t}%</span>' for t in ticks)
    leg = ('<div class="legend"><span class="key"><i class="sw" style="background:var(--ink)"></i>You (line: ±2 SE)</span>'
           '<span class="key"><i class="sw" style="background:var(--ink2);width:3px"></i>Group mean</span>'
           '<span class="key"><i class="sw s1"></i>One dot per person, one read per SNP like you</span></div>')
    chart = leg + f'<div class="pp fp">{"".join(rows)}<div></div><div class="pp-axis">{axis}</div></div>'
    eur = c[c["pop"].isin(["Russian", "French"])]["neanderthal"] * 100
    afr = c[c["pop"].isin(["Yoruba", "Mende"])]["neanderthal"] * 100
    tiles = [tile("Your Neanderthal ancestry", f"{me.neanderthal * 100:.1f}%", f"± {me.neanderthal_se * 100:.1f} (1 SE)"),
             tile("Europeans on the same panel", f"{eur.min():.1f}–{eur.max():.1f}%", "Russian and French controls"),
             tile("Africans (no Neanderthal ancestry)", f"{afr.mean():.1f}%", "Yoruba and Mende: the zero point")]
    body = f'<div class="tiles">{"".join(tiles)}</div>'
    body += card("Neanderthal ancestry, f4(Altai, Chimp; X, Yoruba) / f4(Altai, Chimp; Vindija, Yoruba)", chart,
                 "The introgressing Neanderthals were closer to the Vindija Neanderthal (Croatia) than to the Altai one, so "
                 "the denominator is what 100% Neanderthal ancestry would give; Yoruba mark 0%. Hover for values.")
    body += card("What this can't tell", (
        "<p><b>Which Neanderthal variants you carry, gene by gene.</b> Neanderthal segments average about 50 kb, and "
        "finding them needs genotypes at dense sites. Your sequencing has 1–2 reads per position (1.8×), the AADR SNP "
        "panel has about one SNP per 5 kb and wasn't chosen for archaic variants, and the MyHeritage file lacks the "
        "best-known Neanderthal variants (OAS1 rs10774671, the chr3 COVID-risk haplotype rs35044562).</p>"
        "<p><b>Denisovan ancestry.</b> On this panel the estimate for one genome is too noisy (± ~13 points against a "
        "Papuan scale) to separate a European's expected ~0 from a few percent.</p>"))
    lede = ("All people outside Africa carry about 1–3% Neanderthal DNA from interbreeding some 45,000–55,000 years ago. "
            "Your share is measured against the high-coverage Altai and Vindija Neanderthal genomes in the AADR, with "
            "control individuals processed exactly like your data.")
    return section("neanderthal", "Neanderthal ancestry", esc(lede), body)


SHARED_CTRL = [("Russian", 1), ("Ukrainian", 2), ("Finnish", 3), ("Sardinian", 4), ("Nganasan", 5)]
SHARED_NAMES = {"WHG": "Western hunter-gatherers", "EHG": "Eastern hunter-gatherers", "Iran_N": "Iranian farmers",
                "BOO": "Bolshoy Oleni Ostrov (Kola, 3.5 ka)", "Nganasan": "Nganasan (Siberia, today)",
                "Han": "Han Chinese (today)", "Anatolia_N": "Anatolian farmers", "Yamnaya": "Yamnaya",
                "CordedWare": "Corded Ware"}


def sec_shared(out: Path) -> str:
    rp, vp = out / "shared_alleles" / "rates.csv", out / "shared_alleles" / "variants.csv"
    if not rp.exists() or not vp.exists():
        return ""
    r = pd.read_csv(rp)
    v = pd.read_csv(vp)
    summ = json.loads((out / "shared_alleles" / "summary.json").read_text())
    me = r[r["test"] == "you"].set_index("source")
    used = [s for s in me.index if me.at[s, "diagnostic"] >= 100]
    few = [s for s in me.index if me.at[s, "diagnostic"] < 100]
    lo, hi = 1.0, 70.0
    pos = lambda x: (np.log10(min(max(x, lo), hi)) - np.log10(lo)) / (np.log10(hi) - np.log10(lo)) * 100
    ticks = [1, 2, 5, 10, 20, 50]
    rows = []
    for s in used:
        m = me.loc[s]
        grid = "".join(f'<div class="pp-grid" style="left:{pos(t):.2f}%"></div>' for t in ticks)
        grid += (f'<div class="fp-mean" style="left:{pos(m.expected * 100):.2f}%" tabindex="0" data-tip="Average '
                 f'present-day European&#10;{m.expected * 100:.1f}%"></div>')
        mk = ""
        for pop, slot in SHARED_CTRL:
            for c in r[(r["source"] == s) & (r["pop"] == pop)].itertuples():
                mk += (f'<div class="f4-c s{slot}" style="left:{pos(c.rate * 100):.2f}%" tabindex="0" '
                       f'data-tip="{esc(c.test)} ({pop})&#10;{c.rate * 100:.1f}% of {c.snps} alleles"></div>')
        a, b = pos((m.rate - 2 * m.se) * 100), pos((m.rate + 2 * m.se) * 100)
        mk += (f'<div style="position:absolute;top:50%;height:2px;margin-top:-1px;left:{a:.2f}%;width:{b - a:.2f}%;'
               f'background:var(--ink);opacity:.35"></div><div class="f4-you" style="left:{pos(m.rate * 100):.2f}%" '
               f'tabindex="0" data-tip="You&#10;{m.rate * 100:.1f}% ± {m.se * 100:.1f}&#10;{int(m.snps)} of '
               f'{int(m.diagnostic)} diagnostic SNPs read"></div>')
        rows.append(f'<div class="pp-label">{esc(SHARED_NAMES.get(s, s))} <span class="muted">{int(m.diagnostic)} SNPs'
                    f'</span></div><div class="fp-track">{grid}{mk}</div>')
    axis = "".join(f'<span style="left:{pos(t):.2f}%">{t}%</span>' for t in ticks)
    leg = ('<div class="legend"><span class="key"><i class="sw" style="background:var(--ink)"></i>You (line: ±2 SE)</span>'
           + "".join(f'<span class="key"><i class="sw s{k}"></i>{p}</span>' for p, k in SHARED_CTRL)
           + '<span class="key"><i class="sw" style="background:var(--ink2);width:3px"></i>Average present-day European'
           '</span></div>')
    chart = leg + f'<div class="pp fp">{"".join(rows)}<div></div><div class="pp-axis">{axis}</div></div>'

    ctrl = r[r["test"] != "you"]
    ehg, rus = me.loc["EHG"] if "EHG" in me.index else None, ctrl[(ctrl["source"] == "EHG") & (ctrl["pop"] == "Russian")]
    tiles = [tile("Diagnostic alleles you carry", f"{summ['carried']}", f"seen in your reads, of ~{int(me['snps'].sum()):,} "
                  "diagnostic SNPs read"),
             tile("Also in the imputed CSV", f"{summ['carried_csv_confirmed']}",
                  f"{int(v['csv_copies'].isna().sum())} not in the CSV, {int((v['csv_copies'] == 0).sum())} contradicted")]
    if ehg is not None and len(rus):
        tiles.append(tile("Eastern hunter-gatherer alleles", f"{ehg.rate * 100:.1f}%",
                          f"Russians {rus['rate'].min() * 100:.1f}–{rus['rate'].max() * 100:.1f}%, ± {ehg.se * 100:.1f} (1 SE)"))
    tiles.append(tile("With a ClinVar record", f"{summ['clinvar_exact']}",
                      f"{summ['in_genes']} inside a gene; " + ("all benign" if summ["clinvar_exact"] and
                      v.loc[v["clinvar"].fillna("") != "", "clinvar"].str.contains("enign").all() else "see the table")))
    body = f'<div class="tiles">{"".join(tiles)}</div>'
    body += card("Share of each source's diagnostic alleles you carry", chart,
                 "Diagnostic = at most 5% in present-day Europeans (11 populations pooled) but at least 50% in the source. "
                 "One read per SNP for you and for every control person, whose own alleles are left out of the reference. "
                 "Log scale. Hover for values.")
    if few:
        body += ('<p class="muted">' + ", ".join(SHARED_NAMES.get(s, s) for s in few) + ": fewer than 100 diagnostic "
                 "SNPs (" + ", ".join(str(int(me.at[s, "diagnostic"])) for s in few) + "). Their ancestry is in every "
                 "present-day European, so almost none of their alleles are rare today. That is why the farmer and "
                 "steppe tests use f4 and qpAdm instead.</p>")
    t = v.copy()
    t["Today (Europe)"] = (t["ref_freq"] * 100).map(lambda x: f"{x:.1f}%")
    t["Highest in"] = t["best_source"].map(lambda s: SHARED_NAMES.get(s, s).split(" (")[0]) + " " + \
        (t["best_freq"] * 100).map(lambda x: f"{x:.0f}%")
    t["CSV"] = t["csv_copies"].map(lambda c: "–" if pd.isna(c) else f"{int(c)} cop{'y' if c == 1 else 'ies'}")
    t["Position"] = t["chrom"].astype(str) + ":" + t["pos"].map(lambda p: f"{p:,}")
    t["Gene"] = t["gene"].fillna("")
    t["ClinVar"] = (t["clinvar"].fillna("").str.replace("_", " ") + " " + t["consequence"].fillna("")
                    .str.replace("_", " ")).str.strip()
    t = t.rename(columns={"allele": "Allele"})[["rsid", "Position", "Allele", "Today (Europe)", "Highest in", "CSV",
                                                  "Gene", "ClinVar"]].rename(columns={"rsid": "SNP"})
    conf = t[t["CSV"].str.startswith(("1", "2"))]
    body += card("Diagnostic alleles seen in your reads and in the imputed CSV", table(conf),
                 "The most reliable calls: your reads show the allele and the imputed file agrees. Gene = inside the span of "
                 "that gene's ClinVar records.")
    body += card("All diagnostic alleles in your reads",
                 f'<details><summary>{len(t)} SNPs, sorted by frequency in the source</summary>{table(t)}</details>',
                 "Each rests on a single read at 1.8×, so any one of them may be a sequencing error. The aggregate rates "
                 "above are reliable; single rows are not.")
    body += card("What this can and can't tell", (
        "<p><b>Ancestry: yes, in aggregate.</b> Rare-today alleles are informative only as counts compared with "
        "control people. Nganasan controls carry about half of the Siberian-diagnostic alleles; Russians, Ukrainians "
        "and Finns carry 3–5%; Sardinians 2%. Your Siberian and BOO rates sit with Russians, which fits the few-percent "
        "share from qpAdm. Your Eastern hunter-gatherer rate is at the top of the Russian range.</p>"
        "<p><b>Biology: almost nothing.</b> The Human Origins SNPs were chosen as common, neutral markers for population "
        "history. Most of the diagnostic alleles you carry lie between genes; the few ClinVar records at these positions "
        "are benign. The known trait variants with ancient histories (lactase persistence LCT rs4988235, "
        "eye colour HERC2 rs12913832, skin SLC24A5 rs1426654 and SLC45A2 rs16891982) are not on this panel. Your "
        "genotypes for them are in the Traits section. Their frequencies in ancient groups need the 1240k AADR "
        "release.</p>"))
    lede = ("Which alleles do you share with ancient populations beyond what every European shares? Only alleles that are "
            "rare today but common in an ancient group carry information, and only when compared with people "
            "processed the same way.")
    return section("shared", "Variants shared with ancient populations", esc(lede), body)


def sec_roh(roh: pd.DataFrame | None) -> str:
    if roh is None:
        return ""
    longest = GRCH37_LEN["1"]
    rows = []
    for c in AUTOSOMES:
        segs = roh[roh["CHR"] == c]
        marks = "".join(
            f'<div class="roh-seg s1" style="left:{r.POS1 / GRCH37_LEN[c] * 100:.3f}%;'
            f'width:max(3px,{(r.POS2 - r.POS1) / GRCH37_LEN[c] * 100:.3f}%)" tabindex="0" '
            f'data-tip="chr{c}:{r.POS1:,}–{r.POS2:,}&#10;{r.KB / 1000:.2f} Mb, {r.NSNP} SNPs"></div>'
            for r in segs.itertuples())
        rows.append(f'<div class="ideo-label">{c}</div><div class="ideo-cell">'
                    f'<div class="ideo-track" style="width:{GRCH37_LEN[c] / longest * 100:.2f}%">{marks}</div></div>')
    total = roh["KB"].sum() / 1000
    longest_seg = roh["KB"].max() / 1000
    body = card("Runs of homozygosity along the autosomes",
                f'<div class="ideo">{"".join(rows)}</div>',
                f"{len(roh)} segments, {total:.1f} Mb in total, longest {longest_seg:.1f} Mb. Long runs (over ~10 Mb) "
                "would point to related parents; short ones are normal background in every population.")
    return section("roh", "Runs of homozygosity", "Stretches where both copies of a chromosome are identical.", body)


def sec_traits_prs(cfg, out: Path) -> str:
    cards = []
    tr = out / "traits.csv"
    if tr.exists():
        t = pd.read_csv(tr)
        t["copies"] = t["copies"].map(lambda v: "–" if pd.isna(v) else f"{int(v)}")
        t = t.rename(columns={"rsid": "SNP", "trait": "Trait", "genotype": "Genotype", "effect_allele": "Effect allele",
                              "copies": "Copies", "effect": "Effect"})
        cards.append(card("Single-SNP traits", table(t), "Effect alleles on the GRCh37 forward strand."))
    prs_path = out / "prs" / "prs_summary.csv"
    if prs_path.exists():
        p = pd.read_csv(prs_path)
        pcol = next(c for c in p.columns if c.startswith("percentile"))
        ref = pcol.removeprefix("percentile_vs_")
        meters = "".join(
            f'<div class="meter-row"><div class="meter-head"><b>{esc(r.trait)}</b> <span class="muted">{esc(r.pgs_id)}</span>'
            f'<span class="meter-val">{r[pcol]:.0f}th percentile</span></div>'
            f'<div class="meter"><div class="meter-mid"></div><div class="meter-dot" style="left:{r[pcol]:.1f}%" tabindex="0" '
            f'data-tip="{esc(r.trait)}&#10;percentile {r[pcol]:.1f} vs 1000G {ref}&#10;z = {r.z:+.2f}&#10;'
            f'{r.variants_used:,} of {r.variants_in_score:,} score variants used ({pct(r.coverage, 1)})"></div></div>'
            f'<div class="meter-scale"><span>0</span><span>50</span><span>100</span></div>'
            f'<div class="chart-note">Only {pct(r.coverage)} of the score\'s variants were available, so treat this as rough.</div></div>'
            for _, r in p.iterrows())
        cards.append(card(f"Polygenic scores vs 1000 Genomes {ref}", meters,
                          "Where your score falls among reference people. 50 = average; not a risk estimate."))
    if not cards:
        return ""
    # The traits table needs the full width; the PRS meters sit below it
    return section("traits", "Traits and polygenic scores", "", "".join(cards))


def sec_clinvar(out: Path) -> str:
    path = out / "clinvar_carriers.csv"
    if not path.exists():
        return ""
    cv = pd.read_csv(path)
    sig = cv["CLNSIG"].fillna("not_provided")
    counts = sig.str.replace("_", " ").value_counts()
    top = counts.head(9)
    if len(counts) > 9:
        top = pd.concat([top, pd.Series({"other": counts.iloc[9:].sum()})])
    rows = [(k, v, f"{v:,}", f"{k}\n{v:,} variants") for k, v in top.items()]
    path_mask = sig.str.contains("athogenic", case=True) & ~sig.str.contains("Conflicting")
    n_path = int(path_mask.sum())
    benign = sig.str.split(r"[|/,]").map(lambda xs: all("enign" in x for x in xs))
    keep = cv[~benign].copy()
    keep["condition"] = keep["CLNDN"].fillna("").str.replace("_", " ").str.replace("|", "; ", regex=False)
    keep = keep.sort_values(["stars", "gene"], ascending=[False, True])
    tbl = keep[["gene", "rsid", "genotype", "CLNSIG", "condition", "stars"]].rename(columns={
        "gene": "Gene", "rsid": "SNP", "genotype": "Genotype", "CLNSIG": "Significance", "condition": "Condition",
        "stars": "Review ★"})
    tbl["Significance"] = tbl["Significance"].str.replace("_", " ")
    status = (f'<p class="status {"crit" if n_path else "good"}"><b>{"⚠" if n_path else "✓"} '
              f'{n_path} pathogenic / likely pathogenic</b> variants among {len(cv):,} ClinVar sites you carry.</p>')
    body = card("ClinVar variants you carry, by significance", status + hbars(rows),
                "Most carried ClinVar variants are common benign ones. Confirm anything clinical with a clinical test.")
    body += details(f"Show the {len(tbl)} non-benign variants (drug response, risk factor, conflicting, …)",
                    table(tbl, "dense"))
    return section("clinvar", "ClinVar", "", body)


# ---------------------------------------------------------------- PharmCAT

PGX_SOURCES = ("CPIC", "DPWG", "FDA")
NOT_CALLED = {"", "n/a", "no result", "unknown", "unknown/unknown", "not called", "–"}
# A recommendation is actionable when it asks for a change, and is not a negation of one
PGX_ACTION = ("avoid", "alternative", "not recommended", "increas", "reduc", "decreas", "lower", "≤", "<=",
              "contraindicat", "do not use", "50%")
PGX_NEGATE = ("no need to avoid", "no reason to avoid", "genotype is available", "no recommendation",
              "no adjustments needed", "no action", "label recommended")
PGX_NO_CALL = {"no result", "uncertain susceptibility", "n/a", "indeterminate", ""}


def pgx_actionable(r: dict) -> bool:
    """Asks for a change, is not a negation, and rests on at least one real phenotype call."""
    text = r["rec"].lower()
    called = {p.strip().lower() for p in r["phenotype"].split(",")} - PGX_NO_CALL
    return bool(called) and any(k in text for k in PGX_ACTION) and not any(k in text for k in PGX_NEGATE)
STRENGTH = {"strong": 0, "moderate": 1, "optional": 2}


def _walk(o, path=(), up=()):
    """Every dict inside a JSON value, with its key path and its ancestor dicts."""
    if isinstance(o, dict):
        yield path, o, up
        for k, v in o.items():
            yield from _walk(v, path + (k,), up + (o,))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _walk(v, path + (i,), up)


def _source(path) -> str:
    return next((PGX_SOURCES[[s in str(k).upper() for s in PGX_SOURCES].index(True)]
                 for k in path if any(s in str(k).upper() for s in PGX_SOURCES)), "")


def _dip_label(d: dict) -> str:
    if d.get("label"):
        return str(d["label"])
    names = [(d.get(k) or {}).get("name") for k in ("allele1", "allele2")]
    return "/".join(n for n in names if n) or "–"


def _text(x) -> str:
    """Plain text from PharmCAT fields, which may be lists, dicts, or HTML (DPWG) with entities."""
    if isinstance(x, list):
        return "; ".join(_text(v) for v in x if v)
    if isinstance(x, dict):
        return _text(x.get("text") or x.get("name") or x.get("html") or "")
    t = html.unescape(re.sub(r"<[^>]+>", " ", str(x or "")))
    return re.sub(r"\s+", " ", t).strip().strip('"')


def load_pharmcat(out: Path) -> dict | None:
    """Genes and drug recommendations from PharmCAT's report JSON (or its phenotype JSON). The layout differs
    between PharmCAT releases, so records are found by their keys rather than by fixed paths."""
    d = out / "pharmcat"
    src = next((p for p in (d / "sample.report.json", d / "sample.phenotype.json") if p.exists()), None)
    if src is None:
        return None
    try:
        data = json.loads(src.read_text())
    except (OSError, ValueError) as e:
        print(f"Warning: could not read {src}: {e}")
        return None
    genes: dict[str, dict] = {}
    recs: dict[tuple, dict] = {}
    for path, o, up in _walk(data):
        if "geneSymbol" in o and ("sourceDiplotypes" in o or "recommendationDiplotypes" in o):
            dips = o.get("sourceDiplotypes") or o.get("recommendationDiplotypes") or []
            labels = list(dict.fromkeys(_dip_label(x) for x in dips))
            phen = list(dict.fromkeys(p for x in dips for p in (x.get("phenotypes") or []) if p))
            score = [str(x["activityScore"]) for x in dips if x.get("activityScore") not in (None, "", "n/a")]
            variants = [v for v in (o.get("variants") or []) if isinstance(v, dict)]
            called = (any(l.lower() not in NOT_CALLED and "unknown" not in l.lower() for l in labels)
                      and o.get("callSource") not in ("NONE",))
            g = genes.setdefault(o["geneSymbol"], {"gene": o["geneSymbol"], "called": False, "phen": {},
                                                    "diplotype": "–", "score": "", "pos": None})
            if called and not g["called"]:
                g.update(called=True, diplotype=" or ".join(labels), score=", ".join(score))
            if phen:
                g["phen"].setdefault(str(_source(path) or o.get("phenotypeSource") or ""), ", ".join(phen))
            if variants:
                have = sum(1 for v in variants if v.get("call") not in (None, "", "."))
                g["pos"] = (have, len(variants))
        rec = o.get("drugRecommendation")
        if isinstance(rec, (str, dict)) and _text(rec).strip():
            drug = ""
            if "drugs" in path:
                i = path.index("drugs")
                drug = next((str(k) for k in path[i + 1:i + 3]
                             if isinstance(k, str) and not _source((k,)) and "guideline" not in k.lower()), "")
            if not drug:
                drug = next((_text(a.get("name")) for a in reversed(up) if a.get("name")), "?")
            gene_names = sorted({str(x.get("gene") or x.get("geneSymbol")) for _, x, _ in _walk(o.get("genotypes", []))
                                 if x.get("gene") or x.get("geneSymbol")})
            pheno = sorted({p for _, x, _ in _walk(o.get("genotypes", [])) for p in (x.get("phenotypes") or [])})
            r = {"drug": drug, "source": _source(path), "genes": ", ".join(gene_names),
                 "phenotype": ", ".join(pheno), "class": _text(o.get("classification")),
                 "rec": _text(rec).strip(), "impl": _text(o.get("implications"))}
            recs.setdefault((r["drug"].lower(), r["source"], r["rec"]), r)
    stats = read_json(d / "pgx_input_stats.json")
    return {"genes": sorted(genes.values(), key=lambda g: (not g["called"], g["gene"])),
            "recs": list(recs.values()), "stats": stats, "src": src,
            "html": next((p for p in (d / "sample.report.html",) if p.exists()), None)}


def sec_pharmcat(out: Path, rep: Path) -> str:
    pc = load_pharmcat(out)
    if pc is None:
        if (out / "pharmcat" / "sample.pgx.grch38.vcf").exists():
            return section("pgx", "Drug response (PharmCAT)", "", card("PharmCAT did not finish",
                           f"<p>See {esc(out / 'pharmcat' / 'pharmcat.log')}.</p>"))
        return ""
    genes, recs, st = pc["genes"], pc["recs"], pc["stats"] or {}
    n_called = sum(g["called"] for g in genes)
    link = ""
    if pc["html"]:
        (rep / "pharmcat_full.html").write_bytes(pc["html"].read_bytes())
        link = ' <a href="pharmcat_full.html">PharmCAT\'s full report →</a>'
    tiles = [tile("Genes called", f"{n_called} of {len(genes)}", "the rest: no data or not callable from SNPs")]
    if st:
        tiles.append(tile("PharmCAT positions in your file", f"{st.get('written', 0):,} of {st.get('snv', 0):,}",
                          "SNVs; indels are not in the raw file"))
    actionable = [r for r in recs if pgx_actionable(r)]
    tiles.append(tile("Recommendations that change something", f"{len(actionable)}", f"of {len(recs)} drug–guideline pairs"))
    rows = []
    for g in genes:
        phen = "; ".join(f"{v} ({k})" if k and len(g["phen"]) > 1 else v for k, v in g["phen"].items()) or "–"
        pos = f"{g['pos'][0]} / {g['pos'][1]}" if g["pos"] else ""
        rows.append({"Gene": g["gene"], "Diplotype": g["diplotype"] if g["called"] else "no call",
                     "Phenotype": phen if g["called"] else "–", "Activity score": g["score"] or "",
                     "Positions with data": pos})
    gt = pd.DataFrame(rows)
    if not gt["Positions with data"].astype(bool).any():
        gt = gt.drop(columns="Positions with data")
    if not gt["Activity score"].astype(bool).any():
        gt = gt.drop(columns="Activity score")
    called = gt[gt["Diplotype"] != "no call"]
    body = f'<div class="tiles">{"".join(tiles)}</div>'
    body += card("Your pharmacogenes", table(called, "dense sortable") if len(called) else "<p>No gene could be called.</p>",
                 "Star-allele diplotypes and the phenotype CPIC/DPWG assign to them." + link)
    uncalled = gt[gt["Diplotype"] == "no call"]
    if len(uncalled):
        body += details(f"{len(uncalled)} genes without a call: " + ", ".join(uncalled["Gene"]),
                        "<p class=\"chart-note\">CYP2D6 needs copy-number and structural calls, HLA genes need typing; "
                        "others lack their key positions in the raw file.</p>")
    def rec_table(rs):
        rs = sorted(rs, key=lambda r: (STRENGTH.get(r["class"].lower().split()[0] if r["class"] else "", 9), r["drug"]))
        return table(pd.DataFrame([{"Drug": r["drug"], "Guideline": r["source"], "Genes": r["genes"],
                                    "Your phenotype": r["phenotype"], "Strength": r["class"],
                                    "Recommendation": r["rec"]} for r in rs]), "dense sortable")
    if actionable:
        body += card("Drugs where your genotype changes the advice", rec_table(actionable),
                     "From CPIC / DPWG guidelines as applied by PharmCAT. Talk to a doctor or pharmacist before acting on any of it.")
    routine = [r for r in recs if r not in actionable]
    if routine:
        body += details(f"Show {len(routine)} drugs with standard dosing for your genotype", rec_table(routine))
    lede = ("Imputed SNP data: a gene whose key positions are missing can come out as the reference allele (*1), so "
            "check “Positions with data” before trusting a normal result. Not a clinical test.")
    return section("pgx", "Drug response (PharmCAT)", esc(lede), body)


def sec_cram(cov, qc) -> str:
    if not cov and not qc:
        return ""
    cards = []
    if cov and "per_chrom" in cov:
        pc = cov["per_chrom"]
        rows = [(c, r.meandepth, f"chr{c}\nmean depth {r.meandepth:.2f}×\n{r.coverage:.1f}% of bases covered\n"
                                 f"{int(r.numreads):,} reads") for c, r in pc.iterrows()]
        cards.append(card("Mean read depth per chromosome", columns(rows),
                          "From the CRAM. X is at full depth because you have two X chromosomes."))
    if cov and "hist" in cov:
        h = cov["hist"]
        cap = 12
        b = h[h["depth"] < cap]
        tail = h.loc[h["depth"] >= cap, "n"].sum()
        tot = h["n"].sum()
        rows = [(str(int(r.depth)), r.n, f"depth {int(r.depth)}×\n{r.n:,} bases ({pct(r.n / tot, 1)})")
                for r in b.itertuples()]
        rows.append((f"{cap}+", tail, f"depth ≥ {cap}×\n{tail:,} bases ({pct(tail / tot, 1)})"))
        cards.append(card("Read depth across chromosome 1", columns(rows, y_label="Number of bases at each depth"),
                          f"{pct(h.loc[h['depth'] == 0, 'n'].sum() / tot)} of chr1 has no reads: typical for ~2× "
                          "low-pass sequencing, which is why genotypes are imputed."))
    if qc:
        rows = [(r["chrom"], r["het_rate"], f"chr{r['chrom']}\nheterozygosity {r['het_rate']:.4f}\n{r['n']:,} SNPs")
                for r in qc["per_chrom"]]
        cards.append(card("Heterozygosity per chromosome (raw file)", columns(rows),
                          "Even across autosomes, as expected; X is higher since it has two copies and different SNPs."))
    return section("sequencing", "Data quality", "", '<div class="grid2">' + "".join(cards) + "</div>")


# ---------------------------------------------------------------- page sections: qpAdm

def qp_findings(df: pd.DataFrame, analysis: str) -> str:
    out = []
    a = df[df["analysis"] == analysis]
    for target, t in a.groupby("target", sort=False):
        f = t[t["fits"]]
        if f.empty:
            out.append(f"<li><b>{esc(target)}</b>: none of {len(t)} models fit "
                       f"(best p = {fmt_p(t['p'].max())}).</li>")
            continue
        simple = f[f["n_sources"] == f["n_sources"].min()].sort_values("p", ascending=False).iloc[0]
        best = f.sort_values("p", ascending=False).iloc[0]
        txt = (f"<li><b>{esc(target)}</b>: {len(f)} of {len(t)} models fit. Simplest: "
               f"<b>{esc(simple['sources'])}</b> (p = {fmt_p(simple['p'])})")
        if best["sources"] != simple["sources"]:
            txt += f"; highest p: {esc(best['sources'])} (p = {fmt_p(best['p'])})"
        out.append(txt + ".</li>")
    return "<ul class='findings'>" + "".join(out) + "</ul>"


def simplest_fit(df: pd.DataFrame, analysis: str, target: str):
    """The fitting model with the fewest sources (highest p among those), or None."""
    f = df[(df["analysis"] == analysis) & (df["target"] == target) & df["fits"]]
    if f.empty:
        return None
    return f[f["n_sources"] == f["n_sources"].min()].sort_values("p", ascending=False).iloc[0]


def model_line(r, slots, label: str) -> str:
    verdict = "fits" if r["fits"] else "rejected"
    bar = (model_bar(r, slots, "" if r["fits"] else "faded") if r["valid"]
           else '<div class="segbar-empty">weights outside 0–1</div>')
    return (f'<div class="mr-target">{esc(label)}</div><div class="mr-bar">{bar}</div>'
            f'<div class="mr-p"><span class="badge {verdict}">{"✓" if r["fits"] else "✕"} {verdict}</span> '
            f'p = {fmt_p(r["p"])}</div>')


def glance_card(cfg, df: pd.DataFrame, analysis: str, title: str, sub: str = "") -> str:
    """Simplest fitting model per target, as weight bars (each target's own best model)."""
    a = df[df["analysis"] == analysis]
    slots = source_slots(cfg, analysis, df)
    lines, used = [], []
    for t in dict.fromkeys(a["target"]):
        r = simplest_fit(df, analysis, t)
        if r is None:
            lines.append(f'<div class="mr-target">{esc(t)}</div><div class="mr-bar muted">no model fits</div><div></div>')
            continue
        lines.append(model_line(r, slots, t))
        used += r["src"]
    if not used:
        return card(title, "<p>No model fits for any input.</p>", sub)
    return card(title, slot_legend(slots, used) + f'<div class="mr glance">{"".join(lines)}</div>', sub)


def forest(df: pd.DataFrame, source: str, p_min: float) -> str:
    """Weight of one source (± 95% CI) in every fitting model that includes it, per target."""
    f = df[df["fits"] & df["src"].map(lambda x: source in x and len(x) > 1)].copy()
    if f.empty:
        return ""
    f["w"] = [r["weights"][r["src"].index(source)] for _, r in f.iterrows()]
    f["e"] = [r["se"][r["src"].index(source)] for _, r in f.iterrows()]
    targets = list(dict.fromkeys(df["target"]))
    xmax = min(1.0, max(0.05, float((f["w"] + 1.96 * f["e"]).max()) * 1.15))
    pos = lambda v: max(0.0, min(v, xmax)) / xmax * 100
    keys = f.sort_values(["analysis", "n_sources", "p"], ascending=[True, True, False])[["analysis", "sources"]] \
        .drop_duplicates().itertuples(index=False)
    rows = []
    for analysis, sources in keys:
        m = f[(f["analysis"] == analysis) & (f["sources"] == sources)].set_index("target")
        marks = ""
        for i, t in enumerate(targets):
            if t not in m.index:
                continue
            w, e = m.loc[t, "w"], m.loc[t, "e"]
            top = 30 + 40 * i if len(targets) > 1 else 50
            marks += (f'<div class="fp-ci s{i + 1}" style="left:{pos(w - 1.96 * e):.2f}%;'
                      f'width:{pos(w + 1.96 * e) - pos(w - 1.96 * e):.2f}%;top:{top}%"></div>'
                      f'<div class="fp-dot s{i + 1}" style="left:{pos(w):.2f}%;top:{top}%" tabindex="0" '
                      f'data-tip="{esc(sources)} ({esc(analysis)})&#10;{esc(t)}: {source} {pct(w, 1)} ± {pct(e, 1)}'
                      f'&#10;95% CI {pct(max(w - 1.96 * e, 0), 1)}–{pct(w + 1.96 * e, 1)}&#10;p = {fmt_p(m.loc[t, "p"])}"></div>')
        rows.append(f'<div class="pp-label"><span class="muted">{esc(analysis)}</span> {esc(sources)}</div>'
                    f'<div class="fp-track">{marks}</div>')
    # inverse-variance mean over all fitting estimates
    wts = 1 / np.maximum(f["e"].to_numpy(), 1e-3) ** 2
    mean = float((f["w"] * wts).sum() / wts.sum())
    ticks = nice_ticks(0, xmax, 5)
    tick_lbl = "".join(f'<span style="left:{pos(t):.2f}%">{t * 100:.0f}%</span>' for t in ticks if t <= xmax + 1e-9)
    grid = "".join(f'<div class="pp-grid" style="left:{pos(t):.2f}%"></div>' for t in ticks if t <= xmax + 1e-9)
    body = "".join(r.replace('<div class="fp-track">', f'<div class="fp-track">{grid}'
                             f'<div class="fp-mean" style="left:{pos(mean):.2f}%"></div>') for r in rows)
    return (legend([(t, i + 1) for i, t in enumerate(targets)]) +
            f'<div class="pp fp">{body}<div></div><div class="pp-axis">{tick_lbl}</div></div>'
            f'<div class="chart-note">Dot = weight, line = 95% interval, one row per fitting model (p ≥ {p_min}). '
            f'The vertical line is the precision-weighted mean, {pct(mean, 1)}.</div>')


P_BINS = [(0.5, "q4", "p ≥ 0.5"), (0.2, "q3", "0.2–0.5"), (0.05, "q2", "0.05–0.2"), (0.0, "q1", "fits, p < 0.05")]


def heatmap(df: pd.DataFrame, cfg, analysis: str, target: str) -> str:
    """Upper triangle of source × source: diagonal = that source alone, off-diagonal = the pair."""
    a = df[(df["analysis"] == analysis) & (df["target"] == target)].set_index("sources")
    srcs = cfg["qpadm"]["analyses"][analysis]["sources"]
    n = len(srcs)
    fits = a[a["fits"]]
    best = None
    if len(fits[fits["n_sources"] <= 2]):
        b = fits[fits["n_sources"] <= 2].sort_values(["n_sources", "p"], ascending=[True, False])
        best = b.index[0]
    cells = ['<div></div>'] + [f'<div class="hm-col"><span>{esc(s)}</span></div>' for s in srcs]
    for i, si in enumerate(srcs):
        cells.append(f'<div class="hm-row">{esc(si)}</div>')
        for j, sj in enumerate(srcs):
            if j < i:
                cells.append('<div class="hm-empty"></div>')
                continue
            key = si if i == j else f"{si} + {sj}"
            if key not in a.index:
                cells.append('<div class="hm-empty"></div>')
                continue
            r = a.loc[key]
            if r["fits"]:
                cls = next(c for lo, c, _ in P_BINS if r["p"] >= lo)
            else:
                cls = "rej" if r["valid"] else "inv"
            ws = ", ".join(f"{s} {pct(w)} ± {pct(e)}" for s, w, e in zip(r["src"], r["weights"], r["se"]))
            verdict = "fits" if r["fits"] else ("rejected" if r["valid"] else "rejected: weights outside 0–1")
            tip = f"{key}\n{target}: p = {fmt_p(r['p'])}, {verdict}\n{ws}"
            cells.append(f'<div class="hm-cell {cls}{" best" if key == best else ""}{" diag" if i == j else ""}" '
                         f'tabindex="0" data-tip="{esc(tip)}"></div>')
    return (f'<div class="hm-wrap"><div class="hm-title">{esc(target)}</div>'
            f'<div class="hm" style="grid-template-columns:max-content repeat({n},26px)">{"".join(cells)}</div></div>')


def heatmap_legend() -> str:
    keys = "".join(f'<span class="key"><i class="sw hm-sw {c}"></i>{esc(t)}</span>' for _, c, t in P_BINS)
    keys += '<span class="key"><i class="sw hm-sw rej"></i>rejected</span>'
    keys += '<span class="key"><i class="sw hm-sw inv"></i>weights outside 0–1</span>'
    keys += '<span class="key"><i class="sw hm-sw q3 best"></i>simplest best fit</span>'
    return f'<div class="legend">{keys}</div>'


def qp_heatmaps(cfg, df: pd.DataFrame, analysis: str) -> str:
    targets = list(dict.fromkeys(df.loc[df["analysis"] == analysis, "target"]))
    maps = "".join(heatmap(df, cfg, analysis, t) for t in targets)
    return (heatmap_legend() + f'<div class="hm-row-wrap">{maps}</div>'
            '<div class="chart-note">Diagonal: that population alone. Other cells: the pair of its row and column. '
            'Darker blue = higher p (better fit). Hover a cell for p and weights.</div>')


def qp_model_rows(cfg, df: pd.DataFrame, analysis: str, show: int = 6, min_sources: int = 1) -> str:
    """Each model that fits for any target: one segmented bar per target, rejected ones faded.
    The first `show` models are listed, the rest sit behind a 'show more'."""
    a = df[(df["analysis"] == analysis) & (df["n_sources"] >= min_sources)]
    slots = source_slots(cfg, analysis, df)
    targets = list(dict.fromkeys(a["target"]))
    fit_any = a.groupby("sources")["fits"].any()
    keep = a[a["sources"].isin(fit_any[fit_any].index)]
    order = (keep[keep["target"] == targets[0]].sort_values(["n_sources", "p"], ascending=[True, False])["sources"]
             .tolist())
    order += [s for s in dict.fromkeys(keep["sources"]) if s not in order]
    rows = []
    for sources in order:
        m = keep[keep["sources"] == sources].set_index("target")
        cells = "".join(model_line(m.loc[t], slots, t) for t in targets if t in m.index)
        rows.append(f'<div class="mr"><div class="mr-name">{esc(sources)}</div>{cells}</div>')
    if not rows:
        return "<p>No model fits for any target.</p>"
    html_ = f'<div class="models">{"".join(rows[:show])}</div>'
    if len(rows) > show:
        html_ += details(f"Show {len(rows) - show} more models", f'<div class="models">{"".join(rows[show:])}</div>')
    return slot_legend(slots, [s for x in keep["src"] for s in x]) + html_


def pick_models(a: pd.DataFrame, top: int) -> list[str]:
    """Models that fit for any target first (fewest sources, then highest p), then the best of the rest."""
    g = a.groupby("sources").agg(fits=("fits", "any"), valid=("valid", "any"), n=("n_sources", "first"), p=("p", "max"))
    g = g.sort_values(["fits", "valid", "n", "p"], ascending=[False, False, True, False])
    return g.head(top).index.tolist()


def qp_pplot(df: pd.DataFrame, analysis: str, p_min: float, top: int = 16, only_n: int | None = None,
             floor: int = -20) -> str:
    """p-values on a log axis, one row per model, a dot per target. Right of the line = fits."""
    a = df[df["analysis"] == analysis]
    if only_n:
        a = a[a["n_sources"] == only_n]
    targets = list(dict.fromkeys(a["target"]))
    best = pick_models(a, top)
    a = a[a["sources"].isin(best)]
    lo = max(math.floor(np.log10(max(a["p"].min(), 1e-300))), floor)
    lo = min(lo, -3)
    pos = lambda p: (math.log10(max(p, 10 ** lo)) - lo) / -lo * 100
    step = max(1, math.ceil(-lo / 6))
    ticks = list(range(lo, 1, step))
    if ticks[-1] != 0:
        ticks.append(0)
    grid = "".join(f'<div class="pp-grid" style="left:{(t - lo) / -lo * 100:.2f}%"></div>' for t in ticks)
    clamped = (a["p"] < 10 ** lo).any()
    tick_lbl = "".join(f'<span style="left:{(t - lo) / -lo * 100:.2f}%">'
                       f'{"1" if t == 0 else ("<" if t == lo and clamped else "") + f"1e{t}"}</span>' for t in ticks)
    thr = f'<div class="pp-thr" style="left:{pos(p_min):.2f}%"></div>'
    rows = []
    for sources in best:
        m = a[a["sources"] == sources].set_index("target")
        dots = "".join(
            f'<div class="pp-dot s{i + 1}{" open" if not m.loc[t, "fits"] else ""}" style="left:{pos(m.loc[t, "p"]):.2f}%" '
            f'tabindex="0" data-tip="{esc(sources)}&#10;{esc(t)}: p = {fmt_p(m.loc[t, "p"])}'
            f'{"" if m.loc[t, "valid"] else " (weights outside 0–1)"}"></div>'
            for i, t in enumerate(targets) if t in m.index)
        rows.append(f'<div class="pp-label">{esc(sources)}</div><div class="pp-track">{grid}{thr}{dots}</div>')
    return (legend([(t, i + 1) for i, t in enumerate(targets)]) +
            f'<div class="pp">{"".join(rows)}<div></div><div class="pp-axis">{tick_lbl}'
            f'<span class="pp-thr-label" style="left:{pos(p_min):.2f}%">p = {p_min}</span></div></div>'
            f'<div class="chart-note">{len(best)} models{"" if only_n else ": those that fit for either input, then the best of the rest"}. Log scale. Right of the dashed threshold = not rejected. '
            f'A hollow dot fails (p too low or a weight outside 0–1).</div>')


def qp_table(df: pd.DataFrame) -> str:
    t = df.copy()
    t["weights ± se"] = [", ".join(f"{s} {w:.2f}±{e:.2f}" for s, w, e in zip(r["src"], r["weights"], r["se"]))
                         if r["status"] == "ok" else r["status"] for _, r in t.iterrows()]
    t["p"] = t["p"].map(fmt_p)
    t["fits"] = t["fits"].map({True: "✓", False: ""})
    view = t[["analysis", "target", "n_sources", "sources", "p", "weights ± se", "fits"]]
    attrs = lambda i: (f'data-analysis="{esc(df.iloc[i]["analysis"])}" data-target="{esc(df.iloc[i]["target"])}" '
                       f'data-fits="{int(df.iloc[i]["fits"])}"')
    sel = lambda name, vals: (f'<label>{name} <select data-filter="{name}"><option value="">all</option>' +
                              "".join(f"<option>{esc(v)}</option>" for v in vals) + "</select></label>")
    controls = (f'<div class="filters">{sel("analysis", dict.fromkeys(df["analysis"]))}'
                f'{sel("target", dict.fromkeys(df["target"]))}'
                '<label><input type="checkbox" data-filter="fits"> only models that fit</label>'
                '<span class="muted" id="row-count"></span></div>')
    return controls + table(view, "dense models-table sortable", attrs)



def qp_replicates(cfg, out: Path, per_analysis: int = 4) -> str:
    """Section body: how qpAdm results vary over repeated pseudo-haploid draws (20_qpadm_replicates.py)."""
    path = out / "qpadm" / "replicates" / "summary.csv"
    if not path.exists():
        return ""
    s = pd.read_csv(path)
    R = int(s["replicates"].max())
    fit = s[s["fit_share"] > 0]
    sd_cols = [c for c in s if c.endswith("_sd")]
    jk_cols = [c.replace("_sd", "_jk_se") for c in sd_cols]
    multi = fit[fit["n_sources"] > 1]
    sd = np.nanmedian(multi[sd_cols].to_numpy(float)) if len(multi) else np.nan
    jk = np.nanmedian(multi[jk_cols].to_numpy(float)) if len(multi) else np.nan
    flip = s[(s["fit_share"] > 0) & (s["fit_share"] < 1)]
    tiles = [tile("Replicates", f"{R}", "one random read per SNP, new seed each time"),
             tile("Weight spread between draws", f"{sd * 100:.1f} pts" if np.isfinite(sd) else "–",
                  f"median SD; qpAdm's own SE is {jk * 100:.1f} pts" if np.isfinite(jk) else ""),
             tile("Models whose verdict changes", f"{len(flip)} of {len(s)}", "fit in some draws but not all")]
    rows = []
    for analysis in cfg["qpadm"]["analyses"]:
        d = fit[fit["analysis"] == analysis].sort_values(["n_sources", "fit_share", "p_median"],
                                                         ascending=[True, False, False]).head(per_analysis)
        for _, r in d.iterrows():
            srcs = r["sources"].split(" + ")
            w = "; ".join(f"{src} {r[f'w{j + 1}_mean'] * 100:.1f}% ± {r[f'w{j + 1}_sd'] * 100:.1f} "
                          f"(SE {r[f'w{j + 1}_jk_se'] * 100:.1f})" for j, src in enumerate(srcs)) if len(srcs) > 1 else "–"
            rows.append({"Analysis": analysis, "Model": r["sources"],
                         "Fits in": f"{round(r['fit_share'] * R)} / {R}",
                         "p median [5–95%]": f"{fmt_p(r['p_median'])} [{fmt_p(r['p_q05'])}–{fmt_p(r['p_q95'])}]",
                         "Weight: mean ± SD between draws (jackknife SE)": w})
    tbl = table(pd.DataFrame(rows), "dense sortable") if rows else "<p>No model fits in any replicate.</p>"
    note = ("Only the ~11% of SNPs where your reads show both alleles change between draws, so the spread is small. "
            "Where it is far below qpAdm's jackknife SE, the single draw in the main results is representative.")
    return (f'<div class="tiles">{"".join(tiles)}</div>'
            + card(f"Simplest fitting models per analysis, over {R} draws", tbl, esc(note)))


EAST_NAMES = {"BOO": "Bolshoy Oleni Ostrov (Kola, ~3.7 kya)", "Saami_Historic": "Saami, Chalmny Varre (~150 ya)",
              "Levanluhta": "Levänluhta (Finland, ~1.4 kya)", "SeimaTurbino": "Seima-Turbino (Irtysh, ~3.9 kya)",
              "WSHG": "West Siberian HG (~7.8 kya)", "Botai": "Botai (Kazakhstan, ~5.3 kya)",
              "Baikal_EN": "Baikal Early Neolithic (~7.6 kya)", "Krasnoyarsk_MLBA": "Krasnoyarsk MLBA (~3.6 kya)",
              "Chiyalik": "Chiyalik (Volga-Ural, ~0.7 kya)", "EarlyBulgar": "Volga Bulgars (~0.95 kya)",
              "Khazar": "Khazars (Samara, ~1.2 kya)", "GoldenHorde": "Golden Horde (~0.6 kya)",
              "Conqueror": "Hungarian Conquerors (~1.0 kya)", "Avar_EastAsian": "Avars, East Asian-rich (~1.25 kya)",
              "Mongol_LateMed": "Mongols, Mongol Empire era (~0.7 kya)", "Nganasan": "Nganasan (modern)",
              "Mansi": "Mansi (modern)", "Khanty": "Khanty (modern)", "Selkup": "Selkup (modern)", "Yakut": "Yakut (modern)",
              "(none)": "no East Eurasian source"}


def east_pchart(m: pd.DataFrame, cats: dict, base: str, p_min: float) -> str:
    """p of base + candidate, log axis: you (dot = original draw, line = 5-95% of redraws) and Russians."""
    d = m[(m["mode"] == "single") & (m["base"] == base)]
    lo = -8
    pos = lambda p: (math.log10(min(max(p, 10 ** lo), 1)) - lo) / -lo * 100
    ticks = list(range(lo, 1, 2))
    grid = "".join(f'<div class="pp-grid" style="left:{(t - lo) / -lo * 100:.2f}%"></div>' for t in ticks)
    thr = f'<div class="pp-thr" style="left:{pos(p_min):.2f}%"></div>'
    rows = []
    order = [("", ["(none)"])] + list(cats.items())
    for cat, cands in order:
        if cat:
            rows.append(f'<div class="pp-label f4-cat">{esc(cat)}</div><div></div>')
        for c in cands:
            dc = d[d["candidate"] == c]
            marks = ""
            me = dc[dc["target"] == "me_cram"]
            if len(me) > 1:
                q5, q95 = me["p"].quantile([0.05, 0.95])
                a_, b_ = pos(q5), pos(q95)
                marks += (f'<div style="position:absolute;top:50%;height:3px;margin-top:-1.5px;left:{a_:.2f}%;'
                          f'width:{max(b_ - a_, 0.3):.2f}%;background:var(--s1);opacity:.35;border-radius:2px"></div>')
            for t, cls, lab in (("Russian", "s4", "Russians (AADR, 98 people)"), ("me_cram", "s1", "You")):
                r0 = dc[(dc["target"] == t) & (dc["rep"] == 0)]
                if r0.empty:
                    continue
                r = r0.iloc[0]
                w = float(str(r["weights"]).split()[-1]) if c != "(none)" else float("nan")
                fs = dc[dc["target"] == t]["fits"].mean()
                extra = (f"&#10;fits in {fs:.0%} of {len(dc[dc['target'] == t])} draws" if t == "me_cram" and len(me) > 1 else "")
                marks += (f'<div class="pp-dot {cls}{"" if r["fits"] else " open"}" style="left:{pos(r["p"]):.2f}%" '
                          f'tabindex="0" data-tip="{esc(lab)}: {esc(r["sources"])}&#10;p = {fmt_p(r["p"])}'
                          f'{"" if np.isnan(w) else f"; weight {w * 100:.1f}%"}{extra}'
                          f'{"" if r["fits"] or r["p"] < p_min else " (a weight outside 0–1)"}"></div>')
            rows.append(f'<div class="pp-label">{esc(EAST_NAMES.get(c, c))}</div><div class="pp-track">{grid}{thr}{marks}</div>')
    lbl = "".join(f'<span style="left:{(t - lo) / -lo * 100:.2f}%">{"1" if t == 0 else ("≤" if t == lo else "") + f"1e{t}"}</span>'
                  for t in ticks)
    leg = legend([("You (line: 5–95% of random-read redraws)", 1), ("Russians (population)", 4)])
    return (leg + f'<div class="pp">{"".join(rows)}<div></div><div class="pp-axis">{lbl}'
            f'<span class="pp-thr-label" style="left:{pos(p_min):.2f}%">p = {p_min}</span></div></div>'
            '<div class="chart-note">Log scale. Right of the dashed line = model not rejected; hollow = rejected '
            '(p too low or a weight outside 0–1).</div>')


def qp_east(cfg) -> str:
    work = cfg["out_dir"] / "qpadm" / "east"
    if not (work / "models.csv").exists() or not (work / "compete.csv").exists():
        return ""
    e, p_min = cfg["east"], cfg["qpadm"]["p_min"]
    m = pd.read_csv(work / "models.csv")
    comp = pd.read_csv(work / "compete.csv")
    cats = e["candidates"]
    R = int(m[m["target"] == "me_cram"]["rep"].nunique())
    b2, b3 = (" + ".join(b) for b in e["bases"][:2])
    d2 = m[(m["mode"] == "single") & (m["base"] == b2) & (m["rep"] == 0) & (m["candidate"] != "(none)")]
    you2 = d2[d2["target"] == "me_cram"].sort_values("p", ascending=False)
    ru2 = d2[d2["target"] == "Russian"].sort_values("p", ascending=False)
    n_c = len(you2)
    tiles = [tile("Candidates that fit you", f"{int(you2['fits'].sum())} of {n_c}", f"{b2} + candidate, original draw"),
             tile("Candidates that fit Russians", f"{int(ru2['fits'].sum())} of {n_c}", "same models, 98 genomes"),
             tile("Best for you", EAST_NAMES.get(you2.iloc[0]["candidate"], you2.iloc[0]["candidate"]).split(" (")[0],
                  f"p = {fmt_p(you2.iloc[0]['p'])}"),
             tile("Best for Russians", EAST_NAMES.get(ru2.iloc[0]["candidate"], ru2.iloc[0]["candidate"]).split(" (")[0],
                  f"p = {fmt_p(ru2.iloc[0]['p'])}")]
    body = f'<div class="tiles">{"".join(tiles)}</div>'
    body += card("What this shows", (
        "<p><b>Your genome alone can't pick the source.</b> At 1.8× almost every candidate fits, and head to head "
        "none clearly beats another. Only groups without the right kind of East Eurasian ancestry (West Siberian "
        "hunter-gatherers, Botai, Krasnoyarsk MLBA) are rejected.</p>"
        "<p><b>Russians, with the same models, can.</b> With an Early Slav base, old north-eastern groups fit: "
        "Bolshoy Oleni Ostrov (Kola, ~3.7 kya), Seima-Turbino (~3.9 kya), the Kola Saami, and more weakly the Ob-Ugric and "
        "Samoyedic peoples. Nganasan itself and every medieval steppe group are rejected, from Volga Bulgars and Khazars "
        "to Hungarian Conquerors, Avars, the Golden Horde and Mongol-era Mongolia.</p>"
        "<p><b>Your ranking follows the Russian one</b> (best: Bolshoy Oleni Ostrov, then Seima-Turbino and Saami; "
        "Mongol-era last), and the f4 test on the overview page puts your Siberian signal at a typical Russian's level. "
        "The simplest reading: your ~6–9% is the same component Russians carry, Bronze Age Siberian ancestry that "
        "reached north-eastern Europe with Uralic-associated peoples about 4,000 years ago and came to Russians "
        "through Finnic and Volga neighbours, not admixture from the Mongol period.</p>"))
    body += card(f"{b2} + one East Eurasian candidate", east_pchart(m, cats, b2, p_min),
                 "Each candidate takes Nganasan's place. Medieval groups that are mostly European themselves need a larger "
                 "weight than Nganasan for the same signal, so compare p-values, not weights.")
    body += ("<details><summary>Same with " + esc(b3) + " + candidate</summary>"
             + card(f"{b3} + one East Eurasian candidate", east_pchart(m, cats, b3, p_min),
                    "Adding EHG lets the model take hunter-gatherer ancestry separately, so candidates that carry a "
                    "lot of it (BOO, Saami) lose their advantage.") + "</details>")
    chk = pd.read_csv(work / "outgroup_check.csv") if (work / "outgroup_check.csv").exists() else None
    bad = set(chk.loc[~chk["usable_outgroup"], "candidate"]) if chk is not None else set()
    rows = []
    for t, lab in (("me_cram", "You"), ("Russian", "Russians")):
        d = comp[comp["target"] == t]
        for r in d.itertuples():
            rows.append({"Target": lab, "Candidate": EAST_NAMES.get(r.candidate, r.candidate), "Group": r.category,
                         "Survives": f"{r.survived} of {r.of}" + (" *" if r.candidate in bad else ""),
                         "Rejected when this is an outgroup":
                         ", ".join(EAST_NAMES.get(x, x).split(" (")[0] for x in str(r.loses_to).split()) if
                         isinstance(r.loses_to, str) else ""})
    note = (f"{' + '.join(e['compete_base'])} + A, with candidate B added to the outgroups. A survives B when the model "
            "still fits, meaning B shares no extra drift with the target beyond what A provides. "
            + (f"For you, survival is the share of {R} draws (≥ 50% counts). " if R > 1 else "")
            + "The last column lists candidates that reject A while surviving A themselves. "
            + (f"* Not usable as an outgroup ({', '.join(EAST_NAMES.get(b, b).split(' (')[0] for b in sorted(bad))}): "
               "with any of them added, even Czechs (almost no East Eurasian ancestry) are rejected, because these "
               "groups carry West Eurasian ancestry of their own. They are tested as sources only, so their survival "
               "is not checked in reverse." if bad else ""))
    body += card("Head to head (rotating outgroups)", table(pd.DataFrame(rows), "dense sortable"), esc(note))
    return body


def qpadm_page(cfg, df: pd.DataFrame, source: Path) -> str:
    q = cfg["qpadm"]
    p_min = q["p_min"]
    secs = []
    intro = (
        "<p>qpAdm asks whether your genome can be written as a mix of a few <i>source</i> populations, using "
        "f4-statistics against a fixed set of distant <i>outgroups</i>. Each model gets a p-value: "
        f"<b>p ≥ {p_min} and all weights between 0 and 1 means the model is not rejected</b>. A high p does not "
        "prove a model; when several fit, prefer the one with the fewest sources.</p>"
        "<p>You enter twice: <b>me_cram</b> (one random read per SNP from the raw sequencing, ~489k SNPs) and "
        "<b>me_csv</b> (MyHeritage's imputed genotypes, only ~58k SNPs overlap the AADR panel). me_cram is the "
        "main result; me_csv is a cross-check on far fewer SNPs. Trust a model most when it fits for both.</p>")
    if "fast" in source.name:
        intro += (
            "<p class='muted'>Computed with scripts/13b_qpadm_fast.py (NumPy). On me_cram it reproduces ADMIXTOOLS "
            "qpAdm closely (see outputs/qpadm/fast_vs_admixtools.csv). Each f4-statistic uses the SNPs shared by "
            "its own four groups, so me_csv's statistics all stay on its ~58k SNPs; ADMIXTOOLS' qpfstats mixes "
            "them with statistics on the full SNP set, which rejected almost every me_csv model.</p>")
    else:
        intro += "<p class='muted'>Computed with ADMIXTOOLS qpAdm (scripts/13_qpadm.py).</p>"
    # At a glance: the simplest fitting model of each ancient analysis, the Siberian-related share, closest single
    glance = []
    for analysis, title in (("deep", "Deep ancestry"), ("proximal", "Closer-in-time sources")):
        if analysis in set(df["analysis"]):
            glance.append(glance_card(cfg, df, analysis, title, "Simplest model that fits, for each input."))
    single = next((a for a in ("modern_rotate", "modern_wide", "modern") if a in set(df["analysis"])), None)
    gl = '<div class="grid2">' + "".join(glance) + "</div>" if glance else ""
    if single:
        gl += card("Closest single modern population",
                   qp_pplot(df, single, p_min, top=12, only_n=1, floor=-6),
                   f"Each population alone as your source ({esc(single)} analysis). Higher p = closer fit.")
    fp = forest(df[df["analysis"].isin(["deep", "proximal"])], "Nganasan", p_min)
    if fp:
        gl += card("Siberian-related (Nganasan-like) share across models", fp,
                   "The most stable signal: it appears in every ancient model that fits, at a similar level.")
    if gl:
        secs.append(section("glance", "At a glance", "", gl))
    rep = qp_replicates(cfg, cfg["out_dir"])
    if rep:
        secs.append(section("replicates", "Robustness to the random read",
                            "me_cram takes one random read per SNP. Repeating that draw shows how much of each result "
                            "is luck of the draw.", rep))
    east = qp_east(cfg)
    if east:
        secs.append(section("east", "Where does the Siberian-related share come from?",
                            "Nganasan is a modern stand-in. Here about 20 ancient and modern candidates take its place: "
                            "early Uralic-associated groups of north-eastern Europe, deep Siberian groups, and medieval "
                            "steppe and Volga peoples. Which fit, and which beat the others head to head?", east))
    secs.append(section("about", "How to read this", "", card("qpAdm in short", intro)))
    titles = {"deep": ("Deep ancestry", "Hunter-gatherers (WHG, EHG), early farmers (Anatolia_N), steppe herders "
                                        "(Yamnaya) and a Siberian-related source (Nganasan)."),
              "modern": ("Modern sources", "Is a Slavic source enough, or is a Volga–Uralic / Finnic one needed?"),
              "modern_wide": ("Modern sources, wider", "Is the non-Slavic part Baltic, Finnic, Volga-Turkic or "
                                                       "Central European? Up to two sources per model."),
              "modern_rotate": ("Modern sources, head to head",
                                "Rotating design: the candidates not used in a model become outgroups, so a model "
                                "fits only if it beats all the others. Extra modern outgroups add resolution."),
              "proximal": ("Proximal sources", "Closer-in-time sources: Corded Ware (~4.5 kya, steppe + farmer) "
                                               "and Early Slavs (~1.2 kya).")}
    for analysis in q["analyses"]:
        if analysis not in set(df["analysis"]):
            continue
        title, lede = titles.get(analysis, (analysis, ""))
        spec = q["analyses"][analysis]
        right = esc(", ".join(spec["right"]))
        if spec.get("rotate"):
            right += " <b>+ every candidate that is not a source in the model</b> (rotating design)"
        a = df[df["analysis"] == analysis]
        use_heatmap = len(spec["sources"]) >= 5 and spec["max_sources"] <= 3
        body = card("Findings", qp_findings(df, analysis), f"Outgroups (right): {right}")
        if use_heatmap:
            body += card("Every single source and pair", qp_heatmaps(cfg, df, analysis))
        body += card("Models that fit for at least one input",
                     qp_model_rows(cfg, df, analysis, min_sources=2 if use_heatmap else 1),
                     "Weights of each source; hover a block for weight ± standard error. Faded = rejected for that "
                     "input." + (" Single sources are on the heatmap diagonal." if use_heatmap else ""))
        if (a["n_sources"] >= 3).any():
            body += card("p-values by input", qp_pplot(df, analysis, p_min, top=12))
        secs.append(section(analysis, title, lede, body))
    secs.append(section("all", "All models", "", card(f"{len(df)} qpAdm runs", qp_table(df))))
    nav = ([("glance", "At a glance")] if gl else []) + ([("replicates", "Robustness")] if rep else []) + ([("east", "Siberian source")] if east else [])
    nav += [("about", "How to read")]
    nav += [(a, titles.get(a, (a,))[0]) for a in q["analyses"] if a in set(df["analysis"])] + [("all", "All models")]
    return page("qpAdm results", "qpAdm with the Allen Ancient DNA Resource (v66, Human Origins panel)",
                nav, "".join(secs), back=True)


# ---------------------------------------------------------------- PNG figures (matplotlib)

def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def label_color(hex_color: str) -> str:
    return INK if hex_color == SLOTS[3] else "white"


def fig_stacked(rows: list[tuple[str, list, list, list]], slots: dict[str, int] | None, title: str, path: Path):
    """Horizontal 100% bars: (row label, sources, weights, se)."""
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(rows) + 1.6), facecolor=SURFACE)
    style_ax(ax)
    ax.grid(False)
    for i, (label, src, w, se) in enumerate(rows):
        left = 0
        for j, (s, x, e) in enumerate(zip(src, w, se)):
            color = SLOTS[(slots[s] if slots else j + 1) - 1]
            ax.barh(i, x - 0.004, left=left, height=0.62, color=color, edgecolor="none")
            text = f"{x * 100:.0f}%" if slots else f"{s} {x * 100:.0f}%"
            if x > (0.07 if slots else 0.18):
                ax.text(left + 0.01, i, text, va="center", fontsize=9, color=label_color(color))
            left += x
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    if slots:
        used = sorted({s for r in rows for s in r[1]}, key=slots.get)
        handles = [plt.Rectangle((0, 0), 1, 1, color=SLOTS[slots[s] - 1]) for s in used]
        ax.legend(handles, used, ncol=min(len(used), 6), frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0),
                  fontsize=9, labelcolor=INK, handlelength=1, handleheight=1)
    fig.suptitle(title, color=INK, x=0.01, ha="left", fontsize=12, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_qpadm_models(cfg, df, analysis: str, target: str, path: Path) -> bool:
    f = df[(df["analysis"] == analysis) & (df["target"] == target) & df["fits"]]
    if f.empty:
        return False
    f = f.sort_values(["n_sources", "p"], ascending=[True, False]).head(12)
    rows = [(f"{r['sources']}  (p={fmt_p(r['p'])})", r["src"], r["weights"], r["se"]) for _, r in f.iterrows()]
    fig_stacked(rows, source_slots(cfg, analysis, df), f"qpAdm {analysis} models that fit ({target})", path)
    return True


def fig_qpadm_p(df, analysis: str, p_min: float, path: Path, top: int = 16):
    a = df[df["analysis"] == analysis]
    best = pick_models(a, top)
    targets = list(dict.fromkeys(a["target"]))
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(best) + 1.6), facecolor=SURFACE)
    style_ax(ax)
    for i, t in enumerate(targets):
        m = a[a["target"] == t].set_index("sources").reindex(best)
        filled = m["fits"].fillna(False).astype(bool).to_numpy()
        y = np.arange(len(best))
        ax.scatter(m["p"][filled], y[filled], s=46, color=SLOTS[i], edgecolors=SURFACE, linewidths=1.5, label=t, zorder=3)
        ax.scatter(m["p"][~filled], y[~filled], s=46, facecolors="none", edgecolors=SLOTS[i], linewidths=1.5, zorder=3)
    ax.axvline(p_min, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(p_min, len(best) - 0.4, f" p = {p_min}", color=INK2, fontsize=8, va="top")
    ax.set_xscale("log")
    ax.set_yticks(range(len(best)), best, color=INK, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("qpAdm p-value (right of the line = not rejected; hollow = fails)", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    fig.suptitle(f"qpAdm {analysis}: p-values, CRAM vs CSV", color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_roh(roh: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=SURFACE)
    style_ax(ax)
    ax.grid(False)
    for i, c in enumerate(AUTOSOMES):
        ax.barh(i, GRCH37_LEN[c] / 1e6, height=0.5, color=TRACK)
        for r in roh[roh["CHR"] == c].itertuples():
            ax.barh(i, max((r.POS2 - r.POS1) / 1e6, 0.6), left=r.POS1 / 1e6, height=0.5, color=SLOTS[0])
    ax.set_yticks(range(len(AUTOSOMES)), [f"chr{c}" for c in AUTOSOMES], color=INK, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Position (Mb, GRCh37)", color=MUTED, fontsize=9)
    fig.suptitle(f"Runs of homozygosity: {len(roh)} segments, {roh['KB'].sum() / 1000:.1f} Mb",
                 color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_admixture(anc: dict, path: Path):
    order = ["EUR", "SAS", "EAS", "AFR", "AMR"]
    comps = [(k, anc["admixture"][k]) for k in order if anc["admixture"].get(k, 0) > 0]
    rows = [("you", [k for k, _ in comps], [v for _, v in comps], [0] * len(comps))]
    fig_stacked(rows, {k: i + 1 for i, k in enumerate(order)}, "Supervised ADMIXTURE (1000 Genomes)", path)


def fig_depth(cov: dict, path: Path):
    h = cov["hist"]
    cap = 12
    vals = list(h.loc[h["depth"] < cap, "n"]) + [h.loc[h["depth"] >= cap, "n"].sum()]
    labels = [str(int(d)) for d in h.loc[h["depth"] < cap, "depth"]] + [f"{cap}+"]
    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=SURFACE)
    style_ax(ax)
    ax.grid(axis="x", visible=False)
    ax.bar(labels, np.array(vals) / 1e6, width=0.6, color=SLOTS[0])
    ax.set_xlabel("Read depth", color=MUTED, fontsize=9)
    ax.set_ylabel("Bases (millions)", color=MUTED, fontsize=9)
    fig.suptitle("Read depth across chromosome 1 (CRAM)", color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- page shell

CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--track:#ecebe6;--border:rgba(11,11,11,.10);--link:#256abf;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--s7:#6250d6;--s8:#e34948;
--on4:#0b0b0b;--good:#0ca30c;--good-ink:#006300;--crit:#d03b3b;--pt:#b9b8b1;
--q1:#b7d3f6;--q2:#6da7ec;--q3:#2a78d6;--q4:#184f95}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;
--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--track:#2c2c2a;--border:rgba(255,255,255,.10);
--link:#86b6ef;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767;
--on4:#ffffff;--good-ink:#0ca30c;--pt:#4a4a46;--q1:#184f95;--q2:#256abf;--q3:#5598e7;--q4:#9ec5f4}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--track:#2c2c2a;--border:rgba(255,255,255,.10);--link:#86b6ef;--s1:#3987e5;--s2:#d95926;
--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767;--on4:#ffffff;--good-ink:#0ca30c;--pt:#4a4a46;
--q1:#184f95;--q2:#256abf;--q3:#5598e7;--q4:#9ec5f4}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
a{color:var(--link)}
header{padding:28px 16px 8px;max-width:1120px;margin:0 auto}
header h1{margin:0;font-size:28px;font-weight:600;letter-spacing:-.01em}
header .sub{color:var(--ink2);margin-top:4px}
nav{position:sticky;top:0;z-index:5;background:var(--page);border-bottom:1px solid var(--border)}
nav div{max-width:1120px;margin:0 auto;padding:8px 16px;display:flex;gap:4px 18px;flex-wrap:wrap;font-size:14px}
nav a{color:var(--ink2);text-decoration:none}nav a:hover{color:var(--ink)}
main{max-width:1120px;margin:0 auto;padding:0 16px 48px}
section{margin-top:36px}
h2{font-size:20px;font-weight:600;margin:0 0 4px}
h3{font-size:15px;font-weight:600;margin:0 0 2px}
.lede{color:var(--ink2);margin:0 0 14px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:18px;margin-bottom:16px;min-width:0}
.card-sub{color:var(--ink2);font-size:13.5px;margin-bottom:14px}
.card p{margin:0 0 10px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr));gap:16px}
.grid2>.card{margin-bottom:0}
section>.grid2{margin-bottom:16px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-top:20px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px 16px}
.tile-label{color:var(--ink2);font-size:13px}
.tile-value{font-size:24px;font-weight:600;margin-top:2px}
.tile-note{color:var(--muted);font-size:12.5px}
.muted{color:var(--muted)}
.chart-note{color:var(--muted);font-size:12.5px;margin-top:8px}
.s1{background:var(--s1)}.s2{background:var(--s2)}.s3{background:var(--s3)}.s4{background:var(--s4)}
.s5{background:var(--s5)}.s6{background:var(--s6)}.s7{background:var(--s7)}.s8{background:var(--s8)}
.legend{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:13px;color:var(--ink2);margin:10px 0}
.key{display:inline-flex;align-items:center;gap:6px}
.sw{width:10px;height:10px;border-radius:3px;display:inline-block}
/* segmented bar */
.segbar{display:flex;gap:2px;height:28px;width:100%}
.seg{border-radius:4px;display:flex;align-items:center;padding-left:8px;min-width:2px;overflow:hidden;cursor:default}
.seg span{color:#fff;font-size:12.5px;font-weight:600;white-space:nowrap;font-variant-numeric:tabular-nums}
.seg.s4 span{color:var(--on4)}
.segbar.faded{opacity:.35}
.segbar-empty{height:28px;display:flex;align-items:center;color:var(--muted);font-size:13px;border:1px dashed var(--axis);
border-radius:4px;padding-left:8px}
/* horizontal bars */
.hbars{display:grid;grid-template-columns:minmax(90px,max-content) 1fr;gap:6px 12px;align-items:center;font-size:13.5px}
.hb-label{color:var(--ink2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.hb-track{display:flex;align-items:center;gap:8px;min-width:0}
.hb-bar{height:14px;border-radius:0 4px 4px 0;min-width:2px}
.hb-val{color:var(--ink2);font-size:12.5px;font-variant-numeric:tabular-nums;white-space:nowrap}
/* columns */
.cols-wrap{padding-left:44px}
.cols{position:relative;display:flex;align-items:flex-end;gap:2px;border-bottom:1px solid var(--axis)}
.col{flex:1;height:100%;display:flex;align-items:flex-end;justify-content:center;z-index:1}
.col-bar{width:min(24px,80%);border-radius:4px 4px 0 0;min-height:1px}
.col-grid{position:absolute;left:0;right:0;border-top:1px solid var(--grid)}
.col-grid span{position:absolute;right:calc(100% + 6px);top:-8px;font-size:11px;color:var(--muted);
font-variant-numeric:tabular-nums;white-space:nowrap}
.cols-x{display:flex;gap:2px;margin-top:4px}
.col-x{flex:1;text-align:center;font-size:11px;color:var(--muted);min-width:0;overflow:hidden}
/* ideogram */
.ideo{display:grid;grid-template-columns:28px 1fr;gap:5px 8px;align-items:center}
.ideo-label{color:var(--muted);font-size:12px;text-align:right;font-variant-numeric:tabular-nums}
.ideo-track{position:relative;height:12px;background:var(--track);border-radius:6px}
.roh-seg{position:absolute;top:0;bottom:0;border-radius:2px}
/* meter */
.meter-row{margin-bottom:18px}
.meter-head{display:flex;gap:8px;align-items:baseline;flex-wrap:wrap;margin-bottom:8px}
.meter-val{margin-left:auto;font-weight:600}
.meter{position:relative;height:8px;border-radius:4px;background:var(--track)}
.meter-mid{position:absolute;left:50%;top:-4px;bottom:-4px;width:1px;background:var(--axis)}
.meter-dot{position:absolute;top:50%;width:16px;height:16px;margin:-8px 0 0 -8px;border-radius:50%;background:var(--s1);
box-shadow:0 0 0 2px var(--surface)}
.meter-scale{display:flex;justify-content:space-between;font-size:11px;color:var(--muted);margin-top:4px}
/* status */
.status{padding:8px 12px;border-radius:8px;background:var(--track);font-size:14px}
.status.good b{color:var(--good-ink)}.status.crit b{color:var(--crit)}
/* qpAdm model rows */
.models{display:flex;flex-direction:column;gap:14px;margin-top:6px}
.mr{display:grid;grid-template-columns:70px 1fr minmax(140px,max-content);gap:6px 12px;align-items:center;
padding-bottom:14px;border-bottom:1px solid var(--grid)}
.mr:last-child{border-bottom:0}
.mr-name{grid-column:1/-1;font-weight:600;font-size:14px}
.mr-target{color:var(--ink2);font-size:13px}
.mr-p{font-size:13px;color:var(--ink2);font-variant-numeric:tabular-nums;white-space:nowrap}
.badge{display:inline-block;padding:0 7px;border-radius:9px;font-size:12px;font-weight:600;margin-right:4px}
.badge.fits{color:var(--good-ink);background:color-mix(in srgb,var(--good) 14%,transparent)}
.badge.rejected{color:var(--ink2);background:var(--track)}
.findings{margin:0;padding-left:20px}.findings li{margin-bottom:4px}
/* p-value plot */
.pp{display:grid;grid-template-columns:minmax(120px,max-content) 1fr;gap:4px 12px;align-items:center;font-size:13px}
.pp-label{color:var(--ink2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:280px}
.pp-track{position:relative;height:20px}
.pp-grid{position:absolute;top:0;bottom:0;width:1px;background:var(--grid)}
.pp-thr{position:absolute;top:-2px;bottom:-2px;border-left:1px dashed var(--ink2)}
.pp-dot{position:absolute;top:50%;width:11px;height:11px;margin:-5.5px 0 0 -5.5px;border-radius:50%;
box-shadow:0 0 0 2px var(--surface);z-index:1}
.pp-dot.open{background:var(--surface)!important;border:2px solid currentColor}
.pp-dot.s1.open{color:var(--s1)}.pp-dot.s2.open{color:var(--s2)}.pp-dot.s4.open{color:var(--s4)}
.pp-axis{position:relative;height:30px;font-size:11px;color:var(--muted)}
.pp-axis span{position:absolute;transform:translateX(-50%);top:2px;font-variant-numeric:tabular-nums}
.pp-axis .pp-thr-label{top:15px;color:var(--ink2)}
/* svg charts */
.svg-row{gap:12px}
.svg-fig{margin:0;min-width:0}
.svg-fig svg{width:100%;height:auto;display:block}
.svg-fig figcaption{color:var(--muted);font-size:12.5px;margin-top:4px}
.pt{fill:var(--pt);opacity:.8}.pt.hi{fill:var(--s1);opacity:.75}
.pt:hover{opacity:1;stroke:var(--ink);stroke-width:1}
.me{fill:var(--ink);stroke:var(--surface);stroke-width:2}
.me-label{fill:var(--ink);font-size:13px;font-weight:600;paint-order:stroke;stroke:var(--surface);stroke-width:4px}
.grp{fill:var(--ink2);font-size:11px;font-weight:600;paint-order:stroke;stroke:var(--surface);stroke-width:3px;cursor:default}
.axis{stroke:var(--axis);stroke-width:1}.f4-cat{font-weight:600;color:var(--ink);margin-top:8px}.f4-c{position:absolute;top:50%;width:7px;height:7px;margin:-3.5px 0 0 -3.5px;border-radius:50%;opacity:.75}.f4-you{position:absolute;top:50%;width:13px;height:13px;margin:-6.5px 0 0 -6.5px;border-radius:50%;background:var(--ink);box-shadow:0 0 0 2px var(--surface);z-index:2}.sub-h{font-size:14px;margin:18px 0 6px}tr.hl td{background:var(--track)}.gridl{stroke:var(--grid);stroke-width:1}.ax{fill:var(--muted);font-size:11px}
/* tables */
.table-wrap{overflow-x:auto;max-height:560px;overflow-y:auto;border:1px solid var(--grid);border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th{position:sticky;top:0;background:var(--surface);text-align:left;font-weight:600;color:var(--ink2);
border-bottom:1px solid var(--axis);padding:7px 10px;white-space:nowrap}
td{padding:6px 10px;border-bottom:1px solid var(--grid);vertical-align:top;font-variant-numeric:tabular-nums}
table.dense td{padding:4px 10px;font-size:13px}
tr:hover td{background:color-mix(in srgb,var(--s1) 6%,transparent)}
details{margin:4px 0 16px}summary{cursor:pointer;color:var(--link);margin-bottom:10px}
.filters{display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center;margin-bottom:12px;font-size:14px}
select{font:inherit;padding:3px 6px;border-radius:6px;border:1px solid var(--axis);background:var(--surface);color:var(--ink)}
/* qpAdm: glance rows, heatmap, forest plot */
.mr.glance{border-bottom:0;padding-bottom:0}
.hm-row-wrap{display:flex;flex-wrap:wrap;gap:24px 40px;overflow-x:auto;padding-bottom:4px}
.hm-wrap{min-width:0}
.hm-title{font-weight:600;font-size:13.5px;margin-bottom:6px}
.hm{display:grid;grid-auto-rows:26px;gap:2px;align-items:stretch}
.hm-col{height:auto;display:flex;align-items:flex-end;justify-content:center;padding-bottom:4px}
.hm-col span{writing-mode:vertical-rl;transform:rotate(180deg);font-size:11.5px;color:var(--ink2);white-space:nowrap}
.hm>.hm-col{height:96px}
.hm{grid-template-rows:96px}
.hm-row{font-size:12px;color:var(--ink2);display:flex;align-items:center;justify-content:flex-end;padding-right:8px;white-space:nowrap}
.hm-cell{border-radius:4px;cursor:default}
.hm-cell.diag{box-shadow:inset 0 0 0 1px var(--axis)}
.hm-empty{}
.q1{background:var(--q1)}.q2{background:var(--q2)}.q3{background:var(--q3)}.q4{background:var(--q4)}
.rej{background:var(--track)}
.inv{background:repeating-linear-gradient(45deg,var(--track) 0 3px,var(--grid) 3px 6px)}
.hm-cell.best,.hm-sw.best{outline:2px solid var(--ink);outline-offset:-2px}
.hm-sw{width:12px;height:12px;border-radius:3px}
.hm-cell:hover{outline:2px solid var(--ink2);outline-offset:-2px}
.fp-track{position:relative;height:26px}
.fp-ci{position:absolute;height:2px;margin-top:-1px;border-radius:1px;opacity:.55}
.fp-dot{position:absolute;width:10px;height:10px;margin:-5px 0 0 -5px;border-radius:50%;box-shadow:0 0 0 2px var(--surface);z-index:1}
.fp-mean{position:absolute;top:-2px;bottom:-2px;border-left:1px dashed var(--ink2)}
.fp .pp-label{max-width:340px}
pre.g25{white-space:pre;overflow-x:auto;font-size:12px;background:var(--track);padding:10px;border-radius:8px;margin:6px 0}
.fit-h{margin:14px 0 4px;font-size:14px}
.tile-value{overflow-wrap:anywhere}
/* theme switch and sortable tables */
.theme-btn{margin-left:auto;font:inherit;font-size:13px;color:var(--ink2);background:transparent;border:1px solid var(--axis);
border-radius:6px;padding:1px 8px;cursor:pointer}
.theme-btn:hover{color:var(--ink)}
table.sortable th{cursor:pointer;user-select:none}
table.sortable th[aria-sort]::after{content:" ▾";color:var(--muted)}
table.sortable th[aria-sort="ascending"]::after{content:" ▴"}
/* tooltip */
#tip{position:fixed;z-index:10;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--border);
border-radius:8px;padding:7px 10px;font-size:12.5px;box-shadow:0 4px 16px rgba(0,0,0,.14);max-width:280px;
font-variant-numeric:tabular-nums}
#tip .t1{font-weight:600}#tip div:not(.t1){color:var(--ink2)}
[data-tip]:focus-visible{outline:2px solid var(--ink);outline-offset:1px}
footer{max-width:1120px;margin:0 auto;padding:0 16px 40px;color:var(--muted);font-size:12.5px}
@media (max-width:560px){.mr{grid-template-columns:60px 1fr}.mr-p{grid-column:2}header h1{font-size:23px}}
"""

JS = """
const tip=document.getElementById('tip');
function place(x,y){const r=tip.getBoundingClientRect();let L=x+14,T=y+14;
 if(L+r.width>innerWidth-8)L=x-r.width-14;if(T+r.height>innerHeight-8)T=y-r.height-14;
 tip.style.left=Math.max(8,L)+'px';tip.style.top=Math.max(8,T)+'px'}
function show(el,x,y){tip.replaceChildren(...el.dataset.tip.split('\\n').map((l,i)=>{const d=document.createElement('div');
 d.textContent=l;if(!i)d.className='t1';return d}));tip.hidden=false;place(x,y)}
document.addEventListener('pointerover',e=>{const el=e.target.closest('[data-tip]');el?show(el,e.clientX,e.clientY):tip.hidden=true});
document.addEventListener('pointermove',e=>{if(!tip.hidden)place(e.clientX,e.clientY)});
document.addEventListener('focusin',e=>{const el=e.target.closest('[data-tip]');if(el){const r=el.getBoundingClientRect();
 show(el,r.right,r.top)}});
document.addEventListener('focusout',()=>tip.hidden=true);
// Hide in-bar labels that don't fit their block (never clip them)
function fitLabels(){document.querySelectorAll('.seg span').forEach(s=>{s.style.visibility='';
 if(s.offsetWidth+12>s.parentElement.clientWidth)s.style.visibility='hidden'})}
addEventListener('resize',fitLabels);fitLabels();
// Table filters (qpAdm page)
const filters=[...document.querySelectorAll('[data-filter]')];
function applyFilters(){const rows=document.querySelectorAll('.models-table tbody tr');let n=0;
 rows.forEach(r=>{const ok=filters.every(f=>f.type==='checkbox'?(!f.checked||r.dataset.fits==='1')
  :(!f.value||r.dataset[f.dataset.filter]===f.value));r.hidden=!ok;n+=ok});
 const c=document.getElementById('row-count');if(c)c.textContent=n+' of '+rows.length+' models'}
filters.forEach(f=>f.addEventListener('change',applyFilters));if(filters.length)applyFilters();
// Sortable tables: click a header; numbers (incl. 1.2e-05) sort numerically
document.querySelectorAll('table.sortable').forEach(t=>t.querySelectorAll('th').forEach((th,i)=>th.addEventListener('click',()=>{
 const asc=th.getAttribute('aria-sort')!=='ascending';t.querySelectorAll('th').forEach(h=>h.removeAttribute('aria-sort'));
 th.setAttribute('aria-sort',asc?'ascending':'descending');const body=t.tBodies[0];
 const val=r=>{const s=r.cells[i].textContent.trim(),n=parseFloat(s);return isNaN(n)||!/^[-+\\d.]/.test(s)?s:n};
 [...body.rows].sort((a,b)=>{const x=val(a),y=val(b);const c=typeof x==='number'&&typeof y==='number'?x-y:String(x).localeCompare(String(y));
  return asc?c:-c}).forEach(r=>body.appendChild(r))})));
// Theme: auto (system) -> light -> dark
const themeBtn=document.querySelector('.theme-btn');
function setTheme(m){if(m==='auto')delete document.documentElement.dataset.theme;else document.documentElement.dataset.theme=m;
 if(themeBtn)themeBtn.textContent='Theme: '+m;try{localStorage.setItem('theme',m)}catch(e){}}
if(themeBtn){let m='auto';try{m=localStorage.getItem('theme')||'auto'}catch(e){}setTheme(m);
 themeBtn.addEventListener('click',()=>{const order=['auto','light','dark'];
  setTheme(order[(order.indexOf(themeBtn.textContent.replace('Theme: ',''))+1)%3])})}
"""


def page(title: str, subtitle: str, nav: list[tuple[str, str]], body: str, back: bool = False, head: str = "") -> str:
    links = "".join(f'<a href="#{i}">{esc(t)}</a>' for i, t in nav)
    links = ('<a href="index.html">← Overview</a>' if back else "") + links
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title><style>{CSS}</style>{head}
<script>try{{const m=localStorage.getItem('theme');if(m&&m!=='auto')document.documentElement.dataset.theme=m}}catch(e){{}}</script></head>
<body><header><h1>{esc(title)}</h1><div class="sub">{esc(subtitle)}</div></header>
<nav><div>{links}<button class="theme-btn" type="button">Theme: auto</button></div></nav><main>{body}</main>
<footer>Generated {date.today().isoformat()} by scripts/14_report.py. Not a medical test.</footer>
<div id="tip" hidden></div><script>{JS}</script></body></html>"""


# ---------------------------------------------------------------- main

FIREFOX = Path("/mnt/c/Program Files/Mozilla Firefox/firefox.exe")


def main() -> None:
    plt.switch_backend("Agg")
    parser = config_arg(__doc__)
    parser.add_argument("--no-open", action="store_true", help="don't open the report in Firefox afterwards")
    parser.add_argument("--qpadm-results", help="file in out_dir/qpadm/ (default: qpadm_fast_results.csv if it "
                                                "exists, else qpadm_results.csv)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = cfg["out_dir"]
    rep = out / "report"
    figs = rep / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    qc = read_json(out / "qc.json")
    anc = read_json(out / "ancestry.json")
    pcs = load_pca(cfg)
    roh = load_roh(out)
    cov = load_coverage(out)
    qp_path = qpadm_results_path(out, args.qpadm_results)
    qp = load_qpadm(qp_path)
    f3 = load_f3(out)

    # Overview page
    parts = [("summary", "Summary", sec_summary(cfg, qc, anc, roh, cov)),
             ("ancestry", "Ancestry", sec_ancestry(cfg, anc, pcs)),
             ("deep", "Deep ancestry", sec_deep_teaser(cfg, qp)),
             ("farmer", "Farmer ancestry", sec_farmer(cfg, out)),
             ("f4", "Uralic signal", sec_f4(out)),
             ("neanderthal", "Neanderthal", sec_neanderthal(out)),
             ("shared", "Shared variants", sec_shared(out)),
             ("f3", "Closest ancients", sec_f3_teaser(f3)),
             ("g25", "G25-style", sec_g25(out, rep)),
             ("ibs", "Ancient matches", sec_ibs(out)),
             ("ibs_all", "Ancestrify-style scan", sec_ibs_all(out)),
             ("roh", "ROH", sec_roh(roh)),
             ("traits", "Traits & PRS", sec_traits_prs(cfg, out)),
             ("clinvar", "ClinVar", sec_clinvar(out)),
             ("pgx", "Drugs (PGx)", sec_pharmcat(out, rep)),
             ("sequencing", "Data quality", sec_cram(cov, qc))]
    nav = [(i, t) for i, t, html_ in parts if html_ and i != "summary"]
    sub = f"Sample “{cfg['sample_id']}”"
    if qc:
        sub += f" · {qc['header'].get('method', '')} · {qc['header'].get('reference', '')}"
    index = page("My DNA report", sub, nav, "".join(h for _, _, h in parts))
    if qp is not None:
        index = index.replace('<button class="theme-btn"', '<a href="qpadm.html"><b>qpAdm details →</b></a><button class="theme-btn"', 1)
    if f3 is not None:
        index = index.replace('<button class="theme-btn"', '<a href="map.html"><b>Ancient map →</b></a><button class="theme-btn"', 1)
    (rep / "index.html").write_text(index)
    written = [rep / "index.html"]

    if qp is not None:
        (rep / "qpadm.html").write_text(qpadm_page(cfg, qp, qp_path))
        written.append(rep / "qpadm.html")
    if f3 is not None:
        (rep / "ancient.html").write_text(f3_page(f3).replace(
            '<button class="theme-btn"', '<a href="map.html"><b>Map →</b></a><button class="theme-btn"', 1))
        (rep / "map.html").write_text(map_page(f3))
        written += [rep / "ancient.html", rep / "map.html"]

    # PNG figures
    if qp is not None:
        for analysis in dict.fromkeys(qp["analysis"]):
            for target in dict.fromkeys(qp["target"]):
                path = figs / f"qpadm_{analysis}_{target}.png"
                if fig_qpadm_models(cfg, qp, analysis, target, path):
                    written.append(path)
            path = figs / f"qpadm_{analysis}_pvalues.png"
            fig_qpadm_p(qp, analysis, cfg["qpadm"]["p_min"], path)
            written.append(path)
    if anc:
        fig_admixture(anc, figs / "admixture.png")
        written.append(figs / "admixture.png")
    if roh is not None:
        fig_roh(roh, figs / "roh.png")
        written.append(figs / "roh.png")
    if cov and "hist" in cov:
        fig_depth(cov, figs / "depth_chr1.png")
        written.append(figs / "depth_chr1.png")

    print("Written:")
    for p in written:
        print(f"  {p}")
    index_win = subprocess.run(["wslpath", "-w", str(rep / "index.html")], capture_output=True, text=True).stdout.strip()
    if FIREFOX.exists() and not args.no_open and index_win:
        subprocess.Popen([str(FIREFOX), index_win], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
        print(f"\nOpened in Firefox: {index_win}")
    else:
        print(f"\nOpen in Firefox: \"{FIREFOX}\" \"$(wslpath -w {rep / 'index.html'})\"")


if __name__ == "__main__":
    main()
