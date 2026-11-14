"""Ancient module, qpAdm: the deep-ancestry overview and the full qpAdm page."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from report.core import INK, INK2, MUTED, SLOTS, SURFACE, card, details, esc, fig_stacked, fmt_p, legend, nice_ticks, page, pct, section, seg_bar, style_ax, table, tile

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
    one = df["n_sources"] == 1  # older ancient/qpadm.py runs stored ADMIXTOOLS' 0.000 for single sources
    df.loc[one, "weights"] = pd.Series([[1.0]] * one.sum(), index=df.index[one], dtype=object)
    df.loc[one, "se"] = pd.Series([[0.0]] * one.sum(), index=df.index[one], dtype=object)
    df["src"] = df["sources"].str.split(" + ", regex=False)
    df["valid"] = df["weights"].map(lambda w: bool(w) and all(0 <= x <= 1 for x in w))
    return df


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
    """Overview card: robustness of qpAdm to the random read (ancient/qpadm_replicates.py)."""
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
    """Section body: how qpAdm results vary over repeated pseudo-haploid draws (ancient/qpadm_replicates.py)."""
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
            "<p class='muted'>Computed with scripts/ancient/qpadm_fast.py (NumPy). On me_cram it reproduces ADMIXTOOLS "
            "qpAdm closely (see outputs/qpadm/fast_vs_admixtools.csv). Each f4-statistic uses the SNPs shared by "
            "its own four groups, so me_csv's statistics all stay on its ~58k SNPs; ADMIXTOOLS' qpfstats mixes "
            "them with statistics on the full SNP set, which rejected almost every me_csv model.</p>")
    else:
        intro += "<p class='muted'>Computed with ADMIXTOOLS qpAdm (scripts/ancient/qpadm.py).</p>"
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
                nav, "".join(secs), active="ancient",
                extra=[("ancient_individuals.html", "Ancient individuals"), ("map.html", "Map")])


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
