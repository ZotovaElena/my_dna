"""Ethnicity module: present-day ancestry (ADMIXTURE, PCA, G25-style) and runs of homozygosity."""

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import AUTOSOMES
from ethnicity import plot_ancestry as pca_module
from report.core import GRCH37_LEN, INK, MUTED, SLOTS, SUPERPOP_NAMES, SURFACE, TRACK, card, details, esc, fig_stacked, hbars, legend, pct, section, seg_bar, style_ax, table, tile
from report.sections.f3 import age_txt

def load_roh(out: Path) -> pd.DataFrame | None:
    path = out / "roh.hom"
    if not path.exists():
        return None
    return pd.read_csv(path, sep=r"\s+").astype({"CHR": str})


def load_pca(cfg: dict):
    out = cfg["out_dir"]
    if not (out / "ancestry.eigenvec").exists():
        return None
    try:
        pcs, _ = pca_module.load_pcs(cfg)
    except FileNotFoundError:
        return None
    return pcs


def pca_svg(pcs: pd.DataFrame, me: pd.Series, x: str, y: str, subset=None, label_by: str = "super_pop",
            highlight: str | None = None, title: str = "", colors: dict[str, int] | None = None) -> str:
    """Scatter of reference samples (gray, or one group in slot 1, or coloured by label_by group via colors:
    group -> slot) with group labels at centroids and you."""
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
        if colors and r[label_by] in colors:
            cls = f"pt c{colors[r[label_by]]}"
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
        sp_slot = {k: i + 1 for i, k in enumerate(["EUR", "SAS", "EAS", "AFR", "AMR"])}
        eur = sorted(pcs.loc[pcs["super_pop"] == "EUR", "pop"].dropna().unique())
        a = pca_svg(pcs, me, "PC1", "PC2", label_by="super_pop", colors=sp_slot,
                    title="All 2,504 samples, PC1 vs PC2, coloured by superpopulation (as in the ADMIXTURE bar)")
        b = pca_svg(pcs, me, "PC1", "PC2", subset=lambda d: d["super_pop"] == "EUR", label_by="pop",
                    colors={g: i + 1 for i, g in enumerate(eur)} if len(eur) <= 8 else None,
                    title="Europeans only, PC1 vs PC2, coloured by population (hover a label for its name)")
        body += card("Principal components", f'<div class="grid2 svg-row">{a}{b}</div>',
                     "Each dot is one 1000 Genomes person; hover for details. Labels sit at each group's centre.")
    return section("ancestry", "Ancestry", "Where your genome sits among modern reference populations.", body)


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
             tile("Closest ancient population", str(ap.iloc[0]["name"]).replace("_", " "), f"distance {ap.iloc[0]['distance']:.4f}, "
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
