"""Ancient module: f4 tests, farmer ancestry, Neanderthal ancestry, shared variants."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from report.core import card, esc, section, table, tile

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
