"""Matches module: IBS segment scans with ancient individuals and controls."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from report.core import card, esc, fmt_num, hbars, nice_ticks, read_json, section, table, tile
from report.sections.f3 import CULTURE_RX, PERIOD_NAMES, age_txt, classify_period

IBS_GROUPS = [("you", "You", 0), ("EUR", "Europeans", 1), ("EAS", "East Asians", 2), ("AFR", "Africans", 3)]


IBS_SUPER = {"FIN": "EUR", "GBR": "EUR", "TSI": "EUR", "IBS": "EUR", "CEU": "EUR", "Russian": "EUR",
             "CHB": "EAS", "JPT": "EAS", "Han": "EAS", "YRI": "AFR", "LWK": "AFR", "Mbuti": "AFR", "you": "you",
             "PJL": "SAS", "PEL": "AMR"}


# AADR .anno columns (v66, by position: the headers are long sentences)
ANNO_COLS = {"id": 0, "pub": 6, "doi": 7, "repo": 8, "date_method": 9, "date_full": 12, "anthro": 13, "group": 14,
             "locality": 15, "country": 16, "lat": 17, "lon": 18, "data_type": 21, "coverage": 23, "ho_snps": 27,
             "sex": 30, "family": 31, "y_isogg": 35, "y_manual": 36, "mt_hg": 38, "assessment": 47}
SEX = {"M": "male", "F": "female", "U": "unknown"}


def ancient_annotation(cfg: dict | None, ids) -> pd.DataFrame:
    """AADR annotation for the given Genetic IDs (empty frame if the AADR isn't available)."""
    if cfg is None:
        return pd.DataFrame()
    from ancient.aadr_prepare import aadr_files
    path = aadr_files(cfg)["anno"]
    if not path.exists():
        return pd.DataFrame()
    a = pd.read_csv(path, sep="\t", low_memory=False, dtype=str)
    a = a.iloc[:, list(ANNO_COLS.values())]
    a.columns = list(ANNO_COLS)
    a = a[a["id"].isin(set(ids))].set_index("id")
    return a.replace({"..": "", "n/a": "", "nan": ""}).fillna("")


def culture_of(group: str) -> str:
    hits = [n for n, rx in CULTURE_RX if not n.endswith("(all)") and rx.search(group)]
    return hits[0] if hits else ""


def _row(label: str, value: str) -> str:
    return f'<div class="fact"><span>{esc(label)}</span><b>{value}</b></div>' if value else ""


def ancient_profiles(top: pd.DataFrame, seg: pd.DataFrame, anno: pd.DataFrame, n_ctrl: int, n_rank: int,
                     hs=None) -> str:
    """One card per matched ancient individual: who, when, where, haplogroups, data, publication, your segments."""
    hot, hs_c = (hs[3], hs[0]) if hs else (None, "")
    cards = []
    for r in top.itertuples():
        a = anno.loc[r.id] if r.id in anno.index else pd.Series(dtype=str)
        g = lambda k: str(a.get(k, "") or "").strip()
        period = classify_period(r.group, r.date_bp)
        cult = culture_of(str(r.group))
        full = g("date_full").split("[")[0].strip()
        how = g("date_method").lower()
        how = ("radiocarbon-dated" if how.startswith("direct") else
               "dated by archaeological context" if how.startswith("context") else "")
        when = f"{esc(full)} · {esc(age_txt(r.date_bp))} ago" if full else esc(age_txt(r.date_bp)) + " ago"
        if how:
            when += f'<br><span class="muted">{how}</span>'
        place = ", ".join(x for x in [g("locality"), g("country") or str(r.country)] if x and x != "nan")
        if pd.notna(r.lat) and pd.notna(r.lon):
            place = (f'{esc(place)} <a href="https://www.openstreetmap.org/?mlat={r.lat:.4f}&mlon={r.lon:.4f}#map=8/'
                     f'{r.lat:.4f}/{r.lon:.4f}" target="_blank" rel="noopener">map</a>')
        else:
            place = esc(place)
        sex = SEX.get(g("sex"), g("sex"))
        y = g("y_manual") or g("y_isogg")
        y = "" if y.lower().startswith("n/a") or "female" in y.lower() else y
        hap = " · ".join(x for x in [f"Y {esc(y)}" if y else "", f"mtDNA {esc(g('mt_hg'))}" if g("mt_hg") else ""] if x)
        cov = g("coverage")
        try:
            cov = f"{float(cov):.2f}× coverage" if cov else ""
        except ValueError:
            pass
        dt = g("data_type")
        dt = ("high-coverage shotgun genome" if "diploid" in dt.lower() else "shotgun sequencing" if
              dt.lower().startswith("shotgun") else "1240k capture" if dt.startswith("1240k") else dt[:40])
        data = " · ".join(x for x in [esc(dt), esc(cov),
                                      f"{int(r.ancient_snps):,} SNPs in this scan"] if x)
        doi = g("doi")
        pub = esc(g("pub"))
        if doi.startswith("http"):
            pub = f'<a href="{esc(doi)}" target="_blank" rel="noopener">{pub or "paper"}</a>'
        segs = seg[(seg["ancient"] == r.id) & (seg["pop"] == "you")].sort_values("length_cm", ascending=False)
        def crowd(s):  # segments of anyone, with any ancient, overlapping this one: a hotspot means artifact
            o = seg[(seg["chrom"] == s.chrom) & (seg["start_cm"] < s.end_cm) & (seg["end_cm"] > s.start_cm)]
            if hot is not None and hot[s.Index]:
                return (f'<br><span class="muted">in the chr{hs_c} false-match hotspot: {len(o) - 1:,} other '
                        f'segments of all people overlap it</span>')
            return f'<br><span class="muted">{len(o) - 1:,} other segments in the scan overlap it</span>'

        seg_txt = "<br>".join(f"chr{int(s.chrom)}: {s.start_cm:.1f}–{s.end_cm:.1f} cM ({s.length_cm:.1f} cM, "
                              f"{int(s.snps)} SNPs){crowd(s)}" for s in segs.itertuples())
        fam = g("family")
        if fam:
            deg = fam.split(":")[0]
            rel = [x for x in fam.split(":")[1:] if x and x != str(r.iid)]
            fam = (f"{esc(', '.join(rel))} ({esc(deg)}-degree relative)" if rel and deg.endswith("d")
                   else esc(fam))
        ctrl = int(r.controls_with_segment)
        ctrl_txt = (f"{ctrl} of {n_ctrl}: shared by chance as well" if ctrl else
                    f"none of {n_ctrl}: only you")
        sub = " · ".join(x for x in [esc(cult), esc(period), f"AADR group {esc(r.group)}"] if x)
        cards.append(
            f'<div class="card ap"><h3>{esc(r.iid)}</h3><div class="card-sub">{sub}</div>'
            + _row("When", when) + _row("Where", place) + _row("Sex", esc(sex))
            + _row("Haplogroups", hap) + _row("Relatives in the AADR", fam)
            + _row("Notes", esc(g("anthro")))
            + _row("Data", data) + _row("Published", pub)
            + _row("Your shared segment" + ("s" if len(segs) > 1 else ""), seg_txt)
            + _row("Controls with a segment", esc(ctrl_txt))
            + _row("Outgroup-f3 rank", f"{int(r.rank):,} of {n_rank:,}" if pd.notna(r.rank) else "")
            + "</div>")
    return f'<div class="aps">{"".join(cards)}</div>'


def find_hotspot(seg: pd.DataFrame, bin_cm: float = 5, min_share: float = 0.25):
    """The stretch where segments of everyone pile up (spurious IBS), or None: chrom, start/end cM, mask."""
    if len(seg) < 10:
        return None
    mid = (seg["start_cm"] + seg["end_cm"]) / 2
    key = seg["chrom"].astype(str) + ":" + (mid // bin_cm).astype(int).astype(str)
    top = key.value_counts()
    if top.iat[0] / len(seg) < min_share:
        return None
    c, b = top.index[0].split(":")
    near = (seg["chrom"].astype(str) == c) & ((mid // bin_cm) - int(b)).abs().le(1)
    lo, hi = seg.loc[near, "start_cm"].min(), seg.loc[near, "end_cm"].max()
    mask = (seg["chrom"].astype(str) == c) & (seg["start_cm"] < hi) & (seg["end_cm"] > lo)
    return c, lo, hi, mask


def cm_to_mb(cfg: dict | None, chrom: str, cms: list[float]) -> list[float] | None:
    """Physical positions (Mb) of genetic positions, from the GLIMPSE genetic map, if it exists."""
    if cfg is None:
        return None
    path = cfg["ref_dir"] / "glimpse" / "maps" / f"chr{chrom}.gmap"
    if not path.exists():
        return None
    m = pd.read_csv(path, sep="\t")
    return [float(np.interp(v, m["cM"], m["pos"])) / 1e6 for v in cms]


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


def sec_ibs_all(out: Path, cfg: dict | None = None) -> str:
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
    hs = find_hotspot(seg)
    hot_card = ""
    if hs:
        hc, hlo, hhi, hmask = hs
        mb = cm_to_mb(cfg, hc, [hlo, hhi])
        where = f"chromosome {hc}, {hlo:.0f}–{hhi:.0f} cM" + (f" (~{mb[0]:.0f}–{mb[1]:.0f} Mb)" if mb else "")
        out_seg = seg[~hmask]
        you_out = int((out_seg["pop"] == "you").sum())
        you_in = int(((seg["pop"] == "you") & hmask).sum())
        tiles.append(tile("Outside the one hotspot", f"{you_out}",
                          f"your segments; {int((out_seg['pop'] != 'you').sum())} for all {n_ctrl} controls together"))
        hp = seg[hmask].groupby("pop").size()
        eas_ids = set(people.loc[people["super"] == "EAS", "person"])
        n_eas, eas_in = len(eas_ids), seg.loc[hmask & seg["person_id"].isin(eas_ids), "person_id"].nunique()
        sup_of = people.drop_duplicates("pop").set_index("pop")["super"]
        hot_rows = [("▶ You" if pop == "you" else pop, float(n), f"{n}",
                     f"{pop} ({sup_of.get(pop, '')}): {n} segments in the hotspot")
                    for pop, n in hp.sort_values(ascending=False).items()]
        centro = " It spans the centromere of chromosome 1 and the 1q21 region" if hc == "1" and mb and mb[0] < 121 and mb[1] > 145 else ""
        hot_card = card(f"Why almost every segment is in one place: {where}", hbars(hot_rows) + (
            f'<p style="margin-top:14px">{int(hmask.sum())} of the {len(seg)} passing segments in the whole scan, '
            f'from all {n_ctrl + 1} people, '

            f"fall in this one stretch.{centro}: there are few informative SNPs there and long haplotypes common to "
            "everyone, so a run of SNPs can look half-identical with almost anybody. Studies of IBD flag such "
            "regions as false-positive hotspots and mask them. "
            f"{eas_in} of the {n_eas} East Asian controls have segments there too. {you_in} of your {int(you['segments'])} segments are in it; outside it you have "
            f"{you_out}.</p>"),
            "Segments in the hotspot, per population (summed over people and ancients).")
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
    body += hot_card
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
            "Ancient": top["iid"], "Group": top["group"], "Age": [age_txt(v).replace(" ", "\u00a0") for v in top["date_bp"]],
            "Country": top["country"], "Segments": top["you_n"], "cM": [f"{v:.1f}" for v in top["you_cm"]],
            "Longest": [f"chr{int(c)} {a:.0f}–{b:.0f} cM, {int(n)} SNPs" for c, a, b, n in
                        zip(top["chrom"], top["start_cm"], top["end_cm"], top["snps"])],
            "f3 rank": [f"{int(v):,}" for v in top["rank"]],
            "Controls matching": [f"{int(v)} of {n_ctrl}" for v in top["controls_with_segment"]]})
        body += card("Your matches", table(tt, "dense sortable"),
                     f"Ancient individuals with at least one segment {thr}, longest first. “Controls matching”: how "
                     "many of the present-day controls also have a passing segment with the same ancient genome; "
                     "“f3 rank”: the individual's place in your outgroup-f3 ranking (1 = most shared drift).")
        prof = top.drop_duplicates("id").head(24)
        body += card("Who they were", ancient_profiles(prof, seg, ancient_annotation(cfg, prof["id"]), n_ctrl, n_anc, hs),
                     "Each matched individual as described in the Allen Ancient DNA Resource (AADR v66): dating, "
                     "site, genetic sex, haplogroups, data quality and the paper that published the genome"
                     + (f"; the {len(prof)} with the most shared cM" if len(matched) > len(prof) else "")
                     + ". “map” opens the site on OpenStreetMap.")
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
               f"{'are' if len(chroms) > 1 else 'is'} imputed; rerun matches/ibs_all.py when all 22 are done" if partial else "") + ".")
    return section("ibs_all", "Ancestrify-style scan of every ancient individual", esc(lede), body)
