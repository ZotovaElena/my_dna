"""Overview page: one card per module with its headline results, linking to the module's page."""

from pathlib import Path

import pandas as pd

from report.core import MODULE_ACCENT, esc, icon, pct, read_json
from report.sections.drugs import load_pharmcat, pgx_actionable
from report.sections.f3 import f3_groups
from report.sections.qpadm import deep_best
from report.sections.traits import GENOME_WIDE

ABOUT = {
    "genome": "Your raw data: SNP file, sequencing reads, depth and quality.",
    "health": "Disease-related variants (ClinVar) and polygenic scores.",
    "drugs": "How your genes may change drug response (PharmCAT, CPIC/DPWG guidelines).",
    "traits": "Well-known single-SNP traits and what large studies link to your alleles.",
    "ethnicity": "Where you sit among present-day populations, and runs of homozygosity.",
    "ancient": "Your ancestry in terms of ancient populations: qpAdm, f3/f4 statistics, Neanderthal.",
    "matches": "Shared DNA segments with individual ancient people, checked against controls.",
}


def facts(out: Path, cfg: dict, qc, anc, roh, cov, qp, f3) -> dict[str, list[tuple[str, str]]]:
    """Two or three (label, value) headlines per module, from whatever results exist."""
    f: dict[str, list[tuple[str, str]]] = {m: [] for m in ABOUT}
    if qc:
        f["genome"] += [("SNPs in the raw file", f"{qc['n_variants']:,}"), ("Inferred sex", qc["inferred_sex"])]
    if cov and "per_chrom" in cov:
        pc = cov["per_chrom"]
        pc = pc.loc[[c for c in pc.index if c not in ("X", "Y")]]
        f["genome"].append(("Sequencing depth", f"{(pc['meandepth'] * pc['endpos']).sum() / pc['endpos'].sum():.2f}×"))

    cv = out / "clinvar_carriers.csv"
    if cv.exists():
        sig = pd.read_csv(cv)["CLNSIG"].fillna("")
        n_path = int((sig.str.contains("athogenic") & ~sig.str.contains("Conflicting")).sum())
        f["health"] += [("ClinVar sites you carry", f"{len(sig):,}"),
                        ("Pathogenic / likely pathogenic", f"{n_path}")]
    prs = out / "prs" / "prs_summary.csv"
    if prs.exists():
        p = pd.read_csv(prs)
        pcol = next(c for c in p.columns if c.startswith("percentile"))
        f["health"].append(("Polygenic scores", ", ".join(f"{t} {v:.0f}th pct" for t, v in zip(p["trait"], p[pcol]))))

    pc = load_pharmcat(out)
    if pc:
        f["drugs"] += [("Genes called", f"{sum(g['called'] for g in pc['genes'])} of {len(pc['genes'])}"),
                       ("Recommendations that change something",
                        f"{sum(pgx_actionable(r) for r in pc['recs'])} of {len(pc['recs'])}")]

    if (out / "traits.csv").exists():
        f["traits"].append(("Trait SNPs", f"{len(pd.read_csv(out / 'traits.csv'))}"))
    gm = out / "gwas_trait_map.csv"
    if gm.exists():
        g = pd.read_csv(gm, low_memory=False)
        f["traits"].append(("GWAS associations", f"{len(g):,} ({int((g['neg_log10_p'] >= GENOME_WIDE).sum()):,} "
                                                 "genome-wide)"))
        if (out / "gwas_carried_traits.csv").exists():
            c = pd.read_csv(out / "gwas_carried_traits.csv", low_memory=False)
            f["traits"].append(("Traits linked to alleles you carry", f"{c['trait_label'].nunique()}"))

    if anc:
        top = max(anc["admixture"], key=anc["admixture"].get)
        f["ethnicity"].append(("Main ADMIXTURE component", f"{top} {pct(anc['admixture'][top], 1)}"))
    gd = out / "g25" / "distances.csv"
    if gd.exists():
        d = pd.read_csv(gd)
        d = d[(d["target"] == "you_CRAM") & (d["kind"] == "modern population")].sort_values("distance")
        if len(d):
            f["ethnicity"].append(("Closest present-day populations",
                                   ", ".join(d["name"].head(3).str.replace("_", " "))))
    if roh is not None:
        f["ethnicity"].append(("Runs of homozygosity", f"{roh['KB'].sum() / 1000:.0f} Mb in {len(roh)} segments"))

    if qp is not None:
        best = deep_best(qp)
        if best is not None:
            f["ancient"].append(("Simplest deep model that fits",
                                 " + ".join(f"{s} {w * 100:.0f}%" for s, w in zip(best["src"], best["weights"]))))
    if f3 is not None:
        g = f3_groups(f3)
        f["ancient"].append(("Closest ancient populations (f3)", ", ".join(g["iid"].head(2).str.replace("_", " "))))
    nea = out / "neanderthal" / "neanderthal.csv"
    if nea.exists():
        n = pd.read_csv(nea)
        you = n[n["test"] == "you"]
        if len(you):
            f["ancient"].append(("Neanderthal ancestry", f"{you['neanderthal'].iloc[0] * 100:.1f}%"))

    pp = out / "ibs_all" / "per_person.csv"
    st = read_json(out / "ibs_all" / "settings.json")
    if pp.exists() and st and len(st.get("chromosomes", [])) == 22 and not st.get("limit"):
        p = pd.read_csv(pp)
        you, ctrl = p[p["super"] == "you"].iloc[0], p[p["super"] != "you"]
        f["matches"] += [("Ancient individuals scanned", f"{st['ancients']:,}"),
                         ("Your segments (Ancestrify-style)", f"{int(you['segments'])}; controls' median "
                                                             f"{ctrl['segments'].median():.0f}")]
        from report.sections.matches import find_hotspot
        seg = pd.read_csv(out / "ibs_all" / "segments.csv")
        hs = find_hotspot(seg)
        if hs:
            f["matches"].append(("Outside the chr" + hs[0] + " false-match hotspot",
                                 f"{int(((seg['pop'] == 'you') & ~hs[3]).sum())} for you"))
        return f
    ibs = out / "ibs" / "summary.csv"
    if ibs.exists():
        s = pd.read_csv(ibs)
        you = s[s["modern_pop"] == "you"]
        ctrl = s[s["modern_pop"] != "you"]
        f["matches"] += [("Ancient genomes scanned", f"{you['ancient'].nunique()}"),
                         ("Segments passing both thresholds", f"{int(you['n_segments'].sum())} for you, "
                                                              f"{int(ctrl['n_segments'].sum())} for "
                                                              f"{ctrl['modern'].nunique()} controls")]
    return f


def module_cards(fa: dict[str, list[tuple[str, str]]], titles: dict[str, str]) -> str:
    cards = []
    for m, about in ABOUT.items():
        if m not in titles:
            continue
        rows = "".join(f'<div class="fact"><span>{esc(k)}</span><b>{esc(v)}</b></div>' for k, v in fa[m]) or \
            '<div class="muted">No results yet.</div>'
        acc_l, acc_d = MODULE_ACCENT[m]
        cards.append(f'<a class="card mod" href="{m}.html" style="--acc-l:{acc_l};--acc-d:{acc_d}">'
                     f'<h3><span class="mod-ico">{icon(m)}</span>{esc(titles[m])} <span class="go">→</span></h3>'
                     f'<div class="card-sub">{esc(about)}</div>{rows}</a>')
    return f'<div class="mods">{"".join(cards)}</div>'


def overview(out: Path, cfg: dict, qc, anc, roh, cov, qp, f3, titles: dict[str, str]) -> str:
    lede = ("Personal genome analysis from a MyHeritage SNP file and ~1.9× sequencing reads. Each module has its own "
            "page; the numbers below are the headlines. Population-level results, not a medical test.")
    return (f'<section id="modules"><p class="lede">{esc(lede)}</p>'
            + module_cards(facts(out, cfg, qc, anc, roh, cov, qp, f3), titles) + "</section>")
