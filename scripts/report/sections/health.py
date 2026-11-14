"""Health module: ClinVar variants and polygenic scores."""

from pathlib import Path

import pandas as pd

from report.core import card, details, esc, hbars, pct, section, table

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


def sec_prs(out: Path) -> str:
    prs_path = out / "prs" / "prs_summary.csv"
    if not prs_path.exists():
        return ""
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
    body = card(f"Polygenic scores vs 1000 Genomes {ref}", meters,
                "Where your score falls among reference people. 50 = average; not a risk estimate.")
    return section("prs", "Polygenic scores", "Many small effects added up, compared with reference people.", body)
