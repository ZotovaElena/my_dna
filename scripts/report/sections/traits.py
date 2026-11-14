"""Traits module: single-SNP traits (traits/snp_traits.py) and GWAS Catalog associations (traits/gwas_*.py)."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from report.core import card, details, esc, hbars, section, table, tile

GENOME_WIDE = -np.log10(5e-8)
SIZE_ORDER = ["large", "moderate", "small", "very small", "unknown"]
SIZE_NOTE = {"large": "odds ratio ≥ 3 or ≥ 0.5 SD per copy", "moderate": "odds ratio 1.5–3 or 0.2–0.5 SD",
             "small": "odds ratio 1.1–1.5 or 0.05–0.2 SD", "very small": "odds ratio < 1.1 or < 0.05 SD",
             "unknown": "the study's units can't be compared"}


def load_gwas(out: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """All annotated associations (gwas_trait_map.csv) and the carried genome-wide subset."""
    path = out / "gwas_trait_map.csv"
    if not path.exists():
        return None, None
    gw = pd.read_csv(path, low_memory=False)
    carried_path = out / "gwas_carried_traits.csv"
    carried = pd.read_csv(carried_path, low_memory=False) if carried_path.exists() else gw.iloc[0:0]
    return gw, carried


def main_gene(genes: pd.Series) -> str:
    vals = genes.dropna().astype(str).str.split(r"[;,]\s*").explode().str.strip()
    vals = vals[vals != ""]
    return vals.mode().iat[0] if len(vals) else ""


def sec_snp_traits(out: Path, gw: pd.DataFrame | None) -> str:
    tr = out / "traits.csv"
    if not tr.exists():
        return ""
    t = pd.read_csv(tr)
    rows = []
    for r in t.itertuples():
        row = {"SNP": r.rsid, "Trait": r.trait, "You": r.genotype if isinstance(r.genotype, str) else "–",
               "Effect allele": r.effect_allele + ("" if pd.isna(r.copies) else f" ({int(r.copies)}×)"),
               "Best-known effect": r.effect}
        if gw is not None:
            g = gw[gw["rsid"] == r.rsid]
            sig = g[g["neg_log10_p"] >= GENOME_WIDE]
            row["Gene"] = main_gene(g["mapped_genes"]) or "–"
            row["GWAS results"] = f"{len(g)}" if len(g) else "–"
            row["Genome-wide traits"] = f"{sig['trait_label'].nunique()}" if len(g) else "–"
        rows.append(row)
    df = pd.DataFrame(rows)
    if gw is not None:
        df = df[["SNP", "Gene", "Trait", "You", "Effect allele", "Best-known effect", "GWAS results",
                 "Genome-wide traits"]]
    note = ("Effect alleles on the GRCh37 forward strand, with the number of copies you carry. The best-known effect "
            "comes from the config; " + ("“GWAS results” counts every catalog association for the SNP, “Genome-wide traits” the distinct "
               "traits with p < 5×10⁻⁸." if gw is not None else "GWAS Catalog results appear here once "
               "traits/gwas_catalog.py and traits/gwas_trait_map.py have been run."))
    return section("snps", "Your trait SNPs", "Well-known single-SNP traits and what you carry.",
                   card(f"{len(df)} SNPs", table(df, "dense sortable"), note))


def carried_table(c: pd.DataFrame) -> str:
    """Carried genome-wide associations with filters (category, effect size, SNP) and study links."""
    c = c.assign(size=pd.Categorical(c["effect_size_tier"].fillna("unknown"), SIZE_ORDER, ordered=True))
    c = c.sort_values(["size", "neg_log10_p"], ascending=[True, False])

    def opts(name, key, values):
        o = "".join(f'<option value="{esc(v)}">{esc(v)}</option>' for v in values)
        return (f'<label>{name} <select data-filter="{key}" data-table="#gwas-table" data-count="gwas-count" '
                f'data-noun="associations"><option value="">all</option>{o}</select></label>')

    cats = c["category"].value_counts().index
    sizes = [s for s in SIZE_ORDER if s in set(c["effect_size_tier"].fillna("unknown"))]
    snps = sorted(c["rsid"].unique())
    controls = (f'<div class="filters">{opts("Category", "cat", cats)}{opts("Effect size", "size", sizes)}'
                f'{opts("SNP", "snp", snps)}<span class="muted" id="gwas-count"></span></div>')
    head = "".join(f"<th>{h}</th>" for h in ["Trait", "Category", "SNP", "Gene", "You", "Direction for you",
                                             "Effect size", "p", "Study"])
    body = []
    for r in c.itertuples():
        size = r.effect_size_tier if isinstance(r.effect_size_tier, str) else "unknown"
        direction = r.for_you if isinstance(r.for_you, str) else "unclear"
        p = f"{r.pvalue_mantissa:g}e{int(r.pvalue_exponent)}" if pd.notna(r.pvalue_mantissa) and pd.notna(
            r.pvalue_exponent) else f"1e-{r.neg_log10_p:.0f}"
        pmid = str(int(r.pubmed_id)) if pd.notna(r.pubmed_id) else ""
        author = esc(r.first_author) if isinstance(r.first_author, str) else "study"
        study = (f'<a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}/" target="_blank" rel="noopener">{author}</a>'
                 if pmid else author)
        gene = str(r.mapped_genes).split(",")[0].split(";")[0].strip() if isinstance(r.mapped_genes, str) else ""
        tip = esc(r.explanation) if isinstance(r.explanation, str) else ""
        body.append(
            f'<tr data-cat="{esc(r.category)}" data-size="{esc(size)}" data-snp="{esc(r.rsid)}">'
            f'<td tabindex="0" data-tip="{esc(r.trait_label)}&#10;{tip}">{esc(r.trait_label)}</td>'
            f"<td>{esc(r.category)}</td><td>{esc(r.rsid)}</td><td>{esc(gene)}</td>"
            f'<td class="nw">{esc(r.genotype)} ({int(r.effect_allele_copies)}×{esc(r.effect_allele)})</td>'
            f'<td>{esc(direction)}</td><td>{esc(size)}</td><td class="nw">{esc(p)}</td><td>{study}</td></tr>')
    return (controls + f'<div class="table-wrap"><table id="gwas-table" class="dense sortable"><thead><tr>{head}'
            f'</tr></thead><tbody>{"".join(body)}</tbody></table></div>')


def sec_gwas(out: Path, rep: Path, gw: pd.DataFrame | None, carried: pd.DataFrame | None) -> str:
    if gw is None:
        return ""
    n_snp, n_assoc = gw["rsid"].nunique(), len(gw)
    n_study = gw["accession_id"].nunique() if "accession_id" in gw else gw["pubmed_id"].nunique()
    sig = gw[gw["neg_log10_p"] >= GENOME_WIDE]
    c = carried if carried is not None else gw.iloc[0:0]
    sized = c["effect_size_tier"].fillna("unknown")
    big = int(sized.isin(["moderate", "large"]).sum())
    tiles = [tile("SNPs looked up", f"{n_snp}", f"{n_assoc:,} associations from {n_study:,} studies"),
             tile("Genome-wide associations", f"{len(sig):,}", "p < 5×10⁻⁸, the usual GWAS threshold"),
             tile("Traits linked to alleles you carry", f"{c['trait_label'].nunique() if len(c) else 0}",
                  f"in {c['category'].nunique() if len(c) else 0} categories, genome-wide only"),
             tile("Of these, moderate or large effects", f"{big}",
                  f"of {len(c)}; the rest shift a trait very little" if len(c) else "")]
    body = f'<div class="tiles">{"".join(tiles)}</div>'
    if len(c):
        by_cat = c.groupby("category")["trait_label"].agg(lambda s: list(dict.fromkeys(s)))
        by_cat = by_cat.loc[by_cat.map(len).sort_values(ascending=False).index]
        rows = [(cat, len(t), f"{len(t)}", f"{cat}\n" + "\n".join(t[:6]) + (f"\n… and {len(t) - 6} more" if len(t) > 6
                                                                            else "")) for cat, t in by_cat.items()]
        size_rows = [(f"{s}", int((sized == s).sum()), f"{int((sized == s).sum())}", f"{s}: {SIZE_NOTE[s]}")
                     for s in SIZE_ORDER if (sized == s).any()]
        body += ('<div class="grid2">'
                 + card("Traits linked to your alleles, by category", hbars(rows),
                        "Distinct traits with a genome-wide association to an allele you carry. Hover a bar for the "
                        "traits.")
                 + card("How big are those effects?", hbars(size_rows,
                        note="Per copy of the allele. Most GWAS effects are tiny: each one shifts a trait or risk by "
                             "a few percent, and only many together (a polygenic score) say something about a person."),
                        "Associations you carry, by effect size")
                 + "</div>")
        body += card("Associations with alleles you carry", carried_table(c),
                     "Genome-wide significant results whose effect allele you carry (strand-ambiguous SNPs, "
                     "haplotype and interaction results left out). “Direction for you” is relative to people "
                     "without the allele; hover a trait for the plain-language reading. Study links open PubMed.")
    explorer = out / "gwas_explorer.html"
    link = ""
    if explorer.exists():
        shutil.copy2(explorer, rep / "gwas_explorer.html")
        link = ' The <a href="gwas_explorer.html">GWAS explorer</a> has every association, with paper summaries.'
    body += card("How to read this", (
        "<p>The GWAS Catalog lists associations found in large population studies: people carrying an allele had, "
        "on average, slightly different values of a trait. That is a statement about groups, not a prediction for "
        "you. Effects are mostly very small, depend on the population studied, and a single SNP rarely matters on "
        "its own. Colour and pigmentation studies use their own numeric scales, so their direction is marked "
        "unclear." + link + "</p>"))
    lede = (f"What large studies have linked to your {n_snp} trait SNPs, and which of those results involve alleles "
            "you carry. Only rsIDs were sent to the GWAS Catalog; your genotypes stayed local.")
    return section("gwas", "GWAS Catalog associations", esc(lede), body)
