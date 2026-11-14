"""Genome module: sequencing depth and data quality."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from report.core import GRCH37_LEN, INK, MUTED, SLOTS, SURFACE, card, columns, pct, section, style_ax, tile

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


def sec_qc(qc, cov) -> str:
    """Headline tiles for the raw SNP file and the CRAM."""
    tiles = []
    if qc:
        tiles += [tile("SNPs in the raw file", f"{qc['n_variants']:,}", qc["header"].get("method", "")),
                  tile("Call rate", pct(qc["call_rate"], 2), "SNPs with a genotype"),
                  tile("Inferred sex", qc["inferred_sex"], f"X heterozygosity {qc['het_x_nonpar']:.3f}"),
                  tile("Autosomal heterozygosity", f"{qc['het_autosomal']:.3f}", "share of SNPs that are heterozygous")]
    if cov and "per_chrom" in cov:
        pc = cov["per_chrom"]
        pc = pc.loc[[c for c in pc.index if c not in ("X", "Y")]]
        depth = (pc["meandepth"] * pc["endpos"]).sum() / pc["endpos"].sum()
        tiles.append(tile("Sequencing depth", f"{depth:.2f}×", "autosomes, from the CRAM"))
    if not tiles:
        return ""
    return section("qc", "At a glance", "", f'<div class="tiles">{"".join(tiles)}</div>')
